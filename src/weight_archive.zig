//! Weight Archive — On-disk magic + mmap archive of Records for Own Weights.
//!
//! Phase W1 implementation of docs/BLUEPRINT-20260912-OWN-WEIGHTS-Z3.md:
//!   - Format: Records (20,480 B each: 3,072 B Prefetch Label Area + 17,408 B Cell).
//!   - Alignment: Sector-aligned 4,096 B header + 20,480 B record stride (5 x 4,096 B NVMe sectors).
//!   - Zero sector bleed, zero page misalignment: every Record starts on a 4,096-byte page boundary.
//!   - Paging API: Stolen Cactus `MappedFileRegistry` + `madvise` paging pattern
//!     (`MADV_WILLNEED`, `MADV_DONTNEED`, `MADV_SEQUENTIAL`, `MADV_RANDOM`).
//!   - Explicit rejection: No GGUF, no HuggingFace safetensors bloat, no libz3 linkage,
//!     no hipHostRegister of the cell bank.
//!
//! Toolchain: Zig 0.17 / Zig 0.16.

const std = @import("std");
const geometry = @import("geometry.zig");

// ── On-Disk Format Constants ──────────────────────────────────────────────────

/// FourCC magic: "WARC" (Weight ARChive) = 0x57415243 (little-endian: 'W', 'A', 'R', 'C').
pub const ARCHIVE_MAGIC: u32 = 0x57415243;
/// FourCC magic: "CHPE" (Christopher Hamil Packed Engine) = 0x45504843 (little-endian: 'C', 'H', 'P', 'E').
pub const CHPE_MAGIC: u32 = 0x45504843;

/// Archive format specification version.
pub const ARCHIVE_VERSION: u32 = 1;

/// Sector-aligned header block size (4,096 bytes = 1 NVMe sector = 1 system page).
/// Sector 0 holds the 64-byte `WeightArchiveHeader` followed by 4,032 zero-padded bytes.
/// Placing records at offset 4,096 guarantees that Record 0 and all subsequent records
/// (stride 20,480 B = 5 x 4,096 B) are strictly page-aligned for zero-copy mmap & madvise.
pub const HEADER_BYTES: usize = geometry.SECTOR_BYTES; // 4096

/// Coded tile block size within Cell.fingerprints: 32 vectors x 512 bytes = 16,384 bytes.
pub const TILE_CODE_BYTES: usize = geometry.FINGERPRINT_VECTORS * 512; // 16,384

comptime {
    // Comptime invariants: ensure geometry matches Blueprint & Invariant A-1 / Sector Law
    std.debug.assert(geometry.RECORD_BYTES == 20480);
    std.debug.assert(geometry.CELL_BYTES == 17408);
    std.debug.assert(geometry.PREFETCH_LABEL_BYTES == 3072);
    std.debug.assert(geometry.BYTECODE_HEADER_BYTES == 64);
    std.debug.assert(geometry.ZECKENDORF_SEAL_BYTES == 16);
    std.debug.assert(HEADER_BYTES == 4096);
    std.debug.assert(HEADER_BYTES % geometry.SECTOR_BYTES == 0);
    std.debug.assert(geometry.RECORD_BYTES % geometry.SECTOR_BYTES == 0);
    std.debug.assert(TILE_CODE_BYTES == 16384);
}

// ── Header Definition ─────────────────────────────────────────────────────────

/// 64-byte uniform archive header. Aligned to 1 CPU cache line (64 bytes).
/// Stored at byte offset 0 of the 4,096-byte sector header block.
pub const WeightArchiveHeader = extern struct {
    /// 0..3: Magic number (0x57415243 = "WARC")
    magic: u32 align(64),
    /// 4..7: Archive format version (1)
    version: u32,
    /// 8..15: Header size in bytes (4096 = 1 sector)
    header_bytes: u64,
    /// 16..23: Number of records in archive
    record_count: u64,
    /// 24..31: Number of weight tiles in archive (1 tile per record)
    tile_count: u64,
    /// 32..39: Record stride in bytes (20,480)
    record_bytes: u64,
    /// 40..47: Cell size in bytes (17,408)
    cell_bytes: u64,
    /// 48..55: Prefetch label area size in bytes (3,072)
    prefetch_bytes: u64,
    /// 56..63: Model / Archive flags & provenance bits
    flags: u64,

    comptime {
        std.debug.assert(@sizeOf(WeightArchiveHeader) == 64);
        std.debug.assert(@alignOf(WeightArchiveHeader) == 64);
        std.debug.assert(@offsetOf(WeightArchiveHeader, "magic") == 0);
        std.debug.assert(@offsetOf(WeightArchiveHeader, "version") == 4);
        std.debug.assert(@offsetOf(WeightArchiveHeader, "header_bytes") == 8);
        std.debug.assert(@offsetOf(WeightArchiveHeader, "record_count") == 16);
        std.debug.assert(@offsetOf(WeightArchiveHeader, "tile_count") == 24);
        std.debug.assert(@offsetOf(WeightArchiveHeader, "record_bytes") == 32);
        std.debug.assert(@offsetOf(WeightArchiveHeader, "cell_bytes") == 40);
        std.debug.assert(@offsetOf(WeightArchiveHeader, "prefetch_bytes") == 48);
        std.debug.assert(@offsetOf(WeightArchiveHeader, "flags") == 56);
    }
};

// ── Tile Metadata ─────────────────────────────────────────────────────────────

/// 32-byte tile metadata header embedded inside Cell.semantic_payload (960 B).
/// Tracks neural layer index, expert index, Zeckendorf integer codec index, and quantization budget.
pub const TileMetadata = extern struct {
    layer_index: u32,
    expert_index: u32,
    zeck_index: u64,
    bit_budget: u32,
    quant_bits: u8,
    reserved: [3]u8,
    custom_flags: u64,

    comptime {
        std.debug.assert(@sizeOf(TileMetadata) == 32);
        std.debug.assert(@sizeOf(TileMetadata) <= geometry.SEMANTIC_PAYLOAD_BYTES);
    }
};

/// Custom flag for Group-128 quantization with FP16 outlier channel protection
pub const FLAG_GROUP128_OUTLIERS: u64 = 0x01;
/// Archive-level flag indicating hybrid precision (BF16 Attention + 4-bit MLP)
pub const FLAG_HYBRID_ARCHIVE: u64 = 0x02;
/// Archive-level flag indicating raw contiguous FP16 weights (native ARMv8.2-A fmla.8h line rate)
pub const FLAG_RAW_FP16: u64 = 0x04;

/// Unpacks an FP16 byte slice to FP32, converting IEEE 754 half-precision to 32-bit floats.
pub fn unpackF16SliceToF32(raw_f16_bytes: []const u8, out_f32: []f32) void {
    std.debug.assert(raw_f16_bytes.len >= out_f32.len * 2);
    var i: usize = 0;
    while (i < out_f32.len) : (i += 1) {
        const raw_u16 = std.mem.readInt(u16, raw_f16_bytes[i * 2 ..][0..2], .little);
        const f16_val: f16 = @bitCast(raw_u16);
        out_f32[i] = @floatCast(f16_val);
    }
}


// ── Errors ────────────────────────────────────────────────────────────────────

pub const WeightArchiveError = error{
    InvalidMagic,
    UnsupportedVersion,
    BadGeometry,
    Truncated,
    RecordIndexOutOfRange,
    FileTooSmall,
    AlreadyOpen,
    PathNotFound,
    MmapFailed,
    MisalignedHeader,
};

// ── WeightArchive: Mmap Archive View ──────────────────────────────────────────

/// A read-only zero-copy view over an mmap'd Weight Archive file.
/// All records and tiles are accessed directly from page cache via memory pointer arithmetic.
pub const WeightArchive = struct {
    bytes: []align(std.heap.page_size_min) const u8,
    file_size: usize,
    owns_mapping: bool,
    header_data: WeightArchiveHeader,

    /// Validates raw mapped bytes against WeightArchive geometry and magic.
    pub fn validate(bytes: []const u8) WeightArchiveError!WeightArchiveHeader {
        if (bytes.len < HEADER_BYTES) return WeightArchiveError.FileTooSmall;

        const hdr: *const WeightArchiveHeader = @ptrCast(@alignCast(bytes.ptr));
        if (hdr.magic != ARCHIVE_MAGIC and hdr.magic != CHPE_MAGIC) return WeightArchiveError.InvalidMagic;
        if (hdr.version != ARCHIVE_VERSION) return WeightArchiveError.UnsupportedVersion;
        if (hdr.header_bytes != HEADER_BYTES) return WeightArchiveError.MisalignedHeader;

        const is_standard = (hdr.record_bytes == geometry.RECORD_BYTES and hdr.prefetch_bytes == geometry.PREFETCH_LABEL_BYTES);
        const is_dense = (hdr.record_bytes == geometry.CELL_BYTES and hdr.prefetch_bytes == 0);
        const is_raw = (hdr.record_bytes == TILE_CODE_BYTES and hdr.prefetch_bytes == 0);
        if (!is_standard and !is_dense and !is_raw) return WeightArchiveError.BadGeometry;
        if (!is_raw and hdr.cell_bytes != geometry.CELL_BYTES) return WeightArchiveError.BadGeometry;
        if (is_raw and hdr.cell_bytes != TILE_CODE_BYTES) return WeightArchiveError.BadGeometry;

        const stride: usize = @intCast(hdr.record_bytes);
        const required_len = hdr.header_bytes + hdr.record_count * stride;
        if (required_len > bytes.len) return WeightArchiveError.Truncated;

        return hdr.*;
    }

    /// Open an archive using a standard `std.Io` handle.
    pub fn open(io: std.Io, path: []const u8) !WeightArchive {
        const file = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer file.close(io);

        const stat = try file.stat(io);
        return try openWithFd(file.handle, @intCast(stat.size));
    }

    /// Open an archive using POSIX direct filesystem open (no std.Io required).
    pub fn openPosix(path: [*:0]const u8) !WeightArchive {
        const fd_val = std.os.linux.open(path, .{ .ACCMODE = .RDONLY }, 0);
        if (std.os.linux.errno(fd_val) != .SUCCESS) return WeightArchiveError.FileTooSmall;
        const fd: std.posix.fd_t = @intCast(fd_val);
        defer _ = std.os.linux.close(fd);

        var st: std.os.linux.Statx = undefined;
        if (std.os.linux.statx(fd, "", std.os.linux.AT.EMPTY_PATH, std.os.linux.STATX.BASIC_STATS, &st) != 0) return WeightArchiveError.FileTooSmall;
        return try openWithFd(fd, @intCast(st.size));
    }

    /// Internal helper: maps file descriptor and validates header.
    fn openWithFd(fd: std.posix.fd_t, file_size: usize) !WeightArchive {
        if (file_size < HEADER_BYTES) return WeightArchiveError.FileTooSmall;

        const mapped = try std.posix.mmap(
            null,
            file_size,
            std.posix.PROT{ .READ = true },
            .{ .TYPE = .PRIVATE },
            fd,
            0,
        );
        errdefer std.posix.munmap(mapped);

        if (comptime @import("builtin").os.tag == .linux) {
            _ = std.os.linux.madvise(@alignCast(@constCast(mapped.ptr)), file_size, std.os.linux.MADV.SEQUENTIAL);
            _ = std.os.linux.madvise(@alignCast(@constCast(mapped.ptr)), file_size, std.os.linux.MADV.WILLNEED);
            _ = std.os.linux.madvise(@alignCast(@constCast(mapped.ptr)), file_size, std.os.linux.MADV.HUGEPAGE);
        }

        const hdr = try validate(mapped);
        return .{
            .bytes = mapped,
            .file_size = file_size,
            .owns_mapping = true,
            .header_data = hdr,
        };
    }

    /// Construct a view over existing mapped memory (e.g. for testing or external buffers).
    pub fn openFromBytes(bytes: []align(std.heap.page_size_min) const u8) WeightArchiveError!WeightArchive {
        const hdr = try validate(bytes);
        return .{
            .bytes = bytes,
            .file_size = bytes.len,
            .owns_mapping = false,
            .header_data = hdr,
        };
    }

    /// Releases mapped pages and unmaps file.
    pub fn close(self: *WeightArchive) void {
        if (self.owns_mapping and self.bytes.len > 0) {
            std.posix.munmap(self.bytes);
        }
        self.bytes = &.{};
        self.file_size = 0;
        self.owns_mapping = false;
    }

    /// Archive header view.
    pub inline fn header(self: *const WeightArchive) *const WeightArchiveHeader {
        return &self.header_data;
    }

    /// Total records in archive.
    pub inline fn recordCount(self: *const WeightArchive) usize {
        return @intCast(self.header_data.record_count);
    }

    /// Total tiles in archive.
    pub inline fn tileCount(self: *const WeightArchive) usize {
        return @intCast(self.header_data.tile_count);
    }

    /// True if archive is densely packed without PreFetchLabelArea.
    pub inline fn isDense(self: *const WeightArchive) bool {
        return self.header_data.prefetch_bytes == 0;
    }

    /// Record stride in bytes (20,480 for standard, 17,408 for dense).
    pub inline fn recordBytes(self: *const WeightArchive) usize {
        return @intCast(self.header_data.record_bytes);
    }

    /// Prefetch label size in bytes (3,072 for standard, 0 for dense).
    pub inline fn prefetchBytes(self: *const WeightArchive) usize {
        return @intCast(self.header_data.prefetch_bytes);
    }

    /// Computes the exact byte offset for a record index.
    pub inline fn recordOffset(self: *const WeightArchive, index: usize) usize {
        return HEADER_BYTES + index * self.recordBytes();
    }

    /// Direct reference to a 20,480-byte Record in mapped memory.
    pub fn getRecord(self: *const WeightArchive, index: usize) WeightArchiveError!*const geometry.Record {
        if (self.isDense()) return WeightArchiveError.BadGeometry;
        if (index >= self.recordCount()) return WeightArchiveError.RecordIndexOutOfRange;
        const off = self.recordOffset(index);
        if (off + geometry.RECORD_BYTES > self.bytes.len) return WeightArchiveError.Truncated;
        return @ptrCast(@alignCast(self.bytes.ptr + off));
    }

    /// Direct reference to a 20,480-byte Record in mapped memory (zero-overhead hot path).
    pub inline fn getRecordDirect(self: *const WeightArchive, index: usize) *const geometry.Record {
        const off = HEADER_BYTES + index * self.recordBytes();
        return @ptrCast(@alignCast(self.bytes.ptr + off));
    }

    /// Base pointer to the contiguous array of 20,480-byte Records.
    pub inline fn getRecordsPtr(self: *const WeightArchive) [*]const geometry.Record {
        return @ptrCast(@alignCast(self.bytes.ptr + HEADER_BYTES));
    }

    /// Direct reference to the 17,408-byte Cell (works seamlessly for both standard and dense archives).
    pub inline fn getCellDirect(self: *const WeightArchive, index: usize) *const geometry.Cell {
        const off = HEADER_BYTES + index * self.recordBytes() + self.prefetchBytes();
        return @ptrCast(@alignCast(self.bytes.ptr + off));
    }

    /// True if archive is raw contiguous 16,384-byte blocks (zero padding).
    pub inline fn isRawContiguous(self: *const WeightArchive) bool {
        return self.header_data.record_bytes == TILE_CODE_BYTES and self.header_data.prefetch_bytes == 0;
    }

    /// True if archive is raw contiguous FP16 blocks (native ARMv8.2-A fmla.8h line rate).
    pub inline fn isRawFp16(self: *const WeightArchive) bool {
        return self.isRawContiguous() and (self.header_data.flags & FLAG_RAW_FP16 != 0);
    }

    /// Direct pointer to the 8,192 f16 weights of tile `index` across all archive formats.
    pub inline fn getTileF16Direct(self: *const WeightArchive, index: usize) [*]const f16 {
        return @ptrCast(@alignCast(self.getTileU16Direct(index)));
    }


    /// Direct pointer to the 8,192 u16 weights of tile `index` across all archive formats.
    pub inline fn getTileU16Direct(self: *const WeightArchive, index: usize) [*]const u16 {
        if (self.isRawContiguous()) {
            const off = HEADER_BYTES + index * TILE_CODE_BYTES;
            return @ptrCast(@alignCast(self.bytes.ptr + off));
        } else if (self.isDense()) {
            const off = HEADER_BYTES + index * geometry.CELL_BYTES + geometry.BYTECODE_HEADER_BYTES;
            return @ptrCast(@alignCast(self.bytes.ptr + off));
        } else {
            const off = HEADER_BYTES + index * geometry.RECORD_BYTES + geometry.PREFETCH_LABEL_BYTES + geometry.BYTECODE_HEADER_BYTES;
            return @ptrCast(@alignCast(self.bytes.ptr + off));
        }
    }

    /// Direct pointer to raw f32 weights of tile `index` (e.g. norms, biases) across all archive formats.
    pub inline fn getTileF32Direct(self: *const WeightArchive, index: usize) [*]const f32 {
        return @ptrCast(@alignCast(self.getTileU16Direct(index)));
    }


    /// Unpacks 1D norm or bias weights for tile `index` into an `f32` destination slice across all archive formats.
    /// Safely handles IEEE 754 half-precision conversion when `isRawFp16()` is true.
    pub fn unpackNormTileF32(self: *const WeightArchive, index: usize, out: []f32) void {
        if (self.isRawFp16()) {
            const tile_u16 = self.getTileU16Direct(index);
            const tile_u8: [*]const u8 = @ptrCast(tile_u16);
            unpackF16SliceToF32(tile_u8[0 .. out.len * 2], out);
        } else if (self.isRawContiguous()) {
            // BF16 archives have 1D norm tiles packed as direct FP32
            const tile_f32: [*]const f32 = @ptrCast(@alignCast(self.getTileU16Direct(index)));
            @memcpy(out, tile_f32[0..out.len]);
        } else if (self.isDense()) {
            const cell = self.getCellDirect(index);
            const raw_f32: [*]const f32 = @ptrCast(@alignCast(&cell.fingerprints));
            @memcpy(out, raw_f32[0..out.len]);
        } else {
            const rec = self.getRecord(index) catch return;
            const raw_f32: [*]const f32 = @ptrCast(@alignCast(&rec.cell.fingerprints));
            @memcpy(out, raw_f32[0..out.len]);
        }
    }

    /// Base pointer to contiguous 17,408-byte Cells (valid when isDense() is true).
    pub inline fn getCellsPtr(self: *const WeightArchive) [*]const geometry.Cell {
        return @ptrCast(@alignCast(self.bytes.ptr + HEADER_BYTES));
    }

    /// Direct reference to the 17,408-byte Cell within a Record or Dense Archive.
    pub fn getCell(self: *const WeightArchive, index: usize) WeightArchiveError!*const geometry.Cell {
        if (index >= self.recordCount()) return WeightArchiveError.RecordIndexOutOfRange;
        return self.getCellDirect(index);
    }

    /// Direct reference to the 3,072-byte PreFetchLabelArea within a Record.
    pub fn getPrefetchLabel(self: *const WeightArchive, index: usize) WeightArchiveError!*const geometry.PreFetchLabelArea {
        const rec = try self.getRecord(index);
        return &rec.prefetch_label;
    }

    /// Direct reference to the 16-byte Zeckendorf sequence seal.
    pub fn getZeckendorfSeal(self: *const WeightArchive, index: usize) WeightArchiveError!*const [geometry.ZECKENDORF_SEAL_BYTES]u8 {
        const label = try self.getPrefetchLabel(index);
        return &label.zeckendorf_seal;
    }

    /// Direct reference to the 16,384-byte coded tile block (32 x 512B FingerprintVectors).
    pub fn getTile(self: *const WeightArchive, index: usize) WeightArchiveError!*const [geometry.FINGERPRINT_VECTORS]geometry.FingerprintVector {
        const cell = try self.getCell(index);
        return &cell.fingerprints;
    }

    /// Direct byte-slice view of the 16,384-byte coded tile block.
    pub fn getTileBytes(self: *const WeightArchive, index: usize) WeightArchiveError!*const [TILE_CODE_BYTES]u8 {
        const cell = try self.getCell(index);
        return @ptrCast(&cell.fingerprints);
    }

    /// Direct reference to the 960-byte semantic payload of a Cell.
    pub fn getSemanticPayload(self: *const WeightArchive, index: usize) WeightArchiveError!*const [geometry.SEMANTIC_PAYLOAD_BYTES]u8 {
        const cell = try self.getCell(index);
        return &cell.semantic_payload;
    }

    /// Decodes TileMetadata from the start of the cell's semantic payload.
    pub fn getTileMetadata(self: *const WeightArchive, index: usize) WeightArchiveError!TileMetadata {
        const cell = try self.getCell(index);
        const meta_ptr: *const TileMetadata = @ptrCast(@alignCast(&cell.semantic_payload[0]));
        return meta_ptr.*;
    }

    // ── Cactus-Style Paging & madvise API ──────────────────────────────────────

    /// Advises the kernel to prefetch a specific 20,480-byte Record (`MADV_WILLNEED`).
    /// Because every Record begins on a 4,096-byte page boundary, this call is strictly page-aligned.
    pub fn prefetchRecord(self: *const WeightArchive, index: usize) WeightArchiveError!void {
        if (index >= self.recordCount()) return WeightArchiveError.RecordIndexOutOfRange;
        const off = self.recordOffset(index);
        const ptr: [*]align(std.heap.page_size_min) u8 = @alignCast(@constCast(self.bytes.ptr + off));
        std.posix.madvise(
            ptr,
            geometry.RECORD_BYTES,
            std.posix.MADV.WILLNEED,
        ) catch {};
    }

    /// Releases a specific 20,480-byte Record from physical RAM page cache (`MADV_DONTNEED`).
    /// The page remains mapped and addressable; subsequent reads will fault it back in on demand.
    pub fn releaseRecord(self: *const WeightArchive, index: usize) WeightArchiveError!void {
        if (index >= self.recordCount()) return WeightArchiveError.RecordIndexOutOfRange;
        const off = self.recordOffset(index);
        const ptr: [*]align(std.heap.page_size_min) u8 = @alignCast(@constCast(self.bytes.ptr + off));
        std.posix.madvise(
            ptr,
            geometry.RECORD_BYTES,
            std.posix.MADV.DONTNEED,
        ) catch {};
    }

    /// Prefetches a contiguous range of records into memory.
    pub fn prefetchRange(self: *const WeightArchive, start_index: usize, count: usize) WeightArchiveError!void {
        if (count == 0) return;
        if (start_index + count > self.recordCount()) return WeightArchiveError.RecordIndexOutOfRange;
        const off = self.recordOffset(start_index);
        const len = count * geometry.RECORD_BYTES;
        const ptr: [*]align(std.heap.page_size_min) u8 = @alignCast(@constCast(self.bytes.ptr + off));
        std.posix.madvise(
            ptr,
            len,
            std.posix.MADV.WILLNEED,
        ) catch {};
    }

    /// Evicts a contiguous range of records from memory page cache.
    pub fn releaseRange(self: *const WeightArchive, start_index: usize, count: usize) WeightArchiveError!void {
        if (count == 0) return;
        if (start_index + count > self.recordCount()) return WeightArchiveError.RecordIndexOutOfRange;
        const off = self.recordOffset(start_index);
        const len = count * geometry.RECORD_BYTES;
        const ptr: [*]align(std.heap.page_size_min) u8 = @alignCast(@constCast(self.bytes.ptr + off));
        std.posix.madvise(
            ptr,
            len,
            std.posix.MADV.DONTNEED,
        ) catch {};
    }

    /// Prefetches all records in the archive (`MADV_WILLNEED`).
    pub fn prefetchAll(self: *const WeightArchive) void {
        if (self.bytes.len == 0) return;
        const ptr: [*]align(std.heap.page_size_min) u8 = @alignCast(@constCast(self.bytes.ptr));
        std.posix.madvise(
            ptr,
            self.bytes.len,
            std.posix.MADV.WILLNEED,
        ) catch {};
    }

    /// Evicts all mapped pages in the archive from physical RAM (`MADV_DONTNEED`).
    pub fn releaseAll(self: *const WeightArchive) void {
        if (self.bytes.len == 0) return;
        const ptr: [*]align(std.heap.page_size_min) u8 = @alignCast(@constCast(self.bytes.ptr));
        std.posix.madvise(
            ptr,
            self.bytes.len,
            std.posix.MADV.DONTNEED,
        ) catch {};
    }

    /// Informs the kernel of expected sequential forward access across all records.
    pub fn adviseSequential(self: *const WeightArchive) void {
        if (self.bytes.len == 0) return;
        const ptr: [*]align(std.heap.page_size_min) u8 = @alignCast(@constCast(self.bytes.ptr));
        std.posix.madvise(
            ptr,
            self.bytes.len,
            std.posix.MADV.SEQUENTIAL,
        ) catch {};
    }

    /// Informs the kernel of expected random sparse access across tiles.
    pub fn adviseRandom(self: *const WeightArchive) void {
        if (self.bytes.len == 0) return;
        const ptr: [*]align(std.heap.page_size_min) u8 = @alignCast(@constCast(self.bytes.ptr));
        std.posix.madvise(
            ptr,
            self.bytes.len,
            std.posix.MADV.RANDOM,
        ) catch {};
    }
};

// ── Archive Creation API ──────────────────────────────────────────────────────

/// Creates a new Weight Archive on disk with the specified Records and writes the sector header.
pub fn createArchive(
    io: std.Io,
    path: []const u8,
    records: []const geometry.Record,
    flags: u64,
) !void {
    const dir = std.Io.Dir.cwd();
    var file = try dir.createFile(io, path, .{ .read = true, .truncate = true });
    defer file.close(io);

    const total_bytes = HEADER_BYTES + records.len * geometry.RECORD_BYTES;
    try file.setLength(io, total_bytes);

    const mapped = try std.posix.mmap(
        null,
        total_bytes,
        std.posix.PROT{ .READ = true, .WRITE = true },
        .{ .TYPE = .SHARED },
        file.handle,
        0,
    );
    defer std.posix.munmap(mapped);

    // Initialize 4,096 B sector 0 to zero
    @memset(mapped[0..HEADER_BYTES], 0);

    // Write 64 B header
    const hdr: *WeightArchiveHeader = @ptrCast(@alignCast(mapped.ptr));
    hdr.* = .{
        .magic = ARCHIVE_MAGIC,
        .version = ARCHIVE_VERSION,
        .header_bytes = HEADER_BYTES,
        .record_count = @intCast(records.len),
        .tile_count = @intCast(records.len),
        .record_bytes = geometry.RECORD_BYTES,
        .cell_bytes = geometry.CELL_BYTES,
        .prefetch_bytes = geometry.PREFETCH_LABEL_BYTES,
        .flags = flags,
    };

    // Copy records if any
    if (records.len > 0) {
        const dest = mapped[HEADER_BYTES..total_bytes];
        const src: []const u8 = std.mem.sliceAsBytes(records);
        @memcpy(dest, src);
    }
}

/// Creates a new Weight Archive using POSIX file descriptor primitives.
pub fn createArchivePosix(
    path: []const u8,
    records: []const geometry.Record,
    flags: u64,
) !void {
    const fd = try std.posix.openat(
        std.posix.AT.FDCWD,
        path,
        .{ .ACCMODE = .RDWR, .CREAT = true, .TRUNC = true },
        0o644,
    );
    defer _ = std.os.linux.close(fd);

    const total_bytes = HEADER_BYTES + records.len * geometry.RECORD_BYTES;
    _ = std.os.linux.ftruncate(fd, @intCast(total_bytes));

    const mapped = try std.posix.mmap(
        null,
        total_bytes,
        std.posix.PROT{ .READ = true, .WRITE = true },
        .{ .TYPE = .SHARED },
        fd,
        0,
    );
    defer std.posix.munmap(mapped);

    // Zero entire sector 0
    @memset(mapped[0..HEADER_BYTES], 0);

    // Write 64 B header
    const hdr: *WeightArchiveHeader = @ptrCast(@alignCast(mapped.ptr));
    hdr.* = .{
        .magic = ARCHIVE_MAGIC,
        .version = ARCHIVE_VERSION,
        .header_bytes = HEADER_BYTES,
        .record_count = @intCast(records.len),
        .tile_count = @intCast(records.len),
        .record_bytes = geometry.RECORD_BYTES,
        .cell_bytes = geometry.CELL_BYTES,
        .prefetch_bytes = geometry.PREFETCH_LABEL_BYTES,
        .flags = flags,
    };

    // Copy records
    if (records.len > 0) {
        const dest = mapped[HEADER_BYTES..total_bytes];
        const src: []const u8 = std.mem.sliceAsBytes(records);
        @memcpy(dest, src);
    }
}

/// Converts a standard 20,480-byte record archive to a dense 17,408-byte cell archive by stripping PreFetchLabelArea.
pub fn convertToDensePosix(
    src_path: [*:0]const u8,
    dst_path: [*:0]const u8,
) !void {
    var src_archive = try WeightArchive.openPosix(src_path);
    defer src_archive.close();

    const count = src_archive.recordCount();
    const dst_total_bytes = HEADER_BYTES + count * geometry.CELL_BYTES;

    const fd_val = std.os.linux.open(dst_path, .{ .ACCMODE = .RDWR, .CREAT = true, .TRUNC = true }, 0o644);
    if (std.os.linux.errno(fd_val) != .SUCCESS) return WeightArchiveError.FileTooSmall;
    const fd: std.posix.fd_t = @intCast(fd_val);
    defer _ = std.os.linux.close(fd);

    _ = std.os.linux.ftruncate(fd, @intCast(dst_total_bytes));

    const mapped = try std.posix.mmap(
        null,
        dst_total_bytes,
        std.posix.PROT{ .READ = true, .WRITE = true },
        .{ .TYPE = .SHARED },
        fd,
        0,
    );
    defer std.posix.munmap(mapped);

    // Zero header
    @memset(mapped[0..HEADER_BYTES], 0);

    const hdr: *WeightArchiveHeader = @ptrCast(@alignCast(mapped.ptr));
    hdr.* = src_archive.header_data;
    hdr.record_bytes = geometry.CELL_BYTES;
    hdr.prefetch_bytes = 0;

    const dst_cells: [*]geometry.Cell = @ptrCast(@alignCast(mapped.ptr + HEADER_BYTES));
    for (0..count) |i| {
        const cell = src_archive.getCellDirect(i);
        dst_cells[i] = cell.*;
    }
}

/// Converts an archive to a raw contiguous 16,384-byte weight block archive,
/// stripping ALL PreFetchLabelArea (3,072B) AND Cell bytecode/semantic padding (1,024B).
/// Eliminates 1.54 GB of dead sector padding, leaving only contiguous 16,384B weight tensors.
pub fn convertToRawContiguousPosix(
    src_path: [*:0]const u8,
    dst_path: [*:0]const u8,
) !void {
    var src_archive = try WeightArchive.openPosix(src_path);
    defer src_archive.close();

    const count = src_archive.recordCount();
    const dst_total_bytes = HEADER_BYTES + count * TILE_CODE_BYTES;

    const fd_val = std.os.linux.open(dst_path, .{ .ACCMODE = .RDWR, .CREAT = true, .TRUNC = true }, 0o644);
    if (std.os.linux.errno(fd_val) != .SUCCESS) return WeightArchiveError.FileTooSmall;
    const fd: std.posix.fd_t = @intCast(fd_val);
    defer _ = std.os.linux.close(fd);

    _ = std.os.linux.ftruncate(fd, @intCast(dst_total_bytes));

    const mapped = try std.posix.mmap(
        null,
        dst_total_bytes,
        std.posix.PROT{ .READ = true, .WRITE = true },
        .{ .TYPE = .SHARED },
        fd,
        0,
    );
    defer std.posix.munmap(mapped);

    // Zero header
    @memset(mapped[0..HEADER_BYTES], 0);

    const hdr: *WeightArchiveHeader = @ptrCast(@alignCast(mapped.ptr));
    hdr.* = src_archive.header_data;
    hdr.record_bytes = TILE_CODE_BYTES;
    hdr.cell_bytes = TILE_CODE_BYTES;
    hdr.prefetch_bytes = 0;

    const dst_raw: [*][TILE_CODE_BYTES]u8 = @ptrCast(@alignCast(mapped.ptr + HEADER_BYTES));
    for (0..count) |i| {
        const tile_u16 = src_archive.getTileU16Direct(i);
        const tile_u8: [*]const u8 = @ptrCast(tile_u16);
        @memcpy(&dst_raw[i], tile_u8[0..TILE_CODE_BYTES]);
    }
}

/// Converts an archive to a raw contiguous 16,384-byte FP16 weight block archive,
/// converting all BF16 weights to IEEE 754 half-precision (FP16) floats for ARMv8.2-A native fmla.8h line-rate vector execution.
pub fn convertToRawFp16ContiguousPosix(
    src_path: [*:0]const u8,
    dst_path: [*:0]const u8,
) !void {
    var src_archive = try WeightArchive.openPosix(src_path);
    defer src_archive.close();

    const count = src_archive.recordCount();
    const dst_total_bytes = HEADER_BYTES + count * TILE_CODE_BYTES;

    const fd_val = std.os.linux.open(dst_path, .{ .ACCMODE = .RDWR, .CREAT = true, .TRUNC = true }, 0o644);
    if (std.os.linux.errno(fd_val) != .SUCCESS) return WeightArchiveError.FileTooSmall;
    const fd: std.posix.fd_t = @intCast(fd_val);
    defer _ = std.os.linux.close(fd);

    _ = std.os.linux.ftruncate(fd, @intCast(dst_total_bytes));

    const mapped = try std.posix.mmap(
        null,
        dst_total_bytes,
        std.posix.PROT{ .READ = true, .WRITE = true },
        .{ .TYPE = .SHARED },
        fd,
        0,
    );
    defer std.posix.munmap(mapped);

    // Zero header
    @memset(mapped[0..HEADER_BYTES], 0);

    const hdr: *WeightArchiveHeader = @ptrCast(@alignCast(mapped.ptr));
    hdr.* = src_archive.header_data;
    hdr.record_bytes = TILE_CODE_BYTES;
    hdr.cell_bytes = TILE_CODE_BYTES;
    hdr.prefetch_bytes = 0;
    hdr.flags |= FLAG_RAW_FP16;

    const dst_raw: [*][8192]u16 = @ptrCast(@alignCast(mapped.ptr + HEADER_BYTES));
    for (0..count) |i| {
        const tile_u16 = src_archive.getTileU16Direct(i);
        const is_1d = blk: {
            if (count == 376853) {
                if (i == 376852) break :blk true;
                if (i < 37984) break :blk false;
                const rel = (i - 37984) % 9413;
                if (rel == 0 or rel == 513 or rel == 578 or rel == 643 or rel == 1156) break :blk true;
            }
            if (!src_archive.isRawContiguous()) {
                const cell = src_archive.getCellDirect(i);
                const meta: *const TileMetadata = @ptrCast(@alignCast(&cell.semantic_payload));
                if (meta.quant_bits == 32) break :blk true;
            }
            break :blk false;
        };

        if (is_1d) {
            const tile_u8: [*]const u8 = @ptrCast(tile_u16);
            const dst_u8: [*]u8 = @ptrCast(&dst_raw[i]);
            @memcpy(dst_u8[0..TILE_CODE_BYTES], tile_u8[0..TILE_CODE_BYTES]);
        } else {
            for (0..8192) |k| {
                const bf16_u = tile_u16[k];
                const f32_v: f32 = @bitCast(@as(u32, bf16_u) << 16);
                const f16_v: f16 = @floatCast(f32_v);
                dst_raw[i][k] = @bitCast(f16_v);
            }
        }
    }
}


// ── Cactus-Style MappedFileRegistry ───────────────────────────────────────────

/// Lightweight atomic spinlock for thread-safe registry operations.
pub const SpinLock = struct {
    state: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

    pub fn lock(self: *SpinLock) void {
        while (self.state.cmpxchgWeak(false, true, .acquire, .monotonic) != null) {
            std.atomic.spinLoopHint();
        }
    }

    pub fn unlock(self: *SpinLock) void {
        self.state.store(false, .release);
    }
};

/// Thread-safe registry caching open mmap archive files by canonical path.
/// Steals the Cactus `GraphFile::MappedFileRegistry` pattern:
/// multiple reader components share a single mapped file descriptor and address range,
/// avoiding redundant OS file mappings and memory footprint.
pub const MappedFileRegistry = struct {
    pub const Entry = struct {
        path: []u8,
        archive: WeightArchive,
        ref_count: usize,
    };

    allocator: std.mem.Allocator,
    lock: SpinLock = .{},
    entries: std.ArrayListUnmanaged(Entry) = .empty,

    pub fn init(allocator: std.mem.Allocator) MappedFileRegistry {
        return .{
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MappedFileRegistry) void {
        self.lock.lock();
        defer self.lock.unlock();
        for (self.entries.items) |*entry| {
            entry.archive.close();
            self.allocator.free(entry.path);
        }
        self.entries.deinit(self.allocator);
    }

    /// Retrieves an already-mapped archive or opens and maps it if not loaded.
    pub fn getOrLoad(self: *MappedFileRegistry, io: std.Io, path: []const u8) !*WeightArchive {
        self.lock.lock();
        defer self.lock.unlock();

        for (self.entries.items) |*entry| {
            if (std.mem.eql(u8, entry.path, path)) {
                entry.ref_count += 1;
                return &entry.archive;
            }
        }

        var archive = try WeightArchive.open(io, path);
        errdefer archive.close();

        const path_copy = try self.allocator.dupe(u8, path);
        errdefer self.allocator.free(path_copy);

        try self.entries.append(self.allocator, .{
            .path = path_copy,
            .archive = archive,
            .ref_count = 1,
        });

        return &self.entries.items[self.entries.items.len - 1].archive;
    }

    /// Releases a reference to an archive. If ref_count reaches 0, the mapping is closed and evicted.
    /// Returns true if the archive was completely unmapped and removed.
    pub fn release(self: *MappedFileRegistry, archive: *const WeightArchive) bool {
        self.lock.lock();
        defer self.lock.unlock();

        for (self.entries.items, 0..) |*entry, idx| {
            if (&entry.archive == archive) {
                if (entry.ref_count > 1) {
                    entry.ref_count -= 1;
                    return false;
                } else {
                    entry.archive.close();
                    self.allocator.free(entry.path);
                    _ = self.entries.swapRemove(idx);
                    return true;
                }
            }
        }
        return false;
    }

    /// Returns current number of active archives in registry.
    pub fn count(self: *MappedFileRegistry) usize {
        self.lock.lock();
        defer self.lock.unlock();
        return self.entries.items.len;
    }

    /// Closes and evicts all archives in the registry.
    pub fn clear(self: *MappedFileRegistry) void {
        self.lock.lock();
        defer self.lock.unlock();
        for (self.entries.items) |*entry| {
            entry.archive.close();
            self.allocator.free(entry.path);
        }
        self.entries.clearRetainingCapacity();
    }
};

// ── Unit Tests ────────────────────────────────────────────────────────────────

test "WeightArchiveHeader layout matches cache-line and geometry law" {
    try std.testing.expectEqual(@as(usize, 64), @sizeOf(WeightArchiveHeader));
    try std.testing.expectEqual(@as(usize, 64), @alignOf(WeightArchiveHeader));
    try std.testing.expectEqual(@as(usize, 4096), HEADER_BYTES);
    try std.testing.expectEqual(@as(usize, 20480), geometry.RECORD_BYTES);
    try std.testing.expectEqual(@as(usize, 17408), geometry.CELL_BYTES);
    try std.testing.expectEqual(@as(usize, 3072), geometry.PREFETCH_LABEL_BYTES);
    try std.testing.expectEqual(@as(usize, 16384), TILE_CODE_BYTES);
}

test "TileMetadata layout fits in semantic payload" {
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(TileMetadata));
    try std.testing.expect(@sizeOf(TileMetadata) <= geometry.SEMANTIC_PAYLOAD_BYTES);
}

test "createArchive, mmap open, record slicing, and paging hints" {
    const io = std.testing.io;
    std.Io.Dir.cwd().createDirPath(io, "run") catch {};
    const tmp_path = "run/test_weight_archive_w1.bin";
    defer std.Io.Dir.cwd().deleteFile(io, tmp_path) catch {};

    // Prepare 3 test records with known patterns
    var sample_records: [3]geometry.Record = undefined;
    @memset(std.mem.sliceAsBytes(&sample_records), 0);

    for (&sample_records, 0..) |*rec, i| {
        // Prefetch seal
        @memset(&rec.prefetch_label.zeckendorf_seal, @intCast(0xA0 + i));
        // Cell bytecode header
        rec.cell.header.opcode = @intCast(0x100 + i);
        rec.cell.header.provenance_flags = @intCast(i);
        // Fingerprints (coded tile block)
        for (&rec.cell.fingerprints) |*fp| {
            fp.words[0] = @intCast(0xCAFE0000 + i);
        }
        // Semantic payload metadata
        const meta: *TileMetadata = @ptrCast(@alignCast(&rec.cell.semantic_payload[0]));
        meta.* = .{
            .layer_index = @intCast(i),
            .expert_index = @intCast(10 + i),
            .zeck_index = @intCast(17408 * (i + 1)),
            .bit_budget = 4,
            .quant_bits = 4,
            .reserved = .{ 0, 0, 0 },
            .custom_flags = 0x55AA,
        };
    }

    // Create archive on disk
    try createArchive(io, tmp_path, &sample_records, 0x1234);

    // Verify file size: 4096 (header) + 3 * 20480 (records) = 65,536 bytes (16 pages)
    const expected_file_size = 4096 + 3 * 20480;
    const file = try std.Io.Dir.cwd().openFile(io, tmp_path, .{});
    const stat = try file.stat(io);
    file.close(io);
    try std.testing.expectEqual(expected_file_size, stat.size);

    // Open via WeightArchive
    var archive = try WeightArchive.open(io, tmp_path);
    defer archive.close();

    // Verify header
    try std.testing.expectEqual(ARCHIVE_MAGIC, archive.header().magic);
    try std.testing.expectEqual(ARCHIVE_VERSION, archive.header().version);
    try std.testing.expectEqual(@as(u64, 4096), archive.header().header_bytes);
    try std.testing.expectEqual(@as(u64, 3), archive.header().record_count);
    try std.testing.expectEqual(@as(u64, 3), archive.header().tile_count);
    try std.testing.expectEqual(@as(u64, 20480), archive.header().record_bytes);
    try std.testing.expectEqual(@as(u64, 17408), archive.header().cell_bytes);
    try std.testing.expectEqual(@as(u64, 3072), archive.header().prefetch_bytes);
    try std.testing.expectEqual(@as(u64, 0x1234), archive.header().flags);

    // Verify record counts
    try std.testing.expectEqual(@as(usize, 3), archive.recordCount());
    try std.testing.expectEqual(@as(usize, 3), archive.tileCount());

    // Verify each record content
    for (0..3) |i| {
        const rec = try archive.getRecord(i);
        try std.testing.expectEqual(@as(u8, @intCast(0xA0 + i)), rec.prefetch_label.zeckendorf_seal[0]);

        const cell = try archive.getCell(i);
        try std.testing.expectEqual(@as(u64, @intCast(0x100 + i)), cell.header.opcode);

        const tile_fps = try archive.getTile(i);
        try std.testing.expectEqual(@as(u64, @intCast(0xCAFE0000 + i)), tile_fps[0].words[0]);

        const meta = try archive.getTileMetadata(i);
        try std.testing.expectEqual(@as(u32, @intCast(i)), meta.layer_index);
        try std.testing.expectEqual(@as(u32, @intCast(10 + i)), meta.expert_index);
        try std.testing.expectEqual(@as(u64, @intCast(17408 * (i + 1))), meta.zeck_index);
        try std.testing.expectEqual(@as(u32, 4), meta.bit_budget);
        try std.testing.expectEqual(@as(u8, 4), meta.quant_bits);
        try std.testing.expectEqual(@as(u64, 0x55AA), meta.custom_flags);
    }

    // Out of bounds check
    try std.testing.expectError(WeightArchiveError.RecordIndexOutOfRange, archive.getRecord(3));
    try std.testing.expectError(WeightArchiveError.RecordIndexOutOfRange, archive.getCell(3));

    // Test paging calls (madvise)
    try archive.prefetchRecord(0);
    try archive.releaseRecord(0);
    try archive.prefetchRange(1, 2);
    try archive.releaseRange(1, 2);
    archive.adviseSequential();
    archive.adviseRandom();
    archive.prefetchAll();
    archive.releaseAll();
}

test "MappedFileRegistry caches and shares open archive instances" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    std.Io.Dir.cwd().createDirPath(io, "run") catch {};
    const tmp_path = "run/test_registry_w1.bin";
    defer std.Io.Dir.cwd().deleteFile(io, tmp_path) catch {};

    // Create 1-record archive
    var recs: [1]geometry.Record = undefined;
    @memset(std.mem.sliceAsBytes(&recs), 0);
    recs[0].cell.header.opcode = 0x777;
    try createArchive(io, tmp_path, &recs, 0);

    var reg = MappedFileRegistry.init(allocator);
    defer reg.deinit();

    try std.testing.expectEqual(@as(usize, 0), reg.count());

    // First load
    const a1 = try reg.getOrLoad(io, tmp_path);
    try std.testing.expectEqual(@as(usize, 1), reg.count());
    try std.testing.expectEqual(@as(u64, 0x777), (try a1.getCell(0)).header.opcode);

    // Second load of same path -> returns same cached instance
    const a2 = try reg.getOrLoad(io, tmp_path);
    try std.testing.expectEqual(@as(usize, 1), reg.count());
    try std.testing.expectEqual(a1, a2);

    // Release once -> still 1 entry in registry (ref_count was 2 -> 1)
    const dropped1 = reg.release(a1);
    try std.testing.expect(!dropped1);
    try std.testing.expectEqual(@as(usize, 1), reg.count());

    // Release second time -> unmapped and removed
    const dropped2 = reg.release(a2);
    try std.testing.expect(dropped2);
    try std.testing.expectEqual(@as(usize, 0), reg.count());
}

test "WeightArchive refuses corrupted or malformed files" {
    // Buffer smaller than HEADER_BYTES
    var small_buf: [100]u8 align(std.heap.page_size_min) = @splat(0);
    try std.testing.expectError(WeightArchiveError.FileTooSmall, WeightArchive.openFromBytes(&small_buf));

    // Full sector buffer with invalid magic
    var bad_magic_buf: [HEADER_BYTES]u8 align(std.heap.page_size_min) = @splat(0);
    const hdr_bad: *WeightArchiveHeader = @ptrCast(@alignCast(&bad_magic_buf));
    hdr_bad.* = .{
        .magic = 0xDEADBEEF,
        .version = ARCHIVE_VERSION,
        .header_bytes = HEADER_BYTES,
        .record_count = 0,
        .tile_count = 0,
        .record_bytes = geometry.RECORD_BYTES,
        .cell_bytes = geometry.CELL_BYTES,
        .prefetch_bytes = geometry.PREFETCH_LABEL_BYTES,
        .flags = 0,
    };
    try std.testing.expectError(WeightArchiveError.InvalidMagic, WeightArchive.openFromBytes(&bad_magic_buf));

    // Bad geometry (wrong record stride)
    hdr_bad.magic = ARCHIVE_MAGIC;
    hdr_bad.record_bytes = geometry.RECORD_BYTES + geometry.CACHE_LINE_BYTES;
    try std.testing.expectError(WeightArchiveError.BadGeometry, WeightArchive.openFromBytes(&bad_magic_buf));

    // Truncated (record_count indicates 1 record, but buffer only has header)
    hdr_bad.record_bytes = geometry.RECORD_BYTES;
    hdr_bad.record_count = 1;
    try std.testing.expectError(WeightArchiveError.Truncated, WeightArchive.openFromBytes(&bad_magic_buf));
}

test "WeightArchive supports dense cell archives with prefetch_bytes == 0" {
    var dense_buf: [HEADER_BYTES + geometry.CELL_BYTES]u8 align(std.heap.page_size_min) = @splat(0);
    const hdr: *WeightArchiveHeader = @ptrCast(@alignCast(&dense_buf));
    hdr.* = .{
        .magic = CHPE_MAGIC,
        .version = ARCHIVE_VERSION,
        .header_bytes = HEADER_BYTES,
        .record_count = 1,
        .tile_count = 1,
        .record_bytes = geometry.CELL_BYTES,
        .cell_bytes = geometry.CELL_BYTES,
        .prefetch_bytes = 0,
        .flags = 0,
    };

    const cell_dest: *geometry.Cell = @ptrCast(@alignCast(dense_buf[HEADER_BYTES..].ptr));
    cell_dest.header.opcode = 0x999;

    var archive = try WeightArchive.openFromBytes(&dense_buf);
    defer archive.close();

    try std.testing.expect(archive.isDense());
    try std.testing.expectEqual(@as(usize, 17408), archive.recordBytes());
    try std.testing.expectEqual(@as(usize, 0), archive.prefetchBytes());
    try std.testing.expectEqual(@as(u64, 0x999), (try archive.getCell(0)).header.opcode);
    try std.testing.expectEqual(@as(u64, 0x999), archive.getCellDirect(0).header.opcode);
    try std.testing.expectEqual(@as(u64, 0x999), archive.getCellsPtr()[0].header.opcode);
}
