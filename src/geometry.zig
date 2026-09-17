//! Geometry — the widths every rung of the compression ladder is measured against.
//!
//! This module contains NO algorithm. It is the law, and the compiler is what
//! enforces it: every constant here carries a comptime assertion, so a drift
//! becomes a build failure instead of a corrupted corpus.
//!
//! Sourced from tot_hybrid/BRANCHING.md ("The ladder these branches serve",
//! measured 2026-09-02), Gegenrede drilling plan (2026-09-07), and architectural invariants:
//!   - Invariant A-1: Strict 17,408B cell size (272 x 64B cache lines).
//!   - Invariant A-2: Strict 64B bytecode header (1 cache line).
//!   - Sector Law: 20,480B physical stride (5 x 4096B sectors).
//!   - Pre-Fetch Label Area: 3,072B (with 16B Zeckendorf sequence seal).
//!   - Invariant A-11: 4-hop kill trigger (branch depth > 4 is immediate refusal).
//!   - Invariant A-8: Intent overrules semantics.
//!   - Invariant A-31: ~1.3GB capacity planning observation (not a 1GB hard constraint).
//!
//! Toolchain: Zig 0.17 / Zig 0.16.

const std = @import("std");

// ── Cache-line geometry (RAM side) ───────────────────────────────────────────

/// One 64-byte CPU cache line.
pub const CACHE_LINE_BYTES: usize = 64;

/// Cache lines per cell.
pub const CELL_CACHE_LINES: usize = 272;

/// A cell is L1-resident and cache-line aligned: 272 x 64 (Invariant A-1).
pub const CELL_BYTES: usize = CELL_CACHE_LINES * CACHE_LINE_BYTES; // 17,408

/// Uniform bytecode instruction header width (Invariant A-2).
pub const BYTECODE_HEADER_BYTES: usize = 64;

/// Trailing cryptographic link footer inside semantic_payload.
pub const LINK_FOOTER_BYTES: usize = 24;

/// Routing signature width frozen into fingerprints[0] on write/touch path.
pub const ROUTING_FP_BYTES: usize = 48;

/// Maximum recursive mitosis hop count before hard refusal (Invariant A-11).
pub const MAX_HOP_DEPTH: usize = 4;

/// Invariant A-31: ~1.3GB measured observation for capacity planning, not a 1GB hard constraint.
pub const MODEL_CAPACITY_OBSERVATION_BYTES: usize = 1395864371;

// ── Sector geometry (disk side) ──────────────────────────────────────────────

/// One NVMe sector.
pub const SECTOR_BYTES: usize = 4096;

/// Sectors per on-disk record.
pub const PACKET_SECTORS: usize = 5;

/// A record is sector aligned: 5 x 4096 = 20,480 bytes. Zero sector bleed.
pub const RECORD_BYTES: usize = SECTOR_BYTES * PACKET_SECTORS;

/// The Pre-Fetch Label Area. `CELL_BYTES / SECTOR_BYTES = 4.25` is CORRECT and not a bug.
pub const PREFETCH_LABEL_BYTES: usize = RECORD_BYTES - CELL_BYTES; // 3,072

/// Zeckendorf sequence seal width embedded in PreFetchLabelArea.
pub const ZECKENDORF_SEAL_BYTES: usize = 16;

// ── Core Struct Definitions ──────────────────────────────────────────────────

/// 64-byte uniform bytecode instruction header (Invariant A-2).
/// Aligned to 1 CPU cache line (64 bytes).
pub const BytecodeHeader = extern struct {
    /// 0..7 (8 Bytes): Opcode descriptor (active task type)
    opcode: u64 align(64),
    /// 8..23 (16 Bytes): Subject Identifier token
    subject_id: [16]u8,
    /// 24..39 (16 Bytes): Predicate / Constraint Operator link
    predicate_op: [16]u8,
    /// 40..55 (16 Bytes): Target / Value payload pointer
    target_val: [16]u8,
    /// 56..63 (8 Bytes): Provenance tracking bitmask + Epistemic failure flags + Model epoch ID
    provenance_flags: u64,

    comptime {
        std.debug.assert(@sizeOf(BytecodeHeader) == 64);
        std.debug.assert(@alignOf(BytecodeHeader) == 64);
        std.debug.assert(@offsetOf(BytecodeHeader, "opcode") == 0);
        std.debug.assert(@offsetOf(BytecodeHeader, "subject_id") == 8);
        std.debug.assert(@offsetOf(BytecodeHeader, "predicate_op") == 24);
        std.debug.assert(@offsetOf(BytecodeHeader, "target_val") == 40);
        std.debug.assert(@offsetOf(BytecodeHeader, "provenance_flags") == 56);
    }
};

/// Rel-plane subject_id. Bytes are `lexicon.Term.makeId` — a zero-padding
/// memcpy over a name its caller has ALREADY refused if it was over-wide.
/// This comment used to say "(truncating memcpy)". It no longer truncates:
/// `makeId` asserts `name.len <= 16`, and every caller validates first
/// (`intake_gate.validateIdentifierGbnf`, `scr_grill.parseTermToken`). A stale
/// comment describing a defect as the contract is how the next reader rebuilds
/// the defect.
/// Not comparable to `LstSubjectId`.
pub const RelSubjectId = struct {
    bytes: [16]u8,

    pub fn fromTermBytes(id: [16]u8) RelSubjectId {
        return .{ .bytes = id };
    }

    pub fn eql(self: RelSubjectId, other: RelSubjectId) bool {
        return std.mem.eql(u8, &self.bytes, &other.bytes);
    }
};

/// LST/vault-plane subject_id. Bytes are FNV-1a split hash.
/// Not comparable to `RelSubjectId`.
pub const LstSubjectId = struct {
    bytes: [16]u8,

    pub fn fromHashBytes(id: [16]u8) LstSubjectId {
        return .{ .bytes = id };
    }

    pub fn eql(self: LstSubjectId, other: LstSubjectId) bool {
        return std.mem.eql(u8, &self.bytes, &other.bytes);
    }
};

pub const SubjectIdError = error{
    CrossPlaneIdCompare,
};

/// Always `error.CrossPlaneIdCompare`. Equality is not a byte question.
pub fn compareCrossPlane(rel: RelSubjectId, lst: LstSubjectId) SubjectIdError!void {
    _ = rel;
    _ = lst;
    return error.CrossPlaneIdCompare;
}

// ── provenance_flags bit layout (BytecodeHeader.provenance_flags, 64 bits) ──
//
// Single source of truth for the header bitfield shared by provenance.zig
// (source/outcome/hop/id tokens) and mitosis.zig (chain flags). Both modules
// import these instead of hardcoding shifts, so the two can never collide
// again the way ProvenanceSource (bits 0..3) and the old mitosis bits 0..1
// once did.
//
//   Bits 0..3   (4 bits):  ProvenanceSource        (provenance.zig)
//   Bits 4..7   (4 bits):  Mitosis chain flags      (mitosis.zig)
//   Bits 8..11  (4 bits):  OutcomeKind              (provenance.zig)
//   Bits 12..15 (4 bits):  Hop count (0..4)         (Invariant A-11)
//   Bits 16..31 (16 bits): Embedder ID hash token   (provenance.zig)
//   Bits 32..47 (16 bits): Session ID hash token    (provenance.zig)
//   Bits 48..63 (16 bits): Ticket / Work order ID hash token (provenance.zig)
pub const PROV_SOURCE_SHIFT: u6 = 0;
pub const PROV_SOURCE_BITS: u6 = 4;
pub const MITOSIS_FLAGS_SHIFT: u6 = 4;
pub const MITOSIS_FLAGS_BITS: u6 = 4;
/// Dedicated chain-membership flags. Bits 0..3 remain exclusively owned by
/// ProvenanceSource; these named masks are the single source of truth shared
/// by provenance and mitosis writers.
pub const FLAG_MITOSIS_MEMBER: u64 = 1 << 4;
pub const FLAG_MITOSIS_HAS_NEXT: u64 = 1 << 5;
pub const PROV_SOURCE_MASK: u64 = 0x0F;
pub const MITOSIS_FLAGS_MASK: u64 = FLAG_MITOSIS_MEMBER | FLAG_MITOSIS_HAS_NEXT;
pub const PROV_OUTCOME_SHIFT: u6 = 8;
pub const PROV_OUTCOME_BITS: u6 = 4;
pub const PROV_HOP_SHIFT: u6 = 12;
pub const PROV_HOP_BITS: u6 = 4;
pub const PROV_EMBEDDER_SHIFT: u6 = 16;
pub const PROV_EMBEDDER_BITS: u6 = 16;
pub const PROV_SESSION_SHIFT: u6 = 32;
pub const PROV_SESSION_BITS: u6 = 16;
pub const PROV_TICKET_SHIFT: u6 = 48;
pub const PROV_TICKET_BITS: u6 = 16;

comptime {
    // Every field must fit before the next one starts, and the whole layout
    // must fit exactly in 64 bits with no overlap and no gap.
    // (Widened to usize for the arithmetic: shift/bit-width constants are u6
    // because that's what a u64 shift amount requires, but u6 tops out at 63
    // and the last boundary check below is exactly 64.)
    std.debug.assert(@as(usize, PROV_SOURCE_SHIFT) + PROV_SOURCE_BITS == MITOSIS_FLAGS_SHIFT);
    std.debug.assert(@as(usize, MITOSIS_FLAGS_SHIFT) + MITOSIS_FLAGS_BITS == PROV_OUTCOME_SHIFT);
    std.debug.assert(@as(usize, PROV_OUTCOME_SHIFT) + PROV_OUTCOME_BITS == PROV_HOP_SHIFT);
    std.debug.assert(@as(usize, PROV_HOP_SHIFT) + PROV_HOP_BITS == PROV_EMBEDDER_SHIFT);
    std.debug.assert(@as(usize, PROV_EMBEDDER_SHIFT) + PROV_EMBEDDER_BITS == PROV_SESSION_SHIFT);
    std.debug.assert(@as(usize, PROV_SESSION_SHIFT) + PROV_SESSION_BITS == PROV_TICKET_SHIFT);
    std.debug.assert(@as(usize, PROV_TICKET_SHIFT) + PROV_TICKET_BITS == 64);
    // Hop count must be representable: max value MAX_HOP_DEPTH (4) fits in 4 bits.
    std.debug.assert(MAX_HOP_DEPTH < (@as(usize, 1) << PROV_HOP_BITS));
    std.debug.assert((FLAG_MITOSIS_MEMBER | FLAG_MITOSIS_HAS_NEXT) & PROV_SOURCE_MASK == 0);
    std.debug.assert(MITOSIS_FLAGS_MASK == 0x30);
}

/// 3,072-byte Pre-Fetch Label Area preceding each cell on disk.
pub const PreFetchLabelArea = extern struct {
    /// 16-byte cryptographic Zeckendorf sequence seal (Task I-GEO-02)
    zeckendorf_seal: [ZECKENDORF_SEAL_BYTES]u8,
    /// Remaining reserved label and pre-fetch metadata (3,056 bytes)
    reserved: [PREFETCH_LABEL_BYTES - ZECKENDORF_SEAL_BYTES]u8,

    comptime {
        std.debug.assert(@sizeOf(PreFetchLabelArea) == 3072);
        std.debug.assert(@sizeOf(PreFetchLabelArea) == PREFETCH_LABEL_BYTES);
    }
};

pub const FINGERPRINT_VECTOR_WORDS: usize = 64; // 64 x u64 = 512 bytes
pub const FINGERPRINT_VECTORS: usize = 32;       // 32 x 512B = 16,384 bytes

/// Engine-plane high-float pack (f64 4096-d or f32 8192-d). Not the cell.
/// Two fingerprint tiles. Overflow splits; do not grow CELL_BYTES to 32 KiB.
pub const ENGINE_PACK_BYTES: usize = 32768;

/// Reptile spoke named projection (not silent truncate, not EBM 4096-d).
/// BioCLIP-2.5 `embed_dim`=1024 f32 → 8 × 512B slots at fingerprints[0..8).
/// DINOv3 ViT-S/16 `hidden_size`=384 f32 → 3 × 512B slots at fingerprints[8..11).
/// Slots [11..32) stay zero; that remainder is the 4096-d EBM width only.
pub const BIOCLIP_F32_DIM: usize = 1024;
pub const DINOV3_F32_DIM: usize = 384;
pub const EBM_F32_DIM: usize = 4096;
pub const BIOCLIP_SLOT_BEGIN: usize = 0;
pub const BIOCLIP_SLOT_COUNT: usize = (BIOCLIP_F32_DIM * @sizeOf(f32)) / 512; // 8
pub const DINOV3_SLOT_BEGIN: usize = BIOCLIP_SLOT_BEGIN + BIOCLIP_SLOT_COUNT; // 8
pub const DINOV3_SLOT_COUNT: usize = (DINOV3_F32_DIM * @sizeOf(f32)) / 512; // 3

/// Text embedder (provenance.EMBEDDER_ID = embeddinggemma-300m): hidden_size=768 f32.
/// 768 × 4 = 3072 B = exactly 6 × 512 B slots at fingerprints[0..6).
/// Not BioCLIP (1024-d / 8 slots). Not EBM (4096-d / 32 slots). Do not pad.
pub const EMBEDDINGGEMMA_F32_DIM: usize = 768;
pub const EMBEDDINGGEMMA_SLOT_BEGIN: usize = 0;
pub const EMBEDDINGGEMMA_SLOT_COUNT: usize = (EMBEDDINGGEMMA_F32_DIM * @sizeOf(f32)) / 512; // 6

pub const FingerprintVector = extern struct {
    words: [FINGERPRINT_VECTOR_WORDS]u64 align(64),

    comptime {
        std.debug.assert(@sizeOf(FingerprintVector) == 512);
        std.debug.assert(@alignOf(FingerprintVector) == 64);
    }
};

pub const SEMANTIC_PAYLOAD_BYTES: usize = 960;

/// The work cell: fixed-layout 17,408-byte (272 cache line) record (Invariant A-1).
pub const Cell = extern struct {
    header: BytecodeHeader,
    fingerprints: [FINGERPRINT_VECTORS]FingerprintVector align(64),
    semantic_payload: [SEMANTIC_PAYLOAD_BYTES]u8 align(64),

    comptime {
        std.debug.assert(@sizeOf(Cell) == 17408);
        std.debug.assert(@sizeOf(Cell) == CELL_BYTES);
        std.debug.assert(@alignOf(Cell) == 64);
        std.debug.assert(@offsetOf(Cell, "header") == 0);
        std.debug.assert(@offsetOf(Cell, "fingerprints") == 64);
        std.debug.assert(@offsetOf(Cell, "semantic_payload") == 16448);
    }
};

/// On-disk record: 20,480 bytes = 5 x 4096B sectors.
/// Layout: PreFetchLabelArea (3,072B) + Cell (17,408B).
pub const Record = extern struct {
    prefetch_label: PreFetchLabelArea,
    cell: Cell,

    comptime {
        std.debug.assert(@sizeOf(Record) == 20480);
        std.debug.assert(@sizeOf(Record) == RECORD_BYTES);
        std.debug.assert(@offsetOf(Record, "prefetch_label") == 0);
        std.debug.assert(@offsetOf(Record, "cell") == PREFETCH_LABEL_BYTES);
    }
};

comptime {
    // Two different alignment targets, on purpose. Do not "fix" one to match
    // the other: 17,408 is cache-line law, 20,480 is sector law. Both correct.
    std.debug.assert(CELL_BYTES == 17408);
    std.debug.assert(CELL_BYTES % CACHE_LINE_BYTES == 0);

    std.debug.assert(RECORD_BYTES == 20480);
    std.debug.assert(RECORD_BYTES % SECTOR_BYTES == 0);

    // The record must actually have room for the cell plus its label area.
    std.debug.assert(RECORD_BYTES > CELL_BYTES);
    std.debug.assert(PREFETCH_LABEL_BYTES == 3072);

    // Invariant A-2 validation
    std.debug.assert(BYTECODE_HEADER_BYTES == 64);
    std.debug.assert(@sizeOf(BytecodeHeader) == BYTECODE_HEADER_BYTES);

    // Seal & Footer geometry
    std.debug.assert(ZECKENDORF_SEAL_BYTES == 16);
    std.debug.assert(LINK_FOOTER_BYTES == 24);
    std.debug.assert(ROUTING_FP_BYTES == 48);

    // Invariant A-11 validation
    std.debug.assert(MAX_HOP_DEPTH == 4);

    // Reptile multi-vector layout: 1024-d and 384-d pack exactly; remainder is EBM-only.
    std.debug.assert(BIOCLIP_SLOT_COUNT == 8);
    std.debug.assert(DINOV3_SLOT_COUNT == 3);
    std.debug.assert(DINOV3_SLOT_BEGIN == 8);
    std.debug.assert(BIOCLIP_SLOT_COUNT + DINOV3_SLOT_COUNT < FINGERPRINT_VECTORS);
    std.debug.assert(BIOCLIP_F32_DIM * @sizeOf(f32) == BIOCLIP_SLOT_COUNT * 512);
    std.debug.assert(DINOV3_F32_DIM * @sizeOf(f32) == DINOV3_SLOT_COUNT * 512);
    std.debug.assert(EBM_F32_DIM * @sizeOf(f32) == FINGERPRINT_VECTORS * 512);

    // embeddinggemma-300m: 768 f32 → 6 slots. Distinct from BioCLIP's 8-slot map.
    std.debug.assert(EMBEDDINGGEMMA_F32_DIM == 768);
    std.debug.assert(EMBEDDINGGEMMA_SLOT_BEGIN == 0);
    std.debug.assert(EMBEDDINGGEMMA_SLOT_COUNT == 6);
    std.debug.assert(EMBEDDINGGEMMA_SLOT_COUNT != BIOCLIP_SLOT_COUNT);
    std.debug.assert(EMBEDDINGGEMMA_F32_DIM * @sizeOf(f32) == EMBEDDINGGEMMA_SLOT_COUNT * 512);
    std.debug.assert(EMBEDDINGGEMMA_F32_DIM * @sizeOf(f32) == 3072);
    std.debug.assert(EMBEDDINGGEMMA_F32_DIM != EBM_F32_DIM);
}

// ── The compression ladder ───────────────────────────────────────────────────

/// A rung of the ladder. `width_bytes` is a real width on disk, not a sketch.
pub const Rung = struct {
    ordinal: u8,
    name: []const u8,
    width_bytes: usize,
    /// Compression is EARNED BY USE and never applied on ingest.
    built: bool,
};

/// Rung 1 — forensic intake: fp32 4096-d embedding.
pub const R1_FORENSIC_BYTES: usize = 4096 * @sizeOf(f32);

/// Rung 2 — cell fingerprint: [64]u64 bitvector.
pub const R2_FINGERPRINT_BYTES: usize = 64 * @sizeOf(u64);

/// Rung 3 — EBM proposal: [4]u64 bitvector.
pub const R3_EBM_BYTES: usize = 4 * @sizeOf(u64);

/// Rung 4 — class supercell: one Cell folding `SUPERCELL_MEMBERS` members.
pub const R4_SUPERCELL_BYTES: usize = CELL_BYTES;

/// Members folded into one rung-4 supercell.
pub const SUPERCELL_MEMBERS: usize = 20;

pub const LADDER = [_]Rung{
    .{ .ordinal = 1, .name = "forensic intake", .width_bytes = R1_FORENSIC_BYTES, .built = true },
    .{ .ordinal = 2, .name = "cell fingerprint", .width_bytes = R2_FINGERPRINT_BYTES, .built = true },
    .{ .ordinal = 3, .name = "EBM proposal", .width_bytes = R3_EBM_BYTES, .built = true },
    .{ .ordinal = 4, .name = "class supercell", .width_bytes = R4_SUPERCELL_BYTES, .built = false },
};

comptime {
    std.debug.assert(R1_FORENSIC_BYTES == 16384);
    std.debug.assert(ENGINE_PACK_BYTES == 32768);
    std.debug.assert(ENGINE_PACK_BYTES == 2 * (FINGERPRINT_VECTORS * 512));
    std.debug.assert(ENGINE_PACK_BYTES != CELL_BYTES);
    std.debug.assert(R2_FINGERPRINT_BYTES == 512);
    std.debug.assert(R3_EBM_BYTES == 32);
    std.debug.assert(R4_SUPERCELL_BYTES == CELL_BYTES);

    // Rungs 1-3 compress at the VECTOR level and must strictly narrow.
    std.debug.assert(R1_FORENSIC_BYTES > R2_FINGERPRINT_BYTES);
    std.debug.assert(R2_FINGERPRINT_BYTES > R3_EBM_BYTES);

    // Ordinals are dense and in order.
    for (LADDER, 0..) |rung, i| std.debug.assert(rung.ordinal == i + 1);
}

/// Byte-width reduction, rung 1 -> rung 2.
///
/// NOTE: 16,384 / 512 = 32x, not the "512x" quoted in BRANCHING.md. That 512
/// is the whole vector-level ladder measured in BITS: a 4096-d fp32 embedding
/// (131,072 bits) down to a 256-bit EBM proposal is 512x. Both numbers are
/// right; they measure different spans. Asserted here so the distinction
/// cannot quietly drift.
pub const R1_TO_R2_BYTES_RATIO: usize = R1_FORENSIC_BYTES / R2_FINGERPRINT_BYTES;

/// Vector-level compression across rungs 1 -> 3, in bits. This is the 512x.
pub const VECTOR_COMPRESSION_BITS: usize = (R1_FORENSIC_BYTES * 8) / (R3_EBM_BYTES * 8);

comptime {
    std.debug.assert(R1_TO_R2_BYTES_RATIO == 32);
    std.debug.assert(VECTOR_COMPRESSION_BITS == 512);
}

// ── Fold law ─────────────────────────────────────────────────────────────────

/// How members collapse into a class supercell.
///
/// Measured on two synthetic classes at 10% noise (BRANCHING.md):
///
///   fold      intra    inter    density
///   OR        49.2%     3.3%    0.98 saturated
///   MAJORITY   0.0%    51.0%    0.49
///
/// OR looks like set-union and destroys class separation. Any rung-4 work must
/// carry a test that fails on OR.
pub const Fold = enum { majority, @"or" };

/// The only legal fold.
pub const LEGAL_FOLD: Fold = .majority;

comptime {
    std.debug.assert(LEGAL_FOLD == .majority);
}

// ── Overflow & Kill Trigger law ──────────────────────────────────────────────

/// Never truncate. Demotion down the ladder is lossy by design; splitting is
/// lossless by law. A quantizer must never become a truncator.
///
/// Overflowing a width is a mitosis signal, not a trim instruction.
pub const GeometryError = error{
    /// The payload exceeds its rung width or mitosis exceeds 4 hops. Caller must split, never trim.
    SplitRequired,
};

/// Returns the payload length, or `SplitRequired` if it will not fit `width`.
/// There is deliberately no `fitOrTrim` counterpart.
pub fn fitOrSplit(len: usize, width: usize) GeometryError!usize {
    if (len > width) return GeometryError.SplitRequired;
    return len;
}

/// Enforces the 4-hop kill trigger (Invariant A-11).
/// If mitosis branch depth exceeds 4, immediately throws SplitRequired / hard refusal.
pub fn checkHopDepth(depth: usize) GeometryError!usize {
    if (depth > MAX_HOP_DEPTH) return GeometryError.SplitRequired;
    return depth;
}

// ── Tests ────────────────────────────────────────────────────────────────────

test "cell is cache-line aligned and record is sector aligned" {
    try std.testing.expectEqual(@as(usize, 17408), CELL_BYTES);
    try std.testing.expectEqual(@as(usize, 272), CELL_BYTES / CACHE_LINE_BYTES);
    try std.testing.expectEqual(@as(usize, 20480), RECORD_BYTES);
    try std.testing.expectEqual(@as(usize, 5), RECORD_BYTES / SECTOR_BYTES);
}

test "the 4.25 ratio is correct, not a bug" {
    // A cell does not divide evenly into sectors. That is the point: the pad
    // is the Pre-Fetch Label Area, and the cell never hits disk unpadded.
    try std.testing.expect(CELL_BYTES % SECTOR_BYTES != 0);
    try std.testing.expectEqual(@as(usize, 3072), PREFETCH_LABEL_BYTES);
    try std.testing.expectEqual(CELL_BYTES + PREFETCH_LABEL_BYTES, RECORD_BYTES);
}

test "ladder widths narrow at the vector level" {
    try std.testing.expectEqual(@as(usize, 32), R1_TO_R2_BYTES_RATIO);
    try std.testing.expectEqual(@as(usize, 512), VECTOR_COMPRESSION_BITS);
    try std.testing.expectEqual(@as(usize, 4), LADDER.len);
    try std.testing.expect(!LADDER[3].built); // rung 4 is unbuilt, honestly recorded
}

test "supercell folds 20 members into one cell width" {
    try std.testing.expectEqual(@as(usize, 20), SUPERCELL_MEMBERS);
    try std.testing.expectEqual(CELL_BYTES, R4_SUPERCELL_BYTES);
}

test "fold law is MAJORITY and this test fails if it becomes OR" {
    try std.testing.expectEqual(Fold.majority, LEGAL_FOLD);
    try std.testing.expect(LEGAL_FOLD != .@"or");
}

test "overflow signals a split, never a trim" {
    try std.testing.expectEqual(@as(usize, 100), try fitOrSplit(100, CELL_BYTES));
    try std.testing.expectEqual(CELL_BYTES, try fitOrSplit(CELL_BYTES, CELL_BYTES));
    try std.testing.expectError(
        GeometryError.SplitRequired,
        fitOrSplit(CELL_BYTES + 1, CELL_BYTES),
    );
}

test "bytecode header is exactly 64 bytes and cache-line aligned (Invariant A-2)" {
    try std.testing.expectEqual(@as(usize, 64), BYTECODE_HEADER_BYTES);
    try std.testing.expectEqual(@as(usize, 64), @sizeOf(BytecodeHeader));
    try std.testing.expectEqual(@as(usize, 64), @alignOf(BytecodeHeader));
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(BytecodeHeader, "opcode"));
    try std.testing.expectEqual(@as(usize, 8), @offsetOf(BytecodeHeader, "subject_id"));
    try std.testing.expectEqual(@as(usize, 24), @offsetOf(BytecodeHeader, "predicate_op"));
    try std.testing.expectEqual(@as(usize, 40), @offsetOf(BytecodeHeader, "target_val"));
    try std.testing.expectEqual(@as(usize, 56), @offsetOf(BytecodeHeader, "provenance_flags"));
}

test "cell struct matches 17,408 bytes with exact field offsets (Invariant A-1)" {
    try std.testing.expectEqual(@as(usize, 17408), @sizeOf(Cell));
    try std.testing.expectEqual(@as(usize, 64), @alignOf(Cell));
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(Cell, "header"));
    try std.testing.expectEqual(@as(usize, 64), @offsetOf(Cell, "fingerprints"));
    try std.testing.expectEqual(@as(usize, 16448), @offsetOf(Cell, "semantic_payload"));
    try std.testing.expectEqual(@as(usize, 960), SEMANTIC_PAYLOAD_BYTES);
    try std.testing.expectEqual(@as(usize, 16384), FINGERPRINT_VECTORS * 512);
}

test "record struct matches 20,480 bytes (5 sectors) with 3,072-byte prefetch area" {
    try std.testing.expectEqual(@as(usize, 20480), @sizeOf(Record));
    try std.testing.expectEqual(@as(usize, 3072), @sizeOf(PreFetchLabelArea));
    try std.testing.expectEqual(@as(usize, 16), ZECKENDORF_SEAL_BYTES);
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(Record, "prefetch_label"));
    try std.testing.expectEqual(@as(usize, 3072), @offsetOf(Record, "cell"));
}

test "reptile BioCLIP 1024-d occupies 8 slots and DINOv3 384-d occupies 3" {
    try std.testing.expectEqual(@as(usize, 1024), BIOCLIP_F32_DIM);
    try std.testing.expectEqual(@as(usize, 384), DINOV3_F32_DIM);
    try std.testing.expectEqual(@as(usize, 8), BIOCLIP_SLOT_COUNT);
    try std.testing.expectEqual(@as(usize, 3), DINOV3_SLOT_COUNT);
    try std.testing.expectEqual(@as(usize, 0), BIOCLIP_SLOT_BEGIN);
    try std.testing.expectEqual(@as(usize, 8), DINOV3_SLOT_BEGIN);
    try std.testing.expectEqual(@as(usize, 11), DINOV3_SLOT_BEGIN + DINOV3_SLOT_COUNT);
    try std.testing.expect(DINOV3_SLOT_BEGIN + DINOV3_SLOT_COUNT < FINGERPRINT_VECTORS);
}

test "embeddinggemma-300m 768-d occupies 6 fingerprint slots not BioCLIP 8" {
    try std.testing.expectEqual(@as(usize, 768), EMBEDDINGGEMMA_F32_DIM);
    try std.testing.expectEqual(@as(usize, 6), EMBEDDINGGEMMA_SLOT_COUNT);
    try std.testing.expectEqual(@as(usize, 0), EMBEDDINGGEMMA_SLOT_BEGIN);
    try std.testing.expectEqual(@as(usize, 3072), EMBEDDINGGEMMA_F32_DIM * @sizeOf(f32));
    try std.testing.expect(EMBEDDINGGEMMA_SLOT_COUNT != BIOCLIP_SLOT_COUNT);
    try std.testing.expect(EMBEDDINGGEMMA_F32_DIM != EBM_F32_DIM);
}

test "4-hop kill trigger invariant enforces depth <= 4 (Invariant A-11)" {
    try std.testing.expectEqual(@as(usize, 4), MAX_HOP_DEPTH);
    try std.testing.expectEqual(@as(usize, 0), try checkHopDepth(0));
    try std.testing.expectEqual(@as(usize, 4), try checkHopDepth(4));
    try std.testing.expectError(GeometryError.SplitRequired, checkHopDepth(5));
}

test "RelSubjectId.eql is byte identity" {
    var left: [16]u8 = @splat(0);
    var right: [16]u8 = @splat(0);
    @memcpy(left[0..7], "subject");
    @memcpy(right[0..7], "subject");
    try std.testing.expect(RelSubjectId.fromTermBytes(left).eql(RelSubjectId.fromTermBytes(right)));
    right[0] = 'S';
    try std.testing.expect(!RelSubjectId.fromTermBytes(left).eql(RelSubjectId.fromTermBytes(right)));
}

test "LstSubjectId.eql is byte identity" {
    var left: [16]u8 = @splat(0);
    var right: [16]u8 = @splat(0);
    @memcpy(left[0..7], "symbol_");
    @memcpy(right[0..7], "symbol_");
    try std.testing.expect(LstSubjectId.fromHashBytes(left).eql(LstSubjectId.fromHashBytes(right)));
    right[0] = 'S';
    try std.testing.expect(!LstSubjectId.fromHashBytes(left).eql(LstSubjectId.fromHashBytes(right)));
}

test "compareCrossPlane returns CrossPlaneIdCompare even when both bytes are all-zero" {
    const rel = RelSubjectId.fromTermBytes(@splat(0));
    const lst = LstSubjectId.fromHashBytes(@splat(0));
    try std.testing.expectError(error.CrossPlaneIdCompare, compareCrossPlane(rel, lst));
}

test "wrapper sizes are 16 and BytecodeHeader stays 64 with subject_id at 8" {
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(RelSubjectId));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(LstSubjectId));
    try std.testing.expectEqual(@as(usize, 64), @sizeOf(BytecodeHeader));
    try std.testing.expectEqual(@as(usize, 8), @offsetOf(BytecodeHeader, "subject_id"));
}

test "a raw 17-byte memcpy into a 16-byte id loses byte 16 silently" {
    // This test used to be named "memcpy truncation ... keeps byte 15 as p"
    // and stood as the tree's last artifact ASSERTING that truncation is the
    // contract. It is not, and a test that pins a defect is worse than no test:
    // it fails when someone fixes the defect, so the defect looks load-bearing.
    //
    // What it demonstrates now is WHY every door refuses instead of trimming.
    // `geometry` cannot import `lexicon` (the dependency runs the other way),
    // so the doors are named, not called:
    //   lexicon.Term.makeId          asserts name.len <= 16
    //   intake_gate.validateIdentifierGbnf -> TermLengthExceeded
    //   scr_grill.parseTermToken           -> GrillError.NotEvaluable
    //   mcp_bridge.parseId16               -> IdError.TermLengthExceeded
    // Nothing in this file trims: `RelSubjectId.fromTermBytes` and
    // `LstSubjectId.fromHashBytes` take `[16]u8` and cannot.
    const a = "abcdefghijklmnopq"; // 17
    const b = "abcdefghijklmnopZ"; // 17, differs only at byte 17

    var id_a: [16]u8 = @splat(0);
    var id_b: [16]u8 = @splat(0);
    @memcpy(&id_a, a[0..16]);
    @memcpy(&id_b, b[0..16]);

    // The 16th byte survives; the 17th is what was lost.
    try std.testing.expectEqual(@as(u8, 'p'), id_a[15]);

    // And this is the whole reason for the refusals: two distinct names become
    // one id, so the bank answers a query for either with the other's row
    // instead of reporting an empty rectangle.
    try std.testing.expectEqualSlices(u8, &id_a, &id_b);
}

test "the id wrappers cannot truncate: they take 16 bytes, not a name" {
    const raw: [16]u8 = "sixteen_bytes_ok".*;
    try std.testing.expectEqualSlices(u8, &raw, &RelSubjectId.fromTermBytes(raw).bytes);
    try std.testing.expectEqualSlices(u8, &raw, &LstSubjectId.fromHashBytes(raw).bytes);
}
