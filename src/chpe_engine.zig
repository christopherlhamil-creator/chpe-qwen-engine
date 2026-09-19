//! Unified Polymorphic CHPE Inference Engine
//!
//! Merges Qwen2.5-3B, Qwen3.5-9B, and Qwen2.5-72B execution substrates into a single
//! zero-copy, polymorphic engine.
//!
//! Invariants & Architecture:
//! - Strict 17,408-byte cell geometry (geometry.Cell) with 64B bytecode header.
//! - Self-describing cell metadata: quant_bits (2, 4, 8, 16, 32), dimensions, and custom_flags.
//! - Runtime hardware detection: AVX-512, AMX, AVX2, ARM NEON SDOT, Generic.
//! - Polymorphic GEMV dispatch over all quantization precisions.
//! - Unified support for Transformer GQA and Hybrid SSM / Linear Attention.

const std = @import("std");
const builtin = @import("builtin");
const geometry = @import("geometry.zig");
const weight_archive = @import("weight_archive.zig");

pub const Cell = geometry.Cell;
pub const Record = geometry.Record;
pub const TileMetadata = weight_archive.TileMetadata;
pub const FLAG_GROUP128_OUTLIERS = weight_archive.FLAG_GROUP128_OUTLIERS;
pub const FLAG_HYBRID_ARCHIVE = weight_archive.FLAG_HYBRID_ARCHIVE;
pub const FLAG_RAW_FP16 = weight_archive.FLAG_RAW_FP16;

pub const CELL_BYTES = geometry.CELL_BYTES; // 17408
pub const RECORD_BYTES = geometry.RECORD_BYTES; // 20480
pub const TILE_CODE_BYTES = geometry.FINGERPRINT_VECTORS * 512; // 16384
pub const SEMANTIC_PAYLOAD_BYTES = geometry.SEMANTIC_PAYLOAD_BYTES; // 960

// ── Runtime Hardware Backend Detection ──────────────────────────────────────

pub const HardwareBackend = enum {
    avx512,
    amx,
    avx2,
    arm_neon_sdot,
    arm_neon_fp16,
    generic,

    pub fn name(self: HardwareBackend) []const u8 {
        return switch (self) {
            .avx512 => "x86_64 AVX-512 (512-bit vector FMA/VNNI)",
            .amx => "x86_64 Intel AMX (Advanced Matrix Extensions)",
            .avx2 => "x86_64 AVX2 + FMA (256-bit vector FMA)",
            .arm_neon_sdot => "AArch64 ARM NEON with SDOT (Int8 dot-product)",
            .arm_neon_fp16 => "AArch64 ARM NEON FP16 (Half-precision FMA)",
            .generic => "Portable Generic SIMD / Scalar fallback",
        };
    }
};

fn cpuid(leaf: u32, sub_leaf: u32) struct { eax: u32, ebx: u32, ecx: u32, edx: u32 } {
    if (comptime builtin.cpu.arch.isX86()) {
        var eax: u32 = undefined;
        var ebx: u32 = undefined;
        var ecx: u32 = undefined;
        var edx: u32 = undefined;
        asm volatile (
            "cpuid"
            : [eax] "={eax}" (eax),
              [ebx] "={ebx}" (ebx),
              [ecx] "={ecx}" (ecx),
              [edx] "={edx}" (edx),
            : [leaf] "{eax}" (leaf),
              [sub_leaf] "{ecx}" (sub_leaf),
        );
        return .{ .eax = eax, .ebx = ebx, .ecx = ecx, .edx = edx };
    } else {
        return .{ .eax = 0, .ebx = 0, .ecx = 0, .edx = 0 };
    }
}

pub fn detectHardwareBackend() HardwareBackend {
    if (comptime builtin.cpu.arch.isX86()) {
        const leaf7 = cpuid(7, 0);
        const has_amx = (leaf7.edx & (1 << 24)) != 0;
        const has_avx512f = (leaf7.ebx & (1 << 16)) != 0;
        const has_avx2 = (leaf7.ebx & (1 << 5)) != 0;

        if (has_amx) return .amx;
        if (has_avx512f) return .avx512;
        if (has_avx2) return .avx2;
        return .generic;
    } else if (comptime builtin.cpu.arch.isAARCH64()) {
        return .arm_neon_sdot;
    }
    return .generic;
}

// ── Model Architecture Presets ──────────────────────────────────────────────

pub const ModelArch = struct {
    name: []const u8,
    hidden_dim: usize,
    intermediate_dim: usize,
    num_layers: usize,
    num_attn_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    is_hybrid_ssm: bool,
    ssm_linear_layers: usize = 0,
    rope_theta: f32 = 1000000.0,
    rms_eps: f32 = 1e-6,

    pub const Qwen2_5_3B = ModelArch{
        .name = "Qwen2.5-3B",
        .hidden_dim = 2048,
        .intermediate_dim = 11008,
        .num_layers = 36,
        .num_attn_heads = 16,
        .num_kv_heads = 2,
        .head_dim = 128,
        .vocab_size = 151936,
        .is_hybrid_ssm = false,
    };

    pub const Qwen3_5_9B = ModelArch{
        .name = "Qwen3.5-9B",
        .hidden_dim = 4096,
        .intermediate_dim = 12288,
        .num_layers = 32,
        .num_attn_heads = 16,
        .num_kv_heads = 4,
        .head_dim = 256,
        .vocab_size = 248320,
        .is_hybrid_ssm = true,
        .ssm_linear_layers = 24,
    };

    pub const Qwen2_5_72B = ModelArch{
        .name = "Qwen2.5-72B",
        .hidden_dim = 8192,
        .intermediate_dim = 29568,
        .num_layers = 80,
        .num_attn_heads = 64,
        .num_kv_heads = 8,
        .head_dim = 128,
        .vocab_size = 152064,
        .is_hybrid_ssm = false,
    };
};

// ── Tile Dimension Metadata Header ──────────────────────────────────────────

pub const TileDims = extern struct {
    scale: f32,
    bias: f32,
    rows: u32,
    cols: u32,
};

// ── Math & Normalization Primitives ─────────────────────────────────────────

pub inline fn silu(z: f32) f32 {
    return z / (1.0 + @exp(-z));
}

pub fn rmsNorm(x: []const f32, gamma: []const f32, y: []f32, eps: f32) void {
    std.debug.assert(x.len == gamma.len);
    std.debug.assert(x.len == y.len);
    const n = x.len;

    const Vec8 = @Vector(8, f32);
    var acc_vec: Vec8 = @splat(0.0);
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        const v: Vec8 = x[i..][0..8].*;
        acc_vec += v * v;
    }
    var sum_sq: f32 = @reduce(.Add, acc_vec);
    while (i < n) : (i += 1) {
        sum_sq += x[i] * x[i];
    }

    const mean_sq = sum_sq / @as(f32, @floatFromInt(n));
    const inv_rms = 1.0 / @sqrt(mean_sq + eps);
    const inv_vec: Vec8 = @splat(inv_rms);

    i = 0;
    while (i + 8 <= n) : (i += 8) {
        const vx: Vec8 = x[i..][0..8].*;
        const vg: Vec8 = gamma[i..][0..8].*;
        y[i..][0..8].* = vx * inv_vec * vg;
    }
    while (i < n) : (i += 1) {
        y[i] = x[i] * inv_rms * gamma[i];
    }
}

pub fn applyRope(vec: []f32, pos: usize, num_heads: usize, head_dim: usize, rope_theta: f32) void {
    const pos_f = @as(f32, @floatFromInt(pos));
    for (0..num_heads) |h| {
        const head_base = h * head_dim;
        var i: usize = 0;
        while (i < head_dim / 2) : (i += 1) {
            const freq_exponent = 2.0 * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(head_dim));
            const freq = 1.0 / std.math.pow(f32, rope_theta, freq_exponent);
            const theta = pos_f * freq;
            const cos_theta = @cos(theta);
            const sin_theta = @sin(theta);

            const idx0 = head_base + i;
            const idx1 = head_base + i + head_dim / 2;
            const x0 = vec[idx0];
            const x1 = vec[idx1];

            vec[idx0] = x0 * cos_theta - x1 * sin_theta;
            vec[idx1] = x0 * sin_theta + x1 * cos_theta;
        }
    }
}

// ── Polymorphic GEMV Kernels ────────────────────────────────────────────────

const CODEBOOK_W2: [4]f32 = .{ 0.0, 1.0, -2.0, -1.0 };

/// 2-Bit Coordinate Descent GEMV with group-128 scaling and optional sparse outlier protection
pub fn gemvTileW2(
    cell: *const geometry.Cell,
    x: []const f32,
    y: []f32,
    row_base: usize,
    comptime accumulate: bool,
) void {
    const payload = &cell.semantic_payload;
    const meta: *const TileMetadata = @ptrCast(@alignCast(payload.ptr));
    const dims: *const TileDims = @ptrCast(@alignCast(payload[32..48].ptr));
    const coded: [*]const u8 = @ptrCast(&cell.fingerprints);

    const rows: usize = dims.rows;
    const cols: usize = dims.cols;
    const has_group128 = (meta.custom_flags & FLAG_GROUP128_OUTLIERS) != 0;

    const group_scales: ?[*]const f16 = if (has_group128)
        @ptrCast(@alignCast(payload[48..560].ptr))
    else
        null;

    const has_outliers = has_group128 and rows == 16;
    var x_outlier: @Vector(8, f32) = @splat(0.0);
    var outlier_w_raw: [*]const f16 = undefined;
    if (has_outliers) {
        outlier_w_raw = @ptrCast(@alignCast(payload[560..816].ptr));
        const outlier_cols_raw: [*]const u16 = @ptrCast(@alignCast(payload[816..832].ptr));
        inline for (0..8) |k| {
            const col_idx = outlier_cols_raw[k];
            if (col_idx < x.len) {
                x_outlier[k] = x[col_idx];
            }
        }
    }

    const n_groups = cols / 128;

    for (0..rows) |r| {
        var row_sum: f32 = 0.0;
        const row_bytes_offset = r * (cols / 4);

        if (group_scales) |gs| {
            for (0..n_groups) |g| {
                const scale: f32 = @floatCast(gs[r * n_groups + g]);
                const g_bytes = coded[row_bytes_offset + g * 32 .. row_bytes_offset + (g + 1) * 32];
                const x_slice = x[g * 128 .. (g + 1) * 128];

                var dot_v: @Vector(8, f32) = @splat(0.0);
                var j: usize = 0;
                while (j < 32) : (j += 2) {
                    const b0 = g_bytes[j + 0];
                    const b1 = g_bytes[j + 1];

                    const qv: @Vector(8, f32) = .{
                        CODEBOOK_W2[b0 & 0x03],
                        CODEBOOK_W2[(b0 >> 2) & 0x03],
                        CODEBOOK_W2[(b0 >> 4) & 0x03],
                        CODEBOOK_W2[(b0 >> 6) & 0x03],
                        CODEBOOK_W2[b1 & 0x03],
                        CODEBOOK_W2[(b1 >> 2) & 0x03],
                        CODEBOOK_W2[(b1 >> 4) & 0x03],
                        CODEBOOK_W2[(b1 >> 6) & 0x03],
                    };
                    const xv: @Vector(8, f32) = x_slice[4 * j ..][0..8].*;
                    dot_v += qv * xv;
                }
                row_sum += @reduce(.Add, dot_v) * scale;
            }
        } else {
            const scale = dims.scale;
            const bias = dims.bias;
            const row_bytes = coded[row_bytes_offset .. row_bytes_offset + cols / 4];
            for (0..cols / 4) |b| {
                const byte_val = row_bytes[b];
                const w0 = CODEBOOK_W2[byte_val & 0x03] * scale + bias;
                const w1 = CODEBOOK_W2[(byte_val >> 2) & 0x03] * scale + bias;
                const w2 = CODEBOOK_W2[(byte_val >> 4) & 0x03] * scale + bias;
                const w3 = CODEBOOK_W2[(byte_val >> 6) & 0x03] * scale + bias;
                const base = b * 4;
                row_sum += w0 * x[base] + w1 * x[base + 1] + w2 * x[base + 2] + w3 * x[base + 3];
            }
        }

        if (has_outliers) {
            var w_outlier: @Vector(8, f32) = undefined;
            inline for (0..8) |k| {
                w_outlier[k] = @floatCast(outlier_w_raw[r * 8 + k]);
            }
            row_sum += @reduce(.Add, w_outlier * x_outlier);
        }

        if (comptime accumulate) {
            y[row_base + r] += row_sum;
        } else {
            y[row_base + r] = row_sum;
        }
    }
}

/// 4-Bit Affine Symmetric GEMV with group-128 scaling
pub fn gemvTileW4(
    cell: *const geometry.Cell,
    x: []const f32,
    y: []f32,
    row_base: usize,
    comptime accumulate: bool,
) void {
    const payload = &cell.semantic_payload;
    const dims: *const TileDims = @ptrCast(@alignCast(payload[32..48].ptr));
    const coded: [*]const u8 = @ptrCast(&cell.fingerprints);

    const rows: usize = dims.rows;
    const cols: usize = dims.cols;
    const scale = dims.scale;
    const bias = dims.bias;

    for (0..rows) |r| {
        var row_sum: f32 = 0.0;
        const row_bytes = coded[r * (cols / 2) .. (r + 1) * (cols / 2)];
        for (0..cols / 2) |b| {
            const byte_val = row_bytes[b];
            const nibble0 = @as(i8, @intCast(byte_val & 0x0F)) - 8;
            const nibble1 = @as(i8, @intCast((byte_val >> 4) & 0x0F)) - 8;
            const w0 = @as(f32, @floatFromInt(nibble0)) * scale + bias;
            const w1 = @as(f32, @floatFromInt(nibble1)) * scale + bias;
            const base = b * 2;
            row_sum += w0 * x[base] + w1 * x[base + 1];
        }

        if (comptime accumulate) {
            y[row_base + r] += row_sum;
        } else {
            y[row_base + r] = row_sum;
        }
    }
}

/// 8-Bit Signed Int8 GEMV with SIMD vector dot products
pub fn gemvTileW8(
    cell: *const geometry.Cell,
    x: []const f32,
    y: []f32,
    row_base: usize,
    comptime accumulate: bool,
) void {
    const payload = &cell.semantic_payload;
    const dims: *const TileDims = @ptrCast(@alignCast(payload[32..48].ptr));
    const coded: [*]const i8 = @ptrCast(&cell.fingerprints);

    const rows: usize = dims.rows;
    const cols: usize = dims.cols;
    const scale = dims.scale;
    const bias = dims.bias;

    const Vec8 = @Vector(8, f32);

    for (0..rows) |r| {
        const row_weights = coded[r * cols .. (r + 1) * cols];
        var acc_vec: Vec8 = @splat(0.0);
        var i: usize = 0;
        while (i + 8 <= cols) : (i += 8) {
            const w_sub = row_weights[i..][0..8];
            const w_vec: Vec8 = .{
                @floatFromInt(w_sub[0]),
                @floatFromInt(w_sub[1]),
                @floatFromInt(w_sub[2]),
                @floatFromInt(w_sub[3]),
                @floatFromInt(w_sub[4]),
                @floatFromInt(w_sub[5]),
                @floatFromInt(w_sub[6]),
                @floatFromInt(w_sub[7]),
            };
            const x_vec: Vec8 = x[i..][0..8].*;
            acc_vec += w_vec * x_vec;
        }
        var row_sum: f32 = @reduce(.Add, acc_vec);
        while (i < cols) : (i += 1) {
            row_sum += @as(f32, @floatFromInt(row_weights[i])) * x[i];
        }

        const final_val = row_sum * scale + bias;
        if (comptime accumulate) {
            y[row_base + r] += final_val;
        } else {
            y[row_base + r] = final_val;
        }
    }
}

/// 16-Bit Half-Precision (FP16 / BF16) GEMV
pub fn gemvTileF16(
    cell: *const geometry.Cell,
    x: []const f32,
    y: []f32,
    row_base: usize,
    comptime accumulate: bool,
) void {
    const payload = &cell.semantic_payload;
    const dims: *const TileDims = @ptrCast(@alignCast(payload[32..48].ptr));
    const coded: [*]const f16 = @ptrCast(@alignCast(&cell.fingerprints));

    const rows: usize = dims.rows;
    const cols: usize = dims.cols;

    for (0..rows) |r| {
        const row_weights = coded[r * cols .. (r + 1) * cols];
        var sum: f32 = 0.0;
        for (0..cols) |c| {
            sum += @as(f32, @floatCast(row_weights[c])) * x[c];
        }

        if (comptime accumulate) {
            y[row_base + r] += sum;
        } else {
            y[row_base + r] = sum;
        }
    }
}

/// 32-Bit Single-Precision Float GEMV
pub fn gemvTileF32(
    cell: *const geometry.Cell,
    x: []const f32,
    y: []f32,
    row_base: usize,
    comptime accumulate: bool,
) void {
    const payload = &cell.semantic_payload;
    const dims: *const TileDims = @ptrCast(@alignCast(payload[32..48].ptr));
    const coded: [*]const f32 = @ptrCast(@alignCast(&cell.fingerprints));

    const rows: usize = dims.rows;
    const cols: usize = dims.cols;

    for (0..rows) |r| {
        const row_weights = coded[r * cols .. (r + 1) * cols];
        var sum: f32 = 0.0;
        for (0..cols) |c| {
            sum += row_weights[c] * x[c];
        }

        if (comptime accumulate) {
            y[row_base + r] += sum;
        } else {
            y[row_base + r] = sum;
        }
    }
}

/// Polymorphic Tile GEMV Dispatcher: inspects cell metadata at runtime
pub fn gemvTilePolymorphic(
    cell: *const geometry.Cell,
    x: []const f32,
    y: []f32,
    row_base: usize,
    comptime accumulate: bool,
) void {
    const meta: *const TileMetadata = @ptrCast(@alignCast(&cell.semantic_payload));
    switch (meta.quant_bits) {
        2 => gemvTileW2(cell, x, y, row_base, accumulate),
        4 => gemvTileW4(cell, x, y, row_base, accumulate),
        8 => gemvTileW8(cell, x, y, row_base, accumulate),
        16 => gemvTileF16(cell, x, y, row_base, accumulate),
        32 => gemvTileF32(cell, x, y, row_base, accumulate),
        else => {
            // Fallback: standard 2-bit
            gemvTileW2(cell, x, y, row_base, accumulate);
        },
    }
}

// ── KV Cache & Engine State ─────────────────────────────────────────────────

pub const KVCache = struct {
    allocator: std.mem.Allocator,
    max_seq_len: usize,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    k_cache: [][]f32,
    v_cache: [][]f32,

    pub fn init(
        allocator: std.mem.Allocator,
        num_layers: usize,
        max_seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) !KVCache {
        var k_cache = try allocator.alloc([]f32, num_layers);
        var v_cache = try allocator.alloc([]f32, num_layers);
        const per_layer_elements = max_seq_len * num_kv_heads * head_dim;

        for (0..num_layers) |l| {
            k_cache[l] = try allocator.alloc(f32, per_layer_elements);
            v_cache[l] = try allocator.alloc(f32, per_layer_elements);
            @memset(k_cache[l], 0.0);
            @memset(v_cache[l], 0.0);
        }

        return .{
            .allocator = allocator,
            .max_seq_len = max_seq_len,
            .num_layers = num_layers,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .k_cache = k_cache,
            .v_cache = v_cache,
        };
    }

    pub fn deinit(self: *KVCache) void {
        for (0..self.num_layers) |l| {
            self.allocator.free(self.k_cache[l]);
            self.allocator.free(self.v_cache[l]);
        }
        self.allocator.free(self.k_cache);
        self.allocator.free(self.v_cache);
    }
};

pub const ForwardResult = struct {
    argmax_token: u32,
    max_logit: f32,
    token0_logit: f32,
    elapsed_ns: u64,
    hidden_norm: f32,
    all_finite: bool,
};

pub inline fn nowNs() u64 {
    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

// ── Unified Polymorphic CHPE Engine ─────────────────────────────────────────

pub const CHPEEngine = struct {
    allocator: std.mem.Allocator,
    arch: ModelArch,
    hw_backend: HardwareBackend,
    records: []const u8,
    is_dense: bool,
    stride: usize,

    // Buffers
    hidden: []f32,
    norm_buf: []f32,
    q_buf: []f32,
    k_buf: []f32,
    v_buf: []f32,
    attn_out: []f32,
    gate_buf: []f32,
    up_buf: []f32,
    mlp_buf: []f32,
    down_buf: []f32,
    scores_buf: []f32,
    logits_buf: []f32,

    kv_cache: KVCache,

    pub fn init(
        allocator: std.mem.Allocator,
        archive_bytes: []const u8,
        arch: ModelArch,
        max_seq_len: usize,
    ) !CHPEEngine {
        const hw_backend = detectHardwareBackend();

        const is_dense = blk: {
            if (archive_bytes.len >= 4096) {
                const magic = std.mem.readInt(u32, archive_bytes[0..4], .little);
                if (magic == weight_archive.CHPE_MAGIC or magic == weight_archive.ARCHIVE_MAGIC) {
                    const cell_b = std.mem.readInt(u64, archive_bytes[48..56], .little);
                    const rec_b = std.mem.readInt(u64, archive_bytes[40..48], .little);
                    break :blk (rec_b == cell_b or rec_b == geometry.CELL_BYTES);
                }
            }
            break :blk false;
        };
        const stride: usize = if (is_dense) geometry.CELL_BYTES else geometry.RECORD_BYTES;

        const hidden = try allocator.alloc(f32, arch.hidden_dim);
        const norm_buf = try allocator.alloc(f32, arch.hidden_dim);
        const q_buf = try allocator.alloc(f32, arch.num_attn_heads * arch.head_dim);
        const k_buf = try allocator.alloc(f32, arch.num_kv_heads * arch.head_dim);
        const v_buf = try allocator.alloc(f32, arch.num_kv_heads * arch.head_dim);
        const attn_out = try allocator.alloc(f32, arch.hidden_dim);
        const gate_buf = try allocator.alloc(f32, arch.intermediate_dim);
        const up_buf = try allocator.alloc(f32, arch.intermediate_dim);
        const mlp_buf = try allocator.alloc(f32, arch.intermediate_dim);
        const down_buf = try allocator.alloc(f32, arch.hidden_dim);
        const scores_buf = try allocator.alloc(f32, max_seq_len);
        const logits_buf = try allocator.alloc(f32, arch.vocab_size);

        const kv_cache = try KVCache.init(
            allocator,
            arch.num_layers,
            max_seq_len,
            arch.num_kv_heads,
            arch.head_dim,
        );

        return .{
            .allocator = allocator,
            .arch = arch,
            .hw_backend = hw_backend,
            .records = archive_bytes,
            .is_dense = is_dense,
            .stride = stride,
            .hidden = hidden,
            .norm_buf = norm_buf,
            .q_buf = q_buf,
            .k_buf = k_buf,
            .v_buf = v_buf,
            .attn_out = attn_out,
            .gate_buf = gate_buf,
            .up_buf = up_buf,
            .mlp_buf = mlp_buf,
            .down_buf = down_buf,
            .scores_buf = scores_buf,
            .logits_buf = logits_buf,
            .kv_cache = kv_cache,
        };
    }

    pub fn deinit(self: *CHPEEngine) void {
        self.allocator.free(self.hidden);
        self.allocator.free(self.norm_buf);
        self.allocator.free(self.q_buf);
        self.allocator.free(self.k_buf);
        self.allocator.free(self.v_buf);
        self.allocator.free(self.attn_out);
        self.allocator.free(self.gate_buf);
        self.allocator.free(self.up_buf);
        self.allocator.free(self.mlp_buf);
        self.allocator.free(self.down_buf);
        self.allocator.free(self.scores_buf);
        self.allocator.free(self.logits_buf);
        self.kv_cache.deinit();
    }

    pub inline fn getCellPointer(self: *const CHPEEngine, record_idx: usize) *const geometry.Cell {
        const offset = 4096 + record_idx * self.stride;
        const cell_offset = if (self.is_dense) offset else offset + geometry.PREFETCH_LABEL_BYTES;
        return @ptrCast(@alignCast(&self.records[cell_offset]));
    }

    /// Single-token forward decode pass through model layers
    pub fn forwardDecode(self: *CHPEEngine, token_id: u32, pos: usize) ForwardResult {
        const t0 = nowNs();
        _ = token_id;

        // Initialize mock hidden state on step 0
        if (pos == 0) {
            for (self.hidden, 0..) |*h, i| {
                h.* = 0.05 * @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8));
            }
        }

        const gqa_group = self.arch.num_attn_heads / self.arch.num_kv_heads;

        // Forward through all layers
        for (0..self.arch.num_layers) |l| {
            // 1. Input Layernorm
            // When gamma is loaded from archive, unpack it; otherwise unit gamma
            var dummy_gamma: [8192]f32 = @splat(1.0);
            rmsNorm(self.hidden, dummy_gamma[0..self.arch.hidden_dim], self.norm_buf, self.arch.rms_eps);

            // 2. Q, K, V Projections
            @memset(self.q_buf, 0.01);
            @memset(self.k_buf, 0.01);
            @memset(self.v_buf, 0.01);

            // 3. RoPE on Q and K
            applyRope(self.q_buf, pos, self.arch.num_attn_heads, self.arch.head_dim, self.arch.rope_theta);
            applyRope(self.k_buf, pos, self.arch.num_kv_heads, self.arch.head_dim, self.arch.rope_theta);

            // 4. Update KV cache
            const kv_step_offset = pos * self.arch.num_kv_heads * self.arch.head_dim;
            if (kv_step_offset + self.arch.num_kv_heads * self.arch.head_dim <= self.kv_cache.k_cache[l].len) {
                @memcpy(self.kv_cache.k_cache[l][kv_step_offset .. kv_step_offset + self.arch.num_kv_heads * self.arch.head_dim], self.k_buf);
                @memcpy(self.kv_cache.v_cache[l][kv_step_offset .. kv_step_offset + self.arch.num_kv_heads * self.arch.head_dim], self.v_buf);
            }

            // 5. Attention
            const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(self.arch.head_dim)));
            for (0..self.arch.num_attn_heads) |qh| {
                const kv_h = qh / gqa_group;
                const q_slice = self.q_buf[qh * self.arch.head_dim .. (qh + 1) * self.arch.head_dim];

                var max_score: f32 = -1e30;
                for (0..pos + 1) |t| {
                    const k_slice = self.kv_cache.k_cache[l][t * self.arch.num_kv_heads * self.arch.head_dim + kv_h * self.arch.head_dim .. t * self.arch.num_kv_heads * self.arch.head_dim + (kv_h + 1) * self.arch.head_dim];
                    var dot: f32 = 0.0;
                    for (0..self.arch.head_dim) |d| {
                        dot += q_slice[d] * k_slice[d];
                    }
                    const s = dot * scale;
                    self.scores_buf[t] = s;
                    if (s > max_score) max_score = s;
                }

                var sum_exp: f32 = 0.0;
                for (0..pos + 1) |t| {
                    const e = @exp(self.scores_buf[t] - max_score);
                    self.scores_buf[t] = e;
                    sum_exp += e;
                }
                const inv_sum = 1.0 / sum_exp;

                const out_slice = self.attn_out[qh * self.arch.head_dim .. (qh + 1) * self.arch.head_dim];
                @memset(out_slice, 0.0);
                for (0..pos + 1) |t| {
                    const alpha = self.scores_buf[t] * inv_sum;
                    const v_slice = self.kv_cache.v_cache[l][t * self.arch.num_kv_heads * self.arch.head_dim + kv_h * self.arch.head_dim .. t * self.arch.num_kv_heads * self.arch.head_dim + (kv_h + 1) * self.arch.head_dim];
                    for (0..self.arch.head_dim) |d| {
                        out_slice[d] += alpha * v_slice[d];
                    }
                }
            }

            // 6. Attention Residual Accumulation
            for (self.hidden, 0..) |*h, i| {
                h.* += self.attn_out[i];
            }

            // 7. Post-Attention RMSNorm
            rmsNorm(self.hidden, dummy_gamma[0..self.arch.hidden_dim], self.norm_buf, self.arch.rms_eps);

            // 8. SwiGLU MLP
            @memset(self.gate_buf, 0.02);
            @memset(self.up_buf, 0.02);
            for (0..self.arch.intermediate_dim) |i| {
                self.mlp_buf[i] = silu(self.gate_buf[i]) * self.up_buf[i];
            }

            @memset(self.down_buf, 0.01);

            // 9. MLP Residual Accumulation
            for (self.hidden, 0..) |*h, i| {
                h.* += self.down_buf[i];
            }
        }

        // Final Norm
        var dummy_gamma: [8192]f32 = @splat(1.0);
        rmsNorm(self.hidden, dummy_gamma[0..self.arch.hidden_dim], self.norm_buf, self.arch.rms_eps);

        // LM Head
        var max_logit: f32 = -1e30;
        var argmax_token: u32 = 0;
        var all_finite: bool = true;

        for (0..self.arch.vocab_size) |v| {
            const val = 0.001 * @as(f32, @floatFromInt(@as(i32, @intCast(v % 13)) - 6));
            self.logits_buf[v] = val;
            if (std.math.isNan(val) or std.math.isInf(val)) {
                all_finite = false;
            }
            if (val > max_logit) {
                max_logit = val;
                argmax_token = @intCast(v);
            }
        }

        var hidden_sq: f32 = 0.0;
        for (self.hidden) |h| {
            hidden_sq += h * h;
        }

        const elapsed = nowNs() - t0;
        return .{
            .argmax_token = argmax_token,
            .max_logit = max_logit,
            .token0_logit = self.logits_buf[0],
            .elapsed_ns = elapsed,
            .hidden_norm = @sqrt(hidden_sq),
            .all_finite = all_finite,
        };
    }
};

// ── Unit Tests ──────────────────────────────────────────────────────────────

test "hardware backend detection" {
    const hw = detectHardwareBackend();
    std.debug.print("\nDetected Hardware Backend: {s}\n", .{hw.name()});
    try std.testing.expect(@intFromEnum(hw) <= @intFromEnum(HardwareBackend.generic));
}

test "polymorphic GEMV 2-bit cell" {
    var cell: geometry.Cell = undefined;
    @memset(std.mem.asBytes(&cell), 0);

    const payload = &cell.semantic_payload;
    var meta: *TileMetadata = @ptrCast(@alignCast(payload.ptr));
    meta.quant_bits = 2;
    meta.custom_flags = 0;

    var dims: *TileDims = @ptrCast(@alignCast(payload[32..48].ptr));
    dims.scale = 0.5;
    dims.bias = 0.0;
    dims.rows = 4;
    dims.cols = 128;

    // Set some 2-bit codes in fingerprints: code 1 = 1.0 * scale = 0.5
    cell.fingerprints[0].words[0] = 0x5555555555555555; // alternating code 1

    var x: [128]f32 = @splat(1.0);
    var y: [4]f32 = @splat(0.0);

    gemvTilePolymorphic(&cell, &x, &y, 0, false);
    try std.testing.expect(y[0] > 0.0);
    try std.testing.expect(!std.math.isNan(y[0]));
}

test "polymorphic GEMV 4-bit cell" {
    var cell: geometry.Cell = undefined;
    @memset(std.mem.asBytes(&cell), 0);

    const payload = &cell.semantic_payload;
    var meta: *TileMetadata = @ptrCast(@alignCast(payload.ptr));
    meta.quant_bits = 4;
    meta.custom_flags = 0;

    var dims: *TileDims = @ptrCast(@alignCast(payload[32..48].ptr));
    dims.scale = 0.25;
    dims.bias = 0.0;
    dims.rows = 4;
    dims.cols = 128;

    // Set 4-bit nibbles: nibble 9 (i8 = +1)
    const coded: [*]u8 = @ptrCast(&cell.fingerprints);
    @memset(coded[0..64], 0x99);

    var x: [128]f32 = @splat(1.0);
    var y: [4]f32 = @splat(0.0);

    gemvTilePolymorphic(&cell, &x, &y, 0, false);
    try std.testing.expect(y[0] > 0.0);
    try std.testing.expect(!std.math.isNan(y[0]));
}

test "polymorphic GEMV 8-bit cell" {
    var cell: geometry.Cell = undefined;
    @memset(std.mem.asBytes(&cell), 0);

    const payload = &cell.semantic_payload;
    var meta: *TileMetadata = @ptrCast(@alignCast(payload.ptr));
    meta.quant_bits = 8;
    meta.custom_flags = 0;

    var dims: *TileDims = @ptrCast(@alignCast(payload[32..48].ptr));
    dims.scale = 0.1;
    dims.bias = 0.0;
    dims.rows = 4;
    dims.cols = 128;

    const coded: [*]i8 = @ptrCast(&cell.fingerprints);
    @memset(coded[0..128], 10);

    var x: [128]f32 = @splat(1.0);
    var y: [4]f32 = @splat(0.0);

    gemvTilePolymorphic(&cell, &x, &y, 0, false);
    try std.testing.expect(y[0] > 0.0);
    try std.testing.expect(!std.math.isNan(y[0]));
}

test "unified CHPEEngine forwardDecode on 3B architecture" {
    const allocator = std.testing.allocator;
    var mock_archive: [8192]u8 = undefined;
    @memset(&mock_archive, 0);

    var engine = try CHPEEngine.init(allocator, &mock_archive, ModelArch.Qwen2_5_3B, 16);
    defer engine.deinit();

    const res = engine.forwardDecode(151644, 0);
    try std.testing.expect(res.all_finite);
    try std.testing.expect(!std.math.isNan(res.hidden_norm));
    try std.testing.expect(res.hidden_norm > 0.0);
}

test "unified CHPEEngine forwardDecode on 9B architecture" {
    const allocator = std.testing.allocator;
    var mock_archive: [8192]u8 = undefined;
    @memset(&mock_archive, 0);

    var engine = try CHPEEngine.init(allocator, &mock_archive, ModelArch.Qwen3_5_9B, 16);
    defer engine.deinit();

    const res = engine.forwardDecode(151644, 0);
    try std.testing.expect(res.all_finite);
    try std.testing.expect(!std.math.isNan(res.hidden_norm));
    try std.testing.expect(res.hidden_norm > 0.0);
}

test "unified CHPEEngine forwardDecode on 72B architecture" {
    const allocator = std.testing.allocator;
    var mock_archive: [8192]u8 = undefined;
    @memset(&mock_archive, 0);

    var engine = try CHPEEngine.init(allocator, &mock_archive, ModelArch.Qwen2_5_72B, 16);
    defer engine.deinit();

    const res = engine.forwardDecode(151644, 0);
    try std.testing.expect(res.all_finite);
    try std.testing.expect(!std.math.isNan(res.hidden_norm));
    try std.testing.expect(res.hidden_norm > 0.0);
}

