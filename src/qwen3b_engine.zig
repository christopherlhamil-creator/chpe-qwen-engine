//! Qwen2.5-3B Native Forward Decode Engine
//!
//! Subsystem: tot_hybrid/src/qwen3b_engine.zig
//! Blueprint: docs/BLUEPRINT-20260914-QWEN3B-CHPE-PIPELINE.md

const std = @import("std");
const geometry = @import("geometry.zig");
const weight_archive = @import("weight_archive.zig");
pub const qwen_geom = @import("qwen3b_geometry.zig");
const tensor_map = @import("qwen3b_tensor_map.zig");
const rmsnorm = @import("qwen_rmsnorm.zig");
const rope = @import("qwen_rope.zig");
const attn = @import("qwen_attention.zig");
const mlp = @import("qwen_mlp.zig");
const hw = @import("hardware_config.zig");

pub const HIDDEN_DIM: usize = qwen_geom.HIDDEN_DIM; // 2048
pub const INTERMEDIATE_DIM: usize = qwen_geom.INTERMEDIATE_DIM; // 11008
pub const NUM_LAYERS: usize = qwen_geom.NUM_LAYERS; // 36
pub const VOCAB_SIZE: usize = qwen_geom.VOCAB_SIZE; // 151936
pub const KV_DIM: usize = qwen_geom.KV_DIM; // 256

pub const USE_INT8_SDOT = hw.USE_INT8_SDOT;

pub const worker_tile_ranges = [4][2]usize{
    .{ hw.WORKER_0_TILE_START, hw.WORKER_0_TILE_END },
    .{ hw.WORKER_1_TILE_START, hw.WORKER_1_TILE_END },
    .{ hw.WORKER_2_TILE_START, hw.WORKER_2_TILE_END },
    .{ hw.WORKER_3_TILE_START, hw.WORKER_3_TILE_END },
};

pub const worker_group_ranges = [4][2]usize{
    .{ hw.WORKER_0_TILE_START / hw.TILES_PER_GROUP, hw.WORKER_0_TILE_END / hw.TILES_PER_GROUP },
    .{ hw.WORKER_1_TILE_START / hw.TILES_PER_GROUP, hw.WORKER_1_TILE_END / hw.TILES_PER_GROUP },
    .{ hw.WORKER_2_TILE_START / hw.TILES_PER_GROUP, hw.WORKER_2_TILE_END / hw.TILES_PER_GROUP },
    .{ hw.WORKER_3_TILE_START / hw.TILES_PER_GROUP, hw.WORKER_3_TILE_END / hw.TILES_PER_GROUP },
};

pub var prof_qkv_ns: u64 = 0;
pub var prof_attn_ns: u64 = 0;
pub var prof_oproj_ns: u64 = 0;
pub var prof_norm_ns: u64 = 0;
pub var prof_gateup_ns: u64 = 0;
pub var prof_down_ns: u64 = 0;
pub var prof_head_ns: u64 = 0;

pub fn nowNs() u64 {
    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

pub const ForwardResult = struct {
    argmax_token: u32,
    max_logit: f32,
    token0_logit: f32,
    elapsed_ns: u64,
    hidden_norm: f32,
    all_finite: bool,
    logits: []const f32,
};

pub inline fn prefetchCell(cell: *const geometry.Cell) void {
    const payload: [*]const u8 = @ptrCast(&cell.semantic_payload);
    inline for (.{ 0, 64, 128, 192 }) |off| {
        @prefetch(payload + off, .{ .locality = 3, .cache = .data, .rw = .read });
    }
    const fp: [*]const u8 = @ptrCast(&cell.fingerprints);
    inline for (.{ 0, 64, 128, 192 }) |off| {
        @prefetch(fp + off, .{ .locality = 3, .cache = .data, .rw = .read });
    }
}

/// Prefetches the metadata/scales in semantic_payload and initial weights in fingerprints of an upcoming record.
pub inline fn prefetchRecord(rec: *const geometry.Record) void {
    prefetchCell(&rec.cell);
}

/// Prefetches the initial cache lines of an upcoming raw contiguous 16,384B tile.
pub inline fn prefetchTile(tile_ptr: *const anyopaque) void {
    const p: [*]const u8 = @ptrCast(tile_ptr);
    inline for (.{ 0, 64, 128, 192, 256, 320, 384, 448 }) |off| {
        @prefetch(p + off, .{ .locality = 3, .cache = .data, .rw = .read });
    }
}

/// Unpacks a 1D tensor (e.g. bias, norm gamma) from a cell into an f32 slice.
pub fn unpack1DCell(cell: *const geometry.Cell, out: []f32) void {
    const payload = &cell.semantic_payload;
    const meta: *const weight_archive.TileMetadata = @ptrCast(@alignCast(payload.ptr));
    if (meta.quant_bits == 32) {
        const raw_f32: [*]const f32 = @ptrCast(@alignCast(&cell.fingerprints));
        @memcpy(out, raw_f32[0..out.len]);
        return;
    }

    const scale: f32 = @bitCast(std.mem.readInt(u32, payload[32..36], .little));
    const bias: f32 = @bitCast(std.mem.readInt(u32, payload[36..40], .little));

    const coded: [*]const u8 = @ptrCast(&cell.fingerprints);
    const num_bytes = (out.len + 1) / 2;
    for (0..num_bytes) |j| {
        const b = coded[j];
        const q0: f32 = @floatFromInt(b & 0x0F);
        out[2 * j] = (q0 - 8.0) * scale + bias;
        if (2 * j + 1 < out.len) {
            const q1: f32 = @floatFromInt((b >> 4) & 0x0F);
            out[2 * j + 1] = (q1 - 8.0) * scale + bias;
        }
    }
}


pub inline fn unpack1D(rec: *const geometry.Record, out: []f32) void {
    unpack1DCell(&rec.cell, out);
}

/// Unpacks a specific 2048-weight token embedding row from an embed_tokens cell tile.
pub fn unpackEmbedRowCell(cell: *const geometry.Cell, row_in_tile: usize, out: []f32) void {
    std.debug.assert(out.len == HIDDEN_DIM);

    const payload = &cell.semantic_payload;
    const meta: *const weight_archive.TileMetadata = @ptrCast(@alignCast(payload.ptr));
    if (meta.quant_bits == 8) {
        std.debug.assert(row_in_tile < 8);
        const group_scales_raw: [*]const f16 = @ptrCast(@alignCast(payload[48..304].ptr));
        const coded: [*]const u8 = @ptrCast(&cell.fingerprints);
        const row_bytes = coded[row_in_tile * 2048 .. (row_in_tile + 1) * 2048];

        for (0..16) |g| {
            const scale: f32 = @floatCast(group_scales_raw[row_in_tile * 16 + g]);
            const g_bytes = row_bytes[g * 128 .. (g + 1) * 128];
            const out_g = out[g * 128 .. (g + 1) * 128];

            for (0..128) |j| {
                const w_i8: i8 = @bitCast(g_bytes[j]);
                out_g[j] = @as(f32, @floatFromInt(w_i8)) * scale;
            }
        }
        return;
    }
    if (meta.quant_bits == 2) {
        std.debug.assert(row_in_tile < 32);
        const group_scales_raw: [*]const f16 = @ptrCast(@alignCast(payload[48..560].ptr));
        const coded: [*]const u8 = @ptrCast(&cell.fingerprints);
        const row_bytes = coded[row_in_tile * 512 .. (row_in_tile + 1) * 512];
        const LUT: [4]f32 = .{ 0.0, 1.0, -2.0, -1.0 };

        for (0..16) |g| {
            const scale: f32 = @floatCast(group_scales_raw[row_in_tile * 16 + g]);
            const g_bytes = row_bytes[g * 32 .. (g + 1) * 32];
            const out_g = out[g * 128 .. (g + 1) * 128];

            for (0..32) |j| {
                const b = g_bytes[j];
                out_g[4 * j + 0] = LUT[b & 0x03] * scale;
                out_g[4 * j + 1] = LUT[(b >> 2) & 0x03] * scale;
                out_g[4 * j + 2] = LUT[(b >> 4) & 0x03] * scale;
                out_g[4 * j + 3] = LUT[(b >> 6) & 0x03] * scale;
            }
        }
        if ((meta.custom_flags & weight_archive.FLAG_GROUP128_OUTLIERS) != 0) {
            const outlier_w_raw: [*]const f16 = @ptrCast(@alignCast(payload[560..816].ptr));
            const outlier_cols_raw: [*]const u16 = @ptrCast(@alignCast(payload[816..832].ptr));
            inline for (0..8) |k| {
                const col_idx = outlier_cols_raw[k];
                out[col_idx] = @floatCast(outlier_w_raw[row_in_tile * 8 + k]);
            }
        }
        return;
    }
    if ((meta.custom_flags & weight_archive.FLAG_GROUP128_OUTLIERS) != 0 or meta.quant_bits == 4) {
        std.debug.assert(row_in_tile < 16);
        const group_scales_raw: [*]const f16 = @ptrCast(@alignCast(payload[48..560].ptr));
        const coded: [*]const u8 = @ptrCast(&cell.fingerprints);

        const row_bytes = coded[row_in_tile * 1024 .. (row_in_tile + 1) * 1024];

        for (0..16) |g| {
            const scale: f32 = @floatCast(group_scales_raw[row_in_tile * 16 + g]);
            const g_bytes = row_bytes[g * 64 .. (g + 1) * 64];
            const out_g = out[g * 128 .. (g + 1) * 128];

            for (0..64) |j| {
                const b = g_bytes[j];
                const q0: f32 = @floatFromInt(b & 0x0F);
                const q1: f32 = @floatFromInt((b >> 4) & 0x0F);
                out_g[2 * j] = (q0 - 8.0) * scale;
                out_g[2 * j + 1] = (q1 - 8.0) * scale;
            }
        }

        if ((meta.custom_flags & weight_archive.FLAG_GROUP128_OUTLIERS) != 0) {
            const outlier_w_raw: [*]const f16 = @ptrCast(@alignCast(payload[560..816].ptr));
            const outlier_cols_raw: [*]const u16 = @ptrCast(@alignCast(payload[816..832].ptr));
            inline for (0..8) |k| {
                const col_idx = outlier_cols_raw[k];
                out[col_idx] = @floatCast(outlier_w_raw[row_in_tile * 8 + k]);
            }
        }
        return;
    }

    const scale: f32 = @bitCast(std.mem.readInt(u32, payload[32..36], .little));
    const bias: f32 = @bitCast(std.mem.readInt(u32, payload[36..40], .little));

    const coded: [*]const u8 = @ptrCast(&cell.fingerprints);
    const byte_offset = row_in_tile * 1024;
    for (0..1024) |j| {
        const b = coded[byte_offset + j];
        const q0: f32 = @floatFromInt(b & 0x0F);
        const q1: f32 = @floatFromInt((b >> 4) & 0x0F);
        out[2 * j] = (q0 - 8.0) * scale + bias;
        out[2 * j + 1] = (q1 - 8.0) * scale + bias;
    }
}

pub inline fn unpackEmbedRow(rec: *const geometry.Record, row_in_tile: usize, out: []f32) void {
    unpackEmbedRowCell(&rec.cell, row_in_tile, out);
}

pub fn unpackEmbedRowBF16Direct(tile_u16: [*]const u16, row_in_tile: usize, out: []f32) void {
    std.debug.assert(row_in_tile < 4);
    std.debug.assert(out.len == HIDDEN_DIM);

    const row_u16 = tile_u16[row_in_tile * 2048 .. (row_in_tile + 1) * 2048];

    var i: usize = 0;
    while (i + 8 <= 2048) : (i += 8) {
        const u16_v: @Vector(8, u16) = row_u16[i..][0..8].*;
        const u32_v: @Vector(8, u32) = @as(@Vector(8, u32), u16_v) << @splat(16);
        out[i..][0..8].* = @bitCast(u32_v);
    }
}

pub fn unpackEmbedRowF16Direct(tile_f16: [*]const f16, row_in_tile: usize, out: []f32) void {
    std.debug.assert(row_in_tile < 4);
    std.debug.assert(out.len == HIDDEN_DIM);

    const row_f16 = tile_f16[row_in_tile * 2048 .. (row_in_tile + 1) * 2048];

    var i: usize = 0;
    while (i + 8 <= 2048) : (i += 8) {
        const v: @Vector(8, f16) = row_f16[i..][0..8].*;
        const f32_v: @Vector(8, f32) = @floatCast(v);
        out[i..][0..8].* = @bitCast(f32_v);
    }
}

pub fn unpackEmbedRowBF16(rec: *const geometry.Record, row_in_tile: usize, out: []f32) void {
    const coded_u16: [*]const u16 = @ptrCast(@alignCast(&rec.cell.fingerprints));
    unpackEmbedRowBF16Direct(coded_u16, row_in_tile, out);
}

pub inline fn gemvRowBF16(row_u16: [*]const u16, x: []const f32) f32 {
    var d0: @Vector(4, f32) = @splat(0.0);
    var d1: @Vector(4, f32) = @splat(0.0);
    var d2: @Vector(4, f32) = @splat(0.0);
    var d3: @Vector(4, f32) = @splat(0.0);
    var d4: @Vector(4, f32) = @splat(0.0);
    var d5: @Vector(4, f32) = @splat(0.0);
    var d6: @Vector(4, f32) = @splat(0.0);
    var d7: @Vector(4, f32) = @splat(0.0);

    var c: usize = 0;
    while (c + 32 <= 2048) : (c += 32) {
        if (c + 64 < 2048) {
            @prefetch(&row_u16[c + 64], .{ .locality = 3, .cache = .data, .rw = .read });
        }
        const xa: @Vector(4, f32) = x[c..][0..4].*;
        const xb: @Vector(4, f32) = x[c + 4 ..][0..4].*;
        const xc: @Vector(4, f32) = x[c + 8 ..][0..4].*;
        const xd: @Vector(4, f32) = x[c + 12 ..][0..4].*;
        const xe: @Vector(4, f32) = x[c + 16 ..][0..4].*;
        const xf: @Vector(4, f32) = x[c + 20 ..][0..4].*;
        const xg: @Vector(4, f32) = x[c + 24 ..][0..4].*;
        const xh: @Vector(4, f32) = x[c + 28 ..][0..4].*;

        const wa: @Vector(4, u16) = row_u16[c..][0..4].*;
        const wb: @Vector(4, u16) = row_u16[c + 4 ..][0..4].*;
        const wc: @Vector(4, u16) = row_u16[c + 8 ..][0..4].*;
        const wd: @Vector(4, u16) = row_u16[c + 12 ..][0..4].*;
        const we: @Vector(4, u16) = row_u16[c + 16 ..][0..4].*;
        const wf: @Vector(4, u16) = row_u16[c + 20 ..][0..4].*;
        const wg: @Vector(4, u16) = row_u16[c + 24 ..][0..4].*;
        const wh: @Vector(4, u16) = row_u16[c + 28 ..][0..4].*;

        const fwa: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), wa) << @splat(16));
        const fwb: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), wb) << @splat(16));
        const fwc: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), wc) << @splat(16));
        const fwd: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), wd) << @splat(16));
        const fwe: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), we) << @splat(16));
        const fwf: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), wf) << @splat(16));
        const fwg: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), wg) << @splat(16));
        const fwh: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), wh) << @splat(16));

        d0 = @mulAdd(@Vector(4, f32), fwa, xa, d0);
        d1 = @mulAdd(@Vector(4, f32), fwb, xb, d1);
        d2 = @mulAdd(@Vector(4, f32), fwc, xc, d2);
        d3 = @mulAdd(@Vector(4, f32), fwd, xd, d3);
        d4 = @mulAdd(@Vector(4, f32), fwe, xe, d4);
        d5 = @mulAdd(@Vector(4, f32), fwf, xf, d5);
        d6 = @mulAdd(@Vector(4, f32), fwg, xg, d6);
        d7 = @mulAdd(@Vector(4, f32), fwh, xh, d7);
    }
    const sum4 = (d0 + d1) + (d2 + d3) + (d4 + d5) + (d6 + d7);
    return @reduce(.Add, sum4);
}

pub inline fn gemvQuadRowBF16Values(coded_u16: [*]const u16, x: []const f32) [4]f32 {
    const r0 = coded_u16 + 0 * 2048;
    const r1 = coded_u16 + 1 * 2048;
    const r2 = coded_u16 + 2 * 2048;
    const r3 = coded_u16 + 3 * 2048;

    var d0a: @Vector(4, f32) = @splat(0.0);
    var d0b: @Vector(4, f32) = @splat(0.0);
    var d0c: @Vector(4, f32) = @splat(0.0);
    var d0d: @Vector(4, f32) = @splat(0.0);

    var d1a: @Vector(4, f32) = @splat(0.0);
    var d1b: @Vector(4, f32) = @splat(0.0);
    var d1c: @Vector(4, f32) = @splat(0.0);
    var d1d: @Vector(4, f32) = @splat(0.0);

    var d2a: @Vector(4, f32) = @splat(0.0);
    var d2b: @Vector(4, f32) = @splat(0.0);
    var d2c: @Vector(4, f32) = @splat(0.0);
    var d2d: @Vector(4, f32) = @splat(0.0);

    var d3a: @Vector(4, f32) = @splat(0.0);
    var d3b: @Vector(4, f32) = @splat(0.0);
    var d3c: @Vector(4, f32) = @splat(0.0);
    var d3d: @Vector(4, f32) = @splat(0.0);

    var c: usize = 0;
    while (c < 2048) : (c += 32) {
        const xa0: @Vector(4, f32) = x[c + 0 ..][0..4].*;
        const xa1: @Vector(4, f32) = x[c + 4 ..][0..4].*;
        const xa2: @Vector(4, f32) = x[c + 8 ..][0..4].*;
        const xa3: @Vector(4, f32) = x[c + 12 ..][0..4].*;
        const xa4: @Vector(4, f32) = x[c + 16 ..][0..4].*;
        const xa5: @Vector(4, f32) = x[c + 20 ..][0..4].*;
        const xa6: @Vector(4, f32) = x[c + 24 ..][0..4].*;
        const xa7: @Vector(4, f32) = x[c + 28 ..][0..4].*;

        // Row 0
        const w0_0: @Vector(4, u16) = r0[c + 0 ..][0..4].*;
        const w0_1: @Vector(4, u16) = r0[c + 4 ..][0..4].*;
        const w0_2: @Vector(4, u16) = r0[c + 8 ..][0..4].*;
        const w0_3: @Vector(4, u16) = r0[c + 12 ..][0..4].*;
        const w0_4: @Vector(4, u16) = r0[c + 16 ..][0..4].*;
        const w0_5: @Vector(4, u16) = r0[c + 20 ..][0..4].*;
        const w0_6: @Vector(4, u16) = r0[c + 24 ..][0..4].*;
        const w0_7: @Vector(4, u16) = r0[c + 28 ..][0..4].*;

        const fw0_0: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0_0) << @splat(16));
        const fw0_1: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0_1) << @splat(16));
        const fw0_2: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0_2) << @splat(16));
        const fw0_3: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0_3) << @splat(16));
        const fw0_4: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0_4) << @splat(16));
        const fw0_5: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0_5) << @splat(16));
        const fw0_6: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0_6) << @splat(16));
        const fw0_7: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0_7) << @splat(16));

        d0a = @mulAdd(@Vector(4, f32), fw0_0, xa0, d0a);
        d0b = @mulAdd(@Vector(4, f32), fw0_1, xa1, d0b);
        d0c = @mulAdd(@Vector(4, f32), fw0_2, xa2, d0c);
        d0d = @mulAdd(@Vector(4, f32), fw0_3, xa3, d0d);
        d0a = @mulAdd(@Vector(4, f32), fw0_4, xa4, d0a);
        d0b = @mulAdd(@Vector(4, f32), fw0_5, xa5, d0b);
        d0c = @mulAdd(@Vector(4, f32), fw0_6, xa6, d0c);
        d0d = @mulAdd(@Vector(4, f32), fw0_7, xa7, d0d);

        // Row 1
        const w1_0: @Vector(4, u16) = r1[c + 0 ..][0..4].*;
        const w1_1: @Vector(4, u16) = r1[c + 4 ..][0..4].*;
        const w1_2: @Vector(4, u16) = r1[c + 8 ..][0..4].*;
        const w1_3: @Vector(4, u16) = r1[c + 12 ..][0..4].*;
        const w1_4: @Vector(4, u16) = r1[c + 16 ..][0..4].*;
        const w1_5: @Vector(4, u16) = r1[c + 20 ..][0..4].*;
        const w1_6: @Vector(4, u16) = r1[c + 24 ..][0..4].*;
        const w1_7: @Vector(4, u16) = r1[c + 28 ..][0..4].*;

        const fw1_0: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1_0) << @splat(16));
        const fw1_1: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1_1) << @splat(16));
        const fw1_2: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1_2) << @splat(16));
        const fw1_3: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1_3) << @splat(16));
        const fw1_4: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1_4) << @splat(16));
        const fw1_5: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1_5) << @splat(16));
        const fw1_6: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1_6) << @splat(16));
        const fw1_7: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1_7) << @splat(16));

        d1a = @mulAdd(@Vector(4, f32), fw1_0, xa0, d1a);
        d1b = @mulAdd(@Vector(4, f32), fw1_1, xa1, d1b);
        d1c = @mulAdd(@Vector(4, f32), fw1_2, xa2, d1c);
        d1d = @mulAdd(@Vector(4, f32), fw1_3, xa3, d1d);
        d1a = @mulAdd(@Vector(4, f32), fw1_4, xa4, d1a);
        d1b = @mulAdd(@Vector(4, f32), fw1_5, xa5, d1b);
        d1c = @mulAdd(@Vector(4, f32), fw1_6, xa6, d1c);
        d1d = @mulAdd(@Vector(4, f32), fw1_7, xa7, d1d);

        // Row 2
        const w2_0: @Vector(4, u16) = r2[c + 0 ..][0..4].*;
        const w2_1: @Vector(4, u16) = r2[c + 4 ..][0..4].*;
        const w2_2: @Vector(4, u16) = r2[c + 8 ..][0..4].*;
        const w2_3: @Vector(4, u16) = r2[c + 12 ..][0..4].*;
        const w2_4: @Vector(4, u16) = r2[c + 16 ..][0..4].*;
        const w2_5: @Vector(4, u16) = r2[c + 20 ..][0..4].*;
        const w2_6: @Vector(4, u16) = r2[c + 24 ..][0..4].*;
        const w2_7: @Vector(4, u16) = r2[c + 28 ..][0..4].*;

        const fw2_0: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w2_0) << @splat(16));
        const fw2_1: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w2_1) << @splat(16));
        const fw2_2: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w2_2) << @splat(16));
        const fw2_3: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w2_3) << @splat(16));
        const fw2_4: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w2_4) << @splat(16));
        const fw2_5: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w2_5) << @splat(16));
        const fw2_6: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w2_6) << @splat(16));
        const fw2_7: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w2_7) << @splat(16));

        d2a = @mulAdd(@Vector(4, f32), fw2_0, xa0, d2a);
        d2b = @mulAdd(@Vector(4, f32), fw2_1, xa1, d2b);
        d2c = @mulAdd(@Vector(4, f32), fw2_2, xa2, d2c);
        d2d = @mulAdd(@Vector(4, f32), fw2_3, xa3, d2d);
        d2a = @mulAdd(@Vector(4, f32), fw2_4, xa4, d2a);
        d2b = @mulAdd(@Vector(4, f32), fw2_5, xa5, d2b);
        d2c = @mulAdd(@Vector(4, f32), fw2_6, xa6, d2c);
        d2d = @mulAdd(@Vector(4, f32), fw2_7, xa7, d2d);

        // Row 3
        const w3_0: @Vector(4, u16) = r3[c + 0 ..][0..4].*;
        const w3_1: @Vector(4, u16) = r3[c + 4 ..][0..4].*;
        const w3_2: @Vector(4, u16) = r3[c + 8 ..][0..4].*;
        const w3_3: @Vector(4, u16) = r3[c + 12 ..][0..4].*;
        const w3_4: @Vector(4, u16) = r3[c + 16 ..][0..4].*;
        const w3_5: @Vector(4, u16) = r3[c + 20 ..][0..4].*;
        const w3_6: @Vector(4, u16) = r3[c + 24 ..][0..4].*;
        const w3_7: @Vector(4, u16) = r3[c + 28 ..][0..4].*;

        const fw3_0: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w3_0) << @splat(16));
        const fw3_1: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w3_1) << @splat(16));
        const fw3_2: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w3_2) << @splat(16));
        const fw3_3: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w3_3) << @splat(16));
        const fw3_4: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w3_4) << @splat(16));
        const fw3_5: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w3_5) << @splat(16));
        const fw3_6: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w3_6) << @splat(16));
        const fw3_7: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w3_7) << @splat(16));

        d3a = @mulAdd(@Vector(4, f32), fw3_0, xa0, d3a);
        d3b = @mulAdd(@Vector(4, f32), fw3_1, xa1, d3b);
        d3c = @mulAdd(@Vector(4, f32), fw3_2, xa2, d3c);
        d3d = @mulAdd(@Vector(4, f32), fw3_3, xa3, d3d);
        d3a = @mulAdd(@Vector(4, f32), fw3_4, xa4, d3a);
        d3b = @mulAdd(@Vector(4, f32), fw3_5, xa5, d3b);
        d3c = @mulAdd(@Vector(4, f32), fw3_6, xa6, d3c);
        d3d = @mulAdd(@Vector(4, f32), fw3_7, xa7, d3d);
    }

    return .{
        @reduce(.Add, (d0a + d0b) + (d0c + d0d)),
        @reduce(.Add, (d1a + d1b) + (d1c + d1d)),
        @reduce(.Add, (d2a + d2b) + (d2c + d2d)),
        @reduce(.Add, (d3a + d3b) + (d3c + d3d)),
    };
}

pub inline fn dotRowBF16(w: [*]const u16, x: [*]const f32) f32 {
    var d0: @Vector(4, f32) = @splat(0.0);
    var d1: @Vector(4, f32) = @splat(0.0);
    var d2: @Vector(4, f32) = @splat(0.0);
    var d3: @Vector(4, f32) = @splat(0.0);
    var d4: @Vector(4, f32) = @splat(0.0);
    var d5: @Vector(4, f32) = @splat(0.0);
    var d6: @Vector(4, f32) = @splat(0.0);
    var d7: @Vector(4, f32) = @splat(0.0);
    var d8: @Vector(4, f32) = @splat(0.0);
    var d9: @Vector(4, f32) = @splat(0.0);
    var d10: @Vector(4, f32) = @splat(0.0);
    var d11: @Vector(4, f32) = @splat(0.0);
    var d12: @Vector(4, f32) = @splat(0.0);
    var d13: @Vector(4, f32) = @splat(0.0);
    var d14: @Vector(4, f32) = @splat(0.0);
    var d15: @Vector(4, f32) = @splat(0.0);

    var c: usize = 0;
    while (c + 64 <= 2048) : (c += 64) {
        const w0: @Vector(4, u16) = w[c + 0 ..][0..4].*;
        const w1: @Vector(4, u16) = w[c + 4 ..][0..4].*;
        const w2: @Vector(4, u16) = w[c + 8 ..][0..4].*;
        const w3: @Vector(4, u16) = w[c + 12 ..][0..4].*;
        const w4: @Vector(4, u16) = w[c + 16 ..][0..4].*;
        const w5: @Vector(4, u16) = w[c + 20 ..][0..4].*;
        const w6: @Vector(4, u16) = w[c + 24 ..][0..4].*;
        const w7: @Vector(4, u16) = w[c + 28 ..][0..4].*;
        const w8: @Vector(4, u16) = w[c + 32 ..][0..4].*;
        const w9: @Vector(4, u16) = w[c + 36 ..][0..4].*;
        const w10: @Vector(4, u16) = w[c + 40 ..][0..4].*;
        const w11: @Vector(4, u16) = w[c + 44 ..][0..4].*;
        const w12: @Vector(4, u16) = w[c + 48 ..][0..4].*;
        const w13: @Vector(4, u16) = w[c + 52 ..][0..4].*;
        const w14: @Vector(4, u16) = w[c + 56 ..][0..4].*;
        const w15: @Vector(4, u16) = w[c + 60 ..][0..4].*;

        const fw0: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0) << @splat(16));
        const fw1: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1) << @splat(16));
        const fw2: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w2) << @splat(16));
        const fw3: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w3) << @splat(16));
        const fw4: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w4) << @splat(16));
        const fw5: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w5) << @splat(16));
        const fw6: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w6) << @splat(16));
        const fw7: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w7) << @splat(16));
        const fw8: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w8) << @splat(16));
        const fw9: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w9) << @splat(16));
        const fw10: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w10) << @splat(16));
        const fw11: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w11) << @splat(16));
        const fw12: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w12) << @splat(16));
        const fw13: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w13) << @splat(16));
        const fw14: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w14) << @splat(16));
        const fw15: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w15) << @splat(16));

        const x0: @Vector(4, f32) = x[c + 0 ..][0..4].*;
        const x1: @Vector(4, f32) = x[c + 4 ..][0..4].*;
        const x2: @Vector(4, f32) = x[c + 8 ..][0..4].*;
        const x3: @Vector(4, f32) = x[c + 12 ..][0..4].*;
        const x4: @Vector(4, f32) = x[c + 16 ..][0..4].*;
        const x5: @Vector(4, f32) = x[c + 20 ..][0..4].*;
        const x6: @Vector(4, f32) = x[c + 24 ..][0..4].*;
        const x7: @Vector(4, f32) = x[c + 28 ..][0..4].*;
        const x8: @Vector(4, f32) = x[c + 32 ..][0..4].*;
        const x9: @Vector(4, f32) = x[c + 36 ..][0..4].*;
        const x10: @Vector(4, f32) = x[c + 40 ..][0..4].*;
        const x11: @Vector(4, f32) = x[c + 44 ..][0..4].*;
        const x12: @Vector(4, f32) = x[c + 48 ..][0..4].*;
        const x13: @Vector(4, f32) = x[c + 52 ..][0..4].*;
        const x14: @Vector(4, f32) = x[c + 56 ..][0..4].*;
        const x15: @Vector(4, f32) = x[c + 60 ..][0..4].*;

        d0 = @mulAdd(@Vector(4, f32), fw0, x0, d0);
        d1 = @mulAdd(@Vector(4, f32), fw1, x1, d1);
        d2 = @mulAdd(@Vector(4, f32), fw2, x2, d2);
        d3 = @mulAdd(@Vector(4, f32), fw3, x3, d3);
        d4 = @mulAdd(@Vector(4, f32), fw4, x4, d4);
        d5 = @mulAdd(@Vector(4, f32), fw5, x5, d5);
        d6 = @mulAdd(@Vector(4, f32), fw6, x6, d6);
        d7 = @mulAdd(@Vector(4, f32), fw7, x7, d7);
        d8 = @mulAdd(@Vector(4, f32), fw8, x8, d8);
        d9 = @mulAdd(@Vector(4, f32), fw9, x9, d9);
        d10 = @mulAdd(@Vector(4, f32), fw10, x10, d10);
        d11 = @mulAdd(@Vector(4, f32), fw11, x11, d11);
        d12 = @mulAdd(@Vector(4, f32), fw12, x12, d12);
        d13 = @mulAdd(@Vector(4, f32), fw13, x13, d13);
        d14 = @mulAdd(@Vector(4, f32), fw14, x14, d14);
        d15 = @mulAdd(@Vector(4, f32), fw15, x15, d15);
    }
    const sum_0_3 = (d0 + d1) + (d2 + d3);
    const sum_4_7 = (d4 + d5) + (d6 + d7);
    const sum_8_11 = (d8 + d9) + (d10 + d11);
    const sum_12_15 = (d12 + d13) + (d14 + d15);
    return @reduce(.Add, (sum_0_3 + sum_4_7) + (sum_8_11 + sum_12_15));
}

pub inline fn dotRow11008BF16(w: [*]const u16, x: [*]const f32) f32 {
    var d0: @Vector(4, f32) = @splat(0.0);
    var d1: @Vector(4, f32) = @splat(0.0);
    var d2: @Vector(4, f32) = @splat(0.0);
    var d3: @Vector(4, f32) = @splat(0.0);
    var d4: @Vector(4, f32) = @splat(0.0);
    var d5: @Vector(4, f32) = @splat(0.0);
    var d6: @Vector(4, f32) = @splat(0.0);
    var d7: @Vector(4, f32) = @splat(0.0);
    var d8: @Vector(4, f32) = @splat(0.0);
    var d9: @Vector(4, f32) = @splat(0.0);
    var d10: @Vector(4, f32) = @splat(0.0);
    var d11: @Vector(4, f32) = @splat(0.0);
    var d12: @Vector(4, f32) = @splat(0.0);
    var d13: @Vector(4, f32) = @splat(0.0);
    var d14: @Vector(4, f32) = @splat(0.0);
    var d15: @Vector(4, f32) = @splat(0.0);

    var c: usize = 0;
    while (c < 11008) : (c += 64) {
        const w0: @Vector(4, u16) = w[c + 0 ..][0..4].*;
        const w1: @Vector(4, u16) = w[c + 4 ..][0..4].*;
        const w2: @Vector(4, u16) = w[c + 8 ..][0..4].*;
        const w3: @Vector(4, u16) = w[c + 12 ..][0..4].*;
        const w4: @Vector(4, u16) = w[c + 16 ..][0..4].*;
        const w5: @Vector(4, u16) = w[c + 20 ..][0..4].*;
        const w6: @Vector(4, u16) = w[c + 24 ..][0..4].*;
        const w7: @Vector(4, u16) = w[c + 28 ..][0..4].*;
        const w8: @Vector(4, u16) = w[c + 32 ..][0..4].*;
        const w9: @Vector(4, u16) = w[c + 36 ..][0..4].*;
        const w10: @Vector(4, u16) = w[c + 40 ..][0..4].*;
        const w11: @Vector(4, u16) = w[c + 44 ..][0..4].*;
        const w12: @Vector(4, u16) = w[c + 48 ..][0..4].*;
        const w13: @Vector(4, u16) = w[c + 52 ..][0..4].*;
        const w14: @Vector(4, u16) = w[c + 56 ..][0..4].*;
        const w15: @Vector(4, u16) = w[c + 60 ..][0..4].*;

        const fw0: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0) << @splat(16));
        const fw1: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1) << @splat(16));
        const fw2: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w2) << @splat(16));
        const fw3: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w3) << @splat(16));
        const fw4: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w4) << @splat(16));
        const fw5: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w5) << @splat(16));
        const fw6: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w6) << @splat(16));
        const fw7: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w7) << @splat(16));
        const fw8: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w8) << @splat(16));
        const fw9: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w9) << @splat(16));
        const fw10: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w10) << @splat(16));
        const fw11: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w11) << @splat(16));
        const fw12: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w12) << @splat(16));
        const fw13: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w13) << @splat(16));
        const fw14: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w14) << @splat(16));
        const fw15: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w15) << @splat(16));

        const x0: @Vector(4, f32) = x[c + 0 ..][0..4].*;
        const x1: @Vector(4, f32) = x[c + 4 ..][0..4].*;
        const x2: @Vector(4, f32) = x[c + 8 ..][0..4].*;
        const x3: @Vector(4, f32) = x[c + 12 ..][0..4].*;
        const x4: @Vector(4, f32) = x[c + 16 ..][0..4].*;
        const x5: @Vector(4, f32) = x[c + 20 ..][0..4].*;
        const x6: @Vector(4, f32) = x[c + 24 ..][0..4].*;
        const x7: @Vector(4, f32) = x[c + 28 ..][0..4].*;
        const x8: @Vector(4, f32) = x[c + 32 ..][0..4].*;
        const x9: @Vector(4, f32) = x[c + 36 ..][0..4].*;
        const x10: @Vector(4, f32) = x[c + 40 ..][0..4].*;
        const x11: @Vector(4, f32) = x[c + 44 ..][0..4].*;
        const x12: @Vector(4, f32) = x[c + 48 ..][0..4].*;
        const x13: @Vector(4, f32) = x[c + 52 ..][0..4].*;
        const x14: @Vector(4, f32) = x[c + 56 ..][0..4].*;
        const x15: @Vector(4, f32) = x[c + 60 ..][0..4].*;

        d0 = @mulAdd(@Vector(4, f32), fw0, x0, d0);
        d1 = @mulAdd(@Vector(4, f32), fw1, x1, d1);
        d2 = @mulAdd(@Vector(4, f32), fw2, x2, d2);
        d3 = @mulAdd(@Vector(4, f32), fw3, x3, d3);
        d4 = @mulAdd(@Vector(4, f32), fw4, x4, d4);
        d5 = @mulAdd(@Vector(4, f32), fw5, x5, d5);
        d6 = @mulAdd(@Vector(4, f32), fw6, x6, d6);
        d7 = @mulAdd(@Vector(4, f32), fw7, x7, d7);
        d8 = @mulAdd(@Vector(4, f32), fw8, x8, d8);
        d9 = @mulAdd(@Vector(4, f32), fw9, x9, d9);
        d10 = @mulAdd(@Vector(4, f32), fw10, x10, d10);
        d11 = @mulAdd(@Vector(4, f32), fw11, x11, d11);
        d12 = @mulAdd(@Vector(4, f32), fw12, x12, d12);
        d13 = @mulAdd(@Vector(4, f32), fw13, x13, d13);
        d14 = @mulAdd(@Vector(4, f32), fw14, x14, d14);
        d15 = @mulAdd(@Vector(4, f32), fw15, x15, d15);
    }
    const sum_0_3 = (d0 + d1) + (d2 + d3);
    const sum_4_7 = (d4 + d5) + (d6 + d7);
    const sum_8_11 = (d8 + d9) + (d10 + d11);
    const sum_12_15 = (d12 + d13) + (d14 + d15);
    return @reduce(.Add, (sum_0_3 + sum_4_7) + (sum_8_11 + sum_12_15));
}

pub inline fn dotRowF16(w: [*]const f16, x: [*]const f16) f32 {
    var d0: @Vector(8, f16) = @splat(0.0);
    var d1: @Vector(8, f16) = @splat(0.0);
    var d2: @Vector(8, f16) = @splat(0.0);
    var d3: @Vector(8, f16) = @splat(0.0);
    var d4: @Vector(8, f16) = @splat(0.0);
    var d5: @Vector(8, f16) = @splat(0.0);
    var d6: @Vector(8, f16) = @splat(0.0);
    var d7: @Vector(8, f16) = @splat(0.0);

    var c: usize = 0;
    while (c + 64 <= 2048) : (c += 64) {
        const w0: @Vector(8, f16) = w[c + 0 ..][0..8].*;
        const w1: @Vector(8, f16) = w[c + 8 ..][0..8].*;
        const w2: @Vector(8, f16) = w[c + 16 ..][0..8].*;
        const w3: @Vector(8, f16) = w[c + 24 ..][0..8].*;
        const w4: @Vector(8, f16) = w[c + 32 ..][0..8].*;
        const w5: @Vector(8, f16) = w[c + 40 ..][0..8].*;
        const w6: @Vector(8, f16) = w[c + 48 ..][0..8].*;
        const w7: @Vector(8, f16) = w[c + 56 ..][0..8].*;

        const x0: @Vector(8, f16) = x[c + 0 ..][0..8].*;
        const x1: @Vector(8, f16) = x[c + 8 ..][0..8].*;
        const x2: @Vector(8, f16) = x[c + 16 ..][0..8].*;
        const x3: @Vector(8, f16) = x[c + 24 ..][0..8].*;
        const x4: @Vector(8, f16) = x[c + 32 ..][0..8].*;
        const x5: @Vector(8, f16) = x[c + 40 ..][0..8].*;
        const x6: @Vector(8, f16) = x[c + 48 ..][0..8].*;
        const x7: @Vector(8, f16) = x[c + 56 ..][0..8].*;

        d0 = @mulAdd(@Vector(8, f16), w0, x0, d0);
        d1 = @mulAdd(@Vector(8, f16), w1, x1, d1);
        d2 = @mulAdd(@Vector(8, f16), w2, x2, d2);
        d3 = @mulAdd(@Vector(8, f16), w3, x3, d3);
        d4 = @mulAdd(@Vector(8, f16), w4, x4, d4);
        d5 = @mulAdd(@Vector(8, f16), w5, x5, d5);
        d6 = @mulAdd(@Vector(8, f16), w6, x6, d6);
        d7 = @mulAdd(@Vector(8, f16), w7, x7, d7);
    }
    const sum_0_3 = (d0 + d1) + (d2 + d3);
    const sum_4_7 = (d4 + d5) + (d6 + d7);
    const total_f16 = sum_0_3 + sum_4_7;
    const f32_v: @Vector(8, f32) = @floatCast(total_f16);
    return @reduce(.Add, f32_v);
}

pub inline fn dotRow11008F16(w: [*]const f16, x: [*]const f16) f32 {
    var d0: @Vector(8, f16) = @splat(0.0);
    var d1: @Vector(8, f16) = @splat(0.0);
    var d2: @Vector(8, f16) = @splat(0.0);
    var d3: @Vector(8, f16) = @splat(0.0);
    var d4: @Vector(8, f16) = @splat(0.0);
    var d5: @Vector(8, f16) = @splat(0.0);
    var d6: @Vector(8, f16) = @splat(0.0);
    var d7: @Vector(8, f16) = @splat(0.0);

    var c: usize = 0;
    while (c < 11008) : (c += 64) {
        const w0: @Vector(8, f16) = w[c + 0 ..][0..8].*;
        const w1: @Vector(8, f16) = w[c + 8 ..][0..8].*;
        const w2: @Vector(8, f16) = w[c + 16 ..][0..8].*;
        const w3: @Vector(8, f16) = w[c + 24 ..][0..8].*;
        const w4: @Vector(8, f16) = w[c + 32 ..][0..8].*;
        const w5: @Vector(8, f16) = w[c + 40 ..][0..8].*;
        const w6: @Vector(8, f16) = w[c + 48 ..][0..8].*;
        const w7: @Vector(8, f16) = w[c + 56 ..][0..8].*;

        const x0: @Vector(8, f16) = x[c + 0 ..][0..8].*;
        const x1: @Vector(8, f16) = x[c + 8 ..][0..8].*;
        const x2: @Vector(8, f16) = x[c + 16 ..][0..8].*;
        const x3: @Vector(8, f16) = x[c + 24 ..][0..8].*;
        const x4: @Vector(8, f16) = x[c + 32 ..][0..8].*;
        const x5: @Vector(8, f16) = x[c + 40 ..][0..8].*;
        const x6: @Vector(8, f16) = x[c + 48 ..][0..8].*;
        const x7: @Vector(8, f16) = x[c + 56 ..][0..8].*;

        d0 = @mulAdd(@Vector(8, f16), w0, x0, d0);
        d1 = @mulAdd(@Vector(8, f16), w1, x1, d1);
        d2 = @mulAdd(@Vector(8, f16), w2, x2, d2);
        d3 = @mulAdd(@Vector(8, f16), w3, x3, d3);
        d4 = @mulAdd(@Vector(8, f16), w4, x4, d4);
        d5 = @mulAdd(@Vector(8, f16), w5, x5, d5);
        d6 = @mulAdd(@Vector(8, f16), w6, x6, d6);
        d7 = @mulAdd(@Vector(8, f16), w7, x7, d7);
    }
    const sum_0_3 = (d0 + d1) + (d2 + d3);
    const sum_4_7 = (d4 + d5) + (d6 + d7);
    const total_f16 = sum_0_3 + sum_4_7;
    const f32_v: @Vector(8, f32) = @floatCast(total_f16);
    return @reduce(.Add, f32_v);
}

pub inline fn gemvDualRowBF16Values(
    r0: [*]const u16,
    r1: [*]const u16,
    x: []const f32,
) [2]f32 {
    var d0a: @Vector(4, f32) = @splat(0.0);
    var d0b: @Vector(4, f32) = @splat(0.0);
    var d0c: @Vector(4, f32) = @splat(0.0);
    var d0d: @Vector(4, f32) = @splat(0.0);

    var d1a: @Vector(4, f32) = @splat(0.0);
    var d1b: @Vector(4, f32) = @splat(0.0);
    var d1c: @Vector(4, f32) = @splat(0.0);
    var d1d: @Vector(4, f32) = @splat(0.0);

    var c: usize = 0;
    while (c < 2048) : (c += 32) {
        const xa0: @Vector(4, f32) = x[c + 0 ..][0..4].*;
        const xa1: @Vector(4, f32) = x[c + 4 ..][0..4].*;
        const xa2: @Vector(4, f32) = x[c + 8 ..][0..4].*;
        const xa3: @Vector(4, f32) = x[c + 12 ..][0..4].*;
        const xa4: @Vector(4, f32) = x[c + 16 ..][0..4].*;
        const xa5: @Vector(4, f32) = x[c + 20 ..][0..4].*;
        const xa6: @Vector(4, f32) = x[c + 24 ..][0..4].*;
        const xa7: @Vector(4, f32) = x[c + 28 ..][0..4].*;

        // Row 0 - 32 elements (64 bytes = 1 cacheline)
        const w0_0: @Vector(4, u16) = r0[c + 0 ..][0..4].*;
        const w0_1: @Vector(4, u16) = r0[c + 4 ..][0..4].*;
        const w0_2: @Vector(4, u16) = r0[c + 8 ..][0..4].*;
        const w0_3: @Vector(4, u16) = r0[c + 12 ..][0..4].*;
        const w0_4: @Vector(4, u16) = r0[c + 16 ..][0..4].*;
        const w0_5: @Vector(4, u16) = r0[c + 20 ..][0..4].*;
        const w0_6: @Vector(4, u16) = r0[c + 24 ..][0..4].*;
        const w0_7: @Vector(4, u16) = r0[c + 28 ..][0..4].*;

        const fw0_0: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0_0) << @splat(16));
        const fw0_1: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0_1) << @splat(16));
        const fw0_2: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0_2) << @splat(16));
        const fw0_3: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0_3) << @splat(16));
        const fw0_4: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0_4) << @splat(16));
        const fw0_5: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0_5) << @splat(16));
        const fw0_6: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0_6) << @splat(16));
        const fw0_7: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0_7) << @splat(16));

        d0a = @mulAdd(@Vector(4, f32), fw0_0, xa0, d0a);
        d0b = @mulAdd(@Vector(4, f32), fw0_1, xa1, d0b);
        d0c = @mulAdd(@Vector(4, f32), fw0_2, xa2, d0c);
        d0d = @mulAdd(@Vector(4, f32), fw0_3, xa3, d0d);
        d0a = @mulAdd(@Vector(4, f32), fw0_4, xa4, d0a);
        d0b = @mulAdd(@Vector(4, f32), fw0_5, xa5, d0b);
        d0c = @mulAdd(@Vector(4, f32), fw0_6, xa6, d0c);
        d0d = @mulAdd(@Vector(4, f32), fw0_7, xa7, d0d);

        // Row 1 - 32 elements (64 bytes = 1 cacheline)
        const w1_0: @Vector(4, u16) = r1[c + 0 ..][0..4].*;
        const w1_1: @Vector(4, u16) = r1[c + 4 ..][0..4].*;
        const w1_2: @Vector(4, u16) = r1[c + 8 ..][0..4].*;
        const w1_3: @Vector(4, u16) = r1[c + 12 ..][0..4].*;
        const w1_4: @Vector(4, u16) = r1[c + 16 ..][0..4].*;
        const w1_5: @Vector(4, u16) = r1[c + 20 ..][0..4].*;
        const w1_6: @Vector(4, u16) = r1[c + 24 ..][0..4].*;
        const w1_7: @Vector(4, u16) = r1[c + 28 ..][0..4].*;

        const fw1_0: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1_0) << @splat(16));
        const fw1_1: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1_1) << @splat(16));
        const fw1_2: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1_2) << @splat(16));
        const fw1_3: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1_3) << @splat(16));
        const fw1_4: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1_4) << @splat(16));
        const fw1_5: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1_5) << @splat(16));
        const fw1_6: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1_6) << @splat(16));
        const fw1_7: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1_7) << @splat(16));

        d1a = @mulAdd(@Vector(4, f32), fw1_0, xa0, d1a);
        d1b = @mulAdd(@Vector(4, f32), fw1_1, xa1, d1b);
        d1c = @mulAdd(@Vector(4, f32), fw1_2, xa2, d1c);
        d1d = @mulAdd(@Vector(4, f32), fw1_3, xa3, d1d);
        d1a = @mulAdd(@Vector(4, f32), fw1_4, xa4, d1a);
        d1b = @mulAdd(@Vector(4, f32), fw1_5, xa5, d1b);
        d1c = @mulAdd(@Vector(4, f32), fw1_6, xa6, d1c);
        d1d = @mulAdd(@Vector(4, f32), fw1_7, xa7, d1d);
    }

    return .{
        @reduce(.Add, (d0a + d0b) + (d0c + d0d)),
        @reduce(.Add, (d1a + d1b) + (d1c + d1d)),
    };
}

pub inline fn dotDualRow11008BF16(
    r0: [*]const u16,
    r1: [*]const u16,
    x: [*]const f32,
) [2]f32 {
    var d0a: @Vector(4, f32) = @splat(0.0);
    var d0b: @Vector(4, f32) = @splat(0.0);
    var d0c: @Vector(4, f32) = @splat(0.0);
    var d0d: @Vector(4, f32) = @splat(0.0);

    var d1a: @Vector(4, f32) = @splat(0.0);
    var d1b: @Vector(4, f32) = @splat(0.0);
    var d1c: @Vector(4, f32) = @splat(0.0);
    var d1d: @Vector(4, f32) = @splat(0.0);

    var c: usize = 0;
    while (c < 11008) : (c += 32) {
        const xa0: @Vector(4, f32) = x[c + 0 ..][0..4].*;
        const xa1: @Vector(4, f32) = x[c + 4 ..][0..4].*;
        const xa2: @Vector(4, f32) = x[c + 8 ..][0..4].*;
        const xa3: @Vector(4, f32) = x[c + 12 ..][0..4].*;
        const xa4: @Vector(4, f32) = x[c + 16 ..][0..4].*;
        const xa5: @Vector(4, f32) = x[c + 20 ..][0..4].*;
        const xa6: @Vector(4, f32) = x[c + 24 ..][0..4].*;
        const xa7: @Vector(4, f32) = x[c + 28 ..][0..4].*;

        // Row 0 - 32 elements (64 bytes = 1 cacheline)
        const w0_0: @Vector(4, u16) = r0[c + 0 ..][0..4].*;
        const w0_1: @Vector(4, u16) = r0[c + 4 ..][0..4].*;
        const w0_2: @Vector(4, u16) = r0[c + 8 ..][0..4].*;
        const w0_3: @Vector(4, u16) = r0[c + 12 ..][0..4].*;
        const w0_4: @Vector(4, u16) = r0[c + 16 ..][0..4].*;
        const w0_5: @Vector(4, u16) = r0[c + 20 ..][0..4].*;
        const w0_6: @Vector(4, u16) = r0[c + 24 ..][0..4].*;
        const w0_7: @Vector(4, u16) = r0[c + 28 ..][0..4].*;

        const fw0_0: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0_0) << @splat(16));
        const fw0_1: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0_1) << @splat(16));
        const fw0_2: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0_2) << @splat(16));
        const fw0_3: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0_3) << @splat(16));
        const fw0_4: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0_4) << @splat(16));
        const fw0_5: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0_5) << @splat(16));
        const fw0_6: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0_6) << @splat(16));
        const fw0_7: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0_7) << @splat(16));

        d0a = @mulAdd(@Vector(4, f32), fw0_0, xa0, d0a);
        d0b = @mulAdd(@Vector(4, f32), fw0_1, xa1, d0b);
        d0c = @mulAdd(@Vector(4, f32), fw0_2, xa2, d0c);
        d0d = @mulAdd(@Vector(4, f32), fw0_3, xa3, d0d);
        d0a = @mulAdd(@Vector(4, f32), fw0_4, xa4, d0a);
        d0b = @mulAdd(@Vector(4, f32), fw0_5, xa5, d0b);
        d0c = @mulAdd(@Vector(4, f32), fw0_6, xa6, d0c);
        d0d = @mulAdd(@Vector(4, f32), fw0_7, xa7, d0d);

        // Row 1 - 32 elements (64 bytes = 1 cacheline)
        const w1_0: @Vector(4, u16) = r1[c + 0 ..][0..4].*;
        const w1_1: @Vector(4, u16) = r1[c + 4 ..][0..4].*;
        const w1_2: @Vector(4, u16) = r1[c + 8 ..][0..4].*;
        const w1_3: @Vector(4, u16) = r1[c + 12 ..][0..4].*;
        const w1_4: @Vector(4, u16) = r1[c + 16 ..][0..4].*;
        const w1_5: @Vector(4, u16) = r1[c + 20 ..][0..4].*;
        const w1_6: @Vector(4, u16) = r1[c + 24 ..][0..4].*;
        const w1_7: @Vector(4, u16) = r1[c + 28 ..][0..4].*;

        const fw1_0: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1_0) << @splat(16));
        const fw1_1: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1_1) << @splat(16));
        const fw1_2: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1_2) << @splat(16));
        const fw1_3: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1_3) << @splat(16));
        const fw1_4: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1_4) << @splat(16));
        const fw1_5: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1_5) << @splat(16));
        const fw1_6: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1_6) << @splat(16));
        const fw1_7: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1_7) << @splat(16));

        d1a = @mulAdd(@Vector(4, f32), fw1_0, xa0, d1a);
        d1b = @mulAdd(@Vector(4, f32), fw1_1, xa1, d1b);
        d1c = @mulAdd(@Vector(4, f32), fw1_2, xa2, d1c);
        d1d = @mulAdd(@Vector(4, f32), fw1_3, xa3, d1d);
        d1a = @mulAdd(@Vector(4, f32), fw1_4, xa4, d1a);
        d1b = @mulAdd(@Vector(4, f32), fw1_5, xa5, d1b);
        d1c = @mulAdd(@Vector(4, f32), fw1_6, xa6, d1c);
        d1d = @mulAdd(@Vector(4, f32), fw1_7, xa7, d1d);
    }

    return .{
        @reduce(.Add, (d0a + d0b) + (d0c + d0d)),
        @reduce(.Add, (d1a + d1b) + (d1c + d1d)),
    };
}

pub inline fn dotQuadRow11008BF16(
    r0: [*]const u16,
    r1: [*]const u16,
    r2: [*]const u16,
    r3: [*]const u16,
    x: [*]const f32,
) [4]f32 {
    var d0a: @Vector(4, f32) = @splat(0.0);
    var d0b: @Vector(4, f32) = @splat(0.0);
    var d0c: @Vector(4, f32) = @splat(0.0);
    var d0d: @Vector(4, f32) = @splat(0.0);

    var d1a: @Vector(4, f32) = @splat(0.0);
    var d1b: @Vector(4, f32) = @splat(0.0);
    var d1c: @Vector(4, f32) = @splat(0.0);
    var d1d: @Vector(4, f32) = @splat(0.0);

    var d2a: @Vector(4, f32) = @splat(0.0);
    var d2b: @Vector(4, f32) = @splat(0.0);
    var d2c: @Vector(4, f32) = @splat(0.0);
    var d2d: @Vector(4, f32) = @splat(0.0);

    var d3a: @Vector(4, f32) = @splat(0.0);
    var d3b: @Vector(4, f32) = @splat(0.0);
    var d3c: @Vector(4, f32) = @splat(0.0);
    var d3d: @Vector(4, f32) = @splat(0.0);

    var c: usize = 0;
    while (c < 11008) : (c += 32) {
        const xa0: @Vector(4, f32) = x[c + 0 ..][0..4].*;
        const xa1: @Vector(4, f32) = x[c + 4 ..][0..4].*;
        const xa2: @Vector(4, f32) = x[c + 8 ..][0..4].*;
        const xa3: @Vector(4, f32) = x[c + 12 ..][0..4].*;
        const xa4: @Vector(4, f32) = x[c + 16 ..][0..4].*;
        const xa5: @Vector(4, f32) = x[c + 20 ..][0..4].*;
        const xa6: @Vector(4, f32) = x[c + 24 ..][0..4].*;
        const xa7: @Vector(4, f32) = x[c + 28 ..][0..4].*;

        // Row 0
        const w0_0: @Vector(4, u16) = r0[c + 0 ..][0..4].*;
        const w0_1: @Vector(4, u16) = r0[c + 4 ..][0..4].*;
        const w0_2: @Vector(4, u16) = r0[c + 8 ..][0..4].*;
        const w0_3: @Vector(4, u16) = r0[c + 12 ..][0..4].*;
        const w0_4: @Vector(4, u16) = r0[c + 16 ..][0..4].*;
        const w0_5: @Vector(4, u16) = r0[c + 20 ..][0..4].*;
        const w0_6: @Vector(4, u16) = r0[c + 24 ..][0..4].*;
        const w0_7: @Vector(4, u16) = r0[c + 28 ..][0..4].*;

        const fw0_0: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0_0) << @splat(16));
        const fw0_1: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0_1) << @splat(16));
        const fw0_2: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0_2) << @splat(16));
        const fw0_3: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0_3) << @splat(16));
        const fw0_4: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0_4) << @splat(16));
        const fw0_5: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0_5) << @splat(16));
        const fw0_6: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0_6) << @splat(16));
        const fw0_7: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0_7) << @splat(16));

        d0a = @mulAdd(@Vector(4, f32), fw0_0, xa0, d0a);
        d0b = @mulAdd(@Vector(4, f32), fw0_1, xa1, d0b);
        d0c = @mulAdd(@Vector(4, f32), fw0_2, xa2, d0c);
        d0d = @mulAdd(@Vector(4, f32), fw0_3, xa3, d0d);
        d0a = @mulAdd(@Vector(4, f32), fw0_4, xa4, d0a);
        d0b = @mulAdd(@Vector(4, f32), fw0_5, xa5, d0b);
        d0c = @mulAdd(@Vector(4, f32), fw0_6, xa6, d0c);
        d0d = @mulAdd(@Vector(4, f32), fw0_7, xa7, d0d);

        // Row 1
        const w1_0: @Vector(4, u16) = r1[c + 0 ..][0..4].*;
        const w1_1: @Vector(4, u16) = r1[c + 4 ..][0..4].*;
        const w1_2: @Vector(4, u16) = r1[c + 8 ..][0..4].*;
        const w1_3: @Vector(4, u16) = r1[c + 12 ..][0..4].*;
        const w1_4: @Vector(4, u16) = r1[c + 16 ..][0..4].*;
        const w1_5: @Vector(4, u16) = r1[c + 20 ..][0..4].*;
        const w1_6: @Vector(4, u16) = r1[c + 24 ..][0..4].*;
        const w1_7: @Vector(4, u16) = r1[c + 28 ..][0..4].*;

        const fw1_0: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1_0) << @splat(16));
        const fw1_1: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1_1) << @splat(16));
        const fw1_2: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1_2) << @splat(16));
        const fw1_3: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1_3) << @splat(16));
        const fw1_4: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1_4) << @splat(16));
        const fw1_5: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1_5) << @splat(16));
        const fw1_6: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1_6) << @splat(16));
        const fw1_7: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1_7) << @splat(16));

        d1a = @mulAdd(@Vector(4, f32), fw1_0, xa0, d1a);
        d1b = @mulAdd(@Vector(4, f32), fw1_1, xa1, d1b);
        d1c = @mulAdd(@Vector(4, f32), fw1_2, xa2, d1c);
        d1d = @mulAdd(@Vector(4, f32), fw1_3, xa3, d1d);
        d1a = @mulAdd(@Vector(4, f32), fw1_4, xa4, d1a);
        d1b = @mulAdd(@Vector(4, f32), fw1_5, xa5, d1b);
        d1c = @mulAdd(@Vector(4, f32), fw1_6, xa6, d1c);
        d1d = @mulAdd(@Vector(4, f32), fw1_7, xa7, d1d);

        // Row 2
        const w2_0: @Vector(4, u16) = r2[c + 0 ..][0..4].*;
        const w2_1: @Vector(4, u16) = r2[c + 4 ..][0..4].*;
        const w2_2: @Vector(4, u16) = r2[c + 8 ..][0..4].*;
        const w2_3: @Vector(4, u16) = r2[c + 12 ..][0..4].*;
        const w2_4: @Vector(4, u16) = r2[c + 16 ..][0..4].*;
        const w2_5: @Vector(4, u16) = r2[c + 20 ..][0..4].*;
        const w2_6: @Vector(4, u16) = r2[c + 24 ..][0..4].*;
        const w2_7: @Vector(4, u16) = r2[c + 28 ..][0..4].*;

        const fw2_0: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w2_0) << @splat(16));
        const fw2_1: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w2_1) << @splat(16));
        const fw2_2: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w2_2) << @splat(16));
        const fw2_3: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w2_3) << @splat(16));
        const fw2_4: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w2_4) << @splat(16));
        const fw2_5: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w2_5) << @splat(16));
        const fw2_6: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w2_6) << @splat(16));
        const fw2_7: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w2_7) << @splat(16));

        d2a = @mulAdd(@Vector(4, f32), fw2_0, xa0, d2a);
        d2b = @mulAdd(@Vector(4, f32), fw2_1, xa1, d2b);
        d2c = @mulAdd(@Vector(4, f32), fw2_2, xa2, d2c);
        d2d = @mulAdd(@Vector(4, f32), fw2_3, xa3, d2d);
        d2a = @mulAdd(@Vector(4, f32), fw2_4, xa4, d2a);
        d2b = @mulAdd(@Vector(4, f32), fw2_5, xa5, d2b);
        d2c = @mulAdd(@Vector(4, f32), fw2_6, xa6, d2c);
        d2d = @mulAdd(@Vector(4, f32), fw2_7, xa7, d2d);

        // Row 3
        const w3_0: @Vector(4, u16) = r3[c + 0 ..][0..4].*;
        const w3_1: @Vector(4, u16) = r3[c + 4 ..][0..4].*;
        const w3_2: @Vector(4, u16) = r3[c + 8 ..][0..4].*;
        const w3_3: @Vector(4, u16) = r3[c + 12 ..][0..4].*;
        const w3_4: @Vector(4, u16) = r3[c + 16 ..][0..4].*;
        const w3_5: @Vector(4, u16) = r3[c + 20 ..][0..4].*;
        const w3_6: @Vector(4, u16) = r3[c + 24 ..][0..4].*;
        const w3_7: @Vector(4, u16) = r3[c + 28 ..][0..4].*;

        const fw3_0: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w3_0) << @splat(16));
        const fw3_1: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w3_1) << @splat(16));
        const fw3_2: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w3_2) << @splat(16));
        const fw3_3: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w3_3) << @splat(16));
        const fw3_4: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w3_4) << @splat(16));
        const fw3_5: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w3_5) << @splat(16));
        const fw3_6: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w3_6) << @splat(16));
        const fw3_7: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w3_7) << @splat(16));

        d3a = @mulAdd(@Vector(4, f32), fw3_0, xa0, d3a);
        d3b = @mulAdd(@Vector(4, f32), fw3_1, xa1, d3b);
        d3c = @mulAdd(@Vector(4, f32), fw3_2, xa2, d3c);
        d3d = @mulAdd(@Vector(4, f32), fw3_3, xa3, d3d);
        d3a = @mulAdd(@Vector(4, f32), fw3_4, xa4, d3a);
        d3b = @mulAdd(@Vector(4, f32), fw3_5, xa5, d3b);
        d3c = @mulAdd(@Vector(4, f32), fw3_6, xa6, d3c);
        d3d = @mulAdd(@Vector(4, f32), fw3_7, xa7, d3d);
    }

    return .{
        @reduce(.Add, (d0a + d0b) + (d0c + d0d)),
        @reduce(.Add, (d1a + d1b) + (d1c + d1d)),
        @reduce(.Add, (d2a + d2b) + (d2c + d2d)),
        @reduce(.Add, (d3a + d3b) + (d3c + d3d)),
    };
}

pub inline fn dotOctRow11008BF16(
    r0: [*]const u16,
    r1: [*]const u16,
    r2: [*]const u16,
    r3: [*]const u16,
    r4: [*]const u16,
    r5: [*]const u16,
    r6: [*]const u16,
    r7: [*]const u16,
    x: [*]const f32,
) [8]f32 {
    var d0a: @Vector(4, f32) = @splat(0.0);
    var d0b: @Vector(4, f32) = @splat(0.0);
    var d1a: @Vector(4, f32) = @splat(0.0);
    var d1b: @Vector(4, f32) = @splat(0.0);
    var d2a: @Vector(4, f32) = @splat(0.0);
    var d2b: @Vector(4, f32) = @splat(0.0);
    var d3a: @Vector(4, f32) = @splat(0.0);
    var d3b: @Vector(4, f32) = @splat(0.0);
    var d4a: @Vector(4, f32) = @splat(0.0);
    var d4b: @Vector(4, f32) = @splat(0.0);
    var d5a: @Vector(4, f32) = @splat(0.0);
    var d5b: @Vector(4, f32) = @splat(0.0);
    var d6a: @Vector(4, f32) = @splat(0.0);
    var d6b: @Vector(4, f32) = @splat(0.0);
    var d7a: @Vector(4, f32) = @splat(0.0);
    var d7b: @Vector(4, f32) = @splat(0.0);

    var c: usize = 0;
    while (c < 11008) : (c += 32) {
        const xa0: @Vector(4, f32) = x[c + 0 ..][0..4].*;
        const xa1: @Vector(4, f32) = x[c + 4 ..][0..4].*;
        const xa2: @Vector(4, f32) = x[c + 8 ..][0..4].*;
        const xa3: @Vector(4, f32) = x[c + 12 ..][0..4].*;
        const xa4: @Vector(4, f32) = x[c + 16 ..][0..4].*;
        const xa5: @Vector(4, f32) = x[c + 20 ..][0..4].*;
        const xa6: @Vector(4, f32) = x[c + 24 ..][0..4].*;
        const xa7: @Vector(4, f32) = x[c + 28 ..][0..4].*;

        inline for (.{
            .{ r0, &d0a, &d0b },
            .{ r1, &d1a, &d1b },
            .{ r2, &d2a, &d2b },
            .{ r3, &d3a, &d3b },
            .{ r4, &d4a, &d4b },
            .{ r5, &d5a, &d5b },
            .{ r6, &d6a, &d6b },
            .{ r7, &d7a, &d7b },
        }) |row_spec| {
            const r = row_spec[0];
            const p_da = row_spec[1];
            const p_db = row_spec[2];

            const w0: @Vector(4, u16) = r[c + 0 ..][0..4].*;
            const w1: @Vector(4, u16) = r[c + 4 ..][0..4].*;
            const w2: @Vector(4, u16) = r[c + 8 ..][0..4].*;
            const w3: @Vector(4, u16) = r[c + 12 ..][0..4].*;
            const w4: @Vector(4, u16) = r[c + 16 ..][0..4].*;
            const w5: @Vector(4, u16) = r[c + 20 ..][0..4].*;
            const w6: @Vector(4, u16) = r[c + 24 ..][0..4].*;
            const w7: @Vector(4, u16) = r[c + 28 ..][0..4].*;

            const fw0: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0) << @splat(16));
            const fw1: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1) << @splat(16));
            const fw2: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w2) << @splat(16));
            const fw3: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w3) << @splat(16));
            const fw4: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w4) << @splat(16));
            const fw5: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w5) << @splat(16));
            const fw6: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w6) << @splat(16));
            const fw7: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w7) << @splat(16));

            p_da.* = @mulAdd(@Vector(4, f32), fw0, xa0, p_da.*);
            p_db.* = @mulAdd(@Vector(4, f32), fw1, xa1, p_db.*);
            p_da.* = @mulAdd(@Vector(4, f32), fw2, xa2, p_da.*);
            p_db.* = @mulAdd(@Vector(4, f32), fw3, xa3, p_db.*);
            p_da.* = @mulAdd(@Vector(4, f32), fw4, xa4, p_da.*);
            p_db.* = @mulAdd(@Vector(4, f32), fw5, xa5, p_db.*);
            p_da.* = @mulAdd(@Vector(4, f32), fw6, xa6, p_da.*);
            p_db.* = @mulAdd(@Vector(4, f32), fw7, xa7, p_db.*);
        }
    }

    return .{
        @reduce(.Add, d0a + d0b),
        @reduce(.Add, d1a + d1b),
        @reduce(.Add, d2a + d2b),
        @reduce(.Add, d3a + d3b),
        @reduce(.Add, d4a + d4b),
        @reduce(.Add, d5a + d5b),
        @reduce(.Add, d6a + d6b),
        @reduce(.Add, d7a + d7b),
    };
}

pub inline fn gemvTileBF16Direct(coded_u16: [*]const u16, x: []const f32) [4]f32 {
    return gemvQuadRowBF16Values(coded_u16, x);
}

pub inline fn gemvDualRowF16Values(
    r0: [*]const f16,
    r1: [*]const f16,
    x: [*]const f16,
) [2]f32 {
    var d0a: @Vector(8, f16) = @splat(0.0);
    var d0b: @Vector(8, f16) = @splat(0.0);
    var d0c: @Vector(8, f16) = @splat(0.0);
    var d0d: @Vector(8, f16) = @splat(0.0);

    var d1a: @Vector(8, f16) = @splat(0.0);
    var d1b: @Vector(8, f16) = @splat(0.0);
    var d1c: @Vector(8, f16) = @splat(0.0);
    var d1d: @Vector(8, f16) = @splat(0.0);

    var c: usize = 0;
    while (c < 2048) : (c += 32) {
        const x0: @Vector(8, f16) = x[c + 0 ..][0..8].*;
        const x1: @Vector(8, f16) = x[c + 8 ..][0..8].*;
        const x2: @Vector(8, f16) = x[c + 16 ..][0..8].*;
        const x3: @Vector(8, f16) = x[c + 24 ..][0..8].*;

        const w0_0: @Vector(8, f16) = r0[c + 0 ..][0..8].*;
        const w0_1: @Vector(8, f16) = r0[c + 8 ..][0..8].*;
        const w0_2: @Vector(8, f16) = r0[c + 16 ..][0..8].*;
        const w0_3: @Vector(8, f16) = r0[c + 24 ..][0..8].*;

        const w1_0: @Vector(8, f16) = r1[c + 0 ..][0..8].*;
        const w1_1: @Vector(8, f16) = r1[c + 8 ..][0..8].*;
        const w1_2: @Vector(8, f16) = r1[c + 16 ..][0..8].*;
        const w1_3: @Vector(8, f16) = r1[c + 24 ..][0..8].*;

        d0a = @mulAdd(@Vector(8, f16), w0_0, x0, d0a);
        d0b = @mulAdd(@Vector(8, f16), w0_1, x1, d0b);
        d0c = @mulAdd(@Vector(8, f16), w0_2, x2, d0c);
        d0d = @mulAdd(@Vector(8, f16), w0_3, x3, d0d);

        d1a = @mulAdd(@Vector(8, f16), w1_0, x0, d1a);
        d1b = @mulAdd(@Vector(8, f16), w1_1, x1, d1b);
        d1c = @mulAdd(@Vector(8, f16), w1_2, x2, d1c);
        d1d = @mulAdd(@Vector(8, f16), w1_3, x3, d1d);
    }

    const sum0 = (d0a + d0b) + (d0c + d0d);
    const sum1 = (d1a + d1b) + (d1c + d1d);
    const f32_sum0: @Vector(8, f32) = @floatCast(sum0);
    const f32_sum1: @Vector(8, f32) = @floatCast(sum1);

    return .{
        @reduce(.Add, f32_sum0),
        @reduce(.Add, f32_sum1),
    };
}

pub inline fn gemvQuadRowF16Values(
    r0: [*]const f16,
    r1: [*]const f16,
    r2: [*]const f16,
    r3: [*]const f16,
    x: [*]const f16,
) [4]f32 {
    var d0a: @Vector(8, f16) = @splat(0.0);
    var d0b: @Vector(8, f16) = @splat(0.0);
    var d0c: @Vector(8, f16) = @splat(0.0);
    var d0d: @Vector(8, f16) = @splat(0.0);

    var d1a: @Vector(8, f16) = @splat(0.0);
    var d1b: @Vector(8, f16) = @splat(0.0);
    var d1c: @Vector(8, f16) = @splat(0.0);
    var d1d: @Vector(8, f16) = @splat(0.0);

    var d2a: @Vector(8, f16) = @splat(0.0);
    var d2b: @Vector(8, f16) = @splat(0.0);
    var d2c: @Vector(8, f16) = @splat(0.0);
    var d2d: @Vector(8, f16) = @splat(0.0);

    var d3a: @Vector(8, f16) = @splat(0.0);
    var d3b: @Vector(8, f16) = @splat(0.0);
    var d3c: @Vector(8, f16) = @splat(0.0);
    var d3d: @Vector(8, f16) = @splat(0.0);

    var c: usize = 0;
    while (c < 2048) : (c += 32) {
        const x0: @Vector(8, f16) = x[c + 0 ..][0..8].*;
        const x1: @Vector(8, f16) = x[c + 8 ..][0..8].*;
        const x2: @Vector(8, f16) = x[c + 16 ..][0..8].*;
        const x3: @Vector(8, f16) = x[c + 24 ..][0..8].*;

        const w0_0: @Vector(8, f16) = r0[c + 0 ..][0..8].*;
        const w0_1: @Vector(8, f16) = r0[c + 8 ..][0..8].*;
        const w0_2: @Vector(8, f16) = r0[c + 16 ..][0..8].*;
        const w0_3: @Vector(8, f16) = r0[c + 24 ..][0..8].*;

        const w1_0: @Vector(8, f16) = r1[c + 0 ..][0..8].*;
        const w1_1: @Vector(8, f16) = r1[c + 8 ..][0..8].*;
        const w1_2: @Vector(8, f16) = r1[c + 16 ..][0..8].*;
        const w1_3: @Vector(8, f16) = r1[c + 24 ..][0..8].*;

        const w2_0: @Vector(8, f16) = r2[c + 0 ..][0..8].*;
        const w2_1: @Vector(8, f16) = r2[c + 8 ..][0..8].*;
        const w2_2: @Vector(8, f16) = r2[c + 16 ..][0..8].*;
        const w2_3: @Vector(8, f16) = r2[c + 24 ..][0..8].*;

        const w3_0: @Vector(8, f16) = r3[c + 0 ..][0..8].*;
        const w3_1: @Vector(8, f16) = r3[c + 8 ..][0..8].*;
        const w3_2: @Vector(8, f16) = r3[c + 16 ..][0..8].*;
        const w3_3: @Vector(8, f16) = r3[c + 24 ..][0..8].*;

        d0a = @mulAdd(@Vector(8, f16), w0_0, x0, d0a);
        d0b = @mulAdd(@Vector(8, f16), w0_1, x1, d0b);
        d0c = @mulAdd(@Vector(8, f16), w0_2, x2, d0c);
        d0d = @mulAdd(@Vector(8, f16), w0_3, x3, d0d);

        d1a = @mulAdd(@Vector(8, f16), w1_0, x0, d1a);
        d1b = @mulAdd(@Vector(8, f16), w1_1, x1, d1b);
        d1c = @mulAdd(@Vector(8, f16), w1_2, x2, d1c);
        d1d = @mulAdd(@Vector(8, f16), w1_3, x3, d1d);

        d2a = @mulAdd(@Vector(8, f16), w2_0, x0, d2a);
        d2b = @mulAdd(@Vector(8, f16), w2_1, x1, d2b);
        d2c = @mulAdd(@Vector(8, f16), w2_2, x2, d2c);
        d2d = @mulAdd(@Vector(8, f16), w2_3, x3, d2d);

        d3a = @mulAdd(@Vector(8, f16), w3_0, x0, d3a);
        d3b = @mulAdd(@Vector(8, f16), w3_1, x1, d3b);
        d3c = @mulAdd(@Vector(8, f16), w3_2, x2, d3c);
        d3d = @mulAdd(@Vector(8, f16), w3_3, x3, d3d);
    }

    const sum0 = (d0a + d0b) + (d0c + d0d);
    const sum1 = (d1a + d1b) + (d1c + d1d);
    const sum2 = (d2a + d2b) + (d2c + d2d);
    const sum3 = (d3a + d3b) + (d3c + d3d);
    const f32_sum0: @Vector(8, f32) = @floatCast(sum0);
    const f32_sum1: @Vector(8, f32) = @floatCast(sum1);
    const f32_sum2: @Vector(8, f32) = @floatCast(sum2);
    const f32_sum3: @Vector(8, f32) = @floatCast(sum3);

    return .{
        @reduce(.Add, f32_sum0),
        @reduce(.Add, f32_sum1),
        @reduce(.Add, f32_sum2),
        @reduce(.Add, f32_sum3),
    };
}

pub inline fn gemvTileF16Direct(coded_f16: [*]const f16, x_f16: [*]const f16) [4]f32 {
    const r0 = coded_f16 + 0 * 2048;
    const r1 = coded_f16 + 1 * 2048;
    const r2 = coded_f16 + 2 * 2048;
    const r3 = coded_f16 + 3 * 2048;
    return gemvQuadRowF16Values(r0, r1, r2, r3, x_f16);
}

pub inline fn dotDualRow11008F16(
    r0: [*]const f16,
    r1: [*]const f16,
    x: [*]const f16,
) [2]f32 {
    var d0a: @Vector(8, f16) = @splat(0.0);
    var d0b: @Vector(8, f16) = @splat(0.0);
    var d0c: @Vector(8, f16) = @splat(0.0);
    var d0d: @Vector(8, f16) = @splat(0.0);

    var d1a: @Vector(8, f16) = @splat(0.0);
    var d1b: @Vector(8, f16) = @splat(0.0);
    var d1c: @Vector(8, f16) = @splat(0.0);
    var d1d: @Vector(8, f16) = @splat(0.0);

    var c: usize = 0;
    while (c < 11008) : (c += 32) {
        const x0: @Vector(8, f16) = x[c + 0 ..][0..8].*;
        const x1: @Vector(8, f16) = x[c + 8 ..][0..8].*;
        const x2: @Vector(8, f16) = x[c + 16 ..][0..8].*;
        const x3: @Vector(8, f16) = x[c + 24 ..][0..8].*;

        const w0_0: @Vector(8, f16) = r0[c + 0 ..][0..8].*;
        const w0_1: @Vector(8, f16) = r0[c + 8 ..][0..8].*;
        const w0_2: @Vector(8, f16) = r0[c + 16 ..][0..8].*;
        const w0_3: @Vector(8, f16) = r0[c + 24 ..][0..8].*;

        const w1_0: @Vector(8, f16) = r1[c + 0 ..][0..8].*;
        const w1_1: @Vector(8, f16) = r1[c + 8 ..][0..8].*;
        const w1_2: @Vector(8, f16) = r1[c + 16 ..][0..8].*;
        const w1_3: @Vector(8, f16) = r1[c + 24 ..][0..8].*;

        d0a = @mulAdd(@Vector(8, f16), w0_0, x0, d0a);
        d0b = @mulAdd(@Vector(8, f16), w0_1, x1, d0b);
        d0c = @mulAdd(@Vector(8, f16), w0_2, x2, d0c);
        d0d = @mulAdd(@Vector(8, f16), w0_3, x3, d0d);

        d1a = @mulAdd(@Vector(8, f16), w1_0, x0, d1a);
        d1b = @mulAdd(@Vector(8, f16), w1_1, x1, d1b);
        d1c = @mulAdd(@Vector(8, f16), w1_2, x2, d1c);
        d1d = @mulAdd(@Vector(8, f16), w1_3, x3, d1d);
    }

    const sum0 = (d0a + d0b) + (d0c + d0d);
    const sum1 = (d1a + d1b) + (d1c + d1d);
    const f32_sum0: @Vector(8, f32) = @floatCast(sum0);
    const f32_sum1: @Vector(8, f32) = @floatCast(sum1);

    return .{
        @reduce(.Add, f32_sum0),
        @reduce(.Add, f32_sum1),
    };
}

pub inline fn dotQuadRow11008F16(
    r0: [*]const f16,
    r1: [*]const f16,
    r2: [*]const f16,
    r3: [*]const f16,
    x: [*]const f16,
) [4]f32 {
    var d0a: @Vector(8, f16) = @splat(0.0);
    var d0b: @Vector(8, f16) = @splat(0.0);
    var d0c: @Vector(8, f16) = @splat(0.0);
    var d0d: @Vector(8, f16) = @splat(0.0);

    var d1a: @Vector(8, f16) = @splat(0.0);
    var d1b: @Vector(8, f16) = @splat(0.0);
    var d1c: @Vector(8, f16) = @splat(0.0);
    var d1d: @Vector(8, f16) = @splat(0.0);

    var d2a: @Vector(8, f16) = @splat(0.0);
    var d2b: @Vector(8, f16) = @splat(0.0);
    var d2c: @Vector(8, f16) = @splat(0.0);
    var d2d: @Vector(8, f16) = @splat(0.0);

    var d3a: @Vector(8, f16) = @splat(0.0);
    var d3b: @Vector(8, f16) = @splat(0.0);
    var d3c: @Vector(8, f16) = @splat(0.0);
    var d3d: @Vector(8, f16) = @splat(0.0);

    var c: usize = 0;
    while (c < 11008) : (c += 32) {
        const x0: @Vector(8, f16) = x[c + 0 ..][0..8].*;
        const x1: @Vector(8, f16) = x[c + 8 ..][0..8].*;
        const x2: @Vector(8, f16) = x[c + 16 ..][0..8].*;
        const x3: @Vector(8, f16) = x[c + 24 ..][0..8].*;

        const w0_0: @Vector(8, f16) = r0[c + 0 ..][0..8].*;
        const w0_1: @Vector(8, f16) = r0[c + 8 ..][0..8].*;
        const w0_2: @Vector(8, f16) = r0[c + 16 ..][0..8].*;
        const w0_3: @Vector(8, f16) = r0[c + 24 ..][0..8].*;

        const w1_0: @Vector(8, f16) = r1[c + 0 ..][0..8].*;
        const w1_1: @Vector(8, f16) = r1[c + 8 ..][0..8].*;
        const w1_2: @Vector(8, f16) = r1[c + 16 ..][0..8].*;
        const w1_3: @Vector(8, f16) = r1[c + 24 ..][0..8].*;

        const w2_0: @Vector(8, f16) = r2[c + 0 ..][0..8].*;
        const w2_1: @Vector(8, f16) = r2[c + 8 ..][0..8].*;
        const w2_2: @Vector(8, f16) = r2[c + 16 ..][0..8].*;
        const w2_3: @Vector(8, f16) = r2[c + 24 ..][0..8].*;

        const w3_0: @Vector(8, f16) = r3[c + 0 ..][0..8].*;
        const w3_1: @Vector(8, f16) = r3[c + 8 ..][0..8].*;
        const w3_2: @Vector(8, f16) = r3[c + 16 ..][0..8].*;
        const w3_3: @Vector(8, f16) = r3[c + 24 ..][0..8].*;

        d0a = @mulAdd(@Vector(8, f16), w0_0, x0, d0a);
        d0b = @mulAdd(@Vector(8, f16), w0_1, x1, d0b);
        d0c = @mulAdd(@Vector(8, f16), w0_2, x2, d0c);
        d0d = @mulAdd(@Vector(8, f16), w0_3, x3, d0d);

        d1a = @mulAdd(@Vector(8, f16), w1_0, x0, d1a);
        d1b = @mulAdd(@Vector(8, f16), w1_1, x1, d1b);
        d1c = @mulAdd(@Vector(8, f16), w1_2, x2, d1c);
        d1d = @mulAdd(@Vector(8, f16), w1_3, x3, d1d);

        d2a = @mulAdd(@Vector(8, f16), w2_0, x0, d2a);
        d2b = @mulAdd(@Vector(8, f16), w2_1, x1, d2b);
        d2c = @mulAdd(@Vector(8, f16), w2_2, x2, d2c);
        d2d = @mulAdd(@Vector(8, f16), w2_3, x3, d2d);

        d3a = @mulAdd(@Vector(8, f16), w3_0, x0, d3a);
        d3b = @mulAdd(@Vector(8, f16), w3_1, x1, d3b);
        d3c = @mulAdd(@Vector(8, f16), w3_2, x2, d3c);
        d3d = @mulAdd(@Vector(8, f16), w3_3, x3, d3d);
    }

    const sum0 = (d0a + d0b) + (d0c + d0d);
    const sum1 = (d1a + d1b) + (d1c + d1d);
    const sum2 = (d2a + d2b) + (d2c + d2d);
    const sum3 = (d3a + d3b) + (d3c + d3d);
    const f32_sum0: @Vector(8, f32) = @floatCast(sum0);
    const f32_sum1: @Vector(8, f32) = @floatCast(sum1);
    const f32_sum2: @Vector(8, f32) = @floatCast(sum2);
    const f32_sum3: @Vector(8, f32) = @floatCast(sum3);

    return .{
        @reduce(.Add, f32_sum0),
        @reduce(.Add, f32_sum1),
        @reduce(.Add, f32_sum2),
        @reduce(.Add, f32_sum3),
    };
}

pub inline fn dotOctRow11008F16(
    r0: [*]const f16,
    r1: [*]const f16,
    r2: [*]const f16,
    r3: [*]const f16,
    r4: [*]const f16,
    r5: [*]const f16,
    r6: [*]const f16,
    r7: [*]const f16,
    x: [*]const f16,
) [8]f32 {
    var d0a: @Vector(8, f16) = @splat(0.0);
    var d0b: @Vector(8, f16) = @splat(0.0);
    var d1a: @Vector(8, f16) = @splat(0.0);
    var d1b: @Vector(8, f16) = @splat(0.0);
    var d2a: @Vector(8, f16) = @splat(0.0);
    var d2b: @Vector(8, f16) = @splat(0.0);
    var d3a: @Vector(8, f16) = @splat(0.0);
    var d3b: @Vector(8, f16) = @splat(0.0);
    var d4a: @Vector(8, f16) = @splat(0.0);
    var d4b: @Vector(8, f16) = @splat(0.0);
    var d5a: @Vector(8, f16) = @splat(0.0);
    var d5b: @Vector(8, f16) = @splat(0.0);
    var d6a: @Vector(8, f16) = @splat(0.0);
    var d6b: @Vector(8, f16) = @splat(0.0);
    var d7a: @Vector(8, f16) = @splat(0.0);
    var d7b: @Vector(8, f16) = @splat(0.0);

    var c: usize = 0;
    while (c < 11008) : (c += 32) {
        const x0: @Vector(8, f16) = x[c + 0 ..][0..8].*;
        const x1: @Vector(8, f16) = x[c + 8 ..][0..8].*;
        const x2: @Vector(8, f16) = x[c + 16 ..][0..8].*;
        const x3: @Vector(8, f16) = x[c + 24 ..][0..8].*;

        inline for (.{
            .{ r0, &d0a, &d0b },
            .{ r1, &d1a, &d1b },
            .{ r2, &d2a, &d2b },
            .{ r3, &d3a, &d3b },
            .{ r4, &d4a, &d4b },
            .{ r5, &d5a, &d5b },
            .{ r6, &d6a, &d6b },
            .{ r7, &d7a, &d7b },
        }) |row_spec| {
            const r = row_spec[0];
            const p_da = row_spec[1];
            const p_db = row_spec[2];

            const w0: @Vector(8, f16) = r[c + 0 ..][0..8].*;
            const w1: @Vector(8, f16) = r[c + 8 ..][0..8].*;
            const w2: @Vector(8, f16) = r[c + 16 ..][0..8].*;
            const w3: @Vector(8, f16) = r[c + 24 ..][0..8].*;

            p_da.* = @mulAdd(@Vector(8, f16), w0, x0, p_da.*);
            p_db.* = @mulAdd(@Vector(8, f16), w1, x1, p_db.*);
            p_da.* = @mulAdd(@Vector(8, f16), w2, x2, p_da.*);
            p_db.* = @mulAdd(@Vector(8, f16), w3, x3, p_db.*);
        }
    }

    const s0: @Vector(8, f32) = @floatCast(d0a + d0b);
    const s1: @Vector(8, f32) = @floatCast(d1a + d1b);
    const s2: @Vector(8, f32) = @floatCast(d2a + d2b);
    const s3: @Vector(8, f32) = @floatCast(d3a + d3b);
    const s4: @Vector(8, f32) = @floatCast(d4a + d4b);
    const s5: @Vector(8, f32) = @floatCast(d5a + d5b);
    const s6: @Vector(8, f32) = @floatCast(d6a + d6b);
    const s7: @Vector(8, f32) = @floatCast(d7a + d7b);

    return .{
        @reduce(.Add, s0),
        @reduce(.Add, s1),
        @reduce(.Add, s2),
        @reduce(.Add, s3),
        @reduce(.Add, s4),
        @reduce(.Add, s5),
        @reduce(.Add, s6),
        @reduce(.Add, s7),
    };
}

pub inline fn gemvTileSequentialBF16(coded_u16: [*]const u16, x: []const f32) [4]f32 {
    const x_ptr: [*]const f32 = x.ptr;
    return .{
        dotRowBF16(coded_u16 + 0 * 2048, x_ptr),
        dotRowBF16(coded_u16 + 1 * 2048, x_ptr),
        dotRowBF16(coded_u16 + 2 * 2048, x_ptr),
        dotRowBF16(coded_u16 + 3 * 2048, x_ptr),
    };
}

pub inline fn gemvTileBF16TileDirect(coded_u16: [*]const u16, x: []const f32, y: []f32, row_base: usize) void {
    const vals = gemvQuadRowBF16Values(coded_u16, x);
    y[row_base + 0] += vals[0];
    y[row_base + 1] += vals[1];
    y[row_base + 2] += vals[2];
    y[row_base + 3] += vals[3];
}

pub inline fn gemvTileBF16CellDirect(cell: *const geometry.Cell, x: []const f32, y: []f32, row_base: usize) void {
    const coded_u16: [*]const u16 = @ptrCast(@alignCast(&cell.fingerprints));
    gemvTileBF16TileDirect(coded_u16, x, y, row_base);
}

pub inline fn gemvOctaTileBF16TileDirect(coded0: [*]const u16, coded1: [*]const u16, x: []const f32, y: []f32, row_base: usize) void {
    const vals = gemvOctaRowBF16Values(coded0, coded1, x);
    y[row_base + 0] += vals[0];
    y[row_base + 1] += vals[1];
    y[row_base + 2] += vals[2];
    y[row_base + 3] += vals[3];
    y[row_base + 4] += vals[4];
    y[row_base + 5] += vals[5];
    y[row_base + 6] += vals[6];
    y[row_base + 7] += vals[7];
}

pub inline fn gemvOctaRowGateUpBF16(
    g_u16: [*]const u16,
    u_u16: [*]const u16,
    x: []const f32,
) @Vector(4, f32) {
    const gr0 = g_u16 + 0 * 2048;
    const gr1 = g_u16 + 1 * 2048;
    const gr2 = g_u16 + 2 * 2048;
    const gr3 = g_u16 + 3 * 2048;

    const ur0 = u_u16 + 0 * 2048;
    const ur1 = u_u16 + 1 * 2048;
    const ur2 = u_u16 + 2 * 2048;
    const ur3 = u_u16 + 3 * 2048;

    var g0a: @Vector(4, f32) = @splat(0.0);
    var g0b: @Vector(4, f32) = @splat(0.0);
    var g1a: @Vector(4, f32) = @splat(0.0);
    var g1b: @Vector(4, f32) = @splat(0.0);
    var g2a: @Vector(4, f32) = @splat(0.0);
    var g2b: @Vector(4, f32) = @splat(0.0);
    var g3a: @Vector(4, f32) = @splat(0.0);
    var g3b: @Vector(4, f32) = @splat(0.0);

    var u0a: @Vector(4, f32) = @splat(0.0);
    var u0b: @Vector(4, f32) = @splat(0.0);
    var u1a: @Vector(4, f32) = @splat(0.0);
    var u1b: @Vector(4, f32) = @splat(0.0);
    var u2a: @Vector(4, f32) = @splat(0.0);
    var u2b: @Vector(4, f32) = @splat(0.0);
    var u3a: @Vector(4, f32) = @splat(0.0);
    var u3b: @Vector(4, f32) = @splat(0.0);

    var c: usize = 0;
    while (c + 16 <= 2048) : (c += 16) {
        if (c + 64 < 2048) {
            @prefetch(&gr0[c + 64], .{ .locality = 3, .cache = .data, .rw = .read });
            @prefetch(&gr1[c + 64], .{ .locality = 3, .cache = .data, .rw = .read });
            @prefetch(&gr2[c + 64], .{ .locality = 3, .cache = .data, .rw = .read });
            @prefetch(&gr3[c + 64], .{ .locality = 3, .cache = .data, .rw = .read });
            @prefetch(&ur0[c + 64], .{ .locality = 3, .cache = .data, .rw = .read });
            @prefetch(&ur1[c + 64], .{ .locality = 3, .cache = .data, .rw = .read });
            @prefetch(&ur2[c + 64], .{ .locality = 3, .cache = .data, .rw = .read });
            @prefetch(&ur3[c + 64], .{ .locality = 3, .cache = .data, .rw = .read });
        }

        const xa: @Vector(4, f32) = x[c..][0..4].*;
        const xb: @Vector(4, f32) = x[c + 4 ..][0..4].*;
        const xc: @Vector(4, f32) = x[c + 8 ..][0..4].*;
        const xd: @Vector(4, f32) = x[c + 12 ..][0..4].*;

        // Gate Row 0
        const gw0a: @Vector(4, u16) = gr0[c..][0..4].*;
        const gw0b: @Vector(4, u16) = gr0[c + 4 ..][0..4].*;
        const gw0c: @Vector(4, u16) = gr0[c + 8 ..][0..4].*;
        const gw0d: @Vector(4, u16) = gr0[c + 12 ..][0..4].*;
        const fgw0a: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), gw0a) << @splat(16));
        const fgw0b: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), gw0b) << @splat(16));
        const fgw0c: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), gw0c) << @splat(16));
        const fgw0d: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), gw0d) << @splat(16));
        g0a = @mulAdd(@Vector(4, f32), fgw0a, xa, g0a);
        g0b = @mulAdd(@Vector(4, f32), fgw0b, xb, g0b);
        g0a = @mulAdd(@Vector(4, f32), fgw0c, xc, g0a);
        g0b = @mulAdd(@Vector(4, f32), fgw0d, xd, g0b);

        // Up Row 0
        const uw0a: @Vector(4, u16) = ur0[c..][0..4].*;
        const uw0b: @Vector(4, u16) = ur0[c + 4 ..][0..4].*;
        const uw0c: @Vector(4, u16) = ur0[c + 8 ..][0..4].*;
        const uw0d: @Vector(4, u16) = ur0[c + 12 ..][0..4].*;
        const fuw0a: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), uw0a) << @splat(16));
        const fuw0b: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), uw0b) << @splat(16));
        const fuw0c: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), uw0c) << @splat(16));
        const fuw0d: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), uw0d) << @splat(16));
        u0a = @mulAdd(@Vector(4, f32), fuw0a, xa, u0a);
        u0b = @mulAdd(@Vector(4, f32), fuw0b, xb, u0b);
        u0a = @mulAdd(@Vector(4, f32), fuw0c, xc, u0a);
        u0b = @mulAdd(@Vector(4, f32), fuw0d, xd, u0b);

        // Gate Row 1
        const gw1a: @Vector(4, u16) = gr1[c..][0..4].*;
        const gw1b: @Vector(4, u16) = gr1[c + 4 ..][0..4].*;
        const gw1c: @Vector(4, u16) = gr1[c + 8 ..][0..4].*;
        const gw1d: @Vector(4, u16) = gr1[c + 12 ..][0..4].*;
        const fgw1a: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), gw1a) << @splat(16));
        const fgw1b: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), gw1b) << @splat(16));
        const fgw1c: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), gw1c) << @splat(16));
        const fgw1d: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), gw1d) << @splat(16));
        g1a = @mulAdd(@Vector(4, f32), fgw1a, xa, g1a);
        g1b = @mulAdd(@Vector(4, f32), fgw1b, xb, g1b);
        g1a = @mulAdd(@Vector(4, f32), fgw1c, xc, g1a);
        g1b = @mulAdd(@Vector(4, f32), fgw1d, xd, g1b);

        // Up Row 1
        const uw1a: @Vector(4, u16) = ur1[c..][0..4].*;
        const uw1b: @Vector(4, u16) = ur1[c + 4 ..][0..4].*;
        const uw1c: @Vector(4, u16) = ur1[c + 8 ..][0..4].*;
        const uw1d: @Vector(4, u16) = ur1[c + 12 ..][0..4].*;
        const fuw1a: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), uw1a) << @splat(16));
        const fuw1b: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), uw1b) << @splat(16));
        const fuw1c: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), uw1c) << @splat(16));
        const fuw1d: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), uw1d) << @splat(16));
        u1a = @mulAdd(@Vector(4, f32), fuw1a, xa, u1a);
        u1b = @mulAdd(@Vector(4, f32), fuw1b, xb, u1b);
        u1a = @mulAdd(@Vector(4, f32), fuw1c, xc, u1a);
        u1b = @mulAdd(@Vector(4, f32), fuw1d, xd, u1b);

        // Gate Row 2
        const gw2a: @Vector(4, u16) = gr2[c..][0..4].*;
        const gw2b: @Vector(4, u16) = gr2[c + 4 ..][0..4].*;
        const gw2c: @Vector(4, u16) = gr2[c + 8 ..][0..4].*;
        const gw2d: @Vector(4, u16) = gr2[c + 12 ..][0..4].*;
        const fgw2a: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), gw2a) << @splat(16));
        const fgw2b: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), gw2b) << @splat(16));
        const fgw2c: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), gw2c) << @splat(16));
        const fgw2d: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), gw2d) << @splat(16));
        g2a = @mulAdd(@Vector(4, f32), fgw2a, xa, g2a);
        g2b = @mulAdd(@Vector(4, f32), fgw2b, xb, g2b);
        g2a = @mulAdd(@Vector(4, f32), fgw2c, xc, g2a);
        g2b = @mulAdd(@Vector(4, f32), fgw2d, xd, g2b);

        // Up Row 2
        const uw2a: @Vector(4, u16) = ur2[c..][0..4].*;
        const uw2b: @Vector(4, u16) = ur2[c + 4 ..][0..4].*;
        const uw2c: @Vector(4, u16) = ur2[c + 8 ..][0..4].*;
        const uw2d: @Vector(4, u16) = ur2[c + 12 ..][0..4].*;
        const fuw2a: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), uw2a) << @splat(16));
        const fuw2b: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), uw2b) << @splat(16));
        const fuw2c: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), uw2c) << @splat(16));
        const fuw2d: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), uw2d) << @splat(16));
        u2a = @mulAdd(@Vector(4, f32), fuw2a, xa, u2a);
        u2b = @mulAdd(@Vector(4, f32), fuw2b, xb, u2b);
        u2a = @mulAdd(@Vector(4, f32), fuw2c, xc, u2a);
        u2b = @mulAdd(@Vector(4, f32), fuw2d, xd, u2b);

        // Gate Row 3
        const gw3a: @Vector(4, u16) = gr3[c..][0..4].*;
        const gw3b: @Vector(4, u16) = gr3[c + 4 ..][0..4].*;
        const gw3c: @Vector(4, u16) = gr3[c + 8 ..][0..4].*;
        const gw3d: @Vector(4, u16) = gr3[c + 12 ..][0..4].*;
        const fgw3a: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), gw3a) << @splat(16));
        const fgw3b: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), gw3b) << @splat(16));
        const fgw3c: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), gw3c) << @splat(16));
        const fgw3d: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), gw3d) << @splat(16));
        g3a = @mulAdd(@Vector(4, f32), fgw3a, xa, g3a);
        g3b = @mulAdd(@Vector(4, f32), fgw3b, xb, g3b);
        g3a = @mulAdd(@Vector(4, f32), fgw3c, xc, g3a);
        g3b = @mulAdd(@Vector(4, f32), fgw3d, xd, g3b);

        // Up Row 3
        const uw3a: @Vector(4, u16) = ur3[c..][0..4].*;
        const uw3b: @Vector(4, u16) = ur3[c + 4 ..][0..4].*;
        const uw3c: @Vector(4, u16) = ur3[c + 8 ..][0..4].*;
        const uw3d: @Vector(4, u16) = ur3[c + 12 ..][0..4].*;
        const fuw3a: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), uw3a) << @splat(16));
        const fuw3b: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), uw3b) << @splat(16));
        const fuw3c: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), uw3c) << @splat(16));
        const fuw3d: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), uw3d) << @splat(16));
        u3a = @mulAdd(@Vector(4, f32), fuw3a, xa, u3a);
        u3b = @mulAdd(@Vector(4, f32), fuw3b, xb, u3b);
        u3a = @mulAdd(@Vector(4, f32), fuw3c, xc, u3a);
        u3b = @mulAdd(@Vector(4, f32), fuw3d, xd, u3b);
    }

    const g_vals = @Vector(4, f32){
        @reduce(.Add, g0a + g0b),
        @reduce(.Add, g1a + g1b),
        @reduce(.Add, g2a + g2b),
        @reduce(.Add, g3a + g3b),
    };
    const u_vals = @Vector(4, f32){
        @reduce(.Add, u0a + u0b),
        @reduce(.Add, u1a + u1b),
        @reduce(.Add, u2a + u2b),
        @reduce(.Add, u3a + u3b),
    };

    return mlp.siluVec4(g_vals) * u_vals;
}

pub inline fn gemvOctaRowBF16Values(
    coded0: [*]const u16,
    coded1: [*]const u16,
    x: []const f32,
) [8]f32 {
    const r0 = coded0 + 0 * 2048;
    const r1 = coded0 + 1 * 2048;
    const r2 = coded0 + 2 * 2048;
    const r3 = coded0 + 3 * 2048;
    const r4 = coded1 + 0 * 2048;
    const r5 = coded1 + 1 * 2048;
    const r6 = coded1 + 2 * 2048;
    const r7 = coded1 + 3 * 2048;

    var d0a: @Vector(4, f32) = @splat(0.0);
    var d0b: @Vector(4, f32) = @splat(0.0);
    var d1a: @Vector(4, f32) = @splat(0.0);
    var d1b: @Vector(4, f32) = @splat(0.0);
    var d2a: @Vector(4, f32) = @splat(0.0);
    var d2b: @Vector(4, f32) = @splat(0.0);
    var d3a: @Vector(4, f32) = @splat(0.0);
    var d3b: @Vector(4, f32) = @splat(0.0);
    var d4a: @Vector(4, f32) = @splat(0.0);
    var d4b: @Vector(4, f32) = @splat(0.0);
    var d5a: @Vector(4, f32) = @splat(0.0);
    var d5b: @Vector(4, f32) = @splat(0.0);
    var d6a: @Vector(4, f32) = @splat(0.0);
    var d6b: @Vector(4, f32) = @splat(0.0);
    var d7a: @Vector(4, f32) = @splat(0.0);
    var d7b: @Vector(4, f32) = @splat(0.0);

    var c: usize = 0;
    while (c + 16 <= 2048) : (c += 16) {
        if (c + 64 < 2048) {
            @prefetch(&r0[c + 64], .{ .locality = 3, .cache = .data, .rw = .read });
            @prefetch(&r1[c + 64], .{ .locality = 3, .cache = .data, .rw = .read });
            @prefetch(&r2[c + 64], .{ .locality = 3, .cache = .data, .rw = .read });
            @prefetch(&r3[c + 64], .{ .locality = 3, .cache = .data, .rw = .read });
            @prefetch(&r4[c + 64], .{ .locality = 3, .cache = .data, .rw = .read });
            @prefetch(&r5[c + 64], .{ .locality = 3, .cache = .data, .rw = .read });
            @prefetch(&r6[c + 64], .{ .locality = 3, .cache = .data, .rw = .read });
            @prefetch(&r7[c + 64], .{ .locality = 3, .cache = .data, .rw = .read });
        }

        const xa: @Vector(4, f32) = x[c..][0..4].*;
        const xb: @Vector(4, f32) = x[c + 4 ..][0..4].*;
        const xc: @Vector(4, f32) = x[c + 8 ..][0..4].*;
        const xd: @Vector(4, f32) = x[c + 12 ..][0..4].*;

        // Row 0
        const w0a: @Vector(4, u16) = r0[c..][0..4].*;
        const w0b: @Vector(4, u16) = r0[c + 4 ..][0..4].*;
        const w0c: @Vector(4, u16) = r0[c + 8 ..][0..4].*;
        const w0d: @Vector(4, u16) = r0[c + 12 ..][0..4].*;
        const fw0a: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0a) << @splat(16));
        const fw0b: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0b) << @splat(16));
        const fw0c: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0c) << @splat(16));
        const fw0d: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w0d) << @splat(16));
        d0a = @mulAdd(@Vector(4, f32), fw0a, xa, d0a);
        d0b = @mulAdd(@Vector(4, f32), fw0b, xb, d0b);
        d0a = @mulAdd(@Vector(4, f32), fw0c, xc, d0a);
        d0b = @mulAdd(@Vector(4, f32), fw0d, xd, d0b);

        // Row 1
        const w1a: @Vector(4, u16) = r1[c..][0..4].*;
        const w1b: @Vector(4, u16) = r1[c + 4 ..][0..4].*;
        const w1c: @Vector(4, u16) = r1[c + 8 ..][0..4].*;
        const w1d: @Vector(4, u16) = r1[c + 12 ..][0..4].*;
        const fw1a: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1a) << @splat(16));
        const fw1b: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1b) << @splat(16));
        const fw1c: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1c) << @splat(16));
        const fw1d: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w1d) << @splat(16));
        d1a = @mulAdd(@Vector(4, f32), fw1a, xa, d1a);
        d1b = @mulAdd(@Vector(4, f32), fw1b, xb, d1b);
        d1a = @mulAdd(@Vector(4, f32), fw1c, xc, d1a);
        d1b = @mulAdd(@Vector(4, f32), fw1d, xd, d1b);

        // Row 2
        const w2a: @Vector(4, u16) = r2[c..][0..4].*;
        const w2b: @Vector(4, u16) = r2[c + 4 ..][0..4].*;
        const w2c: @Vector(4, u16) = r2[c + 8 ..][0..4].*;
        const w2d: @Vector(4, u16) = r2[c + 12 ..][0..4].*;
        const fw2a: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w2a) << @splat(16));
        const fw2b: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w2b) << @splat(16));
        const fw2c: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w2c) << @splat(16));
        const fw2d: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w2d) << @splat(16));
        d2a = @mulAdd(@Vector(4, f32), fw2a, xa, d2a);
        d2b = @mulAdd(@Vector(4, f32), fw2b, xb, d2b);
        d2a = @mulAdd(@Vector(4, f32), fw2c, xc, d2a);
        d2b = @mulAdd(@Vector(4, f32), fw2d, xd, d2b);

        // Row 3
        const w3a: @Vector(4, u16) = r3[c..][0..4].*;
        const w3b: @Vector(4, u16) = r3[c + 4 ..][0..4].*;
        const w3c: @Vector(4, u16) = r3[c + 8 ..][0..4].*;
        const w3d: @Vector(4, u16) = r3[c + 12 ..][0..4].*;
        const fw3a: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w3a) << @splat(16));
        const fw3b: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w3b) << @splat(16));
        const fw3c: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w3c) << @splat(16));
        const fw3d: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w3d) << @splat(16));
        d3a = @mulAdd(@Vector(4, f32), fw3a, xa, d3a);
        d3b = @mulAdd(@Vector(4, f32), fw3b, xb, d3b);
        d3a = @mulAdd(@Vector(4, f32), fw3c, xc, d3a);
        d3b = @mulAdd(@Vector(4, f32), fw3d, xd, d3b);

        // Row 4
        const w4a: @Vector(4, u16) = r4[c..][0..4].*;
        const w4b: @Vector(4, u16) = r4[c + 4 ..][0..4].*;
        const w4c: @Vector(4, u16) = r4[c + 8 ..][0..4].*;
        const w4d: @Vector(4, u16) = r4[c + 12 ..][0..4].*;
        const fw4a: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w4a) << @splat(16));
        const fw4b: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w4b) << @splat(16));
        const fw4c: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w4c) << @splat(16));
        const fw4d: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w4d) << @splat(16));
        d4a = @mulAdd(@Vector(4, f32), fw4a, xa, d4a);
        d4b = @mulAdd(@Vector(4, f32), fw4b, xb, d4b);
        d4a = @mulAdd(@Vector(4, f32), fw4c, xc, d4a);
        d4b = @mulAdd(@Vector(4, f32), fw4d, xd, d4b);

        // Row 5
        const w5a: @Vector(4, u16) = r5[c..][0..4].*;
        const w5b: @Vector(4, u16) = r5[c + 4 ..][0..4].*;
        const w5c: @Vector(4, u16) = r5[c + 8 ..][0..4].*;
        const w5d: @Vector(4, u16) = r5[c + 12 ..][0..4].*;
        const fw5a: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w5a) << @splat(16));
        const fw5b: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w5b) << @splat(16));
        const fw5c: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w5c) << @splat(16));
        const fw5d: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w5d) << @splat(16));
        d5a = @mulAdd(@Vector(4, f32), fw5a, xa, d5a);
        d5b = @mulAdd(@Vector(4, f32), fw5b, xb, d5b);
        d5a = @mulAdd(@Vector(4, f32), fw5c, xc, d5a);
        d5b = @mulAdd(@Vector(4, f32), fw5d, xd, d5b);

        // Row 6
        const w6a: @Vector(4, u16) = r6[c..][0..4].*;
        const w6b: @Vector(4, u16) = r6[c + 4 ..][0..4].*;
        const w6c: @Vector(4, u16) = r6[c + 8 ..][0..4].*;
        const w6d: @Vector(4, u16) = r6[c + 12 ..][0..4].*;
        const fw6a: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w6a) << @splat(16));
        const fw6b: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w6b) << @splat(16));
        const fw6c: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w6c) << @splat(16));
        const fw6d: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w6d) << @splat(16));
        d6a = @mulAdd(@Vector(4, f32), fw6a, xa, d6a);
        d6b = @mulAdd(@Vector(4, f32), fw6b, xb, d6b);
        d6a = @mulAdd(@Vector(4, f32), fw6c, xc, d6a);
        d6b = @mulAdd(@Vector(4, f32), fw6d, xd, d6b);

        // Row 7
        const w7a: @Vector(4, u16) = r7[c..][0..4].*;
        const w7b: @Vector(4, u16) = r7[c + 4 ..][0..4].*;
        const w7c: @Vector(4, u16) = r7[c + 8 ..][0..4].*;
        const w7d: @Vector(4, u16) = r7[c + 12 ..][0..4].*;
        const fw7a: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w7a) << @splat(16));
        const fw7b: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w7b) << @splat(16));
        const fw7c: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w7c) << @splat(16));
        const fw7d: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), w7d) << @splat(16));
        d7a = @mulAdd(@Vector(4, f32), fw7a, xa, d7a);
        d7b = @mulAdd(@Vector(4, f32), fw7b, xb, d7b);
        d7a = @mulAdd(@Vector(4, f32), fw7c, xc, d7a);
        d7b = @mulAdd(@Vector(4, f32), fw7d, xd, d7b);
    }

    return .{
        @reduce(.Add, d0a + d0b),
        @reduce(.Add, d1a + d1b),
        @reduce(.Add, d2a + d2b),
        @reduce(.Add, d3a + d3b),
        @reduce(.Add, d4a + d4b),
        @reduce(.Add, d5a + d5b),
        @reduce(.Add, d6a + d6b),
        @reduce(.Add, d7a + d7b),
    };
}

pub fn gemvTileBF16(rec: *const geometry.Record, x: []const f32, y: []f32, row_base: usize) void {
    gemvTileBF16CellDirect(&rec.cell, x, y, row_base);
}

pub inline fn gemvTileDownBF16TileDirect(coded_u16: [*]const u16, x: []const f32, y: []f32, tile_idx: usize) void {
    const global_k_start: usize = tile_idx * 8192;
    var r: usize = global_k_start / INTERMEDIATE_DIM;
    var c: usize = global_k_start % INTERMEDIATE_DIM;
    var j: usize = 0;

    while (j < 8192) {
        const seg_len = @min(8192 - j, INTERMEDIATE_DIM - c);
        var dot_vec_a0: @Vector(4, f32) = @splat(0.0);
        var dot_vec_a1: @Vector(4, f32) = @splat(0.0);
        var dot_vec_a2: @Vector(4, f32) = @splat(0.0);
        var dot_vec_a3: @Vector(4, f32) = @splat(0.0);
        var dot_vec_a4: @Vector(4, f32) = @splat(0.0);
        var dot_vec_a5: @Vector(4, f32) = @splat(0.0);
        var dot_vec_a6: @Vector(4, f32) = @splat(0.0);
        var dot_vec_a7: @Vector(4, f32) = @splat(0.0);
        var dot_vec_b0: @Vector(4, f32) = @splat(0.0);
        var dot_vec_b1: @Vector(4, f32) = @splat(0.0);
        var dot_vec_b2: @Vector(4, f32) = @splat(0.0);
        var dot_vec_b3: @Vector(4, f32) = @splat(0.0);
        var dot_vec_b4: @Vector(4, f32) = @splat(0.0);
        var dot_vec_b5: @Vector(4, f32) = @splat(0.0);
        var dot_vec_b6: @Vector(4, f32) = @splat(0.0);
        var dot_vec_b7: @Vector(4, f32) = @splat(0.0);
        var k: usize = 0;
        while (k + 64 <= seg_len) : (k += 64) {
            if (k + 128 < seg_len) {
                @prefetch(&coded_u16[j + k + 128], .{ .locality = 3, .cache = .data, .rw = .read });
            }
            const u16_a0: @Vector(4, u16) = coded_u16[j + k ..][0..4].*;
            const u16_a1: @Vector(4, u16) = coded_u16[j + k + 4 ..][0..4].*;
            const u16_a2: @Vector(4, u16) = coded_u16[j + k + 8 ..][0..4].*;
            const u16_a3: @Vector(4, u16) = coded_u16[j + k + 12 ..][0..4].*;
            const u16_a4: @Vector(4, u16) = coded_u16[j + k + 16 ..][0..4].*;
            const u16_a5: @Vector(4, u16) = coded_u16[j + k + 20 ..][0..4].*;
            const u16_a6: @Vector(4, u16) = coded_u16[j + k + 24 ..][0..4].*;
            const u16_a7: @Vector(4, u16) = coded_u16[j + k + 28 ..][0..4].*;
            const u16_b0: @Vector(4, u16) = coded_u16[j + k + 32 ..][0..4].*;
            const u16_b1: @Vector(4, u16) = coded_u16[j + k + 36 ..][0..4].*;
            const u16_b2: @Vector(4, u16) = coded_u16[j + k + 40 ..][0..4].*;
            const u16_b3: @Vector(4, u16) = coded_u16[j + k + 44 ..][0..4].*;
            const u16_b4: @Vector(4, u16) = coded_u16[j + k + 48 ..][0..4].*;
            const u16_b5: @Vector(4, u16) = coded_u16[j + k + 52 ..][0..4].*;
            const u16_b6: @Vector(4, u16) = coded_u16[j + k + 56 ..][0..4].*;
            const u16_b7: @Vector(4, u16) = coded_u16[j + k + 60 ..][0..4].*;

            const f32_wa0: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), u16_a0) << @splat(16));
            const f32_wa1: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), u16_a1) << @splat(16));
            const f32_wa2: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), u16_a2) << @splat(16));
            const f32_wa3: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), u16_a3) << @splat(16));
            const f32_wa4: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), u16_a4) << @splat(16));
            const f32_wa5: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), u16_a5) << @splat(16));
            const f32_wa6: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), u16_a6) << @splat(16));
            const f32_wa7: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), u16_a7) << @splat(16));
            const f32_wb0: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), u16_b0) << @splat(16));
            const f32_wb1: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), u16_b1) << @splat(16));
            const f32_wb2: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), u16_b2) << @splat(16));
            const f32_wb3: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), u16_b3) << @splat(16));
            const f32_wb4: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), u16_b4) << @splat(16));
            const f32_wb5: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), u16_b5) << @splat(16));
            const f32_wb6: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), u16_b6) << @splat(16));
            const f32_wb7: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), u16_b7) << @splat(16));

            const f32_xa0: @Vector(4, f32) = x[c + k ..][0..4].*;
            const f32_xa1: @Vector(4, f32) = x[c + k + 4 ..][0..4].*;
            const f32_xa2: @Vector(4, f32) = x[c + k + 8 ..][0..4].*;
            const f32_xa3: @Vector(4, f32) = x[c + k + 12 ..][0..4].*;
            const f32_xa4: @Vector(4, f32) = x[c + k + 16 ..][0..4].*;
            const f32_xa5: @Vector(4, f32) = x[c + k + 20 ..][0..4].*;
            const f32_xa6: @Vector(4, f32) = x[c + k + 24 ..][0..4].*;
            const f32_xa7: @Vector(4, f32) = x[c + k + 28 ..][0..4].*;
            const f32_xb0: @Vector(4, f32) = x[c + k + 32 ..][0..4].*;
            const f32_xb1: @Vector(4, f32) = x[c + k + 36 ..][0..4].*;
            const f32_xb2: @Vector(4, f32) = x[c + k + 40 ..][0..4].*;
            const f32_xb3: @Vector(4, f32) = x[c + k + 44 ..][0..4].*;
            const f32_xb4: @Vector(4, f32) = x[c + k + 48 ..][0..4].*;
            const f32_xb5: @Vector(4, f32) = x[c + k + 52 ..][0..4].*;
            const f32_xb6: @Vector(4, f32) = x[c + k + 56 ..][0..4].*;
            const f32_xb7: @Vector(4, f32) = x[c + k + 60 ..][0..4].*;

            dot_vec_a0 = @mulAdd(@Vector(4, f32), f32_wa0, f32_xa0, dot_vec_a0);
            dot_vec_a1 = @mulAdd(@Vector(4, f32), f32_wa1, f32_xa1, dot_vec_a1);
            dot_vec_a2 = @mulAdd(@Vector(4, f32), f32_wa2, f32_xa2, dot_vec_a2);
            dot_vec_a3 = @mulAdd(@Vector(4, f32), f32_wa3, f32_xa3, dot_vec_a3);
            dot_vec_a4 = @mulAdd(@Vector(4, f32), f32_wa4, f32_xa4, dot_vec_a4);
            dot_vec_a5 = @mulAdd(@Vector(4, f32), f32_wa5, f32_xa5, dot_vec_a5);
            dot_vec_a6 = @mulAdd(@Vector(4, f32), f32_wa6, f32_xa6, dot_vec_a6);
            dot_vec_a7 = @mulAdd(@Vector(4, f32), f32_wa7, f32_xa7, dot_vec_a7);
            dot_vec_b0 = @mulAdd(@Vector(4, f32), f32_wb0, f32_xb0, dot_vec_b0);
            dot_vec_b1 = @mulAdd(@Vector(4, f32), f32_wb1, f32_xb1, dot_vec_b1);
            dot_vec_b2 = @mulAdd(@Vector(4, f32), f32_wb2, f32_xb2, dot_vec_b2);
            dot_vec_b3 = @mulAdd(@Vector(4, f32), f32_wb3, f32_xb3, dot_vec_b3);
            dot_vec_b4 = @mulAdd(@Vector(4, f32), f32_wb4, f32_xb4, dot_vec_b4);
            dot_vec_b5 = @mulAdd(@Vector(4, f32), f32_wb5, f32_xb5, dot_vec_b5);
            dot_vec_b6 = @mulAdd(@Vector(4, f32), f32_wb6, f32_xb6, dot_vec_b6);
            dot_vec_b7 = @mulAdd(@Vector(4, f32), f32_wb7, f32_xb7, dot_vec_b7);
        }
        while (k + 16 <= seg_len) : (k += 16) {
            const u16_a0: @Vector(4, u16) = coded_u16[j + k ..][0..4].*;
            const u16_a1: @Vector(4, u16) = coded_u16[j + k + 4 ..][0..4].*;
            const u16_a2: @Vector(4, u16) = coded_u16[j + k + 8 ..][0..4].*;
            const u16_a3: @Vector(4, u16) = coded_u16[j + k + 12 ..][0..4].*;
            const f32_wa0: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), u16_a0) << @splat(16));
            const f32_wa1: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), u16_a1) << @splat(16));
            const f32_wa2: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), u16_a2) << @splat(16));
            const f32_wa3: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), u16_a3) << @splat(16));
            const f32_xa0: @Vector(4, f32) = x[c + k ..][0..4].*;
            const f32_xa1: @Vector(4, f32) = x[c + k + 4 ..][0..4].*;
            const f32_xa2: @Vector(4, f32) = x[c + k + 8 ..][0..4].*;
            const f32_xa3: @Vector(4, f32) = x[c + k + 12 ..][0..4].*;
            dot_vec_a0 = @mulAdd(@Vector(4, f32), f32_wa0, f32_xa0, dot_vec_a0);
            dot_vec_a1 = @mulAdd(@Vector(4, f32), f32_wa1, f32_xa1, dot_vec_a1);
            dot_vec_a2 = @mulAdd(@Vector(4, f32), f32_wa2, f32_xa2, dot_vec_a2);
            dot_vec_a3 = @mulAdd(@Vector(4, f32), f32_wa3, f32_xa3, dot_vec_a3);
        }
        while (k + 4 <= seg_len) : (k += 4) {
            const u16_val: @Vector(4, u16) = coded_u16[j + k ..][0..4].*;
            const f32_w: @Vector(4, f32) = @bitCast(@as(@Vector(4, u32), u16_val) << @splat(16));
            const f32_x_val: @Vector(4, f32) = x[c + k ..][0..4].*;
            dot_vec_a0 = @mulAdd(@Vector(4, f32), f32_w, f32_x_val, dot_vec_a0);
        }
        const sum_a = (dot_vec_a0 + dot_vec_a1) + (dot_vec_a2 + dot_vec_a3) + (dot_vec_a4 + dot_vec_a5) + (dot_vec_a6 + dot_vec_a7);
        const sum_b = (dot_vec_b0 + dot_vec_b1) + (dot_vec_b2 + dot_vec_b3) + (dot_vec_b4 + dot_vec_b5) + (dot_vec_b6 + dot_vec_b7);
        var row_sum = @reduce(.Add, sum_a + sum_b);
        while (k < seg_len) : (k += 1) {
            const u16_val = coded_u16[j + k];
            const f32_w: f32 = @as(f32, @bitCast(@as(u32, u16_val) << 16));
            row_sum += f32_w * x[c + k];
        }
        y[r] += row_sum;
        j += seg_len;
        c += seg_len;
        if (c >= INTERMEDIATE_DIM) {
            c = 0;
            r += 1;
        }
    }
}

pub inline fn gemvTileDownBF16CellDirect(cell: *const geometry.Cell, x: []const f32, y: []f32, tile_idx: usize) void {
    const coded_u16: [*]const u16 = @ptrCast(@alignCast(&cell.fingerprints));
    gemvTileDownBF16TileDirect(coded_u16, x, y, tile_idx);
}

pub fn gemvTileDownBF16(rec: *const geometry.Record, x: []const f32, y: []f32, tile_idx: usize) void {
    gemvTileDownBF16CellDirect(&rec.cell, x, y, tile_idx);
}

pub fn computeGroupSums2048(x: []const f32, out_sums: *[16]f32) void {
    for (0..16) |g| {
        var s0: @Vector(8, f32) = @splat(0.0);
        var s1: @Vector(8, f32) = @splat(0.0);
        const xg = x[g * 128 .. (g + 1) * 128];
        var j: usize = 0;
        while (j < 128) : (j += 16) {
            s0 += xg[j..][0..8].*;
            s1 += xg[j + 8 ..][0..8].*;
        }
        out_sums[g] = @reduce(.Add, s0 + s1);
    }
}

pub fn computeGroupSums11008(x: []const f32, out_sums: *[86]f32) void {
    for (0..86) |g| {
        var s0: @Vector(8, f32) = @splat(0.0);
        var s1: @Vector(8, f32) = @splat(0.0);
        const xg = x[g * 128 .. (g + 1) * 128];
        var j: usize = 0;
        while (j < 128) : (j += 16) {
            s0 += xg[j..][0..8].*;
            s1 += xg[j + 8 ..][0..8].*;
        }
        out_sums[g] = @reduce(.Add, s0 + s1);
    }
}

pub const Activation2048 = struct {
    raw: []const f32,
    even: [HIDDEN_DIM / 2]f32,
    odd: [HIDDEN_DIM / 2]f32,
    q_even: [HIDDEN_DIM / 2]i8,
    q_odd: [HIDDEN_DIM / 2]i8,
    scales: [16]f32,
    sums: [16]f32,
    q_sums: [16]i32,

    pub fn initInto(act: *Activation2048, x: []const f32) void {
        act.raw = x;
        const min_v: @Vector(4, i32) = @splat(-128);
        const max_v: @Vector(4, i32) = @splat(127);

        for (0..16) |g| {
            const xg = x[g * 128 .. (g + 1) * 128];
            const eg = act.even[g * 64 .. (g + 1) * 64];
            const og = act.odd[g * 64 .. (g + 1) * 64];
            const qe = act.q_even[g * 64 .. (g + 1) * 64];
            const qo = act.q_odd[g * 64 .. (g + 1) * 64];

            var s0: @Vector(4, f32) = @splat(0.0);
            var s1: @Vector(4, f32) = @splat(0.0);
            var max_vec: @Vector(4, f32) = @splat(0.0);

            var j: usize = 0;
            while (j < 64) : (j += 4) {
                const x8: @Vector(8, f32) = xg[2 * j ..][0..8].*;
                const evens: @Vector(4, f32) = @shuffle(f32, x8, undefined, [4]i32{ 0, 2, 4, 6 });
                const odds: @Vector(4, f32) = @shuffle(f32, x8, undefined, [4]i32{ 1, 3, 5, 7 });
                eg[j..][0..4].* = evens;
                og[j..][0..4].* = odds;
                max_vec = @max(max_vec, @max(@abs(evens), @abs(odds)));
                s0 += evens;
                s1 += odds;
            }
            act.sums[g] = @reduce(.Add, s0 + s1);
            const max_abs = @reduce(.Max, max_vec);

            const scale = if (max_abs > 1e-10) max_abs / 127.0 else 0.0;
            const inv_scale = if (max_abs > 1e-10) 127.0 / max_abs else 0.0;
            act.scales[g] = scale;
            const inv_scale_vec: @Vector(4, f32) = @splat(inv_scale);

            var q_sum_acc: @Vector(4, i32) = @splat(0);
            var k: usize = 0;
            while (k < 64) : (k += 4) {
                const ef: @Vector(4, f32) = eg[k..][0..4].*;
                const of: @Vector(4, f32) = og[k..][0..4].*;
                const qi_e: @Vector(4, i32) = @min(@max(@as(@Vector(4, i32), @intFromFloat(@round(ef * inv_scale_vec))), min_v), max_v);
                const qi_o: @Vector(4, i32) = @min(@max(@as(@Vector(4, i32), @intFromFloat(@round(of * inv_scale_vec))), min_v), max_v);
                const q8_e: @Vector(4, i8) = @intCast(qi_e);
                const q8_o: @Vector(4, i8) = @intCast(qi_o);
                qe[k..][0..4].* = q8_e;
                qo[k..][0..4].* = q8_o;
                q_sum_acc += qi_e + qi_o;
            }
            act.q_sums[g] = @reduce(.Add, q_sum_acc);
        }
    }

    pub fn init(x: []const f32) Activation2048 {
        var act: Activation2048 = undefined;
        initInto(&act, x);
        return act;
    }
};

pub const Activation11008 = struct {
    raw: []const f32,
    even: [INTERMEDIATE_DIM / 2]f32,
    odd: [INTERMEDIATE_DIM / 2]f32,
    q_even: [INTERMEDIATE_DIM / 2]i8,
    q_odd: [INTERMEDIATE_DIM / 2]i8,
    scales: [86]f32,
    sums: [86]f32,
    q_sums: [86]i32,

    pub fn quantizeRange(act: *Activation11008, x: []const f32, start_g: usize, end_g: usize) void {
        const min_v: @Vector(4, i32) = @splat(-128);
        const max_v: @Vector(4, i32) = @splat(127);

        for (start_g..end_g) |g| {
            const xg = x[g * 128 .. (g + 1) * 128];
            const eg = act.even[g * 64 .. (g + 1) * 64];
            const og = act.odd[g * 64 .. (g + 1) * 64];
            const qe = act.q_even[g * 64 .. (g + 1) * 64];
            const qo = act.q_odd[g * 64 .. (g + 1) * 64];

            var s0: @Vector(4, f32) = @splat(0.0);
            var s1: @Vector(4, f32) = @splat(0.0);
            var max_abs: f32 = 0.0;

            var j: usize = 0;
            while (j < 64) : (j += 4) {
                inline for (0..4) |k| {
                    const e_val = xg[2 * (j + k)];
                    const o_val = xg[2 * (j + k) + 1];
                    eg[j + k] = e_val;
                    og[j + k] = o_val;
                    const a_e = @abs(e_val);
                    const a_o = @abs(o_val);
                    if (a_e > max_abs) max_abs = a_e;
                    if (a_o > max_abs) max_abs = a_o;
                }
                s0 += eg[j..][0..4].*;
                s1 += og[j..][0..4].*;
            }
            act.sums[g] = @reduce(.Add, s0 + s1);

            const scale = if (max_abs > 1e-10) max_abs / 127.0 else 0.0;
            const inv_scale = if (max_abs > 1e-10) 127.0 / max_abs else 0.0;
            act.scales[g] = scale;
            const inv_scale_vec: @Vector(4, f32) = @splat(inv_scale);

            var q_sum_acc: @Vector(4, i32) = @splat(0);
            var k: usize = 0;
            while (k < 64) : (k += 4) {
                const ef: @Vector(4, f32) = eg[k..][0..4].*;
                const of: @Vector(4, f32) = og[k..][0..4].*;
                const qi_e: @Vector(4, i32) = @min(@max(@as(@Vector(4, i32), @intFromFloat(@round(ef * inv_scale_vec))), min_v), max_v);
                const qi_o: @Vector(4, i32) = @min(@max(@as(@Vector(4, i32), @intFromFloat(@round(of * inv_scale_vec))), min_v), max_v);
                const q8_e: @Vector(4, i8) = @intCast(qi_e);
                const q8_o: @Vector(4, i8) = @intCast(qi_o);
                qe[k..][0..4].* = q8_e;
                qo[k..][0..4].* = q8_o;
                q_sum_acc += qi_e + qi_o;
            }
            act.q_sums[g] = @reduce(.Add, q_sum_acc);
        }
    }

    pub fn initInto(act: *Activation11008, x: []const f32) void {
        act.raw = x;
        quantizeRange(act, x, 0, 86);
    }

    pub fn init(x: []const f32) Activation11008 {
        var act: Activation11008 = undefined;
        initInto(&act, x);
        return act;
    }
};

pub var global_act2048: Activation2048 = undefined;
pub var global_act11008: Activation11008 = undefined;
pub var worker_down_buffers: [4][HIDDEN_DIM]f32 = undefined;

pub fn gemvTileQuadRow(rec: *const geometry.Record, act: *const Activation2048, y: []f32, row_base: usize) void {
    gemvTileQuadRowCellOpt(&rec.cell, act, y, row_base, USE_INT8_SDOT);
}

pub fn gemvTileQuadRowSDOT(rec: *const geometry.Record, act: *const Activation2048, y: []f32, row_base: usize) void {
    gemvTileQuadRowCellOpt(&rec.cell, act, y, row_base, true);
}

pub fn gemvTileQuadRowFP32(rec: *const geometry.Record, act: *const Activation2048, y: []f32, row_base: usize) void {
    gemvTileQuadRowCellOpt(&rec.cell, act, y, row_base, false);
}

pub fn gemvTileQuadRowOpt(rec: *const geometry.Record, act: *const Activation2048, y: []f32, row_base: usize, comptime use_sdot: bool) void {
    gemvTileQuadRowCellOpt(&rec.cell, act, y, row_base, use_sdot);
}

pub fn gemvTileQuadRowDirect(rec: *const geometry.Record, act: *const Activation2048, y: []f32, row_base: usize, comptime accumulate: bool) void {
    gemvTileQuadRowCellDirect(&rec.cell, act, y, row_base, USE_INT8_SDOT, accumulate);
}

pub fn gemvTileQuadRowCellOpt(cell: *const geometry.Cell, act: *const Activation2048, y: []f32, row_base: usize, comptime use_sdot: bool) void {
    gemvTileQuadRowCellDirect(cell, act, y, row_base, use_sdot, true);
}

pub fn gemvTileQuadRowCellDirect(cell: *const geometry.Cell, act: *const Activation2048, y: []f32, row_base: usize, comptime use_sdot: bool, comptime accumulate: bool) void {
    const payload = &cell.semantic_payload;
    const meta: *const weight_archive.TileMetadata = @ptrCast(@alignCast(payload.ptr));

    if (meta.quant_bits == 2) {
        gemvTileW2CellDirect(cell, act, y, row_base, accumulate);
        return;
    }

    if ((meta.custom_flags & weight_archive.FLAG_GROUP128_OUTLIERS) != 0 or meta.quant_bits == 4) {
        const group_scales_raw: [*]const f16 = @ptrCast(@alignCast(payload[48..560].ptr));
        const coded: [*]const u8 = @ptrCast(&cell.fingerprints);
        const has_outliers = (meta.custom_flags & weight_archive.FLAG_GROUP128_OUTLIERS) != 0;

        var x_outlier: @Vector(8, f32) = @splat(0.0);
        var outlier_w_raw: [*]const f16 = undefined;
        if (has_outliers) {
            outlier_w_raw = @ptrCast(@alignCast(payload[560..816].ptr));
            const outlier_cols_raw: [*]const u16 = @ptrCast(@alignCast(payload[816..832].ptr));
            inline for (0..8) |k| {
                const col_idx = outlier_cols_raw[k];
                x_outlier[k] = if ((col_idx & 1) == 0) act.even[col_idx / 2] else act.odd[col_idx / 2];
            }
        }

        const mask0f_16: @Vector(16, u8) = @splat(0x0F);
        const shift4_16: @Vector(16, u3) = @splat(4);

        if (comptime @import("builtin").cpu.arch == .aarch64 and use_sdot) {
            var r: usize = 0;
            while (r < 16) : (r += 8) {
                var row_dots: @Vector(8, f32) = @splat(0.0);
                const r0_bytes = coded[(r + 0) * 1024 .. (r + 1) * 1024];
                const r1_bytes = coded[(r + 1) * 1024 .. (r + 2) * 1024];
                const r2_bytes = coded[(r + 2) * 1024 .. (r + 3) * 1024];
                const r3_bytes = coded[(r + 3) * 1024 .. (r + 4) * 1024];
                const r4_bytes = coded[(r + 4) * 1024 .. (r + 5) * 1024];
                const r5_bytes = coded[(r + 5) * 1024 .. (r + 6) * 1024];
                const r6_bytes = coded[(r + 6) * 1024 .. (r + 7) * 1024];
                const r7_bytes = coded[(r + 7) * 1024 .. (r + 8) * 1024];

                for (0..16) |g| {
                    if (g + 1 < 16) {
                        const next_g = g + 1;
                        @prefetch(r0_bytes.ptr + next_g * 64, .{ .locality = 3, .cache = .data, .rw = .read });
                        @prefetch(r1_bytes.ptr + next_g * 64, .{ .locality = 3, .cache = .data, .rw = .read });
                        @prefetch(r2_bytes.ptr + next_g * 64, .{ .locality = 3, .cache = .data, .rw = .read });
                        @prefetch(r3_bytes.ptr + next_g * 64, .{ .locality = 3, .cache = .data, .rw = .read });
                        @prefetch(r4_bytes.ptr + next_g * 64, .{ .locality = 3, .cache = .data, .rw = .read });
                        @prefetch(r5_bytes.ptr + next_g * 64, .{ .locality = 3, .cache = .data, .rw = .read });
                        @prefetch(r6_bytes.ptr + next_g * 64, .{ .locality = 3, .cache = .data, .rw = .read });
                        @prefetch(r7_bytes.ptr + next_g * 64, .{ .locality = 3, .cache = .data, .rw = .read });
                    }
                    const s0: f32 = @floatCast(group_scales_raw[(r + 0) * 16 + g]);
                    const s1: f32 = @floatCast(group_scales_raw[(r + 1) * 16 + g]);
                    const s2: f32 = @floatCast(group_scales_raw[(r + 2) * 16 + g]);
                    const s3: f32 = @floatCast(group_scales_raw[(r + 3) * 16 + g]);
                    const s4: f32 = @floatCast(group_scales_raw[(r + 4) * 16 + g]);
                    const s5: f32 = @floatCast(group_scales_raw[(r + 5) * 16 + g]);
                    const s6: f32 = @floatCast(group_scales_raw[(r + 6) * 16 + g]);
                    const s7: f32 = @floatCast(group_scales_raw[(r + 7) * 16 + g]);

                    const g0_bytes = r0_bytes[g * 64 .. (g + 1) * 64];
                    const g1_bytes = r1_bytes[g * 64 .. (g + 1) * 64];
                    const g2_bytes = r2_bytes[g * 64 .. (g + 1) * 64];
                    const g3_bytes = r3_bytes[g * 64 .. (g + 1) * 64];
                    const g4_bytes = r4_bytes[g * 64 .. (g + 1) * 64];
                    const g5_bytes = r5_bytes[g * 64 .. (g + 1) * 64];
                    const g6_bytes = r6_bytes[g * 64 .. (g + 1) * 64];
                    const g7_bytes = r7_bytes[g * 64 .. (g + 1) * 64];

                    const q_even_g = act.q_even[g * 64 .. (g + 1) * 64];
                    const q_odd_g = act.q_odd[g * 64 .. (g + 1) * 64];
                    const act_scale = act.scales[g];

                    var acc0: @Vector(4, i32) = @splat(0);
                    var acc1: @Vector(4, i32) = @splat(0);
                    var acc2: @Vector(4, i32) = @splat(0);
                    var acc3: @Vector(4, i32) = @splat(0);
                    var acc4: @Vector(4, i32) = @splat(0);
                    var acc5: @Vector(4, i32) = @splat(0);
                    var acc6: @Vector(4, i32) = @splat(0);
                    var acc7: @Vector(4, i32) = @splat(0);

                    inline for (.{ 0, 16, 32, 48 }) |j| {
                        const xe: @Vector(16, i8) = q_even_g[j..][0..16].*;
                        const xo: @Vector(16, i8) = q_odd_g[j..][0..16].*;

                        const raw0: @Vector(16, u8) = g0_bytes[j..][0..16].*;
                        const raw1: @Vector(16, u8) = g1_bytes[j..][0..16].*;
                        const raw2: @Vector(16, u8) = g2_bytes[j..][0..16].*;
                        const raw3: @Vector(16, u8) = g3_bytes[j..][0..16].*;
                        const raw4: @Vector(16, u8) = g4_bytes[j..][0..16].*;
                        const raw5: @Vector(16, u8) = g5_bytes[j..][0..16].*;
                        const raw6: @Vector(16, u8) = g6_bytes[j..][0..16].*;
                        const raw7: @Vector(16, u8) = g7_bytes[j..][0..16].*;

                        const low0 = raw0 & mask0f_16;
                        const high0 = raw0 >> shift4_16;
                        const low1 = raw1 & mask0f_16;
                        const high1 = raw1 >> shift4_16;
                        const low2 = raw2 & mask0f_16;
                        const high2 = raw2 >> shift4_16;
                        const low3 = raw3 & mask0f_16;
                        const high3 = raw3 >> shift4_16;
                        const low4 = raw4 & mask0f_16;
                        const high4 = raw4 >> shift4_16;
                        const low5 = raw5 & mask0f_16;
                        const high5 = raw5 >> shift4_16;
                        const low6 = raw6 & mask0f_16;
                        const high6 = raw6 >> shift4_16;
                        const low7 = raw7 & mask0f_16;
                        const high7 = raw7 >> shift4_16;

                        asm volatile (
                            \\ sdot %[acc0].4s, %[low0].16b, %[xe].16b
                            \\ sdot %[acc1].4s, %[low1].16b, %[xe].16b
                            \\ sdot %[acc2].4s, %[low2].16b, %[xe].16b
                            \\ sdot %[acc3].4s, %[low3].16b, %[xe].16b
                            \\ sdot %[acc0].4s, %[high0].16b, %[xo].16b
                            \\ sdot %[acc1].4s, %[high1].16b, %[xo].16b
                            \\ sdot %[acc2].4s, %[high2].16b, %[xo].16b
                            \\ sdot %[acc3].4s, %[high3].16b, %[xo].16b
                            : [acc0] "+w" (acc0),
                              [acc1] "+w" (acc1),
                              [acc2] "+w" (acc2),
                              [acc3] "+w" (acc3),
                            : [xe] "w" (xe),
                              [xo] "w" (xo),
                              [low0] "w" (low0),
                              [high0] "w" (high0),
                              [low1] "w" (low1),
                              [high1] "w" (high1),
                              [low2] "w" (low2),
                              [high2] "w" (high2),
                              [low3] "w" (low3),
                              [high3] "w" (high3),
                        );
                        asm volatile (
                            \\ sdot %[acc4].4s, %[low4].16b, %[xe].16b
                            \\ sdot %[acc5].4s, %[low5].16b, %[xe].16b
                            \\ sdot %[acc6].4s, %[low6].16b, %[xe].16b
                            \\ sdot %[acc7].4s, %[low7].16b, %[xe].16b
                            \\ sdot %[acc4].4s, %[high4].16b, %[xo].16b
                            \\ sdot %[acc5].4s, %[high5].16b, %[xo].16b
                            \\ sdot %[acc6].4s, %[high6].16b, %[xo].16b
                            \\ sdot %[acc7].4s, %[high7].16b, %[xo].16b
                            : [acc4] "+w" (acc4),
                              [acc5] "+w" (acc5),
                              [acc6] "+w" (acc6),
                              [acc7] "+w" (acc7),
                            : [xe] "w" (xe),
                              [xo] "w" (xo),
                              [low4] "w" (low4),
                              [high4] "w" (high4),
                              [low5] "w" (low5),
                              [high5] "w" (high5),
                              [low6] "w" (low6),
                              [high6] "w" (high6),
                              [low7] "w" (low7),
                              [high7] "w" (high7),
                        );
                    }

                    const q_sum_8 = 8 * act.q_sums[g];
                    const int_dots: @Vector(8, i32) = .{
                        @as(i32, @intCast(@reduce(.Add, acc0))) - q_sum_8,
                        @as(i32, @intCast(@reduce(.Add, acc1))) - q_sum_8,
                        @as(i32, @intCast(@reduce(.Add, acc2))) - q_sum_8,
                        @as(i32, @intCast(@reduce(.Add, acc3))) - q_sum_8,
                        @as(i32, @intCast(@reduce(.Add, acc4))) - q_sum_8,
                        @as(i32, @intCast(@reduce(.Add, acc5))) - q_sum_8,
                        @as(i32, @intCast(@reduce(.Add, acc6))) - q_sum_8,
                        @as(i32, @intCast(@reduce(.Add, acc7))) - q_sum_8,
                    };
                    const scales: @Vector(8, f32) = .{ s0, s1, s2, s3, s4, s5, s6, s7 };
                    const act_scale_v: @Vector(8, f32) = @splat(act_scale);
                    row_dots += (@as(@Vector(8, f32), @floatFromInt(int_dots)) * act_scale_v) * scales;
                }

                var out_v: @Vector(8, f32) = @splat(0.0);
                if (has_outliers) {
                    var w_outlier0: @Vector(8, f32) = undefined;
                    var w_outlier1: @Vector(8, f32) = undefined;
                    var w_outlier2: @Vector(8, f32) = undefined;
                    var w_outlier3: @Vector(8, f32) = undefined;
                    var w_outlier4: @Vector(8, f32) = undefined;
                    var w_outlier5: @Vector(8, f32) = undefined;
                    var w_outlier6: @Vector(8, f32) = undefined;
                    var w_outlier7: @Vector(8, f32) = undefined;
                    inline for (0..8) |k| {
                        w_outlier0[k] = @floatCast(outlier_w_raw[(r + 0) * 8 + k]);
                        w_outlier1[k] = @floatCast(outlier_w_raw[(r + 1) * 8 + k]);
                        w_outlier2[k] = @floatCast(outlier_w_raw[(r + 2) * 8 + k]);
                        w_outlier3[k] = @floatCast(outlier_w_raw[(r + 3) * 8 + k]);
                        w_outlier4[k] = @floatCast(outlier_w_raw[(r + 4) * 8 + k]);
                        w_outlier5[k] = @floatCast(outlier_w_raw[(r + 5) * 8 + k]);
                        w_outlier6[k] = @floatCast(outlier_w_raw[(r + 6) * 8 + k]);
                        w_outlier7[k] = @floatCast(outlier_w_raw[(r + 7) * 8 + k]);
                    }
                    out_v = .{
                        @reduce(.Add, w_outlier0 * x_outlier),
                        @reduce(.Add, w_outlier1 * x_outlier),
                        @reduce(.Add, w_outlier2 * x_outlier),
                        @reduce(.Add, w_outlier3 * x_outlier),
                        @reduce(.Add, w_outlier4 * x_outlier),
                        @reduce(.Add, w_outlier5 * x_outlier),
                        @reduce(.Add, w_outlier6 * x_outlier),
                        @reduce(.Add, w_outlier7 * x_outlier),
                    };
                }

                const y_slice = y[row_base + r .. row_base + r + 8];
                if (comptime accumulate) {
                    y_slice[0..8].* = @as(@Vector(8, f32), y_slice[0..8].*) + row_dots + out_v;
                } else {
                    y_slice[0..8].* = row_dots + out_v;
                }
            }
            return;
        }

        var r: usize = 0;
        while (r < 16) : (r += 4) {
            var row_dots: @Vector(4, f32) = @splat(0.0);
            const r0_bytes = coded[(r + 0) * 1024 .. (r + 1) * 1024];
            const r1_bytes = coded[(r + 1) * 1024 .. (r + 2) * 1024];
            const r2_bytes = coded[(r + 2) * 1024 .. (r + 3) * 1024];
            const r3_bytes = coded[(r + 3) * 1024 .. (r + 4) * 1024];

            for (0..16) |g| {
                const s0: f32 = @floatCast(group_scales_raw[(r + 0) * 16 + g]);
                const s1: f32 = @floatCast(group_scales_raw[(r + 1) * 16 + g]);
                const s2: f32 = @floatCast(group_scales_raw[(r + 2) * 16 + g]);
                const s3: f32 = @floatCast(group_scales_raw[(r + 3) * 16 + g]);

                const g0_bytes = r0_bytes[g * 64 .. (g + 1) * 64];
                const g1_bytes = r1_bytes[g * 64 .. (g + 1) * 64];
                const g2_bytes = r2_bytes[g * 64 .. (g + 1) * 64];
                const g3_bytes = r3_bytes[g * 64 .. (g + 1) * 64];

                if (comptime use_sdot) {
                    const q_even_g = act.q_even[g * 64 .. (g + 1) * 64];
                    const q_odd_g = act.q_odd[g * 64 .. (g + 1) * 64];
                    const act_scale = act.scales[g];

                    var acc0_e: @Vector(4, i32) = @splat(0);
                    var acc0_o: @Vector(4, i32) = @splat(0);
                    var acc1_e: @Vector(4, i32) = @splat(0);
                    var acc1_o: @Vector(4, i32) = @splat(0);
                    var acc2_e: @Vector(4, i32) = @splat(0);
                    var acc2_o: @Vector(4, i32) = @splat(0);
                    var acc3_e: @Vector(4, i32) = @splat(0);
                    var acc3_o: @Vector(4, i32) = @splat(0);

                    inline for (.{ 0, 16, 32, 48 }) |j| {
                        const xe: @Vector(16, i8) = q_even_g[j..][0..16].*;
                        const xo: @Vector(16, i8) = q_odd_g[j..][0..16].*;

                        const raw0: @Vector(16, u8) = g0_bytes[j..][0..16].*;
                        const raw1: @Vector(16, u8) = g1_bytes[j..][0..16].*;
                        const raw2: @Vector(16, u8) = g2_bytes[j..][0..16].*;
                        const raw3: @Vector(16, u8) = g3_bytes[j..][0..16].*;

                        const low0 = raw0 & mask0f_16;
                        const high0 = raw0 >> shift4_16;
                        const low1 = raw1 & mask0f_16;
                        const high1 = raw1 >> shift4_16;
                        const low2 = raw2 & mask0f_16;
                        const high2 = raw2 >> shift4_16;
                        const low3 = raw3 & mask0f_16;
                        const high3 = raw3 >> shift4_16;

                        if (comptime @import("builtin").cpu.arch == .aarch64) {
                            asm volatile (
                                \\ sdot %[acc0_e].4s, %[low0].16b, %[xe].16b
                                \\ sdot %[acc0_o].4s, %[high0].16b, %[xo].16b
                                \\ sdot %[acc1_e].4s, %[low1].16b, %[xe].16b
                                \\ sdot %[acc1_o].4s, %[high1].16b, %[xo].16b
                                \\ sdot %[acc2_e].4s, %[low2].16b, %[xe].16b
                                \\ sdot %[acc2_o].4s, %[high2].16b, %[xo].16b
                                \\ sdot %[acc3_e].4s, %[low3].16b, %[xe].16b
                                \\ sdot %[acc3_o].4s, %[high3].16b, %[xo].16b
                                : [acc0_e] "+w" (acc0_e),
                                  [acc0_o] "+w" (acc0_o),
                                  [acc1_e] "+w" (acc1_e),
                                  [acc1_o] "+w" (acc1_o),
                                  [acc2_e] "+w" (acc2_e),
                                  [acc2_o] "+w" (acc2_o),
                                  [acc3_e] "+w" (acc3_e),
                                  [acc3_o] "+w" (acc3_o),
                                : [xe] "w" (xe),
                                  [xo] "w" (xo),
                                  [low0] "w" (low0),
                                  [high0] "w" (high0),
                                  [low1] "w" (low1),
                                  [high1] "w" (high1),
                                  [low2] "w" (low2),
                                  [high2] "w" (high2),
                                  [low3] "w" (low3),
                                  [high3] "w" (high3),
                            );
                        } else {
                            const xe_16: @Vector(16, i16) = xe;
                            const xo_16: @Vector(16, i16) = xo;

                            inline for (.{
                                .{ &acc0_e, &acc0_o, low0, high0 },
                                .{ &acc1_e, &acc1_o, low1, high1 },
                                .{ &acc2_e, &acc2_o, low2, high2 },
                                .{ &acc3_e, &acc3_o, low3, high3 },
                            }) |pair| {
                                const l_i8: @Vector(16, i8) = @bitCast(pair[2]);
                                const h_i8: @Vector(16, i8) = @bitCast(pair[3]);
                                const l_16: @Vector(16, i16) = l_i8;
                                const h_16: @Vector(16, i16) = h_i8;
                                const pe: @Vector(16, i32) = @as(@Vector(16, i32), xe_16) * @as(@Vector(16, i32), l_16);
                                const po: @Vector(16, i32) = @as(@Vector(16, i32), xo_16) * @as(@Vector(16, i32), h_16);
                                pair[0].* += .{ pe[0] + pe[1] + pe[2] + pe[3], pe[4] + pe[5] + pe[6] + pe[7], pe[8] + pe[9] + pe[10] + pe[11], pe[12] + pe[13] + pe[14] + pe[15] };
                                pair[1].* += .{ po[0] + po[1] + po[2] + po[3], po[4] + po[5] + po[6] + po[7], po[8] + po[9] + po[10] + po[11], po[12] + po[13] + po[14] + po[15] };
                            }
                        }
                    }

                    const q_sum_8 = 8 * act.q_sums[g];
                    const int_dots: @Vector(4, i32) = .{
                        @as(i32, @intCast(@reduce(.Add, acc0_e + acc0_o))) - q_sum_8,
                        @as(i32, @intCast(@reduce(.Add, acc1_e + acc1_o))) - q_sum_8,
                        @as(i32, @intCast(@reduce(.Add, acc2_e + acc2_o))) - q_sum_8,
                        @as(i32, @intCast(@reduce(.Add, acc3_e + acc3_o))) - q_sum_8,
                    };
                    const scales: @Vector(4, f32) = .{ s0, s1, s2, s3 };
                    const act_scale_v: @Vector(4, f32) = @splat(act_scale);
                    row_dots += (@as(@Vector(4, f32), @floatFromInt(int_dots)) * act_scale_v) * scales;
                } else {
                    const x_even_g = act.even[g * 64 .. (g + 1) * 64];
                    const x_odd_g = act.odd[g * 64 .. (g + 1) * 64];
                    const mask0f_8: @Vector(8, u8) = @splat(0x0F);
                    const shift4_8: @Vector(8, u3) = @splat(4);

                    var d0_vec0: @Vector(8, f32) = @splat(0.0);
                    var d0_vec1: @Vector(8, f32) = @splat(0.0);
                    var d1_vec0: @Vector(8, f32) = @splat(0.0);
                    var d1_vec1: @Vector(8, f32) = @splat(0.0);
                    var d2_vec0: @Vector(8, f32) = @splat(0.0);
                    var d2_vec1: @Vector(8, f32) = @splat(0.0);
                    var d3_vec0: @Vector(8, f32) = @splat(0.0);
                    var d3_vec1: @Vector(8, f32) = @splat(0.0);

                    inline for (.{ 0, 16, 32, 48 }) |j| {
                        const xe0: @Vector(8, f32) = x_even_g[j..][0..8].*;
                        const xo0: @Vector(8, f32) = x_odd_g[j..][0..8].*;
                        const xe1: @Vector(8, f32) = x_even_g[j + 8 ..][0..8].*;
                        const xo1: @Vector(8, f32) = x_odd_g[j + 8 ..][0..8].*;

                        const raw0_0: @Vector(8, u8) = g0_bytes[j..][0..8].*;
                        d0_vec0 += @as(@Vector(8, f32), @floatFromInt(raw0_0 & mask0f_8)) * xe0;
                        d0_vec1 += @as(@Vector(8, f32), @floatFromInt(raw0_0 >> shift4_8)) * xo0;
                        const raw1_0: @Vector(8, u8) = g1_bytes[j..][0..8].*;
                        d1_vec0 += @as(@Vector(8, f32), @floatFromInt(raw1_0 & mask0f_8)) * xe0;
                        d1_vec1 += @as(@Vector(8, f32), @floatFromInt(raw1_0 >> shift4_8)) * xo0;
                        const raw2_0: @Vector(8, u8) = g2_bytes[j..][0..8].*;
                        d2_vec0 += @as(@Vector(8, f32), @floatFromInt(raw2_0 & mask0f_8)) * xe0;
                        d2_vec1 += @as(@Vector(8, f32), @floatFromInt(raw2_0 >> shift4_8)) * xo0;
                        const raw3_0: @Vector(8, u8) = g3_bytes[j..][0..8].*;
                        d3_vec0 += @as(@Vector(8, f32), @floatFromInt(raw3_0 & mask0f_8)) * xe0;
                        d3_vec1 += @as(@Vector(8, f32), @floatFromInt(raw3_0 >> shift4_8)) * xo0;

                        const raw0_1: @Vector(8, u8) = g0_bytes[j + 8 ..][0..8].*;
                        d0_vec0 += @as(@Vector(8, f32), @floatFromInt(raw0_1 & mask0f_8)) * xe1;
                        d0_vec1 += @as(@Vector(8, f32), @floatFromInt(raw0_1 >> shift4_8)) * xo1;
                        const raw1_1: @Vector(8, u8) = g1_bytes[j + 8 ..][0..8].*;
                        d1_vec0 += @as(@Vector(8, f32), @floatFromInt(raw1_1 & mask0f_8)) * xe1;
                        d1_vec1 += @as(@Vector(8, f32), @floatFromInt(raw1_1 >> shift4_8)) * xo1;
                        const raw2_1: @Vector(8, u8) = g2_bytes[j + 8 ..][0..8].*;
                        d2_vec0 += @as(@Vector(8, f32), @floatFromInt(raw2_1 & mask0f_8)) * xe1;
                        d2_vec1 += @as(@Vector(8, f32), @floatFromInt(raw2_1 >> shift4_8)) * xo1;
                        const raw3_1: @Vector(8, u8) = g3_bytes[j + 8 ..][0..8].*;
                        d3_vec0 += @as(@Vector(8, f32), @floatFromInt(raw3_1 & mask0f_8)) * xe1;
                        d3_vec1 += @as(@Vector(8, f32), @floatFromInt(raw3_1 >> shift4_8)) * xo1;
                    }

                    const dot_raw0 = @reduce(.Add, d0_vec0 + d0_vec1);
                    const dot_raw1 = @reduce(.Add, d1_vec0 + d1_vec1);
                    const dot_raw2 = @reduce(.Add, d2_vec0 + d2_vec1);
                    const dot_raw3 = @reduce(.Add, d3_vec0 + d3_vec1);
                    const dot_raw0_3: @Vector(4, f32) = .{ dot_raw0, dot_raw1, dot_raw2, dot_raw3 };
                    const s_vec0_3: @Vector(4, f32) = .{ s0, s1, s2, s3 };
                    const act_sum8: @Vector(4, f32) = @splat(8.0 * act.sums[g]);
                    row_dots += (dot_raw0_3 - act_sum8) * s_vec0_3;
                }
            }

            var out_v: @Vector(4, f32) = @splat(0.0);
            if (has_outliers) {
                var w_outlier0: @Vector(8, f32) = undefined;
                var w_outlier1: @Vector(8, f32) = undefined;
                var w_outlier2: @Vector(8, f32) = undefined;
                var w_outlier3: @Vector(8, f32) = undefined;
                inline for (0..8) |k| {
                    w_outlier0[k] = @floatCast(outlier_w_raw[(r + 0) * 8 + k]);
                    w_outlier1[k] = @floatCast(outlier_w_raw[(r + 1) * 8 + k]);
                    w_outlier2[k] = @floatCast(outlier_w_raw[(r + 2) * 8 + k]);
                    w_outlier3[k] = @floatCast(outlier_w_raw[(r + 3) * 8 + k]);
                }
                out_v = .{
                    @reduce(.Add, w_outlier0 * x_outlier),
                    @reduce(.Add, w_outlier1 * x_outlier),
                    @reduce(.Add, w_outlier2 * x_outlier),
                    @reduce(.Add, w_outlier3 * x_outlier),
                };
            }

            const y_slice = y[row_base + r .. row_base + r + 4];
            if (comptime accumulate) {
                y_slice[0..4].* = @as(@Vector(4, f32), y_slice[0..4].*) + row_dots + out_v;
            } else {
                y_slice[0..4].* = row_dots + out_v;
            }
        }
        return;
    }

    const scale: f32 = @bitCast(std.mem.readInt(u32, payload[32..36], .little));
    const bias: f32 = @bitCast(std.mem.readInt(u32, payload[36..40], .little));
    const coded: [*]const u8 = @ptrCast(&cell.fingerprints);
    const sum_x = @reduce(.Add, @as(@Vector(16, f32), act.sums));

    for (0..16) |r| {
        const row_bytes = coded[r * 1024 .. (r + 1) * 1024];
        var dot_vec: @Vector(8, f32) = @splat(0.0);
        var j: usize = 0;
        while (j + 4 <= 1024) : (j += 4) {
            const b0 = row_bytes[j];
            const b1 = row_bytes[j + 1];
            const b2 = row_bytes[j + 2];
            const b3 = row_bytes[j + 3];

            const q_vec: @Vector(8, f32) = .{
                @as(f32, @floatFromInt(b0 & 0x0F)) - 8.0,
                @as(f32, @floatFromInt(b0 >> 4)) - 8.0,
                @as(f32, @floatFromInt(b1 & 0x0F)) - 8.0,
                @as(f32, @floatFromInt(b1 >> 4)) - 8.0,
                @as(f32, @floatFromInt(b2 & 0x0F)) - 8.0,
                @as(f32, @floatFromInt(b2 >> 4)) - 8.0,
                @as(f32, @floatFromInt(b3 & 0x0F)) - 8.0,
                @as(f32, @floatFromInt(b3 >> 4)) - 8.0,
            };
            const x_vec: @Vector(8, f32) = act.raw[2 * j ..][0..8].*;
            dot_vec += q_vec * x_vec;
        }
        if (comptime accumulate) {
            y[row_base + r] += @reduce(.Add, dot_vec) * scale + sum_x * bias;
        } else {
            y[row_base + r] = @reduce(.Add, dot_vec) * scale + sum_x * bias;
        }
    }
}

pub fn gemvTileQuadRowGateUpFusedCellDirect(
    g_cell: *const geometry.Cell,
    u_cell: *const geometry.Cell,
    act: *const Activation2048,
    out: []f32,
    row_base: usize,
    comptime use_sdot: bool,
) void {
    const g_payload = &g_cell.semantic_payload;
    const g_meta: *const weight_archive.TileMetadata = @ptrCast(@alignCast(g_payload.ptr));
    const u_payload = &u_cell.semantic_payload;
    const u_meta: *const weight_archive.TileMetadata = @ptrCast(@alignCast(u_payload.ptr));

    if ((g_meta.custom_flags & weight_archive.FLAG_GROUP128_OUTLIERS) != 0 and
        (u_meta.custom_flags & weight_archive.FLAG_GROUP128_OUTLIERS) != 0 and
        comptime @import("builtin").cpu.arch == .aarch64 and use_sdot)
    {
        const g_scales_raw: [*]const f16 = @ptrCast(@alignCast(g_payload[48..560].ptr));
        const g_outlier_w: [*]const f16 = @ptrCast(@alignCast(g_payload[560..816].ptr));
        const g_outlier_cols: [*]const u16 = @ptrCast(@alignCast(g_payload[816..832].ptr));
        const g_coded: [*]const u8 = @ptrCast(&g_cell.fingerprints);

        const u_scales_raw: [*]const f16 = @ptrCast(@alignCast(u_payload[48..560].ptr));
        const u_outlier_w: [*]const f16 = @ptrCast(@alignCast(u_payload[560..816].ptr));
        const u_outlier_cols: [*]const u16 = @ptrCast(@alignCast(u_payload[816..832].ptr));
        const u_coded: [*]const u8 = @ptrCast(&u_cell.fingerprints);

        var g_x_outlier: @Vector(8, f32) = undefined;
        var u_x_outlier: @Vector(8, f32) = undefined;
        inline for (0..8) |k| {
            const g_col = g_outlier_cols[k];
            const u_col = u_outlier_cols[k];
            g_x_outlier[k] = if ((g_col & 1) == 0) act.even[g_col / 2] else act.odd[g_col / 2];
            u_x_outlier[k] = if ((u_col & 1) == 0) act.even[u_col / 2] else act.odd[u_col / 2];
        }

        const mask0f_16: @Vector(16, u8) = @splat(0x0F);
        const shift4_16: @Vector(16, u3) = @splat(4);
        const ones: @Vector(8, f32) = @splat(1.0);

        var r: usize = 0;
        while (r < 16) : (r += 8) {
            var g_dots: @Vector(8, f32) = @splat(0.0);
            var u_dots: @Vector(8, f32) = @splat(0.0);

            // Fused Gate and Up 8 rows: load xe, xo once per subchunk and accumulate both
            const gr0 = g_coded[(r + 0) * 1024 .. (r + 1) * 1024];
            const gr1 = g_coded[(r + 1) * 1024 .. (r + 2) * 1024];
            const gr2 = g_coded[(r + 2) * 1024 .. (r + 3) * 1024];
            const gr3 = g_coded[(r + 3) * 1024 .. (r + 4) * 1024];
            const gr4 = g_coded[(r + 4) * 1024 .. (r + 5) * 1024];
            const gr5 = g_coded[(r + 5) * 1024 .. (r + 6) * 1024];
            const gr6 = g_coded[(r + 6) * 1024 .. (r + 7) * 1024];
            const gr7 = g_coded[(r + 7) * 1024 .. (r + 8) * 1024];

            const ur0 = u_coded[(r + 0) * 1024 .. (r + 1) * 1024];
            const ur1 = u_coded[(r + 1) * 1024 .. (r + 2) * 1024];
            const ur2 = u_coded[(r + 2) * 1024 .. (r + 3) * 1024];
            const ur3 = u_coded[(r + 3) * 1024 .. (r + 4) * 1024];
            const ur4 = u_coded[(r + 4) * 1024 .. (r + 5) * 1024];
            const ur5 = u_coded[(r + 5) * 1024 .. (r + 6) * 1024];
            const ur6 = u_coded[(r + 6) * 1024 .. (r + 7) * 1024];
            const ur7 = u_coded[(r + 7) * 1024 .. (r + 8) * 1024];

            for (0..16) |g| {
                if (g + 2 < 16) {
                    const next_g = g + 2;
                    @prefetch(gr0.ptr + next_g * 64, .{ .locality = 2, .cache = .data, .rw = .read });
                    @prefetch(gr1.ptr + next_g * 64, .{ .locality = 2, .cache = .data, .rw = .read });
                    @prefetch(gr2.ptr + next_g * 64, .{ .locality = 2, .cache = .data, .rw = .read });
                    @prefetch(gr3.ptr + next_g * 64, .{ .locality = 2, .cache = .data, .rw = .read });
                    @prefetch(ur0.ptr + next_g * 64, .{ .locality = 2, .cache = .data, .rw = .read });
                    @prefetch(ur1.ptr + next_g * 64, .{ .locality = 2, .cache = .data, .rw = .read });
                    @prefetch(ur2.ptr + next_g * 64, .{ .locality = 2, .cache = .data, .rw = .read });
                    @prefetch(ur3.ptr + next_g * 64, .{ .locality = 2, .cache = .data, .rw = .read });
                }
                const gs0: f32 = @floatCast(g_scales_raw[(r + 0) * 16 + g]);
                const gs1: f32 = @floatCast(g_scales_raw[(r + 1) * 16 + g]);
                const gs2: f32 = @floatCast(g_scales_raw[(r + 2) * 16 + g]);
                const gs3: f32 = @floatCast(g_scales_raw[(r + 3) * 16 + g]);
                const gs4: f32 = @floatCast(g_scales_raw[(r + 4) * 16 + g]);
                const gs5: f32 = @floatCast(g_scales_raw[(r + 5) * 16 + g]);
                const gs6: f32 = @floatCast(g_scales_raw[(r + 6) * 16 + g]);
                const gs7: f32 = @floatCast(g_scales_raw[(r + 7) * 16 + g]);

                const us0: f32 = @floatCast(u_scales_raw[(r + 0) * 16 + g]);
                const us1: f32 = @floatCast(u_scales_raw[(r + 1) * 16 + g]);
                const us2: f32 = @floatCast(u_scales_raw[(r + 2) * 16 + g]);
                const us3: f32 = @floatCast(u_scales_raw[(r + 3) * 16 + g]);
                const us4: f32 = @floatCast(u_scales_raw[(r + 4) * 16 + g]);
                const us5: f32 = @floatCast(u_scales_raw[(r + 5) * 16 + g]);
                const us6: f32 = @floatCast(u_scales_raw[(r + 6) * 16 + g]);
                const us7: f32 = @floatCast(u_scales_raw[(r + 7) * 16 + g]);

                const g0_bytes = gr0[g * 64 .. (g + 1) * 64];
                const g1_bytes = gr1[g * 64 .. (g + 1) * 64];
                const g2_bytes = gr2[g * 64 .. (g + 1) * 64];
                const g3_bytes = gr3[g * 64 .. (g + 1) * 64];
                const g4_bytes = gr4[g * 64 .. (g + 1) * 64];
                const g5_bytes = gr5[g * 64 .. (g + 1) * 64];
                const g6_bytes = gr6[g * 64 .. (g + 1) * 64];
                const g7_bytes = gr7[g * 64 .. (g + 1) * 64];

                const u0_bytes = ur0[g * 64 .. (g + 1) * 64];
                const u1_bytes = ur1[g * 64 .. (g + 1) * 64];
                const u2_bytes = ur2[g * 64 .. (g + 1) * 64];
                const u3_bytes = ur3[g * 64 .. (g + 1) * 64];
                const u4_bytes = ur4[g * 64 .. (g + 1) * 64];
                const u5_bytes = ur5[g * 64 .. (g + 1) * 64];
                const u6_bytes = ur6[g * 64 .. (g + 1) * 64];
                const u7_bytes = ur7[g * 64 .. (g + 1) * 64];

                const q_even_g = act.q_even[g * 64 .. (g + 1) * 64];
                const q_odd_g = act.q_odd[g * 64 .. (g + 1) * 64];
                const act_scale = act.scales[g];

                var ga0: @Vector(4, i32) = @splat(0);
                var ga1: @Vector(4, i32) = @splat(0);
                var ga2: @Vector(4, i32) = @splat(0);
                var ga3: @Vector(4, i32) = @splat(0);
                var ga4: @Vector(4, i32) = @splat(0);
                var ga5: @Vector(4, i32) = @splat(0);
                var ga6: @Vector(4, i32) = @splat(0);
                var ga7: @Vector(4, i32) = @splat(0);

                var ua0: @Vector(4, i32) = @splat(0);
                var ua1: @Vector(4, i32) = @splat(0);
                var ua2: @Vector(4, i32) = @splat(0);
                var ua3: @Vector(4, i32) = @splat(0);
                var ua4: @Vector(4, i32) = @splat(0);
                var ua5: @Vector(4, i32) = @splat(0);
                var ua6: @Vector(4, i32) = @splat(0);
                var ua7: @Vector(4, i32) = @splat(0);

                inline for (.{ 0, 16, 32, 48 }) |j| {
                    const xe: @Vector(16, i8) = q_even_g[j..][0..16].*;
                    const xo: @Vector(16, i8) = q_odd_g[j..][0..16].*;

                    // Gate rows 0..3
                    {
                        const raw0: @Vector(16, u8) = g0_bytes[j..][0..16].*;
                        const raw1: @Vector(16, u8) = g1_bytes[j..][0..16].*;
                        const raw2: @Vector(16, u8) = g2_bytes[j..][0..16].*;
                        const raw3: @Vector(16, u8) = g3_bytes[j..][0..16].*;
                        const low0 = raw0 & mask0f_16;
                        const high0 = raw0 >> shift4_16;
                        const low1 = raw1 & mask0f_16;
                        const high1 = raw1 >> shift4_16;
                        const low2 = raw2 & mask0f_16;
                        const high2 = raw2 >> shift4_16;
                        const low3 = raw3 & mask0f_16;
                        const high3 = raw3 >> shift4_16;

                        asm volatile (
                            \\ sdot %[acc0].4s, %[low0].16b, %[xe].16b
                            \\ sdot %[acc1].4s, %[low1].16b, %[xe].16b
                            \\ sdot %[acc2].4s, %[low2].16b, %[xe].16b
                            \\ sdot %[acc3].4s, %[low3].16b, %[xe].16b
                            \\ sdot %[acc0].4s, %[high0].16b, %[xo].16b
                            \\ sdot %[acc1].4s, %[high1].16b, %[xo].16b
                            \\ sdot %[acc2].4s, %[high2].16b, %[xo].16b
                            \\ sdot %[acc3].4s, %[high3].16b, %[xo].16b
                            : [acc0] "+w" (ga0),
                              [acc1] "+w" (ga1),
                              [acc2] "+w" (ga2),
                              [acc3] "+w" (ga3),
                            : [xe] "w" (xe),
                              [xo] "w" (xo),
                              [low0] "w" (low0),
                              [high0] "w" (high0),
                              [low1] "w" (low1),
                              [high1] "w" (high1),
                              [low2] "w" (low2),
                              [high2] "w" (high2),
                              [low3] "w" (low3),
                              [high3] "w" (high3),
                        );
                    }

                    // Gate rows 4..7
                    {
                        const raw4: @Vector(16, u8) = g4_bytes[j..][0..16].*;
                        const raw5: @Vector(16, u8) = g5_bytes[j..][0..16].*;
                        const raw6: @Vector(16, u8) = g6_bytes[j..][0..16].*;
                        const raw7: @Vector(16, u8) = g7_bytes[j..][0..16].*;
                        const low4 = raw4 & mask0f_16;
                        const high4 = raw4 >> shift4_16;
                        const low5 = raw5 & mask0f_16;
                        const high5 = raw5 >> shift4_16;
                        const low6 = raw6 & mask0f_16;
                        const high6 = raw6 >> shift4_16;
                        const low7 = raw7 & mask0f_16;
                        const high7 = raw7 >> shift4_16;

                        asm volatile (
                            \\ sdot %[acc4].4s, %[low4].16b, %[xe].16b
                            \\ sdot %[acc5].4s, %[low5].16b, %[xe].16b
                            \\ sdot %[acc6].4s, %[low6].16b, %[xe].16b
                            \\ sdot %[acc7].4s, %[low7].16b, %[xe].16b
                            \\ sdot %[acc4].4s, %[high4].16b, %[xo].16b
                            \\ sdot %[acc5].4s, %[high5].16b, %[xo].16b
                            \\ sdot %[acc6].4s, %[high6].16b, %[xo].16b
                            \\ sdot %[acc7].4s, %[high7].16b, %[xo].16b
                            : [acc4] "+w" (ga4),
                              [acc5] "+w" (ga5),
                              [acc6] "+w" (ga6),
                              [acc7] "+w" (ga7),
                            : [xe] "w" (xe),
                              [xo] "w" (xo),
                              [low4] "w" (low4),
                              [high4] "w" (high4),
                              [low5] "w" (low5),
                              [high5] "w" (high5),
                              [low6] "w" (low6),
                              [high6] "w" (high6),
                              [low7] "w" (low7),
                              [high7] "w" (high7),
                        );
                    }

                    // Up rows 0..3 (reusing xe, xo directly in registers)
                    {
                        const raw0: @Vector(16, u8) = u0_bytes[j..][0..16].*;
                        const raw1: @Vector(16, u8) = u1_bytes[j..][0..16].*;
                        const raw2: @Vector(16, u8) = u2_bytes[j..][0..16].*;
                        const raw3: @Vector(16, u8) = u3_bytes[j..][0..16].*;
                        const low0 = raw0 & mask0f_16;
                        const high0 = raw0 >> shift4_16;
                        const low1 = raw1 & mask0f_16;
                        const high1 = raw1 >> shift4_16;
                        const low2 = raw2 & mask0f_16;
                        const high2 = raw2 >> shift4_16;
                        const low3 = raw3 & mask0f_16;
                        const high3 = raw3 >> shift4_16;

                        asm volatile (
                            \\ sdot %[acc0].4s, %[low0].16b, %[xe].16b
                            \\ sdot %[acc1].4s, %[low1].16b, %[xe].16b
                            \\ sdot %[acc2].4s, %[low2].16b, %[xe].16b
                            \\ sdot %[acc3].4s, %[low3].16b, %[xe].16b
                            \\ sdot %[acc0].4s, %[high0].16b, %[xo].16b
                            \\ sdot %[acc1].4s, %[high1].16b, %[xo].16b
                            \\ sdot %[acc2].4s, %[high2].16b, %[xo].16b
                            \\ sdot %[acc3].4s, %[high3].16b, %[xo].16b
                            : [acc0] "+w" (ua0),
                              [acc1] "+w" (ua1),
                              [acc2] "+w" (ua2),
                              [acc3] "+w" (ua3),
                            : [xe] "w" (xe),
                              [xo] "w" (xo),
                              [low0] "w" (low0),
                              [high0] "w" (high0),
                              [low1] "w" (low1),
                              [high1] "w" (high1),
                              [low2] "w" (low2),
                              [high2] "w" (high2),
                              [low3] "w" (low3),
                              [high3] "w" (high3),
                        );
                    }

                    // Up rows 4..7 (reusing xe, xo directly in registers)
                    {
                        const raw4: @Vector(16, u8) = u4_bytes[j..][0..16].*;
                        const raw5: @Vector(16, u8) = u5_bytes[j..][0..16].*;
                        const raw6: @Vector(16, u8) = u6_bytes[j..][0..16].*;
                        const raw7: @Vector(16, u8) = u7_bytes[j..][0..16].*;
                        const low4 = raw4 & mask0f_16;
                        const high4 = raw4 >> shift4_16;
                        const low5 = raw5 & mask0f_16;
                        const high5 = raw5 >> shift4_16;
                        const low6 = raw6 & mask0f_16;
                        const high6 = raw6 >> shift4_16;
                        const low7 = raw7 & mask0f_16;
                        const high7 = raw7 >> shift4_16;

                        asm volatile (
                            \\ sdot %[acc4].4s, %[low4].16b, %[xe].16b
                            \\ sdot %[acc5].4s, %[low5].16b, %[xe].16b
                            \\ sdot %[acc6].4s, %[low6].16b, %[xe].16b
                            \\ sdot %[acc7].4s, %[low7].16b, %[xe].16b
                            \\ sdot %[acc4].4s, %[high4].16b, %[xo].16b
                            \\ sdot %[acc5].4s, %[high5].16b, %[xo].16b
                            \\ sdot %[acc6].4s, %[high6].16b, %[xo].16b
                            \\ sdot %[acc7].4s, %[high7].16b, %[xo].16b
                            : [acc4] "+w" (ua4),
                              [acc5] "+w" (ua5),
                              [acc6] "+w" (ua6),
                              [acc7] "+w" (ua7),
                            : [xe] "w" (xe),
                              [xo] "w" (xo),
                              [low4] "w" (low4),
                              [high4] "w" (high4),
                              [low5] "w" (low5),
                              [high5] "w" (high5),
                              [low6] "w" (low6),
                              [high6] "w" (high6),
                              [low7] "w" (low7),
                              [high7] "w" (high7),
                        );
                    }
                }

                const q_sum_8 = 8 * act.q_sums[g];
                const act_scale_v: @Vector(8, f32) = @splat(act_scale);

                const g_int_dots: @Vector(8, i32) = .{
                    @as(i32, @intCast(@reduce(.Add, ga0))) - q_sum_8,
                    @as(i32, @intCast(@reduce(.Add, ga1))) - q_sum_8,
                    @as(i32, @intCast(@reduce(.Add, ga2))) - q_sum_8,
                    @as(i32, @intCast(@reduce(.Add, ga3))) - q_sum_8,
                    @as(i32, @intCast(@reduce(.Add, ga4))) - q_sum_8,
                    @as(i32, @intCast(@reduce(.Add, ga5))) - q_sum_8,
                    @as(i32, @intCast(@reduce(.Add, ga6))) - q_sum_8,
                    @as(i32, @intCast(@reduce(.Add, ga7))) - q_sum_8,
                };
                const g_scales: @Vector(8, f32) = .{ gs0, gs1, gs2, gs3, gs4, gs5, gs6, gs7 };
                g_dots += (@as(@Vector(8, f32), @floatFromInt(g_int_dots)) * act_scale_v) * g_scales;

                const u_int_dots: @Vector(8, i32) = .{
                    @as(i32, @intCast(@reduce(.Add, ua0))) - q_sum_8,
                    @as(i32, @intCast(@reduce(.Add, ua1))) - q_sum_8,
                    @as(i32, @intCast(@reduce(.Add, ua2))) - q_sum_8,
                    @as(i32, @intCast(@reduce(.Add, ua3))) - q_sum_8,
                    @as(i32, @intCast(@reduce(.Add, ua4))) - q_sum_8,
                    @as(i32, @intCast(@reduce(.Add, ua5))) - q_sum_8,
                    @as(i32, @intCast(@reduce(.Add, ua6))) - q_sum_8,
                    @as(i32, @intCast(@reduce(.Add, ua7))) - q_sum_8,
                };
                const u_scales: @Vector(8, f32) = .{ us0, us1, us2, us3, us4, us5, us6, us7 };
                u_dots += (@as(@Vector(8, f32), @floatFromInt(u_int_dots)) * act_scale_v) * u_scales;
            }

            // Outliers & SwiGLU
            var gw0: @Vector(8, f32) = undefined;
            var gw1: @Vector(8, f32) = undefined;
            var gw2: @Vector(8, f32) = undefined;
            var gw3: @Vector(8, f32) = undefined;
            var gw4: @Vector(8, f32) = undefined;
            var gw5: @Vector(8, f32) = undefined;
            var gw6: @Vector(8, f32) = undefined;
            var gw7: @Vector(8, f32) = undefined;

            var uw0: @Vector(8, f32) = undefined;
            var uw1: @Vector(8, f32) = undefined;
            var uw2: @Vector(8, f32) = undefined;
            var uw3: @Vector(8, f32) = undefined;
            var uw4: @Vector(8, f32) = undefined;
            var uw5: @Vector(8, f32) = undefined;
            var uw6: @Vector(8, f32) = undefined;
            var uw7: @Vector(8, f32) = undefined;

            inline for (0..8) |k| {
                gw0[k] = @floatCast(g_outlier_w[(r + 0) * 8 + k]);
                gw1[k] = @floatCast(g_outlier_w[(r + 1) * 8 + k]);
                gw2[k] = @floatCast(g_outlier_w[(r + 2) * 8 + k]);
                gw3[k] = @floatCast(g_outlier_w[(r + 3) * 8 + k]);
                gw4[k] = @floatCast(g_outlier_w[(r + 4) * 8 + k]);
                gw5[k] = @floatCast(g_outlier_w[(r + 5) * 8 + k]);
                gw6[k] = @floatCast(g_outlier_w[(r + 6) * 8 + k]);
                gw7[k] = @floatCast(g_outlier_w[(r + 7) * 8 + k]);

                uw0[k] = @floatCast(u_outlier_w[(r + 0) * 8 + k]);
                uw1[k] = @floatCast(u_outlier_w[(r + 1) * 8 + k]);
                uw2[k] = @floatCast(u_outlier_w[(r + 2) * 8 + k]);
                uw3[k] = @floatCast(u_outlier_w[(r + 3) * 8 + k]);
                uw4[k] = @floatCast(u_outlier_w[(r + 4) * 8 + k]);
                uw5[k] = @floatCast(u_outlier_w[(r + 5) * 8 + k]);
                uw6[k] = @floatCast(u_outlier_w[(r + 6) * 8 + k]);
                uw7[k] = @floatCast(u_outlier_w[(r + 7) * 8 + k]);
            }

            const g_out_v: @Vector(8, f32) = .{
                @reduce(.Add, gw0 * g_x_outlier),
                @reduce(.Add, gw1 * g_x_outlier),
                @reduce(.Add, gw2 * g_x_outlier),
                @reduce(.Add, gw3 * g_x_outlier),
                @reduce(.Add, gw4 * g_x_outlier),
                @reduce(.Add, gw5 * g_x_outlier),
                @reduce(.Add, gw6 * g_x_outlier),
                @reduce(.Add, gw7 * g_x_outlier),
            };
            const u_out_v: @Vector(8, f32) = .{
                @reduce(.Add, uw0 * u_x_outlier),
                @reduce(.Add, uw1 * u_x_outlier),
                @reduce(.Add, uw2 * u_x_outlier),
                @reduce(.Add, uw3 * u_x_outlier),
                @reduce(.Add, uw4 * u_x_outlier),
                @reduce(.Add, uw5 * u_x_outlier),
                @reduce(.Add, uw6 * u_x_outlier),
                @reduce(.Add, uw7 * u_x_outlier),
            };

            const total_g = g_dots + g_out_v;
            const total_u = u_dots + u_out_v;

            // In-register vectorized SwiGLU
            const silu_v = total_g / (ones + @exp(-total_g));
            out[row_base + r ..][0..8].* = silu_v * total_u;
        }
        return;
    }

    // Generic fallback for non-AArch64 or non-outlier
    var g_tile: [16]f32 = undefined;
    var u_tile: [16]f32 = undefined;
    gemvTileQuadRowCellDirect(g_cell, act, &g_tile, 0, use_sdot, false);
    gemvTileQuadRowCellDirect(u_cell, act, &u_tile, 0, use_sdot, false);
    mlp.swigluForward(&g_tile, &u_tile, out[row_base .. row_base + 16]);
}

pub const gemvTileGateUpFusedDirect = gemvTileQuadRowGateUpFusedCellDirect;
pub const gemvTileLMHead8RowWithMax = gemvTileQuadRowCellDirectWithMax;

pub fn gemvTileQuadRowCellDirectWithMax(
    cell: *const geometry.Cell,
    act: *const Activation2048,
    y: []f32,
    row_base: usize,
    comptime use_sdot: bool,
    w_max: *f32,
    w_argmax: *u32,
    w_finite: *bool,
) void {
    const payload = &cell.semantic_payload;
    const meta: *const weight_archive.TileMetadata = @ptrCast(@alignCast(payload.ptr));
    if (meta.quant_bits == 2) {
        gemvTileLMHeadW2WithMax(cell, act, y, row_base, w_max, w_argmax, w_finite);
        return;
    }

    if ((meta.custom_flags & weight_archive.FLAG_GROUP128_OUTLIERS) != 0 or meta.quant_bits == 4) {
        const group_scales_raw: [*]const f16 = @ptrCast(@alignCast(payload[48..560].ptr));
        const coded: [*]const u8 = @ptrCast(&cell.fingerprints);
        const has_outliers = (meta.custom_flags & weight_archive.FLAG_GROUP128_OUTLIERS) != 0;

        var x_outlier: @Vector(8, f32) = @splat(0.0);
        var outlier_w_raw: [*]const f16 = undefined;
        if (has_outliers) {
            outlier_w_raw = @ptrCast(@alignCast(payload[560..816].ptr));
            const outlier_cols_raw: [*]const u16 = @ptrCast(@alignCast(payload[816..832].ptr));
            inline for (0..8) |k| {
                const col_idx = outlier_cols_raw[k];
                x_outlier[k] = if ((col_idx & 1) == 0) act.even[col_idx / 2] else act.odd[col_idx / 2];
            }
        }

        const mask0f_16: @Vector(16, u8) = @splat(0x0F);
        const shift4_16: @Vector(16, u3) = @splat(4);

        if (comptime @import("builtin").cpu.arch == .aarch64 and use_sdot) {
            var r: usize = 0;
            while (r < 16) : (r += 8) {
                var row_dots: @Vector(8, f32) = @splat(0.0);
                const r0_bytes = coded[(r + 0) * 1024 .. (r + 1) * 1024];
                const r1_bytes = coded[(r + 1) * 1024 .. (r + 2) * 1024];
                const r2_bytes = coded[(r + 2) * 1024 .. (r + 3) * 1024];
                const r3_bytes = coded[(r + 3) * 1024 .. (r + 4) * 1024];
                const r4_bytes = coded[(r + 4) * 1024 .. (r + 5) * 1024];
                const r5_bytes = coded[(r + 5) * 1024 .. (r + 6) * 1024];
                const r6_bytes = coded[(r + 6) * 1024 .. (r + 7) * 1024];
                const r7_bytes = coded[(r + 7) * 1024 .. (r + 8) * 1024];

                for (0..16) |g| {
                    if (g + 1 < 16) {
                        const next_g = g + 1;
                        @prefetch(r0_bytes.ptr + next_g * 64, .{ .locality = 3, .cache = .data, .rw = .read });
                        @prefetch(r1_bytes.ptr + next_g * 64, .{ .locality = 3, .cache = .data, .rw = .read });
                        @prefetch(r2_bytes.ptr + next_g * 64, .{ .locality = 3, .cache = .data, .rw = .read });
                        @prefetch(r3_bytes.ptr + next_g * 64, .{ .locality = 3, .cache = .data, .rw = .read });
                        @prefetch(r4_bytes.ptr + next_g * 64, .{ .locality = 3, .cache = .data, .rw = .read });
                        @prefetch(r5_bytes.ptr + next_g * 64, .{ .locality = 3, .cache = .data, .rw = .read });
                        @prefetch(r6_bytes.ptr + next_g * 64, .{ .locality = 3, .cache = .data, .rw = .read });
                        @prefetch(r7_bytes.ptr + next_g * 64, .{ .locality = 3, .cache = .data, .rw = .read });
                    }
                    const s0: f32 = @floatCast(group_scales_raw[(r + 0) * 16 + g]);
                    const s1: f32 = @floatCast(group_scales_raw[(r + 1) * 16 + g]);
                    const s2: f32 = @floatCast(group_scales_raw[(r + 2) * 16 + g]);
                    const s3: f32 = @floatCast(group_scales_raw[(r + 3) * 16 + g]);
                    const s4: f32 = @floatCast(group_scales_raw[(r + 4) * 16 + g]);
                    const s5: f32 = @floatCast(group_scales_raw[(r + 5) * 16 + g]);
                    const s6: f32 = @floatCast(group_scales_raw[(r + 6) * 16 + g]);
                    const s7: f32 = @floatCast(group_scales_raw[(r + 7) * 16 + g]);

                    const g0_bytes = r0_bytes[g * 64 .. (g + 1) * 64];
                    const g1_bytes = r1_bytes[g * 64 .. (g + 1) * 64];
                    const g2_bytes = r2_bytes[g * 64 .. (g + 1) * 64];
                    const g3_bytes = r3_bytes[g * 64 .. (g + 1) * 64];
                    const g4_bytes = r4_bytes[g * 64 .. (g + 1) * 64];
                    const g5_bytes = r5_bytes[g * 64 .. (g + 1) * 64];
                    const g6_bytes = r6_bytes[g * 64 .. (g + 1) * 64];
                    const g7_bytes = r7_bytes[g * 64 .. (g + 1) * 64];

                    const q_even_g = act.q_even[g * 64 .. (g + 1) * 64];
                    const q_odd_g = act.q_odd[g * 64 .. (g + 1) * 64];
                    const act_scale = act.scales[g];

                    var acc0: @Vector(4, i32) = @splat(0);
                    var acc1: @Vector(4, i32) = @splat(0);
                    var acc2: @Vector(4, i32) = @splat(0);
                    var acc3: @Vector(4, i32) = @splat(0);
                    var acc4: @Vector(4, i32) = @splat(0);
                    var acc5: @Vector(4, i32) = @splat(0);
                    var acc6: @Vector(4, i32) = @splat(0);
                    var acc7: @Vector(4, i32) = @splat(0);

                    inline for (.{ 0, 16, 32, 48 }) |j| {
                        const xe: @Vector(16, i8) = q_even_g[j..][0..16].*;
                        const xo: @Vector(16, i8) = q_odd_g[j..][0..16].*;

                        const raw0: @Vector(16, u8) = g0_bytes[j..][0..16].*;
                        const low0: @Vector(16, i8) = @bitCast(raw0 & mask0f_16);
                        const high0: @Vector(16, i8) = @bitCast(raw0 >> shift4_16);

                        const raw1: @Vector(16, u8) = g1_bytes[j..][0..16].*;
                        const low1: @Vector(16, i8) = @bitCast(raw1 & mask0f_16);
                        const high1: @Vector(16, i8) = @bitCast(raw1 >> shift4_16);

                        const raw2: @Vector(16, u8) = g2_bytes[j..][0..16].*;
                        const low2: @Vector(16, i8) = @bitCast(raw2 & mask0f_16);
                        const high2: @Vector(16, i8) = @bitCast(raw2 >> shift4_16);

                        const raw3: @Vector(16, u8) = g3_bytes[j..][0..16].*;
                        const low3: @Vector(16, i8) = @bitCast(raw3 & mask0f_16);
                        const high3: @Vector(16, i8) = @bitCast(raw3 >> shift4_16);

                        asm volatile (
                            \\ sdot %[acc0].4s, %[low0].16b, %[xe].16b
                            \\ sdot %[acc1].4s, %[low1].16b, %[xe].16b
                            \\ sdot %[acc2].4s, %[low2].16b, %[xe].16b
                            \\ sdot %[acc3].4s, %[low3].16b, %[xe].16b
                            \\ sdot %[acc0].4s, %[high0].16b, %[xo].16b
                            \\ sdot %[acc1].4s, %[high1].16b, %[xo].16b
                            \\ sdot %[acc2].4s, %[high2].16b, %[xo].16b
                            \\ sdot %[acc3].4s, %[high3].16b, %[xo].16b
                            : [acc0] "+w" (acc0),
                              [acc1] "+w" (acc1),
                              [acc2] "+w" (acc2),
                              [acc3] "+w" (acc3),
                            : [xe] "w" (xe),
                              [xo] "w" (xo),
                              [low0] "w" (low0),
                              [high0] "w" (high0),
                              [low1] "w" (low1),
                              [high1] "w" (high1),
                              [low2] "w" (low2),
                              [high2] "w" (high2),
                              [low3] "w" (low3),
                              [high3] "w" (high3),
                        );

                        const raw4: @Vector(16, u8) = g4_bytes[j..][0..16].*;
                        const low4: @Vector(16, i8) = @bitCast(raw4 & mask0f_16);
                        const high4: @Vector(16, i8) = @bitCast(raw4 >> shift4_16);

                        const raw5: @Vector(16, u8) = g5_bytes[j..][0..16].*;
                        const low5: @Vector(16, i8) = @bitCast(raw5 & mask0f_16);
                        const high5: @Vector(16, i8) = @bitCast(raw5 >> shift4_16);

                        const raw6: @Vector(16, u8) = g6_bytes[j..][0..16].*;
                        const low6: @Vector(16, i8) = @bitCast(raw6 & mask0f_16);
                        const high6: @Vector(16, i8) = @bitCast(raw6 >> shift4_16);

                        const raw7: @Vector(16, u8) = g7_bytes[j..][0..16].*;
                        const low7: @Vector(16, i8) = @bitCast(raw7 & mask0f_16);
                        const high7: @Vector(16, i8) = @bitCast(raw7 >> shift4_16);

                        asm volatile (
                            \\ sdot %[acc4].4s, %[low4].16b, %[xe].16b
                            \\ sdot %[acc5].4s, %[low5].16b, %[xe].16b
                            \\ sdot %[acc6].4s, %[low6].16b, %[xe].16b
                            \\ sdot %[acc7].4s, %[low7].16b, %[xe].16b
                            \\ sdot %[acc4].4s, %[high4].16b, %[xo].16b
                            \\ sdot %[acc5].4s, %[high5].16b, %[xo].16b
                            \\ sdot %[acc6].4s, %[high6].16b, %[xo].16b
                            \\ sdot %[acc7].4s, %[high7].16b, %[xo].16b
                            : [acc4] "+w" (acc4),
                              [acc5] "+w" (acc5),
                              [acc6] "+w" (acc6),
                              [acc7] "+w" (acc7),
                            : [xe] "w" (xe),
                              [xo] "w" (xo),
                              [low4] "w" (low4),
                              [high4] "w" (high4),
                              [low5] "w" (low5),
                              [high5] "w" (high5),
                              [low6] "w" (low6),
                              [high6] "w" (high6),
                              [low7] "w" (low7),
                              [high7] "w" (high7),
                        );
                    }

                    const q_sum_8 = 8 * act.q_sums[g];
                    const int_dots: @Vector(8, i32) = .{
                        @as(i32, @intCast(@reduce(.Add, acc0))) - q_sum_8,
                        @as(i32, @intCast(@reduce(.Add, acc1))) - q_sum_8,
                        @as(i32, @intCast(@reduce(.Add, acc2))) - q_sum_8,
                        @as(i32, @intCast(@reduce(.Add, acc3))) - q_sum_8,
                        @as(i32, @intCast(@reduce(.Add, acc4))) - q_sum_8,
                        @as(i32, @intCast(@reduce(.Add, acc5))) - q_sum_8,
                        @as(i32, @intCast(@reduce(.Add, acc6))) - q_sum_8,
                        @as(i32, @intCast(@reduce(.Add, acc7))) - q_sum_8,
                    };
                    const scales: @Vector(8, f32) = .{ s0, s1, s2, s3, s4, s5, s6, s7 };
                    const act_scale_v: @Vector(8, f32) = @splat(act_scale);
                    row_dots += (@as(@Vector(8, f32), @floatFromInt(int_dots)) * act_scale_v) * scales;
                }

                var out_v: @Vector(8, f32) = @splat(0.0);
                if (has_outliers) {
                    var w_outlier0: @Vector(8, f32) = undefined;
                    var w_outlier1: @Vector(8, f32) = undefined;
                    var w_outlier2: @Vector(8, f32) = undefined;
                    var w_outlier3: @Vector(8, f32) = undefined;
                    var w_outlier4: @Vector(8, f32) = undefined;
                    var w_outlier5: @Vector(8, f32) = undefined;
                    var w_outlier6: @Vector(8, f32) = undefined;
                    var w_outlier7: @Vector(8, f32) = undefined;
                    inline for (0..8) |k| {
                        w_outlier0[k] = @floatCast(outlier_w_raw[(r + 0) * 8 + k]);
                        w_outlier1[k] = @floatCast(outlier_w_raw[(r + 1) * 8 + k]);
                        w_outlier2[k] = @floatCast(outlier_w_raw[(r + 2) * 8 + k]);
                        w_outlier3[k] = @floatCast(outlier_w_raw[(r + 3) * 8 + k]);
                        w_outlier4[k] = @floatCast(outlier_w_raw[(r + 4) * 8 + k]);
                        w_outlier5[k] = @floatCast(outlier_w_raw[(r + 5) * 8 + k]);
                        w_outlier6[k] = @floatCast(outlier_w_raw[(r + 6) * 8 + k]);
                        w_outlier7[k] = @floatCast(outlier_w_raw[(r + 7) * 8 + k]);
                    }
                    out_v = .{
                        @reduce(.Add, w_outlier0 * x_outlier),
                        @reduce(.Add, w_outlier1 * x_outlier),
                        @reduce(.Add, w_outlier2 * x_outlier),
                        @reduce(.Add, w_outlier3 * x_outlier),
                        @reduce(.Add, w_outlier4 * x_outlier),
                        @reduce(.Add, w_outlier5 * x_outlier),
                        @reduce(.Add, w_outlier6 * x_outlier),
                        @reduce(.Add, w_outlier7 * x_outlier),
                    };
                }

                const res = row_dots + out_v;
                const y_slice = y[row_base + r .. row_base + r + 8];
                y_slice[0..8].* = res;

                const m = @reduce(.Max, res);
                if (!std.math.isFinite(m)) w_finite.* = false;
                if (m > w_max.*) {
                    inline for (0..8) |offset| {
                        const idx: u32 = @intCast(row_base + r + offset);
                        if (idx < VOCAB_SIZE) {
                            const val = res[offset];
                            if (val > w_max.*) {
                                w_max.* = val;
                                w_argmax.* = idx;
                            }
                        }
                    }
                }
            }
            return;
        }

        var r: usize = 0;
        while (r < 16) : (r += 4) {
            var row_dots: @Vector(4, f32) = @splat(0.0);
            const r0_bytes = coded[(r + 0) * 1024 .. (r + 1) * 1024];
            const r1_bytes = coded[(r + 1) * 1024 .. (r + 2) * 1024];
            const r2_bytes = coded[(r + 2) * 1024 .. (r + 3) * 1024];
            const r3_bytes = coded[(r + 3) * 1024 .. (r + 4) * 1024];

            for (0..16) |g| {
                const s0: f32 = @floatCast(group_scales_raw[(r + 0) * 16 + g]);
                const s1: f32 = @floatCast(group_scales_raw[(r + 1) * 16 + g]);
                const s2: f32 = @floatCast(group_scales_raw[(r + 2) * 16 + g]);
                const s3: f32 = @floatCast(group_scales_raw[(r + 3) * 16 + g]);

                const g0_bytes = r0_bytes[g * 64 .. (g + 1) * 64];
                const g1_bytes = r1_bytes[g * 64 .. (g + 1) * 64];
                const g2_bytes = r2_bytes[g * 64 .. (g + 1) * 64];
                const g3_bytes = r3_bytes[g * 64 .. (g + 1) * 64];

                if (comptime use_sdot) {
                    const q_even_g = act.q_even[g * 64 .. (g + 1) * 64];
                    const q_odd_g = act.q_odd[g * 64 .. (g + 1) * 64];
                    const act_scale = act.scales[g];

                    var acc0_e: @Vector(4, i32) = @splat(0);
                    var acc0_o: @Vector(4, i32) = @splat(0);
                    var acc1_e: @Vector(4, i32) = @splat(0);
                    var acc1_o: @Vector(4, i32) = @splat(0);
                    var acc2_e: @Vector(4, i32) = @splat(0);
                    var acc2_o: @Vector(4, i32) = @splat(0);
                    var acc3_e: @Vector(4, i32) = @splat(0);
                    var acc3_o: @Vector(4, i32) = @splat(0);

                    inline for (.{ 0, 16, 32, 48 }) |j| {
                        const xe: @Vector(16, i8) = q_even_g[j..][0..16].*;
                        const xo: @Vector(16, i8) = q_odd_g[j..][0..16].*;

                        inline for (.{
                            .{ &acc0_e, &acc0_o, g0_bytes[j..][0..16].* },
                            .{ &acc1_e, &acc1_o, g1_bytes[j..][0..16].* },
                            .{ &acc2_e, &acc2_o, g2_bytes[j..][0..16].* },
                            .{ &acc3_e, &acc3_o, g3_bytes[j..][0..16].* },
                        }) |pair| {
                            const raw: @Vector(16, u8) = pair[2];
                            const low: @Vector(16, i8) = @bitCast(raw & mask0f_16);
                            const high: @Vector(16, i8) = @bitCast(raw >> shift4_16);
                            const l_16: @Vector(16, i16) = low;
                            const h_16: @Vector(16, i16) = high;
                            const xe_16: @Vector(16, i16) = xe;
                            const xo_16: @Vector(16, i16) = xo;
                            const pe: @Vector(16, i32) = @as(@Vector(16, i32), xe_16) * @as(@Vector(16, i32), l_16);
                            const po: @Vector(16, i32) = @as(@Vector(16, i32), xo_16) * @as(@Vector(16, i32), h_16);
                            pair[0].* += .{ pe[0] + pe[1] + pe[2] + pe[3], pe[4] + pe[5] + pe[6] + pe[7], pe[8] + pe[9] + pe[10] + pe[11], pe[12] + pe[13] + pe[14] + pe[15] };
                            pair[1].* += .{ po[0] + po[1] + po[2] + po[3], po[4] + po[5] + po[6] + po[7], po[8] + po[9] + po[10] + po[11], po[12] + po[13] + po[14] + po[15] };
                        }
                    }

                    const q_sum_8 = 8 * act.q_sums[g];
                    const int_dots: @Vector(4, i32) = .{
                        @as(i32, @intCast(@reduce(.Add, acc0_e + acc0_o))) - q_sum_8,
                        @as(i32, @intCast(@reduce(.Add, acc1_e + acc1_o))) - q_sum_8,
                        @as(i32, @intCast(@reduce(.Add, acc2_e + acc2_o))) - q_sum_8,
                        @as(i32, @intCast(@reduce(.Add, acc3_e + acc3_o))) - q_sum_8,
                    };
                    const scales: @Vector(4, f32) = .{ s0, s1, s2, s3 };
                    const act_scale_v: @Vector(4, f32) = @splat(act_scale);
                    row_dots += (@as(@Vector(4, f32), @floatFromInt(int_dots)) * act_scale_v) * scales;
                } else {
                    const x_even_g = act.even[g * 64 .. (g + 1) * 64];
                    const x_odd_g = act.odd[g * 64 .. (g + 1) * 64];
                    const mask0f_8: @Vector(8, u8) = @splat(0x0F);
                    const shift4_8: @Vector(8, u3) = @splat(4);

                    var d0_vec0: @Vector(8, f32) = @splat(0.0);
                    var d0_vec1: @Vector(8, f32) = @splat(0.0);
                    var d1_vec0: @Vector(8, f32) = @splat(0.0);
                    var d1_vec1: @Vector(8, f32) = @splat(0.0);
                    var d2_vec0: @Vector(8, f32) = @splat(0.0);
                    var d2_vec1: @Vector(8, f32) = @splat(0.0);
                    var d3_vec0: @Vector(8, f32) = @splat(0.0);
                    var d3_vec1: @Vector(8, f32) = @splat(0.0);

                    inline for (.{ 0, 16, 32, 48 }) |j| {
                        inline for (.{
                            .{ &d0_vec0, &d0_vec1, g0_bytes[j..][0..16].* },
                            .{ &d1_vec0, &d1_vec1, g1_bytes[j..][0..16].* },
                            .{ &d2_vec0, &d2_vec1, g2_bytes[j..][0..16].* },
                            .{ &d3_vec0, &d3_vec1, g3_bytes[j..][0..16].* },
                        }) |entry| {
                            const raw: @Vector(16, u8) = entry[2];
                            const low0: @Vector(8, u8) = @shuffle(u8, raw, undefined, @Vector(8, i32){ 0, 1, 2, 3, 4, 5, 6, 7 }) & mask0f_8;
                            const high0: @Vector(8, u8) = @shuffle(u8, raw, undefined, @Vector(8, i32){ 0, 1, 2, 3, 4, 5, 6, 7 }) >> shift4_8;
                            const low1: @Vector(8, u8) = @shuffle(u8, raw, undefined, @Vector(8, i32){ 8, 9, 10, 11, 12, 13, 14, 15 }) & mask0f_8;
                            const high1: @Vector(8, u8) = @shuffle(u8, raw, undefined, @Vector(8, i32){ 8, 9, 10, 11, 12, 13, 14, 15 }) >> shift4_8;

                            entry[0].* += (@as(@Vector(8, f32), @floatFromInt(low0)) * x_even_g[j..][0..8].*) +
                                          (@as(@Vector(8, f32), @floatFromInt(high0)) * x_odd_g[j..][0..8].*);
                            entry[1].* += (@as(@Vector(8, f32), @floatFromInt(low1)) * x_even_g[j + 8 ..][0..8].*) +
                                          (@as(@Vector(8, f32), @floatFromInt(high1)) * x_odd_g[j + 8 ..][0..8].*);
                        }
                    }

                    const dot_raw0_3: @Vector(4, f32) = .{
                        @reduce(.Add, d0_vec0 + d0_vec1),
                        @reduce(.Add, d1_vec0 + d1_vec1),
                        @reduce(.Add, d2_vec0 + d2_vec1),
                        @reduce(.Add, d3_vec0 + d3_vec1),
                    };
                    const s_vec0_3: @Vector(4, f32) = .{ s0, s1, s2, s3 };
                    const act_sum8: @Vector(4, f32) = @splat(8.0 * act.sums[g]);
                    row_dots += (dot_raw0_3 - act_sum8) * s_vec0_3;
                }
            }

            var out_v: @Vector(4, f32) = @splat(0.0);
            if (has_outliers) {
                var w_outlier0: @Vector(8, f32) = undefined;
                var w_outlier1: @Vector(8, f32) = undefined;
                var w_outlier2: @Vector(8, f32) = undefined;
                var w_outlier3: @Vector(8, f32) = undefined;
                inline for (0..8) |k| {
                    w_outlier0[k] = @floatCast(outlier_w_raw[(r + 0) * 8 + k]);
                    w_outlier1[k] = @floatCast(outlier_w_raw[(r + 1) * 8 + k]);
                    w_outlier2[k] = @floatCast(outlier_w_raw[(r + 2) * 8 + k]);
                    w_outlier3[k] = @floatCast(outlier_w_raw[(r + 3) * 8 + k]);
                }
                out_v = .{
                    @reduce(.Add, w_outlier0 * x_outlier),
                    @reduce(.Add, w_outlier1 * x_outlier),
                    @reduce(.Add, w_outlier2 * x_outlier),
                    @reduce(.Add, w_outlier3 * x_outlier),
                };
            }

            const res = row_dots + out_v;
            const y_slice = y[row_base + r .. row_base + r + 4];
            y_slice[0..4].* = res;

            const m = @reduce(.Max, res);
            if (!std.math.isFinite(m)) w_finite.* = false;
            if (m > w_max.*) {
                inline for (0..4) |offset| {
                    const idx: u32 = @intCast(row_base + r + offset);
                    if (idx < VOCAB_SIZE) {
                        const val = res[offset];
                        if (val > w_max.*) {
                            w_max.* = val;
                            w_argmax.* = idx;
                        }
                    }
                }
            }
        }
        return;
    }

    const scale: f32 = @bitCast(std.mem.readInt(u32, payload[32..36], .little));
    const bias: f32 = @bitCast(std.mem.readInt(u32, payload[36..40], .little));
    const coded: [*]const u8 = @ptrCast(&cell.fingerprints);
    const sum_x = @reduce(.Add, @as(@Vector(16, f32), act.sums));

    for (0..16) |r| {
        const row_bytes = coded[r * 1024 .. (r + 1) * 1024];
        var dot_vec: @Vector(8, f32) = @splat(0.0);
        var j: usize = 0;
        while (j + 4 <= 1024) : (j += 4) {
            const b0 = row_bytes[j];
            const b1 = row_bytes[j + 1];
            const b2 = row_bytes[j + 2];
            const b3 = row_bytes[j + 3];

            const q_vec: @Vector(8, f32) = .{
                @as(f32, @floatFromInt(b0 & 0x0F)) - 8.0,
                @as(f32, @floatFromInt(b0 >> 4)) - 8.0,
                @as(f32, @floatFromInt(b1 & 0x0F)) - 8.0,
                @as(f32, @floatFromInt(b1 >> 4)) - 8.0,
                @as(f32, @floatFromInt(b2 & 0x0F)) - 8.0,
                @as(f32, @floatFromInt(b2 >> 4)) - 8.0,
                @as(f32, @floatFromInt(b3 & 0x0F)) - 8.0,
                @as(f32, @floatFromInt(b3 >> 4)) - 8.0,
            };
            const x_vec: @Vector(8, f32) = act.raw[2 * j ..][0..8].*;
            dot_vec += q_vec * x_vec;
        }
        const val = @reduce(.Add, dot_vec) * scale + sum_x * bias;
        y[row_base + r] = val;
        if (!std.math.isFinite(val)) w_finite.* = false;
        if (val > w_max.*) {
            w_max.* = val;
            w_argmax.* = @intCast(row_base + r);
        }
    }
}

pub fn gemvTileW8CellDirect(cell: *const geometry.Cell, act: *const Activation2048, y: []f32, row_base: usize, comptime accumulate: bool) void {
    const payload = &cell.semantic_payload;
    const group_scales_raw: [*]const f16 = @ptrCast(@alignCast(payload[48..304].ptr));
    const coded: [*]const u8 = @ptrCast(&cell.fingerprints);

    for (0..8) |r| {
        var row_sum: f32 = 0.0;
        const row_bytes = coded[r * 2048 .. (r + 1) * 2048];
        for (0..16) |g| {
            const scale: f32 = @floatCast(group_scales_raw[r * 16 + g]);
            const g_bytes = row_bytes[g * 128 .. (g + 1) * 128];
            const x_slice = act.raw[g * 128 .. (g + 1) * 128];

            var dot_v0: @Vector(8, f32) = @splat(0.0);
            var dot_v1: @Vector(8, f32) = @splat(0.0);
            var j: usize = 0;
            while (j < 128) : (j += 16) {
                const w8_0: [8]i8 = @bitCast(g_bytes[j..][0..8].*);
                const w8_1: [8]i8 = @bitCast(g_bytes[j + 8..][0..8].*);
                const wf0: @Vector(8, f32) = .{
                    @floatFromInt(w8_0[0]), @floatFromInt(w8_0[1]), @floatFromInt(w8_0[2]), @floatFromInt(w8_0[3]),
                    @floatFromInt(w8_0[4]), @floatFromInt(w8_0[5]), @floatFromInt(w8_0[6]), @floatFromInt(w8_0[7]),
                };
                const wf1: @Vector(8, f32) = .{
                    @floatFromInt(w8_1[0]), @floatFromInt(w8_1[1]), @floatFromInt(w8_1[2]), @floatFromInt(w8_1[3]),
                    @floatFromInt(w8_1[4]), @floatFromInt(w8_1[5]), @floatFromInt(w8_1[6]), @floatFromInt(w8_1[7]),
                };
                const xf0: @Vector(8, f32) = x_slice[j..][0..8].*;
                const xf1: @Vector(8, f32) = x_slice[j + 8..][0..8].*;
                dot_v0 += wf0 * xf0;
                dot_v1 += wf1 * xf1;
            }
            row_sum += @reduce(.Add, dot_v0 + dot_v1) * scale;
        }
        if (comptime accumulate) {
            y[row_base + r] += row_sum;
        } else {
            y[row_base + r] = row_sum;
        }
    }
}

pub fn gemvTileGateUpFusedW8Direct(
    g_cell: *const geometry.Cell,
    u_cell: *const geometry.Cell,
    act: *const Activation2048,
    out: []f32,
    row_base: usize,
) void {
    var g_tile: [8]f32 = undefined;
    var u_tile: [8]f32 = undefined;
    gemvTileW8CellDirect(g_cell, act, &g_tile, 0, false);
    gemvTileW8CellDirect(u_cell, act, &u_tile, 0, false);
    mlp.swigluForward(&g_tile, &u_tile, out[row_base .. row_base + 8]);
}

pub fn gemvTileDownW8(cell: *const geometry.Cell, act: *const Activation11008, y: []f32, tile_idx: usize) void {
    const payload = &cell.semantic_payload;
    const group_scales_raw: [*]const f16 = @ptrCast(@alignCast(payload[48..304].ptr));
    const coded: [*]const u8 = @ptrCast(&cell.fingerprints);

    const global_k_start: usize = tile_idx * 16384;
    var r: usize = global_k_start / INTERMEDIATE_DIM;
    var c: usize = global_k_start % INTERMEDIATE_DIM;

    var row_accum: f32 = 0.0;
    for (0..128) |g| {
        const scale: f32 = @floatCast(group_scales_raw[g]);
        const g_bytes = coded[g * 128 .. (g + 1) * 128];
        const x_slice = act.raw[c .. c + 128];

        var dot_v0: @Vector(8, f32) = @splat(0.0);
        var dot_v1: @Vector(8, f32) = @splat(0.0);
        var j: usize = 0;
        while (j < 128) : (j += 16) {
            const w8_0: [8]i8 = @bitCast(g_bytes[j..][0..8].*);
            const w8_1: [8]i8 = @bitCast(g_bytes[j + 8..][0..8].*);
            const wf0: @Vector(8, f32) = .{
                @floatFromInt(w8_0[0]), @floatFromInt(w8_0[1]), @floatFromInt(w8_0[2]), @floatFromInt(w8_0[3]),
                @floatFromInt(w8_0[4]), @floatFromInt(w8_0[5]), @floatFromInt(w8_0[6]), @floatFromInt(w8_0[7]),
            };
            const wf1: @Vector(8, f32) = .{
                @floatFromInt(w8_1[0]), @floatFromInt(w8_1[1]), @floatFromInt(w8_1[2]), @floatFromInt(w8_1[3]),
                @floatFromInt(w8_1[4]), @floatFromInt(w8_1[5]), @floatFromInt(w8_1[6]), @floatFromInt(w8_1[7]),
            };
            const xf0: @Vector(8, f32) = x_slice[j..][0..8].*;
            const xf1: @Vector(8, f32) = x_slice[j + 8..][0..8].*;
            dot_v0 += wf0 * xf0;
            dot_v1 += wf1 * xf1;
        }
        row_accum += @reduce(.Add, dot_v0 + dot_v1) * scale;

        c += 128;
        if (c == INTERMEDIATE_DIM) {
            if (r < y.len) {
                y[r] += row_accum;
            }
            row_accum = 0.0;
            c = 0;
            r += 1;
        }
    }
    if (row_accum != 0.0 and r < y.len) {
        y[r] += row_accum;
    }
}

pub fn gemvTileLMHeadW8WithMax(
    cell: *const geometry.Cell,
    act: *const Activation2048,
    logits: []f32,
    base_row: usize,
    max_val: *f32,
    argmax: *u32,
    finite: *bool,
) void {
    gemvTileW8CellDirect(cell, act, logits, base_row, false);
    inline for (0..8) |offset| {
        const idx: u32 = @intCast(base_row + offset);
        if (idx < VOCAB_SIZE) {
            const val = logits[idx];
            if (!std.math.isFinite(val)) {
                finite.* = false;
            }
            if (val > max_val.*) {
                max_val.* = val;
                argmax.* = idx;
            }
        }
    }
}

pub fn gemvTileW2CellDirect(cell: *const geometry.Cell, act: *const Activation2048, y: []f32, row_base: usize, comptime accumulate: bool) void {
    const payload = &cell.semantic_payload;
    const meta: *const weight_archive.TileMetadata = @ptrCast(@alignCast(payload.ptr));
    const group_scales_raw: [*]const f16 = @ptrCast(@alignCast(payload[48..560].ptr));
    const coded: [*]const u8 = @ptrCast(&cell.fingerprints);
    const has_outliers = (meta.custom_flags & weight_archive.FLAG_GROUP128_OUTLIERS) != 0;

    var x_outlier: @Vector(8, f32) = @splat(0.0);
    var outlier_w_raw: [*]const f16 = undefined;
    if (has_outliers) {
        outlier_w_raw = @ptrCast(@alignCast(payload[560..816].ptr));
        const outlier_cols_raw: [*]const u16 = @ptrCast(@alignCast(payload[816..832].ptr));
        inline for (0..8) |k| {
            const col_idx = outlier_cols_raw[k];
            x_outlier[k] = if ((col_idx & 1) == 0) act.even[col_idx / 2] else act.odd[col_idx / 2];
        }
    }

    const LUT: [4]f32 = .{ 0.0, 1.0, -2.0, -1.0 };

    for (0..16) |r| {
        var row_sum: f32 = 0.0;
        const row_bytes = coded[r * 512 .. (r + 1) * 512];
        for (0..16) |g| {
            const scale: f32 = @floatCast(group_scales_raw[r * 16 + g]);
            const g_bytes = row_bytes[g * 32 .. (g + 1) * 32];
            const x_slice = act.raw[g * 128 .. (g + 1) * 128];

            var dot_v0: @Vector(8, f32) = @splat(0.0);
            var dot_v1: @Vector(8, f32) = @splat(0.0);
            var j: usize = 0;
            while (j < 32) : (j += 4) {
                const b0 = g_bytes[j + 0];
                const b1 = g_bytes[j + 1];
                const b2 = g_bytes[j + 2];
                const b3 = g_bytes[j + 3];

                const qv0: @Vector(8, f32) = .{
                    LUT[b0 & 0x03],
                    LUT[(b0 >> 2) & 0x03],
                    LUT[(b0 >> 4) & 0x03],
                    LUT[(b0 >> 6) & 0x03],
                    LUT[b1 & 0x03],
                    LUT[(b1 >> 2) & 0x03],
                    LUT[(b1 >> 4) & 0x03],
                    LUT[(b1 >> 6) & 0x03],
                };
                const qv1: @Vector(8, f32) = .{
                    LUT[b2 & 0x03],
                    LUT[(b2 >> 2) & 0x03],
                    LUT[(b2 >> 4) & 0x03],
                    LUT[(b2 >> 6) & 0x03],
                    LUT[b3 & 0x03],
                    LUT[(b3 >> 2) & 0x03],
                    LUT[(b3 >> 4) & 0x03],
                    LUT[(b3 >> 6) & 0x03],
                };
                const xv0: @Vector(8, f32) = x_slice[4 * j ..][0..8].*;
                const xv1: @Vector(8, f32) = x_slice[4 * j + 8 ..][0..8].*;
                dot_v0 += qv0 * xv0;
                dot_v1 += qv1 * xv1;
            }
            row_sum += @reduce(.Add, dot_v0 + dot_v1) * scale;
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

pub fn gemvTileLMHeadW2WithMax(
    cell: *const geometry.Cell,
    act: *const Activation2048,
    logits: []f32,
    base_row: usize,
    max_val: *f32,
    argmax: *u32,
    finite: *bool,
) void {
    gemvTileW2CellDirect(cell, act, logits, base_row, false);
    inline for (0..16) |offset| {
        const idx: u32 = @intCast(base_row + offset);
        if (idx < VOCAB_SIZE) {
            const val = logits[idx];
            if (!std.math.isFinite(val)) {
                finite.* = false;
            }
            if (val > max_val.*) {
                max_val.* = val;
                argmax.* = idx;
            }
        }
    }
}

pub fn gemvTileDownW2(cell: *const geometry.Cell, act: *const Activation11008, y: []f32, tile_idx: usize) void {
    const payload = &cell.semantic_payload;
    const group_scales_raw: [*]const f16 = @ptrCast(@alignCast(payload[48..560].ptr));
    const num_outliers: usize = std.mem.readInt(u16, payload[560..562], .little);
    const outlier_w_raw: [*]const f16 = @ptrCast(@alignCast(payload[562..690].ptr));
    const outlier_offsets_raw: [*]const u16 = @ptrCast(@alignCast(payload[690..818].ptr));
    const coded: [*]const u8 = @ptrCast(&cell.fingerprints);

    const LUT: [4]f32 = .{ 0.0, 1.0, -2.0, -1.0 };

    const global_k_start: usize = tile_idx * 32768;
    var r: usize = global_k_start / INTERMEDIATE_DIM;
    var c: usize = global_k_start % INTERMEDIATE_DIM;

    var row_accum: f32 = 0.0;
    for (0..256) |g| {
        const scale: f32 = @floatCast(group_scales_raw[g]);
        const g_bytes = coded[g * 32 .. (g + 1) * 32];
        const x_slice = act.raw[c .. c + 128];

        var dot_v0: @Vector(8, f32) = @splat(0.0);
        var dot_v1: @Vector(8, f32) = @splat(0.0);
        var j: usize = 0;
        while (j < 32) : (j += 4) {
            const b0 = g_bytes[j + 0];
            const b1 = g_bytes[j + 1];
            const b2 = g_bytes[j + 2];
            const b3 = g_bytes[j + 3];

            const qv0: @Vector(8, f32) = .{
                LUT[b0 & 0x03],
                LUT[(b0 >> 2) & 0x03],
                LUT[(b0 >> 4) & 0x03],
                LUT[(b0 >> 6) & 0x03],
                LUT[b1 & 0x03],
                LUT[(b1 >> 2) & 0x03],
                LUT[(b1 >> 4) & 0x03],
                LUT[(b1 >> 6) & 0x03],
            };
            const qv1: @Vector(8, f32) = .{
                LUT[b2 & 0x03],
                LUT[(b2 >> 2) & 0x03],
                LUT[(b2 >> 4) & 0x03],
                LUT[(b2 >> 6) & 0x03],
                LUT[b3 & 0x03],
                LUT[(b3 >> 2) & 0x03],
                LUT[(b3 >> 4) & 0x03],
                LUT[(b3 >> 6) & 0x03],
            };
            const xv0: @Vector(8, f32) = x_slice[4 * j ..][0..8].*;
            const xv1: @Vector(8, f32) = x_slice[4 * j + 8 ..][0..8].*;
            dot_v0 += qv0 * xv0;
            dot_v1 += qv1 * xv1;
        }
        row_accum += @reduce(.Add, dot_v0 + dot_v1) * scale;
        c += 128;
        if (c == INTERMEDIATE_DIM) {
            if (r < y.len) {
                y[r] += row_accum;
            }
            row_accum = 0.0;
            c = 0;
            r += 1;
        }
    }
    if (row_accum != 0.0 and r < y.len) {
        y[r] += row_accum;
    }

    for (0..num_outliers) |i| {
        const w_idx: usize = outlier_offsets_raw[i];
        const glob_k = global_k_start + w_idx;
        const r_out = glob_k / INTERMEDIATE_DIM;
        const c_out = glob_k - r_out * INTERMEDIATE_DIM;
        const w_val: f32 = @floatCast(outlier_w_raw[i]);
        if (r_out < y.len) {
            y[r_out] += w_val * act.raw[c_out];
        }
    }
}

/// Computes dot products of x (dim 2048) against all 16 rows in rec, accumulating into y[row_base..row_base+16].
pub fn gemvTile(rec: *const geometry.Record, x: []const f32, y: []f32, row_base: usize) void {
    gemvTileOpt(rec, x, y, row_base, null);
}

pub fn gemvTileOpt(rec: *const geometry.Record, x: []const f32, y: []f32, row_base: usize, x_sums: ?*const [16]f32) void {
    _ = x_sums;
    const act = Activation2048.init(x);
    gemvTileQuadRow(rec, &act, y, row_base);
}

pub fn gemvTileDownDeintCellOpt(cell: *const geometry.Cell, act: *const Activation11008, y: []f32, tile_idx: usize, comptime use_sdot: bool) void {
    const payload = &cell.semantic_payload;
    const meta: *const weight_archive.TileMetadata = @ptrCast(@alignCast(payload.ptr));

    if (meta.quant_bits == 2) {
        gemvTileDownW2(cell, act, y, tile_idx);
        return;
    }

    if ((meta.custom_flags & weight_archive.FLAG_GROUP128_OUTLIERS) != 0) {
        const group_scales_raw: [*]const f16 = @ptrCast(@alignCast(payload[48..560].ptr));
        const num_outliers: usize = std.mem.readInt(u16, payload[560..562], .little);
        const outlier_w_raw: [*]const f16 = @ptrCast(@alignCast(payload[562..690].ptr));
        const outlier_offsets_raw: [*]const u16 = @ptrCast(@alignCast(payload[690..818].ptr));
        const coded: [*]const u8 = @ptrCast(&cell.fingerprints);

        const global_k_start: usize = tile_idx * 32768;
        var r: usize = global_k_start / INTERMEDIATE_DIM;
        var c: usize = global_k_start % INTERMEDIATE_DIM;
        var group_in_row: usize = c / 128;

        const mask0f_16: @Vector(16, u8) = @splat(0x0F);
        const shift4_16: @Vector(16, u3) = @splat(4);

        var row_accum: f32 = 0.0;
        for (0..256) |g| {
            if (g + 2 < 256) {
                @prefetch(coded + (g + 2) * 64, .{ .locality = 3, .cache = .data, .rw = .read });
            }
            const scale: f32 = @floatCast(group_scales_raw[g]);
            const g_bytes = coded[g * 64 .. (g + 1) * 64];
            const c_half = c / 2;

            if (comptime use_sdot) {
                const q_even_g = act.q_even[c_half .. c_half + 64];
                const q_odd_g = act.q_odd[c_half .. c_half + 64];
                const act_scale = act.scales[group_in_row];

                var acc_e0: @Vector(4, i32) = @splat(0);
                var acc_e1: @Vector(4, i32) = @splat(0);
                var acc_o0: @Vector(4, i32) = @splat(0);
                var acc_o1: @Vector(4, i32) = @splat(0);

                const xe0: @Vector(16, i8) = q_even_g[0..16].*;
                const xo0: @Vector(16, i8) = q_odd_g[0..16].*;
                const raw0: @Vector(16, u8) = g_bytes[0..16].*;
                const low0 = raw0 & mask0f_16;
                const high0 = raw0 >> shift4_16;

                const xe1: @Vector(16, i8) = q_even_g[16..32].*;
                const xo1: @Vector(16, i8) = q_odd_g[16..32].*;
                const raw1: @Vector(16, u8) = g_bytes[16..32].*;
                const low1 = raw1 & mask0f_16;
                const high1 = raw1 >> shift4_16;

                const xe2: @Vector(16, i8) = q_even_g[32..48].*;
                const xo2: @Vector(16, i8) = q_odd_g[32..48].*;
                const raw2: @Vector(16, u8) = g_bytes[32..48].*;
                const low2 = raw2 & mask0f_16;
                const high2 = raw2 >> shift4_16;

                const xe3: @Vector(16, i8) = q_even_g[48..64].*;
                const xo3: @Vector(16, i8) = q_odd_g[48..64].*;
                const raw3: @Vector(16, u8) = g_bytes[48..64].*;
                const low3 = raw3 & mask0f_16;
                const high3 = raw3 >> shift4_16;

                if (comptime @import("builtin").cpu.arch == .aarch64) {
                    asm volatile (
                        \\ sdot %[acc_e0].4s, %[low0].16b, %[xe0].16b
                        \\ sdot %[acc_o0].4s, %[high0].16b, %[xo0].16b
                        \\ sdot %[acc_e1].4s, %[low1].16b, %[xe1].16b
                        \\ sdot %[acc_o1].4s, %[high1].16b, %[xo1].16b
                        \\ sdot %[acc_e0].4s, %[low2].16b, %[xe2].16b
                        \\ sdot %[acc_o0].4s, %[high2].16b, %[xo2].16b
                        \\ sdot %[acc_e1].4s, %[low3].16b, %[xe3].16b
                        \\ sdot %[acc_o1].4s, %[high3].16b, %[xo3].16b
                        : [acc_e0] "+w" (acc_e0),
                          [acc_o0] "+w" (acc_o0),
                          [acc_e1] "+w" (acc_e1),
                          [acc_o1] "+w" (acc_o1),
                        : [xe0] "w" (xe0),
                          [xo0] "w" (xo0),
                          [low0] "w" (low0),
                          [high0] "w" (high0),
                          [xe1] "w" (xe1),
                          [xo1] "w" (xo1),
                          [low1] "w" (low1),
                          [high1] "w" (high1),
                          [xe2] "w" (xe2),
                          [xo2] "w" (xo2),
                          [low2] "w" (low2),
                          [high2] "w" (high2),
                          [xe3] "w" (xe3),
                          [xo3] "w" (xo3),
                          [low3] "w" (low3),
                          [high3] "w" (high3),
                    );
                } else {
                    inline for (.{ .{ low0, high0, xe0, xo0 }, .{ low2, high2, xe2, xo2 } }) |pair| {
                        const l_i8: @Vector(16, i8) = @bitCast(pair[0]);
                        const h_i8: @Vector(16, i8) = @bitCast(pair[1]);
                        const l_16: @Vector(16, i16) = l_i8;
                        const h_16: @Vector(16, i16) = h_i8;
                        const xe_16: @Vector(16, i16) = pair[2];
                        const xo_16: @Vector(16, i16) = pair[3];
                        const pe: @Vector(16, i32) = @as(@Vector(16, i32), xe_16) * @as(@Vector(16, i32), l_16);
                        const po: @Vector(16, i32) = @as(@Vector(16, i32), xo_16) * @as(@Vector(16, i32), h_16);
                        acc_e0 += .{ pe[0] + pe[1] + pe[2] + pe[3], pe[4] + pe[5] + pe[6] + pe[7], pe[8] + pe[9] + pe[10] + pe[11], pe[12] + pe[13] + pe[14] + pe[15] };
                        acc_o0 += .{ po[0] + po[1] + po[2] + po[3], po[4] + po[5] + po[6] + po[7], po[8] + po[9] + po[10] + po[11], po[12] + po[13] + po[14] + po[15] };
                    }
                    inline for (.{ .{ low1, high1, xe1, xo1 }, .{ low3, high3, xe3, xo3 } }) |pair| {
                        const l_i8: @Vector(16, i8) = @bitCast(pair[0]);
                        const h_i8: @Vector(16, i8) = @bitCast(pair[1]);
                        const l_16: @Vector(16, i16) = l_i8;
                        const h_16: @Vector(16, i16) = h_i8;
                        const xe_16: @Vector(16, i16) = pair[2];
                        const xo_16: @Vector(16, i16) = pair[3];
                        const pe: @Vector(16, i32) = @as(@Vector(16, i32), xe_16) * @as(@Vector(16, i32), l_16);
                        const po: @Vector(16, i32) = @as(@Vector(16, i32), xo_16) * @as(@Vector(16, i32), h_16);
                        acc_e1 += .{ pe[0] + pe[1] + pe[2] + pe[3], pe[4] + pe[5] + pe[6] + pe[7], pe[8] + pe[9] + pe[10] + pe[11], pe[12] + pe[13] + pe[14] + pe[15] };
                        acc_o1 += .{ po[0] + po[1] + po[2] + po[3], po[4] + po[5] + po[6] + po[7], po[8] + po[9] + po[10] + po[11], po[12] + po[13] + po[14] + po[15] };
                    }
                }

                const acc_e = acc_e0 + acc_e1;
                const acc_o = acc_o0 + acc_o1;
                const q_sum_8 = 8 * act.q_sums[group_in_row];
                const int_dot: i32 = @as(i32, @intCast(@reduce(.Add, acc_e + acc_o))) - q_sum_8;
                row_accum += @as(f32, @floatFromInt(int_dot)) * (act_scale * scale);
            } else {
                const x_even_g = act.even[c_half .. c_half + 64];
                const x_odd_g = act.odd[c_half .. c_half + 64];
                const mask0f_8: @Vector(8, u8) = @splat(0x0F);
                const shift4_8: @Vector(8, u3) = @splat(4);

                var dot_vec0: @Vector(8, f32) = @splat(0.0);
                var dot_vec1: @Vector(8, f32) = @splat(0.0);
                var dot_vec2: @Vector(8, f32) = @splat(0.0);
                var dot_vec3: @Vector(8, f32) = @splat(0.0);

                inline for (.{ 0, 16, 32, 48 }) |j| {
                    const raw0: @Vector(8, u8) = g_bytes[j..][0..8].*;
                    const low0 = raw0 & mask0f_8;
                    const high0 = raw0 >> shift4_8;
                    const q_vec0: @Vector(8, f32) = @floatFromInt(low0);
                    const q_vec1: @Vector(8, f32) = @floatFromInt(high0);
                    const x_vec0: @Vector(8, f32) = x_even_g[j..][0..8].*;
                    const x_vec1: @Vector(8, f32) = x_odd_g[j..][0..8].*;
                    dot_vec0 += q_vec0 * x_vec0;
                    dot_vec1 += q_vec1 * x_vec1;

                    const raw1: @Vector(8, u8) = g_bytes[j + 8 ..][0..8].*;
                    const low1 = raw1 & mask0f_8;
                    const high1 = raw1 >> shift4_8;
                    const q_vec2: @Vector(8, f32) = @floatFromInt(low1);
                    const q_vec3: @Vector(8, f32) = @floatFromInt(high1);
                    const x_vec2: @Vector(8, f32) = x_even_g[j + 8 ..][0..8].*;
                    const x_vec3: @Vector(8, f32) = x_odd_g[j + 8 ..][0..8].*;
                    dot_vec2 += q_vec2 * x_vec2;
                    dot_vec3 += q_vec3 * x_vec3;
                }
                const dot_raw = @reduce(.Add, (dot_vec0 + dot_vec1) + (dot_vec2 + dot_vec3));
                row_accum += (dot_raw - 8.0 * act.sums[group_in_row]) * scale;
            }

            c += 128;
            group_in_row += 1;
            if (c == INTERMEDIATE_DIM) {
                if (r < y.len) {
                    y[r] += row_accum;
                }
                row_accum = 0.0;
                c = 0;
                r += 1;
                group_in_row = 0;
            }
        }
        if (row_accum != 0.0 and r < y.len) {
            y[r] += row_accum;
        }

        for (0..num_outliers) |i| {
            const w_idx: usize = outlier_offsets_raw[i];
            const glob_k = global_k_start + w_idx;
            const r_out = glob_k / INTERMEDIATE_DIM;
            const c_out = glob_k - r_out * INTERMEDIATE_DIM;
            const w_val: f32 = @floatCast(outlier_w_raw[i]);
            if (r_out < y.len) {
                y[r_out] += w_val * act.raw[c_out];
            }
        }
        return;
    }

    const scale: f32 = @bitCast(std.mem.readInt(u32, payload[32..36], .little));
    const bias: f32 = @bitCast(std.mem.readInt(u32, payload[36..40], .little));

    const coded: [*]const u8 = @ptrCast(&cell.fingerprints);
    const global_k_start: usize = tile_idx * 32768;
    var r: usize = global_k_start / INTERMEDIATE_DIM;
    var c: usize = global_k_start % INTERMEDIATE_DIM;

    var w_idx: usize = 0;
    while (w_idx < 32768) {
        const span_len = @min(32768 - w_idx, INTERMEDIATE_DIM - c);

        var dot_vec: @Vector(8, f32) = @splat(0.0);
        var sum_x_vec: @Vector(8, f32) = @splat(0.0);

        var k: usize = 0;
        while (k + 8 <= span_len) : (k += 8) {
            const byte_idx = (w_idx + k) / 2;
            const b0 = coded[byte_idx];
            const b1 = coded[byte_idx + 1];
            const b2 = coded[byte_idx + 2];
            const b3 = coded[byte_idx + 3];

            const q_vec: @Vector(8, f32) = .{
                @as(f32, @floatFromInt(b0 & 0x0F)) - 8.0,
                @as(f32, @floatFromInt(b0 >> 4)) - 8.0,
                @as(f32, @floatFromInt(b1 & 0x0F)) - 8.0,
                @as(f32, @floatFromInt(b1 >> 4)) - 8.0,
                @as(f32, @floatFromInt(b2 & 0x0F)) - 8.0,
                @as(f32, @floatFromInt(b2 >> 4)) - 8.0,
                @as(f32, @floatFromInt(b3 & 0x0F)) - 8.0,
                @as(f32, @floatFromInt(b3 >> 4)) - 8.0,
            };
            const x_vec: @Vector(8, f32) = act.raw[c + k ..][0..8].*;
            dot_vec += q_vec * x_vec;
            sum_x_vec += x_vec;
        }

        const dot_acc = @reduce(.Add, dot_vec);
        const sum_x_acc = @reduce(.Add, sum_x_vec);
        if (r < y.len) {
            y[r] += dot_acc * scale + sum_x_acc * bias;
        }

        w_idx += span_len;
        c += span_len;
        if (c == INTERMEDIATE_DIM) {
            c = 0;
            r += 1;
        }
    }
}

pub fn gemvTileDownDeintCell8Acc(cell: *const geometry.Cell, act: *const Activation11008, y: []f32, tile_idx: usize, comptime use_sdot: bool) void {
    const payload = &cell.semantic_payload;
    const meta: *const weight_archive.TileMetadata = @ptrCast(@alignCast(payload.ptr));

    if ((meta.custom_flags & weight_archive.FLAG_GROUP128_OUTLIERS) != 0) {
        const group_scales_raw: [*]const f16 = @ptrCast(@alignCast(payload[48..560].ptr));
        const num_outliers: usize = std.mem.readInt(u16, payload[560..562], .little);
        const outlier_w_raw: [*]const f16 = @ptrCast(@alignCast(payload[562..690].ptr));
        const outlier_offsets_raw: [*]const u16 = @ptrCast(@alignCast(payload[690..818].ptr));
        const coded: [*]const u8 = @ptrCast(&cell.fingerprints);

        const global_k_start: usize = tile_idx * 32768;
        var r: usize = global_k_start / INTERMEDIATE_DIM;
        var c: usize = global_k_start % INTERMEDIATE_DIM;
        var group_in_row: usize = c / 128;

        const mask0f_16: @Vector(16, u8) = @splat(0x0F);
        const shift4_16: @Vector(16, u3) = @splat(4);

        var row_accum: f32 = 0.0;
        for (0..256) |g| {
            if (g + 2 < 256) {
                @prefetch(coded + (g + 2) * 64, .{ .locality = 3, .cache = .data, .rw = .read });
            }
            const scale: f32 = @floatCast(group_scales_raw[g]);
            const g_bytes = coded[g * 64 .. (g + 1) * 64];
            const c_half = c / 2;

            if (comptime use_sdot) {
                const q_even_g = act.q_even[c_half .. c_half + 64];
                const q_odd_g = act.q_odd[c_half .. c_half + 64];
                const act_scale = act.scales[group_in_row];

                var acc_e: @Vector(4, i32) = @splat(0);
                var acc_o: @Vector(4, i32) = @splat(0);

                const xe0: @Vector(16, i8) = q_even_g[0..16].*;
                const xo0: @Vector(16, i8) = q_odd_g[0..16].*;
                const raw0: @Vector(16, u8) = g_bytes[0..16].*;
                const low0 = raw0 & mask0f_16;
                const high0 = raw0 >> shift4_16;

                const xe1: @Vector(16, i8) = q_even_g[16..32].*;
                const xo1: @Vector(16, i8) = q_odd_g[16..32].*;
                const raw1: @Vector(16, u8) = g_bytes[16..32].*;
                const low1 = raw1 & mask0f_16;
                const high1 = raw1 >> shift4_16;

                const xe2: @Vector(16, i8) = q_even_g[32..48].*;
                const xo2: @Vector(16, i8) = q_odd_g[32..48].*;
                const raw2: @Vector(16, u8) = g_bytes[32..48].*;
                const low2 = raw2 & mask0f_16;
                const high2 = raw2 >> shift4_16;

                const xe3: @Vector(16, i8) = q_even_g[48..64].*;
                const xo3: @Vector(16, i8) = q_odd_g[48..64].*;
                const raw3: @Vector(16, u8) = g_bytes[48..64].*;
                const low3 = raw3 & mask0f_16;
                const high3 = raw3 >> shift4_16;

                if (comptime @import("builtin").cpu.arch == .aarch64) {
                    asm volatile (
                        \\ sdot %[acc_e].4s, %[low0].16b, %[xe0].16b
                        \\ sdot %[acc_o].4s, %[high0].16b, %[xo0].16b
                        \\ sdot %[acc_e].4s, %[low1].16b, %[xe1].16b
                        \\ sdot %[acc_o].4s, %[high1].16b, %[xo1].16b
                        \\ sdot %[acc_e].4s, %[low2].16b, %[xe2].16b
                        \\ sdot %[acc_o].4s, %[high2].16b, %[xo2].16b
                        \\ sdot %[acc_e].4s, %[low3].16b, %[xe3].16b
                        \\ sdot %[acc_o].4s, %[high3].16b, %[xo3].16b
                        : [acc_e] "+w" (acc_e),
                          [acc_o] "+w" (acc_o),
                        : [xe0] "w" (xe0),
                          [xo0] "w" (xo0),
                          [low0] "w" (low0),
                          [high0] "w" (high0),
                          [xe1] "w" (xe1),
                          [xo1] "w" (xo1),
                          [low1] "w" (low1),
                          [high1] "w" (high1),
                          [xe2] "w" (xe2),
                          [xo2] "w" (xo2),
                          [low2] "w" (low2),
                          [high2] "w" (high2),
                          [xe3] "w" (xe3),
                          [xo3] "w" (xo3),
                          [low3] "w" (low3),
                          [high3] "w" (high3),
                    );
                } else {
                    inline for (.{ .{ low0, high0, xe0, xo0 }, .{ low1, high1, xe1, xo1 }, .{ low2, high2, xe2, xo2 }, .{ low3, high3, xe3, xo3 } }) |quad| {
                        const l_i8: @Vector(16, i8) = @bitCast(quad[0]);
                        const h_i8: @Vector(16, i8) = @bitCast(quad[1]);
                        const l_16: @Vector(16, i16) = l_i8;
                        const h_16: @Vector(16, i16) = h_i8;
                        const xe_16: @Vector(16, i16) = quad[2];
                        const xo_16: @Vector(16, i16) = quad[3];
                        const pe: @Vector(16, i32) = @as(@Vector(16, i32), xe_16) * @as(@Vector(16, i32), l_16);
                        const po: @Vector(16, i32) = @as(@Vector(16, i32), xo_16) * @as(@Vector(16, i32), h_16);
                        acc_e += .{ pe[0] + pe[1] + pe[2] + pe[3], pe[4] + pe[5] + pe[6] + pe[7], pe[8] + pe[9] + pe[10] + pe[11], pe[12] + pe[13] + pe[14] + pe[15] };
                        acc_o += .{ po[0] + po[1] + po[2] + po[3], po[4] + po[5] + po[6] + po[7], po[8] + po[9] + po[10] + po[11], po[12] + po[13] + po[14] + po[15] };
                    }
                }

                const q_sum_8 = 8 * act.q_sums[group_in_row];
                const int_dot: i32 = @as(i32, @intCast(@reduce(.Add, acc_e + acc_o))) - q_sum_8;
                row_accum += @as(f32, @floatFromInt(int_dot)) * (act_scale * scale);
            } else {
                const x_even_g = act.even[c_half .. c_half + 64];
                const x_odd_g = act.odd[c_half .. c_half + 64];
                const mask0f_8: @Vector(8, u8) = @splat(0x0F);
                const shift4_8: @Vector(8, u3) = @splat(4);

                var dot_vec0: @Vector(8, f32) = @splat(0.0);
                var dot_vec1: @Vector(8, f32) = @splat(0.0);
                var dot_vec2: @Vector(8, f32) = @splat(0.0);
                var dot_vec3: @Vector(8, f32) = @splat(0.0);

                inline for (.{ 0, 16, 32, 48 }) |j| {
                    const raw0: @Vector(8, u8) = g_bytes[j..][0..8].*;
                    const low0 = raw0 & mask0f_8;
                    const high0 = raw0 >> shift4_8;
                    const q_vec0: @Vector(8, f32) = @floatFromInt(low0);
                    const q_vec1: @Vector(8, f32) = @floatFromInt(high0);
                    const x_vec0: @Vector(8, f32) = x_even_g[j..][0..8].*;
                    const x_vec1: @Vector(8, f32) = x_odd_g[j..][0..8].*;
                    dot_vec0 += q_vec0 * x_vec0;
                    dot_vec1 += q_vec1 * x_vec1;

                    const raw1: @Vector(8, u8) = g_bytes[j + 8 ..][0..8].*;
                    const low1 = raw1 & mask0f_8;
                    const high1 = raw1 >> shift4_8;
                    const q_vec2: @Vector(8, f32) = @floatFromInt(low1);
                    const q_vec3: @Vector(8, f32) = @floatFromInt(high1);
                    const x_vec2: @Vector(8, f32) = x_even_g[j + 8 ..][0..8].*;
                    const x_vec3: @Vector(8, f32) = x_odd_g[j + 8 ..][0..8].*;
                    dot_vec2 += q_vec2 * x_vec2;
                    dot_vec3 += q_vec3 * x_vec3;
                }
                const dot_raw = @reduce(.Add, (dot_vec0 + dot_vec1) + (dot_vec2 + dot_vec3));
                row_accum += (dot_raw - 8.0 * act.sums[group_in_row]) * scale;
            }

            c += 128;
            group_in_row += 1;
            if (c == INTERMEDIATE_DIM) {
                if (r < y.len) {
                    y[r] += row_accum;
                }
                row_accum = 0.0;
                c = 0;
                r += 1;
                group_in_row = 0;
            }
        }
        if (row_accum != 0.0 and r < y.len) {
            y[r] += row_accum;
        }

        for (0..num_outliers) |i| {
            const w_idx: usize = outlier_offsets_raw[i];
            const glob_k = global_k_start + w_idx;
            const r_out = glob_k / INTERMEDIATE_DIM;
            const c_out = glob_k - r_out * INTERMEDIATE_DIM;
            const w_val: f32 = @floatCast(outlier_w_raw[i]);
            if (r_out < y.len) {
                y[r_out] += w_val * act.raw[c_out];
            }
        }
        return;
    }

    const scale: f32 = @bitCast(std.mem.readInt(u32, payload[32..36], .little));
    const bias: f32 = @bitCast(std.mem.readInt(u32, payload[36..40], .little));

    const coded: [*]const u8 = @ptrCast(&cell.fingerprints);
    const global_k_start: usize = tile_idx * 32768;
    var r: usize = global_k_start / INTERMEDIATE_DIM;
    var c: usize = global_k_start % INTERMEDIATE_DIM;

    var w_idx: usize = 0;
    while (w_idx < 32768) {
        const span_len = @min(32768 - w_idx, INTERMEDIATE_DIM - c);

        var dot_vec: @Vector(8, f32) = @splat(0.0);
        var sum_x_vec: @Vector(8, f32) = @splat(0.0);

        var off: usize = 0;
        while (off + 16 <= span_len) : (off += 16) {
            const byte_off = (w_idx + off) / 2;
            const raw_bytes = coded[byte_off .. byte_off + 8];

            var q_unpacked: [16]f32 = undefined;
            for (0..8) |b| {
                const byte = raw_bytes[b];
                q_unpacked[b * 2] = @floatFromInt(byte & 0x0F);
                q_unpacked[b * 2 + 1] = @floatFromInt(byte >> 4);
            }

            const q_v0: @Vector(8, f32) = q_unpacked[0..8].*;
            const q_v1: @Vector(8, f32) = q_unpacked[8..16].*;
            const x_v0: @Vector(8, f32) = act.raw[c + off ..][0..8].*;
            const x_v1: @Vector(8, f32) = act.raw[c + off + 8 ..][0..8].*;

            dot_vec += q_v0 * x_v0 + q_v1 * x_v1;
            sum_x_vec += x_v0 + x_v1;
        }

        while (off < span_len) : (off += 1) {
            const byte_off = (w_idx + off) / 2;
            const nibble: u4 = if ((w_idx + off) % 2 == 0)
                @truncate(coded[byte_off] & 0x0F)
            else
                @truncate(coded[byte_off] >> 4);

            const q_f: f32 = @floatFromInt(nibble);
            const x_f = act.raw[c + off];
            dot_vec[0] += q_f * x_f;
            sum_x_vec[0] += x_f;
        }

        const dot_total = @reduce(.Add, dot_vec);
        const sum_x_total = @reduce(.Add, sum_x_vec);
        if (r < y.len) {
            y[r] += dot_total * scale + sum_x_total * bias;
        }

        w_idx += span_len;
        c += span_len;
        if (c == INTERMEDIATE_DIM) {
            c = 0;
            r += 1;
        }
    }
}

pub fn gemvTileDownDeint(rec: *const geometry.Record, act: *const Activation11008, y: []f32, tile_idx: usize) void {
    gemvTileDownDeintCellOpt(&rec.cell, act, y, tile_idx, USE_INT8_SDOT);
}


// ── 4-Core Worker Pool (Z3 & Vampire Proven) ───────────────────────────────────

pub const TaskFn = *const fn (worker_id: usize, ctx: *anyopaque) void;

pub const WorkerPool = struct {
    threads: [3]std.Thread = undefined,
    task_fn: ?TaskFn = null,
    task_ctx: ?*anyopaque = null,
    generation: std.atomic.Value(u32) align(64) = std.atomic.Value(u32).init(0),
    worker_done: [3]std.atomic.Value(u32) align(64) = [_]std.atomic.Value(u32){
        std.atomic.Value(u32).init(0),
        std.atomic.Value(u32).init(0),
        std.atomic.Value(u32).init(0),
    },
    shutdown: std.atomic.Value(bool) align(64) = std.atomic.Value(bool).init(false),
    initialized: bool = false,

    pub fn init(self: *WorkerPool) !void {
        self.task_fn = null;
        self.task_ctx = null;
        self.generation.store(0, .seq_cst);
        for (&self.worker_done) |*slot| {
            slot.store(0, .seq_cst);
        }
        self.shutdown.store(false, .seq_cst);
        if (comptime @import("builtin").os.tag == .linux) {
            var cpuset0: [16]usize = @splat(0);
            cpuset0[0] = 1;
            const rc = std.os.linux.syscall3(std.os.linux.SYS.sched_setaffinity, 0, 128, @intFromPtr(&cpuset0));
            std.debug.assert(rc == 0);
        }
        for (1..4) |i| {
            self.threads[i - 1] = try std.Thread.spawn(.{}, workerLoop, .{ self, i });
        }
        self.initialized = true;
    }

    pub fn dispatch(self: *WorkerPool, func: TaskFn, ctx: *anyopaque) void {
        self.task_fn = func;
        self.task_ctx = ctx;
        const next_gen = self.generation.load(.monotonic) +% 1;
        self.generation.store(next_gen, .release);

        // Run worker 0 on calling thread (already pinned to core 0)
        func(0, ctx);

        // Wait for workers 1, 2, 3 to finish
        while (self.worker_done[0].load(.acquire) != next_gen or
            self.worker_done[1].load(.acquire) != next_gen or
            self.worker_done[2].load(.acquire) != next_gen)
        {
            if (comptime @import("builtin").cpu.arch == .aarch64) {
                asm volatile ("yield");
            } else {
                std.atomic.spinLoopHint();
            }
        }
    }

    pub fn deinit(self: *WorkerPool) void {
        if (!self.initialized) return;
        self.shutdown.store(true, .release);
        self.generation.store(self.generation.load(.monotonic) +% 1, .release);
        for (self.threads) |t| {
            t.join();
        }
        self.initialized = false;
    }

    fn workerLoop(self: *WorkerPool, worker_id: usize) void {
        if (comptime @import("builtin").os.tag == .linux) {
            var cpuset: [16]usize = @splat(0);
            cpuset[0] = @as(usize, 1) << @intCast(worker_id);
            const rc = std.os.linux.syscall3(std.os.linux.SYS.sched_setaffinity, 0, 128, @intFromPtr(&cpuset));
            std.debug.assert(rc == 0);
        }
        var last_gen: u32 = 0;
        const done_slot = &self.worker_done[worker_id - 1];
        while (true) {
            while (true) {
                if (self.shutdown.load(.acquire)) return;
                const cur_gen = self.generation.load(.acquire);
                if (cur_gen != last_gen) {
                    last_gen = cur_gen;
                    break;
                }
                if (comptime @import("builtin").cpu.arch == .aarch64) {
                    asm volatile ("yield");
                } else {
                    std.atomic.spinLoopHint();
                }
            }
            if (self.shutdown.load(.acquire)) return;
            if (self.task_fn) |f| {
                f(worker_id, self.task_ctx.?);
            }
            done_slot.store(last_gen, .release);
        }
    }
};

var global_pool: WorkerPool = .{};
var global_pool_init: bool = false;

pub fn getWorkerPool() *WorkerPool {
    if (!global_pool_init) {
        global_pool.init() catch {
            return &global_pool;
        };
        global_pool_init = true;
    }
    return &global_pool;
}

pub fn forwardLayer(
    archive: *const weight_archive.WeightArchive,
    layer_idx: usize,
    pos: u64,
    hidden: *[HIDDEN_DIM]f32,
    kv_cache: *attn.KvCache,
) !void {
    const map = &tensor_map.LAYERS[layer_idx];
    const pool = getWorkerPool();
    const is_dense = archive.isDense();
    const records = if (!is_dense) archive.getRecordsPtr() else undefined;
    const cells = if (is_dense) archive.getCellsPtr() else undefined;

    // 1. Input RMSNorm
    const t_norm_0 = nowNs();
    var norm_gamma: [HIDDEN_DIM]f32 = undefined;
    const input_norm_cell = archive.getCellDirect(map.input_norm);
    unpack1DCell(input_norm_cell, &norm_gamma);

    var norm_x: [HIDDEN_DIM]f32 = undefined;
    rmsnorm.apply(hidden, &norm_gamma, &norm_x, qwen_geom.RMS_NORM_EPS);

    Activation2048.initInto(&global_act2048, &norm_x);
    prof_norm_ns += nowNs() - t_norm_0;

    // 2. Q, K, V Projections in parallel across 4 workers (40 tiles per worker)
    const t_qkv_0 = nowNs();
    var q: [HIDDEN_DIM]f32 = undefined;
    const q_bias_cell = archive.getCellDirect(map.q_bias);
    unpack1DCell(q_bias_cell, &q);

    var k: [KV_DIM]f32 = undefined;
    const k_bias_cell = archive.getCellDirect(map.k_bias);
    unpack1DCell(k_bias_cell, &k);

    var v: [KV_DIM]f32 = undefined;
    const v_bias_cell = archive.getCellDirect(map.v_bias);
    unpack1DCell(v_bias_cell, &v);

    const QkvProjCtx = struct {
        records: [*]const geometry.Record,
        cells: [*]const geometry.Cell,
        is_dense: bool,
        act: *const Activation2048,
        q: *[HIDDEN_DIM]f32,
        k: *[KV_DIM]f32,
        v: *[KV_DIM]f32,
        q_base: usize,
        k_base: usize,
        v_base: usize,
    };
    const qkv_ctx = QkvProjCtx{
        .records = records,
        .cells = cells,
        .is_dense = is_dense,
        .act = &global_act2048,
        .q = &q,
        .k = &k,
        .v = &v,
        .q_base = map.q_proj,
        .k_base = map.k_proj,
        .v_base = map.v_proj,
    };
    pool.dispatch(struct {
        fn run(worker_id: usize, raw_ctx: *anyopaque) void {
            const c: *const QkvProjCtx = @ptrCast(@alignCast(raw_ctx));
            const q_start = worker_id * 32;
            const q_end = (worker_id + 1) * 32;
            if (c.is_dense) {
                for (q_start..q_end) |t| {
                    if (t + 1 < q_end) {
                        prefetchCell(&c.cells[c.q_base + t + 1]);
                    }
                    const cell = &c.cells[c.q_base + t];
                    gemvTileQuadRowCellDirect(cell, c.act, c.q, t * 16, USE_INT8_SDOT, true);
                }
                if (worker_id == 0) {
                    for (0..8) |t| {
                        if (t + 1 < 8) {
                            prefetchCell(&c.cells[c.k_base + t + 1]);
                        }
                        const cell = &c.cells[c.k_base + t];
                        gemvTileQuadRowCellDirect(cell, c.act, c.k, t * 16, USE_INT8_SDOT, true);
                    }
                } else if (worker_id == 1) {
                    for (8..16) |t| {
                        if (t + 1 < 16) {
                            prefetchCell(&c.cells[c.k_base + t + 1]);
                        }
                        const cell = &c.cells[c.k_base + t];
                        gemvTileQuadRowCellDirect(cell, c.act, c.k, t * 16, USE_INT8_SDOT, true);
                    }
                } else if (worker_id == 2) {
                    for (0..8) |t| {
                        if (t + 1 < 8) {
                            prefetchCell(&c.cells[c.v_base + t + 1]);
                        }
                        const cell = &c.cells[c.v_base + t];
                        gemvTileQuadRowCellDirect(cell, c.act, c.v, t * 16, USE_INT8_SDOT, true);
                    }
                } else {
                    for (8..16) |t| {
                        if (t + 1 < 16) {
                            prefetchCell(&c.cells[c.v_base + t + 1]);
                        }
                        const cell = &c.cells[c.v_base + t];
                        gemvTileQuadRowCellDirect(cell, c.act, c.v, t * 16, USE_INT8_SDOT, true);
                    }
                }
            } else {
                for (q_start..q_end) |t| {
                    if (t + 1 < q_end) {
                        prefetchRecord(&c.records[c.q_base + t + 1]);
                    }
                    const rec = &c.records[c.q_base + t];
                    gemvTileQuadRow(rec, c.act, c.q, t * 16);
                }
                if (worker_id == 0) {
                    for (0..8) |t| {
                        if (t + 1 < 8) {
                            prefetchRecord(&c.records[c.k_base + t + 1]);
                        }
                        const rec = &c.records[c.k_base + t];
                        gemvTileQuadRow(rec, c.act, c.k, t * 16);
                    }
                } else if (worker_id == 1) {
                    for (8..16) |t| {
                        if (t + 1 < 16) {
                            prefetchRecord(&c.records[c.k_base + t + 1]);
                        }
                        const rec = &c.records[c.k_base + t];
                        gemvTileQuadRow(rec, c.act, c.k, t * 16);
                    }
                } else if (worker_id == 2) {
                    for (0..8) |t| {
                        if (t + 1 < 8) {
                            prefetchRecord(&c.records[c.v_base + t + 1]);
                        }
                        const rec = &c.records[c.v_base + t];
                        gemvTileQuadRow(rec, c.act, c.v, t * 16);
                    }
                } else {
                    for (8..16) |t| {
                        if (t + 1 < 16) {
                            prefetchRecord(&c.records[c.v_base + t + 1]);
                        }
                        const rec = &c.records[c.v_base + t];
                        gemvTileQuadRow(rec, c.act, c.v, t * 16);
                    }
                }
            }
        }
    }.run, @constCast(&qkv_ctx));
    prof_qkv_ns += nowNs() - t_qkv_0;

    // 3. RoPE on Q and K
    const t_attn_0 = nowNs();
    rope.applyRopeMultiHead(&q, qwen_geom.Q_HEADS, pos);
    rope.applyRopeMultiHead(&k, qwen_geom.KV_HEADS, pos);

    // 4. Update KV cache
    kv_cache.append(&k, &v);

    // 5. Grouped-Query Attention
    var attn_out: [HIDDEN_DIM]f32 = undefined;
    attn.forwardGqa(&q, kv_cache, &attn_out);
    prof_attn_ns += nowNs() - t_attn_0;

    // 6. Attention Output Projection + In-Place Residual (4 workers * 32 tiles)
    const t_oproj_0 = nowNs();
    Activation2048.initInto(&global_act2048, &attn_out);

    const OProjCtx = struct {
        records: [*]const geometry.Record,
        cells: [*]const geometry.Cell,
        is_dense: bool,
        act: *const Activation2048,
        hidden: *[HIDDEN_DIM]f32,
        base: usize,
    };
    const o_ctx = OProjCtx{
        .records = records,
        .cells = cells,
        .is_dense = is_dense,
        .act = &global_act2048,
        .hidden = hidden,
        .base = map.o_proj,
    };
    pool.dispatch(struct {
        fn run(worker_id: usize, raw_ctx: *anyopaque) void {
            const c: *const OProjCtx = @ptrCast(@alignCast(raw_ctx));
            const start_t = worker_id * 32;
            const end_t = (worker_id + 1) * 32;
            if (c.is_dense) {
                for (start_t..end_t) |t| {
                    if (t + 1 < end_t) {
                        prefetchCell(&c.cells[c.base + t + 1]);
                    }
                    const cell = &c.cells[c.base + t];
                    gemvTileQuadRowCellDirect(cell, c.act, c.hidden, t * 16, USE_INT8_SDOT, true);
                }
            } else {
                for (start_t..end_t) |t| {
                    if (t + 1 < end_t) {
                        prefetchRecord(&c.records[c.base + t + 1]);
                    }
                    const rec = &c.records[c.base + t];
                    gemvTileQuadRow(rec, c.act, c.hidden, t * 16);
                }
            }
        }
    }.run, @constCast(&o_ctx));
    prof_oproj_ns += nowNs() - t_oproj_0;

    // 7. Post-Attention RMSNorm
    const t_norm_1 = nowNs();
    const post_norm_cell = archive.getCellDirect(map.post_norm);
    unpack1DCell(post_norm_cell, &norm_gamma);
    rmsnorm.apply(hidden, &norm_gamma, &norm_x, qwen_geom.RMS_NORM_EPS);

    Activation2048.initInto(&global_act2048, &norm_x);
    prof_norm_ns += nowNs() - t_norm_1;


    // 8. SwiGLU MLP: Gate and Up Projections + Parallel SwiGLU (4 workers)
    const t_gu_0 = nowNs();
    var gate_buf: [INTERMEDIATE_DIM]f32 = @splat(0.0);
    var up_buf: [INTERMEDIATE_DIM]f32 = @splat(0.0);
    const GateUpCtx = struct {
        records: [*]const geometry.Record,
        cells: [*]const geometry.Cell,
        is_dense: bool,
        act: *const Activation2048,
        act11008: *Activation11008,
        gate_buf: *[INTERMEDIATE_DIM]f32,
        up_buf: *[INTERMEDIATE_DIM]f32,
        gate_base: usize,
        up_base: usize,
    };
    const gate_up_ctx = GateUpCtx{
        .records = records,
        .cells = cells,
        .is_dense = is_dense,
        .act = &global_act2048,
        .act11008 = &global_act11008,
        .gate_buf = &gate_buf,
        .up_buf = &up_buf,
        .gate_base = map.gate_proj,
        .up_base = map.up_proj,
    };
    pool.dispatch(struct {
        fn run(worker_id: usize, raw_ctx: *anyopaque) void {
            const c: *const GateUpCtx = @ptrCast(@alignCast(raw_ctx));
            const start_t = worker_tile_ranges[worker_id][0];
            const end_t = worker_tile_ranges[worker_id][1];
            if (c.is_dense) {
                for (start_t..end_t) |t| {
                    if (t + 1 < end_t) {
                        prefetchCell(&c.cells[c.gate_base + t + 1]);
                        prefetchCell(&c.cells[c.up_base + t + 1]);
                    }
                    const grec = &c.cells[c.gate_base + t];
                    const urec = &c.cells[c.up_base + t];
                    const b_idx = t * 16;
                    gemvTileGateUpFusedDirect(grec, urec, c.act, c.gate_buf, b_idx, USE_INT8_SDOT);

                    if ((t + 1) % hw.TILES_PER_GROUP == 0) {
                        const g = t / hw.TILES_PER_GROUP;
                        Activation11008.quantizeRange(c.act11008, c.gate_buf, g, g + 1);
                    }
                }
            } else {
                for (start_t..end_t) |t| {
                    if (t + 1 < end_t) {
                        prefetchRecord(&c.records[c.gate_base + t + 1]);
                        prefetchRecord(&c.records[c.up_base + t + 1]);
                    }
                    const grec = &c.records[c.gate_base + t];
                    gemvTileQuadRowDirect(grec, c.act, c.gate_buf, t * 16, false);
                    const urec = &c.records[c.up_base + t];
                    gemvTileQuadRowDirect(urec, c.act, c.up_buf, t * 16, false);
                    const b_idx = t * 16;
                    mlp.swigluForward(c.gate_buf[b_idx .. b_idx + 16], c.up_buf[b_idx .. b_idx + 16], c.gate_buf[b_idx .. b_idx + 16]);

                    if ((t + 1) % hw.TILES_PER_GROUP == 0) {
                        const g = t / hw.TILES_PER_GROUP;
                        Activation11008.quantizeRange(c.act11008, c.gate_buf, g, g + 1);
                    }
                }
            }
        }
    }.run, @constCast(&gate_up_ctx));
    prof_gateup_ns += nowNs() - t_gu_0;

    // 9. Down Projection in parallel (4 workers into private buffers)
    const t_down_0 = nowNs();
    global_act11008.raw = &gate_buf;

    const DownCtx = struct {
        records: [*]const geometry.Record,
        cells: [*]const geometry.Cell,
        is_dense: bool,
        act: *const Activation11008,
        down_base: usize,
    };
    const down_ctx = DownCtx{
        .records = records,
        .cells = cells,
        .is_dense = is_dense,
        .act = &global_act11008,
        .down_base = map.down_proj,
    };
    pool.dispatch(struct {
        fn run(worker_id: usize, raw_ctx: *anyopaque) void {
            const c: *const DownCtx = @ptrCast(@alignCast(raw_ctx));
            switch (worker_id) {
                0 => @memset(worker_down_buffers[0][0..524], 0.0),
                1 => @memset(worker_down_buffers[1][523..1048], 0.0),
                2 => @memset(worker_down_buffers[2][1047..1548], 0.0),
                3 => @memset(worker_down_buffers[3][1547..2048], 0.0),
                else => @memset(&worker_down_buffers[worker_id], 0.0),
            }
            const start_t = worker_tile_ranges[worker_id][0];
            const end_t = worker_tile_ranges[worker_id][1];
            if (c.is_dense) {
                if (start_t + 1 < end_t) {
                    prefetchCell(&c.cells[c.down_base + start_t + 1]);
                }
                for (start_t..end_t) |t| {
                    if (t + 2 < end_t) {
                        prefetchCell(&c.cells[c.down_base + t + 2]);
                    } else if (t + 1 < end_t) {
                        prefetchCell(&c.cells[c.down_base + t + 1]);
                    }
                    const cell = &c.cells[c.down_base + t];
                    gemvTileDownDeintCell8Acc(cell, c.act, &worker_down_buffers[worker_id], t, USE_INT8_SDOT);
                }
            } else {
                if (start_t + 1 < end_t) {
                    prefetchRecord(&c.records[c.down_base + start_t + 1]);
                }
                for (start_t..end_t) |t| {
                    if (t + 2 < end_t) {
                        prefetchRecord(&c.records[c.down_base + t + 2]);
                    } else if (t + 1 < end_t) {
                        prefetchRecord(&c.records[c.down_base + t + 1]);
                    }
                    const rec = &c.records[c.down_base + t];
                    gemvTileDownDeint(rec, c.act, &worker_down_buffers[worker_id], t);
                }
            }
        }
    }.run, @constCast(&down_ctx));

    // Sum private worker buffers directly into hidden residual via disjoint SIMD slices
    var i: usize = 0;
    while (i + 8 <= 520) : (i += 8) {
        const v0: @Vector(8, f32) = worker_down_buffers[0][i..][0..8].*;
        const h: @Vector(8, f32) = hidden[i..][0..8].*;
        hidden[i..][0..8].* = h + v0;
    }
    while (i < 523) : (i += 1) {
        hidden[i] += worker_down_buffers[0][i];
    }
    hidden[523] += worker_down_buffers[0][523] + worker_down_buffers[1][523];

    i = 524;
    while (i + 8 <= 1044) : (i += 8) {
        const v1: @Vector(8, f32) = worker_down_buffers[1][i..][0..8].*;
        const h: @Vector(8, f32) = hidden[i..][0..8].*;
        hidden[i..][0..8].* = h + v1;
    }
    while (i < 1047) : (i += 1) {
        hidden[i] += worker_down_buffers[1][i];
    }
    hidden[1047] += worker_down_buffers[1][1047] + worker_down_buffers[2][1047];

    i = 1048;
    while (i + 8 <= 1544) : (i += 8) {
        const v2: @Vector(8, f32) = worker_down_buffers[2][i..][0..8].*;
        const h: @Vector(8, f32) = hidden[i..][0..8].*;
        hidden[i..][0..8].* = h + v2;
    }
    while (i < 1547) : (i += 1) {
        hidden[i] += worker_down_buffers[2][i];
    }
    hidden[1547] += worker_down_buffers[2][1547] + worker_down_buffers[3][1547];

    i = 1548;
    while (i + 8 <= HIDDEN_DIM) : (i += 8) {
        const v3: @Vector(8, f32) = worker_down_buffers[3][i..][0..8].*;
        const h: @Vector(8, f32) = hidden[i..][0..8].*;
        hidden[i..][0..8].* = h + v3;
    }
    while (i < HIDDEN_DIM) : (i += 1) {
        hidden[i] += worker_down_buffers[3][i];
    }
    prof_down_ns += nowNs() - t_down_0;
}


pub const HeadMode = enum {
    generative_f64,
    generative_f32,
    embedding_pool,
    judge_verdict,
};

pub fn stableSoftmaxF64(logits: []const f64, probs: []f64) void {
    var max_val: f64 = -std.math.inf(f64);
    for (logits) |v| {
        if (v > max_val) max_val = v;
    }

    var sum_exp: f64 = 0.0;
    for (logits, 0..) |v, i| {
        const exp_v = @exp(v - max_val);
        probs[i] = exp_v;
        sum_exp += exp_v;
    }

    const inv_sum = 1.0 / sum_exp;
    for (probs) |*p| {
        p.* *= inv_sum;
    }
}

pub fn forwardLayerHybrid(
    archive: *const weight_archive.WeightArchive,
    layer_idx: usize,
    pos: u64,
    hidden: *[HIDDEN_DIM]f32,
    kv_cache: *attn.KvCache,
) !void {
    const pool = getWorkerPool();
    const is_dense = archive.isDense();
    const records = if (!is_dense) archive.getRecordsPtr() else undefined;
    const cells = if (is_dense) archive.getCellsPtr() else undefined;

    // 1. Input RMSNorm
    const t_norm_0 = nowNs();
    var norm_gamma: [HIDDEN_DIM]f32 = undefined;
    const in_norm_cell = archive.getCellDirect(qwen_geom.HybridLayerOffsets.inNorm(layer_idx));
    unpack1DCell(in_norm_cell, &norm_gamma);

    var norm_x: [HIDDEN_DIM]f32 = undefined;
    rmsnorm.apply(hidden, &norm_gamma, &norm_x, qwen_geom.RMS_NORM_EPS);
    prof_norm_ns += nowNs() - t_norm_0;

    // 2. Q, K, V Projections (BF16 GEMV in parallel across 4 workers)
    const t_qkv_0 = nowNs();
    var q: [HIDDEN_DIM]f32 = undefined;
    const q_bias_cell = archive.getCellDirect(qwen_geom.HybridLayerOffsets.qBias(layer_idx));
    unpack1DCell(q_bias_cell, &q);

    var k: [KV_DIM]f32 = undefined;
    const k_bias_cell = archive.getCellDirect(qwen_geom.HybridLayerOffsets.kBias(layer_idx));
    unpack1DCell(k_bias_cell, &k);

    var v: [KV_DIM]f32 = undefined;
    const v_bias_cell = archive.getCellDirect(qwen_geom.HybridLayerOffsets.vBias(layer_idx));
    unpack1DCell(v_bias_cell, &v);

    Activation2048.initInto(&global_act2048, &norm_x);

    const HybridQkvCtx = struct {
        records: [*]const geometry.Record,
        cells: [*]const geometry.Cell,
        is_dense: bool,
        act: *const Activation2048,
        norm_x: *const [HIDDEN_DIM]f32,
        q: *[HIDDEN_DIM]f32,
        k: *[KV_DIM]f32,
        v: *[KV_DIM]f32,
        q_base: usize,
        k_base: usize,
        v_base: usize,
    };
    const qkv_ctx = HybridQkvCtx{
        .records = records,
        .cells = cells,
        .is_dense = is_dense,
        .act = &global_act2048,
        .norm_x = &norm_x,
        .q = &q,
        .k = &k,
        .v = &v,
        .q_base = qwen_geom.HybridLayerOffsets.qProj(layer_idx),
        .k_base = qwen_geom.HybridLayerOffsets.kProj(layer_idx),
        .v_base = qwen_geom.HybridLayerOffsets.vProj(layer_idx),
    };
    pool.dispatch(struct {
        fn run(worker_id: usize, raw_ctx: *anyopaque) void {
            const c: *const HybridQkvCtx = @ptrCast(@alignCast(raw_ctx));
            const q_start = worker_id * 128;
            const q_end = (worker_id + 1) * 128;
            if (c.is_dense) {
                for (q_start..q_end) |t| {
                    if (t + 1 < q_end) {
                        prefetchCell(&c.cells[c.q_base + t + 1]);
                    }
                    const cell = &c.cells[c.q_base + t];
                    gemvTileBF16CellDirect(cell, c.norm_x, c.q, t * 4);
                }
                const k_start = worker_id * 16;
                const k_end = (worker_id + 1) * 16;
                for (k_start..k_end) |t| {
                    if (t + 1 < k_end) {
                        prefetchCell(&c.cells[c.k_base + t + 1]);
                    }
                    const cell = &c.cells[c.k_base + t];
                    gemvTileBF16CellDirect(cell, c.norm_x, c.k, t * 4);
                }
                const v_start = worker_id * 16;
                const v_end = (worker_id + 1) * 16;
                for (v_start..v_end) |t| {
                    if (t + 1 < v_end) {
                        prefetchCell(&c.cells[c.v_base + t + 1]);
                    }
                    const cell = &c.cells[c.v_base + t];
                    gemvTileBF16CellDirect(cell, c.norm_x, c.v, t * 4);
                }
            } else {
                for (q_start..q_end) |t| {
                    if (t + 1 < q_end) {
                        prefetchRecord(&c.records[c.q_base + t + 1]);
                    }
                    const rec = &c.records[c.q_base + t];
                    gemvTileBF16(rec, c.norm_x, c.q, t * 4);
                }
                const k_start = worker_id * 16;
                const k_end = (worker_id + 1) * 16;
                for (k_start..k_end) |t| {
                    if (t + 1 < k_end) {
                        prefetchRecord(&c.records[c.k_base + t + 1]);
                    }
                    const rec = &c.records[c.k_base + t];
                    gemvTileBF16(rec, c.norm_x, c.k, t * 4);
                }
                const v_start = worker_id * 16;
                const v_end = (worker_id + 1) * 16;
                for (v_start..v_end) |t| {
                    if (t + 1 < v_end) {
                        prefetchRecord(&c.records[c.v_base + t + 1]);
                    }
                    const rec = &c.records[c.v_base + t];
                    gemvTileBF16(rec, c.norm_x, c.v, t * 4);
                }
            }
        }
    }.run, @constCast(&qkv_ctx));
    prof_qkv_ns += nowNs() - t_qkv_0;

    // 3. RoPE on Q and K
    const t_attn_0 = nowNs();
    rope.applyRopeMultiHead(&q, qwen_geom.Q_HEADS, pos);
    rope.applyRopeMultiHead(&k, qwen_geom.KV_HEADS, pos);

    // 4. Update KV cache
    kv_cache.append(&k, &v);

    // 5. Grouped-Query Attention
    var attn_out: [HIDDEN_DIM]f32 = undefined;
    attn.forwardGqa(&q, kv_cache, &attn_out);
    prof_attn_ns += nowNs() - t_attn_0;

    // 6. Attention Output Projection + In-Place Residual
    const t_oproj_0 = nowNs();
    const is_o_w4 = qwen_geom.HybridLayerOffsets.isOW4(layer_idx);
    var act_attn: Activation2048 = undefined;
    Activation2048.initInto(&act_attn, &attn_out);

    const HybridOProjCtx = struct {
        records: [*]const geometry.Record,
        cells: [*]const geometry.Cell,
        is_dense: bool,
        is_o_w4: bool,
        act: *const Activation2048,
        attn_out: *const [HIDDEN_DIM]f32,
        hidden: *[HIDDEN_DIM]f32,
        base: usize,
    };
    const o_ctx = HybridOProjCtx{
        .records = records,
        .cells = cells,
        .is_dense = is_dense,
        .is_o_w4 = is_o_w4,
        .act = &act_attn,
        .attn_out = &attn_out,
        .hidden = hidden,
        .base = qwen_geom.HybridLayerOffsets.oProj(layer_idx),
    };
    pool.dispatch(struct {
        fn run(worker_id: usize, raw_ctx: *anyopaque) void {
            const c: *const HybridOProjCtx = @ptrCast(@alignCast(raw_ctx));
            if (c.is_o_w4) {
                const start_t = worker_id * 32;
                const end_t = (worker_id + 1) * 32;
                if (c.is_dense) {
                    for (start_t..end_t) |t| {
                        if (t + 1 < end_t) {
                            prefetchCell(&c.cells[c.base + t + 1]);
                        }
                        const cell = &c.cells[c.base + t];
                        gemvTileQuadRowCellDirect(cell, c.act, c.hidden, t * 16, USE_INT8_SDOT, true);
                    }
                } else {
                    for (start_t..end_t) |t| {
                        if (t + 1 < end_t) {
                            prefetchRecord(&c.records[c.base + t + 1]);
                        }
                        const rec = &c.records[c.base + t];
                        gemvTileQuadRowDirect(rec, c.act, c.hidden, t * 16, true);
                    }
                }
            } else {
                const start_t = worker_id * 64;
                const end_t = (worker_id + 1) * 64;
                if (c.is_dense) {
                    for (start_t..end_t) |t| {
                        if (t + 1 < end_t) {
                            prefetchCell(&c.cells[c.base + t + 1]);
                        }
                        const cell = &c.cells[c.base + t];
                        gemvTileW8CellDirect(cell, c.act, c.hidden, t * 8, true);
                    }
                } else {
                    for (start_t..end_t) |t| {
                        if (t + 1 < end_t) {
                            prefetchRecord(&c.records[c.base + t + 1]);
                        }
                        const cell = &c.records[c.base + t].cell;
                        gemvTileW8CellDirect(cell, c.act, c.hidden, t * 8, true);
                    }
                }
            }
        }
    }.run, @constCast(&o_ctx));
    prof_oproj_ns += nowNs() - t_oproj_0;

    // 7. Post-Attention RMSNorm
    const t_norm_1 = nowNs();
    const post_norm_cell = archive.getCellDirect(qwen_geom.HybridLayerOffsets.postNorm(layer_idx));
    unpack1DCell(post_norm_cell, &norm_gamma);
    rmsnorm.apply(hidden, &norm_gamma, &norm_x, qwen_geom.RMS_NORM_EPS);

    Activation2048.initInto(&global_act2048, &norm_x);
    prof_norm_ns += nowNs() - t_norm_1;

    // 8. SwiGLU MLP: Gate and Up Projections + Parallel SwiGLU
    const t_gu_0 = nowNs();
    var gate_buf: [INTERMEDIATE_DIM]f32 = @splat(0.0);
    var up_buf: [INTERMEDIATE_DIM]f32 = @splat(0.0);
    const is_mlp_w8 = qwen_geom.HybridLayerOffsets.isMlpW8(layer_idx);
    const is_down_w8 = qwen_geom.HybridLayerOffsets.isDownW8(layer_idx);

    const GateUpCtx = struct {
        records: [*]const geometry.Record,
        cells: [*]const geometry.Cell,
        is_dense: bool,
        is_mlp_w8: bool,
        act: *const Activation2048,
        act11008: *Activation11008,
        gate_buf: *[INTERMEDIATE_DIM]f32,
        up_buf: *[INTERMEDIATE_DIM]f32,
        gate_base: usize,
        up_base: usize,
    };
    const gate_up_ctx = GateUpCtx{
        .records = records,
        .cells = cells,
        .is_dense = is_dense,
        .is_mlp_w8 = is_mlp_w8,
        .act = &global_act2048,
        .act11008 = &global_act11008,
        .gate_buf = &gate_buf,
        .up_buf = &up_buf,
        .gate_base = qwen_geom.HybridLayerOffsets.gateProj(layer_idx),
        .up_base = qwen_geom.HybridLayerOffsets.upProj(layer_idx),
    };
    pool.dispatch(struct {
        fn run(worker_id: usize, raw_ctx: *anyopaque) void {
            const c: *const GateUpCtx = @ptrCast(@alignCast(raw_ctx));
            if (c.is_mlp_w8) {
                const start_t = worker_id * 344;
                const end_t = (worker_id + 1) * 344;
                if (c.is_dense) {
                    for (start_t..end_t) |t| {
                        if (t + 1 < end_t) {
                            prefetchCell(&c.cells[c.gate_base + t + 1]);
                            prefetchCell(&c.cells[c.up_base + t + 1]);
                        }
                        const grec = &c.cells[c.gate_base + t];
                        const urec = &c.cells[c.up_base + t];
                        const b_idx = t * 8;
                        gemvTileGateUpFusedW8Direct(grec, urec, c.act, c.gate_buf, b_idx);

                        if ((t + 1) % 16 == 0) {
                            const g = (t + 1) / 16 - 1;
                            Activation11008.quantizeRange(c.act11008, c.gate_buf, g, g + 1);
                        }
                    }
                } else {
                    for (start_t..end_t) |t| {
                        if (t + 1 < end_t) {
                            prefetchRecord(&c.records[c.gate_base + t + 1]);
                            prefetchRecord(&c.records[c.up_base + t + 1]);
                        }
                        const grec = &c.records[c.gate_base + t].cell;
                        const urec = &c.records[c.up_base + t].cell;
                        const b_idx = t * 8;
                        gemvTileGateUpFusedW8Direct(grec, urec, c.act, c.gate_buf, b_idx);

                        if ((t + 1) % 16 == 0) {
                            const g = (t + 1) / 16 - 1;
                            Activation11008.quantizeRange(c.act11008, c.gate_buf, g, g + 1);
                        }
                    }
                }
            } else {
                const start_t = worker_tile_ranges[worker_id][0];
                const end_t = worker_tile_ranges[worker_id][1];
                if (c.is_dense) {
                    for (start_t..end_t) |t| {
                        if (t + 1 < end_t) {
                            prefetchCell(&c.cells[c.gate_base + t + 1]);
                            prefetchCell(&c.cells[c.up_base + t + 1]);
                        }
                        const grec = &c.cells[c.gate_base + t];
                        const urec = &c.cells[c.up_base + t];
                        const b_idx = t * 16;
                        gemvTileGateUpFusedDirect(grec, urec, c.act, c.gate_buf, b_idx, USE_INT8_SDOT);

                        if ((t + 1) % hw.TILES_PER_GROUP == 0) {
                            const g = t / hw.TILES_PER_GROUP;
                            Activation11008.quantizeRange(c.act11008, c.gate_buf, g, g + 1);
                        }
                    }
                } else {
                    for (start_t..end_t) |t| {
                        if (t + 1 < end_t) {
                            prefetchRecord(&c.records[c.gate_base + t + 1]);
                            prefetchRecord(&c.records[c.up_base + t + 1]);
                        }
                        const grec = &c.records[c.gate_base + t];
                        gemvTileQuadRowDirect(grec, c.act, c.gate_buf, t * 16, false);
                        const urec = &c.records[c.up_base + t];
                        gemvTileQuadRowDirect(urec, c.act, c.up_buf, t * 16, false);
                        const b_idx = t * 16;
                        mlp.swigluForward(c.gate_buf[b_idx .. b_idx + 16], c.up_buf[b_idx .. b_idx + 16], c.gate_buf[b_idx .. b_idx + 16]);

                        if ((t + 1) % hw.TILES_PER_GROUP == 0) {
                            const g = t / hw.TILES_PER_GROUP;
                            Activation11008.quantizeRange(c.act11008, c.gate_buf, g, g + 1);
                        }
                    }
                }
            }
        }
    }.run, @constCast(&gate_up_ctx));
    prof_gateup_ns += nowNs() - t_gu_0;

    // 9. Down Projection in parallel (4 workers into private buffers)
    const t_down_0 = nowNs();
    global_act11008.raw = &gate_buf;

    const DownCtx = struct {
        records: [*]const geometry.Record,
        cells: [*]const geometry.Cell,
        is_dense: bool,
        is_down_w8: bool,
        act: *const Activation11008,
        down_base: usize,
    };
    const down_ctx = DownCtx{
        .records = records,
        .cells = cells,
        .is_dense = is_dense,
        .is_down_w8 = is_down_w8,
        .act = &global_act11008,
        .down_base = qwen_geom.HybridLayerOffsets.downProj(layer_idx),
    };
    pool.dispatch(struct {
        fn run(worker_id: usize, raw_ctx: *anyopaque) void {
            const c: *const DownCtx = @ptrCast(@alignCast(raw_ctx));
            @memset(&worker_down_buffers[worker_id], 0.0);
            if (c.is_down_w8) {
                const start_t = worker_id * 344;
                const end_t = (worker_id + 1) * 344;
                if (c.is_dense) {
                    if (start_t + 1 < end_t) {
                        prefetchCell(&c.cells[c.down_base + start_t + 1]);
                    }
                    for (start_t..end_t) |t| {
                        if (t + 2 < end_t) {
                            prefetchCell(&c.cells[c.down_base + t + 2]);
                        } else if (t + 1 < end_t) {
                            prefetchCell(&c.cells[c.down_base + t + 1]);
                        }
                        const cell = &c.cells[c.down_base + t];
                        gemvTileDownW8(cell, c.act, &worker_down_buffers[worker_id], t);
                    }
                } else {
                    if (start_t + 1 < end_t) {
                        prefetchRecord(&c.records[c.down_base + start_t + 1]);
                    }
                    for (start_t..end_t) |t| {
                        if (t + 2 < end_t) {
                            prefetchRecord(&c.records[c.down_base + t + 2]);
                        } else if (t + 1 < end_t) {
                            prefetchRecord(&c.records[c.down_base + t + 1]);
                        }
                        const cell = &c.records[c.down_base + t].cell;
                        gemvTileDownW8(cell, c.act, &worker_down_buffers[worker_id], t);
                    }
                }
            } else {
                const start_t = worker_tile_ranges[worker_id][0];
                const end_t = worker_tile_ranges[worker_id][1];
                if (c.is_dense) {
                    if (start_t + 1 < end_t) {
                        prefetchCell(&c.cells[c.down_base + start_t + 1]);
                    }
                    for (start_t..end_t) |t| {
                        if (t + 2 < end_t) {
                            prefetchCell(&c.cells[c.down_base + t + 2]);
                        } else if (t + 1 < end_t) {
                            prefetchCell(&c.cells[c.down_base + t + 1]);
                        }
                        const cell = &c.cells[c.down_base + t];
                        gemvTileDownDeintCell8Acc(cell, c.act, &worker_down_buffers[worker_id], t, USE_INT8_SDOT);
                    }
                } else {
                    if (start_t + 1 < end_t) {
                        prefetchRecord(&c.records[c.down_base + start_t + 1]);
                    }
                    for (start_t..end_t) |t| {
                        if (t + 2 < end_t) {
                            prefetchRecord(&c.records[c.down_base + t + 2]);
                        } else if (t + 1 < end_t) {
                            prefetchRecord(&c.records[c.down_base + t + 1]);
                        }
                        const cell = &c.records[c.down_base + t].cell;
                        gemvTileDownDeintCell8Acc(cell, c.act, &worker_down_buffers[worker_id], t, USE_INT8_SDOT);
                    }
                }
            }
        }
    }.run, @constCast(&down_ctx));

    // Sum private worker buffers directly into hidden residual via SIMD
    var i: usize = 0;
    while (i + 8 <= HIDDEN_DIM) : (i += 8) {
        const v0: @Vector(8, f32) = worker_down_buffers[0][i..][0..8].*;
        const v1: @Vector(8, f32) = worker_down_buffers[1][i..][0..8].*;
        const v2: @Vector(8, f32) = worker_down_buffers[2][i..][0..8].*;
        const v3: @Vector(8, f32) = worker_down_buffers[3][i..][0..8].*;
        const h: @Vector(8, f32) = hidden[i..][0..8].*;
        hidden[i..][0..8].* = h + (v0 + v1) + (v2 + v3);
    }
    prof_down_ns += nowNs() - t_down_0;
}

pub const ArchiveMode = enum {
    raw_fp16,
    raw,
    dense,
    standard,
};

pub inline fn getArchiveMode(archive: *const weight_archive.WeightArchive) ArchiveMode {
    if (archive.isRawFp16()) return .raw_fp16;
    if (archive.isRawContiguous()) return .raw;
    if (archive.isDense()) return .dense;
    return .standard;
}

pub fn forwardLayerBF16(
    archive: *const weight_archive.WeightArchive,
    layer_idx: usize,
    pos: u64,
    hidden: *[HIDDEN_DIM]f32,
    kv_cache: *attn.KvCache,
) !void {
    const pool = getWorkerPool();
    const mode = getArchiveMode(archive);
    const raw_tiles: [*]const [8192]u16 = if (mode == .raw or mode == .raw_fp16) @ptrCast(@alignCast(archive.bytes.ptr + weight_archive.HEADER_BYTES)) else undefined;
    const records = if (mode == .standard) archive.getRecordsPtr() else undefined;
    const cells = if (mode == .dense) archive.getCellsPtr() else undefined;
    const layer_base = qwen_geom.Bf16LayerOffsets.layerBase(layer_idx);

    // 1. Input RMSNorm
    const t_norm_0 = nowNs();
    var norm_gamma: [HIDDEN_DIM]f32 = undefined;
    if (mode == .raw or mode == .raw_fp16) {
        @memcpy(&norm_gamma, archive.getTileF32Direct(layer_base + 0)[0..HIDDEN_DIM]);
    } else if (mode == .dense) {
        unpack1DCell(&cells[layer_base + 0], &norm_gamma);
    } else {
        unpack1D(&records[layer_base + 0], &norm_gamma);
    }

    var norm_x: [HIDDEN_DIM]f32 = undefined;
    rmsnorm.apply(hidden, &norm_gamma, &norm_x, qwen_geom.RMS_NORM_EPS);
    prof_norm_ns += nowNs() - t_norm_0;

    var norm_x_f16: [HIDDEN_DIM]f16 = undefined;
    if (mode == .raw_fp16) {
        var ci: usize = 0;
        while (ci < HIDDEN_DIM) : (ci += 8) {
            const vec: @Vector(8, f32) = norm_x[ci..][0..8].*;
            const f16_v: @Vector(8, f16) = @floatCast(vec);
            norm_x_f16[ci..][0..8].* = @bitCast(f16_v);
        }
    }

    // 2. Q, K, V Projections (Parallel 4 Workers)
    const t_qkv_0 = nowNs();
    var q: [HIDDEN_DIM]f32 = undefined;
    var k: [KV_DIM]f32 = undefined;
    var v: [KV_DIM]f32 = undefined;
    if (mode == .raw or mode == .raw_fp16) {
        @memcpy(&q, archive.getTileF32Direct(layer_base + 513)[0..HIDDEN_DIM]);
        @memcpy(&k, archive.getTileF32Direct(layer_base + 578)[0..KV_DIM]);
        @memcpy(&v, archive.getTileF32Direct(layer_base + 643)[0..KV_DIM]);
    } else if (mode == .dense) {
        unpack1DCell(&cells[layer_base + 513], &q);
        unpack1DCell(&cells[layer_base + 578], &k);
        unpack1DCell(&cells[layer_base + 643], &v);
    } else {
        unpack1D(&records[layer_base + 513], &q);
        unpack1D(&records[layer_base + 578], &k);
        unpack1D(&records[layer_base + 643], &v);
    }

    const QkvBf16Ctx = struct {
        raw_tiles: [*]const [8192]u16,
        records: [*]const geometry.Record,
        cells: [*]const geometry.Cell,
        mode: ArchiveMode,
        norm_x: []const f32,
        norm_x_f16: [*]const f16,
        q: []f32,
        k: []f32,
        v: []f32,
        layer_base: usize,
    };
    const qkv_ctx = QkvBf16Ctx{
        .raw_tiles = raw_tiles,
        .records = records,
        .cells = cells,
        .mode = mode,
        .norm_x = &norm_x,
        .norm_x_f16 = &norm_x_f16,
        .q = &q,
        .k = &k,
        .v = &v,
        .layer_base = layer_base,
    };
    pool.dispatch(struct {
        fn run(worker_id: usize, raw_ctx: *anyopaque) void {
            const c: *const QkvBf16Ctx = @ptrCast(@alignCast(raw_ctx));
            const q_start = worker_id * 128;
            const q_end = (worker_id + 1) * 128;
            const q_base = c.layer_base + 1;
            const k_start = worker_id * 16;
            const k_end = (worker_id + 1) * 16;
            const k_base = c.layer_base + 514;
            const v_start = worker_id * 16;
            const v_end = (worker_id + 1) * 16;
            const v_base = c.layer_base + 579;

            if (c.mode == .raw_fp16) {
                const tiles_f16: [*]const [8192]f16 = @ptrCast(c.raw_tiles);
                for (q_start..q_end) |t| {
                    const vals = gemvTileF16Direct(@ptrCast(&tiles_f16[q_base + t]), c.norm_x_f16);
                    inline for (0..4) |i| c.q[t * 4 + i] += vals[i];
                }
                for (k_start..k_end) |t| {
                    const vals = gemvTileF16Direct(@ptrCast(&tiles_f16[k_base + t]), c.norm_x_f16);
                    inline for (0..4) |i| c.k[t * 4 + i] += vals[i];
                }
                for (v_start..v_end) |t| {
                    const vals = gemvTileF16Direct(@ptrCast(&tiles_f16[v_base + t]), c.norm_x_f16);
                    inline for (0..4) |i| c.v[t * 4 + i] += vals[i];
                }
            } else if (c.mode == .raw) {
                for (q_start..q_end) |t| {
                    const vals = gemvTileBF16Direct(@ptrCast(&c.raw_tiles[q_base + t]), c.norm_x);
                    inline for (0..4) |i| c.q[t * 4 + i] += vals[i];
                }
                for (k_start..k_end) |t| {
                    const vals = gemvTileBF16Direct(@ptrCast(&c.raw_tiles[k_base + t]), c.norm_x);
                    inline for (0..4) |i| c.k[t * 4 + i] += vals[i];
                }
                for (v_start..v_end) |t| {
                    const vals = gemvTileBF16Direct(@ptrCast(&c.raw_tiles[v_base + t]), c.norm_x);
                    inline for (0..4) |i| c.v[t * 4 + i] += vals[i];
                }
            } else if (c.mode == .dense) {
                for (q_start..q_end) |t| {
                    gemvTileBF16CellDirect(&c.cells[q_base + t], c.norm_x, c.q, t * 4);
                }
                for (k_start..k_end) |t| {
                    gemvTileBF16CellDirect(&c.cells[k_base + t], c.norm_x, c.k, t * 4);
                }
                for (v_start..v_end) |t| {
                    gemvTileBF16CellDirect(&c.cells[v_base + t], c.norm_x, c.v, t * 4);
                }
            } else {
                for (q_start..q_end) |t| {
                    gemvTileBF16(&c.records[q_base + t], c.norm_x, c.q, t * 4);
                }
                for (k_start..k_end) |t| {
                    gemvTileBF16(&c.records[k_base + t], c.norm_x, c.k, t * 4);
                }
                for (v_start..v_end) |t| {
                    gemvTileBF16(&c.records[v_base + t], c.norm_x, c.v, t * 4);
                }
            }
        }
    }.run, @constCast(&qkv_ctx));
    prof_qkv_ns += nowNs() - t_qkv_0;

    // 3. RoPE on Q and K
    rope.applyRopeMultiHead(&q, qwen_geom.Q_HEADS, pos);
    rope.applyRopeMultiHead(&k, qwen_geom.KV_HEADS, pos);

    // 4. Update KV cache
    kv_cache.append(&k, &v);

    // 5. Grouped-Query Attention
    const t_attn_0 = nowNs();
    var attn_out: [HIDDEN_DIM]f32 = undefined;
    attn.forwardGqa(&q, kv_cache, &attn_out);
    prof_attn_ns += nowNs() - t_attn_0;

    // 6. Attention Output Projection + Residual (Parallel 4 Workers: 128 tiles each)
    const t_o_0 = nowNs();
    var o_out: [HIDDEN_DIM]f32 = @splat(0.0);
    var attn_out_f16: [HIDDEN_DIM]f16 = undefined;
    if (mode == .raw_fp16) {
        var ci: usize = 0;
        while (ci < HIDDEN_DIM) : (ci += 8) {
            const vec: @Vector(8, f32) = attn_out[ci..][0..8].*;
            const f16_v: @Vector(8, f16) = @floatCast(vec);
            attn_out_f16[ci..][0..8].* = @bitCast(f16_v);
        }
    }
    const OBf16Ctx = struct {
        raw_tiles: [*]const [8192]u16,
        records: [*]const geometry.Record,
        cells: [*]const geometry.Cell,
        mode: ArchiveMode,
        attn_out: []const f32,
        attn_out_f16: [*]const f16,
        o_out: []f32,
        o_base: usize,
    };
    const o_ctx = OBf16Ctx{
        .raw_tiles = raw_tiles,
        .records = records,
        .cells = cells,
        .mode = mode,
        .attn_out = &attn_out,
        .attn_out_f16 = &attn_out_f16,
        .o_out = &o_out,
        .o_base = layer_base + 644,
    };
    pool.dispatch(struct {
        fn run(worker_id: usize, raw_ctx: *anyopaque) void {
            const c: *const OBf16Ctx = @ptrCast(@alignCast(raw_ctx));
            const start_t = worker_id * 128;
            const end_t = (worker_id + 1) * 128;
            if (c.mode == .raw_fp16) {
                const tiles_f16: [*]const [8192]f16 = @ptrCast(c.raw_tiles);
                for (start_t..end_t) |t| {
                    const vals = gemvTileF16Direct(@ptrCast(&tiles_f16[c.o_base + t]), c.attn_out_f16);
                    inline for (0..4) |i| c.o_out[t * 4 + i] += vals[i];
                }
            } else if (c.mode == .raw) {
                for (start_t..end_t) |t| {
                    const vals = gemvTileBF16Direct(@ptrCast(&c.raw_tiles[c.o_base + t]), c.attn_out);
                    inline for (0..4) |i| c.o_out[t * 4 + i] += vals[i];
                }
            } else if (c.mode == .dense) {
                for (start_t..end_t) |t| {
                    gemvTileBF16CellDirect(&c.cells[c.o_base + t], c.attn_out, c.o_out, t * 4);
                }
            } else {
                for (start_t..end_t) |t| {
                    gemvTileBF16(&c.records[c.o_base + t], c.attn_out, c.o_out, t * 4);
                }
            }
        }
    }.run, @constCast(&o_ctx));
    mlp.addResidual(hidden, &o_out);
    prof_oproj_ns += nowNs() - t_o_0;

    // 7. Post-Attention RMSNorm
    const t_norm_1 = nowNs();
    if (mode == .raw or mode == .raw_fp16) {
        @memcpy(&norm_gamma, archive.getTileF32Direct(layer_base + 1156)[0..HIDDEN_DIM]);
    } else if (mode == .dense) {
        unpack1DCell(&cells[layer_base + 1156], &norm_gamma);
    } else {
        unpack1D(&records[layer_base + 1156], &norm_gamma);
    }
    rmsnorm.apply(hidden, &norm_gamma, &norm_x, qwen_geom.RMS_NORM_EPS);
    prof_norm_ns += nowNs() - t_norm_1;

    if (mode == .raw_fp16) {
        var ci: usize = 0;
        while (ci < HIDDEN_DIM) : (ci += 8) {
            const vec: @Vector(8, f32) = norm_x[ci..][0..8].*;
            const f16_v: @Vector(8, f16) = @floatCast(vec);
            norm_x_f16[ci..][0..8].* = @bitCast(f16_v);
        }
    }

    // 8. SwiGLU MLP: Gate and Up Projections (Parallel 4 Workers: 688 tiles each, Fused)
    const t_gu_0 = nowNs();
    var gate_buf: [INTERMEDIATE_DIM]f32 = @splat(0.0);
    const GuBf16Ctx = struct {
        raw_tiles: [*]const [8192]u16,
        records: [*]const geometry.Record,
        cells: [*]const geometry.Cell,
        mode: ArchiveMode,
        norm_x: []const f32,
        norm_x_f16: [*]const f16,
        gate_buf: []f32,
        gate_base: usize,
        up_base: usize,
    };
    const gu_ctx = GuBf16Ctx{
        .raw_tiles = raw_tiles,
        .records = records,
        .cells = cells,
        .mode = mode,
        .norm_x = &norm_x,
        .norm_x_f16 = &norm_x_f16,
        .gate_buf = &gate_buf,
        .gate_base = layer_base + 1157,
        .up_base = layer_base + 3909,
    };
    pool.dispatch(struct {
        fn run(worker_id: usize, raw_ctx: *anyopaque) void {
            const c: *const GuBf16Ctx = @ptrCast(@alignCast(raw_ctx));
            const start_t = worker_id * 688;
            const end_t = (worker_id + 1) * 688;

            if (c.mode == .raw_fp16) {
                const tiles_f16: [*]const [8192]f16 = @ptrCast(c.raw_tiles);
                for (start_t..end_t) |t| {
                    const b_idx = t * 4;
                    const g_vals = gemvTileF16Direct(@ptrCast(&tiles_f16[c.gate_base + t]), c.norm_x_f16);
                    c.gate_buf[b_idx..][0..4].* = g_vals;
                }
                for (start_t..end_t) |t| {
                    const b_idx = t * 4;
                    const u_vals = gemvTileF16Direct(@ptrCast(&tiles_f16[c.up_base + t]), c.norm_x_f16);
                    const g_v: @Vector(4, f32) = c.gate_buf[b_idx..][0..4].*;
                    const u_v: @Vector(4, f32) = u_vals;
                    c.gate_buf[b_idx..][0..4].* = mlp.siluVec4(g_v) * u_v;
                }
            } else if (c.mode == .raw) {
                for (start_t..end_t) |t| {
                    const b_idx = t * 4;
                    const g_vals = gemvTileBF16Direct(@ptrCast(&c.raw_tiles[c.gate_base + t]), c.norm_x);
                    c.gate_buf[b_idx..][0..4].* = g_vals;
                }
                for (start_t..end_t) |t| {
                    const b_idx = t * 4;
                    const u_vals = gemvTileBF16Direct(@ptrCast(&c.raw_tiles[c.up_base + t]), c.norm_x);
                    const g_v: @Vector(4, f32) = c.gate_buf[b_idx..][0..4].*;
                    const u_v: @Vector(4, f32) = u_vals;
                    c.gate_buf[b_idx..][0..4].* = mlp.siluVec4(g_v) * u_v;
                }
            } else {
                for (start_t..end_t) |t| {
                    if (t + 1 < end_t) {
                        if (c.mode == .dense) {
                            prefetchCell(&c.cells[c.gate_base + t + 1]);
                            prefetchCell(&c.cells[c.up_base + t + 1]);
                        } else {
                            prefetchRecord(&c.records[c.gate_base + t + 1]);
                            prefetchRecord(&c.records[c.up_base + t + 1]);
                        }
                    }
                    const b_idx = t * 4;
                    const g_u16: [*]const u16 = if (c.mode == .dense)
                        @ptrCast(@alignCast(&c.cells[c.gate_base + t].fingerprints))
                    else
                        @ptrCast(@alignCast(&c.records[c.gate_base + t].cell.fingerprints));
                    const u_u16: [*]const u16 = if (c.mode == .dense)
                        @ptrCast(@alignCast(&c.cells[c.up_base + t].fingerprints))
                    else
                        @ptrCast(@alignCast(&c.records[c.up_base + t].cell.fingerprints));

                    const g_vals = gemvTileSequentialBF16(g_u16, c.norm_x);
                    const u_vals = gemvTileSequentialBF16(u_u16, c.norm_x);
                    const g_v: @Vector(4, f32) = g_vals;
                    const u_v: @Vector(4, f32) = u_vals;
                    c.gate_buf[b_idx..][0..4].* = mlp.siluVec4(g_v) * u_v;
                }
            }
        }
    }.run, @constCast(&gu_ctx));
    prof_gateup_ns += nowNs() - t_gu_0;

    // 9. Down Projection + Residual (Parallel 4 Workers: 688 tiles each)
    const t_down_0 = nowNs();
    var gate_buf_f16: [INTERMEDIATE_DIM]f16 = undefined;
    if (mode == .raw_fp16) {
        var ci: usize = 0;
        while (ci < INTERMEDIATE_DIM) : (ci += 8) {
            const vec: @Vector(8, f32) = gate_buf[ci..][0..8].*;
            const f16_v: @Vector(8, f16) = @floatCast(vec);
            gate_buf_f16[ci..][0..8].* = @bitCast(f16_v);
        }
    }
    const DownBf16Ctx = struct {
        raw_tiles: [*]const [8192]u16,
        records: [*]const geometry.Record,
        cells: [*]const geometry.Cell,
        mode: ArchiveMode,
        gate_buf: []const f32,
        gate_buf_f16: [*]const f16,
        hidden: []f32,
        down_base: usize,
    };
    const down_ctx = DownBf16Ctx{
        .raw_tiles = raw_tiles,
        .records = records,
        .cells = cells,
        .mode = mode,
        .gate_buf = &gate_buf,
        .gate_buf_f16 = &gate_buf_f16,
        .hidden = hidden,
        .down_base = layer_base + 6661,
    };
    pool.dispatch(struct {
        fn run(worker_id: usize, raw_ctx: *anyopaque) void {
            const c: *const DownBf16Ctx = @ptrCast(@alignCast(raw_ctx));
            const start_t = worker_id * 688;
            const end_t = (worker_id + 1) * 688;

            if (c.mode == .raw_fp16) {
                const worker_weights: [*]const f16 = @ptrCast(&c.raw_tiles[c.down_base + start_t]);
                const x_ptr: [*]const f16 = c.gate_buf_f16;
                for (0..128) |quad_idx| {
                    const local_r = quad_idx * 4;
                    const r = worker_id * 512 + local_r;
                    const res = dotQuadRow11008F16(
                        worker_weights + (local_r + 0) * INTERMEDIATE_DIM,
                        worker_weights + (local_r + 1) * INTERMEDIATE_DIM,
                        worker_weights + (local_r + 2) * INTERMEDIATE_DIM,
                        worker_weights + (local_r + 3) * INTERMEDIATE_DIM,
                        x_ptr,
                    );
                    c.hidden[r + 0] += res[0];
                    c.hidden[r + 1] += res[1];
                    c.hidden[r + 2] += res[2];
                    c.hidden[r + 3] += res[3];
                }
            } else if (c.mode == .raw) {
                const worker_weights: [*]const u16 = @ptrCast(&c.raw_tiles[c.down_base + start_t]);
                const x_ptr: [*]const f32 = c.gate_buf.ptr;
                for (0..128) |quad_idx| {
                    const local_r = quad_idx * 4;
                    const r = worker_id * 512 + local_r;
                    const res = dotQuadRow11008BF16(
                        worker_weights + (local_r + 0) * INTERMEDIATE_DIM,
                        worker_weights + (local_r + 1) * INTERMEDIATE_DIM,
                        worker_weights + (local_r + 2) * INTERMEDIATE_DIM,
                        worker_weights + (local_r + 3) * INTERMEDIATE_DIM,
                        x_ptr,
                    );
                    c.hidden[r + 0] += res[0];
                    c.hidden[r + 1] += res[1];
                    c.hidden[r + 2] += res[2];
                    c.hidden[r + 3] += res[3];
                }
            } else {
                for (start_t..end_t) |t| {
                    if (t + 1 < end_t) {
                        if (c.mode == .dense) prefetchCell(&c.cells[c.down_base + t + 1])
                        else prefetchRecord(&c.records[c.down_base + t + 1]);
                    }
                    if (c.mode == .dense) {
                        gemvTileDownBF16CellDirect(&c.cells[c.down_base + t], c.gate_buf, c.hidden, t);
                    } else {
                        gemvTileDownBF16CellDirect(&c.records[c.down_base + t].cell, c.gate_buf, c.hidden, t);
                    }
                }
            }
        }
    }.run, @constCast(&down_ctx));
    prof_down_ns += nowNs() - t_down_0;
}

var static_kv_caches: [NUM_LAYERS]attn.KvCache = undefined;
var static_logits: [VOCAB_SIZE]f32 = undefined;
var static_logits_f64: [VOCAB_SIZE]f64 = undefined;
var static_probs_f64: [VOCAB_SIZE]f64 = undefined;

pub fn decodeSequenceWithHead(
    archive_path: [*:0]const u8,
    tokens: []const u32,
    head_mode: HeadMode,
) !ForwardResult {
    var archive = try weight_archive.WeightArchive.openPosix(archive_path);
    defer archive.close();
    return decodeSequenceWithArchive(&archive, tokens, head_mode);
}

pub fn decodeSequenceWithArchive(
    archive: *const weight_archive.WeightArchive,
    tokens: []const u32,
    head_mode: HeadMode,
) !ForwardResult {
    std.debug.assert(tokens.len > 0);
    const t0 = nowNs();

    // Verify header magic
    if (archive.header().magic != weight_archive.CHPE_MAGIC and archive.header().magic != weight_archive.ARCHIVE_MAGIC) {
        return error.InvalidMagic;
    }

    const is_hybrid: bool = (archive.header().flags == weight_archive.FLAG_HYBRID_ARCHIVE or archive.header().flags == 0x02 or archive.header().record_count == qwen_geom.HybridLayerOffsets.TOTAL_RECORDS);
    const is_bf16: bool = (!is_hybrid and (archive.header().flags == 16 or archive.header().record_count == qwen_geom.Bf16LayerOffsets.TOTAL_RECORDS));
    if (!is_bf16 and !is_hybrid) {
        // precomputed norms loaded directly
    }

    // Step 1: Initialize 36 layer KV caches
    for (0..NUM_LAYERS) |l| {
        static_kv_caches[l].reset();
    }

    prof_qkv_ns = 0;
    prof_attn_ns = 0;
    prof_oproj_ns = 0;
    prof_norm_ns = 0;
    prof_gateup_ns = 0;
    prof_down_ns = 0;
    prof_head_ns = 0;

    var hidden: [HIDDEN_DIM]f32 = undefined;

    // Step 2: Forward decode each token sequentially
    for (tokens, 0..) |tok_id, pos| {
        std.debug.assert(tok_id < VOCAB_SIZE);
        if (is_bf16) {
            const rec_idx = tok_id / 4;
            const row_in_tile = tok_id % 4;
            if (archive.isRawFp16()) {
                const tile_f16 = archive.getTileF16Direct(rec_idx);
                unpackEmbedRowF16Direct(tile_f16, row_in_tile, &hidden);
            } else {
                const tile_u16 = archive.getTileU16Direct(rec_idx);
                unpackEmbedRowBF16Direct(tile_u16, row_in_tile, &hidden);
            }

            for (0..NUM_LAYERS) |l| {
                try forwardLayerBF16(archive, l, @intCast(pos), &hidden, &static_kv_caches[l]);
            }
        } else if (is_hybrid) {
            const rec_idx = tok_id / 16;
            const row_in_tile = tok_id % 16;
            const emb_cell = archive.getCellDirect(qwen_geom.HybridLayerOffsets.EMBED_TOKENS_START + rec_idx);
            unpackEmbedRowCell(emb_cell, row_in_tile, &hidden);

            for (0..NUM_LAYERS) |l| {
                try forwardLayerHybrid(archive, l, @intCast(pos), &hidden, &static_kv_caches[l]);
            }
        } else {
            const rec_idx = tok_id / 16;
            const row_in_tile = tok_id % 16;
            const emb_cell = archive.getCellDirect(tensor_map.EMBED_TOKENS_START + rec_idx);
            unpackEmbedRowCell(emb_cell, row_in_tile, &hidden);

            for (0..NUM_LAYERS) |l| {
                try forwardLayer(archive, l, @intCast(pos), &hidden, &static_kv_caches[l]);
            }
        }
    }

    // Step 3: Final RMSNorm
    if (is_bf16) {
        var final_gamma: [HIDDEN_DIM]f32 = undefined;
        const norm_f32 = archive.getTileF32Direct(qwen_geom.Bf16LayerOffsets.FINAL_NORM_RECORD);
        @memcpy(&final_gamma, norm_f32[0..HIDDEN_DIM]);
        rmsnorm.applyInPlace(&hidden, &final_gamma, qwen_geom.RMS_NORM_EPS);
    } else if (is_hybrid) {
        const final_norm_cell = archive.getCellDirect(qwen_geom.HybridLayerOffsets.FINAL_NORM_RECORD);
        var final_gamma: [HIDDEN_DIM]f32 = undefined;
        unpack1DCell(final_norm_cell, &final_gamma);
        rmsnorm.applyInPlace(&hidden, &final_gamma, qwen_geom.RMS_NORM_EPS);
    } else {
        const final_norm_cell = archive.getCellDirect(tensor_map.FINAL_NORM_RECORD);
        var final_gamma: [HIDDEN_DIM]f32 = undefined;
        unpack1DCell(final_norm_cell, &final_gamma);
        rmsnorm.applyInPlace(&hidden, &final_gamma, qwen_geom.RMS_NORM_EPS);
    }

    // Compute hidden L2 norm for verification
    var sum_sq: f32 = 0.0;
    for (hidden) |h| sum_sq += h * h;
    const hidden_norm = @sqrt(sum_sq);

    // Step 4: Head dispatch based on HeadMode
    var max_logit: f32 = -std.math.inf(f32);
    var argmax_token: u32 = 0;
    var token0_logit: f32 = 0.0;
    var all_finite: bool = true;

    if (is_bf16) {
        const t_head_0 = nowNs();
        const pool = getWorkerPool();
        const mode = getArchiveMode(archive);
        const raw_tiles: [*]const [8192]u16 = if (mode == .raw or mode == .raw_fp16) @ptrCast(@alignCast(archive.bytes.ptr + weight_archive.HEADER_BYTES)) else undefined;
        const records = if (mode == .standard) archive.getRecordsPtr() else undefined;
        const cells = if (mode == .dense) archive.getCellsPtr() else undefined;

        var hidden_f16: [HIDDEN_DIM]f16 = undefined;
        if (mode == .raw_fp16) {
            var ci: usize = 0;
            while (ci < HIDDEN_DIM) : (ci += 8) {
                const vec: @Vector(8, f32) = hidden[ci..][0..8].*;
                const f16_v: @Vector(8, f16) = @floatCast(vec);
                hidden_f16[ci..][0..8].* = @bitCast(f16_v);
            }
        }

        const HeadBf16Ctx = struct {
            raw_tiles: [*]const [8192]u16,
            records: [*]const geometry.Record,
            cells: [*]const geometry.Cell,
            mode: ArchiveMode,
            hidden: *const [HIDDEN_DIM]f32,
            hidden_f16: [*]const f16,
            local_max: [4]f32 = @splat(-std.math.inf(f32)),
            local_argmax: [4]u32 = @splat(0),
            local_finite: [4]bool = @splat(true),
            token0_logit: f32 = 0.0,
        };
        var head_ctx = HeadBf16Ctx{
            .raw_tiles = raw_tiles,
            .records = records,
            .cells = cells,
            .mode = mode,
            .hidden = &hidden,
            .hidden_f16 = &hidden_f16,
        };

        pool.dispatch(struct {
            fn run(worker_id: usize, raw_ctx: *anyopaque) void {
                const c: *HeadBf16Ctx = @ptrCast(@alignCast(raw_ctx));
                const total_records = qwen_geom.Bf16LayerOffsets.EMBED_TOKENS_RECORDS;
                const recs_per_worker = total_records / 4;
                const start_rec = worker_id * recs_per_worker;
                const end_rec = if (worker_id == 3) total_records else (worker_id + 1) * recs_per_worker;

                var w_max: f32 = -std.math.inf(f32);
                var w_argmax: u32 = 0;
                var w_finite: bool = true;

                if (c.mode == .raw_fp16) {
                    const tiles_f16: [*]const [8192]f16 = @ptrCast(c.raw_tiles);
                    for (start_rec..end_rec) |rec_idx| {
                        const logits = gemvTileF16Direct(
                            @ptrCast(&tiles_f16[rec_idx]),
                            c.hidden_f16,
                        );

                        const base_tok: u32 = @intCast(rec_idx * 4);
                        inline for (logits, 0..) |logit, row| {
                            const token_id = base_tok + @as(u32, @intCast(row));
                            if (token_id < VOCAB_SIZE) {
                                static_logits[token_id] = logit;
                                if (!std.math.isFinite(logit)) w_finite = false;
                                if (token_id == 0) c.token0_logit = logit;
                                if (logit > w_max) {
                                    w_max = logit;
                                    w_argmax = token_id;
                                }
                            }
                        }
                    }
                } else if (c.mode == .raw) {
                    for (start_rec..end_rec) |rec_idx| {
                        const logits = gemvTileBF16Direct(
                            @ptrCast(&c.raw_tiles[rec_idx]),
                            c.hidden,
                        );

                        const base_tok: u32 = @intCast(rec_idx * 4);
                        inline for (logits, 0..) |logit, row| {
                            const token_id = base_tok + @as(u32, @intCast(row));
                            if (token_id < VOCAB_SIZE) {
                                static_logits[token_id] = logit;
                                if (!std.math.isFinite(logit)) w_finite = false;
                                if (token_id == 0) c.token0_logit = logit;
                                if (logit > w_max) {
                                    w_max = logit;
                                    w_argmax = token_id;
                                }
                            }
                        }
                    }
                } else {
                    for (start_rec..end_rec) |rec_idx| {
                        if (rec_idx + 1 < end_rec) {
                            if (c.mode == .dense) {
                                prefetchCell(&c.cells[rec_idx + 1]);
                            } else {
                                prefetchRecord(&c.records[rec_idx + 1]);
                            }
                        }
                        const coded_u16: [*]const u16 = if (c.mode == .dense)
                            @ptrCast(@alignCast(&c.cells[rec_idx].fingerprints))
                        else
                            @ptrCast(@alignCast(&c.records[rec_idx].cell.fingerprints));

                        const logits = gemvTileSequentialBF16(coded_u16, c.hidden);

                        const base_tok: u32 = @intCast(rec_idx * 4);
                        inline for (logits, 0..) |logit, row| {
                            const token_id = base_tok + @as(u32, @intCast(row));
                            if (token_id < VOCAB_SIZE) {
                                static_logits[token_id] = logit;
                                if (!std.math.isFinite(logit)) w_finite = false;
                                if (token_id == 0) c.token0_logit = logit;
                                if (logit > w_max) {
                                    w_max = logit;
                                    w_argmax = token_id;
                                }
                            }
                        }
                    }
                }
                c.local_max[worker_id] = w_max;
                c.local_argmax[worker_id] = w_argmax;
                c.local_finite[worker_id] = w_finite;
            }
        }.run, @constCast(&head_ctx));

        max_logit = head_ctx.local_max[0];
        argmax_token = head_ctx.local_argmax[0];
        all_finite = head_ctx.local_finite[0];
        for (1..4) |w| {
            if (head_ctx.local_max[w] > max_logit) {
                max_logit = head_ctx.local_max[w];
                argmax_token = head_ctx.local_argmax[w];
            }
            if (!head_ctx.local_finite[w]) {
                all_finite = false;
            }
        }
        token0_logit = head_ctx.token0_logit;
        prof_head_ns += nowNs() - t_head_0;
    } else {
        @memset(static_logits[0..VOCAB_SIZE], 0.0);
        const pool = getWorkerPool();
        Activation2048.initInto(&global_act2048, &hidden);

        const is_dense = archive.isDense();
        const records = if (!is_dense) archive.getRecordsPtr() else undefined;
        const cells = if (is_dense) archive.getCellsPtr() else undefined;

        const EmbedCtx = struct {
            records: [*]const geometry.Record,
            cells: [*]const geometry.Cell,
            is_dense: bool,
            is_hybrid: bool,
            act: *const Activation2048,
            logits: []f32,
            local_max: [4]f32 = @splat(-std.math.inf(f32)),
            local_argmax: [4]u32 = @splat(0),
            local_finite: [4]bool = @splat(true),
        };
        var embed_ctx = EmbedCtx{
            .records = records,
            .cells = cells,
            .is_dense = is_dense,
            .is_hybrid = is_hybrid,
            .act = &global_act2048,
            .logits = static_logits[0..VOCAB_SIZE],
        };
        const t_head0 = nowNs();
        pool.dispatch(struct {
            fn run(worker_id: usize, raw_ctx: *anyopaque) void {
                const c: *EmbedCtx = @ptrCast(@alignCast(raw_ctx));
                var w_max: f32 = -std.math.inf(f32);
                var w_argmax: u32 = 0;
                var w_finite: bool = true;

                if (c.is_hybrid or c.is_dense) {
                    const start_rec = worker_id * 2374;
                    const end_rec = (worker_id + 1) * 2374;
                    if (start_rec + 1 < end_rec) {
                        prefetchCell(&c.cells[start_rec + 1]);
                    }
                    for (start_rec..end_rec) |rec_idx| {
                        if (rec_idx + 2 < end_rec) {
                            prefetchCell(&c.cells[rec_idx + 2]);
                        } else if (rec_idx + 1 < end_rec) {
                            prefetchCell(&c.cells[rec_idx + 1]);
                        }
                        const cell = &c.cells[rec_idx];
                        const base_row = rec_idx * 16;
                        gemvTileLMHead8RowWithMax(cell, c.act, c.logits, base_row, USE_INT8_SDOT, &w_max, &w_argmax, &w_finite);
                    }
                } else {
                    const start_rec = worker_id * 2374;
                    const end_rec = (worker_id + 1) * 2374;
                    for (start_rec..end_rec) |rec_idx| {
                        if (rec_idx + 1 < end_rec) {
                            prefetchRecord(&c.records[rec_idx + 1]);
                        }
                        const rec = &c.records[rec_idx];
                        const base_row = rec_idx * 16;
                        gemvTileQuadRowDirect(rec, c.act, c.logits, base_row, false);
                        const v0: @Vector(8, f32) = c.logits[base_row..][0..8].*;
                        const v1: @Vector(8, f32) = c.logits[base_row + 8..][0..8].*;
                        const rec_max = @reduce(.Max, @max(v0, v1));
                        if (!std.math.isFinite(rec_max)) w_finite = false;
                        if (rec_max > w_max) {
                            inline for (0..16) |offset| {
                                const idx: u32 = @intCast(base_row + offset);
                                if (idx < VOCAB_SIZE) {
                                    const val = c.logits[idx];
                                    if (val > w_max) {
                                        w_max = val;
                                        w_argmax = idx;
                                    }
                                }
                            }
                        }
                    }
                }
                c.local_max[worker_id] = w_max;
                c.local_argmax[worker_id] = w_argmax;
                c.local_finite[worker_id] = w_finite;
            }
        }.run, @ptrCast(&embed_ctx));
        prof_head_ns += nowNs() - t_head0;

        token0_logit = static_logits[0];
        for (0..4) |w| {
            if (!embed_ctx.local_finite[w]) all_finite = false;
            if (embed_ctx.local_max[w] > max_logit) {
                max_logit = embed_ctx.local_max[w];
                argmax_token = embed_ctx.local_argmax[w];
            }
        }
    }


    if (head_mode == .generative_f64) {
        for (static_logits[0..VOCAB_SIZE], 0..) |logit, idx| {
            static_logits_f64[idx] = @as(f64, logit);
        }
        stableSoftmaxF64(static_logits_f64[0..VOCAB_SIZE], static_probs_f64[0..VOCAB_SIZE]);
    }

    const elapsed = nowNs() - t0;

    return .{
        .argmax_token = argmax_token,
        .max_logit = max_logit,
        .token0_logit = token0_logit,
        .elapsed_ns = elapsed,
        .hidden_norm = hidden_norm,
        .all_finite = all_finite,
        .logits = static_logits[0..VOCAB_SIZE],
    };
}

pub fn decodeSequence(archive_path: [*:0]const u8, tokens: []const u32) !ForwardResult {
    return decodeSequenceWithHead(archive_path, tokens, .generative_f32);
}

pub fn decodeToken0(archive_path: [*:0]const u8) !ForwardResult {
    const default_tok = [_]u32{0};
    return decodeSequence(archive_path, &default_tok);
}

test "f64 vocabulary softmax stability" {
    const logits = [_]f64{ 1.0, 2.0, 3.0 };
    var probs: [3]f64 = undefined;
    stableSoftmaxF64(&logits, &probs);
    var sum: f64 = 0.0;
    for (probs) |p| sum += p;
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), sum, 1e-6);
}

test "gemvTileQuadRow INT8 SDOT vs FP32 numerical parity" {
    var mock_record: geometry.Record = undefined;
    @memset(std.mem.asBytes(&mock_record), 0);
    const meta: *weight_archive.TileMetadata = @ptrCast(@alignCast(&mock_record.cell.semantic_payload));
    meta.custom_flags = weight_archive.FLAG_GROUP128_OUTLIERS;

    const group_scales_raw: [*]f16 = @ptrCast(@alignCast(mock_record.cell.semantic_payload[48..560].ptr));
    for (0..256) |i| {
        group_scales_raw[i] = @floatCast(0.005);
    }

    const coded: [*]u8 = @ptrCast(&mock_record.cell.fingerprints);
    for (0..16384) |i| {
        coded[i] = @intCast((i * 37 + 13) % 256);
    }

    var in_vec: [2048]f32 = undefined;
    for (0..2048) |i| {
        in_vec[i] = @sin(@as(f32, @floatFromInt(i)) * 0.1);
    }

    const act = Activation2048.init(&in_vec);
    var out_int8: [16]f32 = @splat(0.0);
    gemvTileQuadRow(&mock_record, &act, &out_int8, 0);

    var out_fp32: [16]f32 = @splat(0.0);
    for (0..16) |r| {
        var dot: f32 = 0.0;
        const row_bytes = coded[r * 1024 .. (r + 1) * 1024];
        for (0..16) |g| {
            const s: f32 = @floatCast(group_scales_raw[r * 16 + g]);
            const g_b = row_bytes[g * 64 .. (g + 1) * 64];
            var raw_sum: f32 = 0.0;
            for (0..64) |k| {
                const b = g_b[k];
                const w0: f32 = @floatFromInt(b & 0x0F);
                const w1: f32 = @floatFromInt(b >> 4);
                raw_sum += w0 * in_vec[g * 128 + 2 * k] + w1 * in_vec[g * 128 + 2 * k + 1];
            }
            dot += (raw_sum - 8.0 * act.sums[g]) * s;
        }
        out_fp32[r] = dot;
    }

    for (0..16) |r| {
        const diff = @abs(out_int8[r] - out_fp32[r]);
        const rel = diff / (@abs(out_fp32[r]) + 1e-6);
        std.debug.print("Row {d:2}: FP32={d:.4} INT8={d:.4} diff={d:.4} rel={d:.4}%\n", .{ r, out_fp32[r], out_int8[r], diff, rel * 100.0 });
        try std.testing.expect(diff < 0.05);
    }
}


test "gemvTileBF16 matches known float math" {
    var mock_record: geometry.Record = undefined;
    @memset(std.mem.asBytes(&mock_record), 0);
    const u16_ptr: [*]u16 = @ptrCast(@alignCast(&mock_record.cell.fingerprints));

    // BF16 1.0 is 0x3F80, 2.0 is 0x4000
    for (0..4) |r| {
        for (0..2048) |c| {
            u16_ptr[r * 2048 + c] = if (r % 2 == 0) 0x3F80 else 0x4000;
        }
    }

    var in_vec: [2048]f32 = @splat(0.5);
    var out_vec: [4]f32 = @splat(0.0);

    gemvTileBF16(&mock_record, &in_vec, &out_vec, 0);

    // Row 0: 2048 * (1.0 * 0.5) = 1024.0
    // Row 1: 2048 * (2.0 * 0.5) = 2048.0
    try std.testing.expectApproxEqAbs(@as(f32, 1024.0), out_vec[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 2048.0), out_vec[1], 1e-3);
}

test "gemvTileDownDeintCell8Acc parity vs gemvTileDownDeintCellOpt" {
    var dummy_cell: geometry.Cell = undefined;
    @memset(std.mem.asBytes(&dummy_cell), 0);
    const meta: *weight_archive.TileMetadata = @ptrCast(@alignCast(&dummy_cell.semantic_payload));
    meta.custom_flags = weight_archive.FLAG_GROUP128_OUTLIERS;

    var raw_buf: [INTERMEDIATE_DIM]f32 = @splat(0.0);
    var dummy_act: Activation11008 = undefined;
    @memset(std.mem.asBytes(&dummy_act), 0);
    dummy_act.raw = &raw_buf;
    dummy_act.scales[0] = 1.0;

    var out_opt: [HIDDEN_DIM]f32 = @splat(0.0);
    var out_8row: [HIDDEN_DIM]f32 = @splat(0.0);

    gemvTileDownDeintCellOpt(&dummy_cell, &dummy_act, &out_opt, 0, false);
    gemvTileDownDeintCell8Acc(&dummy_cell, &dummy_act, &out_8row, 0, false);

    try std.testing.expectEqualSlices(f32, out_opt[0..16], out_8row[0..16]);
}

test "gemvTileDownDeintCell8Acc disjoint row partition parity" {
    var cell0: geometry.Cell = undefined;
    @memset(std.mem.asBytes(&cell0), 0);
    const meta0: *weight_archive.TileMetadata = @ptrCast(@alignCast(&cell0.semantic_payload));
    meta0.custom_flags = weight_archive.FLAG_GROUP128_OUTLIERS;

    var cell1: geometry.Cell = undefined;
    @memset(std.mem.asBytes(&cell1), 0);
    const meta1: *weight_archive.TileMetadata = @ptrCast(@alignCast(&cell1.semantic_payload));
    meta1.custom_flags = weight_archive.FLAG_GROUP128_OUTLIERS;

    // Provide non-zero group scales so arithmetic is exercized
    const group_scales0: [*]f16 = @ptrCast(@alignCast(cell0.semantic_payload[48..560].ptr));
    const group_scales1: [*]f16 = @ptrCast(@alignCast(cell1.semantic_payload[48..560].ptr));
    for (0..256) |g| {
        group_scales0[g] = 0.05;
        group_scales1[g] = 0.02;
    }

    var raw_act: [INTERMEDIATE_DIM]f32 = undefined;
    for (&raw_act, 0..) |*x, idx| {
        x.* = @floatFromInt(@as(i32, @intCast(idx % 7)) - 3);
    }
    var act: Activation11008 = undefined;
    @memset(std.mem.asBytes(&act), 0);
    act.raw = &raw_act;
    for (&act.scales) |*s| s.* = 1.0;
    for (&act.sums) |*sm| sm.* = 0.0;

    // Single accumulation
    var out_combined: [HIDDEN_DIM]f32 = @splat(0.0);
    gemvTileDownDeintCell8Acc(&cell0, &act, &out_combined, 0, false);
    gemvTileDownDeintCell8Acc(&cell1, &act, &out_combined, 1, false);

    // Disjoint partitioned accumulation across 2 worker buffers
    var out_worker0: [HIDDEN_DIM]f32 = @splat(0.0);
    var out_worker1: [HIDDEN_DIM]f32 = @splat(0.0);
    gemvTileDownDeintCell8Acc(&cell0, &act, &out_worker0, 0, false);
    gemvTileDownDeintCell8Acc(&cell1, &act, &out_worker1, 1, false);

    var out_summed: [HIDDEN_DIM]f32 = @splat(0.0);
    for (0..HIDDEN_DIM) |r| {
        out_summed[r] = out_worker0[r] + out_worker1[r];
    }

    // Parity: disjoint row partition accumulation must bit-exactly match single accumulation
    try std.testing.expectEqualSlices(f32, out_combined[0..16], out_summed[0..16]);
}

test "gemvTileGateUpFusedDirect parity" {
    var g_cell: geometry.Cell = undefined;
    @memset(std.mem.asBytes(&g_cell), 0);
    const g_meta: *weight_archive.TileMetadata = @ptrCast(@alignCast(&g_cell.semantic_payload));
    g_meta.custom_flags = weight_archive.FLAG_GROUP128_OUTLIERS;

    var u_cell: geometry.Cell = undefined;
    @memset(std.mem.asBytes(&u_cell), 0);
    const u_meta: *weight_archive.TileMetadata = @ptrCast(@alignCast(&u_cell.semantic_payload));
    u_meta.custom_flags = weight_archive.FLAG_GROUP128_OUTLIERS;

    var raw_x: [HIDDEN_DIM]f32 = @splat(0.5);
    var act: Activation2048 = undefined;
    Activation2048.initInto(&act, &raw_x);

    var gate_buf: [16]f32 = @splat(0.0);
    var up_buf: [16]f32 = @splat(0.0);
    var fused_buf: [16]f32 = @splat(0.0);

    gemvTileQuadRowCellDirect(&g_cell, &act, &gate_buf, 0, false, false);
    gemvTileQuadRowCellDirect(&u_cell, &act, &up_buf, 0, false, false);
    mlp.swigluForward(&gate_buf, &up_buf, &gate_buf);

    gemvTileGateUpFusedDirect(&g_cell, &u_cell, &act, &fused_buf, 0, false);

    try std.testing.expectEqualSlices(f32, gate_buf[0..16], fused_buf[0..16]);
}

test "gemvTileLMHead8RowWithMax parity" {
    var cell: geometry.Cell = undefined;
    @memset(std.mem.asBytes(&cell), 0);
    const meta: *weight_archive.TileMetadata = @ptrCast(@alignCast(&cell.semantic_payload));
    meta.custom_flags = weight_archive.FLAG_GROUP128_OUTLIERS;

    var raw_x: [HIDDEN_DIM]f32 = @splat(0.5);
    var act: Activation2048 = undefined;
    Activation2048.initInto(&act, &raw_x);

    var logits: [16]f32 = @splat(0.0);
    var w_max: f32 = -std.math.inf(f32);
    var w_argmax: u32 = 0;
    var w_finite: bool = true;

    gemvTileLMHead8RowWithMax(&cell, &act, &logits, 0, false, &w_max, &w_argmax, &w_finite);
    try std.testing.expect(w_finite);
    try std.testing.expect(std.math.isFinite(w_max));
}

test "gemvTileW2CellDirect synthetic numerical parity" {
    var cell: geometry.Cell = undefined;
    @memset(std.mem.asBytes(&cell), 0);
    const meta: *weight_archive.TileMetadata = @ptrCast(@alignCast(&cell.semantic_payload));
    meta.quant_bits = 2;
    meta.custom_flags = 0;

    const group_scales_raw: [*]f16 = @ptrCast(@alignCast(cell.semantic_payload[48..560].ptr));
    for (0..256) |i| {
        group_scales_raw[i] = 1.0;
    }

    const coded: [*]u8 = @ptrCast(&cell.fingerprints);
    coded[0] = 0x55;

    var raw_x: [HIDDEN_DIM]f32 = @splat(1.0);
    var act: Activation2048 = undefined;
    Activation2048.initInto(&act, &raw_x);

    var y: [16]f32 = @splat(0.0);
    gemvTileW2CellDirect(&cell, &act, &y, 0, false);

    try std.testing.expectApproxEqAbs(@as(f32, 4.0), y[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), y[1], 1e-6);
}

test "gemvTileW2CellDirect with outliers" {
    var cell: geometry.Cell = undefined;
    @memset(std.mem.asBytes(&cell), 0);
    const meta: *weight_archive.TileMetadata = @ptrCast(@alignCast(&cell.semantic_payload));
    meta.quant_bits = 2;
    meta.custom_flags = weight_archive.FLAG_GROUP128_OUTLIERS;

    const group_scales_raw: [*]f16 = @ptrCast(@alignCast(cell.semantic_payload[48..560].ptr));
    for (0..256) |i| {
        group_scales_raw[i] = 1.0;
    }

    const outlier_w_raw: [*]f16 = @ptrCast(@alignCast(cell.semantic_payload[560..816].ptr));
    const outlier_cols_raw: [*]u16 = @ptrCast(@alignCast(cell.semantic_payload[816..832].ptr));
    outlier_cols_raw[0] = 10;
    outlier_w_raw[0] = 5.0; // row 0 outlier 0 is weight 5.0 at col 10

    var raw_x: [HIDDEN_DIM]f32 = @splat(0.0);
    raw_x[10] = 2.0; // activation at col 10 is 2.0
    var act: Activation2048 = undefined;
    Activation2048.initInto(&act, &raw_x);

    var y: [16]f32 = @splat(0.0);
    gemvTileW2CellDirect(&cell, &act, &y, 0, false);

    // Expected: 5.0 * 2.0 = 10.0
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), y[0], 1e-6);
}

test "gemvTileDownW2 synthetic numerical parity" {
    var cell: geometry.Cell = undefined;
    @memset(std.mem.asBytes(&cell), 0);
    const meta: *weight_archive.TileMetadata = @ptrCast(@alignCast(&cell.semantic_payload));
    meta.quant_bits = 2;
    meta.custom_flags = 0;

    const group_scales_raw: [*]f16 = @ptrCast(@alignCast(cell.semantic_payload[48..560].ptr));
    for (0..256) |i| {
        group_scales_raw[i] = 1.0;
    }

    const coded: [*]u8 = @ptrCast(&cell.fingerprints);
    coded[0] = 0x55; // four +1.0 weights in group 0

    var raw_x: [INTERMEDIATE_DIM]f32 = @splat(1.0);
    var act: Activation11008 = undefined;
    act.raw = &raw_x;

    var y: [HIDDEN_DIM]f32 = @splat(0.0);
    gemvTileDownW2(&cell, &act, &y, 0);

    // Group 0 belongs to row 0. 4 * 1.0 = 4.0
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), y[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), y[1], 1e-6);
}

test "gemvTileDownW2 with outliers" {
    var cell: geometry.Cell = undefined;
    @memset(std.mem.asBytes(&cell), 0);
    const meta: *weight_archive.TileMetadata = @ptrCast(@alignCast(&cell.semantic_payload));
    meta.quant_bits = 2;
    meta.custom_flags = weight_archive.FLAG_GROUP128_OUTLIERS;

    const group_scales_raw: [*]f16 = @ptrCast(@alignCast(cell.semantic_payload[48..560].ptr));
    for (0..256) |i| {
        group_scales_raw[i] = 1.0;
    }

    std.mem.writeInt(u16, cell.semantic_payload[560..562], 1, .little);
    const outlier_w_raw: [*]f16 = @ptrCast(@alignCast(cell.semantic_payload[562..690].ptr));
    const outlier_offsets_raw: [*]u16 = @ptrCast(@alignCast(cell.semantic_payload[690..818].ptr));
    outlier_w_raw[0] = 7.0;
    outlier_offsets_raw[0] = 10;

    var raw_x: [INTERMEDIATE_DIM]f32 = @splat(0.0);
    raw_x[10] = 3.0;
    var act: Activation11008 = undefined;
    act.raw = &raw_x;

    var y: [HIDDEN_DIM]f32 = @splat(0.0);
    gemvTileDownW2(&cell, &act, &y, 0);

    try std.testing.expectApproxEqAbs(@as(f32, 21.0), y[0], 1e-6);
}



