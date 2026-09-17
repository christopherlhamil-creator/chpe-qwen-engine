//! Qwen2.5-3B Rotary Position Embedding (RoPE)
//!
//! Subsystem: tot_hybrid/src/qwen_rope.zig
//! Blueprint: docs/BLUEPRINT-20260914-QWEN3B-CHPE-PIPELINE.md
//!
//! Architecture & Geometry Specifications:
//!   - Architecture: Qwen2.5-3B-Instruct
//!   - Base Frequency (rope_theta): 1,000,000.0 (from official config.json)
//!   - Head Dimension: 128 floats
//!   - Number of pairs per head: 64 pairs (x_i, x_{i+64}) [NeoX / Qwen2 rotate_half]
//!   - Frequency formula: theta_i = 1,000,000.0 ^ (-2i / 128)
//!   - Rotation angle at position p: phi_i = p * theta_i
//!   - 2D rotation matrix:
//!       x'_{i}    = x_i * cos(phi_i) - x_{i+64} * sin(phi_i)
//!       x'_{i+64} = x_{i+64} * cos(phi_i) + x_i * sin(phi_i)
//!   - Applied exclusively to Query (Q) and Key (K) vectors.
//!   - Value (V) vectors are NEVER rotated.
//!   - Position 0 is exact identity (cos=1, sin=0) on every pair.
//!   - Zero memory allocations; in-place single-head and multi-head transforms.

const std = @import("std");
const qwen_geom = @import("qwen3b_geometry.zig");

pub const HEAD_DIM: usize = qwen_geom.HEAD_DIM; // 128
pub const PAIRS: usize = HEAD_DIM / 2; // 64
pub const ROPE_THETA: f32 = qwen_geom.ROPE_THETA; // 1,000,000.0

pub const INVERSE_FREQUENCIES: [PAIRS]f32 = initFrequencies();

fn initFrequencies() [PAIRS]f32 {
    @setEvalBranchQuota(100_000);
    var freqs: [PAIRS]f32 = undefined;
    for (0..PAIRS) |i| {
        const exponent = -2.0 * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(HEAD_DIM));
        freqs[i] = std.math.pow(f32, ROPE_THETA, exponent);
    }
    return freqs;
}

comptime {
    std.debug.assert(HEAD_DIM == 128);
    std.debug.assert(PAIRS == 64);
    std.debug.assert(ROPE_THETA == 1000000.0);
}

/// Applies RoPE in-place to a single 128-dim head vector (Q or K).
/// When position == 0, returns immediately without modifying the buffer.
/// Implements NeoX / HuggingFace rotate_half:
///   q_embed = (q * cos) + (rotate_half(q) * sin)
pub fn applyRope(q_or_k: *[HEAD_DIM]f32, position: u64) void {
    if (position == 0) return;

    const pos_f: f32 = @floatFromInt(position);
    var i: usize = 0;
    while (i < PAIRS) : (i += 1) {
        const theta = INVERSE_FREQUENCIES[i];
        const angle = pos_f * theta;
        const cos_val = @cos(angle);
        const sin_val = @sin(angle);

        const x0 = q_or_k[i];
        const x1 = q_or_k[i + PAIRS];

        q_or_k[i] = x0 * cos_val - x1 * sin_val;
        q_or_k[i + PAIRS] = x1 * cos_val + x0 * sin_val;
    }
}

/// Applies RoPE in-place across multiple heads (e.g. 16 Q heads or 2 KV heads).
pub fn applyRopeMultiHead(q_or_k: []f32, num_heads: usize, position: u64) void {
    std.debug.assert(q_or_k.len == num_heads * HEAD_DIM);
    if (position == 0) return;

    for (0..num_heads) |h| {
        const head_slice = q_or_k[h * HEAD_DIM .. (h + 1) * HEAD_DIM];
        applyRope(@ptrCast(head_slice.ptr), position);
    }
}

/// Maximum absolute difference between two 128-dim float buffers.
pub fn maxAbsDiff128(a: *const [HEAD_DIM]f32, b: *const [HEAD_DIM]f32) f32 {
    var max_d: f32 = 0.0;
    for (a, b) |v_a, v_b| {
        const diff = @abs(v_a - v_b);
        if (diff > max_d) max_d = diff;
    }
    return max_d;
}

test "qwen_rope position 0 identity" {
    var vec: [HEAD_DIM]f32 = undefined;
    for (0..HEAD_DIM) |i| {
        vec[i] = @as(f32, @floatFromInt(i + 1)) * 0.1;
    }
    const original = vec;

    applyRope(&vec, 0);

    for (0..HEAD_DIM) |i| {
        try std.testing.expectEqual(original[i], vec[i]);
    }
}

test "qwen_rope position 1 rotation norm invariant" {
    var vec: [HEAD_DIM]f32 = undefined;
    for (0..HEAD_DIM) |i| {
        vec[i] = @as(f32, @floatFromInt(i + 1)) * 0.1;
    }

    var original_norm_sq: f32 = 0.0;
    for (vec) |v| {
        original_norm_sq += v * v;
    }

    applyRope(&vec, 1);

    var rotated_norm_sq: f32 = 0.0;
    for (vec) |v| {
        rotated_norm_sq += v * v;
    }

    // 2D orthogonal rotation preserves L2 norm: ||R x||^2 == ||x||^2
    try std.testing.expectApproxEqRel(original_norm_sq, rotated_norm_sq, 1e-4);
}

test "qwen_rope multi-head batching consistency" {
    const num_heads = 4;
    var multi_vec: [num_heads * HEAD_DIM]f32 = undefined;
    for (0..multi_vec.len) |i| {
        multi_vec[i] = @as(f32, @floatFromInt(i + 1)) * 0.01;
    }

    var single_heads: [num_heads][HEAD_DIM]f32 = undefined;
    for (0..num_heads) |h| {
        @memcpy(&single_heads[h], multi_vec[h * HEAD_DIM .. (h + 1) * HEAD_DIM]);
        applyRope(&single_heads[h], 2);
    }

    applyRopeMultiHead(&multi_vec, num_heads, 2);

    for (0..num_heads) |h| {
        const slice = multi_vec[h * HEAD_DIM .. (h + 1) * HEAD_DIM];
        for (0..HEAD_DIM) |d| {
            try std.testing.expectEqual(single_heads[h][d], slice[d]);
        }
    }
}

test "qwen_rope rotate_half position 1 numerical check against reference" {
    var vec: [HEAD_DIM]f32 = undefined;
    for (0..HEAD_DIM) |i| {
        vec[i] = @as(f32, @floatFromInt(i + 1)) * 0.1;
    }

    applyRope(&vec, 1);

    // Matches Python/HuggingFace rotate_half for Qwen2.5 with theta=1,000,000.0:
    try std.testing.expectApproxEqAbs(@as(f32, -5.415531), vec[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -4.622832), vec[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 3.596112), vec[64], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 4.714808), vec[65], 1e-4);
}
