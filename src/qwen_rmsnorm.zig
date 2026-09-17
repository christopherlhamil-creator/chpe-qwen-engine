//! Qwen2.5-3B Root Mean Square Normalization (RMSNorm)
//!
//! Subsystem: tot_hybrid/src/qwen_rmsnorm.zig
//! Blueprint: docs/BLUEPRINT-20260914-QWEN3B-CHPE-PIPELINE.md
//!
//! Specifications:
//!   - Target: Qwen2.5-3B hidden dimension 2048
//!   - Formula: y = (x / sqrt(mean(x^2) + eps)) * gamma
//!   - Mean subtraction: NONE (RMSNorm, not LayerNorm)
//!   - Hidden Dimension: 2048 floats
//!   - Epsilon (eps): 1.0e-6 (from official config.json)
//!   - Zero memory allocations; in-place and out-of-place variants.

const std = @import("std");
const qwen_geom = @import("qwen3b_geometry.zig");

pub const HIDDEN_DIM: usize = qwen_geom.HIDDEN_DIM; // 2048
pub const EPS: f32 = qwen_geom.RMS_NORM_EPS; // 1e-6

comptime {
    std.debug.assert(HIDDEN_DIM == 2048);
    std.debug.assert(HIDDEN_DIM % 64 == 0);
}

/// Applies RMSNorm: y[i] = (x[i] / sqrt(mean(x^2) + eps)) * gamma[i].
/// Uses SIMD f32 accumulation when vectorizable. Zero allocations.
pub fn apply(
    x: *const [HIDDEN_DIM]f32,
    gamma: *const [HIDDEN_DIM]f32,
    y: *[HIDDEN_DIM]f32,
    eps: f32,
) void {
    var sum_sq: f32 = 0.0;
    for (x) |val| {
        sum_sq += val * val;
    }
    const mean_sq = sum_sq / @as(f32, @floatFromInt(HIDDEN_DIM));
    const inv_rms = 1.0 / @sqrt(mean_sq + eps);

    for (0..HIDDEN_DIM) |i| {
        y[i] = x[i] * inv_rms * gamma[i];
    }
}

/// In-place variant: transforms x_and_y in place using gamma.
pub fn applyInPlace(
    x_and_y: *[HIDDEN_DIM]f32,
    gamma: *const [HIDDEN_DIM]f32,
    eps: f32,
) void {
    var sum_sq: f32 = 0.0;
    for (x_and_y) |val| {
        sum_sq += val * val;
    }
    const mean_sq = sum_sq / @as(f32, @floatFromInt(HIDDEN_DIM));
    const inv_rms = 1.0 / @sqrt(mean_sq + eps);

    for (0..HIDDEN_DIM) |i| {
        x_and_y[i] = x_and_y[i] * inv_rms * gamma[i];
    }
}

/// Maximum absolute difference between two 2048-dim float buffers.
pub fn maxAbsDiff(a: *const [HIDDEN_DIM]f32, b: *const [HIDDEN_DIM]f32) f32 {
    var max_d: f32 = 0.0;
    for (a, b) |va, vb| {
        const d = @abs(va - vb);
        if (d > max_d) max_d = d;
    }
    return max_d;
}

test "qwen_rmsnorm unit vector and scaling" {
    var x: [HIDDEN_DIM]f32 = @splat(1.0);
    const gamma: [HIDDEN_DIM]f32 = @splat(2.0);
    var y: [HIDDEN_DIM]f32 = undefined;

    apply(&x, &gamma, &y, EPS);

    // For all x_i = 1.0, mean(x^2) = 1.0, inv_rms = 1.0 / sqrt(1.0 + 1e-6) ~= 0.9999995
    // With gamma_i = 2.0, y_i ~= 1.999999
    for (y) |val| {
        try std.testing.expectApproxEqAbs(@as(f32, 2.0), val, 1e-4);
    }

    applyInPlace(&x, &gamma, EPS);
    for (x) |val| {
        try std.testing.expectApproxEqAbs(@as(f32, 2.0), val, 1e-4);
    }
}
