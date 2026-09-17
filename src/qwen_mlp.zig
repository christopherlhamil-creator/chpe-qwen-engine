//! Qwen2.5-3B SwiGLU MLP & Elementwise Utilities
//!
//! Subsystem: tot_hybrid/src/qwen_mlp.zig
//! Blueprint: docs/BLUEPRINT-20260914-QWEN3B-CHPE-PIPELINE.md
//!
//! Architecture & Geometry Specifications:
//!   - Hidden Dimension: 2048 floats
//!   - Intermediate Dimension: 11008 floats
//!   - Activation Function: SwiGLU = silu(gate(x)) * up(x)
//!   - silu(x) = x / (1.0 + exp(-x))
//!   - Output = down(swiglu) + residual
//!   - Zero paid cloud, pure Zig portable SIMD.

const std = @import("std");
const qwen_geom = @import("qwen3b_geometry.zig");

pub const HIDDEN_DIM: usize = qwen_geom.HIDDEN_DIM; // 2048
pub const INTERMEDIATE_DIM: usize = qwen_geom.INTERMEDIATE_DIM; // 11008

comptime {
    std.debug.assert(HIDDEN_DIM == 2048);
    std.debug.assert(INTERMEDIATE_DIM == 11008);
}

/// Sigmoid-weighted Linear Unit: silu(x) = x / (1.0 + exp(-x)).
pub inline fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

pub inline fn siluVec4(x: @Vector(4, f32)) @Vector(4, f32) {
    const z = -x;
    const z_clamped = @max(@min(z, @as(@Vector(4, f32), @splat(88.0))), @as(@Vector(4, f32), @splat(-88.0)));
    const log2e: @Vector(4, f32) = @splat(1.4426950408889634);
    const k_f = @round(z_clamped * log2e);
    const ln2: @Vector(4, f32) = @splat(0.6931471805599453);
    const f = z_clamped - k_f * ln2;

    const c5: @Vector(4, f32) = @splat(0.0083333333);
    const c4: @Vector(4, f32) = @splat(0.0416666667);
    const c3: @Vector(4, f32) = @splat(0.1666666667);
    const c2: @Vector(4, f32) = @splat(0.5);
    const c1: @Vector(4, f32) = @splat(1.0);
    const c0: @Vector(4, f32) = @splat(1.0);

    var p = c5 * f + c4;
    p = p * f + c3;
    p = p * f + c2;
    p = p * f + c1;
    p = p * f + c0;

    const k_i = @as(@Vector(4, i32), @intFromFloat(k_f));
    const bias: @Vector(4, i32) = @splat(127);
    const exp_bits = (k_i + bias) << @splat(23);
    const scale = @as(@Vector(4, f32), @bitCast(exp_bits));

    const exp_z = p * scale;
    const ones: @Vector(4, f32) = @splat(1.0);
    return x / (ones + exp_z);
}

/// Elementwise SwiGLU: out[i] = silu(gate[i]) * up[i].
pub fn swigluForward(gate: []const f32, up: []const f32, out: []f32) void {
    std.debug.assert(gate.len == up.len);
    std.debug.assert(gate.len == out.len);

    var i: usize = 0;
    while (i + 4 <= gate.len) : (i += 4) {
        const g: @Vector(4, f32) = gate[i..][0..4].*;
        const u: @Vector(4, f32) = up[i..][0..4].*;
        out[i..][0..4].* = siluVec4(g) * u;
    }
    while (i < gate.len) : (i += 1) {
        out[i] = silu(gate[i]) * up[i];
    }
}

/// Elementwise in-place residual addition: target[i] += residual[i].
pub fn addResidual(target: []f32, residual: []const f32) void {
    std.debug.assert(target.len == residual.len);

    var i: usize = 0;
    while (i + 4 <= target.len) : (i += 4) {
        const t: @Vector(4, f32) = target[i..][0..4].*;
        const r: @Vector(4, f32) = residual[i..][0..4].*;
        target[i..][0..4].* = t + r;
    }
    while (i < target.len) : (i += 1) {
        target[i] += residual[i];
    }
}

test "qwen_mlp silu at zero and positive" {
    // silu(0) = 0 / (1 + 1) = 0
    try std.testing.expectEqual(@as(f32, 0.0), silu(0.0));

    // silu(1) = 1 / (1 + exp(-1)) ~= 0.7310586
    try std.testing.expectApproxEqAbs(@as(f32, 0.7310586), silu(1.0), 1e-5);
}

test "qwen_mlp swiglu forward" {
    const gate = [_]f32{ 0.0, 1.0, -1.0, 2.0 };
    const up = [_]f32{ 5.0, 2.0, 3.0, 0.5 };
    var out: [4]f32 = undefined;

    swigluForward(&gate, &up, &out);

    // out[0] = silu(0) * 5.0 = 0.0
    try std.testing.expectEqual(@as(f32, 0.0), out[0]);
    // out[1] = silu(1.0) * 2.0 ~= 0.7310586 * 2.0 = 1.462117
    try std.testing.expectApproxEqAbs(@as(f32, 1.462117), out[1], 1e-4);
}

test "qwen_mlp residual addition" {
    var target = [_]f32{ 1.0, 2.0, 3.0 };
    const res = [_]f32{ 0.5, -1.0, 4.0 };

    addResidual(&target, &res);

    try std.testing.expectEqual(@as(f32, 1.5), target[0]);
    try std.testing.expectEqual(@as(f32, 1.0), target[1]);
    try std.testing.expectEqual(@as(f32, 7.0), target[2]);
}
