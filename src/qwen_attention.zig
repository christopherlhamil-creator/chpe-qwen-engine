//! Qwen2.5-3B Grouped-Query Attention (GQA)
//!
//! Subsystem: tot_hybrid/src/qwen_attention.zig
//! Blueprint: docs/BLUEPRINT-20260914-QWEN3B-CHPE-PIPELINE.md
//!
//! Architecture & Geometry Specifications:
//!   - Architecture: Qwen2.5-3B-Instruct
//!   - Query Heads (Q): 16 heads
//!   - Key/Value Heads (KV): 2 heads
//!   - Head Dimension: 128 floats
//!   - Grouped-Query Attention (GQA) ratio: 16 / 2 = 8 Query heads per KV head
//!   - Hidden Dimension: 16 * 128 = 2048 floats
//!   - KV Dimension: 2 * 128 = 256 floats
//!   - Scale factor: 1.0 / sqrt(128.0)
//!   - Zero paid cloud, pure Zig portable SIMD accumulation.

const std = @import("std");
const qwen_geom = @import("qwen3b_geometry.zig");

pub const Q_HEADS: usize = qwen_geom.Q_HEADS; // 16
pub const KV_HEADS: usize = qwen_geom.KV_HEADS; // 2
pub const HEAD_DIM: usize = qwen_geom.HEAD_DIM; // 128
pub const GQA_GROUP_SIZE: usize = Q_HEADS / KV_HEADS; // 8
pub const HIDDEN_DIM: usize = qwen_geom.HIDDEN_DIM; // 2048
pub const KV_DIM: usize = qwen_geom.KV_DIM; // 256

pub const ATTENTION_SCALE: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(HEAD_DIM)));
pub const MAX_CAUSAL_SEQ: usize = 4096;

comptime {
    std.debug.assert(Q_HEADS == 16);
    std.debug.assert(KV_HEADS == 2);
    std.debug.assert(HEAD_DIM == 128);
    std.debug.assert(GQA_GROUP_SIZE == 8);
    std.debug.assert(HIDDEN_DIM == 2048);
    std.debug.assert(KV_DIM == 256);
}

/// Static KV buffer structure for zero-allocation generation.
pub const KvCache = struct {
    k: [MAX_CAUSAL_SEQ][KV_DIM]f32,
    v: [MAX_CAUSAL_SEQ][KV_DIM]f32,
    seq_len: usize,

    pub fn init() KvCache {
        return .{
            .k = undefined,
            .v = undefined,
            .seq_len = 0,
        };
    }

    pub fn append(self: *KvCache, k_step: *const [KV_DIM]f32, v_step: *const [KV_DIM]f32) void {
        std.debug.assert(self.seq_len < MAX_CAUSAL_SEQ);
        @memcpy(&self.k[self.seq_len], k_step);
        @memcpy(&self.v[self.seq_len], v_step);
        self.seq_len += 1;
    }
};

/// Computes Causal Grouped-Query Attention for a single generation step.
/// q: Query vector for current token [2048]
/// kv_cache: Cached K and V vectors for all tokens up to and including current step
/// out: Output vector [2048]
pub fn forwardGqa(
    q: *const [HIDDEN_DIM]f32,
    kv_cache: *const KvCache,
    out: *[HIDDEN_DIM]f32,
) void {
    const seq_len = kv_cache.seq_len;
    std.debug.assert(seq_len > 0 and seq_len <= MAX_CAUSAL_SEQ);

    // Fast-path for single-token causal decode (softmax of single score is identically 1.0)
    if (seq_len == 1) {
        for (0..Q_HEADS) |h| {
            const kv_h = h / GQA_GROUP_SIZE;
            const q_offset = h * HEAD_DIM;
            const kv_offset = kv_h * HEAD_DIM;
            @memcpy(out[q_offset .. q_offset + HEAD_DIM], kv_cache.v[0][kv_offset .. kv_offset + HEAD_DIM]);
        }
        return;
    }

    var scores: [MAX_CAUSAL_SEQ]f32 = undefined;

    // Process each Query head
    for (0..Q_HEADS) |h| {
        const q_offset = h * HEAD_DIM;
        const q_head = q[q_offset .. q_offset + HEAD_DIM];

        // Map Q head to KV head: 0..7 -> KV 0, 8..15 -> KV 1
        const kv_h = h / GQA_GROUP_SIZE;
        const kv_offset = kv_h * HEAD_DIM;

        // Compute dot product Q * K^T for all causal tokens 0..seq_len-1 via SIMD
        var max_score: f32 = -std.math.inf(f32);
        for (0..seq_len) |s| {
            const k_head = kv_cache.k[s][kv_offset .. kv_offset + HEAD_DIM];
            var d0: @Vector(8, f32) = @splat(0.0);
            var d1: @Vector(8, f32) = @splat(0.0);
            inline for (0..8) |i| {
                const qv0: @Vector(8, f32) = q_head[i * 16 ..][0..8].*;
                const kv0: @Vector(8, f32) = k_head[i * 16 ..][0..8].*;
                const qv1: @Vector(8, f32) = q_head[i * 16 + 8 ..][0..8].*;
                const kv1: @Vector(8, f32) = k_head[i * 16 + 8 ..][0..8].*;
                d0 += qv0 * kv0;
                d1 += qv1 * kv1;
            }
            const dot = @reduce(.Add, d0 + d1);
            const scaled_dot = dot * ATTENTION_SCALE;
            scores[s] = scaled_dot;
            if (scaled_dot > max_score) {
                max_score = scaled_dot;
            }
        }

        // Softmax with numerical stability
        var sum_exp: f32 = 0.0;
        for (0..seq_len) |s| {
            const exp_val = @exp(scores[s] - max_score);
            scores[s] = exp_val;
            sum_exp += exp_val;
        }
        const inv_sum = 1.0 / sum_exp;
        for (0..seq_len) |s| {
            scores[s] *= inv_sum;
        }

        // Weighted accumulation of V vectors via SIMD
        const out_head = out[q_offset .. q_offset + HEAD_DIM];
        @memset(out_head, 0.0);
        for (0..seq_len) |s| {
            const weight: @Vector(8, f32) = @splat(scores[s]);
            const v_head = kv_cache.v[s][kv_offset .. kv_offset + HEAD_DIM];
            inline for (0..16) |i| {
                const vv: @Vector(8, f32) = v_head[i * 8 ..][0..8].*;
                const ov: @Vector(8, f32) = out_head[i * 8 ..][0..8].*;
                out_head[i * 8 ..][0..8].* = ov + weight * vv;
            }
        }
    }
}

test "qwen_attention seq_len 1 exact value pass-through" {
    var kv_cache = KvCache.init();
    var k0: [KV_DIM]f32 = @splat(0.5);
    var v0: [KV_DIM]f32 = undefined;
    for (0..KV_DIM) |i| {
        v0[i] = @as(f32, @floatFromInt(i + 1)) * 0.1;
    }
    kv_cache.append(&k0, &v0);

    var q: [HIDDEN_DIM]f32 = @splat(1.0);
    var out: [HIDDEN_DIM]f32 = undefined;

    forwardGqa(&q, &kv_cache, &out);

    // With seq_len == 1, softmax weight is identically 1.0
    // Head h in [0..7] maps to kv_h = 0 -> v0[0..128]
    // Head h in [8..15] maps to kv_h = 1 -> v0[128..256]
    for (0..Q_HEADS) |h| {
        const kv_h = h / GQA_GROUP_SIZE;
        const q_offset = h * HEAD_DIM;
        const kv_offset = kv_h * HEAD_DIM;
        for (0..HEAD_DIM) |d| {
            try std.testing.expectEqual(v0[kv_offset + d], out[q_offset + d]);
        }
    }
}

test "qwen_attention seq_len 2 vectorized SIMD path" {
    var kv_cache = KvCache.init();
    var k0: [KV_DIM]f32 = @splat(0.2);
    var v0: [KV_DIM]f32 = @splat(1.0);
    var k1: [KV_DIM]f32 = @splat(0.4);
    var v1: [KV_DIM]f32 = @splat(3.0);
    kv_cache.append(&k0, &v0);
    kv_cache.append(&k1, &v1);

    var q: [HIDDEN_DIM]f32 = @splat(0.5);
    var out: [HIDDEN_DIM]f32 = undefined;

    forwardGqa(&q, &kv_cache, &out);

    // Verify all outputs are finite and within bounds [1.0, 3.0]
    for (0..HIDDEN_DIM) |i| {
        try std.testing.expect(out[i] >= 1.0 and out[i] <= 3.0);
    }
}

