//! Auto-indexed static record table for Qwen2.5-3B-Instruct.w2.chpe.
//! Maps all 36 transformer layers and global tensors to exact record indices.
//! Sequential canonical packing: embed_tokens -> 36 layers (in numerical order 0..35) -> final_norm.

const std = @import("std");

pub const LayerRecordMap = struct {
    input_norm: usize,
    q_proj: usize,
    q_bias: usize,
    k_proj: usize,
    k_bias: usize,
    v_proj: usize,
    v_bias: usize,
    o_proj: usize,
    post_norm: usize,
    gate_proj: usize,
    up_proj: usize,
    down_proj: usize,
};

pub const EMBED_TOKENS_START: usize = 0;
pub const EMBED_TOKENS_RECORDS: usize = 9496;
pub const FINAL_NORM_RECORD: usize = 94348;
pub const TOTAL_RECORDS: usize = 94349;

pub fn makeLayer(l: usize) LayerRecordMap {
    const base = EMBED_TOKENS_RECORDS + l * 2357;
    return LayerRecordMap{
        .input_norm = base,
        .down_proj = base + 1,
        .gate_proj = base + 689,
        .up_proj = base + 1377,
        .post_norm = base + 2065,
        .k_bias = base + 2066,
        .k_proj = base + 2067,
        .o_proj = base + 2083,
        .q_bias = base + 2211,
        .q_proj = base + 2212,
        .v_bias = base + 2340,
        .v_proj = base + 2341,
    };
}

pub const LAYERS: [36]LayerRecordMap = blk: {
    var arr: [36]LayerRecordMap = undefined;
    for (0..36) |l| {
        arr[l] = makeLayer(l);
    }
    break :blk arr;
};

test "layers match sequential packing" {
    try std.testing.expectEqual(@as(usize, 9496), LAYERS[0].input_norm);
    try std.testing.expectEqual(@as(usize, 11853), LAYERS[1].input_norm);
    try std.testing.expectEqual(@as(usize, 14210), LAYERS[2].input_norm);
    try std.testing.expectEqual(@as(usize, 16567), LAYERS[3].input_norm);
    try std.testing.expectEqual(@as(usize, 94347), LAYERS[35].v_proj + 15);
    try std.testing.expectEqual(FINAL_NORM_RECORD, LAYERS[35].v_proj + 16);
}
