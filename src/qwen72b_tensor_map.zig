//! Auto-indexed static record table for Qwen2.5-72B-Instruct.chpe.
//! Maps all 80 transformer layers and global tensors to exact record indices.
//! Sequential canonical packing: embed_tokens -> 80 layers (in numerical order 0..79) -> final_norm -> lm_head.
//!
//! Architecture:
//!   - Layers: 80
//!   - Hidden Dimension: 8192
//!   - Intermediate Dimension: 29568
//!   - Query Heads: 64 (Head Dim = 128)
//!   - Key/Value Heads: 8 (KV Dim = 1024)
//!   - Vocabulary: 152064
//!   - Weights per tile: 32768

const std = @import("std");

pub const HIDDEN_DIM: usize = 8192;
pub const INTERMEDIATE_DIM: usize = 29568;
pub const NUM_LAYERS: usize = 80;
pub const NUM_ATTN_HEADS: usize = 64;
pub const NUM_KV_HEADS: usize = 8;
pub const HEAD_DIM: usize = 128;
pub const VOCAB_SIZE: usize = 152064;
pub const WEIGHTS_PER_TILE: usize = 32768;

// Tile counts per tensor
pub const EMBED_TOKENS_RECORDS: usize = 38016; // 152064 * 8192 / 32768
pub const Q_PROJ_TILES: usize = 2048;          // 8192 * 8192 / 32768
pub const K_PROJ_TILES: usize = 256;           // 1024 * 8192 / 32768
pub const V_PROJ_TILES: usize = 256;           // 1024 * 8192 / 32768
pub const O_PROJ_TILES: usize = 2048;          // 8192 * 8192 / 32768
pub const GATE_PROJ_TILES: usize = 7392;       // 29568 * 8192 / 32768
pub const UP_PROJ_TILES: usize = 7392;         // 29568 * 8192 / 32768
pub const DOWN_PROJ_TILES: usize = 7392;       // 8192 * 29568 / 32768

pub const RECORDS_PER_LAYER: usize = 1 + // input_norm
    Q_PROJ_TILES + 1 +                   // q_proj + q_bias
    K_PROJ_TILES + 1 +                   // k_proj + k_bias
    V_PROJ_TILES + 1 +                   // v_proj + v_bias
    O_PROJ_TILES + 1 +                   // o_proj + post_norm
    GATE_PROJ_TILES + UP_PROJ_TILES + DOWN_PROJ_TILES; // 26789 records

pub const EMBED_TOKENS_START: usize = 0;
pub const FINAL_NORM_RECORD: usize = EMBED_TOKENS_RECORDS + NUM_LAYERS * RECORDS_PER_LAYER; // 2,181,136
pub const LM_HEAD_RECORD_START: usize = FINAL_NORM_RECORD + 1; // 2,181,137
pub const LM_HEAD_TILES: usize = 38016;
pub const TOTAL_RECORDS: usize = LM_HEAD_RECORD_START + LM_HEAD_TILES; // 2,219,153

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

pub fn makeLayer(l: usize) LayerRecordMap {
    const base = EMBED_TOKENS_RECORDS + l * RECORDS_PER_LAYER;
    return LayerRecordMap{
        .input_norm = base,
        .q_proj = base + 1,
        .q_bias = base + 1 + Q_PROJ_TILES,
        .k_proj = base + 2 + Q_PROJ_TILES,
        .k_bias = base + 2 + Q_PROJ_TILES + K_PROJ_TILES,
        .v_proj = base + 3 + Q_PROJ_TILES + K_PROJ_TILES,
        .v_bias = base + 3 + Q_PROJ_TILES + K_PROJ_TILES + V_PROJ_TILES,
        .o_proj = base + 4 + Q_PROJ_TILES + K_PROJ_TILES + V_PROJ_TILES,
        .post_norm = base + 4 + Q_PROJ_TILES + K_PROJ_TILES + V_PROJ_TILES + O_PROJ_TILES,
        .gate_proj = base + 5 + Q_PROJ_TILES + K_PROJ_TILES + V_PROJ_TILES + O_PROJ_TILES,
        .up_proj = base + 5 + Q_PROJ_TILES + K_PROJ_TILES + V_PROJ_TILES + O_PROJ_TILES + GATE_PROJ_TILES,
        .down_proj = base + 5 + Q_PROJ_TILES + K_PROJ_TILES + V_PROJ_TILES + O_PROJ_TILES + GATE_PROJ_TILES + UP_PROJ_TILES,
    };
}

pub const LAYERS: [NUM_LAYERS]LayerRecordMap = blk: {
    var arr: [NUM_LAYERS]LayerRecordMap = undefined;
    for (0..NUM_LAYERS) |l| {
        arr[l] = makeLayer(l);
    }
    break :blk arr;
};

test "qwen72b tensor map geometry invariants" {
    try std.testing.expectEqual(@as(usize, 26789), RECORDS_PER_LAYER);
    try std.testing.expectEqual(@as(usize, 38016), EMBED_TOKENS_RECORDS);
    try std.testing.expectEqual(@as(usize, 38016), LAYERS[0].input_norm);
    try std.testing.expectEqual(@as(usize, 38017), LAYERS[0].q_proj);
    try std.testing.expectEqual(@as(usize, 40065), LAYERS[0].q_bias);
    try std.testing.expectEqual(@as(usize, 40066), LAYERS[0].k_proj);
    try std.testing.expectEqual(@as(usize, 40322), LAYERS[0].k_bias);
    try std.testing.expectEqual(@as(usize, 40323), LAYERS[0].v_proj);
    try std.testing.expectEqual(@as(usize, 40579), LAYERS[0].v_bias);
    try std.testing.expectEqual(@as(usize, 40580), LAYERS[0].o_proj);
    try std.testing.expectEqual(@as(usize, 42628), LAYERS[0].post_norm);
    try std.testing.expectEqual(@as(usize, 42629), LAYERS[0].gate_proj);
    try std.testing.expectEqual(@as(usize, 50021), LAYERS[0].up_proj);
    try std.testing.expectEqual(@as(usize, 57413), LAYERS[0].down_proj);
    // End of layer 0 is down_proj + 7392 = 64805 = LAYERS[1].input_norm
    try std.testing.expectEqual(@as(usize, 64805), LAYERS[1].input_norm);

    // Last layer (layer 79)
    try std.testing.expectEqual(@as(usize, 2181136), FINAL_NORM_RECORD);
    try std.testing.expectEqual(FINAL_NORM_RECORD, LAYERS[79].down_proj + DOWN_PROJ_TILES);
    try std.testing.expectEqual(@as(usize, 2181137), LM_HEAD_RECORD_START);
    try std.testing.expectEqual(@as(usize, 2219153), TOTAL_RECORDS);
}
