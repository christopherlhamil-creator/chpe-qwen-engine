//! Geometry constants and tile indexing for Qwen2.5-3B-Instruct.
//! Matches ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/ config.json.
//! All dimensions strictly respect 64-byte CPU cache line boundaries.

const std = @import("std");

pub const HIDDEN_DIM: usize = 2048;
pub const INTERMEDIATE_DIM: usize = 11008;
pub const NUM_LAYERS: usize = 36;
pub const Q_HEADS: usize = 16;
pub const KV_HEADS: usize = 2;
pub const HEAD_DIM: usize = 128;
pub const VOCAB_SIZE: usize = 151936;
pub const TIE_WORD_EMBEDDINGS: bool = true;
pub const RMS_NORM_EPS: f32 = 1e-6;
pub const ROPE_THETA: f32 = 1000000.0;

// Derived dimensions
pub const KV_DIM: usize = KV_HEADS * HEAD_DIM; // 256
pub const Q_DIM: usize = Q_HEADS * HEAD_DIM; // 2048

// Weights per layer
// Q: 2048 x 2048 = 4,194,304
// K: 256 x 2048  = 524,288
// V: 256 x 2048  = 524,288
// O: 2048 x 2048 = 4,194,304
// Gate: 11008 x 2048 = 22,544,384
// Up:   11008 x 2048 = 22,544,384
// Down: 2048 x 11008 = 22,544,384
// RMSNorm in: 2048
// RMSNorm post: 2048

comptime {
    std.debug.assert(HIDDEN_DIM == Q_HEADS * HEAD_DIM);
    std.debug.assert(KV_DIM == KV_HEADS * HEAD_DIM);
    std.debug.assert(HIDDEN_DIM % 64 == 0);
    std.debug.assert(INTERMEDIATE_DIM % 64 == 0);
    std.debug.assert(HEAD_DIM % 64 == 0);
    std.debug.assert(KV_DIM % 64 == 0);
}

pub const ArchivePrecision = enum {
    w2_4bit,
    bf16,
    hybrid,
};

pub const Bf16LayerOffsets = struct {
    pub const EMBED_TOKENS_START: usize = 0;
    pub const EMBED_TOKENS_RECORDS: usize = 37984; // 151,936 / 4 rows
    pub const RECORDS_PER_LAYER: usize = 9413;
    pub const FINAL_NORM_RECORD: usize = 376852;
    pub const TOTAL_RECORDS: usize = 376853;

    pub const Q_PROJ_TILES: usize = 512;
    pub const K_PROJ_TILES: usize = 64;
    pub const V_PROJ_TILES: usize = 64;
    pub const O_PROJ_TILES: usize = 512;
    pub const GATE_PROJ_TILES: usize = 2752;
    pub const UP_PROJ_TILES: usize = 2752;
    pub const DOWN_PROJ_TILES: usize = 2752;

    pub fn layerBase(l: usize) usize {
        return EMBED_TOKENS_RECORDS + l * RECORDS_PER_LAYER;
    }

    pub fn inNorm(l: usize) usize { return layerBase(l) + 0; }
    pub fn qProj(l: usize) usize { return layerBase(l) + 1; }
    pub fn qBias(l: usize) usize { return layerBase(l) + 513; }
    pub fn kProj(l: usize) usize { return layerBase(l) + 514; }
    pub fn kBias(l: usize) usize { return layerBase(l) + 578; }
    pub fn vProj(l: usize) usize { return layerBase(l) + 579; }
    pub fn vBias(l: usize) usize { return layerBase(l) + 643; }
    pub fn oProj(l: usize) usize { return layerBase(l) + 644; }
    pub fn postNorm(l: usize) usize { return layerBase(l) + 1156; }
    pub fn gateProj(l: usize) usize { return layerBase(l) + 1157; }
    pub fn upProj(l: usize) usize { return layerBase(l) + 3909; }
    pub fn downProj(l: usize) usize { return layerBase(l) + 6661; }
};

pub const HybridLayerOffsets = struct {
    pub const EMBED_TOKENS_START: usize = 0;
    pub const EMBED_TOKENS_RECORDS: usize = 9496; // 151,936 vocab / 16 rows per W4 tile
    pub const FINAL_NORM_RECORD: usize = 120108;
    pub const TOTAL_RECORDS: usize = 120109;

    pub const LayerOffsets = struct {
        in_norm: usize,
        q_proj: usize,
        q_bias: usize,
        k_proj: usize,
        k_bias: usize,
        v_proj: usize,
        v_bias: usize,
        o_proj: usize,
        o_tiles: usize,
        post_norm: usize,
        gate_proj: usize,
        up_proj: usize,
        down_proj: usize,
        mlp_tiles: usize,
        down_tiles: usize,
        is_mlp_w8: bool,
        is_down_w8: bool,
        is_o_w4: bool,
    };

    pub const LAYERS: [36]LayerOffsets = .{
        .{ .in_norm = 9496, .q_proj = 9497, .q_bias = 10009, .k_proj = 10010, .k_bias = 10074, .v_proj = 10075, .v_bias = 10139, .o_proj = 10140, .o_tiles = 256, .post_norm = 10396, .gate_proj = 10397, .up_proj = 11085, .down_proj = 11773, .mlp_tiles = 688, .down_tiles = 688, .is_mlp_w8 = false, .is_down_w8 = false, .is_o_w4 = false },
        .{ .in_norm = 12461, .q_proj = 12462, .q_bias = 12974, .k_proj = 12975, .k_bias = 13039, .v_proj = 13040, .v_bias = 13104, .o_proj = 13105, .o_tiles = 256, .post_norm = 13361, .gate_proj = 13362, .up_proj = 14738, .down_proj = 16114, .mlp_tiles = 1376, .down_tiles = 1376, .is_mlp_w8 = true, .is_down_w8 = true, .is_o_w4 = false },
        .{ .in_norm = 17490, .q_proj = 17491, .q_bias = 18003, .k_proj = 18004, .k_bias = 18068, .v_proj = 18069, .v_bias = 18133, .o_proj = 18134, .o_tiles = 256, .post_norm = 18390, .gate_proj = 18391, .up_proj = 19079, .down_proj = 19767, .mlp_tiles = 688, .down_tiles = 688, .is_mlp_w8 = false, .is_down_w8 = false, .is_o_w4 = false },
        .{ .in_norm = 20455, .q_proj = 20456, .q_bias = 20968, .k_proj = 20969, .k_bias = 21033, .v_proj = 21034, .v_bias = 21098, .o_proj = 21099, .o_tiles = 256, .post_norm = 21355, .gate_proj = 21356, .up_proj = 22044, .down_proj = 22732, .mlp_tiles = 688, .down_tiles = 688, .is_mlp_w8 = false, .is_down_w8 = false, .is_o_w4 = false },
        .{ .in_norm = 23420, .q_proj = 23421, .q_bias = 23933, .k_proj = 23934, .k_bias = 23998, .v_proj = 23999, .v_bias = 24063, .o_proj = 24064, .o_tiles = 256, .post_norm = 24320, .gate_proj = 24321, .up_proj = 25009, .down_proj = 25697, .mlp_tiles = 688, .down_tiles = 688, .is_mlp_w8 = false, .is_down_w8 = false, .is_o_w4 = false },
        .{ .in_norm = 26385, .q_proj = 26386, .q_bias = 26898, .k_proj = 26899, .k_bias = 26963, .v_proj = 26964, .v_bias = 27028, .o_proj = 27029, .o_tiles = 256, .post_norm = 27285, .gate_proj = 27286, .up_proj = 27974, .down_proj = 28662, .mlp_tiles = 688, .down_tiles = 688, .is_mlp_w8 = false, .is_down_w8 = false, .is_o_w4 = false },
        .{ .in_norm = 29350, .q_proj = 29351, .q_bias = 29863, .k_proj = 29864, .k_bias = 29928, .v_proj = 29929, .v_bias = 29993, .o_proj = 29994, .o_tiles = 256, .post_norm = 30250, .gate_proj = 30251, .up_proj = 30939, .down_proj = 31627, .mlp_tiles = 688, .down_tiles = 688, .is_mlp_w8 = false, .is_down_w8 = false, .is_o_w4 = false },
        .{ .in_norm = 32315, .q_proj = 32316, .q_bias = 32828, .k_proj = 32829, .k_bias = 32893, .v_proj = 32894, .v_bias = 32958, .o_proj = 32959, .o_tiles = 256, .post_norm = 33215, .gate_proj = 33216, .up_proj = 33904, .down_proj = 34592, .mlp_tiles = 688, .down_tiles = 688, .is_mlp_w8 = false, .is_down_w8 = false, .is_o_w4 = false },
        .{ .in_norm = 35280, .q_proj = 35281, .q_bias = 35793, .k_proj = 35794, .k_bias = 35858, .v_proj = 35859, .v_bias = 35923, .o_proj = 35924, .o_tiles = 256, .post_norm = 36180, .gate_proj = 36181, .up_proj = 36869, .down_proj = 37557, .mlp_tiles = 688, .down_tiles = 688, .is_mlp_w8 = false, .is_down_w8 = false, .is_o_w4 = false },
        .{ .in_norm = 38245, .q_proj = 38246, .q_bias = 38758, .k_proj = 38759, .k_bias = 38823, .v_proj = 38824, .v_bias = 38888, .o_proj = 38889, .o_tiles = 256, .post_norm = 39145, .gate_proj = 39146, .up_proj = 39834, .down_proj = 40522, .mlp_tiles = 688, .down_tiles = 688, .is_mlp_w8 = false, .is_down_w8 = false, .is_o_w4 = false },
        .{ .in_norm = 41210, .q_proj = 41211, .q_bias = 41723, .k_proj = 41724, .k_bias = 41788, .v_proj = 41789, .v_bias = 41853, .o_proj = 41854, .o_tiles = 256, .post_norm = 42110, .gate_proj = 42111, .up_proj = 42799, .down_proj = 43487, .mlp_tiles = 688, .down_tiles = 688, .is_mlp_w8 = false, .is_down_w8 = false, .is_o_w4 = false },
        .{ .in_norm = 44175, .q_proj = 44176, .q_bias = 44688, .k_proj = 44689, .k_bias = 44753, .v_proj = 44754, .v_bias = 44818, .o_proj = 44819, .o_tiles = 256, .post_norm = 45075, .gate_proj = 45076, .up_proj = 45764, .down_proj = 46452, .mlp_tiles = 688, .down_tiles = 688, .is_mlp_w8 = false, .is_down_w8 = false, .is_o_w4 = false },
        .{ .in_norm = 47140, .q_proj = 47141, .q_bias = 47653, .k_proj = 47654, .k_bias = 47718, .v_proj = 47719, .v_bias = 47783, .o_proj = 47784, .o_tiles = 256, .post_norm = 48040, .gate_proj = 48041, .up_proj = 48729, .down_proj = 49417, .mlp_tiles = 688, .down_tiles = 688, .is_mlp_w8 = false, .is_down_w8 = false, .is_o_w4 = false },
        .{ .in_norm = 50105, .q_proj = 50106, .q_bias = 50618, .k_proj = 50619, .k_bias = 50683, .v_proj = 50684, .v_bias = 50748, .o_proj = 50749, .o_tiles = 256, .post_norm = 51005, .gate_proj = 51006, .up_proj = 51694, .down_proj = 52382, .mlp_tiles = 688, .down_tiles = 688, .is_mlp_w8 = false, .is_down_w8 = false, .is_o_w4 = false },
        .{ .in_norm = 53070, .q_proj = 53071, .q_bias = 53583, .k_proj = 53584, .k_bias = 53648, .v_proj = 53649, .v_bias = 53713, .o_proj = 53714, .o_tiles = 128, .post_norm = 53842, .gate_proj = 53843, .up_proj = 54531, .down_proj = 55219, .mlp_tiles = 688, .down_tiles = 688, .is_mlp_w8 = false, .is_down_w8 = false, .is_o_w4 = true },
        .{ .in_norm = 55907, .q_proj = 55908, .q_bias = 56420, .k_proj = 56421, .k_bias = 56485, .v_proj = 56486, .v_bias = 56550, .o_proj = 56551, .o_tiles = 128, .post_norm = 56679, .gate_proj = 56680, .up_proj = 57368, .down_proj = 58056, .mlp_tiles = 688, .down_tiles = 688, .is_mlp_w8 = false, .is_down_w8 = false, .is_o_w4 = true },
        .{ .in_norm = 58744, .q_proj = 58745, .q_bias = 59257, .k_proj = 59258, .k_bias = 59322, .v_proj = 59323, .v_bias = 59387, .o_proj = 59388, .o_tiles = 256, .post_norm = 59644, .gate_proj = 59645, .up_proj = 60333, .down_proj = 61021, .mlp_tiles = 688, .down_tiles = 688, .is_mlp_w8 = false, .is_down_w8 = false, .is_o_w4 = false },
        .{ .in_norm = 61709, .q_proj = 61710, .q_bias = 62222, .k_proj = 62223, .k_bias = 62287, .v_proj = 62288, .v_bias = 62352, .o_proj = 62353, .o_tiles = 256, .post_norm = 62609, .gate_proj = 62610, .up_proj = 63298, .down_proj = 63986, .mlp_tiles = 688, .down_tiles = 688, .is_mlp_w8 = false, .is_down_w8 = false, .is_o_w4 = false },
        .{ .in_norm = 64674, .q_proj = 64675, .q_bias = 65187, .k_proj = 65188, .k_bias = 65252, .v_proj = 65253, .v_bias = 65317, .o_proj = 65318, .o_tiles = 256, .post_norm = 65574, .gate_proj = 65575, .up_proj = 66263, .down_proj = 66951, .mlp_tiles = 688, .down_tiles = 688, .is_mlp_w8 = false, .is_down_w8 = false, .is_o_w4 = false },
        .{ .in_norm = 67639, .q_proj = 67640, .q_bias = 68152, .k_proj = 68153, .k_bias = 68217, .v_proj = 68218, .v_bias = 68282, .o_proj = 68283, .o_tiles = 256, .post_norm = 68539, .gate_proj = 68540, .up_proj = 69228, .down_proj = 69916, .mlp_tiles = 688, .down_tiles = 688, .is_mlp_w8 = false, .is_down_w8 = false, .is_o_w4 = false },
        .{ .in_norm = 70604, .q_proj = 70605, .q_bias = 71117, .k_proj = 71118, .k_bias = 71182, .v_proj = 71183, .v_bias = 71247, .o_proj = 71248, .o_tiles = 256, .post_norm = 71504, .gate_proj = 71505, .up_proj = 72193, .down_proj = 72881, .mlp_tiles = 688, .down_tiles = 688, .is_mlp_w8 = false, .is_down_w8 = false, .is_o_w4 = false },
        .{ .in_norm = 73569, .q_proj = 73570, .q_bias = 74082, .k_proj = 74083, .k_bias = 74147, .v_proj = 74148, .v_bias = 74212, .o_proj = 74213, .o_tiles = 256, .post_norm = 74469, .gate_proj = 74470, .up_proj = 75158, .down_proj = 75846, .mlp_tiles = 688, .down_tiles = 688, .is_mlp_w8 = false, .is_down_w8 = false, .is_o_w4 = false },
        .{ .in_norm = 76534, .q_proj = 76535, .q_bias = 77047, .k_proj = 77048, .k_bias = 77112, .v_proj = 77113, .v_bias = 77177, .o_proj = 77178, .o_tiles = 256, .post_norm = 77434, .gate_proj = 77435, .up_proj = 78123, .down_proj = 78811, .mlp_tiles = 688, .down_tiles = 688, .is_mlp_w8 = false, .is_down_w8 = false, .is_o_w4 = false },
        .{ .in_norm = 79499, .q_proj = 79500, .q_bias = 80012, .k_proj = 80013, .k_bias = 80077, .v_proj = 80078, .v_bias = 80142, .o_proj = 80143, .o_tiles = 256, .post_norm = 80399, .gate_proj = 80400, .up_proj = 81088, .down_proj = 81776, .mlp_tiles = 688, .down_tiles = 688, .is_mlp_w8 = false, .is_down_w8 = false, .is_o_w4 = false },
        .{ .in_norm = 82464, .q_proj = 82465, .q_bias = 82977, .k_proj = 82978, .k_bias = 83042, .v_proj = 83043, .v_bias = 83107, .o_proj = 83108, .o_tiles = 256, .post_norm = 83364, .gate_proj = 83365, .up_proj = 84053, .down_proj = 84741, .mlp_tiles = 688, .down_tiles = 688, .is_mlp_w8 = false, .is_down_w8 = false, .is_o_w4 = false },
        .{ .in_norm = 85429, .q_proj = 85430, .q_bias = 85942, .k_proj = 85943, .k_bias = 86007, .v_proj = 86008, .v_bias = 86072, .o_proj = 86073, .o_tiles = 256, .post_norm = 86329, .gate_proj = 86330, .up_proj = 87018, .down_proj = 87706, .mlp_tiles = 688, .down_tiles = 688, .is_mlp_w8 = false, .is_down_w8 = false, .is_o_w4 = false },
        .{ .in_norm = 88394, .q_proj = 88395, .q_bias = 88907, .k_proj = 88908, .k_bias = 88972, .v_proj = 88973, .v_bias = 89037, .o_proj = 89038, .o_tiles = 256, .post_norm = 89294, .gate_proj = 89295, .up_proj = 89983, .down_proj = 90671, .mlp_tiles = 688, .down_tiles = 688, .is_mlp_w8 = false, .is_down_w8 = false, .is_o_w4 = false },
        .{ .in_norm = 91359, .q_proj = 91360, .q_bias = 91872, .k_proj = 91873, .k_bias = 91937, .v_proj = 91938, .v_bias = 92002, .o_proj = 92003, .o_tiles = 256, .post_norm = 92259, .gate_proj = 92260, .up_proj = 92948, .down_proj = 93636, .mlp_tiles = 688, .down_tiles = 688, .is_mlp_w8 = false, .is_down_w8 = false, .is_o_w4 = false },
        .{ .in_norm = 94324, .q_proj = 94325, .q_bias = 94837, .k_proj = 94838, .k_bias = 94902, .v_proj = 94903, .v_bias = 94967, .o_proj = 94968, .o_tiles = 256, .post_norm = 95224, .gate_proj = 95225, .up_proj = 95913, .down_proj = 96601, .mlp_tiles = 688, .down_tiles = 688, .is_mlp_w8 = false, .is_down_w8 = false, .is_o_w4 = false },
        .{ .in_norm = 97289, .q_proj = 97290, .q_bias = 97802, .k_proj = 97803, .k_bias = 97867, .v_proj = 97868, .v_bias = 97932, .o_proj = 97933, .o_tiles = 256, .post_norm = 98189, .gate_proj = 98190, .up_proj = 98878, .down_proj = 99566, .mlp_tiles = 688, .down_tiles = 688, .is_mlp_w8 = false, .is_down_w8 = false, .is_o_w4 = false },
        .{ .in_norm = 100254, .q_proj = 100255, .q_bias = 100767, .k_proj = 100768, .k_bias = 100832, .v_proj = 100833, .v_bias = 100897, .o_proj = 100898, .o_tiles = 256, .post_norm = 101154, .gate_proj = 101155, .up_proj = 102531, .down_proj = 103907, .mlp_tiles = 1376, .down_tiles = 1376, .is_mlp_w8 = true, .is_down_w8 = true, .is_o_w4 = false },
        .{ .in_norm = 105283, .q_proj = 105284, .q_bias = 105796, .k_proj = 105797, .k_bias = 105861, .v_proj = 105862, .v_bias = 105926, .o_proj = 105927, .o_tiles = 256, .post_norm = 106183, .gate_proj = 106184, .up_proj = 106872, .down_proj = 107560, .mlp_tiles = 688, .down_tiles = 688, .is_mlp_w8 = false, .is_down_w8 = false, .is_o_w4 = false },
        .{ .in_norm = 108248, .q_proj = 108249, .q_bias = 108761, .k_proj = 108762, .k_bias = 108826, .v_proj = 108827, .v_bias = 108891, .o_proj = 108892, .o_tiles = 256, .post_norm = 109148, .gate_proj = 109149, .up_proj = 109837, .down_proj = 110525, .mlp_tiles = 688, .down_tiles = 688, .is_mlp_w8 = false, .is_down_w8 = false, .is_o_w4 = false },
        .{ .in_norm = 111213, .q_proj = 111214, .q_bias = 111726, .k_proj = 111727, .k_bias = 111791, .v_proj = 111792, .v_bias = 111856, .o_proj = 111857, .o_tiles = 256, .post_norm = 112113, .gate_proj = 112114, .up_proj = 112802, .down_proj = 113490, .mlp_tiles = 688, .down_tiles = 688, .is_mlp_w8 = false, .is_down_w8 = false, .is_o_w4 = false },
        .{ .in_norm = 114178, .q_proj = 114179, .q_bias = 114691, .k_proj = 114692, .k_bias = 114756, .v_proj = 114757, .v_bias = 114821, .o_proj = 114822, .o_tiles = 256, .post_norm = 115078, .gate_proj = 115079, .up_proj = 115767, .down_proj = 116455, .mlp_tiles = 688, .down_tiles = 688, .is_mlp_w8 = false, .is_down_w8 = false, .is_o_w4 = false },
        .{ .in_norm = 117143, .q_proj = 117144, .q_bias = 117656, .k_proj = 117657, .k_bias = 117721, .v_proj = 117722, .v_bias = 117786, .o_proj = 117787, .o_tiles = 256, .post_norm = 118043, .gate_proj = 118044, .up_proj = 118732, .down_proj = 119420, .mlp_tiles = 688, .down_tiles = 688, .is_mlp_w8 = false, .is_down_w8 = false, .is_o_w4 = false },
    };

    pub inline fn inNorm(l: usize) usize { return LAYERS[l].in_norm; }
    pub inline fn qProj(l: usize) usize { return LAYERS[l].q_proj; }
    pub inline fn qBias(l: usize) usize { return LAYERS[l].q_bias; }
    pub inline fn kProj(l: usize) usize { return LAYERS[l].k_proj; }
    pub inline fn kBias(l: usize) usize { return LAYERS[l].k_bias; }
    pub inline fn vProj(l: usize) usize { return LAYERS[l].v_proj; }
    pub inline fn vBias(l: usize) usize { return LAYERS[l].v_bias; }
    pub inline fn oProj(l: usize) usize { return LAYERS[l].o_proj; }
    pub inline fn oTiles(l: usize) usize { return LAYERS[l].o_tiles; }
    pub inline fn postNorm(l: usize) usize { return LAYERS[l].post_norm; }
    pub inline fn gateProj(l: usize) usize { return LAYERS[l].gate_proj; }
    pub inline fn upProj(l: usize) usize { return LAYERS[l].up_proj; }
    pub inline fn downProj(l: usize) usize { return LAYERS[l].down_proj; }
    pub inline fn mlpTiles(l: usize) usize { return LAYERS[l].mlp_tiles; }
    pub inline fn downTiles(l: usize) usize { return LAYERS[l].down_tiles; }
    pub inline fn isOW4(l: usize) bool { return LAYERS[l].is_o_w4; }
    pub inline fn isMlpW8(l: usize) bool { return LAYERS[l].is_mlp_w8; }
    pub inline fn isDownW8(l: usize) bool { return LAYERS[l].is_down_w8; }
};
