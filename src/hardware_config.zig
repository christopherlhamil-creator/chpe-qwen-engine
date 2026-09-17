//! Comptime Hardware Architecture & Kernel Execution Directives
//! Generated automatically by scripts/solve_ebm_hardware_gap.py
//! Verified by Z3 + Vampire + Leo-III + EBM solver stack.

pub const TARGET_HARDWARE = "ARMv8.2-A Neoverse-N1 (4 Cores @ 3.0 GHz) - BF16 CHPE Engine";
pub const TARGET_ISA = "arm64";
pub const ACTIVE_CORES: usize = 4;
pub const VOCAB_ROWS_PER_CORE: usize = 37984;
pub const L1D_WORKING_BUDGET_BYTES: usize = 49152;
pub const TILE_WORKING_SET_BYTES: usize = 24576;

pub const FUSED_SWIGLU_WORKING_SET_BYTES: usize = 128;
pub const UNFUSED_TWOPASS_WORKING_SET_BYTES: usize = 88064;
pub const TILES_PER_GROUP: usize = 1;
pub const TOTAL_INTERMEDIATE_TILES: usize = 2752;
pub const TOTAL_INTERMEDIATE_GROUPS: usize = 2752;
pub const WORKER_0_TILE_START: usize = 0;
pub const WORKER_0_TILE_END: usize = 688;
pub const WORKER_1_TILE_START: usize = 688;
pub const WORKER_1_TILE_END: usize = 1376;
pub const WORKER_2_TILE_START: usize = 1376;
pub const WORKER_2_TILE_END: usize = 2064;
pub const WORKER_3_TILE_START: usize = 2064;
pub const WORKER_3_TILE_END: usize = 2752;
pub const SDOT_VECTOR_REGS_USED: usize = 9;
pub const SDOT_VECTOR_REGS_BUDGET: usize = 32;

pub const SYMMETRICAL_ROW_ISOLATION: bool = true;
pub const SYMMETRICAL_BLOCK_SIZE: usize = 128;

pub const VECTOR_DOT_LANES: usize = 4;
pub const VECTOR_INSTRUCTION = "fmla.4s";
pub const VECTOR_WIDTH_BITS: usize = 128;
pub const UNROLL_ACCUMULATOR_FACTOR: usize = 2;
pub const REDUCTION_STRIDE: usize = 2048;
pub const USE_INT8_SDOT: bool = false;

pub const DRAM_BANDWIDTH_SATURATION_GB_S: f32 = 41.8449;
pub const TARGET_THROUGHPUT_TOK_S: f32 = 6.78;
pub const TARGET_DECODE_MS: f32 = 147.55;
pub const LATEST_MEASURED_DECODE_MS: f32 = 210.02;
pub const LATEST_MEASURED_TOK_S: f32 = 4.761;

// Upstream competitor baseline & remaining squeeze target
pub const LLAMA_BASELINE_MS: f32 = 165.60;
pub const LLAMA_BASELINE_TOK_S: f32 = 6.04;
pub const REMAINING_GAP_TO_LLAMA_MS: f32 = 44.42;
pub const SQUEEZED_TARGET_DECODE_MS: f32 = 155.00;
pub const SQUEEZED_TARGET_TOK_S: f32 = 6.45;

// Profiled layer breakdown (ms)
pub const PROFILE_GATE_UP_MS: f32 = 113.16;
pub const PROFILE_DOWN_PROJ_MS: f32 = 49.62;
pub const PROFILE_LM_HEAD_MS: f32 = 21.86;
pub const PROFILE_QKV_PROJ_MS: f32 = 14.55;
pub const PROFILE_O_PROJ_MS: f32 = 10.98;
pub const PROFILE_RMSNORM_MS: f32 = 0.33;
pub const PROFILE_ATTN_GQA_MS: f32 = 0.05;

// Squeeze targets (ms)
pub const TARGET_GATE_UP_MS: f32 = 80.00;
pub const TARGET_DOWN_PROJ_MS: f32 = 38.00;
pub const TARGET_LM_HEAD_MS: f32 = 16.00;

const std = @import("std");

test "hardware config parameters" {
    try std.testing.expectEqual(@as(usize, 4), ACTIVE_CORES);
    try std.testing.expectEqual(@as(usize, 49152), L1D_WORKING_BUDGET_BYTES);
    try std.testing.expect(SDOT_VECTOR_REGS_USED <= SDOT_VECTOR_REGS_BUDGET);
    try std.testing.expect(SQUEEZED_TARGET_DECODE_MS < LLAMA_BASELINE_MS);
}
