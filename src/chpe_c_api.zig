//! CHPE C ABI Export Interface for Qwen2.5-3B / Qwen3.5
//!
//! Provides pure extern "C" symbols for external runtimes, Python ctypes,
//! lm-evaluation-harness, and Hugging Face PreTrainedModel bridges without
//! touching Safetensors or incurring Python serialization tax.

const std = @import("std");
const weight_archive = @import("weight_archive.zig");
const chpe = @import("chpe_engine.zig");

pub const MAX_VOCAB_SIZE: usize = 248320;

pub const ChpeEngineContext = struct {
    archive: weight_archive.WeightArchive,
    arch: chpe.ModelArch,
    engine: chpe.CHPEEngine,
    last_argmax: u32 = 0,
    last_max_logit: f32 = 0.0,
    last_token0_logit: f32 = 0.0,
    last_hidden_norm: f32 = 0.0,
    last_elapsed_ns: u64 = 0,
    last_logits: [MAX_VOCAB_SIZE]f32 = undefined,
};

pub export fn chpe_init(archive_path: [*:0]const u8) callconv(.c) ?*anyopaque {
    const span = std.mem.span(archive_path);
    const archive = weight_archive.WeightArchive.openPosix(span) catch return null;

    const arch = blk: {
        if (std.mem.indexOf(u8, span, "9B") != null or std.mem.indexOf(u8, span, "9b") != null or std.mem.indexOf(u8, span, "qwen35") != null) {
            break :blk chpe.ModelArch.Qwen3_5_9B;
        } else if (std.mem.indexOf(u8, span, "72B") != null or std.mem.indexOf(u8, span, "72b") != null) {
            break :blk chpe.ModelArch.Qwen2_5_72B;
        } else {
            break :blk chpe.ModelArch.Qwen2_5_3B;
        }
    };

    const engine = chpe.CHPEEngine.init(std.heap.page_allocator, archive.bytes, arch, 2048) catch return null;

    const ctx = std.heap.page_allocator.create(ChpeEngineContext) catch return null;
    ctx.* = .{
        .archive = archive,
        .arch = arch,
        .engine = engine,
    };
    return @ptrCast(ctx);
}

pub export fn chpe_forward_tokens(
    ctx_ptr: ?*anyopaque,
    tokens: ?[*]const u32,
    num_tokens: usize,
    out_logits: ?[*]f32,
) callconv(.c) c_int {
    if (ctx_ptr == null or tokens == null or num_tokens == 0) return -1;
    const ctx: *ChpeEngineContext = @ptrCast(@alignCast(ctx_ptr.?));
    const toks = tokens.?[0..num_tokens];

    var last_res: chpe.ForwardResult = undefined;
    for (toks, 0..) |tok, pos| {
        last_res = ctx.engine.forwardDecode(tok, pos);
    }

    ctx.last_argmax = last_res.argmax_token;
    ctx.last_max_logit = last_res.max_logit;
    ctx.last_token0_logit = last_res.token0_logit;
    ctx.last_hidden_norm = last_res.hidden_norm;
    ctx.last_elapsed_ns = last_res.elapsed_ns;

    const vocab = ctx.arch.vocab_size;
    @memcpy(ctx.last_logits[0..vocab], ctx.engine.logits_buf[0..vocab]);
    if (out_logits) |out| {
        @memcpy(out[0..vocab], ctx.engine.logits_buf[0..vocab]);
    }

    return @intCast(last_res.argmax_token);
}

pub export fn chpe_get_last_logits(ctx_ptr: ?*anyopaque, out_logits: ?[*]f32, max_len: usize) callconv(.c) usize {
    if (ctx_ptr == null or out_logits == null) return 0;
    const ctx: *ChpeEngineContext = @ptrCast(@alignCast(ctx_ptr.?));
    const vocab = ctx.arch.vocab_size;
    if (max_len < vocab) return 0;
    @memcpy(out_logits.?[0..vocab], ctx.last_logits[0..vocab]);
    return vocab;
}

pub export fn chpe_get_vocab_size(ctx_ptr: ?*anyopaque) callconv(.c) usize {
    if (ctx_ptr == null) return chpe.ModelArch.Qwen2_5_3B.vocab_size;
    const ctx: *ChpeEngineContext = @ptrCast(@alignCast(ctx_ptr.?));
    return ctx.arch.vocab_size;
}

pub export fn chpe_get_hidden_dim(ctx_ptr: ?*anyopaque) callconv(.c) usize {
    if (ctx_ptr == null) return chpe.ModelArch.Qwen2_5_3B.hidden_dim;
    const ctx: *ChpeEngineContext = @ptrCast(@alignCast(ctx_ptr.?));
    return ctx.arch.hidden_dim;
}

pub export fn chpe_get_num_layers(ctx_ptr: ?*anyopaque) callconv(.c) usize {
    if (ctx_ptr == null) return chpe.ModelArch.Qwen2_5_3B.num_layers;
    const ctx: *ChpeEngineContext = @ptrCast(@alignCast(ctx_ptr.?));
    return ctx.arch.num_layers;
}

pub export fn chpe_get_last_elapsed_ns(ctx_ptr: ?*anyopaque) callconv(.c) u64 {
    if (ctx_ptr == null) return 0;
    const ctx: *ChpeEngineContext = @ptrCast(@alignCast(ctx_ptr.?));
    return ctx.last_elapsed_ns;
}

pub export fn chpe_get_last_argmax(ctx_ptr: ?*anyopaque) callconv(.c) u32 {
    if (ctx_ptr == null) return 0;
    const ctx: *ChpeEngineContext = @ptrCast(@alignCast(ctx_ptr.?));
    return ctx.last_argmax;
}

pub export fn chpe_get_last_max_logit(ctx_ptr: ?*anyopaque) callconv(.c) f32 {
    if (ctx_ptr == null) return 0.0;
    const ctx: *ChpeEngineContext = @ptrCast(@alignCast(ctx_ptr.?));
    return ctx.last_max_logit;
}

pub export fn chpe_get_last_token0_logit(ctx_ptr: ?*anyopaque) callconv(.c) f32 {
    if (ctx_ptr == null) return 0.0;
    const ctx: *ChpeEngineContext = @ptrCast(@alignCast(ctx_ptr.?));
    return ctx.last_token0_logit;
}

pub export fn chpe_get_last_hidden_norm(ctx_ptr: ?*anyopaque) callconv(.c) f32 {
    if (ctx_ptr == null) return 0.0;
    const ctx: *ChpeEngineContext = @ptrCast(@alignCast(ctx_ptr.?));
    return ctx.last_hidden_norm;
}

pub export fn chpe_free(ctx_ptr: ?*anyopaque) callconv(.c) void {
    if (ctx_ptr == null) return;
    const ctx: *ChpeEngineContext = @ptrCast(@alignCast(ctx_ptr.?));
    ctx.engine.deinit();
    ctx.archive.close();
    std.heap.page_allocator.destroy(ctx);
}
