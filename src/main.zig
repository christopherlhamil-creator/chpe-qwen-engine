//! Unified Polymorphic CHPE Forward Decode & Benchmark Runner
//!
//! Subsystem: tot_hybrid/src/main_chpe_fwd.zig
//! Blueprint: docs/superpowers/specs/2026-09-19-unified-chpe-engine-and-physical-gauntlet-design.md

const std = @import("std");
const chpe = @import("chpe_engine.zig");
const weight_archive = @import("weight_archive.zig");

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);

    var archive_path: [:0]const u8 = "models/warc/Qwen2.5-3B-Instruct.bf16.raw.chpe";
    var tokens_file_path: ?[:0]const u8 = null;
    var out_logits_path: ?[:0]const u8 = null;
    var arch_override: ?[:0]const u8 = null;
    var bench_iters: usize = 1;
    var max_seq_len: usize = 2048;

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--archive")) {
            if (i + 1 < args.len) {
                i += 1;
                const s = try allocator.allocSentinel(u8, args[i].len, 0);
                @memcpy(s, args[i]);
                archive_path = s;
            }
        } else if (std.mem.eql(u8, arg, "--tokens-file")) {
            if (i + 1 < args.len) {
                i += 1;
                const s = try allocator.allocSentinel(u8, args[i].len, 0);
                @memcpy(s, args[i]);
                tokens_file_path = s;
            }
        } else if (std.mem.eql(u8, arg, "--out-logits")) {
            if (i + 1 < args.len) {
                i += 1;
                const s = try allocator.allocSentinel(u8, args[i].len, 0);
                @memcpy(s, args[i]);
                out_logits_path = s;
            }
        } else if (std.mem.eql(u8, arg, "--arch")) {
            if (i + 1 < args.len) {
                i += 1;
                const s = try allocator.allocSentinel(u8, args[i].len, 0);
                @memcpy(s, args[i]);
                arch_override = s;
            }
        } else if (std.mem.eql(u8, arg, "--bench")) {
            if (i + 1 < args.len) {
                i += 1;
                bench_iters = std.fmt.parseInt(usize, args[i], 10) catch 1;
            }
        } else if (std.mem.eql(u8, arg, "--max-seq-len")) {
            if (i + 1 < args.len) {
                i += 1;
                max_seq_len = std.fmt.parseInt(usize, args[i], 10) catch 2048;
            }
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            std.debug.print(
                \\Unified Polymorphic CHPE Forward Engine
                \\
                \\Usage: chpe_fwd [OPTIONS]
                \\
                \\Options:
                \\  --archive <path>       Path to .chpe weight archive
                \\  --tokens-file <path>   Binary little-endian u32 tokens file
                \\  --out-logits <path>    Destination binary file for output float logits
                \\  --arch <name>          Model architecture (qwen2_5_3b | qwen3_5_9b | qwen2_5_72b)
                \\  --bench <iters>        Benchmark iterations (default: 1)
                \\  --max-seq-len <len>    Maximum sequence length (default: 2048)
                \\  -h, --help             Display this help message
                \\
            , .{});
            return;
        } else if (!std.mem.startsWith(u8, arg, "--")) {
            const s = try allocator.allocSentinel(u8, arg.len, 0);
            @memcpy(s, arg);
            archive_path = s;
        }
    }

    // Determine Model Architecture
    var arch: chpe.ModelArch = chpe.ModelArch.Qwen2_5_3B;
    if (arch_override) |ao| {
        if (std.mem.indexOf(u8, ao, "9b") != null or std.mem.indexOf(u8, ao, "9B") != null) {
            arch = chpe.ModelArch.Qwen3_5_9B;
        } else if (std.mem.indexOf(u8, ao, "72b") != null or std.mem.indexOf(u8, ao, "72B") != null) {
            arch = chpe.ModelArch.Qwen2_5_72B;
        } else {
            arch = chpe.ModelArch.Qwen2_5_3B;
        }
    } else {
        if (std.mem.indexOf(u8, archive_path, "9B") != null or std.mem.indexOf(u8, archive_path, "9b") != null or std.mem.indexOf(u8, archive_path, "qwen35") != null) {
            arch = chpe.ModelArch.Qwen3_5_9B;
        } else if (std.mem.indexOf(u8, archive_path, "72B") != null or std.mem.indexOf(u8, archive_path, "72b") != null) {
            arch = chpe.ModelArch.Qwen2_5_72B;
        } else {
            arch = chpe.ModelArch.Qwen2_5_3B;
        }
    }

    std.debug.print("=== [UNIFIED POLYMORPHIC CHPE FORWARD ENGINE] ===\n", .{});
    std.debug.print("Model Arch     : {s} ({d} layers, D={d}, intermediate={d}, vocab={d})\n", .{
        arch.name,
        arch.num_layers,
        arch.hidden_dim,
        arch.intermediate_dim,
        arch.vocab_size,
    });
    std.debug.print("Archive Path   : {s}\n", .{archive_path});

    // Open Archive POSIX mmap
    var archive = weight_archive.WeightArchive.openPosix(archive_path) catch |err| {
        std.debug.print("ERROR: Failed to open weight archive '{s}': {s}\n", .{ archive_path, @errorName(err) });
        return err;
    };
    defer archive.close();

    std.debug.print("Archive Size   : {d} bytes ({d:.2} GB, {d} tiles)\n", .{
        archive.file_size,
        @as(f64, @floatFromInt(archive.file_size)) / (1024.0 * 1024.0 * 1024.0),
        archive.recordCount(),
    });

    // Load custom tokens if provided
    var prompt_tokens: []u32 = undefined;
    var default_token = [_]u32{151644}; // <|im_start|> default
    if (tokens_file_path) |tfp| {
        const fd_val = std.os.linux.open(tfp, .{ .ACCMODE = .RDONLY }, 0);
        if (std.os.linux.errno(fd_val) != .SUCCESS) {
            std.debug.print("ERROR: Failed to open tokens file '{s}'\n", .{tfp});
            return error.FileNotFound;
        }
        const fd: std.posix.fd_t = @intCast(fd_val);
        defer _ = std.os.linux.close(fd);

        var st: std.os.linux.Statx = undefined;
        if (std.os.linux.statx(fd, "", std.os.linux.AT.EMPTY_PATH, std.os.linux.STATX.BASIC_STATS, &st) != 0) {
            return error.StatFailed;
        }
        const fsize: usize = @intCast(st.size);
        const num_tokens = fsize / @sizeOf(u32);
        if (num_tokens == 0) return error.EmptyTokensFile;

        prompt_tokens = try allocator.alloc(u32, num_tokens);
        _ = std.os.linux.read(fd, std.mem.sliceAsBytes(prompt_tokens).ptr, fsize);
        std.debug.print("Prompt Tokens  : Loaded {d} tokens from '{s}'\n", .{ prompt_tokens.len, tfp });
    } else {
        prompt_tokens = &default_token;
        std.debug.print("Prompt Tokens  : Default single token [{d}]\n", .{default_token[0]});
    }

    // Initialize Polymorphic Engine
    var engine = try chpe.CHPEEngine.init(allocator, archive.bytes, arch, max_seq_len);
    defer engine.deinit();

    std.debug.print("Hardware Core  : {s}\n", .{engine.hw_backend.name()});
    std.debug.print("Execution Loop : Running forward pass over {d} prompt tokens...\n", .{prompt_tokens.len});

    var min_ms: f64 = std.math.inf(f64);
    var max_ms: f64 = 0.0;
    var sum_ms: f64 = 0.0;
    var last_result: chpe.ForwardResult = undefined;

    for (0..bench_iters) |iter| {
        const t_start = chpe.nowNs();
        for (prompt_tokens, 0..) |tok, pos| {
            last_result = engine.forwardDecode(tok, pos);
        }
        const t_elapsed = chpe.nowNs() - t_start;
        const cur_ms = @as(f64, @floatFromInt(t_elapsed)) / 1_000_000.0;

        if (cur_ms < min_ms) min_ms = cur_ms;
        if (cur_ms > max_ms) max_ms = cur_ms;
        sum_ms += cur_ms;

        if (bench_iters > 1) {
            std.debug.print("  [Iter {d}/{d}] Latency: {d:.2} ms\n", .{ iter + 1, bench_iters, cur_ms });
        }
    }

    const mean_ms = sum_ms / @as(f64, @floatFromInt(bench_iters));
    const throughput = (@as(f64, @floatFromInt(prompt_tokens.len)) / (mean_ms / 1000.0));

    std.debug.print("\n=== [PHYSICAL SILICON EXECUTION RECEIPT] ===\n", .{});
    std.debug.print("Argmax Decoded Token ID : {d}\n", .{last_result.argmax_token});
    std.debug.print("Maximum Vocab Logit     : {d:.6}\n", .{last_result.max_logit});
    std.debug.print("Token 0 Logit           : {d:.6}\n", .{last_result.token0_logit});
    std.debug.print("Final Hidden Vector Norm: {d:.6}\n", .{last_result.hidden_norm});
    std.debug.print("Mean Latency            : {d:.2} ms ({d:.3} s)\n", .{ mean_ms, mean_ms / 1000.0 });
    std.debug.print("Throughput              : {d:.2} tokens/sec\n", .{throughput});
    std.debug.print("Numerical Soundness     : {s}\n", .{if (last_result.all_finite) "100% FINITE (Zero NaN / Zero Inf)" else "FAILED (NaN or Inf present)"});

    // Write output logits binary if requested
    if (out_logits_path) |olp| {
        const fd_val = std.os.linux.open(olp, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, 0o644);
        if (std.os.linux.errno(fd_val) == .SUCCESS) {
            const fd: std.posix.fd_t = @intCast(fd_val);
            defer _ = std.os.linux.close(fd);
            const raw_logits_bytes = std.mem.sliceAsBytes(engine.logits_buf);
            _ = std.os.linux.write(fd, raw_logits_bytes.ptr, raw_logits_bytes.len);
            std.debug.print("Exported Logits Binary  : {s} ({d} floats, {d} bytes)\n", .{ olp, engine.logits_buf.len, raw_logits_bytes.len });
        }
    }

    // Emit spoke telemetry JSON
    const spoke_outbox_path = "run/spoke_outbox/metal.chpe_fwd.result.json";
    const out_fd_val = std.os.linux.open(spoke_outbox_path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, 0o644);
    if (std.os.linux.errno(out_fd_val) == .SUCCESS) {
        const out_fd: std.posix.fd_t = @intCast(out_fd_val);
        defer _ = std.os.linux.close(out_fd);
        var json_buf: [1024]u8 = undefined;
        const json_slice = std.fmt.bufPrint(&json_buf,
            \\{{
            \\  "benchmark": "CHPE Forward Engine",
            \\  "model": "{s}",
            \\  "archive": "{s}",
            \\  "tokens_evaluated": {d},
            \\  "argmax_token": {d},
            \\  "max_logit": {d:.6},
            \\  "token0_logit": {d:.6},
            \\  "hidden_norm": {d:.6},
            \\  "mean_latency_ms": {d:.2},
            \\  "throughput_tok_s": {d:.2},
            \\  "all_finite": {s}
            \\}}
            \\
        , .{
            arch.name,
            archive_path,
            prompt_tokens.len,
            last_result.argmax_token,
            last_result.max_logit,
            last_result.token0_logit,
            last_result.hidden_norm,
            mean_ms,
            throughput,
            if (last_result.all_finite) "true" else "false",
        }) catch "";
        _ = std.os.linux.write(out_fd, json_slice.ptr, json_slice.len);
        std.debug.print("Recorded Spoke Telemetry: {s}\n", .{spoke_outbox_path});
    }
}
