//! Executable CLI runner and benchmark suite for Qwen2.5-3B .chpe Forward Decode.
//!
//! Subsystem: tot_hybrid/src/main_qwen3b_fwd.zig
//! Blueprint: docs/BLUEPRINT-20260914-QWEN3B-CHPE-PIPELINE.md

const std = @import("std");
const engine = @import("qwen3b_engine.zig");
const weight_archive = @import("weight_archive.zig");

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);

    var archive_path: [:0]const u8 = "models/warc/Qwen2.5-3B-Instruct.bf16.chpe";
    var bench_iters: usize = 1;
    var run_seq: bool = false;
    var custom_tokens: ?[]u32 = null;

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
        } else if (std.mem.eql(u8, arg, "--to-dense")) {
            if (i + 1 < args.len) {
                i += 1;
                const dst_path = try allocator.allocSentinel(u8, args[i].len, 0);
                @memcpy(dst_path, args[i]);
                std.debug.print("=== [CHPE DENSE ARCHIVE CONVERTER] ===\n", .{});
                std.debug.print("Source Archive : {s}\n", .{archive_path});
                std.debug.print("Target Dense   : {s}\n", .{dst_path});
                std.debug.print("Stripping 3,072B PreFetchLabelArea sector padding...\n", .{});
                try weight_archive.convertToDensePosix(archive_path, dst_path);
                std.debug.print("Success: Dense 17,408B-stride archive written. Zero DRAM bus padding.\n", .{});
                return;
            }
        } else if (std.mem.eql(u8, arg, "--to-raw")) {
            if (i + 1 < args.len) {
                i += 1;
                const dst_path = try allocator.allocSentinel(u8, args[i].len, 0);
                @memcpy(dst_path, args[i]);
                std.debug.print("=== [CHPE RAW CONTIGUOUS ARCHIVE CONVERTER] ===\n", .{});
                std.debug.print("Source Archive : {s}\n", .{archive_path});
                std.debug.print("Target Raw     : {s}\n", .{dst_path});
                std.debug.print("Stripping ALL 4,096B sector + cell padding (1.54 GB dead bandwidth)...\n", .{});
                try weight_archive.convertToRawContiguousPosix(archive_path, dst_path);
                std.debug.print("Success: Raw contiguous 16,384B-stride archive written (6.172 GB). Zero DRAM bus padding.\n", .{});
                return;
            }
        } else if (std.mem.eql(u8, arg, "--to-raw-fp16")) {
            if (i + 1 < args.len) {
                i += 1;
                const dst_path = try allocator.allocSentinel(u8, args[i].len, 0);
                @memcpy(dst_path, args[i]);
                std.debug.print("=== [CHPE RAW CONTIGUOUS FP16 ARCHIVE CONVERTER] ===\n", .{});
                std.debug.print("Source Archive : {s}\n", .{archive_path});
                std.debug.print("Target Raw FP16: {s}\n", .{dst_path});
                std.debug.print("Converting BF16 -> IEEE 754 FP16 (ARMv8.2-A fmla.8h line rate)...\n", .{});
                try weight_archive.convertToRawFp16ContiguousPosix(archive_path, dst_path);
                std.debug.print("Success: Raw contiguous FP16 archive written (6.174 GB). Line-rate 8-lane SIMD.\n", .{});
                return;
            }
        } else if (std.mem.eql(u8, arg, "--bench")) {
            if (i + 1 < args.len) {
                i += 1;
                bench_iters = std.fmt.parseInt(usize, args[i], 10) catch 1;
            }
        } else if (std.mem.eql(u8, arg, "--seq")) {
            run_seq = true;
        } else if (std.mem.eql(u8, arg, "--tokens-file")) {
            if (i + 1 < args.len) {
                i += 1;
                const fpath = try allocator.allocSentinel(u8, args[i].len, 0);
                @memcpy(fpath, args[i]);
                const fd_val = std.os.linux.open(fpath, .{ .ACCMODE = .RDONLY }, 0);
                if (std.os.linux.errno(fd_val) == .SUCCESS) {
                    const fd: std.posix.fd_t = @intCast(fd_val);
                    defer _ = std.os.linux.close(fd);
                    var st: std.os.linux.Statx = undefined;
                    if (std.os.linux.statx(fd, "", std.os.linux.AT.EMPTY_PATH, std.os.linux.STATX.BASIC_STATS, &st) == 0) {
                        const fsize: usize = @intCast(st.size);
                        const num_tokens = fsize / @sizeOf(u32);
                        const tok_buf = try allocator.alloc(u32, num_tokens);
                        _ = std.os.linux.read(fd, std.mem.sliceAsBytes(tok_buf).ptr, fsize);
                        custom_tokens = tok_buf;
                    }
                }
            }
        } else if (std.mem.eql(u8, arg, "--tokens")) {
            if (i + 1 < args.len) {
                i += 1;
                const tok_buf = try allocator.alloc(u32, 2048);
                var count: usize = 0;
                var it = std.mem.splitScalar(u8, args[i], ',');
                while (it.next()) |chunk| {
                    const trimmed = std.mem.trim(u8, chunk, " \t\r\n");
                    if (trimmed.len > 0 and count < 2048) {
                        tok_buf[count] = try std.fmt.parseInt(u32, trimmed, 10);
                        count += 1;
                    }
                }
                custom_tokens = tok_buf[0..count];
            }
        } else if (!std.mem.startsWith(u8, arg, "--")) {
            const s = try allocator.allocSentinel(u8, arg.len, 0);
            @memcpy(s, arg);
            archive_path = s;
        }
    }

    std.debug.print("=== [QWEN3B CHPE BENCHMARK RUNNER] ===\n", .{});
    std.debug.print("Model Archive : {s}\n", .{archive_path});
    std.debug.print("Benchmark Runs: {}\n", .{bench_iters});
    if (custom_tokens) |toks| {
        std.debug.print("Custom Prompt : {} tokens\n", .{toks.len});
    }

    var min_ms: f64 = std.math.inf(f64);
    var max_ms: f64 = 0.0;
    var sum_ms: f64 = 0.0;
    var last_res: engine.ForwardResult = undefined;

    const default_tok0 = [_]u32{0};
    const active_prompt = if (custom_tokens) |toks| toks else &default_tok0;

    var archive = try weight_archive.WeightArchive.openPosix(archive_path);
    defer archive.close();

    if (bench_iters > 1) {
        std.debug.print("Warming up archive and cache across all 36 transformer layers...\n", .{});
        _ = engine.decodeSequenceWithArchive(&archive, active_prompt, .generative_f32) catch {};
    }

    for (0..bench_iters) |iter| {
        std.debug.print("Executing Run {}/{} across all 36 transformer layers (prompt_len={})...\n", .{ iter + 1, bench_iters, active_prompt.len });
        const res = engine.decodeSequenceWithArchive(&archive, active_prompt, .generative_f32) catch |err| {
            std.debug.print("ERROR: Decode failed with error: {s}\n", .{@errorName(err)});
            return err;
        };
        last_res = res;
        const cur_ms = @as(f64, @floatFromInt(res.elapsed_ns)) / 1_000_000.0;
        if (cur_ms < min_ms) min_ms = cur_ms;
        if (cur_ms > max_ms) max_ms = cur_ms;
        sum_ms += cur_ms;
        std.debug.print("  -> Run {} latency: {d:.2} ms ({d:.3} s)\n", .{ iter + 1, cur_ms, cur_ms / 1000.0 });
    }

    const avg_ms = sum_ms / @as(f64, @floatFromInt(bench_iters));
    const tokens_per_sec = 1000.0 / avg_ms;

    std.debug.print("\n=== [MEASURED BENCHMARK SUMMARY] ===\n", .{});
    std.debug.print("Argmax Decoded Token ID : {}\n", .{last_res.argmax_token});
    std.debug.print("Maximum Vocab Logit     : {d:.6}\n", .{last_res.max_logit});
    std.debug.print("Token 0 Logit           : {d:.6}\n", .{last_res.token0_logit});
    std.debug.print("Final Hidden Vector Norm: {d:.6}\n", .{last_res.hidden_norm});
    std.debug.print("Min Decode Time         : {d:.2} ms\n", .{min_ms});
    std.debug.print("Mean Decode Time        : {d:.2} ms\n", .{avg_ms});
    std.debug.print("Max Decode Time         : {d:.2} ms\n", .{max_ms});
    std.debug.print("Decode Throughput       : {d:.3} tokens/sec\n", .{tokens_per_sec});
    std.debug.print("Layers Executed         : 36\n", .{});
    std.debug.print("Status                  : {s}\n", .{if (last_res.all_finite) "SUCCESS (All logits finite)" else "FAIL (Non-finite logit detected)"});

    std.debug.print("\n--- [MICROARCHITECTURAL LATENCY BREAKDOWN (Last Run)] ---\n", .{});
    const qkv_ms = @as(f64, @floatFromInt(engine.prof_qkv_ns)) / 1_000_000.0;
    const attn_ms = @as(f64, @floatFromInt(engine.prof_attn_ns)) / 1_000_000.0;
    const oproj_ms = @as(f64, @floatFromInt(engine.prof_oproj_ns)) / 1_000_000.0;
    const norm_ms = @as(f64, @floatFromInt(engine.prof_norm_ns)) / 1_000_000.0;
    const gateup_ms = @as(f64, @floatFromInt(engine.prof_gateup_ns)) / 1_000_000.0;
    const down_ms = @as(f64, @floatFromInt(engine.prof_down_ns)) / 1_000_000.0;
    const head_ms = @as(f64, @floatFromInt(engine.prof_head_ns)) / 1_000_000.0;
    const total_prof_ms = qkv_ms + attn_ms + oproj_ms + norm_ms + gateup_ms + down_ms + head_ms;
    std.debug.print("  QKV Proj (36 layers)    : {d:6.2} ms ({d:4.1}%)\n", .{ qkv_ms, qkv_ms / total_prof_ms * 100.0 });
    std.debug.print("  Attn GQA (36 layers)    : {d:6.2} ms ({d:4.1}%)\n", .{ attn_ms, attn_ms / total_prof_ms * 100.0 });
    std.debug.print("  O Proj   (36 layers)    : {d:6.2} ms ({d:4.1}%)\n", .{ oproj_ms, oproj_ms / total_prof_ms * 100.0 });
    std.debug.print("  RMSNorm  (72 layers)    : {d:6.2} ms ({d:4.1}%)\n", .{ norm_ms, norm_ms / total_prof_ms * 100.0 });
    std.debug.print("  Gate/Up  (36 layers)    : {d:6.2} ms ({d:4.1}%)\n", .{ gateup_ms, gateup_ms / total_prof_ms * 100.0 });
    std.debug.print("  Down     (36 layers)    : {d:6.2} ms ({d:4.1}%)\n", .{ down_ms, down_ms / total_prof_ms * 100.0 });
    std.debug.print("  LM Head  (151k vocab)   : {d:6.2} ms ({d:4.1}%)\n", .{ head_ms, head_ms / total_prof_ms * 100.0 });
    std.debug.print("  Total Profiled Core Time: {d:6.2} ms\n", .{total_prof_ms});

    // Dump all token-0 logits to run/qwen3b_engine_token0_logits.bin and run/qwen3b_engine_logits.bin
    const logits_token0_path: [*:0]const u8 = "run/qwen3b_engine_token0_logits.bin";
    const bin0_fd = std.os.linux.open(logits_token0_path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, 0o644);
    if (std.os.linux.errno(bin0_fd) == .SUCCESS) {
        const fd: std.posix.fd_t = @intCast(bin0_fd);
        defer _ = std.os.linux.close(fd);
        const bytes_slice: []const u8 = std.mem.sliceAsBytes(last_res.logits);
        _ = std.os.linux.write(fd, bytes_slice.ptr, bytes_slice.len);
        std.debug.print("Token-0 Logits ({d} floats) dumped to {s}\n", .{ last_res.logits.len, logits_token0_path });
    }
    const logits_legacy_path: [*:0]const u8 = "run/qwen3b_engine_logits.bin";
    const bin_leg_fd = std.os.linux.open(logits_legacy_path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, 0o644);
    if (std.os.linux.errno(bin_leg_fd) == .SUCCESS) {
        const fd: std.posix.fd_t = @intCast(bin_leg_fd);
        defer _ = std.os.linux.close(fd);
        const bytes_slice: []const u8 = std.mem.sliceAsBytes(last_res.logits);
        _ = std.os.linux.write(fd, bytes_slice.ptr, bytes_slice.len);
    }

    if (run_seq) {
        std.debug.print("\n=== [MULTI-TOKEN SEQUENCE VALIDATION (seq_len=2)] ===\n", .{});
        const prompt_tokens = [_]u32{ 151643, 0 };
        const seq_res = engine.decodeSequenceWithArchive(&archive, &prompt_tokens, .generative_f32) catch |err| {
            std.debug.print("ERROR: Multi-token sequence decode failed: {s}\n", .{@errorName(err)});
            return err;
        };
        const seq_ms = @as(f64, @floatFromInt(seq_res.elapsed_ns)) / 1_000_000.0;
        std.debug.print("Sequence Argmax Token ID: {}\n", .{seq_res.argmax_token});
        std.debug.print("Sequence Max Logit      : {d:.6}\n", .{seq_res.max_logit});
        std.debug.print("Sequence Elapsed Time   : {d:.2} ms\n", .{seq_ms});
        std.debug.print("Sequence Status         : {s}\n", .{if (seq_res.all_finite) "PASS (All logits finite)" else "FAIL"});

        // Dump sequence logits to run/qwen3b_engine_seq2_logits.bin
        const logits_seq2_path: [*:0]const u8 = "run/qwen3b_engine_seq2_logits.bin";
        const bin_seq_fd = std.os.linux.open(logits_seq2_path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, 0o644);
        if (std.os.linux.errno(bin_seq_fd) == .SUCCESS) {
            const fd: std.posix.fd_t = @intCast(bin_seq_fd);
            defer _ = std.os.linux.close(fd);
            const bytes_slice: []const u8 = std.mem.sliceAsBytes(seq_res.logits);
            _ = std.os.linux.write(fd, bytes_slice.ptr, bytes_slice.len);
            std.debug.print("Sequence Logits ({d} floats) dumped to {s}\n", .{ seq_res.logits.len, logits_seq2_path });
        }
    }

    // Write spoke outbox telemetry via POSIX open
    const out_path: [*:0]const u8 = "/home/christopherhamil/tot_hybrid/run/spoke_outbox/metal.qwen3b_fwd.result.json";
    const fd_val = std.os.linux.open(out_path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, 0o644);
    if (std.os.linux.errno(fd_val) == .SUCCESS) {
        const fd: std.posix.fd_t = @intCast(fd_val);
        defer _ = std.os.linux.close(fd);

        var json_buf: [2048]u8 = undefined;
        const json_str = try std.fmt.bufPrint(&json_buf,
            \\{{
            \\  "model": "Qwen2.5-3B-Instruct",
            \\  "archive": "{s}",
            \\  "layers_executed": 36,
            \\  "benchmark_runs": {},
            \\  "min_ms": {d:.2},
            \\  "mean_ms": {d:.2},
            \\  "max_ms": {d:.2},
            \\  "tokens_per_sec": {d:.3},
            \\  "argmax_token": {},
            \\  "max_logit": {d:.6},
            \\  "token0_logit": {d:.6},
            \\  "hidden_norm": {d:.6},
            \\  "all_finite": {},
            \\  "status": "{s}"
            \\}}
            \\
        , .{
            archive_path,
            bench_iters,
            min_ms,
            avg_ms,
            max_ms,
            tokens_per_sec,
            last_res.argmax_token,
            last_res.max_logit,
            last_res.token0_logit,
            last_res.hidden_norm,
            last_res.all_finite,
            if (last_res.all_finite) "PASS" else "FAIL",
        });

        _ = std.os.linux.write(fd, json_str.ptr, json_str.len);
        std.debug.print("Telemetry written to {s}\n", .{out_path});
    }
}
