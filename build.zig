const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Geometry module
    const geom_mod = b.createModule(.{
        .root_source_file = b.path("src/geometry.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Weight archive module
    const warc_mod = b.createModule(.{
        .root_source_file = b.path("src/weight_archive.zig"),
        .target = target,
        .optimize = optimize,
    });
    warc_mod.addImport("geometry", geom_mod);

    // CHPE Engine module
    const chpe_mod = b.createModule(.{
        .root_source_file = b.path("src/chpe_engine.zig"),
        .target = target,
        .optimize = optimize,
    });
    chpe_mod.addImport("geometry", geom_mod);
    chpe_mod.addImport("weight_archive", warc_mod);

    // Main executable
    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe_mod.addImport("geometry", geom_mod);
    exe_mod.addImport("weight_archive", warc_mod);
    exe_mod.addImport("chpe_engine", chpe_mod);

    const exe = b.addExecutable(.{
        .name = "chpe_fwd",
        .root_module = exe_mod,
    });

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    const run_step = b.step("run", "Run unified CHPE forward inference engine");
    run_step.dependOn(&run_cmd.step);
}
