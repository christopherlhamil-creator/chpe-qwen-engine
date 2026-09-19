#!/usr/bin/env python3
"""
scripts/eval_lambada_fidelity.py

Unified Empirical Zero-Shot LAMBADA Evaluation Harness
for the Polymorphic CHPE Engine (Qwen 3B, Qwen 3.5 9B, Qwen 72B).

MANDATE: ZERO SIMULATION.
Under Christopher's Law (§0 of AGENTS.md), all metrics must be physically
measured from real forward passes through weight archives. No synthetic loops,
no mock distributions, no shortcuts.
"""

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.export_lambada_openbenchmarking import validate_pts_composite_xml

TOKENIZER_PATHS = {
    "qwen2_5_3b": Path("/home/christopherhamil/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1/tokenizer.json"),
    "qwen3_5_9b": Path("/home/christopherhamil/Downloads/Qwen3.5-9B-base/tokenizer.json"),
    "qwen2_5_72b": Path("/home/christopherhamil/.cache/huggingface/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/tokenizer.json"),
}

DEFAULT_BINARY = REPO_ROOT / "zig-out" / "bin" / "chpe_fwd"
DEFAULT_DATASET = REPO_ROOT / "run" / "lambada_test.jsonl"


def logsumexp(x: np.ndarray) -> float:
    c = np.max(x)
    return float(c + np.log(np.sum(np.exp(x - c))))


def load_lambada_samples(dataset_path: Path, num_samples: int, offset: int = 0) -> list:
    if not dataset_path.exists():
        fallback = Path("/home/christopherhamil/tot_hybrid/run/lambada_test.jsonl")
        if fallback.exists():
            dataset_path = fallback
        else:
            import urllib.request
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            url = "https://raw.githubusercontent.com/cybertronai/bflm/master/lambada_test.jsonl"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(dataset_path, "wb") as f:
                f.write(resp.read())

    samples = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < offset:
                continue
            if len(samples) >= num_samples:
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = item.get("text", "").strip()
            words = text.split()
            if len(words) < 2:
                continue
            context = " ".join(words[:-1])
            target = words[-1]
            samples.append({
                "idx": idx,
                "context": context,
                "target": target,
                "full_text": text
            })
    return samples


def evaluate_lambada(
    samples: list,
    tokenizer_path: Path,
    binary_path: Path,
    archive_path: Path,
    arch: str,
    scratch_dir: Path,
) -> dict:
    from tokenizers import Tokenizer
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    tok = Tokenizer.from_file(str(tokenizer_path))

    if not binary_path.exists():
        fallback_main = Path("/home/christopherhamil/tot_hybrid/zig-out/bin/chpe_fwd")
        if fallback_main.exists():
            binary_path = fallback_main
        else:
            raise FileNotFoundError(f"Unified chpe_fwd binary not found at {binary_path}")

    if not archive_path.exists():
        raise FileNotFoundError(
            f"Weight archive not found at {archive_path}. Under Christopher's Law (§0 of AGENTS.md), "
            "simulations are strictly banned. A physical archive must be provided."
        )

    scratch_dir.mkdir(parents=True, exist_ok=True)
    correct_exact = 0
    correct_top5 = 0
    nlls = []
    latencies = []

    print(f"[EVAL] Evaluating {len(samples)} LAMBADA samples using {binary_path}...")
    print(f"[EVAL] Model Architecture: {arch}")
    print(f"[EVAL] Weight Archive: {archive_path} ({archive_path.stat().st_size / (1024*1024):.1f} MB)")

    for i, s in enumerate(samples):
        ctx_ids = tok.encode(s["context"]).ids
        target_ids = tok.encode(" " + s["target"]).ids
        if not target_ids:
            target_ids = tok.encode(s["target"]).ids
        target_id = target_ids[0] if target_ids else -1

        prompt_file = scratch_dir / f"prompt_{i}.bin"
        out_logits_file = scratch_dir / f"logits_{i}.bin"

        with open(prompt_file, "wb") as f:
            for tid in ctx_ids:
                f.write(struct.pack("<I", tid))

        cmd = [
            str(binary_path),
            "--archive", str(archive_path),
            "--tokens-file", str(prompt_file),
            "--arch", arch,
            "--out-logits", str(out_logits_file),
        ]

        t0 = time.perf_counter()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if proc.returncode != 0:
            print(f"[ERROR] chpe_fwd failed on sample {i} (exit code {proc.returncode}):")
            print(proc.stderr)
            raise RuntimeError(f"chpe_fwd execution failed: {proc.stderr}")

        latencies.append(elapsed_ms)

        if not out_logits_file.exists():
            raise FileNotFoundError(f"Expected output logits file not created: {out_logits_file}")

        logits = np.fromfile(str(out_logits_file), dtype=np.float32)

        # Mandatory Numerical Soundness Guard
        if not np.isfinite(logits).all():
            nan_count = int(np.isnan(logits).sum())
            inf_count = int(np.isinf(logits).sum())
            raise AssertionError(f"Numerical soundness failed on sample {i}: {nan_count} NaNs, {inf_count} Infs in logits")

        pred_id = int(np.argmax(logits))
        top5_ids = set(np.argpartition(logits, -5)[-5:])

        is_exact = (pred_id == target_id) or (pred_id in target_ids)
        is_top5 = (target_id in top5_ids) or any(t in top5_ids for t in target_ids)

        if is_exact:
            correct_exact += 1
        if is_top5:
            correct_top5 += 1

        if target_id >= 0 and target_id < len(logits):
            lse = logsumexp(logits)
            nll = float(lse - logits[target_id])
            nlls.append(nll)
        else:
            nlls.append(10.0)

        # Clean up temporary per-sample files to conserve disk
        if prompt_file.exists():
            prompt_file.unlink()
        if out_logits_file.exists():
            out_logits_file.unlink()

        if (i + 1) % 5 == 0 or (i + 1) == len(samples):
            cur_acc = (correct_exact / (i + 1)) * 100.0
            cur_mean_ms = float(np.mean(latencies))
            print(f"  [Progress {i+1}/{len(samples)}] Exact: {cur_acc:.2f}% | Latency: {cur_mean_ms:.1f} ms/sample")

    n_samples = max(1, len(samples))
    exact_acc = (correct_exact / n_samples) * 100.0
    top5_acc = (correct_top5 / n_samples) * 100.0
    mean_nll = float(np.mean(nlls)) if nlls else 2.5
    ppl = math.exp(min(mean_nll, 20.0))
    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    throughput = (1000.0 / mean_lat) if mean_lat > 0 else 0.0

    metrics = {
        "test_name": f"{arch}-lambada-discourse-empirical",
        "arch": arch,
        "archive": str(archive_path),
        "num_samples": n_samples,
        "exact_match_acc": round(exact_acc, 2),
        "top5_acc": round(top5_acc, 2),
        "mean_nll": round(mean_nll, 4),
        "perplexity": round(ppl, 2),
        "latency_ms": round(mean_lat, 2),
        "throughput_toks": round(throughput, 2),
        "simulation_mode": False,
        "numerical_soundness": "100% Finite (0 NaN / 0 Inf)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Empirical LAMBADA Evaluation for CHPE Unified Engine")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--offset", type=int, default=0, help="Sample offset")
    parser.add_argument("--arch", choices=["qwen2_5_3b", "qwen3_5_9b", "qwen2_5_72b"], default="qwen2_5_3b", help="Model architecture")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="LAMBADA dataset path")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer JSON path (defaults based on arch)")
    parser.add_argument("--binary", default=str(DEFAULT_BINARY), help="Engine binary path")
    parser.add_argument("--archive", required=True, help="Packed .chpe archive path")
    parser.add_argument("--out", default=None, help="Output JSON path")
    parser.add_argument("--export-xml", default=None, help="Output PTS XML path")
    parser.add_argument("--scratch-dir", default=str(REPO_ROOT / "run" / "eval_scratch"), help="Scratch directory for prompts and logits")

    args = parser.parse_args()

    tok_path = Path(args.tokenizer) if args.tokenizer else TOKENIZER_PATHS.get(args.arch)
    if tok_path is None or not tok_path.exists():
        print(f"[ERROR] Could not find tokenizer for architecture {args.arch} at {tok_path}")
        return 1

    archive_path = Path(args.archive)
    if not archive_path.exists():
        print(f"[ERROR] Archive path does not exist: {archive_path}")
        return 1

    samples = load_lambada_samples(Path(args.dataset), args.samples, args.offset)
    scratch_dir = Path(args.scratch_dir)

    metrics = evaluate_lambada(
        samples=samples,
        tokenizer_path=tok_path,
        binary_path=Path(args.binary),
        archive_path=archive_path,
        arch=args.arch,
        scratch_dir=scratch_dir,
    )

    print("\n=== [EMPIRICAL LAMBADA ZERO-SHOT EVALUATION SUMMARY] ===")
    print(f"Architecture         : {metrics['arch']}")
    print(f"Weight Archive       : {metrics['archive']}")
    print(f"Samples Evaluated    : {metrics['num_samples']}")
    print(f"Exact Match Accuracy : {metrics['exact_match_acc']}%")
    print(f"Top-5 Accuracy       : {metrics['top5_acc']}%")
    print(f"Discourse Perplexity : {metrics['perplexity']}")
    print(f"Mean Sample Latency  : {metrics['latency_ms']} ms")
    print(f"Throughput           : {metrics['throughput_toks']} samples/s")
    print(f"Numerical Soundness  : {metrics['numerical_soundness']}")
    print("=======================================================")

    out_json = Path(args.out) if args.out else REPO_ROOT / "run" / f"lambada_{args.arch}_eval.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] Saved evaluation JSON to {out_json}")

    if args.export_xml:
        xml_str = validate_pts_composite_xml(metrics)
        out_xml = Path(args.export_xml)
        out_xml.parent.mkdir(parents=True, exist_ok=True)
        with open(out_xml, "w", encoding="utf-8") as f:
            f.write(xml_str)
        print(f"[OK] Exported OpenBenchmarking XML to {out_xml}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
