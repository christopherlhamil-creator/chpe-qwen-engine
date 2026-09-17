#!/usr/bin/env python3
"""
fetch_weights.py — Automated weight downloader and integrity verifier for CHPE.

Fetches raw contiguous CHPE weight archives from Hugging Face and verifies
exact streaming SHA256 checksums against models/manifest.json.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ID = "Siddachan/qwen2.5-3b-chpe-raw"
ENGINE_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ENGINE_ROOT / "models"
MANIFEST_PATH = MODELS_DIR / "manifest.json"


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(32 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_file(file_path: Path, expected_size: int, expected_sha: str) -> bool:
    if not file_path.exists():
        return False
    actual_size = file_path.stat().st_size
    if actual_size != expected_size:
        print(f"[ERROR] Size mismatch for {file_path.name}: {actual_size} != {expected_size}")
        return False
    print(f"[VERIFY] Computing SHA256 for {file_path.name} ({actual_size / (1024**3):.2f} GB)...")
    actual_sha = compute_sha256(file_path)
    if actual_sha != expected_sha:
        print(f"[ERROR] Checksum mismatch for {file_path.name}:")
        print(f"  Expected: {expected_sha}")
        print(f"  Actual:   {actual_sha}")
        return False
    print(f"[OK] Checksum verified: {actual_sha}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Fetch and verify CHPE model weights from Hugging Face.")
    parser.add_argument("--precision", choices=["fp16", "bf16", "all"], default="fp16",
                        help="Weight precision archive to fetch (default: fp16)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify existing local files without downloading")
    parser.add_argument("--token", type=str, default=None,
                        help="Hugging Face access token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        print(f"[ERROR] Missing manifest at {MANIFEST_PATH}")
        sys.exit(1)

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    targets = []
    if args.precision in ("fp16", "all") and "Qwen2.5-3B-Instruct.fp16.raw.chpe" in manifest["files"]:
        targets.append("Qwen2.5-3B-Instruct.fp16.raw.chpe")
    if args.precision in ("bf16", "all") and "Qwen2.5-3B-Instruct.bf16.raw.chpe" in manifest["files"]:
        targets.append("Qwen2.5-3B-Instruct.bf16.raw.chpe")

    if not targets:
        print(f"[WARN] No matching files in manifest for precision={args.precision}")
        sys.exit(1)

    # Check local models dir or system cache
    system_warc = Path("/home/christopherhamil/models/warc")

    for filename in targets:
        dest = MODELS_DIR / filename
        file_info = manifest["files"][filename]
        expected_size = file_info["size_bytes"]
        expected_sha = file_info["sha256"]

        # Check if already present in models/
        if dest.exists() and verify_file(dest, expected_size, expected_sha):
            print(f"[EXISTS] {filename} is present and valid in {MODELS_DIR}")
            continue

        # Check if present in system warc directory and symlink/copy
        warc_candidate = system_warc / filename
        if warc_candidate.exists() and warc_candidate.stat().st_size == expected_size:
            print(f"[LINK] Found matching file in {system_warc}. Creating symlink in {MODELS_DIR}...")
            if dest.is_symlink() or dest.exists():
                dest.unlink()
            dest.symlink_to(warc_candidate)
            if verify_file(dest, expected_size, expected_sha):
                print(f"[OK] Successfully linked {filename}")
                continue

        if args.verify_only:
            print(f"[FAIL] {filename} is missing or invalid.")
            sys.exit(1)

        # Download from Hugging Face
        print(f"[DOWNLOAD] Fetching {filename} from Hugging Face repository {REPO_ID}...")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print("[ERROR] huggingface_hub is not installed. Run: pip install huggingface_hub")
            sys.exit(1)

        token = args.token or os.environ.get("HF_TOKEN")
        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="model",
            token=token,
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False
        )
        print(f"[DOWNLOAD COMPLETE] Saved to {downloaded_path}")

        if not verify_file(Path(downloaded_path), expected_size, expected_sha):
            print(f"[FATAL] Downloaded file {filename} failed SHA256 integrity verification!")
            sys.exit(1)

    print("\n[SUCCESS] All requested model weights verified.")


if __name__ == "__main__":
    main()
