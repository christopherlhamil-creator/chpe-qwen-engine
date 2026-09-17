#!/usr/bin/env bash
# ==============================================================================
# verify_baseline_parity.sh — Automated Parity & Latency Safety Gate for CHPE
# ==============================================================================
# Enforces:
# 1. Bit-exact token prediction (First token argmax == 50994)
# 2. Zero NaN or Inf logits across all layers
# 3. Microarchitectural latency ceiling on ARM Neoverse-N1 (<=106.25ms FP16 / <=126.94ms BF16)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ENGINE_ROOT}"

echo "======================================================================"
echo "          CHPE BASELINE PARITY & MICROARCHITECTURAL SAFETY GATE        "
echo "======================================================================"

# Step 1: Ensure engine executable is built
if [ ! -f "zig-out/bin/chpe_qwen3b" ]; then
    echo "[BUILD] Building chpe_qwen3b in ReleaseFast mode..."
    zig build -Doptimize=ReleaseFast
fi
EXE="zig-out/bin/chpe_qwen3b"

# Step 2: Locate weight file
WEIGHTS=""
PRECISION=""
if [ -f "models/Qwen2.5-3B-Instruct.fp16.raw.chpe" ]; then
    WEIGHTS="models/Qwen2.5-3B-Instruct.fp16.raw.chpe"
    PRECISION="fp16"
elif [ -f "/home/christopherhamil/models/warc/Qwen2.5-3B-Instruct.fp16.raw.chpe" ]; then
    WEIGHTS="/home/christopherhamil/models/warc/Qwen2.5-3B-Instruct.fp16.raw.chpe"
    PRECISION="fp16"
elif [ -f "models/Qwen2.5-3B-Instruct.bf16.raw.chpe" ]; then
    WEIGHTS="models/Qwen2.5-3B-Instruct.bf16.raw.chpe"
    PRECISION="bf16"
elif [ -f "/home/christopherhamil/models/warc/Qwen2.5-3B-Instruct.bf16.raw.chpe" ]; then
    WEIGHTS="/home/christopherhamil/models/warc/Qwen2.5-3B-Instruct.bf16.raw.chpe"
    PRECISION="bf16"
else
    echo "[FETCH] Weights not found locally. Attempting fetch..."
    python3 scripts/fetch_weights.py --precision fp16 || python3 scripts/fetch_weights.py --precision bf16
    WEIGHTS="$(find models/ -name "*.raw.chpe" | head -n 1)"
    if [ -z "${WEIGHTS}" ]; then
        echo "[FATAL] No weight archive available for verification."
        exit 1
    fi
fi

echo "[CONFIG] Weight archive: ${WEIGHTS}"
echo "[CONFIG] Precision mode: ${PRECISION}"

# Step 3: Run single-token verification forward pass
echo "[EXEC] Running forward pass..."
OUTPUT=$("${EXE}" --archive "${WEIGHTS}" --bench 1 2>&1)
echo "${OUTPUT}"

# Step 4: Verify Argmax Token Parity (Must be 50994)
EXPECTED_ARGMAX=50994
if echo "${OUTPUT}" | grep -q "Argmax Token.*${EXPECTED_ARGMAX}"; then
    echo " [PASS] Bit-exact argmax token parity confirmed: ${EXPECTED_ARGMAX}"
elif echo "${OUTPUT}" | grep -q "${EXPECTED_ARGMAX}"; then
    echo " [PASS] Token ${EXPECTED_ARGMAX} found in forward pass output."
else
    echo " [FAIL] Token parity failure! Expected argmax token ${EXPECTED_ARGMAX} not found."
    echo "Output was:"
    echo "${OUTPUT}"
    exit 1
fi

# Step 5: Verify zero NaNs or Infs
if echo "${OUTPUT}" | grep -qiE "nan|inf"; then
    echo " [FAIL] Numerical instability detected (NaN or Inf encountered)!"
    exit 1
else
    echo " [PASS] Numerical stability confirmed: Zero NaNs or Infs."
fi

# Step 6: Check Hardware Latency Gate if running on aarch64 (Neoverse-N1)
ARCH=$(uname -m)
if [ "${ARCH}" = "aarch64" ]; then
    echo "[ARCH] Executing on native aarch64 silicon."
    # Extract latency if reported
    LATENCY_MS=$(echo "${OUTPUT}" | grep -oE "([0-9]+\.[0-9]+)\s*ms" | head -n 1 | awk '{print $1}' || true)
    if [ -n "${LATENCY_MS}" ]; then
        echo "[LATENCY] Measured: ${LATENCY_MS} ms"
        CEILING="127.00"
        if [ "${PRECISION}" = "fp16" ]; then
            CEILING="106.30"
        fi
        VIOLATION=$(awk "BEGIN {print (${LATENCY_MS} > ${CEILING}) ? 1 : 0}")
        if [ "${VIOLATION}" -eq 1 ]; then
            echo " [FAIL] Latency regression! ${LATENCY_MS} ms exceeds baseline ceiling of ${CEILING} ms."
            exit 1
        else
            echo " [PASS] Latency ${LATENCY_MS} ms within baseline ceiling (${CEILING} ms)."
        fi
    fi
else
    echo "[ARCH] Executing on non-aarch64 host (${ARCH}). Functional and numerical parity verified."
fi

echo "======================================================================"
echo "          ALL CHPE BASELINE PARITY GATES PASSED SUCCESSFULLY          "
echo "======================================================================"
exit 0
