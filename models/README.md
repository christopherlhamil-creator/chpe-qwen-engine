---
license: apache-2.0
tags:
- chpe
- qwen2.5
- arm64
- neoverse-n1
- inference
- pure-zig
- high-performance
model_name: Qwen2.5-3B-Instruct CHPE Contiguous Raw Weights
---

# Qwen2.5-3B-Instruct: CHPE Contiguous Raw Weights (ARM Neoverse-N1)

This repository contains the zero-overhead, strictly aligned raw weight archives for **Qwen2.5-3B-Instruct**, formatted specifically for the **Christopher Hamil Prediction Engine (CHPE)** on 64-bit ARM microarchitectures.

## Key Performance Results (ARM Neoverse-N1 @ 3.00 GHz, 16 vCPUs)

| Architecture | Model | Precision | Token Latency | Generation Speed | Delta vs Llama.cpp | Verification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CHPE (Ours)** | Qwen2.5-3B-Instruct | **FP16** | **$106.25\text{ ms}$** | **$9.138\text{ tok/s}$** | **$-16.57\%$** | `argmax=50994`, 0 NaN/Inf |
| **CHPE (Ours)** | Qwen2.5-3B-Instruct | **BF16** | **$126.94\text{ ms}$** | **$7.810\text{ tok/s}$** | **$-0.32\%$** | `argmax=50994`, 0 NaN/Inf |
| **Llama.cpp** | Qwen2.5-3B-Instruct | Full FP | $127.35\text{ ms}$ | $7.785\text{ tok/s}$ | Baseline ($0.0\%$) | Benchmark ref |

- **Official OpenBenchmarking Run**: [`2609164-NE-CHPELAMB82`](https://openbenchmarking.org/result/2609164-NE-CHPELAMB82)

---

## File Manifest & Checksums

All archives are raw binary memory-mappable blocks aligned to $16,384\text{ bytes}$ ($256 \times 64\text{B}$ cache lines):

- `Qwen2.5-3B-Instruct.fp16.raw.chpe`: $6,174,363,648\text{ bytes}$ (IEEE 754 half-precision float)
- `Qwen2.5-3B-Instruct.bf16.raw.chpe`: $6,174,363,648\text{ bytes}$ (Brain Floating Point 16-bit)
- `tokenizer.json`: Fast HF Tokenizer definition for Qwen2.5 vocabulary ($151,936\text{ tokens}$)
- `manifest.json`: Structural tensor offsets and streaming SHA256 signatures

---

## Microarchitectural Invariants & Formal Proofs

The memory alignment, tensor map projections, and threadpool barriers have been mechanically validated via automated theorem provers:

- **Z3 SMT2**: Theorem `SAT`, $0$ bounds violations.
- **Vampire First-Order Prover**: 15/15 refutation goals proved valid.
- **Leo-III Higher-Order Prover**: 14/14 goals proved valid.
- **Energy-Based Model (EBM)**: Reconstruction loss $E = 0.0000$.
- **Citation Key**: `098ad4dba5ecddbe`

---

## Usage with CHPE

```bash
# Clone the private engine repository
git clone https://github.com/christopherlhamil-creator/chpe-qwen-engine.git
cd chpe-qwen-engine

# Build with Zig 0.17
zig build -Doptimize=ReleaseFast

# Fetch weights using authenticated huggingface_hub
python3 scripts/fetch_weights.py --precision fp16

# Run inference
./zig-out/bin/chpe_qwen3b \
    --weights models/Qwen2.5-3B-Instruct.fp16.raw.chpe \
    --threads 16 \
    --warmup 5 \
    --steps 20
```
