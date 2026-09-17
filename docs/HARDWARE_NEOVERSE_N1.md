# Hardware Specification: ARM Neoverse-N1 (GCP Tau T2A)

## Architecture Overview

The **Christopher Hamil Prediction Engine (CHPE)** target microarchitecture for this baseline lock-in is the **ARM Neoverse-N1** (ARMv8.2-A microarchitecture, codename *Ares*), provisioned on Google Cloud Platform **Tau T2A** (`t2a-standard-16`).

```
+-------------------------------------------------------------------------+
|                              ARM Neoverse-N1 Core                       |
|  +------------------------+   +--------------------------------------+  |
|  | Execution Pipelines    |   | Memory Hierarchy                     |  |
|  | - 4-wide decode/rename |   | - 64 KB L1 Data Cache (4-way, 64B)   |  |
|  | - 2x 128-bit NEON SIMD |   | - 64 KB L1 Instruction Cache         |  |
|  | - Native BF16 & FP16   |   | - 512 KB Private L2 Cache (8-way)    |  |
|  | - Dual 128b load/store |   |                                      |  |
|  +------------------------+   +--------------------------------------+  |
+-------------------------------------------------------------------------+
                                     |
                                  CMN-600
                        Coherent Mesh Network (SLC)
                                     |
                        32 MB Shared System Level Cache
                                     |
               8-Channel DDR4-3200 Memory Subsystem (~204.8 GB/s)
```

---

## Technical Specifications

| Parameter | Specification | Microarchitectural Significance for CHPE |
| :--- | :--- | :--- |
| **ISA** | ARMv8.2-A + Crypto + DotProd + FP16 + BF16 | Direct vector FMA execution without emulation |
| **Cores / Threads** | 16 vCPUs (1 thread per core, no SMT) | Deterministic cache occupancy without hyperthread contention |
| **L1 Data Cache** | 64 KB per core (4-way associative, 64B line) | Matches activation working set; zero cache spilling |
| **L1 Instruction Cache**| 64 KB per core (4-way associative) | Entire inner forward loop fits in L1i; zero i-cache misses |
| **L2 Unified Cache** | 512 KB per core (private, non-inclusive) | Holds intermediate attention QKV projections and RMSNorm buffers |
| **System Level Cache** | 32 MB shared interconnect (CMN-600) | Buffers KV-cache across token steps |
| **Memory Bus** | 8-Channel DDR4-3200 | Aggregate theoretical peak: ~204.8 GB/s |
| **Execution Units** | 2x 128-bit NEON vector ALUs per core | Throughput of 32 FP16 / BF16 operations per cycle per core |
| **Load/Store Units** | 2x 128-bit loads per cycle | Full saturation of NEON registers from L1/L2 |
| **Huge Pages** | 2 MB Transparent Huge Pages (THP) enabled | Reduces TLB misses during weight archive memory mapping |

---

## Memory Bandwidth & Latency Limits

For single-token autoregressive generation on Qwen2.5-3B ($3.09 \times 10^9$ parameters):

- **BF16 / FP16 Weight Footprint**: $6,174,363,648\text{ bytes} \approx 6.174\text{ GB}$.
- **Theoretical DRAM Read Limit** (@ 204.8 GB/s peak):
  $$\tau_{\text{min}} = \frac{6.174\text{ GB}}{204.8\text{ GB/s}} = 30.15\text{ ms} \quad (\approx 33.17\text{ tok/s})$$
- **Practical 16-Core Saturation Limit** (accounting for memory controller overhead and bus arbitration, ~120–140 GB/s sustained):
  $$\tau_{\text{practical}} \approx 44\text{--}51\text{ ms} \quad (\approx 19.5\text{--}22.7\text{ tok/s})$$
- **Measured CHPE FP16 Baseline**:
  $$\tau_{\text{measured}} = 106.25\text{ ms} \quad (9.138\text{ tok/s})$$
- **Measured CHPE BF16 Baseline**:
  $$\tau_{\text{measured}} = 126.94\text{ ms} \quad (7.810\text{ tok/s})$$
- **Llama.cpp Full FP Baseline**:
  $$\tau_{\text{llama.cpp}} = 127.35\text{ ms} \quad (7.785\text{ tok/s})$$

---

## Cache Geometry & Memory Alignment

CHPE achieves high cache utilization through strict memory alignment:

1. **16,384-Byte Tile Alignment**:
   All weight tensors are partitioned into chunks of 16,384 bytes ($256 \times 64\text{B}$ cache lines), matching the exact page boundaries and NEON vector load alignments.
2. **Deterministic Core Pinning**:
   Each worker thread in the 16-core threadpool is pinned 1:1 to a physical core using `sched_setaffinity()`, preventing kernel scheduler thread migration and L1/L2 cache cold starts.
3. **Prefetch Tuning**:
   Software prefetch directives (`@prefetch(ptr, .{})`) stage upcoming weight blocks into L1/L2 ahead of the inner dot-product loops, hiding memory latency behind NEON vector execution.
