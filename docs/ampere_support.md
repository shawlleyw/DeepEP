# DeepEP Ampere (A100) Low-Latency Support

## Overview

This document describes the changes made to enable DeepEP's low-latency (LL) kernels on NVIDIA A100 (SM80) GPUs, which previously required Hopper (SM90) features. The port removes the TMA and FP8 dependencies for the low-latency path, enabling BF16 intranode low-latency dispatch and combine over NVLink P2P on Ampere hardware.

## Problem Statement

DeepEP's low-latency kernels (`internode_ll.cu`) were originally written exclusively for Hopper GPUs, using:

- **TMA (Tensor Memory Accelerator)**: `cp.async.bulk`, `tma_load_1d`, `tma_store_1d` for bulk shared-memory staging
- **mbarrier**: Hardware async barriers for producer-consumer pipelining
- **FP8 intrinsics**: `__nv_cvt_float2_to_fp8x2` for hardware FP8 conversion
- **Cluster launch**: `cudaLaunchKernelEx` with cluster dimensions

The build system (`setup.py`) enforced mutual exclusion: `DISABLE_SM90_FEATURES=1` asserted `disable_nvshmem`, making it impossible to compile any NVSHMEM-dependent code for SM80.

However, for **intranode-only** deployments (single node, 8 GPUs), the low-latency kernels communicate via NVLink P2P (`nvshmemi_get_p2p_ptr`), not RDMA. The SM90-specific features were used only for **local memory movement** (global ↔ shared), not for inter-GPU communication. This made an SM80 port feasible.

## Changes Made

### DeepEP Changes (8 files, +196/-36 lines)

#### 1. Build System (`setup.py`)
- Removed `assert disable_nvshmem` when `DISABLE_SM90_FEATURES=1`
- When NVSHMEM is available on SM80, added `-rdc=true` and `--ptxas-options=--register-usage-level=10` for NVSHMEM device linking
- Conditionally compile `internode.cu` only when SM90 is available (normal internode dispatch/combine uses TMA without fallback)
- `internode_ll.cu` is always compiled when NVSHMEM is available (has SM80 fallback paths)

#### 2. Low-Latency Kernels (`csrc/kernels/internode_ll.cu`)
- **Dispatch kernel**: No changes needed for BF16 path — it already uses `UNROLLED_WARP_COPY` and `st_na_global` without TMA. Added `#ifdef DISABLE_SM90_FEATURES` to block FP8 template instantiation.
- **Combine send phase**: Added SM80 fallback using `UNROLLED_WARP_COPY` for direct global-to-global copy. The SM90 path uses a 3-stage TMA pipeline with mbarrier; the SM80 path does direct warp copies without pipelining.
- **Combine receive phase**: Added SM80 fallback using direct global memory reads (`__ldg`) and register-based weighted accumulation. The SM90 path uses a producer-consumer TMA pipeline with separate load and reduction warps; the SM80 path has all decode warps reading directly from global memory.
- **Combine wrapper**: Separate `COMBINE_LAUNCH_CASE` macros for SM90 (with `SET_SHARED_MEMORY_FOR_TMA` and LogFMT support) and SM80 (no shared memory needed, BF16 only).

#### 3. Launch Infrastructure (`csrc/kernels/launch.cuh`)
- SM80 `SETUP_LAUNCH_CONFIG` now uses `cudaLaunchKernelEx` with `cudaLaunchAttributeCooperative` (needed for `cg::this_grid().sync()` in the LL kernels)
- Removed the old `<<<>>>` launch path for SM80

#### 4. SM80 Stubs (`csrc/kernels/configs.cuh`, `csrc/kernels/utils.cuh`)
- Added FP8 type stubs: `__nv_fp8x2_storage_t`, `__NV_SATFINITE`, `__nv_cvt_float2_to_fp8x2` no-op
- Added TMA function stubs: `tma_store_fence()`, `tma_store_wait<N>()` (referenced by templates compiled but not executed on SM80)

#### 5. C++ Wrapper Guards (`csrc/deep_ep.cpp`, `csrc/config.hpp`)
- Added `#ifdef DISABLE_SM90_FEATURES` runtime asserts blocking FP8 dispatch and LogFMT combine
- Normal internode dispatch/combine guarded by `#if !defined(DISABLE_NVSHMEM) && !defined(DISABLE_SM90_FEATURES)` (since `internode.cu` is not compiled on SM80)
- `internode::get_source_meta_bytes()` references in `config.hpp` guarded by `DISABLE_SM90_FEATURES`
- Low-latency mask buffer functions guarded by `#ifndef DISABLE_NVSHMEM`

#### 6. Tests (`tests/test_low_latency.py`)
- Skip FP8 dispatch when `is_sm90_compiled() == False`
- Skip LogFMT combine when SM90 is not available
- Performance test uses `use_fp8=sm90` instead of hardcoded `True`

### SGLang Changes (3 files, +117/-6 lines)

#### `python/sglang/srt/layers/moe/moe_runner/triton.py`
- Registered `deepep_ll` → `triton` pre-permute: flattens `[num_local_experts, E_tokens, hidden]` into `[total_rows, hidden]` with per-token expert IDs, then creates triton kernel config via `moe_align_block_size`
- Registered `triton` → `deepep_ll` post-permute: reshapes triton output back to `[num_local_experts, E_tokens, hidden]` for LL combine

This was needed because on A100, the `deep_gemm` MoE runner (which has LL permute functions) is not available — only the `triton` runner is used.

## Feature Support Matrix

| Feature | H100 (SM90) | A100 (SM80) |
|---|---|---|
| Normal intranode dispatch/combine (NVLink IPC) | ✅ | ✅ |
| Normal internode dispatch/combine (RDMA) | ✅ | ❌ Not compiled |
| Low-latency dispatch BF16 | ✅ | ✅ |
| Low-latency dispatch FP8 | ✅ | ❌ Requires SM90 |
| Low-latency combine BF16 | ✅ | ✅ |
| Low-latency combine LogFMT | ✅ | ❌ Requires SM90 |
| Low-latency NVLink P2P (intranode) | ✅ | ✅ |
| Low-latency IBGDA RDMA (internode) | ✅ | ✅ (with driver config) |
| CUDA graph support | ✅ | ✅ |
| Zero-copy combine | ✅ | ✅ |
| Recv hook overlap | ✅ | ✅ |
| Mask buffer (shrink mode) | ✅ | ✅ |

## Build Instructions

### A100 (SM80) with low-latency support

```bash
# NVSHMEM must be available (pip package or NVSHMEM_DIR)
pip install nvidia-nvshmem-cu12

# Build with SM80 target + NVSHMEM
DISABLE_SM90_FEATURES=1 pip install .

# Or with explicit NVSHMEM path
NVSHMEM_DIR=/path/to/nvshmem DISABLE_SM90_FEATURES=1 pip install .
```

### A100 (SM80) without low-latency (normal mode only)

```bash
# Block NVSHMEM detection to disable all internode/LL features
DISABLE_SM90_FEATURES=1 python -c "
import sys; sys.modules['nvidia.nvshmem'] = None
exec(open('setup.py').read())
" install
```

### H100 (SM90) — unchanged

```bash
pip install .
```

## Performance Results

### Standalone DeepEP tests (8×A100-80GB, NVLink)

**Intranode normal** (`test_intranode.py`):
- Dispatch: 185 GB/s, 1687 µs
- Combine: 179 GB/s, 1744 µs

**Low-latency** (`test_low_latency.py`):
- Dispatch + Combine: 80.5 GB/s, 274 µs
- Dispatch only: 50-54 GB/s, 140 µs
- Combine only: 110-122 GB/s, 120-130 µs

### SGLang end-to-end (Qwen3-30B-A3B, 8×A100-80GB, EP auto + CUDA graph)

| Phase | Latency | Throughput |
|---|---|---|
| Prefill | 328 ms | 7,806 tokens/s |
| Decode (median) | 124 ms | 2,058 tokens/s |
| Total | 6.85 s | 2,242 tokens/s |

## NVSHMEM Memory Management

### How SGLang Calculates Required DeepEP Buffer Sizes

SGLang computes buffer sizes in `DeepEPBuffer.get_deepep_buffer()` (`sglang/srt/layers/moe/token_dispatcher/deepep.py`). DeepEP uses two types of buffers:

- **NVL buffer** (`num_nvl_bytes`): Allocated via CUDA IPC (`cudaIpcGetMemHandle`) for intranode NVLink communication. Used only by normal mode. NVSHMEM is NOT involved.
- **RDMA buffer** (`num_rdma_bytes`): Allocated via NVSHMEM symmetric heap (`nvshmem_align`). Used by internode normal mode and all low-latency mode.

NVSHMEM is initialized only when `num_rdma_ranks > 1` (multi-node) or `low_latency_mode == True`. For pure intranode normal mode (single node, ≤8 GPUs), NVSHMEM is never initialized and no symmetric heap is allocated.

### Normal vs Low-Latency Buffer Sizing

The two modes use fundamentally different buffer designs:

**Normal mode** uses a **streaming/chunked** design. The kernel sends data in small chunks and waits for each chunk to be consumed before reusing the buffer space. The buffer is sized for a sliding window of in-flight tokens, controlled by the `Config` parameters `num_max_nvl_chunked_recv_tokens` and `num_max_rdma_chunked_recv_tokens` (typically a few hundred tokens). The formula is roughly `num_channels × num_peers × chunk_window × per_token_bytes`. This is memory-efficient — the buffer size is fixed regardless of total batch size.

**Low-latency mode** uses a **flat pre-allocated** design. The entire batch for all experts must fit in the buffer simultaneously — there is no chunking or flow control. The buffer is sized as `num_experts × num_max_dispatch_tokens_per_rank × per_slot_bytes`, double-buffered (odd/even for overlapping dispatch and combine). Per-slot bytes include `hidden × sizeof(bf16)` plus metadata (LogFMT min/max values, control int4). This scales with maximum batch size and number of experts.

| | Normal NVL | Normal RDMA | Low-Latency RDMA |
|---|---|---|---|
| Sizing principle | channels × peers × chunk_window | channels × rdma_peers × chunk_window | experts × max_tokens × per_slot (×2) |
| Scales with | chunk config, hidden | chunk config, hidden, num_nodes | max batch size, num_experts, hidden |
| Buffer reuse | Yes (sliding window) | Yes (sliding window) | No (entire batch pre-allocated) |
| Example (Qwen3-30B-A3B, 8 GPUs) | ~120 MB | 0 (intranode only) | ~520 MB |

The normal mode trades latency for memory efficiency (chunking adds synchronization overhead). The low-latency mode trades memory for minimum latency (no chunking, no flow control).

### How SGLang Selects Buffer Sizes

SGLang computes buffer sizes in `DeepEPBuffer.get_deepep_buffer()` (`sglang/srt/layers/moe/token_dispatcher/deepep.py`):

**Normal mode** (`deepep_mode.enable_normal()`):
- NVL bytes: `Config.get_nvl_buffer_size_hint(hidden_bytes, num_ranks)` — sized for the chunked intranode IPC window. Always computed when normal mode is enabled.
- RDMA bytes: `Config.get_rdma_buffer_size_hint(hidden_bytes, num_ranks)` — returns **zero** when `num_ranks <= NUM_MAX_NVL_PEERS` (single node, ≤8 GPUs), because normal intranode uses NVL/IPC, not NVSHMEM RDMA. For multi-node, it sizes the internode RDMA chunked buffer. In sglang, this is only calculated when `ENABLE_JIT_DEEPGEMM` is true — the normal internode dispatch/combine path in sglang depends on deep_gemm (Hopper-only). On A100, `ENABLE_JIT_DEEPGEMM` is false, so normal RDMA bytes are always zero.

**Low-latency mode** (`deepep_mode.enable_low_latency()`):
- RDMA bytes: `Buffer.get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts)` — computed by `LowLatencyLayout` in `config.hpp`. Always requires NVSHMEM.

**Auto mode**: Takes `max(normal_rdma, ll_rdma)` for RDMA bytes and uses the normal-mode NVL bytes. On A100 intranode, normal RDMA = 0 (no internode, no deep_gemm), so the RDMA buffer is purely the low-latency requirement. NVSHMEM is initialized because `low_latency_mode=True`.

NVSHMEM is initialized only when `num_rdma_ranks > 1` (multi-node) or `low_latency_mode == True`. For pure intranode normal mode (single node, ≤8 GPUs), NVSHMEM is never initialized and no symmetric heap is allocated.

### NVSHMEM Symmetric Heap Sizing

NVSHMEM pre-allocates a symmetric heap at `nvshmemx_init_attr()` time. The default size is 1 GB per PE, controlled by `NVSHMEM_SYMMETRIC_SIZE`. This heap must be allocated as a single contiguous block — it cannot grow on demand because all PEs must maintain symmetric address spaces.

The actual buffer usage is typically much less than 1 GB. For example, Qwen3-30B-A3B needs ~520 MB RDMA + shrink buffer overhead. The excess (~480 MB per GPU) is wasted.

To mitigate this, `buffer.py` now automatically sets `NVSHMEM_SYMMETRIC_SIZE` to `num_rdma_bytes + 128 MiB` (for internal metadata and alignment headroom) before NVSHMEM initialization, unless the user has explicitly set it via the environment variable. This avoids the default 1 GB reservation while leaving enough room for NVSHMEM's internal allocations. Users can still override with:

```bash
export NVSHMEM_SYMMETRIC_SIZE=600000000  # manual override
```

On memory-constrained A100 deployments (e.g., large models with `--mem-fraction-static 0.9`), this automatic sizing can be the difference between successful NVSHMEM initialization and `cuMemCreate` OOM failures.

## Potential Optimization Points

### 1. Combine receive: `cp.async` pipelining (estimated +20-30% combine throughput)

The SM80 combine receive path reads directly from global memory without pipelining. SM80 supports `cp.async.cg.shared.global` (LDGSTS), which could overlap loads with computation. This would restore some of the throughput lost from removing TMA's 3-stage pipeline:

```
Current SM80:  load → accumulate → load → accumulate (serial)
Optimized:     load[i+1] → accumulate[i] (pipelined via cp.async)
```

### 2. Combine send: async copy pipelining (estimated +10-15% send throughput)

Similarly, the combine send phase uses `UNROLLED_WARP_COPY` which is synchronous. Using `cp.async` for the source read + manual barriers could overlap loads with NVLink P2P stores.

### 3. Idle warp utilization in combine (estimated +10% throughput)

In the SM90 path, one warp per group is the "TMA loader" while others are "reduction warps." In the SM80 path, this loader warp (`decode_warp_idx == num_decode_warps`) is idle. It could be repurposed to contribute to reduction work, increasing parallelism.

### 4. FP8 software conversion for dispatch (feature parity)

The A100 path blocks FP8 dispatch entirely. A software implementation of `__nv_cvt_float2_to_fp8x2` could enable FP8 dispatch on SM80 with minimal overhead (the conversion is not on the critical path — it happens during the dispatch send phase).

### 5. Normal internode on SM80 (feature expansion)

Currently `internode.cu` is not compiled for SM80 because it also uses TMA without fallback. Porting the normal internode kernels (similar TMA → warp-copy replacement) would enable multi-node A100 deployments with DeepEP's normal dispatch/combine.

Note that cross-node support on A100 requires fixes at both layers:
- **DeepEP**: Port `internode.cu` TMA code to SM80 (same approach as the `internode_ll.cu` port — replace TMA pipeline with `UNROLLED_WARP_COPY` or `cp.async`).
- **SGLang**: The normal-mode RDMA buffer calculation is currently gated on `ENABLE_JIT_DEEPGEMM` (Hopper-only). This gate would need to be relaxed so that RDMA buffers are allocated for multi-node A100 deployments regardless of deep_gemm availability. The `deepep_normal` → `triton` runner path (already registered) would handle the MoE computation.

The NVL (IPC) and RDMA buffers are fundamentally separate memory pools — NVL buffers are regular `cudaMalloc` memory shared via `cudaIpcGetMemHandle` (intranode only), while RDMA buffers are NVSHMEM symmetric heap memory registered with the InfiniBand NIC for cross-node access. They cannot be interchanged.

### 6. Triton MoE runner optimization for LL format

The current `deepep_ll` → `triton` pre-permute flattens all expert slots (including padding) into the triton input. A more efficient approach would use `masked_m` to pack only valid tokens, reducing wasted computation on padding rows. This matters when token distribution is skewed across experts.
