# Multi-Stream Implementation Plan for ROCm Backend

## Overview
Port CUDA's multi-stream execution to ROCm to enable concurrent batch processing.

## Current State (ROCm)
- Single default stream (stream 0)
- All batches serialized with mutex lock
- Shared memory for all batches
- Performance limited by sequential execution

## Target State (Like CUDA)
- Per-batch HIP streams
- Concurrent batch execution without locking
- Per-stream memory allocation
- Expected: +10-20% throughput

## Implementation Steps

### 1. Add Backend Option Parsing
- Line ~255: Add `multi_stream_ = options.GetOrDefault<bool>("multi_stream", false);`

### 2. Stream/Event Creation (Constructor)
When `multi_stream_ = false`:
- Create compute_stream_, upload_stream_, download_stream_
- Create synchronization events
- Create rocBLAS handle with compute stream

### 3. Memory Allocation Changes
When `multi_stream_ = true`:
- Don't pre-allocate shared tensor_mem_
- Each InputsOutputs will allocate its own memory
- Set tensor_mem_size_ = maxSize (for per-batch allocation)

### 4. InputsOutputs Structure  (if needed)
- May need to create ROCm equivalent with per-stream resources
- Include: streams, events, rocBLAS handle, memory

### 5. GetNetwork() / LockEval() Changes
- When multi_stream_=true: return empty lock (no synchronization needed)
- When multi_stream_=false: return mutex lock (serialize batches)

### 6. Computation Path Updates
- Use per-stream resources when multi_stream_=true
- Use shared resources when multi_stream_=false

## Files to Modify
- `src/neural/backends/rocm/network_rocm.cc` (main implementation)

## HIP API Equivalents
```cpp
CUDA                      →  HIP
cudaStream_t              →  hipStream_t
cudaEvent_t               →  hipEvent_t
cudaStreamCreate          →  hipStreamCreate
cudaEventCreate           →  hipEventCreate
cudaStreamSynchronize     →  hipStreamSynchronize
cudaEventRecord           →  hipEventRecord
cudaStreamWaitEvent       →  hipStreamWaitEvent
cublasHandle_t            →  rocblas_handle
cublasCreate              →  rocblas_create_handle
cublasSetStream           →  rocblas_set_stream
```

## Testing
1. Test with `--backend-opts="multi_stream=false"` (default, should work as before)
2. Test with `--backend-opts="multi_stream=true"` (new concurrent mode)
3. Verify correctness with both modes
4. Benchmark performance improvement

## Expected Results
- Baseline (multi_stream=false): 2,358 nps
- Target (multi_stream=true): 2,590-2,830 nps (+10-20%)
- Total vs rocBLAS: +30-40%
