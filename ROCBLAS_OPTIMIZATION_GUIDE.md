# rocBLAS Optimization Guide - Where The Real Bottleneck Is

## TL;DR

**Flash attention: 1.5% of GPU time** ✓ Already optimized
**rocBLAS GEMM: 77.6% of GPU time** ← **Optimize here for real gains!**

## What Operations Use rocBLAS?

Based on profiling data, here's what each **encoder block** does:

### Per Encoder Block (15 total in T82):

```
1. Q/K/V Projections (3 separate matrix multiplies)
   ├─ Q = Input × W_q  (768×768 matrix)
   ├─ K = Input × W_k  (768×768 matrix)
   └─ V = Input × W_v  (768×768 matrix)

2. Flash Attention (already optimized)
   └─ Output = softmax(Q·K^T / √d) · V

3. Attention Output Projection
   └─ Output = Attention × W_dense (768×768 matrix)

4. Feed-Forward Network (FFN) - THE BOTTLENECK
   ├─ FFN1 = LayerNorm(Output) × W_ffn1  (768×2304 matrix)
   └─ FFN2 = Activation(FFN1) × W_ffn2   (2304×768 matrix)

5. Layer Normalization (x2)
   └─ Cheap operations, ~7% of time
```

### Time Breakdown (From Profiling):

```
Operation                | Time per batch | Total (15 encoders) | % of GPU time
-------------------------|----------------|---------------------|---------------
FFN Matrix Multiplies    | ~1.6 ms        | ~24 ms              | 60%
Q/K/V Projections        | ~0.5 ms        | ~7.5 ms             | 15%
Attention Output Proj    | ~0.1 ms        | ~1.5 ms             | 2.5%
Flash Attention          | ~0.03 ms       | ~0.5 ms             | 1.5%
Layer Norm               | ~0.15 ms       | ~2.3 ms             | 7%
Other operations         | ~0.2 ms        | ~3 ms               | 13%
```

**Key insight:** FFN layers take **40× longer** than flash attention!

## Where Is rocBLAS Called?

### 1. Q/K/V Projection (lines 1665-1693 in layers.cc)

**Current implementation:** 3 separate `cublasXGemmStridedBatched` calls

```cpp
// Compute Q = Input × W_q
cublasXGemmStridedBatched(cublas, ..., mha_q_w, ..., in_out_tensor, ..., mha_q);

// Compute K = Input × W_k
cublasXGemmStridedBatched(cublas, ..., mha_k_w, ..., in_out_tensor, ..., mha_k);

// Compute V = Input × W_v
cublasXGemmStridedBatched(cublas, ..., mha_v_w, ..., in_out_tensor, ..., mha_v);
```

**Dimensions:**
- Input: [batch × 64, 768] (64 positions, 768 features)
- W_q, W_k, W_v: [768, 768] each
- Output: Q, K, V each [batch × 64, 768]

**Performance:**
- ~0.5 ms per encoder (×15 = 7.5 ms total)
- 15% of GPU time

### 2. Attention Output Projection (lines 1875-1886)

**Current implementation:** Single `cublasXgemm` call

```cpp
// Compute Output = Attention × W_dense
cublasXgemm(cublas, ..., mha_dense_w, ..., buffer2, ..., in_out_tensor);
```

**Dimensions:**
- Attention output: [batch × 64, 768]
- W_dense: [768, 768]
- Output: [batch × 64, 768]

**Performance:**
- ~0.1 ms per encoder (×15 = 1.5 ms total)
- 2.5% of GPU time

### 3. FFN Dense Layer 1 (lines 1912-1922)

**Current implementation:** Single `cublasXgemm` call

```cpp
// Compute FFN1 = Input × W_ffn1
cublasXgemm(cublas, ..., ffn_dense1_w, ..., scratch, ..., in_out_tensor);
addBiasBatched(...);  // Add bias and activation
```

**Dimensions:**
- Input: [batch × 64, 768]
- W_ffn1: [768, 2304] ← **3× expansion**
- Output: [batch × 64, 2304]

**Performance:**
- ~0.8 ms per encoder (×15 = 12 ms total)
- 30% of GPU time ← **MAJOR BOTTLENECK**

### 4. FFN Dense Layer 2 (lines 1925-1938)

**Current implementation:** Single `cublasXgemm` call

```cpp
// Compute FFN2 = FFN1 × W_ffn2
cublasXgemm(cublas, ..., ffn_dense2_w, ..., in_out_tensor, ..., buffer1);
addBiasBatched(...);  // Add bias
```

**Dimensions:**
- Input: [batch × 64, 2304]
- W_ffn2: [2304, 768] ← **Projection back**
- Output: [batch × 64, 768]

**Performance:**
- ~0.8 ms per encoder (×15 = 12 ms total)
- 30% of GPU time ← **MAJOR BOTTLENECK**

## Can We Fuse These Like Flash Attention?

### Q/K/V Projection Fusion - YES! ✓

**Opportunity:** Fuse 3 separate matrix multiplies into one

**Current:**
```
Q = Input × W_q    (768×768)
K = Input × W_k    (768×768)
V = Input × W_v    (768×768)
```

**Optimized:**
```
[Q|K|V] = Input × [W_q|W_k|W_v]  (768×2304)
```

**Already implemented!** See lines 1449-1473:
```cpp
// Big allocation to hold qkv weights one after the other
size_t elements = cpu_weights.mha.q_w.size();
size_t size = elements * sizeof(DataType) * 3;
ReportHIPErrors(hipMalloc(&mha_qkv_w, size));
ReportHIPErrors(hipMemcpy(mha_qkv_w, mha_q_w, size / 3, hipMemcpyDeviceToDevice));
ReportHIPErrors(hipMemcpy(mha_qkv_w + elements, mha_k_w, size / 3, ...));
ReportHIPErrors(hipMemcpy(mha_qkv_w + elements * 2, mha_v_w, size / 3, ...));
```

**But it's not being used!** The code still calls 3 separate GEMMs.

**Expected gain:** 2× speedup on Q/K/V projections (15% × 0.5 = **~7.5% overall**)

### FFN Fusion - PARTIAL ✓

**Opportunity:** Fuse matrix multiply + bias + activation

**Current:**
```
1. FFN1 = Input × W_ffn1        (rocBLAS GEMM)
2. FFN1 = FFN1 + bias           (custom kernel)
3. FFN1 = activation(FFN1)      (custom kernel)
```

**Better:**
```
FFN1 = activation(Input × W_ffn1 + bias)   (single fused kernel)
```

**Challenge:** rocBLAS doesn't support built-in activation functions

**Solutions:**
1. **hipBLASLt** - Supports epilogue fusion (GEMM + bias + activation in one call)
2. **Custom WMMA kernel** - Full control, but complex
3. **rocWMMA library** - AMD's WMMA wrapper

**Expected gain:** 10-20% speedup on FFN layers (60% × 0.15 = **~9% overall**)

### Full Encoder Block Fusion - HARD ❌

**Idea:** Fuse entire encoder block into one mega-kernel

**Why it's hard:**
1. Layer normalization requires global reductions (all threads must sync)
2. Attention mechanism has data dependencies (Q·K^T → softmax → ·V)
3. Residual connections require copying/accumulating intermediate results
4. Memory footprint would be huge

**Verdict:** Not practical. Focus on individual operation fusion instead.

## Optimization Strategies (Ordered by ROI)

### 1. Use hipBLASLt Instead of rocBLAS (HIGH ROI)

**What it is:** AMD's optimized GEMM library with epilogue fusion

**Benefits:**
- GEMM + bias + activation in single kernel call
- Better RDNA 3 optimizations
- Reduced kernel launch overhead

**Where to apply:**
- FFN dense1: `GEMM(W_ffn1, Input) + bias + swish` → 1 call instead of 3
- FFN dense2: `GEMM(W_ffn2, FFN1) + bias` → 1 call instead of 2
- Attention output: `GEMM(W_dense, Attention) + bias` → 1 call instead of 2

**Expected gain:** 10-20% overall performance

**Effort:** Medium (API is similar to rocBLAS but requires learning epilogue syntax)

**Example API:**
```cpp
hipblasLtMatmul(handle, matmul_desc,
                &alpha, A, A_desc, B, B_desc,
                &beta, C, C_desc, C, C_desc,
                &epilogue,  // ← Fusion magic happens here
                stream);
```

### 2. Fuse Q/K/V Projections (MEDIUM ROI)

**What to do:** Use the already-concatenated `mha_qkv_w` weights

**Current code:**
```cpp
// lines 1665-1693 - 3 separate calls
cublasXGemmStridedBatched(..., mha_q_w, ..., mha_q);
cublasXGemmStridedBatched(..., mha_k_w, ..., mha_k);
cublasXGemmStridedBatched(..., mha_v_w, ..., mha_v);
```

**Optimized code:**
```cpp
// Single call with concatenated weights
cublasXGemmStridedBatched(
    cublas, rocblas_operation_transpose, rocblas_operation_none,
    768 * 3,  // Output: 2304 (Q|K|V concatenated)
    N * 64,   // Batch × positions
    768,      // Input features
    1.0f, mha_qkv_w, 768, 768 * 768,
    in_out_tensor, 768, 0,
    0.0f, qkv_output, 768 * 3, 768 * 3 * N * 64,
    1);

// Then split Q, K, V with memory copies (much faster than 2 extra GEMMs)
```

**Expected gain:** ~7% overall (2× speedup on 15% of time)

**Effort:** Low (weights already concatenated, just need to change GEMM call)

### 3. Increase Batch Size (LOW ROI but easy)

**Current:** Batch size varies (1-96 positions)

**Observation:** rocBLAS GEMM efficiency increases with batch size

**Strategy:**
- Pad smaller batches to next power-of-2
- Or, use multi-stream to process multiple independent batches concurrently

**Expected gain:** 5-10% on small batches

**Effort:** Low (padding) to Medium (multi-stream already implemented)

### 4. Custom WMMA Kernels (HIGH EFFORT, UNCERTAIN ROI)

**What it is:** Write hand-tuned matrix multiply kernels using AMD's wave matrix intrinsics

**Benefits:**
- Complete control over fusion
- Can optimize for specific sizes (768, 2304)
- Potentially better than hipBLASLt for fixed sizes

**Drawbacks:**
- Very high development cost
- Need to tune for each RDNA generation
- rocBLAS/hipBLASLt already highly optimized

**Previous attempt:** See `src/neural/backends/rocm/STRIDED_WMMA_ATTEMPT.md`
- Custom WMMA kernels showed 10% **regression** due to strided access patterns
- rocBLAS handles interleaved layout better

**Verdict:** Not recommended. Use hipBLASLt instead.

### 5. Quantization to INT8 (HIGHEST POTENTIAL, HIGHEST EFFORT)

**What it is:** Convert FP16 weights/activations to INT8

**Benefits:**
- 2× less memory bandwidth
- 2× less memory footprint
- Potentially 2× faster compute (INT8 ops are faster)

**Drawbacks:**
- Accuracy degradation (requires validation)
- Need calibration data
- Not all operations benefit equally

**Expected gain:** 30-50% overall performance

**Effort:** Very high (requires model retraining/calibration)

## Recommended Optimization Order

### Phase 1: Low-Hanging Fruit (1-2 weeks)

1. **Fuse Q/K/V projections** using existing `mha_qkv_w`
   - Expected: +7% overall
   - Effort: Low
   - Risk: Very low

2. **Increase minimum batch size** to 64 (already done for FP16)
   - Expected: +5% on small batches
   - Effort: Trivial
   - Risk: None

**Total Phase 1:** +12% improvement

### Phase 2: hipBLASLt Integration (2-4 weeks)

1. **Replace rocBLAS with hipBLASLt** for FFN layers
   - FFN dense1 + bias + swish fusion
   - FFN dense2 + bias fusion
   - Expected: +10-15% overall
   - Effort: Medium
   - Risk: Medium (API learning curve)

2. **Optimize epilogue configurations** for RDNA 3
   - Tune tile sizes
   - Test different algorithms
   - Expected: +2-5% additional
   - Effort: Medium
   - Risk: Low

**Total Phase 2:** +12-20% improvement

### Phase 3: Advanced (if needed, 4-8 weeks)

1. **INT8 quantization**
   - Model calibration
   - Accuracy validation
   - Expected: +30-50%
   - Effort: Very high
   - Risk: High (accuracy loss)

**Total Phase 1+2:** +24-32% improvement without quantization
**Total with Phase 3:** +54-82% improvement with quantization

## Why Not Write Custom Kernels?

**Reason 1: rocBLAS is already highly optimized**
- Years of AMD engineering effort
- Tuned for every RDNA generation
- Automatic kernel selection for different sizes

**Reason 2: Maintenance burden**
- Custom kernels need per-GPU tuning
- Break with new architectures
- Hard to debug

**Reason 3: Fusion is available through hipBLASLt**
- Get fusion benefits without custom kernels
- Maintained by AMD
- Performance comparable to hand-tuned code

**Exception:** Flash attention needed custom kernel because:
- Online softmax algorithm not in any library
- Specific memory access pattern (interleaved multi-head)
- Algorithm-level fusion, not just operation fusion

## Code Locations

**Key files to modify:**

1. **`src/neural/backends/rocm/layers.cc`**
   - Line 1665-1693: Q/K/V projections (fuse these)
   - Line 1912-1922: FFN dense1 (add hipBLASLt)
   - Line 1925-1938: FFN dense2 (add hipBLASLt)

2. **`src/neural/backends/rocm/layers.h`**
   - Add hipBLASLt handle creation
   - Add epilogue descriptor structs

3. **`meson.build`**
   - Add hipBLASLt library dependency

## Performance Ceiling

**Current performance:** 2,333 nps @ batch=64

**Theoretical improvements:**
- Q/K/V fusion: +7% → 2,496 nps
- hipBLASLt FFN: +15% → 2,870 nps
- Both combined: +24% → 2,893 nps

**Hardware ceiling (memory-bound):**
- Radeon 8060S: ~3,500 nps (55% efficiency)
- Titan RTX: ~4,500 nps (best case)

**Conclusion:** Another 24-32% improvement is achievable through rocBLAS optimizations, bringing 8060S performance to ~2,900 nps (72% of Titan RTX performance with only 32% of memory bandwidth).

## Summary

**Flash attention (1.5% of time):** Already optimized ✓

**rocBLAS GEMM (77.6% of time):** Where real gains are ← Focus here

**Next steps:**
1. Fuse Q/K/V projections (easy win)
2. Integrate hipBLASLt for FFN layers (bigger win)
3. Consider quantization if more performance needed (biggest win but hardest)

**Expected total gain:** 24-32% without quantization, 54-82% with quantization
