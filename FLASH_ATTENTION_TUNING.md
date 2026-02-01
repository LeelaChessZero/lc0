# Flash Attention Performance Tuning Guide

## Overview

Flash attention performance depends on several kernel parameters that control parallelism, memory access patterns, and register usage. This guide explains how to tune these parameters for optimal performance on your GPU.

## Tunable Parameters

All parameters can be overridden at compile time using `-D` flags. Default values are optimized for RDNA3 with depth=32 (24-head T82 configuration).

### 1. `FATTN_NTHREADS_D32` (default: 128)
**What it controls**: Number of threads per HIP block

**Impact**:
- **Higher (256, 512)**: More parallelism, but higher register pressure and lower occupancy
- **Lower (64, 32)**: Less parallelism, better occupancy, but may underutilize GPU

**Tuning advice**:
- RDNA3 has 64 threads per SIMD, so multiples of 64 work well
- Try 64, 128, 256
- Monitor with: Check if kernel launches successfully

### 2. `FATTN_OCCUPANCY_D32` (default: 2)
**What it controls**: Target number of concurrent wave groups per compute unit

**Impact**:
- **Higher (4, 8)**: More concurrent work, better latency hiding, but more register/LDS pressure
- **Lower (1)**: Fewer concurrent warps, more resources per warp, less latency hiding

**Tuning advice**:
- RDNA3 benefits from 2-4 occupancy typically
- Try 1, 2, 4, 8
- Higher values may cause kernel launch failure if resource requirements exceed limits

### 3. `FATTN_NBATCH_FA_D32` (default: 64)
**What it controls**: Number of rows processed per softmax rescaling iteration

**Impact**:
- **Higher (128)**: Fewer softmax rescaling operations, less overhead, but more register usage
- **Lower (32)**: More frequent rescaling, lower register pressure, but more computation overhead

**Tuning advice**:
- Should be ≤ sequence length (64 for T82)
- Try 32, 64, 128
- Related to online softmax algorithm efficiency

### 4. `FATTN_NBATCH_K2_D32` / `FATTN_NBATCH_V2_D32` / `FATTN_NBATCH_COMBINE_D32` (default: 32)
**What they control**: Tile sizes for parallel loading and combining operations

**Impact**:
- **Larger (64)**: Bigger tiles, potentially better memory bandwidth utilization, more shared memory
- **Smaller (16)**: Smaller tiles, better cache reuse for small depth=32, less shared memory pressure

**Tuning advice**:
- For depth=32, smaller tiles (16) may perform better than default (32)
- All three should typically be equal
- Try 16, 32, 64
- Limited by depth: cannot exceed depth value

### 5. `FATTN_NSTAGES_D32` (default: 2)
**What it controls**: Number of pipeline stages for overlapping computation and memory access

**Impact**:
- **Higher (4, 8)**: More overlap between memory loads and computation, but higher register usage
- **Lower (1)**: No pipelining, simpler kernel, less register pressure

**Tuning advice**:
- RDNA3 benefits from 2-4 stages typically
- Try 1, 2, 4
- Higher values risk register spilling

## Performance Bottlenecks

Flash attention can be limited by:

1. **Memory Bandwidth** (most common)
   - Symptom: Low GPU utilization, performance scales with batch size
   - Solution: Optimize tile sizes, increase occupancy for better latency hiding
   - Target: >70% of peak memory bandwidth

2. **Register Pressure**
   - Symptom: Low occupancy, kernel launch failures
   - Solution: Reduce nthreads, nstages, or nbatch_fa
   - Target: Kernel launches successfully with occupancy ≥ 2

3. **Shared Memory**
   - Symptom: Kernel launch failure, reduced occupancy
   - Solution: Reduce tile sizes (nbatch_K2, V2, combine)
   - RDNA3 has 64KB LDS per CU

4. **Compute Bound**
   - Symptom: High GPU utilization, doesn't scale with batch size
   - Solution: Increase parallelism (nthreads, occupancy)
   - Less common for attention kernels

## Quick Tuning Workflow

### Step 1: Quick Test (5 configs, ~10 minutes)
```bash
./scripts/tune_rocm_backend.sh
```

Tests:
- Baseline (current settings)
- High occupancy (occ=4)
- More threads (256)
- Small tiles (16×16)
- Aggressive combo

### Step 2: Comprehensive Tuning (~30 minutes)
```bash
./scripts/tune_rocm_backend.sh
```

Tests 15+ configurations systematically:
- Thread counts: 64, 128, 256
- Occupancy: 1, 2, 4, 8
- Tile sizes: 16, 32, 64
- Pipeline stages: 1, 2, 4
- Combinations

### Step 3: Manual Fine-Tuning

If you find a promising configuration, fine-tune around it:
```bash
./build.sh release -Dcpp_args="-DUSE_FLASH_ATTENTION=1 -DRDNA3 \
  -DFATTN_NTHREADS_D32=256 \
  -DFATTN_OCCUPANCY_D32=4 \
  -DFATTN_NBATCH_FA_D32=64 \
  -DFATTN_NBATCH_K2_D32=16 \
  -DFATTN_NBATCH_V2_D32=16 \
  -DFATTN_NBATCH_COMBINE_D32=16 \
  -DFATTN_NSTAGES_D32=2"

./build/release/lc0 backendbench \
  --weights=../models/768x15x24h-t82-swa-7464000.pb.gz \
  --backend=rocm-fp16 \
  --batches=30 \
  --start-batch-size=64 \
  --max-batch-size=64
```

## Expected Performance Ranges

Based on RDNA 3.5 (8060S) with depth=32:

- **Baseline**: ~2,300 nps (current)
- **Well-tuned**: 2,400-2,600 nps (5-15% improvement)
- **Optimal**: 2,600-2,800 nps (15-20% improvement possible)

Performance gains beyond 20% unlikely without algorithmic changes.

## Recommended Starting Points by GPU

### RDNA3 (RX 7900 XTX, 7800 XT)
```
nthreads=128, occupancy=4, nbatch_fa=64
tiles=32, nstages=2
```

### RDNA 3.5 (RX 8060S, Steam Deck OLED)
```
nthreads=128, occupancy=2, nbatch_fa=64
tiles=16, nstages=2
```
(Smaller tiles work better on 3.5 due to cache architecture)

### CDNA3 (MI300)
```
nthreads=256, occupancy=4, nbatch_fa=128
tiles=64, nstages=4
```
(More resources available, benefits from larger configs)

## Troubleshooting

### Kernel Launch Failure
**Error**: Flash attention falls back to rocBLAS immediately

**Causes**:
- Insufficient shared memory (reduce tile sizes)
- Insufficient registers (reduce nthreads or nstages)
- Invalid configuration (e.g., tiles > depth)

**Solutions**:
- Reduce nbatch_K2, nbatch_V2, nbatch_combine to 16
- Reduce nthreads to 64 or 128
- Reduce nstages to 1 or 2
- Check build log for HIP errors

### Performance Regression
**Symptom**: New config slower than baseline

**Common causes**:
- Too high occupancy causing resource contention
- Tile size mismatch with depth=32
- Register spilling from too many stages

**Debug**:
1. Return to baseline config
2. Change one parameter at a time
3. Verify kernel actually runs (check for "Successfully using fused attention")

### Build Failures
**Error**: Compilation errors with tuning parameters

**Solutions**:
- Ensure all parameters are integers
- Check parameter values are reasonable (nthreads > 0, tiles ≤ depth, etc.)
- Verify quotes in `-Dcpp_args="..."` are correct

## Advanced: Multi-Depth Tuning

If you use multiple models with different depths (32, 64, 96, 128), you can define separate parameters:

```bash
# Not yet implemented - would require additional work
-DFATTN_NTHREADS_D64=128 -DFATTN_OCCUPANCY_D64=2 ...
-DFATTN_NTHREADS_D96=128 -DFATTN_OCCUPANCY_D96=2 ...
```

Currently only depth=32 is tunable. Other depths use hardcoded defaults.

## Results Interpretation

From tuning script output:

```
Config              nthreads  occupancy  nbatch_fa  ...  Mean_NPS  Max_NPS  Median_NPS
baseline            128       2          64         ...  2322      2434     2306
high_occupancy      128       4          64         ...  2456      2512     2448
small_tiles         128       2          64         ...  2401      2489     2395
```

**Look for**:
- Highest mean NPS (average performance)
- Low standard deviation (consistent performance)
- Max NPS close to mean (stable, no outliers)

**Best config**: Highest mean NPS with stddev < 5% of mean

## Next Steps

After finding optimal config:
1. Update default values in `flash_attention.hip` (lines 43-69)
2. Document the optimal config in FLASH_ATTENTION_INTEGRATION.md
3. Consider submitting findings to lc0 project if significant improvement

## References

- llama.cpp flash attention tuning: https://github.com/ggerganov/llama.cpp/discussions/5519
- RDNA3 optimization guide: AMD ROCm documentation
- Flash attention paper: https://arxiv.org/abs/2205.14135
