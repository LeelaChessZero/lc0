# Flash Attention for lc0 - Quick Start

## For Your GPU (Radeon 8060S / RDNA 3.5)

**Build command** (already optimized, just use it):
```bash
./build.sh release -Drocm=true -Damd_gfx=gfx1151
```

**Run**:
```bash
./build/release/lc0 --weights=<model>.pb.gz --backend=rocm-fp16
```

**Performance**: 2,355 nps (~18% faster than rocBLAS)

That's it! ✓

---

## For Other GPUs

### 1. Check if your GPU is supported
```bash
python3 scripts/rocm_backend_amd_gfx_configs.py list
```

Look for your GPU in the list:
- `[✓]` = Optimized, ready to use
- `[ ]` = Estimated config, should tune

### 2. Build for your GPU
```bash
./build.sh release -Drocm=true -Damd_gfx=<your_gfx_code>

# Examples:
# RX 7900 XTX:  -Damd_gfx=gfx1100
# MI300:        -Damd_gfx=gfx942
```

### 3. If you see "⚠ Using estimated configuration"
Run the tuner to find optimal settings:
```bash
./scripts/tune_rocm_backend.sh           # Full tuning (~30 min) - DEFAULT
./scripts/tune_rocm_backend.sh --quick   # Quick tuning (~10 min)
```

The script will:
- Test different configurations
- Show you the best one
- Tell you exactly how to use it

---

## Need Help?

### Common Issues

**Error: "The -Darch flag requires ROCm backend"**
→ Add `-Drocm=true` to your build command

**Flash attention not being used**
→ Make sure you use `--backend=rocm-fp16` (not `rocm`)

**Performance worse than expected**
→ Run `./scripts/tune_rocm_backend.sh` to find optimal settings for your GPU

### Documentation

- **ARCH_GUIDE.md** - Complete architecture configuration guide
- **FLASH_ATTENTION_INTEGRATION.md** - Technical implementation details
- **FLASH_ATTENTION_TUNING.md** - In-depth tuning guide
- **OPTIMIZATION_SUMMARY.md** - Performance optimization results

### Find Your GPU Architecture

```bash
rocminfo | grep "Name:" | grep gfx
```

---

## Summary

| What You Want | Command |
|---------------|---------|
| Build for gfx1151 (optimized) | `./build.sh release -Drocm=true -Damd_gfx=gfx1151` |
| Build for other GPU | `./build.sh release -Drocm=true -Damd_gfx=<gfx_code>` |
| List supported GPUs | `python3 scripts/rocm_backend_amd_gfx_configs.py list` |
| Find my GPU | `rocminfo \| grep gfx` |
| Tune for my GPU (full) | `./scripts/tune_rocm_backend.sh` |
| Quick tuning | `./scripts/tune_rocm_backend.sh --quick` |
| Get help | `./scripts/tune_rocm_backend.sh --help` |

**Key point**: Always use `-Drocm=true -Damd_gfx=<gfx_code>` together.
