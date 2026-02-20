# Check 0 Recipe: PJRT MLIR Compile Validation

**Status: PASSED** (2026-01-24)

## Environment Assumptions

- Running inside WSL2 Linux
- JAX CUDA PJRT plugin installed in the active Python env
- NVIDIA GPU available (tested on RTX 2060)

## What Check 0 Proves

1. PJRT plugin accepts `format="mlir"` for StableHLO programs
2. StableHLO bytecode v1.0.0 is compatible with the CUDA plugin
3. The `CompileProgram()` refactor works correctly
4. The entire toolchain (stablehlo-translate -> PJRT compile) is functional

## Plugin Info

```
Plugin StableHLO version range:
  stablehlo_minimum_version = 0, 9, 0   (== 0.9.0)
  stablehlo_current_version = 1, 13, 7  (== 1.13.7)

Bytecode target version: 1.0.0 (within range)
GPU: NVIDIA GeForce RTX 2060
```

## Paths (WSL2)

```bash
# PJRT CUDA Plugin (from JAX)
PLUGIN=/home/n4k3dw4ff13s/lc0_env/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so

# stablehlo-translate (built from v1.0.0)
STABLEHLO_TRANSLATE=/home/n4k3dw4ff13s/stablehlo/build/bin/stablehlo-translate

# Lc0 source (Windows, accessible from WSL2)
LC0=/mnt/c/Users/iftik/Downloads/xla-bytcode/lc0-source
```

## Test Files

| File | Purpose |
|------|---------|
| `tests/stablehlo/add.mlir` | Minimal StableHLO text module |
| `tests/stablehlo/add.mlirbc` | Portable artifact (bytecode) targeting v1.0.0 |

## Create Test Files (Self-Contained)

```bash
# 1. Ensure directory exists (prevents "No such file" on output)
mkdir -p tests/stablehlo

# 2. Create the source file (prevents input error)
cat > tests/stablehlo/add.mlir << 'EOF'
module {
  func.func @main(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = stablehlo.add %a, %b : tensor<2x3xf32>
    func.return %0 : tensor<2x3xf32>
  }
}
EOF
```

## Regenerate Bytecode

```bash
STABLEHLO_TRANSLATE=/home/n4k3dw4ff13s/stablehlo/build/bin/stablehlo-translate

# Serialize
$STABLEHLO_TRANSLATE --serialize tests/stablehlo/add.mlir --target=1.0.0 > tests/stablehlo/add.mlirbc

# Sanity check (should print nothing on success)
$STABLEHLO_TRANSLATE --deserialize tests/stablehlo/add.mlirbc > /dev/null
```

## Build Smoketest

```bash
cd /mnt/c/Users/iftik/Downloads/xla-bytcode/lc0-source
meson setup build --wipe -Dxla=true -Dpjrt_smoketest=true
ninja -C build pjrt_mlir_smoketest

# Sanity check
test -x ./build/pjrt_mlir_smoketest && echo "built OK"
```

## Run Smoketest

```bash
PLUGIN=/home/n4k3dw4ff13s/lc0_env/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so

# Text MLIR
./build/pjrt_mlir_smoketest "$PLUGIN" tests/stablehlo/add.mlir

# Bytecode MLIR
./build/pjrt_mlir_smoketest "$PLUGIN" tests/stablehlo/add.mlirbc
```

## Expected Output

```
=== PJRT MLIR Smoketest (Check 0b) ===
[1/5] Loading PJRT plugin...
      OK
[2/5] Plugin attributes:
      stablehlo_current_version = 1, 13, 7
      stablehlo_minimum_version = 0, 9, 0
      ...
[3/5] Creating PJRT client...
      OK
[4/5] Getting first addressable device...
      Using device id: 0
[5/5] Compiling MLIR program...
      OK - Compiled successfully

=== SUCCESS ===
PJRT plugin compiled the MLIR program.
Executable outputs: 1
```

## Troubleshooting

### CRLF Line Ending Issue

If you see:
```
/usr/bin/env: 'python3\r': No such file or directory
```

Fix with:
```bash
sed -i 's/\r$//' scripts/compile_proto.py
chmod +x scripts/compile_proto.py
```

### RAM/OOM Crash During LLVM Build

If your WSL2 window closes suddenly during the stablehlo/LLVM build, you ran out of RAM.

Resume with limited parallelism:
```bash
cmake --build llvm-build --target all -j 2
```

## Code Changes Made for Check 0

| File | Change |
|------|--------|
| `src/neural/backends/xla/pjrt.h` | Added `CompileProgram()`, `FirstAddressableDeviceId()` |
| `src/neural/backends/xla/pjrt.cc` | Implemented new methods, refactored compile path |
| `src/tools/pjrt_mlir_smoketest.cc` | New smoketest tool |
| `meson_options.txt` | Added `pjrt_smoketest` option |
| `meson.build` | Added smoketest build target |

## Next Step

Check 4: Lc0 parameter ordering and op inventory
