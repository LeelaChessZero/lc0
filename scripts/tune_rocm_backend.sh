#!/bin/bash
# Flash Attention Performance Tuner for ROCm
# Finds optimal configuration for your AMD GPU

set -e

WEIGHTS="../models/768x15x24h-t82-swa-7464000.pb.gz"
BATCH_SIZE=64

# Parse arguments
MODE="full"
if [[ "$1" == "--quick" ]]; then
    MODE="quick"
    BATCHES=15
    echo "=== Quick Tuning Mode ==="
    echo "This will test 5 configurations (~10 minutes)"
    echo ""
    echo "For best results, use: ./scripts/tune_rocm_backend.sh (full mode)"
elif [[ "$1" == "--help" || "$1" == "-h" ]]; then
    cat << 'EOF'
Flash Attention Performance Tuner for ROCm

USAGE:
    ./scripts/tune_rocm_backend.sh           Full tune (15+ configs, ~30 min) - DEFAULT
    ./scripts/tune_rocm_backend.sh --quick   Quick tune (5 configs, ~10 min)
    ./scripts/tune_rocm_backend.sh --help    Show this help

WHAT IT DOES:
    Tests different kernel parameters (threads, occupancy, tile sizes)
    to find the optimal configuration for your AMD GPU.

WHEN TO USE:
    - Your GPU architecture shows "⚠ Using estimated configuration"
    - You want to verify/improve the current tuned settings
    - You have a new GPU not in the database

AFTER TUNING:
    The script will show you the best configuration and how to use it.
    You can either:
    1. Use it temporarily with -Dcpp_args flags
    2. Update scripts/rocm_backend_amd_gfx_configs.py to make it permanent

MODES:
    Full mode (default): Tests 15+ configurations for best results
    Quick mode (--quick): Tests 5 configurations for faster results

EOF
    exit 0
else
    BATCHES=20
    echo "=== Full Tuning Mode ==="
    echo "This will test 15+ configurations (~30 minutes)"
    echo ""
    echo "For faster results, use: ./scripts/tune_rocm_backend.sh --quick"
fi

echo "Model: $WEIGHTS"
echo "Batch size: $BATCH_SIZE"
echo "Batches per test: $BATCHES"
echo ""

# Detect current GPU architecture
CURRENT_ARCH=$(rocminfo 2>/dev/null | grep "Name:" | grep gfx | head -1 | awk '{print $2}' || echo "unknown")
if [[ "$CURRENT_ARCH" != "unknown" ]]; then
    ARCH_NAME=$(python3 scripts/rocm_backend_amd_gfx_configs.py name "$CURRENT_ARCH" 2>/dev/null || echo "Unknown GPU")
    echo "Detected GPU: $ARCH_NAME ($CURRENT_ARCH)"
else
    echo "Warning: Could not detect GPU architecture"
fi
echo ""

# Results file
RESULTS_FILE="/tmp/flash_attention_tune_results.csv"
echo "Config,nthreads,occupancy,nbatch_fa,nbatch_K2,nbatch_V2,nbatch_combine,nstages,Mean_NPS,Max_NPS,Median_NPS,StdDev" > $RESULTS_FILE

# Function to run benchmark with specific configuration
run_benchmark() {
    local name=$1
    local nthreads=$2
    local occupancy=$3
    local nbatch_fa=$4
    local nbatch_K2=$5
    local nbatch_V2=$6
    local nbatch_combine=$7
    local nstages=$8

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    printf "Testing: %-25s" "$name"

    # Build with specific configuration
    local cpp_args="-DUSE_FLASH_ATTENTION=1 -DRDNA3"
    cpp_args="$cpp_args -DFATTN_NTHREADS_D32=$nthreads"
    cpp_args="$cpp_args -DFATTN_OCCUPANCY_D32=$occupancy"
    cpp_args="$cpp_args -DFATTN_NBATCH_FA_D32=$nbatch_fa"
    cpp_args="$cpp_args -DFATTN_NBATCH_K2_D32=$nbatch_K2"
    cpp_args="$cpp_args -DFATTN_NBATCH_V2_D32=$nbatch_V2"
    cpp_args="$cpp_args -DFATTN_NBATCH_COMBINE_D32=$nbatch_combine"
    cpp_args="$cpp_args -DFATTN_NSTAGES_D32=$nstages"

    ./build.sh release -Drocm=true -Dcpp_args="$cpp_args" > /tmp/build_${name}.log 2>&1
    if [ $? -ne 0 ]; then
        echo " ❌ BUILD FAILED"
        echo "$name,$nthreads,$occupancy,$nbatch_fa,$nbatch_K2,$nbatch_V2,$nbatch_combine,$nstages,FAILED,FAILED,FAILED,FAILED" >> $RESULTS_FILE
        return
    fi

    # Run benchmark
    output=$(./build/release/lc0 backendbench \
        --weights=$WEIGHTS \
        --backend=rocm-fp16 \
        --batches=$BATCHES \
        --start-batch-size=$BATCH_SIZE \
        --max-batch-size=$BATCH_SIZE 2>&1)

    # Extract performance metrics
    perf_line=$(echo "$output" | grep "^  $BATCH_SIZE," | head -1)

    if [ -z "$perf_line" ]; then
        echo " ❌ NO DATA"
        echo "$name,$nthreads,$occupancy,$nbatch_fa,$nbatch_K2,$nbatch_V2,$nbatch_combine,$nstages,NO_DATA,NO_DATA,NO_DATA,NO_DATA" >> $RESULTS_FILE
        return
    fi

    # Parse results
    mean_nps=$(echo $perf_line | awk -F',' '{print $2}' | tr -d ' ')
    max_nps=$(echo $perf_line | awk -F',' '{print $6}' | tr -d ' ')
    median_nps=$(echo $perf_line | awk -F',' '{print $7}' | tr -d ' ')
    sdev=$(echo $perf_line | awk -F',' '{print $4}' | tr -d ' ')

    printf " ✓ %s nps\n" "$mean_nps"
    echo "$name,$nthreads,$occupancy,$nbatch_fa,$nbatch_K2,$nbatch_V2,$nbatch_combine,$nstages,$mean_nps,$max_nps,$median_nps,$sdev" >> $RESULTS_FILE
}

# Configuration tests
if [[ "$MODE" == "quick" ]]; then
    # Quick mode: 5 most promising configs
    run_benchmark "baseline"            128 2 64 32 32 32 2
    run_benchmark "more_threads"        256 2 64 32 32 32 2
    run_benchmark "high_occupancy"      128 4 64 32 32 32 2
    run_benchmark "small_tiles"         128 2 64 16 16 16 2
    run_benchmark "aggressive"          256 4 64 32 32 32 4
else
    # Full mode: comprehensive testing
    run_benchmark "baseline"            128 2 64 32 32 32 2

    # Thread count variations
    run_benchmark "threads_64"          64  2 64 32 32 32 2
    run_benchmark "threads_256"         256 2 64 32 32 32 2

    # Occupancy variations
    run_benchmark "occupancy_1"         128 1 64 32 32 32 2
    run_benchmark "occupancy_4"         128 4 64 32 32 32 2
    run_benchmark "occupancy_8"         128 8 64 32 32 32 2

    # Batch size variations
    run_benchmark "nbatch_fa_32"        128 2 32 32 32 32 2
    run_benchmark "nbatch_fa_128"       128 2 128 32 32 32 2

    # Tile size variations
    run_benchmark "tiles_16"            128 2 64 16 16 16 2
    run_benchmark "tiles_64"            128 2 64 64 64 64 2

    # Pipeline stages
    run_benchmark "stages_1"            128 2 64 32 32 32 1
    run_benchmark "stages_4"            128 2 64 32 32 32 4

    # Combinations
    run_benchmark "combo_morethreads"   256 2 64 32 32 32 2
    run_benchmark "combo_highoccupancy" 128 4 64 32 32 32 2
    run_benchmark "combo_smalltiles"    128 2 64 16 16 16 2
    run_benchmark "combo_aggressive"    256 4 64 16 16 16 4
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "                  TUNING COMPLETE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Show top results
echo "Top 3 Configurations (sorted by mean NPS):"
echo ""
(head -1 $RESULTS_FILE; tail -n +2 $RESULTS_FILE | grep -v "FAILED\|NO_DATA" | sort -t',' -k9 -rn | head -3) | column -t -s',' | head -4
echo ""

# Get best config
best_line=$(tail -n +2 $RESULTS_FILE | grep -v "FAILED\|NO_DATA" | sort -t',' -k9 -rn | head -1)
if [ -z "$best_line" ]; then
    echo "❌ No successful configurations found!"
    echo "Check build logs in /tmp/build_*.log"
    exit 1
fi

best_config=$(echo "$best_line" | cut -d',' -f1)
best_nps=$(echo "$best_line" | cut -d',' -f9)
best_nthreads=$(echo "$best_line" | cut -d',' -f2)
best_occupancy=$(echo "$best_line" | cut -d',' -f3)
best_nbatch_fa=$(echo "$best_line" | cut -d',' -f4)
best_nbatch_K2=$(echo "$best_line" | cut -d',' -f5)
best_nbatch_V2=$(echo "$best_line" | cut -d',' -f6)
best_nbatch_combine=$(echo "$best_line" | cut -d',' -f7)
best_nstages=$(echo "$best_line" | cut -d',' -f8)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "                 BEST CONFIGURATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Configuration: $best_config"
echo "Performance:   $best_nps nps"
echo ""
echo "Parameters:"
echo "  nthreads        = $best_nthreads"
echo "  occupancy       = $best_occupancy"
echo "  nbatch_fa       = $best_nbatch_fa"
echo "  nbatch_K2/V2    = $best_nbatch_K2"
echo "  nbatch_combine  = $best_nbatch_combine"
echo "  nstages         = $best_nstages"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "                  HOW TO USE THIS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Option 1: Use these settings temporarily"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Build with these flags:"
echo ""
echo "  ./build.sh release -Drocm=true \\"
echo "    -Dcpp_args=\"-DUSE_FLASH_ATTENTION=1 -DRDNA3 \\"
echo "      -DFATTN_NTHREADS_D32=$best_nthreads \\"
echo "      -DFATTN_OCCUPANCY_D32=$best_occupancy \\"
echo "      -DFATTN_NBATCH_FA_D32=$best_nbatch_fa \\"
echo "      -DFATTN_NBATCH_K2_D32=$best_nbatch_K2 \\"
echo "      -DFATTN_NBATCH_V2_D32=$best_nbatch_V2 \\"
echo "      -DFATTN_NBATCH_COMBINE_D32=$best_nbatch_combine \\"
echo "      -DFATTN_NSTAGES_D32=$best_nstages\""
echo ""
echo "Option 2: Make these settings permanent (recommended)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Edit scripts/rocm_backend_amd_gfx_configs.py"
echo "2. Find your architecture ($CURRENT_ARCH)"
echo "3. Update the flash_attention parameters:"
echo ""
echo "    'flash_attention': {"
echo "        'nthreads_d32': $best_nthreads,"
echo "        'occupancy_d32': $best_occupancy,"
echo "        'nbatch_fa_d32': $best_nbatch_fa,"
echo "        'nbatch_k2_d32': $best_nbatch_K2,"
echo "        'nbatch_v2_d32': $best_nbatch_V2,"
echo "        'nbatch_combine_d32': $best_nbatch_combine,"
echo "        'nstages_d32': $best_nstages,"
echo "    },"
echo ""
echo "4. Set 'tuned': True"
echo "5. Then just use: ./build.sh release -Drocm=true -Darch=$CURRENT_ARCH"
echo ""
echo "Full results saved to: $RESULTS_FILE"
echo ""
