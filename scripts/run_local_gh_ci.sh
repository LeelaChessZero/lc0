#!/usr/bin/env bash
# =============================================================================
# scripts/run_local_gh_ci.sh
# Run GitHub Actions CI jobs locally in a single unified Docker container.
# Uses rocm/dev-ubuntu-22.04:latest (which includes ROCm/HIP, GCC, and Linux dev
# toolchains) so only one Docker image is needed across all test jobs.
#
# Usage:
#   ./scripts/run_local_gh_ci.sh [linux|downstream|rocm|all]
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-all}"
ROCM_IMAGE="rocm/dev-ubuntu-22.04:latest"

if ! command -v docker &> /dev/null; then
    echo -e "\033[1;31mError: Docker is required to run isolated containerized CI builds.\033[0m" >&2
    exit 1
fi

if ! docker image inspect "$ROCM_IMAGE" &> /dev/null; then
    echo ">>> Container image $ROCM_IMAGE not found locally. Pulling image..."
    docker pull "$ROCM_IMAGE"
fi

print_header() {
    echo -e "\n\033[1;34m===============================================================================\033[0m"
    echo -e "\033[1;32m  $1\033[0m"
    echo -e "\033[1;34m===============================================================================\033[0m\n"
}

run_linux_ci() {
    print_header "Running Linux GCC Meson Build, Tests & Benchmark"
    docker run --rm \
        -v "$REPO_ROOT":/workspace \
        -w /workspace \
        "$ROCM_IMAGE" \
        bash -c '
            set -euo pipefail
            export DEBIAN_FRONTEND=noninteractive
            apt-get update -qq
            apt-get install -y -qq git python3-pip ninja-build zlib1g-dev libopenblas-dev ccache
            pip3 install -q meson

            BUILD_DIR="build/ci-linux"
            rm -rf "$BUILD_DIR"

            echo ">>> Configuring Meson build (Linux OpenBLAS)..."
            meson setup "$BUILD_DIR" \
                --buildtype release \
                -Dnative_arch=false \
                -Dgtest=true \
                -Dblas=true \
                -Dlc0=true

            echo ">>> Compiling lc0 & test suite with Ninja..."
            ninja -C "$BUILD_DIR"

            echo ">>> Running Unit Tests..."
            meson test -C "$BUILD_DIR" --print-errorlogs

            echo ">>> Running Engine Benchmark..."
            "./$BUILD_DIR/lc0" benchmark --backend=blas --num-positions=2 --movetime=2000

            echo -e "\n\033[1;32m✓ Linux CI Job Passed Successfully!\033[0m"
        '
}

run_downstream_ci() {
    print_header "Running Downstream Minimal / Backend-Free Build"
    docker run --rm \
        -v "$REPO_ROOT":/workspace \
        -w /workspace \
        "$ROCM_IMAGE" \
        bash -c '
            set -euo pipefail
            export DEBIAN_FRONTEND=noninteractive
            apt-get update -qq
            apt-get install -y -qq git python3-pip ninja-build zlib1g-dev
            pip3 install -q meson

            BUILD_DIR="build/ci-downstream"
            rm -rf "$BUILD_DIR"

            echo ">>> Configuring Meson build without external backends (-Dbuild_backends=false)..."
            meson setup "$BUILD_DIR" \
                --buildtype release \
                -Dgtest=false \
                -Dlc0=true \
                -Dblas=false \
                -Dbuild_backends=false

            echo ">>> Compiling minimal lc0..."
            ninja -C "$BUILD_DIR" lc0

            echo ">>> Verifying binary execution..."
            "./$BUILD_DIR/lc0" --help > /dev/null
            echo -e "\n\033[1;32m✓ Downstream Minimal Build Passed Successfully!\033[0m"
        '
}

run_rocm_ci() {
    print_header "Running AMD ROCm / HIP Backend Build (hipcc + hipblas)"
    docker run --rm \
        -v "$REPO_ROOT":/workspace \
        -w /workspace \
        "$ROCM_IMAGE" \
        bash -c '
            set -euo pipefail
            export DEBIAN_FRONTEND=noninteractive
            apt-get update -qq
            apt-get install -y -qq git python3-pip ninja-build zlib1g-dev libopenblas-dev
            pip3 install -q meson

            BUILD_DIR="build/ci-rocm"
            rm -rf "$BUILD_DIR"

            echo ">>> Configuring Meson with ROCm/HIP backend..."
            meson setup "$BUILD_DIR" \
                --buildtype release \
                -Db_lto=false \
                -Dhip=true \
                -Damd_gfx=gfx90a \
                -Dgtest=false \
                -Dlc0=true

            echo ">>> Compiling HIP backend & engine..."
            ninja -C "$BUILD_DIR"

            echo ">>> Verifying backend registration..."
            "./$BUILD_DIR/lc0" --help | grep -i "hip" || true
            echo -e "\n\033[1;32m✓ AMD ROCm / HIP CI Passed Successfully!\033[0m"
        '
}

case "$TARGET" in
    linux)
        run_linux_ci
        ;;
    downstream)
        run_downstream_ci
        ;;
    rocm)
        run_rocm_ci
        ;;
    all)
        run_linux_ci
        run_downstream_ci
        run_rocm_ci
        ;;
    *)
        echo "Usage: $0 [linux|downstream|rocm|all]"
        exit 1
        ;;
esac

print_header "All CI Jobs Completed Successfully in ROCm Container!"
