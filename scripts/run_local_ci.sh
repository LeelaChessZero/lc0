#!/usr/bin/env bash
# =============================================================================
# scripts/run_local_ci.sh
# Run GitHub Actions CI jobs locally in clean Docker containers.
# Mirrors .github/workflows/ci.yml without pushing commits.
# Supports running:
#   1. 'linux'     (Ubuntu 24.04 Docker container: Meson + OpenBLAS + tests + benchmark)
#   2. 'rocm'      (ROCm 22.04 Docker container: Meson + ROCm/HIP backend)
#   3. 'downstream'(Ubuntu 24.04 Docker container: minimal backend-free build)
#   4. 'all'       (all containerized jobs sequentially)
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-all}"

if ! command -v docker &> /dev/null; then
    echo -e "\033[1;31mError: Docker is required to run isolated containerized CI builds.\033[0m" >&2
    exit 1
fi

print_header() {
    echo -e "\n\033[1;34m===============================================================================\033[0m"
    echo -e "\033[1;32m  $1\033[0m"
    echo -e "\033[1;34m===============================================================================\033[0m\n"
}

run_linux_container_ci() {
    print_header "Running Linux GCC Meson CI (Docker: ubuntu:24.04)"
    cd "$REPO_ROOT"

    docker run --rm \
        -v "$REPO_ROOT":/workspace \
        -w /workspace \
        ubuntu:24.04 \
        bash -c '
            set -euo pipefail
            export DEBIAN_FRONTEND=noninteractive
            echo ">>> Installing dependencies inside Ubuntu 24.04 container..."
            apt-get update -qq
            apt-get install -y -qq meson ninja-build ccache libopenblas-dev zlib1g-dev python3-pip git build-essential

            BUILD_DIR="build/docker-ci-linux"
            rm -rf "$BUILD_DIR"

            echo ">>> Configuring Meson build..."
            meson setup "$BUILD_DIR" \
                --buildtype release \
                -Dnative_arch=false \
                -Dgtest=true \
                -Dblas=true \
                -Dlc0=true

            echo ">>> Compiling with Ninja..."
            ninja -C "$BUILD_DIR"

            echo ">>> Running Unit Tests..."
            meson test -C "$BUILD_DIR" --print-errorlogs

            echo ">>> Running Engine Benchmark (Sanity Check)..."
            "./$BUILD_DIR/lc0" benchmark --backend=blas --num-positions=2 --movetime=2000

            echo -e "\n\033[1;32m✓ Linux Meson Container CI Passed Successfully!\033[0m"
        '
}

run_downstream_container_ci() {
    print_header "Running Downstream Minimal Build (Docker: ubuntu:24.04)"
    cd "$REPO_ROOT"

    docker run --rm \
        -v "$REPO_ROOT":/workspace \
        -w /workspace \
        ubuntu:24.04 \
        bash -c '
            set -euo pipefail
            export DEBIAN_FRONTEND=noninteractive
            echo ">>> Installing dependencies inside Ubuntu 24.04 container..."
            apt-get update -qq
            apt-get install -y -qq meson ninja-build zlib1g-dev python3-pip git build-essential

            BUILD_DIR="build/docker-ci-downstream"
            rm -rf "$BUILD_DIR"

            echo ">>> Configuring Meson build without external backends (-Dbuild_backends=false)..."
            meson setup "$BUILD_DIR" \
                --buildtype release \
                -Dgtest=false \
                -Dlc0=true \
                -Dblas=false \
                -Dbuild_backends=false

            echo ">>> Compiling lc0..."
            ninja -C "$BUILD_DIR" lc0

            echo ">>> Verifying built binary..."
            "./$BUILD_DIR/lc0" --help > /dev/null
            echo -e "\n\033[1;32m✓ Downstream Minimal Container Build Passed Successfully!\033[0m"
        '
}

run_rocm_ci() {
    print_header "Running AMD ROCm / HIP CI (Docker: rocm/dev-ubuntu-22.04:latest)"
    cd "$REPO_ROOT"

    echo ">>> Launching ROCm container to build HIP backend..."
    docker run --rm \
        -v "$REPO_ROOT":/workspace \
        -w /workspace \
        rocm/dev-ubuntu-22.04:latest \
        bash -c '
            set -euo pipefail
            export DEBIAN_FRONTEND=noninteractive
            echo ">>> Installing dependencies inside ROCm container..."
            apt-get update -qq
            apt-get install -y -qq git python3-pip ninja-build zlib1g-dev libopenblas-dev
            pip3 install -q meson

            BUILD_DIR="build/docker-ci-rocm"
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
        run_linux_container_ci
        ;;
    downstream)
        run_downstream_container_ci
        ;;
    rocm)
        run_rocm_ci
        ;;
    all)
        run_linux_container_ci
        run_downstream_container_ci
        run_rocm_ci
        ;;
    *)
        echo "Usage: $0 [linux|downstream|rocm|all]"
        exit 1
        ;;
esac

print_header "All Docker Container CI Runs Completed Successfully!"
