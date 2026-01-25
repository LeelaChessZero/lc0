/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with AMD ROCm libraries from the ROCm toolkit
  (or a modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#pragma once
// This file contains primitives that expose the tensor core instructions for HIP code.
// The primitives can be used in a similar way as the nvcuda::wmma interface but with a well-defined memory layout.
//
// Like with nvcuda::wmma there are three types of matrix tiles: A, B, and C with A @ B = C.
// A is a row-major matrix with shape M x K.
// B is a column-major matrix with shape K x N.
// C is a column-major matrix with shape M x N.
// A, B, and C are represented using the same fundamental data type: a row-major matrix with I rows and J columns.
// Note that J is measured in physical 32 bit elements instead of logical elements.
// The methods get_i and get_j can be used to get the physical 32 bit index of the lth element of a thread within a tile.
// All matrix tiles have ne physical 32 bit elements per warp.
//
// As described in the documentation, all pointers for load operations should be aligned to 16 bytes.
// The API in this file also assumes that the pointers for load_generic are aligned to 16 bytes, unaligned pointers are considered undefined behavior.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Helper macros and constants
#define WARP_SIZE 32
#define GGML_UNUSED(x) (void)(x)

// Error handling for device code
[[noreturn]]
static __device__ void no_device_code(
    const char * file_name, const int line, const char * function_name) {
    printf("%s:%d: ERROR: HIP kernel %s has no device code compatible with this architecture.\n",
           file_name, line, function_name);
    __builtin_trap();
    GGML_UNUSED(no_device_code); // suppress unused function warning
}

#ifdef __HIP_DEVICE_COMPILE__
#define NO_DEVICE_CODE no_device_code(__FILE__, __LINE__, __FUNCTION__)
#else
#define NO_DEVICE_CODE
#endif

// Memory copy helper function
template <int nbytes, int alignment = 0>
static __device__ __forceinline__ void hip_memcpy_1(void * __restrict__ dst, const void * __restrict__ src) {
    if constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr int nb_per_cpy = alignment == 0 ? nbytes : alignment;

#pragma unroll
    for (int i = 0; i < nbytes/nb_per_cpy; ++i) {
        if constexpr (nb_per_cpy == 1) {
            ((char *) dst)[i] = ((const char *) src)[i];
        } else if constexpr (nb_per_cpy == 2) {
            ((short *) dst)[i] = ((const short *) src)[i];
        } else if constexpr (nb_per_cpy == 4) {
            ((int *) dst)[i] = ((const int *) src)[i];
        } else if constexpr (nb_per_cpy == 8) {
            ((int2 *) dst)[i] = ((const int2 *) src)[i];
        } else if constexpr (nb_per_cpy == 16) {
            ((int4 *) dst)[i] = ((const int4 *) src)[i];
        } else {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}

// Get maximum copy bytes for aligned transfers
static __device__ __forceinline__ constexpr int hip_get_max_cpy_bytes() {
    return 16;
}

// Dummy bfloat16 type - not used in lc0 (T82 uses FP16 only)
// We need a unique type to avoid template conflicts with half2
struct nv_bfloat162_dummy { half2 data; };
using nv_bfloat162 = nv_bfloat162_dummy;

namespace lczero_mma {

    // Some architectures like RDNA3 perform multiple matrix multiplications per warp in parallel,
    //     effectively the warp is being split into subgroups of threads that each perform a single mma instruction.
    // In those cases the data can be split in different ways across the warp.
    enum data_layout {
        // By default the data uses the I direction as its major dimension and the J direction as its minor dimension.
        // For the A/C matrices this means I major == row major, J major == column major.
        // For the B matrix this means I major == column major, J major == row major.
        // MIRRORED == Each data value is held exactly once per thread subgroup.
        DATA_LAYOUT_I_MAJOR           =  0, // Always used for matrix A&B for RDNA4 and CDNA.
        DATA_LAYOUT_J_MAJOR           = 10, // Matrix C for CDNA and RDNA4, int and float matrix C for RDNA3.
        DATA_LAYOUT_I_MAJOR_MIRRORED  = 20, // Matrix A&B for RDNA3.
        DATA_LAYOUT_J_MAJOR_MIRRORED  = 30,
    };
    // Implemented mma combinations are:
    //   - (I_MAJOR, I_MAJOR)          -> I_MAJOR
    //   - (I_MAJOR, I_MAJOR_MIRRORED) -> I_MAJOR
    //   - (I_MAJOR, J_MAJOR_MIRRORED) -> I_MAJOR

    static constexpr bool is_i_major(const data_layout dl) {
        return dl == DATA_LAYOUT_I_MAJOR ||
               dl == DATA_LAYOUT_I_MAJOR_MIRRORED;
    }

    static constexpr __device__ data_layout get_input_data_layout() {
#if defined(RDNA3)
        return DATA_LAYOUT_I_MAJOR_MIRRORED;
#else
        return DATA_LAYOUT_I_MAJOR;
#endif // defined(RDNA3)
    }

    template <int I_, int J_, typename T, data_layout ds_=DATA_LAYOUT_I_MAJOR>
    struct tile {};

    template <int I_, int J_, typename T>
    struct tile<I_, J_, T, DATA_LAYOUT_I_MAJOR> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR;

#if defined(AMD_MFMA_AVAILABLE)
        static constexpr int ne = I * J / 64;
        T x[ne] = {0};

        static constexpr __device__ bool supported() {
            if (I == 64 && J ==  2) return true;
            if (I == 16 && J ==  8) return true;
            if (I == 32 && J ==  4) return true;
            if (I == 16 && J == 16) return true;
            if (I == 32 && J == 32) return true;
            return false;
        }

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (I == 64 && J == 2) { // Special tile size to load <16, 4> as <16, 8>
                return threadIdx.x % 16;
            } else if constexpr (I == 16 && J == 8) {
                return threadIdx.x % 16;
            } else if constexpr (I == 32 && J == 4) {
                return threadIdx.x % 32;
            } else if constexpr (I == 16 && J == 16) {
                return threadIdx.x % 16;
            } else if constexpr (I == 32 && J == 32) {
                return threadIdx.x % 32;
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 64 && J == 2) { // Special tile size to load <16, 4> as <16, 8>
                return (2 * ((threadIdx.x / 16) % 2) + l);
            } else if constexpr (I == 16 && J == 8) {
                return 2 * (threadIdx.x / 16) + l;
            } else if constexpr (I == 32 && J == 4) {
                return 2 * (threadIdx.x / 32) + l;
            } else if constexpr (I == 16 && J == 16) {
                return 4 * (threadIdx.x / 16) + l;
            } else if constexpr (I == 32 && J == 32) {
                return 4 * (threadIdx.x / 32) + 8 * (l / 4) + (l % 4);
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }
#elif defined(AMD_WMMA_AVAILABLE)
        static constexpr int ne = I * J / 32;
        T x[ne] = {0};

        static constexpr __device__ bool supported() {
            if (I == 16 && J == 16) return true;
            if (I == 16 && J == 8) return true;
            if (I == 16 && J == 4) return true;
            return false;
        }

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (supported()) {
                return threadIdx.x % 16;
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 16 && J == 16) {
#if defined(RDNA3)
                if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int>) {
                    // matrix C
                    return 2 * l + (threadIdx.x / 16);
                } else {
                    // matrix A&B
                    return l;
                }
#else
                // matrix C is the transposed matrix A&B on RDNA4
                return ne * (threadIdx.x / 16) + l;
#endif // defined(RDNA3)
            } else if constexpr (I == 16 && J == 8) {
                // mmq input for RDNA4
                return ne * (threadIdx.x / 16) + l;
            } else if constexpr (I == 16 && J == 4) {
                return ne * (threadIdx.x / 16) + l;
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }
#else
        static constexpr int ne = I * J / 32;
        T x[ne] = {0};

        static constexpr __device__ bool supported() {
            return false;
        }

        static __device__ __forceinline__ int get_i(const int l) {
            NO_DEVICE_CODE;
            return -1;
        }

        static __device__ __forceinline__ int get_j(const int l) {
            NO_DEVICE_CODE;
            return -1;
        }
#endif // defined(AMD_MFMA_AVAILABLE)
    };

    template <int I_, int J_>
    struct tile<I_, J_, half2, DATA_LAYOUT_I_MAJOR> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR;

#if defined(AMD_WMMA_AVAILABLE)
        static constexpr int ne = I * J / 32;
        half2 x[ne] = {{0.0f, 0.0f}};

        static constexpr __device__ bool supported() {
            if (I == 16 && J == 8) return true;
            return false;
        }

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (I == 16 && J == 8) {
                return threadIdx.x % 16;
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 16 && J == 8) {
                return 4 * (threadIdx.x / 16) + l;
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }
#else
        static constexpr int ne = I * J / WARP_SIZE;
        half2 x[ne] = {{0.0f, 0.0f}};

        static constexpr __device__ bool supported() {
            return false;
        }

        static __device__ __forceinline__ int get_i(const int l) {
            NO_DEVICE_CODE;
            return -1;
        }

        static __device__ __forceinline__ int get_j(const int l) {
            NO_DEVICE_CODE;
            return -1;
        }
#endif // defined(AMD_WMMA_AVAILABLE)
    };

    template <int I_, int J_>
    struct tile<I_, J_, nv_bfloat162, DATA_LAYOUT_I_MAJOR> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR;

#if defined(AMD_WMMA_AVAILABLE)
        static constexpr int ne = I * J / 32;
        nv_bfloat162 x[ne] = {{0.0f, 0.0f}};

        static constexpr __device__ bool supported() {
            return tile<I_, J_, half2, DATA_LAYOUT_I_MAJOR>::supported();
        }

        static __device__ __forceinline__ int get_i(const int l) {
            return tile<I_, J_, half2, DATA_LAYOUT_I_MAJOR>::get_i(l);
        }

        static __device__ __forceinline__ int get_j(const int l) {
            return tile<I_, J_, half2, DATA_LAYOUT_I_MAJOR>::get_j(l);
        }
#else
        static constexpr int ne = I * J / WARP_SIZE;
        nv_bfloat162 x[ne] = {{0.0f, 0.0f}};

        static constexpr __device__ bool supported() {
            return false;
        }

        static __device__ __forceinline__ int get_i(const int l) {
            NO_DEVICE_CODE;
            return -1;
        }

        static __device__ __forceinline__ int get_j(const int l) {
            NO_DEVICE_CODE;
            return -1;
        }
#endif  // defined(AMD_WMMA_AVAILABLE)
    };

    template <int I_, int J_, typename T>
    struct tile<I_, J_, T, DATA_LAYOUT_J_MAJOR> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_J_MAJOR;

        static constexpr int ne = tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::ne;
        T x[ne] = {0};

        static constexpr __device__ bool supported() {
            return tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::supported();
        }

        static __device__ __forceinline__ int get_i(const int l) {
            return tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::get_j(l);
        }

        static __device__ __forceinline__ int get_j(const int l) {
            return tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::get_i(l);
        }
    };

    template <int I_, int J_, typename T>
    struct tile<I_, J_, T, DATA_LAYOUT_I_MAJOR_MIRRORED> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR_MIRRORED;

        // RDNA3
        static constexpr int         ne = I * J / 32 * 2;

        T x[ne] = {0};

        static constexpr __device__ bool supported() {
            if (I == 16 && J == 16) return true;
            if (I == 16 && J == 8)  return true;
            if (I == 16 && J == 4)  return true;
            return false;
        }

        static __device__ __forceinline__ int get_i(const int /*l*/) {
            if constexpr (supported()) {
                return threadIdx.x % 16;
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (supported()) {
                return l;
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }
    };

    template <int I_, int J_>
    struct tile<I_, J_, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR_MIRRORED;
#if defined(RDNA3)
        static constexpr int         ne = tile<I_, J_, float, DATA_LAYOUT_I_MAJOR_MIRRORED>::ne;

        half2 x[ne] = {{0.0f, 0.0f}};

        static constexpr __device__ bool supported() {
            return tile<I_, J_, float, DATA_LAYOUT_I_MAJOR_MIRRORED>::supported();
        }

        static __device__ __forceinline__ int get_i(const int l) {
            return tile<I_, J_, float, DATA_LAYOUT_I_MAJOR_MIRRORED>::get_i(l);
        }

        static __device__ __forceinline__ int get_j(const int l) {
            return tile<I_, J_, float, DATA_LAYOUT_I_MAJOR_MIRRORED>::get_j(l);
        }
#else
        static constexpr int         ne = I * J / (WARP_SIZE/4);

        half2 x[ne] = {{0.0f, 0.0f}};

        static constexpr __device__ bool supported() {
            return false;
        }

        static __device__ __forceinline__ int get_i(const int /*l*/) {
            NO_DEVICE_CODE;
            return -1;
        }

        static __device__ __forceinline__ int get_j(const int l) {
            NO_DEVICE_CODE;
            return -1;
        }
#endif // defined(RDNA3)
    };

    // Stub template for J_MAJOR_MIRRORED - not used in lc0, just for compilation
    template <int I_, int J_>
    struct tile<I_, J_, half2, DATA_LAYOUT_J_MAJOR_MIRRORED> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_J_MAJOR_MIRRORED;
        static constexpr int         ne = 4;  // Minimal stub

        half2 x[ne] = {{0.0f, 0.0f}};

        static constexpr __device__ bool supported() { return false; }
        static __device__ __forceinline__ int get_i(const int) { return 0; }
        static __device__ __forceinline__ int get_j(const int) { return 0; }
    };

    template <int I_, int J_>
    struct tile<I_, J_, nv_bfloat162, DATA_LAYOUT_I_MAJOR_MIRRORED> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR_MIRRORED;
        static constexpr int         ne = tile<I_, J_, float, DATA_LAYOUT_I_MAJOR_MIRRORED>::ne;

        nv_bfloat162 x[ne] = {{0.0f, 0.0f}};

        static constexpr __device__ bool supported() {
            return tile<I_, J_, float, DATA_LAYOUT_I_MAJOR_MIRRORED>::supported();
        }

        static __device__ __forceinline__ int get_i(const int l) {
            return tile<I_, J_, float, DATA_LAYOUT_I_MAJOR_MIRRORED>::get_i(l);
        }

        static __device__ __forceinline__ int get_j(const int l) {
            return tile<I_, J_, float, DATA_LAYOUT_I_MAJOR_MIRRORED>::get_j(l);
        }
    };

#if defined(AMD_WMMA_AVAILABLE)
    template <int I, int J>
    static __device__ __forceinline__ tile<I, J/2, half2> get_half2(const tile<I, J, float> & tile_float) {
        tile<I, J/2, half2> ret;
#pragma unroll
        for (int l0 = 0; l0 < tile_float.ne; l0 += 2) {
            ret.x[l0/2] = make_half2(tile_float.x[l0 + 0], tile_float.x[l0 + 1]);
        }
        return ret;
    }

    static __device__ __forceinline__ tile<8, 8, half2> get_transposed(const tile<16, 4, half2> & t) {
        NO_DEVICE_CODE;
        return tile<8, 8, half2>{};
    }
#endif // defined(AMD_WMMA_AVAILABLE)

    static __device__ __forceinline__ void make_identity_mat(tile<16, 8, half2> & t) {
#if defined(RDNA4)
        const int row = t.get_i(0);
        const int left_right = t.get_j(0) / 4;
        const int up_down = row / 8;
        const int idx = row % 8;
        reinterpret_cast<half*>(t.x)[idx] = left_right == up_down ? 1.0f : 0.0f;
#else
        GGML_UNUSED(t);
        NO_DEVICE_CODE;
#endif // defined(RDNA4)
    }

    template <int I, int J, typename T, data_layout dl>
    static __device__ __forceinline__ void load_generic(tile<I, J, T, dl> & t, const T * __restrict__ xs0, const int stride) {
#if defined(AMD_MFMA_AVAILABLE)
        if constexpr (I == 64 && J == 2) { // Special tile size to load <16, 4> as <16, 8>
#pragma unroll
            for (int l = 0; l < t.ne; ++l) {
                t.x[l] = xs0[t.get_i(l)*stride + t.get_j(l)];
            }
        } else {
            hip_memcpy_1<sizeof(t.x)>(t.x, xs0 + t.get_i(0) * stride + t.get_j(0));
        }
#elif defined(AMD_WMMA_AVAILABLE)
        // All wmma layout has contiguous data when i-major.
        if constexpr (is_i_major(dl)) {
            // the data must be aligned to 16 bytes when bigger than hip_get_max_cpy_bytes()
            constexpr int aligned_copy_bytes = hip_get_max_cpy_bytes();
            if constexpr (sizeof(t.x) > aligned_copy_bytes) {
                static_assert(sizeof(t.x) % aligned_copy_bytes == 0, "bad type size");
                constexpr int aligned_copy_count = sizeof(t.x)/aligned_copy_bytes;
#pragma unroll
                for (int i = 0; i < aligned_copy_count; ++i) {
                    hip_memcpy_1<aligned_copy_bytes>(t.x + t.ne/aligned_copy_count*i, xs0 + t.get_i(0) * stride + t.get_j(t.ne/aligned_copy_count*i));
                }
            } else {
                hip_memcpy_1<sizeof(t.x)>(t.x, xs0 + t.get_i(0) * stride + t.get_j(0));
            }
        } else {
#pragma unroll
            for (int l = 0; l < t.ne; ++l) {
                t.x[l] = xs0[t.get_i(l)*stride + t.get_j(l)];
            }
        }
#else
#pragma unroll
        for (int l = 0; l < t.ne; ++l) {
            t.x[l] = xs0[t.get_i(l)*stride + t.get_j(l)];
        }
#endif // defined(AMD_MFMA_AVAILABLE)
    }

    template <typename T>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<8, 8, T> & t, const T * __restrict__ xs0, const int stride) {
        load_generic(t, xs0, stride);
    }

    template <typename T>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<16, 4, T> & t, const T * __restrict__ xs0, const int stride) {
        load_generic(t, xs0, stride);
    }

    template <typename T, data_layout dl>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<16, 8, T, dl> & t, const T * __restrict__ xs0, const int stride) {
        load_generic(t, xs0, stride);
    }

    static __device__ __forceinline__ void load_ldmatrix(
            tile<8, 4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> & t, const half2 * __restrict__ xs0, const int stride) {
        hip_memcpy_1<4*sizeof(half2)>(t.x, xs0 + t.get_i(0)*stride);
    }

    static __device__ __forceinline__ void load_ldmatrix(
            tile<8, 4, half2, DATA_LAYOUT_J_MAJOR_MIRRORED> & t, const half2 * __restrict__ xs0, const int stride) {
#pragma unroll
        for (int l0 = 0; l0 < t.ne; l0 += 2) {
            hip_memcpy_1<2*sizeof(half2)>(t.x + l0, xs0 + t.get_i(l0)*stride + t.get_j(l0));
        }
    }

    static __device__ __forceinline__ void load_ldmatrix(
            tile<32, 4, half2> & t, const half2 * __restrict__ xs0, const int stride) {
        GGML_UNUSED(t);
        GGML_UNUSED(xs0);
        GGML_UNUSED(stride);
        NO_DEVICE_CODE;
    }

    template <typename T>
    static __device__ __forceinline__ void load_ldmatrix_trans(
            tile<16, 8, T> & t, const T * __restrict__ xs0, const int stride) {
        GGML_UNUSED(t);
        GGML_UNUSED(xs0);
        GGML_UNUSED(stride);
        NO_DEVICE_CODE;
    }

    static __device__ __forceinline__ void mma(
            tile<16, 8, int> & D, const tile<16, 4, int> & A, const tile<8, 4, int> & B) {
        GGML_UNUSED(D);
        GGML_UNUSED(A);
        GGML_UNUSED(B);
        NO_DEVICE_CODE;
    }

    static __device__ __forceinline__ void mma(
            tile<16, 8, int> & D, const tile<16, 8, int> & A, const tile<8, 8, int> & B) {
        GGML_UNUSED(D);
        GGML_UNUSED(A);
        GGML_UNUSED(B);
        NO_DEVICE_CODE;
    }

    static __device__ __forceinline__ void mma(
            tile<16, 4, half2> & D, const tile<16, 8, half2> & A, const tile<8, 8, half2> & B) {
        GGML_UNUSED(D);
        GGML_UNUSED(A);
        GGML_UNUSED(B);
        NO_DEVICE_CODE;
    }

    static __device__ __forceinline__ void mma(
            tile<16, 8, half2> & D, const tile<16, 8, half2> & A, const tile<16, 8, half2> & B) {
#if defined(AMD_WMMA_AVAILABLE)
#if defined(RDNA4)
        using halfx8_t = __attribute__((ext_vector_type(8))) _Float16;
        halfx8_t& acc_frag = reinterpret_cast<halfx8_t&>(D.x[0]);
        const halfx8_t& a_frag = reinterpret_cast<const halfx8_t&>(A.x[0]);
        const halfx8_t& b_frag = reinterpret_cast<const halfx8_t&>(B.x[0]);
        acc_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(a_frag, b_frag, acc_frag);
#else
        GGML_UNUSED(D);
        GGML_UNUSED(A);
        GGML_UNUSED(B);
        NO_DEVICE_CODE;
#endif // defined(RDNA4)
#else
        GGML_UNUSED(D);
        GGML_UNUSED(A);
        GGML_UNUSED(B);
        NO_DEVICE_CODE;
#endif // defined(AMD_WMMA_AVAILABLE)
    }

    template <data_layout dl_ab, data_layout dl_d>
    static __device__ __forceinline__ void mma(
            tile<16, 8, float, dl_d> & D, const tile<16, 8, float, dl_ab> & A, const tile<8, 8, float, dl_ab> & B) {
        GGML_UNUSED(D);
        GGML_UNUSED(A);
        GGML_UNUSED(B);
        NO_DEVICE_CODE;
    }

    static __device__ __forceinline__ void mma_block_scaled(tile<16, 8, float> &     D,
                                                            const tile<16, 8, int> & A,
                                                            const tile<8, 8, int> &  B,
                                                            uint32_t                 a_scale,
                                                            uint32_t                 b_scale) {
        GGML_UNUSED(D);
        GGML_UNUSED(A);
        GGML_UNUSED(B);
        GGML_UNUSED(a_scale);
        GGML_UNUSED(b_scale);
        NO_DEVICE_CODE;
    }

    static __device__ __forceinline__ void mma(
            tile<16, 8, float> & D, const tile<16, 8, half2> & A, const tile<8, 8, half2> & B) {
        GGML_UNUSED(D);
        GGML_UNUSED(A);
        GGML_UNUSED(B);
        NO_DEVICE_CODE;
    }

    static __device__ __forceinline__ void mma(
            tile<16, 8, float> & D, const tile<16, 8, nv_bfloat162> & A, const tile<8, 8, nv_bfloat162> & B) {
        GGML_UNUSED(D);
        GGML_UNUSED(A);
        GGML_UNUSED(B);
        NO_DEVICE_CODE;
    }

    template <data_layout dl_ab, data_layout dl_d>
    static __device__ __forceinline__ void mma(
            tile<16, 16, float, dl_d> & D, const tile<16, 8, half2, dl_ab> & A, const tile<16, 8, half2, dl_ab> & B) {
#if defined(AMD_WMMA_AVAILABLE)
#if defined(RDNA4)
        using halfx8_t = __attribute__((ext_vector_type(8))) _Float16;
        using floatx8_t = __attribute__((ext_vector_type(8))) float;
        floatx8_t& acc_frag = reinterpret_cast<floatx8_t&>(D.x[0]);
        const halfx8_t& a_frag = reinterpret_cast<const halfx8_t&>(A.x[0]);
        const halfx8_t& b_frag = reinterpret_cast<const halfx8_t&>(B.x[0]);
        acc_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_frag, b_frag, acc_frag);
#elif defined(RDNA3)
        using halfx16_t = __attribute__((ext_vector_type(16))) _Float16;
        using floatx8_t = __attribute__((ext_vector_type(8))) float;
        floatx8_t& acc_frag = reinterpret_cast<floatx8_t&>(D.x[0]);
        const halfx16_t& a_frag = reinterpret_cast<const halfx16_t&>(A.x[0]);
        const halfx16_t& b_frag = reinterpret_cast<const halfx16_t&>(B.x[0]);
        acc_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, acc_frag);
#else
        GGML_UNUSED(D);
        GGML_UNUSED(A);
        GGML_UNUSED(B);
        NO_DEVICE_CODE;
#endif // RDNA4
#else
        GGML_UNUSED(D);
        GGML_UNUSED(A);
        GGML_UNUSED(B);
        NO_DEVICE_CODE;
#endif // defined(AMD_WMMA_AVAILABLE)
    }

    template <data_layout dl_ab, data_layout dl_d>
    static __device__ __forceinline__ void mma(
            tile<16, 16, float, dl_d> & D, const tile<16, 8, nv_bfloat162, dl_ab> & A, const tile<16, 8, nv_bfloat162, dl_ab> & B) {
#if defined(AMD_WMMA_AVAILABLE)
#if defined(RDNA4)
        using bf16x8_t = __attribute__((ext_vector_type(8))) __hip_bfloat16;
        using floatx8_t = __attribute__((ext_vector_type(8))) float;
        floatx8_t& acc_frag = reinterpret_cast<floatx8_t&>(D.x[0]);
        const bf16x8_t& a_frag = reinterpret_cast<const bf16x8_t&>(A.x[0]);
        const bf16x8_t& b_frag = reinterpret_cast<const bf16x8_t&>(B.x[0]);
        acc_frag = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a_frag, b_frag, acc_frag);
#elif defined(RDNA3)
        using bf16x16_t = __attribute__((ext_vector_type(16))) __hip_bfloat16;
        using floatx8_t = __attribute__((ext_vector_type(8))) float;
        floatx8_t& acc_frag = reinterpret_cast<floatx8_t&>(D.x[0]);
        const bf16x16_t& a_frag = reinterpret_cast<const bf16x16_t&>(A.x[0]);
        const bf16x16_t& b_frag = reinterpret_cast<const bf16x16_t&>(B.x[0]);
        acc_frag = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_frag, b_frag, acc_frag);
#else
        GGML_UNUSED(D);
        GGML_UNUSED(A);
        GGML_UNUSED(B);
        NO_DEVICE_CODE;
#endif // RDNA4
#else
        GGML_UNUSED(D);
        GGML_UNUSED(A);
        GGML_UNUSED(B);
        NO_DEVICE_CODE;
#endif // defined(AMD_WMMA_AVAILABLE)
    }

    template <data_layout dl_d, data_layout dl_ab>
    static __device__ __forceinline__ void mma(
            tile<16, 16, int, dl_d> & D, const tile<16, 8, int, dl_ab> & A, const tile<16, 8, int, dl_ab> & B) {
#if defined(AMD_MFMA_AVAILABLE)
        using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
        int32x4_t * acc = (int32x4_t *) D.x;
#if defined(CDNA3)
        acc[0] = __builtin_amdgcn_mfma_i32_16x16x32_i8(((int64_t *) A.x)[0],
                                                       ((int64_t *) B.x)[0],
                                                       acc[0],
                                                       0, 0, 0);
#elif defined(CDNA2) || defined(CDNA)
        acc[0] = __builtin_amdgcn_mfma_i32_16x16x16i8(A.x[0],
                                                      B.x[0],
                                                      acc[0],
                                                      0, 0, 0);
        acc[0] = __builtin_amdgcn_mfma_i32_16x16x16i8(A.x[1],
                                                      B.x[1],
                                                      acc[0],
                                                      0, 0, 0);
#endif // defined(CDNA3)

#elif defined(AMD_WMMA_AVAILABLE)

        using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;
        int32x8_t * acc = (int32x8_t *) D.x;

#if defined(RDNA4)
        using int32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
        int32x2_t * a_vec = (int32x2_t *) A.x;
        int32x2_t * b_vec = (int32x2_t *) B.x;

        acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
            true,
            a_vec[0],
            true,
            b_vec[0],
            acc[0],
            true
        );

        acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
            true,
            a_vec[1],
            true,
            b_vec[1],
            acc[0],
            true
        );

#elif defined(RDNA3)
        using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
        int32x4_t * a_vec = (int32x4_t *) A.x;
        int32x4_t * b_vec = (int32x4_t *) B.x;

        acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
            true,
            a_vec[0],
            true,
            b_vec[0],
            acc[0],
            true
        );

        acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
            true,
            a_vec[1],
            true,
            b_vec[1],
            acc[0],
            true
        );
#endif // RDNA4

#else
        GGML_UNUSED(D);
        GGML_UNUSED(A);
        GGML_UNUSED(B);
        NO_DEVICE_CODE;
#endif // AMD_MFMA_AVAILABLE
    }

    static __device__ __forceinline__ void mma(
            tile<32, 32, int> & D, const tile<32, 4, int> & A, const tile<32, 4, int> & B) {
#if defined(AMD_MFMA_AVAILABLE)
        using int32x16_t = __attribute__((__vector_size__(16 * sizeof(int)))) int;
        int32x16_t * acc = (int32x16_t *) D.x;
#if defined(CDNA3)
        acc[0] = __builtin_amdgcn_mfma_i32_32x32x16_i8(((int64_t *) A.x)[0],
                                                       ((int64_t *) B.x)[0],
                                                       acc[0],
                                                       0, 0, 0);
#elif defined(CDNA2) || defined(CDNA)
        acc[0] = __builtin_amdgcn_mfma_i32_32x32x8i8(A.x[0],
                                                     B.x[0],
                                                     acc[0],
                                                     0, 0, 0);
        acc[0] = __builtin_amdgcn_mfma_i32_32x32x8i8(A.x[1],
                                                     B.x[1],
                                                     acc[0],
                                                     0, 0, 0);
#endif // defined(CDNA3)

#else
        GGML_UNUSED(D);
        GGML_UNUSED(A);
        GGML_UNUSED(B);
        NO_DEVICE_CODE;
#endif // AMD_MFMA_AVAILABLE
    }

    template <typename T1, typename T2, int J, int K>
    static __device__ __forceinline__ void mma(
            tile<32, J, T1> & D, const tile<32, K, T2> & A, const tile<J, K, T2> & B) {
        tile      <16, J, T1> * D16 = reinterpret_cast<      tile<16, J, T1> *>(&D);
        const tile<16, K, T2> * A16 = reinterpret_cast<const tile<16, K, T2> *>(&A);
        mma(D16[0], A16[0], B);
        mma(D16[1], A16[1], B);
    }

    static __device__ __forceinline__ void mma(
            tile<32, 8, float> & D, const tile<32, 4, half2> & A, const tile<8, 4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> & B) {
        GGML_UNUSED(D);
        GGML_UNUSED(A);
        GGML_UNUSED(B);
        NO_DEVICE_CODE;
    }

    static __device__ __forceinline__ void mma(
            tile<32, 4, half2> & D, const tile<32, 4, half2> & A, const tile<8, 4, half2, DATA_LAYOUT_J_MAJOR_MIRRORED> & B) {
        GGML_UNUSED(D);
        GGML_UNUSED(A);
        GGML_UNUSED(B);
        NO_DEVICE_CODE;
    }

    template <data_layout dl_d, data_layout dl_ab>
    static __device__ __forceinline__ void mma(
            tile<16, 16, int, dl_d> & D, const tile<16, 4, int, dl_ab> & A, const tile<16, 4, int, dl_ab> & B) {
#if defined(AMD_WMMA_AVAILABLE)
        using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;
        int32x8_t * acc = (int32x8_t *) D.x;
#if defined(RDNA4)
        using int32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
        int32x2_t * a_vec = (int32x2_t *) A.x;
        int32x2_t * b_vec = (int32x2_t *) B.x;

        acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
            true,
            a_vec[0],
            true,
            b_vec[0],
            acc[0],
            false
        );
#elif defined(RDNA3)
        using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
        int32x4_t * a_vec = (int32x4_t *) A.x;
        int32x4_t * b_vec = (int32x4_t *) B.x;

        acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
            true,
            a_vec[0],
            true,
            b_vec[0],
            acc[0],
            false
        );
#endif // RDNA4
#else
        GGML_UNUSED(D);
        GGML_UNUSED(A);
        GGML_UNUSED(B);
        NO_DEVICE_CODE;
#endif // AMD_WMMA_AVAILABLE
    }
}
