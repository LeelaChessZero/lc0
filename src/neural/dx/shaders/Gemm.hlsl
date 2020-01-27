/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2020 The LCZero Authors

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
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include "shader_shared.h"

// ------------------- Matrix Multiply Shader -------------------------//

#define MAT_A_INDEX(b, y, x) ((b)*M*K + (y)*K + (x))
#define MAT_B_INDEX(b, y, x) ((b)*K*N + (y)*N + (x))
#define MAT_C_INDEX(b, y, x) ((b)*M*N + (y)*N + (x))

#if USE_FP16_MATH == 1
RWBuffer<float16_t4>    matrixA    : register(u8);
RWBuffer<float16_t4>    matrixB    : register(u9);
RWBuffer<float16_t4>    matrixC    : register(u10);
#else
RWBuffer<float4>        matrixA    : register(u8);
RWBuffer<float4>        matrixB    : register(u9);
RWBuffer<float4>        matrixC    : register(u10);
#endif

cbuffer consts : register(b0) {
    uint M, N, K;
    uint batch;
};

// All matrices are row-major.


// Use shared memory to load inputs.
// Also multiple elements per thread.

// Double-buffered shared memory buffers
// (so that the GPU can overlap loads into shared memory with the math)
#if USE_FP16_MATH==1
groupshared float16_t sharedA[2][kGemmElPerBlockY][kGemmShMemKChunk];
groupshared float16_t sharedB[2][kGemmShMemKChunk][kGemmElPerBlockX];
#else
groupshared float sharedA[2][kGemmElPerBlockY][kGemmShMemKChunk];
groupshared float sharedB[2][kGemmShMemKChunk][kGemmElPerBlockX];
#endif

#define divUp(a, b) (((a)-1)/(b) + 1)

void loadShmemBuffers(int batch, int hs, int ws, int ks, int tidy, int tidx, int bufIndex)
{
#if USE_FP16_MATH==1
    float16_t4 temp;
#else
    float4 temp;
#endif

    const int iterationsA = divUp((kGemmElPerBlockY * kGemmShMemKChunk),
                                  (kGemmBlockWidth * kGemmBlockHeight * 4));
    int i;
    [unroll]
    for (i = 0; i < iterationsA; i++)
    {
      int index = (kGemmBlockWidth * kGemmBlockHeight) * i +
                  kGemmBlockHeight * tidy + tidx;
        index *= 4;
        int lx = index % kGemmShMemKChunk;
        int ly = index / kGemmShMemKChunk;
        if ((hs + ly < M) && (ks + lx < K))
        {
            temp = matrixA[MAT_A_INDEX(batch, hs + ly, ks + lx) / 4];
        }
        else
        {
            temp = 0;
        }
        sharedA[bufIndex][ly][lx + 0] = temp.x;
        sharedA[bufIndex][ly][lx + 1] = temp.y;
        sharedA[bufIndex][ly][lx + 2] = temp.z;
        sharedA[bufIndex][ly][lx + 3] = temp.w;

    }

    const int iterationsB = divUp((kGemmShMemKChunk * kGemmElPerBlockX),
                                  (kGemmBlockWidth * kGemmBlockHeight * 4));
    [unroll]
    for (i = 0; i < iterationsB; i++)
    {
      int index = (kGemmBlockWidth * kGemmBlockHeight) * i +
                  kGemmBlockHeight * tidy + tidx;
        index *= 4;
        int lx = index % kGemmElPerBlockX;
        int ly = index / kGemmElPerBlockX;
        if ((ks + ly < K) && (ws + lx < N))
        {
            temp = matrixB[MAT_B_INDEX(batch, ks + ly, ws + lx) / 4];
        }
        else
        {
            temp = 0;
        }
        sharedB[bufIndex][ly][lx + 0] = temp.x;
        sharedB[bufIndex][ly][lx + 1] = temp.y;
        sharedB[bufIndex][ly][lx + 2] = temp.z;
        sharedB[bufIndex][ly][lx + 3] = temp.w;
    }
}


[numthreads(kGemmBlockWidth, kGemmBlockHeight, 1)] 
void MatrixMul(
    uint3 g_tid : SV_DispatchThreadID,
    uint3 gid : SV_GroupID,
    uint3 l_tid : SV_GroupThreadID
)
{
    int x, y;

#if USE_FP16_MATH==1
    float16_t S[kGemmElPerThreadY][kGemmElPerThreadX];
#else
    float S[kGemmElPerThreadY][kGemmElPerThreadX];
#endif

    [unroll] for (y = 0; y < kGemmElPerThreadY; y++)
      [unroll] for (x = 0; x < kGemmElPerThreadX; x++)
        S[y][x] = 0;

    int wStartThread = g_tid.x * kGemmElPerThreadX;
    int hStartThread = g_tid.y * kGemmElPerThreadY;

    int wStartBlock = gid.x * kGemmBlockWidth * kGemmElPerThreadX;
    int hStartBlock = gid.y * kGemmBlockHeight * kGemmElPerThreadY;

    for (int ks = 0, index = 0; ks < K; ks += kGemmShMemKChunk, index++)
    {
        int shIndex = index & 1;
        // Load chunks of matrices A and B into shared memory.
        loadShmemBuffers(gid.z, hStartBlock, wStartBlock, ks, l_tid.y, l_tid.x, shIndex);

        GroupMemoryBarrierWithGroupSync();

        // Do the Multiplication for the Tile.
        // Removing this unroll improves performance on Nvidia Turing but makes it slightly slower on AMD Vega 7.
        [unroll]
        for (int k = 0; k < kGemmShMemKChunk; k++)
            [unroll]
            for (y = 0; y < kGemmElPerThreadY; y++)
                [unroll]
                for (x = 0; x < kGemmElPerThreadX; x++)
                {
                    int shy = y + l_tid.y * kGemmElPerThreadY;
                    int shx = x + l_tid.x * kGemmElPerThreadX;
                    S[y][x] += sharedA[shIndex][shy][k] * sharedB[shIndex][k][shx];
                }
    }

    // Write results to output.
#if USE_FP16_MATH==1
    float16_t4 temp;
#else
    float4 temp;
#endif

    [unroll]
    for (y = 0; y < kGemmElPerThreadY; y++)
    {
        int w = wStartThread;
        int h = hStartThread + y;

        if (h < M && w < N)
        {
          temp.x = S[y][0];
          temp.y = S[y][1];
          temp.z = S[y][2];
          temp.w = S[y][3];
          matrixC[MAT_C_INDEX(gid.z, h, w + 0) / 4] = temp;

          temp.x = S[y][4];
          temp.y = S[y][5];
          temp.z = S[y][6];
          temp.w = S[y][7];
          matrixC[MAT_C_INDEX(gid.z, h, w + 4) / 4] = temp;
        }
    }
}

