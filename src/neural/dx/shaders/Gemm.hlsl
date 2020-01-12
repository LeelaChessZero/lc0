#include "shader_shared.h"

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
groupshared float16_t sharedA[2][ELEMENTS_PER_BLOCK_Y][SHARED_MEM_K_CHUNK];
groupshared float16_t sharedB[2][SHARED_MEM_K_CHUNK][ELEMENTS_PER_BLOCK_X];
#else
groupshared float sharedA[2][ELEMENTS_PER_BLOCK_Y][SHARED_MEM_K_CHUNK];
groupshared float sharedB[2][SHARED_MEM_K_CHUNK][ELEMENTS_PER_BLOCK_X];
#endif

#define divUp(a, b) (((a)-1)/(b) + 1)

void loadShmemBuffers(int batch, int hs, int ws, int ks, int tidy, int tidx, int bufIndex)
{
#if USE_FP16_MATH==1
    float16_t4 temp;
#else
    float4 temp;
#endif

    const int iterationsA = divUp((ELEMENTS_PER_BLOCK_Y * SHARED_MEM_K_CHUNK), (BLOCK_WIDTH * BLOCK_HEIGHT * 4));
    int i;
    [unroll]
    for (i = 0; i < iterationsA; i++)
    {
        int index = (BLOCK_WIDTH * BLOCK_HEIGHT) * i + BLOCK_HEIGHT * tidy + tidx;
        index *= 4;
        int lx = index % SHARED_MEM_K_CHUNK;
        int ly = index / SHARED_MEM_K_CHUNK;
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

    const int iterationsB = divUp((SHARED_MEM_K_CHUNK * ELEMENTS_PER_BLOCK_X), (BLOCK_WIDTH * BLOCK_HEIGHT * 4));
    [unroll]
    for (i = 0; i < iterationsB; i++)
    {
        int index = (BLOCK_WIDTH * BLOCK_HEIGHT) * i + BLOCK_HEIGHT * tidy + tidx;
        index *= 4;
        int lx = index % ELEMENTS_PER_BLOCK_X;
        int ly = index / ELEMENTS_PER_BLOCK_X;
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


[numthreads(BLOCK_WIDTH, BLOCK_HEIGHT, 1)]
void MatrixMul
(
    uint3 g_tid : SV_DispatchThreadID,
    uint3 gid : SV_GroupID,
    uint3 l_tid : SV_GroupThreadID
)
{
    int x, y;

#if USE_FP16_MATH==1
    float16_t S[ELEMENTS_PER_THREAD_Y][ELEMENTS_PER_THREAD_X];
#else
    float S[ELEMENTS_PER_THREAD_Y][ELEMENTS_PER_THREAD_X];
#endif

    [unroll]
    for (y = 0; y < ELEMENTS_PER_THREAD_Y; y++)
        [unroll]
        for (x = 0; x < ELEMENTS_PER_THREAD_X; x++)
            S[y][x] = 0;

    int wStartThread = g_tid.x * ELEMENTS_PER_THREAD_X;
    int hStartThread = g_tid.y * ELEMENTS_PER_THREAD_Y;

    int wStartBlock = gid.x * BLOCK_WIDTH * ELEMENTS_PER_THREAD_X;
    int hStartBlock = gid.y * BLOCK_HEIGHT * ELEMENTS_PER_THREAD_Y;

    for (int ks = 0, index = 0; ks < K; ks += SHARED_MEM_K_CHUNK, index++)
    {
        int shIndex = index & 1;
        // Load chunks of matrices A and B into shared memory.
        loadShmemBuffers(gid.z, hStartBlock, wStartBlock, ks, l_tid.y, l_tid.x, shIndex);

        GroupMemoryBarrierWithGroupSync();

        // Do the Multiplication for the Tile.
        // Removing this unroll improves performance on Nvidia Turing but makes it slightly slower on AMD Vega 7.
        [unroll]
        for (int k = 0; k < SHARED_MEM_K_CHUNK; k++)
            [unroll]
            for (y = 0; y < ELEMENTS_PER_THREAD_Y; y++)
                [unroll]
                for (x = 0; x < ELEMENTS_PER_THREAD_X; x++)
                {
                    int shy = y + l_tid.y * ELEMENTS_PER_THREAD_Y;
                    int shx = x + l_tid.x * ELEMENTS_PER_THREAD_X;
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
    for (y = 0; y < ELEMENTS_PER_THREAD_Y; y++)
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

