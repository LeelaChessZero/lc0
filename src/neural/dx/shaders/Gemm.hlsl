#include "shader_shared.h"

#if FP16_IO == 1
RWStructuredBuffer<float16_t>  matrixA    : register(u0);
RWStructuredBuffer<float16_t>  matrixB    : register(u1);
RWStructuredBuffer<float16_t>  matrixC    : register(u2);
#else
RWStructuredBuffer<float>  matrixA    : register(u0);
RWStructuredBuffer<float>  matrixB    : register(u1);
RWStructuredBuffer<float>  matrixC    : register(u2);
#endif


cbuffer consts : register(b0) {
    uint M, N, K;
    uint batch;
};

// All matrices are row-major.
#define MAT_C(b, y, x) matrixC[(b)*M*N + (y)*N + (x)]
#define MAT_A(b, y, x) matrixA[(b)*M*K + (y)*K + (x)]
#define MAT_B(b, y, x) matrixB[(b)*K*N + (y)*N + (x)]


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
    // TODO! Vectorized loads!

    const int iterationsA = divUp((ELEMENTS_PER_BLOCK_Y * SHARED_MEM_K_CHUNK) , (BLOCK_WIDTH * BLOCK_HEIGHT));
    int i;
    [unroll]
    for (i = 0; i < iterationsA; i++)
    {
        int index = (BLOCK_WIDTH * BLOCK_HEIGHT) * i + BLOCK_HEIGHT * tidy + tidx;
        int lx = index % SHARED_MEM_K_CHUNK;
        int ly = index / SHARED_MEM_K_CHUNK;
        if ((hs + ly < M) && (ks + lx < K))
            sharedA[bufIndex][ly][lx] = MAT_A(batch, hs + ly, ks + lx);
        else
            sharedA[bufIndex][ly][lx] = 0;
    }

    const int iterationsB = divUp((SHARED_MEM_K_CHUNK * ELEMENTS_PER_BLOCK_X) , (BLOCK_WIDTH * BLOCK_HEIGHT));
    [unroll]
    for (i = 0; i < iterationsB; i++)
    {
        int index = (BLOCK_WIDTH * BLOCK_HEIGHT) * i + BLOCK_HEIGHT * tidy + tidx;
        int lx = index % ELEMENTS_PER_BLOCK_X;
        int ly = index / ELEMENTS_PER_BLOCK_X;
        if ((ks + ly < K) && (ws + lx < N))
            sharedB[bufIndex][ly][lx] = MAT_B(batch, ks + ly, ws + lx);
        else
            sharedB[bufIndex][ly][lx] = 0;
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
    [unroll]
    for (y = 0; y < ELEMENTS_PER_THREAD_Y; y++)
        [unroll]
        for (x = 0; x < ELEMENTS_PER_THREAD_X; x++)
        {
            int w = wStartThread + x;
            int h = hStartThread + y;
            if (h < M && w < N)
                MAT_C(gid.z, h, w) = S[y][x];
        }
}