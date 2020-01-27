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

#if USE_FP16_MATH == 1

RWBuffer<float16_t4>  input               : register(u8);
RWBuffer<float16_t>   transformedInput    : register(u9);

RWBuffer<float16_t>  transformedOutput    : register(u8);
RWBuffer<float16_t4> output               : register(u9);
RWBuffer<float16_t>  bias                 : register(u10);
RWBuffer<float16_t4> skipConnection       : register(u11);
RWBuffer<float16_t>  se_w1                : register(u12);
RWBuffer<float16_t>  se_b1                : register(u13);
RWBuffer<float16_t>  se_w2                : register(u14);
RWBuffer<float16_t>  se_b2                : register(u15);

#else

RWBuffer<float4>  input               : register(u8);
RWBuffer<float>   transformedInput    : register(u9);

RWBuffer<float>  transformedOutput    : register(u8);
RWBuffer<float4> output               : register(u9);
RWBuffer<float>  bias                 : register(u10);
RWBuffer<float4> skipConnection       : register(u11);
RWBuffer<float>  se_w1                : register(u12);
RWBuffer<float>  se_b1                : register(u13);
RWBuffer<float>  se_w2                : register(u14);
RWBuffer<float>  se_b2                : register(u15);
#endif


cbuffer consts : register(b0) {
    uint N, C;

    // Additional fused ops.
    // Used only by output transform shader.
    uint relu;
    uint useBias;
    uint skipAdd;
    uint fusedSe;

    // The channel count after SE (C / se-ratio)
    uint se_K;
};


// index in input/output tensors
#define INDEX_NCHW(n,c,h,w) ((n)*C*H*W + (c)*H*W + (h)*W + w)

// index in intermediate/temp tensor
// W, H == 6 here! (6x6 transformed blocks)
// N also includes part of dimension (2x2)
#define GemmN (N * 4)
#define TEMP_INDEX_HWNC(h,w,n,c) ((h)*6*GemmN*C + (w)*GemmN*C + (n)*C + c)

//----------------------------- Utility functions for Winograd transform ------------------------------//

// fp16/half math seems a bit slow! - on both Nvidia Turing and AMD Vega 7 (Bugs? Lack of optimizations?)
// These are memory bandwidth bound shaders anyway.
#if USE_FP16_MATH == 1

void matrixMul_gpu_serial_6x6x6(out float16_t c[6][6], in float16_t a[6][6], in float16_t b[6][6])
{
    [unroll]
    for (int i = 0; i < 6; ++i)
        [unroll]
        for (int j = 0; j < 6; ++j)
        {
            float16_t S = 0;
            [unroll]
            for (int k = 0; k < 6; ++k)
                S += a[i][k] * b[k][j];
            c[i][j] = S;
        }
}

void matrixMul_gpu_serial_4x6x6(out float16_t c[4][6], in float16_t a[4][6], in float16_t b[6][6])
{
    [unroll]
    for (int i = 0; i < 4; ++i)
        [unroll]
        for (int j = 0; j < 6; ++j)
        {
            float16_t S = 0;
            [unroll]
            for (int k = 0; k < 6; ++k)
                S += a[i][k] * b[k][j];
            c[i][j] = S;
        }
}

void matrixMul_gpu_serial_4x4x6(out float16_t c[4][4], in float16_t a[4][6], in float16_t b[6][4])
{
    [unroll]
    for (int i = 0; i < 4; ++i)
        [unroll]
        for (int j = 0; j < 4; ++j)
        {
            float16_t S = 0;
            [unroll]
            for (int k = 0; k < 6; ++k)
                S += a[i][k] * b[k][j];
            c[i][j] = S;
        }
}

void inputTransform4x4_gpu(out float16_t op[6][6], in const float16_t ip[6][6])
{
    // transform applied to input tile (of size 4x4 - padded up to 6x6)
    const float16_t Bt[6][6] = 
    {
        4,  0, -5,  0, 1, 0,
        0, -4, -4,  1, 1, 0,
        0,  4, -4, -1, 1, 0,
        0, -2, -1,  2, 1, 0,
        0,  2, -1, -2, 1, 0,
        0,  4,  0, -5, 0, 1
    };

    const float16_t B[6][6] =
    {
        4,  0,  0,  0,  0,  0,
        0, -4,  4, -2,  2,  4,
       -5, -4, -4, -1, -1,  0,
        0,  1, -1,  2, -2, -5,
        1,  1,  1,  1,  1,  0,
        0,  0,  0,  0,  0,  1
    };

    float16_t tempIp1[6][6];
    matrixMul_gpu_serial_6x6x6(tempIp1, Bt, ip);
    matrixMul_gpu_serial_6x6x6(op, tempIp1, B);
}

void outputTransform4x4_gpu(out float16_t output[4][4], in const float16_t transformedOutput[6][6])
{
    // transform applied to result
    const float16_t At[4][6] = {
        1, 1, 1, 1, 1, 0,
        0, 1,-1, 2,-2, 0,
        0, 1, 1, 4, 4, 0,
        0, 1,-1, 8,-8, 1
    };

    const float16_t A[6][4] = {
        1, 0, 0, 0,
        1, 1, 1, 1,
        1,-1, 1,-1,
        1, 2, 4, 8,
        1,-2, 4,-8,
        0, 0, 0, 1
    };

    float16_t tempOp[4][6];
    matrixMul_gpu_serial_4x6x6(tempOp, At, transformedOutput);
    matrixMul_gpu_serial_4x4x6(output, tempOp, A);
}

#else


//----------------------------- FP32 versions of the same code above ------------------------------//

void matrixMul_gpu_serial_6x6x6(out float c[6][6], in float a[6][6], in float b[6][6])
{
    [unroll]
    for (int i = 0; i < 6; ++i)
        [unroll]
        for (int j = 0; j < 6; ++j)
        {
            float S = 0;
            [unroll]
            for (int k = 0; k < 6; ++k)
                S += a[i][k] * b[k][j];
            c[i][j] = S;
        }
}

void matrixMul_gpu_serial_4x6x6(out float c[4][6], in float a[4][6], in float b[6][6])
{
    [unroll]
    for (int i = 0; i < 4; ++i)
        [unroll]
        for (int j = 0; j < 6; ++j)
        {
            float S = 0;
            [unroll]
            for (int k = 0; k < 6; ++k)
                S += a[i][k] * b[k][j];
            c[i][j] = S;
        }
}

void matrixMul_gpu_serial_4x4x6(out float c[4][4], in float a[4][6], in float b[6][4])
{
    [unroll]
    for (int i = 0; i < 4; ++i)
        [unroll]
        for (int j = 0; j < 4; ++j)
        {
            float S = 0;
            [unroll]
            for (int k = 0; k < 6; ++k)
                S += a[i][k] * b[k][j];
            c[i][j] = S;
        }
}


void inputTransform4x4_gpu(out float op[6][6], in const float ip[6][6])
{
    // transform applied to input tile (of size 4x4 - padded up to 6x6)
    const float Bt[6][6] =
    {
        4,  0, -5,  0, 1, 0,
        0, -4, -4,  1, 1, 0,
        0,  4, -4, -1, 1, 0,
        0, -2, -1,  2, 1, 0,
        0,  2, -1, -2, 1, 0,
        0,  4,  0, -5, 0, 1
    };

    const float B[6][6] =
    {
        4,  0,  0,  0,  0,  0,
        0, -4,  4, -2,  2,  4,
       -5, -4, -4, -1, -1,  0,
        0,  1, -1,  2, -2, -5,
        1,  1,  1,  1,  1,  0,
        0,  0,  0,  0,  0,  1
    };

    float tempIp1[6][6];
    matrixMul_gpu_serial_6x6x6(tempIp1, Bt, ip);
    matrixMul_gpu_serial_6x6x6(op, tempIp1, B);
}

void outputTransform4x4_gpu(out float output[4][4], in const float transformedOutput[6][6])
{
    // transform applied to result
    const float At[4][6] = {
        1, 1, 1, 1, 1, 0,
        0, 1,-1, 2,-2, 0,
        0, 1, 1, 4, 4, 0,
        0, 1,-1, 8,-8, 1
    };

    const float A[6][4] = {
        1, 0, 0, 0,
        1, 1, 1, 1,
        1,-1, 1,-1,
        1, 2, 4, 8,
        1,-2, 4,-8,
        0, 0, 0, 1
    };

    float tempOp[4][6];
    matrixMul_gpu_serial_4x6x6(tempOp, At, transformedOutput);
    matrixMul_gpu_serial_4x4x6(output, tempOp, A);
}

#endif