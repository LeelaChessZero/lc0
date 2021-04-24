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

cbuffer consts : register(b0) {
  uint N, C;

  // Additional fused ops.
  // Used only by output transform shader.
  uint relu;
  uint useBias;
  uint skipAdd;

  // The channel count after SE (C / se-ratio)
  uint se_K;
};


RWBuffer<float4> input              : register(u8);
RWBuffer<float4> output             : register(u9);
RWBuffer<float>  bias               : register(u10);
RWBuffer<float4> skipConnection     : register(u11);
RWBuffer<float>  se_w1              : register(u12);
RWBuffer<float>  se_b1              : register(u13);
RWBuffer<float>  se_w2              : register(u14);
RWBuffer<float>  se_b2              : register(u15);

// index in input/output tensors
#define INDEX_NCHW(n, c, h, w) ((n)*C * H * W + (c)*H * W + (h)*W + w)

// Fused SE layer implementation
// Each thread block processes entire 'C' dimension worth of data.
// N thread blocks launched.
// As DX Compute block size needs to be known in advance, we compile multiple
// versions of this same HLSL shader with various standard channel counts (128,
// 256, 320, 384, 512, 640, 768 and 1024) and use the next bigger channel count
// version to handle channel counts not in the list of standard counts.

// Note that the weight matrices are transposed.
#define readw1(row, col) (se_w1[(row)*se_K + (col)])
#define readw2(row, col) (se_w2[(row)*2 * C + (col)])

groupshared float sharedData[BLOCK_SIZE];

[numthreads(BLOCK_SIZE, 1, 1)]
void SE
(
    uint3 gid  : SV_GroupID,
    uint3 tid  : SV_GroupThreadID 
)
{
    const int H = 8, W = 8;

    int n = gid.x;
    int k = tid.x;

    // C is set to K in the constant buffer
    if (k >= C) return;

    // TODO: Try float4 board_r1[8], board_r2[8];
    //       Also try reading skip connection tensor early to get more
    //       math-memory access overlap.
    float board[8][8];
    float b = useBias ? bias[k] : 0;

    // Read input tensor.
    int h;
    [unroll] 
    for (h = 0; h < 8; h++) {
      int index = INDEX_NCHW(n, k, h, 0) / 4;
      float4 r1 = input[index];
      float4 r2 = input[index + 1];
      board[h][0] = r1.x;
      board[h][1] = r1.y;
      board[h][2] = r1.z;
      board[h][3] = r1.w;
      board[h][4] = r2.x;
      board[h][5] = r2.y;
      board[h][6] = r2.z;
      board[h][7] = r2.w;
    }

    // Add bias, and compute the average for SE.
    float S = 0;
    [unroll]
    for (int y = 0; y < 8; y++)
        [unroll]
        for (int x = 0; x < 8; x++)
        {
            board[y][x] += b;
            S += board[y][x];
        }
    float avg = S / 64;
    sharedData[k] = avg;

    GroupMemoryBarrierWithGroupSync();

    // First fully-connected layer for SE
    if (k < se_K) {
      S = 0;

      for (int i = 0; i < C; i++) {
        S += sharedData[i] * readw1(i, k);
      }

      S += se_b1[k];

      // relu
      if (S < 0) S = 0;

      sharedData[k] = S;
    }

    GroupMemoryBarrierWithGroupSync();

    // Second fully-connected layer for SE
    S = 0;
    float B = 0;
    for (int i = 0; i < se_K; i++) {
      float val = sharedData[i];
      S += val * readw2(i, k);
      B += val * readw2(i, k + C);
    }
    S += se_b2[k];
    B += se_b2[k + C];

    // Sigmoid (only on the scale part).
    S = 1.0 / (1.0 + exp(-S));


    // Scale, add skip connection, perform relu, and write to output.
    [unroll]
    for (h = 0; h < 8; h++)
    {
        int index = INDEX_NCHW(n, k, h, 0) / 4;
        // can possibly use uint4 to write entire row at a time?
        // couldn't find half2 to uint re-interpret functions :(
        // same issue for reads.
        float4 r1;
        float4 r2;
        r1.x = board[h][0];
        r1.y = board[h][1];
        r1.z = board[h][2];
        r1.w = board[h][3];
        r2.x = board[h][4];
        r2.y = board[h][5];
        r2.z = board[h][6];
        r2.w = board[h][7];

        // SE scale and bias
        r1 = r1*S + B;
        r2 = r2*S + B;

        // residual add
        if (skipAdd) {
            r1 += skipConnection[index];
            r2 += skipConnection[index+1];
        }

        // relu
        if (relu) {
            float4 zeros = float4(0, 0, 0, 0);
            r1 = max(r1, zeros);
            r2 = max(r2, zeros);
        }

        output[index]     = r1;
        output[index + 1] = r2;
    }
}
