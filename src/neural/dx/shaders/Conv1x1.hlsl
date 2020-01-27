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

// ------------------- Simple 1x1 convolution shader -----------------------------//
RWBuffer<float> output  : register(u8);
RWBuffer<float> input   : register(u9);
RWBuffer<float> filter  : register(u10);
RWBuffer<float> bias    : register(u11);

cbuffer ConvConsts : register(b0) {
  uint N, K, C;       
  uint useBias;
  uint relu;
};


#define MAX_FILTERS 1024
groupshared float sh_filter[MAX_FILTERS];
groupshared float sh_bias;


// N*K thread blocks launched (groupIdx.y, and groupIdx.x resp.)
// Each block has 64 (8x8) thread.
// Each thread writes single output element.
[numthreads(kConv1x1BlockSize, 1, 1)] 
#if FP16_IO == 1
void conv_1x1_shader_fp16
#else
void conv_1x1_shader_fp32
#endif
(
    uint3 gtid  : SV_DispatchThreadID, 
    uint3 tid   : SV_GroupThreadID,
    uint3 gid   : SV_GroupID
) 
{
  int k = gid.x;
  int n = gid.y;

  // load bias into shared memory
  if (tid.x == 0) 
      sh_bias = useBias ? bias[k] : 0;

  // load filter into shared memory
  const int iterations = (C - 1) / kConv1x1BlockSize + 1;
  for (int i = 0; i < iterations; i++)
  {
    int c = i * kConv1x1BlockSize + tid.x;
    if (c < C) 
        sh_filter[c] = filter[k * C + c];
  }

  GroupMemoryBarrierWithGroupSync();

  float op = sh_bias;
  for (int c = 0; c < C; c++)
  {
    float ip = input[n * C * 64 + c * 64 + tid.x];
    float filter = sh_filter[c];
    op += ip * filter;
  }

  if (relu && op < 0) op = 0;

  output[n * K * 64 + k * 64 + tid.x] = op;
}

