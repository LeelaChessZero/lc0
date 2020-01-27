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

// ------------------- Add Vectors kernel -----------------------------//
//
// C = act(A + B)
// A and B can have different lengths, mod size is used to pick the required
// element.
// fp16 version processes 2 elements at a time.

RWStructuredBuffer<uint> A : register(u0);
RWStructuredBuffer<uint> B : register(u1);
RWStructuredBuffer<uint> C : register(u2);

cbuffer AddVectorConsts : register(b0) {
  // sizes are /2 for fp16
  uint a_size;
  uint b_size;
  uint c_size;
  uint relu;
  uint act_tanh;
  uint fp16;
};

float2 extractElements(uint packedVal) {
  return float2(f16tof32(packedVal & 0xFFFF),
                f16tof32((packedVal >> 16) & 0xFFFF));
}

[numthreads(kAddVectorsBlockSize, 1, 1)] 
void add_vectors_shader
(
  uint3 globalThreadIdx : SV_DispatchThreadID
) 
{
  int index = globalThreadIdx.x;
  if (index >= c_size) return;
  uint a = A[index % a_size];
  uint b = B[index % b_size];
  uint opVal;
  if (fp16) {
    float2 f2a = extractElements(a);
    float2 f2b = extractElements(b);
    float2 f2c = f2a + f2b;
    if (relu) {
      if (f2c.x < 0) f2c.x = 0;
      if (f2c.y < 0) f2c.y = 0;
    }
    if (act_tanh) {
      f2c = tanh(f2c);
    }
    uint2 opu = f32tof16(f2c);
    opVal = opu.x | (opu.y << 16);
  } else {
    float c = asfloat(a) + asfloat(b);
    if (relu && c < 0) c = 0;
    if (act_tanh) c = tanh(c);
    opVal = asuint(c);
  }
  C[index] = opVal;
}
