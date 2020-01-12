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
*/

#include "shader_shared.h"

// ------------------- Policy Map Shader -----------------------------//
RWBuffer<float> input : register(u8);

// Output is always fp32.
RWStructuredBuffer<float> output  : register(u1);

// Weights are always int32.
RWStructuredBuffer<int> indices : register(u2);

cbuffer PolicyMapConsts : register(b0) {
  uint N;
  uint inputSize;
  uint usedSize;
  uint outputSize;
};

[numthreads(kPolicyMapBlockSize, 1, 1)] 
void PolicyMapShader
(
  uint3 globalThreadIdx : SV_DispatchThreadID
) 
{
  int tid = globalThreadIdx.x;
  int n = tid / usedSize;
  int i = tid % usedSize;

  if (n >= N) return;

  int j = indices[i];

  if (j >= 0) {
    output[n * outputSize + j] = input[n * inputSize + i];
  }
}
