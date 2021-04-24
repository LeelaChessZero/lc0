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

// ------------------- Expand Planes Shader -----------------------------//


RWStructuredBuffer<float>    output_fp32 : register(u0);
RWStructuredBuffer<uint>     output_fp16 : register(u0);
RWStructuredBuffer<uint64_t> masks       : register(u1);
RWStructuredBuffer<float>    values      : register(u2);

cbuffer ExpandPlanesConsts : register(b0) {
  uint N;             // total no of planes to process
  uint kInputPlanes;  // no of planes per position
};


// Block size of 256, same mask/val for 64 consecutive threads.
#define kNumShmemElements (kExpandPlanesElementsPerBlock / 64)
groupshared uint64_t sh_masks[kNumShmemElements];
groupshared float    sh_vals[kNumShmemElements];

[numthreads(kExpandPlanesFp32BlockSize, 1, 1)] 
void ExpandPlanes_shader_fp32
(
    uint3 globalThreadIdx  : SV_DispatchThreadID, 
    uint3 threadIdxInGroup : SV_GroupThreadID
) 
{

  int global_index = globalThreadIdx.x;
  int local_index  = threadIdxInGroup.x;

  int plane_index = global_index >> 6;

  if (plane_index >= N) return;

  // Load inputs to shared memory.
  if (local_index < kNumShmemElements) {
    sh_masks[local_index] = masks[plane_index + local_index];
    sh_vals[local_index] = values[plane_index + local_index];
  }

  GroupMemoryBarrierWithGroupSync();

  uint64_t mask = sh_masks[local_index >> 6];

  int sq_index = global_index & 0x3F;
  float op = 0;

  bool set = !!(mask & (1ull << sq_index));
  if (set) {
    op = sh_vals[local_index >> 6];
  }
  output_fp32[global_index] = op;
}


// every thread writes two consecutive elements
// NCHW means that the consecutive elements are in W dimension
[numthreads(kExpandPlanesFp16BlockSize, 1, 1)] 
void ExpandPlanes_shader_fp16
(
    uint3 globalThreadIdx  : SV_DispatchThreadID,
    uint3 threadIdxInGroup : SV_GroupThreadID
) 
{
  int global_index = globalThreadIdx.x * 2;
  int local_index = threadIdxInGroup.x * 2;

  int plane_index = global_index >> 6;

  if (plane_index >= N) return;

  // Load inputs to shared memory.
  if (threadIdxInGroup.x < kNumShmemElements) {
    sh_masks[threadIdxInGroup.x] = masks[plane_index + threadIdxInGroup.x];
    sh_vals[threadIdxInGroup.x] = values[plane_index + threadIdxInGroup.x];
  }

  GroupMemoryBarrierWithGroupSync();

  uint64_t mask = sh_masks[local_index >> 6];

  int sq_index0 = global_index & 0x3F;
  int sq_index1 = sq_index0 + 1;

  bool set0 = !!(mask & (1ull << sq_index0));
  bool set1 = !!(mask & (1ull << sq_index1));

  float2 opf = 0;

  if (set0) {
    opf.x = sh_vals[local_index >> 6];
  }

  if (set1) {
    opf.y = sh_vals[local_index >> 6];
  }

  uint2 opu = f32tof16(opf);
  uint opVal = opu.x | (opu.y << 16);
  output_fp16[globalThreadIdx.x] = opVal;
}
