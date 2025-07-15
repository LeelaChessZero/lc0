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
#pragma once

#include <d3d12.h>
#include <dxgi1_6.h>
#include <stdint.h>

#include <cstdint>

#include "d3dx12.h"
#include "utils/fp16_utils.h"

#define DEFAULT_FP16 true

// To debug in case some GPUs can't read from sysmem directly in shader.
//#define COPY_BEFORE_SHADER_READ
// Dump per-layer debug data to find where things go wrong.
//#define DEBUG_DUMP_PER_LAYER_DATA

namespace lczero {

namespace dx_backend {

void DxError(HRESULT status, const char* file, const int& line);
#define ReportDxErrors(status) DxError(status, __FILE__, __LINE__)

struct DXAlloc {
  ID3D12Resource* resource;
  uint32_t offset;
  // Various ways of binding an allocation to shader:
  // 1. RAW/Structured buffer bound as root UAV, use gpu_va directly.
  // 2. Typed buffer UAV bound as 4-component typed format (e.g:
  // R16G16B16A16_FLOAT)
  // 3. Typed buffer UAV bound as single component scalar typed format (e.g:
  // R16_FLOAT)

  uint64_t gpu_va;

  // Handle of UAV created as 4-component vector type.
  D3D12_GPU_DESCRIPTOR_HANDLE desc_handle_vector;

  // Handle of UAV created as scalar type.
  D3D12_GPU_DESCRIPTOR_HANDLE desc_handle_scalar;
};

typedef uint16_t dx_half;

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

}  // namespace dx_backend
}  // namespace lczero
