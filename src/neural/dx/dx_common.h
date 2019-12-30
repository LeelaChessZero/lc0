/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2019 The LCZero Authors

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
#pragma once
#include <cstdint>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <stdint.h>
#include "d3dx12.h"

namespace lczero {

uint16_t FP32toFP16(float f32);
float FP16toFP32(uint16_t f16);

namespace dx_backend {

void DxError(HRESULT status, const char* file, const int& line);
#define ReportDxErrors(status) DxError(status, __FILE__, __LINE__)

struct DXAlloc {
  ID3D12Resource* pResource;
  uint32_t offset;
  uint64_t gpuVA;
  D3D12_GPU_DESCRIPTOR_HANDLE descHandle;
};

typedef uint16_t dx_half;

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

}  // namespace dx_backend
}  // namespace lczero
