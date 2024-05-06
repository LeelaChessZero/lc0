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

namespace lczero {
// Obtained by using EnumerateMetaCommands and EnumerateMetaCommandParameters
// calls. Simplified a bit.

struct TensorDesc {
  uint64_t DataType;
  uint64_t Flags;
  uint64_t DimensionCount;
  uint64_t Size[5];
  uint64_t Stride[5];
  uint64_t StrideAlignment[5];
  uint64_t BaseAlignmentInBytes;
  uint64_t PhysicalSizeInElements;
};

//----------------------------------------------------------------------------------//
// GEMM (Matrix multiply)
//----------------------------------------------------------------------------------//

constexpr GUID GemmGuid = {0x1e52ebab,
                           0x25ba,
                           0x463a,
                           {0xa2, 0x85, 0x0a, 0x78, 0x8e, 0xef, 0x5d, 0x01}};

struct GemmCreateDesc {
  TensorDesc DescA;
  TensorDesc DescB;
  TensorDesc DescC;
  uint64_t cMatrixNull;
  TensorDesc DescOut;

  uint64_t Precision;
  uint64_t TransA;
  uint64_t TransB;
  float Alpha;
  float Beta;

  uint64_t ActivationFunction;
  float ActivationParam1, ActivationParam2;
  uint64_t ActivationIsNull;
  uint64_t BindFlags;
};

struct GemmInitDesc {
  D3D12_GPU_DESCRIPTOR_HANDLE AResource;
  D3D12_GPU_DESCRIPTOR_HANDLE BResource;
  D3D12_GPU_DESCRIPTOR_HANDLE CResource;
  D3D12_GPU_DESCRIPTOR_HANDLE PersistentResource;
  D3D12_GPU_DESCRIPTOR_HANDLE TemporaryResource;
};

struct GemmExecuteDesc {
  D3D12_GPU_DESCRIPTOR_HANDLE AResource;
  D3D12_GPU_DESCRIPTOR_HANDLE BResource;
  D3D12_GPU_DESCRIPTOR_HANDLE CResource;
  D3D12_GPU_DESCRIPTOR_HANDLE OutputResource;

  D3D12_GPU_DESCRIPTOR_HANDLE PersistentResource;
  D3D12_GPU_DESCRIPTOR_HANDLE TemporaryResource;
};

//----------------------------------------------------------------------------------//
// Convolution
//----------------------------------------------------------------------------------//

constexpr GUID ConvGuid = {0x17804d6b,
                           0xebfe,
                           0x426f,
                           {0x88, 0xfc, 0xfe, 0xa7, 0x2e, 0x3f, 0x33, 0x56}};

struct ConvCreateDesc {
  TensorDesc InputDesc;
  TensorDesc FilterDesc;
  TensorDesc BiasDesc;
  uint64_t BiasNull;
  TensorDesc OutputDesc;

  uint64_t Mode;
  uint64_t Direction;
  uint64_t Precision;
  uint64_t Stride[3];
  uint64_t Dilation[3];
  uint64_t StartPadding[3];
  uint64_t EndPadding[3];
  uint64_t DimensionCount;
  uint64_t OutputPadding[5];
  uint64_t GroupCount;
  uint64_t ActivationFunction;
  float ActivationParam1, ActivationParam2;
  uint64_t ActivationIsNull;
  uint64_t BindFlags;
};

struct InitConvDesc {
  D3D12_GPU_DESCRIPTOR_HANDLE InputResource;
  D3D12_GPU_DESCRIPTOR_HANDLE FilterResource;
  D3D12_GPU_DESCRIPTOR_HANDLE BiasResource;
  D3D12_GPU_DESCRIPTOR_HANDLE PersistentResource;
  D3D12_GPU_DESCRIPTOR_HANDLE TemporaryResource;
};

struct ExecuteConvDesc {
  D3D12_GPU_DESCRIPTOR_HANDLE InputResource;
  D3D12_GPU_DESCRIPTOR_HANDLE FilterResource;
  D3D12_GPU_DESCRIPTOR_HANDLE BiasResource;
  D3D12_GPU_DESCRIPTOR_HANDLE OutputResource;
  D3D12_GPU_DESCRIPTOR_HANDLE PersistentResource;
  D3D12_GPU_DESCRIPTOR_HANDLE TemporaryResource;
};

}  // namespace lczero
