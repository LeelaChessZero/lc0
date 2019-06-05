#pragma once

#include <cstdint>

// Can be obtained by using EnumerateMetaCommands and 
// EnumerateMetaCommandParameters calls

GUID ConvGuid = {
0x17804d6b,
0xebfe,
0x426f,
{0x88, 0xfc, 0xfe, 0xa7, 0x2e, 0x3f, 0x33, 0x56}};


struct TensorDesc
{
  uint64_t DataType;
  uint64_t Flags;
  uint64_t DimensionCount;
  uint64_t Size[5];
  uint64_t Stride[5];
  uint64_t StrideAlignment[5];
  uint64_t BaseAlignmentInBytes;
  uint64_t PhysicalSizeInElements;
};

struct ConvCreateDesc
{
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
