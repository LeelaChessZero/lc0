// Ankan - test
#define DEFAULT_FP16 true

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
#include <algorithm>
#include <cassert>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <vector>

#include "layers_dx.h"
#include "shader_wrapper.h"
#include "utils/bititer.h"
#include "utils/exception.h"

#include "network_dx.h"

#define DEBUG_RAW_NPS

namespace lczero {

using namespace dx_backend;

void DxContext::flushAndWait() {
  command_list_->Close();
  command_queue_->ExecuteCommandLists(1, (ID3D12CommandList**)&command_list_);
  command_queue_->Signal(fence_, ++fenceVal);

  // Wait for commands to finish on GPU.
  // (spinloop has lowest latency, we can try event based signal if CPU
  // overhead becomes a bottleneck).
  while (fence_->GetCompletedValue() != fenceVal)
    ;

  command_allocator_->Reset();
  command_list_->Reset(command_allocator_, NULL);

  upload_scratch_mem_.offset = 0;
}

void DxContext::dumpFp32(float* buf, int elements) {
  printf("\n");
  for (int i = 0; i < elements; i++) {
    printf("%8.4f ", buf[i]);
    if ((i % 8) == 7) printf("\n");
  }
  printf("\n");
}

void DxContext::copyTensor(DXAlloc dst, DXAlloc src, int bytes) {
  command_list_->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::Transition(
             src.pResource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
             D3D12_RESOURCE_STATE_COPY_SOURCE));

  command_list_->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::Transition(
             dst.pResource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
             D3D12_RESOURCE_STATE_COPY_DEST));

  command_list_->CopyBufferRegion(dst.pResource, dst.offset, src.pResource,
                                  src.offset, bytes);

  command_list_->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::Transition(
             src.pResource, D3D12_RESOURCE_STATE_COPY_SOURCE,
             D3D12_RESOURCE_STATE_UNORDERED_ACCESS));

  command_list_->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::Transition(
             dst.pResource, D3D12_RESOURCE_STATE_COPY_DEST,
             D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
}

void DxContext::dumpCpuTensor(void* data, int size, bool fp16,
                              bool allnewline) {
  printf("\n");
  float* fp32arr = (float*)data;
  uint16_t* arr = (uint16_t*)data;

  for (int i = 0; i < size; i++) {
    printf("%8.4f ", fp16 ? FP16toFP32(arr[i]) : fp32arr[i]);
    if (allnewline || ((i % 8) == 7)) printf("\n");
  }
  printf("\n");
}

void DxContext::dumpTensor(DXAlloc alloc, int size, bool fp16,
                           bool allnewline) {
  int bytes = size * (fp16 ? sizeof(dx_half) : sizeof(float));
  command_list_->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::Transition(
             alloc.pResource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
             D3D12_RESOURCE_STATE_COPY_SOURCE));

  command_list_->CopyBufferRegion(readback_scratch_mem_.pResource,
                                  readback_scratch_mem_.offset, alloc.pResource,
                                  alloc.offset, bytes);

  command_list_->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::Transition(
             alloc.pResource, D3D12_RESOURCE_STATE_COPY_SOURCE,
             D3D12_RESOURCE_STATE_UNORDERED_ACCESS));

  flushAndWait();
  void* cpuPtr;
  readback_scratch_mem_.pResource->Map(0, nullptr, &cpuPtr);
  dumpCpuTensor(cpuPtr, size, fp16, allnewline);
  readback_scratch_mem_.pResource->Unmap(0, nullptr);
}

DxContext::DxContext(const OptionsDict& options) {
  gpu_id_ = options.GetOrDefault<int>("gpu", 0);

  IDXGIFactory4* pFactory = nullptr;
  IDXGIAdapter* pAdapter = nullptr;
  ReportDxErrors(CreateDXGIFactory2(0, IID_PPV_ARGS(&pFactory)));
  ReportDxErrors(pFactory->EnumAdapters(gpu_id_, &pAdapter));
  pFactory->Release();

  if (!pAdapter) throw Exception("Invalid GPU Id: " + std::to_string(gpu_id_));

  ReportDxErrors(D3D12CreateDevice(pAdapter, D3D_FEATURE_LEVEL_11_0,
                                   IID_PPV_ARGS(&device_)));
  pAdapter->Release();

  D3D12_COMMAND_QUEUE_DESC commandqueueDesc;
  commandqueueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
  commandqueueDesc.NodeMask = 0;
  commandqueueDesc.Priority = 0;
  commandqueueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  ReportDxErrors(device_->CreateCommandQueue(&commandqueueDesc,
                                             IID_PPV_ARGS(&command_queue_)));

  ReportDxErrors(device_->CreateCommandAllocator(
      D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&command_allocator_)));

  ReportDxErrors(device_->CreateCommandList(1, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                            command_allocator_, NULL,
                                            IID_PPV_ARGS(&command_list_)));

  D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
  heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
  heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
  heapDesc.NumDescriptors = kNumDescHeapSlots;
  ReportDxErrors(
      device_->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&desc_heap_)));

  next_slot_in_desc_heap_ = 0;

  fenceVal = 0ull;
  ReportDxErrors(device_->CreateFence(fenceVal, D3D12_FENCE_FLAG_NONE,
                                      IID_PPV_ARGS(&fence_)));

  shader_wrapper_.init(device_);

  int max_batch_size = options.GetOrDefault<int>("max_batch", 1024);

  // Allocate scratch space

  // 256 MB should be enough for uploading weights, etc.
  scratch_size_ = 256 * 1024 * 1024;
  CreateAlloc(scratch_size_, D3D12_HEAP_TYPE_UPLOAD, upload_scratch_mem_);
  CreateAlloc(scratch_size_, D3D12_HEAP_TYPE_READBACK, readback_scratch_mem_);
}

DxContext::~DxContext() {
  // Make sure nothing is in flight
  flushAndWait();

  upload_scratch_mem_.pResource->Release();
  readback_scratch_mem_.pResource->Release();

  shader_wrapper_.destroy();
  command_list_->Release();
  command_allocator_->Release();
  command_queue_->Release();
  fence_->Release();
  desc_heap_->Release();
  device_->Release();
}

void DxContext::CreateAlloc(size_t size, D3D12_HEAP_TYPE type, DXAlloc& alloc) {
  // some alignment
  int factor = DivUp(size, 4);
  size = factor * 4;

  D3D12_HEAP_PROPERTIES heapDesc = {};
  heapDesc.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
  heapDesc.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
  heapDesc.CreationNodeMask = 1;
  heapDesc.VisibleNodeMask = 1;

  if (type == D3D12_HEAP_TYPE_CUSTOM) {
    // Use custom heap type to allow GPU writing to system memory directly
    heapDesc.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
    heapDesc.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
  }

  heapDesc.Type = type;

  D3D12_RESOURCE_DESC bufferDesc = {};
  bufferDesc.MipLevels = 1;
  bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
  bufferDesc.Height = 1;
  if (type == D3D12_HEAP_TYPE_DEFAULT || type == D3D12_HEAP_TYPE_CUSTOM)
    bufferDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  bufferDesc.DepthOrArraySize = 1;
  bufferDesc.SampleDesc.Count = 1;
  bufferDesc.SampleDesc.Quality = 0;
  bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

  D3D12_RESOURCE_STATES resourceState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  if (type == D3D12_HEAP_TYPE_UPLOAD)
    resourceState = D3D12_RESOURCE_STATE_GENERIC_READ;
  else if (type == D3D12_HEAP_TYPE_READBACK)
    resourceState = D3D12_RESOURCE_STATE_COPY_DEST;

  bufferDesc.Width = size;
  ReportDxErrors(device_->CreateCommittedResource(
      &heapDesc, D3D12_HEAP_FLAG_NONE, &bufferDesc, resourceState, nullptr,
      IID_PPV_ARGS(&alloc.pResource)));

  alloc.offset = 0;
  alloc.gpuVA = alloc.pResource->GetGPUVirtualAddress();

  // Create desc heap entry for UAV resources.
  if (resourceState == D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
    int slot = next_slot_in_desc_heap_++;

    int handleIncrementSize = device_->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    CD3DX12_CPU_DESCRIPTOR_HANDLE cpuDescHandle(
        desc_heap_->GetCPUDescriptorHandleForHeapStart(), slot,
        handleIncrementSize);

    CD3DX12_GPU_DESCRIPTOR_HANDLE gpuDescHandle(
        desc_heap_->GetGPUDescriptorHandleForHeapStart(), slot,
        handleIncrementSize);

    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uavDesc.Format = DXGI_FORMAT_R32_TYPELESS;
    uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;

    uavDesc.Buffer.FirstElement = 0;
    uavDesc.Buffer.NumElements = size / 4;

    device_->CreateUnorderedAccessView(alloc.pResource, nullptr, &uavDesc,
                                       cpuDescHandle);

    alloc.descHandle = gpuDescHandle;
  }
}

void DxContext::scheduleUpload(DXAlloc alloc, void* data, size_t size) {
  // Make sure enough space is available in the upload scratch buffer
  assert(size <= scratch_size_);
  if (upload_scratch_mem_.offset + size > scratch_size_) flushAndWait();

  uint8_t* temp;
  upload_scratch_mem_.pResource->Map(0, nullptr, (void**)&temp);

  dx_half* cpuPtr = (dx_half*)(temp + upload_scratch_mem_.offset);
  memcpy(cpuPtr, data, size);

  command_list_->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::Transition(
             alloc.pResource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
             D3D12_RESOURCE_STATE_COPY_DEST));

  command_list_->CopyBufferRegion(alloc.pResource, alloc.offset,
                                  upload_scratch_mem_.pResource,
                                  upload_scratch_mem_.offset, size);

  command_list_->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::Transition(
             alloc.pResource, D3D12_RESOURCE_STATE_COPY_DEST,
             D3D12_RESOURCE_STATE_UNORDERED_ACCESS));

  upload_scratch_mem_.pResource->Unmap(0, nullptr);

  // reset at flush and wait
  upload_scratch_mem_.offset += size;
}

DxNetwork::DxNetwork(const WeightsFile& file, const OptionsDict& options)
    : dx_context_(options),
      capabilities_{file.format().network_format().input()} {
  printf("\ncreating DX backend!\n");  // Ankan - for testing
  LegacyWeights weights(file.weights());

  has_conv_policy_ = file.format().network_format().policy() ==
                     pblczero::NetworkFormat::POLICY_CONVOLUTION;
  max_batch_size_ = options.GetOrDefault<int>("max_batch", 1024);

  // Ankan - control fp16 vs fp32
  fp16_ = options.GetOrDefault<bool>("fp16", DEFAULT_FP16);

  const int kNumInputPlanes = kInputPlanes;
  const int kNumFilters = weights.input.biases.size();

  numBlocks_ = weights.residual.size();
  has_se_ = weights.residual[0].has_se;

  // 2. Build the network, and copy the weights to GPU memory.

  // Unique GEMMs for winograd required by the network.
  input_conv_winograd_gemm_ = new GemmMetaCommand(
      &dx_context_, 0, kNumFilters, kNumInputPlanes, 36, fp16_, false, false);

  residual_block_winograd_gemm_ = new GemmMetaCommand(
      &dx_context_, 0, kNumFilters, kNumFilters, 36, fp16_, false, false);

  auto pol_channels = weights.policy.biases.size();
  if (has_conv_policy_) {
    policy_conv_winograd_gemm_ = new GemmMetaCommand(
        &dx_context_, 0, kNumFilters, pol_channels, 36, fp16_, false, false);
  }

  // input
  {
    auto inputConv = std::make_unique<ConvLayer>(
        fp16_, input_conv_winograd_gemm_, &dx_context_, nullptr, kNumFilters, 8,
        8, 3, kNumInputPlanes, true, true);

    inputConv->LoadWeights(&weights.input.weights[0], &weights.input.biases[0],
                           &dx_context_);

    network_.emplace_back(std::move(inputConv));
  }

  // residual block
  for (size_t block = 0; block < weights.residual.size(); block++) {
    auto conv1 = std::make_unique<ConvLayer>(
        fp16_, residual_block_winograd_gemm_, &dx_context_, getLastLayer(),
        kNumFilters, 8, 8, 3, kNumFilters, true, true);

    conv1->LoadWeights(&weights.residual[block].conv1.weights[0],
                       &weights.residual[block].conv1.biases[0], &dx_context_);

    network_.emplace_back(std::move(conv1));

    int se_k = 0;
    if (has_se_) se_k = weights.residual[block].se.b1.size();

    auto conv2 = std::make_unique<ConvLayer>(
        fp16_, residual_block_winograd_gemm_, &dx_context_, getLastLayer(),
        kNumFilters, 8, 8, 3, kNumFilters, true, true, true, has_se_, se_k);

    conv2->LoadWeights(&weights.residual[block].conv2.weights[0],
                       &weights.residual[block].conv2.biases[0], &dx_context_);

    if (has_se_) {
      conv2->LoadSEWeights(
          &weights.residual[block].se.w1[0], &weights.residual[block].se.b1[0],
          &weights.residual[block].se.w2[0], &weights.residual[block].se.b2[0]);
    }
    network_.emplace_back(std::move(conv2));
  }

  resi_last_ = getLastLayer();

  // policy head
  if (has_conv_policy_) {
    // conv1 is same as residual block convolution.
    auto conv1 = std::make_unique<ConvLayer>(
        fp16_, residual_block_winograd_gemm_, &dx_context_, getLastLayer(),
        kNumFilters, 8, 8, 3, kNumFilters, true, true);
    conv1->LoadWeights(&weights.policy1.weights[0], &weights.policy1.biases[0],
                       &dx_context_);
    network_.emplace_back(std::move(conv1));

    // conv2 has different no. of output filters (pol_channels). No relu.
    auto conv2 = std::make_unique<ConvLayer>(
        fp16_, policy_conv_winograd_gemm_, &dx_context_, getLastLayer(),
        pol_channels, 8, 8, 3, kNumFilters, false, true);

    conv2->LoadWeights(&weights.policy.weights[0], &weights.policy.biases[0],
                       &dx_context_);
    network_.emplace_back(std::move(conv2));

    // Policy map layer, evaluated on CPU

  } else {
    // 1x1 convolution, pol_channels output filters
    auto convPol = std::make_unique<ConvLayer>(fp16_, nullptr, &dx_context_,
                                               getLastLayer(), pol_channels, 8,
                                               8, 1, kNumFilters, true, true);
    convPol->LoadWeights(&weights.policy.weights[0], &weights.policy.biases[0],
                         &dx_context_);
    network_.emplace_back(std::move(convPol));

    // FC with bias, no activation
    // pad up kNumOutputPolicy to be a multiple of 8
    assert(weights.ip_pol_b.size() == kNumOutputPolicy);
    auto FCPol = std::make_unique<FCLayer>(fp16_, &dx_context_, getLastLayer(),
                                           kNumOutputPolicyPadded8, 1, 1, true,
                                           false, false);
    // Copy weights to temp space which is padded in size.
    std::vector<float> tempBias(kNumOutputPolicyPadded8);
    std::vector<float> tempWeight(kNumOutputPolicyPadded8 *
                                  weights.ip_pol_w.size() / kNumOutputPolicy);
    memcpy(tempBias.data(), weights.ip_pol_b.data(),
           weights.ip_pol_b.size() * sizeof(float));
    memcpy(tempWeight.data(), weights.ip_pol_w.data(),
           weights.ip_pol_w.size() * sizeof(float));

    FCPol->LoadWeights(tempWeight.data(), tempBias.data(), &dx_context_);
    network_.emplace_back(std::move(FCPol));
  }
  policy_out_ = getLastLayer();

  // value head
  {
    auto val_channels = weights.value.biases.size();

    // 1x1 convolution, val_channels output filters
    auto convVal = std::make_unique<ConvLayer>(fp16_, nullptr, &dx_context_,
                                               getLastLayer(), val_channels, 8,
                                               8, 1, kNumFilters, true, true);
    convVal->LoadWeights(&weights.value.weights[0], &weights.value.biases[0],
                         &dx_context_);
    network_.emplace_back(std::move(convVal));

    // Bias and relu activation.
    auto FCVal1 = std::make_unique<FCLayer>(fp16_, &dx_context_, getLastLayer(),
                                            weights.ip1_val_b.size(), 1, 1,
                                            true, true, false);
    FCVal1->LoadWeights(&weights.ip1_val_w[0], &weights.ip1_val_b[0],
                        &dx_context_);
    network_.emplace_back(std::move(FCVal1));

    has_wdl_ = file.format().network_format().value() ==
               pblczero::NetworkFormat::VALUE_WDL;

    // Fully connected layer with Bias.
    // tanh activation for non wdl nets, no activation for wdl.
    auto fc2_tanh = !has_wdl_;
    auto FCVal2 = std::make_unique<FCLayer>(fp16_, &dx_context_, getLastLayer(),
                                            kNumOutputValuePadded8, 1, 1, true,
                                            false, fc2_tanh);
    // Pad up the weights
    std::vector<float> tempBias(kNumOutputValuePadded8);
    std::vector<float> tempWeight(kNumOutputValuePadded8 *
                                  weights.ip2_val_w.size() /
                                  weights.ip2_val_b.size());
    memcpy(tempBias.data(), weights.ip2_val_b.data(),
           weights.ip2_val_b.size() * sizeof(float));
    memcpy(tempWeight.data(), weights.ip2_val_w.data(),
           weights.ip2_val_w.size() * sizeof(float));
    FCVal2->LoadWeights(tempWeight.data(), tempBias.data(), &dx_context_);
    network_.emplace_back(std::move(FCVal2));
    if (has_wdl_) {
      // Softmax layer on CPU
    }
  }
  value_out_ = getLastLayer();

  dx_context_.flushAndWait();

  // 3. Allocate GPU memory for running the network
  //    - 4 buffers of max size are enough:
  //       * one to hold input,
  //       * second to hold output
  //       * third to hold skip connection's input
  //       * and fourth to act as scratch space needed by some layers
  size_t max_size = resi_last_->GetOutputSize(max_batch_size_);

  // Winograd transformed inputs/outputs need more space.
  // Every 4x4 block of input/output is transfored to 6x6 block.
  max_size *= 36.0 / 16.0;

  // max_size = 256 * 1024 * 1024; // Ankan for testing! 256 MB!

  for (auto& mem : tensor_mem_) {
    dx_context_.CreateAlloc(max_size, D3D12_HEAP_TYPE_DEFAULT, mem);
  }
}

void DxNetwork::forwardEval(InputsOutputsDx* io, int batchSize) {
  lock_.lock();

#ifdef DEBUG_RAW_NPS
  auto t_start = std::chrono::high_resolution_clock::now();
#endif

  // Expand packed board representation into full planes.
  dx_context_.getShaderWrapper()->expandPlanes(
      dx_context_.getCommandList(), tensor_mem_[0], io->input_masks_mem_gpu_,
      io->input_val_mem_gpu_, batchSize, fp16_);

  dx_context_.getCommandList()->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::UAV(nullptr));

  // Ankan - for testing!
  //printf("\nAfter expand planes");
  //dx_context_.dumpTensor(tensor_mem_[0], 1024, fp16_);

  int l = 0;
  // Input Conv
  network_[l++]->Eval(batchSize, tensor_mem_[2], tensor_mem_[0], DXAlloc(),
                      tensor_mem_[1], tensor_mem_[3],
                      dx_context_.getCommandList());

  dx_context_.getCommandList()->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::UAV(nullptr));

  //printf("\nAfter input conv");
  //dx_context_.dumpTensor(tensor_mem_[2], 1024, fp16_);

  // Residual tower.
  for (int block = 0; block < numBlocks_; block++) {
    // conv1
    network_[l++]->Eval(batchSize, tensor_mem_[0], tensor_mem_[2], DXAlloc(),
                        tensor_mem_[1], tensor_mem_[3],
                        dx_context_.getCommandList());

    dx_context_.getCommandList()->ResourceBarrier(
        1, &CD3DX12_RESOURCE_BARRIER::UAV(nullptr));

    // conv2
    network_[l++]->Eval(batchSize, tensor_mem_[2], tensor_mem_[0],
                        tensor_mem_[2], tensor_mem_[1], tensor_mem_[3],
                        dx_context_.getCommandList());

    dx_context_.getCommandList()->ResourceBarrier(
        1, &CD3DX12_RESOURCE_BARRIER::UAV(nullptr));

      // Ankan - test!
    /*
      if (block == 0) {
        printf("\nAfter conv1");
        dx_context_.dumpTensor(tensor_mem_[0], 1024, fp16_);

        printf("\nAfter conv2");
        dx_context_.dumpTensor(tensor_mem_[2], 1024, fp16_);

        exit(0);
      }
    */
  }

  // printf("\nAfter residual tower");
  // dx_context_.dumpTensor(tensor_mem_[2], 1024, fp16_);

  //-----------------------------------///---------------------------------------

  // Policy head.
  if (has_conv_policy_) {
    // Policy conv1.
    network_[l++]->Eval(batchSize, tensor_mem_[0], tensor_mem_[2], DXAlloc(),
                        tensor_mem_[1], tensor_mem_[3],
                        dx_context_.getCommandList());

    dx_context_.getCommandList()->ResourceBarrier(
        1, &CD3DX12_RESOURCE_BARRIER::UAV(nullptr));

    // Policy conv2 (writes directly to system memory).
    network_[l++]->Eval(batchSize, io->op_policy_mem_gpu_, tensor_mem_[0],
                        DXAlloc(), tensor_mem_[1], tensor_mem_[3],
                        dx_context_.getCommandList());

    // TODO! Policy map layer - run on CPU!
    // read op_policy_mem_, write to op_policy_mem_final_
  } else {
    // Policy conv.
    network_[l++]->Eval(batchSize, tensor_mem_[0], tensor_mem_[2], DXAlloc(),
                        tensor_mem_[1], tensor_mem_[3],
                        dx_context_.getCommandList());

    dx_context_.getCommandList()->ResourceBarrier(
        1, &CD3DX12_RESOURCE_BARRIER::UAV(nullptr));

    // printf("\nAfter policy conv");
    // dx_context_.dumpTensor(tensor_mem_[0], 1024, fp16_);

    // Policy FC (writes directly to system memory).
    network_[l++]->Eval(batchSize, io->op_policy_mem_gpu_, tensor_mem_[0],
                        DXAlloc(), tensor_mem_[1], tensor_mem_[3],
                        dx_context_.getCommandList());

    //dx_context_.dumpTensor(io->op_policy_mem_gpu_, 2048*batchSize, fp16_);

  }

  // Value head.

  // Value conv.
  network_[l++]->Eval(batchSize, tensor_mem_[0], tensor_mem_[2], DXAlloc(),
                      tensor_mem_[1], tensor_mem_[3],
                      dx_context_.getCommandList());

  dx_context_.getCommandList()->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::UAV(nullptr));

  // printf("\nAfter value conv");
  // dx_context_.dumpTensor(tensor_mem_[0], 1024, fp16_);

  // value FC1.
  network_[l++]->Eval(batchSize, tensor_mem_[1], tensor_mem_[0], DXAlloc(),
                      tensor_mem_[2], tensor_mem_[3],
                      dx_context_.getCommandList());

  dx_context_.getCommandList()->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::UAV(nullptr));

  // printf("\nAfter value fc1");
  // dx_context_.dumpTensor(tensor_mem_[1], 1024, fp16_);

  // value FC2.
  network_[l++]->Eval(batchSize, io->op_value_mem_gpu_, tensor_mem_[1],
                      DXAlloc(), tensor_mem_[2], tensor_mem_[3],
                      dx_context_.getCommandList());
  /*
  if (batchSize > 1)
  {
   printf("\nAfter value fc2, batch size of %d", batchSize);
   dx_context_.dumpTensor(io->op_value_mem_gpu_, 128, fp16_);
   exit(0);
  }
  */

  if (has_wdl_) {
    // Value softmax
    // TODO! Need to do softmax on CPU (in place over io->op_value_mem_ to
    // io->op_value_mem_final_)
  }

  // TODO: measure time from start to this point to get an idea of CPU side
  // overhead in recording command list
#ifdef DEBUG_RAW_NPS
  auto t_cpu_end = std::chrono::high_resolution_clock::now();
#endif

  dx_context_.flushAndWait();
  lock_.unlock();

  // Do the value head softmax and policy map layers on CPU.
  // Can do it outside the lock to get some more parallelism.
  int val_vector_size = has_wdl_ ? 3 : 1;
  if (fp16_) {

    // Policy:
    if (has_conv_policy_) {
      // run policy map layer
    } else {
      // Un-pad policy output, and convert to fp32.
      dx_half* padded_pol_fp16 = (dx_half*)io->op_policy_mem_;
      for (int n = 0; n < batchSize; n++)
        for (int i = 0; i < kNumOutputPolicy; i++)
          io->op_policy_mem_final_[n * kNumOutputPolicy + i] =
              FP16toFP32(padded_pol_fp16[n * kNumOutputPolicyPadded8 + i]);
    }

    // Value:
    // Un-pad value output, converting it to fp32.
    dx_half* padded_val_fp16 = (dx_half*)io->op_value_mem_;
    for (int n = 0; n < batchSize; n++)
      for (int i = 0; i < val_vector_size; i++)
        io->op_value_mem_final_[n * val_vector_size + i] =
            FP16toFP32(padded_val_fp16[n * kNumOutputValuePadded8 + i]);

  } else {
    // Policy:
    if (has_conv_policy_) {
      // Run policy map layer.
    } else {
      // Un-pad policy output.
      for (int i = 0; i < batchSize; i++)
        memcpy(io->op_policy_mem_final_ + kNumOutputPolicy * i,
               io->op_policy_mem_ + kNumOutputPolicyPadded8 * i,
               kNumOutputPolicy * sizeof(float));
    }

    // Value:
    // Un-pad value output.
    for (int i = 0; i < batchSize; i++)
      memcpy(io->op_value_mem_final_ + val_vector_size * i,
             io->op_value_mem_ + kNumOutputValuePadded8 * i,
             val_vector_size * sizeof(float));
  }

  /*
  if (batchSize > 1) {
    // Ankan - test!
    printf("\nValue out: ");
    dx_context_.dumpCpuTensor(io->op_value_mem_final_, batchSize, false);

    //printf("\npolicy out: ");
    //dx_context_.dumpCpuTensor(io->op_policy_mem_final_, batchSize * kNumOutputPolicy, false);

    exit(0);
  }
  */


#ifdef DEBUG_RAW_NPS
  const int reportingCalls = 100;
  static int numCalls = 0;
  static int sumBatchSize = 0;
  static double totalTime = 0;

  sumBatchSize += batchSize;
  numCalls++;

  auto t_end = std::chrono::high_resolution_clock::now();
  // auto t_end = t_cpu_end;

  double dt = std::chrono::duration<double>(t_end - t_start).count();
  totalTime += dt;
  if (numCalls == reportingCalls) {
    double avgBatchSize = ((double)sumBatchSize) / numCalls;
    double nps = sumBatchSize / totalTime;
    printf(
        "\nAvg batch size: %lf, NN eval time: %lf seconds per %d evals. "
        "NPS: "
        "%g\n",
        avgBatchSize, totalTime, sumBatchSize, nps);
    sumBatchSize = 0;
    totalTime = 0;
    numCalls = 0;
  }
#endif
}

DxNetwork::~DxNetwork() {
  dx_context_.flushAndWait();
  // Free memory and destroy all dx objects.
  for (auto mem : tensor_mem_) {
    mem.pResource->Release();
  }
}

std::unique_ptr<NetworkComputation> DxNetwork::NewComputation() {
  // TODO: figure out if we need to set correct GPU id here ?
  return std::make_unique<DxNetworkComputation>(this, has_wdl_);
}

std::unique_ptr<InputsOutputsDx> DxNetwork::GetInputsOutputs() {
  std::lock_guard<std::mutex> lock(inputs_outputs_lock_);
  if (free_inputs_outputs_.empty()) {
    return std::make_unique<InputsOutputsDx>(max_batch_size_, &dx_context_,
                                             has_wdl_);
  } else {
    std::unique_ptr<InputsOutputsDx> resource =
        std::move(free_inputs_outputs_.front());
    free_inputs_outputs_.pop_front();
    return resource;
  }
}

void DxNetwork::ReleaseInputsOutputs(
    std::unique_ptr<InputsOutputsDx> resource) {
  std::lock_guard<std::mutex> lock(inputs_outputs_lock_);
  free_inputs_outputs_.push_back(std::move(resource));
}

DxNetworkComputation::DxNetworkComputation(DxNetwork* network, bool wdl)
    : network_(network), wdl_(wdl) {
  batch_size_ = 0;
  inputs_outputs_ = network_->GetInputsOutputs();
}

DxNetworkComputation::~DxNetworkComputation() {
  network_->ReleaseInputsOutputs(std::move(inputs_outputs_));
}

void DxNetworkComputation::AddInput(InputPlanes&& input) {
  auto iter_mask =
      &inputs_outputs_->input_masks_mem_[batch_size_ * kInputPlanes];
  auto iter_val = &inputs_outputs_->input_val_mem_[batch_size_ * kInputPlanes];

  int i = 0;
  for (const auto& plane : input) {
    iter_mask[i] = plane.mask;
    iter_val[i] = plane.value;
    i++;
  }

  batch_size_++;
}

void DxNetworkComputation::ComputeBlocking() {
  network_->forwardEval(inputs_outputs_.get(), GetBatchSize());
}

InputsOutputsDx::InputsOutputsDx(int maxBatchSize, DxContext* pContext,
                                 bool wdl) {
  // CPU accesses on Default heap doesn't work.
  // GPU accesses on Upload heap works.
  pContext->CreateAlloc(maxBatchSize * kInputPlanes * sizeof(uint64_t),
                        D3D12_HEAP_TYPE_UPLOAD /*D3D12_HEAP_TYPE_DEFAULT*/,
                        input_masks_mem_gpu_);

  pContext->CreateAlloc(maxBatchSize * kInputPlanes * sizeof(float),
                        D3D12_HEAP_TYPE_UPLOAD /*D3D12_HEAP_TYPE_DEFAULT*/,
                        input_val_mem_gpu_);

  // CUSTOM heap created to have GPU directly write to system memory
  pContext->CreateAlloc(maxBatchSize * kNumOutputPolicyPadded8 * sizeof(float),
                        D3D12_HEAP_TYPE_CUSTOM, op_policy_mem_gpu_);

  pContext->CreateAlloc(maxBatchSize * kNumOutputValuePadded8 * sizeof(float),
                        D3D12_HEAP_TYPE_CUSTOM, op_value_mem_gpu_);

  ReportDxErrors(input_masks_mem_gpu_.pResource->Map(
      0, nullptr, (void**)&input_masks_mem_));

  ReportDxErrors(
      input_val_mem_gpu_.pResource->Map(0, nullptr, (void**)&input_val_mem_));

  ReportDxErrors(
      op_policy_mem_gpu_.pResource->Map(0, nullptr, (void**)&op_policy_mem_));

  ReportDxErrors(
      op_value_mem_gpu_.pResource->Map(0, nullptr, (void**)&op_value_mem_));

  op_policy_mem_final_ = new float[maxBatchSize * kNumOutputPolicy];
  op_value_mem_final_ = new float[maxBatchSize * (wdl ? 3 : 1)];
}

InputsOutputsDx::~InputsOutputsDx() {
  input_masks_mem_gpu_.pResource->Unmap(0, nullptr);
  input_val_mem_gpu_.pResource->Unmap(0, nullptr);
  op_policy_mem_gpu_.pResource->Unmap(0, nullptr);
  op_value_mem_gpu_.pResource->Unmap(0, nullptr);

  input_masks_mem_gpu_.pResource->Release();
  input_val_mem_gpu_.pResource->Release();
  op_policy_mem_gpu_.pResource->Release();
  op_value_mem_gpu_.pResource->Release();

  delete[] op_policy_mem_final_;
  delete[] op_value_mem_final_;
}

std::unique_ptr<Network> MakeDxNetwork(const WeightsFile& weights,
                                       const OptionsDict& options) {
  return std::make_unique<DxNetwork>(weights, options);
}

REGISTER_NETWORK("dx", MakeDxNetwork, 120)

}  // namespace lczero
