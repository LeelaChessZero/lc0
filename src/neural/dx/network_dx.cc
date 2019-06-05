/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

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

void DxContext::dumpTensor(DXAlloc alloc, int bytes, bool fp16,
                           bool allnewline) {
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
  printf("\n");

  float* fp32arr = (float*)cpuPtr;
  uint16_t* arr = (uint16_t*)cpuPtr;
  int size = bytes / sizeof(float);
  if (fp16) size *= 2;

  for (int i = 0; i < size; i++) {
    printf("%8.4f ", fp16 ? FP16toFP32(arr[i]) : fp32arr[i]);
    if (allnewline || ((i % 8) == 7)) printf("\n");
  }
  printf("\n");
  readback_scratch_mem_.pResource->Unmap(0, nullptr);
}

DxContext::DxContext(const OptionsDict& options) {
  gpu_id_ = options.GetOrDefault<int>("gpu", 0);
  // TODO: find a way to create device on the correct adapter
  // int total_gpus;
  // ReportCUDAErrors(cudaGetDeviceCount(&total_gpus));
  // if (gpu_id_ >= total_gpus)
  //  throw Exception("Invalid GPU Id: " + std::to_string(gpu_id_));

  // Select GPU to run on (for *the current* thread).
  // ReportCUDAErrors(cudaSetDevice(gpu_id_));

  // TODO: create adapter on the gpu we have been asked for!
  ReportDxErrors(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                                   IID_PPV_ARGS(&device_)));

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


  // Allocate scratch space for datatype/layout conversion for weights).
  const int scratchSize = 128 * 1024 * 1024;
  CreateAlloc(scratchSize, D3D12_HEAP_TYPE_UPLOAD, &upload_scratch_mem_);
  CreateAlloc(scratchSize, D3D12_HEAP_TYPE_READBACK, &readback_scratch_mem_);
  CreateAlloc(scratchSize*2, D3D12_HEAP_TYPE_DEFAULT, &default_scratch_mem_);

}

DxContext::~DxContext() {
  // Make sure nothing is in flight
  flushAndWait();

  upload_scratch_mem_.pResource->Release();
  readback_scratch_mem_.pResource->Release();
  default_scratch_mem_.pResource->Release();

  shader_wrapper_.destroy();
  command_list_->Release();
  command_allocator_->Release();
  command_queue_->Release();
  fence_->Release();
  desc_heap_->Release();
  device_->Release();
}

void DxContext::CreateAlloc(size_t size, D3D12_HEAP_TYPE type,
                            DXAlloc* pAlloc) {
  // some alignment
  int factor = DivUp(size, 4);
  size = factor * 4;

  D3D12_HEAP_PROPERTIES heapDesc = {};
  heapDesc.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
  heapDesc.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
  // TODO: do we need any changes on multi-gpu systems?
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
      IID_PPV_ARGS(&pAlloc->pResource)));

  pAlloc->offset = 0;
  pAlloc->gpuVA = pAlloc->pResource->GetGPUVirtualAddress();

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

    device_->CreateUnorderedAccessView(pAlloc->pResource, nullptr, &uavDesc,
                                       cpuDescHandle);

    pAlloc->descHandle = gpuDescHandle;
  }
}

void DxContext::scheduleUpload(DXAlloc alloc, void* data, size_t size) {
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
    : dx_context_(options) {
  printf("\ncreating DX backend!\n");  // Ankan - for testing
  LegacyWeights weights(file.weights());

  has_conv_policy_ = file.format().network_format().policy() ==
                     pblczero::NetworkFormat::POLICY_CONVOLUTION;
  max_batch_size_ = options.GetOrDefault<int>("max_batch", 1024);

  const int kNumInputPlanes = kInputPlanes;
  const int kNumFilters = weights.input.biases.size();

  numBlocks_ = weights.residual.size();
  has_se_ = weights.residual[0].has_se;

  // 2. Build the network, and copy the weights to GPU memory.

  // unique convolutions required by the network
  input_conv_ = new ConvMetaCommand(&dx_context_, kNumInputPlanes, kNumFilters,
                                    8, 8, 3, false, true, true);

  resi_conv_1_ = new ConvMetaCommand(&dx_context_, kNumFilters, kNumFilters, 8,
                                     8, 3, false, true, true);

  resi_conv_2_ = new ConvMetaCommand(&dx_context_, kNumFilters, kNumFilters, 8,
                                     8, 3, true, true, true);

  assert(weights.policy.biases.size() == weights.value.biases.size());
  head_conv_ = new ConvMetaCommand(&dx_context_, kNumFilters,
                                   weights.policy.biases.size(), 8, 8, 1, false,
                                   true, true);

  // input
  {
    auto inputConv = std::make_unique<ConvLayer>(input_conv_, &dx_context_,
                                                 nullptr, kNumFilters, 8, 8, 3,
                                                 kNumInputPlanes, true, true);

    inputConv->LoadWeights(&weights.input.weights[0], &weights.input.biases[0],
                           &dx_context_);

    network_.emplace_back(std::move(inputConv));
  }

  // residual block
  for (size_t block = 0; block < weights.residual.size(); block++) {
    auto conv1 = std::make_unique<ConvLayer>(resi_conv_1_, &dx_context_,
                                             getLastLayer(), kNumFilters, 8, 8,
                                             3, kNumFilters, true, true);

    conv1->LoadWeights(&weights.residual[block].conv1.weights[0],
                       &weights.residual[block].conv1.biases[0], &dx_context_);

    network_.emplace_back(std::move(conv1));

    auto conv2 = std::make_unique<ConvLayer>(resi_conv_2_, &dx_context_,
                                             getLastLayer(), kNumFilters, 8, 8,
                                             3, kNumFilters, true, true, true);

    conv2->LoadWeights(&weights.residual[block].conv2.weights[0],
                       &weights.residual[block].conv2.biases[0], &dx_context_);

    network_.emplace_back(std::move(conv2));
  }

  resi_last_ = getLastLayer();

  // policy head
  {
    auto pol_channels = weights.policy.biases.size();
    auto convPol = std::make_unique<ConvLayer>(head_conv_, &dx_context_,
                                               resi_last_, pol_channels, 8, 8,
                                               1, kNumFilters, true, true);
    convPol->LoadWeights(&weights.policy.weights[0], &weights.policy.biases[0],
                         &dx_context_);
    network_.emplace_back(std::move(convPol));

    auto FCPol = std::make_unique<FCLayer>(
        &dx_context_, getLastLayer(), weights.ip_pol_b.size(), 1, 1, true,
        false, false, true, true);  // sigmoid activation, fp32 out
    FCPol->LoadWeights(&weights.ip_pol_w[0], &weights.ip_pol_b[0],
                       &dx_context_);
    network_.emplace_back(std::move(FCPol));
  }
  policy_out_ = getLastLayer();

  // value head
  {
    auto convVal = std::make_unique<ConvLayer>(
        head_conv_, &dx_context_, resi_last_, weights.value.biases.size(), 8, 8,
        1, kNumFilters, true, true);
    convVal->LoadWeights(&weights.value.weights[0], &weights.value.biases[0],
                         &dx_context_);
    network_.emplace_back(std::move(convVal));

    auto FCVal1 = std::make_unique<FCLayer>(
        &dx_context_, getLastLayer(), weights.ip1_val_b.size(), 1, 1, true,
        true, false, false, false);  // relu activation, fp16 out
    FCVal1->LoadWeights(&weights.ip1_val_w[0], &weights.ip1_val_b[0],
                        &dx_context_);
    network_.emplace_back(std::move(FCVal1));

    auto FCVal2 = std::make_unique<FCLayer>(&dx_context_, getLastLayer(), 1, 1,
                                            1, true, false, true, false,
                                            true);  // tanh activation, fp32 out
    FCVal2->LoadWeights(&weights.ip2_val_w[0], &weights.ip2_val_b[0],
                        &dx_context_);
    network_.emplace_back(std::move(FCVal2));
  }
  value_out_ = getLastLayer();

  dx_context_.flushAndWait();

  // 3. allocate GPU memory for running the network
  //    - three buffers of max size are enough (one to hold input, second to
  //    hold output and third to hold skip connection's input).
  size_t maxSize = resi_last_->GetOutputSize(max_batch_size_);

  for (auto& mem : tensor_mem_) {
    dx_context_.CreateAlloc(maxSize, D3D12_HEAP_TYPE_DEFAULT, &mem);
  }
}

void DxNetwork::forwardEval(InputsOutputsDx* io, int batchSize) {
  std::lock_guard<std::mutex> lock(lock_);

#ifdef DEBUG_RAW_NPS
  auto t_start = std::chrono::high_resolution_clock::now();
#endif

  // expand packed planes to full planes
  dx_context_.getShaderWrapper()->expandPlanes(
      dx_context_.getCommandList(), tensor_mem_[0], io->input_masks_mem_gpu_,
      io->input_val_mem_gpu_, batchSize);

  // printf("\nAfter expand planes");
  // dumpTensor(tensor_mem_[0], 1024);

  dx_context_.getCommandList()->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::UAV(tensor_mem_[0].pResource));

  int l = 0;
  // input
  network_[l++]->Eval(batchSize, &tensor_mem_[2], &tensor_mem_[0], nullptr,
                      dx_context_.getCommandList());  // input conv

  // printf("\nAfter input conv");
  // dumpTensor(tensor_mem_[2], 1024);

  dx_context_.getCommandList()->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::UAV(tensor_mem_[2].pResource));

  // residual blocks
  for (int block = 0; block < numBlocks_; block++) {
    network_[l++]->Eval(batchSize, &tensor_mem_[0], &tensor_mem_[2], nullptr,
                        dx_context_.getCommandList());  // conv1

    dx_context_.getCommandList()->ResourceBarrier(
        1, &CD3DX12_RESOURCE_BARRIER::UAV(tensor_mem_[0].pResource));

    network_[l++]->Eval(batchSize, &tensor_mem_[2], &tensor_mem_[0],
                        &tensor_mem_[2],
                        dx_context_.getCommandList());  // conv2

    //printf("\nAfter conv2");
    //dx_context_.dumpTensor(tensor_mem_[2], 1024);
    //exit(0);

    dx_context_.getCommandList()->ResourceBarrier(
        1, &CD3DX12_RESOURCE_BARRIER::UAV(tensor_mem_[2].pResource));
  }

  int testBatchSize = 777;
  // printf("\nAfter residual tower");
  // dumpTensor(tensor_mem_[2], 1024);

  //-----------------------------------///---------------------------------------

  // policy head
  network_[l++]->Eval(batchSize, &tensor_mem_[0], &tensor_mem_[2], nullptr,
                      dx_context_.getCommandList());  // pol conv

  // printf("\nAfter policy conv");
  // dumpTensor(tensor_mem_[0], 2048*2);

  dx_context_.getCommandList()->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::UAV(tensor_mem_[0].pResource));

  // Writes directly fp32

  // shader writes directly to system memory
  /*
  network_[l++]->Eval(batchSize, &io->op_policy_mem_gpu_, &tensor_mem_[0],
                      nullptr,
                      command_list_);  // pol FC
                      */

  // printf("\nAfter policy fc");
  // dumpTensor(io->op_policy_mem_gpu_, 1858 * 4, false);
  // dumpTensor(io->op_policy_mem_gpu_, 1858 * 2);

  // Alternate path - shader write to video memory and then copy to system
  // memory.
  network_[l++]->Eval(batchSize, &tensor_mem_[1], &tensor_mem_[0], nullptr,
                      dx_context_.getCommandList());  // pol FC

  dx_context_.copyTensor(io->op_policy_mem_gpu_, tensor_mem_[1],
                         batchSize * sizeof(float) * kNumOutputPolicy);

  // printf("\nAfter policy fc");
  // dumpTensor(io->op_policy_mem_gpu_, 1858 * 4, false);

  //-----------------------------------///---------------------------------------

  dx_context_.getCommandList()->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::UAV(tensor_mem_[0].pResource));

  // value head
  network_[l++]->Eval(batchSize, &tensor_mem_[0], &tensor_mem_[2], nullptr,
                      dx_context_.getCommandList());  // value conv

  // printf("\nAfter value conv");
  // dumpTensor(tensor_mem_[0], 2048);

  dx_context_.getCommandList()->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::UAV(tensor_mem_[0].pResource));

  network_[l++]->Eval(batchSize, &tensor_mem_[1], &tensor_mem_[0], nullptr,
                      dx_context_.getCommandList());  // value FC1

  // printf("\nAfter value FC1");
  // dumpTensor(tensor_mem_[1], 512);

  dx_context_.getCommandList()->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::UAV(tensor_mem_[1].pResource));

  // Writes directly fp32 to system memory
  network_[l++]->Eval(batchSize, &io->op_value_mem_gpu_, &tensor_mem_[1],
                      nullptr, dx_context_.getCommandList());  // val FC2

  // TODO: measure time from start to this point to get an idea of CPU side
  // overhead in recording command list
#ifdef DEBUG_RAW_NPS
  auto t_cpu_end = std::chrono::high_resolution_clock::now();
#endif

  dx_context_.flushAndWait();

  if (batchSize == testBatchSize) {
    auto t_end = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(t_end - t_start).count();
    double nps = testBatchSize / dt;
    printf("\nFor %d batch size, NN eval time: %lf seconds. NPS: %g\n",
           testBatchSize, dt, nps);
  }

  // printf("\nvalue head output: %f\n", *(io->op_value_mem_));

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

  delete input_conv_;
  delete resi_conv_1_;
  delete resi_conv_2_;
  delete head_conv_;
}

std::unique_ptr<NetworkComputation> DxNetwork::NewComputation() {
  // set correct gpu id for this computation (as it might have been called
  // from a different thread)
  // ReportCUDAErrors(cudaSetDevice(gpu_id_));
  // TODO: correct GPU id
  return std::make_unique<DxNetworkComputation>(this);
}

std::unique_ptr<InputsOutputsDx> DxNetwork::GetInputsOutputs() {
  std::lock_guard<std::mutex> lock(inputs_outputs_lock_);
  if (free_inputs_outputs_.empty()) {
    return std::make_unique<InputsOutputsDx>(max_batch_size_, &dx_context_);
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

DxNetworkComputation::DxNetworkComputation(DxNetwork* network)
    : network_(network) {
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

InputsOutputsDx::InputsOutputsDx(int maxBatchSize, DxContext* pContext) {
  // CPU accesses on Default heap doesn't work.
  // GPU accesses on Upload heap works.
  pContext->CreateAlloc(maxBatchSize * kInputPlanes * sizeof(uint64_t),
                        D3D12_HEAP_TYPE_UPLOAD /*D3D12_HEAP_TYPE_DEFAULT*/,
                        &input_masks_mem_gpu_);

  pContext->CreateAlloc(maxBatchSize * kInputPlanes * sizeof(float),
                        D3D12_HEAP_TYPE_UPLOAD /*D3D12_HEAP_TYPE_DEFAULT*/,
                        &input_val_mem_gpu_);

  // CUSTOM heap created to have GPU directly write to system memory
  pContext->CreateAlloc(maxBatchSize * kNumOutputPolicy * sizeof(float),
                        D3D12_HEAP_TYPE_CUSTOM, &op_policy_mem_gpu_);

  pContext->CreateAlloc(maxBatchSize * kNumOutputPolicy * sizeof(float),
                        D3D12_HEAP_TYPE_CUSTOM, &op_value_mem_gpu_);

  ReportDxErrors(input_masks_mem_gpu_.pResource->Map(
      0, nullptr, (void**)&input_masks_mem_));

  ReportDxErrors(
      input_val_mem_gpu_.pResource->Map(0, nullptr, (void**)&input_val_mem_));

  ReportDxErrors(
      op_policy_mem_gpu_.pResource->Map(0, nullptr, (void**)&op_policy_mem_));

  ReportDxErrors(
      op_value_mem_gpu_.pResource->Map(0, nullptr, (void**)&op_value_mem_));

#if 0
  // need command lists for each batch size :-/
  // Create command list and bake all commands
  ReportDxErrors(pDevice->CreateCommandAllocator(
      D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&command_allocator_)));

  ReportDxErrors(pDevice->CreateCommandList(1, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                            command_allocator_, NULL,
                                            IID_PPV_ARGS(&command_list_)));

  // expand packed planes to full planes
  expandPlanes(command_list_, network->tensor_mem_[0], input_masks_mem_gpu_,
               input_val_mem_gpu_);

  // TODO: Ankan - important! insert barriers!

  int l = 0;
  // input
  network->network_[l++]->Eval(batchSize, &tensor_mem_[2], &tensor_mem_[0], nullptr,
                               command_list_);  // input conv

  // residual block
  for (int block = 0; block < numBlocks_; block++) {
    network_[l++]->Eval(batchSize, &tensor_mem_[0], &tensor_mem_[2], nullptr,
                        command_list_);  // conv1

    network_[l++]->Eval(batchSize, &tensor_mem_[2], &tensor_mem_[0], nullptr,
                        command_list_);  // conv2
  }

  // policy head
  network_[l++]->Eval(batchSize, &tensor_mem_[0], &tensor_mem_[2], nullptr,
                      command_list_);  // pol conv

  // writes directly fp32 to system memory
  network_[l++]->Eval(batchSize, &io->op_policy_mem_gpu_, &tensor_mem_[0],
                      nullptr, command_list_);  // pol FC

  // value head
  network_[l++]->Eval(batchSize, &tensor_mem_[0], &tensor_mem_[2], nullptr,
                      command_list_);  // value conv

  network_[l++]->Eval(batchSize, &tensor_mem_[1], &tensor_mem_[0], nullptr,
                      command_list_);  // value FC1

  // Writes directly fp32 to system memory
  network_[l++]->Eval(batchSize, &io->op_policy_mem_gpu_, &tensor_mem_[1],
                      nullptr, command_list_);  // val FC2

  command_list_->Close();
#endif
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
}

std::unique_ptr<Network> MakeDxNetwork(const WeightsFile& weights,
                                       const OptionsDict& options) {
  // Ankan - test!
  /*
  if (weights.format().network_format().network() !=
      pblczero::NetworkFormat::NETWORK_CLASSICAL) {
    throw Exception(
        "Network format " +
        std::to_string(weights.format().network_format().network()) +
        " is not supported by DirectML backend.");
  }
  */
  return std::make_unique<DxNetwork>(weights, options);
}

REGISTER_NETWORK("dx-fp16", MakeDxNetwork, 120)

}  // namespace lczero
