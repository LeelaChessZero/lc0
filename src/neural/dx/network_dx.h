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
#pragma once

#include "dx_common.h"
#include "neural/factory.h"
#include "neural/network_legacy.h"

namespace lczero {

using namespace dx_backend;
class DxNetwork;
static constexpr int kNumOutputPolicy = 1858;

struct InputsOutputsDx {
  InputsOutputsDx(int maxBatchSize, DxContext* pContext);
  ~InputsOutputsDx();

  // In default heap (video memory, mapped to support CPU writes too).
  DXAlloc input_masks_mem_gpu_;
  DXAlloc input_val_mem_gpu_;

  // In readback heap (system memory mapped for both CPU and GPU).
  DXAlloc op_policy_mem_gpu_;
  DXAlloc op_value_mem_gpu_;

  // CPU pointers of the above allocations.
  uint64_t* input_masks_mem_;
  float* input_val_mem_;
  float* op_policy_mem_;
  float* op_value_mem_;

  // command list with recorded commands to run the network
  // ID3D12GraphicsCommandList4* command_list_[1024];
  // ID3D12CommandAllocator* command_allocator_[1024];
};

class DxNetworkComputation : public NetworkComputation {
 public:
  DxNetworkComputation(DxNetwork* network);
  ~DxNetworkComputation();

  void AddInput(InputPlanes&& input) override;

  void ComputeBlocking() override;

  int GetBatchSize() const override { return batch_size_; }

  float GetQVal(int sample) const override {
    return inputs_outputs_->op_value_mem_[sample];
  }
  float GetPVal(int sample, int move_id) const override {
    return inputs_outputs_->op_policy_mem_[sample * kNumOutputPolicy + move_id];
  }

  float GetDVal(int sample) const override {
    // TODO: support WDL!
    return 0.0f;
  }

 private:
  // Memory holding inputs, outputs.
  std::unique_ptr<InputsOutputsDx> inputs_outputs_;
  int batch_size_;
  bool wdl_;

  DxNetwork* network_;
};

// D3D12 related stuff.
class DxContext {
 private:
  // Should be enough to hold descriptors for all resources.
  static constexpr int kNumDescHeapSlots = 65536;

  ID3D12Device5* device_;
  ID3D12CommandQueue* command_queue_;
  ID3D12GraphicsCommandList5* command_list_;
  ID3D12CommandAllocator* command_allocator_;
  ID3D12DescriptorHeap* desc_heap_;
  ID3D12Fence* fence_;
  uint64_t fenceVal;
  ShaderWrapper shader_wrapper_;

  std::atomic<unsigned int> next_slot_in_desc_heap_;

  // in system memory (used to copy to/from CPU data).
  DXAlloc upload_scratch_mem_;
  DXAlloc readback_scratch_mem_;
  DXAlloc default_scratch_mem_;

  int gpu_id_;

 public:
  DxContext(const OptionsDict& options);
  ~DxContext();


  ID3D12Device5* getDevice() { return device_; }
  ID3D12GraphicsCommandList5* getCommandList() { return command_list_; }
  ShaderWrapper* getShaderWrapper() { return &shader_wrapper_; }
  DXAlloc* getDefaultScratch() { return &default_scratch_mem_; }

  // util functions
  void CreateAlloc(size_t size, D3D12_HEAP_TYPE type, DXAlloc* pAlloc);
  void flushAndWait();
  void scheduleUpload(DXAlloc alloc, void* data, size_t size);
  void dumpFp32(float* buf, int elements);
  void copyTensor(DXAlloc dst, DXAlloc src, int bytes);
  void dumpTensor(DXAlloc alloc, int bytes, bool fp16 = true,
                  bool allnewline = false);
};

class DxNetwork : public Network {
  friend struct InputsOutputsDx;

 public:
  DxNetwork(const WeightsFile& file, const OptionsDict& options);
  ~DxNetwork();

  void forwardEval(InputsOutputsDx* io, int batchSize);
  std::unique_ptr<NetworkComputation> NewComputation() override;

  std::unique_ptr<InputsOutputsDx> GetInputsOutputs();
  void ReleaseInputsOutputs(std::unique_ptr<InputsOutputsDx> resource);

 private:
  DxContext dx_context_;
  int max_batch_size_;

  // Do we want to use nhwc layout? (fastest with fp16 with tensor
  // cores).
  bool nhwc_;

  // Currently only one NN Eval can happen a time (we can fix this if needed
  // by allocating more memory).
  mutable std::mutex lock_;

  // Network Properties.
  int numBlocks_;
  bool has_se_;
  bool has_wdl_;
  bool has_conv_policy_;

  std::vector<std::unique_ptr<BaseLayer>> network_;
  BaseLayer* getLastLayer() { return network_.back().get(); }

  BaseLayer* resi_last_;
  BaseLayer* policy_out_;
  BaseLayer* value_out_;

  // unique convolution types used by the network
  ConvMetaCommand* input_conv_;
  ConvMetaCommand* resi_conv_1_;
  ConvMetaCommand* resi_conv_2_;
  ConvMetaCommand* head_conv_;

  // in device memory
  DXAlloc tensor_mem_[3];

  size_t scratch_size_;

  mutable std::mutex inputs_outputs_lock_;
  std::list<std::unique_ptr<InputsOutputsDx>> free_inputs_outputs_;
};

};  // namespace lczero
