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

#include "dx_common.h"
#include "neural/factory.h"
#include "neural/network_legacy.h"

// TODO: Consider refactoring common part of this backend's code and cudnn
// backend into some base class(es).

namespace lczero {

using namespace dx_backend;
class DxNetwork;

static constexpr int kNumOutputPolicy = 1858;

// Padding needed as on some HW (e.g: NV) fp16 requires gemm matrix dimensions
// to be multiples of 8
static constexpr int kNumOutputPolicyPadded8 = ((kNumOutputPolicy - 1)/8 + 1)*8;

// Normally 3 when using wdl, and 1 without
static constexpr int kNumOutputValuePadded8 = 8;    

struct InputsOutputsDx {
  InputsOutputsDx(int maxBatchSize, DxContext* pContext, bool wdl);
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

  // separate copy, un-padded and always in fp32
  float* op_policy_mem_final_;
  float* op_value_mem_final_;

  // TODO: This can be useful to bake once - execute multiple times
  // which can significantly reduce CPU side overhead of DX calls.
  // Command list with recorded commands to run the network.
  // ID3D12GraphicsCommandList4* command_list_[1024];
  // ID3D12CommandAllocator* command_allocator_[1024];
};

class DxNetworkComputation : public NetworkComputation {
 public:
  DxNetworkComputation(DxNetwork* network, bool wdl);
  ~DxNetworkComputation();

  void AddInput(InputPlanes&& input) override;

  void ComputeBlocking() override;

  int GetBatchSize() const override { return batch_size_; }

  float GetQVal(int sample) const override {
    if (wdl_) {
      auto w = inputs_outputs_->op_value_mem_final_[3 * sample + 0];
      auto l = inputs_outputs_->op_value_mem_final_[3 * sample + 2];
      return w - l;
    } else {
      return inputs_outputs_->op_value_mem_final_[sample];
    }
  }

  float GetDVal(int sample) const override {
    if (wdl_) {
      auto d = inputs_outputs_->op_value_mem_final_[3 * sample + 1];
      return d;
    } else {
      return 0.0f;
    }
  }

  float GetPVal(int sample, int move_id) const override {
    return inputs_outputs_
        ->op_policy_mem_final_[sample * kNumOutputPolicy + move_id];
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
  size_t scratch_size_;
  DXAlloc upload_scratch_mem_;
  DXAlloc readback_scratch_mem_;

  int gpu_id_;

 public:
  DxContext(const OptionsDict& options);
  ~DxContext();


  ID3D12Device5* getDevice() { return device_; }
  ID3D12GraphicsCommandList5* getCommandList() { return command_list_; }
  ShaderWrapper* getShaderWrapper() { return &shader_wrapper_; }

  // util functions
  void CreateAlloc(size_t size, D3D12_HEAP_TYPE type, DXAlloc& alloc);
  void flushAndWait();
  void scheduleUpload(DXAlloc alloc, void* data, size_t size);
  void dumpFp32(float* buf, int elements);
  void copyTensor(DXAlloc dst, DXAlloc src, int bytes);
  void dumpTensor(DXAlloc alloc, int size, bool fp16 = true,
                  bool allnewline = false);
  void dumpCpuTensor(void* data, int size, bool fp16 = true,
                     bool allnewline = false);
};

class DxNetwork : public Network {
  friend struct InputsOutputsDx;

 public:
  DxNetwork(const WeightsFile& file, const OptionsDict& options);
  ~DxNetwork();

  void forwardEval(InputsOutputsDx* io, int batchSize);
  std::unique_ptr<NetworkComputation> NewComputation() override;
  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }

  std::unique_ptr<InputsOutputsDx> GetInputsOutputs();
  void ReleaseInputsOutputs(std::unique_ptr<InputsOutputsDx> resource);

 private:
  const NetworkCapabilities capabilities_;
  DxContext dx_context_;
  int max_batch_size_;

  // Currently only one NN Eval can happen a time (we can fix this if needed
  // by allocating more memory).
  mutable std::mutex lock_;

  // Network Properties.
  int numBlocks_;
  bool has_se_;
  bool has_wdl_;
  bool has_conv_policy_;
  bool fp16_;

  std::vector<std::unique_ptr<BaseLayer>> network_;
  BaseLayer* getLastLayer() { return network_.back().get(); }

  BaseLayer* resi_last_;
  BaseLayer* policy_out_;
  BaseLayer* value_out_;

  // Unique Metacommands used multiple times in the network.
  GemmMetaCommand* input_conv_winograd_gemm_;
  GemmMetaCommand* residual_block_winograd_gemm_;
  GemmMetaCommand* policy_conv_winograd_gemm_;

  // In device memory.
  DXAlloc tensor_mem_[4];

  mutable std::mutex inputs_outputs_lock_;
  std::list<std::unique_ptr<InputsOutputsDx>> free_inputs_outputs_;
};

};  // namespace lczero
