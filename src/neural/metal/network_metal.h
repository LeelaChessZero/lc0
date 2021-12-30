/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2021 The LCZero Authors

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

#include "neural/factory.h"
#include "neural/network_legacy.h"

namespace lczero {
namespace metal_backend {

class MetalNetwork;
class MetalNetworkDelegate;

class MetalNetworkComputation : public NetworkComputation {
 public:
  MetalNetworkComputation(MetalNetwork* network, bool wdl, bool moves_left)
      : wdl_(wdl), moves_left_(moves_left), network_(network) {
    batch_size_ = 0;
    //inputs_outputs_ = network_->GetInputsOutputs();
  }

  ~MetalNetworkComputation() {
    //network_->ReleaseInputsOutputs(std::move(inputs_outputs_));
  }

  void AddInput(InputPlanes&& input) override {
    /*const auto iter_mask =
        &inputs_outputs_->input_masks_mem_[batch_size_ * kInputPlanes];
    const auto iter_val =
        &inputs_outputs_->input_val_mem_[batch_size_ * kInputPlanes];

    int i = 0;
    for (const auto& plane : input) {
      iter_mask[i] = plane.mask;
      iter_val[i] = plane.value;
      i++;
    }*/

    batch_size_++;
  }

  void ComputeBlocking() override {
    //network_->forwardEval(inputs_outputs_.get(), GetBatchSize());
  }

  int GetBatchSize() const override { return batch_size_; }

  float GetQVal(int sample) const override {
    /*if (wdl_) {
      auto w = inputs_outputs_->op_value_mem_[3 * sample + 0];
      auto l = inputs_outputs_->op_value_mem_[3 * sample + 2];
      return w - l;
    } else {
      return inputs_outputs_->op_value_mem_[sample];
    }*/
    return 0.0f;
  }

  float GetDVal(int sample) const override {
    /*if (wdl_) {
      auto d = inputs_outputs_->op_value_mem_[3 * sample + 1];
      return d;
    } else {
      return 0.0f;
    }*/
    return 0.0f;
  }

  float GetPVal(int sample, int move_id) const override {
    //return inputs_outputs_->op_policy_mem_[sample * kNumOutputPolicy + move_id];
    return 0.0f;
  }

  float GetMVal(int sample) const override {
    /*if (moves_left_) {
      return inputs_outputs_->op_moves_left_mem_[sample];
    }*/
    return 0.0f;
  }

 private:
  // Memory holding inputs, outputs.
  //std::unique_ptr<InputsOutputs> inputs_outputs_;
  int batch_size_;
  bool wdl_;
  bool moves_left_;

  MetalNetwork* network_;
};

class MetalNetwork : public Network {
 public:
  MetalNetwork(const WeightsFile& file, const OptionsDict& options);
  ~MetalNetwork() {
    if ( delegate_ ) { /** @todo clean-up delegate first */ delete delegate_; delegate_ = NULL; }
  }

  //void forwardEval(InputsOutputs* io, int inputBatchSize) override;
  void forwardEval(int* io, int inputBatchSize);

  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<MetalNetworkComputation>(this, wdl_, moves_left_);
  }

  /*std::unique_ptr<InputsOutputs> GetInputsOutputs() {
    std::lock_guard<std::mutex> lock(inputs_outputs_lock_);
    if (free_inputs_outputs_.empty()) {
      return std::make_unique<InputsOutputs>(max_batch_size_, wdl_,
                                             moves_left_);
    } else {
      std::unique_ptr<InputsOutputs> resource =
          std::move(free_inputs_outputs_.front());
      free_inputs_outputs_.pop_front();
      return resource;
    }
  }*/

  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }

 private:
  NetworkCapabilities capabilities_{
      pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE,
      pblczero::NetworkFormat::MOVES_LEFT_NONE
  };
  int max_batch_size_;
  int batch_size_;
  int steps_;
  bool wdl_;
  bool moves_left_;
  std::mutex inputs_outputs_lock_;
  //std::list<std::unique_ptr<InputsOutputs>> free_inputs_outputs_;
  MetalNetworkDelegate* delegate_;
};

}  // namespace metal_backend
}  // namespace lczero
