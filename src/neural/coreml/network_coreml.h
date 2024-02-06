/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2024 The LCZero Authors

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

#include <list>

#include "neural/coreml/CoreML.h"
#include "neural/factory.h"
#include "neural/network_legacy.h"

namespace lczero {
namespace coreml_backend {

static int kNumOutputPolicy = 1858;
static int kInputPlanes = 112;

struct InputsOutputs {
  InputsOutputs(int maxBatchSize, bool wdl, bool moves_left) {
    input_masks_mem_.reserve(maxBatchSize * kInputPlanes);
    input_val_mem_.reserve(maxBatchSize * kInputPlanes);
    input_val_mem_expanded_.reserve(maxBatchSize * kInputPlanes * 64);
    op_policy_mem_.reserve(maxBatchSize * kNumOutputPolicy);
    op_value_mem_.reserve(maxBatchSize * (wdl ? 3 : 1));

    if (moves_left) {
      op_moves_left_mem_.reserve(maxBatchSize);
    };
  }
  ~InputsOutputs() {}

  std::vector<uint64_t> input_masks_mem_;
  std::vector<float> input_val_mem_;
  std::vector<float> input_val_mem_expanded_;
  std::vector<float> op_policy_mem_;
  std::vector<float> op_value_mem_;
  std::vector<float> op_moves_left_mem_;
};

class CoreMLNetwork;

class CoreMLNetworkComputation : public NetworkComputation {
 public:
  CoreMLNetworkComputation(CoreMLNetwork* network, bool wdl, bool moves_left);
  ~CoreMLNetworkComputation();

  void AddInput(InputPlanes&& input) override {
    const auto iter_mask =
        &inputs_outputs_->input_masks_mem_[batch_size_ * kInputPlanes];
    const auto iter_val =
        &inputs_outputs_->input_val_mem_[batch_size_ * kInputPlanes];

    int i = 0;
    for (const auto& plane : input) {
      iter_mask[i] = plane.mask;
      iter_val[i] = plane.value;
      i++;
    }

    batch_size_++;
  }

  void ComputeBlocking() override;

  int GetBatchSize() const override { return batch_size_; }

  float GetQVal(int sample) const override {
    if (wdl_) {
      auto w = inputs_outputs_->op_value_mem_[3 * sample + 0];
      auto l = inputs_outputs_->op_value_mem_[3 * sample + 2];
      return w - l;
    } else {
      return inputs_outputs_->op_value_mem_[sample];
    }
  }

  float GetDVal(int sample) const override {
    if (wdl_) {
      auto d = inputs_outputs_->op_value_mem_[3 * sample + 1];
      return d;
    } else {
      return 0.0f;
    }
  }

  float GetPVal(int sample, int move_id) const override {
    return inputs_outputs_->op_policy_mem_[sample * kNumOutputPolicy + move_id];
  }

  float GetMVal(int sample) const override {
    if (moves_left_) {
      return inputs_outputs_->op_moves_left_mem_[sample];
    }
    return 0.0f;
  }

 private:
  // Memory holding inputs, outputs.
  std::unique_ptr<InputsOutputs> inputs_outputs_;
  int batch_size_;
  bool wdl_;
  bool moves_left_;

  CoreMLNetwork* network_;
};

class CoreMLNetwork : public Network {
 public:
  CoreMLNetwork(const WeightsFile& file, const OptionsDict& options);
  ~CoreMLNetwork() {}

  void forwardEval(InputsOutputs* io, int inputBatchSize);

  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<CoreMLNetworkComputation>(this, wdl_, moves_left_);
  }

  std::unique_ptr<InputsOutputs> GetInputsOutputs() {
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
  }

  void ReleaseInputsOutputs(std::unique_ptr<InputsOutputs> resource) {
    std::lock_guard<std::mutex> lock(inputs_outputs_lock_);
    free_inputs_outputs_.push_back(std::move(resource));
  }

  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }

 private:
  NetworkCapabilities capabilities_{
      pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE,
      pblczero::NetworkFormat::MOVES_LEFT_NONE};
  int max_batch_size_;
  int batch_size_;
  bool wdl_;
  bool moves_left_;
  bool conv_policy_;
  bool attn_policy_;
  std::mutex inputs_outputs_lock_;
  std::list<std::unique_ptr<InputsOutputs>> free_inputs_outputs_;
  std::unique_ptr<coreml_backend::CoreML> coreml_;
  mutable std::mutex lock_;
};

}  // namespace coreml_backend
}  // namespace lczero
