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

#include "network_coreml.h"

namespace lczero {
namespace coreml_backend {

CoreMLNetworkComputation::CoreMLNetworkComputation(CoreMLNetwork* network,
                                                   bool wdl, bool moves_left)
    : wdl_(wdl), moves_left_(moves_left), network_(network) {
  batch_size_ = 0;
  inputs_outputs_ = network_->GetInputsOutputs();
}

CoreMLNetworkComputation::~CoreMLNetworkComputation() {
  network_->ReleaseInputsOutputs(std::move(inputs_outputs_));
}

void CoreMLNetworkComputation::ComputeBlocking() {
  network_->forwardEval(inputs_outputs_.get(), GetBatchSize());
}

CoreMLNetwork::CoreMLNetwork(const WeightsFile& file,
                             const OptionsDict& options)
    : capabilities_{file.format().network_format().input(),
                    file.format().network_format().moves_left()} {
  LegacyWeights weights(file.weights());

  max_batch_size_ = options.GetOrDefault<int>("max_batch", 1024);
  batch_size_ = options.GetOrDefault<int>("batch", 64);

  conv_policy_ = file.format().network_format().policy() ==
                 pblczero::NetworkFormat::POLICY_CONVOLUTION;

  attn_policy_ = file.format().network_format().policy() ==
                 pblczero::NetworkFormat::POLICY_ATTENTION;

  wdl_ = file.format().network_format().value() ==
         pblczero::NetworkFormat::VALUE_WDL;

  moves_left_ = (file.format().network_format().moves_left() ==
                 pblczero::NetworkFormat::MOVES_LEFT_V1) &&
                options.GetOrDefault<bool>("mlh", true);

  coreml_ = std::make_unique<CoreML>(wdl_, moves_left_);
}

void CoreMLNetwork::forwardEval(InputsOutputs* io, int batchSize) {
  // Expand encoded input into N x 112 x 8 x 8.
  float* dptr = &io->input_val_mem_expanded_[0];
  for (int i = 0; i < batchSize; i++) {
    for (int j = 0; j < kInputPlanes; j++) {
      const float value = io->input_val_mem_[j + i * kInputPlanes];
      const uint64_t mask = io->input_masks_mem_[j + i * kInputPlanes];
      for (auto k = 0; k < 64; k++) {
        *(dptr++) = (mask & (((uint64_t)1) << k)) != 0 ? value : 0;
      }
    }
  }

  lock_.lock();

  coreml_->forwardEval(&io->input_val_mem_expanded_[0], batchSize,
                       &io->op_policy_mem_[0], &io->op_value_mem_[0],
                       &io->op_moves_left_mem_[0]);

  lock_.unlock();
}

std::unique_ptr<Network> MakeCoreMLNetwork(const std::optional<WeightsFile>& w,
                                           const OptionsDict& options) {
  const WeightsFile& weights = *w;
  return std::make_unique<CoreMLNetwork>(weights, options);
}

REGISTER_NETWORK("coreml", MakeCoreMLNetwork, 104)

}  // namespace coreml_backend
}  // namespace lczero
