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

#include "neural/network_st_batch.h"

#include <cassert>

namespace lczero {

SingleThreadBatchingNetwork::SingleThreadBatchingNetwork(Network* parent)
    : parent_(parent) {}

std::unique_ptr<NetworkComputation>
SingleThreadBatchingNetwork::NewComputation() {
  return std::make_unique<SingleThreadBatchingNetworkComputation>(this);
}

int SingleThreadBatchingNetwork::GetTotalBatchSize() const {
  return parent_computation_->GetBatchSize();
}

void SingleThreadBatchingNetwork::Reset() {
  assert(computed_ || !parent_computation_ ||
         parent_computation_->GetBatchSize() == 0);
  parent_computation_ = parent_->NewComputation();
  computed_ = false;
}

SingleThreadBatchingNetworkComputation::SingleThreadBatchingNetworkComputation(
    SingleThreadBatchingNetwork* network)
    : network_(network),
      start_idx_(network_->parent_computation_->GetBatchSize()) {}

void SingleThreadBatchingNetworkComputation::AddInput(InputPlanes&& input) {
  assert(start_idx_ + batch_size_ ==
         network_->parent_computation_->GetBatchSize());
  assert(!network_->computed_);
  network_->parent_computation_->AddInput(std::move(input));
  ++batch_size_;
}

void SingleThreadBatchingNetworkComputation::ComputeBlocking() {
  assert(batch_size_ > 0);
  if (!network_->computed_) network_->parent_computation_->ComputeBlocking();
  network_->computed_ = true;
}

float SingleThreadBatchingNetworkComputation::GetQVal(int sample) const {
  return network_->parent_computation_->GetQVal(sample + start_idx_);
}

float SingleThreadBatchingNetworkComputation::GetPVal(int sample,
                                                      int move_id) const {
  return network_->parent_computation_->GetPVal(sample + start_idx_, move_id);
}

}  // namespace lczero
