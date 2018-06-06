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
*/

#include "neural/network_st_batch.h"

#include <cassert>

namespace lczero {

SingleThreadBatchingNetwork::SingleThreadBatchingNetwork(
    std::unique_ptr<Network> parent)
    : parent_(std::move(parent)) {}

std::unique_ptr<NetworkComputation>
SingleThreadBatchingNetwork::NewComputation() {
  ++computations_pending_;
  return std::make_unique<SingleThreadBatchingNetworkComputation>(this);
}

void SingleThreadBatchingNetwork::Reset() {
  assert(computations_pending_ == 0);
  parent_computation_ = parent_->NewComputation();
}

SingleThreadBatchingNetworkComputation::SingleThreadBatchingNetworkComputation(
    SingleThreadBatchingNetwork* network)
    : network_(network),
      start_idx_(network_->parent_computation_->GetBatchSize()) {}

void SingleThreadBatchingNetworkComputation::AddInput(InputPlanes&& input) {
  assert(start_idx_ + batch_size_ ==
         network_->parent_computation_->GetBatchSize());
  ++batch_size_;
  network_->parent_computation_->AddInput(std::move(input));
}

void SingleThreadBatchingNetworkComputation::ComputeBlocking() {
  if (--network_->computations_pending_ == 0)
    network_->parent_computation_->ComputeBlocking();
}

float SingleThreadBatchingNetworkComputation::GetQVal(int sample) const {
  return network_->parent_computation_->GetQVal(sample - start_idx_);
}

float SingleThreadBatchingNetworkComputation::GetPVal(int sample,
                                                      int move_id) const {
  return network_->parent_computation_->GetPVal(sample - start_idx_, move_id);
}

}  // namespace lczero