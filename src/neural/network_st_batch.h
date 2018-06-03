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

#pragma once

#include "neural/network.h"

namespace lczero {

// This is a network that helps to combine batches from multiple games running
// is a single thread. Not thread safe.
// Usage:
//   network.Reset();   // Creates new parent computation
//   computations = []
//   multiple times:
//     x = network.NewComputation()
//     computations += x
//     x.AddInput();
//     x.AddInput();
//     x.AddInput();
//     ...
//   for x in computations:
//     x.ComputeBlocking()   // Only last call actually computes, and they are
//                           // computed together in one batch.
//   for x in computations:
//     use(x)
class SingleThreadBatchingNetwork : public Network {
 public:
  SingleThreadBatchingNetwork(std::unique_ptr<Network> parent);
  std::unique_ptr<NetworkComputation> NewComputation() override;

  // Start a fresh batch.
  void Reset();

 private:
  std::unique_ptr<Network> parent_;
  std::unique_ptr<NetworkComputation> parent_computation_;
  int computations_pending_ = 0;
  friend class SingleThreadBatchingNetworkComputation;
};

class SingleThreadBatchingNetworkComputation : public NetworkComputation {
 public:
  SingleThreadBatchingNetworkComputation(SingleThreadBatchingNetwork* network);

  // Adds a sample to the parent batch.
  void AddInput(InputPlanes&& input) override;
  // May not actualy compute immediately. Instead computes when all computations
  // of the network called this.
  void ComputeBlocking() override;
  // Returns how many times AddInput() was called.
  int GetBatchSize() const override { return batch_size_; }
  // Returns Q value of @sample.
  float GetQVal(int sample) const override;
  // Returns P value @move_id of @sample.
  float GetPVal(int sample, int move_id) const override;

 private:
  SingleThreadBatchingNetwork* const network_;
  int start_idx_;
  int batch_size_ = 0;
};

}  // namespace lczero