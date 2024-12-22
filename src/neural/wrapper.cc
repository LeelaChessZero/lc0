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

#include "neural/wrapper.h"

#include "neural/encoder.h"

namespace lczero {
namespace {

class NetworkAsBackend : public Backend {
 public:
  NetworkAsBackend(std::unique_ptr<Network> network)
      : network_(std::move(network)) {
    const NetworkCapabilities& caps = network_->GetCapabilities();
    attrs_.has_mlh = caps.has_mlh();
    attrs_.has_wdl = caps.has_wdl();
    attrs_.runs_on_cpu = network_->IsCpu();
    attrs_.suggested_num_search_threads = network_->GetThreads();
    attrs_.maximum_batch_size = network_->GetMiniBatchSize();
    input_format_ = caps.input_format;
  }

  BackendAttributes GetAttributes() const override { return attrs_; }
  virtual std::unique_ptr<BackendComputation> CreateComputation() override;

 private:
  std::unique_ptr<Network> network_;
  BackendAttributes attrs_;
  pblczero::NetworkFormat::InputFormat input_format_;
  friend class NetworkAsBackendComputation;
};

class NetworkAsBackendComputation : public BackendComputation {
 public:
  NetworkAsBackendComputation(NetworkAsBackend* backend)
      : backend_(backend), computation_(backend_->network_->NewComputation()) {
    results_.reserve(backend_->attrs_.maximum_batch_size);
    moves_.reserve(backend_->attrs_.maximum_batch_size);
    transforms_.reserve(backend_->attrs_.maximum_batch_size);
  }

  size_t RemainingBatchSize() const override {
    return backend_->attrs_.maximum_batch_size - computation_->GetBatchSize();
  }

  AddInputResult AddInput(const EvalPosition& pos, EvalResultPtr result,
                          FetchMode mode) override {
    if (mode == CACHE_ONLY) return REJECTED;
    int transform;
    computation_->AddInput(EncodePositionForNN(backend_->input_format_, pos.pos,
                                               8, FillEmptyHistory::FEN_ONLY,
                                               &transform));
    results_.push_back(result);
    moves_.emplace_back(pos.legal_moves.begin(), pos.legal_moves.end());
    transforms_.push_back(transform);
    return ENQUEUED_FOR_EVAL;
  }

  void ComputeBlocking() override {
    computation_->ComputeBlocking();
    for (size_t i = 0; i < results_.size(); ++i) {
      const EvalResultPtr& result = results_[i];
      const std::vector<Move>& moves = moves_[i];
      if (result.q) *result.q = computation_->GetQVal(i);
      if (result.d) *result.d = computation_->GetDVal(i);
      if (result.m) *result.m = computation_->GetMVal(i);
      if (!result.p.empty()) {
        auto* p = result.p.data();
        for (const Move& move : moves) {
          *p++ = computation_->GetPVal(i, move.as_nn_index(transforms_[i]));
        }
      }
    }
  }

 private:
  NetworkAsBackend* backend_;
  std::unique_ptr<NetworkComputation> computation_;
  std::vector<std::vector<Move>> moves_;
  std::vector<EvalResultPtr> results_;
  std::vector<int> transforms_;
};

std::unique_ptr<BackendComputation> NetworkAsBackend::CreateComputation() {
  return std::make_unique<NetworkAsBackendComputation>(this);
}

}  // namespace

NetworkAsBackendFactory::NetworkAsBackendFactory(const std::string& name,
                                                 FactoryFunc factory,
                                                 int priority)
    : name_(name), factory_(factory), priority_(priority) {}

std::unique_ptr<Backend> NetworkAsBackendFactory::Create(
    const std::optional<WeightsFile>& weights, const OptionsDict& options) {
  return std::make_unique<NetworkAsBackend>(factory_(weights, options));
}

}  // namespace lczero