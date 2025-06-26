/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2025 The LCZero Authors

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

#include "neural/batchsplit.h"

namespace lczero {
namespace {

class BatchSplittingBackend : public Backend {
 public:
  BatchSplittingBackend(Backend* wrapped) : wrapped_backend_(wrapped) {}

  BackendAttributes GetAttributes() const override {
    return wrapped_backend_->GetAttributes();
  }
  std::optional<EvalResult> GetCachedEvaluation(
      const EvalPosition& pos) override {
    return wrapped_backend_->GetCachedEvaluation(pos);
  }
  std::unique_ptr<BackendComputation> CreateComputation() override;

  UpdateConfigurationResult UpdateConfiguration(
      const OptionsDict& options) override {
    return wrapped_backend_->UpdateConfiguration(options);
  }

 private:
  Backend* wrapped_backend_;
};

class BatchSplittingComputation : public BackendComputation {
 public:
  BatchSplittingComputation(Backend* wrapped_backend)
      : wrapped_backend_(wrapped_backend),
        max_batch_size_(wrapped_backend->GetAttributes().maximum_batch_size) {
    MakeComputation();
  }

  size_t UsedBatchSize() const override {
    return wrapped_computation_->UsedBatchSize();
  }
  AddInputResult AddInput(const EvalPosition& pos,
                          EvalResultPtr result) override {
    if (wrapped_computation_->UsedBatchSize() >= max_batch_size_) {
      ComputeBlocking();
      MakeComputation();
    }
    return wrapped_computation_->AddInput(pos, result);
  }

  void ComputeBlocking() override { wrapped_computation_->ComputeBlocking(); }

 private:
  void MakeComputation() {
    wrapped_computation_ = wrapped_backend_->CreateComputation();
  }

  Backend* wrapped_backend_;
  size_t max_batch_size_;
  std::unique_ptr<BackendComputation> wrapped_computation_;
};

std::unique_ptr<BackendComputation> BatchSplittingBackend::CreateComputation() {
  return std::make_unique<BatchSplittingComputation>(wrapped_backend_);
}

}  // namespace

std::unique_ptr<Backend> CreateBatchSplitingBackend(Backend* parent) {
  return std::make_unique<BatchSplittingBackend>(parent);
}

}  // namespace lczero