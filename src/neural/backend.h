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

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "chess/position.h"
#include "neural/loader.h"
#include "utils/optionsdict.h"

namespace lczero {

// Information about the backend or network that search may need.
struct BackendAttributes {
  bool has_mlh;
  bool has_wdl;
  bool runs_on_cpu;
  size_t suggested_num_search_threads;
  size_t maximum_batch_size;
};

struct EvalResult {
  float q;
  float d;
  float m;
  std::vector<float> p;
};

struct EvalResultPtr {
  float* q = nullptr;
  float* d = nullptr;
  float* m = nullptr;
  std::span<float> p;
};

struct EvalPosition {
  std::span<const Position> pos;
  std::span<const Move> legal_moves;
};

class BackendComputation {
 public:
  virtual ~BackendComputation() = default;
  virtual size_t RemainingBatchSize() const = 0;
  enum AddInputResult {
    FETCHED_IMMEDIATELY = 1,  // Was in cache, the result is
                              // already populated.
    ENQUEUED_FOR_EVAL = 2,    // Will be computed during ComputeBlocking();
    REJECTED = 3,             // The sample won't be evaluated
                              // (i.e. because the request was cache-only).
  };
  enum FetchMode {
    CACHE_ONLY = 1,    // Do not enqueue if not in cache.
    ENQUEUE_ONLY = 2,  // Do not check cache.
    CACHE_OR_ENQUEUE = 3,
  };
  virtual AddInputResult AddInput(
      const EvalPosition& pos,  // Input position.
      EvalResultPtr result,     // Where to fetch data into.
      FetchMode mode) = 0;
  virtual void ComputeBlocking() = 0;
};

class Backend {
 public:
  virtual ~Backend() = default;
  virtual BackendAttributes GetAttributes() const = 0;
  virtual std::unique_ptr<BackendComputation> CreateComputation() = 0;

  // Simple helper with default implementation, to evaluate a batch without
  // creating a computation explicitly. Uses CACHE_OR_ENQUEUE mode.
  virtual std::vector<EvalResult> EvaluateBatch(
      std::span<const EvalPosition> positions);
  // Similarly, default implementation for probing a cache. Default
  // implementation creates a single input computation in CACHE_ONLY mode.
  virtual std::optional<EvalResult> GetCachedEvaluation(
      const EvalPosition& pos);
};

class BackendFactory {
 public:
  virtual ~BackendFactory() = default;
  // Higher priority is higher.
  virtual int GetPriority() const = 0;
  virtual std::string_view GetName() const = 0;
  virtual std::unique_ptr<Backend> Create(const std::optional<WeightsFile>&,
                                          const OptionsDict&) = 0;
};

}  // namespace lczero