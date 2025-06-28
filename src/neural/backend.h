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
  int suggested_num_search_threads;
  int recommended_batch_size;
  int maximum_batch_size;
};

struct EvalResultPtr {
  float* q = nullptr;
  float* d = nullptr;
  float* m = nullptr;
  std::span<float> p = {};
};

struct EvalResult {
  float q;
  float d;
  float m;
  std::vector<float> p;

  EvalResultPtr AsPtr() {
    return EvalResultPtr{.q = &q, .d = &d, .m = &m, .p = p};
  }
};

struct EvalPosition {
  std::span<const Position> pos;
  std::span<const Move> legal_moves;
};

class BackendComputation {
 public:
  virtual ~BackendComputation() = default;
  virtual size_t UsedBatchSize() const = 0;
  enum AddInputResult {
    ENQUEUED_FOR_EVAL = 0,    // Will be computed during ComputeBlocking();
    FETCHED_IMMEDIATELY = 1,  // Was in cache, the result is already populated.
  };
  virtual AddInputResult AddInput(
      const EvalPosition& pos,    // Input position.
      EvalResultPtr result) = 0;  // Where to fetch data into.
  virtual void ComputeBlocking() = 0;
};

class Backend {
 public:
  virtual ~Backend() = default;
  virtual BackendAttributes GetAttributes() const = 0;
  virtual std::unique_ptr<BackendComputation> CreateComputation() = 0;

  // Simple helper with default implementation, to evaluate a batch without
  // creating a computation explicitly.
  virtual std::vector<EvalResult> EvaluateBatch(
      std::span<const EvalPosition> positions);
  // Returns the evaluation if it's possible to do immediately.
  virtual std::optional<EvalResult> GetCachedEvaluation(const EvalPosition&) {
    return std::nullopt;
  }

  // Updates the configuration of the backend. This is between searches.
  // It's up to the backend to detect if the configuration has changed.
  enum UpdateConfigurationResult {
    UPDATE_OK = 0,     // Backend handled the update by itself (if needed).
    NEED_RESTART = 1,  // Recreate the backend.
  };
  virtual UpdateConfigurationResult UpdateConfiguration(
      const OptionsDict& opts) {
    current_config_hash_ = ConfigurationHash(opts);
    return UPDATE_OK;
  }

  virtual bool IsSameConfiguration(const OptionsDict& opts) const {
    return ConfigurationHash(opts) == current_config_hash_;
  }

 private:
  // Gets a hash of the backend configuration, to help detect changes.
  virtual uint64_t ConfigurationHash(const OptionsDict&) const;

  uint64_t current_config_hash_;
};

class BackendFactory {
 public:
  virtual ~BackendFactory() = default;
  // Higher priority is higher.
  virtual int GetPriority() const = 0;
  virtual std::string_view GetName() const = 0;
  virtual std::unique_ptr<Backend> Create(const OptionsDict&) = 0;
};

}  // namespace lczero