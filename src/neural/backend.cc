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

#include "neural/backend.h"

namespace lczero {

std::vector<EvalResult> Backend::EvaluateBatch(
    std::span<const EvalPosition> positions) {
  std::vector<EvalResult> results;
  results.reserve(positions.size());
  std::unique_ptr<BackendComputation> computation = CreateComputation();
  for (const EvalPosition& pos : positions) {
    results.emplace_back();
    EvalResult& result = results.back();
    computation->AddInput(
        pos,
        EvalResultPtr{&result.q, &result.d, &result.m,
                      std::span<float>(result.p.data(), result.p.size())},
        BackendComputation::CACHE_OR_ENQUEUE);
  }
  computation->ComputeBlocking();
  return results;
}

std::optional<EvalResult> Backend::GetCachedEvaluation(
    const EvalPosition& pos) {
  EvalResult result;
  std::unique_ptr<BackendComputation> computation = CreateComputation();
  if (computation->AddInput(
          pos,
          EvalResultPtr{&result.q, &result.d, &result.m,
                        std::span<float>(result.p.data(), result.p.size())},
          BackendComputation::CACHE_ONLY) ==
      BackendComputation::FETCHED_IMMEDIATELY) {
    return result;
  }
  return std::nullopt;
}
}  // namespace lczero