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

#include "gmock/gmock.h"

#include "neural/backend.h"

namespace lczero {

class MockBackendComputation : public BackendComputation {
 public:
  MOCK_METHOD(size_t, UsedBatchSize, (), (const, override));
  MOCK_METHOD(AddInputResult, AddInput,
              (const EvalPosition& pos, EvalResultPtr result), (override));
  MOCK_METHOD(void, ComputeBlocking, (), (override));
};

class MockBackend : public Backend {
 public:
  MOCK_METHOD(BackendAttributes, GetAttributes, (), (const, override));
  MOCK_METHOD(std::unique_ptr<BackendComputation>, CreateComputation, (),
              (override));
  MOCK_METHOD(std::vector<EvalResult>, EvaluateBatch,
              (std::span<const EvalPosition> positions), (override));
  MOCK_METHOD(std::optional<EvalResult>, GetCachedEvaluation,
              (const EvalPosition&), (override));
  MOCK_METHOD(UpdateConfigurationResult, UpdateConfiguration,
              (const OptionsDict&), (override));
};

class MockBackendFactory : public BackendFactory {
 public:
  MOCK_METHOD(int, GetPriority, (), (const, override));
  MOCK_METHOD(std::string_view, GetName, (), (const, override));
  MOCK_METHOD(std::unique_ptr<Backend>, Create, (const OptionsDict&),
              (override));
};

}  // namespace lczero
