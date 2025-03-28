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

#include <memory>
#include <vector>

#include "neural/backend.h"

#pragma once

namespace lczero {

class BackendManager {
 public:
  static BackendManager* Get();
  void AddBackend(std::unique_ptr<BackendFactory> factory) {
    algorithms_.push_back(std::move(factory));
  }

  // Returns list of backend names, sorted by priority (higher priority first).
  std::vector<std::string> GetBackendNames() const;

  // Creates a backend from the parameters. Extracts the weights file and the
  // backend from the options.
  std::unique_ptr<Backend> CreateFromParams(const OptionsDict& options) const;

  // Creates a backend from the name. Backend name from the options is ignored.
  // Note that unlike the WeightsFactory, the "options" parameter contains
  // top-level parameters rather than `backend-opts`.
  std::unique_ptr<Backend> CreateFromName(std::string_view name,
                                          const OptionsDict& options) const;

  // Returns a backend factory by name. Returns nullptr if not found.
  BackendFactory* GetFactoryByName(std::string_view name) const;

  struct Register {
    Register(std::unique_ptr<BackendFactory> factory) {
      BackendManager::Get()->AddBackend(std::move(factory));
    }
  };

 private:
  BackendManager() = default;

  std::vector<std::unique_ptr<BackendFactory>> algorithms_;
};

#define REGISTER_BACKEND(factory)                   \
  namespace {                                       \
  static SearchFactory::Register reg29c93##factory( \
      std::make_unique<factory>());                 \
  }
}  // namespace lczero
