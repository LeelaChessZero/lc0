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

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "neural/xla/hlo.pb.h"

namespace lczero {

class XlaTensor {};

class XlaModule {
 public:
  size_t AddPlaceholder(const pblczero::XlaShapeProto& shape);
  size_t AddSharedTensor(const XlaTensor& buffer);
  void Compile(const pblczero::HloModuleProto& hlo_module);
  std::vector<XlaTensor> Run(std::vector<XlaTensor> inputs);
};

class XlaRunner {
 public:
  void AddModule(size_t max_batch_size, std::unique_ptr<XlaModule> module);

  void AddSharedTensor(const std::string& name,
                       std::unique_ptr<XlaTensor> buffer);
  std::optional<XlaTensor*> GetSharedTensor(const std::string& name);

 private:
};

}  // namespace lczero