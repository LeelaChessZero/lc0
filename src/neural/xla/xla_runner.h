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
#include "neural/xla/pjrt.h"

namespace lczero {

class XlaTensor {
 public:
  virtual ~XlaTensor() = default;
  virtual const std::vector<int64_t>& shape() const = 0;
  virtual std::string_view data() const = 0;
  virtual pblczero::XlaShapeProto::Type type() const = 0;
};

class XlaRunner {
 public:
  XlaRunner(const char* library_path);
  void AddModule(size_t minibatch_size, const pblczero::HloModuleProto& module);
  std::vector<XlaTensor> ExecuteBlocking(
      const std::vector<const XlaTensor*>& inputs);

  // Network weights are passed as inputs, but the buffer is transferred once
  // before any inference.
  void SetFrozenInputs(const std::vector<std::unique_ptr<XlaTensor>> inputs);

 private:
  std::unique_ptr<PjrtClient> pjrt_client_;
  std::vector<std::unique_ptr<PjrtDevice>> devices_;
  std::vector<std::pair<size_t, std::unique_ptr<PjrtExecutable>>> executables_;
  std::vector<std::unique_ptr<PjrtDeviceBuffer>> owned_buffers_;
  std::vector<PjrtDeviceBuffer*> buffers_;
  std::vector<size_t> param_idxs_;
};

}  // namespace lczero