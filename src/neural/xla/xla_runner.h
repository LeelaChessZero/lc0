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
#include "neural/xla/xla_tensor.h"

namespace lczero {

// A class that keeps several XLA executables (for different batch sizes),
// manages common buffers among them, and chooses the right executable for a
// batch size.
class XlaRunner {
 public:
  // The library_path is the path to the PJRT library, and device indx.
  XlaRunner(const char* library_path, int device);
  // Compiles and adds a module for the given batch size.
  void AddModule(size_t minibatch_size, const pblczero::HloModuleProto& module);
  // Transfers inputs to the device and execute the executable corresponding to
  // the batch size. Only non-frozen inputs are passed as arguments.
  // Currnetly only single input is supported (just because we don't need more).
  std::vector<std::unique_ptr<XlaMutableTensor>> ExecuteBlocking(
      const std::vector<XlaMutableTensor*>& inputs);
  // Inputs that are shared between all calls (i.e. network weights passed as
  // parameters). These inputs are transferred to device immediately (and not
  // for each inference).
  void SetFrozenInputs(const std::vector<std::unique_ptr<XlaTensor>> inputs);
  // Maximum supported batch size. It's expected that the capacity (not size) of
  // the input tensors would be able to fit this size.
  size_t GetMaxBatchSize() const;

 private:
  std::unique_ptr<PjrtClient> pjrt_client_;
  std::vector<std::unique_ptr<PjrtDevice>> devices_;
  // Compiled executables per batch size.
  std::vector<std::pair<size_t, std::unique_ptr<PjrtExecutable>>> executables_;
  // Frozen inputs, in no particular order, kept for ownership.
  std::vector<std::unique_ptr<PjrtDeviceBuffer>> owned_buffers_;
  // Vector of pointers to all input buffers, that is passed to PJRT. Frozen
  // parameters (constants) are pre-filled in SetFrozenInputs(), and non-frozen
  // inputs (input planes) are created and filled in every request.
  std::vector<PjrtDeviceBuffer*> buffers_;
  std::vector<size_t> param_idxs_;
  int device_;
};

}  // namespace lczero