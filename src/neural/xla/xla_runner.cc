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

#include "neural/xla/xla_runner.h"

#include "utils/exception.h"
#include "utils/logging.h"

namespace lczero {

XlaRunner::XlaRunner(const char* library_path)
    : pjrt_client_(MakePjrt(library_path)->CreateClient()) {
  CERR << "Devices:";
  devices_ = pjrt_client_->GetDevices();
  for (const auto& device : devices_) {
    CERR << "  " << device->ToString();
  }
  if (devices_.empty()) {
    throw Exception("No devices available");
  }
}

void XlaRunner::AddModule(size_t minibatch_size,
                          const pblczero::HloModuleProto& module) {
  pblczero::CompileOptionsProto options;
  options.mutable_executable_build_options()->set_num_replicas(1);
  options.mutable_executable_build_options()->set_num_partitions(1);
  auto executable = pjrt_client_->CompileHlo(module.OutputAsString(),
                                             options.OutputAsString());
  executables_.push_back({minibatch_size, std::move(executable)});
  std::sort(executables_.begin(), executables_.end());
}

void XlaRunner::SetFrozenInputs(
    const std::vector<std::unique_ptr<XlaTensor>> inputs) {
  param_idxs_.clear();
  std::vector<std::unique_ptr<PjrtHostToDeviceTransfer>> transfers_;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto* input = inputs[i].get();
    if (!input) {
      param_idxs_.push_back(i);
      continue;
    }
    transfers_.push_back(pjrt_client_->HostToDevice(
        input->data(), static_cast<PjrtType>(input->type()), input->shape(),
        devices_[0].get()));
  }

  owned_buffers_.clear();
  buffers_.clear();
  buffers_.resize(inputs.size());
  size_t transfer_idx = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i]) {
      owned_buffers_.push_back(
          transfers_[transfer_idx++]->AwaitAndReleaseBuffer());
      buffers_[i] = owned_buffers_.back().get();
    }
  }
}

}  // namespace lczero