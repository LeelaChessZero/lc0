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

#include <algorithm>
#include <numeric>

#include "utils/exception.h"
#include "utils/logging.h"

namespace lczero {
namespace {

pblczero::XlaShapeProto::Type PjrtTypeToXlaType(PjrtType type) {
  switch (type) {
    case PjrtType::PRED:
    case PjrtType::S8:
    case PjrtType::S16:
    case PjrtType::S32:
    case PjrtType::S64:
    case PjrtType::U8:
    case PjrtType::U16:
    case PjrtType::U32:
    case PjrtType::U64:
    case PjrtType::F16:
    case PjrtType::F32:
    case PjrtType::F64:
      return static_cast<pblczero::XlaShapeProto::Type>(type);
    case PjrtType::C64:
      return pblczero::XlaShapeProto::C64;
    case PjrtType::C128:
      return pblczero::XlaShapeProto::C128;
    case PjrtType::BF16:
      return pblczero::XlaShapeProto::BF16;
    case PjrtType::F8E5M2:
      return pblczero::XlaShapeProto::F8E5M2;
    case PjrtType::F8E4M3FN:
      return pblczero::XlaShapeProto::F8E4M3FN;
    case PjrtType::F8E4M3FNUZ:
      return pblczero::XlaShapeProto::F8E4M3FNUZ;
    case PjrtType::F8E5M2FNUZ:
      return pblczero::XlaShapeProto::F8E5M2FNUZ;
    default:
      throw Exception("Unsupported PjrtType type " +
                      std::to_string(static_cast<int>(type)));
  }
}

PjrtType XlaTypeToPjrtType(pblczero::XlaShapeProto::Type type) {
  switch (type) {
    case pblczero::XlaShapeProto::PRED:
    case pblczero::XlaShapeProto::S8:
    case pblczero::XlaShapeProto::S16:
    case pblczero::XlaShapeProto::S32:
    case pblczero::XlaShapeProto::S64:
    case pblczero::XlaShapeProto::U8:
    case pblczero::XlaShapeProto::U16:
    case pblczero::XlaShapeProto::U32:
    case pblczero::XlaShapeProto::U64:
    case pblczero::XlaShapeProto::F16:
    case pblczero::XlaShapeProto::F32:
    case pblczero::XlaShapeProto::F64:
      return static_cast<PjrtType>(type);
    case pblczero::XlaShapeProto::C64:
      return PjrtType::C64;
    case pblczero::XlaShapeProto::C128:
      return PjrtType::C128;
    case pblczero::XlaShapeProto::BF16:
      return PjrtType::BF16;
    case pblczero::XlaShapeProto::F8E5M2:
      return PjrtType::F8E5M2;
    case pblczero::XlaShapeProto::F8E4M3FN:
      return PjrtType::F8E4M3FN;
    case pblczero::XlaShapeProto::F8E4M3FNUZ:
      return PjrtType::F8E4M3FNUZ;
    case pblczero::XlaShapeProto::F8E5M2FNUZ:
      return PjrtType::F8E5M2FNUZ;
    default:
      throw Exception("Unsupported XLA type " +
                      pblczero::XlaShapeProto::Type_Name(type));
  }
}
}  // namespace

XlaRunner::XlaRunner(const char* library_path, int device)
    : pjrt_client_(Pjrt(library_path).CreateClient()), device_(device) {
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
  options.mutable_executable_build_options()->set_device_ordinal(device_);
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
        {static_cast<const char*>(input->data()), input->size()},
        XlaTypeToPjrtType(input->type()), input->shape(),
        devices_.at(device_).get()));
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

size_t XlaRunner::GetMaxBatchSize() const { return executables_.back().first; }

std::vector<std::unique_ptr<XlaMutableTensor>> XlaRunner::ExecuteBlocking(
    const std::vector<XlaMutableTensor*>& inputs) {
  if (inputs.size() != 1) {
    throw Exception("Only one input is kinda supported.");
  }
  // Find the smallest batch size that fits the input.
  auto iter = std::find_if(
      executables_.begin(), executables_.end(), [&](const auto& e) {
        return e.first >= static_cast<size_t>(inputs[0]->shape()[0]);
      });
  if (iter == executables_.end()) {
    throw Exception("No executable found for batch size " +
                    std::to_string(inputs[0]->shape()[0]));
  }
  const size_t batch_size = iter->first;
  // Update the shape to match the rounded up batch size. After growing, the
  // batch size must fit within tensor buffer capacity (it's fine to have
  // garbage in the tail of that buffer).
  std::vector<int64_t> new_shape = inputs[0]->shape();
  new_shape[0] = batch_size;
  inputs[0]->Reshape(new_shape);
  // Transfer the input to the device.
  auto input_buffer =
      pjrt_client_
          ->HostToDevice(
              {static_cast<const char*>(inputs[0]->data()), inputs[0]->size()},
              XlaTypeToPjrtType(inputs[0]->type()), new_shape,
              devices_[0].get())
          ->AwaitAndReleaseBuffer();
  // Make a copy to support multiple concurrent calls, not sure if it's needed.
  auto input_buffers = buffers_;
  input_buffers[param_idxs_[0]] = input_buffer.get();
  // Execute!
  auto outputs = iter->second->ExecuteBlocking(input_buffers);

  // Now we need to transfer the outputs back to the host.
  std::vector<std::unique_ptr<XlaMutableTensor>> result;
  result.reserve(outputs.size());
  std::vector<std::unique_ptr<PjrtEvent>> done_events;
  done_events.reserve(outputs.size());
  // Initialte transfers from device to host.
  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto& output = outputs[i];
    auto new_tensor = std::make_unique<XlaMutableTensor>(
        PjrtTypeToXlaType(output->GetType()), output->GetDimensions());
    done_events.push_back(
        output->DeviceToHost(new_tensor->mutable_data(), new_tensor->size()));
    result.push_back(std::move(new_tensor));
  }
  // Wait for the transfers to complete.
  for (size_t i = 0; i < outputs.size(); ++i) done_events[i]->Await();
  return result;
}

}  // namespace lczero
