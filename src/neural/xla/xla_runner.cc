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

#include <numeric>

#include "utils/exception.h"
#include "utils/logging.h"

namespace lczero {
namespace {

size_t GetTypeSize(pblczero::XlaShapeProto::Type type) {
  switch (type) {
    case pblczero::XlaShapeProto::F32:
      return sizeof(float);
    case pblczero::XlaShapeProto::F64:
      return sizeof(double);
    case pblczero::XlaShapeProto::S32:
      return sizeof(int32_t);
    case pblczero::XlaShapeProto::S64:
      return sizeof(int64_t);
    default:
      throw Exception("Add size for type " +
                      pblczero::XlaShapeProto::Type_Name(type));
  }
}

std::string AsHexString(std::string_view buf) {
  std::string result;
  result.reserve(buf.size() * 2);
  constexpr char hex[] = "0123456789abcdef";
  for (unsigned char c : buf) {
    result.push_back(hex[c >> 4]);
    result.push_back(hex[c & 0xf]);
  }
  return result;
}

}  // namespace

std::string XlaTensor::DebugString() {
  constexpr size_t kMaxSize = 1000;
  constexpr size_t kSuffixSize = 200;
  std::string result = "XlaTensor(";
  result += "shape=[";
  for (size_t i = 0; i < shape().size(); ++i) {
    if (i > 0) result += ", ";
    result += std::to_string(shape()[i]);
  }
  result += "], type=";
  result += pblczero::XlaShapeProto::Type_Name(type());
  result += ") size=" + std::to_string(size());
  result += " data=";
  if (size() <= kMaxSize) {
    result += AsHexString({static_cast<const char*>(data()), size()});
  } else {
    result += AsHexString(
        {static_cast<const char*>(data()), kMaxSize - kSuffixSize - 2});
    result += "....";
    result += AsHexString(
        {static_cast<const char*>(data()) + size() - kSuffixSize, kSuffixSize});
  }
  return result;
}

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
        static_cast<PjrtType>(input->type()), input->shape(),
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

std::vector<std::unique_ptr<XlaTensor>> XlaRunner::ExecuteBlocking(
    const std::vector<XlaTensor*>& inputs) {
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
  const size_t input_size = std::accumulate(new_shape.begin(), new_shape.end(),
                                            1, std::multiplies<size_t>()) *
                            GetTypeSize(inputs[0]->type());
  if (input_size > inputs[0]->capacity()) {
    throw Exception("Input buffer too small");
  }
  // Transfer the input to the device.
  auto input_buffer =
      pjrt_client_
          ->HostToDevice(
              {static_cast<const char*>(inputs[0]->data()), input_size},
              static_cast<PjrtType>(inputs[0]->type()), new_shape,
              devices_[0].get())
          ->AwaitAndReleaseBuffer();
  // Make a copy to support multiple concurrent calls, not sure if it's needed.
  auto input_buffers = buffers_;
  input_buffers[param_idxs_[0]] = input_buffer.get();
  // Execute!
  auto outputs = iter->second->ExecuteBlocking(input_buffers);

  // Now we need to transfer the outputs back to the host.
  std::vector<std::unique_ptr<XlaTensor>> result;
  result.reserve(outputs.size());
  std::vector<std::string> output_buffers;
  std::vector<std::unique_ptr<PjrtEvent>> done_events;
  output_buffers.reserve(outputs.size());
  done_events.reserve(outputs.size());
  // Initialte transfers from device to host.
  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto& output = outputs[i];
    output_buffers.emplace_back();
    auto& buffer = output_buffers.back();
    buffer.resize(output->GetSize());
    done_events.push_back(output->DeviceToHost(&buffer[0], buffer.size()));
  }
  // Wait for the transfers to complete.
  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto& output = outputs[i];
    done_events[i]->Await();
    result.push_back(std::make_unique<XlaTensorOwned>(
        output->GetDimensions(),
        static_cast<pblczero::XlaShapeProto::Type>(output->GetType()),
        std::move(output_buffers[i])));
  }
  return result;
}

}  // namespace lczero