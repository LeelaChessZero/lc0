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

#include "pjrt.h"

#include <dlfcn.h>

#include <cassert>
#include <cstring>
#include <iostream>
#include <numeric>

#include "pjrt_c_api.h"
#include "utils/logging.h"

namespace lczero {

namespace {
static std::string value_to_string(const std::string& value) { return value; }
static std::string value_to_string(int64_t value) {
  return std::to_string(value);
}
static std::string value_to_string(const std::vector<int64_t>& value) {
  std::string result;
  for (auto v : value) {
    if (!result.empty()) result += ", ";
    result += std::to_string(v);
  }
  return result;
}
static std::string value_to_string(float value) {
  return std::to_string(value);
}
static std::string value_to_string(bool value) {
  return value ? "true" : "false";
}

template <typename T>
T MakeStruct() {
  T t;
  memset(&t, 0, sizeof(t));
  t.struct_size = sizeof(t);
  return t;
}

PJRT_Error_Code GetErrorCode(const PJRT_Api* api, PJRT_Error* error) {
  auto args = MakeStruct<PJRT_Error_GetCode_Args>();
  args.error = error;
  api->PJRT_Error_GetCode(&args);
  return args.code;
}

}  // namespace

std::string PjrtKeyValue::value_as_string() const {
  return std::visit([&](const auto& arg) { return value_to_string(arg); },
                    value_);
}

PjrtKeyValue MakeKeyValue(const PJRT_NamedValue* kv) {
  PjrtKeyValue result;
  result.set_key({kv->name, kv->name_size});
  switch (kv->type) {
    case PJRT_NamedValue_kString:
      result.set_value(std::string(kv->string_value, kv->value_size));
      break;
    case PJRT_NamedValue_kInt64:
      result.set_value(kv->int64_value);
      break;
    case PJRT_NamedValue_kInt64List:
      result.set_value(std::vector<int64_t>(
          kv->int64_array_value, kv->int64_array_value + kv->value_size));
      break;
    case PJRT_NamedValue_kFloat:
      result.set_value(kv->float_value);
      break;
    case PJRT_NamedValue_kBool:
      result.set_value(kv->bool_value);
      break;
  }
  return result;
}

std::string PjrtCommon::GetErrorMessage(PJRT_Error* error) const {
  auto args = MakeStruct<PJRT_Error_Message_Args>();
  args.error = error;
  api_->PJRT_Error_Message(&args);
  return std::string(args.message, args.message_size);
}
void PjrtCommon::DestroyErrorMessage(PJRT_Error* error) const {
  assert(error);
  auto args = MakeStruct<PJRT_Error_Destroy_Args>();
  args.error = error;
  api_->PJRT_Error_Destroy(&args);
}

void PjrtCommon::CheckError(PJRT_Error* error) const {
  if (!error) return;
  PjrtException exception(static_cast<PjrtErrorCode>(GetErrorCode(api_, error)),
                          GetErrorMessage(error));
  DestroyErrorMessage(error);
  throw exception;
}

PjrtExecutable::PjrtExecutable(const PJRT_Api* api,
                               PJRT_LoadedExecutable* executable)
    : PjrtCommon(api), executable_(executable) {
  auto args = MakeStruct<PJRT_LoadedExecutable_GetExecutable_Args>();
  args.loaded_executable = executable_;
  CheckError(api_->PJRT_LoadedExecutable_GetExecutable(&args));

  auto args2 = MakeStruct<PJRT_Executable_NumOutputs_Args>();
  args2.executable = args.executable;
  CheckError(api_->PJRT_Executable_NumOutputs(&args2));
  num_outputs_ = args2.num_outputs;
}

PjrtExecutable::~PjrtExecutable() {
  auto args = MakeStruct<PJRT_LoadedExecutable_Destroy_Args>();
  args.executable = executable_;
  CheckError(api_->PJRT_LoadedExecutable_Destroy(&args));
}

size_t PjrtExecutable::GetNumOutputs() const { return num_outputs_; }

std::vector<std::unique_ptr<PjrtDeviceBuffer>> PjrtExecutable::ExecuteBlocking(
    const std::vector<PjrtDeviceBuffer*>& inputs) {
  auto options = MakeStruct<PJRT_ExecuteOptions>();
  options.num_non_donatable_input_indices = inputs.size();
  std::vector<int64_t> non_donatable_indices(inputs.size());
  // TODO the buffer 0 is actually donatable.
  std::iota(non_donatable_indices.begin(), non_donatable_indices.end(), 0);
  options.non_donatable_input_indices = non_donatable_indices.data();

  auto args = MakeStruct<PJRT_LoadedExecutable_Execute_Args>();
  args.executable = executable_;
  args.options = &options;
  args.num_devices = 1;
  std::vector<PJRT_Buffer*> buffers(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) buffers[i] = inputs[i]->buffer_;
  PJRT_Buffer* const* buffers_ptr = buffers.data();
  args.num_args = inputs.size();
  args.argument_lists = &buffers_ptr;

  std::vector<PJRT_Buffer*> outputs(num_outputs_);
  PJRT_Buffer** outputs_ptr = outputs.data();
  PJRT_Event* event_ptr;
  args.output_lists = &outputs_ptr;
  args.device_complete_events = &event_ptr;
  CheckError(api_->PJRT_LoadedExecutable_Execute(&args));

  PjrtEvent event(api_, event_ptr);
  event.Await();

  std::vector<std::unique_ptr<PjrtDeviceBuffer>> output_buffers;
  output_buffers.reserve(num_outputs_);
  for (size_t i = 0; i < num_outputs_; ++i) {
    output_buffers.push_back(
        std::make_unique<PjrtDeviceBuffer>(api_, outputs[i]));
  }
  return output_buffers;
}

PjrtDevice::PjrtDevice(const PJRT_Api* api, PJRT_Device* device)
    : PjrtCommon(api), device_(device) {
  auto args = MakeStruct<PJRT_Device_GetDescription_Args>();
  args.device = device_;
  CheckError(api_->PJRT_Device_GetDescription(&args));
  description_ = args.device_description;
}

std::string PjrtDevice::ToString() const {
  auto args = MakeStruct<PJRT_DeviceDescription_ToString_Args>();
  args.device_description = description_;
  CheckError(api_->PJRT_DeviceDescription_ToString(&args));
  return {args.to_string, args.to_string_size};
}

PjrtClient::PjrtClient(const PJRT_Api* api, PJRT_Client* client)
    : PjrtCommon(api), client_(client) {}

PjrtClient::~PjrtClient() {
  auto args = MakeStruct<PJRT_Client_Destroy_Args>();
  args.client = client_;
  CheckError(api_->PJRT_Client_Destroy(&args));
}

std::unique_ptr<PjrtExecutable> PjrtClient::CompileHlo(
    std::string_view hlo, std::string_view config) {
  constexpr std::string_view kFormat = "hlo";
  auto program = MakeStruct<PJRT_Program>();
  program.code = const_cast<char*>(hlo.data());
  program.code_size = hlo.size();
  program.format = kFormat.data();
  program.format_size = kFormat.size();

  auto args = MakeStruct<PJRT_Client_Compile_Args>();
  args.client = client_;
  args.program = &program;
  args.compile_options = const_cast<char*>(config.data());
  args.compile_options_size = config.size();
  CheckError(api_->PJRT_Client_Compile(&args));
  return std::make_unique<PjrtExecutable>(api_, args.executable);
}

std::vector<std::unique_ptr<PjrtDevice>> PjrtClient::GetDevices() {
  auto args = MakeStruct<PJRT_Client_Devices_Args>();
  args.client = client_;
  CheckError(api_->PJRT_Client_Devices(&args));
  std::vector<std::unique_ptr<PjrtDevice>> result;
  result.reserve(args.num_devices);
  for (size_t i = 0; i < args.num_devices; ++i) {
    result.push_back(std::make_unique<PjrtDevice>(api_, args.devices[i]));
  }
  return result;
}

PjrtEvent::PjrtEvent(const PJRT_Api* api, PJRT_Event* event)
    : PjrtCommon(api), event_(event) {}

PjrtEvent::~PjrtEvent() {
  auto args = MakeStruct<PJRT_Event_Destroy_Args>();
  args.event = event_;
  CheckError(api_->PJRT_Event_Destroy(&args));
}

void PjrtEvent::Await() {
  auto args = MakeStruct<PJRT_Event_Await_Args>();
  args.event = event_;
  CheckError(api_->PJRT_Event_Await(&args));
}

PjrtDeviceBuffer::PjrtDeviceBuffer(const PJRT_Api* api, PJRT_Buffer* buffer)
    : PjrtCommon(api), buffer_(buffer) {}

PjrtDeviceBuffer::~PjrtDeviceBuffer() {
  auto args = MakeStruct<PJRT_Buffer_Destroy_Args>();
  args.buffer = buffer_;
  CheckError(api_->PJRT_Buffer_Destroy(&args));
}

size_t PjrtDeviceBuffer::GetSize() const {
  auto args = MakeStruct<PJRT_Buffer_ToHostBuffer_Args>();
  args.src = buffer_;
  CheckError(api_->PJRT_Buffer_ToHostBuffer(&args));
  return args.dst_size;
}

PjrtType PjrtDeviceBuffer::GetType() const {
  auto args = MakeStruct<PJRT_Buffer_ElementType_Args>();
  args.buffer = buffer_;
  CheckError(api_->PJRT_Buffer_ElementType(&args));
  return static_cast<PjrtType>(args.type);
}

std::vector<int64_t> PjrtDeviceBuffer::GetDimensions() const {
  auto args = MakeStruct<PJRT_Buffer_Dimensions_Args>();
  args.buffer = buffer_;
  CheckError(api_->PJRT_Buffer_Dimensions(&args));
  return {args.dims, args.dims + args.num_dims};
}

std::unique_ptr<PjrtEvent> PjrtDeviceBuffer::DeviceToHost(void* dst,
                                                          size_t size) {
  auto args = MakeStruct<PJRT_Buffer_ToHostBuffer_Args>();
  args.src = buffer_;
  args.dst = dst;
  args.dst_size = size;
  CheckError(api_->PJRT_Buffer_ToHostBuffer(&args));
  return std::make_unique<PjrtEvent>(api_, args.event);
}

PjrtHostToDeviceTransfer::PjrtHostToDeviceTransfer(
    const PJRT_Api* api, PJRT_Buffer* buffer, std::unique_ptr<PjrtEvent> event)
    : PjrtCommon(api), buffer_(buffer), event_(std::move(event)) {}

void PjrtHostToDeviceTransfer::Await() { event_->Await(); }

std::unique_ptr<PjrtDeviceBuffer>
PjrtHostToDeviceTransfer::AwaitAndReleaseBuffer() {
  if (!buffer_) {
    throw PjrtException(PjrtErrorCode::INVALID_ARGUMENT,
                        "Buffer already released");
  }
  Await();
  auto res = std::make_unique<PjrtDeviceBuffer>(api_, buffer_);
  buffer_ = nullptr;
  return res;
}

PjrtHostToDeviceTransfer::~PjrtHostToDeviceTransfer() {
  Await();
  if (buffer_) {
    auto args = MakeStruct<PJRT_Buffer_Destroy_Args>();
    args.buffer = buffer_;
    CheckError(api_->PJRT_Buffer_Destroy(&args));
  }
}

std::unique_ptr<PjrtHostToDeviceTransfer> PjrtClient::HostToDevice(
    std::string_view buffer, PjrtType type, const std::vector<int64_t>& dims,
    const PjrtDevice* device) {
  auto args = MakeStruct<PJRT_Client_BufferFromHostBuffer_Args>();
  args.client = client_;
  args.data = buffer.data();
  args.type = static_cast<PJRT_Buffer_Type>(type);
  args.dims = dims.data();
  args.num_dims = dims.size();
  args.host_buffer_semantics =
      PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
  args.device = device->device_;
  CheckError(api_->PJRT_Client_BufferFromHostBuffer(&args));
  auto event = std::make_unique<PjrtEvent>(api_, args.done_with_host_buffer);
  return std::make_unique<PjrtHostToDeviceTransfer>(api_, args.buffer,
                                                    std::move(event));
}

Pjrt::Pjrt(const char* library_path) : PjrtCommon(nullptr) {
  // TODO factor out the dlopen/dlsym code into a separate function, and
  // implement for other OSes.
  void* handle = dlopen(library_path, RTLD_LAZY);
  if (!handle) {
    throw PjrtException(PjrtErrorCode::INVALID_ARGUMENT,
                        "Unable to load PJRT library " +
                            std::string(library_path) + ": " + dlerror());
  }
  typedef const PJRT_Api* (*PjrtApiFunc)();
  auto func = reinterpret_cast<PjrtApiFunc>(dlsym(handle, "GetPjrtApi"));
  if (!func) {
    throw PjrtException(PjrtErrorCode::INVALID_ARGUMENT,
                        "Unable to find GetPjrtApi() in PJRT library " +
                            std::string(library_path) + ": " + dlerror());
  }
  api_ = func();
  if (!api_) {
    throw PjrtException(PjrtErrorCode::INVALID_ARGUMENT,
                        "GetPjrtApi() returned nullptr in PJRT library " +
                            std::string(library_path));
  }
  auto [major, minor] = ApiVersion();
  if (major != PJRT_API_MAJOR || minor < PJRT_API_MINOR) {
    throw PjrtException(
        PjrtErrorCode::INVALID_ARGUMENT,
        "PJRT library " + std::string(library_path) +
            " has incompatible API version: " + std::to_string(major) + "." +
            std::to_string(minor) + " vs " + std::to_string(PJRT_API_MAJOR) +
            "." + std::to_string(PJRT_API_MINOR));
  }
  Initialize();
}

std::vector<PjrtKeyValue> Pjrt::GetAttributes() const {
  auto args = MakeStruct<PJRT_Plugin_Attributes_Args>();
  CheckError(api_->PJRT_Plugin_Attributes(&args));
  std::vector<PjrtKeyValue> result;
  result.reserve(args.num_attributes);
  for (size_t i = 0; i < args.num_attributes; ++i) {
    result.push_back(MakeKeyValue(args.attributes + i));
  }
  return result;
}

std::unique_ptr<PjrtClient> Pjrt::CreateClient() {
  auto args = MakeStruct<PJRT_Client_Create_Args>();
  CheckError(api_->PJRT_Client_Create(&args));
  return std::make_unique<PjrtClient>(api_, args.client);
}

std::pair<int, int> Pjrt::ApiVersion() const {
  return std::make_pair(api_->pjrt_api_version.major_version,
                        api_->pjrt_api_version.minor_version);
}

void Pjrt::Initialize() {
  auto args = MakeStruct<PJRT_Plugin_Initialize_Args>();
  CheckError(api_->PJRT_Plugin_Initialize(&args));
}

}  // namespace lczero