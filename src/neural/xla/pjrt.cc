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

#include "pjrt_c_api.h"

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
}  // namespace
std::string PjrtKeyValue::value_as_string() const {
  return std::visit([&](const auto& arg) { return value_to_string(arg); },
                    value_);
}

namespace {
template <typename T>
T MakeStruct() {
  T t;
  memset(&t, 0, sizeof(t));
  t.struct_size = sizeof(t);
  return t;
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

class PjrtCommonImpl {
 public:
  PjrtCommonImpl(const PJRT_Api* api) : api_(api) {}
  virtual ~PjrtCommonImpl() = default;

 protected:
  std::string GetErrorMessage(PJRT_Error* error) const {
    auto args = MakeStruct<PJRT_Error_Message_Args>();
    args.error = error;
    api_->PJRT_Error_Message(&args);
    return std::string(args.message, args.message_size);
  }
  PJRT_Error_Code GetErrorCode(PJRT_Error* error) const {
    auto args = MakeStruct<PJRT_Error_GetCode_Args>();
    args.error = error;
    api_->PJRT_Error_GetCode(&args);
    return args.code;
  }
  void DestroyErrorMessage(PJRT_Error* error) const {
    assert(error);
    auto args = MakeStruct<PJRT_Error_Destroy_Args>();
    args.error = error;
    api_->PJRT_Error_Destroy(&args);
  }
  void CheckError(PJRT_Error* error) const {
    if (!error) return;
    PjrtException exception(static_cast<PjrtErrorCode>(GetErrorCode(error)),
                            GetErrorMessage(error));
    DestroyErrorMessage(error);
    throw exception;
  }

  const PJRT_Api* api_;
};

class PjrtClientImpl : public PjrtClient, public PjrtCommonImpl {
 public:
  explicit PjrtClientImpl(const PJRT_Api* api) : PjrtCommonImpl(api) {}
  ~PjrtClientImpl() override {
    auto args = MakeStruct<PJRT_Client_Destroy_Args>();
    args.client = client_;
    CheckError(api_->PJRT_Client_Destroy(&args));
  }

 private:
  PJRT_Client* client_;
};

class PjrtImpl : public Pjrt, public PjrtCommonImpl {
 public:
  explicit PjrtImpl(const char* library_path) : PjrtCommonImpl(nullptr) {
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

  std::vector<PjrtKeyValue> GetAttributes() const override {
    auto args = MakeStruct<PJRT_Plugin_Attributes_Args>();
    CheckError(api_->PJRT_Plugin_Attributes(&args));
    std::vector<PjrtKeyValue> result;
    result.reserve(args.num_attributes);
    for (size_t i = 0; i < args.num_attributes; ++i) {
      result.push_back(MakeKeyValue(args.attributes + i));
    }
    return result;
  }

  std::unique_ptr<PjrtClient> CreateClient() override {
    auto args = MakeStruct<PJRT_Client_Create_Args>();
    CheckError(api_->PJRT_Client_Create(&args));
    return std::make_unique<PjrtClientImpl>(api_);
  }

 private:
  std::pair<int, int> ApiVersion() const {
    return std::make_pair(api_->pjrt_api_version.major_version,
                          api_->pjrt_api_version.minor_version);
  }

  void Initialize() {
    auto args = MakeStruct<PJRT_Plugin_Initialize_Args>();
    CheckError(api_->PJRT_Plugin_Initialize(&args));
  }

  const PJRT_Api* api_;
};

}  // namespace

std::unique_ptr<Pjrt> MakePjrt(const char* library_path) {
  return std::make_unique<PjrtImpl>(library_path);
}

}  // namespace lczero