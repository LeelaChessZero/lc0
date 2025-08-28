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

// This file contains set of C++ wrappers around the PJRT C API.

#pragma once

#include <memory>
#include <stdexcept>
#include <variant>
#include <vector>

struct PJRT_Api;
struct PJRT_Buffer;
struct PJRT_Client;
struct PJRT_Device;
struct PJRT_DeviceDescription;
struct PJRT_Error;
struct PJRT_Event;
struct PJRT_LoadedExecutable;

namespace lczero {

// PJRT_Error_Code as enum class. Coincidentally, the error codes are the same
// as in absl Status module.
enum class PjrtErrorCode {
  CANCELLED = 1,
  UNKNOWN = 2,
  INVALID_ARGUMENT = 3,
  DEADLINE_EXCEEDED = 4,
  NOT_FOUND = 5,
  ALREADY_EXISTS = 6,
  PERMISSION_DENIED = 7,
  RESOURCE_EXHAUSTED = 8,
  FAILED_PRECONDITION = 9,
  ABORTED = 10,
  OUT_OF_RANGE = 11,
  UNIMPLEMENTED = 12,
  INTERNAL = 13,
  UNAVAILABLE = 14,
  DATA_LOSS = 15,
  UNAUTHENTICATED = 16
};

// PJRT_Type as enum class. Unfortunately, only the first 13 types are the same
// as in XLA and HloModuleProto, so simple cast doesn't work.
enum class PjrtType {
  INVALID,
  PRED,
  S8,
  S16,
  S32,
  S64,
  U8,
  U16,
  U32,
  U64,
  F16,
  F32,
  F64,
  BF16,
  C64,
  C128,
  F8E5M2,
  F8E4M3FN,
  F8E4M3B11FNUZ,
  F8E5M2FNUZ,
  F8E4M3FNUZ,
  S4,
  U4,
};

// PJRT errors as exceptions.
class PjrtException : public std::exception {
 public:
  explicit PjrtException(PjrtErrorCode code, const std::string& message)
      : message_(message), code_(code) {}

  const char* what() const noexcept override { return message_.data(); }
  PjrtErrorCode code() const { return code_; }

 private:
  std::string message_;
  PjrtErrorCode code_;
};

// PJRT_NamedValue wrapper. PJRT_NamedValue is a string-keyed values that are
// used for auxiliary functionality like plugin attributes.
class PjrtKeyValue {
 public:
  PjrtKeyValue() = default;
  PjrtKeyValue(const PjrtKeyValue&) = default;
  PjrtKeyValue(PjrtKeyValue&&) = default;
  template <typename T>
  PjrtKeyValue(const std::string& k, const T& v) : key_(k), value_(v) {}

  const std::string& key() const { return key_; }
  // Converts the value to string. This is useful for logging and debugging.
  std::string value_as_string() const;

  void set_key(const std::string& key) { key_ = key; }
  void set_value(const std::string& value) { value_ = value; }
  void set_value(int64_t value) { value_ = value; }
  void set_value(const std::vector<int64_t>& value) { value_ = value; }
  void set_value(float value) { value_ = value; }
  void set_value(bool value) { value_ = value; }

 private:
  std::string key_;
  std::variant<std::string, int64_t, std::vector<int64_t>, float, bool> value_;
};

// A shared base class for all wrappers. Keeps the API pointer and auxiliary
// functions like error checking.
class PjrtCommon {
 protected:
  PjrtCommon(const PJRT_Api* api) : api_(api) {}
  virtual ~PjrtCommon() = default;

  std::string GetErrorMessage(PJRT_Error* error) const;
  void DestroyErrorMessage(PJRT_Error* error) const;
  void CheckError(PJRT_Error* error) const;

  const PJRT_Api* api_;
};

class PjrtDevice : protected PjrtCommon {
 public:
  PjrtDevice(const PJRT_Api* api, PJRT_Device* device);
  std::string ToString() const;

 private:
  PJRT_Device* device_;
  PJRT_DeviceDescription* description_;
  friend class PjrtExecutable;
  friend class PjrtClient;
};

// An event for waiting for asynchronous operations.
class PjrtEvent : protected PjrtCommon {
 public:
  PjrtEvent(const PJRT_Api* api, PJRT_Event* event);
  // Blocks until the operation is complete.
  void Await();
  ~PjrtEvent();

 private:
  PJRT_Event* event_;
};

// A buffer in the device memory.
class PjrtDeviceBuffer : protected PjrtCommon {
 public:
  PjrtDeviceBuffer(const PJRT_Api* api, PJRT_Buffer* buffer);
  ~PjrtDeviceBuffer();
  // Returns the size of the buffer in bytes.
  size_t GetSize() const;
  // Starts an asynchronous copy from the device to the host memory.
  [[nodiscard]] std::unique_ptr<PjrtEvent> DeviceToHost(void* dst, size_t size);
  PjrtType GetType() const;
  std::vector<int64_t> GetDimensions() const;

 private:
  PJRT_Buffer* buffer_;
  friend class PjrtExecutable;
};

class PjrtExecutable : protected PjrtCommon {
 public:
  PjrtExecutable(const PJRT_Api* api, PJRT_LoadedExecutable* executable);
  ~PjrtExecutable();
  // Executes the executable with the given inputs. The inputs are not owned or
  // modified. The function allocates the output buffers and returns them.
  std::vector<std::unique_ptr<PjrtDeviceBuffer>> ExecuteBlocking(
      const std::vector<PjrtDeviceBuffer*>& inputs);
  size_t GetNumOutputs() const;

 private:
  PJRT_LoadedExecutable* executable_;
  size_t num_outputs_;
};

// Ongoing host-to-device transfer. After the transfer is complete, it's
// possible to fetch the device buffer.
class PjrtHostToDeviceTransfer : protected PjrtCommon {
 public:
  PjrtHostToDeviceTransfer(const PJRT_Api* api, PJRT_Buffer* buffer,
                           std::unique_ptr<PjrtEvent> event);
  ~PjrtHostToDeviceTransfer();
  // Blocks until the transfer is complete. (not really necessary as
  // AwaitAndReleaseBuffer() waits anyway)
  void Await();
  // Waits for the transfer to complete and releases the ownership of the
  // buffer.
  std::unique_ptr<PjrtDeviceBuffer> AwaitAndReleaseBuffer();

 private:
  PJRT_Buffer* buffer_;
  std::unique_ptr<PjrtEvent> event_;
};

class PjrtClient : protected PjrtCommon {
 public:
  PjrtClient(const PJRT_Api* api, PJRT_Client* client);
  ~PjrtClient();
  std::unique_ptr<PjrtExecutable> CompileHlo(std::string_view hlo,
                                             std::string_view config);
  std::vector<std::unique_ptr<PjrtDevice>> GetDevices();
  std::unique_ptr<PjrtHostToDeviceTransfer> HostToDevice(
      std::string_view buffer, PjrtType type, const std::vector<int64_t>& dims,
      const PjrtDevice* device);

 private:
  PJRT_Client* client_;
};

class Pjrt : protected PjrtCommon {
 public:
  Pjrt(const char* library_path);
  std::vector<PjrtKeyValue> GetAttributes() const;
  std::unique_ptr<PjrtClient> CreateClient();
  std::pair<int, int> ApiVersion() const;

 private:
  void Initialize();
};

}  // namespace lczero
