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
#include <stdexcept>
#include <variant>
#include <vector>

namespace lczero {

// PJRT_Error_Code as enum class.
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

// PJRT_NamedValue wrapper.
class PjrtKeyValue {
 public:
  PjrtKeyValue() = default;
  PjrtKeyValue(const PjrtKeyValue&) = default;
  PjrtKeyValue(PjrtKeyValue&&) = default;
  template <typename T>
  PjrtKeyValue(const std::string& k, const T& v) : key_(k), value_(v) {}

  const std::string& key() const { return key_; }
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

class PjrtDeviceBuffer {
 public:
  virtual ~PjrtDeviceBuffer() = default;
};

class PjrtExecutable {
 public:
  virtual ~PjrtExecutable() = default;
};

class PjrtDevice {
 public:
  virtual ~PjrtDevice() = default;
  virtual std::string ToString() const = 0;
};

class PjrtHostToDeviceTransfer {
 public:
  virtual ~PjrtHostToDeviceTransfer() = default;
  virtual void WaitUntilDone() = 0;
  virtual std::unique_ptr<PjrtDeviceBuffer> WaitAndReleaseBuffer() = 0;
};

class PjrtClient {
 public:
  virtual ~PjrtClient() = default;
  virtual std::unique_ptr<PjrtExecutable> CompileHlo(
      std::string_view hlo, std::string_view config) = 0;
  virtual std::vector<std::unique_ptr<PjrtDevice>> GetDevices() = 0;
  virtual std::unique_ptr<PjrtHostToDeviceTransfer> HostToDevice(
      std::string_view buffer, PjrtType type, const std::vector<int64_t>& dims,
      const PjrtDevice* device);
};

class Pjrt {
 public:
  virtual ~Pjrt() = default;
  // PJRT_Plugin_Attributes wrapper.
  virtual std::vector<PjrtKeyValue> GetAttributes() const = 0;
  // PJRT_Client_Create wrapper.
  virtual std::unique_ptr<PjrtClient> CreateClient() = 0;
};

// Loads the PJRT plugin from the given library path.
std::unique_ptr<Pjrt> MakePjrt(const char* library_path);

}  // namespace lczero