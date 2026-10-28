/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "proto/lc0ex.pb.h"

namespace lczero {
namespace lc0ex {

// Handles and descriptor references returned by an Executable remain valid
// until that Executable is destroyed. An Execution and its buffers must not
// outlive their Executable. Distinct Executions may be used concurrently; an
// individual Execution must only be accessed by one host thread at a time.
struct TargetInfo {
  pblczero::Target::Vendor vendor = pblczero::Target::VENDOR_UNKNOWN;
  std::string architecture;
};

struct BufferInfo {
  std::string name;
  pblczero::Buffer::DataType data_type = pblczero::Buffer::DATA_TYPE_UNKNOWN;
  std::vector<std::uint64_t> shape;
  std::vector<std::int64_t> strides;
  std::uint64_t size_bytes = 0;
  std::uint64_t offset_bytes = 0;
};

struct ParameterInfo {
  std::string name;
  pblczero::ParameterType type = pblczero::ParameterType_PARAMETER_TYPE_UNKNOWN;
};

struct ProgramInfo {
  std::string name;
  std::string metadata;
};

class Buffer {
 public:
  virtual ~Buffer() = default;

  virtual const BufferInfo& GetInfo() const = 0;
  // Host copies complete before returning, and therefore accept ordinary
  // pageable host memory. If size_bytes is specified, only that prefix is
  // copied.
  virtual void CopyFromHost(
      std::span<const std::byte> source,
      std::optional<std::size_t> size_bytes = std::nullopt) = 0;
  virtual void CopyToHost(
      std::span<std::byte> destination,
      std::optional<std::size_t> size_bytes = std::nullopt) const = 0;
};

class Parameter {
 public:
  virtual ~Parameter() = default;

  virtual const ParameterInfo& GetInfo() const = 0;
  virtual bool IsSet() const = 0;
  virtual void Set(std::uint32_t value) = 0;
  virtual void Set(const Buffer& buffer) = 0;
  virtual void Reset() = 0;
};

class Program {
 public:
  virtual ~Program() = default;

  virtual const ProgramInfo& GetInfo() const = 0;
  virtual std::span<const BufferInfo> GetBuffers() const = 0;
  virtual const BufferInfo* FindBuffer(std::string_view name) const = 0;
  virtual std::span<const ParameterInfo> GetParameters() const = 0;
};

// A reusable instance of one Program. It exclusively leases a stream and an
// execution allocation until it is destroyed. The allocation may be larger
// than this Program's requirements and may be reused by another Execution.
// Run() submits asynchronously; the Execution may be modified or run again
// only after Synchronize().
class Execution {
 public:
  virtual ~Execution() = default;

  // Only buffers belonging to this Execution's Program are available.
  virtual Buffer& GetBuffer(const BufferInfo& info) = 0;
  virtual Parameter& GetParameter(std::string_view name) = 0;
  virtual void ResetParameters() = 0;
  virtual void Run() = 0;
  virtual void Synchronize() = 0;
};

class Executable {
 public:
  virtual ~Executable() = default;

  virtual const TargetInfo& GetTarget() const = 0;
  virtual std::string_view GetMetadata() const = 0;
  // Only persistent buffers are available through an Executable.
  virtual std::span<const BufferInfo> GetBuffers() const = 0;
  virtual std::span<const ParameterInfo> GetParameters() const = 0;
  virtual std::span<const ProgramInfo> GetPrograms() const = 0;

  virtual const BufferInfo* FindBuffer(std::string_view name) const = 0;
  virtual const ParameterInfo* FindParameter(std::string_view name) const = 0;
  virtual const Program* FindProgram(std::string_view name) const = 0;

  virtual std::size_t GetPersistentAllocationSize() const = 0;
  virtual void CopyPersistentFromHost(
      std::span<const std::byte> source,
      std::optional<std::size_t> size_bytes = std::nullopt) = 0;

  // Persistent storage is shared by all Executions; callers must not modify it
  // while an Execution that may access it is in flight.
  virtual Buffer& GetBuffer(const BufferInfo& info) = 0;

  virtual std::unique_ptr<Execution> CreateExecution(
      const Program& program) = 0;
};

class Runtime {
 public:
  virtual ~Runtime() = default;

  virtual std::unique_ptr<Executable> Load(
      const pblczero::NeuralExecutable& executable) = 0;
};

}  // namespace lc0ex
}  // namespace lczero
