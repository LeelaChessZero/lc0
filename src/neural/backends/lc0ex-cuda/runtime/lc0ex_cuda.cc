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

#include "lc0ex_cuda.h"

#include <cuda.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "utils/exception.h"

namespace lczero {
namespace lc0ex {
namespace {

constexpr std::uint32_t kMagic = 0x1c0e;
constexpr std::uint32_t kFormat = 1;

[[noreturn]] void ThrowCuda(CUresult status, const char* expression,
                            const char* file, int line) {
  const char* name = nullptr;
  const char* description = nullptr;
  cuGetErrorName(status, &name);
  cuGetErrorString(status, &description);

  std::string message = expression;
  message += " failed at ";
  message += file;
  message += ":";
  message += std::to_string(line);
  message += ": ";
  message += name ? name : "unknown CUDA error";
  if (description) {
    message += " (";
    message += description;
    message += ")";
  }
  throw Exception(message);
}

#define LC0EX_CUDA_CHECK(expression)                            \
  do {                                                          \
    const CUresult lc0ex_status = (expression);                 \
    if (lc0ex_status != CUDA_SUCCESS)                           \
      ThrowCuda(lc0ex_status, #expression, __FILE__, __LINE__); \
  } while (false)

void IgnoreCuda(CUresult status) { (void)status; }

std::size_t ElementSize(pblczero::Buffer::DataType type) {
  switch (type) {
    case pblczero::Buffer::DATA_TYPE_F32:
      return sizeof(float);
    case pblczero::Buffer::DATA_TYPE_U8:
      return sizeof(std::uint8_t);
    case pblczero::Buffer::DATA_TYPE_F16:
      return sizeof(std::uint16_t);
    case pblczero::Buffer::DATA_TYPE_U64:
      return sizeof(std::uint64_t);
    case pblczero::Buffer::DATA_TYPE_BF16:
      return sizeof(std::uint16_t);
    case pblczero::Buffer::DATA_TYPE_UNKNOWN:
      break;
  }
  throw Exception("Unsupported or unknown buffer data type.");
}

std::uint64_t BufferSize(const pblczero::Buffer& buffer) {
  std::uint64_t elements = 1;
  for (const auto dimension : buffer.shape()) {
    elements *= dimension;
  }
  return elements * ElementSize(buffer.data_type());
}

std::vector<std::int64_t> DefaultStrides(
    const std::vector<std::uint64_t>& shape) {
  std::vector<std::int64_t> strides(shape.size(), 1);
  if (shape.empty()) return strides;
  for (std::size_t i = shape.size() - 1; i > 0; --i) {
    strides[i - 1] = strides[i] * static_cast<std::int64_t>(shape[i]);
  }
  return strides;
}

BufferInfo MakeBufferInfo(const pblczero::Buffer& buffer) {
  std::vector<std::uint64_t> shape(buffer.shape().begin(),
                                   buffer.shape().end());
  std::vector<std::int64_t> strides;
  if (buffer.has_layout() && buffer.layout().strides_size() > 0) {
    strides.assign(buffer.layout().strides().begin(),
                   buffer.layout().strides().end());
  } else {
    strides = DefaultStrides(shape);
  }
  return {
      std::string(buffer.name()),
      buffer.data_type(),
      std::move(shape),
      std::move(strides),
      BufferSize(buffer),
      buffer.offset(),
  };
}

void CopyStridedHtoD(const std::vector<std::uint64_t>& shape,
                     const std::vector<std::int64_t>& dst_strides,
                     const std::vector<std::int64_t>& src_strides,
                     std::size_t elem_size, const std::byte* src,
                     CUdeviceptr dst) {
  if (shape.empty()) {
    LC0EX_CUDA_CHECK(cuMemcpyHtoD(dst, src, elem_size));
    return;
  }
  std::size_t contiguous_dim = shape.size();
  std::size_t contiguous_bytes = elem_size;
  while (contiguous_dim > 0) {
    std::size_t dim = contiguous_dim - 1;
    if (dim == shape.size() - 1) {
      if (dst_strides[dim] == 1 && src_strides[dim] == 1) {
        contiguous_bytes *= shape[dim];
        contiguous_dim = dim;
      } else {
        break;
      }
    } else {
      if (dst_strides[dim] ==
              dst_strides[dim + 1] *
                  static_cast<std::int64_t>(shape[dim + 1]) &&
          src_strides[dim] ==
              src_strides[dim + 1] *
                  static_cast<std::int64_t>(shape[dim + 1])) {
        contiguous_bytes *= shape[dim];
        contiguous_dim = dim;
      } else {
        break;
      }
    }
  }

  auto copy_dim = [&](auto& self, std::size_t dim, const std::byte* s,
                      CUdeviceptr d) -> void {
    if (dim >= contiguous_dim) {
      LC0EX_CUDA_CHECK(cuMemcpyHtoD(d, s, contiguous_bytes));
      return;
    }
    for (std::uint64_t i = 0; i < shape[dim]; ++i) {
      self(self, dim + 1, s + i * src_strides[dim] * elem_size,
           d + i * dst_strides[dim] * elem_size);
    }
  };
  copy_dim(copy_dim, 0, src, dst);
}

void CopyStridedDtoH(const std::vector<std::uint64_t>& shape,
                     const std::vector<std::int64_t>& dst_strides,
                     const std::vector<std::int64_t>& src_strides,
                     std::size_t elem_size, CUdeviceptr src,
                     std::byte* dst) {
  if (shape.empty()) {
    LC0EX_CUDA_CHECK(cuMemcpyDtoH(dst, src, elem_size));
    return;
  }
  std::size_t contiguous_dim = shape.size();
  std::size_t contiguous_bytes = elem_size;
  while (contiguous_dim > 0) {
    std::size_t dim = contiguous_dim - 1;
    if (dim == shape.size() - 1) {
      if (dst_strides[dim] == 1 && src_strides[dim] == 1) {
        contiguous_bytes *= shape[dim];
        contiguous_dim = dim;
      } else {
        break;
      }
    } else {
      if (dst_strides[dim] ==
              dst_strides[dim + 1] *
                  static_cast<std::int64_t>(shape[dim + 1]) &&
          src_strides[dim] ==
              src_strides[dim + 1] *
                  static_cast<std::int64_t>(shape[dim + 1])) {
        contiguous_bytes *= shape[dim];
        contiguous_dim = dim;
      } else {
        break;
      }
    }
  }

  auto copy_dim = [&](auto& self, std::size_t dim, CUdeviceptr s,
                      std::byte* d) -> void {
    if (dim >= contiguous_dim) {
      LC0EX_CUDA_CHECK(cuMemcpyDtoH(d, s, contiguous_bytes));
      return;
    }
    for (std::uint64_t i = 0; i < shape[dim]; ++i) {
      self(self, dim + 1, s + i * src_strides[dim] * elem_size,
           d + i * dst_strides[dim] * elem_size);
    }
  };
  copy_dim(copy_dim, 0, src, dst);
}

std::array<unsigned int, 3> LaunchDimensions(
    const std::vector<std::uint32_t>& dimensions) {
  std::array<unsigned int, 3> result = {1, 1, 1};
  for (std::size_t i = 0; i < dimensions.size(); ++i) {
    result[i] = dimensions[i];
  }
  return result;
}

std::pair<CUdeviceptr, CUdeviceptr> AllocateDeviceMemory(
    std::uint64_t size_bytes, std::uint64_t alignment_bytes) {
  const auto allocation_size = size_bytes + alignment_bytes - 1;

  CUdeviceptr base = 0;
  LC0EX_CUDA_CHECK(
      cuMemAlloc(&base, static_cast<std::size_t>(allocation_size)));

  const auto base_address = static_cast<std::uint64_t>(base);
  const auto aligned_address =
      (base_address + alignment_bytes - 1) & ~(alignment_bytes - 1);
  return {base, static_cast<CUdeviceptr>(aligned_address)};
}

struct Lc0exCudaAllocation {
  std::uint64_t size_bytes_ = 0;
  std::uint64_t alignment_bytes_ = 0;
  CUdeviceptr base_ = 0;
  CUdeviceptr address_ = 0;
};

struct Lc0exCudaBufferPlan {
  std::uint64_t offset_bytes_ = 0;
};

struct Lc0exCudaKernel {
  CUfunction function_ = nullptr;
  std::vector<pblczero::ParameterType> parameters_;
};

struct Lc0exCudaArgument {
  enum class Kind { kAllocation, kSymbol, kParameter, kNullPointer };

  Kind kind_ = Kind::kAllocation;
  std::size_t index_ = 0;
  pblczero::Node::Argument::AllocationLocation::AllocationKind allocation_kind_ =
      pblczero::Node::Argument::AllocationLocation::ALLOCATION_UNKNOWN;
  std::uint64_t offset_ = 0;
  CUdeviceptr symbol_ = 0;
};

struct Lc0exCudaNode {
  CUfunction function_ = nullptr;
  std::array<unsigned int, 3> grid_ = {1, 1, 1};
  std::array<unsigned int, 3> block_ = {1, 1, 1};
  unsigned int dynamic_shared_memory_bytes_ = 0;
  std::vector<Lc0exCudaArgument> arguments_;
  std::vector<std::size_t> dependencies_;
};

class Lc0exCudaExecutable;
class Lc0exCudaExecution;
class Lc0exCudaProgram;

struct Lc0exCudaExecutionSlot {
  Lc0exCudaAllocation allocation_;
  CUstream stream_ = nullptr;
  CUgraphExec graph_exec_ = nullptr;
  const Lc0exCudaProgram* captured_program_ = nullptr;
  bool in_use_ = false;
};

void DestroyExecutionSlot(Lc0exCudaExecutionSlot& slot) {
  if (slot.stream_) IgnoreCuda(cuStreamSynchronize(slot.stream_));
  if (slot.graph_exec_) IgnoreCuda(cuGraphExecDestroy(slot.graph_exec_));
  if (slot.allocation_.base_) IgnoreCuda(cuMemFree(slot.allocation_.base_));
  if (slot.stream_) IgnoreCuda(cuStreamDestroy(slot.stream_));
}

class Lc0exCudaProgram final : public Program {
 public:
  const ProgramInfo& GetInfo() const override { return info_; }

  std::span<const BufferInfo> GetBuffers() const override {
    return {buffer_infos_.data(), buffer_infos_.size()};
  }

  const BufferInfo* FindBuffer(std::string_view name) const override {
    const auto iter = buffer_indices_.find(std::string(name));
    return iter == buffer_indices_.end() ? nullptr
                                         : &buffer_infos_[iter->second];
  }

  std::span<const ParameterInfo> GetParameters() const override {
    return {parameters_.data(), parameters_.size()};
  }

  ProgramInfo info_;
  std::vector<ParameterInfo> parameters_;
  std::unordered_map<std::string, std::size_t> parameter_indices_;
  Lc0exCudaAllocation execution_allocation_;
  std::vector<BufferInfo> buffer_infos_;
  std::vector<Lc0exCudaBufferPlan> buffer_plans_;
  std::unordered_map<std::string, std::size_t> buffer_indices_;
  std::vector<Lc0exCudaNode> nodes_;
};

class Lc0exCudaBuffer final : public Buffer {
 public:
  Lc0exCudaBuffer(Lc0exCudaExecutable* executable, const BufferInfo* info,
                  CUdeviceptr address)
      : executable_(executable), info_(info), address_(address) {}

  const BufferInfo& GetInfo() const override { return *info_; }
  void CopyFromHost(std::span<const std::byte> source,
                    std::optional<std::size_t> size_bytes) override;
  void CopyToHost(std::span<std::byte> destination,
                  std::optional<std::size_t> size_bytes) const override;

  Lc0exCudaExecutable* executable_;
  const BufferInfo* info_;
  CUdeviceptr address_;
};

class Lc0exCudaParameter final : public Parameter {
 public:
  Lc0exCudaParameter(Lc0exCudaExecution* execution, std::size_t slot,
                     pblczero::ParameterType type)
      : execution_(execution), slot_(slot), type_(type) {}

  const ParameterInfo& GetInfo() const override;
  bool IsSet() const override { return is_set_; }
  void Set(std::uint32_t value) override;
  void Set(const Buffer& buffer) override;
  void Reset() override;

  void* ArgumentAddress() {
    switch (type_) {
      case pblczero::ParameterType_PARAMETER_TYPE_U32:
        return &u32_;
      case pblczero::ParameterType_PARAMETER_TYPE_POINTER:
        return &pointer_;
      case pblczero::ParameterType_PARAMETER_TYPE_NULL_POINTER:
        throw Exception("Null pointer is not a runtime parameter.");
      case pblczero::ParameterType_PARAMETER_TYPE_UNKNOWN:
        break;
    }
    throw Exception("Unknown parameter type.");
  }

  Lc0exCudaExecution* execution_;
  std::size_t slot_;
  pblczero::ParameterType type_;
  bool is_set_ = false;
  std::uint32_t u32_ = 0;
  CUdeviceptr pointer_ = 0;
};

class Lc0exCudaExecutable final : public Executable {
 public:
  explicit Lc0exCudaExecutable(CUdevice device) : device_(device) {}
  ~Lc0exCudaExecutable() override;

  const TargetInfo& GetTarget() const override { return target_; }
  std::string_view GetMetadata() const override { return metadata_; }
  std::span<const BufferInfo> GetBuffers() const override {
    return {buffer_infos_.data(), buffer_infos_.size()};
  }
  std::span<const ParameterInfo> GetParameters() const override {
    return {parameters_.data(), parameters_.size()};
  }
  std::span<const ProgramInfo> GetPrograms() const override {
    return {program_infos_.data(), program_infos_.size()};
  }

  const BufferInfo* FindBuffer(std::string_view name) const override {
    const auto iter = buffer_indices_.find(std::string(name));
    return iter == buffer_indices_.end() ? nullptr
                                         : &buffer_infos_[iter->second];
  }

  const ParameterInfo* FindParameter(std::string_view name) const override {
    const auto iter = parameter_indices_.find(std::string(name));
    return iter == parameter_indices_.end() ? nullptr
                                            : &parameters_[iter->second];
  }

  const Program* FindProgram(std::string_view name) const override {
    const auto iter = program_indices_.find(std::string(name));
    return iter == program_indices_.end() ? nullptr : &programs_[iter->second];
  }

  std::size_t GetPersistentAllocationSize() const override {
    return persistent_allocation_.size_bytes_;
  }

  void CopyPersistentFromHost(
      std::span<const std::byte> source,
      std::optional<std::size_t> size_bytes = std::nullopt) override {
    const std::size_t copy_size = size_bytes.value_or(source.size());
    if (copy_size == 0 || !persistent_allocation_.address_) return;

    SetCurrent();
    LC0EX_CUDA_CHECK(cuMemcpyHtoD(persistent_allocation_.address_,
                                  source.data(), copy_size));
  }

  Buffer& GetBuffer(const BufferInfo& info) override;

  std::unique_ptr<Execution> CreateExecution(const Program& program) override;

  Lc0exCudaExecutionSlot* AcquireExecutionSlot();
  void ReleaseExecutionSlot(Lc0exCudaExecutionSlot* slot);

  void Initialize() {
    LC0EX_CUDA_CHECK(cuDevicePrimaryCtxRetain(&context_, device_));
    context_retained_ = true;
    LC0EX_CUDA_CHECK(cuCtxSetCurrent(context_));
  }

  void SetCurrent() const { LC0EX_CUDA_CHECK(cuCtxSetCurrent(context_)); }

  CUdevice device_ = 0;
  CUcontext context_ = nullptr;
  bool context_retained_ = false;

  TargetInfo target_;
  std::string metadata_;

  std::vector<CUmodule> modules_;

  Lc0exCudaAllocation persistent_allocation_;

  std::vector<ParameterInfo> parameters_;
  std::unordered_map<std::string, std::size_t> parameter_indices_;

  std::vector<BufferInfo> buffer_infos_;
  std::vector<Lc0exCudaBufferPlan> buffer_plans_;
  std::vector<std::unique_ptr<Lc0exCudaBuffer>> persistent_buffers_;
  std::unordered_map<std::string, std::size_t> buffer_indices_;

  std::vector<Lc0exCudaKernel> kernels_;

  std::vector<Lc0exCudaProgram> programs_;
  std::vector<ProgramInfo> program_infos_;
  std::unordered_map<std::string, std::size_t> program_indices_;

  // This is the maximum execution allocation required by any program. Slots
  // use it so that sequential executions can reuse device memory.
  Lc0exCudaAllocation execution_pool_allocation_{0, 1, 0, 0};
  std::mutex execution_slots_mutex_;
  std::vector<std::unique_ptr<Lc0exCudaExecutionSlot>> execution_slots_;
};

class Lc0exCudaExecution final : public Execution {
 public:
  Lc0exCudaExecution(Lc0exCudaExecutable* executable,
                     const Lc0exCudaProgram* program)
      : executable_(executable), program_(program) {}
  ~Lc0exCudaExecution() override;

  Buffer& GetBuffer(const BufferInfo& info) override {
    return *buffers_[program_->buffer_indices_.find(info.name)->second];
  }

  Parameter& GetParameter(std::string_view name) override {
    return parameters_[program_->parameter_indices_.find(std::string(name))
                           ->second];
  }

  void ResetParameters() override {
    for (auto& parameter : parameters_) parameter.Reset();
  }

  void Initialize() {
    executable_->SetCurrent();
    slot_ = executable_->AcquireExecutionSlot();

    buffers_.resize(program_->buffer_plans_.size());
    for (std::size_t i = 0; i < program_->buffer_plans_.size(); ++i) {
      const auto& plan = program_->buffer_plans_[i];
      auto address = slot_->allocation_.address_;
      address += plan.offset_bytes_;
      buffers_[i] = std::make_unique<Lc0exCudaBuffer>(
          executable_, &program_->buffer_infos_[i], address);
    }

    parameters_.reserve(program_->parameters_.size());
    for (std::size_t i = 0; i < program_->parameters_.size(); ++i) {
      parameters_.emplace_back(this, i, program_->parameters_[i].type);
    }

    launch_arguments_.resize(program_->nodes_.size());
    allocation_argument_values_.resize(program_->nodes_.size());
    for (std::size_t node_index = 0; node_index < program_->nodes_.size();
         ++node_index) {
      const auto& node = program_->nodes_[node_index];
      auto& arguments = launch_arguments_[node_index];
      auto& allocation_values = allocation_argument_values_[node_index];
      arguments.resize(node.arguments_.size());
      allocation_values.resize(node.arguments_.size());
      for (std::size_t argument_index = 0;
           argument_index < node.arguments_.size(); ++argument_index) {
        const auto& argument = node.arguments_[argument_index];
        auto& value = allocation_values[argument_index];
        switch (argument.kind_) {
          case Lc0exCudaArgument::Kind::kNullPointer:
            value = 0;
            arguments[argument_index] = &value;
            break;
          case Lc0exCudaArgument::Kind::kParameter:
            arguments[argument_index] =
                parameters_[argument.index_].ArgumentAddress();
            break;
          case Lc0exCudaArgument::Kind::kSymbol:
            value = argument.symbol_;
            arguments[argument_index] = &value;
            break;
          case Lc0exCudaArgument::Kind::kAllocation:
            value = argument.allocation_kind_ ==
                            pblczero::Node::Argument::AllocationLocation::
                                ALLOCATION_PERSISTENT
                        ? executable_->persistent_allocation_.address_
                        : slot_->allocation_.address_;
            value += argument.offset_;
            arguments[argument_index] = &value;
            break;
        }
      }
    }
  }

  void Run() override {
    executable_->SetCurrent();
    in_flight_ = true;
    if (slot_->graph_exec_ != nullptr && slot_->captured_program_ == program_) {
      LC0EX_CUDA_CHECK(cuGraphLaunch(slot_->graph_exec_, slot_->stream_));
      return;
    }
    if (slot_->graph_exec_ != nullptr) {
      cuGraphExecDestroy(slot_->graph_exec_);
      slot_->graph_exec_ = nullptr;
    }
    CUgraph graph = nullptr;
    LC0EX_CUDA_CHECK(cuGraphCreate(&graph, 0));

    std::vector<CUgraphNode> graph_nodes(program_->nodes_.size(), nullptr);
    for (std::size_t i = 0; i < program_->nodes_.size(); ++i) {
      const auto& node = program_->nodes_[i];

      CUDA_KERNEL_NODE_PARAMS params{};
      params.func = node.function_;
      params.gridDimX = node.grid_[0];
      params.gridDimY = node.grid_[1];
      params.gridDimZ = node.grid_[2];
      params.blockDimX = node.block_[0];
      params.blockDimY = node.block_[1];
      params.blockDimZ = node.block_[2];
      params.sharedMemBytes = node.dynamic_shared_memory_bytes_;
      params.kernelParams = const_cast<void**>(launch_arguments_[i].data());
      params.extra = nullptr;

      std::vector<CUgraphNode> deps;
      deps.reserve(node.dependencies_.size());
      for (const auto dep_idx : node.dependencies_) {
        deps.push_back(graph_nodes[dep_idx]);
      }

      LC0EX_CUDA_CHECK(cuGraphAddKernelNode(
          &graph_nodes[i], graph, deps.data(), deps.size(), &params));
    }

    LC0EX_CUDA_CHECK(cuGraphInstantiate(&slot_->graph_exec_, graph, 0));
    LC0EX_CUDA_CHECK(cuGraphDestroy(graph));
    slot_->captured_program_ = program_;
    LC0EX_CUDA_CHECK(cuGraphLaunch(slot_->graph_exec_, slot_->stream_));
  }

  void Synchronize() override {
    if (!in_flight_) return;
    executable_->SetCurrent();
    LC0EX_CUDA_CHECK(cuStreamSynchronize(slot_->stream_));
    in_flight_ = false;
  }

  Lc0exCudaExecutable* executable_;
  const Lc0exCudaProgram* program_;
  Lc0exCudaExecutionSlot* slot_ = nullptr;
  bool in_flight_ = false;
  std::vector<std::unique_ptr<Lc0exCudaBuffer>> buffers_;
  std::vector<Lc0exCudaParameter> parameters_;
  std::vector<std::vector<void*>> launch_arguments_;
  std::vector<std::vector<CUdeviceptr>> allocation_argument_values_;
};

const ParameterInfo& Lc0exCudaParameter::GetInfo() const {
  return execution_->program_->parameters_[slot_];
}

void Lc0exCudaParameter::Set(std::uint32_t value) {
  u32_ = value;
  is_set_ = true;
}

void Lc0exCudaParameter::Set(const Buffer& buffer) {
  pointer_ = static_cast<const Lc0exCudaBuffer&>(buffer).address_;
  is_set_ = true;
}

void Lc0exCudaParameter::Reset() {
  is_set_ = false;
  u32_ = 0;
  pointer_ = 0;
}

void Lc0exCudaBuffer::CopyFromHost(std::span<const std::byte> source,
                                   std::optional<std::size_t> size_bytes) {
  const std::size_t copy_size = size_bytes.value_or(source.size());
  if (copy_size == 0) return;

  executable_->SetCurrent();
  const auto default_strides = DefaultStrides(info_->shape);
  if (info_->strides == default_strides) {
    LC0EX_CUDA_CHECK(cuMemcpyHtoD(address_, source.data(), copy_size));
    return;
  }

  const auto elem_size = ElementSize(info_->data_type);
  CopyStridedHtoD(info_->shape, info_->strides, default_strides, elem_size,
                  source.data(), address_);
}

void Lc0exCudaBuffer::CopyToHost(std::span<std::byte> destination,
                                 std::optional<std::size_t> size_bytes) const {
  const std::size_t copy_size = size_bytes.value_or(destination.size());
  if (copy_size == 0) return;

  executable_->SetCurrent();
  const auto default_strides = DefaultStrides(info_->shape);
  if (info_->strides == default_strides) {
    LC0EX_CUDA_CHECK(cuMemcpyDtoH(destination.data(), address_, copy_size));
    return;
  }

  const auto elem_size = ElementSize(info_->data_type);
  CopyStridedDtoH(info_->shape, default_strides, info_->strides, elem_size,
                  address_, destination.data());
}

Lc0exCudaExecutable::~Lc0exCudaExecutable() {
  if (!context_retained_) return;
  if (cuCtxSetCurrent(context_) == CUDA_SUCCESS) {
    absl::c_for_each(execution_slots_,
                     [](const auto& slot) { DestroyExecutionSlot(*slot); });
    if (persistent_allocation_.base_) {
      IgnoreCuda(cuMemFree(persistent_allocation_.base_));
    }
    for (auto& module : modules_) {
      if (module) IgnoreCuda(cuModuleUnload(module));
    }
  }
  IgnoreCuda(cuDevicePrimaryCtxRelease(device_));
}

Lc0exCudaExecution::~Lc0exCudaExecution() {
  if (!executable_ || !slot_ || !executable_->context_retained_) return;
  if (cuCtxSetCurrent(executable_->context_) == CUDA_SUCCESS) {
    IgnoreCuda(cuStreamSynchronize(slot_->stream_));
    executable_->ReleaseExecutionSlot(slot_);
    slot_ = nullptr;
  }
}

Buffer& Lc0exCudaExecutable::GetBuffer(const BufferInfo& info) {
  return *persistent_buffers_[buffer_indices_.find(info.name)->second];
}

Lc0exCudaExecutionSlot* Lc0exCudaExecutable::AcquireExecutionSlot() {
  std::lock_guard<std::mutex> lock(execution_slots_mutex_);
  const auto free_slot = absl::c_find_if(
      execution_slots_, [](const auto& slot) { return !slot->in_use_; });
  if (free_slot != execution_slots_.end()) {
    (*free_slot)->in_use_ = true;
    return free_slot->get();
  }

  auto slot = std::make_unique<Lc0exCudaExecutionSlot>();
  slot->allocation_.size_bytes_ = execution_pool_allocation_.size_bytes_;
  slot->allocation_.alignment_bytes_ =
      execution_pool_allocation_.alignment_bytes_;

  SetCurrent();
  LC0EX_CUDA_CHECK(cuStreamCreate(&slot->stream_, CU_STREAM_DEFAULT));
  if (slot->allocation_.size_bytes_ != 0) {
    const auto memory = AllocateDeviceMemory(
        slot->allocation_.size_bytes_, slot->allocation_.alignment_bytes_);
    slot->allocation_.base_ = memory.first;
    slot->allocation_.address_ = memory.second;
  }
  auto* result = slot.get();
  execution_slots_.push_back(std::move(slot));
  result->in_use_ = true;
  return result;
}

void Lc0exCudaExecutable::ReleaseExecutionSlot(Lc0exCudaExecutionSlot* slot) {
  std::lock_guard<std::mutex> lock(execution_slots_mutex_);
  slot->in_use_ = false;
}

std::unique_ptr<Execution> Lc0exCudaExecutable::CreateExecution(
    const Program& program) {
  const auto* cuda_program = static_cast<const Lc0exCudaProgram*>(&program);
  auto execution = std::make_unique<Lc0exCudaExecution>(this, cuda_program);
  execution->Initialize();
  return execution;
}

void BuildModules(Lc0exCudaExecutable& executable,
                  const pblczero::NeuralExecutable& source) {
  executable.modules_.reserve(source.binaries_size());
  for (const auto& binary : source.binaries()) {
    CUmodule module = nullptr;
    LC0EX_CUDA_CHECK(cuModuleLoadData(&module, binary.data().data()));
    executable.modules_.push_back(module);
  }
}

void BuildAllocation(Lc0exCudaAllocation& destination,
                     const pblczero::Allocation& source) {
  destination.size_bytes_ = source.size_bytes();
  destination.alignment_bytes_ = source.alignment_bytes();
}

void BuildPersistentAllocation(Lc0exCudaExecutable& executable,
                               const pblczero::NeuralExecutable& source) {
  if (source.has_persistent_allocation()) {
    BuildAllocation(executable.persistent_allocation_,
                    source.persistent_allocation());
    const auto memory = AllocateDeviceMemory(
        executable.persistent_allocation_.size_bytes_,
        executable.persistent_allocation_.alignment_bytes_);
    executable.persistent_allocation_.base_ = memory.first;
    executable.persistent_allocation_.address_ = memory.second;
  }
}

void BuildParameters(Lc0exCudaExecutable& executable,
                     const pblczero::NeuralExecutable& source) {
  executable.parameters_.reserve(source.parameters_size());
  executable.parameter_indices_.reserve(source.parameters_size());
  for (const auto& parameter : source.parameters()) {
    if (parameter.type() ==
        pblczero::ParameterType_PARAMETER_TYPE_NULL_POINTER) {
      throw Exception("Null pointer is not a runtime parameter.");
    }
    const auto name = std::string(parameter.name());
    executable.parameters_.push_back({name, parameter.type()});
    executable.parameter_indices_.emplace(executable.parameters_.back().name,
                                          executable.parameters_.size() - 1);
  }
}

void BuildPersistentBuffers(Lc0exCudaExecutable& executable,
                            const pblczero::NeuralExecutable& source) {
  executable.buffer_infos_.reserve(source.buffers_size());
  executable.buffer_plans_.reserve(source.buffers_size());
  executable.buffer_indices_.reserve(source.buffers_size());
  for (const auto& buffer : source.buffers()) {
    auto info = MakeBufferInfo(buffer);
    Lc0exCudaBufferPlan plan{buffer.offset()};

    executable.buffer_infos_.push_back(std::move(info));
    executable.buffer_plans_.push_back(std::move(plan));
    executable.buffer_indices_.emplace(executable.buffer_infos_.back().name,
                                       executable.buffer_infos_.size() - 1);
  }

  executable.persistent_buffers_.resize(executable.buffer_plans_.size());
  for (std::size_t i = 0; i < executable.buffer_plans_.size(); ++i) {
    const auto& plan = executable.buffer_plans_[i];
    auto address = executable.persistent_allocation_.address_;
    address += plan.offset_bytes_;
    executable.persistent_buffers_[i] = std::make_unique<Lc0exCudaBuffer>(
        &executable, &executable.buffer_infos_[i], address);
  }
}

void BuildProgramBuffers(Lc0exCudaProgram& program,
                         const pblczero::Program& source) {
  program.buffer_infos_.reserve(source.buffers_size());
  program.buffer_plans_.reserve(source.buffers_size());
  program.buffer_indices_.reserve(source.buffers_size());
  for (const auto& buffer : source.buffers()) {
    auto info = MakeBufferInfo(buffer);

    program.buffer_infos_.push_back(std::move(info));
    program.buffer_plans_.push_back({buffer.offset()});
    program.buffer_indices_.emplace(program.buffer_infos_.back().name,
                                    program.buffer_infos_.size() - 1);
  }
}

void BuildKernels(Lc0exCudaExecutable& executable,
                  const pblczero::NeuralExecutable& source) {
  executable.kernels_.reserve(source.kernels_size());
  for (const auto& kernel : source.kernels()) {
    CUfunction function = nullptr;
    LC0EX_CUDA_CHECK(
        cuModuleGetFunction(&function, executable.modules_[kernel.binary_idx()],
                            std::string(kernel.function()).c_str()));

    Lc0exCudaKernel plan;
    plan.function_ = function;
    plan.parameters_.assign(kernel.parameters().begin(),
                            kernel.parameters().end());

    executable.kernels_.push_back(std::move(plan));
  }
}

Lc0exCudaArgument BuildParameterArgument(
    Lc0exCudaExecutable& executable, Lc0exCudaProgram& program,
    const pblczero::Node::Argument& source) {
  const auto& global_parameter = executable.parameters_[
      executable.parameter_indices_.at(std::string(source.parameter_name()))];
  const auto [local_iter, inserted] = program.parameter_indices_.try_emplace(
      global_parameter.name, program.parameters_.size());
  if (inserted) program.parameters_.push_back(global_parameter);

  Lc0exCudaArgument argument;
  argument.kind_ = Lc0exCudaArgument::Kind::kParameter;
  argument.index_ = local_iter->second;
  return argument;
}

Lc0exCudaArgument BuildAllocationArgument(
    const pblczero::Node::Argument& source) {
  const auto& location = source.allocation();
  Lc0exCudaArgument argument;
  argument.kind_ = Lc0exCudaArgument::Kind::kAllocation;
  argument.allocation_kind_ = location.kind();
  argument.offset_ = location.offset();
  return argument;
}

Lc0exCudaArgument BuildSymbolArgument(
    Lc0exCudaExecutable& executable, const pblczero::Node::Argument& source) {
  const auto& symbol = source.symbol();
  Lc0exCudaArgument argument;
  argument.kind_ = Lc0exCudaArgument::Kind::kSymbol;
  std::size_t symbol_size = 0;
  const std::string symbol_name(symbol.symbol_name());
  LC0EX_CUDA_CHECK(cuModuleGetGlobal(
      &argument.symbol_, &symbol_size,
      executable.modules_[symbol.binary_idx()], symbol_name.c_str()));
  return argument;
}

Lc0exCudaArgument BuildArgument(
    Lc0exCudaExecutable& executable, Lc0exCudaProgram& program,
    const pblczero::Node::Argument& source) {
  if (source.has_allocation()) {
    return BuildAllocationArgument(source);
  }
  if (source.has_symbol()) {
    return BuildSymbolArgument(executable, source);
  }
  return BuildParameterArgument(executable, program, source);
}

void BuildArguments(Lc0exCudaExecutable& executable,
                    Lc0exCudaProgram& program,
                    const pblczero::Node& source,
                    const Lc0exCudaKernel& kernel,
                    Lc0exCudaNode* destination) {
  auto source_argument = source.arguments().begin();
  destination->arguments_.reserve(kernel.parameters_.size());
  for (const auto parameter_type : kernel.parameters_) {
    switch (parameter_type) {
      case pblczero::ParameterType_PARAMETER_TYPE_NULL_POINTER: {
        Lc0exCudaArgument argument;
        argument.kind_ = Lc0exCudaArgument::Kind::kNullPointer;
        destination->arguments_.push_back(argument);
        break;
      }
      case pblczero::ParameterType_PARAMETER_TYPE_U32:
      case pblczero::ParameterType_PARAMETER_TYPE_POINTER:
        destination->arguments_.push_back(
            BuildArgument(executable, program, *source_argument++));
        break;
      case pblczero::ParameterType_PARAMETER_TYPE_UNKNOWN:
        throw Exception("The lc0ex kernel has an unknown parameter type.");
    }
  }
}

void BuildProgram(Lc0exCudaExecutable& executable,
                  const pblczero::Program& source,
                  Lc0exCudaProgram* destination) {
  destination->info_.name = std::string(source.name());
  destination->info_.metadata = source.metadata();
  if (source.has_execution_allocation()) {
    BuildAllocation(destination->execution_allocation_,
                    source.execution_allocation());
  }
  BuildProgramBuffers(*destination, source);

  destination->nodes_.reserve(source.nodes_size());
  for (std::size_t node_index = 0; node_index < source.nodes_size();
       ++node_index) {
    const auto& node = source.nodes(node_index);
    const auto& kernel = executable.kernels_[node.kernel_idx()];

    Lc0exCudaNode plan;
    plan.function_ = kernel.function_;
    plan.grid_ = LaunchDimensions(node.grid());
    plan.block_ = LaunchDimensions(node.block());
    plan.dynamic_shared_memory_bytes_ = node.dynamic_shared_memory_bytes();
    if (plan.dynamic_shared_memory_bytes_ != 0) {
      LC0EX_CUDA_CHECK(cuFuncSetAttribute(
          plan.function_, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
          plan.dynamic_shared_memory_bytes_));
    }
    BuildArguments(executable, *destination, node, kernel, &plan);
    plan.dependencies_.assign(node.dependencies().begin(),
                              node.dependencies().end());

    destination->nodes_.push_back(std::move(plan));
  }
}

void BuildPrograms(Lc0exCudaExecutable& executable,
                   const pblczero::NeuralExecutable& source) {
  executable.programs_.reserve(source.programs_size());
  executable.program_infos_.reserve(source.programs_size());
  executable.program_indices_.reserve(source.programs_size());
  for (const auto& program : source.programs()) {
    const auto name = std::string(program.name());

    Lc0exCudaProgram plan;
    BuildProgram(executable, program, &plan);
    executable.execution_pool_allocation_.size_bytes_ =
        std::max(executable.execution_pool_allocation_.size_bytes_,
                 plan.execution_allocation_.size_bytes_);
    executable.execution_pool_allocation_.alignment_bytes_ =
        std::max(executable.execution_pool_allocation_.alignment_bytes_,
                 plan.execution_allocation_.alignment_bytes_);
    executable.program_infos_.push_back(plan.info_);
    executable.programs_.push_back(std::move(plan));
    executable.program_indices_.emplace(name, executable.programs_.size() - 1);
  }
}

class Lc0exCudaRuntime final : public Runtime {
 public:
  explicit Lc0exCudaRuntime(int device_ordinal) {
    LC0EX_CUDA_CHECK(cuInit(0));
    LC0EX_CUDA_CHECK(cuDeviceGet(&device_, device_ordinal));
  }

  std::unique_ptr<Executable> Load(
      const pblczero::NeuralExecutable& source) override {
    if (source.magic() != kMagic) throw Exception("Invalid lc0ex magic.");
    if (source.format() != kFormat) {
      throw Exception("Unsupported lc0ex format generation.");
    }

    auto executable = std::make_unique<Lc0exCudaExecutable>(device_);
    executable->Initialize();
    executable->target_.vendor = pblczero::Target::VENDOR_NVIDIA;
    executable->target_.architecture =
        std::string(source.target().architecture());
    executable->metadata_ = source.metadata();

    BuildModules(*executable, source);
    BuildPersistentAllocation(*executable, source);
    BuildParameters(*executable, source);
    BuildPersistentBuffers(*executable, source);
    BuildKernels(*executable, source);
    BuildPrograms(*executable, source);
    return executable;
  }

 private:
  CUdevice device_ = 0;
};

}  // namespace

std::unique_ptr<Runtime> CreateLc0exCudaRuntime(int device_ordinal) {
  return std::make_unique<Lc0exCudaRuntime>(device_ordinal);
}

}  // namespace lc0ex
}  // namespace lczero
