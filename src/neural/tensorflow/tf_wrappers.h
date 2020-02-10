/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2020 The LCZero Authors

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

#include <tensorflow/c/c_api.h>

#include <algorithm>
#include <string>
#include <vector>

#include "utils/exception.h"

namespace lczero {

class TFStatus {
 public:
  TFStatus() : status_(TF_NewStatus()) {}
  ~TFStatus() { TF_DeleteStatus(status_); }
  void Check() const {
    if (TF_GetCode(status_) != TF_OK) {
      throw Exception("Tensorflow error " +
                      std::to_string(TF_GetCode(status_)) + ": " +
                      TF_Message(status_));
    }
  }
  TF_Status* get_raw() const { return status_; }

 private:
  TF_Status* const status_;
};

class TFBuffer {
 public:
  TFBuffer(const std::string& str)
      : buffer_(TF_NewBufferFromString(str.data(), str.size())) {}
  ~TFBuffer() { TF_DeleteBuffer(buffer_); }
  TF_Buffer* get_raw() const { return buffer_; }

 private:
  TF_Buffer* const buffer_;
};

class TFGraph {
 public:
  TFGraph() : graph_(TF_NewGraph()) {}
  ~TFGraph() { TF_DeleteGraph(graph_); }

  class ImportOptions {
   public:
    ImportOptions() : options_(TF_NewImportGraphDefOptions()) {}
    ~ImportOptions() { TF_DeleteImportGraphDefOptions(options_); }

    TF_ImportGraphDefOptions* get_raw() const { return options_; }

   private:
    TF_ImportGraphDefOptions* const options_;
  };
  void ImportGraphDef(const TFBuffer& buffer, const ImportOptions& opts,
                      TFStatus* status) {
    TF_GraphImportGraphDef(graph_, buffer.get_raw(), opts.get_raw(),
                           status->get_raw());
  }

  void ImportGraphDef(const TFBuffer& buffer,
                      const ImportOptions& opts = ImportOptions()) {
    TFStatus status;
    ImportGraphDef(buffer, opts, &status);
    status.Check();
  }
  TF_Operation* GetOperationByName(const std::string& op_name) {
    auto* op = TF_GraphOperationByName(graph_, op_name.c_str());
    if (op == nullptr) {
      throw Exception("Unable to find operation " + op_name);
    }
    return op;
  }

  TF_Graph* get_raw() const { return graph_; }

 private:
  TF_Graph* const graph_;
};

class TFTensor {
 public:
  TFTensor() = default;
  static TFTensor Adopt(TF_Tensor* tensor) { return TFTensor(tensor); }
  TFTensor(TF_DataType datatype, const std::vector<int64_t> dims)
      : tensor_(TF_AllocateTensor(datatype, dims.data(), dims.size(), [&]() {
          size_t size = TF_DataTypeSize(datatype);
          for (const auto x : dims) size *= x;
          return size;
        }())) {}
  void* GetBuffer() const { return TF_TensorData(tensor_); }

  ~TFTensor() { TF_DeleteTensor(tensor_); }
  TFTensor(TFTensor&& other) : tensor_(other.tensor_) {
    other.tensor_ = nullptr;
  }
  void operator=(TFTensor&& other) {
    tensor_ = other.tensor_;
    other.tensor_ = nullptr;
  }

  std::vector<int64_t> GetDimensions() {
    std::vector<int64_t> res(TF_NumDims(tensor_));
    for (size_t i = 0; i < res.size(); ++i) {
      res[i] = TF_Dim(tensor_, i);
    }
    return res;
  }

  size_t GetByteSize() { return TF_TensorByteSize(tensor_); }

  TF_Tensor* get_raw() const { return tensor_; }

  std::string DebugString() {
    std::string res;
    for (const auto x : GetDimensions()) {
      if (!res.empty()) res += ", ";
      res += std::to_string(x);
    }
    return "[" + res + "] type:" + std::to_string(TF_TensorType(tensor_));
  }

  std::string Dump() {
    if (TF_TensorType(tensor_) != TF_FLOAT) return "(not float)";
    auto* data = static_cast<float*>(GetBuffer());
    std::string res;
    for (size_t i = 0; i < std::min(10000000UL, GetByteSize() / sizeof(float));
         ++i) {
      res += " ";
      res += std::to_string(data[i]);
    }
    return res;
  }

 private:
  TFTensor(TF_Tensor* raw) : tensor_(raw) {}

  TF_Tensor* tensor_ = nullptr;
};

class TFSession {
 public:
  class Options {
   public:
    Options() : options_(TF_NewSessionOptions()) {}
    ~Options() { TF_DeleteSessionOptions(options_); }

    TF_SessionOptions* get_raw() const { return options_; }

   private:
    TF_SessionOptions* const options_;
  };

  TFSession(const TFGraph& graph, const Options& options = Options()) {
    TFStatus status;
    session_ =
        TF_NewSession(graph.get_raw(), options.get_raw(), status.get_raw());
    status.Check();
  }
  ~TFSession() {
    TFStatus status;
    TF_CloseSession(session_, status.get_raw());
    TF_DeleteSession(session_, status.get_raw());
    status.Check();
  }

  std::vector<TFTensor> Run(const std::vector<TF_Output>& inputs,
                            const std::vector<TFTensor*>& input_tensors,
                            const std::vector<TF_Output>& outputs,
                            TFStatus* status) {
    std::vector<TF_Tensor*> raw_input_tensors;
    for (const auto& tensor : input_tensors) {
      raw_input_tensors.push_back(tensor->get_raw());
    }
    std::vector<TF_Tensor*> raw_output_tensors(outputs.size());
    TF_SessionRun(session_, nullptr, inputs.data(), raw_input_tensors.data(),
                  inputs.size(), outputs.data(), raw_output_tensors.data(),
                  outputs.size(), nullptr, 0, nullptr, status->get_raw());

    std::vector<TFTensor> output;
    for (const auto& x : raw_output_tensors) {
      output.emplace_back(TFTensor::Adopt(x));
    }
    return output;
  }

  std::vector<TFTensor> Run(const std::vector<TF_Output>& inputs,
                            const std::vector<TFTensor*>& input_tensors,
                            const std::vector<TF_Output>& outputs) {
    TFStatus status;
    auto res = Run(inputs, input_tensors, outputs, &status);
    status.Check();
    return res;
  }

 private:
  TF_Session* session_;
};

}  // namespace lczero