/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2021 The LCZero Authors

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

#include <initializer_list>

#include "neural/onnx/builder.h"
#include "neural/onnx/onnx.pb.h"
#include "proto/net.pb.h"
#include "utils/weights_adapter.h"

namespace lczero {

class FloatOnnxWeightsAdapter : public OnnxConst {
 public:
  FloatOnnxWeightsAdapter(const std::vector<float>& weights,
                          std::initializer_list<int> dims,
                          std::initializer_list<int> order = {});

 private:
  pblczero::TensorProto::DataType GetDataType() const override;
  std::vector<int> GetDimensions() const override;
  std::string GetRawData() const override;

  const std::vector<float>& weights_;
  std::vector<int> dims_;
  std::vector<int> order_;
};

template <typename T>
class GenericOnnxConst : public OnnxConst {
 public:
  GenericOnnxConst(const std::vector<T> data, std::initializer_list<int> dims)
      : data_(data), dims_(dims) {}

 private:
  std::vector<int> GetDimensions() const override { return dims_; }
  std::string GetRawData() const override {
    return {reinterpret_cast<const char*>(data_.data()),
            reinterpret_cast<const char*>(data_.data() + data_.size())};
  }

  std::vector<T> data_;
  std::vector<int> dims_;
};

class Int32OnnxConst : public GenericOnnxConst<int32_t> {
 public:
  using GenericOnnxConst<int32_t>::GenericOnnxConst;

 private:
  pblczero::TensorProto::DataType GetDataType() const override {
    return pblczero::TensorProto::INT32;
  }
};

class Int64OnnxConst : public GenericOnnxConst<int64_t> {
 public:
  using GenericOnnxConst<int64_t>::GenericOnnxConst;

 private:
  pblczero::TensorProto::DataType GetDataType() const override {
    return pblczero::TensorProto::INT64;
  }
};

}  // namespace lczero