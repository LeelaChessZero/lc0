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

#include <fstream>
#include <string>
#include <vector>

#include "neural/backends/stablehlo/stablehlo_backend.h"
#include "neural/backends/stablehlo/semantic/semantic_ir.h"
#include "utils/exception.h"

namespace lczero {
namespace {

using stablehlo::semantic::ConvParams;
using stablehlo::semantic::ElementType;
using stablehlo::semantic::OpKind;
using stablehlo::semantic::ReduceParams;
using stablehlo::semantic::SemanticFunction;
using stablehlo::semantic::SemanticModule;
using stablehlo::semantic::SemanticOp;
using stablehlo::semantic::TensorType;

bool ContainsAscii(const std::vector<uint8_t>& bytes, const std::string& needle) {
  if (needle.empty() || bytes.size() < needle.size()) return false;
  for (size_t i = 0; i + needle.size() <= bytes.size(); ++i) {
    bool match = true;
    for (size_t j = 0; j < needle.size(); ++j) {
      if (bytes[i + j] != static_cast<uint8_t>(needle[j])) {
        match = false;
        break;
      }
    }
    if (match) return true;
  }
  return false;
}

SemanticModule BuildSmokeModule() {
  SemanticModule module;
  SemanticFunction main_fn;
  main_fn.name = "main";
  // Keep convolution asymmetric (I != O) to guard against kernel input/output
  // feature-dimension mapping regressions.
  main_fn.param_types = {
      TensorType{ElementType::kF32, {1, 8, 8, 112}},
      TensorType{ElementType::kF32, {3, 3, 112, 256}},
      TensorType{ElementType::kF32, {1, 8, 8, 256}},
      TensorType{ElementType::kF32, {}},
  };

  ConvParams conv;
  conv.input_batch_dim = 0;
  conv.input_feature_dim = 3;
  conv.input_spatial_dims = {1, 2};
  conv.kernel_input_feature_dim = 2;
  conv.kernel_output_feature_dim = 3;
  conv.kernel_spatial_dims = {0, 1};
  conv.output_batch_dim = 0;
  conv.output_feature_dim = 3;
  conv.output_spatial_dims = {1, 2};
  conv.window_strides = {1, 1};
  conv.padding = {{1, 1}, {1, 1}};
  conv.lhs_dilation = {1, 1};
  conv.rhs_dilation = {1, 1};

  SemanticOp conv_op;
  conv_op.kind = OpKind::kConvolution;
  conv_op.operands = {0, 1};
  conv_op.result_types = {TensorType{ElementType::kF32, {1, 8, 8, 256}}};
  conv_op.attrs = conv;
  main_fn.ops.push_back(std::move(conv_op));

  SemanticOp add_op;
  add_op.kind = OpKind::kAdd;
  add_op.operands = {4, 2};
  add_op.result_types = {TensorType{ElementType::kF32, {1, 8, 8, 256}}};
  main_fn.ops.push_back(std::move(add_op));

  ReduceParams reduce;
  reduce.dimensions = {1, 2};
  reduce.reduce_op = ReduceParams::ReduceOp::kAdd;

  SemanticOp reduce_op;
  reduce_op.kind = OpKind::kReduce;
  reduce_op.operands = {5, 3};
  reduce_op.result_types = {TensorType{ElementType::kF32, {1, 256}}};
  reduce_op.attrs = reduce;
  main_fn.ops.push_back(std::move(reduce_op));

  SemanticOp ret_op;
  ret_op.kind = OpKind::kReturn;
  ret_op.operands = {6};
  main_fn.ops.push_back(std::move(ret_op));

  module.functions.push_back(std::move(main_fn));
  return module;
}

}  // namespace
}  // namespace lczero

int main(int argc, char** argv) {
  const auto semantic_module = lczero::BuildSmokeModule();
  auto bytes = lczero::stablehlo::SemanticModuleToMlirbc(semantic_module);

  if (bytes.empty()) {
    return 1;
  }
  if (!lczero::ContainsAscii(bytes, "convolution_v1") ||
      !lczero::ContainsAscii(bytes, "add_v1") ||
      !lczero::ContainsAscii(bytes, "reduce_v1")) {
    return 2;
  }

  // Negative test: lowering an empty module must throw.
  try {
    lczero::stablehlo::semantic::SemanticModule empty_module;
    (void)lczero::stablehlo::SemanticModuleToMlirbc(empty_module);
    return 10;  // should not reach here
  } catch (const lczero::Exception&) {
    // expected
  }

  if (argc > 1) {
    std::ofstream out(argv[1], std::ios::binary);
    if (!out) {
      return 3;
    }
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    if (!out.good()) {
      return 4;
    }
  }

  return 0;
}
