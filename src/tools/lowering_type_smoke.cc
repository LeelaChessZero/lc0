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

#include <cstdlib>
#include <string>
#include <vector>

#include "neural/backends/stablehlo/stablehlo_backend.h"
#include "neural/backends/stablehlo/lowering/lowering.h"
#include "neural/backends/stablehlo/lowering/op_specs.h"
#include "stablehlo/types.h"

namespace lczero {
namespace {

bool CheckRankedTensor(const stablehlo::TypePtr& type,
                       stablehlo::ElementType expected_element,
                       const std::vector<int64_t>& expected_shape) {
  if (type->kind() != stablehlo::TypeKind::kRankedTensor) return false;
  const auto& ranked =
      static_cast<const stablehlo::RankedTensorType&>(*type);
  if (ranked.shape() != expected_shape) return false;
  if (ranked.elementType()->kind() != stablehlo::TypeKind::kElement) return false;
  const auto& element =
      static_cast<const stablehlo::ElementTypeWrapper&>(*ranked.elementType());
  return element.elementType() == expected_element;
}

}  // namespace
}  // namespace lczero

static void SetTargetVersionEnv(const char* value) {
#ifdef _WIN32
  _putenv_s("LC0_STABLEHLO_TARGET_VERSION", value);
#else
  if (value[0] == '\0') {
    unsetenv("LC0_STABLEHLO_TARGET_VERSION");
  } else {
    setenv("LC0_STABLEHLO_TARGET_VERSION", value, 1);
  }
#endif
}

int main() {
  using lczero::stablehlo::TypePtr;
  using lczero::stablehlo::WireBlock;
  using lczero::stablehlo::ValueRef;
  using lczero::stablehlo::lowering::LowerType;
  using lczero::stablehlo::lowering::LowerReduceOpIntoBlock;
  using lczero::stablehlo::lowering::LowerConstantOpIntoBlock;
  using lczero::stablehlo::lowering::LowerToWireModule;
  using lczero::stablehlo::lowering::ResolveTargetVersion;
  using lczero::stablehlo::lowering::BuildProducerString;
  using lczero::stablehlo::lowering::ApplyTargetVersionToBytecode;
  using lczero::stablehlo::lowering::TargetVersion;
  using lczero::stablehlo::ParseStableHLOVersion;
  using lczero::stablehlo::PluginStableHLOVersionWindow;
  using lczero::stablehlo::ValidateStableHLOVersion;
  using lczero::stablehlo::VersionCheckResult;
  using lczero::stablehlo::semantic::ElementType;
  using lczero::stablehlo::semantic::SemanticFunction;
  using lczero::stablehlo::semantic::SemanticModule;
  using lczero::stablehlo::semantic::OpKind;
  using lczero::stablehlo::semantic::ReduceParams;
  using lczero::stablehlo::semantic::SemanticOp;
  using lczero::stablehlo::semantic::TensorType;

  const TypePtr scalar = LowerType(TensorType{ElementType::kF32, {}});
  if (!lczero::CheckRankedTensor(scalar, lczero::stablehlo::ElementType::kF32, {})) {
    return 1;
  }

  const TypePtr one_d = LowerType(TensorType{ElementType::kS32, {128}});
  if (!lczero::CheckRankedTensor(one_d, lczero::stablehlo::ElementType::kSI32,
                                 {128})) {
    return 2;
  }

  const TypePtr four_d = LowerType(TensorType{ElementType::kPred, {1, 3, 8, 8}});
  if (!lczero::CheckRankedTensor(four_d, lczero::stablehlo::ElementType::kBool,
                                 {1, 3, 8, 8})) {
    return 3;
  }

  WireBlock block;
  std::vector<ValueRef> value_refs = {0, 1};
  std::vector<TypePtr> value_types = {
      LowerType(TensorType{ElementType::kF32, {2, 2}}),
      LowerType(TensorType{ElementType::kF32, {}}),
  };
  ReduceParams reduce_params;
  reduce_params.dimensions = {1};
  reduce_params.reduce_op = ReduceParams::ReduceOp::kAdd;

  SemanticOp reduce_op;
  reduce_op.kind = OpKind::kReduce;
  reduce_op.operands = {0, 1};
  reduce_op.result_types = {TensorType{ElementType::kF32, {2}}};
  reduce_op.attrs = reduce_params;

  if (!LowerReduceOpIntoBlock(reduce_op, &block, &value_refs, &value_types)) {
    return 4;
  }
  if (block.ops.size() != 1 || block.ops.front().opName != "reduce_v1") {
    return 5;
  }
  if (block.ops.front().regions.size() != 1) {
    return 6;
  }

  SemanticOp constant_op;
  constant_op.kind = OpKind::kConstant;
  constant_op.result_types = {TensorType{ElementType::kF32, {}}};
  constant_op.attrs = std::vector<uint8_t>{0, 0, 0, 0};
  if (!LowerConstantOpIntoBlock(constant_op, &block, &value_refs, &value_types)) {
    return 7;
  }
  if (block.ops.size() != 2 || block.ops.back().opName != "constant_v1") {
    return 8;
  }
  if (!block.ops.back().constantProps.has_value() ||
      block.ops.back().constantProps->rawData.size() != 4) {
    return 9;
  }

  SemanticModule semantic_module;
  SemanticFunction main_fn;
  main_fn.name = "main";
  main_fn.param_types = {TensorType{ElementType::kF32, {}},
                         TensorType{ElementType::kF32, {}}};
  SemanticOp add_op;
  add_op.kind = OpKind::kAdd;
  add_op.operands = {0, 1};
  add_op.result_types = {TensorType{ElementType::kF32, {}}};
  main_fn.ops.push_back(add_op);
  SemanticOp return_op;
  return_op.kind = OpKind::kReturn;
  return_op.operands = {2};
  main_fn.ops.push_back(return_op);
  semantic_module.functions.push_back(std::move(main_fn));

  const auto wire_module = LowerToWireModule(semantic_module);
  if (wire_module.rootOp.opName != "module") {
    return 10;
  }
  if (wire_module.rootOp.regions.size() != 1 ||
      wire_module.rootOp.regions.front().blocks.size() != 1) {
    return 11;
  }
  if (wire_module.rootOp.regions.front().blocks.front().ops.size() != 1) {
    return 12;
  }
  const auto& function_op = wire_module.rootOp.regions.front().blocks.front().ops.front();
  if (function_op.opName != "func_v1" || function_op.regions.size() != 1) {
    return 13;
  }

  SetTargetVersionEnv("");
  const auto default_version = ResolveTargetVersion();
  if (default_version.major != 1 || default_version.minor != 0 ||
      default_version.patch != 0) {
    return 14;
  }
  if (BuildProducerString(default_version) != "StableHLO_v1.0.0") {
    return 15;
  }

  SetTargetVersionEnv("1.13.7");
  const auto override_version = ResolveTargetVersion();
  if (override_version.major != 1 || override_version.minor != 13 ||
      override_version.patch != 7) {
    return 16;
  }
  if (BuildProducerString(override_version) != "StableHLO_v1.13.7") {
    return 17;
  }

  // Version parse / validate checks (Phase 4.4 + 4.5)
  if (!ParseStableHLOVersion("1.13.7").has_value()) {
    return 18;
  }
  if (!ParseStableHLOVersion("1, 13, 7").has_value()) {
    return 19;
  }
  if (ParseStableHLOVersion("1.13").has_value()) {
    return 20;
  }

  PluginStableHLOVersionWindow window{
      TargetVersion{1, 0, 0},
      TargetVersion{1, 5, 0},
  };
  auto check = ValidateStableHLOVersion(window, TargetVersion{0, 9, 0});
  if (check.status != VersionCheckResult::Status::kTargetTooOld) {
    return 21;
  }
  if (check.error_message.find("Error: StableHLO version mismatch") ==
          std::string::npos ||
      check.error_message.find("Lc0 targets: 0.9.0") == std::string::npos ||
      check.error_message.find("Plugin supports: 1.0.0 - 1.5.0") ==
          std::string::npos ||
      check.error_message.find("Status: target too old") == std::string::npos) {
    return 22;
  }
  check = ValidateStableHLOVersion(window, TargetVersion{2, 0, 0});
  if (check.status != VersionCheckResult::Status::kTargetTooNew ||
      check.error_message.find("Status: target too new") == std::string::npos) {
    return 23;
  }
  check = ValidateStableHLOVersion(window, TargetVersion{1, 2, 0});
  if (check.status != VersionCheckResult::Status::kOk ||
      !check.error_message.empty()) {
    return 24;
  }

  PluginStableHLOVersionWindow missing_window{};
  check = ValidateStableHLOVersion(missing_window, TargetVersion{1, 0, 0});
  if (check.status != VersionCheckResult::Status::kMissingAttrs) {
    return 25;
  }
  if (check.error_message.find(
          "Error: Plugin does not expose StableHLO version attributes") ==
          std::string::npos ||
      check.error_message.find("Set LC0_STABLEHLO_TARGET_VERSION= to override") ==
          std::string::npos ||
      check.error_message.find("Or update to a newer PJRT plugin") ==
          std::string::npos) {
    return 26;
  }

  PluginStableHLOVersionWindow partial_window{TargetVersion{1, 0, 0},
                                              std::nullopt};
  check = ValidateStableHLOVersion(partial_window, TargetVersion{1, 0, 0});
  if (check.status != VersionCheckResult::Status::kMissingAttrs) {
    return 27;
  }

  std::vector<uint8_t> fake_bytecode = {'M', 'L', 0xEF, 'R', 0x06};
  const std::string original_producer = "StableHLO_v1.0.0";
  fake_bytecode.insert(fake_bytecode.end(), original_producer.begin(),
                       original_producer.end());
  fake_bytecode.push_back(0);
  fake_bytecode.push_back(0x01);
  fake_bytecode.push_back(0xAA);

  const auto rewritten = ApplyTargetVersionToBytecode(fake_bytecode);
  std::string rewritten_producer;
  for (size_t i = 5; i < rewritten.size() && rewritten[i] != 0; ++i) {
    rewritten_producer.push_back(static_cast<char>(rewritten[i]));
  }
  if (rewritten_producer != "StableHLO_v1.13.7") {
    return 28;
  }
  if (rewritten.size() < 2 || rewritten[rewritten.size() - 2] != 0x01 ||
      rewritten[rewritten.size() - 1] != 0xAA) {
    return 29;
  }

  // Wrong entry-function name must fail fast (Phase 4.6).
  SemanticModule wrong_entry_module;
  SemanticFunction wrong_entry_fn;
  wrong_entry_fn.name = "not_main";
  wrong_entry_fn.param_types = {TensorType{ElementType::kF32, {}}};
  SemanticOp wrong_entry_return;
  wrong_entry_return.kind = OpKind::kReturn;
  wrong_entry_return.operands = {0};
  wrong_entry_fn.ops.push_back(wrong_entry_return);
  wrong_entry_module.functions.push_back(std::move(wrong_entry_fn));

  bool saw_wrong_entry_error = false;
  try {
    (void)LowerToWireModule(wrong_entry_module);
  } catch (const std::exception& e) {
    saw_wrong_entry_error =
        std::string(e.what()).find("must be named 'main'") != std::string::npos;
  }
  if (!saw_wrong_entry_error) {
    return 30;
  }

  // Drift guard: verify every OpKind has a matching OpSpec entry.
  for (size_t i = 0;
       i <= static_cast<size_t>(
                ::lczero::stablehlo::semantic::OpKind::kReturn);
       ++i) {
    const auto kind = static_cast<::lczero::stablehlo::semantic::OpKind>(i);
    (void)::lczero::stablehlo::lowering::GetOpSpec(kind);
  }

  SetTargetVersionEnv("");

  return 0;
}
