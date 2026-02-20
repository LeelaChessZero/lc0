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

#include <string>

#include "stablehlo/portable_module.h"

#include "neural/backends/stablehlo/semantic/semantic_ir.h"

namespace lczero::stablehlo::lowering {

// Converts a semantic tensor type into the writer's ranked tensor type.
TypePtr LowerType(const semantic::TensorType& type);

// Lowers straight-line (non-region, non-constant) semantic ops into a wire
// block and extends ValueId mappings for produced results.
//
// Returns false when the op must be handled by a dedicated lowering pass
// (parameter/constant/reduce/return/module-scaffolding stages).
// Throws Exception on structural errors (null pointers, operand arity
// mismatch, unsupported op kind that should have been dispatched elsewhere).
bool LowerStraightLineOpIntoBlock(const semantic::SemanticOp& op,
                                  WireBlock* block,
                                  std::vector<ValueRef>* value_refs,
                                  std::vector<TypePtr>* value_types,
                                  size_t location_index = kInvalidIndex);

// Lowers reduce ops with synthesized add/max/multiply body regions.
//
// Returns false when op.kind != kReduce (caller should try another dispatcher).
// Throws Exception on structural errors (null pointers, operand mapping
// failures, empty operand list).
bool LowerReduceOpIntoBlock(const semantic::SemanticOp& op, WireBlock* block,
                            std::vector<ValueRef>* value_refs,
                            std::vector<TypePtr>* value_types,
                            size_t location_index = kInvalidIndex);

// Lowers constant ops with raw byte payloads.
//
// Returns false when op.kind != kConstant (caller should try another dispatcher).
// Throws Exception on structural errors (null pointers, missing raw byte
// attrs, unexpected result type count).
bool LowerConstantOpIntoBlock(const semantic::SemanticOp& op, WireBlock* block,
                              std::vector<ValueRef>* value_refs,
                              std::vector<TypePtr>* value_types,
                              size_t location_index = kInvalidIndex);

struct TargetVersion {
  int major = 1;
  int minor = 0;
  int patch = 0;
};

// Reads LC0_STABLEHLO_TARGET_VERSION (major.minor.patch). Defaults to 1.0.0.
TargetVersion ResolveTargetVersion();

// Returns bytecode producer tag, for example: StableHLO_v1.0.0.
std::string BuildProducerString(const TargetVersion& version);

// Rewrites bytecode header producer string to the resolved target version.
std::vector<uint8_t> ApplyTargetVersionToBytecode(
    const std::vector<uint8_t>& bytecode);

// Builds a complete wire module (module -> func -> block -> ops) from semantic IR.
WireModule LowerToWireModule(const semantic::SemanticModule& module);

}  // namespace lczero::stablehlo::lowering
