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

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "neural/backends/stablehlo/lowering/lowering.h"
#include "neural/backends/stablehlo/semantic/semantic_ir.h"

namespace lczero::stablehlo {

struct PluginStableHLOVersionWindow {
  std::optional<lowering::TargetVersion> minimum;
  std::optional<lowering::TargetVersion> current;
};

struct VersionCheckResult {
  enum class Status { kOk, kTargetTooOld, kTargetTooNew, kMissingAttrs };
  Status status = Status::kOk;
  std::string error_message;
};

// Parses StableHLO version text.
// Accepts:
//   - "major.minor.patch"
//   - "major, minor, patch"
std::optional<lowering::TargetVersion> ParseStableHLOVersion(
    std::string_view value);

// Validates StableHLO version compatibility for the chosen target version.
// Pure function: no env-var reads, no PJRT dependencies, no side effects.
VersionCheckResult ValidateStableHLOVersion(
    const PluginStableHLOVersionWindow& plugin_window,
    const lowering::TargetVersion& target_version);

// Converts semantic IR module to StableHLO bytecode (.mlirbc).
std::vector<uint8_t> SemanticModuleToMlirbc(
    const semantic::SemanticModule& module);

}  // namespace lczero::stablehlo
