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

#include "neural/backends/stablehlo/stablehlo_backend.h"

#include <algorithm>
#include <cctype>

#include "neural/backends/stablehlo/lowering/lowering.h"
#include "stablehlo/numbering.h"
#include "utils/exception.h"

namespace lczero::stablehlo {

namespace {

std::string FormatVersion(const lowering::TargetVersion& version) {
  return std::to_string(version.major) + "." + std::to_string(version.minor) +
         "." + std::to_string(version.patch);
}

int CompareVersion(const lowering::TargetVersion& lhs,
                   const lowering::TargetVersion& rhs) {
  if (lhs.major != rhs.major) return lhs.major < rhs.major ? -1 : 1;
  if (lhs.minor != rhs.minor) return lhs.minor < rhs.minor ? -1 : 1;
  if (lhs.patch != rhs.patch) return lhs.patch < rhs.patch ? -1 : 1;
  return 0;
}

std::optional<int> ParseComponent(std::string_view value) {
  if (value.empty()) return std::nullopt;
  if (!std::all_of(value.begin(), value.end(),
                   [](char c) { return std::isdigit(c) != 0; })) {
    return std::nullopt;
  }
  try {
    return std::stoi(std::string(value));
  } catch (...) {
    return std::nullopt;
  }
}

std::string MakeMissingAttrsError() {
  return "Error: Plugin does not expose StableHLO version attributes\n"
         "Set LC0_STABLEHLO_TARGET_VERSION= to override\n"
         "Or update to a newer PJRT plugin";
}

std::string MakeMismatchError(const lowering::TargetVersion& target,
                              const lowering::TargetVersion& minimum,
                              const lowering::TargetVersion& current,
                              std::string_view status) {
  return "Error: StableHLO version mismatch\n"
         "Lc0 targets: " +
         FormatVersion(target) + "\nPlugin supports: " +
         FormatVersion(minimum) + " - " + FormatVersion(current) +
         "\nStatus: " + std::string(status);
}

}  // namespace

std::optional<lowering::TargetVersion> ParseStableHLOVersion(
    std::string_view value) {
  std::string normalized;
  normalized.reserve(value.size());
  for (char c : value) {
    if (std::isspace(static_cast<unsigned char>(c)) != 0) continue;
    normalized.push_back(c == ',' ? '.' : c);
  }
  if (normalized.empty()) return std::nullopt;

  const size_t first = normalized.find('.');
  if (first == std::string::npos) return std::nullopt;
  const size_t second = normalized.find('.', first + 1);
  if (second == std::string::npos) return std::nullopt;
  if (normalized.find('.', second + 1) != std::string::npos) return std::nullopt;

  const auto major = ParseComponent(
      std::string_view(normalized).substr(0, first));
  const auto minor = ParseComponent(
      std::string_view(normalized).substr(first + 1, second - first - 1));
  const auto patch =
      ParseComponent(std::string_view(normalized).substr(second + 1));
  if (!major.has_value() || !minor.has_value() || !patch.has_value()) {
    return std::nullopt;
  }

  return lowering::TargetVersion{*major, *minor, *patch};
}

VersionCheckResult ValidateStableHLOVersion(
    const PluginStableHLOVersionWindow& plugin_window,
    const lowering::TargetVersion& target_version) {
  if (!plugin_window.minimum.has_value() || !plugin_window.current.has_value()) {
    return {VersionCheckResult::Status::kMissingAttrs, MakeMissingAttrsError()};
  }

  const lowering::TargetVersion& minimum = *plugin_window.minimum;
  const lowering::TargetVersion& current = *plugin_window.current;

  if (CompareVersion(target_version, minimum) < 0) {
    return {VersionCheckResult::Status::kTargetTooOld,
            MakeMismatchError(target_version, minimum, current,
                              "target too old")};
  }
  if (CompareVersion(target_version, current) > 0) {
    return {VersionCheckResult::Status::kTargetTooNew,
            MakeMismatchError(target_version, minimum, current,
                              "target too new")};
  }
  return {VersionCheckResult::Status::kOk, ""};
}

std::vector<uint8_t> SemanticModuleToMlirbc(
    const semantic::SemanticModule& module) {
  const WireModule wire_module = lowering::LowerToWireModule(module);

  BytecodeAssembler assembler;
  std::vector<uint8_t> bytes = assembler.assemble(wire_module);
  bytes = lowering::ApplyTargetVersionToBytecode(bytes);
  if (bytes.empty()) {
    throw Exception("StableHLO backend produced empty bytecode");
  }
  return bytes;
}

}  // namespace lczero::stablehlo
