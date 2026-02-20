// =============================================================================
// dialect_section.h - Dialect Section Writer (Section ID = 1)
// =============================================================================
// Implements Section 1 (Dialect) payload encoding for MLIR bytecode v6.
//
// Wire Format (Version 6):
//   numDialects: varint
//   For each dialect:
//     emitVarIntWithFlag(nameStringIndex, hasVersion)
//     [if hasVersion: embedded kDialectVersions section - not used for StableHLO]
//
//   numOpNames: varint
//   For each dialect group (ops grouped by parent dialect):
//     dialectNumber: varint
//     count: varint
//     For each op in group:
//       emitVarIntWithFlag(opNameStringIndex, isRegistered)
//
// VarIntWithFlag encoding:
//   packed = (value << 1) | (flag ? 1 : 0)
//   Then emit packed as varint.

#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "stablehlo/encoding.h"
#include "stablehlo/string_interner.h"

namespace lczero {
namespace stablehlo {

// =============================================================================
// DialectOpInfo - Information about a single operation in a dialect
// =============================================================================
// Note: Named DialectOpInfo to avoid collision with ir_section.h::OpInfo
struct DialectOpInfo {
  std::string name;           // "module", "func_v1", etc. (without dialect prefix)
  bool is_registered = true;  // Always true for StableHLO/VHLO ops
};

// =============================================================================
// DialectInfo - Information about a dialect and its operations
// =============================================================================
struct DialectInfo {
  std::string name;           // "builtin", "vhlo"
  bool has_version = false;   // Always false for our use case
  std::vector<DialectOpInfo> ops;    // Operations in this dialect
};

// =============================================================================
// DialectSectionWriter - Builds and emits the Dialect Section payload
// =============================================================================
class DialectSectionWriter {
 public:
  DialectSectionWriter() = default;

  // Add a dialect and return its dialect number (0-indexed).
  // Dialect numbers are assigned in order of addition.
  size_t addDialect(const std::string& name, bool has_version = false);

  // Add an operation to a dialect.
  // dialect_id must be a valid index returned by addDialect().
  void addOp(size_t dialect_id, const std::string& op_name, bool is_registered = true);

  // Get dialect info by index.
  const DialectInfo& getDialect(size_t index) const;

  // Number of dialects.
  size_t numDialects() const { return dialects_.size(); }

  // Total number of operations across all dialects.
  size_t numOpNames() const;

  // Write the dialect section payload to an emitter.
  // Interns all dialect names and op names into the StringInterner.
  // IMPORTANT: Call this BEFORE writing the string section, so all
  // dialect/op names are interned with the correct indices.
  void write(EncodingEmitter& emitter, StringInterner& strings) const;

  // Get the raw bytes of the dialect section payload (for testing).
  std::vector<uint8_t> toBytes(StringInterner& strings) const;

 private:
  std::vector<DialectInfo> dialects_;
  std::unordered_map<std::string, size_t> name_to_id_;
};

}  // namespace stablehlo
}  // namespace lczero
