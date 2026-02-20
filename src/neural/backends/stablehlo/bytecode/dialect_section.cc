// =============================================================================
// dialect_section.cc - Dialect Section Writer Implementation
// =============================================================================

#include "stablehlo/dialect_section.h"

#include <cassert>

namespace lczero {
namespace stablehlo {

size_t DialectSectionWriter::addDialect(const std::string& name, bool has_version) {
  auto it = name_to_id_.find(name);
  if (it != name_to_id_.end()) return it->second;

  size_t id = dialects_.size();
  dialects_.push_back(DialectInfo{name, has_version, {}});
  name_to_id_[dialects_.back().name] = id;
  return id;
}

void DialectSectionWriter::addOp(size_t dialect_id, const std::string& op_name, bool is_registered) {
  assert(dialect_id < dialects_.size());
  dialects_[dialect_id].ops.push_back(DialectOpInfo{op_name, is_registered});
}

const DialectInfo& DialectSectionWriter::getDialect(size_t index) const {
  assert(index < dialects_.size());
  return dialects_[index];
}

size_t DialectSectionWriter::numOpNames() const {
  size_t total = 0;
  for (const auto& d : dialects_) {
    total += d.ops.size();
  }
  return total;
}

void DialectSectionWriter::write(EncodingEmitter& emitter, StringInterner& strings) const {
  // Wire Format (Version 6):
  //   numDialects: varint
  //   For each dialect:
  //     emitVarIntWithFlag(nameStringIndex, hasVersion)
  //   numOpNames: varint
  //   For each dialect group:
  //     dialectNumber: varint
  //     count: varint
  //     For each op:
  //       emitVarIntWithFlag(opNameStringIndex, isRegistered)

  // 1. Emit numDialects.
  emitter.emitVarInt(dialects_.size());

  // 2. Emit dialect entries.
  //    First, intern all dialect names to get their string indices.
  std::vector<size_t> dialectNameIndices;
  dialectNameIndices.reserve(dialects_.size());
  for (const auto& d : dialects_) {
    size_t idx = strings.intern(d.name);
    dialectNameIndices.push_back(idx);
  }

  // Now emit each dialect: varIntWithFlag(nameIndex, hasVersion).
  for (size_t i = 0; i < dialects_.size(); ++i) {
    emitter.emitVarIntWithFlag(dialectNameIndices[i], dialects_[i].has_version);
    // Note: If hasVersion is true, we'd need to emit an embedded DialectVersions
    // section here. For StableHLO, hasVersion is always false.
    assert(!dialects_[i].has_version && "Dialect versioning not implemented");
  }

  // 3. Emit numOpNames (total across all dialects).
  emitter.emitVarInt(numOpNames());

  // 4. Emit op name groups.
  //    Each group: dialectNumber, count, then op entries.
  for (size_t dialectNum = 0; dialectNum < dialects_.size(); ++dialectNum) {
    const auto& d = dialects_[dialectNum];
    if (d.ops.empty()) continue;  // Skip dialects with no ops.

    // Emit dialectNumber.
    emitter.emitVarInt(dialectNum);

    // Emit count.
    emitter.emitVarInt(d.ops.size());

    // Emit each op: varIntWithFlag(opNameStringIndex, isRegistered).
    for (const auto& op : d.ops) {
      size_t opNameIdx = strings.intern(op.name);
      emitter.emitVarIntWithFlag(opNameIdx, op.is_registered);
    }
  }
}

std::vector<uint8_t> DialectSectionWriter::toBytes(StringInterner& strings) const {
  EncodingEmitter emitter;
  write(emitter, strings);
  return emitter.bytes();
}

}  // namespace stablehlo
}  // namespace lczero
