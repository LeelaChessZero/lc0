// =============================================================================
// attr_type_section.cc - AttrType + AttrTypeOffset Section Writers
// =============================================================================

#include "stablehlo/attr_type_section.h"

#include <algorithm>
#include <cassert>
#include <map>

#include "stablehlo/attrs.h"  // For VhloAttrCode

namespace lczero {
namespace stablehlo {

// =============================================================================
// AttrTypeSectionWriter
// =============================================================================

size_t AttrTypeSectionWriter::addAttr(std::vector<uint8_t> payload,
                                       size_t dialect_id,
                                       bool has_custom_encoding) {
  size_t index = attrs_.size();
  attrs_.push_back(EncodedEntry{std::move(payload), dialect_id, has_custom_encoding});
  return index;
}

size_t AttrTypeSectionWriter::addType(std::vector<uint8_t> payload,
                                       size_t dialect_id,
                                       bool has_custom_encoding) {
  size_t index = types_.size();
  types_.push_back(EncodedEntry{std::move(payload), dialect_id, has_custom_encoding});
  return index;
}

void AttrTypeSectionWriter::setTypePayload(size_t index, std::vector<uint8_t> payload) {
  assert(index < types_.size() && "type index out of range");
  types_[index].payload = std::move(payload);
}

void AttrTypeSectionWriter::setAttrPayload(size_t index, std::vector<uint8_t> payload) {
  assert(index < attrs_.size() && "attr index out of range");
  attrs_[index].payload = std::move(payload);
}

std::vector<std::pair<size_t, std::vector<size_t>>> AttrTypeSectionWriter::groupByDialect(
    const std::vector<EncodedEntry>& entries) const {
  // Group indices by dialect ID.
  std::map<size_t, std::vector<size_t>> groups;
  for (size_t i = 0; i < entries.size(); ++i) {
    groups[entries[i].dialect_id].push_back(i);
  }

  // Convert to sorted vector of pairs.
  std::vector<std::pair<size_t, std::vector<size_t>>> result;
  result.reserve(groups.size());
  for (auto& [dialect_id, indices] : groups) {
    result.emplace_back(dialect_id, std::move(indices));
  }
  return result;
}

void AttrTypeSectionWriter::writeGroupedOffsets(
    EncodingEmitter& emitter,
    const std::vector<EncodedEntry>& entries) const {

  // ================================================================
  // DO NOT reorder attrs/types here. Ordering is finalized in
  // numbering.cc (finalizeAttrOrder + preNumberTypes).
  // Reordering in this writer breaks indices already baked into
  // properties blobs.
  // ================================================================
  auto groups = groupByDialect(entries);

  for (const auto& [dialect_id, indices] : groups) {
    if (indices.empty()) continue;

    // Emit dialectNumber.
    emitter.emitVarInt(dialect_id);

    // Emit count.
    emitter.emitVarInt(indices.size());

    // Emit delta offsets with hasCustomEncoding flag.
    // Delta is the SIZE of each entry (which is the offset delta to the next).
    for (size_t idx : indices) {
      const auto& entry = entries[idx];
      size_t delta = entry.payload.size();
      emitter.emitVarIntWithFlag(delta, entry.has_custom_encoding);
    }
  }
}

void AttrTypeSectionWriter::writeGroupedPayloads(
    EncodingEmitter& emitter,
    const std::vector<EncodedEntry>& entries) const {

  auto groups = groupByDialect(entries);

  for (const auto& [dialect_id, indices] : groups) {
    for (size_t idx : indices) {
      const auto& entry = entries[idx];
      emitter.emitBytes(entry.payload);
    }
  }
}

void AttrTypeSectionWriter::writeOffsetSection(EncodingEmitter& emitter) const {
  // Section 3 (AttrTypeOffset) format:
  //   numAttrs: varint
  //   numTypes: varint
  //   attr groups (dialectNumber, count, deltas...)
  //   type groups (dialectNumber, count, deltas...)

  // 1. Emit counts.
  emitter.emitVarInt(attrs_.size());
  emitter.emitVarInt(types_.size());

  // 2. Emit attr groups.
  writeGroupedOffsets(emitter, attrs_);

  // 3. Emit type groups.
  writeGroupedOffsets(emitter, types_);
}

void AttrTypeSectionWriter::writeAttrTypeSection(EncodingEmitter& emitter) const {
  // Section 2 (AttrType) format:
  //   attr payloads (in dialect-grouped order)
  //   type payloads (in dialect-grouped order)

  // 1. Write attr payloads.
  writeGroupedPayloads(emitter, attrs_);

  // 2. Write type payloads.
  writeGroupedPayloads(emitter, types_);
}

std::vector<uint8_t> AttrTypeSectionWriter::toOffsetBytes() const {
  EncodingEmitter emitter;
  writeOffsetSection(emitter);
  return emitter.bytes();
}

std::vector<uint8_t> AttrTypeSectionWriter::toAttrTypeBytes() const {
  EncodingEmitter emitter;
  writeAttrTypeSection(emitter);
  return emitter.bytes();
}

// =============================================================================
// Type Encoding Helpers
// =============================================================================

std::vector<uint8_t> encodeElementType(ElementType et) {
  EncodingEmitter emitter;
  uint64_t code = getTypeCode(et);
  emitter.emitVarInt(code);
  return emitter.bytes();
}

std::vector<uint8_t> encodeRankedTensor(const std::vector<int64_t>& shape,
                                        size_t element_type_index) {
  EncodingEmitter emitter;

  // Type code for RankedTensorV1Type = 20.
  emitter.emitVarInt(TypeCode::kRankedTensorV1);

  // Shape: count followed by signed varints for each dimension.
  emitter.emitVarInt(shape.size());
  for (int64_t dim : shape) {
    emitter.emitSignedVarInt(dim);
  }

  // Element type index.
  emitter.emitVarInt(element_type_index);

  return emitter.bytes();
}

std::vector<uint8_t> encodeFunctionType(
    const std::vector<size_t>& input_type_indices,
    const std::vector<size_t>& output_type_indices) {
  EncodingEmitter emitter;

  // Type code for FunctionV1Type = 8 (verified against pinned VhloBytecode.cpp).
  emitter.emitVarInt(TypeCode::kFunctionV1);

  // Inputs: count followed by type indices.
  emitter.emitVarInt(input_type_indices.size());
  for (size_t idx : input_type_indices) {
    emitter.emitVarInt(idx);
  }

  // Outputs: count followed by type indices.
  emitter.emitVarInt(output_type_indices.size());
  for (size_t idx : output_type_indices) {
    emitter.emitVarInt(idx);
  }

  return emitter.bytes();
}

std::vector<uint8_t> encodeTupleType(
    const std::vector<size_t>& element_type_indices) {
  EncodingEmitter emitter;

  // Type code for TupleV1Type = 23.
  emitter.emitVarInt(TypeCode::kTupleV1);

  // Elements: count followed by type indices.
  emitter.emitVarInt(element_type_indices.size());
  for (size_t idx : element_type_indices) {
    emitter.emitVarInt(idx);
  }

  return emitter.bytes();
}

// =============================================================================
// Attr Encoding Helpers
// =============================================================================

std::vector<uint8_t> encodeStringAttr(size_t string_index) {
  EncodingEmitter emitter;

  // Attr code for StringV1Attr = 14.
  emitter.emitVarInt(VhloAttrCode::kStringV1);

  // String index.
  emitter.emitVarInt(string_index);

  return emitter.bytes();
}

std::vector<uint8_t> encodeBuiltinStringAttr(size_t string_index) {
  EncodingEmitter emitter;

  // Attr code for builtin StringAttr = 2.
  emitter.emitVarInt(BuiltinAttrCode::kStringAttr);

  // String index.
  emitter.emitVarInt(string_index);

  return emitter.bytes();
}

std::vector<uint8_t> encodeIntegerAttr(size_t type_index, int64_t value) {
  EncodingEmitter emitter;

  // Attr code for IntegerV1Attr = 9 (verified: VhloAttrCode::kIntegerV1).
  emitter.emitVarInt(VhloAttrCode::kIntegerV1);

  // Type index.
  emitter.emitVarInt(type_index);

  // Value as signed varint (zigzag).
  // MLIR bytecode uses writeAPIntWithKnownWidth → emitSignedVarInt for <=64-bit.
  emitter.emitSignedVarInt(value);

  return emitter.bytes();
}

std::vector<uint8_t> encodeTypeAttr(size_t type_index) {
  EncodingEmitter emitter;

  // Attr code for TypeV1Attr = 17.
  emitter.emitVarInt(VhloAttrCode::kTypeV1);

  // Type index.
  emitter.emitVarInt(type_index);

  return emitter.bytes();
}

std::vector<uint8_t> encodeArrayAttr(const std::vector<size_t>& attr_indices) {
  EncodingEmitter emitter;

  // Attr code for ArrayV1Attr = 1.
  emitter.emitVarInt(VhloAttrCode::kArrayV1);

  // Count followed by attr indices.
  emitter.emitVarInt(attr_indices.size());
  for (size_t idx : attr_indices) {
    emitter.emitVarInt(idx);
  }

  return emitter.bytes();
}

// =============================================================================
// NEW: Golden-Verified Attr Encoding Helpers
// =============================================================================

std::vector<uint8_t> encodeBooleanAttr(bool value) {
  EncodingEmitter emitter;

  // Attr code for BooleanV1Attr = 2 (verified: VhloAttrCode::kBooleanV1).
  // Golden proof: gather_i32 has bytes=0501 for false
  emitter.emitVarInt(VhloAttrCode::kBooleanV1);

  // Value: 0 for false, 1 for true (plain varint).
  emitter.emitVarInt(value ? 1 : 0);

  return emitter.bytes();
}

std::vector<uint8_t> encodeComparisonDirectionAttr(uint64_t direction) {
  EncodingEmitter emitter;

  // Attr code for ComparisonDirectionV1Attr = 3 (verified: VhloAttrCode::kComparisonDirectionV1).
  // Golden proof: compare_select entry bytes=0707 → code=3, dir=3
  emitter.emitVarInt(VhloAttrCode::kComparisonDirectionV1);

  // Direction value (plain varint).
  emitter.emitVarInt(direction);

  return emitter.bytes();
}

std::vector<uint8_t> encodeComparisonTypeAttr(uint64_t comparison_type) {
  EncodingEmitter emitter;

  // Attr code for ComparisonTypeV1Attr = 4 (verified: VhloAttrCode::kComparisonTypeV1).
  // Golden proof: compare_select entry bytes=0901 → code=4, type=0
  emitter.emitVarInt(VhloAttrCode::kComparisonTypeV1);

  // Comparison type value (plain varint).
  emitter.emitVarInt(comparison_type);

  return emitter.bytes();
}

std::vector<uint8_t> encodePrecisionAttr(uint64_t precision) {
  EncodingEmitter emitter;

  // Attr code for PrecisionV1Attr = 11 (verified: VhloAttrCode::kPrecisionV1).
  emitter.emitVarInt(VhloAttrCode::kPrecisionV1);

  // Precision enum value (plain varint): 0=DEFAULT, 1=HIGH, 2=HIGHEST.
  emitter.emitVarInt(precision);

  return emitter.bytes();
}

std::vector<uint8_t> encodeTensorAttr(size_t type_index,
                                      const std::vector<uint8_t>& raw_bytes) {
  EncodingEmitter emitter;

  // Attr code for TensorV1Attr = 15 (verified: VhloAttrCode::kTensorV1).
  // Golden proof: reduce_add entry bytes=1f010900000000 → code=15, typeIdx=0, len=4, data
  emitter.emitVarInt(VhloAttrCode::kTensorV1);

  // Type index.
  emitter.emitVarInt(type_index);

  // Byte length followed by raw bytes.
  emitter.emitVarInt(raw_bytes.size());
  emitter.emitBytes(raw_bytes);

  return emitter.bytes();
}

}  // namespace stablehlo
}  // namespace lczero
