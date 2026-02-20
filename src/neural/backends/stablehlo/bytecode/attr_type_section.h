// =============================================================================
// attr_type_section.h - AttrType + AttrTypeOffset Section Writers
// =============================================================================
// Implements Section 3 (AttrTypeOffset) and Section 2 (AttrType) encoding
// for MLIR bytecode v6.
//
// Section IDs:
//   Section 2 = kAttrType (actual attr/type payloads)
//   Section 3 = kAttrTypeOffset (offsets + metadata)
//
// Emission order: Section 3 is emitted BEFORE Section 2 in the bytecode stream.
//
// Wire Format (Section 3 - AttrTypeOffset):
//   numAttrs: varint
//   numTypes: varint
//   For each dialect with attrs:
//     dialectNumber: varint
//     count: varint
//     For each attr:
//       deltaOffsetWithFlag(delta, hasCustomEncoding)
//   For each dialect with types:
//     dialectNumber: varint
//     count: varint
//     For each type:
//       deltaOffsetWithFlag(delta, hasCustomEncoding)
//
// Wire Format (Section 2 - AttrType):
//   Attr payloads in dialect-grouped order
//   Type payloads in dialect-grouped order

#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "stablehlo/encoding.h"
#include "stablehlo/attr_interner.h"
#include "stablehlo/type_interner.h"
#include "stablehlo/string_interner.h"

namespace lczero {
namespace stablehlo {

// =============================================================================
// EncodedEntry - Represents a pre-encoded attr or type with its metadata
// =============================================================================
struct EncodedEntry {
  std::vector<uint8_t> payload;  // The encoded bytes
  size_t dialect_id;             // Dialect this entry belongs to
  bool has_custom_encoding;      // True for VHLO types/attrs
};

// =============================================================================
// AttrTypeSectionWriter - Builds and emits Sections 2 and 3
// =============================================================================
class AttrTypeSectionWriter {
 public:
  AttrTypeSectionWriter() = default;

  // Add a pre-encoded attribute.
  // Returns the attr index.
  size_t addAttr(std::vector<uint8_t> payload, size_t dialect_id,
                 bool has_custom_encoding = true);

  // Add a pre-encoded type.
  // Returns the type index.
  size_t addType(std::vector<uint8_t> payload, size_t dialect_id,
                 bool has_custom_encoding = true);

  // Number of attrs/types.
  size_t numAttrs() const { return attrs_.size(); }
  size_t numTypes() const { return types_.size(); }

  // Update a type payload by index (used when reserving slots before encoding).
  void setTypePayload(size_t index, std::vector<uint8_t> payload);

  // Update an attr payload by index (used when reserving slots before encoding).
  void setAttrPayload(size_t index, std::vector<uint8_t> payload);

  // Write Section 3 (AttrTypeOffset) payload.
  void writeOffsetSection(EncodingEmitter& emitter) const;

  // Write Section 2 (AttrType) payload.
  void writeAttrTypeSection(EncodingEmitter& emitter) const;

  // Get Section 3 (AttrTypeOffset) payload bytes.
  std::vector<uint8_t> toOffsetBytes() const;

  // Get Section 2 (AttrType) payload bytes.
  std::vector<uint8_t> toAttrTypeBytes() const;

 private:
  // Write grouped entries (attrs or types) to the offset section.
  void writeGroupedOffsets(EncodingEmitter& emitter,
                           const std::vector<EncodedEntry>& entries) const;

  // Write grouped entry payloads to the AttrType section.
  void writeGroupedPayloads(EncodingEmitter& emitter,
                            const std::vector<EncodedEntry>& entries) const;

  // Get entries grouped by dialect, in dialect order.
  // Returns [(dialect_id, [entry_indices...]), ...]
  std::vector<std::pair<size_t, std::vector<size_t>>> groupByDialect(
      const std::vector<EncodedEntry>& entries) const;

  std::vector<EncodedEntry> attrs_;
  std::vector<EncodedEntry> types_;
};

// =============================================================================
// Type Encoding Helpers
// =============================================================================

// Encode a simple element type (f32, i64, etc.) with no payload.
std::vector<uint8_t> encodeElementType(ElementType et);

// Encode a RankedTensorType.
// shape: the tensor dimensions
// element_type_index: index of the element type in the type table
std::vector<uint8_t> encodeRankedTensor(const std::vector<int64_t>& shape,
                                        size_t element_type_index);

// Encode a FunctionType.
// input_type_indices: indices of input types in the type table
// output_type_indices: indices of output types in the type table
std::vector<uint8_t> encodeFunctionType(
    const std::vector<size_t>& input_type_indices,
    const std::vector<size_t>& output_type_indices);

// Encode a TupleType.
// element_type_indices: indices of element types in the type table
std::vector<uint8_t> encodeTupleType(
    const std::vector<size_t>& element_type_indices);

// =============================================================================
// Attr Encoding Helpers
// =============================================================================
// NOTE: These helpers use VhloAttrCode constants which are verified against
// golden .mlirbc files. See attrs.h for the verified codes.

// Encode a StringAttr.
// string_index: index in the string table
std::vector<uint8_t> encodeStringAttr(size_t string_index);

// Encode a builtin StringAttr.
// string_index: index in the string table
std::vector<uint8_t> encodeBuiltinStringAttr(size_t string_index);

// Encode an IntegerAttr.
// type_index: index of the integer type in the type table
// value: the integer value (encoded as signed varint, zigzag)
std::vector<uint8_t> encodeIntegerAttr(size_t type_index, int64_t value);

// Encode a TypeAttr.
// type_index: index of the type in the type table
std::vector<uint8_t> encodeTypeAttr(size_t type_index);

// Encode an ArrayAttr.
// attr_indices: indices of element attrs in the attr table
std::vector<uint8_t> encodeArrayAttr(const std::vector<size_t>& attr_indices);

// =============================================================================
// Golden-Verified Attr Encoding Helpers
// =============================================================================

// Encode a BooleanAttr.
// Golden proof: gather_i32 has bytes=0501 for false
// Format: code=2, value=0/1 (plain varints)
std::vector<uint8_t> encodeBooleanAttr(bool value);

// Encode a ComparisonDirectionAttr.
// Golden proof: compare_select entry bytes=0707 → code=3, dir=3
// Format: code=3, direction (plain varint)
std::vector<uint8_t> encodeComparisonDirectionAttr(uint64_t direction);

// Encode a ComparisonTypeAttr.
// Golden proof: compare_select entry bytes=0901 → code=4, type=0
// Format: code=4, type (plain varint)
std::vector<uint8_t> encodeComparisonTypeAttr(uint64_t comparison_type);

// Encode a PrecisionAttr.
// Format: code=11, value (plain varint). Enum: 0=DEFAULT, 1=HIGH, 2=HIGHEST.
std::vector<uint8_t> encodePrecisionAttr(uint64_t precision);

// Encode a TensorAttr (dense constant).
// Golden proof: reduce_add entry bytes=1f010900000000 → code=15, typeIdx=0, len=4, data
// Format: code=15, typeIndex, byteLength, rawBytes
std::vector<uint8_t> encodeTensorAttr(size_t type_index,
                                      const std::vector<uint8_t>& raw_bytes);

}  // namespace stablehlo
}  // namespace lczero
