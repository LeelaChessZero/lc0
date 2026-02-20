// =============================================================================
// string_interner.h - String Table Builder for MLIR Bytecode
// =============================================================================
// The string section has a quirk: sizes are emitted in REVERSE order,
// but string data is emitted in FORWARD order.
//
// Format:
//   numStrings: varint
//   sizes[numStrings]: varint[] (in REVERSE order, size+1 for NUL)
//   data: null-terminated strings (in FORWARD order)

#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>

#include "stablehlo/encoding.h"

namespace lczero {
namespace stablehlo {

// =============================================================================
// StringInterner - Deduplicates strings and assigns stable indices
// =============================================================================
class StringInterner {
 public:
  StringInterner() = default;

  // Intern a string and return its index.
  // If the string was already interned, returns the existing index.
  // Indices are assigned in order of first insertion (deterministic).
  size_t intern(std::string_view s);

  // Get index of an already-interned string.
  // Returns kInvalidIndex if not found.
  size_t lookup(std::string_view s) const;

  // Get string by index.
  std::string_view get(size_t index) const;

  // Number of interned strings.
  size_t size() const { return strings_.size(); }

  // Check if empty.
  bool empty() const { return strings_.empty(); }

  // Write the string section to an emitter.
  // This implements the reverse-size quirk.
  void write(EncodingEmitter& emitter) const;

  // Get the raw bytes of the string section (for testing).
  std::vector<uint8_t> toBytes() const;

  // Iterator access for debugging/testing.
  const std::vector<std::string>& strings() const { return strings_; }

 private:
  // Strings in insertion order (determines indices).
  std::vector<std::string> strings_;

  // Map from string content to index for O(1) lookup.
  // We use std::string as key to avoid dangling string_view references
  // when the strings_ vector reallocates.
  std::unordered_map<std::string, size_t> index_map_;
};

}  // namespace stablehlo
}  // namespace lczero
