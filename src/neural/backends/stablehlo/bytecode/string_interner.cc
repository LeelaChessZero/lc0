// =============================================================================
// string_interner.cc - StringInterner Implementation
// =============================================================================

#include "stablehlo/string_interner.h"
#include "stablehlo/index_constants.h"

#include <algorithm>
#include <cassert>

namespace lczero {
namespace stablehlo {

size_t StringInterner::intern(std::string_view s) {
  // Convert to string for map lookup (string_view can be converted implicitly).
  std::string key(s);
  
  // Check if already interned.
  auto it = index_map_.find(key);
  if (it != index_map_.end()) {
    return it->second;
  }

  // Add new string.
  size_t index = strings_.size();
  strings_.emplace_back(s);
  index_map_[std::move(key)] = index;
  
  return index;
}

size_t StringInterner::lookup(std::string_view s) const {
  std::string key(s);
  auto it = index_map_.find(key);
  if (it != index_map_.end()) {
    return it->second;
  }
  return kInvalidIndex;
}

std::string_view StringInterner::get(size_t index) const {
  assert(index < strings_.size());
  return strings_[index];
}

void StringInterner::write(EncodingEmitter& emitter) const {
  // String section format:
  //   numStrings: varint
  //   sizes[numStrings]: varint[] (in REVERSE order, size+1 for NUL)
  //   data: null-terminated strings (in FORWARD order)

  // 1. Emit count.
  emitter.emitVarInt(strings_.size());

  // 2. Emit sizes in REVERSE order.
  // CRITICAL: This is the "reverse-size quirk" from BytecodeWriter.cpp.
  // Each size is string.size() + 1 to account for the null terminator.
  for (auto it = strings_.rbegin(); it != strings_.rend(); ++it) {
    emitter.emitVarInt(it->size() + 1);  // +1 for NUL
  }

  // 3. Emit string data in FORWARD order (null-terminated).
  for (const auto& s : strings_) {
    emitter.emitNulTerminatedString(s);
  }
}

std::vector<uint8_t> StringInterner::toBytes() const {
  EncodingEmitter emitter;
  write(emitter);
  return emitter.bytes();
}

}  // namespace stablehlo
}  // namespace lczero
