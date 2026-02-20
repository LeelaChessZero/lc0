// =============================================================================
// encoding.h - MLIR Bytecode Constants and Encoding Emitter
// =============================================================================
// Section IDs from: llvm-project/mlir/lib/Bytecode/Encoding.h

#pragma once
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace lczero {
namespace stablehlo {

// Magic bytes: "ML\xEFR" (4 bytes)
inline constexpr uint8_t kMagic[4] = {0x4D, 0x4C, 0xEF, 0x52};

// MLIR bytecode version (from Encoding.h: kVersion = 6)
inline constexpr uint64_t kDefaultBytecodeVersion = 6;

// Alignment byte marker (used for padding)
inline constexpr uint8_t kAlignmentByte = 0xCB;

// =============================================================================
// Section IDs - CORRECT values from MLIR Encoding.h
// =============================================================================
// CRITICAL: These MUST match MLIR's Encoding.h exactly!
// Verified against: llvm-project/mlir/lib/Bytecode/Encoding.h
namespace Section {
  inline constexpr uint8_t kString          = 0;  // String section
  inline constexpr uint8_t kDialect         = 1;  // Dialect section  
  inline constexpr uint8_t kAttrType        = 2;  // Attribute/Type section
  inline constexpr uint8_t kAttrTypeOffset  = 3;  // Attribute/Type offset section
  inline constexpr uint8_t kIR              = 4;  // IR section
  inline constexpr uint8_t kResource        = 5;  // Resource section
  inline constexpr uint8_t kResourceOffset  = 6;  // Resource offset section
  inline constexpr uint8_t kDialectVersions = 7;  // Dialect versions section
  inline constexpr uint8_t kProperties      = 8;  // Properties section
}  // namespace Section

// Section ID to name (for scanner output)
inline const char* getSectionName(uint8_t id) {
  switch (id) {
    case Section::kString:          return "String";
    case Section::kDialect:         return "Dialect";
    case Section::kAttrType:        return "AttrType";
    case Section::kAttrTypeOffset:  return "AttrTypeOffset";
    case Section::kIR:              return "IR";
    case Section::kResource:        return "Resource";
    case Section::kResourceOffset:  return "ResourceOffset";
    case Section::kDialectVersions: return "DialectVersions";
    case Section::kProperties:      return "Properties";
    default:                        return "Unknown";
  }
}

// =============================================================================
// EncodingEmitter - Emits MLIR bytecode primitives
// =============================================================================
class EncodingEmitter {
 public:
  void emitByte(uint8_t b);
  void emitBytes(const uint8_t* data, size_t n);
  void emitBytes(const std::vector<uint8_t>& data);

  void emitU64LE(uint64_t v);

  // Align the emitter to the given alignment, padding with kAlignmentByte.
  void alignTo(unsigned alignment);

  // MLIR bytecode varint encoding (NOT LEB128).
  void emitVarInt(uint64_t value);

  // Encodes (value, flag) into one varint:
  //   packed = (value << 1) | (flag ? 1 : 0)
  //   emitVarInt(packed)
  void emitVarIntWithFlag(uint64_t value, bool flag);

  // ZigZag-encodes signed integer, then emitVarInt().
  void emitSignedVarInt(int64_t value);

  // Emits bytes of s followed by '\0'.
  void emitNulTerminatedString(std::string_view s);

  const std::vector<uint8_t>& bytes() const { return buf_; }
  std::vector<uint8_t>& bytes_mut() { return buf_; }
  void clear() { buf_.clear(); requiredAlignment_ = 1; }
  size_t size() const { return buf_.size(); }
  unsigned requiredAlignment() const { return requiredAlignment_; }

 private:
  void emitMultiByteVarInt(uint64_t value);

  std::vector<uint8_t> buf_;
  unsigned requiredAlignment_ = 1;
};

}  // namespace stablehlo
}  // namespace lczero
