// =============================================================================
// encoding.cc - EncodingEmitter Implementation
// =============================================================================
// Varint algorithm copied from MLIR's BytecodeWriter.cpp

#include "stablehlo/encoding.h"

#include <cassert>
#include <cstdint>

namespace lczero {
namespace stablehlo {

void EncodingEmitter::emitByte(uint8_t b) { buf_.push_back(b); }

void EncodingEmitter::emitBytes(const uint8_t* data, size_t n) {
  buf_.insert(buf_.end(), data, data + n);
}

void EncodingEmitter::emitBytes(const std::vector<uint8_t>& data) {
  emitBytes(data.data(), data.size());
}

void EncodingEmitter::emitU64LE(uint64_t v) {
  for (int i = 0; i < 8; ++i) {
    emitByte(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
  }
}

void EncodingEmitter::alignTo(unsigned alignment) {
  if (alignment < 2) return;
  assert((alignment & (alignment - 1)) == 0 && "alignment must be power of two");

  size_t curOffset = buf_.size();
  size_t paddingSize = ((curOffset + alignment - 1) & ~(alignment - 1)) - curOffset;
  while (paddingSize--) emitByte(kAlignmentByte);

  if (alignment > requiredAlignment_) requiredAlignment_ = alignment;
}

void EncodingEmitter::emitVarInt(uint64_t value) {
  // Single-byte case: value fits in 7 bits.
  if ((value >> 7) == 0) {
    emitByte(static_cast<uint8_t>((value << 1) | 0x1));
    return;
  }
  emitMultiByteVarInt(value);
}

void EncodingEmitter::emitMultiByteVarInt(uint64_t value) {
  // IMPORTANT: This logic is copied from MLIR's BytecodeWriter.cpp
  // EncodingEmitter::emitMultiByteVarInt, adapted to avoid LLVM types.

  // Compute the number of bytes needed to encode the value. Each byte can hold
  // up to 7-bits of data. We only check up to the number of bits we can encode
  // in the first byte (8).
  uint64_t it = value >> 7;
  for (size_t numBytes = 2; numBytes < 9; ++numBytes) {
    if ((it >>= 7) == 0) {
      uint64_t encodedValue = (value << 1) | 0x1;
      encodedValue <<= (numBytes - 1);

      // Emit the low numBytes of encodedValue in little-endian order.
      for (size_t i = 0; i < numBytes; ++i) {
        emitByte(static_cast<uint8_t>((encodedValue >> (8 * i)) & 0xFF));
      }
      return;
    }
  }

  // If the value is too large to encode in a single byte, emit a special all
  // zero marker byte and splat the value directly.
  emitByte(0);
  emitU64LE(value);
}

void EncodingEmitter::emitVarIntWithFlag(uint64_t value, bool flag) {
  uint64_t packed = (value << 1) | (flag ? 1ull : 0ull);
  emitVarInt(packed);
}

void EncodingEmitter::emitSignedVarInt(int64_t value) {
  // ZigZag encode: map signed to unsigned so small magnitudes stay small.
  // zigzag(x) = (x << 1) ^ (x >> 63)
  uint64_t zigzag =
      (static_cast<uint64_t>(value) << 1) ^ static_cast<uint64_t>(value < 0 ? ~0ull : 0ull);
  emitVarInt(zigzag);
}

void EncodingEmitter::emitNulTerminatedString(std::string_view s) {
  for (char c : s) {
    // NUL is reserved as terminator; emitting embedded NUL is almost always a bug.
    assert(c != '\0');
    emitByte(static_cast<uint8_t>(c));
  }
  emitByte(0);
}

}  // namespace stablehlo
}  // namespace lczero
