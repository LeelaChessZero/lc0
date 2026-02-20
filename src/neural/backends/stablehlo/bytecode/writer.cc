// =============================================================================
// writer.cc - BytecodeWriter Implementation
// =============================================================================

#include "stablehlo/writer.h"

#include <cassert>

namespace lczero {
namespace stablehlo {

void BytecodeWriter::writeHeader(uint64_t version, std::string_view producer) {
  emitter_.emitByte(kMagic[0]);
  emitter_.emitByte(kMagic[1]);
  emitter_.emitByte(kMagic[2]);
  emitter_.emitByte(kMagic[3]);

  emitter_.emitVarInt(version);
  emitter_.emitNulTerminatedString(producer);
}

void BytecodeWriter::writeSectionInternal(uint8_t id, std::span<const uint8_t> payload,
                                          unsigned alignment) {
  size_t codeOffset = emitter_.bytes().size();
  emitter_.emitByte(id);
  emitter_.emitVarInt(static_cast<uint64_t>(payload.size()));

  bool didAlign = false;
  if (alignment > 1) {
    assert((alignment & (alignment - 1)) == 0 && "alignment must be power of two");
    size_t curOffset = emitter_.bytes().size();
    if (curOffset & (alignment - 1)) {
      emitter_.emitVarInt(alignment);
      emitter_.alignTo(alignment);
      didAlign = true;
    }
  }

  if (didAlign) {
    emitter_.bytes_mut()[codeOffset] |= 0x80;
  }

  if (!payload.empty()) emitter_.emitBytes(payload.data(), payload.size());
}

void BytecodeWriter::writeSection(uint8_t id, std::span<const uint8_t> payload) {
  writeSectionInternal(id, payload, /*alignment=*/1);
}

void BytecodeWriter::writeSection(uint8_t id, const std::function<void(EncodingEmitter&)>& emitPayload) {
  EncodingEmitter tmp;
  emitPayload(tmp);
  writeSectionInternal(
      id,
      std::span<const uint8_t>(tmp.bytes().data(), tmp.bytes().size()),
      tmp.requiredAlignment());
}

}  // namespace stablehlo
}  // namespace lczero
