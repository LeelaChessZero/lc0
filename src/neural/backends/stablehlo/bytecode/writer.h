// =============================================================================
// writer.h - BytecodeWriter for emitting MLIR bytecode
// =============================================================================

#pragma once
#include <cstdint>
#include <functional>
#include <span>
#include <string_view>
#include <vector>

#include "stablehlo/encoding.h"

namespace lczero {
namespace stablehlo {

class BytecodeWriter {
 public:
  // Writes: magic + version varint + producer\0
  void writeHeader(uint64_t version, std::string_view producer);

  // Convenience: use default version 6
  void writeHeader(std::string_view producer) {
    writeHeader(kDefaultBytecodeVersion, producer);
  }

  // Writes a section envelope: [id][len varint][payload]
  void writeSection(uint8_t id, std::span<const uint8_t> payload);

  // Convenience: build payload with a callback into a temporary emitter.
  void writeSection(uint8_t id, const std::function<void(EncodingEmitter&)>& emitPayload);

  const std::vector<uint8_t>& bytes() const { return emitter_.bytes(); }
  std::vector<uint8_t>& bytes_mut() { return emitter_.bytes_mut(); }

 private:
  void writeSectionInternal(uint8_t id, std::span<const uint8_t> payload,
                            unsigned alignment);

  EncodingEmitter emitter_;
};

}  // namespace stablehlo
}  // namespace lczero
