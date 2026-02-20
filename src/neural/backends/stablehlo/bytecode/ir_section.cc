// =============================================================================
// ir_section.cc - IR Section Writer Implementation
// =============================================================================

#include "stablehlo/ir_section.h"

#include <iomanip>
#include <sstream>

namespace lczero {
namespace stablehlo {

// =============================================================================
// OpInfo::computeMask
// =============================================================================
uint8_t OpInfo::computeMask() const {
  uint8_t mask = 0;
  
  if (attrDictIndex.has_value()) {
    mask |= OpMask::kHasAttrs;
  }
  if (!resultTypeIndices.empty()) {
    mask |= OpMask::kHasResults;
  }
  if (!operandValueIndices.empty()) {
    mask |= OpMask::kHasOperands;
  }
  if (!successorBlockIndices.empty()) {
    mask |= OpMask::kHasSuccessors;
  }
  if (!regions.empty()) {
    mask |= OpMask::kHasInlineRegions;
  }
  // Note: kHasUseListOrders (0x20) is not set by default
  if (propertiesId.has_value()) {
    mask |= OpMask::kHasProperties;
  }
  
  return mask;
}

// =============================================================================
// IRSectionWriter
// =============================================================================

void IRSectionWriter::setRootOp(OpInfo op) {
  rootOp_ = std::move(op);
}

void IRSectionWriter::write(EncodingEmitter& emitter) const {
  // IR section starts like a "block with no args" containing the root op.
  // emitVarIntWithFlag(numOps=1, hasArgs=false)
  emitter.emitVarIntWithFlag(1, false);
  
  // Write the root operation.
  writeOp(emitter, rootOp_);
}

std::vector<uint8_t> IRSectionWriter::toBytes() const {
  EncodingEmitter emitter;
  write(emitter);
  return emitter.bytes();
}

void IRSectionWriter::writeOp(EncodingEmitter& emitter, const OpInfo& op) const {
  // Field order per MLIR BytecodeWriter.cpp writeOp:
  // 1. opNameIndex: varint
  // 2. mask: u8 (written as 0, then patched)
  // 3. locationIndex: varint
  // 4. if kHasAttrs: attrDictIndex: varint
  // 5. if kHasProperties: propertiesId: varint
  // 6. if kHasResults: numResults: varint, then typeIdx: varint repeated
  // 7. if kHasOperands: numOperands: varint, then valueIdx: varint repeated
  // 8. if kHasSuccessors: numSucc: varint, then blockIdx: varint repeated
  // 9. if kHasUseListOrders: use-list payload
  // 10. if kHasInlineRegions: emitVarIntWithFlag(numRegions, isIsolated), then regions
  
  // 1. opNameIndex
  emitter.emitVarInt(op.opNameIndex);
  
  // 2. mask placeholder (patched after emitting fields)
  // This pattern matches MLIR's BytecodeWriter and supports future kHasUseListOrders.
  size_t maskOffset = emitter.bytes().size();
  emitter.emitByte(0);
  
  // 3. locationIndex
  emitter.emitVarInt(op.locationIndex);
  
  // Compute mask from fields
  uint8_t mask = op.computeMask();
  
  // 4. attrDictIndex (if kHasAttrs)
  if (mask & OpMask::kHasAttrs) {
    emitter.emitVarInt(*op.attrDictIndex);
  }
  
  // 5. propertiesId (if kHasProperties)
  if (mask & OpMask::kHasProperties) {
    emitter.emitVarInt(*op.propertiesId);
  }
  
  // 6. results (if kHasResults)
  if (mask & OpMask::kHasResults) {
    emitter.emitVarInt(op.resultTypeIndices.size());
    for (size_t typeIdx : op.resultTypeIndices) {
      emitter.emitVarInt(typeIdx);
    }
  }
  
  // 7. operands (if kHasOperands)
  if (mask & OpMask::kHasOperands) {
    emitter.emitVarInt(op.operandValueIndices.size());
    for (size_t valIdx : op.operandValueIndices) {
      emitter.emitVarInt(valIdx);
    }
  }
  
  // 8. successors (if kHasSuccessors)
  if (mask & OpMask::kHasSuccessors) {
    emitter.emitVarInt(op.successorBlockIndices.size());
    for (size_t blockIdx : op.successorBlockIndices) {
      emitter.emitVarInt(blockIdx);
    }
  }
  
  // 9. use-list orders (if kHasUseListOrders)
  // Not implemented for minimal golden match
  
  // 10. regions (if kHasInlineRegions)
  if (mask & OpMask::kHasInlineRegions) {
    // Emit numRegions with isIsolatedFromAbove flag
    emitter.emitVarIntWithFlag(op.regions.size(), op.isIsolatedFromAbove);
    
    if (op.isIsolatedFromAbove) {
      // Isolated regions use nested IR sections (lazy-loading)
      writeNestedIRSection(emitter, op.regions);
    } else {
      // Non-isolated regions are written inline
      writeRegionsInline(emitter, op.regions);
    }
  }
  
  // Patch the mask byte
  emitter.bytes_mut()[maskOffset] = mask;
}

void IRSectionWriter::writeRegion(EncodingEmitter& emitter, const RegionInfo& region) const {
  if (region.blocks.empty()) {
    // Empty region: just emit numBlocks=0
    emitter.emitVarInt(0);
    return;
  }
  
  // Non-empty region:
  // 1. numBlocks
  emitter.emitVarInt(region.blocks.size());
  
  // 2. numValues (auto-computed from region structure)
  emitter.emitVarInt(region.computeNumValues());
  
  // 3. Write each block
  for (const auto& block : region.blocks) {
    writeBlock(emitter, block);
  }
}

void IRSectionWriter::writeBlock(EncodingEmitter& emitter, const BlockInfo& block) const {
  bool hasArgs = !block.args.empty();
  
  // 1. emitVarIntWithFlag(numOps, hasArgs)
  emitter.emitVarIntWithFlag(block.ops.size(), hasArgs);
  
  // 2. If hasArgs, emit args
  if (hasArgs) {
    // numArgs
    emitter.emitVarInt(block.args.size());
    
    // For each arg (bytecode v6 encoding):
    // emitVarIntWithFlag(typeIndex, hasLoc)
    // if hasLoc: emitVarInt(locIndex)
    for (const auto& arg : block.args) {
      emitter.emitVarIntWithFlag(arg.typeIndex, arg.hasLoc);
      if (arg.hasLoc) {
        emitter.emitVarInt(arg.locationIndex);
      }
    }
    
    // Block-arg use-list placeholder byte (bytecode >= kUseListOrdering)
    // In our goldens, this stays 0x00 and no additional bytes follow.
    emitter.emitByte(0x00);
  }
  
  // 3. Write each op
  for (const auto& op : block.ops) {
    writeOp(emitter, op);
  }
}

void IRSectionWriter::writeNestedIRSection(
    EncodingEmitter& emitter,
    const std::vector<RegionInfo>& regions) const {
  
  // Build the nested section payload first (to know the length)
  EncodingEmitter nestedEmitter;
  for (const auto& region : regions) {
    writeRegion(nestedEmitter, region);
  }
  
  // Emit nested section header: [sectionId=4][length]
  emitter.emitByte(4);  // Section ID for IR
  emitter.emitVarInt(nestedEmitter.bytes().size());
  
  // Emit nested section payload
  emitter.emitBytes(nestedEmitter.bytes());
}

void IRSectionWriter::writeRegionsInline(
    EncodingEmitter& emitter,
    const std::vector<RegionInfo>& regions) const {
  
  for (const auto& region : regions) {
    writeRegion(emitter, region);
  }
}

// =============================================================================
// Debug dump implementations
// =============================================================================

// Helper to create indentation string
static std::string indent(int level) {
  return std::string(level * 2, ' ');
}

// Helper to format a vector of size_t as [0, 1, 2]
static std::string formatIndices(const std::vector<size_t>& indices) {
  std::ostringstream ss;
  ss << "[";
  for (size_t i = 0; i < indices.size(); ++i) {
    if (i > 0) ss << ", ";
    ss << indices[i];
  }
  ss << "]";
  return ss.str();
}

std::string BlockArgInfo::toString() const {
  std::ostringstream ss;
  ss << "BlockArg{typeIdx=" << typeIndex;
  if (hasLoc) {
    ss << ", locIdx=" << locationIndex;
  } else {
    ss << ", noLoc";
  }
  ss << "}";
  return ss.str();
}

std::string OpInfo::toString(int level) const {
  std::ostringstream ss;
  std::string ind = indent(level);
  
  ss << ind << "Op {\n";
  ss << ind << "  opNameIndex: " << opNameIndex << "\n";
  ss << ind << "  locationIndex: " << locationIndex << "\n";
  ss << ind << "  mask: 0x" << std::hex << std::setfill('0') << std::setw(2) 
     << static_cast<int>(computeMask()) << std::dec << "\n";
  
  if (attrDictIndex.has_value()) {
    ss << ind << "  attrDictIndex: " << *attrDictIndex << "\n";
  }
  if (propertiesId.has_value()) {
    ss << ind << "  propertiesId: " << *propertiesId << "\n";
  }
  if (!resultTypeIndices.empty()) {
    ss << ind << "  resultTypeIndices: " << formatIndices(resultTypeIndices) << "\n";
  }
  if (!operandValueIndices.empty()) {
    ss << ind << "  operandValueIndices: " << formatIndices(operandValueIndices) << "\n";
  }
  if (!successorBlockIndices.empty()) {
    ss << ind << "  successorBlockIndices: " << formatIndices(successorBlockIndices) << "\n";
  }
  if (!regions.empty()) {
    ss << ind << "  isIsolatedFromAbove: " << (isIsolatedFromAbove ? "true" : "false") << "\n";
    ss << ind << "  regions: [\n";
    for (size_t i = 0; i < regions.size(); ++i) {
      ss << ind << "    Region #" << i << " {\n";
      ss << regions[i].toString(level + 3);
      ss << ind << "    }\n";
    }
    ss << ind << "  ]\n";
  }
  ss << ind << "}";
  return ss.str();
}

std::string BlockInfo::toString(int level) const {
  std::ostringstream ss;
  std::string ind = indent(level);
  
  if (!args.empty()) {
    ss << ind << "args: [\n";
    for (size_t i = 0; i < args.size(); ++i) {
      ss << ind << "  value#" << i << " = " << args[i].toString() << "\n";
    }
    ss << ind << "]\n";
  }
  
  ss << ind << "ops: [\n";
  size_t valueOffset = args.size();  // Results start after block args
  for (size_t i = 0; i < ops.size(); ++i) {
    ss << ops[i].toString(level + 1) << "\n";
    // Show result value IDs
    if (!ops[i].resultTypeIndices.empty()) {
      ss << ind << "  // results: ";
      for (size_t j = 0; j < ops[i].resultTypeIndices.size(); ++j) {
        if (j > 0) ss << ", ";
        ss << "value#" << (valueOffset + j);
      }
      ss << "\n";
      valueOffset += ops[i].resultTypeIndices.size();
    }
  }
  ss << ind << "]\n";
  
  return ss.str();
}

std::string RegionInfo::toString(int level) const {
  std::ostringstream ss;
  std::string ind = indent(level);
  
  ss << ind << "numBlocks: " << blocks.size() << "\n";
  ss << ind << "numValues: " << computeNumValues() << "\n";
  
  for (size_t i = 0; i < blocks.size(); ++i) {
    ss << ind << "Block #" << i << " {\n";
    ss << blocks[i].toString(level + 1);
    ss << ind << "}\n";
  }
  
  return ss.str();
}

std::string IRSectionWriter::dumpIndices() const {
  std::ostringstream ss;
  ss << "=== IR Section Debug Dump ===\n";
  ss << "Root operation:\n";
  ss << rootOp_.toString(0) << "\n";
  return ss.str();
}

}  // namespace stablehlo
}  // namespace lczero
