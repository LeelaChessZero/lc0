// =============================================================================
// ir_section.h - IR Section Writer (Section ID = 4)
// =============================================================================
// Implements Section 4 (IR) payload encoding for MLIR bytecode v6.
//
// Key concepts:
// - Operations encode: opNameIndex, mask, locationIndex, then optional fields
// - Regions contain blocks; blocks contain args and ops
// - Isolated regions use nested IR sections (lazy-loading)
// - Value numbering resets to 0 for each isolated region
//
// OpEncodingMask bits:
//   0x01 = kHasAttrs
//   0x02 = kHasResults
//   0x04 = kHasOperands
//   0x08 = kHasSuccessors
//   0x10 = kHasInlineRegions
//   0x20 = kHasUseListOrders
//   0x40 = kHasProperties

#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "stablehlo/encoding.h"
#include "stablehlo/index_constants.h"

namespace lczero {
namespace stablehlo {

// =============================================================================
// OpEncodingMask bits (from MLIR Encoding.h)
// =============================================================================
namespace OpMask {
  inline constexpr uint8_t kHasAttrs         = 0x01;
  inline constexpr uint8_t kHasResults       = 0x02;
  inline constexpr uint8_t kHasOperands      = 0x04;
  inline constexpr uint8_t kHasSuccessors    = 0x08;
  inline constexpr uint8_t kHasInlineRegions = 0x10;
  inline constexpr uint8_t kHasUseListOrders = 0x20;
  inline constexpr uint8_t kHasProperties    = 0x40;
}

// =============================================================================
// Bytecode version constants (from MLIR Encoding.h)
// =============================================================================
namespace BytecodeVersion {
  inline constexpr uint64_t kDialectVersioning = 1;
  inline constexpr uint64_t kLazyLoading = 2;
  inline constexpr uint64_t kUseListOrdering = 3;
  inline constexpr uint64_t kElideUnknownBlockArgLocation = 4;
  inline constexpr uint64_t kNativePropertiesEncoding = 5;
  inline constexpr uint64_t kCurrent = 6;
}

// =============================================================================
// Block argument info
// =============================================================================
struct BlockArgInfo {
  size_t typeIndex = kInvalidIndex;
  bool hasLoc = false;
  size_t locationIndex = kInvalidIndex;
  
  // Debug dump: returns string representation of this block arg.
  std::string toString() const;
};

// =============================================================================
// Forward declarations
// =============================================================================
struct RegionInfo;

// =============================================================================
// Operation info - describes an operation for encoding
// =============================================================================
struct OpInfo {
  size_t opNameIndex = kInvalidIndex;
  size_t locationIndex = kInvalidIndex;
  
  // Optional: attribute dictionary index (mask bit 0x01)
  std::optional<size_t> attrDictIndex;
  
  // Optional: properties ID (mask bit 0x40)
  std::optional<size_t> propertiesId;
  
  // Result types (mask bit 0x02 if non-empty)
  std::vector<size_t> resultTypeIndices;
  
  // Operand value indices (mask bit 0x04 if non-empty)
  std::vector<size_t> operandValueIndices;
  
  // Successor block indices (mask bit 0x08 if non-empty)
  std::vector<size_t> successorBlockIndices;
  
  // Regions (mask bit 0x10 if non-empty)
  std::vector<RegionInfo> regions;
  
  // Whether this op's regions are isolated from above
  // (determines nested section vs inline region encoding)
  bool isIsolatedFromAbove = false;
  
  // Compute the encoding mask from the fields
  uint8_t computeMask() const;
  
  // Debug dump: returns string representation of this op.
  // indent controls the nesting level for pretty printing.
  std::string toString(int indent = 0) const;
};

// =============================================================================
// Block info - describes a block for encoding
// =============================================================================
struct BlockInfo {
  std::vector<BlockArgInfo> args;
  std::vector<OpInfo> ops;
  
  // Debug dump: returns string representation of this block.
  std::string toString(int indent = 0) const;
};

// =============================================================================
// Region info - describes a region for encoding
// =============================================================================
struct RegionInfo {
  std::vector<BlockInfo> blocks;
  
  // Auto-compute numValues from the region structure.
  // Values = block args + op results (in all blocks).
  size_t computeNumValues() const {
    size_t count = 0;
    for (const auto& block : blocks) {
      count += block.args.size();
      for (const auto& op : block.ops) {
        count += op.resultTypeIndices.size();
      }
    }
    return count;
  }
  
  // Debug dump: returns string representation of this region.
  std::string toString(int indent = 0) const;
};

// =============================================================================
// IRSectionWriter - Writes Section 4 (IR) payload
// =============================================================================
class IRSectionWriter {
 public:
  IRSectionWriter() = default;
  
  // Set the root operation (typically "module")
  void setRootOp(OpInfo op);
  
  // Write the IR section payload to an emitter.
  // The top-level is encoded as a "block with no args" containing the root op.
  void write(EncodingEmitter& emitter) const;
  
  // Get IR section payload bytes.
  std::vector<uint8_t> toBytes() const;
  
  // Debug dump: returns string representation of the entire IR structure.
  // Call this BEFORE toBytes() to see the computed indices.
  std::string dumpIndices() const;
  
 private:
  // Write an operation.
  void writeOp(EncodingEmitter& emitter, const OpInfo& op) const;
  
  // Write a region (numBlocks, numValues, then blocks).
  void writeRegion(EncodingEmitter& emitter, const RegionInfo& region) const;
  
  // Write a block (numOps+hasArgs flag, args if any, then ops).
  void writeBlock(EncodingEmitter& emitter, const BlockInfo& block) const;
  
  // Write regions as a nested IR section (for isolated regions).
  void writeNestedIRSection(EncodingEmitter& emitter, 
                            const std::vector<RegionInfo>& regions) const;
  
  // Write multiple regions inline (for non-isolated regions).
  void writeRegionsInline(EncodingEmitter& emitter,
                          const std::vector<RegionInfo>& regions) const;
  
  OpInfo rootOp_;
};

// =============================================================================
// Builder helpers for constructing IR structures
// =============================================================================

// Create a block argument with location.
inline BlockArgInfo makeBlockArg(size_t typeIndex, size_t locationIndex) {
  return BlockArgInfo{typeIndex, true, locationIndex};
}

// Create a block argument without location (hasLoc=false).
inline BlockArgInfo makeBlockArgNoLoc(size_t typeIndex) {
  return BlockArgInfo{typeIndex, false, kInvalidIndex};
}

// Create an operation with properties and isolated regions (like module/func).
inline OpInfo makeOpWithPropertiesAndRegions(
    size_t opNameIndex,
    size_t locationIndex,
    size_t propertiesId,
    std::vector<RegionInfo> regions,
    bool isIsolated = true) {
  OpInfo op;
  op.opNameIndex = opNameIndex;
  op.locationIndex = locationIndex;
  op.propertiesId = propertiesId;
  op.regions = std::move(regions);
  op.isIsolatedFromAbove = isIsolated;
  return op;
}

// Create an operation with results and operands (like add_v1).
inline OpInfo makeOpWithResultsAndOperands(
    size_t opNameIndex,
    size_t locationIndex,
    std::vector<size_t> resultTypeIndices,
    std::vector<size_t> operandValueIndices) {
  OpInfo op;
  op.opNameIndex = opNameIndex;
  op.locationIndex = locationIndex;
  op.resultTypeIndices = std::move(resultTypeIndices);
  op.operandValueIndices = std::move(operandValueIndices);
  return op;
}

// Create an operation with only operands (like return_v1).
inline OpInfo makeOpWithOperands(
    size_t opNameIndex,
    size_t locationIndex,
    std::vector<size_t> operandValueIndices) {
  OpInfo op;
  op.opNameIndex = opNameIndex;
  op.locationIndex = locationIndex;
  op.operandValueIndices = std::move(operandValueIndices);
  return op;
}

}  // namespace stablehlo
}  // namespace lczero
