// =============================================================================
// portable_module.h - Wire-Level Structures for Bytecode Emission
// =============================================================================
// These structures represent the IR at the wire level - using semantic objects
// (strings, TypePtr) instead of indices. The NumberingState converts these
// to ir_section structures with computed indices.
//
// This is NOT a full semantic IR. It's the minimal "PortableModule-lite" needed
// to drive bytecode emission without hardcoding indices in tests.
//
// Key concepts:
// - WireOp: Operation with names as strings, types as objects
// - ValueRef: Reference to a value (block arg or op result) within a region
// - Numbering: Isolated regions (module, func) reset value numbering to 0

#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "stablehlo/index_constants.h"
#include "stablehlo/types.h"

namespace lczero {
namespace stablehlo {

// =============================================================================
// Value References
// =============================================================================
// Values within a region are numbered sequentially:
//   - Block args: 0, 1, 2, ...
//   - Op results: continue from args
//
// For isolated regions (module, func), numbering resets to 0.
// ValueRef is just the index within the current region's value space.
using ValueRef = size_t;

// =============================================================================
// WireBlockArg - Block argument at wire level
// =============================================================================
struct WireBlockArg {
  TypePtr type;
  bool hasLoc = false;
  size_t locationIndex = kInvalidIndex;
};

// Forward declarations
struct WireRegion;

// =============================================================================
// Op Properties (typed helpers for known ops)
// =============================================================================
struct CompareProps {
  uint64_t comparisonType = 0;
  uint64_t comparisonDirection = 0;
};

struct GatherProps {
  std::vector<int64_t> offsetDims;
  std::vector<int64_t> collapsedSliceDims;
  std::vector<int64_t> startIndexMap;
  int64_t indexVectorDim = 0;
  std::vector<int64_t> sliceSizes;
  bool indicesAreSorted = false;
};

struct ConstantProps {
  // Raw bytes for dense elements (little-endian).
  std::vector<uint8_t> rawData;
};

struct ReduceProps {
  std::vector<int64_t> dimensions;
};

struct TransposeProps {
  std::vector<int64_t> permutation;
};

struct BroadcastInDimProps {
  std::vector<int64_t> broadcastDimensions;
};

struct ConcatenateProps {
  int64_t dimension = 0;
};

struct SliceProps {
  std::vector<int64_t> startIndices;
  std::vector<int64_t> limitIndices;
  std::vector<int64_t> strides;
};

struct DotGeneralProps {
  std::vector<int64_t> lhsBatchingDimensions;
  std::vector<int64_t> rhsBatchingDimensions;
  std::vector<int64_t> lhsContractingDimensions;
  std::vector<int64_t> rhsContractingDimensions;
  std::vector<uint64_t> precisionConfig;  // 0=DEFAULT, 1=HIGH, 2=HIGHEST
};

struct ConvolutionProps {
  // Scalar i64: dimension indices.
  int64_t inputBatchDimension = 0;
  int64_t inputFeatureDimension = 0;
  int64_t kernelInputFeatureDimension = 0;
  int64_t kernelOutputFeatureDimension = 0;
  int64_t outputBatchDimension = 0;
  int64_t outputFeatureDimension = 0;

  // Scalar i64: group counts.
  int64_t featureGroupCount = 1;
  int64_t batchGroupCount = 1;

  // i64 arrays.
  std::vector<int64_t> windowStrides;
  std::vector<int64_t> lhsDilation;
  std::vector<int64_t> rhsDilation;
  std::vector<int64_t> inputSpatialDimensions;
  std::vector<int64_t> kernelSpatialDimensions;
  std::vector<int64_t> outputSpatialDimensions;

  // padding (encoding resolved from golden scan in H11.4/H11.5).
  std::vector<int64_t> padding;

  // window_reversal (encoding resolved from golden scan in H11.4/H11.5).
  std::vector<bool> windowReversal;

  // 0=DEFAULT, 1=HIGH, 2=HIGHEST.
  std::vector<uint64_t> precisionConfig;
};

// =============================================================================
// WireOp - Operation at wire level
// =============================================================================
struct WireOp {
  // Op identity
  std::string opName;       // e.g. "add_v1", "func_v1", "module"
  std::string dialectName;  // e.g. "vhlo", "builtin"
  bool isRegistered = true;
  
  // Location
  size_t locationIndex = kInvalidIndex;
  
  // Properties payload (opaque bytes)
  // For module/func, this is the properties encoding from Section 8
  std::optional<std::vector<uint8_t>> propertiesPayload;

  // Optional symbol name/visibility (module/func)
  std::optional<std::string> symName;
  std::optional<std::string> symVisibility;

  // Typed properties for known ops (if present, used to auto-generate payloads)
  std::optional<CompareProps> compareProps;
  std::optional<GatherProps> gatherProps;
  std::optional<ConstantProps> constantProps;
  std::optional<ReduceProps> reduceProps;
  std::optional<TransposeProps> transposeProps;
  std::optional<BroadcastInDimProps> broadcastInDimProps;
  std::optional<ConcatenateProps> concatenateProps;
  std::optional<SliceProps> sliceProps;
  std::optional<DotGeneralProps> dotGeneralProps;
  std::optional<ConvolutionProps> convolutionProps;
  
  // Result types (empty for module/func, one for add, etc.)
  std::vector<TypePtr> resultTypes;
  
  // Operand value references (indices in current region's value list)
  // For add: {0, 1} means use values 0 and 1 from the enclosing region
  std::vector<ValueRef> operands;
  
  // Successor block indices (for control flow ops)
  std::vector<size_t> successors;
  
  // Nested regions
  std::vector<WireRegion> regions;
  
  // Whether this op's regions are isolated from above
  // True for module, func - value numbering resets to 0 in their regions
  bool isIsolatedFromAbove = false;
  
  // Debug: return string representation
  std::string toString(int indent = 0) const;
};

// =============================================================================
// WireBlock - Block at wire level
// =============================================================================
struct WireBlock {
  std::vector<WireBlockArg> args;
  std::vector<WireOp> ops;
  
  // Debug: return string representation
  std::string toString(int indent = 0) const;
};

// =============================================================================
// WireRegion - Region at wire level
// =============================================================================
struct WireRegion {
  std::vector<WireBlock> blocks;
  
  // Compute total values in this region (args + results)
  size_t computeNumValues() const {
    size_t count = 0;
    for (const auto& block : blocks) {
      count += block.args.size();
      for (const auto& op : block.ops) {
        count += op.resultTypes.size();
      }
    }
    return count;
  }
  
  // Debug: return string representation
  std::string toString(int indent = 0) const;
};

// =============================================================================
// WireModule - Top-level module at wire level
// =============================================================================
struct WireModule {
  WireOp rootOp;  // Typically a "module" op
  
  // Debug: return string representation
  std::string toString() const;
};

// =============================================================================
// Builder helpers for constructing wire structures
// =============================================================================

// Create a wire block arg
inline WireBlockArg makeWireBlockArg(TypePtr type, size_t locationIndex) {
  return WireBlockArg{std::move(type), locationIndex != kInvalidIndex, locationIndex};
}

// Create a module op (isolated, with properties and one region)
inline WireOp makeWireModuleOp(size_t locationIndex,
                               std::optional<std::vector<uint8_t>> propertiesPayload,
                               WireRegion region,
                               std::optional<std::string> symName = std::nullopt,
                               std::optional<std::string> symVisibility = std::nullopt) {
  WireOp op;
  op.opName = "module";
  op.dialectName = "builtin";
  op.isRegistered = true;
  op.locationIndex = locationIndex;
  op.propertiesPayload = std::move(propertiesPayload);
  op.symName = std::move(symName);
  op.symVisibility = std::move(symVisibility);
  op.isIsolatedFromAbove = true;
  op.regions.push_back(std::move(region));
  return op;
}

// Create a func_v1 op (isolated, with properties and one region)
inline WireOp makeWireFuncOp(size_t locationIndex,
                             std::optional<std::vector<uint8_t>> propertiesPayload,
                             WireRegion region,
                             std::optional<std::string> symName = std::nullopt,
                             std::optional<std::string> symVisibility = std::nullopt) {
  WireOp op;
  op.opName = "func_v1";
  op.dialectName = "vhlo";
  op.isRegistered = true;
  op.locationIndex = locationIndex;
  op.propertiesPayload = std::move(propertiesPayload);
  op.symName = std::move(symName);
  op.symVisibility = std::move(symVisibility);
  op.isIsolatedFromAbove = true;
  op.regions.push_back(std::move(region));
  return op;
}

// Create an op with results and operands (like add_v1)
inline WireOp makeWireOpWithResultsAndOperands(
    const std::string& opName,
    const std::string& dialectName,
    size_t locationIndex,
    std::vector<TypePtr> resultTypes,
    std::vector<ValueRef> operands) {
  WireOp op;
  op.opName = opName;
  op.dialectName = dialectName;
  op.isRegistered = true;
  op.locationIndex = locationIndex;
  op.resultTypes = std::move(resultTypes);
  op.operands = std::move(operands);
  return op;
}

// Create a return_v1 op (operands only)
inline WireOp makeWireReturnOp(size_t locationIndex,
                               std::vector<ValueRef> operands) {
  WireOp op;
  op.opName = "return_v1";
  op.dialectName = "vhlo";
  op.isRegistered = true;
  op.locationIndex = locationIndex;
  op.operands = std::move(operands);
  return op;
}

// Create a compare_v1 op with properties
inline WireOp makeWireCompareOp(size_t locationIndex,
                                std::vector<TypePtr> resultTypes,
                                std::vector<ValueRef> operands,
                                uint64_t comparisonType,
                                uint64_t comparisonDirection) {
  WireOp op = makeWireOpWithResultsAndOperands(
      "compare_v1", "vhlo", locationIndex,
      std::move(resultTypes), std::move(operands));
  op.compareProps = CompareProps{comparisonType, comparisonDirection};
  return op;
}

// Create a gather_v1 op with properties
inline WireOp makeWireGatherOp(size_t locationIndex,
                               std::vector<TypePtr> resultTypes,
                               std::vector<ValueRef> operands,
                               GatherProps props) {
  WireOp op = makeWireOpWithResultsAndOperands(
      "gather_v1", "vhlo", locationIndex,
      std::move(resultTypes), std::move(operands));
  op.gatherProps = std::move(props);
  return op;
}

// Create a constant_v1 op with dense payload
inline WireOp makeWireConstantOp(size_t locationIndex,
                                 TypePtr resultType,
                                 std::vector<uint8_t> rawData) {
  WireOp op;
  op.opName = "constant_v1";
  op.dialectName = "vhlo";
  op.isRegistered = true;
  op.locationIndex = locationIndex;
  op.resultTypes = {std::move(resultType)};
  op.constantProps = ConstantProps{std::move(rawData)};
  return op;
}

// Create a reduce_v1 op with properties and a region
inline WireOp makeWireReduceOp(size_t locationIndex,
                               std::vector<TypePtr> resultTypes,
                               std::vector<ValueRef> operands,
                               ReduceProps props,
                               WireRegion region) {
  WireOp op = makeWireOpWithResultsAndOperands(
      "reduce_v1", "vhlo", locationIndex,
      std::move(resultTypes), std::move(operands));
  op.reduceProps = std::move(props);
  op.regions.push_back(std::move(region));
  // Goldens show reduce_v1 regions are isolated-from-above.
  op.isIsolatedFromAbove = true;
  return op;
}

// Create a transpose_v1 op with properties.
inline WireOp makeWireTransposeOp(size_t locationIndex,
                                  std::vector<TypePtr> resultTypes,
                                  std::vector<ValueRef> operands,
                                  TransposeProps props) {
  WireOp op = makeWireOpWithResultsAndOperands(
      "transpose_v1", "vhlo", locationIndex,
      std::move(resultTypes), std::move(operands));
  op.transposeProps = std::move(props);
  return op;
}

// Create a broadcast_in_dim_v1 op with properties.
inline WireOp makeWireBroadcastInDimOp(size_t locationIndex,
                                       std::vector<TypePtr> resultTypes,
                                       std::vector<ValueRef> operands,
                                       BroadcastInDimProps props) {
  WireOp op = makeWireOpWithResultsAndOperands(
      "broadcast_in_dim_v1", "vhlo", locationIndex,
      std::move(resultTypes), std::move(operands));
  op.broadcastInDimProps = std::move(props);
  return op;
}

// Create a dot_general_v1 op with properties.
inline WireOp makeWireDotGeneralOp(size_t locationIndex,
                                   std::vector<TypePtr> resultTypes,
                                   std::vector<ValueRef> operands,
                                   DotGeneralProps props) {
  WireOp op = makeWireOpWithResultsAndOperands(
      "dot_general_v1", "vhlo", locationIndex,
      std::move(resultTypes), std::move(operands));
  op.dotGeneralProps = std::move(props);
  return op;
}

// Create a convolution_v1 op with properties.
inline WireOp makeWireConvolutionOp(size_t locationIndex,
                                    std::vector<TypePtr> resultTypes,
                                    std::vector<ValueRef> operands,
                                    ConvolutionProps props) {
  WireOp op = makeWireOpWithResultsAndOperands(
      "convolution_v1", "vhlo", locationIndex,
      std::move(resultTypes), std::move(operands));
  op.convolutionProps = std::move(props);
  return op;
}

// Create a slice_v1 op with properties.
inline WireOp makeWireSliceOp(size_t locationIndex,
                              std::vector<TypePtr> resultTypes,
                              std::vector<ValueRef> operands,
                              SliceProps props) {
  WireOp op = makeWireOpWithResultsAndOperands(
      "slice_v1", "vhlo", locationIndex,
      std::move(resultTypes), std::move(operands));
  op.sliceProps = std::move(props);
  return op;
}

// Create a concatenate_v1 op with properties.
inline WireOp makeWireConcatenateOp(size_t locationIndex,
                                    std::vector<TypePtr> resultTypes,
                                    std::vector<ValueRef> operands,
                                    ConcatenateProps props) {
  WireOp op = makeWireOpWithResultsAndOperands(
      "concatenate_v1", "vhlo", locationIndex,
      std::move(resultTypes), std::move(operands));
  op.concatenateProps = std::move(props);
  return op;
}

// Build a reducer body region (isolated) with two block args, a single op,
// and a return. The caller provides operand indices explicitly; no offset logic
// is applied here.
inline WireRegion makeWireReduceBodyRegion(const std::string& opName,
                                           size_t locationIndex,
                                           TypePtr tensorType,
                                           ValueRef lhs,
                                           ValueRef rhs,
                                           ValueRef result) {
  WireBlock block;
  block.args.push_back(makeWireBlockArg(tensorType, locationIndex));
  block.args.push_back(makeWireBlockArg(tensorType, locationIndex));

  block.ops.push_back(makeWireOpWithResultsAndOperands(
      opName, "vhlo", locationIndex,
      {tensorType},
      {lhs, rhs}));

  block.ops.push_back(makeWireReturnOp(locationIndex, {result}));

  WireRegion region;
  region.blocks.push_back(std::move(block));
  return region;
}

// Convenience wrappers for common reducer bodies (add/max).
inline WireRegion makeWireReduceAddBodyRegion(size_t locationIndex,
                                              TypePtr tensorType,
                                              ValueRef lhs,
                                              ValueRef rhs,
                                              ValueRef result) {
  return makeWireReduceBodyRegion("add_v1", locationIndex, std::move(tensorType),
                                  lhs, rhs, result);
}

inline WireRegion makeWireReduceMaxBodyRegion(size_t locationIndex,
                                              TypePtr tensorType,
                                              ValueRef lhs,
                                              ValueRef rhs,
                                              ValueRef result) {
  return makeWireReduceBodyRegion("maximum_v1", locationIndex, std::move(tensorType),
                                  lhs, rhs, result);
}

// M6.1 helpers: reducer body regions using local value numbering (0,1,2).
// These construct a 1-block region with 2 args, one op (add/max), and return.
inline WireRegion makeReduceBodyAdd(TypePtr scalarTensorF32) {
  return makeWireReduceBodyRegion("add_v1", /*locationIndex=*/kInvalidIndex,
                                  scalarTensorF32, /*lhs=*/0, /*rhs=*/1, /*result=*/2);
}

inline WireRegion makeReduceBodyMax(TypePtr scalarTensorF32) {
  return makeWireReduceBodyRegion("maximum_v1", /*locationIndex=*/kInvalidIndex,
                                  scalarTensorF32, /*lhs=*/0, /*rhs=*/1, /*result=*/2);
}

}  // namespace stablehlo
}  // namespace lczero
