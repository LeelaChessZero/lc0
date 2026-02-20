// =============================================================================
// numbering.h - IR Numbering for Bytecode Emission
// =============================================================================
// This implements the numbering pass that bridges WireModule (semantic objects)
// to IRSectionWriter (indices).
//
// Key rules from MLIR's IRNumbering.cpp:
// 1. Value numbering is per-REGION, not per-block
//    - All blocks in a region share continuous value IDs
// 2. Isolated regions (module, func) reset value numbering to 0
//    - This is determined by isIsolatedFromAbove on the op
// 3. Within a region: block args numbered first, then op results, block by block
//
// Example (add.mlirbc):
//   Module region: numValues=0 (no values at module level)
//   Func region: numValues=3 (arg0=0, arg1=1, add_result=2)
//
// Note: In real MLIR, isolation is computed based on cross-region value uses.
// For lc0, we simplify: module and func are always isolated.

#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "stablehlo/index_constants.h"
#include "stablehlo/portable_module.h"
#include "stablehlo/ir_section.h"
#include "stablehlo/dialect_section.h"
#include "stablehlo/attr_type_section.h"
#include "stablehlo/string_interner.h"
#include "stablehlo/type_interner.h"
#include "stablehlo/attr_interner.h"
#include "stablehlo/attrs.h"

namespace lczero {
namespace stablehlo {

// =============================================================================
// NumberingState - Converts WireModule to ir_section structures
// =============================================================================
class NumberingState {
 public:
  NumberingState() = default;
  
  // Number the entire wire module.
  // This populates interners and computes all indices.
  void number(const WireModule& module);
  
  // Convert a numbered WireOp to OpInfo for IRSectionWriter.
  // Must be called after number().
  OpInfo toOpInfo(const WireOp& op) const;
  
  // Access interners for section emission
  StringInterner& strings() { return strings_; }
  const StringInterner& strings() const { return strings_; }
  
  DialectSectionWriter& dialects() { return dialects_; }
  const DialectSectionWriter& dialects() const { return dialects_; }
  
  AttrTypeSectionWriter& attrTypes() { return attrTypes_; }
  const AttrTypeSectionWriter& attrTypes() const { return attrTypes_; }
  
  // Properties table (Section 8)
  const std::vector<std::vector<uint8_t>>& propertiesTable() const {
    return propertiesTable_;
  }
  
  // Get the UnknownLoc attr index (valid after number() called)
  size_t unknownLocAttrIndex() const { return unknownLocAttrIndex_; }
  
  // Build func_v1 properties blob using attr indices.
  // funcTypeIndex: index of the FunctionType in attr/type section
  // funcName: the function name (e.g., "main")
  std::vector<uint8_t> buildFuncV1Props(const TypePtr& funcType,
                                        const std::string& funcName,
                                        const std::string& symVisibility = "");
  
  // Build module properties blob.
  std::vector<uint8_t> buildModuleProps(const WireOp& op);

  // Build compare_v1 properties blob (alphabetical attr order).
  std::vector<uint8_t> buildCompareV1Props(const CompareProps& props);

  // Build gather_v1 properties blob (alphabetical attr order).
  std::vector<uint8_t> buildGatherV1Props(const GatherProps& props);

  // Build constant_v1 properties blob (value attr).
  std::vector<uint8_t> buildConstantV1Props(const WireOp& op,
                                            const ConstantProps& props);

  // Build reduce_v1 properties blob (dimensions attr).
  std::vector<uint8_t> buildReduceV1Props(const ReduceProps& props);

  // Build transpose_v1 properties blob (permutation attr).
  std::vector<uint8_t> buildTransposeV1Props(const TransposeProps& props);

  // Build broadcast_in_dim_v1 properties blob (broadcast_dimensions attr).
  std::vector<uint8_t> buildBroadcastInDimV1Props(
      const BroadcastInDimProps& props);

  // Build concatenate_v1 properties blob (dimension attr).
  std::vector<uint8_t> buildConcatenateV1Props(
      const ConcatenateProps& props);

  // Build slice_v1 properties blob (limit_indices, start_indices, strides).
  std::vector<uint8_t> buildSliceV1Props(const SliceProps& props);

  // Build dot_general_v1 properties blob
  // (lhs_batching_dimensions, lhs_contracting_dimensions, precision_config,
  //  rhs_batching_dimensions, rhs_contracting_dimensions).
  std::vector<uint8_t> buildDotGeneralV1Props(const DotGeneralProps& props);

  // Build convolution_v1 properties blob.
  std::vector<uint8_t> buildConvolutionV1Props(const ConvolutionProps& props);
  
  // Debug: dump the numbering state
  std::string dump() const;
  
 private:
  // Number operations, blocks, and regions
  void numberOp(const WireOp& op);
  void numberBlock(const WireBlock& block);
  void numberRegion(const WireRegion& region, bool isIsolated);
  
  // Convert helpers (produce ir_section structures)
  OpInfo toOpInfoImpl(const WireOp& op, size_t& cursor) const;
  BlockInfo toBlockInfo(const WireBlock& block, size_t& cursor) const;
  RegionInfo toRegionInfo(const WireRegion& region, size_t& cursor) const;
  
  // Ensure builtin UnknownLoc attr is registered (called once during number())
  void ensureUnknownLocAttr();
  
  // Ensure builtin dialect is registered
  void ensureBuiltinDialect();
  
  // Intern a VHLO attribute and return its attr index
  size_t internVhloAttr(const AttrPtr& attr);

  // Encode a VHLO attribute (uses current interner state for indices)
  std::vector<uint8_t> encodeVhloAttr(const Attr& attr);

  // Pre-number types in a deterministic order (e.g. refcount-sorted)
  // before walking the IR. This stabilizes type indices for byte-for-byte
  // matching with pinned MLIR output.
  void preNumberTypes(const WireModule& module);

  // Finalize type payloads after numbering (all type indices are fixed).
  void finalizeTypePayloads();

  // Pre-seed op name ordering (refcount-stable) and pre-intern dialect/op
  // names to stabilize string indices.
  void preNumberOpNames(const WireModule& module);
  void preInternDialectAndOpNames();

  // Pre-collect VHLO attribute usage in MLIR-like first-seen/refcount order.
  void preCollectAttrs(const WireModule& module);

  // Pre-intern builtin attrs that must exist before global attr finalization.
  void preInternBuiltinAttrs(const WireModule& module);

  // Finalize global VHLO attr ordering and internal->serialized index mapping.
  void finalizeAttrOrder();
  
  // Intern a builtin attr and return its attr index
  size_t internBuiltinAttr(const std::vector<uint8_t>& payload);

  // Helpers for i64-backed attrs
  size_t internI64IntegerAttr(int64_t value);
  size_t internI64TensorAttr(const std::vector<int64_t>& values);
  size_t internI64TensorAttrWithShape(const std::vector<int64_t>& shape,
                                      const std::vector<int64_t>& values);
  size_t internBoolTensorAttrWithShape(const std::vector<int64_t>& shape,
                                       const std::vector<bool>& values);
  
  // Get or assign an opName index
  size_t getOpNameIndex(const std::string& dialectName,
                        const std::string& opName,
                        bool isRegistered);
  
  // Get or assign a type index
  size_t getTypeIndex(const TypePtr& type);

  // Encode payload bytes for a type using current numbering state.
  std::vector<uint8_t> encodeTypePayload(const TypePtr& type);
  
  // Lookup an existing type index by structural equality.
  // Returns kInvalidIndex if not found.
  size_t lookupTypeIndex(const TypePtr& type) const;
  
  // Get or assign a properties ID
  size_t getPropertiesId(const std::vector<uint8_t>& payload);
  
  // Interners
  StringInterner strings_;
  DialectSectionWriter dialects_;
  AttrTypeSectionWriter attrTypes_;
  
  // Properties table (Section 8): propertiesId → payload bytes
  std::vector<std::vector<uint8_t>> propertiesTable_;
  
  // Dedup map for properties: payload hash → propertiesId
  std::unordered_map<std::string, size_t> propertiesDedup_;

  // OpName tracking: "dialect.opName" → opNameIndex
  std::unordered_map<std::string, size_t> opNameIndices_;

  // Builtin attr dedup: payload → attr index
  std::unordered_map<std::string, size_t> builtinAttrDedup_;
  
  // Type/Attr interners (single source of truth)
  TypeInterner typeInterner_;
  AttrInterner attrInterner_;
  std::vector<size_t> vhloAttrIndexToGlobal_;
  
  // Current value ID counter (reset per isolated region)
  size_t nextValueID_ = 0;
  
  // UnknownLoc attr index (set during number())
  size_t unknownLocAttrIndex_ = kInvalidIndex;
  
  // Builtin dialect ID (0)
  size_t builtinDialectId_ = kInvalidIndex;

  // If true, type indices are frozen and must already exist.
  bool typesFrozen_ = false;

  // If true, VHLO attrs are frozen and must already exist in finalized order.
  bool attrsFrozen_ = false;

  struct AttrUsageEntry {
    size_t refCount = 0;
    size_t firstSeen = 0;
    AttrPtr attr;
  };

  // Pre-collected VHLO attrs before final ordering.
  std::vector<AttrUsageEntry> preCollectedAttrs_;

  // Properties ids in op preorder traversal (one slot per op).
  // kInvalidIndex means "no properties for this op".
  std::vector<size_t> propsIdPreorder_;
};

// =============================================================================
// BytecodeAssembler - Orchestrates full bytecode emission
// =============================================================================
// This ties together all sections in the correct order:
//   1. Header (magic + version + producer)
//   2. Section 1: Dialect
//   3. Section 3: AttrTypeOffset
//   4. Section 2: AttrType
//   5. Section 4: IR
//   6. Section 6: ResourceOffset
//   7. Section 5: Resource
//   8. Section 0: String
//   9. Section 8: Properties (if non-empty)
//
class BytecodeAssembler {
 public:
  BytecodeAssembler() = default;
  
  // Assemble a complete bytecode from a WireModule.
  // Returns the full bytecode bytes.
  std::vector<uint8_t> assemble(const WireModule& module);
  
  // Access the numbering state (for debugging/inspection)
  const NumberingState& numberingState() const { return numbering_; }
  
 private:
  NumberingState numbering_;
};

}  // namespace stablehlo
}  // namespace lczero
