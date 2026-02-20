// =============================================================================
// numbering.cc - IR Numbering Implementation
// =============================================================================

#include "stablehlo/numbering.h"
#include "stablehlo/writer.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace lczero {
namespace stablehlo {

namespace {

struct TypePtrHash {
  size_t operator()(const TypePtr& t) const {
    return t ? t->hash() : 0;
  }
};

struct TypePtrEqual {
  bool operator()(const TypePtr& a, const TypePtr& b) const {
    if (!a || !b) return a == b;
    return a->equals(*b);
  }
};

struct AttrPtrHash {
  size_t operator()(const AttrPtr& a) const {
    return a ? a->hash() : 0;
  }
};

struct AttrPtrEqual {
  bool operator()(const AttrPtr& a, const AttrPtr& b) const {
    if (!a || !b) return a == b;
    return a->equals(*b);
  }
};

struct TypeUseInfo {
  size_t count = 0;
  size_t order = 0;
  TypePtr type;
};

struct AttrUseInfo {
  size_t count = 0;
  size_t order = 0;
  AttrPtr attr;
};

struct OpNameUseInfo {
  size_t count = 0;
  size_t order = 0;
  std::string opName;
};

class OpNameUsageCollector {
 public:
  void noteOp(const std::string& dialect, const std::string& opName) {
    auto& entry = dialects_[dialect];
    auto it = entry.find(opName);
    if (it == entry.end()) {
      OpNameUseInfo info;
      info.count = 1;
      info.order = nextOrder_++;
      info.opName = opName;
      entry.emplace(opName, info);
    } else {
      ++it->second.count;
    }
  }

  std::vector<std::string> dialectNames() const {
    std::vector<std::string> names;
    names.reserve(dialects_.size());
    for (const auto& kv : dialects_) names.push_back(kv.first);
    std::sort(names.begin(), names.end());
    return names;
  }

  std::vector<OpNameUseInfo> orderedOpsForDialect(const std::string& dialect) const {
    std::vector<OpNameUseInfo> out;
    auto it = dialects_.find(dialect);
    if (it == dialects_.end()) return out;
    out.reserve(it->second.size());
    for (const auto& kv : it->second) out.push_back(kv.second);
    std::stable_sort(out.begin(), out.end(),
                     [](const OpNameUseInfo& a, const OpNameUseInfo& b) {
                       if (a.count != b.count) return a.count > b.count;
                       return a.order < b.order;
                     });
    return out;
  }

 private:
  std::unordered_map<std::string,
                     std::unordered_map<std::string, OpNameUseInfo>> dialects_;
  size_t nextOrder_ = 0;
};

class TypeUsageCollector {
 public:
  void noteType(const TypePtr& type) {
    if (!type) return;
    auto it = uses_.find(type);
    if (it == uses_.end()) {
      TypeUseInfo info;
      info.count = 1;
      info.order = nextOrder_++;
      info.type = type;
      uses_.emplace(type, info);
      // Only recurse into nested types on first sight, matching MLIR's
      // numbering behavior (nested types are discovered when a type is
      // first serialized).
      switch (type->kind()) {
        case TypeKind::kElement:
          return;
        case TypeKind::kRankedTensor: {
          const auto* t = static_cast<const RankedTensorType*>(type.get());
          noteType(t->elementType());
          return;
        }
        case TypeKind::kFunction: {
          const auto* t = static_cast<const FunctionType*>(type.get());
          for (const auto& in : t->inputs()) noteType(in);
          for (const auto& out : t->outputs()) noteType(out);
          return;
        }
        case TypeKind::kTuple: {
          const auto* t = static_cast<const TupleType*>(type.get());
          for (const auto& elem : t->elementTypes()) noteType(elem);
          return;
        }
        default:
          return;
      }
    } else {
      ++it->second.count;
    }
  }

  std::vector<TypeUseInfo> ordered() const {
    std::vector<TypeUseInfo> out;
    out.reserve(uses_.size());
    for (const auto& kv : uses_) out.push_back(kv.second);
    std::stable_sort(out.begin(), out.end(),
                     [](const TypeUseInfo& a, const TypeUseInfo& b) {
                       if (a.count != b.count) return a.count > b.count;
                       return a.order < b.order;
                     });
    return out;
  }

 private:
  std::unordered_map<TypePtr, TypeUseInfo, TypePtrHash, TypePtrEqual> uses_;
  size_t nextOrder_ = 0;
};

class AttrUsageCollector {
 public:
  void noteAttr(const AttrPtr& attr) {
    if (!attr) return;
    auto it = uses_.find(attr);
    if (it == uses_.end()) {
      AttrUseInfo info;
      info.count = 1;
      info.order = nextOrder_++;
      info.attr = attr;
      uses_.emplace(attr, info);

      // Only recurse into nested attrs on first sight, matching MLIR's
      // "duplicate attr -> refCount++, early return" behavior.
      if (attr->kind() == AttrKind::kArray) {
        const auto& elems = static_cast<const ArrayAttr*>(attr.get())->elements();
        for (const auto& e : elems) noteAttr(e);
      }
    } else {
      ++it->second.count;
    }
  }

  std::vector<AttrUseInfo> orderedByRefCount() const {
    std::vector<AttrUseInfo> out;
    out.reserve(uses_.size());
    for (const auto& kv : uses_) out.push_back(kv.second);
    std::stable_sort(out.begin(), out.end(),
                     [](const AttrUseInfo& a, const AttrUseInfo& b) {
                       if (a.count != b.count) return a.count > b.count;
                       return a.order < b.order;
                     });
    return out;
  }

 private:
  std::unordered_map<AttrPtr, AttrUseInfo, AttrPtrHash, AttrPtrEqual> uses_;
  size_t nextOrder_ = 0;
};

static void collectTypesFromOp(const WireOp& op, TypeUsageCollector& c);
static void collectAttrsFromOp(const WireOp& op, AttrUsageCollector& c);
static AttrPtr makeI64TensorAttrForUsage(const std::vector<int64_t>& shape,
                                         const std::vector<int64_t>& values);
static AttrPtr makeI64TensorAttrForUsage(const std::vector<int64_t>& values);
static AttrPtr makeBoolTensorAttrForUsage(const std::vector<int64_t>& shape,
                                          const std::vector<bool>& values);

static void collectTypesFromBlock(const WireBlock& block,
                                  TypeUsageCollector& c) {
  for (const auto& arg : block.args) c.noteType(arg.type);
  for (const auto& op : block.ops) collectTypesFromOp(op, c);
}

static void collectTypesFromRegion(const WireRegion& region,
                                   TypeUsageCollector& c) {
  for (const auto& block : region.blocks) collectTypesFromBlock(block, c);
}

static size_t inferConvolutionSpatialRank(const ConvolutionProps& p) {
  const auto pick = [&](const auto& v) -> size_t { return v.empty() ? 0 : v.size(); };
  const size_t r0 = pick(p.windowStrides);
  if (r0) return r0;
  const size_t r1 = pick(p.lhsDilation);
  if (r1) return r1;
  const size_t r2 = pick(p.rhsDilation);
  if (r2) return r2;
  const size_t r3 = pick(p.inputSpatialDimensions);
  if (r3) return r3;
  const size_t r4 = pick(p.kernelSpatialDimensions);
  if (r4) return r4;
  const size_t r5 = pick(p.outputSpatialDimensions);
  if (r5) return r5;
  return p.windowReversal.size();
}

static std::vector<int64_t> inferConvolutionPaddingShape(
    const ConvolutionProps& p,
    size_t spatialRank) {
  if (p.padding.size() != spatialRank * 2) {
    throw std::runtime_error(
        "convolution_v1 padding size must equal spatial_rank * 2");
  }
  return {static_cast<int64_t>(spatialRank), 2};
}

static std::vector<bool> inferConvolutionWindowReversal(
    const ConvolutionProps& p,
    size_t spatialRank) {
  if (p.windowReversal.empty()) {
    return std::vector<bool>(spatialRank, false);
  }
  if (p.windowReversal.size() != spatialRank) {
    throw std::runtime_error(
        "convolution_v1 window_reversal size must equal spatial_rank");
  }
  return p.windowReversal;
}

static std::vector<uint64_t> inferConvolutionPrecisionConfig(
    const ConvolutionProps& p) {
  if (p.precisionConfig.empty()) {
    return {0, 0};  // DEFAULT, DEFAULT
  }
  return p.precisionConfig;
}

static std::vector<uint8_t> encodeI64DenseRaw(
    const std::vector<int64_t>& values) {
  const bool isSplat = !values.empty() &&
                       std::all_of(values.begin(), values.end(),
                                   [&](int64_t v) { return v == values[0]; });
  const size_t elementCount = (isSplat && values.size() > 1) ? 1 : values.size();
  std::vector<uint8_t> raw;
  raw.reserve(elementCount * 8);
  for (size_t i = 0; i < elementCount; ++i) {
    uint64_t u = static_cast<uint64_t>(values[i]);
    for (int j = 0; j < 8; ++j) {
      raw.push_back(static_cast<uint8_t>((u >> (8 * j)) & 0xFF));
    }
  }
  return raw;
}

static std::vector<uint8_t> encodeBoolDenseRaw(
    const std::vector<bool>& values) {
  if (values.empty()) return {};
  const bool isSplat =
      std::all_of(values.begin(), values.end(),
                  [&](bool v) { return v == values[0]; });
  const size_t logicalCount = (isSplat && values.size() > 1) ? 1 : values.size();
  std::vector<uint8_t> raw((logicalCount + 7) / 8, 0);
  for (size_t i = 0; i < logicalCount; ++i) {
    if (values[i]) raw[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
  }
  return raw;
}

static void collectAttrsFromBlock(const WireBlock& block,
                                  AttrUsageCollector& c) {
  for (const auto& op : block.ops) collectAttrsFromOp(op, c);
}

static void collectAttrsFromRegion(const WireRegion& region,
                                   AttrUsageCollector& c) {
  for (const auto& block : region.blocks) collectAttrsFromBlock(block, c);
}

// Returns a flat list of SSA value types in numbering order:
// block args first, then operation results for each block.
static std::vector<TypePtr> collectValueTypesInRegion(const WireRegion& region) {
  std::vector<TypePtr> values;
  for (const auto& block : region.blocks) {
    for (const auto& arg : block.args) values.push_back(arg.type);
    for (const auto& op : block.ops) {
      for (const auto& t : op.resultTypes) values.push_back(t);
    }
  }
  return values;
}

// Derives func_v1 signature types from the entry region:
// - inputs from entry block arguments
// - outputs from the first return_v1 operation's operands
static std::pair<std::vector<TypePtr>, std::vector<TypePtr>>
deriveFuncV1SignatureTypes(const WireOp& funcOp) {
  std::vector<TypePtr> inputTypes;
  std::vector<TypePtr> outputTypes;
  if (funcOp.regions.empty() || funcOp.regions[0].blocks.empty()) {
    return {std::move(inputTypes), std::move(outputTypes)};
  }

  const auto& entry = funcOp.regions[0].blocks[0];
  inputTypes.reserve(entry.args.size());
  for (const auto& arg : entry.args) inputTypes.push_back(arg.type);

  const auto valueTypes = collectValueTypesInRegion(funcOp.regions[0]);
  for (const auto& block : funcOp.regions[0].blocks) {
    for (const auto& bodyOp : block.ops) {
      if (bodyOp.opName != "return_v1") continue;
      outputTypes.reserve(bodyOp.operands.size());
      for (ValueRef v : bodyOp.operands) {
        if (v < valueTypes.size()) outputTypes.push_back(valueTypes[v]);
      }
      return {std::move(inputTypes), std::move(outputTypes)};
    }
  }
  return {std::move(inputTypes), std::move(outputTypes)};
}

static void collectTypesFromOp(const WireOp& op, TypeUsageCollector& c) {
  for (const auto& t : op.resultTypes) c.noteType(t);

  // func_v1 properties: function type built from arg/result types.
  if (op.dialectName == "vhlo" && op.opName == "func_v1" && !op.regions.empty()) {
    auto [inputTypes, outputTypes] = deriveFuncV1SignatureTypes(op);
    c.noteType(makeFunction(std::move(inputTypes), std::move(outputTypes)));
  }

  // constant_v1 properties: DenseElementsAttr uses the result type.
  if (op.constantProps.has_value() && !op.resultTypes.empty()) {
    c.noteType(op.resultTypes[0]);
  }

  // reduce_v1 properties: dimensions attr uses tensor<rank x i64>.
  if (op.reduceProps.has_value()) {
    auto i64 = makeElementType(ElementType::kSI64);
    int64_t rank = static_cast<int64_t>(op.reduceProps->dimensions.size());
    c.noteType(makeRankedTensor({rank}, i64));
  }

  // transpose_v1 properties: permutation attr uses tensor<rank x i64>.
  if (op.transposeProps.has_value()) {
    auto i64 = makeElementType(ElementType::kSI64);
    int64_t rank = static_cast<int64_t>(op.transposeProps->permutation.size());
    c.noteType(makeRankedTensor({rank}, i64));
  }

  // broadcast_in_dim_v1 properties: broadcast_dimensions uses tensor<rank x i64>.
  if (op.broadcastInDimProps.has_value()) {
    auto i64 = makeElementType(ElementType::kSI64);
    int64_t rank =
        static_cast<int64_t>(op.broadcastInDimProps->broadcastDimensions.size());
    c.noteType(makeRankedTensor({rank}, i64));
  }

  // concatenate_v1 properties: dimension uses IntegerAttr<i64>.
  if (op.concatenateProps.has_value()) {
    c.noteType(makeElementType(ElementType::kSI64));
  }

  // slice_v1 properties: start/limit/strides attrs use tensor<rank x i64>.
  if (op.sliceProps.has_value()) {
    auto i64 = makeElementType(ElementType::kSI64);
    int64_t startRank = static_cast<int64_t>(op.sliceProps->startIndices.size());
    int64_t limitRank = static_cast<int64_t>(op.sliceProps->limitIndices.size());
    int64_t stridesRank = static_cast<int64_t>(op.sliceProps->strides.size());
    c.noteType(makeRankedTensor({startRank}, i64));
    c.noteType(makeRankedTensor({limitRank}, i64));
    c.noteType(makeRankedTensor({stridesRank}, i64));
  }

  // dot_general_v1 properties: 4 dimension arrays use tensor<rank x i64>.
  // precision_config is an enum array and does not reference types directly.
  if (op.dotGeneralProps.has_value()) {
    auto i64 = makeElementType(ElementType::kSI64);

    struct I64TensorAttrKey {
      std::vector<int64_t> shape;
      std::vector<int64_t> values;
    };

    std::vector<I64TensorAttrKey> uniqueI64TensorAttrs;
    auto noteUniqueI64Tensor = [&](const std::vector<int64_t>& values) {
      I64TensorAttrKey key{{static_cast<int64_t>(values.size())}, values};
      auto it = std::find_if(uniqueI64TensorAttrs.begin(), uniqueI64TensorAttrs.end(),
                             [&](const I64TensorAttrKey& existing) {
                               return existing.shape == key.shape &&
                                      existing.values == key.values;
                             });
      if (it != uniqueI64TensorAttrs.end()) return;
      uniqueI64TensorAttrs.push_back(key);
      c.noteType(makeRankedTensor(key.shape, i64));
    };

    noteUniqueI64Tensor(op.dotGeneralProps->lhsBatchingDimensions);
    noteUniqueI64Tensor(op.dotGeneralProps->lhsContractingDimensions);
    noteUniqueI64Tensor(op.dotGeneralProps->rhsBatchingDimensions);
    noteUniqueI64Tensor(op.dotGeneralProps->rhsContractingDimensions);
  }

  // gather_v1 properties: i64 tensor attrs are uniqued by (shape,payload).
  if (op.gatherProps.has_value()) {
    auto i64 = makeElementType(ElementType::kSI64);

    struct I64TensorAttrKey {
      std::vector<int64_t> shape;
      std::vector<int64_t> values;
    };

    std::vector<I64TensorAttrKey> uniqueI64TensorAttrs;
    auto noteUniqueI64Tensor = [&](const std::vector<int64_t>& values) {
      I64TensorAttrKey key{{static_cast<int64_t>(values.size())}, values};
      auto it = std::find_if(uniqueI64TensorAttrs.begin(), uniqueI64TensorAttrs.end(),
                             [&](const I64TensorAttrKey& existing) {
                               return existing.shape == key.shape &&
                                      existing.values == key.values;
                             });
      if (it != uniqueI64TensorAttrs.end()) return;
      uniqueI64TensorAttrs.push_back(key);
      c.noteType(makeRankedTensor(key.shape, i64));
    };

    noteUniqueI64Tensor(op.gatherProps->collapsedSliceDims);
    c.noteType(i64);  // index_vector_dim uses i64 integer attr
    noteUniqueI64Tensor(op.gatherProps->offsetDims);
    noteUniqueI64Tensor(op.gatherProps->sliceSizes);
    noteUniqueI64Tensor(op.gatherProps->startIndexMap);
  }

  // convolution_v1 properties: mirror MLIR attr-uniquing side effects by
  // counting nested types only on first-seen attrs (structural equality).
  if (op.convolutionProps.has_value()) {
    const auto& p = *op.convolutionProps;
    const size_t spatialRank = inferConvolutionSpatialRank(p);
    const auto paddingShape = inferConvolutionPaddingShape(p, spatialRank);
    const auto windowReversal = inferConvolutionWindowReversal(p, spatialRank);
    const auto precisionConfig = inferConvolutionPrecisionConfig(p);
    auto i64 = makeElementType(ElementType::kSI64);

    std::unordered_set<AttrPtr, AttrPtrHash, AttrPtrEqual> seenAttrs;
    std::function<void(const AttrPtr&)> noteTypesFromAttr =
        [&](const AttrPtr& attr) {
          if (!attr) return;
          if (!seenAttrs.insert(attr).second) return;
          switch (attr->kind()) {
            case AttrKind::kArray: {
              const auto& elems =
                  static_cast<const ArrayAttr*>(attr.get())->elements();
              for (const auto& e : elems) noteTypesFromAttr(e);
              return;
            }
            case AttrKind::kType: {
              const auto& t = static_cast<const TypeAttr*>(attr.get())->type();
              c.noteType(t);
              return;
            }
            case AttrKind::kInteger: {
              const auto& t =
                  static_cast<const IntegerAttr*>(attr.get())->type();
              c.noteType(t);
              return;
            }
            case AttrKind::kFloat: {
              const auto& t = static_cast<const FloatAttr*>(attr.get())->type();
              c.noteType(t);
              return;
            }
            case AttrKind::kDenseElements: {
              const auto& t =
                  static_cast<const DenseElementsAttr*>(attr.get())->type();
              c.noteType(t);
              return;
            }
            default:
              return;
          }
        };

    // Attribute order is alphabetical:
    // batch_group_count, feature_group_count,
    // input_batch_dimension, input_feature_dimension, input_spatial_dimensions,
    // kernel_input_feature_dimension, kernel_output_feature_dimension,
    // kernel_spatial_dimensions, lhs_dilation,
    // output_batch_dimension, output_feature_dimension, output_spatial_dimensions,
    // padding, precision_config, rhs_dilation, window_reversal, window_strides.
    noteTypesFromAttr(makeIntegerAttr(p.batchGroupCount, i64));
    noteTypesFromAttr(makeIntegerAttr(p.featureGroupCount, i64));
    noteTypesFromAttr(makeIntegerAttr(p.inputBatchDimension, i64));
    noteTypesFromAttr(makeIntegerAttr(p.inputFeatureDimension, i64));
    noteTypesFromAttr(makeI64TensorAttrForUsage(p.inputSpatialDimensions));
    noteTypesFromAttr(makeIntegerAttr(p.kernelInputFeatureDimension, i64));
    noteTypesFromAttr(makeIntegerAttr(p.kernelOutputFeatureDimension, i64));
    noteTypesFromAttr(makeI64TensorAttrForUsage(p.kernelSpatialDimensions));
    noteTypesFromAttr(makeI64TensorAttrForUsage(p.lhsDilation));
    noteTypesFromAttr(makeIntegerAttr(p.outputBatchDimension, i64));
    noteTypesFromAttr(makeIntegerAttr(p.outputFeatureDimension, i64));
    noteTypesFromAttr(makeI64TensorAttrForUsage(p.outputSpatialDimensions));
    noteTypesFromAttr(makeI64TensorAttrForUsage(paddingShape, p.padding));

    std::vector<AttrPtr> precisionElems;
    precisionElems.reserve(precisionConfig.size());
    for (uint64_t value : precisionConfig) {
      precisionElems.push_back(makePrecisionAttr(value));
    }
    noteTypesFromAttr(makeArrayAttr(std::move(precisionElems)));

    noteTypesFromAttr(makeI64TensorAttrForUsage(p.rhsDilation));
    noteTypesFromAttr(
        makeBoolTensorAttrForUsage({static_cast<int64_t>(spatialRank)},
                                   windowReversal));
    noteTypesFromAttr(makeI64TensorAttrForUsage(p.windowStrides));
  }

  for (const auto& region : op.regions) collectTypesFromRegion(region, c);
}

static AttrPtr makeI64TensorAttrForUsage(const std::vector<int64_t>& shape,
                                         const std::vector<int64_t>& values) {
  auto i64 = makeElementType(ElementType::kSI64);
  auto tensorTy = makeRankedTensor(shape, i64);
  std::vector<uint8_t> raw = encodeI64DenseRaw(values);
  return makeDenseElementsAttr(tensorTy, std::move(raw));
}

static AttrPtr makeI64TensorAttrForUsage(const std::vector<int64_t>& values) {
  return makeI64TensorAttrForUsage(
      {static_cast<int64_t>(values.size())}, values);
}

static AttrPtr makeBoolTensorAttrForUsage(const std::vector<int64_t>& shape,
                                          const std::vector<bool>& values) {
  auto i1 = makeElementType(ElementType::kBool);
  auto tensorTy = makeRankedTensor(shape, i1);
  std::vector<uint8_t> raw = encodeBoolDenseRaw(values);
  return makeDenseElementsAttr(tensorTy, std::move(raw));
}

static void collectAttrsFromOp(const WireOp& op, AttrUsageCollector& c) {
  auto noteVhloAttr = [&](const AttrPtr& attr) {
    if (op.dialectName != "vhlo") {
      throw std::runtime_error(
          "single-dialect invariant violated in attr collector: " +
          op.dialectName + "." + op.opName);
    }
    c.noteAttr(attr);
  };

  if (op.dialectName == "vhlo" && op.opName == "func_v1" && !op.regions.empty()) {
    auto [inputTypes, outputTypes] = deriveFuncV1SignatureTypes(op);
    auto funcType = makeFunction(std::move(inputTypes), std::move(outputTypes));
    noteVhloAttr(makeArrayAttr({}));  // arg_attrs
    noteVhloAttr(makeTypeAttr(funcType));  // function_type
    noteVhloAttr(makeArrayAttr({}));  // res_attrs
    noteVhloAttr(makeStringAttr(op.symName.value_or("main")));  // sym_name
    noteVhloAttr(makeStringAttr(op.symVisibility.value_or("")));  // sym_visibility
  } else if (op.dialectName == "vhlo" && op.opName == "compare_v1" &&
             op.compareProps.has_value()) {
    noteVhloAttr(makeComparisonTypeAttr(op.compareProps->comparisonType));
    noteVhloAttr(makeComparisonDirectionAttr(op.compareProps->comparisonDirection));
  } else if (op.dialectName == "vhlo" && op.opName == "gather_v1" &&
             op.gatherProps.has_value()) {
    auto i64 = makeElementType(ElementType::kSI64);
    noteVhloAttr(makeI64TensorAttrForUsage(op.gatherProps->collapsedSliceDims));
    noteVhloAttr(makeIntegerAttr(op.gatherProps->indexVectorDim, i64));
    noteVhloAttr(makeBooleanAttr(op.gatherProps->indicesAreSorted));
    noteVhloAttr(makeI64TensorAttrForUsage(op.gatherProps->offsetDims));
    noteVhloAttr(makeI64TensorAttrForUsage(op.gatherProps->sliceSizes));
    noteVhloAttr(makeI64TensorAttrForUsage(op.gatherProps->startIndexMap));
  } else if (op.dialectName == "vhlo" && op.opName == "constant_v1" &&
             op.constantProps.has_value() && !op.resultTypes.empty()) {
    noteVhloAttr(makeDenseElementsAttr(op.resultTypes[0], op.constantProps->rawData));
  } else if (op.dialectName == "vhlo" && op.opName == "reduce_v1" &&
             op.reduceProps.has_value()) {
    noteVhloAttr(makeI64TensorAttrForUsage(op.reduceProps->dimensions));
  } else if (op.dialectName == "vhlo" && op.opName == "transpose_v1" &&
             op.transposeProps.has_value()) {
    noteVhloAttr(makeI64TensorAttrForUsage(op.transposeProps->permutation));
  } else if (op.dialectName == "vhlo" && op.opName == "broadcast_in_dim_v1" &&
             op.broadcastInDimProps.has_value()) {
    noteVhloAttr(
        makeI64TensorAttrForUsage(op.broadcastInDimProps->broadcastDimensions));
  } else if (op.dialectName == "vhlo" && op.opName == "concatenate_v1" &&
             op.concatenateProps.has_value()) {
    auto i64 = makeElementType(ElementType::kSI64);
    noteVhloAttr(makeIntegerAttr(op.concatenateProps->dimension, i64));
  } else if (op.dialectName == "vhlo" && op.opName == "slice_v1" &&
             op.sliceProps.has_value()) {
    noteVhloAttr(makeI64TensorAttrForUsage(op.sliceProps->limitIndices));
    noteVhloAttr(makeI64TensorAttrForUsage(op.sliceProps->startIndices));
    noteVhloAttr(makeI64TensorAttrForUsage(op.sliceProps->strides));
  } else if (op.dialectName == "vhlo" && op.opName == "dot_general_v1" &&
             op.dotGeneralProps.has_value()) {
    // Keep this in deterministic/alphabetical property order to match the
    // builder and stabilize first-seen tie-breaks:
    // lhs_batching_dimensions, lhs_contracting_dimensions, precision_config,
    // rhs_batching_dimensions, rhs_contracting_dimensions.
    noteVhloAttr(
        makeI64TensorAttrForUsage(op.dotGeneralProps->lhsBatchingDimensions));
    noteVhloAttr(
        makeI64TensorAttrForUsage(op.dotGeneralProps->lhsContractingDimensions));
    std::vector<AttrPtr> precElems;
    precElems.reserve(op.dotGeneralProps->precisionConfig.size());
    for (uint64_t pv : op.dotGeneralProps->precisionConfig) {
      precElems.push_back(makePrecisionAttr(pv));
    }
    noteVhloAttr(makeArrayAttr(std::move(precElems)));
    noteVhloAttr(
        makeI64TensorAttrForUsage(op.dotGeneralProps->rhsBatchingDimensions));
    noteVhloAttr(
        makeI64TensorAttrForUsage(op.dotGeneralProps->rhsContractingDimensions));
  } else if (op.dialectName == "vhlo" && op.opName == "convolution_v1" &&
             op.convolutionProps.has_value()) {
    const auto& p = *op.convolutionProps;
    const size_t spatialRank = inferConvolutionSpatialRank(p);
    const auto paddingShape = inferConvolutionPaddingShape(p, spatialRank);
    const auto windowReversal = inferConvolutionWindowReversal(p, spatialRank);
    const auto precisionConfig = inferConvolutionPrecisionConfig(p);
    auto i64 = makeElementType(ElementType::kSI64);

    // Attribute order is alphabetical:
    // batch_group_count, feature_group_count,
    // input_batch_dimension, input_feature_dimension, input_spatial_dimensions,
    // kernel_input_feature_dimension, kernel_output_feature_dimension,
    // kernel_spatial_dimensions, lhs_dilation,
    // output_batch_dimension, output_feature_dimension, output_spatial_dimensions,
    // padding, precision_config, rhs_dilation, window_reversal, window_strides.
    noteVhloAttr(makeIntegerAttr(p.batchGroupCount, i64));
    noteVhloAttr(makeIntegerAttr(p.featureGroupCount, i64));
    noteVhloAttr(makeIntegerAttr(p.inputBatchDimension, i64));
    noteVhloAttr(makeIntegerAttr(p.inputFeatureDimension, i64));
    noteVhloAttr(makeI64TensorAttrForUsage(p.inputSpatialDimensions));
    noteVhloAttr(makeIntegerAttr(p.kernelInputFeatureDimension, i64));
    noteVhloAttr(makeIntegerAttr(p.kernelOutputFeatureDimension, i64));
    noteVhloAttr(makeI64TensorAttrForUsage(p.kernelSpatialDimensions));
    noteVhloAttr(makeI64TensorAttrForUsage(p.lhsDilation));
    noteVhloAttr(makeIntegerAttr(p.outputBatchDimension, i64));
    noteVhloAttr(makeIntegerAttr(p.outputFeatureDimension, i64));
    noteVhloAttr(makeI64TensorAttrForUsage(p.outputSpatialDimensions));
    noteVhloAttr(makeI64TensorAttrForUsage(paddingShape, p.padding));

    std::vector<AttrPtr> precisionElems;
    precisionElems.reserve(precisionConfig.size());
    for (uint64_t value : precisionConfig) {
      precisionElems.push_back(makePrecisionAttr(value));
    }
    noteVhloAttr(makeArrayAttr(std::move(precisionElems)));

    noteVhloAttr(makeI64TensorAttrForUsage(p.rhsDilation));
    noteVhloAttr(makeBoolTensorAttrForUsage(
        {static_cast<int64_t>(spatialRank)}, windowReversal));
    noteVhloAttr(makeI64TensorAttrForUsage(p.windowStrides));
  }

  for (const auto& region : op.regions) collectAttrsFromRegion(region, c);
}

static void collectOpNamesFromOp(const WireOp& op, OpNameUsageCollector& c) {
  c.noteOp(op.dialectName, op.opName);
  for (const auto& region : op.regions) {
    for (const auto& block : region.blocks) {
      for (const auto& nested : block.ops) {
        collectOpNamesFromOp(nested, c);
      }
    }
  }
}

}  // namespace

// =============================================================================
// NumberingState Implementation
// =============================================================================

void NumberingState::number(const WireModule& module) {
  // Reset state
  nextValueID_ = 0;
  // Reset interners/section builders to avoid state leakage across runs.
  strings_ = StringInterner();
  dialects_ = DialectSectionWriter();
  attrTypes_ = AttrTypeSectionWriter();
  typeInterner_ = TypeInterner();
  attrInterner_ = AttrInterner();
  vhloAttrIndexToGlobal_.clear();
  opNameIndices_.clear();
  propertiesTable_.clear();
  propertiesDedup_.clear();
  builtinAttrDedup_.clear();
  propsIdPreorder_.clear();
  unknownLocAttrIndex_ = kInvalidIndex;
  builtinDialectId_ = kInvalidIndex;
  typesFrozen_ = false;
  attrsFrozen_ = false;
  preCollectedAttrs_.clear();
  
  // Ensure builtin dialect is registered first (dialect 0)
  ensureBuiltinDialect();
  // Keep UnknownLoc available before section payloads are materialized.
  ensureUnknownLocAttr();
  
  // Pre-number op names and pre-intern dialect/op name strings.
  preNumberOpNames(module);
  preInternDialectAndOpNames();

  // Pre-intern module builtin attrs (e.g. sym_name/sym_visibility) so builtin
  // indices are fixed before VHLO attr finalization.
  preInternBuiltinAttrs(module);

  // Pre-number types in a refcount-stable order before walking IR.
  preNumberTypes(module);

  // Pre-collect attrs in MLIR-like first-seen/refcount order.
  preCollectAttrs(module);

  // Finalize global VHLO attr order and mapping before IR walk.
  finalizeAttrOrder();
  
  // Number the root op (typically "module")
  numberOp(module.rootOp);

  // Finalize type payloads now that all indices are fixed.
  finalizeTypePayloads();
}

void NumberingState::ensureBuiltinDialect() {
  if (builtinDialectId_ != kInvalidIndex) return;
  builtinDialectId_ = dialects_.addDialect("builtin");
}

void NumberingState::ensureUnknownLocAttr() {
  if (unknownLocAttrIndex_ != kInvalidIndex) return;
  
  // BuiltinDialectBytecode.td: UnknownLoc has builtin attr code 15.
  // Attr payload is MLIR-varint(code) => varint(15) == 0x1f.
  const std::vector<uint8_t> unknownLocPayload = {0x1f};
  
  unknownLocAttrIndex_ = internBuiltinAttr(unknownLocPayload);
}

size_t NumberingState::internBuiltinAttr(const std::vector<uint8_t>& payload) {
  ensureBuiltinDialect();
  std::string key(payload.begin(), payload.end());
  auto it = builtinAttrDedup_.find(key);
  if (it != builtinAttrDedup_.end()) return it->second;
  if (attrsFrozen_) {
    throw std::runtime_error("Builtin attr not pre-numbered before numbering pass");
  }
  size_t idx = attrTypes_.addAttr(payload, builtinDialectId_, true);
  builtinAttrDedup_[key] = idx;
  return idx;
}

size_t NumberingState::internVhloAttr(const AttrPtr& attr) {
  if (!attr) return kInvalidIndex;

  size_t existing = attrInterner_.lookup(attr);
  if (existing != kInvalidIndex) {
    return vhloAttrIndexToGlobal_[existing];
  }

  if (attrsFrozen_) {
    throw std::runtime_error("Attr not pre-numbered before numbering pass");
  }

  // Ensure dependent types/attrs are interned first (so indices are stable).
  switch (attr->kind()) {
    case AttrKind::kArray: {
      const auto& elems = static_cast<const ArrayAttr&>(*attr).elements();
      for (const auto& e : elems) {
        internVhloAttr(e);
      }
      break;
    }
    case AttrKind::kType: {
      const auto& t = static_cast<const TypeAttr&>(*attr).type();
      getTypeIndex(t);
      break;
    }
    case AttrKind::kInteger: {
      const auto& t = static_cast<const IntegerAttr&>(*attr).type();
      getTypeIndex(t);
      break;
    }
    case AttrKind::kFloat: {
      const auto& t = static_cast<const FloatAttr&>(*attr).type();
      getTypeIndex(t);
      break;
    }
    case AttrKind::kDenseElements: {
      const auto& t = static_cast<const DenseElementsAttr&>(*attr).type();
      getTypeIndex(t);
      break;
    }
    default:
      break;
  }

  size_t localIndex = attrInterner_.intern(attr, AttrInterner::kDialectVhlo);
  if (localIndex == vhloAttrIndexToGlobal_.size()) {
    // Encode and add to AttrType section.
    std::vector<uint8_t> payload = encodeVhloAttr(*attr);
    size_t vhloDialectId = dialects_.addDialect("vhlo");
    size_t globalIndex = attrTypes_.addAttr(std::move(payload), vhloDialectId, true);
    vhloAttrIndexToGlobal_.push_back(globalIndex);
  }
  return vhloAttrIndexToGlobal_[localIndex];
}

size_t NumberingState::internI64IntegerAttr(int64_t value) {
  auto i64 = makeElementType(ElementType::kSI64);
  auto attr = makeIntegerAttr(value, i64);
  return internVhloAttr(attr);
}

size_t NumberingState::internI64TensorAttr(const std::vector<int64_t>& values) {
  return internI64TensorAttrWithShape(
      {static_cast<int64_t>(values.size())}, values);
}

size_t NumberingState::internI64TensorAttrWithShape(
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& values) {
  auto i64 = makeElementType(ElementType::kSI64);
  auto tensorTy = makeRankedTensor(shape, i64);
  std::vector<uint8_t> raw = encodeI64DenseRaw(values);
  auto attr = makeDenseElementsAttr(tensorTy, std::move(raw));
  return internVhloAttr(attr);
}

size_t NumberingState::internBoolTensorAttrWithShape(
    const std::vector<int64_t>& shape,
    const std::vector<bool>& values) {
  auto i1 = makeElementType(ElementType::kBool);
  auto tensorTy = makeRankedTensor(shape, i1);
  std::vector<uint8_t> raw = encodeBoolDenseRaw(values);
  auto attr = makeDenseElementsAttr(tensorTy, std::move(raw));
  return internVhloAttr(attr);
}

std::vector<uint8_t> NumberingState::encodeVhloAttr(const Attr& attr) {
  switch (attr.kind()) {
    case AttrKind::kString: {
      const auto& a = static_cast<const StringAttr&>(attr);
      size_t strIdx = strings_.intern(a.value());
      return encodeStringAttr(strIdx);
    }
    case AttrKind::kInteger: {
      const auto& a = static_cast<const IntegerAttr&>(attr);
      size_t typeIdx = getTypeIndex(a.type());
      return encodeIntegerAttr(typeIdx, a.value());
    }
    case AttrKind::kBoolean: {
      const auto& a = static_cast<const BooleanAttr&>(attr);
      return encodeBooleanAttr(a.value());
    }
    case AttrKind::kArray: {
      const auto& a = static_cast<const ArrayAttr&>(attr);
      std::vector<size_t> indices;
      indices.reserve(a.elements().size());
      for (const auto& e : a.elements()) {
        size_t idx = internVhloAttr(e);
        assert(idx != kInvalidIndex && "failed to intern array element");
        indices.push_back(idx);
      }
      return encodeArrayAttr(indices);
    }
    case AttrKind::kType: {
      const auto& a = static_cast<const TypeAttr&>(attr);
      size_t typeIdx = getTypeIndex(a.type());
      return encodeTypeAttr(typeIdx);
    }
    case AttrKind::kDenseElements: {
      const auto& a = static_cast<const DenseElementsAttr&>(attr);
      size_t typeIdx = getTypeIndex(a.type());
      return encodeTensorAttr(typeIdx, a.data());
    }
    case AttrKind::kComparisonDirection: {
      const auto& a = static_cast<const ComparisonDirectionAttr&>(attr);
      return encodeComparisonDirectionAttr(a.value());
    }
    case AttrKind::kComparisonType: {
      const auto& a = static_cast<const ComparisonTypeAttr&>(attr);
      return encodeComparisonTypeAttr(a.value());
    }
    case AttrKind::kPrecision: {
      const auto& a = static_cast<const PrecisionAttr&>(attr);
      return encodePrecisionAttr(a.value());
    }
    case AttrKind::kUnit:
    case AttrKind::kDictionary:
    case AttrKind::kSymbolRef:
    case AttrKind::kFloat:
    default:
      throw std::runtime_error("Unsupported or unverified VHLO attr encoding");
  }
}

void NumberingState::numberOp(const WireOp& op) {
  // 1. Register the op name
  getOpNameIndex(op.dialectName, op.opName, op.isRegistered);
  
  // 2. Register result types
  for (const auto& type : op.resultTypes) {
    getTypeIndex(type);
    // Each result consumes a value ID in the current region
    nextValueID_++;
  }
  
  // 3. Generate/register properties for known ops
  // For func_v1 and module, we auto-generate correct properties
  // based on the op structure (rather than using placeholder bytes)
  const size_t myIdx = propsIdPreorder_.size();
  // INVARIANT: numberOp() and toOpInfoImpl() must visit ops in identical
  // preorder so this slot aligns with the corresponding consume.
  propsIdPreorder_.push_back(kInvalidIndex);  // one slot per op, always
  
  bool typedPropsRequired = false;
  std::optional<size_t> typedPropsId = std::nullopt;

  if (op.dialectName == "vhlo" && op.opName == "func_v1") {
    typedPropsRequired = true;
    if (op.regions.empty()) {
      throw std::runtime_error("Missing entry region for typed properties: vhlo.func_v1");
    }

    auto [inputTypes, outputTypes] = deriveFuncV1SignatureTypes(op);
    for (const auto& t : inputTypes) getTypeIndex(t);
    for (const auto& t : outputTypes) getTypeIndex(t);

    // Create function type and get its index
    auto funcType = makeFunction(inputTypes, outputTypes);
    getTypeIndex(funcType);

    // Build and register func properties
    const std::string funcName = op.symName.value_or("main");
    const std::string symVis = op.symVisibility.value_or("");
    auto funcProps = buildFuncV1Props(funcType, funcName, symVis);
    typedPropsId = getPropertiesId(funcProps);

  } else if (op.dialectName == "builtin" && op.opName == "module") {
    typedPropsRequired = true;
    auto moduleProps = buildModuleProps(op);
    typedPropsId = getPropertiesId(moduleProps);

  } else if (op.dialectName == "vhlo" && op.opName == "compare_v1") {
    typedPropsRequired = true;
    if (!op.compareProps.has_value()) {
      throw std::runtime_error("Missing typed properties for op: vhlo.compare_v1");
    }
    auto props = buildCompareV1Props(*op.compareProps);
    typedPropsId = getPropertiesId(props);

  } else if (op.dialectName == "vhlo" && op.opName == "gather_v1") {
    typedPropsRequired = true;
    if (!op.gatherProps.has_value()) {
      throw std::runtime_error("Missing typed properties for op: vhlo.gather_v1");
    }
    auto props = buildGatherV1Props(*op.gatherProps);
    typedPropsId = getPropertiesId(props);

  } else if (op.dialectName == "vhlo" && op.opName == "constant_v1") {
    typedPropsRequired = true;
    if (!op.constantProps.has_value()) {
      throw std::runtime_error("Missing typed properties for op: vhlo.constant_v1");
    }
    auto props = buildConstantV1Props(op, *op.constantProps);
    typedPropsId = getPropertiesId(props);

  } else if (op.dialectName == "vhlo" && op.opName == "reduce_v1") {
    typedPropsRequired = true;
    if (!op.reduceProps.has_value()) {
      throw std::runtime_error("Missing typed properties for op: vhlo.reduce_v1");
    }
    auto props = buildReduceV1Props(*op.reduceProps);
    typedPropsId = getPropertiesId(props);

  } else if (op.dialectName == "vhlo" && op.opName == "transpose_v1") {
    typedPropsRequired = true;
    if (!op.transposeProps.has_value()) {
      throw std::runtime_error("Missing typed properties for op: vhlo.transpose_v1");
    }
    auto props = buildTransposeV1Props(*op.transposeProps);
    typedPropsId = getPropertiesId(props);

  } else if (op.dialectName == "vhlo" && op.opName == "broadcast_in_dim_v1") {
    typedPropsRequired = true;
    if (!op.broadcastInDimProps.has_value()) {
      throw std::runtime_error(
          "Missing typed properties for op: vhlo.broadcast_in_dim_v1");
    }
    auto props = buildBroadcastInDimV1Props(*op.broadcastInDimProps);
    typedPropsId = getPropertiesId(props);

  } else if (op.dialectName == "vhlo" && op.opName == "concatenate_v1") {
    typedPropsRequired = true;
    if (!op.concatenateProps.has_value()) {
      throw std::runtime_error(
          "Missing typed properties for op: vhlo.concatenate_v1");
    }
    auto props = buildConcatenateV1Props(*op.concatenateProps);
    typedPropsId = getPropertiesId(props);

  } else if (op.dialectName == "vhlo" && op.opName == "slice_v1") {
    typedPropsRequired = true;
    if (!op.sliceProps.has_value()) {
      throw std::runtime_error("Missing typed properties for op: vhlo.slice_v1");
    }
    auto props = buildSliceV1Props(*op.sliceProps);
    typedPropsId = getPropertiesId(props);

  } else if (op.dialectName == "vhlo" && op.opName == "dot_general_v1") {
    typedPropsRequired = true;
    if (!op.dotGeneralProps.has_value()) {
      throw std::runtime_error(
          "Missing typed properties for op: vhlo.dot_general_v1");
    }
    auto props = buildDotGeneralV1Props(*op.dotGeneralProps);
    typedPropsId = getPropertiesId(props);

  } else if (op.dialectName == "vhlo" && op.opName == "convolution_v1") {
    typedPropsRequired = true;
    if (!op.convolutionProps.has_value()) {
      throw std::runtime_error(
          "Missing typed properties for op: vhlo.convolution_v1");
    }
    auto props = buildConvolutionV1Props(*op.convolutionProps);
    typedPropsId = getPropertiesId(props);

  } else if (op.propertiesPayload.has_value()) {
    // Opaque properties are not allowed for registered ops.
    if (op.isRegistered) {
      throw std::runtime_error("Opaque properties not allowed for registered op: " +
                               op.dialectName + "." + op.opName);
    }
    // Use provided properties for unregistered ops only.
    typedPropsId = getPropertiesId(*op.propertiesPayload);
  }

  if (typedPropsId.has_value()) {
    propsIdPreorder_[myIdx] = *typedPropsId;
  }

  // Enforce required typed properties for known ops.
  if (typedPropsRequired && !typedPropsId.has_value()) {
    throw std::runtime_error("Missing required properties for op: " +
                             op.dialectName + "." + op.opName);
  }
  
  // 4. Number nested regions
  // Key rule: isolated regions reset value numbering to 0
  for (const auto& region : op.regions) {
    numberRegion(region, op.isIsolatedFromAbove);
  }
}

void NumberingState::numberRegion(const WireRegion& region, bool isIsolated) {
  if (region.blocks.empty()) {
    return;
  }
  
  // Save current value ID if not isolated
  size_t savedValueID = nextValueID_;
  
  // Isolated regions reset to 0
  if (isIsolated) {
    nextValueID_ = 0;
  }
  
  // Number all blocks in the region
  // Value numbering spans ALL blocks (not per-block reset!)
  for (const auto& block : region.blocks) {
    numberBlock(block);
  }
  
  // Isolated regions restore the parent's saved value counter because their
  // numbering is independent. Non-isolated regions keep the incremented value.
  if (!isIsolated) {
    // Non-isolated regions continue the parent's value numbering
    // nextValueID_ already has the new value
  } else {
    // Isolated region - restore parent's counter
    nextValueID_ = savedValueID;
  }
}

void NumberingState::numberBlock(const WireBlock& block) {
  // 1. Number block arguments (they define values)
  for (const auto& arg : block.args) {
    getTypeIndex(arg.type);
    // Block arg consumes a value ID
    nextValueID_++;
  }
  
  // 2. Number operations in the block
  for (const auto& op : block.ops) {
    numberOp(op);
  }
}

size_t NumberingState::getOpNameIndex(const std::string& dialectName,
                                      const std::string& opName,
                                      bool isRegistered) {
  std::string fullName = dialectName + "." + opName;
  auto it = opNameIndices_.find(fullName);
  if (it != opNameIndices_.end()) {
    if (!isRegistered) {
      throw std::runtime_error(
          "op registration mismatch: '" + fullName +
          "' was pre-seeded as registered but WireOp says unregistered");
    }
    return it->second;
  }
  if (isRegistered) {
    throw std::runtime_error(
        "op pre-seed drift: registered op '" + fullName +
        "' was not pre-seeded in preNumberOpNames");
  }
  
  // Ensure dialect is registered
  size_t dialectId = dialects_.addDialect(dialectName);
  
  // Add op to dialect
  dialects_.addOp(dialectId, opName, isRegistered);
  
  // The index is based on order of addition
  size_t index = opNameIndices_.size();
  opNameIndices_[fullName] = index;
  return index;
}

std::vector<uint8_t> NumberingState::encodeTypePayload(const TypePtr& type) {
  if (type->kind() == TypeKind::kElement) {
    auto* elemTypeWrapper = static_cast<const ElementTypeWrapper*>(type.get());
    return encodeElementType(elemTypeWrapper->elementType());
  }
  if (type->kind() == TypeKind::kRankedTensor) {
    auto* tensorType = static_cast<const RankedTensorType*>(type.get());
    size_t elemTypeIndex = getTypeIndex(tensorType->elementType());
    return encodeRankedTensor(tensorType->shape(), elemTypeIndex);
  }
  if (type->kind() == TypeKind::kFunction) {
    auto* funcType = static_cast<const FunctionType*>(type.get());
    std::vector<size_t> inputIndices, outputIndices;
    for (const auto& t : funcType->inputs()) {
      inputIndices.push_back(getTypeIndex(t));
    }
    for (const auto& t : funcType->outputs()) {
      outputIndices.push_back(getTypeIndex(t));
    }
    return encodeFunctionType(inputIndices, outputIndices);
  }
  if (type->kind() == TypeKind::kTuple) {
    auto* tupleType = static_cast<const TupleType*>(type.get());
    std::vector<size_t> elemIndices;
    for (const auto& t : tupleType->elementTypes()) {
      elemIndices.push_back(getTypeIndex(t));
    }
    return encodeTupleType(elemIndices);
  }
  throw std::runtime_error(
      "Unsupported TypeKind in encodeTypePayload: " +
      std::to_string(static_cast<int>(type->kind())));
}

size_t NumberingState::getTypeIndex(const TypePtr& type) {
  // Check for an existing structurally-equal type.
  size_t existing = lookupTypeIndex(type);
  if (existing != kInvalidIndex) return existing;

  if (typesFrozen_) {
    throw std::runtime_error("Type not pre-numbered before numbering pass");
  }

  // Insert into interner before encoding (nested types are numbered during encoding).
  size_t index = typeInterner_.intern(type, TypeInterner::kDialectVhlo);

  // Reserve a slot in the AttrType section so indices match interner order.
  size_t vhloDialectId = dialects_.addDialect("vhlo");
  if (index == attrTypes_.numTypes()) {
    size_t added = attrTypes_.addType({}, vhloDialectId, true);
    assert(added == index && "type index mismatch when reserving slot");
  }

  // Encode the type and add to AttrType section.
  std::vector<uint8_t> encoded = encodeTypePayload(type);

  attrTypes_.setTypePayload(index, std::move(encoded));
  return index;
}

void NumberingState::preNumberTypes(const WireModule& module) {
  TypeUsageCollector collector;
  collectTypesFromOp(module.rootOp, collector);
  auto ordered = collector.ordered();

  size_t vhloDialectId = dialects_.addDialect("vhlo");
  for (const auto& info : ordered) {
    size_t index = typeInterner_.intern(info.type, TypeInterner::kDialectVhlo);
    if (index == attrTypes_.numTypes()) {
      size_t added = attrTypes_.addType({}, vhloDialectId, true);
      assert(added == index && "type index mismatch when reserving slot");
    }
  }
  typesFrozen_ = true;
}

void NumberingState::preCollectAttrs(const WireModule& module) {
  AttrUsageCollector collector;
  collectAttrsFromOp(module.rootOp, collector);
  auto ordered = collector.orderedByRefCount();
  preCollectedAttrs_.clear();
  preCollectedAttrs_.reserve(ordered.size());
  for (const auto& info : ordered) {
    preCollectedAttrs_.push_back(
        AttrUsageEntry{info.count, info.order, info.attr});
  }
}

void NumberingState::preInternBuiltinAttrs(const WireModule& module) {
  const WireOp& root = module.rootOp;
  if (root.dialectName != "builtin" || root.opName != "module") return;

  if (root.symName.has_value()) {
    size_t strIdx = strings_.intern(*root.symName);
    internBuiltinAttr(encodeBuiltinStringAttr(strIdx));
  }
  if (root.symVisibility.has_value()) {
    size_t strIdx = strings_.intern(*root.symVisibility);
    internBuiltinAttr(encodeBuiltinStringAttr(strIdx));
  }
}

void NumberingState::finalizeAttrOrder() {
  if (preCollectedAttrs_.empty()) {
    attrsFrozen_ = true;
    return;
  }

  // 1) Primary ordering: refCount DESC, then firstSeen ASC.
  std::vector<AttrUsageEntry> ordered = preCollectedAttrs_;
  std::stable_sort(ordered.begin(), ordered.end(),
                   [](const AttrUsageEntry& a, const AttrUsageEntry& b) {
                     if (a.refCount != b.refCount) return a.refCount > b.refCount;
                     return a.firstSeen < b.firstSeen;
                   });

  // 2) Secondary ordering: group by dialect within varint-byte buckets
  // (MLIR's groupByDialectPerByte behavior).
  // groupByDialectPerByte: no-op comparator is valid only because
  // attr collection is guarded to VHLO-only in collectAttrsFromOp.
  // If another dialect must participate here, replace this with real
  // dialect-grouping logic.
  auto varIntByteWidth = [](size_t value) -> size_t {
    if (value < (1ull << 7)) return 1;
    if (value < (1ull << 14)) return 2;
    if (value < (1ull << 21)) return 3;
    if (value < (1ull << 28)) return 4;
    if (value < (1ull << 35)) return 5;
    if (value < (1ull << 42)) return 6;
    if (value < (1ull << 49)) return 7;
    if (value < (1ull << 56)) return 8;
    return 9;
  };

  const size_t baseAttrIndex = attrTypes_.numAttrs();
  size_t begin = 0;
  while (begin < ordered.size()) {
    const size_t bucketWidth = varIntByteWidth(baseAttrIndex + begin);
    size_t end = begin;
    while (end < ordered.size() &&
           varIntByteWidth(baseAttrIndex + end) == bucketWidth) {
      ++end;
    }
    std::stable_sort(ordered.begin() + begin, ordered.begin() + end,
                     [](const AttrUsageEntry&, const AttrUsageEntry&) {
                       return false;  // single-dialect (vhlo) collector today
                     });
    begin = end;
  }

  // 3) Build internal VHLO attr order + internal->serialized mapping.
  attrInterner_ = AttrInterner();
  vhloAttrIndexToGlobal_.clear();
  size_t vhloDialectId = dialects_.addDialect("vhlo");

  for (const auto& info : ordered) {
    size_t local = attrInterner_.intern(info.attr, AttrInterner::kDialectVhlo);
    if (local == vhloAttrIndexToGlobal_.size()) {
      size_t global = attrTypes_.addAttr({}, vhloDialectId, true);
      vhloAttrIndexToGlobal_.push_back(global);
    }
  }

  // 4) Encode payloads now that all mapped indices are known.
  for (size_t i = 0; i < attrInterner_.size(); ++i) {
    const AttrPtr& attr = attrInterner_.get(i);
    std::vector<uint8_t> payload = encodeVhloAttr(*attr);
    attrTypes_.setAttrPayload(vhloAttrIndexToGlobal_[i], std::move(payload));
  }

  preCollectedAttrs_ = std::move(ordered);
  attrsFrozen_ = true;
}

void NumberingState::finalizeTypePayloads() {
  for (size_t i = 0; i < typeInterner_.size(); ++i) {
    const auto& type = typeInterner_.get(i);
    std::vector<uint8_t> encoded = encodeTypePayload(type);
    attrTypes_.setTypePayload(i, std::move(encoded));
  }
}

void NumberingState::preNumberOpNames(const WireModule& module) {
  OpNameUsageCollector collector;
  collectOpNamesFromOp(module.rootOp, collector);

  // Ensure builtin dialect first if present in module.
  if (module.rootOp.dialectName == "builtin") {
    ensureBuiltinDialect();
  }

  // Dialect ordering: builtin first, then others by name.
  std::vector<std::string> dialects = collector.dialectNames();
  dialects.erase(std::remove(dialects.begin(), dialects.end(), "builtin"),
                 dialects.end());
  std::sort(dialects.begin(), dialects.end());

  if (module.rootOp.dialectName == "builtin") {
    dialects_.addDialect("builtin");
  }
  for (const auto& d : dialects) {
    if (d != "builtin") dialects_.addDialect(d);
  }

  // Populate ops in refcount order per dialect.
  for (size_t d = 0; d < dialects_.numDialects(); ++d) {
    const auto& dialect = dialects_.getDialect(d);
    auto ordered = collector.orderedOpsForDialect(dialect.name);
    for (const auto& info : ordered) {
      std::string fullName = dialect.name + "." + info.opName;
      if (opNameIndices_.find(fullName) != opNameIndices_.end()) continue;
      dialects_.addOp(d, info.opName, true);
      size_t index = opNameIndices_.size();
      opNameIndices_[fullName] = index;
    }
  }
}

void NumberingState::preInternDialectAndOpNames() {
  // Intern dialect names and op names in the same order they will be emitted.
  for (size_t d = 0; d < dialects_.numDialects(); ++d) {
    const auto& dialect = dialects_.getDialect(d);
    strings_.intern(dialect.name);
  }
  for (size_t d = 0; d < dialects_.numDialects(); ++d) {
    const auto& dialect = dialects_.getDialect(d);
    for (const auto& op : dialect.ops) {
      strings_.intern(op.name);
    }
  }
}

size_t NumberingState::lookupTypeIndex(const TypePtr& type) const {
  return typeInterner_.lookup(type);
}

size_t NumberingState::getPropertiesId(const std::vector<uint8_t>& payload) {
  // Convert to string for dedup key
  std::string key(payload.begin(), payload.end());
  auto it = propertiesDedup_.find(key);
  if (it != propertiesDedup_.end()) {
    return it->second;
  }
  
  size_t id = propertiesTable_.size();
  propertiesTable_.push_back(payload);
  propertiesDedup_[key] = id;
  return id;
}

OpInfo NumberingState::toOpInfo(const WireOp& op) const {
  size_t cursor = 0;
  OpInfo info = toOpInfoImpl(op, cursor);
  if (cursor != propsIdPreorder_.size()) {
    throw std::runtime_error("Traversal divergence: props cursor mismatch");
  }
  return info;
}

OpInfo NumberingState::toOpInfoImpl(const WireOp& op, size_t& cursor) const {
  OpInfo info;
  
  // Look up op name index
  std::string fullName = op.dialectName + "." + op.opName;
  auto it = opNameIndices_.find(fullName);
  assert(it != opNameIndices_.end() && "op not numbered");
  info.opNameIndex = it->second;
  
  if (unknownLocAttrIndex_ == kInvalidIndex) {
    throw std::runtime_error("UnknownLoc attr not initialized before IR conversion");
  }
  // Ops always emit a location index in MLIR bytecode v6.
  info.locationIndex = unknownLocAttrIndex_;
  
  if (cursor >= propsIdPreorder_.size()) {
    throw std::runtime_error("Traversal divergence: props cursor out of bounds");
  }
  // INVARIANT: consume exactly one slot per visited op in the same preorder
  // as numberOp() pushes propsIdPreorder_ entries.
  const size_t propsId = propsIdPreorder_[cursor++];
  if (propsId != kInvalidIndex) {
    info.propertiesId = propsId;
  }
  
  // Result types
  for (const auto& type : op.resultTypes) {
    size_t idx = lookupTypeIndex(type);
    assert(idx != kInvalidIndex && "type not numbered");
    info.resultTypeIndices.push_back(idx);
  }
  
  // Operands (these are already value indices)
  info.operandValueIndices = op.operands;
  
  // Successors
  info.successorBlockIndices = op.successors;
  
  // Regions
  info.isIsolatedFromAbove = op.isIsolatedFromAbove;
  for (const auto& region : op.regions) {
    info.regions.push_back(toRegionInfo(region, cursor));
  }
  
  return info;
}

BlockInfo NumberingState::toBlockInfo(const WireBlock& block, size_t& cursor) const {
  BlockInfo info;
  
  // Block args
  for (const auto& arg : block.args) {
    BlockArgInfo argInfo;
    size_t idx = lookupTypeIndex(arg.type);
    assert(idx != kInvalidIndex && "type not numbered");
    argInfo.typeIndex = idx;
    argInfo.hasLoc = false;
    argInfo.locationIndex = kInvalidIndex;
    info.args.push_back(argInfo);
  }
  
  // Ops
  for (const auto& op : block.ops) {
    info.ops.push_back(toOpInfoImpl(op, cursor));
  }
  
  return info;
}

RegionInfo NumberingState::toRegionInfo(const WireRegion& region, size_t& cursor) const {
  RegionInfo info;
  for (const auto& block : region.blocks) {
    info.blocks.push_back(toBlockInfo(block, cursor));
  }
  return info;
}

std::vector<uint8_t> NumberingState::buildFuncV1Props(const TypePtr& funcType,
                                                      const std::string& funcName,
                                                      const std::string& symVisibility) {
  // func_v1 properties blob = 5 varints of attr indices
  // Order (golden-proven):
  //   [arg_attrs, function_type, res_attrs, sym_name, sym_visibility]
  //   = [Array(empty), TypeAttr(funcType), Array(empty), StringAttr("main"), StringAttr("")]
  
  // Create the attrs we need
  const size_t arrEmpty = internVhloAttr(makeArrayAttr({}));
  const size_t typeAttr = internVhloAttr(makeTypeAttr(funcType));
  const size_t symName = internVhloAttr(makeStringAttr(funcName));
  const size_t symVis = internVhloAttr(makeStringAttr(symVisibility));
  
  // Build the properties blob as varints of attr indices
  EncodingEmitter e;
  e.emitVarInt(arrEmpty);   // arg_attrs
  e.emitVarInt(typeAttr);   // function_type
  e.emitVarInt(arrEmpty);   // res_attrs
  e.emitVarInt(symName);    // sym_name
  e.emitVarInt(symVis);     // sym_visibility
  return e.bytes();
}

std::vector<uint8_t> NumberingState::buildModuleProps(const WireOp& op) {
  // Module properties are optional attributes:
  //   sym_name?, sym_visibility?
  // Encoding uses VarIntWithFlag(attrIndex, present).
  EncodingEmitter e;

  if (op.symName.has_value()) {
    size_t strIdx = strings_.intern(*op.symName);
    size_t attrIdx = internBuiltinAttr(encodeBuiltinStringAttr(strIdx));
    e.emitVarIntWithFlag(attrIdx, true);
  } else {
    e.emitVarIntWithFlag(0, false);
  }

  if (op.symVisibility.has_value()) {
    size_t strIdx = strings_.intern(*op.symVisibility);
    size_t attrIdx = internBuiltinAttr(encodeBuiltinStringAttr(strIdx));
    e.emitVarIntWithFlag(attrIdx, true);
  } else {
    e.emitVarIntWithFlag(0, false);
  }

  return e.bytes();
}

std::vector<uint8_t> NumberingState::buildCompareV1Props(const CompareProps& props) {
  // Attribute order is alphabetical: compare_type, comparison_direction.
  const size_t compareType = internVhloAttr(
      makeComparisonTypeAttr(props.comparisonType));
  const size_t compareDir = internVhloAttr(
      makeComparisonDirectionAttr(props.comparisonDirection));

  EncodingEmitter e;
  e.emitVarInt(compareType);
  e.emitVarInt(compareDir);
  return e.bytes();
}

std::vector<uint8_t> NumberingState::buildGatherV1Props(const GatherProps& props) {
  // Attribute order is alphabetical:
  //   collapsed_slice_dims, index_vector_dim, indices_are_sorted,
  //   offset_dims, slice_sizes, start_index_map.
  const size_t collapsed = internI64TensorAttr(props.collapsedSliceDims);
  const size_t indexVec = internI64IntegerAttr(props.indexVectorDim);
  const size_t sorted = internVhloAttr(makeBooleanAttr(props.indicesAreSorted));
  const size_t offset = internI64TensorAttr(props.offsetDims);
  const size_t sliceSizes = internI64TensorAttr(props.sliceSizes);
  const size_t startMap = internI64TensorAttr(props.startIndexMap);

  EncodingEmitter e;
  e.emitVarInt(collapsed);
  e.emitVarInt(indexVec);
  e.emitVarInt(sorted);
  e.emitVarInt(offset);
  e.emitVarInt(sliceSizes);
  e.emitVarInt(startMap);
  return e.bytes();
}

std::vector<uint8_t> NumberingState::buildConstantV1Props(
    const WireOp& op, const ConstantProps& props) {
  if (op.resultTypes.empty()) return {};
  auto valueAttrObj = makeDenseElementsAttr(op.resultTypes[0], props.rawData);
  size_t valueAttr = internVhloAttr(valueAttrObj);
  EncodingEmitter e;
  e.emitVarInt(valueAttr);
  return e.bytes();
}

std::vector<uint8_t> NumberingState::buildReduceV1Props(const ReduceProps& props) {
  size_t dimsAttr = internI64TensorAttr(props.dimensions);
  EncodingEmitter e;
  e.emitVarInt(dimsAttr);
  return e.bytes();
}

std::vector<uint8_t> NumberingState::buildTransposeV1Props(
    const TransposeProps& props) {
  size_t permAttr = internI64TensorAttr(props.permutation);
  EncodingEmitter e;
  e.emitVarInt(permAttr);
  return e.bytes();
}

std::vector<uint8_t> NumberingState::buildBroadcastInDimV1Props(
    const BroadcastInDimProps& props) {
  size_t dimsAttr = internI64TensorAttr(props.broadcastDimensions);
  EncodingEmitter e;
  e.emitVarInt(dimsAttr);
  return e.bytes();
}

std::vector<uint8_t> NumberingState::buildConcatenateV1Props(
    const ConcatenateProps& props) {
  size_t dimAttr = internI64IntegerAttr(props.dimension);
  EncodingEmitter e;
  e.emitVarInt(dimAttr);
  return e.bytes();
}

std::vector<uint8_t> NumberingState::buildSliceV1Props(
    const SliceProps& props) {
  // Attribute order is alphabetical:
  //   limit_indices, start_indices, strides.
  size_t limitAttr = internI64TensorAttr(props.limitIndices);
  size_t startAttr = internI64TensorAttr(props.startIndices);
  size_t stridesAttr = internI64TensorAttr(props.strides);
  EncodingEmitter e;
  e.emitVarInt(limitAttr);
  e.emitVarInt(startAttr);
  e.emitVarInt(stridesAttr);
  return e.bytes();
}

std::vector<uint8_t> NumberingState::buildDotGeneralV1Props(
    const DotGeneralProps& props) {
  // Attribute order is alphabetical:
  //   lhs_batching_dimensions, lhs_contracting_dimensions, precision_config,
  //   rhs_batching_dimensions, rhs_contracting_dimensions.
  const size_t lhsBatchAttr = internI64TensorAttr(props.lhsBatchingDimensions);
  const size_t lhsContractAttr =
      internI64TensorAttr(props.lhsContractingDimensions);

  std::vector<AttrPtr> precisionElems;
  precisionElems.reserve(props.precisionConfig.size());
  for (uint64_t value : props.precisionConfig) {
    precisionElems.push_back(makePrecisionAttr(value));
  }
  const size_t precisionAttr =
      internVhloAttr(makeArrayAttr(std::move(precisionElems)));

  const size_t rhsBatchAttr = internI64TensorAttr(props.rhsBatchingDimensions);
  const size_t rhsContractAttr =
      internI64TensorAttr(props.rhsContractingDimensions);

  EncodingEmitter e;
  e.emitVarInt(lhsBatchAttr);
  e.emitVarInt(lhsContractAttr);
  e.emitVarInt(precisionAttr);
  e.emitVarInt(rhsBatchAttr);
  e.emitVarInt(rhsContractAttr);
  return e.bytes();
}

std::vector<uint8_t> NumberingState::buildConvolutionV1Props(
    const ConvolutionProps& props) {
  const size_t spatialRank = inferConvolutionSpatialRank(props);
  const auto paddingShape = inferConvolutionPaddingShape(props, spatialRank);
  const auto windowReversal = inferConvolutionWindowReversal(props, spatialRank);
  const auto precisionConfig = inferConvolutionPrecisionConfig(props);

  // Attribute order is alphabetical:
  // batch_group_count, feature_group_count,
  // input_batch_dimension, input_feature_dimension, input_spatial_dimensions,
  // kernel_input_feature_dimension, kernel_output_feature_dimension,
  // kernel_spatial_dimensions, lhs_dilation,
  // output_batch_dimension, output_feature_dimension, output_spatial_dimensions,
  // padding, precision_config, rhs_dilation, window_reversal, window_strides.
  const size_t batchGroupCount = internI64IntegerAttr(props.batchGroupCount);
  const size_t featureGroupCount = internI64IntegerAttr(props.featureGroupCount);
  const size_t inputBatchDim = internI64IntegerAttr(props.inputBatchDimension);
  const size_t inputFeatureDim = internI64IntegerAttr(props.inputFeatureDimension);
  const size_t inputSpatialDims = internI64TensorAttr(props.inputSpatialDimensions);
  const size_t kernelInputFeatureDim =
      internI64IntegerAttr(props.kernelInputFeatureDimension);
  const size_t kernelOutputFeatureDim =
      internI64IntegerAttr(props.kernelOutputFeatureDimension);
  const size_t kernelSpatialDims = internI64TensorAttr(props.kernelSpatialDimensions);
  const size_t lhsDilation = internI64TensorAttr(props.lhsDilation);
  const size_t outputBatchDim = internI64IntegerAttr(props.outputBatchDimension);
  const size_t outputFeatureDim = internI64IntegerAttr(props.outputFeatureDimension);
  const size_t outputSpatialDims = internI64TensorAttr(props.outputSpatialDimensions);
  const size_t padding = internI64TensorAttrWithShape(paddingShape, props.padding);

  std::vector<AttrPtr> precisionElems;
  precisionElems.reserve(precisionConfig.size());
  for (uint64_t value : precisionConfig) {
    precisionElems.push_back(makePrecisionAttr(value));
  }
  const size_t precisionAttr =
      internVhloAttr(makeArrayAttr(std::move(precisionElems)));

  const size_t rhsDilation = internI64TensorAttr(props.rhsDilation);
  const size_t windowReversalAttr = internBoolTensorAttrWithShape(
      {static_cast<int64_t>(spatialRank)}, windowReversal);
  const size_t windowStrides = internI64TensorAttr(props.windowStrides);

  EncodingEmitter e;
  e.emitVarInt(batchGroupCount);
  e.emitVarInt(featureGroupCount);
  e.emitVarInt(inputBatchDim);
  e.emitVarInt(inputFeatureDim);
  e.emitVarInt(inputSpatialDims);
  e.emitVarInt(kernelInputFeatureDim);
  e.emitVarInt(kernelOutputFeatureDim);
  e.emitVarInt(kernelSpatialDims);
  e.emitVarInt(lhsDilation);
  e.emitVarInt(outputBatchDim);
  e.emitVarInt(outputFeatureDim);
  e.emitVarInt(outputSpatialDims);
  e.emitVarInt(padding);
  e.emitVarInt(precisionAttr);
  e.emitVarInt(rhsDilation);
  e.emitVarInt(windowReversalAttr);
  e.emitVarInt(windowStrides);
  return e.bytes();
}

std::string NumberingState::dump() const {
  std::ostringstream ss;
  ss << "=== NumberingState ===\n";
  
  // Strings (in insertion order)
  ss << "Strings (" << strings_.size() << "):\n";
  for (size_t i = 0; i < strings_.size(); ++i) {
    ss << "  [" << i << "] \"" << strings_.get(i) << "\"\n";
  }
  
  // OpNames (sorted by index for stable output)
  ss << "OpNames (" << opNameIndices_.size() << "):\n";
  std::vector<std::pair<size_t, std::string>> ops;
  ops.reserve(opNameIndices_.size());
  for (const auto& kv : opNameIndices_) {
    ops.push_back({kv.second, kv.first});
  }
  std::sort(ops.begin(), ops.end());
  for (const auto& it : ops) {
    ss << "  [" << it.first << "] " << it.second << "\n";
  }
  
  // Types (sorted by index)
  const auto& types = typeInterner_.types();
  ss << "Types (" << types.size() << "):\n";
  for (size_t i = 0; i < types.size(); ++i) {
    size_t h = types[i] ? types[i]->hash() : 0;
    ss << "  [" << i << "] hash=" << h << "\n";
  }
  
  // Properties (with lengths)
  ss << "Properties (" << propertiesTable_.size() << "):\n";
  for (size_t i = 0; i < propertiesTable_.size(); ++i) {
    ss << "  [" << i << "] len=" << propertiesTable_[i].size() << "\n";
  }
  
  return ss.str();
}

// =============================================================================
// BytecodeAssembler Implementation
// =============================================================================

std::vector<uint8_t> BytecodeAssembler::assemble(const WireModule& module) {
  // 1. Number the module (populates interners)
  numbering_.number(module);
  
  // 2. Build Section 1: Dialect (needs StringInterner for op names)
  // Note: write() interns the strings, so call it first
  auto dialectBytes = numbering_.dialects().toBytes(numbering_.strings());
  
  // 3. Build Section 0: String (after dialect so names are interned)
  auto stringBytes = numbering_.strings().toBytes();
  
  // 4. Build Section 3: AttrTypeOffset
  auto attrTypeOffsetBytes = numbering_.attrTypes().toOffsetBytes();
  
  // 5. Build Section 2: AttrType
  auto attrTypeBytes = numbering_.attrTypes().toAttrTypeBytes();
  
  // 6. Build Section 8: Properties
  std::vector<uint8_t> propertiesBytes;
  {
    EncodingEmitter emitter;
    const auto& propsTable = numbering_.propertiesTable();
    // Properties section format: numProperties, then each property (with length prefix)
    emitter.emitVarInt(propsTable.size());
    for (const auto& prop : propsTable) {
      emitter.emitVarInt(prop.size());
      emitter.emitBytes(prop);
    }
    propertiesBytes = emitter.bytes();
  }
  
  // 7. Build Section 4: IR
  IRSectionWriter irWriter;
  irWriter.setRootOp(numbering_.toOpInfo(module.rootOp));
  auto irBytes = irWriter.toBytes();

  // 7.5 Build Sections 6/5: ResourceOffset/Resource
  // Goldens include empty resource sections even when no resources exist.
  std::vector<uint8_t> resourceOffsetBytes;
  std::vector<uint8_t> resourceBytes;
  {
    EncodingEmitter emitter;
    emitter.emitVarInt(0);  // numResources = 0
    resourceOffsetBytes = emitter.bytes();
    resourceBytes = {};     // empty payload
  }

  // 8. Assemble all sections using BytecodeWriter
  BytecodeWriter writer;
  writer.writeHeader("StableHLO_v1.0.0");

  // Pinned/golden-style order:
  // 1 Dialect, 3 AttrTypeOffset, 2 AttrType, 4 IR, (6/5 Resource*), 0 String, 8 Properties
  writer.writeSection(Section::kDialect, dialectBytes);
  writer.writeSection(Section::kAttrTypeOffset, attrTypeOffsetBytes);
  writer.writeSection(Section::kAttrType, attrTypeBytes);
  writer.writeSection(Section::kIR, irBytes);
  writer.writeSection(Section::kResourceOffset, resourceOffsetBytes);
  writer.writeSection(Section::kResource, resourceBytes);
  writer.writeSection(Section::kString, stringBytes);
  if (!propertiesBytes.empty() && propertiesBytes.size() > 1) {
    writer.writeSection(Section::kProperties, propertiesBytes);
  }
  
  return writer.bytes();
}

}  // namespace stablehlo
}  // namespace lczero
