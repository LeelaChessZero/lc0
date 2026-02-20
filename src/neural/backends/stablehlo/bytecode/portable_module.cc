// =============================================================================
// portable_module.cc - Wire-Level Structure Implementations
// =============================================================================

#include "stablehlo/portable_module.h"

#include <sstream>

namespace lczero {
namespace stablehlo {

// Helper for indentation
static std::string indent(int level) {
  return std::string(level * 2, ' ');
}

std::string WireOp::toString(int level) const {
  std::ostringstream ss;
  std::string ind = indent(level);
  
  ss << ind << "WireOp {\n";
  ss << ind << "  name: " << dialectName << "." << opName << "\n";
  ss << ind << "  locationIndex: " << locationIndex << "\n";
  
  if (propertiesPayload.has_value()) {
    ss << ind << "  propertiesPayload: " << propertiesPayload->size() << " bytes\n";
  }

  if (symName.has_value()) {
    ss << ind << "  symName: \"" << *symName << "\"\n";
  }
  if (symVisibility.has_value()) {
    ss << ind << "  symVisibility: \"" << *symVisibility << "\"\n";
  }

  if (compareProps.has_value()) {
    ss << ind << "  compareProps: type=" << compareProps->comparisonType
       << " dir=" << compareProps->comparisonDirection << "\n";
  }
  if (gatherProps.has_value()) {
    ss << ind << "  gatherProps: offset=" << gatherProps->offsetDims.size()
       << " collapsed=" << gatherProps->collapsedSliceDims.size()
       << " start_index_map=" << gatherProps->startIndexMap.size()
       << " slice_sizes=" << gatherProps->sliceSizes.size()
       << " index_vector_dim=" << gatherProps->indexVectorDim
       << " sorted=" << (gatherProps->indicesAreSorted ? "true" : "false") << "\n";
  }
  if (constantProps.has_value()) {
    ss << ind << "  constantProps: rawData=" << constantProps->rawData.size() << " bytes\n";
  }
  if (reduceProps.has_value()) {
    ss << ind << "  reduceProps: dims=" << reduceProps->dimensions.size() << "\n";
  }
  
  if (!resultTypes.empty()) {
    ss << ind << "  resultTypes: [";
    for (size_t i = 0; i < resultTypes.size(); ++i) {
      if (i > 0) ss << ", ";
      ss << resultTypes[i]->toString();
    }
    ss << "]\n";
  }
  
  if (!operands.empty()) {
    ss << ind << "  operands: [";
    for (size_t i = 0; i < operands.size(); ++i) {
      if (i > 0) ss << ", ";
      ss << "%" << operands[i];
    }
    ss << "]\n";
  }
  
  if (!successors.empty()) {
    ss << ind << "  successors: [";
    for (size_t i = 0; i < successors.size(); ++i) {
      if (i > 0) ss << ", ";
      ss << "^bb" << successors[i];
    }
    ss << "]\n";
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

std::string WireBlock::toString(int level) const {
  std::ostringstream ss;
  std::string ind = indent(level);
  
  if (!args.empty()) {
    ss << ind << "args: [";
    for (size_t i = 0; i < args.size(); ++i) {
      if (i > 0) ss << ", ";
      ss << "%" << i << ": " << args[i].type->toString();
      if (args[i].hasLoc) {
        ss << " @loc" << args[i].locationIndex;
      }
    }
    ss << "]\n";
  }
  
  ss << ind << "ops: [\n";
  size_t valueOffset = args.size();
  for (const auto& op : ops) {
    ss << op.toString(level + 1) << "\n";
    // Show what values this op defines
    if (!op.resultTypes.empty()) {
      ss << ind << "  // defines: ";
      for (size_t j = 0; j < op.resultTypes.size(); ++j) {
        if (j > 0) ss << ", ";
        ss << "%" << (valueOffset + j);
      }
      ss << "\n";
      valueOffset += op.resultTypes.size();
    }
  }
  ss << ind << "]\n";
  
  return ss.str();
}

std::string WireRegion::toString(int level) const {
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

std::string WireModule::toString() const {
  std::ostringstream ss;
  ss << "=== WireModule ===\n";
  ss << rootOp.toString(0) << "\n";
  return ss.str();
}

}  // namespace stablehlo
}  // namespace lczero
