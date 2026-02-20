// =============================================================================
// type_interner.cc - TypeInterner Implementation
// =============================================================================

#include "stablehlo/type_interner.h"
#include "stablehlo/index_constants.h"

#include <algorithm>
#include <cassert>
#include <map>

namespace lczero {
namespace stablehlo {

size_t TypeInterner::intern(TypePtr type) {
  // CRITICAL: For StableHLO portable artifacts, ALL types should be in the VHLO
  // dialect, including function types. The goldens show zero builtin-grouped types.
  // This is because function types in VHLO are encoded as FunctionV1Type.
  size_t dialect_id = kDialectVhlo;
  return intern(std::move(type), dialect_id);
}

size_t TypeInterner::intern(TypePtr type, size_t dialect_id) {
  // Check if already interned.
  auto it = index_map_.find(type);
  if (it != index_map_.end()) {
    return it->second;
  }

  // Add new type.
  size_t index = types_.size();
  types_.push_back(type);
  dialect_ids_.push_back(dialect_id);
  index_map_[type] = index;

  return index;
}

size_t TypeInterner::lookup(const TypePtr& type) const {
  auto it = index_map_.find(type);
  if (it != index_map_.end()) {
    return it->second;
  }
  return kInvalidIndex;
}

const TypePtr& TypeInterner::get(size_t index) const {
  assert(index < types_.size());
  return types_[index];
}

std::vector<std::pair<size_t, std::vector<size_t>>> TypeInterner::getGroupedByDialect() const {
  // Group indices by dialect ID.
  std::map<size_t, std::vector<size_t>> groups;
  for (size_t i = 0; i < types_.size(); ++i) {
    groups[dialect_ids_[i]].push_back(i);
  }

  // Convert to sorted vector of pairs.
  std::vector<std::pair<size_t, std::vector<size_t>>> result;
  result.reserve(groups.size());
  for (auto& [dialect_id, indices] : groups) {
    result.emplace_back(dialect_id, std::move(indices));
  }
  return result;
}

}  // namespace stablehlo
}  // namespace lczero
