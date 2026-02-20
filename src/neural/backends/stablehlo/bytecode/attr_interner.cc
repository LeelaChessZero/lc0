// =============================================================================
// attr_interner.cc - AttrInterner Implementation
// =============================================================================

#include "stablehlo/attr_interner.h"
#include "stablehlo/index_constants.h"

#include <algorithm>
#include <cassert>
#include <map>

namespace lczero {
namespace stablehlo {

size_t AttrInterner::intern(AttrPtr attr) {
  // Default to VHLO dialect for most attributes.
  // Some builtin attrs might use kDialectBuiltin.
  size_t dialect_id = kDialectVhlo;
  return intern(std::move(attr), dialect_id);
}

size_t AttrInterner::intern(AttrPtr attr, size_t dialect_id) {
  // Check if already interned.
  auto it = index_map_.find(attr);
  if (it != index_map_.end()) {
    return it->second;
  }

  // Add new attribute.
  size_t index = attrs_.size();
  attrs_.push_back(attr);
  dialect_ids_.push_back(dialect_id);
  index_map_[attr] = index;

  return index;
}

size_t AttrInterner::lookup(const AttrPtr& attr) const {
  auto it = index_map_.find(attr);
  if (it != index_map_.end()) {
    return it->second;
  }
  return kInvalidIndex;
}

const AttrPtr& AttrInterner::get(size_t index) const {
  assert(index < attrs_.size());
  return attrs_[index];
}

std::vector<std::pair<size_t, std::vector<size_t>>> AttrInterner::getGroupedByDialect() const {
  // Group indices by dialect ID.
  std::map<size_t, std::vector<size_t>> groups;
  for (size_t i = 0; i < attrs_.size(); ++i) {
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
