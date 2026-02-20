// =============================================================================
// attr_interner.h - Attribute Interner for MLIR Bytecode
// =============================================================================
// Deduplicates attributes by structural equality and assigns stable indices.
// Attributes are grouped by dialect for the AttrTypeOffset section (M4).

#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

#include "stablehlo/attrs.h"

namespace lczero {
namespace stablehlo {

// =============================================================================
// AttrInterner - Deduplicates attributes and assigns stable indices
// =============================================================================
class AttrInterner {
 public:
  // Dialect ID for grouping (used in AttrTypeOffset section).
  static constexpr size_t kDialectBuiltin = 0;
  static constexpr size_t kDialectVhlo = 1;

  AttrInterner() = default;

  // Intern an attribute and return its index.
  // If the attribute was already interned (structurally equal), returns existing index.
  // Indices are assigned in order of first insertion.
  size_t intern(AttrPtr attr);

  // Intern an attribute with explicit dialect ID.
  size_t intern(AttrPtr attr, size_t dialect_id);

  // Get index of an already-interned attribute.
  // Returns kInvalidIndex if not found.
  size_t lookup(const AttrPtr& attr) const;

  // Get attribute by index.
  const AttrPtr& get(size_t index) const;

  // Number of interned attributes.
  size_t size() const { return attrs_.size(); }

  // Check if empty.
  bool empty() const { return attrs_.empty(); }

  // Get dialect ID for an attribute index.
  size_t dialectId(size_t index) const { return dialect_ids_[index]; }

  // Get attributes grouped by dialect (for AttrTypeOffset section).
  // Returns vector of (dialect_id, indices) pairs, sorted by dialect_id.
  std::vector<std::pair<size_t, std::vector<size_t>>> getGroupedByDialect() const;

  // Iterator access for debugging/testing.
  const std::vector<AttrPtr>& attrs() const { return attrs_; }

 private:
  // Custom hash for AttrPtr using structural hash.
  struct AttrPtrHash {
    size_t operator()(const AttrPtr& a) const {
      return a ? a->hash() : 0;
    }
  };

  // Custom equality for AttrPtr using structural equality.
  struct AttrPtrEqual {
    bool operator()(const AttrPtr& a, const AttrPtr& b) const {
      if (!a || !b) return a == b;
      return a->equals(*b);
    }
  };

  // Attributes in insertion order (determines indices).
  std::vector<AttrPtr> attrs_;

  // Dialect ID for each attribute (parallel to attrs_).
  std::vector<size_t> dialect_ids_;

  // Map from attribute to index for O(1) lookup.
  std::unordered_map<AttrPtr, size_t, AttrPtrHash, AttrPtrEqual> index_map_;
};

}  // namespace stablehlo
}  // namespace lczero
