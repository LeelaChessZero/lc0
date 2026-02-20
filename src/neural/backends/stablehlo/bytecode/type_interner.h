// =============================================================================
// type_interner.h - Type Interner for MLIR Bytecode
// =============================================================================
// Deduplicates types by structural equality and assigns stable indices.
// Types are grouped by dialect for the AttrTypeOffset section (M4).

#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

#include "stablehlo/types.h"

namespace lczero {
namespace stablehlo {

// =============================================================================
// TypeInterner - Deduplicates types and assigns stable indices
// =============================================================================
class TypeInterner {
 public:
  // Dialect ID for grouping (used in AttrTypeOffset section).
  // 0 = builtin, 1 = vhlo, etc.
  static constexpr size_t kDialectBuiltin = 0;
  static constexpr size_t kDialectVhlo = 1;

  TypeInterner() = default;

  // Intern a type and return its index.
  // If the type was already interned (structurally equal), returns existing index.
  // Indices are assigned in order of first insertion.
  size_t intern(TypePtr type);

  // Intern a type with explicit dialect ID.
  size_t intern(TypePtr type, size_t dialect_id);

  // Get index of an already-interned type.
  // Returns kInvalidIndex if not found.
  size_t lookup(const TypePtr& type) const;

  // Get type by index.
  const TypePtr& get(size_t index) const;

  // Number of interned types.
  size_t size() const { return types_.size(); }

  // Check if empty.
  bool empty() const { return types_.empty(); }

  // Get dialect ID for a type index.
  size_t dialectId(size_t index) const { return dialect_ids_[index]; }

  // Get types grouped by dialect (for AttrTypeOffset section).
  // Returns vector of (dialect_id, indices) pairs, sorted by dialect_id.
  std::vector<std::pair<size_t, std::vector<size_t>>> getGroupedByDialect() const;

  // Iterator access for debugging/testing.
  const std::vector<TypePtr>& types() const { return types_; }

 private:
  // Custom hash for TypePtr using structural hash.
  struct TypePtrHash {
    size_t operator()(const TypePtr& t) const {
      return t ? t->hash() : 0;
    }
  };

  // Custom equality for TypePtr using structural equality.
  struct TypePtrEqual {
    bool operator()(const TypePtr& a, const TypePtr& b) const {
      if (!a || !b) return a == b;
      return a->equals(*b);
    }
  };

  // Types in insertion order (determines indices).
  std::vector<TypePtr> types_;

  // Dialect ID for each type (parallel to types_).
  std::vector<size_t> dialect_ids_;

  // Map from type to index for O(1) lookup.
  std::unordered_map<TypePtr, size_t, TypePtrHash, TypePtrEqual> index_map_;
};

}  // namespace stablehlo
}  // namespace lczero
