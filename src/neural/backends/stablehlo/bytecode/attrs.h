// =============================================================================
// attrs.h - VHLO Attribute Definitions for Bytecode
// =============================================================================
// Attribute codes from VhloBytecode.cpp. These are the wire-level codes
// used in MLIR bytecode for StableHLO/VHLO.
//
// M2 scope: Define attrs for structural hashing and stable numbering.
// M4 scope: Actual encoding into AttrType section.

#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "stablehlo/types.h"

namespace lczero {
namespace stablehlo {

// =============================================================================
// VHLO Attribute Codes - VERIFIED from golden .mlirbc files
// =============================================================================
// These codes are proven by byte-level analysis of golden bytecode files.
// Use these for any new encoding helpers.
namespace VhloAttrCode {
  inline constexpr uint64_t kArrayV1               = 1;   // e.g. 0301
  inline constexpr uint64_t kBooleanV1             = 2;   // gather_i32: 0501
  inline constexpr uint64_t kComparisonDirectionV1 = 3;   // compare_select: 0707
  inline constexpr uint64_t kComparisonTypeV1      = 4;   // compare_select: 0901
  inline constexpr uint64_t kPrecisionV1           = 11;  // dot_general: precision enum
  inline constexpr uint64_t kIntegerV1             = 9;   // gather_i32: 130105
  inline constexpr uint64_t kStringV1              = 14;  // e.g. 1d11, 1d13
  inline constexpr uint64_t kTensorV1              = 15;  // reduce_add: 1f0109...
  inline constexpr uint64_t kTypeV1                = 17;  // e.g. 2303, 2305
}  // namespace VhloAttrCode

// =============================================================================
// Builtin Attribute Codes (from BuiltinDialectBytecode.td)
// =============================================================================
// Only include codes we actively use.
namespace BuiltinAttrCode {
  inline constexpr uint64_t kStringAttr = 2;  // builtin::StringAttr
}  // namespace BuiltinAttrCode

// Forward declarations.
class Attr;
using AttrPtr = std::shared_ptr<const Attr>;

// =============================================================================
// Attr - Base class for all attributes
// =============================================================================
enum class AttrKind : uint8_t {
  kString,
  kInteger,
  kFloat,
  kBoolean,
  kArray,
  kType,
  kDenseElements,
  kComparisonDirection,
  kComparisonType,
  kPrecision,
  kUnit,
  kDictionary,
  kSymbolRef,
};

class Attr {
 public:
  virtual ~Attr() = default;

  AttrKind kind() const { return kind_; }

  // Structural equality.
  virtual bool equals(const Attr& other) const = 0;

  // Structural hash.
  virtual size_t hash() const = 0;

  // Debug string.
  virtual std::string toString() const = 0;

 protected:
  explicit Attr(AttrKind kind) : kind_(kind) {}

 private:
  AttrKind kind_;
};

// =============================================================================
// StringAttr - String attribute
// =============================================================================
class StringAttr : public Attr {
 public:
  explicit StringAttr(std::string value)
      : Attr(AttrKind::kString), value_(std::move(value)) {}

  const std::string& value() const { return value_; }

  bool equals(const Attr& other) const override {
    if (other.kind() != AttrKind::kString) return false;
    return value_ == static_cast<const StringAttr&>(other).value_;
  }

  size_t hash() const override {
    return std::hash<std::string>{}(value_);
  }

  std::string toString() const override {
    return "\"" + value_ + "\"";
  }

 private:
  std::string value_;
};

// =============================================================================
// IntegerAttr - Integer attribute with type
// =============================================================================
class IntegerAttr : public Attr {
 public:
  IntegerAttr(int64_t value, TypePtr type)
      : Attr(AttrKind::kInteger), value_(value), type_(std::move(type)) {}

  int64_t value() const { return value_; }
  const TypePtr& type() const { return type_; }

  bool equals(const Attr& other) const override {
    if (other.kind() != AttrKind::kInteger) return false;
    const auto& o = static_cast<const IntegerAttr&>(other);
    return value_ == o.value_ && type_->equals(*o.type_);
  }

  size_t hash() const override {
    size_t h = std::hash<int64_t>{}(value_);
    h ^= type_->hash() + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }

  std::string toString() const override {
    return std::to_string(value_) + " : " + type_->toString();
  }

 private:
  int64_t value_;
  TypePtr type_;
};

// =============================================================================
// FloatAttr - Float attribute with type
// =============================================================================
class FloatAttr : public Attr {
 public:
  FloatAttr(double value, TypePtr type)
      : Attr(AttrKind::kFloat), value_(value), type_(std::move(type)) {}

  double value() const { return value_; }
  const TypePtr& type() const { return type_; }

  bool equals(const Attr& other) const override {
    if (other.kind() != AttrKind::kFloat) return false;
    const auto& o = static_cast<const FloatAttr&>(other);
    return value_ == o.value_ && type_->equals(*o.type_);
  }

  size_t hash() const override {
    size_t h = std::hash<double>{}(value_);
    h ^= type_->hash() + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }

  std::string toString() const override {
    return std::to_string(value_) + " : " + type_->toString();
  }

 private:
  double value_;
  TypePtr type_;
};

// =============================================================================
// BooleanAttr - Boolean attribute
// =============================================================================
class BooleanAttr : public Attr {
 public:
  explicit BooleanAttr(bool value)
      : Attr(AttrKind::kBoolean), value_(value) {}

  bool value() const { return value_; }

  bool equals(const Attr& other) const override {
    if (other.kind() != AttrKind::kBoolean) return false;
    return value_ == static_cast<const BooleanAttr&>(other).value_;
  }

  size_t hash() const override {
    return std::hash<bool>{}(value_);
  }

  std::string toString() const override {
    return value_ ? "true" : "false";
  }

 private:
  bool value_;
};

// =============================================================================
// ArrayAttr - Array of attributes
// =============================================================================
class ArrayAttr : public Attr {
 public:
  explicit ArrayAttr(std::vector<AttrPtr> elements)
      : Attr(AttrKind::kArray), elements_(std::move(elements)) {}

  const std::vector<AttrPtr>& elements() const { return elements_; }

  bool equals(const Attr& other) const override {
    if (other.kind() != AttrKind::kArray) return false;
    const auto& o = static_cast<const ArrayAttr&>(other);
    if (elements_.size() != o.elements_.size()) return false;
    for (size_t i = 0; i < elements_.size(); ++i) {
      if (!elements_[i]->equals(*o.elements_[i])) return false;
    }
    return true;
  }

  size_t hash() const override {
    size_t h = 0xABCDEF01;
    for (const auto& e : elements_) {
      h ^= e->hash() + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    return h;
  }

  std::string toString() const override {
    std::string result = "[";
    for (size_t i = 0; i < elements_.size(); ++i) {
      if (i > 0) result += ", ";
      result += elements_[i]->toString();
    }
    result += "]";
    return result;
  }

 private:
  std::vector<AttrPtr> elements_;
};

// =============================================================================
// TypeAttr - Type as attribute
// =============================================================================
class TypeAttr : public Attr {
 public:
  explicit TypeAttr(TypePtr type)
      : Attr(AttrKind::kType), type_(std::move(type)) {}

  const TypePtr& type() const { return type_; }

  bool equals(const Attr& other) const override {
    if (other.kind() != AttrKind::kType) return false;
    return type_->equals(*static_cast<const TypeAttr&>(other).type_);
  }

  size_t hash() const override {
    return type_->hash() ^ 0x13579BDF;
  }

  std::string toString() const override {
    return type_->toString();
  }

 private:
  TypePtr type_;
};

// =============================================================================
// UnitAttr - Unit attribute (no value)
// =============================================================================
class UnitAttr : public Attr {
 public:
  UnitAttr() : Attr(AttrKind::kUnit) {}

  bool equals(const Attr& other) const override {
    return other.kind() == AttrKind::kUnit;
  }

  size_t hash() const override { return 0x2468ACE0; }

  std::string toString() const override { return "unit"; }
};

// =============================================================================
// DenseElementsAttr - Dense tensor constant
// =============================================================================
class DenseElementsAttr : public Attr {
 public:
  DenseElementsAttr(TypePtr type, std::vector<uint8_t> data)
      : Attr(AttrKind::kDenseElements),
        type_(std::move(type)),
        data_(std::move(data)) {}

  const TypePtr& type() const { return type_; }
  const std::vector<uint8_t>& data() const { return data_; }

  bool equals(const Attr& other) const override {
    if (other.kind() != AttrKind::kDenseElements) return false;
    const auto& o = static_cast<const DenseElementsAttr&>(other);
    return type_->equals(*o.type_) && data_ == o.data_;
  }

  size_t hash() const override {
    size_t h = type_->hash();
    for (uint8_t b : data_) {
      h ^= std::hash<uint8_t>{}(b) + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    return h;
  }

  std::string toString() const override {
    return "dense<...> : " + type_->toString();
  }

 private:
  TypePtr type_;
  std::vector<uint8_t> data_;
};

// =============================================================================
// ComparisonDirectionAttr - Comparison direction enum attribute
// =============================================================================
class ComparisonDirectionAttr : public Attr {
 public:
  explicit ComparisonDirectionAttr(uint64_t value)
      : Attr(AttrKind::kComparisonDirection), value_(value) {}

  uint64_t value() const { return value_; }

  bool equals(const Attr& other) const override {
    if (other.kind() != AttrKind::kComparisonDirection) return false;
    return value_ == static_cast<const ComparisonDirectionAttr&>(other).value_;
  }

  size_t hash() const override {
    return std::hash<uint64_t>{}(value_) ^ 0x4C3F1A9B;
  }

  std::string toString() const override {
    return "cmp_dir(" + std::to_string(value_) + ")";
  }

 private:
  uint64_t value_;
};

// =============================================================================
// ComparisonTypeAttr - Comparison type enum attribute
// =============================================================================
class ComparisonTypeAttr : public Attr {
 public:
  explicit ComparisonTypeAttr(uint64_t value)
      : Attr(AttrKind::kComparisonType), value_(value) {}

  uint64_t value() const { return value_; }

  bool equals(const Attr& other) const override {
    if (other.kind() != AttrKind::kComparisonType) return false;
    return value_ == static_cast<const ComparisonTypeAttr&>(other).value_;
  }

  size_t hash() const override {
    return std::hash<uint64_t>{}(value_) ^ 0x7A1B3D5F;
  }

  std::string toString() const override {
    return "cmp_type(" + std::to_string(value_) + ")";
  }

 private:
  uint64_t value_;
};

// =============================================================================
// PrecisionAttr - Precision enum attribute
// =============================================================================
class PrecisionAttr : public Attr {
 public:
  explicit PrecisionAttr(uint64_t value)
      : Attr(AttrKind::kPrecision), value_(value) {}

  uint64_t value() const { return value_; }

  bool equals(const Attr& other) const override {
    if (other.kind() != AttrKind::kPrecision) return false;
    return value_ == static_cast<const PrecisionAttr&>(other).value_;
  }

  size_t hash() const override {
    return std::hash<uint64_t>{}(value_) ^ 0x91E2D3C4;
  }

  std::string toString() const override {
    return "precision(" + std::to_string(value_) + ")";
  }

 private:
  uint64_t value_;
};

// =============================================================================
// Attr factory functions
// =============================================================================
inline AttrPtr makeStringAttr(std::string value) {
  return std::make_shared<StringAttr>(std::move(value));
}

inline AttrPtr makeIntegerAttr(int64_t value, TypePtr type) {
  return std::make_shared<IntegerAttr>(value, std::move(type));
}

inline AttrPtr makeFloatAttr(double value, TypePtr type) {
  return std::make_shared<FloatAttr>(value, std::move(type));
}

inline AttrPtr makeBooleanAttr(bool value) {
  return std::make_shared<BooleanAttr>(value);
}

inline AttrPtr makeArrayAttr(std::vector<AttrPtr> elements) {
  return std::make_shared<ArrayAttr>(std::move(elements));
}

inline AttrPtr makeTypeAttr(TypePtr type) {
  return std::make_shared<TypeAttr>(std::move(type));
}

inline AttrPtr makeUnitAttr() {
  return std::make_shared<UnitAttr>();
}

inline AttrPtr makeDenseElementsAttr(TypePtr type, std::vector<uint8_t> data) {
  return std::make_shared<DenseElementsAttr>(std::move(type), std::move(data));
}

inline AttrPtr makeComparisonDirectionAttr(uint64_t value) {
  return std::make_shared<ComparisonDirectionAttr>(value);
}

inline AttrPtr makeComparisonTypeAttr(uint64_t value) {
  return std::make_shared<ComparisonTypeAttr>(value);
}

inline AttrPtr makePrecisionAttr(uint64_t value) {
  return std::make_shared<PrecisionAttr>(value);
}

}  // namespace stablehlo
}  // namespace lczero
