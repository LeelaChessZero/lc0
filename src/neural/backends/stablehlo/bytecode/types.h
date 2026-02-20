// =============================================================================
// types.h - VHLO Type Definitions for Bytecode
// =============================================================================
// Type codes from VhloBytecode.cpp. These are the wire-level type codes
// used in MLIR bytecode for StableHLO/VHLO.
//
// M2 scope: Define types for structural hashing and stable numbering.
// M4 scope: Actual encoding into AttrType section.

#pragma once
#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace lczero {
namespace stablehlo {

// Dynamic dimension sentinel for bytecode (INT64_MIN).
inline constexpr int64_t kDynamicDim = std::numeric_limits<int64_t>::min();

// =============================================================================
// VHLO Type Codes (from VhloBytecode.cpp)
// =============================================================================
// NOTE: kFunctionV1 = 8, verified against golden bytes where function type
// encodes to 0x11 (varint decode: 0x11 >> 1 = 8).
//
// IMPORTANT - EXOTIC DTYPE WARNING:
// The type codes below for commonly-used types (F32, F64, SI32, SI64,
// RankedTensor, Function, Tuple) have been verified against golden .mlirbc
// files. However, codes for exotic dtypes (F8 variants, SI2/UI2, TF32, etc.)
// were shifted to maintain sequential ordering after the kFunctionV1=8 fix.
// When you start using these exotic types, RE-VERIFY their codes against the
// pinned VhloBytecode.cpp (stablehlo-v1.0.0/VhloBytecode.cpp lines 200-250).
//
namespace TypeCode {
  inline constexpr uint64_t kBooleanV1        = 0;
  inline constexpr uint64_t kComplexF32V1     = 1;
  inline constexpr uint64_t kComplexF64V1     = 2;
  inline constexpr uint64_t kFloatBF16V1      = 3;
  inline constexpr uint64_t kFloatF32V1       = 4;
  inline constexpr uint64_t kFloatF64V1       = 5;
  inline constexpr uint64_t kFloatF16V1       = 6;
  inline constexpr uint64_t kFloatF8E4M3FNV1  = 7;
  inline constexpr uint64_t kFunctionV1       = 8;   // CRITICAL: Golden shows 0x11 = varint(8)
  inline constexpr uint64_t kFloatF8E5M2V1    = 9;   // Shifted down
  inline constexpr uint64_t kIntegerSI4V1     = 10;
  inline constexpr uint64_t kIntegerSI8V1     = 11;
  inline constexpr uint64_t kIntegerSI16V1    = 12;
  inline constexpr uint64_t kIntegerSI32V1    = 13;
  inline constexpr uint64_t kIntegerSI64V1    = 14;
  inline constexpr uint64_t kIntegerUI4V1     = 15;
  inline constexpr uint64_t kIntegerUI8V1     = 16;
  inline constexpr uint64_t kIntegerUI16V1    = 17;
  inline constexpr uint64_t kIntegerUI32V1    = 18;
  inline constexpr uint64_t kIntegerUI64V1    = 19;
  inline constexpr uint64_t kRankedTensorV1   = 20;
  inline constexpr uint64_t kRankedTensorV1WithEncoding = 21;
  inline constexpr uint64_t kTokenV1          = 22;
  inline constexpr uint64_t kTupleV1          = 23;
  inline constexpr uint64_t kUnrankedTensorV1 = 24;
  inline constexpr uint64_t kFloatF8E4M3FNUZV1 = 25;
  inline constexpr uint64_t kFloatF8E4M3B11FNUZV1 = 26;
  inline constexpr uint64_t kFloatF8E5M2FNUZV1 = 27;
  inline constexpr uint64_t kWitnessV1        = 28;
  inline constexpr uint64_t kIntegerSI2V1     = 29;
  inline constexpr uint64_t kIntegerUI2V1     = 30;
  inline constexpr uint64_t kFloatTF32V1      = 31;
}  // namespace TypeCode

// Forward declarations.
class Type;
using TypePtr = std::shared_ptr<const Type>;

// =============================================================================
// Element Types (scalar types)
// =============================================================================
enum class ElementType : uint8_t {
  kBool,
  kF16, kBF16, kF32, kF64, kTF32,
  kF8E4M3FN, kF8E5M2, kF8E4M3FNUZ, kF8E4M3B11FNUZ, kF8E5M2FNUZ,
  kSI2, kSI4, kSI8, kSI16, kSI32, kSI64,
  kUI2, kUI4, kUI8, kUI16, kUI32, kUI64,
  kComplexF32, kComplexF64,
  kToken,
};

// Get the VHLO type code for an element type.
uint64_t getTypeCode(ElementType et);

// Get element type name for debugging.
const char* getElementTypeName(ElementType et);

// =============================================================================
// Type - Base class for all types
// =============================================================================
// Types are immutable and structurally compared/hashed.

enum class TypeKind : uint8_t {
  kElement,       // Scalar element type
  kRankedTensor,  // tensor<shape x element>
  kTuple,         // tuple<types...>
  kFunction,      // (inputs) -> (outputs)
  kUnrankedTensor,// tensor<* x element>
  kWitness,       // !stablehlo.witness
};

class Type {
 public:
  virtual ~Type() = default;
  
  TypeKind kind() const { return kind_; }
  
  // Structural equality.
  virtual bool equals(const Type& other) const = 0;
  
  // Structural hash.
  virtual size_t hash() const = 0;
  
  // Debug string.
  virtual std::string toString() const = 0;

 protected:
  explicit Type(TypeKind kind) : kind_(kind) {}

 private:
  TypeKind kind_;
};

// =============================================================================
// ElementTypeWrapper - Wraps a scalar element type
// =============================================================================
class ElementTypeWrapper : public Type {
 public:
  explicit ElementTypeWrapper(ElementType et)
      : Type(TypeKind::kElement), element_type_(et) {}
  
  ElementType elementType() const { return element_type_; }
  
  bool equals(const Type& other) const override {
    if (other.kind() != TypeKind::kElement) return false;
    return element_type_ == static_cast<const ElementTypeWrapper&>(other).element_type_;
  }
  
  size_t hash() const override {
    return std::hash<uint8_t>{}(static_cast<uint8_t>(element_type_));
  }
  
  std::string toString() const override {
    return getElementTypeName(element_type_);
  }

 private:
  ElementType element_type_;
};

// =============================================================================
// RankedTensorType - tensor<shape x element>
// =============================================================================
class RankedTensorType : public Type {
 public:
  RankedTensorType(std::vector<int64_t> shape, TypePtr element_type)
      : Type(TypeKind::kRankedTensor),
        shape_(std::move(shape)),
        element_type_(std::move(element_type)) {
    for (auto& dim : shape_) {
      if (dim == -1) dim = kDynamicDim;
      assert(dim >= 0 || dim == kDynamicDim);
    }
  }
  
  const std::vector<int64_t>& shape() const { return shape_; }
  const TypePtr& elementType() const { return element_type_; }
  
  bool equals(const Type& other) const override {
    if (other.kind() != TypeKind::kRankedTensor) return false;
    const auto& o = static_cast<const RankedTensorType&>(other);
    return shape_ == o.shape_ && element_type_->equals(*o.element_type_);
  }
  
  size_t hash() const override {
    size_t h = element_type_->hash();
    for (int64_t dim : shape_) {
      h ^= std::hash<int64_t>{}(dim) + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    return h;
  }
  
  std::string toString() const override {
    std::string result = "tensor<";
    for (size_t i = 0; i < shape_.size(); ++i) {
      if (i > 0) result += "x";
      if (shape_[i] < 0) {
        result += "?";  // Dynamic dimension
      } else {
        result += std::to_string(shape_[i]);
      }
    }
    result += "x" + element_type_->toString() + ">";
    return result;
  }

 private:
  std::vector<int64_t> shape_;
  TypePtr element_type_;
};

// =============================================================================
// TupleType - tuple<types...>
// =============================================================================
class TupleType : public Type {
 public:
  explicit TupleType(std::vector<TypePtr> element_types)
      : Type(TypeKind::kTuple), element_types_(std::move(element_types)) {}
  
  const std::vector<TypePtr>& elementTypes() const { return element_types_; }
  
  bool equals(const Type& other) const override {
    if (other.kind() != TypeKind::kTuple) return false;
    const auto& o = static_cast<const TupleType&>(other);
    if (element_types_.size() != o.element_types_.size()) return false;
    for (size_t i = 0; i < element_types_.size(); ++i) {
      if (!element_types_[i]->equals(*o.element_types_[i])) return false;
    }
    return true;
  }
  
  size_t hash() const override {
    size_t h = 0x12345678;
    for (const auto& et : element_types_) {
      h ^= et->hash() + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    return h;
  }
  
  std::string toString() const override {
    std::string result = "tuple<";
    for (size_t i = 0; i < element_types_.size(); ++i) {
      if (i > 0) result += ", ";
      result += element_types_[i]->toString();
    }
    result += ">";
    return result;
  }

 private:
  std::vector<TypePtr> element_types_;
};

// =============================================================================
// FunctionType - (inputs) -> (outputs)
// =============================================================================
class FunctionType : public Type {
 public:
  FunctionType(std::vector<TypePtr> inputs, std::vector<TypePtr> outputs)
      : Type(TypeKind::kFunction),
        inputs_(std::move(inputs)),
        outputs_(std::move(outputs)) {}
  
  const std::vector<TypePtr>& inputs() const { return inputs_; }
  const std::vector<TypePtr>& outputs() const { return outputs_; }
  
  bool equals(const Type& other) const override {
    if (other.kind() != TypeKind::kFunction) return false;
    const auto& o = static_cast<const FunctionType&>(other);
    if (inputs_.size() != o.inputs_.size()) return false;
    if (outputs_.size() != o.outputs_.size()) return false;
    for (size_t i = 0; i < inputs_.size(); ++i) {
      if (!inputs_[i]->equals(*o.inputs_[i])) return false;
    }
    for (size_t i = 0; i < outputs_.size(); ++i) {
      if (!outputs_[i]->equals(*o.outputs_[i])) return false;
    }
    return true;
  }
  
  size_t hash() const override {
    size_t h = 0xDEADBEEF;
    for (const auto& t : inputs_) {
      h ^= t->hash() + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    h ^= 0xCAFEBABE;
    for (const auto& t : outputs_) {
      h ^= t->hash() + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    return h;
  }
  
  std::string toString() const override {
    std::string result = "(";
    for (size_t i = 0; i < inputs_.size(); ++i) {
      if (i > 0) result += ", ";
      result += inputs_[i]->toString();
    }
    result += ") -> (";
    for (size_t i = 0; i < outputs_.size(); ++i) {
      if (i > 0) result += ", ";
      result += outputs_[i]->toString();
    }
    result += ")";
    return result;
  }

 private:
  std::vector<TypePtr> inputs_;
  std::vector<TypePtr> outputs_;
};

// =============================================================================
// Type factory functions
// =============================================================================
inline TypePtr makeElementType(ElementType et) {
  return std::make_shared<ElementTypeWrapper>(et);
}

inline TypePtr makeRankedTensor(std::vector<int64_t> shape, TypePtr element_type) {
  // Normalize dynamic dims to INT64_MIN to match VHLO bytecode sentinel.
  for (auto& dim : shape) {
    if (dim == -1) dim = std::numeric_limits<int64_t>::min();
  }
  return std::make_shared<RankedTensorType>(std::move(shape), std::move(element_type));
}

inline TypePtr makeTuple(std::vector<TypePtr> element_types) {
  return std::make_shared<TupleType>(std::move(element_types));
}

inline TypePtr makeFunction(std::vector<TypePtr> inputs, std::vector<TypePtr> outputs) {
  return std::make_shared<FunctionType>(std::move(inputs), std::move(outputs));
}

// Convenience: common element types.
inline TypePtr f32Type() { return makeElementType(ElementType::kF32); }
inline TypePtr f64Type() { return makeElementType(ElementType::kF64); }
inline TypePtr f16Type() { return makeElementType(ElementType::kF16); }
inline TypePtr bf16Type() { return makeElementType(ElementType::kBF16); }
inline TypePtr i32Type() { return makeElementType(ElementType::kSI32); }
inline TypePtr i64Type() { return makeElementType(ElementType::kSI64); }
inline TypePtr boolType() { return makeElementType(ElementType::kBool); }

}  // namespace stablehlo
}  // namespace lczero
