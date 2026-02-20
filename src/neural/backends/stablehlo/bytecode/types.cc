// =============================================================================
// types.cc - Type Implementation
// =============================================================================

#include "stablehlo/types.h"

namespace lczero {
namespace stablehlo {

uint64_t getTypeCode(ElementType et) {
  switch (et) {
    case ElementType::kBool:        return TypeCode::kBooleanV1;
    case ElementType::kF16:         return TypeCode::kFloatF16V1;
    case ElementType::kBF16:        return TypeCode::kFloatBF16V1;
    case ElementType::kF32:         return TypeCode::kFloatF32V1;
    case ElementType::kF64:         return TypeCode::kFloatF64V1;
    case ElementType::kTF32:        return TypeCode::kFloatTF32V1;
    case ElementType::kF8E4M3FN:    return TypeCode::kFloatF8E4M3FNV1;
    case ElementType::kF8E5M2:      return TypeCode::kFloatF8E5M2V1;
    case ElementType::kF8E4M3FNUZ:  return TypeCode::kFloatF8E4M3FNUZV1;
    case ElementType::kF8E4M3B11FNUZ: return TypeCode::kFloatF8E4M3B11FNUZV1;
    case ElementType::kF8E5M2FNUZ:  return TypeCode::kFloatF8E5M2FNUZV1;
    case ElementType::kSI2:         return TypeCode::kIntegerSI2V1;
    case ElementType::kSI4:         return TypeCode::kIntegerSI4V1;
    case ElementType::kSI8:         return TypeCode::kIntegerSI8V1;
    case ElementType::kSI16:        return TypeCode::kIntegerSI16V1;
    case ElementType::kSI32:        return TypeCode::kIntegerSI32V1;
    case ElementType::kSI64:        return TypeCode::kIntegerSI64V1;
    case ElementType::kUI2:         return TypeCode::kIntegerUI2V1;
    case ElementType::kUI4:         return TypeCode::kIntegerUI4V1;
    case ElementType::kUI8:         return TypeCode::kIntegerUI8V1;
    case ElementType::kUI16:        return TypeCode::kIntegerUI16V1;
    case ElementType::kUI32:        return TypeCode::kIntegerUI32V1;
    case ElementType::kUI64:        return TypeCode::kIntegerUI64V1;
    case ElementType::kComplexF32:  return TypeCode::kComplexF32V1;
    case ElementType::kComplexF64:  return TypeCode::kComplexF64V1;
    case ElementType::kToken:       return TypeCode::kTokenV1;
  }
  return 0;  // Should never reach here.
}

const char* getElementTypeName(ElementType et) {
  switch (et) {
    case ElementType::kBool:        return "i1";
    case ElementType::kF16:         return "f16";
    case ElementType::kBF16:        return "bf16";
    case ElementType::kF32:         return "f32";
    case ElementType::kF64:         return "f64";
    case ElementType::kTF32:        return "tf32";
    case ElementType::kF8E4M3FN:    return "f8E4M3FN";
    case ElementType::kF8E5M2:      return "f8E5M2";
    case ElementType::kF8E4M3FNUZ:  return "f8E4M3FNUZ";
    case ElementType::kF8E4M3B11FNUZ: return "f8E4M3B11FNUZ";
    case ElementType::kF8E5M2FNUZ:  return "f8E5M2FNUZ";
    case ElementType::kSI2:         return "si2";
    case ElementType::kSI4:         return "si4";
    case ElementType::kSI8:         return "si8";
    case ElementType::kSI16:        return "si16";
    case ElementType::kSI32:        return "si32";
    case ElementType::kSI64:        return "si64";
    case ElementType::kUI2:         return "ui2";
    case ElementType::kUI4:         return "ui4";
    case ElementType::kUI8:         return "ui8";
    case ElementType::kUI16:        return "ui16";
    case ElementType::kUI32:        return "ui32";
    case ElementType::kUI64:        return "ui64";
    case ElementType::kComplexF32:  return "complex<f32>";
    case ElementType::kComplexF64:  return "complex<f64>";
    case ElementType::kToken:       return "token";
  }
  return "unknown";
}

}  // namespace stablehlo
}  // namespace lczero
