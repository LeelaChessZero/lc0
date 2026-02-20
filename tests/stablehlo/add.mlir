// Minimal StableHLO test module for Check 0
// Serialize with: stablehlo-translate --serialize add.mlir --target=1.0.0 > add.mlirbc
// Verify with:    stablehlo-translate --deserialize add.mlirbc

module {
  func.func @main(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = stablehlo.add %a, %b : tensor<2x3xf32>
    func.return %0 : tensor<2x3xf32>
  }
}
