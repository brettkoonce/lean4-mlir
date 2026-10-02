module @m {
  func.func @add(%x: tensor<4xf32>, %y: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.add %x, %y : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
