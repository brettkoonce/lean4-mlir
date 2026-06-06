module @m {
  func.func @linear_fwd(%x: tensor<128x784xf32>, %W0: tensor<784x10xf32>, %b0: tensor<10xf32>) -> tensor<128x10xf32> {
    %v0 = stablehlo.dot_general %x, %W0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x784xf32>, tensor<784x10xf32>) -> tensor<128x10xf32>
    %v1 = stablehlo.broadcast_in_dim %b0, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v2 = stablehlo.add %v0, %v1 : tensor<128x10xf32>
    return %v2 : tensor<128x10xf32>
  }
}
