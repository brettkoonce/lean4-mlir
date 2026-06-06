module @m {
  func.func @mlp_fwd(%x: tensor<128x784xf32>, %W0: tensor<784x512xf32>, %b0: tensor<512xf32>, %W1: tensor<512x512xf32>, %b1: tensor<512xf32>, %W2: tensor<512x10xf32>, %b2: tensor<10xf32>) -> tensor<128x10xf32> {
    %v0 = stablehlo.dot_general %x, %W0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x784xf32>, tensor<784x512xf32>) -> tensor<128x512xf32>
    %v1 = stablehlo.broadcast_in_dim %b0, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v2 = stablehlo.add %v0, %v1 : tensor<128x512xf32>
    %v3 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v4 = stablehlo.maximum %v2, %v3 : tensor<128x512xf32>
    %v5 = stablehlo.dot_general %v4, %W1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %v6 = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v7 = stablehlo.add %v5, %v6 : tensor<128x512xf32>
    %v8 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v9 = stablehlo.maximum %v7, %v8 : tensor<128x512xf32>
    %v10 = stablehlo.dot_general %v9, %W2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x10xf32>) -> tensor<128x10xf32>
    %v11 = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v12 = stablehlo.add %v10, %v11 : tensor<128x10xf32>
    return %v12 : tensor<128x10xf32>
  }
}
