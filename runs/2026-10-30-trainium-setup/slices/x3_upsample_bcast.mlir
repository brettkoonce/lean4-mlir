module @m {
  func.func @main(%v71: tensor<128x32x4x4xf32>, %v115: tensor<128x128xf32>) -> tensor<32x32x3x3xf32> {
    %g = stablehlo.reshape %v115 : (tensor<128x128xf32>) -> tensor<128x32x2x2xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %u = stablehlo.broadcast_in_dim %g, dims = [0, 1, 2, 4] : (tensor<128x32x2x2xf32>) -> tensor<128x32x2x2x2x2xf32>
    %rhs = stablehlo.reshape %u : (tensor<128x32x2x2x2x2xf32>) -> tensor<128x32x4x4xf32>
    %l = stablehlo.transpose %v71, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %r = stablehlo.transpose %rhs, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %w = stablehlo.convolution(%l, %r) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    return %w : tensor<32x32x3x3xf32>
  }
}
