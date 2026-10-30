module {
  func.func @main(%x: tensor<32x3x224x224xf32>, %dy: tensor<32x192x14x14xf32>) -> tensor<192x3x16x16xf32> {
    %xt = stablehlo.transpose %x, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %pt = stablehlo.transpose %dy, dims = [1, 0, 2, 3] : (tensor<32x192x14x14xf32>) -> tensor<192x32x14x14xf32>
    %c = stablehlo.convolution(%xt, %pt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [16, 16]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<192x32x14x14xf32>) -> tensor<3x192x16x16xf32>
    %r = stablehlo.transpose %c, dims = [1, 0, 2, 3] : (tensor<3x192x16x16xf32>) -> tensor<192x3x16x16xf32>
    return %r : tensor<192x3x16x16xf32>
  }
}
