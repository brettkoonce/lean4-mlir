module {
  func.func @main(%x: tensor<32x3x224x224xf32>, %dy: tensor<32x192x14x14xf32>) -> tensor<192x3x16x16xf32> {
    %xr = stablehlo.reshape %x : (tensor<32x3x224x224xf32>) -> tensor<32x3x14x16x14x16xf32>
    %xt = stablehlo.transpose %xr, dims = [0, 2, 4, 1, 3, 5] : (tensor<32x3x14x16x14x16xf32>) -> tensor<32x14x14x3x16x16xf32>
    %xp = stablehlo.reshape %xt : (tensor<32x14x14x3x16x16xf32>) -> tensor<6272x768xf32>
    %dt = stablehlo.transpose %dy, dims = [0, 2, 3, 1] : (tensor<32x192x14x14xf32>) -> tensor<32x14x14x192xf32>
    %dp = stablehlo.reshape %dt : (tensor<32x14x14x192xf32>) -> tensor<6272x192xf32>
    %g = stablehlo.dot_general %dp, %xp, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<6272x192xf32>, tensor<6272x768xf32>) -> tensor<192x768xf32>
    %r = stablehlo.reshape %g : (tensor<192x768xf32>) -> tensor<192x3x16x16xf32>
    return %r : tensor<192x3x16x16xf32>
  }
}
