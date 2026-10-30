module @m {
  func.func @main(%b8: tensor<32xf32>, %v115: tensor<128x128xf32>, %v71: tensor<128x32x4x4xf32>, %v76: tensor<128x32x4x4xf32>) -> tensor<32x32x3x3xf32> {
    %v77 = stablehlo.broadcast_in_dim %b8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v78 = stablehlo.add %v76, %v77 : tensor<128x32x4x4xf32>
    %v117 = stablehlo.reshape %v115 : (tensor<128x128xf32>) -> tensor<128x32x2x2xf32>
    %v118 = stablehlo.constant dense<0.0> : tensor<f32>
    %nsas1_z = stablehlo.constant dense<0.0> : tensor<f32>
    %nsas1_zs = stablehlo.constant dense<0.0> : tensor<128x32x2x2xf32>
    %nsas1_s00 = stablehlo.slice %v78 [0:128, 0:32, 0:4:2, 0:4:2] : (tensor<128x32x4x4xf32>) -> tensor<128x32x2x2xf32>
    %nsas1_s01 = stablehlo.slice %v78 [0:128, 0:32, 0:4:2, 1:4:2] : (tensor<128x32x4x4xf32>) -> tensor<128x32x2x2xf32>
    %nsas1_s10 = stablehlo.slice %v78 [0:128, 0:32, 1:4:2, 0:4:2] : (tensor<128x32x4x4xf32>) -> tensor<128x32x2x2xf32>
    %nsas1_s11 = stablehlo.slice %v78 [0:128, 0:32, 1:4:2, 1:4:2] : (tensor<128x32x4x4xf32>) -> tensor<128x32x2x2xf32>
    %nsas1_m1 = stablehlo.maximum %nsas1_s00, %nsas1_s01 : tensor<128x32x2x2xf32>
    %nsas1_m2 = stablehlo.maximum %nsas1_m1, %nsas1_s10 : tensor<128x32x2x2xf32>
    %nsas1_m = stablehlo.maximum %nsas1_m2, %nsas1_s11 : tensor<128x32x2x2xf32>
    %nsas1_e00 = stablehlo.compare EQ, %nsas1_s00, %nsas1_m : (tensor<128x32x2x2xf32>, tensor<128x32x2x2xf32>) -> tensor<128x32x2x2xi1>
    %nsas1_c00 = stablehlo.select %nsas1_e00, %v117, %nsas1_zs : tensor<128x32x2x2xi1>, tensor<128x32x2x2xf32>
    %nsas1_p00 = stablehlo.pad %nsas1_c00, %nsas1_z, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<128x32x2x2xf32>, tensor<f32>) -> tensor<128x32x4x4xf32>
    %nsas1_ib = stablehlo.broadcast_in_dim %v118, dims = [] : (tensor<f32>) -> tensor<128x32x4x4xf32>
    %v119 = stablehlo.add %nsas1_p00, %nsas1_ib : tensor<128x32x4x4xf32>
    %v326 = stablehlo.transpose %v71, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v327 = stablehlo.transpose %v119, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v328 = stablehlo.convolution(%v326, %v327)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    return %v328 : tensor<32x32x3x3xf32>
  }
}
