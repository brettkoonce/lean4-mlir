module @m {
  func.func @cifar_fwd(%x: tensor<128x3072xf32>, %W1: tensor<32x3x3x3xf32>, %b1: tensor<32xf32>, %W2: tensor<32x32x3x3xf32>, %b2: tensor<32xf32>, %W3: tensor<64x32x3x3xf32>, %b3: tensor<64xf32>, %W4: tensor<64x64x3x3xf32>, %b4: tensor<64xf32>, %W5: tensor<4096x512xf32>, %b5: tensor<512xf32>, %W6: tensor<512x512xf32>, %b6: tensor<512xf32>, %W7: tensor<512x10xf32>, %b7: tensor<10xf32>) -> tensor<128x10xf32> {
    %v0 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v1 = stablehlo.convolution(%v0, %W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x3x32x32xf32>, tensor<32x3x3x3xf32>) -> tensor<128x32x32x32xf32>
    %v2 = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<32xf32>) -> tensor<128x32x32x32xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<128x32x32x32xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %v5 = stablehlo.constant dense<0.0> : tensor<128x32768xf32>
    %v6 = stablehlo.maximum %v4, %v5 : tensor<128x32768xf32>
    %v7 = stablehlo.reshape %v6 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v8 = stablehlo.convolution(%v7, %W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x32x32xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x32x32xf32>
    %v9 = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<32xf32>) -> tensor<128x32x32x32xf32>
    %v10 = stablehlo.add %v8, %v9 : tensor<128x32x32x32xf32>
    %v11 = stablehlo.reshape %v10 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %v12 = stablehlo.constant dense<0.0> : tensor<128x32768xf32>
    %v13 = stablehlo.maximum %v11, %v12 : tensor<128x32768xf32>
    %v14 = stablehlo.reshape %v13 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v15 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v16 = "stablehlo.reduce_window"(%v14, %v15) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32x16x16xf32>
    %v17 = stablehlo.reshape %v16 : (tensor<128x32x16x16xf32>) -> tensor<128x8192xf32>
    %v18 = stablehlo.reshape %v17 : (tensor<128x8192xf32>) -> tensor<128x32x16x16xf32>
    %v19 = stablehlo.convolution(%v18, %W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x16x16xf32>, tensor<64x32x3x3xf32>) -> tensor<128x64x16x16xf32>
    %v20 = stablehlo.broadcast_in_dim %b3, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %v21 = stablehlo.add %v19, %v20 : tensor<128x64x16x16xf32>
    %v22 = stablehlo.reshape %v21 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v23 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v24 = stablehlo.maximum %v22, %v23 : tensor<128x16384xf32>
    %v25 = stablehlo.reshape %v24 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v26 = stablehlo.convolution(%v25, %W4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x16x16xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x16x16xf32>
    %v27 = stablehlo.broadcast_in_dim %b4, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %v28 = stablehlo.add %v26, %v27 : tensor<128x64x16x16xf32>
    %v29 = stablehlo.reshape %v28 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v30 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v31 = stablehlo.maximum %v29, %v30 : tensor<128x16384xf32>
    %v32 = stablehlo.reshape %v31 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v33 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v34 = "stablehlo.reduce_window"(%v32, %v33) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64x8x8xf32>
    %v35 = stablehlo.reshape %v34 : (tensor<128x64x8x8xf32>) -> tensor<128x4096xf32>
    %v36 = stablehlo.dot_general %v35, %W5, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x4096xf32>, tensor<4096x512xf32>) -> tensor<128x512xf32>
    %v37 = stablehlo.broadcast_in_dim %b5, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v38 = stablehlo.add %v36, %v37 : tensor<128x512xf32>
    %v39 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v40 = stablehlo.maximum %v38, %v39 : tensor<128x512xf32>
    %v41 = stablehlo.dot_general %v40, %W6, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %v42 = stablehlo.broadcast_in_dim %b6, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v43 = stablehlo.add %v41, %v42 : tensor<128x512xf32>
    %v44 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v45 = stablehlo.maximum %v43, %v44 : tensor<128x512xf32>
    %v46 = stablehlo.dot_general %v45, %W7, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x10xf32>) -> tensor<128x10xf32>
    %v47 = stablehlo.broadcast_in_dim %b7, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v48 = stablehlo.add %v46, %v47 : tensor<128x10xf32>
    return %v48 : tensor<128x10xf32>
  }
}
