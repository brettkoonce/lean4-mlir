module @m {
  func.func @cnn_fwd(%x: tensor<128x784xf32>, %W1: tensor<32x1x3x3xf32>, %b1: tensor<32xf32>, %W2: tensor<32x32x3x3xf32>, %b2: tensor<32xf32>, %W3: tensor<6272x512xf32>, %b3: tensor<512xf32>, %W4: tensor<512x512xf32>, %b4: tensor<512xf32>, %W5: tensor<512x10xf32>, %b5: tensor<10xf32>) -> tensor<128x10xf32> {
    %v0 = stablehlo.reshape %x : (tensor<128x784xf32>) -> tensor<128x1x28x28xf32>
    %v1 = stablehlo.convolution(%v0, %W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x1x28x28xf32>, tensor<32x1x3x3xf32>) -> tensor<128x32x28x28xf32>
    %v2 = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<32xf32>) -> tensor<128x32x28x28xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<128x32x28x28xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<128x32x28x28xf32>) -> tensor<128x25088xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<128x25088xf32>) -> tensor<128x32x28x28xf32>
    %v6 = stablehlo.constant dense<0.0> : tensor<128x32x28x28xf32>
    %v7 = stablehlo.maximum %v5, %v6 : tensor<128x32x28x28xf32>
    %v8 = stablehlo.reshape %v7 : (tensor<128x32x28x28xf32>) -> tensor<128x25088xf32>
    %v9 = stablehlo.reshape %v8 : (tensor<128x25088xf32>) -> tensor<128x32x28x28xf32>
    %v10 = stablehlo.convolution(%v9, %W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x28x28xf32>
    %v11 = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<32xf32>) -> tensor<128x32x28x28xf32>
    %v12 = stablehlo.add %v10, %v11 : tensor<128x32x28x28xf32>
    %v13 = stablehlo.reshape %v12 : (tensor<128x32x28x28xf32>) -> tensor<128x25088xf32>
    %v14 = stablehlo.reshape %v13 : (tensor<128x25088xf32>) -> tensor<128x32x28x28xf32>
    %v15 = stablehlo.constant dense<0.0> : tensor<128x32x28x28xf32>
    %v16 = stablehlo.maximum %v14, %v15 : tensor<128x32x28x28xf32>
    %v17 = stablehlo.reshape %v16 : (tensor<128x32x28x28xf32>) -> tensor<128x25088xf32>
    %v18 = stablehlo.reshape %v17 : (tensor<128x25088xf32>) -> tensor<128x32x28x28xf32>
    %v19 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v20 = "stablehlo.reduce_window"(%v18, %v19) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x28x28xf32>, tensor<f32>) -> tensor<128x32x14x14xf32>
    %v21 = stablehlo.reshape %v20 : (tensor<128x32x14x14xf32>) -> tensor<128x6272xf32>
    %v22 = stablehlo.dot_general %v21, %W3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x6272xf32>, tensor<6272x512xf32>) -> tensor<128x512xf32>
    %v23 = stablehlo.broadcast_in_dim %b3, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v24 = stablehlo.add %v22, %v23 : tensor<128x512xf32>
    %v25 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v26 = stablehlo.maximum %v24, %v25 : tensor<128x512xf32>
    %v27 = stablehlo.dot_general %v26, %W4, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %v28 = stablehlo.broadcast_in_dim %b4, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v29 = stablehlo.add %v27, %v28 : tensor<128x512xf32>
    %v30 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v31 = stablehlo.maximum %v29, %v30 : tensor<128x512xf32>
    %v32 = stablehlo.dot_general %v31, %W5, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x10xf32>) -> tensor<128x10xf32>
    %v33 = stablehlo.broadcast_in_dim %b5, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v34 = stablehlo.add %v32, %v33 : tensor<128x10xf32>
    return %v34 : tensor<128x10xf32>
  }
}
