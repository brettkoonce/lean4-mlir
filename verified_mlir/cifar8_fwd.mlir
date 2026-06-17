module @m {
  func.func @cifar8_fwd(%x: tensor<128x3072xf32>, %W1: tensor<16x3x3x3xf32>, %b1: tensor<16xf32>, %W2: tensor<16x16x3x3xf32>, %b2: tensor<16xf32>, %W3: tensor<16x16x3x3xf32>, %b3: tensor<16xf32>, %W4: tensor<16x16x3x3xf32>, %b4: tensor<16xf32>, %W5: tensor<32x16x3x3xf32>, %b5: tensor<32xf32>, %W6: tensor<32x32x3x3xf32>, %b6: tensor<32xf32>, %W7: tensor<32x32x3x3xf32>, %b7: tensor<32xf32>, %W8: tensor<32x32x3x3xf32>, %b8: tensor<32xf32>, %W9: tensor<128x64xf32>, %b9: tensor<64xf32>, %Wa: tensor<64x64xf32>, %ba: tensor<64xf32>, %Wb: tensor<64x10xf32>, %bb: tensor<10xf32>) -> tensor<128x10xf32> {
    %v0 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v1 = stablehlo.convolution(%v0, %W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x3x32x32xf32>, tensor<16x3x3x3xf32>) -> tensor<128x16x32x32xf32>
    %v2 = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<128x16x32x32xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v5 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v6 = stablehlo.maximum %v4, %v5 : tensor<128x16384xf32>
    %v7 = stablehlo.reshape %v6 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v8 = stablehlo.convolution(%v7, %W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x32x32xf32>
    %v9 = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v10 = stablehlo.add %v8, %v9 : tensor<128x16x32x32xf32>
    %v11 = stablehlo.reshape %v10 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v12 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v13 = stablehlo.maximum %v11, %v12 : tensor<128x16384xf32>
    %v14 = stablehlo.reshape %v13 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v15 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v16 = "stablehlo.reduce_window"(%v14, %v15) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16x16x16xf32>
    %v17 = stablehlo.reshape %v16 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v18 = stablehlo.reshape %v17 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v19 = stablehlo.convolution(%v18, %W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v20 = stablehlo.broadcast_in_dim %b3, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v21 = stablehlo.add %v19, %v20 : tensor<128x16x16x16xf32>
    %v22 = stablehlo.reshape %v21 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v23 = stablehlo.constant dense<0.0> : tensor<128x4096xf32>
    %v24 = stablehlo.maximum %v22, %v23 : tensor<128x4096xf32>
    %v25 = stablehlo.reshape %v24 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v26 = stablehlo.convolution(%v25, %W4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v27 = stablehlo.broadcast_in_dim %b4, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v28 = stablehlo.add %v26, %v27 : tensor<128x16x16x16xf32>
    %v29 = stablehlo.reshape %v28 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v30 = stablehlo.constant dense<0.0> : tensor<128x4096xf32>
    %v31 = stablehlo.maximum %v29, %v30 : tensor<128x4096xf32>
    %v32 = stablehlo.reshape %v31 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v33 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v34 = "stablehlo.reduce_window"(%v32, %v33) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x8x8xf32>
    %v35 = stablehlo.reshape %v34 : (tensor<128x16x8x8xf32>) -> tensor<128x1024xf32>
    %v36 = stablehlo.reshape %v35 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v37 = stablehlo.convolution(%v36, %W5)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x8x8xf32>, tensor<32x16x3x3xf32>) -> tensor<128x32x8x8xf32>
    %v38 = stablehlo.broadcast_in_dim %b5, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v39 = stablehlo.add %v37, %v38 : tensor<128x32x8x8xf32>
    %v40 = stablehlo.reshape %v39 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v41 = stablehlo.constant dense<0.0> : tensor<128x2048xf32>
    %v42 = stablehlo.maximum %v40, %v41 : tensor<128x2048xf32>
    %v43 = stablehlo.reshape %v42 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v44 = stablehlo.convolution(%v43, %W6)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x8x8xf32>
    %v45 = stablehlo.broadcast_in_dim %b6, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v46 = stablehlo.add %v44, %v45 : tensor<128x32x8x8xf32>
    %v47 = stablehlo.reshape %v46 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v48 = stablehlo.constant dense<0.0> : tensor<128x2048xf32>
    %v49 = stablehlo.maximum %v47, %v48 : tensor<128x2048xf32>
    %v50 = stablehlo.reshape %v49 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v51 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v52 = "stablehlo.reduce_window"(%v50, %v51) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32x4x4xf32>
    %v53 = stablehlo.reshape %v52 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v54 = stablehlo.reshape %v53 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v55 = stablehlo.convolution(%v54, %W7)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v56 = stablehlo.broadcast_in_dim %b7, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v57 = stablehlo.add %v55, %v56 : tensor<128x32x4x4xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v59 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v60 = stablehlo.maximum %v58, %v59 : tensor<128x512xf32>
    %v61 = stablehlo.reshape %v60 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v62 = stablehlo.convolution(%v61, %W8)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v63 = stablehlo.broadcast_in_dim %b8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v64 = stablehlo.add %v62, %v63 : tensor<128x32x4x4xf32>
    %v65 = stablehlo.reshape %v64 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v66 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v67 = stablehlo.maximum %v65, %v66 : tensor<128x512xf32>
    %v68 = stablehlo.reshape %v67 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v69 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v70 = "stablehlo.reduce_window"(%v68, %v69) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32x2x2xf32>
    %v71 = stablehlo.reshape %v70 : (tensor<128x32x2x2xf32>) -> tensor<128x128xf32>
    %v72 = stablehlo.dot_general %v71, %W9, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %v73 = stablehlo.broadcast_in_dim %b9, dims = [1] : (tensor<64xf32>) -> tensor<128x64xf32>
    %v74 = stablehlo.add %v72, %v73 : tensor<128x64xf32>
    %v75 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v76 = stablehlo.maximum %v74, %v75 : tensor<128x64xf32>
    %v77 = stablehlo.dot_general %v76, %Wa, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
    %v78 = stablehlo.broadcast_in_dim %ba, dims = [1] : (tensor<64xf32>) -> tensor<128x64xf32>
    %v79 = stablehlo.add %v77, %v78 : tensor<128x64xf32>
    %v80 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v81 = stablehlo.maximum %v79, %v80 : tensor<128x64xf32>
    %v82 = stablehlo.dot_general %v81, %Wb, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x10xf32>) -> tensor<128x10xf32>
    %v83 = stablehlo.broadcast_in_dim %bb, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v84 = stablehlo.add %v82, %v83 : tensor<128x10xf32>
    return %v84 : tensor<128x10xf32>
  }
}
