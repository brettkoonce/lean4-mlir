module @m {
  func.func @cifar_bn_fwd(%x: tensor<128x3072xf32>, %W1: tensor<32x3x3x3xf32>, %b1: tensor<32xf32>, %g1: tensor<f32>, %bt1: tensor<f32>, %W2: tensor<32x32x3x3xf32>, %b2: tensor<32xf32>, %g2: tensor<f32>, %bt2: tensor<f32>, %W3: tensor<64x32x3x3xf32>, %b3: tensor<64xf32>, %g3: tensor<f32>, %bt3: tensor<f32>, %W4: tensor<64x64x3x3xf32>, %b4: tensor<64xf32>, %g4: tensor<f32>, %bt4: tensor<f32>, %W5: tensor<4096x512xf32>, %b5: tensor<512xf32>, %W6: tensor<512x512xf32>, %b6: tensor<512xf32>, %W7: tensor<512x10xf32>, %b7: tensor<10xf32>) -> tensor<128x10xf32> {
    %v0 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v1 = stablehlo.convolution(%v0, %W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x3x32x32xf32>, tensor<32x3x3x3xf32>) -> tensor<128x32x32x32xf32>
    %v2 = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<32xf32>) -> tensor<128x32x32x32xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<128x32x32x32xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %v5 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6 = stablehlo.constant dense<32768.0> : tensor<128x32768xf32>
    %v7 = stablehlo.constant dense<1.0e-05> : tensor<128x32768xf32>
    %v8 = stablehlo.reduce(%v4 init: %v5) applies stablehlo.add across dimensions = [1] : (tensor<128x32768xf32>, tensor<f32>) -> tensor<128xf32>
    %v9 = stablehlo.broadcast_in_dim %v8, dims = [0] : (tensor<128xf32>) -> tensor<128x32768xf32>
    %v10 = stablehlo.divide %v9, %v6 : tensor<128x32768xf32>
    %v11 = stablehlo.subtract %v4, %v10 : tensor<128x32768xf32>
    %v12 = stablehlo.multiply %v11, %v11 : tensor<128x32768xf32>
    %v13 = stablehlo.reduce(%v12 init: %v5) applies stablehlo.add across dimensions = [1] : (tensor<128x32768xf32>, tensor<f32>) -> tensor<128xf32>
    %v14 = stablehlo.broadcast_in_dim %v13, dims = [0] : (tensor<128xf32>) -> tensor<128x32768xf32>
    %v15 = stablehlo.divide %v14, %v6 : tensor<128x32768xf32>
    %v16 = stablehlo.add %v15, %v7 : tensor<128x32768xf32>
    %v17 = stablehlo.rsqrt %v16 : tensor<128x32768xf32>
    %v18 = stablehlo.multiply %v11, %v17 : tensor<128x32768xf32>
    %v19 = stablehlo.broadcast_in_dim %g1, dims = [] : (tensor<f32>) -> tensor<128x32768xf32>
    %v20 = stablehlo.broadcast_in_dim %bt1, dims = [] : (tensor<f32>) -> tensor<128x32768xf32>
    %v21 = stablehlo.multiply %v18, %v19 : tensor<128x32768xf32>
    %v22 = stablehlo.add %v21, %v20 : tensor<128x32768xf32>
    %v23 = stablehlo.constant dense<0.0> : tensor<128x32768xf32>
    %v24 = stablehlo.maximum %v22, %v23 : tensor<128x32768xf32>
    %v25 = stablehlo.reshape %v24 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v26 = stablehlo.convolution(%v25, %W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x32x32xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x32x32xf32>
    %v27 = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<32xf32>) -> tensor<128x32x32x32xf32>
    %v28 = stablehlo.add %v26, %v27 : tensor<128x32x32x32xf32>
    %v29 = stablehlo.reshape %v28 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %v30 = stablehlo.constant dense<0.0> : tensor<f32>
    %v31 = stablehlo.constant dense<32768.0> : tensor<128x32768xf32>
    %v32 = stablehlo.constant dense<1.0e-05> : tensor<128x32768xf32>
    %v33 = stablehlo.reduce(%v29 init: %v30) applies stablehlo.add across dimensions = [1] : (tensor<128x32768xf32>, tensor<f32>) -> tensor<128xf32>
    %v34 = stablehlo.broadcast_in_dim %v33, dims = [0] : (tensor<128xf32>) -> tensor<128x32768xf32>
    %v35 = stablehlo.divide %v34, %v31 : tensor<128x32768xf32>
    %v36 = stablehlo.subtract %v29, %v35 : tensor<128x32768xf32>
    %v37 = stablehlo.multiply %v36, %v36 : tensor<128x32768xf32>
    %v38 = stablehlo.reduce(%v37 init: %v30) applies stablehlo.add across dimensions = [1] : (tensor<128x32768xf32>, tensor<f32>) -> tensor<128xf32>
    %v39 = stablehlo.broadcast_in_dim %v38, dims = [0] : (tensor<128xf32>) -> tensor<128x32768xf32>
    %v40 = stablehlo.divide %v39, %v31 : tensor<128x32768xf32>
    %v41 = stablehlo.add %v40, %v32 : tensor<128x32768xf32>
    %v42 = stablehlo.rsqrt %v41 : tensor<128x32768xf32>
    %v43 = stablehlo.multiply %v36, %v42 : tensor<128x32768xf32>
    %v44 = stablehlo.broadcast_in_dim %g2, dims = [] : (tensor<f32>) -> tensor<128x32768xf32>
    %v45 = stablehlo.broadcast_in_dim %bt2, dims = [] : (tensor<f32>) -> tensor<128x32768xf32>
    %v46 = stablehlo.multiply %v43, %v44 : tensor<128x32768xf32>
    %v47 = stablehlo.add %v46, %v45 : tensor<128x32768xf32>
    %v48 = stablehlo.constant dense<0.0> : tensor<128x32768xf32>
    %v49 = stablehlo.maximum %v47, %v48 : tensor<128x32768xf32>
    %v50 = stablehlo.reshape %v49 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v51 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v52 = "stablehlo.reduce_window"(%v50, %v51) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32x16x16xf32>
    %v53 = stablehlo.reshape %v52 : (tensor<128x32x16x16xf32>) -> tensor<128x8192xf32>
    %v54 = stablehlo.reshape %v53 : (tensor<128x8192xf32>) -> tensor<128x32x16x16xf32>
    %v55 = stablehlo.convolution(%v54, %W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x16x16xf32>, tensor<64x32x3x3xf32>) -> tensor<128x64x16x16xf32>
    %v56 = stablehlo.broadcast_in_dim %b3, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %v57 = stablehlo.add %v55, %v56 : tensor<128x64x16x16xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v59 = stablehlo.constant dense<0.0> : tensor<f32>
    %v60 = stablehlo.constant dense<16384.0> : tensor<128x16384xf32>
    %v61 = stablehlo.constant dense<1.0e-05> : tensor<128x16384xf32>
    %v62 = stablehlo.reduce(%v58 init: %v59) applies stablehlo.add across dimensions = [1] : (tensor<128x16384xf32>, tensor<f32>) -> tensor<128xf32>
    %v63 = stablehlo.broadcast_in_dim %v62, dims = [0] : (tensor<128xf32>) -> tensor<128x16384xf32>
    %v64 = stablehlo.divide %v63, %v60 : tensor<128x16384xf32>
    %v65 = stablehlo.subtract %v58, %v64 : tensor<128x16384xf32>
    %v66 = stablehlo.multiply %v65, %v65 : tensor<128x16384xf32>
    %v67 = stablehlo.reduce(%v66 init: %v59) applies stablehlo.add across dimensions = [1] : (tensor<128x16384xf32>, tensor<f32>) -> tensor<128xf32>
    %v68 = stablehlo.broadcast_in_dim %v67, dims = [0] : (tensor<128xf32>) -> tensor<128x16384xf32>
    %v69 = stablehlo.divide %v68, %v60 : tensor<128x16384xf32>
    %v70 = stablehlo.add %v69, %v61 : tensor<128x16384xf32>
    %v71 = stablehlo.rsqrt %v70 : tensor<128x16384xf32>
    %v72 = stablehlo.multiply %v65, %v71 : tensor<128x16384xf32>
    %v73 = stablehlo.broadcast_in_dim %g3, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
    %v74 = stablehlo.broadcast_in_dim %bt3, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
    %v75 = stablehlo.multiply %v72, %v73 : tensor<128x16384xf32>
    %v76 = stablehlo.add %v75, %v74 : tensor<128x16384xf32>
    %v77 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v78 = stablehlo.maximum %v76, %v77 : tensor<128x16384xf32>
    %v79 = stablehlo.reshape %v78 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v80 = stablehlo.convolution(%v79, %W4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x16x16xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x16x16xf32>
    %v81 = stablehlo.broadcast_in_dim %b4, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %v82 = stablehlo.add %v80, %v81 : tensor<128x64x16x16xf32>
    %v83 = stablehlo.reshape %v82 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v84 = stablehlo.constant dense<0.0> : tensor<f32>
    %v85 = stablehlo.constant dense<16384.0> : tensor<128x16384xf32>
    %v86 = stablehlo.constant dense<1.0e-05> : tensor<128x16384xf32>
    %v87 = stablehlo.reduce(%v83 init: %v84) applies stablehlo.add across dimensions = [1] : (tensor<128x16384xf32>, tensor<f32>) -> tensor<128xf32>
    %v88 = stablehlo.broadcast_in_dim %v87, dims = [0] : (tensor<128xf32>) -> tensor<128x16384xf32>
    %v89 = stablehlo.divide %v88, %v85 : tensor<128x16384xf32>
    %v90 = stablehlo.subtract %v83, %v89 : tensor<128x16384xf32>
    %v91 = stablehlo.multiply %v90, %v90 : tensor<128x16384xf32>
    %v92 = stablehlo.reduce(%v91 init: %v84) applies stablehlo.add across dimensions = [1] : (tensor<128x16384xf32>, tensor<f32>) -> tensor<128xf32>
    %v93 = stablehlo.broadcast_in_dim %v92, dims = [0] : (tensor<128xf32>) -> tensor<128x16384xf32>
    %v94 = stablehlo.divide %v93, %v85 : tensor<128x16384xf32>
    %v95 = stablehlo.add %v94, %v86 : tensor<128x16384xf32>
    %v96 = stablehlo.rsqrt %v95 : tensor<128x16384xf32>
    %v97 = stablehlo.multiply %v90, %v96 : tensor<128x16384xf32>
    %v98 = stablehlo.broadcast_in_dim %g4, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
    %v99 = stablehlo.broadcast_in_dim %bt4, dims = [] : (tensor<f32>) -> tensor<128x16384xf32>
    %v100 = stablehlo.multiply %v97, %v98 : tensor<128x16384xf32>
    %v101 = stablehlo.add %v100, %v99 : tensor<128x16384xf32>
    %v102 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v103 = stablehlo.maximum %v101, %v102 : tensor<128x16384xf32>
    %v104 = stablehlo.reshape %v103 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v105 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v106 = "stablehlo.reduce_window"(%v104, %v105) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64x8x8xf32>
    %v107 = stablehlo.reshape %v106 : (tensor<128x64x8x8xf32>) -> tensor<128x4096xf32>
    %v108 = stablehlo.dot_general %v107, %W5, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x4096xf32>, tensor<4096x512xf32>) -> tensor<128x512xf32>
    %v109 = stablehlo.broadcast_in_dim %b5, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v110 = stablehlo.add %v108, %v109 : tensor<128x512xf32>
    %v111 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v112 = stablehlo.maximum %v110, %v111 : tensor<128x512xf32>
    %v113 = stablehlo.dot_general %v112, %W6, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %v114 = stablehlo.broadcast_in_dim %b6, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v115 = stablehlo.add %v113, %v114 : tensor<128x512xf32>
    %v116 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v117 = stablehlo.maximum %v115, %v116 : tensor<128x512xf32>
    %v118 = stablehlo.dot_general %v117, %W7, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x10xf32>) -> tensor<128x10xf32>
    %v119 = stablehlo.broadcast_in_dim %b7, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v120 = stablehlo.add %v118, %v119 : tensor<128x10xf32>
    return %v120 : tensor<128x10xf32>
  }
}
