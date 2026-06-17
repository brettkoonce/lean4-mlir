module @m {
  func.func @cifar8_bn_train_step(%x: tensor<128x3072xf32>, %W1: tensor<16x3x3x3xf32>, %b1: tensor<16xf32>, %g1: tensor<16xf32>, %bt1: tensor<16xf32>, %W2: tensor<16x16x3x3xf32>, %b2: tensor<16xf32>, %g2: tensor<16xf32>, %bt2: tensor<16xf32>, %W3: tensor<16x16x3x3xf32>, %b3: tensor<16xf32>, %g3: tensor<16xf32>, %bt3: tensor<16xf32>, %W4: tensor<16x16x3x3xf32>, %b4: tensor<16xf32>, %g4: tensor<16xf32>, %bt4: tensor<16xf32>, %W5: tensor<32x16x3x3xf32>, %b5: tensor<32xf32>, %g5: tensor<32xf32>, %bt5: tensor<32xf32>, %W6: tensor<32x32x3x3xf32>, %b6: tensor<32xf32>, %g6: tensor<32xf32>, %bt6: tensor<32xf32>, %W7: tensor<32x32x3x3xf32>, %b7: tensor<32xf32>, %g7: tensor<32xf32>, %bt7: tensor<32xf32>, %W8: tensor<32x32x3x3xf32>, %b8: tensor<32xf32>, %g8: tensor<32xf32>, %bt8: tensor<32xf32>, %W9: tensor<128x64xf32>, %b9: tensor<64xf32>, %Wa: tensor<64x64xf32>, %ba: tensor<64xf32>, %Wb: tensor<64x10xf32>, %bb: tensor<10xf32>, %onehot: tensor<128x10xf32>) -> (tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>) {
    // ── cifar8-bn train step: every line is pretty(verified AST node) ──
    %v0 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v1 = stablehlo.convolution(%v0, %W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x3x32x32xf32>, tensor<16x3x3x3xf32>) -> tensor<128x16x32x32xf32>
    %v2 = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<128x16x32x32xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v6 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v8 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v9 = stablehlo.reduce(%v5 init: %v6) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v10 = stablehlo.broadcast_in_dim %v9, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v11 = stablehlo.divide %v10, %v7 : tensor<128x16x32x32xf32>
    %v12 = stablehlo.subtract %v5, %v11 : tensor<128x16x32x32xf32>
    %v13 = stablehlo.multiply %v12, %v12 : tensor<128x16x32x32xf32>
    %v14 = stablehlo.reduce(%v13 init: %v6) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v15 = stablehlo.broadcast_in_dim %v14, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v16 = stablehlo.divide %v15, %v7 : tensor<128x16x32x32xf32>
    %v17 = stablehlo.add %v16, %v8 : tensor<128x16x32x32xf32>
    %v18 = stablehlo.rsqrt %v17 : tensor<128x16x32x32xf32>
    %v19 = stablehlo.multiply %v12, %v18 : tensor<128x16x32x32xf32>
    %v20 = stablehlo.broadcast_in_dim %g1, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v21 = stablehlo.broadcast_in_dim %bt1, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v22 = stablehlo.multiply %v19, %v20 : tensor<128x16x32x32xf32>
    %v23 = stablehlo.add %v22, %v21 : tensor<128x16x32x32xf32>
    %v24 = stablehlo.reshape %v23 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v25 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v26 = stablehlo.maximum %v24, %v25 : tensor<128x16384xf32>
    %v27 = stablehlo.reshape %v26 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v28 = stablehlo.convolution(%v27, %W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x32x32xf32>
    %v29 = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v30 = stablehlo.add %v28, %v29 : tensor<128x16x32x32xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v32 = stablehlo.reshape %v31 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v33 = stablehlo.constant dense<0.0> : tensor<f32>
    %v34 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v35 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v36 = stablehlo.reduce(%v32 init: %v33) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v37 = stablehlo.broadcast_in_dim %v36, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v38 = stablehlo.divide %v37, %v34 : tensor<128x16x32x32xf32>
    %v39 = stablehlo.subtract %v32, %v38 : tensor<128x16x32x32xf32>
    %v40 = stablehlo.multiply %v39, %v39 : tensor<128x16x32x32xf32>
    %v41 = stablehlo.reduce(%v40 init: %v33) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v42 = stablehlo.broadcast_in_dim %v41, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v43 = stablehlo.divide %v42, %v34 : tensor<128x16x32x32xf32>
    %v44 = stablehlo.add %v43, %v35 : tensor<128x16x32x32xf32>
    %v45 = stablehlo.rsqrt %v44 : tensor<128x16x32x32xf32>
    %v46 = stablehlo.multiply %v39, %v45 : tensor<128x16x32x32xf32>
    %v47 = stablehlo.broadcast_in_dim %g2, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v48 = stablehlo.broadcast_in_dim %bt2, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v49 = stablehlo.multiply %v46, %v47 : tensor<128x16x32x32xf32>
    %v50 = stablehlo.add %v49, %v48 : tensor<128x16x32x32xf32>
    %v51 = stablehlo.reshape %v50 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v52 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v53 = stablehlo.maximum %v51, %v52 : tensor<128x16384xf32>
    %v54 = stablehlo.reshape %v53 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v55 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v56 = "stablehlo.reduce_window"(%v54, %v55) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16x16x16xf32>
    %v57 = stablehlo.reshape %v56 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v59 = stablehlo.convolution(%v58, %W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v60 = stablehlo.broadcast_in_dim %b3, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v61 = stablehlo.add %v59, %v60 : tensor<128x16x16x16xf32>
    %v62 = stablehlo.reshape %v61 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v63 = stablehlo.reshape %v62 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v64 = stablehlo.constant dense<0.0> : tensor<f32>
    %v65 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v66 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v67 = stablehlo.reduce(%v63 init: %v64) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v68 = stablehlo.broadcast_in_dim %v67, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v69 = stablehlo.divide %v68, %v65 : tensor<128x16x16x16xf32>
    %v70 = stablehlo.subtract %v63, %v69 : tensor<128x16x16x16xf32>
    %v71 = stablehlo.multiply %v70, %v70 : tensor<128x16x16x16xf32>
    %v72 = stablehlo.reduce(%v71 init: %v64) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v73 = stablehlo.broadcast_in_dim %v72, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v74 = stablehlo.divide %v73, %v65 : tensor<128x16x16x16xf32>
    %v75 = stablehlo.add %v74, %v66 : tensor<128x16x16x16xf32>
    %v76 = stablehlo.rsqrt %v75 : tensor<128x16x16x16xf32>
    %v77 = stablehlo.multiply %v70, %v76 : tensor<128x16x16x16xf32>
    %v78 = stablehlo.broadcast_in_dim %g3, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v79 = stablehlo.broadcast_in_dim %bt3, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v80 = stablehlo.multiply %v77, %v78 : tensor<128x16x16x16xf32>
    %v81 = stablehlo.add %v80, %v79 : tensor<128x16x16x16xf32>
    %v82 = stablehlo.reshape %v81 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v83 = stablehlo.constant dense<0.0> : tensor<128x4096xf32>
    %v84 = stablehlo.maximum %v82, %v83 : tensor<128x4096xf32>
    %v85 = stablehlo.reshape %v84 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v86 = stablehlo.convolution(%v85, %W4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v87 = stablehlo.broadcast_in_dim %b4, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v88 = stablehlo.add %v86, %v87 : tensor<128x16x16x16xf32>
    %v89 = stablehlo.reshape %v88 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v90 = stablehlo.reshape %v89 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v91 = stablehlo.constant dense<0.0> : tensor<f32>
    %v92 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v93 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v94 = stablehlo.reduce(%v90 init: %v91) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v95 = stablehlo.broadcast_in_dim %v94, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v96 = stablehlo.divide %v95, %v92 : tensor<128x16x16x16xf32>
    %v97 = stablehlo.subtract %v90, %v96 : tensor<128x16x16x16xf32>
    %v98 = stablehlo.multiply %v97, %v97 : tensor<128x16x16x16xf32>
    %v99 = stablehlo.reduce(%v98 init: %v91) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v100 = stablehlo.broadcast_in_dim %v99, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v101 = stablehlo.divide %v100, %v92 : tensor<128x16x16x16xf32>
    %v102 = stablehlo.add %v101, %v93 : tensor<128x16x16x16xf32>
    %v103 = stablehlo.rsqrt %v102 : tensor<128x16x16x16xf32>
    %v104 = stablehlo.multiply %v97, %v103 : tensor<128x16x16x16xf32>
    %v105 = stablehlo.broadcast_in_dim %g4, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v106 = stablehlo.broadcast_in_dim %bt4, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v107 = stablehlo.multiply %v104, %v105 : tensor<128x16x16x16xf32>
    %v108 = stablehlo.add %v107, %v106 : tensor<128x16x16x16xf32>
    %v109 = stablehlo.reshape %v108 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v110 = stablehlo.constant dense<0.0> : tensor<128x4096xf32>
    %v111 = stablehlo.maximum %v109, %v110 : tensor<128x4096xf32>
    %v112 = stablehlo.reshape %v111 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v113 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v114 = "stablehlo.reduce_window"(%v112, %v113) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x8x8xf32>
    %v115 = stablehlo.reshape %v114 : (tensor<128x16x8x8xf32>) -> tensor<128x1024xf32>
    %v116 = stablehlo.reshape %v115 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v117 = stablehlo.convolution(%v116, %W5)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x8x8xf32>, tensor<32x16x3x3xf32>) -> tensor<128x32x8x8xf32>
    %v118 = stablehlo.broadcast_in_dim %b5, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v119 = stablehlo.add %v117, %v118 : tensor<128x32x8x8xf32>
    %v120 = stablehlo.reshape %v119 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v121 = stablehlo.reshape %v120 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v122 = stablehlo.constant dense<0.0> : tensor<f32>
    %v123 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v124 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v125 = stablehlo.reduce(%v121 init: %v122) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v126 = stablehlo.broadcast_in_dim %v125, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v127 = stablehlo.divide %v126, %v123 : tensor<128x32x8x8xf32>
    %v128 = stablehlo.subtract %v121, %v127 : tensor<128x32x8x8xf32>
    %v129 = stablehlo.multiply %v128, %v128 : tensor<128x32x8x8xf32>
    %v130 = stablehlo.reduce(%v129 init: %v122) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v131 = stablehlo.broadcast_in_dim %v130, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v132 = stablehlo.divide %v131, %v123 : tensor<128x32x8x8xf32>
    %v133 = stablehlo.add %v132, %v124 : tensor<128x32x8x8xf32>
    %v134 = stablehlo.rsqrt %v133 : tensor<128x32x8x8xf32>
    %v135 = stablehlo.multiply %v128, %v134 : tensor<128x32x8x8xf32>
    %v136 = stablehlo.broadcast_in_dim %g5, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v137 = stablehlo.broadcast_in_dim %bt5, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v138 = stablehlo.multiply %v135, %v136 : tensor<128x32x8x8xf32>
    %v139 = stablehlo.add %v138, %v137 : tensor<128x32x8x8xf32>
    %v140 = stablehlo.reshape %v139 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v141 = stablehlo.constant dense<0.0> : tensor<128x2048xf32>
    %v142 = stablehlo.maximum %v140, %v141 : tensor<128x2048xf32>
    %v143 = stablehlo.reshape %v142 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v144 = stablehlo.convolution(%v143, %W6)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x8x8xf32>
    %v145 = stablehlo.broadcast_in_dim %b6, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v146 = stablehlo.add %v144, %v145 : tensor<128x32x8x8xf32>
    %v147 = stablehlo.reshape %v146 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v148 = stablehlo.reshape %v147 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v149 = stablehlo.constant dense<0.0> : tensor<f32>
    %v150 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v151 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v152 = stablehlo.reduce(%v148 init: %v149) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v153 = stablehlo.broadcast_in_dim %v152, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v154 = stablehlo.divide %v153, %v150 : tensor<128x32x8x8xf32>
    %v155 = stablehlo.subtract %v148, %v154 : tensor<128x32x8x8xf32>
    %v156 = stablehlo.multiply %v155, %v155 : tensor<128x32x8x8xf32>
    %v157 = stablehlo.reduce(%v156 init: %v149) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v158 = stablehlo.broadcast_in_dim %v157, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v159 = stablehlo.divide %v158, %v150 : tensor<128x32x8x8xf32>
    %v160 = stablehlo.add %v159, %v151 : tensor<128x32x8x8xf32>
    %v161 = stablehlo.rsqrt %v160 : tensor<128x32x8x8xf32>
    %v162 = stablehlo.multiply %v155, %v161 : tensor<128x32x8x8xf32>
    %v163 = stablehlo.broadcast_in_dim %g6, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v164 = stablehlo.broadcast_in_dim %bt6, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v165 = stablehlo.multiply %v162, %v163 : tensor<128x32x8x8xf32>
    %v166 = stablehlo.add %v165, %v164 : tensor<128x32x8x8xf32>
    %v167 = stablehlo.reshape %v166 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v168 = stablehlo.constant dense<0.0> : tensor<128x2048xf32>
    %v169 = stablehlo.maximum %v167, %v168 : tensor<128x2048xf32>
    %v170 = stablehlo.reshape %v169 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v171 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v172 = "stablehlo.reduce_window"(%v170, %v171) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32x4x4xf32>
    %v173 = stablehlo.reshape %v172 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v174 = stablehlo.reshape %v173 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v175 = stablehlo.convolution(%v174, %W7)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v176 = stablehlo.broadcast_in_dim %b7, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v177 = stablehlo.add %v175, %v176 : tensor<128x32x4x4xf32>
    %v178 = stablehlo.reshape %v177 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v179 = stablehlo.reshape %v178 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v180 = stablehlo.constant dense<0.0> : tensor<f32>
    %v181 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v182 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v183 = stablehlo.reduce(%v179 init: %v180) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v184 = stablehlo.broadcast_in_dim %v183, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v185 = stablehlo.divide %v184, %v181 : tensor<128x32x4x4xf32>
    %v186 = stablehlo.subtract %v179, %v185 : tensor<128x32x4x4xf32>
    %v187 = stablehlo.multiply %v186, %v186 : tensor<128x32x4x4xf32>
    %v188 = stablehlo.reduce(%v187 init: %v180) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v189 = stablehlo.broadcast_in_dim %v188, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v190 = stablehlo.divide %v189, %v181 : tensor<128x32x4x4xf32>
    %v191 = stablehlo.add %v190, %v182 : tensor<128x32x4x4xf32>
    %v192 = stablehlo.rsqrt %v191 : tensor<128x32x4x4xf32>
    %v193 = stablehlo.multiply %v186, %v192 : tensor<128x32x4x4xf32>
    %v194 = stablehlo.broadcast_in_dim %g7, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v195 = stablehlo.broadcast_in_dim %bt7, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v196 = stablehlo.multiply %v193, %v194 : tensor<128x32x4x4xf32>
    %v197 = stablehlo.add %v196, %v195 : tensor<128x32x4x4xf32>
    %v198 = stablehlo.reshape %v197 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v199 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v200 = stablehlo.maximum %v198, %v199 : tensor<128x512xf32>
    %v201 = stablehlo.reshape %v200 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v202 = stablehlo.convolution(%v201, %W8)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v203 = stablehlo.broadcast_in_dim %b8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v204 = stablehlo.add %v202, %v203 : tensor<128x32x4x4xf32>
    %v205 = stablehlo.reshape %v204 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v206 = stablehlo.reshape %v205 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v207 = stablehlo.constant dense<0.0> : tensor<f32>
    %v208 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v209 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v210 = stablehlo.reduce(%v206 init: %v207) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v211 = stablehlo.broadcast_in_dim %v210, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v212 = stablehlo.divide %v211, %v208 : tensor<128x32x4x4xf32>
    %v213 = stablehlo.subtract %v206, %v212 : tensor<128x32x4x4xf32>
    %v214 = stablehlo.multiply %v213, %v213 : tensor<128x32x4x4xf32>
    %v215 = stablehlo.reduce(%v214 init: %v207) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v216 = stablehlo.broadcast_in_dim %v215, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v217 = stablehlo.divide %v216, %v208 : tensor<128x32x4x4xf32>
    %v218 = stablehlo.add %v217, %v209 : tensor<128x32x4x4xf32>
    %v219 = stablehlo.rsqrt %v218 : tensor<128x32x4x4xf32>
    %v220 = stablehlo.multiply %v213, %v219 : tensor<128x32x4x4xf32>
    %v221 = stablehlo.broadcast_in_dim %g8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v222 = stablehlo.broadcast_in_dim %bt8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v223 = stablehlo.multiply %v220, %v221 : tensor<128x32x4x4xf32>
    %v224 = stablehlo.add %v223, %v222 : tensor<128x32x4x4xf32>
    %v225 = stablehlo.reshape %v224 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v226 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v227 = stablehlo.maximum %v225, %v226 : tensor<128x512xf32>
    %v228 = stablehlo.reshape %v227 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v229 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v230 = "stablehlo.reduce_window"(%v228, %v229) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32x2x2xf32>
    %v231 = stablehlo.reshape %v230 : (tensor<128x32x2x2xf32>) -> tensor<128x128xf32>
    %v232 = stablehlo.dot_general %v231, %W9, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %v233 = stablehlo.broadcast_in_dim %b9, dims = [1] : (tensor<64xf32>) -> tensor<128x64xf32>
    %v234 = stablehlo.add %v232, %v233 : tensor<128x64xf32>
    %v235 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v236 = stablehlo.maximum %v234, %v235 : tensor<128x64xf32>
    %v237 = stablehlo.dot_general %v236, %Wa, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
    %v238 = stablehlo.broadcast_in_dim %ba, dims = [1] : (tensor<64xf32>) -> tensor<128x64xf32>
    %v239 = stablehlo.add %v237, %v238 : tensor<128x64xf32>
    %v240 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v241 = stablehlo.maximum %v239, %v240 : tensor<128x64xf32>
    %v242 = stablehlo.dot_general %v241, %Wb, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x10xf32>) -> tensor<128x10xf32>
    %v243 = stablehlo.broadcast_in_dim %bb, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v244 = stablehlo.add %v242, %v243 : tensor<128x10xf32>
    %v245 = stablehlo.exponential %v244 : tensor<128x10xf32>
    %v246 = stablehlo.constant dense<0.0> : tensor<f32>
    %v247 = stablehlo.reduce(%v245 init: %v246) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %v248 = stablehlo.broadcast_in_dim %v247, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %v249 = stablehlo.divide %v245, %v248 : tensor<128x10xf32>
    %v250 = stablehlo.subtract %v249, %onehot : tensor<128x10xf32>
    %v251 = stablehlo.dot_general %v250, %Wb, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x10xf32>, tensor<64x10xf32>) -> tensor<128x64xf32>
    %v252 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v253 = stablehlo.compare GT, %v239, %v252 : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xi1>
    %v254 = stablehlo.select %v253, %v251, %v252 : tensor<128x64xi1>, tensor<128x64xf32>
    %v255 = stablehlo.dot_general %v254, %Wa, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
    %v256 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v257 = stablehlo.compare GT, %v234, %v256 : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xi1>
    %v258 = stablehlo.select %v257, %v255, %v256 : tensor<128x64xi1>, tensor<128x64xf32>
    %v259 = stablehlo.dot_general %v258, %W9, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x128xf32>
    %v260 = stablehlo.reshape %v227 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v261 = stablehlo.reshape %v259 : (tensor<128x128xf32>) -> tensor<128x32x2x2xf32>
    %v262 = stablehlo.constant dense<0.0> : tensor<f32>
    %v263 = "stablehlo.select_and_scatter"(%v260, %v261, %v262) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x4x4xf32>, tensor<128x32x2x2xf32>, tensor<f32>) -> tensor<128x32x4x4xf32>
    %v264 = stablehlo.reshape %v263 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v265 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v266 = stablehlo.compare GT, %v225, %v265 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v267 = stablehlo.select %v266, %v264, %v265 : tensor<128x512xi1>, tensor<128x512xf32>
    %v268 = stablehlo.reshape %v267 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v269 = stablehlo.reshape %v205 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v270 = stablehlo.constant dense<0.0> : tensor<f32>
    %v271 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v272 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v273 = stablehlo.reduce(%v269 init: %v270) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v274 = stablehlo.broadcast_in_dim %v273, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v275 = stablehlo.divide %v274, %v271 : tensor<128x32x4x4xf32>
    %v276 = stablehlo.subtract %v269, %v275 : tensor<128x32x4x4xf32>
    %v277 = stablehlo.multiply %v276, %v276 : tensor<128x32x4x4xf32>
    %v278 = stablehlo.reduce(%v277 init: %v270) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v279 = stablehlo.broadcast_in_dim %v278, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v280 = stablehlo.divide %v279, %v271 : tensor<128x32x4x4xf32>
    %v281 = stablehlo.add %v280, %v272 : tensor<128x32x4x4xf32>
    %v282 = stablehlo.rsqrt %v281 : tensor<128x32x4x4xf32>
    %v283 = stablehlo.multiply %v276, %v282 : tensor<128x32x4x4xf32>
    %v284 = stablehlo.broadcast_in_dim %g8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v285 = stablehlo.multiply %v284, %v268 : tensor<128x32x4x4xf32>
    %v286 = stablehlo.reduce(%v285 init: %v270) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v287 = stablehlo.broadcast_in_dim %v286, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v288 = stablehlo.multiply %v283, %v285 : tensor<128x32x4x4xf32>
    %v289 = stablehlo.reduce(%v288 init: %v270) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v290 = stablehlo.broadcast_in_dim %v289, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v291 = stablehlo.multiply %v285, %v271 : tensor<128x32x4x4xf32>
    %v292 = stablehlo.subtract %v291, %v287 : tensor<128x32x4x4xf32>
    %v293 = stablehlo.multiply %v283, %v290 : tensor<128x32x4x4xf32>
    %v294 = stablehlo.subtract %v292, %v293 : tensor<128x32x4x4xf32>
    %v295 = stablehlo.divide %v282, %v271 : tensor<128x32x4x4xf32>
    %v296 = stablehlo.multiply %v295, %v294 : tensor<128x32x4x4xf32>
    %v297 = stablehlo.reshape %v296 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v298 = stablehlo.reshape %v297 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v299 = stablehlo.transpose %W8, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v300 = stablehlo.reverse %v299, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v301 = stablehlo.convolution(%v298, %v300)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v302 = stablehlo.reshape %v301 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v303 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v304 = stablehlo.compare GT, %v198, %v303 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v305 = stablehlo.select %v304, %v302, %v303 : tensor<128x512xi1>, tensor<128x512xf32>
    %v306 = stablehlo.reshape %v305 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v307 = stablehlo.reshape %v178 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v308 = stablehlo.constant dense<0.0> : tensor<f32>
    %v309 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v310 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v311 = stablehlo.reduce(%v307 init: %v308) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v312 = stablehlo.broadcast_in_dim %v311, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v313 = stablehlo.divide %v312, %v309 : tensor<128x32x4x4xf32>
    %v314 = stablehlo.subtract %v307, %v313 : tensor<128x32x4x4xf32>
    %v315 = stablehlo.multiply %v314, %v314 : tensor<128x32x4x4xf32>
    %v316 = stablehlo.reduce(%v315 init: %v308) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v317 = stablehlo.broadcast_in_dim %v316, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v318 = stablehlo.divide %v317, %v309 : tensor<128x32x4x4xf32>
    %v319 = stablehlo.add %v318, %v310 : tensor<128x32x4x4xf32>
    %v320 = stablehlo.rsqrt %v319 : tensor<128x32x4x4xf32>
    %v321 = stablehlo.multiply %v314, %v320 : tensor<128x32x4x4xf32>
    %v322 = stablehlo.broadcast_in_dim %g7, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v323 = stablehlo.multiply %v322, %v306 : tensor<128x32x4x4xf32>
    %v324 = stablehlo.reduce(%v323 init: %v308) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v325 = stablehlo.broadcast_in_dim %v324, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v326 = stablehlo.multiply %v321, %v323 : tensor<128x32x4x4xf32>
    %v327 = stablehlo.reduce(%v326 init: %v308) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v328 = stablehlo.broadcast_in_dim %v327, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v329 = stablehlo.multiply %v323, %v309 : tensor<128x32x4x4xf32>
    %v330 = stablehlo.subtract %v329, %v325 : tensor<128x32x4x4xf32>
    %v331 = stablehlo.multiply %v321, %v328 : tensor<128x32x4x4xf32>
    %v332 = stablehlo.subtract %v330, %v331 : tensor<128x32x4x4xf32>
    %v333 = stablehlo.divide %v320, %v309 : tensor<128x32x4x4xf32>
    %v334 = stablehlo.multiply %v333, %v332 : tensor<128x32x4x4xf32>
    %v335 = stablehlo.reshape %v334 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v336 = stablehlo.reshape %v335 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v337 = stablehlo.transpose %W7, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v338 = stablehlo.reverse %v337, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v339 = stablehlo.convolution(%v336, %v338)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v340 = stablehlo.reshape %v339 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v341 = stablehlo.reshape %v169 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v342 = stablehlo.reshape %v340 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v343 = stablehlo.constant dense<0.0> : tensor<f32>
    %v344 = "stablehlo.select_and_scatter"(%v341, %v342, %v343) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x8x8xf32>, tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32x8x8xf32>
    %v345 = stablehlo.reshape %v344 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v346 = stablehlo.constant dense<0.0> : tensor<128x2048xf32>
    %v347 = stablehlo.compare GT, %v167, %v346 : (tensor<128x2048xf32>, tensor<128x2048xf32>) -> tensor<128x2048xi1>
    %v348 = stablehlo.select %v347, %v345, %v346 : tensor<128x2048xi1>, tensor<128x2048xf32>
    %v349 = stablehlo.reshape %v348 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v350 = stablehlo.reshape %v147 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v351 = stablehlo.constant dense<0.0> : tensor<f32>
    %v352 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v353 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v354 = stablehlo.reduce(%v350 init: %v351) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v355 = stablehlo.broadcast_in_dim %v354, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v356 = stablehlo.divide %v355, %v352 : tensor<128x32x8x8xf32>
    %v357 = stablehlo.subtract %v350, %v356 : tensor<128x32x8x8xf32>
    %v358 = stablehlo.multiply %v357, %v357 : tensor<128x32x8x8xf32>
    %v359 = stablehlo.reduce(%v358 init: %v351) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v360 = stablehlo.broadcast_in_dim %v359, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v361 = stablehlo.divide %v360, %v352 : tensor<128x32x8x8xf32>
    %v362 = stablehlo.add %v361, %v353 : tensor<128x32x8x8xf32>
    %v363 = stablehlo.rsqrt %v362 : tensor<128x32x8x8xf32>
    %v364 = stablehlo.multiply %v357, %v363 : tensor<128x32x8x8xf32>
    %v365 = stablehlo.broadcast_in_dim %g6, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v366 = stablehlo.multiply %v365, %v349 : tensor<128x32x8x8xf32>
    %v367 = stablehlo.reduce(%v366 init: %v351) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v368 = stablehlo.broadcast_in_dim %v367, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v369 = stablehlo.multiply %v364, %v366 : tensor<128x32x8x8xf32>
    %v370 = stablehlo.reduce(%v369 init: %v351) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v371 = stablehlo.broadcast_in_dim %v370, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v372 = stablehlo.multiply %v366, %v352 : tensor<128x32x8x8xf32>
    %v373 = stablehlo.subtract %v372, %v368 : tensor<128x32x8x8xf32>
    %v374 = stablehlo.multiply %v364, %v371 : tensor<128x32x8x8xf32>
    %v375 = stablehlo.subtract %v373, %v374 : tensor<128x32x8x8xf32>
    %v376 = stablehlo.divide %v363, %v352 : tensor<128x32x8x8xf32>
    %v377 = stablehlo.multiply %v376, %v375 : tensor<128x32x8x8xf32>
    %v378 = stablehlo.reshape %v377 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v379 = stablehlo.reshape %v378 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v380 = stablehlo.transpose %W6, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v381 = stablehlo.reverse %v380, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v382 = stablehlo.convolution(%v379, %v381)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x8x8xf32>
    %v383 = stablehlo.reshape %v382 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v384 = stablehlo.constant dense<0.0> : tensor<128x2048xf32>
    %v385 = stablehlo.compare GT, %v140, %v384 : (tensor<128x2048xf32>, tensor<128x2048xf32>) -> tensor<128x2048xi1>
    %v386 = stablehlo.select %v385, %v383, %v384 : tensor<128x2048xi1>, tensor<128x2048xf32>
    %v387 = stablehlo.reshape %v386 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v388 = stablehlo.reshape %v120 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v389 = stablehlo.constant dense<0.0> : tensor<f32>
    %v390 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v391 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v392 = stablehlo.reduce(%v388 init: %v389) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v393 = stablehlo.broadcast_in_dim %v392, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v394 = stablehlo.divide %v393, %v390 : tensor<128x32x8x8xf32>
    %v395 = stablehlo.subtract %v388, %v394 : tensor<128x32x8x8xf32>
    %v396 = stablehlo.multiply %v395, %v395 : tensor<128x32x8x8xf32>
    %v397 = stablehlo.reduce(%v396 init: %v389) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v398 = stablehlo.broadcast_in_dim %v397, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v399 = stablehlo.divide %v398, %v390 : tensor<128x32x8x8xf32>
    %v400 = stablehlo.add %v399, %v391 : tensor<128x32x8x8xf32>
    %v401 = stablehlo.rsqrt %v400 : tensor<128x32x8x8xf32>
    %v402 = stablehlo.multiply %v395, %v401 : tensor<128x32x8x8xf32>
    %v403 = stablehlo.broadcast_in_dim %g5, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v404 = stablehlo.multiply %v403, %v387 : tensor<128x32x8x8xf32>
    %v405 = stablehlo.reduce(%v404 init: %v389) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v406 = stablehlo.broadcast_in_dim %v405, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v407 = stablehlo.multiply %v402, %v404 : tensor<128x32x8x8xf32>
    %v408 = stablehlo.reduce(%v407 init: %v389) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v409 = stablehlo.broadcast_in_dim %v408, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v410 = stablehlo.multiply %v404, %v390 : tensor<128x32x8x8xf32>
    %v411 = stablehlo.subtract %v410, %v406 : tensor<128x32x8x8xf32>
    %v412 = stablehlo.multiply %v402, %v409 : tensor<128x32x8x8xf32>
    %v413 = stablehlo.subtract %v411, %v412 : tensor<128x32x8x8xf32>
    %v414 = stablehlo.divide %v401, %v390 : tensor<128x32x8x8xf32>
    %v415 = stablehlo.multiply %v414, %v413 : tensor<128x32x8x8xf32>
    %v416 = stablehlo.reshape %v415 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v417 = stablehlo.reshape %v416 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v418 = stablehlo.transpose %W5, dims = [1, 0, 2, 3] : (tensor<32x16x3x3xf32>) -> tensor<16x32x3x3xf32>
    %v419 = stablehlo.reverse %v418, dims = [2, 3] : tensor<16x32x3x3xf32>
    %v420 = stablehlo.convolution(%v417, %v419)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<16x32x3x3xf32>) -> tensor<128x16x8x8xf32>
    %v421 = stablehlo.reshape %v420 : (tensor<128x16x8x8xf32>) -> tensor<128x1024xf32>
    %v422 = stablehlo.reshape %v111 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v423 = stablehlo.reshape %v421 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v424 = stablehlo.constant dense<0.0> : tensor<f32>
    %v425 = "stablehlo.select_and_scatter"(%v422, %v423, %v424) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x16x16xf32>, tensor<128x16x8x8xf32>, tensor<f32>) -> tensor<128x16x16x16xf32>
    %v426 = stablehlo.reshape %v425 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v427 = stablehlo.constant dense<0.0> : tensor<128x4096xf32>
    %v428 = stablehlo.compare GT, %v109, %v427 : (tensor<128x4096xf32>, tensor<128x4096xf32>) -> tensor<128x4096xi1>
    %v429 = stablehlo.select %v428, %v426, %v427 : tensor<128x4096xi1>, tensor<128x4096xf32>
    %v430 = stablehlo.reshape %v429 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v431 = stablehlo.reshape %v89 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v432 = stablehlo.constant dense<0.0> : tensor<f32>
    %v433 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v434 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v435 = stablehlo.reduce(%v431 init: %v432) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v436 = stablehlo.broadcast_in_dim %v435, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v437 = stablehlo.divide %v436, %v433 : tensor<128x16x16x16xf32>
    %v438 = stablehlo.subtract %v431, %v437 : tensor<128x16x16x16xf32>
    %v439 = stablehlo.multiply %v438, %v438 : tensor<128x16x16x16xf32>
    %v440 = stablehlo.reduce(%v439 init: %v432) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v441 = stablehlo.broadcast_in_dim %v440, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v442 = stablehlo.divide %v441, %v433 : tensor<128x16x16x16xf32>
    %v443 = stablehlo.add %v442, %v434 : tensor<128x16x16x16xf32>
    %v444 = stablehlo.rsqrt %v443 : tensor<128x16x16x16xf32>
    %v445 = stablehlo.multiply %v438, %v444 : tensor<128x16x16x16xf32>
    %v446 = stablehlo.broadcast_in_dim %g4, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v447 = stablehlo.multiply %v446, %v430 : tensor<128x16x16x16xf32>
    %v448 = stablehlo.reduce(%v447 init: %v432) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v449 = stablehlo.broadcast_in_dim %v448, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v450 = stablehlo.multiply %v445, %v447 : tensor<128x16x16x16xf32>
    %v451 = stablehlo.reduce(%v450 init: %v432) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v452 = stablehlo.broadcast_in_dim %v451, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v453 = stablehlo.multiply %v447, %v433 : tensor<128x16x16x16xf32>
    %v454 = stablehlo.subtract %v453, %v449 : tensor<128x16x16x16xf32>
    %v455 = stablehlo.multiply %v445, %v452 : tensor<128x16x16x16xf32>
    %v456 = stablehlo.subtract %v454, %v455 : tensor<128x16x16x16xf32>
    %v457 = stablehlo.divide %v444, %v433 : tensor<128x16x16x16xf32>
    %v458 = stablehlo.multiply %v457, %v456 : tensor<128x16x16x16xf32>
    %v459 = stablehlo.reshape %v458 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v460 = stablehlo.reshape %v459 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v461 = stablehlo.transpose %W4, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v462 = stablehlo.reverse %v461, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v463 = stablehlo.convolution(%v460, %v462)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v464 = stablehlo.reshape %v463 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v465 = stablehlo.constant dense<0.0> : tensor<128x4096xf32>
    %v466 = stablehlo.compare GT, %v82, %v465 : (tensor<128x4096xf32>, tensor<128x4096xf32>) -> tensor<128x4096xi1>
    %v467 = stablehlo.select %v466, %v464, %v465 : tensor<128x4096xi1>, tensor<128x4096xf32>
    %v468 = stablehlo.reshape %v467 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v469 = stablehlo.reshape %v62 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v470 = stablehlo.constant dense<0.0> : tensor<f32>
    %v471 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v472 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v473 = stablehlo.reduce(%v469 init: %v470) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v474 = stablehlo.broadcast_in_dim %v473, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v475 = stablehlo.divide %v474, %v471 : tensor<128x16x16x16xf32>
    %v476 = stablehlo.subtract %v469, %v475 : tensor<128x16x16x16xf32>
    %v477 = stablehlo.multiply %v476, %v476 : tensor<128x16x16x16xf32>
    %v478 = stablehlo.reduce(%v477 init: %v470) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v479 = stablehlo.broadcast_in_dim %v478, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v480 = stablehlo.divide %v479, %v471 : tensor<128x16x16x16xf32>
    %v481 = stablehlo.add %v480, %v472 : tensor<128x16x16x16xf32>
    %v482 = stablehlo.rsqrt %v481 : tensor<128x16x16x16xf32>
    %v483 = stablehlo.multiply %v476, %v482 : tensor<128x16x16x16xf32>
    %v484 = stablehlo.broadcast_in_dim %g3, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v485 = stablehlo.multiply %v484, %v468 : tensor<128x16x16x16xf32>
    %v486 = stablehlo.reduce(%v485 init: %v470) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v487 = stablehlo.broadcast_in_dim %v486, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v488 = stablehlo.multiply %v483, %v485 : tensor<128x16x16x16xf32>
    %v489 = stablehlo.reduce(%v488 init: %v470) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v490 = stablehlo.broadcast_in_dim %v489, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v491 = stablehlo.multiply %v485, %v471 : tensor<128x16x16x16xf32>
    %v492 = stablehlo.subtract %v491, %v487 : tensor<128x16x16x16xf32>
    %v493 = stablehlo.multiply %v483, %v490 : tensor<128x16x16x16xf32>
    %v494 = stablehlo.subtract %v492, %v493 : tensor<128x16x16x16xf32>
    %v495 = stablehlo.divide %v482, %v471 : tensor<128x16x16x16xf32>
    %v496 = stablehlo.multiply %v495, %v494 : tensor<128x16x16x16xf32>
    %v497 = stablehlo.reshape %v496 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v498 = stablehlo.reshape %v497 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v499 = stablehlo.transpose %W3, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v500 = stablehlo.reverse %v499, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v501 = stablehlo.convolution(%v498, %v500)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v502 = stablehlo.reshape %v501 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v503 = stablehlo.reshape %v53 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v504 = stablehlo.reshape %v502 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v505 = stablehlo.constant dense<0.0> : tensor<f32>
    %v506 = "stablehlo.select_and_scatter"(%v503, %v504, %v505) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x32x32xf32>, tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x32x32xf32>
    %v507 = stablehlo.reshape %v506 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v508 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v509 = stablehlo.compare GT, %v51, %v508 : (tensor<128x16384xf32>, tensor<128x16384xf32>) -> tensor<128x16384xi1>
    %v510 = stablehlo.select %v509, %v507, %v508 : tensor<128x16384xi1>, tensor<128x16384xf32>
    %v511 = stablehlo.reshape %v510 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v512 = stablehlo.reshape %v31 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v513 = stablehlo.constant dense<0.0> : tensor<f32>
    %v514 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v515 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v516 = stablehlo.reduce(%v512 init: %v513) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v517 = stablehlo.broadcast_in_dim %v516, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v518 = stablehlo.divide %v517, %v514 : tensor<128x16x32x32xf32>
    %v519 = stablehlo.subtract %v512, %v518 : tensor<128x16x32x32xf32>
    %v520 = stablehlo.multiply %v519, %v519 : tensor<128x16x32x32xf32>
    %v521 = stablehlo.reduce(%v520 init: %v513) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v522 = stablehlo.broadcast_in_dim %v521, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v523 = stablehlo.divide %v522, %v514 : tensor<128x16x32x32xf32>
    %v524 = stablehlo.add %v523, %v515 : tensor<128x16x32x32xf32>
    %v525 = stablehlo.rsqrt %v524 : tensor<128x16x32x32xf32>
    %v526 = stablehlo.multiply %v519, %v525 : tensor<128x16x32x32xf32>
    %v527 = stablehlo.broadcast_in_dim %g2, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v528 = stablehlo.multiply %v527, %v511 : tensor<128x16x32x32xf32>
    %v529 = stablehlo.reduce(%v528 init: %v513) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v530 = stablehlo.broadcast_in_dim %v529, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v531 = stablehlo.multiply %v526, %v528 : tensor<128x16x32x32xf32>
    %v532 = stablehlo.reduce(%v531 init: %v513) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v533 = stablehlo.broadcast_in_dim %v532, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v534 = stablehlo.multiply %v528, %v514 : tensor<128x16x32x32xf32>
    %v535 = stablehlo.subtract %v534, %v530 : tensor<128x16x32x32xf32>
    %v536 = stablehlo.multiply %v526, %v533 : tensor<128x16x32x32xf32>
    %v537 = stablehlo.subtract %v535, %v536 : tensor<128x16x32x32xf32>
    %v538 = stablehlo.divide %v525, %v514 : tensor<128x16x32x32xf32>
    %v539 = stablehlo.multiply %v538, %v537 : tensor<128x16x32x32xf32>
    %v540 = stablehlo.reshape %v539 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v541 = stablehlo.reshape %v540 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v542 = stablehlo.transpose %W2, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v543 = stablehlo.reverse %v542, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v544 = stablehlo.convolution(%v541, %v543)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x32x32xf32>
    %v545 = stablehlo.reshape %v544 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v546 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v547 = stablehlo.compare GT, %v24, %v546 : (tensor<128x16384xf32>, tensor<128x16384xf32>) -> tensor<128x16384xi1>
    %v548 = stablehlo.select %v547, %v545, %v546 : tensor<128x16384xi1>, tensor<128x16384xf32>
    %v549 = stablehlo.reshape %v548 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v550 = stablehlo.reshape %v4 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v551 = stablehlo.constant dense<0.0> : tensor<f32>
    %v552 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v553 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v554 = stablehlo.reduce(%v550 init: %v551) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v555 = stablehlo.broadcast_in_dim %v554, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v556 = stablehlo.divide %v555, %v552 : tensor<128x16x32x32xf32>
    %v557 = stablehlo.subtract %v550, %v556 : tensor<128x16x32x32xf32>
    %v558 = stablehlo.multiply %v557, %v557 : tensor<128x16x32x32xf32>
    %v559 = stablehlo.reduce(%v558 init: %v551) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v560 = stablehlo.broadcast_in_dim %v559, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v561 = stablehlo.divide %v560, %v552 : tensor<128x16x32x32xf32>
    %v562 = stablehlo.add %v561, %v553 : tensor<128x16x32x32xf32>
    %v563 = stablehlo.rsqrt %v562 : tensor<128x16x32x32xf32>
    %v564 = stablehlo.multiply %v557, %v563 : tensor<128x16x32x32xf32>
    %v565 = stablehlo.broadcast_in_dim %g1, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v566 = stablehlo.multiply %v565, %v549 : tensor<128x16x32x32xf32>
    %v567 = stablehlo.reduce(%v566 init: %v551) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v568 = stablehlo.broadcast_in_dim %v567, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v569 = stablehlo.multiply %v564, %v566 : tensor<128x16x32x32xf32>
    %v570 = stablehlo.reduce(%v569 init: %v551) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v571 = stablehlo.broadcast_in_dim %v570, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v572 = stablehlo.multiply %v566, %v552 : tensor<128x16x32x32xf32>
    %v573 = stablehlo.subtract %v572, %v568 : tensor<128x16x32x32xf32>
    %v574 = stablehlo.multiply %v564, %v571 : tensor<128x16x32x32xf32>
    %v575 = stablehlo.subtract %v573, %v574 : tensor<128x16x32x32xf32>
    %v576 = stablehlo.divide %v563, %v552 : tensor<128x16x32x32xf32>
    %v577 = stablehlo.multiply %v576, %v575 : tensor<128x16x32x32xf32>
    %v578 = stablehlo.reshape %v577 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v579 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v580 = stablehlo.reshape %v578 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v581 = stablehlo.transpose %v579, dims = [1, 0, 2, 3] : (tensor<128x3x32x32xf32>) -> tensor<3x128x32x32xf32>
    %v582 = stablehlo.transpose %v580, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v583 = stablehlo.convolution(%v581, %v582)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<3x16x3x3xf32>
    %v584 = stablehlo.transpose %v583, dims = [1, 0, 2, 3] : (tensor<3x16x3x3xf32>) -> tensor<16x3x3x3xf32>
    %v585 = stablehlo.constant dense<0.00078125> : tensor<16x3x3x3xf32>
    %v586 = stablehlo.multiply %v584, %v585 : tensor<16x3x3x3xf32>
    %v587 = stablehlo.subtract %W1, %v586 : tensor<16x3x3x3xf32>
    %v588 = stablehlo.reshape %v578 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v589 = stablehlo.constant dense<0.0> : tensor<f32>
    %v590 = stablehlo.reduce(%v588 init: %v589) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v591 = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %v592 = stablehlo.multiply %v590, %v591 : tensor<16xf32>
    %v593 = stablehlo.subtract %b1, %v592 : tensor<16xf32>
    %v594 = stablehlo.constant dense<0.0> : tensor<f32>
    %v595 = stablehlo.reshape %v4 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v596 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v597 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v598 = stablehlo.reduce(%v595 init: %v594) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v599 = stablehlo.broadcast_in_dim %v598, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v600 = stablehlo.divide %v599, %v596 : tensor<128x16x32x32xf32>
    %v601 = stablehlo.subtract %v595, %v600 : tensor<128x16x32x32xf32>
    %v602 = stablehlo.multiply %v601, %v601 : tensor<128x16x32x32xf32>
    %v603 = stablehlo.reduce(%v602 init: %v594) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v604 = stablehlo.broadcast_in_dim %v603, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v605 = stablehlo.divide %v604, %v596 : tensor<128x16x32x32xf32>
    %v606 = stablehlo.add %v605, %v597 : tensor<128x16x32x32xf32>
    %v607 = stablehlo.rsqrt %v606 : tensor<128x16x32x32xf32>
    %v608 = stablehlo.multiply %v601, %v607 : tensor<128x16x32x32xf32>
    %v609 = stablehlo.reshape %v548 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v610 = stablehlo.multiply %v609, %v608 : tensor<128x16x32x32xf32>
    %v611 = stablehlo.reduce(%v610 init: %v594) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v612 = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %v613 = stablehlo.multiply %v611, %v612 : tensor<16xf32>
    %v614 = stablehlo.subtract %g1, %v613 : tensor<16xf32>
    %v615 = stablehlo.constant dense<0.0> : tensor<f32>
    %v616 = stablehlo.reshape %v548 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v617 = stablehlo.reduce(%v616 init: %v615) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v618 = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %v619 = stablehlo.multiply %v617, %v618 : tensor<16xf32>
    %v620 = stablehlo.subtract %bt1, %v619 : tensor<16xf32>
    %v621 = stablehlo.reshape %v26 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v622 = stablehlo.reshape %v540 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v623 = stablehlo.transpose %v621, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v624 = stablehlo.transpose %v622, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v625 = stablehlo.convolution(%v623, %v624)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x16x3x3xf32>
    %v626 = stablehlo.transpose %v625, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v627 = stablehlo.constant dense<0.00078125> : tensor<16x16x3x3xf32>
    %v628 = stablehlo.multiply %v626, %v627 : tensor<16x16x3x3xf32>
    %v629 = stablehlo.subtract %W2, %v628 : tensor<16x16x3x3xf32>
    %v630 = stablehlo.reshape %v540 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v631 = stablehlo.constant dense<0.0> : tensor<f32>
    %v632 = stablehlo.reduce(%v630 init: %v631) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v633 = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %v634 = stablehlo.multiply %v632, %v633 : tensor<16xf32>
    %v635 = stablehlo.subtract %b2, %v634 : tensor<16xf32>
    %v636 = stablehlo.constant dense<0.0> : tensor<f32>
    %v637 = stablehlo.reshape %v31 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v638 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v639 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v640 = stablehlo.reduce(%v637 init: %v636) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v641 = stablehlo.broadcast_in_dim %v640, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v642 = stablehlo.divide %v641, %v638 : tensor<128x16x32x32xf32>
    %v643 = stablehlo.subtract %v637, %v642 : tensor<128x16x32x32xf32>
    %v644 = stablehlo.multiply %v643, %v643 : tensor<128x16x32x32xf32>
    %v645 = stablehlo.reduce(%v644 init: %v636) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v646 = stablehlo.broadcast_in_dim %v645, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v647 = stablehlo.divide %v646, %v638 : tensor<128x16x32x32xf32>
    %v648 = stablehlo.add %v647, %v639 : tensor<128x16x32x32xf32>
    %v649 = stablehlo.rsqrt %v648 : tensor<128x16x32x32xf32>
    %v650 = stablehlo.multiply %v643, %v649 : tensor<128x16x32x32xf32>
    %v651 = stablehlo.reshape %v510 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v652 = stablehlo.multiply %v651, %v650 : tensor<128x16x32x32xf32>
    %v653 = stablehlo.reduce(%v652 init: %v636) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v654 = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %v655 = stablehlo.multiply %v653, %v654 : tensor<16xf32>
    %v656 = stablehlo.subtract %g2, %v655 : tensor<16xf32>
    %v657 = stablehlo.constant dense<0.0> : tensor<f32>
    %v658 = stablehlo.reshape %v510 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v659 = stablehlo.reduce(%v658 init: %v657) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v660 = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %v661 = stablehlo.multiply %v659, %v660 : tensor<16xf32>
    %v662 = stablehlo.subtract %bt2, %v661 : tensor<16xf32>
    %v663 = stablehlo.reshape %v57 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v664 = stablehlo.reshape %v497 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v665 = stablehlo.transpose %v663, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v666 = stablehlo.transpose %v664, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v667 = stablehlo.convolution(%v665, %v666)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v668 = stablehlo.transpose %v667, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v669 = stablehlo.constant dense<0.00078125> : tensor<16x16x3x3xf32>
    %v670 = stablehlo.multiply %v668, %v669 : tensor<16x16x3x3xf32>
    %v671 = stablehlo.subtract %W3, %v670 : tensor<16x16x3x3xf32>
    %v672 = stablehlo.reshape %v497 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v673 = stablehlo.constant dense<0.0> : tensor<f32>
    %v674 = stablehlo.reduce(%v672 init: %v673) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v675 = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %v676 = stablehlo.multiply %v674, %v675 : tensor<16xf32>
    %v677 = stablehlo.subtract %b3, %v676 : tensor<16xf32>
    %v678 = stablehlo.constant dense<0.0> : tensor<f32>
    %v679 = stablehlo.reshape %v62 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v680 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v681 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v682 = stablehlo.reduce(%v679 init: %v678) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v683 = stablehlo.broadcast_in_dim %v682, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v684 = stablehlo.divide %v683, %v680 : tensor<128x16x16x16xf32>
    %v685 = stablehlo.subtract %v679, %v684 : tensor<128x16x16x16xf32>
    %v686 = stablehlo.multiply %v685, %v685 : tensor<128x16x16x16xf32>
    %v687 = stablehlo.reduce(%v686 init: %v678) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v688 = stablehlo.broadcast_in_dim %v687, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v689 = stablehlo.divide %v688, %v680 : tensor<128x16x16x16xf32>
    %v690 = stablehlo.add %v689, %v681 : tensor<128x16x16x16xf32>
    %v691 = stablehlo.rsqrt %v690 : tensor<128x16x16x16xf32>
    %v692 = stablehlo.multiply %v685, %v691 : tensor<128x16x16x16xf32>
    %v693 = stablehlo.reshape %v467 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v694 = stablehlo.multiply %v693, %v692 : tensor<128x16x16x16xf32>
    %v695 = stablehlo.reduce(%v694 init: %v678) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v696 = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %v697 = stablehlo.multiply %v695, %v696 : tensor<16xf32>
    %v698 = stablehlo.subtract %g3, %v697 : tensor<16xf32>
    %v699 = stablehlo.constant dense<0.0> : tensor<f32>
    %v700 = stablehlo.reshape %v467 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v701 = stablehlo.reduce(%v700 init: %v699) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v702 = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %v703 = stablehlo.multiply %v701, %v702 : tensor<16xf32>
    %v704 = stablehlo.subtract %bt3, %v703 : tensor<16xf32>
    %v705 = stablehlo.reshape %v84 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v706 = stablehlo.reshape %v459 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v707 = stablehlo.transpose %v705, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v708 = stablehlo.transpose %v706, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v709 = stablehlo.convolution(%v707, %v708)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v710 = stablehlo.transpose %v709, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v711 = stablehlo.constant dense<0.00078125> : tensor<16x16x3x3xf32>
    %v712 = stablehlo.multiply %v710, %v711 : tensor<16x16x3x3xf32>
    %v713 = stablehlo.subtract %W4, %v712 : tensor<16x16x3x3xf32>
    %v714 = stablehlo.reshape %v459 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v715 = stablehlo.constant dense<0.0> : tensor<f32>
    %v716 = stablehlo.reduce(%v714 init: %v715) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v717 = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %v718 = stablehlo.multiply %v716, %v717 : tensor<16xf32>
    %v719 = stablehlo.subtract %b4, %v718 : tensor<16xf32>
    %v720 = stablehlo.constant dense<0.0> : tensor<f32>
    %v721 = stablehlo.reshape %v89 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v722 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v723 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v724 = stablehlo.reduce(%v721 init: %v720) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v725 = stablehlo.broadcast_in_dim %v724, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v726 = stablehlo.divide %v725, %v722 : tensor<128x16x16x16xf32>
    %v727 = stablehlo.subtract %v721, %v726 : tensor<128x16x16x16xf32>
    %v728 = stablehlo.multiply %v727, %v727 : tensor<128x16x16x16xf32>
    %v729 = stablehlo.reduce(%v728 init: %v720) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v730 = stablehlo.broadcast_in_dim %v729, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v731 = stablehlo.divide %v730, %v722 : tensor<128x16x16x16xf32>
    %v732 = stablehlo.add %v731, %v723 : tensor<128x16x16x16xf32>
    %v733 = stablehlo.rsqrt %v732 : tensor<128x16x16x16xf32>
    %v734 = stablehlo.multiply %v727, %v733 : tensor<128x16x16x16xf32>
    %v735 = stablehlo.reshape %v429 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v736 = stablehlo.multiply %v735, %v734 : tensor<128x16x16x16xf32>
    %v737 = stablehlo.reduce(%v736 init: %v720) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v738 = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %v739 = stablehlo.multiply %v737, %v738 : tensor<16xf32>
    %v740 = stablehlo.subtract %g4, %v739 : tensor<16xf32>
    %v741 = stablehlo.constant dense<0.0> : tensor<f32>
    %v742 = stablehlo.reshape %v429 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v743 = stablehlo.reduce(%v742 init: %v741) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v744 = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %v745 = stablehlo.multiply %v743, %v744 : tensor<16xf32>
    %v746 = stablehlo.subtract %bt4, %v745 : tensor<16xf32>
    %v747 = stablehlo.reshape %v115 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v748 = stablehlo.reshape %v416 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v749 = stablehlo.transpose %v747, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v750 = stablehlo.transpose %v748, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v751 = stablehlo.convolution(%v749, %v750)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<16x32x3x3xf32>
    %v752 = stablehlo.transpose %v751, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v753 = stablehlo.constant dense<0.00078125> : tensor<32x16x3x3xf32>
    %v754 = stablehlo.multiply %v752, %v753 : tensor<32x16x3x3xf32>
    %v755 = stablehlo.subtract %W5, %v754 : tensor<32x16x3x3xf32>
    %v756 = stablehlo.reshape %v416 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v757 = stablehlo.constant dense<0.0> : tensor<f32>
    %v758 = stablehlo.reduce(%v756 init: %v757) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v759 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v760 = stablehlo.multiply %v758, %v759 : tensor<32xf32>
    %v761 = stablehlo.subtract %b5, %v760 : tensor<32xf32>
    %v762 = stablehlo.constant dense<0.0> : tensor<f32>
    %v763 = stablehlo.reshape %v120 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v764 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v765 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v766 = stablehlo.reduce(%v763 init: %v762) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v767 = stablehlo.broadcast_in_dim %v766, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v768 = stablehlo.divide %v767, %v764 : tensor<128x32x8x8xf32>
    %v769 = stablehlo.subtract %v763, %v768 : tensor<128x32x8x8xf32>
    %v770 = stablehlo.multiply %v769, %v769 : tensor<128x32x8x8xf32>
    %v771 = stablehlo.reduce(%v770 init: %v762) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v772 = stablehlo.broadcast_in_dim %v771, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v773 = stablehlo.divide %v772, %v764 : tensor<128x32x8x8xf32>
    %v774 = stablehlo.add %v773, %v765 : tensor<128x32x8x8xf32>
    %v775 = stablehlo.rsqrt %v774 : tensor<128x32x8x8xf32>
    %v776 = stablehlo.multiply %v769, %v775 : tensor<128x32x8x8xf32>
    %v777 = stablehlo.reshape %v386 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v778 = stablehlo.multiply %v777, %v776 : tensor<128x32x8x8xf32>
    %v779 = stablehlo.reduce(%v778 init: %v762) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v780 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v781 = stablehlo.multiply %v779, %v780 : tensor<32xf32>
    %v782 = stablehlo.subtract %g5, %v781 : tensor<32xf32>
    %v783 = stablehlo.constant dense<0.0> : tensor<f32>
    %v784 = stablehlo.reshape %v386 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v785 = stablehlo.reduce(%v784 init: %v783) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v786 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v787 = stablehlo.multiply %v785, %v786 : tensor<32xf32>
    %v788 = stablehlo.subtract %bt5, %v787 : tensor<32xf32>
    %v789 = stablehlo.reshape %v142 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v790 = stablehlo.reshape %v378 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v791 = stablehlo.transpose %v789, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v792 = stablehlo.transpose %v790, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v793 = stablehlo.convolution(%v791, %v792)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<32x32x3x3xf32>
    %v794 = stablehlo.transpose %v793, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v795 = stablehlo.constant dense<0.00078125> : tensor<32x32x3x3xf32>
    %v796 = stablehlo.multiply %v794, %v795 : tensor<32x32x3x3xf32>
    %v797 = stablehlo.subtract %W6, %v796 : tensor<32x32x3x3xf32>
    %v798 = stablehlo.reshape %v378 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v799 = stablehlo.constant dense<0.0> : tensor<f32>
    %v800 = stablehlo.reduce(%v798 init: %v799) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v801 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v802 = stablehlo.multiply %v800, %v801 : tensor<32xf32>
    %v803 = stablehlo.subtract %b6, %v802 : tensor<32xf32>
    %v804 = stablehlo.constant dense<0.0> : tensor<f32>
    %v805 = stablehlo.reshape %v147 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v806 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v807 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v808 = stablehlo.reduce(%v805 init: %v804) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v809 = stablehlo.broadcast_in_dim %v808, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v810 = stablehlo.divide %v809, %v806 : tensor<128x32x8x8xf32>
    %v811 = stablehlo.subtract %v805, %v810 : tensor<128x32x8x8xf32>
    %v812 = stablehlo.multiply %v811, %v811 : tensor<128x32x8x8xf32>
    %v813 = stablehlo.reduce(%v812 init: %v804) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v814 = stablehlo.broadcast_in_dim %v813, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v815 = stablehlo.divide %v814, %v806 : tensor<128x32x8x8xf32>
    %v816 = stablehlo.add %v815, %v807 : tensor<128x32x8x8xf32>
    %v817 = stablehlo.rsqrt %v816 : tensor<128x32x8x8xf32>
    %v818 = stablehlo.multiply %v811, %v817 : tensor<128x32x8x8xf32>
    %v819 = stablehlo.reshape %v348 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v820 = stablehlo.multiply %v819, %v818 : tensor<128x32x8x8xf32>
    %v821 = stablehlo.reduce(%v820 init: %v804) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v822 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v823 = stablehlo.multiply %v821, %v822 : tensor<32xf32>
    %v824 = stablehlo.subtract %g6, %v823 : tensor<32xf32>
    %v825 = stablehlo.constant dense<0.0> : tensor<f32>
    %v826 = stablehlo.reshape %v348 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v827 = stablehlo.reduce(%v826 init: %v825) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v828 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v829 = stablehlo.multiply %v827, %v828 : tensor<32xf32>
    %v830 = stablehlo.subtract %bt6, %v829 : tensor<32xf32>
    %v831 = stablehlo.reshape %v173 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v832 = stablehlo.reshape %v335 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v833 = stablehlo.transpose %v831, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v834 = stablehlo.transpose %v832, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v835 = stablehlo.convolution(%v833, %v834)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v836 = stablehlo.transpose %v835, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v837 = stablehlo.constant dense<0.00078125> : tensor<32x32x3x3xf32>
    %v838 = stablehlo.multiply %v836, %v837 : tensor<32x32x3x3xf32>
    %v839 = stablehlo.subtract %W7, %v838 : tensor<32x32x3x3xf32>
    %v840 = stablehlo.reshape %v335 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v841 = stablehlo.constant dense<0.0> : tensor<f32>
    %v842 = stablehlo.reduce(%v840 init: %v841) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v843 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v844 = stablehlo.multiply %v842, %v843 : tensor<32xf32>
    %v845 = stablehlo.subtract %b7, %v844 : tensor<32xf32>
    %v846 = stablehlo.constant dense<0.0> : tensor<f32>
    %v847 = stablehlo.reshape %v178 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v848 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v849 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v850 = stablehlo.reduce(%v847 init: %v846) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v851 = stablehlo.broadcast_in_dim %v850, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v852 = stablehlo.divide %v851, %v848 : tensor<128x32x4x4xf32>
    %v853 = stablehlo.subtract %v847, %v852 : tensor<128x32x4x4xf32>
    %v854 = stablehlo.multiply %v853, %v853 : tensor<128x32x4x4xf32>
    %v855 = stablehlo.reduce(%v854 init: %v846) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v856 = stablehlo.broadcast_in_dim %v855, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v857 = stablehlo.divide %v856, %v848 : tensor<128x32x4x4xf32>
    %v858 = stablehlo.add %v857, %v849 : tensor<128x32x4x4xf32>
    %v859 = stablehlo.rsqrt %v858 : tensor<128x32x4x4xf32>
    %v860 = stablehlo.multiply %v853, %v859 : tensor<128x32x4x4xf32>
    %v861 = stablehlo.reshape %v305 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v862 = stablehlo.multiply %v861, %v860 : tensor<128x32x4x4xf32>
    %v863 = stablehlo.reduce(%v862 init: %v846) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v864 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v865 = stablehlo.multiply %v863, %v864 : tensor<32xf32>
    %v866 = stablehlo.subtract %g7, %v865 : tensor<32xf32>
    %v867 = stablehlo.constant dense<0.0> : tensor<f32>
    %v868 = stablehlo.reshape %v305 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v869 = stablehlo.reduce(%v868 init: %v867) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v870 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v871 = stablehlo.multiply %v869, %v870 : tensor<32xf32>
    %v872 = stablehlo.subtract %bt7, %v871 : tensor<32xf32>
    %v873 = stablehlo.reshape %v200 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v874 = stablehlo.reshape %v297 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v875 = stablehlo.transpose %v873, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v876 = stablehlo.transpose %v874, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v877 = stablehlo.convolution(%v875, %v876)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v878 = stablehlo.transpose %v877, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v879 = stablehlo.constant dense<0.00078125> : tensor<32x32x3x3xf32>
    %v880 = stablehlo.multiply %v878, %v879 : tensor<32x32x3x3xf32>
    %v881 = stablehlo.subtract %W8, %v880 : tensor<32x32x3x3xf32>
    %v882 = stablehlo.reshape %v297 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v883 = stablehlo.constant dense<0.0> : tensor<f32>
    %v884 = stablehlo.reduce(%v882 init: %v883) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v885 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v886 = stablehlo.multiply %v884, %v885 : tensor<32xf32>
    %v887 = stablehlo.subtract %b8, %v886 : tensor<32xf32>
    %v888 = stablehlo.constant dense<0.0> : tensor<f32>
    %v889 = stablehlo.reshape %v205 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v890 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v891 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v892 = stablehlo.reduce(%v889 init: %v888) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v893 = stablehlo.broadcast_in_dim %v892, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v894 = stablehlo.divide %v893, %v890 : tensor<128x32x4x4xf32>
    %v895 = stablehlo.subtract %v889, %v894 : tensor<128x32x4x4xf32>
    %v896 = stablehlo.multiply %v895, %v895 : tensor<128x32x4x4xf32>
    %v897 = stablehlo.reduce(%v896 init: %v888) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v898 = stablehlo.broadcast_in_dim %v897, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v899 = stablehlo.divide %v898, %v890 : tensor<128x32x4x4xf32>
    %v900 = stablehlo.add %v899, %v891 : tensor<128x32x4x4xf32>
    %v901 = stablehlo.rsqrt %v900 : tensor<128x32x4x4xf32>
    %v902 = stablehlo.multiply %v895, %v901 : tensor<128x32x4x4xf32>
    %v903 = stablehlo.reshape %v267 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v904 = stablehlo.multiply %v903, %v902 : tensor<128x32x4x4xf32>
    %v905 = stablehlo.reduce(%v904 init: %v888) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v906 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v907 = stablehlo.multiply %v905, %v906 : tensor<32xf32>
    %v908 = stablehlo.subtract %g8, %v907 : tensor<32xf32>
    %v909 = stablehlo.constant dense<0.0> : tensor<f32>
    %v910 = stablehlo.reshape %v267 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v911 = stablehlo.reduce(%v910 init: %v909) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v912 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v913 = stablehlo.multiply %v911, %v912 : tensor<32xf32>
    %v914 = stablehlo.subtract %bt8, %v913 : tensor<32xf32>
    %v915 = stablehlo.dot_general %v231, %v258, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %v916 = stablehlo.constant dense<0.00078125> : tensor<128x64xf32>
    %v917 = stablehlo.multiply %v915, %v916 : tensor<128x64xf32>
    %v918 = stablehlo.subtract %W9, %v917 : tensor<128x64xf32>
    %v919 = stablehlo.constant dense<0.0> : tensor<f32>
    %v920 = stablehlo.reduce(%v258 init: %v919) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v921 = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %v922 = stablehlo.multiply %v920, %v921 : tensor<64xf32>
    %v923 = stablehlo.subtract %b9, %v922 : tensor<64xf32>
    %v924 = stablehlo.dot_general %v236, %v254, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<64x64xf32>
    %v925 = stablehlo.constant dense<0.00078125> : tensor<64x64xf32>
    %v926 = stablehlo.multiply %v924, %v925 : tensor<64x64xf32>
    %v927 = stablehlo.subtract %Wa, %v926 : tensor<64x64xf32>
    %v928 = stablehlo.constant dense<0.0> : tensor<f32>
    %v929 = stablehlo.reduce(%v254 init: %v928) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v930 = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %v931 = stablehlo.multiply %v929, %v930 : tensor<64xf32>
    %v932 = stablehlo.subtract %ba, %v931 : tensor<64xf32>
    %v933 = stablehlo.dot_general %v241, %v250, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x10xf32>) -> tensor<64x10xf32>
    %v934 = stablehlo.constant dense<0.00078125> : tensor<64x10xf32>
    %v935 = stablehlo.multiply %v933, %v934 : tensor<64x10xf32>
    %v936 = stablehlo.subtract %Wb, %v935 : tensor<64x10xf32>
    %v937 = stablehlo.constant dense<0.0> : tensor<f32>
    %v938 = stablehlo.reduce(%v250 init: %v937) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v939 = stablehlo.constant dense<0.00078125> : tensor<10xf32>
    %v940 = stablehlo.multiply %v938, %v939 : tensor<10xf32>
    %v941 = stablehlo.subtract %bb, %v940 : tensor<10xf32>
    return %v587, %v593, %v614, %v620, %v629, %v635, %v656, %v662, %v671, %v677, %v698, %v704, %v713, %v719, %v740, %v746, %v755, %v761, %v782, %v788, %v797, %v803, %v824, %v830, %v839, %v845, %v866, %v872, %v881, %v887, %v908, %v914, %v918, %v923, %v927, %v932, %v936, %v941 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>
  }
}
