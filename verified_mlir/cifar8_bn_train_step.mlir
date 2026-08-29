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
    %v25 = stablehlo.reshape %v24 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v26 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v27 = stablehlo.maximum %v25, %v26 : tensor<128x16x32x32xf32>
    %v28 = stablehlo.reshape %v27 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v29 = stablehlo.reshape %v28 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v30 = stablehlo.convolution(%v29, %W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x32x32xf32>
    %v31 = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v32 = stablehlo.add %v30, %v31 : tensor<128x16x32x32xf32>
    %v33 = stablehlo.reshape %v32 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v34 = stablehlo.reshape %v33 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v35 = stablehlo.constant dense<0.0> : tensor<f32>
    %v36 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v37 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v38 = stablehlo.reduce(%v34 init: %v35) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v39 = stablehlo.broadcast_in_dim %v38, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v40 = stablehlo.divide %v39, %v36 : tensor<128x16x32x32xf32>
    %v41 = stablehlo.subtract %v34, %v40 : tensor<128x16x32x32xf32>
    %v42 = stablehlo.multiply %v41, %v41 : tensor<128x16x32x32xf32>
    %v43 = stablehlo.reduce(%v42 init: %v35) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v44 = stablehlo.broadcast_in_dim %v43, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v45 = stablehlo.divide %v44, %v36 : tensor<128x16x32x32xf32>
    %v46 = stablehlo.add %v45, %v37 : tensor<128x16x32x32xf32>
    %v47 = stablehlo.rsqrt %v46 : tensor<128x16x32x32xf32>
    %v48 = stablehlo.multiply %v41, %v47 : tensor<128x16x32x32xf32>
    %v49 = stablehlo.broadcast_in_dim %g2, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v50 = stablehlo.broadcast_in_dim %bt2, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v51 = stablehlo.multiply %v48, %v49 : tensor<128x16x32x32xf32>
    %v52 = stablehlo.add %v51, %v50 : tensor<128x16x32x32xf32>
    %v53 = stablehlo.reshape %v52 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v54 = stablehlo.reshape %v53 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v55 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v56 = stablehlo.maximum %v54, %v55 : tensor<128x16x32x32xf32>
    %v57 = stablehlo.reshape %v56 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v59 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v60 = "stablehlo.reduce_window"(%v58, %v59) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16x16x16xf32>
    %v61 = stablehlo.reshape %v60 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v62 = stablehlo.reshape %v61 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v63 = stablehlo.convolution(%v62, %W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v64 = stablehlo.broadcast_in_dim %b3, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v65 = stablehlo.add %v63, %v64 : tensor<128x16x16x16xf32>
    %v66 = stablehlo.reshape %v65 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v67 = stablehlo.reshape %v66 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v68 = stablehlo.constant dense<0.0> : tensor<f32>
    %v69 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v70 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v71 = stablehlo.reduce(%v67 init: %v68) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v72 = stablehlo.broadcast_in_dim %v71, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v73 = stablehlo.divide %v72, %v69 : tensor<128x16x16x16xf32>
    %v74 = stablehlo.subtract %v67, %v73 : tensor<128x16x16x16xf32>
    %v75 = stablehlo.multiply %v74, %v74 : tensor<128x16x16x16xf32>
    %v76 = stablehlo.reduce(%v75 init: %v68) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v77 = stablehlo.broadcast_in_dim %v76, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v78 = stablehlo.divide %v77, %v69 : tensor<128x16x16x16xf32>
    %v79 = stablehlo.add %v78, %v70 : tensor<128x16x16x16xf32>
    %v80 = stablehlo.rsqrt %v79 : tensor<128x16x16x16xf32>
    %v81 = stablehlo.multiply %v74, %v80 : tensor<128x16x16x16xf32>
    %v82 = stablehlo.broadcast_in_dim %g3, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v83 = stablehlo.broadcast_in_dim %bt3, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v84 = stablehlo.multiply %v81, %v82 : tensor<128x16x16x16xf32>
    %v85 = stablehlo.add %v84, %v83 : tensor<128x16x16x16xf32>
    %v86 = stablehlo.reshape %v85 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v87 = stablehlo.reshape %v86 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v88 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v89 = stablehlo.maximum %v87, %v88 : tensor<128x16x16x16xf32>
    %v90 = stablehlo.reshape %v89 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v91 = stablehlo.reshape %v90 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v92 = stablehlo.convolution(%v91, %W4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v93 = stablehlo.broadcast_in_dim %b4, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v94 = stablehlo.add %v92, %v93 : tensor<128x16x16x16xf32>
    %v95 = stablehlo.reshape %v94 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v96 = stablehlo.reshape %v95 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v97 = stablehlo.constant dense<0.0> : tensor<f32>
    %v98 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v99 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v100 = stablehlo.reduce(%v96 init: %v97) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v101 = stablehlo.broadcast_in_dim %v100, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v102 = stablehlo.divide %v101, %v98 : tensor<128x16x16x16xf32>
    %v103 = stablehlo.subtract %v96, %v102 : tensor<128x16x16x16xf32>
    %v104 = stablehlo.multiply %v103, %v103 : tensor<128x16x16x16xf32>
    %v105 = stablehlo.reduce(%v104 init: %v97) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v106 = stablehlo.broadcast_in_dim %v105, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v107 = stablehlo.divide %v106, %v98 : tensor<128x16x16x16xf32>
    %v108 = stablehlo.add %v107, %v99 : tensor<128x16x16x16xf32>
    %v109 = stablehlo.rsqrt %v108 : tensor<128x16x16x16xf32>
    %v110 = stablehlo.multiply %v103, %v109 : tensor<128x16x16x16xf32>
    %v111 = stablehlo.broadcast_in_dim %g4, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v112 = stablehlo.broadcast_in_dim %bt4, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v113 = stablehlo.multiply %v110, %v111 : tensor<128x16x16x16xf32>
    %v114 = stablehlo.add %v113, %v112 : tensor<128x16x16x16xf32>
    %v115 = stablehlo.reshape %v114 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v116 = stablehlo.reshape %v115 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v117 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v118 = stablehlo.maximum %v116, %v117 : tensor<128x16x16x16xf32>
    %v119 = stablehlo.reshape %v118 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v120 = stablehlo.reshape %v119 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v121 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v122 = "stablehlo.reduce_window"(%v120, %v121) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x8x8xf32>
    %v123 = stablehlo.reshape %v122 : (tensor<128x16x8x8xf32>) -> tensor<128x1024xf32>
    %v124 = stablehlo.reshape %v123 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v125 = stablehlo.convolution(%v124, %W5)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x8x8xf32>, tensor<32x16x3x3xf32>) -> tensor<128x32x8x8xf32>
    %v126 = stablehlo.broadcast_in_dim %b5, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v127 = stablehlo.add %v125, %v126 : tensor<128x32x8x8xf32>
    %v128 = stablehlo.reshape %v127 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v129 = stablehlo.reshape %v128 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v130 = stablehlo.constant dense<0.0> : tensor<f32>
    %v131 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v132 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v133 = stablehlo.reduce(%v129 init: %v130) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v134 = stablehlo.broadcast_in_dim %v133, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v135 = stablehlo.divide %v134, %v131 : tensor<128x32x8x8xf32>
    %v136 = stablehlo.subtract %v129, %v135 : tensor<128x32x8x8xf32>
    %v137 = stablehlo.multiply %v136, %v136 : tensor<128x32x8x8xf32>
    %v138 = stablehlo.reduce(%v137 init: %v130) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v139 = stablehlo.broadcast_in_dim %v138, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v140 = stablehlo.divide %v139, %v131 : tensor<128x32x8x8xf32>
    %v141 = stablehlo.add %v140, %v132 : tensor<128x32x8x8xf32>
    %v142 = stablehlo.rsqrt %v141 : tensor<128x32x8x8xf32>
    %v143 = stablehlo.multiply %v136, %v142 : tensor<128x32x8x8xf32>
    %v144 = stablehlo.broadcast_in_dim %g5, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v145 = stablehlo.broadcast_in_dim %bt5, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v146 = stablehlo.multiply %v143, %v144 : tensor<128x32x8x8xf32>
    %v147 = stablehlo.add %v146, %v145 : tensor<128x32x8x8xf32>
    %v148 = stablehlo.reshape %v147 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v149 = stablehlo.reshape %v148 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v150 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v151 = stablehlo.maximum %v149, %v150 : tensor<128x32x8x8xf32>
    %v152 = stablehlo.reshape %v151 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v153 = stablehlo.reshape %v152 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v154 = stablehlo.convolution(%v153, %W6)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x8x8xf32>
    %v155 = stablehlo.broadcast_in_dim %b6, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v156 = stablehlo.add %v154, %v155 : tensor<128x32x8x8xf32>
    %v157 = stablehlo.reshape %v156 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v158 = stablehlo.reshape %v157 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v159 = stablehlo.constant dense<0.0> : tensor<f32>
    %v160 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v161 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v162 = stablehlo.reduce(%v158 init: %v159) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v163 = stablehlo.broadcast_in_dim %v162, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v164 = stablehlo.divide %v163, %v160 : tensor<128x32x8x8xf32>
    %v165 = stablehlo.subtract %v158, %v164 : tensor<128x32x8x8xf32>
    %v166 = stablehlo.multiply %v165, %v165 : tensor<128x32x8x8xf32>
    %v167 = stablehlo.reduce(%v166 init: %v159) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v168 = stablehlo.broadcast_in_dim %v167, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v169 = stablehlo.divide %v168, %v160 : tensor<128x32x8x8xf32>
    %v170 = stablehlo.add %v169, %v161 : tensor<128x32x8x8xf32>
    %v171 = stablehlo.rsqrt %v170 : tensor<128x32x8x8xf32>
    %v172 = stablehlo.multiply %v165, %v171 : tensor<128x32x8x8xf32>
    %v173 = stablehlo.broadcast_in_dim %g6, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v174 = stablehlo.broadcast_in_dim %bt6, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v175 = stablehlo.multiply %v172, %v173 : tensor<128x32x8x8xf32>
    %v176 = stablehlo.add %v175, %v174 : tensor<128x32x8x8xf32>
    %v177 = stablehlo.reshape %v176 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v178 = stablehlo.reshape %v177 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v179 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v180 = stablehlo.maximum %v178, %v179 : tensor<128x32x8x8xf32>
    %v181 = stablehlo.reshape %v180 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v182 = stablehlo.reshape %v181 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v183 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v184 = "stablehlo.reduce_window"(%v182, %v183) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32x4x4xf32>
    %v185 = stablehlo.reshape %v184 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v186 = stablehlo.reshape %v185 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v187 = stablehlo.convolution(%v186, %W7)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v188 = stablehlo.broadcast_in_dim %b7, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v189 = stablehlo.add %v187, %v188 : tensor<128x32x4x4xf32>
    %v190 = stablehlo.reshape %v189 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v191 = stablehlo.reshape %v190 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v192 = stablehlo.constant dense<0.0> : tensor<f32>
    %v193 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v194 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v195 = stablehlo.reduce(%v191 init: %v192) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v196 = stablehlo.broadcast_in_dim %v195, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v197 = stablehlo.divide %v196, %v193 : tensor<128x32x4x4xf32>
    %v198 = stablehlo.subtract %v191, %v197 : tensor<128x32x4x4xf32>
    %v199 = stablehlo.multiply %v198, %v198 : tensor<128x32x4x4xf32>
    %v200 = stablehlo.reduce(%v199 init: %v192) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v201 = stablehlo.broadcast_in_dim %v200, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v202 = stablehlo.divide %v201, %v193 : tensor<128x32x4x4xf32>
    %v203 = stablehlo.add %v202, %v194 : tensor<128x32x4x4xf32>
    %v204 = stablehlo.rsqrt %v203 : tensor<128x32x4x4xf32>
    %v205 = stablehlo.multiply %v198, %v204 : tensor<128x32x4x4xf32>
    %v206 = stablehlo.broadcast_in_dim %g7, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v207 = stablehlo.broadcast_in_dim %bt7, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v208 = stablehlo.multiply %v205, %v206 : tensor<128x32x4x4xf32>
    %v209 = stablehlo.add %v208, %v207 : tensor<128x32x4x4xf32>
    %v210 = stablehlo.reshape %v209 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v211 = stablehlo.reshape %v210 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v212 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v213 = stablehlo.maximum %v211, %v212 : tensor<128x32x4x4xf32>
    %v214 = stablehlo.reshape %v213 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v215 = stablehlo.reshape %v214 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v216 = stablehlo.convolution(%v215, %W8)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v217 = stablehlo.broadcast_in_dim %b8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v218 = stablehlo.add %v216, %v217 : tensor<128x32x4x4xf32>
    %v219 = stablehlo.reshape %v218 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v220 = stablehlo.reshape %v219 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v221 = stablehlo.constant dense<0.0> : tensor<f32>
    %v222 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v223 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v224 = stablehlo.reduce(%v220 init: %v221) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v225 = stablehlo.broadcast_in_dim %v224, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v226 = stablehlo.divide %v225, %v222 : tensor<128x32x4x4xf32>
    %v227 = stablehlo.subtract %v220, %v226 : tensor<128x32x4x4xf32>
    %v228 = stablehlo.multiply %v227, %v227 : tensor<128x32x4x4xf32>
    %v229 = stablehlo.reduce(%v228 init: %v221) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v230 = stablehlo.broadcast_in_dim %v229, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v231 = stablehlo.divide %v230, %v222 : tensor<128x32x4x4xf32>
    %v232 = stablehlo.add %v231, %v223 : tensor<128x32x4x4xf32>
    %v233 = stablehlo.rsqrt %v232 : tensor<128x32x4x4xf32>
    %v234 = stablehlo.multiply %v227, %v233 : tensor<128x32x4x4xf32>
    %v235 = stablehlo.broadcast_in_dim %g8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v236 = stablehlo.broadcast_in_dim %bt8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v237 = stablehlo.multiply %v234, %v235 : tensor<128x32x4x4xf32>
    %v238 = stablehlo.add %v237, %v236 : tensor<128x32x4x4xf32>
    %v239 = stablehlo.reshape %v238 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v240 = stablehlo.reshape %v239 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v241 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v242 = stablehlo.maximum %v240, %v241 : tensor<128x32x4x4xf32>
    %v243 = stablehlo.reshape %v242 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v244 = stablehlo.reshape %v243 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v245 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v246 = "stablehlo.reduce_window"(%v244, %v245) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32x2x2xf32>
    %v247 = stablehlo.reshape %v246 : (tensor<128x32x2x2xf32>) -> tensor<128x128xf32>
    %v248 = stablehlo.dot_general %v247, %W9, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %v249 = stablehlo.broadcast_in_dim %b9, dims = [1] : (tensor<64xf32>) -> tensor<128x64xf32>
    %v250 = stablehlo.add %v248, %v249 : tensor<128x64xf32>
    %v251 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v252 = stablehlo.maximum %v250, %v251 : tensor<128x64xf32>
    %v253 = stablehlo.dot_general %v252, %Wa, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
    %v254 = stablehlo.broadcast_in_dim %ba, dims = [1] : (tensor<64xf32>) -> tensor<128x64xf32>
    %v255 = stablehlo.add %v253, %v254 : tensor<128x64xf32>
    %v256 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v257 = stablehlo.maximum %v255, %v256 : tensor<128x64xf32>
    %v258 = stablehlo.dot_general %v257, %Wb, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x10xf32>) -> tensor<128x10xf32>
    %v259 = stablehlo.broadcast_in_dim %bb, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v260 = stablehlo.add %v258, %v259 : tensor<128x10xf32>
    %v261 = stablehlo.exponential %v260 : tensor<128x10xf32>
    %v262 = stablehlo.constant dense<0.0> : tensor<f32>
    %v263 = stablehlo.reduce(%v261 init: %v262) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %v264 = stablehlo.broadcast_in_dim %v263, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %v265 = stablehlo.divide %v261, %v264 : tensor<128x10xf32>
    %v266 = stablehlo.subtract %v265, %onehot : tensor<128x10xf32>
    %v267 = stablehlo.dot_general %v266, %Wb, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x10xf32>, tensor<64x10xf32>) -> tensor<128x64xf32>
    %v268 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v269 = stablehlo.compare GT, %v255, %v268 : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xi1>
    %v270 = stablehlo.select %v269, %v267, %v268 : tensor<128x64xi1>, tensor<128x64xf32>
    %v271 = stablehlo.dot_general %v270, %Wa, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
    %v272 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v273 = stablehlo.compare GT, %v250, %v272 : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xi1>
    %v274 = stablehlo.select %v273, %v271, %v272 : tensor<128x64xi1>, tensor<128x64xf32>
    %v275 = stablehlo.dot_general %v274, %W9, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x128xf32>
    %v276 = stablehlo.reshape %v243 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v277 = stablehlo.reshape %v275 : (tensor<128x128xf32>) -> tensor<128x32x2x2xf32>
    %v278 = stablehlo.constant dense<0.0> : tensor<f32>
    %v279 = "stablehlo.select_and_scatter"(%v276, %v277, %v278) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x4x4xf32>, tensor<128x32x2x2xf32>, tensor<f32>) -> tensor<128x32x4x4xf32>
    %v280 = stablehlo.reshape %v279 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v281 = stablehlo.reshape %v280 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v282 = stablehlo.reshape %v239 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v283 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v284 = stablehlo.compare GT, %v282, %v283 : (tensor<128x32x4x4xf32>, tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xi1>
    %v285 = stablehlo.select %v284, %v281, %v283 : tensor<128x32x4x4xi1>, tensor<128x32x4x4xf32>
    %v286 = stablehlo.reshape %v285 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v287 = stablehlo.reshape %v286 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v288 = stablehlo.reshape %v219 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v289 = stablehlo.constant dense<0.0> : tensor<f32>
    %v290 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v291 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v292 = stablehlo.reduce(%v288 init: %v289) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v293 = stablehlo.broadcast_in_dim %v292, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v294 = stablehlo.divide %v293, %v290 : tensor<128x32x4x4xf32>
    %v295 = stablehlo.subtract %v288, %v294 : tensor<128x32x4x4xf32>
    %v296 = stablehlo.multiply %v295, %v295 : tensor<128x32x4x4xf32>
    %v297 = stablehlo.reduce(%v296 init: %v289) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v298 = stablehlo.broadcast_in_dim %v297, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v299 = stablehlo.divide %v298, %v290 : tensor<128x32x4x4xf32>
    %v300 = stablehlo.add %v299, %v291 : tensor<128x32x4x4xf32>
    %v301 = stablehlo.rsqrt %v300 : tensor<128x32x4x4xf32>
    %v302 = stablehlo.multiply %v295, %v301 : tensor<128x32x4x4xf32>
    %v303 = stablehlo.broadcast_in_dim %g8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v304 = stablehlo.multiply %v303, %v287 : tensor<128x32x4x4xf32>
    %v305 = stablehlo.reduce(%v304 init: %v289) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v306 = stablehlo.broadcast_in_dim %v305, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v307 = stablehlo.multiply %v302, %v304 : tensor<128x32x4x4xf32>
    %v308 = stablehlo.reduce(%v307 init: %v289) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v309 = stablehlo.broadcast_in_dim %v308, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v310 = stablehlo.multiply %v304, %v290 : tensor<128x32x4x4xf32>
    %v311 = stablehlo.subtract %v310, %v306 : tensor<128x32x4x4xf32>
    %v312 = stablehlo.multiply %v302, %v309 : tensor<128x32x4x4xf32>
    %v313 = stablehlo.subtract %v311, %v312 : tensor<128x32x4x4xf32>
    %v314 = stablehlo.divide %v301, %v290 : tensor<128x32x4x4xf32>
    %v315 = stablehlo.multiply %v314, %v313 : tensor<128x32x4x4xf32>
    %v316 = stablehlo.reshape %v315 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v317 = stablehlo.reshape %v316 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v318 = stablehlo.transpose %W8, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v319 = stablehlo.reverse %v318, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v320 = stablehlo.convolution(%v317, %v319)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v321 = stablehlo.reshape %v320 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v322 = stablehlo.reshape %v321 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v323 = stablehlo.reshape %v210 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v324 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v325 = stablehlo.compare GT, %v323, %v324 : (tensor<128x32x4x4xf32>, tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xi1>
    %v326 = stablehlo.select %v325, %v322, %v324 : tensor<128x32x4x4xi1>, tensor<128x32x4x4xf32>
    %v327 = stablehlo.reshape %v326 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v328 = stablehlo.reshape %v327 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v329 = stablehlo.reshape %v190 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v330 = stablehlo.constant dense<0.0> : tensor<f32>
    %v331 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v332 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v333 = stablehlo.reduce(%v329 init: %v330) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v334 = stablehlo.broadcast_in_dim %v333, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v335 = stablehlo.divide %v334, %v331 : tensor<128x32x4x4xf32>
    %v336 = stablehlo.subtract %v329, %v335 : tensor<128x32x4x4xf32>
    %v337 = stablehlo.multiply %v336, %v336 : tensor<128x32x4x4xf32>
    %v338 = stablehlo.reduce(%v337 init: %v330) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v339 = stablehlo.broadcast_in_dim %v338, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v340 = stablehlo.divide %v339, %v331 : tensor<128x32x4x4xf32>
    %v341 = stablehlo.add %v340, %v332 : tensor<128x32x4x4xf32>
    %v342 = stablehlo.rsqrt %v341 : tensor<128x32x4x4xf32>
    %v343 = stablehlo.multiply %v336, %v342 : tensor<128x32x4x4xf32>
    %v344 = stablehlo.broadcast_in_dim %g7, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v345 = stablehlo.multiply %v344, %v328 : tensor<128x32x4x4xf32>
    %v346 = stablehlo.reduce(%v345 init: %v330) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v347 = stablehlo.broadcast_in_dim %v346, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v348 = stablehlo.multiply %v343, %v345 : tensor<128x32x4x4xf32>
    %v349 = stablehlo.reduce(%v348 init: %v330) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v350 = stablehlo.broadcast_in_dim %v349, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v351 = stablehlo.multiply %v345, %v331 : tensor<128x32x4x4xf32>
    %v352 = stablehlo.subtract %v351, %v347 : tensor<128x32x4x4xf32>
    %v353 = stablehlo.multiply %v343, %v350 : tensor<128x32x4x4xf32>
    %v354 = stablehlo.subtract %v352, %v353 : tensor<128x32x4x4xf32>
    %v355 = stablehlo.divide %v342, %v331 : tensor<128x32x4x4xf32>
    %v356 = stablehlo.multiply %v355, %v354 : tensor<128x32x4x4xf32>
    %v357 = stablehlo.reshape %v356 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v358 = stablehlo.reshape %v357 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v359 = stablehlo.transpose %W7, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v360 = stablehlo.reverse %v359, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v361 = stablehlo.convolution(%v358, %v360)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v362 = stablehlo.reshape %v361 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v363 = stablehlo.reshape %v181 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v364 = stablehlo.reshape %v362 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v365 = stablehlo.constant dense<0.0> : tensor<f32>
    %v366 = "stablehlo.select_and_scatter"(%v363, %v364, %v365) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x8x8xf32>, tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32x8x8xf32>
    %v367 = stablehlo.reshape %v366 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v368 = stablehlo.reshape %v367 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v369 = stablehlo.reshape %v177 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v370 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v371 = stablehlo.compare GT, %v369, %v370 : (tensor<128x32x8x8xf32>, tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xi1>
    %v372 = stablehlo.select %v371, %v368, %v370 : tensor<128x32x8x8xi1>, tensor<128x32x8x8xf32>
    %v373 = stablehlo.reshape %v372 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v374 = stablehlo.reshape %v373 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v375 = stablehlo.reshape %v157 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v376 = stablehlo.constant dense<0.0> : tensor<f32>
    %v377 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v378 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v379 = stablehlo.reduce(%v375 init: %v376) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v380 = stablehlo.broadcast_in_dim %v379, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v381 = stablehlo.divide %v380, %v377 : tensor<128x32x8x8xf32>
    %v382 = stablehlo.subtract %v375, %v381 : tensor<128x32x8x8xf32>
    %v383 = stablehlo.multiply %v382, %v382 : tensor<128x32x8x8xf32>
    %v384 = stablehlo.reduce(%v383 init: %v376) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v385 = stablehlo.broadcast_in_dim %v384, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v386 = stablehlo.divide %v385, %v377 : tensor<128x32x8x8xf32>
    %v387 = stablehlo.add %v386, %v378 : tensor<128x32x8x8xf32>
    %v388 = stablehlo.rsqrt %v387 : tensor<128x32x8x8xf32>
    %v389 = stablehlo.multiply %v382, %v388 : tensor<128x32x8x8xf32>
    %v390 = stablehlo.broadcast_in_dim %g6, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v391 = stablehlo.multiply %v390, %v374 : tensor<128x32x8x8xf32>
    %v392 = stablehlo.reduce(%v391 init: %v376) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v393 = stablehlo.broadcast_in_dim %v392, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v394 = stablehlo.multiply %v389, %v391 : tensor<128x32x8x8xf32>
    %v395 = stablehlo.reduce(%v394 init: %v376) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v396 = stablehlo.broadcast_in_dim %v395, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v397 = stablehlo.multiply %v391, %v377 : tensor<128x32x8x8xf32>
    %v398 = stablehlo.subtract %v397, %v393 : tensor<128x32x8x8xf32>
    %v399 = stablehlo.multiply %v389, %v396 : tensor<128x32x8x8xf32>
    %v400 = stablehlo.subtract %v398, %v399 : tensor<128x32x8x8xf32>
    %v401 = stablehlo.divide %v388, %v377 : tensor<128x32x8x8xf32>
    %v402 = stablehlo.multiply %v401, %v400 : tensor<128x32x8x8xf32>
    %v403 = stablehlo.reshape %v402 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v404 = stablehlo.reshape %v403 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v405 = stablehlo.transpose %W6, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v406 = stablehlo.reverse %v405, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v407 = stablehlo.convolution(%v404, %v406)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x8x8xf32>
    %v408 = stablehlo.reshape %v407 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v409 = stablehlo.reshape %v408 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v410 = stablehlo.reshape %v148 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v411 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v412 = stablehlo.compare GT, %v410, %v411 : (tensor<128x32x8x8xf32>, tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xi1>
    %v413 = stablehlo.select %v412, %v409, %v411 : tensor<128x32x8x8xi1>, tensor<128x32x8x8xf32>
    %v414 = stablehlo.reshape %v413 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v415 = stablehlo.reshape %v414 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v416 = stablehlo.reshape %v128 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v417 = stablehlo.constant dense<0.0> : tensor<f32>
    %v418 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v419 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v420 = stablehlo.reduce(%v416 init: %v417) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v421 = stablehlo.broadcast_in_dim %v420, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v422 = stablehlo.divide %v421, %v418 : tensor<128x32x8x8xf32>
    %v423 = stablehlo.subtract %v416, %v422 : tensor<128x32x8x8xf32>
    %v424 = stablehlo.multiply %v423, %v423 : tensor<128x32x8x8xf32>
    %v425 = stablehlo.reduce(%v424 init: %v417) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v426 = stablehlo.broadcast_in_dim %v425, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v427 = stablehlo.divide %v426, %v418 : tensor<128x32x8x8xf32>
    %v428 = stablehlo.add %v427, %v419 : tensor<128x32x8x8xf32>
    %v429 = stablehlo.rsqrt %v428 : tensor<128x32x8x8xf32>
    %v430 = stablehlo.multiply %v423, %v429 : tensor<128x32x8x8xf32>
    %v431 = stablehlo.broadcast_in_dim %g5, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v432 = stablehlo.multiply %v431, %v415 : tensor<128x32x8x8xf32>
    %v433 = stablehlo.reduce(%v432 init: %v417) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v434 = stablehlo.broadcast_in_dim %v433, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v435 = stablehlo.multiply %v430, %v432 : tensor<128x32x8x8xf32>
    %v436 = stablehlo.reduce(%v435 init: %v417) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v437 = stablehlo.broadcast_in_dim %v436, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v438 = stablehlo.multiply %v432, %v418 : tensor<128x32x8x8xf32>
    %v439 = stablehlo.subtract %v438, %v434 : tensor<128x32x8x8xf32>
    %v440 = stablehlo.multiply %v430, %v437 : tensor<128x32x8x8xf32>
    %v441 = stablehlo.subtract %v439, %v440 : tensor<128x32x8x8xf32>
    %v442 = stablehlo.divide %v429, %v418 : tensor<128x32x8x8xf32>
    %v443 = stablehlo.multiply %v442, %v441 : tensor<128x32x8x8xf32>
    %v444 = stablehlo.reshape %v443 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v445 = stablehlo.reshape %v444 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v446 = stablehlo.transpose %W5, dims = [1, 0, 2, 3] : (tensor<32x16x3x3xf32>) -> tensor<16x32x3x3xf32>
    %v447 = stablehlo.reverse %v446, dims = [2, 3] : tensor<16x32x3x3xf32>
    %v448 = stablehlo.convolution(%v445, %v447)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<16x32x3x3xf32>) -> tensor<128x16x8x8xf32>
    %v449 = stablehlo.reshape %v448 : (tensor<128x16x8x8xf32>) -> tensor<128x1024xf32>
    %v450 = stablehlo.reshape %v119 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v451 = stablehlo.reshape %v449 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v452 = stablehlo.constant dense<0.0> : tensor<f32>
    %v453 = "stablehlo.select_and_scatter"(%v450, %v451, %v452) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x16x16xf32>, tensor<128x16x8x8xf32>, tensor<f32>) -> tensor<128x16x16x16xf32>
    %v454 = stablehlo.reshape %v453 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v455 = stablehlo.reshape %v454 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v456 = stablehlo.reshape %v115 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v457 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v458 = stablehlo.compare GT, %v456, %v457 : (tensor<128x16x16x16xf32>, tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xi1>
    %v459 = stablehlo.select %v458, %v455, %v457 : tensor<128x16x16x16xi1>, tensor<128x16x16x16xf32>
    %v460 = stablehlo.reshape %v459 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v461 = stablehlo.reshape %v460 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v462 = stablehlo.reshape %v95 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v463 = stablehlo.constant dense<0.0> : tensor<f32>
    %v464 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v465 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v466 = stablehlo.reduce(%v462 init: %v463) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v467 = stablehlo.broadcast_in_dim %v466, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v468 = stablehlo.divide %v467, %v464 : tensor<128x16x16x16xf32>
    %v469 = stablehlo.subtract %v462, %v468 : tensor<128x16x16x16xf32>
    %v470 = stablehlo.multiply %v469, %v469 : tensor<128x16x16x16xf32>
    %v471 = stablehlo.reduce(%v470 init: %v463) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v472 = stablehlo.broadcast_in_dim %v471, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v473 = stablehlo.divide %v472, %v464 : tensor<128x16x16x16xf32>
    %v474 = stablehlo.add %v473, %v465 : tensor<128x16x16x16xf32>
    %v475 = stablehlo.rsqrt %v474 : tensor<128x16x16x16xf32>
    %v476 = stablehlo.multiply %v469, %v475 : tensor<128x16x16x16xf32>
    %v477 = stablehlo.broadcast_in_dim %g4, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v478 = stablehlo.multiply %v477, %v461 : tensor<128x16x16x16xf32>
    %v479 = stablehlo.reduce(%v478 init: %v463) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v480 = stablehlo.broadcast_in_dim %v479, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v481 = stablehlo.multiply %v476, %v478 : tensor<128x16x16x16xf32>
    %v482 = stablehlo.reduce(%v481 init: %v463) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v483 = stablehlo.broadcast_in_dim %v482, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v484 = stablehlo.multiply %v478, %v464 : tensor<128x16x16x16xf32>
    %v485 = stablehlo.subtract %v484, %v480 : tensor<128x16x16x16xf32>
    %v486 = stablehlo.multiply %v476, %v483 : tensor<128x16x16x16xf32>
    %v487 = stablehlo.subtract %v485, %v486 : tensor<128x16x16x16xf32>
    %v488 = stablehlo.divide %v475, %v464 : tensor<128x16x16x16xf32>
    %v489 = stablehlo.multiply %v488, %v487 : tensor<128x16x16x16xf32>
    %v490 = stablehlo.reshape %v489 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v491 = stablehlo.reshape %v490 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v492 = stablehlo.transpose %W4, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v493 = stablehlo.reverse %v492, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v494 = stablehlo.convolution(%v491, %v493)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v495 = stablehlo.reshape %v494 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v496 = stablehlo.reshape %v495 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v497 = stablehlo.reshape %v86 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v498 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v499 = stablehlo.compare GT, %v497, %v498 : (tensor<128x16x16x16xf32>, tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xi1>
    %v500 = stablehlo.select %v499, %v496, %v498 : tensor<128x16x16x16xi1>, tensor<128x16x16x16xf32>
    %v501 = stablehlo.reshape %v500 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v502 = stablehlo.reshape %v501 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v503 = stablehlo.reshape %v66 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v504 = stablehlo.constant dense<0.0> : tensor<f32>
    %v505 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v506 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v507 = stablehlo.reduce(%v503 init: %v504) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v508 = stablehlo.broadcast_in_dim %v507, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v509 = stablehlo.divide %v508, %v505 : tensor<128x16x16x16xf32>
    %v510 = stablehlo.subtract %v503, %v509 : tensor<128x16x16x16xf32>
    %v511 = stablehlo.multiply %v510, %v510 : tensor<128x16x16x16xf32>
    %v512 = stablehlo.reduce(%v511 init: %v504) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v513 = stablehlo.broadcast_in_dim %v512, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v514 = stablehlo.divide %v513, %v505 : tensor<128x16x16x16xf32>
    %v515 = stablehlo.add %v514, %v506 : tensor<128x16x16x16xf32>
    %v516 = stablehlo.rsqrt %v515 : tensor<128x16x16x16xf32>
    %v517 = stablehlo.multiply %v510, %v516 : tensor<128x16x16x16xf32>
    %v518 = stablehlo.broadcast_in_dim %g3, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v519 = stablehlo.multiply %v518, %v502 : tensor<128x16x16x16xf32>
    %v520 = stablehlo.reduce(%v519 init: %v504) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v521 = stablehlo.broadcast_in_dim %v520, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v522 = stablehlo.multiply %v517, %v519 : tensor<128x16x16x16xf32>
    %v523 = stablehlo.reduce(%v522 init: %v504) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v524 = stablehlo.broadcast_in_dim %v523, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v525 = stablehlo.multiply %v519, %v505 : tensor<128x16x16x16xf32>
    %v526 = stablehlo.subtract %v525, %v521 : tensor<128x16x16x16xf32>
    %v527 = stablehlo.multiply %v517, %v524 : tensor<128x16x16x16xf32>
    %v528 = stablehlo.subtract %v526, %v527 : tensor<128x16x16x16xf32>
    %v529 = stablehlo.divide %v516, %v505 : tensor<128x16x16x16xf32>
    %v530 = stablehlo.multiply %v529, %v528 : tensor<128x16x16x16xf32>
    %v531 = stablehlo.reshape %v530 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v532 = stablehlo.reshape %v531 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v533 = stablehlo.transpose %W3, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v534 = stablehlo.reverse %v533, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v535 = stablehlo.convolution(%v532, %v534)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v536 = stablehlo.reshape %v535 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v537 = stablehlo.reshape %v57 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v538 = stablehlo.reshape %v536 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v539 = stablehlo.constant dense<0.0> : tensor<f32>
    %v540 = "stablehlo.select_and_scatter"(%v537, %v538, %v539) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x32x32xf32>, tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x32x32xf32>
    %v541 = stablehlo.reshape %v540 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v542 = stablehlo.reshape %v541 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v543 = stablehlo.reshape %v53 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v544 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v545 = stablehlo.compare GT, %v543, %v544 : (tensor<128x16x32x32xf32>, tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xi1>
    %v546 = stablehlo.select %v545, %v542, %v544 : tensor<128x16x32x32xi1>, tensor<128x16x32x32xf32>
    %v547 = stablehlo.reshape %v546 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v548 = stablehlo.reshape %v547 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v549 = stablehlo.reshape %v33 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v550 = stablehlo.constant dense<0.0> : tensor<f32>
    %v551 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v552 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v553 = stablehlo.reduce(%v549 init: %v550) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v554 = stablehlo.broadcast_in_dim %v553, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v555 = stablehlo.divide %v554, %v551 : tensor<128x16x32x32xf32>
    %v556 = stablehlo.subtract %v549, %v555 : tensor<128x16x32x32xf32>
    %v557 = stablehlo.multiply %v556, %v556 : tensor<128x16x32x32xf32>
    %v558 = stablehlo.reduce(%v557 init: %v550) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v559 = stablehlo.broadcast_in_dim %v558, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v560 = stablehlo.divide %v559, %v551 : tensor<128x16x32x32xf32>
    %v561 = stablehlo.add %v560, %v552 : tensor<128x16x32x32xf32>
    %v562 = stablehlo.rsqrt %v561 : tensor<128x16x32x32xf32>
    %v563 = stablehlo.multiply %v556, %v562 : tensor<128x16x32x32xf32>
    %v564 = stablehlo.broadcast_in_dim %g2, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v565 = stablehlo.multiply %v564, %v548 : tensor<128x16x32x32xf32>
    %v566 = stablehlo.reduce(%v565 init: %v550) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v567 = stablehlo.broadcast_in_dim %v566, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v568 = stablehlo.multiply %v563, %v565 : tensor<128x16x32x32xf32>
    %v569 = stablehlo.reduce(%v568 init: %v550) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v570 = stablehlo.broadcast_in_dim %v569, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v571 = stablehlo.multiply %v565, %v551 : tensor<128x16x32x32xf32>
    %v572 = stablehlo.subtract %v571, %v567 : tensor<128x16x32x32xf32>
    %v573 = stablehlo.multiply %v563, %v570 : tensor<128x16x32x32xf32>
    %v574 = stablehlo.subtract %v572, %v573 : tensor<128x16x32x32xf32>
    %v575 = stablehlo.divide %v562, %v551 : tensor<128x16x32x32xf32>
    %v576 = stablehlo.multiply %v575, %v574 : tensor<128x16x32x32xf32>
    %v577 = stablehlo.reshape %v576 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v578 = stablehlo.reshape %v577 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v579 = stablehlo.transpose %W2, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v580 = stablehlo.reverse %v579, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v581 = stablehlo.convolution(%v578, %v580)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x32x32xf32>
    %v582 = stablehlo.reshape %v581 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v583 = stablehlo.reshape %v582 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v584 = stablehlo.reshape %v24 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v585 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v586 = stablehlo.compare GT, %v584, %v585 : (tensor<128x16x32x32xf32>, tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xi1>
    %v587 = stablehlo.select %v586, %v583, %v585 : tensor<128x16x32x32xi1>, tensor<128x16x32x32xf32>
    %v588 = stablehlo.reshape %v587 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v589 = stablehlo.reshape %v588 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v590 = stablehlo.reshape %v4 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v591 = stablehlo.constant dense<0.0> : tensor<f32>
    %v592 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v593 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v594 = stablehlo.reduce(%v590 init: %v591) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v595 = stablehlo.broadcast_in_dim %v594, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v596 = stablehlo.divide %v595, %v592 : tensor<128x16x32x32xf32>
    %v597 = stablehlo.subtract %v590, %v596 : tensor<128x16x32x32xf32>
    %v598 = stablehlo.multiply %v597, %v597 : tensor<128x16x32x32xf32>
    %v599 = stablehlo.reduce(%v598 init: %v591) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v600 = stablehlo.broadcast_in_dim %v599, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v601 = stablehlo.divide %v600, %v592 : tensor<128x16x32x32xf32>
    %v602 = stablehlo.add %v601, %v593 : tensor<128x16x32x32xf32>
    %v603 = stablehlo.rsqrt %v602 : tensor<128x16x32x32xf32>
    %v604 = stablehlo.multiply %v597, %v603 : tensor<128x16x32x32xf32>
    %v605 = stablehlo.broadcast_in_dim %g1, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v606 = stablehlo.multiply %v605, %v589 : tensor<128x16x32x32xf32>
    %v607 = stablehlo.reduce(%v606 init: %v591) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v608 = stablehlo.broadcast_in_dim %v607, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v609 = stablehlo.multiply %v604, %v606 : tensor<128x16x32x32xf32>
    %v610 = stablehlo.reduce(%v609 init: %v591) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v611 = stablehlo.broadcast_in_dim %v610, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v612 = stablehlo.multiply %v606, %v592 : tensor<128x16x32x32xf32>
    %v613 = stablehlo.subtract %v612, %v608 : tensor<128x16x32x32xf32>
    %v614 = stablehlo.multiply %v604, %v611 : tensor<128x16x32x32xf32>
    %v615 = stablehlo.subtract %v613, %v614 : tensor<128x16x32x32xf32>
    %v616 = stablehlo.divide %v603, %v592 : tensor<128x16x32x32xf32>
    %v617 = stablehlo.multiply %v616, %v615 : tensor<128x16x32x32xf32>
    %v618 = stablehlo.reshape %v617 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v619 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v620 = stablehlo.reshape %v618 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v621 = stablehlo.transpose %v619, dims = [1, 0, 2, 3] : (tensor<128x3x32x32xf32>) -> tensor<3x128x32x32xf32>
    %v622 = stablehlo.transpose %v620, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v623 = stablehlo.convolution(%v621, %v622)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<3x16x3x3xf32>
    %v624 = stablehlo.transpose %v623, dims = [1, 0, 2, 3] : (tensor<3x16x3x3xf32>) -> tensor<16x3x3x3xf32>
    %v625 = stablehlo.constant dense<0.00078125> : tensor<16x3x3x3xf32>
    %v626 = stablehlo.multiply %v624, %v625 : tensor<16x3x3x3xf32>
    %v627 = stablehlo.subtract %W1, %v626 : tensor<16x3x3x3xf32>
    %v628 = stablehlo.reshape %v618 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v629 = stablehlo.constant dense<0.0> : tensor<f32>
    %v630 = stablehlo.reduce(%v628 init: %v629) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v631 = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %v632 = stablehlo.multiply %v630, %v631 : tensor<16xf32>
    %v633 = stablehlo.subtract %b1, %v632 : tensor<16xf32>
    %v634 = stablehlo.constant dense<0.0> : tensor<f32>
    %v635 = stablehlo.reshape %v4 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v636 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v637 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v638 = stablehlo.reduce(%v635 init: %v634) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v639 = stablehlo.broadcast_in_dim %v638, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v640 = stablehlo.divide %v639, %v636 : tensor<128x16x32x32xf32>
    %v641 = stablehlo.subtract %v635, %v640 : tensor<128x16x32x32xf32>
    %v642 = stablehlo.multiply %v641, %v641 : tensor<128x16x32x32xf32>
    %v643 = stablehlo.reduce(%v642 init: %v634) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v644 = stablehlo.broadcast_in_dim %v643, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v645 = stablehlo.divide %v644, %v636 : tensor<128x16x32x32xf32>
    %v646 = stablehlo.add %v645, %v637 : tensor<128x16x32x32xf32>
    %v647 = stablehlo.rsqrt %v646 : tensor<128x16x32x32xf32>
    %v648 = stablehlo.multiply %v641, %v647 : tensor<128x16x32x32xf32>
    %v649 = stablehlo.reshape %v588 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v650 = stablehlo.multiply %v649, %v648 : tensor<128x16x32x32xf32>
    %v651 = stablehlo.reduce(%v650 init: %v634) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v652 = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %v653 = stablehlo.multiply %v651, %v652 : tensor<16xf32>
    %v654 = stablehlo.subtract %g1, %v653 : tensor<16xf32>
    %v655 = stablehlo.constant dense<0.0> : tensor<f32>
    %v656 = stablehlo.reshape %v588 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v657 = stablehlo.reduce(%v656 init: %v655) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v658 = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %v659 = stablehlo.multiply %v657, %v658 : tensor<16xf32>
    %v660 = stablehlo.subtract %bt1, %v659 : tensor<16xf32>
    %v661 = stablehlo.reshape %v28 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v662 = stablehlo.reshape %v577 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v663 = stablehlo.transpose %v661, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v664 = stablehlo.transpose %v662, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v665 = stablehlo.convolution(%v663, %v664)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x16x3x3xf32>
    %v666 = stablehlo.transpose %v665, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v667 = stablehlo.constant dense<0.00078125> : tensor<16x16x3x3xf32>
    %v668 = stablehlo.multiply %v666, %v667 : tensor<16x16x3x3xf32>
    %v669 = stablehlo.subtract %W2, %v668 : tensor<16x16x3x3xf32>
    %v670 = stablehlo.reshape %v577 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v671 = stablehlo.constant dense<0.0> : tensor<f32>
    %v672 = stablehlo.reduce(%v670 init: %v671) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v673 = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %v674 = stablehlo.multiply %v672, %v673 : tensor<16xf32>
    %v675 = stablehlo.subtract %b2, %v674 : tensor<16xf32>
    %v676 = stablehlo.constant dense<0.0> : tensor<f32>
    %v677 = stablehlo.reshape %v33 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v678 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v679 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v680 = stablehlo.reduce(%v677 init: %v676) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v681 = stablehlo.broadcast_in_dim %v680, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v682 = stablehlo.divide %v681, %v678 : tensor<128x16x32x32xf32>
    %v683 = stablehlo.subtract %v677, %v682 : tensor<128x16x32x32xf32>
    %v684 = stablehlo.multiply %v683, %v683 : tensor<128x16x32x32xf32>
    %v685 = stablehlo.reduce(%v684 init: %v676) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v686 = stablehlo.broadcast_in_dim %v685, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v687 = stablehlo.divide %v686, %v678 : tensor<128x16x32x32xf32>
    %v688 = stablehlo.add %v687, %v679 : tensor<128x16x32x32xf32>
    %v689 = stablehlo.rsqrt %v688 : tensor<128x16x32x32xf32>
    %v690 = stablehlo.multiply %v683, %v689 : tensor<128x16x32x32xf32>
    %v691 = stablehlo.reshape %v547 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v692 = stablehlo.multiply %v691, %v690 : tensor<128x16x32x32xf32>
    %v693 = stablehlo.reduce(%v692 init: %v676) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v694 = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %v695 = stablehlo.multiply %v693, %v694 : tensor<16xf32>
    %v696 = stablehlo.subtract %g2, %v695 : tensor<16xf32>
    %v697 = stablehlo.constant dense<0.0> : tensor<f32>
    %v698 = stablehlo.reshape %v547 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v699 = stablehlo.reduce(%v698 init: %v697) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v700 = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %v701 = stablehlo.multiply %v699, %v700 : tensor<16xf32>
    %v702 = stablehlo.subtract %bt2, %v701 : tensor<16xf32>
    %v703 = stablehlo.reshape %v61 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v704 = stablehlo.reshape %v531 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v705 = stablehlo.transpose %v703, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v706 = stablehlo.transpose %v704, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v707 = stablehlo.convolution(%v705, %v706)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v708 = stablehlo.transpose %v707, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v709 = stablehlo.constant dense<0.00078125> : tensor<16x16x3x3xf32>
    %v710 = stablehlo.multiply %v708, %v709 : tensor<16x16x3x3xf32>
    %v711 = stablehlo.subtract %W3, %v710 : tensor<16x16x3x3xf32>
    %v712 = stablehlo.reshape %v531 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v713 = stablehlo.constant dense<0.0> : tensor<f32>
    %v714 = stablehlo.reduce(%v712 init: %v713) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v715 = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %v716 = stablehlo.multiply %v714, %v715 : tensor<16xf32>
    %v717 = stablehlo.subtract %b3, %v716 : tensor<16xf32>
    %v718 = stablehlo.constant dense<0.0> : tensor<f32>
    %v719 = stablehlo.reshape %v66 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v720 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v721 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v722 = stablehlo.reduce(%v719 init: %v718) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v723 = stablehlo.broadcast_in_dim %v722, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v724 = stablehlo.divide %v723, %v720 : tensor<128x16x16x16xf32>
    %v725 = stablehlo.subtract %v719, %v724 : tensor<128x16x16x16xf32>
    %v726 = stablehlo.multiply %v725, %v725 : tensor<128x16x16x16xf32>
    %v727 = stablehlo.reduce(%v726 init: %v718) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v728 = stablehlo.broadcast_in_dim %v727, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v729 = stablehlo.divide %v728, %v720 : tensor<128x16x16x16xf32>
    %v730 = stablehlo.add %v729, %v721 : tensor<128x16x16x16xf32>
    %v731 = stablehlo.rsqrt %v730 : tensor<128x16x16x16xf32>
    %v732 = stablehlo.multiply %v725, %v731 : tensor<128x16x16x16xf32>
    %v733 = stablehlo.reshape %v501 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v734 = stablehlo.multiply %v733, %v732 : tensor<128x16x16x16xf32>
    %v735 = stablehlo.reduce(%v734 init: %v718) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v736 = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %v737 = stablehlo.multiply %v735, %v736 : tensor<16xf32>
    %v738 = stablehlo.subtract %g3, %v737 : tensor<16xf32>
    %v739 = stablehlo.constant dense<0.0> : tensor<f32>
    %v740 = stablehlo.reshape %v501 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v741 = stablehlo.reduce(%v740 init: %v739) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v742 = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %v743 = stablehlo.multiply %v741, %v742 : tensor<16xf32>
    %v744 = stablehlo.subtract %bt3, %v743 : tensor<16xf32>
    %v745 = stablehlo.reshape %v90 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v746 = stablehlo.reshape %v490 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v747 = stablehlo.transpose %v745, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v748 = stablehlo.transpose %v746, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v749 = stablehlo.convolution(%v747, %v748)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v750 = stablehlo.transpose %v749, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v751 = stablehlo.constant dense<0.00078125> : tensor<16x16x3x3xf32>
    %v752 = stablehlo.multiply %v750, %v751 : tensor<16x16x3x3xf32>
    %v753 = stablehlo.subtract %W4, %v752 : tensor<16x16x3x3xf32>
    %v754 = stablehlo.reshape %v490 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v755 = stablehlo.constant dense<0.0> : tensor<f32>
    %v756 = stablehlo.reduce(%v754 init: %v755) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v757 = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %v758 = stablehlo.multiply %v756, %v757 : tensor<16xf32>
    %v759 = stablehlo.subtract %b4, %v758 : tensor<16xf32>
    %v760 = stablehlo.constant dense<0.0> : tensor<f32>
    %v761 = stablehlo.reshape %v95 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v762 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v763 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v764 = stablehlo.reduce(%v761 init: %v760) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v765 = stablehlo.broadcast_in_dim %v764, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v766 = stablehlo.divide %v765, %v762 : tensor<128x16x16x16xf32>
    %v767 = stablehlo.subtract %v761, %v766 : tensor<128x16x16x16xf32>
    %v768 = stablehlo.multiply %v767, %v767 : tensor<128x16x16x16xf32>
    %v769 = stablehlo.reduce(%v768 init: %v760) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v770 = stablehlo.broadcast_in_dim %v769, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v771 = stablehlo.divide %v770, %v762 : tensor<128x16x16x16xf32>
    %v772 = stablehlo.add %v771, %v763 : tensor<128x16x16x16xf32>
    %v773 = stablehlo.rsqrt %v772 : tensor<128x16x16x16xf32>
    %v774 = stablehlo.multiply %v767, %v773 : tensor<128x16x16x16xf32>
    %v775 = stablehlo.reshape %v460 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v776 = stablehlo.multiply %v775, %v774 : tensor<128x16x16x16xf32>
    %v777 = stablehlo.reduce(%v776 init: %v760) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v778 = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %v779 = stablehlo.multiply %v777, %v778 : tensor<16xf32>
    %v780 = stablehlo.subtract %g4, %v779 : tensor<16xf32>
    %v781 = stablehlo.constant dense<0.0> : tensor<f32>
    %v782 = stablehlo.reshape %v460 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v783 = stablehlo.reduce(%v782 init: %v781) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v784 = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %v785 = stablehlo.multiply %v783, %v784 : tensor<16xf32>
    %v786 = stablehlo.subtract %bt4, %v785 : tensor<16xf32>
    %v787 = stablehlo.reshape %v123 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v788 = stablehlo.reshape %v444 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v789 = stablehlo.transpose %v787, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v790 = stablehlo.transpose %v788, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v791 = stablehlo.convolution(%v789, %v790)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<16x32x3x3xf32>
    %v792 = stablehlo.transpose %v791, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v793 = stablehlo.constant dense<0.00078125> : tensor<32x16x3x3xf32>
    %v794 = stablehlo.multiply %v792, %v793 : tensor<32x16x3x3xf32>
    %v795 = stablehlo.subtract %W5, %v794 : tensor<32x16x3x3xf32>
    %v796 = stablehlo.reshape %v444 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v797 = stablehlo.constant dense<0.0> : tensor<f32>
    %v798 = stablehlo.reduce(%v796 init: %v797) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v799 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v800 = stablehlo.multiply %v798, %v799 : tensor<32xf32>
    %v801 = stablehlo.subtract %b5, %v800 : tensor<32xf32>
    %v802 = stablehlo.constant dense<0.0> : tensor<f32>
    %v803 = stablehlo.reshape %v128 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v804 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v805 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v806 = stablehlo.reduce(%v803 init: %v802) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v807 = stablehlo.broadcast_in_dim %v806, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v808 = stablehlo.divide %v807, %v804 : tensor<128x32x8x8xf32>
    %v809 = stablehlo.subtract %v803, %v808 : tensor<128x32x8x8xf32>
    %v810 = stablehlo.multiply %v809, %v809 : tensor<128x32x8x8xf32>
    %v811 = stablehlo.reduce(%v810 init: %v802) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v812 = stablehlo.broadcast_in_dim %v811, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v813 = stablehlo.divide %v812, %v804 : tensor<128x32x8x8xf32>
    %v814 = stablehlo.add %v813, %v805 : tensor<128x32x8x8xf32>
    %v815 = stablehlo.rsqrt %v814 : tensor<128x32x8x8xf32>
    %v816 = stablehlo.multiply %v809, %v815 : tensor<128x32x8x8xf32>
    %v817 = stablehlo.reshape %v414 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v818 = stablehlo.multiply %v817, %v816 : tensor<128x32x8x8xf32>
    %v819 = stablehlo.reduce(%v818 init: %v802) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v820 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v821 = stablehlo.multiply %v819, %v820 : tensor<32xf32>
    %v822 = stablehlo.subtract %g5, %v821 : tensor<32xf32>
    %v823 = stablehlo.constant dense<0.0> : tensor<f32>
    %v824 = stablehlo.reshape %v414 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v825 = stablehlo.reduce(%v824 init: %v823) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v826 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v827 = stablehlo.multiply %v825, %v826 : tensor<32xf32>
    %v828 = stablehlo.subtract %bt5, %v827 : tensor<32xf32>
    %v829 = stablehlo.reshape %v152 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v830 = stablehlo.reshape %v403 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v831 = stablehlo.transpose %v829, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v832 = stablehlo.transpose %v830, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v833 = stablehlo.convolution(%v831, %v832)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<32x32x3x3xf32>
    %v834 = stablehlo.transpose %v833, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v835 = stablehlo.constant dense<0.00078125> : tensor<32x32x3x3xf32>
    %v836 = stablehlo.multiply %v834, %v835 : tensor<32x32x3x3xf32>
    %v837 = stablehlo.subtract %W6, %v836 : tensor<32x32x3x3xf32>
    %v838 = stablehlo.reshape %v403 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v839 = stablehlo.constant dense<0.0> : tensor<f32>
    %v840 = stablehlo.reduce(%v838 init: %v839) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v841 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v842 = stablehlo.multiply %v840, %v841 : tensor<32xf32>
    %v843 = stablehlo.subtract %b6, %v842 : tensor<32xf32>
    %v844 = stablehlo.constant dense<0.0> : tensor<f32>
    %v845 = stablehlo.reshape %v157 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v846 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v847 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v848 = stablehlo.reduce(%v845 init: %v844) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v849 = stablehlo.broadcast_in_dim %v848, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v850 = stablehlo.divide %v849, %v846 : tensor<128x32x8x8xf32>
    %v851 = stablehlo.subtract %v845, %v850 : tensor<128x32x8x8xf32>
    %v852 = stablehlo.multiply %v851, %v851 : tensor<128x32x8x8xf32>
    %v853 = stablehlo.reduce(%v852 init: %v844) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v854 = stablehlo.broadcast_in_dim %v853, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v855 = stablehlo.divide %v854, %v846 : tensor<128x32x8x8xf32>
    %v856 = stablehlo.add %v855, %v847 : tensor<128x32x8x8xf32>
    %v857 = stablehlo.rsqrt %v856 : tensor<128x32x8x8xf32>
    %v858 = stablehlo.multiply %v851, %v857 : tensor<128x32x8x8xf32>
    %v859 = stablehlo.reshape %v373 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v860 = stablehlo.multiply %v859, %v858 : tensor<128x32x8x8xf32>
    %v861 = stablehlo.reduce(%v860 init: %v844) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v862 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v863 = stablehlo.multiply %v861, %v862 : tensor<32xf32>
    %v864 = stablehlo.subtract %g6, %v863 : tensor<32xf32>
    %v865 = stablehlo.constant dense<0.0> : tensor<f32>
    %v866 = stablehlo.reshape %v373 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v867 = stablehlo.reduce(%v866 init: %v865) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v868 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v869 = stablehlo.multiply %v867, %v868 : tensor<32xf32>
    %v870 = stablehlo.subtract %bt6, %v869 : tensor<32xf32>
    %v871 = stablehlo.reshape %v185 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v872 = stablehlo.reshape %v357 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v873 = stablehlo.transpose %v871, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v874 = stablehlo.transpose %v872, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v875 = stablehlo.convolution(%v873, %v874)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v876 = stablehlo.transpose %v875, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v877 = stablehlo.constant dense<0.00078125> : tensor<32x32x3x3xf32>
    %v878 = stablehlo.multiply %v876, %v877 : tensor<32x32x3x3xf32>
    %v879 = stablehlo.subtract %W7, %v878 : tensor<32x32x3x3xf32>
    %v880 = stablehlo.reshape %v357 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v881 = stablehlo.constant dense<0.0> : tensor<f32>
    %v882 = stablehlo.reduce(%v880 init: %v881) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v883 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v884 = stablehlo.multiply %v882, %v883 : tensor<32xf32>
    %v885 = stablehlo.subtract %b7, %v884 : tensor<32xf32>
    %v886 = stablehlo.constant dense<0.0> : tensor<f32>
    %v887 = stablehlo.reshape %v190 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v888 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v889 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v890 = stablehlo.reduce(%v887 init: %v886) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v891 = stablehlo.broadcast_in_dim %v890, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v892 = stablehlo.divide %v891, %v888 : tensor<128x32x4x4xf32>
    %v893 = stablehlo.subtract %v887, %v892 : tensor<128x32x4x4xf32>
    %v894 = stablehlo.multiply %v893, %v893 : tensor<128x32x4x4xf32>
    %v895 = stablehlo.reduce(%v894 init: %v886) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v896 = stablehlo.broadcast_in_dim %v895, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v897 = stablehlo.divide %v896, %v888 : tensor<128x32x4x4xf32>
    %v898 = stablehlo.add %v897, %v889 : tensor<128x32x4x4xf32>
    %v899 = stablehlo.rsqrt %v898 : tensor<128x32x4x4xf32>
    %v900 = stablehlo.multiply %v893, %v899 : tensor<128x32x4x4xf32>
    %v901 = stablehlo.reshape %v327 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v902 = stablehlo.multiply %v901, %v900 : tensor<128x32x4x4xf32>
    %v903 = stablehlo.reduce(%v902 init: %v886) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v904 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v905 = stablehlo.multiply %v903, %v904 : tensor<32xf32>
    %v906 = stablehlo.subtract %g7, %v905 : tensor<32xf32>
    %v907 = stablehlo.constant dense<0.0> : tensor<f32>
    %v908 = stablehlo.reshape %v327 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v909 = stablehlo.reduce(%v908 init: %v907) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v910 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v911 = stablehlo.multiply %v909, %v910 : tensor<32xf32>
    %v912 = stablehlo.subtract %bt7, %v911 : tensor<32xf32>
    %v913 = stablehlo.reshape %v214 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v914 = stablehlo.reshape %v316 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v915 = stablehlo.transpose %v913, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v916 = stablehlo.transpose %v914, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v917 = stablehlo.convolution(%v915, %v916)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v918 = stablehlo.transpose %v917, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v919 = stablehlo.constant dense<0.00078125> : tensor<32x32x3x3xf32>
    %v920 = stablehlo.multiply %v918, %v919 : tensor<32x32x3x3xf32>
    %v921 = stablehlo.subtract %W8, %v920 : tensor<32x32x3x3xf32>
    %v922 = stablehlo.reshape %v316 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v923 = stablehlo.constant dense<0.0> : tensor<f32>
    %v924 = stablehlo.reduce(%v922 init: %v923) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v925 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v926 = stablehlo.multiply %v924, %v925 : tensor<32xf32>
    %v927 = stablehlo.subtract %b8, %v926 : tensor<32xf32>
    %v928 = stablehlo.constant dense<0.0> : tensor<f32>
    %v929 = stablehlo.reshape %v219 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v930 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v931 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v932 = stablehlo.reduce(%v929 init: %v928) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v933 = stablehlo.broadcast_in_dim %v932, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v934 = stablehlo.divide %v933, %v930 : tensor<128x32x4x4xf32>
    %v935 = stablehlo.subtract %v929, %v934 : tensor<128x32x4x4xf32>
    %v936 = stablehlo.multiply %v935, %v935 : tensor<128x32x4x4xf32>
    %v937 = stablehlo.reduce(%v936 init: %v928) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v938 = stablehlo.broadcast_in_dim %v937, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v939 = stablehlo.divide %v938, %v930 : tensor<128x32x4x4xf32>
    %v940 = stablehlo.add %v939, %v931 : tensor<128x32x4x4xf32>
    %v941 = stablehlo.rsqrt %v940 : tensor<128x32x4x4xf32>
    %v942 = stablehlo.multiply %v935, %v941 : tensor<128x32x4x4xf32>
    %v943 = stablehlo.reshape %v286 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v944 = stablehlo.multiply %v943, %v942 : tensor<128x32x4x4xf32>
    %v945 = stablehlo.reduce(%v944 init: %v928) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v946 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v947 = stablehlo.multiply %v945, %v946 : tensor<32xf32>
    %v948 = stablehlo.subtract %g8, %v947 : tensor<32xf32>
    %v949 = stablehlo.constant dense<0.0> : tensor<f32>
    %v950 = stablehlo.reshape %v286 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v951 = stablehlo.reduce(%v950 init: %v949) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v952 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v953 = stablehlo.multiply %v951, %v952 : tensor<32xf32>
    %v954 = stablehlo.subtract %bt8, %v953 : tensor<32xf32>
    %v955 = stablehlo.dot_general %v247, %v274, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %v956 = stablehlo.constant dense<0.00078125> : tensor<128x64xf32>
    %v957 = stablehlo.multiply %v955, %v956 : tensor<128x64xf32>
    %v958 = stablehlo.subtract %W9, %v957 : tensor<128x64xf32>
    %v959 = stablehlo.constant dense<0.0> : tensor<f32>
    %v960 = stablehlo.reduce(%v274 init: %v959) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v961 = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %v962 = stablehlo.multiply %v960, %v961 : tensor<64xf32>
    %v963 = stablehlo.subtract %b9, %v962 : tensor<64xf32>
    %v964 = stablehlo.dot_general %v252, %v270, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<64x64xf32>
    %v965 = stablehlo.constant dense<0.00078125> : tensor<64x64xf32>
    %v966 = stablehlo.multiply %v964, %v965 : tensor<64x64xf32>
    %v967 = stablehlo.subtract %Wa, %v966 : tensor<64x64xf32>
    %v968 = stablehlo.constant dense<0.0> : tensor<f32>
    %v969 = stablehlo.reduce(%v270 init: %v968) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v970 = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %v971 = stablehlo.multiply %v969, %v970 : tensor<64xf32>
    %v972 = stablehlo.subtract %ba, %v971 : tensor<64xf32>
    %v973 = stablehlo.dot_general %v257, %v266, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x10xf32>) -> tensor<64x10xf32>
    %v974 = stablehlo.constant dense<0.00078125> : tensor<64x10xf32>
    %v975 = stablehlo.multiply %v973, %v974 : tensor<64x10xf32>
    %v976 = stablehlo.subtract %Wb, %v975 : tensor<64x10xf32>
    %v977 = stablehlo.constant dense<0.0> : tensor<f32>
    %v978 = stablehlo.reduce(%v266 init: %v977) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v979 = stablehlo.constant dense<0.00078125> : tensor<10xf32>
    %v980 = stablehlo.multiply %v978, %v979 : tensor<10xf32>
    %v981 = stablehlo.subtract %bb, %v980 : tensor<10xf32>
    return %v627, %v633, %v654, %v660, %v669, %v675, %v696, %v702, %v711, %v717, %v738, %v744, %v753, %v759, %v780, %v786, %v795, %v801, %v822, %v828, %v837, %v843, %v864, %v870, %v879, %v885, %v906, %v912, %v921, %v927, %v948, %v954, %v958, %v963, %v967, %v972, %v976, %v981 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>
  }
}
