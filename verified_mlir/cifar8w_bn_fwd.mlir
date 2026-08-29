module @m {
  func.func @cifar8w_bn_fwd(%x: tensor<128x3072xf32>, %W1: tensor<16x3x3x3xf32>, %b1: tensor<16xf32>, %g1: tensor<16xf32>, %bt1: tensor<16xf32>, %W2: tensor<16x16x3x3xf32>, %b2: tensor<16xf32>, %g2: tensor<16xf32>, %bt2: tensor<16xf32>, %W3: tensor<16x16x3x3xf32>, %b3: tensor<16xf32>, %g3: tensor<16xf32>, %bt3: tensor<16xf32>, %W4: tensor<16x16x3x3xf32>, %b4: tensor<16xf32>, %g4: tensor<16xf32>, %bt4: tensor<16xf32>, %W5: tensor<32x16x3x3xf32>, %b5: tensor<32xf32>, %g5: tensor<32xf32>, %bt5: tensor<32xf32>, %W6: tensor<32x32x3x3xf32>, %b6: tensor<32xf32>, %g6: tensor<32xf32>, %bt6: tensor<32xf32>, %W7: tensor<32x32x3x3xf32>, %b7: tensor<32xf32>, %g7: tensor<32xf32>, %bt7: tensor<32xf32>, %W8: tensor<32x32x3x3xf32>, %b8: tensor<32xf32>, %g8: tensor<32xf32>, %bt8: tensor<32xf32>, %W9: tensor<128x512xf32>, %b9: tensor<512xf32>, %Wa: tensor<512x512xf32>, %ba: tensor<512xf32>, %Wb: tensor<512x10xf32>, %bb: tensor<10xf32>) -> tensor<128x10xf32> {
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
    %v248 = stablehlo.dot_general %v247, %W9, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x512xf32>) -> tensor<128x512xf32>
    %v249 = stablehlo.broadcast_in_dim %b9, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v250 = stablehlo.add %v248, %v249 : tensor<128x512xf32>
    %v251 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v252 = stablehlo.maximum %v250, %v251 : tensor<128x512xf32>
    %v253 = stablehlo.dot_general %v252, %Wa, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %v254 = stablehlo.broadcast_in_dim %ba, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v255 = stablehlo.add %v253, %v254 : tensor<128x512xf32>
    %v256 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v257 = stablehlo.maximum %v255, %v256 : tensor<128x512xf32>
    %v258 = stablehlo.dot_general %v257, %Wb, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x10xf32>) -> tensor<128x10xf32>
    %v259 = stablehlo.broadcast_in_dim %bb, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v260 = stablehlo.add %v258, %v259 : tensor<128x10xf32>
    return %v260 : tensor<128x10xf32>
  }
}
