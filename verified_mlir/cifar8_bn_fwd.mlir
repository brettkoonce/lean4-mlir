module @m {
  func.func @cifar8_bn_fwd(%x: tensor<128x3072xf32>, %W1: tensor<16x3x3x3xf32>, %b1: tensor<16xf32>, %g1: tensor<16xf32>, %bt1: tensor<16xf32>, %W2: tensor<16x16x3x3xf32>, %b2: tensor<16xf32>, %g2: tensor<16xf32>, %bt2: tensor<16xf32>, %W3: tensor<16x16x3x3xf32>, %b3: tensor<16xf32>, %g3: tensor<16xf32>, %bt3: tensor<16xf32>, %W4: tensor<16x16x3x3xf32>, %b4: tensor<16xf32>, %g4: tensor<16xf32>, %bt4: tensor<16xf32>, %W5: tensor<32x16x3x3xf32>, %b5: tensor<32xf32>, %g5: tensor<32xf32>, %bt5: tensor<32xf32>, %W6: tensor<32x32x3x3xf32>, %b6: tensor<32xf32>, %g6: tensor<32xf32>, %bt6: tensor<32xf32>, %W7: tensor<32x32x3x3xf32>, %b7: tensor<32xf32>, %g7: tensor<32xf32>, %bt7: tensor<32xf32>, %W8: tensor<32x32x3x3xf32>, %b8: tensor<32xf32>, %g8: tensor<32xf32>, %bt8: tensor<32xf32>, %W9: tensor<128x64xf32>, %b9: tensor<64xf32>, %Wa: tensor<64x64xf32>, %ba: tensor<64xf32>, %Wb: tensor<64x10xf32>, %bb: tensor<10xf32>) -> tensor<128x10xf32> {
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
    return %v244 : tensor<128x10xf32>
  }
}
