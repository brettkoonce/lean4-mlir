module @m {
  func.func @cifar_bn_train_step(%x: tensor<128x3072xf32>, %W1: tensor<32x3x3x3xf32>, %b1: tensor<32xf32>, %g1: tensor<32xf32>, %bt1: tensor<32xf32>, %W2: tensor<32x32x3x3xf32>, %b2: tensor<32xf32>, %g2: tensor<32xf32>, %bt2: tensor<32xf32>, %W3: tensor<64x32x3x3xf32>, %b3: tensor<64xf32>, %g3: tensor<64xf32>, %bt3: tensor<64xf32>, %W4: tensor<64x64x3x3xf32>, %b4: tensor<64xf32>, %g4: tensor<64xf32>, %bt4: tensor<64xf32>, %W5: tensor<4096x512xf32>, %b5: tensor<512xf32>, %W6: tensor<512x512xf32>, %b6: tensor<512xf32>, %W7: tensor<512x10xf32>, %b7: tensor<10xf32>, %onehot: tensor<128x10xf32>) -> (tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<64x32x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<4096x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>) {
    // ── cifar-bn train step: every line is pretty(verified AST node) ──
    %v0 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v1 = stablehlo.convolution(%v0, %W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x3x32x32xf32>, tensor<32x3x3x3xf32>) -> tensor<128x32x32x32xf32>
    %v2 = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<32xf32>) -> tensor<128x32x32x32xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<128x32x32x32xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v6 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7 = stablehlo.constant dense<1024.0> : tensor<128x32x32x32xf32>
    %v8 = stablehlo.constant dense<1.0e-05> : tensor<128x32x32x32xf32>
    %v9 = stablehlo.reduce(%v5 init: %v6) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v10 = stablehlo.broadcast_in_dim %v9, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x32x32xf32>
    %v11 = stablehlo.divide %v10, %v7 : tensor<128x32x32x32xf32>
    %v12 = stablehlo.subtract %v5, %v11 : tensor<128x32x32x32xf32>
    %v13 = stablehlo.multiply %v12, %v12 : tensor<128x32x32x32xf32>
    %v14 = stablehlo.reduce(%v13 init: %v6) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v15 = stablehlo.broadcast_in_dim %v14, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x32x32xf32>
    %v16 = stablehlo.divide %v15, %v7 : tensor<128x32x32x32xf32>
    %v17 = stablehlo.add %v16, %v8 : tensor<128x32x32x32xf32>
    %v18 = stablehlo.rsqrt %v17 : tensor<128x32x32x32xf32>
    %v19 = stablehlo.multiply %v12, %v18 : tensor<128x32x32x32xf32>
    %v20 = stablehlo.broadcast_in_dim %g1, dims = [1] : (tensor<32xf32>) -> tensor<128x32x32x32xf32>
    %v21 = stablehlo.broadcast_in_dim %bt1, dims = [1] : (tensor<32xf32>) -> tensor<128x32x32x32xf32>
    %v22 = stablehlo.multiply %v19, %v20 : tensor<128x32x32x32xf32>
    %v23 = stablehlo.add %v22, %v21 : tensor<128x32x32x32xf32>
    %v24 = stablehlo.reshape %v23 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %v25 = stablehlo.constant dense<0.0> : tensor<128x32768xf32>
    %v26 = stablehlo.maximum %v24, %v25 : tensor<128x32768xf32>
    %v27 = stablehlo.reshape %v26 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v28 = stablehlo.convolution(%v27, %W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x32x32xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x32x32xf32>
    %v29 = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<32xf32>) -> tensor<128x32x32x32xf32>
    %v30 = stablehlo.add %v28, %v29 : tensor<128x32x32x32xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %v32 = stablehlo.reshape %v31 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v33 = stablehlo.constant dense<0.0> : tensor<f32>
    %v34 = stablehlo.constant dense<1024.0> : tensor<128x32x32x32xf32>
    %v35 = stablehlo.constant dense<1.0e-05> : tensor<128x32x32x32xf32>
    %v36 = stablehlo.reduce(%v32 init: %v33) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v37 = stablehlo.broadcast_in_dim %v36, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x32x32xf32>
    %v38 = stablehlo.divide %v37, %v34 : tensor<128x32x32x32xf32>
    %v39 = stablehlo.subtract %v32, %v38 : tensor<128x32x32x32xf32>
    %v40 = stablehlo.multiply %v39, %v39 : tensor<128x32x32x32xf32>
    %v41 = stablehlo.reduce(%v40 init: %v33) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v42 = stablehlo.broadcast_in_dim %v41, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x32x32xf32>
    %v43 = stablehlo.divide %v42, %v34 : tensor<128x32x32x32xf32>
    %v44 = stablehlo.add %v43, %v35 : tensor<128x32x32x32xf32>
    %v45 = stablehlo.rsqrt %v44 : tensor<128x32x32x32xf32>
    %v46 = stablehlo.multiply %v39, %v45 : tensor<128x32x32x32xf32>
    %v47 = stablehlo.broadcast_in_dim %g2, dims = [1] : (tensor<32xf32>) -> tensor<128x32x32x32xf32>
    %v48 = stablehlo.broadcast_in_dim %bt2, dims = [1] : (tensor<32xf32>) -> tensor<128x32x32x32xf32>
    %v49 = stablehlo.multiply %v46, %v47 : tensor<128x32x32x32xf32>
    %v50 = stablehlo.add %v49, %v48 : tensor<128x32x32x32xf32>
    %v51 = stablehlo.reshape %v50 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %v52 = stablehlo.constant dense<0.0> : tensor<128x32768xf32>
    %v53 = stablehlo.maximum %v51, %v52 : tensor<128x32768xf32>
    %v54 = stablehlo.reshape %v53 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v55 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v56 = "stablehlo.reduce_window"(%v54, %v55) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32x16x16xf32>
    %v57 = stablehlo.reshape %v56 : (tensor<128x32x16x16xf32>) -> tensor<128x8192xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<128x8192xf32>) -> tensor<128x32x16x16xf32>
    %v59 = stablehlo.convolution(%v58, %W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x16x16xf32>, tensor<64x32x3x3xf32>) -> tensor<128x64x16x16xf32>
    %v60 = stablehlo.broadcast_in_dim %b3, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %v61 = stablehlo.add %v59, %v60 : tensor<128x64x16x16xf32>
    %v62 = stablehlo.reshape %v61 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v63 = stablehlo.reshape %v62 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v64 = stablehlo.constant dense<0.0> : tensor<f32>
    %v65 = stablehlo.constant dense<256.0> : tensor<128x64x16x16xf32>
    %v66 = stablehlo.constant dense<1.0e-05> : tensor<128x64x16x16xf32>
    %v67 = stablehlo.reduce(%v63 init: %v64) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %v68 = stablehlo.broadcast_in_dim %v67, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x16x16xf32>
    %v69 = stablehlo.divide %v68, %v65 : tensor<128x64x16x16xf32>
    %v70 = stablehlo.subtract %v63, %v69 : tensor<128x64x16x16xf32>
    %v71 = stablehlo.multiply %v70, %v70 : tensor<128x64x16x16xf32>
    %v72 = stablehlo.reduce(%v71 init: %v64) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %v73 = stablehlo.broadcast_in_dim %v72, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x16x16xf32>
    %v74 = stablehlo.divide %v73, %v65 : tensor<128x64x16x16xf32>
    %v75 = stablehlo.add %v74, %v66 : tensor<128x64x16x16xf32>
    %v76 = stablehlo.rsqrt %v75 : tensor<128x64x16x16xf32>
    %v77 = stablehlo.multiply %v70, %v76 : tensor<128x64x16x16xf32>
    %v78 = stablehlo.broadcast_in_dim %g3, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %v79 = stablehlo.broadcast_in_dim %bt3, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %v80 = stablehlo.multiply %v77, %v78 : tensor<128x64x16x16xf32>
    %v81 = stablehlo.add %v80, %v79 : tensor<128x64x16x16xf32>
    %v82 = stablehlo.reshape %v81 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v83 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v84 = stablehlo.maximum %v82, %v83 : tensor<128x16384xf32>
    %v85 = stablehlo.reshape %v84 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v86 = stablehlo.convolution(%v85, %W4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x16x16xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x16x16xf32>
    %v87 = stablehlo.broadcast_in_dim %b4, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %v88 = stablehlo.add %v86, %v87 : tensor<128x64x16x16xf32>
    %v89 = stablehlo.reshape %v88 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v90 = stablehlo.reshape %v89 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v91 = stablehlo.constant dense<0.0> : tensor<f32>
    %v92 = stablehlo.constant dense<256.0> : tensor<128x64x16x16xf32>
    %v93 = stablehlo.constant dense<1.0e-05> : tensor<128x64x16x16xf32>
    %v94 = stablehlo.reduce(%v90 init: %v91) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %v95 = stablehlo.broadcast_in_dim %v94, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x16x16xf32>
    %v96 = stablehlo.divide %v95, %v92 : tensor<128x64x16x16xf32>
    %v97 = stablehlo.subtract %v90, %v96 : tensor<128x64x16x16xf32>
    %v98 = stablehlo.multiply %v97, %v97 : tensor<128x64x16x16xf32>
    %v99 = stablehlo.reduce(%v98 init: %v91) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %v100 = stablehlo.broadcast_in_dim %v99, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x16x16xf32>
    %v101 = stablehlo.divide %v100, %v92 : tensor<128x64x16x16xf32>
    %v102 = stablehlo.add %v101, %v93 : tensor<128x64x16x16xf32>
    %v103 = stablehlo.rsqrt %v102 : tensor<128x64x16x16xf32>
    %v104 = stablehlo.multiply %v97, %v103 : tensor<128x64x16x16xf32>
    %v105 = stablehlo.broadcast_in_dim %g4, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %v106 = stablehlo.broadcast_in_dim %bt4, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %v107 = stablehlo.multiply %v104, %v105 : tensor<128x64x16x16xf32>
    %v108 = stablehlo.add %v107, %v106 : tensor<128x64x16x16xf32>
    %v109 = stablehlo.reshape %v108 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v110 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v111 = stablehlo.maximum %v109, %v110 : tensor<128x16384xf32>
    %v112 = stablehlo.reshape %v111 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v113 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v114 = "stablehlo.reduce_window"(%v112, %v113) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64x8x8xf32>
    %v115 = stablehlo.reshape %v114 : (tensor<128x64x8x8xf32>) -> tensor<128x4096xf32>
    %v116 = stablehlo.dot_general %v115, %W5, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x4096xf32>, tensor<4096x512xf32>) -> tensor<128x512xf32>
    %v117 = stablehlo.broadcast_in_dim %b5, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v118 = stablehlo.add %v116, %v117 : tensor<128x512xf32>
    %v119 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v120 = stablehlo.maximum %v118, %v119 : tensor<128x512xf32>
    %v121 = stablehlo.dot_general %v120, %W6, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %v122 = stablehlo.broadcast_in_dim %b6, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v123 = stablehlo.add %v121, %v122 : tensor<128x512xf32>
    %v124 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v125 = stablehlo.maximum %v123, %v124 : tensor<128x512xf32>
    %v126 = stablehlo.dot_general %v125, %W7, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x10xf32>) -> tensor<128x10xf32>
    %v127 = stablehlo.broadcast_in_dim %b7, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v128 = stablehlo.add %v126, %v127 : tensor<128x10xf32>
    %v129 = stablehlo.exponential %v128 : tensor<128x10xf32>
    %v130 = stablehlo.constant dense<0.0> : tensor<f32>
    %v131 = stablehlo.reduce(%v129 init: %v130) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %v132 = stablehlo.broadcast_in_dim %v131, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %v133 = stablehlo.divide %v129, %v132 : tensor<128x10xf32>
    %v134 = stablehlo.subtract %v133, %onehot : tensor<128x10xf32>
    %v135 = stablehlo.dot_general %v134, %W7, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x10xf32>, tensor<512x10xf32>) -> tensor<128x512xf32>
    %v136 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v137 = stablehlo.compare GT, %v123, %v136 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v138 = stablehlo.select %v137, %v135, %v136 : tensor<128x512xi1>, tensor<128x512xf32>
    %v139 = stablehlo.dot_general %v138, %W6, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %v140 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v141 = stablehlo.compare GT, %v118, %v140 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v142 = stablehlo.select %v141, %v139, %v140 : tensor<128x512xi1>, tensor<128x512xf32>
    %v143 = stablehlo.dot_general %v142, %W5, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<4096x512xf32>) -> tensor<128x4096xf32>
    %v144 = stablehlo.reshape %v111 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v145 = stablehlo.reshape %v143 : (tensor<128x4096xf32>) -> tensor<128x64x8x8xf32>
    %v146 = stablehlo.constant dense<0.0> : tensor<f32>
    %v147 = "stablehlo.select_and_scatter"(%v144, %v145, %v146) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x64x16x16xf32>, tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<128x64x16x16xf32>
    %v148 = stablehlo.reshape %v147 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v149 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v150 = stablehlo.compare GT, %v109, %v149 : (tensor<128x16384xf32>, tensor<128x16384xf32>) -> tensor<128x16384xi1>
    %v151 = stablehlo.select %v150, %v148, %v149 : tensor<128x16384xi1>, tensor<128x16384xf32>
    %v152 = stablehlo.reshape %v151 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v153 = stablehlo.reshape %v89 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v154 = stablehlo.constant dense<0.0> : tensor<f32>
    %v155 = stablehlo.constant dense<256.0> : tensor<128x64x16x16xf32>
    %v156 = stablehlo.constant dense<1.0e-05> : tensor<128x64x16x16xf32>
    %v157 = stablehlo.reduce(%v153 init: %v154) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %v158 = stablehlo.broadcast_in_dim %v157, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x16x16xf32>
    %v159 = stablehlo.divide %v158, %v155 : tensor<128x64x16x16xf32>
    %v160 = stablehlo.subtract %v153, %v159 : tensor<128x64x16x16xf32>
    %v161 = stablehlo.multiply %v160, %v160 : tensor<128x64x16x16xf32>
    %v162 = stablehlo.reduce(%v161 init: %v154) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %v163 = stablehlo.broadcast_in_dim %v162, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x16x16xf32>
    %v164 = stablehlo.divide %v163, %v155 : tensor<128x64x16x16xf32>
    %v165 = stablehlo.add %v164, %v156 : tensor<128x64x16x16xf32>
    %v166 = stablehlo.rsqrt %v165 : tensor<128x64x16x16xf32>
    %v167 = stablehlo.multiply %v160, %v166 : tensor<128x64x16x16xf32>
    %v168 = stablehlo.broadcast_in_dim %g4, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %v169 = stablehlo.multiply %v168, %v152 : tensor<128x64x16x16xf32>
    %v170 = stablehlo.reduce(%v169 init: %v154) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %v171 = stablehlo.broadcast_in_dim %v170, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x16x16xf32>
    %v172 = stablehlo.multiply %v167, %v169 : tensor<128x64x16x16xf32>
    %v173 = stablehlo.reduce(%v172 init: %v154) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %v174 = stablehlo.broadcast_in_dim %v173, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x16x16xf32>
    %v175 = stablehlo.multiply %v169, %v155 : tensor<128x64x16x16xf32>
    %v176 = stablehlo.subtract %v175, %v171 : tensor<128x64x16x16xf32>
    %v177 = stablehlo.multiply %v167, %v174 : tensor<128x64x16x16xf32>
    %v178 = stablehlo.subtract %v176, %v177 : tensor<128x64x16x16xf32>
    %v179 = stablehlo.divide %v166, %v155 : tensor<128x64x16x16xf32>
    %v180 = stablehlo.multiply %v179, %v178 : tensor<128x64x16x16xf32>
    %v181 = stablehlo.reshape %v180 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v182 = stablehlo.reshape %v181 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v183 = stablehlo.transpose %W4, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v184 = stablehlo.reverse %v183, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v185 = stablehlo.convolution(%v182, %v184)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x16x16xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x16x16xf32>
    %v186 = stablehlo.reshape %v185 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v187 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v188 = stablehlo.compare GT, %v82, %v187 : (tensor<128x16384xf32>, tensor<128x16384xf32>) -> tensor<128x16384xi1>
    %v189 = stablehlo.select %v188, %v186, %v187 : tensor<128x16384xi1>, tensor<128x16384xf32>
    %v190 = stablehlo.reshape %v189 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v191 = stablehlo.reshape %v62 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v192 = stablehlo.constant dense<0.0> : tensor<f32>
    %v193 = stablehlo.constant dense<256.0> : tensor<128x64x16x16xf32>
    %v194 = stablehlo.constant dense<1.0e-05> : tensor<128x64x16x16xf32>
    %v195 = stablehlo.reduce(%v191 init: %v192) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %v196 = stablehlo.broadcast_in_dim %v195, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x16x16xf32>
    %v197 = stablehlo.divide %v196, %v193 : tensor<128x64x16x16xf32>
    %v198 = stablehlo.subtract %v191, %v197 : tensor<128x64x16x16xf32>
    %v199 = stablehlo.multiply %v198, %v198 : tensor<128x64x16x16xf32>
    %v200 = stablehlo.reduce(%v199 init: %v192) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %v201 = stablehlo.broadcast_in_dim %v200, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x16x16xf32>
    %v202 = stablehlo.divide %v201, %v193 : tensor<128x64x16x16xf32>
    %v203 = stablehlo.add %v202, %v194 : tensor<128x64x16x16xf32>
    %v204 = stablehlo.rsqrt %v203 : tensor<128x64x16x16xf32>
    %v205 = stablehlo.multiply %v198, %v204 : tensor<128x64x16x16xf32>
    %v206 = stablehlo.broadcast_in_dim %g3, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %v207 = stablehlo.multiply %v206, %v190 : tensor<128x64x16x16xf32>
    %v208 = stablehlo.reduce(%v207 init: %v192) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %v209 = stablehlo.broadcast_in_dim %v208, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x16x16xf32>
    %v210 = stablehlo.multiply %v205, %v207 : tensor<128x64x16x16xf32>
    %v211 = stablehlo.reduce(%v210 init: %v192) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %v212 = stablehlo.broadcast_in_dim %v211, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x16x16xf32>
    %v213 = stablehlo.multiply %v207, %v193 : tensor<128x64x16x16xf32>
    %v214 = stablehlo.subtract %v213, %v209 : tensor<128x64x16x16xf32>
    %v215 = stablehlo.multiply %v205, %v212 : tensor<128x64x16x16xf32>
    %v216 = stablehlo.subtract %v214, %v215 : tensor<128x64x16x16xf32>
    %v217 = stablehlo.divide %v204, %v193 : tensor<128x64x16x16xf32>
    %v218 = stablehlo.multiply %v217, %v216 : tensor<128x64x16x16xf32>
    %v219 = stablehlo.reshape %v218 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v220 = stablehlo.reshape %v219 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v221 = stablehlo.transpose %W3, dims = [1, 0, 2, 3] : (tensor<64x32x3x3xf32>) -> tensor<32x64x3x3xf32>
    %v222 = stablehlo.reverse %v221, dims = [2, 3] : tensor<32x64x3x3xf32>
    %v223 = stablehlo.convolution(%v220, %v222)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x16x16xf32>, tensor<32x64x3x3xf32>) -> tensor<128x32x16x16xf32>
    %v224 = stablehlo.reshape %v223 : (tensor<128x32x16x16xf32>) -> tensor<128x8192xf32>
    %v225 = stablehlo.reshape %v53 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v226 = stablehlo.reshape %v224 : (tensor<128x8192xf32>) -> tensor<128x32x16x16xf32>
    %v227 = stablehlo.constant dense<0.0> : tensor<f32>
    %v228 = "stablehlo.select_and_scatter"(%v225, %v226, %v227) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x32x32xf32>, tensor<128x32x16x16xf32>, tensor<f32>) -> tensor<128x32x32x32xf32>
    %v229 = stablehlo.reshape %v228 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %v230 = stablehlo.constant dense<0.0> : tensor<128x32768xf32>
    %v231 = stablehlo.compare GT, %v51, %v230 : (tensor<128x32768xf32>, tensor<128x32768xf32>) -> tensor<128x32768xi1>
    %v232 = stablehlo.select %v231, %v229, %v230 : tensor<128x32768xi1>, tensor<128x32768xf32>
    %v233 = stablehlo.reshape %v232 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v234 = stablehlo.reshape %v31 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v235 = stablehlo.constant dense<0.0> : tensor<f32>
    %v236 = stablehlo.constant dense<1024.0> : tensor<128x32x32x32xf32>
    %v237 = stablehlo.constant dense<1.0e-05> : tensor<128x32x32x32xf32>
    %v238 = stablehlo.reduce(%v234 init: %v235) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v239 = stablehlo.broadcast_in_dim %v238, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x32x32xf32>
    %v240 = stablehlo.divide %v239, %v236 : tensor<128x32x32x32xf32>
    %v241 = stablehlo.subtract %v234, %v240 : tensor<128x32x32x32xf32>
    %v242 = stablehlo.multiply %v241, %v241 : tensor<128x32x32x32xf32>
    %v243 = stablehlo.reduce(%v242 init: %v235) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v244 = stablehlo.broadcast_in_dim %v243, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x32x32xf32>
    %v245 = stablehlo.divide %v244, %v236 : tensor<128x32x32x32xf32>
    %v246 = stablehlo.add %v245, %v237 : tensor<128x32x32x32xf32>
    %v247 = stablehlo.rsqrt %v246 : tensor<128x32x32x32xf32>
    %v248 = stablehlo.multiply %v241, %v247 : tensor<128x32x32x32xf32>
    %v249 = stablehlo.broadcast_in_dim %g2, dims = [1] : (tensor<32xf32>) -> tensor<128x32x32x32xf32>
    %v250 = stablehlo.multiply %v249, %v233 : tensor<128x32x32x32xf32>
    %v251 = stablehlo.reduce(%v250 init: %v235) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v252 = stablehlo.broadcast_in_dim %v251, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x32x32xf32>
    %v253 = stablehlo.multiply %v248, %v250 : tensor<128x32x32x32xf32>
    %v254 = stablehlo.reduce(%v253 init: %v235) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v255 = stablehlo.broadcast_in_dim %v254, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x32x32xf32>
    %v256 = stablehlo.multiply %v250, %v236 : tensor<128x32x32x32xf32>
    %v257 = stablehlo.subtract %v256, %v252 : tensor<128x32x32x32xf32>
    %v258 = stablehlo.multiply %v248, %v255 : tensor<128x32x32x32xf32>
    %v259 = stablehlo.subtract %v257, %v258 : tensor<128x32x32x32xf32>
    %v260 = stablehlo.divide %v247, %v236 : tensor<128x32x32x32xf32>
    %v261 = stablehlo.multiply %v260, %v259 : tensor<128x32x32x32xf32>
    %v262 = stablehlo.reshape %v261 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %v263 = stablehlo.reshape %v262 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v264 = stablehlo.transpose %W2, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v265 = stablehlo.reverse %v264, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v266 = stablehlo.convolution(%v263, %v265)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x32x32xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x32x32xf32>
    %v267 = stablehlo.reshape %v266 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %v268 = stablehlo.constant dense<0.0> : tensor<128x32768xf32>
    %v269 = stablehlo.compare GT, %v24, %v268 : (tensor<128x32768xf32>, tensor<128x32768xf32>) -> tensor<128x32768xi1>
    %v270 = stablehlo.select %v269, %v267, %v268 : tensor<128x32768xi1>, tensor<128x32768xf32>
    %v271 = stablehlo.reshape %v270 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v272 = stablehlo.reshape %v4 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v273 = stablehlo.constant dense<0.0> : tensor<f32>
    %v274 = stablehlo.constant dense<1024.0> : tensor<128x32x32x32xf32>
    %v275 = stablehlo.constant dense<1.0e-05> : tensor<128x32x32x32xf32>
    %v276 = stablehlo.reduce(%v272 init: %v273) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v277 = stablehlo.broadcast_in_dim %v276, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x32x32xf32>
    %v278 = stablehlo.divide %v277, %v274 : tensor<128x32x32x32xf32>
    %v279 = stablehlo.subtract %v272, %v278 : tensor<128x32x32x32xf32>
    %v280 = stablehlo.multiply %v279, %v279 : tensor<128x32x32x32xf32>
    %v281 = stablehlo.reduce(%v280 init: %v273) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v282 = stablehlo.broadcast_in_dim %v281, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x32x32xf32>
    %v283 = stablehlo.divide %v282, %v274 : tensor<128x32x32x32xf32>
    %v284 = stablehlo.add %v283, %v275 : tensor<128x32x32x32xf32>
    %v285 = stablehlo.rsqrt %v284 : tensor<128x32x32x32xf32>
    %v286 = stablehlo.multiply %v279, %v285 : tensor<128x32x32x32xf32>
    %v287 = stablehlo.broadcast_in_dim %g1, dims = [1] : (tensor<32xf32>) -> tensor<128x32x32x32xf32>
    %v288 = stablehlo.multiply %v287, %v271 : tensor<128x32x32x32xf32>
    %v289 = stablehlo.reduce(%v288 init: %v273) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v290 = stablehlo.broadcast_in_dim %v289, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x32x32xf32>
    %v291 = stablehlo.multiply %v286, %v288 : tensor<128x32x32x32xf32>
    %v292 = stablehlo.reduce(%v291 init: %v273) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v293 = stablehlo.broadcast_in_dim %v292, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x32x32xf32>
    %v294 = stablehlo.multiply %v288, %v274 : tensor<128x32x32x32xf32>
    %v295 = stablehlo.subtract %v294, %v290 : tensor<128x32x32x32xf32>
    %v296 = stablehlo.multiply %v286, %v293 : tensor<128x32x32x32xf32>
    %v297 = stablehlo.subtract %v295, %v296 : tensor<128x32x32x32xf32>
    %v298 = stablehlo.divide %v285, %v274 : tensor<128x32x32x32xf32>
    %v299 = stablehlo.multiply %v298, %v297 : tensor<128x32x32x32xf32>
    %v300 = stablehlo.reshape %v299 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %v301 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v302 = stablehlo.reshape %v300 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v303 = stablehlo.transpose %v301, dims = [1, 0, 2, 3] : (tensor<128x3x32x32xf32>) -> tensor<3x128x32x32xf32>
    %v304 = stablehlo.transpose %v302, dims = [1, 0, 2, 3] : (tensor<128x32x32x32xf32>) -> tensor<32x128x32x32xf32>
    %v305 = stablehlo.convolution(%v303, %v304)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x128x32x32xf32>, tensor<32x128x32x32xf32>) -> tensor<3x32x3x3xf32>
    %v306 = stablehlo.transpose %v305, dims = [1, 0, 2, 3] : (tensor<3x32x3x3xf32>) -> tensor<32x3x3x3xf32>
    %v307 = stablehlo.constant dense<0.00078125> : tensor<32x3x3x3xf32>
    %v308 = stablehlo.multiply %v306, %v307 : tensor<32x3x3x3xf32>
    %v309 = stablehlo.subtract %W1, %v308 : tensor<32x3x3x3xf32>
    %v310 = stablehlo.reshape %v300 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v311 = stablehlo.constant dense<0.0> : tensor<f32>
    %v312 = stablehlo.reduce(%v310 init: %v311) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<32xf32>
    %v313 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v314 = stablehlo.multiply %v312, %v313 : tensor<32xf32>
    %v315 = stablehlo.subtract %b1, %v314 : tensor<32xf32>
    %v316 = stablehlo.constant dense<0.0> : tensor<f32>
    %v317 = stablehlo.reshape %v4 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v318 = stablehlo.constant dense<1024.0> : tensor<128x32x32x32xf32>
    %v319 = stablehlo.constant dense<1.0e-05> : tensor<128x32x32x32xf32>
    %v320 = stablehlo.reduce(%v317 init: %v316) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v321 = stablehlo.broadcast_in_dim %v320, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x32x32xf32>
    %v322 = stablehlo.divide %v321, %v318 : tensor<128x32x32x32xf32>
    %v323 = stablehlo.subtract %v317, %v322 : tensor<128x32x32x32xf32>
    %v324 = stablehlo.multiply %v323, %v323 : tensor<128x32x32x32xf32>
    %v325 = stablehlo.reduce(%v324 init: %v316) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v326 = stablehlo.broadcast_in_dim %v325, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x32x32xf32>
    %v327 = stablehlo.divide %v326, %v318 : tensor<128x32x32x32xf32>
    %v328 = stablehlo.add %v327, %v319 : tensor<128x32x32x32xf32>
    %v329 = stablehlo.rsqrt %v328 : tensor<128x32x32x32xf32>
    %v330 = stablehlo.multiply %v323, %v329 : tensor<128x32x32x32xf32>
    %v331 = stablehlo.reshape %v270 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v332 = stablehlo.multiply %v331, %v330 : tensor<128x32x32x32xf32>
    %v333 = stablehlo.reduce(%v332 init: %v316) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<32xf32>
    %v334 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v335 = stablehlo.multiply %v333, %v334 : tensor<32xf32>
    %v336 = stablehlo.subtract %g1, %v335 : tensor<32xf32>
    %v337 = stablehlo.constant dense<0.0> : tensor<f32>
    %v338 = stablehlo.reshape %v270 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v339 = stablehlo.reduce(%v338 init: %v337) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<32xf32>
    %v340 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v341 = stablehlo.multiply %v339, %v340 : tensor<32xf32>
    %v342 = stablehlo.subtract %bt1, %v341 : tensor<32xf32>
    %v343 = stablehlo.reshape %v26 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v344 = stablehlo.reshape %v262 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v345 = stablehlo.transpose %v343, dims = [1, 0, 2, 3] : (tensor<128x32x32x32xf32>) -> tensor<32x128x32x32xf32>
    %v346 = stablehlo.transpose %v344, dims = [1, 0, 2, 3] : (tensor<128x32x32x32xf32>) -> tensor<32x128x32x32xf32>
    %v347 = stablehlo.convolution(%v345, %v346)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x32x32xf32>, tensor<32x128x32x32xf32>) -> tensor<32x32x3x3xf32>
    %v348 = stablehlo.transpose %v347, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v349 = stablehlo.constant dense<0.00078125> : tensor<32x32x3x3xf32>
    %v350 = stablehlo.multiply %v348, %v349 : tensor<32x32x3x3xf32>
    %v351 = stablehlo.subtract %W2, %v350 : tensor<32x32x3x3xf32>
    %v352 = stablehlo.reshape %v262 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v353 = stablehlo.constant dense<0.0> : tensor<f32>
    %v354 = stablehlo.reduce(%v352 init: %v353) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<32xf32>
    %v355 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v356 = stablehlo.multiply %v354, %v355 : tensor<32xf32>
    %v357 = stablehlo.subtract %b2, %v356 : tensor<32xf32>
    %v358 = stablehlo.constant dense<0.0> : tensor<f32>
    %v359 = stablehlo.reshape %v31 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v360 = stablehlo.constant dense<1024.0> : tensor<128x32x32x32xf32>
    %v361 = stablehlo.constant dense<1.0e-05> : tensor<128x32x32x32xf32>
    %v362 = stablehlo.reduce(%v359 init: %v358) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v363 = stablehlo.broadcast_in_dim %v362, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x32x32xf32>
    %v364 = stablehlo.divide %v363, %v360 : tensor<128x32x32x32xf32>
    %v365 = stablehlo.subtract %v359, %v364 : tensor<128x32x32x32xf32>
    %v366 = stablehlo.multiply %v365, %v365 : tensor<128x32x32x32xf32>
    %v367 = stablehlo.reduce(%v366 init: %v358) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v368 = stablehlo.broadcast_in_dim %v367, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x32x32xf32>
    %v369 = stablehlo.divide %v368, %v360 : tensor<128x32x32x32xf32>
    %v370 = stablehlo.add %v369, %v361 : tensor<128x32x32x32xf32>
    %v371 = stablehlo.rsqrt %v370 : tensor<128x32x32x32xf32>
    %v372 = stablehlo.multiply %v365, %v371 : tensor<128x32x32x32xf32>
    %v373 = stablehlo.reshape %v232 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v374 = stablehlo.multiply %v373, %v372 : tensor<128x32x32x32xf32>
    %v375 = stablehlo.reduce(%v374 init: %v358) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<32xf32>
    %v376 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v377 = stablehlo.multiply %v375, %v376 : tensor<32xf32>
    %v378 = stablehlo.subtract %g2, %v377 : tensor<32xf32>
    %v379 = stablehlo.constant dense<0.0> : tensor<f32>
    %v380 = stablehlo.reshape %v232 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v381 = stablehlo.reduce(%v380 init: %v379) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<32xf32>
    %v382 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v383 = stablehlo.multiply %v381, %v382 : tensor<32xf32>
    %v384 = stablehlo.subtract %bt2, %v383 : tensor<32xf32>
    %v385 = stablehlo.reshape %v57 : (tensor<128x8192xf32>) -> tensor<128x32x16x16xf32>
    %v386 = stablehlo.reshape %v219 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v387 = stablehlo.transpose %v385, dims = [1, 0, 2, 3] : (tensor<128x32x16x16xf32>) -> tensor<32x128x16x16xf32>
    %v388 = stablehlo.transpose %v386, dims = [1, 0, 2, 3] : (tensor<128x64x16x16xf32>) -> tensor<64x128x16x16xf32>
    %v389 = stablehlo.convolution(%v387, %v388)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x16x16xf32>, tensor<64x128x16x16xf32>) -> tensor<32x64x3x3xf32>
    %v390 = stablehlo.transpose %v389, dims = [1, 0, 2, 3] : (tensor<32x64x3x3xf32>) -> tensor<64x32x3x3xf32>
    %v391 = stablehlo.constant dense<0.00078125> : tensor<64x32x3x3xf32>
    %v392 = stablehlo.multiply %v390, %v391 : tensor<64x32x3x3xf32>
    %v393 = stablehlo.subtract %W3, %v392 : tensor<64x32x3x3xf32>
    %v394 = stablehlo.reshape %v219 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v395 = stablehlo.constant dense<0.0> : tensor<f32>
    %v396 = stablehlo.reduce(%v394 init: %v395) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<64xf32>
    %v397 = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %v398 = stablehlo.multiply %v396, %v397 : tensor<64xf32>
    %v399 = stablehlo.subtract %b3, %v398 : tensor<64xf32>
    %v400 = stablehlo.constant dense<0.0> : tensor<f32>
    %v401 = stablehlo.reshape %v62 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v402 = stablehlo.constant dense<256.0> : tensor<128x64x16x16xf32>
    %v403 = stablehlo.constant dense<1.0e-05> : tensor<128x64x16x16xf32>
    %v404 = stablehlo.reduce(%v401 init: %v400) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %v405 = stablehlo.broadcast_in_dim %v404, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x16x16xf32>
    %v406 = stablehlo.divide %v405, %v402 : tensor<128x64x16x16xf32>
    %v407 = stablehlo.subtract %v401, %v406 : tensor<128x64x16x16xf32>
    %v408 = stablehlo.multiply %v407, %v407 : tensor<128x64x16x16xf32>
    %v409 = stablehlo.reduce(%v408 init: %v400) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %v410 = stablehlo.broadcast_in_dim %v409, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x16x16xf32>
    %v411 = stablehlo.divide %v410, %v402 : tensor<128x64x16x16xf32>
    %v412 = stablehlo.add %v411, %v403 : tensor<128x64x16x16xf32>
    %v413 = stablehlo.rsqrt %v412 : tensor<128x64x16x16xf32>
    %v414 = stablehlo.multiply %v407, %v413 : tensor<128x64x16x16xf32>
    %v415 = stablehlo.reshape %v189 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v416 = stablehlo.multiply %v415, %v414 : tensor<128x64x16x16xf32>
    %v417 = stablehlo.reduce(%v416 init: %v400) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<64xf32>
    %v418 = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %v419 = stablehlo.multiply %v417, %v418 : tensor<64xf32>
    %v420 = stablehlo.subtract %g3, %v419 : tensor<64xf32>
    %v421 = stablehlo.constant dense<0.0> : tensor<f32>
    %v422 = stablehlo.reshape %v189 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v423 = stablehlo.reduce(%v422 init: %v421) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<64xf32>
    %v424 = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %v425 = stablehlo.multiply %v423, %v424 : tensor<64xf32>
    %v426 = stablehlo.subtract %bt3, %v425 : tensor<64xf32>
    %v427 = stablehlo.reshape %v84 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v428 = stablehlo.reshape %v181 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v429 = stablehlo.transpose %v427, dims = [1, 0, 2, 3] : (tensor<128x64x16x16xf32>) -> tensor<64x128x16x16xf32>
    %v430 = stablehlo.transpose %v428, dims = [1, 0, 2, 3] : (tensor<128x64x16x16xf32>) -> tensor<64x128x16x16xf32>
    %v431 = stablehlo.convolution(%v429, %v430)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x16x16xf32>, tensor<64x128x16x16xf32>) -> tensor<64x64x3x3xf32>
    %v432 = stablehlo.transpose %v431, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v433 = stablehlo.constant dense<0.00078125> : tensor<64x64x3x3xf32>
    %v434 = stablehlo.multiply %v432, %v433 : tensor<64x64x3x3xf32>
    %v435 = stablehlo.subtract %W4, %v434 : tensor<64x64x3x3xf32>
    %v436 = stablehlo.reshape %v181 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v437 = stablehlo.constant dense<0.0> : tensor<f32>
    %v438 = stablehlo.reduce(%v436 init: %v437) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<64xf32>
    %v439 = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %v440 = stablehlo.multiply %v438, %v439 : tensor<64xf32>
    %v441 = stablehlo.subtract %b4, %v440 : tensor<64xf32>
    %v442 = stablehlo.constant dense<0.0> : tensor<f32>
    %v443 = stablehlo.reshape %v89 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v444 = stablehlo.constant dense<256.0> : tensor<128x64x16x16xf32>
    %v445 = stablehlo.constant dense<1.0e-05> : tensor<128x64x16x16xf32>
    %v446 = stablehlo.reduce(%v443 init: %v442) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %v447 = stablehlo.broadcast_in_dim %v446, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x16x16xf32>
    %v448 = stablehlo.divide %v447, %v444 : tensor<128x64x16x16xf32>
    %v449 = stablehlo.subtract %v443, %v448 : tensor<128x64x16x16xf32>
    %v450 = stablehlo.multiply %v449, %v449 : tensor<128x64x16x16xf32>
    %v451 = stablehlo.reduce(%v450 init: %v442) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %v452 = stablehlo.broadcast_in_dim %v451, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x16x16xf32>
    %v453 = stablehlo.divide %v452, %v444 : tensor<128x64x16x16xf32>
    %v454 = stablehlo.add %v453, %v445 : tensor<128x64x16x16xf32>
    %v455 = stablehlo.rsqrt %v454 : tensor<128x64x16x16xf32>
    %v456 = stablehlo.multiply %v449, %v455 : tensor<128x64x16x16xf32>
    %v457 = stablehlo.reshape %v151 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v458 = stablehlo.multiply %v457, %v456 : tensor<128x64x16x16xf32>
    %v459 = stablehlo.reduce(%v458 init: %v442) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<64xf32>
    %v460 = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %v461 = stablehlo.multiply %v459, %v460 : tensor<64xf32>
    %v462 = stablehlo.subtract %g4, %v461 : tensor<64xf32>
    %v463 = stablehlo.constant dense<0.0> : tensor<f32>
    %v464 = stablehlo.reshape %v151 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v465 = stablehlo.reduce(%v464 init: %v463) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<64xf32>
    %v466 = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %v467 = stablehlo.multiply %v465, %v466 : tensor<64xf32>
    %v468 = stablehlo.subtract %bt4, %v467 : tensor<64xf32>
    %v469 = stablehlo.dot_general %v115, %v142, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x4096xf32>, tensor<128x512xf32>) -> tensor<4096x512xf32>
    %v470 = stablehlo.constant dense<0.00078125> : tensor<4096x512xf32>
    %v471 = stablehlo.multiply %v469, %v470 : tensor<4096x512xf32>
    %v472 = stablehlo.subtract %W5, %v471 : tensor<4096x512xf32>
    %v473 = stablehlo.constant dense<0.0> : tensor<f32>
    %v474 = stablehlo.reduce(%v142 init: %v473) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v475 = stablehlo.constant dense<0.00078125> : tensor<512xf32>
    %v476 = stablehlo.multiply %v474, %v475 : tensor<512xf32>
    %v477 = stablehlo.subtract %b5, %v476 : tensor<512xf32>
    %v478 = stablehlo.dot_general %v120, %v138, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<512x512xf32>
    %v479 = stablehlo.constant dense<0.00078125> : tensor<512x512xf32>
    %v480 = stablehlo.multiply %v478, %v479 : tensor<512x512xf32>
    %v481 = stablehlo.subtract %W6, %v480 : tensor<512x512xf32>
    %v482 = stablehlo.constant dense<0.0> : tensor<f32>
    %v483 = stablehlo.reduce(%v138 init: %v482) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v484 = stablehlo.constant dense<0.00078125> : tensor<512xf32>
    %v485 = stablehlo.multiply %v483, %v484 : tensor<512xf32>
    %v486 = stablehlo.subtract %b6, %v485 : tensor<512xf32>
    %v487 = stablehlo.dot_general %v125, %v134, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x10xf32>) -> tensor<512x10xf32>
    %v488 = stablehlo.constant dense<0.00078125> : tensor<512x10xf32>
    %v489 = stablehlo.multiply %v487, %v488 : tensor<512x10xf32>
    %v490 = stablehlo.subtract %W7, %v489 : tensor<512x10xf32>
    %v491 = stablehlo.constant dense<0.0> : tensor<f32>
    %v492 = stablehlo.reduce(%v134 init: %v491) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v493 = stablehlo.constant dense<0.00078125> : tensor<10xf32>
    %v494 = stablehlo.multiply %v492, %v493 : tensor<10xf32>
    %v495 = stablehlo.subtract %b7, %v494 : tensor<10xf32>
    return %v309, %v315, %v336, %v342, %v351, %v357, %v378, %v384, %v393, %v399, %v420, %v426, %v435, %v441, %v462, %v468, %v472, %v477, %v481, %v486, %v490, %v495 : tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<64x32x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<4096x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>
  }
}
