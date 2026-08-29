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
    %v25 = stablehlo.reshape %v24 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v26 = stablehlo.constant dense<0.0> : tensor<128x32x32x32xf32>
    %v27 = stablehlo.maximum %v25, %v26 : tensor<128x32x32x32xf32>
    %v28 = stablehlo.reshape %v27 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %v29 = stablehlo.reshape %v28 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v30 = stablehlo.convolution(%v29, %W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x32x32xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x32x32xf32>
    %v31 = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<32xf32>) -> tensor<128x32x32x32xf32>
    %v32 = stablehlo.add %v30, %v31 : tensor<128x32x32x32xf32>
    %v33 = stablehlo.reshape %v32 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %v34 = stablehlo.reshape %v33 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v35 = stablehlo.constant dense<0.0> : tensor<f32>
    %v36 = stablehlo.constant dense<1024.0> : tensor<128x32x32x32xf32>
    %v37 = stablehlo.constant dense<1.0e-05> : tensor<128x32x32x32xf32>
    %v38 = stablehlo.reduce(%v34 init: %v35) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v39 = stablehlo.broadcast_in_dim %v38, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x32x32xf32>
    %v40 = stablehlo.divide %v39, %v36 : tensor<128x32x32x32xf32>
    %v41 = stablehlo.subtract %v34, %v40 : tensor<128x32x32x32xf32>
    %v42 = stablehlo.multiply %v41, %v41 : tensor<128x32x32x32xf32>
    %v43 = stablehlo.reduce(%v42 init: %v35) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v44 = stablehlo.broadcast_in_dim %v43, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x32x32xf32>
    %v45 = stablehlo.divide %v44, %v36 : tensor<128x32x32x32xf32>
    %v46 = stablehlo.add %v45, %v37 : tensor<128x32x32x32xf32>
    %v47 = stablehlo.rsqrt %v46 : tensor<128x32x32x32xf32>
    %v48 = stablehlo.multiply %v41, %v47 : tensor<128x32x32x32xf32>
    %v49 = stablehlo.broadcast_in_dim %g2, dims = [1] : (tensor<32xf32>) -> tensor<128x32x32x32xf32>
    %v50 = stablehlo.broadcast_in_dim %bt2, dims = [1] : (tensor<32xf32>) -> tensor<128x32x32x32xf32>
    %v51 = stablehlo.multiply %v48, %v49 : tensor<128x32x32x32xf32>
    %v52 = stablehlo.add %v51, %v50 : tensor<128x32x32x32xf32>
    %v53 = stablehlo.reshape %v52 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %v54 = stablehlo.reshape %v53 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v55 = stablehlo.constant dense<0.0> : tensor<128x32x32x32xf32>
    %v56 = stablehlo.maximum %v54, %v55 : tensor<128x32x32x32xf32>
    %v57 = stablehlo.reshape %v56 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v59 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v60 = "stablehlo.reduce_window"(%v58, %v59) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32x16x16xf32>
    %v61 = stablehlo.reshape %v60 : (tensor<128x32x16x16xf32>) -> tensor<128x8192xf32>
    %v62 = stablehlo.reshape %v61 : (tensor<128x8192xf32>) -> tensor<128x32x16x16xf32>
    %v63 = stablehlo.convolution(%v62, %W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x16x16xf32>, tensor<64x32x3x3xf32>) -> tensor<128x64x16x16xf32>
    %v64 = stablehlo.broadcast_in_dim %b3, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %v65 = stablehlo.add %v63, %v64 : tensor<128x64x16x16xf32>
    %v66 = stablehlo.reshape %v65 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v67 = stablehlo.reshape %v66 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v68 = stablehlo.constant dense<0.0> : tensor<f32>
    %v69 = stablehlo.constant dense<256.0> : tensor<128x64x16x16xf32>
    %v70 = stablehlo.constant dense<1.0e-05> : tensor<128x64x16x16xf32>
    %v71 = stablehlo.reduce(%v67 init: %v68) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %v72 = stablehlo.broadcast_in_dim %v71, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x16x16xf32>
    %v73 = stablehlo.divide %v72, %v69 : tensor<128x64x16x16xf32>
    %v74 = stablehlo.subtract %v67, %v73 : tensor<128x64x16x16xf32>
    %v75 = stablehlo.multiply %v74, %v74 : tensor<128x64x16x16xf32>
    %v76 = stablehlo.reduce(%v75 init: %v68) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %v77 = stablehlo.broadcast_in_dim %v76, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x16x16xf32>
    %v78 = stablehlo.divide %v77, %v69 : tensor<128x64x16x16xf32>
    %v79 = stablehlo.add %v78, %v70 : tensor<128x64x16x16xf32>
    %v80 = stablehlo.rsqrt %v79 : tensor<128x64x16x16xf32>
    %v81 = stablehlo.multiply %v74, %v80 : tensor<128x64x16x16xf32>
    %v82 = stablehlo.broadcast_in_dim %g3, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %v83 = stablehlo.broadcast_in_dim %bt3, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %v84 = stablehlo.multiply %v81, %v82 : tensor<128x64x16x16xf32>
    %v85 = stablehlo.add %v84, %v83 : tensor<128x64x16x16xf32>
    %v86 = stablehlo.reshape %v85 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v87 = stablehlo.reshape %v86 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v88 = stablehlo.constant dense<0.0> : tensor<128x64x16x16xf32>
    %v89 = stablehlo.maximum %v87, %v88 : tensor<128x64x16x16xf32>
    %v90 = stablehlo.reshape %v89 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v91 = stablehlo.reshape %v90 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v92 = stablehlo.convolution(%v91, %W4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x16x16xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x16x16xf32>
    %v93 = stablehlo.broadcast_in_dim %b4, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %v94 = stablehlo.add %v92, %v93 : tensor<128x64x16x16xf32>
    %v95 = stablehlo.reshape %v94 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v96 = stablehlo.reshape %v95 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v97 = stablehlo.constant dense<0.0> : tensor<f32>
    %v98 = stablehlo.constant dense<256.0> : tensor<128x64x16x16xf32>
    %v99 = stablehlo.constant dense<1.0e-05> : tensor<128x64x16x16xf32>
    %v100 = stablehlo.reduce(%v96 init: %v97) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %v101 = stablehlo.broadcast_in_dim %v100, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x16x16xf32>
    %v102 = stablehlo.divide %v101, %v98 : tensor<128x64x16x16xf32>
    %v103 = stablehlo.subtract %v96, %v102 : tensor<128x64x16x16xf32>
    %v104 = stablehlo.multiply %v103, %v103 : tensor<128x64x16x16xf32>
    %v105 = stablehlo.reduce(%v104 init: %v97) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %v106 = stablehlo.broadcast_in_dim %v105, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x16x16xf32>
    %v107 = stablehlo.divide %v106, %v98 : tensor<128x64x16x16xf32>
    %v108 = stablehlo.add %v107, %v99 : tensor<128x64x16x16xf32>
    %v109 = stablehlo.rsqrt %v108 : tensor<128x64x16x16xf32>
    %v110 = stablehlo.multiply %v103, %v109 : tensor<128x64x16x16xf32>
    %v111 = stablehlo.broadcast_in_dim %g4, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %v112 = stablehlo.broadcast_in_dim %bt4, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %v113 = stablehlo.multiply %v110, %v111 : tensor<128x64x16x16xf32>
    %v114 = stablehlo.add %v113, %v112 : tensor<128x64x16x16xf32>
    %v115 = stablehlo.reshape %v114 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v116 = stablehlo.reshape %v115 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v117 = stablehlo.constant dense<0.0> : tensor<128x64x16x16xf32>
    %v118 = stablehlo.maximum %v116, %v117 : tensor<128x64x16x16xf32>
    %v119 = stablehlo.reshape %v118 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v120 = stablehlo.reshape %v119 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v121 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v122 = "stablehlo.reduce_window"(%v120, %v121) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64x8x8xf32>
    %v123 = stablehlo.reshape %v122 : (tensor<128x64x8x8xf32>) -> tensor<128x4096xf32>
    %v124 = stablehlo.dot_general %v123, %W5, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x4096xf32>, tensor<4096x512xf32>) -> tensor<128x512xf32>
    %v125 = stablehlo.broadcast_in_dim %b5, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v126 = stablehlo.add %v124, %v125 : tensor<128x512xf32>
    %v127 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v128 = stablehlo.maximum %v126, %v127 : tensor<128x512xf32>
    %v129 = stablehlo.dot_general %v128, %W6, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %v130 = stablehlo.broadcast_in_dim %b6, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v131 = stablehlo.add %v129, %v130 : tensor<128x512xf32>
    %v132 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v133 = stablehlo.maximum %v131, %v132 : tensor<128x512xf32>
    %v134 = stablehlo.dot_general %v133, %W7, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x10xf32>) -> tensor<128x10xf32>
    %v135 = stablehlo.broadcast_in_dim %b7, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v136 = stablehlo.add %v134, %v135 : tensor<128x10xf32>
    %v137 = stablehlo.exponential %v136 : tensor<128x10xf32>
    %v138 = stablehlo.constant dense<0.0> : tensor<f32>
    %v139 = stablehlo.reduce(%v137 init: %v138) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %v140 = stablehlo.broadcast_in_dim %v139, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %v141 = stablehlo.divide %v137, %v140 : tensor<128x10xf32>
    %v142 = stablehlo.subtract %v141, %onehot : tensor<128x10xf32>
    %v143 = stablehlo.dot_general %v142, %W7, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x10xf32>, tensor<512x10xf32>) -> tensor<128x512xf32>
    %v144 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v145 = stablehlo.compare GT, %v131, %v144 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v146 = stablehlo.select %v145, %v143, %v144 : tensor<128x512xi1>, tensor<128x512xf32>
    %v147 = stablehlo.dot_general %v146, %W6, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %v148 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v149 = stablehlo.compare GT, %v126, %v148 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v150 = stablehlo.select %v149, %v147, %v148 : tensor<128x512xi1>, tensor<128x512xf32>
    %v151 = stablehlo.dot_general %v150, %W5, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<4096x512xf32>) -> tensor<128x4096xf32>
    %v152 = stablehlo.reshape %v119 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v153 = stablehlo.reshape %v151 : (tensor<128x4096xf32>) -> tensor<128x64x8x8xf32>
    %v154 = stablehlo.constant dense<0.0> : tensor<f32>
    %v155 = "stablehlo.select_and_scatter"(%v152, %v153, %v154) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x64x16x16xf32>, tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<128x64x16x16xf32>
    %v156 = stablehlo.reshape %v155 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v157 = stablehlo.reshape %v156 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v158 = stablehlo.reshape %v115 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v159 = stablehlo.constant dense<0.0> : tensor<128x64x16x16xf32>
    %v160 = stablehlo.compare GT, %v158, %v159 : (tensor<128x64x16x16xf32>, tensor<128x64x16x16xf32>) -> tensor<128x64x16x16xi1>
    %v161 = stablehlo.select %v160, %v157, %v159 : tensor<128x64x16x16xi1>, tensor<128x64x16x16xf32>
    %v162 = stablehlo.reshape %v161 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v163 = stablehlo.reshape %v162 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v164 = stablehlo.reshape %v95 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v165 = stablehlo.constant dense<0.0> : tensor<f32>
    %v166 = stablehlo.constant dense<256.0> : tensor<128x64x16x16xf32>
    %v167 = stablehlo.constant dense<1.0e-05> : tensor<128x64x16x16xf32>
    %v168 = stablehlo.reduce(%v164 init: %v165) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %v169 = stablehlo.broadcast_in_dim %v168, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x16x16xf32>
    %v170 = stablehlo.divide %v169, %v166 : tensor<128x64x16x16xf32>
    %v171 = stablehlo.subtract %v164, %v170 : tensor<128x64x16x16xf32>
    %v172 = stablehlo.multiply %v171, %v171 : tensor<128x64x16x16xf32>
    %v173 = stablehlo.reduce(%v172 init: %v165) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %v174 = stablehlo.broadcast_in_dim %v173, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x16x16xf32>
    %v175 = stablehlo.divide %v174, %v166 : tensor<128x64x16x16xf32>
    %v176 = stablehlo.add %v175, %v167 : tensor<128x64x16x16xf32>
    %v177 = stablehlo.rsqrt %v176 : tensor<128x64x16x16xf32>
    %v178 = stablehlo.multiply %v171, %v177 : tensor<128x64x16x16xf32>
    %v179 = stablehlo.broadcast_in_dim %g4, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %v180 = stablehlo.multiply %v179, %v163 : tensor<128x64x16x16xf32>
    %v181 = stablehlo.reduce(%v180 init: %v165) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %v182 = stablehlo.broadcast_in_dim %v181, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x16x16xf32>
    %v183 = stablehlo.multiply %v178, %v180 : tensor<128x64x16x16xf32>
    %v184 = stablehlo.reduce(%v183 init: %v165) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %v185 = stablehlo.broadcast_in_dim %v184, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x16x16xf32>
    %v186 = stablehlo.multiply %v180, %v166 : tensor<128x64x16x16xf32>
    %v187 = stablehlo.subtract %v186, %v182 : tensor<128x64x16x16xf32>
    %v188 = stablehlo.multiply %v178, %v185 : tensor<128x64x16x16xf32>
    %v189 = stablehlo.subtract %v187, %v188 : tensor<128x64x16x16xf32>
    %v190 = stablehlo.divide %v177, %v166 : tensor<128x64x16x16xf32>
    %v191 = stablehlo.multiply %v190, %v189 : tensor<128x64x16x16xf32>
    %v192 = stablehlo.reshape %v191 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v193 = stablehlo.reshape %v192 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v194 = stablehlo.transpose %W4, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v195 = stablehlo.reverse %v194, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v196 = stablehlo.convolution(%v193, %v195)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x16x16xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x16x16xf32>
    %v197 = stablehlo.reshape %v196 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v198 = stablehlo.reshape %v197 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v199 = stablehlo.reshape %v86 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v200 = stablehlo.constant dense<0.0> : tensor<128x64x16x16xf32>
    %v201 = stablehlo.compare GT, %v199, %v200 : (tensor<128x64x16x16xf32>, tensor<128x64x16x16xf32>) -> tensor<128x64x16x16xi1>
    %v202 = stablehlo.select %v201, %v198, %v200 : tensor<128x64x16x16xi1>, tensor<128x64x16x16xf32>
    %v203 = stablehlo.reshape %v202 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v204 = stablehlo.reshape %v203 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v205 = stablehlo.reshape %v66 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v206 = stablehlo.constant dense<0.0> : tensor<f32>
    %v207 = stablehlo.constant dense<256.0> : tensor<128x64x16x16xf32>
    %v208 = stablehlo.constant dense<1.0e-05> : tensor<128x64x16x16xf32>
    %v209 = stablehlo.reduce(%v205 init: %v206) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %v210 = stablehlo.broadcast_in_dim %v209, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x16x16xf32>
    %v211 = stablehlo.divide %v210, %v207 : tensor<128x64x16x16xf32>
    %v212 = stablehlo.subtract %v205, %v211 : tensor<128x64x16x16xf32>
    %v213 = stablehlo.multiply %v212, %v212 : tensor<128x64x16x16xf32>
    %v214 = stablehlo.reduce(%v213 init: %v206) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %v215 = stablehlo.broadcast_in_dim %v214, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x16x16xf32>
    %v216 = stablehlo.divide %v215, %v207 : tensor<128x64x16x16xf32>
    %v217 = stablehlo.add %v216, %v208 : tensor<128x64x16x16xf32>
    %v218 = stablehlo.rsqrt %v217 : tensor<128x64x16x16xf32>
    %v219 = stablehlo.multiply %v212, %v218 : tensor<128x64x16x16xf32>
    %v220 = stablehlo.broadcast_in_dim %g3, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %v221 = stablehlo.multiply %v220, %v204 : tensor<128x64x16x16xf32>
    %v222 = stablehlo.reduce(%v221 init: %v206) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %v223 = stablehlo.broadcast_in_dim %v222, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x16x16xf32>
    %v224 = stablehlo.multiply %v219, %v221 : tensor<128x64x16x16xf32>
    %v225 = stablehlo.reduce(%v224 init: %v206) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %v226 = stablehlo.broadcast_in_dim %v225, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x16x16xf32>
    %v227 = stablehlo.multiply %v221, %v207 : tensor<128x64x16x16xf32>
    %v228 = stablehlo.subtract %v227, %v223 : tensor<128x64x16x16xf32>
    %v229 = stablehlo.multiply %v219, %v226 : tensor<128x64x16x16xf32>
    %v230 = stablehlo.subtract %v228, %v229 : tensor<128x64x16x16xf32>
    %v231 = stablehlo.divide %v218, %v207 : tensor<128x64x16x16xf32>
    %v232 = stablehlo.multiply %v231, %v230 : tensor<128x64x16x16xf32>
    %v233 = stablehlo.reshape %v232 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v234 = stablehlo.reshape %v233 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v235 = stablehlo.transpose %W3, dims = [1, 0, 2, 3] : (tensor<64x32x3x3xf32>) -> tensor<32x64x3x3xf32>
    %v236 = stablehlo.reverse %v235, dims = [2, 3] : tensor<32x64x3x3xf32>
    %v237 = stablehlo.convolution(%v234, %v236)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x16x16xf32>, tensor<32x64x3x3xf32>) -> tensor<128x32x16x16xf32>
    %v238 = stablehlo.reshape %v237 : (tensor<128x32x16x16xf32>) -> tensor<128x8192xf32>
    %v239 = stablehlo.reshape %v57 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v240 = stablehlo.reshape %v238 : (tensor<128x8192xf32>) -> tensor<128x32x16x16xf32>
    %v241 = stablehlo.constant dense<0.0> : tensor<f32>
    %v242 = "stablehlo.select_and_scatter"(%v239, %v240, %v241) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x32x32xf32>, tensor<128x32x16x16xf32>, tensor<f32>) -> tensor<128x32x32x32xf32>
    %v243 = stablehlo.reshape %v242 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %v244 = stablehlo.reshape %v243 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v245 = stablehlo.reshape %v53 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v246 = stablehlo.constant dense<0.0> : tensor<128x32x32x32xf32>
    %v247 = stablehlo.compare GT, %v245, %v246 : (tensor<128x32x32x32xf32>, tensor<128x32x32x32xf32>) -> tensor<128x32x32x32xi1>
    %v248 = stablehlo.select %v247, %v244, %v246 : tensor<128x32x32x32xi1>, tensor<128x32x32x32xf32>
    %v249 = stablehlo.reshape %v248 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %v250 = stablehlo.reshape %v249 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v251 = stablehlo.reshape %v33 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v252 = stablehlo.constant dense<0.0> : tensor<f32>
    %v253 = stablehlo.constant dense<1024.0> : tensor<128x32x32x32xf32>
    %v254 = stablehlo.constant dense<1.0e-05> : tensor<128x32x32x32xf32>
    %v255 = stablehlo.reduce(%v251 init: %v252) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v256 = stablehlo.broadcast_in_dim %v255, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x32x32xf32>
    %v257 = stablehlo.divide %v256, %v253 : tensor<128x32x32x32xf32>
    %v258 = stablehlo.subtract %v251, %v257 : tensor<128x32x32x32xf32>
    %v259 = stablehlo.multiply %v258, %v258 : tensor<128x32x32x32xf32>
    %v260 = stablehlo.reduce(%v259 init: %v252) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v261 = stablehlo.broadcast_in_dim %v260, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x32x32xf32>
    %v262 = stablehlo.divide %v261, %v253 : tensor<128x32x32x32xf32>
    %v263 = stablehlo.add %v262, %v254 : tensor<128x32x32x32xf32>
    %v264 = stablehlo.rsqrt %v263 : tensor<128x32x32x32xf32>
    %v265 = stablehlo.multiply %v258, %v264 : tensor<128x32x32x32xf32>
    %v266 = stablehlo.broadcast_in_dim %g2, dims = [1] : (tensor<32xf32>) -> tensor<128x32x32x32xf32>
    %v267 = stablehlo.multiply %v266, %v250 : tensor<128x32x32x32xf32>
    %v268 = stablehlo.reduce(%v267 init: %v252) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v269 = stablehlo.broadcast_in_dim %v268, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x32x32xf32>
    %v270 = stablehlo.multiply %v265, %v267 : tensor<128x32x32x32xf32>
    %v271 = stablehlo.reduce(%v270 init: %v252) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v272 = stablehlo.broadcast_in_dim %v271, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x32x32xf32>
    %v273 = stablehlo.multiply %v267, %v253 : tensor<128x32x32x32xf32>
    %v274 = stablehlo.subtract %v273, %v269 : tensor<128x32x32x32xf32>
    %v275 = stablehlo.multiply %v265, %v272 : tensor<128x32x32x32xf32>
    %v276 = stablehlo.subtract %v274, %v275 : tensor<128x32x32x32xf32>
    %v277 = stablehlo.divide %v264, %v253 : tensor<128x32x32x32xf32>
    %v278 = stablehlo.multiply %v277, %v276 : tensor<128x32x32x32xf32>
    %v279 = stablehlo.reshape %v278 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %v280 = stablehlo.reshape %v279 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v281 = stablehlo.transpose %W2, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v282 = stablehlo.reverse %v281, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v283 = stablehlo.convolution(%v280, %v282)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x32x32xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x32x32xf32>
    %v284 = stablehlo.reshape %v283 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %v285 = stablehlo.reshape %v284 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v286 = stablehlo.reshape %v24 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v287 = stablehlo.constant dense<0.0> : tensor<128x32x32x32xf32>
    %v288 = stablehlo.compare GT, %v286, %v287 : (tensor<128x32x32x32xf32>, tensor<128x32x32x32xf32>) -> tensor<128x32x32x32xi1>
    %v289 = stablehlo.select %v288, %v285, %v287 : tensor<128x32x32x32xi1>, tensor<128x32x32x32xf32>
    %v290 = stablehlo.reshape %v289 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %v291 = stablehlo.reshape %v290 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v292 = stablehlo.reshape %v4 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v293 = stablehlo.constant dense<0.0> : tensor<f32>
    %v294 = stablehlo.constant dense<1024.0> : tensor<128x32x32x32xf32>
    %v295 = stablehlo.constant dense<1.0e-05> : tensor<128x32x32x32xf32>
    %v296 = stablehlo.reduce(%v292 init: %v293) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v297 = stablehlo.broadcast_in_dim %v296, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x32x32xf32>
    %v298 = stablehlo.divide %v297, %v294 : tensor<128x32x32x32xf32>
    %v299 = stablehlo.subtract %v292, %v298 : tensor<128x32x32x32xf32>
    %v300 = stablehlo.multiply %v299, %v299 : tensor<128x32x32x32xf32>
    %v301 = stablehlo.reduce(%v300 init: %v293) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v302 = stablehlo.broadcast_in_dim %v301, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x32x32xf32>
    %v303 = stablehlo.divide %v302, %v294 : tensor<128x32x32x32xf32>
    %v304 = stablehlo.add %v303, %v295 : tensor<128x32x32x32xf32>
    %v305 = stablehlo.rsqrt %v304 : tensor<128x32x32x32xf32>
    %v306 = stablehlo.multiply %v299, %v305 : tensor<128x32x32x32xf32>
    %v307 = stablehlo.broadcast_in_dim %g1, dims = [1] : (tensor<32xf32>) -> tensor<128x32x32x32xf32>
    %v308 = stablehlo.multiply %v307, %v291 : tensor<128x32x32x32xf32>
    %v309 = stablehlo.reduce(%v308 init: %v293) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v310 = stablehlo.broadcast_in_dim %v309, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x32x32xf32>
    %v311 = stablehlo.multiply %v306, %v308 : tensor<128x32x32x32xf32>
    %v312 = stablehlo.reduce(%v311 init: %v293) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v313 = stablehlo.broadcast_in_dim %v312, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x32x32xf32>
    %v314 = stablehlo.multiply %v308, %v294 : tensor<128x32x32x32xf32>
    %v315 = stablehlo.subtract %v314, %v310 : tensor<128x32x32x32xf32>
    %v316 = stablehlo.multiply %v306, %v313 : tensor<128x32x32x32xf32>
    %v317 = stablehlo.subtract %v315, %v316 : tensor<128x32x32x32xf32>
    %v318 = stablehlo.divide %v305, %v294 : tensor<128x32x32x32xf32>
    %v319 = stablehlo.multiply %v318, %v317 : tensor<128x32x32x32xf32>
    %v320 = stablehlo.reshape %v319 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %v321 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v322 = stablehlo.reshape %v320 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v323 = stablehlo.transpose %v321, dims = [1, 0, 2, 3] : (tensor<128x3x32x32xf32>) -> tensor<3x128x32x32xf32>
    %v324 = stablehlo.transpose %v322, dims = [1, 0, 2, 3] : (tensor<128x32x32x32xf32>) -> tensor<32x128x32x32xf32>
    %v325 = stablehlo.convolution(%v323, %v324)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x128x32x32xf32>, tensor<32x128x32x32xf32>) -> tensor<3x32x3x3xf32>
    %v326 = stablehlo.transpose %v325, dims = [1, 0, 2, 3] : (tensor<3x32x3x3xf32>) -> tensor<32x3x3x3xf32>
    %v327 = stablehlo.constant dense<0.00078125> : tensor<32x3x3x3xf32>
    %v328 = stablehlo.multiply %v326, %v327 : tensor<32x3x3x3xf32>
    %v329 = stablehlo.subtract %W1, %v328 : tensor<32x3x3x3xf32>
    %v330 = stablehlo.reshape %v320 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v331 = stablehlo.constant dense<0.0> : tensor<f32>
    %v332 = stablehlo.reduce(%v330 init: %v331) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<32xf32>
    %v333 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v334 = stablehlo.multiply %v332, %v333 : tensor<32xf32>
    %v335 = stablehlo.subtract %b1, %v334 : tensor<32xf32>
    %v336 = stablehlo.constant dense<0.0> : tensor<f32>
    %v337 = stablehlo.reshape %v4 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v338 = stablehlo.constant dense<1024.0> : tensor<128x32x32x32xf32>
    %v339 = stablehlo.constant dense<1.0e-05> : tensor<128x32x32x32xf32>
    %v340 = stablehlo.reduce(%v337 init: %v336) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v341 = stablehlo.broadcast_in_dim %v340, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x32x32xf32>
    %v342 = stablehlo.divide %v341, %v338 : tensor<128x32x32x32xf32>
    %v343 = stablehlo.subtract %v337, %v342 : tensor<128x32x32x32xf32>
    %v344 = stablehlo.multiply %v343, %v343 : tensor<128x32x32x32xf32>
    %v345 = stablehlo.reduce(%v344 init: %v336) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v346 = stablehlo.broadcast_in_dim %v345, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x32x32xf32>
    %v347 = stablehlo.divide %v346, %v338 : tensor<128x32x32x32xf32>
    %v348 = stablehlo.add %v347, %v339 : tensor<128x32x32x32xf32>
    %v349 = stablehlo.rsqrt %v348 : tensor<128x32x32x32xf32>
    %v350 = stablehlo.multiply %v343, %v349 : tensor<128x32x32x32xf32>
    %v351 = stablehlo.reshape %v290 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v352 = stablehlo.multiply %v351, %v350 : tensor<128x32x32x32xf32>
    %v353 = stablehlo.reduce(%v352 init: %v336) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<32xf32>
    %v354 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v355 = stablehlo.multiply %v353, %v354 : tensor<32xf32>
    %v356 = stablehlo.subtract %g1, %v355 : tensor<32xf32>
    %v357 = stablehlo.constant dense<0.0> : tensor<f32>
    %v358 = stablehlo.reshape %v290 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v359 = stablehlo.reduce(%v358 init: %v357) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<32xf32>
    %v360 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v361 = stablehlo.multiply %v359, %v360 : tensor<32xf32>
    %v362 = stablehlo.subtract %bt1, %v361 : tensor<32xf32>
    %v363 = stablehlo.reshape %v28 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v364 = stablehlo.reshape %v279 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v365 = stablehlo.transpose %v363, dims = [1, 0, 2, 3] : (tensor<128x32x32x32xf32>) -> tensor<32x128x32x32xf32>
    %v366 = stablehlo.transpose %v364, dims = [1, 0, 2, 3] : (tensor<128x32x32x32xf32>) -> tensor<32x128x32x32xf32>
    %v367 = stablehlo.convolution(%v365, %v366)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x32x32xf32>, tensor<32x128x32x32xf32>) -> tensor<32x32x3x3xf32>
    %v368 = stablehlo.transpose %v367, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v369 = stablehlo.constant dense<0.00078125> : tensor<32x32x3x3xf32>
    %v370 = stablehlo.multiply %v368, %v369 : tensor<32x32x3x3xf32>
    %v371 = stablehlo.subtract %W2, %v370 : tensor<32x32x3x3xf32>
    %v372 = stablehlo.reshape %v279 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v373 = stablehlo.constant dense<0.0> : tensor<f32>
    %v374 = stablehlo.reduce(%v372 init: %v373) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<32xf32>
    %v375 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v376 = stablehlo.multiply %v374, %v375 : tensor<32xf32>
    %v377 = stablehlo.subtract %b2, %v376 : tensor<32xf32>
    %v378 = stablehlo.constant dense<0.0> : tensor<f32>
    %v379 = stablehlo.reshape %v33 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v380 = stablehlo.constant dense<1024.0> : tensor<128x32x32x32xf32>
    %v381 = stablehlo.constant dense<1.0e-05> : tensor<128x32x32x32xf32>
    %v382 = stablehlo.reduce(%v379 init: %v378) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v383 = stablehlo.broadcast_in_dim %v382, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x32x32xf32>
    %v384 = stablehlo.divide %v383, %v380 : tensor<128x32x32x32xf32>
    %v385 = stablehlo.subtract %v379, %v384 : tensor<128x32x32x32xf32>
    %v386 = stablehlo.multiply %v385, %v385 : tensor<128x32x32x32xf32>
    %v387 = stablehlo.reduce(%v386 init: %v378) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v388 = stablehlo.broadcast_in_dim %v387, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x32x32xf32>
    %v389 = stablehlo.divide %v388, %v380 : tensor<128x32x32x32xf32>
    %v390 = stablehlo.add %v389, %v381 : tensor<128x32x32x32xf32>
    %v391 = stablehlo.rsqrt %v390 : tensor<128x32x32x32xf32>
    %v392 = stablehlo.multiply %v385, %v391 : tensor<128x32x32x32xf32>
    %v393 = stablehlo.reshape %v249 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v394 = stablehlo.multiply %v393, %v392 : tensor<128x32x32x32xf32>
    %v395 = stablehlo.reduce(%v394 init: %v378) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<32xf32>
    %v396 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v397 = stablehlo.multiply %v395, %v396 : tensor<32xf32>
    %v398 = stablehlo.subtract %g2, %v397 : tensor<32xf32>
    %v399 = stablehlo.constant dense<0.0> : tensor<f32>
    %v400 = stablehlo.reshape %v249 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v401 = stablehlo.reduce(%v400 init: %v399) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<32xf32>
    %v402 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v403 = stablehlo.multiply %v401, %v402 : tensor<32xf32>
    %v404 = stablehlo.subtract %bt2, %v403 : tensor<32xf32>
    %v405 = stablehlo.reshape %v61 : (tensor<128x8192xf32>) -> tensor<128x32x16x16xf32>
    %v406 = stablehlo.reshape %v233 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v407 = stablehlo.transpose %v405, dims = [1, 0, 2, 3] : (tensor<128x32x16x16xf32>) -> tensor<32x128x16x16xf32>
    %v408 = stablehlo.transpose %v406, dims = [1, 0, 2, 3] : (tensor<128x64x16x16xf32>) -> tensor<64x128x16x16xf32>
    %v409 = stablehlo.convolution(%v407, %v408)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x16x16xf32>, tensor<64x128x16x16xf32>) -> tensor<32x64x3x3xf32>
    %v410 = stablehlo.transpose %v409, dims = [1, 0, 2, 3] : (tensor<32x64x3x3xf32>) -> tensor<64x32x3x3xf32>
    %v411 = stablehlo.constant dense<0.00078125> : tensor<64x32x3x3xf32>
    %v412 = stablehlo.multiply %v410, %v411 : tensor<64x32x3x3xf32>
    %v413 = stablehlo.subtract %W3, %v412 : tensor<64x32x3x3xf32>
    %v414 = stablehlo.reshape %v233 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v415 = stablehlo.constant dense<0.0> : tensor<f32>
    %v416 = stablehlo.reduce(%v414 init: %v415) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<64xf32>
    %v417 = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %v418 = stablehlo.multiply %v416, %v417 : tensor<64xf32>
    %v419 = stablehlo.subtract %b3, %v418 : tensor<64xf32>
    %v420 = stablehlo.constant dense<0.0> : tensor<f32>
    %v421 = stablehlo.reshape %v66 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v422 = stablehlo.constant dense<256.0> : tensor<128x64x16x16xf32>
    %v423 = stablehlo.constant dense<1.0e-05> : tensor<128x64x16x16xf32>
    %v424 = stablehlo.reduce(%v421 init: %v420) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %v425 = stablehlo.broadcast_in_dim %v424, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x16x16xf32>
    %v426 = stablehlo.divide %v425, %v422 : tensor<128x64x16x16xf32>
    %v427 = stablehlo.subtract %v421, %v426 : tensor<128x64x16x16xf32>
    %v428 = stablehlo.multiply %v427, %v427 : tensor<128x64x16x16xf32>
    %v429 = stablehlo.reduce(%v428 init: %v420) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %v430 = stablehlo.broadcast_in_dim %v429, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x16x16xf32>
    %v431 = stablehlo.divide %v430, %v422 : tensor<128x64x16x16xf32>
    %v432 = stablehlo.add %v431, %v423 : tensor<128x64x16x16xf32>
    %v433 = stablehlo.rsqrt %v432 : tensor<128x64x16x16xf32>
    %v434 = stablehlo.multiply %v427, %v433 : tensor<128x64x16x16xf32>
    %v435 = stablehlo.reshape %v203 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v436 = stablehlo.multiply %v435, %v434 : tensor<128x64x16x16xf32>
    %v437 = stablehlo.reduce(%v436 init: %v420) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<64xf32>
    %v438 = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %v439 = stablehlo.multiply %v437, %v438 : tensor<64xf32>
    %v440 = stablehlo.subtract %g3, %v439 : tensor<64xf32>
    %v441 = stablehlo.constant dense<0.0> : tensor<f32>
    %v442 = stablehlo.reshape %v203 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v443 = stablehlo.reduce(%v442 init: %v441) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<64xf32>
    %v444 = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %v445 = stablehlo.multiply %v443, %v444 : tensor<64xf32>
    %v446 = stablehlo.subtract %bt3, %v445 : tensor<64xf32>
    %v447 = stablehlo.reshape %v90 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v448 = stablehlo.reshape %v192 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v449 = stablehlo.transpose %v447, dims = [1, 0, 2, 3] : (tensor<128x64x16x16xf32>) -> tensor<64x128x16x16xf32>
    %v450 = stablehlo.transpose %v448, dims = [1, 0, 2, 3] : (tensor<128x64x16x16xf32>) -> tensor<64x128x16x16xf32>
    %v451 = stablehlo.convolution(%v449, %v450)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x16x16xf32>, tensor<64x128x16x16xf32>) -> tensor<64x64x3x3xf32>
    %v452 = stablehlo.transpose %v451, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v453 = stablehlo.constant dense<0.00078125> : tensor<64x64x3x3xf32>
    %v454 = stablehlo.multiply %v452, %v453 : tensor<64x64x3x3xf32>
    %v455 = stablehlo.subtract %W4, %v454 : tensor<64x64x3x3xf32>
    %v456 = stablehlo.reshape %v192 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v457 = stablehlo.constant dense<0.0> : tensor<f32>
    %v458 = stablehlo.reduce(%v456 init: %v457) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<64xf32>
    %v459 = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %v460 = stablehlo.multiply %v458, %v459 : tensor<64xf32>
    %v461 = stablehlo.subtract %b4, %v460 : tensor<64xf32>
    %v462 = stablehlo.constant dense<0.0> : tensor<f32>
    %v463 = stablehlo.reshape %v95 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v464 = stablehlo.constant dense<256.0> : tensor<128x64x16x16xf32>
    %v465 = stablehlo.constant dense<1.0e-05> : tensor<128x64x16x16xf32>
    %v466 = stablehlo.reduce(%v463 init: %v462) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %v467 = stablehlo.broadcast_in_dim %v466, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x16x16xf32>
    %v468 = stablehlo.divide %v467, %v464 : tensor<128x64x16x16xf32>
    %v469 = stablehlo.subtract %v463, %v468 : tensor<128x64x16x16xf32>
    %v470 = stablehlo.multiply %v469, %v469 : tensor<128x64x16x16xf32>
    %v471 = stablehlo.reduce(%v470 init: %v462) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64xf32>
    %v472 = stablehlo.broadcast_in_dim %v471, dims = [0, 1] : (tensor<128x64xf32>) -> tensor<128x64x16x16xf32>
    %v473 = stablehlo.divide %v472, %v464 : tensor<128x64x16x16xf32>
    %v474 = stablehlo.add %v473, %v465 : tensor<128x64x16x16xf32>
    %v475 = stablehlo.rsqrt %v474 : tensor<128x64x16x16xf32>
    %v476 = stablehlo.multiply %v469, %v475 : tensor<128x64x16x16xf32>
    %v477 = stablehlo.reshape %v162 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v478 = stablehlo.multiply %v477, %v476 : tensor<128x64x16x16xf32>
    %v479 = stablehlo.reduce(%v478 init: %v462) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<64xf32>
    %v480 = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %v481 = stablehlo.multiply %v479, %v480 : tensor<64xf32>
    %v482 = stablehlo.subtract %g4, %v481 : tensor<64xf32>
    %v483 = stablehlo.constant dense<0.0> : tensor<f32>
    %v484 = stablehlo.reshape %v162 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v485 = stablehlo.reduce(%v484 init: %v483) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<64xf32>
    %v486 = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %v487 = stablehlo.multiply %v485, %v486 : tensor<64xf32>
    %v488 = stablehlo.subtract %bt4, %v487 : tensor<64xf32>
    %v489 = stablehlo.dot_general %v123, %v150, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x4096xf32>, tensor<128x512xf32>) -> tensor<4096x512xf32>
    %v490 = stablehlo.constant dense<0.00078125> : tensor<4096x512xf32>
    %v491 = stablehlo.multiply %v489, %v490 : tensor<4096x512xf32>
    %v492 = stablehlo.subtract %W5, %v491 : tensor<4096x512xf32>
    %v493 = stablehlo.constant dense<0.0> : tensor<f32>
    %v494 = stablehlo.reduce(%v150 init: %v493) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v495 = stablehlo.constant dense<0.00078125> : tensor<512xf32>
    %v496 = stablehlo.multiply %v494, %v495 : tensor<512xf32>
    %v497 = stablehlo.subtract %b5, %v496 : tensor<512xf32>
    %v498 = stablehlo.dot_general %v128, %v146, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<512x512xf32>
    %v499 = stablehlo.constant dense<0.00078125> : tensor<512x512xf32>
    %v500 = stablehlo.multiply %v498, %v499 : tensor<512x512xf32>
    %v501 = stablehlo.subtract %W6, %v500 : tensor<512x512xf32>
    %v502 = stablehlo.constant dense<0.0> : tensor<f32>
    %v503 = stablehlo.reduce(%v146 init: %v502) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v504 = stablehlo.constant dense<0.00078125> : tensor<512xf32>
    %v505 = stablehlo.multiply %v503, %v504 : tensor<512xf32>
    %v506 = stablehlo.subtract %b6, %v505 : tensor<512xf32>
    %v507 = stablehlo.dot_general %v133, %v142, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x10xf32>) -> tensor<512x10xf32>
    %v508 = stablehlo.constant dense<0.00078125> : tensor<512x10xf32>
    %v509 = stablehlo.multiply %v507, %v508 : tensor<512x10xf32>
    %v510 = stablehlo.subtract %W7, %v509 : tensor<512x10xf32>
    %v511 = stablehlo.constant dense<0.0> : tensor<f32>
    %v512 = stablehlo.reduce(%v142 init: %v511) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v513 = stablehlo.constant dense<0.00078125> : tensor<10xf32>
    %v514 = stablehlo.multiply %v512, %v513 : tensor<10xf32>
    %v515 = stablehlo.subtract %b7, %v514 : tensor<10xf32>
    return %v329, %v335, %v356, %v362, %v371, %v377, %v398, %v404, %v413, %v419, %v440, %v446, %v455, %v461, %v482, %v488, %v492, %v497, %v501, %v506, %v510, %v515 : tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<64x32x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<4096x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>
  }
}
