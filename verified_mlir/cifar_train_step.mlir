module @m {
  func.func @cifar_train_step(%x: tensor<128x3072xf32>, %W1: tensor<32x3x3x3xf32>, %b1: tensor<32xf32>, %W2: tensor<32x32x3x3xf32>, %b2: tensor<32xf32>, %W3: tensor<64x32x3x3xf32>, %b3: tensor<64xf32>, %W4: tensor<64x64x3x3xf32>, %b4: tensor<64xf32>, %W5: tensor<4096x512xf32>, %b5: tensor<512xf32>, %W6: tensor<512x512xf32>, %b6: tensor<512xf32>, %W7: tensor<512x10xf32>, %b7: tensor<10xf32>, %onehot: tensor<128x10xf32>) -> (tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<64x32x3x3xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<4096x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>) {
    // ── cifar train step: every line is pretty(verified AST node) ──
    %v0 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v1 = stablehlo.convolution(%v0, %W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x3x32x32xf32>, tensor<32x3x3x3xf32>) -> tensor<128x32x32x32xf32>
    %v2 = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<32xf32>) -> tensor<128x32x32x32xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<128x32x32x32xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v6 = stablehlo.constant dense<0.0> : tensor<128x32x32x32xf32>
    %v7 = stablehlo.maximum %v5, %v6 : tensor<128x32x32x32xf32>
    %v8 = stablehlo.reshape %v7 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %v9 = stablehlo.reshape %v8 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v10 = stablehlo.convolution(%v9, %W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x32x32xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x32x32xf32>
    %v11 = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<32xf32>) -> tensor<128x32x32x32xf32>
    %v12 = stablehlo.add %v10, %v11 : tensor<128x32x32x32xf32>
    %v13 = stablehlo.reshape %v12 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %v14 = stablehlo.reshape %v13 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v15 = stablehlo.constant dense<0.0> : tensor<128x32x32x32xf32>
    %v16 = stablehlo.maximum %v14, %v15 : tensor<128x32x32x32xf32>
    %v17 = stablehlo.reshape %v16 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %v18 = stablehlo.reshape %v17 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v19 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v20 = "stablehlo.reduce_window"(%v18, %v19) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32x16x16xf32>
    %v21 = stablehlo.reshape %v20 : (tensor<128x32x16x16xf32>) -> tensor<128x8192xf32>
    %v22 = stablehlo.reshape %v21 : (tensor<128x8192xf32>) -> tensor<128x32x16x16xf32>
    %v23 = stablehlo.convolution(%v22, %W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x16x16xf32>, tensor<64x32x3x3xf32>) -> tensor<128x64x16x16xf32>
    %v24 = stablehlo.broadcast_in_dim %b3, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %v25 = stablehlo.add %v23, %v24 : tensor<128x64x16x16xf32>
    %v26 = stablehlo.reshape %v25 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v27 = stablehlo.reshape %v26 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v28 = stablehlo.constant dense<0.0> : tensor<128x64x16x16xf32>
    %v29 = stablehlo.maximum %v27, %v28 : tensor<128x64x16x16xf32>
    %v30 = stablehlo.reshape %v29 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v32 = stablehlo.convolution(%v31, %W4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x16x16xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x16x16xf32>
    %v33 = stablehlo.broadcast_in_dim %b4, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %v34 = stablehlo.add %v32, %v33 : tensor<128x64x16x16xf32>
    %v35 = stablehlo.reshape %v34 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v36 = stablehlo.reshape %v35 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v37 = stablehlo.constant dense<0.0> : tensor<128x64x16x16xf32>
    %v38 = stablehlo.maximum %v36, %v37 : tensor<128x64x16x16xf32>
    %v39 = stablehlo.reshape %v38 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v40 = stablehlo.reshape %v39 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v41 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v42 = "stablehlo.reduce_window"(%v40, %v41) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64x8x8xf32>
    %v43 = stablehlo.reshape %v42 : (tensor<128x64x8x8xf32>) -> tensor<128x4096xf32>
    %v44 = stablehlo.dot_general %v43, %W5, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x4096xf32>, tensor<4096x512xf32>) -> tensor<128x512xf32>
    %v45 = stablehlo.broadcast_in_dim %b5, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v46 = stablehlo.add %v44, %v45 : tensor<128x512xf32>
    %v47 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v48 = stablehlo.maximum %v46, %v47 : tensor<128x512xf32>
    %v49 = stablehlo.dot_general %v48, %W6, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %v50 = stablehlo.broadcast_in_dim %b6, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v51 = stablehlo.add %v49, %v50 : tensor<128x512xf32>
    %v52 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v53 = stablehlo.maximum %v51, %v52 : tensor<128x512xf32>
    %v54 = stablehlo.dot_general %v53, %W7, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x10xf32>) -> tensor<128x10xf32>
    %v55 = stablehlo.broadcast_in_dim %b7, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v56 = stablehlo.add %v54, %v55 : tensor<128x10xf32>
    %v57 = stablehlo.exponential %v56 : tensor<128x10xf32>
    %v58 = stablehlo.constant dense<0.0> : tensor<f32>
    %v59 = stablehlo.reduce(%v57 init: %v58) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %v60 = stablehlo.broadcast_in_dim %v59, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %v61 = stablehlo.divide %v57, %v60 : tensor<128x10xf32>
    %v62 = stablehlo.subtract %v61, %onehot : tensor<128x10xf32>
    %v63 = stablehlo.dot_general %v62, %W7, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x10xf32>, tensor<512x10xf32>) -> tensor<128x512xf32>
    %v64 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v65 = stablehlo.compare GT, %v51, %v64 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v66 = stablehlo.select %v65, %v63, %v64 : tensor<128x512xi1>, tensor<128x512xf32>
    %v67 = stablehlo.dot_general %v66, %W6, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %v68 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v69 = stablehlo.compare GT, %v46, %v68 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v70 = stablehlo.select %v69, %v67, %v68 : tensor<128x512xi1>, tensor<128x512xf32>
    %v71 = stablehlo.dot_general %v70, %W5, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<4096x512xf32>) -> tensor<128x4096xf32>
    %v72 = stablehlo.reshape %v39 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v73 = stablehlo.reshape %v71 : (tensor<128x4096xf32>) -> tensor<128x64x8x8xf32>
    %v74 = stablehlo.constant dense<0.0> : tensor<f32>
    %v75 = "stablehlo.select_and_scatter"(%v72, %v73, %v74) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x64x16x16xf32>, tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<128x64x16x16xf32>
    %v76 = stablehlo.reshape %v75 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v77 = stablehlo.reshape %v76 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v78 = stablehlo.reshape %v35 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v79 = stablehlo.constant dense<0.0> : tensor<128x64x16x16xf32>
    %v80 = stablehlo.compare GT, %v78, %v79 : (tensor<128x64x16x16xf32>, tensor<128x64x16x16xf32>) -> tensor<128x64x16x16xi1>
    %v81 = stablehlo.select %v80, %v77, %v79 : tensor<128x64x16x16xi1>, tensor<128x64x16x16xf32>
    %v82 = stablehlo.reshape %v81 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v83 = stablehlo.reshape %v82 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v84 = stablehlo.transpose %W4, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v85 = stablehlo.reverse %v84, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v86 = stablehlo.convolution(%v83, %v85)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x16x16xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x16x16xf32>
    %v87 = stablehlo.reshape %v86 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v88 = stablehlo.reshape %v87 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v89 = stablehlo.reshape %v26 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v90 = stablehlo.constant dense<0.0> : tensor<128x64x16x16xf32>
    %v91 = stablehlo.compare GT, %v89, %v90 : (tensor<128x64x16x16xf32>, tensor<128x64x16x16xf32>) -> tensor<128x64x16x16xi1>
    %v92 = stablehlo.select %v91, %v88, %v90 : tensor<128x64x16x16xi1>, tensor<128x64x16x16xf32>
    %v93 = stablehlo.reshape %v92 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v94 = stablehlo.reshape %v93 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v95 = stablehlo.transpose %W3, dims = [1, 0, 2, 3] : (tensor<64x32x3x3xf32>) -> tensor<32x64x3x3xf32>
    %v96 = stablehlo.reverse %v95, dims = [2, 3] : tensor<32x64x3x3xf32>
    %v97 = stablehlo.convolution(%v94, %v96)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x16x16xf32>, tensor<32x64x3x3xf32>) -> tensor<128x32x16x16xf32>
    %v98 = stablehlo.reshape %v97 : (tensor<128x32x16x16xf32>) -> tensor<128x8192xf32>
    %v99 = stablehlo.reshape %v17 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v100 = stablehlo.reshape %v98 : (tensor<128x8192xf32>) -> tensor<128x32x16x16xf32>
    %v101 = stablehlo.constant dense<0.0> : tensor<f32>
    %v102 = "stablehlo.select_and_scatter"(%v99, %v100, %v101) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x32x32xf32>, tensor<128x32x16x16xf32>, tensor<f32>) -> tensor<128x32x32x32xf32>
    %v103 = stablehlo.reshape %v102 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %v104 = stablehlo.reshape %v103 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v105 = stablehlo.reshape %v13 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v106 = stablehlo.constant dense<0.0> : tensor<128x32x32x32xf32>
    %v107 = stablehlo.compare GT, %v105, %v106 : (tensor<128x32x32x32xf32>, tensor<128x32x32x32xf32>) -> tensor<128x32x32x32xi1>
    %v108 = stablehlo.select %v107, %v104, %v106 : tensor<128x32x32x32xi1>, tensor<128x32x32x32xf32>
    %v109 = stablehlo.reshape %v108 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %v110 = stablehlo.reshape %v109 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v111 = stablehlo.transpose %W2, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v112 = stablehlo.reverse %v111, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v113 = stablehlo.convolution(%v110, %v112)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x32x32xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x32x32xf32>
    %v114 = stablehlo.reshape %v113 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %v115 = stablehlo.reshape %v114 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v116 = stablehlo.reshape %v4 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v117 = stablehlo.constant dense<0.0> : tensor<128x32x32x32xf32>
    %v118 = stablehlo.compare GT, %v116, %v117 : (tensor<128x32x32x32xf32>, tensor<128x32x32x32xf32>) -> tensor<128x32x32x32xi1>
    %v119 = stablehlo.select %v118, %v115, %v117 : tensor<128x32x32x32xi1>, tensor<128x32x32x32xf32>
    %v120 = stablehlo.reshape %v119 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %v121 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v122 = stablehlo.reshape %v120 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v123 = stablehlo.transpose %v121, dims = [1, 0, 2, 3] : (tensor<128x3x32x32xf32>) -> tensor<3x128x32x32xf32>
    %v124 = stablehlo.transpose %v122, dims = [1, 0, 2, 3] : (tensor<128x32x32x32xf32>) -> tensor<32x128x32x32xf32>
    %v125 = stablehlo.convolution(%v123, %v124)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x128x32x32xf32>, tensor<32x128x32x32xf32>) -> tensor<3x32x3x3xf32>
    %v126 = stablehlo.transpose %v125, dims = [1, 0, 2, 3] : (tensor<3x32x3x3xf32>) -> tensor<32x3x3x3xf32>
    %v127 = stablehlo.constant dense<0.00078125> : tensor<32x3x3x3xf32>
    %v128 = stablehlo.multiply %v126, %v127 : tensor<32x3x3x3xf32>
    %v129 = stablehlo.subtract %W1, %v128 : tensor<32x3x3x3xf32>
    %v130 = stablehlo.reshape %v120 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v131 = stablehlo.constant dense<0.0> : tensor<f32>
    %v132 = stablehlo.reduce(%v130 init: %v131) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<32xf32>
    %v133 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v134 = stablehlo.multiply %v132, %v133 : tensor<32xf32>
    %v135 = stablehlo.subtract %b1, %v134 : tensor<32xf32>
    %v136 = stablehlo.reshape %v8 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v137 = stablehlo.reshape %v109 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v138 = stablehlo.transpose %v136, dims = [1, 0, 2, 3] : (tensor<128x32x32x32xf32>) -> tensor<32x128x32x32xf32>
    %v139 = stablehlo.transpose %v137, dims = [1, 0, 2, 3] : (tensor<128x32x32x32xf32>) -> tensor<32x128x32x32xf32>
    %v140 = stablehlo.convolution(%v138, %v139)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x32x32xf32>, tensor<32x128x32x32xf32>) -> tensor<32x32x3x3xf32>
    %v141 = stablehlo.transpose %v140, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v142 = stablehlo.constant dense<0.00078125> : tensor<32x32x3x3xf32>
    %v143 = stablehlo.multiply %v141, %v142 : tensor<32x32x3x3xf32>
    %v144 = stablehlo.subtract %W2, %v143 : tensor<32x32x3x3xf32>
    %v145 = stablehlo.reshape %v109 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v146 = stablehlo.constant dense<0.0> : tensor<f32>
    %v147 = stablehlo.reduce(%v145 init: %v146) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<32xf32>
    %v148 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v149 = stablehlo.multiply %v147, %v148 : tensor<32xf32>
    %v150 = stablehlo.subtract %b2, %v149 : tensor<32xf32>
    %v151 = stablehlo.reshape %v21 : (tensor<128x8192xf32>) -> tensor<128x32x16x16xf32>
    %v152 = stablehlo.reshape %v93 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v153 = stablehlo.transpose %v151, dims = [1, 0, 2, 3] : (tensor<128x32x16x16xf32>) -> tensor<32x128x16x16xf32>
    %v154 = stablehlo.transpose %v152, dims = [1, 0, 2, 3] : (tensor<128x64x16x16xf32>) -> tensor<64x128x16x16xf32>
    %v155 = stablehlo.convolution(%v153, %v154)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x16x16xf32>, tensor<64x128x16x16xf32>) -> tensor<32x64x3x3xf32>
    %v156 = stablehlo.transpose %v155, dims = [1, 0, 2, 3] : (tensor<32x64x3x3xf32>) -> tensor<64x32x3x3xf32>
    %v157 = stablehlo.constant dense<0.00078125> : tensor<64x32x3x3xf32>
    %v158 = stablehlo.multiply %v156, %v157 : tensor<64x32x3x3xf32>
    %v159 = stablehlo.subtract %W3, %v158 : tensor<64x32x3x3xf32>
    %v160 = stablehlo.reshape %v93 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v161 = stablehlo.constant dense<0.0> : tensor<f32>
    %v162 = stablehlo.reduce(%v160 init: %v161) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<64xf32>
    %v163 = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %v164 = stablehlo.multiply %v162, %v163 : tensor<64xf32>
    %v165 = stablehlo.subtract %b3, %v164 : tensor<64xf32>
    %v166 = stablehlo.reshape %v30 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v167 = stablehlo.reshape %v82 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v168 = stablehlo.transpose %v166, dims = [1, 0, 2, 3] : (tensor<128x64x16x16xf32>) -> tensor<64x128x16x16xf32>
    %v169 = stablehlo.transpose %v167, dims = [1, 0, 2, 3] : (tensor<128x64x16x16xf32>) -> tensor<64x128x16x16xf32>
    %v170 = stablehlo.convolution(%v168, %v169)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x16x16xf32>, tensor<64x128x16x16xf32>) -> tensor<64x64x3x3xf32>
    %v171 = stablehlo.transpose %v170, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v172 = stablehlo.constant dense<0.00078125> : tensor<64x64x3x3xf32>
    %v173 = stablehlo.multiply %v171, %v172 : tensor<64x64x3x3xf32>
    %v174 = stablehlo.subtract %W4, %v173 : tensor<64x64x3x3xf32>
    %v175 = stablehlo.reshape %v82 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v176 = stablehlo.constant dense<0.0> : tensor<f32>
    %v177 = stablehlo.reduce(%v175 init: %v176) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<64xf32>
    %v178 = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %v179 = stablehlo.multiply %v177, %v178 : tensor<64xf32>
    %v180 = stablehlo.subtract %b4, %v179 : tensor<64xf32>
    %v181 = stablehlo.dot_general %v43, %v70, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x4096xf32>, tensor<128x512xf32>) -> tensor<4096x512xf32>
    %v182 = stablehlo.constant dense<0.00078125> : tensor<4096x512xf32>
    %v183 = stablehlo.multiply %v181, %v182 : tensor<4096x512xf32>
    %v184 = stablehlo.subtract %W5, %v183 : tensor<4096x512xf32>
    %v185 = stablehlo.constant dense<0.0> : tensor<f32>
    %v186 = stablehlo.reduce(%v70 init: %v185) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v187 = stablehlo.constant dense<0.00078125> : tensor<512xf32>
    %v188 = stablehlo.multiply %v186, %v187 : tensor<512xf32>
    %v189 = stablehlo.subtract %b5, %v188 : tensor<512xf32>
    %v190 = stablehlo.dot_general %v48, %v66, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<512x512xf32>
    %v191 = stablehlo.constant dense<0.00078125> : tensor<512x512xf32>
    %v192 = stablehlo.multiply %v190, %v191 : tensor<512x512xf32>
    %v193 = stablehlo.subtract %W6, %v192 : tensor<512x512xf32>
    %v194 = stablehlo.constant dense<0.0> : tensor<f32>
    %v195 = stablehlo.reduce(%v66 init: %v194) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v196 = stablehlo.constant dense<0.00078125> : tensor<512xf32>
    %v197 = stablehlo.multiply %v195, %v196 : tensor<512xf32>
    %v198 = stablehlo.subtract %b6, %v197 : tensor<512xf32>
    %v199 = stablehlo.dot_general %v53, %v62, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x10xf32>) -> tensor<512x10xf32>
    %v200 = stablehlo.constant dense<0.00078125> : tensor<512x10xf32>
    %v201 = stablehlo.multiply %v199, %v200 : tensor<512x10xf32>
    %v202 = stablehlo.subtract %W7, %v201 : tensor<512x10xf32>
    %v203 = stablehlo.constant dense<0.0> : tensor<f32>
    %v204 = stablehlo.reduce(%v62 init: %v203) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v205 = stablehlo.constant dense<0.00078125> : tensor<10xf32>
    %v206 = stablehlo.multiply %v204, %v205 : tensor<10xf32>
    %v207 = stablehlo.subtract %b7, %v206 : tensor<10xf32>
    return %v129, %v135, %v144, %v150, %v159, %v165, %v174, %v180, %v184, %v189, %v193, %v198, %v202, %v207 : tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<64x32x3x3xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<4096x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>
  }
}
