module @m {
  func.func @cifar8_train_step(%x: tensor<128x3072xf32>, %W1: tensor<16x3x3x3xf32>, %b1: tensor<16xf32>, %W2: tensor<16x16x3x3xf32>, %b2: tensor<16xf32>, %W3: tensor<16x16x3x3xf32>, %b3: tensor<16xf32>, %W4: tensor<16x16x3x3xf32>, %b4: tensor<16xf32>, %W5: tensor<32x16x3x3xf32>, %b5: tensor<32xf32>, %W6: tensor<32x32x3x3xf32>, %b6: tensor<32xf32>, %W7: tensor<32x32x3x3xf32>, %b7: tensor<32xf32>, %W8: tensor<32x32x3x3xf32>, %b8: tensor<32xf32>, %W9: tensor<128x64xf32>, %b9: tensor<64xf32>, %Wa: tensor<64x64xf32>, %ba: tensor<64xf32>, %Wb: tensor<64x10xf32>, %bb: tensor<10xf32>, %onehot: tensor<128x10xf32>) -> (tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>) {
    // ── cifar8 train step: every line is pretty(verified AST node) ──
    %v0 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v1 = stablehlo.convolution(%v0, %W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x3x32x32xf32>, tensor<16x3x3x3xf32>) -> tensor<128x16x32x32xf32>
    %v2 = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<128x16x32x32xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v6 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v7 = stablehlo.maximum %v5, %v6 : tensor<128x16x32x32xf32>
    %v8 = stablehlo.reshape %v7 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v9 = stablehlo.reshape %v8 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v10 = stablehlo.convolution(%v9, %W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x32x32xf32>
    %v11 = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v12 = stablehlo.add %v10, %v11 : tensor<128x16x32x32xf32>
    %v13 = stablehlo.reshape %v12 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v14 = stablehlo.reshape %v13 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v15 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v16 = stablehlo.maximum %v14, %v15 : tensor<128x16x32x32xf32>
    %v17 = stablehlo.reshape %v16 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v18 = stablehlo.reshape %v17 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v19 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v20 = "stablehlo.reduce_window"(%v18, %v19) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16x16x16xf32>
    %v21 = stablehlo.reshape %v20 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v22 = stablehlo.reshape %v21 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v23 = stablehlo.convolution(%v22, %W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v24 = stablehlo.broadcast_in_dim %b3, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v25 = stablehlo.add %v23, %v24 : tensor<128x16x16x16xf32>
    %v26 = stablehlo.reshape %v25 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v27 = stablehlo.reshape %v26 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v28 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v29 = stablehlo.maximum %v27, %v28 : tensor<128x16x16x16xf32>
    %v30 = stablehlo.reshape %v29 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v32 = stablehlo.convolution(%v31, %W4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v33 = stablehlo.broadcast_in_dim %b4, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v34 = stablehlo.add %v32, %v33 : tensor<128x16x16x16xf32>
    %v35 = stablehlo.reshape %v34 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v36 = stablehlo.reshape %v35 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v37 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v38 = stablehlo.maximum %v36, %v37 : tensor<128x16x16x16xf32>
    %v39 = stablehlo.reshape %v38 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v40 = stablehlo.reshape %v39 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v41 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v42 = "stablehlo.reduce_window"(%v40, %v41) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x8x8xf32>
    %v43 = stablehlo.reshape %v42 : (tensor<128x16x8x8xf32>) -> tensor<128x1024xf32>
    %v44 = stablehlo.reshape %v43 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v45 = stablehlo.convolution(%v44, %W5)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x8x8xf32>, tensor<32x16x3x3xf32>) -> tensor<128x32x8x8xf32>
    %v46 = stablehlo.broadcast_in_dim %b5, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v47 = stablehlo.add %v45, %v46 : tensor<128x32x8x8xf32>
    %v48 = stablehlo.reshape %v47 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v49 = stablehlo.reshape %v48 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v50 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v51 = stablehlo.maximum %v49, %v50 : tensor<128x32x8x8xf32>
    %v52 = stablehlo.reshape %v51 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v53 = stablehlo.reshape %v52 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v54 = stablehlo.convolution(%v53, %W6)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x8x8xf32>
    %v55 = stablehlo.broadcast_in_dim %b6, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v56 = stablehlo.add %v54, %v55 : tensor<128x32x8x8xf32>
    %v57 = stablehlo.reshape %v56 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v59 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v60 = stablehlo.maximum %v58, %v59 : tensor<128x32x8x8xf32>
    %v61 = stablehlo.reshape %v60 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v62 = stablehlo.reshape %v61 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v63 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v64 = "stablehlo.reduce_window"(%v62, %v63) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32x4x4xf32>
    %v65 = stablehlo.reshape %v64 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v66 = stablehlo.reshape %v65 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v67 = stablehlo.convolution(%v66, %W7)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v68 = stablehlo.broadcast_in_dim %b7, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v69 = stablehlo.add %v67, %v68 : tensor<128x32x4x4xf32>
    %v70 = stablehlo.reshape %v69 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v71 = stablehlo.reshape %v70 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v72 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v73 = stablehlo.maximum %v71, %v72 : tensor<128x32x4x4xf32>
    %v74 = stablehlo.reshape %v73 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v75 = stablehlo.reshape %v74 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v76 = stablehlo.convolution(%v75, %W8)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v77 = stablehlo.broadcast_in_dim %b8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v78 = stablehlo.add %v76, %v77 : tensor<128x32x4x4xf32>
    %v79 = stablehlo.reshape %v78 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v80 = stablehlo.reshape %v79 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v81 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v82 = stablehlo.maximum %v80, %v81 : tensor<128x32x4x4xf32>
    %v83 = stablehlo.reshape %v82 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v84 = stablehlo.reshape %v83 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v85 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v86 = "stablehlo.reduce_window"(%v84, %v85) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32x2x2xf32>
    %v87 = stablehlo.reshape %v86 : (tensor<128x32x2x2xf32>) -> tensor<128x128xf32>
    %v88 = stablehlo.dot_general %v87, %W9, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %v89 = stablehlo.broadcast_in_dim %b9, dims = [1] : (tensor<64xf32>) -> tensor<128x64xf32>
    %v90 = stablehlo.add %v88, %v89 : tensor<128x64xf32>
    %v91 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v92 = stablehlo.maximum %v90, %v91 : tensor<128x64xf32>
    %v93 = stablehlo.dot_general %v92, %Wa, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
    %v94 = stablehlo.broadcast_in_dim %ba, dims = [1] : (tensor<64xf32>) -> tensor<128x64xf32>
    %v95 = stablehlo.add %v93, %v94 : tensor<128x64xf32>
    %v96 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v97 = stablehlo.maximum %v95, %v96 : tensor<128x64xf32>
    %v98 = stablehlo.dot_general %v97, %Wb, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x10xf32>) -> tensor<128x10xf32>
    %v99 = stablehlo.broadcast_in_dim %bb, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v100 = stablehlo.add %v98, %v99 : tensor<128x10xf32>
    %v101 = stablehlo.exponential %v100 : tensor<128x10xf32>
    %v102 = stablehlo.constant dense<0.0> : tensor<f32>
    %v103 = stablehlo.reduce(%v101 init: %v102) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %v104 = stablehlo.broadcast_in_dim %v103, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %v105 = stablehlo.divide %v101, %v104 : tensor<128x10xf32>
    %v106 = stablehlo.subtract %v105, %onehot : tensor<128x10xf32>
    %v107 = stablehlo.dot_general %v106, %Wb, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x10xf32>, tensor<64x10xf32>) -> tensor<128x64xf32>
    %v108 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v109 = stablehlo.compare GT, %v95, %v108 : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xi1>
    %v110 = stablehlo.select %v109, %v107, %v108 : tensor<128x64xi1>, tensor<128x64xf32>
    %v111 = stablehlo.dot_general %v110, %Wa, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
    %v112 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v113 = stablehlo.compare GT, %v90, %v112 : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xi1>
    %v114 = stablehlo.select %v113, %v111, %v112 : tensor<128x64xi1>, tensor<128x64xf32>
    %v115 = stablehlo.dot_general %v114, %W9, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x128xf32>
    %v116 = stablehlo.reshape %v83 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v117 = stablehlo.reshape %v115 : (tensor<128x128xf32>) -> tensor<128x32x2x2xf32>
    %v118 = stablehlo.constant dense<0.0> : tensor<f32>
    %v119 = "stablehlo.select_and_scatter"(%v116, %v117, %v118) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x4x4xf32>, tensor<128x32x2x2xf32>, tensor<f32>) -> tensor<128x32x4x4xf32>
    %v120 = stablehlo.reshape %v119 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v121 = stablehlo.reshape %v120 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v122 = stablehlo.reshape %v79 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v123 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v124 = stablehlo.compare GT, %v122, %v123 : (tensor<128x32x4x4xf32>, tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xi1>
    %v125 = stablehlo.select %v124, %v121, %v123 : tensor<128x32x4x4xi1>, tensor<128x32x4x4xf32>
    %v126 = stablehlo.reshape %v125 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v127 = stablehlo.reshape %v126 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v128 = stablehlo.transpose %W8, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v129 = stablehlo.reverse %v128, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v130 = stablehlo.convolution(%v127, %v129)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v131 = stablehlo.reshape %v130 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v132 = stablehlo.reshape %v131 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v133 = stablehlo.reshape %v70 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v134 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v135 = stablehlo.compare GT, %v133, %v134 : (tensor<128x32x4x4xf32>, tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xi1>
    %v136 = stablehlo.select %v135, %v132, %v134 : tensor<128x32x4x4xi1>, tensor<128x32x4x4xf32>
    %v137 = stablehlo.reshape %v136 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v138 = stablehlo.reshape %v137 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v139 = stablehlo.transpose %W7, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v140 = stablehlo.reverse %v139, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v141 = stablehlo.convolution(%v138, %v140)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v142 = stablehlo.reshape %v141 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v143 = stablehlo.reshape %v61 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v144 = stablehlo.reshape %v142 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v145 = stablehlo.constant dense<0.0> : tensor<f32>
    %v146 = "stablehlo.select_and_scatter"(%v143, %v144, %v145) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x8x8xf32>, tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32x8x8xf32>
    %v147 = stablehlo.reshape %v146 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v148 = stablehlo.reshape %v147 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v149 = stablehlo.reshape %v57 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v150 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v151 = stablehlo.compare GT, %v149, %v150 : (tensor<128x32x8x8xf32>, tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xi1>
    %v152 = stablehlo.select %v151, %v148, %v150 : tensor<128x32x8x8xi1>, tensor<128x32x8x8xf32>
    %v153 = stablehlo.reshape %v152 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v154 = stablehlo.reshape %v153 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v155 = stablehlo.transpose %W6, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v156 = stablehlo.reverse %v155, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v157 = stablehlo.convolution(%v154, %v156)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x8x8xf32>
    %v158 = stablehlo.reshape %v157 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v159 = stablehlo.reshape %v158 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v160 = stablehlo.reshape %v48 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v161 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v162 = stablehlo.compare GT, %v160, %v161 : (tensor<128x32x8x8xf32>, tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xi1>
    %v163 = stablehlo.select %v162, %v159, %v161 : tensor<128x32x8x8xi1>, tensor<128x32x8x8xf32>
    %v164 = stablehlo.reshape %v163 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v165 = stablehlo.reshape %v164 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v166 = stablehlo.transpose %W5, dims = [1, 0, 2, 3] : (tensor<32x16x3x3xf32>) -> tensor<16x32x3x3xf32>
    %v167 = stablehlo.reverse %v166, dims = [2, 3] : tensor<16x32x3x3xf32>
    %v168 = stablehlo.convolution(%v165, %v167)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<16x32x3x3xf32>) -> tensor<128x16x8x8xf32>
    %v169 = stablehlo.reshape %v168 : (tensor<128x16x8x8xf32>) -> tensor<128x1024xf32>
    %v170 = stablehlo.reshape %v39 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v171 = stablehlo.reshape %v169 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v172 = stablehlo.constant dense<0.0> : tensor<f32>
    %v173 = "stablehlo.select_and_scatter"(%v170, %v171, %v172) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x16x16xf32>, tensor<128x16x8x8xf32>, tensor<f32>) -> tensor<128x16x16x16xf32>
    %v174 = stablehlo.reshape %v173 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v175 = stablehlo.reshape %v174 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v176 = stablehlo.reshape %v35 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v177 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v178 = stablehlo.compare GT, %v176, %v177 : (tensor<128x16x16x16xf32>, tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xi1>
    %v179 = stablehlo.select %v178, %v175, %v177 : tensor<128x16x16x16xi1>, tensor<128x16x16x16xf32>
    %v180 = stablehlo.reshape %v179 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v181 = stablehlo.reshape %v180 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v182 = stablehlo.transpose %W4, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v183 = stablehlo.reverse %v182, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v184 = stablehlo.convolution(%v181, %v183)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v185 = stablehlo.reshape %v184 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v186 = stablehlo.reshape %v185 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v187 = stablehlo.reshape %v26 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v188 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v189 = stablehlo.compare GT, %v187, %v188 : (tensor<128x16x16x16xf32>, tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xi1>
    %v190 = stablehlo.select %v189, %v186, %v188 : tensor<128x16x16x16xi1>, tensor<128x16x16x16xf32>
    %v191 = stablehlo.reshape %v190 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v192 = stablehlo.reshape %v191 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v193 = stablehlo.transpose %W3, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v194 = stablehlo.reverse %v193, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v195 = stablehlo.convolution(%v192, %v194)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v196 = stablehlo.reshape %v195 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v197 = stablehlo.reshape %v17 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v198 = stablehlo.reshape %v196 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v199 = stablehlo.constant dense<0.0> : tensor<f32>
    %v200 = "stablehlo.select_and_scatter"(%v197, %v198, %v199) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x32x32xf32>, tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x32x32xf32>
    %v201 = stablehlo.reshape %v200 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v202 = stablehlo.reshape %v201 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v203 = stablehlo.reshape %v13 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v204 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v205 = stablehlo.compare GT, %v203, %v204 : (tensor<128x16x32x32xf32>, tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xi1>
    %v206 = stablehlo.select %v205, %v202, %v204 : tensor<128x16x32x32xi1>, tensor<128x16x32x32xf32>
    %v207 = stablehlo.reshape %v206 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v208 = stablehlo.reshape %v207 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v209 = stablehlo.transpose %W2, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v210 = stablehlo.reverse %v209, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v211 = stablehlo.convolution(%v208, %v210)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x32x32xf32>
    %v212 = stablehlo.reshape %v211 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v213 = stablehlo.reshape %v212 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v214 = stablehlo.reshape %v4 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v215 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v216 = stablehlo.compare GT, %v214, %v215 : (tensor<128x16x32x32xf32>, tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xi1>
    %v217 = stablehlo.select %v216, %v213, %v215 : tensor<128x16x32x32xi1>, tensor<128x16x32x32xf32>
    %v218 = stablehlo.reshape %v217 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v219 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v220 = stablehlo.reshape %v218 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v221 = stablehlo.transpose %v219, dims = [1, 0, 2, 3] : (tensor<128x3x32x32xf32>) -> tensor<3x128x32x32xf32>
    %v222 = stablehlo.transpose %v220, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v223 = stablehlo.convolution(%v221, %v222)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<3x16x3x3xf32>
    %v224 = stablehlo.transpose %v223, dims = [1, 0, 2, 3] : (tensor<3x16x3x3xf32>) -> tensor<16x3x3x3xf32>
    %v225 = stablehlo.constant dense<0.00078125> : tensor<16x3x3x3xf32>
    %v226 = stablehlo.multiply %v224, %v225 : tensor<16x3x3x3xf32>
    %v227 = stablehlo.subtract %W1, %v226 : tensor<16x3x3x3xf32>
    %v228 = stablehlo.reshape %v218 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v229 = stablehlo.constant dense<0.0> : tensor<f32>
    %v230 = stablehlo.reduce(%v228 init: %v229) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v231 = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %v232 = stablehlo.multiply %v230, %v231 : tensor<16xf32>
    %v233 = stablehlo.subtract %b1, %v232 : tensor<16xf32>
    %v234 = stablehlo.reshape %v8 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v235 = stablehlo.reshape %v207 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v236 = stablehlo.transpose %v234, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v237 = stablehlo.transpose %v235, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v238 = stablehlo.convolution(%v236, %v237)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x16x3x3xf32>
    %v239 = stablehlo.transpose %v238, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v240 = stablehlo.constant dense<0.00078125> : tensor<16x16x3x3xf32>
    %v241 = stablehlo.multiply %v239, %v240 : tensor<16x16x3x3xf32>
    %v242 = stablehlo.subtract %W2, %v241 : tensor<16x16x3x3xf32>
    %v243 = stablehlo.reshape %v207 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v244 = stablehlo.constant dense<0.0> : tensor<f32>
    %v245 = stablehlo.reduce(%v243 init: %v244) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v246 = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %v247 = stablehlo.multiply %v245, %v246 : tensor<16xf32>
    %v248 = stablehlo.subtract %b2, %v247 : tensor<16xf32>
    %v249 = stablehlo.reshape %v21 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v250 = stablehlo.reshape %v191 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v251 = stablehlo.transpose %v249, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v252 = stablehlo.transpose %v250, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v253 = stablehlo.convolution(%v251, %v252)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v254 = stablehlo.transpose %v253, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v255 = stablehlo.constant dense<0.00078125> : tensor<16x16x3x3xf32>
    %v256 = stablehlo.multiply %v254, %v255 : tensor<16x16x3x3xf32>
    %v257 = stablehlo.subtract %W3, %v256 : tensor<16x16x3x3xf32>
    %v258 = stablehlo.reshape %v191 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v259 = stablehlo.constant dense<0.0> : tensor<f32>
    %v260 = stablehlo.reduce(%v258 init: %v259) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v261 = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %v262 = stablehlo.multiply %v260, %v261 : tensor<16xf32>
    %v263 = stablehlo.subtract %b3, %v262 : tensor<16xf32>
    %v264 = stablehlo.reshape %v30 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v265 = stablehlo.reshape %v180 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v266 = stablehlo.transpose %v264, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v267 = stablehlo.transpose %v265, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v268 = stablehlo.convolution(%v266, %v267)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v269 = stablehlo.transpose %v268, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v270 = stablehlo.constant dense<0.00078125> : tensor<16x16x3x3xf32>
    %v271 = stablehlo.multiply %v269, %v270 : tensor<16x16x3x3xf32>
    %v272 = stablehlo.subtract %W4, %v271 : tensor<16x16x3x3xf32>
    %v273 = stablehlo.reshape %v180 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v274 = stablehlo.constant dense<0.0> : tensor<f32>
    %v275 = stablehlo.reduce(%v273 init: %v274) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v276 = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %v277 = stablehlo.multiply %v275, %v276 : tensor<16xf32>
    %v278 = stablehlo.subtract %b4, %v277 : tensor<16xf32>
    %v279 = stablehlo.reshape %v43 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v280 = stablehlo.reshape %v164 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v281 = stablehlo.transpose %v279, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v282 = stablehlo.transpose %v280, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v283 = stablehlo.convolution(%v281, %v282)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<16x32x3x3xf32>
    %v284 = stablehlo.transpose %v283, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v285 = stablehlo.constant dense<0.00078125> : tensor<32x16x3x3xf32>
    %v286 = stablehlo.multiply %v284, %v285 : tensor<32x16x3x3xf32>
    %v287 = stablehlo.subtract %W5, %v286 : tensor<32x16x3x3xf32>
    %v288 = stablehlo.reshape %v164 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v289 = stablehlo.constant dense<0.0> : tensor<f32>
    %v290 = stablehlo.reduce(%v288 init: %v289) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v291 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v292 = stablehlo.multiply %v290, %v291 : tensor<32xf32>
    %v293 = stablehlo.subtract %b5, %v292 : tensor<32xf32>
    %v294 = stablehlo.reshape %v52 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v295 = stablehlo.reshape %v153 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v296 = stablehlo.transpose %v294, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v297 = stablehlo.transpose %v295, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v298 = stablehlo.convolution(%v296, %v297)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<32x32x3x3xf32>
    %v299 = stablehlo.transpose %v298, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v300 = stablehlo.constant dense<0.00078125> : tensor<32x32x3x3xf32>
    %v301 = stablehlo.multiply %v299, %v300 : tensor<32x32x3x3xf32>
    %v302 = stablehlo.subtract %W6, %v301 : tensor<32x32x3x3xf32>
    %v303 = stablehlo.reshape %v153 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v304 = stablehlo.constant dense<0.0> : tensor<f32>
    %v305 = stablehlo.reduce(%v303 init: %v304) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v306 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v307 = stablehlo.multiply %v305, %v306 : tensor<32xf32>
    %v308 = stablehlo.subtract %b6, %v307 : tensor<32xf32>
    %v309 = stablehlo.reshape %v65 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v310 = stablehlo.reshape %v137 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v311 = stablehlo.transpose %v309, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v312 = stablehlo.transpose %v310, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v313 = stablehlo.convolution(%v311, %v312)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v314 = stablehlo.transpose %v313, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v315 = stablehlo.constant dense<0.00078125> : tensor<32x32x3x3xf32>
    %v316 = stablehlo.multiply %v314, %v315 : tensor<32x32x3x3xf32>
    %v317 = stablehlo.subtract %W7, %v316 : tensor<32x32x3x3xf32>
    %v318 = stablehlo.reshape %v137 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v319 = stablehlo.constant dense<0.0> : tensor<f32>
    %v320 = stablehlo.reduce(%v318 init: %v319) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v321 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v322 = stablehlo.multiply %v320, %v321 : tensor<32xf32>
    %v323 = stablehlo.subtract %b7, %v322 : tensor<32xf32>
    %v324 = stablehlo.reshape %v74 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v325 = stablehlo.reshape %v126 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v326 = stablehlo.transpose %v324, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v327 = stablehlo.transpose %v325, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v328 = stablehlo.convolution(%v326, %v327)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v329 = stablehlo.transpose %v328, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v330 = stablehlo.constant dense<0.00078125> : tensor<32x32x3x3xf32>
    %v331 = stablehlo.multiply %v329, %v330 : tensor<32x32x3x3xf32>
    %v332 = stablehlo.subtract %W8, %v331 : tensor<32x32x3x3xf32>
    %v333 = stablehlo.reshape %v126 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v334 = stablehlo.constant dense<0.0> : tensor<f32>
    %v335 = stablehlo.reduce(%v333 init: %v334) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v336 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v337 = stablehlo.multiply %v335, %v336 : tensor<32xf32>
    %v338 = stablehlo.subtract %b8, %v337 : tensor<32xf32>
    %v339 = stablehlo.dot_general %v87, %v114, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %v340 = stablehlo.constant dense<0.00078125> : tensor<128x64xf32>
    %v341 = stablehlo.multiply %v339, %v340 : tensor<128x64xf32>
    %v342 = stablehlo.subtract %W9, %v341 : tensor<128x64xf32>
    %v343 = stablehlo.constant dense<0.0> : tensor<f32>
    %v344 = stablehlo.reduce(%v114 init: %v343) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v345 = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %v346 = stablehlo.multiply %v344, %v345 : tensor<64xf32>
    %v347 = stablehlo.subtract %b9, %v346 : tensor<64xf32>
    %v348 = stablehlo.dot_general %v92, %v110, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<64x64xf32>
    %v349 = stablehlo.constant dense<0.00078125> : tensor<64x64xf32>
    %v350 = stablehlo.multiply %v348, %v349 : tensor<64x64xf32>
    %v351 = stablehlo.subtract %Wa, %v350 : tensor<64x64xf32>
    %v352 = stablehlo.constant dense<0.0> : tensor<f32>
    %v353 = stablehlo.reduce(%v110 init: %v352) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v354 = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %v355 = stablehlo.multiply %v353, %v354 : tensor<64xf32>
    %v356 = stablehlo.subtract %ba, %v355 : tensor<64xf32>
    %v357 = stablehlo.dot_general %v97, %v106, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x10xf32>) -> tensor<64x10xf32>
    %v358 = stablehlo.constant dense<0.00078125> : tensor<64x10xf32>
    %v359 = stablehlo.multiply %v357, %v358 : tensor<64x10xf32>
    %v360 = stablehlo.subtract %Wb, %v359 : tensor<64x10xf32>
    %v361 = stablehlo.constant dense<0.0> : tensor<f32>
    %v362 = stablehlo.reduce(%v106 init: %v361) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v363 = stablehlo.constant dense<0.00078125> : tensor<10xf32>
    %v364 = stablehlo.multiply %v362, %v363 : tensor<10xf32>
    %v365 = stablehlo.subtract %bb, %v364 : tensor<10xf32>
    return %v227, %v233, %v242, %v248, %v257, %v263, %v272, %v278, %v287, %v293, %v302, %v308, %v317, %v323, %v332, %v338, %v342, %v347, %v351, %v356, %v360, %v365 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>
  }
}
