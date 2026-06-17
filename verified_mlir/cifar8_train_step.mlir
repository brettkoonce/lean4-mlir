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
    %v85 = stablehlo.exponential %v84 : tensor<128x10xf32>
    %v86 = stablehlo.constant dense<0.0> : tensor<f32>
    %v87 = stablehlo.reduce(%v85 init: %v86) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %v88 = stablehlo.broadcast_in_dim %v87, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %v89 = stablehlo.divide %v85, %v88 : tensor<128x10xf32>
    %v90 = stablehlo.subtract %v89, %onehot : tensor<128x10xf32>
    %v91 = stablehlo.dot_general %v90, %Wb, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x10xf32>, tensor<64x10xf32>) -> tensor<128x64xf32>
    %v92 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v93 = stablehlo.compare GT, %v79, %v92 : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xi1>
    %v94 = stablehlo.select %v93, %v91, %v92 : tensor<128x64xi1>, tensor<128x64xf32>
    %v95 = stablehlo.dot_general %v94, %Wa, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
    %v96 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v97 = stablehlo.compare GT, %v74, %v96 : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xi1>
    %v98 = stablehlo.select %v97, %v95, %v96 : tensor<128x64xi1>, tensor<128x64xf32>
    %v99 = stablehlo.dot_general %v98, %W9, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x128xf32>
    %v100 = stablehlo.reshape %v67 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v101 = stablehlo.reshape %v99 : (tensor<128x128xf32>) -> tensor<128x32x2x2xf32>
    %v102 = stablehlo.constant dense<0.0> : tensor<f32>
    %v103 = "stablehlo.select_and_scatter"(%v100, %v101, %v102) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x4x4xf32>, tensor<128x32x2x2xf32>, tensor<f32>) -> tensor<128x32x4x4xf32>
    %v104 = stablehlo.reshape %v103 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v105 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v106 = stablehlo.compare GT, %v65, %v105 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v107 = stablehlo.select %v106, %v104, %v105 : tensor<128x512xi1>, tensor<128x512xf32>
    %v108 = stablehlo.reshape %v107 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v109 = stablehlo.transpose %W8, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v110 = stablehlo.reverse %v109, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v111 = stablehlo.convolution(%v108, %v110)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v112 = stablehlo.reshape %v111 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v113 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v114 = stablehlo.compare GT, %v58, %v113 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v115 = stablehlo.select %v114, %v112, %v113 : tensor<128x512xi1>, tensor<128x512xf32>
    %v116 = stablehlo.reshape %v115 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v117 = stablehlo.transpose %W7, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v118 = stablehlo.reverse %v117, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v119 = stablehlo.convolution(%v116, %v118)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v120 = stablehlo.reshape %v119 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v121 = stablehlo.reshape %v49 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v122 = stablehlo.reshape %v120 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v123 = stablehlo.constant dense<0.0> : tensor<f32>
    %v124 = "stablehlo.select_and_scatter"(%v121, %v122, %v123) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x8x8xf32>, tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32x8x8xf32>
    %v125 = stablehlo.reshape %v124 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v126 = stablehlo.constant dense<0.0> : tensor<128x2048xf32>
    %v127 = stablehlo.compare GT, %v47, %v126 : (tensor<128x2048xf32>, tensor<128x2048xf32>) -> tensor<128x2048xi1>
    %v128 = stablehlo.select %v127, %v125, %v126 : tensor<128x2048xi1>, tensor<128x2048xf32>
    %v129 = stablehlo.reshape %v128 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v130 = stablehlo.transpose %W6, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v131 = stablehlo.reverse %v130, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v132 = stablehlo.convolution(%v129, %v131)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x8x8xf32>
    %v133 = stablehlo.reshape %v132 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v134 = stablehlo.constant dense<0.0> : tensor<128x2048xf32>
    %v135 = stablehlo.compare GT, %v40, %v134 : (tensor<128x2048xf32>, tensor<128x2048xf32>) -> tensor<128x2048xi1>
    %v136 = stablehlo.select %v135, %v133, %v134 : tensor<128x2048xi1>, tensor<128x2048xf32>
    %v137 = stablehlo.reshape %v136 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v138 = stablehlo.transpose %W5, dims = [1, 0, 2, 3] : (tensor<32x16x3x3xf32>) -> tensor<16x32x3x3xf32>
    %v139 = stablehlo.reverse %v138, dims = [2, 3] : tensor<16x32x3x3xf32>
    %v140 = stablehlo.convolution(%v137, %v139)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<16x32x3x3xf32>) -> tensor<128x16x8x8xf32>
    %v141 = stablehlo.reshape %v140 : (tensor<128x16x8x8xf32>) -> tensor<128x1024xf32>
    %v142 = stablehlo.reshape %v31 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v143 = stablehlo.reshape %v141 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v144 = stablehlo.constant dense<0.0> : tensor<f32>
    %v145 = "stablehlo.select_and_scatter"(%v142, %v143, %v144) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x16x16xf32>, tensor<128x16x8x8xf32>, tensor<f32>) -> tensor<128x16x16x16xf32>
    %v146 = stablehlo.reshape %v145 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v147 = stablehlo.constant dense<0.0> : tensor<128x4096xf32>
    %v148 = stablehlo.compare GT, %v29, %v147 : (tensor<128x4096xf32>, tensor<128x4096xf32>) -> tensor<128x4096xi1>
    %v149 = stablehlo.select %v148, %v146, %v147 : tensor<128x4096xi1>, tensor<128x4096xf32>
    %v150 = stablehlo.reshape %v149 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v151 = stablehlo.transpose %W4, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v152 = stablehlo.reverse %v151, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v153 = stablehlo.convolution(%v150, %v152)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v154 = stablehlo.reshape %v153 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v155 = stablehlo.constant dense<0.0> : tensor<128x4096xf32>
    %v156 = stablehlo.compare GT, %v22, %v155 : (tensor<128x4096xf32>, tensor<128x4096xf32>) -> tensor<128x4096xi1>
    %v157 = stablehlo.select %v156, %v154, %v155 : tensor<128x4096xi1>, tensor<128x4096xf32>
    %v158 = stablehlo.reshape %v157 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v159 = stablehlo.transpose %W3, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v160 = stablehlo.reverse %v159, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v161 = stablehlo.convolution(%v158, %v160)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v162 = stablehlo.reshape %v161 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v163 = stablehlo.reshape %v13 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v164 = stablehlo.reshape %v162 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v165 = stablehlo.constant dense<0.0> : tensor<f32>
    %v166 = "stablehlo.select_and_scatter"(%v163, %v164, %v165) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x32x32xf32>, tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x32x32xf32>
    %v167 = stablehlo.reshape %v166 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v168 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v169 = stablehlo.compare GT, %v11, %v168 : (tensor<128x16384xf32>, tensor<128x16384xf32>) -> tensor<128x16384xi1>
    %v170 = stablehlo.select %v169, %v167, %v168 : tensor<128x16384xi1>, tensor<128x16384xf32>
    %v171 = stablehlo.reshape %v170 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v172 = stablehlo.transpose %W2, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v173 = stablehlo.reverse %v172, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v174 = stablehlo.convolution(%v171, %v173)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x32x32xf32>
    %v175 = stablehlo.reshape %v174 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v176 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v177 = stablehlo.compare GT, %v4, %v176 : (tensor<128x16384xf32>, tensor<128x16384xf32>) -> tensor<128x16384xi1>
    %v178 = stablehlo.select %v177, %v175, %v176 : tensor<128x16384xi1>, tensor<128x16384xf32>
    %v179 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v180 = stablehlo.reshape %v178 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v181 = stablehlo.transpose %v179, dims = [1, 0, 2, 3] : (tensor<128x3x32x32xf32>) -> tensor<3x128x32x32xf32>
    %v182 = stablehlo.transpose %v180, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v183 = stablehlo.convolution(%v181, %v182)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<3x16x3x3xf32>
    %v184 = stablehlo.transpose %v183, dims = [1, 0, 2, 3] : (tensor<3x16x3x3xf32>) -> tensor<16x3x3x3xf32>
    %v185 = stablehlo.constant dense<0.00078125> : tensor<16x3x3x3xf32>
    %v186 = stablehlo.multiply %v184, %v185 : tensor<16x3x3x3xf32>
    %v187 = stablehlo.subtract %W1, %v186 : tensor<16x3x3x3xf32>
    %v188 = stablehlo.reshape %v178 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v189 = stablehlo.constant dense<0.0> : tensor<f32>
    %v190 = stablehlo.reduce(%v188 init: %v189) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v191 = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %v192 = stablehlo.multiply %v190, %v191 : tensor<16xf32>
    %v193 = stablehlo.subtract %b1, %v192 : tensor<16xf32>
    %v194 = stablehlo.reshape %v6 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v195 = stablehlo.reshape %v170 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v196 = stablehlo.transpose %v194, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v197 = stablehlo.transpose %v195, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v198 = stablehlo.convolution(%v196, %v197)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x16x3x3xf32>
    %v199 = stablehlo.transpose %v198, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v200 = stablehlo.constant dense<0.00078125> : tensor<16x16x3x3xf32>
    %v201 = stablehlo.multiply %v199, %v200 : tensor<16x16x3x3xf32>
    %v202 = stablehlo.subtract %W2, %v201 : tensor<16x16x3x3xf32>
    %v203 = stablehlo.reshape %v170 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v204 = stablehlo.constant dense<0.0> : tensor<f32>
    %v205 = stablehlo.reduce(%v203 init: %v204) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v206 = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %v207 = stablehlo.multiply %v205, %v206 : tensor<16xf32>
    %v208 = stablehlo.subtract %b2, %v207 : tensor<16xf32>
    %v209 = stablehlo.reshape %v17 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v210 = stablehlo.reshape %v157 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v211 = stablehlo.transpose %v209, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v212 = stablehlo.transpose %v210, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v213 = stablehlo.convolution(%v211, %v212)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v214 = stablehlo.transpose %v213, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v215 = stablehlo.constant dense<0.00078125> : tensor<16x16x3x3xf32>
    %v216 = stablehlo.multiply %v214, %v215 : tensor<16x16x3x3xf32>
    %v217 = stablehlo.subtract %W3, %v216 : tensor<16x16x3x3xf32>
    %v218 = stablehlo.reshape %v157 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v219 = stablehlo.constant dense<0.0> : tensor<f32>
    %v220 = stablehlo.reduce(%v218 init: %v219) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v221 = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %v222 = stablehlo.multiply %v220, %v221 : tensor<16xf32>
    %v223 = stablehlo.subtract %b3, %v222 : tensor<16xf32>
    %v224 = stablehlo.reshape %v24 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v225 = stablehlo.reshape %v149 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v226 = stablehlo.transpose %v224, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v227 = stablehlo.transpose %v225, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v228 = stablehlo.convolution(%v226, %v227)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v229 = stablehlo.transpose %v228, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v230 = stablehlo.constant dense<0.00078125> : tensor<16x16x3x3xf32>
    %v231 = stablehlo.multiply %v229, %v230 : tensor<16x16x3x3xf32>
    %v232 = stablehlo.subtract %W4, %v231 : tensor<16x16x3x3xf32>
    %v233 = stablehlo.reshape %v149 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v234 = stablehlo.constant dense<0.0> : tensor<f32>
    %v235 = stablehlo.reduce(%v233 init: %v234) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v236 = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %v237 = stablehlo.multiply %v235, %v236 : tensor<16xf32>
    %v238 = stablehlo.subtract %b4, %v237 : tensor<16xf32>
    %v239 = stablehlo.reshape %v35 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v240 = stablehlo.reshape %v136 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v241 = stablehlo.transpose %v239, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v242 = stablehlo.transpose %v240, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v243 = stablehlo.convolution(%v241, %v242)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<16x32x3x3xf32>
    %v244 = stablehlo.transpose %v243, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v245 = stablehlo.constant dense<0.00078125> : tensor<32x16x3x3xf32>
    %v246 = stablehlo.multiply %v244, %v245 : tensor<32x16x3x3xf32>
    %v247 = stablehlo.subtract %W5, %v246 : tensor<32x16x3x3xf32>
    %v248 = stablehlo.reshape %v136 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v249 = stablehlo.constant dense<0.0> : tensor<f32>
    %v250 = stablehlo.reduce(%v248 init: %v249) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v251 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v252 = stablehlo.multiply %v250, %v251 : tensor<32xf32>
    %v253 = stablehlo.subtract %b5, %v252 : tensor<32xf32>
    %v254 = stablehlo.reshape %v42 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v255 = stablehlo.reshape %v128 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v256 = stablehlo.transpose %v254, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v257 = stablehlo.transpose %v255, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v258 = stablehlo.convolution(%v256, %v257)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<32x32x3x3xf32>
    %v259 = stablehlo.transpose %v258, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v260 = stablehlo.constant dense<0.00078125> : tensor<32x32x3x3xf32>
    %v261 = stablehlo.multiply %v259, %v260 : tensor<32x32x3x3xf32>
    %v262 = stablehlo.subtract %W6, %v261 : tensor<32x32x3x3xf32>
    %v263 = stablehlo.reshape %v128 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v264 = stablehlo.constant dense<0.0> : tensor<f32>
    %v265 = stablehlo.reduce(%v263 init: %v264) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v266 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v267 = stablehlo.multiply %v265, %v266 : tensor<32xf32>
    %v268 = stablehlo.subtract %b6, %v267 : tensor<32xf32>
    %v269 = stablehlo.reshape %v53 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v270 = stablehlo.reshape %v115 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v271 = stablehlo.transpose %v269, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v272 = stablehlo.transpose %v270, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v273 = stablehlo.convolution(%v271, %v272)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v274 = stablehlo.transpose %v273, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v275 = stablehlo.constant dense<0.00078125> : tensor<32x32x3x3xf32>
    %v276 = stablehlo.multiply %v274, %v275 : tensor<32x32x3x3xf32>
    %v277 = stablehlo.subtract %W7, %v276 : tensor<32x32x3x3xf32>
    %v278 = stablehlo.reshape %v115 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v279 = stablehlo.constant dense<0.0> : tensor<f32>
    %v280 = stablehlo.reduce(%v278 init: %v279) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v281 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v282 = stablehlo.multiply %v280, %v281 : tensor<32xf32>
    %v283 = stablehlo.subtract %b7, %v282 : tensor<32xf32>
    %v284 = stablehlo.reshape %v60 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v285 = stablehlo.reshape %v107 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v286 = stablehlo.transpose %v284, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v287 = stablehlo.transpose %v285, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v288 = stablehlo.convolution(%v286, %v287)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v289 = stablehlo.transpose %v288, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v290 = stablehlo.constant dense<0.00078125> : tensor<32x32x3x3xf32>
    %v291 = stablehlo.multiply %v289, %v290 : tensor<32x32x3x3xf32>
    %v292 = stablehlo.subtract %W8, %v291 : tensor<32x32x3x3xf32>
    %v293 = stablehlo.reshape %v107 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v294 = stablehlo.constant dense<0.0> : tensor<f32>
    %v295 = stablehlo.reduce(%v293 init: %v294) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v296 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v297 = stablehlo.multiply %v295, %v296 : tensor<32xf32>
    %v298 = stablehlo.subtract %b8, %v297 : tensor<32xf32>
    %v299 = stablehlo.dot_general %v71, %v98, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %v300 = stablehlo.constant dense<0.00078125> : tensor<128x64xf32>
    %v301 = stablehlo.multiply %v299, %v300 : tensor<128x64xf32>
    %v302 = stablehlo.subtract %W9, %v301 : tensor<128x64xf32>
    %v303 = stablehlo.constant dense<0.0> : tensor<f32>
    %v304 = stablehlo.reduce(%v98 init: %v303) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v305 = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %v306 = stablehlo.multiply %v304, %v305 : tensor<64xf32>
    %v307 = stablehlo.subtract %b9, %v306 : tensor<64xf32>
    %v308 = stablehlo.dot_general %v76, %v94, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<64x64xf32>
    %v309 = stablehlo.constant dense<0.00078125> : tensor<64x64xf32>
    %v310 = stablehlo.multiply %v308, %v309 : tensor<64x64xf32>
    %v311 = stablehlo.subtract %Wa, %v310 : tensor<64x64xf32>
    %v312 = stablehlo.constant dense<0.0> : tensor<f32>
    %v313 = stablehlo.reduce(%v94 init: %v312) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v314 = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %v315 = stablehlo.multiply %v313, %v314 : tensor<64xf32>
    %v316 = stablehlo.subtract %ba, %v315 : tensor<64xf32>
    %v317 = stablehlo.dot_general %v81, %v90, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x10xf32>) -> tensor<64x10xf32>
    %v318 = stablehlo.constant dense<0.00078125> : tensor<64x10xf32>
    %v319 = stablehlo.multiply %v317, %v318 : tensor<64x10xf32>
    %v320 = stablehlo.subtract %Wb, %v319 : tensor<64x10xf32>
    %v321 = stablehlo.constant dense<0.0> : tensor<f32>
    %v322 = stablehlo.reduce(%v90 init: %v321) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v323 = stablehlo.constant dense<0.00078125> : tensor<10xf32>
    %v324 = stablehlo.multiply %v322, %v323 : tensor<10xf32>
    %v325 = stablehlo.subtract %bb, %v324 : tensor<10xf32>
    return %v187, %v193, %v202, %v208, %v217, %v223, %v232, %v238, %v247, %v253, %v262, %v268, %v277, %v283, %v292, %v298, %v302, %v307, %v311, %v316, %v320, %v325 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>
  }
}
