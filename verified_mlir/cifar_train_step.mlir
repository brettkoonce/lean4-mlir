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
    %v5 = stablehlo.constant dense<0.0> : tensor<128x32768xf32>
    %v6 = stablehlo.maximum %v4, %v5 : tensor<128x32768xf32>
    %v7 = stablehlo.reshape %v6 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v8 = stablehlo.convolution(%v7, %W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x32x32xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x32x32xf32>
    %v9 = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<32xf32>) -> tensor<128x32x32x32xf32>
    %v10 = stablehlo.add %v8, %v9 : tensor<128x32x32x32xf32>
    %v11 = stablehlo.reshape %v10 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %v12 = stablehlo.constant dense<0.0> : tensor<128x32768xf32>
    %v13 = stablehlo.maximum %v11, %v12 : tensor<128x32768xf32>
    %v14 = stablehlo.reshape %v13 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v15 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v16 = "stablehlo.reduce_window"(%v14, %v15) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<128x32x16x16xf32>
    %v17 = stablehlo.reshape %v16 : (tensor<128x32x16x16xf32>) -> tensor<128x8192xf32>
    %v18 = stablehlo.reshape %v17 : (tensor<128x8192xf32>) -> tensor<128x32x16x16xf32>
    %v19 = stablehlo.convolution(%v18, %W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x16x16xf32>, tensor<64x32x3x3xf32>) -> tensor<128x64x16x16xf32>
    %v20 = stablehlo.broadcast_in_dim %b3, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %v21 = stablehlo.add %v19, %v20 : tensor<128x64x16x16xf32>
    %v22 = stablehlo.reshape %v21 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v23 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v24 = stablehlo.maximum %v22, %v23 : tensor<128x16384xf32>
    %v25 = stablehlo.reshape %v24 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v26 = stablehlo.convolution(%v25, %W4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x16x16xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x16x16xf32>
    %v27 = stablehlo.broadcast_in_dim %b4, dims = [1] : (tensor<64xf32>) -> tensor<128x64x16x16xf32>
    %v28 = stablehlo.add %v26, %v27 : tensor<128x64x16x16xf32>
    %v29 = stablehlo.reshape %v28 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v30 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v31 = stablehlo.maximum %v29, %v30 : tensor<128x16384xf32>
    %v32 = stablehlo.reshape %v31 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v33 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v34 = "stablehlo.reduce_window"(%v32, %v33) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<128x64x8x8xf32>
    %v35 = stablehlo.reshape %v34 : (tensor<128x64x8x8xf32>) -> tensor<128x4096xf32>
    %v36 = stablehlo.dot_general %v35, %W5, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x4096xf32>, tensor<4096x512xf32>) -> tensor<128x512xf32>
    %v37 = stablehlo.broadcast_in_dim %b5, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v38 = stablehlo.add %v36, %v37 : tensor<128x512xf32>
    %v39 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v40 = stablehlo.maximum %v38, %v39 : tensor<128x512xf32>
    %v41 = stablehlo.dot_general %v40, %W6, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %v42 = stablehlo.broadcast_in_dim %b6, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v43 = stablehlo.add %v41, %v42 : tensor<128x512xf32>
    %v44 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v45 = stablehlo.maximum %v43, %v44 : tensor<128x512xf32>
    %v46 = stablehlo.dot_general %v45, %W7, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x10xf32>) -> tensor<128x10xf32>
    %v47 = stablehlo.broadcast_in_dim %b7, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v48 = stablehlo.add %v46, %v47 : tensor<128x10xf32>
    %v49 = stablehlo.exponential %v48 : tensor<128x10xf32>
    %v50 = stablehlo.constant dense<0.0> : tensor<f32>
    %v51 = stablehlo.reduce(%v49 init: %v50) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %v52 = stablehlo.broadcast_in_dim %v51, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %v53 = stablehlo.divide %v49, %v52 : tensor<128x10xf32>
    %v54 = stablehlo.subtract %v53, %onehot : tensor<128x10xf32>
    %v55 = stablehlo.dot_general %v54, %W7, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x10xf32>, tensor<512x10xf32>) -> tensor<128x512xf32>
    %v56 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v57 = stablehlo.compare GT, %v43, %v56 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v58 = stablehlo.select %v57, %v55, %v56 : tensor<128x512xi1>, tensor<128x512xf32>
    %v59 = stablehlo.dot_general %v58, %W6, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %v60 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v61 = stablehlo.compare GT, %v38, %v60 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v62 = stablehlo.select %v61, %v59, %v60 : tensor<128x512xi1>, tensor<128x512xf32>
    %v63 = stablehlo.dot_general %v62, %W5, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<4096x512xf32>) -> tensor<128x4096xf32>
    %v64 = stablehlo.reshape %v31 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v65 = stablehlo.reshape %v63 : (tensor<128x4096xf32>) -> tensor<128x64x8x8xf32>
    %v66 = stablehlo.constant dense<0.0> : tensor<f32>
    %v67 = "stablehlo.select_and_scatter"(%v64, %v65, %v66) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x64x16x16xf32>, tensor<128x64x8x8xf32>, tensor<f32>) -> tensor<128x64x16x16xf32>
    %v68 = stablehlo.reshape %v67 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v69 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v70 = stablehlo.compare GT, %v29, %v69 : (tensor<128x16384xf32>, tensor<128x16384xf32>) -> tensor<128x16384xi1>
    %v71 = stablehlo.select %v70, %v68, %v69 : tensor<128x16384xi1>, tensor<128x16384xf32>
    %v72 = stablehlo.reshape %v71 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v73 = stablehlo.transpose %W4, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v74 = stablehlo.reverse %v73, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v75 = stablehlo.convolution(%v72, %v74)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x16x16xf32>, tensor<64x64x3x3xf32>) -> tensor<128x64x16x16xf32>
    %v76 = stablehlo.reshape %v75 : (tensor<128x64x16x16xf32>) -> tensor<128x16384xf32>
    %v77 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v78 = stablehlo.compare GT, %v22, %v77 : (tensor<128x16384xf32>, tensor<128x16384xf32>) -> tensor<128x16384xi1>
    %v79 = stablehlo.select %v78, %v76, %v77 : tensor<128x16384xi1>, tensor<128x16384xf32>
    %v80 = stablehlo.reshape %v79 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v81 = stablehlo.transpose %W3, dims = [1, 0, 2, 3] : (tensor<64x32x3x3xf32>) -> tensor<32x64x3x3xf32>
    %v82 = stablehlo.reverse %v81, dims = [2, 3] : tensor<32x64x3x3xf32>
    %v83 = stablehlo.convolution(%v80, %v82)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x16x16xf32>, tensor<32x64x3x3xf32>) -> tensor<128x32x16x16xf32>
    %v84 = stablehlo.reshape %v83 : (tensor<128x32x16x16xf32>) -> tensor<128x8192xf32>
    %v85 = stablehlo.reshape %v13 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v86 = stablehlo.reshape %v84 : (tensor<128x8192xf32>) -> tensor<128x32x16x16xf32>
    %v87 = stablehlo.constant dense<0.0> : tensor<f32>
    %v88 = "stablehlo.select_and_scatter"(%v85, %v86, %v87) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x32x32xf32>, tensor<128x32x16x16xf32>, tensor<f32>) -> tensor<128x32x32x32xf32>
    %v89 = stablehlo.reshape %v88 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %v90 = stablehlo.constant dense<0.0> : tensor<128x32768xf32>
    %v91 = stablehlo.compare GT, %v11, %v90 : (tensor<128x32768xf32>, tensor<128x32768xf32>) -> tensor<128x32768xi1>
    %v92 = stablehlo.select %v91, %v89, %v90 : tensor<128x32768xi1>, tensor<128x32768xf32>
    %v93 = stablehlo.reshape %v92 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v94 = stablehlo.transpose %W2, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v95 = stablehlo.reverse %v94, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v96 = stablehlo.convolution(%v93, %v95)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x32x32xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x32x32xf32>
    %v97 = stablehlo.reshape %v96 : (tensor<128x32x32x32xf32>) -> tensor<128x32768xf32>
    %v98 = stablehlo.constant dense<0.0> : tensor<128x32768xf32>
    %v99 = stablehlo.compare GT, %v4, %v98 : (tensor<128x32768xf32>, tensor<128x32768xf32>) -> tensor<128x32768xi1>
    %v100 = stablehlo.select %v99, %v97, %v98 : tensor<128x32768xi1>, tensor<128x32768xf32>
    %v101 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v102 = stablehlo.reshape %v100 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v103 = stablehlo.transpose %v101, dims = [1, 0, 2, 3] : (tensor<128x3x32x32xf32>) -> tensor<3x128x32x32xf32>
    %v104 = stablehlo.transpose %v102, dims = [1, 0, 2, 3] : (tensor<128x32x32x32xf32>) -> tensor<32x128x32x32xf32>
    %v105 = stablehlo.convolution(%v103, %v104)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x128x32x32xf32>, tensor<32x128x32x32xf32>) -> tensor<3x32x3x3xf32>
    %v106 = stablehlo.transpose %v105, dims = [1, 0, 2, 3] : (tensor<3x32x3x3xf32>) -> tensor<32x3x3x3xf32>
    %v107 = stablehlo.constant dense<0.00078125> : tensor<32x3x3x3xf32>
    %v108 = stablehlo.multiply %v106, %v107 : tensor<32x3x3x3xf32>
    %v109 = stablehlo.subtract %W1, %v108 : tensor<32x3x3x3xf32>
    %v110 = stablehlo.reshape %v100 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v111 = stablehlo.constant dense<0.0> : tensor<f32>
    %v112 = stablehlo.reduce(%v110 init: %v111) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<32xf32>
    %v113 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v114 = stablehlo.multiply %v112, %v113 : tensor<32xf32>
    %v115 = stablehlo.subtract %b1, %v114 : tensor<32xf32>
    %v116 = stablehlo.reshape %v6 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v117 = stablehlo.reshape %v92 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v118 = stablehlo.transpose %v116, dims = [1, 0, 2, 3] : (tensor<128x32x32x32xf32>) -> tensor<32x128x32x32xf32>
    %v119 = stablehlo.transpose %v117, dims = [1, 0, 2, 3] : (tensor<128x32x32x32xf32>) -> tensor<32x128x32x32xf32>
    %v120 = stablehlo.convolution(%v118, %v119)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x32x32xf32>, tensor<32x128x32x32xf32>) -> tensor<32x32x3x3xf32>
    %v121 = stablehlo.transpose %v120, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v122 = stablehlo.constant dense<0.00078125> : tensor<32x32x3x3xf32>
    %v123 = stablehlo.multiply %v121, %v122 : tensor<32x32x3x3xf32>
    %v124 = stablehlo.subtract %W2, %v123 : tensor<32x32x3x3xf32>
    %v125 = stablehlo.reshape %v92 : (tensor<128x32768xf32>) -> tensor<128x32x32x32xf32>
    %v126 = stablehlo.constant dense<0.0> : tensor<f32>
    %v127 = stablehlo.reduce(%v125 init: %v126) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x32x32xf32>, tensor<f32>) -> tensor<32xf32>
    %v128 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v129 = stablehlo.multiply %v127, %v128 : tensor<32xf32>
    %v130 = stablehlo.subtract %b2, %v129 : tensor<32xf32>
    %v131 = stablehlo.reshape %v17 : (tensor<128x8192xf32>) -> tensor<128x32x16x16xf32>
    %v132 = stablehlo.reshape %v79 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v133 = stablehlo.transpose %v131, dims = [1, 0, 2, 3] : (tensor<128x32x16x16xf32>) -> tensor<32x128x16x16xf32>
    %v134 = stablehlo.transpose %v132, dims = [1, 0, 2, 3] : (tensor<128x64x16x16xf32>) -> tensor<64x128x16x16xf32>
    %v135 = stablehlo.convolution(%v133, %v134)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x16x16xf32>, tensor<64x128x16x16xf32>) -> tensor<32x64x3x3xf32>
    %v136 = stablehlo.transpose %v135, dims = [1, 0, 2, 3] : (tensor<32x64x3x3xf32>) -> tensor<64x32x3x3xf32>
    %v137 = stablehlo.constant dense<0.00078125> : tensor<64x32x3x3xf32>
    %v138 = stablehlo.multiply %v136, %v137 : tensor<64x32x3x3xf32>
    %v139 = stablehlo.subtract %W3, %v138 : tensor<64x32x3x3xf32>
    %v140 = stablehlo.reshape %v79 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v141 = stablehlo.constant dense<0.0> : tensor<f32>
    %v142 = stablehlo.reduce(%v140 init: %v141) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<64xf32>
    %v143 = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %v144 = stablehlo.multiply %v142, %v143 : tensor<64xf32>
    %v145 = stablehlo.subtract %b3, %v144 : tensor<64xf32>
    %v146 = stablehlo.reshape %v24 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v147 = stablehlo.reshape %v71 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v148 = stablehlo.transpose %v146, dims = [1, 0, 2, 3] : (tensor<128x64x16x16xf32>) -> tensor<64x128x16x16xf32>
    %v149 = stablehlo.transpose %v147, dims = [1, 0, 2, 3] : (tensor<128x64x16x16xf32>) -> tensor<64x128x16x16xf32>
    %v150 = stablehlo.convolution(%v148, %v149)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x16x16xf32>, tensor<64x128x16x16xf32>) -> tensor<64x64x3x3xf32>
    %v151 = stablehlo.transpose %v150, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v152 = stablehlo.constant dense<0.00078125> : tensor<64x64x3x3xf32>
    %v153 = stablehlo.multiply %v151, %v152 : tensor<64x64x3x3xf32>
    %v154 = stablehlo.subtract %W4, %v153 : tensor<64x64x3x3xf32>
    %v155 = stablehlo.reshape %v71 : (tensor<128x16384xf32>) -> tensor<128x64x16x16xf32>
    %v156 = stablehlo.constant dense<0.0> : tensor<f32>
    %v157 = stablehlo.reduce(%v155 init: %v156) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x64x16x16xf32>, tensor<f32>) -> tensor<64xf32>
    %v158 = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %v159 = stablehlo.multiply %v157, %v158 : tensor<64xf32>
    %v160 = stablehlo.subtract %b4, %v159 : tensor<64xf32>
    %v161 = stablehlo.dot_general %v35, %v62, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x4096xf32>, tensor<128x512xf32>) -> tensor<4096x512xf32>
    %v162 = stablehlo.constant dense<0.00078125> : tensor<4096x512xf32>
    %v163 = stablehlo.multiply %v161, %v162 : tensor<4096x512xf32>
    %v164 = stablehlo.subtract %W5, %v163 : tensor<4096x512xf32>
    %v165 = stablehlo.constant dense<0.0> : tensor<f32>
    %v166 = stablehlo.reduce(%v62 init: %v165) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v167 = stablehlo.constant dense<0.00078125> : tensor<512xf32>
    %v168 = stablehlo.multiply %v166, %v167 : tensor<512xf32>
    %v169 = stablehlo.subtract %b5, %v168 : tensor<512xf32>
    %v170 = stablehlo.dot_general %v40, %v58, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<512x512xf32>
    %v171 = stablehlo.constant dense<0.00078125> : tensor<512x512xf32>
    %v172 = stablehlo.multiply %v170, %v171 : tensor<512x512xf32>
    %v173 = stablehlo.subtract %W6, %v172 : tensor<512x512xf32>
    %v174 = stablehlo.constant dense<0.0> : tensor<f32>
    %v175 = stablehlo.reduce(%v58 init: %v174) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v176 = stablehlo.constant dense<0.00078125> : tensor<512xf32>
    %v177 = stablehlo.multiply %v175, %v176 : tensor<512xf32>
    %v178 = stablehlo.subtract %b6, %v177 : tensor<512xf32>
    %v179 = stablehlo.dot_general %v45, %v54, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x10xf32>) -> tensor<512x10xf32>
    %v180 = stablehlo.constant dense<0.00078125> : tensor<512x10xf32>
    %v181 = stablehlo.multiply %v179, %v180 : tensor<512x10xf32>
    %v182 = stablehlo.subtract %W7, %v181 : tensor<512x10xf32>
    %v183 = stablehlo.constant dense<0.0> : tensor<f32>
    %v184 = stablehlo.reduce(%v54 init: %v183) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v185 = stablehlo.constant dense<0.00078125> : tensor<10xf32>
    %v186 = stablehlo.multiply %v184, %v185 : tensor<10xf32>
    %v187 = stablehlo.subtract %b7, %v186 : tensor<10xf32>
    return %v109, %v115, %v124, %v130, %v139, %v145, %v154, %v160, %v164, %v169, %v173, %v178, %v182, %v187 : tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<64x32x3x3xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<4096x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>
  }
}
