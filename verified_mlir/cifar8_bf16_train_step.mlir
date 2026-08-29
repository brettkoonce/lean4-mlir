module @m {
  func.func @cifar8_bf16_train_step(%x: tensor<128x3072xf32>, %W1: tensor<16x3x3x3xf32>, %b1: tensor<16xf32>, %W2: tensor<16x16x3x3xf32>, %b2: tensor<16xf32>, %W3: tensor<16x16x3x3xf32>, %b3: tensor<16xf32>, %W4: tensor<16x16x3x3xf32>, %b4: tensor<16xf32>, %W5: tensor<32x16x3x3xf32>, %b5: tensor<32xf32>, %W6: tensor<32x32x3x3xf32>, %b6: tensor<32xf32>, %W7: tensor<32x32x3x3xf32>, %b7: tensor<32xf32>, %W8: tensor<32x32x3x3xf32>, %b8: tensor<32xf32>, %W9: tensor<128x64xf32>, %b9: tensor<64xf32>, %Wa: tensor<64x64xf32>, %ba: tensor<64xf32>, %Wb: tensor<64x10xf32>, %bb: tensor<10xf32>, %onehot: tensor<128x10xf32>) -> (tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>) {
    // ── cifar8 train step: every line is pretty(verified AST node) ──
    %v0 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v1 = stablehlo.convert %v0 : (tensor<128x3x32x32xf32>) -> tensor<128x3x32x32xbf16>
    %v2 = stablehlo.convert %W1 : (tensor<16x3x3x3xf32>) -> tensor<16x3x3x3xbf16>
    %v3 = stablehlo.convolution(%v1, %v2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x3x32x32xbf16>, tensor<16x3x3x3xbf16>) -> tensor<128x16x32x32xbf16>
    %v4 = stablehlo.convert %v3 : (tensor<128x16x32x32xbf16>) -> tensor<128x16x32x32xf32>
    %v5 = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v6 = stablehlo.add %v4, %v5 : tensor<128x16x32x32xf32>
    %v7 = stablehlo.reshape %v6 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v8 = stablehlo.reshape %v7 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v9 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v10 = stablehlo.maximum %v8, %v9 : tensor<128x16x32x32xf32>
    %v11 = stablehlo.reshape %v10 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v12 = stablehlo.reshape %v11 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v13 = stablehlo.convert %v12 : (tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xbf16>
    %v14 = stablehlo.convert %W2 : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xbf16>
    %v15 = stablehlo.convolution(%v13, %v14)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xbf16>, tensor<16x16x3x3xbf16>) -> tensor<128x16x32x32xbf16>
    %v16 = stablehlo.convert %v15 : (tensor<128x16x32x32xbf16>) -> tensor<128x16x32x32xf32>
    %v17 = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v18 = stablehlo.add %v16, %v17 : tensor<128x16x32x32xf32>
    %v19 = stablehlo.reshape %v18 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v20 = stablehlo.reshape %v19 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v21 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v22 = stablehlo.maximum %v20, %v21 : tensor<128x16x32x32xf32>
    %v23 = stablehlo.reshape %v22 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v24 = stablehlo.reshape %v23 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v25 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v26 = "stablehlo.reduce_window"(%v24, %v25) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16x16x16xf32>
    %v27 = stablehlo.reshape %v26 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v28 = stablehlo.reshape %v27 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v29 = stablehlo.convert %v28 : (tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xbf16>
    %v30 = stablehlo.convert %W3 : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xbf16>
    %v31 = stablehlo.convolution(%v29, %v30)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xbf16>, tensor<16x16x3x3xbf16>) -> tensor<128x16x16x16xbf16>
    %v32 = stablehlo.convert %v31 : (tensor<128x16x16x16xbf16>) -> tensor<128x16x16x16xf32>
    %v33 = stablehlo.broadcast_in_dim %b3, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v34 = stablehlo.add %v32, %v33 : tensor<128x16x16x16xf32>
    %v35 = stablehlo.reshape %v34 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v36 = stablehlo.reshape %v35 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v37 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v38 = stablehlo.maximum %v36, %v37 : tensor<128x16x16x16xf32>
    %v39 = stablehlo.reshape %v38 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v40 = stablehlo.reshape %v39 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v41 = stablehlo.convert %v40 : (tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xbf16>
    %v42 = stablehlo.convert %W4 : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xbf16>
    %v43 = stablehlo.convolution(%v41, %v42)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xbf16>, tensor<16x16x3x3xbf16>) -> tensor<128x16x16x16xbf16>
    %v44 = stablehlo.convert %v43 : (tensor<128x16x16x16xbf16>) -> tensor<128x16x16x16xf32>
    %v45 = stablehlo.broadcast_in_dim %b4, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v46 = stablehlo.add %v44, %v45 : tensor<128x16x16x16xf32>
    %v47 = stablehlo.reshape %v46 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v48 = stablehlo.reshape %v47 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v49 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v50 = stablehlo.maximum %v48, %v49 : tensor<128x16x16x16xf32>
    %v51 = stablehlo.reshape %v50 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v52 = stablehlo.reshape %v51 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v53 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v54 = "stablehlo.reduce_window"(%v52, %v53) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x8x8xf32>
    %v55 = stablehlo.reshape %v54 : (tensor<128x16x8x8xf32>) -> tensor<128x1024xf32>
    %v56 = stablehlo.reshape %v55 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v57 = stablehlo.convert %v56 : (tensor<128x16x8x8xf32>) -> tensor<128x16x8x8xbf16>
    %v58 = stablehlo.convert %W5 : (tensor<32x16x3x3xf32>) -> tensor<32x16x3x3xbf16>
    %v59 = stablehlo.convolution(%v57, %v58)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x8x8xbf16>, tensor<32x16x3x3xbf16>) -> tensor<128x32x8x8xbf16>
    %v60 = stablehlo.convert %v59 : (tensor<128x32x8x8xbf16>) -> tensor<128x32x8x8xf32>
    %v61 = stablehlo.broadcast_in_dim %b5, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v62 = stablehlo.add %v60, %v61 : tensor<128x32x8x8xf32>
    %v63 = stablehlo.reshape %v62 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v64 = stablehlo.reshape %v63 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v65 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v66 = stablehlo.maximum %v64, %v65 : tensor<128x32x8x8xf32>
    %v67 = stablehlo.reshape %v66 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v68 = stablehlo.reshape %v67 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v69 = stablehlo.convert %v68 : (tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xbf16>
    %v70 = stablehlo.convert %W6 : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xbf16>
    %v71 = stablehlo.convolution(%v69, %v70)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xbf16>, tensor<32x32x3x3xbf16>) -> tensor<128x32x8x8xbf16>
    %v72 = stablehlo.convert %v71 : (tensor<128x32x8x8xbf16>) -> tensor<128x32x8x8xf32>
    %v73 = stablehlo.broadcast_in_dim %b6, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v74 = stablehlo.add %v72, %v73 : tensor<128x32x8x8xf32>
    %v75 = stablehlo.reshape %v74 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v76 = stablehlo.reshape %v75 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v77 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v78 = stablehlo.maximum %v76, %v77 : tensor<128x32x8x8xf32>
    %v79 = stablehlo.reshape %v78 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v80 = stablehlo.reshape %v79 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v81 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v82 = "stablehlo.reduce_window"(%v80, %v81) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32x4x4xf32>
    %v83 = stablehlo.reshape %v82 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v84 = stablehlo.reshape %v83 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v85 = stablehlo.convert %v84 : (tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xbf16>
    %v86 = stablehlo.convert %W7 : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xbf16>
    %v87 = stablehlo.convolution(%v85, %v86)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xbf16>, tensor<32x32x3x3xbf16>) -> tensor<128x32x4x4xbf16>
    %v88 = stablehlo.convert %v87 : (tensor<128x32x4x4xbf16>) -> tensor<128x32x4x4xf32>
    %v89 = stablehlo.broadcast_in_dim %b7, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v90 = stablehlo.add %v88, %v89 : tensor<128x32x4x4xf32>
    %v91 = stablehlo.reshape %v90 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v92 = stablehlo.reshape %v91 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v93 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v94 = stablehlo.maximum %v92, %v93 : tensor<128x32x4x4xf32>
    %v95 = stablehlo.reshape %v94 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v96 = stablehlo.reshape %v95 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v97 = stablehlo.convert %v96 : (tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xbf16>
    %v98 = stablehlo.convert %W8 : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xbf16>
    %v99 = stablehlo.convolution(%v97, %v98)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xbf16>, tensor<32x32x3x3xbf16>) -> tensor<128x32x4x4xbf16>
    %v100 = stablehlo.convert %v99 : (tensor<128x32x4x4xbf16>) -> tensor<128x32x4x4xf32>
    %v101 = stablehlo.broadcast_in_dim %b8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v102 = stablehlo.add %v100, %v101 : tensor<128x32x4x4xf32>
    %v103 = stablehlo.reshape %v102 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v104 = stablehlo.reshape %v103 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v105 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v106 = stablehlo.maximum %v104, %v105 : tensor<128x32x4x4xf32>
    %v107 = stablehlo.reshape %v106 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v108 = stablehlo.reshape %v107 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v109 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v110 = "stablehlo.reduce_window"(%v108, %v109) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32x2x2xf32>
    %v111 = stablehlo.reshape %v110 : (tensor<128x32x2x2xf32>) -> tensor<128x128xf32>
    %v112 = stablehlo.dot_general %v111, %W9, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %v113 = stablehlo.broadcast_in_dim %b9, dims = [1] : (tensor<64xf32>) -> tensor<128x64xf32>
    %v114 = stablehlo.add %v112, %v113 : tensor<128x64xf32>
    %v115 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v116 = stablehlo.maximum %v114, %v115 : tensor<128x64xf32>
    %v117 = stablehlo.dot_general %v116, %Wa, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
    %v118 = stablehlo.broadcast_in_dim %ba, dims = [1] : (tensor<64xf32>) -> tensor<128x64xf32>
    %v119 = stablehlo.add %v117, %v118 : tensor<128x64xf32>
    %v120 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v121 = stablehlo.maximum %v119, %v120 : tensor<128x64xf32>
    %v122 = stablehlo.dot_general %v121, %Wb, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x10xf32>) -> tensor<128x10xf32>
    %v123 = stablehlo.broadcast_in_dim %bb, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v124 = stablehlo.add %v122, %v123 : tensor<128x10xf32>
    %v125 = stablehlo.exponential %v124 : tensor<128x10xf32>
    %v126 = stablehlo.constant dense<0.0> : tensor<f32>
    %v127 = stablehlo.reduce(%v125 init: %v126) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %v128 = stablehlo.broadcast_in_dim %v127, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %v129 = stablehlo.divide %v125, %v128 : tensor<128x10xf32>
    %v130 = stablehlo.subtract %v129, %onehot : tensor<128x10xf32>
    %v131 = stablehlo.dot_general %v130, %Wb, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x10xf32>, tensor<64x10xf32>) -> tensor<128x64xf32>
    %v132 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v133 = stablehlo.compare GT, %v119, %v132 : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xi1>
    %v134 = stablehlo.select %v133, %v131, %v132 : tensor<128x64xi1>, tensor<128x64xf32>
    %v135 = stablehlo.dot_general %v134, %Wa, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
    %v136 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v137 = stablehlo.compare GT, %v114, %v136 : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xi1>
    %v138 = stablehlo.select %v137, %v135, %v136 : tensor<128x64xi1>, tensor<128x64xf32>
    %v139 = stablehlo.dot_general %v138, %W9, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x128xf32>
    %v140 = stablehlo.reshape %v107 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v141 = stablehlo.reshape %v139 : (tensor<128x128xf32>) -> tensor<128x32x2x2xf32>
    %v142 = stablehlo.constant dense<0.0> : tensor<f32>
    %v143 = "stablehlo.select_and_scatter"(%v140, %v141, %v142) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x4x4xf32>, tensor<128x32x2x2xf32>, tensor<f32>) -> tensor<128x32x4x4xf32>
    %v144 = stablehlo.reshape %v143 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v145 = stablehlo.reshape %v144 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v146 = stablehlo.reshape %v103 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v147 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v148 = stablehlo.compare GT, %v146, %v147 : (tensor<128x32x4x4xf32>, tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xi1>
    %v149 = stablehlo.select %v148, %v145, %v147 : tensor<128x32x4x4xi1>, tensor<128x32x4x4xf32>
    %v150 = stablehlo.reshape %v149 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v151 = stablehlo.reshape %v150 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v152 = stablehlo.transpose %W8, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v153 = stablehlo.reverse %v152, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v154 = stablehlo.convolution(%v151, %v153)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v155 = stablehlo.reshape %v154 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v156 = stablehlo.reshape %v155 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v157 = stablehlo.reshape %v91 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v158 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v159 = stablehlo.compare GT, %v157, %v158 : (tensor<128x32x4x4xf32>, tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xi1>
    %v160 = stablehlo.select %v159, %v156, %v158 : tensor<128x32x4x4xi1>, tensor<128x32x4x4xf32>
    %v161 = stablehlo.reshape %v160 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v162 = stablehlo.reshape %v161 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v163 = stablehlo.transpose %W7, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v164 = stablehlo.reverse %v163, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v165 = stablehlo.convolution(%v162, %v164)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v166 = stablehlo.reshape %v165 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v167 = stablehlo.reshape %v79 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v168 = stablehlo.reshape %v166 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v169 = stablehlo.constant dense<0.0> : tensor<f32>
    %v170 = "stablehlo.select_and_scatter"(%v167, %v168, %v169) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x8x8xf32>, tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32x8x8xf32>
    %v171 = stablehlo.reshape %v170 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v172 = stablehlo.reshape %v171 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v173 = stablehlo.reshape %v75 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v174 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v175 = stablehlo.compare GT, %v173, %v174 : (tensor<128x32x8x8xf32>, tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xi1>
    %v176 = stablehlo.select %v175, %v172, %v174 : tensor<128x32x8x8xi1>, tensor<128x32x8x8xf32>
    %v177 = stablehlo.reshape %v176 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v178 = stablehlo.reshape %v177 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v179 = stablehlo.transpose %W6, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v180 = stablehlo.reverse %v179, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v181 = stablehlo.convolution(%v178, %v180)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x8x8xf32>
    %v182 = stablehlo.reshape %v181 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v183 = stablehlo.reshape %v182 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v184 = stablehlo.reshape %v63 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v185 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v186 = stablehlo.compare GT, %v184, %v185 : (tensor<128x32x8x8xf32>, tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xi1>
    %v187 = stablehlo.select %v186, %v183, %v185 : tensor<128x32x8x8xi1>, tensor<128x32x8x8xf32>
    %v188 = stablehlo.reshape %v187 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v189 = stablehlo.reshape %v188 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v190 = stablehlo.transpose %W5, dims = [1, 0, 2, 3] : (tensor<32x16x3x3xf32>) -> tensor<16x32x3x3xf32>
    %v191 = stablehlo.reverse %v190, dims = [2, 3] : tensor<16x32x3x3xf32>
    %v192 = stablehlo.convolution(%v189, %v191)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<16x32x3x3xf32>) -> tensor<128x16x8x8xf32>
    %v193 = stablehlo.reshape %v192 : (tensor<128x16x8x8xf32>) -> tensor<128x1024xf32>
    %v194 = stablehlo.reshape %v51 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v195 = stablehlo.reshape %v193 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v196 = stablehlo.constant dense<0.0> : tensor<f32>
    %v197 = "stablehlo.select_and_scatter"(%v194, %v195, %v196) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x16x16xf32>, tensor<128x16x8x8xf32>, tensor<f32>) -> tensor<128x16x16x16xf32>
    %v198 = stablehlo.reshape %v197 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v199 = stablehlo.reshape %v198 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v200 = stablehlo.reshape %v47 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v201 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v202 = stablehlo.compare GT, %v200, %v201 : (tensor<128x16x16x16xf32>, tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xi1>
    %v203 = stablehlo.select %v202, %v199, %v201 : tensor<128x16x16x16xi1>, tensor<128x16x16x16xf32>
    %v204 = stablehlo.reshape %v203 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v205 = stablehlo.reshape %v204 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v206 = stablehlo.transpose %W4, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v207 = stablehlo.reverse %v206, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v208 = stablehlo.convolution(%v205, %v207)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v209 = stablehlo.reshape %v208 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v210 = stablehlo.reshape %v209 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v211 = stablehlo.reshape %v35 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v212 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v213 = stablehlo.compare GT, %v211, %v212 : (tensor<128x16x16x16xf32>, tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xi1>
    %v214 = stablehlo.select %v213, %v210, %v212 : tensor<128x16x16x16xi1>, tensor<128x16x16x16xf32>
    %v215 = stablehlo.reshape %v214 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v216 = stablehlo.reshape %v215 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v217 = stablehlo.transpose %W3, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v218 = stablehlo.reverse %v217, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v219 = stablehlo.convolution(%v216, %v218)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v220 = stablehlo.reshape %v219 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v221 = stablehlo.reshape %v23 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v222 = stablehlo.reshape %v220 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v223 = stablehlo.constant dense<0.0> : tensor<f32>
    %v224 = "stablehlo.select_and_scatter"(%v221, %v222, %v223) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x32x32xf32>, tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x32x32xf32>
    %v225 = stablehlo.reshape %v224 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v226 = stablehlo.reshape %v225 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v227 = stablehlo.reshape %v19 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v228 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v229 = stablehlo.compare GT, %v227, %v228 : (tensor<128x16x32x32xf32>, tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xi1>
    %v230 = stablehlo.select %v229, %v226, %v228 : tensor<128x16x32x32xi1>, tensor<128x16x32x32xf32>
    %v231 = stablehlo.reshape %v230 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v232 = stablehlo.reshape %v231 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v233 = stablehlo.transpose %W2, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v234 = stablehlo.reverse %v233, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v235 = stablehlo.convolution(%v232, %v234)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x32x32xf32>
    %v236 = stablehlo.reshape %v235 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v237 = stablehlo.reshape %v236 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v238 = stablehlo.reshape %v7 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v239 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v240 = stablehlo.compare GT, %v238, %v239 : (tensor<128x16x32x32xf32>, tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xi1>
    %v241 = stablehlo.select %v240, %v237, %v239 : tensor<128x16x32x32xi1>, tensor<128x16x32x32xf32>
    %v242 = stablehlo.reshape %v241 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v243 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v244 = stablehlo.reshape %v242 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v245 = stablehlo.transpose %v243, dims = [1, 0, 2, 3] : (tensor<128x3x32x32xf32>) -> tensor<3x128x32x32xf32>
    %v246 = stablehlo.transpose %v244, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v247 = stablehlo.convolution(%v245, %v246)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<3x16x3x3xf32>
    %v248 = stablehlo.transpose %v247, dims = [1, 0, 2, 3] : (tensor<3x16x3x3xf32>) -> tensor<16x3x3x3xf32>
    %v249 = stablehlo.constant dense<0.00078125> : tensor<16x3x3x3xf32>
    %v250 = stablehlo.multiply %v248, %v249 : tensor<16x3x3x3xf32>
    %v251 = stablehlo.subtract %W1, %v250 : tensor<16x3x3x3xf32>
    %v252 = stablehlo.reshape %v242 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v253 = stablehlo.constant dense<0.0> : tensor<f32>
    %v254 = stablehlo.reduce(%v252 init: %v253) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v255 = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %v256 = stablehlo.multiply %v254, %v255 : tensor<16xf32>
    %v257 = stablehlo.subtract %b1, %v256 : tensor<16xf32>
    %v258 = stablehlo.reshape %v11 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v259 = stablehlo.reshape %v231 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v260 = stablehlo.transpose %v258, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v261 = stablehlo.transpose %v259, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v262 = stablehlo.convolution(%v260, %v261)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x16x3x3xf32>
    %v263 = stablehlo.transpose %v262, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v264 = stablehlo.constant dense<0.00078125> : tensor<16x16x3x3xf32>
    %v265 = stablehlo.multiply %v263, %v264 : tensor<16x16x3x3xf32>
    %v266 = stablehlo.subtract %W2, %v265 : tensor<16x16x3x3xf32>
    %v267 = stablehlo.reshape %v231 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v268 = stablehlo.constant dense<0.0> : tensor<f32>
    %v269 = stablehlo.reduce(%v267 init: %v268) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v270 = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %v271 = stablehlo.multiply %v269, %v270 : tensor<16xf32>
    %v272 = stablehlo.subtract %b2, %v271 : tensor<16xf32>
    %v273 = stablehlo.reshape %v27 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v274 = stablehlo.reshape %v215 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v275 = stablehlo.transpose %v273, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v276 = stablehlo.transpose %v274, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v277 = stablehlo.convolution(%v275, %v276)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v278 = stablehlo.transpose %v277, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v279 = stablehlo.constant dense<0.00078125> : tensor<16x16x3x3xf32>
    %v280 = stablehlo.multiply %v278, %v279 : tensor<16x16x3x3xf32>
    %v281 = stablehlo.subtract %W3, %v280 : tensor<16x16x3x3xf32>
    %v282 = stablehlo.reshape %v215 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v283 = stablehlo.constant dense<0.0> : tensor<f32>
    %v284 = stablehlo.reduce(%v282 init: %v283) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v285 = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %v286 = stablehlo.multiply %v284, %v285 : tensor<16xf32>
    %v287 = stablehlo.subtract %b3, %v286 : tensor<16xf32>
    %v288 = stablehlo.reshape %v39 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v289 = stablehlo.reshape %v204 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v290 = stablehlo.transpose %v288, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v291 = stablehlo.transpose %v289, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v292 = stablehlo.convolution(%v290, %v291)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v293 = stablehlo.transpose %v292, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v294 = stablehlo.constant dense<0.00078125> : tensor<16x16x3x3xf32>
    %v295 = stablehlo.multiply %v293, %v294 : tensor<16x16x3x3xf32>
    %v296 = stablehlo.subtract %W4, %v295 : tensor<16x16x3x3xf32>
    %v297 = stablehlo.reshape %v204 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v298 = stablehlo.constant dense<0.0> : tensor<f32>
    %v299 = stablehlo.reduce(%v297 init: %v298) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v300 = stablehlo.constant dense<0.00078125> : tensor<16xf32>
    %v301 = stablehlo.multiply %v299, %v300 : tensor<16xf32>
    %v302 = stablehlo.subtract %b4, %v301 : tensor<16xf32>
    %v303 = stablehlo.reshape %v55 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v304 = stablehlo.reshape %v188 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v305 = stablehlo.transpose %v303, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v306 = stablehlo.transpose %v304, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v307 = stablehlo.convolution(%v305, %v306)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<16x32x3x3xf32>
    %v308 = stablehlo.transpose %v307, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v309 = stablehlo.constant dense<0.00078125> : tensor<32x16x3x3xf32>
    %v310 = stablehlo.multiply %v308, %v309 : tensor<32x16x3x3xf32>
    %v311 = stablehlo.subtract %W5, %v310 : tensor<32x16x3x3xf32>
    %v312 = stablehlo.reshape %v188 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v313 = stablehlo.constant dense<0.0> : tensor<f32>
    %v314 = stablehlo.reduce(%v312 init: %v313) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v315 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v316 = stablehlo.multiply %v314, %v315 : tensor<32xf32>
    %v317 = stablehlo.subtract %b5, %v316 : tensor<32xf32>
    %v318 = stablehlo.reshape %v67 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v319 = stablehlo.reshape %v177 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v320 = stablehlo.transpose %v318, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v321 = stablehlo.transpose %v319, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v322 = stablehlo.convolution(%v320, %v321)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<32x32x3x3xf32>
    %v323 = stablehlo.transpose %v322, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v324 = stablehlo.constant dense<0.00078125> : tensor<32x32x3x3xf32>
    %v325 = stablehlo.multiply %v323, %v324 : tensor<32x32x3x3xf32>
    %v326 = stablehlo.subtract %W6, %v325 : tensor<32x32x3x3xf32>
    %v327 = stablehlo.reshape %v177 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v328 = stablehlo.constant dense<0.0> : tensor<f32>
    %v329 = stablehlo.reduce(%v327 init: %v328) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v330 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v331 = stablehlo.multiply %v329, %v330 : tensor<32xf32>
    %v332 = stablehlo.subtract %b6, %v331 : tensor<32xf32>
    %v333 = stablehlo.reshape %v83 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v334 = stablehlo.reshape %v161 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v335 = stablehlo.transpose %v333, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v336 = stablehlo.transpose %v334, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v337 = stablehlo.convolution(%v335, %v336)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v338 = stablehlo.transpose %v337, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v339 = stablehlo.constant dense<0.00078125> : tensor<32x32x3x3xf32>
    %v340 = stablehlo.multiply %v338, %v339 : tensor<32x32x3x3xf32>
    %v341 = stablehlo.subtract %W7, %v340 : tensor<32x32x3x3xf32>
    %v342 = stablehlo.reshape %v161 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v343 = stablehlo.constant dense<0.0> : tensor<f32>
    %v344 = stablehlo.reduce(%v342 init: %v343) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v345 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v346 = stablehlo.multiply %v344, %v345 : tensor<32xf32>
    %v347 = stablehlo.subtract %b7, %v346 : tensor<32xf32>
    %v348 = stablehlo.reshape %v95 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v349 = stablehlo.reshape %v150 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v350 = stablehlo.transpose %v348, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v351 = stablehlo.transpose %v349, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v352 = stablehlo.convolution(%v350, %v351)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v353 = stablehlo.transpose %v352, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v354 = stablehlo.constant dense<0.00078125> : tensor<32x32x3x3xf32>
    %v355 = stablehlo.multiply %v353, %v354 : tensor<32x32x3x3xf32>
    %v356 = stablehlo.subtract %W8, %v355 : tensor<32x32x3x3xf32>
    %v357 = stablehlo.reshape %v150 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v358 = stablehlo.constant dense<0.0> : tensor<f32>
    %v359 = stablehlo.reduce(%v357 init: %v358) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v360 = stablehlo.constant dense<0.00078125> : tensor<32xf32>
    %v361 = stablehlo.multiply %v359, %v360 : tensor<32xf32>
    %v362 = stablehlo.subtract %b8, %v361 : tensor<32xf32>
    %v363 = stablehlo.dot_general %v111, %v138, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %v364 = stablehlo.constant dense<0.00078125> : tensor<128x64xf32>
    %v365 = stablehlo.multiply %v363, %v364 : tensor<128x64xf32>
    %v366 = stablehlo.subtract %W9, %v365 : tensor<128x64xf32>
    %v367 = stablehlo.constant dense<0.0> : tensor<f32>
    %v368 = stablehlo.reduce(%v138 init: %v367) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v369 = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %v370 = stablehlo.multiply %v368, %v369 : tensor<64xf32>
    %v371 = stablehlo.subtract %b9, %v370 : tensor<64xf32>
    %v372 = stablehlo.dot_general %v116, %v134, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<64x64xf32>
    %v373 = stablehlo.constant dense<0.00078125> : tensor<64x64xf32>
    %v374 = stablehlo.multiply %v372, %v373 : tensor<64x64xf32>
    %v375 = stablehlo.subtract %Wa, %v374 : tensor<64x64xf32>
    %v376 = stablehlo.constant dense<0.0> : tensor<f32>
    %v377 = stablehlo.reduce(%v134 init: %v376) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v378 = stablehlo.constant dense<0.00078125> : tensor<64xf32>
    %v379 = stablehlo.multiply %v377, %v378 : tensor<64xf32>
    %v380 = stablehlo.subtract %ba, %v379 : tensor<64xf32>
    %v381 = stablehlo.dot_general %v121, %v130, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x10xf32>) -> tensor<64x10xf32>
    %v382 = stablehlo.constant dense<0.00078125> : tensor<64x10xf32>
    %v383 = stablehlo.multiply %v381, %v382 : tensor<64x10xf32>
    %v384 = stablehlo.subtract %Wb, %v383 : tensor<64x10xf32>
    %v385 = stablehlo.constant dense<0.0> : tensor<f32>
    %v386 = stablehlo.reduce(%v130 init: %v385) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v387 = stablehlo.constant dense<0.00078125> : tensor<10xf32>
    %v388 = stablehlo.multiply %v386, %v387 : tensor<10xf32>
    %v389 = stablehlo.subtract %bb, %v388 : tensor<10xf32>
    return %v251, %v257, %v266, %v272, %v281, %v287, %v296, %v302, %v311, %v317, %v326, %v332, %v341, %v347, %v356, %v362, %v366, %v371, %v375, %v380, %v384, %v389 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>
  }
}
