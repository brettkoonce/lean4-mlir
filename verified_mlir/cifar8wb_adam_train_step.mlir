module @m {
  func.func @cifar8wb_adam_train_step(%x: tensor<128x3072xf32>, %W1: tensor<16x3x3x3xf32>, %cb1: tensor<16xf32>, %W2: tensor<16x16x3x3xf32>, %cb2: tensor<16xf32>, %W3: tensor<16x16x3x3xf32>, %cb3: tensor<16xf32>, %W4: tensor<16x16x3x3xf32>, %cb4: tensor<16xf32>, %W5: tensor<32x16x3x3xf32>, %cb5: tensor<32xf32>, %W6: tensor<32x32x3x3xf32>, %cb6: tensor<32xf32>, %W7: tensor<32x32x3x3xf32>, %cb7: tensor<32xf32>, %W8: tensor<32x32x3x3xf32>, %cb8: tensor<32xf32>, %W9: tensor<128x512xf32>, %b9: tensor<512xf32>, %Wa: tensor<512x512xf32>, %ba: tensor<512xf32>, %Wb: tensor<512x10xf32>, %bb: tensor<10xf32>, %W1m: tensor<16x3x3x3xf32>, %cb1m: tensor<16xf32>, %W2m: tensor<16x16x3x3xf32>, %cb2m: tensor<16xf32>, %W3m: tensor<16x16x3x3xf32>, %cb3m: tensor<16xf32>, %W4m: tensor<16x16x3x3xf32>, %cb4m: tensor<16xf32>, %W5m: tensor<32x16x3x3xf32>, %cb5m: tensor<32xf32>, %W6m: tensor<32x32x3x3xf32>, %cb6m: tensor<32xf32>, %W7m: tensor<32x32x3x3xf32>, %cb7m: tensor<32xf32>, %W8m: tensor<32x32x3x3xf32>, %cb8m: tensor<32xf32>, %W9m: tensor<128x512xf32>, %b9m: tensor<512xf32>, %Wam: tensor<512x512xf32>, %bam: tensor<512xf32>, %Wbm: tensor<512x10xf32>, %bbm: tensor<10xf32>, %W1v: tensor<16x3x3x3xf32>, %cb1v: tensor<16xf32>, %W2v: tensor<16x16x3x3xf32>, %cb2v: tensor<16xf32>, %W3v: tensor<16x16x3x3xf32>, %cb3v: tensor<16xf32>, %W4v: tensor<16x16x3x3xf32>, %cb4v: tensor<16xf32>, %W5v: tensor<32x16x3x3xf32>, %cb5v: tensor<32xf32>, %W6v: tensor<32x32x3x3xf32>, %cb6v: tensor<32xf32>, %W7v: tensor<32x32x3x3xf32>, %cb7v: tensor<32xf32>, %W8v: tensor<32x32x3x3xf32>, %cb8v: tensor<32xf32>, %W9v: tensor<128x512xf32>, %b9v: tensor<512xf32>, %Wav: tensor<512x512xf32>, %bav: tensor<512xf32>, %Wbv: tensor<512x10xf32>, %bbv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: tensor<128x10xf32>) -> (tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>) {
    // ── cifar8 AdamW train step: every line is pretty(verified AST node), except the
    //    marked report-only loss + the %bc passthroughs ──
    %lzero = stablehlo.constant dense<0.0> : tensor<f32>
    %b1 = stablehlo.constant dense<0.9> : tensor<f32>
    %ob1 = stablehlo.constant dense<0.1> : tensor<f32>
    %b2 = stablehlo.constant dense<0.999> : tensor<f32>
    %ob2 = stablehlo.constant dense<0.001> : tensor<f32>
    %eps = stablehlo.constant dense<1.0e-8> : tensor<f32>
    %wd = stablehlo.constant dense<0.0001> : tensor<f32>
    %v0 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v1 = stablehlo.convolution(%v0, %W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x3x32x32xf32>, tensor<16x3x3x3xf32>) -> tensor<128x16x32x32xf32>
    %v2 = stablehlo.broadcast_in_dim %cb1, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
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
    %v11 = stablehlo.broadcast_in_dim %cb2, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
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
    %v24 = stablehlo.broadcast_in_dim %cb3, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
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
    %v33 = stablehlo.broadcast_in_dim %cb4, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
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
    %v46 = stablehlo.broadcast_in_dim %cb5, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
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
    %v55 = stablehlo.broadcast_in_dim %cb6, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
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
    %v68 = stablehlo.broadcast_in_dim %cb7, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
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
    %v77 = stablehlo.broadcast_in_dim %cb8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
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
    %v88 = stablehlo.dot_general %v87, %W9, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x512xf32>) -> tensor<128x512xf32>
    %v89 = stablehlo.broadcast_in_dim %b9, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v90 = stablehlo.add %v88, %v89 : tensor<128x512xf32>
    %v91 = stablehlo.reshape %v90 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v92 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v93 = stablehlo.maximum %v91, %v92 : tensor<128x32x4x4xf32>
    %v94 = stablehlo.reshape %v93 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v95 = stablehlo.dot_general %v94, %Wa, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %v96 = stablehlo.broadcast_in_dim %ba, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v97 = stablehlo.add %v95, %v96 : tensor<128x512xf32>
    %v98 = stablehlo.reshape %v97 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v99 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v100 = stablehlo.maximum %v98, %v99 : tensor<128x32x4x4xf32>
    %v101 = stablehlo.reshape %v100 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v102 = stablehlo.dot_general %v101, %Wb, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x10xf32>) -> tensor<128x10xf32>
    %v103 = stablehlo.broadcast_in_dim %bb, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v104 = stablehlo.add %v102, %v103 : tensor<128x10xf32>
    %v105 = stablehlo.reshape %v104 : (tensor<128x10xf32>) -> tensor<128x1x10xf32>
    %v106 = stablehlo.constant dense<0.0> : tensor<f32>
    %v107 = stablehlo.exponential %v105 : tensor<128x1x10xf32>
    %v108 = stablehlo.reduce(%v107 init: %v106) applies stablehlo.add across dimensions = [2] : (tensor<128x1x10xf32>, tensor<f32>) -> tensor<128x1xf32>
    %v109 = stablehlo.broadcast_in_dim %v108, dims = [0, 1] : (tensor<128x1xf32>) -> tensor<128x1x10xf32>
    %v110 = stablehlo.divide %v107, %v109 : tensor<128x1x10xf32>
    %v111 = stablehlo.reshape %v110 : (tensor<128x1x10xf32>) -> tensor<128x10xf32>
    %v112 = stablehlo.subtract %v111, %onehot : tensor<128x10xf32>
    %v113 = stablehlo.constant dense<0.0078125> : tensor<128x10xf32>
    %v114 = stablehlo.multiply %v112, %v113 : tensor<128x10xf32>
    // ── report-only scalar loss (NOT pretty(AST): the kit has no rank-0 loss op; it
    //    feeds no parameter, only the driver's progress line) ──
    %llog = stablehlo.log %v111 : tensor<128x10xf32>
    %ohll = stablehlo.multiply %onehot, %llog : tensor<128x10xf32>
    %csum = stablehlo.reduce(%ohll init: %lzero) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<f32>
    %cneg = stablehlo.negate %csum : tensor<f32>
    %lbf = stablehlo.constant dense<128.0> : tensor<f32>
    %loss = stablehlo.divide %cneg, %lbf : tensor<f32>
    %v115 = stablehlo.reshape %v114 : (tensor<128x10xf32>) -> tensor<128x1x10xf32>
    %v116 = stablehlo.dot_general %v115, %Wb, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x10xf32>, tensor<512x10xf32>) -> tensor<128x1x512xf32>
    %v117 = stablehlo.reshape %v116 : (tensor<128x1x512xf32>) -> tensor<128x512xf32>
    %v118 = stablehlo.reshape %v117 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v119 = stablehlo.reshape %v97 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v120 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v121 = stablehlo.compare GT, %v119, %v120 : (tensor<128x32x4x4xf32>, tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xi1>
    %v122 = stablehlo.select %v121, %v118, %v120 : tensor<128x32x4x4xi1>, tensor<128x32x4x4xf32>
    %v123 = stablehlo.reshape %v122 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v124 = stablehlo.reshape %v123 : (tensor<128x512xf32>) -> tensor<128x1x512xf32>
    %v125 = stablehlo.dot_general %v124, %Wa, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x512xf32>, tensor<512x512xf32>) -> tensor<128x1x512xf32>
    %v126 = stablehlo.reshape %v125 : (tensor<128x1x512xf32>) -> tensor<128x512xf32>
    %v127 = stablehlo.reshape %v126 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v128 = stablehlo.reshape %v90 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v129 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v130 = stablehlo.compare GT, %v128, %v129 : (tensor<128x32x4x4xf32>, tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xi1>
    %v131 = stablehlo.select %v130, %v127, %v129 : tensor<128x32x4x4xi1>, tensor<128x32x4x4xf32>
    %v132 = stablehlo.reshape %v131 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v133 = stablehlo.reshape %v132 : (tensor<128x512xf32>) -> tensor<128x1x512xf32>
    %v134 = stablehlo.dot_general %v133, %W9, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x512xf32>, tensor<128x512xf32>) -> tensor<128x1x128xf32>
    %v135 = stablehlo.reshape %v134 : (tensor<128x1x128xf32>) -> tensor<128x128xf32>
    %v136 = stablehlo.reshape %v83 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v137 = stablehlo.reshape %v135 : (tensor<128x128xf32>) -> tensor<128x32x2x2xf32>
    %v138 = stablehlo.constant dense<0.0> : tensor<f32>
    %v139 = "stablehlo.select_and_scatter"(%v136, %v137, %v138) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x4x4xf32>, tensor<128x32x2x2xf32>, tensor<f32>) -> tensor<128x32x4x4xf32>
    %v140 = stablehlo.reshape %v139 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v141 = stablehlo.reshape %v140 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v142 = stablehlo.reshape %v79 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v143 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v144 = stablehlo.compare GT, %v142, %v143 : (tensor<128x32x4x4xf32>, tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xi1>
    %v145 = stablehlo.select %v144, %v141, %v143 : tensor<128x32x4x4xi1>, tensor<128x32x4x4xf32>
    %v146 = stablehlo.reshape %v145 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v147 = stablehlo.reshape %v146 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v148 = stablehlo.reverse %W8, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v149 = stablehlo.transpose %v148, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v150 = stablehlo.convolution(%v147, %v149)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v151 = stablehlo.reshape %v150 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v152 = stablehlo.reshape %v151 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v153 = stablehlo.reshape %v70 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v154 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v155 = stablehlo.compare GT, %v153, %v154 : (tensor<128x32x4x4xf32>, tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xi1>
    %v156 = stablehlo.select %v155, %v152, %v154 : tensor<128x32x4x4xi1>, tensor<128x32x4x4xf32>
    %v157 = stablehlo.reshape %v156 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v158 = stablehlo.reshape %v157 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v159 = stablehlo.reverse %W7, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v160 = stablehlo.transpose %v159, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v161 = stablehlo.convolution(%v158, %v160)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v162 = stablehlo.reshape %v161 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v163 = stablehlo.reshape %v61 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v164 = stablehlo.reshape %v162 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v165 = stablehlo.constant dense<0.0> : tensor<f32>
    %v166 = "stablehlo.select_and_scatter"(%v163, %v164, %v165) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x8x8xf32>, tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32x8x8xf32>
    %v167 = stablehlo.reshape %v166 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v168 = stablehlo.reshape %v167 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v169 = stablehlo.reshape %v57 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v170 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v171 = stablehlo.compare GT, %v169, %v170 : (tensor<128x32x8x8xf32>, tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xi1>
    %v172 = stablehlo.select %v171, %v168, %v170 : tensor<128x32x8x8xi1>, tensor<128x32x8x8xf32>
    %v173 = stablehlo.reshape %v172 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v174 = stablehlo.reshape %v173 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v175 = stablehlo.reverse %W6, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v176 = stablehlo.transpose %v175, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v177 = stablehlo.convolution(%v174, %v176)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x8x8xf32>
    %v178 = stablehlo.reshape %v177 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v179 = stablehlo.reshape %v178 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v180 = stablehlo.reshape %v48 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v181 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v182 = stablehlo.compare GT, %v180, %v181 : (tensor<128x32x8x8xf32>, tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xi1>
    %v183 = stablehlo.select %v182, %v179, %v181 : tensor<128x32x8x8xi1>, tensor<128x32x8x8xf32>
    %v184 = stablehlo.reshape %v183 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v185 = stablehlo.reshape %v184 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v186 = stablehlo.reverse %W5, dims = [2, 3] : tensor<32x16x3x3xf32>
    %v187 = stablehlo.transpose %v186, dims = [1, 0, 2, 3] : (tensor<32x16x3x3xf32>) -> tensor<16x32x3x3xf32>
    %v188 = stablehlo.convolution(%v185, %v187)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<16x32x3x3xf32>) -> tensor<128x16x8x8xf32>
    %v189 = stablehlo.reshape %v188 : (tensor<128x16x8x8xf32>) -> tensor<128x1024xf32>
    %v190 = stablehlo.reshape %v39 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v191 = stablehlo.reshape %v189 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v192 = stablehlo.constant dense<0.0> : tensor<f32>
    %v193 = "stablehlo.select_and_scatter"(%v190, %v191, %v192) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x16x16xf32>, tensor<128x16x8x8xf32>, tensor<f32>) -> tensor<128x16x16x16xf32>
    %v194 = stablehlo.reshape %v193 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v195 = stablehlo.reshape %v194 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v196 = stablehlo.reshape %v35 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v197 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v198 = stablehlo.compare GT, %v196, %v197 : (tensor<128x16x16x16xf32>, tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xi1>
    %v199 = stablehlo.select %v198, %v195, %v197 : tensor<128x16x16x16xi1>, tensor<128x16x16x16xf32>
    %v200 = stablehlo.reshape %v199 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v201 = stablehlo.reshape %v200 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v202 = stablehlo.reverse %W4, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v203 = stablehlo.transpose %v202, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v204 = stablehlo.convolution(%v201, %v203)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v205 = stablehlo.reshape %v204 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v206 = stablehlo.reshape %v205 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v207 = stablehlo.reshape %v26 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v208 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v209 = stablehlo.compare GT, %v207, %v208 : (tensor<128x16x16x16xf32>, tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xi1>
    %v210 = stablehlo.select %v209, %v206, %v208 : tensor<128x16x16x16xi1>, tensor<128x16x16x16xf32>
    %v211 = stablehlo.reshape %v210 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v212 = stablehlo.reshape %v211 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v213 = stablehlo.reverse %W3, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v214 = stablehlo.transpose %v213, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v215 = stablehlo.convolution(%v212, %v214)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v216 = stablehlo.reshape %v215 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v217 = stablehlo.reshape %v17 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v218 = stablehlo.reshape %v216 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v219 = stablehlo.constant dense<0.0> : tensor<f32>
    %v220 = "stablehlo.select_and_scatter"(%v217, %v218, %v219) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x32x32xf32>, tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x32x32xf32>
    %v221 = stablehlo.reshape %v220 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v222 = stablehlo.reshape %v221 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v223 = stablehlo.reshape %v13 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v224 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v225 = stablehlo.compare GT, %v223, %v224 : (tensor<128x16x32x32xf32>, tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xi1>
    %v226 = stablehlo.select %v225, %v222, %v224 : tensor<128x16x32x32xi1>, tensor<128x16x32x32xf32>
    %v227 = stablehlo.reshape %v226 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v228 = stablehlo.reshape %v227 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v229 = stablehlo.reverse %W2, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v230 = stablehlo.transpose %v229, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v231 = stablehlo.convolution(%v228, %v230)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x32x32xf32>
    %v232 = stablehlo.reshape %v231 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v233 = stablehlo.reshape %v232 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v234 = stablehlo.reshape %v4 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v235 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v236 = stablehlo.compare GT, %v234, %v235 : (tensor<128x16x32x32xf32>, tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xi1>
    %v237 = stablehlo.select %v236, %v233, %v235 : tensor<128x16x32x32xi1>, tensor<128x16x32x32xf32>
    %v238 = stablehlo.reshape %v237 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v239 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v240 = stablehlo.reshape %v238 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v241 = stablehlo.transpose %v239, dims = [1, 0, 2, 3] : (tensor<128x3x32x32xf32>) -> tensor<3x128x32x32xf32>
    %v242 = stablehlo.transpose %v240, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v243 = stablehlo.convolution(%v241, %v242)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<3x16x3x3xf32>
    %v244 = stablehlo.transpose %v243, dims = [1, 0, 2, 3] : (tensor<3x16x3x3xf32>) -> tensor<16x3x3x3xf32>
    %v245 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v246 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v247 = stablehlo.multiply %v245, %W1m : tensor<16x3x3x3xf32>
    %v248 = stablehlo.multiply %v246, %v244 : tensor<16x3x3x3xf32>
    %v249 = stablehlo.add %v247, %v248 : tensor<16x3x3x3xf32>
    %v250 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v251 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v252 = stablehlo.multiply %v250, %W1v : tensor<16x3x3x3xf32>
    %v253 = stablehlo.multiply %v244, %v244 : tensor<16x3x3x3xf32>
    %v254 = stablehlo.multiply %v251, %v253 : tensor<16x3x3x3xf32>
    %v255 = stablehlo.add %v252, %v254 : tensor<16x3x3x3xf32>
    %v256 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v257 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v258 = stablehlo.divide %v249, %v256 : tensor<16x3x3x3xf32>
    %v259 = stablehlo.divide %v255, %v257 : tensor<16x3x3x3xf32>
    %v260 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v261 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v262 = stablehlo.sqrt %v259 : tensor<16x3x3x3xf32>
    %v263 = stablehlo.add %v262, %v261 : tensor<16x3x3x3xf32>
    %v264 = stablehlo.divide %v258, %v263 : tensor<16x3x3x3xf32>
    %v265 = stablehlo.multiply %v260, %v264 : tensor<16x3x3x3xf32>
    %v266 = stablehlo.subtract %W1, %v265 : tensor<16x3x3x3xf32>
    %v267 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v268 = stablehlo.multiply %v267, %v260 : tensor<16x3x3x3xf32>
    %v269 = stablehlo.multiply %v268, %W1 : tensor<16x3x3x3xf32>
    %v270 = stablehlo.subtract %v266, %v269 : tensor<16x3x3x3xf32>
    %v271 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v272 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v273 = stablehlo.multiply %v271, %W1m : tensor<16x3x3x3xf32>
    %v274 = stablehlo.multiply %v272, %v244 : tensor<16x3x3x3xf32>
    %v275 = stablehlo.add %v273, %v274 : tensor<16x3x3x3xf32>
    %v276 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v277 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v278 = stablehlo.multiply %v276, %W1v : tensor<16x3x3x3xf32>
    %v279 = stablehlo.multiply %v244, %v244 : tensor<16x3x3x3xf32>
    %v280 = stablehlo.multiply %v277, %v279 : tensor<16x3x3x3xf32>
    %v281 = stablehlo.add %v278, %v280 : tensor<16x3x3x3xf32>
    %v282 = stablehlo.reshape %v238 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v283 = stablehlo.constant dense<0.0> : tensor<f32>
    %v284 = stablehlo.reduce(%v282 init: %v283) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v285 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v286 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v287 = stablehlo.multiply %v285, %cb1m : tensor<16xf32>
    %v288 = stablehlo.multiply %v286, %v284 : tensor<16xf32>
    %v289 = stablehlo.add %v287, %v288 : tensor<16xf32>
    %v290 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v291 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v292 = stablehlo.multiply %v290, %cb1v : tensor<16xf32>
    %v293 = stablehlo.multiply %v284, %v284 : tensor<16xf32>
    %v294 = stablehlo.multiply %v291, %v293 : tensor<16xf32>
    %v295 = stablehlo.add %v292, %v294 : tensor<16xf32>
    %v296 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v297 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v298 = stablehlo.divide %v289, %v296 : tensor<16xf32>
    %v299 = stablehlo.divide %v295, %v297 : tensor<16xf32>
    %v300 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v301 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v302 = stablehlo.sqrt %v299 : tensor<16xf32>
    %v303 = stablehlo.add %v302, %v301 : tensor<16xf32>
    %v304 = stablehlo.divide %v298, %v303 : tensor<16xf32>
    %v305 = stablehlo.multiply %v300, %v304 : tensor<16xf32>
    %v306 = stablehlo.subtract %cb1, %v305 : tensor<16xf32>
    %v307 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v308 = stablehlo.multiply %v307, %v300 : tensor<16xf32>
    %v309 = stablehlo.multiply %v308, %cb1 : tensor<16xf32>
    %v310 = stablehlo.subtract %v306, %v309 : tensor<16xf32>
    %v311 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v312 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v313 = stablehlo.multiply %v311, %cb1m : tensor<16xf32>
    %v314 = stablehlo.multiply %v312, %v284 : tensor<16xf32>
    %v315 = stablehlo.add %v313, %v314 : tensor<16xf32>
    %v316 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v317 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v318 = stablehlo.multiply %v316, %cb1v : tensor<16xf32>
    %v319 = stablehlo.multiply %v284, %v284 : tensor<16xf32>
    %v320 = stablehlo.multiply %v317, %v319 : tensor<16xf32>
    %v321 = stablehlo.add %v318, %v320 : tensor<16xf32>
    %v322 = stablehlo.reshape %v8 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v323 = stablehlo.reshape %v227 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v324 = stablehlo.transpose %v322, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v325 = stablehlo.transpose %v323, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v326 = stablehlo.convolution(%v324, %v325)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x16x3x3xf32>
    %v327 = stablehlo.transpose %v326, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v328 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v329 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v330 = stablehlo.multiply %v328, %W2m : tensor<16x16x3x3xf32>
    %v331 = stablehlo.multiply %v329, %v327 : tensor<16x16x3x3xf32>
    %v332 = stablehlo.add %v330, %v331 : tensor<16x16x3x3xf32>
    %v333 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v334 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v335 = stablehlo.multiply %v333, %W2v : tensor<16x16x3x3xf32>
    %v336 = stablehlo.multiply %v327, %v327 : tensor<16x16x3x3xf32>
    %v337 = stablehlo.multiply %v334, %v336 : tensor<16x16x3x3xf32>
    %v338 = stablehlo.add %v335, %v337 : tensor<16x16x3x3xf32>
    %v339 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v340 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v341 = stablehlo.divide %v332, %v339 : tensor<16x16x3x3xf32>
    %v342 = stablehlo.divide %v338, %v340 : tensor<16x16x3x3xf32>
    %v343 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v344 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v345 = stablehlo.sqrt %v342 : tensor<16x16x3x3xf32>
    %v346 = stablehlo.add %v345, %v344 : tensor<16x16x3x3xf32>
    %v347 = stablehlo.divide %v341, %v346 : tensor<16x16x3x3xf32>
    %v348 = stablehlo.multiply %v343, %v347 : tensor<16x16x3x3xf32>
    %v349 = stablehlo.subtract %W2, %v348 : tensor<16x16x3x3xf32>
    %v350 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v351 = stablehlo.multiply %v350, %v343 : tensor<16x16x3x3xf32>
    %v352 = stablehlo.multiply %v351, %W2 : tensor<16x16x3x3xf32>
    %v353 = stablehlo.subtract %v349, %v352 : tensor<16x16x3x3xf32>
    %v354 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v355 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v356 = stablehlo.multiply %v354, %W2m : tensor<16x16x3x3xf32>
    %v357 = stablehlo.multiply %v355, %v327 : tensor<16x16x3x3xf32>
    %v358 = stablehlo.add %v356, %v357 : tensor<16x16x3x3xf32>
    %v359 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v360 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v361 = stablehlo.multiply %v359, %W2v : tensor<16x16x3x3xf32>
    %v362 = stablehlo.multiply %v327, %v327 : tensor<16x16x3x3xf32>
    %v363 = stablehlo.multiply %v360, %v362 : tensor<16x16x3x3xf32>
    %v364 = stablehlo.add %v361, %v363 : tensor<16x16x3x3xf32>
    %v365 = stablehlo.reshape %v227 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v366 = stablehlo.constant dense<0.0> : tensor<f32>
    %v367 = stablehlo.reduce(%v365 init: %v366) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v368 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v369 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v370 = stablehlo.multiply %v368, %cb2m : tensor<16xf32>
    %v371 = stablehlo.multiply %v369, %v367 : tensor<16xf32>
    %v372 = stablehlo.add %v370, %v371 : tensor<16xf32>
    %v373 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v374 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v375 = stablehlo.multiply %v373, %cb2v : tensor<16xf32>
    %v376 = stablehlo.multiply %v367, %v367 : tensor<16xf32>
    %v377 = stablehlo.multiply %v374, %v376 : tensor<16xf32>
    %v378 = stablehlo.add %v375, %v377 : tensor<16xf32>
    %v379 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v380 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v381 = stablehlo.divide %v372, %v379 : tensor<16xf32>
    %v382 = stablehlo.divide %v378, %v380 : tensor<16xf32>
    %v383 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v384 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v385 = stablehlo.sqrt %v382 : tensor<16xf32>
    %v386 = stablehlo.add %v385, %v384 : tensor<16xf32>
    %v387 = stablehlo.divide %v381, %v386 : tensor<16xf32>
    %v388 = stablehlo.multiply %v383, %v387 : tensor<16xf32>
    %v389 = stablehlo.subtract %cb2, %v388 : tensor<16xf32>
    %v390 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v391 = stablehlo.multiply %v390, %v383 : tensor<16xf32>
    %v392 = stablehlo.multiply %v391, %cb2 : tensor<16xf32>
    %v393 = stablehlo.subtract %v389, %v392 : tensor<16xf32>
    %v394 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v395 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v396 = stablehlo.multiply %v394, %cb2m : tensor<16xf32>
    %v397 = stablehlo.multiply %v395, %v367 : tensor<16xf32>
    %v398 = stablehlo.add %v396, %v397 : tensor<16xf32>
    %v399 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v400 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v401 = stablehlo.multiply %v399, %cb2v : tensor<16xf32>
    %v402 = stablehlo.multiply %v367, %v367 : tensor<16xf32>
    %v403 = stablehlo.multiply %v400, %v402 : tensor<16xf32>
    %v404 = stablehlo.add %v401, %v403 : tensor<16xf32>
    %v405 = stablehlo.reshape %v21 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v406 = stablehlo.reshape %v211 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v407 = stablehlo.transpose %v405, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v408 = stablehlo.transpose %v406, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v409 = stablehlo.convolution(%v407, %v408)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v410 = stablehlo.transpose %v409, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v411 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v412 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v413 = stablehlo.multiply %v411, %W3m : tensor<16x16x3x3xf32>
    %v414 = stablehlo.multiply %v412, %v410 : tensor<16x16x3x3xf32>
    %v415 = stablehlo.add %v413, %v414 : tensor<16x16x3x3xf32>
    %v416 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v417 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v418 = stablehlo.multiply %v416, %W3v : tensor<16x16x3x3xf32>
    %v419 = stablehlo.multiply %v410, %v410 : tensor<16x16x3x3xf32>
    %v420 = stablehlo.multiply %v417, %v419 : tensor<16x16x3x3xf32>
    %v421 = stablehlo.add %v418, %v420 : tensor<16x16x3x3xf32>
    %v422 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v423 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v424 = stablehlo.divide %v415, %v422 : tensor<16x16x3x3xf32>
    %v425 = stablehlo.divide %v421, %v423 : tensor<16x16x3x3xf32>
    %v426 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v427 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v428 = stablehlo.sqrt %v425 : tensor<16x16x3x3xf32>
    %v429 = stablehlo.add %v428, %v427 : tensor<16x16x3x3xf32>
    %v430 = stablehlo.divide %v424, %v429 : tensor<16x16x3x3xf32>
    %v431 = stablehlo.multiply %v426, %v430 : tensor<16x16x3x3xf32>
    %v432 = stablehlo.subtract %W3, %v431 : tensor<16x16x3x3xf32>
    %v433 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v434 = stablehlo.multiply %v433, %v426 : tensor<16x16x3x3xf32>
    %v435 = stablehlo.multiply %v434, %W3 : tensor<16x16x3x3xf32>
    %v436 = stablehlo.subtract %v432, %v435 : tensor<16x16x3x3xf32>
    %v437 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v438 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v439 = stablehlo.multiply %v437, %W3m : tensor<16x16x3x3xf32>
    %v440 = stablehlo.multiply %v438, %v410 : tensor<16x16x3x3xf32>
    %v441 = stablehlo.add %v439, %v440 : tensor<16x16x3x3xf32>
    %v442 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v443 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v444 = stablehlo.multiply %v442, %W3v : tensor<16x16x3x3xf32>
    %v445 = stablehlo.multiply %v410, %v410 : tensor<16x16x3x3xf32>
    %v446 = stablehlo.multiply %v443, %v445 : tensor<16x16x3x3xf32>
    %v447 = stablehlo.add %v444, %v446 : tensor<16x16x3x3xf32>
    %v448 = stablehlo.reshape %v211 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v449 = stablehlo.constant dense<0.0> : tensor<f32>
    %v450 = stablehlo.reduce(%v448 init: %v449) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v451 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v452 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v453 = stablehlo.multiply %v451, %cb3m : tensor<16xf32>
    %v454 = stablehlo.multiply %v452, %v450 : tensor<16xf32>
    %v455 = stablehlo.add %v453, %v454 : tensor<16xf32>
    %v456 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v457 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v458 = stablehlo.multiply %v456, %cb3v : tensor<16xf32>
    %v459 = stablehlo.multiply %v450, %v450 : tensor<16xf32>
    %v460 = stablehlo.multiply %v457, %v459 : tensor<16xf32>
    %v461 = stablehlo.add %v458, %v460 : tensor<16xf32>
    %v462 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v463 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v464 = stablehlo.divide %v455, %v462 : tensor<16xf32>
    %v465 = stablehlo.divide %v461, %v463 : tensor<16xf32>
    %v466 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v467 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v468 = stablehlo.sqrt %v465 : tensor<16xf32>
    %v469 = stablehlo.add %v468, %v467 : tensor<16xf32>
    %v470 = stablehlo.divide %v464, %v469 : tensor<16xf32>
    %v471 = stablehlo.multiply %v466, %v470 : tensor<16xf32>
    %v472 = stablehlo.subtract %cb3, %v471 : tensor<16xf32>
    %v473 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v474 = stablehlo.multiply %v473, %v466 : tensor<16xf32>
    %v475 = stablehlo.multiply %v474, %cb3 : tensor<16xf32>
    %v476 = stablehlo.subtract %v472, %v475 : tensor<16xf32>
    %v477 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v478 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v479 = stablehlo.multiply %v477, %cb3m : tensor<16xf32>
    %v480 = stablehlo.multiply %v478, %v450 : tensor<16xf32>
    %v481 = stablehlo.add %v479, %v480 : tensor<16xf32>
    %v482 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v483 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v484 = stablehlo.multiply %v482, %cb3v : tensor<16xf32>
    %v485 = stablehlo.multiply %v450, %v450 : tensor<16xf32>
    %v486 = stablehlo.multiply %v483, %v485 : tensor<16xf32>
    %v487 = stablehlo.add %v484, %v486 : tensor<16xf32>
    %v488 = stablehlo.reshape %v30 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v489 = stablehlo.reshape %v200 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v490 = stablehlo.transpose %v488, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v491 = stablehlo.transpose %v489, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v492 = stablehlo.convolution(%v490, %v491)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v493 = stablehlo.transpose %v492, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v494 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v495 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v496 = stablehlo.multiply %v494, %W4m : tensor<16x16x3x3xf32>
    %v497 = stablehlo.multiply %v495, %v493 : tensor<16x16x3x3xf32>
    %v498 = stablehlo.add %v496, %v497 : tensor<16x16x3x3xf32>
    %v499 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v500 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v501 = stablehlo.multiply %v499, %W4v : tensor<16x16x3x3xf32>
    %v502 = stablehlo.multiply %v493, %v493 : tensor<16x16x3x3xf32>
    %v503 = stablehlo.multiply %v500, %v502 : tensor<16x16x3x3xf32>
    %v504 = stablehlo.add %v501, %v503 : tensor<16x16x3x3xf32>
    %v505 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v506 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v507 = stablehlo.divide %v498, %v505 : tensor<16x16x3x3xf32>
    %v508 = stablehlo.divide %v504, %v506 : tensor<16x16x3x3xf32>
    %v509 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v510 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v511 = stablehlo.sqrt %v508 : tensor<16x16x3x3xf32>
    %v512 = stablehlo.add %v511, %v510 : tensor<16x16x3x3xf32>
    %v513 = stablehlo.divide %v507, %v512 : tensor<16x16x3x3xf32>
    %v514 = stablehlo.multiply %v509, %v513 : tensor<16x16x3x3xf32>
    %v515 = stablehlo.subtract %W4, %v514 : tensor<16x16x3x3xf32>
    %v516 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v517 = stablehlo.multiply %v516, %v509 : tensor<16x16x3x3xf32>
    %v518 = stablehlo.multiply %v517, %W4 : tensor<16x16x3x3xf32>
    %v519 = stablehlo.subtract %v515, %v518 : tensor<16x16x3x3xf32>
    %v520 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v521 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v522 = stablehlo.multiply %v520, %W4m : tensor<16x16x3x3xf32>
    %v523 = stablehlo.multiply %v521, %v493 : tensor<16x16x3x3xf32>
    %v524 = stablehlo.add %v522, %v523 : tensor<16x16x3x3xf32>
    %v525 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v526 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v527 = stablehlo.multiply %v525, %W4v : tensor<16x16x3x3xf32>
    %v528 = stablehlo.multiply %v493, %v493 : tensor<16x16x3x3xf32>
    %v529 = stablehlo.multiply %v526, %v528 : tensor<16x16x3x3xf32>
    %v530 = stablehlo.add %v527, %v529 : tensor<16x16x3x3xf32>
    %v531 = stablehlo.reshape %v200 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v532 = stablehlo.constant dense<0.0> : tensor<f32>
    %v533 = stablehlo.reduce(%v531 init: %v532) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v534 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v535 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v536 = stablehlo.multiply %v534, %cb4m : tensor<16xf32>
    %v537 = stablehlo.multiply %v535, %v533 : tensor<16xf32>
    %v538 = stablehlo.add %v536, %v537 : tensor<16xf32>
    %v539 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v540 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v541 = stablehlo.multiply %v539, %cb4v : tensor<16xf32>
    %v542 = stablehlo.multiply %v533, %v533 : tensor<16xf32>
    %v543 = stablehlo.multiply %v540, %v542 : tensor<16xf32>
    %v544 = stablehlo.add %v541, %v543 : tensor<16xf32>
    %v545 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v546 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v547 = stablehlo.divide %v538, %v545 : tensor<16xf32>
    %v548 = stablehlo.divide %v544, %v546 : tensor<16xf32>
    %v549 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v550 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v551 = stablehlo.sqrt %v548 : tensor<16xf32>
    %v552 = stablehlo.add %v551, %v550 : tensor<16xf32>
    %v553 = stablehlo.divide %v547, %v552 : tensor<16xf32>
    %v554 = stablehlo.multiply %v549, %v553 : tensor<16xf32>
    %v555 = stablehlo.subtract %cb4, %v554 : tensor<16xf32>
    %v556 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v557 = stablehlo.multiply %v556, %v549 : tensor<16xf32>
    %v558 = stablehlo.multiply %v557, %cb4 : tensor<16xf32>
    %v559 = stablehlo.subtract %v555, %v558 : tensor<16xf32>
    %v560 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v561 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v562 = stablehlo.multiply %v560, %cb4m : tensor<16xf32>
    %v563 = stablehlo.multiply %v561, %v533 : tensor<16xf32>
    %v564 = stablehlo.add %v562, %v563 : tensor<16xf32>
    %v565 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v566 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v567 = stablehlo.multiply %v565, %cb4v : tensor<16xf32>
    %v568 = stablehlo.multiply %v533, %v533 : tensor<16xf32>
    %v569 = stablehlo.multiply %v566, %v568 : tensor<16xf32>
    %v570 = stablehlo.add %v567, %v569 : tensor<16xf32>
    %v571 = stablehlo.reshape %v43 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v572 = stablehlo.reshape %v184 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v573 = stablehlo.transpose %v571, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v574 = stablehlo.transpose %v572, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v575 = stablehlo.convolution(%v573, %v574)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<16x32x3x3xf32>
    %v576 = stablehlo.transpose %v575, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v577 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v578 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v579 = stablehlo.multiply %v577, %W5m : tensor<32x16x3x3xf32>
    %v580 = stablehlo.multiply %v578, %v576 : tensor<32x16x3x3xf32>
    %v581 = stablehlo.add %v579, %v580 : tensor<32x16x3x3xf32>
    %v582 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v583 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v584 = stablehlo.multiply %v582, %W5v : tensor<32x16x3x3xf32>
    %v585 = stablehlo.multiply %v576, %v576 : tensor<32x16x3x3xf32>
    %v586 = stablehlo.multiply %v583, %v585 : tensor<32x16x3x3xf32>
    %v587 = stablehlo.add %v584, %v586 : tensor<32x16x3x3xf32>
    %v588 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v589 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v590 = stablehlo.divide %v581, %v588 : tensor<32x16x3x3xf32>
    %v591 = stablehlo.divide %v587, %v589 : tensor<32x16x3x3xf32>
    %v592 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v593 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v594 = stablehlo.sqrt %v591 : tensor<32x16x3x3xf32>
    %v595 = stablehlo.add %v594, %v593 : tensor<32x16x3x3xf32>
    %v596 = stablehlo.divide %v590, %v595 : tensor<32x16x3x3xf32>
    %v597 = stablehlo.multiply %v592, %v596 : tensor<32x16x3x3xf32>
    %v598 = stablehlo.subtract %W5, %v597 : tensor<32x16x3x3xf32>
    %v599 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v600 = stablehlo.multiply %v599, %v592 : tensor<32x16x3x3xf32>
    %v601 = stablehlo.multiply %v600, %W5 : tensor<32x16x3x3xf32>
    %v602 = stablehlo.subtract %v598, %v601 : tensor<32x16x3x3xf32>
    %v603 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v604 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v605 = stablehlo.multiply %v603, %W5m : tensor<32x16x3x3xf32>
    %v606 = stablehlo.multiply %v604, %v576 : tensor<32x16x3x3xf32>
    %v607 = stablehlo.add %v605, %v606 : tensor<32x16x3x3xf32>
    %v608 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v609 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v610 = stablehlo.multiply %v608, %W5v : tensor<32x16x3x3xf32>
    %v611 = stablehlo.multiply %v576, %v576 : tensor<32x16x3x3xf32>
    %v612 = stablehlo.multiply %v609, %v611 : tensor<32x16x3x3xf32>
    %v613 = stablehlo.add %v610, %v612 : tensor<32x16x3x3xf32>
    %v614 = stablehlo.reshape %v184 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v615 = stablehlo.constant dense<0.0> : tensor<f32>
    %v616 = stablehlo.reduce(%v614 init: %v615) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v617 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v618 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v619 = stablehlo.multiply %v617, %cb5m : tensor<32xf32>
    %v620 = stablehlo.multiply %v618, %v616 : tensor<32xf32>
    %v621 = stablehlo.add %v619, %v620 : tensor<32xf32>
    %v622 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v623 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v624 = stablehlo.multiply %v622, %cb5v : tensor<32xf32>
    %v625 = stablehlo.multiply %v616, %v616 : tensor<32xf32>
    %v626 = stablehlo.multiply %v623, %v625 : tensor<32xf32>
    %v627 = stablehlo.add %v624, %v626 : tensor<32xf32>
    %v628 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v629 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v630 = stablehlo.divide %v621, %v628 : tensor<32xf32>
    %v631 = stablehlo.divide %v627, %v629 : tensor<32xf32>
    %v632 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v633 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v634 = stablehlo.sqrt %v631 : tensor<32xf32>
    %v635 = stablehlo.add %v634, %v633 : tensor<32xf32>
    %v636 = stablehlo.divide %v630, %v635 : tensor<32xf32>
    %v637 = stablehlo.multiply %v632, %v636 : tensor<32xf32>
    %v638 = stablehlo.subtract %cb5, %v637 : tensor<32xf32>
    %v639 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v640 = stablehlo.multiply %v639, %v632 : tensor<32xf32>
    %v641 = stablehlo.multiply %v640, %cb5 : tensor<32xf32>
    %v642 = stablehlo.subtract %v638, %v641 : tensor<32xf32>
    %v643 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v644 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v645 = stablehlo.multiply %v643, %cb5m : tensor<32xf32>
    %v646 = stablehlo.multiply %v644, %v616 : tensor<32xf32>
    %v647 = stablehlo.add %v645, %v646 : tensor<32xf32>
    %v648 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v649 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v650 = stablehlo.multiply %v648, %cb5v : tensor<32xf32>
    %v651 = stablehlo.multiply %v616, %v616 : tensor<32xf32>
    %v652 = stablehlo.multiply %v649, %v651 : tensor<32xf32>
    %v653 = stablehlo.add %v650, %v652 : tensor<32xf32>
    %v654 = stablehlo.reshape %v52 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v655 = stablehlo.reshape %v173 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v656 = stablehlo.transpose %v654, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v657 = stablehlo.transpose %v655, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v658 = stablehlo.convolution(%v656, %v657)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<32x32x3x3xf32>
    %v659 = stablehlo.transpose %v658, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v660 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v661 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v662 = stablehlo.multiply %v660, %W6m : tensor<32x32x3x3xf32>
    %v663 = stablehlo.multiply %v661, %v659 : tensor<32x32x3x3xf32>
    %v664 = stablehlo.add %v662, %v663 : tensor<32x32x3x3xf32>
    %v665 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v666 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v667 = stablehlo.multiply %v665, %W6v : tensor<32x32x3x3xf32>
    %v668 = stablehlo.multiply %v659, %v659 : tensor<32x32x3x3xf32>
    %v669 = stablehlo.multiply %v666, %v668 : tensor<32x32x3x3xf32>
    %v670 = stablehlo.add %v667, %v669 : tensor<32x32x3x3xf32>
    %v671 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v672 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v673 = stablehlo.divide %v664, %v671 : tensor<32x32x3x3xf32>
    %v674 = stablehlo.divide %v670, %v672 : tensor<32x32x3x3xf32>
    %v675 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v676 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v677 = stablehlo.sqrt %v674 : tensor<32x32x3x3xf32>
    %v678 = stablehlo.add %v677, %v676 : tensor<32x32x3x3xf32>
    %v679 = stablehlo.divide %v673, %v678 : tensor<32x32x3x3xf32>
    %v680 = stablehlo.multiply %v675, %v679 : tensor<32x32x3x3xf32>
    %v681 = stablehlo.subtract %W6, %v680 : tensor<32x32x3x3xf32>
    %v682 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v683 = stablehlo.multiply %v682, %v675 : tensor<32x32x3x3xf32>
    %v684 = stablehlo.multiply %v683, %W6 : tensor<32x32x3x3xf32>
    %v685 = stablehlo.subtract %v681, %v684 : tensor<32x32x3x3xf32>
    %v686 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v687 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v688 = stablehlo.multiply %v686, %W6m : tensor<32x32x3x3xf32>
    %v689 = stablehlo.multiply %v687, %v659 : tensor<32x32x3x3xf32>
    %v690 = stablehlo.add %v688, %v689 : tensor<32x32x3x3xf32>
    %v691 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v692 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v693 = stablehlo.multiply %v691, %W6v : tensor<32x32x3x3xf32>
    %v694 = stablehlo.multiply %v659, %v659 : tensor<32x32x3x3xf32>
    %v695 = stablehlo.multiply %v692, %v694 : tensor<32x32x3x3xf32>
    %v696 = stablehlo.add %v693, %v695 : tensor<32x32x3x3xf32>
    %v697 = stablehlo.reshape %v173 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v698 = stablehlo.constant dense<0.0> : tensor<f32>
    %v699 = stablehlo.reduce(%v697 init: %v698) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v700 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v701 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v702 = stablehlo.multiply %v700, %cb6m : tensor<32xf32>
    %v703 = stablehlo.multiply %v701, %v699 : tensor<32xf32>
    %v704 = stablehlo.add %v702, %v703 : tensor<32xf32>
    %v705 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v706 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v707 = stablehlo.multiply %v705, %cb6v : tensor<32xf32>
    %v708 = stablehlo.multiply %v699, %v699 : tensor<32xf32>
    %v709 = stablehlo.multiply %v706, %v708 : tensor<32xf32>
    %v710 = stablehlo.add %v707, %v709 : tensor<32xf32>
    %v711 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v712 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v713 = stablehlo.divide %v704, %v711 : tensor<32xf32>
    %v714 = stablehlo.divide %v710, %v712 : tensor<32xf32>
    %v715 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v716 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v717 = stablehlo.sqrt %v714 : tensor<32xf32>
    %v718 = stablehlo.add %v717, %v716 : tensor<32xf32>
    %v719 = stablehlo.divide %v713, %v718 : tensor<32xf32>
    %v720 = stablehlo.multiply %v715, %v719 : tensor<32xf32>
    %v721 = stablehlo.subtract %cb6, %v720 : tensor<32xf32>
    %v722 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v723 = stablehlo.multiply %v722, %v715 : tensor<32xf32>
    %v724 = stablehlo.multiply %v723, %cb6 : tensor<32xf32>
    %v725 = stablehlo.subtract %v721, %v724 : tensor<32xf32>
    %v726 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v727 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v728 = stablehlo.multiply %v726, %cb6m : tensor<32xf32>
    %v729 = stablehlo.multiply %v727, %v699 : tensor<32xf32>
    %v730 = stablehlo.add %v728, %v729 : tensor<32xf32>
    %v731 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v732 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v733 = stablehlo.multiply %v731, %cb6v : tensor<32xf32>
    %v734 = stablehlo.multiply %v699, %v699 : tensor<32xf32>
    %v735 = stablehlo.multiply %v732, %v734 : tensor<32xf32>
    %v736 = stablehlo.add %v733, %v735 : tensor<32xf32>
    %v737 = stablehlo.reshape %v65 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v738 = stablehlo.reshape %v157 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v739 = stablehlo.transpose %v737, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v740 = stablehlo.transpose %v738, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v741 = stablehlo.convolution(%v739, %v740)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v742 = stablehlo.transpose %v741, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v743 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v744 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v745 = stablehlo.multiply %v743, %W7m : tensor<32x32x3x3xf32>
    %v746 = stablehlo.multiply %v744, %v742 : tensor<32x32x3x3xf32>
    %v747 = stablehlo.add %v745, %v746 : tensor<32x32x3x3xf32>
    %v748 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v749 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v750 = stablehlo.multiply %v748, %W7v : tensor<32x32x3x3xf32>
    %v751 = stablehlo.multiply %v742, %v742 : tensor<32x32x3x3xf32>
    %v752 = stablehlo.multiply %v749, %v751 : tensor<32x32x3x3xf32>
    %v753 = stablehlo.add %v750, %v752 : tensor<32x32x3x3xf32>
    %v754 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v755 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v756 = stablehlo.divide %v747, %v754 : tensor<32x32x3x3xf32>
    %v757 = stablehlo.divide %v753, %v755 : tensor<32x32x3x3xf32>
    %v758 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v759 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v760 = stablehlo.sqrt %v757 : tensor<32x32x3x3xf32>
    %v761 = stablehlo.add %v760, %v759 : tensor<32x32x3x3xf32>
    %v762 = stablehlo.divide %v756, %v761 : tensor<32x32x3x3xf32>
    %v763 = stablehlo.multiply %v758, %v762 : tensor<32x32x3x3xf32>
    %v764 = stablehlo.subtract %W7, %v763 : tensor<32x32x3x3xf32>
    %v765 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v766 = stablehlo.multiply %v765, %v758 : tensor<32x32x3x3xf32>
    %v767 = stablehlo.multiply %v766, %W7 : tensor<32x32x3x3xf32>
    %v768 = stablehlo.subtract %v764, %v767 : tensor<32x32x3x3xf32>
    %v769 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v770 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v771 = stablehlo.multiply %v769, %W7m : tensor<32x32x3x3xf32>
    %v772 = stablehlo.multiply %v770, %v742 : tensor<32x32x3x3xf32>
    %v773 = stablehlo.add %v771, %v772 : tensor<32x32x3x3xf32>
    %v774 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v775 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v776 = stablehlo.multiply %v774, %W7v : tensor<32x32x3x3xf32>
    %v777 = stablehlo.multiply %v742, %v742 : tensor<32x32x3x3xf32>
    %v778 = stablehlo.multiply %v775, %v777 : tensor<32x32x3x3xf32>
    %v779 = stablehlo.add %v776, %v778 : tensor<32x32x3x3xf32>
    %v780 = stablehlo.reshape %v157 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v781 = stablehlo.constant dense<0.0> : tensor<f32>
    %v782 = stablehlo.reduce(%v780 init: %v781) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v783 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v784 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v785 = stablehlo.multiply %v783, %cb7m : tensor<32xf32>
    %v786 = stablehlo.multiply %v784, %v782 : tensor<32xf32>
    %v787 = stablehlo.add %v785, %v786 : tensor<32xf32>
    %v788 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v789 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v790 = stablehlo.multiply %v788, %cb7v : tensor<32xf32>
    %v791 = stablehlo.multiply %v782, %v782 : tensor<32xf32>
    %v792 = stablehlo.multiply %v789, %v791 : tensor<32xf32>
    %v793 = stablehlo.add %v790, %v792 : tensor<32xf32>
    %v794 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v795 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v796 = stablehlo.divide %v787, %v794 : tensor<32xf32>
    %v797 = stablehlo.divide %v793, %v795 : tensor<32xf32>
    %v798 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v799 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v800 = stablehlo.sqrt %v797 : tensor<32xf32>
    %v801 = stablehlo.add %v800, %v799 : tensor<32xf32>
    %v802 = stablehlo.divide %v796, %v801 : tensor<32xf32>
    %v803 = stablehlo.multiply %v798, %v802 : tensor<32xf32>
    %v804 = stablehlo.subtract %cb7, %v803 : tensor<32xf32>
    %v805 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v806 = stablehlo.multiply %v805, %v798 : tensor<32xf32>
    %v807 = stablehlo.multiply %v806, %cb7 : tensor<32xf32>
    %v808 = stablehlo.subtract %v804, %v807 : tensor<32xf32>
    %v809 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v810 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v811 = stablehlo.multiply %v809, %cb7m : tensor<32xf32>
    %v812 = stablehlo.multiply %v810, %v782 : tensor<32xf32>
    %v813 = stablehlo.add %v811, %v812 : tensor<32xf32>
    %v814 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v815 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v816 = stablehlo.multiply %v814, %cb7v : tensor<32xf32>
    %v817 = stablehlo.multiply %v782, %v782 : tensor<32xf32>
    %v818 = stablehlo.multiply %v815, %v817 : tensor<32xf32>
    %v819 = stablehlo.add %v816, %v818 : tensor<32xf32>
    %v820 = stablehlo.reshape %v74 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v821 = stablehlo.reshape %v146 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v822 = stablehlo.transpose %v820, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v823 = stablehlo.transpose %v821, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v824 = stablehlo.convolution(%v822, %v823)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v825 = stablehlo.transpose %v824, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v826 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v827 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v828 = stablehlo.multiply %v826, %W8m : tensor<32x32x3x3xf32>
    %v829 = stablehlo.multiply %v827, %v825 : tensor<32x32x3x3xf32>
    %v830 = stablehlo.add %v828, %v829 : tensor<32x32x3x3xf32>
    %v831 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v832 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v833 = stablehlo.multiply %v831, %W8v : tensor<32x32x3x3xf32>
    %v834 = stablehlo.multiply %v825, %v825 : tensor<32x32x3x3xf32>
    %v835 = stablehlo.multiply %v832, %v834 : tensor<32x32x3x3xf32>
    %v836 = stablehlo.add %v833, %v835 : tensor<32x32x3x3xf32>
    %v837 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v838 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v839 = stablehlo.divide %v830, %v837 : tensor<32x32x3x3xf32>
    %v840 = stablehlo.divide %v836, %v838 : tensor<32x32x3x3xf32>
    %v841 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v842 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v843 = stablehlo.sqrt %v840 : tensor<32x32x3x3xf32>
    %v844 = stablehlo.add %v843, %v842 : tensor<32x32x3x3xf32>
    %v845 = stablehlo.divide %v839, %v844 : tensor<32x32x3x3xf32>
    %v846 = stablehlo.multiply %v841, %v845 : tensor<32x32x3x3xf32>
    %v847 = stablehlo.subtract %W8, %v846 : tensor<32x32x3x3xf32>
    %v848 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v849 = stablehlo.multiply %v848, %v841 : tensor<32x32x3x3xf32>
    %v850 = stablehlo.multiply %v849, %W8 : tensor<32x32x3x3xf32>
    %v851 = stablehlo.subtract %v847, %v850 : tensor<32x32x3x3xf32>
    %v852 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v853 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v854 = stablehlo.multiply %v852, %W8m : tensor<32x32x3x3xf32>
    %v855 = stablehlo.multiply %v853, %v825 : tensor<32x32x3x3xf32>
    %v856 = stablehlo.add %v854, %v855 : tensor<32x32x3x3xf32>
    %v857 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v858 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v859 = stablehlo.multiply %v857, %W8v : tensor<32x32x3x3xf32>
    %v860 = stablehlo.multiply %v825, %v825 : tensor<32x32x3x3xf32>
    %v861 = stablehlo.multiply %v858, %v860 : tensor<32x32x3x3xf32>
    %v862 = stablehlo.add %v859, %v861 : tensor<32x32x3x3xf32>
    %v863 = stablehlo.reshape %v146 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v864 = stablehlo.constant dense<0.0> : tensor<f32>
    %v865 = stablehlo.reduce(%v863 init: %v864) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v866 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v867 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v868 = stablehlo.multiply %v866, %cb8m : tensor<32xf32>
    %v869 = stablehlo.multiply %v867, %v865 : tensor<32xf32>
    %v870 = stablehlo.add %v868, %v869 : tensor<32xf32>
    %v871 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v872 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v873 = stablehlo.multiply %v871, %cb8v : tensor<32xf32>
    %v874 = stablehlo.multiply %v865, %v865 : tensor<32xf32>
    %v875 = stablehlo.multiply %v872, %v874 : tensor<32xf32>
    %v876 = stablehlo.add %v873, %v875 : tensor<32xf32>
    %v877 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v878 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v879 = stablehlo.divide %v870, %v877 : tensor<32xf32>
    %v880 = stablehlo.divide %v876, %v878 : tensor<32xf32>
    %v881 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v882 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v883 = stablehlo.sqrt %v880 : tensor<32xf32>
    %v884 = stablehlo.add %v883, %v882 : tensor<32xf32>
    %v885 = stablehlo.divide %v879, %v884 : tensor<32xf32>
    %v886 = stablehlo.multiply %v881, %v885 : tensor<32xf32>
    %v887 = stablehlo.subtract %cb8, %v886 : tensor<32xf32>
    %v888 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v889 = stablehlo.multiply %v888, %v881 : tensor<32xf32>
    %v890 = stablehlo.multiply %v889, %cb8 : tensor<32xf32>
    %v891 = stablehlo.subtract %v887, %v890 : tensor<32xf32>
    %v892 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v893 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v894 = stablehlo.multiply %v892, %cb8m : tensor<32xf32>
    %v895 = stablehlo.multiply %v893, %v865 : tensor<32xf32>
    %v896 = stablehlo.add %v894, %v895 : tensor<32xf32>
    %v897 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v898 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v899 = stablehlo.multiply %v897, %cb8v : tensor<32xf32>
    %v900 = stablehlo.multiply %v865, %v865 : tensor<32xf32>
    %v901 = stablehlo.multiply %v898, %v900 : tensor<32xf32>
    %v902 = stablehlo.add %v899, %v901 : tensor<32xf32>
    %v903 = stablehlo.dot_general %v87, %v132, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x512xf32>) -> tensor<128x512xf32>
    %v904 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v905 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v906 = stablehlo.multiply %v904, %W9m : tensor<128x512xf32>
    %v907 = stablehlo.multiply %v905, %v903 : tensor<128x512xf32>
    %v908 = stablehlo.add %v906, %v907 : tensor<128x512xf32>
    %v909 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v910 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v911 = stablehlo.multiply %v909, %W9v : tensor<128x512xf32>
    %v912 = stablehlo.multiply %v903, %v903 : tensor<128x512xf32>
    %v913 = stablehlo.multiply %v910, %v912 : tensor<128x512xf32>
    %v914 = stablehlo.add %v911, %v913 : tensor<128x512xf32>
    %v915 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v916 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v917 = stablehlo.divide %v908, %v915 : tensor<128x512xf32>
    %v918 = stablehlo.divide %v914, %v916 : tensor<128x512xf32>
    %v919 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v920 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v921 = stablehlo.sqrt %v918 : tensor<128x512xf32>
    %v922 = stablehlo.add %v921, %v920 : tensor<128x512xf32>
    %v923 = stablehlo.divide %v917, %v922 : tensor<128x512xf32>
    %v924 = stablehlo.multiply %v919, %v923 : tensor<128x512xf32>
    %v925 = stablehlo.subtract %W9, %v924 : tensor<128x512xf32>
    %v926 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v927 = stablehlo.multiply %v926, %v919 : tensor<128x512xf32>
    %v928 = stablehlo.multiply %v927, %W9 : tensor<128x512xf32>
    %v929 = stablehlo.subtract %v925, %v928 : tensor<128x512xf32>
    %v930 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v931 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v932 = stablehlo.multiply %v930, %W9m : tensor<128x512xf32>
    %v933 = stablehlo.multiply %v931, %v903 : tensor<128x512xf32>
    %v934 = stablehlo.add %v932, %v933 : tensor<128x512xf32>
    %v935 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v936 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v937 = stablehlo.multiply %v935, %W9v : tensor<128x512xf32>
    %v938 = stablehlo.multiply %v903, %v903 : tensor<128x512xf32>
    %v939 = stablehlo.multiply %v936, %v938 : tensor<128x512xf32>
    %v940 = stablehlo.add %v937, %v939 : tensor<128x512xf32>
    %v941 = stablehlo.constant dense<0.0> : tensor<f32>
    %v942 = stablehlo.reduce(%v132 init: %v941) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v943 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v944 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v945 = stablehlo.multiply %v943, %b9m : tensor<512xf32>
    %v946 = stablehlo.multiply %v944, %v942 : tensor<512xf32>
    %v947 = stablehlo.add %v945, %v946 : tensor<512xf32>
    %v948 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v949 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v950 = stablehlo.multiply %v948, %b9v : tensor<512xf32>
    %v951 = stablehlo.multiply %v942, %v942 : tensor<512xf32>
    %v952 = stablehlo.multiply %v949, %v951 : tensor<512xf32>
    %v953 = stablehlo.add %v950, %v952 : tensor<512xf32>
    %v954 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v955 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v956 = stablehlo.divide %v947, %v954 : tensor<512xf32>
    %v957 = stablehlo.divide %v953, %v955 : tensor<512xf32>
    %v958 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v959 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v960 = stablehlo.sqrt %v957 : tensor<512xf32>
    %v961 = stablehlo.add %v960, %v959 : tensor<512xf32>
    %v962 = stablehlo.divide %v956, %v961 : tensor<512xf32>
    %v963 = stablehlo.multiply %v958, %v962 : tensor<512xf32>
    %v964 = stablehlo.subtract %b9, %v963 : tensor<512xf32>
    %v965 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v966 = stablehlo.multiply %v965, %v958 : tensor<512xf32>
    %v967 = stablehlo.multiply %v966, %b9 : tensor<512xf32>
    %v968 = stablehlo.subtract %v964, %v967 : tensor<512xf32>
    %v969 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v970 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v971 = stablehlo.multiply %v969, %b9m : tensor<512xf32>
    %v972 = stablehlo.multiply %v970, %v942 : tensor<512xf32>
    %v973 = stablehlo.add %v971, %v972 : tensor<512xf32>
    %v974 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v975 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v976 = stablehlo.multiply %v974, %b9v : tensor<512xf32>
    %v977 = stablehlo.multiply %v942, %v942 : tensor<512xf32>
    %v978 = stablehlo.multiply %v975, %v977 : tensor<512xf32>
    %v979 = stablehlo.add %v976, %v978 : tensor<512xf32>
    %v980 = stablehlo.dot_general %v94, %v123, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<512x512xf32>
    %v981 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v982 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v983 = stablehlo.multiply %v981, %Wam : tensor<512x512xf32>
    %v984 = stablehlo.multiply %v982, %v980 : tensor<512x512xf32>
    %v985 = stablehlo.add %v983, %v984 : tensor<512x512xf32>
    %v986 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v987 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v988 = stablehlo.multiply %v986, %Wav : tensor<512x512xf32>
    %v989 = stablehlo.multiply %v980, %v980 : tensor<512x512xf32>
    %v990 = stablehlo.multiply %v987, %v989 : tensor<512x512xf32>
    %v991 = stablehlo.add %v988, %v990 : tensor<512x512xf32>
    %v992 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v993 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v994 = stablehlo.divide %v985, %v992 : tensor<512x512xf32>
    %v995 = stablehlo.divide %v991, %v993 : tensor<512x512xf32>
    %v996 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v997 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v998 = stablehlo.sqrt %v995 : tensor<512x512xf32>
    %v999 = stablehlo.add %v998, %v997 : tensor<512x512xf32>
    %v1000 = stablehlo.divide %v994, %v999 : tensor<512x512xf32>
    %v1001 = stablehlo.multiply %v996, %v1000 : tensor<512x512xf32>
    %v1002 = stablehlo.subtract %Wa, %v1001 : tensor<512x512xf32>
    %v1003 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1004 = stablehlo.multiply %v1003, %v996 : tensor<512x512xf32>
    %v1005 = stablehlo.multiply %v1004, %Wa : tensor<512x512xf32>
    %v1006 = stablehlo.subtract %v1002, %v1005 : tensor<512x512xf32>
    %v1007 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1008 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1009 = stablehlo.multiply %v1007, %Wam : tensor<512x512xf32>
    %v1010 = stablehlo.multiply %v1008, %v980 : tensor<512x512xf32>
    %v1011 = stablehlo.add %v1009, %v1010 : tensor<512x512xf32>
    %v1012 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1013 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1014 = stablehlo.multiply %v1012, %Wav : tensor<512x512xf32>
    %v1015 = stablehlo.multiply %v980, %v980 : tensor<512x512xf32>
    %v1016 = stablehlo.multiply %v1013, %v1015 : tensor<512x512xf32>
    %v1017 = stablehlo.add %v1014, %v1016 : tensor<512x512xf32>
    %v1018 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1019 = stablehlo.reduce(%v123 init: %v1018) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v1020 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1021 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1022 = stablehlo.multiply %v1020, %bam : tensor<512xf32>
    %v1023 = stablehlo.multiply %v1021, %v1019 : tensor<512xf32>
    %v1024 = stablehlo.add %v1022, %v1023 : tensor<512xf32>
    %v1025 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1026 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1027 = stablehlo.multiply %v1025, %bav : tensor<512xf32>
    %v1028 = stablehlo.multiply %v1019, %v1019 : tensor<512xf32>
    %v1029 = stablehlo.multiply %v1026, %v1028 : tensor<512xf32>
    %v1030 = stablehlo.add %v1027, %v1029 : tensor<512xf32>
    %v1031 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1032 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1033 = stablehlo.divide %v1024, %v1031 : tensor<512xf32>
    %v1034 = stablehlo.divide %v1030, %v1032 : tensor<512xf32>
    %v1035 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1036 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1037 = stablehlo.sqrt %v1034 : tensor<512xf32>
    %v1038 = stablehlo.add %v1037, %v1036 : tensor<512xf32>
    %v1039 = stablehlo.divide %v1033, %v1038 : tensor<512xf32>
    %v1040 = stablehlo.multiply %v1035, %v1039 : tensor<512xf32>
    %v1041 = stablehlo.subtract %ba, %v1040 : tensor<512xf32>
    %v1042 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1043 = stablehlo.multiply %v1042, %v1035 : tensor<512xf32>
    %v1044 = stablehlo.multiply %v1043, %ba : tensor<512xf32>
    %v1045 = stablehlo.subtract %v1041, %v1044 : tensor<512xf32>
    %v1046 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1047 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1048 = stablehlo.multiply %v1046, %bam : tensor<512xf32>
    %v1049 = stablehlo.multiply %v1047, %v1019 : tensor<512xf32>
    %v1050 = stablehlo.add %v1048, %v1049 : tensor<512xf32>
    %v1051 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1052 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1053 = stablehlo.multiply %v1051, %bav : tensor<512xf32>
    %v1054 = stablehlo.multiply %v1019, %v1019 : tensor<512xf32>
    %v1055 = stablehlo.multiply %v1052, %v1054 : tensor<512xf32>
    %v1056 = stablehlo.add %v1053, %v1055 : tensor<512xf32>
    %v1057 = stablehlo.dot_general %v101, %v114, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x10xf32>) -> tensor<512x10xf32>
    %v1058 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1059 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1060 = stablehlo.multiply %v1058, %Wbm : tensor<512x10xf32>
    %v1061 = stablehlo.multiply %v1059, %v1057 : tensor<512x10xf32>
    %v1062 = stablehlo.add %v1060, %v1061 : tensor<512x10xf32>
    %v1063 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1064 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1065 = stablehlo.multiply %v1063, %Wbv : tensor<512x10xf32>
    %v1066 = stablehlo.multiply %v1057, %v1057 : tensor<512x10xf32>
    %v1067 = stablehlo.multiply %v1064, %v1066 : tensor<512x10xf32>
    %v1068 = stablehlo.add %v1065, %v1067 : tensor<512x10xf32>
    %v1069 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1070 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1071 = stablehlo.divide %v1062, %v1069 : tensor<512x10xf32>
    %v1072 = stablehlo.divide %v1068, %v1070 : tensor<512x10xf32>
    %v1073 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1074 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1075 = stablehlo.sqrt %v1072 : tensor<512x10xf32>
    %v1076 = stablehlo.add %v1075, %v1074 : tensor<512x10xf32>
    %v1077 = stablehlo.divide %v1071, %v1076 : tensor<512x10xf32>
    %v1078 = stablehlo.multiply %v1073, %v1077 : tensor<512x10xf32>
    %v1079 = stablehlo.subtract %Wb, %v1078 : tensor<512x10xf32>
    %v1080 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1081 = stablehlo.multiply %v1080, %v1073 : tensor<512x10xf32>
    %v1082 = stablehlo.multiply %v1081, %Wb : tensor<512x10xf32>
    %v1083 = stablehlo.subtract %v1079, %v1082 : tensor<512x10xf32>
    %v1084 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1085 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1086 = stablehlo.multiply %v1084, %Wbm : tensor<512x10xf32>
    %v1087 = stablehlo.multiply %v1085, %v1057 : tensor<512x10xf32>
    %v1088 = stablehlo.add %v1086, %v1087 : tensor<512x10xf32>
    %v1089 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1090 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1091 = stablehlo.multiply %v1089, %Wbv : tensor<512x10xf32>
    %v1092 = stablehlo.multiply %v1057, %v1057 : tensor<512x10xf32>
    %v1093 = stablehlo.multiply %v1090, %v1092 : tensor<512x10xf32>
    %v1094 = stablehlo.add %v1091, %v1093 : tensor<512x10xf32>
    %v1095 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1096 = stablehlo.reduce(%v114 init: %v1095) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1097 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1098 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1099 = stablehlo.multiply %v1097, %bbm : tensor<10xf32>
    %v1100 = stablehlo.multiply %v1098, %v1096 : tensor<10xf32>
    %v1101 = stablehlo.add %v1099, %v1100 : tensor<10xf32>
    %v1102 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1103 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1104 = stablehlo.multiply %v1102, %bbv : tensor<10xf32>
    %v1105 = stablehlo.multiply %v1096, %v1096 : tensor<10xf32>
    %v1106 = stablehlo.multiply %v1103, %v1105 : tensor<10xf32>
    %v1107 = stablehlo.add %v1104, %v1106 : tensor<10xf32>
    %v1108 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1109 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1110 = stablehlo.divide %v1101, %v1108 : tensor<10xf32>
    %v1111 = stablehlo.divide %v1107, %v1109 : tensor<10xf32>
    %v1112 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1113 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1114 = stablehlo.sqrt %v1111 : tensor<10xf32>
    %v1115 = stablehlo.add %v1114, %v1113 : tensor<10xf32>
    %v1116 = stablehlo.divide %v1110, %v1115 : tensor<10xf32>
    %v1117 = stablehlo.multiply %v1112, %v1116 : tensor<10xf32>
    %v1118 = stablehlo.subtract %bb, %v1117 : tensor<10xf32>
    %v1119 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1120 = stablehlo.multiply %v1119, %v1112 : tensor<10xf32>
    %v1121 = stablehlo.multiply %v1120, %bb : tensor<10xf32>
    %v1122 = stablehlo.subtract %v1118, %v1121 : tensor<10xf32>
    %v1123 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1124 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1125 = stablehlo.multiply %v1123, %bbm : tensor<10xf32>
    %v1126 = stablehlo.multiply %v1124, %v1096 : tensor<10xf32>
    %v1127 = stablehlo.add %v1125, %v1126 : tensor<10xf32>
    %v1128 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1129 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1130 = stablehlo.multiply %v1128, %bbv : tensor<10xf32>
    %v1131 = stablehlo.multiply %v1096, %v1096 : tensor<10xf32>
    %v1132 = stablehlo.multiply %v1129, %v1131 : tensor<10xf32>
    %v1133 = stablehlo.add %v1130, %v1132 : tensor<10xf32>
    return %v270, %v310, %v353, %v393, %v436, %v476, %v519, %v559, %v602, %v642, %v685, %v725, %v768, %v808, %v851, %v891, %v929, %v968, %v1006, %v1045, %v1083, %v1122, %v275, %v315, %v358, %v398, %v441, %v481, %v524, %v564, %v607, %v647, %v690, %v730, %v773, %v813, %v856, %v896, %v934, %v973, %v1011, %v1050, %v1088, %v1127, %v281, %v321, %v364, %v404, %v447, %v487, %v530, %v570, %v613, %v653, %v696, %v736, %v779, %v819, %v862, %v902, %v940, %v979, %v1017, %v1056, %v1094, %v1133, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
