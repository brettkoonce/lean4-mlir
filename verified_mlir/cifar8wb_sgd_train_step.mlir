module @m {
  func.func @cifar8wb_sgd_train_step(%x: tensor<128x3072xf32>, %W1: tensor<16x3x3x3xf32>, %cb1: tensor<16xf32>, %W2: tensor<16x16x3x3xf32>, %cb2: tensor<16xf32>, %W3: tensor<16x16x3x3xf32>, %cb3: tensor<16xf32>, %W4: tensor<16x16x3x3xf32>, %cb4: tensor<16xf32>, %W5: tensor<32x16x3x3xf32>, %cb5: tensor<32xf32>, %W6: tensor<32x32x3x3xf32>, %cb6: tensor<32xf32>, %W7: tensor<32x32x3x3xf32>, %cb7: tensor<32xf32>, %W8: tensor<32x32x3x3xf32>, %cb8: tensor<32xf32>, %W9: tensor<128x512xf32>, %b9: tensor<512xf32>, %Wa: tensor<512x512xf32>, %ba: tensor<512xf32>, %Wb: tensor<512x10xf32>, %bb: tensor<10xf32>, %W1m: tensor<16x3x3x3xf32>, %cb1m: tensor<16xf32>, %W2m: tensor<16x16x3x3xf32>, %cb2m: tensor<16xf32>, %W3m: tensor<16x16x3x3xf32>, %cb3m: tensor<16xf32>, %W4m: tensor<16x16x3x3xf32>, %cb4m: tensor<16xf32>, %W5m: tensor<32x16x3x3xf32>, %cb5m: tensor<32xf32>, %W6m: tensor<32x32x3x3xf32>, %cb6m: tensor<32xf32>, %W7m: tensor<32x32x3x3xf32>, %cb7m: tensor<32xf32>, %W8m: tensor<32x32x3x3xf32>, %cb8m: tensor<32xf32>, %W9m: tensor<128x512xf32>, %b9m: tensor<512xf32>, %Wam: tensor<512x512xf32>, %bam: tensor<512xf32>, %Wbm: tensor<512x10xf32>, %bbm: tensor<10xf32>, %W1v: tensor<16x3x3x3xf32>, %cb1v: tensor<16xf32>, %W2v: tensor<16x16x3x3xf32>, %cb2v: tensor<16xf32>, %W3v: tensor<16x16x3x3xf32>, %cb3v: tensor<16xf32>, %W4v: tensor<16x16x3x3xf32>, %cb4v: tensor<16xf32>, %W5v: tensor<32x16x3x3xf32>, %cb5v: tensor<32xf32>, %W6v: tensor<32x32x3x3xf32>, %cb6v: tensor<32xf32>, %W7v: tensor<32x32x3x3xf32>, %cb7v: tensor<32xf32>, %W8v: tensor<32x32x3x3xf32>, %cb8v: tensor<32xf32>, %W9v: tensor<128x512xf32>, %b9v: tensor<512xf32>, %Wav: tensor<512x512xf32>, %bav: tensor<512xf32>, %Wbv: tensor<512x10xf32>, %bbv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: tensor<128x10xf32>) -> (tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>) {
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
    %v91 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v92 = stablehlo.maximum %v90, %v91 : tensor<128x512xf32>
    %v93 = stablehlo.dot_general %v92, %Wa, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %v94 = stablehlo.broadcast_in_dim %ba, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v95 = stablehlo.add %v93, %v94 : tensor<128x512xf32>
    %v96 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v97 = stablehlo.maximum %v95, %v96 : tensor<128x512xf32>
    %v98 = stablehlo.dot_general %v97, %Wb, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x10xf32>) -> tensor<128x10xf32>
    %v99 = stablehlo.broadcast_in_dim %bb, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v100 = stablehlo.add %v98, %v99 : tensor<128x10xf32>
    %v101 = stablehlo.reshape %v100 : (tensor<128x10xf32>) -> tensor<128x1x10xf32>
    %v102 = stablehlo.constant dense<0.0> : tensor<f32>
    %v103 = stablehlo.exponential %v101 : tensor<128x1x10xf32>
    %v104 = stablehlo.reduce(%v103 init: %v102) applies stablehlo.add across dimensions = [2] : (tensor<128x1x10xf32>, tensor<f32>) -> tensor<128x1xf32>
    %v105 = stablehlo.broadcast_in_dim %v104, dims = [0, 1] : (tensor<128x1xf32>) -> tensor<128x1x10xf32>
    %v106 = stablehlo.divide %v103, %v105 : tensor<128x1x10xf32>
    %v107 = stablehlo.reshape %v106 : (tensor<128x1x10xf32>) -> tensor<128x10xf32>
    %v108 = stablehlo.subtract %v107, %onehot : tensor<128x10xf32>
    %v109 = stablehlo.constant dense<0.0078125> : tensor<128x10xf32>
    %v110 = stablehlo.multiply %v108, %v109 : tensor<128x10xf32>
    // ── report-only scalar loss (NOT pretty(AST): the kit has no rank-0 loss op; it
    //    feeds no parameter, only the driver's progress line) ──
    %llog = stablehlo.log %v107 : tensor<128x10xf32>
    %ohll = stablehlo.multiply %onehot, %llog : tensor<128x10xf32>
    %csum = stablehlo.reduce(%ohll init: %lzero) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<f32>
    %cneg = stablehlo.negate %csum : tensor<f32>
    %lbf = stablehlo.constant dense<128.0> : tensor<f32>
    %loss = stablehlo.divide %cneg, %lbf : tensor<f32>
    %v111 = stablehlo.reshape %v110 : (tensor<128x10xf32>) -> tensor<128x1x10xf32>
    %v112 = stablehlo.dot_general %v111, %Wb, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x10xf32>, tensor<512x10xf32>) -> tensor<128x1x512xf32>
    %v113 = stablehlo.reshape %v112 : (tensor<128x1x512xf32>) -> tensor<128x512xf32>
    %v114 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v115 = stablehlo.compare GT, %v95, %v114 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v116 = stablehlo.select %v115, %v113, %v114 : tensor<128x512xi1>, tensor<128x512xf32>
    %v117 = stablehlo.reshape %v116 : (tensor<128x512xf32>) -> tensor<128x1x512xf32>
    %v118 = stablehlo.dot_general %v117, %Wa, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x512xf32>, tensor<512x512xf32>) -> tensor<128x1x512xf32>
    %v119 = stablehlo.reshape %v118 : (tensor<128x1x512xf32>) -> tensor<128x512xf32>
    %v120 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v121 = stablehlo.compare GT, %v90, %v120 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v122 = stablehlo.select %v121, %v119, %v120 : tensor<128x512xi1>, tensor<128x512xf32>
    %v123 = stablehlo.reshape %v122 : (tensor<128x512xf32>) -> tensor<128x1x512xf32>
    %v124 = stablehlo.dot_general %v123, %W9, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x512xf32>, tensor<128x512xf32>) -> tensor<128x1x128xf32>
    %v125 = stablehlo.reshape %v124 : (tensor<128x1x128xf32>) -> tensor<128x128xf32>
    %v126 = stablehlo.reshape %v83 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v127 = stablehlo.reshape %v125 : (tensor<128x128xf32>) -> tensor<128x32x2x2xf32>
    %v128 = stablehlo.constant dense<0.0> : tensor<f32>
    %v129 = "stablehlo.select_and_scatter"(%v126, %v127, %v128) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x4x4xf32>, tensor<128x32x2x2xf32>, tensor<f32>) -> tensor<128x32x4x4xf32>
    %v130 = stablehlo.reshape %v129 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v131 = stablehlo.reshape %v130 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v132 = stablehlo.reshape %v79 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v133 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v134 = stablehlo.compare GT, %v132, %v133 : (tensor<128x32x4x4xf32>, tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xi1>
    %v135 = stablehlo.select %v134, %v131, %v133 : tensor<128x32x4x4xi1>, tensor<128x32x4x4xf32>
    %v136 = stablehlo.reshape %v135 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v137 = stablehlo.reshape %v136 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v138 = stablehlo.reverse %W8, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v139 = stablehlo.transpose %v138, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v140 = stablehlo.convolution(%v137, %v139)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v141 = stablehlo.reshape %v140 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v142 = stablehlo.reshape %v141 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v143 = stablehlo.reshape %v70 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v144 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v145 = stablehlo.compare GT, %v143, %v144 : (tensor<128x32x4x4xf32>, tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xi1>
    %v146 = stablehlo.select %v145, %v142, %v144 : tensor<128x32x4x4xi1>, tensor<128x32x4x4xf32>
    %v147 = stablehlo.reshape %v146 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v148 = stablehlo.reshape %v147 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v149 = stablehlo.reverse %W7, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v150 = stablehlo.transpose %v149, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v151 = stablehlo.convolution(%v148, %v150)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v152 = stablehlo.reshape %v151 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v153 = stablehlo.reshape %v61 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v154 = stablehlo.reshape %v152 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v155 = stablehlo.constant dense<0.0> : tensor<f32>
    %v156 = "stablehlo.select_and_scatter"(%v153, %v154, %v155) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x8x8xf32>, tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32x8x8xf32>
    %v157 = stablehlo.reshape %v156 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v158 = stablehlo.reshape %v157 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v159 = stablehlo.reshape %v57 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v160 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v161 = stablehlo.compare GT, %v159, %v160 : (tensor<128x32x8x8xf32>, tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xi1>
    %v162 = stablehlo.select %v161, %v158, %v160 : tensor<128x32x8x8xi1>, tensor<128x32x8x8xf32>
    %v163 = stablehlo.reshape %v162 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v164 = stablehlo.reshape %v163 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v165 = stablehlo.reverse %W6, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v166 = stablehlo.transpose %v165, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v167 = stablehlo.convolution(%v164, %v166)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x8x8xf32>
    %v168 = stablehlo.reshape %v167 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v169 = stablehlo.reshape %v168 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v170 = stablehlo.reshape %v48 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v171 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v172 = stablehlo.compare GT, %v170, %v171 : (tensor<128x32x8x8xf32>, tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xi1>
    %v173 = stablehlo.select %v172, %v169, %v171 : tensor<128x32x8x8xi1>, tensor<128x32x8x8xf32>
    %v174 = stablehlo.reshape %v173 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v175 = stablehlo.reshape %v174 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v176 = stablehlo.reverse %W5, dims = [2, 3] : tensor<32x16x3x3xf32>
    %v177 = stablehlo.transpose %v176, dims = [1, 0, 2, 3] : (tensor<32x16x3x3xf32>) -> tensor<16x32x3x3xf32>
    %v178 = stablehlo.convolution(%v175, %v177)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<16x32x3x3xf32>) -> tensor<128x16x8x8xf32>
    %v179 = stablehlo.reshape %v178 : (tensor<128x16x8x8xf32>) -> tensor<128x1024xf32>
    %v180 = stablehlo.reshape %v39 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v181 = stablehlo.reshape %v179 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v182 = stablehlo.constant dense<0.0> : tensor<f32>
    %v183 = "stablehlo.select_and_scatter"(%v180, %v181, %v182) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x16x16xf32>, tensor<128x16x8x8xf32>, tensor<f32>) -> tensor<128x16x16x16xf32>
    %v184 = stablehlo.reshape %v183 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v185 = stablehlo.reshape %v184 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v186 = stablehlo.reshape %v35 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v187 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v188 = stablehlo.compare GT, %v186, %v187 : (tensor<128x16x16x16xf32>, tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xi1>
    %v189 = stablehlo.select %v188, %v185, %v187 : tensor<128x16x16x16xi1>, tensor<128x16x16x16xf32>
    %v190 = stablehlo.reshape %v189 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v191 = stablehlo.reshape %v190 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v192 = stablehlo.reverse %W4, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v193 = stablehlo.transpose %v192, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v194 = stablehlo.convolution(%v191, %v193)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v195 = stablehlo.reshape %v194 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v196 = stablehlo.reshape %v195 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v197 = stablehlo.reshape %v26 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v198 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v199 = stablehlo.compare GT, %v197, %v198 : (tensor<128x16x16x16xf32>, tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xi1>
    %v200 = stablehlo.select %v199, %v196, %v198 : tensor<128x16x16x16xi1>, tensor<128x16x16x16xf32>
    %v201 = stablehlo.reshape %v200 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v202 = stablehlo.reshape %v201 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v203 = stablehlo.reverse %W3, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v204 = stablehlo.transpose %v203, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v205 = stablehlo.convolution(%v202, %v204)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v206 = stablehlo.reshape %v205 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v207 = stablehlo.reshape %v17 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v208 = stablehlo.reshape %v206 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v209 = stablehlo.constant dense<0.0> : tensor<f32>
    %v210 = "stablehlo.select_and_scatter"(%v207, %v208, %v209) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x32x32xf32>, tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x32x32xf32>
    %v211 = stablehlo.reshape %v210 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v212 = stablehlo.reshape %v211 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v213 = stablehlo.reshape %v13 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v214 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v215 = stablehlo.compare GT, %v213, %v214 : (tensor<128x16x32x32xf32>, tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xi1>
    %v216 = stablehlo.select %v215, %v212, %v214 : tensor<128x16x32x32xi1>, tensor<128x16x32x32xf32>
    %v217 = stablehlo.reshape %v216 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v218 = stablehlo.reshape %v217 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v219 = stablehlo.reverse %W2, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v220 = stablehlo.transpose %v219, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v221 = stablehlo.convolution(%v218, %v220)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x32x32xf32>
    %v222 = stablehlo.reshape %v221 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v223 = stablehlo.reshape %v222 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v224 = stablehlo.reshape %v4 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v225 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v226 = stablehlo.compare GT, %v224, %v225 : (tensor<128x16x32x32xf32>, tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xi1>
    %v227 = stablehlo.select %v226, %v223, %v225 : tensor<128x16x32x32xi1>, tensor<128x16x32x32xf32>
    %v228 = stablehlo.reshape %v227 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v229 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v230 = stablehlo.reshape %v228 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v231 = stablehlo.transpose %v229, dims = [1, 0, 2, 3] : (tensor<128x3x32x32xf32>) -> tensor<3x128x32x32xf32>
    %v232 = stablehlo.transpose %v230, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v233 = stablehlo.convolution(%v231, %v232)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<3x16x3x3xf32>
    %v234 = stablehlo.transpose %v233, dims = [1, 0, 2, 3] : (tensor<3x16x3x3xf32>) -> tensor<16x3x3x3xf32>
    %v235 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v236 = stablehlo.multiply %v235, %v234 : tensor<16x3x3x3xf32>
    %v237 = stablehlo.subtract %W1, %v236 : tensor<16x3x3x3xf32>
    %v238 = stablehlo.reshape %v228 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v239 = stablehlo.constant dense<0.0> : tensor<f32>
    %v240 = stablehlo.reduce(%v238 init: %v239) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v241 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v242 = stablehlo.multiply %v241, %v240 : tensor<16xf32>
    %v243 = stablehlo.subtract %cb1, %v242 : tensor<16xf32>
    %v244 = stablehlo.reshape %v8 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v245 = stablehlo.reshape %v217 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v246 = stablehlo.transpose %v244, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v247 = stablehlo.transpose %v245, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v248 = stablehlo.convolution(%v246, %v247)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x16x3x3xf32>
    %v249 = stablehlo.transpose %v248, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v250 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v251 = stablehlo.multiply %v250, %v249 : tensor<16x16x3x3xf32>
    %v252 = stablehlo.subtract %W2, %v251 : tensor<16x16x3x3xf32>
    %v253 = stablehlo.reshape %v217 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v254 = stablehlo.constant dense<0.0> : tensor<f32>
    %v255 = stablehlo.reduce(%v253 init: %v254) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v256 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v257 = stablehlo.multiply %v256, %v255 : tensor<16xf32>
    %v258 = stablehlo.subtract %cb2, %v257 : tensor<16xf32>
    %v259 = stablehlo.reshape %v21 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v260 = stablehlo.reshape %v201 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v261 = stablehlo.transpose %v259, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v262 = stablehlo.transpose %v260, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v263 = stablehlo.convolution(%v261, %v262)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v264 = stablehlo.transpose %v263, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v265 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v266 = stablehlo.multiply %v265, %v264 : tensor<16x16x3x3xf32>
    %v267 = stablehlo.subtract %W3, %v266 : tensor<16x16x3x3xf32>
    %v268 = stablehlo.reshape %v201 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v269 = stablehlo.constant dense<0.0> : tensor<f32>
    %v270 = stablehlo.reduce(%v268 init: %v269) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v271 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v272 = stablehlo.multiply %v271, %v270 : tensor<16xf32>
    %v273 = stablehlo.subtract %cb3, %v272 : tensor<16xf32>
    %v274 = stablehlo.reshape %v30 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v275 = stablehlo.reshape %v190 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v276 = stablehlo.transpose %v274, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v277 = stablehlo.transpose %v275, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v278 = stablehlo.convolution(%v276, %v277)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v279 = stablehlo.transpose %v278, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v280 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v281 = stablehlo.multiply %v280, %v279 : tensor<16x16x3x3xf32>
    %v282 = stablehlo.subtract %W4, %v281 : tensor<16x16x3x3xf32>
    %v283 = stablehlo.reshape %v190 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v284 = stablehlo.constant dense<0.0> : tensor<f32>
    %v285 = stablehlo.reduce(%v283 init: %v284) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v286 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v287 = stablehlo.multiply %v286, %v285 : tensor<16xf32>
    %v288 = stablehlo.subtract %cb4, %v287 : tensor<16xf32>
    %v289 = stablehlo.reshape %v43 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v290 = stablehlo.reshape %v174 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v291 = stablehlo.transpose %v289, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v292 = stablehlo.transpose %v290, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v293 = stablehlo.convolution(%v291, %v292)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<16x32x3x3xf32>
    %v294 = stablehlo.transpose %v293, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v295 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v296 = stablehlo.multiply %v295, %v294 : tensor<32x16x3x3xf32>
    %v297 = stablehlo.subtract %W5, %v296 : tensor<32x16x3x3xf32>
    %v298 = stablehlo.reshape %v174 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v299 = stablehlo.constant dense<0.0> : tensor<f32>
    %v300 = stablehlo.reduce(%v298 init: %v299) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v301 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v302 = stablehlo.multiply %v301, %v300 : tensor<32xf32>
    %v303 = stablehlo.subtract %cb5, %v302 : tensor<32xf32>
    %v304 = stablehlo.reshape %v52 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v305 = stablehlo.reshape %v163 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v306 = stablehlo.transpose %v304, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v307 = stablehlo.transpose %v305, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v308 = stablehlo.convolution(%v306, %v307)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<32x32x3x3xf32>
    %v309 = stablehlo.transpose %v308, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v310 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v311 = stablehlo.multiply %v310, %v309 : tensor<32x32x3x3xf32>
    %v312 = stablehlo.subtract %W6, %v311 : tensor<32x32x3x3xf32>
    %v313 = stablehlo.reshape %v163 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v314 = stablehlo.constant dense<0.0> : tensor<f32>
    %v315 = stablehlo.reduce(%v313 init: %v314) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v316 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v317 = stablehlo.multiply %v316, %v315 : tensor<32xf32>
    %v318 = stablehlo.subtract %cb6, %v317 : tensor<32xf32>
    %v319 = stablehlo.reshape %v65 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v320 = stablehlo.reshape %v147 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v321 = stablehlo.transpose %v319, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v322 = stablehlo.transpose %v320, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v323 = stablehlo.convolution(%v321, %v322)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v324 = stablehlo.transpose %v323, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v325 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v326 = stablehlo.multiply %v325, %v324 : tensor<32x32x3x3xf32>
    %v327 = stablehlo.subtract %W7, %v326 : tensor<32x32x3x3xf32>
    %v328 = stablehlo.reshape %v147 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v329 = stablehlo.constant dense<0.0> : tensor<f32>
    %v330 = stablehlo.reduce(%v328 init: %v329) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v331 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v332 = stablehlo.multiply %v331, %v330 : tensor<32xf32>
    %v333 = stablehlo.subtract %cb7, %v332 : tensor<32xf32>
    %v334 = stablehlo.reshape %v74 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v335 = stablehlo.reshape %v136 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v336 = stablehlo.transpose %v334, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v337 = stablehlo.transpose %v335, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v338 = stablehlo.convolution(%v336, %v337)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v339 = stablehlo.transpose %v338, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v340 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v341 = stablehlo.multiply %v340, %v339 : tensor<32x32x3x3xf32>
    %v342 = stablehlo.subtract %W8, %v341 : tensor<32x32x3x3xf32>
    %v343 = stablehlo.reshape %v136 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v344 = stablehlo.constant dense<0.0> : tensor<f32>
    %v345 = stablehlo.reduce(%v343 init: %v344) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v346 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v347 = stablehlo.multiply %v346, %v345 : tensor<32xf32>
    %v348 = stablehlo.subtract %cb8, %v347 : tensor<32xf32>
    %v349 = stablehlo.dot_general %v87, %v122, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x512xf32>) -> tensor<128x512xf32>
    %v350 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v351 = stablehlo.multiply %v350, %v349 : tensor<128x512xf32>
    %v352 = stablehlo.subtract %W9, %v351 : tensor<128x512xf32>
    %v353 = stablehlo.constant dense<0.0> : tensor<f32>
    %v354 = stablehlo.reduce(%v122 init: %v353) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v355 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v356 = stablehlo.multiply %v355, %v354 : tensor<512xf32>
    %v357 = stablehlo.subtract %b9, %v356 : tensor<512xf32>
    %v358 = stablehlo.dot_general %v92, %v116, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<512x512xf32>
    %v359 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v360 = stablehlo.multiply %v359, %v358 : tensor<512x512xf32>
    %v361 = stablehlo.subtract %Wa, %v360 : tensor<512x512xf32>
    %v362 = stablehlo.constant dense<0.0> : tensor<f32>
    %v363 = stablehlo.reduce(%v116 init: %v362) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v364 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v365 = stablehlo.multiply %v364, %v363 : tensor<512xf32>
    %v366 = stablehlo.subtract %ba, %v365 : tensor<512xf32>
    %v367 = stablehlo.dot_general %v97, %v110, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x10xf32>) -> tensor<512x10xf32>
    %v368 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v369 = stablehlo.multiply %v368, %v367 : tensor<512x10xf32>
    %v370 = stablehlo.subtract %Wb, %v369 : tensor<512x10xf32>
    %v371 = stablehlo.constant dense<0.0> : tensor<f32>
    %v372 = stablehlo.reduce(%v110 init: %v371) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v373 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v374 = stablehlo.multiply %v373, %v372 : tensor<10xf32>
    %v375 = stablehlo.subtract %bb, %v374 : tensor<10xf32>
    return %v237, %v243, %v252, %v258, %v267, %v273, %v282, %v288, %v297, %v303, %v312, %v318, %v327, %v333, %v342, %v348, %v352, %v357, %v361, %v366, %v370, %v375, %W1m, %cb1m, %W2m, %cb2m, %W3m, %cb3m, %W4m, %cb4m, %W5m, %cb5m, %W6m, %cb6m, %W7m, %cb7m, %W8m, %cb8m, %W9m, %b9m, %Wam, %bam, %Wbm, %bbm, %W1v, %cb1v, %W2v, %cb2v, %W3v, %cb3v, %W4v, %cb4v, %W5v, %cb5v, %W6v, %cb6v, %W7v, %cb7v, %W8v, %cb8v, %W9v, %b9v, %Wav, %bav, %Wbv, %bbv, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
