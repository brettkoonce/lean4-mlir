module @m {
  func.func @cifar8wb_mom_train_step(%x: tensor<128x3072xf32>, %W1: tensor<16x3x3x3xf32>, %cb1: tensor<16xf32>, %W2: tensor<16x16x3x3xf32>, %cb2: tensor<16xf32>, %W3: tensor<16x16x3x3xf32>, %cb3: tensor<16xf32>, %W4: tensor<16x16x3x3xf32>, %cb4: tensor<16xf32>, %W5: tensor<32x16x3x3xf32>, %cb5: tensor<32xf32>, %W6: tensor<32x32x3x3xf32>, %cb6: tensor<32xf32>, %W7: tensor<32x32x3x3xf32>, %cb7: tensor<32xf32>, %W8: tensor<32x32x3x3xf32>, %cb8: tensor<32xf32>, %W9: tensor<128x512xf32>, %b9: tensor<512xf32>, %Wa: tensor<512x512xf32>, %ba: tensor<512xf32>, %Wb: tensor<512x10xf32>, %bb: tensor<10xf32>, %W1m: tensor<16x3x3x3xf32>, %cb1m: tensor<16xf32>, %W2m: tensor<16x16x3x3xf32>, %cb2m: tensor<16xf32>, %W3m: tensor<16x16x3x3xf32>, %cb3m: tensor<16xf32>, %W4m: tensor<16x16x3x3xf32>, %cb4m: tensor<16xf32>, %W5m: tensor<32x16x3x3xf32>, %cb5m: tensor<32xf32>, %W6m: tensor<32x32x3x3xf32>, %cb6m: tensor<32xf32>, %W7m: tensor<32x32x3x3xf32>, %cb7m: tensor<32xf32>, %W8m: tensor<32x32x3x3xf32>, %cb8m: tensor<32xf32>, %W9m: tensor<128x512xf32>, %b9m: tensor<512xf32>, %Wam: tensor<512x512xf32>, %bam: tensor<512xf32>, %Wbm: tensor<512x10xf32>, %bbm: tensor<10xf32>, %W1v: tensor<16x3x3x3xf32>, %cb1v: tensor<16xf32>, %W2v: tensor<16x16x3x3xf32>, %cb2v: tensor<16xf32>, %W3v: tensor<16x16x3x3xf32>, %cb3v: tensor<16xf32>, %W4v: tensor<16x16x3x3xf32>, %cb4v: tensor<16xf32>, %W5v: tensor<32x16x3x3xf32>, %cb5v: tensor<32xf32>, %W6v: tensor<32x32x3x3xf32>, %cb6v: tensor<32xf32>, %W7v: tensor<32x32x3x3xf32>, %cb7v: tensor<32xf32>, %W8v: tensor<32x32x3x3xf32>, %cb8v: tensor<32xf32>, %W9v: tensor<128x512xf32>, %b9v: tensor<512xf32>, %Wav: tensor<512x512xf32>, %bav: tensor<512xf32>, %Wbv: tensor<512x10xf32>, %bbv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: tensor<128x10xf32>) -> (tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>) {
    // ── cifar8 AdamW train step: every line is pretty(verified AST node), except the
    //    marked report-only loss + the %bc passthroughs ──
    %lzero = stablehlo.constant dense<0.0> : tensor<f32>
    %b1 = stablehlo.constant dense<0.9> : tensor<f32>
    %ob1 = stablehlo.constant dense<0.1> : tensor<f32>
    %b2 = stablehlo.constant dense<0.999> : tensor<f32>
    %ob2 = stablehlo.constant dense<0.001> : tensor<f32>
    %eps = stablehlo.constant dense<1.0e-8> : tensor<f32>
    %wd = stablehlo.constant dense<0.0001> : tensor<f32>
    %mu = stablehlo.constant dense<0.9> : tensor<f32>
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
    %v235 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v236 = stablehlo.multiply %v235, %W1v : tensor<16x3x3x3xf32>
    %v237 = stablehlo.add %v236, %v234 : tensor<16x3x3x3xf32>
    %v238 = stablehlo.multiply %v235, %v237 : tensor<16x3x3x3xf32>
    %v239 = stablehlo.add %v238, %v234 : tensor<16x3x3x3xf32>
    %v240 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v241 = stablehlo.multiply %v240, %v239 : tensor<16x3x3x3xf32>
    %v242 = stablehlo.subtract %W1, %v241 : tensor<16x3x3x3xf32>
    %v243 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v244 = stablehlo.multiply %v243, %W1v : tensor<16x3x3x3xf32>
    %v245 = stablehlo.add %v244, %v234 : tensor<16x3x3x3xf32>
    %v246 = stablehlo.reshape %v228 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v247 = stablehlo.constant dense<0.0> : tensor<f32>
    %v248 = stablehlo.reduce(%v246 init: %v247) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v249 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v250 = stablehlo.multiply %v249, %cb1v : tensor<16xf32>
    %v251 = stablehlo.add %v250, %v248 : tensor<16xf32>
    %v252 = stablehlo.multiply %v249, %v251 : tensor<16xf32>
    %v253 = stablehlo.add %v252, %v248 : tensor<16xf32>
    %v254 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v255 = stablehlo.multiply %v254, %v253 : tensor<16xf32>
    %v256 = stablehlo.subtract %cb1, %v255 : tensor<16xf32>
    %v257 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v258 = stablehlo.multiply %v257, %cb1v : tensor<16xf32>
    %v259 = stablehlo.add %v258, %v248 : tensor<16xf32>
    %v260 = stablehlo.reshape %v8 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v261 = stablehlo.reshape %v217 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v262 = stablehlo.transpose %v260, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v263 = stablehlo.transpose %v261, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v264 = stablehlo.convolution(%v262, %v263)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x16x3x3xf32>
    %v265 = stablehlo.transpose %v264, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v266 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v267 = stablehlo.multiply %v266, %W2v : tensor<16x16x3x3xf32>
    %v268 = stablehlo.add %v267, %v265 : tensor<16x16x3x3xf32>
    %v269 = stablehlo.multiply %v266, %v268 : tensor<16x16x3x3xf32>
    %v270 = stablehlo.add %v269, %v265 : tensor<16x16x3x3xf32>
    %v271 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v272 = stablehlo.multiply %v271, %v270 : tensor<16x16x3x3xf32>
    %v273 = stablehlo.subtract %W2, %v272 : tensor<16x16x3x3xf32>
    %v274 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v275 = stablehlo.multiply %v274, %W2v : tensor<16x16x3x3xf32>
    %v276 = stablehlo.add %v275, %v265 : tensor<16x16x3x3xf32>
    %v277 = stablehlo.reshape %v217 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v278 = stablehlo.constant dense<0.0> : tensor<f32>
    %v279 = stablehlo.reduce(%v277 init: %v278) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v280 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v281 = stablehlo.multiply %v280, %cb2v : tensor<16xf32>
    %v282 = stablehlo.add %v281, %v279 : tensor<16xf32>
    %v283 = stablehlo.multiply %v280, %v282 : tensor<16xf32>
    %v284 = stablehlo.add %v283, %v279 : tensor<16xf32>
    %v285 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v286 = stablehlo.multiply %v285, %v284 : tensor<16xf32>
    %v287 = stablehlo.subtract %cb2, %v286 : tensor<16xf32>
    %v288 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v289 = stablehlo.multiply %v288, %cb2v : tensor<16xf32>
    %v290 = stablehlo.add %v289, %v279 : tensor<16xf32>
    %v291 = stablehlo.reshape %v21 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v292 = stablehlo.reshape %v201 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v293 = stablehlo.transpose %v291, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v294 = stablehlo.transpose %v292, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v295 = stablehlo.convolution(%v293, %v294)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v296 = stablehlo.transpose %v295, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v297 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v298 = stablehlo.multiply %v297, %W3v : tensor<16x16x3x3xf32>
    %v299 = stablehlo.add %v298, %v296 : tensor<16x16x3x3xf32>
    %v300 = stablehlo.multiply %v297, %v299 : tensor<16x16x3x3xf32>
    %v301 = stablehlo.add %v300, %v296 : tensor<16x16x3x3xf32>
    %v302 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v303 = stablehlo.multiply %v302, %v301 : tensor<16x16x3x3xf32>
    %v304 = stablehlo.subtract %W3, %v303 : tensor<16x16x3x3xf32>
    %v305 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v306 = stablehlo.multiply %v305, %W3v : tensor<16x16x3x3xf32>
    %v307 = stablehlo.add %v306, %v296 : tensor<16x16x3x3xf32>
    %v308 = stablehlo.reshape %v201 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v309 = stablehlo.constant dense<0.0> : tensor<f32>
    %v310 = stablehlo.reduce(%v308 init: %v309) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v311 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v312 = stablehlo.multiply %v311, %cb3v : tensor<16xf32>
    %v313 = stablehlo.add %v312, %v310 : tensor<16xf32>
    %v314 = stablehlo.multiply %v311, %v313 : tensor<16xf32>
    %v315 = stablehlo.add %v314, %v310 : tensor<16xf32>
    %v316 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v317 = stablehlo.multiply %v316, %v315 : tensor<16xf32>
    %v318 = stablehlo.subtract %cb3, %v317 : tensor<16xf32>
    %v319 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v320 = stablehlo.multiply %v319, %cb3v : tensor<16xf32>
    %v321 = stablehlo.add %v320, %v310 : tensor<16xf32>
    %v322 = stablehlo.reshape %v30 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v323 = stablehlo.reshape %v190 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v324 = stablehlo.transpose %v322, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v325 = stablehlo.transpose %v323, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v326 = stablehlo.convolution(%v324, %v325)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v327 = stablehlo.transpose %v326, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v328 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v329 = stablehlo.multiply %v328, %W4v : tensor<16x16x3x3xf32>
    %v330 = stablehlo.add %v329, %v327 : tensor<16x16x3x3xf32>
    %v331 = stablehlo.multiply %v328, %v330 : tensor<16x16x3x3xf32>
    %v332 = stablehlo.add %v331, %v327 : tensor<16x16x3x3xf32>
    %v333 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v334 = stablehlo.multiply %v333, %v332 : tensor<16x16x3x3xf32>
    %v335 = stablehlo.subtract %W4, %v334 : tensor<16x16x3x3xf32>
    %v336 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v337 = stablehlo.multiply %v336, %W4v : tensor<16x16x3x3xf32>
    %v338 = stablehlo.add %v337, %v327 : tensor<16x16x3x3xf32>
    %v339 = stablehlo.reshape %v190 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v340 = stablehlo.constant dense<0.0> : tensor<f32>
    %v341 = stablehlo.reduce(%v339 init: %v340) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v342 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v343 = stablehlo.multiply %v342, %cb4v : tensor<16xf32>
    %v344 = stablehlo.add %v343, %v341 : tensor<16xf32>
    %v345 = stablehlo.multiply %v342, %v344 : tensor<16xf32>
    %v346 = stablehlo.add %v345, %v341 : tensor<16xf32>
    %v347 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v348 = stablehlo.multiply %v347, %v346 : tensor<16xf32>
    %v349 = stablehlo.subtract %cb4, %v348 : tensor<16xf32>
    %v350 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v351 = stablehlo.multiply %v350, %cb4v : tensor<16xf32>
    %v352 = stablehlo.add %v351, %v341 : tensor<16xf32>
    %v353 = stablehlo.reshape %v43 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v354 = stablehlo.reshape %v174 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v355 = stablehlo.transpose %v353, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v356 = stablehlo.transpose %v354, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v357 = stablehlo.convolution(%v355, %v356)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<16x32x3x3xf32>
    %v358 = stablehlo.transpose %v357, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v359 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v360 = stablehlo.multiply %v359, %W5v : tensor<32x16x3x3xf32>
    %v361 = stablehlo.add %v360, %v358 : tensor<32x16x3x3xf32>
    %v362 = stablehlo.multiply %v359, %v361 : tensor<32x16x3x3xf32>
    %v363 = stablehlo.add %v362, %v358 : tensor<32x16x3x3xf32>
    %v364 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v365 = stablehlo.multiply %v364, %v363 : tensor<32x16x3x3xf32>
    %v366 = stablehlo.subtract %W5, %v365 : tensor<32x16x3x3xf32>
    %v367 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v368 = stablehlo.multiply %v367, %W5v : tensor<32x16x3x3xf32>
    %v369 = stablehlo.add %v368, %v358 : tensor<32x16x3x3xf32>
    %v370 = stablehlo.reshape %v174 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v371 = stablehlo.constant dense<0.0> : tensor<f32>
    %v372 = stablehlo.reduce(%v370 init: %v371) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v373 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v374 = stablehlo.multiply %v373, %cb5v : tensor<32xf32>
    %v375 = stablehlo.add %v374, %v372 : tensor<32xf32>
    %v376 = stablehlo.multiply %v373, %v375 : tensor<32xf32>
    %v377 = stablehlo.add %v376, %v372 : tensor<32xf32>
    %v378 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v379 = stablehlo.multiply %v378, %v377 : tensor<32xf32>
    %v380 = stablehlo.subtract %cb5, %v379 : tensor<32xf32>
    %v381 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v382 = stablehlo.multiply %v381, %cb5v : tensor<32xf32>
    %v383 = stablehlo.add %v382, %v372 : tensor<32xf32>
    %v384 = stablehlo.reshape %v52 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v385 = stablehlo.reshape %v163 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v386 = stablehlo.transpose %v384, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v387 = stablehlo.transpose %v385, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v388 = stablehlo.convolution(%v386, %v387)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<32x32x3x3xf32>
    %v389 = stablehlo.transpose %v388, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v390 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v391 = stablehlo.multiply %v390, %W6v : tensor<32x32x3x3xf32>
    %v392 = stablehlo.add %v391, %v389 : tensor<32x32x3x3xf32>
    %v393 = stablehlo.multiply %v390, %v392 : tensor<32x32x3x3xf32>
    %v394 = stablehlo.add %v393, %v389 : tensor<32x32x3x3xf32>
    %v395 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v396 = stablehlo.multiply %v395, %v394 : tensor<32x32x3x3xf32>
    %v397 = stablehlo.subtract %W6, %v396 : tensor<32x32x3x3xf32>
    %v398 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v399 = stablehlo.multiply %v398, %W6v : tensor<32x32x3x3xf32>
    %v400 = stablehlo.add %v399, %v389 : tensor<32x32x3x3xf32>
    %v401 = stablehlo.reshape %v163 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v402 = stablehlo.constant dense<0.0> : tensor<f32>
    %v403 = stablehlo.reduce(%v401 init: %v402) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v404 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v405 = stablehlo.multiply %v404, %cb6v : tensor<32xf32>
    %v406 = stablehlo.add %v405, %v403 : tensor<32xf32>
    %v407 = stablehlo.multiply %v404, %v406 : tensor<32xf32>
    %v408 = stablehlo.add %v407, %v403 : tensor<32xf32>
    %v409 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v410 = stablehlo.multiply %v409, %v408 : tensor<32xf32>
    %v411 = stablehlo.subtract %cb6, %v410 : tensor<32xf32>
    %v412 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v413 = stablehlo.multiply %v412, %cb6v : tensor<32xf32>
    %v414 = stablehlo.add %v413, %v403 : tensor<32xf32>
    %v415 = stablehlo.reshape %v65 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v416 = stablehlo.reshape %v147 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v417 = stablehlo.transpose %v415, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v418 = stablehlo.transpose %v416, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v419 = stablehlo.convolution(%v417, %v418)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v420 = stablehlo.transpose %v419, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v421 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v422 = stablehlo.multiply %v421, %W7v : tensor<32x32x3x3xf32>
    %v423 = stablehlo.add %v422, %v420 : tensor<32x32x3x3xf32>
    %v424 = stablehlo.multiply %v421, %v423 : tensor<32x32x3x3xf32>
    %v425 = stablehlo.add %v424, %v420 : tensor<32x32x3x3xf32>
    %v426 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v427 = stablehlo.multiply %v426, %v425 : tensor<32x32x3x3xf32>
    %v428 = stablehlo.subtract %W7, %v427 : tensor<32x32x3x3xf32>
    %v429 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v430 = stablehlo.multiply %v429, %W7v : tensor<32x32x3x3xf32>
    %v431 = stablehlo.add %v430, %v420 : tensor<32x32x3x3xf32>
    %v432 = stablehlo.reshape %v147 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v433 = stablehlo.constant dense<0.0> : tensor<f32>
    %v434 = stablehlo.reduce(%v432 init: %v433) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v435 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v436 = stablehlo.multiply %v435, %cb7v : tensor<32xf32>
    %v437 = stablehlo.add %v436, %v434 : tensor<32xf32>
    %v438 = stablehlo.multiply %v435, %v437 : tensor<32xf32>
    %v439 = stablehlo.add %v438, %v434 : tensor<32xf32>
    %v440 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v441 = stablehlo.multiply %v440, %v439 : tensor<32xf32>
    %v442 = stablehlo.subtract %cb7, %v441 : tensor<32xf32>
    %v443 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v444 = stablehlo.multiply %v443, %cb7v : tensor<32xf32>
    %v445 = stablehlo.add %v444, %v434 : tensor<32xf32>
    %v446 = stablehlo.reshape %v74 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v447 = stablehlo.reshape %v136 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v448 = stablehlo.transpose %v446, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v449 = stablehlo.transpose %v447, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v450 = stablehlo.convolution(%v448, %v449)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v451 = stablehlo.transpose %v450, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v452 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v453 = stablehlo.multiply %v452, %W8v : tensor<32x32x3x3xf32>
    %v454 = stablehlo.add %v453, %v451 : tensor<32x32x3x3xf32>
    %v455 = stablehlo.multiply %v452, %v454 : tensor<32x32x3x3xf32>
    %v456 = stablehlo.add %v455, %v451 : tensor<32x32x3x3xf32>
    %v457 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v458 = stablehlo.multiply %v457, %v456 : tensor<32x32x3x3xf32>
    %v459 = stablehlo.subtract %W8, %v458 : tensor<32x32x3x3xf32>
    %v460 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v461 = stablehlo.multiply %v460, %W8v : tensor<32x32x3x3xf32>
    %v462 = stablehlo.add %v461, %v451 : tensor<32x32x3x3xf32>
    %v463 = stablehlo.reshape %v136 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v464 = stablehlo.constant dense<0.0> : tensor<f32>
    %v465 = stablehlo.reduce(%v463 init: %v464) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v466 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v467 = stablehlo.multiply %v466, %cb8v : tensor<32xf32>
    %v468 = stablehlo.add %v467, %v465 : tensor<32xf32>
    %v469 = stablehlo.multiply %v466, %v468 : tensor<32xf32>
    %v470 = stablehlo.add %v469, %v465 : tensor<32xf32>
    %v471 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v472 = stablehlo.multiply %v471, %v470 : tensor<32xf32>
    %v473 = stablehlo.subtract %cb8, %v472 : tensor<32xf32>
    %v474 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v475 = stablehlo.multiply %v474, %cb8v : tensor<32xf32>
    %v476 = stablehlo.add %v475, %v465 : tensor<32xf32>
    %v477 = stablehlo.dot_general %v87, %v122, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x512xf32>) -> tensor<128x512xf32>
    %v478 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v479 = stablehlo.multiply %v478, %W9v : tensor<128x512xf32>
    %v480 = stablehlo.add %v479, %v477 : tensor<128x512xf32>
    %v481 = stablehlo.multiply %v478, %v480 : tensor<128x512xf32>
    %v482 = stablehlo.add %v481, %v477 : tensor<128x512xf32>
    %v483 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v484 = stablehlo.multiply %v483, %v482 : tensor<128x512xf32>
    %v485 = stablehlo.subtract %W9, %v484 : tensor<128x512xf32>
    %v486 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v487 = stablehlo.multiply %v486, %W9v : tensor<128x512xf32>
    %v488 = stablehlo.add %v487, %v477 : tensor<128x512xf32>
    %v489 = stablehlo.constant dense<0.0> : tensor<f32>
    %v490 = stablehlo.reduce(%v122 init: %v489) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v491 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v492 = stablehlo.multiply %v491, %b9v : tensor<512xf32>
    %v493 = stablehlo.add %v492, %v490 : tensor<512xf32>
    %v494 = stablehlo.multiply %v491, %v493 : tensor<512xf32>
    %v495 = stablehlo.add %v494, %v490 : tensor<512xf32>
    %v496 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v497 = stablehlo.multiply %v496, %v495 : tensor<512xf32>
    %v498 = stablehlo.subtract %b9, %v497 : tensor<512xf32>
    %v499 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v500 = stablehlo.multiply %v499, %b9v : tensor<512xf32>
    %v501 = stablehlo.add %v500, %v490 : tensor<512xf32>
    %v502 = stablehlo.dot_general %v92, %v116, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<512x512xf32>
    %v503 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v504 = stablehlo.multiply %v503, %Wav : tensor<512x512xf32>
    %v505 = stablehlo.add %v504, %v502 : tensor<512x512xf32>
    %v506 = stablehlo.multiply %v503, %v505 : tensor<512x512xf32>
    %v507 = stablehlo.add %v506, %v502 : tensor<512x512xf32>
    %v508 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v509 = stablehlo.multiply %v508, %v507 : tensor<512x512xf32>
    %v510 = stablehlo.subtract %Wa, %v509 : tensor<512x512xf32>
    %v511 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v512 = stablehlo.multiply %v511, %Wav : tensor<512x512xf32>
    %v513 = stablehlo.add %v512, %v502 : tensor<512x512xf32>
    %v514 = stablehlo.constant dense<0.0> : tensor<f32>
    %v515 = stablehlo.reduce(%v116 init: %v514) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v516 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v517 = stablehlo.multiply %v516, %bav : tensor<512xf32>
    %v518 = stablehlo.add %v517, %v515 : tensor<512xf32>
    %v519 = stablehlo.multiply %v516, %v518 : tensor<512xf32>
    %v520 = stablehlo.add %v519, %v515 : tensor<512xf32>
    %v521 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v522 = stablehlo.multiply %v521, %v520 : tensor<512xf32>
    %v523 = stablehlo.subtract %ba, %v522 : tensor<512xf32>
    %v524 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v525 = stablehlo.multiply %v524, %bav : tensor<512xf32>
    %v526 = stablehlo.add %v525, %v515 : tensor<512xf32>
    %v527 = stablehlo.dot_general %v97, %v110, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x10xf32>) -> tensor<512x10xf32>
    %v528 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v529 = stablehlo.multiply %v528, %Wbv : tensor<512x10xf32>
    %v530 = stablehlo.add %v529, %v527 : tensor<512x10xf32>
    %v531 = stablehlo.multiply %v528, %v530 : tensor<512x10xf32>
    %v532 = stablehlo.add %v531, %v527 : tensor<512x10xf32>
    %v533 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v534 = stablehlo.multiply %v533, %v532 : tensor<512x10xf32>
    %v535 = stablehlo.subtract %Wb, %v534 : tensor<512x10xf32>
    %v536 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v537 = stablehlo.multiply %v536, %Wbv : tensor<512x10xf32>
    %v538 = stablehlo.add %v537, %v527 : tensor<512x10xf32>
    %v539 = stablehlo.constant dense<0.0> : tensor<f32>
    %v540 = stablehlo.reduce(%v110 init: %v539) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v541 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v542 = stablehlo.multiply %v541, %bbv : tensor<10xf32>
    %v543 = stablehlo.add %v542, %v540 : tensor<10xf32>
    %v544 = stablehlo.multiply %v541, %v543 : tensor<10xf32>
    %v545 = stablehlo.add %v544, %v540 : tensor<10xf32>
    %v546 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v547 = stablehlo.multiply %v546, %v545 : tensor<10xf32>
    %v548 = stablehlo.subtract %bb, %v547 : tensor<10xf32>
    %v549 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v550 = stablehlo.multiply %v549, %bbv : tensor<10xf32>
    %v551 = stablehlo.add %v550, %v540 : tensor<10xf32>
    return %v242, %v256, %v273, %v287, %v304, %v318, %v335, %v349, %v366, %v380, %v397, %v411, %v428, %v442, %v459, %v473, %v485, %v498, %v510, %v523, %v535, %v548, %W1m, %cb1m, %W2m, %cb2m, %W3m, %cb3m, %W4m, %cb4m, %W5m, %cb5m, %W6m, %cb6m, %W7m, %cb7m, %W8m, %cb8m, %W9m, %b9m, %Wam, %bam, %Wbm, %bbm, %v245, %v259, %v276, %v290, %v307, %v321, %v338, %v352, %v369, %v383, %v400, %v414, %v431, %v445, %v462, %v476, %v488, %v501, %v513, %v526, %v538, %v551, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
