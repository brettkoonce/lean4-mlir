module @m {
  func.func @cifar8b_adam_train_step(%x: tensor<128x3072xf32>, %W1: tensor<16x3x3x3xf32>, %cb1: tensor<16xf32>, %W2: tensor<16x16x3x3xf32>, %cb2: tensor<16xf32>, %W3: tensor<16x16x3x3xf32>, %cb3: tensor<16xf32>, %W4: tensor<16x16x3x3xf32>, %cb4: tensor<16xf32>, %W5: tensor<32x16x3x3xf32>, %cb5: tensor<32xf32>, %W6: tensor<32x32x3x3xf32>, %cb6: tensor<32xf32>, %W7: tensor<32x32x3x3xf32>, %cb7: tensor<32xf32>, %W8: tensor<32x32x3x3xf32>, %cb8: tensor<32xf32>, %W9: tensor<128x64xf32>, %b9: tensor<64xf32>, %Wa: tensor<64x64xf32>, %ba: tensor<64xf32>, %Wb: tensor<64x10xf32>, %bb: tensor<10xf32>, %W1m: tensor<16x3x3x3xf32>, %cb1m: tensor<16xf32>, %W2m: tensor<16x16x3x3xf32>, %cb2m: tensor<16xf32>, %W3m: tensor<16x16x3x3xf32>, %cb3m: tensor<16xf32>, %W4m: tensor<16x16x3x3xf32>, %cb4m: tensor<16xf32>, %W5m: tensor<32x16x3x3xf32>, %cb5m: tensor<32xf32>, %W6m: tensor<32x32x3x3xf32>, %cb6m: tensor<32xf32>, %W7m: tensor<32x32x3x3xf32>, %cb7m: tensor<32xf32>, %W8m: tensor<32x32x3x3xf32>, %cb8m: tensor<32xf32>, %W9m: tensor<128x64xf32>, %b9m: tensor<64xf32>, %Wam: tensor<64x64xf32>, %bam: tensor<64xf32>, %Wbm: tensor<64x10xf32>, %bbm: tensor<10xf32>, %W1v: tensor<16x3x3x3xf32>, %cb1v: tensor<16xf32>, %W2v: tensor<16x16x3x3xf32>, %cb2v: tensor<16xf32>, %W3v: tensor<16x16x3x3xf32>, %cb3v: tensor<16xf32>, %W4v: tensor<16x16x3x3xf32>, %cb4v: tensor<16xf32>, %W5v: tensor<32x16x3x3xf32>, %cb5v: tensor<32xf32>, %W6v: tensor<32x32x3x3xf32>, %cb6v: tensor<32xf32>, %W7v: tensor<32x32x3x3xf32>, %cb7v: tensor<32xf32>, %W8v: tensor<32x32x3x3xf32>, %cb8v: tensor<32xf32>, %W9v: tensor<128x64xf32>, %b9v: tensor<64xf32>, %Wav: tensor<64x64xf32>, %bav: tensor<64xf32>, %Wbv: tensor<64x10xf32>, %bbv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: tensor<128x10xf32>) -> (tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>) {
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
    %v112 = stablehlo.dot_general %v111, %Wb, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x10xf32>, tensor<64x10xf32>) -> tensor<128x1x64xf32>
    %v113 = stablehlo.reshape %v112 : (tensor<128x1x64xf32>) -> tensor<128x64xf32>
    %v114 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v115 = stablehlo.compare GT, %v95, %v114 : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xi1>
    %v116 = stablehlo.select %v115, %v113, %v114 : tensor<128x64xi1>, tensor<128x64xf32>
    %v117 = stablehlo.reshape %v116 : (tensor<128x64xf32>) -> tensor<128x1x64xf32>
    %v118 = stablehlo.dot_general %v117, %Wa, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x64xf32>, tensor<64x64xf32>) -> tensor<128x1x64xf32>
    %v119 = stablehlo.reshape %v118 : (tensor<128x1x64xf32>) -> tensor<128x64xf32>
    %v120 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v121 = stablehlo.compare GT, %v90, %v120 : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xi1>
    %v122 = stablehlo.select %v121, %v119, %v120 : tensor<128x64xi1>, tensor<128x64xf32>
    %v123 = stablehlo.reshape %v122 : (tensor<128x64xf32>) -> tensor<128x1x64xf32>
    %v124 = stablehlo.dot_general %v123, %W9, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x64xf32>, tensor<128x64xf32>) -> tensor<128x1x128xf32>
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
    %v235 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v236 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v237 = stablehlo.multiply %v235, %W1m : tensor<16x3x3x3xf32>
    %v238 = stablehlo.multiply %v236, %v234 : tensor<16x3x3x3xf32>
    %v239 = stablehlo.add %v237, %v238 : tensor<16x3x3x3xf32>
    %v240 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v241 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v242 = stablehlo.multiply %v240, %W1v : tensor<16x3x3x3xf32>
    %v243 = stablehlo.multiply %v234, %v234 : tensor<16x3x3x3xf32>
    %v244 = stablehlo.multiply %v241, %v243 : tensor<16x3x3x3xf32>
    %v245 = stablehlo.add %v242, %v244 : tensor<16x3x3x3xf32>
    %v246 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v247 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v248 = stablehlo.divide %v239, %v246 : tensor<16x3x3x3xf32>
    %v249 = stablehlo.divide %v245, %v247 : tensor<16x3x3x3xf32>
    %v250 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v251 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v252 = stablehlo.sqrt %v249 : tensor<16x3x3x3xf32>
    %v253 = stablehlo.add %v252, %v251 : tensor<16x3x3x3xf32>
    %v254 = stablehlo.divide %v248, %v253 : tensor<16x3x3x3xf32>
    %v255 = stablehlo.multiply %v250, %v254 : tensor<16x3x3x3xf32>
    %v256 = stablehlo.subtract %W1, %v255 : tensor<16x3x3x3xf32>
    %v257 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v258 = stablehlo.multiply %v257, %v250 : tensor<16x3x3x3xf32>
    %v259 = stablehlo.multiply %v258, %W1 : tensor<16x3x3x3xf32>
    %v260 = stablehlo.subtract %v256, %v259 : tensor<16x3x3x3xf32>
    %v261 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v262 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v263 = stablehlo.multiply %v261, %W1m : tensor<16x3x3x3xf32>
    %v264 = stablehlo.multiply %v262, %v234 : tensor<16x3x3x3xf32>
    %v265 = stablehlo.add %v263, %v264 : tensor<16x3x3x3xf32>
    %v266 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v267 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v268 = stablehlo.multiply %v266, %W1v : tensor<16x3x3x3xf32>
    %v269 = stablehlo.multiply %v234, %v234 : tensor<16x3x3x3xf32>
    %v270 = stablehlo.multiply %v267, %v269 : tensor<16x3x3x3xf32>
    %v271 = stablehlo.add %v268, %v270 : tensor<16x3x3x3xf32>
    %v272 = stablehlo.reshape %v228 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v273 = stablehlo.constant dense<0.0> : tensor<f32>
    %v274 = stablehlo.reduce(%v272 init: %v273) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v275 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v276 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v277 = stablehlo.multiply %v275, %cb1m : tensor<16xf32>
    %v278 = stablehlo.multiply %v276, %v274 : tensor<16xf32>
    %v279 = stablehlo.add %v277, %v278 : tensor<16xf32>
    %v280 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v281 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v282 = stablehlo.multiply %v280, %cb1v : tensor<16xf32>
    %v283 = stablehlo.multiply %v274, %v274 : tensor<16xf32>
    %v284 = stablehlo.multiply %v281, %v283 : tensor<16xf32>
    %v285 = stablehlo.add %v282, %v284 : tensor<16xf32>
    %v286 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v287 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v288 = stablehlo.divide %v279, %v286 : tensor<16xf32>
    %v289 = stablehlo.divide %v285, %v287 : tensor<16xf32>
    %v290 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v291 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v292 = stablehlo.sqrt %v289 : tensor<16xf32>
    %v293 = stablehlo.add %v292, %v291 : tensor<16xf32>
    %v294 = stablehlo.divide %v288, %v293 : tensor<16xf32>
    %v295 = stablehlo.multiply %v290, %v294 : tensor<16xf32>
    %v296 = stablehlo.subtract %cb1, %v295 : tensor<16xf32>
    %v297 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v298 = stablehlo.multiply %v297, %v290 : tensor<16xf32>
    %v299 = stablehlo.multiply %v298, %cb1 : tensor<16xf32>
    %v300 = stablehlo.subtract %v296, %v299 : tensor<16xf32>
    %v301 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v302 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v303 = stablehlo.multiply %v301, %cb1m : tensor<16xf32>
    %v304 = stablehlo.multiply %v302, %v274 : tensor<16xf32>
    %v305 = stablehlo.add %v303, %v304 : tensor<16xf32>
    %v306 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v307 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v308 = stablehlo.multiply %v306, %cb1v : tensor<16xf32>
    %v309 = stablehlo.multiply %v274, %v274 : tensor<16xf32>
    %v310 = stablehlo.multiply %v307, %v309 : tensor<16xf32>
    %v311 = stablehlo.add %v308, %v310 : tensor<16xf32>
    %v312 = stablehlo.reshape %v8 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v313 = stablehlo.reshape %v217 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v314 = stablehlo.transpose %v312, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v315 = stablehlo.transpose %v313, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v316 = stablehlo.convolution(%v314, %v315)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x16x3x3xf32>
    %v317 = stablehlo.transpose %v316, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v318 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v319 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v320 = stablehlo.multiply %v318, %W2m : tensor<16x16x3x3xf32>
    %v321 = stablehlo.multiply %v319, %v317 : tensor<16x16x3x3xf32>
    %v322 = stablehlo.add %v320, %v321 : tensor<16x16x3x3xf32>
    %v323 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v324 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v325 = stablehlo.multiply %v323, %W2v : tensor<16x16x3x3xf32>
    %v326 = stablehlo.multiply %v317, %v317 : tensor<16x16x3x3xf32>
    %v327 = stablehlo.multiply %v324, %v326 : tensor<16x16x3x3xf32>
    %v328 = stablehlo.add %v325, %v327 : tensor<16x16x3x3xf32>
    %v329 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v330 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v331 = stablehlo.divide %v322, %v329 : tensor<16x16x3x3xf32>
    %v332 = stablehlo.divide %v328, %v330 : tensor<16x16x3x3xf32>
    %v333 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v334 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v335 = stablehlo.sqrt %v332 : tensor<16x16x3x3xf32>
    %v336 = stablehlo.add %v335, %v334 : tensor<16x16x3x3xf32>
    %v337 = stablehlo.divide %v331, %v336 : tensor<16x16x3x3xf32>
    %v338 = stablehlo.multiply %v333, %v337 : tensor<16x16x3x3xf32>
    %v339 = stablehlo.subtract %W2, %v338 : tensor<16x16x3x3xf32>
    %v340 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v341 = stablehlo.multiply %v340, %v333 : tensor<16x16x3x3xf32>
    %v342 = stablehlo.multiply %v341, %W2 : tensor<16x16x3x3xf32>
    %v343 = stablehlo.subtract %v339, %v342 : tensor<16x16x3x3xf32>
    %v344 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v345 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v346 = stablehlo.multiply %v344, %W2m : tensor<16x16x3x3xf32>
    %v347 = stablehlo.multiply %v345, %v317 : tensor<16x16x3x3xf32>
    %v348 = stablehlo.add %v346, %v347 : tensor<16x16x3x3xf32>
    %v349 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v350 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v351 = stablehlo.multiply %v349, %W2v : tensor<16x16x3x3xf32>
    %v352 = stablehlo.multiply %v317, %v317 : tensor<16x16x3x3xf32>
    %v353 = stablehlo.multiply %v350, %v352 : tensor<16x16x3x3xf32>
    %v354 = stablehlo.add %v351, %v353 : tensor<16x16x3x3xf32>
    %v355 = stablehlo.reshape %v217 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v356 = stablehlo.constant dense<0.0> : tensor<f32>
    %v357 = stablehlo.reduce(%v355 init: %v356) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v358 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v359 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v360 = stablehlo.multiply %v358, %cb2m : tensor<16xf32>
    %v361 = stablehlo.multiply %v359, %v357 : tensor<16xf32>
    %v362 = stablehlo.add %v360, %v361 : tensor<16xf32>
    %v363 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v364 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v365 = stablehlo.multiply %v363, %cb2v : tensor<16xf32>
    %v366 = stablehlo.multiply %v357, %v357 : tensor<16xf32>
    %v367 = stablehlo.multiply %v364, %v366 : tensor<16xf32>
    %v368 = stablehlo.add %v365, %v367 : tensor<16xf32>
    %v369 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v370 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v371 = stablehlo.divide %v362, %v369 : tensor<16xf32>
    %v372 = stablehlo.divide %v368, %v370 : tensor<16xf32>
    %v373 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v374 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v375 = stablehlo.sqrt %v372 : tensor<16xf32>
    %v376 = stablehlo.add %v375, %v374 : tensor<16xf32>
    %v377 = stablehlo.divide %v371, %v376 : tensor<16xf32>
    %v378 = stablehlo.multiply %v373, %v377 : tensor<16xf32>
    %v379 = stablehlo.subtract %cb2, %v378 : tensor<16xf32>
    %v380 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v381 = stablehlo.multiply %v380, %v373 : tensor<16xf32>
    %v382 = stablehlo.multiply %v381, %cb2 : tensor<16xf32>
    %v383 = stablehlo.subtract %v379, %v382 : tensor<16xf32>
    %v384 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v385 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v386 = stablehlo.multiply %v384, %cb2m : tensor<16xf32>
    %v387 = stablehlo.multiply %v385, %v357 : tensor<16xf32>
    %v388 = stablehlo.add %v386, %v387 : tensor<16xf32>
    %v389 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v390 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v391 = stablehlo.multiply %v389, %cb2v : tensor<16xf32>
    %v392 = stablehlo.multiply %v357, %v357 : tensor<16xf32>
    %v393 = stablehlo.multiply %v390, %v392 : tensor<16xf32>
    %v394 = stablehlo.add %v391, %v393 : tensor<16xf32>
    %v395 = stablehlo.reshape %v21 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v396 = stablehlo.reshape %v201 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v397 = stablehlo.transpose %v395, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v398 = stablehlo.transpose %v396, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v399 = stablehlo.convolution(%v397, %v398)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v400 = stablehlo.transpose %v399, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v401 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v402 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v403 = stablehlo.multiply %v401, %W3m : tensor<16x16x3x3xf32>
    %v404 = stablehlo.multiply %v402, %v400 : tensor<16x16x3x3xf32>
    %v405 = stablehlo.add %v403, %v404 : tensor<16x16x3x3xf32>
    %v406 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v407 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v408 = stablehlo.multiply %v406, %W3v : tensor<16x16x3x3xf32>
    %v409 = stablehlo.multiply %v400, %v400 : tensor<16x16x3x3xf32>
    %v410 = stablehlo.multiply %v407, %v409 : tensor<16x16x3x3xf32>
    %v411 = stablehlo.add %v408, %v410 : tensor<16x16x3x3xf32>
    %v412 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v413 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v414 = stablehlo.divide %v405, %v412 : tensor<16x16x3x3xf32>
    %v415 = stablehlo.divide %v411, %v413 : tensor<16x16x3x3xf32>
    %v416 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v417 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v418 = stablehlo.sqrt %v415 : tensor<16x16x3x3xf32>
    %v419 = stablehlo.add %v418, %v417 : tensor<16x16x3x3xf32>
    %v420 = stablehlo.divide %v414, %v419 : tensor<16x16x3x3xf32>
    %v421 = stablehlo.multiply %v416, %v420 : tensor<16x16x3x3xf32>
    %v422 = stablehlo.subtract %W3, %v421 : tensor<16x16x3x3xf32>
    %v423 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v424 = stablehlo.multiply %v423, %v416 : tensor<16x16x3x3xf32>
    %v425 = stablehlo.multiply %v424, %W3 : tensor<16x16x3x3xf32>
    %v426 = stablehlo.subtract %v422, %v425 : tensor<16x16x3x3xf32>
    %v427 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v428 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v429 = stablehlo.multiply %v427, %W3m : tensor<16x16x3x3xf32>
    %v430 = stablehlo.multiply %v428, %v400 : tensor<16x16x3x3xf32>
    %v431 = stablehlo.add %v429, %v430 : tensor<16x16x3x3xf32>
    %v432 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v433 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v434 = stablehlo.multiply %v432, %W3v : tensor<16x16x3x3xf32>
    %v435 = stablehlo.multiply %v400, %v400 : tensor<16x16x3x3xf32>
    %v436 = stablehlo.multiply %v433, %v435 : tensor<16x16x3x3xf32>
    %v437 = stablehlo.add %v434, %v436 : tensor<16x16x3x3xf32>
    %v438 = stablehlo.reshape %v201 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v439 = stablehlo.constant dense<0.0> : tensor<f32>
    %v440 = stablehlo.reduce(%v438 init: %v439) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v441 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v442 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v443 = stablehlo.multiply %v441, %cb3m : tensor<16xf32>
    %v444 = stablehlo.multiply %v442, %v440 : tensor<16xf32>
    %v445 = stablehlo.add %v443, %v444 : tensor<16xf32>
    %v446 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v447 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v448 = stablehlo.multiply %v446, %cb3v : tensor<16xf32>
    %v449 = stablehlo.multiply %v440, %v440 : tensor<16xf32>
    %v450 = stablehlo.multiply %v447, %v449 : tensor<16xf32>
    %v451 = stablehlo.add %v448, %v450 : tensor<16xf32>
    %v452 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v453 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v454 = stablehlo.divide %v445, %v452 : tensor<16xf32>
    %v455 = stablehlo.divide %v451, %v453 : tensor<16xf32>
    %v456 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v457 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v458 = stablehlo.sqrt %v455 : tensor<16xf32>
    %v459 = stablehlo.add %v458, %v457 : tensor<16xf32>
    %v460 = stablehlo.divide %v454, %v459 : tensor<16xf32>
    %v461 = stablehlo.multiply %v456, %v460 : tensor<16xf32>
    %v462 = stablehlo.subtract %cb3, %v461 : tensor<16xf32>
    %v463 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v464 = stablehlo.multiply %v463, %v456 : tensor<16xf32>
    %v465 = stablehlo.multiply %v464, %cb3 : tensor<16xf32>
    %v466 = stablehlo.subtract %v462, %v465 : tensor<16xf32>
    %v467 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v468 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v469 = stablehlo.multiply %v467, %cb3m : tensor<16xf32>
    %v470 = stablehlo.multiply %v468, %v440 : tensor<16xf32>
    %v471 = stablehlo.add %v469, %v470 : tensor<16xf32>
    %v472 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v473 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v474 = stablehlo.multiply %v472, %cb3v : tensor<16xf32>
    %v475 = stablehlo.multiply %v440, %v440 : tensor<16xf32>
    %v476 = stablehlo.multiply %v473, %v475 : tensor<16xf32>
    %v477 = stablehlo.add %v474, %v476 : tensor<16xf32>
    %v478 = stablehlo.reshape %v30 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v479 = stablehlo.reshape %v190 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v480 = stablehlo.transpose %v478, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v481 = stablehlo.transpose %v479, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v482 = stablehlo.convolution(%v480, %v481)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v483 = stablehlo.transpose %v482, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v484 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v485 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v486 = stablehlo.multiply %v484, %W4m : tensor<16x16x3x3xf32>
    %v487 = stablehlo.multiply %v485, %v483 : tensor<16x16x3x3xf32>
    %v488 = stablehlo.add %v486, %v487 : tensor<16x16x3x3xf32>
    %v489 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v490 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v491 = stablehlo.multiply %v489, %W4v : tensor<16x16x3x3xf32>
    %v492 = stablehlo.multiply %v483, %v483 : tensor<16x16x3x3xf32>
    %v493 = stablehlo.multiply %v490, %v492 : tensor<16x16x3x3xf32>
    %v494 = stablehlo.add %v491, %v493 : tensor<16x16x3x3xf32>
    %v495 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v496 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v497 = stablehlo.divide %v488, %v495 : tensor<16x16x3x3xf32>
    %v498 = stablehlo.divide %v494, %v496 : tensor<16x16x3x3xf32>
    %v499 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v500 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v501 = stablehlo.sqrt %v498 : tensor<16x16x3x3xf32>
    %v502 = stablehlo.add %v501, %v500 : tensor<16x16x3x3xf32>
    %v503 = stablehlo.divide %v497, %v502 : tensor<16x16x3x3xf32>
    %v504 = stablehlo.multiply %v499, %v503 : tensor<16x16x3x3xf32>
    %v505 = stablehlo.subtract %W4, %v504 : tensor<16x16x3x3xf32>
    %v506 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v507 = stablehlo.multiply %v506, %v499 : tensor<16x16x3x3xf32>
    %v508 = stablehlo.multiply %v507, %W4 : tensor<16x16x3x3xf32>
    %v509 = stablehlo.subtract %v505, %v508 : tensor<16x16x3x3xf32>
    %v510 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v511 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v512 = stablehlo.multiply %v510, %W4m : tensor<16x16x3x3xf32>
    %v513 = stablehlo.multiply %v511, %v483 : tensor<16x16x3x3xf32>
    %v514 = stablehlo.add %v512, %v513 : tensor<16x16x3x3xf32>
    %v515 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v516 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v517 = stablehlo.multiply %v515, %W4v : tensor<16x16x3x3xf32>
    %v518 = stablehlo.multiply %v483, %v483 : tensor<16x16x3x3xf32>
    %v519 = stablehlo.multiply %v516, %v518 : tensor<16x16x3x3xf32>
    %v520 = stablehlo.add %v517, %v519 : tensor<16x16x3x3xf32>
    %v521 = stablehlo.reshape %v190 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v522 = stablehlo.constant dense<0.0> : tensor<f32>
    %v523 = stablehlo.reduce(%v521 init: %v522) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v524 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v525 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v526 = stablehlo.multiply %v524, %cb4m : tensor<16xf32>
    %v527 = stablehlo.multiply %v525, %v523 : tensor<16xf32>
    %v528 = stablehlo.add %v526, %v527 : tensor<16xf32>
    %v529 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v530 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v531 = stablehlo.multiply %v529, %cb4v : tensor<16xf32>
    %v532 = stablehlo.multiply %v523, %v523 : tensor<16xf32>
    %v533 = stablehlo.multiply %v530, %v532 : tensor<16xf32>
    %v534 = stablehlo.add %v531, %v533 : tensor<16xf32>
    %v535 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v536 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v537 = stablehlo.divide %v528, %v535 : tensor<16xf32>
    %v538 = stablehlo.divide %v534, %v536 : tensor<16xf32>
    %v539 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v540 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v541 = stablehlo.sqrt %v538 : tensor<16xf32>
    %v542 = stablehlo.add %v541, %v540 : tensor<16xf32>
    %v543 = stablehlo.divide %v537, %v542 : tensor<16xf32>
    %v544 = stablehlo.multiply %v539, %v543 : tensor<16xf32>
    %v545 = stablehlo.subtract %cb4, %v544 : tensor<16xf32>
    %v546 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v547 = stablehlo.multiply %v546, %v539 : tensor<16xf32>
    %v548 = stablehlo.multiply %v547, %cb4 : tensor<16xf32>
    %v549 = stablehlo.subtract %v545, %v548 : tensor<16xf32>
    %v550 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v551 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v552 = stablehlo.multiply %v550, %cb4m : tensor<16xf32>
    %v553 = stablehlo.multiply %v551, %v523 : tensor<16xf32>
    %v554 = stablehlo.add %v552, %v553 : tensor<16xf32>
    %v555 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v556 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v557 = stablehlo.multiply %v555, %cb4v : tensor<16xf32>
    %v558 = stablehlo.multiply %v523, %v523 : tensor<16xf32>
    %v559 = stablehlo.multiply %v556, %v558 : tensor<16xf32>
    %v560 = stablehlo.add %v557, %v559 : tensor<16xf32>
    %v561 = stablehlo.reshape %v43 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v562 = stablehlo.reshape %v174 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v563 = stablehlo.transpose %v561, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v564 = stablehlo.transpose %v562, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v565 = stablehlo.convolution(%v563, %v564)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<16x32x3x3xf32>
    %v566 = stablehlo.transpose %v565, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v567 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v568 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v569 = stablehlo.multiply %v567, %W5m : tensor<32x16x3x3xf32>
    %v570 = stablehlo.multiply %v568, %v566 : tensor<32x16x3x3xf32>
    %v571 = stablehlo.add %v569, %v570 : tensor<32x16x3x3xf32>
    %v572 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v573 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v574 = stablehlo.multiply %v572, %W5v : tensor<32x16x3x3xf32>
    %v575 = stablehlo.multiply %v566, %v566 : tensor<32x16x3x3xf32>
    %v576 = stablehlo.multiply %v573, %v575 : tensor<32x16x3x3xf32>
    %v577 = stablehlo.add %v574, %v576 : tensor<32x16x3x3xf32>
    %v578 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v579 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v580 = stablehlo.divide %v571, %v578 : tensor<32x16x3x3xf32>
    %v581 = stablehlo.divide %v577, %v579 : tensor<32x16x3x3xf32>
    %v582 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v583 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v584 = stablehlo.sqrt %v581 : tensor<32x16x3x3xf32>
    %v585 = stablehlo.add %v584, %v583 : tensor<32x16x3x3xf32>
    %v586 = stablehlo.divide %v580, %v585 : tensor<32x16x3x3xf32>
    %v587 = stablehlo.multiply %v582, %v586 : tensor<32x16x3x3xf32>
    %v588 = stablehlo.subtract %W5, %v587 : tensor<32x16x3x3xf32>
    %v589 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v590 = stablehlo.multiply %v589, %v582 : tensor<32x16x3x3xf32>
    %v591 = stablehlo.multiply %v590, %W5 : tensor<32x16x3x3xf32>
    %v592 = stablehlo.subtract %v588, %v591 : tensor<32x16x3x3xf32>
    %v593 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v594 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v595 = stablehlo.multiply %v593, %W5m : tensor<32x16x3x3xf32>
    %v596 = stablehlo.multiply %v594, %v566 : tensor<32x16x3x3xf32>
    %v597 = stablehlo.add %v595, %v596 : tensor<32x16x3x3xf32>
    %v598 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v599 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v600 = stablehlo.multiply %v598, %W5v : tensor<32x16x3x3xf32>
    %v601 = stablehlo.multiply %v566, %v566 : tensor<32x16x3x3xf32>
    %v602 = stablehlo.multiply %v599, %v601 : tensor<32x16x3x3xf32>
    %v603 = stablehlo.add %v600, %v602 : tensor<32x16x3x3xf32>
    %v604 = stablehlo.reshape %v174 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v605 = stablehlo.constant dense<0.0> : tensor<f32>
    %v606 = stablehlo.reduce(%v604 init: %v605) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v607 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v608 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v609 = stablehlo.multiply %v607, %cb5m : tensor<32xf32>
    %v610 = stablehlo.multiply %v608, %v606 : tensor<32xf32>
    %v611 = stablehlo.add %v609, %v610 : tensor<32xf32>
    %v612 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v613 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v614 = stablehlo.multiply %v612, %cb5v : tensor<32xf32>
    %v615 = stablehlo.multiply %v606, %v606 : tensor<32xf32>
    %v616 = stablehlo.multiply %v613, %v615 : tensor<32xf32>
    %v617 = stablehlo.add %v614, %v616 : tensor<32xf32>
    %v618 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v619 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v620 = stablehlo.divide %v611, %v618 : tensor<32xf32>
    %v621 = stablehlo.divide %v617, %v619 : tensor<32xf32>
    %v622 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v623 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v624 = stablehlo.sqrt %v621 : tensor<32xf32>
    %v625 = stablehlo.add %v624, %v623 : tensor<32xf32>
    %v626 = stablehlo.divide %v620, %v625 : tensor<32xf32>
    %v627 = stablehlo.multiply %v622, %v626 : tensor<32xf32>
    %v628 = stablehlo.subtract %cb5, %v627 : tensor<32xf32>
    %v629 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v630 = stablehlo.multiply %v629, %v622 : tensor<32xf32>
    %v631 = stablehlo.multiply %v630, %cb5 : tensor<32xf32>
    %v632 = stablehlo.subtract %v628, %v631 : tensor<32xf32>
    %v633 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v634 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v635 = stablehlo.multiply %v633, %cb5m : tensor<32xf32>
    %v636 = stablehlo.multiply %v634, %v606 : tensor<32xf32>
    %v637 = stablehlo.add %v635, %v636 : tensor<32xf32>
    %v638 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v639 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v640 = stablehlo.multiply %v638, %cb5v : tensor<32xf32>
    %v641 = stablehlo.multiply %v606, %v606 : tensor<32xf32>
    %v642 = stablehlo.multiply %v639, %v641 : tensor<32xf32>
    %v643 = stablehlo.add %v640, %v642 : tensor<32xf32>
    %v644 = stablehlo.reshape %v52 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v645 = stablehlo.reshape %v163 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v646 = stablehlo.transpose %v644, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v647 = stablehlo.transpose %v645, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v648 = stablehlo.convolution(%v646, %v647)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<32x32x3x3xf32>
    %v649 = stablehlo.transpose %v648, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v650 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v651 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v652 = stablehlo.multiply %v650, %W6m : tensor<32x32x3x3xf32>
    %v653 = stablehlo.multiply %v651, %v649 : tensor<32x32x3x3xf32>
    %v654 = stablehlo.add %v652, %v653 : tensor<32x32x3x3xf32>
    %v655 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v656 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v657 = stablehlo.multiply %v655, %W6v : tensor<32x32x3x3xf32>
    %v658 = stablehlo.multiply %v649, %v649 : tensor<32x32x3x3xf32>
    %v659 = stablehlo.multiply %v656, %v658 : tensor<32x32x3x3xf32>
    %v660 = stablehlo.add %v657, %v659 : tensor<32x32x3x3xf32>
    %v661 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v662 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v663 = stablehlo.divide %v654, %v661 : tensor<32x32x3x3xf32>
    %v664 = stablehlo.divide %v660, %v662 : tensor<32x32x3x3xf32>
    %v665 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v666 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v667 = stablehlo.sqrt %v664 : tensor<32x32x3x3xf32>
    %v668 = stablehlo.add %v667, %v666 : tensor<32x32x3x3xf32>
    %v669 = stablehlo.divide %v663, %v668 : tensor<32x32x3x3xf32>
    %v670 = stablehlo.multiply %v665, %v669 : tensor<32x32x3x3xf32>
    %v671 = stablehlo.subtract %W6, %v670 : tensor<32x32x3x3xf32>
    %v672 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v673 = stablehlo.multiply %v672, %v665 : tensor<32x32x3x3xf32>
    %v674 = stablehlo.multiply %v673, %W6 : tensor<32x32x3x3xf32>
    %v675 = stablehlo.subtract %v671, %v674 : tensor<32x32x3x3xf32>
    %v676 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v677 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v678 = stablehlo.multiply %v676, %W6m : tensor<32x32x3x3xf32>
    %v679 = stablehlo.multiply %v677, %v649 : tensor<32x32x3x3xf32>
    %v680 = stablehlo.add %v678, %v679 : tensor<32x32x3x3xf32>
    %v681 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v682 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v683 = stablehlo.multiply %v681, %W6v : tensor<32x32x3x3xf32>
    %v684 = stablehlo.multiply %v649, %v649 : tensor<32x32x3x3xf32>
    %v685 = stablehlo.multiply %v682, %v684 : tensor<32x32x3x3xf32>
    %v686 = stablehlo.add %v683, %v685 : tensor<32x32x3x3xf32>
    %v687 = stablehlo.reshape %v163 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v688 = stablehlo.constant dense<0.0> : tensor<f32>
    %v689 = stablehlo.reduce(%v687 init: %v688) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v690 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v691 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v692 = stablehlo.multiply %v690, %cb6m : tensor<32xf32>
    %v693 = stablehlo.multiply %v691, %v689 : tensor<32xf32>
    %v694 = stablehlo.add %v692, %v693 : tensor<32xf32>
    %v695 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v696 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v697 = stablehlo.multiply %v695, %cb6v : tensor<32xf32>
    %v698 = stablehlo.multiply %v689, %v689 : tensor<32xf32>
    %v699 = stablehlo.multiply %v696, %v698 : tensor<32xf32>
    %v700 = stablehlo.add %v697, %v699 : tensor<32xf32>
    %v701 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v702 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v703 = stablehlo.divide %v694, %v701 : tensor<32xf32>
    %v704 = stablehlo.divide %v700, %v702 : tensor<32xf32>
    %v705 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v706 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v707 = stablehlo.sqrt %v704 : tensor<32xf32>
    %v708 = stablehlo.add %v707, %v706 : tensor<32xf32>
    %v709 = stablehlo.divide %v703, %v708 : tensor<32xf32>
    %v710 = stablehlo.multiply %v705, %v709 : tensor<32xf32>
    %v711 = stablehlo.subtract %cb6, %v710 : tensor<32xf32>
    %v712 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v713 = stablehlo.multiply %v712, %v705 : tensor<32xf32>
    %v714 = stablehlo.multiply %v713, %cb6 : tensor<32xf32>
    %v715 = stablehlo.subtract %v711, %v714 : tensor<32xf32>
    %v716 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v717 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v718 = stablehlo.multiply %v716, %cb6m : tensor<32xf32>
    %v719 = stablehlo.multiply %v717, %v689 : tensor<32xf32>
    %v720 = stablehlo.add %v718, %v719 : tensor<32xf32>
    %v721 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v722 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v723 = stablehlo.multiply %v721, %cb6v : tensor<32xf32>
    %v724 = stablehlo.multiply %v689, %v689 : tensor<32xf32>
    %v725 = stablehlo.multiply %v722, %v724 : tensor<32xf32>
    %v726 = stablehlo.add %v723, %v725 : tensor<32xf32>
    %v727 = stablehlo.reshape %v65 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v728 = stablehlo.reshape %v147 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v729 = stablehlo.transpose %v727, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v730 = stablehlo.transpose %v728, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v731 = stablehlo.convolution(%v729, %v730)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v732 = stablehlo.transpose %v731, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v733 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v734 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v735 = stablehlo.multiply %v733, %W7m : tensor<32x32x3x3xf32>
    %v736 = stablehlo.multiply %v734, %v732 : tensor<32x32x3x3xf32>
    %v737 = stablehlo.add %v735, %v736 : tensor<32x32x3x3xf32>
    %v738 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v739 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v740 = stablehlo.multiply %v738, %W7v : tensor<32x32x3x3xf32>
    %v741 = stablehlo.multiply %v732, %v732 : tensor<32x32x3x3xf32>
    %v742 = stablehlo.multiply %v739, %v741 : tensor<32x32x3x3xf32>
    %v743 = stablehlo.add %v740, %v742 : tensor<32x32x3x3xf32>
    %v744 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v745 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v746 = stablehlo.divide %v737, %v744 : tensor<32x32x3x3xf32>
    %v747 = stablehlo.divide %v743, %v745 : tensor<32x32x3x3xf32>
    %v748 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v749 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v750 = stablehlo.sqrt %v747 : tensor<32x32x3x3xf32>
    %v751 = stablehlo.add %v750, %v749 : tensor<32x32x3x3xf32>
    %v752 = stablehlo.divide %v746, %v751 : tensor<32x32x3x3xf32>
    %v753 = stablehlo.multiply %v748, %v752 : tensor<32x32x3x3xf32>
    %v754 = stablehlo.subtract %W7, %v753 : tensor<32x32x3x3xf32>
    %v755 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v756 = stablehlo.multiply %v755, %v748 : tensor<32x32x3x3xf32>
    %v757 = stablehlo.multiply %v756, %W7 : tensor<32x32x3x3xf32>
    %v758 = stablehlo.subtract %v754, %v757 : tensor<32x32x3x3xf32>
    %v759 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v760 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v761 = stablehlo.multiply %v759, %W7m : tensor<32x32x3x3xf32>
    %v762 = stablehlo.multiply %v760, %v732 : tensor<32x32x3x3xf32>
    %v763 = stablehlo.add %v761, %v762 : tensor<32x32x3x3xf32>
    %v764 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v765 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v766 = stablehlo.multiply %v764, %W7v : tensor<32x32x3x3xf32>
    %v767 = stablehlo.multiply %v732, %v732 : tensor<32x32x3x3xf32>
    %v768 = stablehlo.multiply %v765, %v767 : tensor<32x32x3x3xf32>
    %v769 = stablehlo.add %v766, %v768 : tensor<32x32x3x3xf32>
    %v770 = stablehlo.reshape %v147 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v771 = stablehlo.constant dense<0.0> : tensor<f32>
    %v772 = stablehlo.reduce(%v770 init: %v771) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v773 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v774 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v775 = stablehlo.multiply %v773, %cb7m : tensor<32xf32>
    %v776 = stablehlo.multiply %v774, %v772 : tensor<32xf32>
    %v777 = stablehlo.add %v775, %v776 : tensor<32xf32>
    %v778 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v779 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v780 = stablehlo.multiply %v778, %cb7v : tensor<32xf32>
    %v781 = stablehlo.multiply %v772, %v772 : tensor<32xf32>
    %v782 = stablehlo.multiply %v779, %v781 : tensor<32xf32>
    %v783 = stablehlo.add %v780, %v782 : tensor<32xf32>
    %v784 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v785 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v786 = stablehlo.divide %v777, %v784 : tensor<32xf32>
    %v787 = stablehlo.divide %v783, %v785 : tensor<32xf32>
    %v788 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v789 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v790 = stablehlo.sqrt %v787 : tensor<32xf32>
    %v791 = stablehlo.add %v790, %v789 : tensor<32xf32>
    %v792 = stablehlo.divide %v786, %v791 : tensor<32xf32>
    %v793 = stablehlo.multiply %v788, %v792 : tensor<32xf32>
    %v794 = stablehlo.subtract %cb7, %v793 : tensor<32xf32>
    %v795 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v796 = stablehlo.multiply %v795, %v788 : tensor<32xf32>
    %v797 = stablehlo.multiply %v796, %cb7 : tensor<32xf32>
    %v798 = stablehlo.subtract %v794, %v797 : tensor<32xf32>
    %v799 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v800 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v801 = stablehlo.multiply %v799, %cb7m : tensor<32xf32>
    %v802 = stablehlo.multiply %v800, %v772 : tensor<32xf32>
    %v803 = stablehlo.add %v801, %v802 : tensor<32xf32>
    %v804 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v805 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v806 = stablehlo.multiply %v804, %cb7v : tensor<32xf32>
    %v807 = stablehlo.multiply %v772, %v772 : tensor<32xf32>
    %v808 = stablehlo.multiply %v805, %v807 : tensor<32xf32>
    %v809 = stablehlo.add %v806, %v808 : tensor<32xf32>
    %v810 = stablehlo.reshape %v74 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v811 = stablehlo.reshape %v136 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v812 = stablehlo.transpose %v810, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v813 = stablehlo.transpose %v811, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v814 = stablehlo.convolution(%v812, %v813)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v815 = stablehlo.transpose %v814, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v816 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v817 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v818 = stablehlo.multiply %v816, %W8m : tensor<32x32x3x3xf32>
    %v819 = stablehlo.multiply %v817, %v815 : tensor<32x32x3x3xf32>
    %v820 = stablehlo.add %v818, %v819 : tensor<32x32x3x3xf32>
    %v821 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v822 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v823 = stablehlo.multiply %v821, %W8v : tensor<32x32x3x3xf32>
    %v824 = stablehlo.multiply %v815, %v815 : tensor<32x32x3x3xf32>
    %v825 = stablehlo.multiply %v822, %v824 : tensor<32x32x3x3xf32>
    %v826 = stablehlo.add %v823, %v825 : tensor<32x32x3x3xf32>
    %v827 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v828 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v829 = stablehlo.divide %v820, %v827 : tensor<32x32x3x3xf32>
    %v830 = stablehlo.divide %v826, %v828 : tensor<32x32x3x3xf32>
    %v831 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v832 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v833 = stablehlo.sqrt %v830 : tensor<32x32x3x3xf32>
    %v834 = stablehlo.add %v833, %v832 : tensor<32x32x3x3xf32>
    %v835 = stablehlo.divide %v829, %v834 : tensor<32x32x3x3xf32>
    %v836 = stablehlo.multiply %v831, %v835 : tensor<32x32x3x3xf32>
    %v837 = stablehlo.subtract %W8, %v836 : tensor<32x32x3x3xf32>
    %v838 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v839 = stablehlo.multiply %v838, %v831 : tensor<32x32x3x3xf32>
    %v840 = stablehlo.multiply %v839, %W8 : tensor<32x32x3x3xf32>
    %v841 = stablehlo.subtract %v837, %v840 : tensor<32x32x3x3xf32>
    %v842 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v843 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v844 = stablehlo.multiply %v842, %W8m : tensor<32x32x3x3xf32>
    %v845 = stablehlo.multiply %v843, %v815 : tensor<32x32x3x3xf32>
    %v846 = stablehlo.add %v844, %v845 : tensor<32x32x3x3xf32>
    %v847 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v848 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v849 = stablehlo.multiply %v847, %W8v : tensor<32x32x3x3xf32>
    %v850 = stablehlo.multiply %v815, %v815 : tensor<32x32x3x3xf32>
    %v851 = stablehlo.multiply %v848, %v850 : tensor<32x32x3x3xf32>
    %v852 = stablehlo.add %v849, %v851 : tensor<32x32x3x3xf32>
    %v853 = stablehlo.reshape %v136 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v854 = stablehlo.constant dense<0.0> : tensor<f32>
    %v855 = stablehlo.reduce(%v853 init: %v854) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v856 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v857 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v858 = stablehlo.multiply %v856, %cb8m : tensor<32xf32>
    %v859 = stablehlo.multiply %v857, %v855 : tensor<32xf32>
    %v860 = stablehlo.add %v858, %v859 : tensor<32xf32>
    %v861 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v862 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v863 = stablehlo.multiply %v861, %cb8v : tensor<32xf32>
    %v864 = stablehlo.multiply %v855, %v855 : tensor<32xf32>
    %v865 = stablehlo.multiply %v862, %v864 : tensor<32xf32>
    %v866 = stablehlo.add %v863, %v865 : tensor<32xf32>
    %v867 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v868 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v869 = stablehlo.divide %v860, %v867 : tensor<32xf32>
    %v870 = stablehlo.divide %v866, %v868 : tensor<32xf32>
    %v871 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v872 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v873 = stablehlo.sqrt %v870 : tensor<32xf32>
    %v874 = stablehlo.add %v873, %v872 : tensor<32xf32>
    %v875 = stablehlo.divide %v869, %v874 : tensor<32xf32>
    %v876 = stablehlo.multiply %v871, %v875 : tensor<32xf32>
    %v877 = stablehlo.subtract %cb8, %v876 : tensor<32xf32>
    %v878 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v879 = stablehlo.multiply %v878, %v871 : tensor<32xf32>
    %v880 = stablehlo.multiply %v879, %cb8 : tensor<32xf32>
    %v881 = stablehlo.subtract %v877, %v880 : tensor<32xf32>
    %v882 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v883 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v884 = stablehlo.multiply %v882, %cb8m : tensor<32xf32>
    %v885 = stablehlo.multiply %v883, %v855 : tensor<32xf32>
    %v886 = stablehlo.add %v884, %v885 : tensor<32xf32>
    %v887 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v888 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v889 = stablehlo.multiply %v887, %cb8v : tensor<32xf32>
    %v890 = stablehlo.multiply %v855, %v855 : tensor<32xf32>
    %v891 = stablehlo.multiply %v888, %v890 : tensor<32xf32>
    %v892 = stablehlo.add %v889, %v891 : tensor<32xf32>
    %v893 = stablehlo.dot_general %v87, %v122, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %v894 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v895 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v896 = stablehlo.multiply %v894, %W9m : tensor<128x64xf32>
    %v897 = stablehlo.multiply %v895, %v893 : tensor<128x64xf32>
    %v898 = stablehlo.add %v896, %v897 : tensor<128x64xf32>
    %v899 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v900 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v901 = stablehlo.multiply %v899, %W9v : tensor<128x64xf32>
    %v902 = stablehlo.multiply %v893, %v893 : tensor<128x64xf32>
    %v903 = stablehlo.multiply %v900, %v902 : tensor<128x64xf32>
    %v904 = stablehlo.add %v901, %v903 : tensor<128x64xf32>
    %v905 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v906 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v907 = stablehlo.divide %v898, %v905 : tensor<128x64xf32>
    %v908 = stablehlo.divide %v904, %v906 : tensor<128x64xf32>
    %v909 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v910 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v911 = stablehlo.sqrt %v908 : tensor<128x64xf32>
    %v912 = stablehlo.add %v911, %v910 : tensor<128x64xf32>
    %v913 = stablehlo.divide %v907, %v912 : tensor<128x64xf32>
    %v914 = stablehlo.multiply %v909, %v913 : tensor<128x64xf32>
    %v915 = stablehlo.subtract %W9, %v914 : tensor<128x64xf32>
    %v916 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v917 = stablehlo.multiply %v916, %v909 : tensor<128x64xf32>
    %v918 = stablehlo.multiply %v917, %W9 : tensor<128x64xf32>
    %v919 = stablehlo.subtract %v915, %v918 : tensor<128x64xf32>
    %v920 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v921 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v922 = stablehlo.multiply %v920, %W9m : tensor<128x64xf32>
    %v923 = stablehlo.multiply %v921, %v893 : tensor<128x64xf32>
    %v924 = stablehlo.add %v922, %v923 : tensor<128x64xf32>
    %v925 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v926 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v927 = stablehlo.multiply %v925, %W9v : tensor<128x64xf32>
    %v928 = stablehlo.multiply %v893, %v893 : tensor<128x64xf32>
    %v929 = stablehlo.multiply %v926, %v928 : tensor<128x64xf32>
    %v930 = stablehlo.add %v927, %v929 : tensor<128x64xf32>
    %v931 = stablehlo.constant dense<0.0> : tensor<f32>
    %v932 = stablehlo.reduce(%v122 init: %v931) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v933 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v934 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v935 = stablehlo.multiply %v933, %b9m : tensor<64xf32>
    %v936 = stablehlo.multiply %v934, %v932 : tensor<64xf32>
    %v937 = stablehlo.add %v935, %v936 : tensor<64xf32>
    %v938 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v939 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v940 = stablehlo.multiply %v938, %b9v : tensor<64xf32>
    %v941 = stablehlo.multiply %v932, %v932 : tensor<64xf32>
    %v942 = stablehlo.multiply %v939, %v941 : tensor<64xf32>
    %v943 = stablehlo.add %v940, %v942 : tensor<64xf32>
    %v944 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v945 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v946 = stablehlo.divide %v937, %v944 : tensor<64xf32>
    %v947 = stablehlo.divide %v943, %v945 : tensor<64xf32>
    %v948 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v949 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v950 = stablehlo.sqrt %v947 : tensor<64xf32>
    %v951 = stablehlo.add %v950, %v949 : tensor<64xf32>
    %v952 = stablehlo.divide %v946, %v951 : tensor<64xf32>
    %v953 = stablehlo.multiply %v948, %v952 : tensor<64xf32>
    %v954 = stablehlo.subtract %b9, %v953 : tensor<64xf32>
    %v955 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v956 = stablehlo.multiply %v955, %v948 : tensor<64xf32>
    %v957 = stablehlo.multiply %v956, %b9 : tensor<64xf32>
    %v958 = stablehlo.subtract %v954, %v957 : tensor<64xf32>
    %v959 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v960 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v961 = stablehlo.multiply %v959, %b9m : tensor<64xf32>
    %v962 = stablehlo.multiply %v960, %v932 : tensor<64xf32>
    %v963 = stablehlo.add %v961, %v962 : tensor<64xf32>
    %v964 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v965 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v966 = stablehlo.multiply %v964, %b9v : tensor<64xf32>
    %v967 = stablehlo.multiply %v932, %v932 : tensor<64xf32>
    %v968 = stablehlo.multiply %v965, %v967 : tensor<64xf32>
    %v969 = stablehlo.add %v966, %v968 : tensor<64xf32>
    %v970 = stablehlo.dot_general %v92, %v116, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<64x64xf32>
    %v971 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v972 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v973 = stablehlo.multiply %v971, %Wam : tensor<64x64xf32>
    %v974 = stablehlo.multiply %v972, %v970 : tensor<64x64xf32>
    %v975 = stablehlo.add %v973, %v974 : tensor<64x64xf32>
    %v976 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v977 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v978 = stablehlo.multiply %v976, %Wav : tensor<64x64xf32>
    %v979 = stablehlo.multiply %v970, %v970 : tensor<64x64xf32>
    %v980 = stablehlo.multiply %v977, %v979 : tensor<64x64xf32>
    %v981 = stablehlo.add %v978, %v980 : tensor<64x64xf32>
    %v982 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v983 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v984 = stablehlo.divide %v975, %v982 : tensor<64x64xf32>
    %v985 = stablehlo.divide %v981, %v983 : tensor<64x64xf32>
    %v986 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v987 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v988 = stablehlo.sqrt %v985 : tensor<64x64xf32>
    %v989 = stablehlo.add %v988, %v987 : tensor<64x64xf32>
    %v990 = stablehlo.divide %v984, %v989 : tensor<64x64xf32>
    %v991 = stablehlo.multiply %v986, %v990 : tensor<64x64xf32>
    %v992 = stablehlo.subtract %Wa, %v991 : tensor<64x64xf32>
    %v993 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v994 = stablehlo.multiply %v993, %v986 : tensor<64x64xf32>
    %v995 = stablehlo.multiply %v994, %Wa : tensor<64x64xf32>
    %v996 = stablehlo.subtract %v992, %v995 : tensor<64x64xf32>
    %v997 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v998 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v999 = stablehlo.multiply %v997, %Wam : tensor<64x64xf32>
    %v1000 = stablehlo.multiply %v998, %v970 : tensor<64x64xf32>
    %v1001 = stablehlo.add %v999, %v1000 : tensor<64x64xf32>
    %v1002 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1003 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1004 = stablehlo.multiply %v1002, %Wav : tensor<64x64xf32>
    %v1005 = stablehlo.multiply %v970, %v970 : tensor<64x64xf32>
    %v1006 = stablehlo.multiply %v1003, %v1005 : tensor<64x64xf32>
    %v1007 = stablehlo.add %v1004, %v1006 : tensor<64x64xf32>
    %v1008 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1009 = stablehlo.reduce(%v116 init: %v1008) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v1010 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1011 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1012 = stablehlo.multiply %v1010, %bam : tensor<64xf32>
    %v1013 = stablehlo.multiply %v1011, %v1009 : tensor<64xf32>
    %v1014 = stablehlo.add %v1012, %v1013 : tensor<64xf32>
    %v1015 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1016 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1017 = stablehlo.multiply %v1015, %bav : tensor<64xf32>
    %v1018 = stablehlo.multiply %v1009, %v1009 : tensor<64xf32>
    %v1019 = stablehlo.multiply %v1016, %v1018 : tensor<64xf32>
    %v1020 = stablehlo.add %v1017, %v1019 : tensor<64xf32>
    %v1021 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1022 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1023 = stablehlo.divide %v1014, %v1021 : tensor<64xf32>
    %v1024 = stablehlo.divide %v1020, %v1022 : tensor<64xf32>
    %v1025 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1026 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1027 = stablehlo.sqrt %v1024 : tensor<64xf32>
    %v1028 = stablehlo.add %v1027, %v1026 : tensor<64xf32>
    %v1029 = stablehlo.divide %v1023, %v1028 : tensor<64xf32>
    %v1030 = stablehlo.multiply %v1025, %v1029 : tensor<64xf32>
    %v1031 = stablehlo.subtract %ba, %v1030 : tensor<64xf32>
    %v1032 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1033 = stablehlo.multiply %v1032, %v1025 : tensor<64xf32>
    %v1034 = stablehlo.multiply %v1033, %ba : tensor<64xf32>
    %v1035 = stablehlo.subtract %v1031, %v1034 : tensor<64xf32>
    %v1036 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1037 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1038 = stablehlo.multiply %v1036, %bam : tensor<64xf32>
    %v1039 = stablehlo.multiply %v1037, %v1009 : tensor<64xf32>
    %v1040 = stablehlo.add %v1038, %v1039 : tensor<64xf32>
    %v1041 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1042 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1043 = stablehlo.multiply %v1041, %bav : tensor<64xf32>
    %v1044 = stablehlo.multiply %v1009, %v1009 : tensor<64xf32>
    %v1045 = stablehlo.multiply %v1042, %v1044 : tensor<64xf32>
    %v1046 = stablehlo.add %v1043, %v1045 : tensor<64xf32>
    %v1047 = stablehlo.dot_general %v97, %v110, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x10xf32>) -> tensor<64x10xf32>
    %v1048 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1049 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1050 = stablehlo.multiply %v1048, %Wbm : tensor<64x10xf32>
    %v1051 = stablehlo.multiply %v1049, %v1047 : tensor<64x10xf32>
    %v1052 = stablehlo.add %v1050, %v1051 : tensor<64x10xf32>
    %v1053 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1054 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1055 = stablehlo.multiply %v1053, %Wbv : tensor<64x10xf32>
    %v1056 = stablehlo.multiply %v1047, %v1047 : tensor<64x10xf32>
    %v1057 = stablehlo.multiply %v1054, %v1056 : tensor<64x10xf32>
    %v1058 = stablehlo.add %v1055, %v1057 : tensor<64x10xf32>
    %v1059 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1060 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1061 = stablehlo.divide %v1052, %v1059 : tensor<64x10xf32>
    %v1062 = stablehlo.divide %v1058, %v1060 : tensor<64x10xf32>
    %v1063 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1064 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1065 = stablehlo.sqrt %v1062 : tensor<64x10xf32>
    %v1066 = stablehlo.add %v1065, %v1064 : tensor<64x10xf32>
    %v1067 = stablehlo.divide %v1061, %v1066 : tensor<64x10xf32>
    %v1068 = stablehlo.multiply %v1063, %v1067 : tensor<64x10xf32>
    %v1069 = stablehlo.subtract %Wb, %v1068 : tensor<64x10xf32>
    %v1070 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1071 = stablehlo.multiply %v1070, %v1063 : tensor<64x10xf32>
    %v1072 = stablehlo.multiply %v1071, %Wb : tensor<64x10xf32>
    %v1073 = stablehlo.subtract %v1069, %v1072 : tensor<64x10xf32>
    %v1074 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1075 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1076 = stablehlo.multiply %v1074, %Wbm : tensor<64x10xf32>
    %v1077 = stablehlo.multiply %v1075, %v1047 : tensor<64x10xf32>
    %v1078 = stablehlo.add %v1076, %v1077 : tensor<64x10xf32>
    %v1079 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1080 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1081 = stablehlo.multiply %v1079, %Wbv : tensor<64x10xf32>
    %v1082 = stablehlo.multiply %v1047, %v1047 : tensor<64x10xf32>
    %v1083 = stablehlo.multiply %v1080, %v1082 : tensor<64x10xf32>
    %v1084 = stablehlo.add %v1081, %v1083 : tensor<64x10xf32>
    %v1085 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1086 = stablehlo.reduce(%v110 init: %v1085) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1087 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1088 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1089 = stablehlo.multiply %v1087, %bbm : tensor<10xf32>
    %v1090 = stablehlo.multiply %v1088, %v1086 : tensor<10xf32>
    %v1091 = stablehlo.add %v1089, %v1090 : tensor<10xf32>
    %v1092 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1093 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1094 = stablehlo.multiply %v1092, %bbv : tensor<10xf32>
    %v1095 = stablehlo.multiply %v1086, %v1086 : tensor<10xf32>
    %v1096 = stablehlo.multiply %v1093, %v1095 : tensor<10xf32>
    %v1097 = stablehlo.add %v1094, %v1096 : tensor<10xf32>
    %v1098 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1099 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1100 = stablehlo.divide %v1091, %v1098 : tensor<10xf32>
    %v1101 = stablehlo.divide %v1097, %v1099 : tensor<10xf32>
    %v1102 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1103 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1104 = stablehlo.sqrt %v1101 : tensor<10xf32>
    %v1105 = stablehlo.add %v1104, %v1103 : tensor<10xf32>
    %v1106 = stablehlo.divide %v1100, %v1105 : tensor<10xf32>
    %v1107 = stablehlo.multiply %v1102, %v1106 : tensor<10xf32>
    %v1108 = stablehlo.subtract %bb, %v1107 : tensor<10xf32>
    %v1109 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1110 = stablehlo.multiply %v1109, %v1102 : tensor<10xf32>
    %v1111 = stablehlo.multiply %v1110, %bb : tensor<10xf32>
    %v1112 = stablehlo.subtract %v1108, %v1111 : tensor<10xf32>
    %v1113 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1114 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1115 = stablehlo.multiply %v1113, %bbm : tensor<10xf32>
    %v1116 = stablehlo.multiply %v1114, %v1086 : tensor<10xf32>
    %v1117 = stablehlo.add %v1115, %v1116 : tensor<10xf32>
    %v1118 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1119 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1120 = stablehlo.multiply %v1118, %bbv : tensor<10xf32>
    %v1121 = stablehlo.multiply %v1086, %v1086 : tensor<10xf32>
    %v1122 = stablehlo.multiply %v1119, %v1121 : tensor<10xf32>
    %v1123 = stablehlo.add %v1120, %v1122 : tensor<10xf32>
    return %v260, %v300, %v343, %v383, %v426, %v466, %v509, %v549, %v592, %v632, %v675, %v715, %v758, %v798, %v841, %v881, %v919, %v958, %v996, %v1035, %v1073, %v1112, %v265, %v305, %v348, %v388, %v431, %v471, %v514, %v554, %v597, %v637, %v680, %v720, %v763, %v803, %v846, %v886, %v924, %v963, %v1001, %v1040, %v1078, %v1117, %v271, %v311, %v354, %v394, %v437, %v477, %v520, %v560, %v603, %v643, %v686, %v726, %v769, %v809, %v852, %v892, %v930, %v969, %v1007, %v1046, %v1084, %v1123, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
