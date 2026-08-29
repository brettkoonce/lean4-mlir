module @m {
  func.func @cifar8_sgd_train_step(%x: tensor<128x3072xf32>, %W1: tensor<16x3x3x3xf32>, %cb1: tensor<16xf32>, %W2: tensor<16x16x3x3xf32>, %cb2: tensor<16xf32>, %W3: tensor<16x16x3x3xf32>, %cb3: tensor<16xf32>, %W4: tensor<16x16x3x3xf32>, %cb4: tensor<16xf32>, %W5: tensor<32x16x3x3xf32>, %cb5: tensor<32xf32>, %W6: tensor<32x32x3x3xf32>, %cb6: tensor<32xf32>, %W7: tensor<32x32x3x3xf32>, %cb7: tensor<32xf32>, %W8: tensor<32x32x3x3xf32>, %cb8: tensor<32xf32>, %W9: tensor<128x64xf32>, %b9: tensor<64xf32>, %Wa: tensor<64x64xf32>, %ba: tensor<64xf32>, %Wb: tensor<64x10xf32>, %bb: tensor<10xf32>, %W1m: tensor<16x3x3x3xf32>, %cb1m: tensor<16xf32>, %W2m: tensor<16x16x3x3xf32>, %cb2m: tensor<16xf32>, %W3m: tensor<16x16x3x3xf32>, %cb3m: tensor<16xf32>, %W4m: tensor<16x16x3x3xf32>, %cb4m: tensor<16xf32>, %W5m: tensor<32x16x3x3xf32>, %cb5m: tensor<32xf32>, %W6m: tensor<32x32x3x3xf32>, %cb6m: tensor<32xf32>, %W7m: tensor<32x32x3x3xf32>, %cb7m: tensor<32xf32>, %W8m: tensor<32x32x3x3xf32>, %cb8m: tensor<32xf32>, %W9m: tensor<128x64xf32>, %b9m: tensor<64xf32>, %Wam: tensor<64x64xf32>, %bam: tensor<64xf32>, %Wbm: tensor<64x10xf32>, %bbm: tensor<10xf32>, %W1v: tensor<16x3x3x3xf32>, %cb1v: tensor<16xf32>, %W2v: tensor<16x16x3x3xf32>, %cb2v: tensor<16xf32>, %W3v: tensor<16x16x3x3xf32>, %cb3v: tensor<16xf32>, %W4v: tensor<16x16x3x3xf32>, %cb4v: tensor<16xf32>, %W5v: tensor<32x16x3x3xf32>, %cb5v: tensor<32xf32>, %W6v: tensor<32x32x3x3xf32>, %cb6v: tensor<32xf32>, %W7v: tensor<32x32x3x3xf32>, %cb7v: tensor<32xf32>, %W8v: tensor<32x32x3x3xf32>, %cb8v: tensor<32xf32>, %W9v: tensor<128x64xf32>, %b9v: tensor<64xf32>, %Wav: tensor<64x64xf32>, %bav: tensor<64xf32>, %Wbv: tensor<64x10xf32>, %bbv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: tensor<128x10xf32>) -> (tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>) {
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
    %v101 = stablehlo.exponential %v100 : tensor<128x10xf32>
    %v102 = stablehlo.constant dense<0.0> : tensor<f32>
    %v103 = stablehlo.reduce(%v101 init: %v102) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %v104 = stablehlo.broadcast_in_dim %v103, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %v105 = stablehlo.divide %v101, %v104 : tensor<128x10xf32>
    %v106 = stablehlo.subtract %v105, %onehot : tensor<128x10xf32>
    %v107 = stablehlo.constant dense<0.0078125> : tensor<128x10xf32>
    %v108 = stablehlo.multiply %v106, %v107 : tensor<128x10xf32>
    // ── report-only scalar loss (NOT pretty(AST): the kit has no rank-0 loss op; it
    //    feeds no parameter, only the driver's progress line) ──
    %llog = stablehlo.log %v105 : tensor<128x10xf32>
    %ohll = stablehlo.multiply %onehot, %llog : tensor<128x10xf32>
    %csum = stablehlo.reduce(%ohll init: %lzero) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<f32>
    %cneg = stablehlo.negate %csum : tensor<f32>
    %lbf = stablehlo.constant dense<128.0> : tensor<f32>
    %loss = stablehlo.divide %cneg, %lbf : tensor<f32>
    %v109 = stablehlo.dot_general %v108, %Wb, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x10xf32>, tensor<64x10xf32>) -> tensor<128x64xf32>
    %v110 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v111 = stablehlo.compare GT, %v95, %v110 : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xi1>
    %v112 = stablehlo.select %v111, %v109, %v110 : tensor<128x64xi1>, tensor<128x64xf32>
    %v113 = stablehlo.dot_general %v112, %Wa, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
    %v114 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v115 = stablehlo.compare GT, %v90, %v114 : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xi1>
    %v116 = stablehlo.select %v115, %v113, %v114 : tensor<128x64xi1>, tensor<128x64xf32>
    %v117 = stablehlo.dot_general %v116, %W9, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x128xf32>
    %v118 = stablehlo.reshape %v83 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v119 = stablehlo.reshape %v117 : (tensor<128x128xf32>) -> tensor<128x32x2x2xf32>
    %v120 = stablehlo.constant dense<0.0> : tensor<f32>
    %v121 = "stablehlo.select_and_scatter"(%v118, %v119, %v120) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x4x4xf32>, tensor<128x32x2x2xf32>, tensor<f32>) -> tensor<128x32x4x4xf32>
    %v122 = stablehlo.reshape %v121 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v123 = stablehlo.reshape %v122 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v124 = stablehlo.reshape %v79 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v125 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v126 = stablehlo.compare GT, %v124, %v125 : (tensor<128x32x4x4xf32>, tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xi1>
    %v127 = stablehlo.select %v126, %v123, %v125 : tensor<128x32x4x4xi1>, tensor<128x32x4x4xf32>
    %v128 = stablehlo.reshape %v127 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v129 = stablehlo.reshape %v128 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v130 = stablehlo.transpose %W8, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v131 = stablehlo.reverse %v130, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v132 = stablehlo.convolution(%v129, %v131)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v133 = stablehlo.reshape %v132 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v134 = stablehlo.reshape %v133 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v135 = stablehlo.reshape %v70 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v136 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v137 = stablehlo.compare GT, %v135, %v136 : (tensor<128x32x4x4xf32>, tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xi1>
    %v138 = stablehlo.select %v137, %v134, %v136 : tensor<128x32x4x4xi1>, tensor<128x32x4x4xf32>
    %v139 = stablehlo.reshape %v138 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v140 = stablehlo.reshape %v139 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v141 = stablehlo.transpose %W7, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v142 = stablehlo.reverse %v141, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v143 = stablehlo.convolution(%v140, %v142)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v144 = stablehlo.reshape %v143 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v145 = stablehlo.reshape %v61 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v146 = stablehlo.reshape %v144 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v147 = stablehlo.constant dense<0.0> : tensor<f32>
    %v148 = "stablehlo.select_and_scatter"(%v145, %v146, %v147) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x8x8xf32>, tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32x8x8xf32>
    %v149 = stablehlo.reshape %v148 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v150 = stablehlo.reshape %v149 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v151 = stablehlo.reshape %v57 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v152 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v153 = stablehlo.compare GT, %v151, %v152 : (tensor<128x32x8x8xf32>, tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xi1>
    %v154 = stablehlo.select %v153, %v150, %v152 : tensor<128x32x8x8xi1>, tensor<128x32x8x8xf32>
    %v155 = stablehlo.reshape %v154 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v156 = stablehlo.reshape %v155 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v157 = stablehlo.transpose %W6, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v158 = stablehlo.reverse %v157, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v159 = stablehlo.convolution(%v156, %v158)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x8x8xf32>
    %v160 = stablehlo.reshape %v159 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v161 = stablehlo.reshape %v160 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v162 = stablehlo.reshape %v48 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v163 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v164 = stablehlo.compare GT, %v162, %v163 : (tensor<128x32x8x8xf32>, tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xi1>
    %v165 = stablehlo.select %v164, %v161, %v163 : tensor<128x32x8x8xi1>, tensor<128x32x8x8xf32>
    %v166 = stablehlo.reshape %v165 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v167 = stablehlo.reshape %v166 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v168 = stablehlo.transpose %W5, dims = [1, 0, 2, 3] : (tensor<32x16x3x3xf32>) -> tensor<16x32x3x3xf32>
    %v169 = stablehlo.reverse %v168, dims = [2, 3] : tensor<16x32x3x3xf32>
    %v170 = stablehlo.convolution(%v167, %v169)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<16x32x3x3xf32>) -> tensor<128x16x8x8xf32>
    %v171 = stablehlo.reshape %v170 : (tensor<128x16x8x8xf32>) -> tensor<128x1024xf32>
    %v172 = stablehlo.reshape %v39 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v173 = stablehlo.reshape %v171 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v174 = stablehlo.constant dense<0.0> : tensor<f32>
    %v175 = "stablehlo.select_and_scatter"(%v172, %v173, %v174) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x16x16xf32>, tensor<128x16x8x8xf32>, tensor<f32>) -> tensor<128x16x16x16xf32>
    %v176 = stablehlo.reshape %v175 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v177 = stablehlo.reshape %v176 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v178 = stablehlo.reshape %v35 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v179 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v180 = stablehlo.compare GT, %v178, %v179 : (tensor<128x16x16x16xf32>, tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xi1>
    %v181 = stablehlo.select %v180, %v177, %v179 : tensor<128x16x16x16xi1>, tensor<128x16x16x16xf32>
    %v182 = stablehlo.reshape %v181 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v183 = stablehlo.reshape %v182 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v184 = stablehlo.transpose %W4, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v185 = stablehlo.reverse %v184, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v186 = stablehlo.convolution(%v183, %v185)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v187 = stablehlo.reshape %v186 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v188 = stablehlo.reshape %v187 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v189 = stablehlo.reshape %v26 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v190 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v191 = stablehlo.compare GT, %v189, %v190 : (tensor<128x16x16x16xf32>, tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xi1>
    %v192 = stablehlo.select %v191, %v188, %v190 : tensor<128x16x16x16xi1>, tensor<128x16x16x16xf32>
    %v193 = stablehlo.reshape %v192 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v194 = stablehlo.reshape %v193 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v195 = stablehlo.transpose %W3, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v196 = stablehlo.reverse %v195, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v197 = stablehlo.convolution(%v194, %v196)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v198 = stablehlo.reshape %v197 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v199 = stablehlo.reshape %v17 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v200 = stablehlo.reshape %v198 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v201 = stablehlo.constant dense<0.0> : tensor<f32>
    %v202 = "stablehlo.select_and_scatter"(%v199, %v200, %v201) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x32x32xf32>, tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x32x32xf32>
    %v203 = stablehlo.reshape %v202 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v204 = stablehlo.reshape %v203 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v205 = stablehlo.reshape %v13 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v206 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v207 = stablehlo.compare GT, %v205, %v206 : (tensor<128x16x32x32xf32>, tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xi1>
    %v208 = stablehlo.select %v207, %v204, %v206 : tensor<128x16x32x32xi1>, tensor<128x16x32x32xf32>
    %v209 = stablehlo.reshape %v208 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v210 = stablehlo.reshape %v209 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v211 = stablehlo.transpose %W2, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v212 = stablehlo.reverse %v211, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v213 = stablehlo.convolution(%v210, %v212)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x32x32xf32>
    %v214 = stablehlo.reshape %v213 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v215 = stablehlo.reshape %v214 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v216 = stablehlo.reshape %v4 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v217 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v218 = stablehlo.compare GT, %v216, %v217 : (tensor<128x16x32x32xf32>, tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xi1>
    %v219 = stablehlo.select %v218, %v215, %v217 : tensor<128x16x32x32xi1>, tensor<128x16x32x32xf32>
    %v220 = stablehlo.reshape %v219 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v221 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v222 = stablehlo.reshape %v220 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v223 = stablehlo.transpose %v221, dims = [1, 0, 2, 3] : (tensor<128x3x32x32xf32>) -> tensor<3x128x32x32xf32>
    %v224 = stablehlo.transpose %v222, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v225 = stablehlo.convolution(%v223, %v224)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<3x16x3x3xf32>
    %v226 = stablehlo.transpose %v225, dims = [1, 0, 2, 3] : (tensor<3x16x3x3xf32>) -> tensor<16x3x3x3xf32>
    %v227 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v228 = stablehlo.multiply %v227, %v226 : tensor<16x3x3x3xf32>
    %v229 = stablehlo.subtract %W1, %v228 : tensor<16x3x3x3xf32>
    %v230 = stablehlo.reshape %v220 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v231 = stablehlo.constant dense<0.0> : tensor<f32>
    %v232 = stablehlo.reduce(%v230 init: %v231) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v233 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v234 = stablehlo.multiply %v233, %v232 : tensor<16xf32>
    %v235 = stablehlo.subtract %cb1, %v234 : tensor<16xf32>
    %v236 = stablehlo.reshape %v8 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v237 = stablehlo.reshape %v209 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v238 = stablehlo.transpose %v236, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v239 = stablehlo.transpose %v237, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v240 = stablehlo.convolution(%v238, %v239)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x16x3x3xf32>
    %v241 = stablehlo.transpose %v240, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v242 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v243 = stablehlo.multiply %v242, %v241 : tensor<16x16x3x3xf32>
    %v244 = stablehlo.subtract %W2, %v243 : tensor<16x16x3x3xf32>
    %v245 = stablehlo.reshape %v209 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v246 = stablehlo.constant dense<0.0> : tensor<f32>
    %v247 = stablehlo.reduce(%v245 init: %v246) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v248 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v249 = stablehlo.multiply %v248, %v247 : tensor<16xf32>
    %v250 = stablehlo.subtract %cb2, %v249 : tensor<16xf32>
    %v251 = stablehlo.reshape %v21 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v252 = stablehlo.reshape %v193 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v253 = stablehlo.transpose %v251, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v254 = stablehlo.transpose %v252, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v255 = stablehlo.convolution(%v253, %v254)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v256 = stablehlo.transpose %v255, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v257 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v258 = stablehlo.multiply %v257, %v256 : tensor<16x16x3x3xf32>
    %v259 = stablehlo.subtract %W3, %v258 : tensor<16x16x3x3xf32>
    %v260 = stablehlo.reshape %v193 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v261 = stablehlo.constant dense<0.0> : tensor<f32>
    %v262 = stablehlo.reduce(%v260 init: %v261) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v263 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v264 = stablehlo.multiply %v263, %v262 : tensor<16xf32>
    %v265 = stablehlo.subtract %cb3, %v264 : tensor<16xf32>
    %v266 = stablehlo.reshape %v30 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v267 = stablehlo.reshape %v182 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v268 = stablehlo.transpose %v266, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v269 = stablehlo.transpose %v267, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v270 = stablehlo.convolution(%v268, %v269)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v271 = stablehlo.transpose %v270, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v272 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v273 = stablehlo.multiply %v272, %v271 : tensor<16x16x3x3xf32>
    %v274 = stablehlo.subtract %W4, %v273 : tensor<16x16x3x3xf32>
    %v275 = stablehlo.reshape %v182 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v276 = stablehlo.constant dense<0.0> : tensor<f32>
    %v277 = stablehlo.reduce(%v275 init: %v276) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v278 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v279 = stablehlo.multiply %v278, %v277 : tensor<16xf32>
    %v280 = stablehlo.subtract %cb4, %v279 : tensor<16xf32>
    %v281 = stablehlo.reshape %v43 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v282 = stablehlo.reshape %v166 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v283 = stablehlo.transpose %v281, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v284 = stablehlo.transpose %v282, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v285 = stablehlo.convolution(%v283, %v284)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<16x32x3x3xf32>
    %v286 = stablehlo.transpose %v285, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v287 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v288 = stablehlo.multiply %v287, %v286 : tensor<32x16x3x3xf32>
    %v289 = stablehlo.subtract %W5, %v288 : tensor<32x16x3x3xf32>
    %v290 = stablehlo.reshape %v166 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v291 = stablehlo.constant dense<0.0> : tensor<f32>
    %v292 = stablehlo.reduce(%v290 init: %v291) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v293 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v294 = stablehlo.multiply %v293, %v292 : tensor<32xf32>
    %v295 = stablehlo.subtract %cb5, %v294 : tensor<32xf32>
    %v296 = stablehlo.reshape %v52 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v297 = stablehlo.reshape %v155 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v298 = stablehlo.transpose %v296, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v299 = stablehlo.transpose %v297, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v300 = stablehlo.convolution(%v298, %v299)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<32x32x3x3xf32>
    %v301 = stablehlo.transpose %v300, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v302 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v303 = stablehlo.multiply %v302, %v301 : tensor<32x32x3x3xf32>
    %v304 = stablehlo.subtract %W6, %v303 : tensor<32x32x3x3xf32>
    %v305 = stablehlo.reshape %v155 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v306 = stablehlo.constant dense<0.0> : tensor<f32>
    %v307 = stablehlo.reduce(%v305 init: %v306) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v308 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v309 = stablehlo.multiply %v308, %v307 : tensor<32xf32>
    %v310 = stablehlo.subtract %cb6, %v309 : tensor<32xf32>
    %v311 = stablehlo.reshape %v65 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v312 = stablehlo.reshape %v139 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v313 = stablehlo.transpose %v311, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v314 = stablehlo.transpose %v312, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v315 = stablehlo.convolution(%v313, %v314)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v316 = stablehlo.transpose %v315, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v317 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v318 = stablehlo.multiply %v317, %v316 : tensor<32x32x3x3xf32>
    %v319 = stablehlo.subtract %W7, %v318 : tensor<32x32x3x3xf32>
    %v320 = stablehlo.reshape %v139 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v321 = stablehlo.constant dense<0.0> : tensor<f32>
    %v322 = stablehlo.reduce(%v320 init: %v321) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v323 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v324 = stablehlo.multiply %v323, %v322 : tensor<32xf32>
    %v325 = stablehlo.subtract %cb7, %v324 : tensor<32xf32>
    %v326 = stablehlo.reshape %v74 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v327 = stablehlo.reshape %v128 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v328 = stablehlo.transpose %v326, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v329 = stablehlo.transpose %v327, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v330 = stablehlo.convolution(%v328, %v329)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v331 = stablehlo.transpose %v330, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v332 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v333 = stablehlo.multiply %v332, %v331 : tensor<32x32x3x3xf32>
    %v334 = stablehlo.subtract %W8, %v333 : tensor<32x32x3x3xf32>
    %v335 = stablehlo.reshape %v128 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v336 = stablehlo.constant dense<0.0> : tensor<f32>
    %v337 = stablehlo.reduce(%v335 init: %v336) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v338 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v339 = stablehlo.multiply %v338, %v337 : tensor<32xf32>
    %v340 = stablehlo.subtract %cb8, %v339 : tensor<32xf32>
    %v341 = stablehlo.dot_general %v87, %v116, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %v342 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v343 = stablehlo.multiply %v342, %v341 : tensor<128x64xf32>
    %v344 = stablehlo.subtract %W9, %v343 : tensor<128x64xf32>
    %v345 = stablehlo.constant dense<0.0> : tensor<f32>
    %v346 = stablehlo.reduce(%v116 init: %v345) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v347 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v348 = stablehlo.multiply %v347, %v346 : tensor<64xf32>
    %v349 = stablehlo.subtract %b9, %v348 : tensor<64xf32>
    %v350 = stablehlo.dot_general %v92, %v112, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<64x64xf32>
    %v351 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v352 = stablehlo.multiply %v351, %v350 : tensor<64x64xf32>
    %v353 = stablehlo.subtract %Wa, %v352 : tensor<64x64xf32>
    %v354 = stablehlo.constant dense<0.0> : tensor<f32>
    %v355 = stablehlo.reduce(%v112 init: %v354) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v356 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v357 = stablehlo.multiply %v356, %v355 : tensor<64xf32>
    %v358 = stablehlo.subtract %ba, %v357 : tensor<64xf32>
    %v359 = stablehlo.dot_general %v97, %v108, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x10xf32>) -> tensor<64x10xf32>
    %v360 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v361 = stablehlo.multiply %v360, %v359 : tensor<64x10xf32>
    %v362 = stablehlo.subtract %Wb, %v361 : tensor<64x10xf32>
    %v363 = stablehlo.constant dense<0.0> : tensor<f32>
    %v364 = stablehlo.reduce(%v108 init: %v363) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v365 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v366 = stablehlo.multiply %v365, %v364 : tensor<10xf32>
    %v367 = stablehlo.subtract %bb, %v366 : tensor<10xf32>
    return %v229, %v235, %v244, %v250, %v259, %v265, %v274, %v280, %v289, %v295, %v304, %v310, %v319, %v325, %v334, %v340, %v344, %v349, %v353, %v358, %v362, %v367, %W1m, %cb1m, %W2m, %cb2m, %W3m, %cb3m, %W4m, %cb4m, %W5m, %cb5m, %W6m, %cb6m, %W7m, %cb7m, %W8m, %cb8m, %W9m, %b9m, %Wam, %bam, %Wbm, %bbm, %W1v, %cb1v, %W2v, %cb2v, %W3v, %cb3v, %W4v, %cb4v, %W5v, %cb5v, %W6v, %cb6v, %W7v, %cb7v, %W8v, %cb8v, %W9v, %b9v, %Wav, %bav, %Wbv, %bbv, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
