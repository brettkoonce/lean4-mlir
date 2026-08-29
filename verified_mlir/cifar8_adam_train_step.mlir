module @m {
  func.func @cifar8_adam_train_step(%x: tensor<128x3072xf32>, %W1: tensor<16x3x3x3xf32>, %cb1: tensor<16xf32>, %W2: tensor<16x16x3x3xf32>, %cb2: tensor<16xf32>, %W3: tensor<16x16x3x3xf32>, %cb3: tensor<16xf32>, %W4: tensor<16x16x3x3xf32>, %cb4: tensor<16xf32>, %W5: tensor<32x16x3x3xf32>, %cb5: tensor<32xf32>, %W6: tensor<32x32x3x3xf32>, %cb6: tensor<32xf32>, %W7: tensor<32x32x3x3xf32>, %cb7: tensor<32xf32>, %W8: tensor<32x32x3x3xf32>, %cb8: tensor<32xf32>, %W9: tensor<128x64xf32>, %b9: tensor<64xf32>, %Wa: tensor<64x64xf32>, %ba: tensor<64xf32>, %Wb: tensor<64x10xf32>, %bb: tensor<10xf32>, %W1m: tensor<16x3x3x3xf32>, %cb1m: tensor<16xf32>, %W2m: tensor<16x16x3x3xf32>, %cb2m: tensor<16xf32>, %W3m: tensor<16x16x3x3xf32>, %cb3m: tensor<16xf32>, %W4m: tensor<16x16x3x3xf32>, %cb4m: tensor<16xf32>, %W5m: tensor<32x16x3x3xf32>, %cb5m: tensor<32xf32>, %W6m: tensor<32x32x3x3xf32>, %cb6m: tensor<32xf32>, %W7m: tensor<32x32x3x3xf32>, %cb7m: tensor<32xf32>, %W8m: tensor<32x32x3x3xf32>, %cb8m: tensor<32xf32>, %W9m: tensor<128x64xf32>, %b9m: tensor<64xf32>, %Wam: tensor<64x64xf32>, %bam: tensor<64xf32>, %Wbm: tensor<64x10xf32>, %bbm: tensor<10xf32>, %W1v: tensor<16x3x3x3xf32>, %cb1v: tensor<16xf32>, %W2v: tensor<16x16x3x3xf32>, %cb2v: tensor<16xf32>, %W3v: tensor<16x16x3x3xf32>, %cb3v: tensor<16xf32>, %W4v: tensor<16x16x3x3xf32>, %cb4v: tensor<16xf32>, %W5v: tensor<32x16x3x3xf32>, %cb5v: tensor<32xf32>, %W6v: tensor<32x32x3x3xf32>, %cb6v: tensor<32xf32>, %W7v: tensor<32x32x3x3xf32>, %cb7v: tensor<32xf32>, %W8v: tensor<32x32x3x3xf32>, %cb8v: tensor<32xf32>, %W9v: tensor<128x64xf32>, %b9v: tensor<64xf32>, %Wav: tensor<64x64xf32>, %bav: tensor<64xf32>, %Wbv: tensor<64x10xf32>, %bbv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: tensor<128x10xf32>) -> (tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>) {
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
    %v227 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v228 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v229 = stablehlo.multiply %v227, %W1m : tensor<16x3x3x3xf32>
    %v230 = stablehlo.multiply %v228, %v226 : tensor<16x3x3x3xf32>
    %v231 = stablehlo.add %v229, %v230 : tensor<16x3x3x3xf32>
    %v232 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v233 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v234 = stablehlo.multiply %v232, %W1v : tensor<16x3x3x3xf32>
    %v235 = stablehlo.multiply %v226, %v226 : tensor<16x3x3x3xf32>
    %v236 = stablehlo.multiply %v233, %v235 : tensor<16x3x3x3xf32>
    %v237 = stablehlo.add %v234, %v236 : tensor<16x3x3x3xf32>
    %v238 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v239 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v240 = stablehlo.divide %v231, %v238 : tensor<16x3x3x3xf32>
    %v241 = stablehlo.divide %v237, %v239 : tensor<16x3x3x3xf32>
    %v242 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v243 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v244 = stablehlo.sqrt %v241 : tensor<16x3x3x3xf32>
    %v245 = stablehlo.add %v244, %v243 : tensor<16x3x3x3xf32>
    %v246 = stablehlo.divide %v240, %v245 : tensor<16x3x3x3xf32>
    %v247 = stablehlo.multiply %v242, %v246 : tensor<16x3x3x3xf32>
    %v248 = stablehlo.subtract %W1, %v247 : tensor<16x3x3x3xf32>
    %v249 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v250 = stablehlo.multiply %v249, %v242 : tensor<16x3x3x3xf32>
    %v251 = stablehlo.multiply %v250, %W1 : tensor<16x3x3x3xf32>
    %v252 = stablehlo.subtract %v248, %v251 : tensor<16x3x3x3xf32>
    %v253 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v254 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v255 = stablehlo.multiply %v253, %W1m : tensor<16x3x3x3xf32>
    %v256 = stablehlo.multiply %v254, %v226 : tensor<16x3x3x3xf32>
    %v257 = stablehlo.add %v255, %v256 : tensor<16x3x3x3xf32>
    %v258 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v259 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v260 = stablehlo.multiply %v258, %W1v : tensor<16x3x3x3xf32>
    %v261 = stablehlo.multiply %v226, %v226 : tensor<16x3x3x3xf32>
    %v262 = stablehlo.multiply %v259, %v261 : tensor<16x3x3x3xf32>
    %v263 = stablehlo.add %v260, %v262 : tensor<16x3x3x3xf32>
    %v264 = stablehlo.reshape %v220 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v265 = stablehlo.constant dense<0.0> : tensor<f32>
    %v266 = stablehlo.reduce(%v264 init: %v265) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v267 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v268 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v269 = stablehlo.multiply %v267, %cb1m : tensor<16xf32>
    %v270 = stablehlo.multiply %v268, %v266 : tensor<16xf32>
    %v271 = stablehlo.add %v269, %v270 : tensor<16xf32>
    %v272 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v273 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v274 = stablehlo.multiply %v272, %cb1v : tensor<16xf32>
    %v275 = stablehlo.multiply %v266, %v266 : tensor<16xf32>
    %v276 = stablehlo.multiply %v273, %v275 : tensor<16xf32>
    %v277 = stablehlo.add %v274, %v276 : tensor<16xf32>
    %v278 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v279 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v280 = stablehlo.divide %v271, %v278 : tensor<16xf32>
    %v281 = stablehlo.divide %v277, %v279 : tensor<16xf32>
    %v282 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v283 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v284 = stablehlo.sqrt %v281 : tensor<16xf32>
    %v285 = stablehlo.add %v284, %v283 : tensor<16xf32>
    %v286 = stablehlo.divide %v280, %v285 : tensor<16xf32>
    %v287 = stablehlo.multiply %v282, %v286 : tensor<16xf32>
    %v288 = stablehlo.subtract %cb1, %v287 : tensor<16xf32>
    %v289 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v290 = stablehlo.multiply %v289, %v282 : tensor<16xf32>
    %v291 = stablehlo.multiply %v290, %cb1 : tensor<16xf32>
    %v292 = stablehlo.subtract %v288, %v291 : tensor<16xf32>
    %v293 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v294 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v295 = stablehlo.multiply %v293, %cb1m : tensor<16xf32>
    %v296 = stablehlo.multiply %v294, %v266 : tensor<16xf32>
    %v297 = stablehlo.add %v295, %v296 : tensor<16xf32>
    %v298 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v299 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v300 = stablehlo.multiply %v298, %cb1v : tensor<16xf32>
    %v301 = stablehlo.multiply %v266, %v266 : tensor<16xf32>
    %v302 = stablehlo.multiply %v299, %v301 : tensor<16xf32>
    %v303 = stablehlo.add %v300, %v302 : tensor<16xf32>
    %v304 = stablehlo.reshape %v8 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v305 = stablehlo.reshape %v209 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v306 = stablehlo.transpose %v304, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v307 = stablehlo.transpose %v305, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v308 = stablehlo.convolution(%v306, %v307)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x16x3x3xf32>
    %v309 = stablehlo.transpose %v308, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v310 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v311 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v312 = stablehlo.multiply %v310, %W2m : tensor<16x16x3x3xf32>
    %v313 = stablehlo.multiply %v311, %v309 : tensor<16x16x3x3xf32>
    %v314 = stablehlo.add %v312, %v313 : tensor<16x16x3x3xf32>
    %v315 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v316 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v317 = stablehlo.multiply %v315, %W2v : tensor<16x16x3x3xf32>
    %v318 = stablehlo.multiply %v309, %v309 : tensor<16x16x3x3xf32>
    %v319 = stablehlo.multiply %v316, %v318 : tensor<16x16x3x3xf32>
    %v320 = stablehlo.add %v317, %v319 : tensor<16x16x3x3xf32>
    %v321 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v322 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v323 = stablehlo.divide %v314, %v321 : tensor<16x16x3x3xf32>
    %v324 = stablehlo.divide %v320, %v322 : tensor<16x16x3x3xf32>
    %v325 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v326 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v327 = stablehlo.sqrt %v324 : tensor<16x16x3x3xf32>
    %v328 = stablehlo.add %v327, %v326 : tensor<16x16x3x3xf32>
    %v329 = stablehlo.divide %v323, %v328 : tensor<16x16x3x3xf32>
    %v330 = stablehlo.multiply %v325, %v329 : tensor<16x16x3x3xf32>
    %v331 = stablehlo.subtract %W2, %v330 : tensor<16x16x3x3xf32>
    %v332 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v333 = stablehlo.multiply %v332, %v325 : tensor<16x16x3x3xf32>
    %v334 = stablehlo.multiply %v333, %W2 : tensor<16x16x3x3xf32>
    %v335 = stablehlo.subtract %v331, %v334 : tensor<16x16x3x3xf32>
    %v336 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v337 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v338 = stablehlo.multiply %v336, %W2m : tensor<16x16x3x3xf32>
    %v339 = stablehlo.multiply %v337, %v309 : tensor<16x16x3x3xf32>
    %v340 = stablehlo.add %v338, %v339 : tensor<16x16x3x3xf32>
    %v341 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v342 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v343 = stablehlo.multiply %v341, %W2v : tensor<16x16x3x3xf32>
    %v344 = stablehlo.multiply %v309, %v309 : tensor<16x16x3x3xf32>
    %v345 = stablehlo.multiply %v342, %v344 : tensor<16x16x3x3xf32>
    %v346 = stablehlo.add %v343, %v345 : tensor<16x16x3x3xf32>
    %v347 = stablehlo.reshape %v209 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v348 = stablehlo.constant dense<0.0> : tensor<f32>
    %v349 = stablehlo.reduce(%v347 init: %v348) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v350 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v351 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v352 = stablehlo.multiply %v350, %cb2m : tensor<16xf32>
    %v353 = stablehlo.multiply %v351, %v349 : tensor<16xf32>
    %v354 = stablehlo.add %v352, %v353 : tensor<16xf32>
    %v355 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v356 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v357 = stablehlo.multiply %v355, %cb2v : tensor<16xf32>
    %v358 = stablehlo.multiply %v349, %v349 : tensor<16xf32>
    %v359 = stablehlo.multiply %v356, %v358 : tensor<16xf32>
    %v360 = stablehlo.add %v357, %v359 : tensor<16xf32>
    %v361 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v362 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v363 = stablehlo.divide %v354, %v361 : tensor<16xf32>
    %v364 = stablehlo.divide %v360, %v362 : tensor<16xf32>
    %v365 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v366 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v367 = stablehlo.sqrt %v364 : tensor<16xf32>
    %v368 = stablehlo.add %v367, %v366 : tensor<16xf32>
    %v369 = stablehlo.divide %v363, %v368 : tensor<16xf32>
    %v370 = stablehlo.multiply %v365, %v369 : tensor<16xf32>
    %v371 = stablehlo.subtract %cb2, %v370 : tensor<16xf32>
    %v372 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v373 = stablehlo.multiply %v372, %v365 : tensor<16xf32>
    %v374 = stablehlo.multiply %v373, %cb2 : tensor<16xf32>
    %v375 = stablehlo.subtract %v371, %v374 : tensor<16xf32>
    %v376 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v377 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v378 = stablehlo.multiply %v376, %cb2m : tensor<16xf32>
    %v379 = stablehlo.multiply %v377, %v349 : tensor<16xf32>
    %v380 = stablehlo.add %v378, %v379 : tensor<16xf32>
    %v381 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v382 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v383 = stablehlo.multiply %v381, %cb2v : tensor<16xf32>
    %v384 = stablehlo.multiply %v349, %v349 : tensor<16xf32>
    %v385 = stablehlo.multiply %v382, %v384 : tensor<16xf32>
    %v386 = stablehlo.add %v383, %v385 : tensor<16xf32>
    %v387 = stablehlo.reshape %v21 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v388 = stablehlo.reshape %v193 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v389 = stablehlo.transpose %v387, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v390 = stablehlo.transpose %v388, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v391 = stablehlo.convolution(%v389, %v390)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v392 = stablehlo.transpose %v391, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v393 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v394 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v395 = stablehlo.multiply %v393, %W3m : tensor<16x16x3x3xf32>
    %v396 = stablehlo.multiply %v394, %v392 : tensor<16x16x3x3xf32>
    %v397 = stablehlo.add %v395, %v396 : tensor<16x16x3x3xf32>
    %v398 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v399 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v400 = stablehlo.multiply %v398, %W3v : tensor<16x16x3x3xf32>
    %v401 = stablehlo.multiply %v392, %v392 : tensor<16x16x3x3xf32>
    %v402 = stablehlo.multiply %v399, %v401 : tensor<16x16x3x3xf32>
    %v403 = stablehlo.add %v400, %v402 : tensor<16x16x3x3xf32>
    %v404 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v405 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v406 = stablehlo.divide %v397, %v404 : tensor<16x16x3x3xf32>
    %v407 = stablehlo.divide %v403, %v405 : tensor<16x16x3x3xf32>
    %v408 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v409 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v410 = stablehlo.sqrt %v407 : tensor<16x16x3x3xf32>
    %v411 = stablehlo.add %v410, %v409 : tensor<16x16x3x3xf32>
    %v412 = stablehlo.divide %v406, %v411 : tensor<16x16x3x3xf32>
    %v413 = stablehlo.multiply %v408, %v412 : tensor<16x16x3x3xf32>
    %v414 = stablehlo.subtract %W3, %v413 : tensor<16x16x3x3xf32>
    %v415 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v416 = stablehlo.multiply %v415, %v408 : tensor<16x16x3x3xf32>
    %v417 = stablehlo.multiply %v416, %W3 : tensor<16x16x3x3xf32>
    %v418 = stablehlo.subtract %v414, %v417 : tensor<16x16x3x3xf32>
    %v419 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v420 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v421 = stablehlo.multiply %v419, %W3m : tensor<16x16x3x3xf32>
    %v422 = stablehlo.multiply %v420, %v392 : tensor<16x16x3x3xf32>
    %v423 = stablehlo.add %v421, %v422 : tensor<16x16x3x3xf32>
    %v424 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v425 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v426 = stablehlo.multiply %v424, %W3v : tensor<16x16x3x3xf32>
    %v427 = stablehlo.multiply %v392, %v392 : tensor<16x16x3x3xf32>
    %v428 = stablehlo.multiply %v425, %v427 : tensor<16x16x3x3xf32>
    %v429 = stablehlo.add %v426, %v428 : tensor<16x16x3x3xf32>
    %v430 = stablehlo.reshape %v193 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v431 = stablehlo.constant dense<0.0> : tensor<f32>
    %v432 = stablehlo.reduce(%v430 init: %v431) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v433 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v434 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v435 = stablehlo.multiply %v433, %cb3m : tensor<16xf32>
    %v436 = stablehlo.multiply %v434, %v432 : tensor<16xf32>
    %v437 = stablehlo.add %v435, %v436 : tensor<16xf32>
    %v438 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v439 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v440 = stablehlo.multiply %v438, %cb3v : tensor<16xf32>
    %v441 = stablehlo.multiply %v432, %v432 : tensor<16xf32>
    %v442 = stablehlo.multiply %v439, %v441 : tensor<16xf32>
    %v443 = stablehlo.add %v440, %v442 : tensor<16xf32>
    %v444 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v445 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v446 = stablehlo.divide %v437, %v444 : tensor<16xf32>
    %v447 = stablehlo.divide %v443, %v445 : tensor<16xf32>
    %v448 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v449 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v450 = stablehlo.sqrt %v447 : tensor<16xf32>
    %v451 = stablehlo.add %v450, %v449 : tensor<16xf32>
    %v452 = stablehlo.divide %v446, %v451 : tensor<16xf32>
    %v453 = stablehlo.multiply %v448, %v452 : tensor<16xf32>
    %v454 = stablehlo.subtract %cb3, %v453 : tensor<16xf32>
    %v455 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v456 = stablehlo.multiply %v455, %v448 : tensor<16xf32>
    %v457 = stablehlo.multiply %v456, %cb3 : tensor<16xf32>
    %v458 = stablehlo.subtract %v454, %v457 : tensor<16xf32>
    %v459 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v460 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v461 = stablehlo.multiply %v459, %cb3m : tensor<16xf32>
    %v462 = stablehlo.multiply %v460, %v432 : tensor<16xf32>
    %v463 = stablehlo.add %v461, %v462 : tensor<16xf32>
    %v464 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v465 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v466 = stablehlo.multiply %v464, %cb3v : tensor<16xf32>
    %v467 = stablehlo.multiply %v432, %v432 : tensor<16xf32>
    %v468 = stablehlo.multiply %v465, %v467 : tensor<16xf32>
    %v469 = stablehlo.add %v466, %v468 : tensor<16xf32>
    %v470 = stablehlo.reshape %v30 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v471 = stablehlo.reshape %v182 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v472 = stablehlo.transpose %v470, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v473 = stablehlo.transpose %v471, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v474 = stablehlo.convolution(%v472, %v473)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v475 = stablehlo.transpose %v474, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v476 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v477 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v478 = stablehlo.multiply %v476, %W4m : tensor<16x16x3x3xf32>
    %v479 = stablehlo.multiply %v477, %v475 : tensor<16x16x3x3xf32>
    %v480 = stablehlo.add %v478, %v479 : tensor<16x16x3x3xf32>
    %v481 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v482 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v483 = stablehlo.multiply %v481, %W4v : tensor<16x16x3x3xf32>
    %v484 = stablehlo.multiply %v475, %v475 : tensor<16x16x3x3xf32>
    %v485 = stablehlo.multiply %v482, %v484 : tensor<16x16x3x3xf32>
    %v486 = stablehlo.add %v483, %v485 : tensor<16x16x3x3xf32>
    %v487 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v488 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v489 = stablehlo.divide %v480, %v487 : tensor<16x16x3x3xf32>
    %v490 = stablehlo.divide %v486, %v488 : tensor<16x16x3x3xf32>
    %v491 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v492 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v493 = stablehlo.sqrt %v490 : tensor<16x16x3x3xf32>
    %v494 = stablehlo.add %v493, %v492 : tensor<16x16x3x3xf32>
    %v495 = stablehlo.divide %v489, %v494 : tensor<16x16x3x3xf32>
    %v496 = stablehlo.multiply %v491, %v495 : tensor<16x16x3x3xf32>
    %v497 = stablehlo.subtract %W4, %v496 : tensor<16x16x3x3xf32>
    %v498 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v499 = stablehlo.multiply %v498, %v491 : tensor<16x16x3x3xf32>
    %v500 = stablehlo.multiply %v499, %W4 : tensor<16x16x3x3xf32>
    %v501 = stablehlo.subtract %v497, %v500 : tensor<16x16x3x3xf32>
    %v502 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v503 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v504 = stablehlo.multiply %v502, %W4m : tensor<16x16x3x3xf32>
    %v505 = stablehlo.multiply %v503, %v475 : tensor<16x16x3x3xf32>
    %v506 = stablehlo.add %v504, %v505 : tensor<16x16x3x3xf32>
    %v507 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v508 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v509 = stablehlo.multiply %v507, %W4v : tensor<16x16x3x3xf32>
    %v510 = stablehlo.multiply %v475, %v475 : tensor<16x16x3x3xf32>
    %v511 = stablehlo.multiply %v508, %v510 : tensor<16x16x3x3xf32>
    %v512 = stablehlo.add %v509, %v511 : tensor<16x16x3x3xf32>
    %v513 = stablehlo.reshape %v182 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v514 = stablehlo.constant dense<0.0> : tensor<f32>
    %v515 = stablehlo.reduce(%v513 init: %v514) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v516 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v517 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v518 = stablehlo.multiply %v516, %cb4m : tensor<16xf32>
    %v519 = stablehlo.multiply %v517, %v515 : tensor<16xf32>
    %v520 = stablehlo.add %v518, %v519 : tensor<16xf32>
    %v521 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v522 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v523 = stablehlo.multiply %v521, %cb4v : tensor<16xf32>
    %v524 = stablehlo.multiply %v515, %v515 : tensor<16xf32>
    %v525 = stablehlo.multiply %v522, %v524 : tensor<16xf32>
    %v526 = stablehlo.add %v523, %v525 : tensor<16xf32>
    %v527 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v528 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v529 = stablehlo.divide %v520, %v527 : tensor<16xf32>
    %v530 = stablehlo.divide %v526, %v528 : tensor<16xf32>
    %v531 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v532 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v533 = stablehlo.sqrt %v530 : tensor<16xf32>
    %v534 = stablehlo.add %v533, %v532 : tensor<16xf32>
    %v535 = stablehlo.divide %v529, %v534 : tensor<16xf32>
    %v536 = stablehlo.multiply %v531, %v535 : tensor<16xf32>
    %v537 = stablehlo.subtract %cb4, %v536 : tensor<16xf32>
    %v538 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v539 = stablehlo.multiply %v538, %v531 : tensor<16xf32>
    %v540 = stablehlo.multiply %v539, %cb4 : tensor<16xf32>
    %v541 = stablehlo.subtract %v537, %v540 : tensor<16xf32>
    %v542 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v543 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v544 = stablehlo.multiply %v542, %cb4m : tensor<16xf32>
    %v545 = stablehlo.multiply %v543, %v515 : tensor<16xf32>
    %v546 = stablehlo.add %v544, %v545 : tensor<16xf32>
    %v547 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v548 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v549 = stablehlo.multiply %v547, %cb4v : tensor<16xf32>
    %v550 = stablehlo.multiply %v515, %v515 : tensor<16xf32>
    %v551 = stablehlo.multiply %v548, %v550 : tensor<16xf32>
    %v552 = stablehlo.add %v549, %v551 : tensor<16xf32>
    %v553 = stablehlo.reshape %v43 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v554 = stablehlo.reshape %v166 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v555 = stablehlo.transpose %v553, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v556 = stablehlo.transpose %v554, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v557 = stablehlo.convolution(%v555, %v556)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<16x32x3x3xf32>
    %v558 = stablehlo.transpose %v557, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v559 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v560 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v561 = stablehlo.multiply %v559, %W5m : tensor<32x16x3x3xf32>
    %v562 = stablehlo.multiply %v560, %v558 : tensor<32x16x3x3xf32>
    %v563 = stablehlo.add %v561, %v562 : tensor<32x16x3x3xf32>
    %v564 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v565 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v566 = stablehlo.multiply %v564, %W5v : tensor<32x16x3x3xf32>
    %v567 = stablehlo.multiply %v558, %v558 : tensor<32x16x3x3xf32>
    %v568 = stablehlo.multiply %v565, %v567 : tensor<32x16x3x3xf32>
    %v569 = stablehlo.add %v566, %v568 : tensor<32x16x3x3xf32>
    %v570 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v571 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v572 = stablehlo.divide %v563, %v570 : tensor<32x16x3x3xf32>
    %v573 = stablehlo.divide %v569, %v571 : tensor<32x16x3x3xf32>
    %v574 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v575 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v576 = stablehlo.sqrt %v573 : tensor<32x16x3x3xf32>
    %v577 = stablehlo.add %v576, %v575 : tensor<32x16x3x3xf32>
    %v578 = stablehlo.divide %v572, %v577 : tensor<32x16x3x3xf32>
    %v579 = stablehlo.multiply %v574, %v578 : tensor<32x16x3x3xf32>
    %v580 = stablehlo.subtract %W5, %v579 : tensor<32x16x3x3xf32>
    %v581 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v582 = stablehlo.multiply %v581, %v574 : tensor<32x16x3x3xf32>
    %v583 = stablehlo.multiply %v582, %W5 : tensor<32x16x3x3xf32>
    %v584 = stablehlo.subtract %v580, %v583 : tensor<32x16x3x3xf32>
    %v585 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v586 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v587 = stablehlo.multiply %v585, %W5m : tensor<32x16x3x3xf32>
    %v588 = stablehlo.multiply %v586, %v558 : tensor<32x16x3x3xf32>
    %v589 = stablehlo.add %v587, %v588 : tensor<32x16x3x3xf32>
    %v590 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v591 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v592 = stablehlo.multiply %v590, %W5v : tensor<32x16x3x3xf32>
    %v593 = stablehlo.multiply %v558, %v558 : tensor<32x16x3x3xf32>
    %v594 = stablehlo.multiply %v591, %v593 : tensor<32x16x3x3xf32>
    %v595 = stablehlo.add %v592, %v594 : tensor<32x16x3x3xf32>
    %v596 = stablehlo.reshape %v166 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v597 = stablehlo.constant dense<0.0> : tensor<f32>
    %v598 = stablehlo.reduce(%v596 init: %v597) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v599 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v600 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v601 = stablehlo.multiply %v599, %cb5m : tensor<32xf32>
    %v602 = stablehlo.multiply %v600, %v598 : tensor<32xf32>
    %v603 = stablehlo.add %v601, %v602 : tensor<32xf32>
    %v604 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v605 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v606 = stablehlo.multiply %v604, %cb5v : tensor<32xf32>
    %v607 = stablehlo.multiply %v598, %v598 : tensor<32xf32>
    %v608 = stablehlo.multiply %v605, %v607 : tensor<32xf32>
    %v609 = stablehlo.add %v606, %v608 : tensor<32xf32>
    %v610 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v611 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v612 = stablehlo.divide %v603, %v610 : tensor<32xf32>
    %v613 = stablehlo.divide %v609, %v611 : tensor<32xf32>
    %v614 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v615 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v616 = stablehlo.sqrt %v613 : tensor<32xf32>
    %v617 = stablehlo.add %v616, %v615 : tensor<32xf32>
    %v618 = stablehlo.divide %v612, %v617 : tensor<32xf32>
    %v619 = stablehlo.multiply %v614, %v618 : tensor<32xf32>
    %v620 = stablehlo.subtract %cb5, %v619 : tensor<32xf32>
    %v621 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v622 = stablehlo.multiply %v621, %v614 : tensor<32xf32>
    %v623 = stablehlo.multiply %v622, %cb5 : tensor<32xf32>
    %v624 = stablehlo.subtract %v620, %v623 : tensor<32xf32>
    %v625 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v626 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v627 = stablehlo.multiply %v625, %cb5m : tensor<32xf32>
    %v628 = stablehlo.multiply %v626, %v598 : tensor<32xf32>
    %v629 = stablehlo.add %v627, %v628 : tensor<32xf32>
    %v630 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v631 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v632 = stablehlo.multiply %v630, %cb5v : tensor<32xf32>
    %v633 = stablehlo.multiply %v598, %v598 : tensor<32xf32>
    %v634 = stablehlo.multiply %v631, %v633 : tensor<32xf32>
    %v635 = stablehlo.add %v632, %v634 : tensor<32xf32>
    %v636 = stablehlo.reshape %v52 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v637 = stablehlo.reshape %v155 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v638 = stablehlo.transpose %v636, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v639 = stablehlo.transpose %v637, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v640 = stablehlo.convolution(%v638, %v639)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<32x32x3x3xf32>
    %v641 = stablehlo.transpose %v640, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v642 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v643 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v644 = stablehlo.multiply %v642, %W6m : tensor<32x32x3x3xf32>
    %v645 = stablehlo.multiply %v643, %v641 : tensor<32x32x3x3xf32>
    %v646 = stablehlo.add %v644, %v645 : tensor<32x32x3x3xf32>
    %v647 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v648 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v649 = stablehlo.multiply %v647, %W6v : tensor<32x32x3x3xf32>
    %v650 = stablehlo.multiply %v641, %v641 : tensor<32x32x3x3xf32>
    %v651 = stablehlo.multiply %v648, %v650 : tensor<32x32x3x3xf32>
    %v652 = stablehlo.add %v649, %v651 : tensor<32x32x3x3xf32>
    %v653 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v654 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v655 = stablehlo.divide %v646, %v653 : tensor<32x32x3x3xf32>
    %v656 = stablehlo.divide %v652, %v654 : tensor<32x32x3x3xf32>
    %v657 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v658 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v659 = stablehlo.sqrt %v656 : tensor<32x32x3x3xf32>
    %v660 = stablehlo.add %v659, %v658 : tensor<32x32x3x3xf32>
    %v661 = stablehlo.divide %v655, %v660 : tensor<32x32x3x3xf32>
    %v662 = stablehlo.multiply %v657, %v661 : tensor<32x32x3x3xf32>
    %v663 = stablehlo.subtract %W6, %v662 : tensor<32x32x3x3xf32>
    %v664 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v665 = stablehlo.multiply %v664, %v657 : tensor<32x32x3x3xf32>
    %v666 = stablehlo.multiply %v665, %W6 : tensor<32x32x3x3xf32>
    %v667 = stablehlo.subtract %v663, %v666 : tensor<32x32x3x3xf32>
    %v668 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v669 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v670 = stablehlo.multiply %v668, %W6m : tensor<32x32x3x3xf32>
    %v671 = stablehlo.multiply %v669, %v641 : tensor<32x32x3x3xf32>
    %v672 = stablehlo.add %v670, %v671 : tensor<32x32x3x3xf32>
    %v673 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v674 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v675 = stablehlo.multiply %v673, %W6v : tensor<32x32x3x3xf32>
    %v676 = stablehlo.multiply %v641, %v641 : tensor<32x32x3x3xf32>
    %v677 = stablehlo.multiply %v674, %v676 : tensor<32x32x3x3xf32>
    %v678 = stablehlo.add %v675, %v677 : tensor<32x32x3x3xf32>
    %v679 = stablehlo.reshape %v155 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v680 = stablehlo.constant dense<0.0> : tensor<f32>
    %v681 = stablehlo.reduce(%v679 init: %v680) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v682 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v683 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v684 = stablehlo.multiply %v682, %cb6m : tensor<32xf32>
    %v685 = stablehlo.multiply %v683, %v681 : tensor<32xf32>
    %v686 = stablehlo.add %v684, %v685 : tensor<32xf32>
    %v687 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v688 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v689 = stablehlo.multiply %v687, %cb6v : tensor<32xf32>
    %v690 = stablehlo.multiply %v681, %v681 : tensor<32xf32>
    %v691 = stablehlo.multiply %v688, %v690 : tensor<32xf32>
    %v692 = stablehlo.add %v689, %v691 : tensor<32xf32>
    %v693 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v694 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v695 = stablehlo.divide %v686, %v693 : tensor<32xf32>
    %v696 = stablehlo.divide %v692, %v694 : tensor<32xf32>
    %v697 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v698 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v699 = stablehlo.sqrt %v696 : tensor<32xf32>
    %v700 = stablehlo.add %v699, %v698 : tensor<32xf32>
    %v701 = stablehlo.divide %v695, %v700 : tensor<32xf32>
    %v702 = stablehlo.multiply %v697, %v701 : tensor<32xf32>
    %v703 = stablehlo.subtract %cb6, %v702 : tensor<32xf32>
    %v704 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v705 = stablehlo.multiply %v704, %v697 : tensor<32xf32>
    %v706 = stablehlo.multiply %v705, %cb6 : tensor<32xf32>
    %v707 = stablehlo.subtract %v703, %v706 : tensor<32xf32>
    %v708 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v709 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v710 = stablehlo.multiply %v708, %cb6m : tensor<32xf32>
    %v711 = stablehlo.multiply %v709, %v681 : tensor<32xf32>
    %v712 = stablehlo.add %v710, %v711 : tensor<32xf32>
    %v713 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v714 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v715 = stablehlo.multiply %v713, %cb6v : tensor<32xf32>
    %v716 = stablehlo.multiply %v681, %v681 : tensor<32xf32>
    %v717 = stablehlo.multiply %v714, %v716 : tensor<32xf32>
    %v718 = stablehlo.add %v715, %v717 : tensor<32xf32>
    %v719 = stablehlo.reshape %v65 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v720 = stablehlo.reshape %v139 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v721 = stablehlo.transpose %v719, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v722 = stablehlo.transpose %v720, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v723 = stablehlo.convolution(%v721, %v722)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v724 = stablehlo.transpose %v723, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v725 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v726 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v727 = stablehlo.multiply %v725, %W7m : tensor<32x32x3x3xf32>
    %v728 = stablehlo.multiply %v726, %v724 : tensor<32x32x3x3xf32>
    %v729 = stablehlo.add %v727, %v728 : tensor<32x32x3x3xf32>
    %v730 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v731 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v732 = stablehlo.multiply %v730, %W7v : tensor<32x32x3x3xf32>
    %v733 = stablehlo.multiply %v724, %v724 : tensor<32x32x3x3xf32>
    %v734 = stablehlo.multiply %v731, %v733 : tensor<32x32x3x3xf32>
    %v735 = stablehlo.add %v732, %v734 : tensor<32x32x3x3xf32>
    %v736 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v737 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v738 = stablehlo.divide %v729, %v736 : tensor<32x32x3x3xf32>
    %v739 = stablehlo.divide %v735, %v737 : tensor<32x32x3x3xf32>
    %v740 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v741 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v742 = stablehlo.sqrt %v739 : tensor<32x32x3x3xf32>
    %v743 = stablehlo.add %v742, %v741 : tensor<32x32x3x3xf32>
    %v744 = stablehlo.divide %v738, %v743 : tensor<32x32x3x3xf32>
    %v745 = stablehlo.multiply %v740, %v744 : tensor<32x32x3x3xf32>
    %v746 = stablehlo.subtract %W7, %v745 : tensor<32x32x3x3xf32>
    %v747 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v748 = stablehlo.multiply %v747, %v740 : tensor<32x32x3x3xf32>
    %v749 = stablehlo.multiply %v748, %W7 : tensor<32x32x3x3xf32>
    %v750 = stablehlo.subtract %v746, %v749 : tensor<32x32x3x3xf32>
    %v751 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v752 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v753 = stablehlo.multiply %v751, %W7m : tensor<32x32x3x3xf32>
    %v754 = stablehlo.multiply %v752, %v724 : tensor<32x32x3x3xf32>
    %v755 = stablehlo.add %v753, %v754 : tensor<32x32x3x3xf32>
    %v756 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v757 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v758 = stablehlo.multiply %v756, %W7v : tensor<32x32x3x3xf32>
    %v759 = stablehlo.multiply %v724, %v724 : tensor<32x32x3x3xf32>
    %v760 = stablehlo.multiply %v757, %v759 : tensor<32x32x3x3xf32>
    %v761 = stablehlo.add %v758, %v760 : tensor<32x32x3x3xf32>
    %v762 = stablehlo.reshape %v139 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v763 = stablehlo.constant dense<0.0> : tensor<f32>
    %v764 = stablehlo.reduce(%v762 init: %v763) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v765 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v766 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v767 = stablehlo.multiply %v765, %cb7m : tensor<32xf32>
    %v768 = stablehlo.multiply %v766, %v764 : tensor<32xf32>
    %v769 = stablehlo.add %v767, %v768 : tensor<32xf32>
    %v770 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v771 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v772 = stablehlo.multiply %v770, %cb7v : tensor<32xf32>
    %v773 = stablehlo.multiply %v764, %v764 : tensor<32xf32>
    %v774 = stablehlo.multiply %v771, %v773 : tensor<32xf32>
    %v775 = stablehlo.add %v772, %v774 : tensor<32xf32>
    %v776 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v777 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v778 = stablehlo.divide %v769, %v776 : tensor<32xf32>
    %v779 = stablehlo.divide %v775, %v777 : tensor<32xf32>
    %v780 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v781 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v782 = stablehlo.sqrt %v779 : tensor<32xf32>
    %v783 = stablehlo.add %v782, %v781 : tensor<32xf32>
    %v784 = stablehlo.divide %v778, %v783 : tensor<32xf32>
    %v785 = stablehlo.multiply %v780, %v784 : tensor<32xf32>
    %v786 = stablehlo.subtract %cb7, %v785 : tensor<32xf32>
    %v787 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v788 = stablehlo.multiply %v787, %v780 : tensor<32xf32>
    %v789 = stablehlo.multiply %v788, %cb7 : tensor<32xf32>
    %v790 = stablehlo.subtract %v786, %v789 : tensor<32xf32>
    %v791 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v792 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v793 = stablehlo.multiply %v791, %cb7m : tensor<32xf32>
    %v794 = stablehlo.multiply %v792, %v764 : tensor<32xf32>
    %v795 = stablehlo.add %v793, %v794 : tensor<32xf32>
    %v796 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v797 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v798 = stablehlo.multiply %v796, %cb7v : tensor<32xf32>
    %v799 = stablehlo.multiply %v764, %v764 : tensor<32xf32>
    %v800 = stablehlo.multiply %v797, %v799 : tensor<32xf32>
    %v801 = stablehlo.add %v798, %v800 : tensor<32xf32>
    %v802 = stablehlo.reshape %v74 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v803 = stablehlo.reshape %v128 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v804 = stablehlo.transpose %v802, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v805 = stablehlo.transpose %v803, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v806 = stablehlo.convolution(%v804, %v805)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v807 = stablehlo.transpose %v806, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v808 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v809 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v810 = stablehlo.multiply %v808, %W8m : tensor<32x32x3x3xf32>
    %v811 = stablehlo.multiply %v809, %v807 : tensor<32x32x3x3xf32>
    %v812 = stablehlo.add %v810, %v811 : tensor<32x32x3x3xf32>
    %v813 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v814 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v815 = stablehlo.multiply %v813, %W8v : tensor<32x32x3x3xf32>
    %v816 = stablehlo.multiply %v807, %v807 : tensor<32x32x3x3xf32>
    %v817 = stablehlo.multiply %v814, %v816 : tensor<32x32x3x3xf32>
    %v818 = stablehlo.add %v815, %v817 : tensor<32x32x3x3xf32>
    %v819 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v820 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v821 = stablehlo.divide %v812, %v819 : tensor<32x32x3x3xf32>
    %v822 = stablehlo.divide %v818, %v820 : tensor<32x32x3x3xf32>
    %v823 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v824 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v825 = stablehlo.sqrt %v822 : tensor<32x32x3x3xf32>
    %v826 = stablehlo.add %v825, %v824 : tensor<32x32x3x3xf32>
    %v827 = stablehlo.divide %v821, %v826 : tensor<32x32x3x3xf32>
    %v828 = stablehlo.multiply %v823, %v827 : tensor<32x32x3x3xf32>
    %v829 = stablehlo.subtract %W8, %v828 : tensor<32x32x3x3xf32>
    %v830 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v831 = stablehlo.multiply %v830, %v823 : tensor<32x32x3x3xf32>
    %v832 = stablehlo.multiply %v831, %W8 : tensor<32x32x3x3xf32>
    %v833 = stablehlo.subtract %v829, %v832 : tensor<32x32x3x3xf32>
    %v834 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v835 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v836 = stablehlo.multiply %v834, %W8m : tensor<32x32x3x3xf32>
    %v837 = stablehlo.multiply %v835, %v807 : tensor<32x32x3x3xf32>
    %v838 = stablehlo.add %v836, %v837 : tensor<32x32x3x3xf32>
    %v839 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v840 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v841 = stablehlo.multiply %v839, %W8v : tensor<32x32x3x3xf32>
    %v842 = stablehlo.multiply %v807, %v807 : tensor<32x32x3x3xf32>
    %v843 = stablehlo.multiply %v840, %v842 : tensor<32x32x3x3xf32>
    %v844 = stablehlo.add %v841, %v843 : tensor<32x32x3x3xf32>
    %v845 = stablehlo.reshape %v128 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v846 = stablehlo.constant dense<0.0> : tensor<f32>
    %v847 = stablehlo.reduce(%v845 init: %v846) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v848 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v849 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v850 = stablehlo.multiply %v848, %cb8m : tensor<32xf32>
    %v851 = stablehlo.multiply %v849, %v847 : tensor<32xf32>
    %v852 = stablehlo.add %v850, %v851 : tensor<32xf32>
    %v853 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v854 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v855 = stablehlo.multiply %v853, %cb8v : tensor<32xf32>
    %v856 = stablehlo.multiply %v847, %v847 : tensor<32xf32>
    %v857 = stablehlo.multiply %v854, %v856 : tensor<32xf32>
    %v858 = stablehlo.add %v855, %v857 : tensor<32xf32>
    %v859 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v860 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v861 = stablehlo.divide %v852, %v859 : tensor<32xf32>
    %v862 = stablehlo.divide %v858, %v860 : tensor<32xf32>
    %v863 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v864 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v865 = stablehlo.sqrt %v862 : tensor<32xf32>
    %v866 = stablehlo.add %v865, %v864 : tensor<32xf32>
    %v867 = stablehlo.divide %v861, %v866 : tensor<32xf32>
    %v868 = stablehlo.multiply %v863, %v867 : tensor<32xf32>
    %v869 = stablehlo.subtract %cb8, %v868 : tensor<32xf32>
    %v870 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v871 = stablehlo.multiply %v870, %v863 : tensor<32xf32>
    %v872 = stablehlo.multiply %v871, %cb8 : tensor<32xf32>
    %v873 = stablehlo.subtract %v869, %v872 : tensor<32xf32>
    %v874 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v875 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v876 = stablehlo.multiply %v874, %cb8m : tensor<32xf32>
    %v877 = stablehlo.multiply %v875, %v847 : tensor<32xf32>
    %v878 = stablehlo.add %v876, %v877 : tensor<32xf32>
    %v879 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v880 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v881 = stablehlo.multiply %v879, %cb8v : tensor<32xf32>
    %v882 = stablehlo.multiply %v847, %v847 : tensor<32xf32>
    %v883 = stablehlo.multiply %v880, %v882 : tensor<32xf32>
    %v884 = stablehlo.add %v881, %v883 : tensor<32xf32>
    %v885 = stablehlo.dot_general %v87, %v116, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %v886 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v887 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v888 = stablehlo.multiply %v886, %W9m : tensor<128x64xf32>
    %v889 = stablehlo.multiply %v887, %v885 : tensor<128x64xf32>
    %v890 = stablehlo.add %v888, %v889 : tensor<128x64xf32>
    %v891 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v892 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v893 = stablehlo.multiply %v891, %W9v : tensor<128x64xf32>
    %v894 = stablehlo.multiply %v885, %v885 : tensor<128x64xf32>
    %v895 = stablehlo.multiply %v892, %v894 : tensor<128x64xf32>
    %v896 = stablehlo.add %v893, %v895 : tensor<128x64xf32>
    %v897 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v898 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v899 = stablehlo.divide %v890, %v897 : tensor<128x64xf32>
    %v900 = stablehlo.divide %v896, %v898 : tensor<128x64xf32>
    %v901 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v902 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v903 = stablehlo.sqrt %v900 : tensor<128x64xf32>
    %v904 = stablehlo.add %v903, %v902 : tensor<128x64xf32>
    %v905 = stablehlo.divide %v899, %v904 : tensor<128x64xf32>
    %v906 = stablehlo.multiply %v901, %v905 : tensor<128x64xf32>
    %v907 = stablehlo.subtract %W9, %v906 : tensor<128x64xf32>
    %v908 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v909 = stablehlo.multiply %v908, %v901 : tensor<128x64xf32>
    %v910 = stablehlo.multiply %v909, %W9 : tensor<128x64xf32>
    %v911 = stablehlo.subtract %v907, %v910 : tensor<128x64xf32>
    %v912 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v913 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v914 = stablehlo.multiply %v912, %W9m : tensor<128x64xf32>
    %v915 = stablehlo.multiply %v913, %v885 : tensor<128x64xf32>
    %v916 = stablehlo.add %v914, %v915 : tensor<128x64xf32>
    %v917 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v918 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v919 = stablehlo.multiply %v917, %W9v : tensor<128x64xf32>
    %v920 = stablehlo.multiply %v885, %v885 : tensor<128x64xf32>
    %v921 = stablehlo.multiply %v918, %v920 : tensor<128x64xf32>
    %v922 = stablehlo.add %v919, %v921 : tensor<128x64xf32>
    %v923 = stablehlo.constant dense<0.0> : tensor<f32>
    %v924 = stablehlo.reduce(%v116 init: %v923) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v925 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v926 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v927 = stablehlo.multiply %v925, %b9m : tensor<64xf32>
    %v928 = stablehlo.multiply %v926, %v924 : tensor<64xf32>
    %v929 = stablehlo.add %v927, %v928 : tensor<64xf32>
    %v930 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v931 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v932 = stablehlo.multiply %v930, %b9v : tensor<64xf32>
    %v933 = stablehlo.multiply %v924, %v924 : tensor<64xf32>
    %v934 = stablehlo.multiply %v931, %v933 : tensor<64xf32>
    %v935 = stablehlo.add %v932, %v934 : tensor<64xf32>
    %v936 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v937 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v938 = stablehlo.divide %v929, %v936 : tensor<64xf32>
    %v939 = stablehlo.divide %v935, %v937 : tensor<64xf32>
    %v940 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v941 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v942 = stablehlo.sqrt %v939 : tensor<64xf32>
    %v943 = stablehlo.add %v942, %v941 : tensor<64xf32>
    %v944 = stablehlo.divide %v938, %v943 : tensor<64xf32>
    %v945 = stablehlo.multiply %v940, %v944 : tensor<64xf32>
    %v946 = stablehlo.subtract %b9, %v945 : tensor<64xf32>
    %v947 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v948 = stablehlo.multiply %v947, %v940 : tensor<64xf32>
    %v949 = stablehlo.multiply %v948, %b9 : tensor<64xf32>
    %v950 = stablehlo.subtract %v946, %v949 : tensor<64xf32>
    %v951 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v952 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v953 = stablehlo.multiply %v951, %b9m : tensor<64xf32>
    %v954 = stablehlo.multiply %v952, %v924 : tensor<64xf32>
    %v955 = stablehlo.add %v953, %v954 : tensor<64xf32>
    %v956 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v957 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v958 = stablehlo.multiply %v956, %b9v : tensor<64xf32>
    %v959 = stablehlo.multiply %v924, %v924 : tensor<64xf32>
    %v960 = stablehlo.multiply %v957, %v959 : tensor<64xf32>
    %v961 = stablehlo.add %v958, %v960 : tensor<64xf32>
    %v962 = stablehlo.dot_general %v92, %v112, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<64x64xf32>
    %v963 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v964 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v965 = stablehlo.multiply %v963, %Wam : tensor<64x64xf32>
    %v966 = stablehlo.multiply %v964, %v962 : tensor<64x64xf32>
    %v967 = stablehlo.add %v965, %v966 : tensor<64x64xf32>
    %v968 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v969 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v970 = stablehlo.multiply %v968, %Wav : tensor<64x64xf32>
    %v971 = stablehlo.multiply %v962, %v962 : tensor<64x64xf32>
    %v972 = stablehlo.multiply %v969, %v971 : tensor<64x64xf32>
    %v973 = stablehlo.add %v970, %v972 : tensor<64x64xf32>
    %v974 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v975 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v976 = stablehlo.divide %v967, %v974 : tensor<64x64xf32>
    %v977 = stablehlo.divide %v973, %v975 : tensor<64x64xf32>
    %v978 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v979 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v980 = stablehlo.sqrt %v977 : tensor<64x64xf32>
    %v981 = stablehlo.add %v980, %v979 : tensor<64x64xf32>
    %v982 = stablehlo.divide %v976, %v981 : tensor<64x64xf32>
    %v983 = stablehlo.multiply %v978, %v982 : tensor<64x64xf32>
    %v984 = stablehlo.subtract %Wa, %v983 : tensor<64x64xf32>
    %v985 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v986 = stablehlo.multiply %v985, %v978 : tensor<64x64xf32>
    %v987 = stablehlo.multiply %v986, %Wa : tensor<64x64xf32>
    %v988 = stablehlo.subtract %v984, %v987 : tensor<64x64xf32>
    %v989 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v990 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v991 = stablehlo.multiply %v989, %Wam : tensor<64x64xf32>
    %v992 = stablehlo.multiply %v990, %v962 : tensor<64x64xf32>
    %v993 = stablehlo.add %v991, %v992 : tensor<64x64xf32>
    %v994 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v995 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v996 = stablehlo.multiply %v994, %Wav : tensor<64x64xf32>
    %v997 = stablehlo.multiply %v962, %v962 : tensor<64x64xf32>
    %v998 = stablehlo.multiply %v995, %v997 : tensor<64x64xf32>
    %v999 = stablehlo.add %v996, %v998 : tensor<64x64xf32>
    %v1000 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1001 = stablehlo.reduce(%v112 init: %v1000) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v1002 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1003 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1004 = stablehlo.multiply %v1002, %bam : tensor<64xf32>
    %v1005 = stablehlo.multiply %v1003, %v1001 : tensor<64xf32>
    %v1006 = stablehlo.add %v1004, %v1005 : tensor<64xf32>
    %v1007 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1008 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1009 = stablehlo.multiply %v1007, %bav : tensor<64xf32>
    %v1010 = stablehlo.multiply %v1001, %v1001 : tensor<64xf32>
    %v1011 = stablehlo.multiply %v1008, %v1010 : tensor<64xf32>
    %v1012 = stablehlo.add %v1009, %v1011 : tensor<64xf32>
    %v1013 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1014 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1015 = stablehlo.divide %v1006, %v1013 : tensor<64xf32>
    %v1016 = stablehlo.divide %v1012, %v1014 : tensor<64xf32>
    %v1017 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1018 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1019 = stablehlo.sqrt %v1016 : tensor<64xf32>
    %v1020 = stablehlo.add %v1019, %v1018 : tensor<64xf32>
    %v1021 = stablehlo.divide %v1015, %v1020 : tensor<64xf32>
    %v1022 = stablehlo.multiply %v1017, %v1021 : tensor<64xf32>
    %v1023 = stablehlo.subtract %ba, %v1022 : tensor<64xf32>
    %v1024 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1025 = stablehlo.multiply %v1024, %v1017 : tensor<64xf32>
    %v1026 = stablehlo.multiply %v1025, %ba : tensor<64xf32>
    %v1027 = stablehlo.subtract %v1023, %v1026 : tensor<64xf32>
    %v1028 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1029 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1030 = stablehlo.multiply %v1028, %bam : tensor<64xf32>
    %v1031 = stablehlo.multiply %v1029, %v1001 : tensor<64xf32>
    %v1032 = stablehlo.add %v1030, %v1031 : tensor<64xf32>
    %v1033 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1034 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1035 = stablehlo.multiply %v1033, %bav : tensor<64xf32>
    %v1036 = stablehlo.multiply %v1001, %v1001 : tensor<64xf32>
    %v1037 = stablehlo.multiply %v1034, %v1036 : tensor<64xf32>
    %v1038 = stablehlo.add %v1035, %v1037 : tensor<64xf32>
    %v1039 = stablehlo.dot_general %v97, %v108, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x10xf32>) -> tensor<64x10xf32>
    %v1040 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1041 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1042 = stablehlo.multiply %v1040, %Wbm : tensor<64x10xf32>
    %v1043 = stablehlo.multiply %v1041, %v1039 : tensor<64x10xf32>
    %v1044 = stablehlo.add %v1042, %v1043 : tensor<64x10xf32>
    %v1045 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1046 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1047 = stablehlo.multiply %v1045, %Wbv : tensor<64x10xf32>
    %v1048 = stablehlo.multiply %v1039, %v1039 : tensor<64x10xf32>
    %v1049 = stablehlo.multiply %v1046, %v1048 : tensor<64x10xf32>
    %v1050 = stablehlo.add %v1047, %v1049 : tensor<64x10xf32>
    %v1051 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1052 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1053 = stablehlo.divide %v1044, %v1051 : tensor<64x10xf32>
    %v1054 = stablehlo.divide %v1050, %v1052 : tensor<64x10xf32>
    %v1055 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1056 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1057 = stablehlo.sqrt %v1054 : tensor<64x10xf32>
    %v1058 = stablehlo.add %v1057, %v1056 : tensor<64x10xf32>
    %v1059 = stablehlo.divide %v1053, %v1058 : tensor<64x10xf32>
    %v1060 = stablehlo.multiply %v1055, %v1059 : tensor<64x10xf32>
    %v1061 = stablehlo.subtract %Wb, %v1060 : tensor<64x10xf32>
    %v1062 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1063 = stablehlo.multiply %v1062, %v1055 : tensor<64x10xf32>
    %v1064 = stablehlo.multiply %v1063, %Wb : tensor<64x10xf32>
    %v1065 = stablehlo.subtract %v1061, %v1064 : tensor<64x10xf32>
    %v1066 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1067 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1068 = stablehlo.multiply %v1066, %Wbm : tensor<64x10xf32>
    %v1069 = stablehlo.multiply %v1067, %v1039 : tensor<64x10xf32>
    %v1070 = stablehlo.add %v1068, %v1069 : tensor<64x10xf32>
    %v1071 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1072 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1073 = stablehlo.multiply %v1071, %Wbv : tensor<64x10xf32>
    %v1074 = stablehlo.multiply %v1039, %v1039 : tensor<64x10xf32>
    %v1075 = stablehlo.multiply %v1072, %v1074 : tensor<64x10xf32>
    %v1076 = stablehlo.add %v1073, %v1075 : tensor<64x10xf32>
    %v1077 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1078 = stablehlo.reduce(%v108 init: %v1077) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1079 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1080 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1081 = stablehlo.multiply %v1079, %bbm : tensor<10xf32>
    %v1082 = stablehlo.multiply %v1080, %v1078 : tensor<10xf32>
    %v1083 = stablehlo.add %v1081, %v1082 : tensor<10xf32>
    %v1084 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1085 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1086 = stablehlo.multiply %v1084, %bbv : tensor<10xf32>
    %v1087 = stablehlo.multiply %v1078, %v1078 : tensor<10xf32>
    %v1088 = stablehlo.multiply %v1085, %v1087 : tensor<10xf32>
    %v1089 = stablehlo.add %v1086, %v1088 : tensor<10xf32>
    %v1090 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1091 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1092 = stablehlo.divide %v1083, %v1090 : tensor<10xf32>
    %v1093 = stablehlo.divide %v1089, %v1091 : tensor<10xf32>
    %v1094 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1095 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1096 = stablehlo.sqrt %v1093 : tensor<10xf32>
    %v1097 = stablehlo.add %v1096, %v1095 : tensor<10xf32>
    %v1098 = stablehlo.divide %v1092, %v1097 : tensor<10xf32>
    %v1099 = stablehlo.multiply %v1094, %v1098 : tensor<10xf32>
    %v1100 = stablehlo.subtract %bb, %v1099 : tensor<10xf32>
    %v1101 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1102 = stablehlo.multiply %v1101, %v1094 : tensor<10xf32>
    %v1103 = stablehlo.multiply %v1102, %bb : tensor<10xf32>
    %v1104 = stablehlo.subtract %v1100, %v1103 : tensor<10xf32>
    %v1105 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1106 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1107 = stablehlo.multiply %v1105, %bbm : tensor<10xf32>
    %v1108 = stablehlo.multiply %v1106, %v1078 : tensor<10xf32>
    %v1109 = stablehlo.add %v1107, %v1108 : tensor<10xf32>
    %v1110 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1111 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1112 = stablehlo.multiply %v1110, %bbv : tensor<10xf32>
    %v1113 = stablehlo.multiply %v1078, %v1078 : tensor<10xf32>
    %v1114 = stablehlo.multiply %v1111, %v1113 : tensor<10xf32>
    %v1115 = stablehlo.add %v1112, %v1114 : tensor<10xf32>
    return %v252, %v292, %v335, %v375, %v418, %v458, %v501, %v541, %v584, %v624, %v667, %v707, %v750, %v790, %v833, %v873, %v911, %v950, %v988, %v1027, %v1065, %v1104, %v257, %v297, %v340, %v380, %v423, %v463, %v506, %v546, %v589, %v629, %v672, %v712, %v755, %v795, %v838, %v878, %v916, %v955, %v993, %v1032, %v1070, %v1109, %v263, %v303, %v346, %v386, %v429, %v469, %v512, %v552, %v595, %v635, %v678, %v718, %v761, %v801, %v844, %v884, %v922, %v961, %v999, %v1038, %v1076, %v1115, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
