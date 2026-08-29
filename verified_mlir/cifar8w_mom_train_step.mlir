module @m {
  func.func @cifar8w_mom_train_step(%x: tensor<128x3072xf32>, %W1: tensor<16x3x3x3xf32>, %cb1: tensor<16xf32>, %W2: tensor<16x16x3x3xf32>, %cb2: tensor<16xf32>, %W3: tensor<16x16x3x3xf32>, %cb3: tensor<16xf32>, %W4: tensor<16x16x3x3xf32>, %cb4: tensor<16xf32>, %W5: tensor<32x16x3x3xf32>, %cb5: tensor<32xf32>, %W6: tensor<32x32x3x3xf32>, %cb6: tensor<32xf32>, %W7: tensor<32x32x3x3xf32>, %cb7: tensor<32xf32>, %W8: tensor<32x32x3x3xf32>, %cb8: tensor<32xf32>, %W9: tensor<128x512xf32>, %b9: tensor<512xf32>, %Wa: tensor<512x512xf32>, %ba: tensor<512xf32>, %Wb: tensor<512x10xf32>, %bb: tensor<10xf32>, %W1m: tensor<16x3x3x3xf32>, %cb1m: tensor<16xf32>, %W2m: tensor<16x16x3x3xf32>, %cb2m: tensor<16xf32>, %W3m: tensor<16x16x3x3xf32>, %cb3m: tensor<16xf32>, %W4m: tensor<16x16x3x3xf32>, %cb4m: tensor<16xf32>, %W5m: tensor<32x16x3x3xf32>, %cb5m: tensor<32xf32>, %W6m: tensor<32x32x3x3xf32>, %cb6m: tensor<32xf32>, %W7m: tensor<32x32x3x3xf32>, %cb7m: tensor<32xf32>, %W8m: tensor<32x32x3x3xf32>, %cb8m: tensor<32xf32>, %W9m: tensor<128x512xf32>, %b9m: tensor<512xf32>, %Wam: tensor<512x512xf32>, %bam: tensor<512xf32>, %Wbm: tensor<512x10xf32>, %bbm: tensor<10xf32>, %W1v: tensor<16x3x3x3xf32>, %cb1v: tensor<16xf32>, %W2v: tensor<16x16x3x3xf32>, %cb2v: tensor<16xf32>, %W3v: tensor<16x16x3x3xf32>, %cb3v: tensor<16xf32>, %W4v: tensor<16x16x3x3xf32>, %cb4v: tensor<16xf32>, %W5v: tensor<32x16x3x3xf32>, %cb5v: tensor<32xf32>, %W6v: tensor<32x32x3x3xf32>, %cb6v: tensor<32xf32>, %W7v: tensor<32x32x3x3xf32>, %cb7v: tensor<32xf32>, %W8v: tensor<32x32x3x3xf32>, %cb8v: tensor<32xf32>, %W9v: tensor<128x512xf32>, %b9v: tensor<512xf32>, %Wav: tensor<512x512xf32>, %bav: tensor<512xf32>, %Wbv: tensor<512x10xf32>, %bbv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: tensor<128x10xf32>) -> (tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>) {
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
    %v109 = stablehlo.dot_general %v108, %Wb, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x10xf32>, tensor<512x10xf32>) -> tensor<128x512xf32>
    %v110 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v111 = stablehlo.compare GT, %v95, %v110 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v112 = stablehlo.select %v111, %v109, %v110 : tensor<128x512xi1>, tensor<128x512xf32>
    %v113 = stablehlo.dot_general %v112, %Wa, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %v114 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v115 = stablehlo.compare GT, %v90, %v114 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v116 = stablehlo.select %v115, %v113, %v114 : tensor<128x512xi1>, tensor<128x512xf32>
    %v117 = stablehlo.dot_general %v116, %W9, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x128xf32>
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
    %v227 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v228 = stablehlo.multiply %v227, %W1v : tensor<16x3x3x3xf32>
    %v229 = stablehlo.add %v228, %v226 : tensor<16x3x3x3xf32>
    %v230 = stablehlo.multiply %v227, %v229 : tensor<16x3x3x3xf32>
    %v231 = stablehlo.add %v230, %v226 : tensor<16x3x3x3xf32>
    %v232 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v233 = stablehlo.multiply %v232, %v231 : tensor<16x3x3x3xf32>
    %v234 = stablehlo.subtract %W1, %v233 : tensor<16x3x3x3xf32>
    %v235 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v236 = stablehlo.multiply %v235, %W1v : tensor<16x3x3x3xf32>
    %v237 = stablehlo.add %v236, %v226 : tensor<16x3x3x3xf32>
    %v238 = stablehlo.reshape %v220 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v239 = stablehlo.constant dense<0.0> : tensor<f32>
    %v240 = stablehlo.reduce(%v238 init: %v239) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v241 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v242 = stablehlo.multiply %v241, %cb1v : tensor<16xf32>
    %v243 = stablehlo.add %v242, %v240 : tensor<16xf32>
    %v244 = stablehlo.multiply %v241, %v243 : tensor<16xf32>
    %v245 = stablehlo.add %v244, %v240 : tensor<16xf32>
    %v246 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v247 = stablehlo.multiply %v246, %v245 : tensor<16xf32>
    %v248 = stablehlo.subtract %cb1, %v247 : tensor<16xf32>
    %v249 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v250 = stablehlo.multiply %v249, %cb1v : tensor<16xf32>
    %v251 = stablehlo.add %v250, %v240 : tensor<16xf32>
    %v252 = stablehlo.reshape %v8 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v253 = stablehlo.reshape %v209 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v254 = stablehlo.transpose %v252, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v255 = stablehlo.transpose %v253, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v256 = stablehlo.convolution(%v254, %v255)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x16x3x3xf32>
    %v257 = stablehlo.transpose %v256, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v258 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v259 = stablehlo.multiply %v258, %W2v : tensor<16x16x3x3xf32>
    %v260 = stablehlo.add %v259, %v257 : tensor<16x16x3x3xf32>
    %v261 = stablehlo.multiply %v258, %v260 : tensor<16x16x3x3xf32>
    %v262 = stablehlo.add %v261, %v257 : tensor<16x16x3x3xf32>
    %v263 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v264 = stablehlo.multiply %v263, %v262 : tensor<16x16x3x3xf32>
    %v265 = stablehlo.subtract %W2, %v264 : tensor<16x16x3x3xf32>
    %v266 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v267 = stablehlo.multiply %v266, %W2v : tensor<16x16x3x3xf32>
    %v268 = stablehlo.add %v267, %v257 : tensor<16x16x3x3xf32>
    %v269 = stablehlo.reshape %v209 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v270 = stablehlo.constant dense<0.0> : tensor<f32>
    %v271 = stablehlo.reduce(%v269 init: %v270) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v272 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v273 = stablehlo.multiply %v272, %cb2v : tensor<16xf32>
    %v274 = stablehlo.add %v273, %v271 : tensor<16xf32>
    %v275 = stablehlo.multiply %v272, %v274 : tensor<16xf32>
    %v276 = stablehlo.add %v275, %v271 : tensor<16xf32>
    %v277 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v278 = stablehlo.multiply %v277, %v276 : tensor<16xf32>
    %v279 = stablehlo.subtract %cb2, %v278 : tensor<16xf32>
    %v280 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v281 = stablehlo.multiply %v280, %cb2v : tensor<16xf32>
    %v282 = stablehlo.add %v281, %v271 : tensor<16xf32>
    %v283 = stablehlo.reshape %v21 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v284 = stablehlo.reshape %v193 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v285 = stablehlo.transpose %v283, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v286 = stablehlo.transpose %v284, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v287 = stablehlo.convolution(%v285, %v286)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v288 = stablehlo.transpose %v287, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v289 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v290 = stablehlo.multiply %v289, %W3v : tensor<16x16x3x3xf32>
    %v291 = stablehlo.add %v290, %v288 : tensor<16x16x3x3xf32>
    %v292 = stablehlo.multiply %v289, %v291 : tensor<16x16x3x3xf32>
    %v293 = stablehlo.add %v292, %v288 : tensor<16x16x3x3xf32>
    %v294 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v295 = stablehlo.multiply %v294, %v293 : tensor<16x16x3x3xf32>
    %v296 = stablehlo.subtract %W3, %v295 : tensor<16x16x3x3xf32>
    %v297 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v298 = stablehlo.multiply %v297, %W3v : tensor<16x16x3x3xf32>
    %v299 = stablehlo.add %v298, %v288 : tensor<16x16x3x3xf32>
    %v300 = stablehlo.reshape %v193 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v301 = stablehlo.constant dense<0.0> : tensor<f32>
    %v302 = stablehlo.reduce(%v300 init: %v301) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v303 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v304 = stablehlo.multiply %v303, %cb3v : tensor<16xf32>
    %v305 = stablehlo.add %v304, %v302 : tensor<16xf32>
    %v306 = stablehlo.multiply %v303, %v305 : tensor<16xf32>
    %v307 = stablehlo.add %v306, %v302 : tensor<16xf32>
    %v308 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v309 = stablehlo.multiply %v308, %v307 : tensor<16xf32>
    %v310 = stablehlo.subtract %cb3, %v309 : tensor<16xf32>
    %v311 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v312 = stablehlo.multiply %v311, %cb3v : tensor<16xf32>
    %v313 = stablehlo.add %v312, %v302 : tensor<16xf32>
    %v314 = stablehlo.reshape %v30 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v315 = stablehlo.reshape %v182 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v316 = stablehlo.transpose %v314, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v317 = stablehlo.transpose %v315, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v318 = stablehlo.convolution(%v316, %v317)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v319 = stablehlo.transpose %v318, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v320 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v321 = stablehlo.multiply %v320, %W4v : tensor<16x16x3x3xf32>
    %v322 = stablehlo.add %v321, %v319 : tensor<16x16x3x3xf32>
    %v323 = stablehlo.multiply %v320, %v322 : tensor<16x16x3x3xf32>
    %v324 = stablehlo.add %v323, %v319 : tensor<16x16x3x3xf32>
    %v325 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v326 = stablehlo.multiply %v325, %v324 : tensor<16x16x3x3xf32>
    %v327 = stablehlo.subtract %W4, %v326 : tensor<16x16x3x3xf32>
    %v328 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v329 = stablehlo.multiply %v328, %W4v : tensor<16x16x3x3xf32>
    %v330 = stablehlo.add %v329, %v319 : tensor<16x16x3x3xf32>
    %v331 = stablehlo.reshape %v182 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v332 = stablehlo.constant dense<0.0> : tensor<f32>
    %v333 = stablehlo.reduce(%v331 init: %v332) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v334 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v335 = stablehlo.multiply %v334, %cb4v : tensor<16xf32>
    %v336 = stablehlo.add %v335, %v333 : tensor<16xf32>
    %v337 = stablehlo.multiply %v334, %v336 : tensor<16xf32>
    %v338 = stablehlo.add %v337, %v333 : tensor<16xf32>
    %v339 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v340 = stablehlo.multiply %v339, %v338 : tensor<16xf32>
    %v341 = stablehlo.subtract %cb4, %v340 : tensor<16xf32>
    %v342 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v343 = stablehlo.multiply %v342, %cb4v : tensor<16xf32>
    %v344 = stablehlo.add %v343, %v333 : tensor<16xf32>
    %v345 = stablehlo.reshape %v43 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v346 = stablehlo.reshape %v166 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v347 = stablehlo.transpose %v345, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v348 = stablehlo.transpose %v346, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v349 = stablehlo.convolution(%v347, %v348)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<16x32x3x3xf32>
    %v350 = stablehlo.transpose %v349, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v351 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v352 = stablehlo.multiply %v351, %W5v : tensor<32x16x3x3xf32>
    %v353 = stablehlo.add %v352, %v350 : tensor<32x16x3x3xf32>
    %v354 = stablehlo.multiply %v351, %v353 : tensor<32x16x3x3xf32>
    %v355 = stablehlo.add %v354, %v350 : tensor<32x16x3x3xf32>
    %v356 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v357 = stablehlo.multiply %v356, %v355 : tensor<32x16x3x3xf32>
    %v358 = stablehlo.subtract %W5, %v357 : tensor<32x16x3x3xf32>
    %v359 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v360 = stablehlo.multiply %v359, %W5v : tensor<32x16x3x3xf32>
    %v361 = stablehlo.add %v360, %v350 : tensor<32x16x3x3xf32>
    %v362 = stablehlo.reshape %v166 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v363 = stablehlo.constant dense<0.0> : tensor<f32>
    %v364 = stablehlo.reduce(%v362 init: %v363) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v365 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v366 = stablehlo.multiply %v365, %cb5v : tensor<32xf32>
    %v367 = stablehlo.add %v366, %v364 : tensor<32xf32>
    %v368 = stablehlo.multiply %v365, %v367 : tensor<32xf32>
    %v369 = stablehlo.add %v368, %v364 : tensor<32xf32>
    %v370 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v371 = stablehlo.multiply %v370, %v369 : tensor<32xf32>
    %v372 = stablehlo.subtract %cb5, %v371 : tensor<32xf32>
    %v373 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v374 = stablehlo.multiply %v373, %cb5v : tensor<32xf32>
    %v375 = stablehlo.add %v374, %v364 : tensor<32xf32>
    %v376 = stablehlo.reshape %v52 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v377 = stablehlo.reshape %v155 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v378 = stablehlo.transpose %v376, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v379 = stablehlo.transpose %v377, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v380 = stablehlo.convolution(%v378, %v379)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<32x32x3x3xf32>
    %v381 = stablehlo.transpose %v380, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v382 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v383 = stablehlo.multiply %v382, %W6v : tensor<32x32x3x3xf32>
    %v384 = stablehlo.add %v383, %v381 : tensor<32x32x3x3xf32>
    %v385 = stablehlo.multiply %v382, %v384 : tensor<32x32x3x3xf32>
    %v386 = stablehlo.add %v385, %v381 : tensor<32x32x3x3xf32>
    %v387 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v388 = stablehlo.multiply %v387, %v386 : tensor<32x32x3x3xf32>
    %v389 = stablehlo.subtract %W6, %v388 : tensor<32x32x3x3xf32>
    %v390 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v391 = stablehlo.multiply %v390, %W6v : tensor<32x32x3x3xf32>
    %v392 = stablehlo.add %v391, %v381 : tensor<32x32x3x3xf32>
    %v393 = stablehlo.reshape %v155 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v394 = stablehlo.constant dense<0.0> : tensor<f32>
    %v395 = stablehlo.reduce(%v393 init: %v394) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v396 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v397 = stablehlo.multiply %v396, %cb6v : tensor<32xf32>
    %v398 = stablehlo.add %v397, %v395 : tensor<32xf32>
    %v399 = stablehlo.multiply %v396, %v398 : tensor<32xf32>
    %v400 = stablehlo.add %v399, %v395 : tensor<32xf32>
    %v401 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v402 = stablehlo.multiply %v401, %v400 : tensor<32xf32>
    %v403 = stablehlo.subtract %cb6, %v402 : tensor<32xf32>
    %v404 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v405 = stablehlo.multiply %v404, %cb6v : tensor<32xf32>
    %v406 = stablehlo.add %v405, %v395 : tensor<32xf32>
    %v407 = stablehlo.reshape %v65 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v408 = stablehlo.reshape %v139 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v409 = stablehlo.transpose %v407, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v410 = stablehlo.transpose %v408, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v411 = stablehlo.convolution(%v409, %v410)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v412 = stablehlo.transpose %v411, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v413 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v414 = stablehlo.multiply %v413, %W7v : tensor<32x32x3x3xf32>
    %v415 = stablehlo.add %v414, %v412 : tensor<32x32x3x3xf32>
    %v416 = stablehlo.multiply %v413, %v415 : tensor<32x32x3x3xf32>
    %v417 = stablehlo.add %v416, %v412 : tensor<32x32x3x3xf32>
    %v418 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v419 = stablehlo.multiply %v418, %v417 : tensor<32x32x3x3xf32>
    %v420 = stablehlo.subtract %W7, %v419 : tensor<32x32x3x3xf32>
    %v421 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v422 = stablehlo.multiply %v421, %W7v : tensor<32x32x3x3xf32>
    %v423 = stablehlo.add %v422, %v412 : tensor<32x32x3x3xf32>
    %v424 = stablehlo.reshape %v139 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v425 = stablehlo.constant dense<0.0> : tensor<f32>
    %v426 = stablehlo.reduce(%v424 init: %v425) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v427 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v428 = stablehlo.multiply %v427, %cb7v : tensor<32xf32>
    %v429 = stablehlo.add %v428, %v426 : tensor<32xf32>
    %v430 = stablehlo.multiply %v427, %v429 : tensor<32xf32>
    %v431 = stablehlo.add %v430, %v426 : tensor<32xf32>
    %v432 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v433 = stablehlo.multiply %v432, %v431 : tensor<32xf32>
    %v434 = stablehlo.subtract %cb7, %v433 : tensor<32xf32>
    %v435 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v436 = stablehlo.multiply %v435, %cb7v : tensor<32xf32>
    %v437 = stablehlo.add %v436, %v426 : tensor<32xf32>
    %v438 = stablehlo.reshape %v74 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v439 = stablehlo.reshape %v128 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v440 = stablehlo.transpose %v438, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v441 = stablehlo.transpose %v439, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v442 = stablehlo.convolution(%v440, %v441)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v443 = stablehlo.transpose %v442, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v444 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v445 = stablehlo.multiply %v444, %W8v : tensor<32x32x3x3xf32>
    %v446 = stablehlo.add %v445, %v443 : tensor<32x32x3x3xf32>
    %v447 = stablehlo.multiply %v444, %v446 : tensor<32x32x3x3xf32>
    %v448 = stablehlo.add %v447, %v443 : tensor<32x32x3x3xf32>
    %v449 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v450 = stablehlo.multiply %v449, %v448 : tensor<32x32x3x3xf32>
    %v451 = stablehlo.subtract %W8, %v450 : tensor<32x32x3x3xf32>
    %v452 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v453 = stablehlo.multiply %v452, %W8v : tensor<32x32x3x3xf32>
    %v454 = stablehlo.add %v453, %v443 : tensor<32x32x3x3xf32>
    %v455 = stablehlo.reshape %v128 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v456 = stablehlo.constant dense<0.0> : tensor<f32>
    %v457 = stablehlo.reduce(%v455 init: %v456) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v458 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v459 = stablehlo.multiply %v458, %cb8v : tensor<32xf32>
    %v460 = stablehlo.add %v459, %v457 : tensor<32xf32>
    %v461 = stablehlo.multiply %v458, %v460 : tensor<32xf32>
    %v462 = stablehlo.add %v461, %v457 : tensor<32xf32>
    %v463 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v464 = stablehlo.multiply %v463, %v462 : tensor<32xf32>
    %v465 = stablehlo.subtract %cb8, %v464 : tensor<32xf32>
    %v466 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v467 = stablehlo.multiply %v466, %cb8v : tensor<32xf32>
    %v468 = stablehlo.add %v467, %v457 : tensor<32xf32>
    %v469 = stablehlo.dot_general %v87, %v116, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x512xf32>) -> tensor<128x512xf32>
    %v470 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v471 = stablehlo.multiply %v470, %W9v : tensor<128x512xf32>
    %v472 = stablehlo.add %v471, %v469 : tensor<128x512xf32>
    %v473 = stablehlo.multiply %v470, %v472 : tensor<128x512xf32>
    %v474 = stablehlo.add %v473, %v469 : tensor<128x512xf32>
    %v475 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v476 = stablehlo.multiply %v475, %v474 : tensor<128x512xf32>
    %v477 = stablehlo.subtract %W9, %v476 : tensor<128x512xf32>
    %v478 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v479 = stablehlo.multiply %v478, %W9v : tensor<128x512xf32>
    %v480 = stablehlo.add %v479, %v469 : tensor<128x512xf32>
    %v481 = stablehlo.constant dense<0.0> : tensor<f32>
    %v482 = stablehlo.reduce(%v116 init: %v481) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v483 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v484 = stablehlo.multiply %v483, %b9v : tensor<512xf32>
    %v485 = stablehlo.add %v484, %v482 : tensor<512xf32>
    %v486 = stablehlo.multiply %v483, %v485 : tensor<512xf32>
    %v487 = stablehlo.add %v486, %v482 : tensor<512xf32>
    %v488 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v489 = stablehlo.multiply %v488, %v487 : tensor<512xf32>
    %v490 = stablehlo.subtract %b9, %v489 : tensor<512xf32>
    %v491 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v492 = stablehlo.multiply %v491, %b9v : tensor<512xf32>
    %v493 = stablehlo.add %v492, %v482 : tensor<512xf32>
    %v494 = stablehlo.dot_general %v92, %v112, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<512x512xf32>
    %v495 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v496 = stablehlo.multiply %v495, %Wav : tensor<512x512xf32>
    %v497 = stablehlo.add %v496, %v494 : tensor<512x512xf32>
    %v498 = stablehlo.multiply %v495, %v497 : tensor<512x512xf32>
    %v499 = stablehlo.add %v498, %v494 : tensor<512x512xf32>
    %v500 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v501 = stablehlo.multiply %v500, %v499 : tensor<512x512xf32>
    %v502 = stablehlo.subtract %Wa, %v501 : tensor<512x512xf32>
    %v503 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v504 = stablehlo.multiply %v503, %Wav : tensor<512x512xf32>
    %v505 = stablehlo.add %v504, %v494 : tensor<512x512xf32>
    %v506 = stablehlo.constant dense<0.0> : tensor<f32>
    %v507 = stablehlo.reduce(%v112 init: %v506) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v508 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v509 = stablehlo.multiply %v508, %bav : tensor<512xf32>
    %v510 = stablehlo.add %v509, %v507 : tensor<512xf32>
    %v511 = stablehlo.multiply %v508, %v510 : tensor<512xf32>
    %v512 = stablehlo.add %v511, %v507 : tensor<512xf32>
    %v513 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v514 = stablehlo.multiply %v513, %v512 : tensor<512xf32>
    %v515 = stablehlo.subtract %ba, %v514 : tensor<512xf32>
    %v516 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v517 = stablehlo.multiply %v516, %bav : tensor<512xf32>
    %v518 = stablehlo.add %v517, %v507 : tensor<512xf32>
    %v519 = stablehlo.dot_general %v97, %v108, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x10xf32>) -> tensor<512x10xf32>
    %v520 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v521 = stablehlo.multiply %v520, %Wbv : tensor<512x10xf32>
    %v522 = stablehlo.add %v521, %v519 : tensor<512x10xf32>
    %v523 = stablehlo.multiply %v520, %v522 : tensor<512x10xf32>
    %v524 = stablehlo.add %v523, %v519 : tensor<512x10xf32>
    %v525 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v526 = stablehlo.multiply %v525, %v524 : tensor<512x10xf32>
    %v527 = stablehlo.subtract %Wb, %v526 : tensor<512x10xf32>
    %v528 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v529 = stablehlo.multiply %v528, %Wbv : tensor<512x10xf32>
    %v530 = stablehlo.add %v529, %v519 : tensor<512x10xf32>
    %v531 = stablehlo.constant dense<0.0> : tensor<f32>
    %v532 = stablehlo.reduce(%v108 init: %v531) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v533 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v534 = stablehlo.multiply %v533, %bbv : tensor<10xf32>
    %v535 = stablehlo.add %v534, %v532 : tensor<10xf32>
    %v536 = stablehlo.multiply %v533, %v535 : tensor<10xf32>
    %v537 = stablehlo.add %v536, %v532 : tensor<10xf32>
    %v538 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v539 = stablehlo.multiply %v538, %v537 : tensor<10xf32>
    %v540 = stablehlo.subtract %bb, %v539 : tensor<10xf32>
    %v541 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v542 = stablehlo.multiply %v541, %bbv : tensor<10xf32>
    %v543 = stablehlo.add %v542, %v532 : tensor<10xf32>
    return %v234, %v248, %v265, %v279, %v296, %v310, %v327, %v341, %v358, %v372, %v389, %v403, %v420, %v434, %v451, %v465, %v477, %v490, %v502, %v515, %v527, %v540, %W1m, %cb1m, %W2m, %cb2m, %W3m, %cb3m, %W4m, %cb4m, %W5m, %cb5m, %W6m, %cb6m, %W7m, %cb7m, %W8m, %cb8m, %W9m, %b9m, %Wam, %bam, %Wbm, %bbm, %v237, %v251, %v268, %v282, %v299, %v313, %v330, %v344, %v361, %v375, %v392, %v406, %v423, %v437, %v454, %v468, %v480, %v493, %v505, %v518, %v530, %v543, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
