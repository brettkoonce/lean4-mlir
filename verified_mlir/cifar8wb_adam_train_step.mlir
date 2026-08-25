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
    %v5 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v6 = stablehlo.maximum %v4, %v5 : tensor<128x16384xf32>
    %v7 = stablehlo.reshape %v6 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v8 = stablehlo.convolution(%v7, %W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x32x32xf32>
    %v9 = stablehlo.broadcast_in_dim %cb2, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
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
    %v20 = stablehlo.broadcast_in_dim %cb3, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v21 = stablehlo.add %v19, %v20 : tensor<128x16x16x16xf32>
    %v22 = stablehlo.reshape %v21 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v23 = stablehlo.constant dense<0.0> : tensor<128x4096xf32>
    %v24 = stablehlo.maximum %v22, %v23 : tensor<128x4096xf32>
    %v25 = stablehlo.reshape %v24 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v26 = stablehlo.convolution(%v25, %W4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v27 = stablehlo.broadcast_in_dim %cb4, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
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
    %v38 = stablehlo.broadcast_in_dim %cb5, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v39 = stablehlo.add %v37, %v38 : tensor<128x32x8x8xf32>
    %v40 = stablehlo.reshape %v39 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v41 = stablehlo.constant dense<0.0> : tensor<128x2048xf32>
    %v42 = stablehlo.maximum %v40, %v41 : tensor<128x2048xf32>
    %v43 = stablehlo.reshape %v42 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v44 = stablehlo.convolution(%v43, %W6)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x8x8xf32>
    %v45 = stablehlo.broadcast_in_dim %cb6, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
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
    %v56 = stablehlo.broadcast_in_dim %cb7, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v57 = stablehlo.add %v55, %v56 : tensor<128x32x4x4xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v59 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v60 = stablehlo.maximum %v58, %v59 : tensor<128x512xf32>
    %v61 = stablehlo.reshape %v60 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v62 = stablehlo.convolution(%v61, %W8)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v63 = stablehlo.broadcast_in_dim %cb8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
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
    %v72 = stablehlo.dot_general %v71, %W9, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x512xf32>) -> tensor<128x512xf32>
    %v73 = stablehlo.broadcast_in_dim %b9, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v74 = stablehlo.add %v72, %v73 : tensor<128x512xf32>
    %v75 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v76 = stablehlo.maximum %v74, %v75 : tensor<128x512xf32>
    %v77 = stablehlo.dot_general %v76, %Wa, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %v78 = stablehlo.broadcast_in_dim %ba, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v79 = stablehlo.add %v77, %v78 : tensor<128x512xf32>
    %v80 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v81 = stablehlo.maximum %v79, %v80 : tensor<128x512xf32>
    %v82 = stablehlo.dot_general %v81, %Wb, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x10xf32>) -> tensor<128x10xf32>
    %v83 = stablehlo.broadcast_in_dim %bb, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v84 = stablehlo.add %v82, %v83 : tensor<128x10xf32>
    %v85 = stablehlo.reshape %v84 : (tensor<128x10xf32>) -> tensor<128x1x10xf32>
    %v86 = stablehlo.constant dense<0.0> : tensor<f32>
    %v87 = stablehlo.exponential %v85 : tensor<128x1x10xf32>
    %v88 = stablehlo.reduce(%v87 init: %v86) applies stablehlo.add across dimensions = [2] : (tensor<128x1x10xf32>, tensor<f32>) -> tensor<128x1xf32>
    %v89 = stablehlo.broadcast_in_dim %v88, dims = [0, 1] : (tensor<128x1xf32>) -> tensor<128x1x10xf32>
    %v90 = stablehlo.divide %v87, %v89 : tensor<128x1x10xf32>
    %v91 = stablehlo.reshape %v90 : (tensor<128x1x10xf32>) -> tensor<128x10xf32>
    %v92 = stablehlo.subtract %v91, %onehot : tensor<128x10xf32>
    %v93 = stablehlo.constant dense<0.0078125> : tensor<128x10xf32>
    %v94 = stablehlo.multiply %v92, %v93 : tensor<128x10xf32>
    // ── report-only scalar loss (NOT pretty(AST): the kit has no rank-0 loss op; it
    //    feeds no parameter, only the driver's progress line) ──
    %llog = stablehlo.log %v91 : tensor<128x10xf32>
    %ohll = stablehlo.multiply %onehot, %llog : tensor<128x10xf32>
    %csum = stablehlo.reduce(%ohll init: %lzero) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<f32>
    %cneg = stablehlo.negate %csum : tensor<f32>
    %lbf = stablehlo.constant dense<128.0> : tensor<f32>
    %loss = stablehlo.divide %cneg, %lbf : tensor<f32>
    %v95 = stablehlo.reshape %v94 : (tensor<128x10xf32>) -> tensor<128x1x10xf32>
    %v96 = stablehlo.dot_general %v95, %Wb, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x10xf32>, tensor<512x10xf32>) -> tensor<128x1x512xf32>
    %v97 = stablehlo.reshape %v96 : (tensor<128x1x512xf32>) -> tensor<128x512xf32>
    %v98 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v99 = stablehlo.compare GT, %v79, %v98 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v100 = stablehlo.select %v99, %v97, %v98 : tensor<128x512xi1>, tensor<128x512xf32>
    %v101 = stablehlo.reshape %v100 : (tensor<128x512xf32>) -> tensor<128x1x512xf32>
    %v102 = stablehlo.dot_general %v101, %Wa, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x512xf32>, tensor<512x512xf32>) -> tensor<128x1x512xf32>
    %v103 = stablehlo.reshape %v102 : (tensor<128x1x512xf32>) -> tensor<128x512xf32>
    %v104 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v105 = stablehlo.compare GT, %v74, %v104 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v106 = stablehlo.select %v105, %v103, %v104 : tensor<128x512xi1>, tensor<128x512xf32>
    %v107 = stablehlo.reshape %v106 : (tensor<128x512xf32>) -> tensor<128x1x512xf32>
    %v108 = stablehlo.dot_general %v107, %W9, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x512xf32>, tensor<128x512xf32>) -> tensor<128x1x128xf32>
    %v109 = stablehlo.reshape %v108 : (tensor<128x1x128xf32>) -> tensor<128x128xf32>
    %v110 = stablehlo.reshape %v67 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v111 = stablehlo.reshape %v109 : (tensor<128x128xf32>) -> tensor<128x32x2x2xf32>
    %v112 = stablehlo.constant dense<0.0> : tensor<f32>
    %v113 = "stablehlo.select_and_scatter"(%v110, %v111, %v112) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x4x4xf32>, tensor<128x32x2x2xf32>, tensor<f32>) -> tensor<128x32x4x4xf32>
    %v114 = stablehlo.reshape %v113 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v115 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v116 = stablehlo.compare GT, %v65, %v115 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v117 = stablehlo.select %v116, %v114, %v115 : tensor<128x512xi1>, tensor<128x512xf32>
    %v118 = stablehlo.reshape %v117 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v119 = stablehlo.reverse %W8, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v120 = stablehlo.transpose %v119, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v121 = stablehlo.convolution(%v118, %v120)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v122 = stablehlo.reshape %v121 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v123 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v124 = stablehlo.compare GT, %v58, %v123 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v125 = stablehlo.select %v124, %v122, %v123 : tensor<128x512xi1>, tensor<128x512xf32>
    %v126 = stablehlo.reshape %v125 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v127 = stablehlo.reverse %W7, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v128 = stablehlo.transpose %v127, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v129 = stablehlo.convolution(%v126, %v128)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v130 = stablehlo.reshape %v129 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v131 = stablehlo.reshape %v49 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v132 = stablehlo.reshape %v130 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v133 = stablehlo.constant dense<0.0> : tensor<f32>
    %v134 = "stablehlo.select_and_scatter"(%v131, %v132, %v133) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x8x8xf32>, tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32x8x8xf32>
    %v135 = stablehlo.reshape %v134 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v136 = stablehlo.constant dense<0.0> : tensor<128x2048xf32>
    %v137 = stablehlo.compare GT, %v47, %v136 : (tensor<128x2048xf32>, tensor<128x2048xf32>) -> tensor<128x2048xi1>
    %v138 = stablehlo.select %v137, %v135, %v136 : tensor<128x2048xi1>, tensor<128x2048xf32>
    %v139 = stablehlo.reshape %v138 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v140 = stablehlo.reverse %W6, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v141 = stablehlo.transpose %v140, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v142 = stablehlo.convolution(%v139, %v141)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x8x8xf32>
    %v143 = stablehlo.reshape %v142 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v144 = stablehlo.constant dense<0.0> : tensor<128x2048xf32>
    %v145 = stablehlo.compare GT, %v40, %v144 : (tensor<128x2048xf32>, tensor<128x2048xf32>) -> tensor<128x2048xi1>
    %v146 = stablehlo.select %v145, %v143, %v144 : tensor<128x2048xi1>, tensor<128x2048xf32>
    %v147 = stablehlo.reshape %v146 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v148 = stablehlo.reverse %W5, dims = [2, 3] : tensor<32x16x3x3xf32>
    %v149 = stablehlo.transpose %v148, dims = [1, 0, 2, 3] : (tensor<32x16x3x3xf32>) -> tensor<16x32x3x3xf32>
    %v150 = stablehlo.convolution(%v147, %v149)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<16x32x3x3xf32>) -> tensor<128x16x8x8xf32>
    %v151 = stablehlo.reshape %v150 : (tensor<128x16x8x8xf32>) -> tensor<128x1024xf32>
    %v152 = stablehlo.reshape %v31 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v153 = stablehlo.reshape %v151 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v154 = stablehlo.constant dense<0.0> : tensor<f32>
    %v155 = "stablehlo.select_and_scatter"(%v152, %v153, %v154) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x16x16xf32>, tensor<128x16x8x8xf32>, tensor<f32>) -> tensor<128x16x16x16xf32>
    %v156 = stablehlo.reshape %v155 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v157 = stablehlo.constant dense<0.0> : tensor<128x4096xf32>
    %v158 = stablehlo.compare GT, %v29, %v157 : (tensor<128x4096xf32>, tensor<128x4096xf32>) -> tensor<128x4096xi1>
    %v159 = stablehlo.select %v158, %v156, %v157 : tensor<128x4096xi1>, tensor<128x4096xf32>
    %v160 = stablehlo.reshape %v159 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v161 = stablehlo.reverse %W4, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v162 = stablehlo.transpose %v161, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v163 = stablehlo.convolution(%v160, %v162)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v164 = stablehlo.reshape %v163 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v165 = stablehlo.constant dense<0.0> : tensor<128x4096xf32>
    %v166 = stablehlo.compare GT, %v22, %v165 : (tensor<128x4096xf32>, tensor<128x4096xf32>) -> tensor<128x4096xi1>
    %v167 = stablehlo.select %v166, %v164, %v165 : tensor<128x4096xi1>, tensor<128x4096xf32>
    %v168 = stablehlo.reshape %v167 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v169 = stablehlo.reverse %W3, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v170 = stablehlo.transpose %v169, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v171 = stablehlo.convolution(%v168, %v170)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v172 = stablehlo.reshape %v171 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v173 = stablehlo.reshape %v13 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v174 = stablehlo.reshape %v172 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v175 = stablehlo.constant dense<0.0> : tensor<f32>
    %v176 = "stablehlo.select_and_scatter"(%v173, %v174, %v175) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x32x32xf32>, tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x32x32xf32>
    %v177 = stablehlo.reshape %v176 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v178 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v179 = stablehlo.compare GT, %v11, %v178 : (tensor<128x16384xf32>, tensor<128x16384xf32>) -> tensor<128x16384xi1>
    %v180 = stablehlo.select %v179, %v177, %v178 : tensor<128x16384xi1>, tensor<128x16384xf32>
    %v181 = stablehlo.reshape %v180 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v182 = stablehlo.reverse %W2, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v183 = stablehlo.transpose %v182, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v184 = stablehlo.convolution(%v181, %v183)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x32x32xf32>
    %v185 = stablehlo.reshape %v184 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v186 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v187 = stablehlo.compare GT, %v4, %v186 : (tensor<128x16384xf32>, tensor<128x16384xf32>) -> tensor<128x16384xi1>
    %v188 = stablehlo.select %v187, %v185, %v186 : tensor<128x16384xi1>, tensor<128x16384xf32>
    %v189 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v190 = stablehlo.reshape %v188 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v191 = stablehlo.transpose %v189, dims = [1, 0, 2, 3] : (tensor<128x3x32x32xf32>) -> tensor<3x128x32x32xf32>
    %v192 = stablehlo.transpose %v190, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v193 = stablehlo.convolution(%v191, %v192)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<3x16x3x3xf32>
    %v194 = stablehlo.transpose %v193, dims = [1, 0, 2, 3] : (tensor<3x16x3x3xf32>) -> tensor<16x3x3x3xf32>
    %v195 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v196 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v197 = stablehlo.multiply %v195, %W1m : tensor<16x3x3x3xf32>
    %v198 = stablehlo.multiply %v196, %v194 : tensor<16x3x3x3xf32>
    %v199 = stablehlo.add %v197, %v198 : tensor<16x3x3x3xf32>
    %v200 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v201 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v202 = stablehlo.multiply %v200, %W1v : tensor<16x3x3x3xf32>
    %v203 = stablehlo.multiply %v194, %v194 : tensor<16x3x3x3xf32>
    %v204 = stablehlo.multiply %v201, %v203 : tensor<16x3x3x3xf32>
    %v205 = stablehlo.add %v202, %v204 : tensor<16x3x3x3xf32>
    %v206 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v207 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v208 = stablehlo.divide %v199, %v206 : tensor<16x3x3x3xf32>
    %v209 = stablehlo.divide %v205, %v207 : tensor<16x3x3x3xf32>
    %v210 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v211 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v212 = stablehlo.sqrt %v209 : tensor<16x3x3x3xf32>
    %v213 = stablehlo.add %v212, %v211 : tensor<16x3x3x3xf32>
    %v214 = stablehlo.divide %v208, %v213 : tensor<16x3x3x3xf32>
    %v215 = stablehlo.multiply %v210, %v214 : tensor<16x3x3x3xf32>
    %v216 = stablehlo.subtract %W1, %v215 : tensor<16x3x3x3xf32>
    %v217 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v218 = stablehlo.multiply %v217, %v210 : tensor<16x3x3x3xf32>
    %v219 = stablehlo.multiply %v218, %W1 : tensor<16x3x3x3xf32>
    %v220 = stablehlo.subtract %v216, %v219 : tensor<16x3x3x3xf32>
    %v221 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v222 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v223 = stablehlo.multiply %v221, %W1m : tensor<16x3x3x3xf32>
    %v224 = stablehlo.multiply %v222, %v194 : tensor<16x3x3x3xf32>
    %v225 = stablehlo.add %v223, %v224 : tensor<16x3x3x3xf32>
    %v226 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v227 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v228 = stablehlo.multiply %v226, %W1v : tensor<16x3x3x3xf32>
    %v229 = stablehlo.multiply %v194, %v194 : tensor<16x3x3x3xf32>
    %v230 = stablehlo.multiply %v227, %v229 : tensor<16x3x3x3xf32>
    %v231 = stablehlo.add %v228, %v230 : tensor<16x3x3x3xf32>
    %v232 = stablehlo.reshape %v188 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v233 = stablehlo.constant dense<0.0> : tensor<f32>
    %v234 = stablehlo.reduce(%v232 init: %v233) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v235 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v236 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v237 = stablehlo.multiply %v235, %cb1m : tensor<16xf32>
    %v238 = stablehlo.multiply %v236, %v234 : tensor<16xf32>
    %v239 = stablehlo.add %v237, %v238 : tensor<16xf32>
    %v240 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v241 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v242 = stablehlo.multiply %v240, %cb1v : tensor<16xf32>
    %v243 = stablehlo.multiply %v234, %v234 : tensor<16xf32>
    %v244 = stablehlo.multiply %v241, %v243 : tensor<16xf32>
    %v245 = stablehlo.add %v242, %v244 : tensor<16xf32>
    %v246 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v247 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v248 = stablehlo.divide %v239, %v246 : tensor<16xf32>
    %v249 = stablehlo.divide %v245, %v247 : tensor<16xf32>
    %v250 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v251 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v252 = stablehlo.sqrt %v249 : tensor<16xf32>
    %v253 = stablehlo.add %v252, %v251 : tensor<16xf32>
    %v254 = stablehlo.divide %v248, %v253 : tensor<16xf32>
    %v255 = stablehlo.multiply %v250, %v254 : tensor<16xf32>
    %v256 = stablehlo.subtract %cb1, %v255 : tensor<16xf32>
    %v257 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v258 = stablehlo.multiply %v257, %v250 : tensor<16xf32>
    %v259 = stablehlo.multiply %v258, %cb1 : tensor<16xf32>
    %v260 = stablehlo.subtract %v256, %v259 : tensor<16xf32>
    %v261 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v262 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v263 = stablehlo.multiply %v261, %cb1m : tensor<16xf32>
    %v264 = stablehlo.multiply %v262, %v234 : tensor<16xf32>
    %v265 = stablehlo.add %v263, %v264 : tensor<16xf32>
    %v266 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v267 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v268 = stablehlo.multiply %v266, %cb1v : tensor<16xf32>
    %v269 = stablehlo.multiply %v234, %v234 : tensor<16xf32>
    %v270 = stablehlo.multiply %v267, %v269 : tensor<16xf32>
    %v271 = stablehlo.add %v268, %v270 : tensor<16xf32>
    %v272 = stablehlo.reshape %v6 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v273 = stablehlo.reshape %v180 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v274 = stablehlo.transpose %v272, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v275 = stablehlo.transpose %v273, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v276 = stablehlo.convolution(%v274, %v275)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x16x3x3xf32>
    %v277 = stablehlo.transpose %v276, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v278 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v279 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v280 = stablehlo.multiply %v278, %W2m : tensor<16x16x3x3xf32>
    %v281 = stablehlo.multiply %v279, %v277 : tensor<16x16x3x3xf32>
    %v282 = stablehlo.add %v280, %v281 : tensor<16x16x3x3xf32>
    %v283 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v284 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v285 = stablehlo.multiply %v283, %W2v : tensor<16x16x3x3xf32>
    %v286 = stablehlo.multiply %v277, %v277 : tensor<16x16x3x3xf32>
    %v287 = stablehlo.multiply %v284, %v286 : tensor<16x16x3x3xf32>
    %v288 = stablehlo.add %v285, %v287 : tensor<16x16x3x3xf32>
    %v289 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v290 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v291 = stablehlo.divide %v282, %v289 : tensor<16x16x3x3xf32>
    %v292 = stablehlo.divide %v288, %v290 : tensor<16x16x3x3xf32>
    %v293 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v294 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v295 = stablehlo.sqrt %v292 : tensor<16x16x3x3xf32>
    %v296 = stablehlo.add %v295, %v294 : tensor<16x16x3x3xf32>
    %v297 = stablehlo.divide %v291, %v296 : tensor<16x16x3x3xf32>
    %v298 = stablehlo.multiply %v293, %v297 : tensor<16x16x3x3xf32>
    %v299 = stablehlo.subtract %W2, %v298 : tensor<16x16x3x3xf32>
    %v300 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v301 = stablehlo.multiply %v300, %v293 : tensor<16x16x3x3xf32>
    %v302 = stablehlo.multiply %v301, %W2 : tensor<16x16x3x3xf32>
    %v303 = stablehlo.subtract %v299, %v302 : tensor<16x16x3x3xf32>
    %v304 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v305 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v306 = stablehlo.multiply %v304, %W2m : tensor<16x16x3x3xf32>
    %v307 = stablehlo.multiply %v305, %v277 : tensor<16x16x3x3xf32>
    %v308 = stablehlo.add %v306, %v307 : tensor<16x16x3x3xf32>
    %v309 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v310 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v311 = stablehlo.multiply %v309, %W2v : tensor<16x16x3x3xf32>
    %v312 = stablehlo.multiply %v277, %v277 : tensor<16x16x3x3xf32>
    %v313 = stablehlo.multiply %v310, %v312 : tensor<16x16x3x3xf32>
    %v314 = stablehlo.add %v311, %v313 : tensor<16x16x3x3xf32>
    %v315 = stablehlo.reshape %v180 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v316 = stablehlo.constant dense<0.0> : tensor<f32>
    %v317 = stablehlo.reduce(%v315 init: %v316) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v318 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v319 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v320 = stablehlo.multiply %v318, %cb2m : tensor<16xf32>
    %v321 = stablehlo.multiply %v319, %v317 : tensor<16xf32>
    %v322 = stablehlo.add %v320, %v321 : tensor<16xf32>
    %v323 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v324 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v325 = stablehlo.multiply %v323, %cb2v : tensor<16xf32>
    %v326 = stablehlo.multiply %v317, %v317 : tensor<16xf32>
    %v327 = stablehlo.multiply %v324, %v326 : tensor<16xf32>
    %v328 = stablehlo.add %v325, %v327 : tensor<16xf32>
    %v329 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v330 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v331 = stablehlo.divide %v322, %v329 : tensor<16xf32>
    %v332 = stablehlo.divide %v328, %v330 : tensor<16xf32>
    %v333 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v334 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v335 = stablehlo.sqrt %v332 : tensor<16xf32>
    %v336 = stablehlo.add %v335, %v334 : tensor<16xf32>
    %v337 = stablehlo.divide %v331, %v336 : tensor<16xf32>
    %v338 = stablehlo.multiply %v333, %v337 : tensor<16xf32>
    %v339 = stablehlo.subtract %cb2, %v338 : tensor<16xf32>
    %v340 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v341 = stablehlo.multiply %v340, %v333 : tensor<16xf32>
    %v342 = stablehlo.multiply %v341, %cb2 : tensor<16xf32>
    %v343 = stablehlo.subtract %v339, %v342 : tensor<16xf32>
    %v344 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v345 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v346 = stablehlo.multiply %v344, %cb2m : tensor<16xf32>
    %v347 = stablehlo.multiply %v345, %v317 : tensor<16xf32>
    %v348 = stablehlo.add %v346, %v347 : tensor<16xf32>
    %v349 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v350 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v351 = stablehlo.multiply %v349, %cb2v : tensor<16xf32>
    %v352 = stablehlo.multiply %v317, %v317 : tensor<16xf32>
    %v353 = stablehlo.multiply %v350, %v352 : tensor<16xf32>
    %v354 = stablehlo.add %v351, %v353 : tensor<16xf32>
    %v355 = stablehlo.reshape %v17 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v356 = stablehlo.reshape %v167 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v357 = stablehlo.transpose %v355, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v358 = stablehlo.transpose %v356, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v359 = stablehlo.convolution(%v357, %v358)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v360 = stablehlo.transpose %v359, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v361 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v362 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v363 = stablehlo.multiply %v361, %W3m : tensor<16x16x3x3xf32>
    %v364 = stablehlo.multiply %v362, %v360 : tensor<16x16x3x3xf32>
    %v365 = stablehlo.add %v363, %v364 : tensor<16x16x3x3xf32>
    %v366 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v367 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v368 = stablehlo.multiply %v366, %W3v : tensor<16x16x3x3xf32>
    %v369 = stablehlo.multiply %v360, %v360 : tensor<16x16x3x3xf32>
    %v370 = stablehlo.multiply %v367, %v369 : tensor<16x16x3x3xf32>
    %v371 = stablehlo.add %v368, %v370 : tensor<16x16x3x3xf32>
    %v372 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v373 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v374 = stablehlo.divide %v365, %v372 : tensor<16x16x3x3xf32>
    %v375 = stablehlo.divide %v371, %v373 : tensor<16x16x3x3xf32>
    %v376 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v377 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v378 = stablehlo.sqrt %v375 : tensor<16x16x3x3xf32>
    %v379 = stablehlo.add %v378, %v377 : tensor<16x16x3x3xf32>
    %v380 = stablehlo.divide %v374, %v379 : tensor<16x16x3x3xf32>
    %v381 = stablehlo.multiply %v376, %v380 : tensor<16x16x3x3xf32>
    %v382 = stablehlo.subtract %W3, %v381 : tensor<16x16x3x3xf32>
    %v383 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v384 = stablehlo.multiply %v383, %v376 : tensor<16x16x3x3xf32>
    %v385 = stablehlo.multiply %v384, %W3 : tensor<16x16x3x3xf32>
    %v386 = stablehlo.subtract %v382, %v385 : tensor<16x16x3x3xf32>
    %v387 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v388 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v389 = stablehlo.multiply %v387, %W3m : tensor<16x16x3x3xf32>
    %v390 = stablehlo.multiply %v388, %v360 : tensor<16x16x3x3xf32>
    %v391 = stablehlo.add %v389, %v390 : tensor<16x16x3x3xf32>
    %v392 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v393 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v394 = stablehlo.multiply %v392, %W3v : tensor<16x16x3x3xf32>
    %v395 = stablehlo.multiply %v360, %v360 : tensor<16x16x3x3xf32>
    %v396 = stablehlo.multiply %v393, %v395 : tensor<16x16x3x3xf32>
    %v397 = stablehlo.add %v394, %v396 : tensor<16x16x3x3xf32>
    %v398 = stablehlo.reshape %v167 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v399 = stablehlo.constant dense<0.0> : tensor<f32>
    %v400 = stablehlo.reduce(%v398 init: %v399) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v401 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v402 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v403 = stablehlo.multiply %v401, %cb3m : tensor<16xf32>
    %v404 = stablehlo.multiply %v402, %v400 : tensor<16xf32>
    %v405 = stablehlo.add %v403, %v404 : tensor<16xf32>
    %v406 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v407 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v408 = stablehlo.multiply %v406, %cb3v : tensor<16xf32>
    %v409 = stablehlo.multiply %v400, %v400 : tensor<16xf32>
    %v410 = stablehlo.multiply %v407, %v409 : tensor<16xf32>
    %v411 = stablehlo.add %v408, %v410 : tensor<16xf32>
    %v412 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v413 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v414 = stablehlo.divide %v405, %v412 : tensor<16xf32>
    %v415 = stablehlo.divide %v411, %v413 : tensor<16xf32>
    %v416 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v417 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v418 = stablehlo.sqrt %v415 : tensor<16xf32>
    %v419 = stablehlo.add %v418, %v417 : tensor<16xf32>
    %v420 = stablehlo.divide %v414, %v419 : tensor<16xf32>
    %v421 = stablehlo.multiply %v416, %v420 : tensor<16xf32>
    %v422 = stablehlo.subtract %cb3, %v421 : tensor<16xf32>
    %v423 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v424 = stablehlo.multiply %v423, %v416 : tensor<16xf32>
    %v425 = stablehlo.multiply %v424, %cb3 : tensor<16xf32>
    %v426 = stablehlo.subtract %v422, %v425 : tensor<16xf32>
    %v427 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v428 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v429 = stablehlo.multiply %v427, %cb3m : tensor<16xf32>
    %v430 = stablehlo.multiply %v428, %v400 : tensor<16xf32>
    %v431 = stablehlo.add %v429, %v430 : tensor<16xf32>
    %v432 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v433 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v434 = stablehlo.multiply %v432, %cb3v : tensor<16xf32>
    %v435 = stablehlo.multiply %v400, %v400 : tensor<16xf32>
    %v436 = stablehlo.multiply %v433, %v435 : tensor<16xf32>
    %v437 = stablehlo.add %v434, %v436 : tensor<16xf32>
    %v438 = stablehlo.reshape %v24 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v439 = stablehlo.reshape %v159 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v440 = stablehlo.transpose %v438, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v441 = stablehlo.transpose %v439, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v442 = stablehlo.convolution(%v440, %v441)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v443 = stablehlo.transpose %v442, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v444 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v445 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v446 = stablehlo.multiply %v444, %W4m : tensor<16x16x3x3xf32>
    %v447 = stablehlo.multiply %v445, %v443 : tensor<16x16x3x3xf32>
    %v448 = stablehlo.add %v446, %v447 : tensor<16x16x3x3xf32>
    %v449 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v450 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v451 = stablehlo.multiply %v449, %W4v : tensor<16x16x3x3xf32>
    %v452 = stablehlo.multiply %v443, %v443 : tensor<16x16x3x3xf32>
    %v453 = stablehlo.multiply %v450, %v452 : tensor<16x16x3x3xf32>
    %v454 = stablehlo.add %v451, %v453 : tensor<16x16x3x3xf32>
    %v455 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v456 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v457 = stablehlo.divide %v448, %v455 : tensor<16x16x3x3xf32>
    %v458 = stablehlo.divide %v454, %v456 : tensor<16x16x3x3xf32>
    %v459 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v460 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v461 = stablehlo.sqrt %v458 : tensor<16x16x3x3xf32>
    %v462 = stablehlo.add %v461, %v460 : tensor<16x16x3x3xf32>
    %v463 = stablehlo.divide %v457, %v462 : tensor<16x16x3x3xf32>
    %v464 = stablehlo.multiply %v459, %v463 : tensor<16x16x3x3xf32>
    %v465 = stablehlo.subtract %W4, %v464 : tensor<16x16x3x3xf32>
    %v466 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v467 = stablehlo.multiply %v466, %v459 : tensor<16x16x3x3xf32>
    %v468 = stablehlo.multiply %v467, %W4 : tensor<16x16x3x3xf32>
    %v469 = stablehlo.subtract %v465, %v468 : tensor<16x16x3x3xf32>
    %v470 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v471 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v472 = stablehlo.multiply %v470, %W4m : tensor<16x16x3x3xf32>
    %v473 = stablehlo.multiply %v471, %v443 : tensor<16x16x3x3xf32>
    %v474 = stablehlo.add %v472, %v473 : tensor<16x16x3x3xf32>
    %v475 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v476 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v477 = stablehlo.multiply %v475, %W4v : tensor<16x16x3x3xf32>
    %v478 = stablehlo.multiply %v443, %v443 : tensor<16x16x3x3xf32>
    %v479 = stablehlo.multiply %v476, %v478 : tensor<16x16x3x3xf32>
    %v480 = stablehlo.add %v477, %v479 : tensor<16x16x3x3xf32>
    %v481 = stablehlo.reshape %v159 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v482 = stablehlo.constant dense<0.0> : tensor<f32>
    %v483 = stablehlo.reduce(%v481 init: %v482) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v484 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v485 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v486 = stablehlo.multiply %v484, %cb4m : tensor<16xf32>
    %v487 = stablehlo.multiply %v485, %v483 : tensor<16xf32>
    %v488 = stablehlo.add %v486, %v487 : tensor<16xf32>
    %v489 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v490 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v491 = stablehlo.multiply %v489, %cb4v : tensor<16xf32>
    %v492 = stablehlo.multiply %v483, %v483 : tensor<16xf32>
    %v493 = stablehlo.multiply %v490, %v492 : tensor<16xf32>
    %v494 = stablehlo.add %v491, %v493 : tensor<16xf32>
    %v495 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v496 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v497 = stablehlo.divide %v488, %v495 : tensor<16xf32>
    %v498 = stablehlo.divide %v494, %v496 : tensor<16xf32>
    %v499 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v500 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v501 = stablehlo.sqrt %v498 : tensor<16xf32>
    %v502 = stablehlo.add %v501, %v500 : tensor<16xf32>
    %v503 = stablehlo.divide %v497, %v502 : tensor<16xf32>
    %v504 = stablehlo.multiply %v499, %v503 : tensor<16xf32>
    %v505 = stablehlo.subtract %cb4, %v504 : tensor<16xf32>
    %v506 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v507 = stablehlo.multiply %v506, %v499 : tensor<16xf32>
    %v508 = stablehlo.multiply %v507, %cb4 : tensor<16xf32>
    %v509 = stablehlo.subtract %v505, %v508 : tensor<16xf32>
    %v510 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v511 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v512 = stablehlo.multiply %v510, %cb4m : tensor<16xf32>
    %v513 = stablehlo.multiply %v511, %v483 : tensor<16xf32>
    %v514 = stablehlo.add %v512, %v513 : tensor<16xf32>
    %v515 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v516 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v517 = stablehlo.multiply %v515, %cb4v : tensor<16xf32>
    %v518 = stablehlo.multiply %v483, %v483 : tensor<16xf32>
    %v519 = stablehlo.multiply %v516, %v518 : tensor<16xf32>
    %v520 = stablehlo.add %v517, %v519 : tensor<16xf32>
    %v521 = stablehlo.reshape %v35 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v522 = stablehlo.reshape %v146 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v523 = stablehlo.transpose %v521, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v524 = stablehlo.transpose %v522, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v525 = stablehlo.convolution(%v523, %v524)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<16x32x3x3xf32>
    %v526 = stablehlo.transpose %v525, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v527 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v528 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v529 = stablehlo.multiply %v527, %W5m : tensor<32x16x3x3xf32>
    %v530 = stablehlo.multiply %v528, %v526 : tensor<32x16x3x3xf32>
    %v531 = stablehlo.add %v529, %v530 : tensor<32x16x3x3xf32>
    %v532 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v533 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v534 = stablehlo.multiply %v532, %W5v : tensor<32x16x3x3xf32>
    %v535 = stablehlo.multiply %v526, %v526 : tensor<32x16x3x3xf32>
    %v536 = stablehlo.multiply %v533, %v535 : tensor<32x16x3x3xf32>
    %v537 = stablehlo.add %v534, %v536 : tensor<32x16x3x3xf32>
    %v538 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v539 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v540 = stablehlo.divide %v531, %v538 : tensor<32x16x3x3xf32>
    %v541 = stablehlo.divide %v537, %v539 : tensor<32x16x3x3xf32>
    %v542 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v543 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v544 = stablehlo.sqrt %v541 : tensor<32x16x3x3xf32>
    %v545 = stablehlo.add %v544, %v543 : tensor<32x16x3x3xf32>
    %v546 = stablehlo.divide %v540, %v545 : tensor<32x16x3x3xf32>
    %v547 = stablehlo.multiply %v542, %v546 : tensor<32x16x3x3xf32>
    %v548 = stablehlo.subtract %W5, %v547 : tensor<32x16x3x3xf32>
    %v549 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v550 = stablehlo.multiply %v549, %v542 : tensor<32x16x3x3xf32>
    %v551 = stablehlo.multiply %v550, %W5 : tensor<32x16x3x3xf32>
    %v552 = stablehlo.subtract %v548, %v551 : tensor<32x16x3x3xf32>
    %v553 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v554 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v555 = stablehlo.multiply %v553, %W5m : tensor<32x16x3x3xf32>
    %v556 = stablehlo.multiply %v554, %v526 : tensor<32x16x3x3xf32>
    %v557 = stablehlo.add %v555, %v556 : tensor<32x16x3x3xf32>
    %v558 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v559 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v560 = stablehlo.multiply %v558, %W5v : tensor<32x16x3x3xf32>
    %v561 = stablehlo.multiply %v526, %v526 : tensor<32x16x3x3xf32>
    %v562 = stablehlo.multiply %v559, %v561 : tensor<32x16x3x3xf32>
    %v563 = stablehlo.add %v560, %v562 : tensor<32x16x3x3xf32>
    %v564 = stablehlo.reshape %v146 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v565 = stablehlo.constant dense<0.0> : tensor<f32>
    %v566 = stablehlo.reduce(%v564 init: %v565) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v567 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v568 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v569 = stablehlo.multiply %v567, %cb5m : tensor<32xf32>
    %v570 = stablehlo.multiply %v568, %v566 : tensor<32xf32>
    %v571 = stablehlo.add %v569, %v570 : tensor<32xf32>
    %v572 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v573 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v574 = stablehlo.multiply %v572, %cb5v : tensor<32xf32>
    %v575 = stablehlo.multiply %v566, %v566 : tensor<32xf32>
    %v576 = stablehlo.multiply %v573, %v575 : tensor<32xf32>
    %v577 = stablehlo.add %v574, %v576 : tensor<32xf32>
    %v578 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v579 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v580 = stablehlo.divide %v571, %v578 : tensor<32xf32>
    %v581 = stablehlo.divide %v577, %v579 : tensor<32xf32>
    %v582 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v583 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v584 = stablehlo.sqrt %v581 : tensor<32xf32>
    %v585 = stablehlo.add %v584, %v583 : tensor<32xf32>
    %v586 = stablehlo.divide %v580, %v585 : tensor<32xf32>
    %v587 = stablehlo.multiply %v582, %v586 : tensor<32xf32>
    %v588 = stablehlo.subtract %cb5, %v587 : tensor<32xf32>
    %v589 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v590 = stablehlo.multiply %v589, %v582 : tensor<32xf32>
    %v591 = stablehlo.multiply %v590, %cb5 : tensor<32xf32>
    %v592 = stablehlo.subtract %v588, %v591 : tensor<32xf32>
    %v593 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v594 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v595 = stablehlo.multiply %v593, %cb5m : tensor<32xf32>
    %v596 = stablehlo.multiply %v594, %v566 : tensor<32xf32>
    %v597 = stablehlo.add %v595, %v596 : tensor<32xf32>
    %v598 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v599 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v600 = stablehlo.multiply %v598, %cb5v : tensor<32xf32>
    %v601 = stablehlo.multiply %v566, %v566 : tensor<32xf32>
    %v602 = stablehlo.multiply %v599, %v601 : tensor<32xf32>
    %v603 = stablehlo.add %v600, %v602 : tensor<32xf32>
    %v604 = stablehlo.reshape %v42 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v605 = stablehlo.reshape %v138 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v606 = stablehlo.transpose %v604, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v607 = stablehlo.transpose %v605, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v608 = stablehlo.convolution(%v606, %v607)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<32x32x3x3xf32>
    %v609 = stablehlo.transpose %v608, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v610 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v611 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v612 = stablehlo.multiply %v610, %W6m : tensor<32x32x3x3xf32>
    %v613 = stablehlo.multiply %v611, %v609 : tensor<32x32x3x3xf32>
    %v614 = stablehlo.add %v612, %v613 : tensor<32x32x3x3xf32>
    %v615 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v616 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v617 = stablehlo.multiply %v615, %W6v : tensor<32x32x3x3xf32>
    %v618 = stablehlo.multiply %v609, %v609 : tensor<32x32x3x3xf32>
    %v619 = stablehlo.multiply %v616, %v618 : tensor<32x32x3x3xf32>
    %v620 = stablehlo.add %v617, %v619 : tensor<32x32x3x3xf32>
    %v621 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v622 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v623 = stablehlo.divide %v614, %v621 : tensor<32x32x3x3xf32>
    %v624 = stablehlo.divide %v620, %v622 : tensor<32x32x3x3xf32>
    %v625 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v626 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v627 = stablehlo.sqrt %v624 : tensor<32x32x3x3xf32>
    %v628 = stablehlo.add %v627, %v626 : tensor<32x32x3x3xf32>
    %v629 = stablehlo.divide %v623, %v628 : tensor<32x32x3x3xf32>
    %v630 = stablehlo.multiply %v625, %v629 : tensor<32x32x3x3xf32>
    %v631 = stablehlo.subtract %W6, %v630 : tensor<32x32x3x3xf32>
    %v632 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v633 = stablehlo.multiply %v632, %v625 : tensor<32x32x3x3xf32>
    %v634 = stablehlo.multiply %v633, %W6 : tensor<32x32x3x3xf32>
    %v635 = stablehlo.subtract %v631, %v634 : tensor<32x32x3x3xf32>
    %v636 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v637 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v638 = stablehlo.multiply %v636, %W6m : tensor<32x32x3x3xf32>
    %v639 = stablehlo.multiply %v637, %v609 : tensor<32x32x3x3xf32>
    %v640 = stablehlo.add %v638, %v639 : tensor<32x32x3x3xf32>
    %v641 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v642 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v643 = stablehlo.multiply %v641, %W6v : tensor<32x32x3x3xf32>
    %v644 = stablehlo.multiply %v609, %v609 : tensor<32x32x3x3xf32>
    %v645 = stablehlo.multiply %v642, %v644 : tensor<32x32x3x3xf32>
    %v646 = stablehlo.add %v643, %v645 : tensor<32x32x3x3xf32>
    %v647 = stablehlo.reshape %v138 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v648 = stablehlo.constant dense<0.0> : tensor<f32>
    %v649 = stablehlo.reduce(%v647 init: %v648) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v650 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v651 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v652 = stablehlo.multiply %v650, %cb6m : tensor<32xf32>
    %v653 = stablehlo.multiply %v651, %v649 : tensor<32xf32>
    %v654 = stablehlo.add %v652, %v653 : tensor<32xf32>
    %v655 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v656 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v657 = stablehlo.multiply %v655, %cb6v : tensor<32xf32>
    %v658 = stablehlo.multiply %v649, %v649 : tensor<32xf32>
    %v659 = stablehlo.multiply %v656, %v658 : tensor<32xf32>
    %v660 = stablehlo.add %v657, %v659 : tensor<32xf32>
    %v661 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v662 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v663 = stablehlo.divide %v654, %v661 : tensor<32xf32>
    %v664 = stablehlo.divide %v660, %v662 : tensor<32xf32>
    %v665 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v666 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v667 = stablehlo.sqrt %v664 : tensor<32xf32>
    %v668 = stablehlo.add %v667, %v666 : tensor<32xf32>
    %v669 = stablehlo.divide %v663, %v668 : tensor<32xf32>
    %v670 = stablehlo.multiply %v665, %v669 : tensor<32xf32>
    %v671 = stablehlo.subtract %cb6, %v670 : tensor<32xf32>
    %v672 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v673 = stablehlo.multiply %v672, %v665 : tensor<32xf32>
    %v674 = stablehlo.multiply %v673, %cb6 : tensor<32xf32>
    %v675 = stablehlo.subtract %v671, %v674 : tensor<32xf32>
    %v676 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v677 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v678 = stablehlo.multiply %v676, %cb6m : tensor<32xf32>
    %v679 = stablehlo.multiply %v677, %v649 : tensor<32xf32>
    %v680 = stablehlo.add %v678, %v679 : tensor<32xf32>
    %v681 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v682 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v683 = stablehlo.multiply %v681, %cb6v : tensor<32xf32>
    %v684 = stablehlo.multiply %v649, %v649 : tensor<32xf32>
    %v685 = stablehlo.multiply %v682, %v684 : tensor<32xf32>
    %v686 = stablehlo.add %v683, %v685 : tensor<32xf32>
    %v687 = stablehlo.reshape %v53 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v688 = stablehlo.reshape %v125 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v689 = stablehlo.transpose %v687, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v690 = stablehlo.transpose %v688, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v691 = stablehlo.convolution(%v689, %v690)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v692 = stablehlo.transpose %v691, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v693 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v694 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v695 = stablehlo.multiply %v693, %W7m : tensor<32x32x3x3xf32>
    %v696 = stablehlo.multiply %v694, %v692 : tensor<32x32x3x3xf32>
    %v697 = stablehlo.add %v695, %v696 : tensor<32x32x3x3xf32>
    %v698 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v699 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v700 = stablehlo.multiply %v698, %W7v : tensor<32x32x3x3xf32>
    %v701 = stablehlo.multiply %v692, %v692 : tensor<32x32x3x3xf32>
    %v702 = stablehlo.multiply %v699, %v701 : tensor<32x32x3x3xf32>
    %v703 = stablehlo.add %v700, %v702 : tensor<32x32x3x3xf32>
    %v704 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v705 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v706 = stablehlo.divide %v697, %v704 : tensor<32x32x3x3xf32>
    %v707 = stablehlo.divide %v703, %v705 : tensor<32x32x3x3xf32>
    %v708 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v709 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v710 = stablehlo.sqrt %v707 : tensor<32x32x3x3xf32>
    %v711 = stablehlo.add %v710, %v709 : tensor<32x32x3x3xf32>
    %v712 = stablehlo.divide %v706, %v711 : tensor<32x32x3x3xf32>
    %v713 = stablehlo.multiply %v708, %v712 : tensor<32x32x3x3xf32>
    %v714 = stablehlo.subtract %W7, %v713 : tensor<32x32x3x3xf32>
    %v715 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v716 = stablehlo.multiply %v715, %v708 : tensor<32x32x3x3xf32>
    %v717 = stablehlo.multiply %v716, %W7 : tensor<32x32x3x3xf32>
    %v718 = stablehlo.subtract %v714, %v717 : tensor<32x32x3x3xf32>
    %v719 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v720 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v721 = stablehlo.multiply %v719, %W7m : tensor<32x32x3x3xf32>
    %v722 = stablehlo.multiply %v720, %v692 : tensor<32x32x3x3xf32>
    %v723 = stablehlo.add %v721, %v722 : tensor<32x32x3x3xf32>
    %v724 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v725 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v726 = stablehlo.multiply %v724, %W7v : tensor<32x32x3x3xf32>
    %v727 = stablehlo.multiply %v692, %v692 : tensor<32x32x3x3xf32>
    %v728 = stablehlo.multiply %v725, %v727 : tensor<32x32x3x3xf32>
    %v729 = stablehlo.add %v726, %v728 : tensor<32x32x3x3xf32>
    %v730 = stablehlo.reshape %v125 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v731 = stablehlo.constant dense<0.0> : tensor<f32>
    %v732 = stablehlo.reduce(%v730 init: %v731) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v733 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v734 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v735 = stablehlo.multiply %v733, %cb7m : tensor<32xf32>
    %v736 = stablehlo.multiply %v734, %v732 : tensor<32xf32>
    %v737 = stablehlo.add %v735, %v736 : tensor<32xf32>
    %v738 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v739 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v740 = stablehlo.multiply %v738, %cb7v : tensor<32xf32>
    %v741 = stablehlo.multiply %v732, %v732 : tensor<32xf32>
    %v742 = stablehlo.multiply %v739, %v741 : tensor<32xf32>
    %v743 = stablehlo.add %v740, %v742 : tensor<32xf32>
    %v744 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v745 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v746 = stablehlo.divide %v737, %v744 : tensor<32xf32>
    %v747 = stablehlo.divide %v743, %v745 : tensor<32xf32>
    %v748 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v749 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v750 = stablehlo.sqrt %v747 : tensor<32xf32>
    %v751 = stablehlo.add %v750, %v749 : tensor<32xf32>
    %v752 = stablehlo.divide %v746, %v751 : tensor<32xf32>
    %v753 = stablehlo.multiply %v748, %v752 : tensor<32xf32>
    %v754 = stablehlo.subtract %cb7, %v753 : tensor<32xf32>
    %v755 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v756 = stablehlo.multiply %v755, %v748 : tensor<32xf32>
    %v757 = stablehlo.multiply %v756, %cb7 : tensor<32xf32>
    %v758 = stablehlo.subtract %v754, %v757 : tensor<32xf32>
    %v759 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v760 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v761 = stablehlo.multiply %v759, %cb7m : tensor<32xf32>
    %v762 = stablehlo.multiply %v760, %v732 : tensor<32xf32>
    %v763 = stablehlo.add %v761, %v762 : tensor<32xf32>
    %v764 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v765 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v766 = stablehlo.multiply %v764, %cb7v : tensor<32xf32>
    %v767 = stablehlo.multiply %v732, %v732 : tensor<32xf32>
    %v768 = stablehlo.multiply %v765, %v767 : tensor<32xf32>
    %v769 = stablehlo.add %v766, %v768 : tensor<32xf32>
    %v770 = stablehlo.reshape %v60 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v771 = stablehlo.reshape %v117 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v772 = stablehlo.transpose %v770, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v773 = stablehlo.transpose %v771, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v774 = stablehlo.convolution(%v772, %v773)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v775 = stablehlo.transpose %v774, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v776 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v777 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v778 = stablehlo.multiply %v776, %W8m : tensor<32x32x3x3xf32>
    %v779 = stablehlo.multiply %v777, %v775 : tensor<32x32x3x3xf32>
    %v780 = stablehlo.add %v778, %v779 : tensor<32x32x3x3xf32>
    %v781 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v782 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v783 = stablehlo.multiply %v781, %W8v : tensor<32x32x3x3xf32>
    %v784 = stablehlo.multiply %v775, %v775 : tensor<32x32x3x3xf32>
    %v785 = stablehlo.multiply %v782, %v784 : tensor<32x32x3x3xf32>
    %v786 = stablehlo.add %v783, %v785 : tensor<32x32x3x3xf32>
    %v787 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v788 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v789 = stablehlo.divide %v780, %v787 : tensor<32x32x3x3xf32>
    %v790 = stablehlo.divide %v786, %v788 : tensor<32x32x3x3xf32>
    %v791 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v792 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v793 = stablehlo.sqrt %v790 : tensor<32x32x3x3xf32>
    %v794 = stablehlo.add %v793, %v792 : tensor<32x32x3x3xf32>
    %v795 = stablehlo.divide %v789, %v794 : tensor<32x32x3x3xf32>
    %v796 = stablehlo.multiply %v791, %v795 : tensor<32x32x3x3xf32>
    %v797 = stablehlo.subtract %W8, %v796 : tensor<32x32x3x3xf32>
    %v798 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v799 = stablehlo.multiply %v798, %v791 : tensor<32x32x3x3xf32>
    %v800 = stablehlo.multiply %v799, %W8 : tensor<32x32x3x3xf32>
    %v801 = stablehlo.subtract %v797, %v800 : tensor<32x32x3x3xf32>
    %v802 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v803 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v804 = stablehlo.multiply %v802, %W8m : tensor<32x32x3x3xf32>
    %v805 = stablehlo.multiply %v803, %v775 : tensor<32x32x3x3xf32>
    %v806 = stablehlo.add %v804, %v805 : tensor<32x32x3x3xf32>
    %v807 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v808 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v809 = stablehlo.multiply %v807, %W8v : tensor<32x32x3x3xf32>
    %v810 = stablehlo.multiply %v775, %v775 : tensor<32x32x3x3xf32>
    %v811 = stablehlo.multiply %v808, %v810 : tensor<32x32x3x3xf32>
    %v812 = stablehlo.add %v809, %v811 : tensor<32x32x3x3xf32>
    %v813 = stablehlo.reshape %v117 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v814 = stablehlo.constant dense<0.0> : tensor<f32>
    %v815 = stablehlo.reduce(%v813 init: %v814) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v816 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v817 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v818 = stablehlo.multiply %v816, %cb8m : tensor<32xf32>
    %v819 = stablehlo.multiply %v817, %v815 : tensor<32xf32>
    %v820 = stablehlo.add %v818, %v819 : tensor<32xf32>
    %v821 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v822 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v823 = stablehlo.multiply %v821, %cb8v : tensor<32xf32>
    %v824 = stablehlo.multiply %v815, %v815 : tensor<32xf32>
    %v825 = stablehlo.multiply %v822, %v824 : tensor<32xf32>
    %v826 = stablehlo.add %v823, %v825 : tensor<32xf32>
    %v827 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v828 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v829 = stablehlo.divide %v820, %v827 : tensor<32xf32>
    %v830 = stablehlo.divide %v826, %v828 : tensor<32xf32>
    %v831 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v832 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v833 = stablehlo.sqrt %v830 : tensor<32xf32>
    %v834 = stablehlo.add %v833, %v832 : tensor<32xf32>
    %v835 = stablehlo.divide %v829, %v834 : tensor<32xf32>
    %v836 = stablehlo.multiply %v831, %v835 : tensor<32xf32>
    %v837 = stablehlo.subtract %cb8, %v836 : tensor<32xf32>
    %v838 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v839 = stablehlo.multiply %v838, %v831 : tensor<32xf32>
    %v840 = stablehlo.multiply %v839, %cb8 : tensor<32xf32>
    %v841 = stablehlo.subtract %v837, %v840 : tensor<32xf32>
    %v842 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v843 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v844 = stablehlo.multiply %v842, %cb8m : tensor<32xf32>
    %v845 = stablehlo.multiply %v843, %v815 : tensor<32xf32>
    %v846 = stablehlo.add %v844, %v845 : tensor<32xf32>
    %v847 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v848 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v849 = stablehlo.multiply %v847, %cb8v : tensor<32xf32>
    %v850 = stablehlo.multiply %v815, %v815 : tensor<32xf32>
    %v851 = stablehlo.multiply %v848, %v850 : tensor<32xf32>
    %v852 = stablehlo.add %v849, %v851 : tensor<32xf32>
    %v853 = stablehlo.dot_general %v71, %v106, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x512xf32>) -> tensor<128x512xf32>
    %v854 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v855 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v856 = stablehlo.multiply %v854, %W9m : tensor<128x512xf32>
    %v857 = stablehlo.multiply %v855, %v853 : tensor<128x512xf32>
    %v858 = stablehlo.add %v856, %v857 : tensor<128x512xf32>
    %v859 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v860 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v861 = stablehlo.multiply %v859, %W9v : tensor<128x512xf32>
    %v862 = stablehlo.multiply %v853, %v853 : tensor<128x512xf32>
    %v863 = stablehlo.multiply %v860, %v862 : tensor<128x512xf32>
    %v864 = stablehlo.add %v861, %v863 : tensor<128x512xf32>
    %v865 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v866 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v867 = stablehlo.divide %v858, %v865 : tensor<128x512xf32>
    %v868 = stablehlo.divide %v864, %v866 : tensor<128x512xf32>
    %v869 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v870 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v871 = stablehlo.sqrt %v868 : tensor<128x512xf32>
    %v872 = stablehlo.add %v871, %v870 : tensor<128x512xf32>
    %v873 = stablehlo.divide %v867, %v872 : tensor<128x512xf32>
    %v874 = stablehlo.multiply %v869, %v873 : tensor<128x512xf32>
    %v875 = stablehlo.subtract %W9, %v874 : tensor<128x512xf32>
    %v876 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v877 = stablehlo.multiply %v876, %v869 : tensor<128x512xf32>
    %v878 = stablehlo.multiply %v877, %W9 : tensor<128x512xf32>
    %v879 = stablehlo.subtract %v875, %v878 : tensor<128x512xf32>
    %v880 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v881 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v882 = stablehlo.multiply %v880, %W9m : tensor<128x512xf32>
    %v883 = stablehlo.multiply %v881, %v853 : tensor<128x512xf32>
    %v884 = stablehlo.add %v882, %v883 : tensor<128x512xf32>
    %v885 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v886 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v887 = stablehlo.multiply %v885, %W9v : tensor<128x512xf32>
    %v888 = stablehlo.multiply %v853, %v853 : tensor<128x512xf32>
    %v889 = stablehlo.multiply %v886, %v888 : tensor<128x512xf32>
    %v890 = stablehlo.add %v887, %v889 : tensor<128x512xf32>
    %v891 = stablehlo.constant dense<0.0> : tensor<f32>
    %v892 = stablehlo.reduce(%v106 init: %v891) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v893 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v894 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v895 = stablehlo.multiply %v893, %b9m : tensor<512xf32>
    %v896 = stablehlo.multiply %v894, %v892 : tensor<512xf32>
    %v897 = stablehlo.add %v895, %v896 : tensor<512xf32>
    %v898 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v899 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v900 = stablehlo.multiply %v898, %b9v : tensor<512xf32>
    %v901 = stablehlo.multiply %v892, %v892 : tensor<512xf32>
    %v902 = stablehlo.multiply %v899, %v901 : tensor<512xf32>
    %v903 = stablehlo.add %v900, %v902 : tensor<512xf32>
    %v904 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v905 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v906 = stablehlo.divide %v897, %v904 : tensor<512xf32>
    %v907 = stablehlo.divide %v903, %v905 : tensor<512xf32>
    %v908 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v909 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v910 = stablehlo.sqrt %v907 : tensor<512xf32>
    %v911 = stablehlo.add %v910, %v909 : tensor<512xf32>
    %v912 = stablehlo.divide %v906, %v911 : tensor<512xf32>
    %v913 = stablehlo.multiply %v908, %v912 : tensor<512xf32>
    %v914 = stablehlo.subtract %b9, %v913 : tensor<512xf32>
    %v915 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v916 = stablehlo.multiply %v915, %v908 : tensor<512xf32>
    %v917 = stablehlo.multiply %v916, %b9 : tensor<512xf32>
    %v918 = stablehlo.subtract %v914, %v917 : tensor<512xf32>
    %v919 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v920 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v921 = stablehlo.multiply %v919, %b9m : tensor<512xf32>
    %v922 = stablehlo.multiply %v920, %v892 : tensor<512xf32>
    %v923 = stablehlo.add %v921, %v922 : tensor<512xf32>
    %v924 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v925 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v926 = stablehlo.multiply %v924, %b9v : tensor<512xf32>
    %v927 = stablehlo.multiply %v892, %v892 : tensor<512xf32>
    %v928 = stablehlo.multiply %v925, %v927 : tensor<512xf32>
    %v929 = stablehlo.add %v926, %v928 : tensor<512xf32>
    %v930 = stablehlo.dot_general %v76, %v100, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<512x512xf32>
    %v931 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v932 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v933 = stablehlo.multiply %v931, %Wam : tensor<512x512xf32>
    %v934 = stablehlo.multiply %v932, %v930 : tensor<512x512xf32>
    %v935 = stablehlo.add %v933, %v934 : tensor<512x512xf32>
    %v936 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v937 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v938 = stablehlo.multiply %v936, %Wav : tensor<512x512xf32>
    %v939 = stablehlo.multiply %v930, %v930 : tensor<512x512xf32>
    %v940 = stablehlo.multiply %v937, %v939 : tensor<512x512xf32>
    %v941 = stablehlo.add %v938, %v940 : tensor<512x512xf32>
    %v942 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v943 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v944 = stablehlo.divide %v935, %v942 : tensor<512x512xf32>
    %v945 = stablehlo.divide %v941, %v943 : tensor<512x512xf32>
    %v946 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v947 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v948 = stablehlo.sqrt %v945 : tensor<512x512xf32>
    %v949 = stablehlo.add %v948, %v947 : tensor<512x512xf32>
    %v950 = stablehlo.divide %v944, %v949 : tensor<512x512xf32>
    %v951 = stablehlo.multiply %v946, %v950 : tensor<512x512xf32>
    %v952 = stablehlo.subtract %Wa, %v951 : tensor<512x512xf32>
    %v953 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v954 = stablehlo.multiply %v953, %v946 : tensor<512x512xf32>
    %v955 = stablehlo.multiply %v954, %Wa : tensor<512x512xf32>
    %v956 = stablehlo.subtract %v952, %v955 : tensor<512x512xf32>
    %v957 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v958 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v959 = stablehlo.multiply %v957, %Wam : tensor<512x512xf32>
    %v960 = stablehlo.multiply %v958, %v930 : tensor<512x512xf32>
    %v961 = stablehlo.add %v959, %v960 : tensor<512x512xf32>
    %v962 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v963 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v964 = stablehlo.multiply %v962, %Wav : tensor<512x512xf32>
    %v965 = stablehlo.multiply %v930, %v930 : tensor<512x512xf32>
    %v966 = stablehlo.multiply %v963, %v965 : tensor<512x512xf32>
    %v967 = stablehlo.add %v964, %v966 : tensor<512x512xf32>
    %v968 = stablehlo.constant dense<0.0> : tensor<f32>
    %v969 = stablehlo.reduce(%v100 init: %v968) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v970 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v971 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v972 = stablehlo.multiply %v970, %bam : tensor<512xf32>
    %v973 = stablehlo.multiply %v971, %v969 : tensor<512xf32>
    %v974 = stablehlo.add %v972, %v973 : tensor<512xf32>
    %v975 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v976 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v977 = stablehlo.multiply %v975, %bav : tensor<512xf32>
    %v978 = stablehlo.multiply %v969, %v969 : tensor<512xf32>
    %v979 = stablehlo.multiply %v976, %v978 : tensor<512xf32>
    %v980 = stablehlo.add %v977, %v979 : tensor<512xf32>
    %v981 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v982 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v983 = stablehlo.divide %v974, %v981 : tensor<512xf32>
    %v984 = stablehlo.divide %v980, %v982 : tensor<512xf32>
    %v985 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v986 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v987 = stablehlo.sqrt %v984 : tensor<512xf32>
    %v988 = stablehlo.add %v987, %v986 : tensor<512xf32>
    %v989 = stablehlo.divide %v983, %v988 : tensor<512xf32>
    %v990 = stablehlo.multiply %v985, %v989 : tensor<512xf32>
    %v991 = stablehlo.subtract %ba, %v990 : tensor<512xf32>
    %v992 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v993 = stablehlo.multiply %v992, %v985 : tensor<512xf32>
    %v994 = stablehlo.multiply %v993, %ba : tensor<512xf32>
    %v995 = stablehlo.subtract %v991, %v994 : tensor<512xf32>
    %v996 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v997 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v998 = stablehlo.multiply %v996, %bam : tensor<512xf32>
    %v999 = stablehlo.multiply %v997, %v969 : tensor<512xf32>
    %v1000 = stablehlo.add %v998, %v999 : tensor<512xf32>
    %v1001 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1002 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1003 = stablehlo.multiply %v1001, %bav : tensor<512xf32>
    %v1004 = stablehlo.multiply %v969, %v969 : tensor<512xf32>
    %v1005 = stablehlo.multiply %v1002, %v1004 : tensor<512xf32>
    %v1006 = stablehlo.add %v1003, %v1005 : tensor<512xf32>
    %v1007 = stablehlo.dot_general %v81, %v94, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x10xf32>) -> tensor<512x10xf32>
    %v1008 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1009 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1010 = stablehlo.multiply %v1008, %Wbm : tensor<512x10xf32>
    %v1011 = stablehlo.multiply %v1009, %v1007 : tensor<512x10xf32>
    %v1012 = stablehlo.add %v1010, %v1011 : tensor<512x10xf32>
    %v1013 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1014 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1015 = stablehlo.multiply %v1013, %Wbv : tensor<512x10xf32>
    %v1016 = stablehlo.multiply %v1007, %v1007 : tensor<512x10xf32>
    %v1017 = stablehlo.multiply %v1014, %v1016 : tensor<512x10xf32>
    %v1018 = stablehlo.add %v1015, %v1017 : tensor<512x10xf32>
    %v1019 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1020 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1021 = stablehlo.divide %v1012, %v1019 : tensor<512x10xf32>
    %v1022 = stablehlo.divide %v1018, %v1020 : tensor<512x10xf32>
    %v1023 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1024 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1025 = stablehlo.sqrt %v1022 : tensor<512x10xf32>
    %v1026 = stablehlo.add %v1025, %v1024 : tensor<512x10xf32>
    %v1027 = stablehlo.divide %v1021, %v1026 : tensor<512x10xf32>
    %v1028 = stablehlo.multiply %v1023, %v1027 : tensor<512x10xf32>
    %v1029 = stablehlo.subtract %Wb, %v1028 : tensor<512x10xf32>
    %v1030 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1031 = stablehlo.multiply %v1030, %v1023 : tensor<512x10xf32>
    %v1032 = stablehlo.multiply %v1031, %Wb : tensor<512x10xf32>
    %v1033 = stablehlo.subtract %v1029, %v1032 : tensor<512x10xf32>
    %v1034 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1035 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1036 = stablehlo.multiply %v1034, %Wbm : tensor<512x10xf32>
    %v1037 = stablehlo.multiply %v1035, %v1007 : tensor<512x10xf32>
    %v1038 = stablehlo.add %v1036, %v1037 : tensor<512x10xf32>
    %v1039 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1040 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1041 = stablehlo.multiply %v1039, %Wbv : tensor<512x10xf32>
    %v1042 = stablehlo.multiply %v1007, %v1007 : tensor<512x10xf32>
    %v1043 = stablehlo.multiply %v1040, %v1042 : tensor<512x10xf32>
    %v1044 = stablehlo.add %v1041, %v1043 : tensor<512x10xf32>
    %v1045 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1046 = stablehlo.reduce(%v94 init: %v1045) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1047 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1048 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1049 = stablehlo.multiply %v1047, %bbm : tensor<10xf32>
    %v1050 = stablehlo.multiply %v1048, %v1046 : tensor<10xf32>
    %v1051 = stablehlo.add %v1049, %v1050 : tensor<10xf32>
    %v1052 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1053 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1054 = stablehlo.multiply %v1052, %bbv : tensor<10xf32>
    %v1055 = stablehlo.multiply %v1046, %v1046 : tensor<10xf32>
    %v1056 = stablehlo.multiply %v1053, %v1055 : tensor<10xf32>
    %v1057 = stablehlo.add %v1054, %v1056 : tensor<10xf32>
    %v1058 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1059 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1060 = stablehlo.divide %v1051, %v1058 : tensor<10xf32>
    %v1061 = stablehlo.divide %v1057, %v1059 : tensor<10xf32>
    %v1062 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1063 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1064 = stablehlo.sqrt %v1061 : tensor<10xf32>
    %v1065 = stablehlo.add %v1064, %v1063 : tensor<10xf32>
    %v1066 = stablehlo.divide %v1060, %v1065 : tensor<10xf32>
    %v1067 = stablehlo.multiply %v1062, %v1066 : tensor<10xf32>
    %v1068 = stablehlo.subtract %bb, %v1067 : tensor<10xf32>
    %v1069 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1070 = stablehlo.multiply %v1069, %v1062 : tensor<10xf32>
    %v1071 = stablehlo.multiply %v1070, %bb : tensor<10xf32>
    %v1072 = stablehlo.subtract %v1068, %v1071 : tensor<10xf32>
    %v1073 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1074 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1075 = stablehlo.multiply %v1073, %bbm : tensor<10xf32>
    %v1076 = stablehlo.multiply %v1074, %v1046 : tensor<10xf32>
    %v1077 = stablehlo.add %v1075, %v1076 : tensor<10xf32>
    %v1078 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1079 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1080 = stablehlo.multiply %v1078, %bbv : tensor<10xf32>
    %v1081 = stablehlo.multiply %v1046, %v1046 : tensor<10xf32>
    %v1082 = stablehlo.multiply %v1079, %v1081 : tensor<10xf32>
    %v1083 = stablehlo.add %v1080, %v1082 : tensor<10xf32>
    return %v220, %v260, %v303, %v343, %v386, %v426, %v469, %v509, %v552, %v592, %v635, %v675, %v718, %v758, %v801, %v841, %v879, %v918, %v956, %v995, %v1033, %v1072, %v225, %v265, %v308, %v348, %v391, %v431, %v474, %v514, %v557, %v597, %v640, %v680, %v723, %v763, %v806, %v846, %v884, %v923, %v961, %v1000, %v1038, %v1077, %v231, %v271, %v314, %v354, %v397, %v437, %v480, %v520, %v563, %v603, %v646, %v686, %v729, %v769, %v812, %v852, %v890, %v929, %v967, %v1006, %v1044, %v1083, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
