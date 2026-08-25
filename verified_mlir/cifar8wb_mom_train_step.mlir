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
    %v195 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v196 = stablehlo.multiply %v195, %W1v : tensor<16x3x3x3xf32>
    %v197 = stablehlo.add %v196, %v194 : tensor<16x3x3x3xf32>
    %v198 = stablehlo.multiply %v195, %v197 : tensor<16x3x3x3xf32>
    %v199 = stablehlo.add %v198, %v194 : tensor<16x3x3x3xf32>
    %v200 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v201 = stablehlo.multiply %v200, %v199 : tensor<16x3x3x3xf32>
    %v202 = stablehlo.subtract %W1, %v201 : tensor<16x3x3x3xf32>
    %v203 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v204 = stablehlo.multiply %v203, %W1v : tensor<16x3x3x3xf32>
    %v205 = stablehlo.add %v204, %v194 : tensor<16x3x3x3xf32>
    %v206 = stablehlo.reshape %v188 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v207 = stablehlo.constant dense<0.0> : tensor<f32>
    %v208 = stablehlo.reduce(%v206 init: %v207) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v209 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v210 = stablehlo.multiply %v209, %cb1v : tensor<16xf32>
    %v211 = stablehlo.add %v210, %v208 : tensor<16xf32>
    %v212 = stablehlo.multiply %v209, %v211 : tensor<16xf32>
    %v213 = stablehlo.add %v212, %v208 : tensor<16xf32>
    %v214 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v215 = stablehlo.multiply %v214, %v213 : tensor<16xf32>
    %v216 = stablehlo.subtract %cb1, %v215 : tensor<16xf32>
    %v217 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v218 = stablehlo.multiply %v217, %cb1v : tensor<16xf32>
    %v219 = stablehlo.add %v218, %v208 : tensor<16xf32>
    %v220 = stablehlo.reshape %v6 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v221 = stablehlo.reshape %v180 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v222 = stablehlo.transpose %v220, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v223 = stablehlo.transpose %v221, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v224 = stablehlo.convolution(%v222, %v223)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x16x3x3xf32>
    %v225 = stablehlo.transpose %v224, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v226 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v227 = stablehlo.multiply %v226, %W2v : tensor<16x16x3x3xf32>
    %v228 = stablehlo.add %v227, %v225 : tensor<16x16x3x3xf32>
    %v229 = stablehlo.multiply %v226, %v228 : tensor<16x16x3x3xf32>
    %v230 = stablehlo.add %v229, %v225 : tensor<16x16x3x3xf32>
    %v231 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v232 = stablehlo.multiply %v231, %v230 : tensor<16x16x3x3xf32>
    %v233 = stablehlo.subtract %W2, %v232 : tensor<16x16x3x3xf32>
    %v234 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v235 = stablehlo.multiply %v234, %W2v : tensor<16x16x3x3xf32>
    %v236 = stablehlo.add %v235, %v225 : tensor<16x16x3x3xf32>
    %v237 = stablehlo.reshape %v180 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v238 = stablehlo.constant dense<0.0> : tensor<f32>
    %v239 = stablehlo.reduce(%v237 init: %v238) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v240 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v241 = stablehlo.multiply %v240, %cb2v : tensor<16xf32>
    %v242 = stablehlo.add %v241, %v239 : tensor<16xf32>
    %v243 = stablehlo.multiply %v240, %v242 : tensor<16xf32>
    %v244 = stablehlo.add %v243, %v239 : tensor<16xf32>
    %v245 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v246 = stablehlo.multiply %v245, %v244 : tensor<16xf32>
    %v247 = stablehlo.subtract %cb2, %v246 : tensor<16xf32>
    %v248 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v249 = stablehlo.multiply %v248, %cb2v : tensor<16xf32>
    %v250 = stablehlo.add %v249, %v239 : tensor<16xf32>
    %v251 = stablehlo.reshape %v17 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v252 = stablehlo.reshape %v167 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v253 = stablehlo.transpose %v251, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v254 = stablehlo.transpose %v252, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v255 = stablehlo.convolution(%v253, %v254)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v256 = stablehlo.transpose %v255, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v257 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v258 = stablehlo.multiply %v257, %W3v : tensor<16x16x3x3xf32>
    %v259 = stablehlo.add %v258, %v256 : tensor<16x16x3x3xf32>
    %v260 = stablehlo.multiply %v257, %v259 : tensor<16x16x3x3xf32>
    %v261 = stablehlo.add %v260, %v256 : tensor<16x16x3x3xf32>
    %v262 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v263 = stablehlo.multiply %v262, %v261 : tensor<16x16x3x3xf32>
    %v264 = stablehlo.subtract %W3, %v263 : tensor<16x16x3x3xf32>
    %v265 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v266 = stablehlo.multiply %v265, %W3v : tensor<16x16x3x3xf32>
    %v267 = stablehlo.add %v266, %v256 : tensor<16x16x3x3xf32>
    %v268 = stablehlo.reshape %v167 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v269 = stablehlo.constant dense<0.0> : tensor<f32>
    %v270 = stablehlo.reduce(%v268 init: %v269) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v271 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v272 = stablehlo.multiply %v271, %cb3v : tensor<16xf32>
    %v273 = stablehlo.add %v272, %v270 : tensor<16xf32>
    %v274 = stablehlo.multiply %v271, %v273 : tensor<16xf32>
    %v275 = stablehlo.add %v274, %v270 : tensor<16xf32>
    %v276 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v277 = stablehlo.multiply %v276, %v275 : tensor<16xf32>
    %v278 = stablehlo.subtract %cb3, %v277 : tensor<16xf32>
    %v279 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v280 = stablehlo.multiply %v279, %cb3v : tensor<16xf32>
    %v281 = stablehlo.add %v280, %v270 : tensor<16xf32>
    %v282 = stablehlo.reshape %v24 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v283 = stablehlo.reshape %v159 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v284 = stablehlo.transpose %v282, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v285 = stablehlo.transpose %v283, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v286 = stablehlo.convolution(%v284, %v285)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v287 = stablehlo.transpose %v286, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v288 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v289 = stablehlo.multiply %v288, %W4v : tensor<16x16x3x3xf32>
    %v290 = stablehlo.add %v289, %v287 : tensor<16x16x3x3xf32>
    %v291 = stablehlo.multiply %v288, %v290 : tensor<16x16x3x3xf32>
    %v292 = stablehlo.add %v291, %v287 : tensor<16x16x3x3xf32>
    %v293 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v294 = stablehlo.multiply %v293, %v292 : tensor<16x16x3x3xf32>
    %v295 = stablehlo.subtract %W4, %v294 : tensor<16x16x3x3xf32>
    %v296 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v297 = stablehlo.multiply %v296, %W4v : tensor<16x16x3x3xf32>
    %v298 = stablehlo.add %v297, %v287 : tensor<16x16x3x3xf32>
    %v299 = stablehlo.reshape %v159 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v300 = stablehlo.constant dense<0.0> : tensor<f32>
    %v301 = stablehlo.reduce(%v299 init: %v300) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v302 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v303 = stablehlo.multiply %v302, %cb4v : tensor<16xf32>
    %v304 = stablehlo.add %v303, %v301 : tensor<16xf32>
    %v305 = stablehlo.multiply %v302, %v304 : tensor<16xf32>
    %v306 = stablehlo.add %v305, %v301 : tensor<16xf32>
    %v307 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v308 = stablehlo.multiply %v307, %v306 : tensor<16xf32>
    %v309 = stablehlo.subtract %cb4, %v308 : tensor<16xf32>
    %v310 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v311 = stablehlo.multiply %v310, %cb4v : tensor<16xf32>
    %v312 = stablehlo.add %v311, %v301 : tensor<16xf32>
    %v313 = stablehlo.reshape %v35 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v314 = stablehlo.reshape %v146 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v315 = stablehlo.transpose %v313, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v316 = stablehlo.transpose %v314, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v317 = stablehlo.convolution(%v315, %v316)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<16x32x3x3xf32>
    %v318 = stablehlo.transpose %v317, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v319 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v320 = stablehlo.multiply %v319, %W5v : tensor<32x16x3x3xf32>
    %v321 = stablehlo.add %v320, %v318 : tensor<32x16x3x3xf32>
    %v322 = stablehlo.multiply %v319, %v321 : tensor<32x16x3x3xf32>
    %v323 = stablehlo.add %v322, %v318 : tensor<32x16x3x3xf32>
    %v324 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v325 = stablehlo.multiply %v324, %v323 : tensor<32x16x3x3xf32>
    %v326 = stablehlo.subtract %W5, %v325 : tensor<32x16x3x3xf32>
    %v327 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v328 = stablehlo.multiply %v327, %W5v : tensor<32x16x3x3xf32>
    %v329 = stablehlo.add %v328, %v318 : tensor<32x16x3x3xf32>
    %v330 = stablehlo.reshape %v146 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v331 = stablehlo.constant dense<0.0> : tensor<f32>
    %v332 = stablehlo.reduce(%v330 init: %v331) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v333 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v334 = stablehlo.multiply %v333, %cb5v : tensor<32xf32>
    %v335 = stablehlo.add %v334, %v332 : tensor<32xf32>
    %v336 = stablehlo.multiply %v333, %v335 : tensor<32xf32>
    %v337 = stablehlo.add %v336, %v332 : tensor<32xf32>
    %v338 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v339 = stablehlo.multiply %v338, %v337 : tensor<32xf32>
    %v340 = stablehlo.subtract %cb5, %v339 : tensor<32xf32>
    %v341 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v342 = stablehlo.multiply %v341, %cb5v : tensor<32xf32>
    %v343 = stablehlo.add %v342, %v332 : tensor<32xf32>
    %v344 = stablehlo.reshape %v42 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v345 = stablehlo.reshape %v138 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v346 = stablehlo.transpose %v344, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v347 = stablehlo.transpose %v345, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v348 = stablehlo.convolution(%v346, %v347)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<32x32x3x3xf32>
    %v349 = stablehlo.transpose %v348, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v350 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v351 = stablehlo.multiply %v350, %W6v : tensor<32x32x3x3xf32>
    %v352 = stablehlo.add %v351, %v349 : tensor<32x32x3x3xf32>
    %v353 = stablehlo.multiply %v350, %v352 : tensor<32x32x3x3xf32>
    %v354 = stablehlo.add %v353, %v349 : tensor<32x32x3x3xf32>
    %v355 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v356 = stablehlo.multiply %v355, %v354 : tensor<32x32x3x3xf32>
    %v357 = stablehlo.subtract %W6, %v356 : tensor<32x32x3x3xf32>
    %v358 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v359 = stablehlo.multiply %v358, %W6v : tensor<32x32x3x3xf32>
    %v360 = stablehlo.add %v359, %v349 : tensor<32x32x3x3xf32>
    %v361 = stablehlo.reshape %v138 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v362 = stablehlo.constant dense<0.0> : tensor<f32>
    %v363 = stablehlo.reduce(%v361 init: %v362) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v364 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v365 = stablehlo.multiply %v364, %cb6v : tensor<32xf32>
    %v366 = stablehlo.add %v365, %v363 : tensor<32xf32>
    %v367 = stablehlo.multiply %v364, %v366 : tensor<32xf32>
    %v368 = stablehlo.add %v367, %v363 : tensor<32xf32>
    %v369 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v370 = stablehlo.multiply %v369, %v368 : tensor<32xf32>
    %v371 = stablehlo.subtract %cb6, %v370 : tensor<32xf32>
    %v372 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v373 = stablehlo.multiply %v372, %cb6v : tensor<32xf32>
    %v374 = stablehlo.add %v373, %v363 : tensor<32xf32>
    %v375 = stablehlo.reshape %v53 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v376 = stablehlo.reshape %v125 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v377 = stablehlo.transpose %v375, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v378 = stablehlo.transpose %v376, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v379 = stablehlo.convolution(%v377, %v378)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v380 = stablehlo.transpose %v379, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v381 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v382 = stablehlo.multiply %v381, %W7v : tensor<32x32x3x3xf32>
    %v383 = stablehlo.add %v382, %v380 : tensor<32x32x3x3xf32>
    %v384 = stablehlo.multiply %v381, %v383 : tensor<32x32x3x3xf32>
    %v385 = stablehlo.add %v384, %v380 : tensor<32x32x3x3xf32>
    %v386 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v387 = stablehlo.multiply %v386, %v385 : tensor<32x32x3x3xf32>
    %v388 = stablehlo.subtract %W7, %v387 : tensor<32x32x3x3xf32>
    %v389 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v390 = stablehlo.multiply %v389, %W7v : tensor<32x32x3x3xf32>
    %v391 = stablehlo.add %v390, %v380 : tensor<32x32x3x3xf32>
    %v392 = stablehlo.reshape %v125 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v393 = stablehlo.constant dense<0.0> : tensor<f32>
    %v394 = stablehlo.reduce(%v392 init: %v393) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v395 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v396 = stablehlo.multiply %v395, %cb7v : tensor<32xf32>
    %v397 = stablehlo.add %v396, %v394 : tensor<32xf32>
    %v398 = stablehlo.multiply %v395, %v397 : tensor<32xf32>
    %v399 = stablehlo.add %v398, %v394 : tensor<32xf32>
    %v400 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v401 = stablehlo.multiply %v400, %v399 : tensor<32xf32>
    %v402 = stablehlo.subtract %cb7, %v401 : tensor<32xf32>
    %v403 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v404 = stablehlo.multiply %v403, %cb7v : tensor<32xf32>
    %v405 = stablehlo.add %v404, %v394 : tensor<32xf32>
    %v406 = stablehlo.reshape %v60 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v407 = stablehlo.reshape %v117 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v408 = stablehlo.transpose %v406, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v409 = stablehlo.transpose %v407, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v410 = stablehlo.convolution(%v408, %v409)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v411 = stablehlo.transpose %v410, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v412 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v413 = stablehlo.multiply %v412, %W8v : tensor<32x32x3x3xf32>
    %v414 = stablehlo.add %v413, %v411 : tensor<32x32x3x3xf32>
    %v415 = stablehlo.multiply %v412, %v414 : tensor<32x32x3x3xf32>
    %v416 = stablehlo.add %v415, %v411 : tensor<32x32x3x3xf32>
    %v417 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v418 = stablehlo.multiply %v417, %v416 : tensor<32x32x3x3xf32>
    %v419 = stablehlo.subtract %W8, %v418 : tensor<32x32x3x3xf32>
    %v420 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v421 = stablehlo.multiply %v420, %W8v : tensor<32x32x3x3xf32>
    %v422 = stablehlo.add %v421, %v411 : tensor<32x32x3x3xf32>
    %v423 = stablehlo.reshape %v117 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v424 = stablehlo.constant dense<0.0> : tensor<f32>
    %v425 = stablehlo.reduce(%v423 init: %v424) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v426 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v427 = stablehlo.multiply %v426, %cb8v : tensor<32xf32>
    %v428 = stablehlo.add %v427, %v425 : tensor<32xf32>
    %v429 = stablehlo.multiply %v426, %v428 : tensor<32xf32>
    %v430 = stablehlo.add %v429, %v425 : tensor<32xf32>
    %v431 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v432 = stablehlo.multiply %v431, %v430 : tensor<32xf32>
    %v433 = stablehlo.subtract %cb8, %v432 : tensor<32xf32>
    %v434 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v435 = stablehlo.multiply %v434, %cb8v : tensor<32xf32>
    %v436 = stablehlo.add %v435, %v425 : tensor<32xf32>
    %v437 = stablehlo.dot_general %v71, %v106, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x512xf32>) -> tensor<128x512xf32>
    %v438 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v439 = stablehlo.multiply %v438, %W9v : tensor<128x512xf32>
    %v440 = stablehlo.add %v439, %v437 : tensor<128x512xf32>
    %v441 = stablehlo.multiply %v438, %v440 : tensor<128x512xf32>
    %v442 = stablehlo.add %v441, %v437 : tensor<128x512xf32>
    %v443 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v444 = stablehlo.multiply %v443, %v442 : tensor<128x512xf32>
    %v445 = stablehlo.subtract %W9, %v444 : tensor<128x512xf32>
    %v446 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v447 = stablehlo.multiply %v446, %W9v : tensor<128x512xf32>
    %v448 = stablehlo.add %v447, %v437 : tensor<128x512xf32>
    %v449 = stablehlo.constant dense<0.0> : tensor<f32>
    %v450 = stablehlo.reduce(%v106 init: %v449) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v451 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v452 = stablehlo.multiply %v451, %b9v : tensor<512xf32>
    %v453 = stablehlo.add %v452, %v450 : tensor<512xf32>
    %v454 = stablehlo.multiply %v451, %v453 : tensor<512xf32>
    %v455 = stablehlo.add %v454, %v450 : tensor<512xf32>
    %v456 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v457 = stablehlo.multiply %v456, %v455 : tensor<512xf32>
    %v458 = stablehlo.subtract %b9, %v457 : tensor<512xf32>
    %v459 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v460 = stablehlo.multiply %v459, %b9v : tensor<512xf32>
    %v461 = stablehlo.add %v460, %v450 : tensor<512xf32>
    %v462 = stablehlo.dot_general %v76, %v100, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<512x512xf32>
    %v463 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v464 = stablehlo.multiply %v463, %Wav : tensor<512x512xf32>
    %v465 = stablehlo.add %v464, %v462 : tensor<512x512xf32>
    %v466 = stablehlo.multiply %v463, %v465 : tensor<512x512xf32>
    %v467 = stablehlo.add %v466, %v462 : tensor<512x512xf32>
    %v468 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v469 = stablehlo.multiply %v468, %v467 : tensor<512x512xf32>
    %v470 = stablehlo.subtract %Wa, %v469 : tensor<512x512xf32>
    %v471 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v472 = stablehlo.multiply %v471, %Wav : tensor<512x512xf32>
    %v473 = stablehlo.add %v472, %v462 : tensor<512x512xf32>
    %v474 = stablehlo.constant dense<0.0> : tensor<f32>
    %v475 = stablehlo.reduce(%v100 init: %v474) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v476 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v477 = stablehlo.multiply %v476, %bav : tensor<512xf32>
    %v478 = stablehlo.add %v477, %v475 : tensor<512xf32>
    %v479 = stablehlo.multiply %v476, %v478 : tensor<512xf32>
    %v480 = stablehlo.add %v479, %v475 : tensor<512xf32>
    %v481 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v482 = stablehlo.multiply %v481, %v480 : tensor<512xf32>
    %v483 = stablehlo.subtract %ba, %v482 : tensor<512xf32>
    %v484 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v485 = stablehlo.multiply %v484, %bav : tensor<512xf32>
    %v486 = stablehlo.add %v485, %v475 : tensor<512xf32>
    %v487 = stablehlo.dot_general %v81, %v94, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x10xf32>) -> tensor<512x10xf32>
    %v488 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v489 = stablehlo.multiply %v488, %Wbv : tensor<512x10xf32>
    %v490 = stablehlo.add %v489, %v487 : tensor<512x10xf32>
    %v491 = stablehlo.multiply %v488, %v490 : tensor<512x10xf32>
    %v492 = stablehlo.add %v491, %v487 : tensor<512x10xf32>
    %v493 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v494 = stablehlo.multiply %v493, %v492 : tensor<512x10xf32>
    %v495 = stablehlo.subtract %Wb, %v494 : tensor<512x10xf32>
    %v496 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v497 = stablehlo.multiply %v496, %Wbv : tensor<512x10xf32>
    %v498 = stablehlo.add %v497, %v487 : tensor<512x10xf32>
    %v499 = stablehlo.constant dense<0.0> : tensor<f32>
    %v500 = stablehlo.reduce(%v94 init: %v499) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v501 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v502 = stablehlo.multiply %v501, %bbv : tensor<10xf32>
    %v503 = stablehlo.add %v502, %v500 : tensor<10xf32>
    %v504 = stablehlo.multiply %v501, %v503 : tensor<10xf32>
    %v505 = stablehlo.add %v504, %v500 : tensor<10xf32>
    %v506 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v507 = stablehlo.multiply %v506, %v505 : tensor<10xf32>
    %v508 = stablehlo.subtract %bb, %v507 : tensor<10xf32>
    %v509 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v510 = stablehlo.multiply %v509, %bbv : tensor<10xf32>
    %v511 = stablehlo.add %v510, %v500 : tensor<10xf32>
    return %v202, %v216, %v233, %v247, %v264, %v278, %v295, %v309, %v326, %v340, %v357, %v371, %v388, %v402, %v419, %v433, %v445, %v458, %v470, %v483, %v495, %v508, %W1m, %cb1m, %W2m, %cb2m, %W3m, %cb3m, %W4m, %cb4m, %W5m, %cb5m, %W6m, %cb6m, %W7m, %cb7m, %W8m, %cb8m, %W9m, %b9m, %Wam, %bam, %Wbm, %bbm, %v205, %v219, %v236, %v250, %v267, %v281, %v298, %v312, %v329, %v343, %v360, %v374, %v391, %v405, %v422, %v436, %v448, %v461, %v473, %v486, %v498, %v511, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
