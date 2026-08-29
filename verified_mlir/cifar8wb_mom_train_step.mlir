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
    %v245 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v246 = stablehlo.multiply %v245, %W1v : tensor<16x3x3x3xf32>
    %v247 = stablehlo.add %v246, %v244 : tensor<16x3x3x3xf32>
    %v248 = stablehlo.multiply %v245, %v247 : tensor<16x3x3x3xf32>
    %v249 = stablehlo.add %v248, %v244 : tensor<16x3x3x3xf32>
    %v250 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v251 = stablehlo.multiply %v250, %v249 : tensor<16x3x3x3xf32>
    %v252 = stablehlo.subtract %W1, %v251 : tensor<16x3x3x3xf32>
    %v253 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v254 = stablehlo.multiply %v253, %W1v : tensor<16x3x3x3xf32>
    %v255 = stablehlo.add %v254, %v244 : tensor<16x3x3x3xf32>
    %v256 = stablehlo.reshape %v238 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v257 = stablehlo.constant dense<0.0> : tensor<f32>
    %v258 = stablehlo.reduce(%v256 init: %v257) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v259 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v260 = stablehlo.multiply %v259, %cb1v : tensor<16xf32>
    %v261 = stablehlo.add %v260, %v258 : tensor<16xf32>
    %v262 = stablehlo.multiply %v259, %v261 : tensor<16xf32>
    %v263 = stablehlo.add %v262, %v258 : tensor<16xf32>
    %v264 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v265 = stablehlo.multiply %v264, %v263 : tensor<16xf32>
    %v266 = stablehlo.subtract %cb1, %v265 : tensor<16xf32>
    %v267 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v268 = stablehlo.multiply %v267, %cb1v : tensor<16xf32>
    %v269 = stablehlo.add %v268, %v258 : tensor<16xf32>
    %v270 = stablehlo.reshape %v8 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v271 = stablehlo.reshape %v227 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v272 = stablehlo.transpose %v270, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v273 = stablehlo.transpose %v271, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v274 = stablehlo.convolution(%v272, %v273)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x16x3x3xf32>
    %v275 = stablehlo.transpose %v274, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v276 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v277 = stablehlo.multiply %v276, %W2v : tensor<16x16x3x3xf32>
    %v278 = stablehlo.add %v277, %v275 : tensor<16x16x3x3xf32>
    %v279 = stablehlo.multiply %v276, %v278 : tensor<16x16x3x3xf32>
    %v280 = stablehlo.add %v279, %v275 : tensor<16x16x3x3xf32>
    %v281 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v282 = stablehlo.multiply %v281, %v280 : tensor<16x16x3x3xf32>
    %v283 = stablehlo.subtract %W2, %v282 : tensor<16x16x3x3xf32>
    %v284 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v285 = stablehlo.multiply %v284, %W2v : tensor<16x16x3x3xf32>
    %v286 = stablehlo.add %v285, %v275 : tensor<16x16x3x3xf32>
    %v287 = stablehlo.reshape %v227 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v288 = stablehlo.constant dense<0.0> : tensor<f32>
    %v289 = stablehlo.reduce(%v287 init: %v288) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v290 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v291 = stablehlo.multiply %v290, %cb2v : tensor<16xf32>
    %v292 = stablehlo.add %v291, %v289 : tensor<16xf32>
    %v293 = stablehlo.multiply %v290, %v292 : tensor<16xf32>
    %v294 = stablehlo.add %v293, %v289 : tensor<16xf32>
    %v295 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v296 = stablehlo.multiply %v295, %v294 : tensor<16xf32>
    %v297 = stablehlo.subtract %cb2, %v296 : tensor<16xf32>
    %v298 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v299 = stablehlo.multiply %v298, %cb2v : tensor<16xf32>
    %v300 = stablehlo.add %v299, %v289 : tensor<16xf32>
    %v301 = stablehlo.reshape %v21 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v302 = stablehlo.reshape %v211 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v303 = stablehlo.transpose %v301, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v304 = stablehlo.transpose %v302, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v305 = stablehlo.convolution(%v303, %v304)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v306 = stablehlo.transpose %v305, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v307 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v308 = stablehlo.multiply %v307, %W3v : tensor<16x16x3x3xf32>
    %v309 = stablehlo.add %v308, %v306 : tensor<16x16x3x3xf32>
    %v310 = stablehlo.multiply %v307, %v309 : tensor<16x16x3x3xf32>
    %v311 = stablehlo.add %v310, %v306 : tensor<16x16x3x3xf32>
    %v312 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v313 = stablehlo.multiply %v312, %v311 : tensor<16x16x3x3xf32>
    %v314 = stablehlo.subtract %W3, %v313 : tensor<16x16x3x3xf32>
    %v315 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v316 = stablehlo.multiply %v315, %W3v : tensor<16x16x3x3xf32>
    %v317 = stablehlo.add %v316, %v306 : tensor<16x16x3x3xf32>
    %v318 = stablehlo.reshape %v211 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v319 = stablehlo.constant dense<0.0> : tensor<f32>
    %v320 = stablehlo.reduce(%v318 init: %v319) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v321 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v322 = stablehlo.multiply %v321, %cb3v : tensor<16xf32>
    %v323 = stablehlo.add %v322, %v320 : tensor<16xf32>
    %v324 = stablehlo.multiply %v321, %v323 : tensor<16xf32>
    %v325 = stablehlo.add %v324, %v320 : tensor<16xf32>
    %v326 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v327 = stablehlo.multiply %v326, %v325 : tensor<16xf32>
    %v328 = stablehlo.subtract %cb3, %v327 : tensor<16xf32>
    %v329 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v330 = stablehlo.multiply %v329, %cb3v : tensor<16xf32>
    %v331 = stablehlo.add %v330, %v320 : tensor<16xf32>
    %v332 = stablehlo.reshape %v30 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v333 = stablehlo.reshape %v200 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v334 = stablehlo.transpose %v332, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v335 = stablehlo.transpose %v333, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v336 = stablehlo.convolution(%v334, %v335)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v337 = stablehlo.transpose %v336, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v338 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v339 = stablehlo.multiply %v338, %W4v : tensor<16x16x3x3xf32>
    %v340 = stablehlo.add %v339, %v337 : tensor<16x16x3x3xf32>
    %v341 = stablehlo.multiply %v338, %v340 : tensor<16x16x3x3xf32>
    %v342 = stablehlo.add %v341, %v337 : tensor<16x16x3x3xf32>
    %v343 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v344 = stablehlo.multiply %v343, %v342 : tensor<16x16x3x3xf32>
    %v345 = stablehlo.subtract %W4, %v344 : tensor<16x16x3x3xf32>
    %v346 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v347 = stablehlo.multiply %v346, %W4v : tensor<16x16x3x3xf32>
    %v348 = stablehlo.add %v347, %v337 : tensor<16x16x3x3xf32>
    %v349 = stablehlo.reshape %v200 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v350 = stablehlo.constant dense<0.0> : tensor<f32>
    %v351 = stablehlo.reduce(%v349 init: %v350) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v352 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v353 = stablehlo.multiply %v352, %cb4v : tensor<16xf32>
    %v354 = stablehlo.add %v353, %v351 : tensor<16xf32>
    %v355 = stablehlo.multiply %v352, %v354 : tensor<16xf32>
    %v356 = stablehlo.add %v355, %v351 : tensor<16xf32>
    %v357 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v358 = stablehlo.multiply %v357, %v356 : tensor<16xf32>
    %v359 = stablehlo.subtract %cb4, %v358 : tensor<16xf32>
    %v360 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v361 = stablehlo.multiply %v360, %cb4v : tensor<16xf32>
    %v362 = stablehlo.add %v361, %v351 : tensor<16xf32>
    %v363 = stablehlo.reshape %v43 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v364 = stablehlo.reshape %v184 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v365 = stablehlo.transpose %v363, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v366 = stablehlo.transpose %v364, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v367 = stablehlo.convolution(%v365, %v366)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<16x32x3x3xf32>
    %v368 = stablehlo.transpose %v367, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v369 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v370 = stablehlo.multiply %v369, %W5v : tensor<32x16x3x3xf32>
    %v371 = stablehlo.add %v370, %v368 : tensor<32x16x3x3xf32>
    %v372 = stablehlo.multiply %v369, %v371 : tensor<32x16x3x3xf32>
    %v373 = stablehlo.add %v372, %v368 : tensor<32x16x3x3xf32>
    %v374 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v375 = stablehlo.multiply %v374, %v373 : tensor<32x16x3x3xf32>
    %v376 = stablehlo.subtract %W5, %v375 : tensor<32x16x3x3xf32>
    %v377 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v378 = stablehlo.multiply %v377, %W5v : tensor<32x16x3x3xf32>
    %v379 = stablehlo.add %v378, %v368 : tensor<32x16x3x3xf32>
    %v380 = stablehlo.reshape %v184 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v381 = stablehlo.constant dense<0.0> : tensor<f32>
    %v382 = stablehlo.reduce(%v380 init: %v381) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v383 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v384 = stablehlo.multiply %v383, %cb5v : tensor<32xf32>
    %v385 = stablehlo.add %v384, %v382 : tensor<32xf32>
    %v386 = stablehlo.multiply %v383, %v385 : tensor<32xf32>
    %v387 = stablehlo.add %v386, %v382 : tensor<32xf32>
    %v388 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v389 = stablehlo.multiply %v388, %v387 : tensor<32xf32>
    %v390 = stablehlo.subtract %cb5, %v389 : tensor<32xf32>
    %v391 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v392 = stablehlo.multiply %v391, %cb5v : tensor<32xf32>
    %v393 = stablehlo.add %v392, %v382 : tensor<32xf32>
    %v394 = stablehlo.reshape %v52 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v395 = stablehlo.reshape %v173 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v396 = stablehlo.transpose %v394, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v397 = stablehlo.transpose %v395, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v398 = stablehlo.convolution(%v396, %v397)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<32x32x3x3xf32>
    %v399 = stablehlo.transpose %v398, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v400 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v401 = stablehlo.multiply %v400, %W6v : tensor<32x32x3x3xf32>
    %v402 = stablehlo.add %v401, %v399 : tensor<32x32x3x3xf32>
    %v403 = stablehlo.multiply %v400, %v402 : tensor<32x32x3x3xf32>
    %v404 = stablehlo.add %v403, %v399 : tensor<32x32x3x3xf32>
    %v405 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v406 = stablehlo.multiply %v405, %v404 : tensor<32x32x3x3xf32>
    %v407 = stablehlo.subtract %W6, %v406 : tensor<32x32x3x3xf32>
    %v408 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v409 = stablehlo.multiply %v408, %W6v : tensor<32x32x3x3xf32>
    %v410 = stablehlo.add %v409, %v399 : tensor<32x32x3x3xf32>
    %v411 = stablehlo.reshape %v173 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v412 = stablehlo.constant dense<0.0> : tensor<f32>
    %v413 = stablehlo.reduce(%v411 init: %v412) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v414 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v415 = stablehlo.multiply %v414, %cb6v : tensor<32xf32>
    %v416 = stablehlo.add %v415, %v413 : tensor<32xf32>
    %v417 = stablehlo.multiply %v414, %v416 : tensor<32xf32>
    %v418 = stablehlo.add %v417, %v413 : tensor<32xf32>
    %v419 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v420 = stablehlo.multiply %v419, %v418 : tensor<32xf32>
    %v421 = stablehlo.subtract %cb6, %v420 : tensor<32xf32>
    %v422 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v423 = stablehlo.multiply %v422, %cb6v : tensor<32xf32>
    %v424 = stablehlo.add %v423, %v413 : tensor<32xf32>
    %v425 = stablehlo.reshape %v65 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v426 = stablehlo.reshape %v157 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v427 = stablehlo.transpose %v425, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v428 = stablehlo.transpose %v426, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v429 = stablehlo.convolution(%v427, %v428)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v430 = stablehlo.transpose %v429, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v431 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v432 = stablehlo.multiply %v431, %W7v : tensor<32x32x3x3xf32>
    %v433 = stablehlo.add %v432, %v430 : tensor<32x32x3x3xf32>
    %v434 = stablehlo.multiply %v431, %v433 : tensor<32x32x3x3xf32>
    %v435 = stablehlo.add %v434, %v430 : tensor<32x32x3x3xf32>
    %v436 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v437 = stablehlo.multiply %v436, %v435 : tensor<32x32x3x3xf32>
    %v438 = stablehlo.subtract %W7, %v437 : tensor<32x32x3x3xf32>
    %v439 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v440 = stablehlo.multiply %v439, %W7v : tensor<32x32x3x3xf32>
    %v441 = stablehlo.add %v440, %v430 : tensor<32x32x3x3xf32>
    %v442 = stablehlo.reshape %v157 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v443 = stablehlo.constant dense<0.0> : tensor<f32>
    %v444 = stablehlo.reduce(%v442 init: %v443) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v445 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v446 = stablehlo.multiply %v445, %cb7v : tensor<32xf32>
    %v447 = stablehlo.add %v446, %v444 : tensor<32xf32>
    %v448 = stablehlo.multiply %v445, %v447 : tensor<32xf32>
    %v449 = stablehlo.add %v448, %v444 : tensor<32xf32>
    %v450 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v451 = stablehlo.multiply %v450, %v449 : tensor<32xf32>
    %v452 = stablehlo.subtract %cb7, %v451 : tensor<32xf32>
    %v453 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v454 = stablehlo.multiply %v453, %cb7v : tensor<32xf32>
    %v455 = stablehlo.add %v454, %v444 : tensor<32xf32>
    %v456 = stablehlo.reshape %v74 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v457 = stablehlo.reshape %v146 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v458 = stablehlo.transpose %v456, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v459 = stablehlo.transpose %v457, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v460 = stablehlo.convolution(%v458, %v459)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v461 = stablehlo.transpose %v460, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v462 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v463 = stablehlo.multiply %v462, %W8v : tensor<32x32x3x3xf32>
    %v464 = stablehlo.add %v463, %v461 : tensor<32x32x3x3xf32>
    %v465 = stablehlo.multiply %v462, %v464 : tensor<32x32x3x3xf32>
    %v466 = stablehlo.add %v465, %v461 : tensor<32x32x3x3xf32>
    %v467 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v468 = stablehlo.multiply %v467, %v466 : tensor<32x32x3x3xf32>
    %v469 = stablehlo.subtract %W8, %v468 : tensor<32x32x3x3xf32>
    %v470 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v471 = stablehlo.multiply %v470, %W8v : tensor<32x32x3x3xf32>
    %v472 = stablehlo.add %v471, %v461 : tensor<32x32x3x3xf32>
    %v473 = stablehlo.reshape %v146 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v474 = stablehlo.constant dense<0.0> : tensor<f32>
    %v475 = stablehlo.reduce(%v473 init: %v474) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v476 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v477 = stablehlo.multiply %v476, %cb8v : tensor<32xf32>
    %v478 = stablehlo.add %v477, %v475 : tensor<32xf32>
    %v479 = stablehlo.multiply %v476, %v478 : tensor<32xf32>
    %v480 = stablehlo.add %v479, %v475 : tensor<32xf32>
    %v481 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v482 = stablehlo.multiply %v481, %v480 : tensor<32xf32>
    %v483 = stablehlo.subtract %cb8, %v482 : tensor<32xf32>
    %v484 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v485 = stablehlo.multiply %v484, %cb8v : tensor<32xf32>
    %v486 = stablehlo.add %v485, %v475 : tensor<32xf32>
    %v487 = stablehlo.dot_general %v87, %v132, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x512xf32>) -> tensor<128x512xf32>
    %v488 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v489 = stablehlo.multiply %v488, %W9v : tensor<128x512xf32>
    %v490 = stablehlo.add %v489, %v487 : tensor<128x512xf32>
    %v491 = stablehlo.multiply %v488, %v490 : tensor<128x512xf32>
    %v492 = stablehlo.add %v491, %v487 : tensor<128x512xf32>
    %v493 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v494 = stablehlo.multiply %v493, %v492 : tensor<128x512xf32>
    %v495 = stablehlo.subtract %W9, %v494 : tensor<128x512xf32>
    %v496 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v497 = stablehlo.multiply %v496, %W9v : tensor<128x512xf32>
    %v498 = stablehlo.add %v497, %v487 : tensor<128x512xf32>
    %v499 = stablehlo.constant dense<0.0> : tensor<f32>
    %v500 = stablehlo.reduce(%v132 init: %v499) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v501 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v502 = stablehlo.multiply %v501, %b9v : tensor<512xf32>
    %v503 = stablehlo.add %v502, %v500 : tensor<512xf32>
    %v504 = stablehlo.multiply %v501, %v503 : tensor<512xf32>
    %v505 = stablehlo.add %v504, %v500 : tensor<512xf32>
    %v506 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v507 = stablehlo.multiply %v506, %v505 : tensor<512xf32>
    %v508 = stablehlo.subtract %b9, %v507 : tensor<512xf32>
    %v509 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v510 = stablehlo.multiply %v509, %b9v : tensor<512xf32>
    %v511 = stablehlo.add %v510, %v500 : tensor<512xf32>
    %v512 = stablehlo.dot_general %v94, %v123, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<512x512xf32>
    %v513 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v514 = stablehlo.multiply %v513, %Wav : tensor<512x512xf32>
    %v515 = stablehlo.add %v514, %v512 : tensor<512x512xf32>
    %v516 = stablehlo.multiply %v513, %v515 : tensor<512x512xf32>
    %v517 = stablehlo.add %v516, %v512 : tensor<512x512xf32>
    %v518 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v519 = stablehlo.multiply %v518, %v517 : tensor<512x512xf32>
    %v520 = stablehlo.subtract %Wa, %v519 : tensor<512x512xf32>
    %v521 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v522 = stablehlo.multiply %v521, %Wav : tensor<512x512xf32>
    %v523 = stablehlo.add %v522, %v512 : tensor<512x512xf32>
    %v524 = stablehlo.constant dense<0.0> : tensor<f32>
    %v525 = stablehlo.reduce(%v123 init: %v524) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v526 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v527 = stablehlo.multiply %v526, %bav : tensor<512xf32>
    %v528 = stablehlo.add %v527, %v525 : tensor<512xf32>
    %v529 = stablehlo.multiply %v526, %v528 : tensor<512xf32>
    %v530 = stablehlo.add %v529, %v525 : tensor<512xf32>
    %v531 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v532 = stablehlo.multiply %v531, %v530 : tensor<512xf32>
    %v533 = stablehlo.subtract %ba, %v532 : tensor<512xf32>
    %v534 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v535 = stablehlo.multiply %v534, %bav : tensor<512xf32>
    %v536 = stablehlo.add %v535, %v525 : tensor<512xf32>
    %v537 = stablehlo.dot_general %v101, %v114, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x10xf32>) -> tensor<512x10xf32>
    %v538 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v539 = stablehlo.multiply %v538, %Wbv : tensor<512x10xf32>
    %v540 = stablehlo.add %v539, %v537 : tensor<512x10xf32>
    %v541 = stablehlo.multiply %v538, %v540 : tensor<512x10xf32>
    %v542 = stablehlo.add %v541, %v537 : tensor<512x10xf32>
    %v543 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v544 = stablehlo.multiply %v543, %v542 : tensor<512x10xf32>
    %v545 = stablehlo.subtract %Wb, %v544 : tensor<512x10xf32>
    %v546 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v547 = stablehlo.multiply %v546, %Wbv : tensor<512x10xf32>
    %v548 = stablehlo.add %v547, %v537 : tensor<512x10xf32>
    %v549 = stablehlo.constant dense<0.0> : tensor<f32>
    %v550 = stablehlo.reduce(%v114 init: %v549) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v551 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v552 = stablehlo.multiply %v551, %bbv : tensor<10xf32>
    %v553 = stablehlo.add %v552, %v550 : tensor<10xf32>
    %v554 = stablehlo.multiply %v551, %v553 : tensor<10xf32>
    %v555 = stablehlo.add %v554, %v550 : tensor<10xf32>
    %v556 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v557 = stablehlo.multiply %v556, %v555 : tensor<10xf32>
    %v558 = stablehlo.subtract %bb, %v557 : tensor<10xf32>
    %v559 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v560 = stablehlo.multiply %v559, %bbv : tensor<10xf32>
    %v561 = stablehlo.add %v560, %v550 : tensor<10xf32>
    return %v252, %v266, %v283, %v297, %v314, %v328, %v345, %v359, %v376, %v390, %v407, %v421, %v438, %v452, %v469, %v483, %v495, %v508, %v520, %v533, %v545, %v558, %W1m, %cb1m, %W2m, %cb2m, %W3m, %cb3m, %W4m, %cb4m, %W5m, %cb5m, %W6m, %cb6m, %W7m, %cb7m, %W8m, %cb8m, %W9m, %b9m, %Wam, %bam, %Wbm, %bbm, %v255, %v269, %v286, %v300, %v317, %v331, %v348, %v362, %v379, %v393, %v410, %v424, %v441, %v455, %v472, %v486, %v498, %v511, %v523, %v536, %v548, %v561, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
