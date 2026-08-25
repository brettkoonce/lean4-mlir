module @m {
  func.func @cifar8wb_bf16sgd_train_step(%x: tensor<128x3072xf32>, %W1: tensor<16x3x3x3xf32>, %cb1: tensor<16xf32>, %W2: tensor<16x16x3x3xf32>, %cb2: tensor<16xf32>, %W3: tensor<16x16x3x3xf32>, %cb3: tensor<16xf32>, %W4: tensor<16x16x3x3xf32>, %cb4: tensor<16xf32>, %W5: tensor<32x16x3x3xf32>, %cb5: tensor<32xf32>, %W6: tensor<32x32x3x3xf32>, %cb6: tensor<32xf32>, %W7: tensor<32x32x3x3xf32>, %cb7: tensor<32xf32>, %W8: tensor<32x32x3x3xf32>, %cb8: tensor<32xf32>, %W9: tensor<128x512xf32>, %b9: tensor<512xf32>, %Wa: tensor<512x512xf32>, %ba: tensor<512xf32>, %Wb: tensor<512x10xf32>, %bb: tensor<10xf32>, %W1m: tensor<16x3x3x3xf32>, %cb1m: tensor<16xf32>, %W2m: tensor<16x16x3x3xf32>, %cb2m: tensor<16xf32>, %W3m: tensor<16x16x3x3xf32>, %cb3m: tensor<16xf32>, %W4m: tensor<16x16x3x3xf32>, %cb4m: tensor<16xf32>, %W5m: tensor<32x16x3x3xf32>, %cb5m: tensor<32xf32>, %W6m: tensor<32x32x3x3xf32>, %cb6m: tensor<32xf32>, %W7m: tensor<32x32x3x3xf32>, %cb7m: tensor<32xf32>, %W8m: tensor<32x32x3x3xf32>, %cb8m: tensor<32xf32>, %W9m: tensor<128x512xf32>, %b9m: tensor<512xf32>, %Wam: tensor<512x512xf32>, %bam: tensor<512xf32>, %Wbm: tensor<512x10xf32>, %bbm: tensor<10xf32>, %W1v: tensor<16x3x3x3xf32>, %cb1v: tensor<16xf32>, %W2v: tensor<16x16x3x3xf32>, %cb2v: tensor<16xf32>, %W3v: tensor<16x16x3x3xf32>, %cb3v: tensor<16xf32>, %W4v: tensor<16x16x3x3xf32>, %cb4v: tensor<16xf32>, %W5v: tensor<32x16x3x3xf32>, %cb5v: tensor<32xf32>, %W6v: tensor<32x32x3x3xf32>, %cb6v: tensor<32xf32>, %W7v: tensor<32x32x3x3xf32>, %cb7v: tensor<32xf32>, %W8v: tensor<32x32x3x3xf32>, %cb8v: tensor<32xf32>, %W9v: tensor<128x512xf32>, %b9v: tensor<512xf32>, %Wav: tensor<512x512xf32>, %bav: tensor<512xf32>, %Wbv: tensor<512x10xf32>, %bbv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: tensor<128x10xf32>) -> (tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>) {
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
    %v1 = stablehlo.convert %v0 : (tensor<128x3x32x32xf32>) -> tensor<128x3x32x32xbf16>
    %v2 = stablehlo.convert %W1 : (tensor<16x3x3x3xf32>) -> tensor<16x3x3x3xbf16>
    %v3 = stablehlo.convolution(%v1, %v2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x3x32x32xbf16>, tensor<16x3x3x3xbf16>) -> tensor<128x16x32x32xbf16>
    %v4 = stablehlo.convert %v3 : (tensor<128x16x32x32xbf16>) -> tensor<128x16x32x32xf32>
    %v5 = stablehlo.broadcast_in_dim %cb1, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v6 = stablehlo.add %v4, %v5 : tensor<128x16x32x32xf32>
    %v7 = stablehlo.reshape %v6 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v8 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v9 = stablehlo.maximum %v7, %v8 : tensor<128x16384xf32>
    %v10 = stablehlo.reshape %v9 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v11 = stablehlo.convert %v10 : (tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xbf16>
    %v12 = stablehlo.convert %W2 : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xbf16>
    %v13 = stablehlo.convolution(%v11, %v12)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xbf16>, tensor<16x16x3x3xbf16>) -> tensor<128x16x32x32xbf16>
    %v14 = stablehlo.convert %v13 : (tensor<128x16x32x32xbf16>) -> tensor<128x16x32x32xf32>
    %v15 = stablehlo.broadcast_in_dim %cb2, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v16 = stablehlo.add %v14, %v15 : tensor<128x16x32x32xf32>
    %v17 = stablehlo.reshape %v16 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v18 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v19 = stablehlo.maximum %v17, %v18 : tensor<128x16384xf32>
    %v20 = stablehlo.reshape %v19 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v21 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v22 = "stablehlo.reduce_window"(%v20, %v21) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16x16x16xf32>
    %v23 = stablehlo.reshape %v22 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v24 = stablehlo.reshape %v23 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v25 = stablehlo.convert %v24 : (tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xbf16>
    %v26 = stablehlo.convert %W3 : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xbf16>
    %v27 = stablehlo.convolution(%v25, %v26)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xbf16>, tensor<16x16x3x3xbf16>) -> tensor<128x16x16x16xbf16>
    %v28 = stablehlo.convert %v27 : (tensor<128x16x16x16xbf16>) -> tensor<128x16x16x16xf32>
    %v29 = stablehlo.broadcast_in_dim %cb3, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v30 = stablehlo.add %v28, %v29 : tensor<128x16x16x16xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v32 = stablehlo.constant dense<0.0> : tensor<128x4096xf32>
    %v33 = stablehlo.maximum %v31, %v32 : tensor<128x4096xf32>
    %v34 = stablehlo.reshape %v33 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v35 = stablehlo.convert %v34 : (tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xbf16>
    %v36 = stablehlo.convert %W4 : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xbf16>
    %v37 = stablehlo.convolution(%v35, %v36)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xbf16>, tensor<16x16x3x3xbf16>) -> tensor<128x16x16x16xbf16>
    %v38 = stablehlo.convert %v37 : (tensor<128x16x16x16xbf16>) -> tensor<128x16x16x16xf32>
    %v39 = stablehlo.broadcast_in_dim %cb4, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v40 = stablehlo.add %v38, %v39 : tensor<128x16x16x16xf32>
    %v41 = stablehlo.reshape %v40 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v42 = stablehlo.constant dense<0.0> : tensor<128x4096xf32>
    %v43 = stablehlo.maximum %v41, %v42 : tensor<128x4096xf32>
    %v44 = stablehlo.reshape %v43 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v45 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v46 = "stablehlo.reduce_window"(%v44, %v45) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x8x8xf32>
    %v47 = stablehlo.reshape %v46 : (tensor<128x16x8x8xf32>) -> tensor<128x1024xf32>
    %v48 = stablehlo.reshape %v47 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v49 = stablehlo.convert %v48 : (tensor<128x16x8x8xf32>) -> tensor<128x16x8x8xbf16>
    %v50 = stablehlo.convert %W5 : (tensor<32x16x3x3xf32>) -> tensor<32x16x3x3xbf16>
    %v51 = stablehlo.convolution(%v49, %v50)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x8x8xbf16>, tensor<32x16x3x3xbf16>) -> tensor<128x32x8x8xbf16>
    %v52 = stablehlo.convert %v51 : (tensor<128x32x8x8xbf16>) -> tensor<128x32x8x8xf32>
    %v53 = stablehlo.broadcast_in_dim %cb5, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v54 = stablehlo.add %v52, %v53 : tensor<128x32x8x8xf32>
    %v55 = stablehlo.reshape %v54 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v56 = stablehlo.constant dense<0.0> : tensor<128x2048xf32>
    %v57 = stablehlo.maximum %v55, %v56 : tensor<128x2048xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v59 = stablehlo.convert %v58 : (tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xbf16>
    %v60 = stablehlo.convert %W6 : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xbf16>
    %v61 = stablehlo.convolution(%v59, %v60)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xbf16>, tensor<32x32x3x3xbf16>) -> tensor<128x32x8x8xbf16>
    %v62 = stablehlo.convert %v61 : (tensor<128x32x8x8xbf16>) -> tensor<128x32x8x8xf32>
    %v63 = stablehlo.broadcast_in_dim %cb6, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v64 = stablehlo.add %v62, %v63 : tensor<128x32x8x8xf32>
    %v65 = stablehlo.reshape %v64 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v66 = stablehlo.constant dense<0.0> : tensor<128x2048xf32>
    %v67 = stablehlo.maximum %v65, %v66 : tensor<128x2048xf32>
    %v68 = stablehlo.reshape %v67 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v69 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v70 = "stablehlo.reduce_window"(%v68, %v69) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32x4x4xf32>
    %v71 = stablehlo.reshape %v70 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v72 = stablehlo.reshape %v71 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v73 = stablehlo.convert %v72 : (tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xbf16>
    %v74 = stablehlo.convert %W7 : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xbf16>
    %v75 = stablehlo.convolution(%v73, %v74)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xbf16>, tensor<32x32x3x3xbf16>) -> tensor<128x32x4x4xbf16>
    %v76 = stablehlo.convert %v75 : (tensor<128x32x4x4xbf16>) -> tensor<128x32x4x4xf32>
    %v77 = stablehlo.broadcast_in_dim %cb7, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v78 = stablehlo.add %v76, %v77 : tensor<128x32x4x4xf32>
    %v79 = stablehlo.reshape %v78 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v80 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v81 = stablehlo.maximum %v79, %v80 : tensor<128x512xf32>
    %v82 = stablehlo.reshape %v81 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v83 = stablehlo.convert %v82 : (tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xbf16>
    %v84 = stablehlo.convert %W8 : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xbf16>
    %v85 = stablehlo.convolution(%v83, %v84)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xbf16>, tensor<32x32x3x3xbf16>) -> tensor<128x32x4x4xbf16>
    %v86 = stablehlo.convert %v85 : (tensor<128x32x4x4xbf16>) -> tensor<128x32x4x4xf32>
    %v87 = stablehlo.broadcast_in_dim %cb8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v88 = stablehlo.add %v86, %v87 : tensor<128x32x4x4xf32>
    %v89 = stablehlo.reshape %v88 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v90 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v91 = stablehlo.maximum %v89, %v90 : tensor<128x512xf32>
    %v92 = stablehlo.reshape %v91 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v93 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v94 = "stablehlo.reduce_window"(%v92, %v93) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32x2x2xf32>
    %v95 = stablehlo.reshape %v94 : (tensor<128x32x2x2xf32>) -> tensor<128x128xf32>
    %v96 = stablehlo.dot_general %v95, %W9, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x512xf32>) -> tensor<128x512xf32>
    %v97 = stablehlo.broadcast_in_dim %b9, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v98 = stablehlo.add %v96, %v97 : tensor<128x512xf32>
    %v99 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v100 = stablehlo.maximum %v98, %v99 : tensor<128x512xf32>
    %v101 = stablehlo.dot_general %v100, %Wa, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %v102 = stablehlo.broadcast_in_dim %ba, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v103 = stablehlo.add %v101, %v102 : tensor<128x512xf32>
    %v104 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v105 = stablehlo.maximum %v103, %v104 : tensor<128x512xf32>
    %v106 = stablehlo.dot_general %v105, %Wb, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x10xf32>) -> tensor<128x10xf32>
    %v107 = stablehlo.broadcast_in_dim %bb, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v108 = stablehlo.add %v106, %v107 : tensor<128x10xf32>
    %v109 = stablehlo.reshape %v108 : (tensor<128x10xf32>) -> tensor<128x1x10xf32>
    %v110 = stablehlo.constant dense<0.0> : tensor<f32>
    %v111 = stablehlo.exponential %v109 : tensor<128x1x10xf32>
    %v112 = stablehlo.reduce(%v111 init: %v110) applies stablehlo.add across dimensions = [2] : (tensor<128x1x10xf32>, tensor<f32>) -> tensor<128x1xf32>
    %v113 = stablehlo.broadcast_in_dim %v112, dims = [0, 1] : (tensor<128x1xf32>) -> tensor<128x1x10xf32>
    %v114 = stablehlo.divide %v111, %v113 : tensor<128x1x10xf32>
    %v115 = stablehlo.reshape %v114 : (tensor<128x1x10xf32>) -> tensor<128x10xf32>
    %v116 = stablehlo.subtract %v115, %onehot : tensor<128x10xf32>
    %v117 = stablehlo.constant dense<0.0078125> : tensor<128x10xf32>
    %v118 = stablehlo.multiply %v116, %v117 : tensor<128x10xf32>
    // ── report-only scalar loss (NOT pretty(AST): the kit has no rank-0 loss op; it
    //    feeds no parameter, only the driver's progress line) ──
    %llog = stablehlo.log %v115 : tensor<128x10xf32>
    %ohll = stablehlo.multiply %onehot, %llog : tensor<128x10xf32>
    %csum = stablehlo.reduce(%ohll init: %lzero) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<f32>
    %cneg = stablehlo.negate %csum : tensor<f32>
    %lbf = stablehlo.constant dense<128.0> : tensor<f32>
    %loss = stablehlo.divide %cneg, %lbf : tensor<f32>
    %v119 = stablehlo.reshape %v118 : (tensor<128x10xf32>) -> tensor<128x1x10xf32>
    %v120 = stablehlo.dot_general %v119, %Wb, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x10xf32>, tensor<512x10xf32>) -> tensor<128x1x512xf32>
    %v121 = stablehlo.reshape %v120 : (tensor<128x1x512xf32>) -> tensor<128x512xf32>
    %v122 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v123 = stablehlo.compare GT, %v103, %v122 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v124 = stablehlo.select %v123, %v121, %v122 : tensor<128x512xi1>, tensor<128x512xf32>
    %v125 = stablehlo.reshape %v124 : (tensor<128x512xf32>) -> tensor<128x1x512xf32>
    %v126 = stablehlo.dot_general %v125, %Wa, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x512xf32>, tensor<512x512xf32>) -> tensor<128x1x512xf32>
    %v127 = stablehlo.reshape %v126 : (tensor<128x1x512xf32>) -> tensor<128x512xf32>
    %v128 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v129 = stablehlo.compare GT, %v98, %v128 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v130 = stablehlo.select %v129, %v127, %v128 : tensor<128x512xi1>, tensor<128x512xf32>
    %v131 = stablehlo.reshape %v130 : (tensor<128x512xf32>) -> tensor<128x1x512xf32>
    %v132 = stablehlo.dot_general %v131, %W9, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x512xf32>, tensor<128x512xf32>) -> tensor<128x1x128xf32>
    %v133 = stablehlo.reshape %v132 : (tensor<128x1x128xf32>) -> tensor<128x128xf32>
    %v134 = stablehlo.reshape %v91 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v135 = stablehlo.reshape %v133 : (tensor<128x128xf32>) -> tensor<128x32x2x2xf32>
    %v136 = stablehlo.constant dense<0.0> : tensor<f32>
    %v137 = "stablehlo.select_and_scatter"(%v134, %v135, %v136) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x4x4xf32>, tensor<128x32x2x2xf32>, tensor<f32>) -> tensor<128x32x4x4xf32>
    %v138 = stablehlo.reshape %v137 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v139 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v140 = stablehlo.compare GT, %v89, %v139 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v141 = stablehlo.select %v140, %v138, %v139 : tensor<128x512xi1>, tensor<128x512xf32>
    %v142 = stablehlo.reshape %v141 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v143 = stablehlo.reverse %W8, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v144 = stablehlo.transpose %v143, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v145 = stablehlo.convert %v142 : (tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xbf16>
    %v146 = stablehlo.convert %v144 : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xbf16>
    %v147 = stablehlo.convolution(%v145, %v146)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xbf16>, tensor<32x32x3x3xbf16>) -> tensor<128x32x4x4xbf16>
    %v148 = stablehlo.convert %v147 : (tensor<128x32x4x4xbf16>) -> tensor<128x32x4x4xf32>
    %v149 = stablehlo.reshape %v148 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v150 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v151 = stablehlo.compare GT, %v79, %v150 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v152 = stablehlo.select %v151, %v149, %v150 : tensor<128x512xi1>, tensor<128x512xf32>
    %v153 = stablehlo.reshape %v152 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v154 = stablehlo.reverse %W7, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v155 = stablehlo.transpose %v154, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v156 = stablehlo.convert %v153 : (tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xbf16>
    %v157 = stablehlo.convert %v155 : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xbf16>
    %v158 = stablehlo.convolution(%v156, %v157)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xbf16>, tensor<32x32x3x3xbf16>) -> tensor<128x32x4x4xbf16>
    %v159 = stablehlo.convert %v158 : (tensor<128x32x4x4xbf16>) -> tensor<128x32x4x4xf32>
    %v160 = stablehlo.reshape %v159 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v161 = stablehlo.reshape %v67 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v162 = stablehlo.reshape %v160 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v163 = stablehlo.constant dense<0.0> : tensor<f32>
    %v164 = "stablehlo.select_and_scatter"(%v161, %v162, %v163) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x8x8xf32>, tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32x8x8xf32>
    %v165 = stablehlo.reshape %v164 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v166 = stablehlo.constant dense<0.0> : tensor<128x2048xf32>
    %v167 = stablehlo.compare GT, %v65, %v166 : (tensor<128x2048xf32>, tensor<128x2048xf32>) -> tensor<128x2048xi1>
    %v168 = stablehlo.select %v167, %v165, %v166 : tensor<128x2048xi1>, tensor<128x2048xf32>
    %v169 = stablehlo.reshape %v168 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v170 = stablehlo.reverse %W6, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v171 = stablehlo.transpose %v170, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v172 = stablehlo.convert %v169 : (tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xbf16>
    %v173 = stablehlo.convert %v171 : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xbf16>
    %v174 = stablehlo.convolution(%v172, %v173)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xbf16>, tensor<32x32x3x3xbf16>) -> tensor<128x32x8x8xbf16>
    %v175 = stablehlo.convert %v174 : (tensor<128x32x8x8xbf16>) -> tensor<128x32x8x8xf32>
    %v176 = stablehlo.reshape %v175 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v177 = stablehlo.constant dense<0.0> : tensor<128x2048xf32>
    %v178 = stablehlo.compare GT, %v55, %v177 : (tensor<128x2048xf32>, tensor<128x2048xf32>) -> tensor<128x2048xi1>
    %v179 = stablehlo.select %v178, %v176, %v177 : tensor<128x2048xi1>, tensor<128x2048xf32>
    %v180 = stablehlo.reshape %v179 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v181 = stablehlo.reverse %W5, dims = [2, 3] : tensor<32x16x3x3xf32>
    %v182 = stablehlo.transpose %v181, dims = [1, 0, 2, 3] : (tensor<32x16x3x3xf32>) -> tensor<16x32x3x3xf32>
    %v183 = stablehlo.convert %v180 : (tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xbf16>
    %v184 = stablehlo.convert %v182 : (tensor<16x32x3x3xf32>) -> tensor<16x32x3x3xbf16>
    %v185 = stablehlo.convolution(%v183, %v184)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xbf16>, tensor<16x32x3x3xbf16>) -> tensor<128x16x8x8xbf16>
    %v186 = stablehlo.convert %v185 : (tensor<128x16x8x8xbf16>) -> tensor<128x16x8x8xf32>
    %v187 = stablehlo.reshape %v186 : (tensor<128x16x8x8xf32>) -> tensor<128x1024xf32>
    %v188 = stablehlo.reshape %v43 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v189 = stablehlo.reshape %v187 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v190 = stablehlo.constant dense<0.0> : tensor<f32>
    %v191 = "stablehlo.select_and_scatter"(%v188, %v189, %v190) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x16x16xf32>, tensor<128x16x8x8xf32>, tensor<f32>) -> tensor<128x16x16x16xf32>
    %v192 = stablehlo.reshape %v191 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v193 = stablehlo.constant dense<0.0> : tensor<128x4096xf32>
    %v194 = stablehlo.compare GT, %v41, %v193 : (tensor<128x4096xf32>, tensor<128x4096xf32>) -> tensor<128x4096xi1>
    %v195 = stablehlo.select %v194, %v192, %v193 : tensor<128x4096xi1>, tensor<128x4096xf32>
    %v196 = stablehlo.reshape %v195 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v197 = stablehlo.reverse %W4, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v198 = stablehlo.transpose %v197, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v199 = stablehlo.convert %v196 : (tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xbf16>
    %v200 = stablehlo.convert %v198 : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xbf16>
    %v201 = stablehlo.convolution(%v199, %v200)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xbf16>, tensor<16x16x3x3xbf16>) -> tensor<128x16x16x16xbf16>
    %v202 = stablehlo.convert %v201 : (tensor<128x16x16x16xbf16>) -> tensor<128x16x16x16xf32>
    %v203 = stablehlo.reshape %v202 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v204 = stablehlo.constant dense<0.0> : tensor<128x4096xf32>
    %v205 = stablehlo.compare GT, %v31, %v204 : (tensor<128x4096xf32>, tensor<128x4096xf32>) -> tensor<128x4096xi1>
    %v206 = stablehlo.select %v205, %v203, %v204 : tensor<128x4096xi1>, tensor<128x4096xf32>
    %v207 = stablehlo.reshape %v206 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v208 = stablehlo.reverse %W3, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v209 = stablehlo.transpose %v208, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v210 = stablehlo.convert %v207 : (tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xbf16>
    %v211 = stablehlo.convert %v209 : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xbf16>
    %v212 = stablehlo.convolution(%v210, %v211)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xbf16>, tensor<16x16x3x3xbf16>) -> tensor<128x16x16x16xbf16>
    %v213 = stablehlo.convert %v212 : (tensor<128x16x16x16xbf16>) -> tensor<128x16x16x16xf32>
    %v214 = stablehlo.reshape %v213 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v215 = stablehlo.reshape %v19 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v216 = stablehlo.reshape %v214 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v217 = stablehlo.constant dense<0.0> : tensor<f32>
    %v218 = "stablehlo.select_and_scatter"(%v215, %v216, %v217) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x32x32xf32>, tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x32x32xf32>
    %v219 = stablehlo.reshape %v218 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v220 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v221 = stablehlo.compare GT, %v17, %v220 : (tensor<128x16384xf32>, tensor<128x16384xf32>) -> tensor<128x16384xi1>
    %v222 = stablehlo.select %v221, %v219, %v220 : tensor<128x16384xi1>, tensor<128x16384xf32>
    %v223 = stablehlo.reshape %v222 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v224 = stablehlo.reverse %W2, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v225 = stablehlo.transpose %v224, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v226 = stablehlo.convert %v223 : (tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xbf16>
    %v227 = stablehlo.convert %v225 : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xbf16>
    %v228 = stablehlo.convolution(%v226, %v227)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xbf16>, tensor<16x16x3x3xbf16>) -> tensor<128x16x32x32xbf16>
    %v229 = stablehlo.convert %v228 : (tensor<128x16x32x32xbf16>) -> tensor<128x16x32x32xf32>
    %v230 = stablehlo.reshape %v229 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v231 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v232 = stablehlo.compare GT, %v7, %v231 : (tensor<128x16384xf32>, tensor<128x16384xf32>) -> tensor<128x16384xi1>
    %v233 = stablehlo.select %v232, %v230, %v231 : tensor<128x16384xi1>, tensor<128x16384xf32>
    %v234 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v235 = stablehlo.reshape %v233 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v236 = stablehlo.transpose %v234, dims = [1, 0, 2, 3] : (tensor<128x3x32x32xf32>) -> tensor<3x128x32x32xf32>
    %v237 = stablehlo.transpose %v235, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v238 = stablehlo.convert %v236 : (tensor<3x128x32x32xf32>) -> tensor<3x128x32x32xbf16>
    %v239 = stablehlo.convert %v237 : (tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xbf16>
    %v240 = stablehlo.convolution(%v238, %v239)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x128x32x32xbf16>, tensor<16x128x32x32xbf16>) -> tensor<3x16x3x3xbf16>
    %v241 = stablehlo.convert %v240 : (tensor<3x16x3x3xbf16>) -> tensor<3x16x3x3xf32>
    %v242 = stablehlo.transpose %v241, dims = [1, 0, 2, 3] : (tensor<3x16x3x3xf32>) -> tensor<16x3x3x3xf32>
    %v243 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v244 = stablehlo.multiply %v243, %v242 : tensor<16x3x3x3xf32>
    %v245 = stablehlo.subtract %W1, %v244 : tensor<16x3x3x3xf32>
    %v246 = stablehlo.reshape %v233 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v247 = stablehlo.constant dense<0.0> : tensor<f32>
    %v248 = stablehlo.reduce(%v246 init: %v247) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v249 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v250 = stablehlo.multiply %v249, %v248 : tensor<16xf32>
    %v251 = stablehlo.subtract %cb1, %v250 : tensor<16xf32>
    %v252 = stablehlo.reshape %v9 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v253 = stablehlo.reshape %v222 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v254 = stablehlo.transpose %v252, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v255 = stablehlo.transpose %v253, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v256 = stablehlo.convert %v254 : (tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xbf16>
    %v257 = stablehlo.convert %v255 : (tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xbf16>
    %v258 = stablehlo.convolution(%v256, %v257)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xbf16>, tensor<16x128x32x32xbf16>) -> tensor<16x16x3x3xbf16>
    %v259 = stablehlo.convert %v258 : (tensor<16x16x3x3xbf16>) -> tensor<16x16x3x3xf32>
    %v260 = stablehlo.transpose %v259, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v261 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v262 = stablehlo.multiply %v261, %v260 : tensor<16x16x3x3xf32>
    %v263 = stablehlo.subtract %W2, %v262 : tensor<16x16x3x3xf32>
    %v264 = stablehlo.reshape %v222 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v265 = stablehlo.constant dense<0.0> : tensor<f32>
    %v266 = stablehlo.reduce(%v264 init: %v265) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v267 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v268 = stablehlo.multiply %v267, %v266 : tensor<16xf32>
    %v269 = stablehlo.subtract %cb2, %v268 : tensor<16xf32>
    %v270 = stablehlo.reshape %v23 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v271 = stablehlo.reshape %v206 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v272 = stablehlo.transpose %v270, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v273 = stablehlo.transpose %v271, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v274 = stablehlo.convert %v272 : (tensor<16x128x16x16xf32>) -> tensor<16x128x16x16xbf16>
    %v275 = stablehlo.convert %v273 : (tensor<16x128x16x16xf32>) -> tensor<16x128x16x16xbf16>
    %v276 = stablehlo.convolution(%v274, %v275)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xbf16>, tensor<16x128x16x16xbf16>) -> tensor<16x16x3x3xbf16>
    %v277 = stablehlo.convert %v276 : (tensor<16x16x3x3xbf16>) -> tensor<16x16x3x3xf32>
    %v278 = stablehlo.transpose %v277, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v279 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v280 = stablehlo.multiply %v279, %v278 : tensor<16x16x3x3xf32>
    %v281 = stablehlo.subtract %W3, %v280 : tensor<16x16x3x3xf32>
    %v282 = stablehlo.reshape %v206 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v283 = stablehlo.constant dense<0.0> : tensor<f32>
    %v284 = stablehlo.reduce(%v282 init: %v283) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v285 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v286 = stablehlo.multiply %v285, %v284 : tensor<16xf32>
    %v287 = stablehlo.subtract %cb3, %v286 : tensor<16xf32>
    %v288 = stablehlo.reshape %v33 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v289 = stablehlo.reshape %v195 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v290 = stablehlo.transpose %v288, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v291 = stablehlo.transpose %v289, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v292 = stablehlo.convert %v290 : (tensor<16x128x16x16xf32>) -> tensor<16x128x16x16xbf16>
    %v293 = stablehlo.convert %v291 : (tensor<16x128x16x16xf32>) -> tensor<16x128x16x16xbf16>
    %v294 = stablehlo.convolution(%v292, %v293)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xbf16>, tensor<16x128x16x16xbf16>) -> tensor<16x16x3x3xbf16>
    %v295 = stablehlo.convert %v294 : (tensor<16x16x3x3xbf16>) -> tensor<16x16x3x3xf32>
    %v296 = stablehlo.transpose %v295, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v297 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v298 = stablehlo.multiply %v297, %v296 : tensor<16x16x3x3xf32>
    %v299 = stablehlo.subtract %W4, %v298 : tensor<16x16x3x3xf32>
    %v300 = stablehlo.reshape %v195 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v301 = stablehlo.constant dense<0.0> : tensor<f32>
    %v302 = stablehlo.reduce(%v300 init: %v301) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v303 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v304 = stablehlo.multiply %v303, %v302 : tensor<16xf32>
    %v305 = stablehlo.subtract %cb4, %v304 : tensor<16xf32>
    %v306 = stablehlo.reshape %v47 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v307 = stablehlo.reshape %v179 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v308 = stablehlo.transpose %v306, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v309 = stablehlo.transpose %v307, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v310 = stablehlo.convert %v308 : (tensor<16x128x8x8xf32>) -> tensor<16x128x8x8xbf16>
    %v311 = stablehlo.convert %v309 : (tensor<32x128x8x8xf32>) -> tensor<32x128x8x8xbf16>
    %v312 = stablehlo.convolution(%v310, %v311)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xbf16>, tensor<32x128x8x8xbf16>) -> tensor<16x32x3x3xbf16>
    %v313 = stablehlo.convert %v312 : (tensor<16x32x3x3xbf16>) -> tensor<16x32x3x3xf32>
    %v314 = stablehlo.transpose %v313, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v315 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v316 = stablehlo.multiply %v315, %v314 : tensor<32x16x3x3xf32>
    %v317 = stablehlo.subtract %W5, %v316 : tensor<32x16x3x3xf32>
    %v318 = stablehlo.reshape %v179 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v319 = stablehlo.constant dense<0.0> : tensor<f32>
    %v320 = stablehlo.reduce(%v318 init: %v319) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v321 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v322 = stablehlo.multiply %v321, %v320 : tensor<32xf32>
    %v323 = stablehlo.subtract %cb5, %v322 : tensor<32xf32>
    %v324 = stablehlo.reshape %v57 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v325 = stablehlo.reshape %v168 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v326 = stablehlo.transpose %v324, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v327 = stablehlo.transpose %v325, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v328 = stablehlo.convert %v326 : (tensor<32x128x8x8xf32>) -> tensor<32x128x8x8xbf16>
    %v329 = stablehlo.convert %v327 : (tensor<32x128x8x8xf32>) -> tensor<32x128x8x8xbf16>
    %v330 = stablehlo.convolution(%v328, %v329)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xbf16>, tensor<32x128x8x8xbf16>) -> tensor<32x32x3x3xbf16>
    %v331 = stablehlo.convert %v330 : (tensor<32x32x3x3xbf16>) -> tensor<32x32x3x3xf32>
    %v332 = stablehlo.transpose %v331, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v333 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v334 = stablehlo.multiply %v333, %v332 : tensor<32x32x3x3xf32>
    %v335 = stablehlo.subtract %W6, %v334 : tensor<32x32x3x3xf32>
    %v336 = stablehlo.reshape %v168 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v337 = stablehlo.constant dense<0.0> : tensor<f32>
    %v338 = stablehlo.reduce(%v336 init: %v337) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v339 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v340 = stablehlo.multiply %v339, %v338 : tensor<32xf32>
    %v341 = stablehlo.subtract %cb6, %v340 : tensor<32xf32>
    %v342 = stablehlo.reshape %v71 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v343 = stablehlo.reshape %v152 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v344 = stablehlo.transpose %v342, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v345 = stablehlo.transpose %v343, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v346 = stablehlo.convert %v344 : (tensor<32x128x4x4xf32>) -> tensor<32x128x4x4xbf16>
    %v347 = stablehlo.convert %v345 : (tensor<32x128x4x4xf32>) -> tensor<32x128x4x4xbf16>
    %v348 = stablehlo.convolution(%v346, %v347)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xbf16>, tensor<32x128x4x4xbf16>) -> tensor<32x32x3x3xbf16>
    %v349 = stablehlo.convert %v348 : (tensor<32x32x3x3xbf16>) -> tensor<32x32x3x3xf32>
    %v350 = stablehlo.transpose %v349, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v351 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v352 = stablehlo.multiply %v351, %v350 : tensor<32x32x3x3xf32>
    %v353 = stablehlo.subtract %W7, %v352 : tensor<32x32x3x3xf32>
    %v354 = stablehlo.reshape %v152 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v355 = stablehlo.constant dense<0.0> : tensor<f32>
    %v356 = stablehlo.reduce(%v354 init: %v355) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v357 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v358 = stablehlo.multiply %v357, %v356 : tensor<32xf32>
    %v359 = stablehlo.subtract %cb7, %v358 : tensor<32xf32>
    %v360 = stablehlo.reshape %v81 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v361 = stablehlo.reshape %v141 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v362 = stablehlo.transpose %v360, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v363 = stablehlo.transpose %v361, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v364 = stablehlo.convert %v362 : (tensor<32x128x4x4xf32>) -> tensor<32x128x4x4xbf16>
    %v365 = stablehlo.convert %v363 : (tensor<32x128x4x4xf32>) -> tensor<32x128x4x4xbf16>
    %v366 = stablehlo.convolution(%v364, %v365)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xbf16>, tensor<32x128x4x4xbf16>) -> tensor<32x32x3x3xbf16>
    %v367 = stablehlo.convert %v366 : (tensor<32x32x3x3xbf16>) -> tensor<32x32x3x3xf32>
    %v368 = stablehlo.transpose %v367, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v369 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v370 = stablehlo.multiply %v369, %v368 : tensor<32x32x3x3xf32>
    %v371 = stablehlo.subtract %W8, %v370 : tensor<32x32x3x3xf32>
    %v372 = stablehlo.reshape %v141 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v373 = stablehlo.constant dense<0.0> : tensor<f32>
    %v374 = stablehlo.reduce(%v372 init: %v373) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v375 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v376 = stablehlo.multiply %v375, %v374 : tensor<32xf32>
    %v377 = stablehlo.subtract %cb8, %v376 : tensor<32xf32>
    %v378 = stablehlo.dot_general %v95, %v130, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x512xf32>) -> tensor<128x512xf32>
    %v379 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v380 = stablehlo.multiply %v379, %v378 : tensor<128x512xf32>
    %v381 = stablehlo.subtract %W9, %v380 : tensor<128x512xf32>
    %v382 = stablehlo.constant dense<0.0> : tensor<f32>
    %v383 = stablehlo.reduce(%v130 init: %v382) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v384 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v385 = stablehlo.multiply %v384, %v383 : tensor<512xf32>
    %v386 = stablehlo.subtract %b9, %v385 : tensor<512xf32>
    %v387 = stablehlo.dot_general %v100, %v124, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<512x512xf32>
    %v388 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v389 = stablehlo.multiply %v388, %v387 : tensor<512x512xf32>
    %v390 = stablehlo.subtract %Wa, %v389 : tensor<512x512xf32>
    %v391 = stablehlo.constant dense<0.0> : tensor<f32>
    %v392 = stablehlo.reduce(%v124 init: %v391) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v393 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v394 = stablehlo.multiply %v393, %v392 : tensor<512xf32>
    %v395 = stablehlo.subtract %ba, %v394 : tensor<512xf32>
    %v396 = stablehlo.dot_general %v105, %v118, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x10xf32>) -> tensor<512x10xf32>
    %v397 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v398 = stablehlo.multiply %v397, %v396 : tensor<512x10xf32>
    %v399 = stablehlo.subtract %Wb, %v398 : tensor<512x10xf32>
    %v400 = stablehlo.constant dense<0.0> : tensor<f32>
    %v401 = stablehlo.reduce(%v118 init: %v400) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v402 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v403 = stablehlo.multiply %v402, %v401 : tensor<10xf32>
    %v404 = stablehlo.subtract %bb, %v403 : tensor<10xf32>
    return %v245, %v251, %v263, %v269, %v281, %v287, %v299, %v305, %v317, %v323, %v335, %v341, %v353, %v359, %v371, %v377, %v381, %v386, %v390, %v395, %v399, %v404, %W1m, %cb1m, %W2m, %cb2m, %W3m, %cb3m, %W4m, %cb4m, %W5m, %cb5m, %W6m, %cb6m, %W7m, %cb7m, %W8m, %cb8m, %W9m, %b9m, %Wam, %bam, %Wbm, %bbm, %W1v, %cb1v, %W2v, %cb2v, %W3v, %cb3v, %W4v, %cb4v, %W5v, %cb5v, %W6v, %cb6v, %W7v, %cb7v, %W8v, %cb8v, %W9v, %b9v, %Wav, %bav, %Wbv, %bbv, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
