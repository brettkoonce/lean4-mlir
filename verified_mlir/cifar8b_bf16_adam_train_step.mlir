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
    %v96 = stablehlo.dot_general %v95, %W9, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %v97 = stablehlo.broadcast_in_dim %b9, dims = [1] : (tensor<64xf32>) -> tensor<128x64xf32>
    %v98 = stablehlo.add %v96, %v97 : tensor<128x64xf32>
    %v99 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v100 = stablehlo.maximum %v98, %v99 : tensor<128x64xf32>
    %v101 = stablehlo.dot_general %v100, %Wa, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
    %v102 = stablehlo.broadcast_in_dim %ba, dims = [1] : (tensor<64xf32>) -> tensor<128x64xf32>
    %v103 = stablehlo.add %v101, %v102 : tensor<128x64xf32>
    %v104 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v105 = stablehlo.maximum %v103, %v104 : tensor<128x64xf32>
    %v106 = stablehlo.dot_general %v105, %Wb, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x10xf32>) -> tensor<128x10xf32>
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
    %v120 = stablehlo.dot_general %v119, %Wb, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x10xf32>, tensor<64x10xf32>) -> tensor<128x1x64xf32>
    %v121 = stablehlo.reshape %v120 : (tensor<128x1x64xf32>) -> tensor<128x64xf32>
    %v122 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v123 = stablehlo.compare GT, %v103, %v122 : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xi1>
    %v124 = stablehlo.select %v123, %v121, %v122 : tensor<128x64xi1>, tensor<128x64xf32>
    %v125 = stablehlo.reshape %v124 : (tensor<128x64xf32>) -> tensor<128x1x64xf32>
    %v126 = stablehlo.dot_general %v125, %Wa, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x64xf32>, tensor<64x64xf32>) -> tensor<128x1x64xf32>
    %v127 = stablehlo.reshape %v126 : (tensor<128x1x64xf32>) -> tensor<128x64xf32>
    %v128 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v129 = stablehlo.compare GT, %v98, %v128 : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xi1>
    %v130 = stablehlo.select %v129, %v127, %v128 : tensor<128x64xi1>, tensor<128x64xf32>
    %v131 = stablehlo.reshape %v130 : (tensor<128x64xf32>) -> tensor<128x1x64xf32>
    %v132 = stablehlo.dot_general %v131, %W9, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x64xf32>, tensor<128x64xf32>) -> tensor<128x1x128xf32>
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
    %v243 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v244 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v245 = stablehlo.multiply %v243, %W1m : tensor<16x3x3x3xf32>
    %v246 = stablehlo.multiply %v244, %v242 : tensor<16x3x3x3xf32>
    %v247 = stablehlo.add %v245, %v246 : tensor<16x3x3x3xf32>
    %v248 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v249 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v250 = stablehlo.multiply %v248, %W1v : tensor<16x3x3x3xf32>
    %v251 = stablehlo.multiply %v242, %v242 : tensor<16x3x3x3xf32>
    %v252 = stablehlo.multiply %v249, %v251 : tensor<16x3x3x3xf32>
    %v253 = stablehlo.add %v250, %v252 : tensor<16x3x3x3xf32>
    %v254 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v255 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v256 = stablehlo.divide %v247, %v254 : tensor<16x3x3x3xf32>
    %v257 = stablehlo.divide %v253, %v255 : tensor<16x3x3x3xf32>
    %v258 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v259 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v260 = stablehlo.sqrt %v257 : tensor<16x3x3x3xf32>
    %v261 = stablehlo.add %v260, %v259 : tensor<16x3x3x3xf32>
    %v262 = stablehlo.divide %v256, %v261 : tensor<16x3x3x3xf32>
    %v263 = stablehlo.multiply %v258, %v262 : tensor<16x3x3x3xf32>
    %v264 = stablehlo.subtract %W1, %v263 : tensor<16x3x3x3xf32>
    %v265 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v266 = stablehlo.multiply %v265, %v258 : tensor<16x3x3x3xf32>
    %v267 = stablehlo.multiply %v266, %W1 : tensor<16x3x3x3xf32>
    %v268 = stablehlo.subtract %v264, %v267 : tensor<16x3x3x3xf32>
    %v269 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v270 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v271 = stablehlo.multiply %v269, %W1m : tensor<16x3x3x3xf32>
    %v272 = stablehlo.multiply %v270, %v242 : tensor<16x3x3x3xf32>
    %v273 = stablehlo.add %v271, %v272 : tensor<16x3x3x3xf32>
    %v274 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v275 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v276 = stablehlo.multiply %v274, %W1v : tensor<16x3x3x3xf32>
    %v277 = stablehlo.multiply %v242, %v242 : tensor<16x3x3x3xf32>
    %v278 = stablehlo.multiply %v275, %v277 : tensor<16x3x3x3xf32>
    %v279 = stablehlo.add %v276, %v278 : tensor<16x3x3x3xf32>
    %v280 = stablehlo.reshape %v233 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v281 = stablehlo.constant dense<0.0> : tensor<f32>
    %v282 = stablehlo.reduce(%v280 init: %v281) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v283 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v284 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v285 = stablehlo.multiply %v283, %cb1m : tensor<16xf32>
    %v286 = stablehlo.multiply %v284, %v282 : tensor<16xf32>
    %v287 = stablehlo.add %v285, %v286 : tensor<16xf32>
    %v288 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v289 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v290 = stablehlo.multiply %v288, %cb1v : tensor<16xf32>
    %v291 = stablehlo.multiply %v282, %v282 : tensor<16xf32>
    %v292 = stablehlo.multiply %v289, %v291 : tensor<16xf32>
    %v293 = stablehlo.add %v290, %v292 : tensor<16xf32>
    %v294 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v295 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v296 = stablehlo.divide %v287, %v294 : tensor<16xf32>
    %v297 = stablehlo.divide %v293, %v295 : tensor<16xf32>
    %v298 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v299 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v300 = stablehlo.sqrt %v297 : tensor<16xf32>
    %v301 = stablehlo.add %v300, %v299 : tensor<16xf32>
    %v302 = stablehlo.divide %v296, %v301 : tensor<16xf32>
    %v303 = stablehlo.multiply %v298, %v302 : tensor<16xf32>
    %v304 = stablehlo.subtract %cb1, %v303 : tensor<16xf32>
    %v305 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v306 = stablehlo.multiply %v305, %v298 : tensor<16xf32>
    %v307 = stablehlo.multiply %v306, %cb1 : tensor<16xf32>
    %v308 = stablehlo.subtract %v304, %v307 : tensor<16xf32>
    %v309 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v310 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v311 = stablehlo.multiply %v309, %cb1m : tensor<16xf32>
    %v312 = stablehlo.multiply %v310, %v282 : tensor<16xf32>
    %v313 = stablehlo.add %v311, %v312 : tensor<16xf32>
    %v314 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v315 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v316 = stablehlo.multiply %v314, %cb1v : tensor<16xf32>
    %v317 = stablehlo.multiply %v282, %v282 : tensor<16xf32>
    %v318 = stablehlo.multiply %v315, %v317 : tensor<16xf32>
    %v319 = stablehlo.add %v316, %v318 : tensor<16xf32>
    %v320 = stablehlo.reshape %v9 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v321 = stablehlo.reshape %v222 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v322 = stablehlo.transpose %v320, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v323 = stablehlo.transpose %v321, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v324 = stablehlo.convert %v322 : (tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xbf16>
    %v325 = stablehlo.convert %v323 : (tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xbf16>
    %v326 = stablehlo.convolution(%v324, %v325)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xbf16>, tensor<16x128x32x32xbf16>) -> tensor<16x16x3x3xbf16>
    %v327 = stablehlo.convert %v326 : (tensor<16x16x3x3xbf16>) -> tensor<16x16x3x3xf32>
    %v328 = stablehlo.transpose %v327, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v329 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v330 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v331 = stablehlo.multiply %v329, %W2m : tensor<16x16x3x3xf32>
    %v332 = stablehlo.multiply %v330, %v328 : tensor<16x16x3x3xf32>
    %v333 = stablehlo.add %v331, %v332 : tensor<16x16x3x3xf32>
    %v334 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v335 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v336 = stablehlo.multiply %v334, %W2v : tensor<16x16x3x3xf32>
    %v337 = stablehlo.multiply %v328, %v328 : tensor<16x16x3x3xf32>
    %v338 = stablehlo.multiply %v335, %v337 : tensor<16x16x3x3xf32>
    %v339 = stablehlo.add %v336, %v338 : tensor<16x16x3x3xf32>
    %v340 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v341 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v342 = stablehlo.divide %v333, %v340 : tensor<16x16x3x3xf32>
    %v343 = stablehlo.divide %v339, %v341 : tensor<16x16x3x3xf32>
    %v344 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v345 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v346 = stablehlo.sqrt %v343 : tensor<16x16x3x3xf32>
    %v347 = stablehlo.add %v346, %v345 : tensor<16x16x3x3xf32>
    %v348 = stablehlo.divide %v342, %v347 : tensor<16x16x3x3xf32>
    %v349 = stablehlo.multiply %v344, %v348 : tensor<16x16x3x3xf32>
    %v350 = stablehlo.subtract %W2, %v349 : tensor<16x16x3x3xf32>
    %v351 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v352 = stablehlo.multiply %v351, %v344 : tensor<16x16x3x3xf32>
    %v353 = stablehlo.multiply %v352, %W2 : tensor<16x16x3x3xf32>
    %v354 = stablehlo.subtract %v350, %v353 : tensor<16x16x3x3xf32>
    %v355 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v356 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v357 = stablehlo.multiply %v355, %W2m : tensor<16x16x3x3xf32>
    %v358 = stablehlo.multiply %v356, %v328 : tensor<16x16x3x3xf32>
    %v359 = stablehlo.add %v357, %v358 : tensor<16x16x3x3xf32>
    %v360 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v361 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v362 = stablehlo.multiply %v360, %W2v : tensor<16x16x3x3xf32>
    %v363 = stablehlo.multiply %v328, %v328 : tensor<16x16x3x3xf32>
    %v364 = stablehlo.multiply %v361, %v363 : tensor<16x16x3x3xf32>
    %v365 = stablehlo.add %v362, %v364 : tensor<16x16x3x3xf32>
    %v366 = stablehlo.reshape %v222 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v367 = stablehlo.constant dense<0.0> : tensor<f32>
    %v368 = stablehlo.reduce(%v366 init: %v367) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v369 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v370 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v371 = stablehlo.multiply %v369, %cb2m : tensor<16xf32>
    %v372 = stablehlo.multiply %v370, %v368 : tensor<16xf32>
    %v373 = stablehlo.add %v371, %v372 : tensor<16xf32>
    %v374 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v375 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v376 = stablehlo.multiply %v374, %cb2v : tensor<16xf32>
    %v377 = stablehlo.multiply %v368, %v368 : tensor<16xf32>
    %v378 = stablehlo.multiply %v375, %v377 : tensor<16xf32>
    %v379 = stablehlo.add %v376, %v378 : tensor<16xf32>
    %v380 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v381 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v382 = stablehlo.divide %v373, %v380 : tensor<16xf32>
    %v383 = stablehlo.divide %v379, %v381 : tensor<16xf32>
    %v384 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v385 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v386 = stablehlo.sqrt %v383 : tensor<16xf32>
    %v387 = stablehlo.add %v386, %v385 : tensor<16xf32>
    %v388 = stablehlo.divide %v382, %v387 : tensor<16xf32>
    %v389 = stablehlo.multiply %v384, %v388 : tensor<16xf32>
    %v390 = stablehlo.subtract %cb2, %v389 : tensor<16xf32>
    %v391 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v392 = stablehlo.multiply %v391, %v384 : tensor<16xf32>
    %v393 = stablehlo.multiply %v392, %cb2 : tensor<16xf32>
    %v394 = stablehlo.subtract %v390, %v393 : tensor<16xf32>
    %v395 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v396 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v397 = stablehlo.multiply %v395, %cb2m : tensor<16xf32>
    %v398 = stablehlo.multiply %v396, %v368 : tensor<16xf32>
    %v399 = stablehlo.add %v397, %v398 : tensor<16xf32>
    %v400 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v401 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v402 = stablehlo.multiply %v400, %cb2v : tensor<16xf32>
    %v403 = stablehlo.multiply %v368, %v368 : tensor<16xf32>
    %v404 = stablehlo.multiply %v401, %v403 : tensor<16xf32>
    %v405 = stablehlo.add %v402, %v404 : tensor<16xf32>
    %v406 = stablehlo.reshape %v23 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v407 = stablehlo.reshape %v206 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v408 = stablehlo.transpose %v406, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v409 = stablehlo.transpose %v407, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v410 = stablehlo.convert %v408 : (tensor<16x128x16x16xf32>) -> tensor<16x128x16x16xbf16>
    %v411 = stablehlo.convert %v409 : (tensor<16x128x16x16xf32>) -> tensor<16x128x16x16xbf16>
    %v412 = stablehlo.convolution(%v410, %v411)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xbf16>, tensor<16x128x16x16xbf16>) -> tensor<16x16x3x3xbf16>
    %v413 = stablehlo.convert %v412 : (tensor<16x16x3x3xbf16>) -> tensor<16x16x3x3xf32>
    %v414 = stablehlo.transpose %v413, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v415 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v416 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v417 = stablehlo.multiply %v415, %W3m : tensor<16x16x3x3xf32>
    %v418 = stablehlo.multiply %v416, %v414 : tensor<16x16x3x3xf32>
    %v419 = stablehlo.add %v417, %v418 : tensor<16x16x3x3xf32>
    %v420 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v421 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v422 = stablehlo.multiply %v420, %W3v : tensor<16x16x3x3xf32>
    %v423 = stablehlo.multiply %v414, %v414 : tensor<16x16x3x3xf32>
    %v424 = stablehlo.multiply %v421, %v423 : tensor<16x16x3x3xf32>
    %v425 = stablehlo.add %v422, %v424 : tensor<16x16x3x3xf32>
    %v426 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v427 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v428 = stablehlo.divide %v419, %v426 : tensor<16x16x3x3xf32>
    %v429 = stablehlo.divide %v425, %v427 : tensor<16x16x3x3xf32>
    %v430 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v431 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v432 = stablehlo.sqrt %v429 : tensor<16x16x3x3xf32>
    %v433 = stablehlo.add %v432, %v431 : tensor<16x16x3x3xf32>
    %v434 = stablehlo.divide %v428, %v433 : tensor<16x16x3x3xf32>
    %v435 = stablehlo.multiply %v430, %v434 : tensor<16x16x3x3xf32>
    %v436 = stablehlo.subtract %W3, %v435 : tensor<16x16x3x3xf32>
    %v437 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v438 = stablehlo.multiply %v437, %v430 : tensor<16x16x3x3xf32>
    %v439 = stablehlo.multiply %v438, %W3 : tensor<16x16x3x3xf32>
    %v440 = stablehlo.subtract %v436, %v439 : tensor<16x16x3x3xf32>
    %v441 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v442 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v443 = stablehlo.multiply %v441, %W3m : tensor<16x16x3x3xf32>
    %v444 = stablehlo.multiply %v442, %v414 : tensor<16x16x3x3xf32>
    %v445 = stablehlo.add %v443, %v444 : tensor<16x16x3x3xf32>
    %v446 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v447 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v448 = stablehlo.multiply %v446, %W3v : tensor<16x16x3x3xf32>
    %v449 = stablehlo.multiply %v414, %v414 : tensor<16x16x3x3xf32>
    %v450 = stablehlo.multiply %v447, %v449 : tensor<16x16x3x3xf32>
    %v451 = stablehlo.add %v448, %v450 : tensor<16x16x3x3xf32>
    %v452 = stablehlo.reshape %v206 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v453 = stablehlo.constant dense<0.0> : tensor<f32>
    %v454 = stablehlo.reduce(%v452 init: %v453) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v455 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v456 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v457 = stablehlo.multiply %v455, %cb3m : tensor<16xf32>
    %v458 = stablehlo.multiply %v456, %v454 : tensor<16xf32>
    %v459 = stablehlo.add %v457, %v458 : tensor<16xf32>
    %v460 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v461 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v462 = stablehlo.multiply %v460, %cb3v : tensor<16xf32>
    %v463 = stablehlo.multiply %v454, %v454 : tensor<16xf32>
    %v464 = stablehlo.multiply %v461, %v463 : tensor<16xf32>
    %v465 = stablehlo.add %v462, %v464 : tensor<16xf32>
    %v466 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v467 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v468 = stablehlo.divide %v459, %v466 : tensor<16xf32>
    %v469 = stablehlo.divide %v465, %v467 : tensor<16xf32>
    %v470 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v471 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v472 = stablehlo.sqrt %v469 : tensor<16xf32>
    %v473 = stablehlo.add %v472, %v471 : tensor<16xf32>
    %v474 = stablehlo.divide %v468, %v473 : tensor<16xf32>
    %v475 = stablehlo.multiply %v470, %v474 : tensor<16xf32>
    %v476 = stablehlo.subtract %cb3, %v475 : tensor<16xf32>
    %v477 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v478 = stablehlo.multiply %v477, %v470 : tensor<16xf32>
    %v479 = stablehlo.multiply %v478, %cb3 : tensor<16xf32>
    %v480 = stablehlo.subtract %v476, %v479 : tensor<16xf32>
    %v481 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v482 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v483 = stablehlo.multiply %v481, %cb3m : tensor<16xf32>
    %v484 = stablehlo.multiply %v482, %v454 : tensor<16xf32>
    %v485 = stablehlo.add %v483, %v484 : tensor<16xf32>
    %v486 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v487 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v488 = stablehlo.multiply %v486, %cb3v : tensor<16xf32>
    %v489 = stablehlo.multiply %v454, %v454 : tensor<16xf32>
    %v490 = stablehlo.multiply %v487, %v489 : tensor<16xf32>
    %v491 = stablehlo.add %v488, %v490 : tensor<16xf32>
    %v492 = stablehlo.reshape %v33 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v493 = stablehlo.reshape %v195 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v494 = stablehlo.transpose %v492, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v495 = stablehlo.transpose %v493, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v496 = stablehlo.convert %v494 : (tensor<16x128x16x16xf32>) -> tensor<16x128x16x16xbf16>
    %v497 = stablehlo.convert %v495 : (tensor<16x128x16x16xf32>) -> tensor<16x128x16x16xbf16>
    %v498 = stablehlo.convolution(%v496, %v497)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xbf16>, tensor<16x128x16x16xbf16>) -> tensor<16x16x3x3xbf16>
    %v499 = stablehlo.convert %v498 : (tensor<16x16x3x3xbf16>) -> tensor<16x16x3x3xf32>
    %v500 = stablehlo.transpose %v499, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v501 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v502 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v503 = stablehlo.multiply %v501, %W4m : tensor<16x16x3x3xf32>
    %v504 = stablehlo.multiply %v502, %v500 : tensor<16x16x3x3xf32>
    %v505 = stablehlo.add %v503, %v504 : tensor<16x16x3x3xf32>
    %v506 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v507 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v508 = stablehlo.multiply %v506, %W4v : tensor<16x16x3x3xf32>
    %v509 = stablehlo.multiply %v500, %v500 : tensor<16x16x3x3xf32>
    %v510 = stablehlo.multiply %v507, %v509 : tensor<16x16x3x3xf32>
    %v511 = stablehlo.add %v508, %v510 : tensor<16x16x3x3xf32>
    %v512 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v513 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v514 = stablehlo.divide %v505, %v512 : tensor<16x16x3x3xf32>
    %v515 = stablehlo.divide %v511, %v513 : tensor<16x16x3x3xf32>
    %v516 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v517 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v518 = stablehlo.sqrt %v515 : tensor<16x16x3x3xf32>
    %v519 = stablehlo.add %v518, %v517 : tensor<16x16x3x3xf32>
    %v520 = stablehlo.divide %v514, %v519 : tensor<16x16x3x3xf32>
    %v521 = stablehlo.multiply %v516, %v520 : tensor<16x16x3x3xf32>
    %v522 = stablehlo.subtract %W4, %v521 : tensor<16x16x3x3xf32>
    %v523 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v524 = stablehlo.multiply %v523, %v516 : tensor<16x16x3x3xf32>
    %v525 = stablehlo.multiply %v524, %W4 : tensor<16x16x3x3xf32>
    %v526 = stablehlo.subtract %v522, %v525 : tensor<16x16x3x3xf32>
    %v527 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v528 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v529 = stablehlo.multiply %v527, %W4m : tensor<16x16x3x3xf32>
    %v530 = stablehlo.multiply %v528, %v500 : tensor<16x16x3x3xf32>
    %v531 = stablehlo.add %v529, %v530 : tensor<16x16x3x3xf32>
    %v532 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v533 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v534 = stablehlo.multiply %v532, %W4v : tensor<16x16x3x3xf32>
    %v535 = stablehlo.multiply %v500, %v500 : tensor<16x16x3x3xf32>
    %v536 = stablehlo.multiply %v533, %v535 : tensor<16x16x3x3xf32>
    %v537 = stablehlo.add %v534, %v536 : tensor<16x16x3x3xf32>
    %v538 = stablehlo.reshape %v195 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v539 = stablehlo.constant dense<0.0> : tensor<f32>
    %v540 = stablehlo.reduce(%v538 init: %v539) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v541 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v542 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v543 = stablehlo.multiply %v541, %cb4m : tensor<16xf32>
    %v544 = stablehlo.multiply %v542, %v540 : tensor<16xf32>
    %v545 = stablehlo.add %v543, %v544 : tensor<16xf32>
    %v546 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v547 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v548 = stablehlo.multiply %v546, %cb4v : tensor<16xf32>
    %v549 = stablehlo.multiply %v540, %v540 : tensor<16xf32>
    %v550 = stablehlo.multiply %v547, %v549 : tensor<16xf32>
    %v551 = stablehlo.add %v548, %v550 : tensor<16xf32>
    %v552 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v553 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v554 = stablehlo.divide %v545, %v552 : tensor<16xf32>
    %v555 = stablehlo.divide %v551, %v553 : tensor<16xf32>
    %v556 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v557 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v558 = stablehlo.sqrt %v555 : tensor<16xf32>
    %v559 = stablehlo.add %v558, %v557 : tensor<16xf32>
    %v560 = stablehlo.divide %v554, %v559 : tensor<16xf32>
    %v561 = stablehlo.multiply %v556, %v560 : tensor<16xf32>
    %v562 = stablehlo.subtract %cb4, %v561 : tensor<16xf32>
    %v563 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v564 = stablehlo.multiply %v563, %v556 : tensor<16xf32>
    %v565 = stablehlo.multiply %v564, %cb4 : tensor<16xf32>
    %v566 = stablehlo.subtract %v562, %v565 : tensor<16xf32>
    %v567 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v568 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v569 = stablehlo.multiply %v567, %cb4m : tensor<16xf32>
    %v570 = stablehlo.multiply %v568, %v540 : tensor<16xf32>
    %v571 = stablehlo.add %v569, %v570 : tensor<16xf32>
    %v572 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v573 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v574 = stablehlo.multiply %v572, %cb4v : tensor<16xf32>
    %v575 = stablehlo.multiply %v540, %v540 : tensor<16xf32>
    %v576 = stablehlo.multiply %v573, %v575 : tensor<16xf32>
    %v577 = stablehlo.add %v574, %v576 : tensor<16xf32>
    %v578 = stablehlo.reshape %v47 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v579 = stablehlo.reshape %v179 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v580 = stablehlo.transpose %v578, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v581 = stablehlo.transpose %v579, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v582 = stablehlo.convert %v580 : (tensor<16x128x8x8xf32>) -> tensor<16x128x8x8xbf16>
    %v583 = stablehlo.convert %v581 : (tensor<32x128x8x8xf32>) -> tensor<32x128x8x8xbf16>
    %v584 = stablehlo.convolution(%v582, %v583)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xbf16>, tensor<32x128x8x8xbf16>) -> tensor<16x32x3x3xbf16>
    %v585 = stablehlo.convert %v584 : (tensor<16x32x3x3xbf16>) -> tensor<16x32x3x3xf32>
    %v586 = stablehlo.transpose %v585, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v587 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v588 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v589 = stablehlo.multiply %v587, %W5m : tensor<32x16x3x3xf32>
    %v590 = stablehlo.multiply %v588, %v586 : tensor<32x16x3x3xf32>
    %v591 = stablehlo.add %v589, %v590 : tensor<32x16x3x3xf32>
    %v592 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v593 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v594 = stablehlo.multiply %v592, %W5v : tensor<32x16x3x3xf32>
    %v595 = stablehlo.multiply %v586, %v586 : tensor<32x16x3x3xf32>
    %v596 = stablehlo.multiply %v593, %v595 : tensor<32x16x3x3xf32>
    %v597 = stablehlo.add %v594, %v596 : tensor<32x16x3x3xf32>
    %v598 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v599 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v600 = stablehlo.divide %v591, %v598 : tensor<32x16x3x3xf32>
    %v601 = stablehlo.divide %v597, %v599 : tensor<32x16x3x3xf32>
    %v602 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v603 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v604 = stablehlo.sqrt %v601 : tensor<32x16x3x3xf32>
    %v605 = stablehlo.add %v604, %v603 : tensor<32x16x3x3xf32>
    %v606 = stablehlo.divide %v600, %v605 : tensor<32x16x3x3xf32>
    %v607 = stablehlo.multiply %v602, %v606 : tensor<32x16x3x3xf32>
    %v608 = stablehlo.subtract %W5, %v607 : tensor<32x16x3x3xf32>
    %v609 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v610 = stablehlo.multiply %v609, %v602 : tensor<32x16x3x3xf32>
    %v611 = stablehlo.multiply %v610, %W5 : tensor<32x16x3x3xf32>
    %v612 = stablehlo.subtract %v608, %v611 : tensor<32x16x3x3xf32>
    %v613 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v614 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v615 = stablehlo.multiply %v613, %W5m : tensor<32x16x3x3xf32>
    %v616 = stablehlo.multiply %v614, %v586 : tensor<32x16x3x3xf32>
    %v617 = stablehlo.add %v615, %v616 : tensor<32x16x3x3xf32>
    %v618 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v619 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v620 = stablehlo.multiply %v618, %W5v : tensor<32x16x3x3xf32>
    %v621 = stablehlo.multiply %v586, %v586 : tensor<32x16x3x3xf32>
    %v622 = stablehlo.multiply %v619, %v621 : tensor<32x16x3x3xf32>
    %v623 = stablehlo.add %v620, %v622 : tensor<32x16x3x3xf32>
    %v624 = stablehlo.reshape %v179 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v625 = stablehlo.constant dense<0.0> : tensor<f32>
    %v626 = stablehlo.reduce(%v624 init: %v625) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v627 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v628 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v629 = stablehlo.multiply %v627, %cb5m : tensor<32xf32>
    %v630 = stablehlo.multiply %v628, %v626 : tensor<32xf32>
    %v631 = stablehlo.add %v629, %v630 : tensor<32xf32>
    %v632 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v633 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v634 = stablehlo.multiply %v632, %cb5v : tensor<32xf32>
    %v635 = stablehlo.multiply %v626, %v626 : tensor<32xf32>
    %v636 = stablehlo.multiply %v633, %v635 : tensor<32xf32>
    %v637 = stablehlo.add %v634, %v636 : tensor<32xf32>
    %v638 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v639 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v640 = stablehlo.divide %v631, %v638 : tensor<32xf32>
    %v641 = stablehlo.divide %v637, %v639 : tensor<32xf32>
    %v642 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v643 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v644 = stablehlo.sqrt %v641 : tensor<32xf32>
    %v645 = stablehlo.add %v644, %v643 : tensor<32xf32>
    %v646 = stablehlo.divide %v640, %v645 : tensor<32xf32>
    %v647 = stablehlo.multiply %v642, %v646 : tensor<32xf32>
    %v648 = stablehlo.subtract %cb5, %v647 : tensor<32xf32>
    %v649 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v650 = stablehlo.multiply %v649, %v642 : tensor<32xf32>
    %v651 = stablehlo.multiply %v650, %cb5 : tensor<32xf32>
    %v652 = stablehlo.subtract %v648, %v651 : tensor<32xf32>
    %v653 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v654 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v655 = stablehlo.multiply %v653, %cb5m : tensor<32xf32>
    %v656 = stablehlo.multiply %v654, %v626 : tensor<32xf32>
    %v657 = stablehlo.add %v655, %v656 : tensor<32xf32>
    %v658 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v659 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v660 = stablehlo.multiply %v658, %cb5v : tensor<32xf32>
    %v661 = stablehlo.multiply %v626, %v626 : tensor<32xf32>
    %v662 = stablehlo.multiply %v659, %v661 : tensor<32xf32>
    %v663 = stablehlo.add %v660, %v662 : tensor<32xf32>
    %v664 = stablehlo.reshape %v57 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v665 = stablehlo.reshape %v168 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v666 = stablehlo.transpose %v664, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v667 = stablehlo.transpose %v665, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v668 = stablehlo.convert %v666 : (tensor<32x128x8x8xf32>) -> tensor<32x128x8x8xbf16>
    %v669 = stablehlo.convert %v667 : (tensor<32x128x8x8xf32>) -> tensor<32x128x8x8xbf16>
    %v670 = stablehlo.convolution(%v668, %v669)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xbf16>, tensor<32x128x8x8xbf16>) -> tensor<32x32x3x3xbf16>
    %v671 = stablehlo.convert %v670 : (tensor<32x32x3x3xbf16>) -> tensor<32x32x3x3xf32>
    %v672 = stablehlo.transpose %v671, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v673 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v674 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v675 = stablehlo.multiply %v673, %W6m : tensor<32x32x3x3xf32>
    %v676 = stablehlo.multiply %v674, %v672 : tensor<32x32x3x3xf32>
    %v677 = stablehlo.add %v675, %v676 : tensor<32x32x3x3xf32>
    %v678 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v679 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v680 = stablehlo.multiply %v678, %W6v : tensor<32x32x3x3xf32>
    %v681 = stablehlo.multiply %v672, %v672 : tensor<32x32x3x3xf32>
    %v682 = stablehlo.multiply %v679, %v681 : tensor<32x32x3x3xf32>
    %v683 = stablehlo.add %v680, %v682 : tensor<32x32x3x3xf32>
    %v684 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v685 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v686 = stablehlo.divide %v677, %v684 : tensor<32x32x3x3xf32>
    %v687 = stablehlo.divide %v683, %v685 : tensor<32x32x3x3xf32>
    %v688 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v689 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v690 = stablehlo.sqrt %v687 : tensor<32x32x3x3xf32>
    %v691 = stablehlo.add %v690, %v689 : tensor<32x32x3x3xf32>
    %v692 = stablehlo.divide %v686, %v691 : tensor<32x32x3x3xf32>
    %v693 = stablehlo.multiply %v688, %v692 : tensor<32x32x3x3xf32>
    %v694 = stablehlo.subtract %W6, %v693 : tensor<32x32x3x3xf32>
    %v695 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v696 = stablehlo.multiply %v695, %v688 : tensor<32x32x3x3xf32>
    %v697 = stablehlo.multiply %v696, %W6 : tensor<32x32x3x3xf32>
    %v698 = stablehlo.subtract %v694, %v697 : tensor<32x32x3x3xf32>
    %v699 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v700 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v701 = stablehlo.multiply %v699, %W6m : tensor<32x32x3x3xf32>
    %v702 = stablehlo.multiply %v700, %v672 : tensor<32x32x3x3xf32>
    %v703 = stablehlo.add %v701, %v702 : tensor<32x32x3x3xf32>
    %v704 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v705 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v706 = stablehlo.multiply %v704, %W6v : tensor<32x32x3x3xf32>
    %v707 = stablehlo.multiply %v672, %v672 : tensor<32x32x3x3xf32>
    %v708 = stablehlo.multiply %v705, %v707 : tensor<32x32x3x3xf32>
    %v709 = stablehlo.add %v706, %v708 : tensor<32x32x3x3xf32>
    %v710 = stablehlo.reshape %v168 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v711 = stablehlo.constant dense<0.0> : tensor<f32>
    %v712 = stablehlo.reduce(%v710 init: %v711) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v713 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v714 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v715 = stablehlo.multiply %v713, %cb6m : tensor<32xf32>
    %v716 = stablehlo.multiply %v714, %v712 : tensor<32xf32>
    %v717 = stablehlo.add %v715, %v716 : tensor<32xf32>
    %v718 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v719 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v720 = stablehlo.multiply %v718, %cb6v : tensor<32xf32>
    %v721 = stablehlo.multiply %v712, %v712 : tensor<32xf32>
    %v722 = stablehlo.multiply %v719, %v721 : tensor<32xf32>
    %v723 = stablehlo.add %v720, %v722 : tensor<32xf32>
    %v724 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v725 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v726 = stablehlo.divide %v717, %v724 : tensor<32xf32>
    %v727 = stablehlo.divide %v723, %v725 : tensor<32xf32>
    %v728 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v729 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v730 = stablehlo.sqrt %v727 : tensor<32xf32>
    %v731 = stablehlo.add %v730, %v729 : tensor<32xf32>
    %v732 = stablehlo.divide %v726, %v731 : tensor<32xf32>
    %v733 = stablehlo.multiply %v728, %v732 : tensor<32xf32>
    %v734 = stablehlo.subtract %cb6, %v733 : tensor<32xf32>
    %v735 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v736 = stablehlo.multiply %v735, %v728 : tensor<32xf32>
    %v737 = stablehlo.multiply %v736, %cb6 : tensor<32xf32>
    %v738 = stablehlo.subtract %v734, %v737 : tensor<32xf32>
    %v739 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v740 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v741 = stablehlo.multiply %v739, %cb6m : tensor<32xf32>
    %v742 = stablehlo.multiply %v740, %v712 : tensor<32xf32>
    %v743 = stablehlo.add %v741, %v742 : tensor<32xf32>
    %v744 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v745 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v746 = stablehlo.multiply %v744, %cb6v : tensor<32xf32>
    %v747 = stablehlo.multiply %v712, %v712 : tensor<32xf32>
    %v748 = stablehlo.multiply %v745, %v747 : tensor<32xf32>
    %v749 = stablehlo.add %v746, %v748 : tensor<32xf32>
    %v750 = stablehlo.reshape %v71 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v751 = stablehlo.reshape %v152 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v752 = stablehlo.transpose %v750, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v753 = stablehlo.transpose %v751, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v754 = stablehlo.convert %v752 : (tensor<32x128x4x4xf32>) -> tensor<32x128x4x4xbf16>
    %v755 = stablehlo.convert %v753 : (tensor<32x128x4x4xf32>) -> tensor<32x128x4x4xbf16>
    %v756 = stablehlo.convolution(%v754, %v755)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xbf16>, tensor<32x128x4x4xbf16>) -> tensor<32x32x3x3xbf16>
    %v757 = stablehlo.convert %v756 : (tensor<32x32x3x3xbf16>) -> tensor<32x32x3x3xf32>
    %v758 = stablehlo.transpose %v757, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v759 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v760 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v761 = stablehlo.multiply %v759, %W7m : tensor<32x32x3x3xf32>
    %v762 = stablehlo.multiply %v760, %v758 : tensor<32x32x3x3xf32>
    %v763 = stablehlo.add %v761, %v762 : tensor<32x32x3x3xf32>
    %v764 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v765 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v766 = stablehlo.multiply %v764, %W7v : tensor<32x32x3x3xf32>
    %v767 = stablehlo.multiply %v758, %v758 : tensor<32x32x3x3xf32>
    %v768 = stablehlo.multiply %v765, %v767 : tensor<32x32x3x3xf32>
    %v769 = stablehlo.add %v766, %v768 : tensor<32x32x3x3xf32>
    %v770 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v771 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v772 = stablehlo.divide %v763, %v770 : tensor<32x32x3x3xf32>
    %v773 = stablehlo.divide %v769, %v771 : tensor<32x32x3x3xf32>
    %v774 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v775 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v776 = stablehlo.sqrt %v773 : tensor<32x32x3x3xf32>
    %v777 = stablehlo.add %v776, %v775 : tensor<32x32x3x3xf32>
    %v778 = stablehlo.divide %v772, %v777 : tensor<32x32x3x3xf32>
    %v779 = stablehlo.multiply %v774, %v778 : tensor<32x32x3x3xf32>
    %v780 = stablehlo.subtract %W7, %v779 : tensor<32x32x3x3xf32>
    %v781 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v782 = stablehlo.multiply %v781, %v774 : tensor<32x32x3x3xf32>
    %v783 = stablehlo.multiply %v782, %W7 : tensor<32x32x3x3xf32>
    %v784 = stablehlo.subtract %v780, %v783 : tensor<32x32x3x3xf32>
    %v785 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v786 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v787 = stablehlo.multiply %v785, %W7m : tensor<32x32x3x3xf32>
    %v788 = stablehlo.multiply %v786, %v758 : tensor<32x32x3x3xf32>
    %v789 = stablehlo.add %v787, %v788 : tensor<32x32x3x3xf32>
    %v790 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v791 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v792 = stablehlo.multiply %v790, %W7v : tensor<32x32x3x3xf32>
    %v793 = stablehlo.multiply %v758, %v758 : tensor<32x32x3x3xf32>
    %v794 = stablehlo.multiply %v791, %v793 : tensor<32x32x3x3xf32>
    %v795 = stablehlo.add %v792, %v794 : tensor<32x32x3x3xf32>
    %v796 = stablehlo.reshape %v152 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v797 = stablehlo.constant dense<0.0> : tensor<f32>
    %v798 = stablehlo.reduce(%v796 init: %v797) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v799 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v800 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v801 = stablehlo.multiply %v799, %cb7m : tensor<32xf32>
    %v802 = stablehlo.multiply %v800, %v798 : tensor<32xf32>
    %v803 = stablehlo.add %v801, %v802 : tensor<32xf32>
    %v804 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v805 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v806 = stablehlo.multiply %v804, %cb7v : tensor<32xf32>
    %v807 = stablehlo.multiply %v798, %v798 : tensor<32xf32>
    %v808 = stablehlo.multiply %v805, %v807 : tensor<32xf32>
    %v809 = stablehlo.add %v806, %v808 : tensor<32xf32>
    %v810 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v811 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v812 = stablehlo.divide %v803, %v810 : tensor<32xf32>
    %v813 = stablehlo.divide %v809, %v811 : tensor<32xf32>
    %v814 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v815 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v816 = stablehlo.sqrt %v813 : tensor<32xf32>
    %v817 = stablehlo.add %v816, %v815 : tensor<32xf32>
    %v818 = stablehlo.divide %v812, %v817 : tensor<32xf32>
    %v819 = stablehlo.multiply %v814, %v818 : tensor<32xf32>
    %v820 = stablehlo.subtract %cb7, %v819 : tensor<32xf32>
    %v821 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v822 = stablehlo.multiply %v821, %v814 : tensor<32xf32>
    %v823 = stablehlo.multiply %v822, %cb7 : tensor<32xf32>
    %v824 = stablehlo.subtract %v820, %v823 : tensor<32xf32>
    %v825 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v826 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v827 = stablehlo.multiply %v825, %cb7m : tensor<32xf32>
    %v828 = stablehlo.multiply %v826, %v798 : tensor<32xf32>
    %v829 = stablehlo.add %v827, %v828 : tensor<32xf32>
    %v830 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v831 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v832 = stablehlo.multiply %v830, %cb7v : tensor<32xf32>
    %v833 = stablehlo.multiply %v798, %v798 : tensor<32xf32>
    %v834 = stablehlo.multiply %v831, %v833 : tensor<32xf32>
    %v835 = stablehlo.add %v832, %v834 : tensor<32xf32>
    %v836 = stablehlo.reshape %v81 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v837 = stablehlo.reshape %v141 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v838 = stablehlo.transpose %v836, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v839 = stablehlo.transpose %v837, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v840 = stablehlo.convert %v838 : (tensor<32x128x4x4xf32>) -> tensor<32x128x4x4xbf16>
    %v841 = stablehlo.convert %v839 : (tensor<32x128x4x4xf32>) -> tensor<32x128x4x4xbf16>
    %v842 = stablehlo.convolution(%v840, %v841)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xbf16>, tensor<32x128x4x4xbf16>) -> tensor<32x32x3x3xbf16>
    %v843 = stablehlo.convert %v842 : (tensor<32x32x3x3xbf16>) -> tensor<32x32x3x3xf32>
    %v844 = stablehlo.transpose %v843, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v845 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v846 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v847 = stablehlo.multiply %v845, %W8m : tensor<32x32x3x3xf32>
    %v848 = stablehlo.multiply %v846, %v844 : tensor<32x32x3x3xf32>
    %v849 = stablehlo.add %v847, %v848 : tensor<32x32x3x3xf32>
    %v850 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v851 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v852 = stablehlo.multiply %v850, %W8v : tensor<32x32x3x3xf32>
    %v853 = stablehlo.multiply %v844, %v844 : tensor<32x32x3x3xf32>
    %v854 = stablehlo.multiply %v851, %v853 : tensor<32x32x3x3xf32>
    %v855 = stablehlo.add %v852, %v854 : tensor<32x32x3x3xf32>
    %v856 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v857 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v858 = stablehlo.divide %v849, %v856 : tensor<32x32x3x3xf32>
    %v859 = stablehlo.divide %v855, %v857 : tensor<32x32x3x3xf32>
    %v860 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v861 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v862 = stablehlo.sqrt %v859 : tensor<32x32x3x3xf32>
    %v863 = stablehlo.add %v862, %v861 : tensor<32x32x3x3xf32>
    %v864 = stablehlo.divide %v858, %v863 : tensor<32x32x3x3xf32>
    %v865 = stablehlo.multiply %v860, %v864 : tensor<32x32x3x3xf32>
    %v866 = stablehlo.subtract %W8, %v865 : tensor<32x32x3x3xf32>
    %v867 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v868 = stablehlo.multiply %v867, %v860 : tensor<32x32x3x3xf32>
    %v869 = stablehlo.multiply %v868, %W8 : tensor<32x32x3x3xf32>
    %v870 = stablehlo.subtract %v866, %v869 : tensor<32x32x3x3xf32>
    %v871 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v872 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v873 = stablehlo.multiply %v871, %W8m : tensor<32x32x3x3xf32>
    %v874 = stablehlo.multiply %v872, %v844 : tensor<32x32x3x3xf32>
    %v875 = stablehlo.add %v873, %v874 : tensor<32x32x3x3xf32>
    %v876 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v877 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v878 = stablehlo.multiply %v876, %W8v : tensor<32x32x3x3xf32>
    %v879 = stablehlo.multiply %v844, %v844 : tensor<32x32x3x3xf32>
    %v880 = stablehlo.multiply %v877, %v879 : tensor<32x32x3x3xf32>
    %v881 = stablehlo.add %v878, %v880 : tensor<32x32x3x3xf32>
    %v882 = stablehlo.reshape %v141 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v883 = stablehlo.constant dense<0.0> : tensor<f32>
    %v884 = stablehlo.reduce(%v882 init: %v883) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v885 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v886 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v887 = stablehlo.multiply %v885, %cb8m : tensor<32xf32>
    %v888 = stablehlo.multiply %v886, %v884 : tensor<32xf32>
    %v889 = stablehlo.add %v887, %v888 : tensor<32xf32>
    %v890 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v891 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v892 = stablehlo.multiply %v890, %cb8v : tensor<32xf32>
    %v893 = stablehlo.multiply %v884, %v884 : tensor<32xf32>
    %v894 = stablehlo.multiply %v891, %v893 : tensor<32xf32>
    %v895 = stablehlo.add %v892, %v894 : tensor<32xf32>
    %v896 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v897 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v898 = stablehlo.divide %v889, %v896 : tensor<32xf32>
    %v899 = stablehlo.divide %v895, %v897 : tensor<32xf32>
    %v900 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v901 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v902 = stablehlo.sqrt %v899 : tensor<32xf32>
    %v903 = stablehlo.add %v902, %v901 : tensor<32xf32>
    %v904 = stablehlo.divide %v898, %v903 : tensor<32xf32>
    %v905 = stablehlo.multiply %v900, %v904 : tensor<32xf32>
    %v906 = stablehlo.subtract %cb8, %v905 : tensor<32xf32>
    %v907 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v908 = stablehlo.multiply %v907, %v900 : tensor<32xf32>
    %v909 = stablehlo.multiply %v908, %cb8 : tensor<32xf32>
    %v910 = stablehlo.subtract %v906, %v909 : tensor<32xf32>
    %v911 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v912 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v913 = stablehlo.multiply %v911, %cb8m : tensor<32xf32>
    %v914 = stablehlo.multiply %v912, %v884 : tensor<32xf32>
    %v915 = stablehlo.add %v913, %v914 : tensor<32xf32>
    %v916 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v917 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v918 = stablehlo.multiply %v916, %cb8v : tensor<32xf32>
    %v919 = stablehlo.multiply %v884, %v884 : tensor<32xf32>
    %v920 = stablehlo.multiply %v917, %v919 : tensor<32xf32>
    %v921 = stablehlo.add %v918, %v920 : tensor<32xf32>
    %v922 = stablehlo.dot_general %v95, %v130, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %v923 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v924 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v925 = stablehlo.multiply %v923, %W9m : tensor<128x64xf32>
    %v926 = stablehlo.multiply %v924, %v922 : tensor<128x64xf32>
    %v927 = stablehlo.add %v925, %v926 : tensor<128x64xf32>
    %v928 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v929 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v930 = stablehlo.multiply %v928, %W9v : tensor<128x64xf32>
    %v931 = stablehlo.multiply %v922, %v922 : tensor<128x64xf32>
    %v932 = stablehlo.multiply %v929, %v931 : tensor<128x64xf32>
    %v933 = stablehlo.add %v930, %v932 : tensor<128x64xf32>
    %v934 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v935 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v936 = stablehlo.divide %v927, %v934 : tensor<128x64xf32>
    %v937 = stablehlo.divide %v933, %v935 : tensor<128x64xf32>
    %v938 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v939 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v940 = stablehlo.sqrt %v937 : tensor<128x64xf32>
    %v941 = stablehlo.add %v940, %v939 : tensor<128x64xf32>
    %v942 = stablehlo.divide %v936, %v941 : tensor<128x64xf32>
    %v943 = stablehlo.multiply %v938, %v942 : tensor<128x64xf32>
    %v944 = stablehlo.subtract %W9, %v943 : tensor<128x64xf32>
    %v945 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v946 = stablehlo.multiply %v945, %v938 : tensor<128x64xf32>
    %v947 = stablehlo.multiply %v946, %W9 : tensor<128x64xf32>
    %v948 = stablehlo.subtract %v944, %v947 : tensor<128x64xf32>
    %v949 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v950 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v951 = stablehlo.multiply %v949, %W9m : tensor<128x64xf32>
    %v952 = stablehlo.multiply %v950, %v922 : tensor<128x64xf32>
    %v953 = stablehlo.add %v951, %v952 : tensor<128x64xf32>
    %v954 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v955 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v956 = stablehlo.multiply %v954, %W9v : tensor<128x64xf32>
    %v957 = stablehlo.multiply %v922, %v922 : tensor<128x64xf32>
    %v958 = stablehlo.multiply %v955, %v957 : tensor<128x64xf32>
    %v959 = stablehlo.add %v956, %v958 : tensor<128x64xf32>
    %v960 = stablehlo.constant dense<0.0> : tensor<f32>
    %v961 = stablehlo.reduce(%v130 init: %v960) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v962 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v963 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v964 = stablehlo.multiply %v962, %b9m : tensor<64xf32>
    %v965 = stablehlo.multiply %v963, %v961 : tensor<64xf32>
    %v966 = stablehlo.add %v964, %v965 : tensor<64xf32>
    %v967 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v968 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v969 = stablehlo.multiply %v967, %b9v : tensor<64xf32>
    %v970 = stablehlo.multiply %v961, %v961 : tensor<64xf32>
    %v971 = stablehlo.multiply %v968, %v970 : tensor<64xf32>
    %v972 = stablehlo.add %v969, %v971 : tensor<64xf32>
    %v973 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v974 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v975 = stablehlo.divide %v966, %v973 : tensor<64xf32>
    %v976 = stablehlo.divide %v972, %v974 : tensor<64xf32>
    %v977 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v978 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v979 = stablehlo.sqrt %v976 : tensor<64xf32>
    %v980 = stablehlo.add %v979, %v978 : tensor<64xf32>
    %v981 = stablehlo.divide %v975, %v980 : tensor<64xf32>
    %v982 = stablehlo.multiply %v977, %v981 : tensor<64xf32>
    %v983 = stablehlo.subtract %b9, %v982 : tensor<64xf32>
    %v984 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v985 = stablehlo.multiply %v984, %v977 : tensor<64xf32>
    %v986 = stablehlo.multiply %v985, %b9 : tensor<64xf32>
    %v987 = stablehlo.subtract %v983, %v986 : tensor<64xf32>
    %v988 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v989 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v990 = stablehlo.multiply %v988, %b9m : tensor<64xf32>
    %v991 = stablehlo.multiply %v989, %v961 : tensor<64xf32>
    %v992 = stablehlo.add %v990, %v991 : tensor<64xf32>
    %v993 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v994 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v995 = stablehlo.multiply %v993, %b9v : tensor<64xf32>
    %v996 = stablehlo.multiply %v961, %v961 : tensor<64xf32>
    %v997 = stablehlo.multiply %v994, %v996 : tensor<64xf32>
    %v998 = stablehlo.add %v995, %v997 : tensor<64xf32>
    %v999 = stablehlo.dot_general %v100, %v124, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<64x64xf32>
    %v1000 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1001 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1002 = stablehlo.multiply %v1000, %Wam : tensor<64x64xf32>
    %v1003 = stablehlo.multiply %v1001, %v999 : tensor<64x64xf32>
    %v1004 = stablehlo.add %v1002, %v1003 : tensor<64x64xf32>
    %v1005 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1006 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1007 = stablehlo.multiply %v1005, %Wav : tensor<64x64xf32>
    %v1008 = stablehlo.multiply %v999, %v999 : tensor<64x64xf32>
    %v1009 = stablehlo.multiply %v1006, %v1008 : tensor<64x64xf32>
    %v1010 = stablehlo.add %v1007, %v1009 : tensor<64x64xf32>
    %v1011 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1012 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1013 = stablehlo.divide %v1004, %v1011 : tensor<64x64xf32>
    %v1014 = stablehlo.divide %v1010, %v1012 : tensor<64x64xf32>
    %v1015 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1016 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1017 = stablehlo.sqrt %v1014 : tensor<64x64xf32>
    %v1018 = stablehlo.add %v1017, %v1016 : tensor<64x64xf32>
    %v1019 = stablehlo.divide %v1013, %v1018 : tensor<64x64xf32>
    %v1020 = stablehlo.multiply %v1015, %v1019 : tensor<64x64xf32>
    %v1021 = stablehlo.subtract %Wa, %v1020 : tensor<64x64xf32>
    %v1022 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1023 = stablehlo.multiply %v1022, %v1015 : tensor<64x64xf32>
    %v1024 = stablehlo.multiply %v1023, %Wa : tensor<64x64xf32>
    %v1025 = stablehlo.subtract %v1021, %v1024 : tensor<64x64xf32>
    %v1026 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1027 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1028 = stablehlo.multiply %v1026, %Wam : tensor<64x64xf32>
    %v1029 = stablehlo.multiply %v1027, %v999 : tensor<64x64xf32>
    %v1030 = stablehlo.add %v1028, %v1029 : tensor<64x64xf32>
    %v1031 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1032 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1033 = stablehlo.multiply %v1031, %Wav : tensor<64x64xf32>
    %v1034 = stablehlo.multiply %v999, %v999 : tensor<64x64xf32>
    %v1035 = stablehlo.multiply %v1032, %v1034 : tensor<64x64xf32>
    %v1036 = stablehlo.add %v1033, %v1035 : tensor<64x64xf32>
    %v1037 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1038 = stablehlo.reduce(%v124 init: %v1037) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v1039 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1040 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1041 = stablehlo.multiply %v1039, %bam : tensor<64xf32>
    %v1042 = stablehlo.multiply %v1040, %v1038 : tensor<64xf32>
    %v1043 = stablehlo.add %v1041, %v1042 : tensor<64xf32>
    %v1044 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1045 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1046 = stablehlo.multiply %v1044, %bav : tensor<64xf32>
    %v1047 = stablehlo.multiply %v1038, %v1038 : tensor<64xf32>
    %v1048 = stablehlo.multiply %v1045, %v1047 : tensor<64xf32>
    %v1049 = stablehlo.add %v1046, %v1048 : tensor<64xf32>
    %v1050 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1051 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1052 = stablehlo.divide %v1043, %v1050 : tensor<64xf32>
    %v1053 = stablehlo.divide %v1049, %v1051 : tensor<64xf32>
    %v1054 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1055 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1056 = stablehlo.sqrt %v1053 : tensor<64xf32>
    %v1057 = stablehlo.add %v1056, %v1055 : tensor<64xf32>
    %v1058 = stablehlo.divide %v1052, %v1057 : tensor<64xf32>
    %v1059 = stablehlo.multiply %v1054, %v1058 : tensor<64xf32>
    %v1060 = stablehlo.subtract %ba, %v1059 : tensor<64xf32>
    %v1061 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1062 = stablehlo.multiply %v1061, %v1054 : tensor<64xf32>
    %v1063 = stablehlo.multiply %v1062, %ba : tensor<64xf32>
    %v1064 = stablehlo.subtract %v1060, %v1063 : tensor<64xf32>
    %v1065 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1066 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1067 = stablehlo.multiply %v1065, %bam : tensor<64xf32>
    %v1068 = stablehlo.multiply %v1066, %v1038 : tensor<64xf32>
    %v1069 = stablehlo.add %v1067, %v1068 : tensor<64xf32>
    %v1070 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1071 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1072 = stablehlo.multiply %v1070, %bav : tensor<64xf32>
    %v1073 = stablehlo.multiply %v1038, %v1038 : tensor<64xf32>
    %v1074 = stablehlo.multiply %v1071, %v1073 : tensor<64xf32>
    %v1075 = stablehlo.add %v1072, %v1074 : tensor<64xf32>
    %v1076 = stablehlo.dot_general %v105, %v118, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x10xf32>) -> tensor<64x10xf32>
    %v1077 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1078 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1079 = stablehlo.multiply %v1077, %Wbm : tensor<64x10xf32>
    %v1080 = stablehlo.multiply %v1078, %v1076 : tensor<64x10xf32>
    %v1081 = stablehlo.add %v1079, %v1080 : tensor<64x10xf32>
    %v1082 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1083 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1084 = stablehlo.multiply %v1082, %Wbv : tensor<64x10xf32>
    %v1085 = stablehlo.multiply %v1076, %v1076 : tensor<64x10xf32>
    %v1086 = stablehlo.multiply %v1083, %v1085 : tensor<64x10xf32>
    %v1087 = stablehlo.add %v1084, %v1086 : tensor<64x10xf32>
    %v1088 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1089 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1090 = stablehlo.divide %v1081, %v1088 : tensor<64x10xf32>
    %v1091 = stablehlo.divide %v1087, %v1089 : tensor<64x10xf32>
    %v1092 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1093 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1094 = stablehlo.sqrt %v1091 : tensor<64x10xf32>
    %v1095 = stablehlo.add %v1094, %v1093 : tensor<64x10xf32>
    %v1096 = stablehlo.divide %v1090, %v1095 : tensor<64x10xf32>
    %v1097 = stablehlo.multiply %v1092, %v1096 : tensor<64x10xf32>
    %v1098 = stablehlo.subtract %Wb, %v1097 : tensor<64x10xf32>
    %v1099 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1100 = stablehlo.multiply %v1099, %v1092 : tensor<64x10xf32>
    %v1101 = stablehlo.multiply %v1100, %Wb : tensor<64x10xf32>
    %v1102 = stablehlo.subtract %v1098, %v1101 : tensor<64x10xf32>
    %v1103 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1104 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1105 = stablehlo.multiply %v1103, %Wbm : tensor<64x10xf32>
    %v1106 = stablehlo.multiply %v1104, %v1076 : tensor<64x10xf32>
    %v1107 = stablehlo.add %v1105, %v1106 : tensor<64x10xf32>
    %v1108 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1109 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1110 = stablehlo.multiply %v1108, %Wbv : tensor<64x10xf32>
    %v1111 = stablehlo.multiply %v1076, %v1076 : tensor<64x10xf32>
    %v1112 = stablehlo.multiply %v1109, %v1111 : tensor<64x10xf32>
    %v1113 = stablehlo.add %v1110, %v1112 : tensor<64x10xf32>
    %v1114 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1115 = stablehlo.reduce(%v118 init: %v1114) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1116 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1117 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1118 = stablehlo.multiply %v1116, %bbm : tensor<10xf32>
    %v1119 = stablehlo.multiply %v1117, %v1115 : tensor<10xf32>
    %v1120 = stablehlo.add %v1118, %v1119 : tensor<10xf32>
    %v1121 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1122 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1123 = stablehlo.multiply %v1121, %bbv : tensor<10xf32>
    %v1124 = stablehlo.multiply %v1115, %v1115 : tensor<10xf32>
    %v1125 = stablehlo.multiply %v1122, %v1124 : tensor<10xf32>
    %v1126 = stablehlo.add %v1123, %v1125 : tensor<10xf32>
    %v1127 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1128 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1129 = stablehlo.divide %v1120, %v1127 : tensor<10xf32>
    %v1130 = stablehlo.divide %v1126, %v1128 : tensor<10xf32>
    %v1131 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1132 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1133 = stablehlo.sqrt %v1130 : tensor<10xf32>
    %v1134 = stablehlo.add %v1133, %v1132 : tensor<10xf32>
    %v1135 = stablehlo.divide %v1129, %v1134 : tensor<10xf32>
    %v1136 = stablehlo.multiply %v1131, %v1135 : tensor<10xf32>
    %v1137 = stablehlo.subtract %bb, %v1136 : tensor<10xf32>
    %v1138 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1139 = stablehlo.multiply %v1138, %v1131 : tensor<10xf32>
    %v1140 = stablehlo.multiply %v1139, %bb : tensor<10xf32>
    %v1141 = stablehlo.subtract %v1137, %v1140 : tensor<10xf32>
    %v1142 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1143 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1144 = stablehlo.multiply %v1142, %bbm : tensor<10xf32>
    %v1145 = stablehlo.multiply %v1143, %v1115 : tensor<10xf32>
    %v1146 = stablehlo.add %v1144, %v1145 : tensor<10xf32>
    %v1147 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1148 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1149 = stablehlo.multiply %v1147, %bbv : tensor<10xf32>
    %v1150 = stablehlo.multiply %v1115, %v1115 : tensor<10xf32>
    %v1151 = stablehlo.multiply %v1148, %v1150 : tensor<10xf32>
    %v1152 = stablehlo.add %v1149, %v1151 : tensor<10xf32>
    return %v268, %v308, %v354, %v394, %v440, %v480, %v526, %v566, %v612, %v652, %v698, %v738, %v784, %v824, %v870, %v910, %v948, %v987, %v1025, %v1064, %v1102, %v1141, %v273, %v313, %v359, %v399, %v445, %v485, %v531, %v571, %v617, %v657, %v703, %v743, %v789, %v829, %v875, %v915, %v953, %v992, %v1030, %v1069, %v1107, %v1146, %v279, %v319, %v365, %v405, %v451, %v491, %v537, %v577, %v623, %v663, %v709, %v749, %v795, %v835, %v881, %v921, %v959, %v998, %v1036, %v1075, %v1113, %v1152, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
