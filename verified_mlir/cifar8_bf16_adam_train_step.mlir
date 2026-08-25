module @m {
  func.func @cifar8_bf16_adam_train_step(%x: tensor<128x3072xf32>, %W1: tensor<16x3x3x3xf32>, %cb1: tensor<16xf32>, %W2: tensor<16x16x3x3xf32>, %cb2: tensor<16xf32>, %W3: tensor<16x16x3x3xf32>, %cb3: tensor<16xf32>, %W4: tensor<16x16x3x3xf32>, %cb4: tensor<16xf32>, %W5: tensor<32x16x3x3xf32>, %cb5: tensor<32xf32>, %W6: tensor<32x32x3x3xf32>, %cb6: tensor<32xf32>, %W7: tensor<32x32x3x3xf32>, %cb7: tensor<32xf32>, %W8: tensor<32x32x3x3xf32>, %cb8: tensor<32xf32>, %W9: tensor<128x64xf32>, %b9: tensor<64xf32>, %Wa: tensor<64x64xf32>, %ba: tensor<64xf32>, %Wb: tensor<64x10xf32>, %bb: tensor<10xf32>, %W1m: tensor<16x3x3x3xf32>, %cb1m: tensor<16xf32>, %W2m: tensor<16x16x3x3xf32>, %cb2m: tensor<16xf32>, %W3m: tensor<16x16x3x3xf32>, %cb3m: tensor<16xf32>, %W4m: tensor<16x16x3x3xf32>, %cb4m: tensor<16xf32>, %W5m: tensor<32x16x3x3xf32>, %cb5m: tensor<32xf32>, %W6m: tensor<32x32x3x3xf32>, %cb6m: tensor<32xf32>, %W7m: tensor<32x32x3x3xf32>, %cb7m: tensor<32xf32>, %W8m: tensor<32x32x3x3xf32>, %cb8m: tensor<32xf32>, %W9m: tensor<128x64xf32>, %b9m: tensor<64xf32>, %Wam: tensor<64x64xf32>, %bam: tensor<64xf32>, %Wbm: tensor<64x10xf32>, %bbm: tensor<10xf32>, %W1v: tensor<16x3x3x3xf32>, %cb1v: tensor<16xf32>, %W2v: tensor<16x16x3x3xf32>, %cb2v: tensor<16xf32>, %W3v: tensor<16x16x3x3xf32>, %cb3v: tensor<16xf32>, %W4v: tensor<16x16x3x3xf32>, %cb4v: tensor<16xf32>, %W5v: tensor<32x16x3x3xf32>, %cb5v: tensor<32xf32>, %W6v: tensor<32x32x3x3xf32>, %cb6v: tensor<32xf32>, %W7v: tensor<32x32x3x3xf32>, %cb7v: tensor<32xf32>, %W8v: tensor<32x32x3x3xf32>, %cb8v: tensor<32xf32>, %W9v: tensor<128x64xf32>, %b9v: tensor<64xf32>, %Wav: tensor<64x64xf32>, %bav: tensor<64xf32>, %Wbv: tensor<64x10xf32>, %bbv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: tensor<128x10xf32>) -> (tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>) {
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
    %v109 = stablehlo.exponential %v108 : tensor<128x10xf32>
    %v110 = stablehlo.constant dense<0.0> : tensor<f32>
    %v111 = stablehlo.reduce(%v109 init: %v110) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %v112 = stablehlo.broadcast_in_dim %v111, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %v113 = stablehlo.divide %v109, %v112 : tensor<128x10xf32>
    %v114 = stablehlo.subtract %v113, %onehot : tensor<128x10xf32>
    %v115 = stablehlo.constant dense<0.0078125> : tensor<128x10xf32>
    %v116 = stablehlo.multiply %v114, %v115 : tensor<128x10xf32>
    // ── report-only scalar loss (NOT pretty(AST): the kit has no rank-0 loss op; it
    //    feeds no parameter, only the driver's progress line) ──
    %llog = stablehlo.log %v113 : tensor<128x10xf32>
    %ohll = stablehlo.multiply %onehot, %llog : tensor<128x10xf32>
    %csum = stablehlo.reduce(%ohll init: %lzero) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<f32>
    %cneg = stablehlo.negate %csum : tensor<f32>
    %lbf = stablehlo.constant dense<128.0> : tensor<f32>
    %loss = stablehlo.divide %cneg, %lbf : tensor<f32>
    %v117 = stablehlo.dot_general %v116, %Wb, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x10xf32>, tensor<64x10xf32>) -> tensor<128x64xf32>
    %v118 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v119 = stablehlo.compare GT, %v103, %v118 : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xi1>
    %v120 = stablehlo.select %v119, %v117, %v118 : tensor<128x64xi1>, tensor<128x64xf32>
    %v121 = stablehlo.dot_general %v120, %Wa, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
    %v122 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v123 = stablehlo.compare GT, %v98, %v122 : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xi1>
    %v124 = stablehlo.select %v123, %v121, %v122 : tensor<128x64xi1>, tensor<128x64xf32>
    %v125 = stablehlo.dot_general %v124, %W9, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x128xf32>
    %v126 = stablehlo.reshape %v91 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
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
    %v131 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v132 = stablehlo.compare GT, %v89, %v131 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v133 = stablehlo.select %v132, %v130, %v131 : tensor<128x512xi1>, tensor<128x512xf32>
    %v134 = stablehlo.reshape %v133 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v135 = stablehlo.transpose %W8, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v136 = stablehlo.reverse %v135, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v137 = stablehlo.convolution(%v134, %v136)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v138 = stablehlo.reshape %v137 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v139 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v140 = stablehlo.compare GT, %v79, %v139 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v141 = stablehlo.select %v140, %v138, %v139 : tensor<128x512xi1>, tensor<128x512xf32>
    %v142 = stablehlo.reshape %v141 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v143 = stablehlo.transpose %W7, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v144 = stablehlo.reverse %v143, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v145 = stablehlo.convolution(%v142, %v144)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v146 = stablehlo.reshape %v145 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v147 = stablehlo.reshape %v67 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v148 = stablehlo.reshape %v146 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v149 = stablehlo.constant dense<0.0> : tensor<f32>
    %v150 = "stablehlo.select_and_scatter"(%v147, %v148, %v149) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x8x8xf32>, tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32x8x8xf32>
    %v151 = stablehlo.reshape %v150 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v152 = stablehlo.constant dense<0.0> : tensor<128x2048xf32>
    %v153 = stablehlo.compare GT, %v65, %v152 : (tensor<128x2048xf32>, tensor<128x2048xf32>) -> tensor<128x2048xi1>
    %v154 = stablehlo.select %v153, %v151, %v152 : tensor<128x2048xi1>, tensor<128x2048xf32>
    %v155 = stablehlo.reshape %v154 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v156 = stablehlo.transpose %W6, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v157 = stablehlo.reverse %v156, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v158 = stablehlo.convolution(%v155, %v157)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x8x8xf32>
    %v159 = stablehlo.reshape %v158 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v160 = stablehlo.constant dense<0.0> : tensor<128x2048xf32>
    %v161 = stablehlo.compare GT, %v55, %v160 : (tensor<128x2048xf32>, tensor<128x2048xf32>) -> tensor<128x2048xi1>
    %v162 = stablehlo.select %v161, %v159, %v160 : tensor<128x2048xi1>, tensor<128x2048xf32>
    %v163 = stablehlo.reshape %v162 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v164 = stablehlo.transpose %W5, dims = [1, 0, 2, 3] : (tensor<32x16x3x3xf32>) -> tensor<16x32x3x3xf32>
    %v165 = stablehlo.reverse %v164, dims = [2, 3] : tensor<16x32x3x3xf32>
    %v166 = stablehlo.convolution(%v163, %v165)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<16x32x3x3xf32>) -> tensor<128x16x8x8xf32>
    %v167 = stablehlo.reshape %v166 : (tensor<128x16x8x8xf32>) -> tensor<128x1024xf32>
    %v168 = stablehlo.reshape %v43 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v169 = stablehlo.reshape %v167 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v170 = stablehlo.constant dense<0.0> : tensor<f32>
    %v171 = "stablehlo.select_and_scatter"(%v168, %v169, %v170) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x16x16xf32>, tensor<128x16x8x8xf32>, tensor<f32>) -> tensor<128x16x16x16xf32>
    %v172 = stablehlo.reshape %v171 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v173 = stablehlo.constant dense<0.0> : tensor<128x4096xf32>
    %v174 = stablehlo.compare GT, %v41, %v173 : (tensor<128x4096xf32>, tensor<128x4096xf32>) -> tensor<128x4096xi1>
    %v175 = stablehlo.select %v174, %v172, %v173 : tensor<128x4096xi1>, tensor<128x4096xf32>
    %v176 = stablehlo.reshape %v175 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v177 = stablehlo.transpose %W4, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v178 = stablehlo.reverse %v177, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v179 = stablehlo.convolution(%v176, %v178)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v180 = stablehlo.reshape %v179 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v181 = stablehlo.constant dense<0.0> : tensor<128x4096xf32>
    %v182 = stablehlo.compare GT, %v31, %v181 : (tensor<128x4096xf32>, tensor<128x4096xf32>) -> tensor<128x4096xi1>
    %v183 = stablehlo.select %v182, %v180, %v181 : tensor<128x4096xi1>, tensor<128x4096xf32>
    %v184 = stablehlo.reshape %v183 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v185 = stablehlo.transpose %W3, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v186 = stablehlo.reverse %v185, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v187 = stablehlo.convolution(%v184, %v186)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v188 = stablehlo.reshape %v187 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v189 = stablehlo.reshape %v19 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v190 = stablehlo.reshape %v188 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v191 = stablehlo.constant dense<0.0> : tensor<f32>
    %v192 = "stablehlo.select_and_scatter"(%v189, %v190, %v191) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x32x32xf32>, tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x32x32xf32>
    %v193 = stablehlo.reshape %v192 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v194 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v195 = stablehlo.compare GT, %v17, %v194 : (tensor<128x16384xf32>, tensor<128x16384xf32>) -> tensor<128x16384xi1>
    %v196 = stablehlo.select %v195, %v193, %v194 : tensor<128x16384xi1>, tensor<128x16384xf32>
    %v197 = stablehlo.reshape %v196 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v198 = stablehlo.transpose %W2, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v199 = stablehlo.reverse %v198, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v200 = stablehlo.convolution(%v197, %v199)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x32x32xf32>
    %v201 = stablehlo.reshape %v200 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v202 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v203 = stablehlo.compare GT, %v7, %v202 : (tensor<128x16384xf32>, tensor<128x16384xf32>) -> tensor<128x16384xi1>
    %v204 = stablehlo.select %v203, %v201, %v202 : tensor<128x16384xi1>, tensor<128x16384xf32>
    %v205 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v206 = stablehlo.reshape %v204 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v207 = stablehlo.transpose %v205, dims = [1, 0, 2, 3] : (tensor<128x3x32x32xf32>) -> tensor<3x128x32x32xf32>
    %v208 = stablehlo.transpose %v206, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v209 = stablehlo.convolution(%v207, %v208)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<3x16x3x3xf32>
    %v210 = stablehlo.transpose %v209, dims = [1, 0, 2, 3] : (tensor<3x16x3x3xf32>) -> tensor<16x3x3x3xf32>
    %v211 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v212 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v213 = stablehlo.multiply %v211, %W1m : tensor<16x3x3x3xf32>
    %v214 = stablehlo.multiply %v212, %v210 : tensor<16x3x3x3xf32>
    %v215 = stablehlo.add %v213, %v214 : tensor<16x3x3x3xf32>
    %v216 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v217 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v218 = stablehlo.multiply %v216, %W1v : tensor<16x3x3x3xf32>
    %v219 = stablehlo.multiply %v210, %v210 : tensor<16x3x3x3xf32>
    %v220 = stablehlo.multiply %v217, %v219 : tensor<16x3x3x3xf32>
    %v221 = stablehlo.add %v218, %v220 : tensor<16x3x3x3xf32>
    %v222 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v223 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v224 = stablehlo.divide %v215, %v222 : tensor<16x3x3x3xf32>
    %v225 = stablehlo.divide %v221, %v223 : tensor<16x3x3x3xf32>
    %v226 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v227 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v228 = stablehlo.sqrt %v225 : tensor<16x3x3x3xf32>
    %v229 = stablehlo.add %v228, %v227 : tensor<16x3x3x3xf32>
    %v230 = stablehlo.divide %v224, %v229 : tensor<16x3x3x3xf32>
    %v231 = stablehlo.multiply %v226, %v230 : tensor<16x3x3x3xf32>
    %v232 = stablehlo.subtract %W1, %v231 : tensor<16x3x3x3xf32>
    %v233 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v234 = stablehlo.multiply %v233, %v226 : tensor<16x3x3x3xf32>
    %v235 = stablehlo.multiply %v234, %W1 : tensor<16x3x3x3xf32>
    %v236 = stablehlo.subtract %v232, %v235 : tensor<16x3x3x3xf32>
    %v237 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v238 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v239 = stablehlo.multiply %v237, %W1m : tensor<16x3x3x3xf32>
    %v240 = stablehlo.multiply %v238, %v210 : tensor<16x3x3x3xf32>
    %v241 = stablehlo.add %v239, %v240 : tensor<16x3x3x3xf32>
    %v242 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v243 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v244 = stablehlo.multiply %v242, %W1v : tensor<16x3x3x3xf32>
    %v245 = stablehlo.multiply %v210, %v210 : tensor<16x3x3x3xf32>
    %v246 = stablehlo.multiply %v243, %v245 : tensor<16x3x3x3xf32>
    %v247 = stablehlo.add %v244, %v246 : tensor<16x3x3x3xf32>
    %v248 = stablehlo.reshape %v204 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v249 = stablehlo.constant dense<0.0> : tensor<f32>
    %v250 = stablehlo.reduce(%v248 init: %v249) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v251 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v252 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v253 = stablehlo.multiply %v251, %cb1m : tensor<16xf32>
    %v254 = stablehlo.multiply %v252, %v250 : tensor<16xf32>
    %v255 = stablehlo.add %v253, %v254 : tensor<16xf32>
    %v256 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v257 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v258 = stablehlo.multiply %v256, %cb1v : tensor<16xf32>
    %v259 = stablehlo.multiply %v250, %v250 : tensor<16xf32>
    %v260 = stablehlo.multiply %v257, %v259 : tensor<16xf32>
    %v261 = stablehlo.add %v258, %v260 : tensor<16xf32>
    %v262 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v263 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v264 = stablehlo.divide %v255, %v262 : tensor<16xf32>
    %v265 = stablehlo.divide %v261, %v263 : tensor<16xf32>
    %v266 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v267 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v268 = stablehlo.sqrt %v265 : tensor<16xf32>
    %v269 = stablehlo.add %v268, %v267 : tensor<16xf32>
    %v270 = stablehlo.divide %v264, %v269 : tensor<16xf32>
    %v271 = stablehlo.multiply %v266, %v270 : tensor<16xf32>
    %v272 = stablehlo.subtract %cb1, %v271 : tensor<16xf32>
    %v273 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v274 = stablehlo.multiply %v273, %v266 : tensor<16xf32>
    %v275 = stablehlo.multiply %v274, %cb1 : tensor<16xf32>
    %v276 = stablehlo.subtract %v272, %v275 : tensor<16xf32>
    %v277 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v278 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v279 = stablehlo.multiply %v277, %cb1m : tensor<16xf32>
    %v280 = stablehlo.multiply %v278, %v250 : tensor<16xf32>
    %v281 = stablehlo.add %v279, %v280 : tensor<16xf32>
    %v282 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v283 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v284 = stablehlo.multiply %v282, %cb1v : tensor<16xf32>
    %v285 = stablehlo.multiply %v250, %v250 : tensor<16xf32>
    %v286 = stablehlo.multiply %v283, %v285 : tensor<16xf32>
    %v287 = stablehlo.add %v284, %v286 : tensor<16xf32>
    %v288 = stablehlo.reshape %v9 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v289 = stablehlo.reshape %v196 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v290 = stablehlo.transpose %v288, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v291 = stablehlo.transpose %v289, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v292 = stablehlo.convolution(%v290, %v291)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x16x3x3xf32>
    %v293 = stablehlo.transpose %v292, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v294 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v295 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v296 = stablehlo.multiply %v294, %W2m : tensor<16x16x3x3xf32>
    %v297 = stablehlo.multiply %v295, %v293 : tensor<16x16x3x3xf32>
    %v298 = stablehlo.add %v296, %v297 : tensor<16x16x3x3xf32>
    %v299 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v300 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v301 = stablehlo.multiply %v299, %W2v : tensor<16x16x3x3xf32>
    %v302 = stablehlo.multiply %v293, %v293 : tensor<16x16x3x3xf32>
    %v303 = stablehlo.multiply %v300, %v302 : tensor<16x16x3x3xf32>
    %v304 = stablehlo.add %v301, %v303 : tensor<16x16x3x3xf32>
    %v305 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v306 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v307 = stablehlo.divide %v298, %v305 : tensor<16x16x3x3xf32>
    %v308 = stablehlo.divide %v304, %v306 : tensor<16x16x3x3xf32>
    %v309 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v310 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v311 = stablehlo.sqrt %v308 : tensor<16x16x3x3xf32>
    %v312 = stablehlo.add %v311, %v310 : tensor<16x16x3x3xf32>
    %v313 = stablehlo.divide %v307, %v312 : tensor<16x16x3x3xf32>
    %v314 = stablehlo.multiply %v309, %v313 : tensor<16x16x3x3xf32>
    %v315 = stablehlo.subtract %W2, %v314 : tensor<16x16x3x3xf32>
    %v316 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v317 = stablehlo.multiply %v316, %v309 : tensor<16x16x3x3xf32>
    %v318 = stablehlo.multiply %v317, %W2 : tensor<16x16x3x3xf32>
    %v319 = stablehlo.subtract %v315, %v318 : tensor<16x16x3x3xf32>
    %v320 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v321 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v322 = stablehlo.multiply %v320, %W2m : tensor<16x16x3x3xf32>
    %v323 = stablehlo.multiply %v321, %v293 : tensor<16x16x3x3xf32>
    %v324 = stablehlo.add %v322, %v323 : tensor<16x16x3x3xf32>
    %v325 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v326 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v327 = stablehlo.multiply %v325, %W2v : tensor<16x16x3x3xf32>
    %v328 = stablehlo.multiply %v293, %v293 : tensor<16x16x3x3xf32>
    %v329 = stablehlo.multiply %v326, %v328 : tensor<16x16x3x3xf32>
    %v330 = stablehlo.add %v327, %v329 : tensor<16x16x3x3xf32>
    %v331 = stablehlo.reshape %v196 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v332 = stablehlo.constant dense<0.0> : tensor<f32>
    %v333 = stablehlo.reduce(%v331 init: %v332) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v334 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v335 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v336 = stablehlo.multiply %v334, %cb2m : tensor<16xf32>
    %v337 = stablehlo.multiply %v335, %v333 : tensor<16xf32>
    %v338 = stablehlo.add %v336, %v337 : tensor<16xf32>
    %v339 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v340 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v341 = stablehlo.multiply %v339, %cb2v : tensor<16xf32>
    %v342 = stablehlo.multiply %v333, %v333 : tensor<16xf32>
    %v343 = stablehlo.multiply %v340, %v342 : tensor<16xf32>
    %v344 = stablehlo.add %v341, %v343 : tensor<16xf32>
    %v345 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v346 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v347 = stablehlo.divide %v338, %v345 : tensor<16xf32>
    %v348 = stablehlo.divide %v344, %v346 : tensor<16xf32>
    %v349 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v350 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v351 = stablehlo.sqrt %v348 : tensor<16xf32>
    %v352 = stablehlo.add %v351, %v350 : tensor<16xf32>
    %v353 = stablehlo.divide %v347, %v352 : tensor<16xf32>
    %v354 = stablehlo.multiply %v349, %v353 : tensor<16xf32>
    %v355 = stablehlo.subtract %cb2, %v354 : tensor<16xf32>
    %v356 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v357 = stablehlo.multiply %v356, %v349 : tensor<16xf32>
    %v358 = stablehlo.multiply %v357, %cb2 : tensor<16xf32>
    %v359 = stablehlo.subtract %v355, %v358 : tensor<16xf32>
    %v360 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v361 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v362 = stablehlo.multiply %v360, %cb2m : tensor<16xf32>
    %v363 = stablehlo.multiply %v361, %v333 : tensor<16xf32>
    %v364 = stablehlo.add %v362, %v363 : tensor<16xf32>
    %v365 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v366 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v367 = stablehlo.multiply %v365, %cb2v : tensor<16xf32>
    %v368 = stablehlo.multiply %v333, %v333 : tensor<16xf32>
    %v369 = stablehlo.multiply %v366, %v368 : tensor<16xf32>
    %v370 = stablehlo.add %v367, %v369 : tensor<16xf32>
    %v371 = stablehlo.reshape %v23 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v372 = stablehlo.reshape %v183 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v373 = stablehlo.transpose %v371, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v374 = stablehlo.transpose %v372, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v375 = stablehlo.convolution(%v373, %v374)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v376 = stablehlo.transpose %v375, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v377 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v378 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v379 = stablehlo.multiply %v377, %W3m : tensor<16x16x3x3xf32>
    %v380 = stablehlo.multiply %v378, %v376 : tensor<16x16x3x3xf32>
    %v381 = stablehlo.add %v379, %v380 : tensor<16x16x3x3xf32>
    %v382 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v383 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v384 = stablehlo.multiply %v382, %W3v : tensor<16x16x3x3xf32>
    %v385 = stablehlo.multiply %v376, %v376 : tensor<16x16x3x3xf32>
    %v386 = stablehlo.multiply %v383, %v385 : tensor<16x16x3x3xf32>
    %v387 = stablehlo.add %v384, %v386 : tensor<16x16x3x3xf32>
    %v388 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v389 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v390 = stablehlo.divide %v381, %v388 : tensor<16x16x3x3xf32>
    %v391 = stablehlo.divide %v387, %v389 : tensor<16x16x3x3xf32>
    %v392 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v393 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v394 = stablehlo.sqrt %v391 : tensor<16x16x3x3xf32>
    %v395 = stablehlo.add %v394, %v393 : tensor<16x16x3x3xf32>
    %v396 = stablehlo.divide %v390, %v395 : tensor<16x16x3x3xf32>
    %v397 = stablehlo.multiply %v392, %v396 : tensor<16x16x3x3xf32>
    %v398 = stablehlo.subtract %W3, %v397 : tensor<16x16x3x3xf32>
    %v399 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v400 = stablehlo.multiply %v399, %v392 : tensor<16x16x3x3xf32>
    %v401 = stablehlo.multiply %v400, %W3 : tensor<16x16x3x3xf32>
    %v402 = stablehlo.subtract %v398, %v401 : tensor<16x16x3x3xf32>
    %v403 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v404 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v405 = stablehlo.multiply %v403, %W3m : tensor<16x16x3x3xf32>
    %v406 = stablehlo.multiply %v404, %v376 : tensor<16x16x3x3xf32>
    %v407 = stablehlo.add %v405, %v406 : tensor<16x16x3x3xf32>
    %v408 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v409 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v410 = stablehlo.multiply %v408, %W3v : tensor<16x16x3x3xf32>
    %v411 = stablehlo.multiply %v376, %v376 : tensor<16x16x3x3xf32>
    %v412 = stablehlo.multiply %v409, %v411 : tensor<16x16x3x3xf32>
    %v413 = stablehlo.add %v410, %v412 : tensor<16x16x3x3xf32>
    %v414 = stablehlo.reshape %v183 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v415 = stablehlo.constant dense<0.0> : tensor<f32>
    %v416 = stablehlo.reduce(%v414 init: %v415) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v417 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v418 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v419 = stablehlo.multiply %v417, %cb3m : tensor<16xf32>
    %v420 = stablehlo.multiply %v418, %v416 : tensor<16xf32>
    %v421 = stablehlo.add %v419, %v420 : tensor<16xf32>
    %v422 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v423 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v424 = stablehlo.multiply %v422, %cb3v : tensor<16xf32>
    %v425 = stablehlo.multiply %v416, %v416 : tensor<16xf32>
    %v426 = stablehlo.multiply %v423, %v425 : tensor<16xf32>
    %v427 = stablehlo.add %v424, %v426 : tensor<16xf32>
    %v428 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v429 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v430 = stablehlo.divide %v421, %v428 : tensor<16xf32>
    %v431 = stablehlo.divide %v427, %v429 : tensor<16xf32>
    %v432 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v433 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v434 = stablehlo.sqrt %v431 : tensor<16xf32>
    %v435 = stablehlo.add %v434, %v433 : tensor<16xf32>
    %v436 = stablehlo.divide %v430, %v435 : tensor<16xf32>
    %v437 = stablehlo.multiply %v432, %v436 : tensor<16xf32>
    %v438 = stablehlo.subtract %cb3, %v437 : tensor<16xf32>
    %v439 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v440 = stablehlo.multiply %v439, %v432 : tensor<16xf32>
    %v441 = stablehlo.multiply %v440, %cb3 : tensor<16xf32>
    %v442 = stablehlo.subtract %v438, %v441 : tensor<16xf32>
    %v443 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v444 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v445 = stablehlo.multiply %v443, %cb3m : tensor<16xf32>
    %v446 = stablehlo.multiply %v444, %v416 : tensor<16xf32>
    %v447 = stablehlo.add %v445, %v446 : tensor<16xf32>
    %v448 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v449 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v450 = stablehlo.multiply %v448, %cb3v : tensor<16xf32>
    %v451 = stablehlo.multiply %v416, %v416 : tensor<16xf32>
    %v452 = stablehlo.multiply %v449, %v451 : tensor<16xf32>
    %v453 = stablehlo.add %v450, %v452 : tensor<16xf32>
    %v454 = stablehlo.reshape %v33 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v455 = stablehlo.reshape %v175 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v456 = stablehlo.transpose %v454, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v457 = stablehlo.transpose %v455, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v458 = stablehlo.convolution(%v456, %v457)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v459 = stablehlo.transpose %v458, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v460 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v461 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v462 = stablehlo.multiply %v460, %W4m : tensor<16x16x3x3xf32>
    %v463 = stablehlo.multiply %v461, %v459 : tensor<16x16x3x3xf32>
    %v464 = stablehlo.add %v462, %v463 : tensor<16x16x3x3xf32>
    %v465 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v466 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v467 = stablehlo.multiply %v465, %W4v : tensor<16x16x3x3xf32>
    %v468 = stablehlo.multiply %v459, %v459 : tensor<16x16x3x3xf32>
    %v469 = stablehlo.multiply %v466, %v468 : tensor<16x16x3x3xf32>
    %v470 = stablehlo.add %v467, %v469 : tensor<16x16x3x3xf32>
    %v471 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v472 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v473 = stablehlo.divide %v464, %v471 : tensor<16x16x3x3xf32>
    %v474 = stablehlo.divide %v470, %v472 : tensor<16x16x3x3xf32>
    %v475 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v476 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v477 = stablehlo.sqrt %v474 : tensor<16x16x3x3xf32>
    %v478 = stablehlo.add %v477, %v476 : tensor<16x16x3x3xf32>
    %v479 = stablehlo.divide %v473, %v478 : tensor<16x16x3x3xf32>
    %v480 = stablehlo.multiply %v475, %v479 : tensor<16x16x3x3xf32>
    %v481 = stablehlo.subtract %W4, %v480 : tensor<16x16x3x3xf32>
    %v482 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v483 = stablehlo.multiply %v482, %v475 : tensor<16x16x3x3xf32>
    %v484 = stablehlo.multiply %v483, %W4 : tensor<16x16x3x3xf32>
    %v485 = stablehlo.subtract %v481, %v484 : tensor<16x16x3x3xf32>
    %v486 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v487 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v488 = stablehlo.multiply %v486, %W4m : tensor<16x16x3x3xf32>
    %v489 = stablehlo.multiply %v487, %v459 : tensor<16x16x3x3xf32>
    %v490 = stablehlo.add %v488, %v489 : tensor<16x16x3x3xf32>
    %v491 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v492 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v493 = stablehlo.multiply %v491, %W4v : tensor<16x16x3x3xf32>
    %v494 = stablehlo.multiply %v459, %v459 : tensor<16x16x3x3xf32>
    %v495 = stablehlo.multiply %v492, %v494 : tensor<16x16x3x3xf32>
    %v496 = stablehlo.add %v493, %v495 : tensor<16x16x3x3xf32>
    %v497 = stablehlo.reshape %v175 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v498 = stablehlo.constant dense<0.0> : tensor<f32>
    %v499 = stablehlo.reduce(%v497 init: %v498) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v500 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v501 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v502 = stablehlo.multiply %v500, %cb4m : tensor<16xf32>
    %v503 = stablehlo.multiply %v501, %v499 : tensor<16xf32>
    %v504 = stablehlo.add %v502, %v503 : tensor<16xf32>
    %v505 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v506 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v507 = stablehlo.multiply %v505, %cb4v : tensor<16xf32>
    %v508 = stablehlo.multiply %v499, %v499 : tensor<16xf32>
    %v509 = stablehlo.multiply %v506, %v508 : tensor<16xf32>
    %v510 = stablehlo.add %v507, %v509 : tensor<16xf32>
    %v511 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v512 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v513 = stablehlo.divide %v504, %v511 : tensor<16xf32>
    %v514 = stablehlo.divide %v510, %v512 : tensor<16xf32>
    %v515 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v516 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v517 = stablehlo.sqrt %v514 : tensor<16xf32>
    %v518 = stablehlo.add %v517, %v516 : tensor<16xf32>
    %v519 = stablehlo.divide %v513, %v518 : tensor<16xf32>
    %v520 = stablehlo.multiply %v515, %v519 : tensor<16xf32>
    %v521 = stablehlo.subtract %cb4, %v520 : tensor<16xf32>
    %v522 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v523 = stablehlo.multiply %v522, %v515 : tensor<16xf32>
    %v524 = stablehlo.multiply %v523, %cb4 : tensor<16xf32>
    %v525 = stablehlo.subtract %v521, %v524 : tensor<16xf32>
    %v526 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v527 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v528 = stablehlo.multiply %v526, %cb4m : tensor<16xf32>
    %v529 = stablehlo.multiply %v527, %v499 : tensor<16xf32>
    %v530 = stablehlo.add %v528, %v529 : tensor<16xf32>
    %v531 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v532 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v533 = stablehlo.multiply %v531, %cb4v : tensor<16xf32>
    %v534 = stablehlo.multiply %v499, %v499 : tensor<16xf32>
    %v535 = stablehlo.multiply %v532, %v534 : tensor<16xf32>
    %v536 = stablehlo.add %v533, %v535 : tensor<16xf32>
    %v537 = stablehlo.reshape %v47 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v538 = stablehlo.reshape %v162 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v539 = stablehlo.transpose %v537, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v540 = stablehlo.transpose %v538, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v541 = stablehlo.convolution(%v539, %v540)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<16x32x3x3xf32>
    %v542 = stablehlo.transpose %v541, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v543 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v544 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v545 = stablehlo.multiply %v543, %W5m : tensor<32x16x3x3xf32>
    %v546 = stablehlo.multiply %v544, %v542 : tensor<32x16x3x3xf32>
    %v547 = stablehlo.add %v545, %v546 : tensor<32x16x3x3xf32>
    %v548 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v549 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v550 = stablehlo.multiply %v548, %W5v : tensor<32x16x3x3xf32>
    %v551 = stablehlo.multiply %v542, %v542 : tensor<32x16x3x3xf32>
    %v552 = stablehlo.multiply %v549, %v551 : tensor<32x16x3x3xf32>
    %v553 = stablehlo.add %v550, %v552 : tensor<32x16x3x3xf32>
    %v554 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v555 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v556 = stablehlo.divide %v547, %v554 : tensor<32x16x3x3xf32>
    %v557 = stablehlo.divide %v553, %v555 : tensor<32x16x3x3xf32>
    %v558 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v559 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v560 = stablehlo.sqrt %v557 : tensor<32x16x3x3xf32>
    %v561 = stablehlo.add %v560, %v559 : tensor<32x16x3x3xf32>
    %v562 = stablehlo.divide %v556, %v561 : tensor<32x16x3x3xf32>
    %v563 = stablehlo.multiply %v558, %v562 : tensor<32x16x3x3xf32>
    %v564 = stablehlo.subtract %W5, %v563 : tensor<32x16x3x3xf32>
    %v565 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v566 = stablehlo.multiply %v565, %v558 : tensor<32x16x3x3xf32>
    %v567 = stablehlo.multiply %v566, %W5 : tensor<32x16x3x3xf32>
    %v568 = stablehlo.subtract %v564, %v567 : tensor<32x16x3x3xf32>
    %v569 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v570 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v571 = stablehlo.multiply %v569, %W5m : tensor<32x16x3x3xf32>
    %v572 = stablehlo.multiply %v570, %v542 : tensor<32x16x3x3xf32>
    %v573 = stablehlo.add %v571, %v572 : tensor<32x16x3x3xf32>
    %v574 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v575 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v576 = stablehlo.multiply %v574, %W5v : tensor<32x16x3x3xf32>
    %v577 = stablehlo.multiply %v542, %v542 : tensor<32x16x3x3xf32>
    %v578 = stablehlo.multiply %v575, %v577 : tensor<32x16x3x3xf32>
    %v579 = stablehlo.add %v576, %v578 : tensor<32x16x3x3xf32>
    %v580 = stablehlo.reshape %v162 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v581 = stablehlo.constant dense<0.0> : tensor<f32>
    %v582 = stablehlo.reduce(%v580 init: %v581) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v583 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v584 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v585 = stablehlo.multiply %v583, %cb5m : tensor<32xf32>
    %v586 = stablehlo.multiply %v584, %v582 : tensor<32xf32>
    %v587 = stablehlo.add %v585, %v586 : tensor<32xf32>
    %v588 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v589 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v590 = stablehlo.multiply %v588, %cb5v : tensor<32xf32>
    %v591 = stablehlo.multiply %v582, %v582 : tensor<32xf32>
    %v592 = stablehlo.multiply %v589, %v591 : tensor<32xf32>
    %v593 = stablehlo.add %v590, %v592 : tensor<32xf32>
    %v594 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v595 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v596 = stablehlo.divide %v587, %v594 : tensor<32xf32>
    %v597 = stablehlo.divide %v593, %v595 : tensor<32xf32>
    %v598 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v599 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v600 = stablehlo.sqrt %v597 : tensor<32xf32>
    %v601 = stablehlo.add %v600, %v599 : tensor<32xf32>
    %v602 = stablehlo.divide %v596, %v601 : tensor<32xf32>
    %v603 = stablehlo.multiply %v598, %v602 : tensor<32xf32>
    %v604 = stablehlo.subtract %cb5, %v603 : tensor<32xf32>
    %v605 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v606 = stablehlo.multiply %v605, %v598 : tensor<32xf32>
    %v607 = stablehlo.multiply %v606, %cb5 : tensor<32xf32>
    %v608 = stablehlo.subtract %v604, %v607 : tensor<32xf32>
    %v609 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v610 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v611 = stablehlo.multiply %v609, %cb5m : tensor<32xf32>
    %v612 = stablehlo.multiply %v610, %v582 : tensor<32xf32>
    %v613 = stablehlo.add %v611, %v612 : tensor<32xf32>
    %v614 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v615 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v616 = stablehlo.multiply %v614, %cb5v : tensor<32xf32>
    %v617 = stablehlo.multiply %v582, %v582 : tensor<32xf32>
    %v618 = stablehlo.multiply %v615, %v617 : tensor<32xf32>
    %v619 = stablehlo.add %v616, %v618 : tensor<32xf32>
    %v620 = stablehlo.reshape %v57 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v621 = stablehlo.reshape %v154 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v622 = stablehlo.transpose %v620, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v623 = stablehlo.transpose %v621, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v624 = stablehlo.convolution(%v622, %v623)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<32x32x3x3xf32>
    %v625 = stablehlo.transpose %v624, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v626 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v627 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v628 = stablehlo.multiply %v626, %W6m : tensor<32x32x3x3xf32>
    %v629 = stablehlo.multiply %v627, %v625 : tensor<32x32x3x3xf32>
    %v630 = stablehlo.add %v628, %v629 : tensor<32x32x3x3xf32>
    %v631 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v632 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v633 = stablehlo.multiply %v631, %W6v : tensor<32x32x3x3xf32>
    %v634 = stablehlo.multiply %v625, %v625 : tensor<32x32x3x3xf32>
    %v635 = stablehlo.multiply %v632, %v634 : tensor<32x32x3x3xf32>
    %v636 = stablehlo.add %v633, %v635 : tensor<32x32x3x3xf32>
    %v637 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v638 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v639 = stablehlo.divide %v630, %v637 : tensor<32x32x3x3xf32>
    %v640 = stablehlo.divide %v636, %v638 : tensor<32x32x3x3xf32>
    %v641 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v642 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v643 = stablehlo.sqrt %v640 : tensor<32x32x3x3xf32>
    %v644 = stablehlo.add %v643, %v642 : tensor<32x32x3x3xf32>
    %v645 = stablehlo.divide %v639, %v644 : tensor<32x32x3x3xf32>
    %v646 = stablehlo.multiply %v641, %v645 : tensor<32x32x3x3xf32>
    %v647 = stablehlo.subtract %W6, %v646 : tensor<32x32x3x3xf32>
    %v648 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v649 = stablehlo.multiply %v648, %v641 : tensor<32x32x3x3xf32>
    %v650 = stablehlo.multiply %v649, %W6 : tensor<32x32x3x3xf32>
    %v651 = stablehlo.subtract %v647, %v650 : tensor<32x32x3x3xf32>
    %v652 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v653 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v654 = stablehlo.multiply %v652, %W6m : tensor<32x32x3x3xf32>
    %v655 = stablehlo.multiply %v653, %v625 : tensor<32x32x3x3xf32>
    %v656 = stablehlo.add %v654, %v655 : tensor<32x32x3x3xf32>
    %v657 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v658 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v659 = stablehlo.multiply %v657, %W6v : tensor<32x32x3x3xf32>
    %v660 = stablehlo.multiply %v625, %v625 : tensor<32x32x3x3xf32>
    %v661 = stablehlo.multiply %v658, %v660 : tensor<32x32x3x3xf32>
    %v662 = stablehlo.add %v659, %v661 : tensor<32x32x3x3xf32>
    %v663 = stablehlo.reshape %v154 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v664 = stablehlo.constant dense<0.0> : tensor<f32>
    %v665 = stablehlo.reduce(%v663 init: %v664) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v666 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v667 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v668 = stablehlo.multiply %v666, %cb6m : tensor<32xf32>
    %v669 = stablehlo.multiply %v667, %v665 : tensor<32xf32>
    %v670 = stablehlo.add %v668, %v669 : tensor<32xf32>
    %v671 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v672 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v673 = stablehlo.multiply %v671, %cb6v : tensor<32xf32>
    %v674 = stablehlo.multiply %v665, %v665 : tensor<32xf32>
    %v675 = stablehlo.multiply %v672, %v674 : tensor<32xf32>
    %v676 = stablehlo.add %v673, %v675 : tensor<32xf32>
    %v677 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v678 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v679 = stablehlo.divide %v670, %v677 : tensor<32xf32>
    %v680 = stablehlo.divide %v676, %v678 : tensor<32xf32>
    %v681 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v682 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v683 = stablehlo.sqrt %v680 : tensor<32xf32>
    %v684 = stablehlo.add %v683, %v682 : tensor<32xf32>
    %v685 = stablehlo.divide %v679, %v684 : tensor<32xf32>
    %v686 = stablehlo.multiply %v681, %v685 : tensor<32xf32>
    %v687 = stablehlo.subtract %cb6, %v686 : tensor<32xf32>
    %v688 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v689 = stablehlo.multiply %v688, %v681 : tensor<32xf32>
    %v690 = stablehlo.multiply %v689, %cb6 : tensor<32xf32>
    %v691 = stablehlo.subtract %v687, %v690 : tensor<32xf32>
    %v692 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v693 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v694 = stablehlo.multiply %v692, %cb6m : tensor<32xf32>
    %v695 = stablehlo.multiply %v693, %v665 : tensor<32xf32>
    %v696 = stablehlo.add %v694, %v695 : tensor<32xf32>
    %v697 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v698 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v699 = stablehlo.multiply %v697, %cb6v : tensor<32xf32>
    %v700 = stablehlo.multiply %v665, %v665 : tensor<32xf32>
    %v701 = stablehlo.multiply %v698, %v700 : tensor<32xf32>
    %v702 = stablehlo.add %v699, %v701 : tensor<32xf32>
    %v703 = stablehlo.reshape %v71 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v704 = stablehlo.reshape %v141 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v705 = stablehlo.transpose %v703, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v706 = stablehlo.transpose %v704, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v707 = stablehlo.convolution(%v705, %v706)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v708 = stablehlo.transpose %v707, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v709 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v710 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v711 = stablehlo.multiply %v709, %W7m : tensor<32x32x3x3xf32>
    %v712 = stablehlo.multiply %v710, %v708 : tensor<32x32x3x3xf32>
    %v713 = stablehlo.add %v711, %v712 : tensor<32x32x3x3xf32>
    %v714 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v715 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v716 = stablehlo.multiply %v714, %W7v : tensor<32x32x3x3xf32>
    %v717 = stablehlo.multiply %v708, %v708 : tensor<32x32x3x3xf32>
    %v718 = stablehlo.multiply %v715, %v717 : tensor<32x32x3x3xf32>
    %v719 = stablehlo.add %v716, %v718 : tensor<32x32x3x3xf32>
    %v720 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v721 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v722 = stablehlo.divide %v713, %v720 : tensor<32x32x3x3xf32>
    %v723 = stablehlo.divide %v719, %v721 : tensor<32x32x3x3xf32>
    %v724 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v725 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v726 = stablehlo.sqrt %v723 : tensor<32x32x3x3xf32>
    %v727 = stablehlo.add %v726, %v725 : tensor<32x32x3x3xf32>
    %v728 = stablehlo.divide %v722, %v727 : tensor<32x32x3x3xf32>
    %v729 = stablehlo.multiply %v724, %v728 : tensor<32x32x3x3xf32>
    %v730 = stablehlo.subtract %W7, %v729 : tensor<32x32x3x3xf32>
    %v731 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v732 = stablehlo.multiply %v731, %v724 : tensor<32x32x3x3xf32>
    %v733 = stablehlo.multiply %v732, %W7 : tensor<32x32x3x3xf32>
    %v734 = stablehlo.subtract %v730, %v733 : tensor<32x32x3x3xf32>
    %v735 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v736 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v737 = stablehlo.multiply %v735, %W7m : tensor<32x32x3x3xf32>
    %v738 = stablehlo.multiply %v736, %v708 : tensor<32x32x3x3xf32>
    %v739 = stablehlo.add %v737, %v738 : tensor<32x32x3x3xf32>
    %v740 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v741 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v742 = stablehlo.multiply %v740, %W7v : tensor<32x32x3x3xf32>
    %v743 = stablehlo.multiply %v708, %v708 : tensor<32x32x3x3xf32>
    %v744 = stablehlo.multiply %v741, %v743 : tensor<32x32x3x3xf32>
    %v745 = stablehlo.add %v742, %v744 : tensor<32x32x3x3xf32>
    %v746 = stablehlo.reshape %v141 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v747 = stablehlo.constant dense<0.0> : tensor<f32>
    %v748 = stablehlo.reduce(%v746 init: %v747) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v749 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v750 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v751 = stablehlo.multiply %v749, %cb7m : tensor<32xf32>
    %v752 = stablehlo.multiply %v750, %v748 : tensor<32xf32>
    %v753 = stablehlo.add %v751, %v752 : tensor<32xf32>
    %v754 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v755 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v756 = stablehlo.multiply %v754, %cb7v : tensor<32xf32>
    %v757 = stablehlo.multiply %v748, %v748 : tensor<32xf32>
    %v758 = stablehlo.multiply %v755, %v757 : tensor<32xf32>
    %v759 = stablehlo.add %v756, %v758 : tensor<32xf32>
    %v760 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v761 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v762 = stablehlo.divide %v753, %v760 : tensor<32xf32>
    %v763 = stablehlo.divide %v759, %v761 : tensor<32xf32>
    %v764 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v765 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v766 = stablehlo.sqrt %v763 : tensor<32xf32>
    %v767 = stablehlo.add %v766, %v765 : tensor<32xf32>
    %v768 = stablehlo.divide %v762, %v767 : tensor<32xf32>
    %v769 = stablehlo.multiply %v764, %v768 : tensor<32xf32>
    %v770 = stablehlo.subtract %cb7, %v769 : tensor<32xf32>
    %v771 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v772 = stablehlo.multiply %v771, %v764 : tensor<32xf32>
    %v773 = stablehlo.multiply %v772, %cb7 : tensor<32xf32>
    %v774 = stablehlo.subtract %v770, %v773 : tensor<32xf32>
    %v775 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v776 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v777 = stablehlo.multiply %v775, %cb7m : tensor<32xf32>
    %v778 = stablehlo.multiply %v776, %v748 : tensor<32xf32>
    %v779 = stablehlo.add %v777, %v778 : tensor<32xf32>
    %v780 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v781 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v782 = stablehlo.multiply %v780, %cb7v : tensor<32xf32>
    %v783 = stablehlo.multiply %v748, %v748 : tensor<32xf32>
    %v784 = stablehlo.multiply %v781, %v783 : tensor<32xf32>
    %v785 = stablehlo.add %v782, %v784 : tensor<32xf32>
    %v786 = stablehlo.reshape %v81 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v787 = stablehlo.reshape %v133 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v788 = stablehlo.transpose %v786, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v789 = stablehlo.transpose %v787, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v790 = stablehlo.convolution(%v788, %v789)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v791 = stablehlo.transpose %v790, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v792 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v793 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v794 = stablehlo.multiply %v792, %W8m : tensor<32x32x3x3xf32>
    %v795 = stablehlo.multiply %v793, %v791 : tensor<32x32x3x3xf32>
    %v796 = stablehlo.add %v794, %v795 : tensor<32x32x3x3xf32>
    %v797 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v798 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v799 = stablehlo.multiply %v797, %W8v : tensor<32x32x3x3xf32>
    %v800 = stablehlo.multiply %v791, %v791 : tensor<32x32x3x3xf32>
    %v801 = stablehlo.multiply %v798, %v800 : tensor<32x32x3x3xf32>
    %v802 = stablehlo.add %v799, %v801 : tensor<32x32x3x3xf32>
    %v803 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v804 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v805 = stablehlo.divide %v796, %v803 : tensor<32x32x3x3xf32>
    %v806 = stablehlo.divide %v802, %v804 : tensor<32x32x3x3xf32>
    %v807 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v808 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v809 = stablehlo.sqrt %v806 : tensor<32x32x3x3xf32>
    %v810 = stablehlo.add %v809, %v808 : tensor<32x32x3x3xf32>
    %v811 = stablehlo.divide %v805, %v810 : tensor<32x32x3x3xf32>
    %v812 = stablehlo.multiply %v807, %v811 : tensor<32x32x3x3xf32>
    %v813 = stablehlo.subtract %W8, %v812 : tensor<32x32x3x3xf32>
    %v814 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v815 = stablehlo.multiply %v814, %v807 : tensor<32x32x3x3xf32>
    %v816 = stablehlo.multiply %v815, %W8 : tensor<32x32x3x3xf32>
    %v817 = stablehlo.subtract %v813, %v816 : tensor<32x32x3x3xf32>
    %v818 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v819 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v820 = stablehlo.multiply %v818, %W8m : tensor<32x32x3x3xf32>
    %v821 = stablehlo.multiply %v819, %v791 : tensor<32x32x3x3xf32>
    %v822 = stablehlo.add %v820, %v821 : tensor<32x32x3x3xf32>
    %v823 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v824 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v825 = stablehlo.multiply %v823, %W8v : tensor<32x32x3x3xf32>
    %v826 = stablehlo.multiply %v791, %v791 : tensor<32x32x3x3xf32>
    %v827 = stablehlo.multiply %v824, %v826 : tensor<32x32x3x3xf32>
    %v828 = stablehlo.add %v825, %v827 : tensor<32x32x3x3xf32>
    %v829 = stablehlo.reshape %v133 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v830 = stablehlo.constant dense<0.0> : tensor<f32>
    %v831 = stablehlo.reduce(%v829 init: %v830) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v832 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v833 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v834 = stablehlo.multiply %v832, %cb8m : tensor<32xf32>
    %v835 = stablehlo.multiply %v833, %v831 : tensor<32xf32>
    %v836 = stablehlo.add %v834, %v835 : tensor<32xf32>
    %v837 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v838 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v839 = stablehlo.multiply %v837, %cb8v : tensor<32xf32>
    %v840 = stablehlo.multiply %v831, %v831 : tensor<32xf32>
    %v841 = stablehlo.multiply %v838, %v840 : tensor<32xf32>
    %v842 = stablehlo.add %v839, %v841 : tensor<32xf32>
    %v843 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v844 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v845 = stablehlo.divide %v836, %v843 : tensor<32xf32>
    %v846 = stablehlo.divide %v842, %v844 : tensor<32xf32>
    %v847 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v848 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v849 = stablehlo.sqrt %v846 : tensor<32xf32>
    %v850 = stablehlo.add %v849, %v848 : tensor<32xf32>
    %v851 = stablehlo.divide %v845, %v850 : tensor<32xf32>
    %v852 = stablehlo.multiply %v847, %v851 : tensor<32xf32>
    %v853 = stablehlo.subtract %cb8, %v852 : tensor<32xf32>
    %v854 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v855 = stablehlo.multiply %v854, %v847 : tensor<32xf32>
    %v856 = stablehlo.multiply %v855, %cb8 : tensor<32xf32>
    %v857 = stablehlo.subtract %v853, %v856 : tensor<32xf32>
    %v858 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v859 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v860 = stablehlo.multiply %v858, %cb8m : tensor<32xf32>
    %v861 = stablehlo.multiply %v859, %v831 : tensor<32xf32>
    %v862 = stablehlo.add %v860, %v861 : tensor<32xf32>
    %v863 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v864 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v865 = stablehlo.multiply %v863, %cb8v : tensor<32xf32>
    %v866 = stablehlo.multiply %v831, %v831 : tensor<32xf32>
    %v867 = stablehlo.multiply %v864, %v866 : tensor<32xf32>
    %v868 = stablehlo.add %v865, %v867 : tensor<32xf32>
    %v869 = stablehlo.dot_general %v95, %v124, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %v870 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v871 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v872 = stablehlo.multiply %v870, %W9m : tensor<128x64xf32>
    %v873 = stablehlo.multiply %v871, %v869 : tensor<128x64xf32>
    %v874 = stablehlo.add %v872, %v873 : tensor<128x64xf32>
    %v875 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v876 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v877 = stablehlo.multiply %v875, %W9v : tensor<128x64xf32>
    %v878 = stablehlo.multiply %v869, %v869 : tensor<128x64xf32>
    %v879 = stablehlo.multiply %v876, %v878 : tensor<128x64xf32>
    %v880 = stablehlo.add %v877, %v879 : tensor<128x64xf32>
    %v881 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v882 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v883 = stablehlo.divide %v874, %v881 : tensor<128x64xf32>
    %v884 = stablehlo.divide %v880, %v882 : tensor<128x64xf32>
    %v885 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v886 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v887 = stablehlo.sqrt %v884 : tensor<128x64xf32>
    %v888 = stablehlo.add %v887, %v886 : tensor<128x64xf32>
    %v889 = stablehlo.divide %v883, %v888 : tensor<128x64xf32>
    %v890 = stablehlo.multiply %v885, %v889 : tensor<128x64xf32>
    %v891 = stablehlo.subtract %W9, %v890 : tensor<128x64xf32>
    %v892 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v893 = stablehlo.multiply %v892, %v885 : tensor<128x64xf32>
    %v894 = stablehlo.multiply %v893, %W9 : tensor<128x64xf32>
    %v895 = stablehlo.subtract %v891, %v894 : tensor<128x64xf32>
    %v896 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v897 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v898 = stablehlo.multiply %v896, %W9m : tensor<128x64xf32>
    %v899 = stablehlo.multiply %v897, %v869 : tensor<128x64xf32>
    %v900 = stablehlo.add %v898, %v899 : tensor<128x64xf32>
    %v901 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v902 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v903 = stablehlo.multiply %v901, %W9v : tensor<128x64xf32>
    %v904 = stablehlo.multiply %v869, %v869 : tensor<128x64xf32>
    %v905 = stablehlo.multiply %v902, %v904 : tensor<128x64xf32>
    %v906 = stablehlo.add %v903, %v905 : tensor<128x64xf32>
    %v907 = stablehlo.constant dense<0.0> : tensor<f32>
    %v908 = stablehlo.reduce(%v124 init: %v907) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v909 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v910 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v911 = stablehlo.multiply %v909, %b9m : tensor<64xf32>
    %v912 = stablehlo.multiply %v910, %v908 : tensor<64xf32>
    %v913 = stablehlo.add %v911, %v912 : tensor<64xf32>
    %v914 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v915 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v916 = stablehlo.multiply %v914, %b9v : tensor<64xf32>
    %v917 = stablehlo.multiply %v908, %v908 : tensor<64xf32>
    %v918 = stablehlo.multiply %v915, %v917 : tensor<64xf32>
    %v919 = stablehlo.add %v916, %v918 : tensor<64xf32>
    %v920 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v921 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v922 = stablehlo.divide %v913, %v920 : tensor<64xf32>
    %v923 = stablehlo.divide %v919, %v921 : tensor<64xf32>
    %v924 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v925 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v926 = stablehlo.sqrt %v923 : tensor<64xf32>
    %v927 = stablehlo.add %v926, %v925 : tensor<64xf32>
    %v928 = stablehlo.divide %v922, %v927 : tensor<64xf32>
    %v929 = stablehlo.multiply %v924, %v928 : tensor<64xf32>
    %v930 = stablehlo.subtract %b9, %v929 : tensor<64xf32>
    %v931 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v932 = stablehlo.multiply %v931, %v924 : tensor<64xf32>
    %v933 = stablehlo.multiply %v932, %b9 : tensor<64xf32>
    %v934 = stablehlo.subtract %v930, %v933 : tensor<64xf32>
    %v935 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v936 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v937 = stablehlo.multiply %v935, %b9m : tensor<64xf32>
    %v938 = stablehlo.multiply %v936, %v908 : tensor<64xf32>
    %v939 = stablehlo.add %v937, %v938 : tensor<64xf32>
    %v940 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v941 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v942 = stablehlo.multiply %v940, %b9v : tensor<64xf32>
    %v943 = stablehlo.multiply %v908, %v908 : tensor<64xf32>
    %v944 = stablehlo.multiply %v941, %v943 : tensor<64xf32>
    %v945 = stablehlo.add %v942, %v944 : tensor<64xf32>
    %v946 = stablehlo.dot_general %v100, %v120, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<64x64xf32>
    %v947 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v948 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v949 = stablehlo.multiply %v947, %Wam : tensor<64x64xf32>
    %v950 = stablehlo.multiply %v948, %v946 : tensor<64x64xf32>
    %v951 = stablehlo.add %v949, %v950 : tensor<64x64xf32>
    %v952 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v953 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v954 = stablehlo.multiply %v952, %Wav : tensor<64x64xf32>
    %v955 = stablehlo.multiply %v946, %v946 : tensor<64x64xf32>
    %v956 = stablehlo.multiply %v953, %v955 : tensor<64x64xf32>
    %v957 = stablehlo.add %v954, %v956 : tensor<64x64xf32>
    %v958 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v959 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v960 = stablehlo.divide %v951, %v958 : tensor<64x64xf32>
    %v961 = stablehlo.divide %v957, %v959 : tensor<64x64xf32>
    %v962 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v963 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v964 = stablehlo.sqrt %v961 : tensor<64x64xf32>
    %v965 = stablehlo.add %v964, %v963 : tensor<64x64xf32>
    %v966 = stablehlo.divide %v960, %v965 : tensor<64x64xf32>
    %v967 = stablehlo.multiply %v962, %v966 : tensor<64x64xf32>
    %v968 = stablehlo.subtract %Wa, %v967 : tensor<64x64xf32>
    %v969 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v970 = stablehlo.multiply %v969, %v962 : tensor<64x64xf32>
    %v971 = stablehlo.multiply %v970, %Wa : tensor<64x64xf32>
    %v972 = stablehlo.subtract %v968, %v971 : tensor<64x64xf32>
    %v973 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v974 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v975 = stablehlo.multiply %v973, %Wam : tensor<64x64xf32>
    %v976 = stablehlo.multiply %v974, %v946 : tensor<64x64xf32>
    %v977 = stablehlo.add %v975, %v976 : tensor<64x64xf32>
    %v978 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v979 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v980 = stablehlo.multiply %v978, %Wav : tensor<64x64xf32>
    %v981 = stablehlo.multiply %v946, %v946 : tensor<64x64xf32>
    %v982 = stablehlo.multiply %v979, %v981 : tensor<64x64xf32>
    %v983 = stablehlo.add %v980, %v982 : tensor<64x64xf32>
    %v984 = stablehlo.constant dense<0.0> : tensor<f32>
    %v985 = stablehlo.reduce(%v120 init: %v984) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v986 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v987 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v988 = stablehlo.multiply %v986, %bam : tensor<64xf32>
    %v989 = stablehlo.multiply %v987, %v985 : tensor<64xf32>
    %v990 = stablehlo.add %v988, %v989 : tensor<64xf32>
    %v991 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v992 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v993 = stablehlo.multiply %v991, %bav : tensor<64xf32>
    %v994 = stablehlo.multiply %v985, %v985 : tensor<64xf32>
    %v995 = stablehlo.multiply %v992, %v994 : tensor<64xf32>
    %v996 = stablehlo.add %v993, %v995 : tensor<64xf32>
    %v997 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v998 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v999 = stablehlo.divide %v990, %v997 : tensor<64xf32>
    %v1000 = stablehlo.divide %v996, %v998 : tensor<64xf32>
    %v1001 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1002 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1003 = stablehlo.sqrt %v1000 : tensor<64xf32>
    %v1004 = stablehlo.add %v1003, %v1002 : tensor<64xf32>
    %v1005 = stablehlo.divide %v999, %v1004 : tensor<64xf32>
    %v1006 = stablehlo.multiply %v1001, %v1005 : tensor<64xf32>
    %v1007 = stablehlo.subtract %ba, %v1006 : tensor<64xf32>
    %v1008 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1009 = stablehlo.multiply %v1008, %v1001 : tensor<64xf32>
    %v1010 = stablehlo.multiply %v1009, %ba : tensor<64xf32>
    %v1011 = stablehlo.subtract %v1007, %v1010 : tensor<64xf32>
    %v1012 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1013 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1014 = stablehlo.multiply %v1012, %bam : tensor<64xf32>
    %v1015 = stablehlo.multiply %v1013, %v985 : tensor<64xf32>
    %v1016 = stablehlo.add %v1014, %v1015 : tensor<64xf32>
    %v1017 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1018 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1019 = stablehlo.multiply %v1017, %bav : tensor<64xf32>
    %v1020 = stablehlo.multiply %v985, %v985 : tensor<64xf32>
    %v1021 = stablehlo.multiply %v1018, %v1020 : tensor<64xf32>
    %v1022 = stablehlo.add %v1019, %v1021 : tensor<64xf32>
    %v1023 = stablehlo.dot_general %v105, %v116, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x10xf32>) -> tensor<64x10xf32>
    %v1024 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1025 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1026 = stablehlo.multiply %v1024, %Wbm : tensor<64x10xf32>
    %v1027 = stablehlo.multiply %v1025, %v1023 : tensor<64x10xf32>
    %v1028 = stablehlo.add %v1026, %v1027 : tensor<64x10xf32>
    %v1029 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1030 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1031 = stablehlo.multiply %v1029, %Wbv : tensor<64x10xf32>
    %v1032 = stablehlo.multiply %v1023, %v1023 : tensor<64x10xf32>
    %v1033 = stablehlo.multiply %v1030, %v1032 : tensor<64x10xf32>
    %v1034 = stablehlo.add %v1031, %v1033 : tensor<64x10xf32>
    %v1035 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1036 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1037 = stablehlo.divide %v1028, %v1035 : tensor<64x10xf32>
    %v1038 = stablehlo.divide %v1034, %v1036 : tensor<64x10xf32>
    %v1039 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1040 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1041 = stablehlo.sqrt %v1038 : tensor<64x10xf32>
    %v1042 = stablehlo.add %v1041, %v1040 : tensor<64x10xf32>
    %v1043 = stablehlo.divide %v1037, %v1042 : tensor<64x10xf32>
    %v1044 = stablehlo.multiply %v1039, %v1043 : tensor<64x10xf32>
    %v1045 = stablehlo.subtract %Wb, %v1044 : tensor<64x10xf32>
    %v1046 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1047 = stablehlo.multiply %v1046, %v1039 : tensor<64x10xf32>
    %v1048 = stablehlo.multiply %v1047, %Wb : tensor<64x10xf32>
    %v1049 = stablehlo.subtract %v1045, %v1048 : tensor<64x10xf32>
    %v1050 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1051 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1052 = stablehlo.multiply %v1050, %Wbm : tensor<64x10xf32>
    %v1053 = stablehlo.multiply %v1051, %v1023 : tensor<64x10xf32>
    %v1054 = stablehlo.add %v1052, %v1053 : tensor<64x10xf32>
    %v1055 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1056 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1057 = stablehlo.multiply %v1055, %Wbv : tensor<64x10xf32>
    %v1058 = stablehlo.multiply %v1023, %v1023 : tensor<64x10xf32>
    %v1059 = stablehlo.multiply %v1056, %v1058 : tensor<64x10xf32>
    %v1060 = stablehlo.add %v1057, %v1059 : tensor<64x10xf32>
    %v1061 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1062 = stablehlo.reduce(%v116 init: %v1061) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1063 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1064 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1065 = stablehlo.multiply %v1063, %bbm : tensor<10xf32>
    %v1066 = stablehlo.multiply %v1064, %v1062 : tensor<10xf32>
    %v1067 = stablehlo.add %v1065, %v1066 : tensor<10xf32>
    %v1068 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1069 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1070 = stablehlo.multiply %v1068, %bbv : tensor<10xf32>
    %v1071 = stablehlo.multiply %v1062, %v1062 : tensor<10xf32>
    %v1072 = stablehlo.multiply %v1069, %v1071 : tensor<10xf32>
    %v1073 = stablehlo.add %v1070, %v1072 : tensor<10xf32>
    %v1074 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1075 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1076 = stablehlo.divide %v1067, %v1074 : tensor<10xf32>
    %v1077 = stablehlo.divide %v1073, %v1075 : tensor<10xf32>
    %v1078 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1079 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1080 = stablehlo.sqrt %v1077 : tensor<10xf32>
    %v1081 = stablehlo.add %v1080, %v1079 : tensor<10xf32>
    %v1082 = stablehlo.divide %v1076, %v1081 : tensor<10xf32>
    %v1083 = stablehlo.multiply %v1078, %v1082 : tensor<10xf32>
    %v1084 = stablehlo.subtract %bb, %v1083 : tensor<10xf32>
    %v1085 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1086 = stablehlo.multiply %v1085, %v1078 : tensor<10xf32>
    %v1087 = stablehlo.multiply %v1086, %bb : tensor<10xf32>
    %v1088 = stablehlo.subtract %v1084, %v1087 : tensor<10xf32>
    %v1089 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1090 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1091 = stablehlo.multiply %v1089, %bbm : tensor<10xf32>
    %v1092 = stablehlo.multiply %v1090, %v1062 : tensor<10xf32>
    %v1093 = stablehlo.add %v1091, %v1092 : tensor<10xf32>
    %v1094 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1095 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1096 = stablehlo.multiply %v1094, %bbv : tensor<10xf32>
    %v1097 = stablehlo.multiply %v1062, %v1062 : tensor<10xf32>
    %v1098 = stablehlo.multiply %v1095, %v1097 : tensor<10xf32>
    %v1099 = stablehlo.add %v1096, %v1098 : tensor<10xf32>
    return %v236, %v276, %v319, %v359, %v402, %v442, %v485, %v525, %v568, %v608, %v651, %v691, %v734, %v774, %v817, %v857, %v895, %v934, %v972, %v1011, %v1049, %v1088, %v241, %v281, %v324, %v364, %v407, %v447, %v490, %v530, %v573, %v613, %v656, %v696, %v739, %v779, %v822, %v862, %v900, %v939, %v977, %v1016, %v1054, %v1093, %v247, %v287, %v330, %v370, %v413, %v453, %v496, %v536, %v579, %v619, %v662, %v702, %v745, %v785, %v828, %v868, %v906, %v945, %v983, %v1022, %v1060, %v1099, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
