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
    %v8 = stablehlo.reshape %v7 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v9 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v10 = stablehlo.maximum %v8, %v9 : tensor<128x16x32x32xf32>
    %v11 = stablehlo.reshape %v10 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v12 = stablehlo.reshape %v11 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v13 = stablehlo.convert %v12 : (tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xbf16>
    %v14 = stablehlo.convert %W2 : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xbf16>
    %v15 = stablehlo.convolution(%v13, %v14)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xbf16>, tensor<16x16x3x3xbf16>) -> tensor<128x16x32x32xbf16>
    %v16 = stablehlo.convert %v15 : (tensor<128x16x32x32xbf16>) -> tensor<128x16x32x32xf32>
    %v17 = stablehlo.broadcast_in_dim %cb2, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v18 = stablehlo.add %v16, %v17 : tensor<128x16x32x32xf32>
    %v19 = stablehlo.reshape %v18 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v20 = stablehlo.reshape %v19 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v21 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v22 = stablehlo.maximum %v20, %v21 : tensor<128x16x32x32xf32>
    %v23 = stablehlo.reshape %v22 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v24 = stablehlo.reshape %v23 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v25 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v26 = "stablehlo.reduce_window"(%v24, %v25) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16x16x16xf32>
    %v27 = stablehlo.reshape %v26 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v28 = stablehlo.reshape %v27 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v29 = stablehlo.convert %v28 : (tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xbf16>
    %v30 = stablehlo.convert %W3 : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xbf16>
    %v31 = stablehlo.convolution(%v29, %v30)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xbf16>, tensor<16x16x3x3xbf16>) -> tensor<128x16x16x16xbf16>
    %v32 = stablehlo.convert %v31 : (tensor<128x16x16x16xbf16>) -> tensor<128x16x16x16xf32>
    %v33 = stablehlo.broadcast_in_dim %cb3, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v34 = stablehlo.add %v32, %v33 : tensor<128x16x16x16xf32>
    %v35 = stablehlo.reshape %v34 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v36 = stablehlo.reshape %v35 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v37 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v38 = stablehlo.maximum %v36, %v37 : tensor<128x16x16x16xf32>
    %v39 = stablehlo.reshape %v38 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v40 = stablehlo.reshape %v39 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v41 = stablehlo.convert %v40 : (tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xbf16>
    %v42 = stablehlo.convert %W4 : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xbf16>
    %v43 = stablehlo.convolution(%v41, %v42)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xbf16>, tensor<16x16x3x3xbf16>) -> tensor<128x16x16x16xbf16>
    %v44 = stablehlo.convert %v43 : (tensor<128x16x16x16xbf16>) -> tensor<128x16x16x16xf32>
    %v45 = stablehlo.broadcast_in_dim %cb4, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v46 = stablehlo.add %v44, %v45 : tensor<128x16x16x16xf32>
    %v47 = stablehlo.reshape %v46 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v48 = stablehlo.reshape %v47 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v49 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v50 = stablehlo.maximum %v48, %v49 : tensor<128x16x16x16xf32>
    %v51 = stablehlo.reshape %v50 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v52 = stablehlo.reshape %v51 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v53 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v54 = "stablehlo.reduce_window"(%v52, %v53) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x8x8xf32>
    %v55 = stablehlo.reshape %v54 : (tensor<128x16x8x8xf32>) -> tensor<128x1024xf32>
    %v56 = stablehlo.reshape %v55 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v57 = stablehlo.convert %v56 : (tensor<128x16x8x8xf32>) -> tensor<128x16x8x8xbf16>
    %v58 = stablehlo.convert %W5 : (tensor<32x16x3x3xf32>) -> tensor<32x16x3x3xbf16>
    %v59 = stablehlo.convolution(%v57, %v58)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x8x8xbf16>, tensor<32x16x3x3xbf16>) -> tensor<128x32x8x8xbf16>
    %v60 = stablehlo.convert %v59 : (tensor<128x32x8x8xbf16>) -> tensor<128x32x8x8xf32>
    %v61 = stablehlo.broadcast_in_dim %cb5, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v62 = stablehlo.add %v60, %v61 : tensor<128x32x8x8xf32>
    %v63 = stablehlo.reshape %v62 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v64 = stablehlo.reshape %v63 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v65 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v66 = stablehlo.maximum %v64, %v65 : tensor<128x32x8x8xf32>
    %v67 = stablehlo.reshape %v66 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v68 = stablehlo.reshape %v67 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v69 = stablehlo.convert %v68 : (tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xbf16>
    %v70 = stablehlo.convert %W6 : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xbf16>
    %v71 = stablehlo.convolution(%v69, %v70)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xbf16>, tensor<32x32x3x3xbf16>) -> tensor<128x32x8x8xbf16>
    %v72 = stablehlo.convert %v71 : (tensor<128x32x8x8xbf16>) -> tensor<128x32x8x8xf32>
    %v73 = stablehlo.broadcast_in_dim %cb6, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v74 = stablehlo.add %v72, %v73 : tensor<128x32x8x8xf32>
    %v75 = stablehlo.reshape %v74 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v76 = stablehlo.reshape %v75 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v77 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v78 = stablehlo.maximum %v76, %v77 : tensor<128x32x8x8xf32>
    %v79 = stablehlo.reshape %v78 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v80 = stablehlo.reshape %v79 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v81 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v82 = "stablehlo.reduce_window"(%v80, %v81) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32x4x4xf32>
    %v83 = stablehlo.reshape %v82 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v84 = stablehlo.reshape %v83 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v85 = stablehlo.convert %v84 : (tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xbf16>
    %v86 = stablehlo.convert %W7 : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xbf16>
    %v87 = stablehlo.convolution(%v85, %v86)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xbf16>, tensor<32x32x3x3xbf16>) -> tensor<128x32x4x4xbf16>
    %v88 = stablehlo.convert %v87 : (tensor<128x32x4x4xbf16>) -> tensor<128x32x4x4xf32>
    %v89 = stablehlo.broadcast_in_dim %cb7, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v90 = stablehlo.add %v88, %v89 : tensor<128x32x4x4xf32>
    %v91 = stablehlo.reshape %v90 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v92 = stablehlo.reshape %v91 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v93 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v94 = stablehlo.maximum %v92, %v93 : tensor<128x32x4x4xf32>
    %v95 = stablehlo.reshape %v94 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v96 = stablehlo.reshape %v95 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v97 = stablehlo.convert %v96 : (tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xbf16>
    %v98 = stablehlo.convert %W8 : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xbf16>
    %v99 = stablehlo.convolution(%v97, %v98)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xbf16>, tensor<32x32x3x3xbf16>) -> tensor<128x32x4x4xbf16>
    %v100 = stablehlo.convert %v99 : (tensor<128x32x4x4xbf16>) -> tensor<128x32x4x4xf32>
    %v101 = stablehlo.broadcast_in_dim %cb8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v102 = stablehlo.add %v100, %v101 : tensor<128x32x4x4xf32>
    %v103 = stablehlo.reshape %v102 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v104 = stablehlo.reshape %v103 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v105 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v106 = stablehlo.maximum %v104, %v105 : tensor<128x32x4x4xf32>
    %v107 = stablehlo.reshape %v106 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v108 = stablehlo.reshape %v107 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v109 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v110 = "stablehlo.reduce_window"(%v108, %v109) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32x2x2xf32>
    %v111 = stablehlo.reshape %v110 : (tensor<128x32x2x2xf32>) -> tensor<128x128xf32>
    %v112 = stablehlo.dot_general %v111, %W9, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %v113 = stablehlo.broadcast_in_dim %b9, dims = [1] : (tensor<64xf32>) -> tensor<128x64xf32>
    %v114 = stablehlo.add %v112, %v113 : tensor<128x64xf32>
    %v115 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v116 = stablehlo.maximum %v114, %v115 : tensor<128x64xf32>
    %v117 = stablehlo.dot_general %v116, %Wa, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
    %v118 = stablehlo.broadcast_in_dim %ba, dims = [1] : (tensor<64xf32>) -> tensor<128x64xf32>
    %v119 = stablehlo.add %v117, %v118 : tensor<128x64xf32>
    %v120 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v121 = stablehlo.maximum %v119, %v120 : tensor<128x64xf32>
    %v122 = stablehlo.dot_general %v121, %Wb, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x10xf32>) -> tensor<128x10xf32>
    %v123 = stablehlo.broadcast_in_dim %bb, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v124 = stablehlo.add %v122, %v123 : tensor<128x10xf32>
    %v125 = stablehlo.exponential %v124 : tensor<128x10xf32>
    %v126 = stablehlo.constant dense<0.0> : tensor<f32>
    %v127 = stablehlo.reduce(%v125 init: %v126) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %v128 = stablehlo.broadcast_in_dim %v127, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %v129 = stablehlo.divide %v125, %v128 : tensor<128x10xf32>
    %v130 = stablehlo.subtract %v129, %onehot : tensor<128x10xf32>
    %v131 = stablehlo.constant dense<0.0078125> : tensor<128x10xf32>
    %v132 = stablehlo.multiply %v130, %v131 : tensor<128x10xf32>
    // ── report-only scalar loss (NOT pretty(AST): the kit has no rank-0 loss op; it
    //    feeds no parameter, only the driver's progress line) ──
    %llog = stablehlo.log %v129 : tensor<128x10xf32>
    %ohll = stablehlo.multiply %onehot, %llog : tensor<128x10xf32>
    %csum = stablehlo.reduce(%ohll init: %lzero) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<f32>
    %cneg = stablehlo.negate %csum : tensor<f32>
    %lbf = stablehlo.constant dense<128.0> : tensor<f32>
    %loss = stablehlo.divide %cneg, %lbf : tensor<f32>
    %v133 = stablehlo.dot_general %v132, %Wb, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x10xf32>, tensor<64x10xf32>) -> tensor<128x64xf32>
    %v134 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v135 = stablehlo.compare GT, %v119, %v134 : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xi1>
    %v136 = stablehlo.select %v135, %v133, %v134 : tensor<128x64xi1>, tensor<128x64xf32>
    %v137 = stablehlo.dot_general %v136, %Wa, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
    %v138 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v139 = stablehlo.compare GT, %v114, %v138 : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xi1>
    %v140 = stablehlo.select %v139, %v137, %v138 : tensor<128x64xi1>, tensor<128x64xf32>
    %v141 = stablehlo.dot_general %v140, %W9, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x128xf32>
    %v142 = stablehlo.reshape %v107 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v143 = stablehlo.reshape %v141 : (tensor<128x128xf32>) -> tensor<128x32x2x2xf32>
    %v144 = stablehlo.constant dense<0.0> : tensor<f32>
    %v145 = "stablehlo.select_and_scatter"(%v142, %v143, %v144) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x4x4xf32>, tensor<128x32x2x2xf32>, tensor<f32>) -> tensor<128x32x4x4xf32>
    %v146 = stablehlo.reshape %v145 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v147 = stablehlo.reshape %v146 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v148 = stablehlo.reshape %v103 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v149 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v150 = stablehlo.compare GT, %v148, %v149 : (tensor<128x32x4x4xf32>, tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xi1>
    %v151 = stablehlo.select %v150, %v147, %v149 : tensor<128x32x4x4xi1>, tensor<128x32x4x4xf32>
    %v152 = stablehlo.reshape %v151 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v153 = stablehlo.reshape %v152 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v154 = stablehlo.transpose %W8, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v155 = stablehlo.reverse %v154, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v156 = stablehlo.convolution(%v153, %v155)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v157 = stablehlo.reshape %v156 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v158 = stablehlo.reshape %v157 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v159 = stablehlo.reshape %v91 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v160 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v161 = stablehlo.compare GT, %v159, %v160 : (tensor<128x32x4x4xf32>, tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xi1>
    %v162 = stablehlo.select %v161, %v158, %v160 : tensor<128x32x4x4xi1>, tensor<128x32x4x4xf32>
    %v163 = stablehlo.reshape %v162 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v164 = stablehlo.reshape %v163 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v165 = stablehlo.transpose %W7, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v166 = stablehlo.reverse %v165, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v167 = stablehlo.convolution(%v164, %v166)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v168 = stablehlo.reshape %v167 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v169 = stablehlo.reshape %v79 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v170 = stablehlo.reshape %v168 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v171 = stablehlo.constant dense<0.0> : tensor<f32>
    %v172 = "stablehlo.select_and_scatter"(%v169, %v170, %v171) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x8x8xf32>, tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32x8x8xf32>
    %v173 = stablehlo.reshape %v172 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v174 = stablehlo.reshape %v173 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v175 = stablehlo.reshape %v75 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v176 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v177 = stablehlo.compare GT, %v175, %v176 : (tensor<128x32x8x8xf32>, tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xi1>
    %v178 = stablehlo.select %v177, %v174, %v176 : tensor<128x32x8x8xi1>, tensor<128x32x8x8xf32>
    %v179 = stablehlo.reshape %v178 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v180 = stablehlo.reshape %v179 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v181 = stablehlo.transpose %W6, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v182 = stablehlo.reverse %v181, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v183 = stablehlo.convolution(%v180, %v182)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x8x8xf32>
    %v184 = stablehlo.reshape %v183 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v185 = stablehlo.reshape %v184 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v186 = stablehlo.reshape %v63 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v187 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v188 = stablehlo.compare GT, %v186, %v187 : (tensor<128x32x8x8xf32>, tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xi1>
    %v189 = stablehlo.select %v188, %v185, %v187 : tensor<128x32x8x8xi1>, tensor<128x32x8x8xf32>
    %v190 = stablehlo.reshape %v189 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v191 = stablehlo.reshape %v190 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v192 = stablehlo.transpose %W5, dims = [1, 0, 2, 3] : (tensor<32x16x3x3xf32>) -> tensor<16x32x3x3xf32>
    %v193 = stablehlo.reverse %v192, dims = [2, 3] : tensor<16x32x3x3xf32>
    %v194 = stablehlo.convolution(%v191, %v193)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<16x32x3x3xf32>) -> tensor<128x16x8x8xf32>
    %v195 = stablehlo.reshape %v194 : (tensor<128x16x8x8xf32>) -> tensor<128x1024xf32>
    %v196 = stablehlo.reshape %v51 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v197 = stablehlo.reshape %v195 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v198 = stablehlo.constant dense<0.0> : tensor<f32>
    %v199 = "stablehlo.select_and_scatter"(%v196, %v197, %v198) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x16x16xf32>, tensor<128x16x8x8xf32>, tensor<f32>) -> tensor<128x16x16x16xf32>
    %v200 = stablehlo.reshape %v199 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v201 = stablehlo.reshape %v200 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v202 = stablehlo.reshape %v47 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v203 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v204 = stablehlo.compare GT, %v202, %v203 : (tensor<128x16x16x16xf32>, tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xi1>
    %v205 = stablehlo.select %v204, %v201, %v203 : tensor<128x16x16x16xi1>, tensor<128x16x16x16xf32>
    %v206 = stablehlo.reshape %v205 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v207 = stablehlo.reshape %v206 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v208 = stablehlo.transpose %W4, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v209 = stablehlo.reverse %v208, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v210 = stablehlo.convolution(%v207, %v209)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v211 = stablehlo.reshape %v210 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v212 = stablehlo.reshape %v211 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v213 = stablehlo.reshape %v35 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v214 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v215 = stablehlo.compare GT, %v213, %v214 : (tensor<128x16x16x16xf32>, tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xi1>
    %v216 = stablehlo.select %v215, %v212, %v214 : tensor<128x16x16x16xi1>, tensor<128x16x16x16xf32>
    %v217 = stablehlo.reshape %v216 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v218 = stablehlo.reshape %v217 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v219 = stablehlo.transpose %W3, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v220 = stablehlo.reverse %v219, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v221 = stablehlo.convolution(%v218, %v220)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v222 = stablehlo.reshape %v221 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v223 = stablehlo.reshape %v23 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v224 = stablehlo.reshape %v222 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v225 = stablehlo.constant dense<0.0> : tensor<f32>
    %v226 = "stablehlo.select_and_scatter"(%v223, %v224, %v225) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x32x32xf32>, tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x32x32xf32>
    %v227 = stablehlo.reshape %v226 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v228 = stablehlo.reshape %v227 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v229 = stablehlo.reshape %v19 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v230 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v231 = stablehlo.compare GT, %v229, %v230 : (tensor<128x16x32x32xf32>, tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xi1>
    %v232 = stablehlo.select %v231, %v228, %v230 : tensor<128x16x32x32xi1>, tensor<128x16x32x32xf32>
    %v233 = stablehlo.reshape %v232 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v234 = stablehlo.reshape %v233 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v235 = stablehlo.transpose %W2, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v236 = stablehlo.reverse %v235, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v237 = stablehlo.convolution(%v234, %v236)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x32x32xf32>
    %v238 = stablehlo.reshape %v237 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v239 = stablehlo.reshape %v238 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v240 = stablehlo.reshape %v7 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v241 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v242 = stablehlo.compare GT, %v240, %v241 : (tensor<128x16x32x32xf32>, tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xi1>
    %v243 = stablehlo.select %v242, %v239, %v241 : tensor<128x16x32x32xi1>, tensor<128x16x32x32xf32>
    %v244 = stablehlo.reshape %v243 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v245 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v246 = stablehlo.reshape %v244 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v247 = stablehlo.transpose %v245, dims = [1, 0, 2, 3] : (tensor<128x3x32x32xf32>) -> tensor<3x128x32x32xf32>
    %v248 = stablehlo.transpose %v246, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v249 = stablehlo.convolution(%v247, %v248)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<3x16x3x3xf32>
    %v250 = stablehlo.transpose %v249, dims = [1, 0, 2, 3] : (tensor<3x16x3x3xf32>) -> tensor<16x3x3x3xf32>
    %v251 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v252 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v253 = stablehlo.multiply %v251, %W1m : tensor<16x3x3x3xf32>
    %v254 = stablehlo.multiply %v252, %v250 : tensor<16x3x3x3xf32>
    %v255 = stablehlo.add %v253, %v254 : tensor<16x3x3x3xf32>
    %v256 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v257 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v258 = stablehlo.multiply %v256, %W1v : tensor<16x3x3x3xf32>
    %v259 = stablehlo.multiply %v250, %v250 : tensor<16x3x3x3xf32>
    %v260 = stablehlo.multiply %v257, %v259 : tensor<16x3x3x3xf32>
    %v261 = stablehlo.add %v258, %v260 : tensor<16x3x3x3xf32>
    %v262 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v263 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v264 = stablehlo.divide %v255, %v262 : tensor<16x3x3x3xf32>
    %v265 = stablehlo.divide %v261, %v263 : tensor<16x3x3x3xf32>
    %v266 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v267 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v268 = stablehlo.sqrt %v265 : tensor<16x3x3x3xf32>
    %v269 = stablehlo.add %v268, %v267 : tensor<16x3x3x3xf32>
    %v270 = stablehlo.divide %v264, %v269 : tensor<16x3x3x3xf32>
    %v271 = stablehlo.multiply %v266, %v270 : tensor<16x3x3x3xf32>
    %v272 = stablehlo.subtract %W1, %v271 : tensor<16x3x3x3xf32>
    %v273 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v274 = stablehlo.multiply %v273, %v266 : tensor<16x3x3x3xf32>
    %v275 = stablehlo.multiply %v274, %W1 : tensor<16x3x3x3xf32>
    %v276 = stablehlo.subtract %v272, %v275 : tensor<16x3x3x3xf32>
    %v277 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v278 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v279 = stablehlo.multiply %v277, %W1m : tensor<16x3x3x3xf32>
    %v280 = stablehlo.multiply %v278, %v250 : tensor<16x3x3x3xf32>
    %v281 = stablehlo.add %v279, %v280 : tensor<16x3x3x3xf32>
    %v282 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v283 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v284 = stablehlo.multiply %v282, %W1v : tensor<16x3x3x3xf32>
    %v285 = stablehlo.multiply %v250, %v250 : tensor<16x3x3x3xf32>
    %v286 = stablehlo.multiply %v283, %v285 : tensor<16x3x3x3xf32>
    %v287 = stablehlo.add %v284, %v286 : tensor<16x3x3x3xf32>
    %v288 = stablehlo.reshape %v244 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v289 = stablehlo.constant dense<0.0> : tensor<f32>
    %v290 = stablehlo.reduce(%v288 init: %v289) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v291 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v292 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v293 = stablehlo.multiply %v291, %cb1m : tensor<16xf32>
    %v294 = stablehlo.multiply %v292, %v290 : tensor<16xf32>
    %v295 = stablehlo.add %v293, %v294 : tensor<16xf32>
    %v296 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v297 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v298 = stablehlo.multiply %v296, %cb1v : tensor<16xf32>
    %v299 = stablehlo.multiply %v290, %v290 : tensor<16xf32>
    %v300 = stablehlo.multiply %v297, %v299 : tensor<16xf32>
    %v301 = stablehlo.add %v298, %v300 : tensor<16xf32>
    %v302 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v303 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v304 = stablehlo.divide %v295, %v302 : tensor<16xf32>
    %v305 = stablehlo.divide %v301, %v303 : tensor<16xf32>
    %v306 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v307 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v308 = stablehlo.sqrt %v305 : tensor<16xf32>
    %v309 = stablehlo.add %v308, %v307 : tensor<16xf32>
    %v310 = stablehlo.divide %v304, %v309 : tensor<16xf32>
    %v311 = stablehlo.multiply %v306, %v310 : tensor<16xf32>
    %v312 = stablehlo.subtract %cb1, %v311 : tensor<16xf32>
    %v313 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v314 = stablehlo.multiply %v313, %v306 : tensor<16xf32>
    %v315 = stablehlo.multiply %v314, %cb1 : tensor<16xf32>
    %v316 = stablehlo.subtract %v312, %v315 : tensor<16xf32>
    %v317 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v318 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v319 = stablehlo.multiply %v317, %cb1m : tensor<16xf32>
    %v320 = stablehlo.multiply %v318, %v290 : tensor<16xf32>
    %v321 = stablehlo.add %v319, %v320 : tensor<16xf32>
    %v322 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v323 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v324 = stablehlo.multiply %v322, %cb1v : tensor<16xf32>
    %v325 = stablehlo.multiply %v290, %v290 : tensor<16xf32>
    %v326 = stablehlo.multiply %v323, %v325 : tensor<16xf32>
    %v327 = stablehlo.add %v324, %v326 : tensor<16xf32>
    %v328 = stablehlo.reshape %v11 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v329 = stablehlo.reshape %v233 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v330 = stablehlo.transpose %v328, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v331 = stablehlo.transpose %v329, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v332 = stablehlo.convolution(%v330, %v331)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x16x3x3xf32>
    %v333 = stablehlo.transpose %v332, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v334 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v335 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v336 = stablehlo.multiply %v334, %W2m : tensor<16x16x3x3xf32>
    %v337 = stablehlo.multiply %v335, %v333 : tensor<16x16x3x3xf32>
    %v338 = stablehlo.add %v336, %v337 : tensor<16x16x3x3xf32>
    %v339 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v340 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v341 = stablehlo.multiply %v339, %W2v : tensor<16x16x3x3xf32>
    %v342 = stablehlo.multiply %v333, %v333 : tensor<16x16x3x3xf32>
    %v343 = stablehlo.multiply %v340, %v342 : tensor<16x16x3x3xf32>
    %v344 = stablehlo.add %v341, %v343 : tensor<16x16x3x3xf32>
    %v345 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v346 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v347 = stablehlo.divide %v338, %v345 : tensor<16x16x3x3xf32>
    %v348 = stablehlo.divide %v344, %v346 : tensor<16x16x3x3xf32>
    %v349 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v350 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v351 = stablehlo.sqrt %v348 : tensor<16x16x3x3xf32>
    %v352 = stablehlo.add %v351, %v350 : tensor<16x16x3x3xf32>
    %v353 = stablehlo.divide %v347, %v352 : tensor<16x16x3x3xf32>
    %v354 = stablehlo.multiply %v349, %v353 : tensor<16x16x3x3xf32>
    %v355 = stablehlo.subtract %W2, %v354 : tensor<16x16x3x3xf32>
    %v356 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v357 = stablehlo.multiply %v356, %v349 : tensor<16x16x3x3xf32>
    %v358 = stablehlo.multiply %v357, %W2 : tensor<16x16x3x3xf32>
    %v359 = stablehlo.subtract %v355, %v358 : tensor<16x16x3x3xf32>
    %v360 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v361 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v362 = stablehlo.multiply %v360, %W2m : tensor<16x16x3x3xf32>
    %v363 = stablehlo.multiply %v361, %v333 : tensor<16x16x3x3xf32>
    %v364 = stablehlo.add %v362, %v363 : tensor<16x16x3x3xf32>
    %v365 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v366 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v367 = stablehlo.multiply %v365, %W2v : tensor<16x16x3x3xf32>
    %v368 = stablehlo.multiply %v333, %v333 : tensor<16x16x3x3xf32>
    %v369 = stablehlo.multiply %v366, %v368 : tensor<16x16x3x3xf32>
    %v370 = stablehlo.add %v367, %v369 : tensor<16x16x3x3xf32>
    %v371 = stablehlo.reshape %v233 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v372 = stablehlo.constant dense<0.0> : tensor<f32>
    %v373 = stablehlo.reduce(%v371 init: %v372) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v374 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v375 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v376 = stablehlo.multiply %v374, %cb2m : tensor<16xf32>
    %v377 = stablehlo.multiply %v375, %v373 : tensor<16xf32>
    %v378 = stablehlo.add %v376, %v377 : tensor<16xf32>
    %v379 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v380 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v381 = stablehlo.multiply %v379, %cb2v : tensor<16xf32>
    %v382 = stablehlo.multiply %v373, %v373 : tensor<16xf32>
    %v383 = stablehlo.multiply %v380, %v382 : tensor<16xf32>
    %v384 = stablehlo.add %v381, %v383 : tensor<16xf32>
    %v385 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v386 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v387 = stablehlo.divide %v378, %v385 : tensor<16xf32>
    %v388 = stablehlo.divide %v384, %v386 : tensor<16xf32>
    %v389 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v390 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v391 = stablehlo.sqrt %v388 : tensor<16xf32>
    %v392 = stablehlo.add %v391, %v390 : tensor<16xf32>
    %v393 = stablehlo.divide %v387, %v392 : tensor<16xf32>
    %v394 = stablehlo.multiply %v389, %v393 : tensor<16xf32>
    %v395 = stablehlo.subtract %cb2, %v394 : tensor<16xf32>
    %v396 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v397 = stablehlo.multiply %v396, %v389 : tensor<16xf32>
    %v398 = stablehlo.multiply %v397, %cb2 : tensor<16xf32>
    %v399 = stablehlo.subtract %v395, %v398 : tensor<16xf32>
    %v400 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v401 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v402 = stablehlo.multiply %v400, %cb2m : tensor<16xf32>
    %v403 = stablehlo.multiply %v401, %v373 : tensor<16xf32>
    %v404 = stablehlo.add %v402, %v403 : tensor<16xf32>
    %v405 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v406 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v407 = stablehlo.multiply %v405, %cb2v : tensor<16xf32>
    %v408 = stablehlo.multiply %v373, %v373 : tensor<16xf32>
    %v409 = stablehlo.multiply %v406, %v408 : tensor<16xf32>
    %v410 = stablehlo.add %v407, %v409 : tensor<16xf32>
    %v411 = stablehlo.reshape %v27 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v412 = stablehlo.reshape %v217 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v413 = stablehlo.transpose %v411, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v414 = stablehlo.transpose %v412, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v415 = stablehlo.convolution(%v413, %v414)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v416 = stablehlo.transpose %v415, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v417 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v418 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v419 = stablehlo.multiply %v417, %W3m : tensor<16x16x3x3xf32>
    %v420 = stablehlo.multiply %v418, %v416 : tensor<16x16x3x3xf32>
    %v421 = stablehlo.add %v419, %v420 : tensor<16x16x3x3xf32>
    %v422 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v423 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v424 = stablehlo.multiply %v422, %W3v : tensor<16x16x3x3xf32>
    %v425 = stablehlo.multiply %v416, %v416 : tensor<16x16x3x3xf32>
    %v426 = stablehlo.multiply %v423, %v425 : tensor<16x16x3x3xf32>
    %v427 = stablehlo.add %v424, %v426 : tensor<16x16x3x3xf32>
    %v428 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v429 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v430 = stablehlo.divide %v421, %v428 : tensor<16x16x3x3xf32>
    %v431 = stablehlo.divide %v427, %v429 : tensor<16x16x3x3xf32>
    %v432 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v433 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v434 = stablehlo.sqrt %v431 : tensor<16x16x3x3xf32>
    %v435 = stablehlo.add %v434, %v433 : tensor<16x16x3x3xf32>
    %v436 = stablehlo.divide %v430, %v435 : tensor<16x16x3x3xf32>
    %v437 = stablehlo.multiply %v432, %v436 : tensor<16x16x3x3xf32>
    %v438 = stablehlo.subtract %W3, %v437 : tensor<16x16x3x3xf32>
    %v439 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v440 = stablehlo.multiply %v439, %v432 : tensor<16x16x3x3xf32>
    %v441 = stablehlo.multiply %v440, %W3 : tensor<16x16x3x3xf32>
    %v442 = stablehlo.subtract %v438, %v441 : tensor<16x16x3x3xf32>
    %v443 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v444 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v445 = stablehlo.multiply %v443, %W3m : tensor<16x16x3x3xf32>
    %v446 = stablehlo.multiply %v444, %v416 : tensor<16x16x3x3xf32>
    %v447 = stablehlo.add %v445, %v446 : tensor<16x16x3x3xf32>
    %v448 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v449 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v450 = stablehlo.multiply %v448, %W3v : tensor<16x16x3x3xf32>
    %v451 = stablehlo.multiply %v416, %v416 : tensor<16x16x3x3xf32>
    %v452 = stablehlo.multiply %v449, %v451 : tensor<16x16x3x3xf32>
    %v453 = stablehlo.add %v450, %v452 : tensor<16x16x3x3xf32>
    %v454 = stablehlo.reshape %v217 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v455 = stablehlo.constant dense<0.0> : tensor<f32>
    %v456 = stablehlo.reduce(%v454 init: %v455) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v457 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v458 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v459 = stablehlo.multiply %v457, %cb3m : tensor<16xf32>
    %v460 = stablehlo.multiply %v458, %v456 : tensor<16xf32>
    %v461 = stablehlo.add %v459, %v460 : tensor<16xf32>
    %v462 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v463 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v464 = stablehlo.multiply %v462, %cb3v : tensor<16xf32>
    %v465 = stablehlo.multiply %v456, %v456 : tensor<16xf32>
    %v466 = stablehlo.multiply %v463, %v465 : tensor<16xf32>
    %v467 = stablehlo.add %v464, %v466 : tensor<16xf32>
    %v468 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v469 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v470 = stablehlo.divide %v461, %v468 : tensor<16xf32>
    %v471 = stablehlo.divide %v467, %v469 : tensor<16xf32>
    %v472 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v473 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v474 = stablehlo.sqrt %v471 : tensor<16xf32>
    %v475 = stablehlo.add %v474, %v473 : tensor<16xf32>
    %v476 = stablehlo.divide %v470, %v475 : tensor<16xf32>
    %v477 = stablehlo.multiply %v472, %v476 : tensor<16xf32>
    %v478 = stablehlo.subtract %cb3, %v477 : tensor<16xf32>
    %v479 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v480 = stablehlo.multiply %v479, %v472 : tensor<16xf32>
    %v481 = stablehlo.multiply %v480, %cb3 : tensor<16xf32>
    %v482 = stablehlo.subtract %v478, %v481 : tensor<16xf32>
    %v483 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v484 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v485 = stablehlo.multiply %v483, %cb3m : tensor<16xf32>
    %v486 = stablehlo.multiply %v484, %v456 : tensor<16xf32>
    %v487 = stablehlo.add %v485, %v486 : tensor<16xf32>
    %v488 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v489 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v490 = stablehlo.multiply %v488, %cb3v : tensor<16xf32>
    %v491 = stablehlo.multiply %v456, %v456 : tensor<16xf32>
    %v492 = stablehlo.multiply %v489, %v491 : tensor<16xf32>
    %v493 = stablehlo.add %v490, %v492 : tensor<16xf32>
    %v494 = stablehlo.reshape %v39 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v495 = stablehlo.reshape %v206 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v496 = stablehlo.transpose %v494, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v497 = stablehlo.transpose %v495, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v498 = stablehlo.convolution(%v496, %v497)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v499 = stablehlo.transpose %v498, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v500 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v501 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v502 = stablehlo.multiply %v500, %W4m : tensor<16x16x3x3xf32>
    %v503 = stablehlo.multiply %v501, %v499 : tensor<16x16x3x3xf32>
    %v504 = stablehlo.add %v502, %v503 : tensor<16x16x3x3xf32>
    %v505 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v506 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v507 = stablehlo.multiply %v505, %W4v : tensor<16x16x3x3xf32>
    %v508 = stablehlo.multiply %v499, %v499 : tensor<16x16x3x3xf32>
    %v509 = stablehlo.multiply %v506, %v508 : tensor<16x16x3x3xf32>
    %v510 = stablehlo.add %v507, %v509 : tensor<16x16x3x3xf32>
    %v511 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v512 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v513 = stablehlo.divide %v504, %v511 : tensor<16x16x3x3xf32>
    %v514 = stablehlo.divide %v510, %v512 : tensor<16x16x3x3xf32>
    %v515 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v516 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v517 = stablehlo.sqrt %v514 : tensor<16x16x3x3xf32>
    %v518 = stablehlo.add %v517, %v516 : tensor<16x16x3x3xf32>
    %v519 = stablehlo.divide %v513, %v518 : tensor<16x16x3x3xf32>
    %v520 = stablehlo.multiply %v515, %v519 : tensor<16x16x3x3xf32>
    %v521 = stablehlo.subtract %W4, %v520 : tensor<16x16x3x3xf32>
    %v522 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v523 = stablehlo.multiply %v522, %v515 : tensor<16x16x3x3xf32>
    %v524 = stablehlo.multiply %v523, %W4 : tensor<16x16x3x3xf32>
    %v525 = stablehlo.subtract %v521, %v524 : tensor<16x16x3x3xf32>
    %v526 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v527 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v528 = stablehlo.multiply %v526, %W4m : tensor<16x16x3x3xf32>
    %v529 = stablehlo.multiply %v527, %v499 : tensor<16x16x3x3xf32>
    %v530 = stablehlo.add %v528, %v529 : tensor<16x16x3x3xf32>
    %v531 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v532 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v533 = stablehlo.multiply %v531, %W4v : tensor<16x16x3x3xf32>
    %v534 = stablehlo.multiply %v499, %v499 : tensor<16x16x3x3xf32>
    %v535 = stablehlo.multiply %v532, %v534 : tensor<16x16x3x3xf32>
    %v536 = stablehlo.add %v533, %v535 : tensor<16x16x3x3xf32>
    %v537 = stablehlo.reshape %v206 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v538 = stablehlo.constant dense<0.0> : tensor<f32>
    %v539 = stablehlo.reduce(%v537 init: %v538) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v540 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v541 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v542 = stablehlo.multiply %v540, %cb4m : tensor<16xf32>
    %v543 = stablehlo.multiply %v541, %v539 : tensor<16xf32>
    %v544 = stablehlo.add %v542, %v543 : tensor<16xf32>
    %v545 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v546 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v547 = stablehlo.multiply %v545, %cb4v : tensor<16xf32>
    %v548 = stablehlo.multiply %v539, %v539 : tensor<16xf32>
    %v549 = stablehlo.multiply %v546, %v548 : tensor<16xf32>
    %v550 = stablehlo.add %v547, %v549 : tensor<16xf32>
    %v551 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v552 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v553 = stablehlo.divide %v544, %v551 : tensor<16xf32>
    %v554 = stablehlo.divide %v550, %v552 : tensor<16xf32>
    %v555 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v556 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v557 = stablehlo.sqrt %v554 : tensor<16xf32>
    %v558 = stablehlo.add %v557, %v556 : tensor<16xf32>
    %v559 = stablehlo.divide %v553, %v558 : tensor<16xf32>
    %v560 = stablehlo.multiply %v555, %v559 : tensor<16xf32>
    %v561 = stablehlo.subtract %cb4, %v560 : tensor<16xf32>
    %v562 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v563 = stablehlo.multiply %v562, %v555 : tensor<16xf32>
    %v564 = stablehlo.multiply %v563, %cb4 : tensor<16xf32>
    %v565 = stablehlo.subtract %v561, %v564 : tensor<16xf32>
    %v566 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v567 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v568 = stablehlo.multiply %v566, %cb4m : tensor<16xf32>
    %v569 = stablehlo.multiply %v567, %v539 : tensor<16xf32>
    %v570 = stablehlo.add %v568, %v569 : tensor<16xf32>
    %v571 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v572 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v573 = stablehlo.multiply %v571, %cb4v : tensor<16xf32>
    %v574 = stablehlo.multiply %v539, %v539 : tensor<16xf32>
    %v575 = stablehlo.multiply %v572, %v574 : tensor<16xf32>
    %v576 = stablehlo.add %v573, %v575 : tensor<16xf32>
    %v577 = stablehlo.reshape %v55 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v578 = stablehlo.reshape %v190 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v579 = stablehlo.transpose %v577, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v580 = stablehlo.transpose %v578, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v581 = stablehlo.convolution(%v579, %v580)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<16x32x3x3xf32>
    %v582 = stablehlo.transpose %v581, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v583 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v584 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v585 = stablehlo.multiply %v583, %W5m : tensor<32x16x3x3xf32>
    %v586 = stablehlo.multiply %v584, %v582 : tensor<32x16x3x3xf32>
    %v587 = stablehlo.add %v585, %v586 : tensor<32x16x3x3xf32>
    %v588 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v589 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v590 = stablehlo.multiply %v588, %W5v : tensor<32x16x3x3xf32>
    %v591 = stablehlo.multiply %v582, %v582 : tensor<32x16x3x3xf32>
    %v592 = stablehlo.multiply %v589, %v591 : tensor<32x16x3x3xf32>
    %v593 = stablehlo.add %v590, %v592 : tensor<32x16x3x3xf32>
    %v594 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v595 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v596 = stablehlo.divide %v587, %v594 : tensor<32x16x3x3xf32>
    %v597 = stablehlo.divide %v593, %v595 : tensor<32x16x3x3xf32>
    %v598 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v599 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v600 = stablehlo.sqrt %v597 : tensor<32x16x3x3xf32>
    %v601 = stablehlo.add %v600, %v599 : tensor<32x16x3x3xf32>
    %v602 = stablehlo.divide %v596, %v601 : tensor<32x16x3x3xf32>
    %v603 = stablehlo.multiply %v598, %v602 : tensor<32x16x3x3xf32>
    %v604 = stablehlo.subtract %W5, %v603 : tensor<32x16x3x3xf32>
    %v605 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v606 = stablehlo.multiply %v605, %v598 : tensor<32x16x3x3xf32>
    %v607 = stablehlo.multiply %v606, %W5 : tensor<32x16x3x3xf32>
    %v608 = stablehlo.subtract %v604, %v607 : tensor<32x16x3x3xf32>
    %v609 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v610 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v611 = stablehlo.multiply %v609, %W5m : tensor<32x16x3x3xf32>
    %v612 = stablehlo.multiply %v610, %v582 : tensor<32x16x3x3xf32>
    %v613 = stablehlo.add %v611, %v612 : tensor<32x16x3x3xf32>
    %v614 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v615 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v616 = stablehlo.multiply %v614, %W5v : tensor<32x16x3x3xf32>
    %v617 = stablehlo.multiply %v582, %v582 : tensor<32x16x3x3xf32>
    %v618 = stablehlo.multiply %v615, %v617 : tensor<32x16x3x3xf32>
    %v619 = stablehlo.add %v616, %v618 : tensor<32x16x3x3xf32>
    %v620 = stablehlo.reshape %v190 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v621 = stablehlo.constant dense<0.0> : tensor<f32>
    %v622 = stablehlo.reduce(%v620 init: %v621) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v623 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v624 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v625 = stablehlo.multiply %v623, %cb5m : tensor<32xf32>
    %v626 = stablehlo.multiply %v624, %v622 : tensor<32xf32>
    %v627 = stablehlo.add %v625, %v626 : tensor<32xf32>
    %v628 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v629 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v630 = stablehlo.multiply %v628, %cb5v : tensor<32xf32>
    %v631 = stablehlo.multiply %v622, %v622 : tensor<32xf32>
    %v632 = stablehlo.multiply %v629, %v631 : tensor<32xf32>
    %v633 = stablehlo.add %v630, %v632 : tensor<32xf32>
    %v634 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v635 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v636 = stablehlo.divide %v627, %v634 : tensor<32xf32>
    %v637 = stablehlo.divide %v633, %v635 : tensor<32xf32>
    %v638 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v639 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v640 = stablehlo.sqrt %v637 : tensor<32xf32>
    %v641 = stablehlo.add %v640, %v639 : tensor<32xf32>
    %v642 = stablehlo.divide %v636, %v641 : tensor<32xf32>
    %v643 = stablehlo.multiply %v638, %v642 : tensor<32xf32>
    %v644 = stablehlo.subtract %cb5, %v643 : tensor<32xf32>
    %v645 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v646 = stablehlo.multiply %v645, %v638 : tensor<32xf32>
    %v647 = stablehlo.multiply %v646, %cb5 : tensor<32xf32>
    %v648 = stablehlo.subtract %v644, %v647 : tensor<32xf32>
    %v649 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v650 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v651 = stablehlo.multiply %v649, %cb5m : tensor<32xf32>
    %v652 = stablehlo.multiply %v650, %v622 : tensor<32xf32>
    %v653 = stablehlo.add %v651, %v652 : tensor<32xf32>
    %v654 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v655 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v656 = stablehlo.multiply %v654, %cb5v : tensor<32xf32>
    %v657 = stablehlo.multiply %v622, %v622 : tensor<32xf32>
    %v658 = stablehlo.multiply %v655, %v657 : tensor<32xf32>
    %v659 = stablehlo.add %v656, %v658 : tensor<32xf32>
    %v660 = stablehlo.reshape %v67 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v661 = stablehlo.reshape %v179 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v662 = stablehlo.transpose %v660, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v663 = stablehlo.transpose %v661, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v664 = stablehlo.convolution(%v662, %v663)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<32x32x3x3xf32>
    %v665 = stablehlo.transpose %v664, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v666 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v667 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v668 = stablehlo.multiply %v666, %W6m : tensor<32x32x3x3xf32>
    %v669 = stablehlo.multiply %v667, %v665 : tensor<32x32x3x3xf32>
    %v670 = stablehlo.add %v668, %v669 : tensor<32x32x3x3xf32>
    %v671 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v672 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v673 = stablehlo.multiply %v671, %W6v : tensor<32x32x3x3xf32>
    %v674 = stablehlo.multiply %v665, %v665 : tensor<32x32x3x3xf32>
    %v675 = stablehlo.multiply %v672, %v674 : tensor<32x32x3x3xf32>
    %v676 = stablehlo.add %v673, %v675 : tensor<32x32x3x3xf32>
    %v677 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v678 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v679 = stablehlo.divide %v670, %v677 : tensor<32x32x3x3xf32>
    %v680 = stablehlo.divide %v676, %v678 : tensor<32x32x3x3xf32>
    %v681 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v682 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v683 = stablehlo.sqrt %v680 : tensor<32x32x3x3xf32>
    %v684 = stablehlo.add %v683, %v682 : tensor<32x32x3x3xf32>
    %v685 = stablehlo.divide %v679, %v684 : tensor<32x32x3x3xf32>
    %v686 = stablehlo.multiply %v681, %v685 : tensor<32x32x3x3xf32>
    %v687 = stablehlo.subtract %W6, %v686 : tensor<32x32x3x3xf32>
    %v688 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v689 = stablehlo.multiply %v688, %v681 : tensor<32x32x3x3xf32>
    %v690 = stablehlo.multiply %v689, %W6 : tensor<32x32x3x3xf32>
    %v691 = stablehlo.subtract %v687, %v690 : tensor<32x32x3x3xf32>
    %v692 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v693 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v694 = stablehlo.multiply %v692, %W6m : tensor<32x32x3x3xf32>
    %v695 = stablehlo.multiply %v693, %v665 : tensor<32x32x3x3xf32>
    %v696 = stablehlo.add %v694, %v695 : tensor<32x32x3x3xf32>
    %v697 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v698 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v699 = stablehlo.multiply %v697, %W6v : tensor<32x32x3x3xf32>
    %v700 = stablehlo.multiply %v665, %v665 : tensor<32x32x3x3xf32>
    %v701 = stablehlo.multiply %v698, %v700 : tensor<32x32x3x3xf32>
    %v702 = stablehlo.add %v699, %v701 : tensor<32x32x3x3xf32>
    %v703 = stablehlo.reshape %v179 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v704 = stablehlo.constant dense<0.0> : tensor<f32>
    %v705 = stablehlo.reduce(%v703 init: %v704) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v706 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v707 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v708 = stablehlo.multiply %v706, %cb6m : tensor<32xf32>
    %v709 = stablehlo.multiply %v707, %v705 : tensor<32xf32>
    %v710 = stablehlo.add %v708, %v709 : tensor<32xf32>
    %v711 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v712 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v713 = stablehlo.multiply %v711, %cb6v : tensor<32xf32>
    %v714 = stablehlo.multiply %v705, %v705 : tensor<32xf32>
    %v715 = stablehlo.multiply %v712, %v714 : tensor<32xf32>
    %v716 = stablehlo.add %v713, %v715 : tensor<32xf32>
    %v717 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v718 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v719 = stablehlo.divide %v710, %v717 : tensor<32xf32>
    %v720 = stablehlo.divide %v716, %v718 : tensor<32xf32>
    %v721 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v722 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v723 = stablehlo.sqrt %v720 : tensor<32xf32>
    %v724 = stablehlo.add %v723, %v722 : tensor<32xf32>
    %v725 = stablehlo.divide %v719, %v724 : tensor<32xf32>
    %v726 = stablehlo.multiply %v721, %v725 : tensor<32xf32>
    %v727 = stablehlo.subtract %cb6, %v726 : tensor<32xf32>
    %v728 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v729 = stablehlo.multiply %v728, %v721 : tensor<32xf32>
    %v730 = stablehlo.multiply %v729, %cb6 : tensor<32xf32>
    %v731 = stablehlo.subtract %v727, %v730 : tensor<32xf32>
    %v732 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v733 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v734 = stablehlo.multiply %v732, %cb6m : tensor<32xf32>
    %v735 = stablehlo.multiply %v733, %v705 : tensor<32xf32>
    %v736 = stablehlo.add %v734, %v735 : tensor<32xf32>
    %v737 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v738 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v739 = stablehlo.multiply %v737, %cb6v : tensor<32xf32>
    %v740 = stablehlo.multiply %v705, %v705 : tensor<32xf32>
    %v741 = stablehlo.multiply %v738, %v740 : tensor<32xf32>
    %v742 = stablehlo.add %v739, %v741 : tensor<32xf32>
    %v743 = stablehlo.reshape %v83 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v744 = stablehlo.reshape %v163 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v745 = stablehlo.transpose %v743, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v746 = stablehlo.transpose %v744, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v747 = stablehlo.convolution(%v745, %v746)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v748 = stablehlo.transpose %v747, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v749 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v750 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v751 = stablehlo.multiply %v749, %W7m : tensor<32x32x3x3xf32>
    %v752 = stablehlo.multiply %v750, %v748 : tensor<32x32x3x3xf32>
    %v753 = stablehlo.add %v751, %v752 : tensor<32x32x3x3xf32>
    %v754 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v755 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v756 = stablehlo.multiply %v754, %W7v : tensor<32x32x3x3xf32>
    %v757 = stablehlo.multiply %v748, %v748 : tensor<32x32x3x3xf32>
    %v758 = stablehlo.multiply %v755, %v757 : tensor<32x32x3x3xf32>
    %v759 = stablehlo.add %v756, %v758 : tensor<32x32x3x3xf32>
    %v760 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v761 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v762 = stablehlo.divide %v753, %v760 : tensor<32x32x3x3xf32>
    %v763 = stablehlo.divide %v759, %v761 : tensor<32x32x3x3xf32>
    %v764 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v765 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v766 = stablehlo.sqrt %v763 : tensor<32x32x3x3xf32>
    %v767 = stablehlo.add %v766, %v765 : tensor<32x32x3x3xf32>
    %v768 = stablehlo.divide %v762, %v767 : tensor<32x32x3x3xf32>
    %v769 = stablehlo.multiply %v764, %v768 : tensor<32x32x3x3xf32>
    %v770 = stablehlo.subtract %W7, %v769 : tensor<32x32x3x3xf32>
    %v771 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v772 = stablehlo.multiply %v771, %v764 : tensor<32x32x3x3xf32>
    %v773 = stablehlo.multiply %v772, %W7 : tensor<32x32x3x3xf32>
    %v774 = stablehlo.subtract %v770, %v773 : tensor<32x32x3x3xf32>
    %v775 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v776 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v777 = stablehlo.multiply %v775, %W7m : tensor<32x32x3x3xf32>
    %v778 = stablehlo.multiply %v776, %v748 : tensor<32x32x3x3xf32>
    %v779 = stablehlo.add %v777, %v778 : tensor<32x32x3x3xf32>
    %v780 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v781 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v782 = stablehlo.multiply %v780, %W7v : tensor<32x32x3x3xf32>
    %v783 = stablehlo.multiply %v748, %v748 : tensor<32x32x3x3xf32>
    %v784 = stablehlo.multiply %v781, %v783 : tensor<32x32x3x3xf32>
    %v785 = stablehlo.add %v782, %v784 : tensor<32x32x3x3xf32>
    %v786 = stablehlo.reshape %v163 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v787 = stablehlo.constant dense<0.0> : tensor<f32>
    %v788 = stablehlo.reduce(%v786 init: %v787) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v789 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v790 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v791 = stablehlo.multiply %v789, %cb7m : tensor<32xf32>
    %v792 = stablehlo.multiply %v790, %v788 : tensor<32xf32>
    %v793 = stablehlo.add %v791, %v792 : tensor<32xf32>
    %v794 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v795 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v796 = stablehlo.multiply %v794, %cb7v : tensor<32xf32>
    %v797 = stablehlo.multiply %v788, %v788 : tensor<32xf32>
    %v798 = stablehlo.multiply %v795, %v797 : tensor<32xf32>
    %v799 = stablehlo.add %v796, %v798 : tensor<32xf32>
    %v800 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v801 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v802 = stablehlo.divide %v793, %v800 : tensor<32xf32>
    %v803 = stablehlo.divide %v799, %v801 : tensor<32xf32>
    %v804 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v805 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v806 = stablehlo.sqrt %v803 : tensor<32xf32>
    %v807 = stablehlo.add %v806, %v805 : tensor<32xf32>
    %v808 = stablehlo.divide %v802, %v807 : tensor<32xf32>
    %v809 = stablehlo.multiply %v804, %v808 : tensor<32xf32>
    %v810 = stablehlo.subtract %cb7, %v809 : tensor<32xf32>
    %v811 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v812 = stablehlo.multiply %v811, %v804 : tensor<32xf32>
    %v813 = stablehlo.multiply %v812, %cb7 : tensor<32xf32>
    %v814 = stablehlo.subtract %v810, %v813 : tensor<32xf32>
    %v815 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v816 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v817 = stablehlo.multiply %v815, %cb7m : tensor<32xf32>
    %v818 = stablehlo.multiply %v816, %v788 : tensor<32xf32>
    %v819 = stablehlo.add %v817, %v818 : tensor<32xf32>
    %v820 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v821 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v822 = stablehlo.multiply %v820, %cb7v : tensor<32xf32>
    %v823 = stablehlo.multiply %v788, %v788 : tensor<32xf32>
    %v824 = stablehlo.multiply %v821, %v823 : tensor<32xf32>
    %v825 = stablehlo.add %v822, %v824 : tensor<32xf32>
    %v826 = stablehlo.reshape %v95 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v827 = stablehlo.reshape %v152 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v828 = stablehlo.transpose %v826, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v829 = stablehlo.transpose %v827, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v830 = stablehlo.convolution(%v828, %v829)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v831 = stablehlo.transpose %v830, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v832 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v833 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v834 = stablehlo.multiply %v832, %W8m : tensor<32x32x3x3xf32>
    %v835 = stablehlo.multiply %v833, %v831 : tensor<32x32x3x3xf32>
    %v836 = stablehlo.add %v834, %v835 : tensor<32x32x3x3xf32>
    %v837 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v838 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v839 = stablehlo.multiply %v837, %W8v : tensor<32x32x3x3xf32>
    %v840 = stablehlo.multiply %v831, %v831 : tensor<32x32x3x3xf32>
    %v841 = stablehlo.multiply %v838, %v840 : tensor<32x32x3x3xf32>
    %v842 = stablehlo.add %v839, %v841 : tensor<32x32x3x3xf32>
    %v843 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v844 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v845 = stablehlo.divide %v836, %v843 : tensor<32x32x3x3xf32>
    %v846 = stablehlo.divide %v842, %v844 : tensor<32x32x3x3xf32>
    %v847 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v848 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v849 = stablehlo.sqrt %v846 : tensor<32x32x3x3xf32>
    %v850 = stablehlo.add %v849, %v848 : tensor<32x32x3x3xf32>
    %v851 = stablehlo.divide %v845, %v850 : tensor<32x32x3x3xf32>
    %v852 = stablehlo.multiply %v847, %v851 : tensor<32x32x3x3xf32>
    %v853 = stablehlo.subtract %W8, %v852 : tensor<32x32x3x3xf32>
    %v854 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v855 = stablehlo.multiply %v854, %v847 : tensor<32x32x3x3xf32>
    %v856 = stablehlo.multiply %v855, %W8 : tensor<32x32x3x3xf32>
    %v857 = stablehlo.subtract %v853, %v856 : tensor<32x32x3x3xf32>
    %v858 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v859 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v860 = stablehlo.multiply %v858, %W8m : tensor<32x32x3x3xf32>
    %v861 = stablehlo.multiply %v859, %v831 : tensor<32x32x3x3xf32>
    %v862 = stablehlo.add %v860, %v861 : tensor<32x32x3x3xf32>
    %v863 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v864 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v865 = stablehlo.multiply %v863, %W8v : tensor<32x32x3x3xf32>
    %v866 = stablehlo.multiply %v831, %v831 : tensor<32x32x3x3xf32>
    %v867 = stablehlo.multiply %v864, %v866 : tensor<32x32x3x3xf32>
    %v868 = stablehlo.add %v865, %v867 : tensor<32x32x3x3xf32>
    %v869 = stablehlo.reshape %v152 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v870 = stablehlo.constant dense<0.0> : tensor<f32>
    %v871 = stablehlo.reduce(%v869 init: %v870) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v872 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v873 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v874 = stablehlo.multiply %v872, %cb8m : tensor<32xf32>
    %v875 = stablehlo.multiply %v873, %v871 : tensor<32xf32>
    %v876 = stablehlo.add %v874, %v875 : tensor<32xf32>
    %v877 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v878 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v879 = stablehlo.multiply %v877, %cb8v : tensor<32xf32>
    %v880 = stablehlo.multiply %v871, %v871 : tensor<32xf32>
    %v881 = stablehlo.multiply %v878, %v880 : tensor<32xf32>
    %v882 = stablehlo.add %v879, %v881 : tensor<32xf32>
    %v883 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v884 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v885 = stablehlo.divide %v876, %v883 : tensor<32xf32>
    %v886 = stablehlo.divide %v882, %v884 : tensor<32xf32>
    %v887 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v888 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v889 = stablehlo.sqrt %v886 : tensor<32xf32>
    %v890 = stablehlo.add %v889, %v888 : tensor<32xf32>
    %v891 = stablehlo.divide %v885, %v890 : tensor<32xf32>
    %v892 = stablehlo.multiply %v887, %v891 : tensor<32xf32>
    %v893 = stablehlo.subtract %cb8, %v892 : tensor<32xf32>
    %v894 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v895 = stablehlo.multiply %v894, %v887 : tensor<32xf32>
    %v896 = stablehlo.multiply %v895, %cb8 : tensor<32xf32>
    %v897 = stablehlo.subtract %v893, %v896 : tensor<32xf32>
    %v898 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v899 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v900 = stablehlo.multiply %v898, %cb8m : tensor<32xf32>
    %v901 = stablehlo.multiply %v899, %v871 : tensor<32xf32>
    %v902 = stablehlo.add %v900, %v901 : tensor<32xf32>
    %v903 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v904 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v905 = stablehlo.multiply %v903, %cb8v : tensor<32xf32>
    %v906 = stablehlo.multiply %v871, %v871 : tensor<32xf32>
    %v907 = stablehlo.multiply %v904, %v906 : tensor<32xf32>
    %v908 = stablehlo.add %v905, %v907 : tensor<32xf32>
    %v909 = stablehlo.dot_general %v111, %v140, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %v910 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v911 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v912 = stablehlo.multiply %v910, %W9m : tensor<128x64xf32>
    %v913 = stablehlo.multiply %v911, %v909 : tensor<128x64xf32>
    %v914 = stablehlo.add %v912, %v913 : tensor<128x64xf32>
    %v915 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v916 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v917 = stablehlo.multiply %v915, %W9v : tensor<128x64xf32>
    %v918 = stablehlo.multiply %v909, %v909 : tensor<128x64xf32>
    %v919 = stablehlo.multiply %v916, %v918 : tensor<128x64xf32>
    %v920 = stablehlo.add %v917, %v919 : tensor<128x64xf32>
    %v921 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v922 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v923 = stablehlo.divide %v914, %v921 : tensor<128x64xf32>
    %v924 = stablehlo.divide %v920, %v922 : tensor<128x64xf32>
    %v925 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v926 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v927 = stablehlo.sqrt %v924 : tensor<128x64xf32>
    %v928 = stablehlo.add %v927, %v926 : tensor<128x64xf32>
    %v929 = stablehlo.divide %v923, %v928 : tensor<128x64xf32>
    %v930 = stablehlo.multiply %v925, %v929 : tensor<128x64xf32>
    %v931 = stablehlo.subtract %W9, %v930 : tensor<128x64xf32>
    %v932 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v933 = stablehlo.multiply %v932, %v925 : tensor<128x64xf32>
    %v934 = stablehlo.multiply %v933, %W9 : tensor<128x64xf32>
    %v935 = stablehlo.subtract %v931, %v934 : tensor<128x64xf32>
    %v936 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v937 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v938 = stablehlo.multiply %v936, %W9m : tensor<128x64xf32>
    %v939 = stablehlo.multiply %v937, %v909 : tensor<128x64xf32>
    %v940 = stablehlo.add %v938, %v939 : tensor<128x64xf32>
    %v941 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v942 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v943 = stablehlo.multiply %v941, %W9v : tensor<128x64xf32>
    %v944 = stablehlo.multiply %v909, %v909 : tensor<128x64xf32>
    %v945 = stablehlo.multiply %v942, %v944 : tensor<128x64xf32>
    %v946 = stablehlo.add %v943, %v945 : tensor<128x64xf32>
    %v947 = stablehlo.constant dense<0.0> : tensor<f32>
    %v948 = stablehlo.reduce(%v140 init: %v947) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v949 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v950 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v951 = stablehlo.multiply %v949, %b9m : tensor<64xf32>
    %v952 = stablehlo.multiply %v950, %v948 : tensor<64xf32>
    %v953 = stablehlo.add %v951, %v952 : tensor<64xf32>
    %v954 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v955 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v956 = stablehlo.multiply %v954, %b9v : tensor<64xf32>
    %v957 = stablehlo.multiply %v948, %v948 : tensor<64xf32>
    %v958 = stablehlo.multiply %v955, %v957 : tensor<64xf32>
    %v959 = stablehlo.add %v956, %v958 : tensor<64xf32>
    %v960 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v961 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v962 = stablehlo.divide %v953, %v960 : tensor<64xf32>
    %v963 = stablehlo.divide %v959, %v961 : tensor<64xf32>
    %v964 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v965 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v966 = stablehlo.sqrt %v963 : tensor<64xf32>
    %v967 = stablehlo.add %v966, %v965 : tensor<64xf32>
    %v968 = stablehlo.divide %v962, %v967 : tensor<64xf32>
    %v969 = stablehlo.multiply %v964, %v968 : tensor<64xf32>
    %v970 = stablehlo.subtract %b9, %v969 : tensor<64xf32>
    %v971 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v972 = stablehlo.multiply %v971, %v964 : tensor<64xf32>
    %v973 = stablehlo.multiply %v972, %b9 : tensor<64xf32>
    %v974 = stablehlo.subtract %v970, %v973 : tensor<64xf32>
    %v975 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v976 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v977 = stablehlo.multiply %v975, %b9m : tensor<64xf32>
    %v978 = stablehlo.multiply %v976, %v948 : tensor<64xf32>
    %v979 = stablehlo.add %v977, %v978 : tensor<64xf32>
    %v980 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v981 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v982 = stablehlo.multiply %v980, %b9v : tensor<64xf32>
    %v983 = stablehlo.multiply %v948, %v948 : tensor<64xf32>
    %v984 = stablehlo.multiply %v981, %v983 : tensor<64xf32>
    %v985 = stablehlo.add %v982, %v984 : tensor<64xf32>
    %v986 = stablehlo.dot_general %v116, %v136, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<64x64xf32>
    %v987 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v988 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v989 = stablehlo.multiply %v987, %Wam : tensor<64x64xf32>
    %v990 = stablehlo.multiply %v988, %v986 : tensor<64x64xf32>
    %v991 = stablehlo.add %v989, %v990 : tensor<64x64xf32>
    %v992 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v993 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v994 = stablehlo.multiply %v992, %Wav : tensor<64x64xf32>
    %v995 = stablehlo.multiply %v986, %v986 : tensor<64x64xf32>
    %v996 = stablehlo.multiply %v993, %v995 : tensor<64x64xf32>
    %v997 = stablehlo.add %v994, %v996 : tensor<64x64xf32>
    %v998 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v999 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1000 = stablehlo.divide %v991, %v998 : tensor<64x64xf32>
    %v1001 = stablehlo.divide %v997, %v999 : tensor<64x64xf32>
    %v1002 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1003 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1004 = stablehlo.sqrt %v1001 : tensor<64x64xf32>
    %v1005 = stablehlo.add %v1004, %v1003 : tensor<64x64xf32>
    %v1006 = stablehlo.divide %v1000, %v1005 : tensor<64x64xf32>
    %v1007 = stablehlo.multiply %v1002, %v1006 : tensor<64x64xf32>
    %v1008 = stablehlo.subtract %Wa, %v1007 : tensor<64x64xf32>
    %v1009 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1010 = stablehlo.multiply %v1009, %v1002 : tensor<64x64xf32>
    %v1011 = stablehlo.multiply %v1010, %Wa : tensor<64x64xf32>
    %v1012 = stablehlo.subtract %v1008, %v1011 : tensor<64x64xf32>
    %v1013 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1014 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1015 = stablehlo.multiply %v1013, %Wam : tensor<64x64xf32>
    %v1016 = stablehlo.multiply %v1014, %v986 : tensor<64x64xf32>
    %v1017 = stablehlo.add %v1015, %v1016 : tensor<64x64xf32>
    %v1018 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1019 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v1020 = stablehlo.multiply %v1018, %Wav : tensor<64x64xf32>
    %v1021 = stablehlo.multiply %v986, %v986 : tensor<64x64xf32>
    %v1022 = stablehlo.multiply %v1019, %v1021 : tensor<64x64xf32>
    %v1023 = stablehlo.add %v1020, %v1022 : tensor<64x64xf32>
    %v1024 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1025 = stablehlo.reduce(%v136 init: %v1024) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v1026 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1027 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1028 = stablehlo.multiply %v1026, %bam : tensor<64xf32>
    %v1029 = stablehlo.multiply %v1027, %v1025 : tensor<64xf32>
    %v1030 = stablehlo.add %v1028, %v1029 : tensor<64xf32>
    %v1031 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1032 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1033 = stablehlo.multiply %v1031, %bav : tensor<64xf32>
    %v1034 = stablehlo.multiply %v1025, %v1025 : tensor<64xf32>
    %v1035 = stablehlo.multiply %v1032, %v1034 : tensor<64xf32>
    %v1036 = stablehlo.add %v1033, %v1035 : tensor<64xf32>
    %v1037 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1038 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1039 = stablehlo.divide %v1030, %v1037 : tensor<64xf32>
    %v1040 = stablehlo.divide %v1036, %v1038 : tensor<64xf32>
    %v1041 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1042 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1043 = stablehlo.sqrt %v1040 : tensor<64xf32>
    %v1044 = stablehlo.add %v1043, %v1042 : tensor<64xf32>
    %v1045 = stablehlo.divide %v1039, %v1044 : tensor<64xf32>
    %v1046 = stablehlo.multiply %v1041, %v1045 : tensor<64xf32>
    %v1047 = stablehlo.subtract %ba, %v1046 : tensor<64xf32>
    %v1048 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1049 = stablehlo.multiply %v1048, %v1041 : tensor<64xf32>
    %v1050 = stablehlo.multiply %v1049, %ba : tensor<64xf32>
    %v1051 = stablehlo.subtract %v1047, %v1050 : tensor<64xf32>
    %v1052 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1053 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1054 = stablehlo.multiply %v1052, %bam : tensor<64xf32>
    %v1055 = stablehlo.multiply %v1053, %v1025 : tensor<64xf32>
    %v1056 = stablehlo.add %v1054, %v1055 : tensor<64xf32>
    %v1057 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1058 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v1059 = stablehlo.multiply %v1057, %bav : tensor<64xf32>
    %v1060 = stablehlo.multiply %v1025, %v1025 : tensor<64xf32>
    %v1061 = stablehlo.multiply %v1058, %v1060 : tensor<64xf32>
    %v1062 = stablehlo.add %v1059, %v1061 : tensor<64xf32>
    %v1063 = stablehlo.dot_general %v121, %v132, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x10xf32>) -> tensor<64x10xf32>
    %v1064 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1065 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1066 = stablehlo.multiply %v1064, %Wbm : tensor<64x10xf32>
    %v1067 = stablehlo.multiply %v1065, %v1063 : tensor<64x10xf32>
    %v1068 = stablehlo.add %v1066, %v1067 : tensor<64x10xf32>
    %v1069 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1070 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1071 = stablehlo.multiply %v1069, %Wbv : tensor<64x10xf32>
    %v1072 = stablehlo.multiply %v1063, %v1063 : tensor<64x10xf32>
    %v1073 = stablehlo.multiply %v1070, %v1072 : tensor<64x10xf32>
    %v1074 = stablehlo.add %v1071, %v1073 : tensor<64x10xf32>
    %v1075 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1076 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1077 = stablehlo.divide %v1068, %v1075 : tensor<64x10xf32>
    %v1078 = stablehlo.divide %v1074, %v1076 : tensor<64x10xf32>
    %v1079 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1080 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1081 = stablehlo.sqrt %v1078 : tensor<64x10xf32>
    %v1082 = stablehlo.add %v1081, %v1080 : tensor<64x10xf32>
    %v1083 = stablehlo.divide %v1077, %v1082 : tensor<64x10xf32>
    %v1084 = stablehlo.multiply %v1079, %v1083 : tensor<64x10xf32>
    %v1085 = stablehlo.subtract %Wb, %v1084 : tensor<64x10xf32>
    %v1086 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1087 = stablehlo.multiply %v1086, %v1079 : tensor<64x10xf32>
    %v1088 = stablehlo.multiply %v1087, %Wb : tensor<64x10xf32>
    %v1089 = stablehlo.subtract %v1085, %v1088 : tensor<64x10xf32>
    %v1090 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1091 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1092 = stablehlo.multiply %v1090, %Wbm : tensor<64x10xf32>
    %v1093 = stablehlo.multiply %v1091, %v1063 : tensor<64x10xf32>
    %v1094 = stablehlo.add %v1092, %v1093 : tensor<64x10xf32>
    %v1095 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1096 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v1097 = stablehlo.multiply %v1095, %Wbv : tensor<64x10xf32>
    %v1098 = stablehlo.multiply %v1063, %v1063 : tensor<64x10xf32>
    %v1099 = stablehlo.multiply %v1096, %v1098 : tensor<64x10xf32>
    %v1100 = stablehlo.add %v1097, %v1099 : tensor<64x10xf32>
    %v1101 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1102 = stablehlo.reduce(%v132 init: %v1101) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1103 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1104 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1105 = stablehlo.multiply %v1103, %bbm : tensor<10xf32>
    %v1106 = stablehlo.multiply %v1104, %v1102 : tensor<10xf32>
    %v1107 = stablehlo.add %v1105, %v1106 : tensor<10xf32>
    %v1108 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1109 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1110 = stablehlo.multiply %v1108, %bbv : tensor<10xf32>
    %v1111 = stablehlo.multiply %v1102, %v1102 : tensor<10xf32>
    %v1112 = stablehlo.multiply %v1109, %v1111 : tensor<10xf32>
    %v1113 = stablehlo.add %v1110, %v1112 : tensor<10xf32>
    %v1114 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1115 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1116 = stablehlo.divide %v1107, %v1114 : tensor<10xf32>
    %v1117 = stablehlo.divide %v1113, %v1115 : tensor<10xf32>
    %v1118 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1119 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1120 = stablehlo.sqrt %v1117 : tensor<10xf32>
    %v1121 = stablehlo.add %v1120, %v1119 : tensor<10xf32>
    %v1122 = stablehlo.divide %v1116, %v1121 : tensor<10xf32>
    %v1123 = stablehlo.multiply %v1118, %v1122 : tensor<10xf32>
    %v1124 = stablehlo.subtract %bb, %v1123 : tensor<10xf32>
    %v1125 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1126 = stablehlo.multiply %v1125, %v1118 : tensor<10xf32>
    %v1127 = stablehlo.multiply %v1126, %bb : tensor<10xf32>
    %v1128 = stablehlo.subtract %v1124, %v1127 : tensor<10xf32>
    %v1129 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1130 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1131 = stablehlo.multiply %v1129, %bbm : tensor<10xf32>
    %v1132 = stablehlo.multiply %v1130, %v1102 : tensor<10xf32>
    %v1133 = stablehlo.add %v1131, %v1132 : tensor<10xf32>
    %v1134 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1135 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1136 = stablehlo.multiply %v1134, %bbv : tensor<10xf32>
    %v1137 = stablehlo.multiply %v1102, %v1102 : tensor<10xf32>
    %v1138 = stablehlo.multiply %v1135, %v1137 : tensor<10xf32>
    %v1139 = stablehlo.add %v1136, %v1138 : tensor<10xf32>
    return %v276, %v316, %v359, %v399, %v442, %v482, %v525, %v565, %v608, %v648, %v691, %v731, %v774, %v814, %v857, %v897, %v935, %v974, %v1012, %v1051, %v1089, %v1128, %v281, %v321, %v364, %v404, %v447, %v487, %v530, %v570, %v613, %v653, %v696, %v736, %v779, %v819, %v862, %v902, %v940, %v979, %v1017, %v1056, %v1094, %v1133, %v287, %v327, %v370, %v410, %v453, %v493, %v536, %v576, %v619, %v659, %v702, %v742, %v785, %v825, %v868, %v908, %v946, %v985, %v1023, %v1062, %v1100, %v1139, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
