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
    %v112 = stablehlo.dot_general %v111, %W9, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x512xf32>) -> tensor<128x512xf32>
    %v113 = stablehlo.broadcast_in_dim %b9, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v114 = stablehlo.add %v112, %v113 : tensor<128x512xf32>
    %v115 = stablehlo.reshape %v114 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v116 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v117 = stablehlo.maximum %v115, %v116 : tensor<128x32x4x4xf32>
    %v118 = stablehlo.reshape %v117 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v119 = stablehlo.dot_general %v118, %Wa, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %v120 = stablehlo.broadcast_in_dim %ba, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v121 = stablehlo.add %v119, %v120 : tensor<128x512xf32>
    %v122 = stablehlo.reshape %v121 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v123 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v124 = stablehlo.maximum %v122, %v123 : tensor<128x32x4x4xf32>
    %v125 = stablehlo.reshape %v124 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v126 = stablehlo.dot_general %v125, %Wb, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x10xf32>) -> tensor<128x10xf32>
    %v127 = stablehlo.broadcast_in_dim %bb, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v128 = stablehlo.add %v126, %v127 : tensor<128x10xf32>
    %v129 = stablehlo.reshape %v128 : (tensor<128x10xf32>) -> tensor<128x1x10xf32>
    %v130 = stablehlo.constant dense<0.0> : tensor<f32>
    %v131 = stablehlo.exponential %v129 : tensor<128x1x10xf32>
    %v132 = stablehlo.reduce(%v131 init: %v130) applies stablehlo.add across dimensions = [2] : (tensor<128x1x10xf32>, tensor<f32>) -> tensor<128x1xf32>
    %v133 = stablehlo.broadcast_in_dim %v132, dims = [0, 1] : (tensor<128x1xf32>) -> tensor<128x1x10xf32>
    %v134 = stablehlo.divide %v131, %v133 : tensor<128x1x10xf32>
    %v135 = stablehlo.reshape %v134 : (tensor<128x1x10xf32>) -> tensor<128x10xf32>
    %v136 = stablehlo.subtract %v135, %onehot : tensor<128x10xf32>
    %v137 = stablehlo.constant dense<0.0078125> : tensor<128x10xf32>
    %v138 = stablehlo.multiply %v136, %v137 : tensor<128x10xf32>
    // ── report-only scalar loss (NOT pretty(AST): the kit has no rank-0 loss op; it
    //    feeds no parameter, only the driver's progress line) ──
    %llog = stablehlo.log %v135 : tensor<128x10xf32>
    %ohll = stablehlo.multiply %onehot, %llog : tensor<128x10xf32>
    %csum = stablehlo.reduce(%ohll init: %lzero) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<f32>
    %cneg = stablehlo.negate %csum : tensor<f32>
    %lbf = stablehlo.constant dense<128.0> : tensor<f32>
    %loss = stablehlo.divide %cneg, %lbf : tensor<f32>
    %v139 = stablehlo.reshape %v138 : (tensor<128x10xf32>) -> tensor<128x1x10xf32>
    %v140 = stablehlo.dot_general %v139, %Wb, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x10xf32>, tensor<512x10xf32>) -> tensor<128x1x512xf32>
    %v141 = stablehlo.reshape %v140 : (tensor<128x1x512xf32>) -> tensor<128x512xf32>
    %v142 = stablehlo.reshape %v141 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v143 = stablehlo.reshape %v121 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v144 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v145 = stablehlo.compare GT, %v143, %v144 : (tensor<128x32x4x4xf32>, tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xi1>
    %v146 = stablehlo.select %v145, %v142, %v144 : tensor<128x32x4x4xi1>, tensor<128x32x4x4xf32>
    %v147 = stablehlo.reshape %v146 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v148 = stablehlo.reshape %v147 : (tensor<128x512xf32>) -> tensor<128x1x512xf32>
    %v149 = stablehlo.dot_general %v148, %Wa, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x512xf32>, tensor<512x512xf32>) -> tensor<128x1x512xf32>
    %v150 = stablehlo.reshape %v149 : (tensor<128x1x512xf32>) -> tensor<128x512xf32>
    %v151 = stablehlo.reshape %v150 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v152 = stablehlo.reshape %v114 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v153 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v154 = stablehlo.compare GT, %v152, %v153 : (tensor<128x32x4x4xf32>, tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xi1>
    %v155 = stablehlo.select %v154, %v151, %v153 : tensor<128x32x4x4xi1>, tensor<128x32x4x4xf32>
    %v156 = stablehlo.reshape %v155 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v157 = stablehlo.reshape %v156 : (tensor<128x512xf32>) -> tensor<128x1x512xf32>
    %v158 = stablehlo.dot_general %v157, %W9, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x512xf32>, tensor<128x512xf32>) -> tensor<128x1x128xf32>
    %v159 = stablehlo.reshape %v158 : (tensor<128x1x128xf32>) -> tensor<128x128xf32>
    %v160 = stablehlo.reshape %v107 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v161 = stablehlo.reshape %v159 : (tensor<128x128xf32>) -> tensor<128x32x2x2xf32>
    %v162 = stablehlo.constant dense<0.0> : tensor<f32>
    %v163 = "stablehlo.select_and_scatter"(%v160, %v161, %v162) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x4x4xf32>, tensor<128x32x2x2xf32>, tensor<f32>) -> tensor<128x32x4x4xf32>
    %v164 = stablehlo.reshape %v163 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v165 = stablehlo.reshape %v164 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v166 = stablehlo.reshape %v103 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v167 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v168 = stablehlo.compare GT, %v166, %v167 : (tensor<128x32x4x4xf32>, tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xi1>
    %v169 = stablehlo.select %v168, %v165, %v167 : tensor<128x32x4x4xi1>, tensor<128x32x4x4xf32>
    %v170 = stablehlo.reshape %v169 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v171 = stablehlo.reshape %v170 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v172 = stablehlo.reverse %W8, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v173 = stablehlo.transpose %v172, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v174 = stablehlo.convert %v171 : (tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xbf16>
    %v175 = stablehlo.convert %v173 : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xbf16>
    %v176 = stablehlo.convolution(%v174, %v175)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xbf16>, tensor<32x32x3x3xbf16>) -> tensor<128x32x4x4xbf16>
    %v177 = stablehlo.convert %v176 : (tensor<128x32x4x4xbf16>) -> tensor<128x32x4x4xf32>
    %v178 = stablehlo.reshape %v177 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v179 = stablehlo.reshape %v178 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v180 = stablehlo.reshape %v91 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v181 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v182 = stablehlo.compare GT, %v180, %v181 : (tensor<128x32x4x4xf32>, tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xi1>
    %v183 = stablehlo.select %v182, %v179, %v181 : tensor<128x32x4x4xi1>, tensor<128x32x4x4xf32>
    %v184 = stablehlo.reshape %v183 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v185 = stablehlo.reshape %v184 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v186 = stablehlo.reverse %W7, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v187 = stablehlo.transpose %v186, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v188 = stablehlo.convert %v185 : (tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xbf16>
    %v189 = stablehlo.convert %v187 : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xbf16>
    %v190 = stablehlo.convolution(%v188, %v189)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xbf16>, tensor<32x32x3x3xbf16>) -> tensor<128x32x4x4xbf16>
    %v191 = stablehlo.convert %v190 : (tensor<128x32x4x4xbf16>) -> tensor<128x32x4x4xf32>
    %v192 = stablehlo.reshape %v191 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v193 = stablehlo.reshape %v79 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v194 = stablehlo.reshape %v192 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v195 = stablehlo.constant dense<0.0> : tensor<f32>
    %v196 = "stablehlo.select_and_scatter"(%v193, %v194, %v195) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x8x8xf32>, tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32x8x8xf32>
    %v197 = stablehlo.reshape %v196 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v198 = stablehlo.reshape %v197 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v199 = stablehlo.reshape %v75 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v200 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v201 = stablehlo.compare GT, %v199, %v200 : (tensor<128x32x8x8xf32>, tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xi1>
    %v202 = stablehlo.select %v201, %v198, %v200 : tensor<128x32x8x8xi1>, tensor<128x32x8x8xf32>
    %v203 = stablehlo.reshape %v202 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v204 = stablehlo.reshape %v203 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v205 = stablehlo.reverse %W6, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v206 = stablehlo.transpose %v205, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v207 = stablehlo.convert %v204 : (tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xbf16>
    %v208 = stablehlo.convert %v206 : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xbf16>
    %v209 = stablehlo.convolution(%v207, %v208)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xbf16>, tensor<32x32x3x3xbf16>) -> tensor<128x32x8x8xbf16>
    %v210 = stablehlo.convert %v209 : (tensor<128x32x8x8xbf16>) -> tensor<128x32x8x8xf32>
    %v211 = stablehlo.reshape %v210 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v212 = stablehlo.reshape %v211 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v213 = stablehlo.reshape %v63 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v214 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v215 = stablehlo.compare GT, %v213, %v214 : (tensor<128x32x8x8xf32>, tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xi1>
    %v216 = stablehlo.select %v215, %v212, %v214 : tensor<128x32x8x8xi1>, tensor<128x32x8x8xf32>
    %v217 = stablehlo.reshape %v216 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v218 = stablehlo.reshape %v217 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v219 = stablehlo.reverse %W5, dims = [2, 3] : tensor<32x16x3x3xf32>
    %v220 = stablehlo.transpose %v219, dims = [1, 0, 2, 3] : (tensor<32x16x3x3xf32>) -> tensor<16x32x3x3xf32>
    %v221 = stablehlo.convert %v218 : (tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xbf16>
    %v222 = stablehlo.convert %v220 : (tensor<16x32x3x3xf32>) -> tensor<16x32x3x3xbf16>
    %v223 = stablehlo.convolution(%v221, %v222)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xbf16>, tensor<16x32x3x3xbf16>) -> tensor<128x16x8x8xbf16>
    %v224 = stablehlo.convert %v223 : (tensor<128x16x8x8xbf16>) -> tensor<128x16x8x8xf32>
    %v225 = stablehlo.reshape %v224 : (tensor<128x16x8x8xf32>) -> tensor<128x1024xf32>
    %v226 = stablehlo.reshape %v51 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v227 = stablehlo.reshape %v225 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v228 = stablehlo.constant dense<0.0> : tensor<f32>
    %v229 = "stablehlo.select_and_scatter"(%v226, %v227, %v228) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x16x16xf32>, tensor<128x16x8x8xf32>, tensor<f32>) -> tensor<128x16x16x16xf32>
    %v230 = stablehlo.reshape %v229 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v231 = stablehlo.reshape %v230 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v232 = stablehlo.reshape %v47 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v233 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v234 = stablehlo.compare GT, %v232, %v233 : (tensor<128x16x16x16xf32>, tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xi1>
    %v235 = stablehlo.select %v234, %v231, %v233 : tensor<128x16x16x16xi1>, tensor<128x16x16x16xf32>
    %v236 = stablehlo.reshape %v235 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v237 = stablehlo.reshape %v236 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v238 = stablehlo.reverse %W4, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v239 = stablehlo.transpose %v238, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v240 = stablehlo.convert %v237 : (tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xbf16>
    %v241 = stablehlo.convert %v239 : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xbf16>
    %v242 = stablehlo.convolution(%v240, %v241)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xbf16>, tensor<16x16x3x3xbf16>) -> tensor<128x16x16x16xbf16>
    %v243 = stablehlo.convert %v242 : (tensor<128x16x16x16xbf16>) -> tensor<128x16x16x16xf32>
    %v244 = stablehlo.reshape %v243 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v245 = stablehlo.reshape %v244 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v246 = stablehlo.reshape %v35 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v247 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v248 = stablehlo.compare GT, %v246, %v247 : (tensor<128x16x16x16xf32>, tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xi1>
    %v249 = stablehlo.select %v248, %v245, %v247 : tensor<128x16x16x16xi1>, tensor<128x16x16x16xf32>
    %v250 = stablehlo.reshape %v249 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v251 = stablehlo.reshape %v250 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v252 = stablehlo.reverse %W3, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v253 = stablehlo.transpose %v252, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v254 = stablehlo.convert %v251 : (tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xbf16>
    %v255 = stablehlo.convert %v253 : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xbf16>
    %v256 = stablehlo.convolution(%v254, %v255)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xbf16>, tensor<16x16x3x3xbf16>) -> tensor<128x16x16x16xbf16>
    %v257 = stablehlo.convert %v256 : (tensor<128x16x16x16xbf16>) -> tensor<128x16x16x16xf32>
    %v258 = stablehlo.reshape %v257 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v259 = stablehlo.reshape %v23 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v260 = stablehlo.reshape %v258 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v261 = stablehlo.constant dense<0.0> : tensor<f32>
    %v262 = "stablehlo.select_and_scatter"(%v259, %v260, %v261) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x32x32xf32>, tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x32x32xf32>
    %v263 = stablehlo.reshape %v262 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v264 = stablehlo.reshape %v263 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v265 = stablehlo.reshape %v19 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v266 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v267 = stablehlo.compare GT, %v265, %v266 : (tensor<128x16x32x32xf32>, tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xi1>
    %v268 = stablehlo.select %v267, %v264, %v266 : tensor<128x16x32x32xi1>, tensor<128x16x32x32xf32>
    %v269 = stablehlo.reshape %v268 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v270 = stablehlo.reshape %v269 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v271 = stablehlo.reverse %W2, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v272 = stablehlo.transpose %v271, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v273 = stablehlo.convert %v270 : (tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xbf16>
    %v274 = stablehlo.convert %v272 : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xbf16>
    %v275 = stablehlo.convolution(%v273, %v274)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xbf16>, tensor<16x16x3x3xbf16>) -> tensor<128x16x32x32xbf16>
    %v276 = stablehlo.convert %v275 : (tensor<128x16x32x32xbf16>) -> tensor<128x16x32x32xf32>
    %v277 = stablehlo.reshape %v276 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v278 = stablehlo.reshape %v277 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v279 = stablehlo.reshape %v7 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v280 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v281 = stablehlo.compare GT, %v279, %v280 : (tensor<128x16x32x32xf32>, tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xi1>
    %v282 = stablehlo.select %v281, %v278, %v280 : tensor<128x16x32x32xi1>, tensor<128x16x32x32xf32>
    %v283 = stablehlo.reshape %v282 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v284 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v285 = stablehlo.reshape %v283 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v286 = stablehlo.transpose %v284, dims = [1, 0, 2, 3] : (tensor<128x3x32x32xf32>) -> tensor<3x128x32x32xf32>
    %v287 = stablehlo.transpose %v285, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v288 = stablehlo.convert %v286 : (tensor<3x128x32x32xf32>) -> tensor<3x128x32x32xbf16>
    %v289 = stablehlo.convert %v287 : (tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xbf16>
    %v290 = stablehlo.convolution(%v288, %v289)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x128x32x32xbf16>, tensor<16x128x32x32xbf16>) -> tensor<3x16x3x3xbf16>
    %v291 = stablehlo.convert %v290 : (tensor<3x16x3x3xbf16>) -> tensor<3x16x3x3xf32>
    %v292 = stablehlo.transpose %v291, dims = [1, 0, 2, 3] : (tensor<3x16x3x3xf32>) -> tensor<16x3x3x3xf32>
    %v293 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v294 = stablehlo.multiply %v293, %v292 : tensor<16x3x3x3xf32>
    %v295 = stablehlo.subtract %W1, %v294 : tensor<16x3x3x3xf32>
    %v296 = stablehlo.reshape %v283 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v297 = stablehlo.constant dense<0.0> : tensor<f32>
    %v298 = stablehlo.reduce(%v296 init: %v297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v299 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v300 = stablehlo.multiply %v299, %v298 : tensor<16xf32>
    %v301 = stablehlo.subtract %cb1, %v300 : tensor<16xf32>
    %v302 = stablehlo.reshape %v11 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v303 = stablehlo.reshape %v269 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v304 = stablehlo.transpose %v302, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v305 = stablehlo.transpose %v303, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v306 = stablehlo.convert %v304 : (tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xbf16>
    %v307 = stablehlo.convert %v305 : (tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xbf16>
    %v308 = stablehlo.convolution(%v306, %v307)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xbf16>, tensor<16x128x32x32xbf16>) -> tensor<16x16x3x3xbf16>
    %v309 = stablehlo.convert %v308 : (tensor<16x16x3x3xbf16>) -> tensor<16x16x3x3xf32>
    %v310 = stablehlo.transpose %v309, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v311 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v312 = stablehlo.multiply %v311, %v310 : tensor<16x16x3x3xf32>
    %v313 = stablehlo.subtract %W2, %v312 : tensor<16x16x3x3xf32>
    %v314 = stablehlo.reshape %v269 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v315 = stablehlo.constant dense<0.0> : tensor<f32>
    %v316 = stablehlo.reduce(%v314 init: %v315) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v317 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v318 = stablehlo.multiply %v317, %v316 : tensor<16xf32>
    %v319 = stablehlo.subtract %cb2, %v318 : tensor<16xf32>
    %v320 = stablehlo.reshape %v27 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v321 = stablehlo.reshape %v250 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v322 = stablehlo.transpose %v320, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v323 = stablehlo.transpose %v321, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v324 = stablehlo.convert %v322 : (tensor<16x128x16x16xf32>) -> tensor<16x128x16x16xbf16>
    %v325 = stablehlo.convert %v323 : (tensor<16x128x16x16xf32>) -> tensor<16x128x16x16xbf16>
    %v326 = stablehlo.convolution(%v324, %v325)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xbf16>, tensor<16x128x16x16xbf16>) -> tensor<16x16x3x3xbf16>
    %v327 = stablehlo.convert %v326 : (tensor<16x16x3x3xbf16>) -> tensor<16x16x3x3xf32>
    %v328 = stablehlo.transpose %v327, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v329 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v330 = stablehlo.multiply %v329, %v328 : tensor<16x16x3x3xf32>
    %v331 = stablehlo.subtract %W3, %v330 : tensor<16x16x3x3xf32>
    %v332 = stablehlo.reshape %v250 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v333 = stablehlo.constant dense<0.0> : tensor<f32>
    %v334 = stablehlo.reduce(%v332 init: %v333) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v335 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v336 = stablehlo.multiply %v335, %v334 : tensor<16xf32>
    %v337 = stablehlo.subtract %cb3, %v336 : tensor<16xf32>
    %v338 = stablehlo.reshape %v39 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v339 = stablehlo.reshape %v236 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v340 = stablehlo.transpose %v338, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v341 = stablehlo.transpose %v339, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v342 = stablehlo.convert %v340 : (tensor<16x128x16x16xf32>) -> tensor<16x128x16x16xbf16>
    %v343 = stablehlo.convert %v341 : (tensor<16x128x16x16xf32>) -> tensor<16x128x16x16xbf16>
    %v344 = stablehlo.convolution(%v342, %v343)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xbf16>, tensor<16x128x16x16xbf16>) -> tensor<16x16x3x3xbf16>
    %v345 = stablehlo.convert %v344 : (tensor<16x16x3x3xbf16>) -> tensor<16x16x3x3xf32>
    %v346 = stablehlo.transpose %v345, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v347 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v348 = stablehlo.multiply %v347, %v346 : tensor<16x16x3x3xf32>
    %v349 = stablehlo.subtract %W4, %v348 : tensor<16x16x3x3xf32>
    %v350 = stablehlo.reshape %v236 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v351 = stablehlo.constant dense<0.0> : tensor<f32>
    %v352 = stablehlo.reduce(%v350 init: %v351) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v353 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v354 = stablehlo.multiply %v353, %v352 : tensor<16xf32>
    %v355 = stablehlo.subtract %cb4, %v354 : tensor<16xf32>
    %v356 = stablehlo.reshape %v55 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v357 = stablehlo.reshape %v217 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v358 = stablehlo.transpose %v356, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v359 = stablehlo.transpose %v357, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v360 = stablehlo.convert %v358 : (tensor<16x128x8x8xf32>) -> tensor<16x128x8x8xbf16>
    %v361 = stablehlo.convert %v359 : (tensor<32x128x8x8xf32>) -> tensor<32x128x8x8xbf16>
    %v362 = stablehlo.convolution(%v360, %v361)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xbf16>, tensor<32x128x8x8xbf16>) -> tensor<16x32x3x3xbf16>
    %v363 = stablehlo.convert %v362 : (tensor<16x32x3x3xbf16>) -> tensor<16x32x3x3xf32>
    %v364 = stablehlo.transpose %v363, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v365 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v366 = stablehlo.multiply %v365, %v364 : tensor<32x16x3x3xf32>
    %v367 = stablehlo.subtract %W5, %v366 : tensor<32x16x3x3xf32>
    %v368 = stablehlo.reshape %v217 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v369 = stablehlo.constant dense<0.0> : tensor<f32>
    %v370 = stablehlo.reduce(%v368 init: %v369) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v371 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v372 = stablehlo.multiply %v371, %v370 : tensor<32xf32>
    %v373 = stablehlo.subtract %cb5, %v372 : tensor<32xf32>
    %v374 = stablehlo.reshape %v67 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v375 = stablehlo.reshape %v203 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v376 = stablehlo.transpose %v374, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v377 = stablehlo.transpose %v375, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v378 = stablehlo.convert %v376 : (tensor<32x128x8x8xf32>) -> tensor<32x128x8x8xbf16>
    %v379 = stablehlo.convert %v377 : (tensor<32x128x8x8xf32>) -> tensor<32x128x8x8xbf16>
    %v380 = stablehlo.convolution(%v378, %v379)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xbf16>, tensor<32x128x8x8xbf16>) -> tensor<32x32x3x3xbf16>
    %v381 = stablehlo.convert %v380 : (tensor<32x32x3x3xbf16>) -> tensor<32x32x3x3xf32>
    %v382 = stablehlo.transpose %v381, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v383 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v384 = stablehlo.multiply %v383, %v382 : tensor<32x32x3x3xf32>
    %v385 = stablehlo.subtract %W6, %v384 : tensor<32x32x3x3xf32>
    %v386 = stablehlo.reshape %v203 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v387 = stablehlo.constant dense<0.0> : tensor<f32>
    %v388 = stablehlo.reduce(%v386 init: %v387) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v389 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v390 = stablehlo.multiply %v389, %v388 : tensor<32xf32>
    %v391 = stablehlo.subtract %cb6, %v390 : tensor<32xf32>
    %v392 = stablehlo.reshape %v83 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v393 = stablehlo.reshape %v184 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v394 = stablehlo.transpose %v392, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v395 = stablehlo.transpose %v393, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v396 = stablehlo.convert %v394 : (tensor<32x128x4x4xf32>) -> tensor<32x128x4x4xbf16>
    %v397 = stablehlo.convert %v395 : (tensor<32x128x4x4xf32>) -> tensor<32x128x4x4xbf16>
    %v398 = stablehlo.convolution(%v396, %v397)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xbf16>, tensor<32x128x4x4xbf16>) -> tensor<32x32x3x3xbf16>
    %v399 = stablehlo.convert %v398 : (tensor<32x32x3x3xbf16>) -> tensor<32x32x3x3xf32>
    %v400 = stablehlo.transpose %v399, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v401 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v402 = stablehlo.multiply %v401, %v400 : tensor<32x32x3x3xf32>
    %v403 = stablehlo.subtract %W7, %v402 : tensor<32x32x3x3xf32>
    %v404 = stablehlo.reshape %v184 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v405 = stablehlo.constant dense<0.0> : tensor<f32>
    %v406 = stablehlo.reduce(%v404 init: %v405) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v407 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v408 = stablehlo.multiply %v407, %v406 : tensor<32xf32>
    %v409 = stablehlo.subtract %cb7, %v408 : tensor<32xf32>
    %v410 = stablehlo.reshape %v95 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v411 = stablehlo.reshape %v170 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v412 = stablehlo.transpose %v410, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v413 = stablehlo.transpose %v411, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v414 = stablehlo.convert %v412 : (tensor<32x128x4x4xf32>) -> tensor<32x128x4x4xbf16>
    %v415 = stablehlo.convert %v413 : (tensor<32x128x4x4xf32>) -> tensor<32x128x4x4xbf16>
    %v416 = stablehlo.convolution(%v414, %v415)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xbf16>, tensor<32x128x4x4xbf16>) -> tensor<32x32x3x3xbf16>
    %v417 = stablehlo.convert %v416 : (tensor<32x32x3x3xbf16>) -> tensor<32x32x3x3xf32>
    %v418 = stablehlo.transpose %v417, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v419 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v420 = stablehlo.multiply %v419, %v418 : tensor<32x32x3x3xf32>
    %v421 = stablehlo.subtract %W8, %v420 : tensor<32x32x3x3xf32>
    %v422 = stablehlo.reshape %v170 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v423 = stablehlo.constant dense<0.0> : tensor<f32>
    %v424 = stablehlo.reduce(%v422 init: %v423) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v425 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v426 = stablehlo.multiply %v425, %v424 : tensor<32xf32>
    %v427 = stablehlo.subtract %cb8, %v426 : tensor<32xf32>
    %v428 = stablehlo.dot_general %v111, %v156, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x512xf32>) -> tensor<128x512xf32>
    %v429 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v430 = stablehlo.multiply %v429, %v428 : tensor<128x512xf32>
    %v431 = stablehlo.subtract %W9, %v430 : tensor<128x512xf32>
    %v432 = stablehlo.constant dense<0.0> : tensor<f32>
    %v433 = stablehlo.reduce(%v156 init: %v432) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v434 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v435 = stablehlo.multiply %v434, %v433 : tensor<512xf32>
    %v436 = stablehlo.subtract %b9, %v435 : tensor<512xf32>
    %v437 = stablehlo.dot_general %v118, %v147, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<512x512xf32>
    %v438 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v439 = stablehlo.multiply %v438, %v437 : tensor<512x512xf32>
    %v440 = stablehlo.subtract %Wa, %v439 : tensor<512x512xf32>
    %v441 = stablehlo.constant dense<0.0> : tensor<f32>
    %v442 = stablehlo.reduce(%v147 init: %v441) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v443 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v444 = stablehlo.multiply %v443, %v442 : tensor<512xf32>
    %v445 = stablehlo.subtract %ba, %v444 : tensor<512xf32>
    %v446 = stablehlo.dot_general %v125, %v138, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x10xf32>) -> tensor<512x10xf32>
    %v447 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v448 = stablehlo.multiply %v447, %v446 : tensor<512x10xf32>
    %v449 = stablehlo.subtract %Wb, %v448 : tensor<512x10xf32>
    %v450 = stablehlo.constant dense<0.0> : tensor<f32>
    %v451 = stablehlo.reduce(%v138 init: %v450) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v452 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v453 = stablehlo.multiply %v452, %v451 : tensor<10xf32>
    %v454 = stablehlo.subtract %bb, %v453 : tensor<10xf32>
    return %v295, %v301, %v313, %v319, %v331, %v337, %v349, %v355, %v367, %v373, %v385, %v391, %v403, %v409, %v421, %v427, %v431, %v436, %v440, %v445, %v449, %v454, %W1m, %cb1m, %W2m, %cb2m, %W3m, %cb3m, %W4m, %cb4m, %W5m, %cb5m, %W6m, %cb6m, %W7m, %cb7m, %W8m, %cb8m, %W9m, %b9m, %Wam, %bam, %Wbm, %bbm, %W1v, %cb1v, %W2v, %cb2v, %W3v, %cb3v, %W4v, %cb4v, %W5v, %cb5v, %W6v, %cb6v, %W7v, %cb7v, %W8v, %cb8v, %W9v, %b9v, %Wav, %bav, %Wbv, %bbv, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
