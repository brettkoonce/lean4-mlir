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
    %v115 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v116 = stablehlo.maximum %v114, %v115 : tensor<128x512xf32>
    %v117 = stablehlo.dot_general %v116, %Wa, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %v118 = stablehlo.broadcast_in_dim %ba, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v119 = stablehlo.add %v117, %v118 : tensor<128x512xf32>
    %v120 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v121 = stablehlo.maximum %v119, %v120 : tensor<128x512xf32>
    %v122 = stablehlo.dot_general %v121, %Wb, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x10xf32>) -> tensor<128x10xf32>
    %v123 = stablehlo.broadcast_in_dim %bb, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v124 = stablehlo.add %v122, %v123 : tensor<128x10xf32>
    %v125 = stablehlo.reshape %v124 : (tensor<128x10xf32>) -> tensor<128x1x10xf32>
    %v126 = stablehlo.constant dense<0.0> : tensor<f32>
    %v127 = stablehlo.exponential %v125 : tensor<128x1x10xf32>
    %v128 = stablehlo.reduce(%v127 init: %v126) applies stablehlo.add across dimensions = [2] : (tensor<128x1x10xf32>, tensor<f32>) -> tensor<128x1xf32>
    %v129 = stablehlo.broadcast_in_dim %v128, dims = [0, 1] : (tensor<128x1xf32>) -> tensor<128x1x10xf32>
    %v130 = stablehlo.divide %v127, %v129 : tensor<128x1x10xf32>
    %v131 = stablehlo.reshape %v130 : (tensor<128x1x10xf32>) -> tensor<128x10xf32>
    %v132 = stablehlo.subtract %v131, %onehot : tensor<128x10xf32>
    %v133 = stablehlo.constant dense<0.0078125> : tensor<128x10xf32>
    %v134 = stablehlo.multiply %v132, %v133 : tensor<128x10xf32>
    // ── report-only scalar loss (NOT pretty(AST): the kit has no rank-0 loss op; it
    //    feeds no parameter, only the driver's progress line) ──
    %llog = stablehlo.log %v131 : tensor<128x10xf32>
    %ohll = stablehlo.multiply %onehot, %llog : tensor<128x10xf32>
    %csum = stablehlo.reduce(%ohll init: %lzero) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<f32>
    %cneg = stablehlo.negate %csum : tensor<f32>
    %lbf = stablehlo.constant dense<128.0> : tensor<f32>
    %loss = stablehlo.divide %cneg, %lbf : tensor<f32>
    %v135 = stablehlo.reshape %v134 : (tensor<128x10xf32>) -> tensor<128x1x10xf32>
    %v136 = stablehlo.dot_general %v135, %Wb, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x10xf32>, tensor<512x10xf32>) -> tensor<128x1x512xf32>
    %v137 = stablehlo.reshape %v136 : (tensor<128x1x512xf32>) -> tensor<128x512xf32>
    %v138 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v139 = stablehlo.compare GT, %v119, %v138 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v140 = stablehlo.select %v139, %v137, %v138 : tensor<128x512xi1>, tensor<128x512xf32>
    %v141 = stablehlo.reshape %v140 : (tensor<128x512xf32>) -> tensor<128x1x512xf32>
    %v142 = stablehlo.dot_general %v141, %Wa, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x512xf32>, tensor<512x512xf32>) -> tensor<128x1x512xf32>
    %v143 = stablehlo.reshape %v142 : (tensor<128x1x512xf32>) -> tensor<128x512xf32>
    %v144 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v145 = stablehlo.compare GT, %v114, %v144 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v146 = stablehlo.select %v145, %v143, %v144 : tensor<128x512xi1>, tensor<128x512xf32>
    %v147 = stablehlo.reshape %v146 : (tensor<128x512xf32>) -> tensor<128x1x512xf32>
    %v148 = stablehlo.dot_general %v147, %W9, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x512xf32>, tensor<128x512xf32>) -> tensor<128x1x128xf32>
    %v149 = stablehlo.reshape %v148 : (tensor<128x1x128xf32>) -> tensor<128x128xf32>
    %v150 = stablehlo.reshape %v107 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v151 = stablehlo.reshape %v149 : (tensor<128x128xf32>) -> tensor<128x32x2x2xf32>
    %v152 = stablehlo.constant dense<0.0> : tensor<f32>
    %v153 = "stablehlo.select_and_scatter"(%v150, %v151, %v152) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x4x4xf32>, tensor<128x32x2x2xf32>, tensor<f32>) -> tensor<128x32x4x4xf32>
    %v154 = stablehlo.reshape %v153 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v155 = stablehlo.reshape %v154 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v156 = stablehlo.reshape %v103 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v157 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v158 = stablehlo.compare GT, %v156, %v157 : (tensor<128x32x4x4xf32>, tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xi1>
    %v159 = stablehlo.select %v158, %v155, %v157 : tensor<128x32x4x4xi1>, tensor<128x32x4x4xf32>
    %v160 = stablehlo.reshape %v159 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v161 = stablehlo.reshape %v160 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v162 = stablehlo.reverse %W8, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v163 = stablehlo.transpose %v162, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v164 = stablehlo.convert %v161 : (tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xbf16>
    %v165 = stablehlo.convert %v163 : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xbf16>
    %v166 = stablehlo.convolution(%v164, %v165)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xbf16>, tensor<32x32x3x3xbf16>) -> tensor<128x32x4x4xbf16>
    %v167 = stablehlo.convert %v166 : (tensor<128x32x4x4xbf16>) -> tensor<128x32x4x4xf32>
    %v168 = stablehlo.reshape %v167 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v169 = stablehlo.reshape %v168 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v170 = stablehlo.reshape %v91 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v171 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v172 = stablehlo.compare GT, %v170, %v171 : (tensor<128x32x4x4xf32>, tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xi1>
    %v173 = stablehlo.select %v172, %v169, %v171 : tensor<128x32x4x4xi1>, tensor<128x32x4x4xf32>
    %v174 = stablehlo.reshape %v173 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v175 = stablehlo.reshape %v174 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v176 = stablehlo.reverse %W7, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v177 = stablehlo.transpose %v176, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v178 = stablehlo.convert %v175 : (tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xbf16>
    %v179 = stablehlo.convert %v177 : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xbf16>
    %v180 = stablehlo.convolution(%v178, %v179)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xbf16>, tensor<32x32x3x3xbf16>) -> tensor<128x32x4x4xbf16>
    %v181 = stablehlo.convert %v180 : (tensor<128x32x4x4xbf16>) -> tensor<128x32x4x4xf32>
    %v182 = stablehlo.reshape %v181 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v183 = stablehlo.reshape %v79 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v184 = stablehlo.reshape %v182 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v185 = stablehlo.constant dense<0.0> : tensor<f32>
    %v186 = "stablehlo.select_and_scatter"(%v183, %v184, %v185) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x8x8xf32>, tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32x8x8xf32>
    %v187 = stablehlo.reshape %v186 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v188 = stablehlo.reshape %v187 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v189 = stablehlo.reshape %v75 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v190 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v191 = stablehlo.compare GT, %v189, %v190 : (tensor<128x32x8x8xf32>, tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xi1>
    %v192 = stablehlo.select %v191, %v188, %v190 : tensor<128x32x8x8xi1>, tensor<128x32x8x8xf32>
    %v193 = stablehlo.reshape %v192 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v194 = stablehlo.reshape %v193 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v195 = stablehlo.reverse %W6, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v196 = stablehlo.transpose %v195, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v197 = stablehlo.convert %v194 : (tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xbf16>
    %v198 = stablehlo.convert %v196 : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xbf16>
    %v199 = stablehlo.convolution(%v197, %v198)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xbf16>, tensor<32x32x3x3xbf16>) -> tensor<128x32x8x8xbf16>
    %v200 = stablehlo.convert %v199 : (tensor<128x32x8x8xbf16>) -> tensor<128x32x8x8xf32>
    %v201 = stablehlo.reshape %v200 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v202 = stablehlo.reshape %v201 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v203 = stablehlo.reshape %v63 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v204 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v205 = stablehlo.compare GT, %v203, %v204 : (tensor<128x32x8x8xf32>, tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xi1>
    %v206 = stablehlo.select %v205, %v202, %v204 : tensor<128x32x8x8xi1>, tensor<128x32x8x8xf32>
    %v207 = stablehlo.reshape %v206 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v208 = stablehlo.reshape %v207 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v209 = stablehlo.reverse %W5, dims = [2, 3] : tensor<32x16x3x3xf32>
    %v210 = stablehlo.transpose %v209, dims = [1, 0, 2, 3] : (tensor<32x16x3x3xf32>) -> tensor<16x32x3x3xf32>
    %v211 = stablehlo.convert %v208 : (tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xbf16>
    %v212 = stablehlo.convert %v210 : (tensor<16x32x3x3xf32>) -> tensor<16x32x3x3xbf16>
    %v213 = stablehlo.convolution(%v211, %v212)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xbf16>, tensor<16x32x3x3xbf16>) -> tensor<128x16x8x8xbf16>
    %v214 = stablehlo.convert %v213 : (tensor<128x16x8x8xbf16>) -> tensor<128x16x8x8xf32>
    %v215 = stablehlo.reshape %v214 : (tensor<128x16x8x8xf32>) -> tensor<128x1024xf32>
    %v216 = stablehlo.reshape %v51 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v217 = stablehlo.reshape %v215 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v218 = stablehlo.constant dense<0.0> : tensor<f32>
    %v219 = "stablehlo.select_and_scatter"(%v216, %v217, %v218) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x16x16xf32>, tensor<128x16x8x8xf32>, tensor<f32>) -> tensor<128x16x16x16xf32>
    %v220 = stablehlo.reshape %v219 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v221 = stablehlo.reshape %v220 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v222 = stablehlo.reshape %v47 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v223 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v224 = stablehlo.compare GT, %v222, %v223 : (tensor<128x16x16x16xf32>, tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xi1>
    %v225 = stablehlo.select %v224, %v221, %v223 : tensor<128x16x16x16xi1>, tensor<128x16x16x16xf32>
    %v226 = stablehlo.reshape %v225 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v227 = stablehlo.reshape %v226 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v228 = stablehlo.reverse %W4, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v229 = stablehlo.transpose %v228, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v230 = stablehlo.convert %v227 : (tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xbf16>
    %v231 = stablehlo.convert %v229 : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xbf16>
    %v232 = stablehlo.convolution(%v230, %v231)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xbf16>, tensor<16x16x3x3xbf16>) -> tensor<128x16x16x16xbf16>
    %v233 = stablehlo.convert %v232 : (tensor<128x16x16x16xbf16>) -> tensor<128x16x16x16xf32>
    %v234 = stablehlo.reshape %v233 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v235 = stablehlo.reshape %v234 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v236 = stablehlo.reshape %v35 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v237 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v238 = stablehlo.compare GT, %v236, %v237 : (tensor<128x16x16x16xf32>, tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xi1>
    %v239 = stablehlo.select %v238, %v235, %v237 : tensor<128x16x16x16xi1>, tensor<128x16x16x16xf32>
    %v240 = stablehlo.reshape %v239 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v241 = stablehlo.reshape %v240 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v242 = stablehlo.reverse %W3, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v243 = stablehlo.transpose %v242, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v244 = stablehlo.convert %v241 : (tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xbf16>
    %v245 = stablehlo.convert %v243 : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xbf16>
    %v246 = stablehlo.convolution(%v244, %v245)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xbf16>, tensor<16x16x3x3xbf16>) -> tensor<128x16x16x16xbf16>
    %v247 = stablehlo.convert %v246 : (tensor<128x16x16x16xbf16>) -> tensor<128x16x16x16xf32>
    %v248 = stablehlo.reshape %v247 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v249 = stablehlo.reshape %v23 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v250 = stablehlo.reshape %v248 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v251 = stablehlo.constant dense<0.0> : tensor<f32>
    %v252 = "stablehlo.select_and_scatter"(%v249, %v250, %v251) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x32x32xf32>, tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x32x32xf32>
    %v253 = stablehlo.reshape %v252 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v254 = stablehlo.reshape %v253 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v255 = stablehlo.reshape %v19 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v256 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v257 = stablehlo.compare GT, %v255, %v256 : (tensor<128x16x32x32xf32>, tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xi1>
    %v258 = stablehlo.select %v257, %v254, %v256 : tensor<128x16x32x32xi1>, tensor<128x16x32x32xf32>
    %v259 = stablehlo.reshape %v258 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v260 = stablehlo.reshape %v259 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v261 = stablehlo.reverse %W2, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v262 = stablehlo.transpose %v261, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v263 = stablehlo.convert %v260 : (tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xbf16>
    %v264 = stablehlo.convert %v262 : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xbf16>
    %v265 = stablehlo.convolution(%v263, %v264)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xbf16>, tensor<16x16x3x3xbf16>) -> tensor<128x16x32x32xbf16>
    %v266 = stablehlo.convert %v265 : (tensor<128x16x32x32xbf16>) -> tensor<128x16x32x32xf32>
    %v267 = stablehlo.reshape %v266 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v268 = stablehlo.reshape %v267 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v269 = stablehlo.reshape %v7 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v270 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v271 = stablehlo.compare GT, %v269, %v270 : (tensor<128x16x32x32xf32>, tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xi1>
    %v272 = stablehlo.select %v271, %v268, %v270 : tensor<128x16x32x32xi1>, tensor<128x16x32x32xf32>
    %v273 = stablehlo.reshape %v272 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v274 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v275 = stablehlo.reshape %v273 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v276 = stablehlo.transpose %v274, dims = [1, 0, 2, 3] : (tensor<128x3x32x32xf32>) -> tensor<3x128x32x32xf32>
    %v277 = stablehlo.transpose %v275, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v278 = stablehlo.convert %v276 : (tensor<3x128x32x32xf32>) -> tensor<3x128x32x32xbf16>
    %v279 = stablehlo.convert %v277 : (tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xbf16>
    %v280 = stablehlo.convolution(%v278, %v279)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x128x32x32xbf16>, tensor<16x128x32x32xbf16>) -> tensor<3x16x3x3xbf16>
    %v281 = stablehlo.convert %v280 : (tensor<3x16x3x3xbf16>) -> tensor<3x16x3x3xf32>
    %v282 = stablehlo.transpose %v281, dims = [1, 0, 2, 3] : (tensor<3x16x3x3xf32>) -> tensor<16x3x3x3xf32>
    %v283 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v284 = stablehlo.multiply %v283, %v282 : tensor<16x3x3x3xf32>
    %v285 = stablehlo.subtract %W1, %v284 : tensor<16x3x3x3xf32>
    %v286 = stablehlo.reshape %v273 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v287 = stablehlo.constant dense<0.0> : tensor<f32>
    %v288 = stablehlo.reduce(%v286 init: %v287) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v289 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v290 = stablehlo.multiply %v289, %v288 : tensor<16xf32>
    %v291 = stablehlo.subtract %cb1, %v290 : tensor<16xf32>
    %v292 = stablehlo.reshape %v11 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v293 = stablehlo.reshape %v259 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v294 = stablehlo.transpose %v292, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v295 = stablehlo.transpose %v293, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v296 = stablehlo.convert %v294 : (tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xbf16>
    %v297 = stablehlo.convert %v295 : (tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xbf16>
    %v298 = stablehlo.convolution(%v296, %v297)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xbf16>, tensor<16x128x32x32xbf16>) -> tensor<16x16x3x3xbf16>
    %v299 = stablehlo.convert %v298 : (tensor<16x16x3x3xbf16>) -> tensor<16x16x3x3xf32>
    %v300 = stablehlo.transpose %v299, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v301 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v302 = stablehlo.multiply %v301, %v300 : tensor<16x16x3x3xf32>
    %v303 = stablehlo.subtract %W2, %v302 : tensor<16x16x3x3xf32>
    %v304 = stablehlo.reshape %v259 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v305 = stablehlo.constant dense<0.0> : tensor<f32>
    %v306 = stablehlo.reduce(%v304 init: %v305) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v307 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v308 = stablehlo.multiply %v307, %v306 : tensor<16xf32>
    %v309 = stablehlo.subtract %cb2, %v308 : tensor<16xf32>
    %v310 = stablehlo.reshape %v27 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v311 = stablehlo.reshape %v240 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v312 = stablehlo.transpose %v310, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v313 = stablehlo.transpose %v311, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v314 = stablehlo.convert %v312 : (tensor<16x128x16x16xf32>) -> tensor<16x128x16x16xbf16>
    %v315 = stablehlo.convert %v313 : (tensor<16x128x16x16xf32>) -> tensor<16x128x16x16xbf16>
    %v316 = stablehlo.convolution(%v314, %v315)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xbf16>, tensor<16x128x16x16xbf16>) -> tensor<16x16x3x3xbf16>
    %v317 = stablehlo.convert %v316 : (tensor<16x16x3x3xbf16>) -> tensor<16x16x3x3xf32>
    %v318 = stablehlo.transpose %v317, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v319 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v320 = stablehlo.multiply %v319, %v318 : tensor<16x16x3x3xf32>
    %v321 = stablehlo.subtract %W3, %v320 : tensor<16x16x3x3xf32>
    %v322 = stablehlo.reshape %v240 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v323 = stablehlo.constant dense<0.0> : tensor<f32>
    %v324 = stablehlo.reduce(%v322 init: %v323) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v325 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v326 = stablehlo.multiply %v325, %v324 : tensor<16xf32>
    %v327 = stablehlo.subtract %cb3, %v326 : tensor<16xf32>
    %v328 = stablehlo.reshape %v39 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v329 = stablehlo.reshape %v226 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v330 = stablehlo.transpose %v328, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v331 = stablehlo.transpose %v329, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v332 = stablehlo.convert %v330 : (tensor<16x128x16x16xf32>) -> tensor<16x128x16x16xbf16>
    %v333 = stablehlo.convert %v331 : (tensor<16x128x16x16xf32>) -> tensor<16x128x16x16xbf16>
    %v334 = stablehlo.convolution(%v332, %v333)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xbf16>, tensor<16x128x16x16xbf16>) -> tensor<16x16x3x3xbf16>
    %v335 = stablehlo.convert %v334 : (tensor<16x16x3x3xbf16>) -> tensor<16x16x3x3xf32>
    %v336 = stablehlo.transpose %v335, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v337 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v338 = stablehlo.multiply %v337, %v336 : tensor<16x16x3x3xf32>
    %v339 = stablehlo.subtract %W4, %v338 : tensor<16x16x3x3xf32>
    %v340 = stablehlo.reshape %v226 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v341 = stablehlo.constant dense<0.0> : tensor<f32>
    %v342 = stablehlo.reduce(%v340 init: %v341) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v343 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v344 = stablehlo.multiply %v343, %v342 : tensor<16xf32>
    %v345 = stablehlo.subtract %cb4, %v344 : tensor<16xf32>
    %v346 = stablehlo.reshape %v55 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v347 = stablehlo.reshape %v207 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v348 = stablehlo.transpose %v346, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v349 = stablehlo.transpose %v347, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v350 = stablehlo.convert %v348 : (tensor<16x128x8x8xf32>) -> tensor<16x128x8x8xbf16>
    %v351 = stablehlo.convert %v349 : (tensor<32x128x8x8xf32>) -> tensor<32x128x8x8xbf16>
    %v352 = stablehlo.convolution(%v350, %v351)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xbf16>, tensor<32x128x8x8xbf16>) -> tensor<16x32x3x3xbf16>
    %v353 = stablehlo.convert %v352 : (tensor<16x32x3x3xbf16>) -> tensor<16x32x3x3xf32>
    %v354 = stablehlo.transpose %v353, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v355 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v356 = stablehlo.multiply %v355, %v354 : tensor<32x16x3x3xf32>
    %v357 = stablehlo.subtract %W5, %v356 : tensor<32x16x3x3xf32>
    %v358 = stablehlo.reshape %v207 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v359 = stablehlo.constant dense<0.0> : tensor<f32>
    %v360 = stablehlo.reduce(%v358 init: %v359) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v361 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v362 = stablehlo.multiply %v361, %v360 : tensor<32xf32>
    %v363 = stablehlo.subtract %cb5, %v362 : tensor<32xf32>
    %v364 = stablehlo.reshape %v67 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v365 = stablehlo.reshape %v193 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v366 = stablehlo.transpose %v364, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v367 = stablehlo.transpose %v365, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v368 = stablehlo.convert %v366 : (tensor<32x128x8x8xf32>) -> tensor<32x128x8x8xbf16>
    %v369 = stablehlo.convert %v367 : (tensor<32x128x8x8xf32>) -> tensor<32x128x8x8xbf16>
    %v370 = stablehlo.convolution(%v368, %v369)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xbf16>, tensor<32x128x8x8xbf16>) -> tensor<32x32x3x3xbf16>
    %v371 = stablehlo.convert %v370 : (tensor<32x32x3x3xbf16>) -> tensor<32x32x3x3xf32>
    %v372 = stablehlo.transpose %v371, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v373 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v374 = stablehlo.multiply %v373, %v372 : tensor<32x32x3x3xf32>
    %v375 = stablehlo.subtract %W6, %v374 : tensor<32x32x3x3xf32>
    %v376 = stablehlo.reshape %v193 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v377 = stablehlo.constant dense<0.0> : tensor<f32>
    %v378 = stablehlo.reduce(%v376 init: %v377) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v379 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v380 = stablehlo.multiply %v379, %v378 : tensor<32xf32>
    %v381 = stablehlo.subtract %cb6, %v380 : tensor<32xf32>
    %v382 = stablehlo.reshape %v83 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v383 = stablehlo.reshape %v174 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v384 = stablehlo.transpose %v382, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v385 = stablehlo.transpose %v383, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v386 = stablehlo.convert %v384 : (tensor<32x128x4x4xf32>) -> tensor<32x128x4x4xbf16>
    %v387 = stablehlo.convert %v385 : (tensor<32x128x4x4xf32>) -> tensor<32x128x4x4xbf16>
    %v388 = stablehlo.convolution(%v386, %v387)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xbf16>, tensor<32x128x4x4xbf16>) -> tensor<32x32x3x3xbf16>
    %v389 = stablehlo.convert %v388 : (tensor<32x32x3x3xbf16>) -> tensor<32x32x3x3xf32>
    %v390 = stablehlo.transpose %v389, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v391 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v392 = stablehlo.multiply %v391, %v390 : tensor<32x32x3x3xf32>
    %v393 = stablehlo.subtract %W7, %v392 : tensor<32x32x3x3xf32>
    %v394 = stablehlo.reshape %v174 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v395 = stablehlo.constant dense<0.0> : tensor<f32>
    %v396 = stablehlo.reduce(%v394 init: %v395) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v397 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v398 = stablehlo.multiply %v397, %v396 : tensor<32xf32>
    %v399 = stablehlo.subtract %cb7, %v398 : tensor<32xf32>
    %v400 = stablehlo.reshape %v95 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v401 = stablehlo.reshape %v160 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v402 = stablehlo.transpose %v400, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v403 = stablehlo.transpose %v401, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v404 = stablehlo.convert %v402 : (tensor<32x128x4x4xf32>) -> tensor<32x128x4x4xbf16>
    %v405 = stablehlo.convert %v403 : (tensor<32x128x4x4xf32>) -> tensor<32x128x4x4xbf16>
    %v406 = stablehlo.convolution(%v404, %v405)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xbf16>, tensor<32x128x4x4xbf16>) -> tensor<32x32x3x3xbf16>
    %v407 = stablehlo.convert %v406 : (tensor<32x32x3x3xbf16>) -> tensor<32x32x3x3xf32>
    %v408 = stablehlo.transpose %v407, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v409 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v410 = stablehlo.multiply %v409, %v408 : tensor<32x32x3x3xf32>
    %v411 = stablehlo.subtract %W8, %v410 : tensor<32x32x3x3xf32>
    %v412 = stablehlo.reshape %v160 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v413 = stablehlo.constant dense<0.0> : tensor<f32>
    %v414 = stablehlo.reduce(%v412 init: %v413) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v415 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v416 = stablehlo.multiply %v415, %v414 : tensor<32xf32>
    %v417 = stablehlo.subtract %cb8, %v416 : tensor<32xf32>
    %v418 = stablehlo.dot_general %v111, %v146, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x512xf32>) -> tensor<128x512xf32>
    %v419 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v420 = stablehlo.multiply %v419, %v418 : tensor<128x512xf32>
    %v421 = stablehlo.subtract %W9, %v420 : tensor<128x512xf32>
    %v422 = stablehlo.constant dense<0.0> : tensor<f32>
    %v423 = stablehlo.reduce(%v146 init: %v422) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v424 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v425 = stablehlo.multiply %v424, %v423 : tensor<512xf32>
    %v426 = stablehlo.subtract %b9, %v425 : tensor<512xf32>
    %v427 = stablehlo.dot_general %v116, %v140, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<512x512xf32>
    %v428 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v429 = stablehlo.multiply %v428, %v427 : tensor<512x512xf32>
    %v430 = stablehlo.subtract %Wa, %v429 : tensor<512x512xf32>
    %v431 = stablehlo.constant dense<0.0> : tensor<f32>
    %v432 = stablehlo.reduce(%v140 init: %v431) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v433 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v434 = stablehlo.multiply %v433, %v432 : tensor<512xf32>
    %v435 = stablehlo.subtract %ba, %v434 : tensor<512xf32>
    %v436 = stablehlo.dot_general %v121, %v134, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x10xf32>) -> tensor<512x10xf32>
    %v437 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v438 = stablehlo.multiply %v437, %v436 : tensor<512x10xf32>
    %v439 = stablehlo.subtract %Wb, %v438 : tensor<512x10xf32>
    %v440 = stablehlo.constant dense<0.0> : tensor<f32>
    %v441 = stablehlo.reduce(%v134 init: %v440) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v442 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v443 = stablehlo.multiply %v442, %v441 : tensor<10xf32>
    %v444 = stablehlo.subtract %bb, %v443 : tensor<10xf32>
    return %v285, %v291, %v303, %v309, %v321, %v327, %v339, %v345, %v357, %v363, %v375, %v381, %v393, %v399, %v411, %v417, %v421, %v426, %v430, %v435, %v439, %v444, %W1m, %cb1m, %W2m, %cb2m, %W3m, %cb3m, %W4m, %cb4m, %W5m, %cb5m, %W6m, %cb6m, %W7m, %cb7m, %W8m, %cb8m, %W9m, %b9m, %Wam, %bam, %Wbm, %bbm, %W1v, %cb1v, %W2v, %cb2v, %W3v, %cb3v, %W4v, %cb4v, %W5v, %cb5v, %W6v, %cb6v, %W7v, %cb7v, %W8v, %cb8v, %W9v, %b9v, %Wav, %bav, %Wbv, %bbv, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
