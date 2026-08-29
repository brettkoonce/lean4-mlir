module @m {
  func.func @cifar8wb_bf16adam_train_step(%x: tensor<128x3072xf32>, %W1: tensor<16x3x3x3xf32>, %cb1: tensor<16xf32>, %W2: tensor<16x16x3x3xf32>, %cb2: tensor<16xf32>, %W3: tensor<16x16x3x3xf32>, %cb3: tensor<16xf32>, %W4: tensor<16x16x3x3xf32>, %cb4: tensor<16xf32>, %W5: tensor<32x16x3x3xf32>, %cb5: tensor<32xf32>, %W6: tensor<32x32x3x3xf32>, %cb6: tensor<32xf32>, %W7: tensor<32x32x3x3xf32>, %cb7: tensor<32xf32>, %W8: tensor<32x32x3x3xf32>, %cb8: tensor<32xf32>, %W9: tensor<128x512xf32>, %b9: tensor<512xf32>, %Wa: tensor<512x512xf32>, %ba: tensor<512xf32>, %Wb: tensor<512x10xf32>, %bb: tensor<10xf32>, %W1m: tensor<16x3x3x3xf32>, %cb1m: tensor<16xf32>, %W2m: tensor<16x16x3x3xf32>, %cb2m: tensor<16xf32>, %W3m: tensor<16x16x3x3xf32>, %cb3m: tensor<16xf32>, %W4m: tensor<16x16x3x3xf32>, %cb4m: tensor<16xf32>, %W5m: tensor<32x16x3x3xf32>, %cb5m: tensor<32xf32>, %W6m: tensor<32x32x3x3xf32>, %cb6m: tensor<32xf32>, %W7m: tensor<32x32x3x3xf32>, %cb7m: tensor<32xf32>, %W8m: tensor<32x32x3x3xf32>, %cb8m: tensor<32xf32>, %W9m: tensor<128x512xf32>, %b9m: tensor<512xf32>, %Wam: tensor<512x512xf32>, %bam: tensor<512xf32>, %Wbm: tensor<512x10xf32>, %bbm: tensor<10xf32>, %W1v: tensor<16x3x3x3xf32>, %cb1v: tensor<16xf32>, %W2v: tensor<16x16x3x3xf32>, %cb2v: tensor<16xf32>, %W3v: tensor<16x16x3x3xf32>, %cb3v: tensor<16xf32>, %W4v: tensor<16x16x3x3xf32>, %cb4v: tensor<16xf32>, %W5v: tensor<32x16x3x3xf32>, %cb5v: tensor<32xf32>, %W6v: tensor<32x32x3x3xf32>, %cb6v: tensor<32xf32>, %W7v: tensor<32x32x3x3xf32>, %cb7v: tensor<32xf32>, %W8v: tensor<32x32x3x3xf32>, %cb8v: tensor<32xf32>, %W9v: tensor<128x512xf32>, %b9v: tensor<512xf32>, %Wav: tensor<512x512xf32>, %bav: tensor<512xf32>, %Wbv: tensor<512x10xf32>, %bbv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: tensor<128x10xf32>) -> (tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>) {
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
    %v293 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v294 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v295 = stablehlo.multiply %v293, %W1m : tensor<16x3x3x3xf32>
    %v296 = stablehlo.multiply %v294, %v292 : tensor<16x3x3x3xf32>
    %v297 = stablehlo.add %v295, %v296 : tensor<16x3x3x3xf32>
    %v298 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v299 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v300 = stablehlo.multiply %v298, %W1v : tensor<16x3x3x3xf32>
    %v301 = stablehlo.multiply %v292, %v292 : tensor<16x3x3x3xf32>
    %v302 = stablehlo.multiply %v299, %v301 : tensor<16x3x3x3xf32>
    %v303 = stablehlo.add %v300, %v302 : tensor<16x3x3x3xf32>
    %v304 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v305 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v306 = stablehlo.divide %v297, %v304 : tensor<16x3x3x3xf32>
    %v307 = stablehlo.divide %v303, %v305 : tensor<16x3x3x3xf32>
    %v308 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v309 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v310 = stablehlo.sqrt %v307 : tensor<16x3x3x3xf32>
    %v311 = stablehlo.add %v310, %v309 : tensor<16x3x3x3xf32>
    %v312 = stablehlo.divide %v306, %v311 : tensor<16x3x3x3xf32>
    %v313 = stablehlo.multiply %v308, %v312 : tensor<16x3x3x3xf32>
    %v314 = stablehlo.subtract %W1, %v313 : tensor<16x3x3x3xf32>
    %v315 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v316 = stablehlo.multiply %v315, %v308 : tensor<16x3x3x3xf32>
    %v317 = stablehlo.multiply %v316, %W1 : tensor<16x3x3x3xf32>
    %v318 = stablehlo.subtract %v314, %v317 : tensor<16x3x3x3xf32>
    %v319 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v320 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v321 = stablehlo.multiply %v319, %W1m : tensor<16x3x3x3xf32>
    %v322 = stablehlo.multiply %v320, %v292 : tensor<16x3x3x3xf32>
    %v323 = stablehlo.add %v321, %v322 : tensor<16x3x3x3xf32>
    %v324 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v325 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v326 = stablehlo.multiply %v324, %W1v : tensor<16x3x3x3xf32>
    %v327 = stablehlo.multiply %v292, %v292 : tensor<16x3x3x3xf32>
    %v328 = stablehlo.multiply %v325, %v327 : tensor<16x3x3x3xf32>
    %v329 = stablehlo.add %v326, %v328 : tensor<16x3x3x3xf32>
    %v330 = stablehlo.reshape %v283 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v331 = stablehlo.constant dense<0.0> : tensor<f32>
    %v332 = stablehlo.reduce(%v330 init: %v331) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v333 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v334 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v335 = stablehlo.multiply %v333, %cb1m : tensor<16xf32>
    %v336 = stablehlo.multiply %v334, %v332 : tensor<16xf32>
    %v337 = stablehlo.add %v335, %v336 : tensor<16xf32>
    %v338 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v339 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v340 = stablehlo.multiply %v338, %cb1v : tensor<16xf32>
    %v341 = stablehlo.multiply %v332, %v332 : tensor<16xf32>
    %v342 = stablehlo.multiply %v339, %v341 : tensor<16xf32>
    %v343 = stablehlo.add %v340, %v342 : tensor<16xf32>
    %v344 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v345 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v346 = stablehlo.divide %v337, %v344 : tensor<16xf32>
    %v347 = stablehlo.divide %v343, %v345 : tensor<16xf32>
    %v348 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v349 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v350 = stablehlo.sqrt %v347 : tensor<16xf32>
    %v351 = stablehlo.add %v350, %v349 : tensor<16xf32>
    %v352 = stablehlo.divide %v346, %v351 : tensor<16xf32>
    %v353 = stablehlo.multiply %v348, %v352 : tensor<16xf32>
    %v354 = stablehlo.subtract %cb1, %v353 : tensor<16xf32>
    %v355 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v356 = stablehlo.multiply %v355, %v348 : tensor<16xf32>
    %v357 = stablehlo.multiply %v356, %cb1 : tensor<16xf32>
    %v358 = stablehlo.subtract %v354, %v357 : tensor<16xf32>
    %v359 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v360 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v361 = stablehlo.multiply %v359, %cb1m : tensor<16xf32>
    %v362 = stablehlo.multiply %v360, %v332 : tensor<16xf32>
    %v363 = stablehlo.add %v361, %v362 : tensor<16xf32>
    %v364 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v365 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v366 = stablehlo.multiply %v364, %cb1v : tensor<16xf32>
    %v367 = stablehlo.multiply %v332, %v332 : tensor<16xf32>
    %v368 = stablehlo.multiply %v365, %v367 : tensor<16xf32>
    %v369 = stablehlo.add %v366, %v368 : tensor<16xf32>
    %v370 = stablehlo.reshape %v11 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v371 = stablehlo.reshape %v269 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v372 = stablehlo.transpose %v370, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v373 = stablehlo.transpose %v371, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v374 = stablehlo.convert %v372 : (tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xbf16>
    %v375 = stablehlo.convert %v373 : (tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xbf16>
    %v376 = stablehlo.convolution(%v374, %v375)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xbf16>, tensor<16x128x32x32xbf16>) -> tensor<16x16x3x3xbf16>
    %v377 = stablehlo.convert %v376 : (tensor<16x16x3x3xbf16>) -> tensor<16x16x3x3xf32>
    %v378 = stablehlo.transpose %v377, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v379 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v380 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v381 = stablehlo.multiply %v379, %W2m : tensor<16x16x3x3xf32>
    %v382 = stablehlo.multiply %v380, %v378 : tensor<16x16x3x3xf32>
    %v383 = stablehlo.add %v381, %v382 : tensor<16x16x3x3xf32>
    %v384 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v385 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v386 = stablehlo.multiply %v384, %W2v : tensor<16x16x3x3xf32>
    %v387 = stablehlo.multiply %v378, %v378 : tensor<16x16x3x3xf32>
    %v388 = stablehlo.multiply %v385, %v387 : tensor<16x16x3x3xf32>
    %v389 = stablehlo.add %v386, %v388 : tensor<16x16x3x3xf32>
    %v390 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v391 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v392 = stablehlo.divide %v383, %v390 : tensor<16x16x3x3xf32>
    %v393 = stablehlo.divide %v389, %v391 : tensor<16x16x3x3xf32>
    %v394 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v395 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v396 = stablehlo.sqrt %v393 : tensor<16x16x3x3xf32>
    %v397 = stablehlo.add %v396, %v395 : tensor<16x16x3x3xf32>
    %v398 = stablehlo.divide %v392, %v397 : tensor<16x16x3x3xf32>
    %v399 = stablehlo.multiply %v394, %v398 : tensor<16x16x3x3xf32>
    %v400 = stablehlo.subtract %W2, %v399 : tensor<16x16x3x3xf32>
    %v401 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v402 = stablehlo.multiply %v401, %v394 : tensor<16x16x3x3xf32>
    %v403 = stablehlo.multiply %v402, %W2 : tensor<16x16x3x3xf32>
    %v404 = stablehlo.subtract %v400, %v403 : tensor<16x16x3x3xf32>
    %v405 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v406 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v407 = stablehlo.multiply %v405, %W2m : tensor<16x16x3x3xf32>
    %v408 = stablehlo.multiply %v406, %v378 : tensor<16x16x3x3xf32>
    %v409 = stablehlo.add %v407, %v408 : tensor<16x16x3x3xf32>
    %v410 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v411 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v412 = stablehlo.multiply %v410, %W2v : tensor<16x16x3x3xf32>
    %v413 = stablehlo.multiply %v378, %v378 : tensor<16x16x3x3xf32>
    %v414 = stablehlo.multiply %v411, %v413 : tensor<16x16x3x3xf32>
    %v415 = stablehlo.add %v412, %v414 : tensor<16x16x3x3xf32>
    %v416 = stablehlo.reshape %v269 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v417 = stablehlo.constant dense<0.0> : tensor<f32>
    %v418 = stablehlo.reduce(%v416 init: %v417) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v419 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v420 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v421 = stablehlo.multiply %v419, %cb2m : tensor<16xf32>
    %v422 = stablehlo.multiply %v420, %v418 : tensor<16xf32>
    %v423 = stablehlo.add %v421, %v422 : tensor<16xf32>
    %v424 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v425 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v426 = stablehlo.multiply %v424, %cb2v : tensor<16xf32>
    %v427 = stablehlo.multiply %v418, %v418 : tensor<16xf32>
    %v428 = stablehlo.multiply %v425, %v427 : tensor<16xf32>
    %v429 = stablehlo.add %v426, %v428 : tensor<16xf32>
    %v430 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v431 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v432 = stablehlo.divide %v423, %v430 : tensor<16xf32>
    %v433 = stablehlo.divide %v429, %v431 : tensor<16xf32>
    %v434 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v435 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v436 = stablehlo.sqrt %v433 : tensor<16xf32>
    %v437 = stablehlo.add %v436, %v435 : tensor<16xf32>
    %v438 = stablehlo.divide %v432, %v437 : tensor<16xf32>
    %v439 = stablehlo.multiply %v434, %v438 : tensor<16xf32>
    %v440 = stablehlo.subtract %cb2, %v439 : tensor<16xf32>
    %v441 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v442 = stablehlo.multiply %v441, %v434 : tensor<16xf32>
    %v443 = stablehlo.multiply %v442, %cb2 : tensor<16xf32>
    %v444 = stablehlo.subtract %v440, %v443 : tensor<16xf32>
    %v445 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v446 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v447 = stablehlo.multiply %v445, %cb2m : tensor<16xf32>
    %v448 = stablehlo.multiply %v446, %v418 : tensor<16xf32>
    %v449 = stablehlo.add %v447, %v448 : tensor<16xf32>
    %v450 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v451 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v452 = stablehlo.multiply %v450, %cb2v : tensor<16xf32>
    %v453 = stablehlo.multiply %v418, %v418 : tensor<16xf32>
    %v454 = stablehlo.multiply %v451, %v453 : tensor<16xf32>
    %v455 = stablehlo.add %v452, %v454 : tensor<16xf32>
    %v456 = stablehlo.reshape %v27 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v457 = stablehlo.reshape %v250 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v458 = stablehlo.transpose %v456, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v459 = stablehlo.transpose %v457, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v460 = stablehlo.convert %v458 : (tensor<16x128x16x16xf32>) -> tensor<16x128x16x16xbf16>
    %v461 = stablehlo.convert %v459 : (tensor<16x128x16x16xf32>) -> tensor<16x128x16x16xbf16>
    %v462 = stablehlo.convolution(%v460, %v461)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xbf16>, tensor<16x128x16x16xbf16>) -> tensor<16x16x3x3xbf16>
    %v463 = stablehlo.convert %v462 : (tensor<16x16x3x3xbf16>) -> tensor<16x16x3x3xf32>
    %v464 = stablehlo.transpose %v463, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v465 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v466 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v467 = stablehlo.multiply %v465, %W3m : tensor<16x16x3x3xf32>
    %v468 = stablehlo.multiply %v466, %v464 : tensor<16x16x3x3xf32>
    %v469 = stablehlo.add %v467, %v468 : tensor<16x16x3x3xf32>
    %v470 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v471 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v472 = stablehlo.multiply %v470, %W3v : tensor<16x16x3x3xf32>
    %v473 = stablehlo.multiply %v464, %v464 : tensor<16x16x3x3xf32>
    %v474 = stablehlo.multiply %v471, %v473 : tensor<16x16x3x3xf32>
    %v475 = stablehlo.add %v472, %v474 : tensor<16x16x3x3xf32>
    %v476 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v477 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v478 = stablehlo.divide %v469, %v476 : tensor<16x16x3x3xf32>
    %v479 = stablehlo.divide %v475, %v477 : tensor<16x16x3x3xf32>
    %v480 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v481 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v482 = stablehlo.sqrt %v479 : tensor<16x16x3x3xf32>
    %v483 = stablehlo.add %v482, %v481 : tensor<16x16x3x3xf32>
    %v484 = stablehlo.divide %v478, %v483 : tensor<16x16x3x3xf32>
    %v485 = stablehlo.multiply %v480, %v484 : tensor<16x16x3x3xf32>
    %v486 = stablehlo.subtract %W3, %v485 : tensor<16x16x3x3xf32>
    %v487 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v488 = stablehlo.multiply %v487, %v480 : tensor<16x16x3x3xf32>
    %v489 = stablehlo.multiply %v488, %W3 : tensor<16x16x3x3xf32>
    %v490 = stablehlo.subtract %v486, %v489 : tensor<16x16x3x3xf32>
    %v491 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v492 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v493 = stablehlo.multiply %v491, %W3m : tensor<16x16x3x3xf32>
    %v494 = stablehlo.multiply %v492, %v464 : tensor<16x16x3x3xf32>
    %v495 = stablehlo.add %v493, %v494 : tensor<16x16x3x3xf32>
    %v496 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v497 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v498 = stablehlo.multiply %v496, %W3v : tensor<16x16x3x3xf32>
    %v499 = stablehlo.multiply %v464, %v464 : tensor<16x16x3x3xf32>
    %v500 = stablehlo.multiply %v497, %v499 : tensor<16x16x3x3xf32>
    %v501 = stablehlo.add %v498, %v500 : tensor<16x16x3x3xf32>
    %v502 = stablehlo.reshape %v250 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v503 = stablehlo.constant dense<0.0> : tensor<f32>
    %v504 = stablehlo.reduce(%v502 init: %v503) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v505 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v506 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v507 = stablehlo.multiply %v505, %cb3m : tensor<16xf32>
    %v508 = stablehlo.multiply %v506, %v504 : tensor<16xf32>
    %v509 = stablehlo.add %v507, %v508 : tensor<16xf32>
    %v510 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v511 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v512 = stablehlo.multiply %v510, %cb3v : tensor<16xf32>
    %v513 = stablehlo.multiply %v504, %v504 : tensor<16xf32>
    %v514 = stablehlo.multiply %v511, %v513 : tensor<16xf32>
    %v515 = stablehlo.add %v512, %v514 : tensor<16xf32>
    %v516 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v517 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v518 = stablehlo.divide %v509, %v516 : tensor<16xf32>
    %v519 = stablehlo.divide %v515, %v517 : tensor<16xf32>
    %v520 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v521 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v522 = stablehlo.sqrt %v519 : tensor<16xf32>
    %v523 = stablehlo.add %v522, %v521 : tensor<16xf32>
    %v524 = stablehlo.divide %v518, %v523 : tensor<16xf32>
    %v525 = stablehlo.multiply %v520, %v524 : tensor<16xf32>
    %v526 = stablehlo.subtract %cb3, %v525 : tensor<16xf32>
    %v527 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v528 = stablehlo.multiply %v527, %v520 : tensor<16xf32>
    %v529 = stablehlo.multiply %v528, %cb3 : tensor<16xf32>
    %v530 = stablehlo.subtract %v526, %v529 : tensor<16xf32>
    %v531 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v532 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v533 = stablehlo.multiply %v531, %cb3m : tensor<16xf32>
    %v534 = stablehlo.multiply %v532, %v504 : tensor<16xf32>
    %v535 = stablehlo.add %v533, %v534 : tensor<16xf32>
    %v536 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v537 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v538 = stablehlo.multiply %v536, %cb3v : tensor<16xf32>
    %v539 = stablehlo.multiply %v504, %v504 : tensor<16xf32>
    %v540 = stablehlo.multiply %v537, %v539 : tensor<16xf32>
    %v541 = stablehlo.add %v538, %v540 : tensor<16xf32>
    %v542 = stablehlo.reshape %v39 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v543 = stablehlo.reshape %v236 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v544 = stablehlo.transpose %v542, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v545 = stablehlo.transpose %v543, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v546 = stablehlo.convert %v544 : (tensor<16x128x16x16xf32>) -> tensor<16x128x16x16xbf16>
    %v547 = stablehlo.convert %v545 : (tensor<16x128x16x16xf32>) -> tensor<16x128x16x16xbf16>
    %v548 = stablehlo.convolution(%v546, %v547)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xbf16>, tensor<16x128x16x16xbf16>) -> tensor<16x16x3x3xbf16>
    %v549 = stablehlo.convert %v548 : (tensor<16x16x3x3xbf16>) -> tensor<16x16x3x3xf32>
    %v550 = stablehlo.transpose %v549, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v551 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v552 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v553 = stablehlo.multiply %v551, %W4m : tensor<16x16x3x3xf32>
    %v554 = stablehlo.multiply %v552, %v550 : tensor<16x16x3x3xf32>
    %v555 = stablehlo.add %v553, %v554 : tensor<16x16x3x3xf32>
    %v556 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v557 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v558 = stablehlo.multiply %v556, %W4v : tensor<16x16x3x3xf32>
    %v559 = stablehlo.multiply %v550, %v550 : tensor<16x16x3x3xf32>
    %v560 = stablehlo.multiply %v557, %v559 : tensor<16x16x3x3xf32>
    %v561 = stablehlo.add %v558, %v560 : tensor<16x16x3x3xf32>
    %v562 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v563 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v564 = stablehlo.divide %v555, %v562 : tensor<16x16x3x3xf32>
    %v565 = stablehlo.divide %v561, %v563 : tensor<16x16x3x3xf32>
    %v566 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v567 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v568 = stablehlo.sqrt %v565 : tensor<16x16x3x3xf32>
    %v569 = stablehlo.add %v568, %v567 : tensor<16x16x3x3xf32>
    %v570 = stablehlo.divide %v564, %v569 : tensor<16x16x3x3xf32>
    %v571 = stablehlo.multiply %v566, %v570 : tensor<16x16x3x3xf32>
    %v572 = stablehlo.subtract %W4, %v571 : tensor<16x16x3x3xf32>
    %v573 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v574 = stablehlo.multiply %v573, %v566 : tensor<16x16x3x3xf32>
    %v575 = stablehlo.multiply %v574, %W4 : tensor<16x16x3x3xf32>
    %v576 = stablehlo.subtract %v572, %v575 : tensor<16x16x3x3xf32>
    %v577 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v578 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v579 = stablehlo.multiply %v577, %W4m : tensor<16x16x3x3xf32>
    %v580 = stablehlo.multiply %v578, %v550 : tensor<16x16x3x3xf32>
    %v581 = stablehlo.add %v579, %v580 : tensor<16x16x3x3xf32>
    %v582 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v583 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v584 = stablehlo.multiply %v582, %W4v : tensor<16x16x3x3xf32>
    %v585 = stablehlo.multiply %v550, %v550 : tensor<16x16x3x3xf32>
    %v586 = stablehlo.multiply %v583, %v585 : tensor<16x16x3x3xf32>
    %v587 = stablehlo.add %v584, %v586 : tensor<16x16x3x3xf32>
    %v588 = stablehlo.reshape %v236 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v589 = stablehlo.constant dense<0.0> : tensor<f32>
    %v590 = stablehlo.reduce(%v588 init: %v589) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v591 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v592 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v593 = stablehlo.multiply %v591, %cb4m : tensor<16xf32>
    %v594 = stablehlo.multiply %v592, %v590 : tensor<16xf32>
    %v595 = stablehlo.add %v593, %v594 : tensor<16xf32>
    %v596 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v597 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v598 = stablehlo.multiply %v596, %cb4v : tensor<16xf32>
    %v599 = stablehlo.multiply %v590, %v590 : tensor<16xf32>
    %v600 = stablehlo.multiply %v597, %v599 : tensor<16xf32>
    %v601 = stablehlo.add %v598, %v600 : tensor<16xf32>
    %v602 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v603 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v604 = stablehlo.divide %v595, %v602 : tensor<16xf32>
    %v605 = stablehlo.divide %v601, %v603 : tensor<16xf32>
    %v606 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v607 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v608 = stablehlo.sqrt %v605 : tensor<16xf32>
    %v609 = stablehlo.add %v608, %v607 : tensor<16xf32>
    %v610 = stablehlo.divide %v604, %v609 : tensor<16xf32>
    %v611 = stablehlo.multiply %v606, %v610 : tensor<16xf32>
    %v612 = stablehlo.subtract %cb4, %v611 : tensor<16xf32>
    %v613 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v614 = stablehlo.multiply %v613, %v606 : tensor<16xf32>
    %v615 = stablehlo.multiply %v614, %cb4 : tensor<16xf32>
    %v616 = stablehlo.subtract %v612, %v615 : tensor<16xf32>
    %v617 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v618 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v619 = stablehlo.multiply %v617, %cb4m : tensor<16xf32>
    %v620 = stablehlo.multiply %v618, %v590 : tensor<16xf32>
    %v621 = stablehlo.add %v619, %v620 : tensor<16xf32>
    %v622 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v623 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v624 = stablehlo.multiply %v622, %cb4v : tensor<16xf32>
    %v625 = stablehlo.multiply %v590, %v590 : tensor<16xf32>
    %v626 = stablehlo.multiply %v623, %v625 : tensor<16xf32>
    %v627 = stablehlo.add %v624, %v626 : tensor<16xf32>
    %v628 = stablehlo.reshape %v55 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v629 = stablehlo.reshape %v217 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v630 = stablehlo.transpose %v628, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v631 = stablehlo.transpose %v629, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v632 = stablehlo.convert %v630 : (tensor<16x128x8x8xf32>) -> tensor<16x128x8x8xbf16>
    %v633 = stablehlo.convert %v631 : (tensor<32x128x8x8xf32>) -> tensor<32x128x8x8xbf16>
    %v634 = stablehlo.convolution(%v632, %v633)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xbf16>, tensor<32x128x8x8xbf16>) -> tensor<16x32x3x3xbf16>
    %v635 = stablehlo.convert %v634 : (tensor<16x32x3x3xbf16>) -> tensor<16x32x3x3xf32>
    %v636 = stablehlo.transpose %v635, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v637 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v638 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v639 = stablehlo.multiply %v637, %W5m : tensor<32x16x3x3xf32>
    %v640 = stablehlo.multiply %v638, %v636 : tensor<32x16x3x3xf32>
    %v641 = stablehlo.add %v639, %v640 : tensor<32x16x3x3xf32>
    %v642 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v643 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v644 = stablehlo.multiply %v642, %W5v : tensor<32x16x3x3xf32>
    %v645 = stablehlo.multiply %v636, %v636 : tensor<32x16x3x3xf32>
    %v646 = stablehlo.multiply %v643, %v645 : tensor<32x16x3x3xf32>
    %v647 = stablehlo.add %v644, %v646 : tensor<32x16x3x3xf32>
    %v648 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v649 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v650 = stablehlo.divide %v641, %v648 : tensor<32x16x3x3xf32>
    %v651 = stablehlo.divide %v647, %v649 : tensor<32x16x3x3xf32>
    %v652 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v653 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v654 = stablehlo.sqrt %v651 : tensor<32x16x3x3xf32>
    %v655 = stablehlo.add %v654, %v653 : tensor<32x16x3x3xf32>
    %v656 = stablehlo.divide %v650, %v655 : tensor<32x16x3x3xf32>
    %v657 = stablehlo.multiply %v652, %v656 : tensor<32x16x3x3xf32>
    %v658 = stablehlo.subtract %W5, %v657 : tensor<32x16x3x3xf32>
    %v659 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v660 = stablehlo.multiply %v659, %v652 : tensor<32x16x3x3xf32>
    %v661 = stablehlo.multiply %v660, %W5 : tensor<32x16x3x3xf32>
    %v662 = stablehlo.subtract %v658, %v661 : tensor<32x16x3x3xf32>
    %v663 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v664 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v665 = stablehlo.multiply %v663, %W5m : tensor<32x16x3x3xf32>
    %v666 = stablehlo.multiply %v664, %v636 : tensor<32x16x3x3xf32>
    %v667 = stablehlo.add %v665, %v666 : tensor<32x16x3x3xf32>
    %v668 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v669 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v670 = stablehlo.multiply %v668, %W5v : tensor<32x16x3x3xf32>
    %v671 = stablehlo.multiply %v636, %v636 : tensor<32x16x3x3xf32>
    %v672 = stablehlo.multiply %v669, %v671 : tensor<32x16x3x3xf32>
    %v673 = stablehlo.add %v670, %v672 : tensor<32x16x3x3xf32>
    %v674 = stablehlo.reshape %v217 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v675 = stablehlo.constant dense<0.0> : tensor<f32>
    %v676 = stablehlo.reduce(%v674 init: %v675) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v677 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v678 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v679 = stablehlo.multiply %v677, %cb5m : tensor<32xf32>
    %v680 = stablehlo.multiply %v678, %v676 : tensor<32xf32>
    %v681 = stablehlo.add %v679, %v680 : tensor<32xf32>
    %v682 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v683 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v684 = stablehlo.multiply %v682, %cb5v : tensor<32xf32>
    %v685 = stablehlo.multiply %v676, %v676 : tensor<32xf32>
    %v686 = stablehlo.multiply %v683, %v685 : tensor<32xf32>
    %v687 = stablehlo.add %v684, %v686 : tensor<32xf32>
    %v688 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v689 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v690 = stablehlo.divide %v681, %v688 : tensor<32xf32>
    %v691 = stablehlo.divide %v687, %v689 : tensor<32xf32>
    %v692 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v693 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v694 = stablehlo.sqrt %v691 : tensor<32xf32>
    %v695 = stablehlo.add %v694, %v693 : tensor<32xf32>
    %v696 = stablehlo.divide %v690, %v695 : tensor<32xf32>
    %v697 = stablehlo.multiply %v692, %v696 : tensor<32xf32>
    %v698 = stablehlo.subtract %cb5, %v697 : tensor<32xf32>
    %v699 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v700 = stablehlo.multiply %v699, %v692 : tensor<32xf32>
    %v701 = stablehlo.multiply %v700, %cb5 : tensor<32xf32>
    %v702 = stablehlo.subtract %v698, %v701 : tensor<32xf32>
    %v703 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v704 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v705 = stablehlo.multiply %v703, %cb5m : tensor<32xf32>
    %v706 = stablehlo.multiply %v704, %v676 : tensor<32xf32>
    %v707 = stablehlo.add %v705, %v706 : tensor<32xf32>
    %v708 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v709 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v710 = stablehlo.multiply %v708, %cb5v : tensor<32xf32>
    %v711 = stablehlo.multiply %v676, %v676 : tensor<32xf32>
    %v712 = stablehlo.multiply %v709, %v711 : tensor<32xf32>
    %v713 = stablehlo.add %v710, %v712 : tensor<32xf32>
    %v714 = stablehlo.reshape %v67 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v715 = stablehlo.reshape %v203 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v716 = stablehlo.transpose %v714, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v717 = stablehlo.transpose %v715, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v718 = stablehlo.convert %v716 : (tensor<32x128x8x8xf32>) -> tensor<32x128x8x8xbf16>
    %v719 = stablehlo.convert %v717 : (tensor<32x128x8x8xf32>) -> tensor<32x128x8x8xbf16>
    %v720 = stablehlo.convolution(%v718, %v719)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xbf16>, tensor<32x128x8x8xbf16>) -> tensor<32x32x3x3xbf16>
    %v721 = stablehlo.convert %v720 : (tensor<32x32x3x3xbf16>) -> tensor<32x32x3x3xf32>
    %v722 = stablehlo.transpose %v721, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v723 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v724 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v725 = stablehlo.multiply %v723, %W6m : tensor<32x32x3x3xf32>
    %v726 = stablehlo.multiply %v724, %v722 : tensor<32x32x3x3xf32>
    %v727 = stablehlo.add %v725, %v726 : tensor<32x32x3x3xf32>
    %v728 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v729 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v730 = stablehlo.multiply %v728, %W6v : tensor<32x32x3x3xf32>
    %v731 = stablehlo.multiply %v722, %v722 : tensor<32x32x3x3xf32>
    %v732 = stablehlo.multiply %v729, %v731 : tensor<32x32x3x3xf32>
    %v733 = stablehlo.add %v730, %v732 : tensor<32x32x3x3xf32>
    %v734 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v735 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v736 = stablehlo.divide %v727, %v734 : tensor<32x32x3x3xf32>
    %v737 = stablehlo.divide %v733, %v735 : tensor<32x32x3x3xf32>
    %v738 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v739 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v740 = stablehlo.sqrt %v737 : tensor<32x32x3x3xf32>
    %v741 = stablehlo.add %v740, %v739 : tensor<32x32x3x3xf32>
    %v742 = stablehlo.divide %v736, %v741 : tensor<32x32x3x3xf32>
    %v743 = stablehlo.multiply %v738, %v742 : tensor<32x32x3x3xf32>
    %v744 = stablehlo.subtract %W6, %v743 : tensor<32x32x3x3xf32>
    %v745 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v746 = stablehlo.multiply %v745, %v738 : tensor<32x32x3x3xf32>
    %v747 = stablehlo.multiply %v746, %W6 : tensor<32x32x3x3xf32>
    %v748 = stablehlo.subtract %v744, %v747 : tensor<32x32x3x3xf32>
    %v749 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v750 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v751 = stablehlo.multiply %v749, %W6m : tensor<32x32x3x3xf32>
    %v752 = stablehlo.multiply %v750, %v722 : tensor<32x32x3x3xf32>
    %v753 = stablehlo.add %v751, %v752 : tensor<32x32x3x3xf32>
    %v754 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v755 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v756 = stablehlo.multiply %v754, %W6v : tensor<32x32x3x3xf32>
    %v757 = stablehlo.multiply %v722, %v722 : tensor<32x32x3x3xf32>
    %v758 = stablehlo.multiply %v755, %v757 : tensor<32x32x3x3xf32>
    %v759 = stablehlo.add %v756, %v758 : tensor<32x32x3x3xf32>
    %v760 = stablehlo.reshape %v203 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v761 = stablehlo.constant dense<0.0> : tensor<f32>
    %v762 = stablehlo.reduce(%v760 init: %v761) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v763 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v764 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v765 = stablehlo.multiply %v763, %cb6m : tensor<32xf32>
    %v766 = stablehlo.multiply %v764, %v762 : tensor<32xf32>
    %v767 = stablehlo.add %v765, %v766 : tensor<32xf32>
    %v768 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v769 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v770 = stablehlo.multiply %v768, %cb6v : tensor<32xf32>
    %v771 = stablehlo.multiply %v762, %v762 : tensor<32xf32>
    %v772 = stablehlo.multiply %v769, %v771 : tensor<32xf32>
    %v773 = stablehlo.add %v770, %v772 : tensor<32xf32>
    %v774 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v775 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v776 = stablehlo.divide %v767, %v774 : tensor<32xf32>
    %v777 = stablehlo.divide %v773, %v775 : tensor<32xf32>
    %v778 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v779 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v780 = stablehlo.sqrt %v777 : tensor<32xf32>
    %v781 = stablehlo.add %v780, %v779 : tensor<32xf32>
    %v782 = stablehlo.divide %v776, %v781 : tensor<32xf32>
    %v783 = stablehlo.multiply %v778, %v782 : tensor<32xf32>
    %v784 = stablehlo.subtract %cb6, %v783 : tensor<32xf32>
    %v785 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v786 = stablehlo.multiply %v785, %v778 : tensor<32xf32>
    %v787 = stablehlo.multiply %v786, %cb6 : tensor<32xf32>
    %v788 = stablehlo.subtract %v784, %v787 : tensor<32xf32>
    %v789 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v790 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v791 = stablehlo.multiply %v789, %cb6m : tensor<32xf32>
    %v792 = stablehlo.multiply %v790, %v762 : tensor<32xf32>
    %v793 = stablehlo.add %v791, %v792 : tensor<32xf32>
    %v794 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v795 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v796 = stablehlo.multiply %v794, %cb6v : tensor<32xf32>
    %v797 = stablehlo.multiply %v762, %v762 : tensor<32xf32>
    %v798 = stablehlo.multiply %v795, %v797 : tensor<32xf32>
    %v799 = stablehlo.add %v796, %v798 : tensor<32xf32>
    %v800 = stablehlo.reshape %v83 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v801 = stablehlo.reshape %v184 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v802 = stablehlo.transpose %v800, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v803 = stablehlo.transpose %v801, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v804 = stablehlo.convert %v802 : (tensor<32x128x4x4xf32>) -> tensor<32x128x4x4xbf16>
    %v805 = stablehlo.convert %v803 : (tensor<32x128x4x4xf32>) -> tensor<32x128x4x4xbf16>
    %v806 = stablehlo.convolution(%v804, %v805)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xbf16>, tensor<32x128x4x4xbf16>) -> tensor<32x32x3x3xbf16>
    %v807 = stablehlo.convert %v806 : (tensor<32x32x3x3xbf16>) -> tensor<32x32x3x3xf32>
    %v808 = stablehlo.transpose %v807, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v809 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v810 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v811 = stablehlo.multiply %v809, %W7m : tensor<32x32x3x3xf32>
    %v812 = stablehlo.multiply %v810, %v808 : tensor<32x32x3x3xf32>
    %v813 = stablehlo.add %v811, %v812 : tensor<32x32x3x3xf32>
    %v814 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v815 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v816 = stablehlo.multiply %v814, %W7v : tensor<32x32x3x3xf32>
    %v817 = stablehlo.multiply %v808, %v808 : tensor<32x32x3x3xf32>
    %v818 = stablehlo.multiply %v815, %v817 : tensor<32x32x3x3xf32>
    %v819 = stablehlo.add %v816, %v818 : tensor<32x32x3x3xf32>
    %v820 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v821 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v822 = stablehlo.divide %v813, %v820 : tensor<32x32x3x3xf32>
    %v823 = stablehlo.divide %v819, %v821 : tensor<32x32x3x3xf32>
    %v824 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v825 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v826 = stablehlo.sqrt %v823 : tensor<32x32x3x3xf32>
    %v827 = stablehlo.add %v826, %v825 : tensor<32x32x3x3xf32>
    %v828 = stablehlo.divide %v822, %v827 : tensor<32x32x3x3xf32>
    %v829 = stablehlo.multiply %v824, %v828 : tensor<32x32x3x3xf32>
    %v830 = stablehlo.subtract %W7, %v829 : tensor<32x32x3x3xf32>
    %v831 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v832 = stablehlo.multiply %v831, %v824 : tensor<32x32x3x3xf32>
    %v833 = stablehlo.multiply %v832, %W7 : tensor<32x32x3x3xf32>
    %v834 = stablehlo.subtract %v830, %v833 : tensor<32x32x3x3xf32>
    %v835 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v836 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v837 = stablehlo.multiply %v835, %W7m : tensor<32x32x3x3xf32>
    %v838 = stablehlo.multiply %v836, %v808 : tensor<32x32x3x3xf32>
    %v839 = stablehlo.add %v837, %v838 : tensor<32x32x3x3xf32>
    %v840 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v841 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v842 = stablehlo.multiply %v840, %W7v : tensor<32x32x3x3xf32>
    %v843 = stablehlo.multiply %v808, %v808 : tensor<32x32x3x3xf32>
    %v844 = stablehlo.multiply %v841, %v843 : tensor<32x32x3x3xf32>
    %v845 = stablehlo.add %v842, %v844 : tensor<32x32x3x3xf32>
    %v846 = stablehlo.reshape %v184 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v847 = stablehlo.constant dense<0.0> : tensor<f32>
    %v848 = stablehlo.reduce(%v846 init: %v847) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v849 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v850 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v851 = stablehlo.multiply %v849, %cb7m : tensor<32xf32>
    %v852 = stablehlo.multiply %v850, %v848 : tensor<32xf32>
    %v853 = stablehlo.add %v851, %v852 : tensor<32xf32>
    %v854 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v855 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v856 = stablehlo.multiply %v854, %cb7v : tensor<32xf32>
    %v857 = stablehlo.multiply %v848, %v848 : tensor<32xf32>
    %v858 = stablehlo.multiply %v855, %v857 : tensor<32xf32>
    %v859 = stablehlo.add %v856, %v858 : tensor<32xf32>
    %v860 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v861 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v862 = stablehlo.divide %v853, %v860 : tensor<32xf32>
    %v863 = stablehlo.divide %v859, %v861 : tensor<32xf32>
    %v864 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v865 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v866 = stablehlo.sqrt %v863 : tensor<32xf32>
    %v867 = stablehlo.add %v866, %v865 : tensor<32xf32>
    %v868 = stablehlo.divide %v862, %v867 : tensor<32xf32>
    %v869 = stablehlo.multiply %v864, %v868 : tensor<32xf32>
    %v870 = stablehlo.subtract %cb7, %v869 : tensor<32xf32>
    %v871 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v872 = stablehlo.multiply %v871, %v864 : tensor<32xf32>
    %v873 = stablehlo.multiply %v872, %cb7 : tensor<32xf32>
    %v874 = stablehlo.subtract %v870, %v873 : tensor<32xf32>
    %v875 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v876 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v877 = stablehlo.multiply %v875, %cb7m : tensor<32xf32>
    %v878 = stablehlo.multiply %v876, %v848 : tensor<32xf32>
    %v879 = stablehlo.add %v877, %v878 : tensor<32xf32>
    %v880 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v881 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v882 = stablehlo.multiply %v880, %cb7v : tensor<32xf32>
    %v883 = stablehlo.multiply %v848, %v848 : tensor<32xf32>
    %v884 = stablehlo.multiply %v881, %v883 : tensor<32xf32>
    %v885 = stablehlo.add %v882, %v884 : tensor<32xf32>
    %v886 = stablehlo.reshape %v95 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v887 = stablehlo.reshape %v170 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v888 = stablehlo.transpose %v886, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v889 = stablehlo.transpose %v887, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v890 = stablehlo.convert %v888 : (tensor<32x128x4x4xf32>) -> tensor<32x128x4x4xbf16>
    %v891 = stablehlo.convert %v889 : (tensor<32x128x4x4xf32>) -> tensor<32x128x4x4xbf16>
    %v892 = stablehlo.convolution(%v890, %v891)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xbf16>, tensor<32x128x4x4xbf16>) -> tensor<32x32x3x3xbf16>
    %v893 = stablehlo.convert %v892 : (tensor<32x32x3x3xbf16>) -> tensor<32x32x3x3xf32>
    %v894 = stablehlo.transpose %v893, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v895 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v896 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v897 = stablehlo.multiply %v895, %W8m : tensor<32x32x3x3xf32>
    %v898 = stablehlo.multiply %v896, %v894 : tensor<32x32x3x3xf32>
    %v899 = stablehlo.add %v897, %v898 : tensor<32x32x3x3xf32>
    %v900 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v901 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v902 = stablehlo.multiply %v900, %W8v : tensor<32x32x3x3xf32>
    %v903 = stablehlo.multiply %v894, %v894 : tensor<32x32x3x3xf32>
    %v904 = stablehlo.multiply %v901, %v903 : tensor<32x32x3x3xf32>
    %v905 = stablehlo.add %v902, %v904 : tensor<32x32x3x3xf32>
    %v906 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v907 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v908 = stablehlo.divide %v899, %v906 : tensor<32x32x3x3xf32>
    %v909 = stablehlo.divide %v905, %v907 : tensor<32x32x3x3xf32>
    %v910 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v911 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v912 = stablehlo.sqrt %v909 : tensor<32x32x3x3xf32>
    %v913 = stablehlo.add %v912, %v911 : tensor<32x32x3x3xf32>
    %v914 = stablehlo.divide %v908, %v913 : tensor<32x32x3x3xf32>
    %v915 = stablehlo.multiply %v910, %v914 : tensor<32x32x3x3xf32>
    %v916 = stablehlo.subtract %W8, %v915 : tensor<32x32x3x3xf32>
    %v917 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v918 = stablehlo.multiply %v917, %v910 : tensor<32x32x3x3xf32>
    %v919 = stablehlo.multiply %v918, %W8 : tensor<32x32x3x3xf32>
    %v920 = stablehlo.subtract %v916, %v919 : tensor<32x32x3x3xf32>
    %v921 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v922 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v923 = stablehlo.multiply %v921, %W8m : tensor<32x32x3x3xf32>
    %v924 = stablehlo.multiply %v922, %v894 : tensor<32x32x3x3xf32>
    %v925 = stablehlo.add %v923, %v924 : tensor<32x32x3x3xf32>
    %v926 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v927 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v928 = stablehlo.multiply %v926, %W8v : tensor<32x32x3x3xf32>
    %v929 = stablehlo.multiply %v894, %v894 : tensor<32x32x3x3xf32>
    %v930 = stablehlo.multiply %v927, %v929 : tensor<32x32x3x3xf32>
    %v931 = stablehlo.add %v928, %v930 : tensor<32x32x3x3xf32>
    %v932 = stablehlo.reshape %v170 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v933 = stablehlo.constant dense<0.0> : tensor<f32>
    %v934 = stablehlo.reduce(%v932 init: %v933) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v935 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v936 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v937 = stablehlo.multiply %v935, %cb8m : tensor<32xf32>
    %v938 = stablehlo.multiply %v936, %v934 : tensor<32xf32>
    %v939 = stablehlo.add %v937, %v938 : tensor<32xf32>
    %v940 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v941 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v942 = stablehlo.multiply %v940, %cb8v : tensor<32xf32>
    %v943 = stablehlo.multiply %v934, %v934 : tensor<32xf32>
    %v944 = stablehlo.multiply %v941, %v943 : tensor<32xf32>
    %v945 = stablehlo.add %v942, %v944 : tensor<32xf32>
    %v946 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v947 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v948 = stablehlo.divide %v939, %v946 : tensor<32xf32>
    %v949 = stablehlo.divide %v945, %v947 : tensor<32xf32>
    %v950 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v951 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v952 = stablehlo.sqrt %v949 : tensor<32xf32>
    %v953 = stablehlo.add %v952, %v951 : tensor<32xf32>
    %v954 = stablehlo.divide %v948, %v953 : tensor<32xf32>
    %v955 = stablehlo.multiply %v950, %v954 : tensor<32xf32>
    %v956 = stablehlo.subtract %cb8, %v955 : tensor<32xf32>
    %v957 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v958 = stablehlo.multiply %v957, %v950 : tensor<32xf32>
    %v959 = stablehlo.multiply %v958, %cb8 : tensor<32xf32>
    %v960 = stablehlo.subtract %v956, %v959 : tensor<32xf32>
    %v961 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v962 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v963 = stablehlo.multiply %v961, %cb8m : tensor<32xf32>
    %v964 = stablehlo.multiply %v962, %v934 : tensor<32xf32>
    %v965 = stablehlo.add %v963, %v964 : tensor<32xf32>
    %v966 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v967 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v968 = stablehlo.multiply %v966, %cb8v : tensor<32xf32>
    %v969 = stablehlo.multiply %v934, %v934 : tensor<32xf32>
    %v970 = stablehlo.multiply %v967, %v969 : tensor<32xf32>
    %v971 = stablehlo.add %v968, %v970 : tensor<32xf32>
    %v972 = stablehlo.dot_general %v111, %v156, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x512xf32>) -> tensor<128x512xf32>
    %v973 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v974 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v975 = stablehlo.multiply %v973, %W9m : tensor<128x512xf32>
    %v976 = stablehlo.multiply %v974, %v972 : tensor<128x512xf32>
    %v977 = stablehlo.add %v975, %v976 : tensor<128x512xf32>
    %v978 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v979 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v980 = stablehlo.multiply %v978, %W9v : tensor<128x512xf32>
    %v981 = stablehlo.multiply %v972, %v972 : tensor<128x512xf32>
    %v982 = stablehlo.multiply %v979, %v981 : tensor<128x512xf32>
    %v983 = stablehlo.add %v980, %v982 : tensor<128x512xf32>
    %v984 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v985 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v986 = stablehlo.divide %v977, %v984 : tensor<128x512xf32>
    %v987 = stablehlo.divide %v983, %v985 : tensor<128x512xf32>
    %v988 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v989 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v990 = stablehlo.sqrt %v987 : tensor<128x512xf32>
    %v991 = stablehlo.add %v990, %v989 : tensor<128x512xf32>
    %v992 = stablehlo.divide %v986, %v991 : tensor<128x512xf32>
    %v993 = stablehlo.multiply %v988, %v992 : tensor<128x512xf32>
    %v994 = stablehlo.subtract %W9, %v993 : tensor<128x512xf32>
    %v995 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v996 = stablehlo.multiply %v995, %v988 : tensor<128x512xf32>
    %v997 = stablehlo.multiply %v996, %W9 : tensor<128x512xf32>
    %v998 = stablehlo.subtract %v994, %v997 : tensor<128x512xf32>
    %v999 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v1000 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v1001 = stablehlo.multiply %v999, %W9m : tensor<128x512xf32>
    %v1002 = stablehlo.multiply %v1000, %v972 : tensor<128x512xf32>
    %v1003 = stablehlo.add %v1001, %v1002 : tensor<128x512xf32>
    %v1004 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v1005 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v1006 = stablehlo.multiply %v1004, %W9v : tensor<128x512xf32>
    %v1007 = stablehlo.multiply %v972, %v972 : tensor<128x512xf32>
    %v1008 = stablehlo.multiply %v1005, %v1007 : tensor<128x512xf32>
    %v1009 = stablehlo.add %v1006, %v1008 : tensor<128x512xf32>
    %v1010 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1011 = stablehlo.reduce(%v156 init: %v1010) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v1012 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1013 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1014 = stablehlo.multiply %v1012, %b9m : tensor<512xf32>
    %v1015 = stablehlo.multiply %v1013, %v1011 : tensor<512xf32>
    %v1016 = stablehlo.add %v1014, %v1015 : tensor<512xf32>
    %v1017 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1018 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1019 = stablehlo.multiply %v1017, %b9v : tensor<512xf32>
    %v1020 = stablehlo.multiply %v1011, %v1011 : tensor<512xf32>
    %v1021 = stablehlo.multiply %v1018, %v1020 : tensor<512xf32>
    %v1022 = stablehlo.add %v1019, %v1021 : tensor<512xf32>
    %v1023 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1024 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1025 = stablehlo.divide %v1016, %v1023 : tensor<512xf32>
    %v1026 = stablehlo.divide %v1022, %v1024 : tensor<512xf32>
    %v1027 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1028 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1029 = stablehlo.sqrt %v1026 : tensor<512xf32>
    %v1030 = stablehlo.add %v1029, %v1028 : tensor<512xf32>
    %v1031 = stablehlo.divide %v1025, %v1030 : tensor<512xf32>
    %v1032 = stablehlo.multiply %v1027, %v1031 : tensor<512xf32>
    %v1033 = stablehlo.subtract %b9, %v1032 : tensor<512xf32>
    %v1034 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1035 = stablehlo.multiply %v1034, %v1027 : tensor<512xf32>
    %v1036 = stablehlo.multiply %v1035, %b9 : tensor<512xf32>
    %v1037 = stablehlo.subtract %v1033, %v1036 : tensor<512xf32>
    %v1038 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1039 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1040 = stablehlo.multiply %v1038, %b9m : tensor<512xf32>
    %v1041 = stablehlo.multiply %v1039, %v1011 : tensor<512xf32>
    %v1042 = stablehlo.add %v1040, %v1041 : tensor<512xf32>
    %v1043 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1044 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1045 = stablehlo.multiply %v1043, %b9v : tensor<512xf32>
    %v1046 = stablehlo.multiply %v1011, %v1011 : tensor<512xf32>
    %v1047 = stablehlo.multiply %v1044, %v1046 : tensor<512xf32>
    %v1048 = stablehlo.add %v1045, %v1047 : tensor<512xf32>
    %v1049 = stablehlo.dot_general %v118, %v147, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<512x512xf32>
    %v1050 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1051 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1052 = stablehlo.multiply %v1050, %Wam : tensor<512x512xf32>
    %v1053 = stablehlo.multiply %v1051, %v1049 : tensor<512x512xf32>
    %v1054 = stablehlo.add %v1052, %v1053 : tensor<512x512xf32>
    %v1055 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1056 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1057 = stablehlo.multiply %v1055, %Wav : tensor<512x512xf32>
    %v1058 = stablehlo.multiply %v1049, %v1049 : tensor<512x512xf32>
    %v1059 = stablehlo.multiply %v1056, %v1058 : tensor<512x512xf32>
    %v1060 = stablehlo.add %v1057, %v1059 : tensor<512x512xf32>
    %v1061 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1062 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1063 = stablehlo.divide %v1054, %v1061 : tensor<512x512xf32>
    %v1064 = stablehlo.divide %v1060, %v1062 : tensor<512x512xf32>
    %v1065 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1066 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1067 = stablehlo.sqrt %v1064 : tensor<512x512xf32>
    %v1068 = stablehlo.add %v1067, %v1066 : tensor<512x512xf32>
    %v1069 = stablehlo.divide %v1063, %v1068 : tensor<512x512xf32>
    %v1070 = stablehlo.multiply %v1065, %v1069 : tensor<512x512xf32>
    %v1071 = stablehlo.subtract %Wa, %v1070 : tensor<512x512xf32>
    %v1072 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1073 = stablehlo.multiply %v1072, %v1065 : tensor<512x512xf32>
    %v1074 = stablehlo.multiply %v1073, %Wa : tensor<512x512xf32>
    %v1075 = stablehlo.subtract %v1071, %v1074 : tensor<512x512xf32>
    %v1076 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1077 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1078 = stablehlo.multiply %v1076, %Wam : tensor<512x512xf32>
    %v1079 = stablehlo.multiply %v1077, %v1049 : tensor<512x512xf32>
    %v1080 = stablehlo.add %v1078, %v1079 : tensor<512x512xf32>
    %v1081 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1082 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1083 = stablehlo.multiply %v1081, %Wav : tensor<512x512xf32>
    %v1084 = stablehlo.multiply %v1049, %v1049 : tensor<512x512xf32>
    %v1085 = stablehlo.multiply %v1082, %v1084 : tensor<512x512xf32>
    %v1086 = stablehlo.add %v1083, %v1085 : tensor<512x512xf32>
    %v1087 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1088 = stablehlo.reduce(%v147 init: %v1087) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v1089 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1090 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1091 = stablehlo.multiply %v1089, %bam : tensor<512xf32>
    %v1092 = stablehlo.multiply %v1090, %v1088 : tensor<512xf32>
    %v1093 = stablehlo.add %v1091, %v1092 : tensor<512xf32>
    %v1094 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1095 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1096 = stablehlo.multiply %v1094, %bav : tensor<512xf32>
    %v1097 = stablehlo.multiply %v1088, %v1088 : tensor<512xf32>
    %v1098 = stablehlo.multiply %v1095, %v1097 : tensor<512xf32>
    %v1099 = stablehlo.add %v1096, %v1098 : tensor<512xf32>
    %v1100 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1101 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1102 = stablehlo.divide %v1093, %v1100 : tensor<512xf32>
    %v1103 = stablehlo.divide %v1099, %v1101 : tensor<512xf32>
    %v1104 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1105 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1106 = stablehlo.sqrt %v1103 : tensor<512xf32>
    %v1107 = stablehlo.add %v1106, %v1105 : tensor<512xf32>
    %v1108 = stablehlo.divide %v1102, %v1107 : tensor<512xf32>
    %v1109 = stablehlo.multiply %v1104, %v1108 : tensor<512xf32>
    %v1110 = stablehlo.subtract %ba, %v1109 : tensor<512xf32>
    %v1111 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1112 = stablehlo.multiply %v1111, %v1104 : tensor<512xf32>
    %v1113 = stablehlo.multiply %v1112, %ba : tensor<512xf32>
    %v1114 = stablehlo.subtract %v1110, %v1113 : tensor<512xf32>
    %v1115 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1116 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1117 = stablehlo.multiply %v1115, %bam : tensor<512xf32>
    %v1118 = stablehlo.multiply %v1116, %v1088 : tensor<512xf32>
    %v1119 = stablehlo.add %v1117, %v1118 : tensor<512xf32>
    %v1120 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1121 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1122 = stablehlo.multiply %v1120, %bav : tensor<512xf32>
    %v1123 = stablehlo.multiply %v1088, %v1088 : tensor<512xf32>
    %v1124 = stablehlo.multiply %v1121, %v1123 : tensor<512xf32>
    %v1125 = stablehlo.add %v1122, %v1124 : tensor<512xf32>
    %v1126 = stablehlo.dot_general %v125, %v138, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x10xf32>) -> tensor<512x10xf32>
    %v1127 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1128 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1129 = stablehlo.multiply %v1127, %Wbm : tensor<512x10xf32>
    %v1130 = stablehlo.multiply %v1128, %v1126 : tensor<512x10xf32>
    %v1131 = stablehlo.add %v1129, %v1130 : tensor<512x10xf32>
    %v1132 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1133 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1134 = stablehlo.multiply %v1132, %Wbv : tensor<512x10xf32>
    %v1135 = stablehlo.multiply %v1126, %v1126 : tensor<512x10xf32>
    %v1136 = stablehlo.multiply %v1133, %v1135 : tensor<512x10xf32>
    %v1137 = stablehlo.add %v1134, %v1136 : tensor<512x10xf32>
    %v1138 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1139 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1140 = stablehlo.divide %v1131, %v1138 : tensor<512x10xf32>
    %v1141 = stablehlo.divide %v1137, %v1139 : tensor<512x10xf32>
    %v1142 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1143 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1144 = stablehlo.sqrt %v1141 : tensor<512x10xf32>
    %v1145 = stablehlo.add %v1144, %v1143 : tensor<512x10xf32>
    %v1146 = stablehlo.divide %v1140, %v1145 : tensor<512x10xf32>
    %v1147 = stablehlo.multiply %v1142, %v1146 : tensor<512x10xf32>
    %v1148 = stablehlo.subtract %Wb, %v1147 : tensor<512x10xf32>
    %v1149 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1150 = stablehlo.multiply %v1149, %v1142 : tensor<512x10xf32>
    %v1151 = stablehlo.multiply %v1150, %Wb : tensor<512x10xf32>
    %v1152 = stablehlo.subtract %v1148, %v1151 : tensor<512x10xf32>
    %v1153 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1154 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1155 = stablehlo.multiply %v1153, %Wbm : tensor<512x10xf32>
    %v1156 = stablehlo.multiply %v1154, %v1126 : tensor<512x10xf32>
    %v1157 = stablehlo.add %v1155, %v1156 : tensor<512x10xf32>
    %v1158 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1159 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1160 = stablehlo.multiply %v1158, %Wbv : tensor<512x10xf32>
    %v1161 = stablehlo.multiply %v1126, %v1126 : tensor<512x10xf32>
    %v1162 = stablehlo.multiply %v1159, %v1161 : tensor<512x10xf32>
    %v1163 = stablehlo.add %v1160, %v1162 : tensor<512x10xf32>
    %v1164 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1165 = stablehlo.reduce(%v138 init: %v1164) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1166 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1167 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1168 = stablehlo.multiply %v1166, %bbm : tensor<10xf32>
    %v1169 = stablehlo.multiply %v1167, %v1165 : tensor<10xf32>
    %v1170 = stablehlo.add %v1168, %v1169 : tensor<10xf32>
    %v1171 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1172 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1173 = stablehlo.multiply %v1171, %bbv : tensor<10xf32>
    %v1174 = stablehlo.multiply %v1165, %v1165 : tensor<10xf32>
    %v1175 = stablehlo.multiply %v1172, %v1174 : tensor<10xf32>
    %v1176 = stablehlo.add %v1173, %v1175 : tensor<10xf32>
    %v1177 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1178 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1179 = stablehlo.divide %v1170, %v1177 : tensor<10xf32>
    %v1180 = stablehlo.divide %v1176, %v1178 : tensor<10xf32>
    %v1181 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1182 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1183 = stablehlo.sqrt %v1180 : tensor<10xf32>
    %v1184 = stablehlo.add %v1183, %v1182 : tensor<10xf32>
    %v1185 = stablehlo.divide %v1179, %v1184 : tensor<10xf32>
    %v1186 = stablehlo.multiply %v1181, %v1185 : tensor<10xf32>
    %v1187 = stablehlo.subtract %bb, %v1186 : tensor<10xf32>
    %v1188 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1189 = stablehlo.multiply %v1188, %v1181 : tensor<10xf32>
    %v1190 = stablehlo.multiply %v1189, %bb : tensor<10xf32>
    %v1191 = stablehlo.subtract %v1187, %v1190 : tensor<10xf32>
    %v1192 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1193 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1194 = stablehlo.multiply %v1192, %bbm : tensor<10xf32>
    %v1195 = stablehlo.multiply %v1193, %v1165 : tensor<10xf32>
    %v1196 = stablehlo.add %v1194, %v1195 : tensor<10xf32>
    %v1197 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1198 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1199 = stablehlo.multiply %v1197, %bbv : tensor<10xf32>
    %v1200 = stablehlo.multiply %v1165, %v1165 : tensor<10xf32>
    %v1201 = stablehlo.multiply %v1198, %v1200 : tensor<10xf32>
    %v1202 = stablehlo.add %v1199, %v1201 : tensor<10xf32>
    return %v318, %v358, %v404, %v444, %v490, %v530, %v576, %v616, %v662, %v702, %v748, %v788, %v834, %v874, %v920, %v960, %v998, %v1037, %v1075, %v1114, %v1152, %v1191, %v323, %v363, %v409, %v449, %v495, %v535, %v581, %v621, %v667, %v707, %v753, %v793, %v839, %v879, %v925, %v965, %v1003, %v1042, %v1080, %v1119, %v1157, %v1196, %v329, %v369, %v415, %v455, %v501, %v541, %v587, %v627, %v673, %v713, %v759, %v799, %v845, %v885, %v931, %v971, %v1009, %v1048, %v1086, %v1125, %v1163, %v1202, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
