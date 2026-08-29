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
    %v283 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v284 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v285 = stablehlo.multiply %v283, %W1m : tensor<16x3x3x3xf32>
    %v286 = stablehlo.multiply %v284, %v282 : tensor<16x3x3x3xf32>
    %v287 = stablehlo.add %v285, %v286 : tensor<16x3x3x3xf32>
    %v288 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v289 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v290 = stablehlo.multiply %v288, %W1v : tensor<16x3x3x3xf32>
    %v291 = stablehlo.multiply %v282, %v282 : tensor<16x3x3x3xf32>
    %v292 = stablehlo.multiply %v289, %v291 : tensor<16x3x3x3xf32>
    %v293 = stablehlo.add %v290, %v292 : tensor<16x3x3x3xf32>
    %v294 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v295 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v296 = stablehlo.divide %v287, %v294 : tensor<16x3x3x3xf32>
    %v297 = stablehlo.divide %v293, %v295 : tensor<16x3x3x3xf32>
    %v298 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v299 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v300 = stablehlo.sqrt %v297 : tensor<16x3x3x3xf32>
    %v301 = stablehlo.add %v300, %v299 : tensor<16x3x3x3xf32>
    %v302 = stablehlo.divide %v296, %v301 : tensor<16x3x3x3xf32>
    %v303 = stablehlo.multiply %v298, %v302 : tensor<16x3x3x3xf32>
    %v304 = stablehlo.subtract %W1, %v303 : tensor<16x3x3x3xf32>
    %v305 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v306 = stablehlo.multiply %v305, %v298 : tensor<16x3x3x3xf32>
    %v307 = stablehlo.multiply %v306, %W1 : tensor<16x3x3x3xf32>
    %v308 = stablehlo.subtract %v304, %v307 : tensor<16x3x3x3xf32>
    %v309 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v310 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v311 = stablehlo.multiply %v309, %W1m : tensor<16x3x3x3xf32>
    %v312 = stablehlo.multiply %v310, %v282 : tensor<16x3x3x3xf32>
    %v313 = stablehlo.add %v311, %v312 : tensor<16x3x3x3xf32>
    %v314 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v315 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v316 = stablehlo.multiply %v314, %W1v : tensor<16x3x3x3xf32>
    %v317 = stablehlo.multiply %v282, %v282 : tensor<16x3x3x3xf32>
    %v318 = stablehlo.multiply %v315, %v317 : tensor<16x3x3x3xf32>
    %v319 = stablehlo.add %v316, %v318 : tensor<16x3x3x3xf32>
    %v320 = stablehlo.reshape %v273 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v321 = stablehlo.constant dense<0.0> : tensor<f32>
    %v322 = stablehlo.reduce(%v320 init: %v321) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v323 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v324 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v325 = stablehlo.multiply %v323, %cb1m : tensor<16xf32>
    %v326 = stablehlo.multiply %v324, %v322 : tensor<16xf32>
    %v327 = stablehlo.add %v325, %v326 : tensor<16xf32>
    %v328 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v329 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v330 = stablehlo.multiply %v328, %cb1v : tensor<16xf32>
    %v331 = stablehlo.multiply %v322, %v322 : tensor<16xf32>
    %v332 = stablehlo.multiply %v329, %v331 : tensor<16xf32>
    %v333 = stablehlo.add %v330, %v332 : tensor<16xf32>
    %v334 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v335 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v336 = stablehlo.divide %v327, %v334 : tensor<16xf32>
    %v337 = stablehlo.divide %v333, %v335 : tensor<16xf32>
    %v338 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v339 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v340 = stablehlo.sqrt %v337 : tensor<16xf32>
    %v341 = stablehlo.add %v340, %v339 : tensor<16xf32>
    %v342 = stablehlo.divide %v336, %v341 : tensor<16xf32>
    %v343 = stablehlo.multiply %v338, %v342 : tensor<16xf32>
    %v344 = stablehlo.subtract %cb1, %v343 : tensor<16xf32>
    %v345 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v346 = stablehlo.multiply %v345, %v338 : tensor<16xf32>
    %v347 = stablehlo.multiply %v346, %cb1 : tensor<16xf32>
    %v348 = stablehlo.subtract %v344, %v347 : tensor<16xf32>
    %v349 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v350 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v351 = stablehlo.multiply %v349, %cb1m : tensor<16xf32>
    %v352 = stablehlo.multiply %v350, %v322 : tensor<16xf32>
    %v353 = stablehlo.add %v351, %v352 : tensor<16xf32>
    %v354 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v355 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v356 = stablehlo.multiply %v354, %cb1v : tensor<16xf32>
    %v357 = stablehlo.multiply %v322, %v322 : tensor<16xf32>
    %v358 = stablehlo.multiply %v355, %v357 : tensor<16xf32>
    %v359 = stablehlo.add %v356, %v358 : tensor<16xf32>
    %v360 = stablehlo.reshape %v11 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v361 = stablehlo.reshape %v259 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v362 = stablehlo.transpose %v360, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v363 = stablehlo.transpose %v361, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v364 = stablehlo.convert %v362 : (tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xbf16>
    %v365 = stablehlo.convert %v363 : (tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xbf16>
    %v366 = stablehlo.convolution(%v364, %v365)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xbf16>, tensor<16x128x32x32xbf16>) -> tensor<16x16x3x3xbf16>
    %v367 = stablehlo.convert %v366 : (tensor<16x16x3x3xbf16>) -> tensor<16x16x3x3xf32>
    %v368 = stablehlo.transpose %v367, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v369 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v370 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v371 = stablehlo.multiply %v369, %W2m : tensor<16x16x3x3xf32>
    %v372 = stablehlo.multiply %v370, %v368 : tensor<16x16x3x3xf32>
    %v373 = stablehlo.add %v371, %v372 : tensor<16x16x3x3xf32>
    %v374 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v375 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v376 = stablehlo.multiply %v374, %W2v : tensor<16x16x3x3xf32>
    %v377 = stablehlo.multiply %v368, %v368 : tensor<16x16x3x3xf32>
    %v378 = stablehlo.multiply %v375, %v377 : tensor<16x16x3x3xf32>
    %v379 = stablehlo.add %v376, %v378 : tensor<16x16x3x3xf32>
    %v380 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v381 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v382 = stablehlo.divide %v373, %v380 : tensor<16x16x3x3xf32>
    %v383 = stablehlo.divide %v379, %v381 : tensor<16x16x3x3xf32>
    %v384 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v385 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v386 = stablehlo.sqrt %v383 : tensor<16x16x3x3xf32>
    %v387 = stablehlo.add %v386, %v385 : tensor<16x16x3x3xf32>
    %v388 = stablehlo.divide %v382, %v387 : tensor<16x16x3x3xf32>
    %v389 = stablehlo.multiply %v384, %v388 : tensor<16x16x3x3xf32>
    %v390 = stablehlo.subtract %W2, %v389 : tensor<16x16x3x3xf32>
    %v391 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v392 = stablehlo.multiply %v391, %v384 : tensor<16x16x3x3xf32>
    %v393 = stablehlo.multiply %v392, %W2 : tensor<16x16x3x3xf32>
    %v394 = stablehlo.subtract %v390, %v393 : tensor<16x16x3x3xf32>
    %v395 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v396 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v397 = stablehlo.multiply %v395, %W2m : tensor<16x16x3x3xf32>
    %v398 = stablehlo.multiply %v396, %v368 : tensor<16x16x3x3xf32>
    %v399 = stablehlo.add %v397, %v398 : tensor<16x16x3x3xf32>
    %v400 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v401 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v402 = stablehlo.multiply %v400, %W2v : tensor<16x16x3x3xf32>
    %v403 = stablehlo.multiply %v368, %v368 : tensor<16x16x3x3xf32>
    %v404 = stablehlo.multiply %v401, %v403 : tensor<16x16x3x3xf32>
    %v405 = stablehlo.add %v402, %v404 : tensor<16x16x3x3xf32>
    %v406 = stablehlo.reshape %v259 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v407 = stablehlo.constant dense<0.0> : tensor<f32>
    %v408 = stablehlo.reduce(%v406 init: %v407) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v409 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v410 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v411 = stablehlo.multiply %v409, %cb2m : tensor<16xf32>
    %v412 = stablehlo.multiply %v410, %v408 : tensor<16xf32>
    %v413 = stablehlo.add %v411, %v412 : tensor<16xf32>
    %v414 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v415 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v416 = stablehlo.multiply %v414, %cb2v : tensor<16xf32>
    %v417 = stablehlo.multiply %v408, %v408 : tensor<16xf32>
    %v418 = stablehlo.multiply %v415, %v417 : tensor<16xf32>
    %v419 = stablehlo.add %v416, %v418 : tensor<16xf32>
    %v420 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v421 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v422 = stablehlo.divide %v413, %v420 : tensor<16xf32>
    %v423 = stablehlo.divide %v419, %v421 : tensor<16xf32>
    %v424 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v425 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v426 = stablehlo.sqrt %v423 : tensor<16xf32>
    %v427 = stablehlo.add %v426, %v425 : tensor<16xf32>
    %v428 = stablehlo.divide %v422, %v427 : tensor<16xf32>
    %v429 = stablehlo.multiply %v424, %v428 : tensor<16xf32>
    %v430 = stablehlo.subtract %cb2, %v429 : tensor<16xf32>
    %v431 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v432 = stablehlo.multiply %v431, %v424 : tensor<16xf32>
    %v433 = stablehlo.multiply %v432, %cb2 : tensor<16xf32>
    %v434 = stablehlo.subtract %v430, %v433 : tensor<16xf32>
    %v435 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v436 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v437 = stablehlo.multiply %v435, %cb2m : tensor<16xf32>
    %v438 = stablehlo.multiply %v436, %v408 : tensor<16xf32>
    %v439 = stablehlo.add %v437, %v438 : tensor<16xf32>
    %v440 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v441 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v442 = stablehlo.multiply %v440, %cb2v : tensor<16xf32>
    %v443 = stablehlo.multiply %v408, %v408 : tensor<16xf32>
    %v444 = stablehlo.multiply %v441, %v443 : tensor<16xf32>
    %v445 = stablehlo.add %v442, %v444 : tensor<16xf32>
    %v446 = stablehlo.reshape %v27 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v447 = stablehlo.reshape %v240 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v448 = stablehlo.transpose %v446, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v449 = stablehlo.transpose %v447, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v450 = stablehlo.convert %v448 : (tensor<16x128x16x16xf32>) -> tensor<16x128x16x16xbf16>
    %v451 = stablehlo.convert %v449 : (tensor<16x128x16x16xf32>) -> tensor<16x128x16x16xbf16>
    %v452 = stablehlo.convolution(%v450, %v451)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xbf16>, tensor<16x128x16x16xbf16>) -> tensor<16x16x3x3xbf16>
    %v453 = stablehlo.convert %v452 : (tensor<16x16x3x3xbf16>) -> tensor<16x16x3x3xf32>
    %v454 = stablehlo.transpose %v453, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v455 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v456 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v457 = stablehlo.multiply %v455, %W3m : tensor<16x16x3x3xf32>
    %v458 = stablehlo.multiply %v456, %v454 : tensor<16x16x3x3xf32>
    %v459 = stablehlo.add %v457, %v458 : tensor<16x16x3x3xf32>
    %v460 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v461 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v462 = stablehlo.multiply %v460, %W3v : tensor<16x16x3x3xf32>
    %v463 = stablehlo.multiply %v454, %v454 : tensor<16x16x3x3xf32>
    %v464 = stablehlo.multiply %v461, %v463 : tensor<16x16x3x3xf32>
    %v465 = stablehlo.add %v462, %v464 : tensor<16x16x3x3xf32>
    %v466 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v467 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v468 = stablehlo.divide %v459, %v466 : tensor<16x16x3x3xf32>
    %v469 = stablehlo.divide %v465, %v467 : tensor<16x16x3x3xf32>
    %v470 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v471 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v472 = stablehlo.sqrt %v469 : tensor<16x16x3x3xf32>
    %v473 = stablehlo.add %v472, %v471 : tensor<16x16x3x3xf32>
    %v474 = stablehlo.divide %v468, %v473 : tensor<16x16x3x3xf32>
    %v475 = stablehlo.multiply %v470, %v474 : tensor<16x16x3x3xf32>
    %v476 = stablehlo.subtract %W3, %v475 : tensor<16x16x3x3xf32>
    %v477 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v478 = stablehlo.multiply %v477, %v470 : tensor<16x16x3x3xf32>
    %v479 = stablehlo.multiply %v478, %W3 : tensor<16x16x3x3xf32>
    %v480 = stablehlo.subtract %v476, %v479 : tensor<16x16x3x3xf32>
    %v481 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v482 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v483 = stablehlo.multiply %v481, %W3m : tensor<16x16x3x3xf32>
    %v484 = stablehlo.multiply %v482, %v454 : tensor<16x16x3x3xf32>
    %v485 = stablehlo.add %v483, %v484 : tensor<16x16x3x3xf32>
    %v486 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v487 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v488 = stablehlo.multiply %v486, %W3v : tensor<16x16x3x3xf32>
    %v489 = stablehlo.multiply %v454, %v454 : tensor<16x16x3x3xf32>
    %v490 = stablehlo.multiply %v487, %v489 : tensor<16x16x3x3xf32>
    %v491 = stablehlo.add %v488, %v490 : tensor<16x16x3x3xf32>
    %v492 = stablehlo.reshape %v240 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v493 = stablehlo.constant dense<0.0> : tensor<f32>
    %v494 = stablehlo.reduce(%v492 init: %v493) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v495 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v496 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v497 = stablehlo.multiply %v495, %cb3m : tensor<16xf32>
    %v498 = stablehlo.multiply %v496, %v494 : tensor<16xf32>
    %v499 = stablehlo.add %v497, %v498 : tensor<16xf32>
    %v500 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v501 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v502 = stablehlo.multiply %v500, %cb3v : tensor<16xf32>
    %v503 = stablehlo.multiply %v494, %v494 : tensor<16xf32>
    %v504 = stablehlo.multiply %v501, %v503 : tensor<16xf32>
    %v505 = stablehlo.add %v502, %v504 : tensor<16xf32>
    %v506 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v507 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v508 = stablehlo.divide %v499, %v506 : tensor<16xf32>
    %v509 = stablehlo.divide %v505, %v507 : tensor<16xf32>
    %v510 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v511 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v512 = stablehlo.sqrt %v509 : tensor<16xf32>
    %v513 = stablehlo.add %v512, %v511 : tensor<16xf32>
    %v514 = stablehlo.divide %v508, %v513 : tensor<16xf32>
    %v515 = stablehlo.multiply %v510, %v514 : tensor<16xf32>
    %v516 = stablehlo.subtract %cb3, %v515 : tensor<16xf32>
    %v517 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v518 = stablehlo.multiply %v517, %v510 : tensor<16xf32>
    %v519 = stablehlo.multiply %v518, %cb3 : tensor<16xf32>
    %v520 = stablehlo.subtract %v516, %v519 : tensor<16xf32>
    %v521 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v522 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v523 = stablehlo.multiply %v521, %cb3m : tensor<16xf32>
    %v524 = stablehlo.multiply %v522, %v494 : tensor<16xf32>
    %v525 = stablehlo.add %v523, %v524 : tensor<16xf32>
    %v526 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v527 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v528 = stablehlo.multiply %v526, %cb3v : tensor<16xf32>
    %v529 = stablehlo.multiply %v494, %v494 : tensor<16xf32>
    %v530 = stablehlo.multiply %v527, %v529 : tensor<16xf32>
    %v531 = stablehlo.add %v528, %v530 : tensor<16xf32>
    %v532 = stablehlo.reshape %v39 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v533 = stablehlo.reshape %v226 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v534 = stablehlo.transpose %v532, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v535 = stablehlo.transpose %v533, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v536 = stablehlo.convert %v534 : (tensor<16x128x16x16xf32>) -> tensor<16x128x16x16xbf16>
    %v537 = stablehlo.convert %v535 : (tensor<16x128x16x16xf32>) -> tensor<16x128x16x16xbf16>
    %v538 = stablehlo.convolution(%v536, %v537)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xbf16>, tensor<16x128x16x16xbf16>) -> tensor<16x16x3x3xbf16>
    %v539 = stablehlo.convert %v538 : (tensor<16x16x3x3xbf16>) -> tensor<16x16x3x3xf32>
    %v540 = stablehlo.transpose %v539, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v541 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v542 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v543 = stablehlo.multiply %v541, %W4m : tensor<16x16x3x3xf32>
    %v544 = stablehlo.multiply %v542, %v540 : tensor<16x16x3x3xf32>
    %v545 = stablehlo.add %v543, %v544 : tensor<16x16x3x3xf32>
    %v546 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v547 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v548 = stablehlo.multiply %v546, %W4v : tensor<16x16x3x3xf32>
    %v549 = stablehlo.multiply %v540, %v540 : tensor<16x16x3x3xf32>
    %v550 = stablehlo.multiply %v547, %v549 : tensor<16x16x3x3xf32>
    %v551 = stablehlo.add %v548, %v550 : tensor<16x16x3x3xf32>
    %v552 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v553 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v554 = stablehlo.divide %v545, %v552 : tensor<16x16x3x3xf32>
    %v555 = stablehlo.divide %v551, %v553 : tensor<16x16x3x3xf32>
    %v556 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v557 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v558 = stablehlo.sqrt %v555 : tensor<16x16x3x3xf32>
    %v559 = stablehlo.add %v558, %v557 : tensor<16x16x3x3xf32>
    %v560 = stablehlo.divide %v554, %v559 : tensor<16x16x3x3xf32>
    %v561 = stablehlo.multiply %v556, %v560 : tensor<16x16x3x3xf32>
    %v562 = stablehlo.subtract %W4, %v561 : tensor<16x16x3x3xf32>
    %v563 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v564 = stablehlo.multiply %v563, %v556 : tensor<16x16x3x3xf32>
    %v565 = stablehlo.multiply %v564, %W4 : tensor<16x16x3x3xf32>
    %v566 = stablehlo.subtract %v562, %v565 : tensor<16x16x3x3xf32>
    %v567 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v568 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v569 = stablehlo.multiply %v567, %W4m : tensor<16x16x3x3xf32>
    %v570 = stablehlo.multiply %v568, %v540 : tensor<16x16x3x3xf32>
    %v571 = stablehlo.add %v569, %v570 : tensor<16x16x3x3xf32>
    %v572 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v573 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v574 = stablehlo.multiply %v572, %W4v : tensor<16x16x3x3xf32>
    %v575 = stablehlo.multiply %v540, %v540 : tensor<16x16x3x3xf32>
    %v576 = stablehlo.multiply %v573, %v575 : tensor<16x16x3x3xf32>
    %v577 = stablehlo.add %v574, %v576 : tensor<16x16x3x3xf32>
    %v578 = stablehlo.reshape %v226 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v579 = stablehlo.constant dense<0.0> : tensor<f32>
    %v580 = stablehlo.reduce(%v578 init: %v579) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v581 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v582 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v583 = stablehlo.multiply %v581, %cb4m : tensor<16xf32>
    %v584 = stablehlo.multiply %v582, %v580 : tensor<16xf32>
    %v585 = stablehlo.add %v583, %v584 : tensor<16xf32>
    %v586 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v587 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v588 = stablehlo.multiply %v586, %cb4v : tensor<16xf32>
    %v589 = stablehlo.multiply %v580, %v580 : tensor<16xf32>
    %v590 = stablehlo.multiply %v587, %v589 : tensor<16xf32>
    %v591 = stablehlo.add %v588, %v590 : tensor<16xf32>
    %v592 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v593 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v594 = stablehlo.divide %v585, %v592 : tensor<16xf32>
    %v595 = stablehlo.divide %v591, %v593 : tensor<16xf32>
    %v596 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v597 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v598 = stablehlo.sqrt %v595 : tensor<16xf32>
    %v599 = stablehlo.add %v598, %v597 : tensor<16xf32>
    %v600 = stablehlo.divide %v594, %v599 : tensor<16xf32>
    %v601 = stablehlo.multiply %v596, %v600 : tensor<16xf32>
    %v602 = stablehlo.subtract %cb4, %v601 : tensor<16xf32>
    %v603 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v604 = stablehlo.multiply %v603, %v596 : tensor<16xf32>
    %v605 = stablehlo.multiply %v604, %cb4 : tensor<16xf32>
    %v606 = stablehlo.subtract %v602, %v605 : tensor<16xf32>
    %v607 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v608 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v609 = stablehlo.multiply %v607, %cb4m : tensor<16xf32>
    %v610 = stablehlo.multiply %v608, %v580 : tensor<16xf32>
    %v611 = stablehlo.add %v609, %v610 : tensor<16xf32>
    %v612 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v613 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v614 = stablehlo.multiply %v612, %cb4v : tensor<16xf32>
    %v615 = stablehlo.multiply %v580, %v580 : tensor<16xf32>
    %v616 = stablehlo.multiply %v613, %v615 : tensor<16xf32>
    %v617 = stablehlo.add %v614, %v616 : tensor<16xf32>
    %v618 = stablehlo.reshape %v55 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v619 = stablehlo.reshape %v207 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v620 = stablehlo.transpose %v618, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v621 = stablehlo.transpose %v619, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v622 = stablehlo.convert %v620 : (tensor<16x128x8x8xf32>) -> tensor<16x128x8x8xbf16>
    %v623 = stablehlo.convert %v621 : (tensor<32x128x8x8xf32>) -> tensor<32x128x8x8xbf16>
    %v624 = stablehlo.convolution(%v622, %v623)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xbf16>, tensor<32x128x8x8xbf16>) -> tensor<16x32x3x3xbf16>
    %v625 = stablehlo.convert %v624 : (tensor<16x32x3x3xbf16>) -> tensor<16x32x3x3xf32>
    %v626 = stablehlo.transpose %v625, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v627 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v628 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v629 = stablehlo.multiply %v627, %W5m : tensor<32x16x3x3xf32>
    %v630 = stablehlo.multiply %v628, %v626 : tensor<32x16x3x3xf32>
    %v631 = stablehlo.add %v629, %v630 : tensor<32x16x3x3xf32>
    %v632 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v633 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v634 = stablehlo.multiply %v632, %W5v : tensor<32x16x3x3xf32>
    %v635 = stablehlo.multiply %v626, %v626 : tensor<32x16x3x3xf32>
    %v636 = stablehlo.multiply %v633, %v635 : tensor<32x16x3x3xf32>
    %v637 = stablehlo.add %v634, %v636 : tensor<32x16x3x3xf32>
    %v638 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v639 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v640 = stablehlo.divide %v631, %v638 : tensor<32x16x3x3xf32>
    %v641 = stablehlo.divide %v637, %v639 : tensor<32x16x3x3xf32>
    %v642 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v643 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v644 = stablehlo.sqrt %v641 : tensor<32x16x3x3xf32>
    %v645 = stablehlo.add %v644, %v643 : tensor<32x16x3x3xf32>
    %v646 = stablehlo.divide %v640, %v645 : tensor<32x16x3x3xf32>
    %v647 = stablehlo.multiply %v642, %v646 : tensor<32x16x3x3xf32>
    %v648 = stablehlo.subtract %W5, %v647 : tensor<32x16x3x3xf32>
    %v649 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v650 = stablehlo.multiply %v649, %v642 : tensor<32x16x3x3xf32>
    %v651 = stablehlo.multiply %v650, %W5 : tensor<32x16x3x3xf32>
    %v652 = stablehlo.subtract %v648, %v651 : tensor<32x16x3x3xf32>
    %v653 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v654 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v655 = stablehlo.multiply %v653, %W5m : tensor<32x16x3x3xf32>
    %v656 = stablehlo.multiply %v654, %v626 : tensor<32x16x3x3xf32>
    %v657 = stablehlo.add %v655, %v656 : tensor<32x16x3x3xf32>
    %v658 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v659 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v660 = stablehlo.multiply %v658, %W5v : tensor<32x16x3x3xf32>
    %v661 = stablehlo.multiply %v626, %v626 : tensor<32x16x3x3xf32>
    %v662 = stablehlo.multiply %v659, %v661 : tensor<32x16x3x3xf32>
    %v663 = stablehlo.add %v660, %v662 : tensor<32x16x3x3xf32>
    %v664 = stablehlo.reshape %v207 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v665 = stablehlo.constant dense<0.0> : tensor<f32>
    %v666 = stablehlo.reduce(%v664 init: %v665) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v667 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v668 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v669 = stablehlo.multiply %v667, %cb5m : tensor<32xf32>
    %v670 = stablehlo.multiply %v668, %v666 : tensor<32xf32>
    %v671 = stablehlo.add %v669, %v670 : tensor<32xf32>
    %v672 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v673 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v674 = stablehlo.multiply %v672, %cb5v : tensor<32xf32>
    %v675 = stablehlo.multiply %v666, %v666 : tensor<32xf32>
    %v676 = stablehlo.multiply %v673, %v675 : tensor<32xf32>
    %v677 = stablehlo.add %v674, %v676 : tensor<32xf32>
    %v678 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v679 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v680 = stablehlo.divide %v671, %v678 : tensor<32xf32>
    %v681 = stablehlo.divide %v677, %v679 : tensor<32xf32>
    %v682 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v683 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v684 = stablehlo.sqrt %v681 : tensor<32xf32>
    %v685 = stablehlo.add %v684, %v683 : tensor<32xf32>
    %v686 = stablehlo.divide %v680, %v685 : tensor<32xf32>
    %v687 = stablehlo.multiply %v682, %v686 : tensor<32xf32>
    %v688 = stablehlo.subtract %cb5, %v687 : tensor<32xf32>
    %v689 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v690 = stablehlo.multiply %v689, %v682 : tensor<32xf32>
    %v691 = stablehlo.multiply %v690, %cb5 : tensor<32xf32>
    %v692 = stablehlo.subtract %v688, %v691 : tensor<32xf32>
    %v693 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v694 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v695 = stablehlo.multiply %v693, %cb5m : tensor<32xf32>
    %v696 = stablehlo.multiply %v694, %v666 : tensor<32xf32>
    %v697 = stablehlo.add %v695, %v696 : tensor<32xf32>
    %v698 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v699 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v700 = stablehlo.multiply %v698, %cb5v : tensor<32xf32>
    %v701 = stablehlo.multiply %v666, %v666 : tensor<32xf32>
    %v702 = stablehlo.multiply %v699, %v701 : tensor<32xf32>
    %v703 = stablehlo.add %v700, %v702 : tensor<32xf32>
    %v704 = stablehlo.reshape %v67 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v705 = stablehlo.reshape %v193 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v706 = stablehlo.transpose %v704, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v707 = stablehlo.transpose %v705, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v708 = stablehlo.convert %v706 : (tensor<32x128x8x8xf32>) -> tensor<32x128x8x8xbf16>
    %v709 = stablehlo.convert %v707 : (tensor<32x128x8x8xf32>) -> tensor<32x128x8x8xbf16>
    %v710 = stablehlo.convolution(%v708, %v709)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xbf16>, tensor<32x128x8x8xbf16>) -> tensor<32x32x3x3xbf16>
    %v711 = stablehlo.convert %v710 : (tensor<32x32x3x3xbf16>) -> tensor<32x32x3x3xf32>
    %v712 = stablehlo.transpose %v711, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v713 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v714 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v715 = stablehlo.multiply %v713, %W6m : tensor<32x32x3x3xf32>
    %v716 = stablehlo.multiply %v714, %v712 : tensor<32x32x3x3xf32>
    %v717 = stablehlo.add %v715, %v716 : tensor<32x32x3x3xf32>
    %v718 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v719 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v720 = stablehlo.multiply %v718, %W6v : tensor<32x32x3x3xf32>
    %v721 = stablehlo.multiply %v712, %v712 : tensor<32x32x3x3xf32>
    %v722 = stablehlo.multiply %v719, %v721 : tensor<32x32x3x3xf32>
    %v723 = stablehlo.add %v720, %v722 : tensor<32x32x3x3xf32>
    %v724 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v725 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v726 = stablehlo.divide %v717, %v724 : tensor<32x32x3x3xf32>
    %v727 = stablehlo.divide %v723, %v725 : tensor<32x32x3x3xf32>
    %v728 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v729 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v730 = stablehlo.sqrt %v727 : tensor<32x32x3x3xf32>
    %v731 = stablehlo.add %v730, %v729 : tensor<32x32x3x3xf32>
    %v732 = stablehlo.divide %v726, %v731 : tensor<32x32x3x3xf32>
    %v733 = stablehlo.multiply %v728, %v732 : tensor<32x32x3x3xf32>
    %v734 = stablehlo.subtract %W6, %v733 : tensor<32x32x3x3xf32>
    %v735 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v736 = stablehlo.multiply %v735, %v728 : tensor<32x32x3x3xf32>
    %v737 = stablehlo.multiply %v736, %W6 : tensor<32x32x3x3xf32>
    %v738 = stablehlo.subtract %v734, %v737 : tensor<32x32x3x3xf32>
    %v739 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v740 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v741 = stablehlo.multiply %v739, %W6m : tensor<32x32x3x3xf32>
    %v742 = stablehlo.multiply %v740, %v712 : tensor<32x32x3x3xf32>
    %v743 = stablehlo.add %v741, %v742 : tensor<32x32x3x3xf32>
    %v744 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v745 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v746 = stablehlo.multiply %v744, %W6v : tensor<32x32x3x3xf32>
    %v747 = stablehlo.multiply %v712, %v712 : tensor<32x32x3x3xf32>
    %v748 = stablehlo.multiply %v745, %v747 : tensor<32x32x3x3xf32>
    %v749 = stablehlo.add %v746, %v748 : tensor<32x32x3x3xf32>
    %v750 = stablehlo.reshape %v193 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v751 = stablehlo.constant dense<0.0> : tensor<f32>
    %v752 = stablehlo.reduce(%v750 init: %v751) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v753 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v754 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v755 = stablehlo.multiply %v753, %cb6m : tensor<32xf32>
    %v756 = stablehlo.multiply %v754, %v752 : tensor<32xf32>
    %v757 = stablehlo.add %v755, %v756 : tensor<32xf32>
    %v758 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v759 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v760 = stablehlo.multiply %v758, %cb6v : tensor<32xf32>
    %v761 = stablehlo.multiply %v752, %v752 : tensor<32xf32>
    %v762 = stablehlo.multiply %v759, %v761 : tensor<32xf32>
    %v763 = stablehlo.add %v760, %v762 : tensor<32xf32>
    %v764 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v765 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v766 = stablehlo.divide %v757, %v764 : tensor<32xf32>
    %v767 = stablehlo.divide %v763, %v765 : tensor<32xf32>
    %v768 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v769 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v770 = stablehlo.sqrt %v767 : tensor<32xf32>
    %v771 = stablehlo.add %v770, %v769 : tensor<32xf32>
    %v772 = stablehlo.divide %v766, %v771 : tensor<32xf32>
    %v773 = stablehlo.multiply %v768, %v772 : tensor<32xf32>
    %v774 = stablehlo.subtract %cb6, %v773 : tensor<32xf32>
    %v775 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v776 = stablehlo.multiply %v775, %v768 : tensor<32xf32>
    %v777 = stablehlo.multiply %v776, %cb6 : tensor<32xf32>
    %v778 = stablehlo.subtract %v774, %v777 : tensor<32xf32>
    %v779 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v780 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v781 = stablehlo.multiply %v779, %cb6m : tensor<32xf32>
    %v782 = stablehlo.multiply %v780, %v752 : tensor<32xf32>
    %v783 = stablehlo.add %v781, %v782 : tensor<32xf32>
    %v784 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v785 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v786 = stablehlo.multiply %v784, %cb6v : tensor<32xf32>
    %v787 = stablehlo.multiply %v752, %v752 : tensor<32xf32>
    %v788 = stablehlo.multiply %v785, %v787 : tensor<32xf32>
    %v789 = stablehlo.add %v786, %v788 : tensor<32xf32>
    %v790 = stablehlo.reshape %v83 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v791 = stablehlo.reshape %v174 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v792 = stablehlo.transpose %v790, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v793 = stablehlo.transpose %v791, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v794 = stablehlo.convert %v792 : (tensor<32x128x4x4xf32>) -> tensor<32x128x4x4xbf16>
    %v795 = stablehlo.convert %v793 : (tensor<32x128x4x4xf32>) -> tensor<32x128x4x4xbf16>
    %v796 = stablehlo.convolution(%v794, %v795)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xbf16>, tensor<32x128x4x4xbf16>) -> tensor<32x32x3x3xbf16>
    %v797 = stablehlo.convert %v796 : (tensor<32x32x3x3xbf16>) -> tensor<32x32x3x3xf32>
    %v798 = stablehlo.transpose %v797, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v799 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v800 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v801 = stablehlo.multiply %v799, %W7m : tensor<32x32x3x3xf32>
    %v802 = stablehlo.multiply %v800, %v798 : tensor<32x32x3x3xf32>
    %v803 = stablehlo.add %v801, %v802 : tensor<32x32x3x3xf32>
    %v804 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v805 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v806 = stablehlo.multiply %v804, %W7v : tensor<32x32x3x3xf32>
    %v807 = stablehlo.multiply %v798, %v798 : tensor<32x32x3x3xf32>
    %v808 = stablehlo.multiply %v805, %v807 : tensor<32x32x3x3xf32>
    %v809 = stablehlo.add %v806, %v808 : tensor<32x32x3x3xf32>
    %v810 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v811 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v812 = stablehlo.divide %v803, %v810 : tensor<32x32x3x3xf32>
    %v813 = stablehlo.divide %v809, %v811 : tensor<32x32x3x3xf32>
    %v814 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v815 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v816 = stablehlo.sqrt %v813 : tensor<32x32x3x3xf32>
    %v817 = stablehlo.add %v816, %v815 : tensor<32x32x3x3xf32>
    %v818 = stablehlo.divide %v812, %v817 : tensor<32x32x3x3xf32>
    %v819 = stablehlo.multiply %v814, %v818 : tensor<32x32x3x3xf32>
    %v820 = stablehlo.subtract %W7, %v819 : tensor<32x32x3x3xf32>
    %v821 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v822 = stablehlo.multiply %v821, %v814 : tensor<32x32x3x3xf32>
    %v823 = stablehlo.multiply %v822, %W7 : tensor<32x32x3x3xf32>
    %v824 = stablehlo.subtract %v820, %v823 : tensor<32x32x3x3xf32>
    %v825 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v826 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v827 = stablehlo.multiply %v825, %W7m : tensor<32x32x3x3xf32>
    %v828 = stablehlo.multiply %v826, %v798 : tensor<32x32x3x3xf32>
    %v829 = stablehlo.add %v827, %v828 : tensor<32x32x3x3xf32>
    %v830 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v831 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v832 = stablehlo.multiply %v830, %W7v : tensor<32x32x3x3xf32>
    %v833 = stablehlo.multiply %v798, %v798 : tensor<32x32x3x3xf32>
    %v834 = stablehlo.multiply %v831, %v833 : tensor<32x32x3x3xf32>
    %v835 = stablehlo.add %v832, %v834 : tensor<32x32x3x3xf32>
    %v836 = stablehlo.reshape %v174 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v837 = stablehlo.constant dense<0.0> : tensor<f32>
    %v838 = stablehlo.reduce(%v836 init: %v837) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v839 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v840 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v841 = stablehlo.multiply %v839, %cb7m : tensor<32xf32>
    %v842 = stablehlo.multiply %v840, %v838 : tensor<32xf32>
    %v843 = stablehlo.add %v841, %v842 : tensor<32xf32>
    %v844 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v845 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v846 = stablehlo.multiply %v844, %cb7v : tensor<32xf32>
    %v847 = stablehlo.multiply %v838, %v838 : tensor<32xf32>
    %v848 = stablehlo.multiply %v845, %v847 : tensor<32xf32>
    %v849 = stablehlo.add %v846, %v848 : tensor<32xf32>
    %v850 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v851 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v852 = stablehlo.divide %v843, %v850 : tensor<32xf32>
    %v853 = stablehlo.divide %v849, %v851 : tensor<32xf32>
    %v854 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v855 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v856 = stablehlo.sqrt %v853 : tensor<32xf32>
    %v857 = stablehlo.add %v856, %v855 : tensor<32xf32>
    %v858 = stablehlo.divide %v852, %v857 : tensor<32xf32>
    %v859 = stablehlo.multiply %v854, %v858 : tensor<32xf32>
    %v860 = stablehlo.subtract %cb7, %v859 : tensor<32xf32>
    %v861 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v862 = stablehlo.multiply %v861, %v854 : tensor<32xf32>
    %v863 = stablehlo.multiply %v862, %cb7 : tensor<32xf32>
    %v864 = stablehlo.subtract %v860, %v863 : tensor<32xf32>
    %v865 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v866 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v867 = stablehlo.multiply %v865, %cb7m : tensor<32xf32>
    %v868 = stablehlo.multiply %v866, %v838 : tensor<32xf32>
    %v869 = stablehlo.add %v867, %v868 : tensor<32xf32>
    %v870 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v871 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v872 = stablehlo.multiply %v870, %cb7v : tensor<32xf32>
    %v873 = stablehlo.multiply %v838, %v838 : tensor<32xf32>
    %v874 = stablehlo.multiply %v871, %v873 : tensor<32xf32>
    %v875 = stablehlo.add %v872, %v874 : tensor<32xf32>
    %v876 = stablehlo.reshape %v95 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v877 = stablehlo.reshape %v160 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v878 = stablehlo.transpose %v876, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v879 = stablehlo.transpose %v877, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v880 = stablehlo.convert %v878 : (tensor<32x128x4x4xf32>) -> tensor<32x128x4x4xbf16>
    %v881 = stablehlo.convert %v879 : (tensor<32x128x4x4xf32>) -> tensor<32x128x4x4xbf16>
    %v882 = stablehlo.convolution(%v880, %v881)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xbf16>, tensor<32x128x4x4xbf16>) -> tensor<32x32x3x3xbf16>
    %v883 = stablehlo.convert %v882 : (tensor<32x32x3x3xbf16>) -> tensor<32x32x3x3xf32>
    %v884 = stablehlo.transpose %v883, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v885 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v886 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v887 = stablehlo.multiply %v885, %W8m : tensor<32x32x3x3xf32>
    %v888 = stablehlo.multiply %v886, %v884 : tensor<32x32x3x3xf32>
    %v889 = stablehlo.add %v887, %v888 : tensor<32x32x3x3xf32>
    %v890 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v891 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v892 = stablehlo.multiply %v890, %W8v : tensor<32x32x3x3xf32>
    %v893 = stablehlo.multiply %v884, %v884 : tensor<32x32x3x3xf32>
    %v894 = stablehlo.multiply %v891, %v893 : tensor<32x32x3x3xf32>
    %v895 = stablehlo.add %v892, %v894 : tensor<32x32x3x3xf32>
    %v896 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v897 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v898 = stablehlo.divide %v889, %v896 : tensor<32x32x3x3xf32>
    %v899 = stablehlo.divide %v895, %v897 : tensor<32x32x3x3xf32>
    %v900 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v901 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v902 = stablehlo.sqrt %v899 : tensor<32x32x3x3xf32>
    %v903 = stablehlo.add %v902, %v901 : tensor<32x32x3x3xf32>
    %v904 = stablehlo.divide %v898, %v903 : tensor<32x32x3x3xf32>
    %v905 = stablehlo.multiply %v900, %v904 : tensor<32x32x3x3xf32>
    %v906 = stablehlo.subtract %W8, %v905 : tensor<32x32x3x3xf32>
    %v907 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v908 = stablehlo.multiply %v907, %v900 : tensor<32x32x3x3xf32>
    %v909 = stablehlo.multiply %v908, %W8 : tensor<32x32x3x3xf32>
    %v910 = stablehlo.subtract %v906, %v909 : tensor<32x32x3x3xf32>
    %v911 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v912 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v913 = stablehlo.multiply %v911, %W8m : tensor<32x32x3x3xf32>
    %v914 = stablehlo.multiply %v912, %v884 : tensor<32x32x3x3xf32>
    %v915 = stablehlo.add %v913, %v914 : tensor<32x32x3x3xf32>
    %v916 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v917 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v918 = stablehlo.multiply %v916, %W8v : tensor<32x32x3x3xf32>
    %v919 = stablehlo.multiply %v884, %v884 : tensor<32x32x3x3xf32>
    %v920 = stablehlo.multiply %v917, %v919 : tensor<32x32x3x3xf32>
    %v921 = stablehlo.add %v918, %v920 : tensor<32x32x3x3xf32>
    %v922 = stablehlo.reshape %v160 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v923 = stablehlo.constant dense<0.0> : tensor<f32>
    %v924 = stablehlo.reduce(%v922 init: %v923) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v925 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v926 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v927 = stablehlo.multiply %v925, %cb8m : tensor<32xf32>
    %v928 = stablehlo.multiply %v926, %v924 : tensor<32xf32>
    %v929 = stablehlo.add %v927, %v928 : tensor<32xf32>
    %v930 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v931 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v932 = stablehlo.multiply %v930, %cb8v : tensor<32xf32>
    %v933 = stablehlo.multiply %v924, %v924 : tensor<32xf32>
    %v934 = stablehlo.multiply %v931, %v933 : tensor<32xf32>
    %v935 = stablehlo.add %v932, %v934 : tensor<32xf32>
    %v936 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v937 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v938 = stablehlo.divide %v929, %v936 : tensor<32xf32>
    %v939 = stablehlo.divide %v935, %v937 : tensor<32xf32>
    %v940 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v941 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v942 = stablehlo.sqrt %v939 : tensor<32xf32>
    %v943 = stablehlo.add %v942, %v941 : tensor<32xf32>
    %v944 = stablehlo.divide %v938, %v943 : tensor<32xf32>
    %v945 = stablehlo.multiply %v940, %v944 : tensor<32xf32>
    %v946 = stablehlo.subtract %cb8, %v945 : tensor<32xf32>
    %v947 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v948 = stablehlo.multiply %v947, %v940 : tensor<32xf32>
    %v949 = stablehlo.multiply %v948, %cb8 : tensor<32xf32>
    %v950 = stablehlo.subtract %v946, %v949 : tensor<32xf32>
    %v951 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v952 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v953 = stablehlo.multiply %v951, %cb8m : tensor<32xf32>
    %v954 = stablehlo.multiply %v952, %v924 : tensor<32xf32>
    %v955 = stablehlo.add %v953, %v954 : tensor<32xf32>
    %v956 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v957 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v958 = stablehlo.multiply %v956, %cb8v : tensor<32xf32>
    %v959 = stablehlo.multiply %v924, %v924 : tensor<32xf32>
    %v960 = stablehlo.multiply %v957, %v959 : tensor<32xf32>
    %v961 = stablehlo.add %v958, %v960 : tensor<32xf32>
    %v962 = stablehlo.dot_general %v111, %v146, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x512xf32>) -> tensor<128x512xf32>
    %v963 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v964 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v965 = stablehlo.multiply %v963, %W9m : tensor<128x512xf32>
    %v966 = stablehlo.multiply %v964, %v962 : tensor<128x512xf32>
    %v967 = stablehlo.add %v965, %v966 : tensor<128x512xf32>
    %v968 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v969 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v970 = stablehlo.multiply %v968, %W9v : tensor<128x512xf32>
    %v971 = stablehlo.multiply %v962, %v962 : tensor<128x512xf32>
    %v972 = stablehlo.multiply %v969, %v971 : tensor<128x512xf32>
    %v973 = stablehlo.add %v970, %v972 : tensor<128x512xf32>
    %v974 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v975 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v976 = stablehlo.divide %v967, %v974 : tensor<128x512xf32>
    %v977 = stablehlo.divide %v973, %v975 : tensor<128x512xf32>
    %v978 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v979 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v980 = stablehlo.sqrt %v977 : tensor<128x512xf32>
    %v981 = stablehlo.add %v980, %v979 : tensor<128x512xf32>
    %v982 = stablehlo.divide %v976, %v981 : tensor<128x512xf32>
    %v983 = stablehlo.multiply %v978, %v982 : tensor<128x512xf32>
    %v984 = stablehlo.subtract %W9, %v983 : tensor<128x512xf32>
    %v985 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v986 = stablehlo.multiply %v985, %v978 : tensor<128x512xf32>
    %v987 = stablehlo.multiply %v986, %W9 : tensor<128x512xf32>
    %v988 = stablehlo.subtract %v984, %v987 : tensor<128x512xf32>
    %v989 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v990 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v991 = stablehlo.multiply %v989, %W9m : tensor<128x512xf32>
    %v992 = stablehlo.multiply %v990, %v962 : tensor<128x512xf32>
    %v993 = stablehlo.add %v991, %v992 : tensor<128x512xf32>
    %v994 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v995 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v996 = stablehlo.multiply %v994, %W9v : tensor<128x512xf32>
    %v997 = stablehlo.multiply %v962, %v962 : tensor<128x512xf32>
    %v998 = stablehlo.multiply %v995, %v997 : tensor<128x512xf32>
    %v999 = stablehlo.add %v996, %v998 : tensor<128x512xf32>
    %v1000 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1001 = stablehlo.reduce(%v146 init: %v1000) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v1002 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1003 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1004 = stablehlo.multiply %v1002, %b9m : tensor<512xf32>
    %v1005 = stablehlo.multiply %v1003, %v1001 : tensor<512xf32>
    %v1006 = stablehlo.add %v1004, %v1005 : tensor<512xf32>
    %v1007 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1008 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1009 = stablehlo.multiply %v1007, %b9v : tensor<512xf32>
    %v1010 = stablehlo.multiply %v1001, %v1001 : tensor<512xf32>
    %v1011 = stablehlo.multiply %v1008, %v1010 : tensor<512xf32>
    %v1012 = stablehlo.add %v1009, %v1011 : tensor<512xf32>
    %v1013 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1014 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1015 = stablehlo.divide %v1006, %v1013 : tensor<512xf32>
    %v1016 = stablehlo.divide %v1012, %v1014 : tensor<512xf32>
    %v1017 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1018 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1019 = stablehlo.sqrt %v1016 : tensor<512xf32>
    %v1020 = stablehlo.add %v1019, %v1018 : tensor<512xf32>
    %v1021 = stablehlo.divide %v1015, %v1020 : tensor<512xf32>
    %v1022 = stablehlo.multiply %v1017, %v1021 : tensor<512xf32>
    %v1023 = stablehlo.subtract %b9, %v1022 : tensor<512xf32>
    %v1024 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1025 = stablehlo.multiply %v1024, %v1017 : tensor<512xf32>
    %v1026 = stablehlo.multiply %v1025, %b9 : tensor<512xf32>
    %v1027 = stablehlo.subtract %v1023, %v1026 : tensor<512xf32>
    %v1028 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1029 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1030 = stablehlo.multiply %v1028, %b9m : tensor<512xf32>
    %v1031 = stablehlo.multiply %v1029, %v1001 : tensor<512xf32>
    %v1032 = stablehlo.add %v1030, %v1031 : tensor<512xf32>
    %v1033 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1034 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1035 = stablehlo.multiply %v1033, %b9v : tensor<512xf32>
    %v1036 = stablehlo.multiply %v1001, %v1001 : tensor<512xf32>
    %v1037 = stablehlo.multiply %v1034, %v1036 : tensor<512xf32>
    %v1038 = stablehlo.add %v1035, %v1037 : tensor<512xf32>
    %v1039 = stablehlo.dot_general %v116, %v140, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<512x512xf32>
    %v1040 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1041 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1042 = stablehlo.multiply %v1040, %Wam : tensor<512x512xf32>
    %v1043 = stablehlo.multiply %v1041, %v1039 : tensor<512x512xf32>
    %v1044 = stablehlo.add %v1042, %v1043 : tensor<512x512xf32>
    %v1045 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1046 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1047 = stablehlo.multiply %v1045, %Wav : tensor<512x512xf32>
    %v1048 = stablehlo.multiply %v1039, %v1039 : tensor<512x512xf32>
    %v1049 = stablehlo.multiply %v1046, %v1048 : tensor<512x512xf32>
    %v1050 = stablehlo.add %v1047, %v1049 : tensor<512x512xf32>
    %v1051 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1052 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1053 = stablehlo.divide %v1044, %v1051 : tensor<512x512xf32>
    %v1054 = stablehlo.divide %v1050, %v1052 : tensor<512x512xf32>
    %v1055 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1056 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1057 = stablehlo.sqrt %v1054 : tensor<512x512xf32>
    %v1058 = stablehlo.add %v1057, %v1056 : tensor<512x512xf32>
    %v1059 = stablehlo.divide %v1053, %v1058 : tensor<512x512xf32>
    %v1060 = stablehlo.multiply %v1055, %v1059 : tensor<512x512xf32>
    %v1061 = stablehlo.subtract %Wa, %v1060 : tensor<512x512xf32>
    %v1062 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1063 = stablehlo.multiply %v1062, %v1055 : tensor<512x512xf32>
    %v1064 = stablehlo.multiply %v1063, %Wa : tensor<512x512xf32>
    %v1065 = stablehlo.subtract %v1061, %v1064 : tensor<512x512xf32>
    %v1066 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1067 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1068 = stablehlo.multiply %v1066, %Wam : tensor<512x512xf32>
    %v1069 = stablehlo.multiply %v1067, %v1039 : tensor<512x512xf32>
    %v1070 = stablehlo.add %v1068, %v1069 : tensor<512x512xf32>
    %v1071 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1072 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1073 = stablehlo.multiply %v1071, %Wav : tensor<512x512xf32>
    %v1074 = stablehlo.multiply %v1039, %v1039 : tensor<512x512xf32>
    %v1075 = stablehlo.multiply %v1072, %v1074 : tensor<512x512xf32>
    %v1076 = stablehlo.add %v1073, %v1075 : tensor<512x512xf32>
    %v1077 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1078 = stablehlo.reduce(%v140 init: %v1077) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v1079 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1080 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1081 = stablehlo.multiply %v1079, %bam : tensor<512xf32>
    %v1082 = stablehlo.multiply %v1080, %v1078 : tensor<512xf32>
    %v1083 = stablehlo.add %v1081, %v1082 : tensor<512xf32>
    %v1084 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1085 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1086 = stablehlo.multiply %v1084, %bav : tensor<512xf32>
    %v1087 = stablehlo.multiply %v1078, %v1078 : tensor<512xf32>
    %v1088 = stablehlo.multiply %v1085, %v1087 : tensor<512xf32>
    %v1089 = stablehlo.add %v1086, %v1088 : tensor<512xf32>
    %v1090 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1091 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1092 = stablehlo.divide %v1083, %v1090 : tensor<512xf32>
    %v1093 = stablehlo.divide %v1089, %v1091 : tensor<512xf32>
    %v1094 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1095 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1096 = stablehlo.sqrt %v1093 : tensor<512xf32>
    %v1097 = stablehlo.add %v1096, %v1095 : tensor<512xf32>
    %v1098 = stablehlo.divide %v1092, %v1097 : tensor<512xf32>
    %v1099 = stablehlo.multiply %v1094, %v1098 : tensor<512xf32>
    %v1100 = stablehlo.subtract %ba, %v1099 : tensor<512xf32>
    %v1101 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1102 = stablehlo.multiply %v1101, %v1094 : tensor<512xf32>
    %v1103 = stablehlo.multiply %v1102, %ba : tensor<512xf32>
    %v1104 = stablehlo.subtract %v1100, %v1103 : tensor<512xf32>
    %v1105 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1106 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1107 = stablehlo.multiply %v1105, %bam : tensor<512xf32>
    %v1108 = stablehlo.multiply %v1106, %v1078 : tensor<512xf32>
    %v1109 = stablehlo.add %v1107, %v1108 : tensor<512xf32>
    %v1110 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1111 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1112 = stablehlo.multiply %v1110, %bav : tensor<512xf32>
    %v1113 = stablehlo.multiply %v1078, %v1078 : tensor<512xf32>
    %v1114 = stablehlo.multiply %v1111, %v1113 : tensor<512xf32>
    %v1115 = stablehlo.add %v1112, %v1114 : tensor<512xf32>
    %v1116 = stablehlo.dot_general %v121, %v134, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x10xf32>) -> tensor<512x10xf32>
    %v1117 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1118 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1119 = stablehlo.multiply %v1117, %Wbm : tensor<512x10xf32>
    %v1120 = stablehlo.multiply %v1118, %v1116 : tensor<512x10xf32>
    %v1121 = stablehlo.add %v1119, %v1120 : tensor<512x10xf32>
    %v1122 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1123 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1124 = stablehlo.multiply %v1122, %Wbv : tensor<512x10xf32>
    %v1125 = stablehlo.multiply %v1116, %v1116 : tensor<512x10xf32>
    %v1126 = stablehlo.multiply %v1123, %v1125 : tensor<512x10xf32>
    %v1127 = stablehlo.add %v1124, %v1126 : tensor<512x10xf32>
    %v1128 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1129 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1130 = stablehlo.divide %v1121, %v1128 : tensor<512x10xf32>
    %v1131 = stablehlo.divide %v1127, %v1129 : tensor<512x10xf32>
    %v1132 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1133 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1134 = stablehlo.sqrt %v1131 : tensor<512x10xf32>
    %v1135 = stablehlo.add %v1134, %v1133 : tensor<512x10xf32>
    %v1136 = stablehlo.divide %v1130, %v1135 : tensor<512x10xf32>
    %v1137 = stablehlo.multiply %v1132, %v1136 : tensor<512x10xf32>
    %v1138 = stablehlo.subtract %Wb, %v1137 : tensor<512x10xf32>
    %v1139 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1140 = stablehlo.multiply %v1139, %v1132 : tensor<512x10xf32>
    %v1141 = stablehlo.multiply %v1140, %Wb : tensor<512x10xf32>
    %v1142 = stablehlo.subtract %v1138, %v1141 : tensor<512x10xf32>
    %v1143 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1144 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1145 = stablehlo.multiply %v1143, %Wbm : tensor<512x10xf32>
    %v1146 = stablehlo.multiply %v1144, %v1116 : tensor<512x10xf32>
    %v1147 = stablehlo.add %v1145, %v1146 : tensor<512x10xf32>
    %v1148 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1149 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1150 = stablehlo.multiply %v1148, %Wbv : tensor<512x10xf32>
    %v1151 = stablehlo.multiply %v1116, %v1116 : tensor<512x10xf32>
    %v1152 = stablehlo.multiply %v1149, %v1151 : tensor<512x10xf32>
    %v1153 = stablehlo.add %v1150, %v1152 : tensor<512x10xf32>
    %v1154 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1155 = stablehlo.reduce(%v134 init: %v1154) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1156 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1157 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1158 = stablehlo.multiply %v1156, %bbm : tensor<10xf32>
    %v1159 = stablehlo.multiply %v1157, %v1155 : tensor<10xf32>
    %v1160 = stablehlo.add %v1158, %v1159 : tensor<10xf32>
    %v1161 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1162 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1163 = stablehlo.multiply %v1161, %bbv : tensor<10xf32>
    %v1164 = stablehlo.multiply %v1155, %v1155 : tensor<10xf32>
    %v1165 = stablehlo.multiply %v1162, %v1164 : tensor<10xf32>
    %v1166 = stablehlo.add %v1163, %v1165 : tensor<10xf32>
    %v1167 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1168 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1169 = stablehlo.divide %v1160, %v1167 : tensor<10xf32>
    %v1170 = stablehlo.divide %v1166, %v1168 : tensor<10xf32>
    %v1171 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1172 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1173 = stablehlo.sqrt %v1170 : tensor<10xf32>
    %v1174 = stablehlo.add %v1173, %v1172 : tensor<10xf32>
    %v1175 = stablehlo.divide %v1169, %v1174 : tensor<10xf32>
    %v1176 = stablehlo.multiply %v1171, %v1175 : tensor<10xf32>
    %v1177 = stablehlo.subtract %bb, %v1176 : tensor<10xf32>
    %v1178 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1179 = stablehlo.multiply %v1178, %v1171 : tensor<10xf32>
    %v1180 = stablehlo.multiply %v1179, %bb : tensor<10xf32>
    %v1181 = stablehlo.subtract %v1177, %v1180 : tensor<10xf32>
    %v1182 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1183 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1184 = stablehlo.multiply %v1182, %bbm : tensor<10xf32>
    %v1185 = stablehlo.multiply %v1183, %v1155 : tensor<10xf32>
    %v1186 = stablehlo.add %v1184, %v1185 : tensor<10xf32>
    %v1187 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1188 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1189 = stablehlo.multiply %v1187, %bbv : tensor<10xf32>
    %v1190 = stablehlo.multiply %v1155, %v1155 : tensor<10xf32>
    %v1191 = stablehlo.multiply %v1188, %v1190 : tensor<10xf32>
    %v1192 = stablehlo.add %v1189, %v1191 : tensor<10xf32>
    return %v308, %v348, %v394, %v434, %v480, %v520, %v566, %v606, %v652, %v692, %v738, %v778, %v824, %v864, %v910, %v950, %v988, %v1027, %v1065, %v1104, %v1142, %v1181, %v313, %v353, %v399, %v439, %v485, %v525, %v571, %v611, %v657, %v697, %v743, %v783, %v829, %v869, %v915, %v955, %v993, %v1032, %v1070, %v1109, %v1147, %v1186, %v319, %v359, %v405, %v445, %v491, %v531, %v577, %v617, %v663, %v703, %v749, %v789, %v835, %v875, %v921, %v961, %v999, %v1038, %v1076, %v1115, %v1153, %v1192, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
