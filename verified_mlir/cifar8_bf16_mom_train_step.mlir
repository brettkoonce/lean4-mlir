module @m {
  func.func @cifar8_bf16_mom_train_step(%x: tensor<128x3072xf32>, %W1: tensor<16x3x3x3xf32>, %cb1: tensor<16xf32>, %W2: tensor<16x16x3x3xf32>, %cb2: tensor<16xf32>, %W3: tensor<16x16x3x3xf32>, %cb3: tensor<16xf32>, %W4: tensor<16x16x3x3xf32>, %cb4: tensor<16xf32>, %W5: tensor<32x16x3x3xf32>, %cb5: tensor<32xf32>, %W6: tensor<32x32x3x3xf32>, %cb6: tensor<32xf32>, %W7: tensor<32x32x3x3xf32>, %cb7: tensor<32xf32>, %W8: tensor<32x32x3x3xf32>, %cb8: tensor<32xf32>, %W9: tensor<128x64xf32>, %b9: tensor<64xf32>, %Wa: tensor<64x64xf32>, %ba: tensor<64xf32>, %Wb: tensor<64x10xf32>, %bb: tensor<10xf32>, %W1m: tensor<16x3x3x3xf32>, %cb1m: tensor<16xf32>, %W2m: tensor<16x16x3x3xf32>, %cb2m: tensor<16xf32>, %W3m: tensor<16x16x3x3xf32>, %cb3m: tensor<16xf32>, %W4m: tensor<16x16x3x3xf32>, %cb4m: tensor<16xf32>, %W5m: tensor<32x16x3x3xf32>, %cb5m: tensor<32xf32>, %W6m: tensor<32x32x3x3xf32>, %cb6m: tensor<32xf32>, %W7m: tensor<32x32x3x3xf32>, %cb7m: tensor<32xf32>, %W8m: tensor<32x32x3x3xf32>, %cb8m: tensor<32xf32>, %W9m: tensor<128x64xf32>, %b9m: tensor<64xf32>, %Wam: tensor<64x64xf32>, %bam: tensor<64xf32>, %Wbm: tensor<64x10xf32>, %bbm: tensor<10xf32>, %W1v: tensor<16x3x3x3xf32>, %cb1v: tensor<16xf32>, %W2v: tensor<16x16x3x3xf32>, %cb2v: tensor<16xf32>, %W3v: tensor<16x16x3x3xf32>, %cb3v: tensor<16xf32>, %W4v: tensor<16x16x3x3xf32>, %cb4v: tensor<16xf32>, %W5v: tensor<32x16x3x3xf32>, %cb5v: tensor<32xf32>, %W6v: tensor<32x32x3x3xf32>, %cb6v: tensor<32xf32>, %W7v: tensor<32x32x3x3xf32>, %cb7v: tensor<32xf32>, %W8v: tensor<32x32x3x3xf32>, %cb8v: tensor<32xf32>, %W9v: tensor<128x64xf32>, %b9v: tensor<64xf32>, %Wav: tensor<64x64xf32>, %bav: tensor<64xf32>, %Wbv: tensor<64x10xf32>, %bbv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: tensor<128x10xf32>) -> (tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>) {
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
    %v251 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v252 = stablehlo.multiply %v251, %W1v : tensor<16x3x3x3xf32>
    %v253 = stablehlo.add %v252, %v250 : tensor<16x3x3x3xf32>
    %v254 = stablehlo.multiply %v251, %v253 : tensor<16x3x3x3xf32>
    %v255 = stablehlo.add %v254, %v250 : tensor<16x3x3x3xf32>
    %v256 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v257 = stablehlo.multiply %v256, %v255 : tensor<16x3x3x3xf32>
    %v258 = stablehlo.subtract %W1, %v257 : tensor<16x3x3x3xf32>
    %v259 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v260 = stablehlo.multiply %v259, %W1v : tensor<16x3x3x3xf32>
    %v261 = stablehlo.add %v260, %v250 : tensor<16x3x3x3xf32>
    %v262 = stablehlo.reshape %v244 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v263 = stablehlo.constant dense<0.0> : tensor<f32>
    %v264 = stablehlo.reduce(%v262 init: %v263) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v265 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v266 = stablehlo.multiply %v265, %cb1v : tensor<16xf32>
    %v267 = stablehlo.add %v266, %v264 : tensor<16xf32>
    %v268 = stablehlo.multiply %v265, %v267 : tensor<16xf32>
    %v269 = stablehlo.add %v268, %v264 : tensor<16xf32>
    %v270 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v271 = stablehlo.multiply %v270, %v269 : tensor<16xf32>
    %v272 = stablehlo.subtract %cb1, %v271 : tensor<16xf32>
    %v273 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v274 = stablehlo.multiply %v273, %cb1v : tensor<16xf32>
    %v275 = stablehlo.add %v274, %v264 : tensor<16xf32>
    %v276 = stablehlo.reshape %v11 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v277 = stablehlo.reshape %v233 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v278 = stablehlo.transpose %v276, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v279 = stablehlo.transpose %v277, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v280 = stablehlo.convolution(%v278, %v279)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x16x3x3xf32>
    %v281 = stablehlo.transpose %v280, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v282 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v283 = stablehlo.multiply %v282, %W2v : tensor<16x16x3x3xf32>
    %v284 = stablehlo.add %v283, %v281 : tensor<16x16x3x3xf32>
    %v285 = stablehlo.multiply %v282, %v284 : tensor<16x16x3x3xf32>
    %v286 = stablehlo.add %v285, %v281 : tensor<16x16x3x3xf32>
    %v287 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v288 = stablehlo.multiply %v287, %v286 : tensor<16x16x3x3xf32>
    %v289 = stablehlo.subtract %W2, %v288 : tensor<16x16x3x3xf32>
    %v290 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v291 = stablehlo.multiply %v290, %W2v : tensor<16x16x3x3xf32>
    %v292 = stablehlo.add %v291, %v281 : tensor<16x16x3x3xf32>
    %v293 = stablehlo.reshape %v233 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v294 = stablehlo.constant dense<0.0> : tensor<f32>
    %v295 = stablehlo.reduce(%v293 init: %v294) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v296 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v297 = stablehlo.multiply %v296, %cb2v : tensor<16xf32>
    %v298 = stablehlo.add %v297, %v295 : tensor<16xf32>
    %v299 = stablehlo.multiply %v296, %v298 : tensor<16xf32>
    %v300 = stablehlo.add %v299, %v295 : tensor<16xf32>
    %v301 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v302 = stablehlo.multiply %v301, %v300 : tensor<16xf32>
    %v303 = stablehlo.subtract %cb2, %v302 : tensor<16xf32>
    %v304 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v305 = stablehlo.multiply %v304, %cb2v : tensor<16xf32>
    %v306 = stablehlo.add %v305, %v295 : tensor<16xf32>
    %v307 = stablehlo.reshape %v27 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v308 = stablehlo.reshape %v217 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v309 = stablehlo.transpose %v307, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v310 = stablehlo.transpose %v308, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v311 = stablehlo.convolution(%v309, %v310)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v312 = stablehlo.transpose %v311, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v313 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v314 = stablehlo.multiply %v313, %W3v : tensor<16x16x3x3xf32>
    %v315 = stablehlo.add %v314, %v312 : tensor<16x16x3x3xf32>
    %v316 = stablehlo.multiply %v313, %v315 : tensor<16x16x3x3xf32>
    %v317 = stablehlo.add %v316, %v312 : tensor<16x16x3x3xf32>
    %v318 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v319 = stablehlo.multiply %v318, %v317 : tensor<16x16x3x3xf32>
    %v320 = stablehlo.subtract %W3, %v319 : tensor<16x16x3x3xf32>
    %v321 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v322 = stablehlo.multiply %v321, %W3v : tensor<16x16x3x3xf32>
    %v323 = stablehlo.add %v322, %v312 : tensor<16x16x3x3xf32>
    %v324 = stablehlo.reshape %v217 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v325 = stablehlo.constant dense<0.0> : tensor<f32>
    %v326 = stablehlo.reduce(%v324 init: %v325) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v327 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v328 = stablehlo.multiply %v327, %cb3v : tensor<16xf32>
    %v329 = stablehlo.add %v328, %v326 : tensor<16xf32>
    %v330 = stablehlo.multiply %v327, %v329 : tensor<16xf32>
    %v331 = stablehlo.add %v330, %v326 : tensor<16xf32>
    %v332 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v333 = stablehlo.multiply %v332, %v331 : tensor<16xf32>
    %v334 = stablehlo.subtract %cb3, %v333 : tensor<16xf32>
    %v335 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v336 = stablehlo.multiply %v335, %cb3v : tensor<16xf32>
    %v337 = stablehlo.add %v336, %v326 : tensor<16xf32>
    %v338 = stablehlo.reshape %v39 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v339 = stablehlo.reshape %v206 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v340 = stablehlo.transpose %v338, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v341 = stablehlo.transpose %v339, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v342 = stablehlo.convolution(%v340, %v341)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v343 = stablehlo.transpose %v342, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v344 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v345 = stablehlo.multiply %v344, %W4v : tensor<16x16x3x3xf32>
    %v346 = stablehlo.add %v345, %v343 : tensor<16x16x3x3xf32>
    %v347 = stablehlo.multiply %v344, %v346 : tensor<16x16x3x3xf32>
    %v348 = stablehlo.add %v347, %v343 : tensor<16x16x3x3xf32>
    %v349 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v350 = stablehlo.multiply %v349, %v348 : tensor<16x16x3x3xf32>
    %v351 = stablehlo.subtract %W4, %v350 : tensor<16x16x3x3xf32>
    %v352 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v353 = stablehlo.multiply %v352, %W4v : tensor<16x16x3x3xf32>
    %v354 = stablehlo.add %v353, %v343 : tensor<16x16x3x3xf32>
    %v355 = stablehlo.reshape %v206 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v356 = stablehlo.constant dense<0.0> : tensor<f32>
    %v357 = stablehlo.reduce(%v355 init: %v356) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v358 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v359 = stablehlo.multiply %v358, %cb4v : tensor<16xf32>
    %v360 = stablehlo.add %v359, %v357 : tensor<16xf32>
    %v361 = stablehlo.multiply %v358, %v360 : tensor<16xf32>
    %v362 = stablehlo.add %v361, %v357 : tensor<16xf32>
    %v363 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v364 = stablehlo.multiply %v363, %v362 : tensor<16xf32>
    %v365 = stablehlo.subtract %cb4, %v364 : tensor<16xf32>
    %v366 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v367 = stablehlo.multiply %v366, %cb4v : tensor<16xf32>
    %v368 = stablehlo.add %v367, %v357 : tensor<16xf32>
    %v369 = stablehlo.reshape %v55 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v370 = stablehlo.reshape %v190 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v371 = stablehlo.transpose %v369, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v372 = stablehlo.transpose %v370, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v373 = stablehlo.convolution(%v371, %v372)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<16x32x3x3xf32>
    %v374 = stablehlo.transpose %v373, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v375 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v376 = stablehlo.multiply %v375, %W5v : tensor<32x16x3x3xf32>
    %v377 = stablehlo.add %v376, %v374 : tensor<32x16x3x3xf32>
    %v378 = stablehlo.multiply %v375, %v377 : tensor<32x16x3x3xf32>
    %v379 = stablehlo.add %v378, %v374 : tensor<32x16x3x3xf32>
    %v380 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v381 = stablehlo.multiply %v380, %v379 : tensor<32x16x3x3xf32>
    %v382 = stablehlo.subtract %W5, %v381 : tensor<32x16x3x3xf32>
    %v383 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v384 = stablehlo.multiply %v383, %W5v : tensor<32x16x3x3xf32>
    %v385 = stablehlo.add %v384, %v374 : tensor<32x16x3x3xf32>
    %v386 = stablehlo.reshape %v190 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v387 = stablehlo.constant dense<0.0> : tensor<f32>
    %v388 = stablehlo.reduce(%v386 init: %v387) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v389 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v390 = stablehlo.multiply %v389, %cb5v : tensor<32xf32>
    %v391 = stablehlo.add %v390, %v388 : tensor<32xf32>
    %v392 = stablehlo.multiply %v389, %v391 : tensor<32xf32>
    %v393 = stablehlo.add %v392, %v388 : tensor<32xf32>
    %v394 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v395 = stablehlo.multiply %v394, %v393 : tensor<32xf32>
    %v396 = stablehlo.subtract %cb5, %v395 : tensor<32xf32>
    %v397 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v398 = stablehlo.multiply %v397, %cb5v : tensor<32xf32>
    %v399 = stablehlo.add %v398, %v388 : tensor<32xf32>
    %v400 = stablehlo.reshape %v67 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v401 = stablehlo.reshape %v179 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v402 = stablehlo.transpose %v400, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v403 = stablehlo.transpose %v401, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v404 = stablehlo.convolution(%v402, %v403)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<32x32x3x3xf32>
    %v405 = stablehlo.transpose %v404, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v406 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v407 = stablehlo.multiply %v406, %W6v : tensor<32x32x3x3xf32>
    %v408 = stablehlo.add %v407, %v405 : tensor<32x32x3x3xf32>
    %v409 = stablehlo.multiply %v406, %v408 : tensor<32x32x3x3xf32>
    %v410 = stablehlo.add %v409, %v405 : tensor<32x32x3x3xf32>
    %v411 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v412 = stablehlo.multiply %v411, %v410 : tensor<32x32x3x3xf32>
    %v413 = stablehlo.subtract %W6, %v412 : tensor<32x32x3x3xf32>
    %v414 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v415 = stablehlo.multiply %v414, %W6v : tensor<32x32x3x3xf32>
    %v416 = stablehlo.add %v415, %v405 : tensor<32x32x3x3xf32>
    %v417 = stablehlo.reshape %v179 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v418 = stablehlo.constant dense<0.0> : tensor<f32>
    %v419 = stablehlo.reduce(%v417 init: %v418) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v420 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v421 = stablehlo.multiply %v420, %cb6v : tensor<32xf32>
    %v422 = stablehlo.add %v421, %v419 : tensor<32xf32>
    %v423 = stablehlo.multiply %v420, %v422 : tensor<32xf32>
    %v424 = stablehlo.add %v423, %v419 : tensor<32xf32>
    %v425 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v426 = stablehlo.multiply %v425, %v424 : tensor<32xf32>
    %v427 = stablehlo.subtract %cb6, %v426 : tensor<32xf32>
    %v428 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v429 = stablehlo.multiply %v428, %cb6v : tensor<32xf32>
    %v430 = stablehlo.add %v429, %v419 : tensor<32xf32>
    %v431 = stablehlo.reshape %v83 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v432 = stablehlo.reshape %v163 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v433 = stablehlo.transpose %v431, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v434 = stablehlo.transpose %v432, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v435 = stablehlo.convolution(%v433, %v434)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v436 = stablehlo.transpose %v435, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v437 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v438 = stablehlo.multiply %v437, %W7v : tensor<32x32x3x3xf32>
    %v439 = stablehlo.add %v438, %v436 : tensor<32x32x3x3xf32>
    %v440 = stablehlo.multiply %v437, %v439 : tensor<32x32x3x3xf32>
    %v441 = stablehlo.add %v440, %v436 : tensor<32x32x3x3xf32>
    %v442 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v443 = stablehlo.multiply %v442, %v441 : tensor<32x32x3x3xf32>
    %v444 = stablehlo.subtract %W7, %v443 : tensor<32x32x3x3xf32>
    %v445 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v446 = stablehlo.multiply %v445, %W7v : tensor<32x32x3x3xf32>
    %v447 = stablehlo.add %v446, %v436 : tensor<32x32x3x3xf32>
    %v448 = stablehlo.reshape %v163 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v449 = stablehlo.constant dense<0.0> : tensor<f32>
    %v450 = stablehlo.reduce(%v448 init: %v449) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v451 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v452 = stablehlo.multiply %v451, %cb7v : tensor<32xf32>
    %v453 = stablehlo.add %v452, %v450 : tensor<32xf32>
    %v454 = stablehlo.multiply %v451, %v453 : tensor<32xf32>
    %v455 = stablehlo.add %v454, %v450 : tensor<32xf32>
    %v456 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v457 = stablehlo.multiply %v456, %v455 : tensor<32xf32>
    %v458 = stablehlo.subtract %cb7, %v457 : tensor<32xf32>
    %v459 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v460 = stablehlo.multiply %v459, %cb7v : tensor<32xf32>
    %v461 = stablehlo.add %v460, %v450 : tensor<32xf32>
    %v462 = stablehlo.reshape %v95 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v463 = stablehlo.reshape %v152 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v464 = stablehlo.transpose %v462, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v465 = stablehlo.transpose %v463, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v466 = stablehlo.convolution(%v464, %v465)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v467 = stablehlo.transpose %v466, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v468 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v469 = stablehlo.multiply %v468, %W8v : tensor<32x32x3x3xf32>
    %v470 = stablehlo.add %v469, %v467 : tensor<32x32x3x3xf32>
    %v471 = stablehlo.multiply %v468, %v470 : tensor<32x32x3x3xf32>
    %v472 = stablehlo.add %v471, %v467 : tensor<32x32x3x3xf32>
    %v473 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v474 = stablehlo.multiply %v473, %v472 : tensor<32x32x3x3xf32>
    %v475 = stablehlo.subtract %W8, %v474 : tensor<32x32x3x3xf32>
    %v476 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v477 = stablehlo.multiply %v476, %W8v : tensor<32x32x3x3xf32>
    %v478 = stablehlo.add %v477, %v467 : tensor<32x32x3x3xf32>
    %v479 = stablehlo.reshape %v152 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v480 = stablehlo.constant dense<0.0> : tensor<f32>
    %v481 = stablehlo.reduce(%v479 init: %v480) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v482 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v483 = stablehlo.multiply %v482, %cb8v : tensor<32xf32>
    %v484 = stablehlo.add %v483, %v481 : tensor<32xf32>
    %v485 = stablehlo.multiply %v482, %v484 : tensor<32xf32>
    %v486 = stablehlo.add %v485, %v481 : tensor<32xf32>
    %v487 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v488 = stablehlo.multiply %v487, %v486 : tensor<32xf32>
    %v489 = stablehlo.subtract %cb8, %v488 : tensor<32xf32>
    %v490 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v491 = stablehlo.multiply %v490, %cb8v : tensor<32xf32>
    %v492 = stablehlo.add %v491, %v481 : tensor<32xf32>
    %v493 = stablehlo.dot_general %v111, %v140, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %v494 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v495 = stablehlo.multiply %v494, %W9v : tensor<128x64xf32>
    %v496 = stablehlo.add %v495, %v493 : tensor<128x64xf32>
    %v497 = stablehlo.multiply %v494, %v496 : tensor<128x64xf32>
    %v498 = stablehlo.add %v497, %v493 : tensor<128x64xf32>
    %v499 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v500 = stablehlo.multiply %v499, %v498 : tensor<128x64xf32>
    %v501 = stablehlo.subtract %W9, %v500 : tensor<128x64xf32>
    %v502 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v503 = stablehlo.multiply %v502, %W9v : tensor<128x64xf32>
    %v504 = stablehlo.add %v503, %v493 : tensor<128x64xf32>
    %v505 = stablehlo.constant dense<0.0> : tensor<f32>
    %v506 = stablehlo.reduce(%v140 init: %v505) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v507 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v508 = stablehlo.multiply %v507, %b9v : tensor<64xf32>
    %v509 = stablehlo.add %v508, %v506 : tensor<64xf32>
    %v510 = stablehlo.multiply %v507, %v509 : tensor<64xf32>
    %v511 = stablehlo.add %v510, %v506 : tensor<64xf32>
    %v512 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v513 = stablehlo.multiply %v512, %v511 : tensor<64xf32>
    %v514 = stablehlo.subtract %b9, %v513 : tensor<64xf32>
    %v515 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v516 = stablehlo.multiply %v515, %b9v : tensor<64xf32>
    %v517 = stablehlo.add %v516, %v506 : tensor<64xf32>
    %v518 = stablehlo.dot_general %v116, %v136, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<64x64xf32>
    %v519 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v520 = stablehlo.multiply %v519, %Wav : tensor<64x64xf32>
    %v521 = stablehlo.add %v520, %v518 : tensor<64x64xf32>
    %v522 = stablehlo.multiply %v519, %v521 : tensor<64x64xf32>
    %v523 = stablehlo.add %v522, %v518 : tensor<64x64xf32>
    %v524 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v525 = stablehlo.multiply %v524, %v523 : tensor<64x64xf32>
    %v526 = stablehlo.subtract %Wa, %v525 : tensor<64x64xf32>
    %v527 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v528 = stablehlo.multiply %v527, %Wav : tensor<64x64xf32>
    %v529 = stablehlo.add %v528, %v518 : tensor<64x64xf32>
    %v530 = stablehlo.constant dense<0.0> : tensor<f32>
    %v531 = stablehlo.reduce(%v136 init: %v530) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v532 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v533 = stablehlo.multiply %v532, %bav : tensor<64xf32>
    %v534 = stablehlo.add %v533, %v531 : tensor<64xf32>
    %v535 = stablehlo.multiply %v532, %v534 : tensor<64xf32>
    %v536 = stablehlo.add %v535, %v531 : tensor<64xf32>
    %v537 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v538 = stablehlo.multiply %v537, %v536 : tensor<64xf32>
    %v539 = stablehlo.subtract %ba, %v538 : tensor<64xf32>
    %v540 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v541 = stablehlo.multiply %v540, %bav : tensor<64xf32>
    %v542 = stablehlo.add %v541, %v531 : tensor<64xf32>
    %v543 = stablehlo.dot_general %v121, %v132, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x10xf32>) -> tensor<64x10xf32>
    %v544 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v545 = stablehlo.multiply %v544, %Wbv : tensor<64x10xf32>
    %v546 = stablehlo.add %v545, %v543 : tensor<64x10xf32>
    %v547 = stablehlo.multiply %v544, %v546 : tensor<64x10xf32>
    %v548 = stablehlo.add %v547, %v543 : tensor<64x10xf32>
    %v549 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v550 = stablehlo.multiply %v549, %v548 : tensor<64x10xf32>
    %v551 = stablehlo.subtract %Wb, %v550 : tensor<64x10xf32>
    %v552 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v553 = stablehlo.multiply %v552, %Wbv : tensor<64x10xf32>
    %v554 = stablehlo.add %v553, %v543 : tensor<64x10xf32>
    %v555 = stablehlo.constant dense<0.0> : tensor<f32>
    %v556 = stablehlo.reduce(%v132 init: %v555) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v557 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v558 = stablehlo.multiply %v557, %bbv : tensor<10xf32>
    %v559 = stablehlo.add %v558, %v556 : tensor<10xf32>
    %v560 = stablehlo.multiply %v557, %v559 : tensor<10xf32>
    %v561 = stablehlo.add %v560, %v556 : tensor<10xf32>
    %v562 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v563 = stablehlo.multiply %v562, %v561 : tensor<10xf32>
    %v564 = stablehlo.subtract %bb, %v563 : tensor<10xf32>
    %v565 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v566 = stablehlo.multiply %v565, %bbv : tensor<10xf32>
    %v567 = stablehlo.add %v566, %v556 : tensor<10xf32>
    return %v258, %v272, %v289, %v303, %v320, %v334, %v351, %v365, %v382, %v396, %v413, %v427, %v444, %v458, %v475, %v489, %v501, %v514, %v526, %v539, %v551, %v564, %W1m, %cb1m, %W2m, %cb2m, %W3m, %cb3m, %W4m, %cb4m, %W5m, %cb5m, %W6m, %cb6m, %W7m, %cb7m, %W8m, %cb8m, %W9m, %b9m, %Wam, %bam, %Wbm, %bbm, %v261, %v275, %v292, %v306, %v323, %v337, %v354, %v368, %v385, %v399, %v416, %v430, %v447, %v461, %v478, %v492, %v504, %v517, %v529, %v542, %v554, %v567, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
