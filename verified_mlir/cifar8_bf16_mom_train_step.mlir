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
    %v211 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v212 = stablehlo.multiply %v211, %W1v : tensor<16x3x3x3xf32>
    %v213 = stablehlo.add %v212, %v210 : tensor<16x3x3x3xf32>
    %v214 = stablehlo.multiply %v211, %v213 : tensor<16x3x3x3xf32>
    %v215 = stablehlo.add %v214, %v210 : tensor<16x3x3x3xf32>
    %v216 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v217 = stablehlo.multiply %v216, %v215 : tensor<16x3x3x3xf32>
    %v218 = stablehlo.subtract %W1, %v217 : tensor<16x3x3x3xf32>
    %v219 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v220 = stablehlo.multiply %v219, %W1v : tensor<16x3x3x3xf32>
    %v221 = stablehlo.add %v220, %v210 : tensor<16x3x3x3xf32>
    %v222 = stablehlo.reshape %v204 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v223 = stablehlo.constant dense<0.0> : tensor<f32>
    %v224 = stablehlo.reduce(%v222 init: %v223) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v225 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v226 = stablehlo.multiply %v225, %cb1v : tensor<16xf32>
    %v227 = stablehlo.add %v226, %v224 : tensor<16xf32>
    %v228 = stablehlo.multiply %v225, %v227 : tensor<16xf32>
    %v229 = stablehlo.add %v228, %v224 : tensor<16xf32>
    %v230 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v231 = stablehlo.multiply %v230, %v229 : tensor<16xf32>
    %v232 = stablehlo.subtract %cb1, %v231 : tensor<16xf32>
    %v233 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v234 = stablehlo.multiply %v233, %cb1v : tensor<16xf32>
    %v235 = stablehlo.add %v234, %v224 : tensor<16xf32>
    %v236 = stablehlo.reshape %v9 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v237 = stablehlo.reshape %v196 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v238 = stablehlo.transpose %v236, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v239 = stablehlo.transpose %v237, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v240 = stablehlo.convolution(%v238, %v239)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x16x3x3xf32>
    %v241 = stablehlo.transpose %v240, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v242 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v243 = stablehlo.multiply %v242, %W2v : tensor<16x16x3x3xf32>
    %v244 = stablehlo.add %v243, %v241 : tensor<16x16x3x3xf32>
    %v245 = stablehlo.multiply %v242, %v244 : tensor<16x16x3x3xf32>
    %v246 = stablehlo.add %v245, %v241 : tensor<16x16x3x3xf32>
    %v247 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v248 = stablehlo.multiply %v247, %v246 : tensor<16x16x3x3xf32>
    %v249 = stablehlo.subtract %W2, %v248 : tensor<16x16x3x3xf32>
    %v250 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v251 = stablehlo.multiply %v250, %W2v : tensor<16x16x3x3xf32>
    %v252 = stablehlo.add %v251, %v241 : tensor<16x16x3x3xf32>
    %v253 = stablehlo.reshape %v196 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v254 = stablehlo.constant dense<0.0> : tensor<f32>
    %v255 = stablehlo.reduce(%v253 init: %v254) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v256 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v257 = stablehlo.multiply %v256, %cb2v : tensor<16xf32>
    %v258 = stablehlo.add %v257, %v255 : tensor<16xf32>
    %v259 = stablehlo.multiply %v256, %v258 : tensor<16xf32>
    %v260 = stablehlo.add %v259, %v255 : tensor<16xf32>
    %v261 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v262 = stablehlo.multiply %v261, %v260 : tensor<16xf32>
    %v263 = stablehlo.subtract %cb2, %v262 : tensor<16xf32>
    %v264 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v265 = stablehlo.multiply %v264, %cb2v : tensor<16xf32>
    %v266 = stablehlo.add %v265, %v255 : tensor<16xf32>
    %v267 = stablehlo.reshape %v23 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v268 = stablehlo.reshape %v183 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v269 = stablehlo.transpose %v267, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v270 = stablehlo.transpose %v268, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v271 = stablehlo.convolution(%v269, %v270)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v272 = stablehlo.transpose %v271, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v273 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v274 = stablehlo.multiply %v273, %W3v : tensor<16x16x3x3xf32>
    %v275 = stablehlo.add %v274, %v272 : tensor<16x16x3x3xf32>
    %v276 = stablehlo.multiply %v273, %v275 : tensor<16x16x3x3xf32>
    %v277 = stablehlo.add %v276, %v272 : tensor<16x16x3x3xf32>
    %v278 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v279 = stablehlo.multiply %v278, %v277 : tensor<16x16x3x3xf32>
    %v280 = stablehlo.subtract %W3, %v279 : tensor<16x16x3x3xf32>
    %v281 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v282 = stablehlo.multiply %v281, %W3v : tensor<16x16x3x3xf32>
    %v283 = stablehlo.add %v282, %v272 : tensor<16x16x3x3xf32>
    %v284 = stablehlo.reshape %v183 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v285 = stablehlo.constant dense<0.0> : tensor<f32>
    %v286 = stablehlo.reduce(%v284 init: %v285) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v287 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v288 = stablehlo.multiply %v287, %cb3v : tensor<16xf32>
    %v289 = stablehlo.add %v288, %v286 : tensor<16xf32>
    %v290 = stablehlo.multiply %v287, %v289 : tensor<16xf32>
    %v291 = stablehlo.add %v290, %v286 : tensor<16xf32>
    %v292 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v293 = stablehlo.multiply %v292, %v291 : tensor<16xf32>
    %v294 = stablehlo.subtract %cb3, %v293 : tensor<16xf32>
    %v295 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v296 = stablehlo.multiply %v295, %cb3v : tensor<16xf32>
    %v297 = stablehlo.add %v296, %v286 : tensor<16xf32>
    %v298 = stablehlo.reshape %v33 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v299 = stablehlo.reshape %v175 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v300 = stablehlo.transpose %v298, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v301 = stablehlo.transpose %v299, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v302 = stablehlo.convolution(%v300, %v301)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v303 = stablehlo.transpose %v302, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v304 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v305 = stablehlo.multiply %v304, %W4v : tensor<16x16x3x3xf32>
    %v306 = stablehlo.add %v305, %v303 : tensor<16x16x3x3xf32>
    %v307 = stablehlo.multiply %v304, %v306 : tensor<16x16x3x3xf32>
    %v308 = stablehlo.add %v307, %v303 : tensor<16x16x3x3xf32>
    %v309 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v310 = stablehlo.multiply %v309, %v308 : tensor<16x16x3x3xf32>
    %v311 = stablehlo.subtract %W4, %v310 : tensor<16x16x3x3xf32>
    %v312 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v313 = stablehlo.multiply %v312, %W4v : tensor<16x16x3x3xf32>
    %v314 = stablehlo.add %v313, %v303 : tensor<16x16x3x3xf32>
    %v315 = stablehlo.reshape %v175 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v316 = stablehlo.constant dense<0.0> : tensor<f32>
    %v317 = stablehlo.reduce(%v315 init: %v316) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v318 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v319 = stablehlo.multiply %v318, %cb4v : tensor<16xf32>
    %v320 = stablehlo.add %v319, %v317 : tensor<16xf32>
    %v321 = stablehlo.multiply %v318, %v320 : tensor<16xf32>
    %v322 = stablehlo.add %v321, %v317 : tensor<16xf32>
    %v323 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v324 = stablehlo.multiply %v323, %v322 : tensor<16xf32>
    %v325 = stablehlo.subtract %cb4, %v324 : tensor<16xf32>
    %v326 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v327 = stablehlo.multiply %v326, %cb4v : tensor<16xf32>
    %v328 = stablehlo.add %v327, %v317 : tensor<16xf32>
    %v329 = stablehlo.reshape %v47 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v330 = stablehlo.reshape %v162 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v331 = stablehlo.transpose %v329, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v332 = stablehlo.transpose %v330, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v333 = stablehlo.convolution(%v331, %v332)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<16x32x3x3xf32>
    %v334 = stablehlo.transpose %v333, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v335 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v336 = stablehlo.multiply %v335, %W5v : tensor<32x16x3x3xf32>
    %v337 = stablehlo.add %v336, %v334 : tensor<32x16x3x3xf32>
    %v338 = stablehlo.multiply %v335, %v337 : tensor<32x16x3x3xf32>
    %v339 = stablehlo.add %v338, %v334 : tensor<32x16x3x3xf32>
    %v340 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v341 = stablehlo.multiply %v340, %v339 : tensor<32x16x3x3xf32>
    %v342 = stablehlo.subtract %W5, %v341 : tensor<32x16x3x3xf32>
    %v343 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v344 = stablehlo.multiply %v343, %W5v : tensor<32x16x3x3xf32>
    %v345 = stablehlo.add %v344, %v334 : tensor<32x16x3x3xf32>
    %v346 = stablehlo.reshape %v162 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v347 = stablehlo.constant dense<0.0> : tensor<f32>
    %v348 = stablehlo.reduce(%v346 init: %v347) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v349 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v350 = stablehlo.multiply %v349, %cb5v : tensor<32xf32>
    %v351 = stablehlo.add %v350, %v348 : tensor<32xf32>
    %v352 = stablehlo.multiply %v349, %v351 : tensor<32xf32>
    %v353 = stablehlo.add %v352, %v348 : tensor<32xf32>
    %v354 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v355 = stablehlo.multiply %v354, %v353 : tensor<32xf32>
    %v356 = stablehlo.subtract %cb5, %v355 : tensor<32xf32>
    %v357 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v358 = stablehlo.multiply %v357, %cb5v : tensor<32xf32>
    %v359 = stablehlo.add %v358, %v348 : tensor<32xf32>
    %v360 = stablehlo.reshape %v57 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v361 = stablehlo.reshape %v154 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v362 = stablehlo.transpose %v360, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v363 = stablehlo.transpose %v361, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v364 = stablehlo.convolution(%v362, %v363)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<32x32x3x3xf32>
    %v365 = stablehlo.transpose %v364, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v366 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v367 = stablehlo.multiply %v366, %W6v : tensor<32x32x3x3xf32>
    %v368 = stablehlo.add %v367, %v365 : tensor<32x32x3x3xf32>
    %v369 = stablehlo.multiply %v366, %v368 : tensor<32x32x3x3xf32>
    %v370 = stablehlo.add %v369, %v365 : tensor<32x32x3x3xf32>
    %v371 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v372 = stablehlo.multiply %v371, %v370 : tensor<32x32x3x3xf32>
    %v373 = stablehlo.subtract %W6, %v372 : tensor<32x32x3x3xf32>
    %v374 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v375 = stablehlo.multiply %v374, %W6v : tensor<32x32x3x3xf32>
    %v376 = stablehlo.add %v375, %v365 : tensor<32x32x3x3xf32>
    %v377 = stablehlo.reshape %v154 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v378 = stablehlo.constant dense<0.0> : tensor<f32>
    %v379 = stablehlo.reduce(%v377 init: %v378) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v380 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v381 = stablehlo.multiply %v380, %cb6v : tensor<32xf32>
    %v382 = stablehlo.add %v381, %v379 : tensor<32xf32>
    %v383 = stablehlo.multiply %v380, %v382 : tensor<32xf32>
    %v384 = stablehlo.add %v383, %v379 : tensor<32xf32>
    %v385 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v386 = stablehlo.multiply %v385, %v384 : tensor<32xf32>
    %v387 = stablehlo.subtract %cb6, %v386 : tensor<32xf32>
    %v388 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v389 = stablehlo.multiply %v388, %cb6v : tensor<32xf32>
    %v390 = stablehlo.add %v389, %v379 : tensor<32xf32>
    %v391 = stablehlo.reshape %v71 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v392 = stablehlo.reshape %v141 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v393 = stablehlo.transpose %v391, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v394 = stablehlo.transpose %v392, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v395 = stablehlo.convolution(%v393, %v394)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v396 = stablehlo.transpose %v395, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v397 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v398 = stablehlo.multiply %v397, %W7v : tensor<32x32x3x3xf32>
    %v399 = stablehlo.add %v398, %v396 : tensor<32x32x3x3xf32>
    %v400 = stablehlo.multiply %v397, %v399 : tensor<32x32x3x3xf32>
    %v401 = stablehlo.add %v400, %v396 : tensor<32x32x3x3xf32>
    %v402 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v403 = stablehlo.multiply %v402, %v401 : tensor<32x32x3x3xf32>
    %v404 = stablehlo.subtract %W7, %v403 : tensor<32x32x3x3xf32>
    %v405 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v406 = stablehlo.multiply %v405, %W7v : tensor<32x32x3x3xf32>
    %v407 = stablehlo.add %v406, %v396 : tensor<32x32x3x3xf32>
    %v408 = stablehlo.reshape %v141 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v409 = stablehlo.constant dense<0.0> : tensor<f32>
    %v410 = stablehlo.reduce(%v408 init: %v409) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v411 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v412 = stablehlo.multiply %v411, %cb7v : tensor<32xf32>
    %v413 = stablehlo.add %v412, %v410 : tensor<32xf32>
    %v414 = stablehlo.multiply %v411, %v413 : tensor<32xf32>
    %v415 = stablehlo.add %v414, %v410 : tensor<32xf32>
    %v416 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v417 = stablehlo.multiply %v416, %v415 : tensor<32xf32>
    %v418 = stablehlo.subtract %cb7, %v417 : tensor<32xf32>
    %v419 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v420 = stablehlo.multiply %v419, %cb7v : tensor<32xf32>
    %v421 = stablehlo.add %v420, %v410 : tensor<32xf32>
    %v422 = stablehlo.reshape %v81 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v423 = stablehlo.reshape %v133 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v424 = stablehlo.transpose %v422, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v425 = stablehlo.transpose %v423, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v426 = stablehlo.convolution(%v424, %v425)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v427 = stablehlo.transpose %v426, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v428 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v429 = stablehlo.multiply %v428, %W8v : tensor<32x32x3x3xf32>
    %v430 = stablehlo.add %v429, %v427 : tensor<32x32x3x3xf32>
    %v431 = stablehlo.multiply %v428, %v430 : tensor<32x32x3x3xf32>
    %v432 = stablehlo.add %v431, %v427 : tensor<32x32x3x3xf32>
    %v433 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v434 = stablehlo.multiply %v433, %v432 : tensor<32x32x3x3xf32>
    %v435 = stablehlo.subtract %W8, %v434 : tensor<32x32x3x3xf32>
    %v436 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v437 = stablehlo.multiply %v436, %W8v : tensor<32x32x3x3xf32>
    %v438 = stablehlo.add %v437, %v427 : tensor<32x32x3x3xf32>
    %v439 = stablehlo.reshape %v133 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v440 = stablehlo.constant dense<0.0> : tensor<f32>
    %v441 = stablehlo.reduce(%v439 init: %v440) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v442 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v443 = stablehlo.multiply %v442, %cb8v : tensor<32xf32>
    %v444 = stablehlo.add %v443, %v441 : tensor<32xf32>
    %v445 = stablehlo.multiply %v442, %v444 : tensor<32xf32>
    %v446 = stablehlo.add %v445, %v441 : tensor<32xf32>
    %v447 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v448 = stablehlo.multiply %v447, %v446 : tensor<32xf32>
    %v449 = stablehlo.subtract %cb8, %v448 : tensor<32xf32>
    %v450 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v451 = stablehlo.multiply %v450, %cb8v : tensor<32xf32>
    %v452 = stablehlo.add %v451, %v441 : tensor<32xf32>
    %v453 = stablehlo.dot_general %v95, %v124, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %v454 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v455 = stablehlo.multiply %v454, %W9v : tensor<128x64xf32>
    %v456 = stablehlo.add %v455, %v453 : tensor<128x64xf32>
    %v457 = stablehlo.multiply %v454, %v456 : tensor<128x64xf32>
    %v458 = stablehlo.add %v457, %v453 : tensor<128x64xf32>
    %v459 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v460 = stablehlo.multiply %v459, %v458 : tensor<128x64xf32>
    %v461 = stablehlo.subtract %W9, %v460 : tensor<128x64xf32>
    %v462 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v463 = stablehlo.multiply %v462, %W9v : tensor<128x64xf32>
    %v464 = stablehlo.add %v463, %v453 : tensor<128x64xf32>
    %v465 = stablehlo.constant dense<0.0> : tensor<f32>
    %v466 = stablehlo.reduce(%v124 init: %v465) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v467 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v468 = stablehlo.multiply %v467, %b9v : tensor<64xf32>
    %v469 = stablehlo.add %v468, %v466 : tensor<64xf32>
    %v470 = stablehlo.multiply %v467, %v469 : tensor<64xf32>
    %v471 = stablehlo.add %v470, %v466 : tensor<64xf32>
    %v472 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v473 = stablehlo.multiply %v472, %v471 : tensor<64xf32>
    %v474 = stablehlo.subtract %b9, %v473 : tensor<64xf32>
    %v475 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v476 = stablehlo.multiply %v475, %b9v : tensor<64xf32>
    %v477 = stablehlo.add %v476, %v466 : tensor<64xf32>
    %v478 = stablehlo.dot_general %v100, %v120, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<64x64xf32>
    %v479 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v480 = stablehlo.multiply %v479, %Wav : tensor<64x64xf32>
    %v481 = stablehlo.add %v480, %v478 : tensor<64x64xf32>
    %v482 = stablehlo.multiply %v479, %v481 : tensor<64x64xf32>
    %v483 = stablehlo.add %v482, %v478 : tensor<64x64xf32>
    %v484 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v485 = stablehlo.multiply %v484, %v483 : tensor<64x64xf32>
    %v486 = stablehlo.subtract %Wa, %v485 : tensor<64x64xf32>
    %v487 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v488 = stablehlo.multiply %v487, %Wav : tensor<64x64xf32>
    %v489 = stablehlo.add %v488, %v478 : tensor<64x64xf32>
    %v490 = stablehlo.constant dense<0.0> : tensor<f32>
    %v491 = stablehlo.reduce(%v120 init: %v490) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v492 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v493 = stablehlo.multiply %v492, %bav : tensor<64xf32>
    %v494 = stablehlo.add %v493, %v491 : tensor<64xf32>
    %v495 = stablehlo.multiply %v492, %v494 : tensor<64xf32>
    %v496 = stablehlo.add %v495, %v491 : tensor<64xf32>
    %v497 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v498 = stablehlo.multiply %v497, %v496 : tensor<64xf32>
    %v499 = stablehlo.subtract %ba, %v498 : tensor<64xf32>
    %v500 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v501 = stablehlo.multiply %v500, %bav : tensor<64xf32>
    %v502 = stablehlo.add %v501, %v491 : tensor<64xf32>
    %v503 = stablehlo.dot_general %v105, %v116, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x10xf32>) -> tensor<64x10xf32>
    %v504 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v505 = stablehlo.multiply %v504, %Wbv : tensor<64x10xf32>
    %v506 = stablehlo.add %v505, %v503 : tensor<64x10xf32>
    %v507 = stablehlo.multiply %v504, %v506 : tensor<64x10xf32>
    %v508 = stablehlo.add %v507, %v503 : tensor<64x10xf32>
    %v509 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v510 = stablehlo.multiply %v509, %v508 : tensor<64x10xf32>
    %v511 = stablehlo.subtract %Wb, %v510 : tensor<64x10xf32>
    %v512 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v513 = stablehlo.multiply %v512, %Wbv : tensor<64x10xf32>
    %v514 = stablehlo.add %v513, %v503 : tensor<64x10xf32>
    %v515 = stablehlo.constant dense<0.0> : tensor<f32>
    %v516 = stablehlo.reduce(%v116 init: %v515) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v517 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v518 = stablehlo.multiply %v517, %bbv : tensor<10xf32>
    %v519 = stablehlo.add %v518, %v516 : tensor<10xf32>
    %v520 = stablehlo.multiply %v517, %v519 : tensor<10xf32>
    %v521 = stablehlo.add %v520, %v516 : tensor<10xf32>
    %v522 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v523 = stablehlo.multiply %v522, %v521 : tensor<10xf32>
    %v524 = stablehlo.subtract %bb, %v523 : tensor<10xf32>
    %v525 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v526 = stablehlo.multiply %v525, %bbv : tensor<10xf32>
    %v527 = stablehlo.add %v526, %v516 : tensor<10xf32>
    return %v218, %v232, %v249, %v263, %v280, %v294, %v311, %v325, %v342, %v356, %v373, %v387, %v404, %v418, %v435, %v449, %v461, %v474, %v486, %v499, %v511, %v524, %W1m, %cb1m, %W2m, %cb2m, %W3m, %cb3m, %W4m, %cb4m, %W5m, %cb5m, %W6m, %cb6m, %W7m, %cb7m, %W8m, %cb8m, %W9m, %b9m, %Wam, %bam, %Wbm, %bbm, %v221, %v235, %v252, %v266, %v283, %v297, %v314, %v328, %v345, %v359, %v376, %v390, %v407, %v421, %v438, %v452, %v464, %v477, %v489, %v502, %v514, %v527, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
