module @m {
  func.func @cifar8wb_bn_bf16adam_train_step(%x: tensor<128x3072xf32>, %W1: tensor<16x3x3x3xf32>, %cb1: tensor<16xf32>, %g1: tensor<16xf32>, %bt1: tensor<16xf32>, %W2: tensor<16x16x3x3xf32>, %cb2: tensor<16xf32>, %g2: tensor<16xf32>, %bt2: tensor<16xf32>, %W3: tensor<16x16x3x3xf32>, %cb3: tensor<16xf32>, %g3: tensor<16xf32>, %bt3: tensor<16xf32>, %W4: tensor<16x16x3x3xf32>, %cb4: tensor<16xf32>, %g4: tensor<16xf32>, %bt4: tensor<16xf32>, %W5: tensor<32x16x3x3xf32>, %cb5: tensor<32xf32>, %g5: tensor<32xf32>, %bt5: tensor<32xf32>, %W6: tensor<32x32x3x3xf32>, %cb6: tensor<32xf32>, %g6: tensor<32xf32>, %bt6: tensor<32xf32>, %W7: tensor<32x32x3x3xf32>, %cb7: tensor<32xf32>, %g7: tensor<32xf32>, %bt7: tensor<32xf32>, %W8: tensor<32x32x3x3xf32>, %cb8: tensor<32xf32>, %g8: tensor<32xf32>, %bt8: tensor<32xf32>, %W9: tensor<128x512xf32>, %b9: tensor<512xf32>, %Wa: tensor<512x512xf32>, %ba: tensor<512xf32>, %Wb: tensor<512x10xf32>, %bb: tensor<10xf32>, %W1m: tensor<16x3x3x3xf32>, %cb1m: tensor<16xf32>, %g1m: tensor<16xf32>, %bt1m: tensor<16xf32>, %W2m: tensor<16x16x3x3xf32>, %cb2m: tensor<16xf32>, %g2m: tensor<16xf32>, %bt2m: tensor<16xf32>, %W3m: tensor<16x16x3x3xf32>, %cb3m: tensor<16xf32>, %g3m: tensor<16xf32>, %bt3m: tensor<16xf32>, %W4m: tensor<16x16x3x3xf32>, %cb4m: tensor<16xf32>, %g4m: tensor<16xf32>, %bt4m: tensor<16xf32>, %W5m: tensor<32x16x3x3xf32>, %cb5m: tensor<32xf32>, %g5m: tensor<32xf32>, %bt5m: tensor<32xf32>, %W6m: tensor<32x32x3x3xf32>, %cb6m: tensor<32xf32>, %g6m: tensor<32xf32>, %bt6m: tensor<32xf32>, %W7m: tensor<32x32x3x3xf32>, %cb7m: tensor<32xf32>, %g7m: tensor<32xf32>, %bt7m: tensor<32xf32>, %W8m: tensor<32x32x3x3xf32>, %cb8m: tensor<32xf32>, %g8m: tensor<32xf32>, %bt8m: tensor<32xf32>, %W9m: tensor<128x512xf32>, %b9m: tensor<512xf32>, %Wam: tensor<512x512xf32>, %bam: tensor<512xf32>, %Wbm: tensor<512x10xf32>, %bbm: tensor<10xf32>, %W1v: tensor<16x3x3x3xf32>, %cb1v: tensor<16xf32>, %g1v: tensor<16xf32>, %bt1v: tensor<16xf32>, %W2v: tensor<16x16x3x3xf32>, %cb2v: tensor<16xf32>, %g2v: tensor<16xf32>, %bt2v: tensor<16xf32>, %W3v: tensor<16x16x3x3xf32>, %cb3v: tensor<16xf32>, %g3v: tensor<16xf32>, %bt3v: tensor<16xf32>, %W4v: tensor<16x16x3x3xf32>, %cb4v: tensor<16xf32>, %g4v: tensor<16xf32>, %bt4v: tensor<16xf32>, %W5v: tensor<32x16x3x3xf32>, %cb5v: tensor<32xf32>, %g5v: tensor<32xf32>, %bt5v: tensor<32xf32>, %W6v: tensor<32x32x3x3xf32>, %cb6v: tensor<32xf32>, %g6v: tensor<32xf32>, %bt6v: tensor<32xf32>, %W7v: tensor<32x32x3x3xf32>, %cb7v: tensor<32xf32>, %g7v: tensor<32xf32>, %bt7v: tensor<32xf32>, %W8v: tensor<32x32x3x3xf32>, %cb8v: tensor<32xf32>, %g8v: tensor<32xf32>, %bt8v: tensor<32xf32>, %W9v: tensor<128x512xf32>, %b9v: tensor<512xf32>, %Wav: tensor<512x512xf32>, %bav: tensor<512xf32>, %Wbv: tensor<512x10xf32>, %bbv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: tensor<128x10xf32>) -> (tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>) {
    // ── cifar8-BN train step, BATCHED op family: every line is pretty(verified AST
    //    node), except the marked report-only loss + the %bc passthroughs ──
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
    %v9 = stablehlo.constant dense<0.0> : tensor<f32>
    %v10 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v11 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v12 = stablehlo.reduce(%v8 init: %v9) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v13 = stablehlo.broadcast_in_dim %v12, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v14 = stablehlo.divide %v13, %v10 : tensor<128x16x32x32xf32>
    %v15 = stablehlo.subtract %v8, %v14 : tensor<128x16x32x32xf32>
    %v16 = stablehlo.multiply %v15, %v15 : tensor<128x16x32x32xf32>
    %v17 = stablehlo.reduce(%v16 init: %v9) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v18 = stablehlo.broadcast_in_dim %v17, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v19 = stablehlo.divide %v18, %v10 : tensor<128x16x32x32xf32>
    %v20 = stablehlo.add %v19, %v11 : tensor<128x16x32x32xf32>
    %v21 = stablehlo.rsqrt %v20 : tensor<128x16x32x32xf32>
    %v22 = stablehlo.multiply %v15, %v21 : tensor<128x16x32x32xf32>
    %v23 = stablehlo.broadcast_in_dim %g1, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v24 = stablehlo.broadcast_in_dim %bt1, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v25 = stablehlo.multiply %v22, %v23 : tensor<128x16x32x32xf32>
    %v26 = stablehlo.add %v25, %v24 : tensor<128x16x32x32xf32>
    %v27 = stablehlo.reshape %v26 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v28 = stablehlo.reshape %v27 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v29 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v30 = stablehlo.maximum %v28, %v29 : tensor<128x16x32x32xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v32 = stablehlo.reshape %v31 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v33 = stablehlo.convert %v32 : (tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xbf16>
    %v34 = stablehlo.convert %W2 : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xbf16>
    %v35 = stablehlo.convolution(%v33, %v34)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xbf16>, tensor<16x16x3x3xbf16>) -> tensor<128x16x32x32xbf16>
    %v36 = stablehlo.convert %v35 : (tensor<128x16x32x32xbf16>) -> tensor<128x16x32x32xf32>
    %v37 = stablehlo.broadcast_in_dim %cb2, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v38 = stablehlo.add %v36, %v37 : tensor<128x16x32x32xf32>
    %v39 = stablehlo.reshape %v38 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v40 = stablehlo.reshape %v39 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v41 = stablehlo.constant dense<0.0> : tensor<f32>
    %v42 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v43 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v44 = stablehlo.reduce(%v40 init: %v41) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v45 = stablehlo.broadcast_in_dim %v44, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v46 = stablehlo.divide %v45, %v42 : tensor<128x16x32x32xf32>
    %v47 = stablehlo.subtract %v40, %v46 : tensor<128x16x32x32xf32>
    %v48 = stablehlo.multiply %v47, %v47 : tensor<128x16x32x32xf32>
    %v49 = stablehlo.reduce(%v48 init: %v41) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v50 = stablehlo.broadcast_in_dim %v49, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v51 = stablehlo.divide %v50, %v42 : tensor<128x16x32x32xf32>
    %v52 = stablehlo.add %v51, %v43 : tensor<128x16x32x32xf32>
    %v53 = stablehlo.rsqrt %v52 : tensor<128x16x32x32xf32>
    %v54 = stablehlo.multiply %v47, %v53 : tensor<128x16x32x32xf32>
    %v55 = stablehlo.broadcast_in_dim %g2, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v56 = stablehlo.broadcast_in_dim %bt2, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v57 = stablehlo.multiply %v54, %v55 : tensor<128x16x32x32xf32>
    %v58 = stablehlo.add %v57, %v56 : tensor<128x16x32x32xf32>
    %v59 = stablehlo.reshape %v58 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v60 = stablehlo.reshape %v59 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v61 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v62 = stablehlo.maximum %v60, %v61 : tensor<128x16x32x32xf32>
    %v63 = stablehlo.reshape %v62 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v64 = stablehlo.reshape %v63 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v65 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v66 = "stablehlo.reduce_window"(%v64, %v65) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16x16x16xf32>
    %v67 = stablehlo.reshape %v66 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v68 = stablehlo.reshape %v67 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v69 = stablehlo.convert %v68 : (tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xbf16>
    %v70 = stablehlo.convert %W3 : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xbf16>
    %v71 = stablehlo.convolution(%v69, %v70)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xbf16>, tensor<16x16x3x3xbf16>) -> tensor<128x16x16x16xbf16>
    %v72 = stablehlo.convert %v71 : (tensor<128x16x16x16xbf16>) -> tensor<128x16x16x16xf32>
    %v73 = stablehlo.broadcast_in_dim %cb3, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v74 = stablehlo.add %v72, %v73 : tensor<128x16x16x16xf32>
    %v75 = stablehlo.reshape %v74 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v76 = stablehlo.reshape %v75 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v77 = stablehlo.constant dense<0.0> : tensor<f32>
    %v78 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v79 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v80 = stablehlo.reduce(%v76 init: %v77) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v81 = stablehlo.broadcast_in_dim %v80, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v82 = stablehlo.divide %v81, %v78 : tensor<128x16x16x16xf32>
    %v83 = stablehlo.subtract %v76, %v82 : tensor<128x16x16x16xf32>
    %v84 = stablehlo.multiply %v83, %v83 : tensor<128x16x16x16xf32>
    %v85 = stablehlo.reduce(%v84 init: %v77) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v86 = stablehlo.broadcast_in_dim %v85, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v87 = stablehlo.divide %v86, %v78 : tensor<128x16x16x16xf32>
    %v88 = stablehlo.add %v87, %v79 : tensor<128x16x16x16xf32>
    %v89 = stablehlo.rsqrt %v88 : tensor<128x16x16x16xf32>
    %v90 = stablehlo.multiply %v83, %v89 : tensor<128x16x16x16xf32>
    %v91 = stablehlo.broadcast_in_dim %g3, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v92 = stablehlo.broadcast_in_dim %bt3, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v93 = stablehlo.multiply %v90, %v91 : tensor<128x16x16x16xf32>
    %v94 = stablehlo.add %v93, %v92 : tensor<128x16x16x16xf32>
    %v95 = stablehlo.reshape %v94 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v96 = stablehlo.reshape %v95 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v97 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v98 = stablehlo.maximum %v96, %v97 : tensor<128x16x16x16xf32>
    %v99 = stablehlo.reshape %v98 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v100 = stablehlo.reshape %v99 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v101 = stablehlo.convert %v100 : (tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xbf16>
    %v102 = stablehlo.convert %W4 : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xbf16>
    %v103 = stablehlo.convolution(%v101, %v102)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xbf16>, tensor<16x16x3x3xbf16>) -> tensor<128x16x16x16xbf16>
    %v104 = stablehlo.convert %v103 : (tensor<128x16x16x16xbf16>) -> tensor<128x16x16x16xf32>
    %v105 = stablehlo.broadcast_in_dim %cb4, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v106 = stablehlo.add %v104, %v105 : tensor<128x16x16x16xf32>
    %v107 = stablehlo.reshape %v106 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v108 = stablehlo.reshape %v107 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v109 = stablehlo.constant dense<0.0> : tensor<f32>
    %v110 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v111 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v112 = stablehlo.reduce(%v108 init: %v109) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v113 = stablehlo.broadcast_in_dim %v112, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v114 = stablehlo.divide %v113, %v110 : tensor<128x16x16x16xf32>
    %v115 = stablehlo.subtract %v108, %v114 : tensor<128x16x16x16xf32>
    %v116 = stablehlo.multiply %v115, %v115 : tensor<128x16x16x16xf32>
    %v117 = stablehlo.reduce(%v116 init: %v109) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v118 = stablehlo.broadcast_in_dim %v117, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v119 = stablehlo.divide %v118, %v110 : tensor<128x16x16x16xf32>
    %v120 = stablehlo.add %v119, %v111 : tensor<128x16x16x16xf32>
    %v121 = stablehlo.rsqrt %v120 : tensor<128x16x16x16xf32>
    %v122 = stablehlo.multiply %v115, %v121 : tensor<128x16x16x16xf32>
    %v123 = stablehlo.broadcast_in_dim %g4, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v124 = stablehlo.broadcast_in_dim %bt4, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v125 = stablehlo.multiply %v122, %v123 : tensor<128x16x16x16xf32>
    %v126 = stablehlo.add %v125, %v124 : tensor<128x16x16x16xf32>
    %v127 = stablehlo.reshape %v126 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v128 = stablehlo.reshape %v127 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v129 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v130 = stablehlo.maximum %v128, %v129 : tensor<128x16x16x16xf32>
    %v131 = stablehlo.reshape %v130 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v132 = stablehlo.reshape %v131 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v133 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v134 = "stablehlo.reduce_window"(%v132, %v133) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x8x8xf32>
    %v135 = stablehlo.reshape %v134 : (tensor<128x16x8x8xf32>) -> tensor<128x1024xf32>
    %v136 = stablehlo.reshape %v135 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v137 = stablehlo.convert %v136 : (tensor<128x16x8x8xf32>) -> tensor<128x16x8x8xbf16>
    %v138 = stablehlo.convert %W5 : (tensor<32x16x3x3xf32>) -> tensor<32x16x3x3xbf16>
    %v139 = stablehlo.convolution(%v137, %v138)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x8x8xbf16>, tensor<32x16x3x3xbf16>) -> tensor<128x32x8x8xbf16>
    %v140 = stablehlo.convert %v139 : (tensor<128x32x8x8xbf16>) -> tensor<128x32x8x8xf32>
    %v141 = stablehlo.broadcast_in_dim %cb5, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v142 = stablehlo.add %v140, %v141 : tensor<128x32x8x8xf32>
    %v143 = stablehlo.reshape %v142 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v144 = stablehlo.reshape %v143 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v145 = stablehlo.constant dense<0.0> : tensor<f32>
    %v146 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v147 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v148 = stablehlo.reduce(%v144 init: %v145) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v149 = stablehlo.broadcast_in_dim %v148, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v150 = stablehlo.divide %v149, %v146 : tensor<128x32x8x8xf32>
    %v151 = stablehlo.subtract %v144, %v150 : tensor<128x32x8x8xf32>
    %v152 = stablehlo.multiply %v151, %v151 : tensor<128x32x8x8xf32>
    %v153 = stablehlo.reduce(%v152 init: %v145) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v154 = stablehlo.broadcast_in_dim %v153, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v155 = stablehlo.divide %v154, %v146 : tensor<128x32x8x8xf32>
    %v156 = stablehlo.add %v155, %v147 : tensor<128x32x8x8xf32>
    %v157 = stablehlo.rsqrt %v156 : tensor<128x32x8x8xf32>
    %v158 = stablehlo.multiply %v151, %v157 : tensor<128x32x8x8xf32>
    %v159 = stablehlo.broadcast_in_dim %g5, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v160 = stablehlo.broadcast_in_dim %bt5, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v161 = stablehlo.multiply %v158, %v159 : tensor<128x32x8x8xf32>
    %v162 = stablehlo.add %v161, %v160 : tensor<128x32x8x8xf32>
    %v163 = stablehlo.reshape %v162 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v164 = stablehlo.reshape %v163 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v165 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v166 = stablehlo.maximum %v164, %v165 : tensor<128x32x8x8xf32>
    %v167 = stablehlo.reshape %v166 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v168 = stablehlo.reshape %v167 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v169 = stablehlo.convert %v168 : (tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xbf16>
    %v170 = stablehlo.convert %W6 : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xbf16>
    %v171 = stablehlo.convolution(%v169, %v170)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xbf16>, tensor<32x32x3x3xbf16>) -> tensor<128x32x8x8xbf16>
    %v172 = stablehlo.convert %v171 : (tensor<128x32x8x8xbf16>) -> tensor<128x32x8x8xf32>
    %v173 = stablehlo.broadcast_in_dim %cb6, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v174 = stablehlo.add %v172, %v173 : tensor<128x32x8x8xf32>
    %v175 = stablehlo.reshape %v174 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v176 = stablehlo.reshape %v175 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v177 = stablehlo.constant dense<0.0> : tensor<f32>
    %v178 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v179 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v180 = stablehlo.reduce(%v176 init: %v177) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v181 = stablehlo.broadcast_in_dim %v180, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v182 = stablehlo.divide %v181, %v178 : tensor<128x32x8x8xf32>
    %v183 = stablehlo.subtract %v176, %v182 : tensor<128x32x8x8xf32>
    %v184 = stablehlo.multiply %v183, %v183 : tensor<128x32x8x8xf32>
    %v185 = stablehlo.reduce(%v184 init: %v177) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v186 = stablehlo.broadcast_in_dim %v185, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v187 = stablehlo.divide %v186, %v178 : tensor<128x32x8x8xf32>
    %v188 = stablehlo.add %v187, %v179 : tensor<128x32x8x8xf32>
    %v189 = stablehlo.rsqrt %v188 : tensor<128x32x8x8xf32>
    %v190 = stablehlo.multiply %v183, %v189 : tensor<128x32x8x8xf32>
    %v191 = stablehlo.broadcast_in_dim %g6, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v192 = stablehlo.broadcast_in_dim %bt6, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v193 = stablehlo.multiply %v190, %v191 : tensor<128x32x8x8xf32>
    %v194 = stablehlo.add %v193, %v192 : tensor<128x32x8x8xf32>
    %v195 = stablehlo.reshape %v194 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v196 = stablehlo.reshape %v195 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v197 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v198 = stablehlo.maximum %v196, %v197 : tensor<128x32x8x8xf32>
    %v199 = stablehlo.reshape %v198 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v200 = stablehlo.reshape %v199 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v201 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v202 = "stablehlo.reduce_window"(%v200, %v201) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32x4x4xf32>
    %v203 = stablehlo.reshape %v202 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v204 = stablehlo.reshape %v203 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v205 = stablehlo.convert %v204 : (tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xbf16>
    %v206 = stablehlo.convert %W7 : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xbf16>
    %v207 = stablehlo.convolution(%v205, %v206)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xbf16>, tensor<32x32x3x3xbf16>) -> tensor<128x32x4x4xbf16>
    %v208 = stablehlo.convert %v207 : (tensor<128x32x4x4xbf16>) -> tensor<128x32x4x4xf32>
    %v209 = stablehlo.broadcast_in_dim %cb7, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v210 = stablehlo.add %v208, %v209 : tensor<128x32x4x4xf32>
    %v211 = stablehlo.reshape %v210 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v212 = stablehlo.reshape %v211 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v213 = stablehlo.constant dense<0.0> : tensor<f32>
    %v214 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v215 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v216 = stablehlo.reduce(%v212 init: %v213) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v217 = stablehlo.broadcast_in_dim %v216, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v218 = stablehlo.divide %v217, %v214 : tensor<128x32x4x4xf32>
    %v219 = stablehlo.subtract %v212, %v218 : tensor<128x32x4x4xf32>
    %v220 = stablehlo.multiply %v219, %v219 : tensor<128x32x4x4xf32>
    %v221 = stablehlo.reduce(%v220 init: %v213) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v222 = stablehlo.broadcast_in_dim %v221, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v223 = stablehlo.divide %v222, %v214 : tensor<128x32x4x4xf32>
    %v224 = stablehlo.add %v223, %v215 : tensor<128x32x4x4xf32>
    %v225 = stablehlo.rsqrt %v224 : tensor<128x32x4x4xf32>
    %v226 = stablehlo.multiply %v219, %v225 : tensor<128x32x4x4xf32>
    %v227 = stablehlo.broadcast_in_dim %g7, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v228 = stablehlo.broadcast_in_dim %bt7, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v229 = stablehlo.multiply %v226, %v227 : tensor<128x32x4x4xf32>
    %v230 = stablehlo.add %v229, %v228 : tensor<128x32x4x4xf32>
    %v231 = stablehlo.reshape %v230 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v232 = stablehlo.reshape %v231 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v233 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v234 = stablehlo.maximum %v232, %v233 : tensor<128x32x4x4xf32>
    %v235 = stablehlo.reshape %v234 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v236 = stablehlo.reshape %v235 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v237 = stablehlo.convert %v236 : (tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xbf16>
    %v238 = stablehlo.convert %W8 : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xbf16>
    %v239 = stablehlo.convolution(%v237, %v238)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xbf16>, tensor<32x32x3x3xbf16>) -> tensor<128x32x4x4xbf16>
    %v240 = stablehlo.convert %v239 : (tensor<128x32x4x4xbf16>) -> tensor<128x32x4x4xf32>
    %v241 = stablehlo.broadcast_in_dim %cb8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v242 = stablehlo.add %v240, %v241 : tensor<128x32x4x4xf32>
    %v243 = stablehlo.reshape %v242 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v244 = stablehlo.reshape %v243 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v245 = stablehlo.constant dense<0.0> : tensor<f32>
    %v246 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v247 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v248 = stablehlo.reduce(%v244 init: %v245) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v249 = stablehlo.broadcast_in_dim %v248, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v250 = stablehlo.divide %v249, %v246 : tensor<128x32x4x4xf32>
    %v251 = stablehlo.subtract %v244, %v250 : tensor<128x32x4x4xf32>
    %v252 = stablehlo.multiply %v251, %v251 : tensor<128x32x4x4xf32>
    %v253 = stablehlo.reduce(%v252 init: %v245) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v254 = stablehlo.broadcast_in_dim %v253, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v255 = stablehlo.divide %v254, %v246 : tensor<128x32x4x4xf32>
    %v256 = stablehlo.add %v255, %v247 : tensor<128x32x4x4xf32>
    %v257 = stablehlo.rsqrt %v256 : tensor<128x32x4x4xf32>
    %v258 = stablehlo.multiply %v251, %v257 : tensor<128x32x4x4xf32>
    %v259 = stablehlo.broadcast_in_dim %g8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v260 = stablehlo.broadcast_in_dim %bt8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v261 = stablehlo.multiply %v258, %v259 : tensor<128x32x4x4xf32>
    %v262 = stablehlo.add %v261, %v260 : tensor<128x32x4x4xf32>
    %v263 = stablehlo.reshape %v262 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v264 = stablehlo.reshape %v263 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v265 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v266 = stablehlo.maximum %v264, %v265 : tensor<128x32x4x4xf32>
    %v267 = stablehlo.reshape %v266 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v268 = stablehlo.reshape %v267 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v269 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v270 = "stablehlo.reduce_window"(%v268, %v269) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32x2x2xf32>
    %v271 = stablehlo.reshape %v270 : (tensor<128x32x2x2xf32>) -> tensor<128x128xf32>
    %v272 = stablehlo.dot_general %v271, %W9, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x512xf32>) -> tensor<128x512xf32>
    %v273 = stablehlo.broadcast_in_dim %b9, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v274 = stablehlo.add %v272, %v273 : tensor<128x512xf32>
    %v275 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v276 = stablehlo.maximum %v274, %v275 : tensor<128x512xf32>
    %v277 = stablehlo.dot_general %v276, %Wa, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %v278 = stablehlo.broadcast_in_dim %ba, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v279 = stablehlo.add %v277, %v278 : tensor<128x512xf32>
    %v280 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v281 = stablehlo.maximum %v279, %v280 : tensor<128x512xf32>
    %v282 = stablehlo.dot_general %v281, %Wb, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x10xf32>) -> tensor<128x10xf32>
    %v283 = stablehlo.broadcast_in_dim %bb, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v284 = stablehlo.add %v282, %v283 : tensor<128x10xf32>
    %v285 = stablehlo.reshape %v284 : (tensor<128x10xf32>) -> tensor<128x1x10xf32>
    %v286 = stablehlo.constant dense<0.0> : tensor<f32>
    %v287 = stablehlo.exponential %v285 : tensor<128x1x10xf32>
    %v288 = stablehlo.reduce(%v287 init: %v286) applies stablehlo.add across dimensions = [2] : (tensor<128x1x10xf32>, tensor<f32>) -> tensor<128x1xf32>
    %v289 = stablehlo.broadcast_in_dim %v288, dims = [0, 1] : (tensor<128x1xf32>) -> tensor<128x1x10xf32>
    %v290 = stablehlo.divide %v287, %v289 : tensor<128x1x10xf32>
    %v291 = stablehlo.reshape %v290 : (tensor<128x1x10xf32>) -> tensor<128x10xf32>
    %v292 = stablehlo.subtract %v291, %onehot : tensor<128x10xf32>
    %v293 = stablehlo.constant dense<0.0078125> : tensor<128x10xf32>
    %v294 = stablehlo.multiply %v292, %v293 : tensor<128x10xf32>
    // ── report-only scalar loss (NOT pretty(AST): the kit has no rank-0 loss op; it
    //    feeds no parameter, only the driver's progress line) ──
    %llog = stablehlo.log %v291 : tensor<128x10xf32>
    %ohll = stablehlo.multiply %onehot, %llog : tensor<128x10xf32>
    %csum = stablehlo.reduce(%ohll init: %lzero) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<f32>
    %cneg = stablehlo.negate %csum : tensor<f32>
    %lbf = stablehlo.constant dense<128.0> : tensor<f32>
    %loss = stablehlo.divide %cneg, %lbf : tensor<f32>
    %v295 = stablehlo.reshape %v294 : (tensor<128x10xf32>) -> tensor<128x1x10xf32>
    %v296 = stablehlo.dot_general %v295, %Wb, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x10xf32>, tensor<512x10xf32>) -> tensor<128x1x512xf32>
    %v297 = stablehlo.reshape %v296 : (tensor<128x1x512xf32>) -> tensor<128x512xf32>
    %v298 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v299 = stablehlo.compare GT, %v279, %v298 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v300 = stablehlo.select %v299, %v297, %v298 : tensor<128x512xi1>, tensor<128x512xf32>
    %v301 = stablehlo.reshape %v300 : (tensor<128x512xf32>) -> tensor<128x1x512xf32>
    %v302 = stablehlo.dot_general %v301, %Wa, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x512xf32>, tensor<512x512xf32>) -> tensor<128x1x512xf32>
    %v303 = stablehlo.reshape %v302 : (tensor<128x1x512xf32>) -> tensor<128x512xf32>
    %v304 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v305 = stablehlo.compare GT, %v274, %v304 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v306 = stablehlo.select %v305, %v303, %v304 : tensor<128x512xi1>, tensor<128x512xf32>
    %v307 = stablehlo.reshape %v306 : (tensor<128x512xf32>) -> tensor<128x1x512xf32>
    %v308 = stablehlo.dot_general %v307, %W9, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x512xf32>, tensor<128x512xf32>) -> tensor<128x1x128xf32>
    %v309 = stablehlo.reshape %v308 : (tensor<128x1x128xf32>) -> tensor<128x128xf32>
    %v310 = stablehlo.reshape %v267 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v311 = stablehlo.reshape %v309 : (tensor<128x128xf32>) -> tensor<128x32x2x2xf32>
    %v312 = stablehlo.constant dense<0.0> : tensor<f32>
    %v313 = "stablehlo.select_and_scatter"(%v310, %v311, %v312) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x4x4xf32>, tensor<128x32x2x2xf32>, tensor<f32>) -> tensor<128x32x4x4xf32>
    %v314 = stablehlo.reshape %v313 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v315 = stablehlo.reshape %v314 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v316 = stablehlo.reshape %v263 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v317 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v318 = stablehlo.compare GT, %v316, %v317 : (tensor<128x32x4x4xf32>, tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xi1>
    %v319 = stablehlo.select %v318, %v315, %v317 : tensor<128x32x4x4xi1>, tensor<128x32x4x4xf32>
    %v320 = stablehlo.reshape %v319 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v321 = stablehlo.reshape %v320 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v322 = stablehlo.reshape %v243 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v323 = stablehlo.constant dense<0.0> : tensor<f32>
    %v324 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v325 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v326 = stablehlo.reduce(%v322 init: %v323) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v327 = stablehlo.broadcast_in_dim %v326, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v328 = stablehlo.divide %v327, %v324 : tensor<128x32x4x4xf32>
    %v329 = stablehlo.subtract %v322, %v328 : tensor<128x32x4x4xf32>
    %v330 = stablehlo.multiply %v329, %v329 : tensor<128x32x4x4xf32>
    %v331 = stablehlo.reduce(%v330 init: %v323) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v332 = stablehlo.broadcast_in_dim %v331, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v333 = stablehlo.divide %v332, %v324 : tensor<128x32x4x4xf32>
    %v334 = stablehlo.add %v333, %v325 : tensor<128x32x4x4xf32>
    %v335 = stablehlo.rsqrt %v334 : tensor<128x32x4x4xf32>
    %v336 = stablehlo.multiply %v329, %v335 : tensor<128x32x4x4xf32>
    %v337 = stablehlo.broadcast_in_dim %g8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v338 = stablehlo.multiply %v337, %v321 : tensor<128x32x4x4xf32>
    %v339 = stablehlo.reduce(%v338 init: %v323) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v340 = stablehlo.broadcast_in_dim %v339, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v341 = stablehlo.multiply %v336, %v338 : tensor<128x32x4x4xf32>
    %v342 = stablehlo.reduce(%v341 init: %v323) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v343 = stablehlo.broadcast_in_dim %v342, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v344 = stablehlo.multiply %v338, %v324 : tensor<128x32x4x4xf32>
    %v345 = stablehlo.subtract %v344, %v340 : tensor<128x32x4x4xf32>
    %v346 = stablehlo.multiply %v336, %v343 : tensor<128x32x4x4xf32>
    %v347 = stablehlo.subtract %v345, %v346 : tensor<128x32x4x4xf32>
    %v348 = stablehlo.divide %v335, %v324 : tensor<128x32x4x4xf32>
    %v349 = stablehlo.multiply %v348, %v347 : tensor<128x32x4x4xf32>
    %v350 = stablehlo.reshape %v349 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v351 = stablehlo.reshape %v350 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v352 = stablehlo.reverse %W8, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v353 = stablehlo.transpose %v352, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v354 = stablehlo.convert %v351 : (tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xbf16>
    %v355 = stablehlo.convert %v353 : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xbf16>
    %v356 = stablehlo.convolution(%v354, %v355)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xbf16>, tensor<32x32x3x3xbf16>) -> tensor<128x32x4x4xbf16>
    %v357 = stablehlo.convert %v356 : (tensor<128x32x4x4xbf16>) -> tensor<128x32x4x4xf32>
    %v358 = stablehlo.reshape %v357 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v359 = stablehlo.reshape %v358 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v360 = stablehlo.reshape %v231 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v361 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v362 = stablehlo.compare GT, %v360, %v361 : (tensor<128x32x4x4xf32>, tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xi1>
    %v363 = stablehlo.select %v362, %v359, %v361 : tensor<128x32x4x4xi1>, tensor<128x32x4x4xf32>
    %v364 = stablehlo.reshape %v363 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v365 = stablehlo.reshape %v364 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v366 = stablehlo.reshape %v211 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v367 = stablehlo.constant dense<0.0> : tensor<f32>
    %v368 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v369 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v370 = stablehlo.reduce(%v366 init: %v367) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v371 = stablehlo.broadcast_in_dim %v370, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v372 = stablehlo.divide %v371, %v368 : tensor<128x32x4x4xf32>
    %v373 = stablehlo.subtract %v366, %v372 : tensor<128x32x4x4xf32>
    %v374 = stablehlo.multiply %v373, %v373 : tensor<128x32x4x4xf32>
    %v375 = stablehlo.reduce(%v374 init: %v367) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v376 = stablehlo.broadcast_in_dim %v375, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v377 = stablehlo.divide %v376, %v368 : tensor<128x32x4x4xf32>
    %v378 = stablehlo.add %v377, %v369 : tensor<128x32x4x4xf32>
    %v379 = stablehlo.rsqrt %v378 : tensor<128x32x4x4xf32>
    %v380 = stablehlo.multiply %v373, %v379 : tensor<128x32x4x4xf32>
    %v381 = stablehlo.broadcast_in_dim %g7, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v382 = stablehlo.multiply %v381, %v365 : tensor<128x32x4x4xf32>
    %v383 = stablehlo.reduce(%v382 init: %v367) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v384 = stablehlo.broadcast_in_dim %v383, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v385 = stablehlo.multiply %v380, %v382 : tensor<128x32x4x4xf32>
    %v386 = stablehlo.reduce(%v385 init: %v367) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v387 = stablehlo.broadcast_in_dim %v386, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v388 = stablehlo.multiply %v382, %v368 : tensor<128x32x4x4xf32>
    %v389 = stablehlo.subtract %v388, %v384 : tensor<128x32x4x4xf32>
    %v390 = stablehlo.multiply %v380, %v387 : tensor<128x32x4x4xf32>
    %v391 = stablehlo.subtract %v389, %v390 : tensor<128x32x4x4xf32>
    %v392 = stablehlo.divide %v379, %v368 : tensor<128x32x4x4xf32>
    %v393 = stablehlo.multiply %v392, %v391 : tensor<128x32x4x4xf32>
    %v394 = stablehlo.reshape %v393 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v395 = stablehlo.reshape %v394 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v396 = stablehlo.reverse %W7, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v397 = stablehlo.transpose %v396, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v398 = stablehlo.convert %v395 : (tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xbf16>
    %v399 = stablehlo.convert %v397 : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xbf16>
    %v400 = stablehlo.convolution(%v398, %v399)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xbf16>, tensor<32x32x3x3xbf16>) -> tensor<128x32x4x4xbf16>
    %v401 = stablehlo.convert %v400 : (tensor<128x32x4x4xbf16>) -> tensor<128x32x4x4xf32>
    %v402 = stablehlo.reshape %v401 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v403 = stablehlo.reshape %v199 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v404 = stablehlo.reshape %v402 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v405 = stablehlo.constant dense<0.0> : tensor<f32>
    %v406 = "stablehlo.select_and_scatter"(%v403, %v404, %v405) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x8x8xf32>, tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32x8x8xf32>
    %v407 = stablehlo.reshape %v406 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v408 = stablehlo.reshape %v407 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v409 = stablehlo.reshape %v195 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v410 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v411 = stablehlo.compare GT, %v409, %v410 : (tensor<128x32x8x8xf32>, tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xi1>
    %v412 = stablehlo.select %v411, %v408, %v410 : tensor<128x32x8x8xi1>, tensor<128x32x8x8xf32>
    %v413 = stablehlo.reshape %v412 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v414 = stablehlo.reshape %v413 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v415 = stablehlo.reshape %v175 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v416 = stablehlo.constant dense<0.0> : tensor<f32>
    %v417 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v418 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v419 = stablehlo.reduce(%v415 init: %v416) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v420 = stablehlo.broadcast_in_dim %v419, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v421 = stablehlo.divide %v420, %v417 : tensor<128x32x8x8xf32>
    %v422 = stablehlo.subtract %v415, %v421 : tensor<128x32x8x8xf32>
    %v423 = stablehlo.multiply %v422, %v422 : tensor<128x32x8x8xf32>
    %v424 = stablehlo.reduce(%v423 init: %v416) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v425 = stablehlo.broadcast_in_dim %v424, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v426 = stablehlo.divide %v425, %v417 : tensor<128x32x8x8xf32>
    %v427 = stablehlo.add %v426, %v418 : tensor<128x32x8x8xf32>
    %v428 = stablehlo.rsqrt %v427 : tensor<128x32x8x8xf32>
    %v429 = stablehlo.multiply %v422, %v428 : tensor<128x32x8x8xf32>
    %v430 = stablehlo.broadcast_in_dim %g6, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v431 = stablehlo.multiply %v430, %v414 : tensor<128x32x8x8xf32>
    %v432 = stablehlo.reduce(%v431 init: %v416) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v433 = stablehlo.broadcast_in_dim %v432, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v434 = stablehlo.multiply %v429, %v431 : tensor<128x32x8x8xf32>
    %v435 = stablehlo.reduce(%v434 init: %v416) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v436 = stablehlo.broadcast_in_dim %v435, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v437 = stablehlo.multiply %v431, %v417 : tensor<128x32x8x8xf32>
    %v438 = stablehlo.subtract %v437, %v433 : tensor<128x32x8x8xf32>
    %v439 = stablehlo.multiply %v429, %v436 : tensor<128x32x8x8xf32>
    %v440 = stablehlo.subtract %v438, %v439 : tensor<128x32x8x8xf32>
    %v441 = stablehlo.divide %v428, %v417 : tensor<128x32x8x8xf32>
    %v442 = stablehlo.multiply %v441, %v440 : tensor<128x32x8x8xf32>
    %v443 = stablehlo.reshape %v442 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v444 = stablehlo.reshape %v443 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v445 = stablehlo.reverse %W6, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v446 = stablehlo.transpose %v445, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v447 = stablehlo.convert %v444 : (tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xbf16>
    %v448 = stablehlo.convert %v446 : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xbf16>
    %v449 = stablehlo.convolution(%v447, %v448)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xbf16>, tensor<32x32x3x3xbf16>) -> tensor<128x32x8x8xbf16>
    %v450 = stablehlo.convert %v449 : (tensor<128x32x8x8xbf16>) -> tensor<128x32x8x8xf32>
    %v451 = stablehlo.reshape %v450 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v452 = stablehlo.reshape %v451 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v453 = stablehlo.reshape %v163 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v454 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v455 = stablehlo.compare GT, %v453, %v454 : (tensor<128x32x8x8xf32>, tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xi1>
    %v456 = stablehlo.select %v455, %v452, %v454 : tensor<128x32x8x8xi1>, tensor<128x32x8x8xf32>
    %v457 = stablehlo.reshape %v456 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v458 = stablehlo.reshape %v457 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v459 = stablehlo.reshape %v143 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v460 = stablehlo.constant dense<0.0> : tensor<f32>
    %v461 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v462 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v463 = stablehlo.reduce(%v459 init: %v460) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v464 = stablehlo.broadcast_in_dim %v463, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v465 = stablehlo.divide %v464, %v461 : tensor<128x32x8x8xf32>
    %v466 = stablehlo.subtract %v459, %v465 : tensor<128x32x8x8xf32>
    %v467 = stablehlo.multiply %v466, %v466 : tensor<128x32x8x8xf32>
    %v468 = stablehlo.reduce(%v467 init: %v460) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v469 = stablehlo.broadcast_in_dim %v468, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v470 = stablehlo.divide %v469, %v461 : tensor<128x32x8x8xf32>
    %v471 = stablehlo.add %v470, %v462 : tensor<128x32x8x8xf32>
    %v472 = stablehlo.rsqrt %v471 : tensor<128x32x8x8xf32>
    %v473 = stablehlo.multiply %v466, %v472 : tensor<128x32x8x8xf32>
    %v474 = stablehlo.broadcast_in_dim %g5, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v475 = stablehlo.multiply %v474, %v458 : tensor<128x32x8x8xf32>
    %v476 = stablehlo.reduce(%v475 init: %v460) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v477 = stablehlo.broadcast_in_dim %v476, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v478 = stablehlo.multiply %v473, %v475 : tensor<128x32x8x8xf32>
    %v479 = stablehlo.reduce(%v478 init: %v460) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v480 = stablehlo.broadcast_in_dim %v479, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v481 = stablehlo.multiply %v475, %v461 : tensor<128x32x8x8xf32>
    %v482 = stablehlo.subtract %v481, %v477 : tensor<128x32x8x8xf32>
    %v483 = stablehlo.multiply %v473, %v480 : tensor<128x32x8x8xf32>
    %v484 = stablehlo.subtract %v482, %v483 : tensor<128x32x8x8xf32>
    %v485 = stablehlo.divide %v472, %v461 : tensor<128x32x8x8xf32>
    %v486 = stablehlo.multiply %v485, %v484 : tensor<128x32x8x8xf32>
    %v487 = stablehlo.reshape %v486 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v488 = stablehlo.reshape %v487 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v489 = stablehlo.reverse %W5, dims = [2, 3] : tensor<32x16x3x3xf32>
    %v490 = stablehlo.transpose %v489, dims = [1, 0, 2, 3] : (tensor<32x16x3x3xf32>) -> tensor<16x32x3x3xf32>
    %v491 = stablehlo.convert %v488 : (tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xbf16>
    %v492 = stablehlo.convert %v490 : (tensor<16x32x3x3xf32>) -> tensor<16x32x3x3xbf16>
    %v493 = stablehlo.convolution(%v491, %v492)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xbf16>, tensor<16x32x3x3xbf16>) -> tensor<128x16x8x8xbf16>
    %v494 = stablehlo.convert %v493 : (tensor<128x16x8x8xbf16>) -> tensor<128x16x8x8xf32>
    %v495 = stablehlo.reshape %v494 : (tensor<128x16x8x8xf32>) -> tensor<128x1024xf32>
    %v496 = stablehlo.reshape %v131 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v497 = stablehlo.reshape %v495 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v498 = stablehlo.constant dense<0.0> : tensor<f32>
    %v499 = "stablehlo.select_and_scatter"(%v496, %v497, %v498) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x16x16xf32>, tensor<128x16x8x8xf32>, tensor<f32>) -> tensor<128x16x16x16xf32>
    %v500 = stablehlo.reshape %v499 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v501 = stablehlo.reshape %v500 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v502 = stablehlo.reshape %v127 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v503 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v504 = stablehlo.compare GT, %v502, %v503 : (tensor<128x16x16x16xf32>, tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xi1>
    %v505 = stablehlo.select %v504, %v501, %v503 : tensor<128x16x16x16xi1>, tensor<128x16x16x16xf32>
    %v506 = stablehlo.reshape %v505 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v507 = stablehlo.reshape %v506 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v508 = stablehlo.reshape %v107 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v509 = stablehlo.constant dense<0.0> : tensor<f32>
    %v510 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v511 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v512 = stablehlo.reduce(%v508 init: %v509) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v513 = stablehlo.broadcast_in_dim %v512, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v514 = stablehlo.divide %v513, %v510 : tensor<128x16x16x16xf32>
    %v515 = stablehlo.subtract %v508, %v514 : tensor<128x16x16x16xf32>
    %v516 = stablehlo.multiply %v515, %v515 : tensor<128x16x16x16xf32>
    %v517 = stablehlo.reduce(%v516 init: %v509) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v518 = stablehlo.broadcast_in_dim %v517, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v519 = stablehlo.divide %v518, %v510 : tensor<128x16x16x16xf32>
    %v520 = stablehlo.add %v519, %v511 : tensor<128x16x16x16xf32>
    %v521 = stablehlo.rsqrt %v520 : tensor<128x16x16x16xf32>
    %v522 = stablehlo.multiply %v515, %v521 : tensor<128x16x16x16xf32>
    %v523 = stablehlo.broadcast_in_dim %g4, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v524 = stablehlo.multiply %v523, %v507 : tensor<128x16x16x16xf32>
    %v525 = stablehlo.reduce(%v524 init: %v509) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v526 = stablehlo.broadcast_in_dim %v525, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v527 = stablehlo.multiply %v522, %v524 : tensor<128x16x16x16xf32>
    %v528 = stablehlo.reduce(%v527 init: %v509) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v529 = stablehlo.broadcast_in_dim %v528, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v530 = stablehlo.multiply %v524, %v510 : tensor<128x16x16x16xf32>
    %v531 = stablehlo.subtract %v530, %v526 : tensor<128x16x16x16xf32>
    %v532 = stablehlo.multiply %v522, %v529 : tensor<128x16x16x16xf32>
    %v533 = stablehlo.subtract %v531, %v532 : tensor<128x16x16x16xf32>
    %v534 = stablehlo.divide %v521, %v510 : tensor<128x16x16x16xf32>
    %v535 = stablehlo.multiply %v534, %v533 : tensor<128x16x16x16xf32>
    %v536 = stablehlo.reshape %v535 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v537 = stablehlo.reshape %v536 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v538 = stablehlo.reverse %W4, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v539 = stablehlo.transpose %v538, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v540 = stablehlo.convert %v537 : (tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xbf16>
    %v541 = stablehlo.convert %v539 : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xbf16>
    %v542 = stablehlo.convolution(%v540, %v541)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xbf16>, tensor<16x16x3x3xbf16>) -> tensor<128x16x16x16xbf16>
    %v543 = stablehlo.convert %v542 : (tensor<128x16x16x16xbf16>) -> tensor<128x16x16x16xf32>
    %v544 = stablehlo.reshape %v543 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v545 = stablehlo.reshape %v544 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v546 = stablehlo.reshape %v95 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v547 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v548 = stablehlo.compare GT, %v546, %v547 : (tensor<128x16x16x16xf32>, tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xi1>
    %v549 = stablehlo.select %v548, %v545, %v547 : tensor<128x16x16x16xi1>, tensor<128x16x16x16xf32>
    %v550 = stablehlo.reshape %v549 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v551 = stablehlo.reshape %v550 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v552 = stablehlo.reshape %v75 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v553 = stablehlo.constant dense<0.0> : tensor<f32>
    %v554 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v555 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v556 = stablehlo.reduce(%v552 init: %v553) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v557 = stablehlo.broadcast_in_dim %v556, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v558 = stablehlo.divide %v557, %v554 : tensor<128x16x16x16xf32>
    %v559 = stablehlo.subtract %v552, %v558 : tensor<128x16x16x16xf32>
    %v560 = stablehlo.multiply %v559, %v559 : tensor<128x16x16x16xf32>
    %v561 = stablehlo.reduce(%v560 init: %v553) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v562 = stablehlo.broadcast_in_dim %v561, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v563 = stablehlo.divide %v562, %v554 : tensor<128x16x16x16xf32>
    %v564 = stablehlo.add %v563, %v555 : tensor<128x16x16x16xf32>
    %v565 = stablehlo.rsqrt %v564 : tensor<128x16x16x16xf32>
    %v566 = stablehlo.multiply %v559, %v565 : tensor<128x16x16x16xf32>
    %v567 = stablehlo.broadcast_in_dim %g3, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v568 = stablehlo.multiply %v567, %v551 : tensor<128x16x16x16xf32>
    %v569 = stablehlo.reduce(%v568 init: %v553) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v570 = stablehlo.broadcast_in_dim %v569, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v571 = stablehlo.multiply %v566, %v568 : tensor<128x16x16x16xf32>
    %v572 = stablehlo.reduce(%v571 init: %v553) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v573 = stablehlo.broadcast_in_dim %v572, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v574 = stablehlo.multiply %v568, %v554 : tensor<128x16x16x16xf32>
    %v575 = stablehlo.subtract %v574, %v570 : tensor<128x16x16x16xf32>
    %v576 = stablehlo.multiply %v566, %v573 : tensor<128x16x16x16xf32>
    %v577 = stablehlo.subtract %v575, %v576 : tensor<128x16x16x16xf32>
    %v578 = stablehlo.divide %v565, %v554 : tensor<128x16x16x16xf32>
    %v579 = stablehlo.multiply %v578, %v577 : tensor<128x16x16x16xf32>
    %v580 = stablehlo.reshape %v579 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v581 = stablehlo.reshape %v580 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v582 = stablehlo.reverse %W3, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v583 = stablehlo.transpose %v582, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v584 = stablehlo.convert %v581 : (tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xbf16>
    %v585 = stablehlo.convert %v583 : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xbf16>
    %v586 = stablehlo.convolution(%v584, %v585)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xbf16>, tensor<16x16x3x3xbf16>) -> tensor<128x16x16x16xbf16>
    %v587 = stablehlo.convert %v586 : (tensor<128x16x16x16xbf16>) -> tensor<128x16x16x16xf32>
    %v588 = stablehlo.reshape %v587 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v589 = stablehlo.reshape %v63 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v590 = stablehlo.reshape %v588 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v591 = stablehlo.constant dense<0.0> : tensor<f32>
    %v592 = "stablehlo.select_and_scatter"(%v589, %v590, %v591) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x32x32xf32>, tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x32x32xf32>
    %v593 = stablehlo.reshape %v592 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v594 = stablehlo.reshape %v593 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v595 = stablehlo.reshape %v59 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v596 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v597 = stablehlo.compare GT, %v595, %v596 : (tensor<128x16x32x32xf32>, tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xi1>
    %v598 = stablehlo.select %v597, %v594, %v596 : tensor<128x16x32x32xi1>, tensor<128x16x32x32xf32>
    %v599 = stablehlo.reshape %v598 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v600 = stablehlo.reshape %v599 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v601 = stablehlo.reshape %v39 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v602 = stablehlo.constant dense<0.0> : tensor<f32>
    %v603 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v604 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v605 = stablehlo.reduce(%v601 init: %v602) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v606 = stablehlo.broadcast_in_dim %v605, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v607 = stablehlo.divide %v606, %v603 : tensor<128x16x32x32xf32>
    %v608 = stablehlo.subtract %v601, %v607 : tensor<128x16x32x32xf32>
    %v609 = stablehlo.multiply %v608, %v608 : tensor<128x16x32x32xf32>
    %v610 = stablehlo.reduce(%v609 init: %v602) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v611 = stablehlo.broadcast_in_dim %v610, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v612 = stablehlo.divide %v611, %v603 : tensor<128x16x32x32xf32>
    %v613 = stablehlo.add %v612, %v604 : tensor<128x16x32x32xf32>
    %v614 = stablehlo.rsqrt %v613 : tensor<128x16x32x32xf32>
    %v615 = stablehlo.multiply %v608, %v614 : tensor<128x16x32x32xf32>
    %v616 = stablehlo.broadcast_in_dim %g2, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v617 = stablehlo.multiply %v616, %v600 : tensor<128x16x32x32xf32>
    %v618 = stablehlo.reduce(%v617 init: %v602) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v619 = stablehlo.broadcast_in_dim %v618, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v620 = stablehlo.multiply %v615, %v617 : tensor<128x16x32x32xf32>
    %v621 = stablehlo.reduce(%v620 init: %v602) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v622 = stablehlo.broadcast_in_dim %v621, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v623 = stablehlo.multiply %v617, %v603 : tensor<128x16x32x32xf32>
    %v624 = stablehlo.subtract %v623, %v619 : tensor<128x16x32x32xf32>
    %v625 = stablehlo.multiply %v615, %v622 : tensor<128x16x32x32xf32>
    %v626 = stablehlo.subtract %v624, %v625 : tensor<128x16x32x32xf32>
    %v627 = stablehlo.divide %v614, %v603 : tensor<128x16x32x32xf32>
    %v628 = stablehlo.multiply %v627, %v626 : tensor<128x16x32x32xf32>
    %v629 = stablehlo.reshape %v628 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v630 = stablehlo.reshape %v629 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v631 = stablehlo.reverse %W2, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v632 = stablehlo.transpose %v631, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v633 = stablehlo.convert %v630 : (tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xbf16>
    %v634 = stablehlo.convert %v632 : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xbf16>
    %v635 = stablehlo.convolution(%v633, %v634)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xbf16>, tensor<16x16x3x3xbf16>) -> tensor<128x16x32x32xbf16>
    %v636 = stablehlo.convert %v635 : (tensor<128x16x32x32xbf16>) -> tensor<128x16x32x32xf32>
    %v637 = stablehlo.reshape %v636 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v638 = stablehlo.reshape %v637 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v639 = stablehlo.reshape %v27 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v640 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v641 = stablehlo.compare GT, %v639, %v640 : (tensor<128x16x32x32xf32>, tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xi1>
    %v642 = stablehlo.select %v641, %v638, %v640 : tensor<128x16x32x32xi1>, tensor<128x16x32x32xf32>
    %v643 = stablehlo.reshape %v642 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v644 = stablehlo.reshape %v643 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v645 = stablehlo.reshape %v7 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v646 = stablehlo.constant dense<0.0> : tensor<f32>
    %v647 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v648 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v649 = stablehlo.reduce(%v645 init: %v646) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v650 = stablehlo.broadcast_in_dim %v649, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v651 = stablehlo.divide %v650, %v647 : tensor<128x16x32x32xf32>
    %v652 = stablehlo.subtract %v645, %v651 : tensor<128x16x32x32xf32>
    %v653 = stablehlo.multiply %v652, %v652 : tensor<128x16x32x32xf32>
    %v654 = stablehlo.reduce(%v653 init: %v646) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v655 = stablehlo.broadcast_in_dim %v654, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v656 = stablehlo.divide %v655, %v647 : tensor<128x16x32x32xf32>
    %v657 = stablehlo.add %v656, %v648 : tensor<128x16x32x32xf32>
    %v658 = stablehlo.rsqrt %v657 : tensor<128x16x32x32xf32>
    %v659 = stablehlo.multiply %v652, %v658 : tensor<128x16x32x32xf32>
    %v660 = stablehlo.broadcast_in_dim %g1, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v661 = stablehlo.multiply %v660, %v644 : tensor<128x16x32x32xf32>
    %v662 = stablehlo.reduce(%v661 init: %v646) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v663 = stablehlo.broadcast_in_dim %v662, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v664 = stablehlo.multiply %v659, %v661 : tensor<128x16x32x32xf32>
    %v665 = stablehlo.reduce(%v664 init: %v646) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v666 = stablehlo.broadcast_in_dim %v665, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v667 = stablehlo.multiply %v661, %v647 : tensor<128x16x32x32xf32>
    %v668 = stablehlo.subtract %v667, %v663 : tensor<128x16x32x32xf32>
    %v669 = stablehlo.multiply %v659, %v666 : tensor<128x16x32x32xf32>
    %v670 = stablehlo.subtract %v668, %v669 : tensor<128x16x32x32xf32>
    %v671 = stablehlo.divide %v658, %v647 : tensor<128x16x32x32xf32>
    %v672 = stablehlo.multiply %v671, %v670 : tensor<128x16x32x32xf32>
    %v673 = stablehlo.reshape %v672 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v674 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v675 = stablehlo.reshape %v673 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v676 = stablehlo.transpose %v674, dims = [1, 0, 2, 3] : (tensor<128x3x32x32xf32>) -> tensor<3x128x32x32xf32>
    %v677 = stablehlo.transpose %v675, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v678 = stablehlo.convert %v676 : (tensor<3x128x32x32xf32>) -> tensor<3x128x32x32xbf16>
    %v679 = stablehlo.convert %v677 : (tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xbf16>
    %v680 = stablehlo.convolution(%v678, %v679)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x128x32x32xbf16>, tensor<16x128x32x32xbf16>) -> tensor<3x16x3x3xbf16>
    %v681 = stablehlo.convert %v680 : (tensor<3x16x3x3xbf16>) -> tensor<3x16x3x3xf32>
    %v682 = stablehlo.transpose %v681, dims = [1, 0, 2, 3] : (tensor<3x16x3x3xf32>) -> tensor<16x3x3x3xf32>
    %v683 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v684 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v685 = stablehlo.multiply %v683, %W1m : tensor<16x3x3x3xf32>
    %v686 = stablehlo.multiply %v684, %v682 : tensor<16x3x3x3xf32>
    %v687 = stablehlo.add %v685, %v686 : tensor<16x3x3x3xf32>
    %v688 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v689 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v690 = stablehlo.multiply %v688, %W1v : tensor<16x3x3x3xf32>
    %v691 = stablehlo.multiply %v682, %v682 : tensor<16x3x3x3xf32>
    %v692 = stablehlo.multiply %v689, %v691 : tensor<16x3x3x3xf32>
    %v693 = stablehlo.add %v690, %v692 : tensor<16x3x3x3xf32>
    %v694 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v695 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v696 = stablehlo.divide %v687, %v694 : tensor<16x3x3x3xf32>
    %v697 = stablehlo.divide %v693, %v695 : tensor<16x3x3x3xf32>
    %v698 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v699 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v700 = stablehlo.sqrt %v697 : tensor<16x3x3x3xf32>
    %v701 = stablehlo.add %v700, %v699 : tensor<16x3x3x3xf32>
    %v702 = stablehlo.divide %v696, %v701 : tensor<16x3x3x3xf32>
    %v703 = stablehlo.multiply %v698, %v702 : tensor<16x3x3x3xf32>
    %v704 = stablehlo.subtract %W1, %v703 : tensor<16x3x3x3xf32>
    %v705 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v706 = stablehlo.multiply %v705, %v698 : tensor<16x3x3x3xf32>
    %v707 = stablehlo.multiply %v706, %W1 : tensor<16x3x3x3xf32>
    %v708 = stablehlo.subtract %v704, %v707 : tensor<16x3x3x3xf32>
    %v709 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v710 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v711 = stablehlo.multiply %v709, %W1m : tensor<16x3x3x3xf32>
    %v712 = stablehlo.multiply %v710, %v682 : tensor<16x3x3x3xf32>
    %v713 = stablehlo.add %v711, %v712 : tensor<16x3x3x3xf32>
    %v714 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v715 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v716 = stablehlo.multiply %v714, %W1v : tensor<16x3x3x3xf32>
    %v717 = stablehlo.multiply %v682, %v682 : tensor<16x3x3x3xf32>
    %v718 = stablehlo.multiply %v715, %v717 : tensor<16x3x3x3xf32>
    %v719 = stablehlo.add %v716, %v718 : tensor<16x3x3x3xf32>
    %v720 = stablehlo.reshape %v673 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v721 = stablehlo.constant dense<0.0> : tensor<f32>
    %v722 = stablehlo.reduce(%v720 init: %v721) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v723 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v724 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v725 = stablehlo.multiply %v723, %cb1m : tensor<16xf32>
    %v726 = stablehlo.multiply %v724, %v722 : tensor<16xf32>
    %v727 = stablehlo.add %v725, %v726 : tensor<16xf32>
    %v728 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v729 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v730 = stablehlo.multiply %v728, %cb1v : tensor<16xf32>
    %v731 = stablehlo.multiply %v722, %v722 : tensor<16xf32>
    %v732 = stablehlo.multiply %v729, %v731 : tensor<16xf32>
    %v733 = stablehlo.add %v730, %v732 : tensor<16xf32>
    %v734 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v735 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v736 = stablehlo.divide %v727, %v734 : tensor<16xf32>
    %v737 = stablehlo.divide %v733, %v735 : tensor<16xf32>
    %v738 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v739 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v740 = stablehlo.sqrt %v737 : tensor<16xf32>
    %v741 = stablehlo.add %v740, %v739 : tensor<16xf32>
    %v742 = stablehlo.divide %v736, %v741 : tensor<16xf32>
    %v743 = stablehlo.multiply %v738, %v742 : tensor<16xf32>
    %v744 = stablehlo.subtract %cb1, %v743 : tensor<16xf32>
    %v745 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v746 = stablehlo.multiply %v745, %v738 : tensor<16xf32>
    %v747 = stablehlo.multiply %v746, %cb1 : tensor<16xf32>
    %v748 = stablehlo.subtract %v744, %v747 : tensor<16xf32>
    %v749 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v750 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v751 = stablehlo.multiply %v749, %cb1m : tensor<16xf32>
    %v752 = stablehlo.multiply %v750, %v722 : tensor<16xf32>
    %v753 = stablehlo.add %v751, %v752 : tensor<16xf32>
    %v754 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v755 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v756 = stablehlo.multiply %v754, %cb1v : tensor<16xf32>
    %v757 = stablehlo.multiply %v722, %v722 : tensor<16xf32>
    %v758 = stablehlo.multiply %v755, %v757 : tensor<16xf32>
    %v759 = stablehlo.add %v756, %v758 : tensor<16xf32>
    %v760 = stablehlo.constant dense<0.0> : tensor<f32>
    %v761 = stablehlo.reshape %v7 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v762 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v763 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v764 = stablehlo.reduce(%v761 init: %v760) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v765 = stablehlo.broadcast_in_dim %v764, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v766 = stablehlo.divide %v765, %v762 : tensor<128x16x32x32xf32>
    %v767 = stablehlo.subtract %v761, %v766 : tensor<128x16x32x32xf32>
    %v768 = stablehlo.multiply %v767, %v767 : tensor<128x16x32x32xf32>
    %v769 = stablehlo.reduce(%v768 init: %v760) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v770 = stablehlo.broadcast_in_dim %v769, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v771 = stablehlo.divide %v770, %v762 : tensor<128x16x32x32xf32>
    %v772 = stablehlo.add %v771, %v763 : tensor<128x16x32x32xf32>
    %v773 = stablehlo.rsqrt %v772 : tensor<128x16x32x32xf32>
    %v774 = stablehlo.multiply %v767, %v773 : tensor<128x16x32x32xf32>
    %v775 = stablehlo.reshape %v643 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v776 = stablehlo.multiply %v775, %v774 : tensor<128x16x32x32xf32>
    %v777 = stablehlo.reduce(%v776 init: %v760) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v778 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v779 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v780 = stablehlo.multiply %v778, %g1m : tensor<16xf32>
    %v781 = stablehlo.multiply %v779, %v777 : tensor<16xf32>
    %v782 = stablehlo.add %v780, %v781 : tensor<16xf32>
    %v783 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v784 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v785 = stablehlo.multiply %v783, %g1v : tensor<16xf32>
    %v786 = stablehlo.multiply %v777, %v777 : tensor<16xf32>
    %v787 = stablehlo.multiply %v784, %v786 : tensor<16xf32>
    %v788 = stablehlo.add %v785, %v787 : tensor<16xf32>
    %v789 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v790 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v791 = stablehlo.divide %v782, %v789 : tensor<16xf32>
    %v792 = stablehlo.divide %v788, %v790 : tensor<16xf32>
    %v793 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v794 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v795 = stablehlo.sqrt %v792 : tensor<16xf32>
    %v796 = stablehlo.add %v795, %v794 : tensor<16xf32>
    %v797 = stablehlo.divide %v791, %v796 : tensor<16xf32>
    %v798 = stablehlo.multiply %v793, %v797 : tensor<16xf32>
    %v799 = stablehlo.subtract %g1, %v798 : tensor<16xf32>
    %v800 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v801 = stablehlo.multiply %v800, %v793 : tensor<16xf32>
    %v802 = stablehlo.multiply %v801, %g1 : tensor<16xf32>
    %v803 = stablehlo.subtract %v799, %v802 : tensor<16xf32>
    %v804 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v805 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v806 = stablehlo.multiply %v804, %g1m : tensor<16xf32>
    %v807 = stablehlo.multiply %v805, %v777 : tensor<16xf32>
    %v808 = stablehlo.add %v806, %v807 : tensor<16xf32>
    %v809 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v810 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v811 = stablehlo.multiply %v809, %g1v : tensor<16xf32>
    %v812 = stablehlo.multiply %v777, %v777 : tensor<16xf32>
    %v813 = stablehlo.multiply %v810, %v812 : tensor<16xf32>
    %v814 = stablehlo.add %v811, %v813 : tensor<16xf32>
    %v815 = stablehlo.constant dense<0.0> : tensor<f32>
    %v816 = stablehlo.reshape %v643 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v817 = stablehlo.reduce(%v816 init: %v815) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v818 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v819 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v820 = stablehlo.multiply %v818, %bt1m : tensor<16xf32>
    %v821 = stablehlo.multiply %v819, %v817 : tensor<16xf32>
    %v822 = stablehlo.add %v820, %v821 : tensor<16xf32>
    %v823 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v824 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v825 = stablehlo.multiply %v823, %bt1v : tensor<16xf32>
    %v826 = stablehlo.multiply %v817, %v817 : tensor<16xf32>
    %v827 = stablehlo.multiply %v824, %v826 : tensor<16xf32>
    %v828 = stablehlo.add %v825, %v827 : tensor<16xf32>
    %v829 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v830 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v831 = stablehlo.divide %v822, %v829 : tensor<16xf32>
    %v832 = stablehlo.divide %v828, %v830 : tensor<16xf32>
    %v833 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v834 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v835 = stablehlo.sqrt %v832 : tensor<16xf32>
    %v836 = stablehlo.add %v835, %v834 : tensor<16xf32>
    %v837 = stablehlo.divide %v831, %v836 : tensor<16xf32>
    %v838 = stablehlo.multiply %v833, %v837 : tensor<16xf32>
    %v839 = stablehlo.subtract %bt1, %v838 : tensor<16xf32>
    %v840 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v841 = stablehlo.multiply %v840, %v833 : tensor<16xf32>
    %v842 = stablehlo.multiply %v841, %bt1 : tensor<16xf32>
    %v843 = stablehlo.subtract %v839, %v842 : tensor<16xf32>
    %v844 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v845 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v846 = stablehlo.multiply %v844, %bt1m : tensor<16xf32>
    %v847 = stablehlo.multiply %v845, %v817 : tensor<16xf32>
    %v848 = stablehlo.add %v846, %v847 : tensor<16xf32>
    %v849 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v850 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v851 = stablehlo.multiply %v849, %bt1v : tensor<16xf32>
    %v852 = stablehlo.multiply %v817, %v817 : tensor<16xf32>
    %v853 = stablehlo.multiply %v850, %v852 : tensor<16xf32>
    %v854 = stablehlo.add %v851, %v853 : tensor<16xf32>
    %v855 = stablehlo.reshape %v31 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v856 = stablehlo.reshape %v629 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v857 = stablehlo.transpose %v855, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v858 = stablehlo.transpose %v856, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v859 = stablehlo.convert %v857 : (tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xbf16>
    %v860 = stablehlo.convert %v858 : (tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xbf16>
    %v861 = stablehlo.convolution(%v859, %v860)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xbf16>, tensor<16x128x32x32xbf16>) -> tensor<16x16x3x3xbf16>
    %v862 = stablehlo.convert %v861 : (tensor<16x16x3x3xbf16>) -> tensor<16x16x3x3xf32>
    %v863 = stablehlo.transpose %v862, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v864 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v865 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v866 = stablehlo.multiply %v864, %W2m : tensor<16x16x3x3xf32>
    %v867 = stablehlo.multiply %v865, %v863 : tensor<16x16x3x3xf32>
    %v868 = stablehlo.add %v866, %v867 : tensor<16x16x3x3xf32>
    %v869 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v870 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v871 = stablehlo.multiply %v869, %W2v : tensor<16x16x3x3xf32>
    %v872 = stablehlo.multiply %v863, %v863 : tensor<16x16x3x3xf32>
    %v873 = stablehlo.multiply %v870, %v872 : tensor<16x16x3x3xf32>
    %v874 = stablehlo.add %v871, %v873 : tensor<16x16x3x3xf32>
    %v875 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v876 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v877 = stablehlo.divide %v868, %v875 : tensor<16x16x3x3xf32>
    %v878 = stablehlo.divide %v874, %v876 : tensor<16x16x3x3xf32>
    %v879 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v880 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v881 = stablehlo.sqrt %v878 : tensor<16x16x3x3xf32>
    %v882 = stablehlo.add %v881, %v880 : tensor<16x16x3x3xf32>
    %v883 = stablehlo.divide %v877, %v882 : tensor<16x16x3x3xf32>
    %v884 = stablehlo.multiply %v879, %v883 : tensor<16x16x3x3xf32>
    %v885 = stablehlo.subtract %W2, %v884 : tensor<16x16x3x3xf32>
    %v886 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v887 = stablehlo.multiply %v886, %v879 : tensor<16x16x3x3xf32>
    %v888 = stablehlo.multiply %v887, %W2 : tensor<16x16x3x3xf32>
    %v889 = stablehlo.subtract %v885, %v888 : tensor<16x16x3x3xf32>
    %v890 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v891 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v892 = stablehlo.multiply %v890, %W2m : tensor<16x16x3x3xf32>
    %v893 = stablehlo.multiply %v891, %v863 : tensor<16x16x3x3xf32>
    %v894 = stablehlo.add %v892, %v893 : tensor<16x16x3x3xf32>
    %v895 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v896 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v897 = stablehlo.multiply %v895, %W2v : tensor<16x16x3x3xf32>
    %v898 = stablehlo.multiply %v863, %v863 : tensor<16x16x3x3xf32>
    %v899 = stablehlo.multiply %v896, %v898 : tensor<16x16x3x3xf32>
    %v900 = stablehlo.add %v897, %v899 : tensor<16x16x3x3xf32>
    %v901 = stablehlo.reshape %v629 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v902 = stablehlo.constant dense<0.0> : tensor<f32>
    %v903 = stablehlo.reduce(%v901 init: %v902) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v904 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v905 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v906 = stablehlo.multiply %v904, %cb2m : tensor<16xf32>
    %v907 = stablehlo.multiply %v905, %v903 : tensor<16xf32>
    %v908 = stablehlo.add %v906, %v907 : tensor<16xf32>
    %v909 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v910 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v911 = stablehlo.multiply %v909, %cb2v : tensor<16xf32>
    %v912 = stablehlo.multiply %v903, %v903 : tensor<16xf32>
    %v913 = stablehlo.multiply %v910, %v912 : tensor<16xf32>
    %v914 = stablehlo.add %v911, %v913 : tensor<16xf32>
    %v915 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v916 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v917 = stablehlo.divide %v908, %v915 : tensor<16xf32>
    %v918 = stablehlo.divide %v914, %v916 : tensor<16xf32>
    %v919 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v920 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v921 = stablehlo.sqrt %v918 : tensor<16xf32>
    %v922 = stablehlo.add %v921, %v920 : tensor<16xf32>
    %v923 = stablehlo.divide %v917, %v922 : tensor<16xf32>
    %v924 = stablehlo.multiply %v919, %v923 : tensor<16xf32>
    %v925 = stablehlo.subtract %cb2, %v924 : tensor<16xf32>
    %v926 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v927 = stablehlo.multiply %v926, %v919 : tensor<16xf32>
    %v928 = stablehlo.multiply %v927, %cb2 : tensor<16xf32>
    %v929 = stablehlo.subtract %v925, %v928 : tensor<16xf32>
    %v930 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v931 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v932 = stablehlo.multiply %v930, %cb2m : tensor<16xf32>
    %v933 = stablehlo.multiply %v931, %v903 : tensor<16xf32>
    %v934 = stablehlo.add %v932, %v933 : tensor<16xf32>
    %v935 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v936 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v937 = stablehlo.multiply %v935, %cb2v : tensor<16xf32>
    %v938 = stablehlo.multiply %v903, %v903 : tensor<16xf32>
    %v939 = stablehlo.multiply %v936, %v938 : tensor<16xf32>
    %v940 = stablehlo.add %v937, %v939 : tensor<16xf32>
    %v941 = stablehlo.constant dense<0.0> : tensor<f32>
    %v942 = stablehlo.reshape %v39 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v943 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v944 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v945 = stablehlo.reduce(%v942 init: %v941) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v946 = stablehlo.broadcast_in_dim %v945, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v947 = stablehlo.divide %v946, %v943 : tensor<128x16x32x32xf32>
    %v948 = stablehlo.subtract %v942, %v947 : tensor<128x16x32x32xf32>
    %v949 = stablehlo.multiply %v948, %v948 : tensor<128x16x32x32xf32>
    %v950 = stablehlo.reduce(%v949 init: %v941) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v951 = stablehlo.broadcast_in_dim %v950, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v952 = stablehlo.divide %v951, %v943 : tensor<128x16x32x32xf32>
    %v953 = stablehlo.add %v952, %v944 : tensor<128x16x32x32xf32>
    %v954 = stablehlo.rsqrt %v953 : tensor<128x16x32x32xf32>
    %v955 = stablehlo.multiply %v948, %v954 : tensor<128x16x32x32xf32>
    %v956 = stablehlo.reshape %v599 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v957 = stablehlo.multiply %v956, %v955 : tensor<128x16x32x32xf32>
    %v958 = stablehlo.reduce(%v957 init: %v941) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v959 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v960 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v961 = stablehlo.multiply %v959, %g2m : tensor<16xf32>
    %v962 = stablehlo.multiply %v960, %v958 : tensor<16xf32>
    %v963 = stablehlo.add %v961, %v962 : tensor<16xf32>
    %v964 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v965 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v966 = stablehlo.multiply %v964, %g2v : tensor<16xf32>
    %v967 = stablehlo.multiply %v958, %v958 : tensor<16xf32>
    %v968 = stablehlo.multiply %v965, %v967 : tensor<16xf32>
    %v969 = stablehlo.add %v966, %v968 : tensor<16xf32>
    %v970 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v971 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v972 = stablehlo.divide %v963, %v970 : tensor<16xf32>
    %v973 = stablehlo.divide %v969, %v971 : tensor<16xf32>
    %v974 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v975 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v976 = stablehlo.sqrt %v973 : tensor<16xf32>
    %v977 = stablehlo.add %v976, %v975 : tensor<16xf32>
    %v978 = stablehlo.divide %v972, %v977 : tensor<16xf32>
    %v979 = stablehlo.multiply %v974, %v978 : tensor<16xf32>
    %v980 = stablehlo.subtract %g2, %v979 : tensor<16xf32>
    %v981 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v982 = stablehlo.multiply %v981, %v974 : tensor<16xf32>
    %v983 = stablehlo.multiply %v982, %g2 : tensor<16xf32>
    %v984 = stablehlo.subtract %v980, %v983 : tensor<16xf32>
    %v985 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v986 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v987 = stablehlo.multiply %v985, %g2m : tensor<16xf32>
    %v988 = stablehlo.multiply %v986, %v958 : tensor<16xf32>
    %v989 = stablehlo.add %v987, %v988 : tensor<16xf32>
    %v990 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v991 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v992 = stablehlo.multiply %v990, %g2v : tensor<16xf32>
    %v993 = stablehlo.multiply %v958, %v958 : tensor<16xf32>
    %v994 = stablehlo.multiply %v991, %v993 : tensor<16xf32>
    %v995 = stablehlo.add %v992, %v994 : tensor<16xf32>
    %v996 = stablehlo.constant dense<0.0> : tensor<f32>
    %v997 = stablehlo.reshape %v599 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v998 = stablehlo.reduce(%v997 init: %v996) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v999 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1000 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1001 = stablehlo.multiply %v999, %bt2m : tensor<16xf32>
    %v1002 = stablehlo.multiply %v1000, %v998 : tensor<16xf32>
    %v1003 = stablehlo.add %v1001, %v1002 : tensor<16xf32>
    %v1004 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1005 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1006 = stablehlo.multiply %v1004, %bt2v : tensor<16xf32>
    %v1007 = stablehlo.multiply %v998, %v998 : tensor<16xf32>
    %v1008 = stablehlo.multiply %v1005, %v1007 : tensor<16xf32>
    %v1009 = stablehlo.add %v1006, %v1008 : tensor<16xf32>
    %v1010 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1011 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1012 = stablehlo.divide %v1003, %v1010 : tensor<16xf32>
    %v1013 = stablehlo.divide %v1009, %v1011 : tensor<16xf32>
    %v1014 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1015 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1016 = stablehlo.sqrt %v1013 : tensor<16xf32>
    %v1017 = stablehlo.add %v1016, %v1015 : tensor<16xf32>
    %v1018 = stablehlo.divide %v1012, %v1017 : tensor<16xf32>
    %v1019 = stablehlo.multiply %v1014, %v1018 : tensor<16xf32>
    %v1020 = stablehlo.subtract %bt2, %v1019 : tensor<16xf32>
    %v1021 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1022 = stablehlo.multiply %v1021, %v1014 : tensor<16xf32>
    %v1023 = stablehlo.multiply %v1022, %bt2 : tensor<16xf32>
    %v1024 = stablehlo.subtract %v1020, %v1023 : tensor<16xf32>
    %v1025 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1026 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1027 = stablehlo.multiply %v1025, %bt2m : tensor<16xf32>
    %v1028 = stablehlo.multiply %v1026, %v998 : tensor<16xf32>
    %v1029 = stablehlo.add %v1027, %v1028 : tensor<16xf32>
    %v1030 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1031 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1032 = stablehlo.multiply %v1030, %bt2v : tensor<16xf32>
    %v1033 = stablehlo.multiply %v998, %v998 : tensor<16xf32>
    %v1034 = stablehlo.multiply %v1031, %v1033 : tensor<16xf32>
    %v1035 = stablehlo.add %v1032, %v1034 : tensor<16xf32>
    %v1036 = stablehlo.reshape %v67 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1037 = stablehlo.reshape %v580 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1038 = stablehlo.transpose %v1036, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v1039 = stablehlo.transpose %v1037, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v1040 = stablehlo.convert %v1038 : (tensor<16x128x16x16xf32>) -> tensor<16x128x16x16xbf16>
    %v1041 = stablehlo.convert %v1039 : (tensor<16x128x16x16xf32>) -> tensor<16x128x16x16xbf16>
    %v1042 = stablehlo.convolution(%v1040, %v1041)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xbf16>, tensor<16x128x16x16xbf16>) -> tensor<16x16x3x3xbf16>
    %v1043 = stablehlo.convert %v1042 : (tensor<16x16x3x3xbf16>) -> tensor<16x16x3x3xf32>
    %v1044 = stablehlo.transpose %v1043, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v1045 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1046 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1047 = stablehlo.multiply %v1045, %W3m : tensor<16x16x3x3xf32>
    %v1048 = stablehlo.multiply %v1046, %v1044 : tensor<16x16x3x3xf32>
    %v1049 = stablehlo.add %v1047, %v1048 : tensor<16x16x3x3xf32>
    %v1050 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1051 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1052 = stablehlo.multiply %v1050, %W3v : tensor<16x16x3x3xf32>
    %v1053 = stablehlo.multiply %v1044, %v1044 : tensor<16x16x3x3xf32>
    %v1054 = stablehlo.multiply %v1051, %v1053 : tensor<16x16x3x3xf32>
    %v1055 = stablehlo.add %v1052, %v1054 : tensor<16x16x3x3xf32>
    %v1056 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1057 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1058 = stablehlo.divide %v1049, %v1056 : tensor<16x16x3x3xf32>
    %v1059 = stablehlo.divide %v1055, %v1057 : tensor<16x16x3x3xf32>
    %v1060 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1061 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1062 = stablehlo.sqrt %v1059 : tensor<16x16x3x3xf32>
    %v1063 = stablehlo.add %v1062, %v1061 : tensor<16x16x3x3xf32>
    %v1064 = stablehlo.divide %v1058, %v1063 : tensor<16x16x3x3xf32>
    %v1065 = stablehlo.multiply %v1060, %v1064 : tensor<16x16x3x3xf32>
    %v1066 = stablehlo.subtract %W3, %v1065 : tensor<16x16x3x3xf32>
    %v1067 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1068 = stablehlo.multiply %v1067, %v1060 : tensor<16x16x3x3xf32>
    %v1069 = stablehlo.multiply %v1068, %W3 : tensor<16x16x3x3xf32>
    %v1070 = stablehlo.subtract %v1066, %v1069 : tensor<16x16x3x3xf32>
    %v1071 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1072 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1073 = stablehlo.multiply %v1071, %W3m : tensor<16x16x3x3xf32>
    %v1074 = stablehlo.multiply %v1072, %v1044 : tensor<16x16x3x3xf32>
    %v1075 = stablehlo.add %v1073, %v1074 : tensor<16x16x3x3xf32>
    %v1076 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1077 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1078 = stablehlo.multiply %v1076, %W3v : tensor<16x16x3x3xf32>
    %v1079 = stablehlo.multiply %v1044, %v1044 : tensor<16x16x3x3xf32>
    %v1080 = stablehlo.multiply %v1077, %v1079 : tensor<16x16x3x3xf32>
    %v1081 = stablehlo.add %v1078, %v1080 : tensor<16x16x3x3xf32>
    %v1082 = stablehlo.reshape %v580 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1083 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1084 = stablehlo.reduce(%v1082 init: %v1083) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v1085 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1086 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1087 = stablehlo.multiply %v1085, %cb3m : tensor<16xf32>
    %v1088 = stablehlo.multiply %v1086, %v1084 : tensor<16xf32>
    %v1089 = stablehlo.add %v1087, %v1088 : tensor<16xf32>
    %v1090 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1091 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1092 = stablehlo.multiply %v1090, %cb3v : tensor<16xf32>
    %v1093 = stablehlo.multiply %v1084, %v1084 : tensor<16xf32>
    %v1094 = stablehlo.multiply %v1091, %v1093 : tensor<16xf32>
    %v1095 = stablehlo.add %v1092, %v1094 : tensor<16xf32>
    %v1096 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1097 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1098 = stablehlo.divide %v1089, %v1096 : tensor<16xf32>
    %v1099 = stablehlo.divide %v1095, %v1097 : tensor<16xf32>
    %v1100 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1101 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1102 = stablehlo.sqrt %v1099 : tensor<16xf32>
    %v1103 = stablehlo.add %v1102, %v1101 : tensor<16xf32>
    %v1104 = stablehlo.divide %v1098, %v1103 : tensor<16xf32>
    %v1105 = stablehlo.multiply %v1100, %v1104 : tensor<16xf32>
    %v1106 = stablehlo.subtract %cb3, %v1105 : tensor<16xf32>
    %v1107 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1108 = stablehlo.multiply %v1107, %v1100 : tensor<16xf32>
    %v1109 = stablehlo.multiply %v1108, %cb3 : tensor<16xf32>
    %v1110 = stablehlo.subtract %v1106, %v1109 : tensor<16xf32>
    %v1111 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1112 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1113 = stablehlo.multiply %v1111, %cb3m : tensor<16xf32>
    %v1114 = stablehlo.multiply %v1112, %v1084 : tensor<16xf32>
    %v1115 = stablehlo.add %v1113, %v1114 : tensor<16xf32>
    %v1116 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1117 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1118 = stablehlo.multiply %v1116, %cb3v : tensor<16xf32>
    %v1119 = stablehlo.multiply %v1084, %v1084 : tensor<16xf32>
    %v1120 = stablehlo.multiply %v1117, %v1119 : tensor<16xf32>
    %v1121 = stablehlo.add %v1118, %v1120 : tensor<16xf32>
    %v1122 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1123 = stablehlo.reshape %v75 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1124 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v1125 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v1126 = stablehlo.reduce(%v1123 init: %v1122) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v1127 = stablehlo.broadcast_in_dim %v1126, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v1128 = stablehlo.divide %v1127, %v1124 : tensor<128x16x16x16xf32>
    %v1129 = stablehlo.subtract %v1123, %v1128 : tensor<128x16x16x16xf32>
    %v1130 = stablehlo.multiply %v1129, %v1129 : tensor<128x16x16x16xf32>
    %v1131 = stablehlo.reduce(%v1130 init: %v1122) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v1132 = stablehlo.broadcast_in_dim %v1131, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v1133 = stablehlo.divide %v1132, %v1124 : tensor<128x16x16x16xf32>
    %v1134 = stablehlo.add %v1133, %v1125 : tensor<128x16x16x16xf32>
    %v1135 = stablehlo.rsqrt %v1134 : tensor<128x16x16x16xf32>
    %v1136 = stablehlo.multiply %v1129, %v1135 : tensor<128x16x16x16xf32>
    %v1137 = stablehlo.reshape %v550 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1138 = stablehlo.multiply %v1137, %v1136 : tensor<128x16x16x16xf32>
    %v1139 = stablehlo.reduce(%v1138 init: %v1122) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v1140 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1141 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1142 = stablehlo.multiply %v1140, %g3m : tensor<16xf32>
    %v1143 = stablehlo.multiply %v1141, %v1139 : tensor<16xf32>
    %v1144 = stablehlo.add %v1142, %v1143 : tensor<16xf32>
    %v1145 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1146 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1147 = stablehlo.multiply %v1145, %g3v : tensor<16xf32>
    %v1148 = stablehlo.multiply %v1139, %v1139 : tensor<16xf32>
    %v1149 = stablehlo.multiply %v1146, %v1148 : tensor<16xf32>
    %v1150 = stablehlo.add %v1147, %v1149 : tensor<16xf32>
    %v1151 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1152 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1153 = stablehlo.divide %v1144, %v1151 : tensor<16xf32>
    %v1154 = stablehlo.divide %v1150, %v1152 : tensor<16xf32>
    %v1155 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1156 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1157 = stablehlo.sqrt %v1154 : tensor<16xf32>
    %v1158 = stablehlo.add %v1157, %v1156 : tensor<16xf32>
    %v1159 = stablehlo.divide %v1153, %v1158 : tensor<16xf32>
    %v1160 = stablehlo.multiply %v1155, %v1159 : tensor<16xf32>
    %v1161 = stablehlo.subtract %g3, %v1160 : tensor<16xf32>
    %v1162 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1163 = stablehlo.multiply %v1162, %v1155 : tensor<16xf32>
    %v1164 = stablehlo.multiply %v1163, %g3 : tensor<16xf32>
    %v1165 = stablehlo.subtract %v1161, %v1164 : tensor<16xf32>
    %v1166 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1167 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1168 = stablehlo.multiply %v1166, %g3m : tensor<16xf32>
    %v1169 = stablehlo.multiply %v1167, %v1139 : tensor<16xf32>
    %v1170 = stablehlo.add %v1168, %v1169 : tensor<16xf32>
    %v1171 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1172 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1173 = stablehlo.multiply %v1171, %g3v : tensor<16xf32>
    %v1174 = stablehlo.multiply %v1139, %v1139 : tensor<16xf32>
    %v1175 = stablehlo.multiply %v1172, %v1174 : tensor<16xf32>
    %v1176 = stablehlo.add %v1173, %v1175 : tensor<16xf32>
    %v1177 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1178 = stablehlo.reshape %v550 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1179 = stablehlo.reduce(%v1178 init: %v1177) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v1180 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1181 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1182 = stablehlo.multiply %v1180, %bt3m : tensor<16xf32>
    %v1183 = stablehlo.multiply %v1181, %v1179 : tensor<16xf32>
    %v1184 = stablehlo.add %v1182, %v1183 : tensor<16xf32>
    %v1185 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1186 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1187 = stablehlo.multiply %v1185, %bt3v : tensor<16xf32>
    %v1188 = stablehlo.multiply %v1179, %v1179 : tensor<16xf32>
    %v1189 = stablehlo.multiply %v1186, %v1188 : tensor<16xf32>
    %v1190 = stablehlo.add %v1187, %v1189 : tensor<16xf32>
    %v1191 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1192 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1193 = stablehlo.divide %v1184, %v1191 : tensor<16xf32>
    %v1194 = stablehlo.divide %v1190, %v1192 : tensor<16xf32>
    %v1195 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1196 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1197 = stablehlo.sqrt %v1194 : tensor<16xf32>
    %v1198 = stablehlo.add %v1197, %v1196 : tensor<16xf32>
    %v1199 = stablehlo.divide %v1193, %v1198 : tensor<16xf32>
    %v1200 = stablehlo.multiply %v1195, %v1199 : tensor<16xf32>
    %v1201 = stablehlo.subtract %bt3, %v1200 : tensor<16xf32>
    %v1202 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1203 = stablehlo.multiply %v1202, %v1195 : tensor<16xf32>
    %v1204 = stablehlo.multiply %v1203, %bt3 : tensor<16xf32>
    %v1205 = stablehlo.subtract %v1201, %v1204 : tensor<16xf32>
    %v1206 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1207 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1208 = stablehlo.multiply %v1206, %bt3m : tensor<16xf32>
    %v1209 = stablehlo.multiply %v1207, %v1179 : tensor<16xf32>
    %v1210 = stablehlo.add %v1208, %v1209 : tensor<16xf32>
    %v1211 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1212 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1213 = stablehlo.multiply %v1211, %bt3v : tensor<16xf32>
    %v1214 = stablehlo.multiply %v1179, %v1179 : tensor<16xf32>
    %v1215 = stablehlo.multiply %v1212, %v1214 : tensor<16xf32>
    %v1216 = stablehlo.add %v1213, %v1215 : tensor<16xf32>
    %v1217 = stablehlo.reshape %v99 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1218 = stablehlo.reshape %v536 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1219 = stablehlo.transpose %v1217, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v1220 = stablehlo.transpose %v1218, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v1221 = stablehlo.convert %v1219 : (tensor<16x128x16x16xf32>) -> tensor<16x128x16x16xbf16>
    %v1222 = stablehlo.convert %v1220 : (tensor<16x128x16x16xf32>) -> tensor<16x128x16x16xbf16>
    %v1223 = stablehlo.convolution(%v1221, %v1222)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xbf16>, tensor<16x128x16x16xbf16>) -> tensor<16x16x3x3xbf16>
    %v1224 = stablehlo.convert %v1223 : (tensor<16x16x3x3xbf16>) -> tensor<16x16x3x3xf32>
    %v1225 = stablehlo.transpose %v1224, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v1226 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1227 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1228 = stablehlo.multiply %v1226, %W4m : tensor<16x16x3x3xf32>
    %v1229 = stablehlo.multiply %v1227, %v1225 : tensor<16x16x3x3xf32>
    %v1230 = stablehlo.add %v1228, %v1229 : tensor<16x16x3x3xf32>
    %v1231 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1232 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1233 = stablehlo.multiply %v1231, %W4v : tensor<16x16x3x3xf32>
    %v1234 = stablehlo.multiply %v1225, %v1225 : tensor<16x16x3x3xf32>
    %v1235 = stablehlo.multiply %v1232, %v1234 : tensor<16x16x3x3xf32>
    %v1236 = stablehlo.add %v1233, %v1235 : tensor<16x16x3x3xf32>
    %v1237 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1238 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1239 = stablehlo.divide %v1230, %v1237 : tensor<16x16x3x3xf32>
    %v1240 = stablehlo.divide %v1236, %v1238 : tensor<16x16x3x3xf32>
    %v1241 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1242 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1243 = stablehlo.sqrt %v1240 : tensor<16x16x3x3xf32>
    %v1244 = stablehlo.add %v1243, %v1242 : tensor<16x16x3x3xf32>
    %v1245 = stablehlo.divide %v1239, %v1244 : tensor<16x16x3x3xf32>
    %v1246 = stablehlo.multiply %v1241, %v1245 : tensor<16x16x3x3xf32>
    %v1247 = stablehlo.subtract %W4, %v1246 : tensor<16x16x3x3xf32>
    %v1248 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1249 = stablehlo.multiply %v1248, %v1241 : tensor<16x16x3x3xf32>
    %v1250 = stablehlo.multiply %v1249, %W4 : tensor<16x16x3x3xf32>
    %v1251 = stablehlo.subtract %v1247, %v1250 : tensor<16x16x3x3xf32>
    %v1252 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1253 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1254 = stablehlo.multiply %v1252, %W4m : tensor<16x16x3x3xf32>
    %v1255 = stablehlo.multiply %v1253, %v1225 : tensor<16x16x3x3xf32>
    %v1256 = stablehlo.add %v1254, %v1255 : tensor<16x16x3x3xf32>
    %v1257 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1258 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1259 = stablehlo.multiply %v1257, %W4v : tensor<16x16x3x3xf32>
    %v1260 = stablehlo.multiply %v1225, %v1225 : tensor<16x16x3x3xf32>
    %v1261 = stablehlo.multiply %v1258, %v1260 : tensor<16x16x3x3xf32>
    %v1262 = stablehlo.add %v1259, %v1261 : tensor<16x16x3x3xf32>
    %v1263 = stablehlo.reshape %v536 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1264 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1265 = stablehlo.reduce(%v1263 init: %v1264) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v1266 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1267 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1268 = stablehlo.multiply %v1266, %cb4m : tensor<16xf32>
    %v1269 = stablehlo.multiply %v1267, %v1265 : tensor<16xf32>
    %v1270 = stablehlo.add %v1268, %v1269 : tensor<16xf32>
    %v1271 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1272 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1273 = stablehlo.multiply %v1271, %cb4v : tensor<16xf32>
    %v1274 = stablehlo.multiply %v1265, %v1265 : tensor<16xf32>
    %v1275 = stablehlo.multiply %v1272, %v1274 : tensor<16xf32>
    %v1276 = stablehlo.add %v1273, %v1275 : tensor<16xf32>
    %v1277 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1278 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1279 = stablehlo.divide %v1270, %v1277 : tensor<16xf32>
    %v1280 = stablehlo.divide %v1276, %v1278 : tensor<16xf32>
    %v1281 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1282 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1283 = stablehlo.sqrt %v1280 : tensor<16xf32>
    %v1284 = stablehlo.add %v1283, %v1282 : tensor<16xf32>
    %v1285 = stablehlo.divide %v1279, %v1284 : tensor<16xf32>
    %v1286 = stablehlo.multiply %v1281, %v1285 : tensor<16xf32>
    %v1287 = stablehlo.subtract %cb4, %v1286 : tensor<16xf32>
    %v1288 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1289 = stablehlo.multiply %v1288, %v1281 : tensor<16xf32>
    %v1290 = stablehlo.multiply %v1289, %cb4 : tensor<16xf32>
    %v1291 = stablehlo.subtract %v1287, %v1290 : tensor<16xf32>
    %v1292 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1293 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1294 = stablehlo.multiply %v1292, %cb4m : tensor<16xf32>
    %v1295 = stablehlo.multiply %v1293, %v1265 : tensor<16xf32>
    %v1296 = stablehlo.add %v1294, %v1295 : tensor<16xf32>
    %v1297 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1298 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1299 = stablehlo.multiply %v1297, %cb4v : tensor<16xf32>
    %v1300 = stablehlo.multiply %v1265, %v1265 : tensor<16xf32>
    %v1301 = stablehlo.multiply %v1298, %v1300 : tensor<16xf32>
    %v1302 = stablehlo.add %v1299, %v1301 : tensor<16xf32>
    %v1303 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1304 = stablehlo.reshape %v107 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1305 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v1306 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v1307 = stablehlo.reduce(%v1304 init: %v1303) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v1308 = stablehlo.broadcast_in_dim %v1307, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v1309 = stablehlo.divide %v1308, %v1305 : tensor<128x16x16x16xf32>
    %v1310 = stablehlo.subtract %v1304, %v1309 : tensor<128x16x16x16xf32>
    %v1311 = stablehlo.multiply %v1310, %v1310 : tensor<128x16x16x16xf32>
    %v1312 = stablehlo.reduce(%v1311 init: %v1303) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v1313 = stablehlo.broadcast_in_dim %v1312, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v1314 = stablehlo.divide %v1313, %v1305 : tensor<128x16x16x16xf32>
    %v1315 = stablehlo.add %v1314, %v1306 : tensor<128x16x16x16xf32>
    %v1316 = stablehlo.rsqrt %v1315 : tensor<128x16x16x16xf32>
    %v1317 = stablehlo.multiply %v1310, %v1316 : tensor<128x16x16x16xf32>
    %v1318 = stablehlo.reshape %v506 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1319 = stablehlo.multiply %v1318, %v1317 : tensor<128x16x16x16xf32>
    %v1320 = stablehlo.reduce(%v1319 init: %v1303) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v1321 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1322 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1323 = stablehlo.multiply %v1321, %g4m : tensor<16xf32>
    %v1324 = stablehlo.multiply %v1322, %v1320 : tensor<16xf32>
    %v1325 = stablehlo.add %v1323, %v1324 : tensor<16xf32>
    %v1326 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1327 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1328 = stablehlo.multiply %v1326, %g4v : tensor<16xf32>
    %v1329 = stablehlo.multiply %v1320, %v1320 : tensor<16xf32>
    %v1330 = stablehlo.multiply %v1327, %v1329 : tensor<16xf32>
    %v1331 = stablehlo.add %v1328, %v1330 : tensor<16xf32>
    %v1332 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1333 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1334 = stablehlo.divide %v1325, %v1332 : tensor<16xf32>
    %v1335 = stablehlo.divide %v1331, %v1333 : tensor<16xf32>
    %v1336 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1337 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1338 = stablehlo.sqrt %v1335 : tensor<16xf32>
    %v1339 = stablehlo.add %v1338, %v1337 : tensor<16xf32>
    %v1340 = stablehlo.divide %v1334, %v1339 : tensor<16xf32>
    %v1341 = stablehlo.multiply %v1336, %v1340 : tensor<16xf32>
    %v1342 = stablehlo.subtract %g4, %v1341 : tensor<16xf32>
    %v1343 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1344 = stablehlo.multiply %v1343, %v1336 : tensor<16xf32>
    %v1345 = stablehlo.multiply %v1344, %g4 : tensor<16xf32>
    %v1346 = stablehlo.subtract %v1342, %v1345 : tensor<16xf32>
    %v1347 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1348 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1349 = stablehlo.multiply %v1347, %g4m : tensor<16xf32>
    %v1350 = stablehlo.multiply %v1348, %v1320 : tensor<16xf32>
    %v1351 = stablehlo.add %v1349, %v1350 : tensor<16xf32>
    %v1352 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1353 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1354 = stablehlo.multiply %v1352, %g4v : tensor<16xf32>
    %v1355 = stablehlo.multiply %v1320, %v1320 : tensor<16xf32>
    %v1356 = stablehlo.multiply %v1353, %v1355 : tensor<16xf32>
    %v1357 = stablehlo.add %v1354, %v1356 : tensor<16xf32>
    %v1358 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1359 = stablehlo.reshape %v506 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1360 = stablehlo.reduce(%v1359 init: %v1358) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v1361 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1362 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1363 = stablehlo.multiply %v1361, %bt4m : tensor<16xf32>
    %v1364 = stablehlo.multiply %v1362, %v1360 : tensor<16xf32>
    %v1365 = stablehlo.add %v1363, %v1364 : tensor<16xf32>
    %v1366 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1367 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1368 = stablehlo.multiply %v1366, %bt4v : tensor<16xf32>
    %v1369 = stablehlo.multiply %v1360, %v1360 : tensor<16xf32>
    %v1370 = stablehlo.multiply %v1367, %v1369 : tensor<16xf32>
    %v1371 = stablehlo.add %v1368, %v1370 : tensor<16xf32>
    %v1372 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1373 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1374 = stablehlo.divide %v1365, %v1372 : tensor<16xf32>
    %v1375 = stablehlo.divide %v1371, %v1373 : tensor<16xf32>
    %v1376 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1377 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1378 = stablehlo.sqrt %v1375 : tensor<16xf32>
    %v1379 = stablehlo.add %v1378, %v1377 : tensor<16xf32>
    %v1380 = stablehlo.divide %v1374, %v1379 : tensor<16xf32>
    %v1381 = stablehlo.multiply %v1376, %v1380 : tensor<16xf32>
    %v1382 = stablehlo.subtract %bt4, %v1381 : tensor<16xf32>
    %v1383 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1384 = stablehlo.multiply %v1383, %v1376 : tensor<16xf32>
    %v1385 = stablehlo.multiply %v1384, %bt4 : tensor<16xf32>
    %v1386 = stablehlo.subtract %v1382, %v1385 : tensor<16xf32>
    %v1387 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1388 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1389 = stablehlo.multiply %v1387, %bt4m : tensor<16xf32>
    %v1390 = stablehlo.multiply %v1388, %v1360 : tensor<16xf32>
    %v1391 = stablehlo.add %v1389, %v1390 : tensor<16xf32>
    %v1392 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1393 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1394 = stablehlo.multiply %v1392, %bt4v : tensor<16xf32>
    %v1395 = stablehlo.multiply %v1360, %v1360 : tensor<16xf32>
    %v1396 = stablehlo.multiply %v1393, %v1395 : tensor<16xf32>
    %v1397 = stablehlo.add %v1394, %v1396 : tensor<16xf32>
    %v1398 = stablehlo.reshape %v135 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v1399 = stablehlo.reshape %v487 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1400 = stablehlo.transpose %v1398, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v1401 = stablehlo.transpose %v1399, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v1402 = stablehlo.convert %v1400 : (tensor<16x128x8x8xf32>) -> tensor<16x128x8x8xbf16>
    %v1403 = stablehlo.convert %v1401 : (tensor<32x128x8x8xf32>) -> tensor<32x128x8x8xbf16>
    %v1404 = stablehlo.convolution(%v1402, %v1403)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xbf16>, tensor<32x128x8x8xbf16>) -> tensor<16x32x3x3xbf16>
    %v1405 = stablehlo.convert %v1404 : (tensor<16x32x3x3xbf16>) -> tensor<16x32x3x3xf32>
    %v1406 = stablehlo.transpose %v1405, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v1407 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1408 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1409 = stablehlo.multiply %v1407, %W5m : tensor<32x16x3x3xf32>
    %v1410 = stablehlo.multiply %v1408, %v1406 : tensor<32x16x3x3xf32>
    %v1411 = stablehlo.add %v1409, %v1410 : tensor<32x16x3x3xf32>
    %v1412 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1413 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1414 = stablehlo.multiply %v1412, %W5v : tensor<32x16x3x3xf32>
    %v1415 = stablehlo.multiply %v1406, %v1406 : tensor<32x16x3x3xf32>
    %v1416 = stablehlo.multiply %v1413, %v1415 : tensor<32x16x3x3xf32>
    %v1417 = stablehlo.add %v1414, %v1416 : tensor<32x16x3x3xf32>
    %v1418 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1419 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1420 = stablehlo.divide %v1411, %v1418 : tensor<32x16x3x3xf32>
    %v1421 = stablehlo.divide %v1417, %v1419 : tensor<32x16x3x3xf32>
    %v1422 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1423 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1424 = stablehlo.sqrt %v1421 : tensor<32x16x3x3xf32>
    %v1425 = stablehlo.add %v1424, %v1423 : tensor<32x16x3x3xf32>
    %v1426 = stablehlo.divide %v1420, %v1425 : tensor<32x16x3x3xf32>
    %v1427 = stablehlo.multiply %v1422, %v1426 : tensor<32x16x3x3xf32>
    %v1428 = stablehlo.subtract %W5, %v1427 : tensor<32x16x3x3xf32>
    %v1429 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1430 = stablehlo.multiply %v1429, %v1422 : tensor<32x16x3x3xf32>
    %v1431 = stablehlo.multiply %v1430, %W5 : tensor<32x16x3x3xf32>
    %v1432 = stablehlo.subtract %v1428, %v1431 : tensor<32x16x3x3xf32>
    %v1433 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1434 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1435 = stablehlo.multiply %v1433, %W5m : tensor<32x16x3x3xf32>
    %v1436 = stablehlo.multiply %v1434, %v1406 : tensor<32x16x3x3xf32>
    %v1437 = stablehlo.add %v1435, %v1436 : tensor<32x16x3x3xf32>
    %v1438 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1439 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1440 = stablehlo.multiply %v1438, %W5v : tensor<32x16x3x3xf32>
    %v1441 = stablehlo.multiply %v1406, %v1406 : tensor<32x16x3x3xf32>
    %v1442 = stablehlo.multiply %v1439, %v1441 : tensor<32x16x3x3xf32>
    %v1443 = stablehlo.add %v1440, %v1442 : tensor<32x16x3x3xf32>
    %v1444 = stablehlo.reshape %v487 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1445 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1446 = stablehlo.reduce(%v1444 init: %v1445) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1447 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1448 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1449 = stablehlo.multiply %v1447, %cb5m : tensor<32xf32>
    %v1450 = stablehlo.multiply %v1448, %v1446 : tensor<32xf32>
    %v1451 = stablehlo.add %v1449, %v1450 : tensor<32xf32>
    %v1452 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1453 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1454 = stablehlo.multiply %v1452, %cb5v : tensor<32xf32>
    %v1455 = stablehlo.multiply %v1446, %v1446 : tensor<32xf32>
    %v1456 = stablehlo.multiply %v1453, %v1455 : tensor<32xf32>
    %v1457 = stablehlo.add %v1454, %v1456 : tensor<32xf32>
    %v1458 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1459 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1460 = stablehlo.divide %v1451, %v1458 : tensor<32xf32>
    %v1461 = stablehlo.divide %v1457, %v1459 : tensor<32xf32>
    %v1462 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1463 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1464 = stablehlo.sqrt %v1461 : tensor<32xf32>
    %v1465 = stablehlo.add %v1464, %v1463 : tensor<32xf32>
    %v1466 = stablehlo.divide %v1460, %v1465 : tensor<32xf32>
    %v1467 = stablehlo.multiply %v1462, %v1466 : tensor<32xf32>
    %v1468 = stablehlo.subtract %cb5, %v1467 : tensor<32xf32>
    %v1469 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1470 = stablehlo.multiply %v1469, %v1462 : tensor<32xf32>
    %v1471 = stablehlo.multiply %v1470, %cb5 : tensor<32xf32>
    %v1472 = stablehlo.subtract %v1468, %v1471 : tensor<32xf32>
    %v1473 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1474 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1475 = stablehlo.multiply %v1473, %cb5m : tensor<32xf32>
    %v1476 = stablehlo.multiply %v1474, %v1446 : tensor<32xf32>
    %v1477 = stablehlo.add %v1475, %v1476 : tensor<32xf32>
    %v1478 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1479 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1480 = stablehlo.multiply %v1478, %cb5v : tensor<32xf32>
    %v1481 = stablehlo.multiply %v1446, %v1446 : tensor<32xf32>
    %v1482 = stablehlo.multiply %v1479, %v1481 : tensor<32xf32>
    %v1483 = stablehlo.add %v1480, %v1482 : tensor<32xf32>
    %v1484 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1485 = stablehlo.reshape %v143 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1486 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v1487 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v1488 = stablehlo.reduce(%v1485 init: %v1484) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1489 = stablehlo.broadcast_in_dim %v1488, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v1490 = stablehlo.divide %v1489, %v1486 : tensor<128x32x8x8xf32>
    %v1491 = stablehlo.subtract %v1485, %v1490 : tensor<128x32x8x8xf32>
    %v1492 = stablehlo.multiply %v1491, %v1491 : tensor<128x32x8x8xf32>
    %v1493 = stablehlo.reduce(%v1492 init: %v1484) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1494 = stablehlo.broadcast_in_dim %v1493, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v1495 = stablehlo.divide %v1494, %v1486 : tensor<128x32x8x8xf32>
    %v1496 = stablehlo.add %v1495, %v1487 : tensor<128x32x8x8xf32>
    %v1497 = stablehlo.rsqrt %v1496 : tensor<128x32x8x8xf32>
    %v1498 = stablehlo.multiply %v1491, %v1497 : tensor<128x32x8x8xf32>
    %v1499 = stablehlo.reshape %v457 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1500 = stablehlo.multiply %v1499, %v1498 : tensor<128x32x8x8xf32>
    %v1501 = stablehlo.reduce(%v1500 init: %v1484) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1502 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1503 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1504 = stablehlo.multiply %v1502, %g5m : tensor<32xf32>
    %v1505 = stablehlo.multiply %v1503, %v1501 : tensor<32xf32>
    %v1506 = stablehlo.add %v1504, %v1505 : tensor<32xf32>
    %v1507 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1508 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1509 = stablehlo.multiply %v1507, %g5v : tensor<32xf32>
    %v1510 = stablehlo.multiply %v1501, %v1501 : tensor<32xf32>
    %v1511 = stablehlo.multiply %v1508, %v1510 : tensor<32xf32>
    %v1512 = stablehlo.add %v1509, %v1511 : tensor<32xf32>
    %v1513 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1514 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1515 = stablehlo.divide %v1506, %v1513 : tensor<32xf32>
    %v1516 = stablehlo.divide %v1512, %v1514 : tensor<32xf32>
    %v1517 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1518 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1519 = stablehlo.sqrt %v1516 : tensor<32xf32>
    %v1520 = stablehlo.add %v1519, %v1518 : tensor<32xf32>
    %v1521 = stablehlo.divide %v1515, %v1520 : tensor<32xf32>
    %v1522 = stablehlo.multiply %v1517, %v1521 : tensor<32xf32>
    %v1523 = stablehlo.subtract %g5, %v1522 : tensor<32xf32>
    %v1524 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1525 = stablehlo.multiply %v1524, %v1517 : tensor<32xf32>
    %v1526 = stablehlo.multiply %v1525, %g5 : tensor<32xf32>
    %v1527 = stablehlo.subtract %v1523, %v1526 : tensor<32xf32>
    %v1528 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1529 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1530 = stablehlo.multiply %v1528, %g5m : tensor<32xf32>
    %v1531 = stablehlo.multiply %v1529, %v1501 : tensor<32xf32>
    %v1532 = stablehlo.add %v1530, %v1531 : tensor<32xf32>
    %v1533 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1534 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1535 = stablehlo.multiply %v1533, %g5v : tensor<32xf32>
    %v1536 = stablehlo.multiply %v1501, %v1501 : tensor<32xf32>
    %v1537 = stablehlo.multiply %v1534, %v1536 : tensor<32xf32>
    %v1538 = stablehlo.add %v1535, %v1537 : tensor<32xf32>
    %v1539 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1540 = stablehlo.reshape %v457 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1541 = stablehlo.reduce(%v1540 init: %v1539) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1542 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1543 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1544 = stablehlo.multiply %v1542, %bt5m : tensor<32xf32>
    %v1545 = stablehlo.multiply %v1543, %v1541 : tensor<32xf32>
    %v1546 = stablehlo.add %v1544, %v1545 : tensor<32xf32>
    %v1547 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1548 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1549 = stablehlo.multiply %v1547, %bt5v : tensor<32xf32>
    %v1550 = stablehlo.multiply %v1541, %v1541 : tensor<32xf32>
    %v1551 = stablehlo.multiply %v1548, %v1550 : tensor<32xf32>
    %v1552 = stablehlo.add %v1549, %v1551 : tensor<32xf32>
    %v1553 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1554 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1555 = stablehlo.divide %v1546, %v1553 : tensor<32xf32>
    %v1556 = stablehlo.divide %v1552, %v1554 : tensor<32xf32>
    %v1557 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1558 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1559 = stablehlo.sqrt %v1556 : tensor<32xf32>
    %v1560 = stablehlo.add %v1559, %v1558 : tensor<32xf32>
    %v1561 = stablehlo.divide %v1555, %v1560 : tensor<32xf32>
    %v1562 = stablehlo.multiply %v1557, %v1561 : tensor<32xf32>
    %v1563 = stablehlo.subtract %bt5, %v1562 : tensor<32xf32>
    %v1564 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1565 = stablehlo.multiply %v1564, %v1557 : tensor<32xf32>
    %v1566 = stablehlo.multiply %v1565, %bt5 : tensor<32xf32>
    %v1567 = stablehlo.subtract %v1563, %v1566 : tensor<32xf32>
    %v1568 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1569 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1570 = stablehlo.multiply %v1568, %bt5m : tensor<32xf32>
    %v1571 = stablehlo.multiply %v1569, %v1541 : tensor<32xf32>
    %v1572 = stablehlo.add %v1570, %v1571 : tensor<32xf32>
    %v1573 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1574 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1575 = stablehlo.multiply %v1573, %bt5v : tensor<32xf32>
    %v1576 = stablehlo.multiply %v1541, %v1541 : tensor<32xf32>
    %v1577 = stablehlo.multiply %v1574, %v1576 : tensor<32xf32>
    %v1578 = stablehlo.add %v1575, %v1577 : tensor<32xf32>
    %v1579 = stablehlo.reshape %v167 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1580 = stablehlo.reshape %v443 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1581 = stablehlo.transpose %v1579, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v1582 = stablehlo.transpose %v1580, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v1583 = stablehlo.convert %v1581 : (tensor<32x128x8x8xf32>) -> tensor<32x128x8x8xbf16>
    %v1584 = stablehlo.convert %v1582 : (tensor<32x128x8x8xf32>) -> tensor<32x128x8x8xbf16>
    %v1585 = stablehlo.convolution(%v1583, %v1584)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xbf16>, tensor<32x128x8x8xbf16>) -> tensor<32x32x3x3xbf16>
    %v1586 = stablehlo.convert %v1585 : (tensor<32x32x3x3xbf16>) -> tensor<32x32x3x3xf32>
    %v1587 = stablehlo.transpose %v1586, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v1588 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1589 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1590 = stablehlo.multiply %v1588, %W6m : tensor<32x32x3x3xf32>
    %v1591 = stablehlo.multiply %v1589, %v1587 : tensor<32x32x3x3xf32>
    %v1592 = stablehlo.add %v1590, %v1591 : tensor<32x32x3x3xf32>
    %v1593 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1594 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1595 = stablehlo.multiply %v1593, %W6v : tensor<32x32x3x3xf32>
    %v1596 = stablehlo.multiply %v1587, %v1587 : tensor<32x32x3x3xf32>
    %v1597 = stablehlo.multiply %v1594, %v1596 : tensor<32x32x3x3xf32>
    %v1598 = stablehlo.add %v1595, %v1597 : tensor<32x32x3x3xf32>
    %v1599 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1600 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1601 = stablehlo.divide %v1592, %v1599 : tensor<32x32x3x3xf32>
    %v1602 = stablehlo.divide %v1598, %v1600 : tensor<32x32x3x3xf32>
    %v1603 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1604 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1605 = stablehlo.sqrt %v1602 : tensor<32x32x3x3xf32>
    %v1606 = stablehlo.add %v1605, %v1604 : tensor<32x32x3x3xf32>
    %v1607 = stablehlo.divide %v1601, %v1606 : tensor<32x32x3x3xf32>
    %v1608 = stablehlo.multiply %v1603, %v1607 : tensor<32x32x3x3xf32>
    %v1609 = stablehlo.subtract %W6, %v1608 : tensor<32x32x3x3xf32>
    %v1610 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1611 = stablehlo.multiply %v1610, %v1603 : tensor<32x32x3x3xf32>
    %v1612 = stablehlo.multiply %v1611, %W6 : tensor<32x32x3x3xf32>
    %v1613 = stablehlo.subtract %v1609, %v1612 : tensor<32x32x3x3xf32>
    %v1614 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1615 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1616 = stablehlo.multiply %v1614, %W6m : tensor<32x32x3x3xf32>
    %v1617 = stablehlo.multiply %v1615, %v1587 : tensor<32x32x3x3xf32>
    %v1618 = stablehlo.add %v1616, %v1617 : tensor<32x32x3x3xf32>
    %v1619 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1620 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1621 = stablehlo.multiply %v1619, %W6v : tensor<32x32x3x3xf32>
    %v1622 = stablehlo.multiply %v1587, %v1587 : tensor<32x32x3x3xf32>
    %v1623 = stablehlo.multiply %v1620, %v1622 : tensor<32x32x3x3xf32>
    %v1624 = stablehlo.add %v1621, %v1623 : tensor<32x32x3x3xf32>
    %v1625 = stablehlo.reshape %v443 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1626 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1627 = stablehlo.reduce(%v1625 init: %v1626) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1628 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1629 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1630 = stablehlo.multiply %v1628, %cb6m : tensor<32xf32>
    %v1631 = stablehlo.multiply %v1629, %v1627 : tensor<32xf32>
    %v1632 = stablehlo.add %v1630, %v1631 : tensor<32xf32>
    %v1633 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1634 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1635 = stablehlo.multiply %v1633, %cb6v : tensor<32xf32>
    %v1636 = stablehlo.multiply %v1627, %v1627 : tensor<32xf32>
    %v1637 = stablehlo.multiply %v1634, %v1636 : tensor<32xf32>
    %v1638 = stablehlo.add %v1635, %v1637 : tensor<32xf32>
    %v1639 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1640 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1641 = stablehlo.divide %v1632, %v1639 : tensor<32xf32>
    %v1642 = stablehlo.divide %v1638, %v1640 : tensor<32xf32>
    %v1643 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1644 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1645 = stablehlo.sqrt %v1642 : tensor<32xf32>
    %v1646 = stablehlo.add %v1645, %v1644 : tensor<32xf32>
    %v1647 = stablehlo.divide %v1641, %v1646 : tensor<32xf32>
    %v1648 = stablehlo.multiply %v1643, %v1647 : tensor<32xf32>
    %v1649 = stablehlo.subtract %cb6, %v1648 : tensor<32xf32>
    %v1650 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1651 = stablehlo.multiply %v1650, %v1643 : tensor<32xf32>
    %v1652 = stablehlo.multiply %v1651, %cb6 : tensor<32xf32>
    %v1653 = stablehlo.subtract %v1649, %v1652 : tensor<32xf32>
    %v1654 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1655 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1656 = stablehlo.multiply %v1654, %cb6m : tensor<32xf32>
    %v1657 = stablehlo.multiply %v1655, %v1627 : tensor<32xf32>
    %v1658 = stablehlo.add %v1656, %v1657 : tensor<32xf32>
    %v1659 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1660 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1661 = stablehlo.multiply %v1659, %cb6v : tensor<32xf32>
    %v1662 = stablehlo.multiply %v1627, %v1627 : tensor<32xf32>
    %v1663 = stablehlo.multiply %v1660, %v1662 : tensor<32xf32>
    %v1664 = stablehlo.add %v1661, %v1663 : tensor<32xf32>
    %v1665 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1666 = stablehlo.reshape %v175 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1667 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v1668 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v1669 = stablehlo.reduce(%v1666 init: %v1665) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1670 = stablehlo.broadcast_in_dim %v1669, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v1671 = stablehlo.divide %v1670, %v1667 : tensor<128x32x8x8xf32>
    %v1672 = stablehlo.subtract %v1666, %v1671 : tensor<128x32x8x8xf32>
    %v1673 = stablehlo.multiply %v1672, %v1672 : tensor<128x32x8x8xf32>
    %v1674 = stablehlo.reduce(%v1673 init: %v1665) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1675 = stablehlo.broadcast_in_dim %v1674, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v1676 = stablehlo.divide %v1675, %v1667 : tensor<128x32x8x8xf32>
    %v1677 = stablehlo.add %v1676, %v1668 : tensor<128x32x8x8xf32>
    %v1678 = stablehlo.rsqrt %v1677 : tensor<128x32x8x8xf32>
    %v1679 = stablehlo.multiply %v1672, %v1678 : tensor<128x32x8x8xf32>
    %v1680 = stablehlo.reshape %v413 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1681 = stablehlo.multiply %v1680, %v1679 : tensor<128x32x8x8xf32>
    %v1682 = stablehlo.reduce(%v1681 init: %v1665) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1683 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1684 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1685 = stablehlo.multiply %v1683, %g6m : tensor<32xf32>
    %v1686 = stablehlo.multiply %v1684, %v1682 : tensor<32xf32>
    %v1687 = stablehlo.add %v1685, %v1686 : tensor<32xf32>
    %v1688 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1689 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1690 = stablehlo.multiply %v1688, %g6v : tensor<32xf32>
    %v1691 = stablehlo.multiply %v1682, %v1682 : tensor<32xf32>
    %v1692 = stablehlo.multiply %v1689, %v1691 : tensor<32xf32>
    %v1693 = stablehlo.add %v1690, %v1692 : tensor<32xf32>
    %v1694 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1695 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1696 = stablehlo.divide %v1687, %v1694 : tensor<32xf32>
    %v1697 = stablehlo.divide %v1693, %v1695 : tensor<32xf32>
    %v1698 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1699 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1700 = stablehlo.sqrt %v1697 : tensor<32xf32>
    %v1701 = stablehlo.add %v1700, %v1699 : tensor<32xf32>
    %v1702 = stablehlo.divide %v1696, %v1701 : tensor<32xf32>
    %v1703 = stablehlo.multiply %v1698, %v1702 : tensor<32xf32>
    %v1704 = stablehlo.subtract %g6, %v1703 : tensor<32xf32>
    %v1705 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1706 = stablehlo.multiply %v1705, %v1698 : tensor<32xf32>
    %v1707 = stablehlo.multiply %v1706, %g6 : tensor<32xf32>
    %v1708 = stablehlo.subtract %v1704, %v1707 : tensor<32xf32>
    %v1709 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1710 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1711 = stablehlo.multiply %v1709, %g6m : tensor<32xf32>
    %v1712 = stablehlo.multiply %v1710, %v1682 : tensor<32xf32>
    %v1713 = stablehlo.add %v1711, %v1712 : tensor<32xf32>
    %v1714 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1715 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1716 = stablehlo.multiply %v1714, %g6v : tensor<32xf32>
    %v1717 = stablehlo.multiply %v1682, %v1682 : tensor<32xf32>
    %v1718 = stablehlo.multiply %v1715, %v1717 : tensor<32xf32>
    %v1719 = stablehlo.add %v1716, %v1718 : tensor<32xf32>
    %v1720 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1721 = stablehlo.reshape %v413 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1722 = stablehlo.reduce(%v1721 init: %v1720) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1723 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1724 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1725 = stablehlo.multiply %v1723, %bt6m : tensor<32xf32>
    %v1726 = stablehlo.multiply %v1724, %v1722 : tensor<32xf32>
    %v1727 = stablehlo.add %v1725, %v1726 : tensor<32xf32>
    %v1728 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1729 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1730 = stablehlo.multiply %v1728, %bt6v : tensor<32xf32>
    %v1731 = stablehlo.multiply %v1722, %v1722 : tensor<32xf32>
    %v1732 = stablehlo.multiply %v1729, %v1731 : tensor<32xf32>
    %v1733 = stablehlo.add %v1730, %v1732 : tensor<32xf32>
    %v1734 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1735 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1736 = stablehlo.divide %v1727, %v1734 : tensor<32xf32>
    %v1737 = stablehlo.divide %v1733, %v1735 : tensor<32xf32>
    %v1738 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1739 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1740 = stablehlo.sqrt %v1737 : tensor<32xf32>
    %v1741 = stablehlo.add %v1740, %v1739 : tensor<32xf32>
    %v1742 = stablehlo.divide %v1736, %v1741 : tensor<32xf32>
    %v1743 = stablehlo.multiply %v1738, %v1742 : tensor<32xf32>
    %v1744 = stablehlo.subtract %bt6, %v1743 : tensor<32xf32>
    %v1745 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1746 = stablehlo.multiply %v1745, %v1738 : tensor<32xf32>
    %v1747 = stablehlo.multiply %v1746, %bt6 : tensor<32xf32>
    %v1748 = stablehlo.subtract %v1744, %v1747 : tensor<32xf32>
    %v1749 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1750 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1751 = stablehlo.multiply %v1749, %bt6m : tensor<32xf32>
    %v1752 = stablehlo.multiply %v1750, %v1722 : tensor<32xf32>
    %v1753 = stablehlo.add %v1751, %v1752 : tensor<32xf32>
    %v1754 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1755 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1756 = stablehlo.multiply %v1754, %bt6v : tensor<32xf32>
    %v1757 = stablehlo.multiply %v1722, %v1722 : tensor<32xf32>
    %v1758 = stablehlo.multiply %v1755, %v1757 : tensor<32xf32>
    %v1759 = stablehlo.add %v1756, %v1758 : tensor<32xf32>
    %v1760 = stablehlo.reshape %v203 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1761 = stablehlo.reshape %v394 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1762 = stablehlo.transpose %v1760, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v1763 = stablehlo.transpose %v1761, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v1764 = stablehlo.convert %v1762 : (tensor<32x128x4x4xf32>) -> tensor<32x128x4x4xbf16>
    %v1765 = stablehlo.convert %v1763 : (tensor<32x128x4x4xf32>) -> tensor<32x128x4x4xbf16>
    %v1766 = stablehlo.convolution(%v1764, %v1765)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xbf16>, tensor<32x128x4x4xbf16>) -> tensor<32x32x3x3xbf16>
    %v1767 = stablehlo.convert %v1766 : (tensor<32x32x3x3xbf16>) -> tensor<32x32x3x3xf32>
    %v1768 = stablehlo.transpose %v1767, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v1769 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1770 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1771 = stablehlo.multiply %v1769, %W7m : tensor<32x32x3x3xf32>
    %v1772 = stablehlo.multiply %v1770, %v1768 : tensor<32x32x3x3xf32>
    %v1773 = stablehlo.add %v1771, %v1772 : tensor<32x32x3x3xf32>
    %v1774 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1775 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1776 = stablehlo.multiply %v1774, %W7v : tensor<32x32x3x3xf32>
    %v1777 = stablehlo.multiply %v1768, %v1768 : tensor<32x32x3x3xf32>
    %v1778 = stablehlo.multiply %v1775, %v1777 : tensor<32x32x3x3xf32>
    %v1779 = stablehlo.add %v1776, %v1778 : tensor<32x32x3x3xf32>
    %v1780 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1781 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1782 = stablehlo.divide %v1773, %v1780 : tensor<32x32x3x3xf32>
    %v1783 = stablehlo.divide %v1779, %v1781 : tensor<32x32x3x3xf32>
    %v1784 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1785 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1786 = stablehlo.sqrt %v1783 : tensor<32x32x3x3xf32>
    %v1787 = stablehlo.add %v1786, %v1785 : tensor<32x32x3x3xf32>
    %v1788 = stablehlo.divide %v1782, %v1787 : tensor<32x32x3x3xf32>
    %v1789 = stablehlo.multiply %v1784, %v1788 : tensor<32x32x3x3xf32>
    %v1790 = stablehlo.subtract %W7, %v1789 : tensor<32x32x3x3xf32>
    %v1791 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1792 = stablehlo.multiply %v1791, %v1784 : tensor<32x32x3x3xf32>
    %v1793 = stablehlo.multiply %v1792, %W7 : tensor<32x32x3x3xf32>
    %v1794 = stablehlo.subtract %v1790, %v1793 : tensor<32x32x3x3xf32>
    %v1795 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1796 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1797 = stablehlo.multiply %v1795, %W7m : tensor<32x32x3x3xf32>
    %v1798 = stablehlo.multiply %v1796, %v1768 : tensor<32x32x3x3xf32>
    %v1799 = stablehlo.add %v1797, %v1798 : tensor<32x32x3x3xf32>
    %v1800 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1801 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1802 = stablehlo.multiply %v1800, %W7v : tensor<32x32x3x3xf32>
    %v1803 = stablehlo.multiply %v1768, %v1768 : tensor<32x32x3x3xf32>
    %v1804 = stablehlo.multiply %v1801, %v1803 : tensor<32x32x3x3xf32>
    %v1805 = stablehlo.add %v1802, %v1804 : tensor<32x32x3x3xf32>
    %v1806 = stablehlo.reshape %v394 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1807 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1808 = stablehlo.reduce(%v1806 init: %v1807) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1809 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1810 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1811 = stablehlo.multiply %v1809, %cb7m : tensor<32xf32>
    %v1812 = stablehlo.multiply %v1810, %v1808 : tensor<32xf32>
    %v1813 = stablehlo.add %v1811, %v1812 : tensor<32xf32>
    %v1814 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1815 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1816 = stablehlo.multiply %v1814, %cb7v : tensor<32xf32>
    %v1817 = stablehlo.multiply %v1808, %v1808 : tensor<32xf32>
    %v1818 = stablehlo.multiply %v1815, %v1817 : tensor<32xf32>
    %v1819 = stablehlo.add %v1816, %v1818 : tensor<32xf32>
    %v1820 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1821 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1822 = stablehlo.divide %v1813, %v1820 : tensor<32xf32>
    %v1823 = stablehlo.divide %v1819, %v1821 : tensor<32xf32>
    %v1824 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1825 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1826 = stablehlo.sqrt %v1823 : tensor<32xf32>
    %v1827 = stablehlo.add %v1826, %v1825 : tensor<32xf32>
    %v1828 = stablehlo.divide %v1822, %v1827 : tensor<32xf32>
    %v1829 = stablehlo.multiply %v1824, %v1828 : tensor<32xf32>
    %v1830 = stablehlo.subtract %cb7, %v1829 : tensor<32xf32>
    %v1831 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1832 = stablehlo.multiply %v1831, %v1824 : tensor<32xf32>
    %v1833 = stablehlo.multiply %v1832, %cb7 : tensor<32xf32>
    %v1834 = stablehlo.subtract %v1830, %v1833 : tensor<32xf32>
    %v1835 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1836 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1837 = stablehlo.multiply %v1835, %cb7m : tensor<32xf32>
    %v1838 = stablehlo.multiply %v1836, %v1808 : tensor<32xf32>
    %v1839 = stablehlo.add %v1837, %v1838 : tensor<32xf32>
    %v1840 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1841 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1842 = stablehlo.multiply %v1840, %cb7v : tensor<32xf32>
    %v1843 = stablehlo.multiply %v1808, %v1808 : tensor<32xf32>
    %v1844 = stablehlo.multiply %v1841, %v1843 : tensor<32xf32>
    %v1845 = stablehlo.add %v1842, %v1844 : tensor<32xf32>
    %v1846 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1847 = stablehlo.reshape %v211 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1848 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v1849 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v1850 = stablehlo.reduce(%v1847 init: %v1846) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1851 = stablehlo.broadcast_in_dim %v1850, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v1852 = stablehlo.divide %v1851, %v1848 : tensor<128x32x4x4xf32>
    %v1853 = stablehlo.subtract %v1847, %v1852 : tensor<128x32x4x4xf32>
    %v1854 = stablehlo.multiply %v1853, %v1853 : tensor<128x32x4x4xf32>
    %v1855 = stablehlo.reduce(%v1854 init: %v1846) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1856 = stablehlo.broadcast_in_dim %v1855, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v1857 = stablehlo.divide %v1856, %v1848 : tensor<128x32x4x4xf32>
    %v1858 = stablehlo.add %v1857, %v1849 : tensor<128x32x4x4xf32>
    %v1859 = stablehlo.rsqrt %v1858 : tensor<128x32x4x4xf32>
    %v1860 = stablehlo.multiply %v1853, %v1859 : tensor<128x32x4x4xf32>
    %v1861 = stablehlo.reshape %v364 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1862 = stablehlo.multiply %v1861, %v1860 : tensor<128x32x4x4xf32>
    %v1863 = stablehlo.reduce(%v1862 init: %v1846) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1864 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1865 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1866 = stablehlo.multiply %v1864, %g7m : tensor<32xf32>
    %v1867 = stablehlo.multiply %v1865, %v1863 : tensor<32xf32>
    %v1868 = stablehlo.add %v1866, %v1867 : tensor<32xf32>
    %v1869 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1870 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1871 = stablehlo.multiply %v1869, %g7v : tensor<32xf32>
    %v1872 = stablehlo.multiply %v1863, %v1863 : tensor<32xf32>
    %v1873 = stablehlo.multiply %v1870, %v1872 : tensor<32xf32>
    %v1874 = stablehlo.add %v1871, %v1873 : tensor<32xf32>
    %v1875 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1876 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1877 = stablehlo.divide %v1868, %v1875 : tensor<32xf32>
    %v1878 = stablehlo.divide %v1874, %v1876 : tensor<32xf32>
    %v1879 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1880 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1881 = stablehlo.sqrt %v1878 : tensor<32xf32>
    %v1882 = stablehlo.add %v1881, %v1880 : tensor<32xf32>
    %v1883 = stablehlo.divide %v1877, %v1882 : tensor<32xf32>
    %v1884 = stablehlo.multiply %v1879, %v1883 : tensor<32xf32>
    %v1885 = stablehlo.subtract %g7, %v1884 : tensor<32xf32>
    %v1886 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1887 = stablehlo.multiply %v1886, %v1879 : tensor<32xf32>
    %v1888 = stablehlo.multiply %v1887, %g7 : tensor<32xf32>
    %v1889 = stablehlo.subtract %v1885, %v1888 : tensor<32xf32>
    %v1890 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1891 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1892 = stablehlo.multiply %v1890, %g7m : tensor<32xf32>
    %v1893 = stablehlo.multiply %v1891, %v1863 : tensor<32xf32>
    %v1894 = stablehlo.add %v1892, %v1893 : tensor<32xf32>
    %v1895 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1896 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1897 = stablehlo.multiply %v1895, %g7v : tensor<32xf32>
    %v1898 = stablehlo.multiply %v1863, %v1863 : tensor<32xf32>
    %v1899 = stablehlo.multiply %v1896, %v1898 : tensor<32xf32>
    %v1900 = stablehlo.add %v1897, %v1899 : tensor<32xf32>
    %v1901 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1902 = stablehlo.reshape %v364 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1903 = stablehlo.reduce(%v1902 init: %v1901) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1904 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1905 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1906 = stablehlo.multiply %v1904, %bt7m : tensor<32xf32>
    %v1907 = stablehlo.multiply %v1905, %v1903 : tensor<32xf32>
    %v1908 = stablehlo.add %v1906, %v1907 : tensor<32xf32>
    %v1909 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1910 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1911 = stablehlo.multiply %v1909, %bt7v : tensor<32xf32>
    %v1912 = stablehlo.multiply %v1903, %v1903 : tensor<32xf32>
    %v1913 = stablehlo.multiply %v1910, %v1912 : tensor<32xf32>
    %v1914 = stablehlo.add %v1911, %v1913 : tensor<32xf32>
    %v1915 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1916 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1917 = stablehlo.divide %v1908, %v1915 : tensor<32xf32>
    %v1918 = stablehlo.divide %v1914, %v1916 : tensor<32xf32>
    %v1919 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1920 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1921 = stablehlo.sqrt %v1918 : tensor<32xf32>
    %v1922 = stablehlo.add %v1921, %v1920 : tensor<32xf32>
    %v1923 = stablehlo.divide %v1917, %v1922 : tensor<32xf32>
    %v1924 = stablehlo.multiply %v1919, %v1923 : tensor<32xf32>
    %v1925 = stablehlo.subtract %bt7, %v1924 : tensor<32xf32>
    %v1926 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1927 = stablehlo.multiply %v1926, %v1919 : tensor<32xf32>
    %v1928 = stablehlo.multiply %v1927, %bt7 : tensor<32xf32>
    %v1929 = stablehlo.subtract %v1925, %v1928 : tensor<32xf32>
    %v1930 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1931 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1932 = stablehlo.multiply %v1930, %bt7m : tensor<32xf32>
    %v1933 = stablehlo.multiply %v1931, %v1903 : tensor<32xf32>
    %v1934 = stablehlo.add %v1932, %v1933 : tensor<32xf32>
    %v1935 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1936 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1937 = stablehlo.multiply %v1935, %bt7v : tensor<32xf32>
    %v1938 = stablehlo.multiply %v1903, %v1903 : tensor<32xf32>
    %v1939 = stablehlo.multiply %v1936, %v1938 : tensor<32xf32>
    %v1940 = stablehlo.add %v1937, %v1939 : tensor<32xf32>
    %v1941 = stablehlo.reshape %v235 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1942 = stablehlo.reshape %v350 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1943 = stablehlo.transpose %v1941, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v1944 = stablehlo.transpose %v1942, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v1945 = stablehlo.convert %v1943 : (tensor<32x128x4x4xf32>) -> tensor<32x128x4x4xbf16>
    %v1946 = stablehlo.convert %v1944 : (tensor<32x128x4x4xf32>) -> tensor<32x128x4x4xbf16>
    %v1947 = stablehlo.convolution(%v1945, %v1946)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xbf16>, tensor<32x128x4x4xbf16>) -> tensor<32x32x3x3xbf16>
    %v1948 = stablehlo.convert %v1947 : (tensor<32x32x3x3xbf16>) -> tensor<32x32x3x3xf32>
    %v1949 = stablehlo.transpose %v1948, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v1950 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1951 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1952 = stablehlo.multiply %v1950, %W8m : tensor<32x32x3x3xf32>
    %v1953 = stablehlo.multiply %v1951, %v1949 : tensor<32x32x3x3xf32>
    %v1954 = stablehlo.add %v1952, %v1953 : tensor<32x32x3x3xf32>
    %v1955 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1956 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1957 = stablehlo.multiply %v1955, %W8v : tensor<32x32x3x3xf32>
    %v1958 = stablehlo.multiply %v1949, %v1949 : tensor<32x32x3x3xf32>
    %v1959 = stablehlo.multiply %v1956, %v1958 : tensor<32x32x3x3xf32>
    %v1960 = stablehlo.add %v1957, %v1959 : tensor<32x32x3x3xf32>
    %v1961 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1962 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1963 = stablehlo.divide %v1954, %v1961 : tensor<32x32x3x3xf32>
    %v1964 = stablehlo.divide %v1960, %v1962 : tensor<32x32x3x3xf32>
    %v1965 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1966 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1967 = stablehlo.sqrt %v1964 : tensor<32x32x3x3xf32>
    %v1968 = stablehlo.add %v1967, %v1966 : tensor<32x32x3x3xf32>
    %v1969 = stablehlo.divide %v1963, %v1968 : tensor<32x32x3x3xf32>
    %v1970 = stablehlo.multiply %v1965, %v1969 : tensor<32x32x3x3xf32>
    %v1971 = stablehlo.subtract %W8, %v1970 : tensor<32x32x3x3xf32>
    %v1972 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1973 = stablehlo.multiply %v1972, %v1965 : tensor<32x32x3x3xf32>
    %v1974 = stablehlo.multiply %v1973, %W8 : tensor<32x32x3x3xf32>
    %v1975 = stablehlo.subtract %v1971, %v1974 : tensor<32x32x3x3xf32>
    %v1976 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1977 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1978 = stablehlo.multiply %v1976, %W8m : tensor<32x32x3x3xf32>
    %v1979 = stablehlo.multiply %v1977, %v1949 : tensor<32x32x3x3xf32>
    %v1980 = stablehlo.add %v1978, %v1979 : tensor<32x32x3x3xf32>
    %v1981 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1982 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1983 = stablehlo.multiply %v1981, %W8v : tensor<32x32x3x3xf32>
    %v1984 = stablehlo.multiply %v1949, %v1949 : tensor<32x32x3x3xf32>
    %v1985 = stablehlo.multiply %v1982, %v1984 : tensor<32x32x3x3xf32>
    %v1986 = stablehlo.add %v1983, %v1985 : tensor<32x32x3x3xf32>
    %v1987 = stablehlo.reshape %v350 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1988 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1989 = stablehlo.reduce(%v1987 init: %v1988) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1990 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1991 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1992 = stablehlo.multiply %v1990, %cb8m : tensor<32xf32>
    %v1993 = stablehlo.multiply %v1991, %v1989 : tensor<32xf32>
    %v1994 = stablehlo.add %v1992, %v1993 : tensor<32xf32>
    %v1995 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1996 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1997 = stablehlo.multiply %v1995, %cb8v : tensor<32xf32>
    %v1998 = stablehlo.multiply %v1989, %v1989 : tensor<32xf32>
    %v1999 = stablehlo.multiply %v1996, %v1998 : tensor<32xf32>
    %v2000 = stablehlo.add %v1997, %v1999 : tensor<32xf32>
    %v2001 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2002 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2003 = stablehlo.divide %v1994, %v2001 : tensor<32xf32>
    %v2004 = stablehlo.divide %v2000, %v2002 : tensor<32xf32>
    %v2005 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2006 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2007 = stablehlo.sqrt %v2004 : tensor<32xf32>
    %v2008 = stablehlo.add %v2007, %v2006 : tensor<32xf32>
    %v2009 = stablehlo.divide %v2003, %v2008 : tensor<32xf32>
    %v2010 = stablehlo.multiply %v2005, %v2009 : tensor<32xf32>
    %v2011 = stablehlo.subtract %cb8, %v2010 : tensor<32xf32>
    %v2012 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2013 = stablehlo.multiply %v2012, %v2005 : tensor<32xf32>
    %v2014 = stablehlo.multiply %v2013, %cb8 : tensor<32xf32>
    %v2015 = stablehlo.subtract %v2011, %v2014 : tensor<32xf32>
    %v2016 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2017 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2018 = stablehlo.multiply %v2016, %cb8m : tensor<32xf32>
    %v2019 = stablehlo.multiply %v2017, %v1989 : tensor<32xf32>
    %v2020 = stablehlo.add %v2018, %v2019 : tensor<32xf32>
    %v2021 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2022 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2023 = stablehlo.multiply %v2021, %cb8v : tensor<32xf32>
    %v2024 = stablehlo.multiply %v1989, %v1989 : tensor<32xf32>
    %v2025 = stablehlo.multiply %v2022, %v2024 : tensor<32xf32>
    %v2026 = stablehlo.add %v2023, %v2025 : tensor<32xf32>
    %v2027 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2028 = stablehlo.reshape %v243 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v2029 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v2030 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v2031 = stablehlo.reduce(%v2028 init: %v2027) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v2032 = stablehlo.broadcast_in_dim %v2031, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v2033 = stablehlo.divide %v2032, %v2029 : tensor<128x32x4x4xf32>
    %v2034 = stablehlo.subtract %v2028, %v2033 : tensor<128x32x4x4xf32>
    %v2035 = stablehlo.multiply %v2034, %v2034 : tensor<128x32x4x4xf32>
    %v2036 = stablehlo.reduce(%v2035 init: %v2027) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v2037 = stablehlo.broadcast_in_dim %v2036, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v2038 = stablehlo.divide %v2037, %v2029 : tensor<128x32x4x4xf32>
    %v2039 = stablehlo.add %v2038, %v2030 : tensor<128x32x4x4xf32>
    %v2040 = stablehlo.rsqrt %v2039 : tensor<128x32x4x4xf32>
    %v2041 = stablehlo.multiply %v2034, %v2040 : tensor<128x32x4x4xf32>
    %v2042 = stablehlo.reshape %v320 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v2043 = stablehlo.multiply %v2042, %v2041 : tensor<128x32x4x4xf32>
    %v2044 = stablehlo.reduce(%v2043 init: %v2027) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v2045 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2046 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2047 = stablehlo.multiply %v2045, %g8m : tensor<32xf32>
    %v2048 = stablehlo.multiply %v2046, %v2044 : tensor<32xf32>
    %v2049 = stablehlo.add %v2047, %v2048 : tensor<32xf32>
    %v2050 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2051 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2052 = stablehlo.multiply %v2050, %g8v : tensor<32xf32>
    %v2053 = stablehlo.multiply %v2044, %v2044 : tensor<32xf32>
    %v2054 = stablehlo.multiply %v2051, %v2053 : tensor<32xf32>
    %v2055 = stablehlo.add %v2052, %v2054 : tensor<32xf32>
    %v2056 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2057 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2058 = stablehlo.divide %v2049, %v2056 : tensor<32xf32>
    %v2059 = stablehlo.divide %v2055, %v2057 : tensor<32xf32>
    %v2060 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2061 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2062 = stablehlo.sqrt %v2059 : tensor<32xf32>
    %v2063 = stablehlo.add %v2062, %v2061 : tensor<32xf32>
    %v2064 = stablehlo.divide %v2058, %v2063 : tensor<32xf32>
    %v2065 = stablehlo.multiply %v2060, %v2064 : tensor<32xf32>
    %v2066 = stablehlo.subtract %g8, %v2065 : tensor<32xf32>
    %v2067 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2068 = stablehlo.multiply %v2067, %v2060 : tensor<32xf32>
    %v2069 = stablehlo.multiply %v2068, %g8 : tensor<32xf32>
    %v2070 = stablehlo.subtract %v2066, %v2069 : tensor<32xf32>
    %v2071 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2072 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2073 = stablehlo.multiply %v2071, %g8m : tensor<32xf32>
    %v2074 = stablehlo.multiply %v2072, %v2044 : tensor<32xf32>
    %v2075 = stablehlo.add %v2073, %v2074 : tensor<32xf32>
    %v2076 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2077 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2078 = stablehlo.multiply %v2076, %g8v : tensor<32xf32>
    %v2079 = stablehlo.multiply %v2044, %v2044 : tensor<32xf32>
    %v2080 = stablehlo.multiply %v2077, %v2079 : tensor<32xf32>
    %v2081 = stablehlo.add %v2078, %v2080 : tensor<32xf32>
    %v2082 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2083 = stablehlo.reshape %v320 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v2084 = stablehlo.reduce(%v2083 init: %v2082) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v2085 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2086 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2087 = stablehlo.multiply %v2085, %bt8m : tensor<32xf32>
    %v2088 = stablehlo.multiply %v2086, %v2084 : tensor<32xf32>
    %v2089 = stablehlo.add %v2087, %v2088 : tensor<32xf32>
    %v2090 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2091 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2092 = stablehlo.multiply %v2090, %bt8v : tensor<32xf32>
    %v2093 = stablehlo.multiply %v2084, %v2084 : tensor<32xf32>
    %v2094 = stablehlo.multiply %v2091, %v2093 : tensor<32xf32>
    %v2095 = stablehlo.add %v2092, %v2094 : tensor<32xf32>
    %v2096 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2097 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2098 = stablehlo.divide %v2089, %v2096 : tensor<32xf32>
    %v2099 = stablehlo.divide %v2095, %v2097 : tensor<32xf32>
    %v2100 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2101 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2102 = stablehlo.sqrt %v2099 : tensor<32xf32>
    %v2103 = stablehlo.add %v2102, %v2101 : tensor<32xf32>
    %v2104 = stablehlo.divide %v2098, %v2103 : tensor<32xf32>
    %v2105 = stablehlo.multiply %v2100, %v2104 : tensor<32xf32>
    %v2106 = stablehlo.subtract %bt8, %v2105 : tensor<32xf32>
    %v2107 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2108 = stablehlo.multiply %v2107, %v2100 : tensor<32xf32>
    %v2109 = stablehlo.multiply %v2108, %bt8 : tensor<32xf32>
    %v2110 = stablehlo.subtract %v2106, %v2109 : tensor<32xf32>
    %v2111 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2112 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2113 = stablehlo.multiply %v2111, %bt8m : tensor<32xf32>
    %v2114 = stablehlo.multiply %v2112, %v2084 : tensor<32xf32>
    %v2115 = stablehlo.add %v2113, %v2114 : tensor<32xf32>
    %v2116 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2117 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2118 = stablehlo.multiply %v2116, %bt8v : tensor<32xf32>
    %v2119 = stablehlo.multiply %v2084, %v2084 : tensor<32xf32>
    %v2120 = stablehlo.multiply %v2117, %v2119 : tensor<32xf32>
    %v2121 = stablehlo.add %v2118, %v2120 : tensor<32xf32>
    %v2122 = stablehlo.dot_general %v271, %v306, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x512xf32>) -> tensor<128x512xf32>
    %v2123 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v2124 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v2125 = stablehlo.multiply %v2123, %W9m : tensor<128x512xf32>
    %v2126 = stablehlo.multiply %v2124, %v2122 : tensor<128x512xf32>
    %v2127 = stablehlo.add %v2125, %v2126 : tensor<128x512xf32>
    %v2128 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v2129 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v2130 = stablehlo.multiply %v2128, %W9v : tensor<128x512xf32>
    %v2131 = stablehlo.multiply %v2122, %v2122 : tensor<128x512xf32>
    %v2132 = stablehlo.multiply %v2129, %v2131 : tensor<128x512xf32>
    %v2133 = stablehlo.add %v2130, %v2132 : tensor<128x512xf32>
    %v2134 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v2135 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v2136 = stablehlo.divide %v2127, %v2134 : tensor<128x512xf32>
    %v2137 = stablehlo.divide %v2133, %v2135 : tensor<128x512xf32>
    %v2138 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v2139 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v2140 = stablehlo.sqrt %v2137 : tensor<128x512xf32>
    %v2141 = stablehlo.add %v2140, %v2139 : tensor<128x512xf32>
    %v2142 = stablehlo.divide %v2136, %v2141 : tensor<128x512xf32>
    %v2143 = stablehlo.multiply %v2138, %v2142 : tensor<128x512xf32>
    %v2144 = stablehlo.subtract %W9, %v2143 : tensor<128x512xf32>
    %v2145 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v2146 = stablehlo.multiply %v2145, %v2138 : tensor<128x512xf32>
    %v2147 = stablehlo.multiply %v2146, %W9 : tensor<128x512xf32>
    %v2148 = stablehlo.subtract %v2144, %v2147 : tensor<128x512xf32>
    %v2149 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v2150 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v2151 = stablehlo.multiply %v2149, %W9m : tensor<128x512xf32>
    %v2152 = stablehlo.multiply %v2150, %v2122 : tensor<128x512xf32>
    %v2153 = stablehlo.add %v2151, %v2152 : tensor<128x512xf32>
    %v2154 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v2155 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v2156 = stablehlo.multiply %v2154, %W9v : tensor<128x512xf32>
    %v2157 = stablehlo.multiply %v2122, %v2122 : tensor<128x512xf32>
    %v2158 = stablehlo.multiply %v2155, %v2157 : tensor<128x512xf32>
    %v2159 = stablehlo.add %v2156, %v2158 : tensor<128x512xf32>
    %v2160 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2161 = stablehlo.reduce(%v306 init: %v2160) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v2162 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2163 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2164 = stablehlo.multiply %v2162, %b9m : tensor<512xf32>
    %v2165 = stablehlo.multiply %v2163, %v2161 : tensor<512xf32>
    %v2166 = stablehlo.add %v2164, %v2165 : tensor<512xf32>
    %v2167 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2168 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2169 = stablehlo.multiply %v2167, %b9v : tensor<512xf32>
    %v2170 = stablehlo.multiply %v2161, %v2161 : tensor<512xf32>
    %v2171 = stablehlo.multiply %v2168, %v2170 : tensor<512xf32>
    %v2172 = stablehlo.add %v2169, %v2171 : tensor<512xf32>
    %v2173 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2174 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2175 = stablehlo.divide %v2166, %v2173 : tensor<512xf32>
    %v2176 = stablehlo.divide %v2172, %v2174 : tensor<512xf32>
    %v2177 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2178 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2179 = stablehlo.sqrt %v2176 : tensor<512xf32>
    %v2180 = stablehlo.add %v2179, %v2178 : tensor<512xf32>
    %v2181 = stablehlo.divide %v2175, %v2180 : tensor<512xf32>
    %v2182 = stablehlo.multiply %v2177, %v2181 : tensor<512xf32>
    %v2183 = stablehlo.subtract %b9, %v2182 : tensor<512xf32>
    %v2184 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2185 = stablehlo.multiply %v2184, %v2177 : tensor<512xf32>
    %v2186 = stablehlo.multiply %v2185, %b9 : tensor<512xf32>
    %v2187 = stablehlo.subtract %v2183, %v2186 : tensor<512xf32>
    %v2188 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2189 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2190 = stablehlo.multiply %v2188, %b9m : tensor<512xf32>
    %v2191 = stablehlo.multiply %v2189, %v2161 : tensor<512xf32>
    %v2192 = stablehlo.add %v2190, %v2191 : tensor<512xf32>
    %v2193 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2194 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2195 = stablehlo.multiply %v2193, %b9v : tensor<512xf32>
    %v2196 = stablehlo.multiply %v2161, %v2161 : tensor<512xf32>
    %v2197 = stablehlo.multiply %v2194, %v2196 : tensor<512xf32>
    %v2198 = stablehlo.add %v2195, %v2197 : tensor<512xf32>
    %v2199 = stablehlo.dot_general %v276, %v300, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<512x512xf32>
    %v2200 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v2201 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v2202 = stablehlo.multiply %v2200, %Wam : tensor<512x512xf32>
    %v2203 = stablehlo.multiply %v2201, %v2199 : tensor<512x512xf32>
    %v2204 = stablehlo.add %v2202, %v2203 : tensor<512x512xf32>
    %v2205 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v2206 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v2207 = stablehlo.multiply %v2205, %Wav : tensor<512x512xf32>
    %v2208 = stablehlo.multiply %v2199, %v2199 : tensor<512x512xf32>
    %v2209 = stablehlo.multiply %v2206, %v2208 : tensor<512x512xf32>
    %v2210 = stablehlo.add %v2207, %v2209 : tensor<512x512xf32>
    %v2211 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v2212 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v2213 = stablehlo.divide %v2204, %v2211 : tensor<512x512xf32>
    %v2214 = stablehlo.divide %v2210, %v2212 : tensor<512x512xf32>
    %v2215 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v2216 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v2217 = stablehlo.sqrt %v2214 : tensor<512x512xf32>
    %v2218 = stablehlo.add %v2217, %v2216 : tensor<512x512xf32>
    %v2219 = stablehlo.divide %v2213, %v2218 : tensor<512x512xf32>
    %v2220 = stablehlo.multiply %v2215, %v2219 : tensor<512x512xf32>
    %v2221 = stablehlo.subtract %Wa, %v2220 : tensor<512x512xf32>
    %v2222 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v2223 = stablehlo.multiply %v2222, %v2215 : tensor<512x512xf32>
    %v2224 = stablehlo.multiply %v2223, %Wa : tensor<512x512xf32>
    %v2225 = stablehlo.subtract %v2221, %v2224 : tensor<512x512xf32>
    %v2226 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v2227 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v2228 = stablehlo.multiply %v2226, %Wam : tensor<512x512xf32>
    %v2229 = stablehlo.multiply %v2227, %v2199 : tensor<512x512xf32>
    %v2230 = stablehlo.add %v2228, %v2229 : tensor<512x512xf32>
    %v2231 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v2232 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v2233 = stablehlo.multiply %v2231, %Wav : tensor<512x512xf32>
    %v2234 = stablehlo.multiply %v2199, %v2199 : tensor<512x512xf32>
    %v2235 = stablehlo.multiply %v2232, %v2234 : tensor<512x512xf32>
    %v2236 = stablehlo.add %v2233, %v2235 : tensor<512x512xf32>
    %v2237 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2238 = stablehlo.reduce(%v300 init: %v2237) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v2239 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2240 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2241 = stablehlo.multiply %v2239, %bam : tensor<512xf32>
    %v2242 = stablehlo.multiply %v2240, %v2238 : tensor<512xf32>
    %v2243 = stablehlo.add %v2241, %v2242 : tensor<512xf32>
    %v2244 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2245 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2246 = stablehlo.multiply %v2244, %bav : tensor<512xf32>
    %v2247 = stablehlo.multiply %v2238, %v2238 : tensor<512xf32>
    %v2248 = stablehlo.multiply %v2245, %v2247 : tensor<512xf32>
    %v2249 = stablehlo.add %v2246, %v2248 : tensor<512xf32>
    %v2250 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2251 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2252 = stablehlo.divide %v2243, %v2250 : tensor<512xf32>
    %v2253 = stablehlo.divide %v2249, %v2251 : tensor<512xf32>
    %v2254 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2255 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2256 = stablehlo.sqrt %v2253 : tensor<512xf32>
    %v2257 = stablehlo.add %v2256, %v2255 : tensor<512xf32>
    %v2258 = stablehlo.divide %v2252, %v2257 : tensor<512xf32>
    %v2259 = stablehlo.multiply %v2254, %v2258 : tensor<512xf32>
    %v2260 = stablehlo.subtract %ba, %v2259 : tensor<512xf32>
    %v2261 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2262 = stablehlo.multiply %v2261, %v2254 : tensor<512xf32>
    %v2263 = stablehlo.multiply %v2262, %ba : tensor<512xf32>
    %v2264 = stablehlo.subtract %v2260, %v2263 : tensor<512xf32>
    %v2265 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2266 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2267 = stablehlo.multiply %v2265, %bam : tensor<512xf32>
    %v2268 = stablehlo.multiply %v2266, %v2238 : tensor<512xf32>
    %v2269 = stablehlo.add %v2267, %v2268 : tensor<512xf32>
    %v2270 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2271 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2272 = stablehlo.multiply %v2270, %bav : tensor<512xf32>
    %v2273 = stablehlo.multiply %v2238, %v2238 : tensor<512xf32>
    %v2274 = stablehlo.multiply %v2271, %v2273 : tensor<512xf32>
    %v2275 = stablehlo.add %v2272, %v2274 : tensor<512xf32>
    %v2276 = stablehlo.dot_general %v281, %v294, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x10xf32>) -> tensor<512x10xf32>
    %v2277 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v2278 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v2279 = stablehlo.multiply %v2277, %Wbm : tensor<512x10xf32>
    %v2280 = stablehlo.multiply %v2278, %v2276 : tensor<512x10xf32>
    %v2281 = stablehlo.add %v2279, %v2280 : tensor<512x10xf32>
    %v2282 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v2283 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v2284 = stablehlo.multiply %v2282, %Wbv : tensor<512x10xf32>
    %v2285 = stablehlo.multiply %v2276, %v2276 : tensor<512x10xf32>
    %v2286 = stablehlo.multiply %v2283, %v2285 : tensor<512x10xf32>
    %v2287 = stablehlo.add %v2284, %v2286 : tensor<512x10xf32>
    %v2288 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v2289 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v2290 = stablehlo.divide %v2281, %v2288 : tensor<512x10xf32>
    %v2291 = stablehlo.divide %v2287, %v2289 : tensor<512x10xf32>
    %v2292 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v2293 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v2294 = stablehlo.sqrt %v2291 : tensor<512x10xf32>
    %v2295 = stablehlo.add %v2294, %v2293 : tensor<512x10xf32>
    %v2296 = stablehlo.divide %v2290, %v2295 : tensor<512x10xf32>
    %v2297 = stablehlo.multiply %v2292, %v2296 : tensor<512x10xf32>
    %v2298 = stablehlo.subtract %Wb, %v2297 : tensor<512x10xf32>
    %v2299 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v2300 = stablehlo.multiply %v2299, %v2292 : tensor<512x10xf32>
    %v2301 = stablehlo.multiply %v2300, %Wb : tensor<512x10xf32>
    %v2302 = stablehlo.subtract %v2298, %v2301 : tensor<512x10xf32>
    %v2303 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v2304 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v2305 = stablehlo.multiply %v2303, %Wbm : tensor<512x10xf32>
    %v2306 = stablehlo.multiply %v2304, %v2276 : tensor<512x10xf32>
    %v2307 = stablehlo.add %v2305, %v2306 : tensor<512x10xf32>
    %v2308 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v2309 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v2310 = stablehlo.multiply %v2308, %Wbv : tensor<512x10xf32>
    %v2311 = stablehlo.multiply %v2276, %v2276 : tensor<512x10xf32>
    %v2312 = stablehlo.multiply %v2309, %v2311 : tensor<512x10xf32>
    %v2313 = stablehlo.add %v2310, %v2312 : tensor<512x10xf32>
    %v2314 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2315 = stablehlo.reduce(%v294 init: %v2314) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v2316 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2317 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2318 = stablehlo.multiply %v2316, %bbm : tensor<10xf32>
    %v2319 = stablehlo.multiply %v2317, %v2315 : tensor<10xf32>
    %v2320 = stablehlo.add %v2318, %v2319 : tensor<10xf32>
    %v2321 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2322 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2323 = stablehlo.multiply %v2321, %bbv : tensor<10xf32>
    %v2324 = stablehlo.multiply %v2315, %v2315 : tensor<10xf32>
    %v2325 = stablehlo.multiply %v2322, %v2324 : tensor<10xf32>
    %v2326 = stablehlo.add %v2323, %v2325 : tensor<10xf32>
    %v2327 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2328 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2329 = stablehlo.divide %v2320, %v2327 : tensor<10xf32>
    %v2330 = stablehlo.divide %v2326, %v2328 : tensor<10xf32>
    %v2331 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2332 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2333 = stablehlo.sqrt %v2330 : tensor<10xf32>
    %v2334 = stablehlo.add %v2333, %v2332 : tensor<10xf32>
    %v2335 = stablehlo.divide %v2329, %v2334 : tensor<10xf32>
    %v2336 = stablehlo.multiply %v2331, %v2335 : tensor<10xf32>
    %v2337 = stablehlo.subtract %bb, %v2336 : tensor<10xf32>
    %v2338 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2339 = stablehlo.multiply %v2338, %v2331 : tensor<10xf32>
    %v2340 = stablehlo.multiply %v2339, %bb : tensor<10xf32>
    %v2341 = stablehlo.subtract %v2337, %v2340 : tensor<10xf32>
    %v2342 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2343 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2344 = stablehlo.multiply %v2342, %bbm : tensor<10xf32>
    %v2345 = stablehlo.multiply %v2343, %v2315 : tensor<10xf32>
    %v2346 = stablehlo.add %v2344, %v2345 : tensor<10xf32>
    %v2347 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2348 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2349 = stablehlo.multiply %v2347, %bbv : tensor<10xf32>
    %v2350 = stablehlo.multiply %v2315, %v2315 : tensor<10xf32>
    %v2351 = stablehlo.multiply %v2348, %v2350 : tensor<10xf32>
    %v2352 = stablehlo.add %v2349, %v2351 : tensor<10xf32>
    return %v708, %v748, %v803, %v843, %v889, %v929, %v984, %v1024, %v1070, %v1110, %v1165, %v1205, %v1251, %v1291, %v1346, %v1386, %v1432, %v1472, %v1527, %v1567, %v1613, %v1653, %v1708, %v1748, %v1794, %v1834, %v1889, %v1929, %v1975, %v2015, %v2070, %v2110, %v2148, %v2187, %v2225, %v2264, %v2302, %v2341, %v713, %v753, %v808, %v848, %v894, %v934, %v989, %v1029, %v1075, %v1115, %v1170, %v1210, %v1256, %v1296, %v1351, %v1391, %v1437, %v1477, %v1532, %v1572, %v1618, %v1658, %v1713, %v1753, %v1799, %v1839, %v1894, %v1934, %v1980, %v2020, %v2075, %v2115, %v2153, %v2192, %v2230, %v2269, %v2307, %v2346, %v719, %v759, %v814, %v854, %v900, %v940, %v995, %v1035, %v1081, %v1121, %v1176, %v1216, %v1262, %v1302, %v1357, %v1397, %v1443, %v1483, %v1538, %v1578, %v1624, %v1664, %v1719, %v1759, %v1805, %v1845, %v1900, %v1940, %v1986, %v2026, %v2081, %v2121, %v2159, %v2198, %v2236, %v2275, %v2313, %v2352, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
