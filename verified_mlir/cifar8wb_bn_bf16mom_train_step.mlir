module @m {
  func.func @cifar8wb_bn_bf16mom_train_step(%x: tensor<128x3072xf32>, %W1: tensor<16x3x3x3xf32>, %cb1: tensor<16xf32>, %g1: tensor<16xf32>, %bt1: tensor<16xf32>, %W2: tensor<16x16x3x3xf32>, %cb2: tensor<16xf32>, %g2: tensor<16xf32>, %bt2: tensor<16xf32>, %W3: tensor<16x16x3x3xf32>, %cb3: tensor<16xf32>, %g3: tensor<16xf32>, %bt3: tensor<16xf32>, %W4: tensor<16x16x3x3xf32>, %cb4: tensor<16xf32>, %g4: tensor<16xf32>, %bt4: tensor<16xf32>, %W5: tensor<32x16x3x3xf32>, %cb5: tensor<32xf32>, %g5: tensor<32xf32>, %bt5: tensor<32xf32>, %W6: tensor<32x32x3x3xf32>, %cb6: tensor<32xf32>, %g6: tensor<32xf32>, %bt6: tensor<32xf32>, %W7: tensor<32x32x3x3xf32>, %cb7: tensor<32xf32>, %g7: tensor<32xf32>, %bt7: tensor<32xf32>, %W8: tensor<32x32x3x3xf32>, %cb8: tensor<32xf32>, %g8: tensor<32xf32>, %bt8: tensor<32xf32>, %W9: tensor<128x512xf32>, %b9: tensor<512xf32>, %Wa: tensor<512x512xf32>, %ba: tensor<512xf32>, %Wb: tensor<512x10xf32>, %bb: tensor<10xf32>, %W1m: tensor<16x3x3x3xf32>, %cb1m: tensor<16xf32>, %g1m: tensor<16xf32>, %bt1m: tensor<16xf32>, %W2m: tensor<16x16x3x3xf32>, %cb2m: tensor<16xf32>, %g2m: tensor<16xf32>, %bt2m: tensor<16xf32>, %W3m: tensor<16x16x3x3xf32>, %cb3m: tensor<16xf32>, %g3m: tensor<16xf32>, %bt3m: tensor<16xf32>, %W4m: tensor<16x16x3x3xf32>, %cb4m: tensor<16xf32>, %g4m: tensor<16xf32>, %bt4m: tensor<16xf32>, %W5m: tensor<32x16x3x3xf32>, %cb5m: tensor<32xf32>, %g5m: tensor<32xf32>, %bt5m: tensor<32xf32>, %W6m: tensor<32x32x3x3xf32>, %cb6m: tensor<32xf32>, %g6m: tensor<32xf32>, %bt6m: tensor<32xf32>, %W7m: tensor<32x32x3x3xf32>, %cb7m: tensor<32xf32>, %g7m: tensor<32xf32>, %bt7m: tensor<32xf32>, %W8m: tensor<32x32x3x3xf32>, %cb8m: tensor<32xf32>, %g8m: tensor<32xf32>, %bt8m: tensor<32xf32>, %W9m: tensor<128x512xf32>, %b9m: tensor<512xf32>, %Wam: tensor<512x512xf32>, %bam: tensor<512xf32>, %Wbm: tensor<512x10xf32>, %bbm: tensor<10xf32>, %W1v: tensor<16x3x3x3xf32>, %cb1v: tensor<16xf32>, %g1v: tensor<16xf32>, %bt1v: tensor<16xf32>, %W2v: tensor<16x16x3x3xf32>, %cb2v: tensor<16xf32>, %g2v: tensor<16xf32>, %bt2v: tensor<16xf32>, %W3v: tensor<16x16x3x3xf32>, %cb3v: tensor<16xf32>, %g3v: tensor<16xf32>, %bt3v: tensor<16xf32>, %W4v: tensor<16x16x3x3xf32>, %cb4v: tensor<16xf32>, %g4v: tensor<16xf32>, %bt4v: tensor<16xf32>, %W5v: tensor<32x16x3x3xf32>, %cb5v: tensor<32xf32>, %g5v: tensor<32xf32>, %bt5v: tensor<32xf32>, %W6v: tensor<32x32x3x3xf32>, %cb6v: tensor<32xf32>, %g6v: tensor<32xf32>, %bt6v: tensor<32xf32>, %W7v: tensor<32x32x3x3xf32>, %cb7v: tensor<32xf32>, %g7v: tensor<32xf32>, %bt7v: tensor<32xf32>, %W8v: tensor<32x32x3x3xf32>, %cb8v: tensor<32xf32>, %g8v: tensor<32xf32>, %bt8v: tensor<32xf32>, %W9v: tensor<128x512xf32>, %b9v: tensor<512xf32>, %Wav: tensor<512x512xf32>, %bav: tensor<512xf32>, %Wbv: tensor<512x10xf32>, %bbv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: tensor<128x10xf32>) -> (tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>) {
    // ── cifar8-BN train step, BATCHED op family: every line is pretty(verified AST
    //    node), except the marked report-only loss + the %bc passthroughs ──
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
    %v683 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v684 = stablehlo.multiply %v683, %W1v : tensor<16x3x3x3xf32>
    %v685 = stablehlo.add %v684, %v682 : tensor<16x3x3x3xf32>
    %v686 = stablehlo.multiply %v683, %v685 : tensor<16x3x3x3xf32>
    %v687 = stablehlo.add %v686, %v682 : tensor<16x3x3x3xf32>
    %v688 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v689 = stablehlo.multiply %v688, %v687 : tensor<16x3x3x3xf32>
    %v690 = stablehlo.subtract %W1, %v689 : tensor<16x3x3x3xf32>
    %v691 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v692 = stablehlo.multiply %v691, %W1v : tensor<16x3x3x3xf32>
    %v693 = stablehlo.add %v692, %v682 : tensor<16x3x3x3xf32>
    %v694 = stablehlo.reshape %v673 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v695 = stablehlo.constant dense<0.0> : tensor<f32>
    %v696 = stablehlo.reduce(%v694 init: %v695) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v697 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v698 = stablehlo.multiply %v697, %cb1v : tensor<16xf32>
    %v699 = stablehlo.add %v698, %v696 : tensor<16xf32>
    %v700 = stablehlo.multiply %v697, %v699 : tensor<16xf32>
    %v701 = stablehlo.add %v700, %v696 : tensor<16xf32>
    %v702 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v703 = stablehlo.multiply %v702, %v701 : tensor<16xf32>
    %v704 = stablehlo.subtract %cb1, %v703 : tensor<16xf32>
    %v705 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v706 = stablehlo.multiply %v705, %cb1v : tensor<16xf32>
    %v707 = stablehlo.add %v706, %v696 : tensor<16xf32>
    %v708 = stablehlo.constant dense<0.0> : tensor<f32>
    %v709 = stablehlo.reshape %v7 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v710 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v711 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v712 = stablehlo.reduce(%v709 init: %v708) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v713 = stablehlo.broadcast_in_dim %v712, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v714 = stablehlo.divide %v713, %v710 : tensor<128x16x32x32xf32>
    %v715 = stablehlo.subtract %v709, %v714 : tensor<128x16x32x32xf32>
    %v716 = stablehlo.multiply %v715, %v715 : tensor<128x16x32x32xf32>
    %v717 = stablehlo.reduce(%v716 init: %v708) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v718 = stablehlo.broadcast_in_dim %v717, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v719 = stablehlo.divide %v718, %v710 : tensor<128x16x32x32xf32>
    %v720 = stablehlo.add %v719, %v711 : tensor<128x16x32x32xf32>
    %v721 = stablehlo.rsqrt %v720 : tensor<128x16x32x32xf32>
    %v722 = stablehlo.multiply %v715, %v721 : tensor<128x16x32x32xf32>
    %v723 = stablehlo.reshape %v643 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v724 = stablehlo.multiply %v723, %v722 : tensor<128x16x32x32xf32>
    %v725 = stablehlo.reduce(%v724 init: %v708) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v726 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v727 = stablehlo.multiply %v726, %g1v : tensor<16xf32>
    %v728 = stablehlo.add %v727, %v725 : tensor<16xf32>
    %v729 = stablehlo.multiply %v726, %v728 : tensor<16xf32>
    %v730 = stablehlo.add %v729, %v725 : tensor<16xf32>
    %v731 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v732 = stablehlo.multiply %v731, %v730 : tensor<16xf32>
    %v733 = stablehlo.subtract %g1, %v732 : tensor<16xf32>
    %v734 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v735 = stablehlo.multiply %v734, %g1v : tensor<16xf32>
    %v736 = stablehlo.add %v735, %v725 : tensor<16xf32>
    %v737 = stablehlo.constant dense<0.0> : tensor<f32>
    %v738 = stablehlo.reshape %v643 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v739 = stablehlo.reduce(%v738 init: %v737) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v740 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v741 = stablehlo.multiply %v740, %bt1v : tensor<16xf32>
    %v742 = stablehlo.add %v741, %v739 : tensor<16xf32>
    %v743 = stablehlo.multiply %v740, %v742 : tensor<16xf32>
    %v744 = stablehlo.add %v743, %v739 : tensor<16xf32>
    %v745 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v746 = stablehlo.multiply %v745, %v744 : tensor<16xf32>
    %v747 = stablehlo.subtract %bt1, %v746 : tensor<16xf32>
    %v748 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v749 = stablehlo.multiply %v748, %bt1v : tensor<16xf32>
    %v750 = stablehlo.add %v749, %v739 : tensor<16xf32>
    %v751 = stablehlo.reshape %v31 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v752 = stablehlo.reshape %v629 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v753 = stablehlo.transpose %v751, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v754 = stablehlo.transpose %v752, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v755 = stablehlo.convert %v753 : (tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xbf16>
    %v756 = stablehlo.convert %v754 : (tensor<16x128x32x32xf32>) -> tensor<16x128x32x32xbf16>
    %v757 = stablehlo.convolution(%v755, %v756)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xbf16>, tensor<16x128x32x32xbf16>) -> tensor<16x16x3x3xbf16>
    %v758 = stablehlo.convert %v757 : (tensor<16x16x3x3xbf16>) -> tensor<16x16x3x3xf32>
    %v759 = stablehlo.transpose %v758, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v760 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v761 = stablehlo.multiply %v760, %W2v : tensor<16x16x3x3xf32>
    %v762 = stablehlo.add %v761, %v759 : tensor<16x16x3x3xf32>
    %v763 = stablehlo.multiply %v760, %v762 : tensor<16x16x3x3xf32>
    %v764 = stablehlo.add %v763, %v759 : tensor<16x16x3x3xf32>
    %v765 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v766 = stablehlo.multiply %v765, %v764 : tensor<16x16x3x3xf32>
    %v767 = stablehlo.subtract %W2, %v766 : tensor<16x16x3x3xf32>
    %v768 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v769 = stablehlo.multiply %v768, %W2v : tensor<16x16x3x3xf32>
    %v770 = stablehlo.add %v769, %v759 : tensor<16x16x3x3xf32>
    %v771 = stablehlo.reshape %v629 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v772 = stablehlo.constant dense<0.0> : tensor<f32>
    %v773 = stablehlo.reduce(%v771 init: %v772) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v774 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v775 = stablehlo.multiply %v774, %cb2v : tensor<16xf32>
    %v776 = stablehlo.add %v775, %v773 : tensor<16xf32>
    %v777 = stablehlo.multiply %v774, %v776 : tensor<16xf32>
    %v778 = stablehlo.add %v777, %v773 : tensor<16xf32>
    %v779 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v780 = stablehlo.multiply %v779, %v778 : tensor<16xf32>
    %v781 = stablehlo.subtract %cb2, %v780 : tensor<16xf32>
    %v782 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v783 = stablehlo.multiply %v782, %cb2v : tensor<16xf32>
    %v784 = stablehlo.add %v783, %v773 : tensor<16xf32>
    %v785 = stablehlo.constant dense<0.0> : tensor<f32>
    %v786 = stablehlo.reshape %v39 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v787 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v788 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v789 = stablehlo.reduce(%v786 init: %v785) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v790 = stablehlo.broadcast_in_dim %v789, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v791 = stablehlo.divide %v790, %v787 : tensor<128x16x32x32xf32>
    %v792 = stablehlo.subtract %v786, %v791 : tensor<128x16x32x32xf32>
    %v793 = stablehlo.multiply %v792, %v792 : tensor<128x16x32x32xf32>
    %v794 = stablehlo.reduce(%v793 init: %v785) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v795 = stablehlo.broadcast_in_dim %v794, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v796 = stablehlo.divide %v795, %v787 : tensor<128x16x32x32xf32>
    %v797 = stablehlo.add %v796, %v788 : tensor<128x16x32x32xf32>
    %v798 = stablehlo.rsqrt %v797 : tensor<128x16x32x32xf32>
    %v799 = stablehlo.multiply %v792, %v798 : tensor<128x16x32x32xf32>
    %v800 = stablehlo.reshape %v599 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v801 = stablehlo.multiply %v800, %v799 : tensor<128x16x32x32xf32>
    %v802 = stablehlo.reduce(%v801 init: %v785) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v803 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v804 = stablehlo.multiply %v803, %g2v : tensor<16xf32>
    %v805 = stablehlo.add %v804, %v802 : tensor<16xf32>
    %v806 = stablehlo.multiply %v803, %v805 : tensor<16xf32>
    %v807 = stablehlo.add %v806, %v802 : tensor<16xf32>
    %v808 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v809 = stablehlo.multiply %v808, %v807 : tensor<16xf32>
    %v810 = stablehlo.subtract %g2, %v809 : tensor<16xf32>
    %v811 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v812 = stablehlo.multiply %v811, %g2v : tensor<16xf32>
    %v813 = stablehlo.add %v812, %v802 : tensor<16xf32>
    %v814 = stablehlo.constant dense<0.0> : tensor<f32>
    %v815 = stablehlo.reshape %v599 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v816 = stablehlo.reduce(%v815 init: %v814) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v817 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v818 = stablehlo.multiply %v817, %bt2v : tensor<16xf32>
    %v819 = stablehlo.add %v818, %v816 : tensor<16xf32>
    %v820 = stablehlo.multiply %v817, %v819 : tensor<16xf32>
    %v821 = stablehlo.add %v820, %v816 : tensor<16xf32>
    %v822 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v823 = stablehlo.multiply %v822, %v821 : tensor<16xf32>
    %v824 = stablehlo.subtract %bt2, %v823 : tensor<16xf32>
    %v825 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v826 = stablehlo.multiply %v825, %bt2v : tensor<16xf32>
    %v827 = stablehlo.add %v826, %v816 : tensor<16xf32>
    %v828 = stablehlo.reshape %v67 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v829 = stablehlo.reshape %v580 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v830 = stablehlo.transpose %v828, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v831 = stablehlo.transpose %v829, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v832 = stablehlo.convert %v830 : (tensor<16x128x16x16xf32>) -> tensor<16x128x16x16xbf16>
    %v833 = stablehlo.convert %v831 : (tensor<16x128x16x16xf32>) -> tensor<16x128x16x16xbf16>
    %v834 = stablehlo.convolution(%v832, %v833)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xbf16>, tensor<16x128x16x16xbf16>) -> tensor<16x16x3x3xbf16>
    %v835 = stablehlo.convert %v834 : (tensor<16x16x3x3xbf16>) -> tensor<16x16x3x3xf32>
    %v836 = stablehlo.transpose %v835, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v837 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v838 = stablehlo.multiply %v837, %W3v : tensor<16x16x3x3xf32>
    %v839 = stablehlo.add %v838, %v836 : tensor<16x16x3x3xf32>
    %v840 = stablehlo.multiply %v837, %v839 : tensor<16x16x3x3xf32>
    %v841 = stablehlo.add %v840, %v836 : tensor<16x16x3x3xf32>
    %v842 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v843 = stablehlo.multiply %v842, %v841 : tensor<16x16x3x3xf32>
    %v844 = stablehlo.subtract %W3, %v843 : tensor<16x16x3x3xf32>
    %v845 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v846 = stablehlo.multiply %v845, %W3v : tensor<16x16x3x3xf32>
    %v847 = stablehlo.add %v846, %v836 : tensor<16x16x3x3xf32>
    %v848 = stablehlo.reshape %v580 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v849 = stablehlo.constant dense<0.0> : tensor<f32>
    %v850 = stablehlo.reduce(%v848 init: %v849) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v851 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v852 = stablehlo.multiply %v851, %cb3v : tensor<16xf32>
    %v853 = stablehlo.add %v852, %v850 : tensor<16xf32>
    %v854 = stablehlo.multiply %v851, %v853 : tensor<16xf32>
    %v855 = stablehlo.add %v854, %v850 : tensor<16xf32>
    %v856 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v857 = stablehlo.multiply %v856, %v855 : tensor<16xf32>
    %v858 = stablehlo.subtract %cb3, %v857 : tensor<16xf32>
    %v859 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v860 = stablehlo.multiply %v859, %cb3v : tensor<16xf32>
    %v861 = stablehlo.add %v860, %v850 : tensor<16xf32>
    %v862 = stablehlo.constant dense<0.0> : tensor<f32>
    %v863 = stablehlo.reshape %v75 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v864 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v865 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v866 = stablehlo.reduce(%v863 init: %v862) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v867 = stablehlo.broadcast_in_dim %v866, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v868 = stablehlo.divide %v867, %v864 : tensor<128x16x16x16xf32>
    %v869 = stablehlo.subtract %v863, %v868 : tensor<128x16x16x16xf32>
    %v870 = stablehlo.multiply %v869, %v869 : tensor<128x16x16x16xf32>
    %v871 = stablehlo.reduce(%v870 init: %v862) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v872 = stablehlo.broadcast_in_dim %v871, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v873 = stablehlo.divide %v872, %v864 : tensor<128x16x16x16xf32>
    %v874 = stablehlo.add %v873, %v865 : tensor<128x16x16x16xf32>
    %v875 = stablehlo.rsqrt %v874 : tensor<128x16x16x16xf32>
    %v876 = stablehlo.multiply %v869, %v875 : tensor<128x16x16x16xf32>
    %v877 = stablehlo.reshape %v550 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v878 = stablehlo.multiply %v877, %v876 : tensor<128x16x16x16xf32>
    %v879 = stablehlo.reduce(%v878 init: %v862) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v880 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v881 = stablehlo.multiply %v880, %g3v : tensor<16xf32>
    %v882 = stablehlo.add %v881, %v879 : tensor<16xf32>
    %v883 = stablehlo.multiply %v880, %v882 : tensor<16xf32>
    %v884 = stablehlo.add %v883, %v879 : tensor<16xf32>
    %v885 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v886 = stablehlo.multiply %v885, %v884 : tensor<16xf32>
    %v887 = stablehlo.subtract %g3, %v886 : tensor<16xf32>
    %v888 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v889 = stablehlo.multiply %v888, %g3v : tensor<16xf32>
    %v890 = stablehlo.add %v889, %v879 : tensor<16xf32>
    %v891 = stablehlo.constant dense<0.0> : tensor<f32>
    %v892 = stablehlo.reshape %v550 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v893 = stablehlo.reduce(%v892 init: %v891) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v894 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v895 = stablehlo.multiply %v894, %bt3v : tensor<16xf32>
    %v896 = stablehlo.add %v895, %v893 : tensor<16xf32>
    %v897 = stablehlo.multiply %v894, %v896 : tensor<16xf32>
    %v898 = stablehlo.add %v897, %v893 : tensor<16xf32>
    %v899 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v900 = stablehlo.multiply %v899, %v898 : tensor<16xf32>
    %v901 = stablehlo.subtract %bt3, %v900 : tensor<16xf32>
    %v902 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v903 = stablehlo.multiply %v902, %bt3v : tensor<16xf32>
    %v904 = stablehlo.add %v903, %v893 : tensor<16xf32>
    %v905 = stablehlo.reshape %v99 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v906 = stablehlo.reshape %v536 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v907 = stablehlo.transpose %v905, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v908 = stablehlo.transpose %v906, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v909 = stablehlo.convert %v907 : (tensor<16x128x16x16xf32>) -> tensor<16x128x16x16xbf16>
    %v910 = stablehlo.convert %v908 : (tensor<16x128x16x16xf32>) -> tensor<16x128x16x16xbf16>
    %v911 = stablehlo.convolution(%v909, %v910)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xbf16>, tensor<16x128x16x16xbf16>) -> tensor<16x16x3x3xbf16>
    %v912 = stablehlo.convert %v911 : (tensor<16x16x3x3xbf16>) -> tensor<16x16x3x3xf32>
    %v913 = stablehlo.transpose %v912, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v914 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v915 = stablehlo.multiply %v914, %W4v : tensor<16x16x3x3xf32>
    %v916 = stablehlo.add %v915, %v913 : tensor<16x16x3x3xf32>
    %v917 = stablehlo.multiply %v914, %v916 : tensor<16x16x3x3xf32>
    %v918 = stablehlo.add %v917, %v913 : tensor<16x16x3x3xf32>
    %v919 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v920 = stablehlo.multiply %v919, %v918 : tensor<16x16x3x3xf32>
    %v921 = stablehlo.subtract %W4, %v920 : tensor<16x16x3x3xf32>
    %v922 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v923 = stablehlo.multiply %v922, %W4v : tensor<16x16x3x3xf32>
    %v924 = stablehlo.add %v923, %v913 : tensor<16x16x3x3xf32>
    %v925 = stablehlo.reshape %v536 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v926 = stablehlo.constant dense<0.0> : tensor<f32>
    %v927 = stablehlo.reduce(%v925 init: %v926) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v928 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v929 = stablehlo.multiply %v928, %cb4v : tensor<16xf32>
    %v930 = stablehlo.add %v929, %v927 : tensor<16xf32>
    %v931 = stablehlo.multiply %v928, %v930 : tensor<16xf32>
    %v932 = stablehlo.add %v931, %v927 : tensor<16xf32>
    %v933 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v934 = stablehlo.multiply %v933, %v932 : tensor<16xf32>
    %v935 = stablehlo.subtract %cb4, %v934 : tensor<16xf32>
    %v936 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v937 = stablehlo.multiply %v936, %cb4v : tensor<16xf32>
    %v938 = stablehlo.add %v937, %v927 : tensor<16xf32>
    %v939 = stablehlo.constant dense<0.0> : tensor<f32>
    %v940 = stablehlo.reshape %v107 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v941 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v942 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v943 = stablehlo.reduce(%v940 init: %v939) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v944 = stablehlo.broadcast_in_dim %v943, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v945 = stablehlo.divide %v944, %v941 : tensor<128x16x16x16xf32>
    %v946 = stablehlo.subtract %v940, %v945 : tensor<128x16x16x16xf32>
    %v947 = stablehlo.multiply %v946, %v946 : tensor<128x16x16x16xf32>
    %v948 = stablehlo.reduce(%v947 init: %v939) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v949 = stablehlo.broadcast_in_dim %v948, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v950 = stablehlo.divide %v949, %v941 : tensor<128x16x16x16xf32>
    %v951 = stablehlo.add %v950, %v942 : tensor<128x16x16x16xf32>
    %v952 = stablehlo.rsqrt %v951 : tensor<128x16x16x16xf32>
    %v953 = stablehlo.multiply %v946, %v952 : tensor<128x16x16x16xf32>
    %v954 = stablehlo.reshape %v506 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v955 = stablehlo.multiply %v954, %v953 : tensor<128x16x16x16xf32>
    %v956 = stablehlo.reduce(%v955 init: %v939) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v957 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v958 = stablehlo.multiply %v957, %g4v : tensor<16xf32>
    %v959 = stablehlo.add %v958, %v956 : tensor<16xf32>
    %v960 = stablehlo.multiply %v957, %v959 : tensor<16xf32>
    %v961 = stablehlo.add %v960, %v956 : tensor<16xf32>
    %v962 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v963 = stablehlo.multiply %v962, %v961 : tensor<16xf32>
    %v964 = stablehlo.subtract %g4, %v963 : tensor<16xf32>
    %v965 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v966 = stablehlo.multiply %v965, %g4v : tensor<16xf32>
    %v967 = stablehlo.add %v966, %v956 : tensor<16xf32>
    %v968 = stablehlo.constant dense<0.0> : tensor<f32>
    %v969 = stablehlo.reshape %v506 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v970 = stablehlo.reduce(%v969 init: %v968) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v971 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v972 = stablehlo.multiply %v971, %bt4v : tensor<16xf32>
    %v973 = stablehlo.add %v972, %v970 : tensor<16xf32>
    %v974 = stablehlo.multiply %v971, %v973 : tensor<16xf32>
    %v975 = stablehlo.add %v974, %v970 : tensor<16xf32>
    %v976 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v977 = stablehlo.multiply %v976, %v975 : tensor<16xf32>
    %v978 = stablehlo.subtract %bt4, %v977 : tensor<16xf32>
    %v979 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v980 = stablehlo.multiply %v979, %bt4v : tensor<16xf32>
    %v981 = stablehlo.add %v980, %v970 : tensor<16xf32>
    %v982 = stablehlo.reshape %v135 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v983 = stablehlo.reshape %v487 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v984 = stablehlo.transpose %v982, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v985 = stablehlo.transpose %v983, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v986 = stablehlo.convert %v984 : (tensor<16x128x8x8xf32>) -> tensor<16x128x8x8xbf16>
    %v987 = stablehlo.convert %v985 : (tensor<32x128x8x8xf32>) -> tensor<32x128x8x8xbf16>
    %v988 = stablehlo.convolution(%v986, %v987)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xbf16>, tensor<32x128x8x8xbf16>) -> tensor<16x32x3x3xbf16>
    %v989 = stablehlo.convert %v988 : (tensor<16x32x3x3xbf16>) -> tensor<16x32x3x3xf32>
    %v990 = stablehlo.transpose %v989, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v991 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v992 = stablehlo.multiply %v991, %W5v : tensor<32x16x3x3xf32>
    %v993 = stablehlo.add %v992, %v990 : tensor<32x16x3x3xf32>
    %v994 = stablehlo.multiply %v991, %v993 : tensor<32x16x3x3xf32>
    %v995 = stablehlo.add %v994, %v990 : tensor<32x16x3x3xf32>
    %v996 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v997 = stablehlo.multiply %v996, %v995 : tensor<32x16x3x3xf32>
    %v998 = stablehlo.subtract %W5, %v997 : tensor<32x16x3x3xf32>
    %v999 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1000 = stablehlo.multiply %v999, %W5v : tensor<32x16x3x3xf32>
    %v1001 = stablehlo.add %v1000, %v990 : tensor<32x16x3x3xf32>
    %v1002 = stablehlo.reshape %v487 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1003 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1004 = stablehlo.reduce(%v1002 init: %v1003) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1005 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1006 = stablehlo.multiply %v1005, %cb5v : tensor<32xf32>
    %v1007 = stablehlo.add %v1006, %v1004 : tensor<32xf32>
    %v1008 = stablehlo.multiply %v1005, %v1007 : tensor<32xf32>
    %v1009 = stablehlo.add %v1008, %v1004 : tensor<32xf32>
    %v1010 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1011 = stablehlo.multiply %v1010, %v1009 : tensor<32xf32>
    %v1012 = stablehlo.subtract %cb5, %v1011 : tensor<32xf32>
    %v1013 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1014 = stablehlo.multiply %v1013, %cb5v : tensor<32xf32>
    %v1015 = stablehlo.add %v1014, %v1004 : tensor<32xf32>
    %v1016 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1017 = stablehlo.reshape %v143 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1018 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v1019 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v1020 = stablehlo.reduce(%v1017 init: %v1016) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1021 = stablehlo.broadcast_in_dim %v1020, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v1022 = stablehlo.divide %v1021, %v1018 : tensor<128x32x8x8xf32>
    %v1023 = stablehlo.subtract %v1017, %v1022 : tensor<128x32x8x8xf32>
    %v1024 = stablehlo.multiply %v1023, %v1023 : tensor<128x32x8x8xf32>
    %v1025 = stablehlo.reduce(%v1024 init: %v1016) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1026 = stablehlo.broadcast_in_dim %v1025, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v1027 = stablehlo.divide %v1026, %v1018 : tensor<128x32x8x8xf32>
    %v1028 = stablehlo.add %v1027, %v1019 : tensor<128x32x8x8xf32>
    %v1029 = stablehlo.rsqrt %v1028 : tensor<128x32x8x8xf32>
    %v1030 = stablehlo.multiply %v1023, %v1029 : tensor<128x32x8x8xf32>
    %v1031 = stablehlo.reshape %v457 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1032 = stablehlo.multiply %v1031, %v1030 : tensor<128x32x8x8xf32>
    %v1033 = stablehlo.reduce(%v1032 init: %v1016) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1034 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1035 = stablehlo.multiply %v1034, %g5v : tensor<32xf32>
    %v1036 = stablehlo.add %v1035, %v1033 : tensor<32xf32>
    %v1037 = stablehlo.multiply %v1034, %v1036 : tensor<32xf32>
    %v1038 = stablehlo.add %v1037, %v1033 : tensor<32xf32>
    %v1039 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1040 = stablehlo.multiply %v1039, %v1038 : tensor<32xf32>
    %v1041 = stablehlo.subtract %g5, %v1040 : tensor<32xf32>
    %v1042 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1043 = stablehlo.multiply %v1042, %g5v : tensor<32xf32>
    %v1044 = stablehlo.add %v1043, %v1033 : tensor<32xf32>
    %v1045 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1046 = stablehlo.reshape %v457 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1047 = stablehlo.reduce(%v1046 init: %v1045) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1048 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1049 = stablehlo.multiply %v1048, %bt5v : tensor<32xf32>
    %v1050 = stablehlo.add %v1049, %v1047 : tensor<32xf32>
    %v1051 = stablehlo.multiply %v1048, %v1050 : tensor<32xf32>
    %v1052 = stablehlo.add %v1051, %v1047 : tensor<32xf32>
    %v1053 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1054 = stablehlo.multiply %v1053, %v1052 : tensor<32xf32>
    %v1055 = stablehlo.subtract %bt5, %v1054 : tensor<32xf32>
    %v1056 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1057 = stablehlo.multiply %v1056, %bt5v : tensor<32xf32>
    %v1058 = stablehlo.add %v1057, %v1047 : tensor<32xf32>
    %v1059 = stablehlo.reshape %v167 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1060 = stablehlo.reshape %v443 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1061 = stablehlo.transpose %v1059, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v1062 = stablehlo.transpose %v1060, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v1063 = stablehlo.convert %v1061 : (tensor<32x128x8x8xf32>) -> tensor<32x128x8x8xbf16>
    %v1064 = stablehlo.convert %v1062 : (tensor<32x128x8x8xf32>) -> tensor<32x128x8x8xbf16>
    %v1065 = stablehlo.convolution(%v1063, %v1064)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xbf16>, tensor<32x128x8x8xbf16>) -> tensor<32x32x3x3xbf16>
    %v1066 = stablehlo.convert %v1065 : (tensor<32x32x3x3xbf16>) -> tensor<32x32x3x3xf32>
    %v1067 = stablehlo.transpose %v1066, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v1068 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1069 = stablehlo.multiply %v1068, %W6v : tensor<32x32x3x3xf32>
    %v1070 = stablehlo.add %v1069, %v1067 : tensor<32x32x3x3xf32>
    %v1071 = stablehlo.multiply %v1068, %v1070 : tensor<32x32x3x3xf32>
    %v1072 = stablehlo.add %v1071, %v1067 : tensor<32x32x3x3xf32>
    %v1073 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1074 = stablehlo.multiply %v1073, %v1072 : tensor<32x32x3x3xf32>
    %v1075 = stablehlo.subtract %W6, %v1074 : tensor<32x32x3x3xf32>
    %v1076 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1077 = stablehlo.multiply %v1076, %W6v : tensor<32x32x3x3xf32>
    %v1078 = stablehlo.add %v1077, %v1067 : tensor<32x32x3x3xf32>
    %v1079 = stablehlo.reshape %v443 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1080 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1081 = stablehlo.reduce(%v1079 init: %v1080) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1082 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1083 = stablehlo.multiply %v1082, %cb6v : tensor<32xf32>
    %v1084 = stablehlo.add %v1083, %v1081 : tensor<32xf32>
    %v1085 = stablehlo.multiply %v1082, %v1084 : tensor<32xf32>
    %v1086 = stablehlo.add %v1085, %v1081 : tensor<32xf32>
    %v1087 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1088 = stablehlo.multiply %v1087, %v1086 : tensor<32xf32>
    %v1089 = stablehlo.subtract %cb6, %v1088 : tensor<32xf32>
    %v1090 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1091 = stablehlo.multiply %v1090, %cb6v : tensor<32xf32>
    %v1092 = stablehlo.add %v1091, %v1081 : tensor<32xf32>
    %v1093 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1094 = stablehlo.reshape %v175 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1095 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v1096 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v1097 = stablehlo.reduce(%v1094 init: %v1093) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1098 = stablehlo.broadcast_in_dim %v1097, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v1099 = stablehlo.divide %v1098, %v1095 : tensor<128x32x8x8xf32>
    %v1100 = stablehlo.subtract %v1094, %v1099 : tensor<128x32x8x8xf32>
    %v1101 = stablehlo.multiply %v1100, %v1100 : tensor<128x32x8x8xf32>
    %v1102 = stablehlo.reduce(%v1101 init: %v1093) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1103 = stablehlo.broadcast_in_dim %v1102, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v1104 = stablehlo.divide %v1103, %v1095 : tensor<128x32x8x8xf32>
    %v1105 = stablehlo.add %v1104, %v1096 : tensor<128x32x8x8xf32>
    %v1106 = stablehlo.rsqrt %v1105 : tensor<128x32x8x8xf32>
    %v1107 = stablehlo.multiply %v1100, %v1106 : tensor<128x32x8x8xf32>
    %v1108 = stablehlo.reshape %v413 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1109 = stablehlo.multiply %v1108, %v1107 : tensor<128x32x8x8xf32>
    %v1110 = stablehlo.reduce(%v1109 init: %v1093) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1111 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1112 = stablehlo.multiply %v1111, %g6v : tensor<32xf32>
    %v1113 = stablehlo.add %v1112, %v1110 : tensor<32xf32>
    %v1114 = stablehlo.multiply %v1111, %v1113 : tensor<32xf32>
    %v1115 = stablehlo.add %v1114, %v1110 : tensor<32xf32>
    %v1116 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1117 = stablehlo.multiply %v1116, %v1115 : tensor<32xf32>
    %v1118 = stablehlo.subtract %g6, %v1117 : tensor<32xf32>
    %v1119 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1120 = stablehlo.multiply %v1119, %g6v : tensor<32xf32>
    %v1121 = stablehlo.add %v1120, %v1110 : tensor<32xf32>
    %v1122 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1123 = stablehlo.reshape %v413 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1124 = stablehlo.reduce(%v1123 init: %v1122) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1125 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1126 = stablehlo.multiply %v1125, %bt6v : tensor<32xf32>
    %v1127 = stablehlo.add %v1126, %v1124 : tensor<32xf32>
    %v1128 = stablehlo.multiply %v1125, %v1127 : tensor<32xf32>
    %v1129 = stablehlo.add %v1128, %v1124 : tensor<32xf32>
    %v1130 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1131 = stablehlo.multiply %v1130, %v1129 : tensor<32xf32>
    %v1132 = stablehlo.subtract %bt6, %v1131 : tensor<32xf32>
    %v1133 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1134 = stablehlo.multiply %v1133, %bt6v : tensor<32xf32>
    %v1135 = stablehlo.add %v1134, %v1124 : tensor<32xf32>
    %v1136 = stablehlo.reshape %v203 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1137 = stablehlo.reshape %v394 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1138 = stablehlo.transpose %v1136, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v1139 = stablehlo.transpose %v1137, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v1140 = stablehlo.convert %v1138 : (tensor<32x128x4x4xf32>) -> tensor<32x128x4x4xbf16>
    %v1141 = stablehlo.convert %v1139 : (tensor<32x128x4x4xf32>) -> tensor<32x128x4x4xbf16>
    %v1142 = stablehlo.convolution(%v1140, %v1141)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xbf16>, tensor<32x128x4x4xbf16>) -> tensor<32x32x3x3xbf16>
    %v1143 = stablehlo.convert %v1142 : (tensor<32x32x3x3xbf16>) -> tensor<32x32x3x3xf32>
    %v1144 = stablehlo.transpose %v1143, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v1145 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1146 = stablehlo.multiply %v1145, %W7v : tensor<32x32x3x3xf32>
    %v1147 = stablehlo.add %v1146, %v1144 : tensor<32x32x3x3xf32>
    %v1148 = stablehlo.multiply %v1145, %v1147 : tensor<32x32x3x3xf32>
    %v1149 = stablehlo.add %v1148, %v1144 : tensor<32x32x3x3xf32>
    %v1150 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1151 = stablehlo.multiply %v1150, %v1149 : tensor<32x32x3x3xf32>
    %v1152 = stablehlo.subtract %W7, %v1151 : tensor<32x32x3x3xf32>
    %v1153 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1154 = stablehlo.multiply %v1153, %W7v : tensor<32x32x3x3xf32>
    %v1155 = stablehlo.add %v1154, %v1144 : tensor<32x32x3x3xf32>
    %v1156 = stablehlo.reshape %v394 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1157 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1158 = stablehlo.reduce(%v1156 init: %v1157) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1159 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1160 = stablehlo.multiply %v1159, %cb7v : tensor<32xf32>
    %v1161 = stablehlo.add %v1160, %v1158 : tensor<32xf32>
    %v1162 = stablehlo.multiply %v1159, %v1161 : tensor<32xf32>
    %v1163 = stablehlo.add %v1162, %v1158 : tensor<32xf32>
    %v1164 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1165 = stablehlo.multiply %v1164, %v1163 : tensor<32xf32>
    %v1166 = stablehlo.subtract %cb7, %v1165 : tensor<32xf32>
    %v1167 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1168 = stablehlo.multiply %v1167, %cb7v : tensor<32xf32>
    %v1169 = stablehlo.add %v1168, %v1158 : tensor<32xf32>
    %v1170 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1171 = stablehlo.reshape %v211 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1172 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v1173 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v1174 = stablehlo.reduce(%v1171 init: %v1170) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1175 = stablehlo.broadcast_in_dim %v1174, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v1176 = stablehlo.divide %v1175, %v1172 : tensor<128x32x4x4xf32>
    %v1177 = stablehlo.subtract %v1171, %v1176 : tensor<128x32x4x4xf32>
    %v1178 = stablehlo.multiply %v1177, %v1177 : tensor<128x32x4x4xf32>
    %v1179 = stablehlo.reduce(%v1178 init: %v1170) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1180 = stablehlo.broadcast_in_dim %v1179, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v1181 = stablehlo.divide %v1180, %v1172 : tensor<128x32x4x4xf32>
    %v1182 = stablehlo.add %v1181, %v1173 : tensor<128x32x4x4xf32>
    %v1183 = stablehlo.rsqrt %v1182 : tensor<128x32x4x4xf32>
    %v1184 = stablehlo.multiply %v1177, %v1183 : tensor<128x32x4x4xf32>
    %v1185 = stablehlo.reshape %v364 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1186 = stablehlo.multiply %v1185, %v1184 : tensor<128x32x4x4xf32>
    %v1187 = stablehlo.reduce(%v1186 init: %v1170) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1188 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1189 = stablehlo.multiply %v1188, %g7v : tensor<32xf32>
    %v1190 = stablehlo.add %v1189, %v1187 : tensor<32xf32>
    %v1191 = stablehlo.multiply %v1188, %v1190 : tensor<32xf32>
    %v1192 = stablehlo.add %v1191, %v1187 : tensor<32xf32>
    %v1193 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1194 = stablehlo.multiply %v1193, %v1192 : tensor<32xf32>
    %v1195 = stablehlo.subtract %g7, %v1194 : tensor<32xf32>
    %v1196 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1197 = stablehlo.multiply %v1196, %g7v : tensor<32xf32>
    %v1198 = stablehlo.add %v1197, %v1187 : tensor<32xf32>
    %v1199 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1200 = stablehlo.reshape %v364 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1201 = stablehlo.reduce(%v1200 init: %v1199) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1202 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1203 = stablehlo.multiply %v1202, %bt7v : tensor<32xf32>
    %v1204 = stablehlo.add %v1203, %v1201 : tensor<32xf32>
    %v1205 = stablehlo.multiply %v1202, %v1204 : tensor<32xf32>
    %v1206 = stablehlo.add %v1205, %v1201 : tensor<32xf32>
    %v1207 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1208 = stablehlo.multiply %v1207, %v1206 : tensor<32xf32>
    %v1209 = stablehlo.subtract %bt7, %v1208 : tensor<32xf32>
    %v1210 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1211 = stablehlo.multiply %v1210, %bt7v : tensor<32xf32>
    %v1212 = stablehlo.add %v1211, %v1201 : tensor<32xf32>
    %v1213 = stablehlo.reshape %v235 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1214 = stablehlo.reshape %v350 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1215 = stablehlo.transpose %v1213, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v1216 = stablehlo.transpose %v1214, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v1217 = stablehlo.convert %v1215 : (tensor<32x128x4x4xf32>) -> tensor<32x128x4x4xbf16>
    %v1218 = stablehlo.convert %v1216 : (tensor<32x128x4x4xf32>) -> tensor<32x128x4x4xbf16>
    %v1219 = stablehlo.convolution(%v1217, %v1218)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xbf16>, tensor<32x128x4x4xbf16>) -> tensor<32x32x3x3xbf16>
    %v1220 = stablehlo.convert %v1219 : (tensor<32x32x3x3xbf16>) -> tensor<32x32x3x3xf32>
    %v1221 = stablehlo.transpose %v1220, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v1222 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1223 = stablehlo.multiply %v1222, %W8v : tensor<32x32x3x3xf32>
    %v1224 = stablehlo.add %v1223, %v1221 : tensor<32x32x3x3xf32>
    %v1225 = stablehlo.multiply %v1222, %v1224 : tensor<32x32x3x3xf32>
    %v1226 = stablehlo.add %v1225, %v1221 : tensor<32x32x3x3xf32>
    %v1227 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1228 = stablehlo.multiply %v1227, %v1226 : tensor<32x32x3x3xf32>
    %v1229 = stablehlo.subtract %W8, %v1228 : tensor<32x32x3x3xf32>
    %v1230 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1231 = stablehlo.multiply %v1230, %W8v : tensor<32x32x3x3xf32>
    %v1232 = stablehlo.add %v1231, %v1221 : tensor<32x32x3x3xf32>
    %v1233 = stablehlo.reshape %v350 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1234 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1235 = stablehlo.reduce(%v1233 init: %v1234) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1236 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1237 = stablehlo.multiply %v1236, %cb8v : tensor<32xf32>
    %v1238 = stablehlo.add %v1237, %v1235 : tensor<32xf32>
    %v1239 = stablehlo.multiply %v1236, %v1238 : tensor<32xf32>
    %v1240 = stablehlo.add %v1239, %v1235 : tensor<32xf32>
    %v1241 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1242 = stablehlo.multiply %v1241, %v1240 : tensor<32xf32>
    %v1243 = stablehlo.subtract %cb8, %v1242 : tensor<32xf32>
    %v1244 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1245 = stablehlo.multiply %v1244, %cb8v : tensor<32xf32>
    %v1246 = stablehlo.add %v1245, %v1235 : tensor<32xf32>
    %v1247 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1248 = stablehlo.reshape %v243 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1249 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v1250 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v1251 = stablehlo.reduce(%v1248 init: %v1247) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1252 = stablehlo.broadcast_in_dim %v1251, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v1253 = stablehlo.divide %v1252, %v1249 : tensor<128x32x4x4xf32>
    %v1254 = stablehlo.subtract %v1248, %v1253 : tensor<128x32x4x4xf32>
    %v1255 = stablehlo.multiply %v1254, %v1254 : tensor<128x32x4x4xf32>
    %v1256 = stablehlo.reduce(%v1255 init: %v1247) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1257 = stablehlo.broadcast_in_dim %v1256, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v1258 = stablehlo.divide %v1257, %v1249 : tensor<128x32x4x4xf32>
    %v1259 = stablehlo.add %v1258, %v1250 : tensor<128x32x4x4xf32>
    %v1260 = stablehlo.rsqrt %v1259 : tensor<128x32x4x4xf32>
    %v1261 = stablehlo.multiply %v1254, %v1260 : tensor<128x32x4x4xf32>
    %v1262 = stablehlo.reshape %v320 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1263 = stablehlo.multiply %v1262, %v1261 : tensor<128x32x4x4xf32>
    %v1264 = stablehlo.reduce(%v1263 init: %v1247) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1265 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1266 = stablehlo.multiply %v1265, %g8v : tensor<32xf32>
    %v1267 = stablehlo.add %v1266, %v1264 : tensor<32xf32>
    %v1268 = stablehlo.multiply %v1265, %v1267 : tensor<32xf32>
    %v1269 = stablehlo.add %v1268, %v1264 : tensor<32xf32>
    %v1270 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1271 = stablehlo.multiply %v1270, %v1269 : tensor<32xf32>
    %v1272 = stablehlo.subtract %g8, %v1271 : tensor<32xf32>
    %v1273 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1274 = stablehlo.multiply %v1273, %g8v : tensor<32xf32>
    %v1275 = stablehlo.add %v1274, %v1264 : tensor<32xf32>
    %v1276 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1277 = stablehlo.reshape %v320 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1278 = stablehlo.reduce(%v1277 init: %v1276) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1279 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1280 = stablehlo.multiply %v1279, %bt8v : tensor<32xf32>
    %v1281 = stablehlo.add %v1280, %v1278 : tensor<32xf32>
    %v1282 = stablehlo.multiply %v1279, %v1281 : tensor<32xf32>
    %v1283 = stablehlo.add %v1282, %v1278 : tensor<32xf32>
    %v1284 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1285 = stablehlo.multiply %v1284, %v1283 : tensor<32xf32>
    %v1286 = stablehlo.subtract %bt8, %v1285 : tensor<32xf32>
    %v1287 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1288 = stablehlo.multiply %v1287, %bt8v : tensor<32xf32>
    %v1289 = stablehlo.add %v1288, %v1278 : tensor<32xf32>
    %v1290 = stablehlo.dot_general %v271, %v306, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x512xf32>) -> tensor<128x512xf32>
    %v1291 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v1292 = stablehlo.multiply %v1291, %W9v : tensor<128x512xf32>
    %v1293 = stablehlo.add %v1292, %v1290 : tensor<128x512xf32>
    %v1294 = stablehlo.multiply %v1291, %v1293 : tensor<128x512xf32>
    %v1295 = stablehlo.add %v1294, %v1290 : tensor<128x512xf32>
    %v1296 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v1297 = stablehlo.multiply %v1296, %v1295 : tensor<128x512xf32>
    %v1298 = stablehlo.subtract %W9, %v1297 : tensor<128x512xf32>
    %v1299 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v1300 = stablehlo.multiply %v1299, %W9v : tensor<128x512xf32>
    %v1301 = stablehlo.add %v1300, %v1290 : tensor<128x512xf32>
    %v1302 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1303 = stablehlo.reduce(%v306 init: %v1302) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v1304 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1305 = stablehlo.multiply %v1304, %b9v : tensor<512xf32>
    %v1306 = stablehlo.add %v1305, %v1303 : tensor<512xf32>
    %v1307 = stablehlo.multiply %v1304, %v1306 : tensor<512xf32>
    %v1308 = stablehlo.add %v1307, %v1303 : tensor<512xf32>
    %v1309 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1310 = stablehlo.multiply %v1309, %v1308 : tensor<512xf32>
    %v1311 = stablehlo.subtract %b9, %v1310 : tensor<512xf32>
    %v1312 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1313 = stablehlo.multiply %v1312, %b9v : tensor<512xf32>
    %v1314 = stablehlo.add %v1313, %v1303 : tensor<512xf32>
    %v1315 = stablehlo.dot_general %v276, %v300, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<512x512xf32>
    %v1316 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1317 = stablehlo.multiply %v1316, %Wav : tensor<512x512xf32>
    %v1318 = stablehlo.add %v1317, %v1315 : tensor<512x512xf32>
    %v1319 = stablehlo.multiply %v1316, %v1318 : tensor<512x512xf32>
    %v1320 = stablehlo.add %v1319, %v1315 : tensor<512x512xf32>
    %v1321 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1322 = stablehlo.multiply %v1321, %v1320 : tensor<512x512xf32>
    %v1323 = stablehlo.subtract %Wa, %v1322 : tensor<512x512xf32>
    %v1324 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1325 = stablehlo.multiply %v1324, %Wav : tensor<512x512xf32>
    %v1326 = stablehlo.add %v1325, %v1315 : tensor<512x512xf32>
    %v1327 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1328 = stablehlo.reduce(%v300 init: %v1327) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v1329 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1330 = stablehlo.multiply %v1329, %bav : tensor<512xf32>
    %v1331 = stablehlo.add %v1330, %v1328 : tensor<512xf32>
    %v1332 = stablehlo.multiply %v1329, %v1331 : tensor<512xf32>
    %v1333 = stablehlo.add %v1332, %v1328 : tensor<512xf32>
    %v1334 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1335 = stablehlo.multiply %v1334, %v1333 : tensor<512xf32>
    %v1336 = stablehlo.subtract %ba, %v1335 : tensor<512xf32>
    %v1337 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1338 = stablehlo.multiply %v1337, %bav : tensor<512xf32>
    %v1339 = stablehlo.add %v1338, %v1328 : tensor<512xf32>
    %v1340 = stablehlo.dot_general %v281, %v294, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x10xf32>) -> tensor<512x10xf32>
    %v1341 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1342 = stablehlo.multiply %v1341, %Wbv : tensor<512x10xf32>
    %v1343 = stablehlo.add %v1342, %v1340 : tensor<512x10xf32>
    %v1344 = stablehlo.multiply %v1341, %v1343 : tensor<512x10xf32>
    %v1345 = stablehlo.add %v1344, %v1340 : tensor<512x10xf32>
    %v1346 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1347 = stablehlo.multiply %v1346, %v1345 : tensor<512x10xf32>
    %v1348 = stablehlo.subtract %Wb, %v1347 : tensor<512x10xf32>
    %v1349 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1350 = stablehlo.multiply %v1349, %Wbv : tensor<512x10xf32>
    %v1351 = stablehlo.add %v1350, %v1340 : tensor<512x10xf32>
    %v1352 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1353 = stablehlo.reduce(%v294 init: %v1352) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1354 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1355 = stablehlo.multiply %v1354, %bbv : tensor<10xf32>
    %v1356 = stablehlo.add %v1355, %v1353 : tensor<10xf32>
    %v1357 = stablehlo.multiply %v1354, %v1356 : tensor<10xf32>
    %v1358 = stablehlo.add %v1357, %v1353 : tensor<10xf32>
    %v1359 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1360 = stablehlo.multiply %v1359, %v1358 : tensor<10xf32>
    %v1361 = stablehlo.subtract %bb, %v1360 : tensor<10xf32>
    %v1362 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1363 = stablehlo.multiply %v1362, %bbv : tensor<10xf32>
    %v1364 = stablehlo.add %v1363, %v1353 : tensor<10xf32>
    return %v690, %v704, %v733, %v747, %v767, %v781, %v810, %v824, %v844, %v858, %v887, %v901, %v921, %v935, %v964, %v978, %v998, %v1012, %v1041, %v1055, %v1075, %v1089, %v1118, %v1132, %v1152, %v1166, %v1195, %v1209, %v1229, %v1243, %v1272, %v1286, %v1298, %v1311, %v1323, %v1336, %v1348, %v1361, %W1m, %cb1m, %g1m, %bt1m, %W2m, %cb2m, %g2m, %bt2m, %W3m, %cb3m, %g3m, %bt3m, %W4m, %cb4m, %g4m, %bt4m, %W5m, %cb5m, %g5m, %bt5m, %W6m, %cb6m, %g6m, %bt6m, %W7m, %cb7m, %g7m, %bt7m, %W8m, %cb8m, %g8m, %bt8m, %W9m, %b9m, %Wam, %bam, %Wbm, %bbm, %v693, %v707, %v736, %v750, %v770, %v784, %v813, %v827, %v847, %v861, %v890, %v904, %v924, %v938, %v967, %v981, %v1001, %v1015, %v1044, %v1058, %v1078, %v1092, %v1121, %v1135, %v1155, %v1169, %v1198, %v1212, %v1232, %v1246, %v1275, %v1289, %v1301, %v1314, %v1326, %v1339, %v1351, %v1364, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
