module @m {
  func.func @cifar8wb_bn_adam_train_step(%x: tensor<128x3072xf32>, %W1: tensor<16x3x3x3xf32>, %cb1: tensor<16xf32>, %g1: tensor<16xf32>, %bt1: tensor<16xf32>, %W2: tensor<16x16x3x3xf32>, %cb2: tensor<16xf32>, %g2: tensor<16xf32>, %bt2: tensor<16xf32>, %W3: tensor<16x16x3x3xf32>, %cb3: tensor<16xf32>, %g3: tensor<16xf32>, %bt3: tensor<16xf32>, %W4: tensor<16x16x3x3xf32>, %cb4: tensor<16xf32>, %g4: tensor<16xf32>, %bt4: tensor<16xf32>, %W5: tensor<32x16x3x3xf32>, %cb5: tensor<32xf32>, %g5: tensor<32xf32>, %bt5: tensor<32xf32>, %W6: tensor<32x32x3x3xf32>, %cb6: tensor<32xf32>, %g6: tensor<32xf32>, %bt6: tensor<32xf32>, %W7: tensor<32x32x3x3xf32>, %cb7: tensor<32xf32>, %g7: tensor<32xf32>, %bt7: tensor<32xf32>, %W8: tensor<32x32x3x3xf32>, %cb8: tensor<32xf32>, %g8: tensor<32xf32>, %bt8: tensor<32xf32>, %W9: tensor<128x512xf32>, %b9: tensor<512xf32>, %Wa: tensor<512x512xf32>, %ba: tensor<512xf32>, %Wb: tensor<512x10xf32>, %bb: tensor<10xf32>, %W1m: tensor<16x3x3x3xf32>, %cb1m: tensor<16xf32>, %g1m: tensor<16xf32>, %bt1m: tensor<16xf32>, %W2m: tensor<16x16x3x3xf32>, %cb2m: tensor<16xf32>, %g2m: tensor<16xf32>, %bt2m: tensor<16xf32>, %W3m: tensor<16x16x3x3xf32>, %cb3m: tensor<16xf32>, %g3m: tensor<16xf32>, %bt3m: tensor<16xf32>, %W4m: tensor<16x16x3x3xf32>, %cb4m: tensor<16xf32>, %g4m: tensor<16xf32>, %bt4m: tensor<16xf32>, %W5m: tensor<32x16x3x3xf32>, %cb5m: tensor<32xf32>, %g5m: tensor<32xf32>, %bt5m: tensor<32xf32>, %W6m: tensor<32x32x3x3xf32>, %cb6m: tensor<32xf32>, %g6m: tensor<32xf32>, %bt6m: tensor<32xf32>, %W7m: tensor<32x32x3x3xf32>, %cb7m: tensor<32xf32>, %g7m: tensor<32xf32>, %bt7m: tensor<32xf32>, %W8m: tensor<32x32x3x3xf32>, %cb8m: tensor<32xf32>, %g8m: tensor<32xf32>, %bt8m: tensor<32xf32>, %W9m: tensor<128x512xf32>, %b9m: tensor<512xf32>, %Wam: tensor<512x512xf32>, %bam: tensor<512xf32>, %Wbm: tensor<512x10xf32>, %bbm: tensor<10xf32>, %W1v: tensor<16x3x3x3xf32>, %cb1v: tensor<16xf32>, %g1v: tensor<16xf32>, %bt1v: tensor<16xf32>, %W2v: tensor<16x16x3x3xf32>, %cb2v: tensor<16xf32>, %g2v: tensor<16xf32>, %bt2v: tensor<16xf32>, %W3v: tensor<16x16x3x3xf32>, %cb3v: tensor<16xf32>, %g3v: tensor<16xf32>, %bt3v: tensor<16xf32>, %W4v: tensor<16x16x3x3xf32>, %cb4v: tensor<16xf32>, %g4v: tensor<16xf32>, %bt4v: tensor<16xf32>, %W5v: tensor<32x16x3x3xf32>, %cb5v: tensor<32xf32>, %g5v: tensor<32xf32>, %bt5v: tensor<32xf32>, %W6v: tensor<32x32x3x3xf32>, %cb6v: tensor<32xf32>, %g6v: tensor<32xf32>, %bt6v: tensor<32xf32>, %W7v: tensor<32x32x3x3xf32>, %cb7v: tensor<32xf32>, %g7v: tensor<32xf32>, %bt7v: tensor<32xf32>, %W8v: tensor<32x32x3x3xf32>, %cb8v: tensor<32xf32>, %g8v: tensor<32xf32>, %bt8v: tensor<32xf32>, %W9v: tensor<128x512xf32>, %b9v: tensor<512xf32>, %Wav: tensor<512x512xf32>, %bav: tensor<512xf32>, %Wbv: tensor<512x10xf32>, %bbv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: tensor<128x10xf32>) -> (tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>) {
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
    %v1 = stablehlo.convolution(%v0, %W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x3x32x32xf32>, tensor<16x3x3x3xf32>) -> tensor<128x16x32x32xf32>
    %v2 = stablehlo.broadcast_in_dim %cb1, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<128x16x32x32xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v6 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v8 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v9 = stablehlo.reduce(%v5 init: %v6) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v10 = stablehlo.broadcast_in_dim %v9, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v11 = stablehlo.divide %v10, %v7 : tensor<128x16x32x32xf32>
    %v12 = stablehlo.subtract %v5, %v11 : tensor<128x16x32x32xf32>
    %v13 = stablehlo.multiply %v12, %v12 : tensor<128x16x32x32xf32>
    %v14 = stablehlo.reduce(%v13 init: %v6) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v15 = stablehlo.broadcast_in_dim %v14, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v16 = stablehlo.divide %v15, %v7 : tensor<128x16x32x32xf32>
    %v17 = stablehlo.add %v16, %v8 : tensor<128x16x32x32xf32>
    %v18 = stablehlo.rsqrt %v17 : tensor<128x16x32x32xf32>
    %v19 = stablehlo.multiply %v12, %v18 : tensor<128x16x32x32xf32>
    %v20 = stablehlo.broadcast_in_dim %g1, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v21 = stablehlo.broadcast_in_dim %bt1, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v22 = stablehlo.multiply %v19, %v20 : tensor<128x16x32x32xf32>
    %v23 = stablehlo.add %v22, %v21 : tensor<128x16x32x32xf32>
    %v24 = stablehlo.reshape %v23 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v25 = stablehlo.reshape %v24 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v26 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v27 = stablehlo.maximum %v25, %v26 : tensor<128x16x32x32xf32>
    %v28 = stablehlo.reshape %v27 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v29 = stablehlo.reshape %v28 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v30 = stablehlo.convolution(%v29, %W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x32x32xf32>
    %v31 = stablehlo.broadcast_in_dim %cb2, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v32 = stablehlo.add %v30, %v31 : tensor<128x16x32x32xf32>
    %v33 = stablehlo.reshape %v32 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v34 = stablehlo.reshape %v33 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v35 = stablehlo.constant dense<0.0> : tensor<f32>
    %v36 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v37 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v38 = stablehlo.reduce(%v34 init: %v35) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v39 = stablehlo.broadcast_in_dim %v38, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v40 = stablehlo.divide %v39, %v36 : tensor<128x16x32x32xf32>
    %v41 = stablehlo.subtract %v34, %v40 : tensor<128x16x32x32xf32>
    %v42 = stablehlo.multiply %v41, %v41 : tensor<128x16x32x32xf32>
    %v43 = stablehlo.reduce(%v42 init: %v35) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v44 = stablehlo.broadcast_in_dim %v43, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v45 = stablehlo.divide %v44, %v36 : tensor<128x16x32x32xf32>
    %v46 = stablehlo.add %v45, %v37 : tensor<128x16x32x32xf32>
    %v47 = stablehlo.rsqrt %v46 : tensor<128x16x32x32xf32>
    %v48 = stablehlo.multiply %v41, %v47 : tensor<128x16x32x32xf32>
    %v49 = stablehlo.broadcast_in_dim %g2, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v50 = stablehlo.broadcast_in_dim %bt2, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v51 = stablehlo.multiply %v48, %v49 : tensor<128x16x32x32xf32>
    %v52 = stablehlo.add %v51, %v50 : tensor<128x16x32x32xf32>
    %v53 = stablehlo.reshape %v52 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v54 = stablehlo.reshape %v53 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v55 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v56 = stablehlo.maximum %v54, %v55 : tensor<128x16x32x32xf32>
    %v57 = stablehlo.reshape %v56 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v59 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v60 = "stablehlo.reduce_window"(%v58, %v59) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16x16x16xf32>
    %v61 = stablehlo.reshape %v60 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v62 = stablehlo.reshape %v61 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v63 = stablehlo.convolution(%v62, %W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v64 = stablehlo.broadcast_in_dim %cb3, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v65 = stablehlo.add %v63, %v64 : tensor<128x16x16x16xf32>
    %v66 = stablehlo.reshape %v65 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v67 = stablehlo.reshape %v66 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v68 = stablehlo.constant dense<0.0> : tensor<f32>
    %v69 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v70 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v71 = stablehlo.reduce(%v67 init: %v68) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v72 = stablehlo.broadcast_in_dim %v71, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v73 = stablehlo.divide %v72, %v69 : tensor<128x16x16x16xf32>
    %v74 = stablehlo.subtract %v67, %v73 : tensor<128x16x16x16xf32>
    %v75 = stablehlo.multiply %v74, %v74 : tensor<128x16x16x16xf32>
    %v76 = stablehlo.reduce(%v75 init: %v68) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v77 = stablehlo.broadcast_in_dim %v76, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v78 = stablehlo.divide %v77, %v69 : tensor<128x16x16x16xf32>
    %v79 = stablehlo.add %v78, %v70 : tensor<128x16x16x16xf32>
    %v80 = stablehlo.rsqrt %v79 : tensor<128x16x16x16xf32>
    %v81 = stablehlo.multiply %v74, %v80 : tensor<128x16x16x16xf32>
    %v82 = stablehlo.broadcast_in_dim %g3, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v83 = stablehlo.broadcast_in_dim %bt3, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v84 = stablehlo.multiply %v81, %v82 : tensor<128x16x16x16xf32>
    %v85 = stablehlo.add %v84, %v83 : tensor<128x16x16x16xf32>
    %v86 = stablehlo.reshape %v85 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v87 = stablehlo.reshape %v86 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v88 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v89 = stablehlo.maximum %v87, %v88 : tensor<128x16x16x16xf32>
    %v90 = stablehlo.reshape %v89 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v91 = stablehlo.reshape %v90 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v92 = stablehlo.convolution(%v91, %W4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v93 = stablehlo.broadcast_in_dim %cb4, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v94 = stablehlo.add %v92, %v93 : tensor<128x16x16x16xf32>
    %v95 = stablehlo.reshape %v94 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v96 = stablehlo.reshape %v95 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v97 = stablehlo.constant dense<0.0> : tensor<f32>
    %v98 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v99 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v100 = stablehlo.reduce(%v96 init: %v97) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v101 = stablehlo.broadcast_in_dim %v100, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v102 = stablehlo.divide %v101, %v98 : tensor<128x16x16x16xf32>
    %v103 = stablehlo.subtract %v96, %v102 : tensor<128x16x16x16xf32>
    %v104 = stablehlo.multiply %v103, %v103 : tensor<128x16x16x16xf32>
    %v105 = stablehlo.reduce(%v104 init: %v97) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v106 = stablehlo.broadcast_in_dim %v105, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v107 = stablehlo.divide %v106, %v98 : tensor<128x16x16x16xf32>
    %v108 = stablehlo.add %v107, %v99 : tensor<128x16x16x16xf32>
    %v109 = stablehlo.rsqrt %v108 : tensor<128x16x16x16xf32>
    %v110 = stablehlo.multiply %v103, %v109 : tensor<128x16x16x16xf32>
    %v111 = stablehlo.broadcast_in_dim %g4, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v112 = stablehlo.broadcast_in_dim %bt4, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v113 = stablehlo.multiply %v110, %v111 : tensor<128x16x16x16xf32>
    %v114 = stablehlo.add %v113, %v112 : tensor<128x16x16x16xf32>
    %v115 = stablehlo.reshape %v114 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v116 = stablehlo.reshape %v115 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v117 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v118 = stablehlo.maximum %v116, %v117 : tensor<128x16x16x16xf32>
    %v119 = stablehlo.reshape %v118 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v120 = stablehlo.reshape %v119 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v121 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v122 = "stablehlo.reduce_window"(%v120, %v121) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x8x8xf32>
    %v123 = stablehlo.reshape %v122 : (tensor<128x16x8x8xf32>) -> tensor<128x1024xf32>
    %v124 = stablehlo.reshape %v123 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v125 = stablehlo.convolution(%v124, %W5)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x8x8xf32>, tensor<32x16x3x3xf32>) -> tensor<128x32x8x8xf32>
    %v126 = stablehlo.broadcast_in_dim %cb5, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v127 = stablehlo.add %v125, %v126 : tensor<128x32x8x8xf32>
    %v128 = stablehlo.reshape %v127 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v129 = stablehlo.reshape %v128 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v130 = stablehlo.constant dense<0.0> : tensor<f32>
    %v131 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v132 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v133 = stablehlo.reduce(%v129 init: %v130) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v134 = stablehlo.broadcast_in_dim %v133, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v135 = stablehlo.divide %v134, %v131 : tensor<128x32x8x8xf32>
    %v136 = stablehlo.subtract %v129, %v135 : tensor<128x32x8x8xf32>
    %v137 = stablehlo.multiply %v136, %v136 : tensor<128x32x8x8xf32>
    %v138 = stablehlo.reduce(%v137 init: %v130) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v139 = stablehlo.broadcast_in_dim %v138, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v140 = stablehlo.divide %v139, %v131 : tensor<128x32x8x8xf32>
    %v141 = stablehlo.add %v140, %v132 : tensor<128x32x8x8xf32>
    %v142 = stablehlo.rsqrt %v141 : tensor<128x32x8x8xf32>
    %v143 = stablehlo.multiply %v136, %v142 : tensor<128x32x8x8xf32>
    %v144 = stablehlo.broadcast_in_dim %g5, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v145 = stablehlo.broadcast_in_dim %bt5, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v146 = stablehlo.multiply %v143, %v144 : tensor<128x32x8x8xf32>
    %v147 = stablehlo.add %v146, %v145 : tensor<128x32x8x8xf32>
    %v148 = stablehlo.reshape %v147 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v149 = stablehlo.reshape %v148 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v150 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v151 = stablehlo.maximum %v149, %v150 : tensor<128x32x8x8xf32>
    %v152 = stablehlo.reshape %v151 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v153 = stablehlo.reshape %v152 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v154 = stablehlo.convolution(%v153, %W6)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x8x8xf32>
    %v155 = stablehlo.broadcast_in_dim %cb6, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v156 = stablehlo.add %v154, %v155 : tensor<128x32x8x8xf32>
    %v157 = stablehlo.reshape %v156 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v158 = stablehlo.reshape %v157 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v159 = stablehlo.constant dense<0.0> : tensor<f32>
    %v160 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v161 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v162 = stablehlo.reduce(%v158 init: %v159) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v163 = stablehlo.broadcast_in_dim %v162, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v164 = stablehlo.divide %v163, %v160 : tensor<128x32x8x8xf32>
    %v165 = stablehlo.subtract %v158, %v164 : tensor<128x32x8x8xf32>
    %v166 = stablehlo.multiply %v165, %v165 : tensor<128x32x8x8xf32>
    %v167 = stablehlo.reduce(%v166 init: %v159) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v168 = stablehlo.broadcast_in_dim %v167, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v169 = stablehlo.divide %v168, %v160 : tensor<128x32x8x8xf32>
    %v170 = stablehlo.add %v169, %v161 : tensor<128x32x8x8xf32>
    %v171 = stablehlo.rsqrt %v170 : tensor<128x32x8x8xf32>
    %v172 = stablehlo.multiply %v165, %v171 : tensor<128x32x8x8xf32>
    %v173 = stablehlo.broadcast_in_dim %g6, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v174 = stablehlo.broadcast_in_dim %bt6, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v175 = stablehlo.multiply %v172, %v173 : tensor<128x32x8x8xf32>
    %v176 = stablehlo.add %v175, %v174 : tensor<128x32x8x8xf32>
    %v177 = stablehlo.reshape %v176 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v178 = stablehlo.reshape %v177 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v179 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v180 = stablehlo.maximum %v178, %v179 : tensor<128x32x8x8xf32>
    %v181 = stablehlo.reshape %v180 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v182 = stablehlo.reshape %v181 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v183 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v184 = "stablehlo.reduce_window"(%v182, %v183) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32x4x4xf32>
    %v185 = stablehlo.reshape %v184 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v186 = stablehlo.reshape %v185 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v187 = stablehlo.convolution(%v186, %W7)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v188 = stablehlo.broadcast_in_dim %cb7, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v189 = stablehlo.add %v187, %v188 : tensor<128x32x4x4xf32>
    %v190 = stablehlo.reshape %v189 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v191 = stablehlo.reshape %v190 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v192 = stablehlo.constant dense<0.0> : tensor<f32>
    %v193 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v194 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v195 = stablehlo.reduce(%v191 init: %v192) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v196 = stablehlo.broadcast_in_dim %v195, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v197 = stablehlo.divide %v196, %v193 : tensor<128x32x4x4xf32>
    %v198 = stablehlo.subtract %v191, %v197 : tensor<128x32x4x4xf32>
    %v199 = stablehlo.multiply %v198, %v198 : tensor<128x32x4x4xf32>
    %v200 = stablehlo.reduce(%v199 init: %v192) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v201 = stablehlo.broadcast_in_dim %v200, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v202 = stablehlo.divide %v201, %v193 : tensor<128x32x4x4xf32>
    %v203 = stablehlo.add %v202, %v194 : tensor<128x32x4x4xf32>
    %v204 = stablehlo.rsqrt %v203 : tensor<128x32x4x4xf32>
    %v205 = stablehlo.multiply %v198, %v204 : tensor<128x32x4x4xf32>
    %v206 = stablehlo.broadcast_in_dim %g7, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v207 = stablehlo.broadcast_in_dim %bt7, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v208 = stablehlo.multiply %v205, %v206 : tensor<128x32x4x4xf32>
    %v209 = stablehlo.add %v208, %v207 : tensor<128x32x4x4xf32>
    %v210 = stablehlo.reshape %v209 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v211 = stablehlo.reshape %v210 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v212 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v213 = stablehlo.maximum %v211, %v212 : tensor<128x32x4x4xf32>
    %v214 = stablehlo.reshape %v213 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v215 = stablehlo.reshape %v214 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v216 = stablehlo.convolution(%v215, %W8)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v217 = stablehlo.broadcast_in_dim %cb8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v218 = stablehlo.add %v216, %v217 : tensor<128x32x4x4xf32>
    %v219 = stablehlo.reshape %v218 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v220 = stablehlo.reshape %v219 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v221 = stablehlo.constant dense<0.0> : tensor<f32>
    %v222 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v223 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v224 = stablehlo.reduce(%v220 init: %v221) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v225 = stablehlo.broadcast_in_dim %v224, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v226 = stablehlo.divide %v225, %v222 : tensor<128x32x4x4xf32>
    %v227 = stablehlo.subtract %v220, %v226 : tensor<128x32x4x4xf32>
    %v228 = stablehlo.multiply %v227, %v227 : tensor<128x32x4x4xf32>
    %v229 = stablehlo.reduce(%v228 init: %v221) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v230 = stablehlo.broadcast_in_dim %v229, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v231 = stablehlo.divide %v230, %v222 : tensor<128x32x4x4xf32>
    %v232 = stablehlo.add %v231, %v223 : tensor<128x32x4x4xf32>
    %v233 = stablehlo.rsqrt %v232 : tensor<128x32x4x4xf32>
    %v234 = stablehlo.multiply %v227, %v233 : tensor<128x32x4x4xf32>
    %v235 = stablehlo.broadcast_in_dim %g8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v236 = stablehlo.broadcast_in_dim %bt8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v237 = stablehlo.multiply %v234, %v235 : tensor<128x32x4x4xf32>
    %v238 = stablehlo.add %v237, %v236 : tensor<128x32x4x4xf32>
    %v239 = stablehlo.reshape %v238 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v240 = stablehlo.reshape %v239 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v241 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v242 = stablehlo.maximum %v240, %v241 : tensor<128x32x4x4xf32>
    %v243 = stablehlo.reshape %v242 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v244 = stablehlo.reshape %v243 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v245 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v246 = "stablehlo.reduce_window"(%v244, %v245) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32x2x2xf32>
    %v247 = stablehlo.reshape %v246 : (tensor<128x32x2x2xf32>) -> tensor<128x128xf32>
    %v248 = stablehlo.dot_general %v247, %W9, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x512xf32>) -> tensor<128x512xf32>
    %v249 = stablehlo.broadcast_in_dim %b9, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v250 = stablehlo.add %v248, %v249 : tensor<128x512xf32>
    %v251 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v252 = stablehlo.maximum %v250, %v251 : tensor<128x512xf32>
    %v253 = stablehlo.dot_general %v252, %Wa, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %v254 = stablehlo.broadcast_in_dim %ba, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v255 = stablehlo.add %v253, %v254 : tensor<128x512xf32>
    %v256 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v257 = stablehlo.maximum %v255, %v256 : tensor<128x512xf32>
    %v258 = stablehlo.dot_general %v257, %Wb, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x10xf32>) -> tensor<128x10xf32>
    %v259 = stablehlo.broadcast_in_dim %bb, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v260 = stablehlo.add %v258, %v259 : tensor<128x10xf32>
    %v261 = stablehlo.reshape %v260 : (tensor<128x10xf32>) -> tensor<128x1x10xf32>
    %v262 = stablehlo.constant dense<0.0> : tensor<f32>
    %v263 = stablehlo.exponential %v261 : tensor<128x1x10xf32>
    %v264 = stablehlo.reduce(%v263 init: %v262) applies stablehlo.add across dimensions = [2] : (tensor<128x1x10xf32>, tensor<f32>) -> tensor<128x1xf32>
    %v265 = stablehlo.broadcast_in_dim %v264, dims = [0, 1] : (tensor<128x1xf32>) -> tensor<128x1x10xf32>
    %v266 = stablehlo.divide %v263, %v265 : tensor<128x1x10xf32>
    %v267 = stablehlo.reshape %v266 : (tensor<128x1x10xf32>) -> tensor<128x10xf32>
    %v268 = stablehlo.subtract %v267, %onehot : tensor<128x10xf32>
    %v269 = stablehlo.constant dense<0.0078125> : tensor<128x10xf32>
    %v270 = stablehlo.multiply %v268, %v269 : tensor<128x10xf32>
    // ── report-only scalar loss (NOT pretty(AST): the kit has no rank-0 loss op; it
    //    feeds no parameter, only the driver's progress line) ──
    %llog = stablehlo.log %v267 : tensor<128x10xf32>
    %ohll = stablehlo.multiply %onehot, %llog : tensor<128x10xf32>
    %csum = stablehlo.reduce(%ohll init: %lzero) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<f32>
    %cneg = stablehlo.negate %csum : tensor<f32>
    %lbf = stablehlo.constant dense<128.0> : tensor<f32>
    %loss = stablehlo.divide %cneg, %lbf : tensor<f32>
    %v271 = stablehlo.reshape %v270 : (tensor<128x10xf32>) -> tensor<128x1x10xf32>
    %v272 = stablehlo.dot_general %v271, %Wb, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x10xf32>, tensor<512x10xf32>) -> tensor<128x1x512xf32>
    %v273 = stablehlo.reshape %v272 : (tensor<128x1x512xf32>) -> tensor<128x512xf32>
    %v274 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v275 = stablehlo.compare GT, %v255, %v274 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v276 = stablehlo.select %v275, %v273, %v274 : tensor<128x512xi1>, tensor<128x512xf32>
    %v277 = stablehlo.reshape %v276 : (tensor<128x512xf32>) -> tensor<128x1x512xf32>
    %v278 = stablehlo.dot_general %v277, %Wa, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x512xf32>, tensor<512x512xf32>) -> tensor<128x1x512xf32>
    %v279 = stablehlo.reshape %v278 : (tensor<128x1x512xf32>) -> tensor<128x512xf32>
    %v280 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v281 = stablehlo.compare GT, %v250, %v280 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v282 = stablehlo.select %v281, %v279, %v280 : tensor<128x512xi1>, tensor<128x512xf32>
    %v283 = stablehlo.reshape %v282 : (tensor<128x512xf32>) -> tensor<128x1x512xf32>
    %v284 = stablehlo.dot_general %v283, %W9, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x1x512xf32>, tensor<128x512xf32>) -> tensor<128x1x128xf32>
    %v285 = stablehlo.reshape %v284 : (tensor<128x1x128xf32>) -> tensor<128x128xf32>
    %v286 = stablehlo.reshape %v243 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v287 = stablehlo.reshape %v285 : (tensor<128x128xf32>) -> tensor<128x32x2x2xf32>
    %v288 = stablehlo.constant dense<0.0> : tensor<f32>
    %v289 = "stablehlo.select_and_scatter"(%v286, %v287, %v288) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x4x4xf32>, tensor<128x32x2x2xf32>, tensor<f32>) -> tensor<128x32x4x4xf32>
    %v290 = stablehlo.reshape %v289 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v291 = stablehlo.reshape %v290 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v292 = stablehlo.reshape %v239 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v293 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v294 = stablehlo.compare GT, %v292, %v293 : (tensor<128x32x4x4xf32>, tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xi1>
    %v295 = stablehlo.select %v294, %v291, %v293 : tensor<128x32x4x4xi1>, tensor<128x32x4x4xf32>
    %v296 = stablehlo.reshape %v295 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v297 = stablehlo.reshape %v296 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v298 = stablehlo.reshape %v219 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v299 = stablehlo.constant dense<0.0> : tensor<f32>
    %v300 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v301 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v302 = stablehlo.reduce(%v298 init: %v299) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v303 = stablehlo.broadcast_in_dim %v302, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v304 = stablehlo.divide %v303, %v300 : tensor<128x32x4x4xf32>
    %v305 = stablehlo.subtract %v298, %v304 : tensor<128x32x4x4xf32>
    %v306 = stablehlo.multiply %v305, %v305 : tensor<128x32x4x4xf32>
    %v307 = stablehlo.reduce(%v306 init: %v299) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v308 = stablehlo.broadcast_in_dim %v307, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v309 = stablehlo.divide %v308, %v300 : tensor<128x32x4x4xf32>
    %v310 = stablehlo.add %v309, %v301 : tensor<128x32x4x4xf32>
    %v311 = stablehlo.rsqrt %v310 : tensor<128x32x4x4xf32>
    %v312 = stablehlo.multiply %v305, %v311 : tensor<128x32x4x4xf32>
    %v313 = stablehlo.broadcast_in_dim %g8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v314 = stablehlo.multiply %v313, %v297 : tensor<128x32x4x4xf32>
    %v315 = stablehlo.reduce(%v314 init: %v299) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v316 = stablehlo.broadcast_in_dim %v315, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v317 = stablehlo.multiply %v312, %v314 : tensor<128x32x4x4xf32>
    %v318 = stablehlo.reduce(%v317 init: %v299) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v319 = stablehlo.broadcast_in_dim %v318, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v320 = stablehlo.multiply %v314, %v300 : tensor<128x32x4x4xf32>
    %v321 = stablehlo.subtract %v320, %v316 : tensor<128x32x4x4xf32>
    %v322 = stablehlo.multiply %v312, %v319 : tensor<128x32x4x4xf32>
    %v323 = stablehlo.subtract %v321, %v322 : tensor<128x32x4x4xf32>
    %v324 = stablehlo.divide %v311, %v300 : tensor<128x32x4x4xf32>
    %v325 = stablehlo.multiply %v324, %v323 : tensor<128x32x4x4xf32>
    %v326 = stablehlo.reshape %v325 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v327 = stablehlo.reshape %v326 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v328 = stablehlo.reverse %W8, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v329 = stablehlo.transpose %v328, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v330 = stablehlo.convolution(%v327, %v329)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v331 = stablehlo.reshape %v330 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v332 = stablehlo.reshape %v331 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v333 = stablehlo.reshape %v210 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v334 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v335 = stablehlo.compare GT, %v333, %v334 : (tensor<128x32x4x4xf32>, tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xi1>
    %v336 = stablehlo.select %v335, %v332, %v334 : tensor<128x32x4x4xi1>, tensor<128x32x4x4xf32>
    %v337 = stablehlo.reshape %v336 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v338 = stablehlo.reshape %v337 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v339 = stablehlo.reshape %v190 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v340 = stablehlo.constant dense<0.0> : tensor<f32>
    %v341 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v342 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v343 = stablehlo.reduce(%v339 init: %v340) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v344 = stablehlo.broadcast_in_dim %v343, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v345 = stablehlo.divide %v344, %v341 : tensor<128x32x4x4xf32>
    %v346 = stablehlo.subtract %v339, %v345 : tensor<128x32x4x4xf32>
    %v347 = stablehlo.multiply %v346, %v346 : tensor<128x32x4x4xf32>
    %v348 = stablehlo.reduce(%v347 init: %v340) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v349 = stablehlo.broadcast_in_dim %v348, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v350 = stablehlo.divide %v349, %v341 : tensor<128x32x4x4xf32>
    %v351 = stablehlo.add %v350, %v342 : tensor<128x32x4x4xf32>
    %v352 = stablehlo.rsqrt %v351 : tensor<128x32x4x4xf32>
    %v353 = stablehlo.multiply %v346, %v352 : tensor<128x32x4x4xf32>
    %v354 = stablehlo.broadcast_in_dim %g7, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v355 = stablehlo.multiply %v354, %v338 : tensor<128x32x4x4xf32>
    %v356 = stablehlo.reduce(%v355 init: %v340) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v357 = stablehlo.broadcast_in_dim %v356, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v358 = stablehlo.multiply %v353, %v355 : tensor<128x32x4x4xf32>
    %v359 = stablehlo.reduce(%v358 init: %v340) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v360 = stablehlo.broadcast_in_dim %v359, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v361 = stablehlo.multiply %v355, %v341 : tensor<128x32x4x4xf32>
    %v362 = stablehlo.subtract %v361, %v357 : tensor<128x32x4x4xf32>
    %v363 = stablehlo.multiply %v353, %v360 : tensor<128x32x4x4xf32>
    %v364 = stablehlo.subtract %v362, %v363 : tensor<128x32x4x4xf32>
    %v365 = stablehlo.divide %v352, %v341 : tensor<128x32x4x4xf32>
    %v366 = stablehlo.multiply %v365, %v364 : tensor<128x32x4x4xf32>
    %v367 = stablehlo.reshape %v366 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v368 = stablehlo.reshape %v367 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v369 = stablehlo.reverse %W7, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v370 = stablehlo.transpose %v369, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v371 = stablehlo.convolution(%v368, %v370)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v372 = stablehlo.reshape %v371 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v373 = stablehlo.reshape %v181 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v374 = stablehlo.reshape %v372 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v375 = stablehlo.constant dense<0.0> : tensor<f32>
    %v376 = "stablehlo.select_and_scatter"(%v373, %v374, %v375) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x8x8xf32>, tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32x8x8xf32>
    %v377 = stablehlo.reshape %v376 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v378 = stablehlo.reshape %v377 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v379 = stablehlo.reshape %v177 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v380 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v381 = stablehlo.compare GT, %v379, %v380 : (tensor<128x32x8x8xf32>, tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xi1>
    %v382 = stablehlo.select %v381, %v378, %v380 : tensor<128x32x8x8xi1>, tensor<128x32x8x8xf32>
    %v383 = stablehlo.reshape %v382 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v384 = stablehlo.reshape %v383 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v385 = stablehlo.reshape %v157 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v386 = stablehlo.constant dense<0.0> : tensor<f32>
    %v387 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v388 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v389 = stablehlo.reduce(%v385 init: %v386) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v390 = stablehlo.broadcast_in_dim %v389, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v391 = stablehlo.divide %v390, %v387 : tensor<128x32x8x8xf32>
    %v392 = stablehlo.subtract %v385, %v391 : tensor<128x32x8x8xf32>
    %v393 = stablehlo.multiply %v392, %v392 : tensor<128x32x8x8xf32>
    %v394 = stablehlo.reduce(%v393 init: %v386) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v395 = stablehlo.broadcast_in_dim %v394, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v396 = stablehlo.divide %v395, %v387 : tensor<128x32x8x8xf32>
    %v397 = stablehlo.add %v396, %v388 : tensor<128x32x8x8xf32>
    %v398 = stablehlo.rsqrt %v397 : tensor<128x32x8x8xf32>
    %v399 = stablehlo.multiply %v392, %v398 : tensor<128x32x8x8xf32>
    %v400 = stablehlo.broadcast_in_dim %g6, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v401 = stablehlo.multiply %v400, %v384 : tensor<128x32x8x8xf32>
    %v402 = stablehlo.reduce(%v401 init: %v386) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v403 = stablehlo.broadcast_in_dim %v402, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v404 = stablehlo.multiply %v399, %v401 : tensor<128x32x8x8xf32>
    %v405 = stablehlo.reduce(%v404 init: %v386) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v406 = stablehlo.broadcast_in_dim %v405, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v407 = stablehlo.multiply %v401, %v387 : tensor<128x32x8x8xf32>
    %v408 = stablehlo.subtract %v407, %v403 : tensor<128x32x8x8xf32>
    %v409 = stablehlo.multiply %v399, %v406 : tensor<128x32x8x8xf32>
    %v410 = stablehlo.subtract %v408, %v409 : tensor<128x32x8x8xf32>
    %v411 = stablehlo.divide %v398, %v387 : tensor<128x32x8x8xf32>
    %v412 = stablehlo.multiply %v411, %v410 : tensor<128x32x8x8xf32>
    %v413 = stablehlo.reshape %v412 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v414 = stablehlo.reshape %v413 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v415 = stablehlo.reverse %W6, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v416 = stablehlo.transpose %v415, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v417 = stablehlo.convolution(%v414, %v416)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x8x8xf32>
    %v418 = stablehlo.reshape %v417 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v419 = stablehlo.reshape %v418 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v420 = stablehlo.reshape %v148 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v421 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v422 = stablehlo.compare GT, %v420, %v421 : (tensor<128x32x8x8xf32>, tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xi1>
    %v423 = stablehlo.select %v422, %v419, %v421 : tensor<128x32x8x8xi1>, tensor<128x32x8x8xf32>
    %v424 = stablehlo.reshape %v423 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v425 = stablehlo.reshape %v424 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v426 = stablehlo.reshape %v128 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v427 = stablehlo.constant dense<0.0> : tensor<f32>
    %v428 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v429 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v430 = stablehlo.reduce(%v426 init: %v427) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v431 = stablehlo.broadcast_in_dim %v430, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v432 = stablehlo.divide %v431, %v428 : tensor<128x32x8x8xf32>
    %v433 = stablehlo.subtract %v426, %v432 : tensor<128x32x8x8xf32>
    %v434 = stablehlo.multiply %v433, %v433 : tensor<128x32x8x8xf32>
    %v435 = stablehlo.reduce(%v434 init: %v427) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v436 = stablehlo.broadcast_in_dim %v435, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v437 = stablehlo.divide %v436, %v428 : tensor<128x32x8x8xf32>
    %v438 = stablehlo.add %v437, %v429 : tensor<128x32x8x8xf32>
    %v439 = stablehlo.rsqrt %v438 : tensor<128x32x8x8xf32>
    %v440 = stablehlo.multiply %v433, %v439 : tensor<128x32x8x8xf32>
    %v441 = stablehlo.broadcast_in_dim %g5, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v442 = stablehlo.multiply %v441, %v425 : tensor<128x32x8x8xf32>
    %v443 = stablehlo.reduce(%v442 init: %v427) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v444 = stablehlo.broadcast_in_dim %v443, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v445 = stablehlo.multiply %v440, %v442 : tensor<128x32x8x8xf32>
    %v446 = stablehlo.reduce(%v445 init: %v427) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v447 = stablehlo.broadcast_in_dim %v446, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v448 = stablehlo.multiply %v442, %v428 : tensor<128x32x8x8xf32>
    %v449 = stablehlo.subtract %v448, %v444 : tensor<128x32x8x8xf32>
    %v450 = stablehlo.multiply %v440, %v447 : tensor<128x32x8x8xf32>
    %v451 = stablehlo.subtract %v449, %v450 : tensor<128x32x8x8xf32>
    %v452 = stablehlo.divide %v439, %v428 : tensor<128x32x8x8xf32>
    %v453 = stablehlo.multiply %v452, %v451 : tensor<128x32x8x8xf32>
    %v454 = stablehlo.reshape %v453 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v455 = stablehlo.reshape %v454 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v456 = stablehlo.reverse %W5, dims = [2, 3] : tensor<32x16x3x3xf32>
    %v457 = stablehlo.transpose %v456, dims = [1, 0, 2, 3] : (tensor<32x16x3x3xf32>) -> tensor<16x32x3x3xf32>
    %v458 = stablehlo.convolution(%v455, %v457)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<16x32x3x3xf32>) -> tensor<128x16x8x8xf32>
    %v459 = stablehlo.reshape %v458 : (tensor<128x16x8x8xf32>) -> tensor<128x1024xf32>
    %v460 = stablehlo.reshape %v119 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v461 = stablehlo.reshape %v459 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v462 = stablehlo.constant dense<0.0> : tensor<f32>
    %v463 = "stablehlo.select_and_scatter"(%v460, %v461, %v462) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x16x16xf32>, tensor<128x16x8x8xf32>, tensor<f32>) -> tensor<128x16x16x16xf32>
    %v464 = stablehlo.reshape %v463 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v465 = stablehlo.reshape %v464 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v466 = stablehlo.reshape %v115 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v467 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v468 = stablehlo.compare GT, %v466, %v467 : (tensor<128x16x16x16xf32>, tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xi1>
    %v469 = stablehlo.select %v468, %v465, %v467 : tensor<128x16x16x16xi1>, tensor<128x16x16x16xf32>
    %v470 = stablehlo.reshape %v469 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v471 = stablehlo.reshape %v470 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v472 = stablehlo.reshape %v95 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v473 = stablehlo.constant dense<0.0> : tensor<f32>
    %v474 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v475 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v476 = stablehlo.reduce(%v472 init: %v473) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v477 = stablehlo.broadcast_in_dim %v476, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v478 = stablehlo.divide %v477, %v474 : tensor<128x16x16x16xf32>
    %v479 = stablehlo.subtract %v472, %v478 : tensor<128x16x16x16xf32>
    %v480 = stablehlo.multiply %v479, %v479 : tensor<128x16x16x16xf32>
    %v481 = stablehlo.reduce(%v480 init: %v473) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v482 = stablehlo.broadcast_in_dim %v481, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v483 = stablehlo.divide %v482, %v474 : tensor<128x16x16x16xf32>
    %v484 = stablehlo.add %v483, %v475 : tensor<128x16x16x16xf32>
    %v485 = stablehlo.rsqrt %v484 : tensor<128x16x16x16xf32>
    %v486 = stablehlo.multiply %v479, %v485 : tensor<128x16x16x16xf32>
    %v487 = stablehlo.broadcast_in_dim %g4, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v488 = stablehlo.multiply %v487, %v471 : tensor<128x16x16x16xf32>
    %v489 = stablehlo.reduce(%v488 init: %v473) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v490 = stablehlo.broadcast_in_dim %v489, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v491 = stablehlo.multiply %v486, %v488 : tensor<128x16x16x16xf32>
    %v492 = stablehlo.reduce(%v491 init: %v473) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v493 = stablehlo.broadcast_in_dim %v492, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v494 = stablehlo.multiply %v488, %v474 : tensor<128x16x16x16xf32>
    %v495 = stablehlo.subtract %v494, %v490 : tensor<128x16x16x16xf32>
    %v496 = stablehlo.multiply %v486, %v493 : tensor<128x16x16x16xf32>
    %v497 = stablehlo.subtract %v495, %v496 : tensor<128x16x16x16xf32>
    %v498 = stablehlo.divide %v485, %v474 : tensor<128x16x16x16xf32>
    %v499 = stablehlo.multiply %v498, %v497 : tensor<128x16x16x16xf32>
    %v500 = stablehlo.reshape %v499 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v501 = stablehlo.reshape %v500 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v502 = stablehlo.reverse %W4, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v503 = stablehlo.transpose %v502, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v504 = stablehlo.convolution(%v501, %v503)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v505 = stablehlo.reshape %v504 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v506 = stablehlo.reshape %v505 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v507 = stablehlo.reshape %v86 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v508 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v509 = stablehlo.compare GT, %v507, %v508 : (tensor<128x16x16x16xf32>, tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xi1>
    %v510 = stablehlo.select %v509, %v506, %v508 : tensor<128x16x16x16xi1>, tensor<128x16x16x16xf32>
    %v511 = stablehlo.reshape %v510 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v512 = stablehlo.reshape %v511 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v513 = stablehlo.reshape %v66 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v514 = stablehlo.constant dense<0.0> : tensor<f32>
    %v515 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v516 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v517 = stablehlo.reduce(%v513 init: %v514) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v518 = stablehlo.broadcast_in_dim %v517, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v519 = stablehlo.divide %v518, %v515 : tensor<128x16x16x16xf32>
    %v520 = stablehlo.subtract %v513, %v519 : tensor<128x16x16x16xf32>
    %v521 = stablehlo.multiply %v520, %v520 : tensor<128x16x16x16xf32>
    %v522 = stablehlo.reduce(%v521 init: %v514) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v523 = stablehlo.broadcast_in_dim %v522, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v524 = stablehlo.divide %v523, %v515 : tensor<128x16x16x16xf32>
    %v525 = stablehlo.add %v524, %v516 : tensor<128x16x16x16xf32>
    %v526 = stablehlo.rsqrt %v525 : tensor<128x16x16x16xf32>
    %v527 = stablehlo.multiply %v520, %v526 : tensor<128x16x16x16xf32>
    %v528 = stablehlo.broadcast_in_dim %g3, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v529 = stablehlo.multiply %v528, %v512 : tensor<128x16x16x16xf32>
    %v530 = stablehlo.reduce(%v529 init: %v514) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v531 = stablehlo.broadcast_in_dim %v530, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v532 = stablehlo.multiply %v527, %v529 : tensor<128x16x16x16xf32>
    %v533 = stablehlo.reduce(%v532 init: %v514) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v534 = stablehlo.broadcast_in_dim %v533, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v535 = stablehlo.multiply %v529, %v515 : tensor<128x16x16x16xf32>
    %v536 = stablehlo.subtract %v535, %v531 : tensor<128x16x16x16xf32>
    %v537 = stablehlo.multiply %v527, %v534 : tensor<128x16x16x16xf32>
    %v538 = stablehlo.subtract %v536, %v537 : tensor<128x16x16x16xf32>
    %v539 = stablehlo.divide %v526, %v515 : tensor<128x16x16x16xf32>
    %v540 = stablehlo.multiply %v539, %v538 : tensor<128x16x16x16xf32>
    %v541 = stablehlo.reshape %v540 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v542 = stablehlo.reshape %v541 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v543 = stablehlo.reverse %W3, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v544 = stablehlo.transpose %v543, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v545 = stablehlo.convolution(%v542, %v544)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v546 = stablehlo.reshape %v545 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v547 = stablehlo.reshape %v57 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v548 = stablehlo.reshape %v546 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v549 = stablehlo.constant dense<0.0> : tensor<f32>
    %v550 = "stablehlo.select_and_scatter"(%v547, %v548, %v549) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x32x32xf32>, tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x32x32xf32>
    %v551 = stablehlo.reshape %v550 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v552 = stablehlo.reshape %v551 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v553 = stablehlo.reshape %v53 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v554 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v555 = stablehlo.compare GT, %v553, %v554 : (tensor<128x16x32x32xf32>, tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xi1>
    %v556 = stablehlo.select %v555, %v552, %v554 : tensor<128x16x32x32xi1>, tensor<128x16x32x32xf32>
    %v557 = stablehlo.reshape %v556 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v558 = stablehlo.reshape %v557 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v559 = stablehlo.reshape %v33 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v560 = stablehlo.constant dense<0.0> : tensor<f32>
    %v561 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v562 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v563 = stablehlo.reduce(%v559 init: %v560) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v564 = stablehlo.broadcast_in_dim %v563, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v565 = stablehlo.divide %v564, %v561 : tensor<128x16x32x32xf32>
    %v566 = stablehlo.subtract %v559, %v565 : tensor<128x16x32x32xf32>
    %v567 = stablehlo.multiply %v566, %v566 : tensor<128x16x32x32xf32>
    %v568 = stablehlo.reduce(%v567 init: %v560) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v569 = stablehlo.broadcast_in_dim %v568, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v570 = stablehlo.divide %v569, %v561 : tensor<128x16x32x32xf32>
    %v571 = stablehlo.add %v570, %v562 : tensor<128x16x32x32xf32>
    %v572 = stablehlo.rsqrt %v571 : tensor<128x16x32x32xf32>
    %v573 = stablehlo.multiply %v566, %v572 : tensor<128x16x32x32xf32>
    %v574 = stablehlo.broadcast_in_dim %g2, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v575 = stablehlo.multiply %v574, %v558 : tensor<128x16x32x32xf32>
    %v576 = stablehlo.reduce(%v575 init: %v560) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v577 = stablehlo.broadcast_in_dim %v576, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v578 = stablehlo.multiply %v573, %v575 : tensor<128x16x32x32xf32>
    %v579 = stablehlo.reduce(%v578 init: %v560) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v580 = stablehlo.broadcast_in_dim %v579, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v581 = stablehlo.multiply %v575, %v561 : tensor<128x16x32x32xf32>
    %v582 = stablehlo.subtract %v581, %v577 : tensor<128x16x32x32xf32>
    %v583 = stablehlo.multiply %v573, %v580 : tensor<128x16x32x32xf32>
    %v584 = stablehlo.subtract %v582, %v583 : tensor<128x16x32x32xf32>
    %v585 = stablehlo.divide %v572, %v561 : tensor<128x16x32x32xf32>
    %v586 = stablehlo.multiply %v585, %v584 : tensor<128x16x32x32xf32>
    %v587 = stablehlo.reshape %v586 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v588 = stablehlo.reshape %v587 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v589 = stablehlo.reverse %W2, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v590 = stablehlo.transpose %v589, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v591 = stablehlo.convolution(%v588, %v590)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x32x32xf32>
    %v592 = stablehlo.reshape %v591 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v593 = stablehlo.reshape %v592 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v594 = stablehlo.reshape %v24 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v595 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v596 = stablehlo.compare GT, %v594, %v595 : (tensor<128x16x32x32xf32>, tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xi1>
    %v597 = stablehlo.select %v596, %v593, %v595 : tensor<128x16x32x32xi1>, tensor<128x16x32x32xf32>
    %v598 = stablehlo.reshape %v597 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v599 = stablehlo.reshape %v598 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v600 = stablehlo.reshape %v4 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v601 = stablehlo.constant dense<0.0> : tensor<f32>
    %v602 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v603 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v604 = stablehlo.reduce(%v600 init: %v601) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v605 = stablehlo.broadcast_in_dim %v604, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v606 = stablehlo.divide %v605, %v602 : tensor<128x16x32x32xf32>
    %v607 = stablehlo.subtract %v600, %v606 : tensor<128x16x32x32xf32>
    %v608 = stablehlo.multiply %v607, %v607 : tensor<128x16x32x32xf32>
    %v609 = stablehlo.reduce(%v608 init: %v601) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v610 = stablehlo.broadcast_in_dim %v609, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v611 = stablehlo.divide %v610, %v602 : tensor<128x16x32x32xf32>
    %v612 = stablehlo.add %v611, %v603 : tensor<128x16x32x32xf32>
    %v613 = stablehlo.rsqrt %v612 : tensor<128x16x32x32xf32>
    %v614 = stablehlo.multiply %v607, %v613 : tensor<128x16x32x32xf32>
    %v615 = stablehlo.broadcast_in_dim %g1, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v616 = stablehlo.multiply %v615, %v599 : tensor<128x16x32x32xf32>
    %v617 = stablehlo.reduce(%v616 init: %v601) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v618 = stablehlo.broadcast_in_dim %v617, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v619 = stablehlo.multiply %v614, %v616 : tensor<128x16x32x32xf32>
    %v620 = stablehlo.reduce(%v619 init: %v601) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v621 = stablehlo.broadcast_in_dim %v620, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v622 = stablehlo.multiply %v616, %v602 : tensor<128x16x32x32xf32>
    %v623 = stablehlo.subtract %v622, %v618 : tensor<128x16x32x32xf32>
    %v624 = stablehlo.multiply %v614, %v621 : tensor<128x16x32x32xf32>
    %v625 = stablehlo.subtract %v623, %v624 : tensor<128x16x32x32xf32>
    %v626 = stablehlo.divide %v613, %v602 : tensor<128x16x32x32xf32>
    %v627 = stablehlo.multiply %v626, %v625 : tensor<128x16x32x32xf32>
    %v628 = stablehlo.reshape %v627 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v629 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v630 = stablehlo.reshape %v628 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v631 = stablehlo.transpose %v629, dims = [1, 0, 2, 3] : (tensor<128x3x32x32xf32>) -> tensor<3x128x32x32xf32>
    %v632 = stablehlo.transpose %v630, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v633 = stablehlo.convolution(%v631, %v632)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<3x16x3x3xf32>
    %v634 = stablehlo.transpose %v633, dims = [1, 0, 2, 3] : (tensor<3x16x3x3xf32>) -> tensor<16x3x3x3xf32>
    %v635 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v636 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v637 = stablehlo.multiply %v635, %W1m : tensor<16x3x3x3xf32>
    %v638 = stablehlo.multiply %v636, %v634 : tensor<16x3x3x3xf32>
    %v639 = stablehlo.add %v637, %v638 : tensor<16x3x3x3xf32>
    %v640 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v641 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v642 = stablehlo.multiply %v640, %W1v : tensor<16x3x3x3xf32>
    %v643 = stablehlo.multiply %v634, %v634 : tensor<16x3x3x3xf32>
    %v644 = stablehlo.multiply %v641, %v643 : tensor<16x3x3x3xf32>
    %v645 = stablehlo.add %v642, %v644 : tensor<16x3x3x3xf32>
    %v646 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v647 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v648 = stablehlo.divide %v639, %v646 : tensor<16x3x3x3xf32>
    %v649 = stablehlo.divide %v645, %v647 : tensor<16x3x3x3xf32>
    %v650 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v651 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v652 = stablehlo.sqrt %v649 : tensor<16x3x3x3xf32>
    %v653 = stablehlo.add %v652, %v651 : tensor<16x3x3x3xf32>
    %v654 = stablehlo.divide %v648, %v653 : tensor<16x3x3x3xf32>
    %v655 = stablehlo.multiply %v650, %v654 : tensor<16x3x3x3xf32>
    %v656 = stablehlo.subtract %W1, %v655 : tensor<16x3x3x3xf32>
    %v657 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v658 = stablehlo.multiply %v657, %v650 : tensor<16x3x3x3xf32>
    %v659 = stablehlo.multiply %v658, %W1 : tensor<16x3x3x3xf32>
    %v660 = stablehlo.subtract %v656, %v659 : tensor<16x3x3x3xf32>
    %v661 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v662 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v663 = stablehlo.multiply %v661, %W1m : tensor<16x3x3x3xf32>
    %v664 = stablehlo.multiply %v662, %v634 : tensor<16x3x3x3xf32>
    %v665 = stablehlo.add %v663, %v664 : tensor<16x3x3x3xf32>
    %v666 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v667 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v668 = stablehlo.multiply %v666, %W1v : tensor<16x3x3x3xf32>
    %v669 = stablehlo.multiply %v634, %v634 : tensor<16x3x3x3xf32>
    %v670 = stablehlo.multiply %v667, %v669 : tensor<16x3x3x3xf32>
    %v671 = stablehlo.add %v668, %v670 : tensor<16x3x3x3xf32>
    %v672 = stablehlo.reshape %v628 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v673 = stablehlo.constant dense<0.0> : tensor<f32>
    %v674 = stablehlo.reduce(%v672 init: %v673) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v675 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v676 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v677 = stablehlo.multiply %v675, %cb1m : tensor<16xf32>
    %v678 = stablehlo.multiply %v676, %v674 : tensor<16xf32>
    %v679 = stablehlo.add %v677, %v678 : tensor<16xf32>
    %v680 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v681 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v682 = stablehlo.multiply %v680, %cb1v : tensor<16xf32>
    %v683 = stablehlo.multiply %v674, %v674 : tensor<16xf32>
    %v684 = stablehlo.multiply %v681, %v683 : tensor<16xf32>
    %v685 = stablehlo.add %v682, %v684 : tensor<16xf32>
    %v686 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v687 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v688 = stablehlo.divide %v679, %v686 : tensor<16xf32>
    %v689 = stablehlo.divide %v685, %v687 : tensor<16xf32>
    %v690 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v691 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v692 = stablehlo.sqrt %v689 : tensor<16xf32>
    %v693 = stablehlo.add %v692, %v691 : tensor<16xf32>
    %v694 = stablehlo.divide %v688, %v693 : tensor<16xf32>
    %v695 = stablehlo.multiply %v690, %v694 : tensor<16xf32>
    %v696 = stablehlo.subtract %cb1, %v695 : tensor<16xf32>
    %v697 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v698 = stablehlo.multiply %v697, %v690 : tensor<16xf32>
    %v699 = stablehlo.multiply %v698, %cb1 : tensor<16xf32>
    %v700 = stablehlo.subtract %v696, %v699 : tensor<16xf32>
    %v701 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v702 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v703 = stablehlo.multiply %v701, %cb1m : tensor<16xf32>
    %v704 = stablehlo.multiply %v702, %v674 : tensor<16xf32>
    %v705 = stablehlo.add %v703, %v704 : tensor<16xf32>
    %v706 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v707 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v708 = stablehlo.multiply %v706, %cb1v : tensor<16xf32>
    %v709 = stablehlo.multiply %v674, %v674 : tensor<16xf32>
    %v710 = stablehlo.multiply %v707, %v709 : tensor<16xf32>
    %v711 = stablehlo.add %v708, %v710 : tensor<16xf32>
    %v712 = stablehlo.constant dense<0.0> : tensor<f32>
    %v713 = stablehlo.reshape %v4 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v714 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v715 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v716 = stablehlo.reduce(%v713 init: %v712) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v717 = stablehlo.broadcast_in_dim %v716, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v718 = stablehlo.divide %v717, %v714 : tensor<128x16x32x32xf32>
    %v719 = stablehlo.subtract %v713, %v718 : tensor<128x16x32x32xf32>
    %v720 = stablehlo.multiply %v719, %v719 : tensor<128x16x32x32xf32>
    %v721 = stablehlo.reduce(%v720 init: %v712) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v722 = stablehlo.broadcast_in_dim %v721, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v723 = stablehlo.divide %v722, %v714 : tensor<128x16x32x32xf32>
    %v724 = stablehlo.add %v723, %v715 : tensor<128x16x32x32xf32>
    %v725 = stablehlo.rsqrt %v724 : tensor<128x16x32x32xf32>
    %v726 = stablehlo.multiply %v719, %v725 : tensor<128x16x32x32xf32>
    %v727 = stablehlo.reshape %v598 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v728 = stablehlo.multiply %v727, %v726 : tensor<128x16x32x32xf32>
    %v729 = stablehlo.reduce(%v728 init: %v712) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v730 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v731 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v732 = stablehlo.multiply %v730, %g1m : tensor<16xf32>
    %v733 = stablehlo.multiply %v731, %v729 : tensor<16xf32>
    %v734 = stablehlo.add %v732, %v733 : tensor<16xf32>
    %v735 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v736 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v737 = stablehlo.multiply %v735, %g1v : tensor<16xf32>
    %v738 = stablehlo.multiply %v729, %v729 : tensor<16xf32>
    %v739 = stablehlo.multiply %v736, %v738 : tensor<16xf32>
    %v740 = stablehlo.add %v737, %v739 : tensor<16xf32>
    %v741 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v742 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v743 = stablehlo.divide %v734, %v741 : tensor<16xf32>
    %v744 = stablehlo.divide %v740, %v742 : tensor<16xf32>
    %v745 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v746 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v747 = stablehlo.sqrt %v744 : tensor<16xf32>
    %v748 = stablehlo.add %v747, %v746 : tensor<16xf32>
    %v749 = stablehlo.divide %v743, %v748 : tensor<16xf32>
    %v750 = stablehlo.multiply %v745, %v749 : tensor<16xf32>
    %v751 = stablehlo.subtract %g1, %v750 : tensor<16xf32>
    %v752 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v753 = stablehlo.multiply %v752, %v745 : tensor<16xf32>
    %v754 = stablehlo.multiply %v753, %g1 : tensor<16xf32>
    %v755 = stablehlo.subtract %v751, %v754 : tensor<16xf32>
    %v756 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v757 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v758 = stablehlo.multiply %v756, %g1m : tensor<16xf32>
    %v759 = stablehlo.multiply %v757, %v729 : tensor<16xf32>
    %v760 = stablehlo.add %v758, %v759 : tensor<16xf32>
    %v761 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v762 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v763 = stablehlo.multiply %v761, %g1v : tensor<16xf32>
    %v764 = stablehlo.multiply %v729, %v729 : tensor<16xf32>
    %v765 = stablehlo.multiply %v762, %v764 : tensor<16xf32>
    %v766 = stablehlo.add %v763, %v765 : tensor<16xf32>
    %v767 = stablehlo.constant dense<0.0> : tensor<f32>
    %v768 = stablehlo.reshape %v598 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v769 = stablehlo.reduce(%v768 init: %v767) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v770 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v771 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v772 = stablehlo.multiply %v770, %bt1m : tensor<16xf32>
    %v773 = stablehlo.multiply %v771, %v769 : tensor<16xf32>
    %v774 = stablehlo.add %v772, %v773 : tensor<16xf32>
    %v775 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v776 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v777 = stablehlo.multiply %v775, %bt1v : tensor<16xf32>
    %v778 = stablehlo.multiply %v769, %v769 : tensor<16xf32>
    %v779 = stablehlo.multiply %v776, %v778 : tensor<16xf32>
    %v780 = stablehlo.add %v777, %v779 : tensor<16xf32>
    %v781 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v782 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v783 = stablehlo.divide %v774, %v781 : tensor<16xf32>
    %v784 = stablehlo.divide %v780, %v782 : tensor<16xf32>
    %v785 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v786 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v787 = stablehlo.sqrt %v784 : tensor<16xf32>
    %v788 = stablehlo.add %v787, %v786 : tensor<16xf32>
    %v789 = stablehlo.divide %v783, %v788 : tensor<16xf32>
    %v790 = stablehlo.multiply %v785, %v789 : tensor<16xf32>
    %v791 = stablehlo.subtract %bt1, %v790 : tensor<16xf32>
    %v792 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v793 = stablehlo.multiply %v792, %v785 : tensor<16xf32>
    %v794 = stablehlo.multiply %v793, %bt1 : tensor<16xf32>
    %v795 = stablehlo.subtract %v791, %v794 : tensor<16xf32>
    %v796 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v797 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v798 = stablehlo.multiply %v796, %bt1m : tensor<16xf32>
    %v799 = stablehlo.multiply %v797, %v769 : tensor<16xf32>
    %v800 = stablehlo.add %v798, %v799 : tensor<16xf32>
    %v801 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v802 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v803 = stablehlo.multiply %v801, %bt1v : tensor<16xf32>
    %v804 = stablehlo.multiply %v769, %v769 : tensor<16xf32>
    %v805 = stablehlo.multiply %v802, %v804 : tensor<16xf32>
    %v806 = stablehlo.add %v803, %v805 : tensor<16xf32>
    %v807 = stablehlo.reshape %v28 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v808 = stablehlo.reshape %v587 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v809 = stablehlo.transpose %v807, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v810 = stablehlo.transpose %v808, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v811 = stablehlo.convolution(%v809, %v810)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x16x3x3xf32>
    %v812 = stablehlo.transpose %v811, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v813 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v814 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v815 = stablehlo.multiply %v813, %W2m : tensor<16x16x3x3xf32>
    %v816 = stablehlo.multiply %v814, %v812 : tensor<16x16x3x3xf32>
    %v817 = stablehlo.add %v815, %v816 : tensor<16x16x3x3xf32>
    %v818 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v819 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v820 = stablehlo.multiply %v818, %W2v : tensor<16x16x3x3xf32>
    %v821 = stablehlo.multiply %v812, %v812 : tensor<16x16x3x3xf32>
    %v822 = stablehlo.multiply %v819, %v821 : tensor<16x16x3x3xf32>
    %v823 = stablehlo.add %v820, %v822 : tensor<16x16x3x3xf32>
    %v824 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v825 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v826 = stablehlo.divide %v817, %v824 : tensor<16x16x3x3xf32>
    %v827 = stablehlo.divide %v823, %v825 : tensor<16x16x3x3xf32>
    %v828 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v829 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v830 = stablehlo.sqrt %v827 : tensor<16x16x3x3xf32>
    %v831 = stablehlo.add %v830, %v829 : tensor<16x16x3x3xf32>
    %v832 = stablehlo.divide %v826, %v831 : tensor<16x16x3x3xf32>
    %v833 = stablehlo.multiply %v828, %v832 : tensor<16x16x3x3xf32>
    %v834 = stablehlo.subtract %W2, %v833 : tensor<16x16x3x3xf32>
    %v835 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v836 = stablehlo.multiply %v835, %v828 : tensor<16x16x3x3xf32>
    %v837 = stablehlo.multiply %v836, %W2 : tensor<16x16x3x3xf32>
    %v838 = stablehlo.subtract %v834, %v837 : tensor<16x16x3x3xf32>
    %v839 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v840 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v841 = stablehlo.multiply %v839, %W2m : tensor<16x16x3x3xf32>
    %v842 = stablehlo.multiply %v840, %v812 : tensor<16x16x3x3xf32>
    %v843 = stablehlo.add %v841, %v842 : tensor<16x16x3x3xf32>
    %v844 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v845 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v846 = stablehlo.multiply %v844, %W2v : tensor<16x16x3x3xf32>
    %v847 = stablehlo.multiply %v812, %v812 : tensor<16x16x3x3xf32>
    %v848 = stablehlo.multiply %v845, %v847 : tensor<16x16x3x3xf32>
    %v849 = stablehlo.add %v846, %v848 : tensor<16x16x3x3xf32>
    %v850 = stablehlo.reshape %v587 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v851 = stablehlo.constant dense<0.0> : tensor<f32>
    %v852 = stablehlo.reduce(%v850 init: %v851) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v853 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v854 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v855 = stablehlo.multiply %v853, %cb2m : tensor<16xf32>
    %v856 = stablehlo.multiply %v854, %v852 : tensor<16xf32>
    %v857 = stablehlo.add %v855, %v856 : tensor<16xf32>
    %v858 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v859 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v860 = stablehlo.multiply %v858, %cb2v : tensor<16xf32>
    %v861 = stablehlo.multiply %v852, %v852 : tensor<16xf32>
    %v862 = stablehlo.multiply %v859, %v861 : tensor<16xf32>
    %v863 = stablehlo.add %v860, %v862 : tensor<16xf32>
    %v864 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v865 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v866 = stablehlo.divide %v857, %v864 : tensor<16xf32>
    %v867 = stablehlo.divide %v863, %v865 : tensor<16xf32>
    %v868 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v869 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v870 = stablehlo.sqrt %v867 : tensor<16xf32>
    %v871 = stablehlo.add %v870, %v869 : tensor<16xf32>
    %v872 = stablehlo.divide %v866, %v871 : tensor<16xf32>
    %v873 = stablehlo.multiply %v868, %v872 : tensor<16xf32>
    %v874 = stablehlo.subtract %cb2, %v873 : tensor<16xf32>
    %v875 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v876 = stablehlo.multiply %v875, %v868 : tensor<16xf32>
    %v877 = stablehlo.multiply %v876, %cb2 : tensor<16xf32>
    %v878 = stablehlo.subtract %v874, %v877 : tensor<16xf32>
    %v879 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v880 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v881 = stablehlo.multiply %v879, %cb2m : tensor<16xf32>
    %v882 = stablehlo.multiply %v880, %v852 : tensor<16xf32>
    %v883 = stablehlo.add %v881, %v882 : tensor<16xf32>
    %v884 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v885 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v886 = stablehlo.multiply %v884, %cb2v : tensor<16xf32>
    %v887 = stablehlo.multiply %v852, %v852 : tensor<16xf32>
    %v888 = stablehlo.multiply %v885, %v887 : tensor<16xf32>
    %v889 = stablehlo.add %v886, %v888 : tensor<16xf32>
    %v890 = stablehlo.constant dense<0.0> : tensor<f32>
    %v891 = stablehlo.reshape %v33 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v892 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v893 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v894 = stablehlo.reduce(%v891 init: %v890) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v895 = stablehlo.broadcast_in_dim %v894, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v896 = stablehlo.divide %v895, %v892 : tensor<128x16x32x32xf32>
    %v897 = stablehlo.subtract %v891, %v896 : tensor<128x16x32x32xf32>
    %v898 = stablehlo.multiply %v897, %v897 : tensor<128x16x32x32xf32>
    %v899 = stablehlo.reduce(%v898 init: %v890) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v900 = stablehlo.broadcast_in_dim %v899, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v901 = stablehlo.divide %v900, %v892 : tensor<128x16x32x32xf32>
    %v902 = stablehlo.add %v901, %v893 : tensor<128x16x32x32xf32>
    %v903 = stablehlo.rsqrt %v902 : tensor<128x16x32x32xf32>
    %v904 = stablehlo.multiply %v897, %v903 : tensor<128x16x32x32xf32>
    %v905 = stablehlo.reshape %v557 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v906 = stablehlo.multiply %v905, %v904 : tensor<128x16x32x32xf32>
    %v907 = stablehlo.reduce(%v906 init: %v890) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v908 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v909 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v910 = stablehlo.multiply %v908, %g2m : tensor<16xf32>
    %v911 = stablehlo.multiply %v909, %v907 : tensor<16xf32>
    %v912 = stablehlo.add %v910, %v911 : tensor<16xf32>
    %v913 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v914 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v915 = stablehlo.multiply %v913, %g2v : tensor<16xf32>
    %v916 = stablehlo.multiply %v907, %v907 : tensor<16xf32>
    %v917 = stablehlo.multiply %v914, %v916 : tensor<16xf32>
    %v918 = stablehlo.add %v915, %v917 : tensor<16xf32>
    %v919 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v920 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v921 = stablehlo.divide %v912, %v919 : tensor<16xf32>
    %v922 = stablehlo.divide %v918, %v920 : tensor<16xf32>
    %v923 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v924 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v925 = stablehlo.sqrt %v922 : tensor<16xf32>
    %v926 = stablehlo.add %v925, %v924 : tensor<16xf32>
    %v927 = stablehlo.divide %v921, %v926 : tensor<16xf32>
    %v928 = stablehlo.multiply %v923, %v927 : tensor<16xf32>
    %v929 = stablehlo.subtract %g2, %v928 : tensor<16xf32>
    %v930 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v931 = stablehlo.multiply %v930, %v923 : tensor<16xf32>
    %v932 = stablehlo.multiply %v931, %g2 : tensor<16xf32>
    %v933 = stablehlo.subtract %v929, %v932 : tensor<16xf32>
    %v934 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v935 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v936 = stablehlo.multiply %v934, %g2m : tensor<16xf32>
    %v937 = stablehlo.multiply %v935, %v907 : tensor<16xf32>
    %v938 = stablehlo.add %v936, %v937 : tensor<16xf32>
    %v939 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v940 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v941 = stablehlo.multiply %v939, %g2v : tensor<16xf32>
    %v942 = stablehlo.multiply %v907, %v907 : tensor<16xf32>
    %v943 = stablehlo.multiply %v940, %v942 : tensor<16xf32>
    %v944 = stablehlo.add %v941, %v943 : tensor<16xf32>
    %v945 = stablehlo.constant dense<0.0> : tensor<f32>
    %v946 = stablehlo.reshape %v557 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v947 = stablehlo.reduce(%v946 init: %v945) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v948 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v949 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v950 = stablehlo.multiply %v948, %bt2m : tensor<16xf32>
    %v951 = stablehlo.multiply %v949, %v947 : tensor<16xf32>
    %v952 = stablehlo.add %v950, %v951 : tensor<16xf32>
    %v953 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v954 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v955 = stablehlo.multiply %v953, %bt2v : tensor<16xf32>
    %v956 = stablehlo.multiply %v947, %v947 : tensor<16xf32>
    %v957 = stablehlo.multiply %v954, %v956 : tensor<16xf32>
    %v958 = stablehlo.add %v955, %v957 : tensor<16xf32>
    %v959 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v960 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v961 = stablehlo.divide %v952, %v959 : tensor<16xf32>
    %v962 = stablehlo.divide %v958, %v960 : tensor<16xf32>
    %v963 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v964 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v965 = stablehlo.sqrt %v962 : tensor<16xf32>
    %v966 = stablehlo.add %v965, %v964 : tensor<16xf32>
    %v967 = stablehlo.divide %v961, %v966 : tensor<16xf32>
    %v968 = stablehlo.multiply %v963, %v967 : tensor<16xf32>
    %v969 = stablehlo.subtract %bt2, %v968 : tensor<16xf32>
    %v970 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v971 = stablehlo.multiply %v970, %v963 : tensor<16xf32>
    %v972 = stablehlo.multiply %v971, %bt2 : tensor<16xf32>
    %v973 = stablehlo.subtract %v969, %v972 : tensor<16xf32>
    %v974 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v975 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v976 = stablehlo.multiply %v974, %bt2m : tensor<16xf32>
    %v977 = stablehlo.multiply %v975, %v947 : tensor<16xf32>
    %v978 = stablehlo.add %v976, %v977 : tensor<16xf32>
    %v979 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v980 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v981 = stablehlo.multiply %v979, %bt2v : tensor<16xf32>
    %v982 = stablehlo.multiply %v947, %v947 : tensor<16xf32>
    %v983 = stablehlo.multiply %v980, %v982 : tensor<16xf32>
    %v984 = stablehlo.add %v981, %v983 : tensor<16xf32>
    %v985 = stablehlo.reshape %v61 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v986 = stablehlo.reshape %v541 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v987 = stablehlo.transpose %v985, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v988 = stablehlo.transpose %v986, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v989 = stablehlo.convolution(%v987, %v988)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v990 = stablehlo.transpose %v989, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v991 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v992 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v993 = stablehlo.multiply %v991, %W3m : tensor<16x16x3x3xf32>
    %v994 = stablehlo.multiply %v992, %v990 : tensor<16x16x3x3xf32>
    %v995 = stablehlo.add %v993, %v994 : tensor<16x16x3x3xf32>
    %v996 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v997 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v998 = stablehlo.multiply %v996, %W3v : tensor<16x16x3x3xf32>
    %v999 = stablehlo.multiply %v990, %v990 : tensor<16x16x3x3xf32>
    %v1000 = stablehlo.multiply %v997, %v999 : tensor<16x16x3x3xf32>
    %v1001 = stablehlo.add %v998, %v1000 : tensor<16x16x3x3xf32>
    %v1002 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1003 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1004 = stablehlo.divide %v995, %v1002 : tensor<16x16x3x3xf32>
    %v1005 = stablehlo.divide %v1001, %v1003 : tensor<16x16x3x3xf32>
    %v1006 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1007 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1008 = stablehlo.sqrt %v1005 : tensor<16x16x3x3xf32>
    %v1009 = stablehlo.add %v1008, %v1007 : tensor<16x16x3x3xf32>
    %v1010 = stablehlo.divide %v1004, %v1009 : tensor<16x16x3x3xf32>
    %v1011 = stablehlo.multiply %v1006, %v1010 : tensor<16x16x3x3xf32>
    %v1012 = stablehlo.subtract %W3, %v1011 : tensor<16x16x3x3xf32>
    %v1013 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1014 = stablehlo.multiply %v1013, %v1006 : tensor<16x16x3x3xf32>
    %v1015 = stablehlo.multiply %v1014, %W3 : tensor<16x16x3x3xf32>
    %v1016 = stablehlo.subtract %v1012, %v1015 : tensor<16x16x3x3xf32>
    %v1017 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1018 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1019 = stablehlo.multiply %v1017, %W3m : tensor<16x16x3x3xf32>
    %v1020 = stablehlo.multiply %v1018, %v990 : tensor<16x16x3x3xf32>
    %v1021 = stablehlo.add %v1019, %v1020 : tensor<16x16x3x3xf32>
    %v1022 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1023 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1024 = stablehlo.multiply %v1022, %W3v : tensor<16x16x3x3xf32>
    %v1025 = stablehlo.multiply %v990, %v990 : tensor<16x16x3x3xf32>
    %v1026 = stablehlo.multiply %v1023, %v1025 : tensor<16x16x3x3xf32>
    %v1027 = stablehlo.add %v1024, %v1026 : tensor<16x16x3x3xf32>
    %v1028 = stablehlo.reshape %v541 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1029 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1030 = stablehlo.reduce(%v1028 init: %v1029) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v1031 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1032 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1033 = stablehlo.multiply %v1031, %cb3m : tensor<16xf32>
    %v1034 = stablehlo.multiply %v1032, %v1030 : tensor<16xf32>
    %v1035 = stablehlo.add %v1033, %v1034 : tensor<16xf32>
    %v1036 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1037 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1038 = stablehlo.multiply %v1036, %cb3v : tensor<16xf32>
    %v1039 = stablehlo.multiply %v1030, %v1030 : tensor<16xf32>
    %v1040 = stablehlo.multiply %v1037, %v1039 : tensor<16xf32>
    %v1041 = stablehlo.add %v1038, %v1040 : tensor<16xf32>
    %v1042 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1043 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1044 = stablehlo.divide %v1035, %v1042 : tensor<16xf32>
    %v1045 = stablehlo.divide %v1041, %v1043 : tensor<16xf32>
    %v1046 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1047 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1048 = stablehlo.sqrt %v1045 : tensor<16xf32>
    %v1049 = stablehlo.add %v1048, %v1047 : tensor<16xf32>
    %v1050 = stablehlo.divide %v1044, %v1049 : tensor<16xf32>
    %v1051 = stablehlo.multiply %v1046, %v1050 : tensor<16xf32>
    %v1052 = stablehlo.subtract %cb3, %v1051 : tensor<16xf32>
    %v1053 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1054 = stablehlo.multiply %v1053, %v1046 : tensor<16xf32>
    %v1055 = stablehlo.multiply %v1054, %cb3 : tensor<16xf32>
    %v1056 = stablehlo.subtract %v1052, %v1055 : tensor<16xf32>
    %v1057 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1058 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1059 = stablehlo.multiply %v1057, %cb3m : tensor<16xf32>
    %v1060 = stablehlo.multiply %v1058, %v1030 : tensor<16xf32>
    %v1061 = stablehlo.add %v1059, %v1060 : tensor<16xf32>
    %v1062 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1063 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1064 = stablehlo.multiply %v1062, %cb3v : tensor<16xf32>
    %v1065 = stablehlo.multiply %v1030, %v1030 : tensor<16xf32>
    %v1066 = stablehlo.multiply %v1063, %v1065 : tensor<16xf32>
    %v1067 = stablehlo.add %v1064, %v1066 : tensor<16xf32>
    %v1068 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1069 = stablehlo.reshape %v66 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1070 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v1071 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v1072 = stablehlo.reduce(%v1069 init: %v1068) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v1073 = stablehlo.broadcast_in_dim %v1072, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v1074 = stablehlo.divide %v1073, %v1070 : tensor<128x16x16x16xf32>
    %v1075 = stablehlo.subtract %v1069, %v1074 : tensor<128x16x16x16xf32>
    %v1076 = stablehlo.multiply %v1075, %v1075 : tensor<128x16x16x16xf32>
    %v1077 = stablehlo.reduce(%v1076 init: %v1068) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v1078 = stablehlo.broadcast_in_dim %v1077, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v1079 = stablehlo.divide %v1078, %v1070 : tensor<128x16x16x16xf32>
    %v1080 = stablehlo.add %v1079, %v1071 : tensor<128x16x16x16xf32>
    %v1081 = stablehlo.rsqrt %v1080 : tensor<128x16x16x16xf32>
    %v1082 = stablehlo.multiply %v1075, %v1081 : tensor<128x16x16x16xf32>
    %v1083 = stablehlo.reshape %v511 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1084 = stablehlo.multiply %v1083, %v1082 : tensor<128x16x16x16xf32>
    %v1085 = stablehlo.reduce(%v1084 init: %v1068) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v1086 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1087 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1088 = stablehlo.multiply %v1086, %g3m : tensor<16xf32>
    %v1089 = stablehlo.multiply %v1087, %v1085 : tensor<16xf32>
    %v1090 = stablehlo.add %v1088, %v1089 : tensor<16xf32>
    %v1091 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1092 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1093 = stablehlo.multiply %v1091, %g3v : tensor<16xf32>
    %v1094 = stablehlo.multiply %v1085, %v1085 : tensor<16xf32>
    %v1095 = stablehlo.multiply %v1092, %v1094 : tensor<16xf32>
    %v1096 = stablehlo.add %v1093, %v1095 : tensor<16xf32>
    %v1097 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1098 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1099 = stablehlo.divide %v1090, %v1097 : tensor<16xf32>
    %v1100 = stablehlo.divide %v1096, %v1098 : tensor<16xf32>
    %v1101 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1102 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1103 = stablehlo.sqrt %v1100 : tensor<16xf32>
    %v1104 = stablehlo.add %v1103, %v1102 : tensor<16xf32>
    %v1105 = stablehlo.divide %v1099, %v1104 : tensor<16xf32>
    %v1106 = stablehlo.multiply %v1101, %v1105 : tensor<16xf32>
    %v1107 = stablehlo.subtract %g3, %v1106 : tensor<16xf32>
    %v1108 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1109 = stablehlo.multiply %v1108, %v1101 : tensor<16xf32>
    %v1110 = stablehlo.multiply %v1109, %g3 : tensor<16xf32>
    %v1111 = stablehlo.subtract %v1107, %v1110 : tensor<16xf32>
    %v1112 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1113 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1114 = stablehlo.multiply %v1112, %g3m : tensor<16xf32>
    %v1115 = stablehlo.multiply %v1113, %v1085 : tensor<16xf32>
    %v1116 = stablehlo.add %v1114, %v1115 : tensor<16xf32>
    %v1117 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1118 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1119 = stablehlo.multiply %v1117, %g3v : tensor<16xf32>
    %v1120 = stablehlo.multiply %v1085, %v1085 : tensor<16xf32>
    %v1121 = stablehlo.multiply %v1118, %v1120 : tensor<16xf32>
    %v1122 = stablehlo.add %v1119, %v1121 : tensor<16xf32>
    %v1123 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1124 = stablehlo.reshape %v511 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1125 = stablehlo.reduce(%v1124 init: %v1123) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v1126 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1127 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1128 = stablehlo.multiply %v1126, %bt3m : tensor<16xf32>
    %v1129 = stablehlo.multiply %v1127, %v1125 : tensor<16xf32>
    %v1130 = stablehlo.add %v1128, %v1129 : tensor<16xf32>
    %v1131 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1132 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1133 = stablehlo.multiply %v1131, %bt3v : tensor<16xf32>
    %v1134 = stablehlo.multiply %v1125, %v1125 : tensor<16xf32>
    %v1135 = stablehlo.multiply %v1132, %v1134 : tensor<16xf32>
    %v1136 = stablehlo.add %v1133, %v1135 : tensor<16xf32>
    %v1137 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1138 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1139 = stablehlo.divide %v1130, %v1137 : tensor<16xf32>
    %v1140 = stablehlo.divide %v1136, %v1138 : tensor<16xf32>
    %v1141 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1142 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1143 = stablehlo.sqrt %v1140 : tensor<16xf32>
    %v1144 = stablehlo.add %v1143, %v1142 : tensor<16xf32>
    %v1145 = stablehlo.divide %v1139, %v1144 : tensor<16xf32>
    %v1146 = stablehlo.multiply %v1141, %v1145 : tensor<16xf32>
    %v1147 = stablehlo.subtract %bt3, %v1146 : tensor<16xf32>
    %v1148 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1149 = stablehlo.multiply %v1148, %v1141 : tensor<16xf32>
    %v1150 = stablehlo.multiply %v1149, %bt3 : tensor<16xf32>
    %v1151 = stablehlo.subtract %v1147, %v1150 : tensor<16xf32>
    %v1152 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1153 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1154 = stablehlo.multiply %v1152, %bt3m : tensor<16xf32>
    %v1155 = stablehlo.multiply %v1153, %v1125 : tensor<16xf32>
    %v1156 = stablehlo.add %v1154, %v1155 : tensor<16xf32>
    %v1157 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1158 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1159 = stablehlo.multiply %v1157, %bt3v : tensor<16xf32>
    %v1160 = stablehlo.multiply %v1125, %v1125 : tensor<16xf32>
    %v1161 = stablehlo.multiply %v1158, %v1160 : tensor<16xf32>
    %v1162 = stablehlo.add %v1159, %v1161 : tensor<16xf32>
    %v1163 = stablehlo.reshape %v90 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1164 = stablehlo.reshape %v500 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1165 = stablehlo.transpose %v1163, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v1166 = stablehlo.transpose %v1164, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v1167 = stablehlo.convolution(%v1165, %v1166)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v1168 = stablehlo.transpose %v1167, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v1169 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1170 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1171 = stablehlo.multiply %v1169, %W4m : tensor<16x16x3x3xf32>
    %v1172 = stablehlo.multiply %v1170, %v1168 : tensor<16x16x3x3xf32>
    %v1173 = stablehlo.add %v1171, %v1172 : tensor<16x16x3x3xf32>
    %v1174 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1175 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1176 = stablehlo.multiply %v1174, %W4v : tensor<16x16x3x3xf32>
    %v1177 = stablehlo.multiply %v1168, %v1168 : tensor<16x16x3x3xf32>
    %v1178 = stablehlo.multiply %v1175, %v1177 : tensor<16x16x3x3xf32>
    %v1179 = stablehlo.add %v1176, %v1178 : tensor<16x16x3x3xf32>
    %v1180 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1181 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1182 = stablehlo.divide %v1173, %v1180 : tensor<16x16x3x3xf32>
    %v1183 = stablehlo.divide %v1179, %v1181 : tensor<16x16x3x3xf32>
    %v1184 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1185 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1186 = stablehlo.sqrt %v1183 : tensor<16x16x3x3xf32>
    %v1187 = stablehlo.add %v1186, %v1185 : tensor<16x16x3x3xf32>
    %v1188 = stablehlo.divide %v1182, %v1187 : tensor<16x16x3x3xf32>
    %v1189 = stablehlo.multiply %v1184, %v1188 : tensor<16x16x3x3xf32>
    %v1190 = stablehlo.subtract %W4, %v1189 : tensor<16x16x3x3xf32>
    %v1191 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1192 = stablehlo.multiply %v1191, %v1184 : tensor<16x16x3x3xf32>
    %v1193 = stablehlo.multiply %v1192, %W4 : tensor<16x16x3x3xf32>
    %v1194 = stablehlo.subtract %v1190, %v1193 : tensor<16x16x3x3xf32>
    %v1195 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1196 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1197 = stablehlo.multiply %v1195, %W4m : tensor<16x16x3x3xf32>
    %v1198 = stablehlo.multiply %v1196, %v1168 : tensor<16x16x3x3xf32>
    %v1199 = stablehlo.add %v1197, %v1198 : tensor<16x16x3x3xf32>
    %v1200 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1201 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1202 = stablehlo.multiply %v1200, %W4v : tensor<16x16x3x3xf32>
    %v1203 = stablehlo.multiply %v1168, %v1168 : tensor<16x16x3x3xf32>
    %v1204 = stablehlo.multiply %v1201, %v1203 : tensor<16x16x3x3xf32>
    %v1205 = stablehlo.add %v1202, %v1204 : tensor<16x16x3x3xf32>
    %v1206 = stablehlo.reshape %v500 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1207 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1208 = stablehlo.reduce(%v1206 init: %v1207) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v1209 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1210 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1211 = stablehlo.multiply %v1209, %cb4m : tensor<16xf32>
    %v1212 = stablehlo.multiply %v1210, %v1208 : tensor<16xf32>
    %v1213 = stablehlo.add %v1211, %v1212 : tensor<16xf32>
    %v1214 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1215 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1216 = stablehlo.multiply %v1214, %cb4v : tensor<16xf32>
    %v1217 = stablehlo.multiply %v1208, %v1208 : tensor<16xf32>
    %v1218 = stablehlo.multiply %v1215, %v1217 : tensor<16xf32>
    %v1219 = stablehlo.add %v1216, %v1218 : tensor<16xf32>
    %v1220 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1221 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1222 = stablehlo.divide %v1213, %v1220 : tensor<16xf32>
    %v1223 = stablehlo.divide %v1219, %v1221 : tensor<16xf32>
    %v1224 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1225 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1226 = stablehlo.sqrt %v1223 : tensor<16xf32>
    %v1227 = stablehlo.add %v1226, %v1225 : tensor<16xf32>
    %v1228 = stablehlo.divide %v1222, %v1227 : tensor<16xf32>
    %v1229 = stablehlo.multiply %v1224, %v1228 : tensor<16xf32>
    %v1230 = stablehlo.subtract %cb4, %v1229 : tensor<16xf32>
    %v1231 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1232 = stablehlo.multiply %v1231, %v1224 : tensor<16xf32>
    %v1233 = stablehlo.multiply %v1232, %cb4 : tensor<16xf32>
    %v1234 = stablehlo.subtract %v1230, %v1233 : tensor<16xf32>
    %v1235 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1236 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1237 = stablehlo.multiply %v1235, %cb4m : tensor<16xf32>
    %v1238 = stablehlo.multiply %v1236, %v1208 : tensor<16xf32>
    %v1239 = stablehlo.add %v1237, %v1238 : tensor<16xf32>
    %v1240 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1241 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1242 = stablehlo.multiply %v1240, %cb4v : tensor<16xf32>
    %v1243 = stablehlo.multiply %v1208, %v1208 : tensor<16xf32>
    %v1244 = stablehlo.multiply %v1241, %v1243 : tensor<16xf32>
    %v1245 = stablehlo.add %v1242, %v1244 : tensor<16xf32>
    %v1246 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1247 = stablehlo.reshape %v95 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1248 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v1249 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v1250 = stablehlo.reduce(%v1247 init: %v1246) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v1251 = stablehlo.broadcast_in_dim %v1250, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v1252 = stablehlo.divide %v1251, %v1248 : tensor<128x16x16x16xf32>
    %v1253 = stablehlo.subtract %v1247, %v1252 : tensor<128x16x16x16xf32>
    %v1254 = stablehlo.multiply %v1253, %v1253 : tensor<128x16x16x16xf32>
    %v1255 = stablehlo.reduce(%v1254 init: %v1246) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v1256 = stablehlo.broadcast_in_dim %v1255, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v1257 = stablehlo.divide %v1256, %v1248 : tensor<128x16x16x16xf32>
    %v1258 = stablehlo.add %v1257, %v1249 : tensor<128x16x16x16xf32>
    %v1259 = stablehlo.rsqrt %v1258 : tensor<128x16x16x16xf32>
    %v1260 = stablehlo.multiply %v1253, %v1259 : tensor<128x16x16x16xf32>
    %v1261 = stablehlo.reshape %v470 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1262 = stablehlo.multiply %v1261, %v1260 : tensor<128x16x16x16xf32>
    %v1263 = stablehlo.reduce(%v1262 init: %v1246) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v1264 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1265 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1266 = stablehlo.multiply %v1264, %g4m : tensor<16xf32>
    %v1267 = stablehlo.multiply %v1265, %v1263 : tensor<16xf32>
    %v1268 = stablehlo.add %v1266, %v1267 : tensor<16xf32>
    %v1269 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1270 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1271 = stablehlo.multiply %v1269, %g4v : tensor<16xf32>
    %v1272 = stablehlo.multiply %v1263, %v1263 : tensor<16xf32>
    %v1273 = stablehlo.multiply %v1270, %v1272 : tensor<16xf32>
    %v1274 = stablehlo.add %v1271, %v1273 : tensor<16xf32>
    %v1275 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1276 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1277 = stablehlo.divide %v1268, %v1275 : tensor<16xf32>
    %v1278 = stablehlo.divide %v1274, %v1276 : tensor<16xf32>
    %v1279 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1280 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1281 = stablehlo.sqrt %v1278 : tensor<16xf32>
    %v1282 = stablehlo.add %v1281, %v1280 : tensor<16xf32>
    %v1283 = stablehlo.divide %v1277, %v1282 : tensor<16xf32>
    %v1284 = stablehlo.multiply %v1279, %v1283 : tensor<16xf32>
    %v1285 = stablehlo.subtract %g4, %v1284 : tensor<16xf32>
    %v1286 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1287 = stablehlo.multiply %v1286, %v1279 : tensor<16xf32>
    %v1288 = stablehlo.multiply %v1287, %g4 : tensor<16xf32>
    %v1289 = stablehlo.subtract %v1285, %v1288 : tensor<16xf32>
    %v1290 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1291 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1292 = stablehlo.multiply %v1290, %g4m : tensor<16xf32>
    %v1293 = stablehlo.multiply %v1291, %v1263 : tensor<16xf32>
    %v1294 = stablehlo.add %v1292, %v1293 : tensor<16xf32>
    %v1295 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1296 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1297 = stablehlo.multiply %v1295, %g4v : tensor<16xf32>
    %v1298 = stablehlo.multiply %v1263, %v1263 : tensor<16xf32>
    %v1299 = stablehlo.multiply %v1296, %v1298 : tensor<16xf32>
    %v1300 = stablehlo.add %v1297, %v1299 : tensor<16xf32>
    %v1301 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1302 = stablehlo.reshape %v470 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1303 = stablehlo.reduce(%v1302 init: %v1301) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v1304 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1305 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1306 = stablehlo.multiply %v1304, %bt4m : tensor<16xf32>
    %v1307 = stablehlo.multiply %v1305, %v1303 : tensor<16xf32>
    %v1308 = stablehlo.add %v1306, %v1307 : tensor<16xf32>
    %v1309 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1310 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1311 = stablehlo.multiply %v1309, %bt4v : tensor<16xf32>
    %v1312 = stablehlo.multiply %v1303, %v1303 : tensor<16xf32>
    %v1313 = stablehlo.multiply %v1310, %v1312 : tensor<16xf32>
    %v1314 = stablehlo.add %v1311, %v1313 : tensor<16xf32>
    %v1315 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1316 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1317 = stablehlo.divide %v1308, %v1315 : tensor<16xf32>
    %v1318 = stablehlo.divide %v1314, %v1316 : tensor<16xf32>
    %v1319 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1320 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1321 = stablehlo.sqrt %v1318 : tensor<16xf32>
    %v1322 = stablehlo.add %v1321, %v1320 : tensor<16xf32>
    %v1323 = stablehlo.divide %v1317, %v1322 : tensor<16xf32>
    %v1324 = stablehlo.multiply %v1319, %v1323 : tensor<16xf32>
    %v1325 = stablehlo.subtract %bt4, %v1324 : tensor<16xf32>
    %v1326 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1327 = stablehlo.multiply %v1326, %v1319 : tensor<16xf32>
    %v1328 = stablehlo.multiply %v1327, %bt4 : tensor<16xf32>
    %v1329 = stablehlo.subtract %v1325, %v1328 : tensor<16xf32>
    %v1330 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1331 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1332 = stablehlo.multiply %v1330, %bt4m : tensor<16xf32>
    %v1333 = stablehlo.multiply %v1331, %v1303 : tensor<16xf32>
    %v1334 = stablehlo.add %v1332, %v1333 : tensor<16xf32>
    %v1335 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1336 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1337 = stablehlo.multiply %v1335, %bt4v : tensor<16xf32>
    %v1338 = stablehlo.multiply %v1303, %v1303 : tensor<16xf32>
    %v1339 = stablehlo.multiply %v1336, %v1338 : tensor<16xf32>
    %v1340 = stablehlo.add %v1337, %v1339 : tensor<16xf32>
    %v1341 = stablehlo.reshape %v123 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v1342 = stablehlo.reshape %v454 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1343 = stablehlo.transpose %v1341, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v1344 = stablehlo.transpose %v1342, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v1345 = stablehlo.convolution(%v1343, %v1344)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<16x32x3x3xf32>
    %v1346 = stablehlo.transpose %v1345, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v1347 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1348 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1349 = stablehlo.multiply %v1347, %W5m : tensor<32x16x3x3xf32>
    %v1350 = stablehlo.multiply %v1348, %v1346 : tensor<32x16x3x3xf32>
    %v1351 = stablehlo.add %v1349, %v1350 : tensor<32x16x3x3xf32>
    %v1352 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1353 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1354 = stablehlo.multiply %v1352, %W5v : tensor<32x16x3x3xf32>
    %v1355 = stablehlo.multiply %v1346, %v1346 : tensor<32x16x3x3xf32>
    %v1356 = stablehlo.multiply %v1353, %v1355 : tensor<32x16x3x3xf32>
    %v1357 = stablehlo.add %v1354, %v1356 : tensor<32x16x3x3xf32>
    %v1358 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1359 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1360 = stablehlo.divide %v1351, %v1358 : tensor<32x16x3x3xf32>
    %v1361 = stablehlo.divide %v1357, %v1359 : tensor<32x16x3x3xf32>
    %v1362 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1363 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1364 = stablehlo.sqrt %v1361 : tensor<32x16x3x3xf32>
    %v1365 = stablehlo.add %v1364, %v1363 : tensor<32x16x3x3xf32>
    %v1366 = stablehlo.divide %v1360, %v1365 : tensor<32x16x3x3xf32>
    %v1367 = stablehlo.multiply %v1362, %v1366 : tensor<32x16x3x3xf32>
    %v1368 = stablehlo.subtract %W5, %v1367 : tensor<32x16x3x3xf32>
    %v1369 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1370 = stablehlo.multiply %v1369, %v1362 : tensor<32x16x3x3xf32>
    %v1371 = stablehlo.multiply %v1370, %W5 : tensor<32x16x3x3xf32>
    %v1372 = stablehlo.subtract %v1368, %v1371 : tensor<32x16x3x3xf32>
    %v1373 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1374 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1375 = stablehlo.multiply %v1373, %W5m : tensor<32x16x3x3xf32>
    %v1376 = stablehlo.multiply %v1374, %v1346 : tensor<32x16x3x3xf32>
    %v1377 = stablehlo.add %v1375, %v1376 : tensor<32x16x3x3xf32>
    %v1378 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1379 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1380 = stablehlo.multiply %v1378, %W5v : tensor<32x16x3x3xf32>
    %v1381 = stablehlo.multiply %v1346, %v1346 : tensor<32x16x3x3xf32>
    %v1382 = stablehlo.multiply %v1379, %v1381 : tensor<32x16x3x3xf32>
    %v1383 = stablehlo.add %v1380, %v1382 : tensor<32x16x3x3xf32>
    %v1384 = stablehlo.reshape %v454 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1385 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1386 = stablehlo.reduce(%v1384 init: %v1385) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1387 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1388 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1389 = stablehlo.multiply %v1387, %cb5m : tensor<32xf32>
    %v1390 = stablehlo.multiply %v1388, %v1386 : tensor<32xf32>
    %v1391 = stablehlo.add %v1389, %v1390 : tensor<32xf32>
    %v1392 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1393 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1394 = stablehlo.multiply %v1392, %cb5v : tensor<32xf32>
    %v1395 = stablehlo.multiply %v1386, %v1386 : tensor<32xf32>
    %v1396 = stablehlo.multiply %v1393, %v1395 : tensor<32xf32>
    %v1397 = stablehlo.add %v1394, %v1396 : tensor<32xf32>
    %v1398 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1399 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1400 = stablehlo.divide %v1391, %v1398 : tensor<32xf32>
    %v1401 = stablehlo.divide %v1397, %v1399 : tensor<32xf32>
    %v1402 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1403 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1404 = stablehlo.sqrt %v1401 : tensor<32xf32>
    %v1405 = stablehlo.add %v1404, %v1403 : tensor<32xf32>
    %v1406 = stablehlo.divide %v1400, %v1405 : tensor<32xf32>
    %v1407 = stablehlo.multiply %v1402, %v1406 : tensor<32xf32>
    %v1408 = stablehlo.subtract %cb5, %v1407 : tensor<32xf32>
    %v1409 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1410 = stablehlo.multiply %v1409, %v1402 : tensor<32xf32>
    %v1411 = stablehlo.multiply %v1410, %cb5 : tensor<32xf32>
    %v1412 = stablehlo.subtract %v1408, %v1411 : tensor<32xf32>
    %v1413 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1414 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1415 = stablehlo.multiply %v1413, %cb5m : tensor<32xf32>
    %v1416 = stablehlo.multiply %v1414, %v1386 : tensor<32xf32>
    %v1417 = stablehlo.add %v1415, %v1416 : tensor<32xf32>
    %v1418 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1419 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1420 = stablehlo.multiply %v1418, %cb5v : tensor<32xf32>
    %v1421 = stablehlo.multiply %v1386, %v1386 : tensor<32xf32>
    %v1422 = stablehlo.multiply %v1419, %v1421 : tensor<32xf32>
    %v1423 = stablehlo.add %v1420, %v1422 : tensor<32xf32>
    %v1424 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1425 = stablehlo.reshape %v128 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1426 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v1427 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v1428 = stablehlo.reduce(%v1425 init: %v1424) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1429 = stablehlo.broadcast_in_dim %v1428, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v1430 = stablehlo.divide %v1429, %v1426 : tensor<128x32x8x8xf32>
    %v1431 = stablehlo.subtract %v1425, %v1430 : tensor<128x32x8x8xf32>
    %v1432 = stablehlo.multiply %v1431, %v1431 : tensor<128x32x8x8xf32>
    %v1433 = stablehlo.reduce(%v1432 init: %v1424) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1434 = stablehlo.broadcast_in_dim %v1433, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v1435 = stablehlo.divide %v1434, %v1426 : tensor<128x32x8x8xf32>
    %v1436 = stablehlo.add %v1435, %v1427 : tensor<128x32x8x8xf32>
    %v1437 = stablehlo.rsqrt %v1436 : tensor<128x32x8x8xf32>
    %v1438 = stablehlo.multiply %v1431, %v1437 : tensor<128x32x8x8xf32>
    %v1439 = stablehlo.reshape %v424 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1440 = stablehlo.multiply %v1439, %v1438 : tensor<128x32x8x8xf32>
    %v1441 = stablehlo.reduce(%v1440 init: %v1424) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1442 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1443 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1444 = stablehlo.multiply %v1442, %g5m : tensor<32xf32>
    %v1445 = stablehlo.multiply %v1443, %v1441 : tensor<32xf32>
    %v1446 = stablehlo.add %v1444, %v1445 : tensor<32xf32>
    %v1447 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1448 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1449 = stablehlo.multiply %v1447, %g5v : tensor<32xf32>
    %v1450 = stablehlo.multiply %v1441, %v1441 : tensor<32xf32>
    %v1451 = stablehlo.multiply %v1448, %v1450 : tensor<32xf32>
    %v1452 = stablehlo.add %v1449, %v1451 : tensor<32xf32>
    %v1453 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1454 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1455 = stablehlo.divide %v1446, %v1453 : tensor<32xf32>
    %v1456 = stablehlo.divide %v1452, %v1454 : tensor<32xf32>
    %v1457 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1458 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1459 = stablehlo.sqrt %v1456 : tensor<32xf32>
    %v1460 = stablehlo.add %v1459, %v1458 : tensor<32xf32>
    %v1461 = stablehlo.divide %v1455, %v1460 : tensor<32xf32>
    %v1462 = stablehlo.multiply %v1457, %v1461 : tensor<32xf32>
    %v1463 = stablehlo.subtract %g5, %v1462 : tensor<32xf32>
    %v1464 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1465 = stablehlo.multiply %v1464, %v1457 : tensor<32xf32>
    %v1466 = stablehlo.multiply %v1465, %g5 : tensor<32xf32>
    %v1467 = stablehlo.subtract %v1463, %v1466 : tensor<32xf32>
    %v1468 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1469 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1470 = stablehlo.multiply %v1468, %g5m : tensor<32xf32>
    %v1471 = stablehlo.multiply %v1469, %v1441 : tensor<32xf32>
    %v1472 = stablehlo.add %v1470, %v1471 : tensor<32xf32>
    %v1473 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1474 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1475 = stablehlo.multiply %v1473, %g5v : tensor<32xf32>
    %v1476 = stablehlo.multiply %v1441, %v1441 : tensor<32xf32>
    %v1477 = stablehlo.multiply %v1474, %v1476 : tensor<32xf32>
    %v1478 = stablehlo.add %v1475, %v1477 : tensor<32xf32>
    %v1479 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1480 = stablehlo.reshape %v424 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1481 = stablehlo.reduce(%v1480 init: %v1479) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1482 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1483 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1484 = stablehlo.multiply %v1482, %bt5m : tensor<32xf32>
    %v1485 = stablehlo.multiply %v1483, %v1481 : tensor<32xf32>
    %v1486 = stablehlo.add %v1484, %v1485 : tensor<32xf32>
    %v1487 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1488 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1489 = stablehlo.multiply %v1487, %bt5v : tensor<32xf32>
    %v1490 = stablehlo.multiply %v1481, %v1481 : tensor<32xf32>
    %v1491 = stablehlo.multiply %v1488, %v1490 : tensor<32xf32>
    %v1492 = stablehlo.add %v1489, %v1491 : tensor<32xf32>
    %v1493 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1494 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1495 = stablehlo.divide %v1486, %v1493 : tensor<32xf32>
    %v1496 = stablehlo.divide %v1492, %v1494 : tensor<32xf32>
    %v1497 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1498 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1499 = stablehlo.sqrt %v1496 : tensor<32xf32>
    %v1500 = stablehlo.add %v1499, %v1498 : tensor<32xf32>
    %v1501 = stablehlo.divide %v1495, %v1500 : tensor<32xf32>
    %v1502 = stablehlo.multiply %v1497, %v1501 : tensor<32xf32>
    %v1503 = stablehlo.subtract %bt5, %v1502 : tensor<32xf32>
    %v1504 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1505 = stablehlo.multiply %v1504, %v1497 : tensor<32xf32>
    %v1506 = stablehlo.multiply %v1505, %bt5 : tensor<32xf32>
    %v1507 = stablehlo.subtract %v1503, %v1506 : tensor<32xf32>
    %v1508 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1509 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1510 = stablehlo.multiply %v1508, %bt5m : tensor<32xf32>
    %v1511 = stablehlo.multiply %v1509, %v1481 : tensor<32xf32>
    %v1512 = stablehlo.add %v1510, %v1511 : tensor<32xf32>
    %v1513 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1514 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1515 = stablehlo.multiply %v1513, %bt5v : tensor<32xf32>
    %v1516 = stablehlo.multiply %v1481, %v1481 : tensor<32xf32>
    %v1517 = stablehlo.multiply %v1514, %v1516 : tensor<32xf32>
    %v1518 = stablehlo.add %v1515, %v1517 : tensor<32xf32>
    %v1519 = stablehlo.reshape %v152 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1520 = stablehlo.reshape %v413 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1521 = stablehlo.transpose %v1519, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v1522 = stablehlo.transpose %v1520, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v1523 = stablehlo.convolution(%v1521, %v1522)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<32x32x3x3xf32>
    %v1524 = stablehlo.transpose %v1523, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v1525 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1526 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1527 = stablehlo.multiply %v1525, %W6m : tensor<32x32x3x3xf32>
    %v1528 = stablehlo.multiply %v1526, %v1524 : tensor<32x32x3x3xf32>
    %v1529 = stablehlo.add %v1527, %v1528 : tensor<32x32x3x3xf32>
    %v1530 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1531 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1532 = stablehlo.multiply %v1530, %W6v : tensor<32x32x3x3xf32>
    %v1533 = stablehlo.multiply %v1524, %v1524 : tensor<32x32x3x3xf32>
    %v1534 = stablehlo.multiply %v1531, %v1533 : tensor<32x32x3x3xf32>
    %v1535 = stablehlo.add %v1532, %v1534 : tensor<32x32x3x3xf32>
    %v1536 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1537 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1538 = stablehlo.divide %v1529, %v1536 : tensor<32x32x3x3xf32>
    %v1539 = stablehlo.divide %v1535, %v1537 : tensor<32x32x3x3xf32>
    %v1540 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1541 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1542 = stablehlo.sqrt %v1539 : tensor<32x32x3x3xf32>
    %v1543 = stablehlo.add %v1542, %v1541 : tensor<32x32x3x3xf32>
    %v1544 = stablehlo.divide %v1538, %v1543 : tensor<32x32x3x3xf32>
    %v1545 = stablehlo.multiply %v1540, %v1544 : tensor<32x32x3x3xf32>
    %v1546 = stablehlo.subtract %W6, %v1545 : tensor<32x32x3x3xf32>
    %v1547 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1548 = stablehlo.multiply %v1547, %v1540 : tensor<32x32x3x3xf32>
    %v1549 = stablehlo.multiply %v1548, %W6 : tensor<32x32x3x3xf32>
    %v1550 = stablehlo.subtract %v1546, %v1549 : tensor<32x32x3x3xf32>
    %v1551 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1552 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1553 = stablehlo.multiply %v1551, %W6m : tensor<32x32x3x3xf32>
    %v1554 = stablehlo.multiply %v1552, %v1524 : tensor<32x32x3x3xf32>
    %v1555 = stablehlo.add %v1553, %v1554 : tensor<32x32x3x3xf32>
    %v1556 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1557 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1558 = stablehlo.multiply %v1556, %W6v : tensor<32x32x3x3xf32>
    %v1559 = stablehlo.multiply %v1524, %v1524 : tensor<32x32x3x3xf32>
    %v1560 = stablehlo.multiply %v1557, %v1559 : tensor<32x32x3x3xf32>
    %v1561 = stablehlo.add %v1558, %v1560 : tensor<32x32x3x3xf32>
    %v1562 = stablehlo.reshape %v413 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1563 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1564 = stablehlo.reduce(%v1562 init: %v1563) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1565 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1566 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1567 = stablehlo.multiply %v1565, %cb6m : tensor<32xf32>
    %v1568 = stablehlo.multiply %v1566, %v1564 : tensor<32xf32>
    %v1569 = stablehlo.add %v1567, %v1568 : tensor<32xf32>
    %v1570 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1571 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1572 = stablehlo.multiply %v1570, %cb6v : tensor<32xf32>
    %v1573 = stablehlo.multiply %v1564, %v1564 : tensor<32xf32>
    %v1574 = stablehlo.multiply %v1571, %v1573 : tensor<32xf32>
    %v1575 = stablehlo.add %v1572, %v1574 : tensor<32xf32>
    %v1576 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1577 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1578 = stablehlo.divide %v1569, %v1576 : tensor<32xf32>
    %v1579 = stablehlo.divide %v1575, %v1577 : tensor<32xf32>
    %v1580 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1581 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1582 = stablehlo.sqrt %v1579 : tensor<32xf32>
    %v1583 = stablehlo.add %v1582, %v1581 : tensor<32xf32>
    %v1584 = stablehlo.divide %v1578, %v1583 : tensor<32xf32>
    %v1585 = stablehlo.multiply %v1580, %v1584 : tensor<32xf32>
    %v1586 = stablehlo.subtract %cb6, %v1585 : tensor<32xf32>
    %v1587 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1588 = stablehlo.multiply %v1587, %v1580 : tensor<32xf32>
    %v1589 = stablehlo.multiply %v1588, %cb6 : tensor<32xf32>
    %v1590 = stablehlo.subtract %v1586, %v1589 : tensor<32xf32>
    %v1591 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1592 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1593 = stablehlo.multiply %v1591, %cb6m : tensor<32xf32>
    %v1594 = stablehlo.multiply %v1592, %v1564 : tensor<32xf32>
    %v1595 = stablehlo.add %v1593, %v1594 : tensor<32xf32>
    %v1596 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1597 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1598 = stablehlo.multiply %v1596, %cb6v : tensor<32xf32>
    %v1599 = stablehlo.multiply %v1564, %v1564 : tensor<32xf32>
    %v1600 = stablehlo.multiply %v1597, %v1599 : tensor<32xf32>
    %v1601 = stablehlo.add %v1598, %v1600 : tensor<32xf32>
    %v1602 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1603 = stablehlo.reshape %v157 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1604 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v1605 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v1606 = stablehlo.reduce(%v1603 init: %v1602) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1607 = stablehlo.broadcast_in_dim %v1606, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v1608 = stablehlo.divide %v1607, %v1604 : tensor<128x32x8x8xf32>
    %v1609 = stablehlo.subtract %v1603, %v1608 : tensor<128x32x8x8xf32>
    %v1610 = stablehlo.multiply %v1609, %v1609 : tensor<128x32x8x8xf32>
    %v1611 = stablehlo.reduce(%v1610 init: %v1602) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1612 = stablehlo.broadcast_in_dim %v1611, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v1613 = stablehlo.divide %v1612, %v1604 : tensor<128x32x8x8xf32>
    %v1614 = stablehlo.add %v1613, %v1605 : tensor<128x32x8x8xf32>
    %v1615 = stablehlo.rsqrt %v1614 : tensor<128x32x8x8xf32>
    %v1616 = stablehlo.multiply %v1609, %v1615 : tensor<128x32x8x8xf32>
    %v1617 = stablehlo.reshape %v383 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1618 = stablehlo.multiply %v1617, %v1616 : tensor<128x32x8x8xf32>
    %v1619 = stablehlo.reduce(%v1618 init: %v1602) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1620 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1621 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1622 = stablehlo.multiply %v1620, %g6m : tensor<32xf32>
    %v1623 = stablehlo.multiply %v1621, %v1619 : tensor<32xf32>
    %v1624 = stablehlo.add %v1622, %v1623 : tensor<32xf32>
    %v1625 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1626 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1627 = stablehlo.multiply %v1625, %g6v : tensor<32xf32>
    %v1628 = stablehlo.multiply %v1619, %v1619 : tensor<32xf32>
    %v1629 = stablehlo.multiply %v1626, %v1628 : tensor<32xf32>
    %v1630 = stablehlo.add %v1627, %v1629 : tensor<32xf32>
    %v1631 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1632 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1633 = stablehlo.divide %v1624, %v1631 : tensor<32xf32>
    %v1634 = stablehlo.divide %v1630, %v1632 : tensor<32xf32>
    %v1635 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1636 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1637 = stablehlo.sqrt %v1634 : tensor<32xf32>
    %v1638 = stablehlo.add %v1637, %v1636 : tensor<32xf32>
    %v1639 = stablehlo.divide %v1633, %v1638 : tensor<32xf32>
    %v1640 = stablehlo.multiply %v1635, %v1639 : tensor<32xf32>
    %v1641 = stablehlo.subtract %g6, %v1640 : tensor<32xf32>
    %v1642 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1643 = stablehlo.multiply %v1642, %v1635 : tensor<32xf32>
    %v1644 = stablehlo.multiply %v1643, %g6 : tensor<32xf32>
    %v1645 = stablehlo.subtract %v1641, %v1644 : tensor<32xf32>
    %v1646 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1647 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1648 = stablehlo.multiply %v1646, %g6m : tensor<32xf32>
    %v1649 = stablehlo.multiply %v1647, %v1619 : tensor<32xf32>
    %v1650 = stablehlo.add %v1648, %v1649 : tensor<32xf32>
    %v1651 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1652 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1653 = stablehlo.multiply %v1651, %g6v : tensor<32xf32>
    %v1654 = stablehlo.multiply %v1619, %v1619 : tensor<32xf32>
    %v1655 = stablehlo.multiply %v1652, %v1654 : tensor<32xf32>
    %v1656 = stablehlo.add %v1653, %v1655 : tensor<32xf32>
    %v1657 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1658 = stablehlo.reshape %v383 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1659 = stablehlo.reduce(%v1658 init: %v1657) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1660 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1661 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1662 = stablehlo.multiply %v1660, %bt6m : tensor<32xf32>
    %v1663 = stablehlo.multiply %v1661, %v1659 : tensor<32xf32>
    %v1664 = stablehlo.add %v1662, %v1663 : tensor<32xf32>
    %v1665 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1666 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1667 = stablehlo.multiply %v1665, %bt6v : tensor<32xf32>
    %v1668 = stablehlo.multiply %v1659, %v1659 : tensor<32xf32>
    %v1669 = stablehlo.multiply %v1666, %v1668 : tensor<32xf32>
    %v1670 = stablehlo.add %v1667, %v1669 : tensor<32xf32>
    %v1671 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1672 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1673 = stablehlo.divide %v1664, %v1671 : tensor<32xf32>
    %v1674 = stablehlo.divide %v1670, %v1672 : tensor<32xf32>
    %v1675 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1676 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1677 = stablehlo.sqrt %v1674 : tensor<32xf32>
    %v1678 = stablehlo.add %v1677, %v1676 : tensor<32xf32>
    %v1679 = stablehlo.divide %v1673, %v1678 : tensor<32xf32>
    %v1680 = stablehlo.multiply %v1675, %v1679 : tensor<32xf32>
    %v1681 = stablehlo.subtract %bt6, %v1680 : tensor<32xf32>
    %v1682 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1683 = stablehlo.multiply %v1682, %v1675 : tensor<32xf32>
    %v1684 = stablehlo.multiply %v1683, %bt6 : tensor<32xf32>
    %v1685 = stablehlo.subtract %v1681, %v1684 : tensor<32xf32>
    %v1686 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1687 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1688 = stablehlo.multiply %v1686, %bt6m : tensor<32xf32>
    %v1689 = stablehlo.multiply %v1687, %v1659 : tensor<32xf32>
    %v1690 = stablehlo.add %v1688, %v1689 : tensor<32xf32>
    %v1691 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1692 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1693 = stablehlo.multiply %v1691, %bt6v : tensor<32xf32>
    %v1694 = stablehlo.multiply %v1659, %v1659 : tensor<32xf32>
    %v1695 = stablehlo.multiply %v1692, %v1694 : tensor<32xf32>
    %v1696 = stablehlo.add %v1693, %v1695 : tensor<32xf32>
    %v1697 = stablehlo.reshape %v185 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1698 = stablehlo.reshape %v367 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1699 = stablehlo.transpose %v1697, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v1700 = stablehlo.transpose %v1698, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v1701 = stablehlo.convolution(%v1699, %v1700)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v1702 = stablehlo.transpose %v1701, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v1703 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1704 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1705 = stablehlo.multiply %v1703, %W7m : tensor<32x32x3x3xf32>
    %v1706 = stablehlo.multiply %v1704, %v1702 : tensor<32x32x3x3xf32>
    %v1707 = stablehlo.add %v1705, %v1706 : tensor<32x32x3x3xf32>
    %v1708 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1709 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1710 = stablehlo.multiply %v1708, %W7v : tensor<32x32x3x3xf32>
    %v1711 = stablehlo.multiply %v1702, %v1702 : tensor<32x32x3x3xf32>
    %v1712 = stablehlo.multiply %v1709, %v1711 : tensor<32x32x3x3xf32>
    %v1713 = stablehlo.add %v1710, %v1712 : tensor<32x32x3x3xf32>
    %v1714 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1715 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1716 = stablehlo.divide %v1707, %v1714 : tensor<32x32x3x3xf32>
    %v1717 = stablehlo.divide %v1713, %v1715 : tensor<32x32x3x3xf32>
    %v1718 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1719 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1720 = stablehlo.sqrt %v1717 : tensor<32x32x3x3xf32>
    %v1721 = stablehlo.add %v1720, %v1719 : tensor<32x32x3x3xf32>
    %v1722 = stablehlo.divide %v1716, %v1721 : tensor<32x32x3x3xf32>
    %v1723 = stablehlo.multiply %v1718, %v1722 : tensor<32x32x3x3xf32>
    %v1724 = stablehlo.subtract %W7, %v1723 : tensor<32x32x3x3xf32>
    %v1725 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1726 = stablehlo.multiply %v1725, %v1718 : tensor<32x32x3x3xf32>
    %v1727 = stablehlo.multiply %v1726, %W7 : tensor<32x32x3x3xf32>
    %v1728 = stablehlo.subtract %v1724, %v1727 : tensor<32x32x3x3xf32>
    %v1729 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1730 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1731 = stablehlo.multiply %v1729, %W7m : tensor<32x32x3x3xf32>
    %v1732 = stablehlo.multiply %v1730, %v1702 : tensor<32x32x3x3xf32>
    %v1733 = stablehlo.add %v1731, %v1732 : tensor<32x32x3x3xf32>
    %v1734 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1735 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1736 = stablehlo.multiply %v1734, %W7v : tensor<32x32x3x3xf32>
    %v1737 = stablehlo.multiply %v1702, %v1702 : tensor<32x32x3x3xf32>
    %v1738 = stablehlo.multiply %v1735, %v1737 : tensor<32x32x3x3xf32>
    %v1739 = stablehlo.add %v1736, %v1738 : tensor<32x32x3x3xf32>
    %v1740 = stablehlo.reshape %v367 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1741 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1742 = stablehlo.reduce(%v1740 init: %v1741) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1743 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1744 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1745 = stablehlo.multiply %v1743, %cb7m : tensor<32xf32>
    %v1746 = stablehlo.multiply %v1744, %v1742 : tensor<32xf32>
    %v1747 = stablehlo.add %v1745, %v1746 : tensor<32xf32>
    %v1748 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1749 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1750 = stablehlo.multiply %v1748, %cb7v : tensor<32xf32>
    %v1751 = stablehlo.multiply %v1742, %v1742 : tensor<32xf32>
    %v1752 = stablehlo.multiply %v1749, %v1751 : tensor<32xf32>
    %v1753 = stablehlo.add %v1750, %v1752 : tensor<32xf32>
    %v1754 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1755 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1756 = stablehlo.divide %v1747, %v1754 : tensor<32xf32>
    %v1757 = stablehlo.divide %v1753, %v1755 : tensor<32xf32>
    %v1758 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1759 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1760 = stablehlo.sqrt %v1757 : tensor<32xf32>
    %v1761 = stablehlo.add %v1760, %v1759 : tensor<32xf32>
    %v1762 = stablehlo.divide %v1756, %v1761 : tensor<32xf32>
    %v1763 = stablehlo.multiply %v1758, %v1762 : tensor<32xf32>
    %v1764 = stablehlo.subtract %cb7, %v1763 : tensor<32xf32>
    %v1765 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1766 = stablehlo.multiply %v1765, %v1758 : tensor<32xf32>
    %v1767 = stablehlo.multiply %v1766, %cb7 : tensor<32xf32>
    %v1768 = stablehlo.subtract %v1764, %v1767 : tensor<32xf32>
    %v1769 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1770 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1771 = stablehlo.multiply %v1769, %cb7m : tensor<32xf32>
    %v1772 = stablehlo.multiply %v1770, %v1742 : tensor<32xf32>
    %v1773 = stablehlo.add %v1771, %v1772 : tensor<32xf32>
    %v1774 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1775 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1776 = stablehlo.multiply %v1774, %cb7v : tensor<32xf32>
    %v1777 = stablehlo.multiply %v1742, %v1742 : tensor<32xf32>
    %v1778 = stablehlo.multiply %v1775, %v1777 : tensor<32xf32>
    %v1779 = stablehlo.add %v1776, %v1778 : tensor<32xf32>
    %v1780 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1781 = stablehlo.reshape %v190 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1782 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v1783 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v1784 = stablehlo.reduce(%v1781 init: %v1780) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1785 = stablehlo.broadcast_in_dim %v1784, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v1786 = stablehlo.divide %v1785, %v1782 : tensor<128x32x4x4xf32>
    %v1787 = stablehlo.subtract %v1781, %v1786 : tensor<128x32x4x4xf32>
    %v1788 = stablehlo.multiply %v1787, %v1787 : tensor<128x32x4x4xf32>
    %v1789 = stablehlo.reduce(%v1788 init: %v1780) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1790 = stablehlo.broadcast_in_dim %v1789, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v1791 = stablehlo.divide %v1790, %v1782 : tensor<128x32x4x4xf32>
    %v1792 = stablehlo.add %v1791, %v1783 : tensor<128x32x4x4xf32>
    %v1793 = stablehlo.rsqrt %v1792 : tensor<128x32x4x4xf32>
    %v1794 = stablehlo.multiply %v1787, %v1793 : tensor<128x32x4x4xf32>
    %v1795 = stablehlo.reshape %v337 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1796 = stablehlo.multiply %v1795, %v1794 : tensor<128x32x4x4xf32>
    %v1797 = stablehlo.reduce(%v1796 init: %v1780) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1798 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1799 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1800 = stablehlo.multiply %v1798, %g7m : tensor<32xf32>
    %v1801 = stablehlo.multiply %v1799, %v1797 : tensor<32xf32>
    %v1802 = stablehlo.add %v1800, %v1801 : tensor<32xf32>
    %v1803 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1804 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1805 = stablehlo.multiply %v1803, %g7v : tensor<32xf32>
    %v1806 = stablehlo.multiply %v1797, %v1797 : tensor<32xf32>
    %v1807 = stablehlo.multiply %v1804, %v1806 : tensor<32xf32>
    %v1808 = stablehlo.add %v1805, %v1807 : tensor<32xf32>
    %v1809 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1810 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1811 = stablehlo.divide %v1802, %v1809 : tensor<32xf32>
    %v1812 = stablehlo.divide %v1808, %v1810 : tensor<32xf32>
    %v1813 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1814 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1815 = stablehlo.sqrt %v1812 : tensor<32xf32>
    %v1816 = stablehlo.add %v1815, %v1814 : tensor<32xf32>
    %v1817 = stablehlo.divide %v1811, %v1816 : tensor<32xf32>
    %v1818 = stablehlo.multiply %v1813, %v1817 : tensor<32xf32>
    %v1819 = stablehlo.subtract %g7, %v1818 : tensor<32xf32>
    %v1820 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1821 = stablehlo.multiply %v1820, %v1813 : tensor<32xf32>
    %v1822 = stablehlo.multiply %v1821, %g7 : tensor<32xf32>
    %v1823 = stablehlo.subtract %v1819, %v1822 : tensor<32xf32>
    %v1824 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1825 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1826 = stablehlo.multiply %v1824, %g7m : tensor<32xf32>
    %v1827 = stablehlo.multiply %v1825, %v1797 : tensor<32xf32>
    %v1828 = stablehlo.add %v1826, %v1827 : tensor<32xf32>
    %v1829 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1830 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1831 = stablehlo.multiply %v1829, %g7v : tensor<32xf32>
    %v1832 = stablehlo.multiply %v1797, %v1797 : tensor<32xf32>
    %v1833 = stablehlo.multiply %v1830, %v1832 : tensor<32xf32>
    %v1834 = stablehlo.add %v1831, %v1833 : tensor<32xf32>
    %v1835 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1836 = stablehlo.reshape %v337 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1837 = stablehlo.reduce(%v1836 init: %v1835) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1838 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1839 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1840 = stablehlo.multiply %v1838, %bt7m : tensor<32xf32>
    %v1841 = stablehlo.multiply %v1839, %v1837 : tensor<32xf32>
    %v1842 = stablehlo.add %v1840, %v1841 : tensor<32xf32>
    %v1843 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1844 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1845 = stablehlo.multiply %v1843, %bt7v : tensor<32xf32>
    %v1846 = stablehlo.multiply %v1837, %v1837 : tensor<32xf32>
    %v1847 = stablehlo.multiply %v1844, %v1846 : tensor<32xf32>
    %v1848 = stablehlo.add %v1845, %v1847 : tensor<32xf32>
    %v1849 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1850 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1851 = stablehlo.divide %v1842, %v1849 : tensor<32xf32>
    %v1852 = stablehlo.divide %v1848, %v1850 : tensor<32xf32>
    %v1853 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1854 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1855 = stablehlo.sqrt %v1852 : tensor<32xf32>
    %v1856 = stablehlo.add %v1855, %v1854 : tensor<32xf32>
    %v1857 = stablehlo.divide %v1851, %v1856 : tensor<32xf32>
    %v1858 = stablehlo.multiply %v1853, %v1857 : tensor<32xf32>
    %v1859 = stablehlo.subtract %bt7, %v1858 : tensor<32xf32>
    %v1860 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1861 = stablehlo.multiply %v1860, %v1853 : tensor<32xf32>
    %v1862 = stablehlo.multiply %v1861, %bt7 : tensor<32xf32>
    %v1863 = stablehlo.subtract %v1859, %v1862 : tensor<32xf32>
    %v1864 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1865 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1866 = stablehlo.multiply %v1864, %bt7m : tensor<32xf32>
    %v1867 = stablehlo.multiply %v1865, %v1837 : tensor<32xf32>
    %v1868 = stablehlo.add %v1866, %v1867 : tensor<32xf32>
    %v1869 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1870 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1871 = stablehlo.multiply %v1869, %bt7v : tensor<32xf32>
    %v1872 = stablehlo.multiply %v1837, %v1837 : tensor<32xf32>
    %v1873 = stablehlo.multiply %v1870, %v1872 : tensor<32xf32>
    %v1874 = stablehlo.add %v1871, %v1873 : tensor<32xf32>
    %v1875 = stablehlo.reshape %v214 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1876 = stablehlo.reshape %v326 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1877 = stablehlo.transpose %v1875, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v1878 = stablehlo.transpose %v1876, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v1879 = stablehlo.convolution(%v1877, %v1878)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v1880 = stablehlo.transpose %v1879, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v1881 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1882 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1883 = stablehlo.multiply %v1881, %W8m : tensor<32x32x3x3xf32>
    %v1884 = stablehlo.multiply %v1882, %v1880 : tensor<32x32x3x3xf32>
    %v1885 = stablehlo.add %v1883, %v1884 : tensor<32x32x3x3xf32>
    %v1886 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1887 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1888 = stablehlo.multiply %v1886, %W8v : tensor<32x32x3x3xf32>
    %v1889 = stablehlo.multiply %v1880, %v1880 : tensor<32x32x3x3xf32>
    %v1890 = stablehlo.multiply %v1887, %v1889 : tensor<32x32x3x3xf32>
    %v1891 = stablehlo.add %v1888, %v1890 : tensor<32x32x3x3xf32>
    %v1892 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1893 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1894 = stablehlo.divide %v1885, %v1892 : tensor<32x32x3x3xf32>
    %v1895 = stablehlo.divide %v1891, %v1893 : tensor<32x32x3x3xf32>
    %v1896 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1897 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1898 = stablehlo.sqrt %v1895 : tensor<32x32x3x3xf32>
    %v1899 = stablehlo.add %v1898, %v1897 : tensor<32x32x3x3xf32>
    %v1900 = stablehlo.divide %v1894, %v1899 : tensor<32x32x3x3xf32>
    %v1901 = stablehlo.multiply %v1896, %v1900 : tensor<32x32x3x3xf32>
    %v1902 = stablehlo.subtract %W8, %v1901 : tensor<32x32x3x3xf32>
    %v1903 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1904 = stablehlo.multiply %v1903, %v1896 : tensor<32x32x3x3xf32>
    %v1905 = stablehlo.multiply %v1904, %W8 : tensor<32x32x3x3xf32>
    %v1906 = stablehlo.subtract %v1902, %v1905 : tensor<32x32x3x3xf32>
    %v1907 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1908 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1909 = stablehlo.multiply %v1907, %W8m : tensor<32x32x3x3xf32>
    %v1910 = stablehlo.multiply %v1908, %v1880 : tensor<32x32x3x3xf32>
    %v1911 = stablehlo.add %v1909, %v1910 : tensor<32x32x3x3xf32>
    %v1912 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1913 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1914 = stablehlo.multiply %v1912, %W8v : tensor<32x32x3x3xf32>
    %v1915 = stablehlo.multiply %v1880, %v1880 : tensor<32x32x3x3xf32>
    %v1916 = stablehlo.multiply %v1913, %v1915 : tensor<32x32x3x3xf32>
    %v1917 = stablehlo.add %v1914, %v1916 : tensor<32x32x3x3xf32>
    %v1918 = stablehlo.reshape %v326 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1919 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1920 = stablehlo.reduce(%v1918 init: %v1919) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1921 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1922 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1923 = stablehlo.multiply %v1921, %cb8m : tensor<32xf32>
    %v1924 = stablehlo.multiply %v1922, %v1920 : tensor<32xf32>
    %v1925 = stablehlo.add %v1923, %v1924 : tensor<32xf32>
    %v1926 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1927 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1928 = stablehlo.multiply %v1926, %cb8v : tensor<32xf32>
    %v1929 = stablehlo.multiply %v1920, %v1920 : tensor<32xf32>
    %v1930 = stablehlo.multiply %v1927, %v1929 : tensor<32xf32>
    %v1931 = stablehlo.add %v1928, %v1930 : tensor<32xf32>
    %v1932 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1933 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1934 = stablehlo.divide %v1925, %v1932 : tensor<32xf32>
    %v1935 = stablehlo.divide %v1931, %v1933 : tensor<32xf32>
    %v1936 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1937 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1938 = stablehlo.sqrt %v1935 : tensor<32xf32>
    %v1939 = stablehlo.add %v1938, %v1937 : tensor<32xf32>
    %v1940 = stablehlo.divide %v1934, %v1939 : tensor<32xf32>
    %v1941 = stablehlo.multiply %v1936, %v1940 : tensor<32xf32>
    %v1942 = stablehlo.subtract %cb8, %v1941 : tensor<32xf32>
    %v1943 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1944 = stablehlo.multiply %v1943, %v1936 : tensor<32xf32>
    %v1945 = stablehlo.multiply %v1944, %cb8 : tensor<32xf32>
    %v1946 = stablehlo.subtract %v1942, %v1945 : tensor<32xf32>
    %v1947 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1948 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1949 = stablehlo.multiply %v1947, %cb8m : tensor<32xf32>
    %v1950 = stablehlo.multiply %v1948, %v1920 : tensor<32xf32>
    %v1951 = stablehlo.add %v1949, %v1950 : tensor<32xf32>
    %v1952 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1953 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1954 = stablehlo.multiply %v1952, %cb8v : tensor<32xf32>
    %v1955 = stablehlo.multiply %v1920, %v1920 : tensor<32xf32>
    %v1956 = stablehlo.multiply %v1953, %v1955 : tensor<32xf32>
    %v1957 = stablehlo.add %v1954, %v1956 : tensor<32xf32>
    %v1958 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1959 = stablehlo.reshape %v219 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1960 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v1961 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v1962 = stablehlo.reduce(%v1959 init: %v1958) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1963 = stablehlo.broadcast_in_dim %v1962, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v1964 = stablehlo.divide %v1963, %v1960 : tensor<128x32x4x4xf32>
    %v1965 = stablehlo.subtract %v1959, %v1964 : tensor<128x32x4x4xf32>
    %v1966 = stablehlo.multiply %v1965, %v1965 : tensor<128x32x4x4xf32>
    %v1967 = stablehlo.reduce(%v1966 init: %v1958) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1968 = stablehlo.broadcast_in_dim %v1967, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v1969 = stablehlo.divide %v1968, %v1960 : tensor<128x32x4x4xf32>
    %v1970 = stablehlo.add %v1969, %v1961 : tensor<128x32x4x4xf32>
    %v1971 = stablehlo.rsqrt %v1970 : tensor<128x32x4x4xf32>
    %v1972 = stablehlo.multiply %v1965, %v1971 : tensor<128x32x4x4xf32>
    %v1973 = stablehlo.reshape %v296 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1974 = stablehlo.multiply %v1973, %v1972 : tensor<128x32x4x4xf32>
    %v1975 = stablehlo.reduce(%v1974 init: %v1958) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1976 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1977 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1978 = stablehlo.multiply %v1976, %g8m : tensor<32xf32>
    %v1979 = stablehlo.multiply %v1977, %v1975 : tensor<32xf32>
    %v1980 = stablehlo.add %v1978, %v1979 : tensor<32xf32>
    %v1981 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1982 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1983 = stablehlo.multiply %v1981, %g8v : tensor<32xf32>
    %v1984 = stablehlo.multiply %v1975, %v1975 : tensor<32xf32>
    %v1985 = stablehlo.multiply %v1982, %v1984 : tensor<32xf32>
    %v1986 = stablehlo.add %v1983, %v1985 : tensor<32xf32>
    %v1987 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1988 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1989 = stablehlo.divide %v1980, %v1987 : tensor<32xf32>
    %v1990 = stablehlo.divide %v1986, %v1988 : tensor<32xf32>
    %v1991 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1992 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1993 = stablehlo.sqrt %v1990 : tensor<32xf32>
    %v1994 = stablehlo.add %v1993, %v1992 : tensor<32xf32>
    %v1995 = stablehlo.divide %v1989, %v1994 : tensor<32xf32>
    %v1996 = stablehlo.multiply %v1991, %v1995 : tensor<32xf32>
    %v1997 = stablehlo.subtract %g8, %v1996 : tensor<32xf32>
    %v1998 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1999 = stablehlo.multiply %v1998, %v1991 : tensor<32xf32>
    %v2000 = stablehlo.multiply %v1999, %g8 : tensor<32xf32>
    %v2001 = stablehlo.subtract %v1997, %v2000 : tensor<32xf32>
    %v2002 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2003 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2004 = stablehlo.multiply %v2002, %g8m : tensor<32xf32>
    %v2005 = stablehlo.multiply %v2003, %v1975 : tensor<32xf32>
    %v2006 = stablehlo.add %v2004, %v2005 : tensor<32xf32>
    %v2007 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2008 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2009 = stablehlo.multiply %v2007, %g8v : tensor<32xf32>
    %v2010 = stablehlo.multiply %v1975, %v1975 : tensor<32xf32>
    %v2011 = stablehlo.multiply %v2008, %v2010 : tensor<32xf32>
    %v2012 = stablehlo.add %v2009, %v2011 : tensor<32xf32>
    %v2013 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2014 = stablehlo.reshape %v296 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v2015 = stablehlo.reduce(%v2014 init: %v2013) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v2016 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2017 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2018 = stablehlo.multiply %v2016, %bt8m : tensor<32xf32>
    %v2019 = stablehlo.multiply %v2017, %v2015 : tensor<32xf32>
    %v2020 = stablehlo.add %v2018, %v2019 : tensor<32xf32>
    %v2021 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2022 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2023 = stablehlo.multiply %v2021, %bt8v : tensor<32xf32>
    %v2024 = stablehlo.multiply %v2015, %v2015 : tensor<32xf32>
    %v2025 = stablehlo.multiply %v2022, %v2024 : tensor<32xf32>
    %v2026 = stablehlo.add %v2023, %v2025 : tensor<32xf32>
    %v2027 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2028 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2029 = stablehlo.divide %v2020, %v2027 : tensor<32xf32>
    %v2030 = stablehlo.divide %v2026, %v2028 : tensor<32xf32>
    %v2031 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2032 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2033 = stablehlo.sqrt %v2030 : tensor<32xf32>
    %v2034 = stablehlo.add %v2033, %v2032 : tensor<32xf32>
    %v2035 = stablehlo.divide %v2029, %v2034 : tensor<32xf32>
    %v2036 = stablehlo.multiply %v2031, %v2035 : tensor<32xf32>
    %v2037 = stablehlo.subtract %bt8, %v2036 : tensor<32xf32>
    %v2038 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2039 = stablehlo.multiply %v2038, %v2031 : tensor<32xf32>
    %v2040 = stablehlo.multiply %v2039, %bt8 : tensor<32xf32>
    %v2041 = stablehlo.subtract %v2037, %v2040 : tensor<32xf32>
    %v2042 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2043 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2044 = stablehlo.multiply %v2042, %bt8m : tensor<32xf32>
    %v2045 = stablehlo.multiply %v2043, %v2015 : tensor<32xf32>
    %v2046 = stablehlo.add %v2044, %v2045 : tensor<32xf32>
    %v2047 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2048 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2049 = stablehlo.multiply %v2047, %bt8v : tensor<32xf32>
    %v2050 = stablehlo.multiply %v2015, %v2015 : tensor<32xf32>
    %v2051 = stablehlo.multiply %v2048, %v2050 : tensor<32xf32>
    %v2052 = stablehlo.add %v2049, %v2051 : tensor<32xf32>
    %v2053 = stablehlo.dot_general %v247, %v282, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x512xf32>) -> tensor<128x512xf32>
    %v2054 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v2055 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v2056 = stablehlo.multiply %v2054, %W9m : tensor<128x512xf32>
    %v2057 = stablehlo.multiply %v2055, %v2053 : tensor<128x512xf32>
    %v2058 = stablehlo.add %v2056, %v2057 : tensor<128x512xf32>
    %v2059 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v2060 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v2061 = stablehlo.multiply %v2059, %W9v : tensor<128x512xf32>
    %v2062 = stablehlo.multiply %v2053, %v2053 : tensor<128x512xf32>
    %v2063 = stablehlo.multiply %v2060, %v2062 : tensor<128x512xf32>
    %v2064 = stablehlo.add %v2061, %v2063 : tensor<128x512xf32>
    %v2065 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v2066 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v2067 = stablehlo.divide %v2058, %v2065 : tensor<128x512xf32>
    %v2068 = stablehlo.divide %v2064, %v2066 : tensor<128x512xf32>
    %v2069 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v2070 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v2071 = stablehlo.sqrt %v2068 : tensor<128x512xf32>
    %v2072 = stablehlo.add %v2071, %v2070 : tensor<128x512xf32>
    %v2073 = stablehlo.divide %v2067, %v2072 : tensor<128x512xf32>
    %v2074 = stablehlo.multiply %v2069, %v2073 : tensor<128x512xf32>
    %v2075 = stablehlo.subtract %W9, %v2074 : tensor<128x512xf32>
    %v2076 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v2077 = stablehlo.multiply %v2076, %v2069 : tensor<128x512xf32>
    %v2078 = stablehlo.multiply %v2077, %W9 : tensor<128x512xf32>
    %v2079 = stablehlo.subtract %v2075, %v2078 : tensor<128x512xf32>
    %v2080 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v2081 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v2082 = stablehlo.multiply %v2080, %W9m : tensor<128x512xf32>
    %v2083 = stablehlo.multiply %v2081, %v2053 : tensor<128x512xf32>
    %v2084 = stablehlo.add %v2082, %v2083 : tensor<128x512xf32>
    %v2085 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v2086 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v2087 = stablehlo.multiply %v2085, %W9v : tensor<128x512xf32>
    %v2088 = stablehlo.multiply %v2053, %v2053 : tensor<128x512xf32>
    %v2089 = stablehlo.multiply %v2086, %v2088 : tensor<128x512xf32>
    %v2090 = stablehlo.add %v2087, %v2089 : tensor<128x512xf32>
    %v2091 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2092 = stablehlo.reduce(%v282 init: %v2091) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v2093 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2094 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2095 = stablehlo.multiply %v2093, %b9m : tensor<512xf32>
    %v2096 = stablehlo.multiply %v2094, %v2092 : tensor<512xf32>
    %v2097 = stablehlo.add %v2095, %v2096 : tensor<512xf32>
    %v2098 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2099 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2100 = stablehlo.multiply %v2098, %b9v : tensor<512xf32>
    %v2101 = stablehlo.multiply %v2092, %v2092 : tensor<512xf32>
    %v2102 = stablehlo.multiply %v2099, %v2101 : tensor<512xf32>
    %v2103 = stablehlo.add %v2100, %v2102 : tensor<512xf32>
    %v2104 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2105 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2106 = stablehlo.divide %v2097, %v2104 : tensor<512xf32>
    %v2107 = stablehlo.divide %v2103, %v2105 : tensor<512xf32>
    %v2108 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2109 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2110 = stablehlo.sqrt %v2107 : tensor<512xf32>
    %v2111 = stablehlo.add %v2110, %v2109 : tensor<512xf32>
    %v2112 = stablehlo.divide %v2106, %v2111 : tensor<512xf32>
    %v2113 = stablehlo.multiply %v2108, %v2112 : tensor<512xf32>
    %v2114 = stablehlo.subtract %b9, %v2113 : tensor<512xf32>
    %v2115 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2116 = stablehlo.multiply %v2115, %v2108 : tensor<512xf32>
    %v2117 = stablehlo.multiply %v2116, %b9 : tensor<512xf32>
    %v2118 = stablehlo.subtract %v2114, %v2117 : tensor<512xf32>
    %v2119 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2120 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2121 = stablehlo.multiply %v2119, %b9m : tensor<512xf32>
    %v2122 = stablehlo.multiply %v2120, %v2092 : tensor<512xf32>
    %v2123 = stablehlo.add %v2121, %v2122 : tensor<512xf32>
    %v2124 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2125 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2126 = stablehlo.multiply %v2124, %b9v : tensor<512xf32>
    %v2127 = stablehlo.multiply %v2092, %v2092 : tensor<512xf32>
    %v2128 = stablehlo.multiply %v2125, %v2127 : tensor<512xf32>
    %v2129 = stablehlo.add %v2126, %v2128 : tensor<512xf32>
    %v2130 = stablehlo.dot_general %v252, %v276, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<512x512xf32>
    %v2131 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v2132 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v2133 = stablehlo.multiply %v2131, %Wam : tensor<512x512xf32>
    %v2134 = stablehlo.multiply %v2132, %v2130 : tensor<512x512xf32>
    %v2135 = stablehlo.add %v2133, %v2134 : tensor<512x512xf32>
    %v2136 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v2137 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v2138 = stablehlo.multiply %v2136, %Wav : tensor<512x512xf32>
    %v2139 = stablehlo.multiply %v2130, %v2130 : tensor<512x512xf32>
    %v2140 = stablehlo.multiply %v2137, %v2139 : tensor<512x512xf32>
    %v2141 = stablehlo.add %v2138, %v2140 : tensor<512x512xf32>
    %v2142 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v2143 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v2144 = stablehlo.divide %v2135, %v2142 : tensor<512x512xf32>
    %v2145 = stablehlo.divide %v2141, %v2143 : tensor<512x512xf32>
    %v2146 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v2147 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v2148 = stablehlo.sqrt %v2145 : tensor<512x512xf32>
    %v2149 = stablehlo.add %v2148, %v2147 : tensor<512x512xf32>
    %v2150 = stablehlo.divide %v2144, %v2149 : tensor<512x512xf32>
    %v2151 = stablehlo.multiply %v2146, %v2150 : tensor<512x512xf32>
    %v2152 = stablehlo.subtract %Wa, %v2151 : tensor<512x512xf32>
    %v2153 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v2154 = stablehlo.multiply %v2153, %v2146 : tensor<512x512xf32>
    %v2155 = stablehlo.multiply %v2154, %Wa : tensor<512x512xf32>
    %v2156 = stablehlo.subtract %v2152, %v2155 : tensor<512x512xf32>
    %v2157 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v2158 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v2159 = stablehlo.multiply %v2157, %Wam : tensor<512x512xf32>
    %v2160 = stablehlo.multiply %v2158, %v2130 : tensor<512x512xf32>
    %v2161 = stablehlo.add %v2159, %v2160 : tensor<512x512xf32>
    %v2162 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v2163 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v2164 = stablehlo.multiply %v2162, %Wav : tensor<512x512xf32>
    %v2165 = stablehlo.multiply %v2130, %v2130 : tensor<512x512xf32>
    %v2166 = stablehlo.multiply %v2163, %v2165 : tensor<512x512xf32>
    %v2167 = stablehlo.add %v2164, %v2166 : tensor<512x512xf32>
    %v2168 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2169 = stablehlo.reduce(%v276 init: %v2168) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v2170 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2171 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2172 = stablehlo.multiply %v2170, %bam : tensor<512xf32>
    %v2173 = stablehlo.multiply %v2171, %v2169 : tensor<512xf32>
    %v2174 = stablehlo.add %v2172, %v2173 : tensor<512xf32>
    %v2175 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2176 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2177 = stablehlo.multiply %v2175, %bav : tensor<512xf32>
    %v2178 = stablehlo.multiply %v2169, %v2169 : tensor<512xf32>
    %v2179 = stablehlo.multiply %v2176, %v2178 : tensor<512xf32>
    %v2180 = stablehlo.add %v2177, %v2179 : tensor<512xf32>
    %v2181 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2182 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2183 = stablehlo.divide %v2174, %v2181 : tensor<512xf32>
    %v2184 = stablehlo.divide %v2180, %v2182 : tensor<512xf32>
    %v2185 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2186 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2187 = stablehlo.sqrt %v2184 : tensor<512xf32>
    %v2188 = stablehlo.add %v2187, %v2186 : tensor<512xf32>
    %v2189 = stablehlo.divide %v2183, %v2188 : tensor<512xf32>
    %v2190 = stablehlo.multiply %v2185, %v2189 : tensor<512xf32>
    %v2191 = stablehlo.subtract %ba, %v2190 : tensor<512xf32>
    %v2192 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2193 = stablehlo.multiply %v2192, %v2185 : tensor<512xf32>
    %v2194 = stablehlo.multiply %v2193, %ba : tensor<512xf32>
    %v2195 = stablehlo.subtract %v2191, %v2194 : tensor<512xf32>
    %v2196 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2197 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2198 = stablehlo.multiply %v2196, %bam : tensor<512xf32>
    %v2199 = stablehlo.multiply %v2197, %v2169 : tensor<512xf32>
    %v2200 = stablehlo.add %v2198, %v2199 : tensor<512xf32>
    %v2201 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2202 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v2203 = stablehlo.multiply %v2201, %bav : tensor<512xf32>
    %v2204 = stablehlo.multiply %v2169, %v2169 : tensor<512xf32>
    %v2205 = stablehlo.multiply %v2202, %v2204 : tensor<512xf32>
    %v2206 = stablehlo.add %v2203, %v2205 : tensor<512xf32>
    %v2207 = stablehlo.dot_general %v257, %v270, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x10xf32>) -> tensor<512x10xf32>
    %v2208 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v2209 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v2210 = stablehlo.multiply %v2208, %Wbm : tensor<512x10xf32>
    %v2211 = stablehlo.multiply %v2209, %v2207 : tensor<512x10xf32>
    %v2212 = stablehlo.add %v2210, %v2211 : tensor<512x10xf32>
    %v2213 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v2214 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v2215 = stablehlo.multiply %v2213, %Wbv : tensor<512x10xf32>
    %v2216 = stablehlo.multiply %v2207, %v2207 : tensor<512x10xf32>
    %v2217 = stablehlo.multiply %v2214, %v2216 : tensor<512x10xf32>
    %v2218 = stablehlo.add %v2215, %v2217 : tensor<512x10xf32>
    %v2219 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v2220 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v2221 = stablehlo.divide %v2212, %v2219 : tensor<512x10xf32>
    %v2222 = stablehlo.divide %v2218, %v2220 : tensor<512x10xf32>
    %v2223 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v2224 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v2225 = stablehlo.sqrt %v2222 : tensor<512x10xf32>
    %v2226 = stablehlo.add %v2225, %v2224 : tensor<512x10xf32>
    %v2227 = stablehlo.divide %v2221, %v2226 : tensor<512x10xf32>
    %v2228 = stablehlo.multiply %v2223, %v2227 : tensor<512x10xf32>
    %v2229 = stablehlo.subtract %Wb, %v2228 : tensor<512x10xf32>
    %v2230 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v2231 = stablehlo.multiply %v2230, %v2223 : tensor<512x10xf32>
    %v2232 = stablehlo.multiply %v2231, %Wb : tensor<512x10xf32>
    %v2233 = stablehlo.subtract %v2229, %v2232 : tensor<512x10xf32>
    %v2234 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v2235 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v2236 = stablehlo.multiply %v2234, %Wbm : tensor<512x10xf32>
    %v2237 = stablehlo.multiply %v2235, %v2207 : tensor<512x10xf32>
    %v2238 = stablehlo.add %v2236, %v2237 : tensor<512x10xf32>
    %v2239 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v2240 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v2241 = stablehlo.multiply %v2239, %Wbv : tensor<512x10xf32>
    %v2242 = stablehlo.multiply %v2207, %v2207 : tensor<512x10xf32>
    %v2243 = stablehlo.multiply %v2240, %v2242 : tensor<512x10xf32>
    %v2244 = stablehlo.add %v2241, %v2243 : tensor<512x10xf32>
    %v2245 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2246 = stablehlo.reduce(%v270 init: %v2245) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v2247 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2248 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2249 = stablehlo.multiply %v2247, %bbm : tensor<10xf32>
    %v2250 = stablehlo.multiply %v2248, %v2246 : tensor<10xf32>
    %v2251 = stablehlo.add %v2249, %v2250 : tensor<10xf32>
    %v2252 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2253 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2254 = stablehlo.multiply %v2252, %bbv : tensor<10xf32>
    %v2255 = stablehlo.multiply %v2246, %v2246 : tensor<10xf32>
    %v2256 = stablehlo.multiply %v2253, %v2255 : tensor<10xf32>
    %v2257 = stablehlo.add %v2254, %v2256 : tensor<10xf32>
    %v2258 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2259 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2260 = stablehlo.divide %v2251, %v2258 : tensor<10xf32>
    %v2261 = stablehlo.divide %v2257, %v2259 : tensor<10xf32>
    %v2262 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2263 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2264 = stablehlo.sqrt %v2261 : tensor<10xf32>
    %v2265 = stablehlo.add %v2264, %v2263 : tensor<10xf32>
    %v2266 = stablehlo.divide %v2260, %v2265 : tensor<10xf32>
    %v2267 = stablehlo.multiply %v2262, %v2266 : tensor<10xf32>
    %v2268 = stablehlo.subtract %bb, %v2267 : tensor<10xf32>
    %v2269 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2270 = stablehlo.multiply %v2269, %v2262 : tensor<10xf32>
    %v2271 = stablehlo.multiply %v2270, %bb : tensor<10xf32>
    %v2272 = stablehlo.subtract %v2268, %v2271 : tensor<10xf32>
    %v2273 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2274 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2275 = stablehlo.multiply %v2273, %bbm : tensor<10xf32>
    %v2276 = stablehlo.multiply %v2274, %v2246 : tensor<10xf32>
    %v2277 = stablehlo.add %v2275, %v2276 : tensor<10xf32>
    %v2278 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2279 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2280 = stablehlo.multiply %v2278, %bbv : tensor<10xf32>
    %v2281 = stablehlo.multiply %v2246, %v2246 : tensor<10xf32>
    %v2282 = stablehlo.multiply %v2279, %v2281 : tensor<10xf32>
    %v2283 = stablehlo.add %v2280, %v2282 : tensor<10xf32>
    return %v660, %v700, %v755, %v795, %v838, %v878, %v933, %v973, %v1016, %v1056, %v1111, %v1151, %v1194, %v1234, %v1289, %v1329, %v1372, %v1412, %v1467, %v1507, %v1550, %v1590, %v1645, %v1685, %v1728, %v1768, %v1823, %v1863, %v1906, %v1946, %v2001, %v2041, %v2079, %v2118, %v2156, %v2195, %v2233, %v2272, %v665, %v705, %v760, %v800, %v843, %v883, %v938, %v978, %v1021, %v1061, %v1116, %v1156, %v1199, %v1239, %v1294, %v1334, %v1377, %v1417, %v1472, %v1512, %v1555, %v1595, %v1650, %v1690, %v1733, %v1773, %v1828, %v1868, %v1911, %v1951, %v2006, %v2046, %v2084, %v2123, %v2161, %v2200, %v2238, %v2277, %v671, %v711, %v766, %v806, %v849, %v889, %v944, %v984, %v1027, %v1067, %v1122, %v1162, %v1205, %v1245, %v1300, %v1340, %v1383, %v1423, %v1478, %v1518, %v1561, %v1601, %v1656, %v1696, %v1739, %v1779, %v1834, %v1874, %v1917, %v1957, %v2012, %v2052, %v2090, %v2129, %v2167, %v2206, %v2244, %v2283, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
