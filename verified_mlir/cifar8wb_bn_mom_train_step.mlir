module @m {
  func.func @cifar8wb_bn_mom_train_step(%x: tensor<128x3072xf32>, %W1: tensor<16x3x3x3xf32>, %cb1: tensor<16xf32>, %g1: tensor<16xf32>, %bt1: tensor<16xf32>, %W2: tensor<16x16x3x3xf32>, %cb2: tensor<16xf32>, %g2: tensor<16xf32>, %bt2: tensor<16xf32>, %W3: tensor<16x16x3x3xf32>, %cb3: tensor<16xf32>, %g3: tensor<16xf32>, %bt3: tensor<16xf32>, %W4: tensor<16x16x3x3xf32>, %cb4: tensor<16xf32>, %g4: tensor<16xf32>, %bt4: tensor<16xf32>, %W5: tensor<32x16x3x3xf32>, %cb5: tensor<32xf32>, %g5: tensor<32xf32>, %bt5: tensor<32xf32>, %W6: tensor<32x32x3x3xf32>, %cb6: tensor<32xf32>, %g6: tensor<32xf32>, %bt6: tensor<32xf32>, %W7: tensor<32x32x3x3xf32>, %cb7: tensor<32xf32>, %g7: tensor<32xf32>, %bt7: tensor<32xf32>, %W8: tensor<32x32x3x3xf32>, %cb8: tensor<32xf32>, %g8: tensor<32xf32>, %bt8: tensor<32xf32>, %W9: tensor<128x512xf32>, %b9: tensor<512xf32>, %Wa: tensor<512x512xf32>, %ba: tensor<512xf32>, %Wb: tensor<512x10xf32>, %bb: tensor<10xf32>, %W1m: tensor<16x3x3x3xf32>, %cb1m: tensor<16xf32>, %g1m: tensor<16xf32>, %bt1m: tensor<16xf32>, %W2m: tensor<16x16x3x3xf32>, %cb2m: tensor<16xf32>, %g2m: tensor<16xf32>, %bt2m: tensor<16xf32>, %W3m: tensor<16x16x3x3xf32>, %cb3m: tensor<16xf32>, %g3m: tensor<16xf32>, %bt3m: tensor<16xf32>, %W4m: tensor<16x16x3x3xf32>, %cb4m: tensor<16xf32>, %g4m: tensor<16xf32>, %bt4m: tensor<16xf32>, %W5m: tensor<32x16x3x3xf32>, %cb5m: tensor<32xf32>, %g5m: tensor<32xf32>, %bt5m: tensor<32xf32>, %W6m: tensor<32x32x3x3xf32>, %cb6m: tensor<32xf32>, %g6m: tensor<32xf32>, %bt6m: tensor<32xf32>, %W7m: tensor<32x32x3x3xf32>, %cb7m: tensor<32xf32>, %g7m: tensor<32xf32>, %bt7m: tensor<32xf32>, %W8m: tensor<32x32x3x3xf32>, %cb8m: tensor<32xf32>, %g8m: tensor<32xf32>, %bt8m: tensor<32xf32>, %W9m: tensor<128x512xf32>, %b9m: tensor<512xf32>, %Wam: tensor<512x512xf32>, %bam: tensor<512xf32>, %Wbm: tensor<512x10xf32>, %bbm: tensor<10xf32>, %W1v: tensor<16x3x3x3xf32>, %cb1v: tensor<16xf32>, %g1v: tensor<16xf32>, %bt1v: tensor<16xf32>, %W2v: tensor<16x16x3x3xf32>, %cb2v: tensor<16xf32>, %g2v: tensor<16xf32>, %bt2v: tensor<16xf32>, %W3v: tensor<16x16x3x3xf32>, %cb3v: tensor<16xf32>, %g3v: tensor<16xf32>, %bt3v: tensor<16xf32>, %W4v: tensor<16x16x3x3xf32>, %cb4v: tensor<16xf32>, %g4v: tensor<16xf32>, %bt4v: tensor<16xf32>, %W5v: tensor<32x16x3x3xf32>, %cb5v: tensor<32xf32>, %g5v: tensor<32xf32>, %bt5v: tensor<32xf32>, %W6v: tensor<32x32x3x3xf32>, %cb6v: tensor<32xf32>, %g6v: tensor<32xf32>, %bt6v: tensor<32xf32>, %W7v: tensor<32x32x3x3xf32>, %cb7v: tensor<32xf32>, %g7v: tensor<32xf32>, %bt7v: tensor<32xf32>, %W8v: tensor<32x32x3x3xf32>, %cb8v: tensor<32xf32>, %g8v: tensor<32xf32>, %bt8v: tensor<32xf32>, %W9v: tensor<128x512xf32>, %b9v: tensor<512xf32>, %Wav: tensor<512x512xf32>, %bav: tensor<512xf32>, %Wbv: tensor<512x10xf32>, %bbv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: tensor<128x10xf32>) -> (tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>) {
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
    %v635 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v636 = stablehlo.multiply %v635, %W1v : tensor<16x3x3x3xf32>
    %v637 = stablehlo.add %v636, %v634 : tensor<16x3x3x3xf32>
    %v638 = stablehlo.multiply %v635, %v637 : tensor<16x3x3x3xf32>
    %v639 = stablehlo.add %v638, %v634 : tensor<16x3x3x3xf32>
    %v640 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v641 = stablehlo.multiply %v640, %v639 : tensor<16x3x3x3xf32>
    %v642 = stablehlo.subtract %W1, %v641 : tensor<16x3x3x3xf32>
    %v643 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v644 = stablehlo.multiply %v643, %W1v : tensor<16x3x3x3xf32>
    %v645 = stablehlo.add %v644, %v634 : tensor<16x3x3x3xf32>
    %v646 = stablehlo.reshape %v628 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v647 = stablehlo.constant dense<0.0> : tensor<f32>
    %v648 = stablehlo.reduce(%v646 init: %v647) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v649 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v650 = stablehlo.multiply %v649, %cb1v : tensor<16xf32>
    %v651 = stablehlo.add %v650, %v648 : tensor<16xf32>
    %v652 = stablehlo.multiply %v649, %v651 : tensor<16xf32>
    %v653 = stablehlo.add %v652, %v648 : tensor<16xf32>
    %v654 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v655 = stablehlo.multiply %v654, %v653 : tensor<16xf32>
    %v656 = stablehlo.subtract %cb1, %v655 : tensor<16xf32>
    %v657 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v658 = stablehlo.multiply %v657, %cb1v : tensor<16xf32>
    %v659 = stablehlo.add %v658, %v648 : tensor<16xf32>
    %v660 = stablehlo.constant dense<0.0> : tensor<f32>
    %v661 = stablehlo.reshape %v4 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v662 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v663 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v664 = stablehlo.reduce(%v661 init: %v660) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v665 = stablehlo.broadcast_in_dim %v664, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v666 = stablehlo.divide %v665, %v662 : tensor<128x16x32x32xf32>
    %v667 = stablehlo.subtract %v661, %v666 : tensor<128x16x32x32xf32>
    %v668 = stablehlo.multiply %v667, %v667 : tensor<128x16x32x32xf32>
    %v669 = stablehlo.reduce(%v668 init: %v660) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v670 = stablehlo.broadcast_in_dim %v669, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v671 = stablehlo.divide %v670, %v662 : tensor<128x16x32x32xf32>
    %v672 = stablehlo.add %v671, %v663 : tensor<128x16x32x32xf32>
    %v673 = stablehlo.rsqrt %v672 : tensor<128x16x32x32xf32>
    %v674 = stablehlo.multiply %v667, %v673 : tensor<128x16x32x32xf32>
    %v675 = stablehlo.reshape %v598 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v676 = stablehlo.multiply %v675, %v674 : tensor<128x16x32x32xf32>
    %v677 = stablehlo.reduce(%v676 init: %v660) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v678 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v679 = stablehlo.multiply %v678, %g1v : tensor<16xf32>
    %v680 = stablehlo.add %v679, %v677 : tensor<16xf32>
    %v681 = stablehlo.multiply %v678, %v680 : tensor<16xf32>
    %v682 = stablehlo.add %v681, %v677 : tensor<16xf32>
    %v683 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v684 = stablehlo.multiply %v683, %v682 : tensor<16xf32>
    %v685 = stablehlo.subtract %g1, %v684 : tensor<16xf32>
    %v686 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v687 = stablehlo.multiply %v686, %g1v : tensor<16xf32>
    %v688 = stablehlo.add %v687, %v677 : tensor<16xf32>
    %v689 = stablehlo.constant dense<0.0> : tensor<f32>
    %v690 = stablehlo.reshape %v598 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v691 = stablehlo.reduce(%v690 init: %v689) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v692 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v693 = stablehlo.multiply %v692, %bt1v : tensor<16xf32>
    %v694 = stablehlo.add %v693, %v691 : tensor<16xf32>
    %v695 = stablehlo.multiply %v692, %v694 : tensor<16xf32>
    %v696 = stablehlo.add %v695, %v691 : tensor<16xf32>
    %v697 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v698 = stablehlo.multiply %v697, %v696 : tensor<16xf32>
    %v699 = stablehlo.subtract %bt1, %v698 : tensor<16xf32>
    %v700 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v701 = stablehlo.multiply %v700, %bt1v : tensor<16xf32>
    %v702 = stablehlo.add %v701, %v691 : tensor<16xf32>
    %v703 = stablehlo.reshape %v28 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v704 = stablehlo.reshape %v587 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v705 = stablehlo.transpose %v703, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v706 = stablehlo.transpose %v704, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v707 = stablehlo.convolution(%v705, %v706)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x16x3x3xf32>
    %v708 = stablehlo.transpose %v707, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v709 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v710 = stablehlo.multiply %v709, %W2v : tensor<16x16x3x3xf32>
    %v711 = stablehlo.add %v710, %v708 : tensor<16x16x3x3xf32>
    %v712 = stablehlo.multiply %v709, %v711 : tensor<16x16x3x3xf32>
    %v713 = stablehlo.add %v712, %v708 : tensor<16x16x3x3xf32>
    %v714 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v715 = stablehlo.multiply %v714, %v713 : tensor<16x16x3x3xf32>
    %v716 = stablehlo.subtract %W2, %v715 : tensor<16x16x3x3xf32>
    %v717 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v718 = stablehlo.multiply %v717, %W2v : tensor<16x16x3x3xf32>
    %v719 = stablehlo.add %v718, %v708 : tensor<16x16x3x3xf32>
    %v720 = stablehlo.reshape %v587 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v721 = stablehlo.constant dense<0.0> : tensor<f32>
    %v722 = stablehlo.reduce(%v720 init: %v721) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v723 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v724 = stablehlo.multiply %v723, %cb2v : tensor<16xf32>
    %v725 = stablehlo.add %v724, %v722 : tensor<16xf32>
    %v726 = stablehlo.multiply %v723, %v725 : tensor<16xf32>
    %v727 = stablehlo.add %v726, %v722 : tensor<16xf32>
    %v728 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v729 = stablehlo.multiply %v728, %v727 : tensor<16xf32>
    %v730 = stablehlo.subtract %cb2, %v729 : tensor<16xf32>
    %v731 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v732 = stablehlo.multiply %v731, %cb2v : tensor<16xf32>
    %v733 = stablehlo.add %v732, %v722 : tensor<16xf32>
    %v734 = stablehlo.constant dense<0.0> : tensor<f32>
    %v735 = stablehlo.reshape %v33 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v736 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v737 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v738 = stablehlo.reduce(%v735 init: %v734) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v739 = stablehlo.broadcast_in_dim %v738, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v740 = stablehlo.divide %v739, %v736 : tensor<128x16x32x32xf32>
    %v741 = stablehlo.subtract %v735, %v740 : tensor<128x16x32x32xf32>
    %v742 = stablehlo.multiply %v741, %v741 : tensor<128x16x32x32xf32>
    %v743 = stablehlo.reduce(%v742 init: %v734) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v744 = stablehlo.broadcast_in_dim %v743, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v745 = stablehlo.divide %v744, %v736 : tensor<128x16x32x32xf32>
    %v746 = stablehlo.add %v745, %v737 : tensor<128x16x32x32xf32>
    %v747 = stablehlo.rsqrt %v746 : tensor<128x16x32x32xf32>
    %v748 = stablehlo.multiply %v741, %v747 : tensor<128x16x32x32xf32>
    %v749 = stablehlo.reshape %v557 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v750 = stablehlo.multiply %v749, %v748 : tensor<128x16x32x32xf32>
    %v751 = stablehlo.reduce(%v750 init: %v734) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v752 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v753 = stablehlo.multiply %v752, %g2v : tensor<16xf32>
    %v754 = stablehlo.add %v753, %v751 : tensor<16xf32>
    %v755 = stablehlo.multiply %v752, %v754 : tensor<16xf32>
    %v756 = stablehlo.add %v755, %v751 : tensor<16xf32>
    %v757 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v758 = stablehlo.multiply %v757, %v756 : tensor<16xf32>
    %v759 = stablehlo.subtract %g2, %v758 : tensor<16xf32>
    %v760 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v761 = stablehlo.multiply %v760, %g2v : tensor<16xf32>
    %v762 = stablehlo.add %v761, %v751 : tensor<16xf32>
    %v763 = stablehlo.constant dense<0.0> : tensor<f32>
    %v764 = stablehlo.reshape %v557 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v765 = stablehlo.reduce(%v764 init: %v763) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v766 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v767 = stablehlo.multiply %v766, %bt2v : tensor<16xf32>
    %v768 = stablehlo.add %v767, %v765 : tensor<16xf32>
    %v769 = stablehlo.multiply %v766, %v768 : tensor<16xf32>
    %v770 = stablehlo.add %v769, %v765 : tensor<16xf32>
    %v771 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v772 = stablehlo.multiply %v771, %v770 : tensor<16xf32>
    %v773 = stablehlo.subtract %bt2, %v772 : tensor<16xf32>
    %v774 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v775 = stablehlo.multiply %v774, %bt2v : tensor<16xf32>
    %v776 = stablehlo.add %v775, %v765 : tensor<16xf32>
    %v777 = stablehlo.reshape %v61 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v778 = stablehlo.reshape %v541 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v779 = stablehlo.transpose %v777, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v780 = stablehlo.transpose %v778, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v781 = stablehlo.convolution(%v779, %v780)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v782 = stablehlo.transpose %v781, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v783 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v784 = stablehlo.multiply %v783, %W3v : tensor<16x16x3x3xf32>
    %v785 = stablehlo.add %v784, %v782 : tensor<16x16x3x3xf32>
    %v786 = stablehlo.multiply %v783, %v785 : tensor<16x16x3x3xf32>
    %v787 = stablehlo.add %v786, %v782 : tensor<16x16x3x3xf32>
    %v788 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v789 = stablehlo.multiply %v788, %v787 : tensor<16x16x3x3xf32>
    %v790 = stablehlo.subtract %W3, %v789 : tensor<16x16x3x3xf32>
    %v791 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v792 = stablehlo.multiply %v791, %W3v : tensor<16x16x3x3xf32>
    %v793 = stablehlo.add %v792, %v782 : tensor<16x16x3x3xf32>
    %v794 = stablehlo.reshape %v541 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v795 = stablehlo.constant dense<0.0> : tensor<f32>
    %v796 = stablehlo.reduce(%v794 init: %v795) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v797 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v798 = stablehlo.multiply %v797, %cb3v : tensor<16xf32>
    %v799 = stablehlo.add %v798, %v796 : tensor<16xf32>
    %v800 = stablehlo.multiply %v797, %v799 : tensor<16xf32>
    %v801 = stablehlo.add %v800, %v796 : tensor<16xf32>
    %v802 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v803 = stablehlo.multiply %v802, %v801 : tensor<16xf32>
    %v804 = stablehlo.subtract %cb3, %v803 : tensor<16xf32>
    %v805 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v806 = stablehlo.multiply %v805, %cb3v : tensor<16xf32>
    %v807 = stablehlo.add %v806, %v796 : tensor<16xf32>
    %v808 = stablehlo.constant dense<0.0> : tensor<f32>
    %v809 = stablehlo.reshape %v66 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v810 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v811 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v812 = stablehlo.reduce(%v809 init: %v808) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v813 = stablehlo.broadcast_in_dim %v812, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v814 = stablehlo.divide %v813, %v810 : tensor<128x16x16x16xf32>
    %v815 = stablehlo.subtract %v809, %v814 : tensor<128x16x16x16xf32>
    %v816 = stablehlo.multiply %v815, %v815 : tensor<128x16x16x16xf32>
    %v817 = stablehlo.reduce(%v816 init: %v808) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v818 = stablehlo.broadcast_in_dim %v817, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v819 = stablehlo.divide %v818, %v810 : tensor<128x16x16x16xf32>
    %v820 = stablehlo.add %v819, %v811 : tensor<128x16x16x16xf32>
    %v821 = stablehlo.rsqrt %v820 : tensor<128x16x16x16xf32>
    %v822 = stablehlo.multiply %v815, %v821 : tensor<128x16x16x16xf32>
    %v823 = stablehlo.reshape %v511 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v824 = stablehlo.multiply %v823, %v822 : tensor<128x16x16x16xf32>
    %v825 = stablehlo.reduce(%v824 init: %v808) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v826 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v827 = stablehlo.multiply %v826, %g3v : tensor<16xf32>
    %v828 = stablehlo.add %v827, %v825 : tensor<16xf32>
    %v829 = stablehlo.multiply %v826, %v828 : tensor<16xf32>
    %v830 = stablehlo.add %v829, %v825 : tensor<16xf32>
    %v831 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v832 = stablehlo.multiply %v831, %v830 : tensor<16xf32>
    %v833 = stablehlo.subtract %g3, %v832 : tensor<16xf32>
    %v834 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v835 = stablehlo.multiply %v834, %g3v : tensor<16xf32>
    %v836 = stablehlo.add %v835, %v825 : tensor<16xf32>
    %v837 = stablehlo.constant dense<0.0> : tensor<f32>
    %v838 = stablehlo.reshape %v511 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v839 = stablehlo.reduce(%v838 init: %v837) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v840 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v841 = stablehlo.multiply %v840, %bt3v : tensor<16xf32>
    %v842 = stablehlo.add %v841, %v839 : tensor<16xf32>
    %v843 = stablehlo.multiply %v840, %v842 : tensor<16xf32>
    %v844 = stablehlo.add %v843, %v839 : tensor<16xf32>
    %v845 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v846 = stablehlo.multiply %v845, %v844 : tensor<16xf32>
    %v847 = stablehlo.subtract %bt3, %v846 : tensor<16xf32>
    %v848 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v849 = stablehlo.multiply %v848, %bt3v : tensor<16xf32>
    %v850 = stablehlo.add %v849, %v839 : tensor<16xf32>
    %v851 = stablehlo.reshape %v90 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v852 = stablehlo.reshape %v500 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v853 = stablehlo.transpose %v851, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v854 = stablehlo.transpose %v852, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v855 = stablehlo.convolution(%v853, %v854)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v856 = stablehlo.transpose %v855, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v857 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v858 = stablehlo.multiply %v857, %W4v : tensor<16x16x3x3xf32>
    %v859 = stablehlo.add %v858, %v856 : tensor<16x16x3x3xf32>
    %v860 = stablehlo.multiply %v857, %v859 : tensor<16x16x3x3xf32>
    %v861 = stablehlo.add %v860, %v856 : tensor<16x16x3x3xf32>
    %v862 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v863 = stablehlo.multiply %v862, %v861 : tensor<16x16x3x3xf32>
    %v864 = stablehlo.subtract %W4, %v863 : tensor<16x16x3x3xf32>
    %v865 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v866 = stablehlo.multiply %v865, %W4v : tensor<16x16x3x3xf32>
    %v867 = stablehlo.add %v866, %v856 : tensor<16x16x3x3xf32>
    %v868 = stablehlo.reshape %v500 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v869 = stablehlo.constant dense<0.0> : tensor<f32>
    %v870 = stablehlo.reduce(%v868 init: %v869) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v871 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v872 = stablehlo.multiply %v871, %cb4v : tensor<16xf32>
    %v873 = stablehlo.add %v872, %v870 : tensor<16xf32>
    %v874 = stablehlo.multiply %v871, %v873 : tensor<16xf32>
    %v875 = stablehlo.add %v874, %v870 : tensor<16xf32>
    %v876 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v877 = stablehlo.multiply %v876, %v875 : tensor<16xf32>
    %v878 = stablehlo.subtract %cb4, %v877 : tensor<16xf32>
    %v879 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v880 = stablehlo.multiply %v879, %cb4v : tensor<16xf32>
    %v881 = stablehlo.add %v880, %v870 : tensor<16xf32>
    %v882 = stablehlo.constant dense<0.0> : tensor<f32>
    %v883 = stablehlo.reshape %v95 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v884 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v885 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v886 = stablehlo.reduce(%v883 init: %v882) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v887 = stablehlo.broadcast_in_dim %v886, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v888 = stablehlo.divide %v887, %v884 : tensor<128x16x16x16xf32>
    %v889 = stablehlo.subtract %v883, %v888 : tensor<128x16x16x16xf32>
    %v890 = stablehlo.multiply %v889, %v889 : tensor<128x16x16x16xf32>
    %v891 = stablehlo.reduce(%v890 init: %v882) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v892 = stablehlo.broadcast_in_dim %v891, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v893 = stablehlo.divide %v892, %v884 : tensor<128x16x16x16xf32>
    %v894 = stablehlo.add %v893, %v885 : tensor<128x16x16x16xf32>
    %v895 = stablehlo.rsqrt %v894 : tensor<128x16x16x16xf32>
    %v896 = stablehlo.multiply %v889, %v895 : tensor<128x16x16x16xf32>
    %v897 = stablehlo.reshape %v470 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v898 = stablehlo.multiply %v897, %v896 : tensor<128x16x16x16xf32>
    %v899 = stablehlo.reduce(%v898 init: %v882) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v900 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v901 = stablehlo.multiply %v900, %g4v : tensor<16xf32>
    %v902 = stablehlo.add %v901, %v899 : tensor<16xf32>
    %v903 = stablehlo.multiply %v900, %v902 : tensor<16xf32>
    %v904 = stablehlo.add %v903, %v899 : tensor<16xf32>
    %v905 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v906 = stablehlo.multiply %v905, %v904 : tensor<16xf32>
    %v907 = stablehlo.subtract %g4, %v906 : tensor<16xf32>
    %v908 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v909 = stablehlo.multiply %v908, %g4v : tensor<16xf32>
    %v910 = stablehlo.add %v909, %v899 : tensor<16xf32>
    %v911 = stablehlo.constant dense<0.0> : tensor<f32>
    %v912 = stablehlo.reshape %v470 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v913 = stablehlo.reduce(%v912 init: %v911) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v914 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v915 = stablehlo.multiply %v914, %bt4v : tensor<16xf32>
    %v916 = stablehlo.add %v915, %v913 : tensor<16xf32>
    %v917 = stablehlo.multiply %v914, %v916 : tensor<16xf32>
    %v918 = stablehlo.add %v917, %v913 : tensor<16xf32>
    %v919 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v920 = stablehlo.multiply %v919, %v918 : tensor<16xf32>
    %v921 = stablehlo.subtract %bt4, %v920 : tensor<16xf32>
    %v922 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v923 = stablehlo.multiply %v922, %bt4v : tensor<16xf32>
    %v924 = stablehlo.add %v923, %v913 : tensor<16xf32>
    %v925 = stablehlo.reshape %v123 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v926 = stablehlo.reshape %v454 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v927 = stablehlo.transpose %v925, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v928 = stablehlo.transpose %v926, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v929 = stablehlo.convolution(%v927, %v928)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<16x32x3x3xf32>
    %v930 = stablehlo.transpose %v929, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v931 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v932 = stablehlo.multiply %v931, %W5v : tensor<32x16x3x3xf32>
    %v933 = stablehlo.add %v932, %v930 : tensor<32x16x3x3xf32>
    %v934 = stablehlo.multiply %v931, %v933 : tensor<32x16x3x3xf32>
    %v935 = stablehlo.add %v934, %v930 : tensor<32x16x3x3xf32>
    %v936 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v937 = stablehlo.multiply %v936, %v935 : tensor<32x16x3x3xf32>
    %v938 = stablehlo.subtract %W5, %v937 : tensor<32x16x3x3xf32>
    %v939 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v940 = stablehlo.multiply %v939, %W5v : tensor<32x16x3x3xf32>
    %v941 = stablehlo.add %v940, %v930 : tensor<32x16x3x3xf32>
    %v942 = stablehlo.reshape %v454 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v943 = stablehlo.constant dense<0.0> : tensor<f32>
    %v944 = stablehlo.reduce(%v942 init: %v943) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v945 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v946 = stablehlo.multiply %v945, %cb5v : tensor<32xf32>
    %v947 = stablehlo.add %v946, %v944 : tensor<32xf32>
    %v948 = stablehlo.multiply %v945, %v947 : tensor<32xf32>
    %v949 = stablehlo.add %v948, %v944 : tensor<32xf32>
    %v950 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v951 = stablehlo.multiply %v950, %v949 : tensor<32xf32>
    %v952 = stablehlo.subtract %cb5, %v951 : tensor<32xf32>
    %v953 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v954 = stablehlo.multiply %v953, %cb5v : tensor<32xf32>
    %v955 = stablehlo.add %v954, %v944 : tensor<32xf32>
    %v956 = stablehlo.constant dense<0.0> : tensor<f32>
    %v957 = stablehlo.reshape %v128 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v958 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v959 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v960 = stablehlo.reduce(%v957 init: %v956) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v961 = stablehlo.broadcast_in_dim %v960, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v962 = stablehlo.divide %v961, %v958 : tensor<128x32x8x8xf32>
    %v963 = stablehlo.subtract %v957, %v962 : tensor<128x32x8x8xf32>
    %v964 = stablehlo.multiply %v963, %v963 : tensor<128x32x8x8xf32>
    %v965 = stablehlo.reduce(%v964 init: %v956) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v966 = stablehlo.broadcast_in_dim %v965, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v967 = stablehlo.divide %v966, %v958 : tensor<128x32x8x8xf32>
    %v968 = stablehlo.add %v967, %v959 : tensor<128x32x8x8xf32>
    %v969 = stablehlo.rsqrt %v968 : tensor<128x32x8x8xf32>
    %v970 = stablehlo.multiply %v963, %v969 : tensor<128x32x8x8xf32>
    %v971 = stablehlo.reshape %v424 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v972 = stablehlo.multiply %v971, %v970 : tensor<128x32x8x8xf32>
    %v973 = stablehlo.reduce(%v972 init: %v956) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v974 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v975 = stablehlo.multiply %v974, %g5v : tensor<32xf32>
    %v976 = stablehlo.add %v975, %v973 : tensor<32xf32>
    %v977 = stablehlo.multiply %v974, %v976 : tensor<32xf32>
    %v978 = stablehlo.add %v977, %v973 : tensor<32xf32>
    %v979 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v980 = stablehlo.multiply %v979, %v978 : tensor<32xf32>
    %v981 = stablehlo.subtract %g5, %v980 : tensor<32xf32>
    %v982 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v983 = stablehlo.multiply %v982, %g5v : tensor<32xf32>
    %v984 = stablehlo.add %v983, %v973 : tensor<32xf32>
    %v985 = stablehlo.constant dense<0.0> : tensor<f32>
    %v986 = stablehlo.reshape %v424 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v987 = stablehlo.reduce(%v986 init: %v985) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v988 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v989 = stablehlo.multiply %v988, %bt5v : tensor<32xf32>
    %v990 = stablehlo.add %v989, %v987 : tensor<32xf32>
    %v991 = stablehlo.multiply %v988, %v990 : tensor<32xf32>
    %v992 = stablehlo.add %v991, %v987 : tensor<32xf32>
    %v993 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v994 = stablehlo.multiply %v993, %v992 : tensor<32xf32>
    %v995 = stablehlo.subtract %bt5, %v994 : tensor<32xf32>
    %v996 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v997 = stablehlo.multiply %v996, %bt5v : tensor<32xf32>
    %v998 = stablehlo.add %v997, %v987 : tensor<32xf32>
    %v999 = stablehlo.reshape %v152 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1000 = stablehlo.reshape %v413 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1001 = stablehlo.transpose %v999, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v1002 = stablehlo.transpose %v1000, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v1003 = stablehlo.convolution(%v1001, %v1002)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<32x32x3x3xf32>
    %v1004 = stablehlo.transpose %v1003, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v1005 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1006 = stablehlo.multiply %v1005, %W6v : tensor<32x32x3x3xf32>
    %v1007 = stablehlo.add %v1006, %v1004 : tensor<32x32x3x3xf32>
    %v1008 = stablehlo.multiply %v1005, %v1007 : tensor<32x32x3x3xf32>
    %v1009 = stablehlo.add %v1008, %v1004 : tensor<32x32x3x3xf32>
    %v1010 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1011 = stablehlo.multiply %v1010, %v1009 : tensor<32x32x3x3xf32>
    %v1012 = stablehlo.subtract %W6, %v1011 : tensor<32x32x3x3xf32>
    %v1013 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1014 = stablehlo.multiply %v1013, %W6v : tensor<32x32x3x3xf32>
    %v1015 = stablehlo.add %v1014, %v1004 : tensor<32x32x3x3xf32>
    %v1016 = stablehlo.reshape %v413 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1017 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1018 = stablehlo.reduce(%v1016 init: %v1017) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1019 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1020 = stablehlo.multiply %v1019, %cb6v : tensor<32xf32>
    %v1021 = stablehlo.add %v1020, %v1018 : tensor<32xf32>
    %v1022 = stablehlo.multiply %v1019, %v1021 : tensor<32xf32>
    %v1023 = stablehlo.add %v1022, %v1018 : tensor<32xf32>
    %v1024 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1025 = stablehlo.multiply %v1024, %v1023 : tensor<32xf32>
    %v1026 = stablehlo.subtract %cb6, %v1025 : tensor<32xf32>
    %v1027 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1028 = stablehlo.multiply %v1027, %cb6v : tensor<32xf32>
    %v1029 = stablehlo.add %v1028, %v1018 : tensor<32xf32>
    %v1030 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1031 = stablehlo.reshape %v157 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1032 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v1033 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v1034 = stablehlo.reduce(%v1031 init: %v1030) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1035 = stablehlo.broadcast_in_dim %v1034, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v1036 = stablehlo.divide %v1035, %v1032 : tensor<128x32x8x8xf32>
    %v1037 = stablehlo.subtract %v1031, %v1036 : tensor<128x32x8x8xf32>
    %v1038 = stablehlo.multiply %v1037, %v1037 : tensor<128x32x8x8xf32>
    %v1039 = stablehlo.reduce(%v1038 init: %v1030) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1040 = stablehlo.broadcast_in_dim %v1039, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v1041 = stablehlo.divide %v1040, %v1032 : tensor<128x32x8x8xf32>
    %v1042 = stablehlo.add %v1041, %v1033 : tensor<128x32x8x8xf32>
    %v1043 = stablehlo.rsqrt %v1042 : tensor<128x32x8x8xf32>
    %v1044 = stablehlo.multiply %v1037, %v1043 : tensor<128x32x8x8xf32>
    %v1045 = stablehlo.reshape %v383 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1046 = stablehlo.multiply %v1045, %v1044 : tensor<128x32x8x8xf32>
    %v1047 = stablehlo.reduce(%v1046 init: %v1030) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1048 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1049 = stablehlo.multiply %v1048, %g6v : tensor<32xf32>
    %v1050 = stablehlo.add %v1049, %v1047 : tensor<32xf32>
    %v1051 = stablehlo.multiply %v1048, %v1050 : tensor<32xf32>
    %v1052 = stablehlo.add %v1051, %v1047 : tensor<32xf32>
    %v1053 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1054 = stablehlo.multiply %v1053, %v1052 : tensor<32xf32>
    %v1055 = stablehlo.subtract %g6, %v1054 : tensor<32xf32>
    %v1056 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1057 = stablehlo.multiply %v1056, %g6v : tensor<32xf32>
    %v1058 = stablehlo.add %v1057, %v1047 : tensor<32xf32>
    %v1059 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1060 = stablehlo.reshape %v383 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1061 = stablehlo.reduce(%v1060 init: %v1059) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1062 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1063 = stablehlo.multiply %v1062, %bt6v : tensor<32xf32>
    %v1064 = stablehlo.add %v1063, %v1061 : tensor<32xf32>
    %v1065 = stablehlo.multiply %v1062, %v1064 : tensor<32xf32>
    %v1066 = stablehlo.add %v1065, %v1061 : tensor<32xf32>
    %v1067 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1068 = stablehlo.multiply %v1067, %v1066 : tensor<32xf32>
    %v1069 = stablehlo.subtract %bt6, %v1068 : tensor<32xf32>
    %v1070 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1071 = stablehlo.multiply %v1070, %bt6v : tensor<32xf32>
    %v1072 = stablehlo.add %v1071, %v1061 : tensor<32xf32>
    %v1073 = stablehlo.reshape %v185 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1074 = stablehlo.reshape %v367 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1075 = stablehlo.transpose %v1073, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v1076 = stablehlo.transpose %v1074, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v1077 = stablehlo.convolution(%v1075, %v1076)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v1078 = stablehlo.transpose %v1077, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v1079 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1080 = stablehlo.multiply %v1079, %W7v : tensor<32x32x3x3xf32>
    %v1081 = stablehlo.add %v1080, %v1078 : tensor<32x32x3x3xf32>
    %v1082 = stablehlo.multiply %v1079, %v1081 : tensor<32x32x3x3xf32>
    %v1083 = stablehlo.add %v1082, %v1078 : tensor<32x32x3x3xf32>
    %v1084 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1085 = stablehlo.multiply %v1084, %v1083 : tensor<32x32x3x3xf32>
    %v1086 = stablehlo.subtract %W7, %v1085 : tensor<32x32x3x3xf32>
    %v1087 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1088 = stablehlo.multiply %v1087, %W7v : tensor<32x32x3x3xf32>
    %v1089 = stablehlo.add %v1088, %v1078 : tensor<32x32x3x3xf32>
    %v1090 = stablehlo.reshape %v367 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1091 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1092 = stablehlo.reduce(%v1090 init: %v1091) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1093 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1094 = stablehlo.multiply %v1093, %cb7v : tensor<32xf32>
    %v1095 = stablehlo.add %v1094, %v1092 : tensor<32xf32>
    %v1096 = stablehlo.multiply %v1093, %v1095 : tensor<32xf32>
    %v1097 = stablehlo.add %v1096, %v1092 : tensor<32xf32>
    %v1098 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1099 = stablehlo.multiply %v1098, %v1097 : tensor<32xf32>
    %v1100 = stablehlo.subtract %cb7, %v1099 : tensor<32xf32>
    %v1101 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1102 = stablehlo.multiply %v1101, %cb7v : tensor<32xf32>
    %v1103 = stablehlo.add %v1102, %v1092 : tensor<32xf32>
    %v1104 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1105 = stablehlo.reshape %v190 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1106 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v1107 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v1108 = stablehlo.reduce(%v1105 init: %v1104) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1109 = stablehlo.broadcast_in_dim %v1108, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v1110 = stablehlo.divide %v1109, %v1106 : tensor<128x32x4x4xf32>
    %v1111 = stablehlo.subtract %v1105, %v1110 : tensor<128x32x4x4xf32>
    %v1112 = stablehlo.multiply %v1111, %v1111 : tensor<128x32x4x4xf32>
    %v1113 = stablehlo.reduce(%v1112 init: %v1104) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1114 = stablehlo.broadcast_in_dim %v1113, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v1115 = stablehlo.divide %v1114, %v1106 : tensor<128x32x4x4xf32>
    %v1116 = stablehlo.add %v1115, %v1107 : tensor<128x32x4x4xf32>
    %v1117 = stablehlo.rsqrt %v1116 : tensor<128x32x4x4xf32>
    %v1118 = stablehlo.multiply %v1111, %v1117 : tensor<128x32x4x4xf32>
    %v1119 = stablehlo.reshape %v337 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1120 = stablehlo.multiply %v1119, %v1118 : tensor<128x32x4x4xf32>
    %v1121 = stablehlo.reduce(%v1120 init: %v1104) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1122 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1123 = stablehlo.multiply %v1122, %g7v : tensor<32xf32>
    %v1124 = stablehlo.add %v1123, %v1121 : tensor<32xf32>
    %v1125 = stablehlo.multiply %v1122, %v1124 : tensor<32xf32>
    %v1126 = stablehlo.add %v1125, %v1121 : tensor<32xf32>
    %v1127 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1128 = stablehlo.multiply %v1127, %v1126 : tensor<32xf32>
    %v1129 = stablehlo.subtract %g7, %v1128 : tensor<32xf32>
    %v1130 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1131 = stablehlo.multiply %v1130, %g7v : tensor<32xf32>
    %v1132 = stablehlo.add %v1131, %v1121 : tensor<32xf32>
    %v1133 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1134 = stablehlo.reshape %v337 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1135 = stablehlo.reduce(%v1134 init: %v1133) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1136 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1137 = stablehlo.multiply %v1136, %bt7v : tensor<32xf32>
    %v1138 = stablehlo.add %v1137, %v1135 : tensor<32xf32>
    %v1139 = stablehlo.multiply %v1136, %v1138 : tensor<32xf32>
    %v1140 = stablehlo.add %v1139, %v1135 : tensor<32xf32>
    %v1141 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1142 = stablehlo.multiply %v1141, %v1140 : tensor<32xf32>
    %v1143 = stablehlo.subtract %bt7, %v1142 : tensor<32xf32>
    %v1144 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1145 = stablehlo.multiply %v1144, %bt7v : tensor<32xf32>
    %v1146 = stablehlo.add %v1145, %v1135 : tensor<32xf32>
    %v1147 = stablehlo.reshape %v214 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1148 = stablehlo.reshape %v326 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1149 = stablehlo.transpose %v1147, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v1150 = stablehlo.transpose %v1148, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v1151 = stablehlo.convolution(%v1149, %v1150)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v1152 = stablehlo.transpose %v1151, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v1153 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1154 = stablehlo.multiply %v1153, %W8v : tensor<32x32x3x3xf32>
    %v1155 = stablehlo.add %v1154, %v1152 : tensor<32x32x3x3xf32>
    %v1156 = stablehlo.multiply %v1153, %v1155 : tensor<32x32x3x3xf32>
    %v1157 = stablehlo.add %v1156, %v1152 : tensor<32x32x3x3xf32>
    %v1158 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1159 = stablehlo.multiply %v1158, %v1157 : tensor<32x32x3x3xf32>
    %v1160 = stablehlo.subtract %W8, %v1159 : tensor<32x32x3x3xf32>
    %v1161 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1162 = stablehlo.multiply %v1161, %W8v : tensor<32x32x3x3xf32>
    %v1163 = stablehlo.add %v1162, %v1152 : tensor<32x32x3x3xf32>
    %v1164 = stablehlo.reshape %v326 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1165 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1166 = stablehlo.reduce(%v1164 init: %v1165) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1167 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1168 = stablehlo.multiply %v1167, %cb8v : tensor<32xf32>
    %v1169 = stablehlo.add %v1168, %v1166 : tensor<32xf32>
    %v1170 = stablehlo.multiply %v1167, %v1169 : tensor<32xf32>
    %v1171 = stablehlo.add %v1170, %v1166 : tensor<32xf32>
    %v1172 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1173 = stablehlo.multiply %v1172, %v1171 : tensor<32xf32>
    %v1174 = stablehlo.subtract %cb8, %v1173 : tensor<32xf32>
    %v1175 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1176 = stablehlo.multiply %v1175, %cb8v : tensor<32xf32>
    %v1177 = stablehlo.add %v1176, %v1166 : tensor<32xf32>
    %v1178 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1179 = stablehlo.reshape %v219 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1180 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v1181 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v1182 = stablehlo.reduce(%v1179 init: %v1178) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1183 = stablehlo.broadcast_in_dim %v1182, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v1184 = stablehlo.divide %v1183, %v1180 : tensor<128x32x4x4xf32>
    %v1185 = stablehlo.subtract %v1179, %v1184 : tensor<128x32x4x4xf32>
    %v1186 = stablehlo.multiply %v1185, %v1185 : tensor<128x32x4x4xf32>
    %v1187 = stablehlo.reduce(%v1186 init: %v1178) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1188 = stablehlo.broadcast_in_dim %v1187, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v1189 = stablehlo.divide %v1188, %v1180 : tensor<128x32x4x4xf32>
    %v1190 = stablehlo.add %v1189, %v1181 : tensor<128x32x4x4xf32>
    %v1191 = stablehlo.rsqrt %v1190 : tensor<128x32x4x4xf32>
    %v1192 = stablehlo.multiply %v1185, %v1191 : tensor<128x32x4x4xf32>
    %v1193 = stablehlo.reshape %v296 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1194 = stablehlo.multiply %v1193, %v1192 : tensor<128x32x4x4xf32>
    %v1195 = stablehlo.reduce(%v1194 init: %v1178) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1196 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1197 = stablehlo.multiply %v1196, %g8v : tensor<32xf32>
    %v1198 = stablehlo.add %v1197, %v1195 : tensor<32xf32>
    %v1199 = stablehlo.multiply %v1196, %v1198 : tensor<32xf32>
    %v1200 = stablehlo.add %v1199, %v1195 : tensor<32xf32>
    %v1201 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1202 = stablehlo.multiply %v1201, %v1200 : tensor<32xf32>
    %v1203 = stablehlo.subtract %g8, %v1202 : tensor<32xf32>
    %v1204 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1205 = stablehlo.multiply %v1204, %g8v : tensor<32xf32>
    %v1206 = stablehlo.add %v1205, %v1195 : tensor<32xf32>
    %v1207 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1208 = stablehlo.reshape %v296 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1209 = stablehlo.reduce(%v1208 init: %v1207) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1210 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1211 = stablehlo.multiply %v1210, %bt8v : tensor<32xf32>
    %v1212 = stablehlo.add %v1211, %v1209 : tensor<32xf32>
    %v1213 = stablehlo.multiply %v1210, %v1212 : tensor<32xf32>
    %v1214 = stablehlo.add %v1213, %v1209 : tensor<32xf32>
    %v1215 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1216 = stablehlo.multiply %v1215, %v1214 : tensor<32xf32>
    %v1217 = stablehlo.subtract %bt8, %v1216 : tensor<32xf32>
    %v1218 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1219 = stablehlo.multiply %v1218, %bt8v : tensor<32xf32>
    %v1220 = stablehlo.add %v1219, %v1209 : tensor<32xf32>
    %v1221 = stablehlo.dot_general %v247, %v282, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x512xf32>) -> tensor<128x512xf32>
    %v1222 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v1223 = stablehlo.multiply %v1222, %W9v : tensor<128x512xf32>
    %v1224 = stablehlo.add %v1223, %v1221 : tensor<128x512xf32>
    %v1225 = stablehlo.multiply %v1222, %v1224 : tensor<128x512xf32>
    %v1226 = stablehlo.add %v1225, %v1221 : tensor<128x512xf32>
    %v1227 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v1228 = stablehlo.multiply %v1227, %v1226 : tensor<128x512xf32>
    %v1229 = stablehlo.subtract %W9, %v1228 : tensor<128x512xf32>
    %v1230 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v1231 = stablehlo.multiply %v1230, %W9v : tensor<128x512xf32>
    %v1232 = stablehlo.add %v1231, %v1221 : tensor<128x512xf32>
    %v1233 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1234 = stablehlo.reduce(%v282 init: %v1233) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v1235 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1236 = stablehlo.multiply %v1235, %b9v : tensor<512xf32>
    %v1237 = stablehlo.add %v1236, %v1234 : tensor<512xf32>
    %v1238 = stablehlo.multiply %v1235, %v1237 : tensor<512xf32>
    %v1239 = stablehlo.add %v1238, %v1234 : tensor<512xf32>
    %v1240 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1241 = stablehlo.multiply %v1240, %v1239 : tensor<512xf32>
    %v1242 = stablehlo.subtract %b9, %v1241 : tensor<512xf32>
    %v1243 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1244 = stablehlo.multiply %v1243, %b9v : tensor<512xf32>
    %v1245 = stablehlo.add %v1244, %v1234 : tensor<512xf32>
    %v1246 = stablehlo.dot_general %v252, %v276, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<512x512xf32>
    %v1247 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1248 = stablehlo.multiply %v1247, %Wav : tensor<512x512xf32>
    %v1249 = stablehlo.add %v1248, %v1246 : tensor<512x512xf32>
    %v1250 = stablehlo.multiply %v1247, %v1249 : tensor<512x512xf32>
    %v1251 = stablehlo.add %v1250, %v1246 : tensor<512x512xf32>
    %v1252 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1253 = stablehlo.multiply %v1252, %v1251 : tensor<512x512xf32>
    %v1254 = stablehlo.subtract %Wa, %v1253 : tensor<512x512xf32>
    %v1255 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1256 = stablehlo.multiply %v1255, %Wav : tensor<512x512xf32>
    %v1257 = stablehlo.add %v1256, %v1246 : tensor<512x512xf32>
    %v1258 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1259 = stablehlo.reduce(%v276 init: %v1258) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v1260 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1261 = stablehlo.multiply %v1260, %bav : tensor<512xf32>
    %v1262 = stablehlo.add %v1261, %v1259 : tensor<512xf32>
    %v1263 = stablehlo.multiply %v1260, %v1262 : tensor<512xf32>
    %v1264 = stablehlo.add %v1263, %v1259 : tensor<512xf32>
    %v1265 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1266 = stablehlo.multiply %v1265, %v1264 : tensor<512xf32>
    %v1267 = stablehlo.subtract %ba, %v1266 : tensor<512xf32>
    %v1268 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1269 = stablehlo.multiply %v1268, %bav : tensor<512xf32>
    %v1270 = stablehlo.add %v1269, %v1259 : tensor<512xf32>
    %v1271 = stablehlo.dot_general %v257, %v270, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x10xf32>) -> tensor<512x10xf32>
    %v1272 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1273 = stablehlo.multiply %v1272, %Wbv : tensor<512x10xf32>
    %v1274 = stablehlo.add %v1273, %v1271 : tensor<512x10xf32>
    %v1275 = stablehlo.multiply %v1272, %v1274 : tensor<512x10xf32>
    %v1276 = stablehlo.add %v1275, %v1271 : tensor<512x10xf32>
    %v1277 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1278 = stablehlo.multiply %v1277, %v1276 : tensor<512x10xf32>
    %v1279 = stablehlo.subtract %Wb, %v1278 : tensor<512x10xf32>
    %v1280 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1281 = stablehlo.multiply %v1280, %Wbv : tensor<512x10xf32>
    %v1282 = stablehlo.add %v1281, %v1271 : tensor<512x10xf32>
    %v1283 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1284 = stablehlo.reduce(%v270 init: %v1283) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1285 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1286 = stablehlo.multiply %v1285, %bbv : tensor<10xf32>
    %v1287 = stablehlo.add %v1286, %v1284 : tensor<10xf32>
    %v1288 = stablehlo.multiply %v1285, %v1287 : tensor<10xf32>
    %v1289 = stablehlo.add %v1288, %v1284 : tensor<10xf32>
    %v1290 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1291 = stablehlo.multiply %v1290, %v1289 : tensor<10xf32>
    %v1292 = stablehlo.subtract %bb, %v1291 : tensor<10xf32>
    %v1293 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1294 = stablehlo.multiply %v1293, %bbv : tensor<10xf32>
    %v1295 = stablehlo.add %v1294, %v1284 : tensor<10xf32>
    return %v642, %v656, %v685, %v699, %v716, %v730, %v759, %v773, %v790, %v804, %v833, %v847, %v864, %v878, %v907, %v921, %v938, %v952, %v981, %v995, %v1012, %v1026, %v1055, %v1069, %v1086, %v1100, %v1129, %v1143, %v1160, %v1174, %v1203, %v1217, %v1229, %v1242, %v1254, %v1267, %v1279, %v1292, %W1m, %cb1m, %g1m, %bt1m, %W2m, %cb2m, %g2m, %bt2m, %W3m, %cb3m, %g3m, %bt3m, %W4m, %cb4m, %g4m, %bt4m, %W5m, %cb5m, %g5m, %bt5m, %W6m, %cb6m, %g6m, %bt6m, %W7m, %cb7m, %g7m, %bt7m, %W8m, %cb8m, %g8m, %bt8m, %W9m, %b9m, %Wam, %bam, %Wbm, %bbm, %v645, %v659, %v688, %v702, %v719, %v733, %v762, %v776, %v793, %v807, %v836, %v850, %v867, %v881, %v910, %v924, %v941, %v955, %v984, %v998, %v1015, %v1029, %v1058, %v1072, %v1089, %v1103, %v1132, %v1146, %v1163, %v1177, %v1206, %v1220, %v1232, %v1245, %v1257, %v1270, %v1282, %v1295, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
