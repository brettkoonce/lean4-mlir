module @m {
  func.func @cifar8_bn_sgd_train_step(%x: tensor<128x3072xf32>, %W1: tensor<16x3x3x3xf32>, %cb1: tensor<16xf32>, %g1: tensor<16xf32>, %bt1: tensor<16xf32>, %W2: tensor<16x16x3x3xf32>, %cb2: tensor<16xf32>, %g2: tensor<16xf32>, %bt2: tensor<16xf32>, %W3: tensor<16x16x3x3xf32>, %cb3: tensor<16xf32>, %g3: tensor<16xf32>, %bt3: tensor<16xf32>, %W4: tensor<16x16x3x3xf32>, %cb4: tensor<16xf32>, %g4: tensor<16xf32>, %bt4: tensor<16xf32>, %W5: tensor<32x16x3x3xf32>, %cb5: tensor<32xf32>, %g5: tensor<32xf32>, %bt5: tensor<32xf32>, %W6: tensor<32x32x3x3xf32>, %cb6: tensor<32xf32>, %g6: tensor<32xf32>, %bt6: tensor<32xf32>, %W7: tensor<32x32x3x3xf32>, %cb7: tensor<32xf32>, %g7: tensor<32xf32>, %bt7: tensor<32xf32>, %W8: tensor<32x32x3x3xf32>, %cb8: tensor<32xf32>, %g8: tensor<32xf32>, %bt8: tensor<32xf32>, %W9: tensor<128x64xf32>, %b9: tensor<64xf32>, %Wa: tensor<64x64xf32>, %ba: tensor<64xf32>, %Wb: tensor<64x10xf32>, %bb: tensor<10xf32>, %W1m: tensor<16x3x3x3xf32>, %cb1m: tensor<16xf32>, %g1m: tensor<16xf32>, %bt1m: tensor<16xf32>, %W2m: tensor<16x16x3x3xf32>, %cb2m: tensor<16xf32>, %g2m: tensor<16xf32>, %bt2m: tensor<16xf32>, %W3m: tensor<16x16x3x3xf32>, %cb3m: tensor<16xf32>, %g3m: tensor<16xf32>, %bt3m: tensor<16xf32>, %W4m: tensor<16x16x3x3xf32>, %cb4m: tensor<16xf32>, %g4m: tensor<16xf32>, %bt4m: tensor<16xf32>, %W5m: tensor<32x16x3x3xf32>, %cb5m: tensor<32xf32>, %g5m: tensor<32xf32>, %bt5m: tensor<32xf32>, %W6m: tensor<32x32x3x3xf32>, %cb6m: tensor<32xf32>, %g6m: tensor<32xf32>, %bt6m: tensor<32xf32>, %W7m: tensor<32x32x3x3xf32>, %cb7m: tensor<32xf32>, %g7m: tensor<32xf32>, %bt7m: tensor<32xf32>, %W8m: tensor<32x32x3x3xf32>, %cb8m: tensor<32xf32>, %g8m: tensor<32xf32>, %bt8m: tensor<32xf32>, %W9m: tensor<128x64xf32>, %b9m: tensor<64xf32>, %Wam: tensor<64x64xf32>, %bam: tensor<64xf32>, %Wbm: tensor<64x10xf32>, %bbm: tensor<10xf32>, %W1v: tensor<16x3x3x3xf32>, %cb1v: tensor<16xf32>, %g1v: tensor<16xf32>, %bt1v: tensor<16xf32>, %W2v: tensor<16x16x3x3xf32>, %cb2v: tensor<16xf32>, %g2v: tensor<16xf32>, %bt2v: tensor<16xf32>, %W3v: tensor<16x16x3x3xf32>, %cb3v: tensor<16xf32>, %g3v: tensor<16xf32>, %bt3v: tensor<16xf32>, %W4v: tensor<16x16x3x3xf32>, %cb4v: tensor<16xf32>, %g4v: tensor<16xf32>, %bt4v: tensor<16xf32>, %W5v: tensor<32x16x3x3xf32>, %cb5v: tensor<32xf32>, %g5v: tensor<32xf32>, %bt5v: tensor<32xf32>, %W6v: tensor<32x32x3x3xf32>, %cb6v: tensor<32xf32>, %g6v: tensor<32xf32>, %bt6v: tensor<32xf32>, %W7v: tensor<32x32x3x3xf32>, %cb7v: tensor<32xf32>, %g7v: tensor<32xf32>, %bt7v: tensor<32xf32>, %W8v: tensor<32x32x3x3xf32>, %cb8v: tensor<32xf32>, %g8v: tensor<32xf32>, %bt8v: tensor<32xf32>, %W9v: tensor<128x64xf32>, %b9v: tensor<64xf32>, %Wav: tensor<64x64xf32>, %bav: tensor<64xf32>, %Wbv: tensor<64x10xf32>, %bbv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: tensor<128x10xf32>) -> (tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>) {
    // ── cifar8-bn train step: every line is pretty(verified AST node), except the
    //    marked report-only loss + the %bc passthroughs ──
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
    %v248 = stablehlo.dot_general %v247, %W9, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %v249 = stablehlo.broadcast_in_dim %b9, dims = [1] : (tensor<64xf32>) -> tensor<128x64xf32>
    %v250 = stablehlo.add %v248, %v249 : tensor<128x64xf32>
    %v251 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v252 = stablehlo.maximum %v250, %v251 : tensor<128x64xf32>
    %v253 = stablehlo.dot_general %v252, %Wa, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
    %v254 = stablehlo.broadcast_in_dim %ba, dims = [1] : (tensor<64xf32>) -> tensor<128x64xf32>
    %v255 = stablehlo.add %v253, %v254 : tensor<128x64xf32>
    %v256 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v257 = stablehlo.maximum %v255, %v256 : tensor<128x64xf32>
    %v258 = stablehlo.dot_general %v257, %Wb, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x10xf32>) -> tensor<128x10xf32>
    %v259 = stablehlo.broadcast_in_dim %bb, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v260 = stablehlo.add %v258, %v259 : tensor<128x10xf32>
    %v261 = stablehlo.exponential %v260 : tensor<128x10xf32>
    %v262 = stablehlo.constant dense<0.0> : tensor<f32>
    %v263 = stablehlo.reduce(%v261 init: %v262) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %v264 = stablehlo.broadcast_in_dim %v263, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %v265 = stablehlo.divide %v261, %v264 : tensor<128x10xf32>
    %v266 = stablehlo.subtract %v265, %onehot : tensor<128x10xf32>
    %v267 = stablehlo.constant dense<0.0078125> : tensor<128x10xf32>
    %v268 = stablehlo.multiply %v266, %v267 : tensor<128x10xf32>
    // ── report-only scalar loss (NOT pretty(AST): no rank-0 loss op; feeds no
    //    parameter, only the driver's progress line) ──
    %llog = stablehlo.log %v265 : tensor<128x10xf32>
    %ohll = stablehlo.multiply %onehot, %llog : tensor<128x10xf32>
    %csum = stablehlo.reduce(%ohll init: %lzero) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<f32>
    %cneg = stablehlo.negate %csum : tensor<f32>
    %lbf = stablehlo.constant dense<128.0> : tensor<f32>
    %loss = stablehlo.divide %cneg, %lbf : tensor<f32>
    %v269 = stablehlo.dot_general %v268, %Wb, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x10xf32>, tensor<64x10xf32>) -> tensor<128x64xf32>
    %v270 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v271 = stablehlo.compare GT, %v255, %v270 : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xi1>
    %v272 = stablehlo.select %v271, %v269, %v270 : tensor<128x64xi1>, tensor<128x64xf32>
    %v273 = stablehlo.dot_general %v272, %Wa, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
    %v274 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v275 = stablehlo.compare GT, %v250, %v274 : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xi1>
    %v276 = stablehlo.select %v275, %v273, %v274 : tensor<128x64xi1>, tensor<128x64xf32>
    %v277 = stablehlo.dot_general %v276, %W9, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x128xf32>
    %v278 = stablehlo.reshape %v243 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v279 = stablehlo.reshape %v277 : (tensor<128x128xf32>) -> tensor<128x32x2x2xf32>
    %v280 = stablehlo.constant dense<0.0> : tensor<f32>
    %v281 = "stablehlo.select_and_scatter"(%v278, %v279, %v280) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x4x4xf32>, tensor<128x32x2x2xf32>, tensor<f32>) -> tensor<128x32x4x4xf32>
    %v282 = stablehlo.reshape %v281 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v283 = stablehlo.reshape %v282 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v284 = stablehlo.reshape %v239 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v285 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v286 = stablehlo.compare GT, %v284, %v285 : (tensor<128x32x4x4xf32>, tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xi1>
    %v287 = stablehlo.select %v286, %v283, %v285 : tensor<128x32x4x4xi1>, tensor<128x32x4x4xf32>
    %v288 = stablehlo.reshape %v287 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v289 = stablehlo.reshape %v288 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v290 = stablehlo.reshape %v219 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v291 = stablehlo.constant dense<0.0> : tensor<f32>
    %v292 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v293 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v294 = stablehlo.reduce(%v290 init: %v291) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v295 = stablehlo.broadcast_in_dim %v294, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v296 = stablehlo.divide %v295, %v292 : tensor<128x32x4x4xf32>
    %v297 = stablehlo.subtract %v290, %v296 : tensor<128x32x4x4xf32>
    %v298 = stablehlo.multiply %v297, %v297 : tensor<128x32x4x4xf32>
    %v299 = stablehlo.reduce(%v298 init: %v291) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v300 = stablehlo.broadcast_in_dim %v299, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v301 = stablehlo.divide %v300, %v292 : tensor<128x32x4x4xf32>
    %v302 = stablehlo.add %v301, %v293 : tensor<128x32x4x4xf32>
    %v303 = stablehlo.rsqrt %v302 : tensor<128x32x4x4xf32>
    %v304 = stablehlo.multiply %v297, %v303 : tensor<128x32x4x4xf32>
    %v305 = stablehlo.broadcast_in_dim %g8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v306 = stablehlo.multiply %v305, %v289 : tensor<128x32x4x4xf32>
    %v307 = stablehlo.reduce(%v306 init: %v291) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v308 = stablehlo.broadcast_in_dim %v307, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v309 = stablehlo.multiply %v304, %v306 : tensor<128x32x4x4xf32>
    %v310 = stablehlo.reduce(%v309 init: %v291) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v311 = stablehlo.broadcast_in_dim %v310, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v312 = stablehlo.multiply %v306, %v292 : tensor<128x32x4x4xf32>
    %v313 = stablehlo.subtract %v312, %v308 : tensor<128x32x4x4xf32>
    %v314 = stablehlo.multiply %v304, %v311 : tensor<128x32x4x4xf32>
    %v315 = stablehlo.subtract %v313, %v314 : tensor<128x32x4x4xf32>
    %v316 = stablehlo.divide %v303, %v292 : tensor<128x32x4x4xf32>
    %v317 = stablehlo.multiply %v316, %v315 : tensor<128x32x4x4xf32>
    %v318 = stablehlo.reshape %v317 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v319 = stablehlo.reshape %v318 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v320 = stablehlo.transpose %W8, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v321 = stablehlo.reverse %v320, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v322 = stablehlo.convolution(%v319, %v321)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v323 = stablehlo.reshape %v322 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v324 = stablehlo.reshape %v323 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v325 = stablehlo.reshape %v210 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v326 = stablehlo.constant dense<0.0> : tensor<128x32x4x4xf32>
    %v327 = stablehlo.compare GT, %v325, %v326 : (tensor<128x32x4x4xf32>, tensor<128x32x4x4xf32>) -> tensor<128x32x4x4xi1>
    %v328 = stablehlo.select %v327, %v324, %v326 : tensor<128x32x4x4xi1>, tensor<128x32x4x4xf32>
    %v329 = stablehlo.reshape %v328 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v330 = stablehlo.reshape %v329 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v331 = stablehlo.reshape %v190 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v332 = stablehlo.constant dense<0.0> : tensor<f32>
    %v333 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v334 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v335 = stablehlo.reduce(%v331 init: %v332) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v336 = stablehlo.broadcast_in_dim %v335, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v337 = stablehlo.divide %v336, %v333 : tensor<128x32x4x4xf32>
    %v338 = stablehlo.subtract %v331, %v337 : tensor<128x32x4x4xf32>
    %v339 = stablehlo.multiply %v338, %v338 : tensor<128x32x4x4xf32>
    %v340 = stablehlo.reduce(%v339 init: %v332) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v341 = stablehlo.broadcast_in_dim %v340, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v342 = stablehlo.divide %v341, %v333 : tensor<128x32x4x4xf32>
    %v343 = stablehlo.add %v342, %v334 : tensor<128x32x4x4xf32>
    %v344 = stablehlo.rsqrt %v343 : tensor<128x32x4x4xf32>
    %v345 = stablehlo.multiply %v338, %v344 : tensor<128x32x4x4xf32>
    %v346 = stablehlo.broadcast_in_dim %g7, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v347 = stablehlo.multiply %v346, %v330 : tensor<128x32x4x4xf32>
    %v348 = stablehlo.reduce(%v347 init: %v332) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v349 = stablehlo.broadcast_in_dim %v348, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v350 = stablehlo.multiply %v345, %v347 : tensor<128x32x4x4xf32>
    %v351 = stablehlo.reduce(%v350 init: %v332) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v352 = stablehlo.broadcast_in_dim %v351, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v353 = stablehlo.multiply %v347, %v333 : tensor<128x32x4x4xf32>
    %v354 = stablehlo.subtract %v353, %v349 : tensor<128x32x4x4xf32>
    %v355 = stablehlo.multiply %v345, %v352 : tensor<128x32x4x4xf32>
    %v356 = stablehlo.subtract %v354, %v355 : tensor<128x32x4x4xf32>
    %v357 = stablehlo.divide %v344, %v333 : tensor<128x32x4x4xf32>
    %v358 = stablehlo.multiply %v357, %v356 : tensor<128x32x4x4xf32>
    %v359 = stablehlo.reshape %v358 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v360 = stablehlo.reshape %v359 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v361 = stablehlo.transpose %W7, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v362 = stablehlo.reverse %v361, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v363 = stablehlo.convolution(%v360, %v362)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v364 = stablehlo.reshape %v363 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v365 = stablehlo.reshape %v181 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v366 = stablehlo.reshape %v364 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v367 = stablehlo.constant dense<0.0> : tensor<f32>
    %v368 = "stablehlo.select_and_scatter"(%v365, %v366, %v367) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x8x8xf32>, tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32x8x8xf32>
    %v369 = stablehlo.reshape %v368 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v370 = stablehlo.reshape %v369 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v371 = stablehlo.reshape %v177 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v372 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v373 = stablehlo.compare GT, %v371, %v372 : (tensor<128x32x8x8xf32>, tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xi1>
    %v374 = stablehlo.select %v373, %v370, %v372 : tensor<128x32x8x8xi1>, tensor<128x32x8x8xf32>
    %v375 = stablehlo.reshape %v374 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v376 = stablehlo.reshape %v375 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v377 = stablehlo.reshape %v157 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v378 = stablehlo.constant dense<0.0> : tensor<f32>
    %v379 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v380 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v381 = stablehlo.reduce(%v377 init: %v378) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v382 = stablehlo.broadcast_in_dim %v381, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v383 = stablehlo.divide %v382, %v379 : tensor<128x32x8x8xf32>
    %v384 = stablehlo.subtract %v377, %v383 : tensor<128x32x8x8xf32>
    %v385 = stablehlo.multiply %v384, %v384 : tensor<128x32x8x8xf32>
    %v386 = stablehlo.reduce(%v385 init: %v378) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v387 = stablehlo.broadcast_in_dim %v386, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v388 = stablehlo.divide %v387, %v379 : tensor<128x32x8x8xf32>
    %v389 = stablehlo.add %v388, %v380 : tensor<128x32x8x8xf32>
    %v390 = stablehlo.rsqrt %v389 : tensor<128x32x8x8xf32>
    %v391 = stablehlo.multiply %v384, %v390 : tensor<128x32x8x8xf32>
    %v392 = stablehlo.broadcast_in_dim %g6, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v393 = stablehlo.multiply %v392, %v376 : tensor<128x32x8x8xf32>
    %v394 = stablehlo.reduce(%v393 init: %v378) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v395 = stablehlo.broadcast_in_dim %v394, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v396 = stablehlo.multiply %v391, %v393 : tensor<128x32x8x8xf32>
    %v397 = stablehlo.reduce(%v396 init: %v378) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v398 = stablehlo.broadcast_in_dim %v397, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v399 = stablehlo.multiply %v393, %v379 : tensor<128x32x8x8xf32>
    %v400 = stablehlo.subtract %v399, %v395 : tensor<128x32x8x8xf32>
    %v401 = stablehlo.multiply %v391, %v398 : tensor<128x32x8x8xf32>
    %v402 = stablehlo.subtract %v400, %v401 : tensor<128x32x8x8xf32>
    %v403 = stablehlo.divide %v390, %v379 : tensor<128x32x8x8xf32>
    %v404 = stablehlo.multiply %v403, %v402 : tensor<128x32x8x8xf32>
    %v405 = stablehlo.reshape %v404 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v406 = stablehlo.reshape %v405 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v407 = stablehlo.transpose %W6, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v408 = stablehlo.reverse %v407, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v409 = stablehlo.convolution(%v406, %v408)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x8x8xf32>
    %v410 = stablehlo.reshape %v409 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v411 = stablehlo.reshape %v410 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v412 = stablehlo.reshape %v148 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v413 = stablehlo.constant dense<0.0> : tensor<128x32x8x8xf32>
    %v414 = stablehlo.compare GT, %v412, %v413 : (tensor<128x32x8x8xf32>, tensor<128x32x8x8xf32>) -> tensor<128x32x8x8xi1>
    %v415 = stablehlo.select %v414, %v411, %v413 : tensor<128x32x8x8xi1>, tensor<128x32x8x8xf32>
    %v416 = stablehlo.reshape %v415 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v417 = stablehlo.reshape %v416 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v418 = stablehlo.reshape %v128 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v419 = stablehlo.constant dense<0.0> : tensor<f32>
    %v420 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v421 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v422 = stablehlo.reduce(%v418 init: %v419) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v423 = stablehlo.broadcast_in_dim %v422, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v424 = stablehlo.divide %v423, %v420 : tensor<128x32x8x8xf32>
    %v425 = stablehlo.subtract %v418, %v424 : tensor<128x32x8x8xf32>
    %v426 = stablehlo.multiply %v425, %v425 : tensor<128x32x8x8xf32>
    %v427 = stablehlo.reduce(%v426 init: %v419) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v428 = stablehlo.broadcast_in_dim %v427, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v429 = stablehlo.divide %v428, %v420 : tensor<128x32x8x8xf32>
    %v430 = stablehlo.add %v429, %v421 : tensor<128x32x8x8xf32>
    %v431 = stablehlo.rsqrt %v430 : tensor<128x32x8x8xf32>
    %v432 = stablehlo.multiply %v425, %v431 : tensor<128x32x8x8xf32>
    %v433 = stablehlo.broadcast_in_dim %g5, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v434 = stablehlo.multiply %v433, %v417 : tensor<128x32x8x8xf32>
    %v435 = stablehlo.reduce(%v434 init: %v419) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v436 = stablehlo.broadcast_in_dim %v435, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v437 = stablehlo.multiply %v432, %v434 : tensor<128x32x8x8xf32>
    %v438 = stablehlo.reduce(%v437 init: %v419) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v439 = stablehlo.broadcast_in_dim %v438, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v440 = stablehlo.multiply %v434, %v420 : tensor<128x32x8x8xf32>
    %v441 = stablehlo.subtract %v440, %v436 : tensor<128x32x8x8xf32>
    %v442 = stablehlo.multiply %v432, %v439 : tensor<128x32x8x8xf32>
    %v443 = stablehlo.subtract %v441, %v442 : tensor<128x32x8x8xf32>
    %v444 = stablehlo.divide %v431, %v420 : tensor<128x32x8x8xf32>
    %v445 = stablehlo.multiply %v444, %v443 : tensor<128x32x8x8xf32>
    %v446 = stablehlo.reshape %v445 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v447 = stablehlo.reshape %v446 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v448 = stablehlo.transpose %W5, dims = [1, 0, 2, 3] : (tensor<32x16x3x3xf32>) -> tensor<16x32x3x3xf32>
    %v449 = stablehlo.reverse %v448, dims = [2, 3] : tensor<16x32x3x3xf32>
    %v450 = stablehlo.convolution(%v447, %v449)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<16x32x3x3xf32>) -> tensor<128x16x8x8xf32>
    %v451 = stablehlo.reshape %v450 : (tensor<128x16x8x8xf32>) -> tensor<128x1024xf32>
    %v452 = stablehlo.reshape %v119 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v453 = stablehlo.reshape %v451 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v454 = stablehlo.constant dense<0.0> : tensor<f32>
    %v455 = "stablehlo.select_and_scatter"(%v452, %v453, %v454) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x16x16xf32>, tensor<128x16x8x8xf32>, tensor<f32>) -> tensor<128x16x16x16xf32>
    %v456 = stablehlo.reshape %v455 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v457 = stablehlo.reshape %v456 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v458 = stablehlo.reshape %v115 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v459 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v460 = stablehlo.compare GT, %v458, %v459 : (tensor<128x16x16x16xf32>, tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xi1>
    %v461 = stablehlo.select %v460, %v457, %v459 : tensor<128x16x16x16xi1>, tensor<128x16x16x16xf32>
    %v462 = stablehlo.reshape %v461 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v463 = stablehlo.reshape %v462 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v464 = stablehlo.reshape %v95 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v465 = stablehlo.constant dense<0.0> : tensor<f32>
    %v466 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v467 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v468 = stablehlo.reduce(%v464 init: %v465) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v469 = stablehlo.broadcast_in_dim %v468, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v470 = stablehlo.divide %v469, %v466 : tensor<128x16x16x16xf32>
    %v471 = stablehlo.subtract %v464, %v470 : tensor<128x16x16x16xf32>
    %v472 = stablehlo.multiply %v471, %v471 : tensor<128x16x16x16xf32>
    %v473 = stablehlo.reduce(%v472 init: %v465) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v474 = stablehlo.broadcast_in_dim %v473, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v475 = stablehlo.divide %v474, %v466 : tensor<128x16x16x16xf32>
    %v476 = stablehlo.add %v475, %v467 : tensor<128x16x16x16xf32>
    %v477 = stablehlo.rsqrt %v476 : tensor<128x16x16x16xf32>
    %v478 = stablehlo.multiply %v471, %v477 : tensor<128x16x16x16xf32>
    %v479 = stablehlo.broadcast_in_dim %g4, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v480 = stablehlo.multiply %v479, %v463 : tensor<128x16x16x16xf32>
    %v481 = stablehlo.reduce(%v480 init: %v465) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v482 = stablehlo.broadcast_in_dim %v481, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v483 = stablehlo.multiply %v478, %v480 : tensor<128x16x16x16xf32>
    %v484 = stablehlo.reduce(%v483 init: %v465) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v485 = stablehlo.broadcast_in_dim %v484, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v486 = stablehlo.multiply %v480, %v466 : tensor<128x16x16x16xf32>
    %v487 = stablehlo.subtract %v486, %v482 : tensor<128x16x16x16xf32>
    %v488 = stablehlo.multiply %v478, %v485 : tensor<128x16x16x16xf32>
    %v489 = stablehlo.subtract %v487, %v488 : tensor<128x16x16x16xf32>
    %v490 = stablehlo.divide %v477, %v466 : tensor<128x16x16x16xf32>
    %v491 = stablehlo.multiply %v490, %v489 : tensor<128x16x16x16xf32>
    %v492 = stablehlo.reshape %v491 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v493 = stablehlo.reshape %v492 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v494 = stablehlo.transpose %W4, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v495 = stablehlo.reverse %v494, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v496 = stablehlo.convolution(%v493, %v495)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v497 = stablehlo.reshape %v496 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v498 = stablehlo.reshape %v497 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v499 = stablehlo.reshape %v86 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v500 = stablehlo.constant dense<0.0> : tensor<128x16x16x16xf32>
    %v501 = stablehlo.compare GT, %v499, %v500 : (tensor<128x16x16x16xf32>, tensor<128x16x16x16xf32>) -> tensor<128x16x16x16xi1>
    %v502 = stablehlo.select %v501, %v498, %v500 : tensor<128x16x16x16xi1>, tensor<128x16x16x16xf32>
    %v503 = stablehlo.reshape %v502 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v504 = stablehlo.reshape %v503 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v505 = stablehlo.reshape %v66 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v506 = stablehlo.constant dense<0.0> : tensor<f32>
    %v507 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v508 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v509 = stablehlo.reduce(%v505 init: %v506) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v510 = stablehlo.broadcast_in_dim %v509, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v511 = stablehlo.divide %v510, %v507 : tensor<128x16x16x16xf32>
    %v512 = stablehlo.subtract %v505, %v511 : tensor<128x16x16x16xf32>
    %v513 = stablehlo.multiply %v512, %v512 : tensor<128x16x16x16xf32>
    %v514 = stablehlo.reduce(%v513 init: %v506) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v515 = stablehlo.broadcast_in_dim %v514, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v516 = stablehlo.divide %v515, %v507 : tensor<128x16x16x16xf32>
    %v517 = stablehlo.add %v516, %v508 : tensor<128x16x16x16xf32>
    %v518 = stablehlo.rsqrt %v517 : tensor<128x16x16x16xf32>
    %v519 = stablehlo.multiply %v512, %v518 : tensor<128x16x16x16xf32>
    %v520 = stablehlo.broadcast_in_dim %g3, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v521 = stablehlo.multiply %v520, %v504 : tensor<128x16x16x16xf32>
    %v522 = stablehlo.reduce(%v521 init: %v506) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v523 = stablehlo.broadcast_in_dim %v522, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v524 = stablehlo.multiply %v519, %v521 : tensor<128x16x16x16xf32>
    %v525 = stablehlo.reduce(%v524 init: %v506) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v526 = stablehlo.broadcast_in_dim %v525, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v527 = stablehlo.multiply %v521, %v507 : tensor<128x16x16x16xf32>
    %v528 = stablehlo.subtract %v527, %v523 : tensor<128x16x16x16xf32>
    %v529 = stablehlo.multiply %v519, %v526 : tensor<128x16x16x16xf32>
    %v530 = stablehlo.subtract %v528, %v529 : tensor<128x16x16x16xf32>
    %v531 = stablehlo.divide %v518, %v507 : tensor<128x16x16x16xf32>
    %v532 = stablehlo.multiply %v531, %v530 : tensor<128x16x16x16xf32>
    %v533 = stablehlo.reshape %v532 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v534 = stablehlo.reshape %v533 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v535 = stablehlo.transpose %W3, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v536 = stablehlo.reverse %v535, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v537 = stablehlo.convolution(%v534, %v536)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v538 = stablehlo.reshape %v537 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v539 = stablehlo.reshape %v57 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v540 = stablehlo.reshape %v538 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v541 = stablehlo.constant dense<0.0> : tensor<f32>
    %v542 = "stablehlo.select_and_scatter"(%v539, %v540, %v541) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x32x32xf32>, tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x32x32xf32>
    %v543 = stablehlo.reshape %v542 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v544 = stablehlo.reshape %v543 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v545 = stablehlo.reshape %v53 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v546 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v547 = stablehlo.compare GT, %v545, %v546 : (tensor<128x16x32x32xf32>, tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xi1>
    %v548 = stablehlo.select %v547, %v544, %v546 : tensor<128x16x32x32xi1>, tensor<128x16x32x32xf32>
    %v549 = stablehlo.reshape %v548 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v550 = stablehlo.reshape %v549 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v551 = stablehlo.reshape %v33 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v552 = stablehlo.constant dense<0.0> : tensor<f32>
    %v553 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v554 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v555 = stablehlo.reduce(%v551 init: %v552) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v556 = stablehlo.broadcast_in_dim %v555, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v557 = stablehlo.divide %v556, %v553 : tensor<128x16x32x32xf32>
    %v558 = stablehlo.subtract %v551, %v557 : tensor<128x16x32x32xf32>
    %v559 = stablehlo.multiply %v558, %v558 : tensor<128x16x32x32xf32>
    %v560 = stablehlo.reduce(%v559 init: %v552) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v561 = stablehlo.broadcast_in_dim %v560, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v562 = stablehlo.divide %v561, %v553 : tensor<128x16x32x32xf32>
    %v563 = stablehlo.add %v562, %v554 : tensor<128x16x32x32xf32>
    %v564 = stablehlo.rsqrt %v563 : tensor<128x16x32x32xf32>
    %v565 = stablehlo.multiply %v558, %v564 : tensor<128x16x32x32xf32>
    %v566 = stablehlo.broadcast_in_dim %g2, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v567 = stablehlo.multiply %v566, %v550 : tensor<128x16x32x32xf32>
    %v568 = stablehlo.reduce(%v567 init: %v552) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v569 = stablehlo.broadcast_in_dim %v568, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v570 = stablehlo.multiply %v565, %v567 : tensor<128x16x32x32xf32>
    %v571 = stablehlo.reduce(%v570 init: %v552) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v572 = stablehlo.broadcast_in_dim %v571, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v573 = stablehlo.multiply %v567, %v553 : tensor<128x16x32x32xf32>
    %v574 = stablehlo.subtract %v573, %v569 : tensor<128x16x32x32xf32>
    %v575 = stablehlo.multiply %v565, %v572 : tensor<128x16x32x32xf32>
    %v576 = stablehlo.subtract %v574, %v575 : tensor<128x16x32x32xf32>
    %v577 = stablehlo.divide %v564, %v553 : tensor<128x16x32x32xf32>
    %v578 = stablehlo.multiply %v577, %v576 : tensor<128x16x32x32xf32>
    %v579 = stablehlo.reshape %v578 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v580 = stablehlo.reshape %v579 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v581 = stablehlo.transpose %W2, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v582 = stablehlo.reverse %v581, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v583 = stablehlo.convolution(%v580, %v582)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x32x32xf32>
    %v584 = stablehlo.reshape %v583 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v585 = stablehlo.reshape %v584 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v586 = stablehlo.reshape %v24 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v587 = stablehlo.constant dense<0.0> : tensor<128x16x32x32xf32>
    %v588 = stablehlo.compare GT, %v586, %v587 : (tensor<128x16x32x32xf32>, tensor<128x16x32x32xf32>) -> tensor<128x16x32x32xi1>
    %v589 = stablehlo.select %v588, %v585, %v587 : tensor<128x16x32x32xi1>, tensor<128x16x32x32xf32>
    %v590 = stablehlo.reshape %v589 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v591 = stablehlo.reshape %v590 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v592 = stablehlo.reshape %v4 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v593 = stablehlo.constant dense<0.0> : tensor<f32>
    %v594 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v595 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v596 = stablehlo.reduce(%v592 init: %v593) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v597 = stablehlo.broadcast_in_dim %v596, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v598 = stablehlo.divide %v597, %v594 : tensor<128x16x32x32xf32>
    %v599 = stablehlo.subtract %v592, %v598 : tensor<128x16x32x32xf32>
    %v600 = stablehlo.multiply %v599, %v599 : tensor<128x16x32x32xf32>
    %v601 = stablehlo.reduce(%v600 init: %v593) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v602 = stablehlo.broadcast_in_dim %v601, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v603 = stablehlo.divide %v602, %v594 : tensor<128x16x32x32xf32>
    %v604 = stablehlo.add %v603, %v595 : tensor<128x16x32x32xf32>
    %v605 = stablehlo.rsqrt %v604 : tensor<128x16x32x32xf32>
    %v606 = stablehlo.multiply %v599, %v605 : tensor<128x16x32x32xf32>
    %v607 = stablehlo.broadcast_in_dim %g1, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v608 = stablehlo.multiply %v607, %v591 : tensor<128x16x32x32xf32>
    %v609 = stablehlo.reduce(%v608 init: %v593) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v610 = stablehlo.broadcast_in_dim %v609, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v611 = stablehlo.multiply %v606, %v608 : tensor<128x16x32x32xf32>
    %v612 = stablehlo.reduce(%v611 init: %v593) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v613 = stablehlo.broadcast_in_dim %v612, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v614 = stablehlo.multiply %v608, %v594 : tensor<128x16x32x32xf32>
    %v615 = stablehlo.subtract %v614, %v610 : tensor<128x16x32x32xf32>
    %v616 = stablehlo.multiply %v606, %v613 : tensor<128x16x32x32xf32>
    %v617 = stablehlo.subtract %v615, %v616 : tensor<128x16x32x32xf32>
    %v618 = stablehlo.divide %v605, %v594 : tensor<128x16x32x32xf32>
    %v619 = stablehlo.multiply %v618, %v617 : tensor<128x16x32x32xf32>
    %v620 = stablehlo.reshape %v619 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v621 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v622 = stablehlo.reshape %v620 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v623 = stablehlo.transpose %v621, dims = [1, 0, 2, 3] : (tensor<128x3x32x32xf32>) -> tensor<3x128x32x32xf32>
    %v624 = stablehlo.transpose %v622, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v625 = stablehlo.convolution(%v623, %v624)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<3x16x3x3xf32>
    %v626 = stablehlo.transpose %v625, dims = [1, 0, 2, 3] : (tensor<3x16x3x3xf32>) -> tensor<16x3x3x3xf32>
    %v627 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v628 = stablehlo.multiply %v627, %v626 : tensor<16x3x3x3xf32>
    %v629 = stablehlo.subtract %W1, %v628 : tensor<16x3x3x3xf32>
    %v630 = stablehlo.reshape %v620 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v631 = stablehlo.constant dense<0.0> : tensor<f32>
    %v632 = stablehlo.reduce(%v630 init: %v631) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v633 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v634 = stablehlo.multiply %v633, %v632 : tensor<16xf32>
    %v635 = stablehlo.subtract %cb1, %v634 : tensor<16xf32>
    %v636 = stablehlo.constant dense<0.0> : tensor<f32>
    %v637 = stablehlo.reshape %v4 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v638 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v639 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v640 = stablehlo.reduce(%v637 init: %v636) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v641 = stablehlo.broadcast_in_dim %v640, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v642 = stablehlo.divide %v641, %v638 : tensor<128x16x32x32xf32>
    %v643 = stablehlo.subtract %v637, %v642 : tensor<128x16x32x32xf32>
    %v644 = stablehlo.multiply %v643, %v643 : tensor<128x16x32x32xf32>
    %v645 = stablehlo.reduce(%v644 init: %v636) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v646 = stablehlo.broadcast_in_dim %v645, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v647 = stablehlo.divide %v646, %v638 : tensor<128x16x32x32xf32>
    %v648 = stablehlo.add %v647, %v639 : tensor<128x16x32x32xf32>
    %v649 = stablehlo.rsqrt %v648 : tensor<128x16x32x32xf32>
    %v650 = stablehlo.multiply %v643, %v649 : tensor<128x16x32x32xf32>
    %v651 = stablehlo.reshape %v590 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v652 = stablehlo.multiply %v651, %v650 : tensor<128x16x32x32xf32>
    %v653 = stablehlo.reduce(%v652 init: %v636) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v654 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v655 = stablehlo.multiply %v654, %v653 : tensor<16xf32>
    %v656 = stablehlo.subtract %g1, %v655 : tensor<16xf32>
    %v657 = stablehlo.constant dense<0.0> : tensor<f32>
    %v658 = stablehlo.reshape %v590 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v659 = stablehlo.reduce(%v658 init: %v657) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v660 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v661 = stablehlo.multiply %v660, %v659 : tensor<16xf32>
    %v662 = stablehlo.subtract %bt1, %v661 : tensor<16xf32>
    %v663 = stablehlo.reshape %v28 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v664 = stablehlo.reshape %v579 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v665 = stablehlo.transpose %v663, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v666 = stablehlo.transpose %v664, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v667 = stablehlo.convolution(%v665, %v666)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x16x3x3xf32>
    %v668 = stablehlo.transpose %v667, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v669 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v670 = stablehlo.multiply %v669, %v668 : tensor<16x16x3x3xf32>
    %v671 = stablehlo.subtract %W2, %v670 : tensor<16x16x3x3xf32>
    %v672 = stablehlo.reshape %v579 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v673 = stablehlo.constant dense<0.0> : tensor<f32>
    %v674 = stablehlo.reduce(%v672 init: %v673) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v675 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v676 = stablehlo.multiply %v675, %v674 : tensor<16xf32>
    %v677 = stablehlo.subtract %cb2, %v676 : tensor<16xf32>
    %v678 = stablehlo.constant dense<0.0> : tensor<f32>
    %v679 = stablehlo.reshape %v33 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v680 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v681 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v682 = stablehlo.reduce(%v679 init: %v678) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v683 = stablehlo.broadcast_in_dim %v682, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v684 = stablehlo.divide %v683, %v680 : tensor<128x16x32x32xf32>
    %v685 = stablehlo.subtract %v679, %v684 : tensor<128x16x32x32xf32>
    %v686 = stablehlo.multiply %v685, %v685 : tensor<128x16x32x32xf32>
    %v687 = stablehlo.reduce(%v686 init: %v678) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v688 = stablehlo.broadcast_in_dim %v687, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v689 = stablehlo.divide %v688, %v680 : tensor<128x16x32x32xf32>
    %v690 = stablehlo.add %v689, %v681 : tensor<128x16x32x32xf32>
    %v691 = stablehlo.rsqrt %v690 : tensor<128x16x32x32xf32>
    %v692 = stablehlo.multiply %v685, %v691 : tensor<128x16x32x32xf32>
    %v693 = stablehlo.reshape %v549 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v694 = stablehlo.multiply %v693, %v692 : tensor<128x16x32x32xf32>
    %v695 = stablehlo.reduce(%v694 init: %v678) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v696 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v697 = stablehlo.multiply %v696, %v695 : tensor<16xf32>
    %v698 = stablehlo.subtract %g2, %v697 : tensor<16xf32>
    %v699 = stablehlo.constant dense<0.0> : tensor<f32>
    %v700 = stablehlo.reshape %v549 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v701 = stablehlo.reduce(%v700 init: %v699) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v702 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v703 = stablehlo.multiply %v702, %v701 : tensor<16xf32>
    %v704 = stablehlo.subtract %bt2, %v703 : tensor<16xf32>
    %v705 = stablehlo.reshape %v61 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v706 = stablehlo.reshape %v533 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v707 = stablehlo.transpose %v705, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v708 = stablehlo.transpose %v706, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v709 = stablehlo.convolution(%v707, %v708)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v710 = stablehlo.transpose %v709, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v711 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v712 = stablehlo.multiply %v711, %v710 : tensor<16x16x3x3xf32>
    %v713 = stablehlo.subtract %W3, %v712 : tensor<16x16x3x3xf32>
    %v714 = stablehlo.reshape %v533 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v715 = stablehlo.constant dense<0.0> : tensor<f32>
    %v716 = stablehlo.reduce(%v714 init: %v715) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v717 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v718 = stablehlo.multiply %v717, %v716 : tensor<16xf32>
    %v719 = stablehlo.subtract %cb3, %v718 : tensor<16xf32>
    %v720 = stablehlo.constant dense<0.0> : tensor<f32>
    %v721 = stablehlo.reshape %v66 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v722 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v723 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v724 = stablehlo.reduce(%v721 init: %v720) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v725 = stablehlo.broadcast_in_dim %v724, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v726 = stablehlo.divide %v725, %v722 : tensor<128x16x16x16xf32>
    %v727 = stablehlo.subtract %v721, %v726 : tensor<128x16x16x16xf32>
    %v728 = stablehlo.multiply %v727, %v727 : tensor<128x16x16x16xf32>
    %v729 = stablehlo.reduce(%v728 init: %v720) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v730 = stablehlo.broadcast_in_dim %v729, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v731 = stablehlo.divide %v730, %v722 : tensor<128x16x16x16xf32>
    %v732 = stablehlo.add %v731, %v723 : tensor<128x16x16x16xf32>
    %v733 = stablehlo.rsqrt %v732 : tensor<128x16x16x16xf32>
    %v734 = stablehlo.multiply %v727, %v733 : tensor<128x16x16x16xf32>
    %v735 = stablehlo.reshape %v503 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v736 = stablehlo.multiply %v735, %v734 : tensor<128x16x16x16xf32>
    %v737 = stablehlo.reduce(%v736 init: %v720) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v738 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v739 = stablehlo.multiply %v738, %v737 : tensor<16xf32>
    %v740 = stablehlo.subtract %g3, %v739 : tensor<16xf32>
    %v741 = stablehlo.constant dense<0.0> : tensor<f32>
    %v742 = stablehlo.reshape %v503 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v743 = stablehlo.reduce(%v742 init: %v741) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v744 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v745 = stablehlo.multiply %v744, %v743 : tensor<16xf32>
    %v746 = stablehlo.subtract %bt3, %v745 : tensor<16xf32>
    %v747 = stablehlo.reshape %v90 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v748 = stablehlo.reshape %v492 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v749 = stablehlo.transpose %v747, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v750 = stablehlo.transpose %v748, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v751 = stablehlo.convolution(%v749, %v750)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v752 = stablehlo.transpose %v751, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v753 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v754 = stablehlo.multiply %v753, %v752 : tensor<16x16x3x3xf32>
    %v755 = stablehlo.subtract %W4, %v754 : tensor<16x16x3x3xf32>
    %v756 = stablehlo.reshape %v492 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v757 = stablehlo.constant dense<0.0> : tensor<f32>
    %v758 = stablehlo.reduce(%v756 init: %v757) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v759 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v760 = stablehlo.multiply %v759, %v758 : tensor<16xf32>
    %v761 = stablehlo.subtract %cb4, %v760 : tensor<16xf32>
    %v762 = stablehlo.constant dense<0.0> : tensor<f32>
    %v763 = stablehlo.reshape %v95 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v764 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v765 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v766 = stablehlo.reduce(%v763 init: %v762) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v767 = stablehlo.broadcast_in_dim %v766, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v768 = stablehlo.divide %v767, %v764 : tensor<128x16x16x16xf32>
    %v769 = stablehlo.subtract %v763, %v768 : tensor<128x16x16x16xf32>
    %v770 = stablehlo.multiply %v769, %v769 : tensor<128x16x16x16xf32>
    %v771 = stablehlo.reduce(%v770 init: %v762) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v772 = stablehlo.broadcast_in_dim %v771, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v773 = stablehlo.divide %v772, %v764 : tensor<128x16x16x16xf32>
    %v774 = stablehlo.add %v773, %v765 : tensor<128x16x16x16xf32>
    %v775 = stablehlo.rsqrt %v774 : tensor<128x16x16x16xf32>
    %v776 = stablehlo.multiply %v769, %v775 : tensor<128x16x16x16xf32>
    %v777 = stablehlo.reshape %v462 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v778 = stablehlo.multiply %v777, %v776 : tensor<128x16x16x16xf32>
    %v779 = stablehlo.reduce(%v778 init: %v762) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v780 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v781 = stablehlo.multiply %v780, %v779 : tensor<16xf32>
    %v782 = stablehlo.subtract %g4, %v781 : tensor<16xf32>
    %v783 = stablehlo.constant dense<0.0> : tensor<f32>
    %v784 = stablehlo.reshape %v462 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v785 = stablehlo.reduce(%v784 init: %v783) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v786 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v787 = stablehlo.multiply %v786, %v785 : tensor<16xf32>
    %v788 = stablehlo.subtract %bt4, %v787 : tensor<16xf32>
    %v789 = stablehlo.reshape %v123 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v790 = stablehlo.reshape %v446 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v791 = stablehlo.transpose %v789, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v792 = stablehlo.transpose %v790, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v793 = stablehlo.convolution(%v791, %v792)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<16x32x3x3xf32>
    %v794 = stablehlo.transpose %v793, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v795 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v796 = stablehlo.multiply %v795, %v794 : tensor<32x16x3x3xf32>
    %v797 = stablehlo.subtract %W5, %v796 : tensor<32x16x3x3xf32>
    %v798 = stablehlo.reshape %v446 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v799 = stablehlo.constant dense<0.0> : tensor<f32>
    %v800 = stablehlo.reduce(%v798 init: %v799) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v801 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v802 = stablehlo.multiply %v801, %v800 : tensor<32xf32>
    %v803 = stablehlo.subtract %cb5, %v802 : tensor<32xf32>
    %v804 = stablehlo.constant dense<0.0> : tensor<f32>
    %v805 = stablehlo.reshape %v128 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v806 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v807 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v808 = stablehlo.reduce(%v805 init: %v804) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v809 = stablehlo.broadcast_in_dim %v808, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v810 = stablehlo.divide %v809, %v806 : tensor<128x32x8x8xf32>
    %v811 = stablehlo.subtract %v805, %v810 : tensor<128x32x8x8xf32>
    %v812 = stablehlo.multiply %v811, %v811 : tensor<128x32x8x8xf32>
    %v813 = stablehlo.reduce(%v812 init: %v804) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v814 = stablehlo.broadcast_in_dim %v813, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v815 = stablehlo.divide %v814, %v806 : tensor<128x32x8x8xf32>
    %v816 = stablehlo.add %v815, %v807 : tensor<128x32x8x8xf32>
    %v817 = stablehlo.rsqrt %v816 : tensor<128x32x8x8xf32>
    %v818 = stablehlo.multiply %v811, %v817 : tensor<128x32x8x8xf32>
    %v819 = stablehlo.reshape %v416 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v820 = stablehlo.multiply %v819, %v818 : tensor<128x32x8x8xf32>
    %v821 = stablehlo.reduce(%v820 init: %v804) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v822 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v823 = stablehlo.multiply %v822, %v821 : tensor<32xf32>
    %v824 = stablehlo.subtract %g5, %v823 : tensor<32xf32>
    %v825 = stablehlo.constant dense<0.0> : tensor<f32>
    %v826 = stablehlo.reshape %v416 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v827 = stablehlo.reduce(%v826 init: %v825) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v828 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v829 = stablehlo.multiply %v828, %v827 : tensor<32xf32>
    %v830 = stablehlo.subtract %bt5, %v829 : tensor<32xf32>
    %v831 = stablehlo.reshape %v152 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v832 = stablehlo.reshape %v405 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v833 = stablehlo.transpose %v831, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v834 = stablehlo.transpose %v832, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v835 = stablehlo.convolution(%v833, %v834)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<32x32x3x3xf32>
    %v836 = stablehlo.transpose %v835, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v837 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v838 = stablehlo.multiply %v837, %v836 : tensor<32x32x3x3xf32>
    %v839 = stablehlo.subtract %W6, %v838 : tensor<32x32x3x3xf32>
    %v840 = stablehlo.reshape %v405 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v841 = stablehlo.constant dense<0.0> : tensor<f32>
    %v842 = stablehlo.reduce(%v840 init: %v841) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v843 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v844 = stablehlo.multiply %v843, %v842 : tensor<32xf32>
    %v845 = stablehlo.subtract %cb6, %v844 : tensor<32xf32>
    %v846 = stablehlo.constant dense<0.0> : tensor<f32>
    %v847 = stablehlo.reshape %v157 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v848 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v849 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v850 = stablehlo.reduce(%v847 init: %v846) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v851 = stablehlo.broadcast_in_dim %v850, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v852 = stablehlo.divide %v851, %v848 : tensor<128x32x8x8xf32>
    %v853 = stablehlo.subtract %v847, %v852 : tensor<128x32x8x8xf32>
    %v854 = stablehlo.multiply %v853, %v853 : tensor<128x32x8x8xf32>
    %v855 = stablehlo.reduce(%v854 init: %v846) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v856 = stablehlo.broadcast_in_dim %v855, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v857 = stablehlo.divide %v856, %v848 : tensor<128x32x8x8xf32>
    %v858 = stablehlo.add %v857, %v849 : tensor<128x32x8x8xf32>
    %v859 = stablehlo.rsqrt %v858 : tensor<128x32x8x8xf32>
    %v860 = stablehlo.multiply %v853, %v859 : tensor<128x32x8x8xf32>
    %v861 = stablehlo.reshape %v375 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v862 = stablehlo.multiply %v861, %v860 : tensor<128x32x8x8xf32>
    %v863 = stablehlo.reduce(%v862 init: %v846) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v864 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v865 = stablehlo.multiply %v864, %v863 : tensor<32xf32>
    %v866 = stablehlo.subtract %g6, %v865 : tensor<32xf32>
    %v867 = stablehlo.constant dense<0.0> : tensor<f32>
    %v868 = stablehlo.reshape %v375 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v869 = stablehlo.reduce(%v868 init: %v867) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v870 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v871 = stablehlo.multiply %v870, %v869 : tensor<32xf32>
    %v872 = stablehlo.subtract %bt6, %v871 : tensor<32xf32>
    %v873 = stablehlo.reshape %v185 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v874 = stablehlo.reshape %v359 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v875 = stablehlo.transpose %v873, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v876 = stablehlo.transpose %v874, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v877 = stablehlo.convolution(%v875, %v876)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v878 = stablehlo.transpose %v877, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v879 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v880 = stablehlo.multiply %v879, %v878 : tensor<32x32x3x3xf32>
    %v881 = stablehlo.subtract %W7, %v880 : tensor<32x32x3x3xf32>
    %v882 = stablehlo.reshape %v359 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v883 = stablehlo.constant dense<0.0> : tensor<f32>
    %v884 = stablehlo.reduce(%v882 init: %v883) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v885 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v886 = stablehlo.multiply %v885, %v884 : tensor<32xf32>
    %v887 = stablehlo.subtract %cb7, %v886 : tensor<32xf32>
    %v888 = stablehlo.constant dense<0.0> : tensor<f32>
    %v889 = stablehlo.reshape %v190 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v890 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v891 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v892 = stablehlo.reduce(%v889 init: %v888) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v893 = stablehlo.broadcast_in_dim %v892, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v894 = stablehlo.divide %v893, %v890 : tensor<128x32x4x4xf32>
    %v895 = stablehlo.subtract %v889, %v894 : tensor<128x32x4x4xf32>
    %v896 = stablehlo.multiply %v895, %v895 : tensor<128x32x4x4xf32>
    %v897 = stablehlo.reduce(%v896 init: %v888) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v898 = stablehlo.broadcast_in_dim %v897, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v899 = stablehlo.divide %v898, %v890 : tensor<128x32x4x4xf32>
    %v900 = stablehlo.add %v899, %v891 : tensor<128x32x4x4xf32>
    %v901 = stablehlo.rsqrt %v900 : tensor<128x32x4x4xf32>
    %v902 = stablehlo.multiply %v895, %v901 : tensor<128x32x4x4xf32>
    %v903 = stablehlo.reshape %v329 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v904 = stablehlo.multiply %v903, %v902 : tensor<128x32x4x4xf32>
    %v905 = stablehlo.reduce(%v904 init: %v888) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v906 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v907 = stablehlo.multiply %v906, %v905 : tensor<32xf32>
    %v908 = stablehlo.subtract %g7, %v907 : tensor<32xf32>
    %v909 = stablehlo.constant dense<0.0> : tensor<f32>
    %v910 = stablehlo.reshape %v329 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v911 = stablehlo.reduce(%v910 init: %v909) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v912 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v913 = stablehlo.multiply %v912, %v911 : tensor<32xf32>
    %v914 = stablehlo.subtract %bt7, %v913 : tensor<32xf32>
    %v915 = stablehlo.reshape %v214 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v916 = stablehlo.reshape %v318 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v917 = stablehlo.transpose %v915, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v918 = stablehlo.transpose %v916, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v919 = stablehlo.convolution(%v917, %v918)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v920 = stablehlo.transpose %v919, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v921 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v922 = stablehlo.multiply %v921, %v920 : tensor<32x32x3x3xf32>
    %v923 = stablehlo.subtract %W8, %v922 : tensor<32x32x3x3xf32>
    %v924 = stablehlo.reshape %v318 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v925 = stablehlo.constant dense<0.0> : tensor<f32>
    %v926 = stablehlo.reduce(%v924 init: %v925) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v927 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v928 = stablehlo.multiply %v927, %v926 : tensor<32xf32>
    %v929 = stablehlo.subtract %cb8, %v928 : tensor<32xf32>
    %v930 = stablehlo.constant dense<0.0> : tensor<f32>
    %v931 = stablehlo.reshape %v219 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v932 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v933 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v934 = stablehlo.reduce(%v931 init: %v930) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v935 = stablehlo.broadcast_in_dim %v934, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v936 = stablehlo.divide %v935, %v932 : tensor<128x32x4x4xf32>
    %v937 = stablehlo.subtract %v931, %v936 : tensor<128x32x4x4xf32>
    %v938 = stablehlo.multiply %v937, %v937 : tensor<128x32x4x4xf32>
    %v939 = stablehlo.reduce(%v938 init: %v930) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v940 = stablehlo.broadcast_in_dim %v939, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v941 = stablehlo.divide %v940, %v932 : tensor<128x32x4x4xf32>
    %v942 = stablehlo.add %v941, %v933 : tensor<128x32x4x4xf32>
    %v943 = stablehlo.rsqrt %v942 : tensor<128x32x4x4xf32>
    %v944 = stablehlo.multiply %v937, %v943 : tensor<128x32x4x4xf32>
    %v945 = stablehlo.reshape %v288 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v946 = stablehlo.multiply %v945, %v944 : tensor<128x32x4x4xf32>
    %v947 = stablehlo.reduce(%v946 init: %v930) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v948 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v949 = stablehlo.multiply %v948, %v947 : tensor<32xf32>
    %v950 = stablehlo.subtract %g8, %v949 : tensor<32xf32>
    %v951 = stablehlo.constant dense<0.0> : tensor<f32>
    %v952 = stablehlo.reshape %v288 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v953 = stablehlo.reduce(%v952 init: %v951) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v954 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v955 = stablehlo.multiply %v954, %v953 : tensor<32xf32>
    %v956 = stablehlo.subtract %bt8, %v955 : tensor<32xf32>
    %v957 = stablehlo.dot_general %v247, %v276, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %v958 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v959 = stablehlo.multiply %v958, %v957 : tensor<128x64xf32>
    %v960 = stablehlo.subtract %W9, %v959 : tensor<128x64xf32>
    %v961 = stablehlo.constant dense<0.0> : tensor<f32>
    %v962 = stablehlo.reduce(%v276 init: %v961) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v963 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v964 = stablehlo.multiply %v963, %v962 : tensor<64xf32>
    %v965 = stablehlo.subtract %b9, %v964 : tensor<64xf32>
    %v966 = stablehlo.dot_general %v252, %v272, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<64x64xf32>
    %v967 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v968 = stablehlo.multiply %v967, %v966 : tensor<64x64xf32>
    %v969 = stablehlo.subtract %Wa, %v968 : tensor<64x64xf32>
    %v970 = stablehlo.constant dense<0.0> : tensor<f32>
    %v971 = stablehlo.reduce(%v272 init: %v970) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v972 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v973 = stablehlo.multiply %v972, %v971 : tensor<64xf32>
    %v974 = stablehlo.subtract %ba, %v973 : tensor<64xf32>
    %v975 = stablehlo.dot_general %v257, %v268, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x10xf32>) -> tensor<64x10xf32>
    %v976 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v977 = stablehlo.multiply %v976, %v975 : tensor<64x10xf32>
    %v978 = stablehlo.subtract %Wb, %v977 : tensor<64x10xf32>
    %v979 = stablehlo.constant dense<0.0> : tensor<f32>
    %v980 = stablehlo.reduce(%v268 init: %v979) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v981 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v982 = stablehlo.multiply %v981, %v980 : tensor<10xf32>
    %v983 = stablehlo.subtract %bb, %v982 : tensor<10xf32>
    return %v629, %v635, %v656, %v662, %v671, %v677, %v698, %v704, %v713, %v719, %v740, %v746, %v755, %v761, %v782, %v788, %v797, %v803, %v824, %v830, %v839, %v845, %v866, %v872, %v881, %v887, %v908, %v914, %v923, %v929, %v950, %v956, %v960, %v965, %v969, %v974, %v978, %v983, %W1m, %cb1m, %g1m, %bt1m, %W2m, %cb2m, %g2m, %bt2m, %W3m, %cb3m, %g3m, %bt3m, %W4m, %cb4m, %g4m, %bt4m, %W5m, %cb5m, %g5m, %bt5m, %W6m, %cb6m, %g6m, %bt6m, %W7m, %cb7m, %g7m, %bt7m, %W8m, %cb8m, %g8m, %bt8m, %W9m, %b9m, %Wam, %bam, %Wbm, %bbm, %W1v, %cb1v, %g1v, %bt1v, %W2v, %cb2v, %g2v, %bt2v, %W3v, %cb3v, %g3v, %bt3v, %W4v, %cb4v, %g4v, %bt4v, %W5v, %cb5v, %g5v, %bt5v, %W6v, %cb6v, %g6v, %bt6v, %W7v, %cb7v, %g7v, %bt7v, %W8v, %cb8v, %g8v, %bt8v, %W9v, %b9v, %Wav, %bav, %Wbv, %bbv, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
