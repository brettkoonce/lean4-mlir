module @m {
  func.func @cifar8_bn_adam_train_step(%x: tensor<128x3072xf32>, %W1: tensor<16x3x3x3xf32>, %cb1: tensor<16xf32>, %g1: tensor<16xf32>, %bt1: tensor<16xf32>, %W2: tensor<16x16x3x3xf32>, %cb2: tensor<16xf32>, %g2: tensor<16xf32>, %bt2: tensor<16xf32>, %W3: tensor<16x16x3x3xf32>, %cb3: tensor<16xf32>, %g3: tensor<16xf32>, %bt3: tensor<16xf32>, %W4: tensor<16x16x3x3xf32>, %cb4: tensor<16xf32>, %g4: tensor<16xf32>, %bt4: tensor<16xf32>, %W5: tensor<32x16x3x3xf32>, %cb5: tensor<32xf32>, %g5: tensor<32xf32>, %bt5: tensor<32xf32>, %W6: tensor<32x32x3x3xf32>, %cb6: tensor<32xf32>, %g6: tensor<32xf32>, %bt6: tensor<32xf32>, %W7: tensor<32x32x3x3xf32>, %cb7: tensor<32xf32>, %g7: tensor<32xf32>, %bt7: tensor<32xf32>, %W8: tensor<32x32x3x3xf32>, %cb8: tensor<32xf32>, %g8: tensor<32xf32>, %bt8: tensor<32xf32>, %W9: tensor<128x64xf32>, %b9: tensor<64xf32>, %Wa: tensor<64x64xf32>, %ba: tensor<64xf32>, %Wb: tensor<64x10xf32>, %bb: tensor<10xf32>, %W1m: tensor<16x3x3x3xf32>, %cb1m: tensor<16xf32>, %g1m: tensor<16xf32>, %bt1m: tensor<16xf32>, %W2m: tensor<16x16x3x3xf32>, %cb2m: tensor<16xf32>, %g2m: tensor<16xf32>, %bt2m: tensor<16xf32>, %W3m: tensor<16x16x3x3xf32>, %cb3m: tensor<16xf32>, %g3m: tensor<16xf32>, %bt3m: tensor<16xf32>, %W4m: tensor<16x16x3x3xf32>, %cb4m: tensor<16xf32>, %g4m: tensor<16xf32>, %bt4m: tensor<16xf32>, %W5m: tensor<32x16x3x3xf32>, %cb5m: tensor<32xf32>, %g5m: tensor<32xf32>, %bt5m: tensor<32xf32>, %W6m: tensor<32x32x3x3xf32>, %cb6m: tensor<32xf32>, %g6m: tensor<32xf32>, %bt6m: tensor<32xf32>, %W7m: tensor<32x32x3x3xf32>, %cb7m: tensor<32xf32>, %g7m: tensor<32xf32>, %bt7m: tensor<32xf32>, %W8m: tensor<32x32x3x3xf32>, %cb8m: tensor<32xf32>, %g8m: tensor<32xf32>, %bt8m: tensor<32xf32>, %W9m: tensor<128x64xf32>, %b9m: tensor<64xf32>, %Wam: tensor<64x64xf32>, %bam: tensor<64xf32>, %Wbm: tensor<64x10xf32>, %bbm: tensor<10xf32>, %W1v: tensor<16x3x3x3xf32>, %cb1v: tensor<16xf32>, %g1v: tensor<16xf32>, %bt1v: tensor<16xf32>, %W2v: tensor<16x16x3x3xf32>, %cb2v: tensor<16xf32>, %g2v: tensor<16xf32>, %bt2v: tensor<16xf32>, %W3v: tensor<16x16x3x3xf32>, %cb3v: tensor<16xf32>, %g3v: tensor<16xf32>, %bt3v: tensor<16xf32>, %W4v: tensor<16x16x3x3xf32>, %cb4v: tensor<16xf32>, %g4v: tensor<16xf32>, %bt4v: tensor<16xf32>, %W5v: tensor<32x16x3x3xf32>, %cb5v: tensor<32xf32>, %g5v: tensor<32xf32>, %bt5v: tensor<32xf32>, %W6v: tensor<32x32x3x3xf32>, %cb6v: tensor<32xf32>, %g6v: tensor<32xf32>, %bt6v: tensor<32xf32>, %W7v: tensor<32x32x3x3xf32>, %cb7v: tensor<32xf32>, %g7v: tensor<32xf32>, %bt7v: tensor<32xf32>, %W8v: tensor<32x32x3x3xf32>, %cb8v: tensor<32xf32>, %g8v: tensor<32xf32>, %bt8v: tensor<32xf32>, %W9v: tensor<128x64xf32>, %b9v: tensor<64xf32>, %Wav: tensor<64x64xf32>, %bav: tensor<64xf32>, %Wbv: tensor<64x10xf32>, %bbv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: tensor<128x10xf32>) -> (tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>) {
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
    %v627 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v628 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v629 = stablehlo.multiply %v627, %W1m : tensor<16x3x3x3xf32>
    %v630 = stablehlo.multiply %v628, %v626 : tensor<16x3x3x3xf32>
    %v631 = stablehlo.add %v629, %v630 : tensor<16x3x3x3xf32>
    %v632 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v633 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v634 = stablehlo.multiply %v632, %W1v : tensor<16x3x3x3xf32>
    %v635 = stablehlo.multiply %v626, %v626 : tensor<16x3x3x3xf32>
    %v636 = stablehlo.multiply %v633, %v635 : tensor<16x3x3x3xf32>
    %v637 = stablehlo.add %v634, %v636 : tensor<16x3x3x3xf32>
    %v638 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v639 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v640 = stablehlo.divide %v631, %v638 : tensor<16x3x3x3xf32>
    %v641 = stablehlo.divide %v637, %v639 : tensor<16x3x3x3xf32>
    %v642 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v643 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v644 = stablehlo.sqrt %v641 : tensor<16x3x3x3xf32>
    %v645 = stablehlo.add %v644, %v643 : tensor<16x3x3x3xf32>
    %v646 = stablehlo.divide %v640, %v645 : tensor<16x3x3x3xf32>
    %v647 = stablehlo.multiply %v642, %v646 : tensor<16x3x3x3xf32>
    %v648 = stablehlo.subtract %W1, %v647 : tensor<16x3x3x3xf32>
    %v649 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v650 = stablehlo.multiply %v649, %v642 : tensor<16x3x3x3xf32>
    %v651 = stablehlo.multiply %v650, %W1 : tensor<16x3x3x3xf32>
    %v652 = stablehlo.subtract %v648, %v651 : tensor<16x3x3x3xf32>
    %v653 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v654 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v655 = stablehlo.multiply %v653, %W1m : tensor<16x3x3x3xf32>
    %v656 = stablehlo.multiply %v654, %v626 : tensor<16x3x3x3xf32>
    %v657 = stablehlo.add %v655, %v656 : tensor<16x3x3x3xf32>
    %v658 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v659 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v660 = stablehlo.multiply %v658, %W1v : tensor<16x3x3x3xf32>
    %v661 = stablehlo.multiply %v626, %v626 : tensor<16x3x3x3xf32>
    %v662 = stablehlo.multiply %v659, %v661 : tensor<16x3x3x3xf32>
    %v663 = stablehlo.add %v660, %v662 : tensor<16x3x3x3xf32>
    %v664 = stablehlo.reshape %v620 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v665 = stablehlo.constant dense<0.0> : tensor<f32>
    %v666 = stablehlo.reduce(%v664 init: %v665) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v667 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v668 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v669 = stablehlo.multiply %v667, %cb1m : tensor<16xf32>
    %v670 = stablehlo.multiply %v668, %v666 : tensor<16xf32>
    %v671 = stablehlo.add %v669, %v670 : tensor<16xf32>
    %v672 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v673 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v674 = stablehlo.multiply %v672, %cb1v : tensor<16xf32>
    %v675 = stablehlo.multiply %v666, %v666 : tensor<16xf32>
    %v676 = stablehlo.multiply %v673, %v675 : tensor<16xf32>
    %v677 = stablehlo.add %v674, %v676 : tensor<16xf32>
    %v678 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v679 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v680 = stablehlo.divide %v671, %v678 : tensor<16xf32>
    %v681 = stablehlo.divide %v677, %v679 : tensor<16xf32>
    %v682 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v683 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v684 = stablehlo.sqrt %v681 : tensor<16xf32>
    %v685 = stablehlo.add %v684, %v683 : tensor<16xf32>
    %v686 = stablehlo.divide %v680, %v685 : tensor<16xf32>
    %v687 = stablehlo.multiply %v682, %v686 : tensor<16xf32>
    %v688 = stablehlo.subtract %cb1, %v687 : tensor<16xf32>
    %v689 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v690 = stablehlo.multiply %v689, %v682 : tensor<16xf32>
    %v691 = stablehlo.multiply %v690, %cb1 : tensor<16xf32>
    %v692 = stablehlo.subtract %v688, %v691 : tensor<16xf32>
    %v693 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v694 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v695 = stablehlo.multiply %v693, %cb1m : tensor<16xf32>
    %v696 = stablehlo.multiply %v694, %v666 : tensor<16xf32>
    %v697 = stablehlo.add %v695, %v696 : tensor<16xf32>
    %v698 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v699 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v700 = stablehlo.multiply %v698, %cb1v : tensor<16xf32>
    %v701 = stablehlo.multiply %v666, %v666 : tensor<16xf32>
    %v702 = stablehlo.multiply %v699, %v701 : tensor<16xf32>
    %v703 = stablehlo.add %v700, %v702 : tensor<16xf32>
    %v704 = stablehlo.constant dense<0.0> : tensor<f32>
    %v705 = stablehlo.reshape %v4 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v706 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v707 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v708 = stablehlo.reduce(%v705 init: %v704) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v709 = stablehlo.broadcast_in_dim %v708, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v710 = stablehlo.divide %v709, %v706 : tensor<128x16x32x32xf32>
    %v711 = stablehlo.subtract %v705, %v710 : tensor<128x16x32x32xf32>
    %v712 = stablehlo.multiply %v711, %v711 : tensor<128x16x32x32xf32>
    %v713 = stablehlo.reduce(%v712 init: %v704) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v714 = stablehlo.broadcast_in_dim %v713, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v715 = stablehlo.divide %v714, %v706 : tensor<128x16x32x32xf32>
    %v716 = stablehlo.add %v715, %v707 : tensor<128x16x32x32xf32>
    %v717 = stablehlo.rsqrt %v716 : tensor<128x16x32x32xf32>
    %v718 = stablehlo.multiply %v711, %v717 : tensor<128x16x32x32xf32>
    %v719 = stablehlo.reshape %v590 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v720 = stablehlo.multiply %v719, %v718 : tensor<128x16x32x32xf32>
    %v721 = stablehlo.reduce(%v720 init: %v704) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v722 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v723 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v724 = stablehlo.multiply %v722, %g1m : tensor<16xf32>
    %v725 = stablehlo.multiply %v723, %v721 : tensor<16xf32>
    %v726 = stablehlo.add %v724, %v725 : tensor<16xf32>
    %v727 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v728 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v729 = stablehlo.multiply %v727, %g1v : tensor<16xf32>
    %v730 = stablehlo.multiply %v721, %v721 : tensor<16xf32>
    %v731 = stablehlo.multiply %v728, %v730 : tensor<16xf32>
    %v732 = stablehlo.add %v729, %v731 : tensor<16xf32>
    %v733 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v734 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v735 = stablehlo.divide %v726, %v733 : tensor<16xf32>
    %v736 = stablehlo.divide %v732, %v734 : tensor<16xf32>
    %v737 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v738 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v739 = stablehlo.sqrt %v736 : tensor<16xf32>
    %v740 = stablehlo.add %v739, %v738 : tensor<16xf32>
    %v741 = stablehlo.divide %v735, %v740 : tensor<16xf32>
    %v742 = stablehlo.multiply %v737, %v741 : tensor<16xf32>
    %v743 = stablehlo.subtract %g1, %v742 : tensor<16xf32>
    %v744 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v745 = stablehlo.multiply %v744, %v737 : tensor<16xf32>
    %v746 = stablehlo.multiply %v745, %g1 : tensor<16xf32>
    %v747 = stablehlo.subtract %v743, %v746 : tensor<16xf32>
    %v748 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v749 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v750 = stablehlo.multiply %v748, %g1m : tensor<16xf32>
    %v751 = stablehlo.multiply %v749, %v721 : tensor<16xf32>
    %v752 = stablehlo.add %v750, %v751 : tensor<16xf32>
    %v753 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v754 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v755 = stablehlo.multiply %v753, %g1v : tensor<16xf32>
    %v756 = stablehlo.multiply %v721, %v721 : tensor<16xf32>
    %v757 = stablehlo.multiply %v754, %v756 : tensor<16xf32>
    %v758 = stablehlo.add %v755, %v757 : tensor<16xf32>
    %v759 = stablehlo.constant dense<0.0> : tensor<f32>
    %v760 = stablehlo.reshape %v590 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v761 = stablehlo.reduce(%v760 init: %v759) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v762 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v763 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v764 = stablehlo.multiply %v762, %bt1m : tensor<16xf32>
    %v765 = stablehlo.multiply %v763, %v761 : tensor<16xf32>
    %v766 = stablehlo.add %v764, %v765 : tensor<16xf32>
    %v767 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v768 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v769 = stablehlo.multiply %v767, %bt1v : tensor<16xf32>
    %v770 = stablehlo.multiply %v761, %v761 : tensor<16xf32>
    %v771 = stablehlo.multiply %v768, %v770 : tensor<16xf32>
    %v772 = stablehlo.add %v769, %v771 : tensor<16xf32>
    %v773 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v774 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v775 = stablehlo.divide %v766, %v773 : tensor<16xf32>
    %v776 = stablehlo.divide %v772, %v774 : tensor<16xf32>
    %v777 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v778 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v779 = stablehlo.sqrt %v776 : tensor<16xf32>
    %v780 = stablehlo.add %v779, %v778 : tensor<16xf32>
    %v781 = stablehlo.divide %v775, %v780 : tensor<16xf32>
    %v782 = stablehlo.multiply %v777, %v781 : tensor<16xf32>
    %v783 = stablehlo.subtract %bt1, %v782 : tensor<16xf32>
    %v784 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v785 = stablehlo.multiply %v784, %v777 : tensor<16xf32>
    %v786 = stablehlo.multiply %v785, %bt1 : tensor<16xf32>
    %v787 = stablehlo.subtract %v783, %v786 : tensor<16xf32>
    %v788 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v789 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v790 = stablehlo.multiply %v788, %bt1m : tensor<16xf32>
    %v791 = stablehlo.multiply %v789, %v761 : tensor<16xf32>
    %v792 = stablehlo.add %v790, %v791 : tensor<16xf32>
    %v793 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v794 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v795 = stablehlo.multiply %v793, %bt1v : tensor<16xf32>
    %v796 = stablehlo.multiply %v761, %v761 : tensor<16xf32>
    %v797 = stablehlo.multiply %v794, %v796 : tensor<16xf32>
    %v798 = stablehlo.add %v795, %v797 : tensor<16xf32>
    %v799 = stablehlo.reshape %v28 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v800 = stablehlo.reshape %v579 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v801 = stablehlo.transpose %v799, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v802 = stablehlo.transpose %v800, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v803 = stablehlo.convolution(%v801, %v802)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x16x3x3xf32>
    %v804 = stablehlo.transpose %v803, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v805 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v806 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v807 = stablehlo.multiply %v805, %W2m : tensor<16x16x3x3xf32>
    %v808 = stablehlo.multiply %v806, %v804 : tensor<16x16x3x3xf32>
    %v809 = stablehlo.add %v807, %v808 : tensor<16x16x3x3xf32>
    %v810 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v811 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v812 = stablehlo.multiply %v810, %W2v : tensor<16x16x3x3xf32>
    %v813 = stablehlo.multiply %v804, %v804 : tensor<16x16x3x3xf32>
    %v814 = stablehlo.multiply %v811, %v813 : tensor<16x16x3x3xf32>
    %v815 = stablehlo.add %v812, %v814 : tensor<16x16x3x3xf32>
    %v816 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v817 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v818 = stablehlo.divide %v809, %v816 : tensor<16x16x3x3xf32>
    %v819 = stablehlo.divide %v815, %v817 : tensor<16x16x3x3xf32>
    %v820 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v821 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v822 = stablehlo.sqrt %v819 : tensor<16x16x3x3xf32>
    %v823 = stablehlo.add %v822, %v821 : tensor<16x16x3x3xf32>
    %v824 = stablehlo.divide %v818, %v823 : tensor<16x16x3x3xf32>
    %v825 = stablehlo.multiply %v820, %v824 : tensor<16x16x3x3xf32>
    %v826 = stablehlo.subtract %W2, %v825 : tensor<16x16x3x3xf32>
    %v827 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v828 = stablehlo.multiply %v827, %v820 : tensor<16x16x3x3xf32>
    %v829 = stablehlo.multiply %v828, %W2 : tensor<16x16x3x3xf32>
    %v830 = stablehlo.subtract %v826, %v829 : tensor<16x16x3x3xf32>
    %v831 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v832 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v833 = stablehlo.multiply %v831, %W2m : tensor<16x16x3x3xf32>
    %v834 = stablehlo.multiply %v832, %v804 : tensor<16x16x3x3xf32>
    %v835 = stablehlo.add %v833, %v834 : tensor<16x16x3x3xf32>
    %v836 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v837 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v838 = stablehlo.multiply %v836, %W2v : tensor<16x16x3x3xf32>
    %v839 = stablehlo.multiply %v804, %v804 : tensor<16x16x3x3xf32>
    %v840 = stablehlo.multiply %v837, %v839 : tensor<16x16x3x3xf32>
    %v841 = stablehlo.add %v838, %v840 : tensor<16x16x3x3xf32>
    %v842 = stablehlo.reshape %v579 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v843 = stablehlo.constant dense<0.0> : tensor<f32>
    %v844 = stablehlo.reduce(%v842 init: %v843) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v845 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v846 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v847 = stablehlo.multiply %v845, %cb2m : tensor<16xf32>
    %v848 = stablehlo.multiply %v846, %v844 : tensor<16xf32>
    %v849 = stablehlo.add %v847, %v848 : tensor<16xf32>
    %v850 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v851 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v852 = stablehlo.multiply %v850, %cb2v : tensor<16xf32>
    %v853 = stablehlo.multiply %v844, %v844 : tensor<16xf32>
    %v854 = stablehlo.multiply %v851, %v853 : tensor<16xf32>
    %v855 = stablehlo.add %v852, %v854 : tensor<16xf32>
    %v856 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v857 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v858 = stablehlo.divide %v849, %v856 : tensor<16xf32>
    %v859 = stablehlo.divide %v855, %v857 : tensor<16xf32>
    %v860 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v861 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v862 = stablehlo.sqrt %v859 : tensor<16xf32>
    %v863 = stablehlo.add %v862, %v861 : tensor<16xf32>
    %v864 = stablehlo.divide %v858, %v863 : tensor<16xf32>
    %v865 = stablehlo.multiply %v860, %v864 : tensor<16xf32>
    %v866 = stablehlo.subtract %cb2, %v865 : tensor<16xf32>
    %v867 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v868 = stablehlo.multiply %v867, %v860 : tensor<16xf32>
    %v869 = stablehlo.multiply %v868, %cb2 : tensor<16xf32>
    %v870 = stablehlo.subtract %v866, %v869 : tensor<16xf32>
    %v871 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v872 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v873 = stablehlo.multiply %v871, %cb2m : tensor<16xf32>
    %v874 = stablehlo.multiply %v872, %v844 : tensor<16xf32>
    %v875 = stablehlo.add %v873, %v874 : tensor<16xf32>
    %v876 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v877 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v878 = stablehlo.multiply %v876, %cb2v : tensor<16xf32>
    %v879 = stablehlo.multiply %v844, %v844 : tensor<16xf32>
    %v880 = stablehlo.multiply %v877, %v879 : tensor<16xf32>
    %v881 = stablehlo.add %v878, %v880 : tensor<16xf32>
    %v882 = stablehlo.constant dense<0.0> : tensor<f32>
    %v883 = stablehlo.reshape %v33 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v884 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v885 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v886 = stablehlo.reduce(%v883 init: %v882) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v887 = stablehlo.broadcast_in_dim %v886, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v888 = stablehlo.divide %v887, %v884 : tensor<128x16x32x32xf32>
    %v889 = stablehlo.subtract %v883, %v888 : tensor<128x16x32x32xf32>
    %v890 = stablehlo.multiply %v889, %v889 : tensor<128x16x32x32xf32>
    %v891 = stablehlo.reduce(%v890 init: %v882) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v892 = stablehlo.broadcast_in_dim %v891, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v893 = stablehlo.divide %v892, %v884 : tensor<128x16x32x32xf32>
    %v894 = stablehlo.add %v893, %v885 : tensor<128x16x32x32xf32>
    %v895 = stablehlo.rsqrt %v894 : tensor<128x16x32x32xf32>
    %v896 = stablehlo.multiply %v889, %v895 : tensor<128x16x32x32xf32>
    %v897 = stablehlo.reshape %v549 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v898 = stablehlo.multiply %v897, %v896 : tensor<128x16x32x32xf32>
    %v899 = stablehlo.reduce(%v898 init: %v882) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v900 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v901 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v902 = stablehlo.multiply %v900, %g2m : tensor<16xf32>
    %v903 = stablehlo.multiply %v901, %v899 : tensor<16xf32>
    %v904 = stablehlo.add %v902, %v903 : tensor<16xf32>
    %v905 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v906 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v907 = stablehlo.multiply %v905, %g2v : tensor<16xf32>
    %v908 = stablehlo.multiply %v899, %v899 : tensor<16xf32>
    %v909 = stablehlo.multiply %v906, %v908 : tensor<16xf32>
    %v910 = stablehlo.add %v907, %v909 : tensor<16xf32>
    %v911 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v912 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v913 = stablehlo.divide %v904, %v911 : tensor<16xf32>
    %v914 = stablehlo.divide %v910, %v912 : tensor<16xf32>
    %v915 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v916 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v917 = stablehlo.sqrt %v914 : tensor<16xf32>
    %v918 = stablehlo.add %v917, %v916 : tensor<16xf32>
    %v919 = stablehlo.divide %v913, %v918 : tensor<16xf32>
    %v920 = stablehlo.multiply %v915, %v919 : tensor<16xf32>
    %v921 = stablehlo.subtract %g2, %v920 : tensor<16xf32>
    %v922 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v923 = stablehlo.multiply %v922, %v915 : tensor<16xf32>
    %v924 = stablehlo.multiply %v923, %g2 : tensor<16xf32>
    %v925 = stablehlo.subtract %v921, %v924 : tensor<16xf32>
    %v926 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v927 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v928 = stablehlo.multiply %v926, %g2m : tensor<16xf32>
    %v929 = stablehlo.multiply %v927, %v899 : tensor<16xf32>
    %v930 = stablehlo.add %v928, %v929 : tensor<16xf32>
    %v931 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v932 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v933 = stablehlo.multiply %v931, %g2v : tensor<16xf32>
    %v934 = stablehlo.multiply %v899, %v899 : tensor<16xf32>
    %v935 = stablehlo.multiply %v932, %v934 : tensor<16xf32>
    %v936 = stablehlo.add %v933, %v935 : tensor<16xf32>
    %v937 = stablehlo.constant dense<0.0> : tensor<f32>
    %v938 = stablehlo.reshape %v549 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v939 = stablehlo.reduce(%v938 init: %v937) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v940 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v941 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v942 = stablehlo.multiply %v940, %bt2m : tensor<16xf32>
    %v943 = stablehlo.multiply %v941, %v939 : tensor<16xf32>
    %v944 = stablehlo.add %v942, %v943 : tensor<16xf32>
    %v945 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v946 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v947 = stablehlo.multiply %v945, %bt2v : tensor<16xf32>
    %v948 = stablehlo.multiply %v939, %v939 : tensor<16xf32>
    %v949 = stablehlo.multiply %v946, %v948 : tensor<16xf32>
    %v950 = stablehlo.add %v947, %v949 : tensor<16xf32>
    %v951 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v952 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v953 = stablehlo.divide %v944, %v951 : tensor<16xf32>
    %v954 = stablehlo.divide %v950, %v952 : tensor<16xf32>
    %v955 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v956 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v957 = stablehlo.sqrt %v954 : tensor<16xf32>
    %v958 = stablehlo.add %v957, %v956 : tensor<16xf32>
    %v959 = stablehlo.divide %v953, %v958 : tensor<16xf32>
    %v960 = stablehlo.multiply %v955, %v959 : tensor<16xf32>
    %v961 = stablehlo.subtract %bt2, %v960 : tensor<16xf32>
    %v962 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v963 = stablehlo.multiply %v962, %v955 : tensor<16xf32>
    %v964 = stablehlo.multiply %v963, %bt2 : tensor<16xf32>
    %v965 = stablehlo.subtract %v961, %v964 : tensor<16xf32>
    %v966 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v967 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v968 = stablehlo.multiply %v966, %bt2m : tensor<16xf32>
    %v969 = stablehlo.multiply %v967, %v939 : tensor<16xf32>
    %v970 = stablehlo.add %v968, %v969 : tensor<16xf32>
    %v971 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v972 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v973 = stablehlo.multiply %v971, %bt2v : tensor<16xf32>
    %v974 = stablehlo.multiply %v939, %v939 : tensor<16xf32>
    %v975 = stablehlo.multiply %v972, %v974 : tensor<16xf32>
    %v976 = stablehlo.add %v973, %v975 : tensor<16xf32>
    %v977 = stablehlo.reshape %v61 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v978 = stablehlo.reshape %v533 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v979 = stablehlo.transpose %v977, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v980 = stablehlo.transpose %v978, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v981 = stablehlo.convolution(%v979, %v980)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v982 = stablehlo.transpose %v981, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v983 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v984 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v985 = stablehlo.multiply %v983, %W3m : tensor<16x16x3x3xf32>
    %v986 = stablehlo.multiply %v984, %v982 : tensor<16x16x3x3xf32>
    %v987 = stablehlo.add %v985, %v986 : tensor<16x16x3x3xf32>
    %v988 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v989 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v990 = stablehlo.multiply %v988, %W3v : tensor<16x16x3x3xf32>
    %v991 = stablehlo.multiply %v982, %v982 : tensor<16x16x3x3xf32>
    %v992 = stablehlo.multiply %v989, %v991 : tensor<16x16x3x3xf32>
    %v993 = stablehlo.add %v990, %v992 : tensor<16x16x3x3xf32>
    %v994 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v995 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v996 = stablehlo.divide %v987, %v994 : tensor<16x16x3x3xf32>
    %v997 = stablehlo.divide %v993, %v995 : tensor<16x16x3x3xf32>
    %v998 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v999 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1000 = stablehlo.sqrt %v997 : tensor<16x16x3x3xf32>
    %v1001 = stablehlo.add %v1000, %v999 : tensor<16x16x3x3xf32>
    %v1002 = stablehlo.divide %v996, %v1001 : tensor<16x16x3x3xf32>
    %v1003 = stablehlo.multiply %v998, %v1002 : tensor<16x16x3x3xf32>
    %v1004 = stablehlo.subtract %W3, %v1003 : tensor<16x16x3x3xf32>
    %v1005 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1006 = stablehlo.multiply %v1005, %v998 : tensor<16x16x3x3xf32>
    %v1007 = stablehlo.multiply %v1006, %W3 : tensor<16x16x3x3xf32>
    %v1008 = stablehlo.subtract %v1004, %v1007 : tensor<16x16x3x3xf32>
    %v1009 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1010 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1011 = stablehlo.multiply %v1009, %W3m : tensor<16x16x3x3xf32>
    %v1012 = stablehlo.multiply %v1010, %v982 : tensor<16x16x3x3xf32>
    %v1013 = stablehlo.add %v1011, %v1012 : tensor<16x16x3x3xf32>
    %v1014 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1015 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1016 = stablehlo.multiply %v1014, %W3v : tensor<16x16x3x3xf32>
    %v1017 = stablehlo.multiply %v982, %v982 : tensor<16x16x3x3xf32>
    %v1018 = stablehlo.multiply %v1015, %v1017 : tensor<16x16x3x3xf32>
    %v1019 = stablehlo.add %v1016, %v1018 : tensor<16x16x3x3xf32>
    %v1020 = stablehlo.reshape %v533 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1021 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1022 = stablehlo.reduce(%v1020 init: %v1021) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v1023 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1024 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1025 = stablehlo.multiply %v1023, %cb3m : tensor<16xf32>
    %v1026 = stablehlo.multiply %v1024, %v1022 : tensor<16xf32>
    %v1027 = stablehlo.add %v1025, %v1026 : tensor<16xf32>
    %v1028 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1029 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1030 = stablehlo.multiply %v1028, %cb3v : tensor<16xf32>
    %v1031 = stablehlo.multiply %v1022, %v1022 : tensor<16xf32>
    %v1032 = stablehlo.multiply %v1029, %v1031 : tensor<16xf32>
    %v1033 = stablehlo.add %v1030, %v1032 : tensor<16xf32>
    %v1034 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1035 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1036 = stablehlo.divide %v1027, %v1034 : tensor<16xf32>
    %v1037 = stablehlo.divide %v1033, %v1035 : tensor<16xf32>
    %v1038 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1039 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1040 = stablehlo.sqrt %v1037 : tensor<16xf32>
    %v1041 = stablehlo.add %v1040, %v1039 : tensor<16xf32>
    %v1042 = stablehlo.divide %v1036, %v1041 : tensor<16xf32>
    %v1043 = stablehlo.multiply %v1038, %v1042 : tensor<16xf32>
    %v1044 = stablehlo.subtract %cb3, %v1043 : tensor<16xf32>
    %v1045 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1046 = stablehlo.multiply %v1045, %v1038 : tensor<16xf32>
    %v1047 = stablehlo.multiply %v1046, %cb3 : tensor<16xf32>
    %v1048 = stablehlo.subtract %v1044, %v1047 : tensor<16xf32>
    %v1049 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1050 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1051 = stablehlo.multiply %v1049, %cb3m : tensor<16xf32>
    %v1052 = stablehlo.multiply %v1050, %v1022 : tensor<16xf32>
    %v1053 = stablehlo.add %v1051, %v1052 : tensor<16xf32>
    %v1054 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1055 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1056 = stablehlo.multiply %v1054, %cb3v : tensor<16xf32>
    %v1057 = stablehlo.multiply %v1022, %v1022 : tensor<16xf32>
    %v1058 = stablehlo.multiply %v1055, %v1057 : tensor<16xf32>
    %v1059 = stablehlo.add %v1056, %v1058 : tensor<16xf32>
    %v1060 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1061 = stablehlo.reshape %v66 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1062 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v1063 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v1064 = stablehlo.reduce(%v1061 init: %v1060) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v1065 = stablehlo.broadcast_in_dim %v1064, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v1066 = stablehlo.divide %v1065, %v1062 : tensor<128x16x16x16xf32>
    %v1067 = stablehlo.subtract %v1061, %v1066 : tensor<128x16x16x16xf32>
    %v1068 = stablehlo.multiply %v1067, %v1067 : tensor<128x16x16x16xf32>
    %v1069 = stablehlo.reduce(%v1068 init: %v1060) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v1070 = stablehlo.broadcast_in_dim %v1069, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v1071 = stablehlo.divide %v1070, %v1062 : tensor<128x16x16x16xf32>
    %v1072 = stablehlo.add %v1071, %v1063 : tensor<128x16x16x16xf32>
    %v1073 = stablehlo.rsqrt %v1072 : tensor<128x16x16x16xf32>
    %v1074 = stablehlo.multiply %v1067, %v1073 : tensor<128x16x16x16xf32>
    %v1075 = stablehlo.reshape %v503 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1076 = stablehlo.multiply %v1075, %v1074 : tensor<128x16x16x16xf32>
    %v1077 = stablehlo.reduce(%v1076 init: %v1060) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v1078 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1079 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1080 = stablehlo.multiply %v1078, %g3m : tensor<16xf32>
    %v1081 = stablehlo.multiply %v1079, %v1077 : tensor<16xf32>
    %v1082 = stablehlo.add %v1080, %v1081 : tensor<16xf32>
    %v1083 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1084 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1085 = stablehlo.multiply %v1083, %g3v : tensor<16xf32>
    %v1086 = stablehlo.multiply %v1077, %v1077 : tensor<16xf32>
    %v1087 = stablehlo.multiply %v1084, %v1086 : tensor<16xf32>
    %v1088 = stablehlo.add %v1085, %v1087 : tensor<16xf32>
    %v1089 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1090 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1091 = stablehlo.divide %v1082, %v1089 : tensor<16xf32>
    %v1092 = stablehlo.divide %v1088, %v1090 : tensor<16xf32>
    %v1093 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1094 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1095 = stablehlo.sqrt %v1092 : tensor<16xf32>
    %v1096 = stablehlo.add %v1095, %v1094 : tensor<16xf32>
    %v1097 = stablehlo.divide %v1091, %v1096 : tensor<16xf32>
    %v1098 = stablehlo.multiply %v1093, %v1097 : tensor<16xf32>
    %v1099 = stablehlo.subtract %g3, %v1098 : tensor<16xf32>
    %v1100 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1101 = stablehlo.multiply %v1100, %v1093 : tensor<16xf32>
    %v1102 = stablehlo.multiply %v1101, %g3 : tensor<16xf32>
    %v1103 = stablehlo.subtract %v1099, %v1102 : tensor<16xf32>
    %v1104 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1105 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1106 = stablehlo.multiply %v1104, %g3m : tensor<16xf32>
    %v1107 = stablehlo.multiply %v1105, %v1077 : tensor<16xf32>
    %v1108 = stablehlo.add %v1106, %v1107 : tensor<16xf32>
    %v1109 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1110 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1111 = stablehlo.multiply %v1109, %g3v : tensor<16xf32>
    %v1112 = stablehlo.multiply %v1077, %v1077 : tensor<16xf32>
    %v1113 = stablehlo.multiply %v1110, %v1112 : tensor<16xf32>
    %v1114 = stablehlo.add %v1111, %v1113 : tensor<16xf32>
    %v1115 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1116 = stablehlo.reshape %v503 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1117 = stablehlo.reduce(%v1116 init: %v1115) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v1118 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1119 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1120 = stablehlo.multiply %v1118, %bt3m : tensor<16xf32>
    %v1121 = stablehlo.multiply %v1119, %v1117 : tensor<16xf32>
    %v1122 = stablehlo.add %v1120, %v1121 : tensor<16xf32>
    %v1123 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1124 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1125 = stablehlo.multiply %v1123, %bt3v : tensor<16xf32>
    %v1126 = stablehlo.multiply %v1117, %v1117 : tensor<16xf32>
    %v1127 = stablehlo.multiply %v1124, %v1126 : tensor<16xf32>
    %v1128 = stablehlo.add %v1125, %v1127 : tensor<16xf32>
    %v1129 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1130 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1131 = stablehlo.divide %v1122, %v1129 : tensor<16xf32>
    %v1132 = stablehlo.divide %v1128, %v1130 : tensor<16xf32>
    %v1133 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1134 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1135 = stablehlo.sqrt %v1132 : tensor<16xf32>
    %v1136 = stablehlo.add %v1135, %v1134 : tensor<16xf32>
    %v1137 = stablehlo.divide %v1131, %v1136 : tensor<16xf32>
    %v1138 = stablehlo.multiply %v1133, %v1137 : tensor<16xf32>
    %v1139 = stablehlo.subtract %bt3, %v1138 : tensor<16xf32>
    %v1140 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1141 = stablehlo.multiply %v1140, %v1133 : tensor<16xf32>
    %v1142 = stablehlo.multiply %v1141, %bt3 : tensor<16xf32>
    %v1143 = stablehlo.subtract %v1139, %v1142 : tensor<16xf32>
    %v1144 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1145 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1146 = stablehlo.multiply %v1144, %bt3m : tensor<16xf32>
    %v1147 = stablehlo.multiply %v1145, %v1117 : tensor<16xf32>
    %v1148 = stablehlo.add %v1146, %v1147 : tensor<16xf32>
    %v1149 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1150 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1151 = stablehlo.multiply %v1149, %bt3v : tensor<16xf32>
    %v1152 = stablehlo.multiply %v1117, %v1117 : tensor<16xf32>
    %v1153 = stablehlo.multiply %v1150, %v1152 : tensor<16xf32>
    %v1154 = stablehlo.add %v1151, %v1153 : tensor<16xf32>
    %v1155 = stablehlo.reshape %v90 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1156 = stablehlo.reshape %v492 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1157 = stablehlo.transpose %v1155, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v1158 = stablehlo.transpose %v1156, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v1159 = stablehlo.convolution(%v1157, %v1158)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v1160 = stablehlo.transpose %v1159, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v1161 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1162 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1163 = stablehlo.multiply %v1161, %W4m : tensor<16x16x3x3xf32>
    %v1164 = stablehlo.multiply %v1162, %v1160 : tensor<16x16x3x3xf32>
    %v1165 = stablehlo.add %v1163, %v1164 : tensor<16x16x3x3xf32>
    %v1166 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1167 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1168 = stablehlo.multiply %v1166, %W4v : tensor<16x16x3x3xf32>
    %v1169 = stablehlo.multiply %v1160, %v1160 : tensor<16x16x3x3xf32>
    %v1170 = stablehlo.multiply %v1167, %v1169 : tensor<16x16x3x3xf32>
    %v1171 = stablehlo.add %v1168, %v1170 : tensor<16x16x3x3xf32>
    %v1172 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1173 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1174 = stablehlo.divide %v1165, %v1172 : tensor<16x16x3x3xf32>
    %v1175 = stablehlo.divide %v1171, %v1173 : tensor<16x16x3x3xf32>
    %v1176 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1177 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1178 = stablehlo.sqrt %v1175 : tensor<16x16x3x3xf32>
    %v1179 = stablehlo.add %v1178, %v1177 : tensor<16x16x3x3xf32>
    %v1180 = stablehlo.divide %v1174, %v1179 : tensor<16x16x3x3xf32>
    %v1181 = stablehlo.multiply %v1176, %v1180 : tensor<16x16x3x3xf32>
    %v1182 = stablehlo.subtract %W4, %v1181 : tensor<16x16x3x3xf32>
    %v1183 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1184 = stablehlo.multiply %v1183, %v1176 : tensor<16x16x3x3xf32>
    %v1185 = stablehlo.multiply %v1184, %W4 : tensor<16x16x3x3xf32>
    %v1186 = stablehlo.subtract %v1182, %v1185 : tensor<16x16x3x3xf32>
    %v1187 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1188 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1189 = stablehlo.multiply %v1187, %W4m : tensor<16x16x3x3xf32>
    %v1190 = stablehlo.multiply %v1188, %v1160 : tensor<16x16x3x3xf32>
    %v1191 = stablehlo.add %v1189, %v1190 : tensor<16x16x3x3xf32>
    %v1192 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1193 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1194 = stablehlo.multiply %v1192, %W4v : tensor<16x16x3x3xf32>
    %v1195 = stablehlo.multiply %v1160, %v1160 : tensor<16x16x3x3xf32>
    %v1196 = stablehlo.multiply %v1193, %v1195 : tensor<16x16x3x3xf32>
    %v1197 = stablehlo.add %v1194, %v1196 : tensor<16x16x3x3xf32>
    %v1198 = stablehlo.reshape %v492 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1199 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1200 = stablehlo.reduce(%v1198 init: %v1199) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v1201 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1202 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1203 = stablehlo.multiply %v1201, %cb4m : tensor<16xf32>
    %v1204 = stablehlo.multiply %v1202, %v1200 : tensor<16xf32>
    %v1205 = stablehlo.add %v1203, %v1204 : tensor<16xf32>
    %v1206 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1207 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1208 = stablehlo.multiply %v1206, %cb4v : tensor<16xf32>
    %v1209 = stablehlo.multiply %v1200, %v1200 : tensor<16xf32>
    %v1210 = stablehlo.multiply %v1207, %v1209 : tensor<16xf32>
    %v1211 = stablehlo.add %v1208, %v1210 : tensor<16xf32>
    %v1212 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1213 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1214 = stablehlo.divide %v1205, %v1212 : tensor<16xf32>
    %v1215 = stablehlo.divide %v1211, %v1213 : tensor<16xf32>
    %v1216 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1217 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1218 = stablehlo.sqrt %v1215 : tensor<16xf32>
    %v1219 = stablehlo.add %v1218, %v1217 : tensor<16xf32>
    %v1220 = stablehlo.divide %v1214, %v1219 : tensor<16xf32>
    %v1221 = stablehlo.multiply %v1216, %v1220 : tensor<16xf32>
    %v1222 = stablehlo.subtract %cb4, %v1221 : tensor<16xf32>
    %v1223 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1224 = stablehlo.multiply %v1223, %v1216 : tensor<16xf32>
    %v1225 = stablehlo.multiply %v1224, %cb4 : tensor<16xf32>
    %v1226 = stablehlo.subtract %v1222, %v1225 : tensor<16xf32>
    %v1227 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1228 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1229 = stablehlo.multiply %v1227, %cb4m : tensor<16xf32>
    %v1230 = stablehlo.multiply %v1228, %v1200 : tensor<16xf32>
    %v1231 = stablehlo.add %v1229, %v1230 : tensor<16xf32>
    %v1232 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1233 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1234 = stablehlo.multiply %v1232, %cb4v : tensor<16xf32>
    %v1235 = stablehlo.multiply %v1200, %v1200 : tensor<16xf32>
    %v1236 = stablehlo.multiply %v1233, %v1235 : tensor<16xf32>
    %v1237 = stablehlo.add %v1234, %v1236 : tensor<16xf32>
    %v1238 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1239 = stablehlo.reshape %v95 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1240 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v1241 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v1242 = stablehlo.reduce(%v1239 init: %v1238) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v1243 = stablehlo.broadcast_in_dim %v1242, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v1244 = stablehlo.divide %v1243, %v1240 : tensor<128x16x16x16xf32>
    %v1245 = stablehlo.subtract %v1239, %v1244 : tensor<128x16x16x16xf32>
    %v1246 = stablehlo.multiply %v1245, %v1245 : tensor<128x16x16x16xf32>
    %v1247 = stablehlo.reduce(%v1246 init: %v1238) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v1248 = stablehlo.broadcast_in_dim %v1247, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v1249 = stablehlo.divide %v1248, %v1240 : tensor<128x16x16x16xf32>
    %v1250 = stablehlo.add %v1249, %v1241 : tensor<128x16x16x16xf32>
    %v1251 = stablehlo.rsqrt %v1250 : tensor<128x16x16x16xf32>
    %v1252 = stablehlo.multiply %v1245, %v1251 : tensor<128x16x16x16xf32>
    %v1253 = stablehlo.reshape %v462 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1254 = stablehlo.multiply %v1253, %v1252 : tensor<128x16x16x16xf32>
    %v1255 = stablehlo.reduce(%v1254 init: %v1238) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v1256 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1257 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1258 = stablehlo.multiply %v1256, %g4m : tensor<16xf32>
    %v1259 = stablehlo.multiply %v1257, %v1255 : tensor<16xf32>
    %v1260 = stablehlo.add %v1258, %v1259 : tensor<16xf32>
    %v1261 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1262 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1263 = stablehlo.multiply %v1261, %g4v : tensor<16xf32>
    %v1264 = stablehlo.multiply %v1255, %v1255 : tensor<16xf32>
    %v1265 = stablehlo.multiply %v1262, %v1264 : tensor<16xf32>
    %v1266 = stablehlo.add %v1263, %v1265 : tensor<16xf32>
    %v1267 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1268 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1269 = stablehlo.divide %v1260, %v1267 : tensor<16xf32>
    %v1270 = stablehlo.divide %v1266, %v1268 : tensor<16xf32>
    %v1271 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1272 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1273 = stablehlo.sqrt %v1270 : tensor<16xf32>
    %v1274 = stablehlo.add %v1273, %v1272 : tensor<16xf32>
    %v1275 = stablehlo.divide %v1269, %v1274 : tensor<16xf32>
    %v1276 = stablehlo.multiply %v1271, %v1275 : tensor<16xf32>
    %v1277 = stablehlo.subtract %g4, %v1276 : tensor<16xf32>
    %v1278 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1279 = stablehlo.multiply %v1278, %v1271 : tensor<16xf32>
    %v1280 = stablehlo.multiply %v1279, %g4 : tensor<16xf32>
    %v1281 = stablehlo.subtract %v1277, %v1280 : tensor<16xf32>
    %v1282 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1283 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1284 = stablehlo.multiply %v1282, %g4m : tensor<16xf32>
    %v1285 = stablehlo.multiply %v1283, %v1255 : tensor<16xf32>
    %v1286 = stablehlo.add %v1284, %v1285 : tensor<16xf32>
    %v1287 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1288 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1289 = stablehlo.multiply %v1287, %g4v : tensor<16xf32>
    %v1290 = stablehlo.multiply %v1255, %v1255 : tensor<16xf32>
    %v1291 = stablehlo.multiply %v1288, %v1290 : tensor<16xf32>
    %v1292 = stablehlo.add %v1289, %v1291 : tensor<16xf32>
    %v1293 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1294 = stablehlo.reshape %v462 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1295 = stablehlo.reduce(%v1294 init: %v1293) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v1296 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1297 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1298 = stablehlo.multiply %v1296, %bt4m : tensor<16xf32>
    %v1299 = stablehlo.multiply %v1297, %v1295 : tensor<16xf32>
    %v1300 = stablehlo.add %v1298, %v1299 : tensor<16xf32>
    %v1301 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1302 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1303 = stablehlo.multiply %v1301, %bt4v : tensor<16xf32>
    %v1304 = stablehlo.multiply %v1295, %v1295 : tensor<16xf32>
    %v1305 = stablehlo.multiply %v1302, %v1304 : tensor<16xf32>
    %v1306 = stablehlo.add %v1303, %v1305 : tensor<16xf32>
    %v1307 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1308 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1309 = stablehlo.divide %v1300, %v1307 : tensor<16xf32>
    %v1310 = stablehlo.divide %v1306, %v1308 : tensor<16xf32>
    %v1311 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1312 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1313 = stablehlo.sqrt %v1310 : tensor<16xf32>
    %v1314 = stablehlo.add %v1313, %v1312 : tensor<16xf32>
    %v1315 = stablehlo.divide %v1309, %v1314 : tensor<16xf32>
    %v1316 = stablehlo.multiply %v1311, %v1315 : tensor<16xf32>
    %v1317 = stablehlo.subtract %bt4, %v1316 : tensor<16xf32>
    %v1318 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1319 = stablehlo.multiply %v1318, %v1311 : tensor<16xf32>
    %v1320 = stablehlo.multiply %v1319, %bt4 : tensor<16xf32>
    %v1321 = stablehlo.subtract %v1317, %v1320 : tensor<16xf32>
    %v1322 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1323 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1324 = stablehlo.multiply %v1322, %bt4m : tensor<16xf32>
    %v1325 = stablehlo.multiply %v1323, %v1295 : tensor<16xf32>
    %v1326 = stablehlo.add %v1324, %v1325 : tensor<16xf32>
    %v1327 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1328 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1329 = stablehlo.multiply %v1327, %bt4v : tensor<16xf32>
    %v1330 = stablehlo.multiply %v1295, %v1295 : tensor<16xf32>
    %v1331 = stablehlo.multiply %v1328, %v1330 : tensor<16xf32>
    %v1332 = stablehlo.add %v1329, %v1331 : tensor<16xf32>
    %v1333 = stablehlo.reshape %v123 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v1334 = stablehlo.reshape %v446 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1335 = stablehlo.transpose %v1333, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v1336 = stablehlo.transpose %v1334, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v1337 = stablehlo.convolution(%v1335, %v1336)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<16x32x3x3xf32>
    %v1338 = stablehlo.transpose %v1337, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v1339 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1340 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1341 = stablehlo.multiply %v1339, %W5m : tensor<32x16x3x3xf32>
    %v1342 = stablehlo.multiply %v1340, %v1338 : tensor<32x16x3x3xf32>
    %v1343 = stablehlo.add %v1341, %v1342 : tensor<32x16x3x3xf32>
    %v1344 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1345 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1346 = stablehlo.multiply %v1344, %W5v : tensor<32x16x3x3xf32>
    %v1347 = stablehlo.multiply %v1338, %v1338 : tensor<32x16x3x3xf32>
    %v1348 = stablehlo.multiply %v1345, %v1347 : tensor<32x16x3x3xf32>
    %v1349 = stablehlo.add %v1346, %v1348 : tensor<32x16x3x3xf32>
    %v1350 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1351 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1352 = stablehlo.divide %v1343, %v1350 : tensor<32x16x3x3xf32>
    %v1353 = stablehlo.divide %v1349, %v1351 : tensor<32x16x3x3xf32>
    %v1354 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1355 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1356 = stablehlo.sqrt %v1353 : tensor<32x16x3x3xf32>
    %v1357 = stablehlo.add %v1356, %v1355 : tensor<32x16x3x3xf32>
    %v1358 = stablehlo.divide %v1352, %v1357 : tensor<32x16x3x3xf32>
    %v1359 = stablehlo.multiply %v1354, %v1358 : tensor<32x16x3x3xf32>
    %v1360 = stablehlo.subtract %W5, %v1359 : tensor<32x16x3x3xf32>
    %v1361 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1362 = stablehlo.multiply %v1361, %v1354 : tensor<32x16x3x3xf32>
    %v1363 = stablehlo.multiply %v1362, %W5 : tensor<32x16x3x3xf32>
    %v1364 = stablehlo.subtract %v1360, %v1363 : tensor<32x16x3x3xf32>
    %v1365 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1366 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1367 = stablehlo.multiply %v1365, %W5m : tensor<32x16x3x3xf32>
    %v1368 = stablehlo.multiply %v1366, %v1338 : tensor<32x16x3x3xf32>
    %v1369 = stablehlo.add %v1367, %v1368 : tensor<32x16x3x3xf32>
    %v1370 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1371 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1372 = stablehlo.multiply %v1370, %W5v : tensor<32x16x3x3xf32>
    %v1373 = stablehlo.multiply %v1338, %v1338 : tensor<32x16x3x3xf32>
    %v1374 = stablehlo.multiply %v1371, %v1373 : tensor<32x16x3x3xf32>
    %v1375 = stablehlo.add %v1372, %v1374 : tensor<32x16x3x3xf32>
    %v1376 = stablehlo.reshape %v446 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1377 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1378 = stablehlo.reduce(%v1376 init: %v1377) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1379 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1380 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1381 = stablehlo.multiply %v1379, %cb5m : tensor<32xf32>
    %v1382 = stablehlo.multiply %v1380, %v1378 : tensor<32xf32>
    %v1383 = stablehlo.add %v1381, %v1382 : tensor<32xf32>
    %v1384 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1385 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1386 = stablehlo.multiply %v1384, %cb5v : tensor<32xf32>
    %v1387 = stablehlo.multiply %v1378, %v1378 : tensor<32xf32>
    %v1388 = stablehlo.multiply %v1385, %v1387 : tensor<32xf32>
    %v1389 = stablehlo.add %v1386, %v1388 : tensor<32xf32>
    %v1390 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1391 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1392 = stablehlo.divide %v1383, %v1390 : tensor<32xf32>
    %v1393 = stablehlo.divide %v1389, %v1391 : tensor<32xf32>
    %v1394 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1395 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1396 = stablehlo.sqrt %v1393 : tensor<32xf32>
    %v1397 = stablehlo.add %v1396, %v1395 : tensor<32xf32>
    %v1398 = stablehlo.divide %v1392, %v1397 : tensor<32xf32>
    %v1399 = stablehlo.multiply %v1394, %v1398 : tensor<32xf32>
    %v1400 = stablehlo.subtract %cb5, %v1399 : tensor<32xf32>
    %v1401 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1402 = stablehlo.multiply %v1401, %v1394 : tensor<32xf32>
    %v1403 = stablehlo.multiply %v1402, %cb5 : tensor<32xf32>
    %v1404 = stablehlo.subtract %v1400, %v1403 : tensor<32xf32>
    %v1405 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1406 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1407 = stablehlo.multiply %v1405, %cb5m : tensor<32xf32>
    %v1408 = stablehlo.multiply %v1406, %v1378 : tensor<32xf32>
    %v1409 = stablehlo.add %v1407, %v1408 : tensor<32xf32>
    %v1410 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1411 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1412 = stablehlo.multiply %v1410, %cb5v : tensor<32xf32>
    %v1413 = stablehlo.multiply %v1378, %v1378 : tensor<32xf32>
    %v1414 = stablehlo.multiply %v1411, %v1413 : tensor<32xf32>
    %v1415 = stablehlo.add %v1412, %v1414 : tensor<32xf32>
    %v1416 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1417 = stablehlo.reshape %v128 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1418 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v1419 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v1420 = stablehlo.reduce(%v1417 init: %v1416) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1421 = stablehlo.broadcast_in_dim %v1420, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v1422 = stablehlo.divide %v1421, %v1418 : tensor<128x32x8x8xf32>
    %v1423 = stablehlo.subtract %v1417, %v1422 : tensor<128x32x8x8xf32>
    %v1424 = stablehlo.multiply %v1423, %v1423 : tensor<128x32x8x8xf32>
    %v1425 = stablehlo.reduce(%v1424 init: %v1416) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1426 = stablehlo.broadcast_in_dim %v1425, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v1427 = stablehlo.divide %v1426, %v1418 : tensor<128x32x8x8xf32>
    %v1428 = stablehlo.add %v1427, %v1419 : tensor<128x32x8x8xf32>
    %v1429 = stablehlo.rsqrt %v1428 : tensor<128x32x8x8xf32>
    %v1430 = stablehlo.multiply %v1423, %v1429 : tensor<128x32x8x8xf32>
    %v1431 = stablehlo.reshape %v416 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1432 = stablehlo.multiply %v1431, %v1430 : tensor<128x32x8x8xf32>
    %v1433 = stablehlo.reduce(%v1432 init: %v1416) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1434 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1435 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1436 = stablehlo.multiply %v1434, %g5m : tensor<32xf32>
    %v1437 = stablehlo.multiply %v1435, %v1433 : tensor<32xf32>
    %v1438 = stablehlo.add %v1436, %v1437 : tensor<32xf32>
    %v1439 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1440 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1441 = stablehlo.multiply %v1439, %g5v : tensor<32xf32>
    %v1442 = stablehlo.multiply %v1433, %v1433 : tensor<32xf32>
    %v1443 = stablehlo.multiply %v1440, %v1442 : tensor<32xf32>
    %v1444 = stablehlo.add %v1441, %v1443 : tensor<32xf32>
    %v1445 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1446 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1447 = stablehlo.divide %v1438, %v1445 : tensor<32xf32>
    %v1448 = stablehlo.divide %v1444, %v1446 : tensor<32xf32>
    %v1449 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1450 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1451 = stablehlo.sqrt %v1448 : tensor<32xf32>
    %v1452 = stablehlo.add %v1451, %v1450 : tensor<32xf32>
    %v1453 = stablehlo.divide %v1447, %v1452 : tensor<32xf32>
    %v1454 = stablehlo.multiply %v1449, %v1453 : tensor<32xf32>
    %v1455 = stablehlo.subtract %g5, %v1454 : tensor<32xf32>
    %v1456 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1457 = stablehlo.multiply %v1456, %v1449 : tensor<32xf32>
    %v1458 = stablehlo.multiply %v1457, %g5 : tensor<32xf32>
    %v1459 = stablehlo.subtract %v1455, %v1458 : tensor<32xf32>
    %v1460 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1461 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1462 = stablehlo.multiply %v1460, %g5m : tensor<32xf32>
    %v1463 = stablehlo.multiply %v1461, %v1433 : tensor<32xf32>
    %v1464 = stablehlo.add %v1462, %v1463 : tensor<32xf32>
    %v1465 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1466 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1467 = stablehlo.multiply %v1465, %g5v : tensor<32xf32>
    %v1468 = stablehlo.multiply %v1433, %v1433 : tensor<32xf32>
    %v1469 = stablehlo.multiply %v1466, %v1468 : tensor<32xf32>
    %v1470 = stablehlo.add %v1467, %v1469 : tensor<32xf32>
    %v1471 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1472 = stablehlo.reshape %v416 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1473 = stablehlo.reduce(%v1472 init: %v1471) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1474 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1475 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1476 = stablehlo.multiply %v1474, %bt5m : tensor<32xf32>
    %v1477 = stablehlo.multiply %v1475, %v1473 : tensor<32xf32>
    %v1478 = stablehlo.add %v1476, %v1477 : tensor<32xf32>
    %v1479 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1480 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1481 = stablehlo.multiply %v1479, %bt5v : tensor<32xf32>
    %v1482 = stablehlo.multiply %v1473, %v1473 : tensor<32xf32>
    %v1483 = stablehlo.multiply %v1480, %v1482 : tensor<32xf32>
    %v1484 = stablehlo.add %v1481, %v1483 : tensor<32xf32>
    %v1485 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1486 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1487 = stablehlo.divide %v1478, %v1485 : tensor<32xf32>
    %v1488 = stablehlo.divide %v1484, %v1486 : tensor<32xf32>
    %v1489 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1490 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1491 = stablehlo.sqrt %v1488 : tensor<32xf32>
    %v1492 = stablehlo.add %v1491, %v1490 : tensor<32xf32>
    %v1493 = stablehlo.divide %v1487, %v1492 : tensor<32xf32>
    %v1494 = stablehlo.multiply %v1489, %v1493 : tensor<32xf32>
    %v1495 = stablehlo.subtract %bt5, %v1494 : tensor<32xf32>
    %v1496 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1497 = stablehlo.multiply %v1496, %v1489 : tensor<32xf32>
    %v1498 = stablehlo.multiply %v1497, %bt5 : tensor<32xf32>
    %v1499 = stablehlo.subtract %v1495, %v1498 : tensor<32xf32>
    %v1500 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1501 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1502 = stablehlo.multiply %v1500, %bt5m : tensor<32xf32>
    %v1503 = stablehlo.multiply %v1501, %v1473 : tensor<32xf32>
    %v1504 = stablehlo.add %v1502, %v1503 : tensor<32xf32>
    %v1505 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1506 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1507 = stablehlo.multiply %v1505, %bt5v : tensor<32xf32>
    %v1508 = stablehlo.multiply %v1473, %v1473 : tensor<32xf32>
    %v1509 = stablehlo.multiply %v1506, %v1508 : tensor<32xf32>
    %v1510 = stablehlo.add %v1507, %v1509 : tensor<32xf32>
    %v1511 = stablehlo.reshape %v152 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1512 = stablehlo.reshape %v405 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1513 = stablehlo.transpose %v1511, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v1514 = stablehlo.transpose %v1512, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v1515 = stablehlo.convolution(%v1513, %v1514)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<32x32x3x3xf32>
    %v1516 = stablehlo.transpose %v1515, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v1517 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1518 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1519 = stablehlo.multiply %v1517, %W6m : tensor<32x32x3x3xf32>
    %v1520 = stablehlo.multiply %v1518, %v1516 : tensor<32x32x3x3xf32>
    %v1521 = stablehlo.add %v1519, %v1520 : tensor<32x32x3x3xf32>
    %v1522 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1523 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1524 = stablehlo.multiply %v1522, %W6v : tensor<32x32x3x3xf32>
    %v1525 = stablehlo.multiply %v1516, %v1516 : tensor<32x32x3x3xf32>
    %v1526 = stablehlo.multiply %v1523, %v1525 : tensor<32x32x3x3xf32>
    %v1527 = stablehlo.add %v1524, %v1526 : tensor<32x32x3x3xf32>
    %v1528 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1529 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1530 = stablehlo.divide %v1521, %v1528 : tensor<32x32x3x3xf32>
    %v1531 = stablehlo.divide %v1527, %v1529 : tensor<32x32x3x3xf32>
    %v1532 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1533 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1534 = stablehlo.sqrt %v1531 : tensor<32x32x3x3xf32>
    %v1535 = stablehlo.add %v1534, %v1533 : tensor<32x32x3x3xf32>
    %v1536 = stablehlo.divide %v1530, %v1535 : tensor<32x32x3x3xf32>
    %v1537 = stablehlo.multiply %v1532, %v1536 : tensor<32x32x3x3xf32>
    %v1538 = stablehlo.subtract %W6, %v1537 : tensor<32x32x3x3xf32>
    %v1539 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1540 = stablehlo.multiply %v1539, %v1532 : tensor<32x32x3x3xf32>
    %v1541 = stablehlo.multiply %v1540, %W6 : tensor<32x32x3x3xf32>
    %v1542 = stablehlo.subtract %v1538, %v1541 : tensor<32x32x3x3xf32>
    %v1543 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1544 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1545 = stablehlo.multiply %v1543, %W6m : tensor<32x32x3x3xf32>
    %v1546 = stablehlo.multiply %v1544, %v1516 : tensor<32x32x3x3xf32>
    %v1547 = stablehlo.add %v1545, %v1546 : tensor<32x32x3x3xf32>
    %v1548 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1549 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1550 = stablehlo.multiply %v1548, %W6v : tensor<32x32x3x3xf32>
    %v1551 = stablehlo.multiply %v1516, %v1516 : tensor<32x32x3x3xf32>
    %v1552 = stablehlo.multiply %v1549, %v1551 : tensor<32x32x3x3xf32>
    %v1553 = stablehlo.add %v1550, %v1552 : tensor<32x32x3x3xf32>
    %v1554 = stablehlo.reshape %v405 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1555 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1556 = stablehlo.reduce(%v1554 init: %v1555) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1557 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1558 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1559 = stablehlo.multiply %v1557, %cb6m : tensor<32xf32>
    %v1560 = stablehlo.multiply %v1558, %v1556 : tensor<32xf32>
    %v1561 = stablehlo.add %v1559, %v1560 : tensor<32xf32>
    %v1562 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1563 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1564 = stablehlo.multiply %v1562, %cb6v : tensor<32xf32>
    %v1565 = stablehlo.multiply %v1556, %v1556 : tensor<32xf32>
    %v1566 = stablehlo.multiply %v1563, %v1565 : tensor<32xf32>
    %v1567 = stablehlo.add %v1564, %v1566 : tensor<32xf32>
    %v1568 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1569 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1570 = stablehlo.divide %v1561, %v1568 : tensor<32xf32>
    %v1571 = stablehlo.divide %v1567, %v1569 : tensor<32xf32>
    %v1572 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1573 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1574 = stablehlo.sqrt %v1571 : tensor<32xf32>
    %v1575 = stablehlo.add %v1574, %v1573 : tensor<32xf32>
    %v1576 = stablehlo.divide %v1570, %v1575 : tensor<32xf32>
    %v1577 = stablehlo.multiply %v1572, %v1576 : tensor<32xf32>
    %v1578 = stablehlo.subtract %cb6, %v1577 : tensor<32xf32>
    %v1579 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1580 = stablehlo.multiply %v1579, %v1572 : tensor<32xf32>
    %v1581 = stablehlo.multiply %v1580, %cb6 : tensor<32xf32>
    %v1582 = stablehlo.subtract %v1578, %v1581 : tensor<32xf32>
    %v1583 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1584 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1585 = stablehlo.multiply %v1583, %cb6m : tensor<32xf32>
    %v1586 = stablehlo.multiply %v1584, %v1556 : tensor<32xf32>
    %v1587 = stablehlo.add %v1585, %v1586 : tensor<32xf32>
    %v1588 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1589 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1590 = stablehlo.multiply %v1588, %cb6v : tensor<32xf32>
    %v1591 = stablehlo.multiply %v1556, %v1556 : tensor<32xf32>
    %v1592 = stablehlo.multiply %v1589, %v1591 : tensor<32xf32>
    %v1593 = stablehlo.add %v1590, %v1592 : tensor<32xf32>
    %v1594 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1595 = stablehlo.reshape %v157 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1596 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v1597 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v1598 = stablehlo.reduce(%v1595 init: %v1594) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1599 = stablehlo.broadcast_in_dim %v1598, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v1600 = stablehlo.divide %v1599, %v1596 : tensor<128x32x8x8xf32>
    %v1601 = stablehlo.subtract %v1595, %v1600 : tensor<128x32x8x8xf32>
    %v1602 = stablehlo.multiply %v1601, %v1601 : tensor<128x32x8x8xf32>
    %v1603 = stablehlo.reduce(%v1602 init: %v1594) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1604 = stablehlo.broadcast_in_dim %v1603, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v1605 = stablehlo.divide %v1604, %v1596 : tensor<128x32x8x8xf32>
    %v1606 = stablehlo.add %v1605, %v1597 : tensor<128x32x8x8xf32>
    %v1607 = stablehlo.rsqrt %v1606 : tensor<128x32x8x8xf32>
    %v1608 = stablehlo.multiply %v1601, %v1607 : tensor<128x32x8x8xf32>
    %v1609 = stablehlo.reshape %v375 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1610 = stablehlo.multiply %v1609, %v1608 : tensor<128x32x8x8xf32>
    %v1611 = stablehlo.reduce(%v1610 init: %v1594) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1612 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1613 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1614 = stablehlo.multiply %v1612, %g6m : tensor<32xf32>
    %v1615 = stablehlo.multiply %v1613, %v1611 : tensor<32xf32>
    %v1616 = stablehlo.add %v1614, %v1615 : tensor<32xf32>
    %v1617 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1618 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1619 = stablehlo.multiply %v1617, %g6v : tensor<32xf32>
    %v1620 = stablehlo.multiply %v1611, %v1611 : tensor<32xf32>
    %v1621 = stablehlo.multiply %v1618, %v1620 : tensor<32xf32>
    %v1622 = stablehlo.add %v1619, %v1621 : tensor<32xf32>
    %v1623 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1624 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1625 = stablehlo.divide %v1616, %v1623 : tensor<32xf32>
    %v1626 = stablehlo.divide %v1622, %v1624 : tensor<32xf32>
    %v1627 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1628 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1629 = stablehlo.sqrt %v1626 : tensor<32xf32>
    %v1630 = stablehlo.add %v1629, %v1628 : tensor<32xf32>
    %v1631 = stablehlo.divide %v1625, %v1630 : tensor<32xf32>
    %v1632 = stablehlo.multiply %v1627, %v1631 : tensor<32xf32>
    %v1633 = stablehlo.subtract %g6, %v1632 : tensor<32xf32>
    %v1634 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1635 = stablehlo.multiply %v1634, %v1627 : tensor<32xf32>
    %v1636 = stablehlo.multiply %v1635, %g6 : tensor<32xf32>
    %v1637 = stablehlo.subtract %v1633, %v1636 : tensor<32xf32>
    %v1638 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1639 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1640 = stablehlo.multiply %v1638, %g6m : tensor<32xf32>
    %v1641 = stablehlo.multiply %v1639, %v1611 : tensor<32xf32>
    %v1642 = stablehlo.add %v1640, %v1641 : tensor<32xf32>
    %v1643 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1644 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1645 = stablehlo.multiply %v1643, %g6v : tensor<32xf32>
    %v1646 = stablehlo.multiply %v1611, %v1611 : tensor<32xf32>
    %v1647 = stablehlo.multiply %v1644, %v1646 : tensor<32xf32>
    %v1648 = stablehlo.add %v1645, %v1647 : tensor<32xf32>
    %v1649 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1650 = stablehlo.reshape %v375 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1651 = stablehlo.reduce(%v1650 init: %v1649) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1652 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1653 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1654 = stablehlo.multiply %v1652, %bt6m : tensor<32xf32>
    %v1655 = stablehlo.multiply %v1653, %v1651 : tensor<32xf32>
    %v1656 = stablehlo.add %v1654, %v1655 : tensor<32xf32>
    %v1657 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1658 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1659 = stablehlo.multiply %v1657, %bt6v : tensor<32xf32>
    %v1660 = stablehlo.multiply %v1651, %v1651 : tensor<32xf32>
    %v1661 = stablehlo.multiply %v1658, %v1660 : tensor<32xf32>
    %v1662 = stablehlo.add %v1659, %v1661 : tensor<32xf32>
    %v1663 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1664 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1665 = stablehlo.divide %v1656, %v1663 : tensor<32xf32>
    %v1666 = stablehlo.divide %v1662, %v1664 : tensor<32xf32>
    %v1667 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1668 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1669 = stablehlo.sqrt %v1666 : tensor<32xf32>
    %v1670 = stablehlo.add %v1669, %v1668 : tensor<32xf32>
    %v1671 = stablehlo.divide %v1665, %v1670 : tensor<32xf32>
    %v1672 = stablehlo.multiply %v1667, %v1671 : tensor<32xf32>
    %v1673 = stablehlo.subtract %bt6, %v1672 : tensor<32xf32>
    %v1674 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1675 = stablehlo.multiply %v1674, %v1667 : tensor<32xf32>
    %v1676 = stablehlo.multiply %v1675, %bt6 : tensor<32xf32>
    %v1677 = stablehlo.subtract %v1673, %v1676 : tensor<32xf32>
    %v1678 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1679 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1680 = stablehlo.multiply %v1678, %bt6m : tensor<32xf32>
    %v1681 = stablehlo.multiply %v1679, %v1651 : tensor<32xf32>
    %v1682 = stablehlo.add %v1680, %v1681 : tensor<32xf32>
    %v1683 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1684 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1685 = stablehlo.multiply %v1683, %bt6v : tensor<32xf32>
    %v1686 = stablehlo.multiply %v1651, %v1651 : tensor<32xf32>
    %v1687 = stablehlo.multiply %v1684, %v1686 : tensor<32xf32>
    %v1688 = stablehlo.add %v1685, %v1687 : tensor<32xf32>
    %v1689 = stablehlo.reshape %v185 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1690 = stablehlo.reshape %v359 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1691 = stablehlo.transpose %v1689, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v1692 = stablehlo.transpose %v1690, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v1693 = stablehlo.convolution(%v1691, %v1692)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v1694 = stablehlo.transpose %v1693, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v1695 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1696 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1697 = stablehlo.multiply %v1695, %W7m : tensor<32x32x3x3xf32>
    %v1698 = stablehlo.multiply %v1696, %v1694 : tensor<32x32x3x3xf32>
    %v1699 = stablehlo.add %v1697, %v1698 : tensor<32x32x3x3xf32>
    %v1700 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1701 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1702 = stablehlo.multiply %v1700, %W7v : tensor<32x32x3x3xf32>
    %v1703 = stablehlo.multiply %v1694, %v1694 : tensor<32x32x3x3xf32>
    %v1704 = stablehlo.multiply %v1701, %v1703 : tensor<32x32x3x3xf32>
    %v1705 = stablehlo.add %v1702, %v1704 : tensor<32x32x3x3xf32>
    %v1706 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1707 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1708 = stablehlo.divide %v1699, %v1706 : tensor<32x32x3x3xf32>
    %v1709 = stablehlo.divide %v1705, %v1707 : tensor<32x32x3x3xf32>
    %v1710 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1711 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1712 = stablehlo.sqrt %v1709 : tensor<32x32x3x3xf32>
    %v1713 = stablehlo.add %v1712, %v1711 : tensor<32x32x3x3xf32>
    %v1714 = stablehlo.divide %v1708, %v1713 : tensor<32x32x3x3xf32>
    %v1715 = stablehlo.multiply %v1710, %v1714 : tensor<32x32x3x3xf32>
    %v1716 = stablehlo.subtract %W7, %v1715 : tensor<32x32x3x3xf32>
    %v1717 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1718 = stablehlo.multiply %v1717, %v1710 : tensor<32x32x3x3xf32>
    %v1719 = stablehlo.multiply %v1718, %W7 : tensor<32x32x3x3xf32>
    %v1720 = stablehlo.subtract %v1716, %v1719 : tensor<32x32x3x3xf32>
    %v1721 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1722 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1723 = stablehlo.multiply %v1721, %W7m : tensor<32x32x3x3xf32>
    %v1724 = stablehlo.multiply %v1722, %v1694 : tensor<32x32x3x3xf32>
    %v1725 = stablehlo.add %v1723, %v1724 : tensor<32x32x3x3xf32>
    %v1726 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1727 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1728 = stablehlo.multiply %v1726, %W7v : tensor<32x32x3x3xf32>
    %v1729 = stablehlo.multiply %v1694, %v1694 : tensor<32x32x3x3xf32>
    %v1730 = stablehlo.multiply %v1727, %v1729 : tensor<32x32x3x3xf32>
    %v1731 = stablehlo.add %v1728, %v1730 : tensor<32x32x3x3xf32>
    %v1732 = stablehlo.reshape %v359 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1733 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1734 = stablehlo.reduce(%v1732 init: %v1733) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1735 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1736 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1737 = stablehlo.multiply %v1735, %cb7m : tensor<32xf32>
    %v1738 = stablehlo.multiply %v1736, %v1734 : tensor<32xf32>
    %v1739 = stablehlo.add %v1737, %v1738 : tensor<32xf32>
    %v1740 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1741 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1742 = stablehlo.multiply %v1740, %cb7v : tensor<32xf32>
    %v1743 = stablehlo.multiply %v1734, %v1734 : tensor<32xf32>
    %v1744 = stablehlo.multiply %v1741, %v1743 : tensor<32xf32>
    %v1745 = stablehlo.add %v1742, %v1744 : tensor<32xf32>
    %v1746 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1747 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1748 = stablehlo.divide %v1739, %v1746 : tensor<32xf32>
    %v1749 = stablehlo.divide %v1745, %v1747 : tensor<32xf32>
    %v1750 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1751 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1752 = stablehlo.sqrt %v1749 : tensor<32xf32>
    %v1753 = stablehlo.add %v1752, %v1751 : tensor<32xf32>
    %v1754 = stablehlo.divide %v1748, %v1753 : tensor<32xf32>
    %v1755 = stablehlo.multiply %v1750, %v1754 : tensor<32xf32>
    %v1756 = stablehlo.subtract %cb7, %v1755 : tensor<32xf32>
    %v1757 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1758 = stablehlo.multiply %v1757, %v1750 : tensor<32xf32>
    %v1759 = stablehlo.multiply %v1758, %cb7 : tensor<32xf32>
    %v1760 = stablehlo.subtract %v1756, %v1759 : tensor<32xf32>
    %v1761 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1762 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1763 = stablehlo.multiply %v1761, %cb7m : tensor<32xf32>
    %v1764 = stablehlo.multiply %v1762, %v1734 : tensor<32xf32>
    %v1765 = stablehlo.add %v1763, %v1764 : tensor<32xf32>
    %v1766 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1767 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1768 = stablehlo.multiply %v1766, %cb7v : tensor<32xf32>
    %v1769 = stablehlo.multiply %v1734, %v1734 : tensor<32xf32>
    %v1770 = stablehlo.multiply %v1767, %v1769 : tensor<32xf32>
    %v1771 = stablehlo.add %v1768, %v1770 : tensor<32xf32>
    %v1772 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1773 = stablehlo.reshape %v190 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1774 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v1775 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v1776 = stablehlo.reduce(%v1773 init: %v1772) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1777 = stablehlo.broadcast_in_dim %v1776, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v1778 = stablehlo.divide %v1777, %v1774 : tensor<128x32x4x4xf32>
    %v1779 = stablehlo.subtract %v1773, %v1778 : tensor<128x32x4x4xf32>
    %v1780 = stablehlo.multiply %v1779, %v1779 : tensor<128x32x4x4xf32>
    %v1781 = stablehlo.reduce(%v1780 init: %v1772) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1782 = stablehlo.broadcast_in_dim %v1781, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v1783 = stablehlo.divide %v1782, %v1774 : tensor<128x32x4x4xf32>
    %v1784 = stablehlo.add %v1783, %v1775 : tensor<128x32x4x4xf32>
    %v1785 = stablehlo.rsqrt %v1784 : tensor<128x32x4x4xf32>
    %v1786 = stablehlo.multiply %v1779, %v1785 : tensor<128x32x4x4xf32>
    %v1787 = stablehlo.reshape %v329 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1788 = stablehlo.multiply %v1787, %v1786 : tensor<128x32x4x4xf32>
    %v1789 = stablehlo.reduce(%v1788 init: %v1772) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1790 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1791 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1792 = stablehlo.multiply %v1790, %g7m : tensor<32xf32>
    %v1793 = stablehlo.multiply %v1791, %v1789 : tensor<32xf32>
    %v1794 = stablehlo.add %v1792, %v1793 : tensor<32xf32>
    %v1795 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1796 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1797 = stablehlo.multiply %v1795, %g7v : tensor<32xf32>
    %v1798 = stablehlo.multiply %v1789, %v1789 : tensor<32xf32>
    %v1799 = stablehlo.multiply %v1796, %v1798 : tensor<32xf32>
    %v1800 = stablehlo.add %v1797, %v1799 : tensor<32xf32>
    %v1801 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1802 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1803 = stablehlo.divide %v1794, %v1801 : tensor<32xf32>
    %v1804 = stablehlo.divide %v1800, %v1802 : tensor<32xf32>
    %v1805 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1806 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1807 = stablehlo.sqrt %v1804 : tensor<32xf32>
    %v1808 = stablehlo.add %v1807, %v1806 : tensor<32xf32>
    %v1809 = stablehlo.divide %v1803, %v1808 : tensor<32xf32>
    %v1810 = stablehlo.multiply %v1805, %v1809 : tensor<32xf32>
    %v1811 = stablehlo.subtract %g7, %v1810 : tensor<32xf32>
    %v1812 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1813 = stablehlo.multiply %v1812, %v1805 : tensor<32xf32>
    %v1814 = stablehlo.multiply %v1813, %g7 : tensor<32xf32>
    %v1815 = stablehlo.subtract %v1811, %v1814 : tensor<32xf32>
    %v1816 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1817 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1818 = stablehlo.multiply %v1816, %g7m : tensor<32xf32>
    %v1819 = stablehlo.multiply %v1817, %v1789 : tensor<32xf32>
    %v1820 = stablehlo.add %v1818, %v1819 : tensor<32xf32>
    %v1821 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1822 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1823 = stablehlo.multiply %v1821, %g7v : tensor<32xf32>
    %v1824 = stablehlo.multiply %v1789, %v1789 : tensor<32xf32>
    %v1825 = stablehlo.multiply %v1822, %v1824 : tensor<32xf32>
    %v1826 = stablehlo.add %v1823, %v1825 : tensor<32xf32>
    %v1827 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1828 = stablehlo.reshape %v329 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1829 = stablehlo.reduce(%v1828 init: %v1827) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1830 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1831 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1832 = stablehlo.multiply %v1830, %bt7m : tensor<32xf32>
    %v1833 = stablehlo.multiply %v1831, %v1829 : tensor<32xf32>
    %v1834 = stablehlo.add %v1832, %v1833 : tensor<32xf32>
    %v1835 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1836 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1837 = stablehlo.multiply %v1835, %bt7v : tensor<32xf32>
    %v1838 = stablehlo.multiply %v1829, %v1829 : tensor<32xf32>
    %v1839 = stablehlo.multiply %v1836, %v1838 : tensor<32xf32>
    %v1840 = stablehlo.add %v1837, %v1839 : tensor<32xf32>
    %v1841 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1842 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1843 = stablehlo.divide %v1834, %v1841 : tensor<32xf32>
    %v1844 = stablehlo.divide %v1840, %v1842 : tensor<32xf32>
    %v1845 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1846 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1847 = stablehlo.sqrt %v1844 : tensor<32xf32>
    %v1848 = stablehlo.add %v1847, %v1846 : tensor<32xf32>
    %v1849 = stablehlo.divide %v1843, %v1848 : tensor<32xf32>
    %v1850 = stablehlo.multiply %v1845, %v1849 : tensor<32xf32>
    %v1851 = stablehlo.subtract %bt7, %v1850 : tensor<32xf32>
    %v1852 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1853 = stablehlo.multiply %v1852, %v1845 : tensor<32xf32>
    %v1854 = stablehlo.multiply %v1853, %bt7 : tensor<32xf32>
    %v1855 = stablehlo.subtract %v1851, %v1854 : tensor<32xf32>
    %v1856 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1857 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1858 = stablehlo.multiply %v1856, %bt7m : tensor<32xf32>
    %v1859 = stablehlo.multiply %v1857, %v1829 : tensor<32xf32>
    %v1860 = stablehlo.add %v1858, %v1859 : tensor<32xf32>
    %v1861 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1862 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1863 = stablehlo.multiply %v1861, %bt7v : tensor<32xf32>
    %v1864 = stablehlo.multiply %v1829, %v1829 : tensor<32xf32>
    %v1865 = stablehlo.multiply %v1862, %v1864 : tensor<32xf32>
    %v1866 = stablehlo.add %v1863, %v1865 : tensor<32xf32>
    %v1867 = stablehlo.reshape %v214 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1868 = stablehlo.reshape %v318 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1869 = stablehlo.transpose %v1867, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v1870 = stablehlo.transpose %v1868, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v1871 = stablehlo.convolution(%v1869, %v1870)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v1872 = stablehlo.transpose %v1871, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v1873 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1874 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1875 = stablehlo.multiply %v1873, %W8m : tensor<32x32x3x3xf32>
    %v1876 = stablehlo.multiply %v1874, %v1872 : tensor<32x32x3x3xf32>
    %v1877 = stablehlo.add %v1875, %v1876 : tensor<32x32x3x3xf32>
    %v1878 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1879 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1880 = stablehlo.multiply %v1878, %W8v : tensor<32x32x3x3xf32>
    %v1881 = stablehlo.multiply %v1872, %v1872 : tensor<32x32x3x3xf32>
    %v1882 = stablehlo.multiply %v1879, %v1881 : tensor<32x32x3x3xf32>
    %v1883 = stablehlo.add %v1880, %v1882 : tensor<32x32x3x3xf32>
    %v1884 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1885 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1886 = stablehlo.divide %v1877, %v1884 : tensor<32x32x3x3xf32>
    %v1887 = stablehlo.divide %v1883, %v1885 : tensor<32x32x3x3xf32>
    %v1888 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1889 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1890 = stablehlo.sqrt %v1887 : tensor<32x32x3x3xf32>
    %v1891 = stablehlo.add %v1890, %v1889 : tensor<32x32x3x3xf32>
    %v1892 = stablehlo.divide %v1886, %v1891 : tensor<32x32x3x3xf32>
    %v1893 = stablehlo.multiply %v1888, %v1892 : tensor<32x32x3x3xf32>
    %v1894 = stablehlo.subtract %W8, %v1893 : tensor<32x32x3x3xf32>
    %v1895 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1896 = stablehlo.multiply %v1895, %v1888 : tensor<32x32x3x3xf32>
    %v1897 = stablehlo.multiply %v1896, %W8 : tensor<32x32x3x3xf32>
    %v1898 = stablehlo.subtract %v1894, %v1897 : tensor<32x32x3x3xf32>
    %v1899 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1900 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1901 = stablehlo.multiply %v1899, %W8m : tensor<32x32x3x3xf32>
    %v1902 = stablehlo.multiply %v1900, %v1872 : tensor<32x32x3x3xf32>
    %v1903 = stablehlo.add %v1901, %v1902 : tensor<32x32x3x3xf32>
    %v1904 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1905 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1906 = stablehlo.multiply %v1904, %W8v : tensor<32x32x3x3xf32>
    %v1907 = stablehlo.multiply %v1872, %v1872 : tensor<32x32x3x3xf32>
    %v1908 = stablehlo.multiply %v1905, %v1907 : tensor<32x32x3x3xf32>
    %v1909 = stablehlo.add %v1906, %v1908 : tensor<32x32x3x3xf32>
    %v1910 = stablehlo.reshape %v318 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1911 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1912 = stablehlo.reduce(%v1910 init: %v1911) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1913 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1914 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1915 = stablehlo.multiply %v1913, %cb8m : tensor<32xf32>
    %v1916 = stablehlo.multiply %v1914, %v1912 : tensor<32xf32>
    %v1917 = stablehlo.add %v1915, %v1916 : tensor<32xf32>
    %v1918 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1919 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1920 = stablehlo.multiply %v1918, %cb8v : tensor<32xf32>
    %v1921 = stablehlo.multiply %v1912, %v1912 : tensor<32xf32>
    %v1922 = stablehlo.multiply %v1919, %v1921 : tensor<32xf32>
    %v1923 = stablehlo.add %v1920, %v1922 : tensor<32xf32>
    %v1924 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1925 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1926 = stablehlo.divide %v1917, %v1924 : tensor<32xf32>
    %v1927 = stablehlo.divide %v1923, %v1925 : tensor<32xf32>
    %v1928 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1929 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1930 = stablehlo.sqrt %v1927 : tensor<32xf32>
    %v1931 = stablehlo.add %v1930, %v1929 : tensor<32xf32>
    %v1932 = stablehlo.divide %v1926, %v1931 : tensor<32xf32>
    %v1933 = stablehlo.multiply %v1928, %v1932 : tensor<32xf32>
    %v1934 = stablehlo.subtract %cb8, %v1933 : tensor<32xf32>
    %v1935 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1936 = stablehlo.multiply %v1935, %v1928 : tensor<32xf32>
    %v1937 = stablehlo.multiply %v1936, %cb8 : tensor<32xf32>
    %v1938 = stablehlo.subtract %v1934, %v1937 : tensor<32xf32>
    %v1939 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1940 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1941 = stablehlo.multiply %v1939, %cb8m : tensor<32xf32>
    %v1942 = stablehlo.multiply %v1940, %v1912 : tensor<32xf32>
    %v1943 = stablehlo.add %v1941, %v1942 : tensor<32xf32>
    %v1944 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1945 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1946 = stablehlo.multiply %v1944, %cb8v : tensor<32xf32>
    %v1947 = stablehlo.multiply %v1912, %v1912 : tensor<32xf32>
    %v1948 = stablehlo.multiply %v1945, %v1947 : tensor<32xf32>
    %v1949 = stablehlo.add %v1946, %v1948 : tensor<32xf32>
    %v1950 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1951 = stablehlo.reshape %v219 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1952 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v1953 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v1954 = stablehlo.reduce(%v1951 init: %v1950) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1955 = stablehlo.broadcast_in_dim %v1954, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v1956 = stablehlo.divide %v1955, %v1952 : tensor<128x32x4x4xf32>
    %v1957 = stablehlo.subtract %v1951, %v1956 : tensor<128x32x4x4xf32>
    %v1958 = stablehlo.multiply %v1957, %v1957 : tensor<128x32x4x4xf32>
    %v1959 = stablehlo.reduce(%v1958 init: %v1950) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1960 = stablehlo.broadcast_in_dim %v1959, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v1961 = stablehlo.divide %v1960, %v1952 : tensor<128x32x4x4xf32>
    %v1962 = stablehlo.add %v1961, %v1953 : tensor<128x32x4x4xf32>
    %v1963 = stablehlo.rsqrt %v1962 : tensor<128x32x4x4xf32>
    %v1964 = stablehlo.multiply %v1957, %v1963 : tensor<128x32x4x4xf32>
    %v1965 = stablehlo.reshape %v288 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1966 = stablehlo.multiply %v1965, %v1964 : tensor<128x32x4x4xf32>
    %v1967 = stablehlo.reduce(%v1966 init: %v1950) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1968 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1969 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1970 = stablehlo.multiply %v1968, %g8m : tensor<32xf32>
    %v1971 = stablehlo.multiply %v1969, %v1967 : tensor<32xf32>
    %v1972 = stablehlo.add %v1970, %v1971 : tensor<32xf32>
    %v1973 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1974 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1975 = stablehlo.multiply %v1973, %g8v : tensor<32xf32>
    %v1976 = stablehlo.multiply %v1967, %v1967 : tensor<32xf32>
    %v1977 = stablehlo.multiply %v1974, %v1976 : tensor<32xf32>
    %v1978 = stablehlo.add %v1975, %v1977 : tensor<32xf32>
    %v1979 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1980 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1981 = stablehlo.divide %v1972, %v1979 : tensor<32xf32>
    %v1982 = stablehlo.divide %v1978, %v1980 : tensor<32xf32>
    %v1983 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1984 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1985 = stablehlo.sqrt %v1982 : tensor<32xf32>
    %v1986 = stablehlo.add %v1985, %v1984 : tensor<32xf32>
    %v1987 = stablehlo.divide %v1981, %v1986 : tensor<32xf32>
    %v1988 = stablehlo.multiply %v1983, %v1987 : tensor<32xf32>
    %v1989 = stablehlo.subtract %g8, %v1988 : tensor<32xf32>
    %v1990 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1991 = stablehlo.multiply %v1990, %v1983 : tensor<32xf32>
    %v1992 = stablehlo.multiply %v1991, %g8 : tensor<32xf32>
    %v1993 = stablehlo.subtract %v1989, %v1992 : tensor<32xf32>
    %v1994 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1995 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1996 = stablehlo.multiply %v1994, %g8m : tensor<32xf32>
    %v1997 = stablehlo.multiply %v1995, %v1967 : tensor<32xf32>
    %v1998 = stablehlo.add %v1996, %v1997 : tensor<32xf32>
    %v1999 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2000 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2001 = stablehlo.multiply %v1999, %g8v : tensor<32xf32>
    %v2002 = stablehlo.multiply %v1967, %v1967 : tensor<32xf32>
    %v2003 = stablehlo.multiply %v2000, %v2002 : tensor<32xf32>
    %v2004 = stablehlo.add %v2001, %v2003 : tensor<32xf32>
    %v2005 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2006 = stablehlo.reshape %v288 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v2007 = stablehlo.reduce(%v2006 init: %v2005) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v2008 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2009 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2010 = stablehlo.multiply %v2008, %bt8m : tensor<32xf32>
    %v2011 = stablehlo.multiply %v2009, %v2007 : tensor<32xf32>
    %v2012 = stablehlo.add %v2010, %v2011 : tensor<32xf32>
    %v2013 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2014 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2015 = stablehlo.multiply %v2013, %bt8v : tensor<32xf32>
    %v2016 = stablehlo.multiply %v2007, %v2007 : tensor<32xf32>
    %v2017 = stablehlo.multiply %v2014, %v2016 : tensor<32xf32>
    %v2018 = stablehlo.add %v2015, %v2017 : tensor<32xf32>
    %v2019 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2020 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2021 = stablehlo.divide %v2012, %v2019 : tensor<32xf32>
    %v2022 = stablehlo.divide %v2018, %v2020 : tensor<32xf32>
    %v2023 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2024 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2025 = stablehlo.sqrt %v2022 : tensor<32xf32>
    %v2026 = stablehlo.add %v2025, %v2024 : tensor<32xf32>
    %v2027 = stablehlo.divide %v2021, %v2026 : tensor<32xf32>
    %v2028 = stablehlo.multiply %v2023, %v2027 : tensor<32xf32>
    %v2029 = stablehlo.subtract %bt8, %v2028 : tensor<32xf32>
    %v2030 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2031 = stablehlo.multiply %v2030, %v2023 : tensor<32xf32>
    %v2032 = stablehlo.multiply %v2031, %bt8 : tensor<32xf32>
    %v2033 = stablehlo.subtract %v2029, %v2032 : tensor<32xf32>
    %v2034 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2035 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2036 = stablehlo.multiply %v2034, %bt8m : tensor<32xf32>
    %v2037 = stablehlo.multiply %v2035, %v2007 : tensor<32xf32>
    %v2038 = stablehlo.add %v2036, %v2037 : tensor<32xf32>
    %v2039 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2040 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2041 = stablehlo.multiply %v2039, %bt8v : tensor<32xf32>
    %v2042 = stablehlo.multiply %v2007, %v2007 : tensor<32xf32>
    %v2043 = stablehlo.multiply %v2040, %v2042 : tensor<32xf32>
    %v2044 = stablehlo.add %v2041, %v2043 : tensor<32xf32>
    %v2045 = stablehlo.dot_general %v247, %v276, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %v2046 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v2047 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v2048 = stablehlo.multiply %v2046, %W9m : tensor<128x64xf32>
    %v2049 = stablehlo.multiply %v2047, %v2045 : tensor<128x64xf32>
    %v2050 = stablehlo.add %v2048, %v2049 : tensor<128x64xf32>
    %v2051 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v2052 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v2053 = stablehlo.multiply %v2051, %W9v : tensor<128x64xf32>
    %v2054 = stablehlo.multiply %v2045, %v2045 : tensor<128x64xf32>
    %v2055 = stablehlo.multiply %v2052, %v2054 : tensor<128x64xf32>
    %v2056 = stablehlo.add %v2053, %v2055 : tensor<128x64xf32>
    %v2057 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v2058 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v2059 = stablehlo.divide %v2050, %v2057 : tensor<128x64xf32>
    %v2060 = stablehlo.divide %v2056, %v2058 : tensor<128x64xf32>
    %v2061 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v2062 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v2063 = stablehlo.sqrt %v2060 : tensor<128x64xf32>
    %v2064 = stablehlo.add %v2063, %v2062 : tensor<128x64xf32>
    %v2065 = stablehlo.divide %v2059, %v2064 : tensor<128x64xf32>
    %v2066 = stablehlo.multiply %v2061, %v2065 : tensor<128x64xf32>
    %v2067 = stablehlo.subtract %W9, %v2066 : tensor<128x64xf32>
    %v2068 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v2069 = stablehlo.multiply %v2068, %v2061 : tensor<128x64xf32>
    %v2070 = stablehlo.multiply %v2069, %W9 : tensor<128x64xf32>
    %v2071 = stablehlo.subtract %v2067, %v2070 : tensor<128x64xf32>
    %v2072 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v2073 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v2074 = stablehlo.multiply %v2072, %W9m : tensor<128x64xf32>
    %v2075 = stablehlo.multiply %v2073, %v2045 : tensor<128x64xf32>
    %v2076 = stablehlo.add %v2074, %v2075 : tensor<128x64xf32>
    %v2077 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v2078 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v2079 = stablehlo.multiply %v2077, %W9v : tensor<128x64xf32>
    %v2080 = stablehlo.multiply %v2045, %v2045 : tensor<128x64xf32>
    %v2081 = stablehlo.multiply %v2078, %v2080 : tensor<128x64xf32>
    %v2082 = stablehlo.add %v2079, %v2081 : tensor<128x64xf32>
    %v2083 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2084 = stablehlo.reduce(%v276 init: %v2083) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v2085 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2086 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2087 = stablehlo.multiply %v2085, %b9m : tensor<64xf32>
    %v2088 = stablehlo.multiply %v2086, %v2084 : tensor<64xf32>
    %v2089 = stablehlo.add %v2087, %v2088 : tensor<64xf32>
    %v2090 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2091 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2092 = stablehlo.multiply %v2090, %b9v : tensor<64xf32>
    %v2093 = stablehlo.multiply %v2084, %v2084 : tensor<64xf32>
    %v2094 = stablehlo.multiply %v2091, %v2093 : tensor<64xf32>
    %v2095 = stablehlo.add %v2092, %v2094 : tensor<64xf32>
    %v2096 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2097 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2098 = stablehlo.divide %v2089, %v2096 : tensor<64xf32>
    %v2099 = stablehlo.divide %v2095, %v2097 : tensor<64xf32>
    %v2100 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2101 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2102 = stablehlo.sqrt %v2099 : tensor<64xf32>
    %v2103 = stablehlo.add %v2102, %v2101 : tensor<64xf32>
    %v2104 = stablehlo.divide %v2098, %v2103 : tensor<64xf32>
    %v2105 = stablehlo.multiply %v2100, %v2104 : tensor<64xf32>
    %v2106 = stablehlo.subtract %b9, %v2105 : tensor<64xf32>
    %v2107 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2108 = stablehlo.multiply %v2107, %v2100 : tensor<64xf32>
    %v2109 = stablehlo.multiply %v2108, %b9 : tensor<64xf32>
    %v2110 = stablehlo.subtract %v2106, %v2109 : tensor<64xf32>
    %v2111 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2112 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2113 = stablehlo.multiply %v2111, %b9m : tensor<64xf32>
    %v2114 = stablehlo.multiply %v2112, %v2084 : tensor<64xf32>
    %v2115 = stablehlo.add %v2113, %v2114 : tensor<64xf32>
    %v2116 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2117 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2118 = stablehlo.multiply %v2116, %b9v : tensor<64xf32>
    %v2119 = stablehlo.multiply %v2084, %v2084 : tensor<64xf32>
    %v2120 = stablehlo.multiply %v2117, %v2119 : tensor<64xf32>
    %v2121 = stablehlo.add %v2118, %v2120 : tensor<64xf32>
    %v2122 = stablehlo.dot_general %v252, %v272, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<64x64xf32>
    %v2123 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v2124 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v2125 = stablehlo.multiply %v2123, %Wam : tensor<64x64xf32>
    %v2126 = stablehlo.multiply %v2124, %v2122 : tensor<64x64xf32>
    %v2127 = stablehlo.add %v2125, %v2126 : tensor<64x64xf32>
    %v2128 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v2129 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v2130 = stablehlo.multiply %v2128, %Wav : tensor<64x64xf32>
    %v2131 = stablehlo.multiply %v2122, %v2122 : tensor<64x64xf32>
    %v2132 = stablehlo.multiply %v2129, %v2131 : tensor<64x64xf32>
    %v2133 = stablehlo.add %v2130, %v2132 : tensor<64x64xf32>
    %v2134 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v2135 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v2136 = stablehlo.divide %v2127, %v2134 : tensor<64x64xf32>
    %v2137 = stablehlo.divide %v2133, %v2135 : tensor<64x64xf32>
    %v2138 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v2139 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v2140 = stablehlo.sqrt %v2137 : tensor<64x64xf32>
    %v2141 = stablehlo.add %v2140, %v2139 : tensor<64x64xf32>
    %v2142 = stablehlo.divide %v2136, %v2141 : tensor<64x64xf32>
    %v2143 = stablehlo.multiply %v2138, %v2142 : tensor<64x64xf32>
    %v2144 = stablehlo.subtract %Wa, %v2143 : tensor<64x64xf32>
    %v2145 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v2146 = stablehlo.multiply %v2145, %v2138 : tensor<64x64xf32>
    %v2147 = stablehlo.multiply %v2146, %Wa : tensor<64x64xf32>
    %v2148 = stablehlo.subtract %v2144, %v2147 : tensor<64x64xf32>
    %v2149 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v2150 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v2151 = stablehlo.multiply %v2149, %Wam : tensor<64x64xf32>
    %v2152 = stablehlo.multiply %v2150, %v2122 : tensor<64x64xf32>
    %v2153 = stablehlo.add %v2151, %v2152 : tensor<64x64xf32>
    %v2154 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v2155 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v2156 = stablehlo.multiply %v2154, %Wav : tensor<64x64xf32>
    %v2157 = stablehlo.multiply %v2122, %v2122 : tensor<64x64xf32>
    %v2158 = stablehlo.multiply %v2155, %v2157 : tensor<64x64xf32>
    %v2159 = stablehlo.add %v2156, %v2158 : tensor<64x64xf32>
    %v2160 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2161 = stablehlo.reduce(%v272 init: %v2160) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v2162 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2163 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2164 = stablehlo.multiply %v2162, %bam : tensor<64xf32>
    %v2165 = stablehlo.multiply %v2163, %v2161 : tensor<64xf32>
    %v2166 = stablehlo.add %v2164, %v2165 : tensor<64xf32>
    %v2167 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2168 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2169 = stablehlo.multiply %v2167, %bav : tensor<64xf32>
    %v2170 = stablehlo.multiply %v2161, %v2161 : tensor<64xf32>
    %v2171 = stablehlo.multiply %v2168, %v2170 : tensor<64xf32>
    %v2172 = stablehlo.add %v2169, %v2171 : tensor<64xf32>
    %v2173 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2174 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2175 = stablehlo.divide %v2166, %v2173 : tensor<64xf32>
    %v2176 = stablehlo.divide %v2172, %v2174 : tensor<64xf32>
    %v2177 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2178 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2179 = stablehlo.sqrt %v2176 : tensor<64xf32>
    %v2180 = stablehlo.add %v2179, %v2178 : tensor<64xf32>
    %v2181 = stablehlo.divide %v2175, %v2180 : tensor<64xf32>
    %v2182 = stablehlo.multiply %v2177, %v2181 : tensor<64xf32>
    %v2183 = stablehlo.subtract %ba, %v2182 : tensor<64xf32>
    %v2184 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2185 = stablehlo.multiply %v2184, %v2177 : tensor<64xf32>
    %v2186 = stablehlo.multiply %v2185, %ba : tensor<64xf32>
    %v2187 = stablehlo.subtract %v2183, %v2186 : tensor<64xf32>
    %v2188 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2189 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2190 = stablehlo.multiply %v2188, %bam : tensor<64xf32>
    %v2191 = stablehlo.multiply %v2189, %v2161 : tensor<64xf32>
    %v2192 = stablehlo.add %v2190, %v2191 : tensor<64xf32>
    %v2193 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2194 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2195 = stablehlo.multiply %v2193, %bav : tensor<64xf32>
    %v2196 = stablehlo.multiply %v2161, %v2161 : tensor<64xf32>
    %v2197 = stablehlo.multiply %v2194, %v2196 : tensor<64xf32>
    %v2198 = stablehlo.add %v2195, %v2197 : tensor<64xf32>
    %v2199 = stablehlo.dot_general %v257, %v268, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x10xf32>) -> tensor<64x10xf32>
    %v2200 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v2201 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v2202 = stablehlo.multiply %v2200, %Wbm : tensor<64x10xf32>
    %v2203 = stablehlo.multiply %v2201, %v2199 : tensor<64x10xf32>
    %v2204 = stablehlo.add %v2202, %v2203 : tensor<64x10xf32>
    %v2205 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v2206 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v2207 = stablehlo.multiply %v2205, %Wbv : tensor<64x10xf32>
    %v2208 = stablehlo.multiply %v2199, %v2199 : tensor<64x10xf32>
    %v2209 = stablehlo.multiply %v2206, %v2208 : tensor<64x10xf32>
    %v2210 = stablehlo.add %v2207, %v2209 : tensor<64x10xf32>
    %v2211 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v2212 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v2213 = stablehlo.divide %v2204, %v2211 : tensor<64x10xf32>
    %v2214 = stablehlo.divide %v2210, %v2212 : tensor<64x10xf32>
    %v2215 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v2216 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v2217 = stablehlo.sqrt %v2214 : tensor<64x10xf32>
    %v2218 = stablehlo.add %v2217, %v2216 : tensor<64x10xf32>
    %v2219 = stablehlo.divide %v2213, %v2218 : tensor<64x10xf32>
    %v2220 = stablehlo.multiply %v2215, %v2219 : tensor<64x10xf32>
    %v2221 = stablehlo.subtract %Wb, %v2220 : tensor<64x10xf32>
    %v2222 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v2223 = stablehlo.multiply %v2222, %v2215 : tensor<64x10xf32>
    %v2224 = stablehlo.multiply %v2223, %Wb : tensor<64x10xf32>
    %v2225 = stablehlo.subtract %v2221, %v2224 : tensor<64x10xf32>
    %v2226 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v2227 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v2228 = stablehlo.multiply %v2226, %Wbm : tensor<64x10xf32>
    %v2229 = stablehlo.multiply %v2227, %v2199 : tensor<64x10xf32>
    %v2230 = stablehlo.add %v2228, %v2229 : tensor<64x10xf32>
    %v2231 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v2232 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v2233 = stablehlo.multiply %v2231, %Wbv : tensor<64x10xf32>
    %v2234 = stablehlo.multiply %v2199, %v2199 : tensor<64x10xf32>
    %v2235 = stablehlo.multiply %v2232, %v2234 : tensor<64x10xf32>
    %v2236 = stablehlo.add %v2233, %v2235 : tensor<64x10xf32>
    %v2237 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2238 = stablehlo.reduce(%v268 init: %v2237) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v2239 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2240 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2241 = stablehlo.multiply %v2239, %bbm : tensor<10xf32>
    %v2242 = stablehlo.multiply %v2240, %v2238 : tensor<10xf32>
    %v2243 = stablehlo.add %v2241, %v2242 : tensor<10xf32>
    %v2244 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2245 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2246 = stablehlo.multiply %v2244, %bbv : tensor<10xf32>
    %v2247 = stablehlo.multiply %v2238, %v2238 : tensor<10xf32>
    %v2248 = stablehlo.multiply %v2245, %v2247 : tensor<10xf32>
    %v2249 = stablehlo.add %v2246, %v2248 : tensor<10xf32>
    %v2250 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2251 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2252 = stablehlo.divide %v2243, %v2250 : tensor<10xf32>
    %v2253 = stablehlo.divide %v2249, %v2251 : tensor<10xf32>
    %v2254 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2255 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2256 = stablehlo.sqrt %v2253 : tensor<10xf32>
    %v2257 = stablehlo.add %v2256, %v2255 : tensor<10xf32>
    %v2258 = stablehlo.divide %v2252, %v2257 : tensor<10xf32>
    %v2259 = stablehlo.multiply %v2254, %v2258 : tensor<10xf32>
    %v2260 = stablehlo.subtract %bb, %v2259 : tensor<10xf32>
    %v2261 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2262 = stablehlo.multiply %v2261, %v2254 : tensor<10xf32>
    %v2263 = stablehlo.multiply %v2262, %bb : tensor<10xf32>
    %v2264 = stablehlo.subtract %v2260, %v2263 : tensor<10xf32>
    %v2265 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2266 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2267 = stablehlo.multiply %v2265, %bbm : tensor<10xf32>
    %v2268 = stablehlo.multiply %v2266, %v2238 : tensor<10xf32>
    %v2269 = stablehlo.add %v2267, %v2268 : tensor<10xf32>
    %v2270 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2271 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2272 = stablehlo.multiply %v2270, %bbv : tensor<10xf32>
    %v2273 = stablehlo.multiply %v2238, %v2238 : tensor<10xf32>
    %v2274 = stablehlo.multiply %v2271, %v2273 : tensor<10xf32>
    %v2275 = stablehlo.add %v2272, %v2274 : tensor<10xf32>
    return %v652, %v692, %v747, %v787, %v830, %v870, %v925, %v965, %v1008, %v1048, %v1103, %v1143, %v1186, %v1226, %v1281, %v1321, %v1364, %v1404, %v1459, %v1499, %v1542, %v1582, %v1637, %v1677, %v1720, %v1760, %v1815, %v1855, %v1898, %v1938, %v1993, %v2033, %v2071, %v2110, %v2148, %v2187, %v2225, %v2264, %v657, %v697, %v752, %v792, %v835, %v875, %v930, %v970, %v1013, %v1053, %v1108, %v1148, %v1191, %v1231, %v1286, %v1326, %v1369, %v1409, %v1464, %v1504, %v1547, %v1587, %v1642, %v1682, %v1725, %v1765, %v1820, %v1860, %v1903, %v1943, %v1998, %v2038, %v2076, %v2115, %v2153, %v2192, %v2230, %v2269, %v663, %v703, %v758, %v798, %v841, %v881, %v936, %v976, %v1019, %v1059, %v1114, %v1154, %v1197, %v1237, %v1292, %v1332, %v1375, %v1415, %v1470, %v1510, %v1553, %v1593, %v1648, %v1688, %v1731, %v1771, %v1826, %v1866, %v1909, %v1949, %v2004, %v2044, %v2082, %v2121, %v2159, %v2198, %v2236, %v2275, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
