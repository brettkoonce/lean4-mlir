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
    %v25 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v26 = stablehlo.maximum %v24, %v25 : tensor<128x16384xf32>
    %v27 = stablehlo.reshape %v26 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v28 = stablehlo.convolution(%v27, %W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x32x32xf32>
    %v29 = stablehlo.broadcast_in_dim %cb2, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v30 = stablehlo.add %v28, %v29 : tensor<128x16x32x32xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v32 = stablehlo.reshape %v31 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v33 = stablehlo.constant dense<0.0> : tensor<f32>
    %v34 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v35 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v36 = stablehlo.reduce(%v32 init: %v33) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v37 = stablehlo.broadcast_in_dim %v36, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v38 = stablehlo.divide %v37, %v34 : tensor<128x16x32x32xf32>
    %v39 = stablehlo.subtract %v32, %v38 : tensor<128x16x32x32xf32>
    %v40 = stablehlo.multiply %v39, %v39 : tensor<128x16x32x32xf32>
    %v41 = stablehlo.reduce(%v40 init: %v33) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v42 = stablehlo.broadcast_in_dim %v41, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v43 = stablehlo.divide %v42, %v34 : tensor<128x16x32x32xf32>
    %v44 = stablehlo.add %v43, %v35 : tensor<128x16x32x32xf32>
    %v45 = stablehlo.rsqrt %v44 : tensor<128x16x32x32xf32>
    %v46 = stablehlo.multiply %v39, %v45 : tensor<128x16x32x32xf32>
    %v47 = stablehlo.broadcast_in_dim %g2, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v48 = stablehlo.broadcast_in_dim %bt2, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v49 = stablehlo.multiply %v46, %v47 : tensor<128x16x32x32xf32>
    %v50 = stablehlo.add %v49, %v48 : tensor<128x16x32x32xf32>
    %v51 = stablehlo.reshape %v50 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v52 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v53 = stablehlo.maximum %v51, %v52 : tensor<128x16384xf32>
    %v54 = stablehlo.reshape %v53 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v55 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v56 = "stablehlo.reduce_window"(%v54, %v55) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16x16x16xf32>
    %v57 = stablehlo.reshape %v56 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v59 = stablehlo.convolution(%v58, %W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v60 = stablehlo.broadcast_in_dim %cb3, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v61 = stablehlo.add %v59, %v60 : tensor<128x16x16x16xf32>
    %v62 = stablehlo.reshape %v61 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v63 = stablehlo.reshape %v62 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v64 = stablehlo.constant dense<0.0> : tensor<f32>
    %v65 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v66 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v67 = stablehlo.reduce(%v63 init: %v64) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v68 = stablehlo.broadcast_in_dim %v67, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v69 = stablehlo.divide %v68, %v65 : tensor<128x16x16x16xf32>
    %v70 = stablehlo.subtract %v63, %v69 : tensor<128x16x16x16xf32>
    %v71 = stablehlo.multiply %v70, %v70 : tensor<128x16x16x16xf32>
    %v72 = stablehlo.reduce(%v71 init: %v64) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v73 = stablehlo.broadcast_in_dim %v72, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v74 = stablehlo.divide %v73, %v65 : tensor<128x16x16x16xf32>
    %v75 = stablehlo.add %v74, %v66 : tensor<128x16x16x16xf32>
    %v76 = stablehlo.rsqrt %v75 : tensor<128x16x16x16xf32>
    %v77 = stablehlo.multiply %v70, %v76 : tensor<128x16x16x16xf32>
    %v78 = stablehlo.broadcast_in_dim %g3, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v79 = stablehlo.broadcast_in_dim %bt3, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v80 = stablehlo.multiply %v77, %v78 : tensor<128x16x16x16xf32>
    %v81 = stablehlo.add %v80, %v79 : tensor<128x16x16x16xf32>
    %v82 = stablehlo.reshape %v81 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v83 = stablehlo.constant dense<0.0> : tensor<128x4096xf32>
    %v84 = stablehlo.maximum %v82, %v83 : tensor<128x4096xf32>
    %v85 = stablehlo.reshape %v84 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v86 = stablehlo.convolution(%v85, %W4)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v87 = stablehlo.broadcast_in_dim %cb4, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v88 = stablehlo.add %v86, %v87 : tensor<128x16x16x16xf32>
    %v89 = stablehlo.reshape %v88 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v90 = stablehlo.reshape %v89 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v91 = stablehlo.constant dense<0.0> : tensor<f32>
    %v92 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v93 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v94 = stablehlo.reduce(%v90 init: %v91) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v95 = stablehlo.broadcast_in_dim %v94, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v96 = stablehlo.divide %v95, %v92 : tensor<128x16x16x16xf32>
    %v97 = stablehlo.subtract %v90, %v96 : tensor<128x16x16x16xf32>
    %v98 = stablehlo.multiply %v97, %v97 : tensor<128x16x16x16xf32>
    %v99 = stablehlo.reduce(%v98 init: %v91) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v100 = stablehlo.broadcast_in_dim %v99, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v101 = stablehlo.divide %v100, %v92 : tensor<128x16x16x16xf32>
    %v102 = stablehlo.add %v101, %v93 : tensor<128x16x16x16xf32>
    %v103 = stablehlo.rsqrt %v102 : tensor<128x16x16x16xf32>
    %v104 = stablehlo.multiply %v97, %v103 : tensor<128x16x16x16xf32>
    %v105 = stablehlo.broadcast_in_dim %g4, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v106 = stablehlo.broadcast_in_dim %bt4, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v107 = stablehlo.multiply %v104, %v105 : tensor<128x16x16x16xf32>
    %v108 = stablehlo.add %v107, %v106 : tensor<128x16x16x16xf32>
    %v109 = stablehlo.reshape %v108 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v110 = stablehlo.constant dense<0.0> : tensor<128x4096xf32>
    %v111 = stablehlo.maximum %v109, %v110 : tensor<128x4096xf32>
    %v112 = stablehlo.reshape %v111 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v113 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v114 = "stablehlo.reduce_window"(%v112, %v113) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x8x8xf32>
    %v115 = stablehlo.reshape %v114 : (tensor<128x16x8x8xf32>) -> tensor<128x1024xf32>
    %v116 = stablehlo.reshape %v115 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v117 = stablehlo.convolution(%v116, %W5)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x8x8xf32>, tensor<32x16x3x3xf32>) -> tensor<128x32x8x8xf32>
    %v118 = stablehlo.broadcast_in_dim %cb5, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v119 = stablehlo.add %v117, %v118 : tensor<128x32x8x8xf32>
    %v120 = stablehlo.reshape %v119 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v121 = stablehlo.reshape %v120 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v122 = stablehlo.constant dense<0.0> : tensor<f32>
    %v123 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v124 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v125 = stablehlo.reduce(%v121 init: %v122) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v126 = stablehlo.broadcast_in_dim %v125, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v127 = stablehlo.divide %v126, %v123 : tensor<128x32x8x8xf32>
    %v128 = stablehlo.subtract %v121, %v127 : tensor<128x32x8x8xf32>
    %v129 = stablehlo.multiply %v128, %v128 : tensor<128x32x8x8xf32>
    %v130 = stablehlo.reduce(%v129 init: %v122) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v131 = stablehlo.broadcast_in_dim %v130, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v132 = stablehlo.divide %v131, %v123 : tensor<128x32x8x8xf32>
    %v133 = stablehlo.add %v132, %v124 : tensor<128x32x8x8xf32>
    %v134 = stablehlo.rsqrt %v133 : tensor<128x32x8x8xf32>
    %v135 = stablehlo.multiply %v128, %v134 : tensor<128x32x8x8xf32>
    %v136 = stablehlo.broadcast_in_dim %g5, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v137 = stablehlo.broadcast_in_dim %bt5, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v138 = stablehlo.multiply %v135, %v136 : tensor<128x32x8x8xf32>
    %v139 = stablehlo.add %v138, %v137 : tensor<128x32x8x8xf32>
    %v140 = stablehlo.reshape %v139 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v141 = stablehlo.constant dense<0.0> : tensor<128x2048xf32>
    %v142 = stablehlo.maximum %v140, %v141 : tensor<128x2048xf32>
    %v143 = stablehlo.reshape %v142 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v144 = stablehlo.convolution(%v143, %W6)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x8x8xf32>
    %v145 = stablehlo.broadcast_in_dim %cb6, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v146 = stablehlo.add %v144, %v145 : tensor<128x32x8x8xf32>
    %v147 = stablehlo.reshape %v146 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v148 = stablehlo.reshape %v147 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v149 = stablehlo.constant dense<0.0> : tensor<f32>
    %v150 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v151 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v152 = stablehlo.reduce(%v148 init: %v149) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v153 = stablehlo.broadcast_in_dim %v152, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v154 = stablehlo.divide %v153, %v150 : tensor<128x32x8x8xf32>
    %v155 = stablehlo.subtract %v148, %v154 : tensor<128x32x8x8xf32>
    %v156 = stablehlo.multiply %v155, %v155 : tensor<128x32x8x8xf32>
    %v157 = stablehlo.reduce(%v156 init: %v149) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v158 = stablehlo.broadcast_in_dim %v157, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v159 = stablehlo.divide %v158, %v150 : tensor<128x32x8x8xf32>
    %v160 = stablehlo.add %v159, %v151 : tensor<128x32x8x8xf32>
    %v161 = stablehlo.rsqrt %v160 : tensor<128x32x8x8xf32>
    %v162 = stablehlo.multiply %v155, %v161 : tensor<128x32x8x8xf32>
    %v163 = stablehlo.broadcast_in_dim %g6, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v164 = stablehlo.broadcast_in_dim %bt6, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v165 = stablehlo.multiply %v162, %v163 : tensor<128x32x8x8xf32>
    %v166 = stablehlo.add %v165, %v164 : tensor<128x32x8x8xf32>
    %v167 = stablehlo.reshape %v166 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v168 = stablehlo.constant dense<0.0> : tensor<128x2048xf32>
    %v169 = stablehlo.maximum %v167, %v168 : tensor<128x2048xf32>
    %v170 = stablehlo.reshape %v169 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v171 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v172 = "stablehlo.reduce_window"(%v170, %v171) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32x4x4xf32>
    %v173 = stablehlo.reshape %v172 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v174 = stablehlo.reshape %v173 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v175 = stablehlo.convolution(%v174, %W7)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v176 = stablehlo.broadcast_in_dim %cb7, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v177 = stablehlo.add %v175, %v176 : tensor<128x32x4x4xf32>
    %v178 = stablehlo.reshape %v177 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v179 = stablehlo.reshape %v178 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v180 = stablehlo.constant dense<0.0> : tensor<f32>
    %v181 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v182 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v183 = stablehlo.reduce(%v179 init: %v180) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v184 = stablehlo.broadcast_in_dim %v183, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v185 = stablehlo.divide %v184, %v181 : tensor<128x32x4x4xf32>
    %v186 = stablehlo.subtract %v179, %v185 : tensor<128x32x4x4xf32>
    %v187 = stablehlo.multiply %v186, %v186 : tensor<128x32x4x4xf32>
    %v188 = stablehlo.reduce(%v187 init: %v180) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v189 = stablehlo.broadcast_in_dim %v188, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v190 = stablehlo.divide %v189, %v181 : tensor<128x32x4x4xf32>
    %v191 = stablehlo.add %v190, %v182 : tensor<128x32x4x4xf32>
    %v192 = stablehlo.rsqrt %v191 : tensor<128x32x4x4xf32>
    %v193 = stablehlo.multiply %v186, %v192 : tensor<128x32x4x4xf32>
    %v194 = stablehlo.broadcast_in_dim %g7, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v195 = stablehlo.broadcast_in_dim %bt7, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v196 = stablehlo.multiply %v193, %v194 : tensor<128x32x4x4xf32>
    %v197 = stablehlo.add %v196, %v195 : tensor<128x32x4x4xf32>
    %v198 = stablehlo.reshape %v197 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v199 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v200 = stablehlo.maximum %v198, %v199 : tensor<128x512xf32>
    %v201 = stablehlo.reshape %v200 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v202 = stablehlo.convolution(%v201, %W8)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v203 = stablehlo.broadcast_in_dim %cb8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v204 = stablehlo.add %v202, %v203 : tensor<128x32x4x4xf32>
    %v205 = stablehlo.reshape %v204 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v206 = stablehlo.reshape %v205 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v207 = stablehlo.constant dense<0.0> : tensor<f32>
    %v208 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v209 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v210 = stablehlo.reduce(%v206 init: %v207) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v211 = stablehlo.broadcast_in_dim %v210, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v212 = stablehlo.divide %v211, %v208 : tensor<128x32x4x4xf32>
    %v213 = stablehlo.subtract %v206, %v212 : tensor<128x32x4x4xf32>
    %v214 = stablehlo.multiply %v213, %v213 : tensor<128x32x4x4xf32>
    %v215 = stablehlo.reduce(%v214 init: %v207) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v216 = stablehlo.broadcast_in_dim %v215, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v217 = stablehlo.divide %v216, %v208 : tensor<128x32x4x4xf32>
    %v218 = stablehlo.add %v217, %v209 : tensor<128x32x4x4xf32>
    %v219 = stablehlo.rsqrt %v218 : tensor<128x32x4x4xf32>
    %v220 = stablehlo.multiply %v213, %v219 : tensor<128x32x4x4xf32>
    %v221 = stablehlo.broadcast_in_dim %g8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v222 = stablehlo.broadcast_in_dim %bt8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v223 = stablehlo.multiply %v220, %v221 : tensor<128x32x4x4xf32>
    %v224 = stablehlo.add %v223, %v222 : tensor<128x32x4x4xf32>
    %v225 = stablehlo.reshape %v224 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v226 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v227 = stablehlo.maximum %v225, %v226 : tensor<128x512xf32>
    %v228 = stablehlo.reshape %v227 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v229 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v230 = "stablehlo.reduce_window"(%v228, %v229) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32x2x2xf32>
    %v231 = stablehlo.reshape %v230 : (tensor<128x32x2x2xf32>) -> tensor<128x128xf32>
    %v232 = stablehlo.dot_general %v231, %W9, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %v233 = stablehlo.broadcast_in_dim %b9, dims = [1] : (tensor<64xf32>) -> tensor<128x64xf32>
    %v234 = stablehlo.add %v232, %v233 : tensor<128x64xf32>
    %v235 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v236 = stablehlo.maximum %v234, %v235 : tensor<128x64xf32>
    %v237 = stablehlo.dot_general %v236, %Wa, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
    %v238 = stablehlo.broadcast_in_dim %ba, dims = [1] : (tensor<64xf32>) -> tensor<128x64xf32>
    %v239 = stablehlo.add %v237, %v238 : tensor<128x64xf32>
    %v240 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v241 = stablehlo.maximum %v239, %v240 : tensor<128x64xf32>
    %v242 = stablehlo.dot_general %v241, %Wb, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x10xf32>) -> tensor<128x10xf32>
    %v243 = stablehlo.broadcast_in_dim %bb, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
    %v244 = stablehlo.add %v242, %v243 : tensor<128x10xf32>
    %v245 = stablehlo.exponential %v244 : tensor<128x10xf32>
    %v246 = stablehlo.constant dense<0.0> : tensor<f32>
    %v247 = stablehlo.reduce(%v245 init: %v246) applies stablehlo.add across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    %v248 = stablehlo.broadcast_in_dim %v247, dims = [0] : (tensor<128xf32>) -> tensor<128x10xf32>
    %v249 = stablehlo.divide %v245, %v248 : tensor<128x10xf32>
    %v250 = stablehlo.subtract %v249, %onehot : tensor<128x10xf32>
    %v251 = stablehlo.constant dense<0.0078125> : tensor<128x10xf32>
    %v252 = stablehlo.multiply %v250, %v251 : tensor<128x10xf32>
    // ── report-only scalar loss (NOT pretty(AST): no rank-0 loss op; feeds no
    //    parameter, only the driver's progress line) ──
    %llog = stablehlo.log %v249 : tensor<128x10xf32>
    %ohll = stablehlo.multiply %onehot, %llog : tensor<128x10xf32>
    %csum = stablehlo.reduce(%ohll init: %lzero) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<f32>
    %cneg = stablehlo.negate %csum : tensor<f32>
    %lbf = stablehlo.constant dense<128.0> : tensor<f32>
    %loss = stablehlo.divide %cneg, %lbf : tensor<f32>
    %v253 = stablehlo.dot_general %v252, %Wb, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x10xf32>, tensor<64x10xf32>) -> tensor<128x64xf32>
    %v254 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v255 = stablehlo.compare GT, %v239, %v254 : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xi1>
    %v256 = stablehlo.select %v255, %v253, %v254 : tensor<128x64xi1>, tensor<128x64xf32>
    %v257 = stablehlo.dot_general %v256, %Wa, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
    %v258 = stablehlo.constant dense<0.0> : tensor<128x64xf32>
    %v259 = stablehlo.compare GT, %v234, %v258 : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x64xi1>
    %v260 = stablehlo.select %v259, %v257, %v258 : tensor<128x64xi1>, tensor<128x64xf32>
    %v261 = stablehlo.dot_general %v260, %W9, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x128xf32>
    %v262 = stablehlo.reshape %v227 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v263 = stablehlo.reshape %v261 : (tensor<128x128xf32>) -> tensor<128x32x2x2xf32>
    %v264 = stablehlo.constant dense<0.0> : tensor<f32>
    %v265 = "stablehlo.select_and_scatter"(%v262, %v263, %v264) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x4x4xf32>, tensor<128x32x2x2xf32>, tensor<f32>) -> tensor<128x32x4x4xf32>
    %v266 = stablehlo.reshape %v265 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v267 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v268 = stablehlo.compare GT, %v225, %v267 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v269 = stablehlo.select %v268, %v266, %v267 : tensor<128x512xi1>, tensor<128x512xf32>
    %v270 = stablehlo.reshape %v269 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v271 = stablehlo.reshape %v205 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v272 = stablehlo.constant dense<0.0> : tensor<f32>
    %v273 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v274 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v275 = stablehlo.reduce(%v271 init: %v272) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v276 = stablehlo.broadcast_in_dim %v275, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v277 = stablehlo.divide %v276, %v273 : tensor<128x32x4x4xf32>
    %v278 = stablehlo.subtract %v271, %v277 : tensor<128x32x4x4xf32>
    %v279 = stablehlo.multiply %v278, %v278 : tensor<128x32x4x4xf32>
    %v280 = stablehlo.reduce(%v279 init: %v272) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v281 = stablehlo.broadcast_in_dim %v280, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v282 = stablehlo.divide %v281, %v273 : tensor<128x32x4x4xf32>
    %v283 = stablehlo.add %v282, %v274 : tensor<128x32x4x4xf32>
    %v284 = stablehlo.rsqrt %v283 : tensor<128x32x4x4xf32>
    %v285 = stablehlo.multiply %v278, %v284 : tensor<128x32x4x4xf32>
    %v286 = stablehlo.broadcast_in_dim %g8, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v287 = stablehlo.multiply %v286, %v270 : tensor<128x32x4x4xf32>
    %v288 = stablehlo.reduce(%v287 init: %v272) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v289 = stablehlo.broadcast_in_dim %v288, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v290 = stablehlo.multiply %v285, %v287 : tensor<128x32x4x4xf32>
    %v291 = stablehlo.reduce(%v290 init: %v272) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v292 = stablehlo.broadcast_in_dim %v291, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v293 = stablehlo.multiply %v287, %v273 : tensor<128x32x4x4xf32>
    %v294 = stablehlo.subtract %v293, %v289 : tensor<128x32x4x4xf32>
    %v295 = stablehlo.multiply %v285, %v292 : tensor<128x32x4x4xf32>
    %v296 = stablehlo.subtract %v294, %v295 : tensor<128x32x4x4xf32>
    %v297 = stablehlo.divide %v284, %v273 : tensor<128x32x4x4xf32>
    %v298 = stablehlo.multiply %v297, %v296 : tensor<128x32x4x4xf32>
    %v299 = stablehlo.reshape %v298 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v300 = stablehlo.reshape %v299 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v301 = stablehlo.transpose %W8, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v302 = stablehlo.reverse %v301, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v303 = stablehlo.convolution(%v300, %v302)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v304 = stablehlo.reshape %v303 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v305 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v306 = stablehlo.compare GT, %v198, %v305 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v307 = stablehlo.select %v306, %v304, %v305 : tensor<128x512xi1>, tensor<128x512xf32>
    %v308 = stablehlo.reshape %v307 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v309 = stablehlo.reshape %v178 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v310 = stablehlo.constant dense<0.0> : tensor<f32>
    %v311 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v312 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v313 = stablehlo.reduce(%v309 init: %v310) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v314 = stablehlo.broadcast_in_dim %v313, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v315 = stablehlo.divide %v314, %v311 : tensor<128x32x4x4xf32>
    %v316 = stablehlo.subtract %v309, %v315 : tensor<128x32x4x4xf32>
    %v317 = stablehlo.multiply %v316, %v316 : tensor<128x32x4x4xf32>
    %v318 = stablehlo.reduce(%v317 init: %v310) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v319 = stablehlo.broadcast_in_dim %v318, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v320 = stablehlo.divide %v319, %v311 : tensor<128x32x4x4xf32>
    %v321 = stablehlo.add %v320, %v312 : tensor<128x32x4x4xf32>
    %v322 = stablehlo.rsqrt %v321 : tensor<128x32x4x4xf32>
    %v323 = stablehlo.multiply %v316, %v322 : tensor<128x32x4x4xf32>
    %v324 = stablehlo.broadcast_in_dim %g7, dims = [1] : (tensor<32xf32>) -> tensor<128x32x4x4xf32>
    %v325 = stablehlo.multiply %v324, %v308 : tensor<128x32x4x4xf32>
    %v326 = stablehlo.reduce(%v325 init: %v310) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v327 = stablehlo.broadcast_in_dim %v326, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v328 = stablehlo.multiply %v323, %v325 : tensor<128x32x4x4xf32>
    %v329 = stablehlo.reduce(%v328 init: %v310) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v330 = stablehlo.broadcast_in_dim %v329, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v331 = stablehlo.multiply %v325, %v311 : tensor<128x32x4x4xf32>
    %v332 = stablehlo.subtract %v331, %v327 : tensor<128x32x4x4xf32>
    %v333 = stablehlo.multiply %v323, %v330 : tensor<128x32x4x4xf32>
    %v334 = stablehlo.subtract %v332, %v333 : tensor<128x32x4x4xf32>
    %v335 = stablehlo.divide %v322, %v311 : tensor<128x32x4x4xf32>
    %v336 = stablehlo.multiply %v335, %v334 : tensor<128x32x4x4xf32>
    %v337 = stablehlo.reshape %v336 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v338 = stablehlo.reshape %v337 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v339 = stablehlo.transpose %W7, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v340 = stablehlo.reverse %v339, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v341 = stablehlo.convolution(%v338, %v340)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x4x4xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x4x4xf32>
    %v342 = stablehlo.reshape %v341 : (tensor<128x32x4x4xf32>) -> tensor<128x512xf32>
    %v343 = stablehlo.reshape %v169 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v344 = stablehlo.reshape %v342 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v345 = stablehlo.constant dense<0.0> : tensor<f32>
    %v346 = "stablehlo.select_and_scatter"(%v343, %v344, %v345) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x32x8x8xf32>, tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32x8x8xf32>
    %v347 = stablehlo.reshape %v346 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v348 = stablehlo.constant dense<0.0> : tensor<128x2048xf32>
    %v349 = stablehlo.compare GT, %v167, %v348 : (tensor<128x2048xf32>, tensor<128x2048xf32>) -> tensor<128x2048xi1>
    %v350 = stablehlo.select %v349, %v347, %v348 : tensor<128x2048xi1>, tensor<128x2048xf32>
    %v351 = stablehlo.reshape %v350 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v352 = stablehlo.reshape %v147 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v353 = stablehlo.constant dense<0.0> : tensor<f32>
    %v354 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v355 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v356 = stablehlo.reduce(%v352 init: %v353) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v357 = stablehlo.broadcast_in_dim %v356, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v358 = stablehlo.divide %v357, %v354 : tensor<128x32x8x8xf32>
    %v359 = stablehlo.subtract %v352, %v358 : tensor<128x32x8x8xf32>
    %v360 = stablehlo.multiply %v359, %v359 : tensor<128x32x8x8xf32>
    %v361 = stablehlo.reduce(%v360 init: %v353) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v362 = stablehlo.broadcast_in_dim %v361, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v363 = stablehlo.divide %v362, %v354 : tensor<128x32x8x8xf32>
    %v364 = stablehlo.add %v363, %v355 : tensor<128x32x8x8xf32>
    %v365 = stablehlo.rsqrt %v364 : tensor<128x32x8x8xf32>
    %v366 = stablehlo.multiply %v359, %v365 : tensor<128x32x8x8xf32>
    %v367 = stablehlo.broadcast_in_dim %g6, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v368 = stablehlo.multiply %v367, %v351 : tensor<128x32x8x8xf32>
    %v369 = stablehlo.reduce(%v368 init: %v353) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v370 = stablehlo.broadcast_in_dim %v369, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v371 = stablehlo.multiply %v366, %v368 : tensor<128x32x8x8xf32>
    %v372 = stablehlo.reduce(%v371 init: %v353) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v373 = stablehlo.broadcast_in_dim %v372, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v374 = stablehlo.multiply %v368, %v354 : tensor<128x32x8x8xf32>
    %v375 = stablehlo.subtract %v374, %v370 : tensor<128x32x8x8xf32>
    %v376 = stablehlo.multiply %v366, %v373 : tensor<128x32x8x8xf32>
    %v377 = stablehlo.subtract %v375, %v376 : tensor<128x32x8x8xf32>
    %v378 = stablehlo.divide %v365, %v354 : tensor<128x32x8x8xf32>
    %v379 = stablehlo.multiply %v378, %v377 : tensor<128x32x8x8xf32>
    %v380 = stablehlo.reshape %v379 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v381 = stablehlo.reshape %v380 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v382 = stablehlo.transpose %W6, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v383 = stablehlo.reverse %v382, dims = [2, 3] : tensor<32x32x3x3xf32>
    %v384 = stablehlo.convolution(%v381, %v383)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x8x8xf32>
    %v385 = stablehlo.reshape %v384 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v386 = stablehlo.constant dense<0.0> : tensor<128x2048xf32>
    %v387 = stablehlo.compare GT, %v140, %v386 : (tensor<128x2048xf32>, tensor<128x2048xf32>) -> tensor<128x2048xi1>
    %v388 = stablehlo.select %v387, %v385, %v386 : tensor<128x2048xi1>, tensor<128x2048xf32>
    %v389 = stablehlo.reshape %v388 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v390 = stablehlo.reshape %v120 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v391 = stablehlo.constant dense<0.0> : tensor<f32>
    %v392 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v393 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v394 = stablehlo.reduce(%v390 init: %v391) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v395 = stablehlo.broadcast_in_dim %v394, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v396 = stablehlo.divide %v395, %v392 : tensor<128x32x8x8xf32>
    %v397 = stablehlo.subtract %v390, %v396 : tensor<128x32x8x8xf32>
    %v398 = stablehlo.multiply %v397, %v397 : tensor<128x32x8x8xf32>
    %v399 = stablehlo.reduce(%v398 init: %v391) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v400 = stablehlo.broadcast_in_dim %v399, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v401 = stablehlo.divide %v400, %v392 : tensor<128x32x8x8xf32>
    %v402 = stablehlo.add %v401, %v393 : tensor<128x32x8x8xf32>
    %v403 = stablehlo.rsqrt %v402 : tensor<128x32x8x8xf32>
    %v404 = stablehlo.multiply %v397, %v403 : tensor<128x32x8x8xf32>
    %v405 = stablehlo.broadcast_in_dim %g5, dims = [1] : (tensor<32xf32>) -> tensor<128x32x8x8xf32>
    %v406 = stablehlo.multiply %v405, %v389 : tensor<128x32x8x8xf32>
    %v407 = stablehlo.reduce(%v406 init: %v391) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v408 = stablehlo.broadcast_in_dim %v407, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v409 = stablehlo.multiply %v404, %v406 : tensor<128x32x8x8xf32>
    %v410 = stablehlo.reduce(%v409 init: %v391) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v411 = stablehlo.broadcast_in_dim %v410, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v412 = stablehlo.multiply %v406, %v392 : tensor<128x32x8x8xf32>
    %v413 = stablehlo.subtract %v412, %v408 : tensor<128x32x8x8xf32>
    %v414 = stablehlo.multiply %v404, %v411 : tensor<128x32x8x8xf32>
    %v415 = stablehlo.subtract %v413, %v414 : tensor<128x32x8x8xf32>
    %v416 = stablehlo.divide %v403, %v392 : tensor<128x32x8x8xf32>
    %v417 = stablehlo.multiply %v416, %v415 : tensor<128x32x8x8xf32>
    %v418 = stablehlo.reshape %v417 : (tensor<128x32x8x8xf32>) -> tensor<128x2048xf32>
    %v419 = stablehlo.reshape %v418 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v420 = stablehlo.transpose %W5, dims = [1, 0, 2, 3] : (tensor<32x16x3x3xf32>) -> tensor<16x32x3x3xf32>
    %v421 = stablehlo.reverse %v420, dims = [2, 3] : tensor<16x32x3x3xf32>
    %v422 = stablehlo.convolution(%v419, %v421)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x8x8xf32>, tensor<16x32x3x3xf32>) -> tensor<128x16x8x8xf32>
    %v423 = stablehlo.reshape %v422 : (tensor<128x16x8x8xf32>) -> tensor<128x1024xf32>
    %v424 = stablehlo.reshape %v111 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v425 = stablehlo.reshape %v423 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v426 = stablehlo.constant dense<0.0> : tensor<f32>
    %v427 = "stablehlo.select_and_scatter"(%v424, %v425, %v426) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x16x16xf32>, tensor<128x16x8x8xf32>, tensor<f32>) -> tensor<128x16x16x16xf32>
    %v428 = stablehlo.reshape %v427 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v429 = stablehlo.constant dense<0.0> : tensor<128x4096xf32>
    %v430 = stablehlo.compare GT, %v109, %v429 : (tensor<128x4096xf32>, tensor<128x4096xf32>) -> tensor<128x4096xi1>
    %v431 = stablehlo.select %v430, %v428, %v429 : tensor<128x4096xi1>, tensor<128x4096xf32>
    %v432 = stablehlo.reshape %v431 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v433 = stablehlo.reshape %v89 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v434 = stablehlo.constant dense<0.0> : tensor<f32>
    %v435 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v436 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v437 = stablehlo.reduce(%v433 init: %v434) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v438 = stablehlo.broadcast_in_dim %v437, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v439 = stablehlo.divide %v438, %v435 : tensor<128x16x16x16xf32>
    %v440 = stablehlo.subtract %v433, %v439 : tensor<128x16x16x16xf32>
    %v441 = stablehlo.multiply %v440, %v440 : tensor<128x16x16x16xf32>
    %v442 = stablehlo.reduce(%v441 init: %v434) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v443 = stablehlo.broadcast_in_dim %v442, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v444 = stablehlo.divide %v443, %v435 : tensor<128x16x16x16xf32>
    %v445 = stablehlo.add %v444, %v436 : tensor<128x16x16x16xf32>
    %v446 = stablehlo.rsqrt %v445 : tensor<128x16x16x16xf32>
    %v447 = stablehlo.multiply %v440, %v446 : tensor<128x16x16x16xf32>
    %v448 = stablehlo.broadcast_in_dim %g4, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v449 = stablehlo.multiply %v448, %v432 : tensor<128x16x16x16xf32>
    %v450 = stablehlo.reduce(%v449 init: %v434) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v451 = stablehlo.broadcast_in_dim %v450, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v452 = stablehlo.multiply %v447, %v449 : tensor<128x16x16x16xf32>
    %v453 = stablehlo.reduce(%v452 init: %v434) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v454 = stablehlo.broadcast_in_dim %v453, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v455 = stablehlo.multiply %v449, %v435 : tensor<128x16x16x16xf32>
    %v456 = stablehlo.subtract %v455, %v451 : tensor<128x16x16x16xf32>
    %v457 = stablehlo.multiply %v447, %v454 : tensor<128x16x16x16xf32>
    %v458 = stablehlo.subtract %v456, %v457 : tensor<128x16x16x16xf32>
    %v459 = stablehlo.divide %v446, %v435 : tensor<128x16x16x16xf32>
    %v460 = stablehlo.multiply %v459, %v458 : tensor<128x16x16x16xf32>
    %v461 = stablehlo.reshape %v460 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v462 = stablehlo.reshape %v461 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v463 = stablehlo.transpose %W4, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v464 = stablehlo.reverse %v463, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v465 = stablehlo.convolution(%v462, %v464)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v466 = stablehlo.reshape %v465 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v467 = stablehlo.constant dense<0.0> : tensor<128x4096xf32>
    %v468 = stablehlo.compare GT, %v82, %v467 : (tensor<128x4096xf32>, tensor<128x4096xf32>) -> tensor<128x4096xi1>
    %v469 = stablehlo.select %v468, %v466, %v467 : tensor<128x4096xi1>, tensor<128x4096xf32>
    %v470 = stablehlo.reshape %v469 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v471 = stablehlo.reshape %v62 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v472 = stablehlo.constant dense<0.0> : tensor<f32>
    %v473 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v474 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v475 = stablehlo.reduce(%v471 init: %v472) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v476 = stablehlo.broadcast_in_dim %v475, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v477 = stablehlo.divide %v476, %v473 : tensor<128x16x16x16xf32>
    %v478 = stablehlo.subtract %v471, %v477 : tensor<128x16x16x16xf32>
    %v479 = stablehlo.multiply %v478, %v478 : tensor<128x16x16x16xf32>
    %v480 = stablehlo.reduce(%v479 init: %v472) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v481 = stablehlo.broadcast_in_dim %v480, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v482 = stablehlo.divide %v481, %v473 : tensor<128x16x16x16xf32>
    %v483 = stablehlo.add %v482, %v474 : tensor<128x16x16x16xf32>
    %v484 = stablehlo.rsqrt %v483 : tensor<128x16x16x16xf32>
    %v485 = stablehlo.multiply %v478, %v484 : tensor<128x16x16x16xf32>
    %v486 = stablehlo.broadcast_in_dim %g3, dims = [1] : (tensor<16xf32>) -> tensor<128x16x16x16xf32>
    %v487 = stablehlo.multiply %v486, %v470 : tensor<128x16x16x16xf32>
    %v488 = stablehlo.reduce(%v487 init: %v472) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v489 = stablehlo.broadcast_in_dim %v488, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v490 = stablehlo.multiply %v485, %v487 : tensor<128x16x16x16xf32>
    %v491 = stablehlo.reduce(%v490 init: %v472) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v492 = stablehlo.broadcast_in_dim %v491, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v493 = stablehlo.multiply %v487, %v473 : tensor<128x16x16x16xf32>
    %v494 = stablehlo.subtract %v493, %v489 : tensor<128x16x16x16xf32>
    %v495 = stablehlo.multiply %v485, %v492 : tensor<128x16x16x16xf32>
    %v496 = stablehlo.subtract %v494, %v495 : tensor<128x16x16x16xf32>
    %v497 = stablehlo.divide %v484, %v473 : tensor<128x16x16x16xf32>
    %v498 = stablehlo.multiply %v497, %v496 : tensor<128x16x16x16xf32>
    %v499 = stablehlo.reshape %v498 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v500 = stablehlo.reshape %v499 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v501 = stablehlo.transpose %W3, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v502 = stablehlo.reverse %v501, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v503 = stablehlo.convolution(%v500, %v502)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x16x16xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x16x16xf32>
    %v504 = stablehlo.reshape %v503 : (tensor<128x16x16x16xf32>) -> tensor<128x4096xf32>
    %v505 = stablehlo.reshape %v53 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v506 = stablehlo.reshape %v504 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v507 = stablehlo.constant dense<0.0> : tensor<f32>
    %v508 = "stablehlo.select_and_scatter"(%v505, %v506, %v507) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<128x16x32x32xf32>, tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16x32x32xf32>
    %v509 = stablehlo.reshape %v508 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v510 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v511 = stablehlo.compare GT, %v51, %v510 : (tensor<128x16384xf32>, tensor<128x16384xf32>) -> tensor<128x16384xi1>
    %v512 = stablehlo.select %v511, %v509, %v510 : tensor<128x16384xi1>, tensor<128x16384xf32>
    %v513 = stablehlo.reshape %v512 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v514 = stablehlo.reshape %v31 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v515 = stablehlo.constant dense<0.0> : tensor<f32>
    %v516 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v517 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v518 = stablehlo.reduce(%v514 init: %v515) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v519 = stablehlo.broadcast_in_dim %v518, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v520 = stablehlo.divide %v519, %v516 : tensor<128x16x32x32xf32>
    %v521 = stablehlo.subtract %v514, %v520 : tensor<128x16x32x32xf32>
    %v522 = stablehlo.multiply %v521, %v521 : tensor<128x16x32x32xf32>
    %v523 = stablehlo.reduce(%v522 init: %v515) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v524 = stablehlo.broadcast_in_dim %v523, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v525 = stablehlo.divide %v524, %v516 : tensor<128x16x32x32xf32>
    %v526 = stablehlo.add %v525, %v517 : tensor<128x16x32x32xf32>
    %v527 = stablehlo.rsqrt %v526 : tensor<128x16x32x32xf32>
    %v528 = stablehlo.multiply %v521, %v527 : tensor<128x16x32x32xf32>
    %v529 = stablehlo.broadcast_in_dim %g2, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v530 = stablehlo.multiply %v529, %v513 : tensor<128x16x32x32xf32>
    %v531 = stablehlo.reduce(%v530 init: %v515) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v532 = stablehlo.broadcast_in_dim %v531, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v533 = stablehlo.multiply %v528, %v530 : tensor<128x16x32x32xf32>
    %v534 = stablehlo.reduce(%v533 init: %v515) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v535 = stablehlo.broadcast_in_dim %v534, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v536 = stablehlo.multiply %v530, %v516 : tensor<128x16x32x32xf32>
    %v537 = stablehlo.subtract %v536, %v532 : tensor<128x16x32x32xf32>
    %v538 = stablehlo.multiply %v528, %v535 : tensor<128x16x32x32xf32>
    %v539 = stablehlo.subtract %v537, %v538 : tensor<128x16x32x32xf32>
    %v540 = stablehlo.divide %v527, %v516 : tensor<128x16x32x32xf32>
    %v541 = stablehlo.multiply %v540, %v539 : tensor<128x16x32x32xf32>
    %v542 = stablehlo.reshape %v541 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v543 = stablehlo.reshape %v542 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v544 = stablehlo.transpose %W2, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v545 = stablehlo.reverse %v544, dims = [2, 3] : tensor<16x16x3x3xf32>
    %v546 = stablehlo.convolution(%v543, %v545)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x16x32x32xf32>, tensor<16x16x3x3xf32>) -> tensor<128x16x32x32xf32>
    %v547 = stablehlo.reshape %v546 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v548 = stablehlo.constant dense<0.0> : tensor<128x16384xf32>
    %v549 = stablehlo.compare GT, %v24, %v548 : (tensor<128x16384xf32>, tensor<128x16384xf32>) -> tensor<128x16384xi1>
    %v550 = stablehlo.select %v549, %v547, %v548 : tensor<128x16384xi1>, tensor<128x16384xf32>
    %v551 = stablehlo.reshape %v550 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v552 = stablehlo.reshape %v4 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v553 = stablehlo.constant dense<0.0> : tensor<f32>
    %v554 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v555 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v556 = stablehlo.reduce(%v552 init: %v553) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v557 = stablehlo.broadcast_in_dim %v556, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v558 = stablehlo.divide %v557, %v554 : tensor<128x16x32x32xf32>
    %v559 = stablehlo.subtract %v552, %v558 : tensor<128x16x32x32xf32>
    %v560 = stablehlo.multiply %v559, %v559 : tensor<128x16x32x32xf32>
    %v561 = stablehlo.reduce(%v560 init: %v553) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v562 = stablehlo.broadcast_in_dim %v561, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v563 = stablehlo.divide %v562, %v554 : tensor<128x16x32x32xf32>
    %v564 = stablehlo.add %v563, %v555 : tensor<128x16x32x32xf32>
    %v565 = stablehlo.rsqrt %v564 : tensor<128x16x32x32xf32>
    %v566 = stablehlo.multiply %v559, %v565 : tensor<128x16x32x32xf32>
    %v567 = stablehlo.broadcast_in_dim %g1, dims = [1] : (tensor<16xf32>) -> tensor<128x16x32x32xf32>
    %v568 = stablehlo.multiply %v567, %v551 : tensor<128x16x32x32xf32>
    %v569 = stablehlo.reduce(%v568 init: %v553) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v570 = stablehlo.broadcast_in_dim %v569, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v571 = stablehlo.multiply %v566, %v568 : tensor<128x16x32x32xf32>
    %v572 = stablehlo.reduce(%v571 init: %v553) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v573 = stablehlo.broadcast_in_dim %v572, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v574 = stablehlo.multiply %v568, %v554 : tensor<128x16x32x32xf32>
    %v575 = stablehlo.subtract %v574, %v570 : tensor<128x16x32x32xf32>
    %v576 = stablehlo.multiply %v566, %v573 : tensor<128x16x32x32xf32>
    %v577 = stablehlo.subtract %v575, %v576 : tensor<128x16x32x32xf32>
    %v578 = stablehlo.divide %v565, %v554 : tensor<128x16x32x32xf32>
    %v579 = stablehlo.multiply %v578, %v577 : tensor<128x16x32x32xf32>
    %v580 = stablehlo.reshape %v579 : (tensor<128x16x32x32xf32>) -> tensor<128x16384xf32>
    %v581 = stablehlo.reshape %x : (tensor<128x3072xf32>) -> tensor<128x3x32x32xf32>
    %v582 = stablehlo.reshape %v580 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v583 = stablehlo.transpose %v581, dims = [1, 0, 2, 3] : (tensor<128x3x32x32xf32>) -> tensor<3x128x32x32xf32>
    %v584 = stablehlo.transpose %v582, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v585 = stablehlo.convolution(%v583, %v584)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<3x16x3x3xf32>
    %v586 = stablehlo.transpose %v585, dims = [1, 0, 2, 3] : (tensor<3x16x3x3xf32>) -> tensor<16x3x3x3xf32>
    %v587 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v588 = stablehlo.multiply %v587, %v586 : tensor<16x3x3x3xf32>
    %v589 = stablehlo.subtract %W1, %v588 : tensor<16x3x3x3xf32>
    %v590 = stablehlo.reshape %v580 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v591 = stablehlo.constant dense<0.0> : tensor<f32>
    %v592 = stablehlo.reduce(%v590 init: %v591) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v593 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v594 = stablehlo.multiply %v593, %v592 : tensor<16xf32>
    %v595 = stablehlo.subtract %cb1, %v594 : tensor<16xf32>
    %v596 = stablehlo.constant dense<0.0> : tensor<f32>
    %v597 = stablehlo.reshape %v4 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v598 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v599 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v600 = stablehlo.reduce(%v597 init: %v596) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v601 = stablehlo.broadcast_in_dim %v600, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v602 = stablehlo.divide %v601, %v598 : tensor<128x16x32x32xf32>
    %v603 = stablehlo.subtract %v597, %v602 : tensor<128x16x32x32xf32>
    %v604 = stablehlo.multiply %v603, %v603 : tensor<128x16x32x32xf32>
    %v605 = stablehlo.reduce(%v604 init: %v596) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v606 = stablehlo.broadcast_in_dim %v605, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v607 = stablehlo.divide %v606, %v598 : tensor<128x16x32x32xf32>
    %v608 = stablehlo.add %v607, %v599 : tensor<128x16x32x32xf32>
    %v609 = stablehlo.rsqrt %v608 : tensor<128x16x32x32xf32>
    %v610 = stablehlo.multiply %v603, %v609 : tensor<128x16x32x32xf32>
    %v611 = stablehlo.reshape %v550 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v612 = stablehlo.multiply %v611, %v610 : tensor<128x16x32x32xf32>
    %v613 = stablehlo.reduce(%v612 init: %v596) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v614 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v615 = stablehlo.multiply %v614, %v613 : tensor<16xf32>
    %v616 = stablehlo.subtract %g1, %v615 : tensor<16xf32>
    %v617 = stablehlo.constant dense<0.0> : tensor<f32>
    %v618 = stablehlo.reshape %v550 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v619 = stablehlo.reduce(%v618 init: %v617) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v620 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v621 = stablehlo.multiply %v620, %v619 : tensor<16xf32>
    %v622 = stablehlo.subtract %bt1, %v621 : tensor<16xf32>
    %v623 = stablehlo.reshape %v26 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v624 = stablehlo.reshape %v542 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v625 = stablehlo.transpose %v623, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v626 = stablehlo.transpose %v624, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v627 = stablehlo.convolution(%v625, %v626)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x16x3x3xf32>
    %v628 = stablehlo.transpose %v627, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v629 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v630 = stablehlo.multiply %v629, %v628 : tensor<16x16x3x3xf32>
    %v631 = stablehlo.subtract %W2, %v630 : tensor<16x16x3x3xf32>
    %v632 = stablehlo.reshape %v542 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v633 = stablehlo.constant dense<0.0> : tensor<f32>
    %v634 = stablehlo.reduce(%v632 init: %v633) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v635 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v636 = stablehlo.multiply %v635, %v634 : tensor<16xf32>
    %v637 = stablehlo.subtract %cb2, %v636 : tensor<16xf32>
    %v638 = stablehlo.constant dense<0.0> : tensor<f32>
    %v639 = stablehlo.reshape %v31 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v640 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v641 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v642 = stablehlo.reduce(%v639 init: %v638) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v643 = stablehlo.broadcast_in_dim %v642, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v644 = stablehlo.divide %v643, %v640 : tensor<128x16x32x32xf32>
    %v645 = stablehlo.subtract %v639, %v644 : tensor<128x16x32x32xf32>
    %v646 = stablehlo.multiply %v645, %v645 : tensor<128x16x32x32xf32>
    %v647 = stablehlo.reduce(%v646 init: %v638) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v648 = stablehlo.broadcast_in_dim %v647, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v649 = stablehlo.divide %v648, %v640 : tensor<128x16x32x32xf32>
    %v650 = stablehlo.add %v649, %v641 : tensor<128x16x32x32xf32>
    %v651 = stablehlo.rsqrt %v650 : tensor<128x16x32x32xf32>
    %v652 = stablehlo.multiply %v645, %v651 : tensor<128x16x32x32xf32>
    %v653 = stablehlo.reshape %v512 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v654 = stablehlo.multiply %v653, %v652 : tensor<128x16x32x32xf32>
    %v655 = stablehlo.reduce(%v654 init: %v638) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v656 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v657 = stablehlo.multiply %v656, %v655 : tensor<16xf32>
    %v658 = stablehlo.subtract %g2, %v657 : tensor<16xf32>
    %v659 = stablehlo.constant dense<0.0> : tensor<f32>
    %v660 = stablehlo.reshape %v512 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v661 = stablehlo.reduce(%v660 init: %v659) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v662 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v663 = stablehlo.multiply %v662, %v661 : tensor<16xf32>
    %v664 = stablehlo.subtract %bt2, %v663 : tensor<16xf32>
    %v665 = stablehlo.reshape %v57 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v666 = stablehlo.reshape %v499 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v667 = stablehlo.transpose %v665, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v668 = stablehlo.transpose %v666, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v669 = stablehlo.convolution(%v667, %v668)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v670 = stablehlo.transpose %v669, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v671 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v672 = stablehlo.multiply %v671, %v670 : tensor<16x16x3x3xf32>
    %v673 = stablehlo.subtract %W3, %v672 : tensor<16x16x3x3xf32>
    %v674 = stablehlo.reshape %v499 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v675 = stablehlo.constant dense<0.0> : tensor<f32>
    %v676 = stablehlo.reduce(%v674 init: %v675) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v677 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v678 = stablehlo.multiply %v677, %v676 : tensor<16xf32>
    %v679 = stablehlo.subtract %cb3, %v678 : tensor<16xf32>
    %v680 = stablehlo.constant dense<0.0> : tensor<f32>
    %v681 = stablehlo.reshape %v62 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v682 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v683 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v684 = stablehlo.reduce(%v681 init: %v680) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v685 = stablehlo.broadcast_in_dim %v684, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v686 = stablehlo.divide %v685, %v682 : tensor<128x16x16x16xf32>
    %v687 = stablehlo.subtract %v681, %v686 : tensor<128x16x16x16xf32>
    %v688 = stablehlo.multiply %v687, %v687 : tensor<128x16x16x16xf32>
    %v689 = stablehlo.reduce(%v688 init: %v680) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v690 = stablehlo.broadcast_in_dim %v689, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v691 = stablehlo.divide %v690, %v682 : tensor<128x16x16x16xf32>
    %v692 = stablehlo.add %v691, %v683 : tensor<128x16x16x16xf32>
    %v693 = stablehlo.rsqrt %v692 : tensor<128x16x16x16xf32>
    %v694 = stablehlo.multiply %v687, %v693 : tensor<128x16x16x16xf32>
    %v695 = stablehlo.reshape %v469 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v696 = stablehlo.multiply %v695, %v694 : tensor<128x16x16x16xf32>
    %v697 = stablehlo.reduce(%v696 init: %v680) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v698 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v699 = stablehlo.multiply %v698, %v697 : tensor<16xf32>
    %v700 = stablehlo.subtract %g3, %v699 : tensor<16xf32>
    %v701 = stablehlo.constant dense<0.0> : tensor<f32>
    %v702 = stablehlo.reshape %v469 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v703 = stablehlo.reduce(%v702 init: %v701) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v704 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v705 = stablehlo.multiply %v704, %v703 : tensor<16xf32>
    %v706 = stablehlo.subtract %bt3, %v705 : tensor<16xf32>
    %v707 = stablehlo.reshape %v84 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v708 = stablehlo.reshape %v461 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v709 = stablehlo.transpose %v707, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v710 = stablehlo.transpose %v708, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v711 = stablehlo.convolution(%v709, %v710)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v712 = stablehlo.transpose %v711, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v713 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v714 = stablehlo.multiply %v713, %v712 : tensor<16x16x3x3xf32>
    %v715 = stablehlo.subtract %W4, %v714 : tensor<16x16x3x3xf32>
    %v716 = stablehlo.reshape %v461 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v717 = stablehlo.constant dense<0.0> : tensor<f32>
    %v718 = stablehlo.reduce(%v716 init: %v717) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v719 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v720 = stablehlo.multiply %v719, %v718 : tensor<16xf32>
    %v721 = stablehlo.subtract %cb4, %v720 : tensor<16xf32>
    %v722 = stablehlo.constant dense<0.0> : tensor<f32>
    %v723 = stablehlo.reshape %v89 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v724 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v725 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v726 = stablehlo.reduce(%v723 init: %v722) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v727 = stablehlo.broadcast_in_dim %v726, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v728 = stablehlo.divide %v727, %v724 : tensor<128x16x16x16xf32>
    %v729 = stablehlo.subtract %v723, %v728 : tensor<128x16x16x16xf32>
    %v730 = stablehlo.multiply %v729, %v729 : tensor<128x16x16x16xf32>
    %v731 = stablehlo.reduce(%v730 init: %v722) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v732 = stablehlo.broadcast_in_dim %v731, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v733 = stablehlo.divide %v732, %v724 : tensor<128x16x16x16xf32>
    %v734 = stablehlo.add %v733, %v725 : tensor<128x16x16x16xf32>
    %v735 = stablehlo.rsqrt %v734 : tensor<128x16x16x16xf32>
    %v736 = stablehlo.multiply %v729, %v735 : tensor<128x16x16x16xf32>
    %v737 = stablehlo.reshape %v431 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v738 = stablehlo.multiply %v737, %v736 : tensor<128x16x16x16xf32>
    %v739 = stablehlo.reduce(%v738 init: %v722) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v740 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v741 = stablehlo.multiply %v740, %v739 : tensor<16xf32>
    %v742 = stablehlo.subtract %g4, %v741 : tensor<16xf32>
    %v743 = stablehlo.constant dense<0.0> : tensor<f32>
    %v744 = stablehlo.reshape %v431 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v745 = stablehlo.reduce(%v744 init: %v743) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v746 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v747 = stablehlo.multiply %v746, %v745 : tensor<16xf32>
    %v748 = stablehlo.subtract %bt4, %v747 : tensor<16xf32>
    %v749 = stablehlo.reshape %v115 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v750 = stablehlo.reshape %v418 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v751 = stablehlo.transpose %v749, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v752 = stablehlo.transpose %v750, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v753 = stablehlo.convolution(%v751, %v752)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<16x32x3x3xf32>
    %v754 = stablehlo.transpose %v753, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v755 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v756 = stablehlo.multiply %v755, %v754 : tensor<32x16x3x3xf32>
    %v757 = stablehlo.subtract %W5, %v756 : tensor<32x16x3x3xf32>
    %v758 = stablehlo.reshape %v418 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v759 = stablehlo.constant dense<0.0> : tensor<f32>
    %v760 = stablehlo.reduce(%v758 init: %v759) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v761 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v762 = stablehlo.multiply %v761, %v760 : tensor<32xf32>
    %v763 = stablehlo.subtract %cb5, %v762 : tensor<32xf32>
    %v764 = stablehlo.constant dense<0.0> : tensor<f32>
    %v765 = stablehlo.reshape %v120 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v766 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v767 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v768 = stablehlo.reduce(%v765 init: %v764) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v769 = stablehlo.broadcast_in_dim %v768, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v770 = stablehlo.divide %v769, %v766 : tensor<128x32x8x8xf32>
    %v771 = stablehlo.subtract %v765, %v770 : tensor<128x32x8x8xf32>
    %v772 = stablehlo.multiply %v771, %v771 : tensor<128x32x8x8xf32>
    %v773 = stablehlo.reduce(%v772 init: %v764) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v774 = stablehlo.broadcast_in_dim %v773, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v775 = stablehlo.divide %v774, %v766 : tensor<128x32x8x8xf32>
    %v776 = stablehlo.add %v775, %v767 : tensor<128x32x8x8xf32>
    %v777 = stablehlo.rsqrt %v776 : tensor<128x32x8x8xf32>
    %v778 = stablehlo.multiply %v771, %v777 : tensor<128x32x8x8xf32>
    %v779 = stablehlo.reshape %v388 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v780 = stablehlo.multiply %v779, %v778 : tensor<128x32x8x8xf32>
    %v781 = stablehlo.reduce(%v780 init: %v764) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v782 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v783 = stablehlo.multiply %v782, %v781 : tensor<32xf32>
    %v784 = stablehlo.subtract %g5, %v783 : tensor<32xf32>
    %v785 = stablehlo.constant dense<0.0> : tensor<f32>
    %v786 = stablehlo.reshape %v388 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v787 = stablehlo.reduce(%v786 init: %v785) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v788 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v789 = stablehlo.multiply %v788, %v787 : tensor<32xf32>
    %v790 = stablehlo.subtract %bt5, %v789 : tensor<32xf32>
    %v791 = stablehlo.reshape %v142 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v792 = stablehlo.reshape %v380 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v793 = stablehlo.transpose %v791, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v794 = stablehlo.transpose %v792, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v795 = stablehlo.convolution(%v793, %v794)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<32x32x3x3xf32>
    %v796 = stablehlo.transpose %v795, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v797 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v798 = stablehlo.multiply %v797, %v796 : tensor<32x32x3x3xf32>
    %v799 = stablehlo.subtract %W6, %v798 : tensor<32x32x3x3xf32>
    %v800 = stablehlo.reshape %v380 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v801 = stablehlo.constant dense<0.0> : tensor<f32>
    %v802 = stablehlo.reduce(%v800 init: %v801) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v803 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v804 = stablehlo.multiply %v803, %v802 : tensor<32xf32>
    %v805 = stablehlo.subtract %cb6, %v804 : tensor<32xf32>
    %v806 = stablehlo.constant dense<0.0> : tensor<f32>
    %v807 = stablehlo.reshape %v147 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v808 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v809 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v810 = stablehlo.reduce(%v807 init: %v806) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v811 = stablehlo.broadcast_in_dim %v810, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v812 = stablehlo.divide %v811, %v808 : tensor<128x32x8x8xf32>
    %v813 = stablehlo.subtract %v807, %v812 : tensor<128x32x8x8xf32>
    %v814 = stablehlo.multiply %v813, %v813 : tensor<128x32x8x8xf32>
    %v815 = stablehlo.reduce(%v814 init: %v806) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v816 = stablehlo.broadcast_in_dim %v815, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v817 = stablehlo.divide %v816, %v808 : tensor<128x32x8x8xf32>
    %v818 = stablehlo.add %v817, %v809 : tensor<128x32x8x8xf32>
    %v819 = stablehlo.rsqrt %v818 : tensor<128x32x8x8xf32>
    %v820 = stablehlo.multiply %v813, %v819 : tensor<128x32x8x8xf32>
    %v821 = stablehlo.reshape %v350 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v822 = stablehlo.multiply %v821, %v820 : tensor<128x32x8x8xf32>
    %v823 = stablehlo.reduce(%v822 init: %v806) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v824 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v825 = stablehlo.multiply %v824, %v823 : tensor<32xf32>
    %v826 = stablehlo.subtract %g6, %v825 : tensor<32xf32>
    %v827 = stablehlo.constant dense<0.0> : tensor<f32>
    %v828 = stablehlo.reshape %v350 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v829 = stablehlo.reduce(%v828 init: %v827) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v830 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v831 = stablehlo.multiply %v830, %v829 : tensor<32xf32>
    %v832 = stablehlo.subtract %bt6, %v831 : tensor<32xf32>
    %v833 = stablehlo.reshape %v173 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v834 = stablehlo.reshape %v337 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v835 = stablehlo.transpose %v833, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v836 = stablehlo.transpose %v834, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v837 = stablehlo.convolution(%v835, %v836)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v838 = stablehlo.transpose %v837, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v839 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v840 = stablehlo.multiply %v839, %v838 : tensor<32x32x3x3xf32>
    %v841 = stablehlo.subtract %W7, %v840 : tensor<32x32x3x3xf32>
    %v842 = stablehlo.reshape %v337 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v843 = stablehlo.constant dense<0.0> : tensor<f32>
    %v844 = stablehlo.reduce(%v842 init: %v843) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v845 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v846 = stablehlo.multiply %v845, %v844 : tensor<32xf32>
    %v847 = stablehlo.subtract %cb7, %v846 : tensor<32xf32>
    %v848 = stablehlo.constant dense<0.0> : tensor<f32>
    %v849 = stablehlo.reshape %v178 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v850 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v851 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v852 = stablehlo.reduce(%v849 init: %v848) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v853 = stablehlo.broadcast_in_dim %v852, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v854 = stablehlo.divide %v853, %v850 : tensor<128x32x4x4xf32>
    %v855 = stablehlo.subtract %v849, %v854 : tensor<128x32x4x4xf32>
    %v856 = stablehlo.multiply %v855, %v855 : tensor<128x32x4x4xf32>
    %v857 = stablehlo.reduce(%v856 init: %v848) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v858 = stablehlo.broadcast_in_dim %v857, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v859 = stablehlo.divide %v858, %v850 : tensor<128x32x4x4xf32>
    %v860 = stablehlo.add %v859, %v851 : tensor<128x32x4x4xf32>
    %v861 = stablehlo.rsqrt %v860 : tensor<128x32x4x4xf32>
    %v862 = stablehlo.multiply %v855, %v861 : tensor<128x32x4x4xf32>
    %v863 = stablehlo.reshape %v307 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v864 = stablehlo.multiply %v863, %v862 : tensor<128x32x4x4xf32>
    %v865 = stablehlo.reduce(%v864 init: %v848) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v866 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v867 = stablehlo.multiply %v866, %v865 : tensor<32xf32>
    %v868 = stablehlo.subtract %g7, %v867 : tensor<32xf32>
    %v869 = stablehlo.constant dense<0.0> : tensor<f32>
    %v870 = stablehlo.reshape %v307 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v871 = stablehlo.reduce(%v870 init: %v869) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v872 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v873 = stablehlo.multiply %v872, %v871 : tensor<32xf32>
    %v874 = stablehlo.subtract %bt7, %v873 : tensor<32xf32>
    %v875 = stablehlo.reshape %v200 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v876 = stablehlo.reshape %v299 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v877 = stablehlo.transpose %v875, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v878 = stablehlo.transpose %v876, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v879 = stablehlo.convolution(%v877, %v878)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v880 = stablehlo.transpose %v879, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v881 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v882 = stablehlo.multiply %v881, %v880 : tensor<32x32x3x3xf32>
    %v883 = stablehlo.subtract %W8, %v882 : tensor<32x32x3x3xf32>
    %v884 = stablehlo.reshape %v299 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v885 = stablehlo.constant dense<0.0> : tensor<f32>
    %v886 = stablehlo.reduce(%v884 init: %v885) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v887 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v888 = stablehlo.multiply %v887, %v886 : tensor<32xf32>
    %v889 = stablehlo.subtract %cb8, %v888 : tensor<32xf32>
    %v890 = stablehlo.constant dense<0.0> : tensor<f32>
    %v891 = stablehlo.reshape %v205 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v892 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v893 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v894 = stablehlo.reduce(%v891 init: %v890) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v895 = stablehlo.broadcast_in_dim %v894, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v896 = stablehlo.divide %v895, %v892 : tensor<128x32x4x4xf32>
    %v897 = stablehlo.subtract %v891, %v896 : tensor<128x32x4x4xf32>
    %v898 = stablehlo.multiply %v897, %v897 : tensor<128x32x4x4xf32>
    %v899 = stablehlo.reduce(%v898 init: %v890) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v900 = stablehlo.broadcast_in_dim %v899, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v901 = stablehlo.divide %v900, %v892 : tensor<128x32x4x4xf32>
    %v902 = stablehlo.add %v901, %v893 : tensor<128x32x4x4xf32>
    %v903 = stablehlo.rsqrt %v902 : tensor<128x32x4x4xf32>
    %v904 = stablehlo.multiply %v897, %v903 : tensor<128x32x4x4xf32>
    %v905 = stablehlo.reshape %v269 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v906 = stablehlo.multiply %v905, %v904 : tensor<128x32x4x4xf32>
    %v907 = stablehlo.reduce(%v906 init: %v890) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v908 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v909 = stablehlo.multiply %v908, %v907 : tensor<32xf32>
    %v910 = stablehlo.subtract %g8, %v909 : tensor<32xf32>
    %v911 = stablehlo.constant dense<0.0> : tensor<f32>
    %v912 = stablehlo.reshape %v269 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v913 = stablehlo.reduce(%v912 init: %v911) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v914 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v915 = stablehlo.multiply %v914, %v913 : tensor<32xf32>
    %v916 = stablehlo.subtract %bt8, %v915 : tensor<32xf32>
    %v917 = stablehlo.dot_general %v231, %v260, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %v918 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v919 = stablehlo.multiply %v918, %v917 : tensor<128x64xf32>
    %v920 = stablehlo.subtract %W9, %v919 : tensor<128x64xf32>
    %v921 = stablehlo.constant dense<0.0> : tensor<f32>
    %v922 = stablehlo.reduce(%v260 init: %v921) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v923 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v924 = stablehlo.multiply %v923, %v922 : tensor<64xf32>
    %v925 = stablehlo.subtract %b9, %v924 : tensor<64xf32>
    %v926 = stablehlo.dot_general %v236, %v256, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<64x64xf32>
    %v927 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v928 = stablehlo.multiply %v927, %v926 : tensor<64x64xf32>
    %v929 = stablehlo.subtract %Wa, %v928 : tensor<64x64xf32>
    %v930 = stablehlo.constant dense<0.0> : tensor<f32>
    %v931 = stablehlo.reduce(%v256 init: %v930) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v932 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v933 = stablehlo.multiply %v932, %v931 : tensor<64xf32>
    %v934 = stablehlo.subtract %ba, %v933 : tensor<64xf32>
    %v935 = stablehlo.dot_general %v241, %v252, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x10xf32>) -> tensor<64x10xf32>
    %v936 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v937 = stablehlo.multiply %v936, %v935 : tensor<64x10xf32>
    %v938 = stablehlo.subtract %Wb, %v937 : tensor<64x10xf32>
    %v939 = stablehlo.constant dense<0.0> : tensor<f32>
    %v940 = stablehlo.reduce(%v252 init: %v939) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v941 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v942 = stablehlo.multiply %v941, %v940 : tensor<10xf32>
    %v943 = stablehlo.subtract %bb, %v942 : tensor<10xf32>
    return %v589, %v595, %v616, %v622, %v631, %v637, %v658, %v664, %v673, %v679, %v700, %v706, %v715, %v721, %v742, %v748, %v757, %v763, %v784, %v790, %v799, %v805, %v826, %v832, %v841, %v847, %v868, %v874, %v883, %v889, %v910, %v916, %v920, %v925, %v929, %v934, %v938, %v943, %W1m, %cb1m, %g1m, %bt1m, %W2m, %cb2m, %g2m, %bt2m, %W3m, %cb3m, %g3m, %bt3m, %W4m, %cb4m, %g4m, %bt4m, %W5m, %cb5m, %g5m, %bt5m, %W6m, %cb6m, %g6m, %bt6m, %W7m, %cb7m, %g7m, %bt7m, %W8m, %cb8m, %g8m, %bt8m, %W9m, %b9m, %Wam, %bam, %Wbm, %bbm, %W1v, %cb1v, %g1v, %bt1v, %W2v, %cb2v, %g2v, %bt2v, %W3v, %cb3v, %g3v, %bt3v, %W4v, %cb4v, %g4v, %bt4v, %W5v, %cb5v, %g5v, %bt5v, %W6v, %cb6v, %g6v, %bt6v, %W7v, %cb7v, %g7v, %bt7v, %W8v, %cb8v, %g8v, %bt8v, %W9v, %b9v, %Wav, %bav, %Wbv, %bbv, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
