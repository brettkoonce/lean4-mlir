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
    %v587 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v588 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v589 = stablehlo.multiply %v587, %W1m : tensor<16x3x3x3xf32>
    %v590 = stablehlo.multiply %v588, %v586 : tensor<16x3x3x3xf32>
    %v591 = stablehlo.add %v589, %v590 : tensor<16x3x3x3xf32>
    %v592 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v593 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v594 = stablehlo.multiply %v592, %W1v : tensor<16x3x3x3xf32>
    %v595 = stablehlo.multiply %v586, %v586 : tensor<16x3x3x3xf32>
    %v596 = stablehlo.multiply %v593, %v595 : tensor<16x3x3x3xf32>
    %v597 = stablehlo.add %v594, %v596 : tensor<16x3x3x3xf32>
    %v598 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v599 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v600 = stablehlo.divide %v591, %v598 : tensor<16x3x3x3xf32>
    %v601 = stablehlo.divide %v597, %v599 : tensor<16x3x3x3xf32>
    %v602 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v603 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v604 = stablehlo.sqrt %v601 : tensor<16x3x3x3xf32>
    %v605 = stablehlo.add %v604, %v603 : tensor<16x3x3x3xf32>
    %v606 = stablehlo.divide %v600, %v605 : tensor<16x3x3x3xf32>
    %v607 = stablehlo.multiply %v602, %v606 : tensor<16x3x3x3xf32>
    %v608 = stablehlo.subtract %W1, %v607 : tensor<16x3x3x3xf32>
    %v609 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v610 = stablehlo.multiply %v609, %v602 : tensor<16x3x3x3xf32>
    %v611 = stablehlo.multiply %v610, %W1 : tensor<16x3x3x3xf32>
    %v612 = stablehlo.subtract %v608, %v611 : tensor<16x3x3x3xf32>
    %v613 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v614 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v615 = stablehlo.multiply %v613, %W1m : tensor<16x3x3x3xf32>
    %v616 = stablehlo.multiply %v614, %v586 : tensor<16x3x3x3xf32>
    %v617 = stablehlo.add %v615, %v616 : tensor<16x3x3x3xf32>
    %v618 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v619 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v620 = stablehlo.multiply %v618, %W1v : tensor<16x3x3x3xf32>
    %v621 = stablehlo.multiply %v586, %v586 : tensor<16x3x3x3xf32>
    %v622 = stablehlo.multiply %v619, %v621 : tensor<16x3x3x3xf32>
    %v623 = stablehlo.add %v620, %v622 : tensor<16x3x3x3xf32>
    %v624 = stablehlo.reshape %v580 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v625 = stablehlo.constant dense<0.0> : tensor<f32>
    %v626 = stablehlo.reduce(%v624 init: %v625) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v627 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v628 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v629 = stablehlo.multiply %v627, %cb1m : tensor<16xf32>
    %v630 = stablehlo.multiply %v628, %v626 : tensor<16xf32>
    %v631 = stablehlo.add %v629, %v630 : tensor<16xf32>
    %v632 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v633 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v634 = stablehlo.multiply %v632, %cb1v : tensor<16xf32>
    %v635 = stablehlo.multiply %v626, %v626 : tensor<16xf32>
    %v636 = stablehlo.multiply %v633, %v635 : tensor<16xf32>
    %v637 = stablehlo.add %v634, %v636 : tensor<16xf32>
    %v638 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v639 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v640 = stablehlo.divide %v631, %v638 : tensor<16xf32>
    %v641 = stablehlo.divide %v637, %v639 : tensor<16xf32>
    %v642 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v643 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v644 = stablehlo.sqrt %v641 : tensor<16xf32>
    %v645 = stablehlo.add %v644, %v643 : tensor<16xf32>
    %v646 = stablehlo.divide %v640, %v645 : tensor<16xf32>
    %v647 = stablehlo.multiply %v642, %v646 : tensor<16xf32>
    %v648 = stablehlo.subtract %cb1, %v647 : tensor<16xf32>
    %v649 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v650 = stablehlo.multiply %v649, %v642 : tensor<16xf32>
    %v651 = stablehlo.multiply %v650, %cb1 : tensor<16xf32>
    %v652 = stablehlo.subtract %v648, %v651 : tensor<16xf32>
    %v653 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v654 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v655 = stablehlo.multiply %v653, %cb1m : tensor<16xf32>
    %v656 = stablehlo.multiply %v654, %v626 : tensor<16xf32>
    %v657 = stablehlo.add %v655, %v656 : tensor<16xf32>
    %v658 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v659 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v660 = stablehlo.multiply %v658, %cb1v : tensor<16xf32>
    %v661 = stablehlo.multiply %v626, %v626 : tensor<16xf32>
    %v662 = stablehlo.multiply %v659, %v661 : tensor<16xf32>
    %v663 = stablehlo.add %v660, %v662 : tensor<16xf32>
    %v664 = stablehlo.constant dense<0.0> : tensor<f32>
    %v665 = stablehlo.reshape %v4 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v666 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v667 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v668 = stablehlo.reduce(%v665 init: %v664) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v669 = stablehlo.broadcast_in_dim %v668, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v670 = stablehlo.divide %v669, %v666 : tensor<128x16x32x32xf32>
    %v671 = stablehlo.subtract %v665, %v670 : tensor<128x16x32x32xf32>
    %v672 = stablehlo.multiply %v671, %v671 : tensor<128x16x32x32xf32>
    %v673 = stablehlo.reduce(%v672 init: %v664) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v674 = stablehlo.broadcast_in_dim %v673, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v675 = stablehlo.divide %v674, %v666 : tensor<128x16x32x32xf32>
    %v676 = stablehlo.add %v675, %v667 : tensor<128x16x32x32xf32>
    %v677 = stablehlo.rsqrt %v676 : tensor<128x16x32x32xf32>
    %v678 = stablehlo.multiply %v671, %v677 : tensor<128x16x32x32xf32>
    %v679 = stablehlo.reshape %v550 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v680 = stablehlo.multiply %v679, %v678 : tensor<128x16x32x32xf32>
    %v681 = stablehlo.reduce(%v680 init: %v664) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v682 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v683 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v684 = stablehlo.multiply %v682, %g1m : tensor<16xf32>
    %v685 = stablehlo.multiply %v683, %v681 : tensor<16xf32>
    %v686 = stablehlo.add %v684, %v685 : tensor<16xf32>
    %v687 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v688 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v689 = stablehlo.multiply %v687, %g1v : tensor<16xf32>
    %v690 = stablehlo.multiply %v681, %v681 : tensor<16xf32>
    %v691 = stablehlo.multiply %v688, %v690 : tensor<16xf32>
    %v692 = stablehlo.add %v689, %v691 : tensor<16xf32>
    %v693 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v694 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v695 = stablehlo.divide %v686, %v693 : tensor<16xf32>
    %v696 = stablehlo.divide %v692, %v694 : tensor<16xf32>
    %v697 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v698 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v699 = stablehlo.sqrt %v696 : tensor<16xf32>
    %v700 = stablehlo.add %v699, %v698 : tensor<16xf32>
    %v701 = stablehlo.divide %v695, %v700 : tensor<16xf32>
    %v702 = stablehlo.multiply %v697, %v701 : tensor<16xf32>
    %v703 = stablehlo.subtract %g1, %v702 : tensor<16xf32>
    %v704 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v705 = stablehlo.multiply %v704, %v697 : tensor<16xf32>
    %v706 = stablehlo.multiply %v705, %g1 : tensor<16xf32>
    %v707 = stablehlo.subtract %v703, %v706 : tensor<16xf32>
    %v708 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v709 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v710 = stablehlo.multiply %v708, %g1m : tensor<16xf32>
    %v711 = stablehlo.multiply %v709, %v681 : tensor<16xf32>
    %v712 = stablehlo.add %v710, %v711 : tensor<16xf32>
    %v713 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v714 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v715 = stablehlo.multiply %v713, %g1v : tensor<16xf32>
    %v716 = stablehlo.multiply %v681, %v681 : tensor<16xf32>
    %v717 = stablehlo.multiply %v714, %v716 : tensor<16xf32>
    %v718 = stablehlo.add %v715, %v717 : tensor<16xf32>
    %v719 = stablehlo.constant dense<0.0> : tensor<f32>
    %v720 = stablehlo.reshape %v550 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v721 = stablehlo.reduce(%v720 init: %v719) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v722 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v723 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v724 = stablehlo.multiply %v722, %bt1m : tensor<16xf32>
    %v725 = stablehlo.multiply %v723, %v721 : tensor<16xf32>
    %v726 = stablehlo.add %v724, %v725 : tensor<16xf32>
    %v727 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v728 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v729 = stablehlo.multiply %v727, %bt1v : tensor<16xf32>
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
    %v743 = stablehlo.subtract %bt1, %v742 : tensor<16xf32>
    %v744 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v745 = stablehlo.multiply %v744, %v737 : tensor<16xf32>
    %v746 = stablehlo.multiply %v745, %bt1 : tensor<16xf32>
    %v747 = stablehlo.subtract %v743, %v746 : tensor<16xf32>
    %v748 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v749 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v750 = stablehlo.multiply %v748, %bt1m : tensor<16xf32>
    %v751 = stablehlo.multiply %v749, %v721 : tensor<16xf32>
    %v752 = stablehlo.add %v750, %v751 : tensor<16xf32>
    %v753 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v754 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v755 = stablehlo.multiply %v753, %bt1v : tensor<16xf32>
    %v756 = stablehlo.multiply %v721, %v721 : tensor<16xf32>
    %v757 = stablehlo.multiply %v754, %v756 : tensor<16xf32>
    %v758 = stablehlo.add %v755, %v757 : tensor<16xf32>
    %v759 = stablehlo.reshape %v26 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v760 = stablehlo.reshape %v542 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v761 = stablehlo.transpose %v759, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v762 = stablehlo.transpose %v760, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v763 = stablehlo.convolution(%v761, %v762)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x16x3x3xf32>
    %v764 = stablehlo.transpose %v763, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v765 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v766 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v767 = stablehlo.multiply %v765, %W2m : tensor<16x16x3x3xf32>
    %v768 = stablehlo.multiply %v766, %v764 : tensor<16x16x3x3xf32>
    %v769 = stablehlo.add %v767, %v768 : tensor<16x16x3x3xf32>
    %v770 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v771 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v772 = stablehlo.multiply %v770, %W2v : tensor<16x16x3x3xf32>
    %v773 = stablehlo.multiply %v764, %v764 : tensor<16x16x3x3xf32>
    %v774 = stablehlo.multiply %v771, %v773 : tensor<16x16x3x3xf32>
    %v775 = stablehlo.add %v772, %v774 : tensor<16x16x3x3xf32>
    %v776 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v777 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v778 = stablehlo.divide %v769, %v776 : tensor<16x16x3x3xf32>
    %v779 = stablehlo.divide %v775, %v777 : tensor<16x16x3x3xf32>
    %v780 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v781 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v782 = stablehlo.sqrt %v779 : tensor<16x16x3x3xf32>
    %v783 = stablehlo.add %v782, %v781 : tensor<16x16x3x3xf32>
    %v784 = stablehlo.divide %v778, %v783 : tensor<16x16x3x3xf32>
    %v785 = stablehlo.multiply %v780, %v784 : tensor<16x16x3x3xf32>
    %v786 = stablehlo.subtract %W2, %v785 : tensor<16x16x3x3xf32>
    %v787 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v788 = stablehlo.multiply %v787, %v780 : tensor<16x16x3x3xf32>
    %v789 = stablehlo.multiply %v788, %W2 : tensor<16x16x3x3xf32>
    %v790 = stablehlo.subtract %v786, %v789 : tensor<16x16x3x3xf32>
    %v791 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v792 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v793 = stablehlo.multiply %v791, %W2m : tensor<16x16x3x3xf32>
    %v794 = stablehlo.multiply %v792, %v764 : tensor<16x16x3x3xf32>
    %v795 = stablehlo.add %v793, %v794 : tensor<16x16x3x3xf32>
    %v796 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v797 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v798 = stablehlo.multiply %v796, %W2v : tensor<16x16x3x3xf32>
    %v799 = stablehlo.multiply %v764, %v764 : tensor<16x16x3x3xf32>
    %v800 = stablehlo.multiply %v797, %v799 : tensor<16x16x3x3xf32>
    %v801 = stablehlo.add %v798, %v800 : tensor<16x16x3x3xf32>
    %v802 = stablehlo.reshape %v542 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v803 = stablehlo.constant dense<0.0> : tensor<f32>
    %v804 = stablehlo.reduce(%v802 init: %v803) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v805 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v806 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v807 = stablehlo.multiply %v805, %cb2m : tensor<16xf32>
    %v808 = stablehlo.multiply %v806, %v804 : tensor<16xf32>
    %v809 = stablehlo.add %v807, %v808 : tensor<16xf32>
    %v810 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v811 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v812 = stablehlo.multiply %v810, %cb2v : tensor<16xf32>
    %v813 = stablehlo.multiply %v804, %v804 : tensor<16xf32>
    %v814 = stablehlo.multiply %v811, %v813 : tensor<16xf32>
    %v815 = stablehlo.add %v812, %v814 : tensor<16xf32>
    %v816 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v817 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v818 = stablehlo.divide %v809, %v816 : tensor<16xf32>
    %v819 = stablehlo.divide %v815, %v817 : tensor<16xf32>
    %v820 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v821 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v822 = stablehlo.sqrt %v819 : tensor<16xf32>
    %v823 = stablehlo.add %v822, %v821 : tensor<16xf32>
    %v824 = stablehlo.divide %v818, %v823 : tensor<16xf32>
    %v825 = stablehlo.multiply %v820, %v824 : tensor<16xf32>
    %v826 = stablehlo.subtract %cb2, %v825 : tensor<16xf32>
    %v827 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v828 = stablehlo.multiply %v827, %v820 : tensor<16xf32>
    %v829 = stablehlo.multiply %v828, %cb2 : tensor<16xf32>
    %v830 = stablehlo.subtract %v826, %v829 : tensor<16xf32>
    %v831 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v832 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v833 = stablehlo.multiply %v831, %cb2m : tensor<16xf32>
    %v834 = stablehlo.multiply %v832, %v804 : tensor<16xf32>
    %v835 = stablehlo.add %v833, %v834 : tensor<16xf32>
    %v836 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v837 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v838 = stablehlo.multiply %v836, %cb2v : tensor<16xf32>
    %v839 = stablehlo.multiply %v804, %v804 : tensor<16xf32>
    %v840 = stablehlo.multiply %v837, %v839 : tensor<16xf32>
    %v841 = stablehlo.add %v838, %v840 : tensor<16xf32>
    %v842 = stablehlo.constant dense<0.0> : tensor<f32>
    %v843 = stablehlo.reshape %v31 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v844 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v845 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v846 = stablehlo.reduce(%v843 init: %v842) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v847 = stablehlo.broadcast_in_dim %v846, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v848 = stablehlo.divide %v847, %v844 : tensor<128x16x32x32xf32>
    %v849 = stablehlo.subtract %v843, %v848 : tensor<128x16x32x32xf32>
    %v850 = stablehlo.multiply %v849, %v849 : tensor<128x16x32x32xf32>
    %v851 = stablehlo.reduce(%v850 init: %v842) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v852 = stablehlo.broadcast_in_dim %v851, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v853 = stablehlo.divide %v852, %v844 : tensor<128x16x32x32xf32>
    %v854 = stablehlo.add %v853, %v845 : tensor<128x16x32x32xf32>
    %v855 = stablehlo.rsqrt %v854 : tensor<128x16x32x32xf32>
    %v856 = stablehlo.multiply %v849, %v855 : tensor<128x16x32x32xf32>
    %v857 = stablehlo.reshape %v512 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v858 = stablehlo.multiply %v857, %v856 : tensor<128x16x32x32xf32>
    %v859 = stablehlo.reduce(%v858 init: %v842) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v860 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v861 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v862 = stablehlo.multiply %v860, %g2m : tensor<16xf32>
    %v863 = stablehlo.multiply %v861, %v859 : tensor<16xf32>
    %v864 = stablehlo.add %v862, %v863 : tensor<16xf32>
    %v865 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v866 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v867 = stablehlo.multiply %v865, %g2v : tensor<16xf32>
    %v868 = stablehlo.multiply %v859, %v859 : tensor<16xf32>
    %v869 = stablehlo.multiply %v866, %v868 : tensor<16xf32>
    %v870 = stablehlo.add %v867, %v869 : tensor<16xf32>
    %v871 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v872 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v873 = stablehlo.divide %v864, %v871 : tensor<16xf32>
    %v874 = stablehlo.divide %v870, %v872 : tensor<16xf32>
    %v875 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v876 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v877 = stablehlo.sqrt %v874 : tensor<16xf32>
    %v878 = stablehlo.add %v877, %v876 : tensor<16xf32>
    %v879 = stablehlo.divide %v873, %v878 : tensor<16xf32>
    %v880 = stablehlo.multiply %v875, %v879 : tensor<16xf32>
    %v881 = stablehlo.subtract %g2, %v880 : tensor<16xf32>
    %v882 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v883 = stablehlo.multiply %v882, %v875 : tensor<16xf32>
    %v884 = stablehlo.multiply %v883, %g2 : tensor<16xf32>
    %v885 = stablehlo.subtract %v881, %v884 : tensor<16xf32>
    %v886 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v887 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v888 = stablehlo.multiply %v886, %g2m : tensor<16xf32>
    %v889 = stablehlo.multiply %v887, %v859 : tensor<16xf32>
    %v890 = stablehlo.add %v888, %v889 : tensor<16xf32>
    %v891 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v892 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v893 = stablehlo.multiply %v891, %g2v : tensor<16xf32>
    %v894 = stablehlo.multiply %v859, %v859 : tensor<16xf32>
    %v895 = stablehlo.multiply %v892, %v894 : tensor<16xf32>
    %v896 = stablehlo.add %v893, %v895 : tensor<16xf32>
    %v897 = stablehlo.constant dense<0.0> : tensor<f32>
    %v898 = stablehlo.reshape %v512 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v899 = stablehlo.reduce(%v898 init: %v897) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v900 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v901 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v902 = stablehlo.multiply %v900, %bt2m : tensor<16xf32>
    %v903 = stablehlo.multiply %v901, %v899 : tensor<16xf32>
    %v904 = stablehlo.add %v902, %v903 : tensor<16xf32>
    %v905 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v906 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v907 = stablehlo.multiply %v905, %bt2v : tensor<16xf32>
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
    %v921 = stablehlo.subtract %bt2, %v920 : tensor<16xf32>
    %v922 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v923 = stablehlo.multiply %v922, %v915 : tensor<16xf32>
    %v924 = stablehlo.multiply %v923, %bt2 : tensor<16xf32>
    %v925 = stablehlo.subtract %v921, %v924 : tensor<16xf32>
    %v926 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v927 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v928 = stablehlo.multiply %v926, %bt2m : tensor<16xf32>
    %v929 = stablehlo.multiply %v927, %v899 : tensor<16xf32>
    %v930 = stablehlo.add %v928, %v929 : tensor<16xf32>
    %v931 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v932 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v933 = stablehlo.multiply %v931, %bt2v : tensor<16xf32>
    %v934 = stablehlo.multiply %v899, %v899 : tensor<16xf32>
    %v935 = stablehlo.multiply %v932, %v934 : tensor<16xf32>
    %v936 = stablehlo.add %v933, %v935 : tensor<16xf32>
    %v937 = stablehlo.reshape %v57 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v938 = stablehlo.reshape %v499 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v939 = stablehlo.transpose %v937, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v940 = stablehlo.transpose %v938, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v941 = stablehlo.convolution(%v939, %v940)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v942 = stablehlo.transpose %v941, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v943 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v944 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v945 = stablehlo.multiply %v943, %W3m : tensor<16x16x3x3xf32>
    %v946 = stablehlo.multiply %v944, %v942 : tensor<16x16x3x3xf32>
    %v947 = stablehlo.add %v945, %v946 : tensor<16x16x3x3xf32>
    %v948 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v949 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v950 = stablehlo.multiply %v948, %W3v : tensor<16x16x3x3xf32>
    %v951 = stablehlo.multiply %v942, %v942 : tensor<16x16x3x3xf32>
    %v952 = stablehlo.multiply %v949, %v951 : tensor<16x16x3x3xf32>
    %v953 = stablehlo.add %v950, %v952 : tensor<16x16x3x3xf32>
    %v954 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v955 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v956 = stablehlo.divide %v947, %v954 : tensor<16x16x3x3xf32>
    %v957 = stablehlo.divide %v953, %v955 : tensor<16x16x3x3xf32>
    %v958 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v959 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v960 = stablehlo.sqrt %v957 : tensor<16x16x3x3xf32>
    %v961 = stablehlo.add %v960, %v959 : tensor<16x16x3x3xf32>
    %v962 = stablehlo.divide %v956, %v961 : tensor<16x16x3x3xf32>
    %v963 = stablehlo.multiply %v958, %v962 : tensor<16x16x3x3xf32>
    %v964 = stablehlo.subtract %W3, %v963 : tensor<16x16x3x3xf32>
    %v965 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v966 = stablehlo.multiply %v965, %v958 : tensor<16x16x3x3xf32>
    %v967 = stablehlo.multiply %v966, %W3 : tensor<16x16x3x3xf32>
    %v968 = stablehlo.subtract %v964, %v967 : tensor<16x16x3x3xf32>
    %v969 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v970 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v971 = stablehlo.multiply %v969, %W3m : tensor<16x16x3x3xf32>
    %v972 = stablehlo.multiply %v970, %v942 : tensor<16x16x3x3xf32>
    %v973 = stablehlo.add %v971, %v972 : tensor<16x16x3x3xf32>
    %v974 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v975 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v976 = stablehlo.multiply %v974, %W3v : tensor<16x16x3x3xf32>
    %v977 = stablehlo.multiply %v942, %v942 : tensor<16x16x3x3xf32>
    %v978 = stablehlo.multiply %v975, %v977 : tensor<16x16x3x3xf32>
    %v979 = stablehlo.add %v976, %v978 : tensor<16x16x3x3xf32>
    %v980 = stablehlo.reshape %v499 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v981 = stablehlo.constant dense<0.0> : tensor<f32>
    %v982 = stablehlo.reduce(%v980 init: %v981) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v983 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v984 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v985 = stablehlo.multiply %v983, %cb3m : tensor<16xf32>
    %v986 = stablehlo.multiply %v984, %v982 : tensor<16xf32>
    %v987 = stablehlo.add %v985, %v986 : tensor<16xf32>
    %v988 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v989 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v990 = stablehlo.multiply %v988, %cb3v : tensor<16xf32>
    %v991 = stablehlo.multiply %v982, %v982 : tensor<16xf32>
    %v992 = stablehlo.multiply %v989, %v991 : tensor<16xf32>
    %v993 = stablehlo.add %v990, %v992 : tensor<16xf32>
    %v994 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v995 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v996 = stablehlo.divide %v987, %v994 : tensor<16xf32>
    %v997 = stablehlo.divide %v993, %v995 : tensor<16xf32>
    %v998 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v999 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1000 = stablehlo.sqrt %v997 : tensor<16xf32>
    %v1001 = stablehlo.add %v1000, %v999 : tensor<16xf32>
    %v1002 = stablehlo.divide %v996, %v1001 : tensor<16xf32>
    %v1003 = stablehlo.multiply %v998, %v1002 : tensor<16xf32>
    %v1004 = stablehlo.subtract %cb3, %v1003 : tensor<16xf32>
    %v1005 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1006 = stablehlo.multiply %v1005, %v998 : tensor<16xf32>
    %v1007 = stablehlo.multiply %v1006, %cb3 : tensor<16xf32>
    %v1008 = stablehlo.subtract %v1004, %v1007 : tensor<16xf32>
    %v1009 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1010 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1011 = stablehlo.multiply %v1009, %cb3m : tensor<16xf32>
    %v1012 = stablehlo.multiply %v1010, %v982 : tensor<16xf32>
    %v1013 = stablehlo.add %v1011, %v1012 : tensor<16xf32>
    %v1014 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1015 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1016 = stablehlo.multiply %v1014, %cb3v : tensor<16xf32>
    %v1017 = stablehlo.multiply %v982, %v982 : tensor<16xf32>
    %v1018 = stablehlo.multiply %v1015, %v1017 : tensor<16xf32>
    %v1019 = stablehlo.add %v1016, %v1018 : tensor<16xf32>
    %v1020 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1021 = stablehlo.reshape %v62 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1022 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v1023 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v1024 = stablehlo.reduce(%v1021 init: %v1020) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v1025 = stablehlo.broadcast_in_dim %v1024, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v1026 = stablehlo.divide %v1025, %v1022 : tensor<128x16x16x16xf32>
    %v1027 = stablehlo.subtract %v1021, %v1026 : tensor<128x16x16x16xf32>
    %v1028 = stablehlo.multiply %v1027, %v1027 : tensor<128x16x16x16xf32>
    %v1029 = stablehlo.reduce(%v1028 init: %v1020) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v1030 = stablehlo.broadcast_in_dim %v1029, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v1031 = stablehlo.divide %v1030, %v1022 : tensor<128x16x16x16xf32>
    %v1032 = stablehlo.add %v1031, %v1023 : tensor<128x16x16x16xf32>
    %v1033 = stablehlo.rsqrt %v1032 : tensor<128x16x16x16xf32>
    %v1034 = stablehlo.multiply %v1027, %v1033 : tensor<128x16x16x16xf32>
    %v1035 = stablehlo.reshape %v469 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1036 = stablehlo.multiply %v1035, %v1034 : tensor<128x16x16x16xf32>
    %v1037 = stablehlo.reduce(%v1036 init: %v1020) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v1038 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1039 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1040 = stablehlo.multiply %v1038, %g3m : tensor<16xf32>
    %v1041 = stablehlo.multiply %v1039, %v1037 : tensor<16xf32>
    %v1042 = stablehlo.add %v1040, %v1041 : tensor<16xf32>
    %v1043 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1044 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1045 = stablehlo.multiply %v1043, %g3v : tensor<16xf32>
    %v1046 = stablehlo.multiply %v1037, %v1037 : tensor<16xf32>
    %v1047 = stablehlo.multiply %v1044, %v1046 : tensor<16xf32>
    %v1048 = stablehlo.add %v1045, %v1047 : tensor<16xf32>
    %v1049 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1050 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1051 = stablehlo.divide %v1042, %v1049 : tensor<16xf32>
    %v1052 = stablehlo.divide %v1048, %v1050 : tensor<16xf32>
    %v1053 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1054 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1055 = stablehlo.sqrt %v1052 : tensor<16xf32>
    %v1056 = stablehlo.add %v1055, %v1054 : tensor<16xf32>
    %v1057 = stablehlo.divide %v1051, %v1056 : tensor<16xf32>
    %v1058 = stablehlo.multiply %v1053, %v1057 : tensor<16xf32>
    %v1059 = stablehlo.subtract %g3, %v1058 : tensor<16xf32>
    %v1060 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1061 = stablehlo.multiply %v1060, %v1053 : tensor<16xf32>
    %v1062 = stablehlo.multiply %v1061, %g3 : tensor<16xf32>
    %v1063 = stablehlo.subtract %v1059, %v1062 : tensor<16xf32>
    %v1064 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1065 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1066 = stablehlo.multiply %v1064, %g3m : tensor<16xf32>
    %v1067 = stablehlo.multiply %v1065, %v1037 : tensor<16xf32>
    %v1068 = stablehlo.add %v1066, %v1067 : tensor<16xf32>
    %v1069 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1070 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1071 = stablehlo.multiply %v1069, %g3v : tensor<16xf32>
    %v1072 = stablehlo.multiply %v1037, %v1037 : tensor<16xf32>
    %v1073 = stablehlo.multiply %v1070, %v1072 : tensor<16xf32>
    %v1074 = stablehlo.add %v1071, %v1073 : tensor<16xf32>
    %v1075 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1076 = stablehlo.reshape %v469 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1077 = stablehlo.reduce(%v1076 init: %v1075) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v1078 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1079 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1080 = stablehlo.multiply %v1078, %bt3m : tensor<16xf32>
    %v1081 = stablehlo.multiply %v1079, %v1077 : tensor<16xf32>
    %v1082 = stablehlo.add %v1080, %v1081 : tensor<16xf32>
    %v1083 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1084 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1085 = stablehlo.multiply %v1083, %bt3v : tensor<16xf32>
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
    %v1099 = stablehlo.subtract %bt3, %v1098 : tensor<16xf32>
    %v1100 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1101 = stablehlo.multiply %v1100, %v1093 : tensor<16xf32>
    %v1102 = stablehlo.multiply %v1101, %bt3 : tensor<16xf32>
    %v1103 = stablehlo.subtract %v1099, %v1102 : tensor<16xf32>
    %v1104 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1105 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1106 = stablehlo.multiply %v1104, %bt3m : tensor<16xf32>
    %v1107 = stablehlo.multiply %v1105, %v1077 : tensor<16xf32>
    %v1108 = stablehlo.add %v1106, %v1107 : tensor<16xf32>
    %v1109 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1110 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1111 = stablehlo.multiply %v1109, %bt3v : tensor<16xf32>
    %v1112 = stablehlo.multiply %v1077, %v1077 : tensor<16xf32>
    %v1113 = stablehlo.multiply %v1110, %v1112 : tensor<16xf32>
    %v1114 = stablehlo.add %v1111, %v1113 : tensor<16xf32>
    %v1115 = stablehlo.reshape %v84 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1116 = stablehlo.reshape %v461 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1117 = stablehlo.transpose %v1115, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v1118 = stablehlo.transpose %v1116, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v1119 = stablehlo.convolution(%v1117, %v1118)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v1120 = stablehlo.transpose %v1119, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v1121 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1122 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1123 = stablehlo.multiply %v1121, %W4m : tensor<16x16x3x3xf32>
    %v1124 = stablehlo.multiply %v1122, %v1120 : tensor<16x16x3x3xf32>
    %v1125 = stablehlo.add %v1123, %v1124 : tensor<16x16x3x3xf32>
    %v1126 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1127 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1128 = stablehlo.multiply %v1126, %W4v : tensor<16x16x3x3xf32>
    %v1129 = stablehlo.multiply %v1120, %v1120 : tensor<16x16x3x3xf32>
    %v1130 = stablehlo.multiply %v1127, %v1129 : tensor<16x16x3x3xf32>
    %v1131 = stablehlo.add %v1128, %v1130 : tensor<16x16x3x3xf32>
    %v1132 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1133 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1134 = stablehlo.divide %v1125, %v1132 : tensor<16x16x3x3xf32>
    %v1135 = stablehlo.divide %v1131, %v1133 : tensor<16x16x3x3xf32>
    %v1136 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1137 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1138 = stablehlo.sqrt %v1135 : tensor<16x16x3x3xf32>
    %v1139 = stablehlo.add %v1138, %v1137 : tensor<16x16x3x3xf32>
    %v1140 = stablehlo.divide %v1134, %v1139 : tensor<16x16x3x3xf32>
    %v1141 = stablehlo.multiply %v1136, %v1140 : tensor<16x16x3x3xf32>
    %v1142 = stablehlo.subtract %W4, %v1141 : tensor<16x16x3x3xf32>
    %v1143 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1144 = stablehlo.multiply %v1143, %v1136 : tensor<16x16x3x3xf32>
    %v1145 = stablehlo.multiply %v1144, %W4 : tensor<16x16x3x3xf32>
    %v1146 = stablehlo.subtract %v1142, %v1145 : tensor<16x16x3x3xf32>
    %v1147 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1148 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1149 = stablehlo.multiply %v1147, %W4m : tensor<16x16x3x3xf32>
    %v1150 = stablehlo.multiply %v1148, %v1120 : tensor<16x16x3x3xf32>
    %v1151 = stablehlo.add %v1149, %v1150 : tensor<16x16x3x3xf32>
    %v1152 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1153 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v1154 = stablehlo.multiply %v1152, %W4v : tensor<16x16x3x3xf32>
    %v1155 = stablehlo.multiply %v1120, %v1120 : tensor<16x16x3x3xf32>
    %v1156 = stablehlo.multiply %v1153, %v1155 : tensor<16x16x3x3xf32>
    %v1157 = stablehlo.add %v1154, %v1156 : tensor<16x16x3x3xf32>
    %v1158 = stablehlo.reshape %v461 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1159 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1160 = stablehlo.reduce(%v1158 init: %v1159) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v1161 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1162 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1163 = stablehlo.multiply %v1161, %cb4m : tensor<16xf32>
    %v1164 = stablehlo.multiply %v1162, %v1160 : tensor<16xf32>
    %v1165 = stablehlo.add %v1163, %v1164 : tensor<16xf32>
    %v1166 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1167 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1168 = stablehlo.multiply %v1166, %cb4v : tensor<16xf32>
    %v1169 = stablehlo.multiply %v1160, %v1160 : tensor<16xf32>
    %v1170 = stablehlo.multiply %v1167, %v1169 : tensor<16xf32>
    %v1171 = stablehlo.add %v1168, %v1170 : tensor<16xf32>
    %v1172 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1173 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1174 = stablehlo.divide %v1165, %v1172 : tensor<16xf32>
    %v1175 = stablehlo.divide %v1171, %v1173 : tensor<16xf32>
    %v1176 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1177 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1178 = stablehlo.sqrt %v1175 : tensor<16xf32>
    %v1179 = stablehlo.add %v1178, %v1177 : tensor<16xf32>
    %v1180 = stablehlo.divide %v1174, %v1179 : tensor<16xf32>
    %v1181 = stablehlo.multiply %v1176, %v1180 : tensor<16xf32>
    %v1182 = stablehlo.subtract %cb4, %v1181 : tensor<16xf32>
    %v1183 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1184 = stablehlo.multiply %v1183, %v1176 : tensor<16xf32>
    %v1185 = stablehlo.multiply %v1184, %cb4 : tensor<16xf32>
    %v1186 = stablehlo.subtract %v1182, %v1185 : tensor<16xf32>
    %v1187 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1188 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1189 = stablehlo.multiply %v1187, %cb4m : tensor<16xf32>
    %v1190 = stablehlo.multiply %v1188, %v1160 : tensor<16xf32>
    %v1191 = stablehlo.add %v1189, %v1190 : tensor<16xf32>
    %v1192 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1193 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1194 = stablehlo.multiply %v1192, %cb4v : tensor<16xf32>
    %v1195 = stablehlo.multiply %v1160, %v1160 : tensor<16xf32>
    %v1196 = stablehlo.multiply %v1193, %v1195 : tensor<16xf32>
    %v1197 = stablehlo.add %v1194, %v1196 : tensor<16xf32>
    %v1198 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1199 = stablehlo.reshape %v89 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1200 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v1201 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v1202 = stablehlo.reduce(%v1199 init: %v1198) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v1203 = stablehlo.broadcast_in_dim %v1202, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v1204 = stablehlo.divide %v1203, %v1200 : tensor<128x16x16x16xf32>
    %v1205 = stablehlo.subtract %v1199, %v1204 : tensor<128x16x16x16xf32>
    %v1206 = stablehlo.multiply %v1205, %v1205 : tensor<128x16x16x16xf32>
    %v1207 = stablehlo.reduce(%v1206 init: %v1198) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v1208 = stablehlo.broadcast_in_dim %v1207, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v1209 = stablehlo.divide %v1208, %v1200 : tensor<128x16x16x16xf32>
    %v1210 = stablehlo.add %v1209, %v1201 : tensor<128x16x16x16xf32>
    %v1211 = stablehlo.rsqrt %v1210 : tensor<128x16x16x16xf32>
    %v1212 = stablehlo.multiply %v1205, %v1211 : tensor<128x16x16x16xf32>
    %v1213 = stablehlo.reshape %v431 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1214 = stablehlo.multiply %v1213, %v1212 : tensor<128x16x16x16xf32>
    %v1215 = stablehlo.reduce(%v1214 init: %v1198) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v1216 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1217 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1218 = stablehlo.multiply %v1216, %g4m : tensor<16xf32>
    %v1219 = stablehlo.multiply %v1217, %v1215 : tensor<16xf32>
    %v1220 = stablehlo.add %v1218, %v1219 : tensor<16xf32>
    %v1221 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1222 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1223 = stablehlo.multiply %v1221, %g4v : tensor<16xf32>
    %v1224 = stablehlo.multiply %v1215, %v1215 : tensor<16xf32>
    %v1225 = stablehlo.multiply %v1222, %v1224 : tensor<16xf32>
    %v1226 = stablehlo.add %v1223, %v1225 : tensor<16xf32>
    %v1227 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1228 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1229 = stablehlo.divide %v1220, %v1227 : tensor<16xf32>
    %v1230 = stablehlo.divide %v1226, %v1228 : tensor<16xf32>
    %v1231 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1232 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1233 = stablehlo.sqrt %v1230 : tensor<16xf32>
    %v1234 = stablehlo.add %v1233, %v1232 : tensor<16xf32>
    %v1235 = stablehlo.divide %v1229, %v1234 : tensor<16xf32>
    %v1236 = stablehlo.multiply %v1231, %v1235 : tensor<16xf32>
    %v1237 = stablehlo.subtract %g4, %v1236 : tensor<16xf32>
    %v1238 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1239 = stablehlo.multiply %v1238, %v1231 : tensor<16xf32>
    %v1240 = stablehlo.multiply %v1239, %g4 : tensor<16xf32>
    %v1241 = stablehlo.subtract %v1237, %v1240 : tensor<16xf32>
    %v1242 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1243 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1244 = stablehlo.multiply %v1242, %g4m : tensor<16xf32>
    %v1245 = stablehlo.multiply %v1243, %v1215 : tensor<16xf32>
    %v1246 = stablehlo.add %v1244, %v1245 : tensor<16xf32>
    %v1247 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1248 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1249 = stablehlo.multiply %v1247, %g4v : tensor<16xf32>
    %v1250 = stablehlo.multiply %v1215, %v1215 : tensor<16xf32>
    %v1251 = stablehlo.multiply %v1248, %v1250 : tensor<16xf32>
    %v1252 = stablehlo.add %v1249, %v1251 : tensor<16xf32>
    %v1253 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1254 = stablehlo.reshape %v431 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v1255 = stablehlo.reduce(%v1254 init: %v1253) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v1256 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1257 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1258 = stablehlo.multiply %v1256, %bt4m : tensor<16xf32>
    %v1259 = stablehlo.multiply %v1257, %v1255 : tensor<16xf32>
    %v1260 = stablehlo.add %v1258, %v1259 : tensor<16xf32>
    %v1261 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1262 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1263 = stablehlo.multiply %v1261, %bt4v : tensor<16xf32>
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
    %v1277 = stablehlo.subtract %bt4, %v1276 : tensor<16xf32>
    %v1278 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1279 = stablehlo.multiply %v1278, %v1271 : tensor<16xf32>
    %v1280 = stablehlo.multiply %v1279, %bt4 : tensor<16xf32>
    %v1281 = stablehlo.subtract %v1277, %v1280 : tensor<16xf32>
    %v1282 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1283 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1284 = stablehlo.multiply %v1282, %bt4m : tensor<16xf32>
    %v1285 = stablehlo.multiply %v1283, %v1255 : tensor<16xf32>
    %v1286 = stablehlo.add %v1284, %v1285 : tensor<16xf32>
    %v1287 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1288 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v1289 = stablehlo.multiply %v1287, %bt4v : tensor<16xf32>
    %v1290 = stablehlo.multiply %v1255, %v1255 : tensor<16xf32>
    %v1291 = stablehlo.multiply %v1288, %v1290 : tensor<16xf32>
    %v1292 = stablehlo.add %v1289, %v1291 : tensor<16xf32>
    %v1293 = stablehlo.reshape %v115 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v1294 = stablehlo.reshape %v418 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1295 = stablehlo.transpose %v1293, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v1296 = stablehlo.transpose %v1294, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v1297 = stablehlo.convolution(%v1295, %v1296)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<16x32x3x3xf32>
    %v1298 = stablehlo.transpose %v1297, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v1299 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1300 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1301 = stablehlo.multiply %v1299, %W5m : tensor<32x16x3x3xf32>
    %v1302 = stablehlo.multiply %v1300, %v1298 : tensor<32x16x3x3xf32>
    %v1303 = stablehlo.add %v1301, %v1302 : tensor<32x16x3x3xf32>
    %v1304 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1305 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1306 = stablehlo.multiply %v1304, %W5v : tensor<32x16x3x3xf32>
    %v1307 = stablehlo.multiply %v1298, %v1298 : tensor<32x16x3x3xf32>
    %v1308 = stablehlo.multiply %v1305, %v1307 : tensor<32x16x3x3xf32>
    %v1309 = stablehlo.add %v1306, %v1308 : tensor<32x16x3x3xf32>
    %v1310 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1311 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1312 = stablehlo.divide %v1303, %v1310 : tensor<32x16x3x3xf32>
    %v1313 = stablehlo.divide %v1309, %v1311 : tensor<32x16x3x3xf32>
    %v1314 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1315 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1316 = stablehlo.sqrt %v1313 : tensor<32x16x3x3xf32>
    %v1317 = stablehlo.add %v1316, %v1315 : tensor<32x16x3x3xf32>
    %v1318 = stablehlo.divide %v1312, %v1317 : tensor<32x16x3x3xf32>
    %v1319 = stablehlo.multiply %v1314, %v1318 : tensor<32x16x3x3xf32>
    %v1320 = stablehlo.subtract %W5, %v1319 : tensor<32x16x3x3xf32>
    %v1321 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1322 = stablehlo.multiply %v1321, %v1314 : tensor<32x16x3x3xf32>
    %v1323 = stablehlo.multiply %v1322, %W5 : tensor<32x16x3x3xf32>
    %v1324 = stablehlo.subtract %v1320, %v1323 : tensor<32x16x3x3xf32>
    %v1325 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1326 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1327 = stablehlo.multiply %v1325, %W5m : tensor<32x16x3x3xf32>
    %v1328 = stablehlo.multiply %v1326, %v1298 : tensor<32x16x3x3xf32>
    %v1329 = stablehlo.add %v1327, %v1328 : tensor<32x16x3x3xf32>
    %v1330 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1331 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v1332 = stablehlo.multiply %v1330, %W5v : tensor<32x16x3x3xf32>
    %v1333 = stablehlo.multiply %v1298, %v1298 : tensor<32x16x3x3xf32>
    %v1334 = stablehlo.multiply %v1331, %v1333 : tensor<32x16x3x3xf32>
    %v1335 = stablehlo.add %v1332, %v1334 : tensor<32x16x3x3xf32>
    %v1336 = stablehlo.reshape %v418 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1337 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1338 = stablehlo.reduce(%v1336 init: %v1337) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1339 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1340 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1341 = stablehlo.multiply %v1339, %cb5m : tensor<32xf32>
    %v1342 = stablehlo.multiply %v1340, %v1338 : tensor<32xf32>
    %v1343 = stablehlo.add %v1341, %v1342 : tensor<32xf32>
    %v1344 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1345 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1346 = stablehlo.multiply %v1344, %cb5v : tensor<32xf32>
    %v1347 = stablehlo.multiply %v1338, %v1338 : tensor<32xf32>
    %v1348 = stablehlo.multiply %v1345, %v1347 : tensor<32xf32>
    %v1349 = stablehlo.add %v1346, %v1348 : tensor<32xf32>
    %v1350 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1351 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1352 = stablehlo.divide %v1343, %v1350 : tensor<32xf32>
    %v1353 = stablehlo.divide %v1349, %v1351 : tensor<32xf32>
    %v1354 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1355 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1356 = stablehlo.sqrt %v1353 : tensor<32xf32>
    %v1357 = stablehlo.add %v1356, %v1355 : tensor<32xf32>
    %v1358 = stablehlo.divide %v1352, %v1357 : tensor<32xf32>
    %v1359 = stablehlo.multiply %v1354, %v1358 : tensor<32xf32>
    %v1360 = stablehlo.subtract %cb5, %v1359 : tensor<32xf32>
    %v1361 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1362 = stablehlo.multiply %v1361, %v1354 : tensor<32xf32>
    %v1363 = stablehlo.multiply %v1362, %cb5 : tensor<32xf32>
    %v1364 = stablehlo.subtract %v1360, %v1363 : tensor<32xf32>
    %v1365 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1366 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1367 = stablehlo.multiply %v1365, %cb5m : tensor<32xf32>
    %v1368 = stablehlo.multiply %v1366, %v1338 : tensor<32xf32>
    %v1369 = stablehlo.add %v1367, %v1368 : tensor<32xf32>
    %v1370 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1371 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1372 = stablehlo.multiply %v1370, %cb5v : tensor<32xf32>
    %v1373 = stablehlo.multiply %v1338, %v1338 : tensor<32xf32>
    %v1374 = stablehlo.multiply %v1371, %v1373 : tensor<32xf32>
    %v1375 = stablehlo.add %v1372, %v1374 : tensor<32xf32>
    %v1376 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1377 = stablehlo.reshape %v120 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1378 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v1379 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v1380 = stablehlo.reduce(%v1377 init: %v1376) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1381 = stablehlo.broadcast_in_dim %v1380, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v1382 = stablehlo.divide %v1381, %v1378 : tensor<128x32x8x8xf32>
    %v1383 = stablehlo.subtract %v1377, %v1382 : tensor<128x32x8x8xf32>
    %v1384 = stablehlo.multiply %v1383, %v1383 : tensor<128x32x8x8xf32>
    %v1385 = stablehlo.reduce(%v1384 init: %v1376) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1386 = stablehlo.broadcast_in_dim %v1385, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v1387 = stablehlo.divide %v1386, %v1378 : tensor<128x32x8x8xf32>
    %v1388 = stablehlo.add %v1387, %v1379 : tensor<128x32x8x8xf32>
    %v1389 = stablehlo.rsqrt %v1388 : tensor<128x32x8x8xf32>
    %v1390 = stablehlo.multiply %v1383, %v1389 : tensor<128x32x8x8xf32>
    %v1391 = stablehlo.reshape %v388 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1392 = stablehlo.multiply %v1391, %v1390 : tensor<128x32x8x8xf32>
    %v1393 = stablehlo.reduce(%v1392 init: %v1376) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1394 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1395 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1396 = stablehlo.multiply %v1394, %g5m : tensor<32xf32>
    %v1397 = stablehlo.multiply %v1395, %v1393 : tensor<32xf32>
    %v1398 = stablehlo.add %v1396, %v1397 : tensor<32xf32>
    %v1399 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1400 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1401 = stablehlo.multiply %v1399, %g5v : tensor<32xf32>
    %v1402 = stablehlo.multiply %v1393, %v1393 : tensor<32xf32>
    %v1403 = stablehlo.multiply %v1400, %v1402 : tensor<32xf32>
    %v1404 = stablehlo.add %v1401, %v1403 : tensor<32xf32>
    %v1405 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1406 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1407 = stablehlo.divide %v1398, %v1405 : tensor<32xf32>
    %v1408 = stablehlo.divide %v1404, %v1406 : tensor<32xf32>
    %v1409 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1410 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1411 = stablehlo.sqrt %v1408 : tensor<32xf32>
    %v1412 = stablehlo.add %v1411, %v1410 : tensor<32xf32>
    %v1413 = stablehlo.divide %v1407, %v1412 : tensor<32xf32>
    %v1414 = stablehlo.multiply %v1409, %v1413 : tensor<32xf32>
    %v1415 = stablehlo.subtract %g5, %v1414 : tensor<32xf32>
    %v1416 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1417 = stablehlo.multiply %v1416, %v1409 : tensor<32xf32>
    %v1418 = stablehlo.multiply %v1417, %g5 : tensor<32xf32>
    %v1419 = stablehlo.subtract %v1415, %v1418 : tensor<32xf32>
    %v1420 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1421 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1422 = stablehlo.multiply %v1420, %g5m : tensor<32xf32>
    %v1423 = stablehlo.multiply %v1421, %v1393 : tensor<32xf32>
    %v1424 = stablehlo.add %v1422, %v1423 : tensor<32xf32>
    %v1425 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1426 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1427 = stablehlo.multiply %v1425, %g5v : tensor<32xf32>
    %v1428 = stablehlo.multiply %v1393, %v1393 : tensor<32xf32>
    %v1429 = stablehlo.multiply %v1426, %v1428 : tensor<32xf32>
    %v1430 = stablehlo.add %v1427, %v1429 : tensor<32xf32>
    %v1431 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1432 = stablehlo.reshape %v388 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1433 = stablehlo.reduce(%v1432 init: %v1431) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1434 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1435 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1436 = stablehlo.multiply %v1434, %bt5m : tensor<32xf32>
    %v1437 = stablehlo.multiply %v1435, %v1433 : tensor<32xf32>
    %v1438 = stablehlo.add %v1436, %v1437 : tensor<32xf32>
    %v1439 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1440 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1441 = stablehlo.multiply %v1439, %bt5v : tensor<32xf32>
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
    %v1455 = stablehlo.subtract %bt5, %v1454 : tensor<32xf32>
    %v1456 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1457 = stablehlo.multiply %v1456, %v1449 : tensor<32xf32>
    %v1458 = stablehlo.multiply %v1457, %bt5 : tensor<32xf32>
    %v1459 = stablehlo.subtract %v1455, %v1458 : tensor<32xf32>
    %v1460 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1461 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1462 = stablehlo.multiply %v1460, %bt5m : tensor<32xf32>
    %v1463 = stablehlo.multiply %v1461, %v1433 : tensor<32xf32>
    %v1464 = stablehlo.add %v1462, %v1463 : tensor<32xf32>
    %v1465 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1466 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1467 = stablehlo.multiply %v1465, %bt5v : tensor<32xf32>
    %v1468 = stablehlo.multiply %v1433, %v1433 : tensor<32xf32>
    %v1469 = stablehlo.multiply %v1466, %v1468 : tensor<32xf32>
    %v1470 = stablehlo.add %v1467, %v1469 : tensor<32xf32>
    %v1471 = stablehlo.reshape %v142 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1472 = stablehlo.reshape %v380 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1473 = stablehlo.transpose %v1471, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v1474 = stablehlo.transpose %v1472, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v1475 = stablehlo.convolution(%v1473, %v1474)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<32x32x3x3xf32>
    %v1476 = stablehlo.transpose %v1475, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v1477 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1478 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1479 = stablehlo.multiply %v1477, %W6m : tensor<32x32x3x3xf32>
    %v1480 = stablehlo.multiply %v1478, %v1476 : tensor<32x32x3x3xf32>
    %v1481 = stablehlo.add %v1479, %v1480 : tensor<32x32x3x3xf32>
    %v1482 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1483 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1484 = stablehlo.multiply %v1482, %W6v : tensor<32x32x3x3xf32>
    %v1485 = stablehlo.multiply %v1476, %v1476 : tensor<32x32x3x3xf32>
    %v1486 = stablehlo.multiply %v1483, %v1485 : tensor<32x32x3x3xf32>
    %v1487 = stablehlo.add %v1484, %v1486 : tensor<32x32x3x3xf32>
    %v1488 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1489 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1490 = stablehlo.divide %v1481, %v1488 : tensor<32x32x3x3xf32>
    %v1491 = stablehlo.divide %v1487, %v1489 : tensor<32x32x3x3xf32>
    %v1492 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1493 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1494 = stablehlo.sqrt %v1491 : tensor<32x32x3x3xf32>
    %v1495 = stablehlo.add %v1494, %v1493 : tensor<32x32x3x3xf32>
    %v1496 = stablehlo.divide %v1490, %v1495 : tensor<32x32x3x3xf32>
    %v1497 = stablehlo.multiply %v1492, %v1496 : tensor<32x32x3x3xf32>
    %v1498 = stablehlo.subtract %W6, %v1497 : tensor<32x32x3x3xf32>
    %v1499 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1500 = stablehlo.multiply %v1499, %v1492 : tensor<32x32x3x3xf32>
    %v1501 = stablehlo.multiply %v1500, %W6 : tensor<32x32x3x3xf32>
    %v1502 = stablehlo.subtract %v1498, %v1501 : tensor<32x32x3x3xf32>
    %v1503 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1504 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1505 = stablehlo.multiply %v1503, %W6m : tensor<32x32x3x3xf32>
    %v1506 = stablehlo.multiply %v1504, %v1476 : tensor<32x32x3x3xf32>
    %v1507 = stablehlo.add %v1505, %v1506 : tensor<32x32x3x3xf32>
    %v1508 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1509 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1510 = stablehlo.multiply %v1508, %W6v : tensor<32x32x3x3xf32>
    %v1511 = stablehlo.multiply %v1476, %v1476 : tensor<32x32x3x3xf32>
    %v1512 = stablehlo.multiply %v1509, %v1511 : tensor<32x32x3x3xf32>
    %v1513 = stablehlo.add %v1510, %v1512 : tensor<32x32x3x3xf32>
    %v1514 = stablehlo.reshape %v380 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1515 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1516 = stablehlo.reduce(%v1514 init: %v1515) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1517 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1518 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1519 = stablehlo.multiply %v1517, %cb6m : tensor<32xf32>
    %v1520 = stablehlo.multiply %v1518, %v1516 : tensor<32xf32>
    %v1521 = stablehlo.add %v1519, %v1520 : tensor<32xf32>
    %v1522 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1523 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1524 = stablehlo.multiply %v1522, %cb6v : tensor<32xf32>
    %v1525 = stablehlo.multiply %v1516, %v1516 : tensor<32xf32>
    %v1526 = stablehlo.multiply %v1523, %v1525 : tensor<32xf32>
    %v1527 = stablehlo.add %v1524, %v1526 : tensor<32xf32>
    %v1528 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1529 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1530 = stablehlo.divide %v1521, %v1528 : tensor<32xf32>
    %v1531 = stablehlo.divide %v1527, %v1529 : tensor<32xf32>
    %v1532 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1533 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1534 = stablehlo.sqrt %v1531 : tensor<32xf32>
    %v1535 = stablehlo.add %v1534, %v1533 : tensor<32xf32>
    %v1536 = stablehlo.divide %v1530, %v1535 : tensor<32xf32>
    %v1537 = stablehlo.multiply %v1532, %v1536 : tensor<32xf32>
    %v1538 = stablehlo.subtract %cb6, %v1537 : tensor<32xf32>
    %v1539 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1540 = stablehlo.multiply %v1539, %v1532 : tensor<32xf32>
    %v1541 = stablehlo.multiply %v1540, %cb6 : tensor<32xf32>
    %v1542 = stablehlo.subtract %v1538, %v1541 : tensor<32xf32>
    %v1543 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1544 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1545 = stablehlo.multiply %v1543, %cb6m : tensor<32xf32>
    %v1546 = stablehlo.multiply %v1544, %v1516 : tensor<32xf32>
    %v1547 = stablehlo.add %v1545, %v1546 : tensor<32xf32>
    %v1548 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1549 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1550 = stablehlo.multiply %v1548, %cb6v : tensor<32xf32>
    %v1551 = stablehlo.multiply %v1516, %v1516 : tensor<32xf32>
    %v1552 = stablehlo.multiply %v1549, %v1551 : tensor<32xf32>
    %v1553 = stablehlo.add %v1550, %v1552 : tensor<32xf32>
    %v1554 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1555 = stablehlo.reshape %v147 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1556 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v1557 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v1558 = stablehlo.reduce(%v1555 init: %v1554) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1559 = stablehlo.broadcast_in_dim %v1558, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v1560 = stablehlo.divide %v1559, %v1556 : tensor<128x32x8x8xf32>
    %v1561 = stablehlo.subtract %v1555, %v1560 : tensor<128x32x8x8xf32>
    %v1562 = stablehlo.multiply %v1561, %v1561 : tensor<128x32x8x8xf32>
    %v1563 = stablehlo.reduce(%v1562 init: %v1554) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1564 = stablehlo.broadcast_in_dim %v1563, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v1565 = stablehlo.divide %v1564, %v1556 : tensor<128x32x8x8xf32>
    %v1566 = stablehlo.add %v1565, %v1557 : tensor<128x32x8x8xf32>
    %v1567 = stablehlo.rsqrt %v1566 : tensor<128x32x8x8xf32>
    %v1568 = stablehlo.multiply %v1561, %v1567 : tensor<128x32x8x8xf32>
    %v1569 = stablehlo.reshape %v350 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1570 = stablehlo.multiply %v1569, %v1568 : tensor<128x32x8x8xf32>
    %v1571 = stablehlo.reduce(%v1570 init: %v1554) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1572 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1573 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1574 = stablehlo.multiply %v1572, %g6m : tensor<32xf32>
    %v1575 = stablehlo.multiply %v1573, %v1571 : tensor<32xf32>
    %v1576 = stablehlo.add %v1574, %v1575 : tensor<32xf32>
    %v1577 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1578 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1579 = stablehlo.multiply %v1577, %g6v : tensor<32xf32>
    %v1580 = stablehlo.multiply %v1571, %v1571 : tensor<32xf32>
    %v1581 = stablehlo.multiply %v1578, %v1580 : tensor<32xf32>
    %v1582 = stablehlo.add %v1579, %v1581 : tensor<32xf32>
    %v1583 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1584 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1585 = stablehlo.divide %v1576, %v1583 : tensor<32xf32>
    %v1586 = stablehlo.divide %v1582, %v1584 : tensor<32xf32>
    %v1587 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1588 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1589 = stablehlo.sqrt %v1586 : tensor<32xf32>
    %v1590 = stablehlo.add %v1589, %v1588 : tensor<32xf32>
    %v1591 = stablehlo.divide %v1585, %v1590 : tensor<32xf32>
    %v1592 = stablehlo.multiply %v1587, %v1591 : tensor<32xf32>
    %v1593 = stablehlo.subtract %g6, %v1592 : tensor<32xf32>
    %v1594 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1595 = stablehlo.multiply %v1594, %v1587 : tensor<32xf32>
    %v1596 = stablehlo.multiply %v1595, %g6 : tensor<32xf32>
    %v1597 = stablehlo.subtract %v1593, %v1596 : tensor<32xf32>
    %v1598 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1599 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1600 = stablehlo.multiply %v1598, %g6m : tensor<32xf32>
    %v1601 = stablehlo.multiply %v1599, %v1571 : tensor<32xf32>
    %v1602 = stablehlo.add %v1600, %v1601 : tensor<32xf32>
    %v1603 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1604 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1605 = stablehlo.multiply %v1603, %g6v : tensor<32xf32>
    %v1606 = stablehlo.multiply %v1571, %v1571 : tensor<32xf32>
    %v1607 = stablehlo.multiply %v1604, %v1606 : tensor<32xf32>
    %v1608 = stablehlo.add %v1605, %v1607 : tensor<32xf32>
    %v1609 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1610 = stablehlo.reshape %v350 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1611 = stablehlo.reduce(%v1610 init: %v1609) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1612 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1613 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1614 = stablehlo.multiply %v1612, %bt6m : tensor<32xf32>
    %v1615 = stablehlo.multiply %v1613, %v1611 : tensor<32xf32>
    %v1616 = stablehlo.add %v1614, %v1615 : tensor<32xf32>
    %v1617 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1618 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1619 = stablehlo.multiply %v1617, %bt6v : tensor<32xf32>
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
    %v1633 = stablehlo.subtract %bt6, %v1632 : tensor<32xf32>
    %v1634 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1635 = stablehlo.multiply %v1634, %v1627 : tensor<32xf32>
    %v1636 = stablehlo.multiply %v1635, %bt6 : tensor<32xf32>
    %v1637 = stablehlo.subtract %v1633, %v1636 : tensor<32xf32>
    %v1638 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1639 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1640 = stablehlo.multiply %v1638, %bt6m : tensor<32xf32>
    %v1641 = stablehlo.multiply %v1639, %v1611 : tensor<32xf32>
    %v1642 = stablehlo.add %v1640, %v1641 : tensor<32xf32>
    %v1643 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1644 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1645 = stablehlo.multiply %v1643, %bt6v : tensor<32xf32>
    %v1646 = stablehlo.multiply %v1611, %v1611 : tensor<32xf32>
    %v1647 = stablehlo.multiply %v1644, %v1646 : tensor<32xf32>
    %v1648 = stablehlo.add %v1645, %v1647 : tensor<32xf32>
    %v1649 = stablehlo.reshape %v173 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1650 = stablehlo.reshape %v337 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1651 = stablehlo.transpose %v1649, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v1652 = stablehlo.transpose %v1650, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v1653 = stablehlo.convolution(%v1651, %v1652)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v1654 = stablehlo.transpose %v1653, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v1655 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1656 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1657 = stablehlo.multiply %v1655, %W7m : tensor<32x32x3x3xf32>
    %v1658 = stablehlo.multiply %v1656, %v1654 : tensor<32x32x3x3xf32>
    %v1659 = stablehlo.add %v1657, %v1658 : tensor<32x32x3x3xf32>
    %v1660 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1661 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1662 = stablehlo.multiply %v1660, %W7v : tensor<32x32x3x3xf32>
    %v1663 = stablehlo.multiply %v1654, %v1654 : tensor<32x32x3x3xf32>
    %v1664 = stablehlo.multiply %v1661, %v1663 : tensor<32x32x3x3xf32>
    %v1665 = stablehlo.add %v1662, %v1664 : tensor<32x32x3x3xf32>
    %v1666 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1667 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1668 = stablehlo.divide %v1659, %v1666 : tensor<32x32x3x3xf32>
    %v1669 = stablehlo.divide %v1665, %v1667 : tensor<32x32x3x3xf32>
    %v1670 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1671 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1672 = stablehlo.sqrt %v1669 : tensor<32x32x3x3xf32>
    %v1673 = stablehlo.add %v1672, %v1671 : tensor<32x32x3x3xf32>
    %v1674 = stablehlo.divide %v1668, %v1673 : tensor<32x32x3x3xf32>
    %v1675 = stablehlo.multiply %v1670, %v1674 : tensor<32x32x3x3xf32>
    %v1676 = stablehlo.subtract %W7, %v1675 : tensor<32x32x3x3xf32>
    %v1677 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1678 = stablehlo.multiply %v1677, %v1670 : tensor<32x32x3x3xf32>
    %v1679 = stablehlo.multiply %v1678, %W7 : tensor<32x32x3x3xf32>
    %v1680 = stablehlo.subtract %v1676, %v1679 : tensor<32x32x3x3xf32>
    %v1681 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1682 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1683 = stablehlo.multiply %v1681, %W7m : tensor<32x32x3x3xf32>
    %v1684 = stablehlo.multiply %v1682, %v1654 : tensor<32x32x3x3xf32>
    %v1685 = stablehlo.add %v1683, %v1684 : tensor<32x32x3x3xf32>
    %v1686 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1687 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1688 = stablehlo.multiply %v1686, %W7v : tensor<32x32x3x3xf32>
    %v1689 = stablehlo.multiply %v1654, %v1654 : tensor<32x32x3x3xf32>
    %v1690 = stablehlo.multiply %v1687, %v1689 : tensor<32x32x3x3xf32>
    %v1691 = stablehlo.add %v1688, %v1690 : tensor<32x32x3x3xf32>
    %v1692 = stablehlo.reshape %v337 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1693 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1694 = stablehlo.reduce(%v1692 init: %v1693) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1695 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1696 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1697 = stablehlo.multiply %v1695, %cb7m : tensor<32xf32>
    %v1698 = stablehlo.multiply %v1696, %v1694 : tensor<32xf32>
    %v1699 = stablehlo.add %v1697, %v1698 : tensor<32xf32>
    %v1700 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1701 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1702 = stablehlo.multiply %v1700, %cb7v : tensor<32xf32>
    %v1703 = stablehlo.multiply %v1694, %v1694 : tensor<32xf32>
    %v1704 = stablehlo.multiply %v1701, %v1703 : tensor<32xf32>
    %v1705 = stablehlo.add %v1702, %v1704 : tensor<32xf32>
    %v1706 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1707 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1708 = stablehlo.divide %v1699, %v1706 : tensor<32xf32>
    %v1709 = stablehlo.divide %v1705, %v1707 : tensor<32xf32>
    %v1710 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1711 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1712 = stablehlo.sqrt %v1709 : tensor<32xf32>
    %v1713 = stablehlo.add %v1712, %v1711 : tensor<32xf32>
    %v1714 = stablehlo.divide %v1708, %v1713 : tensor<32xf32>
    %v1715 = stablehlo.multiply %v1710, %v1714 : tensor<32xf32>
    %v1716 = stablehlo.subtract %cb7, %v1715 : tensor<32xf32>
    %v1717 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1718 = stablehlo.multiply %v1717, %v1710 : tensor<32xf32>
    %v1719 = stablehlo.multiply %v1718, %cb7 : tensor<32xf32>
    %v1720 = stablehlo.subtract %v1716, %v1719 : tensor<32xf32>
    %v1721 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1722 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1723 = stablehlo.multiply %v1721, %cb7m : tensor<32xf32>
    %v1724 = stablehlo.multiply %v1722, %v1694 : tensor<32xf32>
    %v1725 = stablehlo.add %v1723, %v1724 : tensor<32xf32>
    %v1726 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1727 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1728 = stablehlo.multiply %v1726, %cb7v : tensor<32xf32>
    %v1729 = stablehlo.multiply %v1694, %v1694 : tensor<32xf32>
    %v1730 = stablehlo.multiply %v1727, %v1729 : tensor<32xf32>
    %v1731 = stablehlo.add %v1728, %v1730 : tensor<32xf32>
    %v1732 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1733 = stablehlo.reshape %v178 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1734 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v1735 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v1736 = stablehlo.reduce(%v1733 init: %v1732) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1737 = stablehlo.broadcast_in_dim %v1736, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v1738 = stablehlo.divide %v1737, %v1734 : tensor<128x32x4x4xf32>
    %v1739 = stablehlo.subtract %v1733, %v1738 : tensor<128x32x4x4xf32>
    %v1740 = stablehlo.multiply %v1739, %v1739 : tensor<128x32x4x4xf32>
    %v1741 = stablehlo.reduce(%v1740 init: %v1732) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1742 = stablehlo.broadcast_in_dim %v1741, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v1743 = stablehlo.divide %v1742, %v1734 : tensor<128x32x4x4xf32>
    %v1744 = stablehlo.add %v1743, %v1735 : tensor<128x32x4x4xf32>
    %v1745 = stablehlo.rsqrt %v1744 : tensor<128x32x4x4xf32>
    %v1746 = stablehlo.multiply %v1739, %v1745 : tensor<128x32x4x4xf32>
    %v1747 = stablehlo.reshape %v307 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1748 = stablehlo.multiply %v1747, %v1746 : tensor<128x32x4x4xf32>
    %v1749 = stablehlo.reduce(%v1748 init: %v1732) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1750 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1751 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1752 = stablehlo.multiply %v1750, %g7m : tensor<32xf32>
    %v1753 = stablehlo.multiply %v1751, %v1749 : tensor<32xf32>
    %v1754 = stablehlo.add %v1752, %v1753 : tensor<32xf32>
    %v1755 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1756 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1757 = stablehlo.multiply %v1755, %g7v : tensor<32xf32>
    %v1758 = stablehlo.multiply %v1749, %v1749 : tensor<32xf32>
    %v1759 = stablehlo.multiply %v1756, %v1758 : tensor<32xf32>
    %v1760 = stablehlo.add %v1757, %v1759 : tensor<32xf32>
    %v1761 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1762 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1763 = stablehlo.divide %v1754, %v1761 : tensor<32xf32>
    %v1764 = stablehlo.divide %v1760, %v1762 : tensor<32xf32>
    %v1765 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1766 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1767 = stablehlo.sqrt %v1764 : tensor<32xf32>
    %v1768 = stablehlo.add %v1767, %v1766 : tensor<32xf32>
    %v1769 = stablehlo.divide %v1763, %v1768 : tensor<32xf32>
    %v1770 = stablehlo.multiply %v1765, %v1769 : tensor<32xf32>
    %v1771 = stablehlo.subtract %g7, %v1770 : tensor<32xf32>
    %v1772 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1773 = stablehlo.multiply %v1772, %v1765 : tensor<32xf32>
    %v1774 = stablehlo.multiply %v1773, %g7 : tensor<32xf32>
    %v1775 = stablehlo.subtract %v1771, %v1774 : tensor<32xf32>
    %v1776 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1777 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1778 = stablehlo.multiply %v1776, %g7m : tensor<32xf32>
    %v1779 = stablehlo.multiply %v1777, %v1749 : tensor<32xf32>
    %v1780 = stablehlo.add %v1778, %v1779 : tensor<32xf32>
    %v1781 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1782 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1783 = stablehlo.multiply %v1781, %g7v : tensor<32xf32>
    %v1784 = stablehlo.multiply %v1749, %v1749 : tensor<32xf32>
    %v1785 = stablehlo.multiply %v1782, %v1784 : tensor<32xf32>
    %v1786 = stablehlo.add %v1783, %v1785 : tensor<32xf32>
    %v1787 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1788 = stablehlo.reshape %v307 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1789 = stablehlo.reduce(%v1788 init: %v1787) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1790 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1791 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1792 = stablehlo.multiply %v1790, %bt7m : tensor<32xf32>
    %v1793 = stablehlo.multiply %v1791, %v1789 : tensor<32xf32>
    %v1794 = stablehlo.add %v1792, %v1793 : tensor<32xf32>
    %v1795 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1796 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1797 = stablehlo.multiply %v1795, %bt7v : tensor<32xf32>
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
    %v1811 = stablehlo.subtract %bt7, %v1810 : tensor<32xf32>
    %v1812 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1813 = stablehlo.multiply %v1812, %v1805 : tensor<32xf32>
    %v1814 = stablehlo.multiply %v1813, %bt7 : tensor<32xf32>
    %v1815 = stablehlo.subtract %v1811, %v1814 : tensor<32xf32>
    %v1816 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1817 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1818 = stablehlo.multiply %v1816, %bt7m : tensor<32xf32>
    %v1819 = stablehlo.multiply %v1817, %v1789 : tensor<32xf32>
    %v1820 = stablehlo.add %v1818, %v1819 : tensor<32xf32>
    %v1821 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1822 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1823 = stablehlo.multiply %v1821, %bt7v : tensor<32xf32>
    %v1824 = stablehlo.multiply %v1789, %v1789 : tensor<32xf32>
    %v1825 = stablehlo.multiply %v1822, %v1824 : tensor<32xf32>
    %v1826 = stablehlo.add %v1823, %v1825 : tensor<32xf32>
    %v1827 = stablehlo.reshape %v200 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1828 = stablehlo.reshape %v299 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1829 = stablehlo.transpose %v1827, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v1830 = stablehlo.transpose %v1828, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v1831 = stablehlo.convolution(%v1829, %v1830)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v1832 = stablehlo.transpose %v1831, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v1833 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1834 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1835 = stablehlo.multiply %v1833, %W8m : tensor<32x32x3x3xf32>
    %v1836 = stablehlo.multiply %v1834, %v1832 : tensor<32x32x3x3xf32>
    %v1837 = stablehlo.add %v1835, %v1836 : tensor<32x32x3x3xf32>
    %v1838 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1839 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1840 = stablehlo.multiply %v1838, %W8v : tensor<32x32x3x3xf32>
    %v1841 = stablehlo.multiply %v1832, %v1832 : tensor<32x32x3x3xf32>
    %v1842 = stablehlo.multiply %v1839, %v1841 : tensor<32x32x3x3xf32>
    %v1843 = stablehlo.add %v1840, %v1842 : tensor<32x32x3x3xf32>
    %v1844 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1845 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1846 = stablehlo.divide %v1837, %v1844 : tensor<32x32x3x3xf32>
    %v1847 = stablehlo.divide %v1843, %v1845 : tensor<32x32x3x3xf32>
    %v1848 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1849 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1850 = stablehlo.sqrt %v1847 : tensor<32x32x3x3xf32>
    %v1851 = stablehlo.add %v1850, %v1849 : tensor<32x32x3x3xf32>
    %v1852 = stablehlo.divide %v1846, %v1851 : tensor<32x32x3x3xf32>
    %v1853 = stablehlo.multiply %v1848, %v1852 : tensor<32x32x3x3xf32>
    %v1854 = stablehlo.subtract %W8, %v1853 : tensor<32x32x3x3xf32>
    %v1855 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1856 = stablehlo.multiply %v1855, %v1848 : tensor<32x32x3x3xf32>
    %v1857 = stablehlo.multiply %v1856, %W8 : tensor<32x32x3x3xf32>
    %v1858 = stablehlo.subtract %v1854, %v1857 : tensor<32x32x3x3xf32>
    %v1859 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1860 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1861 = stablehlo.multiply %v1859, %W8m : tensor<32x32x3x3xf32>
    %v1862 = stablehlo.multiply %v1860, %v1832 : tensor<32x32x3x3xf32>
    %v1863 = stablehlo.add %v1861, %v1862 : tensor<32x32x3x3xf32>
    %v1864 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1865 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1866 = stablehlo.multiply %v1864, %W8v : tensor<32x32x3x3xf32>
    %v1867 = stablehlo.multiply %v1832, %v1832 : tensor<32x32x3x3xf32>
    %v1868 = stablehlo.multiply %v1865, %v1867 : tensor<32x32x3x3xf32>
    %v1869 = stablehlo.add %v1866, %v1868 : tensor<32x32x3x3xf32>
    %v1870 = stablehlo.reshape %v299 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1871 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1872 = stablehlo.reduce(%v1870 init: %v1871) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1873 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1874 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1875 = stablehlo.multiply %v1873, %cb8m : tensor<32xf32>
    %v1876 = stablehlo.multiply %v1874, %v1872 : tensor<32xf32>
    %v1877 = stablehlo.add %v1875, %v1876 : tensor<32xf32>
    %v1878 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1879 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1880 = stablehlo.multiply %v1878, %cb8v : tensor<32xf32>
    %v1881 = stablehlo.multiply %v1872, %v1872 : tensor<32xf32>
    %v1882 = stablehlo.multiply %v1879, %v1881 : tensor<32xf32>
    %v1883 = stablehlo.add %v1880, %v1882 : tensor<32xf32>
    %v1884 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1885 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1886 = stablehlo.divide %v1877, %v1884 : tensor<32xf32>
    %v1887 = stablehlo.divide %v1883, %v1885 : tensor<32xf32>
    %v1888 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1889 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1890 = stablehlo.sqrt %v1887 : tensor<32xf32>
    %v1891 = stablehlo.add %v1890, %v1889 : tensor<32xf32>
    %v1892 = stablehlo.divide %v1886, %v1891 : tensor<32xf32>
    %v1893 = stablehlo.multiply %v1888, %v1892 : tensor<32xf32>
    %v1894 = stablehlo.subtract %cb8, %v1893 : tensor<32xf32>
    %v1895 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1896 = stablehlo.multiply %v1895, %v1888 : tensor<32xf32>
    %v1897 = stablehlo.multiply %v1896, %cb8 : tensor<32xf32>
    %v1898 = stablehlo.subtract %v1894, %v1897 : tensor<32xf32>
    %v1899 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1900 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1901 = stablehlo.multiply %v1899, %cb8m : tensor<32xf32>
    %v1902 = stablehlo.multiply %v1900, %v1872 : tensor<32xf32>
    %v1903 = stablehlo.add %v1901, %v1902 : tensor<32xf32>
    %v1904 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1905 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1906 = stablehlo.multiply %v1904, %cb8v : tensor<32xf32>
    %v1907 = stablehlo.multiply %v1872, %v1872 : tensor<32xf32>
    %v1908 = stablehlo.multiply %v1905, %v1907 : tensor<32xf32>
    %v1909 = stablehlo.add %v1906, %v1908 : tensor<32xf32>
    %v1910 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1911 = stablehlo.reshape %v205 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1912 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v1913 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v1914 = stablehlo.reduce(%v1911 init: %v1910) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1915 = stablehlo.broadcast_in_dim %v1914, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v1916 = stablehlo.divide %v1915, %v1912 : tensor<128x32x4x4xf32>
    %v1917 = stablehlo.subtract %v1911, %v1916 : tensor<128x32x4x4xf32>
    %v1918 = stablehlo.multiply %v1917, %v1917 : tensor<128x32x4x4xf32>
    %v1919 = stablehlo.reduce(%v1918 init: %v1910) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1920 = stablehlo.broadcast_in_dim %v1919, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v1921 = stablehlo.divide %v1920, %v1912 : tensor<128x32x4x4xf32>
    %v1922 = stablehlo.add %v1921, %v1913 : tensor<128x32x4x4xf32>
    %v1923 = stablehlo.rsqrt %v1922 : tensor<128x32x4x4xf32>
    %v1924 = stablehlo.multiply %v1917, %v1923 : tensor<128x32x4x4xf32>
    %v1925 = stablehlo.reshape %v269 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1926 = stablehlo.multiply %v1925, %v1924 : tensor<128x32x4x4xf32>
    %v1927 = stablehlo.reduce(%v1926 init: %v1910) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1928 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1929 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1930 = stablehlo.multiply %v1928, %g8m : tensor<32xf32>
    %v1931 = stablehlo.multiply %v1929, %v1927 : tensor<32xf32>
    %v1932 = stablehlo.add %v1930, %v1931 : tensor<32xf32>
    %v1933 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1934 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1935 = stablehlo.multiply %v1933, %g8v : tensor<32xf32>
    %v1936 = stablehlo.multiply %v1927, %v1927 : tensor<32xf32>
    %v1937 = stablehlo.multiply %v1934, %v1936 : tensor<32xf32>
    %v1938 = stablehlo.add %v1935, %v1937 : tensor<32xf32>
    %v1939 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1940 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1941 = stablehlo.divide %v1932, %v1939 : tensor<32xf32>
    %v1942 = stablehlo.divide %v1938, %v1940 : tensor<32xf32>
    %v1943 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1944 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1945 = stablehlo.sqrt %v1942 : tensor<32xf32>
    %v1946 = stablehlo.add %v1945, %v1944 : tensor<32xf32>
    %v1947 = stablehlo.divide %v1941, %v1946 : tensor<32xf32>
    %v1948 = stablehlo.multiply %v1943, %v1947 : tensor<32xf32>
    %v1949 = stablehlo.subtract %g8, %v1948 : tensor<32xf32>
    %v1950 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1951 = stablehlo.multiply %v1950, %v1943 : tensor<32xf32>
    %v1952 = stablehlo.multiply %v1951, %g8 : tensor<32xf32>
    %v1953 = stablehlo.subtract %v1949, %v1952 : tensor<32xf32>
    %v1954 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1955 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1956 = stablehlo.multiply %v1954, %g8m : tensor<32xf32>
    %v1957 = stablehlo.multiply %v1955, %v1927 : tensor<32xf32>
    %v1958 = stablehlo.add %v1956, %v1957 : tensor<32xf32>
    %v1959 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1960 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1961 = stablehlo.multiply %v1959, %g8v : tensor<32xf32>
    %v1962 = stablehlo.multiply %v1927, %v1927 : tensor<32xf32>
    %v1963 = stablehlo.multiply %v1960, %v1962 : tensor<32xf32>
    %v1964 = stablehlo.add %v1961, %v1963 : tensor<32xf32>
    %v1965 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1966 = stablehlo.reshape %v269 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1967 = stablehlo.reduce(%v1966 init: %v1965) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1968 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1969 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1970 = stablehlo.multiply %v1968, %bt8m : tensor<32xf32>
    %v1971 = stablehlo.multiply %v1969, %v1967 : tensor<32xf32>
    %v1972 = stablehlo.add %v1970, %v1971 : tensor<32xf32>
    %v1973 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1974 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1975 = stablehlo.multiply %v1973, %bt8v : tensor<32xf32>
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
    %v1989 = stablehlo.subtract %bt8, %v1988 : tensor<32xf32>
    %v1990 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1991 = stablehlo.multiply %v1990, %v1983 : tensor<32xf32>
    %v1992 = stablehlo.multiply %v1991, %bt8 : tensor<32xf32>
    %v1993 = stablehlo.subtract %v1989, %v1992 : tensor<32xf32>
    %v1994 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1995 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1996 = stablehlo.multiply %v1994, %bt8m : tensor<32xf32>
    %v1997 = stablehlo.multiply %v1995, %v1967 : tensor<32xf32>
    %v1998 = stablehlo.add %v1996, %v1997 : tensor<32xf32>
    %v1999 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2000 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v2001 = stablehlo.multiply %v1999, %bt8v : tensor<32xf32>
    %v2002 = stablehlo.multiply %v1967, %v1967 : tensor<32xf32>
    %v2003 = stablehlo.multiply %v2000, %v2002 : tensor<32xf32>
    %v2004 = stablehlo.add %v2001, %v2003 : tensor<32xf32>
    %v2005 = stablehlo.dot_general %v231, %v260, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %v2006 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v2007 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v2008 = stablehlo.multiply %v2006, %W9m : tensor<128x64xf32>
    %v2009 = stablehlo.multiply %v2007, %v2005 : tensor<128x64xf32>
    %v2010 = stablehlo.add %v2008, %v2009 : tensor<128x64xf32>
    %v2011 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v2012 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v2013 = stablehlo.multiply %v2011, %W9v : tensor<128x64xf32>
    %v2014 = stablehlo.multiply %v2005, %v2005 : tensor<128x64xf32>
    %v2015 = stablehlo.multiply %v2012, %v2014 : tensor<128x64xf32>
    %v2016 = stablehlo.add %v2013, %v2015 : tensor<128x64xf32>
    %v2017 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v2018 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v2019 = stablehlo.divide %v2010, %v2017 : tensor<128x64xf32>
    %v2020 = stablehlo.divide %v2016, %v2018 : tensor<128x64xf32>
    %v2021 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v2022 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v2023 = stablehlo.sqrt %v2020 : tensor<128x64xf32>
    %v2024 = stablehlo.add %v2023, %v2022 : tensor<128x64xf32>
    %v2025 = stablehlo.divide %v2019, %v2024 : tensor<128x64xf32>
    %v2026 = stablehlo.multiply %v2021, %v2025 : tensor<128x64xf32>
    %v2027 = stablehlo.subtract %W9, %v2026 : tensor<128x64xf32>
    %v2028 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v2029 = stablehlo.multiply %v2028, %v2021 : tensor<128x64xf32>
    %v2030 = stablehlo.multiply %v2029, %W9 : tensor<128x64xf32>
    %v2031 = stablehlo.subtract %v2027, %v2030 : tensor<128x64xf32>
    %v2032 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v2033 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v2034 = stablehlo.multiply %v2032, %W9m : tensor<128x64xf32>
    %v2035 = stablehlo.multiply %v2033, %v2005 : tensor<128x64xf32>
    %v2036 = stablehlo.add %v2034, %v2035 : tensor<128x64xf32>
    %v2037 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v2038 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64xf32>
    %v2039 = stablehlo.multiply %v2037, %W9v : tensor<128x64xf32>
    %v2040 = stablehlo.multiply %v2005, %v2005 : tensor<128x64xf32>
    %v2041 = stablehlo.multiply %v2038, %v2040 : tensor<128x64xf32>
    %v2042 = stablehlo.add %v2039, %v2041 : tensor<128x64xf32>
    %v2043 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2044 = stablehlo.reduce(%v260 init: %v2043) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v2045 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2046 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2047 = stablehlo.multiply %v2045, %b9m : tensor<64xf32>
    %v2048 = stablehlo.multiply %v2046, %v2044 : tensor<64xf32>
    %v2049 = stablehlo.add %v2047, %v2048 : tensor<64xf32>
    %v2050 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2051 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2052 = stablehlo.multiply %v2050, %b9v : tensor<64xf32>
    %v2053 = stablehlo.multiply %v2044, %v2044 : tensor<64xf32>
    %v2054 = stablehlo.multiply %v2051, %v2053 : tensor<64xf32>
    %v2055 = stablehlo.add %v2052, %v2054 : tensor<64xf32>
    %v2056 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2057 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2058 = stablehlo.divide %v2049, %v2056 : tensor<64xf32>
    %v2059 = stablehlo.divide %v2055, %v2057 : tensor<64xf32>
    %v2060 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2061 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2062 = stablehlo.sqrt %v2059 : tensor<64xf32>
    %v2063 = stablehlo.add %v2062, %v2061 : tensor<64xf32>
    %v2064 = stablehlo.divide %v2058, %v2063 : tensor<64xf32>
    %v2065 = stablehlo.multiply %v2060, %v2064 : tensor<64xf32>
    %v2066 = stablehlo.subtract %b9, %v2065 : tensor<64xf32>
    %v2067 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2068 = stablehlo.multiply %v2067, %v2060 : tensor<64xf32>
    %v2069 = stablehlo.multiply %v2068, %b9 : tensor<64xf32>
    %v2070 = stablehlo.subtract %v2066, %v2069 : tensor<64xf32>
    %v2071 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2072 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2073 = stablehlo.multiply %v2071, %b9m : tensor<64xf32>
    %v2074 = stablehlo.multiply %v2072, %v2044 : tensor<64xf32>
    %v2075 = stablehlo.add %v2073, %v2074 : tensor<64xf32>
    %v2076 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2077 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2078 = stablehlo.multiply %v2076, %b9v : tensor<64xf32>
    %v2079 = stablehlo.multiply %v2044, %v2044 : tensor<64xf32>
    %v2080 = stablehlo.multiply %v2077, %v2079 : tensor<64xf32>
    %v2081 = stablehlo.add %v2078, %v2080 : tensor<64xf32>
    %v2082 = stablehlo.dot_general %v236, %v256, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<64x64xf32>
    %v2083 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v2084 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v2085 = stablehlo.multiply %v2083, %Wam : tensor<64x64xf32>
    %v2086 = stablehlo.multiply %v2084, %v2082 : tensor<64x64xf32>
    %v2087 = stablehlo.add %v2085, %v2086 : tensor<64x64xf32>
    %v2088 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v2089 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v2090 = stablehlo.multiply %v2088, %Wav : tensor<64x64xf32>
    %v2091 = stablehlo.multiply %v2082, %v2082 : tensor<64x64xf32>
    %v2092 = stablehlo.multiply %v2089, %v2091 : tensor<64x64xf32>
    %v2093 = stablehlo.add %v2090, %v2092 : tensor<64x64xf32>
    %v2094 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v2095 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v2096 = stablehlo.divide %v2087, %v2094 : tensor<64x64xf32>
    %v2097 = stablehlo.divide %v2093, %v2095 : tensor<64x64xf32>
    %v2098 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v2099 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v2100 = stablehlo.sqrt %v2097 : tensor<64x64xf32>
    %v2101 = stablehlo.add %v2100, %v2099 : tensor<64x64xf32>
    %v2102 = stablehlo.divide %v2096, %v2101 : tensor<64x64xf32>
    %v2103 = stablehlo.multiply %v2098, %v2102 : tensor<64x64xf32>
    %v2104 = stablehlo.subtract %Wa, %v2103 : tensor<64x64xf32>
    %v2105 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v2106 = stablehlo.multiply %v2105, %v2098 : tensor<64x64xf32>
    %v2107 = stablehlo.multiply %v2106, %Wa : tensor<64x64xf32>
    %v2108 = stablehlo.subtract %v2104, %v2107 : tensor<64x64xf32>
    %v2109 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v2110 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v2111 = stablehlo.multiply %v2109, %Wam : tensor<64x64xf32>
    %v2112 = stablehlo.multiply %v2110, %v2082 : tensor<64x64xf32>
    %v2113 = stablehlo.add %v2111, %v2112 : tensor<64x64xf32>
    %v2114 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v2115 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %v2116 = stablehlo.multiply %v2114, %Wav : tensor<64x64xf32>
    %v2117 = stablehlo.multiply %v2082, %v2082 : tensor<64x64xf32>
    %v2118 = stablehlo.multiply %v2115, %v2117 : tensor<64x64xf32>
    %v2119 = stablehlo.add %v2116, %v2118 : tensor<64x64xf32>
    %v2120 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2121 = stablehlo.reduce(%v256 init: %v2120) applies stablehlo.add across dimensions = [0] : (tensor<128x64xf32>, tensor<f32>) -> tensor<64xf32>
    %v2122 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2123 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2124 = stablehlo.multiply %v2122, %bam : tensor<64xf32>
    %v2125 = stablehlo.multiply %v2123, %v2121 : tensor<64xf32>
    %v2126 = stablehlo.add %v2124, %v2125 : tensor<64xf32>
    %v2127 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2128 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2129 = stablehlo.multiply %v2127, %bav : tensor<64xf32>
    %v2130 = stablehlo.multiply %v2121, %v2121 : tensor<64xf32>
    %v2131 = stablehlo.multiply %v2128, %v2130 : tensor<64xf32>
    %v2132 = stablehlo.add %v2129, %v2131 : tensor<64xf32>
    %v2133 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2134 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2135 = stablehlo.divide %v2126, %v2133 : tensor<64xf32>
    %v2136 = stablehlo.divide %v2132, %v2134 : tensor<64xf32>
    %v2137 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2138 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2139 = stablehlo.sqrt %v2136 : tensor<64xf32>
    %v2140 = stablehlo.add %v2139, %v2138 : tensor<64xf32>
    %v2141 = stablehlo.divide %v2135, %v2140 : tensor<64xf32>
    %v2142 = stablehlo.multiply %v2137, %v2141 : tensor<64xf32>
    %v2143 = stablehlo.subtract %ba, %v2142 : tensor<64xf32>
    %v2144 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2145 = stablehlo.multiply %v2144, %v2137 : tensor<64xf32>
    %v2146 = stablehlo.multiply %v2145, %ba : tensor<64xf32>
    %v2147 = stablehlo.subtract %v2143, %v2146 : tensor<64xf32>
    %v2148 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2149 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2150 = stablehlo.multiply %v2148, %bam : tensor<64xf32>
    %v2151 = stablehlo.multiply %v2149, %v2121 : tensor<64xf32>
    %v2152 = stablehlo.add %v2150, %v2151 : tensor<64xf32>
    %v2153 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2154 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v2155 = stablehlo.multiply %v2153, %bav : tensor<64xf32>
    %v2156 = stablehlo.multiply %v2121, %v2121 : tensor<64xf32>
    %v2157 = stablehlo.multiply %v2154, %v2156 : tensor<64xf32>
    %v2158 = stablehlo.add %v2155, %v2157 : tensor<64xf32>
    %v2159 = stablehlo.dot_general %v241, %v252, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x64xf32>, tensor<128x10xf32>) -> tensor<64x10xf32>
    %v2160 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v2161 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v2162 = stablehlo.multiply %v2160, %Wbm : tensor<64x10xf32>
    %v2163 = stablehlo.multiply %v2161, %v2159 : tensor<64x10xf32>
    %v2164 = stablehlo.add %v2162, %v2163 : tensor<64x10xf32>
    %v2165 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v2166 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v2167 = stablehlo.multiply %v2165, %Wbv : tensor<64x10xf32>
    %v2168 = stablehlo.multiply %v2159, %v2159 : tensor<64x10xf32>
    %v2169 = stablehlo.multiply %v2166, %v2168 : tensor<64x10xf32>
    %v2170 = stablehlo.add %v2167, %v2169 : tensor<64x10xf32>
    %v2171 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v2172 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v2173 = stablehlo.divide %v2164, %v2171 : tensor<64x10xf32>
    %v2174 = stablehlo.divide %v2170, %v2172 : tensor<64x10xf32>
    %v2175 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v2176 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v2177 = stablehlo.sqrt %v2174 : tensor<64x10xf32>
    %v2178 = stablehlo.add %v2177, %v2176 : tensor<64x10xf32>
    %v2179 = stablehlo.divide %v2173, %v2178 : tensor<64x10xf32>
    %v2180 = stablehlo.multiply %v2175, %v2179 : tensor<64x10xf32>
    %v2181 = stablehlo.subtract %Wb, %v2180 : tensor<64x10xf32>
    %v2182 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v2183 = stablehlo.multiply %v2182, %v2175 : tensor<64x10xf32>
    %v2184 = stablehlo.multiply %v2183, %Wb : tensor<64x10xf32>
    %v2185 = stablehlo.subtract %v2181, %v2184 : tensor<64x10xf32>
    %v2186 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v2187 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v2188 = stablehlo.multiply %v2186, %Wbm : tensor<64x10xf32>
    %v2189 = stablehlo.multiply %v2187, %v2159 : tensor<64x10xf32>
    %v2190 = stablehlo.add %v2188, %v2189 : tensor<64x10xf32>
    %v2191 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v2192 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
    %v2193 = stablehlo.multiply %v2191, %Wbv : tensor<64x10xf32>
    %v2194 = stablehlo.multiply %v2159, %v2159 : tensor<64x10xf32>
    %v2195 = stablehlo.multiply %v2192, %v2194 : tensor<64x10xf32>
    %v2196 = stablehlo.add %v2193, %v2195 : tensor<64x10xf32>
    %v2197 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2198 = stablehlo.reduce(%v252 init: %v2197) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v2199 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2200 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2201 = stablehlo.multiply %v2199, %bbm : tensor<10xf32>
    %v2202 = stablehlo.multiply %v2200, %v2198 : tensor<10xf32>
    %v2203 = stablehlo.add %v2201, %v2202 : tensor<10xf32>
    %v2204 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2205 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2206 = stablehlo.multiply %v2204, %bbv : tensor<10xf32>
    %v2207 = stablehlo.multiply %v2198, %v2198 : tensor<10xf32>
    %v2208 = stablehlo.multiply %v2205, %v2207 : tensor<10xf32>
    %v2209 = stablehlo.add %v2206, %v2208 : tensor<10xf32>
    %v2210 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2211 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2212 = stablehlo.divide %v2203, %v2210 : tensor<10xf32>
    %v2213 = stablehlo.divide %v2209, %v2211 : tensor<10xf32>
    %v2214 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2215 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2216 = stablehlo.sqrt %v2213 : tensor<10xf32>
    %v2217 = stablehlo.add %v2216, %v2215 : tensor<10xf32>
    %v2218 = stablehlo.divide %v2212, %v2217 : tensor<10xf32>
    %v2219 = stablehlo.multiply %v2214, %v2218 : tensor<10xf32>
    %v2220 = stablehlo.subtract %bb, %v2219 : tensor<10xf32>
    %v2221 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2222 = stablehlo.multiply %v2221, %v2214 : tensor<10xf32>
    %v2223 = stablehlo.multiply %v2222, %bb : tensor<10xf32>
    %v2224 = stablehlo.subtract %v2220, %v2223 : tensor<10xf32>
    %v2225 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2226 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2227 = stablehlo.multiply %v2225, %bbm : tensor<10xf32>
    %v2228 = stablehlo.multiply %v2226, %v2198 : tensor<10xf32>
    %v2229 = stablehlo.add %v2227, %v2228 : tensor<10xf32>
    %v2230 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2231 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v2232 = stablehlo.multiply %v2230, %bbv : tensor<10xf32>
    %v2233 = stablehlo.multiply %v2198, %v2198 : tensor<10xf32>
    %v2234 = stablehlo.multiply %v2231, %v2233 : tensor<10xf32>
    %v2235 = stablehlo.add %v2232, %v2234 : tensor<10xf32>
    return %v612, %v652, %v707, %v747, %v790, %v830, %v885, %v925, %v968, %v1008, %v1063, %v1103, %v1146, %v1186, %v1241, %v1281, %v1324, %v1364, %v1419, %v1459, %v1502, %v1542, %v1597, %v1637, %v1680, %v1720, %v1775, %v1815, %v1858, %v1898, %v1953, %v1993, %v2031, %v2070, %v2108, %v2147, %v2185, %v2224, %v617, %v657, %v712, %v752, %v795, %v835, %v890, %v930, %v973, %v1013, %v1068, %v1108, %v1151, %v1191, %v1246, %v1286, %v1329, %v1369, %v1424, %v1464, %v1507, %v1547, %v1602, %v1642, %v1685, %v1725, %v1780, %v1820, %v1863, %v1903, %v1958, %v1998, %v2036, %v2075, %v2113, %v2152, %v2190, %v2229, %v623, %v663, %v718, %v758, %v801, %v841, %v896, %v936, %v979, %v1019, %v1074, %v1114, %v1157, %v1197, %v1252, %v1292, %v1335, %v1375, %v1430, %v1470, %v1513, %v1553, %v1608, %v1648, %v1691, %v1731, %v1786, %v1826, %v1869, %v1909, %v1964, %v2004, %v2042, %v2081, %v2119, %v2158, %v2196, %v2235, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, tensor<64x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
