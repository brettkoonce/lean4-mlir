module @m {
  func.func @cifar8w_bn_mom_train_step(%x: tensor<128x3072xf32>, %W1: tensor<16x3x3x3xf32>, %cb1: tensor<16xf32>, %g1: tensor<16xf32>, %bt1: tensor<16xf32>, %W2: tensor<16x16x3x3xf32>, %cb2: tensor<16xf32>, %g2: tensor<16xf32>, %bt2: tensor<16xf32>, %W3: tensor<16x16x3x3xf32>, %cb3: tensor<16xf32>, %g3: tensor<16xf32>, %bt3: tensor<16xf32>, %W4: tensor<16x16x3x3xf32>, %cb4: tensor<16xf32>, %g4: tensor<16xf32>, %bt4: tensor<16xf32>, %W5: tensor<32x16x3x3xf32>, %cb5: tensor<32xf32>, %g5: tensor<32xf32>, %bt5: tensor<32xf32>, %W6: tensor<32x32x3x3xf32>, %cb6: tensor<32xf32>, %g6: tensor<32xf32>, %bt6: tensor<32xf32>, %W7: tensor<32x32x3x3xf32>, %cb7: tensor<32xf32>, %g7: tensor<32xf32>, %bt7: tensor<32xf32>, %W8: tensor<32x32x3x3xf32>, %cb8: tensor<32xf32>, %g8: tensor<32xf32>, %bt8: tensor<32xf32>, %W9: tensor<128x512xf32>, %b9: tensor<512xf32>, %Wa: tensor<512x512xf32>, %ba: tensor<512xf32>, %Wb: tensor<512x10xf32>, %bb: tensor<10xf32>, %W1m: tensor<16x3x3x3xf32>, %cb1m: tensor<16xf32>, %g1m: tensor<16xf32>, %bt1m: tensor<16xf32>, %W2m: tensor<16x16x3x3xf32>, %cb2m: tensor<16xf32>, %g2m: tensor<16xf32>, %bt2m: tensor<16xf32>, %W3m: tensor<16x16x3x3xf32>, %cb3m: tensor<16xf32>, %g3m: tensor<16xf32>, %bt3m: tensor<16xf32>, %W4m: tensor<16x16x3x3xf32>, %cb4m: tensor<16xf32>, %g4m: tensor<16xf32>, %bt4m: tensor<16xf32>, %W5m: tensor<32x16x3x3xf32>, %cb5m: tensor<32xf32>, %g5m: tensor<32xf32>, %bt5m: tensor<32xf32>, %W6m: tensor<32x32x3x3xf32>, %cb6m: tensor<32xf32>, %g6m: tensor<32xf32>, %bt6m: tensor<32xf32>, %W7m: tensor<32x32x3x3xf32>, %cb7m: tensor<32xf32>, %g7m: tensor<32xf32>, %bt7m: tensor<32xf32>, %W8m: tensor<32x32x3x3xf32>, %cb8m: tensor<32xf32>, %g8m: tensor<32xf32>, %bt8m: tensor<32xf32>, %W9m: tensor<128x512xf32>, %b9m: tensor<512xf32>, %Wam: tensor<512x512xf32>, %bam: tensor<512xf32>, %Wbm: tensor<512x10xf32>, %bbm: tensor<10xf32>, %W1v: tensor<16x3x3x3xf32>, %cb1v: tensor<16xf32>, %g1v: tensor<16xf32>, %bt1v: tensor<16xf32>, %W2v: tensor<16x16x3x3xf32>, %cb2v: tensor<16xf32>, %g2v: tensor<16xf32>, %bt2v: tensor<16xf32>, %W3v: tensor<16x16x3x3xf32>, %cb3v: tensor<16xf32>, %g3v: tensor<16xf32>, %bt3v: tensor<16xf32>, %W4v: tensor<16x16x3x3xf32>, %cb4v: tensor<16xf32>, %g4v: tensor<16xf32>, %bt4v: tensor<16xf32>, %W5v: tensor<32x16x3x3xf32>, %cb5v: tensor<32xf32>, %g5v: tensor<32xf32>, %bt5v: tensor<32xf32>, %W6v: tensor<32x32x3x3xf32>, %cb6v: tensor<32xf32>, %g6v: tensor<32xf32>, %bt6v: tensor<32xf32>, %W7v: tensor<32x32x3x3xf32>, %cb7v: tensor<32xf32>, %g7v: tensor<32xf32>, %bt7v: tensor<32xf32>, %W8v: tensor<32x32x3x3xf32>, %cb8v: tensor<32xf32>, %g8v: tensor<32xf32>, %bt8v: tensor<32xf32>, %W9v: tensor<128x512xf32>, %b9v: tensor<512xf32>, %Wav: tensor<512x512xf32>, %bav: tensor<512xf32>, %Wbv: tensor<512x10xf32>, %bbv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: tensor<128x10xf32>) -> (tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>) {
    // ── cifar8-bn train step: every line is pretty(verified AST node), except the
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
    %v232 = stablehlo.dot_general %v231, %W9, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x512xf32>) -> tensor<128x512xf32>
    %v233 = stablehlo.broadcast_in_dim %b9, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v234 = stablehlo.add %v232, %v233 : tensor<128x512xf32>
    %v235 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v236 = stablehlo.maximum %v234, %v235 : tensor<128x512xf32>
    %v237 = stablehlo.dot_general %v236, %Wa, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %v238 = stablehlo.broadcast_in_dim %ba, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %v239 = stablehlo.add %v237, %v238 : tensor<128x512xf32>
    %v240 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v241 = stablehlo.maximum %v239, %v240 : tensor<128x512xf32>
    %v242 = stablehlo.dot_general %v241, %Wb, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x10xf32>) -> tensor<128x10xf32>
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
    %v253 = stablehlo.dot_general %v252, %Wb, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x10xf32>, tensor<512x10xf32>) -> tensor<128x512xf32>
    %v254 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v255 = stablehlo.compare GT, %v239, %v254 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v256 = stablehlo.select %v255, %v253, %v254 : tensor<128x512xi1>, tensor<128x512xf32>
    %v257 = stablehlo.dot_general %v256, %Wa, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %v258 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v259 = stablehlo.compare GT, %v234, %v258 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v260 = stablehlo.select %v259, %v257, %v258 : tensor<128x512xi1>, tensor<128x512xf32>
    %v261 = stablehlo.dot_general %v260, %W9, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x128xf32>
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
    %v587 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v588 = stablehlo.multiply %v587, %W1v : tensor<16x3x3x3xf32>
    %v589 = stablehlo.add %v588, %v586 : tensor<16x3x3x3xf32>
    %v590 = stablehlo.multiply %v587, %v589 : tensor<16x3x3x3xf32>
    %v591 = stablehlo.add %v590, %v586 : tensor<16x3x3x3xf32>
    %v592 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v593 = stablehlo.multiply %v592, %v591 : tensor<16x3x3x3xf32>
    %v594 = stablehlo.subtract %W1, %v593 : tensor<16x3x3x3xf32>
    %v595 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v596 = stablehlo.multiply %v595, %W1v : tensor<16x3x3x3xf32>
    %v597 = stablehlo.add %v596, %v586 : tensor<16x3x3x3xf32>
    %v598 = stablehlo.reshape %v580 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v599 = stablehlo.constant dense<0.0> : tensor<f32>
    %v600 = stablehlo.reduce(%v598 init: %v599) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v601 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v602 = stablehlo.multiply %v601, %cb1v : tensor<16xf32>
    %v603 = stablehlo.add %v602, %v600 : tensor<16xf32>
    %v604 = stablehlo.multiply %v601, %v603 : tensor<16xf32>
    %v605 = stablehlo.add %v604, %v600 : tensor<16xf32>
    %v606 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v607 = stablehlo.multiply %v606, %v605 : tensor<16xf32>
    %v608 = stablehlo.subtract %cb1, %v607 : tensor<16xf32>
    %v609 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v610 = stablehlo.multiply %v609, %cb1v : tensor<16xf32>
    %v611 = stablehlo.add %v610, %v600 : tensor<16xf32>
    %v612 = stablehlo.constant dense<0.0> : tensor<f32>
    %v613 = stablehlo.reshape %v4 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v614 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v615 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v616 = stablehlo.reduce(%v613 init: %v612) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v617 = stablehlo.broadcast_in_dim %v616, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v618 = stablehlo.divide %v617, %v614 : tensor<128x16x32x32xf32>
    %v619 = stablehlo.subtract %v613, %v618 : tensor<128x16x32x32xf32>
    %v620 = stablehlo.multiply %v619, %v619 : tensor<128x16x32x32xf32>
    %v621 = stablehlo.reduce(%v620 init: %v612) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v622 = stablehlo.broadcast_in_dim %v621, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v623 = stablehlo.divide %v622, %v614 : tensor<128x16x32x32xf32>
    %v624 = stablehlo.add %v623, %v615 : tensor<128x16x32x32xf32>
    %v625 = stablehlo.rsqrt %v624 : tensor<128x16x32x32xf32>
    %v626 = stablehlo.multiply %v619, %v625 : tensor<128x16x32x32xf32>
    %v627 = stablehlo.reshape %v550 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v628 = stablehlo.multiply %v627, %v626 : tensor<128x16x32x32xf32>
    %v629 = stablehlo.reduce(%v628 init: %v612) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v630 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v631 = stablehlo.multiply %v630, %g1v : tensor<16xf32>
    %v632 = stablehlo.add %v631, %v629 : tensor<16xf32>
    %v633 = stablehlo.multiply %v630, %v632 : tensor<16xf32>
    %v634 = stablehlo.add %v633, %v629 : tensor<16xf32>
    %v635 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v636 = stablehlo.multiply %v635, %v634 : tensor<16xf32>
    %v637 = stablehlo.subtract %g1, %v636 : tensor<16xf32>
    %v638 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v639 = stablehlo.multiply %v638, %g1v : tensor<16xf32>
    %v640 = stablehlo.add %v639, %v629 : tensor<16xf32>
    %v641 = stablehlo.constant dense<0.0> : tensor<f32>
    %v642 = stablehlo.reshape %v550 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v643 = stablehlo.reduce(%v642 init: %v641) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v644 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v645 = stablehlo.multiply %v644, %bt1v : tensor<16xf32>
    %v646 = stablehlo.add %v645, %v643 : tensor<16xf32>
    %v647 = stablehlo.multiply %v644, %v646 : tensor<16xf32>
    %v648 = stablehlo.add %v647, %v643 : tensor<16xf32>
    %v649 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v650 = stablehlo.multiply %v649, %v648 : tensor<16xf32>
    %v651 = stablehlo.subtract %bt1, %v650 : tensor<16xf32>
    %v652 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v653 = stablehlo.multiply %v652, %bt1v : tensor<16xf32>
    %v654 = stablehlo.add %v653, %v643 : tensor<16xf32>
    %v655 = stablehlo.reshape %v26 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v656 = stablehlo.reshape %v542 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v657 = stablehlo.transpose %v655, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v658 = stablehlo.transpose %v656, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v659 = stablehlo.convolution(%v657, %v658)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x16x3x3xf32>
    %v660 = stablehlo.transpose %v659, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v661 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v662 = stablehlo.multiply %v661, %W2v : tensor<16x16x3x3xf32>
    %v663 = stablehlo.add %v662, %v660 : tensor<16x16x3x3xf32>
    %v664 = stablehlo.multiply %v661, %v663 : tensor<16x16x3x3xf32>
    %v665 = stablehlo.add %v664, %v660 : tensor<16x16x3x3xf32>
    %v666 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v667 = stablehlo.multiply %v666, %v665 : tensor<16x16x3x3xf32>
    %v668 = stablehlo.subtract %W2, %v667 : tensor<16x16x3x3xf32>
    %v669 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v670 = stablehlo.multiply %v669, %W2v : tensor<16x16x3x3xf32>
    %v671 = stablehlo.add %v670, %v660 : tensor<16x16x3x3xf32>
    %v672 = stablehlo.reshape %v542 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v673 = stablehlo.constant dense<0.0> : tensor<f32>
    %v674 = stablehlo.reduce(%v672 init: %v673) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v675 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v676 = stablehlo.multiply %v675, %cb2v : tensor<16xf32>
    %v677 = stablehlo.add %v676, %v674 : tensor<16xf32>
    %v678 = stablehlo.multiply %v675, %v677 : tensor<16xf32>
    %v679 = stablehlo.add %v678, %v674 : tensor<16xf32>
    %v680 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v681 = stablehlo.multiply %v680, %v679 : tensor<16xf32>
    %v682 = stablehlo.subtract %cb2, %v681 : tensor<16xf32>
    %v683 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v684 = stablehlo.multiply %v683, %cb2v : tensor<16xf32>
    %v685 = stablehlo.add %v684, %v674 : tensor<16xf32>
    %v686 = stablehlo.constant dense<0.0> : tensor<f32>
    %v687 = stablehlo.reshape %v31 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v688 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v689 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v690 = stablehlo.reduce(%v687 init: %v686) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v691 = stablehlo.broadcast_in_dim %v690, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v692 = stablehlo.divide %v691, %v688 : tensor<128x16x32x32xf32>
    %v693 = stablehlo.subtract %v687, %v692 : tensor<128x16x32x32xf32>
    %v694 = stablehlo.multiply %v693, %v693 : tensor<128x16x32x32xf32>
    %v695 = stablehlo.reduce(%v694 init: %v686) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v696 = stablehlo.broadcast_in_dim %v695, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v697 = stablehlo.divide %v696, %v688 : tensor<128x16x32x32xf32>
    %v698 = stablehlo.add %v697, %v689 : tensor<128x16x32x32xf32>
    %v699 = stablehlo.rsqrt %v698 : tensor<128x16x32x32xf32>
    %v700 = stablehlo.multiply %v693, %v699 : tensor<128x16x32x32xf32>
    %v701 = stablehlo.reshape %v512 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v702 = stablehlo.multiply %v701, %v700 : tensor<128x16x32x32xf32>
    %v703 = stablehlo.reduce(%v702 init: %v686) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v704 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v705 = stablehlo.multiply %v704, %g2v : tensor<16xf32>
    %v706 = stablehlo.add %v705, %v703 : tensor<16xf32>
    %v707 = stablehlo.multiply %v704, %v706 : tensor<16xf32>
    %v708 = stablehlo.add %v707, %v703 : tensor<16xf32>
    %v709 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v710 = stablehlo.multiply %v709, %v708 : tensor<16xf32>
    %v711 = stablehlo.subtract %g2, %v710 : tensor<16xf32>
    %v712 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v713 = stablehlo.multiply %v712, %g2v : tensor<16xf32>
    %v714 = stablehlo.add %v713, %v703 : tensor<16xf32>
    %v715 = stablehlo.constant dense<0.0> : tensor<f32>
    %v716 = stablehlo.reshape %v512 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v717 = stablehlo.reduce(%v716 init: %v715) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v718 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v719 = stablehlo.multiply %v718, %bt2v : tensor<16xf32>
    %v720 = stablehlo.add %v719, %v717 : tensor<16xf32>
    %v721 = stablehlo.multiply %v718, %v720 : tensor<16xf32>
    %v722 = stablehlo.add %v721, %v717 : tensor<16xf32>
    %v723 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v724 = stablehlo.multiply %v723, %v722 : tensor<16xf32>
    %v725 = stablehlo.subtract %bt2, %v724 : tensor<16xf32>
    %v726 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v727 = stablehlo.multiply %v726, %bt2v : tensor<16xf32>
    %v728 = stablehlo.add %v727, %v717 : tensor<16xf32>
    %v729 = stablehlo.reshape %v57 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v730 = stablehlo.reshape %v499 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v731 = stablehlo.transpose %v729, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v732 = stablehlo.transpose %v730, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v733 = stablehlo.convolution(%v731, %v732)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v734 = stablehlo.transpose %v733, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v735 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v736 = stablehlo.multiply %v735, %W3v : tensor<16x16x3x3xf32>
    %v737 = stablehlo.add %v736, %v734 : tensor<16x16x3x3xf32>
    %v738 = stablehlo.multiply %v735, %v737 : tensor<16x16x3x3xf32>
    %v739 = stablehlo.add %v738, %v734 : tensor<16x16x3x3xf32>
    %v740 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v741 = stablehlo.multiply %v740, %v739 : tensor<16x16x3x3xf32>
    %v742 = stablehlo.subtract %W3, %v741 : tensor<16x16x3x3xf32>
    %v743 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v744 = stablehlo.multiply %v743, %W3v : tensor<16x16x3x3xf32>
    %v745 = stablehlo.add %v744, %v734 : tensor<16x16x3x3xf32>
    %v746 = stablehlo.reshape %v499 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v747 = stablehlo.constant dense<0.0> : tensor<f32>
    %v748 = stablehlo.reduce(%v746 init: %v747) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v749 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v750 = stablehlo.multiply %v749, %cb3v : tensor<16xf32>
    %v751 = stablehlo.add %v750, %v748 : tensor<16xf32>
    %v752 = stablehlo.multiply %v749, %v751 : tensor<16xf32>
    %v753 = stablehlo.add %v752, %v748 : tensor<16xf32>
    %v754 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v755 = stablehlo.multiply %v754, %v753 : tensor<16xf32>
    %v756 = stablehlo.subtract %cb3, %v755 : tensor<16xf32>
    %v757 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v758 = stablehlo.multiply %v757, %cb3v : tensor<16xf32>
    %v759 = stablehlo.add %v758, %v748 : tensor<16xf32>
    %v760 = stablehlo.constant dense<0.0> : tensor<f32>
    %v761 = stablehlo.reshape %v62 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v762 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v763 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v764 = stablehlo.reduce(%v761 init: %v760) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v765 = stablehlo.broadcast_in_dim %v764, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v766 = stablehlo.divide %v765, %v762 : tensor<128x16x16x16xf32>
    %v767 = stablehlo.subtract %v761, %v766 : tensor<128x16x16x16xf32>
    %v768 = stablehlo.multiply %v767, %v767 : tensor<128x16x16x16xf32>
    %v769 = stablehlo.reduce(%v768 init: %v760) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v770 = stablehlo.broadcast_in_dim %v769, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v771 = stablehlo.divide %v770, %v762 : tensor<128x16x16x16xf32>
    %v772 = stablehlo.add %v771, %v763 : tensor<128x16x16x16xf32>
    %v773 = stablehlo.rsqrt %v772 : tensor<128x16x16x16xf32>
    %v774 = stablehlo.multiply %v767, %v773 : tensor<128x16x16x16xf32>
    %v775 = stablehlo.reshape %v469 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v776 = stablehlo.multiply %v775, %v774 : tensor<128x16x16x16xf32>
    %v777 = stablehlo.reduce(%v776 init: %v760) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v778 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v779 = stablehlo.multiply %v778, %g3v : tensor<16xf32>
    %v780 = stablehlo.add %v779, %v777 : tensor<16xf32>
    %v781 = stablehlo.multiply %v778, %v780 : tensor<16xf32>
    %v782 = stablehlo.add %v781, %v777 : tensor<16xf32>
    %v783 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v784 = stablehlo.multiply %v783, %v782 : tensor<16xf32>
    %v785 = stablehlo.subtract %g3, %v784 : tensor<16xf32>
    %v786 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v787 = stablehlo.multiply %v786, %g3v : tensor<16xf32>
    %v788 = stablehlo.add %v787, %v777 : tensor<16xf32>
    %v789 = stablehlo.constant dense<0.0> : tensor<f32>
    %v790 = stablehlo.reshape %v469 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v791 = stablehlo.reduce(%v790 init: %v789) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v792 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v793 = stablehlo.multiply %v792, %bt3v : tensor<16xf32>
    %v794 = stablehlo.add %v793, %v791 : tensor<16xf32>
    %v795 = stablehlo.multiply %v792, %v794 : tensor<16xf32>
    %v796 = stablehlo.add %v795, %v791 : tensor<16xf32>
    %v797 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v798 = stablehlo.multiply %v797, %v796 : tensor<16xf32>
    %v799 = stablehlo.subtract %bt3, %v798 : tensor<16xf32>
    %v800 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v801 = stablehlo.multiply %v800, %bt3v : tensor<16xf32>
    %v802 = stablehlo.add %v801, %v791 : tensor<16xf32>
    %v803 = stablehlo.reshape %v84 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v804 = stablehlo.reshape %v461 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v805 = stablehlo.transpose %v803, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v806 = stablehlo.transpose %v804, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v807 = stablehlo.convolution(%v805, %v806)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v808 = stablehlo.transpose %v807, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v809 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v810 = stablehlo.multiply %v809, %W4v : tensor<16x16x3x3xf32>
    %v811 = stablehlo.add %v810, %v808 : tensor<16x16x3x3xf32>
    %v812 = stablehlo.multiply %v809, %v811 : tensor<16x16x3x3xf32>
    %v813 = stablehlo.add %v812, %v808 : tensor<16x16x3x3xf32>
    %v814 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v815 = stablehlo.multiply %v814, %v813 : tensor<16x16x3x3xf32>
    %v816 = stablehlo.subtract %W4, %v815 : tensor<16x16x3x3xf32>
    %v817 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v818 = stablehlo.multiply %v817, %W4v : tensor<16x16x3x3xf32>
    %v819 = stablehlo.add %v818, %v808 : tensor<16x16x3x3xf32>
    %v820 = stablehlo.reshape %v461 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v821 = stablehlo.constant dense<0.0> : tensor<f32>
    %v822 = stablehlo.reduce(%v820 init: %v821) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v823 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v824 = stablehlo.multiply %v823, %cb4v : tensor<16xf32>
    %v825 = stablehlo.add %v824, %v822 : tensor<16xf32>
    %v826 = stablehlo.multiply %v823, %v825 : tensor<16xf32>
    %v827 = stablehlo.add %v826, %v822 : tensor<16xf32>
    %v828 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v829 = stablehlo.multiply %v828, %v827 : tensor<16xf32>
    %v830 = stablehlo.subtract %cb4, %v829 : tensor<16xf32>
    %v831 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v832 = stablehlo.multiply %v831, %cb4v : tensor<16xf32>
    %v833 = stablehlo.add %v832, %v822 : tensor<16xf32>
    %v834 = stablehlo.constant dense<0.0> : tensor<f32>
    %v835 = stablehlo.reshape %v89 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v836 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v837 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v838 = stablehlo.reduce(%v835 init: %v834) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v839 = stablehlo.broadcast_in_dim %v838, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v840 = stablehlo.divide %v839, %v836 : tensor<128x16x16x16xf32>
    %v841 = stablehlo.subtract %v835, %v840 : tensor<128x16x16x16xf32>
    %v842 = stablehlo.multiply %v841, %v841 : tensor<128x16x16x16xf32>
    %v843 = stablehlo.reduce(%v842 init: %v834) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v844 = stablehlo.broadcast_in_dim %v843, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v845 = stablehlo.divide %v844, %v836 : tensor<128x16x16x16xf32>
    %v846 = stablehlo.add %v845, %v837 : tensor<128x16x16x16xf32>
    %v847 = stablehlo.rsqrt %v846 : tensor<128x16x16x16xf32>
    %v848 = stablehlo.multiply %v841, %v847 : tensor<128x16x16x16xf32>
    %v849 = stablehlo.reshape %v431 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v850 = stablehlo.multiply %v849, %v848 : tensor<128x16x16x16xf32>
    %v851 = stablehlo.reduce(%v850 init: %v834) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v852 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v853 = stablehlo.multiply %v852, %g4v : tensor<16xf32>
    %v854 = stablehlo.add %v853, %v851 : tensor<16xf32>
    %v855 = stablehlo.multiply %v852, %v854 : tensor<16xf32>
    %v856 = stablehlo.add %v855, %v851 : tensor<16xf32>
    %v857 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v858 = stablehlo.multiply %v857, %v856 : tensor<16xf32>
    %v859 = stablehlo.subtract %g4, %v858 : tensor<16xf32>
    %v860 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v861 = stablehlo.multiply %v860, %g4v : tensor<16xf32>
    %v862 = stablehlo.add %v861, %v851 : tensor<16xf32>
    %v863 = stablehlo.constant dense<0.0> : tensor<f32>
    %v864 = stablehlo.reshape %v431 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v865 = stablehlo.reduce(%v864 init: %v863) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v866 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v867 = stablehlo.multiply %v866, %bt4v : tensor<16xf32>
    %v868 = stablehlo.add %v867, %v865 : tensor<16xf32>
    %v869 = stablehlo.multiply %v866, %v868 : tensor<16xf32>
    %v870 = stablehlo.add %v869, %v865 : tensor<16xf32>
    %v871 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v872 = stablehlo.multiply %v871, %v870 : tensor<16xf32>
    %v873 = stablehlo.subtract %bt4, %v872 : tensor<16xf32>
    %v874 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v875 = stablehlo.multiply %v874, %bt4v : tensor<16xf32>
    %v876 = stablehlo.add %v875, %v865 : tensor<16xf32>
    %v877 = stablehlo.reshape %v115 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v878 = stablehlo.reshape %v418 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v879 = stablehlo.transpose %v877, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v880 = stablehlo.transpose %v878, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v881 = stablehlo.convolution(%v879, %v880)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<16x32x3x3xf32>
    %v882 = stablehlo.transpose %v881, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v883 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v884 = stablehlo.multiply %v883, %W5v : tensor<32x16x3x3xf32>
    %v885 = stablehlo.add %v884, %v882 : tensor<32x16x3x3xf32>
    %v886 = stablehlo.multiply %v883, %v885 : tensor<32x16x3x3xf32>
    %v887 = stablehlo.add %v886, %v882 : tensor<32x16x3x3xf32>
    %v888 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v889 = stablehlo.multiply %v888, %v887 : tensor<32x16x3x3xf32>
    %v890 = stablehlo.subtract %W5, %v889 : tensor<32x16x3x3xf32>
    %v891 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v892 = stablehlo.multiply %v891, %W5v : tensor<32x16x3x3xf32>
    %v893 = stablehlo.add %v892, %v882 : tensor<32x16x3x3xf32>
    %v894 = stablehlo.reshape %v418 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v895 = stablehlo.constant dense<0.0> : tensor<f32>
    %v896 = stablehlo.reduce(%v894 init: %v895) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v897 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v898 = stablehlo.multiply %v897, %cb5v : tensor<32xf32>
    %v899 = stablehlo.add %v898, %v896 : tensor<32xf32>
    %v900 = stablehlo.multiply %v897, %v899 : tensor<32xf32>
    %v901 = stablehlo.add %v900, %v896 : tensor<32xf32>
    %v902 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v903 = stablehlo.multiply %v902, %v901 : tensor<32xf32>
    %v904 = stablehlo.subtract %cb5, %v903 : tensor<32xf32>
    %v905 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v906 = stablehlo.multiply %v905, %cb5v : tensor<32xf32>
    %v907 = stablehlo.add %v906, %v896 : tensor<32xf32>
    %v908 = stablehlo.constant dense<0.0> : tensor<f32>
    %v909 = stablehlo.reshape %v120 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v910 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v911 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v912 = stablehlo.reduce(%v909 init: %v908) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v913 = stablehlo.broadcast_in_dim %v912, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v914 = stablehlo.divide %v913, %v910 : tensor<128x32x8x8xf32>
    %v915 = stablehlo.subtract %v909, %v914 : tensor<128x32x8x8xf32>
    %v916 = stablehlo.multiply %v915, %v915 : tensor<128x32x8x8xf32>
    %v917 = stablehlo.reduce(%v916 init: %v908) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v918 = stablehlo.broadcast_in_dim %v917, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v919 = stablehlo.divide %v918, %v910 : tensor<128x32x8x8xf32>
    %v920 = stablehlo.add %v919, %v911 : tensor<128x32x8x8xf32>
    %v921 = stablehlo.rsqrt %v920 : tensor<128x32x8x8xf32>
    %v922 = stablehlo.multiply %v915, %v921 : tensor<128x32x8x8xf32>
    %v923 = stablehlo.reshape %v388 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v924 = stablehlo.multiply %v923, %v922 : tensor<128x32x8x8xf32>
    %v925 = stablehlo.reduce(%v924 init: %v908) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v926 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v927 = stablehlo.multiply %v926, %g5v : tensor<32xf32>
    %v928 = stablehlo.add %v927, %v925 : tensor<32xf32>
    %v929 = stablehlo.multiply %v926, %v928 : tensor<32xf32>
    %v930 = stablehlo.add %v929, %v925 : tensor<32xf32>
    %v931 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v932 = stablehlo.multiply %v931, %v930 : tensor<32xf32>
    %v933 = stablehlo.subtract %g5, %v932 : tensor<32xf32>
    %v934 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v935 = stablehlo.multiply %v934, %g5v : tensor<32xf32>
    %v936 = stablehlo.add %v935, %v925 : tensor<32xf32>
    %v937 = stablehlo.constant dense<0.0> : tensor<f32>
    %v938 = stablehlo.reshape %v388 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v939 = stablehlo.reduce(%v938 init: %v937) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v940 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v941 = stablehlo.multiply %v940, %bt5v : tensor<32xf32>
    %v942 = stablehlo.add %v941, %v939 : tensor<32xf32>
    %v943 = stablehlo.multiply %v940, %v942 : tensor<32xf32>
    %v944 = stablehlo.add %v943, %v939 : tensor<32xf32>
    %v945 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v946 = stablehlo.multiply %v945, %v944 : tensor<32xf32>
    %v947 = stablehlo.subtract %bt5, %v946 : tensor<32xf32>
    %v948 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v949 = stablehlo.multiply %v948, %bt5v : tensor<32xf32>
    %v950 = stablehlo.add %v949, %v939 : tensor<32xf32>
    %v951 = stablehlo.reshape %v142 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v952 = stablehlo.reshape %v380 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v953 = stablehlo.transpose %v951, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v954 = stablehlo.transpose %v952, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v955 = stablehlo.convolution(%v953, %v954)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<32x32x3x3xf32>
    %v956 = stablehlo.transpose %v955, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v957 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v958 = stablehlo.multiply %v957, %W6v : tensor<32x32x3x3xf32>
    %v959 = stablehlo.add %v958, %v956 : tensor<32x32x3x3xf32>
    %v960 = stablehlo.multiply %v957, %v959 : tensor<32x32x3x3xf32>
    %v961 = stablehlo.add %v960, %v956 : tensor<32x32x3x3xf32>
    %v962 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v963 = stablehlo.multiply %v962, %v961 : tensor<32x32x3x3xf32>
    %v964 = stablehlo.subtract %W6, %v963 : tensor<32x32x3x3xf32>
    %v965 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v966 = stablehlo.multiply %v965, %W6v : tensor<32x32x3x3xf32>
    %v967 = stablehlo.add %v966, %v956 : tensor<32x32x3x3xf32>
    %v968 = stablehlo.reshape %v380 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v969 = stablehlo.constant dense<0.0> : tensor<f32>
    %v970 = stablehlo.reduce(%v968 init: %v969) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v971 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v972 = stablehlo.multiply %v971, %cb6v : tensor<32xf32>
    %v973 = stablehlo.add %v972, %v970 : tensor<32xf32>
    %v974 = stablehlo.multiply %v971, %v973 : tensor<32xf32>
    %v975 = stablehlo.add %v974, %v970 : tensor<32xf32>
    %v976 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v977 = stablehlo.multiply %v976, %v975 : tensor<32xf32>
    %v978 = stablehlo.subtract %cb6, %v977 : tensor<32xf32>
    %v979 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v980 = stablehlo.multiply %v979, %cb6v : tensor<32xf32>
    %v981 = stablehlo.add %v980, %v970 : tensor<32xf32>
    %v982 = stablehlo.constant dense<0.0> : tensor<f32>
    %v983 = stablehlo.reshape %v147 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v984 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v985 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v986 = stablehlo.reduce(%v983 init: %v982) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v987 = stablehlo.broadcast_in_dim %v986, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v988 = stablehlo.divide %v987, %v984 : tensor<128x32x8x8xf32>
    %v989 = stablehlo.subtract %v983, %v988 : tensor<128x32x8x8xf32>
    %v990 = stablehlo.multiply %v989, %v989 : tensor<128x32x8x8xf32>
    %v991 = stablehlo.reduce(%v990 init: %v982) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v992 = stablehlo.broadcast_in_dim %v991, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v993 = stablehlo.divide %v992, %v984 : tensor<128x32x8x8xf32>
    %v994 = stablehlo.add %v993, %v985 : tensor<128x32x8x8xf32>
    %v995 = stablehlo.rsqrt %v994 : tensor<128x32x8x8xf32>
    %v996 = stablehlo.multiply %v989, %v995 : tensor<128x32x8x8xf32>
    %v997 = stablehlo.reshape %v350 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v998 = stablehlo.multiply %v997, %v996 : tensor<128x32x8x8xf32>
    %v999 = stablehlo.reduce(%v998 init: %v982) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1000 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1001 = stablehlo.multiply %v1000, %g6v : tensor<32xf32>
    %v1002 = stablehlo.add %v1001, %v999 : tensor<32xf32>
    %v1003 = stablehlo.multiply %v1000, %v1002 : tensor<32xf32>
    %v1004 = stablehlo.add %v1003, %v999 : tensor<32xf32>
    %v1005 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1006 = stablehlo.multiply %v1005, %v1004 : tensor<32xf32>
    %v1007 = stablehlo.subtract %g6, %v1006 : tensor<32xf32>
    %v1008 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1009 = stablehlo.multiply %v1008, %g6v : tensor<32xf32>
    %v1010 = stablehlo.add %v1009, %v999 : tensor<32xf32>
    %v1011 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1012 = stablehlo.reshape %v350 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1013 = stablehlo.reduce(%v1012 init: %v1011) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1014 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1015 = stablehlo.multiply %v1014, %bt6v : tensor<32xf32>
    %v1016 = stablehlo.add %v1015, %v1013 : tensor<32xf32>
    %v1017 = stablehlo.multiply %v1014, %v1016 : tensor<32xf32>
    %v1018 = stablehlo.add %v1017, %v1013 : tensor<32xf32>
    %v1019 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1020 = stablehlo.multiply %v1019, %v1018 : tensor<32xf32>
    %v1021 = stablehlo.subtract %bt6, %v1020 : tensor<32xf32>
    %v1022 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1023 = stablehlo.multiply %v1022, %bt6v : tensor<32xf32>
    %v1024 = stablehlo.add %v1023, %v1013 : tensor<32xf32>
    %v1025 = stablehlo.reshape %v173 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1026 = stablehlo.reshape %v337 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1027 = stablehlo.transpose %v1025, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v1028 = stablehlo.transpose %v1026, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v1029 = stablehlo.convolution(%v1027, %v1028)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v1030 = stablehlo.transpose %v1029, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v1031 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1032 = stablehlo.multiply %v1031, %W7v : tensor<32x32x3x3xf32>
    %v1033 = stablehlo.add %v1032, %v1030 : tensor<32x32x3x3xf32>
    %v1034 = stablehlo.multiply %v1031, %v1033 : tensor<32x32x3x3xf32>
    %v1035 = stablehlo.add %v1034, %v1030 : tensor<32x32x3x3xf32>
    %v1036 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1037 = stablehlo.multiply %v1036, %v1035 : tensor<32x32x3x3xf32>
    %v1038 = stablehlo.subtract %W7, %v1037 : tensor<32x32x3x3xf32>
    %v1039 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1040 = stablehlo.multiply %v1039, %W7v : tensor<32x32x3x3xf32>
    %v1041 = stablehlo.add %v1040, %v1030 : tensor<32x32x3x3xf32>
    %v1042 = stablehlo.reshape %v337 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1043 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1044 = stablehlo.reduce(%v1042 init: %v1043) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1045 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1046 = stablehlo.multiply %v1045, %cb7v : tensor<32xf32>
    %v1047 = stablehlo.add %v1046, %v1044 : tensor<32xf32>
    %v1048 = stablehlo.multiply %v1045, %v1047 : tensor<32xf32>
    %v1049 = stablehlo.add %v1048, %v1044 : tensor<32xf32>
    %v1050 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1051 = stablehlo.multiply %v1050, %v1049 : tensor<32xf32>
    %v1052 = stablehlo.subtract %cb7, %v1051 : tensor<32xf32>
    %v1053 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1054 = stablehlo.multiply %v1053, %cb7v : tensor<32xf32>
    %v1055 = stablehlo.add %v1054, %v1044 : tensor<32xf32>
    %v1056 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1057 = stablehlo.reshape %v178 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1058 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v1059 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v1060 = stablehlo.reduce(%v1057 init: %v1056) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1061 = stablehlo.broadcast_in_dim %v1060, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v1062 = stablehlo.divide %v1061, %v1058 : tensor<128x32x4x4xf32>
    %v1063 = stablehlo.subtract %v1057, %v1062 : tensor<128x32x4x4xf32>
    %v1064 = stablehlo.multiply %v1063, %v1063 : tensor<128x32x4x4xf32>
    %v1065 = stablehlo.reduce(%v1064 init: %v1056) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1066 = stablehlo.broadcast_in_dim %v1065, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v1067 = stablehlo.divide %v1066, %v1058 : tensor<128x32x4x4xf32>
    %v1068 = stablehlo.add %v1067, %v1059 : tensor<128x32x4x4xf32>
    %v1069 = stablehlo.rsqrt %v1068 : tensor<128x32x4x4xf32>
    %v1070 = stablehlo.multiply %v1063, %v1069 : tensor<128x32x4x4xf32>
    %v1071 = stablehlo.reshape %v307 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1072 = stablehlo.multiply %v1071, %v1070 : tensor<128x32x4x4xf32>
    %v1073 = stablehlo.reduce(%v1072 init: %v1056) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1074 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1075 = stablehlo.multiply %v1074, %g7v : tensor<32xf32>
    %v1076 = stablehlo.add %v1075, %v1073 : tensor<32xf32>
    %v1077 = stablehlo.multiply %v1074, %v1076 : tensor<32xf32>
    %v1078 = stablehlo.add %v1077, %v1073 : tensor<32xf32>
    %v1079 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1080 = stablehlo.multiply %v1079, %v1078 : tensor<32xf32>
    %v1081 = stablehlo.subtract %g7, %v1080 : tensor<32xf32>
    %v1082 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1083 = stablehlo.multiply %v1082, %g7v : tensor<32xf32>
    %v1084 = stablehlo.add %v1083, %v1073 : tensor<32xf32>
    %v1085 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1086 = stablehlo.reshape %v307 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1087 = stablehlo.reduce(%v1086 init: %v1085) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1088 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1089 = stablehlo.multiply %v1088, %bt7v : tensor<32xf32>
    %v1090 = stablehlo.add %v1089, %v1087 : tensor<32xf32>
    %v1091 = stablehlo.multiply %v1088, %v1090 : tensor<32xf32>
    %v1092 = stablehlo.add %v1091, %v1087 : tensor<32xf32>
    %v1093 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1094 = stablehlo.multiply %v1093, %v1092 : tensor<32xf32>
    %v1095 = stablehlo.subtract %bt7, %v1094 : tensor<32xf32>
    %v1096 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1097 = stablehlo.multiply %v1096, %bt7v : tensor<32xf32>
    %v1098 = stablehlo.add %v1097, %v1087 : tensor<32xf32>
    %v1099 = stablehlo.reshape %v200 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1100 = stablehlo.reshape %v299 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1101 = stablehlo.transpose %v1099, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v1102 = stablehlo.transpose %v1100, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v1103 = stablehlo.convolution(%v1101, %v1102)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v1104 = stablehlo.transpose %v1103, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v1105 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1106 = stablehlo.multiply %v1105, %W8v : tensor<32x32x3x3xf32>
    %v1107 = stablehlo.add %v1106, %v1104 : tensor<32x32x3x3xf32>
    %v1108 = stablehlo.multiply %v1105, %v1107 : tensor<32x32x3x3xf32>
    %v1109 = stablehlo.add %v1108, %v1104 : tensor<32x32x3x3xf32>
    %v1110 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1111 = stablehlo.multiply %v1110, %v1109 : tensor<32x32x3x3xf32>
    %v1112 = stablehlo.subtract %W8, %v1111 : tensor<32x32x3x3xf32>
    %v1113 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1114 = stablehlo.multiply %v1113, %W8v : tensor<32x32x3x3xf32>
    %v1115 = stablehlo.add %v1114, %v1104 : tensor<32x32x3x3xf32>
    %v1116 = stablehlo.reshape %v299 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1117 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1118 = stablehlo.reduce(%v1116 init: %v1117) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1119 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1120 = stablehlo.multiply %v1119, %cb8v : tensor<32xf32>
    %v1121 = stablehlo.add %v1120, %v1118 : tensor<32xf32>
    %v1122 = stablehlo.multiply %v1119, %v1121 : tensor<32xf32>
    %v1123 = stablehlo.add %v1122, %v1118 : tensor<32xf32>
    %v1124 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1125 = stablehlo.multiply %v1124, %v1123 : tensor<32xf32>
    %v1126 = stablehlo.subtract %cb8, %v1125 : tensor<32xf32>
    %v1127 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1128 = stablehlo.multiply %v1127, %cb8v : tensor<32xf32>
    %v1129 = stablehlo.add %v1128, %v1118 : tensor<32xf32>
    %v1130 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1131 = stablehlo.reshape %v205 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1132 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v1133 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v1134 = stablehlo.reduce(%v1131 init: %v1130) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1135 = stablehlo.broadcast_in_dim %v1134, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v1136 = stablehlo.divide %v1135, %v1132 : tensor<128x32x4x4xf32>
    %v1137 = stablehlo.subtract %v1131, %v1136 : tensor<128x32x4x4xf32>
    %v1138 = stablehlo.multiply %v1137, %v1137 : tensor<128x32x4x4xf32>
    %v1139 = stablehlo.reduce(%v1138 init: %v1130) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1140 = stablehlo.broadcast_in_dim %v1139, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v1141 = stablehlo.divide %v1140, %v1132 : tensor<128x32x4x4xf32>
    %v1142 = stablehlo.add %v1141, %v1133 : tensor<128x32x4x4xf32>
    %v1143 = stablehlo.rsqrt %v1142 : tensor<128x32x4x4xf32>
    %v1144 = stablehlo.multiply %v1137, %v1143 : tensor<128x32x4x4xf32>
    %v1145 = stablehlo.reshape %v269 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1146 = stablehlo.multiply %v1145, %v1144 : tensor<128x32x4x4xf32>
    %v1147 = stablehlo.reduce(%v1146 init: %v1130) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1148 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1149 = stablehlo.multiply %v1148, %g8v : tensor<32xf32>
    %v1150 = stablehlo.add %v1149, %v1147 : tensor<32xf32>
    %v1151 = stablehlo.multiply %v1148, %v1150 : tensor<32xf32>
    %v1152 = stablehlo.add %v1151, %v1147 : tensor<32xf32>
    %v1153 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1154 = stablehlo.multiply %v1153, %v1152 : tensor<32xf32>
    %v1155 = stablehlo.subtract %g8, %v1154 : tensor<32xf32>
    %v1156 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1157 = stablehlo.multiply %v1156, %g8v : tensor<32xf32>
    %v1158 = stablehlo.add %v1157, %v1147 : tensor<32xf32>
    %v1159 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1160 = stablehlo.reshape %v269 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1161 = stablehlo.reduce(%v1160 init: %v1159) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1162 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1163 = stablehlo.multiply %v1162, %bt8v : tensor<32xf32>
    %v1164 = stablehlo.add %v1163, %v1161 : tensor<32xf32>
    %v1165 = stablehlo.multiply %v1162, %v1164 : tensor<32xf32>
    %v1166 = stablehlo.add %v1165, %v1161 : tensor<32xf32>
    %v1167 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1168 = stablehlo.multiply %v1167, %v1166 : tensor<32xf32>
    %v1169 = stablehlo.subtract %bt8, %v1168 : tensor<32xf32>
    %v1170 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1171 = stablehlo.multiply %v1170, %bt8v : tensor<32xf32>
    %v1172 = stablehlo.add %v1171, %v1161 : tensor<32xf32>
    %v1173 = stablehlo.dot_general %v231, %v260, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x512xf32>) -> tensor<128x512xf32>
    %v1174 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v1175 = stablehlo.multiply %v1174, %W9v : tensor<128x512xf32>
    %v1176 = stablehlo.add %v1175, %v1173 : tensor<128x512xf32>
    %v1177 = stablehlo.multiply %v1174, %v1176 : tensor<128x512xf32>
    %v1178 = stablehlo.add %v1177, %v1173 : tensor<128x512xf32>
    %v1179 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v1180 = stablehlo.multiply %v1179, %v1178 : tensor<128x512xf32>
    %v1181 = stablehlo.subtract %W9, %v1180 : tensor<128x512xf32>
    %v1182 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v1183 = stablehlo.multiply %v1182, %W9v : tensor<128x512xf32>
    %v1184 = stablehlo.add %v1183, %v1173 : tensor<128x512xf32>
    %v1185 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1186 = stablehlo.reduce(%v260 init: %v1185) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v1187 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1188 = stablehlo.multiply %v1187, %b9v : tensor<512xf32>
    %v1189 = stablehlo.add %v1188, %v1186 : tensor<512xf32>
    %v1190 = stablehlo.multiply %v1187, %v1189 : tensor<512xf32>
    %v1191 = stablehlo.add %v1190, %v1186 : tensor<512xf32>
    %v1192 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1193 = stablehlo.multiply %v1192, %v1191 : tensor<512xf32>
    %v1194 = stablehlo.subtract %b9, %v1193 : tensor<512xf32>
    %v1195 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1196 = stablehlo.multiply %v1195, %b9v : tensor<512xf32>
    %v1197 = stablehlo.add %v1196, %v1186 : tensor<512xf32>
    %v1198 = stablehlo.dot_general %v236, %v256, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<512x512xf32>
    %v1199 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1200 = stablehlo.multiply %v1199, %Wav : tensor<512x512xf32>
    %v1201 = stablehlo.add %v1200, %v1198 : tensor<512x512xf32>
    %v1202 = stablehlo.multiply %v1199, %v1201 : tensor<512x512xf32>
    %v1203 = stablehlo.add %v1202, %v1198 : tensor<512x512xf32>
    %v1204 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1205 = stablehlo.multiply %v1204, %v1203 : tensor<512x512xf32>
    %v1206 = stablehlo.subtract %Wa, %v1205 : tensor<512x512xf32>
    %v1207 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1208 = stablehlo.multiply %v1207, %Wav : tensor<512x512xf32>
    %v1209 = stablehlo.add %v1208, %v1198 : tensor<512x512xf32>
    %v1210 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1211 = stablehlo.reduce(%v256 init: %v1210) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v1212 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1213 = stablehlo.multiply %v1212, %bav : tensor<512xf32>
    %v1214 = stablehlo.add %v1213, %v1211 : tensor<512xf32>
    %v1215 = stablehlo.multiply %v1212, %v1214 : tensor<512xf32>
    %v1216 = stablehlo.add %v1215, %v1211 : tensor<512xf32>
    %v1217 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1218 = stablehlo.multiply %v1217, %v1216 : tensor<512xf32>
    %v1219 = stablehlo.subtract %ba, %v1218 : tensor<512xf32>
    %v1220 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1221 = stablehlo.multiply %v1220, %bav : tensor<512xf32>
    %v1222 = stablehlo.add %v1221, %v1211 : tensor<512xf32>
    %v1223 = stablehlo.dot_general %v241, %v252, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x10xf32>) -> tensor<512x10xf32>
    %v1224 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1225 = stablehlo.multiply %v1224, %Wbv : tensor<512x10xf32>
    %v1226 = stablehlo.add %v1225, %v1223 : tensor<512x10xf32>
    %v1227 = stablehlo.multiply %v1224, %v1226 : tensor<512x10xf32>
    %v1228 = stablehlo.add %v1227, %v1223 : tensor<512x10xf32>
    %v1229 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1230 = stablehlo.multiply %v1229, %v1228 : tensor<512x10xf32>
    %v1231 = stablehlo.subtract %Wb, %v1230 : tensor<512x10xf32>
    %v1232 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1233 = stablehlo.multiply %v1232, %Wbv : tensor<512x10xf32>
    %v1234 = stablehlo.add %v1233, %v1223 : tensor<512x10xf32>
    %v1235 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1236 = stablehlo.reduce(%v252 init: %v1235) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1237 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1238 = stablehlo.multiply %v1237, %bbv : tensor<10xf32>
    %v1239 = stablehlo.add %v1238, %v1236 : tensor<10xf32>
    %v1240 = stablehlo.multiply %v1237, %v1239 : tensor<10xf32>
    %v1241 = stablehlo.add %v1240, %v1236 : tensor<10xf32>
    %v1242 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1243 = stablehlo.multiply %v1242, %v1241 : tensor<10xf32>
    %v1244 = stablehlo.subtract %bb, %v1243 : tensor<10xf32>
    %v1245 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1246 = stablehlo.multiply %v1245, %bbv : tensor<10xf32>
    %v1247 = stablehlo.add %v1246, %v1236 : tensor<10xf32>
    return %v594, %v608, %v637, %v651, %v668, %v682, %v711, %v725, %v742, %v756, %v785, %v799, %v816, %v830, %v859, %v873, %v890, %v904, %v933, %v947, %v964, %v978, %v1007, %v1021, %v1038, %v1052, %v1081, %v1095, %v1112, %v1126, %v1155, %v1169, %v1181, %v1194, %v1206, %v1219, %v1231, %v1244, %W1m, %cb1m, %g1m, %bt1m, %W2m, %cb2m, %g2m, %bt2m, %W3m, %cb3m, %g3m, %bt3m, %W4m, %cb4m, %g4m, %bt4m, %W5m, %cb5m, %g5m, %bt5m, %W6m, %cb6m, %g6m, %bt6m, %W7m, %cb7m, %g7m, %bt7m, %W8m, %cb8m, %g8m, %bt8m, %W9m, %b9m, %Wam, %bam, %Wbm, %bbm, %v597, %v611, %v640, %v654, %v671, %v685, %v714, %v728, %v745, %v759, %v788, %v802, %v819, %v833, %v862, %v876, %v893, %v907, %v936, %v950, %v967, %v981, %v1010, %v1024, %v1041, %v1055, %v1084, %v1098, %v1115, %v1129, %v1158, %v1172, %v1184, %v1197, %v1209, %v1222, %v1234, %v1247, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
