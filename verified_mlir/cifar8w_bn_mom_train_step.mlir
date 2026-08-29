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
    %v269 = stablehlo.dot_general %v268, %Wb, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x10xf32>, tensor<512x10xf32>) -> tensor<128x512xf32>
    %v270 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v271 = stablehlo.compare GT, %v255, %v270 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v272 = stablehlo.select %v271, %v269, %v270 : tensor<128x512xi1>, tensor<128x512xf32>
    %v273 = stablehlo.dot_general %v272, %Wa, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %v274 = stablehlo.constant dense<0.0> : tensor<128x512xf32>
    %v275 = stablehlo.compare GT, %v250, %v274 : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xi1>
    %v276 = stablehlo.select %v275, %v273, %v274 : tensor<128x512xi1>, tensor<128x512xf32>
    %v277 = stablehlo.dot_general %v276, %W9, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x128xf32>
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
    %v627 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v628 = stablehlo.multiply %v627, %W1v : tensor<16x3x3x3xf32>
    %v629 = stablehlo.add %v628, %v626 : tensor<16x3x3x3xf32>
    %v630 = stablehlo.multiply %v627, %v629 : tensor<16x3x3x3xf32>
    %v631 = stablehlo.add %v630, %v626 : tensor<16x3x3x3xf32>
    %v632 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v633 = stablehlo.multiply %v632, %v631 : tensor<16x3x3x3xf32>
    %v634 = stablehlo.subtract %W1, %v633 : tensor<16x3x3x3xf32>
    %v635 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x3x3x3xf32>
    %v636 = stablehlo.multiply %v635, %W1v : tensor<16x3x3x3xf32>
    %v637 = stablehlo.add %v636, %v626 : tensor<16x3x3x3xf32>
    %v638 = stablehlo.reshape %v620 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v639 = stablehlo.constant dense<0.0> : tensor<f32>
    %v640 = stablehlo.reduce(%v638 init: %v639) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v641 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v642 = stablehlo.multiply %v641, %cb1v : tensor<16xf32>
    %v643 = stablehlo.add %v642, %v640 : tensor<16xf32>
    %v644 = stablehlo.multiply %v641, %v643 : tensor<16xf32>
    %v645 = stablehlo.add %v644, %v640 : tensor<16xf32>
    %v646 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v647 = stablehlo.multiply %v646, %v645 : tensor<16xf32>
    %v648 = stablehlo.subtract %cb1, %v647 : tensor<16xf32>
    %v649 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v650 = stablehlo.multiply %v649, %cb1v : tensor<16xf32>
    %v651 = stablehlo.add %v650, %v640 : tensor<16xf32>
    %v652 = stablehlo.constant dense<0.0> : tensor<f32>
    %v653 = stablehlo.reshape %v4 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v654 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v655 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v656 = stablehlo.reduce(%v653 init: %v652) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v657 = stablehlo.broadcast_in_dim %v656, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v658 = stablehlo.divide %v657, %v654 : tensor<128x16x32x32xf32>
    %v659 = stablehlo.subtract %v653, %v658 : tensor<128x16x32x32xf32>
    %v660 = stablehlo.multiply %v659, %v659 : tensor<128x16x32x32xf32>
    %v661 = stablehlo.reduce(%v660 init: %v652) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v662 = stablehlo.broadcast_in_dim %v661, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v663 = stablehlo.divide %v662, %v654 : tensor<128x16x32x32xf32>
    %v664 = stablehlo.add %v663, %v655 : tensor<128x16x32x32xf32>
    %v665 = stablehlo.rsqrt %v664 : tensor<128x16x32x32xf32>
    %v666 = stablehlo.multiply %v659, %v665 : tensor<128x16x32x32xf32>
    %v667 = stablehlo.reshape %v590 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v668 = stablehlo.multiply %v667, %v666 : tensor<128x16x32x32xf32>
    %v669 = stablehlo.reduce(%v668 init: %v652) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v670 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v671 = stablehlo.multiply %v670, %g1v : tensor<16xf32>
    %v672 = stablehlo.add %v671, %v669 : tensor<16xf32>
    %v673 = stablehlo.multiply %v670, %v672 : tensor<16xf32>
    %v674 = stablehlo.add %v673, %v669 : tensor<16xf32>
    %v675 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v676 = stablehlo.multiply %v675, %v674 : tensor<16xf32>
    %v677 = stablehlo.subtract %g1, %v676 : tensor<16xf32>
    %v678 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v679 = stablehlo.multiply %v678, %g1v : tensor<16xf32>
    %v680 = stablehlo.add %v679, %v669 : tensor<16xf32>
    %v681 = stablehlo.constant dense<0.0> : tensor<f32>
    %v682 = stablehlo.reshape %v590 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v683 = stablehlo.reduce(%v682 init: %v681) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v684 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v685 = stablehlo.multiply %v684, %bt1v : tensor<16xf32>
    %v686 = stablehlo.add %v685, %v683 : tensor<16xf32>
    %v687 = stablehlo.multiply %v684, %v686 : tensor<16xf32>
    %v688 = stablehlo.add %v687, %v683 : tensor<16xf32>
    %v689 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v690 = stablehlo.multiply %v689, %v688 : tensor<16xf32>
    %v691 = stablehlo.subtract %bt1, %v690 : tensor<16xf32>
    %v692 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v693 = stablehlo.multiply %v692, %bt1v : tensor<16xf32>
    %v694 = stablehlo.add %v693, %v683 : tensor<16xf32>
    %v695 = stablehlo.reshape %v28 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v696 = stablehlo.reshape %v579 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v697 = stablehlo.transpose %v695, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v698 = stablehlo.transpose %v696, dims = [1, 0, 2, 3] : (tensor<128x16x32x32xf32>) -> tensor<16x128x32x32xf32>
    %v699 = stablehlo.convolution(%v697, %v698)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x32x32xf32>, tensor<16x128x32x32xf32>) -> tensor<16x16x3x3xf32>
    %v700 = stablehlo.transpose %v699, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v701 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v702 = stablehlo.multiply %v701, %W2v : tensor<16x16x3x3xf32>
    %v703 = stablehlo.add %v702, %v700 : tensor<16x16x3x3xf32>
    %v704 = stablehlo.multiply %v701, %v703 : tensor<16x16x3x3xf32>
    %v705 = stablehlo.add %v704, %v700 : tensor<16x16x3x3xf32>
    %v706 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v707 = stablehlo.multiply %v706, %v705 : tensor<16x16x3x3xf32>
    %v708 = stablehlo.subtract %W2, %v707 : tensor<16x16x3x3xf32>
    %v709 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v710 = stablehlo.multiply %v709, %W2v : tensor<16x16x3x3xf32>
    %v711 = stablehlo.add %v710, %v700 : tensor<16x16x3x3xf32>
    %v712 = stablehlo.reshape %v579 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v713 = stablehlo.constant dense<0.0> : tensor<f32>
    %v714 = stablehlo.reduce(%v712 init: %v713) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v715 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v716 = stablehlo.multiply %v715, %cb2v : tensor<16xf32>
    %v717 = stablehlo.add %v716, %v714 : tensor<16xf32>
    %v718 = stablehlo.multiply %v715, %v717 : tensor<16xf32>
    %v719 = stablehlo.add %v718, %v714 : tensor<16xf32>
    %v720 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v721 = stablehlo.multiply %v720, %v719 : tensor<16xf32>
    %v722 = stablehlo.subtract %cb2, %v721 : tensor<16xf32>
    %v723 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v724 = stablehlo.multiply %v723, %cb2v : tensor<16xf32>
    %v725 = stablehlo.add %v724, %v714 : tensor<16xf32>
    %v726 = stablehlo.constant dense<0.0> : tensor<f32>
    %v727 = stablehlo.reshape %v33 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v728 = stablehlo.constant dense<1024.0> : tensor<128x16x32x32xf32>
    %v729 = stablehlo.constant dense<1.0e-05> : tensor<128x16x32x32xf32>
    %v730 = stablehlo.reduce(%v727 init: %v726) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v731 = stablehlo.broadcast_in_dim %v730, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v732 = stablehlo.divide %v731, %v728 : tensor<128x16x32x32xf32>
    %v733 = stablehlo.subtract %v727, %v732 : tensor<128x16x32x32xf32>
    %v734 = stablehlo.multiply %v733, %v733 : tensor<128x16x32x32xf32>
    %v735 = stablehlo.reduce(%v734 init: %v726) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v736 = stablehlo.broadcast_in_dim %v735, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x32x32xf32>
    %v737 = stablehlo.divide %v736, %v728 : tensor<128x16x32x32xf32>
    %v738 = stablehlo.add %v737, %v729 : tensor<128x16x32x32xf32>
    %v739 = stablehlo.rsqrt %v738 : tensor<128x16x32x32xf32>
    %v740 = stablehlo.multiply %v733, %v739 : tensor<128x16x32x32xf32>
    %v741 = stablehlo.reshape %v549 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v742 = stablehlo.multiply %v741, %v740 : tensor<128x16x32x32xf32>
    %v743 = stablehlo.reduce(%v742 init: %v726) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v744 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v745 = stablehlo.multiply %v744, %g2v : tensor<16xf32>
    %v746 = stablehlo.add %v745, %v743 : tensor<16xf32>
    %v747 = stablehlo.multiply %v744, %v746 : tensor<16xf32>
    %v748 = stablehlo.add %v747, %v743 : tensor<16xf32>
    %v749 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v750 = stablehlo.multiply %v749, %v748 : tensor<16xf32>
    %v751 = stablehlo.subtract %g2, %v750 : tensor<16xf32>
    %v752 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v753 = stablehlo.multiply %v752, %g2v : tensor<16xf32>
    %v754 = stablehlo.add %v753, %v743 : tensor<16xf32>
    %v755 = stablehlo.constant dense<0.0> : tensor<f32>
    %v756 = stablehlo.reshape %v549 : (tensor<128x16384xf32>) -> tensor<128x16x32x32xf32>
    %v757 = stablehlo.reduce(%v756 init: %v755) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x32x32xf32>, tensor<f32>) -> tensor<16xf32>
    %v758 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v759 = stablehlo.multiply %v758, %bt2v : tensor<16xf32>
    %v760 = stablehlo.add %v759, %v757 : tensor<16xf32>
    %v761 = stablehlo.multiply %v758, %v760 : tensor<16xf32>
    %v762 = stablehlo.add %v761, %v757 : tensor<16xf32>
    %v763 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v764 = stablehlo.multiply %v763, %v762 : tensor<16xf32>
    %v765 = stablehlo.subtract %bt2, %v764 : tensor<16xf32>
    %v766 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v767 = stablehlo.multiply %v766, %bt2v : tensor<16xf32>
    %v768 = stablehlo.add %v767, %v757 : tensor<16xf32>
    %v769 = stablehlo.reshape %v61 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v770 = stablehlo.reshape %v533 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v771 = stablehlo.transpose %v769, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v772 = stablehlo.transpose %v770, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v773 = stablehlo.convolution(%v771, %v772)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v774 = stablehlo.transpose %v773, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v775 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v776 = stablehlo.multiply %v775, %W3v : tensor<16x16x3x3xf32>
    %v777 = stablehlo.add %v776, %v774 : tensor<16x16x3x3xf32>
    %v778 = stablehlo.multiply %v775, %v777 : tensor<16x16x3x3xf32>
    %v779 = stablehlo.add %v778, %v774 : tensor<16x16x3x3xf32>
    %v780 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v781 = stablehlo.multiply %v780, %v779 : tensor<16x16x3x3xf32>
    %v782 = stablehlo.subtract %W3, %v781 : tensor<16x16x3x3xf32>
    %v783 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v784 = stablehlo.multiply %v783, %W3v : tensor<16x16x3x3xf32>
    %v785 = stablehlo.add %v784, %v774 : tensor<16x16x3x3xf32>
    %v786 = stablehlo.reshape %v533 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v787 = stablehlo.constant dense<0.0> : tensor<f32>
    %v788 = stablehlo.reduce(%v786 init: %v787) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v789 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v790 = stablehlo.multiply %v789, %cb3v : tensor<16xf32>
    %v791 = stablehlo.add %v790, %v788 : tensor<16xf32>
    %v792 = stablehlo.multiply %v789, %v791 : tensor<16xf32>
    %v793 = stablehlo.add %v792, %v788 : tensor<16xf32>
    %v794 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v795 = stablehlo.multiply %v794, %v793 : tensor<16xf32>
    %v796 = stablehlo.subtract %cb3, %v795 : tensor<16xf32>
    %v797 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v798 = stablehlo.multiply %v797, %cb3v : tensor<16xf32>
    %v799 = stablehlo.add %v798, %v788 : tensor<16xf32>
    %v800 = stablehlo.constant dense<0.0> : tensor<f32>
    %v801 = stablehlo.reshape %v66 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v802 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v803 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v804 = stablehlo.reduce(%v801 init: %v800) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v805 = stablehlo.broadcast_in_dim %v804, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v806 = stablehlo.divide %v805, %v802 : tensor<128x16x16x16xf32>
    %v807 = stablehlo.subtract %v801, %v806 : tensor<128x16x16x16xf32>
    %v808 = stablehlo.multiply %v807, %v807 : tensor<128x16x16x16xf32>
    %v809 = stablehlo.reduce(%v808 init: %v800) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v810 = stablehlo.broadcast_in_dim %v809, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v811 = stablehlo.divide %v810, %v802 : tensor<128x16x16x16xf32>
    %v812 = stablehlo.add %v811, %v803 : tensor<128x16x16x16xf32>
    %v813 = stablehlo.rsqrt %v812 : tensor<128x16x16x16xf32>
    %v814 = stablehlo.multiply %v807, %v813 : tensor<128x16x16x16xf32>
    %v815 = stablehlo.reshape %v503 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v816 = stablehlo.multiply %v815, %v814 : tensor<128x16x16x16xf32>
    %v817 = stablehlo.reduce(%v816 init: %v800) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v818 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v819 = stablehlo.multiply %v818, %g3v : tensor<16xf32>
    %v820 = stablehlo.add %v819, %v817 : tensor<16xf32>
    %v821 = stablehlo.multiply %v818, %v820 : tensor<16xf32>
    %v822 = stablehlo.add %v821, %v817 : tensor<16xf32>
    %v823 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v824 = stablehlo.multiply %v823, %v822 : tensor<16xf32>
    %v825 = stablehlo.subtract %g3, %v824 : tensor<16xf32>
    %v826 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v827 = stablehlo.multiply %v826, %g3v : tensor<16xf32>
    %v828 = stablehlo.add %v827, %v817 : tensor<16xf32>
    %v829 = stablehlo.constant dense<0.0> : tensor<f32>
    %v830 = stablehlo.reshape %v503 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v831 = stablehlo.reduce(%v830 init: %v829) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v832 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v833 = stablehlo.multiply %v832, %bt3v : tensor<16xf32>
    %v834 = stablehlo.add %v833, %v831 : tensor<16xf32>
    %v835 = stablehlo.multiply %v832, %v834 : tensor<16xf32>
    %v836 = stablehlo.add %v835, %v831 : tensor<16xf32>
    %v837 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v838 = stablehlo.multiply %v837, %v836 : tensor<16xf32>
    %v839 = stablehlo.subtract %bt3, %v838 : tensor<16xf32>
    %v840 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v841 = stablehlo.multiply %v840, %bt3v : tensor<16xf32>
    %v842 = stablehlo.add %v841, %v831 : tensor<16xf32>
    %v843 = stablehlo.reshape %v90 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v844 = stablehlo.reshape %v492 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v845 = stablehlo.transpose %v843, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v846 = stablehlo.transpose %v844, dims = [1, 0, 2, 3] : (tensor<128x16x16x16xf32>) -> tensor<16x128x16x16xf32>
    %v847 = stablehlo.convolution(%v845, %v846)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x16x16xf32>, tensor<16x128x16x16xf32>) -> tensor<16x16x3x3xf32>
    %v848 = stablehlo.transpose %v847, dims = [1, 0, 2, 3] : (tensor<16x16x3x3xf32>) -> tensor<16x16x3x3xf32>
    %v849 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v850 = stablehlo.multiply %v849, %W4v : tensor<16x16x3x3xf32>
    %v851 = stablehlo.add %v850, %v848 : tensor<16x16x3x3xf32>
    %v852 = stablehlo.multiply %v849, %v851 : tensor<16x16x3x3xf32>
    %v853 = stablehlo.add %v852, %v848 : tensor<16x16x3x3xf32>
    %v854 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v855 = stablehlo.multiply %v854, %v853 : tensor<16x16x3x3xf32>
    %v856 = stablehlo.subtract %W4, %v855 : tensor<16x16x3x3xf32>
    %v857 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16x16x3x3xf32>
    %v858 = stablehlo.multiply %v857, %W4v : tensor<16x16x3x3xf32>
    %v859 = stablehlo.add %v858, %v848 : tensor<16x16x3x3xf32>
    %v860 = stablehlo.reshape %v492 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v861 = stablehlo.constant dense<0.0> : tensor<f32>
    %v862 = stablehlo.reduce(%v860 init: %v861) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v863 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v864 = stablehlo.multiply %v863, %cb4v : tensor<16xf32>
    %v865 = stablehlo.add %v864, %v862 : tensor<16xf32>
    %v866 = stablehlo.multiply %v863, %v865 : tensor<16xf32>
    %v867 = stablehlo.add %v866, %v862 : tensor<16xf32>
    %v868 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v869 = stablehlo.multiply %v868, %v867 : tensor<16xf32>
    %v870 = stablehlo.subtract %cb4, %v869 : tensor<16xf32>
    %v871 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v872 = stablehlo.multiply %v871, %cb4v : tensor<16xf32>
    %v873 = stablehlo.add %v872, %v862 : tensor<16xf32>
    %v874 = stablehlo.constant dense<0.0> : tensor<f32>
    %v875 = stablehlo.reshape %v95 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v876 = stablehlo.constant dense<256.0> : tensor<128x16x16x16xf32>
    %v877 = stablehlo.constant dense<1.0e-05> : tensor<128x16x16x16xf32>
    %v878 = stablehlo.reduce(%v875 init: %v874) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v879 = stablehlo.broadcast_in_dim %v878, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v880 = stablehlo.divide %v879, %v876 : tensor<128x16x16x16xf32>
    %v881 = stablehlo.subtract %v875, %v880 : tensor<128x16x16x16xf32>
    %v882 = stablehlo.multiply %v881, %v881 : tensor<128x16x16x16xf32>
    %v883 = stablehlo.reduce(%v882 init: %v874) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %v884 = stablehlo.broadcast_in_dim %v883, dims = [0, 1] : (tensor<128x16xf32>) -> tensor<128x16x16x16xf32>
    %v885 = stablehlo.divide %v884, %v876 : tensor<128x16x16x16xf32>
    %v886 = stablehlo.add %v885, %v877 : tensor<128x16x16x16xf32>
    %v887 = stablehlo.rsqrt %v886 : tensor<128x16x16x16xf32>
    %v888 = stablehlo.multiply %v881, %v887 : tensor<128x16x16x16xf32>
    %v889 = stablehlo.reshape %v462 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v890 = stablehlo.multiply %v889, %v888 : tensor<128x16x16x16xf32>
    %v891 = stablehlo.reduce(%v890 init: %v874) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v892 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v893 = stablehlo.multiply %v892, %g4v : tensor<16xf32>
    %v894 = stablehlo.add %v893, %v891 : tensor<16xf32>
    %v895 = stablehlo.multiply %v892, %v894 : tensor<16xf32>
    %v896 = stablehlo.add %v895, %v891 : tensor<16xf32>
    %v897 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v898 = stablehlo.multiply %v897, %v896 : tensor<16xf32>
    %v899 = stablehlo.subtract %g4, %v898 : tensor<16xf32>
    %v900 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v901 = stablehlo.multiply %v900, %g4v : tensor<16xf32>
    %v902 = stablehlo.add %v901, %v891 : tensor<16xf32>
    %v903 = stablehlo.constant dense<0.0> : tensor<f32>
    %v904 = stablehlo.reshape %v462 : (tensor<128x4096xf32>) -> tensor<128x16x16x16xf32>
    %v905 = stablehlo.reduce(%v904 init: %v903) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x16x16x16xf32>, tensor<f32>) -> tensor<16xf32>
    %v906 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v907 = stablehlo.multiply %v906, %bt4v : tensor<16xf32>
    %v908 = stablehlo.add %v907, %v905 : tensor<16xf32>
    %v909 = stablehlo.multiply %v906, %v908 : tensor<16xf32>
    %v910 = stablehlo.add %v909, %v905 : tensor<16xf32>
    %v911 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v912 = stablehlo.multiply %v911, %v910 : tensor<16xf32>
    %v913 = stablehlo.subtract %bt4, %v912 : tensor<16xf32>
    %v914 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %v915 = stablehlo.multiply %v914, %bt4v : tensor<16xf32>
    %v916 = stablehlo.add %v915, %v905 : tensor<16xf32>
    %v917 = stablehlo.reshape %v123 : (tensor<128x1024xf32>) -> tensor<128x16x8x8xf32>
    %v918 = stablehlo.reshape %v446 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v919 = stablehlo.transpose %v917, dims = [1, 0, 2, 3] : (tensor<128x16x8x8xf32>) -> tensor<16x128x8x8xf32>
    %v920 = stablehlo.transpose %v918, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v921 = stablehlo.convolution(%v919, %v920)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<16x32x3x3xf32>
    %v922 = stablehlo.transpose %v921, dims = [1, 0, 2, 3] : (tensor<16x32x3x3xf32>) -> tensor<32x16x3x3xf32>
    %v923 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v924 = stablehlo.multiply %v923, %W5v : tensor<32x16x3x3xf32>
    %v925 = stablehlo.add %v924, %v922 : tensor<32x16x3x3xf32>
    %v926 = stablehlo.multiply %v923, %v925 : tensor<32x16x3x3xf32>
    %v927 = stablehlo.add %v926, %v922 : tensor<32x16x3x3xf32>
    %v928 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v929 = stablehlo.multiply %v928, %v927 : tensor<32x16x3x3xf32>
    %v930 = stablehlo.subtract %W5, %v929 : tensor<32x16x3x3xf32>
    %v931 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x16x3x3xf32>
    %v932 = stablehlo.multiply %v931, %W5v : tensor<32x16x3x3xf32>
    %v933 = stablehlo.add %v932, %v922 : tensor<32x16x3x3xf32>
    %v934 = stablehlo.reshape %v446 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v935 = stablehlo.constant dense<0.0> : tensor<f32>
    %v936 = stablehlo.reduce(%v934 init: %v935) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v937 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v938 = stablehlo.multiply %v937, %cb5v : tensor<32xf32>
    %v939 = stablehlo.add %v938, %v936 : tensor<32xf32>
    %v940 = stablehlo.multiply %v937, %v939 : tensor<32xf32>
    %v941 = stablehlo.add %v940, %v936 : tensor<32xf32>
    %v942 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v943 = stablehlo.multiply %v942, %v941 : tensor<32xf32>
    %v944 = stablehlo.subtract %cb5, %v943 : tensor<32xf32>
    %v945 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v946 = stablehlo.multiply %v945, %cb5v : tensor<32xf32>
    %v947 = stablehlo.add %v946, %v936 : tensor<32xf32>
    %v948 = stablehlo.constant dense<0.0> : tensor<f32>
    %v949 = stablehlo.reshape %v128 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v950 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v951 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v952 = stablehlo.reduce(%v949 init: %v948) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v953 = stablehlo.broadcast_in_dim %v952, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v954 = stablehlo.divide %v953, %v950 : tensor<128x32x8x8xf32>
    %v955 = stablehlo.subtract %v949, %v954 : tensor<128x32x8x8xf32>
    %v956 = stablehlo.multiply %v955, %v955 : tensor<128x32x8x8xf32>
    %v957 = stablehlo.reduce(%v956 init: %v948) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v958 = stablehlo.broadcast_in_dim %v957, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v959 = stablehlo.divide %v958, %v950 : tensor<128x32x8x8xf32>
    %v960 = stablehlo.add %v959, %v951 : tensor<128x32x8x8xf32>
    %v961 = stablehlo.rsqrt %v960 : tensor<128x32x8x8xf32>
    %v962 = stablehlo.multiply %v955, %v961 : tensor<128x32x8x8xf32>
    %v963 = stablehlo.reshape %v416 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v964 = stablehlo.multiply %v963, %v962 : tensor<128x32x8x8xf32>
    %v965 = stablehlo.reduce(%v964 init: %v948) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v966 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v967 = stablehlo.multiply %v966, %g5v : tensor<32xf32>
    %v968 = stablehlo.add %v967, %v965 : tensor<32xf32>
    %v969 = stablehlo.multiply %v966, %v968 : tensor<32xf32>
    %v970 = stablehlo.add %v969, %v965 : tensor<32xf32>
    %v971 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v972 = stablehlo.multiply %v971, %v970 : tensor<32xf32>
    %v973 = stablehlo.subtract %g5, %v972 : tensor<32xf32>
    %v974 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v975 = stablehlo.multiply %v974, %g5v : tensor<32xf32>
    %v976 = stablehlo.add %v975, %v965 : tensor<32xf32>
    %v977 = stablehlo.constant dense<0.0> : tensor<f32>
    %v978 = stablehlo.reshape %v416 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v979 = stablehlo.reduce(%v978 init: %v977) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v980 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v981 = stablehlo.multiply %v980, %bt5v : tensor<32xf32>
    %v982 = stablehlo.add %v981, %v979 : tensor<32xf32>
    %v983 = stablehlo.multiply %v980, %v982 : tensor<32xf32>
    %v984 = stablehlo.add %v983, %v979 : tensor<32xf32>
    %v985 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v986 = stablehlo.multiply %v985, %v984 : tensor<32xf32>
    %v987 = stablehlo.subtract %bt5, %v986 : tensor<32xf32>
    %v988 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v989 = stablehlo.multiply %v988, %bt5v : tensor<32xf32>
    %v990 = stablehlo.add %v989, %v979 : tensor<32xf32>
    %v991 = stablehlo.reshape %v152 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v992 = stablehlo.reshape %v405 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v993 = stablehlo.transpose %v991, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v994 = stablehlo.transpose %v992, dims = [1, 0, 2, 3] : (tensor<128x32x8x8xf32>) -> tensor<32x128x8x8xf32>
    %v995 = stablehlo.convolution(%v993, %v994)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x8x8xf32>, tensor<32x128x8x8xf32>) -> tensor<32x32x3x3xf32>
    %v996 = stablehlo.transpose %v995, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v997 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v998 = stablehlo.multiply %v997, %W6v : tensor<32x32x3x3xf32>
    %v999 = stablehlo.add %v998, %v996 : tensor<32x32x3x3xf32>
    %v1000 = stablehlo.multiply %v997, %v999 : tensor<32x32x3x3xf32>
    %v1001 = stablehlo.add %v1000, %v996 : tensor<32x32x3x3xf32>
    %v1002 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1003 = stablehlo.multiply %v1002, %v1001 : tensor<32x32x3x3xf32>
    %v1004 = stablehlo.subtract %W6, %v1003 : tensor<32x32x3x3xf32>
    %v1005 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1006 = stablehlo.multiply %v1005, %W6v : tensor<32x32x3x3xf32>
    %v1007 = stablehlo.add %v1006, %v996 : tensor<32x32x3x3xf32>
    %v1008 = stablehlo.reshape %v405 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1009 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1010 = stablehlo.reduce(%v1008 init: %v1009) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1011 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1012 = stablehlo.multiply %v1011, %cb6v : tensor<32xf32>
    %v1013 = stablehlo.add %v1012, %v1010 : tensor<32xf32>
    %v1014 = stablehlo.multiply %v1011, %v1013 : tensor<32xf32>
    %v1015 = stablehlo.add %v1014, %v1010 : tensor<32xf32>
    %v1016 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1017 = stablehlo.multiply %v1016, %v1015 : tensor<32xf32>
    %v1018 = stablehlo.subtract %cb6, %v1017 : tensor<32xf32>
    %v1019 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1020 = stablehlo.multiply %v1019, %cb6v : tensor<32xf32>
    %v1021 = stablehlo.add %v1020, %v1010 : tensor<32xf32>
    %v1022 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1023 = stablehlo.reshape %v157 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1024 = stablehlo.constant dense<64.0> : tensor<128x32x8x8xf32>
    %v1025 = stablehlo.constant dense<1.0e-05> : tensor<128x32x8x8xf32>
    %v1026 = stablehlo.reduce(%v1023 init: %v1022) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1027 = stablehlo.broadcast_in_dim %v1026, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v1028 = stablehlo.divide %v1027, %v1024 : tensor<128x32x8x8xf32>
    %v1029 = stablehlo.subtract %v1023, %v1028 : tensor<128x32x8x8xf32>
    %v1030 = stablehlo.multiply %v1029, %v1029 : tensor<128x32x8x8xf32>
    %v1031 = stablehlo.reduce(%v1030 init: %v1022) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1032 = stablehlo.broadcast_in_dim %v1031, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x8x8xf32>
    %v1033 = stablehlo.divide %v1032, %v1024 : tensor<128x32x8x8xf32>
    %v1034 = stablehlo.add %v1033, %v1025 : tensor<128x32x8x8xf32>
    %v1035 = stablehlo.rsqrt %v1034 : tensor<128x32x8x8xf32>
    %v1036 = stablehlo.multiply %v1029, %v1035 : tensor<128x32x8x8xf32>
    %v1037 = stablehlo.reshape %v375 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1038 = stablehlo.multiply %v1037, %v1036 : tensor<128x32x8x8xf32>
    %v1039 = stablehlo.reduce(%v1038 init: %v1022) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1040 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1041 = stablehlo.multiply %v1040, %g6v : tensor<32xf32>
    %v1042 = stablehlo.add %v1041, %v1039 : tensor<32xf32>
    %v1043 = stablehlo.multiply %v1040, %v1042 : tensor<32xf32>
    %v1044 = stablehlo.add %v1043, %v1039 : tensor<32xf32>
    %v1045 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1046 = stablehlo.multiply %v1045, %v1044 : tensor<32xf32>
    %v1047 = stablehlo.subtract %g6, %v1046 : tensor<32xf32>
    %v1048 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1049 = stablehlo.multiply %v1048, %g6v : tensor<32xf32>
    %v1050 = stablehlo.add %v1049, %v1039 : tensor<32xf32>
    %v1051 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1052 = stablehlo.reshape %v375 : (tensor<128x2048xf32>) -> tensor<128x32x8x8xf32>
    %v1053 = stablehlo.reduce(%v1052 init: %v1051) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x8x8xf32>, tensor<f32>) -> tensor<32xf32>
    %v1054 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1055 = stablehlo.multiply %v1054, %bt6v : tensor<32xf32>
    %v1056 = stablehlo.add %v1055, %v1053 : tensor<32xf32>
    %v1057 = stablehlo.multiply %v1054, %v1056 : tensor<32xf32>
    %v1058 = stablehlo.add %v1057, %v1053 : tensor<32xf32>
    %v1059 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1060 = stablehlo.multiply %v1059, %v1058 : tensor<32xf32>
    %v1061 = stablehlo.subtract %bt6, %v1060 : tensor<32xf32>
    %v1062 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1063 = stablehlo.multiply %v1062, %bt6v : tensor<32xf32>
    %v1064 = stablehlo.add %v1063, %v1053 : tensor<32xf32>
    %v1065 = stablehlo.reshape %v185 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1066 = stablehlo.reshape %v359 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1067 = stablehlo.transpose %v1065, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v1068 = stablehlo.transpose %v1066, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v1069 = stablehlo.convolution(%v1067, %v1068)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v1070 = stablehlo.transpose %v1069, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v1071 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1072 = stablehlo.multiply %v1071, %W7v : tensor<32x32x3x3xf32>
    %v1073 = stablehlo.add %v1072, %v1070 : tensor<32x32x3x3xf32>
    %v1074 = stablehlo.multiply %v1071, %v1073 : tensor<32x32x3x3xf32>
    %v1075 = stablehlo.add %v1074, %v1070 : tensor<32x32x3x3xf32>
    %v1076 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1077 = stablehlo.multiply %v1076, %v1075 : tensor<32x32x3x3xf32>
    %v1078 = stablehlo.subtract %W7, %v1077 : tensor<32x32x3x3xf32>
    %v1079 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1080 = stablehlo.multiply %v1079, %W7v : tensor<32x32x3x3xf32>
    %v1081 = stablehlo.add %v1080, %v1070 : tensor<32x32x3x3xf32>
    %v1082 = stablehlo.reshape %v359 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1083 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1084 = stablehlo.reduce(%v1082 init: %v1083) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1085 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1086 = stablehlo.multiply %v1085, %cb7v : tensor<32xf32>
    %v1087 = stablehlo.add %v1086, %v1084 : tensor<32xf32>
    %v1088 = stablehlo.multiply %v1085, %v1087 : tensor<32xf32>
    %v1089 = stablehlo.add %v1088, %v1084 : tensor<32xf32>
    %v1090 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1091 = stablehlo.multiply %v1090, %v1089 : tensor<32xf32>
    %v1092 = stablehlo.subtract %cb7, %v1091 : tensor<32xf32>
    %v1093 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1094 = stablehlo.multiply %v1093, %cb7v : tensor<32xf32>
    %v1095 = stablehlo.add %v1094, %v1084 : tensor<32xf32>
    %v1096 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1097 = stablehlo.reshape %v190 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1098 = stablehlo.constant dense<16.0> : tensor<128x32x4x4xf32>
    %v1099 = stablehlo.constant dense<1.0e-05> : tensor<128x32x4x4xf32>
    %v1100 = stablehlo.reduce(%v1097 init: %v1096) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1101 = stablehlo.broadcast_in_dim %v1100, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v1102 = stablehlo.divide %v1101, %v1098 : tensor<128x32x4x4xf32>
    %v1103 = stablehlo.subtract %v1097, %v1102 : tensor<128x32x4x4xf32>
    %v1104 = stablehlo.multiply %v1103, %v1103 : tensor<128x32x4x4xf32>
    %v1105 = stablehlo.reduce(%v1104 init: %v1096) applies stablehlo.add across dimensions = [2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    %v1106 = stablehlo.broadcast_in_dim %v1105, dims = [0, 1] : (tensor<128x32xf32>) -> tensor<128x32x4x4xf32>
    %v1107 = stablehlo.divide %v1106, %v1098 : tensor<128x32x4x4xf32>
    %v1108 = stablehlo.add %v1107, %v1099 : tensor<128x32x4x4xf32>
    %v1109 = stablehlo.rsqrt %v1108 : tensor<128x32x4x4xf32>
    %v1110 = stablehlo.multiply %v1103, %v1109 : tensor<128x32x4x4xf32>
    %v1111 = stablehlo.reshape %v329 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1112 = stablehlo.multiply %v1111, %v1110 : tensor<128x32x4x4xf32>
    %v1113 = stablehlo.reduce(%v1112 init: %v1096) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1114 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1115 = stablehlo.multiply %v1114, %g7v : tensor<32xf32>
    %v1116 = stablehlo.add %v1115, %v1113 : tensor<32xf32>
    %v1117 = stablehlo.multiply %v1114, %v1116 : tensor<32xf32>
    %v1118 = stablehlo.add %v1117, %v1113 : tensor<32xf32>
    %v1119 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1120 = stablehlo.multiply %v1119, %v1118 : tensor<32xf32>
    %v1121 = stablehlo.subtract %g7, %v1120 : tensor<32xf32>
    %v1122 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1123 = stablehlo.multiply %v1122, %g7v : tensor<32xf32>
    %v1124 = stablehlo.add %v1123, %v1113 : tensor<32xf32>
    %v1125 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1126 = stablehlo.reshape %v329 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1127 = stablehlo.reduce(%v1126 init: %v1125) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1128 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1129 = stablehlo.multiply %v1128, %bt7v : tensor<32xf32>
    %v1130 = stablehlo.add %v1129, %v1127 : tensor<32xf32>
    %v1131 = stablehlo.multiply %v1128, %v1130 : tensor<32xf32>
    %v1132 = stablehlo.add %v1131, %v1127 : tensor<32xf32>
    %v1133 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1134 = stablehlo.multiply %v1133, %v1132 : tensor<32xf32>
    %v1135 = stablehlo.subtract %bt7, %v1134 : tensor<32xf32>
    %v1136 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1137 = stablehlo.multiply %v1136, %bt7v : tensor<32xf32>
    %v1138 = stablehlo.add %v1137, %v1127 : tensor<32xf32>
    %v1139 = stablehlo.reshape %v214 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1140 = stablehlo.reshape %v318 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1141 = stablehlo.transpose %v1139, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v1142 = stablehlo.transpose %v1140, dims = [1, 0, 2, 3] : (tensor<128x32x4x4xf32>) -> tensor<32x128x4x4xf32>
    %v1143 = stablehlo.convolution(%v1141, %v1142)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x4x4xf32>, tensor<32x128x4x4xf32>) -> tensor<32x32x3x3xf32>
    %v1144 = stablehlo.transpose %v1143, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>
    %v1145 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1146 = stablehlo.multiply %v1145, %W8v : tensor<32x32x3x3xf32>
    %v1147 = stablehlo.add %v1146, %v1144 : tensor<32x32x3x3xf32>
    %v1148 = stablehlo.multiply %v1145, %v1147 : tensor<32x32x3x3xf32>
    %v1149 = stablehlo.add %v1148, %v1144 : tensor<32x32x3x3xf32>
    %v1150 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1151 = stablehlo.multiply %v1150, %v1149 : tensor<32x32x3x3xf32>
    %v1152 = stablehlo.subtract %W8, %v1151 : tensor<32x32x3x3xf32>
    %v1153 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32x32x3x3xf32>
    %v1154 = stablehlo.multiply %v1153, %W8v : tensor<32x32x3x3xf32>
    %v1155 = stablehlo.add %v1154, %v1144 : tensor<32x32x3x3xf32>
    %v1156 = stablehlo.reshape %v318 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1157 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1158 = stablehlo.reduce(%v1156 init: %v1157) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1159 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1160 = stablehlo.multiply %v1159, %cb8v : tensor<32xf32>
    %v1161 = stablehlo.add %v1160, %v1158 : tensor<32xf32>
    %v1162 = stablehlo.multiply %v1159, %v1161 : tensor<32xf32>
    %v1163 = stablehlo.add %v1162, %v1158 : tensor<32xf32>
    %v1164 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1165 = stablehlo.multiply %v1164, %v1163 : tensor<32xf32>
    %v1166 = stablehlo.subtract %cb8, %v1165 : tensor<32xf32>
    %v1167 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1168 = stablehlo.multiply %v1167, %cb8v : tensor<32xf32>
    %v1169 = stablehlo.add %v1168, %v1158 : tensor<32xf32>
    %v1170 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1171 = stablehlo.reshape %v219 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
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
    %v1185 = stablehlo.reshape %v288 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1186 = stablehlo.multiply %v1185, %v1184 : tensor<128x32x4x4xf32>
    %v1187 = stablehlo.reduce(%v1186 init: %v1170) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1188 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1189 = stablehlo.multiply %v1188, %g8v : tensor<32xf32>
    %v1190 = stablehlo.add %v1189, %v1187 : tensor<32xf32>
    %v1191 = stablehlo.multiply %v1188, %v1190 : tensor<32xf32>
    %v1192 = stablehlo.add %v1191, %v1187 : tensor<32xf32>
    %v1193 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1194 = stablehlo.multiply %v1193, %v1192 : tensor<32xf32>
    %v1195 = stablehlo.subtract %g8, %v1194 : tensor<32xf32>
    %v1196 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1197 = stablehlo.multiply %v1196, %g8v : tensor<32xf32>
    %v1198 = stablehlo.add %v1197, %v1187 : tensor<32xf32>
    %v1199 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1200 = stablehlo.reshape %v288 : (tensor<128x512xf32>) -> tensor<128x32x4x4xf32>
    %v1201 = stablehlo.reduce(%v1200 init: %v1199) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<128x32x4x4xf32>, tensor<f32>) -> tensor<32xf32>
    %v1202 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1203 = stablehlo.multiply %v1202, %bt8v : tensor<32xf32>
    %v1204 = stablehlo.add %v1203, %v1201 : tensor<32xf32>
    %v1205 = stablehlo.multiply %v1202, %v1204 : tensor<32xf32>
    %v1206 = stablehlo.add %v1205, %v1201 : tensor<32xf32>
    %v1207 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1208 = stablehlo.multiply %v1207, %v1206 : tensor<32xf32>
    %v1209 = stablehlo.subtract %bt8, %v1208 : tensor<32xf32>
    %v1210 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %v1211 = stablehlo.multiply %v1210, %bt8v : tensor<32xf32>
    %v1212 = stablehlo.add %v1211, %v1201 : tensor<32xf32>
    %v1213 = stablehlo.dot_general %v247, %v276, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x128xf32>, tensor<128x512xf32>) -> tensor<128x512xf32>
    %v1214 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v1215 = stablehlo.multiply %v1214, %W9v : tensor<128x512xf32>
    %v1216 = stablehlo.add %v1215, %v1213 : tensor<128x512xf32>
    %v1217 = stablehlo.multiply %v1214, %v1216 : tensor<128x512xf32>
    %v1218 = stablehlo.add %v1217, %v1213 : tensor<128x512xf32>
    %v1219 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v1220 = stablehlo.multiply %v1219, %v1218 : tensor<128x512xf32>
    %v1221 = stablehlo.subtract %W9, %v1220 : tensor<128x512xf32>
    %v1222 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %v1223 = stablehlo.multiply %v1222, %W9v : tensor<128x512xf32>
    %v1224 = stablehlo.add %v1223, %v1213 : tensor<128x512xf32>
    %v1225 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1226 = stablehlo.reduce(%v276 init: %v1225) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v1227 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1228 = stablehlo.multiply %v1227, %b9v : tensor<512xf32>
    %v1229 = stablehlo.add %v1228, %v1226 : tensor<512xf32>
    %v1230 = stablehlo.multiply %v1227, %v1229 : tensor<512xf32>
    %v1231 = stablehlo.add %v1230, %v1226 : tensor<512xf32>
    %v1232 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1233 = stablehlo.multiply %v1232, %v1231 : tensor<512xf32>
    %v1234 = stablehlo.subtract %b9, %v1233 : tensor<512xf32>
    %v1235 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1236 = stablehlo.multiply %v1235, %b9v : tensor<512xf32>
    %v1237 = stablehlo.add %v1236, %v1226 : tensor<512xf32>
    %v1238 = stablehlo.dot_general %v252, %v272, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<512x512xf32>
    %v1239 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1240 = stablehlo.multiply %v1239, %Wav : tensor<512x512xf32>
    %v1241 = stablehlo.add %v1240, %v1238 : tensor<512x512xf32>
    %v1242 = stablehlo.multiply %v1239, %v1241 : tensor<512x512xf32>
    %v1243 = stablehlo.add %v1242, %v1238 : tensor<512x512xf32>
    %v1244 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1245 = stablehlo.multiply %v1244, %v1243 : tensor<512x512xf32>
    %v1246 = stablehlo.subtract %Wa, %v1245 : tensor<512x512xf32>
    %v1247 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %v1248 = stablehlo.multiply %v1247, %Wav : tensor<512x512xf32>
    %v1249 = stablehlo.add %v1248, %v1238 : tensor<512x512xf32>
    %v1250 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1251 = stablehlo.reduce(%v272 init: %v1250) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %v1252 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1253 = stablehlo.multiply %v1252, %bav : tensor<512xf32>
    %v1254 = stablehlo.add %v1253, %v1251 : tensor<512xf32>
    %v1255 = stablehlo.multiply %v1252, %v1254 : tensor<512xf32>
    %v1256 = stablehlo.add %v1255, %v1251 : tensor<512xf32>
    %v1257 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1258 = stablehlo.multiply %v1257, %v1256 : tensor<512xf32>
    %v1259 = stablehlo.subtract %ba, %v1258 : tensor<512xf32>
    %v1260 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v1261 = stablehlo.multiply %v1260, %bav : tensor<512xf32>
    %v1262 = stablehlo.add %v1261, %v1251 : tensor<512xf32>
    %v1263 = stablehlo.dot_general %v257, %v268, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<128x10xf32>) -> tensor<512x10xf32>
    %v1264 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1265 = stablehlo.multiply %v1264, %Wbv : tensor<512x10xf32>
    %v1266 = stablehlo.add %v1265, %v1263 : tensor<512x10xf32>
    %v1267 = stablehlo.multiply %v1264, %v1266 : tensor<512x10xf32>
    %v1268 = stablehlo.add %v1267, %v1263 : tensor<512x10xf32>
    %v1269 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1270 = stablehlo.multiply %v1269, %v1268 : tensor<512x10xf32>
    %v1271 = stablehlo.subtract %Wb, %v1270 : tensor<512x10xf32>
    %v1272 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v1273 = stablehlo.multiply %v1272, %Wbv : tensor<512x10xf32>
    %v1274 = stablehlo.add %v1273, %v1263 : tensor<512x10xf32>
    %v1275 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1276 = stablehlo.reduce(%v268 init: %v1275) applies stablehlo.add across dimensions = [0] : (tensor<128x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1277 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1278 = stablehlo.multiply %v1277, %bbv : tensor<10xf32>
    %v1279 = stablehlo.add %v1278, %v1276 : tensor<10xf32>
    %v1280 = stablehlo.multiply %v1277, %v1279 : tensor<10xf32>
    %v1281 = stablehlo.add %v1280, %v1276 : tensor<10xf32>
    %v1282 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1283 = stablehlo.multiply %v1282, %v1281 : tensor<10xf32>
    %v1284 = stablehlo.subtract %bb, %v1283 : tensor<10xf32>
    %v1285 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v1286 = stablehlo.multiply %v1285, %bbv : tensor<10xf32>
    %v1287 = stablehlo.add %v1286, %v1276 : tensor<10xf32>
    return %v634, %v648, %v677, %v691, %v708, %v722, %v751, %v765, %v782, %v796, %v825, %v839, %v856, %v870, %v899, %v913, %v930, %v944, %v973, %v987, %v1004, %v1018, %v1047, %v1061, %v1078, %v1092, %v1121, %v1135, %v1152, %v1166, %v1195, %v1209, %v1221, %v1234, %v1246, %v1259, %v1271, %v1284, %W1m, %cb1m, %g1m, %bt1m, %W2m, %cb2m, %g2m, %bt2m, %W3m, %cb3m, %g3m, %bt3m, %W4m, %cb4m, %g4m, %bt4m, %W5m, %cb5m, %g5m, %bt5m, %W6m, %cb6m, %g6m, %bt6m, %W7m, %cb7m, %g7m, %bt7m, %W8m, %cb8m, %g8m, %bt8m, %W9m, %b9m, %Wam, %bam, %Wbm, %bbm, %v637, %v651, %v680, %v694, %v711, %v725, %v754, %v768, %v785, %v799, %v828, %v842, %v859, %v873, %v902, %v916, %v933, %v947, %v976, %v990, %v1007, %v1021, %v1050, %v1064, %v1081, %v1095, %v1124, %v1138, %v1155, %v1169, %v1198, %v1212, %v1224, %v1237, %v1249, %v1262, %v1274, %v1287, %loss, %bc1, %bc2 : tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x32x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<128x512xf32>, tensor<512xf32>, tensor<512x512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
