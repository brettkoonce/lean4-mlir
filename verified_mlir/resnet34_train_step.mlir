module @m {
  func.func @resnet34_train_step(%x: tensor<32x150528xf32>, %sW: tensor<64x3x7x7xf32>, %sg: tensor<64xf32>, %sbt: tensor<64xf32>, %s1b0W1: tensor<64x64x3x3xf32>, %s1b0g1: tensor<64xf32>, %s1b0bt1: tensor<64xf32>, %s1b0W2: tensor<64x64x3x3xf32>, %s1b0g2: tensor<64xf32>, %s1b0bt2: tensor<64xf32>, %s1b1W1: tensor<64x64x3x3xf32>, %s1b1g1: tensor<64xf32>, %s1b1bt1: tensor<64xf32>, %s1b1W2: tensor<64x64x3x3xf32>, %s1b1g2: tensor<64xf32>, %s1b1bt2: tensor<64xf32>, %s1b2W1: tensor<64x64x3x3xf32>, %s1b2g1: tensor<64xf32>, %s1b2bt1: tensor<64xf32>, %s1b2W2: tensor<64x64x3x3xf32>, %s1b2g2: tensor<64xf32>, %s1b2bt2: tensor<64xf32>, %d2W1: tensor<128x64x3x3xf32>, %d2g1: tensor<128xf32>, %d2bt1: tensor<128xf32>, %d2W2: tensor<128x128x3x3xf32>, %d2g2: tensor<128xf32>, %d2bt2: tensor<128xf32>, %d2Wp: tensor<128x64x1x1xf32>, %d2gp: tensor<128xf32>, %d2btp: tensor<128xf32>, %s2b0W1: tensor<128x128x3x3xf32>, %s2b0g1: tensor<128xf32>, %s2b0bt1: tensor<128xf32>, %s2b0W2: tensor<128x128x3x3xf32>, %s2b0g2: tensor<128xf32>, %s2b0bt2: tensor<128xf32>, %s2b1W1: tensor<128x128x3x3xf32>, %s2b1g1: tensor<128xf32>, %s2b1bt1: tensor<128xf32>, %s2b1W2: tensor<128x128x3x3xf32>, %s2b1g2: tensor<128xf32>, %s2b1bt2: tensor<128xf32>, %s2b2W1: tensor<128x128x3x3xf32>, %s2b2g1: tensor<128xf32>, %s2b2bt1: tensor<128xf32>, %s2b2W2: tensor<128x128x3x3xf32>, %s2b2g2: tensor<128xf32>, %s2b2bt2: tensor<128xf32>, %d3W1: tensor<256x128x3x3xf32>, %d3g1: tensor<256xf32>, %d3bt1: tensor<256xf32>, %d3W2: tensor<256x256x3x3xf32>, %d3g2: tensor<256xf32>, %d3bt2: tensor<256xf32>, %d3Wp: tensor<256x128x1x1xf32>, %d3gp: tensor<256xf32>, %d3btp: tensor<256xf32>, %s3b0W1: tensor<256x256x3x3xf32>, %s3b0g1: tensor<256xf32>, %s3b0bt1: tensor<256xf32>, %s3b0W2: tensor<256x256x3x3xf32>, %s3b0g2: tensor<256xf32>, %s3b0bt2: tensor<256xf32>, %s3b1W1: tensor<256x256x3x3xf32>, %s3b1g1: tensor<256xf32>, %s3b1bt1: tensor<256xf32>, %s3b1W2: tensor<256x256x3x3xf32>, %s3b1g2: tensor<256xf32>, %s3b1bt2: tensor<256xf32>, %s3b2W1: tensor<256x256x3x3xf32>, %s3b2g1: tensor<256xf32>, %s3b2bt1: tensor<256xf32>, %s3b2W2: tensor<256x256x3x3xf32>, %s3b2g2: tensor<256xf32>, %s3b2bt2: tensor<256xf32>, %s3b3W1: tensor<256x256x3x3xf32>, %s3b3g1: tensor<256xf32>, %s3b3bt1: tensor<256xf32>, %s3b3W2: tensor<256x256x3x3xf32>, %s3b3g2: tensor<256xf32>, %s3b3bt2: tensor<256xf32>, %s3b4W1: tensor<256x256x3x3xf32>, %s3b4g1: tensor<256xf32>, %s3b4bt1: tensor<256xf32>, %s3b4W2: tensor<256x256x3x3xf32>, %s3b4g2: tensor<256xf32>, %s3b4bt2: tensor<256xf32>, %d4W1: tensor<512x256x3x3xf32>, %d4g1: tensor<512xf32>, %d4bt1: tensor<512xf32>, %d4W2: tensor<512x512x3x3xf32>, %d4g2: tensor<512xf32>, %d4bt2: tensor<512xf32>, %d4Wp: tensor<512x256x1x1xf32>, %d4gp: tensor<512xf32>, %d4btp: tensor<512xf32>, %s4b0W1: tensor<512x512x3x3xf32>, %s4b0g1: tensor<512xf32>, %s4b0bt1: tensor<512xf32>, %s4b0W2: tensor<512x512x3x3xf32>, %s4b0g2: tensor<512xf32>, %s4b0bt2: tensor<512xf32>, %s4b1W1: tensor<512x512x3x3xf32>, %s4b1g1: tensor<512xf32>, %s4b1bt1: tensor<512xf32>, %s4b1W2: tensor<512x512x3x3xf32>, %s4b1g2: tensor<512xf32>, %s4b1bt2: tensor<512xf32>, %Wd: tensor<512x10xf32>, %bd: tensor<10xf32>, %onehot: tensor<32x10xf32>) -> (tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>) {
    // ── ResNet-34 train step: every line is pretty(verified AST node) ──
    // §2l step B: the conv biases are gone from the signature (BN removes them; He et al.'s
    // `.convBn` has none). The proven conv ops still take a bias operand, so it is bound to a
    // zero constant here — same op, `bias = 0`, and `x + 0.0` is exact.
    %zb64 = stablehlo.constant dense<0.0> : tensor<64xf32>
    %zb128 = stablehlo.constant dense<0.0> : tensor<128xf32>
    %zb256 = stablehlo.constant dense<0.0> : tensor<256xf32>
    %zb512 = stablehlo.constant dense<0.0> : tensor<512xf32>
    %v0 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v1 = stablehlo.convolution(%v0, %sW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<32x64x112x112xf32>
    %v2 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<32x64x112x112xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v6 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7 = stablehlo.constant dense<12544.0> : tensor<32x64x112x112xf32>
    %v8 = stablehlo.constant dense<1.0e-05> : tensor<32x64x112x112xf32>
    %v9 = stablehlo.reduce(%v5 init: %v6) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v10 = stablehlo.broadcast_in_dim %v9, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v11 = stablehlo.divide %v10, %v7 : tensor<32x64x112x112xf32>
    %v12 = stablehlo.subtract %v5, %v11 : tensor<32x64x112x112xf32>
    %v13 = stablehlo.multiply %v12, %v12 : tensor<32x64x112x112xf32>
    %v14 = stablehlo.reduce(%v13 init: %v6) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v15 = stablehlo.broadcast_in_dim %v14, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v16 = stablehlo.divide %v15, %v7 : tensor<32x64x112x112xf32>
    %v17 = stablehlo.add %v16, %v8 : tensor<32x64x112x112xf32>
    %v18 = stablehlo.rsqrt %v17 : tensor<32x64x112x112xf32>
    %v19 = stablehlo.multiply %v12, %v18 : tensor<32x64x112x112xf32>
    %v20 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v21 = stablehlo.broadcast_in_dim %sbt, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v22 = stablehlo.multiply %v19, %v20 : tensor<32x64x112x112xf32>
    %v23 = stablehlo.add %v22, %v21 : tensor<32x64x112x112xf32>
    %v24 = stablehlo.reshape %v23 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v25 = stablehlo.reshape %v24 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v26 = stablehlo.constant dense<0.0> : tensor<32x64x112x112xf32>
    %v27 = stablehlo.maximum %v25, %v26 : tensor<32x64x112x112xf32>
    %v28 = stablehlo.reshape %v27 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v29 = stablehlo.reshape %v28 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v30 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v31 = "stablehlo.reduce_window"(%v29, %v30) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>} : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64x56x56xf32>
    %v32 = stablehlo.reshape %v31 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v33 = stablehlo.reshape %v32 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v34 = stablehlo.convolution(%v33, %s1b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v35 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v36 = stablehlo.add %v34, %v35 : tensor<32x64x56x56xf32>
    %v37 = stablehlo.reshape %v36 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v38 = stablehlo.reshape %v37 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v39 = stablehlo.constant dense<0.0> : tensor<f32>
    %v40 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v41 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v42 = stablehlo.reduce(%v38 init: %v39) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v43 = stablehlo.broadcast_in_dim %v42, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v44 = stablehlo.divide %v43, %v40 : tensor<32x64x56x56xf32>
    %v45 = stablehlo.subtract %v38, %v44 : tensor<32x64x56x56xf32>
    %v46 = stablehlo.multiply %v45, %v45 : tensor<32x64x56x56xf32>
    %v47 = stablehlo.reduce(%v46 init: %v39) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v48 = stablehlo.broadcast_in_dim %v47, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v49 = stablehlo.divide %v48, %v40 : tensor<32x64x56x56xf32>
    %v50 = stablehlo.add %v49, %v41 : tensor<32x64x56x56xf32>
    %v51 = stablehlo.rsqrt %v50 : tensor<32x64x56x56xf32>
    %v52 = stablehlo.multiply %v45, %v51 : tensor<32x64x56x56xf32>
    %v53 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v54 = stablehlo.broadcast_in_dim %s1b0bt1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v55 = stablehlo.multiply %v52, %v53 : tensor<32x64x56x56xf32>
    %v56 = stablehlo.add %v55, %v54 : tensor<32x64x56x56xf32>
    %v57 = stablehlo.reshape %v56 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v59 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v60 = stablehlo.maximum %v58, %v59 : tensor<32x64x56x56xf32>
    %v61 = stablehlo.reshape %v60 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v62 = stablehlo.reshape %v61 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v63 = stablehlo.convolution(%v62, %s1b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v64 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v65 = stablehlo.add %v63, %v64 : tensor<32x64x56x56xf32>
    %v66 = stablehlo.reshape %v65 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v67 = stablehlo.reshape %v66 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v68 = stablehlo.constant dense<0.0> : tensor<f32>
    %v69 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v70 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v71 = stablehlo.reduce(%v67 init: %v68) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v72 = stablehlo.broadcast_in_dim %v71, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v73 = stablehlo.divide %v72, %v69 : tensor<32x64x56x56xf32>
    %v74 = stablehlo.subtract %v67, %v73 : tensor<32x64x56x56xf32>
    %v75 = stablehlo.multiply %v74, %v74 : tensor<32x64x56x56xf32>
    %v76 = stablehlo.reduce(%v75 init: %v68) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v77 = stablehlo.broadcast_in_dim %v76, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v78 = stablehlo.divide %v77, %v69 : tensor<32x64x56x56xf32>
    %v79 = stablehlo.add %v78, %v70 : tensor<32x64x56x56xf32>
    %v80 = stablehlo.rsqrt %v79 : tensor<32x64x56x56xf32>
    %v81 = stablehlo.multiply %v74, %v80 : tensor<32x64x56x56xf32>
    %v82 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v83 = stablehlo.broadcast_in_dim %s1b0bt2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v84 = stablehlo.multiply %v81, %v82 : tensor<32x64x56x56xf32>
    %v85 = stablehlo.add %v84, %v83 : tensor<32x64x56x56xf32>
    %v86 = stablehlo.reshape %v85 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v87 = stablehlo.reshape %v86 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v88 = stablehlo.reshape %v32 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v89 = stablehlo.add %v87, %v88 : tensor<32x64x56x56xf32>
    %v90 = stablehlo.reshape %v89 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v91 = stablehlo.reshape %v90 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v92 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v93 = stablehlo.maximum %v91, %v92 : tensor<32x64x56x56xf32>
    %v94 = stablehlo.reshape %v93 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v95 = stablehlo.reshape %v94 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v96 = stablehlo.convolution(%v95, %s1b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v97 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v98 = stablehlo.add %v96, %v97 : tensor<32x64x56x56xf32>
    %v99 = stablehlo.reshape %v98 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v100 = stablehlo.reshape %v99 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v101 = stablehlo.constant dense<0.0> : tensor<f32>
    %v102 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v103 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v104 = stablehlo.reduce(%v100 init: %v101) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v105 = stablehlo.broadcast_in_dim %v104, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v106 = stablehlo.divide %v105, %v102 : tensor<32x64x56x56xf32>
    %v107 = stablehlo.subtract %v100, %v106 : tensor<32x64x56x56xf32>
    %v108 = stablehlo.multiply %v107, %v107 : tensor<32x64x56x56xf32>
    %v109 = stablehlo.reduce(%v108 init: %v101) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v110 = stablehlo.broadcast_in_dim %v109, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v111 = stablehlo.divide %v110, %v102 : tensor<32x64x56x56xf32>
    %v112 = stablehlo.add %v111, %v103 : tensor<32x64x56x56xf32>
    %v113 = stablehlo.rsqrt %v112 : tensor<32x64x56x56xf32>
    %v114 = stablehlo.multiply %v107, %v113 : tensor<32x64x56x56xf32>
    %v115 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v116 = stablehlo.broadcast_in_dim %s1b1bt1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v117 = stablehlo.multiply %v114, %v115 : tensor<32x64x56x56xf32>
    %v118 = stablehlo.add %v117, %v116 : tensor<32x64x56x56xf32>
    %v119 = stablehlo.reshape %v118 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v120 = stablehlo.reshape %v119 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v121 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v122 = stablehlo.maximum %v120, %v121 : tensor<32x64x56x56xf32>
    %v123 = stablehlo.reshape %v122 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v124 = stablehlo.reshape %v123 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v125 = stablehlo.convolution(%v124, %s1b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v126 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v127 = stablehlo.add %v125, %v126 : tensor<32x64x56x56xf32>
    %v128 = stablehlo.reshape %v127 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v129 = stablehlo.reshape %v128 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v130 = stablehlo.constant dense<0.0> : tensor<f32>
    %v131 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v132 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v133 = stablehlo.reduce(%v129 init: %v130) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v134 = stablehlo.broadcast_in_dim %v133, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v135 = stablehlo.divide %v134, %v131 : tensor<32x64x56x56xf32>
    %v136 = stablehlo.subtract %v129, %v135 : tensor<32x64x56x56xf32>
    %v137 = stablehlo.multiply %v136, %v136 : tensor<32x64x56x56xf32>
    %v138 = stablehlo.reduce(%v137 init: %v130) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v139 = stablehlo.broadcast_in_dim %v138, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v140 = stablehlo.divide %v139, %v131 : tensor<32x64x56x56xf32>
    %v141 = stablehlo.add %v140, %v132 : tensor<32x64x56x56xf32>
    %v142 = stablehlo.rsqrt %v141 : tensor<32x64x56x56xf32>
    %v143 = stablehlo.multiply %v136, %v142 : tensor<32x64x56x56xf32>
    %v144 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v145 = stablehlo.broadcast_in_dim %s1b1bt2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v146 = stablehlo.multiply %v143, %v144 : tensor<32x64x56x56xf32>
    %v147 = stablehlo.add %v146, %v145 : tensor<32x64x56x56xf32>
    %v148 = stablehlo.reshape %v147 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v149 = stablehlo.reshape %v148 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v150 = stablehlo.reshape %v94 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v151 = stablehlo.add %v149, %v150 : tensor<32x64x56x56xf32>
    %v152 = stablehlo.reshape %v151 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v153 = stablehlo.reshape %v152 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v154 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v155 = stablehlo.maximum %v153, %v154 : tensor<32x64x56x56xf32>
    %v156 = stablehlo.reshape %v155 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v157 = stablehlo.reshape %v156 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v158 = stablehlo.convolution(%v157, %s1b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v159 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v160 = stablehlo.add %v158, %v159 : tensor<32x64x56x56xf32>
    %v161 = stablehlo.reshape %v160 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v162 = stablehlo.reshape %v161 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v163 = stablehlo.constant dense<0.0> : tensor<f32>
    %v164 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v165 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v166 = stablehlo.reduce(%v162 init: %v163) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v167 = stablehlo.broadcast_in_dim %v166, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v168 = stablehlo.divide %v167, %v164 : tensor<32x64x56x56xf32>
    %v169 = stablehlo.subtract %v162, %v168 : tensor<32x64x56x56xf32>
    %v170 = stablehlo.multiply %v169, %v169 : tensor<32x64x56x56xf32>
    %v171 = stablehlo.reduce(%v170 init: %v163) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v172 = stablehlo.broadcast_in_dim %v171, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v173 = stablehlo.divide %v172, %v164 : tensor<32x64x56x56xf32>
    %v174 = stablehlo.add %v173, %v165 : tensor<32x64x56x56xf32>
    %v175 = stablehlo.rsqrt %v174 : tensor<32x64x56x56xf32>
    %v176 = stablehlo.multiply %v169, %v175 : tensor<32x64x56x56xf32>
    %v177 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v178 = stablehlo.broadcast_in_dim %s1b2bt1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v179 = stablehlo.multiply %v176, %v177 : tensor<32x64x56x56xf32>
    %v180 = stablehlo.add %v179, %v178 : tensor<32x64x56x56xf32>
    %v181 = stablehlo.reshape %v180 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v182 = stablehlo.reshape %v181 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v183 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v184 = stablehlo.maximum %v182, %v183 : tensor<32x64x56x56xf32>
    %v185 = stablehlo.reshape %v184 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v186 = stablehlo.reshape %v185 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v187 = stablehlo.convolution(%v186, %s1b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v188 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v189 = stablehlo.add %v187, %v188 : tensor<32x64x56x56xf32>
    %v190 = stablehlo.reshape %v189 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v191 = stablehlo.reshape %v190 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v192 = stablehlo.constant dense<0.0> : tensor<f32>
    %v193 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v194 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v195 = stablehlo.reduce(%v191 init: %v192) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v196 = stablehlo.broadcast_in_dim %v195, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v197 = stablehlo.divide %v196, %v193 : tensor<32x64x56x56xf32>
    %v198 = stablehlo.subtract %v191, %v197 : tensor<32x64x56x56xf32>
    %v199 = stablehlo.multiply %v198, %v198 : tensor<32x64x56x56xf32>
    %v200 = stablehlo.reduce(%v199 init: %v192) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v201 = stablehlo.broadcast_in_dim %v200, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v202 = stablehlo.divide %v201, %v193 : tensor<32x64x56x56xf32>
    %v203 = stablehlo.add %v202, %v194 : tensor<32x64x56x56xf32>
    %v204 = stablehlo.rsqrt %v203 : tensor<32x64x56x56xf32>
    %v205 = stablehlo.multiply %v198, %v204 : tensor<32x64x56x56xf32>
    %v206 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v207 = stablehlo.broadcast_in_dim %s1b2bt2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v208 = stablehlo.multiply %v205, %v206 : tensor<32x64x56x56xf32>
    %v209 = stablehlo.add %v208, %v207 : tensor<32x64x56x56xf32>
    %v210 = stablehlo.reshape %v209 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v211 = stablehlo.reshape %v210 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v212 = stablehlo.reshape %v156 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v213 = stablehlo.add %v211, %v212 : tensor<32x64x56x56xf32>
    %v214 = stablehlo.reshape %v213 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v215 = stablehlo.reshape %v214 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v216 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v217 = stablehlo.maximum %v215, %v216 : tensor<32x64x56x56xf32>
    %v218 = stablehlo.reshape %v217 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v219 = stablehlo.reshape %v218 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v220 = stablehlo.convolution(%v219, %d2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<128x64x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v221 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v222 = stablehlo.add %v220, %v221 : tensor<32x128x28x28xf32>
    %v223 = stablehlo.reshape %v222 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v224 = stablehlo.reshape %v223 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v225 = stablehlo.constant dense<0.0> : tensor<f32>
    %v226 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v227 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v228 = stablehlo.reduce(%v224 init: %v225) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v229 = stablehlo.broadcast_in_dim %v228, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v230 = stablehlo.divide %v229, %v226 : tensor<32x128x28x28xf32>
    %v231 = stablehlo.subtract %v224, %v230 : tensor<32x128x28x28xf32>
    %v232 = stablehlo.multiply %v231, %v231 : tensor<32x128x28x28xf32>
    %v233 = stablehlo.reduce(%v232 init: %v225) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v234 = stablehlo.broadcast_in_dim %v233, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v235 = stablehlo.divide %v234, %v226 : tensor<32x128x28x28xf32>
    %v236 = stablehlo.add %v235, %v227 : tensor<32x128x28x28xf32>
    %v237 = stablehlo.rsqrt %v236 : tensor<32x128x28x28xf32>
    %v238 = stablehlo.multiply %v231, %v237 : tensor<32x128x28x28xf32>
    %v239 = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v240 = stablehlo.broadcast_in_dim %d2bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v241 = stablehlo.multiply %v238, %v239 : tensor<32x128x28x28xf32>
    %v242 = stablehlo.add %v241, %v240 : tensor<32x128x28x28xf32>
    %v243 = stablehlo.reshape %v242 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v244 = stablehlo.reshape %v243 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v245 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v246 = stablehlo.maximum %v244, %v245 : tensor<32x128x28x28xf32>
    %v247 = stablehlo.reshape %v246 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v248 = stablehlo.reshape %v247 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v249 = stablehlo.convolution(%v248, %d2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v250 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v251 = stablehlo.add %v249, %v250 : tensor<32x128x28x28xf32>
    %v252 = stablehlo.reshape %v251 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v253 = stablehlo.reshape %v252 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v254 = stablehlo.constant dense<0.0> : tensor<f32>
    %v255 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v256 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v257 = stablehlo.reduce(%v253 init: %v254) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v258 = stablehlo.broadcast_in_dim %v257, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v259 = stablehlo.divide %v258, %v255 : tensor<32x128x28x28xf32>
    %v260 = stablehlo.subtract %v253, %v259 : tensor<32x128x28x28xf32>
    %v261 = stablehlo.multiply %v260, %v260 : tensor<32x128x28x28xf32>
    %v262 = stablehlo.reduce(%v261 init: %v254) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v263 = stablehlo.broadcast_in_dim %v262, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v264 = stablehlo.divide %v263, %v255 : tensor<32x128x28x28xf32>
    %v265 = stablehlo.add %v264, %v256 : tensor<32x128x28x28xf32>
    %v266 = stablehlo.rsqrt %v265 : tensor<32x128x28x28xf32>
    %v267 = stablehlo.multiply %v260, %v266 : tensor<32x128x28x28xf32>
    %v268 = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v269 = stablehlo.broadcast_in_dim %d2bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v270 = stablehlo.multiply %v267, %v268 : tensor<32x128x28x28xf32>
    %v271 = stablehlo.add %v270, %v269 : tensor<32x128x28x28xf32>
    %v272 = stablehlo.reshape %v271 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v273 = stablehlo.reshape %v218 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v274 = stablehlo.convolution(%v273, %d2Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<128x64x1x1xf32>) -> tensor<32x128x28x28xf32>
    %v275 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v276 = stablehlo.add %v274, %v275 : tensor<32x128x28x28xf32>
    %v277 = stablehlo.reshape %v276 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v278 = stablehlo.reshape %v277 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v279 = stablehlo.constant dense<0.0> : tensor<f32>
    %v280 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v281 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v282 = stablehlo.reduce(%v278 init: %v279) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v283 = stablehlo.broadcast_in_dim %v282, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v284 = stablehlo.divide %v283, %v280 : tensor<32x128x28x28xf32>
    %v285 = stablehlo.subtract %v278, %v284 : tensor<32x128x28x28xf32>
    %v286 = stablehlo.multiply %v285, %v285 : tensor<32x128x28x28xf32>
    %v287 = stablehlo.reduce(%v286 init: %v279) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v288 = stablehlo.broadcast_in_dim %v287, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v289 = stablehlo.divide %v288, %v280 : tensor<32x128x28x28xf32>
    %v290 = stablehlo.add %v289, %v281 : tensor<32x128x28x28xf32>
    %v291 = stablehlo.rsqrt %v290 : tensor<32x128x28x28xf32>
    %v292 = stablehlo.multiply %v285, %v291 : tensor<32x128x28x28xf32>
    %v293 = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v294 = stablehlo.broadcast_in_dim %d2btp, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v295 = stablehlo.multiply %v292, %v293 : tensor<32x128x28x28xf32>
    %v296 = stablehlo.add %v295, %v294 : tensor<32x128x28x28xf32>
    %v297 = stablehlo.reshape %v296 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v298 = stablehlo.reshape %v272 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v299 = stablehlo.reshape %v297 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v300 = stablehlo.add %v298, %v299 : tensor<32x128x28x28xf32>
    %v301 = stablehlo.reshape %v300 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v302 = stablehlo.reshape %v301 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v303 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v304 = stablehlo.maximum %v302, %v303 : tensor<32x128x28x28xf32>
    %v305 = stablehlo.reshape %v304 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v306 = stablehlo.reshape %v305 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v307 = stablehlo.convolution(%v306, %s2b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v308 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v309 = stablehlo.add %v307, %v308 : tensor<32x128x28x28xf32>
    %v310 = stablehlo.reshape %v309 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v311 = stablehlo.reshape %v310 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v312 = stablehlo.constant dense<0.0> : tensor<f32>
    %v313 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v314 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v315 = stablehlo.reduce(%v311 init: %v312) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v316 = stablehlo.broadcast_in_dim %v315, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v317 = stablehlo.divide %v316, %v313 : tensor<32x128x28x28xf32>
    %v318 = stablehlo.subtract %v311, %v317 : tensor<32x128x28x28xf32>
    %v319 = stablehlo.multiply %v318, %v318 : tensor<32x128x28x28xf32>
    %v320 = stablehlo.reduce(%v319 init: %v312) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v321 = stablehlo.broadcast_in_dim %v320, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v322 = stablehlo.divide %v321, %v313 : tensor<32x128x28x28xf32>
    %v323 = stablehlo.add %v322, %v314 : tensor<32x128x28x28xf32>
    %v324 = stablehlo.rsqrt %v323 : tensor<32x128x28x28xf32>
    %v325 = stablehlo.multiply %v318, %v324 : tensor<32x128x28x28xf32>
    %v326 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v327 = stablehlo.broadcast_in_dim %s2b0bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v328 = stablehlo.multiply %v325, %v326 : tensor<32x128x28x28xf32>
    %v329 = stablehlo.add %v328, %v327 : tensor<32x128x28x28xf32>
    %v330 = stablehlo.reshape %v329 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v331 = stablehlo.reshape %v330 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v332 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v333 = stablehlo.maximum %v331, %v332 : tensor<32x128x28x28xf32>
    %v334 = stablehlo.reshape %v333 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v335 = stablehlo.reshape %v334 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v336 = stablehlo.convolution(%v335, %s2b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v337 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v338 = stablehlo.add %v336, %v337 : tensor<32x128x28x28xf32>
    %v339 = stablehlo.reshape %v338 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v340 = stablehlo.reshape %v339 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v341 = stablehlo.constant dense<0.0> : tensor<f32>
    %v342 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v343 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v344 = stablehlo.reduce(%v340 init: %v341) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v345 = stablehlo.broadcast_in_dim %v344, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v346 = stablehlo.divide %v345, %v342 : tensor<32x128x28x28xf32>
    %v347 = stablehlo.subtract %v340, %v346 : tensor<32x128x28x28xf32>
    %v348 = stablehlo.multiply %v347, %v347 : tensor<32x128x28x28xf32>
    %v349 = stablehlo.reduce(%v348 init: %v341) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v350 = stablehlo.broadcast_in_dim %v349, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v351 = stablehlo.divide %v350, %v342 : tensor<32x128x28x28xf32>
    %v352 = stablehlo.add %v351, %v343 : tensor<32x128x28x28xf32>
    %v353 = stablehlo.rsqrt %v352 : tensor<32x128x28x28xf32>
    %v354 = stablehlo.multiply %v347, %v353 : tensor<32x128x28x28xf32>
    %v355 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v356 = stablehlo.broadcast_in_dim %s2b0bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v357 = stablehlo.multiply %v354, %v355 : tensor<32x128x28x28xf32>
    %v358 = stablehlo.add %v357, %v356 : tensor<32x128x28x28xf32>
    %v359 = stablehlo.reshape %v358 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v360 = stablehlo.reshape %v359 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v361 = stablehlo.reshape %v305 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v362 = stablehlo.add %v360, %v361 : tensor<32x128x28x28xf32>
    %v363 = stablehlo.reshape %v362 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v364 = stablehlo.reshape %v363 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v365 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v366 = stablehlo.maximum %v364, %v365 : tensor<32x128x28x28xf32>
    %v367 = stablehlo.reshape %v366 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v368 = stablehlo.reshape %v367 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v369 = stablehlo.convolution(%v368, %s2b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v370 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v371 = stablehlo.add %v369, %v370 : tensor<32x128x28x28xf32>
    %v372 = stablehlo.reshape %v371 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v373 = stablehlo.reshape %v372 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v374 = stablehlo.constant dense<0.0> : tensor<f32>
    %v375 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v376 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v377 = stablehlo.reduce(%v373 init: %v374) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v378 = stablehlo.broadcast_in_dim %v377, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v379 = stablehlo.divide %v378, %v375 : tensor<32x128x28x28xf32>
    %v380 = stablehlo.subtract %v373, %v379 : tensor<32x128x28x28xf32>
    %v381 = stablehlo.multiply %v380, %v380 : tensor<32x128x28x28xf32>
    %v382 = stablehlo.reduce(%v381 init: %v374) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v383 = stablehlo.broadcast_in_dim %v382, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v384 = stablehlo.divide %v383, %v375 : tensor<32x128x28x28xf32>
    %v385 = stablehlo.add %v384, %v376 : tensor<32x128x28x28xf32>
    %v386 = stablehlo.rsqrt %v385 : tensor<32x128x28x28xf32>
    %v387 = stablehlo.multiply %v380, %v386 : tensor<32x128x28x28xf32>
    %v388 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v389 = stablehlo.broadcast_in_dim %s2b1bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v390 = stablehlo.multiply %v387, %v388 : tensor<32x128x28x28xf32>
    %v391 = stablehlo.add %v390, %v389 : tensor<32x128x28x28xf32>
    %v392 = stablehlo.reshape %v391 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v393 = stablehlo.reshape %v392 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v394 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v395 = stablehlo.maximum %v393, %v394 : tensor<32x128x28x28xf32>
    %v396 = stablehlo.reshape %v395 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v397 = stablehlo.reshape %v396 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v398 = stablehlo.convolution(%v397, %s2b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v399 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v400 = stablehlo.add %v398, %v399 : tensor<32x128x28x28xf32>
    %v401 = stablehlo.reshape %v400 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v402 = stablehlo.reshape %v401 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v403 = stablehlo.constant dense<0.0> : tensor<f32>
    %v404 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v405 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v406 = stablehlo.reduce(%v402 init: %v403) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v407 = stablehlo.broadcast_in_dim %v406, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v408 = stablehlo.divide %v407, %v404 : tensor<32x128x28x28xf32>
    %v409 = stablehlo.subtract %v402, %v408 : tensor<32x128x28x28xf32>
    %v410 = stablehlo.multiply %v409, %v409 : tensor<32x128x28x28xf32>
    %v411 = stablehlo.reduce(%v410 init: %v403) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v412 = stablehlo.broadcast_in_dim %v411, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v413 = stablehlo.divide %v412, %v404 : tensor<32x128x28x28xf32>
    %v414 = stablehlo.add %v413, %v405 : tensor<32x128x28x28xf32>
    %v415 = stablehlo.rsqrt %v414 : tensor<32x128x28x28xf32>
    %v416 = stablehlo.multiply %v409, %v415 : tensor<32x128x28x28xf32>
    %v417 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v418 = stablehlo.broadcast_in_dim %s2b1bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v419 = stablehlo.multiply %v416, %v417 : tensor<32x128x28x28xf32>
    %v420 = stablehlo.add %v419, %v418 : tensor<32x128x28x28xf32>
    %v421 = stablehlo.reshape %v420 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v422 = stablehlo.reshape %v421 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v423 = stablehlo.reshape %v367 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v424 = stablehlo.add %v422, %v423 : tensor<32x128x28x28xf32>
    %v425 = stablehlo.reshape %v424 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v426 = stablehlo.reshape %v425 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v427 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v428 = stablehlo.maximum %v426, %v427 : tensor<32x128x28x28xf32>
    %v429 = stablehlo.reshape %v428 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v430 = stablehlo.reshape %v429 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v431 = stablehlo.convolution(%v430, %s2b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v432 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v433 = stablehlo.add %v431, %v432 : tensor<32x128x28x28xf32>
    %v434 = stablehlo.reshape %v433 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v435 = stablehlo.reshape %v434 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v436 = stablehlo.constant dense<0.0> : tensor<f32>
    %v437 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v438 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v439 = stablehlo.reduce(%v435 init: %v436) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v440 = stablehlo.broadcast_in_dim %v439, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v441 = stablehlo.divide %v440, %v437 : tensor<32x128x28x28xf32>
    %v442 = stablehlo.subtract %v435, %v441 : tensor<32x128x28x28xf32>
    %v443 = stablehlo.multiply %v442, %v442 : tensor<32x128x28x28xf32>
    %v444 = stablehlo.reduce(%v443 init: %v436) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v445 = stablehlo.broadcast_in_dim %v444, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v446 = stablehlo.divide %v445, %v437 : tensor<32x128x28x28xf32>
    %v447 = stablehlo.add %v446, %v438 : tensor<32x128x28x28xf32>
    %v448 = stablehlo.rsqrt %v447 : tensor<32x128x28x28xf32>
    %v449 = stablehlo.multiply %v442, %v448 : tensor<32x128x28x28xf32>
    %v450 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v451 = stablehlo.broadcast_in_dim %s2b2bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v452 = stablehlo.multiply %v449, %v450 : tensor<32x128x28x28xf32>
    %v453 = stablehlo.add %v452, %v451 : tensor<32x128x28x28xf32>
    %v454 = stablehlo.reshape %v453 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v455 = stablehlo.reshape %v454 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v456 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v457 = stablehlo.maximum %v455, %v456 : tensor<32x128x28x28xf32>
    %v458 = stablehlo.reshape %v457 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v459 = stablehlo.reshape %v458 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v460 = stablehlo.convolution(%v459, %s2b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v461 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v462 = stablehlo.add %v460, %v461 : tensor<32x128x28x28xf32>
    %v463 = stablehlo.reshape %v462 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v464 = stablehlo.reshape %v463 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v465 = stablehlo.constant dense<0.0> : tensor<f32>
    %v466 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v467 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v468 = stablehlo.reduce(%v464 init: %v465) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v469 = stablehlo.broadcast_in_dim %v468, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v470 = stablehlo.divide %v469, %v466 : tensor<32x128x28x28xf32>
    %v471 = stablehlo.subtract %v464, %v470 : tensor<32x128x28x28xf32>
    %v472 = stablehlo.multiply %v471, %v471 : tensor<32x128x28x28xf32>
    %v473 = stablehlo.reduce(%v472 init: %v465) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v474 = stablehlo.broadcast_in_dim %v473, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v475 = stablehlo.divide %v474, %v466 : tensor<32x128x28x28xf32>
    %v476 = stablehlo.add %v475, %v467 : tensor<32x128x28x28xf32>
    %v477 = stablehlo.rsqrt %v476 : tensor<32x128x28x28xf32>
    %v478 = stablehlo.multiply %v471, %v477 : tensor<32x128x28x28xf32>
    %v479 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v480 = stablehlo.broadcast_in_dim %s2b2bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v481 = stablehlo.multiply %v478, %v479 : tensor<32x128x28x28xf32>
    %v482 = stablehlo.add %v481, %v480 : tensor<32x128x28x28xf32>
    %v483 = stablehlo.reshape %v482 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v484 = stablehlo.reshape %v483 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v485 = stablehlo.reshape %v429 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v486 = stablehlo.add %v484, %v485 : tensor<32x128x28x28xf32>
    %v487 = stablehlo.reshape %v486 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v488 = stablehlo.reshape %v487 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v489 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v490 = stablehlo.maximum %v488, %v489 : tensor<32x128x28x28xf32>
    %v491 = stablehlo.reshape %v490 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v492 = stablehlo.reshape %v491 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v493 = stablehlo.convolution(%v492, %d3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<256x128x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v494 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v495 = stablehlo.add %v493, %v494 : tensor<32x256x14x14xf32>
    %v496 = stablehlo.reshape %v495 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v497 = stablehlo.reshape %v496 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v498 = stablehlo.constant dense<0.0> : tensor<f32>
    %v499 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v500 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v501 = stablehlo.reduce(%v497 init: %v498) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v502 = stablehlo.broadcast_in_dim %v501, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v503 = stablehlo.divide %v502, %v499 : tensor<32x256x14x14xf32>
    %v504 = stablehlo.subtract %v497, %v503 : tensor<32x256x14x14xf32>
    %v505 = stablehlo.multiply %v504, %v504 : tensor<32x256x14x14xf32>
    %v506 = stablehlo.reduce(%v505 init: %v498) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v507 = stablehlo.broadcast_in_dim %v506, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v508 = stablehlo.divide %v507, %v499 : tensor<32x256x14x14xf32>
    %v509 = stablehlo.add %v508, %v500 : tensor<32x256x14x14xf32>
    %v510 = stablehlo.rsqrt %v509 : tensor<32x256x14x14xf32>
    %v511 = stablehlo.multiply %v504, %v510 : tensor<32x256x14x14xf32>
    %v512 = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v513 = stablehlo.broadcast_in_dim %d3bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v514 = stablehlo.multiply %v511, %v512 : tensor<32x256x14x14xf32>
    %v515 = stablehlo.add %v514, %v513 : tensor<32x256x14x14xf32>
    %v516 = stablehlo.reshape %v515 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v517 = stablehlo.reshape %v516 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v518 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v519 = stablehlo.maximum %v517, %v518 : tensor<32x256x14x14xf32>
    %v520 = stablehlo.reshape %v519 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v521 = stablehlo.reshape %v520 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v522 = stablehlo.convolution(%v521, %d3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v523 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v524 = stablehlo.add %v522, %v523 : tensor<32x256x14x14xf32>
    %v525 = stablehlo.reshape %v524 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v526 = stablehlo.reshape %v525 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v527 = stablehlo.constant dense<0.0> : tensor<f32>
    %v528 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v529 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v530 = stablehlo.reduce(%v526 init: %v527) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v531 = stablehlo.broadcast_in_dim %v530, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v532 = stablehlo.divide %v531, %v528 : tensor<32x256x14x14xf32>
    %v533 = stablehlo.subtract %v526, %v532 : tensor<32x256x14x14xf32>
    %v534 = stablehlo.multiply %v533, %v533 : tensor<32x256x14x14xf32>
    %v535 = stablehlo.reduce(%v534 init: %v527) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v536 = stablehlo.broadcast_in_dim %v535, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v537 = stablehlo.divide %v536, %v528 : tensor<32x256x14x14xf32>
    %v538 = stablehlo.add %v537, %v529 : tensor<32x256x14x14xf32>
    %v539 = stablehlo.rsqrt %v538 : tensor<32x256x14x14xf32>
    %v540 = stablehlo.multiply %v533, %v539 : tensor<32x256x14x14xf32>
    %v541 = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v542 = stablehlo.broadcast_in_dim %d3bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v543 = stablehlo.multiply %v540, %v541 : tensor<32x256x14x14xf32>
    %v544 = stablehlo.add %v543, %v542 : tensor<32x256x14x14xf32>
    %v545 = stablehlo.reshape %v544 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v546 = stablehlo.reshape %v491 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v547 = stablehlo.convolution(%v546, %d3Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<256x128x1x1xf32>) -> tensor<32x256x14x14xf32>
    %v548 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v549 = stablehlo.add %v547, %v548 : tensor<32x256x14x14xf32>
    %v550 = stablehlo.reshape %v549 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v551 = stablehlo.reshape %v550 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v552 = stablehlo.constant dense<0.0> : tensor<f32>
    %v553 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v554 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v555 = stablehlo.reduce(%v551 init: %v552) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v556 = stablehlo.broadcast_in_dim %v555, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v557 = stablehlo.divide %v556, %v553 : tensor<32x256x14x14xf32>
    %v558 = stablehlo.subtract %v551, %v557 : tensor<32x256x14x14xf32>
    %v559 = stablehlo.multiply %v558, %v558 : tensor<32x256x14x14xf32>
    %v560 = stablehlo.reduce(%v559 init: %v552) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v561 = stablehlo.broadcast_in_dim %v560, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v562 = stablehlo.divide %v561, %v553 : tensor<32x256x14x14xf32>
    %v563 = stablehlo.add %v562, %v554 : tensor<32x256x14x14xf32>
    %v564 = stablehlo.rsqrt %v563 : tensor<32x256x14x14xf32>
    %v565 = stablehlo.multiply %v558, %v564 : tensor<32x256x14x14xf32>
    %v566 = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v567 = stablehlo.broadcast_in_dim %d3btp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v568 = stablehlo.multiply %v565, %v566 : tensor<32x256x14x14xf32>
    %v569 = stablehlo.add %v568, %v567 : tensor<32x256x14x14xf32>
    %v570 = stablehlo.reshape %v569 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v571 = stablehlo.reshape %v545 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v572 = stablehlo.reshape %v570 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v573 = stablehlo.add %v571, %v572 : tensor<32x256x14x14xf32>
    %v574 = stablehlo.reshape %v573 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v575 = stablehlo.reshape %v574 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v576 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v577 = stablehlo.maximum %v575, %v576 : tensor<32x256x14x14xf32>
    %v578 = stablehlo.reshape %v577 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v579 = stablehlo.reshape %v578 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v580 = stablehlo.convolution(%v579, %s3b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v581 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v582 = stablehlo.add %v580, %v581 : tensor<32x256x14x14xf32>
    %v583 = stablehlo.reshape %v582 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v584 = stablehlo.reshape %v583 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v585 = stablehlo.constant dense<0.0> : tensor<f32>
    %v586 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v587 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v588 = stablehlo.reduce(%v584 init: %v585) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v589 = stablehlo.broadcast_in_dim %v588, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v590 = stablehlo.divide %v589, %v586 : tensor<32x256x14x14xf32>
    %v591 = stablehlo.subtract %v584, %v590 : tensor<32x256x14x14xf32>
    %v592 = stablehlo.multiply %v591, %v591 : tensor<32x256x14x14xf32>
    %v593 = stablehlo.reduce(%v592 init: %v585) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v594 = stablehlo.broadcast_in_dim %v593, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v595 = stablehlo.divide %v594, %v586 : tensor<32x256x14x14xf32>
    %v596 = stablehlo.add %v595, %v587 : tensor<32x256x14x14xf32>
    %v597 = stablehlo.rsqrt %v596 : tensor<32x256x14x14xf32>
    %v598 = stablehlo.multiply %v591, %v597 : tensor<32x256x14x14xf32>
    %v599 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v600 = stablehlo.broadcast_in_dim %s3b0bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v601 = stablehlo.multiply %v598, %v599 : tensor<32x256x14x14xf32>
    %v602 = stablehlo.add %v601, %v600 : tensor<32x256x14x14xf32>
    %v603 = stablehlo.reshape %v602 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v604 = stablehlo.reshape %v603 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v605 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v606 = stablehlo.maximum %v604, %v605 : tensor<32x256x14x14xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v608 = stablehlo.reshape %v607 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v609 = stablehlo.convolution(%v608, %s3b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v610 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v611 = stablehlo.add %v609, %v610 : tensor<32x256x14x14xf32>
    %v612 = stablehlo.reshape %v611 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v613 = stablehlo.reshape %v612 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v614 = stablehlo.constant dense<0.0> : tensor<f32>
    %v615 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v616 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v617 = stablehlo.reduce(%v613 init: %v614) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v618 = stablehlo.broadcast_in_dim %v617, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v619 = stablehlo.divide %v618, %v615 : tensor<32x256x14x14xf32>
    %v620 = stablehlo.subtract %v613, %v619 : tensor<32x256x14x14xf32>
    %v621 = stablehlo.multiply %v620, %v620 : tensor<32x256x14x14xf32>
    %v622 = stablehlo.reduce(%v621 init: %v614) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v623 = stablehlo.broadcast_in_dim %v622, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v624 = stablehlo.divide %v623, %v615 : tensor<32x256x14x14xf32>
    %v625 = stablehlo.add %v624, %v616 : tensor<32x256x14x14xf32>
    %v626 = stablehlo.rsqrt %v625 : tensor<32x256x14x14xf32>
    %v627 = stablehlo.multiply %v620, %v626 : tensor<32x256x14x14xf32>
    %v628 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v629 = stablehlo.broadcast_in_dim %s3b0bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v630 = stablehlo.multiply %v627, %v628 : tensor<32x256x14x14xf32>
    %v631 = stablehlo.add %v630, %v629 : tensor<32x256x14x14xf32>
    %v632 = stablehlo.reshape %v631 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v633 = stablehlo.reshape %v632 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v634 = stablehlo.reshape %v578 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v635 = stablehlo.add %v633, %v634 : tensor<32x256x14x14xf32>
    %v636 = stablehlo.reshape %v635 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v637 = stablehlo.reshape %v636 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v638 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v639 = stablehlo.maximum %v637, %v638 : tensor<32x256x14x14xf32>
    %v640 = stablehlo.reshape %v639 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v641 = stablehlo.reshape %v640 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v642 = stablehlo.convolution(%v641, %s3b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v643 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v644 = stablehlo.add %v642, %v643 : tensor<32x256x14x14xf32>
    %v645 = stablehlo.reshape %v644 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v646 = stablehlo.reshape %v645 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v647 = stablehlo.constant dense<0.0> : tensor<f32>
    %v648 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v649 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v650 = stablehlo.reduce(%v646 init: %v647) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v651 = stablehlo.broadcast_in_dim %v650, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v652 = stablehlo.divide %v651, %v648 : tensor<32x256x14x14xf32>
    %v653 = stablehlo.subtract %v646, %v652 : tensor<32x256x14x14xf32>
    %v654 = stablehlo.multiply %v653, %v653 : tensor<32x256x14x14xf32>
    %v655 = stablehlo.reduce(%v654 init: %v647) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v656 = stablehlo.broadcast_in_dim %v655, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v657 = stablehlo.divide %v656, %v648 : tensor<32x256x14x14xf32>
    %v658 = stablehlo.add %v657, %v649 : tensor<32x256x14x14xf32>
    %v659 = stablehlo.rsqrt %v658 : tensor<32x256x14x14xf32>
    %v660 = stablehlo.multiply %v653, %v659 : tensor<32x256x14x14xf32>
    %v661 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v662 = stablehlo.broadcast_in_dim %s3b1bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v663 = stablehlo.multiply %v660, %v661 : tensor<32x256x14x14xf32>
    %v664 = stablehlo.add %v663, %v662 : tensor<32x256x14x14xf32>
    %v665 = stablehlo.reshape %v664 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v666 = stablehlo.reshape %v665 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v667 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v668 = stablehlo.maximum %v666, %v667 : tensor<32x256x14x14xf32>
    %v669 = stablehlo.reshape %v668 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v670 = stablehlo.reshape %v669 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v671 = stablehlo.convolution(%v670, %s3b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v672 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v673 = stablehlo.add %v671, %v672 : tensor<32x256x14x14xf32>
    %v674 = stablehlo.reshape %v673 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v675 = stablehlo.reshape %v674 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v676 = stablehlo.constant dense<0.0> : tensor<f32>
    %v677 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v678 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v679 = stablehlo.reduce(%v675 init: %v676) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v680 = stablehlo.broadcast_in_dim %v679, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v681 = stablehlo.divide %v680, %v677 : tensor<32x256x14x14xf32>
    %v682 = stablehlo.subtract %v675, %v681 : tensor<32x256x14x14xf32>
    %v683 = stablehlo.multiply %v682, %v682 : tensor<32x256x14x14xf32>
    %v684 = stablehlo.reduce(%v683 init: %v676) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v685 = stablehlo.broadcast_in_dim %v684, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v686 = stablehlo.divide %v685, %v677 : tensor<32x256x14x14xf32>
    %v687 = stablehlo.add %v686, %v678 : tensor<32x256x14x14xf32>
    %v688 = stablehlo.rsqrt %v687 : tensor<32x256x14x14xf32>
    %v689 = stablehlo.multiply %v682, %v688 : tensor<32x256x14x14xf32>
    %v690 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v691 = stablehlo.broadcast_in_dim %s3b1bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v692 = stablehlo.multiply %v689, %v690 : tensor<32x256x14x14xf32>
    %v693 = stablehlo.add %v692, %v691 : tensor<32x256x14x14xf32>
    %v694 = stablehlo.reshape %v693 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v695 = stablehlo.reshape %v694 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v696 = stablehlo.reshape %v640 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v697 = stablehlo.add %v695, %v696 : tensor<32x256x14x14xf32>
    %v698 = stablehlo.reshape %v697 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v699 = stablehlo.reshape %v698 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v700 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v701 = stablehlo.maximum %v699, %v700 : tensor<32x256x14x14xf32>
    %v702 = stablehlo.reshape %v701 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v703 = stablehlo.reshape %v702 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v704 = stablehlo.convolution(%v703, %s3b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v705 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v706 = stablehlo.add %v704, %v705 : tensor<32x256x14x14xf32>
    %v707 = stablehlo.reshape %v706 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v708 = stablehlo.reshape %v707 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v709 = stablehlo.constant dense<0.0> : tensor<f32>
    %v710 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v711 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v712 = stablehlo.reduce(%v708 init: %v709) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v713 = stablehlo.broadcast_in_dim %v712, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v714 = stablehlo.divide %v713, %v710 : tensor<32x256x14x14xf32>
    %v715 = stablehlo.subtract %v708, %v714 : tensor<32x256x14x14xf32>
    %v716 = stablehlo.multiply %v715, %v715 : tensor<32x256x14x14xf32>
    %v717 = stablehlo.reduce(%v716 init: %v709) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v718 = stablehlo.broadcast_in_dim %v717, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v719 = stablehlo.divide %v718, %v710 : tensor<32x256x14x14xf32>
    %v720 = stablehlo.add %v719, %v711 : tensor<32x256x14x14xf32>
    %v721 = stablehlo.rsqrt %v720 : tensor<32x256x14x14xf32>
    %v722 = stablehlo.multiply %v715, %v721 : tensor<32x256x14x14xf32>
    %v723 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v724 = stablehlo.broadcast_in_dim %s3b2bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v725 = stablehlo.multiply %v722, %v723 : tensor<32x256x14x14xf32>
    %v726 = stablehlo.add %v725, %v724 : tensor<32x256x14x14xf32>
    %v727 = stablehlo.reshape %v726 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v728 = stablehlo.reshape %v727 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v729 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v730 = stablehlo.maximum %v728, %v729 : tensor<32x256x14x14xf32>
    %v731 = stablehlo.reshape %v730 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v732 = stablehlo.reshape %v731 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v733 = stablehlo.convolution(%v732, %s3b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v734 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v735 = stablehlo.add %v733, %v734 : tensor<32x256x14x14xf32>
    %v736 = stablehlo.reshape %v735 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v737 = stablehlo.reshape %v736 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v738 = stablehlo.constant dense<0.0> : tensor<f32>
    %v739 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v740 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v741 = stablehlo.reduce(%v737 init: %v738) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v742 = stablehlo.broadcast_in_dim %v741, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v743 = stablehlo.divide %v742, %v739 : tensor<32x256x14x14xf32>
    %v744 = stablehlo.subtract %v737, %v743 : tensor<32x256x14x14xf32>
    %v745 = stablehlo.multiply %v744, %v744 : tensor<32x256x14x14xf32>
    %v746 = stablehlo.reduce(%v745 init: %v738) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v747 = stablehlo.broadcast_in_dim %v746, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v748 = stablehlo.divide %v747, %v739 : tensor<32x256x14x14xf32>
    %v749 = stablehlo.add %v748, %v740 : tensor<32x256x14x14xf32>
    %v750 = stablehlo.rsqrt %v749 : tensor<32x256x14x14xf32>
    %v751 = stablehlo.multiply %v744, %v750 : tensor<32x256x14x14xf32>
    %v752 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v753 = stablehlo.broadcast_in_dim %s3b2bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v754 = stablehlo.multiply %v751, %v752 : tensor<32x256x14x14xf32>
    %v755 = stablehlo.add %v754, %v753 : tensor<32x256x14x14xf32>
    %v756 = stablehlo.reshape %v755 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v757 = stablehlo.reshape %v756 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v758 = stablehlo.reshape %v702 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v759 = stablehlo.add %v757, %v758 : tensor<32x256x14x14xf32>
    %v760 = stablehlo.reshape %v759 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v761 = stablehlo.reshape %v760 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v762 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v763 = stablehlo.maximum %v761, %v762 : tensor<32x256x14x14xf32>
    %v764 = stablehlo.reshape %v763 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v765 = stablehlo.reshape %v764 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v766 = stablehlo.convolution(%v765, %s3b3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v767 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v768 = stablehlo.add %v766, %v767 : tensor<32x256x14x14xf32>
    %v769 = stablehlo.reshape %v768 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v770 = stablehlo.reshape %v769 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v771 = stablehlo.constant dense<0.0> : tensor<f32>
    %v772 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v773 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v774 = stablehlo.reduce(%v770 init: %v771) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v775 = stablehlo.broadcast_in_dim %v774, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v776 = stablehlo.divide %v775, %v772 : tensor<32x256x14x14xf32>
    %v777 = stablehlo.subtract %v770, %v776 : tensor<32x256x14x14xf32>
    %v778 = stablehlo.multiply %v777, %v777 : tensor<32x256x14x14xf32>
    %v779 = stablehlo.reduce(%v778 init: %v771) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v780 = stablehlo.broadcast_in_dim %v779, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v781 = stablehlo.divide %v780, %v772 : tensor<32x256x14x14xf32>
    %v782 = stablehlo.add %v781, %v773 : tensor<32x256x14x14xf32>
    %v783 = stablehlo.rsqrt %v782 : tensor<32x256x14x14xf32>
    %v784 = stablehlo.multiply %v777, %v783 : tensor<32x256x14x14xf32>
    %v785 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v786 = stablehlo.broadcast_in_dim %s3b3bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v787 = stablehlo.multiply %v784, %v785 : tensor<32x256x14x14xf32>
    %v788 = stablehlo.add %v787, %v786 : tensor<32x256x14x14xf32>
    %v789 = stablehlo.reshape %v788 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v790 = stablehlo.reshape %v789 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v791 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v792 = stablehlo.maximum %v790, %v791 : tensor<32x256x14x14xf32>
    %v793 = stablehlo.reshape %v792 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v794 = stablehlo.reshape %v793 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v795 = stablehlo.convolution(%v794, %s3b3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v796 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v797 = stablehlo.add %v795, %v796 : tensor<32x256x14x14xf32>
    %v798 = stablehlo.reshape %v797 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v799 = stablehlo.reshape %v798 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v800 = stablehlo.constant dense<0.0> : tensor<f32>
    %v801 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v802 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v803 = stablehlo.reduce(%v799 init: %v800) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v804 = stablehlo.broadcast_in_dim %v803, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v805 = stablehlo.divide %v804, %v801 : tensor<32x256x14x14xf32>
    %v806 = stablehlo.subtract %v799, %v805 : tensor<32x256x14x14xf32>
    %v807 = stablehlo.multiply %v806, %v806 : tensor<32x256x14x14xf32>
    %v808 = stablehlo.reduce(%v807 init: %v800) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v809 = stablehlo.broadcast_in_dim %v808, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v810 = stablehlo.divide %v809, %v801 : tensor<32x256x14x14xf32>
    %v811 = stablehlo.add %v810, %v802 : tensor<32x256x14x14xf32>
    %v812 = stablehlo.rsqrt %v811 : tensor<32x256x14x14xf32>
    %v813 = stablehlo.multiply %v806, %v812 : tensor<32x256x14x14xf32>
    %v814 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v815 = stablehlo.broadcast_in_dim %s3b3bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v816 = stablehlo.multiply %v813, %v814 : tensor<32x256x14x14xf32>
    %v817 = stablehlo.add %v816, %v815 : tensor<32x256x14x14xf32>
    %v818 = stablehlo.reshape %v817 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v819 = stablehlo.reshape %v818 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v820 = stablehlo.reshape %v764 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v821 = stablehlo.add %v819, %v820 : tensor<32x256x14x14xf32>
    %v822 = stablehlo.reshape %v821 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v823 = stablehlo.reshape %v822 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v824 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v825 = stablehlo.maximum %v823, %v824 : tensor<32x256x14x14xf32>
    %v826 = stablehlo.reshape %v825 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v827 = stablehlo.reshape %v826 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v828 = stablehlo.convolution(%v827, %s3b4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v829 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v830 = stablehlo.add %v828, %v829 : tensor<32x256x14x14xf32>
    %v831 = stablehlo.reshape %v830 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v832 = stablehlo.reshape %v831 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v833 = stablehlo.constant dense<0.0> : tensor<f32>
    %v834 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v835 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v836 = stablehlo.reduce(%v832 init: %v833) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v837 = stablehlo.broadcast_in_dim %v836, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v838 = stablehlo.divide %v837, %v834 : tensor<32x256x14x14xf32>
    %v839 = stablehlo.subtract %v832, %v838 : tensor<32x256x14x14xf32>
    %v840 = stablehlo.multiply %v839, %v839 : tensor<32x256x14x14xf32>
    %v841 = stablehlo.reduce(%v840 init: %v833) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v842 = stablehlo.broadcast_in_dim %v841, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v843 = stablehlo.divide %v842, %v834 : tensor<32x256x14x14xf32>
    %v844 = stablehlo.add %v843, %v835 : tensor<32x256x14x14xf32>
    %v845 = stablehlo.rsqrt %v844 : tensor<32x256x14x14xf32>
    %v846 = stablehlo.multiply %v839, %v845 : tensor<32x256x14x14xf32>
    %v847 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v848 = stablehlo.broadcast_in_dim %s3b4bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v849 = stablehlo.multiply %v846, %v847 : tensor<32x256x14x14xf32>
    %v850 = stablehlo.add %v849, %v848 : tensor<32x256x14x14xf32>
    %v851 = stablehlo.reshape %v850 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v852 = stablehlo.reshape %v851 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v853 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v854 = stablehlo.maximum %v852, %v853 : tensor<32x256x14x14xf32>
    %v855 = stablehlo.reshape %v854 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v856 = stablehlo.reshape %v855 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v857 = stablehlo.convolution(%v856, %s3b4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v858 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v859 = stablehlo.add %v857, %v858 : tensor<32x256x14x14xf32>
    %v860 = stablehlo.reshape %v859 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v861 = stablehlo.reshape %v860 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v862 = stablehlo.constant dense<0.0> : tensor<f32>
    %v863 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v864 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v865 = stablehlo.reduce(%v861 init: %v862) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v866 = stablehlo.broadcast_in_dim %v865, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v867 = stablehlo.divide %v866, %v863 : tensor<32x256x14x14xf32>
    %v868 = stablehlo.subtract %v861, %v867 : tensor<32x256x14x14xf32>
    %v869 = stablehlo.multiply %v868, %v868 : tensor<32x256x14x14xf32>
    %v870 = stablehlo.reduce(%v869 init: %v862) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v871 = stablehlo.broadcast_in_dim %v870, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v872 = stablehlo.divide %v871, %v863 : tensor<32x256x14x14xf32>
    %v873 = stablehlo.add %v872, %v864 : tensor<32x256x14x14xf32>
    %v874 = stablehlo.rsqrt %v873 : tensor<32x256x14x14xf32>
    %v875 = stablehlo.multiply %v868, %v874 : tensor<32x256x14x14xf32>
    %v876 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v877 = stablehlo.broadcast_in_dim %s3b4bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v878 = stablehlo.multiply %v875, %v876 : tensor<32x256x14x14xf32>
    %v879 = stablehlo.add %v878, %v877 : tensor<32x256x14x14xf32>
    %v880 = stablehlo.reshape %v879 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v881 = stablehlo.reshape %v880 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v882 = stablehlo.reshape %v826 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v883 = stablehlo.add %v881, %v882 : tensor<32x256x14x14xf32>
    %v884 = stablehlo.reshape %v883 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v885 = stablehlo.reshape %v884 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v886 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v887 = stablehlo.maximum %v885, %v886 : tensor<32x256x14x14xf32>
    %v888 = stablehlo.reshape %v887 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v889 = stablehlo.reshape %v888 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v890 = stablehlo.convolution(%v889, %d4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<512x256x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v891 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v892 = stablehlo.add %v890, %v891 : tensor<32x512x7x7xf32>
    %v893 = stablehlo.reshape %v892 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v894 = stablehlo.reshape %v893 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v895 = stablehlo.constant dense<0.0> : tensor<f32>
    %v896 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v897 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v898 = stablehlo.reduce(%v894 init: %v895) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v899 = stablehlo.broadcast_in_dim %v898, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v900 = stablehlo.divide %v899, %v896 : tensor<32x512x7x7xf32>
    %v901 = stablehlo.subtract %v894, %v900 : tensor<32x512x7x7xf32>
    %v902 = stablehlo.multiply %v901, %v901 : tensor<32x512x7x7xf32>
    %v903 = stablehlo.reduce(%v902 init: %v895) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v904 = stablehlo.broadcast_in_dim %v903, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v905 = stablehlo.divide %v904, %v896 : tensor<32x512x7x7xf32>
    %v906 = stablehlo.add %v905, %v897 : tensor<32x512x7x7xf32>
    %v907 = stablehlo.rsqrt %v906 : tensor<32x512x7x7xf32>
    %v908 = stablehlo.multiply %v901, %v907 : tensor<32x512x7x7xf32>
    %v909 = stablehlo.broadcast_in_dim %d4g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v910 = stablehlo.broadcast_in_dim %d4bt1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v911 = stablehlo.multiply %v908, %v909 : tensor<32x512x7x7xf32>
    %v912 = stablehlo.add %v911, %v910 : tensor<32x512x7x7xf32>
    %v913 = stablehlo.reshape %v912 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v914 = stablehlo.reshape %v913 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v915 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v916 = stablehlo.maximum %v914, %v915 : tensor<32x512x7x7xf32>
    %v917 = stablehlo.reshape %v916 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v918 = stablehlo.reshape %v917 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v919 = stablehlo.convolution(%v918, %d4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v920 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v921 = stablehlo.add %v919, %v920 : tensor<32x512x7x7xf32>
    %v922 = stablehlo.reshape %v921 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v923 = stablehlo.reshape %v922 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v924 = stablehlo.constant dense<0.0> : tensor<f32>
    %v925 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v926 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v927 = stablehlo.reduce(%v923 init: %v924) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v928 = stablehlo.broadcast_in_dim %v927, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v929 = stablehlo.divide %v928, %v925 : tensor<32x512x7x7xf32>
    %v930 = stablehlo.subtract %v923, %v929 : tensor<32x512x7x7xf32>
    %v931 = stablehlo.multiply %v930, %v930 : tensor<32x512x7x7xf32>
    %v932 = stablehlo.reduce(%v931 init: %v924) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v933 = stablehlo.broadcast_in_dim %v932, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v934 = stablehlo.divide %v933, %v925 : tensor<32x512x7x7xf32>
    %v935 = stablehlo.add %v934, %v926 : tensor<32x512x7x7xf32>
    %v936 = stablehlo.rsqrt %v935 : tensor<32x512x7x7xf32>
    %v937 = stablehlo.multiply %v930, %v936 : tensor<32x512x7x7xf32>
    %v938 = stablehlo.broadcast_in_dim %d4g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v939 = stablehlo.broadcast_in_dim %d4bt2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v940 = stablehlo.multiply %v937, %v938 : tensor<32x512x7x7xf32>
    %v941 = stablehlo.add %v940, %v939 : tensor<32x512x7x7xf32>
    %v942 = stablehlo.reshape %v941 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v943 = stablehlo.reshape %v888 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v944 = stablehlo.convolution(%v943, %d4Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<512x256x1x1xf32>) -> tensor<32x512x7x7xf32>
    %v945 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v946 = stablehlo.add %v944, %v945 : tensor<32x512x7x7xf32>
    %v947 = stablehlo.reshape %v946 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v948 = stablehlo.reshape %v947 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v949 = stablehlo.constant dense<0.0> : tensor<f32>
    %v950 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v951 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v952 = stablehlo.reduce(%v948 init: %v949) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v953 = stablehlo.broadcast_in_dim %v952, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v954 = stablehlo.divide %v953, %v950 : tensor<32x512x7x7xf32>
    %v955 = stablehlo.subtract %v948, %v954 : tensor<32x512x7x7xf32>
    %v956 = stablehlo.multiply %v955, %v955 : tensor<32x512x7x7xf32>
    %v957 = stablehlo.reduce(%v956 init: %v949) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v958 = stablehlo.broadcast_in_dim %v957, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v959 = stablehlo.divide %v958, %v950 : tensor<32x512x7x7xf32>
    %v960 = stablehlo.add %v959, %v951 : tensor<32x512x7x7xf32>
    %v961 = stablehlo.rsqrt %v960 : tensor<32x512x7x7xf32>
    %v962 = stablehlo.multiply %v955, %v961 : tensor<32x512x7x7xf32>
    %v963 = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v964 = stablehlo.broadcast_in_dim %d4btp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v965 = stablehlo.multiply %v962, %v963 : tensor<32x512x7x7xf32>
    %v966 = stablehlo.add %v965, %v964 : tensor<32x512x7x7xf32>
    %v967 = stablehlo.reshape %v966 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v968 = stablehlo.reshape %v942 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v969 = stablehlo.reshape %v967 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v970 = stablehlo.add %v968, %v969 : tensor<32x512x7x7xf32>
    %v971 = stablehlo.reshape %v970 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v972 = stablehlo.reshape %v971 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v973 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v974 = stablehlo.maximum %v972, %v973 : tensor<32x512x7x7xf32>
    %v975 = stablehlo.reshape %v974 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v976 = stablehlo.reshape %v975 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v977 = stablehlo.convolution(%v976, %s4b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v978 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v979 = stablehlo.add %v977, %v978 : tensor<32x512x7x7xf32>
    %v980 = stablehlo.reshape %v979 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v981 = stablehlo.reshape %v980 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v982 = stablehlo.constant dense<0.0> : tensor<f32>
    %v983 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v984 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v985 = stablehlo.reduce(%v981 init: %v982) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v986 = stablehlo.broadcast_in_dim %v985, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v987 = stablehlo.divide %v986, %v983 : tensor<32x512x7x7xf32>
    %v988 = stablehlo.subtract %v981, %v987 : tensor<32x512x7x7xf32>
    %v989 = stablehlo.multiply %v988, %v988 : tensor<32x512x7x7xf32>
    %v990 = stablehlo.reduce(%v989 init: %v982) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v991 = stablehlo.broadcast_in_dim %v990, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v992 = stablehlo.divide %v991, %v983 : tensor<32x512x7x7xf32>
    %v993 = stablehlo.add %v992, %v984 : tensor<32x512x7x7xf32>
    %v994 = stablehlo.rsqrt %v993 : tensor<32x512x7x7xf32>
    %v995 = stablehlo.multiply %v988, %v994 : tensor<32x512x7x7xf32>
    %v996 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v997 = stablehlo.broadcast_in_dim %s4b0bt1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v998 = stablehlo.multiply %v995, %v996 : tensor<32x512x7x7xf32>
    %v999 = stablehlo.add %v998, %v997 : tensor<32x512x7x7xf32>
    %v1000 = stablehlo.reshape %v999 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1001 = stablehlo.reshape %v1000 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1002 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1003 = stablehlo.maximum %v1001, %v1002 : tensor<32x512x7x7xf32>
    %v1004 = stablehlo.reshape %v1003 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1005 = stablehlo.reshape %v1004 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1006 = stablehlo.convolution(%v1005, %s4b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1007 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1008 = stablehlo.add %v1006, %v1007 : tensor<32x512x7x7xf32>
    %v1009 = stablehlo.reshape %v1008 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1010 = stablehlo.reshape %v1009 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1011 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1012 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1013 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1014 = stablehlo.reduce(%v1010 init: %v1011) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1015 = stablehlo.broadcast_in_dim %v1014, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1016 = stablehlo.divide %v1015, %v1012 : tensor<32x512x7x7xf32>
    %v1017 = stablehlo.subtract %v1010, %v1016 : tensor<32x512x7x7xf32>
    %v1018 = stablehlo.multiply %v1017, %v1017 : tensor<32x512x7x7xf32>
    %v1019 = stablehlo.reduce(%v1018 init: %v1011) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1020 = stablehlo.broadcast_in_dim %v1019, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1021 = stablehlo.divide %v1020, %v1012 : tensor<32x512x7x7xf32>
    %v1022 = stablehlo.add %v1021, %v1013 : tensor<32x512x7x7xf32>
    %v1023 = stablehlo.rsqrt %v1022 : tensor<32x512x7x7xf32>
    %v1024 = stablehlo.multiply %v1017, %v1023 : tensor<32x512x7x7xf32>
    %v1025 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1026 = stablehlo.broadcast_in_dim %s4b0bt2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1027 = stablehlo.multiply %v1024, %v1025 : tensor<32x512x7x7xf32>
    %v1028 = stablehlo.add %v1027, %v1026 : tensor<32x512x7x7xf32>
    %v1029 = stablehlo.reshape %v1028 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1030 = stablehlo.reshape %v1029 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1031 = stablehlo.reshape %v975 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1032 = stablehlo.add %v1030, %v1031 : tensor<32x512x7x7xf32>
    %v1033 = stablehlo.reshape %v1032 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1034 = stablehlo.reshape %v1033 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1035 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1036 = stablehlo.maximum %v1034, %v1035 : tensor<32x512x7x7xf32>
    %v1037 = stablehlo.reshape %v1036 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1038 = stablehlo.reshape %v1037 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1039 = stablehlo.convolution(%v1038, %s4b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1040 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1041 = stablehlo.add %v1039, %v1040 : tensor<32x512x7x7xf32>
    %v1042 = stablehlo.reshape %v1041 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1043 = stablehlo.reshape %v1042 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1044 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1045 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1046 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1047 = stablehlo.reduce(%v1043 init: %v1044) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1048 = stablehlo.broadcast_in_dim %v1047, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1049 = stablehlo.divide %v1048, %v1045 : tensor<32x512x7x7xf32>
    %v1050 = stablehlo.subtract %v1043, %v1049 : tensor<32x512x7x7xf32>
    %v1051 = stablehlo.multiply %v1050, %v1050 : tensor<32x512x7x7xf32>
    %v1052 = stablehlo.reduce(%v1051 init: %v1044) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1053 = stablehlo.broadcast_in_dim %v1052, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1054 = stablehlo.divide %v1053, %v1045 : tensor<32x512x7x7xf32>
    %v1055 = stablehlo.add %v1054, %v1046 : tensor<32x512x7x7xf32>
    %v1056 = stablehlo.rsqrt %v1055 : tensor<32x512x7x7xf32>
    %v1057 = stablehlo.multiply %v1050, %v1056 : tensor<32x512x7x7xf32>
    %v1058 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1059 = stablehlo.broadcast_in_dim %s4b1bt1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1060 = stablehlo.multiply %v1057, %v1058 : tensor<32x512x7x7xf32>
    %v1061 = stablehlo.add %v1060, %v1059 : tensor<32x512x7x7xf32>
    %v1062 = stablehlo.reshape %v1061 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1063 = stablehlo.reshape %v1062 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1064 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1065 = stablehlo.maximum %v1063, %v1064 : tensor<32x512x7x7xf32>
    %v1066 = stablehlo.reshape %v1065 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1067 = stablehlo.reshape %v1066 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1068 = stablehlo.convolution(%v1067, %s4b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1069 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1070 = stablehlo.add %v1068, %v1069 : tensor<32x512x7x7xf32>
    %v1071 = stablehlo.reshape %v1070 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1072 = stablehlo.reshape %v1071 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1073 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1074 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1075 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1076 = stablehlo.reduce(%v1072 init: %v1073) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1077 = stablehlo.broadcast_in_dim %v1076, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1078 = stablehlo.divide %v1077, %v1074 : tensor<32x512x7x7xf32>
    %v1079 = stablehlo.subtract %v1072, %v1078 : tensor<32x512x7x7xf32>
    %v1080 = stablehlo.multiply %v1079, %v1079 : tensor<32x512x7x7xf32>
    %v1081 = stablehlo.reduce(%v1080 init: %v1073) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1082 = stablehlo.broadcast_in_dim %v1081, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1083 = stablehlo.divide %v1082, %v1074 : tensor<32x512x7x7xf32>
    %v1084 = stablehlo.add %v1083, %v1075 : tensor<32x512x7x7xf32>
    %v1085 = stablehlo.rsqrt %v1084 : tensor<32x512x7x7xf32>
    %v1086 = stablehlo.multiply %v1079, %v1085 : tensor<32x512x7x7xf32>
    %v1087 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1088 = stablehlo.broadcast_in_dim %s4b1bt2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1089 = stablehlo.multiply %v1086, %v1087 : tensor<32x512x7x7xf32>
    %v1090 = stablehlo.add %v1089, %v1088 : tensor<32x512x7x7xf32>
    %v1091 = stablehlo.reshape %v1090 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1092 = stablehlo.reshape %v1091 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1093 = stablehlo.reshape %v1037 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1094 = stablehlo.add %v1092, %v1093 : tensor<32x512x7x7xf32>
    %v1095 = stablehlo.reshape %v1094 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1096 = stablehlo.reshape %v1095 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1097 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1098 = stablehlo.maximum %v1096, %v1097 : tensor<32x512x7x7xf32>
    %v1099 = stablehlo.reshape %v1098 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1100 = stablehlo.reshape %v1099 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1101 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1102 = stablehlo.reduce(%v1100 init: %v1101) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1103 = stablehlo.constant dense<49.0> : tensor<32x512xf32>
    %v1104 = stablehlo.divide %v1102, %v1103 : tensor<32x512xf32>
    %v1105 = stablehlo.dot_general %v1104, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x512xf32>, tensor<512x10xf32>) -> tensor<32x10xf32>
    %v1106 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v1107 = stablehlo.add %v1105, %v1106 : tensor<32x10xf32>
    %v1108 = stablehlo.exponential %v1107 : tensor<32x10xf32>
    %v1109 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1110 = stablehlo.reduce(%v1108 init: %v1109) applies stablehlo.add across dimensions = [1] : (tensor<32x10xf32>, tensor<f32>) -> tensor<32xf32>
    %v1111 = stablehlo.broadcast_in_dim %v1110, dims = [0] : (tensor<32xf32>) -> tensor<32x10xf32>
    %v1112 = stablehlo.divide %v1108, %v1111 : tensor<32x10xf32>
    %v1113 = stablehlo.subtract %v1112, %onehot : tensor<32x10xf32>
    %v1114 = stablehlo.dot_general %v1113, %Wd, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<512x10xf32>) -> tensor<32x512xf32>
    %v1115 = stablehlo.constant dense<49.0> : tensor<32x512xf32>
    %v1116 = stablehlo.divide %v1114, %v1115 : tensor<32x512xf32>
    %v1117 = stablehlo.broadcast_in_dim %v1116, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1118 = stablehlo.reshape %v1117 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1119 = stablehlo.dot_general %v1104, %v1113, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x512xf32>, tensor<32x10xf32>) -> tensor<512x10xf32>
    %v1120 = stablehlo.constant dense<0.003125> : tensor<512x10xf32>
    %v1121 = stablehlo.multiply %v1119, %v1120 : tensor<512x10xf32>
    %v1122 = stablehlo.subtract %Wd, %v1121 : tensor<512x10xf32>
    %v1123 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1124 = stablehlo.reduce(%v1113 init: %v1123) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1125 = stablehlo.constant dense<0.003125> : tensor<10xf32>
    %v1126 = stablehlo.multiply %v1124, %v1125 : tensor<10xf32>
    %v1127 = stablehlo.subtract %bd, %v1126 : tensor<10xf32>
    %v1128 = stablehlo.reshape %v1118 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1129 = stablehlo.reshape %v1095 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1130 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1131 = stablehlo.compare GT, %v1129, %v1130 : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %v1132 = stablehlo.select %v1131, %v1128, %v1130 : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %v1133 = stablehlo.reshape %v1132 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1134 = stablehlo.reshape %v1133 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1135 = stablehlo.reshape %v1071 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1136 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1137 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1138 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1139 = stablehlo.reduce(%v1135 init: %v1136) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1140 = stablehlo.broadcast_in_dim %v1139, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1141 = stablehlo.divide %v1140, %v1137 : tensor<32x512x7x7xf32>
    %v1142 = stablehlo.subtract %v1135, %v1141 : tensor<32x512x7x7xf32>
    %v1143 = stablehlo.multiply %v1142, %v1142 : tensor<32x512x7x7xf32>
    %v1144 = stablehlo.reduce(%v1143 init: %v1136) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1145 = stablehlo.broadcast_in_dim %v1144, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1146 = stablehlo.divide %v1145, %v1137 : tensor<32x512x7x7xf32>
    %v1147 = stablehlo.add %v1146, %v1138 : tensor<32x512x7x7xf32>
    %v1148 = stablehlo.rsqrt %v1147 : tensor<32x512x7x7xf32>
    %v1149 = stablehlo.multiply %v1142, %v1148 : tensor<32x512x7x7xf32>
    %v1150 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1151 = stablehlo.multiply %v1150, %v1134 : tensor<32x512x7x7xf32>
    %v1152 = stablehlo.reduce(%v1151 init: %v1136) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1153 = stablehlo.broadcast_in_dim %v1152, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1154 = stablehlo.multiply %v1149, %v1151 : tensor<32x512x7x7xf32>
    %v1155 = stablehlo.reduce(%v1154 init: %v1136) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1156 = stablehlo.broadcast_in_dim %v1155, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1157 = stablehlo.multiply %v1151, %v1137 : tensor<32x512x7x7xf32>
    %v1158 = stablehlo.subtract %v1157, %v1153 : tensor<32x512x7x7xf32>
    %v1159 = stablehlo.multiply %v1149, %v1156 : tensor<32x512x7x7xf32>
    %v1160 = stablehlo.subtract %v1158, %v1159 : tensor<32x512x7x7xf32>
    %v1161 = stablehlo.divide %v1148, %v1137 : tensor<32x512x7x7xf32>
    %v1162 = stablehlo.multiply %v1161, %v1160 : tensor<32x512x7x7xf32>
    %v1163 = stablehlo.reshape %v1162 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1164 = stablehlo.reshape %v1163 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1165 = stablehlo.transpose %s4b1W2, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1166 = stablehlo.reverse %v1165, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1167 = stablehlo.convolution(%v1164, %v1166)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1168 = stablehlo.reshape %v1167 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1169 = stablehlo.reshape %v1168 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1170 = stablehlo.reshape %v1062 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1171 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1172 = stablehlo.compare GT, %v1170, %v1171 : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %v1173 = stablehlo.select %v1172, %v1169, %v1171 : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %v1174 = stablehlo.reshape %v1173 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1175 = stablehlo.reshape %v1174 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1176 = stablehlo.reshape %v1042 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1177 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1178 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1179 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1180 = stablehlo.reduce(%v1176 init: %v1177) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1181 = stablehlo.broadcast_in_dim %v1180, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1182 = stablehlo.divide %v1181, %v1178 : tensor<32x512x7x7xf32>
    %v1183 = stablehlo.subtract %v1176, %v1182 : tensor<32x512x7x7xf32>
    %v1184 = stablehlo.multiply %v1183, %v1183 : tensor<32x512x7x7xf32>
    %v1185 = stablehlo.reduce(%v1184 init: %v1177) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1186 = stablehlo.broadcast_in_dim %v1185, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1187 = stablehlo.divide %v1186, %v1178 : tensor<32x512x7x7xf32>
    %v1188 = stablehlo.add %v1187, %v1179 : tensor<32x512x7x7xf32>
    %v1189 = stablehlo.rsqrt %v1188 : tensor<32x512x7x7xf32>
    %v1190 = stablehlo.multiply %v1183, %v1189 : tensor<32x512x7x7xf32>
    %v1191 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1192 = stablehlo.multiply %v1191, %v1175 : tensor<32x512x7x7xf32>
    %v1193 = stablehlo.reduce(%v1192 init: %v1177) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1194 = stablehlo.broadcast_in_dim %v1193, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1195 = stablehlo.multiply %v1190, %v1192 : tensor<32x512x7x7xf32>
    %v1196 = stablehlo.reduce(%v1195 init: %v1177) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1197 = stablehlo.broadcast_in_dim %v1196, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1198 = stablehlo.multiply %v1192, %v1178 : tensor<32x512x7x7xf32>
    %v1199 = stablehlo.subtract %v1198, %v1194 : tensor<32x512x7x7xf32>
    %v1200 = stablehlo.multiply %v1190, %v1197 : tensor<32x512x7x7xf32>
    %v1201 = stablehlo.subtract %v1199, %v1200 : tensor<32x512x7x7xf32>
    %v1202 = stablehlo.divide %v1189, %v1178 : tensor<32x512x7x7xf32>
    %v1203 = stablehlo.multiply %v1202, %v1201 : tensor<32x512x7x7xf32>
    %v1204 = stablehlo.reshape %v1203 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1205 = stablehlo.reshape %v1204 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1206 = stablehlo.transpose %s4b1W1, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1207 = stablehlo.reverse %v1206, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1208 = stablehlo.convolution(%v1205, %v1207)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1209 = stablehlo.reshape %v1208 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1210 = stablehlo.reshape %v1209 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1211 = stablehlo.reshape %v1133 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1212 = stablehlo.add %v1210, %v1211 : tensor<32x512x7x7xf32>
    %v1213 = stablehlo.reshape %v1212 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1214 = stablehlo.reshape %v1037 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1215 = stablehlo.reshape %v1204 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1216 = stablehlo.transpose %v1214, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1217 = stablehlo.transpose %v1215, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1218 = stablehlo.convolution(%v1216, %v1217)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1219 = stablehlo.transpose %v1218, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1220 = stablehlo.constant dense<0.003125> : tensor<512x512x3x3xf32>
    %v1221 = stablehlo.multiply %v1219, %v1220 : tensor<512x512x3x3xf32>
    %v1222 = stablehlo.subtract %s4b1W1, %v1221 : tensor<512x512x3x3xf32>
    %v1223 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1224 = stablehlo.reshape %v1042 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1225 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1226 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1227 = stablehlo.reduce(%v1224 init: %v1223) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1228 = stablehlo.broadcast_in_dim %v1227, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1229 = stablehlo.divide %v1228, %v1225 : tensor<32x512x7x7xf32>
    %v1230 = stablehlo.subtract %v1224, %v1229 : tensor<32x512x7x7xf32>
    %v1231 = stablehlo.multiply %v1230, %v1230 : tensor<32x512x7x7xf32>
    %v1232 = stablehlo.reduce(%v1231 init: %v1223) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1233 = stablehlo.broadcast_in_dim %v1232, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1234 = stablehlo.divide %v1233, %v1225 : tensor<32x512x7x7xf32>
    %v1235 = stablehlo.add %v1234, %v1226 : tensor<32x512x7x7xf32>
    %v1236 = stablehlo.rsqrt %v1235 : tensor<32x512x7x7xf32>
    %v1237 = stablehlo.multiply %v1230, %v1236 : tensor<32x512x7x7xf32>
    %v1238 = stablehlo.reshape %v1174 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1239 = stablehlo.multiply %v1238, %v1237 : tensor<32x512x7x7xf32>
    %v1240 = stablehlo.reduce(%v1239 init: %v1223) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1241 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1242 = stablehlo.multiply %v1240, %v1241 : tensor<512xf32>
    %v1243 = stablehlo.subtract %s4b1g1, %v1242 : tensor<512xf32>
    %v1244 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1245 = stablehlo.reshape %v1174 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1246 = stablehlo.reduce(%v1245 init: %v1244) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1247 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1248 = stablehlo.multiply %v1246, %v1247 : tensor<512xf32>
    %v1249 = stablehlo.subtract %s4b1bt1, %v1248 : tensor<512xf32>
    %v1250 = stablehlo.reshape %v1066 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1251 = stablehlo.reshape %v1163 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1252 = stablehlo.transpose %v1250, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1253 = stablehlo.transpose %v1251, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1254 = stablehlo.convolution(%v1252, %v1253)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1255 = stablehlo.transpose %v1254, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1256 = stablehlo.constant dense<0.003125> : tensor<512x512x3x3xf32>
    %v1257 = stablehlo.multiply %v1255, %v1256 : tensor<512x512x3x3xf32>
    %v1258 = stablehlo.subtract %s4b1W2, %v1257 : tensor<512x512x3x3xf32>
    %v1259 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1260 = stablehlo.reshape %v1071 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1261 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1262 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1263 = stablehlo.reduce(%v1260 init: %v1259) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1264 = stablehlo.broadcast_in_dim %v1263, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1265 = stablehlo.divide %v1264, %v1261 : tensor<32x512x7x7xf32>
    %v1266 = stablehlo.subtract %v1260, %v1265 : tensor<32x512x7x7xf32>
    %v1267 = stablehlo.multiply %v1266, %v1266 : tensor<32x512x7x7xf32>
    %v1268 = stablehlo.reduce(%v1267 init: %v1259) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1269 = stablehlo.broadcast_in_dim %v1268, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1270 = stablehlo.divide %v1269, %v1261 : tensor<32x512x7x7xf32>
    %v1271 = stablehlo.add %v1270, %v1262 : tensor<32x512x7x7xf32>
    %v1272 = stablehlo.rsqrt %v1271 : tensor<32x512x7x7xf32>
    %v1273 = stablehlo.multiply %v1266, %v1272 : tensor<32x512x7x7xf32>
    %v1274 = stablehlo.reshape %v1133 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1275 = stablehlo.multiply %v1274, %v1273 : tensor<32x512x7x7xf32>
    %v1276 = stablehlo.reduce(%v1275 init: %v1259) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1277 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1278 = stablehlo.multiply %v1276, %v1277 : tensor<512xf32>
    %v1279 = stablehlo.subtract %s4b1g2, %v1278 : tensor<512xf32>
    %v1280 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1281 = stablehlo.reshape %v1133 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1282 = stablehlo.reduce(%v1281 init: %v1280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1283 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1284 = stablehlo.multiply %v1282, %v1283 : tensor<512xf32>
    %v1285 = stablehlo.subtract %s4b1bt2, %v1284 : tensor<512xf32>
    %v1286 = stablehlo.reshape %v1213 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1287 = stablehlo.reshape %v1033 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1288 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1289 = stablehlo.compare GT, %v1287, %v1288 : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %v1290 = stablehlo.select %v1289, %v1286, %v1288 : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %v1291 = stablehlo.reshape %v1290 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1292 = stablehlo.reshape %v1291 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1293 = stablehlo.reshape %v1009 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1294 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1295 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1296 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1297 = stablehlo.reduce(%v1293 init: %v1294) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1298 = stablehlo.broadcast_in_dim %v1297, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1299 = stablehlo.divide %v1298, %v1295 : tensor<32x512x7x7xf32>
    %v1300 = stablehlo.subtract %v1293, %v1299 : tensor<32x512x7x7xf32>
    %v1301 = stablehlo.multiply %v1300, %v1300 : tensor<32x512x7x7xf32>
    %v1302 = stablehlo.reduce(%v1301 init: %v1294) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1303 = stablehlo.broadcast_in_dim %v1302, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1304 = stablehlo.divide %v1303, %v1295 : tensor<32x512x7x7xf32>
    %v1305 = stablehlo.add %v1304, %v1296 : tensor<32x512x7x7xf32>
    %v1306 = stablehlo.rsqrt %v1305 : tensor<32x512x7x7xf32>
    %v1307 = stablehlo.multiply %v1300, %v1306 : tensor<32x512x7x7xf32>
    %v1308 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1309 = stablehlo.multiply %v1308, %v1292 : tensor<32x512x7x7xf32>
    %v1310 = stablehlo.reduce(%v1309 init: %v1294) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1311 = stablehlo.broadcast_in_dim %v1310, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1312 = stablehlo.multiply %v1307, %v1309 : tensor<32x512x7x7xf32>
    %v1313 = stablehlo.reduce(%v1312 init: %v1294) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1314 = stablehlo.broadcast_in_dim %v1313, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1315 = stablehlo.multiply %v1309, %v1295 : tensor<32x512x7x7xf32>
    %v1316 = stablehlo.subtract %v1315, %v1311 : tensor<32x512x7x7xf32>
    %v1317 = stablehlo.multiply %v1307, %v1314 : tensor<32x512x7x7xf32>
    %v1318 = stablehlo.subtract %v1316, %v1317 : tensor<32x512x7x7xf32>
    %v1319 = stablehlo.divide %v1306, %v1295 : tensor<32x512x7x7xf32>
    %v1320 = stablehlo.multiply %v1319, %v1318 : tensor<32x512x7x7xf32>
    %v1321 = stablehlo.reshape %v1320 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1322 = stablehlo.reshape %v1321 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1323 = stablehlo.transpose %s4b0W2, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1324 = stablehlo.reverse %v1323, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1325 = stablehlo.convolution(%v1322, %v1324)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1326 = stablehlo.reshape %v1325 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1327 = stablehlo.reshape %v1326 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1328 = stablehlo.reshape %v1000 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1329 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1330 = stablehlo.compare GT, %v1328, %v1329 : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %v1331 = stablehlo.select %v1330, %v1327, %v1329 : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %v1332 = stablehlo.reshape %v1331 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1333 = stablehlo.reshape %v1332 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1334 = stablehlo.reshape %v980 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1335 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1336 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1337 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1338 = stablehlo.reduce(%v1334 init: %v1335) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1339 = stablehlo.broadcast_in_dim %v1338, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1340 = stablehlo.divide %v1339, %v1336 : tensor<32x512x7x7xf32>
    %v1341 = stablehlo.subtract %v1334, %v1340 : tensor<32x512x7x7xf32>
    %v1342 = stablehlo.multiply %v1341, %v1341 : tensor<32x512x7x7xf32>
    %v1343 = stablehlo.reduce(%v1342 init: %v1335) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1344 = stablehlo.broadcast_in_dim %v1343, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1345 = stablehlo.divide %v1344, %v1336 : tensor<32x512x7x7xf32>
    %v1346 = stablehlo.add %v1345, %v1337 : tensor<32x512x7x7xf32>
    %v1347 = stablehlo.rsqrt %v1346 : tensor<32x512x7x7xf32>
    %v1348 = stablehlo.multiply %v1341, %v1347 : tensor<32x512x7x7xf32>
    %v1349 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1350 = stablehlo.multiply %v1349, %v1333 : tensor<32x512x7x7xf32>
    %v1351 = stablehlo.reduce(%v1350 init: %v1335) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1352 = stablehlo.broadcast_in_dim %v1351, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1353 = stablehlo.multiply %v1348, %v1350 : tensor<32x512x7x7xf32>
    %v1354 = stablehlo.reduce(%v1353 init: %v1335) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1355 = stablehlo.broadcast_in_dim %v1354, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1356 = stablehlo.multiply %v1350, %v1336 : tensor<32x512x7x7xf32>
    %v1357 = stablehlo.subtract %v1356, %v1352 : tensor<32x512x7x7xf32>
    %v1358 = stablehlo.multiply %v1348, %v1355 : tensor<32x512x7x7xf32>
    %v1359 = stablehlo.subtract %v1357, %v1358 : tensor<32x512x7x7xf32>
    %v1360 = stablehlo.divide %v1347, %v1336 : tensor<32x512x7x7xf32>
    %v1361 = stablehlo.multiply %v1360, %v1359 : tensor<32x512x7x7xf32>
    %v1362 = stablehlo.reshape %v1361 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1363 = stablehlo.reshape %v1362 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1364 = stablehlo.transpose %s4b0W1, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1365 = stablehlo.reverse %v1364, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1366 = stablehlo.convolution(%v1363, %v1365)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1367 = stablehlo.reshape %v1366 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1368 = stablehlo.reshape %v1367 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1369 = stablehlo.reshape %v1291 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1370 = stablehlo.add %v1368, %v1369 : tensor<32x512x7x7xf32>
    %v1371 = stablehlo.reshape %v1370 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1372 = stablehlo.reshape %v975 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1373 = stablehlo.reshape %v1362 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1374 = stablehlo.transpose %v1372, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1375 = stablehlo.transpose %v1373, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1376 = stablehlo.convolution(%v1374, %v1375)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1377 = stablehlo.transpose %v1376, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1378 = stablehlo.constant dense<0.003125> : tensor<512x512x3x3xf32>
    %v1379 = stablehlo.multiply %v1377, %v1378 : tensor<512x512x3x3xf32>
    %v1380 = stablehlo.subtract %s4b0W1, %v1379 : tensor<512x512x3x3xf32>
    %v1381 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1382 = stablehlo.reshape %v980 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1383 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1384 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1385 = stablehlo.reduce(%v1382 init: %v1381) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1386 = stablehlo.broadcast_in_dim %v1385, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1387 = stablehlo.divide %v1386, %v1383 : tensor<32x512x7x7xf32>
    %v1388 = stablehlo.subtract %v1382, %v1387 : tensor<32x512x7x7xf32>
    %v1389 = stablehlo.multiply %v1388, %v1388 : tensor<32x512x7x7xf32>
    %v1390 = stablehlo.reduce(%v1389 init: %v1381) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1391 = stablehlo.broadcast_in_dim %v1390, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1392 = stablehlo.divide %v1391, %v1383 : tensor<32x512x7x7xf32>
    %v1393 = stablehlo.add %v1392, %v1384 : tensor<32x512x7x7xf32>
    %v1394 = stablehlo.rsqrt %v1393 : tensor<32x512x7x7xf32>
    %v1395 = stablehlo.multiply %v1388, %v1394 : tensor<32x512x7x7xf32>
    %v1396 = stablehlo.reshape %v1332 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1397 = stablehlo.multiply %v1396, %v1395 : tensor<32x512x7x7xf32>
    %v1398 = stablehlo.reduce(%v1397 init: %v1381) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1399 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1400 = stablehlo.multiply %v1398, %v1399 : tensor<512xf32>
    %v1401 = stablehlo.subtract %s4b0g1, %v1400 : tensor<512xf32>
    %v1402 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1403 = stablehlo.reshape %v1332 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1404 = stablehlo.reduce(%v1403 init: %v1402) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1405 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1406 = stablehlo.multiply %v1404, %v1405 : tensor<512xf32>
    %v1407 = stablehlo.subtract %s4b0bt1, %v1406 : tensor<512xf32>
    %v1408 = stablehlo.reshape %v1004 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1409 = stablehlo.reshape %v1321 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1410 = stablehlo.transpose %v1408, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1411 = stablehlo.transpose %v1409, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1412 = stablehlo.convolution(%v1410, %v1411)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1413 = stablehlo.transpose %v1412, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1414 = stablehlo.constant dense<0.003125> : tensor<512x512x3x3xf32>
    %v1415 = stablehlo.multiply %v1413, %v1414 : tensor<512x512x3x3xf32>
    %v1416 = stablehlo.subtract %s4b0W2, %v1415 : tensor<512x512x3x3xf32>
    %v1417 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1418 = stablehlo.reshape %v1009 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1419 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1420 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1421 = stablehlo.reduce(%v1418 init: %v1417) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1422 = stablehlo.broadcast_in_dim %v1421, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1423 = stablehlo.divide %v1422, %v1419 : tensor<32x512x7x7xf32>
    %v1424 = stablehlo.subtract %v1418, %v1423 : tensor<32x512x7x7xf32>
    %v1425 = stablehlo.multiply %v1424, %v1424 : tensor<32x512x7x7xf32>
    %v1426 = stablehlo.reduce(%v1425 init: %v1417) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1427 = stablehlo.broadcast_in_dim %v1426, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1428 = stablehlo.divide %v1427, %v1419 : tensor<32x512x7x7xf32>
    %v1429 = stablehlo.add %v1428, %v1420 : tensor<32x512x7x7xf32>
    %v1430 = stablehlo.rsqrt %v1429 : tensor<32x512x7x7xf32>
    %v1431 = stablehlo.multiply %v1424, %v1430 : tensor<32x512x7x7xf32>
    %v1432 = stablehlo.reshape %v1291 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1433 = stablehlo.multiply %v1432, %v1431 : tensor<32x512x7x7xf32>
    %v1434 = stablehlo.reduce(%v1433 init: %v1417) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1435 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1436 = stablehlo.multiply %v1434, %v1435 : tensor<512xf32>
    %v1437 = stablehlo.subtract %s4b0g2, %v1436 : tensor<512xf32>
    %v1438 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1439 = stablehlo.reshape %v1291 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1440 = stablehlo.reduce(%v1439 init: %v1438) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1441 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1442 = stablehlo.multiply %v1440, %v1441 : tensor<512xf32>
    %v1443 = stablehlo.subtract %s4b0bt2, %v1442 : tensor<512xf32>
    %v1444 = stablehlo.reshape %v1371 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1445 = stablehlo.reshape %v971 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1446 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1447 = stablehlo.compare GT, %v1445, %v1446 : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %v1448 = stablehlo.select %v1447, %v1444, %v1446 : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %v1449 = stablehlo.reshape %v1448 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1450 = stablehlo.reshape %v1449 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1451 = stablehlo.reshape %v922 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1452 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1453 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1454 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1455 = stablehlo.reduce(%v1451 init: %v1452) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1456 = stablehlo.broadcast_in_dim %v1455, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1457 = stablehlo.divide %v1456, %v1453 : tensor<32x512x7x7xf32>
    %v1458 = stablehlo.subtract %v1451, %v1457 : tensor<32x512x7x7xf32>
    %v1459 = stablehlo.multiply %v1458, %v1458 : tensor<32x512x7x7xf32>
    %v1460 = stablehlo.reduce(%v1459 init: %v1452) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1461 = stablehlo.broadcast_in_dim %v1460, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1462 = stablehlo.divide %v1461, %v1453 : tensor<32x512x7x7xf32>
    %v1463 = stablehlo.add %v1462, %v1454 : tensor<32x512x7x7xf32>
    %v1464 = stablehlo.rsqrt %v1463 : tensor<32x512x7x7xf32>
    %v1465 = stablehlo.multiply %v1458, %v1464 : tensor<32x512x7x7xf32>
    %v1466 = stablehlo.broadcast_in_dim %d4g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1467 = stablehlo.multiply %v1466, %v1450 : tensor<32x512x7x7xf32>
    %v1468 = stablehlo.reduce(%v1467 init: %v1452) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1469 = stablehlo.broadcast_in_dim %v1468, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1470 = stablehlo.multiply %v1465, %v1467 : tensor<32x512x7x7xf32>
    %v1471 = stablehlo.reduce(%v1470 init: %v1452) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1472 = stablehlo.broadcast_in_dim %v1471, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1473 = stablehlo.multiply %v1467, %v1453 : tensor<32x512x7x7xf32>
    %v1474 = stablehlo.subtract %v1473, %v1469 : tensor<32x512x7x7xf32>
    %v1475 = stablehlo.multiply %v1465, %v1472 : tensor<32x512x7x7xf32>
    %v1476 = stablehlo.subtract %v1474, %v1475 : tensor<32x512x7x7xf32>
    %v1477 = stablehlo.divide %v1464, %v1453 : tensor<32x512x7x7xf32>
    %v1478 = stablehlo.multiply %v1477, %v1476 : tensor<32x512x7x7xf32>
    %v1479 = stablehlo.reshape %v1478 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1480 = stablehlo.reshape %v1479 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1481 = stablehlo.transpose %d4W2, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1482 = stablehlo.reverse %v1481, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1483 = stablehlo.convolution(%v1480, %v1482)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1484 = stablehlo.reshape %v1483 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1485 = stablehlo.reshape %v1484 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1486 = stablehlo.reshape %v913 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1487 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1488 = stablehlo.compare GT, %v1486, %v1487 : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %v1489 = stablehlo.select %v1488, %v1485, %v1487 : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %v1490 = stablehlo.reshape %v1489 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1491 = stablehlo.reshape %v1490 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1492 = stablehlo.reshape %v893 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1493 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1494 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1495 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1496 = stablehlo.reduce(%v1492 init: %v1493) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1497 = stablehlo.broadcast_in_dim %v1496, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1498 = stablehlo.divide %v1497, %v1494 : tensor<32x512x7x7xf32>
    %v1499 = stablehlo.subtract %v1492, %v1498 : tensor<32x512x7x7xf32>
    %v1500 = stablehlo.multiply %v1499, %v1499 : tensor<32x512x7x7xf32>
    %v1501 = stablehlo.reduce(%v1500 init: %v1493) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1502 = stablehlo.broadcast_in_dim %v1501, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1503 = stablehlo.divide %v1502, %v1494 : tensor<32x512x7x7xf32>
    %v1504 = stablehlo.add %v1503, %v1495 : tensor<32x512x7x7xf32>
    %v1505 = stablehlo.rsqrt %v1504 : tensor<32x512x7x7xf32>
    %v1506 = stablehlo.multiply %v1499, %v1505 : tensor<32x512x7x7xf32>
    %v1507 = stablehlo.broadcast_in_dim %d4g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1508 = stablehlo.multiply %v1507, %v1491 : tensor<32x512x7x7xf32>
    %v1509 = stablehlo.reduce(%v1508 init: %v1493) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1510 = stablehlo.broadcast_in_dim %v1509, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1511 = stablehlo.multiply %v1506, %v1508 : tensor<32x512x7x7xf32>
    %v1512 = stablehlo.reduce(%v1511 init: %v1493) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1513 = stablehlo.broadcast_in_dim %v1512, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1514 = stablehlo.multiply %v1508, %v1494 : tensor<32x512x7x7xf32>
    %v1515 = stablehlo.subtract %v1514, %v1510 : tensor<32x512x7x7xf32>
    %v1516 = stablehlo.multiply %v1506, %v1513 : tensor<32x512x7x7xf32>
    %v1517 = stablehlo.subtract %v1515, %v1516 : tensor<32x512x7x7xf32>
    %v1518 = stablehlo.divide %v1505, %v1494 : tensor<32x512x7x7xf32>
    %v1519 = stablehlo.multiply %v1518, %v1517 : tensor<32x512x7x7xf32>
    %v1520 = stablehlo.reshape %v1519 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1521 = stablehlo.reshape %v1520 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1522 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1523 = stablehlo.pad %v1521, %v1522, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %v1524 = stablehlo.transpose %d4W1, dims = [1, 0, 2, 3] : (tensor<512x256x3x3xf32>) -> tensor<256x512x3x3xf32>
    %v1525 = stablehlo.reverse %v1524, dims = [2, 3] : tensor<256x512x3x3xf32>
    %v1526 = stablehlo.convolution(%v1523, %v1525)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<256x512x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1527 = stablehlo.reshape %v1526 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1528 = stablehlo.reshape %v1449 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1529 = stablehlo.reshape %v947 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1530 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1531 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1532 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1533 = stablehlo.reduce(%v1529 init: %v1530) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1534 = stablehlo.broadcast_in_dim %v1533, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1535 = stablehlo.divide %v1534, %v1531 : tensor<32x512x7x7xf32>
    %v1536 = stablehlo.subtract %v1529, %v1535 : tensor<32x512x7x7xf32>
    %v1537 = stablehlo.multiply %v1536, %v1536 : tensor<32x512x7x7xf32>
    %v1538 = stablehlo.reduce(%v1537 init: %v1530) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1539 = stablehlo.broadcast_in_dim %v1538, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1540 = stablehlo.divide %v1539, %v1531 : tensor<32x512x7x7xf32>
    %v1541 = stablehlo.add %v1540, %v1532 : tensor<32x512x7x7xf32>
    %v1542 = stablehlo.rsqrt %v1541 : tensor<32x512x7x7xf32>
    %v1543 = stablehlo.multiply %v1536, %v1542 : tensor<32x512x7x7xf32>
    %v1544 = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1545 = stablehlo.multiply %v1544, %v1528 : tensor<32x512x7x7xf32>
    %v1546 = stablehlo.reduce(%v1545 init: %v1530) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1547 = stablehlo.broadcast_in_dim %v1546, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1548 = stablehlo.multiply %v1543, %v1545 : tensor<32x512x7x7xf32>
    %v1549 = stablehlo.reduce(%v1548 init: %v1530) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1550 = stablehlo.broadcast_in_dim %v1549, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1551 = stablehlo.multiply %v1545, %v1531 : tensor<32x512x7x7xf32>
    %v1552 = stablehlo.subtract %v1551, %v1547 : tensor<32x512x7x7xf32>
    %v1553 = stablehlo.multiply %v1543, %v1550 : tensor<32x512x7x7xf32>
    %v1554 = stablehlo.subtract %v1552, %v1553 : tensor<32x512x7x7xf32>
    %v1555 = stablehlo.divide %v1542, %v1531 : tensor<32x512x7x7xf32>
    %v1556 = stablehlo.multiply %v1555, %v1554 : tensor<32x512x7x7xf32>
    %v1557 = stablehlo.reshape %v1556 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1558 = stablehlo.reshape %v1557 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1559 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1560 = stablehlo.pad %v1558, %v1559, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %v1561 = stablehlo.transpose %d4Wp, dims = [1, 0, 2, 3] : (tensor<512x256x1x1xf32>) -> tensor<256x512x1x1xf32>
    %v1562 = stablehlo.reverse %v1561, dims = [2, 3] : tensor<256x512x1x1xf32>
    %v1563 = stablehlo.convolution(%v1560, %v1562)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<256x512x1x1xf32>) -> tensor<32x256x14x14xf32>
    %v1564 = stablehlo.reshape %v1563 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1565 = stablehlo.reshape %v1527 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1566 = stablehlo.reshape %v1564 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1567 = stablehlo.add %v1565, %v1566 : tensor<32x256x14x14xf32>
    %v1568 = stablehlo.reshape %v1567 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1569 = stablehlo.reshape %v888 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1570 = stablehlo.reshape %v1520 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1571 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1572 = stablehlo.pad %v1570, %v1571, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %v1573 = stablehlo.transpose %v1569, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1574 = stablehlo.transpose %v1572, dims = [1, 0, 2, 3] : (tensor<32x512x14x14xf32>) -> tensor<512x32x14x14xf32>
    %v1575 = stablehlo.convolution(%v1573, %v1574)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<512x32x14x14xf32>) -> tensor<256x512x3x3xf32>
    %v1576 = stablehlo.transpose %v1575, dims = [1, 0, 2, 3] : (tensor<256x512x3x3xf32>) -> tensor<512x256x3x3xf32>
    %v1577 = stablehlo.constant dense<0.003125> : tensor<512x256x3x3xf32>
    %v1578 = stablehlo.multiply %v1576, %v1577 : tensor<512x256x3x3xf32>
    %v1579 = stablehlo.subtract %d4W1, %v1578 : tensor<512x256x3x3xf32>
    %v1580 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1581 = stablehlo.reshape %v893 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1582 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1583 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1584 = stablehlo.reduce(%v1581 init: %v1580) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1585 = stablehlo.broadcast_in_dim %v1584, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1586 = stablehlo.divide %v1585, %v1582 : tensor<32x512x7x7xf32>
    %v1587 = stablehlo.subtract %v1581, %v1586 : tensor<32x512x7x7xf32>
    %v1588 = stablehlo.multiply %v1587, %v1587 : tensor<32x512x7x7xf32>
    %v1589 = stablehlo.reduce(%v1588 init: %v1580) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1590 = stablehlo.broadcast_in_dim %v1589, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1591 = stablehlo.divide %v1590, %v1582 : tensor<32x512x7x7xf32>
    %v1592 = stablehlo.add %v1591, %v1583 : tensor<32x512x7x7xf32>
    %v1593 = stablehlo.rsqrt %v1592 : tensor<32x512x7x7xf32>
    %v1594 = stablehlo.multiply %v1587, %v1593 : tensor<32x512x7x7xf32>
    %v1595 = stablehlo.reshape %v1490 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1596 = stablehlo.multiply %v1595, %v1594 : tensor<32x512x7x7xf32>
    %v1597 = stablehlo.reduce(%v1596 init: %v1580) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1598 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1599 = stablehlo.multiply %v1597, %v1598 : tensor<512xf32>
    %v1600 = stablehlo.subtract %d4g1, %v1599 : tensor<512xf32>
    %v1601 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1602 = stablehlo.reshape %v1490 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1603 = stablehlo.reduce(%v1602 init: %v1601) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1604 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1605 = stablehlo.multiply %v1603, %v1604 : tensor<512xf32>
    %v1606 = stablehlo.subtract %d4bt1, %v1605 : tensor<512xf32>
    %v1607 = stablehlo.reshape %v917 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1608 = stablehlo.reshape %v1479 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1609 = stablehlo.transpose %v1607, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1610 = stablehlo.transpose %v1608, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1611 = stablehlo.convolution(%v1609, %v1610)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1612 = stablehlo.transpose %v1611, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1613 = stablehlo.constant dense<0.003125> : tensor<512x512x3x3xf32>
    %v1614 = stablehlo.multiply %v1612, %v1613 : tensor<512x512x3x3xf32>
    %v1615 = stablehlo.subtract %d4W2, %v1614 : tensor<512x512x3x3xf32>
    %v1616 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1617 = stablehlo.reshape %v922 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1618 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1619 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1620 = stablehlo.reduce(%v1617 init: %v1616) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1621 = stablehlo.broadcast_in_dim %v1620, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1622 = stablehlo.divide %v1621, %v1618 : tensor<32x512x7x7xf32>
    %v1623 = stablehlo.subtract %v1617, %v1622 : tensor<32x512x7x7xf32>
    %v1624 = stablehlo.multiply %v1623, %v1623 : tensor<32x512x7x7xf32>
    %v1625 = stablehlo.reduce(%v1624 init: %v1616) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1626 = stablehlo.broadcast_in_dim %v1625, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1627 = stablehlo.divide %v1626, %v1618 : tensor<32x512x7x7xf32>
    %v1628 = stablehlo.add %v1627, %v1619 : tensor<32x512x7x7xf32>
    %v1629 = stablehlo.rsqrt %v1628 : tensor<32x512x7x7xf32>
    %v1630 = stablehlo.multiply %v1623, %v1629 : tensor<32x512x7x7xf32>
    %v1631 = stablehlo.reshape %v1449 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1632 = stablehlo.multiply %v1631, %v1630 : tensor<32x512x7x7xf32>
    %v1633 = stablehlo.reduce(%v1632 init: %v1616) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1634 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1635 = stablehlo.multiply %v1633, %v1634 : tensor<512xf32>
    %v1636 = stablehlo.subtract %d4g2, %v1635 : tensor<512xf32>
    %v1637 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1638 = stablehlo.reshape %v1449 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1639 = stablehlo.reduce(%v1638 init: %v1637) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1640 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1641 = stablehlo.multiply %v1639, %v1640 : tensor<512xf32>
    %v1642 = stablehlo.subtract %d4bt2, %v1641 : tensor<512xf32>
    %v1643 = stablehlo.reshape %v888 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1644 = stablehlo.reshape %v1557 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1645 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1646 = stablehlo.pad %v1644, %v1645, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %v1647 = stablehlo.transpose %v1643, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1648 = stablehlo.transpose %v1646, dims = [1, 0, 2, 3] : (tensor<32x512x14x14xf32>) -> tensor<512x32x14x14xf32>
    %v1649 = stablehlo.convolution(%v1647, %v1648)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<512x32x14x14xf32>) -> tensor<256x512x1x1xf32>
    %v1650 = stablehlo.transpose %v1649, dims = [1, 0, 2, 3] : (tensor<256x512x1x1xf32>) -> tensor<512x256x1x1xf32>
    %v1651 = stablehlo.constant dense<0.003125> : tensor<512x256x1x1xf32>
    %v1652 = stablehlo.multiply %v1650, %v1651 : tensor<512x256x1x1xf32>
    %v1653 = stablehlo.subtract %d4Wp, %v1652 : tensor<512x256x1x1xf32>
    %v1654 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1655 = stablehlo.reshape %v947 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1656 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1657 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1658 = stablehlo.reduce(%v1655 init: %v1654) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1659 = stablehlo.broadcast_in_dim %v1658, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1660 = stablehlo.divide %v1659, %v1656 : tensor<32x512x7x7xf32>
    %v1661 = stablehlo.subtract %v1655, %v1660 : tensor<32x512x7x7xf32>
    %v1662 = stablehlo.multiply %v1661, %v1661 : tensor<32x512x7x7xf32>
    %v1663 = stablehlo.reduce(%v1662 init: %v1654) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1664 = stablehlo.broadcast_in_dim %v1663, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1665 = stablehlo.divide %v1664, %v1656 : tensor<32x512x7x7xf32>
    %v1666 = stablehlo.add %v1665, %v1657 : tensor<32x512x7x7xf32>
    %v1667 = stablehlo.rsqrt %v1666 : tensor<32x512x7x7xf32>
    %v1668 = stablehlo.multiply %v1661, %v1667 : tensor<32x512x7x7xf32>
    %v1669 = stablehlo.reshape %v1449 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1670 = stablehlo.multiply %v1669, %v1668 : tensor<32x512x7x7xf32>
    %v1671 = stablehlo.reduce(%v1670 init: %v1654) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1672 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1673 = stablehlo.multiply %v1671, %v1672 : tensor<512xf32>
    %v1674 = stablehlo.subtract %d4gp, %v1673 : tensor<512xf32>
    %v1675 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1676 = stablehlo.reshape %v1449 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1677 = stablehlo.reduce(%v1676 init: %v1675) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1678 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1679 = stablehlo.multiply %v1677, %v1678 : tensor<512xf32>
    %v1680 = stablehlo.subtract %d4btp, %v1679 : tensor<512xf32>
    %v1681 = stablehlo.reshape %v1568 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1682 = stablehlo.reshape %v884 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1683 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v1684 = stablehlo.compare GT, %v1682, %v1683 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v1685 = stablehlo.select %v1684, %v1681, %v1683 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v1686 = stablehlo.reshape %v1685 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1687 = stablehlo.reshape %v1686 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1688 = stablehlo.reshape %v860 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1689 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1690 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v1691 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1692 = stablehlo.reduce(%v1688 init: %v1689) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1693 = stablehlo.broadcast_in_dim %v1692, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1694 = stablehlo.divide %v1693, %v1690 : tensor<32x256x14x14xf32>
    %v1695 = stablehlo.subtract %v1688, %v1694 : tensor<32x256x14x14xf32>
    %v1696 = stablehlo.multiply %v1695, %v1695 : tensor<32x256x14x14xf32>
    %v1697 = stablehlo.reduce(%v1696 init: %v1689) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1698 = stablehlo.broadcast_in_dim %v1697, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1699 = stablehlo.divide %v1698, %v1690 : tensor<32x256x14x14xf32>
    %v1700 = stablehlo.add %v1699, %v1691 : tensor<32x256x14x14xf32>
    %v1701 = stablehlo.rsqrt %v1700 : tensor<32x256x14x14xf32>
    %v1702 = stablehlo.multiply %v1695, %v1701 : tensor<32x256x14x14xf32>
    %v1703 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1704 = stablehlo.multiply %v1703, %v1687 : tensor<32x256x14x14xf32>
    %v1705 = stablehlo.reduce(%v1704 init: %v1689) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1706 = stablehlo.broadcast_in_dim %v1705, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1707 = stablehlo.multiply %v1702, %v1704 : tensor<32x256x14x14xf32>
    %v1708 = stablehlo.reduce(%v1707 init: %v1689) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1709 = stablehlo.broadcast_in_dim %v1708, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1710 = stablehlo.multiply %v1704, %v1690 : tensor<32x256x14x14xf32>
    %v1711 = stablehlo.subtract %v1710, %v1706 : tensor<32x256x14x14xf32>
    %v1712 = stablehlo.multiply %v1702, %v1709 : tensor<32x256x14x14xf32>
    %v1713 = stablehlo.subtract %v1711, %v1712 : tensor<32x256x14x14xf32>
    %v1714 = stablehlo.divide %v1701, %v1690 : tensor<32x256x14x14xf32>
    %v1715 = stablehlo.multiply %v1714, %v1713 : tensor<32x256x14x14xf32>
    %v1716 = stablehlo.reshape %v1715 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1717 = stablehlo.reshape %v1716 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1718 = stablehlo.transpose %s3b4W2, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1719 = stablehlo.reverse %v1718, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1720 = stablehlo.convolution(%v1717, %v1719)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1721 = stablehlo.reshape %v1720 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1722 = stablehlo.reshape %v1721 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1723 = stablehlo.reshape %v851 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1724 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v1725 = stablehlo.compare GT, %v1723, %v1724 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v1726 = stablehlo.select %v1725, %v1722, %v1724 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v1727 = stablehlo.reshape %v1726 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1728 = stablehlo.reshape %v1727 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1729 = stablehlo.reshape %v831 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1730 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1731 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v1732 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1733 = stablehlo.reduce(%v1729 init: %v1730) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1734 = stablehlo.broadcast_in_dim %v1733, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1735 = stablehlo.divide %v1734, %v1731 : tensor<32x256x14x14xf32>
    %v1736 = stablehlo.subtract %v1729, %v1735 : tensor<32x256x14x14xf32>
    %v1737 = stablehlo.multiply %v1736, %v1736 : tensor<32x256x14x14xf32>
    %v1738 = stablehlo.reduce(%v1737 init: %v1730) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1739 = stablehlo.broadcast_in_dim %v1738, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1740 = stablehlo.divide %v1739, %v1731 : tensor<32x256x14x14xf32>
    %v1741 = stablehlo.add %v1740, %v1732 : tensor<32x256x14x14xf32>
    %v1742 = stablehlo.rsqrt %v1741 : tensor<32x256x14x14xf32>
    %v1743 = stablehlo.multiply %v1736, %v1742 : tensor<32x256x14x14xf32>
    %v1744 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1745 = stablehlo.multiply %v1744, %v1728 : tensor<32x256x14x14xf32>
    %v1746 = stablehlo.reduce(%v1745 init: %v1730) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1747 = stablehlo.broadcast_in_dim %v1746, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1748 = stablehlo.multiply %v1743, %v1745 : tensor<32x256x14x14xf32>
    %v1749 = stablehlo.reduce(%v1748 init: %v1730) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1750 = stablehlo.broadcast_in_dim %v1749, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1751 = stablehlo.multiply %v1745, %v1731 : tensor<32x256x14x14xf32>
    %v1752 = stablehlo.subtract %v1751, %v1747 : tensor<32x256x14x14xf32>
    %v1753 = stablehlo.multiply %v1743, %v1750 : tensor<32x256x14x14xf32>
    %v1754 = stablehlo.subtract %v1752, %v1753 : tensor<32x256x14x14xf32>
    %v1755 = stablehlo.divide %v1742, %v1731 : tensor<32x256x14x14xf32>
    %v1756 = stablehlo.multiply %v1755, %v1754 : tensor<32x256x14x14xf32>
    %v1757 = stablehlo.reshape %v1756 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1758 = stablehlo.reshape %v1757 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1759 = stablehlo.transpose %s3b4W1, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1760 = stablehlo.reverse %v1759, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1761 = stablehlo.convolution(%v1758, %v1760)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1762 = stablehlo.reshape %v1761 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1763 = stablehlo.reshape %v1762 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1764 = stablehlo.reshape %v1686 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1765 = stablehlo.add %v1763, %v1764 : tensor<32x256x14x14xf32>
    %v1766 = stablehlo.reshape %v1765 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1767 = stablehlo.reshape %v826 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1768 = stablehlo.reshape %v1757 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1769 = stablehlo.transpose %v1767, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1770 = stablehlo.transpose %v1768, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1771 = stablehlo.convolution(%v1769, %v1770)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1772 = stablehlo.transpose %v1771, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1773 = stablehlo.constant dense<0.003125> : tensor<256x256x3x3xf32>
    %v1774 = stablehlo.multiply %v1772, %v1773 : tensor<256x256x3x3xf32>
    %v1775 = stablehlo.subtract %s3b4W1, %v1774 : tensor<256x256x3x3xf32>
    %v1776 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1777 = stablehlo.reshape %v831 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1778 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v1779 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1780 = stablehlo.reduce(%v1777 init: %v1776) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1781 = stablehlo.broadcast_in_dim %v1780, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1782 = stablehlo.divide %v1781, %v1778 : tensor<32x256x14x14xf32>
    %v1783 = stablehlo.subtract %v1777, %v1782 : tensor<32x256x14x14xf32>
    %v1784 = stablehlo.multiply %v1783, %v1783 : tensor<32x256x14x14xf32>
    %v1785 = stablehlo.reduce(%v1784 init: %v1776) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1786 = stablehlo.broadcast_in_dim %v1785, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1787 = stablehlo.divide %v1786, %v1778 : tensor<32x256x14x14xf32>
    %v1788 = stablehlo.add %v1787, %v1779 : tensor<32x256x14x14xf32>
    %v1789 = stablehlo.rsqrt %v1788 : tensor<32x256x14x14xf32>
    %v1790 = stablehlo.multiply %v1783, %v1789 : tensor<32x256x14x14xf32>
    %v1791 = stablehlo.reshape %v1727 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1792 = stablehlo.multiply %v1791, %v1790 : tensor<32x256x14x14xf32>
    %v1793 = stablehlo.reduce(%v1792 init: %v1776) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1794 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v1795 = stablehlo.multiply %v1793, %v1794 : tensor<256xf32>
    %v1796 = stablehlo.subtract %s3b4g1, %v1795 : tensor<256xf32>
    %v1797 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1798 = stablehlo.reshape %v1727 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1799 = stablehlo.reduce(%v1798 init: %v1797) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1800 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v1801 = stablehlo.multiply %v1799, %v1800 : tensor<256xf32>
    %v1802 = stablehlo.subtract %s3b4bt1, %v1801 : tensor<256xf32>
    %v1803 = stablehlo.reshape %v855 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1804 = stablehlo.reshape %v1716 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1805 = stablehlo.transpose %v1803, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1806 = stablehlo.transpose %v1804, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1807 = stablehlo.convolution(%v1805, %v1806)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1808 = stablehlo.transpose %v1807, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1809 = stablehlo.constant dense<0.003125> : tensor<256x256x3x3xf32>
    %v1810 = stablehlo.multiply %v1808, %v1809 : tensor<256x256x3x3xf32>
    %v1811 = stablehlo.subtract %s3b4W2, %v1810 : tensor<256x256x3x3xf32>
    %v1812 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1813 = stablehlo.reshape %v860 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1814 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v1815 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1816 = stablehlo.reduce(%v1813 init: %v1812) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1817 = stablehlo.broadcast_in_dim %v1816, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1818 = stablehlo.divide %v1817, %v1814 : tensor<32x256x14x14xf32>
    %v1819 = stablehlo.subtract %v1813, %v1818 : tensor<32x256x14x14xf32>
    %v1820 = stablehlo.multiply %v1819, %v1819 : tensor<32x256x14x14xf32>
    %v1821 = stablehlo.reduce(%v1820 init: %v1812) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1822 = stablehlo.broadcast_in_dim %v1821, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1823 = stablehlo.divide %v1822, %v1814 : tensor<32x256x14x14xf32>
    %v1824 = stablehlo.add %v1823, %v1815 : tensor<32x256x14x14xf32>
    %v1825 = stablehlo.rsqrt %v1824 : tensor<32x256x14x14xf32>
    %v1826 = stablehlo.multiply %v1819, %v1825 : tensor<32x256x14x14xf32>
    %v1827 = stablehlo.reshape %v1686 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1828 = stablehlo.multiply %v1827, %v1826 : tensor<32x256x14x14xf32>
    %v1829 = stablehlo.reduce(%v1828 init: %v1812) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1830 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v1831 = stablehlo.multiply %v1829, %v1830 : tensor<256xf32>
    %v1832 = stablehlo.subtract %s3b4g2, %v1831 : tensor<256xf32>
    %v1833 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1834 = stablehlo.reshape %v1686 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1835 = stablehlo.reduce(%v1834 init: %v1833) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1836 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v1837 = stablehlo.multiply %v1835, %v1836 : tensor<256xf32>
    %v1838 = stablehlo.subtract %s3b4bt2, %v1837 : tensor<256xf32>
    %v1839 = stablehlo.reshape %v1766 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1840 = stablehlo.reshape %v822 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1841 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v1842 = stablehlo.compare GT, %v1840, %v1841 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v1843 = stablehlo.select %v1842, %v1839, %v1841 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v1844 = stablehlo.reshape %v1843 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1845 = stablehlo.reshape %v1844 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1846 = stablehlo.reshape %v798 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1847 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1848 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v1849 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1850 = stablehlo.reduce(%v1846 init: %v1847) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1851 = stablehlo.broadcast_in_dim %v1850, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1852 = stablehlo.divide %v1851, %v1848 : tensor<32x256x14x14xf32>
    %v1853 = stablehlo.subtract %v1846, %v1852 : tensor<32x256x14x14xf32>
    %v1854 = stablehlo.multiply %v1853, %v1853 : tensor<32x256x14x14xf32>
    %v1855 = stablehlo.reduce(%v1854 init: %v1847) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1856 = stablehlo.broadcast_in_dim %v1855, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1857 = stablehlo.divide %v1856, %v1848 : tensor<32x256x14x14xf32>
    %v1858 = stablehlo.add %v1857, %v1849 : tensor<32x256x14x14xf32>
    %v1859 = stablehlo.rsqrt %v1858 : tensor<32x256x14x14xf32>
    %v1860 = stablehlo.multiply %v1853, %v1859 : tensor<32x256x14x14xf32>
    %v1861 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1862 = stablehlo.multiply %v1861, %v1845 : tensor<32x256x14x14xf32>
    %v1863 = stablehlo.reduce(%v1862 init: %v1847) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1864 = stablehlo.broadcast_in_dim %v1863, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1865 = stablehlo.multiply %v1860, %v1862 : tensor<32x256x14x14xf32>
    %v1866 = stablehlo.reduce(%v1865 init: %v1847) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1867 = stablehlo.broadcast_in_dim %v1866, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1868 = stablehlo.multiply %v1862, %v1848 : tensor<32x256x14x14xf32>
    %v1869 = stablehlo.subtract %v1868, %v1864 : tensor<32x256x14x14xf32>
    %v1870 = stablehlo.multiply %v1860, %v1867 : tensor<32x256x14x14xf32>
    %v1871 = stablehlo.subtract %v1869, %v1870 : tensor<32x256x14x14xf32>
    %v1872 = stablehlo.divide %v1859, %v1848 : tensor<32x256x14x14xf32>
    %v1873 = stablehlo.multiply %v1872, %v1871 : tensor<32x256x14x14xf32>
    %v1874 = stablehlo.reshape %v1873 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1875 = stablehlo.reshape %v1874 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1876 = stablehlo.transpose %s3b3W2, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1877 = stablehlo.reverse %v1876, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1878 = stablehlo.convolution(%v1875, %v1877)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1879 = stablehlo.reshape %v1878 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1880 = stablehlo.reshape %v1879 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1881 = stablehlo.reshape %v789 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1882 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v1883 = stablehlo.compare GT, %v1881, %v1882 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v1884 = stablehlo.select %v1883, %v1880, %v1882 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v1885 = stablehlo.reshape %v1884 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1886 = stablehlo.reshape %v1885 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1887 = stablehlo.reshape %v769 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1888 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1889 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v1890 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1891 = stablehlo.reduce(%v1887 init: %v1888) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1892 = stablehlo.broadcast_in_dim %v1891, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1893 = stablehlo.divide %v1892, %v1889 : tensor<32x256x14x14xf32>
    %v1894 = stablehlo.subtract %v1887, %v1893 : tensor<32x256x14x14xf32>
    %v1895 = stablehlo.multiply %v1894, %v1894 : tensor<32x256x14x14xf32>
    %v1896 = stablehlo.reduce(%v1895 init: %v1888) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1897 = stablehlo.broadcast_in_dim %v1896, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1898 = stablehlo.divide %v1897, %v1889 : tensor<32x256x14x14xf32>
    %v1899 = stablehlo.add %v1898, %v1890 : tensor<32x256x14x14xf32>
    %v1900 = stablehlo.rsqrt %v1899 : tensor<32x256x14x14xf32>
    %v1901 = stablehlo.multiply %v1894, %v1900 : tensor<32x256x14x14xf32>
    %v1902 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1903 = stablehlo.multiply %v1902, %v1886 : tensor<32x256x14x14xf32>
    %v1904 = stablehlo.reduce(%v1903 init: %v1888) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1905 = stablehlo.broadcast_in_dim %v1904, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1906 = stablehlo.multiply %v1901, %v1903 : tensor<32x256x14x14xf32>
    %v1907 = stablehlo.reduce(%v1906 init: %v1888) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1908 = stablehlo.broadcast_in_dim %v1907, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1909 = stablehlo.multiply %v1903, %v1889 : tensor<32x256x14x14xf32>
    %v1910 = stablehlo.subtract %v1909, %v1905 : tensor<32x256x14x14xf32>
    %v1911 = stablehlo.multiply %v1901, %v1908 : tensor<32x256x14x14xf32>
    %v1912 = stablehlo.subtract %v1910, %v1911 : tensor<32x256x14x14xf32>
    %v1913 = stablehlo.divide %v1900, %v1889 : tensor<32x256x14x14xf32>
    %v1914 = stablehlo.multiply %v1913, %v1912 : tensor<32x256x14x14xf32>
    %v1915 = stablehlo.reshape %v1914 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1916 = stablehlo.reshape %v1915 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1917 = stablehlo.transpose %s3b3W1, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1918 = stablehlo.reverse %v1917, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1919 = stablehlo.convolution(%v1916, %v1918)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1920 = stablehlo.reshape %v1919 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1921 = stablehlo.reshape %v1920 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1922 = stablehlo.reshape %v1844 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1923 = stablehlo.add %v1921, %v1922 : tensor<32x256x14x14xf32>
    %v1924 = stablehlo.reshape %v1923 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1925 = stablehlo.reshape %v764 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1926 = stablehlo.reshape %v1915 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1927 = stablehlo.transpose %v1925, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1928 = stablehlo.transpose %v1926, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1929 = stablehlo.convolution(%v1927, %v1928)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1930 = stablehlo.transpose %v1929, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1931 = stablehlo.constant dense<0.003125> : tensor<256x256x3x3xf32>
    %v1932 = stablehlo.multiply %v1930, %v1931 : tensor<256x256x3x3xf32>
    %v1933 = stablehlo.subtract %s3b3W1, %v1932 : tensor<256x256x3x3xf32>
    %v1934 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1935 = stablehlo.reshape %v769 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1936 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v1937 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1938 = stablehlo.reduce(%v1935 init: %v1934) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1939 = stablehlo.broadcast_in_dim %v1938, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1940 = stablehlo.divide %v1939, %v1936 : tensor<32x256x14x14xf32>
    %v1941 = stablehlo.subtract %v1935, %v1940 : tensor<32x256x14x14xf32>
    %v1942 = stablehlo.multiply %v1941, %v1941 : tensor<32x256x14x14xf32>
    %v1943 = stablehlo.reduce(%v1942 init: %v1934) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1944 = stablehlo.broadcast_in_dim %v1943, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1945 = stablehlo.divide %v1944, %v1936 : tensor<32x256x14x14xf32>
    %v1946 = stablehlo.add %v1945, %v1937 : tensor<32x256x14x14xf32>
    %v1947 = stablehlo.rsqrt %v1946 : tensor<32x256x14x14xf32>
    %v1948 = stablehlo.multiply %v1941, %v1947 : tensor<32x256x14x14xf32>
    %v1949 = stablehlo.reshape %v1885 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1950 = stablehlo.multiply %v1949, %v1948 : tensor<32x256x14x14xf32>
    %v1951 = stablehlo.reduce(%v1950 init: %v1934) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1952 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v1953 = stablehlo.multiply %v1951, %v1952 : tensor<256xf32>
    %v1954 = stablehlo.subtract %s3b3g1, %v1953 : tensor<256xf32>
    %v1955 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1956 = stablehlo.reshape %v1885 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1957 = stablehlo.reduce(%v1956 init: %v1955) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1958 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v1959 = stablehlo.multiply %v1957, %v1958 : tensor<256xf32>
    %v1960 = stablehlo.subtract %s3b3bt1, %v1959 : tensor<256xf32>
    %v1961 = stablehlo.reshape %v793 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1962 = stablehlo.reshape %v1874 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1963 = stablehlo.transpose %v1961, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1964 = stablehlo.transpose %v1962, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1965 = stablehlo.convolution(%v1963, %v1964)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1966 = stablehlo.transpose %v1965, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1967 = stablehlo.constant dense<0.003125> : tensor<256x256x3x3xf32>
    %v1968 = stablehlo.multiply %v1966, %v1967 : tensor<256x256x3x3xf32>
    %v1969 = stablehlo.subtract %s3b3W2, %v1968 : tensor<256x256x3x3xf32>
    %v1970 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1971 = stablehlo.reshape %v798 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1972 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v1973 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1974 = stablehlo.reduce(%v1971 init: %v1970) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1975 = stablehlo.broadcast_in_dim %v1974, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1976 = stablehlo.divide %v1975, %v1972 : tensor<32x256x14x14xf32>
    %v1977 = stablehlo.subtract %v1971, %v1976 : tensor<32x256x14x14xf32>
    %v1978 = stablehlo.multiply %v1977, %v1977 : tensor<32x256x14x14xf32>
    %v1979 = stablehlo.reduce(%v1978 init: %v1970) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1980 = stablehlo.broadcast_in_dim %v1979, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1981 = stablehlo.divide %v1980, %v1972 : tensor<32x256x14x14xf32>
    %v1982 = stablehlo.add %v1981, %v1973 : tensor<32x256x14x14xf32>
    %v1983 = stablehlo.rsqrt %v1982 : tensor<32x256x14x14xf32>
    %v1984 = stablehlo.multiply %v1977, %v1983 : tensor<32x256x14x14xf32>
    %v1985 = stablehlo.reshape %v1844 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1986 = stablehlo.multiply %v1985, %v1984 : tensor<32x256x14x14xf32>
    %v1987 = stablehlo.reduce(%v1986 init: %v1970) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1988 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v1989 = stablehlo.multiply %v1987, %v1988 : tensor<256xf32>
    %v1990 = stablehlo.subtract %s3b3g2, %v1989 : tensor<256xf32>
    %v1991 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1992 = stablehlo.reshape %v1844 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1993 = stablehlo.reduce(%v1992 init: %v1991) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1994 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v1995 = stablehlo.multiply %v1993, %v1994 : tensor<256xf32>
    %v1996 = stablehlo.subtract %s3b3bt2, %v1995 : tensor<256xf32>
    %v1997 = stablehlo.reshape %v1924 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1998 = stablehlo.reshape %v760 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1999 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2000 = stablehlo.compare GT, %v1998, %v1999 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2001 = stablehlo.select %v2000, %v1997, %v1999 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2002 = stablehlo.reshape %v2001 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2003 = stablehlo.reshape %v2002 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2004 = stablehlo.reshape %v736 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2005 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2006 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2007 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2008 = stablehlo.reduce(%v2004 init: %v2005) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2009 = stablehlo.broadcast_in_dim %v2008, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2010 = stablehlo.divide %v2009, %v2006 : tensor<32x256x14x14xf32>
    %v2011 = stablehlo.subtract %v2004, %v2010 : tensor<32x256x14x14xf32>
    %v2012 = stablehlo.multiply %v2011, %v2011 : tensor<32x256x14x14xf32>
    %v2013 = stablehlo.reduce(%v2012 init: %v2005) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2014 = stablehlo.broadcast_in_dim %v2013, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2015 = stablehlo.divide %v2014, %v2006 : tensor<32x256x14x14xf32>
    %v2016 = stablehlo.add %v2015, %v2007 : tensor<32x256x14x14xf32>
    %v2017 = stablehlo.rsqrt %v2016 : tensor<32x256x14x14xf32>
    %v2018 = stablehlo.multiply %v2011, %v2017 : tensor<32x256x14x14xf32>
    %v2019 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2020 = stablehlo.multiply %v2019, %v2003 : tensor<32x256x14x14xf32>
    %v2021 = stablehlo.reduce(%v2020 init: %v2005) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2022 = stablehlo.broadcast_in_dim %v2021, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2023 = stablehlo.multiply %v2018, %v2020 : tensor<32x256x14x14xf32>
    %v2024 = stablehlo.reduce(%v2023 init: %v2005) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2025 = stablehlo.broadcast_in_dim %v2024, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2026 = stablehlo.multiply %v2020, %v2006 : tensor<32x256x14x14xf32>
    %v2027 = stablehlo.subtract %v2026, %v2022 : tensor<32x256x14x14xf32>
    %v2028 = stablehlo.multiply %v2018, %v2025 : tensor<32x256x14x14xf32>
    %v2029 = stablehlo.subtract %v2027, %v2028 : tensor<32x256x14x14xf32>
    %v2030 = stablehlo.divide %v2017, %v2006 : tensor<32x256x14x14xf32>
    %v2031 = stablehlo.multiply %v2030, %v2029 : tensor<32x256x14x14xf32>
    %v2032 = stablehlo.reshape %v2031 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2033 = stablehlo.reshape %v2032 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2034 = stablehlo.transpose %s3b2W2, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2035 = stablehlo.reverse %v2034, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2036 = stablehlo.convolution(%v2033, %v2035)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2037 = stablehlo.reshape %v2036 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2038 = stablehlo.reshape %v2037 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2039 = stablehlo.reshape %v727 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2040 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2041 = stablehlo.compare GT, %v2039, %v2040 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2042 = stablehlo.select %v2041, %v2038, %v2040 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2043 = stablehlo.reshape %v2042 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2044 = stablehlo.reshape %v2043 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2045 = stablehlo.reshape %v707 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2046 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2047 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2048 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2049 = stablehlo.reduce(%v2045 init: %v2046) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2050 = stablehlo.broadcast_in_dim %v2049, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2051 = stablehlo.divide %v2050, %v2047 : tensor<32x256x14x14xf32>
    %v2052 = stablehlo.subtract %v2045, %v2051 : tensor<32x256x14x14xf32>
    %v2053 = stablehlo.multiply %v2052, %v2052 : tensor<32x256x14x14xf32>
    %v2054 = stablehlo.reduce(%v2053 init: %v2046) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2055 = stablehlo.broadcast_in_dim %v2054, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2056 = stablehlo.divide %v2055, %v2047 : tensor<32x256x14x14xf32>
    %v2057 = stablehlo.add %v2056, %v2048 : tensor<32x256x14x14xf32>
    %v2058 = stablehlo.rsqrt %v2057 : tensor<32x256x14x14xf32>
    %v2059 = stablehlo.multiply %v2052, %v2058 : tensor<32x256x14x14xf32>
    %v2060 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2061 = stablehlo.multiply %v2060, %v2044 : tensor<32x256x14x14xf32>
    %v2062 = stablehlo.reduce(%v2061 init: %v2046) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2063 = stablehlo.broadcast_in_dim %v2062, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2064 = stablehlo.multiply %v2059, %v2061 : tensor<32x256x14x14xf32>
    %v2065 = stablehlo.reduce(%v2064 init: %v2046) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2066 = stablehlo.broadcast_in_dim %v2065, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2067 = stablehlo.multiply %v2061, %v2047 : tensor<32x256x14x14xf32>
    %v2068 = stablehlo.subtract %v2067, %v2063 : tensor<32x256x14x14xf32>
    %v2069 = stablehlo.multiply %v2059, %v2066 : tensor<32x256x14x14xf32>
    %v2070 = stablehlo.subtract %v2068, %v2069 : tensor<32x256x14x14xf32>
    %v2071 = stablehlo.divide %v2058, %v2047 : tensor<32x256x14x14xf32>
    %v2072 = stablehlo.multiply %v2071, %v2070 : tensor<32x256x14x14xf32>
    %v2073 = stablehlo.reshape %v2072 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2074 = stablehlo.reshape %v2073 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2075 = stablehlo.transpose %s3b2W1, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2076 = stablehlo.reverse %v2075, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2077 = stablehlo.convolution(%v2074, %v2076)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2078 = stablehlo.reshape %v2077 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2079 = stablehlo.reshape %v2078 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2080 = stablehlo.reshape %v2002 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2081 = stablehlo.add %v2079, %v2080 : tensor<32x256x14x14xf32>
    %v2082 = stablehlo.reshape %v2081 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2083 = stablehlo.reshape %v702 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2084 = stablehlo.reshape %v2073 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2085 = stablehlo.transpose %v2083, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2086 = stablehlo.transpose %v2084, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2087 = stablehlo.convolution(%v2085, %v2086)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2088 = stablehlo.transpose %v2087, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2089 = stablehlo.constant dense<0.003125> : tensor<256x256x3x3xf32>
    %v2090 = stablehlo.multiply %v2088, %v2089 : tensor<256x256x3x3xf32>
    %v2091 = stablehlo.subtract %s3b2W1, %v2090 : tensor<256x256x3x3xf32>
    %v2092 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2093 = stablehlo.reshape %v707 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2094 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2095 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2096 = stablehlo.reduce(%v2093 init: %v2092) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2097 = stablehlo.broadcast_in_dim %v2096, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2098 = stablehlo.divide %v2097, %v2094 : tensor<32x256x14x14xf32>
    %v2099 = stablehlo.subtract %v2093, %v2098 : tensor<32x256x14x14xf32>
    %v2100 = stablehlo.multiply %v2099, %v2099 : tensor<32x256x14x14xf32>
    %v2101 = stablehlo.reduce(%v2100 init: %v2092) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2102 = stablehlo.broadcast_in_dim %v2101, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2103 = stablehlo.divide %v2102, %v2094 : tensor<32x256x14x14xf32>
    %v2104 = stablehlo.add %v2103, %v2095 : tensor<32x256x14x14xf32>
    %v2105 = stablehlo.rsqrt %v2104 : tensor<32x256x14x14xf32>
    %v2106 = stablehlo.multiply %v2099, %v2105 : tensor<32x256x14x14xf32>
    %v2107 = stablehlo.reshape %v2043 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2108 = stablehlo.multiply %v2107, %v2106 : tensor<32x256x14x14xf32>
    %v2109 = stablehlo.reduce(%v2108 init: %v2092) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2110 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2111 = stablehlo.multiply %v2109, %v2110 : tensor<256xf32>
    %v2112 = stablehlo.subtract %s3b2g1, %v2111 : tensor<256xf32>
    %v2113 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2114 = stablehlo.reshape %v2043 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2115 = stablehlo.reduce(%v2114 init: %v2113) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2116 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2117 = stablehlo.multiply %v2115, %v2116 : tensor<256xf32>
    %v2118 = stablehlo.subtract %s3b2bt1, %v2117 : tensor<256xf32>
    %v2119 = stablehlo.reshape %v731 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2120 = stablehlo.reshape %v2032 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2121 = stablehlo.transpose %v2119, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2122 = stablehlo.transpose %v2120, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2123 = stablehlo.convolution(%v2121, %v2122)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2124 = stablehlo.transpose %v2123, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2125 = stablehlo.constant dense<0.003125> : tensor<256x256x3x3xf32>
    %v2126 = stablehlo.multiply %v2124, %v2125 : tensor<256x256x3x3xf32>
    %v2127 = stablehlo.subtract %s3b2W2, %v2126 : tensor<256x256x3x3xf32>
    %v2128 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2129 = stablehlo.reshape %v736 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2130 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2131 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2132 = stablehlo.reduce(%v2129 init: %v2128) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2133 = stablehlo.broadcast_in_dim %v2132, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2134 = stablehlo.divide %v2133, %v2130 : tensor<32x256x14x14xf32>
    %v2135 = stablehlo.subtract %v2129, %v2134 : tensor<32x256x14x14xf32>
    %v2136 = stablehlo.multiply %v2135, %v2135 : tensor<32x256x14x14xf32>
    %v2137 = stablehlo.reduce(%v2136 init: %v2128) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2138 = stablehlo.broadcast_in_dim %v2137, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2139 = stablehlo.divide %v2138, %v2130 : tensor<32x256x14x14xf32>
    %v2140 = stablehlo.add %v2139, %v2131 : tensor<32x256x14x14xf32>
    %v2141 = stablehlo.rsqrt %v2140 : tensor<32x256x14x14xf32>
    %v2142 = stablehlo.multiply %v2135, %v2141 : tensor<32x256x14x14xf32>
    %v2143 = stablehlo.reshape %v2002 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2144 = stablehlo.multiply %v2143, %v2142 : tensor<32x256x14x14xf32>
    %v2145 = stablehlo.reduce(%v2144 init: %v2128) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2146 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2147 = stablehlo.multiply %v2145, %v2146 : tensor<256xf32>
    %v2148 = stablehlo.subtract %s3b2g2, %v2147 : tensor<256xf32>
    %v2149 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2150 = stablehlo.reshape %v2002 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2151 = stablehlo.reduce(%v2150 init: %v2149) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2152 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2153 = stablehlo.multiply %v2151, %v2152 : tensor<256xf32>
    %v2154 = stablehlo.subtract %s3b2bt2, %v2153 : tensor<256xf32>
    %v2155 = stablehlo.reshape %v2082 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2156 = stablehlo.reshape %v698 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2157 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2158 = stablehlo.compare GT, %v2156, %v2157 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2159 = stablehlo.select %v2158, %v2155, %v2157 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2160 = stablehlo.reshape %v2159 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2161 = stablehlo.reshape %v2160 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2162 = stablehlo.reshape %v674 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2163 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2164 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2165 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2166 = stablehlo.reduce(%v2162 init: %v2163) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2167 = stablehlo.broadcast_in_dim %v2166, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2168 = stablehlo.divide %v2167, %v2164 : tensor<32x256x14x14xf32>
    %v2169 = stablehlo.subtract %v2162, %v2168 : tensor<32x256x14x14xf32>
    %v2170 = stablehlo.multiply %v2169, %v2169 : tensor<32x256x14x14xf32>
    %v2171 = stablehlo.reduce(%v2170 init: %v2163) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2172 = stablehlo.broadcast_in_dim %v2171, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2173 = stablehlo.divide %v2172, %v2164 : tensor<32x256x14x14xf32>
    %v2174 = stablehlo.add %v2173, %v2165 : tensor<32x256x14x14xf32>
    %v2175 = stablehlo.rsqrt %v2174 : tensor<32x256x14x14xf32>
    %v2176 = stablehlo.multiply %v2169, %v2175 : tensor<32x256x14x14xf32>
    %v2177 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2178 = stablehlo.multiply %v2177, %v2161 : tensor<32x256x14x14xf32>
    %v2179 = stablehlo.reduce(%v2178 init: %v2163) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2180 = stablehlo.broadcast_in_dim %v2179, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2181 = stablehlo.multiply %v2176, %v2178 : tensor<32x256x14x14xf32>
    %v2182 = stablehlo.reduce(%v2181 init: %v2163) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2183 = stablehlo.broadcast_in_dim %v2182, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2184 = stablehlo.multiply %v2178, %v2164 : tensor<32x256x14x14xf32>
    %v2185 = stablehlo.subtract %v2184, %v2180 : tensor<32x256x14x14xf32>
    %v2186 = stablehlo.multiply %v2176, %v2183 : tensor<32x256x14x14xf32>
    %v2187 = stablehlo.subtract %v2185, %v2186 : tensor<32x256x14x14xf32>
    %v2188 = stablehlo.divide %v2175, %v2164 : tensor<32x256x14x14xf32>
    %v2189 = stablehlo.multiply %v2188, %v2187 : tensor<32x256x14x14xf32>
    %v2190 = stablehlo.reshape %v2189 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2191 = stablehlo.reshape %v2190 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2192 = stablehlo.transpose %s3b1W2, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2193 = stablehlo.reverse %v2192, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2194 = stablehlo.convolution(%v2191, %v2193)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2195 = stablehlo.reshape %v2194 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2196 = stablehlo.reshape %v2195 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2197 = stablehlo.reshape %v665 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2198 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2199 = stablehlo.compare GT, %v2197, %v2198 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2200 = stablehlo.select %v2199, %v2196, %v2198 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2201 = stablehlo.reshape %v2200 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2202 = stablehlo.reshape %v2201 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2203 = stablehlo.reshape %v645 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2204 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2205 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2206 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2207 = stablehlo.reduce(%v2203 init: %v2204) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2208 = stablehlo.broadcast_in_dim %v2207, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2209 = stablehlo.divide %v2208, %v2205 : tensor<32x256x14x14xf32>
    %v2210 = stablehlo.subtract %v2203, %v2209 : tensor<32x256x14x14xf32>
    %v2211 = stablehlo.multiply %v2210, %v2210 : tensor<32x256x14x14xf32>
    %v2212 = stablehlo.reduce(%v2211 init: %v2204) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2213 = stablehlo.broadcast_in_dim %v2212, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2214 = stablehlo.divide %v2213, %v2205 : tensor<32x256x14x14xf32>
    %v2215 = stablehlo.add %v2214, %v2206 : tensor<32x256x14x14xf32>
    %v2216 = stablehlo.rsqrt %v2215 : tensor<32x256x14x14xf32>
    %v2217 = stablehlo.multiply %v2210, %v2216 : tensor<32x256x14x14xf32>
    %v2218 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2219 = stablehlo.multiply %v2218, %v2202 : tensor<32x256x14x14xf32>
    %v2220 = stablehlo.reduce(%v2219 init: %v2204) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2221 = stablehlo.broadcast_in_dim %v2220, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2222 = stablehlo.multiply %v2217, %v2219 : tensor<32x256x14x14xf32>
    %v2223 = stablehlo.reduce(%v2222 init: %v2204) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2224 = stablehlo.broadcast_in_dim %v2223, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2225 = stablehlo.multiply %v2219, %v2205 : tensor<32x256x14x14xf32>
    %v2226 = stablehlo.subtract %v2225, %v2221 : tensor<32x256x14x14xf32>
    %v2227 = stablehlo.multiply %v2217, %v2224 : tensor<32x256x14x14xf32>
    %v2228 = stablehlo.subtract %v2226, %v2227 : tensor<32x256x14x14xf32>
    %v2229 = stablehlo.divide %v2216, %v2205 : tensor<32x256x14x14xf32>
    %v2230 = stablehlo.multiply %v2229, %v2228 : tensor<32x256x14x14xf32>
    %v2231 = stablehlo.reshape %v2230 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2232 = stablehlo.reshape %v2231 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2233 = stablehlo.transpose %s3b1W1, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2234 = stablehlo.reverse %v2233, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2235 = stablehlo.convolution(%v2232, %v2234)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2236 = stablehlo.reshape %v2235 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2237 = stablehlo.reshape %v2236 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2238 = stablehlo.reshape %v2160 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2239 = stablehlo.add %v2237, %v2238 : tensor<32x256x14x14xf32>
    %v2240 = stablehlo.reshape %v2239 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2241 = stablehlo.reshape %v640 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2242 = stablehlo.reshape %v2231 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2243 = stablehlo.transpose %v2241, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2244 = stablehlo.transpose %v2242, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2245 = stablehlo.convolution(%v2243, %v2244)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2246 = stablehlo.transpose %v2245, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2247 = stablehlo.constant dense<0.003125> : tensor<256x256x3x3xf32>
    %v2248 = stablehlo.multiply %v2246, %v2247 : tensor<256x256x3x3xf32>
    %v2249 = stablehlo.subtract %s3b1W1, %v2248 : tensor<256x256x3x3xf32>
    %v2250 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2251 = stablehlo.reshape %v645 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2252 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2253 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2254 = stablehlo.reduce(%v2251 init: %v2250) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2255 = stablehlo.broadcast_in_dim %v2254, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2256 = stablehlo.divide %v2255, %v2252 : tensor<32x256x14x14xf32>
    %v2257 = stablehlo.subtract %v2251, %v2256 : tensor<32x256x14x14xf32>
    %v2258 = stablehlo.multiply %v2257, %v2257 : tensor<32x256x14x14xf32>
    %v2259 = stablehlo.reduce(%v2258 init: %v2250) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2260 = stablehlo.broadcast_in_dim %v2259, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2261 = stablehlo.divide %v2260, %v2252 : tensor<32x256x14x14xf32>
    %v2262 = stablehlo.add %v2261, %v2253 : tensor<32x256x14x14xf32>
    %v2263 = stablehlo.rsqrt %v2262 : tensor<32x256x14x14xf32>
    %v2264 = stablehlo.multiply %v2257, %v2263 : tensor<32x256x14x14xf32>
    %v2265 = stablehlo.reshape %v2201 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2266 = stablehlo.multiply %v2265, %v2264 : tensor<32x256x14x14xf32>
    %v2267 = stablehlo.reduce(%v2266 init: %v2250) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2268 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2269 = stablehlo.multiply %v2267, %v2268 : tensor<256xf32>
    %v2270 = stablehlo.subtract %s3b1g1, %v2269 : tensor<256xf32>
    %v2271 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2272 = stablehlo.reshape %v2201 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2273 = stablehlo.reduce(%v2272 init: %v2271) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2274 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2275 = stablehlo.multiply %v2273, %v2274 : tensor<256xf32>
    %v2276 = stablehlo.subtract %s3b1bt1, %v2275 : tensor<256xf32>
    %v2277 = stablehlo.reshape %v669 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2278 = stablehlo.reshape %v2190 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2279 = stablehlo.transpose %v2277, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2280 = stablehlo.transpose %v2278, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2281 = stablehlo.convolution(%v2279, %v2280)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2282 = stablehlo.transpose %v2281, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2283 = stablehlo.constant dense<0.003125> : tensor<256x256x3x3xf32>
    %v2284 = stablehlo.multiply %v2282, %v2283 : tensor<256x256x3x3xf32>
    %v2285 = stablehlo.subtract %s3b1W2, %v2284 : tensor<256x256x3x3xf32>
    %v2286 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2287 = stablehlo.reshape %v674 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2288 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2289 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2290 = stablehlo.reduce(%v2287 init: %v2286) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2291 = stablehlo.broadcast_in_dim %v2290, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2292 = stablehlo.divide %v2291, %v2288 : tensor<32x256x14x14xf32>
    %v2293 = stablehlo.subtract %v2287, %v2292 : tensor<32x256x14x14xf32>
    %v2294 = stablehlo.multiply %v2293, %v2293 : tensor<32x256x14x14xf32>
    %v2295 = stablehlo.reduce(%v2294 init: %v2286) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2296 = stablehlo.broadcast_in_dim %v2295, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2297 = stablehlo.divide %v2296, %v2288 : tensor<32x256x14x14xf32>
    %v2298 = stablehlo.add %v2297, %v2289 : tensor<32x256x14x14xf32>
    %v2299 = stablehlo.rsqrt %v2298 : tensor<32x256x14x14xf32>
    %v2300 = stablehlo.multiply %v2293, %v2299 : tensor<32x256x14x14xf32>
    %v2301 = stablehlo.reshape %v2160 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2302 = stablehlo.multiply %v2301, %v2300 : tensor<32x256x14x14xf32>
    %v2303 = stablehlo.reduce(%v2302 init: %v2286) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2304 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2305 = stablehlo.multiply %v2303, %v2304 : tensor<256xf32>
    %v2306 = stablehlo.subtract %s3b1g2, %v2305 : tensor<256xf32>
    %v2307 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2308 = stablehlo.reshape %v2160 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2309 = stablehlo.reduce(%v2308 init: %v2307) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2310 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2311 = stablehlo.multiply %v2309, %v2310 : tensor<256xf32>
    %v2312 = stablehlo.subtract %s3b1bt2, %v2311 : tensor<256xf32>
    %v2313 = stablehlo.reshape %v2240 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2314 = stablehlo.reshape %v636 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2315 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2316 = stablehlo.compare GT, %v2314, %v2315 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2317 = stablehlo.select %v2316, %v2313, %v2315 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2318 = stablehlo.reshape %v2317 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2319 = stablehlo.reshape %v2318 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2320 = stablehlo.reshape %v612 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2321 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2322 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2323 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2324 = stablehlo.reduce(%v2320 init: %v2321) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2325 = stablehlo.broadcast_in_dim %v2324, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2326 = stablehlo.divide %v2325, %v2322 : tensor<32x256x14x14xf32>
    %v2327 = stablehlo.subtract %v2320, %v2326 : tensor<32x256x14x14xf32>
    %v2328 = stablehlo.multiply %v2327, %v2327 : tensor<32x256x14x14xf32>
    %v2329 = stablehlo.reduce(%v2328 init: %v2321) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2330 = stablehlo.broadcast_in_dim %v2329, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2331 = stablehlo.divide %v2330, %v2322 : tensor<32x256x14x14xf32>
    %v2332 = stablehlo.add %v2331, %v2323 : tensor<32x256x14x14xf32>
    %v2333 = stablehlo.rsqrt %v2332 : tensor<32x256x14x14xf32>
    %v2334 = stablehlo.multiply %v2327, %v2333 : tensor<32x256x14x14xf32>
    %v2335 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2336 = stablehlo.multiply %v2335, %v2319 : tensor<32x256x14x14xf32>
    %v2337 = stablehlo.reduce(%v2336 init: %v2321) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2338 = stablehlo.broadcast_in_dim %v2337, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2339 = stablehlo.multiply %v2334, %v2336 : tensor<32x256x14x14xf32>
    %v2340 = stablehlo.reduce(%v2339 init: %v2321) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2341 = stablehlo.broadcast_in_dim %v2340, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2342 = stablehlo.multiply %v2336, %v2322 : tensor<32x256x14x14xf32>
    %v2343 = stablehlo.subtract %v2342, %v2338 : tensor<32x256x14x14xf32>
    %v2344 = stablehlo.multiply %v2334, %v2341 : tensor<32x256x14x14xf32>
    %v2345 = stablehlo.subtract %v2343, %v2344 : tensor<32x256x14x14xf32>
    %v2346 = stablehlo.divide %v2333, %v2322 : tensor<32x256x14x14xf32>
    %v2347 = stablehlo.multiply %v2346, %v2345 : tensor<32x256x14x14xf32>
    %v2348 = stablehlo.reshape %v2347 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2349 = stablehlo.reshape %v2348 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2350 = stablehlo.transpose %s3b0W2, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2351 = stablehlo.reverse %v2350, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2352 = stablehlo.convolution(%v2349, %v2351)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2353 = stablehlo.reshape %v2352 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2354 = stablehlo.reshape %v2353 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2355 = stablehlo.reshape %v603 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2356 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2357 = stablehlo.compare GT, %v2355, %v2356 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2358 = stablehlo.select %v2357, %v2354, %v2356 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2359 = stablehlo.reshape %v2358 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2360 = stablehlo.reshape %v2359 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2361 = stablehlo.reshape %v583 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2362 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2363 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2364 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2365 = stablehlo.reduce(%v2361 init: %v2362) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2366 = stablehlo.broadcast_in_dim %v2365, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2367 = stablehlo.divide %v2366, %v2363 : tensor<32x256x14x14xf32>
    %v2368 = stablehlo.subtract %v2361, %v2367 : tensor<32x256x14x14xf32>
    %v2369 = stablehlo.multiply %v2368, %v2368 : tensor<32x256x14x14xf32>
    %v2370 = stablehlo.reduce(%v2369 init: %v2362) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2371 = stablehlo.broadcast_in_dim %v2370, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2372 = stablehlo.divide %v2371, %v2363 : tensor<32x256x14x14xf32>
    %v2373 = stablehlo.add %v2372, %v2364 : tensor<32x256x14x14xf32>
    %v2374 = stablehlo.rsqrt %v2373 : tensor<32x256x14x14xf32>
    %v2375 = stablehlo.multiply %v2368, %v2374 : tensor<32x256x14x14xf32>
    %v2376 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2377 = stablehlo.multiply %v2376, %v2360 : tensor<32x256x14x14xf32>
    %v2378 = stablehlo.reduce(%v2377 init: %v2362) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2379 = stablehlo.broadcast_in_dim %v2378, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2380 = stablehlo.multiply %v2375, %v2377 : tensor<32x256x14x14xf32>
    %v2381 = stablehlo.reduce(%v2380 init: %v2362) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2382 = stablehlo.broadcast_in_dim %v2381, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2383 = stablehlo.multiply %v2377, %v2363 : tensor<32x256x14x14xf32>
    %v2384 = stablehlo.subtract %v2383, %v2379 : tensor<32x256x14x14xf32>
    %v2385 = stablehlo.multiply %v2375, %v2382 : tensor<32x256x14x14xf32>
    %v2386 = stablehlo.subtract %v2384, %v2385 : tensor<32x256x14x14xf32>
    %v2387 = stablehlo.divide %v2374, %v2363 : tensor<32x256x14x14xf32>
    %v2388 = stablehlo.multiply %v2387, %v2386 : tensor<32x256x14x14xf32>
    %v2389 = stablehlo.reshape %v2388 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2390 = stablehlo.reshape %v2389 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2391 = stablehlo.transpose %s3b0W1, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2392 = stablehlo.reverse %v2391, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2393 = stablehlo.convolution(%v2390, %v2392)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2394 = stablehlo.reshape %v2393 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2395 = stablehlo.reshape %v2394 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2396 = stablehlo.reshape %v2318 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2397 = stablehlo.add %v2395, %v2396 : tensor<32x256x14x14xf32>
    %v2398 = stablehlo.reshape %v2397 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2399 = stablehlo.reshape %v578 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2400 = stablehlo.reshape %v2389 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2401 = stablehlo.transpose %v2399, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2402 = stablehlo.transpose %v2400, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2403 = stablehlo.convolution(%v2401, %v2402)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2404 = stablehlo.transpose %v2403, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2405 = stablehlo.constant dense<0.003125> : tensor<256x256x3x3xf32>
    %v2406 = stablehlo.multiply %v2404, %v2405 : tensor<256x256x3x3xf32>
    %v2407 = stablehlo.subtract %s3b0W1, %v2406 : tensor<256x256x3x3xf32>
    %v2408 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2409 = stablehlo.reshape %v583 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2410 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2411 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2412 = stablehlo.reduce(%v2409 init: %v2408) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2413 = stablehlo.broadcast_in_dim %v2412, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2414 = stablehlo.divide %v2413, %v2410 : tensor<32x256x14x14xf32>
    %v2415 = stablehlo.subtract %v2409, %v2414 : tensor<32x256x14x14xf32>
    %v2416 = stablehlo.multiply %v2415, %v2415 : tensor<32x256x14x14xf32>
    %v2417 = stablehlo.reduce(%v2416 init: %v2408) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2418 = stablehlo.broadcast_in_dim %v2417, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2419 = stablehlo.divide %v2418, %v2410 : tensor<32x256x14x14xf32>
    %v2420 = stablehlo.add %v2419, %v2411 : tensor<32x256x14x14xf32>
    %v2421 = stablehlo.rsqrt %v2420 : tensor<32x256x14x14xf32>
    %v2422 = stablehlo.multiply %v2415, %v2421 : tensor<32x256x14x14xf32>
    %v2423 = stablehlo.reshape %v2359 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2424 = stablehlo.multiply %v2423, %v2422 : tensor<32x256x14x14xf32>
    %v2425 = stablehlo.reduce(%v2424 init: %v2408) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2426 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2427 = stablehlo.multiply %v2425, %v2426 : tensor<256xf32>
    %v2428 = stablehlo.subtract %s3b0g1, %v2427 : tensor<256xf32>
    %v2429 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2430 = stablehlo.reshape %v2359 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2431 = stablehlo.reduce(%v2430 init: %v2429) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2432 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2433 = stablehlo.multiply %v2431, %v2432 : tensor<256xf32>
    %v2434 = stablehlo.subtract %s3b0bt1, %v2433 : tensor<256xf32>
    %v2435 = stablehlo.reshape %v607 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2436 = stablehlo.reshape %v2348 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2437 = stablehlo.transpose %v2435, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2438 = stablehlo.transpose %v2436, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2439 = stablehlo.convolution(%v2437, %v2438)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2440 = stablehlo.transpose %v2439, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2441 = stablehlo.constant dense<0.003125> : tensor<256x256x3x3xf32>
    %v2442 = stablehlo.multiply %v2440, %v2441 : tensor<256x256x3x3xf32>
    %v2443 = stablehlo.subtract %s3b0W2, %v2442 : tensor<256x256x3x3xf32>
    %v2444 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2445 = stablehlo.reshape %v612 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2446 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2447 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2448 = stablehlo.reduce(%v2445 init: %v2444) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2449 = stablehlo.broadcast_in_dim %v2448, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2450 = stablehlo.divide %v2449, %v2446 : tensor<32x256x14x14xf32>
    %v2451 = stablehlo.subtract %v2445, %v2450 : tensor<32x256x14x14xf32>
    %v2452 = stablehlo.multiply %v2451, %v2451 : tensor<32x256x14x14xf32>
    %v2453 = stablehlo.reduce(%v2452 init: %v2444) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2454 = stablehlo.broadcast_in_dim %v2453, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2455 = stablehlo.divide %v2454, %v2446 : tensor<32x256x14x14xf32>
    %v2456 = stablehlo.add %v2455, %v2447 : tensor<32x256x14x14xf32>
    %v2457 = stablehlo.rsqrt %v2456 : tensor<32x256x14x14xf32>
    %v2458 = stablehlo.multiply %v2451, %v2457 : tensor<32x256x14x14xf32>
    %v2459 = stablehlo.reshape %v2318 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2460 = stablehlo.multiply %v2459, %v2458 : tensor<32x256x14x14xf32>
    %v2461 = stablehlo.reduce(%v2460 init: %v2444) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2462 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2463 = stablehlo.multiply %v2461, %v2462 : tensor<256xf32>
    %v2464 = stablehlo.subtract %s3b0g2, %v2463 : tensor<256xf32>
    %v2465 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2466 = stablehlo.reshape %v2318 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2467 = stablehlo.reduce(%v2466 init: %v2465) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2468 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2469 = stablehlo.multiply %v2467, %v2468 : tensor<256xf32>
    %v2470 = stablehlo.subtract %s3b0bt2, %v2469 : tensor<256xf32>
    %v2471 = stablehlo.reshape %v2398 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2472 = stablehlo.reshape %v574 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2473 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2474 = stablehlo.compare GT, %v2472, %v2473 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2475 = stablehlo.select %v2474, %v2471, %v2473 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2476 = stablehlo.reshape %v2475 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2477 = stablehlo.reshape %v2476 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2478 = stablehlo.reshape %v525 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2479 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2480 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2481 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2482 = stablehlo.reduce(%v2478 init: %v2479) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2483 = stablehlo.broadcast_in_dim %v2482, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2484 = stablehlo.divide %v2483, %v2480 : tensor<32x256x14x14xf32>
    %v2485 = stablehlo.subtract %v2478, %v2484 : tensor<32x256x14x14xf32>
    %v2486 = stablehlo.multiply %v2485, %v2485 : tensor<32x256x14x14xf32>
    %v2487 = stablehlo.reduce(%v2486 init: %v2479) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2488 = stablehlo.broadcast_in_dim %v2487, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2489 = stablehlo.divide %v2488, %v2480 : tensor<32x256x14x14xf32>
    %v2490 = stablehlo.add %v2489, %v2481 : tensor<32x256x14x14xf32>
    %v2491 = stablehlo.rsqrt %v2490 : tensor<32x256x14x14xf32>
    %v2492 = stablehlo.multiply %v2485, %v2491 : tensor<32x256x14x14xf32>
    %v2493 = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2494 = stablehlo.multiply %v2493, %v2477 : tensor<32x256x14x14xf32>
    %v2495 = stablehlo.reduce(%v2494 init: %v2479) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2496 = stablehlo.broadcast_in_dim %v2495, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2497 = stablehlo.multiply %v2492, %v2494 : tensor<32x256x14x14xf32>
    %v2498 = stablehlo.reduce(%v2497 init: %v2479) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2499 = stablehlo.broadcast_in_dim %v2498, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2500 = stablehlo.multiply %v2494, %v2480 : tensor<32x256x14x14xf32>
    %v2501 = stablehlo.subtract %v2500, %v2496 : tensor<32x256x14x14xf32>
    %v2502 = stablehlo.multiply %v2492, %v2499 : tensor<32x256x14x14xf32>
    %v2503 = stablehlo.subtract %v2501, %v2502 : tensor<32x256x14x14xf32>
    %v2504 = stablehlo.divide %v2491, %v2480 : tensor<32x256x14x14xf32>
    %v2505 = stablehlo.multiply %v2504, %v2503 : tensor<32x256x14x14xf32>
    %v2506 = stablehlo.reshape %v2505 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2507 = stablehlo.reshape %v2506 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2508 = stablehlo.transpose %d3W2, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2509 = stablehlo.reverse %v2508, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2510 = stablehlo.convolution(%v2507, %v2509)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2511 = stablehlo.reshape %v2510 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2512 = stablehlo.reshape %v2511 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2513 = stablehlo.reshape %v516 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2514 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2515 = stablehlo.compare GT, %v2513, %v2514 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2516 = stablehlo.select %v2515, %v2512, %v2514 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2517 = stablehlo.reshape %v2516 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2518 = stablehlo.reshape %v2517 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2519 = stablehlo.reshape %v496 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2520 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2521 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2522 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2523 = stablehlo.reduce(%v2519 init: %v2520) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2524 = stablehlo.broadcast_in_dim %v2523, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2525 = stablehlo.divide %v2524, %v2521 : tensor<32x256x14x14xf32>
    %v2526 = stablehlo.subtract %v2519, %v2525 : tensor<32x256x14x14xf32>
    %v2527 = stablehlo.multiply %v2526, %v2526 : tensor<32x256x14x14xf32>
    %v2528 = stablehlo.reduce(%v2527 init: %v2520) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2529 = stablehlo.broadcast_in_dim %v2528, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2530 = stablehlo.divide %v2529, %v2521 : tensor<32x256x14x14xf32>
    %v2531 = stablehlo.add %v2530, %v2522 : tensor<32x256x14x14xf32>
    %v2532 = stablehlo.rsqrt %v2531 : tensor<32x256x14x14xf32>
    %v2533 = stablehlo.multiply %v2526, %v2532 : tensor<32x256x14x14xf32>
    %v2534 = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2535 = stablehlo.multiply %v2534, %v2518 : tensor<32x256x14x14xf32>
    %v2536 = stablehlo.reduce(%v2535 init: %v2520) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2537 = stablehlo.broadcast_in_dim %v2536, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2538 = stablehlo.multiply %v2533, %v2535 : tensor<32x256x14x14xf32>
    %v2539 = stablehlo.reduce(%v2538 init: %v2520) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2540 = stablehlo.broadcast_in_dim %v2539, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2541 = stablehlo.multiply %v2535, %v2521 : tensor<32x256x14x14xf32>
    %v2542 = stablehlo.subtract %v2541, %v2537 : tensor<32x256x14x14xf32>
    %v2543 = stablehlo.multiply %v2533, %v2540 : tensor<32x256x14x14xf32>
    %v2544 = stablehlo.subtract %v2542, %v2543 : tensor<32x256x14x14xf32>
    %v2545 = stablehlo.divide %v2532, %v2521 : tensor<32x256x14x14xf32>
    %v2546 = stablehlo.multiply %v2545, %v2544 : tensor<32x256x14x14xf32>
    %v2547 = stablehlo.reshape %v2546 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2548 = stablehlo.reshape %v2547 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2549 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2550 = stablehlo.pad %v2548, %v2549, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %v2551 = stablehlo.transpose %d3W1, dims = [1, 0, 2, 3] : (tensor<256x128x3x3xf32>) -> tensor<128x256x3x3xf32>
    %v2552 = stablehlo.reverse %v2551, dims = [2, 3] : tensor<128x256x3x3xf32>
    %v2553 = stablehlo.convolution(%v2550, %v2552)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x28x28xf32>, tensor<128x256x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2554 = stablehlo.reshape %v2553 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2555 = stablehlo.reshape %v2476 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2556 = stablehlo.reshape %v550 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2557 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2558 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2559 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2560 = stablehlo.reduce(%v2556 init: %v2557) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2561 = stablehlo.broadcast_in_dim %v2560, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2562 = stablehlo.divide %v2561, %v2558 : tensor<32x256x14x14xf32>
    %v2563 = stablehlo.subtract %v2556, %v2562 : tensor<32x256x14x14xf32>
    %v2564 = stablehlo.multiply %v2563, %v2563 : tensor<32x256x14x14xf32>
    %v2565 = stablehlo.reduce(%v2564 init: %v2557) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2566 = stablehlo.broadcast_in_dim %v2565, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2567 = stablehlo.divide %v2566, %v2558 : tensor<32x256x14x14xf32>
    %v2568 = stablehlo.add %v2567, %v2559 : tensor<32x256x14x14xf32>
    %v2569 = stablehlo.rsqrt %v2568 : tensor<32x256x14x14xf32>
    %v2570 = stablehlo.multiply %v2563, %v2569 : tensor<32x256x14x14xf32>
    %v2571 = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2572 = stablehlo.multiply %v2571, %v2555 : tensor<32x256x14x14xf32>
    %v2573 = stablehlo.reduce(%v2572 init: %v2557) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2574 = stablehlo.broadcast_in_dim %v2573, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2575 = stablehlo.multiply %v2570, %v2572 : tensor<32x256x14x14xf32>
    %v2576 = stablehlo.reduce(%v2575 init: %v2557) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2577 = stablehlo.broadcast_in_dim %v2576, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2578 = stablehlo.multiply %v2572, %v2558 : tensor<32x256x14x14xf32>
    %v2579 = stablehlo.subtract %v2578, %v2574 : tensor<32x256x14x14xf32>
    %v2580 = stablehlo.multiply %v2570, %v2577 : tensor<32x256x14x14xf32>
    %v2581 = stablehlo.subtract %v2579, %v2580 : tensor<32x256x14x14xf32>
    %v2582 = stablehlo.divide %v2569, %v2558 : tensor<32x256x14x14xf32>
    %v2583 = stablehlo.multiply %v2582, %v2581 : tensor<32x256x14x14xf32>
    %v2584 = stablehlo.reshape %v2583 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2585 = stablehlo.reshape %v2584 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2586 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2587 = stablehlo.pad %v2585, %v2586, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %v2588 = stablehlo.transpose %d3Wp, dims = [1, 0, 2, 3] : (tensor<256x128x1x1xf32>) -> tensor<128x256x1x1xf32>
    %v2589 = stablehlo.reverse %v2588, dims = [2, 3] : tensor<128x256x1x1xf32>
    %v2590 = stablehlo.convolution(%v2587, %v2589)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x28x28xf32>, tensor<128x256x1x1xf32>) -> tensor<32x128x28x28xf32>
    %v2591 = stablehlo.reshape %v2590 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2592 = stablehlo.reshape %v2554 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2593 = stablehlo.reshape %v2591 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2594 = stablehlo.add %v2592, %v2593 : tensor<32x128x28x28xf32>
    %v2595 = stablehlo.reshape %v2594 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2596 = stablehlo.reshape %v491 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2597 = stablehlo.reshape %v2547 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2598 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2599 = stablehlo.pad %v2597, %v2598, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %v2600 = stablehlo.transpose %v2596, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2601 = stablehlo.transpose %v2599, dims = [1, 0, 2, 3] : (tensor<32x256x28x28xf32>) -> tensor<256x32x28x28xf32>
    %v2602 = stablehlo.convolution(%v2600, %v2601)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<256x32x28x28xf32>) -> tensor<128x256x3x3xf32>
    %v2603 = stablehlo.transpose %v2602, dims = [1, 0, 2, 3] : (tensor<128x256x3x3xf32>) -> tensor<256x128x3x3xf32>
    %v2604 = stablehlo.constant dense<0.003125> : tensor<256x128x3x3xf32>
    %v2605 = stablehlo.multiply %v2603, %v2604 : tensor<256x128x3x3xf32>
    %v2606 = stablehlo.subtract %d3W1, %v2605 : tensor<256x128x3x3xf32>
    %v2607 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2608 = stablehlo.reshape %v496 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2609 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2610 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2611 = stablehlo.reduce(%v2608 init: %v2607) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2612 = stablehlo.broadcast_in_dim %v2611, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2613 = stablehlo.divide %v2612, %v2609 : tensor<32x256x14x14xf32>
    %v2614 = stablehlo.subtract %v2608, %v2613 : tensor<32x256x14x14xf32>
    %v2615 = stablehlo.multiply %v2614, %v2614 : tensor<32x256x14x14xf32>
    %v2616 = stablehlo.reduce(%v2615 init: %v2607) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2617 = stablehlo.broadcast_in_dim %v2616, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2618 = stablehlo.divide %v2617, %v2609 : tensor<32x256x14x14xf32>
    %v2619 = stablehlo.add %v2618, %v2610 : tensor<32x256x14x14xf32>
    %v2620 = stablehlo.rsqrt %v2619 : tensor<32x256x14x14xf32>
    %v2621 = stablehlo.multiply %v2614, %v2620 : tensor<32x256x14x14xf32>
    %v2622 = stablehlo.reshape %v2517 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2623 = stablehlo.multiply %v2622, %v2621 : tensor<32x256x14x14xf32>
    %v2624 = stablehlo.reduce(%v2623 init: %v2607) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2625 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2626 = stablehlo.multiply %v2624, %v2625 : tensor<256xf32>
    %v2627 = stablehlo.subtract %d3g1, %v2626 : tensor<256xf32>
    %v2628 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2629 = stablehlo.reshape %v2517 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2630 = stablehlo.reduce(%v2629 init: %v2628) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2631 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2632 = stablehlo.multiply %v2630, %v2631 : tensor<256xf32>
    %v2633 = stablehlo.subtract %d3bt1, %v2632 : tensor<256xf32>
    %v2634 = stablehlo.reshape %v520 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2635 = stablehlo.reshape %v2506 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2636 = stablehlo.transpose %v2634, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2637 = stablehlo.transpose %v2635, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2638 = stablehlo.convolution(%v2636, %v2637)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2639 = stablehlo.transpose %v2638, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2640 = stablehlo.constant dense<0.003125> : tensor<256x256x3x3xf32>
    %v2641 = stablehlo.multiply %v2639, %v2640 : tensor<256x256x3x3xf32>
    %v2642 = stablehlo.subtract %d3W2, %v2641 : tensor<256x256x3x3xf32>
    %v2643 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2644 = stablehlo.reshape %v525 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2645 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2646 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2647 = stablehlo.reduce(%v2644 init: %v2643) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2648 = stablehlo.broadcast_in_dim %v2647, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2649 = stablehlo.divide %v2648, %v2645 : tensor<32x256x14x14xf32>
    %v2650 = stablehlo.subtract %v2644, %v2649 : tensor<32x256x14x14xf32>
    %v2651 = stablehlo.multiply %v2650, %v2650 : tensor<32x256x14x14xf32>
    %v2652 = stablehlo.reduce(%v2651 init: %v2643) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2653 = stablehlo.broadcast_in_dim %v2652, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2654 = stablehlo.divide %v2653, %v2645 : tensor<32x256x14x14xf32>
    %v2655 = stablehlo.add %v2654, %v2646 : tensor<32x256x14x14xf32>
    %v2656 = stablehlo.rsqrt %v2655 : tensor<32x256x14x14xf32>
    %v2657 = stablehlo.multiply %v2650, %v2656 : tensor<32x256x14x14xf32>
    %v2658 = stablehlo.reshape %v2476 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2659 = stablehlo.multiply %v2658, %v2657 : tensor<32x256x14x14xf32>
    %v2660 = stablehlo.reduce(%v2659 init: %v2643) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2661 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2662 = stablehlo.multiply %v2660, %v2661 : tensor<256xf32>
    %v2663 = stablehlo.subtract %d3g2, %v2662 : tensor<256xf32>
    %v2664 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2665 = stablehlo.reshape %v2476 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2666 = stablehlo.reduce(%v2665 init: %v2664) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2667 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2668 = stablehlo.multiply %v2666, %v2667 : tensor<256xf32>
    %v2669 = stablehlo.subtract %d3bt2, %v2668 : tensor<256xf32>
    %v2670 = stablehlo.reshape %v491 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2671 = stablehlo.reshape %v2584 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2672 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2673 = stablehlo.pad %v2671, %v2672, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %v2674 = stablehlo.transpose %v2670, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2675 = stablehlo.transpose %v2673, dims = [1, 0, 2, 3] : (tensor<32x256x28x28xf32>) -> tensor<256x32x28x28xf32>
    %v2676 = stablehlo.convolution(%v2674, %v2675)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<256x32x28x28xf32>) -> tensor<128x256x1x1xf32>
    %v2677 = stablehlo.transpose %v2676, dims = [1, 0, 2, 3] : (tensor<128x256x1x1xf32>) -> tensor<256x128x1x1xf32>
    %v2678 = stablehlo.constant dense<0.003125> : tensor<256x128x1x1xf32>
    %v2679 = stablehlo.multiply %v2677, %v2678 : tensor<256x128x1x1xf32>
    %v2680 = stablehlo.subtract %d3Wp, %v2679 : tensor<256x128x1x1xf32>
    %v2681 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2682 = stablehlo.reshape %v550 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2683 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2684 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2685 = stablehlo.reduce(%v2682 init: %v2681) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2686 = stablehlo.broadcast_in_dim %v2685, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2687 = stablehlo.divide %v2686, %v2683 : tensor<32x256x14x14xf32>
    %v2688 = stablehlo.subtract %v2682, %v2687 : tensor<32x256x14x14xf32>
    %v2689 = stablehlo.multiply %v2688, %v2688 : tensor<32x256x14x14xf32>
    %v2690 = stablehlo.reduce(%v2689 init: %v2681) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2691 = stablehlo.broadcast_in_dim %v2690, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2692 = stablehlo.divide %v2691, %v2683 : tensor<32x256x14x14xf32>
    %v2693 = stablehlo.add %v2692, %v2684 : tensor<32x256x14x14xf32>
    %v2694 = stablehlo.rsqrt %v2693 : tensor<32x256x14x14xf32>
    %v2695 = stablehlo.multiply %v2688, %v2694 : tensor<32x256x14x14xf32>
    %v2696 = stablehlo.reshape %v2476 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2697 = stablehlo.multiply %v2696, %v2695 : tensor<32x256x14x14xf32>
    %v2698 = stablehlo.reduce(%v2697 init: %v2681) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2699 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2700 = stablehlo.multiply %v2698, %v2699 : tensor<256xf32>
    %v2701 = stablehlo.subtract %d3gp, %v2700 : tensor<256xf32>
    %v2702 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2703 = stablehlo.reshape %v2476 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2704 = stablehlo.reduce(%v2703 init: %v2702) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2705 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2706 = stablehlo.multiply %v2704, %v2705 : tensor<256xf32>
    %v2707 = stablehlo.subtract %d3btp, %v2706 : tensor<256xf32>
    %v2708 = stablehlo.reshape %v2595 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2709 = stablehlo.reshape %v487 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2710 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v2711 = stablehlo.compare GT, %v2709, %v2710 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v2712 = stablehlo.select %v2711, %v2708, %v2710 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v2713 = stablehlo.reshape %v2712 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2714 = stablehlo.reshape %v2713 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2715 = stablehlo.reshape %v463 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2716 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2717 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v2718 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2719 = stablehlo.reduce(%v2715 init: %v2716) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2720 = stablehlo.broadcast_in_dim %v2719, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2721 = stablehlo.divide %v2720, %v2717 : tensor<32x128x28x28xf32>
    %v2722 = stablehlo.subtract %v2715, %v2721 : tensor<32x128x28x28xf32>
    %v2723 = stablehlo.multiply %v2722, %v2722 : tensor<32x128x28x28xf32>
    %v2724 = stablehlo.reduce(%v2723 init: %v2716) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2725 = stablehlo.broadcast_in_dim %v2724, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2726 = stablehlo.divide %v2725, %v2717 : tensor<32x128x28x28xf32>
    %v2727 = stablehlo.add %v2726, %v2718 : tensor<32x128x28x28xf32>
    %v2728 = stablehlo.rsqrt %v2727 : tensor<32x128x28x28xf32>
    %v2729 = stablehlo.multiply %v2722, %v2728 : tensor<32x128x28x28xf32>
    %v2730 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2731 = stablehlo.multiply %v2730, %v2714 : tensor<32x128x28x28xf32>
    %v2732 = stablehlo.reduce(%v2731 init: %v2716) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2733 = stablehlo.broadcast_in_dim %v2732, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2734 = stablehlo.multiply %v2729, %v2731 : tensor<32x128x28x28xf32>
    %v2735 = stablehlo.reduce(%v2734 init: %v2716) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2736 = stablehlo.broadcast_in_dim %v2735, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2737 = stablehlo.multiply %v2731, %v2717 : tensor<32x128x28x28xf32>
    %v2738 = stablehlo.subtract %v2737, %v2733 : tensor<32x128x28x28xf32>
    %v2739 = stablehlo.multiply %v2729, %v2736 : tensor<32x128x28x28xf32>
    %v2740 = stablehlo.subtract %v2738, %v2739 : tensor<32x128x28x28xf32>
    %v2741 = stablehlo.divide %v2728, %v2717 : tensor<32x128x28x28xf32>
    %v2742 = stablehlo.multiply %v2741, %v2740 : tensor<32x128x28x28xf32>
    %v2743 = stablehlo.reshape %v2742 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2744 = stablehlo.reshape %v2743 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2745 = stablehlo.transpose %s2b2W2, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2746 = stablehlo.reverse %v2745, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2747 = stablehlo.convolution(%v2744, %v2746)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2748 = stablehlo.reshape %v2747 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2749 = stablehlo.reshape %v2748 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2750 = stablehlo.reshape %v454 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2751 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v2752 = stablehlo.compare GT, %v2750, %v2751 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v2753 = stablehlo.select %v2752, %v2749, %v2751 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v2754 = stablehlo.reshape %v2753 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2755 = stablehlo.reshape %v2754 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2756 = stablehlo.reshape %v434 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2757 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2758 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v2759 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2760 = stablehlo.reduce(%v2756 init: %v2757) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2761 = stablehlo.broadcast_in_dim %v2760, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2762 = stablehlo.divide %v2761, %v2758 : tensor<32x128x28x28xf32>
    %v2763 = stablehlo.subtract %v2756, %v2762 : tensor<32x128x28x28xf32>
    %v2764 = stablehlo.multiply %v2763, %v2763 : tensor<32x128x28x28xf32>
    %v2765 = stablehlo.reduce(%v2764 init: %v2757) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2766 = stablehlo.broadcast_in_dim %v2765, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2767 = stablehlo.divide %v2766, %v2758 : tensor<32x128x28x28xf32>
    %v2768 = stablehlo.add %v2767, %v2759 : tensor<32x128x28x28xf32>
    %v2769 = stablehlo.rsqrt %v2768 : tensor<32x128x28x28xf32>
    %v2770 = stablehlo.multiply %v2763, %v2769 : tensor<32x128x28x28xf32>
    %v2771 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2772 = stablehlo.multiply %v2771, %v2755 : tensor<32x128x28x28xf32>
    %v2773 = stablehlo.reduce(%v2772 init: %v2757) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2774 = stablehlo.broadcast_in_dim %v2773, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2775 = stablehlo.multiply %v2770, %v2772 : tensor<32x128x28x28xf32>
    %v2776 = stablehlo.reduce(%v2775 init: %v2757) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2777 = stablehlo.broadcast_in_dim %v2776, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2778 = stablehlo.multiply %v2772, %v2758 : tensor<32x128x28x28xf32>
    %v2779 = stablehlo.subtract %v2778, %v2774 : tensor<32x128x28x28xf32>
    %v2780 = stablehlo.multiply %v2770, %v2777 : tensor<32x128x28x28xf32>
    %v2781 = stablehlo.subtract %v2779, %v2780 : tensor<32x128x28x28xf32>
    %v2782 = stablehlo.divide %v2769, %v2758 : tensor<32x128x28x28xf32>
    %v2783 = stablehlo.multiply %v2782, %v2781 : tensor<32x128x28x28xf32>
    %v2784 = stablehlo.reshape %v2783 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2785 = stablehlo.reshape %v2784 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2786 = stablehlo.transpose %s2b2W1, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2787 = stablehlo.reverse %v2786, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2788 = stablehlo.convolution(%v2785, %v2787)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2789 = stablehlo.reshape %v2788 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2790 = stablehlo.reshape %v2789 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2791 = stablehlo.reshape %v2713 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2792 = stablehlo.add %v2790, %v2791 : tensor<32x128x28x28xf32>
    %v2793 = stablehlo.reshape %v2792 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2794 = stablehlo.reshape %v429 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2795 = stablehlo.reshape %v2784 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2796 = stablehlo.transpose %v2794, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2797 = stablehlo.transpose %v2795, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2798 = stablehlo.convolution(%v2796, %v2797)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2799 = stablehlo.transpose %v2798, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2800 = stablehlo.constant dense<0.003125> : tensor<128x128x3x3xf32>
    %v2801 = stablehlo.multiply %v2799, %v2800 : tensor<128x128x3x3xf32>
    %v2802 = stablehlo.subtract %s2b2W1, %v2801 : tensor<128x128x3x3xf32>
    %v2803 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2804 = stablehlo.reshape %v434 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2805 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v2806 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2807 = stablehlo.reduce(%v2804 init: %v2803) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2808 = stablehlo.broadcast_in_dim %v2807, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2809 = stablehlo.divide %v2808, %v2805 : tensor<32x128x28x28xf32>
    %v2810 = stablehlo.subtract %v2804, %v2809 : tensor<32x128x28x28xf32>
    %v2811 = stablehlo.multiply %v2810, %v2810 : tensor<32x128x28x28xf32>
    %v2812 = stablehlo.reduce(%v2811 init: %v2803) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2813 = stablehlo.broadcast_in_dim %v2812, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2814 = stablehlo.divide %v2813, %v2805 : tensor<32x128x28x28xf32>
    %v2815 = stablehlo.add %v2814, %v2806 : tensor<32x128x28x28xf32>
    %v2816 = stablehlo.rsqrt %v2815 : tensor<32x128x28x28xf32>
    %v2817 = stablehlo.multiply %v2810, %v2816 : tensor<32x128x28x28xf32>
    %v2818 = stablehlo.reshape %v2754 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2819 = stablehlo.multiply %v2818, %v2817 : tensor<32x128x28x28xf32>
    %v2820 = stablehlo.reduce(%v2819 init: %v2803) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2821 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v2822 = stablehlo.multiply %v2820, %v2821 : tensor<128xf32>
    %v2823 = stablehlo.subtract %s2b2g1, %v2822 : tensor<128xf32>
    %v2824 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2825 = stablehlo.reshape %v2754 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2826 = stablehlo.reduce(%v2825 init: %v2824) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2827 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v2828 = stablehlo.multiply %v2826, %v2827 : tensor<128xf32>
    %v2829 = stablehlo.subtract %s2b2bt1, %v2828 : tensor<128xf32>
    %v2830 = stablehlo.reshape %v458 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2831 = stablehlo.reshape %v2743 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2832 = stablehlo.transpose %v2830, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2833 = stablehlo.transpose %v2831, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2834 = stablehlo.convolution(%v2832, %v2833)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2835 = stablehlo.transpose %v2834, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2836 = stablehlo.constant dense<0.003125> : tensor<128x128x3x3xf32>
    %v2837 = stablehlo.multiply %v2835, %v2836 : tensor<128x128x3x3xf32>
    %v2838 = stablehlo.subtract %s2b2W2, %v2837 : tensor<128x128x3x3xf32>
    %v2839 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2840 = stablehlo.reshape %v463 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2841 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v2842 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2843 = stablehlo.reduce(%v2840 init: %v2839) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2844 = stablehlo.broadcast_in_dim %v2843, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2845 = stablehlo.divide %v2844, %v2841 : tensor<32x128x28x28xf32>
    %v2846 = stablehlo.subtract %v2840, %v2845 : tensor<32x128x28x28xf32>
    %v2847 = stablehlo.multiply %v2846, %v2846 : tensor<32x128x28x28xf32>
    %v2848 = stablehlo.reduce(%v2847 init: %v2839) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2849 = stablehlo.broadcast_in_dim %v2848, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2850 = stablehlo.divide %v2849, %v2841 : tensor<32x128x28x28xf32>
    %v2851 = stablehlo.add %v2850, %v2842 : tensor<32x128x28x28xf32>
    %v2852 = stablehlo.rsqrt %v2851 : tensor<32x128x28x28xf32>
    %v2853 = stablehlo.multiply %v2846, %v2852 : tensor<32x128x28x28xf32>
    %v2854 = stablehlo.reshape %v2713 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2855 = stablehlo.multiply %v2854, %v2853 : tensor<32x128x28x28xf32>
    %v2856 = stablehlo.reduce(%v2855 init: %v2839) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2857 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v2858 = stablehlo.multiply %v2856, %v2857 : tensor<128xf32>
    %v2859 = stablehlo.subtract %s2b2g2, %v2858 : tensor<128xf32>
    %v2860 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2861 = stablehlo.reshape %v2713 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2862 = stablehlo.reduce(%v2861 init: %v2860) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2863 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v2864 = stablehlo.multiply %v2862, %v2863 : tensor<128xf32>
    %v2865 = stablehlo.subtract %s2b2bt2, %v2864 : tensor<128xf32>
    %v2866 = stablehlo.reshape %v2793 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2867 = stablehlo.reshape %v425 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2868 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v2869 = stablehlo.compare GT, %v2867, %v2868 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v2870 = stablehlo.select %v2869, %v2866, %v2868 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v2871 = stablehlo.reshape %v2870 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2872 = stablehlo.reshape %v2871 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2873 = stablehlo.reshape %v401 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2874 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2875 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v2876 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2877 = stablehlo.reduce(%v2873 init: %v2874) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2878 = stablehlo.broadcast_in_dim %v2877, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2879 = stablehlo.divide %v2878, %v2875 : tensor<32x128x28x28xf32>
    %v2880 = stablehlo.subtract %v2873, %v2879 : tensor<32x128x28x28xf32>
    %v2881 = stablehlo.multiply %v2880, %v2880 : tensor<32x128x28x28xf32>
    %v2882 = stablehlo.reduce(%v2881 init: %v2874) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2883 = stablehlo.broadcast_in_dim %v2882, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2884 = stablehlo.divide %v2883, %v2875 : tensor<32x128x28x28xf32>
    %v2885 = stablehlo.add %v2884, %v2876 : tensor<32x128x28x28xf32>
    %v2886 = stablehlo.rsqrt %v2885 : tensor<32x128x28x28xf32>
    %v2887 = stablehlo.multiply %v2880, %v2886 : tensor<32x128x28x28xf32>
    %v2888 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2889 = stablehlo.multiply %v2888, %v2872 : tensor<32x128x28x28xf32>
    %v2890 = stablehlo.reduce(%v2889 init: %v2874) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2891 = stablehlo.broadcast_in_dim %v2890, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2892 = stablehlo.multiply %v2887, %v2889 : tensor<32x128x28x28xf32>
    %v2893 = stablehlo.reduce(%v2892 init: %v2874) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2894 = stablehlo.broadcast_in_dim %v2893, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2895 = stablehlo.multiply %v2889, %v2875 : tensor<32x128x28x28xf32>
    %v2896 = stablehlo.subtract %v2895, %v2891 : tensor<32x128x28x28xf32>
    %v2897 = stablehlo.multiply %v2887, %v2894 : tensor<32x128x28x28xf32>
    %v2898 = stablehlo.subtract %v2896, %v2897 : tensor<32x128x28x28xf32>
    %v2899 = stablehlo.divide %v2886, %v2875 : tensor<32x128x28x28xf32>
    %v2900 = stablehlo.multiply %v2899, %v2898 : tensor<32x128x28x28xf32>
    %v2901 = stablehlo.reshape %v2900 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2902 = stablehlo.reshape %v2901 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2903 = stablehlo.transpose %s2b1W2, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2904 = stablehlo.reverse %v2903, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2905 = stablehlo.convolution(%v2902, %v2904)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2906 = stablehlo.reshape %v2905 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2907 = stablehlo.reshape %v2906 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2908 = stablehlo.reshape %v392 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2909 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v2910 = stablehlo.compare GT, %v2908, %v2909 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v2911 = stablehlo.select %v2910, %v2907, %v2909 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v2912 = stablehlo.reshape %v2911 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2913 = stablehlo.reshape %v2912 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2914 = stablehlo.reshape %v372 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2915 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2916 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v2917 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2918 = stablehlo.reduce(%v2914 init: %v2915) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2919 = stablehlo.broadcast_in_dim %v2918, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2920 = stablehlo.divide %v2919, %v2916 : tensor<32x128x28x28xf32>
    %v2921 = stablehlo.subtract %v2914, %v2920 : tensor<32x128x28x28xf32>
    %v2922 = stablehlo.multiply %v2921, %v2921 : tensor<32x128x28x28xf32>
    %v2923 = stablehlo.reduce(%v2922 init: %v2915) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2924 = stablehlo.broadcast_in_dim %v2923, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2925 = stablehlo.divide %v2924, %v2916 : tensor<32x128x28x28xf32>
    %v2926 = stablehlo.add %v2925, %v2917 : tensor<32x128x28x28xf32>
    %v2927 = stablehlo.rsqrt %v2926 : tensor<32x128x28x28xf32>
    %v2928 = stablehlo.multiply %v2921, %v2927 : tensor<32x128x28x28xf32>
    %v2929 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2930 = stablehlo.multiply %v2929, %v2913 : tensor<32x128x28x28xf32>
    %v2931 = stablehlo.reduce(%v2930 init: %v2915) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2932 = stablehlo.broadcast_in_dim %v2931, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2933 = stablehlo.multiply %v2928, %v2930 : tensor<32x128x28x28xf32>
    %v2934 = stablehlo.reduce(%v2933 init: %v2915) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2935 = stablehlo.broadcast_in_dim %v2934, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2936 = stablehlo.multiply %v2930, %v2916 : tensor<32x128x28x28xf32>
    %v2937 = stablehlo.subtract %v2936, %v2932 : tensor<32x128x28x28xf32>
    %v2938 = stablehlo.multiply %v2928, %v2935 : tensor<32x128x28x28xf32>
    %v2939 = stablehlo.subtract %v2937, %v2938 : tensor<32x128x28x28xf32>
    %v2940 = stablehlo.divide %v2927, %v2916 : tensor<32x128x28x28xf32>
    %v2941 = stablehlo.multiply %v2940, %v2939 : tensor<32x128x28x28xf32>
    %v2942 = stablehlo.reshape %v2941 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2943 = stablehlo.reshape %v2942 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2944 = stablehlo.transpose %s2b1W1, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2945 = stablehlo.reverse %v2944, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2946 = stablehlo.convolution(%v2943, %v2945)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2947 = stablehlo.reshape %v2946 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2948 = stablehlo.reshape %v2947 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2949 = stablehlo.reshape %v2871 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2950 = stablehlo.add %v2948, %v2949 : tensor<32x128x28x28xf32>
    %v2951 = stablehlo.reshape %v2950 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2952 = stablehlo.reshape %v367 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2953 = stablehlo.reshape %v2942 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2954 = stablehlo.transpose %v2952, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2955 = stablehlo.transpose %v2953, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2956 = stablehlo.convolution(%v2954, %v2955)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2957 = stablehlo.transpose %v2956, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2958 = stablehlo.constant dense<0.003125> : tensor<128x128x3x3xf32>
    %v2959 = stablehlo.multiply %v2957, %v2958 : tensor<128x128x3x3xf32>
    %v2960 = stablehlo.subtract %s2b1W1, %v2959 : tensor<128x128x3x3xf32>
    %v2961 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2962 = stablehlo.reshape %v372 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2963 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v2964 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2965 = stablehlo.reduce(%v2962 init: %v2961) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2966 = stablehlo.broadcast_in_dim %v2965, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2967 = stablehlo.divide %v2966, %v2963 : tensor<32x128x28x28xf32>
    %v2968 = stablehlo.subtract %v2962, %v2967 : tensor<32x128x28x28xf32>
    %v2969 = stablehlo.multiply %v2968, %v2968 : tensor<32x128x28x28xf32>
    %v2970 = stablehlo.reduce(%v2969 init: %v2961) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2971 = stablehlo.broadcast_in_dim %v2970, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2972 = stablehlo.divide %v2971, %v2963 : tensor<32x128x28x28xf32>
    %v2973 = stablehlo.add %v2972, %v2964 : tensor<32x128x28x28xf32>
    %v2974 = stablehlo.rsqrt %v2973 : tensor<32x128x28x28xf32>
    %v2975 = stablehlo.multiply %v2968, %v2974 : tensor<32x128x28x28xf32>
    %v2976 = stablehlo.reshape %v2912 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2977 = stablehlo.multiply %v2976, %v2975 : tensor<32x128x28x28xf32>
    %v2978 = stablehlo.reduce(%v2977 init: %v2961) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2979 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v2980 = stablehlo.multiply %v2978, %v2979 : tensor<128xf32>
    %v2981 = stablehlo.subtract %s2b1g1, %v2980 : tensor<128xf32>
    %v2982 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2983 = stablehlo.reshape %v2912 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2984 = stablehlo.reduce(%v2983 init: %v2982) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2985 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v2986 = stablehlo.multiply %v2984, %v2985 : tensor<128xf32>
    %v2987 = stablehlo.subtract %s2b1bt1, %v2986 : tensor<128xf32>
    %v2988 = stablehlo.reshape %v396 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2989 = stablehlo.reshape %v2901 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2990 = stablehlo.transpose %v2988, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2991 = stablehlo.transpose %v2989, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2992 = stablehlo.convolution(%v2990, %v2991)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2993 = stablehlo.transpose %v2992, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2994 = stablehlo.constant dense<0.003125> : tensor<128x128x3x3xf32>
    %v2995 = stablehlo.multiply %v2993, %v2994 : tensor<128x128x3x3xf32>
    %v2996 = stablehlo.subtract %s2b1W2, %v2995 : tensor<128x128x3x3xf32>
    %v2997 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2998 = stablehlo.reshape %v401 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2999 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v3000 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3001 = stablehlo.reduce(%v2998 init: %v2997) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3002 = stablehlo.broadcast_in_dim %v3001, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3003 = stablehlo.divide %v3002, %v2999 : tensor<32x128x28x28xf32>
    %v3004 = stablehlo.subtract %v2998, %v3003 : tensor<32x128x28x28xf32>
    %v3005 = stablehlo.multiply %v3004, %v3004 : tensor<32x128x28x28xf32>
    %v3006 = stablehlo.reduce(%v3005 init: %v2997) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3007 = stablehlo.broadcast_in_dim %v3006, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3008 = stablehlo.divide %v3007, %v2999 : tensor<32x128x28x28xf32>
    %v3009 = stablehlo.add %v3008, %v3000 : tensor<32x128x28x28xf32>
    %v3010 = stablehlo.rsqrt %v3009 : tensor<32x128x28x28xf32>
    %v3011 = stablehlo.multiply %v3004, %v3010 : tensor<32x128x28x28xf32>
    %v3012 = stablehlo.reshape %v2871 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3013 = stablehlo.multiply %v3012, %v3011 : tensor<32x128x28x28xf32>
    %v3014 = stablehlo.reduce(%v3013 init: %v2997) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3015 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v3016 = stablehlo.multiply %v3014, %v3015 : tensor<128xf32>
    %v3017 = stablehlo.subtract %s2b1g2, %v3016 : tensor<128xf32>
    %v3018 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3019 = stablehlo.reshape %v2871 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3020 = stablehlo.reduce(%v3019 init: %v3018) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3021 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v3022 = stablehlo.multiply %v3020, %v3021 : tensor<128xf32>
    %v3023 = stablehlo.subtract %s2b1bt2, %v3022 : tensor<128xf32>
    %v3024 = stablehlo.reshape %v2951 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3025 = stablehlo.reshape %v363 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3026 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v3027 = stablehlo.compare GT, %v3025, %v3026 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v3028 = stablehlo.select %v3027, %v3024, %v3026 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v3029 = stablehlo.reshape %v3028 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3030 = stablehlo.reshape %v3029 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3031 = stablehlo.reshape %v339 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3032 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3033 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v3034 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3035 = stablehlo.reduce(%v3031 init: %v3032) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3036 = stablehlo.broadcast_in_dim %v3035, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3037 = stablehlo.divide %v3036, %v3033 : tensor<32x128x28x28xf32>
    %v3038 = stablehlo.subtract %v3031, %v3037 : tensor<32x128x28x28xf32>
    %v3039 = stablehlo.multiply %v3038, %v3038 : tensor<32x128x28x28xf32>
    %v3040 = stablehlo.reduce(%v3039 init: %v3032) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3041 = stablehlo.broadcast_in_dim %v3040, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3042 = stablehlo.divide %v3041, %v3033 : tensor<32x128x28x28xf32>
    %v3043 = stablehlo.add %v3042, %v3034 : tensor<32x128x28x28xf32>
    %v3044 = stablehlo.rsqrt %v3043 : tensor<32x128x28x28xf32>
    %v3045 = stablehlo.multiply %v3038, %v3044 : tensor<32x128x28x28xf32>
    %v3046 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3047 = stablehlo.multiply %v3046, %v3030 : tensor<32x128x28x28xf32>
    %v3048 = stablehlo.reduce(%v3047 init: %v3032) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3049 = stablehlo.broadcast_in_dim %v3048, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3050 = stablehlo.multiply %v3045, %v3047 : tensor<32x128x28x28xf32>
    %v3051 = stablehlo.reduce(%v3050 init: %v3032) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3052 = stablehlo.broadcast_in_dim %v3051, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3053 = stablehlo.multiply %v3047, %v3033 : tensor<32x128x28x28xf32>
    %v3054 = stablehlo.subtract %v3053, %v3049 : tensor<32x128x28x28xf32>
    %v3055 = stablehlo.multiply %v3045, %v3052 : tensor<32x128x28x28xf32>
    %v3056 = stablehlo.subtract %v3054, %v3055 : tensor<32x128x28x28xf32>
    %v3057 = stablehlo.divide %v3044, %v3033 : tensor<32x128x28x28xf32>
    %v3058 = stablehlo.multiply %v3057, %v3056 : tensor<32x128x28x28xf32>
    %v3059 = stablehlo.reshape %v3058 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3060 = stablehlo.reshape %v3059 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3061 = stablehlo.transpose %s2b0W2, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3062 = stablehlo.reverse %v3061, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3063 = stablehlo.convolution(%v3060, %v3062)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v3064 = stablehlo.reshape %v3063 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3065 = stablehlo.reshape %v3064 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3066 = stablehlo.reshape %v330 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3067 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v3068 = stablehlo.compare GT, %v3066, %v3067 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v3069 = stablehlo.select %v3068, %v3065, %v3067 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v3070 = stablehlo.reshape %v3069 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3071 = stablehlo.reshape %v3070 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3072 = stablehlo.reshape %v310 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3073 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3074 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v3075 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3076 = stablehlo.reduce(%v3072 init: %v3073) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3077 = stablehlo.broadcast_in_dim %v3076, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3078 = stablehlo.divide %v3077, %v3074 : tensor<32x128x28x28xf32>
    %v3079 = stablehlo.subtract %v3072, %v3078 : tensor<32x128x28x28xf32>
    %v3080 = stablehlo.multiply %v3079, %v3079 : tensor<32x128x28x28xf32>
    %v3081 = stablehlo.reduce(%v3080 init: %v3073) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3082 = stablehlo.broadcast_in_dim %v3081, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3083 = stablehlo.divide %v3082, %v3074 : tensor<32x128x28x28xf32>
    %v3084 = stablehlo.add %v3083, %v3075 : tensor<32x128x28x28xf32>
    %v3085 = stablehlo.rsqrt %v3084 : tensor<32x128x28x28xf32>
    %v3086 = stablehlo.multiply %v3079, %v3085 : tensor<32x128x28x28xf32>
    %v3087 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3088 = stablehlo.multiply %v3087, %v3071 : tensor<32x128x28x28xf32>
    %v3089 = stablehlo.reduce(%v3088 init: %v3073) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3090 = stablehlo.broadcast_in_dim %v3089, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3091 = stablehlo.multiply %v3086, %v3088 : tensor<32x128x28x28xf32>
    %v3092 = stablehlo.reduce(%v3091 init: %v3073) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3093 = stablehlo.broadcast_in_dim %v3092, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3094 = stablehlo.multiply %v3088, %v3074 : tensor<32x128x28x28xf32>
    %v3095 = stablehlo.subtract %v3094, %v3090 : tensor<32x128x28x28xf32>
    %v3096 = stablehlo.multiply %v3086, %v3093 : tensor<32x128x28x28xf32>
    %v3097 = stablehlo.subtract %v3095, %v3096 : tensor<32x128x28x28xf32>
    %v3098 = stablehlo.divide %v3085, %v3074 : tensor<32x128x28x28xf32>
    %v3099 = stablehlo.multiply %v3098, %v3097 : tensor<32x128x28x28xf32>
    %v3100 = stablehlo.reshape %v3099 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3101 = stablehlo.reshape %v3100 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3102 = stablehlo.transpose %s2b0W1, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3103 = stablehlo.reverse %v3102, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3104 = stablehlo.convolution(%v3101, %v3103)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v3105 = stablehlo.reshape %v3104 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3106 = stablehlo.reshape %v3105 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3107 = stablehlo.reshape %v3029 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3108 = stablehlo.add %v3106, %v3107 : tensor<32x128x28x28xf32>
    %v3109 = stablehlo.reshape %v3108 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3110 = stablehlo.reshape %v305 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3111 = stablehlo.reshape %v3100 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3112 = stablehlo.transpose %v3110, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3113 = stablehlo.transpose %v3111, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3114 = stablehlo.convolution(%v3112, %v3113)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v3115 = stablehlo.transpose %v3114, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3116 = stablehlo.constant dense<0.003125> : tensor<128x128x3x3xf32>
    %v3117 = stablehlo.multiply %v3115, %v3116 : tensor<128x128x3x3xf32>
    %v3118 = stablehlo.subtract %s2b0W1, %v3117 : tensor<128x128x3x3xf32>
    %v3119 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3120 = stablehlo.reshape %v310 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3121 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v3122 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3123 = stablehlo.reduce(%v3120 init: %v3119) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3124 = stablehlo.broadcast_in_dim %v3123, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3125 = stablehlo.divide %v3124, %v3121 : tensor<32x128x28x28xf32>
    %v3126 = stablehlo.subtract %v3120, %v3125 : tensor<32x128x28x28xf32>
    %v3127 = stablehlo.multiply %v3126, %v3126 : tensor<32x128x28x28xf32>
    %v3128 = stablehlo.reduce(%v3127 init: %v3119) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3129 = stablehlo.broadcast_in_dim %v3128, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3130 = stablehlo.divide %v3129, %v3121 : tensor<32x128x28x28xf32>
    %v3131 = stablehlo.add %v3130, %v3122 : tensor<32x128x28x28xf32>
    %v3132 = stablehlo.rsqrt %v3131 : tensor<32x128x28x28xf32>
    %v3133 = stablehlo.multiply %v3126, %v3132 : tensor<32x128x28x28xf32>
    %v3134 = stablehlo.reshape %v3070 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3135 = stablehlo.multiply %v3134, %v3133 : tensor<32x128x28x28xf32>
    %v3136 = stablehlo.reduce(%v3135 init: %v3119) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3137 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v3138 = stablehlo.multiply %v3136, %v3137 : tensor<128xf32>
    %v3139 = stablehlo.subtract %s2b0g1, %v3138 : tensor<128xf32>
    %v3140 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3141 = stablehlo.reshape %v3070 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3142 = stablehlo.reduce(%v3141 init: %v3140) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3143 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v3144 = stablehlo.multiply %v3142, %v3143 : tensor<128xf32>
    %v3145 = stablehlo.subtract %s2b0bt1, %v3144 : tensor<128xf32>
    %v3146 = stablehlo.reshape %v334 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3147 = stablehlo.reshape %v3059 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3148 = stablehlo.transpose %v3146, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3149 = stablehlo.transpose %v3147, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3150 = stablehlo.convolution(%v3148, %v3149)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v3151 = stablehlo.transpose %v3150, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3152 = stablehlo.constant dense<0.003125> : tensor<128x128x3x3xf32>
    %v3153 = stablehlo.multiply %v3151, %v3152 : tensor<128x128x3x3xf32>
    %v3154 = stablehlo.subtract %s2b0W2, %v3153 : tensor<128x128x3x3xf32>
    %v3155 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3156 = stablehlo.reshape %v339 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3157 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v3158 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3159 = stablehlo.reduce(%v3156 init: %v3155) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3160 = stablehlo.broadcast_in_dim %v3159, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3161 = stablehlo.divide %v3160, %v3157 : tensor<32x128x28x28xf32>
    %v3162 = stablehlo.subtract %v3156, %v3161 : tensor<32x128x28x28xf32>
    %v3163 = stablehlo.multiply %v3162, %v3162 : tensor<32x128x28x28xf32>
    %v3164 = stablehlo.reduce(%v3163 init: %v3155) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3165 = stablehlo.broadcast_in_dim %v3164, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3166 = stablehlo.divide %v3165, %v3157 : tensor<32x128x28x28xf32>
    %v3167 = stablehlo.add %v3166, %v3158 : tensor<32x128x28x28xf32>
    %v3168 = stablehlo.rsqrt %v3167 : tensor<32x128x28x28xf32>
    %v3169 = stablehlo.multiply %v3162, %v3168 : tensor<32x128x28x28xf32>
    %v3170 = stablehlo.reshape %v3029 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3171 = stablehlo.multiply %v3170, %v3169 : tensor<32x128x28x28xf32>
    %v3172 = stablehlo.reduce(%v3171 init: %v3155) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3173 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v3174 = stablehlo.multiply %v3172, %v3173 : tensor<128xf32>
    %v3175 = stablehlo.subtract %s2b0g2, %v3174 : tensor<128xf32>
    %v3176 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3177 = stablehlo.reshape %v3029 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3178 = stablehlo.reduce(%v3177 init: %v3176) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3179 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v3180 = stablehlo.multiply %v3178, %v3179 : tensor<128xf32>
    %v3181 = stablehlo.subtract %s2b0bt2, %v3180 : tensor<128xf32>
    %v3182 = stablehlo.reshape %v3109 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3183 = stablehlo.reshape %v301 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3184 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v3185 = stablehlo.compare GT, %v3183, %v3184 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v3186 = stablehlo.select %v3185, %v3182, %v3184 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v3187 = stablehlo.reshape %v3186 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3188 = stablehlo.reshape %v3187 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3189 = stablehlo.reshape %v252 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3190 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3191 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v3192 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3193 = stablehlo.reduce(%v3189 init: %v3190) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3194 = stablehlo.broadcast_in_dim %v3193, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3195 = stablehlo.divide %v3194, %v3191 : tensor<32x128x28x28xf32>
    %v3196 = stablehlo.subtract %v3189, %v3195 : tensor<32x128x28x28xf32>
    %v3197 = stablehlo.multiply %v3196, %v3196 : tensor<32x128x28x28xf32>
    %v3198 = stablehlo.reduce(%v3197 init: %v3190) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3199 = stablehlo.broadcast_in_dim %v3198, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3200 = stablehlo.divide %v3199, %v3191 : tensor<32x128x28x28xf32>
    %v3201 = stablehlo.add %v3200, %v3192 : tensor<32x128x28x28xf32>
    %v3202 = stablehlo.rsqrt %v3201 : tensor<32x128x28x28xf32>
    %v3203 = stablehlo.multiply %v3196, %v3202 : tensor<32x128x28x28xf32>
    %v3204 = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3205 = stablehlo.multiply %v3204, %v3188 : tensor<32x128x28x28xf32>
    %v3206 = stablehlo.reduce(%v3205 init: %v3190) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3207 = stablehlo.broadcast_in_dim %v3206, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3208 = stablehlo.multiply %v3203, %v3205 : tensor<32x128x28x28xf32>
    %v3209 = stablehlo.reduce(%v3208 init: %v3190) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3210 = stablehlo.broadcast_in_dim %v3209, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3211 = stablehlo.multiply %v3205, %v3191 : tensor<32x128x28x28xf32>
    %v3212 = stablehlo.subtract %v3211, %v3207 : tensor<32x128x28x28xf32>
    %v3213 = stablehlo.multiply %v3203, %v3210 : tensor<32x128x28x28xf32>
    %v3214 = stablehlo.subtract %v3212, %v3213 : tensor<32x128x28x28xf32>
    %v3215 = stablehlo.divide %v3202, %v3191 : tensor<32x128x28x28xf32>
    %v3216 = stablehlo.multiply %v3215, %v3214 : tensor<32x128x28x28xf32>
    %v3217 = stablehlo.reshape %v3216 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3218 = stablehlo.reshape %v3217 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3219 = stablehlo.transpose %d2W2, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3220 = stablehlo.reverse %v3219, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3221 = stablehlo.convolution(%v3218, %v3220)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v3222 = stablehlo.reshape %v3221 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3223 = stablehlo.reshape %v3222 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3224 = stablehlo.reshape %v243 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3225 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v3226 = stablehlo.compare GT, %v3224, %v3225 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v3227 = stablehlo.select %v3226, %v3223, %v3225 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v3228 = stablehlo.reshape %v3227 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3229 = stablehlo.reshape %v3228 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3230 = stablehlo.reshape %v223 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3231 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3232 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v3233 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3234 = stablehlo.reduce(%v3230 init: %v3231) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3235 = stablehlo.broadcast_in_dim %v3234, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3236 = stablehlo.divide %v3235, %v3232 : tensor<32x128x28x28xf32>
    %v3237 = stablehlo.subtract %v3230, %v3236 : tensor<32x128x28x28xf32>
    %v3238 = stablehlo.multiply %v3237, %v3237 : tensor<32x128x28x28xf32>
    %v3239 = stablehlo.reduce(%v3238 init: %v3231) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3240 = stablehlo.broadcast_in_dim %v3239, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3241 = stablehlo.divide %v3240, %v3232 : tensor<32x128x28x28xf32>
    %v3242 = stablehlo.add %v3241, %v3233 : tensor<32x128x28x28xf32>
    %v3243 = stablehlo.rsqrt %v3242 : tensor<32x128x28x28xf32>
    %v3244 = stablehlo.multiply %v3237, %v3243 : tensor<32x128x28x28xf32>
    %v3245 = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3246 = stablehlo.multiply %v3245, %v3229 : tensor<32x128x28x28xf32>
    %v3247 = stablehlo.reduce(%v3246 init: %v3231) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3248 = stablehlo.broadcast_in_dim %v3247, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3249 = stablehlo.multiply %v3244, %v3246 : tensor<32x128x28x28xf32>
    %v3250 = stablehlo.reduce(%v3249 init: %v3231) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3251 = stablehlo.broadcast_in_dim %v3250, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3252 = stablehlo.multiply %v3246, %v3232 : tensor<32x128x28x28xf32>
    %v3253 = stablehlo.subtract %v3252, %v3248 : tensor<32x128x28x28xf32>
    %v3254 = stablehlo.multiply %v3244, %v3251 : tensor<32x128x28x28xf32>
    %v3255 = stablehlo.subtract %v3253, %v3254 : tensor<32x128x28x28xf32>
    %v3256 = stablehlo.divide %v3243, %v3232 : tensor<32x128x28x28xf32>
    %v3257 = stablehlo.multiply %v3256, %v3255 : tensor<32x128x28x28xf32>
    %v3258 = stablehlo.reshape %v3257 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3259 = stablehlo.reshape %v3258 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3260 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3261 = stablehlo.pad %v3259, %v3260, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %v3262 = stablehlo.transpose %d2W1, dims = [1, 0, 2, 3] : (tensor<128x64x3x3xf32>) -> tensor<64x128x3x3xf32>
    %v3263 = stablehlo.reverse %v3262, dims = [2, 3] : tensor<64x128x3x3xf32>
    %v3264 = stablehlo.convolution(%v3261, %v3263)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x56x56xf32>, tensor<64x128x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3265 = stablehlo.reshape %v3264 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3266 = stablehlo.reshape %v3187 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3267 = stablehlo.reshape %v277 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3268 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3269 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v3270 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3271 = stablehlo.reduce(%v3267 init: %v3268) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3272 = stablehlo.broadcast_in_dim %v3271, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3273 = stablehlo.divide %v3272, %v3269 : tensor<32x128x28x28xf32>
    %v3274 = stablehlo.subtract %v3267, %v3273 : tensor<32x128x28x28xf32>
    %v3275 = stablehlo.multiply %v3274, %v3274 : tensor<32x128x28x28xf32>
    %v3276 = stablehlo.reduce(%v3275 init: %v3268) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3277 = stablehlo.broadcast_in_dim %v3276, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3278 = stablehlo.divide %v3277, %v3269 : tensor<32x128x28x28xf32>
    %v3279 = stablehlo.add %v3278, %v3270 : tensor<32x128x28x28xf32>
    %v3280 = stablehlo.rsqrt %v3279 : tensor<32x128x28x28xf32>
    %v3281 = stablehlo.multiply %v3274, %v3280 : tensor<32x128x28x28xf32>
    %v3282 = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3283 = stablehlo.multiply %v3282, %v3266 : tensor<32x128x28x28xf32>
    %v3284 = stablehlo.reduce(%v3283 init: %v3268) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3285 = stablehlo.broadcast_in_dim %v3284, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3286 = stablehlo.multiply %v3281, %v3283 : tensor<32x128x28x28xf32>
    %v3287 = stablehlo.reduce(%v3286 init: %v3268) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3288 = stablehlo.broadcast_in_dim %v3287, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3289 = stablehlo.multiply %v3283, %v3269 : tensor<32x128x28x28xf32>
    %v3290 = stablehlo.subtract %v3289, %v3285 : tensor<32x128x28x28xf32>
    %v3291 = stablehlo.multiply %v3281, %v3288 : tensor<32x128x28x28xf32>
    %v3292 = stablehlo.subtract %v3290, %v3291 : tensor<32x128x28x28xf32>
    %v3293 = stablehlo.divide %v3280, %v3269 : tensor<32x128x28x28xf32>
    %v3294 = stablehlo.multiply %v3293, %v3292 : tensor<32x128x28x28xf32>
    %v3295 = stablehlo.reshape %v3294 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3296 = stablehlo.reshape %v3295 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3297 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3298 = stablehlo.pad %v3296, %v3297, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %v3299 = stablehlo.transpose %d2Wp, dims = [1, 0, 2, 3] : (tensor<128x64x1x1xf32>) -> tensor<64x128x1x1xf32>
    %v3300 = stablehlo.reverse %v3299, dims = [2, 3] : tensor<64x128x1x1xf32>
    %v3301 = stablehlo.convolution(%v3298, %v3300)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x56x56xf32>, tensor<64x128x1x1xf32>) -> tensor<32x64x56x56xf32>
    %v3302 = stablehlo.reshape %v3301 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3303 = stablehlo.reshape %v3265 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3304 = stablehlo.reshape %v3302 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3305 = stablehlo.add %v3303, %v3304 : tensor<32x64x56x56xf32>
    %v3306 = stablehlo.reshape %v3305 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3307 = stablehlo.reshape %v218 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3308 = stablehlo.reshape %v3258 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3309 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3310 = stablehlo.pad %v3308, %v3309, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %v3311 = stablehlo.transpose %v3307, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3312 = stablehlo.transpose %v3310, dims = [1, 0, 2, 3] : (tensor<32x128x56x56xf32>) -> tensor<128x32x56x56xf32>
    %v3313 = stablehlo.convolution(%v3311, %v3312)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<128x32x56x56xf32>) -> tensor<64x128x3x3xf32>
    %v3314 = stablehlo.transpose %v3313, dims = [1, 0, 2, 3] : (tensor<64x128x3x3xf32>) -> tensor<128x64x3x3xf32>
    %v3315 = stablehlo.constant dense<0.003125> : tensor<128x64x3x3xf32>
    %v3316 = stablehlo.multiply %v3314, %v3315 : tensor<128x64x3x3xf32>
    %v3317 = stablehlo.subtract %d2W1, %v3316 : tensor<128x64x3x3xf32>
    %v3318 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3319 = stablehlo.reshape %v223 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3320 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v3321 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3322 = stablehlo.reduce(%v3319 init: %v3318) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3323 = stablehlo.broadcast_in_dim %v3322, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3324 = stablehlo.divide %v3323, %v3320 : tensor<32x128x28x28xf32>
    %v3325 = stablehlo.subtract %v3319, %v3324 : tensor<32x128x28x28xf32>
    %v3326 = stablehlo.multiply %v3325, %v3325 : tensor<32x128x28x28xf32>
    %v3327 = stablehlo.reduce(%v3326 init: %v3318) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3328 = stablehlo.broadcast_in_dim %v3327, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3329 = stablehlo.divide %v3328, %v3320 : tensor<32x128x28x28xf32>
    %v3330 = stablehlo.add %v3329, %v3321 : tensor<32x128x28x28xf32>
    %v3331 = stablehlo.rsqrt %v3330 : tensor<32x128x28x28xf32>
    %v3332 = stablehlo.multiply %v3325, %v3331 : tensor<32x128x28x28xf32>
    %v3333 = stablehlo.reshape %v3228 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3334 = stablehlo.multiply %v3333, %v3332 : tensor<32x128x28x28xf32>
    %v3335 = stablehlo.reduce(%v3334 init: %v3318) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3336 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v3337 = stablehlo.multiply %v3335, %v3336 : tensor<128xf32>
    %v3338 = stablehlo.subtract %d2g1, %v3337 : tensor<128xf32>
    %v3339 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3340 = stablehlo.reshape %v3228 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3341 = stablehlo.reduce(%v3340 init: %v3339) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3342 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v3343 = stablehlo.multiply %v3341, %v3342 : tensor<128xf32>
    %v3344 = stablehlo.subtract %d2bt1, %v3343 : tensor<128xf32>
    %v3345 = stablehlo.reshape %v247 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3346 = stablehlo.reshape %v3217 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3347 = stablehlo.transpose %v3345, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3348 = stablehlo.transpose %v3346, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3349 = stablehlo.convolution(%v3347, %v3348)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v3350 = stablehlo.transpose %v3349, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3351 = stablehlo.constant dense<0.003125> : tensor<128x128x3x3xf32>
    %v3352 = stablehlo.multiply %v3350, %v3351 : tensor<128x128x3x3xf32>
    %v3353 = stablehlo.subtract %d2W2, %v3352 : tensor<128x128x3x3xf32>
    %v3354 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3355 = stablehlo.reshape %v252 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3356 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v3357 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3358 = stablehlo.reduce(%v3355 init: %v3354) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3359 = stablehlo.broadcast_in_dim %v3358, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3360 = stablehlo.divide %v3359, %v3356 : tensor<32x128x28x28xf32>
    %v3361 = stablehlo.subtract %v3355, %v3360 : tensor<32x128x28x28xf32>
    %v3362 = stablehlo.multiply %v3361, %v3361 : tensor<32x128x28x28xf32>
    %v3363 = stablehlo.reduce(%v3362 init: %v3354) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3364 = stablehlo.broadcast_in_dim %v3363, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3365 = stablehlo.divide %v3364, %v3356 : tensor<32x128x28x28xf32>
    %v3366 = stablehlo.add %v3365, %v3357 : tensor<32x128x28x28xf32>
    %v3367 = stablehlo.rsqrt %v3366 : tensor<32x128x28x28xf32>
    %v3368 = stablehlo.multiply %v3361, %v3367 : tensor<32x128x28x28xf32>
    %v3369 = stablehlo.reshape %v3187 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3370 = stablehlo.multiply %v3369, %v3368 : tensor<32x128x28x28xf32>
    %v3371 = stablehlo.reduce(%v3370 init: %v3354) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3372 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v3373 = stablehlo.multiply %v3371, %v3372 : tensor<128xf32>
    %v3374 = stablehlo.subtract %d2g2, %v3373 : tensor<128xf32>
    %v3375 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3376 = stablehlo.reshape %v3187 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3377 = stablehlo.reduce(%v3376 init: %v3375) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3378 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v3379 = stablehlo.multiply %v3377, %v3378 : tensor<128xf32>
    %v3380 = stablehlo.subtract %d2bt2, %v3379 : tensor<128xf32>
    %v3381 = stablehlo.reshape %v218 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3382 = stablehlo.reshape %v3295 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3383 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3384 = stablehlo.pad %v3382, %v3383, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %v3385 = stablehlo.transpose %v3381, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3386 = stablehlo.transpose %v3384, dims = [1, 0, 2, 3] : (tensor<32x128x56x56xf32>) -> tensor<128x32x56x56xf32>
    %v3387 = stablehlo.convolution(%v3385, %v3386)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<128x32x56x56xf32>) -> tensor<64x128x1x1xf32>
    %v3388 = stablehlo.transpose %v3387, dims = [1, 0, 2, 3] : (tensor<64x128x1x1xf32>) -> tensor<128x64x1x1xf32>
    %v3389 = stablehlo.constant dense<0.003125> : tensor<128x64x1x1xf32>
    %v3390 = stablehlo.multiply %v3388, %v3389 : tensor<128x64x1x1xf32>
    %v3391 = stablehlo.subtract %d2Wp, %v3390 : tensor<128x64x1x1xf32>
    %v3392 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3393 = stablehlo.reshape %v277 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3394 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v3395 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3396 = stablehlo.reduce(%v3393 init: %v3392) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3397 = stablehlo.broadcast_in_dim %v3396, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3398 = stablehlo.divide %v3397, %v3394 : tensor<32x128x28x28xf32>
    %v3399 = stablehlo.subtract %v3393, %v3398 : tensor<32x128x28x28xf32>
    %v3400 = stablehlo.multiply %v3399, %v3399 : tensor<32x128x28x28xf32>
    %v3401 = stablehlo.reduce(%v3400 init: %v3392) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3402 = stablehlo.broadcast_in_dim %v3401, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3403 = stablehlo.divide %v3402, %v3394 : tensor<32x128x28x28xf32>
    %v3404 = stablehlo.add %v3403, %v3395 : tensor<32x128x28x28xf32>
    %v3405 = stablehlo.rsqrt %v3404 : tensor<32x128x28x28xf32>
    %v3406 = stablehlo.multiply %v3399, %v3405 : tensor<32x128x28x28xf32>
    %v3407 = stablehlo.reshape %v3187 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3408 = stablehlo.multiply %v3407, %v3406 : tensor<32x128x28x28xf32>
    %v3409 = stablehlo.reduce(%v3408 init: %v3392) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3410 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v3411 = stablehlo.multiply %v3409, %v3410 : tensor<128xf32>
    %v3412 = stablehlo.subtract %d2gp, %v3411 : tensor<128xf32>
    %v3413 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3414 = stablehlo.reshape %v3187 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3415 = stablehlo.reduce(%v3414 init: %v3413) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3416 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v3417 = stablehlo.multiply %v3415, %v3416 : tensor<128xf32>
    %v3418 = stablehlo.subtract %d2btp, %v3417 : tensor<128xf32>
    %v3419 = stablehlo.reshape %v3306 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3420 = stablehlo.reshape %v214 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3421 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3422 = stablehlo.compare GT, %v3420, %v3421 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3423 = stablehlo.select %v3422, %v3419, %v3421 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3424 = stablehlo.reshape %v3423 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3425 = stablehlo.reshape %v3424 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3426 = stablehlo.reshape %v190 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3427 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3428 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v3429 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3430 = stablehlo.reduce(%v3426 init: %v3427) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3431 = stablehlo.broadcast_in_dim %v3430, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3432 = stablehlo.divide %v3431, %v3428 : tensor<32x64x56x56xf32>
    %v3433 = stablehlo.subtract %v3426, %v3432 : tensor<32x64x56x56xf32>
    %v3434 = stablehlo.multiply %v3433, %v3433 : tensor<32x64x56x56xf32>
    %v3435 = stablehlo.reduce(%v3434 init: %v3427) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3436 = stablehlo.broadcast_in_dim %v3435, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3437 = stablehlo.divide %v3436, %v3428 : tensor<32x64x56x56xf32>
    %v3438 = stablehlo.add %v3437, %v3429 : tensor<32x64x56x56xf32>
    %v3439 = stablehlo.rsqrt %v3438 : tensor<32x64x56x56xf32>
    %v3440 = stablehlo.multiply %v3433, %v3439 : tensor<32x64x56x56xf32>
    %v3441 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3442 = stablehlo.multiply %v3441, %v3425 : tensor<32x64x56x56xf32>
    %v3443 = stablehlo.reduce(%v3442 init: %v3427) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3444 = stablehlo.broadcast_in_dim %v3443, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3445 = stablehlo.multiply %v3440, %v3442 : tensor<32x64x56x56xf32>
    %v3446 = stablehlo.reduce(%v3445 init: %v3427) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3447 = stablehlo.broadcast_in_dim %v3446, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3448 = stablehlo.multiply %v3442, %v3428 : tensor<32x64x56x56xf32>
    %v3449 = stablehlo.subtract %v3448, %v3444 : tensor<32x64x56x56xf32>
    %v3450 = stablehlo.multiply %v3440, %v3447 : tensor<32x64x56x56xf32>
    %v3451 = stablehlo.subtract %v3449, %v3450 : tensor<32x64x56x56xf32>
    %v3452 = stablehlo.divide %v3439, %v3428 : tensor<32x64x56x56xf32>
    %v3453 = stablehlo.multiply %v3452, %v3451 : tensor<32x64x56x56xf32>
    %v3454 = stablehlo.reshape %v3453 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3455 = stablehlo.reshape %v3454 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3456 = stablehlo.transpose %s1b2W2, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3457 = stablehlo.reverse %v3456, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3458 = stablehlo.convolution(%v3455, %v3457)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3459 = stablehlo.reshape %v3458 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3460 = stablehlo.reshape %v3459 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3461 = stablehlo.reshape %v181 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3462 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3463 = stablehlo.compare GT, %v3461, %v3462 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3464 = stablehlo.select %v3463, %v3460, %v3462 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3465 = stablehlo.reshape %v3464 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3466 = stablehlo.reshape %v3465 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3467 = stablehlo.reshape %v161 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3468 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3469 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v3470 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3471 = stablehlo.reduce(%v3467 init: %v3468) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3472 = stablehlo.broadcast_in_dim %v3471, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3473 = stablehlo.divide %v3472, %v3469 : tensor<32x64x56x56xf32>
    %v3474 = stablehlo.subtract %v3467, %v3473 : tensor<32x64x56x56xf32>
    %v3475 = stablehlo.multiply %v3474, %v3474 : tensor<32x64x56x56xf32>
    %v3476 = stablehlo.reduce(%v3475 init: %v3468) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3477 = stablehlo.broadcast_in_dim %v3476, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3478 = stablehlo.divide %v3477, %v3469 : tensor<32x64x56x56xf32>
    %v3479 = stablehlo.add %v3478, %v3470 : tensor<32x64x56x56xf32>
    %v3480 = stablehlo.rsqrt %v3479 : tensor<32x64x56x56xf32>
    %v3481 = stablehlo.multiply %v3474, %v3480 : tensor<32x64x56x56xf32>
    %v3482 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3483 = stablehlo.multiply %v3482, %v3466 : tensor<32x64x56x56xf32>
    %v3484 = stablehlo.reduce(%v3483 init: %v3468) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3485 = stablehlo.broadcast_in_dim %v3484, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3486 = stablehlo.multiply %v3481, %v3483 : tensor<32x64x56x56xf32>
    %v3487 = stablehlo.reduce(%v3486 init: %v3468) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3488 = stablehlo.broadcast_in_dim %v3487, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3489 = stablehlo.multiply %v3483, %v3469 : tensor<32x64x56x56xf32>
    %v3490 = stablehlo.subtract %v3489, %v3485 : tensor<32x64x56x56xf32>
    %v3491 = stablehlo.multiply %v3481, %v3488 : tensor<32x64x56x56xf32>
    %v3492 = stablehlo.subtract %v3490, %v3491 : tensor<32x64x56x56xf32>
    %v3493 = stablehlo.divide %v3480, %v3469 : tensor<32x64x56x56xf32>
    %v3494 = stablehlo.multiply %v3493, %v3492 : tensor<32x64x56x56xf32>
    %v3495 = stablehlo.reshape %v3494 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3496 = stablehlo.reshape %v3495 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3497 = stablehlo.transpose %s1b2W1, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3498 = stablehlo.reverse %v3497, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3499 = stablehlo.convolution(%v3496, %v3498)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3500 = stablehlo.reshape %v3499 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3501 = stablehlo.reshape %v3500 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3502 = stablehlo.reshape %v3424 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3503 = stablehlo.add %v3501, %v3502 : tensor<32x64x56x56xf32>
    %v3504 = stablehlo.reshape %v3503 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3505 = stablehlo.reshape %v156 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3506 = stablehlo.reshape %v3495 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3507 = stablehlo.transpose %v3505, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3508 = stablehlo.transpose %v3506, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3509 = stablehlo.convolution(%v3507, %v3508)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3510 = stablehlo.transpose %v3509, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3511 = stablehlo.constant dense<0.003125> : tensor<64x64x3x3xf32>
    %v3512 = stablehlo.multiply %v3510, %v3511 : tensor<64x64x3x3xf32>
    %v3513 = stablehlo.subtract %s1b2W1, %v3512 : tensor<64x64x3x3xf32>
    %v3514 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3515 = stablehlo.reshape %v161 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3516 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v3517 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3518 = stablehlo.reduce(%v3515 init: %v3514) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3519 = stablehlo.broadcast_in_dim %v3518, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3520 = stablehlo.divide %v3519, %v3516 : tensor<32x64x56x56xf32>
    %v3521 = stablehlo.subtract %v3515, %v3520 : tensor<32x64x56x56xf32>
    %v3522 = stablehlo.multiply %v3521, %v3521 : tensor<32x64x56x56xf32>
    %v3523 = stablehlo.reduce(%v3522 init: %v3514) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3524 = stablehlo.broadcast_in_dim %v3523, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3525 = stablehlo.divide %v3524, %v3516 : tensor<32x64x56x56xf32>
    %v3526 = stablehlo.add %v3525, %v3517 : tensor<32x64x56x56xf32>
    %v3527 = stablehlo.rsqrt %v3526 : tensor<32x64x56x56xf32>
    %v3528 = stablehlo.multiply %v3521, %v3527 : tensor<32x64x56x56xf32>
    %v3529 = stablehlo.reshape %v3465 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3530 = stablehlo.multiply %v3529, %v3528 : tensor<32x64x56x56xf32>
    %v3531 = stablehlo.reduce(%v3530 init: %v3514) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3532 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3533 = stablehlo.multiply %v3531, %v3532 : tensor<64xf32>
    %v3534 = stablehlo.subtract %s1b2g1, %v3533 : tensor<64xf32>
    %v3535 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3536 = stablehlo.reshape %v3465 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3537 = stablehlo.reduce(%v3536 init: %v3535) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3538 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3539 = stablehlo.multiply %v3537, %v3538 : tensor<64xf32>
    %v3540 = stablehlo.subtract %s1b2bt1, %v3539 : tensor<64xf32>
    %v3541 = stablehlo.reshape %v185 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3542 = stablehlo.reshape %v3454 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3543 = stablehlo.transpose %v3541, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3544 = stablehlo.transpose %v3542, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3545 = stablehlo.convolution(%v3543, %v3544)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3546 = stablehlo.transpose %v3545, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3547 = stablehlo.constant dense<0.003125> : tensor<64x64x3x3xf32>
    %v3548 = stablehlo.multiply %v3546, %v3547 : tensor<64x64x3x3xf32>
    %v3549 = stablehlo.subtract %s1b2W2, %v3548 : tensor<64x64x3x3xf32>
    %v3550 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3551 = stablehlo.reshape %v190 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3552 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v3553 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3554 = stablehlo.reduce(%v3551 init: %v3550) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3555 = stablehlo.broadcast_in_dim %v3554, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3556 = stablehlo.divide %v3555, %v3552 : tensor<32x64x56x56xf32>
    %v3557 = stablehlo.subtract %v3551, %v3556 : tensor<32x64x56x56xf32>
    %v3558 = stablehlo.multiply %v3557, %v3557 : tensor<32x64x56x56xf32>
    %v3559 = stablehlo.reduce(%v3558 init: %v3550) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3560 = stablehlo.broadcast_in_dim %v3559, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3561 = stablehlo.divide %v3560, %v3552 : tensor<32x64x56x56xf32>
    %v3562 = stablehlo.add %v3561, %v3553 : tensor<32x64x56x56xf32>
    %v3563 = stablehlo.rsqrt %v3562 : tensor<32x64x56x56xf32>
    %v3564 = stablehlo.multiply %v3557, %v3563 : tensor<32x64x56x56xf32>
    %v3565 = stablehlo.reshape %v3424 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3566 = stablehlo.multiply %v3565, %v3564 : tensor<32x64x56x56xf32>
    %v3567 = stablehlo.reduce(%v3566 init: %v3550) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3568 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3569 = stablehlo.multiply %v3567, %v3568 : tensor<64xf32>
    %v3570 = stablehlo.subtract %s1b2g2, %v3569 : tensor<64xf32>
    %v3571 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3572 = stablehlo.reshape %v3424 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3573 = stablehlo.reduce(%v3572 init: %v3571) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3574 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3575 = stablehlo.multiply %v3573, %v3574 : tensor<64xf32>
    %v3576 = stablehlo.subtract %s1b2bt2, %v3575 : tensor<64xf32>
    %v3577 = stablehlo.reshape %v3504 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3578 = stablehlo.reshape %v152 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3579 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3580 = stablehlo.compare GT, %v3578, %v3579 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3581 = stablehlo.select %v3580, %v3577, %v3579 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3582 = stablehlo.reshape %v3581 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3583 = stablehlo.reshape %v3582 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3584 = stablehlo.reshape %v128 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3585 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3586 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v3587 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3588 = stablehlo.reduce(%v3584 init: %v3585) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3589 = stablehlo.broadcast_in_dim %v3588, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3590 = stablehlo.divide %v3589, %v3586 : tensor<32x64x56x56xf32>
    %v3591 = stablehlo.subtract %v3584, %v3590 : tensor<32x64x56x56xf32>
    %v3592 = stablehlo.multiply %v3591, %v3591 : tensor<32x64x56x56xf32>
    %v3593 = stablehlo.reduce(%v3592 init: %v3585) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3594 = stablehlo.broadcast_in_dim %v3593, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3595 = stablehlo.divide %v3594, %v3586 : tensor<32x64x56x56xf32>
    %v3596 = stablehlo.add %v3595, %v3587 : tensor<32x64x56x56xf32>
    %v3597 = stablehlo.rsqrt %v3596 : tensor<32x64x56x56xf32>
    %v3598 = stablehlo.multiply %v3591, %v3597 : tensor<32x64x56x56xf32>
    %v3599 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3600 = stablehlo.multiply %v3599, %v3583 : tensor<32x64x56x56xf32>
    %v3601 = stablehlo.reduce(%v3600 init: %v3585) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3602 = stablehlo.broadcast_in_dim %v3601, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3603 = stablehlo.multiply %v3598, %v3600 : tensor<32x64x56x56xf32>
    %v3604 = stablehlo.reduce(%v3603 init: %v3585) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3605 = stablehlo.broadcast_in_dim %v3604, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3606 = stablehlo.multiply %v3600, %v3586 : tensor<32x64x56x56xf32>
    %v3607 = stablehlo.subtract %v3606, %v3602 : tensor<32x64x56x56xf32>
    %v3608 = stablehlo.multiply %v3598, %v3605 : tensor<32x64x56x56xf32>
    %v3609 = stablehlo.subtract %v3607, %v3608 : tensor<32x64x56x56xf32>
    %v3610 = stablehlo.divide %v3597, %v3586 : tensor<32x64x56x56xf32>
    %v3611 = stablehlo.multiply %v3610, %v3609 : tensor<32x64x56x56xf32>
    %v3612 = stablehlo.reshape %v3611 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3613 = stablehlo.reshape %v3612 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3614 = stablehlo.transpose %s1b1W2, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3615 = stablehlo.reverse %v3614, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3616 = stablehlo.convolution(%v3613, %v3615)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3617 = stablehlo.reshape %v3616 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3618 = stablehlo.reshape %v3617 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3619 = stablehlo.reshape %v119 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3620 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3621 = stablehlo.compare GT, %v3619, %v3620 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3622 = stablehlo.select %v3621, %v3618, %v3620 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3623 = stablehlo.reshape %v3622 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3624 = stablehlo.reshape %v3623 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3625 = stablehlo.reshape %v99 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3626 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3627 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v3628 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3629 = stablehlo.reduce(%v3625 init: %v3626) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3630 = stablehlo.broadcast_in_dim %v3629, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3631 = stablehlo.divide %v3630, %v3627 : tensor<32x64x56x56xf32>
    %v3632 = stablehlo.subtract %v3625, %v3631 : tensor<32x64x56x56xf32>
    %v3633 = stablehlo.multiply %v3632, %v3632 : tensor<32x64x56x56xf32>
    %v3634 = stablehlo.reduce(%v3633 init: %v3626) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3635 = stablehlo.broadcast_in_dim %v3634, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3636 = stablehlo.divide %v3635, %v3627 : tensor<32x64x56x56xf32>
    %v3637 = stablehlo.add %v3636, %v3628 : tensor<32x64x56x56xf32>
    %v3638 = stablehlo.rsqrt %v3637 : tensor<32x64x56x56xf32>
    %v3639 = stablehlo.multiply %v3632, %v3638 : tensor<32x64x56x56xf32>
    %v3640 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3641 = stablehlo.multiply %v3640, %v3624 : tensor<32x64x56x56xf32>
    %v3642 = stablehlo.reduce(%v3641 init: %v3626) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3643 = stablehlo.broadcast_in_dim %v3642, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3644 = stablehlo.multiply %v3639, %v3641 : tensor<32x64x56x56xf32>
    %v3645 = stablehlo.reduce(%v3644 init: %v3626) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3646 = stablehlo.broadcast_in_dim %v3645, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3647 = stablehlo.multiply %v3641, %v3627 : tensor<32x64x56x56xf32>
    %v3648 = stablehlo.subtract %v3647, %v3643 : tensor<32x64x56x56xf32>
    %v3649 = stablehlo.multiply %v3639, %v3646 : tensor<32x64x56x56xf32>
    %v3650 = stablehlo.subtract %v3648, %v3649 : tensor<32x64x56x56xf32>
    %v3651 = stablehlo.divide %v3638, %v3627 : tensor<32x64x56x56xf32>
    %v3652 = stablehlo.multiply %v3651, %v3650 : tensor<32x64x56x56xf32>
    %v3653 = stablehlo.reshape %v3652 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3654 = stablehlo.reshape %v3653 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3655 = stablehlo.transpose %s1b1W1, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3656 = stablehlo.reverse %v3655, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3657 = stablehlo.convolution(%v3654, %v3656)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3658 = stablehlo.reshape %v3657 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3659 = stablehlo.reshape %v3658 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3660 = stablehlo.reshape %v3582 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3661 = stablehlo.add %v3659, %v3660 : tensor<32x64x56x56xf32>
    %v3662 = stablehlo.reshape %v3661 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3663 = stablehlo.reshape %v94 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3664 = stablehlo.reshape %v3653 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3665 = stablehlo.transpose %v3663, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3666 = stablehlo.transpose %v3664, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3667 = stablehlo.convolution(%v3665, %v3666)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3668 = stablehlo.transpose %v3667, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3669 = stablehlo.constant dense<0.003125> : tensor<64x64x3x3xf32>
    %v3670 = stablehlo.multiply %v3668, %v3669 : tensor<64x64x3x3xf32>
    %v3671 = stablehlo.subtract %s1b1W1, %v3670 : tensor<64x64x3x3xf32>
    %v3672 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3673 = stablehlo.reshape %v99 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3674 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v3675 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3676 = stablehlo.reduce(%v3673 init: %v3672) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3677 = stablehlo.broadcast_in_dim %v3676, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3678 = stablehlo.divide %v3677, %v3674 : tensor<32x64x56x56xf32>
    %v3679 = stablehlo.subtract %v3673, %v3678 : tensor<32x64x56x56xf32>
    %v3680 = stablehlo.multiply %v3679, %v3679 : tensor<32x64x56x56xf32>
    %v3681 = stablehlo.reduce(%v3680 init: %v3672) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3682 = stablehlo.broadcast_in_dim %v3681, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3683 = stablehlo.divide %v3682, %v3674 : tensor<32x64x56x56xf32>
    %v3684 = stablehlo.add %v3683, %v3675 : tensor<32x64x56x56xf32>
    %v3685 = stablehlo.rsqrt %v3684 : tensor<32x64x56x56xf32>
    %v3686 = stablehlo.multiply %v3679, %v3685 : tensor<32x64x56x56xf32>
    %v3687 = stablehlo.reshape %v3623 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3688 = stablehlo.multiply %v3687, %v3686 : tensor<32x64x56x56xf32>
    %v3689 = stablehlo.reduce(%v3688 init: %v3672) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3690 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3691 = stablehlo.multiply %v3689, %v3690 : tensor<64xf32>
    %v3692 = stablehlo.subtract %s1b1g1, %v3691 : tensor<64xf32>
    %v3693 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3694 = stablehlo.reshape %v3623 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3695 = stablehlo.reduce(%v3694 init: %v3693) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3696 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3697 = stablehlo.multiply %v3695, %v3696 : tensor<64xf32>
    %v3698 = stablehlo.subtract %s1b1bt1, %v3697 : tensor<64xf32>
    %v3699 = stablehlo.reshape %v123 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3700 = stablehlo.reshape %v3612 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3701 = stablehlo.transpose %v3699, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3702 = stablehlo.transpose %v3700, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3703 = stablehlo.convolution(%v3701, %v3702)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3704 = stablehlo.transpose %v3703, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3705 = stablehlo.constant dense<0.003125> : tensor<64x64x3x3xf32>
    %v3706 = stablehlo.multiply %v3704, %v3705 : tensor<64x64x3x3xf32>
    %v3707 = stablehlo.subtract %s1b1W2, %v3706 : tensor<64x64x3x3xf32>
    %v3708 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3709 = stablehlo.reshape %v128 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3710 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v3711 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3712 = stablehlo.reduce(%v3709 init: %v3708) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3713 = stablehlo.broadcast_in_dim %v3712, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3714 = stablehlo.divide %v3713, %v3710 : tensor<32x64x56x56xf32>
    %v3715 = stablehlo.subtract %v3709, %v3714 : tensor<32x64x56x56xf32>
    %v3716 = stablehlo.multiply %v3715, %v3715 : tensor<32x64x56x56xf32>
    %v3717 = stablehlo.reduce(%v3716 init: %v3708) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3718 = stablehlo.broadcast_in_dim %v3717, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3719 = stablehlo.divide %v3718, %v3710 : tensor<32x64x56x56xf32>
    %v3720 = stablehlo.add %v3719, %v3711 : tensor<32x64x56x56xf32>
    %v3721 = stablehlo.rsqrt %v3720 : tensor<32x64x56x56xf32>
    %v3722 = stablehlo.multiply %v3715, %v3721 : tensor<32x64x56x56xf32>
    %v3723 = stablehlo.reshape %v3582 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3724 = stablehlo.multiply %v3723, %v3722 : tensor<32x64x56x56xf32>
    %v3725 = stablehlo.reduce(%v3724 init: %v3708) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3726 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3727 = stablehlo.multiply %v3725, %v3726 : tensor<64xf32>
    %v3728 = stablehlo.subtract %s1b1g2, %v3727 : tensor<64xf32>
    %v3729 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3730 = stablehlo.reshape %v3582 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3731 = stablehlo.reduce(%v3730 init: %v3729) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3732 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3733 = stablehlo.multiply %v3731, %v3732 : tensor<64xf32>
    %v3734 = stablehlo.subtract %s1b1bt2, %v3733 : tensor<64xf32>
    %v3735 = stablehlo.reshape %v3662 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3736 = stablehlo.reshape %v90 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3737 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3738 = stablehlo.compare GT, %v3736, %v3737 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3739 = stablehlo.select %v3738, %v3735, %v3737 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3740 = stablehlo.reshape %v3739 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3741 = stablehlo.reshape %v3740 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3742 = stablehlo.reshape %v66 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3743 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3744 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v3745 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3746 = stablehlo.reduce(%v3742 init: %v3743) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3747 = stablehlo.broadcast_in_dim %v3746, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3748 = stablehlo.divide %v3747, %v3744 : tensor<32x64x56x56xf32>
    %v3749 = stablehlo.subtract %v3742, %v3748 : tensor<32x64x56x56xf32>
    %v3750 = stablehlo.multiply %v3749, %v3749 : tensor<32x64x56x56xf32>
    %v3751 = stablehlo.reduce(%v3750 init: %v3743) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3752 = stablehlo.broadcast_in_dim %v3751, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3753 = stablehlo.divide %v3752, %v3744 : tensor<32x64x56x56xf32>
    %v3754 = stablehlo.add %v3753, %v3745 : tensor<32x64x56x56xf32>
    %v3755 = stablehlo.rsqrt %v3754 : tensor<32x64x56x56xf32>
    %v3756 = stablehlo.multiply %v3749, %v3755 : tensor<32x64x56x56xf32>
    %v3757 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3758 = stablehlo.multiply %v3757, %v3741 : tensor<32x64x56x56xf32>
    %v3759 = stablehlo.reduce(%v3758 init: %v3743) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3760 = stablehlo.broadcast_in_dim %v3759, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3761 = stablehlo.multiply %v3756, %v3758 : tensor<32x64x56x56xf32>
    %v3762 = stablehlo.reduce(%v3761 init: %v3743) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3763 = stablehlo.broadcast_in_dim %v3762, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3764 = stablehlo.multiply %v3758, %v3744 : tensor<32x64x56x56xf32>
    %v3765 = stablehlo.subtract %v3764, %v3760 : tensor<32x64x56x56xf32>
    %v3766 = stablehlo.multiply %v3756, %v3763 : tensor<32x64x56x56xf32>
    %v3767 = stablehlo.subtract %v3765, %v3766 : tensor<32x64x56x56xf32>
    %v3768 = stablehlo.divide %v3755, %v3744 : tensor<32x64x56x56xf32>
    %v3769 = stablehlo.multiply %v3768, %v3767 : tensor<32x64x56x56xf32>
    %v3770 = stablehlo.reshape %v3769 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3771 = stablehlo.reshape %v3770 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3772 = stablehlo.transpose %s1b0W2, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3773 = stablehlo.reverse %v3772, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3774 = stablehlo.convolution(%v3771, %v3773)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3775 = stablehlo.reshape %v3774 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3776 = stablehlo.reshape %v3775 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3777 = stablehlo.reshape %v57 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3778 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3779 = stablehlo.compare GT, %v3777, %v3778 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3780 = stablehlo.select %v3779, %v3776, %v3778 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3781 = stablehlo.reshape %v3780 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3782 = stablehlo.reshape %v3781 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3783 = stablehlo.reshape %v37 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3784 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3785 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v3786 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3787 = stablehlo.reduce(%v3783 init: %v3784) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3788 = stablehlo.broadcast_in_dim %v3787, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3789 = stablehlo.divide %v3788, %v3785 : tensor<32x64x56x56xf32>
    %v3790 = stablehlo.subtract %v3783, %v3789 : tensor<32x64x56x56xf32>
    %v3791 = stablehlo.multiply %v3790, %v3790 : tensor<32x64x56x56xf32>
    %v3792 = stablehlo.reduce(%v3791 init: %v3784) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3793 = stablehlo.broadcast_in_dim %v3792, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3794 = stablehlo.divide %v3793, %v3785 : tensor<32x64x56x56xf32>
    %v3795 = stablehlo.add %v3794, %v3786 : tensor<32x64x56x56xf32>
    %v3796 = stablehlo.rsqrt %v3795 : tensor<32x64x56x56xf32>
    %v3797 = stablehlo.multiply %v3790, %v3796 : tensor<32x64x56x56xf32>
    %v3798 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3799 = stablehlo.multiply %v3798, %v3782 : tensor<32x64x56x56xf32>
    %v3800 = stablehlo.reduce(%v3799 init: %v3784) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3801 = stablehlo.broadcast_in_dim %v3800, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3802 = stablehlo.multiply %v3797, %v3799 : tensor<32x64x56x56xf32>
    %v3803 = stablehlo.reduce(%v3802 init: %v3784) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3804 = stablehlo.broadcast_in_dim %v3803, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3805 = stablehlo.multiply %v3799, %v3785 : tensor<32x64x56x56xf32>
    %v3806 = stablehlo.subtract %v3805, %v3801 : tensor<32x64x56x56xf32>
    %v3807 = stablehlo.multiply %v3797, %v3804 : tensor<32x64x56x56xf32>
    %v3808 = stablehlo.subtract %v3806, %v3807 : tensor<32x64x56x56xf32>
    %v3809 = stablehlo.divide %v3796, %v3785 : tensor<32x64x56x56xf32>
    %v3810 = stablehlo.multiply %v3809, %v3808 : tensor<32x64x56x56xf32>
    %v3811 = stablehlo.reshape %v3810 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3812 = stablehlo.reshape %v3811 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3813 = stablehlo.transpose %s1b0W1, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3814 = stablehlo.reverse %v3813, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3815 = stablehlo.convolution(%v3812, %v3814)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3816 = stablehlo.reshape %v3815 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3817 = stablehlo.reshape %v3816 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3818 = stablehlo.reshape %v3740 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3819 = stablehlo.add %v3817, %v3818 : tensor<32x64x56x56xf32>
    %v3820 = stablehlo.reshape %v3819 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3821 = stablehlo.reshape %v32 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3822 = stablehlo.reshape %v3811 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3823 = stablehlo.transpose %v3821, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3824 = stablehlo.transpose %v3822, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3825 = stablehlo.convolution(%v3823, %v3824)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3826 = stablehlo.transpose %v3825, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3827 = stablehlo.constant dense<0.003125> : tensor<64x64x3x3xf32>
    %v3828 = stablehlo.multiply %v3826, %v3827 : tensor<64x64x3x3xf32>
    %v3829 = stablehlo.subtract %s1b0W1, %v3828 : tensor<64x64x3x3xf32>
    %v3830 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3831 = stablehlo.reshape %v37 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3832 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v3833 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3834 = stablehlo.reduce(%v3831 init: %v3830) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3835 = stablehlo.broadcast_in_dim %v3834, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3836 = stablehlo.divide %v3835, %v3832 : tensor<32x64x56x56xf32>
    %v3837 = stablehlo.subtract %v3831, %v3836 : tensor<32x64x56x56xf32>
    %v3838 = stablehlo.multiply %v3837, %v3837 : tensor<32x64x56x56xf32>
    %v3839 = stablehlo.reduce(%v3838 init: %v3830) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3840 = stablehlo.broadcast_in_dim %v3839, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3841 = stablehlo.divide %v3840, %v3832 : tensor<32x64x56x56xf32>
    %v3842 = stablehlo.add %v3841, %v3833 : tensor<32x64x56x56xf32>
    %v3843 = stablehlo.rsqrt %v3842 : tensor<32x64x56x56xf32>
    %v3844 = stablehlo.multiply %v3837, %v3843 : tensor<32x64x56x56xf32>
    %v3845 = stablehlo.reshape %v3781 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3846 = stablehlo.multiply %v3845, %v3844 : tensor<32x64x56x56xf32>
    %v3847 = stablehlo.reduce(%v3846 init: %v3830) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3848 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3849 = stablehlo.multiply %v3847, %v3848 : tensor<64xf32>
    %v3850 = stablehlo.subtract %s1b0g1, %v3849 : tensor<64xf32>
    %v3851 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3852 = stablehlo.reshape %v3781 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3853 = stablehlo.reduce(%v3852 init: %v3851) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3854 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3855 = stablehlo.multiply %v3853, %v3854 : tensor<64xf32>
    %v3856 = stablehlo.subtract %s1b0bt1, %v3855 : tensor<64xf32>
    %v3857 = stablehlo.reshape %v61 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3858 = stablehlo.reshape %v3770 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3859 = stablehlo.transpose %v3857, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3860 = stablehlo.transpose %v3858, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3861 = stablehlo.convolution(%v3859, %v3860)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3862 = stablehlo.transpose %v3861, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3863 = stablehlo.constant dense<0.003125> : tensor<64x64x3x3xf32>
    %v3864 = stablehlo.multiply %v3862, %v3863 : tensor<64x64x3x3xf32>
    %v3865 = stablehlo.subtract %s1b0W2, %v3864 : tensor<64x64x3x3xf32>
    %v3866 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3867 = stablehlo.reshape %v66 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3868 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v3869 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3870 = stablehlo.reduce(%v3867 init: %v3866) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3871 = stablehlo.broadcast_in_dim %v3870, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3872 = stablehlo.divide %v3871, %v3868 : tensor<32x64x56x56xf32>
    %v3873 = stablehlo.subtract %v3867, %v3872 : tensor<32x64x56x56xf32>
    %v3874 = stablehlo.multiply %v3873, %v3873 : tensor<32x64x56x56xf32>
    %v3875 = stablehlo.reduce(%v3874 init: %v3866) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3876 = stablehlo.broadcast_in_dim %v3875, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3877 = stablehlo.divide %v3876, %v3868 : tensor<32x64x56x56xf32>
    %v3878 = stablehlo.add %v3877, %v3869 : tensor<32x64x56x56xf32>
    %v3879 = stablehlo.rsqrt %v3878 : tensor<32x64x56x56xf32>
    %v3880 = stablehlo.multiply %v3873, %v3879 : tensor<32x64x56x56xf32>
    %v3881 = stablehlo.reshape %v3740 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3882 = stablehlo.multiply %v3881, %v3880 : tensor<32x64x56x56xf32>
    %v3883 = stablehlo.reduce(%v3882 init: %v3866) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3884 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3885 = stablehlo.multiply %v3883, %v3884 : tensor<64xf32>
    %v3886 = stablehlo.subtract %s1b0g2, %v3885 : tensor<64xf32>
    %v3887 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3888 = stablehlo.reshape %v3740 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3889 = stablehlo.reduce(%v3888 init: %v3887) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3890 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3891 = stablehlo.multiply %v3889, %v3890 : tensor<64xf32>
    %v3892 = stablehlo.subtract %s1b0bt2, %v3891 : tensor<64xf32>
    %v3893 = stablehlo.reshape %v28 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3894 = stablehlo.reshape %v3820 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3895 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3896 = "stablehlo.select_and_scatter"(%v3893, %v3894, %v3895) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>} : (tensor<32x64x112x112xf32>, tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x112x112xf32>
    %v3897 = stablehlo.reshape %v3896 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v3898 = stablehlo.reshape %v3897 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3899 = stablehlo.reshape %v24 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3900 = stablehlo.constant dense<0.0> : tensor<32x64x112x112xf32>
    %v3901 = stablehlo.compare GT, %v3899, %v3900 : (tensor<32x64x112x112xf32>, tensor<32x64x112x112xf32>) -> tensor<32x64x112x112xi1>
    %v3902 = stablehlo.select %v3901, %v3898, %v3900 : tensor<32x64x112x112xi1>, tensor<32x64x112x112xf32>
    %v3903 = stablehlo.reshape %v3902 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v3904 = stablehlo.reshape %v3903 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3905 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3906 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3907 = stablehlo.constant dense<12544.0> : tensor<32x64x112x112xf32>
    %v3908 = stablehlo.constant dense<1.0e-05> : tensor<32x64x112x112xf32>
    %v3909 = stablehlo.reduce(%v3905 init: %v3906) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3910 = stablehlo.broadcast_in_dim %v3909, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v3911 = stablehlo.divide %v3910, %v3907 : tensor<32x64x112x112xf32>
    %v3912 = stablehlo.subtract %v3905, %v3911 : tensor<32x64x112x112xf32>
    %v3913 = stablehlo.multiply %v3912, %v3912 : tensor<32x64x112x112xf32>
    %v3914 = stablehlo.reduce(%v3913 init: %v3906) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3915 = stablehlo.broadcast_in_dim %v3914, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v3916 = stablehlo.divide %v3915, %v3907 : tensor<32x64x112x112xf32>
    %v3917 = stablehlo.add %v3916, %v3908 : tensor<32x64x112x112xf32>
    %v3918 = stablehlo.rsqrt %v3917 : tensor<32x64x112x112xf32>
    %v3919 = stablehlo.multiply %v3912, %v3918 : tensor<32x64x112x112xf32>
    %v3920 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3921 = stablehlo.multiply %v3920, %v3904 : tensor<32x64x112x112xf32>
    %v3922 = stablehlo.reduce(%v3921 init: %v3906) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3923 = stablehlo.broadcast_in_dim %v3922, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v3924 = stablehlo.multiply %v3919, %v3921 : tensor<32x64x112x112xf32>
    %v3925 = stablehlo.reduce(%v3924 init: %v3906) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3926 = stablehlo.broadcast_in_dim %v3925, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v3927 = stablehlo.multiply %v3921, %v3907 : tensor<32x64x112x112xf32>
    %v3928 = stablehlo.subtract %v3927, %v3923 : tensor<32x64x112x112xf32>
    %v3929 = stablehlo.multiply %v3919, %v3926 : tensor<32x64x112x112xf32>
    %v3930 = stablehlo.subtract %v3928, %v3929 : tensor<32x64x112x112xf32>
    %v3931 = stablehlo.divide %v3918, %v3907 : tensor<32x64x112x112xf32>
    %v3932 = stablehlo.multiply %v3931, %v3930 : tensor<32x64x112x112xf32>
    %v3933 = stablehlo.reshape %v3932 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v3934 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v3935 = stablehlo.reshape %v3933 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3936 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3937 = stablehlo.pad %v3935, %v3936, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64x224x224xf32>
    %v3938 = stablehlo.transpose %v3934, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %v3939 = stablehlo.transpose %v3937, dims = [1, 0, 2, 3] : (tensor<32x64x224x224xf32>) -> tensor<64x32x224x224xf32>
    %v3940 = stablehlo.convolution(%v3938, %v3939)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<64x32x224x224xf32>) -> tensor<3x64x7x7xf32>
    %v3941 = stablehlo.transpose %v3940, dims = [1, 0, 2, 3] : (tensor<3x64x7x7xf32>) -> tensor<64x3x7x7xf32>
    %v3942 = stablehlo.constant dense<0.003125> : tensor<64x3x7x7xf32>
    %v3943 = stablehlo.multiply %v3941, %v3942 : tensor<64x3x7x7xf32>
    %v3944 = stablehlo.subtract %sW, %v3943 : tensor<64x3x7x7xf32>
    %v3945 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3946 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3947 = stablehlo.constant dense<12544.0> : tensor<32x64x112x112xf32>
    %v3948 = stablehlo.constant dense<1.0e-05> : tensor<32x64x112x112xf32>
    %v3949 = stablehlo.reduce(%v3946 init: %v3945) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3950 = stablehlo.broadcast_in_dim %v3949, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v3951 = stablehlo.divide %v3950, %v3947 : tensor<32x64x112x112xf32>
    %v3952 = stablehlo.subtract %v3946, %v3951 : tensor<32x64x112x112xf32>
    %v3953 = stablehlo.multiply %v3952, %v3952 : tensor<32x64x112x112xf32>
    %v3954 = stablehlo.reduce(%v3953 init: %v3945) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3955 = stablehlo.broadcast_in_dim %v3954, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v3956 = stablehlo.divide %v3955, %v3947 : tensor<32x64x112x112xf32>
    %v3957 = stablehlo.add %v3956, %v3948 : tensor<32x64x112x112xf32>
    %v3958 = stablehlo.rsqrt %v3957 : tensor<32x64x112x112xf32>
    %v3959 = stablehlo.multiply %v3952, %v3958 : tensor<32x64x112x112xf32>
    %v3960 = stablehlo.reshape %v3903 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3961 = stablehlo.multiply %v3960, %v3959 : tensor<32x64x112x112xf32>
    %v3962 = stablehlo.reduce(%v3961 init: %v3945) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3963 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3964 = stablehlo.multiply %v3962, %v3963 : tensor<64xf32>
    %v3965 = stablehlo.subtract %sg, %v3964 : tensor<64xf32>
    %v3966 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3967 = stablehlo.reshape %v3903 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3968 = stablehlo.reduce(%v3967 init: %v3966) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3969 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3970 = stablehlo.multiply %v3968, %v3969 : tensor<64xf32>
    %v3971 = stablehlo.subtract %sbt, %v3970 : tensor<64xf32>
    return %v3944, %v3965, %v3971, %v3829, %v3850, %v3856, %v3865, %v3886, %v3892, %v3671, %v3692, %v3698, %v3707, %v3728, %v3734, %v3513, %v3534, %v3540, %v3549, %v3570, %v3576, %v3317, %v3338, %v3344, %v3353, %v3374, %v3380, %v3391, %v3412, %v3418, %v3118, %v3139, %v3145, %v3154, %v3175, %v3181, %v2960, %v2981, %v2987, %v2996, %v3017, %v3023, %v2802, %v2823, %v2829, %v2838, %v2859, %v2865, %v2606, %v2627, %v2633, %v2642, %v2663, %v2669, %v2680, %v2701, %v2707, %v2407, %v2428, %v2434, %v2443, %v2464, %v2470, %v2249, %v2270, %v2276, %v2285, %v2306, %v2312, %v2091, %v2112, %v2118, %v2127, %v2148, %v2154, %v1933, %v1954, %v1960, %v1969, %v1990, %v1996, %v1775, %v1796, %v1802, %v1811, %v1832, %v1838, %v1579, %v1600, %v1606, %v1615, %v1636, %v1642, %v1653, %v1674, %v1680, %v1380, %v1401, %v1407, %v1416, %v1437, %v1443, %v1222, %v1243, %v1249, %v1258, %v1279, %v1285, %v1122, %v1127 : tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>
  }
}
