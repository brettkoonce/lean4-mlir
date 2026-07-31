module @m {
  func.func @resnet34_train_step(%x: tensor<32x150528xf32>, %sW: tensor<64x3x7x7xf32>, %sg: tensor<64xf32>, %sbt: tensor<64xf32>, %s1b0W1: tensor<64x64x3x3xf32>, %s1b0g1: tensor<64xf32>, %s1b0bt1: tensor<64xf32>, %s1b0W2: tensor<64x64x3x3xf32>, %s1b0g2: tensor<64xf32>, %s1b0bt2: tensor<64xf32>, %s1b1W1: tensor<64x64x3x3xf32>, %s1b1g1: tensor<64xf32>, %s1b1bt1: tensor<64xf32>, %s1b1W2: tensor<64x64x3x3xf32>, %s1b1g2: tensor<64xf32>, %s1b1bt2: tensor<64xf32>, %s1b2W1: tensor<64x64x3x3xf32>, %s1b2g1: tensor<64xf32>, %s1b2bt1: tensor<64xf32>, %s1b2W2: tensor<64x64x3x3xf32>, %s1b2g2: tensor<64xf32>, %s1b2bt2: tensor<64xf32>, %d2W1: tensor<128x64x3x3xf32>, %d2g1: tensor<128xf32>, %d2bt1: tensor<128xf32>, %d2W2: tensor<128x128x3x3xf32>, %d2g2: tensor<128xf32>, %d2bt2: tensor<128xf32>, %d2Wp: tensor<128x64x3x3xf32>, %d2gp: tensor<128xf32>, %d2btp: tensor<128xf32>, %s2b0W1: tensor<128x128x3x3xf32>, %s2b0g1: tensor<128xf32>, %s2b0bt1: tensor<128xf32>, %s2b0W2: tensor<128x128x3x3xf32>, %s2b0g2: tensor<128xf32>, %s2b0bt2: tensor<128xf32>, %s2b1W1: tensor<128x128x3x3xf32>, %s2b1g1: tensor<128xf32>, %s2b1bt1: tensor<128xf32>, %s2b1W2: tensor<128x128x3x3xf32>, %s2b1g2: tensor<128xf32>, %s2b1bt2: tensor<128xf32>, %s2b2W1: tensor<128x128x3x3xf32>, %s2b2g1: tensor<128xf32>, %s2b2bt1: tensor<128xf32>, %s2b2W2: tensor<128x128x3x3xf32>, %s2b2g2: tensor<128xf32>, %s2b2bt2: tensor<128xf32>, %d3W1: tensor<256x128x3x3xf32>, %d3g1: tensor<256xf32>, %d3bt1: tensor<256xf32>, %d3W2: tensor<256x256x3x3xf32>, %d3g2: tensor<256xf32>, %d3bt2: tensor<256xf32>, %d3Wp: tensor<256x128x3x3xf32>, %d3gp: tensor<256xf32>, %d3btp: tensor<256xf32>, %s3b0W1: tensor<256x256x3x3xf32>, %s3b0g1: tensor<256xf32>, %s3b0bt1: tensor<256xf32>, %s3b0W2: tensor<256x256x3x3xf32>, %s3b0g2: tensor<256xf32>, %s3b0bt2: tensor<256xf32>, %s3b1W1: tensor<256x256x3x3xf32>, %s3b1g1: tensor<256xf32>, %s3b1bt1: tensor<256xf32>, %s3b1W2: tensor<256x256x3x3xf32>, %s3b1g2: tensor<256xf32>, %s3b1bt2: tensor<256xf32>, %s3b2W1: tensor<256x256x3x3xf32>, %s3b2g1: tensor<256xf32>, %s3b2bt1: tensor<256xf32>, %s3b2W2: tensor<256x256x3x3xf32>, %s3b2g2: tensor<256xf32>, %s3b2bt2: tensor<256xf32>, %s3b3W1: tensor<256x256x3x3xf32>, %s3b3g1: tensor<256xf32>, %s3b3bt1: tensor<256xf32>, %s3b3W2: tensor<256x256x3x3xf32>, %s3b3g2: tensor<256xf32>, %s3b3bt2: tensor<256xf32>, %s3b4W1: tensor<256x256x3x3xf32>, %s3b4g1: tensor<256xf32>, %s3b4bt1: tensor<256xf32>, %s3b4W2: tensor<256x256x3x3xf32>, %s3b4g2: tensor<256xf32>, %s3b4bt2: tensor<256xf32>, %d4W1: tensor<512x256x3x3xf32>, %d4g1: tensor<512xf32>, %d4bt1: tensor<512xf32>, %d4W2: tensor<512x512x3x3xf32>, %d4g2: tensor<512xf32>, %d4bt2: tensor<512xf32>, %d4Wp: tensor<512x256x3x3xf32>, %d4gp: tensor<512xf32>, %d4btp: tensor<512xf32>, %s4b0W1: tensor<512x512x3x3xf32>, %s4b0g1: tensor<512xf32>, %s4b0bt1: tensor<512xf32>, %s4b0W2: tensor<512x512x3x3xf32>, %s4b0g2: tensor<512xf32>, %s4b0bt2: tensor<512xf32>, %s4b1W1: tensor<512x512x3x3xf32>, %s4b1g1: tensor<512xf32>, %s4b1bt1: tensor<512xf32>, %s4b1W2: tensor<512x512x3x3xf32>, %s4b1g2: tensor<512xf32>, %s4b1bt2: tensor<512xf32>, %Wd: tensor<512x10xf32>, %bd: tensor<10xf32>, %onehot: tensor<32x10xf32>) -> (tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>) {
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
    %v25 = stablehlo.constant dense<0.0> : tensor<32x802816xf32>
    %v26 = stablehlo.maximum %v24, %v25 : tensor<32x802816xf32>
    %v27 = stablehlo.reshape %v26 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v28 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v29 = "stablehlo.reduce_window"(%v27, %v28) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64x56x56xf32>
    %v30 = stablehlo.reshape %v29 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v32 = stablehlo.convolution(%v31, %s1b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v33 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v34 = stablehlo.add %v32, %v33 : tensor<32x64x56x56xf32>
    %v35 = stablehlo.reshape %v34 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v36 = stablehlo.reshape %v35 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v37 = stablehlo.constant dense<0.0> : tensor<f32>
    %v38 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v39 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v40 = stablehlo.reduce(%v36 init: %v37) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v41 = stablehlo.broadcast_in_dim %v40, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v42 = stablehlo.divide %v41, %v38 : tensor<32x64x56x56xf32>
    %v43 = stablehlo.subtract %v36, %v42 : tensor<32x64x56x56xf32>
    %v44 = stablehlo.multiply %v43, %v43 : tensor<32x64x56x56xf32>
    %v45 = stablehlo.reduce(%v44 init: %v37) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v46 = stablehlo.broadcast_in_dim %v45, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v47 = stablehlo.divide %v46, %v38 : tensor<32x64x56x56xf32>
    %v48 = stablehlo.add %v47, %v39 : tensor<32x64x56x56xf32>
    %v49 = stablehlo.rsqrt %v48 : tensor<32x64x56x56xf32>
    %v50 = stablehlo.multiply %v43, %v49 : tensor<32x64x56x56xf32>
    %v51 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v52 = stablehlo.broadcast_in_dim %s1b0bt1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v53 = stablehlo.multiply %v50, %v51 : tensor<32x64x56x56xf32>
    %v54 = stablehlo.add %v53, %v52 : tensor<32x64x56x56xf32>
    %v55 = stablehlo.reshape %v54 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v56 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v57 = stablehlo.maximum %v55, %v56 : tensor<32x200704xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v59 = stablehlo.convolution(%v58, %s1b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v60 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v61 = stablehlo.add %v59, %v60 : tensor<32x64x56x56xf32>
    %v62 = stablehlo.reshape %v61 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v63 = stablehlo.reshape %v62 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v64 = stablehlo.constant dense<0.0> : tensor<f32>
    %v65 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v66 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v67 = stablehlo.reduce(%v63 init: %v64) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v68 = stablehlo.broadcast_in_dim %v67, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v69 = stablehlo.divide %v68, %v65 : tensor<32x64x56x56xf32>
    %v70 = stablehlo.subtract %v63, %v69 : tensor<32x64x56x56xf32>
    %v71 = stablehlo.multiply %v70, %v70 : tensor<32x64x56x56xf32>
    %v72 = stablehlo.reduce(%v71 init: %v64) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v73 = stablehlo.broadcast_in_dim %v72, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v74 = stablehlo.divide %v73, %v65 : tensor<32x64x56x56xf32>
    %v75 = stablehlo.add %v74, %v66 : tensor<32x64x56x56xf32>
    %v76 = stablehlo.rsqrt %v75 : tensor<32x64x56x56xf32>
    %v77 = stablehlo.multiply %v70, %v76 : tensor<32x64x56x56xf32>
    %v78 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v79 = stablehlo.broadcast_in_dim %s1b0bt2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v80 = stablehlo.multiply %v77, %v78 : tensor<32x64x56x56xf32>
    %v81 = stablehlo.add %v80, %v79 : tensor<32x64x56x56xf32>
    %v82 = stablehlo.reshape %v81 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v83 = stablehlo.add %v82, %v30 : tensor<32x200704xf32>
    %v84 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v85 = stablehlo.maximum %v83, %v84 : tensor<32x200704xf32>
    %v86 = stablehlo.reshape %v85 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v87 = stablehlo.convolution(%v86, %s1b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v88 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v89 = stablehlo.add %v87, %v88 : tensor<32x64x56x56xf32>
    %v90 = stablehlo.reshape %v89 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v91 = stablehlo.reshape %v90 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v92 = stablehlo.constant dense<0.0> : tensor<f32>
    %v93 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v94 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v95 = stablehlo.reduce(%v91 init: %v92) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v96 = stablehlo.broadcast_in_dim %v95, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v97 = stablehlo.divide %v96, %v93 : tensor<32x64x56x56xf32>
    %v98 = stablehlo.subtract %v91, %v97 : tensor<32x64x56x56xf32>
    %v99 = stablehlo.multiply %v98, %v98 : tensor<32x64x56x56xf32>
    %v100 = stablehlo.reduce(%v99 init: %v92) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v101 = stablehlo.broadcast_in_dim %v100, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v102 = stablehlo.divide %v101, %v93 : tensor<32x64x56x56xf32>
    %v103 = stablehlo.add %v102, %v94 : tensor<32x64x56x56xf32>
    %v104 = stablehlo.rsqrt %v103 : tensor<32x64x56x56xf32>
    %v105 = stablehlo.multiply %v98, %v104 : tensor<32x64x56x56xf32>
    %v106 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v107 = stablehlo.broadcast_in_dim %s1b1bt1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v108 = stablehlo.multiply %v105, %v106 : tensor<32x64x56x56xf32>
    %v109 = stablehlo.add %v108, %v107 : tensor<32x64x56x56xf32>
    %v110 = stablehlo.reshape %v109 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v111 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v112 = stablehlo.maximum %v110, %v111 : tensor<32x200704xf32>
    %v113 = stablehlo.reshape %v112 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v114 = stablehlo.convolution(%v113, %s1b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v115 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v116 = stablehlo.add %v114, %v115 : tensor<32x64x56x56xf32>
    %v117 = stablehlo.reshape %v116 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v118 = stablehlo.reshape %v117 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v119 = stablehlo.constant dense<0.0> : tensor<f32>
    %v120 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v121 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v122 = stablehlo.reduce(%v118 init: %v119) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v123 = stablehlo.broadcast_in_dim %v122, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v124 = stablehlo.divide %v123, %v120 : tensor<32x64x56x56xf32>
    %v125 = stablehlo.subtract %v118, %v124 : tensor<32x64x56x56xf32>
    %v126 = stablehlo.multiply %v125, %v125 : tensor<32x64x56x56xf32>
    %v127 = stablehlo.reduce(%v126 init: %v119) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v128 = stablehlo.broadcast_in_dim %v127, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v129 = stablehlo.divide %v128, %v120 : tensor<32x64x56x56xf32>
    %v130 = stablehlo.add %v129, %v121 : tensor<32x64x56x56xf32>
    %v131 = stablehlo.rsqrt %v130 : tensor<32x64x56x56xf32>
    %v132 = stablehlo.multiply %v125, %v131 : tensor<32x64x56x56xf32>
    %v133 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v134 = stablehlo.broadcast_in_dim %s1b1bt2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v135 = stablehlo.multiply %v132, %v133 : tensor<32x64x56x56xf32>
    %v136 = stablehlo.add %v135, %v134 : tensor<32x64x56x56xf32>
    %v137 = stablehlo.reshape %v136 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v138 = stablehlo.add %v137, %v85 : tensor<32x200704xf32>
    %v139 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v140 = stablehlo.maximum %v138, %v139 : tensor<32x200704xf32>
    %v141 = stablehlo.reshape %v140 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v142 = stablehlo.convolution(%v141, %s1b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v143 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v144 = stablehlo.add %v142, %v143 : tensor<32x64x56x56xf32>
    %v145 = stablehlo.reshape %v144 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v146 = stablehlo.reshape %v145 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v147 = stablehlo.constant dense<0.0> : tensor<f32>
    %v148 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v149 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v150 = stablehlo.reduce(%v146 init: %v147) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v151 = stablehlo.broadcast_in_dim %v150, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v152 = stablehlo.divide %v151, %v148 : tensor<32x64x56x56xf32>
    %v153 = stablehlo.subtract %v146, %v152 : tensor<32x64x56x56xf32>
    %v154 = stablehlo.multiply %v153, %v153 : tensor<32x64x56x56xf32>
    %v155 = stablehlo.reduce(%v154 init: %v147) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v156 = stablehlo.broadcast_in_dim %v155, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v157 = stablehlo.divide %v156, %v148 : tensor<32x64x56x56xf32>
    %v158 = stablehlo.add %v157, %v149 : tensor<32x64x56x56xf32>
    %v159 = stablehlo.rsqrt %v158 : tensor<32x64x56x56xf32>
    %v160 = stablehlo.multiply %v153, %v159 : tensor<32x64x56x56xf32>
    %v161 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v162 = stablehlo.broadcast_in_dim %s1b2bt1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v163 = stablehlo.multiply %v160, %v161 : tensor<32x64x56x56xf32>
    %v164 = stablehlo.add %v163, %v162 : tensor<32x64x56x56xf32>
    %v165 = stablehlo.reshape %v164 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v166 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v167 = stablehlo.maximum %v165, %v166 : tensor<32x200704xf32>
    %v168 = stablehlo.reshape %v167 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v169 = stablehlo.convolution(%v168, %s1b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v170 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v171 = stablehlo.add %v169, %v170 : tensor<32x64x56x56xf32>
    %v172 = stablehlo.reshape %v171 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v173 = stablehlo.reshape %v172 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v174 = stablehlo.constant dense<0.0> : tensor<f32>
    %v175 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v176 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v177 = stablehlo.reduce(%v173 init: %v174) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v178 = stablehlo.broadcast_in_dim %v177, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v179 = stablehlo.divide %v178, %v175 : tensor<32x64x56x56xf32>
    %v180 = stablehlo.subtract %v173, %v179 : tensor<32x64x56x56xf32>
    %v181 = stablehlo.multiply %v180, %v180 : tensor<32x64x56x56xf32>
    %v182 = stablehlo.reduce(%v181 init: %v174) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v183 = stablehlo.broadcast_in_dim %v182, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v184 = stablehlo.divide %v183, %v175 : tensor<32x64x56x56xf32>
    %v185 = stablehlo.add %v184, %v176 : tensor<32x64x56x56xf32>
    %v186 = stablehlo.rsqrt %v185 : tensor<32x64x56x56xf32>
    %v187 = stablehlo.multiply %v180, %v186 : tensor<32x64x56x56xf32>
    %v188 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v189 = stablehlo.broadcast_in_dim %s1b2bt2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v190 = stablehlo.multiply %v187, %v188 : tensor<32x64x56x56xf32>
    %v191 = stablehlo.add %v190, %v189 : tensor<32x64x56x56xf32>
    %v192 = stablehlo.reshape %v191 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v193 = stablehlo.add %v192, %v140 : tensor<32x200704xf32>
    %v194 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v195 = stablehlo.maximum %v193, %v194 : tensor<32x200704xf32>
    %v196 = stablehlo.reshape %v195 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v197 = stablehlo.convolution(%v196, %d2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<128x64x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v198 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v199 = stablehlo.add %v197, %v198 : tensor<32x128x28x28xf32>
    %v200 = stablehlo.reshape %v199 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v201 = stablehlo.reshape %v200 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v202 = stablehlo.constant dense<0.0> : tensor<f32>
    %v203 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v204 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v205 = stablehlo.reduce(%v201 init: %v202) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v206 = stablehlo.broadcast_in_dim %v205, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v207 = stablehlo.divide %v206, %v203 : tensor<32x128x28x28xf32>
    %v208 = stablehlo.subtract %v201, %v207 : tensor<32x128x28x28xf32>
    %v209 = stablehlo.multiply %v208, %v208 : tensor<32x128x28x28xf32>
    %v210 = stablehlo.reduce(%v209 init: %v202) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v211 = stablehlo.broadcast_in_dim %v210, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v212 = stablehlo.divide %v211, %v203 : tensor<32x128x28x28xf32>
    %v213 = stablehlo.add %v212, %v204 : tensor<32x128x28x28xf32>
    %v214 = stablehlo.rsqrt %v213 : tensor<32x128x28x28xf32>
    %v215 = stablehlo.multiply %v208, %v214 : tensor<32x128x28x28xf32>
    %v216 = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v217 = stablehlo.broadcast_in_dim %d2bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v218 = stablehlo.multiply %v215, %v216 : tensor<32x128x28x28xf32>
    %v219 = stablehlo.add %v218, %v217 : tensor<32x128x28x28xf32>
    %v220 = stablehlo.reshape %v219 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v221 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v222 = stablehlo.maximum %v220, %v221 : tensor<32x100352xf32>
    %v223 = stablehlo.reshape %v222 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v224 = stablehlo.convolution(%v223, %d2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v225 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v226 = stablehlo.add %v224, %v225 : tensor<32x128x28x28xf32>
    %v227 = stablehlo.reshape %v226 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v228 = stablehlo.reshape %v227 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v229 = stablehlo.constant dense<0.0> : tensor<f32>
    %v230 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v231 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v232 = stablehlo.reduce(%v228 init: %v229) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v233 = stablehlo.broadcast_in_dim %v232, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v234 = stablehlo.divide %v233, %v230 : tensor<32x128x28x28xf32>
    %v235 = stablehlo.subtract %v228, %v234 : tensor<32x128x28x28xf32>
    %v236 = stablehlo.multiply %v235, %v235 : tensor<32x128x28x28xf32>
    %v237 = stablehlo.reduce(%v236 init: %v229) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v238 = stablehlo.broadcast_in_dim %v237, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v239 = stablehlo.divide %v238, %v230 : tensor<32x128x28x28xf32>
    %v240 = stablehlo.add %v239, %v231 : tensor<32x128x28x28xf32>
    %v241 = stablehlo.rsqrt %v240 : tensor<32x128x28x28xf32>
    %v242 = stablehlo.multiply %v235, %v241 : tensor<32x128x28x28xf32>
    %v243 = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v244 = stablehlo.broadcast_in_dim %d2bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v245 = stablehlo.multiply %v242, %v243 : tensor<32x128x28x28xf32>
    %v246 = stablehlo.add %v245, %v244 : tensor<32x128x28x28xf32>
    %v247 = stablehlo.reshape %v246 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v248 = stablehlo.reshape %v195 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v249 = stablehlo.convolution(%v248, %d2Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<128x64x3x3xf32>) -> tensor<32x128x28x28xf32>
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
    %v268 = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v269 = stablehlo.broadcast_in_dim %d2btp, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v270 = stablehlo.multiply %v267, %v268 : tensor<32x128x28x28xf32>
    %v271 = stablehlo.add %v270, %v269 : tensor<32x128x28x28xf32>
    %v272 = stablehlo.reshape %v271 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v273 = stablehlo.add %v247, %v272 : tensor<32x100352xf32>
    %v274 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v275 = stablehlo.maximum %v273, %v274 : tensor<32x100352xf32>
    %v276 = stablehlo.reshape %v275 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v277 = stablehlo.convolution(%v276, %s2b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v278 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v279 = stablehlo.add %v277, %v278 : tensor<32x128x28x28xf32>
    %v280 = stablehlo.reshape %v279 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v281 = stablehlo.reshape %v280 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v282 = stablehlo.constant dense<0.0> : tensor<f32>
    %v283 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v284 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v285 = stablehlo.reduce(%v281 init: %v282) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v286 = stablehlo.broadcast_in_dim %v285, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v287 = stablehlo.divide %v286, %v283 : tensor<32x128x28x28xf32>
    %v288 = stablehlo.subtract %v281, %v287 : tensor<32x128x28x28xf32>
    %v289 = stablehlo.multiply %v288, %v288 : tensor<32x128x28x28xf32>
    %v290 = stablehlo.reduce(%v289 init: %v282) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v291 = stablehlo.broadcast_in_dim %v290, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v292 = stablehlo.divide %v291, %v283 : tensor<32x128x28x28xf32>
    %v293 = stablehlo.add %v292, %v284 : tensor<32x128x28x28xf32>
    %v294 = stablehlo.rsqrt %v293 : tensor<32x128x28x28xf32>
    %v295 = stablehlo.multiply %v288, %v294 : tensor<32x128x28x28xf32>
    %v296 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v297 = stablehlo.broadcast_in_dim %s2b0bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v298 = stablehlo.multiply %v295, %v296 : tensor<32x128x28x28xf32>
    %v299 = stablehlo.add %v298, %v297 : tensor<32x128x28x28xf32>
    %v300 = stablehlo.reshape %v299 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v301 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v302 = stablehlo.maximum %v300, %v301 : tensor<32x100352xf32>
    %v303 = stablehlo.reshape %v302 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v304 = stablehlo.convolution(%v303, %s2b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v305 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v306 = stablehlo.add %v304, %v305 : tensor<32x128x28x28xf32>
    %v307 = stablehlo.reshape %v306 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v308 = stablehlo.reshape %v307 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v309 = stablehlo.constant dense<0.0> : tensor<f32>
    %v310 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v311 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v312 = stablehlo.reduce(%v308 init: %v309) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v313 = stablehlo.broadcast_in_dim %v312, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v314 = stablehlo.divide %v313, %v310 : tensor<32x128x28x28xf32>
    %v315 = stablehlo.subtract %v308, %v314 : tensor<32x128x28x28xf32>
    %v316 = stablehlo.multiply %v315, %v315 : tensor<32x128x28x28xf32>
    %v317 = stablehlo.reduce(%v316 init: %v309) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v318 = stablehlo.broadcast_in_dim %v317, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v319 = stablehlo.divide %v318, %v310 : tensor<32x128x28x28xf32>
    %v320 = stablehlo.add %v319, %v311 : tensor<32x128x28x28xf32>
    %v321 = stablehlo.rsqrt %v320 : tensor<32x128x28x28xf32>
    %v322 = stablehlo.multiply %v315, %v321 : tensor<32x128x28x28xf32>
    %v323 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v324 = stablehlo.broadcast_in_dim %s2b0bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v325 = stablehlo.multiply %v322, %v323 : tensor<32x128x28x28xf32>
    %v326 = stablehlo.add %v325, %v324 : tensor<32x128x28x28xf32>
    %v327 = stablehlo.reshape %v326 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v328 = stablehlo.add %v327, %v275 : tensor<32x100352xf32>
    %v329 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v330 = stablehlo.maximum %v328, %v329 : tensor<32x100352xf32>
    %v331 = stablehlo.reshape %v330 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v332 = stablehlo.convolution(%v331, %s2b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v333 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v334 = stablehlo.add %v332, %v333 : tensor<32x128x28x28xf32>
    %v335 = stablehlo.reshape %v334 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v336 = stablehlo.reshape %v335 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v337 = stablehlo.constant dense<0.0> : tensor<f32>
    %v338 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v339 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v340 = stablehlo.reduce(%v336 init: %v337) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v341 = stablehlo.broadcast_in_dim %v340, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v342 = stablehlo.divide %v341, %v338 : tensor<32x128x28x28xf32>
    %v343 = stablehlo.subtract %v336, %v342 : tensor<32x128x28x28xf32>
    %v344 = stablehlo.multiply %v343, %v343 : tensor<32x128x28x28xf32>
    %v345 = stablehlo.reduce(%v344 init: %v337) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v346 = stablehlo.broadcast_in_dim %v345, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v347 = stablehlo.divide %v346, %v338 : tensor<32x128x28x28xf32>
    %v348 = stablehlo.add %v347, %v339 : tensor<32x128x28x28xf32>
    %v349 = stablehlo.rsqrt %v348 : tensor<32x128x28x28xf32>
    %v350 = stablehlo.multiply %v343, %v349 : tensor<32x128x28x28xf32>
    %v351 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v352 = stablehlo.broadcast_in_dim %s2b1bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v353 = stablehlo.multiply %v350, %v351 : tensor<32x128x28x28xf32>
    %v354 = stablehlo.add %v353, %v352 : tensor<32x128x28x28xf32>
    %v355 = stablehlo.reshape %v354 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v356 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v357 = stablehlo.maximum %v355, %v356 : tensor<32x100352xf32>
    %v358 = stablehlo.reshape %v357 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v359 = stablehlo.convolution(%v358, %s2b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v360 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v361 = stablehlo.add %v359, %v360 : tensor<32x128x28x28xf32>
    %v362 = stablehlo.reshape %v361 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v363 = stablehlo.reshape %v362 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v364 = stablehlo.constant dense<0.0> : tensor<f32>
    %v365 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v366 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v367 = stablehlo.reduce(%v363 init: %v364) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v368 = stablehlo.broadcast_in_dim %v367, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v369 = stablehlo.divide %v368, %v365 : tensor<32x128x28x28xf32>
    %v370 = stablehlo.subtract %v363, %v369 : tensor<32x128x28x28xf32>
    %v371 = stablehlo.multiply %v370, %v370 : tensor<32x128x28x28xf32>
    %v372 = stablehlo.reduce(%v371 init: %v364) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v373 = stablehlo.broadcast_in_dim %v372, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v374 = stablehlo.divide %v373, %v365 : tensor<32x128x28x28xf32>
    %v375 = stablehlo.add %v374, %v366 : tensor<32x128x28x28xf32>
    %v376 = stablehlo.rsqrt %v375 : tensor<32x128x28x28xf32>
    %v377 = stablehlo.multiply %v370, %v376 : tensor<32x128x28x28xf32>
    %v378 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v379 = stablehlo.broadcast_in_dim %s2b1bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v380 = stablehlo.multiply %v377, %v378 : tensor<32x128x28x28xf32>
    %v381 = stablehlo.add %v380, %v379 : tensor<32x128x28x28xf32>
    %v382 = stablehlo.reshape %v381 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v383 = stablehlo.add %v382, %v330 : tensor<32x100352xf32>
    %v384 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v385 = stablehlo.maximum %v383, %v384 : tensor<32x100352xf32>
    %v386 = stablehlo.reshape %v385 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v387 = stablehlo.convolution(%v386, %s2b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v388 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v389 = stablehlo.add %v387, %v388 : tensor<32x128x28x28xf32>
    %v390 = stablehlo.reshape %v389 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v391 = stablehlo.reshape %v390 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v392 = stablehlo.constant dense<0.0> : tensor<f32>
    %v393 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v394 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v395 = stablehlo.reduce(%v391 init: %v392) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v396 = stablehlo.broadcast_in_dim %v395, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v397 = stablehlo.divide %v396, %v393 : tensor<32x128x28x28xf32>
    %v398 = stablehlo.subtract %v391, %v397 : tensor<32x128x28x28xf32>
    %v399 = stablehlo.multiply %v398, %v398 : tensor<32x128x28x28xf32>
    %v400 = stablehlo.reduce(%v399 init: %v392) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v401 = stablehlo.broadcast_in_dim %v400, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v402 = stablehlo.divide %v401, %v393 : tensor<32x128x28x28xf32>
    %v403 = stablehlo.add %v402, %v394 : tensor<32x128x28x28xf32>
    %v404 = stablehlo.rsqrt %v403 : tensor<32x128x28x28xf32>
    %v405 = stablehlo.multiply %v398, %v404 : tensor<32x128x28x28xf32>
    %v406 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v407 = stablehlo.broadcast_in_dim %s2b2bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v408 = stablehlo.multiply %v405, %v406 : tensor<32x128x28x28xf32>
    %v409 = stablehlo.add %v408, %v407 : tensor<32x128x28x28xf32>
    %v410 = stablehlo.reshape %v409 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v411 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v412 = stablehlo.maximum %v410, %v411 : tensor<32x100352xf32>
    %v413 = stablehlo.reshape %v412 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v414 = stablehlo.convolution(%v413, %s2b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v415 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v416 = stablehlo.add %v414, %v415 : tensor<32x128x28x28xf32>
    %v417 = stablehlo.reshape %v416 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v418 = stablehlo.reshape %v417 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v419 = stablehlo.constant dense<0.0> : tensor<f32>
    %v420 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v421 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v422 = stablehlo.reduce(%v418 init: %v419) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v423 = stablehlo.broadcast_in_dim %v422, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v424 = stablehlo.divide %v423, %v420 : tensor<32x128x28x28xf32>
    %v425 = stablehlo.subtract %v418, %v424 : tensor<32x128x28x28xf32>
    %v426 = stablehlo.multiply %v425, %v425 : tensor<32x128x28x28xf32>
    %v427 = stablehlo.reduce(%v426 init: %v419) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v428 = stablehlo.broadcast_in_dim %v427, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v429 = stablehlo.divide %v428, %v420 : tensor<32x128x28x28xf32>
    %v430 = stablehlo.add %v429, %v421 : tensor<32x128x28x28xf32>
    %v431 = stablehlo.rsqrt %v430 : tensor<32x128x28x28xf32>
    %v432 = stablehlo.multiply %v425, %v431 : tensor<32x128x28x28xf32>
    %v433 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v434 = stablehlo.broadcast_in_dim %s2b2bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v435 = stablehlo.multiply %v432, %v433 : tensor<32x128x28x28xf32>
    %v436 = stablehlo.add %v435, %v434 : tensor<32x128x28x28xf32>
    %v437 = stablehlo.reshape %v436 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v438 = stablehlo.add %v437, %v385 : tensor<32x100352xf32>
    %v439 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v440 = stablehlo.maximum %v438, %v439 : tensor<32x100352xf32>
    %v441 = stablehlo.reshape %v440 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v442 = stablehlo.convolution(%v441, %d3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<256x128x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v443 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v444 = stablehlo.add %v442, %v443 : tensor<32x256x14x14xf32>
    %v445 = stablehlo.reshape %v444 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v446 = stablehlo.reshape %v445 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v447 = stablehlo.constant dense<0.0> : tensor<f32>
    %v448 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v449 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v450 = stablehlo.reduce(%v446 init: %v447) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v451 = stablehlo.broadcast_in_dim %v450, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v452 = stablehlo.divide %v451, %v448 : tensor<32x256x14x14xf32>
    %v453 = stablehlo.subtract %v446, %v452 : tensor<32x256x14x14xf32>
    %v454 = stablehlo.multiply %v453, %v453 : tensor<32x256x14x14xf32>
    %v455 = stablehlo.reduce(%v454 init: %v447) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v456 = stablehlo.broadcast_in_dim %v455, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v457 = stablehlo.divide %v456, %v448 : tensor<32x256x14x14xf32>
    %v458 = stablehlo.add %v457, %v449 : tensor<32x256x14x14xf32>
    %v459 = stablehlo.rsqrt %v458 : tensor<32x256x14x14xf32>
    %v460 = stablehlo.multiply %v453, %v459 : tensor<32x256x14x14xf32>
    %v461 = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v462 = stablehlo.broadcast_in_dim %d3bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v463 = stablehlo.multiply %v460, %v461 : tensor<32x256x14x14xf32>
    %v464 = stablehlo.add %v463, %v462 : tensor<32x256x14x14xf32>
    %v465 = stablehlo.reshape %v464 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v466 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v467 = stablehlo.maximum %v465, %v466 : tensor<32x50176xf32>
    %v468 = stablehlo.reshape %v467 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v469 = stablehlo.convolution(%v468, %d3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v470 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v471 = stablehlo.add %v469, %v470 : tensor<32x256x14x14xf32>
    %v472 = stablehlo.reshape %v471 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v473 = stablehlo.reshape %v472 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v474 = stablehlo.constant dense<0.0> : tensor<f32>
    %v475 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v476 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v477 = stablehlo.reduce(%v473 init: %v474) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v478 = stablehlo.broadcast_in_dim %v477, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v479 = stablehlo.divide %v478, %v475 : tensor<32x256x14x14xf32>
    %v480 = stablehlo.subtract %v473, %v479 : tensor<32x256x14x14xf32>
    %v481 = stablehlo.multiply %v480, %v480 : tensor<32x256x14x14xf32>
    %v482 = stablehlo.reduce(%v481 init: %v474) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v483 = stablehlo.broadcast_in_dim %v482, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v484 = stablehlo.divide %v483, %v475 : tensor<32x256x14x14xf32>
    %v485 = stablehlo.add %v484, %v476 : tensor<32x256x14x14xf32>
    %v486 = stablehlo.rsqrt %v485 : tensor<32x256x14x14xf32>
    %v487 = stablehlo.multiply %v480, %v486 : tensor<32x256x14x14xf32>
    %v488 = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v489 = stablehlo.broadcast_in_dim %d3bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v490 = stablehlo.multiply %v487, %v488 : tensor<32x256x14x14xf32>
    %v491 = stablehlo.add %v490, %v489 : tensor<32x256x14x14xf32>
    %v492 = stablehlo.reshape %v491 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v493 = stablehlo.reshape %v440 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v494 = stablehlo.convolution(%v493, %d3Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<256x128x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v495 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v496 = stablehlo.add %v494, %v495 : tensor<32x256x14x14xf32>
    %v497 = stablehlo.reshape %v496 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v498 = stablehlo.reshape %v497 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v499 = stablehlo.constant dense<0.0> : tensor<f32>
    %v500 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v501 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v502 = stablehlo.reduce(%v498 init: %v499) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v503 = stablehlo.broadcast_in_dim %v502, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v504 = stablehlo.divide %v503, %v500 : tensor<32x256x14x14xf32>
    %v505 = stablehlo.subtract %v498, %v504 : tensor<32x256x14x14xf32>
    %v506 = stablehlo.multiply %v505, %v505 : tensor<32x256x14x14xf32>
    %v507 = stablehlo.reduce(%v506 init: %v499) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v508 = stablehlo.broadcast_in_dim %v507, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v509 = stablehlo.divide %v508, %v500 : tensor<32x256x14x14xf32>
    %v510 = stablehlo.add %v509, %v501 : tensor<32x256x14x14xf32>
    %v511 = stablehlo.rsqrt %v510 : tensor<32x256x14x14xf32>
    %v512 = stablehlo.multiply %v505, %v511 : tensor<32x256x14x14xf32>
    %v513 = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v514 = stablehlo.broadcast_in_dim %d3btp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v515 = stablehlo.multiply %v512, %v513 : tensor<32x256x14x14xf32>
    %v516 = stablehlo.add %v515, %v514 : tensor<32x256x14x14xf32>
    %v517 = stablehlo.reshape %v516 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v518 = stablehlo.add %v492, %v517 : tensor<32x50176xf32>
    %v519 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v520 = stablehlo.maximum %v518, %v519 : tensor<32x50176xf32>
    %v521 = stablehlo.reshape %v520 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v522 = stablehlo.convolution(%v521, %s3b0W1)
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
    %v541 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v542 = stablehlo.broadcast_in_dim %s3b0bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v543 = stablehlo.multiply %v540, %v541 : tensor<32x256x14x14xf32>
    %v544 = stablehlo.add %v543, %v542 : tensor<32x256x14x14xf32>
    %v545 = stablehlo.reshape %v544 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v546 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v547 = stablehlo.maximum %v545, %v546 : tensor<32x50176xf32>
    %v548 = stablehlo.reshape %v547 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v549 = stablehlo.convolution(%v548, %s3b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v550 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v551 = stablehlo.add %v549, %v550 : tensor<32x256x14x14xf32>
    %v552 = stablehlo.reshape %v551 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v553 = stablehlo.reshape %v552 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v554 = stablehlo.constant dense<0.0> : tensor<f32>
    %v555 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v556 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v557 = stablehlo.reduce(%v553 init: %v554) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v558 = stablehlo.broadcast_in_dim %v557, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v559 = stablehlo.divide %v558, %v555 : tensor<32x256x14x14xf32>
    %v560 = stablehlo.subtract %v553, %v559 : tensor<32x256x14x14xf32>
    %v561 = stablehlo.multiply %v560, %v560 : tensor<32x256x14x14xf32>
    %v562 = stablehlo.reduce(%v561 init: %v554) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v563 = stablehlo.broadcast_in_dim %v562, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v564 = stablehlo.divide %v563, %v555 : tensor<32x256x14x14xf32>
    %v565 = stablehlo.add %v564, %v556 : tensor<32x256x14x14xf32>
    %v566 = stablehlo.rsqrt %v565 : tensor<32x256x14x14xf32>
    %v567 = stablehlo.multiply %v560, %v566 : tensor<32x256x14x14xf32>
    %v568 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v569 = stablehlo.broadcast_in_dim %s3b0bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v570 = stablehlo.multiply %v567, %v568 : tensor<32x256x14x14xf32>
    %v571 = stablehlo.add %v570, %v569 : tensor<32x256x14x14xf32>
    %v572 = stablehlo.reshape %v571 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v573 = stablehlo.add %v572, %v520 : tensor<32x50176xf32>
    %v574 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v575 = stablehlo.maximum %v573, %v574 : tensor<32x50176xf32>
    %v576 = stablehlo.reshape %v575 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v577 = stablehlo.convolution(%v576, %s3b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v578 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v579 = stablehlo.add %v577, %v578 : tensor<32x256x14x14xf32>
    %v580 = stablehlo.reshape %v579 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v581 = stablehlo.reshape %v580 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v582 = stablehlo.constant dense<0.0> : tensor<f32>
    %v583 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v584 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v585 = stablehlo.reduce(%v581 init: %v582) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v586 = stablehlo.broadcast_in_dim %v585, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v587 = stablehlo.divide %v586, %v583 : tensor<32x256x14x14xf32>
    %v588 = stablehlo.subtract %v581, %v587 : tensor<32x256x14x14xf32>
    %v589 = stablehlo.multiply %v588, %v588 : tensor<32x256x14x14xf32>
    %v590 = stablehlo.reduce(%v589 init: %v582) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v591 = stablehlo.broadcast_in_dim %v590, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v592 = stablehlo.divide %v591, %v583 : tensor<32x256x14x14xf32>
    %v593 = stablehlo.add %v592, %v584 : tensor<32x256x14x14xf32>
    %v594 = stablehlo.rsqrt %v593 : tensor<32x256x14x14xf32>
    %v595 = stablehlo.multiply %v588, %v594 : tensor<32x256x14x14xf32>
    %v596 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v597 = stablehlo.broadcast_in_dim %s3b1bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v598 = stablehlo.multiply %v595, %v596 : tensor<32x256x14x14xf32>
    %v599 = stablehlo.add %v598, %v597 : tensor<32x256x14x14xf32>
    %v600 = stablehlo.reshape %v599 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v601 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v602 = stablehlo.maximum %v600, %v601 : tensor<32x50176xf32>
    %v603 = stablehlo.reshape %v602 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v604 = stablehlo.convolution(%v603, %s3b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v605 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v606 = stablehlo.add %v604, %v605 : tensor<32x256x14x14xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v608 = stablehlo.reshape %v607 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v609 = stablehlo.constant dense<0.0> : tensor<f32>
    %v610 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v611 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v612 = stablehlo.reduce(%v608 init: %v609) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v613 = stablehlo.broadcast_in_dim %v612, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v614 = stablehlo.divide %v613, %v610 : tensor<32x256x14x14xf32>
    %v615 = stablehlo.subtract %v608, %v614 : tensor<32x256x14x14xf32>
    %v616 = stablehlo.multiply %v615, %v615 : tensor<32x256x14x14xf32>
    %v617 = stablehlo.reduce(%v616 init: %v609) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v618 = stablehlo.broadcast_in_dim %v617, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v619 = stablehlo.divide %v618, %v610 : tensor<32x256x14x14xf32>
    %v620 = stablehlo.add %v619, %v611 : tensor<32x256x14x14xf32>
    %v621 = stablehlo.rsqrt %v620 : tensor<32x256x14x14xf32>
    %v622 = stablehlo.multiply %v615, %v621 : tensor<32x256x14x14xf32>
    %v623 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v624 = stablehlo.broadcast_in_dim %s3b1bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v625 = stablehlo.multiply %v622, %v623 : tensor<32x256x14x14xf32>
    %v626 = stablehlo.add %v625, %v624 : tensor<32x256x14x14xf32>
    %v627 = stablehlo.reshape %v626 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v628 = stablehlo.add %v627, %v575 : tensor<32x50176xf32>
    %v629 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v630 = stablehlo.maximum %v628, %v629 : tensor<32x50176xf32>
    %v631 = stablehlo.reshape %v630 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v632 = stablehlo.convolution(%v631, %s3b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v633 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v634 = stablehlo.add %v632, %v633 : tensor<32x256x14x14xf32>
    %v635 = stablehlo.reshape %v634 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v636 = stablehlo.reshape %v635 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v637 = stablehlo.constant dense<0.0> : tensor<f32>
    %v638 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v639 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v640 = stablehlo.reduce(%v636 init: %v637) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v641 = stablehlo.broadcast_in_dim %v640, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v642 = stablehlo.divide %v641, %v638 : tensor<32x256x14x14xf32>
    %v643 = stablehlo.subtract %v636, %v642 : tensor<32x256x14x14xf32>
    %v644 = stablehlo.multiply %v643, %v643 : tensor<32x256x14x14xf32>
    %v645 = stablehlo.reduce(%v644 init: %v637) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v646 = stablehlo.broadcast_in_dim %v645, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v647 = stablehlo.divide %v646, %v638 : tensor<32x256x14x14xf32>
    %v648 = stablehlo.add %v647, %v639 : tensor<32x256x14x14xf32>
    %v649 = stablehlo.rsqrt %v648 : tensor<32x256x14x14xf32>
    %v650 = stablehlo.multiply %v643, %v649 : tensor<32x256x14x14xf32>
    %v651 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v652 = stablehlo.broadcast_in_dim %s3b2bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v653 = stablehlo.multiply %v650, %v651 : tensor<32x256x14x14xf32>
    %v654 = stablehlo.add %v653, %v652 : tensor<32x256x14x14xf32>
    %v655 = stablehlo.reshape %v654 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v656 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v657 = stablehlo.maximum %v655, %v656 : tensor<32x50176xf32>
    %v658 = stablehlo.reshape %v657 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v659 = stablehlo.convolution(%v658, %s3b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v660 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v661 = stablehlo.add %v659, %v660 : tensor<32x256x14x14xf32>
    %v662 = stablehlo.reshape %v661 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v663 = stablehlo.reshape %v662 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v664 = stablehlo.constant dense<0.0> : tensor<f32>
    %v665 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v666 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v667 = stablehlo.reduce(%v663 init: %v664) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v668 = stablehlo.broadcast_in_dim %v667, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v669 = stablehlo.divide %v668, %v665 : tensor<32x256x14x14xf32>
    %v670 = stablehlo.subtract %v663, %v669 : tensor<32x256x14x14xf32>
    %v671 = stablehlo.multiply %v670, %v670 : tensor<32x256x14x14xf32>
    %v672 = stablehlo.reduce(%v671 init: %v664) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v673 = stablehlo.broadcast_in_dim %v672, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v674 = stablehlo.divide %v673, %v665 : tensor<32x256x14x14xf32>
    %v675 = stablehlo.add %v674, %v666 : tensor<32x256x14x14xf32>
    %v676 = stablehlo.rsqrt %v675 : tensor<32x256x14x14xf32>
    %v677 = stablehlo.multiply %v670, %v676 : tensor<32x256x14x14xf32>
    %v678 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v679 = stablehlo.broadcast_in_dim %s3b2bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v680 = stablehlo.multiply %v677, %v678 : tensor<32x256x14x14xf32>
    %v681 = stablehlo.add %v680, %v679 : tensor<32x256x14x14xf32>
    %v682 = stablehlo.reshape %v681 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v683 = stablehlo.add %v682, %v630 : tensor<32x50176xf32>
    %v684 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v685 = stablehlo.maximum %v683, %v684 : tensor<32x50176xf32>
    %v686 = stablehlo.reshape %v685 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v687 = stablehlo.convolution(%v686, %s3b3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v688 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v689 = stablehlo.add %v687, %v688 : tensor<32x256x14x14xf32>
    %v690 = stablehlo.reshape %v689 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v691 = stablehlo.reshape %v690 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v692 = stablehlo.constant dense<0.0> : tensor<f32>
    %v693 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v694 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v695 = stablehlo.reduce(%v691 init: %v692) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v696 = stablehlo.broadcast_in_dim %v695, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v697 = stablehlo.divide %v696, %v693 : tensor<32x256x14x14xf32>
    %v698 = stablehlo.subtract %v691, %v697 : tensor<32x256x14x14xf32>
    %v699 = stablehlo.multiply %v698, %v698 : tensor<32x256x14x14xf32>
    %v700 = stablehlo.reduce(%v699 init: %v692) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v701 = stablehlo.broadcast_in_dim %v700, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v702 = stablehlo.divide %v701, %v693 : tensor<32x256x14x14xf32>
    %v703 = stablehlo.add %v702, %v694 : tensor<32x256x14x14xf32>
    %v704 = stablehlo.rsqrt %v703 : tensor<32x256x14x14xf32>
    %v705 = stablehlo.multiply %v698, %v704 : tensor<32x256x14x14xf32>
    %v706 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v707 = stablehlo.broadcast_in_dim %s3b3bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v708 = stablehlo.multiply %v705, %v706 : tensor<32x256x14x14xf32>
    %v709 = stablehlo.add %v708, %v707 : tensor<32x256x14x14xf32>
    %v710 = stablehlo.reshape %v709 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v711 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v712 = stablehlo.maximum %v710, %v711 : tensor<32x50176xf32>
    %v713 = stablehlo.reshape %v712 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v714 = stablehlo.convolution(%v713, %s3b3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v715 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v716 = stablehlo.add %v714, %v715 : tensor<32x256x14x14xf32>
    %v717 = stablehlo.reshape %v716 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v718 = stablehlo.reshape %v717 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v719 = stablehlo.constant dense<0.0> : tensor<f32>
    %v720 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v721 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v722 = stablehlo.reduce(%v718 init: %v719) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v723 = stablehlo.broadcast_in_dim %v722, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v724 = stablehlo.divide %v723, %v720 : tensor<32x256x14x14xf32>
    %v725 = stablehlo.subtract %v718, %v724 : tensor<32x256x14x14xf32>
    %v726 = stablehlo.multiply %v725, %v725 : tensor<32x256x14x14xf32>
    %v727 = stablehlo.reduce(%v726 init: %v719) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v728 = stablehlo.broadcast_in_dim %v727, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v729 = stablehlo.divide %v728, %v720 : tensor<32x256x14x14xf32>
    %v730 = stablehlo.add %v729, %v721 : tensor<32x256x14x14xf32>
    %v731 = stablehlo.rsqrt %v730 : tensor<32x256x14x14xf32>
    %v732 = stablehlo.multiply %v725, %v731 : tensor<32x256x14x14xf32>
    %v733 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v734 = stablehlo.broadcast_in_dim %s3b3bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v735 = stablehlo.multiply %v732, %v733 : tensor<32x256x14x14xf32>
    %v736 = stablehlo.add %v735, %v734 : tensor<32x256x14x14xf32>
    %v737 = stablehlo.reshape %v736 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v738 = stablehlo.add %v737, %v685 : tensor<32x50176xf32>
    %v739 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v740 = stablehlo.maximum %v738, %v739 : tensor<32x50176xf32>
    %v741 = stablehlo.reshape %v740 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v742 = stablehlo.convolution(%v741, %s3b4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v743 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v744 = stablehlo.add %v742, %v743 : tensor<32x256x14x14xf32>
    %v745 = stablehlo.reshape %v744 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v746 = stablehlo.reshape %v745 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v747 = stablehlo.constant dense<0.0> : tensor<f32>
    %v748 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v749 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v750 = stablehlo.reduce(%v746 init: %v747) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v751 = stablehlo.broadcast_in_dim %v750, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v752 = stablehlo.divide %v751, %v748 : tensor<32x256x14x14xf32>
    %v753 = stablehlo.subtract %v746, %v752 : tensor<32x256x14x14xf32>
    %v754 = stablehlo.multiply %v753, %v753 : tensor<32x256x14x14xf32>
    %v755 = stablehlo.reduce(%v754 init: %v747) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v756 = stablehlo.broadcast_in_dim %v755, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v757 = stablehlo.divide %v756, %v748 : tensor<32x256x14x14xf32>
    %v758 = stablehlo.add %v757, %v749 : tensor<32x256x14x14xf32>
    %v759 = stablehlo.rsqrt %v758 : tensor<32x256x14x14xf32>
    %v760 = stablehlo.multiply %v753, %v759 : tensor<32x256x14x14xf32>
    %v761 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v762 = stablehlo.broadcast_in_dim %s3b4bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v763 = stablehlo.multiply %v760, %v761 : tensor<32x256x14x14xf32>
    %v764 = stablehlo.add %v763, %v762 : tensor<32x256x14x14xf32>
    %v765 = stablehlo.reshape %v764 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v766 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v767 = stablehlo.maximum %v765, %v766 : tensor<32x50176xf32>
    %v768 = stablehlo.reshape %v767 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v769 = stablehlo.convolution(%v768, %s3b4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v770 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v771 = stablehlo.add %v769, %v770 : tensor<32x256x14x14xf32>
    %v772 = stablehlo.reshape %v771 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v773 = stablehlo.reshape %v772 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v774 = stablehlo.constant dense<0.0> : tensor<f32>
    %v775 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v776 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v777 = stablehlo.reduce(%v773 init: %v774) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v778 = stablehlo.broadcast_in_dim %v777, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v779 = stablehlo.divide %v778, %v775 : tensor<32x256x14x14xf32>
    %v780 = stablehlo.subtract %v773, %v779 : tensor<32x256x14x14xf32>
    %v781 = stablehlo.multiply %v780, %v780 : tensor<32x256x14x14xf32>
    %v782 = stablehlo.reduce(%v781 init: %v774) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v783 = stablehlo.broadcast_in_dim %v782, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v784 = stablehlo.divide %v783, %v775 : tensor<32x256x14x14xf32>
    %v785 = stablehlo.add %v784, %v776 : tensor<32x256x14x14xf32>
    %v786 = stablehlo.rsqrt %v785 : tensor<32x256x14x14xf32>
    %v787 = stablehlo.multiply %v780, %v786 : tensor<32x256x14x14xf32>
    %v788 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v789 = stablehlo.broadcast_in_dim %s3b4bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v790 = stablehlo.multiply %v787, %v788 : tensor<32x256x14x14xf32>
    %v791 = stablehlo.add %v790, %v789 : tensor<32x256x14x14xf32>
    %v792 = stablehlo.reshape %v791 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v793 = stablehlo.add %v792, %v740 : tensor<32x50176xf32>
    %v794 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v795 = stablehlo.maximum %v793, %v794 : tensor<32x50176xf32>
    %v796 = stablehlo.reshape %v795 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v797 = stablehlo.convolution(%v796, %d4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<512x256x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v798 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v799 = stablehlo.add %v797, %v798 : tensor<32x512x7x7xf32>
    %v800 = stablehlo.reshape %v799 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v801 = stablehlo.reshape %v800 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v802 = stablehlo.constant dense<0.0> : tensor<f32>
    %v803 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v804 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v805 = stablehlo.reduce(%v801 init: %v802) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v806 = stablehlo.broadcast_in_dim %v805, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v807 = stablehlo.divide %v806, %v803 : tensor<32x512x7x7xf32>
    %v808 = stablehlo.subtract %v801, %v807 : tensor<32x512x7x7xf32>
    %v809 = stablehlo.multiply %v808, %v808 : tensor<32x512x7x7xf32>
    %v810 = stablehlo.reduce(%v809 init: %v802) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v811 = stablehlo.broadcast_in_dim %v810, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v812 = stablehlo.divide %v811, %v803 : tensor<32x512x7x7xf32>
    %v813 = stablehlo.add %v812, %v804 : tensor<32x512x7x7xf32>
    %v814 = stablehlo.rsqrt %v813 : tensor<32x512x7x7xf32>
    %v815 = stablehlo.multiply %v808, %v814 : tensor<32x512x7x7xf32>
    %v816 = stablehlo.broadcast_in_dim %d4g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v817 = stablehlo.broadcast_in_dim %d4bt1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v818 = stablehlo.multiply %v815, %v816 : tensor<32x512x7x7xf32>
    %v819 = stablehlo.add %v818, %v817 : tensor<32x512x7x7xf32>
    %v820 = stablehlo.reshape %v819 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v821 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v822 = stablehlo.maximum %v820, %v821 : tensor<32x25088xf32>
    %v823 = stablehlo.reshape %v822 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v824 = stablehlo.convolution(%v823, %d4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v825 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v826 = stablehlo.add %v824, %v825 : tensor<32x512x7x7xf32>
    %v827 = stablehlo.reshape %v826 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v828 = stablehlo.reshape %v827 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v829 = stablehlo.constant dense<0.0> : tensor<f32>
    %v830 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v831 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v832 = stablehlo.reduce(%v828 init: %v829) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v833 = stablehlo.broadcast_in_dim %v832, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v834 = stablehlo.divide %v833, %v830 : tensor<32x512x7x7xf32>
    %v835 = stablehlo.subtract %v828, %v834 : tensor<32x512x7x7xf32>
    %v836 = stablehlo.multiply %v835, %v835 : tensor<32x512x7x7xf32>
    %v837 = stablehlo.reduce(%v836 init: %v829) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v838 = stablehlo.broadcast_in_dim %v837, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v839 = stablehlo.divide %v838, %v830 : tensor<32x512x7x7xf32>
    %v840 = stablehlo.add %v839, %v831 : tensor<32x512x7x7xf32>
    %v841 = stablehlo.rsqrt %v840 : tensor<32x512x7x7xf32>
    %v842 = stablehlo.multiply %v835, %v841 : tensor<32x512x7x7xf32>
    %v843 = stablehlo.broadcast_in_dim %d4g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v844 = stablehlo.broadcast_in_dim %d4bt2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v845 = stablehlo.multiply %v842, %v843 : tensor<32x512x7x7xf32>
    %v846 = stablehlo.add %v845, %v844 : tensor<32x512x7x7xf32>
    %v847 = stablehlo.reshape %v846 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v848 = stablehlo.reshape %v795 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v849 = stablehlo.convolution(%v848, %d4Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<512x256x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v850 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v851 = stablehlo.add %v849, %v850 : tensor<32x512x7x7xf32>
    %v852 = stablehlo.reshape %v851 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v853 = stablehlo.reshape %v852 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v854 = stablehlo.constant dense<0.0> : tensor<f32>
    %v855 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v856 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v857 = stablehlo.reduce(%v853 init: %v854) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v858 = stablehlo.broadcast_in_dim %v857, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v859 = stablehlo.divide %v858, %v855 : tensor<32x512x7x7xf32>
    %v860 = stablehlo.subtract %v853, %v859 : tensor<32x512x7x7xf32>
    %v861 = stablehlo.multiply %v860, %v860 : tensor<32x512x7x7xf32>
    %v862 = stablehlo.reduce(%v861 init: %v854) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v863 = stablehlo.broadcast_in_dim %v862, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v864 = stablehlo.divide %v863, %v855 : tensor<32x512x7x7xf32>
    %v865 = stablehlo.add %v864, %v856 : tensor<32x512x7x7xf32>
    %v866 = stablehlo.rsqrt %v865 : tensor<32x512x7x7xf32>
    %v867 = stablehlo.multiply %v860, %v866 : tensor<32x512x7x7xf32>
    %v868 = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v869 = stablehlo.broadcast_in_dim %d4btp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v870 = stablehlo.multiply %v867, %v868 : tensor<32x512x7x7xf32>
    %v871 = stablehlo.add %v870, %v869 : tensor<32x512x7x7xf32>
    %v872 = stablehlo.reshape %v871 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v873 = stablehlo.add %v847, %v872 : tensor<32x25088xf32>
    %v874 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v875 = stablehlo.maximum %v873, %v874 : tensor<32x25088xf32>
    %v876 = stablehlo.reshape %v875 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v877 = stablehlo.convolution(%v876, %s4b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v878 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v879 = stablehlo.add %v877, %v878 : tensor<32x512x7x7xf32>
    %v880 = stablehlo.reshape %v879 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v881 = stablehlo.reshape %v880 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v882 = stablehlo.constant dense<0.0> : tensor<f32>
    %v883 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v884 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v885 = stablehlo.reduce(%v881 init: %v882) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v886 = stablehlo.broadcast_in_dim %v885, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v887 = stablehlo.divide %v886, %v883 : tensor<32x512x7x7xf32>
    %v888 = stablehlo.subtract %v881, %v887 : tensor<32x512x7x7xf32>
    %v889 = stablehlo.multiply %v888, %v888 : tensor<32x512x7x7xf32>
    %v890 = stablehlo.reduce(%v889 init: %v882) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v891 = stablehlo.broadcast_in_dim %v890, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v892 = stablehlo.divide %v891, %v883 : tensor<32x512x7x7xf32>
    %v893 = stablehlo.add %v892, %v884 : tensor<32x512x7x7xf32>
    %v894 = stablehlo.rsqrt %v893 : tensor<32x512x7x7xf32>
    %v895 = stablehlo.multiply %v888, %v894 : tensor<32x512x7x7xf32>
    %v896 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v897 = stablehlo.broadcast_in_dim %s4b0bt1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v898 = stablehlo.multiply %v895, %v896 : tensor<32x512x7x7xf32>
    %v899 = stablehlo.add %v898, %v897 : tensor<32x512x7x7xf32>
    %v900 = stablehlo.reshape %v899 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v901 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v902 = stablehlo.maximum %v900, %v901 : tensor<32x25088xf32>
    %v903 = stablehlo.reshape %v902 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v904 = stablehlo.convolution(%v903, %s4b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v905 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v906 = stablehlo.add %v904, %v905 : tensor<32x512x7x7xf32>
    %v907 = stablehlo.reshape %v906 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v908 = stablehlo.reshape %v907 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v909 = stablehlo.constant dense<0.0> : tensor<f32>
    %v910 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v911 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v912 = stablehlo.reduce(%v908 init: %v909) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v913 = stablehlo.broadcast_in_dim %v912, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v914 = stablehlo.divide %v913, %v910 : tensor<32x512x7x7xf32>
    %v915 = stablehlo.subtract %v908, %v914 : tensor<32x512x7x7xf32>
    %v916 = stablehlo.multiply %v915, %v915 : tensor<32x512x7x7xf32>
    %v917 = stablehlo.reduce(%v916 init: %v909) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v918 = stablehlo.broadcast_in_dim %v917, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v919 = stablehlo.divide %v918, %v910 : tensor<32x512x7x7xf32>
    %v920 = stablehlo.add %v919, %v911 : tensor<32x512x7x7xf32>
    %v921 = stablehlo.rsqrt %v920 : tensor<32x512x7x7xf32>
    %v922 = stablehlo.multiply %v915, %v921 : tensor<32x512x7x7xf32>
    %v923 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v924 = stablehlo.broadcast_in_dim %s4b0bt2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v925 = stablehlo.multiply %v922, %v923 : tensor<32x512x7x7xf32>
    %v926 = stablehlo.add %v925, %v924 : tensor<32x512x7x7xf32>
    %v927 = stablehlo.reshape %v926 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v928 = stablehlo.add %v927, %v875 : tensor<32x25088xf32>
    %v929 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v930 = stablehlo.maximum %v928, %v929 : tensor<32x25088xf32>
    %v931 = stablehlo.reshape %v930 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v932 = stablehlo.convolution(%v931, %s4b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v933 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v934 = stablehlo.add %v932, %v933 : tensor<32x512x7x7xf32>
    %v935 = stablehlo.reshape %v934 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v936 = stablehlo.reshape %v935 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v937 = stablehlo.constant dense<0.0> : tensor<f32>
    %v938 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v939 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v940 = stablehlo.reduce(%v936 init: %v937) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v941 = stablehlo.broadcast_in_dim %v940, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v942 = stablehlo.divide %v941, %v938 : tensor<32x512x7x7xf32>
    %v943 = stablehlo.subtract %v936, %v942 : tensor<32x512x7x7xf32>
    %v944 = stablehlo.multiply %v943, %v943 : tensor<32x512x7x7xf32>
    %v945 = stablehlo.reduce(%v944 init: %v937) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v946 = stablehlo.broadcast_in_dim %v945, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v947 = stablehlo.divide %v946, %v938 : tensor<32x512x7x7xf32>
    %v948 = stablehlo.add %v947, %v939 : tensor<32x512x7x7xf32>
    %v949 = stablehlo.rsqrt %v948 : tensor<32x512x7x7xf32>
    %v950 = stablehlo.multiply %v943, %v949 : tensor<32x512x7x7xf32>
    %v951 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v952 = stablehlo.broadcast_in_dim %s4b1bt1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v953 = stablehlo.multiply %v950, %v951 : tensor<32x512x7x7xf32>
    %v954 = stablehlo.add %v953, %v952 : tensor<32x512x7x7xf32>
    %v955 = stablehlo.reshape %v954 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v956 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v957 = stablehlo.maximum %v955, %v956 : tensor<32x25088xf32>
    %v958 = stablehlo.reshape %v957 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v959 = stablehlo.convolution(%v958, %s4b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v960 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v961 = stablehlo.add %v959, %v960 : tensor<32x512x7x7xf32>
    %v962 = stablehlo.reshape %v961 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v963 = stablehlo.reshape %v962 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v964 = stablehlo.constant dense<0.0> : tensor<f32>
    %v965 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v966 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v967 = stablehlo.reduce(%v963 init: %v964) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v968 = stablehlo.broadcast_in_dim %v967, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v969 = stablehlo.divide %v968, %v965 : tensor<32x512x7x7xf32>
    %v970 = stablehlo.subtract %v963, %v969 : tensor<32x512x7x7xf32>
    %v971 = stablehlo.multiply %v970, %v970 : tensor<32x512x7x7xf32>
    %v972 = stablehlo.reduce(%v971 init: %v964) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v973 = stablehlo.broadcast_in_dim %v972, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v974 = stablehlo.divide %v973, %v965 : tensor<32x512x7x7xf32>
    %v975 = stablehlo.add %v974, %v966 : tensor<32x512x7x7xf32>
    %v976 = stablehlo.rsqrt %v975 : tensor<32x512x7x7xf32>
    %v977 = stablehlo.multiply %v970, %v976 : tensor<32x512x7x7xf32>
    %v978 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v979 = stablehlo.broadcast_in_dim %s4b1bt2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v980 = stablehlo.multiply %v977, %v978 : tensor<32x512x7x7xf32>
    %v981 = stablehlo.add %v980, %v979 : tensor<32x512x7x7xf32>
    %v982 = stablehlo.reshape %v981 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v983 = stablehlo.add %v982, %v930 : tensor<32x25088xf32>
    %v984 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v985 = stablehlo.maximum %v983, %v984 : tensor<32x25088xf32>
    %v986 = stablehlo.reshape %v985 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v987 = stablehlo.constant dense<0.0> : tensor<f32>
    %v988 = stablehlo.reduce(%v986 init: %v987) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v989 = stablehlo.constant dense<49.0> : tensor<32x512xf32>
    %v990 = stablehlo.divide %v988, %v989 : tensor<32x512xf32>
    %v991 = stablehlo.dot_general %v990, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x512xf32>, tensor<512x10xf32>) -> tensor<32x10xf32>
    %v992 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v993 = stablehlo.add %v991, %v992 : tensor<32x10xf32>
    %v994 = stablehlo.exponential %v993 : tensor<32x10xf32>
    %v995 = stablehlo.constant dense<0.0> : tensor<f32>
    %v996 = stablehlo.reduce(%v994 init: %v995) applies stablehlo.add across dimensions = [1] : (tensor<32x10xf32>, tensor<f32>) -> tensor<32xf32>
    %v997 = stablehlo.broadcast_in_dim %v996, dims = [0] : (tensor<32xf32>) -> tensor<32x10xf32>
    %v998 = stablehlo.divide %v994, %v997 : tensor<32x10xf32>
    %v999 = stablehlo.subtract %v998, %onehot : tensor<32x10xf32>
    %v1000 = stablehlo.dot_general %v999, %Wd, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<512x10xf32>) -> tensor<32x512xf32>
    %v1001 = stablehlo.constant dense<49.0> : tensor<32x512xf32>
    %v1002 = stablehlo.divide %v1000, %v1001 : tensor<32x512xf32>
    %v1003 = stablehlo.broadcast_in_dim %v1002, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1004 = stablehlo.reshape %v1003 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1005 = stablehlo.dot_general %v990, %v999, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x512xf32>, tensor<32x10xf32>) -> tensor<512x10xf32>
    %v1006 = stablehlo.constant dense<0.003125> : tensor<512x10xf32>
    %v1007 = stablehlo.multiply %v1005, %v1006 : tensor<512x10xf32>
    %v1008 = stablehlo.subtract %Wd, %v1007 : tensor<512x10xf32>
    %v1009 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1010 = stablehlo.reduce(%v999 init: %v1009) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1011 = stablehlo.constant dense<0.003125> : tensor<10xf32>
    %v1012 = stablehlo.multiply %v1010, %v1011 : tensor<10xf32>
    %v1013 = stablehlo.subtract %bd, %v1012 : tensor<10xf32>
    %v1014 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v1015 = stablehlo.compare GT, %v983, %v1014 : (tensor<32x25088xf32>, tensor<32x25088xf32>) -> tensor<32x25088xi1>
    %v1016 = stablehlo.select %v1015, %v1004, %v1014 : tensor<32x25088xi1>, tensor<32x25088xf32>
    %v1017 = stablehlo.reshape %v1016 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1018 = stablehlo.reshape %v962 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1019 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1020 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1021 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1022 = stablehlo.reduce(%v1018 init: %v1019) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1023 = stablehlo.broadcast_in_dim %v1022, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1024 = stablehlo.divide %v1023, %v1020 : tensor<32x512x7x7xf32>
    %v1025 = stablehlo.subtract %v1018, %v1024 : tensor<32x512x7x7xf32>
    %v1026 = stablehlo.multiply %v1025, %v1025 : tensor<32x512x7x7xf32>
    %v1027 = stablehlo.reduce(%v1026 init: %v1019) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1028 = stablehlo.broadcast_in_dim %v1027, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1029 = stablehlo.divide %v1028, %v1020 : tensor<32x512x7x7xf32>
    %v1030 = stablehlo.add %v1029, %v1021 : tensor<32x512x7x7xf32>
    %v1031 = stablehlo.rsqrt %v1030 : tensor<32x512x7x7xf32>
    %v1032 = stablehlo.multiply %v1025, %v1031 : tensor<32x512x7x7xf32>
    %v1033 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1034 = stablehlo.multiply %v1033, %v1017 : tensor<32x512x7x7xf32>
    %v1035 = stablehlo.reduce(%v1034 init: %v1019) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1036 = stablehlo.broadcast_in_dim %v1035, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1037 = stablehlo.multiply %v1032, %v1034 : tensor<32x512x7x7xf32>
    %v1038 = stablehlo.reduce(%v1037 init: %v1019) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1039 = stablehlo.broadcast_in_dim %v1038, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1040 = stablehlo.multiply %v1034, %v1020 : tensor<32x512x7x7xf32>
    %v1041 = stablehlo.subtract %v1040, %v1036 : tensor<32x512x7x7xf32>
    %v1042 = stablehlo.multiply %v1032, %v1039 : tensor<32x512x7x7xf32>
    %v1043 = stablehlo.subtract %v1041, %v1042 : tensor<32x512x7x7xf32>
    %v1044 = stablehlo.divide %v1031, %v1020 : tensor<32x512x7x7xf32>
    %v1045 = stablehlo.multiply %v1044, %v1043 : tensor<32x512x7x7xf32>
    %v1046 = stablehlo.reshape %v1045 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1047 = stablehlo.reshape %v1046 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1048 = stablehlo.transpose %s4b1W2, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1049 = stablehlo.reverse %v1048, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1050 = stablehlo.convolution(%v1047, %v1049)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1051 = stablehlo.reshape %v1050 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1052 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v1053 = stablehlo.compare GT, %v955, %v1052 : (tensor<32x25088xf32>, tensor<32x25088xf32>) -> tensor<32x25088xi1>
    %v1054 = stablehlo.select %v1053, %v1051, %v1052 : tensor<32x25088xi1>, tensor<32x25088xf32>
    %v1055 = stablehlo.reshape %v1054 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1056 = stablehlo.reshape %v935 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1057 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1058 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1059 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1060 = stablehlo.reduce(%v1056 init: %v1057) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1061 = stablehlo.broadcast_in_dim %v1060, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1062 = stablehlo.divide %v1061, %v1058 : tensor<32x512x7x7xf32>
    %v1063 = stablehlo.subtract %v1056, %v1062 : tensor<32x512x7x7xf32>
    %v1064 = stablehlo.multiply %v1063, %v1063 : tensor<32x512x7x7xf32>
    %v1065 = stablehlo.reduce(%v1064 init: %v1057) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1066 = stablehlo.broadcast_in_dim %v1065, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1067 = stablehlo.divide %v1066, %v1058 : tensor<32x512x7x7xf32>
    %v1068 = stablehlo.add %v1067, %v1059 : tensor<32x512x7x7xf32>
    %v1069 = stablehlo.rsqrt %v1068 : tensor<32x512x7x7xf32>
    %v1070 = stablehlo.multiply %v1063, %v1069 : tensor<32x512x7x7xf32>
    %v1071 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1072 = stablehlo.multiply %v1071, %v1055 : tensor<32x512x7x7xf32>
    %v1073 = stablehlo.reduce(%v1072 init: %v1057) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1074 = stablehlo.broadcast_in_dim %v1073, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1075 = stablehlo.multiply %v1070, %v1072 : tensor<32x512x7x7xf32>
    %v1076 = stablehlo.reduce(%v1075 init: %v1057) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1077 = stablehlo.broadcast_in_dim %v1076, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1078 = stablehlo.multiply %v1072, %v1058 : tensor<32x512x7x7xf32>
    %v1079 = stablehlo.subtract %v1078, %v1074 : tensor<32x512x7x7xf32>
    %v1080 = stablehlo.multiply %v1070, %v1077 : tensor<32x512x7x7xf32>
    %v1081 = stablehlo.subtract %v1079, %v1080 : tensor<32x512x7x7xf32>
    %v1082 = stablehlo.divide %v1069, %v1058 : tensor<32x512x7x7xf32>
    %v1083 = stablehlo.multiply %v1082, %v1081 : tensor<32x512x7x7xf32>
    %v1084 = stablehlo.reshape %v1083 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1085 = stablehlo.reshape %v1084 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1086 = stablehlo.transpose %s4b1W1, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1087 = stablehlo.reverse %v1086, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1088 = stablehlo.convolution(%v1085, %v1087)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1089 = stablehlo.reshape %v1088 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1090 = stablehlo.add %v1089, %v1016 : tensor<32x25088xf32>
    %v1091 = stablehlo.reshape %v930 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1092 = stablehlo.reshape %v1084 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1093 = stablehlo.transpose %v1091, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1094 = stablehlo.transpose %v1092, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1095 = stablehlo.convolution(%v1093, %v1094)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1096 = stablehlo.transpose %v1095, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1097 = stablehlo.constant dense<0.003125> : tensor<512x512x3x3xf32>
    %v1098 = stablehlo.multiply %v1096, %v1097 : tensor<512x512x3x3xf32>
    %v1099 = stablehlo.subtract %s4b1W1, %v1098 : tensor<512x512x3x3xf32>
    %v1100 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1101 = stablehlo.reshape %v935 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1102 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1103 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1104 = stablehlo.reduce(%v1101 init: %v1100) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1105 = stablehlo.broadcast_in_dim %v1104, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1106 = stablehlo.divide %v1105, %v1102 : tensor<32x512x7x7xf32>
    %v1107 = stablehlo.subtract %v1101, %v1106 : tensor<32x512x7x7xf32>
    %v1108 = stablehlo.multiply %v1107, %v1107 : tensor<32x512x7x7xf32>
    %v1109 = stablehlo.reduce(%v1108 init: %v1100) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1110 = stablehlo.broadcast_in_dim %v1109, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1111 = stablehlo.divide %v1110, %v1102 : tensor<32x512x7x7xf32>
    %v1112 = stablehlo.add %v1111, %v1103 : tensor<32x512x7x7xf32>
    %v1113 = stablehlo.rsqrt %v1112 : tensor<32x512x7x7xf32>
    %v1114 = stablehlo.multiply %v1107, %v1113 : tensor<32x512x7x7xf32>
    %v1115 = stablehlo.reshape %v1054 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1116 = stablehlo.multiply %v1115, %v1114 : tensor<32x512x7x7xf32>
    %v1117 = stablehlo.reduce(%v1116 init: %v1100) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1118 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1119 = stablehlo.multiply %v1117, %v1118 : tensor<512xf32>
    %v1120 = stablehlo.subtract %s4b1g1, %v1119 : tensor<512xf32>
    %v1121 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1122 = stablehlo.reshape %v1054 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1123 = stablehlo.reduce(%v1122 init: %v1121) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1124 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1125 = stablehlo.multiply %v1123, %v1124 : tensor<512xf32>
    %v1126 = stablehlo.subtract %s4b1bt1, %v1125 : tensor<512xf32>
    %v1127 = stablehlo.reshape %v957 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1128 = stablehlo.reshape %v1046 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1129 = stablehlo.transpose %v1127, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1130 = stablehlo.transpose %v1128, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1131 = stablehlo.convolution(%v1129, %v1130)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1132 = stablehlo.transpose %v1131, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1133 = stablehlo.constant dense<0.003125> : tensor<512x512x3x3xf32>
    %v1134 = stablehlo.multiply %v1132, %v1133 : tensor<512x512x3x3xf32>
    %v1135 = stablehlo.subtract %s4b1W2, %v1134 : tensor<512x512x3x3xf32>
    %v1136 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1137 = stablehlo.reshape %v962 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1138 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1139 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1140 = stablehlo.reduce(%v1137 init: %v1136) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1141 = stablehlo.broadcast_in_dim %v1140, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1142 = stablehlo.divide %v1141, %v1138 : tensor<32x512x7x7xf32>
    %v1143 = stablehlo.subtract %v1137, %v1142 : tensor<32x512x7x7xf32>
    %v1144 = stablehlo.multiply %v1143, %v1143 : tensor<32x512x7x7xf32>
    %v1145 = stablehlo.reduce(%v1144 init: %v1136) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1146 = stablehlo.broadcast_in_dim %v1145, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1147 = stablehlo.divide %v1146, %v1138 : tensor<32x512x7x7xf32>
    %v1148 = stablehlo.add %v1147, %v1139 : tensor<32x512x7x7xf32>
    %v1149 = stablehlo.rsqrt %v1148 : tensor<32x512x7x7xf32>
    %v1150 = stablehlo.multiply %v1143, %v1149 : tensor<32x512x7x7xf32>
    %v1151 = stablehlo.reshape %v1016 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1152 = stablehlo.multiply %v1151, %v1150 : tensor<32x512x7x7xf32>
    %v1153 = stablehlo.reduce(%v1152 init: %v1136) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1154 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1155 = stablehlo.multiply %v1153, %v1154 : tensor<512xf32>
    %v1156 = stablehlo.subtract %s4b1g2, %v1155 : tensor<512xf32>
    %v1157 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1158 = stablehlo.reshape %v1016 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1159 = stablehlo.reduce(%v1158 init: %v1157) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1160 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1161 = stablehlo.multiply %v1159, %v1160 : tensor<512xf32>
    %v1162 = stablehlo.subtract %s4b1bt2, %v1161 : tensor<512xf32>
    %v1163 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v1164 = stablehlo.compare GT, %v928, %v1163 : (tensor<32x25088xf32>, tensor<32x25088xf32>) -> tensor<32x25088xi1>
    %v1165 = stablehlo.select %v1164, %v1090, %v1163 : tensor<32x25088xi1>, tensor<32x25088xf32>
    %v1166 = stablehlo.reshape %v1165 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1167 = stablehlo.reshape %v907 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1168 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1169 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1170 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1171 = stablehlo.reduce(%v1167 init: %v1168) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1172 = stablehlo.broadcast_in_dim %v1171, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1173 = stablehlo.divide %v1172, %v1169 : tensor<32x512x7x7xf32>
    %v1174 = stablehlo.subtract %v1167, %v1173 : tensor<32x512x7x7xf32>
    %v1175 = stablehlo.multiply %v1174, %v1174 : tensor<32x512x7x7xf32>
    %v1176 = stablehlo.reduce(%v1175 init: %v1168) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1177 = stablehlo.broadcast_in_dim %v1176, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1178 = stablehlo.divide %v1177, %v1169 : tensor<32x512x7x7xf32>
    %v1179 = stablehlo.add %v1178, %v1170 : tensor<32x512x7x7xf32>
    %v1180 = stablehlo.rsqrt %v1179 : tensor<32x512x7x7xf32>
    %v1181 = stablehlo.multiply %v1174, %v1180 : tensor<32x512x7x7xf32>
    %v1182 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1183 = stablehlo.multiply %v1182, %v1166 : tensor<32x512x7x7xf32>
    %v1184 = stablehlo.reduce(%v1183 init: %v1168) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1185 = stablehlo.broadcast_in_dim %v1184, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1186 = stablehlo.multiply %v1181, %v1183 : tensor<32x512x7x7xf32>
    %v1187 = stablehlo.reduce(%v1186 init: %v1168) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1188 = stablehlo.broadcast_in_dim %v1187, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1189 = stablehlo.multiply %v1183, %v1169 : tensor<32x512x7x7xf32>
    %v1190 = stablehlo.subtract %v1189, %v1185 : tensor<32x512x7x7xf32>
    %v1191 = stablehlo.multiply %v1181, %v1188 : tensor<32x512x7x7xf32>
    %v1192 = stablehlo.subtract %v1190, %v1191 : tensor<32x512x7x7xf32>
    %v1193 = stablehlo.divide %v1180, %v1169 : tensor<32x512x7x7xf32>
    %v1194 = stablehlo.multiply %v1193, %v1192 : tensor<32x512x7x7xf32>
    %v1195 = stablehlo.reshape %v1194 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1196 = stablehlo.reshape %v1195 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1197 = stablehlo.transpose %s4b0W2, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1198 = stablehlo.reverse %v1197, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1199 = stablehlo.convolution(%v1196, %v1198)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1200 = stablehlo.reshape %v1199 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1201 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v1202 = stablehlo.compare GT, %v900, %v1201 : (tensor<32x25088xf32>, tensor<32x25088xf32>) -> tensor<32x25088xi1>
    %v1203 = stablehlo.select %v1202, %v1200, %v1201 : tensor<32x25088xi1>, tensor<32x25088xf32>
    %v1204 = stablehlo.reshape %v1203 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1205 = stablehlo.reshape %v880 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1206 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1207 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1208 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1209 = stablehlo.reduce(%v1205 init: %v1206) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1210 = stablehlo.broadcast_in_dim %v1209, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1211 = stablehlo.divide %v1210, %v1207 : tensor<32x512x7x7xf32>
    %v1212 = stablehlo.subtract %v1205, %v1211 : tensor<32x512x7x7xf32>
    %v1213 = stablehlo.multiply %v1212, %v1212 : tensor<32x512x7x7xf32>
    %v1214 = stablehlo.reduce(%v1213 init: %v1206) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1215 = stablehlo.broadcast_in_dim %v1214, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1216 = stablehlo.divide %v1215, %v1207 : tensor<32x512x7x7xf32>
    %v1217 = stablehlo.add %v1216, %v1208 : tensor<32x512x7x7xf32>
    %v1218 = stablehlo.rsqrt %v1217 : tensor<32x512x7x7xf32>
    %v1219 = stablehlo.multiply %v1212, %v1218 : tensor<32x512x7x7xf32>
    %v1220 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1221 = stablehlo.multiply %v1220, %v1204 : tensor<32x512x7x7xf32>
    %v1222 = stablehlo.reduce(%v1221 init: %v1206) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1223 = stablehlo.broadcast_in_dim %v1222, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1224 = stablehlo.multiply %v1219, %v1221 : tensor<32x512x7x7xf32>
    %v1225 = stablehlo.reduce(%v1224 init: %v1206) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1226 = stablehlo.broadcast_in_dim %v1225, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1227 = stablehlo.multiply %v1221, %v1207 : tensor<32x512x7x7xf32>
    %v1228 = stablehlo.subtract %v1227, %v1223 : tensor<32x512x7x7xf32>
    %v1229 = stablehlo.multiply %v1219, %v1226 : tensor<32x512x7x7xf32>
    %v1230 = stablehlo.subtract %v1228, %v1229 : tensor<32x512x7x7xf32>
    %v1231 = stablehlo.divide %v1218, %v1207 : tensor<32x512x7x7xf32>
    %v1232 = stablehlo.multiply %v1231, %v1230 : tensor<32x512x7x7xf32>
    %v1233 = stablehlo.reshape %v1232 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1234 = stablehlo.reshape %v1233 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1235 = stablehlo.transpose %s4b0W1, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1236 = stablehlo.reverse %v1235, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1237 = stablehlo.convolution(%v1234, %v1236)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1238 = stablehlo.reshape %v1237 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1239 = stablehlo.add %v1238, %v1165 : tensor<32x25088xf32>
    %v1240 = stablehlo.reshape %v875 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1241 = stablehlo.reshape %v1233 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1242 = stablehlo.transpose %v1240, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1243 = stablehlo.transpose %v1241, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1244 = stablehlo.convolution(%v1242, %v1243)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1245 = stablehlo.transpose %v1244, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1246 = stablehlo.constant dense<0.003125> : tensor<512x512x3x3xf32>
    %v1247 = stablehlo.multiply %v1245, %v1246 : tensor<512x512x3x3xf32>
    %v1248 = stablehlo.subtract %s4b0W1, %v1247 : tensor<512x512x3x3xf32>
    %v1249 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1250 = stablehlo.reshape %v880 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1251 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1252 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1253 = stablehlo.reduce(%v1250 init: %v1249) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1254 = stablehlo.broadcast_in_dim %v1253, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1255 = stablehlo.divide %v1254, %v1251 : tensor<32x512x7x7xf32>
    %v1256 = stablehlo.subtract %v1250, %v1255 : tensor<32x512x7x7xf32>
    %v1257 = stablehlo.multiply %v1256, %v1256 : tensor<32x512x7x7xf32>
    %v1258 = stablehlo.reduce(%v1257 init: %v1249) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1259 = stablehlo.broadcast_in_dim %v1258, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1260 = stablehlo.divide %v1259, %v1251 : tensor<32x512x7x7xf32>
    %v1261 = stablehlo.add %v1260, %v1252 : tensor<32x512x7x7xf32>
    %v1262 = stablehlo.rsqrt %v1261 : tensor<32x512x7x7xf32>
    %v1263 = stablehlo.multiply %v1256, %v1262 : tensor<32x512x7x7xf32>
    %v1264 = stablehlo.reshape %v1203 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1265 = stablehlo.multiply %v1264, %v1263 : tensor<32x512x7x7xf32>
    %v1266 = stablehlo.reduce(%v1265 init: %v1249) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1267 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1268 = stablehlo.multiply %v1266, %v1267 : tensor<512xf32>
    %v1269 = stablehlo.subtract %s4b0g1, %v1268 : tensor<512xf32>
    %v1270 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1271 = stablehlo.reshape %v1203 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1272 = stablehlo.reduce(%v1271 init: %v1270) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1273 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1274 = stablehlo.multiply %v1272, %v1273 : tensor<512xf32>
    %v1275 = stablehlo.subtract %s4b0bt1, %v1274 : tensor<512xf32>
    %v1276 = stablehlo.reshape %v902 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1277 = stablehlo.reshape %v1195 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1278 = stablehlo.transpose %v1276, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1279 = stablehlo.transpose %v1277, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1280 = stablehlo.convolution(%v1278, %v1279)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1281 = stablehlo.transpose %v1280, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1282 = stablehlo.constant dense<0.003125> : tensor<512x512x3x3xf32>
    %v1283 = stablehlo.multiply %v1281, %v1282 : tensor<512x512x3x3xf32>
    %v1284 = stablehlo.subtract %s4b0W2, %v1283 : tensor<512x512x3x3xf32>
    %v1285 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1286 = stablehlo.reshape %v907 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1287 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1288 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1289 = stablehlo.reduce(%v1286 init: %v1285) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1290 = stablehlo.broadcast_in_dim %v1289, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1291 = stablehlo.divide %v1290, %v1287 : tensor<32x512x7x7xf32>
    %v1292 = stablehlo.subtract %v1286, %v1291 : tensor<32x512x7x7xf32>
    %v1293 = stablehlo.multiply %v1292, %v1292 : tensor<32x512x7x7xf32>
    %v1294 = stablehlo.reduce(%v1293 init: %v1285) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1295 = stablehlo.broadcast_in_dim %v1294, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1296 = stablehlo.divide %v1295, %v1287 : tensor<32x512x7x7xf32>
    %v1297 = stablehlo.add %v1296, %v1288 : tensor<32x512x7x7xf32>
    %v1298 = stablehlo.rsqrt %v1297 : tensor<32x512x7x7xf32>
    %v1299 = stablehlo.multiply %v1292, %v1298 : tensor<32x512x7x7xf32>
    %v1300 = stablehlo.reshape %v1165 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1301 = stablehlo.multiply %v1300, %v1299 : tensor<32x512x7x7xf32>
    %v1302 = stablehlo.reduce(%v1301 init: %v1285) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1303 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1304 = stablehlo.multiply %v1302, %v1303 : tensor<512xf32>
    %v1305 = stablehlo.subtract %s4b0g2, %v1304 : tensor<512xf32>
    %v1306 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1307 = stablehlo.reshape %v1165 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1308 = stablehlo.reduce(%v1307 init: %v1306) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1309 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1310 = stablehlo.multiply %v1308, %v1309 : tensor<512xf32>
    %v1311 = stablehlo.subtract %s4b0bt2, %v1310 : tensor<512xf32>
    %v1312 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v1313 = stablehlo.compare GT, %v873, %v1312 : (tensor<32x25088xf32>, tensor<32x25088xf32>) -> tensor<32x25088xi1>
    %v1314 = stablehlo.select %v1313, %v1239, %v1312 : tensor<32x25088xi1>, tensor<32x25088xf32>
    %v1315 = stablehlo.reshape %v1314 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1316 = stablehlo.reshape %v827 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1317 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1318 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1319 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1320 = stablehlo.reduce(%v1316 init: %v1317) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1321 = stablehlo.broadcast_in_dim %v1320, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1322 = stablehlo.divide %v1321, %v1318 : tensor<32x512x7x7xf32>
    %v1323 = stablehlo.subtract %v1316, %v1322 : tensor<32x512x7x7xf32>
    %v1324 = stablehlo.multiply %v1323, %v1323 : tensor<32x512x7x7xf32>
    %v1325 = stablehlo.reduce(%v1324 init: %v1317) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1326 = stablehlo.broadcast_in_dim %v1325, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1327 = stablehlo.divide %v1326, %v1318 : tensor<32x512x7x7xf32>
    %v1328 = stablehlo.add %v1327, %v1319 : tensor<32x512x7x7xf32>
    %v1329 = stablehlo.rsqrt %v1328 : tensor<32x512x7x7xf32>
    %v1330 = stablehlo.multiply %v1323, %v1329 : tensor<32x512x7x7xf32>
    %v1331 = stablehlo.broadcast_in_dim %d4g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1332 = stablehlo.multiply %v1331, %v1315 : tensor<32x512x7x7xf32>
    %v1333 = stablehlo.reduce(%v1332 init: %v1317) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1334 = stablehlo.broadcast_in_dim %v1333, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1335 = stablehlo.multiply %v1330, %v1332 : tensor<32x512x7x7xf32>
    %v1336 = stablehlo.reduce(%v1335 init: %v1317) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1337 = stablehlo.broadcast_in_dim %v1336, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1338 = stablehlo.multiply %v1332, %v1318 : tensor<32x512x7x7xf32>
    %v1339 = stablehlo.subtract %v1338, %v1334 : tensor<32x512x7x7xf32>
    %v1340 = stablehlo.multiply %v1330, %v1337 : tensor<32x512x7x7xf32>
    %v1341 = stablehlo.subtract %v1339, %v1340 : tensor<32x512x7x7xf32>
    %v1342 = stablehlo.divide %v1329, %v1318 : tensor<32x512x7x7xf32>
    %v1343 = stablehlo.multiply %v1342, %v1341 : tensor<32x512x7x7xf32>
    %v1344 = stablehlo.reshape %v1343 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1345 = stablehlo.reshape %v1344 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1346 = stablehlo.transpose %d4W2, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1347 = stablehlo.reverse %v1346, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1348 = stablehlo.convolution(%v1345, %v1347)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1349 = stablehlo.reshape %v1348 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1350 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v1351 = stablehlo.compare GT, %v820, %v1350 : (tensor<32x25088xf32>, tensor<32x25088xf32>) -> tensor<32x25088xi1>
    %v1352 = stablehlo.select %v1351, %v1349, %v1350 : tensor<32x25088xi1>, tensor<32x25088xf32>
    %v1353 = stablehlo.reshape %v1352 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1354 = stablehlo.reshape %v800 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1355 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1356 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1357 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1358 = stablehlo.reduce(%v1354 init: %v1355) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1359 = stablehlo.broadcast_in_dim %v1358, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1360 = stablehlo.divide %v1359, %v1356 : tensor<32x512x7x7xf32>
    %v1361 = stablehlo.subtract %v1354, %v1360 : tensor<32x512x7x7xf32>
    %v1362 = stablehlo.multiply %v1361, %v1361 : tensor<32x512x7x7xf32>
    %v1363 = stablehlo.reduce(%v1362 init: %v1355) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1364 = stablehlo.broadcast_in_dim %v1363, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1365 = stablehlo.divide %v1364, %v1356 : tensor<32x512x7x7xf32>
    %v1366 = stablehlo.add %v1365, %v1357 : tensor<32x512x7x7xf32>
    %v1367 = stablehlo.rsqrt %v1366 : tensor<32x512x7x7xf32>
    %v1368 = stablehlo.multiply %v1361, %v1367 : tensor<32x512x7x7xf32>
    %v1369 = stablehlo.broadcast_in_dim %d4g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1370 = stablehlo.multiply %v1369, %v1353 : tensor<32x512x7x7xf32>
    %v1371 = stablehlo.reduce(%v1370 init: %v1355) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1372 = stablehlo.broadcast_in_dim %v1371, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1373 = stablehlo.multiply %v1368, %v1370 : tensor<32x512x7x7xf32>
    %v1374 = stablehlo.reduce(%v1373 init: %v1355) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1375 = stablehlo.broadcast_in_dim %v1374, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1376 = stablehlo.multiply %v1370, %v1356 : tensor<32x512x7x7xf32>
    %v1377 = stablehlo.subtract %v1376, %v1372 : tensor<32x512x7x7xf32>
    %v1378 = stablehlo.multiply %v1368, %v1375 : tensor<32x512x7x7xf32>
    %v1379 = stablehlo.subtract %v1377, %v1378 : tensor<32x512x7x7xf32>
    %v1380 = stablehlo.divide %v1367, %v1356 : tensor<32x512x7x7xf32>
    %v1381 = stablehlo.multiply %v1380, %v1379 : tensor<32x512x7x7xf32>
    %v1382 = stablehlo.reshape %v1381 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1383 = stablehlo.reshape %v1382 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1384 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1385 = stablehlo.pad %v1383, %v1384, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %v1386 = stablehlo.transpose %d4W1, dims = [1, 0, 2, 3] : (tensor<512x256x3x3xf32>) -> tensor<256x512x3x3xf32>
    %v1387 = stablehlo.reverse %v1386, dims = [2, 3] : tensor<256x512x3x3xf32>
    %v1388 = stablehlo.convolution(%v1385, %v1387)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<256x512x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1389 = stablehlo.reshape %v1388 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1390 = stablehlo.reshape %v1314 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1391 = stablehlo.reshape %v852 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1392 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1393 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1394 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1395 = stablehlo.reduce(%v1391 init: %v1392) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1396 = stablehlo.broadcast_in_dim %v1395, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1397 = stablehlo.divide %v1396, %v1393 : tensor<32x512x7x7xf32>
    %v1398 = stablehlo.subtract %v1391, %v1397 : tensor<32x512x7x7xf32>
    %v1399 = stablehlo.multiply %v1398, %v1398 : tensor<32x512x7x7xf32>
    %v1400 = stablehlo.reduce(%v1399 init: %v1392) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1401 = stablehlo.broadcast_in_dim %v1400, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1402 = stablehlo.divide %v1401, %v1393 : tensor<32x512x7x7xf32>
    %v1403 = stablehlo.add %v1402, %v1394 : tensor<32x512x7x7xf32>
    %v1404 = stablehlo.rsqrt %v1403 : tensor<32x512x7x7xf32>
    %v1405 = stablehlo.multiply %v1398, %v1404 : tensor<32x512x7x7xf32>
    %v1406 = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1407 = stablehlo.multiply %v1406, %v1390 : tensor<32x512x7x7xf32>
    %v1408 = stablehlo.reduce(%v1407 init: %v1392) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1409 = stablehlo.broadcast_in_dim %v1408, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1410 = stablehlo.multiply %v1405, %v1407 : tensor<32x512x7x7xf32>
    %v1411 = stablehlo.reduce(%v1410 init: %v1392) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1412 = stablehlo.broadcast_in_dim %v1411, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1413 = stablehlo.multiply %v1407, %v1393 : tensor<32x512x7x7xf32>
    %v1414 = stablehlo.subtract %v1413, %v1409 : tensor<32x512x7x7xf32>
    %v1415 = stablehlo.multiply %v1405, %v1412 : tensor<32x512x7x7xf32>
    %v1416 = stablehlo.subtract %v1414, %v1415 : tensor<32x512x7x7xf32>
    %v1417 = stablehlo.divide %v1404, %v1393 : tensor<32x512x7x7xf32>
    %v1418 = stablehlo.multiply %v1417, %v1416 : tensor<32x512x7x7xf32>
    %v1419 = stablehlo.reshape %v1418 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1420 = stablehlo.reshape %v1419 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1421 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1422 = stablehlo.pad %v1420, %v1421, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %v1423 = stablehlo.transpose %d4Wp, dims = [1, 0, 2, 3] : (tensor<512x256x3x3xf32>) -> tensor<256x512x3x3xf32>
    %v1424 = stablehlo.reverse %v1423, dims = [2, 3] : tensor<256x512x3x3xf32>
    %v1425 = stablehlo.convolution(%v1422, %v1424)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<256x512x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1426 = stablehlo.reshape %v1425 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1427 = stablehlo.add %v1389, %v1426 : tensor<32x50176xf32>
    %v1428 = stablehlo.reshape %v795 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1429 = stablehlo.reshape %v1382 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1430 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1431 = stablehlo.pad %v1429, %v1430, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %v1432 = stablehlo.transpose %v1428, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1433 = stablehlo.transpose %v1431, dims = [1, 0, 2, 3] : (tensor<32x512x14x14xf32>) -> tensor<512x32x14x14xf32>
    %v1434 = stablehlo.convolution(%v1432, %v1433)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<512x32x14x14xf32>) -> tensor<256x512x3x3xf32>
    %v1435 = stablehlo.transpose %v1434, dims = [1, 0, 2, 3] : (tensor<256x512x3x3xf32>) -> tensor<512x256x3x3xf32>
    %v1436 = stablehlo.constant dense<0.003125> : tensor<512x256x3x3xf32>
    %v1437 = stablehlo.multiply %v1435, %v1436 : tensor<512x256x3x3xf32>
    %v1438 = stablehlo.subtract %d4W1, %v1437 : tensor<512x256x3x3xf32>
    %v1439 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1440 = stablehlo.reshape %v800 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1441 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1442 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1443 = stablehlo.reduce(%v1440 init: %v1439) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1444 = stablehlo.broadcast_in_dim %v1443, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1445 = stablehlo.divide %v1444, %v1441 : tensor<32x512x7x7xf32>
    %v1446 = stablehlo.subtract %v1440, %v1445 : tensor<32x512x7x7xf32>
    %v1447 = stablehlo.multiply %v1446, %v1446 : tensor<32x512x7x7xf32>
    %v1448 = stablehlo.reduce(%v1447 init: %v1439) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1449 = stablehlo.broadcast_in_dim %v1448, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1450 = stablehlo.divide %v1449, %v1441 : tensor<32x512x7x7xf32>
    %v1451 = stablehlo.add %v1450, %v1442 : tensor<32x512x7x7xf32>
    %v1452 = stablehlo.rsqrt %v1451 : tensor<32x512x7x7xf32>
    %v1453 = stablehlo.multiply %v1446, %v1452 : tensor<32x512x7x7xf32>
    %v1454 = stablehlo.reshape %v1352 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1455 = stablehlo.multiply %v1454, %v1453 : tensor<32x512x7x7xf32>
    %v1456 = stablehlo.reduce(%v1455 init: %v1439) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1457 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1458 = stablehlo.multiply %v1456, %v1457 : tensor<512xf32>
    %v1459 = stablehlo.subtract %d4g1, %v1458 : tensor<512xf32>
    %v1460 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1461 = stablehlo.reshape %v1352 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1462 = stablehlo.reduce(%v1461 init: %v1460) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1463 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1464 = stablehlo.multiply %v1462, %v1463 : tensor<512xf32>
    %v1465 = stablehlo.subtract %d4bt1, %v1464 : tensor<512xf32>
    %v1466 = stablehlo.reshape %v822 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1467 = stablehlo.reshape %v1344 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1468 = stablehlo.transpose %v1466, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1469 = stablehlo.transpose %v1467, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1470 = stablehlo.convolution(%v1468, %v1469)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1471 = stablehlo.transpose %v1470, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1472 = stablehlo.constant dense<0.003125> : tensor<512x512x3x3xf32>
    %v1473 = stablehlo.multiply %v1471, %v1472 : tensor<512x512x3x3xf32>
    %v1474 = stablehlo.subtract %d4W2, %v1473 : tensor<512x512x3x3xf32>
    %v1475 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1476 = stablehlo.reshape %v827 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1477 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1478 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1479 = stablehlo.reduce(%v1476 init: %v1475) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1480 = stablehlo.broadcast_in_dim %v1479, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1481 = stablehlo.divide %v1480, %v1477 : tensor<32x512x7x7xf32>
    %v1482 = stablehlo.subtract %v1476, %v1481 : tensor<32x512x7x7xf32>
    %v1483 = stablehlo.multiply %v1482, %v1482 : tensor<32x512x7x7xf32>
    %v1484 = stablehlo.reduce(%v1483 init: %v1475) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1485 = stablehlo.broadcast_in_dim %v1484, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1486 = stablehlo.divide %v1485, %v1477 : tensor<32x512x7x7xf32>
    %v1487 = stablehlo.add %v1486, %v1478 : tensor<32x512x7x7xf32>
    %v1488 = stablehlo.rsqrt %v1487 : tensor<32x512x7x7xf32>
    %v1489 = stablehlo.multiply %v1482, %v1488 : tensor<32x512x7x7xf32>
    %v1490 = stablehlo.reshape %v1314 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1491 = stablehlo.multiply %v1490, %v1489 : tensor<32x512x7x7xf32>
    %v1492 = stablehlo.reduce(%v1491 init: %v1475) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1493 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1494 = stablehlo.multiply %v1492, %v1493 : tensor<512xf32>
    %v1495 = stablehlo.subtract %d4g2, %v1494 : tensor<512xf32>
    %v1496 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1497 = stablehlo.reshape %v1314 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1498 = stablehlo.reduce(%v1497 init: %v1496) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1499 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1500 = stablehlo.multiply %v1498, %v1499 : tensor<512xf32>
    %v1501 = stablehlo.subtract %d4bt2, %v1500 : tensor<512xf32>
    %v1502 = stablehlo.reshape %v795 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1503 = stablehlo.reshape %v1419 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1504 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1505 = stablehlo.pad %v1503, %v1504, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %v1506 = stablehlo.transpose %v1502, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1507 = stablehlo.transpose %v1505, dims = [1, 0, 2, 3] : (tensor<32x512x14x14xf32>) -> tensor<512x32x14x14xf32>
    %v1508 = stablehlo.convolution(%v1506, %v1507)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<512x32x14x14xf32>) -> tensor<256x512x3x3xf32>
    %v1509 = stablehlo.transpose %v1508, dims = [1, 0, 2, 3] : (tensor<256x512x3x3xf32>) -> tensor<512x256x3x3xf32>
    %v1510 = stablehlo.constant dense<0.003125> : tensor<512x256x3x3xf32>
    %v1511 = stablehlo.multiply %v1509, %v1510 : tensor<512x256x3x3xf32>
    %v1512 = stablehlo.subtract %d4Wp, %v1511 : tensor<512x256x3x3xf32>
    %v1513 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1514 = stablehlo.reshape %v852 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1515 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1516 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1517 = stablehlo.reduce(%v1514 init: %v1513) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1518 = stablehlo.broadcast_in_dim %v1517, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1519 = stablehlo.divide %v1518, %v1515 : tensor<32x512x7x7xf32>
    %v1520 = stablehlo.subtract %v1514, %v1519 : tensor<32x512x7x7xf32>
    %v1521 = stablehlo.multiply %v1520, %v1520 : tensor<32x512x7x7xf32>
    %v1522 = stablehlo.reduce(%v1521 init: %v1513) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1523 = stablehlo.broadcast_in_dim %v1522, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1524 = stablehlo.divide %v1523, %v1515 : tensor<32x512x7x7xf32>
    %v1525 = stablehlo.add %v1524, %v1516 : tensor<32x512x7x7xf32>
    %v1526 = stablehlo.rsqrt %v1525 : tensor<32x512x7x7xf32>
    %v1527 = stablehlo.multiply %v1520, %v1526 : tensor<32x512x7x7xf32>
    %v1528 = stablehlo.reshape %v1314 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1529 = stablehlo.multiply %v1528, %v1527 : tensor<32x512x7x7xf32>
    %v1530 = stablehlo.reduce(%v1529 init: %v1513) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1531 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1532 = stablehlo.multiply %v1530, %v1531 : tensor<512xf32>
    %v1533 = stablehlo.subtract %d4gp, %v1532 : tensor<512xf32>
    %v1534 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1535 = stablehlo.reshape %v1314 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1536 = stablehlo.reduce(%v1535 init: %v1534) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1537 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1538 = stablehlo.multiply %v1536, %v1537 : tensor<512xf32>
    %v1539 = stablehlo.subtract %d4btp, %v1538 : tensor<512xf32>
    %v1540 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1541 = stablehlo.compare GT, %v793, %v1540 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v1542 = stablehlo.select %v1541, %v1427, %v1540 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v1543 = stablehlo.reshape %v1542 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1544 = stablehlo.reshape %v772 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1545 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1546 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v1547 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1548 = stablehlo.reduce(%v1544 init: %v1545) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1549 = stablehlo.broadcast_in_dim %v1548, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1550 = stablehlo.divide %v1549, %v1546 : tensor<32x256x14x14xf32>
    %v1551 = stablehlo.subtract %v1544, %v1550 : tensor<32x256x14x14xf32>
    %v1552 = stablehlo.multiply %v1551, %v1551 : tensor<32x256x14x14xf32>
    %v1553 = stablehlo.reduce(%v1552 init: %v1545) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1554 = stablehlo.broadcast_in_dim %v1553, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1555 = stablehlo.divide %v1554, %v1546 : tensor<32x256x14x14xf32>
    %v1556 = stablehlo.add %v1555, %v1547 : tensor<32x256x14x14xf32>
    %v1557 = stablehlo.rsqrt %v1556 : tensor<32x256x14x14xf32>
    %v1558 = stablehlo.multiply %v1551, %v1557 : tensor<32x256x14x14xf32>
    %v1559 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1560 = stablehlo.multiply %v1559, %v1543 : tensor<32x256x14x14xf32>
    %v1561 = stablehlo.reduce(%v1560 init: %v1545) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1562 = stablehlo.broadcast_in_dim %v1561, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1563 = stablehlo.multiply %v1558, %v1560 : tensor<32x256x14x14xf32>
    %v1564 = stablehlo.reduce(%v1563 init: %v1545) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1565 = stablehlo.broadcast_in_dim %v1564, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1566 = stablehlo.multiply %v1560, %v1546 : tensor<32x256x14x14xf32>
    %v1567 = stablehlo.subtract %v1566, %v1562 : tensor<32x256x14x14xf32>
    %v1568 = stablehlo.multiply %v1558, %v1565 : tensor<32x256x14x14xf32>
    %v1569 = stablehlo.subtract %v1567, %v1568 : tensor<32x256x14x14xf32>
    %v1570 = stablehlo.divide %v1557, %v1546 : tensor<32x256x14x14xf32>
    %v1571 = stablehlo.multiply %v1570, %v1569 : tensor<32x256x14x14xf32>
    %v1572 = stablehlo.reshape %v1571 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1573 = stablehlo.reshape %v1572 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1574 = stablehlo.transpose %s3b4W2, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1575 = stablehlo.reverse %v1574, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1576 = stablehlo.convolution(%v1573, %v1575)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1577 = stablehlo.reshape %v1576 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1578 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1579 = stablehlo.compare GT, %v765, %v1578 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v1580 = stablehlo.select %v1579, %v1577, %v1578 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v1581 = stablehlo.reshape %v1580 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1582 = stablehlo.reshape %v745 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1583 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1584 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v1585 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1586 = stablehlo.reduce(%v1582 init: %v1583) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1587 = stablehlo.broadcast_in_dim %v1586, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1588 = stablehlo.divide %v1587, %v1584 : tensor<32x256x14x14xf32>
    %v1589 = stablehlo.subtract %v1582, %v1588 : tensor<32x256x14x14xf32>
    %v1590 = stablehlo.multiply %v1589, %v1589 : tensor<32x256x14x14xf32>
    %v1591 = stablehlo.reduce(%v1590 init: %v1583) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1592 = stablehlo.broadcast_in_dim %v1591, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1593 = stablehlo.divide %v1592, %v1584 : tensor<32x256x14x14xf32>
    %v1594 = stablehlo.add %v1593, %v1585 : tensor<32x256x14x14xf32>
    %v1595 = stablehlo.rsqrt %v1594 : tensor<32x256x14x14xf32>
    %v1596 = stablehlo.multiply %v1589, %v1595 : tensor<32x256x14x14xf32>
    %v1597 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1598 = stablehlo.multiply %v1597, %v1581 : tensor<32x256x14x14xf32>
    %v1599 = stablehlo.reduce(%v1598 init: %v1583) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1600 = stablehlo.broadcast_in_dim %v1599, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1601 = stablehlo.multiply %v1596, %v1598 : tensor<32x256x14x14xf32>
    %v1602 = stablehlo.reduce(%v1601 init: %v1583) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1603 = stablehlo.broadcast_in_dim %v1602, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1604 = stablehlo.multiply %v1598, %v1584 : tensor<32x256x14x14xf32>
    %v1605 = stablehlo.subtract %v1604, %v1600 : tensor<32x256x14x14xf32>
    %v1606 = stablehlo.multiply %v1596, %v1603 : tensor<32x256x14x14xf32>
    %v1607 = stablehlo.subtract %v1605, %v1606 : tensor<32x256x14x14xf32>
    %v1608 = stablehlo.divide %v1595, %v1584 : tensor<32x256x14x14xf32>
    %v1609 = stablehlo.multiply %v1608, %v1607 : tensor<32x256x14x14xf32>
    %v1610 = stablehlo.reshape %v1609 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1611 = stablehlo.reshape %v1610 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1612 = stablehlo.transpose %s3b4W1, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1613 = stablehlo.reverse %v1612, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1614 = stablehlo.convolution(%v1611, %v1613)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1615 = stablehlo.reshape %v1614 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1616 = stablehlo.add %v1615, %v1542 : tensor<32x50176xf32>
    %v1617 = stablehlo.reshape %v740 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1618 = stablehlo.reshape %v1610 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1619 = stablehlo.transpose %v1617, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1620 = stablehlo.transpose %v1618, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1621 = stablehlo.convolution(%v1619, %v1620)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1622 = stablehlo.transpose %v1621, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1623 = stablehlo.constant dense<0.003125> : tensor<256x256x3x3xf32>
    %v1624 = stablehlo.multiply %v1622, %v1623 : tensor<256x256x3x3xf32>
    %v1625 = stablehlo.subtract %s3b4W1, %v1624 : tensor<256x256x3x3xf32>
    %v1626 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1627 = stablehlo.reshape %v745 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1628 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v1629 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1630 = stablehlo.reduce(%v1627 init: %v1626) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1631 = stablehlo.broadcast_in_dim %v1630, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1632 = stablehlo.divide %v1631, %v1628 : tensor<32x256x14x14xf32>
    %v1633 = stablehlo.subtract %v1627, %v1632 : tensor<32x256x14x14xf32>
    %v1634 = stablehlo.multiply %v1633, %v1633 : tensor<32x256x14x14xf32>
    %v1635 = stablehlo.reduce(%v1634 init: %v1626) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1636 = stablehlo.broadcast_in_dim %v1635, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1637 = stablehlo.divide %v1636, %v1628 : tensor<32x256x14x14xf32>
    %v1638 = stablehlo.add %v1637, %v1629 : tensor<32x256x14x14xf32>
    %v1639 = stablehlo.rsqrt %v1638 : tensor<32x256x14x14xf32>
    %v1640 = stablehlo.multiply %v1633, %v1639 : tensor<32x256x14x14xf32>
    %v1641 = stablehlo.reshape %v1580 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1642 = stablehlo.multiply %v1641, %v1640 : tensor<32x256x14x14xf32>
    %v1643 = stablehlo.reduce(%v1642 init: %v1626) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1644 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v1645 = stablehlo.multiply %v1643, %v1644 : tensor<256xf32>
    %v1646 = stablehlo.subtract %s3b4g1, %v1645 : tensor<256xf32>
    %v1647 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1648 = stablehlo.reshape %v1580 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1649 = stablehlo.reduce(%v1648 init: %v1647) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1650 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v1651 = stablehlo.multiply %v1649, %v1650 : tensor<256xf32>
    %v1652 = stablehlo.subtract %s3b4bt1, %v1651 : tensor<256xf32>
    %v1653 = stablehlo.reshape %v767 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1654 = stablehlo.reshape %v1572 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1655 = stablehlo.transpose %v1653, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1656 = stablehlo.transpose %v1654, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1657 = stablehlo.convolution(%v1655, %v1656)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1658 = stablehlo.transpose %v1657, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1659 = stablehlo.constant dense<0.003125> : tensor<256x256x3x3xf32>
    %v1660 = stablehlo.multiply %v1658, %v1659 : tensor<256x256x3x3xf32>
    %v1661 = stablehlo.subtract %s3b4W2, %v1660 : tensor<256x256x3x3xf32>
    %v1662 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1663 = stablehlo.reshape %v772 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1664 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v1665 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1666 = stablehlo.reduce(%v1663 init: %v1662) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1667 = stablehlo.broadcast_in_dim %v1666, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1668 = stablehlo.divide %v1667, %v1664 : tensor<32x256x14x14xf32>
    %v1669 = stablehlo.subtract %v1663, %v1668 : tensor<32x256x14x14xf32>
    %v1670 = stablehlo.multiply %v1669, %v1669 : tensor<32x256x14x14xf32>
    %v1671 = stablehlo.reduce(%v1670 init: %v1662) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1672 = stablehlo.broadcast_in_dim %v1671, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1673 = stablehlo.divide %v1672, %v1664 : tensor<32x256x14x14xf32>
    %v1674 = stablehlo.add %v1673, %v1665 : tensor<32x256x14x14xf32>
    %v1675 = stablehlo.rsqrt %v1674 : tensor<32x256x14x14xf32>
    %v1676 = stablehlo.multiply %v1669, %v1675 : tensor<32x256x14x14xf32>
    %v1677 = stablehlo.reshape %v1542 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1678 = stablehlo.multiply %v1677, %v1676 : tensor<32x256x14x14xf32>
    %v1679 = stablehlo.reduce(%v1678 init: %v1662) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1680 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v1681 = stablehlo.multiply %v1679, %v1680 : tensor<256xf32>
    %v1682 = stablehlo.subtract %s3b4g2, %v1681 : tensor<256xf32>
    %v1683 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1684 = stablehlo.reshape %v1542 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1685 = stablehlo.reduce(%v1684 init: %v1683) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1686 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v1687 = stablehlo.multiply %v1685, %v1686 : tensor<256xf32>
    %v1688 = stablehlo.subtract %s3b4bt2, %v1687 : tensor<256xf32>
    %v1689 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1690 = stablehlo.compare GT, %v738, %v1689 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v1691 = stablehlo.select %v1690, %v1616, %v1689 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v1692 = stablehlo.reshape %v1691 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1693 = stablehlo.reshape %v717 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1694 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1695 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v1696 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1697 = stablehlo.reduce(%v1693 init: %v1694) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1698 = stablehlo.broadcast_in_dim %v1697, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1699 = stablehlo.divide %v1698, %v1695 : tensor<32x256x14x14xf32>
    %v1700 = stablehlo.subtract %v1693, %v1699 : tensor<32x256x14x14xf32>
    %v1701 = stablehlo.multiply %v1700, %v1700 : tensor<32x256x14x14xf32>
    %v1702 = stablehlo.reduce(%v1701 init: %v1694) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1703 = stablehlo.broadcast_in_dim %v1702, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1704 = stablehlo.divide %v1703, %v1695 : tensor<32x256x14x14xf32>
    %v1705 = stablehlo.add %v1704, %v1696 : tensor<32x256x14x14xf32>
    %v1706 = stablehlo.rsqrt %v1705 : tensor<32x256x14x14xf32>
    %v1707 = stablehlo.multiply %v1700, %v1706 : tensor<32x256x14x14xf32>
    %v1708 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1709 = stablehlo.multiply %v1708, %v1692 : tensor<32x256x14x14xf32>
    %v1710 = stablehlo.reduce(%v1709 init: %v1694) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1711 = stablehlo.broadcast_in_dim %v1710, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1712 = stablehlo.multiply %v1707, %v1709 : tensor<32x256x14x14xf32>
    %v1713 = stablehlo.reduce(%v1712 init: %v1694) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1714 = stablehlo.broadcast_in_dim %v1713, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1715 = stablehlo.multiply %v1709, %v1695 : tensor<32x256x14x14xf32>
    %v1716 = stablehlo.subtract %v1715, %v1711 : tensor<32x256x14x14xf32>
    %v1717 = stablehlo.multiply %v1707, %v1714 : tensor<32x256x14x14xf32>
    %v1718 = stablehlo.subtract %v1716, %v1717 : tensor<32x256x14x14xf32>
    %v1719 = stablehlo.divide %v1706, %v1695 : tensor<32x256x14x14xf32>
    %v1720 = stablehlo.multiply %v1719, %v1718 : tensor<32x256x14x14xf32>
    %v1721 = stablehlo.reshape %v1720 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1722 = stablehlo.reshape %v1721 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1723 = stablehlo.transpose %s3b3W2, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1724 = stablehlo.reverse %v1723, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1725 = stablehlo.convolution(%v1722, %v1724)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1726 = stablehlo.reshape %v1725 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1727 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1728 = stablehlo.compare GT, %v710, %v1727 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v1729 = stablehlo.select %v1728, %v1726, %v1727 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v1730 = stablehlo.reshape %v1729 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1731 = stablehlo.reshape %v690 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1732 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1733 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v1734 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1735 = stablehlo.reduce(%v1731 init: %v1732) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1736 = stablehlo.broadcast_in_dim %v1735, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1737 = stablehlo.divide %v1736, %v1733 : tensor<32x256x14x14xf32>
    %v1738 = stablehlo.subtract %v1731, %v1737 : tensor<32x256x14x14xf32>
    %v1739 = stablehlo.multiply %v1738, %v1738 : tensor<32x256x14x14xf32>
    %v1740 = stablehlo.reduce(%v1739 init: %v1732) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1741 = stablehlo.broadcast_in_dim %v1740, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1742 = stablehlo.divide %v1741, %v1733 : tensor<32x256x14x14xf32>
    %v1743 = stablehlo.add %v1742, %v1734 : tensor<32x256x14x14xf32>
    %v1744 = stablehlo.rsqrt %v1743 : tensor<32x256x14x14xf32>
    %v1745 = stablehlo.multiply %v1738, %v1744 : tensor<32x256x14x14xf32>
    %v1746 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1747 = stablehlo.multiply %v1746, %v1730 : tensor<32x256x14x14xf32>
    %v1748 = stablehlo.reduce(%v1747 init: %v1732) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1749 = stablehlo.broadcast_in_dim %v1748, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1750 = stablehlo.multiply %v1745, %v1747 : tensor<32x256x14x14xf32>
    %v1751 = stablehlo.reduce(%v1750 init: %v1732) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1752 = stablehlo.broadcast_in_dim %v1751, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1753 = stablehlo.multiply %v1747, %v1733 : tensor<32x256x14x14xf32>
    %v1754 = stablehlo.subtract %v1753, %v1749 : tensor<32x256x14x14xf32>
    %v1755 = stablehlo.multiply %v1745, %v1752 : tensor<32x256x14x14xf32>
    %v1756 = stablehlo.subtract %v1754, %v1755 : tensor<32x256x14x14xf32>
    %v1757 = stablehlo.divide %v1744, %v1733 : tensor<32x256x14x14xf32>
    %v1758 = stablehlo.multiply %v1757, %v1756 : tensor<32x256x14x14xf32>
    %v1759 = stablehlo.reshape %v1758 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1760 = stablehlo.reshape %v1759 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1761 = stablehlo.transpose %s3b3W1, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1762 = stablehlo.reverse %v1761, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1763 = stablehlo.convolution(%v1760, %v1762)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1764 = stablehlo.reshape %v1763 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1765 = stablehlo.add %v1764, %v1691 : tensor<32x50176xf32>
    %v1766 = stablehlo.reshape %v685 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1767 = stablehlo.reshape %v1759 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1768 = stablehlo.transpose %v1766, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1769 = stablehlo.transpose %v1767, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1770 = stablehlo.convolution(%v1768, %v1769)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1771 = stablehlo.transpose %v1770, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1772 = stablehlo.constant dense<0.003125> : tensor<256x256x3x3xf32>
    %v1773 = stablehlo.multiply %v1771, %v1772 : tensor<256x256x3x3xf32>
    %v1774 = stablehlo.subtract %s3b3W1, %v1773 : tensor<256x256x3x3xf32>
    %v1775 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1776 = stablehlo.reshape %v690 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1777 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v1778 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1779 = stablehlo.reduce(%v1776 init: %v1775) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1780 = stablehlo.broadcast_in_dim %v1779, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1781 = stablehlo.divide %v1780, %v1777 : tensor<32x256x14x14xf32>
    %v1782 = stablehlo.subtract %v1776, %v1781 : tensor<32x256x14x14xf32>
    %v1783 = stablehlo.multiply %v1782, %v1782 : tensor<32x256x14x14xf32>
    %v1784 = stablehlo.reduce(%v1783 init: %v1775) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1785 = stablehlo.broadcast_in_dim %v1784, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1786 = stablehlo.divide %v1785, %v1777 : tensor<32x256x14x14xf32>
    %v1787 = stablehlo.add %v1786, %v1778 : tensor<32x256x14x14xf32>
    %v1788 = stablehlo.rsqrt %v1787 : tensor<32x256x14x14xf32>
    %v1789 = stablehlo.multiply %v1782, %v1788 : tensor<32x256x14x14xf32>
    %v1790 = stablehlo.reshape %v1729 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1791 = stablehlo.multiply %v1790, %v1789 : tensor<32x256x14x14xf32>
    %v1792 = stablehlo.reduce(%v1791 init: %v1775) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1793 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v1794 = stablehlo.multiply %v1792, %v1793 : tensor<256xf32>
    %v1795 = stablehlo.subtract %s3b3g1, %v1794 : tensor<256xf32>
    %v1796 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1797 = stablehlo.reshape %v1729 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1798 = stablehlo.reduce(%v1797 init: %v1796) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1799 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v1800 = stablehlo.multiply %v1798, %v1799 : tensor<256xf32>
    %v1801 = stablehlo.subtract %s3b3bt1, %v1800 : tensor<256xf32>
    %v1802 = stablehlo.reshape %v712 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1803 = stablehlo.reshape %v1721 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1804 = stablehlo.transpose %v1802, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1805 = stablehlo.transpose %v1803, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1806 = stablehlo.convolution(%v1804, %v1805)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1807 = stablehlo.transpose %v1806, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1808 = stablehlo.constant dense<0.003125> : tensor<256x256x3x3xf32>
    %v1809 = stablehlo.multiply %v1807, %v1808 : tensor<256x256x3x3xf32>
    %v1810 = stablehlo.subtract %s3b3W2, %v1809 : tensor<256x256x3x3xf32>
    %v1811 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1812 = stablehlo.reshape %v717 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1813 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v1814 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1815 = stablehlo.reduce(%v1812 init: %v1811) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1816 = stablehlo.broadcast_in_dim %v1815, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1817 = stablehlo.divide %v1816, %v1813 : tensor<32x256x14x14xf32>
    %v1818 = stablehlo.subtract %v1812, %v1817 : tensor<32x256x14x14xf32>
    %v1819 = stablehlo.multiply %v1818, %v1818 : tensor<32x256x14x14xf32>
    %v1820 = stablehlo.reduce(%v1819 init: %v1811) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1821 = stablehlo.broadcast_in_dim %v1820, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1822 = stablehlo.divide %v1821, %v1813 : tensor<32x256x14x14xf32>
    %v1823 = stablehlo.add %v1822, %v1814 : tensor<32x256x14x14xf32>
    %v1824 = stablehlo.rsqrt %v1823 : tensor<32x256x14x14xf32>
    %v1825 = stablehlo.multiply %v1818, %v1824 : tensor<32x256x14x14xf32>
    %v1826 = stablehlo.reshape %v1691 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1827 = stablehlo.multiply %v1826, %v1825 : tensor<32x256x14x14xf32>
    %v1828 = stablehlo.reduce(%v1827 init: %v1811) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1829 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v1830 = stablehlo.multiply %v1828, %v1829 : tensor<256xf32>
    %v1831 = stablehlo.subtract %s3b3g2, %v1830 : tensor<256xf32>
    %v1832 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1833 = stablehlo.reshape %v1691 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1834 = stablehlo.reduce(%v1833 init: %v1832) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1835 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v1836 = stablehlo.multiply %v1834, %v1835 : tensor<256xf32>
    %v1837 = stablehlo.subtract %s3b3bt2, %v1836 : tensor<256xf32>
    %v1838 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1839 = stablehlo.compare GT, %v683, %v1838 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v1840 = stablehlo.select %v1839, %v1765, %v1838 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v1841 = stablehlo.reshape %v1840 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1842 = stablehlo.reshape %v662 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1843 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1844 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v1845 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1846 = stablehlo.reduce(%v1842 init: %v1843) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1847 = stablehlo.broadcast_in_dim %v1846, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1848 = stablehlo.divide %v1847, %v1844 : tensor<32x256x14x14xf32>
    %v1849 = stablehlo.subtract %v1842, %v1848 : tensor<32x256x14x14xf32>
    %v1850 = stablehlo.multiply %v1849, %v1849 : tensor<32x256x14x14xf32>
    %v1851 = stablehlo.reduce(%v1850 init: %v1843) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1852 = stablehlo.broadcast_in_dim %v1851, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1853 = stablehlo.divide %v1852, %v1844 : tensor<32x256x14x14xf32>
    %v1854 = stablehlo.add %v1853, %v1845 : tensor<32x256x14x14xf32>
    %v1855 = stablehlo.rsqrt %v1854 : tensor<32x256x14x14xf32>
    %v1856 = stablehlo.multiply %v1849, %v1855 : tensor<32x256x14x14xf32>
    %v1857 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1858 = stablehlo.multiply %v1857, %v1841 : tensor<32x256x14x14xf32>
    %v1859 = stablehlo.reduce(%v1858 init: %v1843) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1860 = stablehlo.broadcast_in_dim %v1859, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1861 = stablehlo.multiply %v1856, %v1858 : tensor<32x256x14x14xf32>
    %v1862 = stablehlo.reduce(%v1861 init: %v1843) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1863 = stablehlo.broadcast_in_dim %v1862, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1864 = stablehlo.multiply %v1858, %v1844 : tensor<32x256x14x14xf32>
    %v1865 = stablehlo.subtract %v1864, %v1860 : tensor<32x256x14x14xf32>
    %v1866 = stablehlo.multiply %v1856, %v1863 : tensor<32x256x14x14xf32>
    %v1867 = stablehlo.subtract %v1865, %v1866 : tensor<32x256x14x14xf32>
    %v1868 = stablehlo.divide %v1855, %v1844 : tensor<32x256x14x14xf32>
    %v1869 = stablehlo.multiply %v1868, %v1867 : tensor<32x256x14x14xf32>
    %v1870 = stablehlo.reshape %v1869 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1871 = stablehlo.reshape %v1870 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1872 = stablehlo.transpose %s3b2W2, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1873 = stablehlo.reverse %v1872, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1874 = stablehlo.convolution(%v1871, %v1873)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1875 = stablehlo.reshape %v1874 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1876 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1877 = stablehlo.compare GT, %v655, %v1876 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v1878 = stablehlo.select %v1877, %v1875, %v1876 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v1879 = stablehlo.reshape %v1878 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1880 = stablehlo.reshape %v635 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1881 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1882 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v1883 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1884 = stablehlo.reduce(%v1880 init: %v1881) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1885 = stablehlo.broadcast_in_dim %v1884, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1886 = stablehlo.divide %v1885, %v1882 : tensor<32x256x14x14xf32>
    %v1887 = stablehlo.subtract %v1880, %v1886 : tensor<32x256x14x14xf32>
    %v1888 = stablehlo.multiply %v1887, %v1887 : tensor<32x256x14x14xf32>
    %v1889 = stablehlo.reduce(%v1888 init: %v1881) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1890 = stablehlo.broadcast_in_dim %v1889, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1891 = stablehlo.divide %v1890, %v1882 : tensor<32x256x14x14xf32>
    %v1892 = stablehlo.add %v1891, %v1883 : tensor<32x256x14x14xf32>
    %v1893 = stablehlo.rsqrt %v1892 : tensor<32x256x14x14xf32>
    %v1894 = stablehlo.multiply %v1887, %v1893 : tensor<32x256x14x14xf32>
    %v1895 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1896 = stablehlo.multiply %v1895, %v1879 : tensor<32x256x14x14xf32>
    %v1897 = stablehlo.reduce(%v1896 init: %v1881) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1898 = stablehlo.broadcast_in_dim %v1897, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1899 = stablehlo.multiply %v1894, %v1896 : tensor<32x256x14x14xf32>
    %v1900 = stablehlo.reduce(%v1899 init: %v1881) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1901 = stablehlo.broadcast_in_dim %v1900, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1902 = stablehlo.multiply %v1896, %v1882 : tensor<32x256x14x14xf32>
    %v1903 = stablehlo.subtract %v1902, %v1898 : tensor<32x256x14x14xf32>
    %v1904 = stablehlo.multiply %v1894, %v1901 : tensor<32x256x14x14xf32>
    %v1905 = stablehlo.subtract %v1903, %v1904 : tensor<32x256x14x14xf32>
    %v1906 = stablehlo.divide %v1893, %v1882 : tensor<32x256x14x14xf32>
    %v1907 = stablehlo.multiply %v1906, %v1905 : tensor<32x256x14x14xf32>
    %v1908 = stablehlo.reshape %v1907 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1909 = stablehlo.reshape %v1908 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1910 = stablehlo.transpose %s3b2W1, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1911 = stablehlo.reverse %v1910, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1912 = stablehlo.convolution(%v1909, %v1911)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1913 = stablehlo.reshape %v1912 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1914 = stablehlo.add %v1913, %v1840 : tensor<32x50176xf32>
    %v1915 = stablehlo.reshape %v630 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1916 = stablehlo.reshape %v1908 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1917 = stablehlo.transpose %v1915, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1918 = stablehlo.transpose %v1916, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1919 = stablehlo.convolution(%v1917, %v1918)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1920 = stablehlo.transpose %v1919, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1921 = stablehlo.constant dense<0.003125> : tensor<256x256x3x3xf32>
    %v1922 = stablehlo.multiply %v1920, %v1921 : tensor<256x256x3x3xf32>
    %v1923 = stablehlo.subtract %s3b2W1, %v1922 : tensor<256x256x3x3xf32>
    %v1924 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1925 = stablehlo.reshape %v635 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1926 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v1927 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1928 = stablehlo.reduce(%v1925 init: %v1924) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1929 = stablehlo.broadcast_in_dim %v1928, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1930 = stablehlo.divide %v1929, %v1926 : tensor<32x256x14x14xf32>
    %v1931 = stablehlo.subtract %v1925, %v1930 : tensor<32x256x14x14xf32>
    %v1932 = stablehlo.multiply %v1931, %v1931 : tensor<32x256x14x14xf32>
    %v1933 = stablehlo.reduce(%v1932 init: %v1924) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1934 = stablehlo.broadcast_in_dim %v1933, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1935 = stablehlo.divide %v1934, %v1926 : tensor<32x256x14x14xf32>
    %v1936 = stablehlo.add %v1935, %v1927 : tensor<32x256x14x14xf32>
    %v1937 = stablehlo.rsqrt %v1936 : tensor<32x256x14x14xf32>
    %v1938 = stablehlo.multiply %v1931, %v1937 : tensor<32x256x14x14xf32>
    %v1939 = stablehlo.reshape %v1878 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1940 = stablehlo.multiply %v1939, %v1938 : tensor<32x256x14x14xf32>
    %v1941 = stablehlo.reduce(%v1940 init: %v1924) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1942 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v1943 = stablehlo.multiply %v1941, %v1942 : tensor<256xf32>
    %v1944 = stablehlo.subtract %s3b2g1, %v1943 : tensor<256xf32>
    %v1945 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1946 = stablehlo.reshape %v1878 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1947 = stablehlo.reduce(%v1946 init: %v1945) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1948 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v1949 = stablehlo.multiply %v1947, %v1948 : tensor<256xf32>
    %v1950 = stablehlo.subtract %s3b2bt1, %v1949 : tensor<256xf32>
    %v1951 = stablehlo.reshape %v657 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1952 = stablehlo.reshape %v1870 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1953 = stablehlo.transpose %v1951, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1954 = stablehlo.transpose %v1952, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1955 = stablehlo.convolution(%v1953, %v1954)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1956 = stablehlo.transpose %v1955, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1957 = stablehlo.constant dense<0.003125> : tensor<256x256x3x3xf32>
    %v1958 = stablehlo.multiply %v1956, %v1957 : tensor<256x256x3x3xf32>
    %v1959 = stablehlo.subtract %s3b2W2, %v1958 : tensor<256x256x3x3xf32>
    %v1960 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1961 = stablehlo.reshape %v662 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1962 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v1963 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1964 = stablehlo.reduce(%v1961 init: %v1960) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1965 = stablehlo.broadcast_in_dim %v1964, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1966 = stablehlo.divide %v1965, %v1962 : tensor<32x256x14x14xf32>
    %v1967 = stablehlo.subtract %v1961, %v1966 : tensor<32x256x14x14xf32>
    %v1968 = stablehlo.multiply %v1967, %v1967 : tensor<32x256x14x14xf32>
    %v1969 = stablehlo.reduce(%v1968 init: %v1960) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1970 = stablehlo.broadcast_in_dim %v1969, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1971 = stablehlo.divide %v1970, %v1962 : tensor<32x256x14x14xf32>
    %v1972 = stablehlo.add %v1971, %v1963 : tensor<32x256x14x14xf32>
    %v1973 = stablehlo.rsqrt %v1972 : tensor<32x256x14x14xf32>
    %v1974 = stablehlo.multiply %v1967, %v1973 : tensor<32x256x14x14xf32>
    %v1975 = stablehlo.reshape %v1840 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1976 = stablehlo.multiply %v1975, %v1974 : tensor<32x256x14x14xf32>
    %v1977 = stablehlo.reduce(%v1976 init: %v1960) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1978 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v1979 = stablehlo.multiply %v1977, %v1978 : tensor<256xf32>
    %v1980 = stablehlo.subtract %s3b2g2, %v1979 : tensor<256xf32>
    %v1981 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1982 = stablehlo.reshape %v1840 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1983 = stablehlo.reduce(%v1982 init: %v1981) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1984 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v1985 = stablehlo.multiply %v1983, %v1984 : tensor<256xf32>
    %v1986 = stablehlo.subtract %s3b2bt2, %v1985 : tensor<256xf32>
    %v1987 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1988 = stablehlo.compare GT, %v628, %v1987 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v1989 = stablehlo.select %v1988, %v1914, %v1987 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v1990 = stablehlo.reshape %v1989 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1991 = stablehlo.reshape %v607 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1992 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1993 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v1994 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1995 = stablehlo.reduce(%v1991 init: %v1992) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1996 = stablehlo.broadcast_in_dim %v1995, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1997 = stablehlo.divide %v1996, %v1993 : tensor<32x256x14x14xf32>
    %v1998 = stablehlo.subtract %v1991, %v1997 : tensor<32x256x14x14xf32>
    %v1999 = stablehlo.multiply %v1998, %v1998 : tensor<32x256x14x14xf32>
    %v2000 = stablehlo.reduce(%v1999 init: %v1992) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2001 = stablehlo.broadcast_in_dim %v2000, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2002 = stablehlo.divide %v2001, %v1993 : tensor<32x256x14x14xf32>
    %v2003 = stablehlo.add %v2002, %v1994 : tensor<32x256x14x14xf32>
    %v2004 = stablehlo.rsqrt %v2003 : tensor<32x256x14x14xf32>
    %v2005 = stablehlo.multiply %v1998, %v2004 : tensor<32x256x14x14xf32>
    %v2006 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2007 = stablehlo.multiply %v2006, %v1990 : tensor<32x256x14x14xf32>
    %v2008 = stablehlo.reduce(%v2007 init: %v1992) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2009 = stablehlo.broadcast_in_dim %v2008, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2010 = stablehlo.multiply %v2005, %v2007 : tensor<32x256x14x14xf32>
    %v2011 = stablehlo.reduce(%v2010 init: %v1992) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2012 = stablehlo.broadcast_in_dim %v2011, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2013 = stablehlo.multiply %v2007, %v1993 : tensor<32x256x14x14xf32>
    %v2014 = stablehlo.subtract %v2013, %v2009 : tensor<32x256x14x14xf32>
    %v2015 = stablehlo.multiply %v2005, %v2012 : tensor<32x256x14x14xf32>
    %v2016 = stablehlo.subtract %v2014, %v2015 : tensor<32x256x14x14xf32>
    %v2017 = stablehlo.divide %v2004, %v1993 : tensor<32x256x14x14xf32>
    %v2018 = stablehlo.multiply %v2017, %v2016 : tensor<32x256x14x14xf32>
    %v2019 = stablehlo.reshape %v2018 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2020 = stablehlo.reshape %v2019 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2021 = stablehlo.transpose %s3b1W2, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2022 = stablehlo.reverse %v2021, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2023 = stablehlo.convolution(%v2020, %v2022)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2024 = stablehlo.reshape %v2023 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2025 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v2026 = stablehlo.compare GT, %v600, %v2025 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v2027 = stablehlo.select %v2026, %v2024, %v2025 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v2028 = stablehlo.reshape %v2027 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2029 = stablehlo.reshape %v580 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2030 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2031 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2032 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2033 = stablehlo.reduce(%v2029 init: %v2030) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2034 = stablehlo.broadcast_in_dim %v2033, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2035 = stablehlo.divide %v2034, %v2031 : tensor<32x256x14x14xf32>
    %v2036 = stablehlo.subtract %v2029, %v2035 : tensor<32x256x14x14xf32>
    %v2037 = stablehlo.multiply %v2036, %v2036 : tensor<32x256x14x14xf32>
    %v2038 = stablehlo.reduce(%v2037 init: %v2030) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2039 = stablehlo.broadcast_in_dim %v2038, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2040 = stablehlo.divide %v2039, %v2031 : tensor<32x256x14x14xf32>
    %v2041 = stablehlo.add %v2040, %v2032 : tensor<32x256x14x14xf32>
    %v2042 = stablehlo.rsqrt %v2041 : tensor<32x256x14x14xf32>
    %v2043 = stablehlo.multiply %v2036, %v2042 : tensor<32x256x14x14xf32>
    %v2044 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2045 = stablehlo.multiply %v2044, %v2028 : tensor<32x256x14x14xf32>
    %v2046 = stablehlo.reduce(%v2045 init: %v2030) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2047 = stablehlo.broadcast_in_dim %v2046, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2048 = stablehlo.multiply %v2043, %v2045 : tensor<32x256x14x14xf32>
    %v2049 = stablehlo.reduce(%v2048 init: %v2030) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2050 = stablehlo.broadcast_in_dim %v2049, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2051 = stablehlo.multiply %v2045, %v2031 : tensor<32x256x14x14xf32>
    %v2052 = stablehlo.subtract %v2051, %v2047 : tensor<32x256x14x14xf32>
    %v2053 = stablehlo.multiply %v2043, %v2050 : tensor<32x256x14x14xf32>
    %v2054 = stablehlo.subtract %v2052, %v2053 : tensor<32x256x14x14xf32>
    %v2055 = stablehlo.divide %v2042, %v2031 : tensor<32x256x14x14xf32>
    %v2056 = stablehlo.multiply %v2055, %v2054 : tensor<32x256x14x14xf32>
    %v2057 = stablehlo.reshape %v2056 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2058 = stablehlo.reshape %v2057 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2059 = stablehlo.transpose %s3b1W1, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2060 = stablehlo.reverse %v2059, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2061 = stablehlo.convolution(%v2058, %v2060)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2062 = stablehlo.reshape %v2061 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2063 = stablehlo.add %v2062, %v1989 : tensor<32x50176xf32>
    %v2064 = stablehlo.reshape %v575 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2065 = stablehlo.reshape %v2057 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2066 = stablehlo.transpose %v2064, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2067 = stablehlo.transpose %v2065, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2068 = stablehlo.convolution(%v2066, %v2067)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2069 = stablehlo.transpose %v2068, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2070 = stablehlo.constant dense<0.003125> : tensor<256x256x3x3xf32>
    %v2071 = stablehlo.multiply %v2069, %v2070 : tensor<256x256x3x3xf32>
    %v2072 = stablehlo.subtract %s3b1W1, %v2071 : tensor<256x256x3x3xf32>
    %v2073 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2074 = stablehlo.reshape %v580 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2075 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2076 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2077 = stablehlo.reduce(%v2074 init: %v2073) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2078 = stablehlo.broadcast_in_dim %v2077, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2079 = stablehlo.divide %v2078, %v2075 : tensor<32x256x14x14xf32>
    %v2080 = stablehlo.subtract %v2074, %v2079 : tensor<32x256x14x14xf32>
    %v2081 = stablehlo.multiply %v2080, %v2080 : tensor<32x256x14x14xf32>
    %v2082 = stablehlo.reduce(%v2081 init: %v2073) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2083 = stablehlo.broadcast_in_dim %v2082, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2084 = stablehlo.divide %v2083, %v2075 : tensor<32x256x14x14xf32>
    %v2085 = stablehlo.add %v2084, %v2076 : tensor<32x256x14x14xf32>
    %v2086 = stablehlo.rsqrt %v2085 : tensor<32x256x14x14xf32>
    %v2087 = stablehlo.multiply %v2080, %v2086 : tensor<32x256x14x14xf32>
    %v2088 = stablehlo.reshape %v2027 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2089 = stablehlo.multiply %v2088, %v2087 : tensor<32x256x14x14xf32>
    %v2090 = stablehlo.reduce(%v2089 init: %v2073) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2091 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2092 = stablehlo.multiply %v2090, %v2091 : tensor<256xf32>
    %v2093 = stablehlo.subtract %s3b1g1, %v2092 : tensor<256xf32>
    %v2094 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2095 = stablehlo.reshape %v2027 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2096 = stablehlo.reduce(%v2095 init: %v2094) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2097 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2098 = stablehlo.multiply %v2096, %v2097 : tensor<256xf32>
    %v2099 = stablehlo.subtract %s3b1bt1, %v2098 : tensor<256xf32>
    %v2100 = stablehlo.reshape %v602 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2101 = stablehlo.reshape %v2019 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2102 = stablehlo.transpose %v2100, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2103 = stablehlo.transpose %v2101, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2104 = stablehlo.convolution(%v2102, %v2103)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2105 = stablehlo.transpose %v2104, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2106 = stablehlo.constant dense<0.003125> : tensor<256x256x3x3xf32>
    %v2107 = stablehlo.multiply %v2105, %v2106 : tensor<256x256x3x3xf32>
    %v2108 = stablehlo.subtract %s3b1W2, %v2107 : tensor<256x256x3x3xf32>
    %v2109 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2110 = stablehlo.reshape %v607 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2111 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2112 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2113 = stablehlo.reduce(%v2110 init: %v2109) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2114 = stablehlo.broadcast_in_dim %v2113, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2115 = stablehlo.divide %v2114, %v2111 : tensor<32x256x14x14xf32>
    %v2116 = stablehlo.subtract %v2110, %v2115 : tensor<32x256x14x14xf32>
    %v2117 = stablehlo.multiply %v2116, %v2116 : tensor<32x256x14x14xf32>
    %v2118 = stablehlo.reduce(%v2117 init: %v2109) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2119 = stablehlo.broadcast_in_dim %v2118, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2120 = stablehlo.divide %v2119, %v2111 : tensor<32x256x14x14xf32>
    %v2121 = stablehlo.add %v2120, %v2112 : tensor<32x256x14x14xf32>
    %v2122 = stablehlo.rsqrt %v2121 : tensor<32x256x14x14xf32>
    %v2123 = stablehlo.multiply %v2116, %v2122 : tensor<32x256x14x14xf32>
    %v2124 = stablehlo.reshape %v1989 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2125 = stablehlo.multiply %v2124, %v2123 : tensor<32x256x14x14xf32>
    %v2126 = stablehlo.reduce(%v2125 init: %v2109) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2127 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2128 = stablehlo.multiply %v2126, %v2127 : tensor<256xf32>
    %v2129 = stablehlo.subtract %s3b1g2, %v2128 : tensor<256xf32>
    %v2130 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2131 = stablehlo.reshape %v1989 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2132 = stablehlo.reduce(%v2131 init: %v2130) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2133 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2134 = stablehlo.multiply %v2132, %v2133 : tensor<256xf32>
    %v2135 = stablehlo.subtract %s3b1bt2, %v2134 : tensor<256xf32>
    %v2136 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v2137 = stablehlo.compare GT, %v573, %v2136 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v2138 = stablehlo.select %v2137, %v2063, %v2136 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v2139 = stablehlo.reshape %v2138 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2140 = stablehlo.reshape %v552 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2141 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2142 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2143 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2144 = stablehlo.reduce(%v2140 init: %v2141) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2145 = stablehlo.broadcast_in_dim %v2144, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2146 = stablehlo.divide %v2145, %v2142 : tensor<32x256x14x14xf32>
    %v2147 = stablehlo.subtract %v2140, %v2146 : tensor<32x256x14x14xf32>
    %v2148 = stablehlo.multiply %v2147, %v2147 : tensor<32x256x14x14xf32>
    %v2149 = stablehlo.reduce(%v2148 init: %v2141) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2150 = stablehlo.broadcast_in_dim %v2149, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2151 = stablehlo.divide %v2150, %v2142 : tensor<32x256x14x14xf32>
    %v2152 = stablehlo.add %v2151, %v2143 : tensor<32x256x14x14xf32>
    %v2153 = stablehlo.rsqrt %v2152 : tensor<32x256x14x14xf32>
    %v2154 = stablehlo.multiply %v2147, %v2153 : tensor<32x256x14x14xf32>
    %v2155 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2156 = stablehlo.multiply %v2155, %v2139 : tensor<32x256x14x14xf32>
    %v2157 = stablehlo.reduce(%v2156 init: %v2141) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2158 = stablehlo.broadcast_in_dim %v2157, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2159 = stablehlo.multiply %v2154, %v2156 : tensor<32x256x14x14xf32>
    %v2160 = stablehlo.reduce(%v2159 init: %v2141) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2161 = stablehlo.broadcast_in_dim %v2160, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2162 = stablehlo.multiply %v2156, %v2142 : tensor<32x256x14x14xf32>
    %v2163 = stablehlo.subtract %v2162, %v2158 : tensor<32x256x14x14xf32>
    %v2164 = stablehlo.multiply %v2154, %v2161 : tensor<32x256x14x14xf32>
    %v2165 = stablehlo.subtract %v2163, %v2164 : tensor<32x256x14x14xf32>
    %v2166 = stablehlo.divide %v2153, %v2142 : tensor<32x256x14x14xf32>
    %v2167 = stablehlo.multiply %v2166, %v2165 : tensor<32x256x14x14xf32>
    %v2168 = stablehlo.reshape %v2167 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2169 = stablehlo.reshape %v2168 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2170 = stablehlo.transpose %s3b0W2, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2171 = stablehlo.reverse %v2170, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2172 = stablehlo.convolution(%v2169, %v2171)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2173 = stablehlo.reshape %v2172 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2174 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v2175 = stablehlo.compare GT, %v545, %v2174 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v2176 = stablehlo.select %v2175, %v2173, %v2174 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v2177 = stablehlo.reshape %v2176 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2178 = stablehlo.reshape %v525 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2179 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2180 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2181 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2182 = stablehlo.reduce(%v2178 init: %v2179) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2183 = stablehlo.broadcast_in_dim %v2182, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2184 = stablehlo.divide %v2183, %v2180 : tensor<32x256x14x14xf32>
    %v2185 = stablehlo.subtract %v2178, %v2184 : tensor<32x256x14x14xf32>
    %v2186 = stablehlo.multiply %v2185, %v2185 : tensor<32x256x14x14xf32>
    %v2187 = stablehlo.reduce(%v2186 init: %v2179) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2188 = stablehlo.broadcast_in_dim %v2187, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2189 = stablehlo.divide %v2188, %v2180 : tensor<32x256x14x14xf32>
    %v2190 = stablehlo.add %v2189, %v2181 : tensor<32x256x14x14xf32>
    %v2191 = stablehlo.rsqrt %v2190 : tensor<32x256x14x14xf32>
    %v2192 = stablehlo.multiply %v2185, %v2191 : tensor<32x256x14x14xf32>
    %v2193 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2194 = stablehlo.multiply %v2193, %v2177 : tensor<32x256x14x14xf32>
    %v2195 = stablehlo.reduce(%v2194 init: %v2179) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2196 = stablehlo.broadcast_in_dim %v2195, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2197 = stablehlo.multiply %v2192, %v2194 : tensor<32x256x14x14xf32>
    %v2198 = stablehlo.reduce(%v2197 init: %v2179) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2199 = stablehlo.broadcast_in_dim %v2198, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2200 = stablehlo.multiply %v2194, %v2180 : tensor<32x256x14x14xf32>
    %v2201 = stablehlo.subtract %v2200, %v2196 : tensor<32x256x14x14xf32>
    %v2202 = stablehlo.multiply %v2192, %v2199 : tensor<32x256x14x14xf32>
    %v2203 = stablehlo.subtract %v2201, %v2202 : tensor<32x256x14x14xf32>
    %v2204 = stablehlo.divide %v2191, %v2180 : tensor<32x256x14x14xf32>
    %v2205 = stablehlo.multiply %v2204, %v2203 : tensor<32x256x14x14xf32>
    %v2206 = stablehlo.reshape %v2205 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2207 = stablehlo.reshape %v2206 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2208 = stablehlo.transpose %s3b0W1, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2209 = stablehlo.reverse %v2208, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2210 = stablehlo.convolution(%v2207, %v2209)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2211 = stablehlo.reshape %v2210 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2212 = stablehlo.add %v2211, %v2138 : tensor<32x50176xf32>
    %v2213 = stablehlo.reshape %v520 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2214 = stablehlo.reshape %v2206 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2215 = stablehlo.transpose %v2213, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2216 = stablehlo.transpose %v2214, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2217 = stablehlo.convolution(%v2215, %v2216)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2218 = stablehlo.transpose %v2217, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2219 = stablehlo.constant dense<0.003125> : tensor<256x256x3x3xf32>
    %v2220 = stablehlo.multiply %v2218, %v2219 : tensor<256x256x3x3xf32>
    %v2221 = stablehlo.subtract %s3b0W1, %v2220 : tensor<256x256x3x3xf32>
    %v2222 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2223 = stablehlo.reshape %v525 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2224 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2225 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2226 = stablehlo.reduce(%v2223 init: %v2222) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2227 = stablehlo.broadcast_in_dim %v2226, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2228 = stablehlo.divide %v2227, %v2224 : tensor<32x256x14x14xf32>
    %v2229 = stablehlo.subtract %v2223, %v2228 : tensor<32x256x14x14xf32>
    %v2230 = stablehlo.multiply %v2229, %v2229 : tensor<32x256x14x14xf32>
    %v2231 = stablehlo.reduce(%v2230 init: %v2222) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2232 = stablehlo.broadcast_in_dim %v2231, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2233 = stablehlo.divide %v2232, %v2224 : tensor<32x256x14x14xf32>
    %v2234 = stablehlo.add %v2233, %v2225 : tensor<32x256x14x14xf32>
    %v2235 = stablehlo.rsqrt %v2234 : tensor<32x256x14x14xf32>
    %v2236 = stablehlo.multiply %v2229, %v2235 : tensor<32x256x14x14xf32>
    %v2237 = stablehlo.reshape %v2176 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2238 = stablehlo.multiply %v2237, %v2236 : tensor<32x256x14x14xf32>
    %v2239 = stablehlo.reduce(%v2238 init: %v2222) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2240 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2241 = stablehlo.multiply %v2239, %v2240 : tensor<256xf32>
    %v2242 = stablehlo.subtract %s3b0g1, %v2241 : tensor<256xf32>
    %v2243 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2244 = stablehlo.reshape %v2176 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2245 = stablehlo.reduce(%v2244 init: %v2243) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2246 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2247 = stablehlo.multiply %v2245, %v2246 : tensor<256xf32>
    %v2248 = stablehlo.subtract %s3b0bt1, %v2247 : tensor<256xf32>
    %v2249 = stablehlo.reshape %v547 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2250 = stablehlo.reshape %v2168 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2251 = stablehlo.transpose %v2249, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2252 = stablehlo.transpose %v2250, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2253 = stablehlo.convolution(%v2251, %v2252)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2254 = stablehlo.transpose %v2253, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2255 = stablehlo.constant dense<0.003125> : tensor<256x256x3x3xf32>
    %v2256 = stablehlo.multiply %v2254, %v2255 : tensor<256x256x3x3xf32>
    %v2257 = stablehlo.subtract %s3b0W2, %v2256 : tensor<256x256x3x3xf32>
    %v2258 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2259 = stablehlo.reshape %v552 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2260 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2261 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2262 = stablehlo.reduce(%v2259 init: %v2258) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2263 = stablehlo.broadcast_in_dim %v2262, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2264 = stablehlo.divide %v2263, %v2260 : tensor<32x256x14x14xf32>
    %v2265 = stablehlo.subtract %v2259, %v2264 : tensor<32x256x14x14xf32>
    %v2266 = stablehlo.multiply %v2265, %v2265 : tensor<32x256x14x14xf32>
    %v2267 = stablehlo.reduce(%v2266 init: %v2258) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2268 = stablehlo.broadcast_in_dim %v2267, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2269 = stablehlo.divide %v2268, %v2260 : tensor<32x256x14x14xf32>
    %v2270 = stablehlo.add %v2269, %v2261 : tensor<32x256x14x14xf32>
    %v2271 = stablehlo.rsqrt %v2270 : tensor<32x256x14x14xf32>
    %v2272 = stablehlo.multiply %v2265, %v2271 : tensor<32x256x14x14xf32>
    %v2273 = stablehlo.reshape %v2138 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2274 = stablehlo.multiply %v2273, %v2272 : tensor<32x256x14x14xf32>
    %v2275 = stablehlo.reduce(%v2274 init: %v2258) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2276 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2277 = stablehlo.multiply %v2275, %v2276 : tensor<256xf32>
    %v2278 = stablehlo.subtract %s3b0g2, %v2277 : tensor<256xf32>
    %v2279 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2280 = stablehlo.reshape %v2138 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2281 = stablehlo.reduce(%v2280 init: %v2279) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2282 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2283 = stablehlo.multiply %v2281, %v2282 : tensor<256xf32>
    %v2284 = stablehlo.subtract %s3b0bt2, %v2283 : tensor<256xf32>
    %v2285 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v2286 = stablehlo.compare GT, %v518, %v2285 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v2287 = stablehlo.select %v2286, %v2212, %v2285 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v2288 = stablehlo.reshape %v2287 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2289 = stablehlo.reshape %v472 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2290 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2291 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2292 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2293 = stablehlo.reduce(%v2289 init: %v2290) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2294 = stablehlo.broadcast_in_dim %v2293, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2295 = stablehlo.divide %v2294, %v2291 : tensor<32x256x14x14xf32>
    %v2296 = stablehlo.subtract %v2289, %v2295 : tensor<32x256x14x14xf32>
    %v2297 = stablehlo.multiply %v2296, %v2296 : tensor<32x256x14x14xf32>
    %v2298 = stablehlo.reduce(%v2297 init: %v2290) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2299 = stablehlo.broadcast_in_dim %v2298, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2300 = stablehlo.divide %v2299, %v2291 : tensor<32x256x14x14xf32>
    %v2301 = stablehlo.add %v2300, %v2292 : tensor<32x256x14x14xf32>
    %v2302 = stablehlo.rsqrt %v2301 : tensor<32x256x14x14xf32>
    %v2303 = stablehlo.multiply %v2296, %v2302 : tensor<32x256x14x14xf32>
    %v2304 = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2305 = stablehlo.multiply %v2304, %v2288 : tensor<32x256x14x14xf32>
    %v2306 = stablehlo.reduce(%v2305 init: %v2290) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2307 = stablehlo.broadcast_in_dim %v2306, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2308 = stablehlo.multiply %v2303, %v2305 : tensor<32x256x14x14xf32>
    %v2309 = stablehlo.reduce(%v2308 init: %v2290) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2310 = stablehlo.broadcast_in_dim %v2309, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2311 = stablehlo.multiply %v2305, %v2291 : tensor<32x256x14x14xf32>
    %v2312 = stablehlo.subtract %v2311, %v2307 : tensor<32x256x14x14xf32>
    %v2313 = stablehlo.multiply %v2303, %v2310 : tensor<32x256x14x14xf32>
    %v2314 = stablehlo.subtract %v2312, %v2313 : tensor<32x256x14x14xf32>
    %v2315 = stablehlo.divide %v2302, %v2291 : tensor<32x256x14x14xf32>
    %v2316 = stablehlo.multiply %v2315, %v2314 : tensor<32x256x14x14xf32>
    %v2317 = stablehlo.reshape %v2316 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2318 = stablehlo.reshape %v2317 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2319 = stablehlo.transpose %d3W2, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2320 = stablehlo.reverse %v2319, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2321 = stablehlo.convolution(%v2318, %v2320)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2322 = stablehlo.reshape %v2321 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2323 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v2324 = stablehlo.compare GT, %v465, %v2323 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v2325 = stablehlo.select %v2324, %v2322, %v2323 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v2326 = stablehlo.reshape %v2325 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2327 = stablehlo.reshape %v445 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2328 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2329 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2330 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2331 = stablehlo.reduce(%v2327 init: %v2328) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2332 = stablehlo.broadcast_in_dim %v2331, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2333 = stablehlo.divide %v2332, %v2329 : tensor<32x256x14x14xf32>
    %v2334 = stablehlo.subtract %v2327, %v2333 : tensor<32x256x14x14xf32>
    %v2335 = stablehlo.multiply %v2334, %v2334 : tensor<32x256x14x14xf32>
    %v2336 = stablehlo.reduce(%v2335 init: %v2328) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2337 = stablehlo.broadcast_in_dim %v2336, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2338 = stablehlo.divide %v2337, %v2329 : tensor<32x256x14x14xf32>
    %v2339 = stablehlo.add %v2338, %v2330 : tensor<32x256x14x14xf32>
    %v2340 = stablehlo.rsqrt %v2339 : tensor<32x256x14x14xf32>
    %v2341 = stablehlo.multiply %v2334, %v2340 : tensor<32x256x14x14xf32>
    %v2342 = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2343 = stablehlo.multiply %v2342, %v2326 : tensor<32x256x14x14xf32>
    %v2344 = stablehlo.reduce(%v2343 init: %v2328) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2345 = stablehlo.broadcast_in_dim %v2344, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2346 = stablehlo.multiply %v2341, %v2343 : tensor<32x256x14x14xf32>
    %v2347 = stablehlo.reduce(%v2346 init: %v2328) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2348 = stablehlo.broadcast_in_dim %v2347, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2349 = stablehlo.multiply %v2343, %v2329 : tensor<32x256x14x14xf32>
    %v2350 = stablehlo.subtract %v2349, %v2345 : tensor<32x256x14x14xf32>
    %v2351 = stablehlo.multiply %v2341, %v2348 : tensor<32x256x14x14xf32>
    %v2352 = stablehlo.subtract %v2350, %v2351 : tensor<32x256x14x14xf32>
    %v2353 = stablehlo.divide %v2340, %v2329 : tensor<32x256x14x14xf32>
    %v2354 = stablehlo.multiply %v2353, %v2352 : tensor<32x256x14x14xf32>
    %v2355 = stablehlo.reshape %v2354 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2356 = stablehlo.reshape %v2355 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2357 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2358 = stablehlo.pad %v2356, %v2357, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %v2359 = stablehlo.transpose %d3W1, dims = [1, 0, 2, 3] : (tensor<256x128x3x3xf32>) -> tensor<128x256x3x3xf32>
    %v2360 = stablehlo.reverse %v2359, dims = [2, 3] : tensor<128x256x3x3xf32>
    %v2361 = stablehlo.convolution(%v2358, %v2360)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x28x28xf32>, tensor<128x256x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2362 = stablehlo.reshape %v2361 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2363 = stablehlo.reshape %v2287 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2364 = stablehlo.reshape %v497 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2365 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2366 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2367 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2368 = stablehlo.reduce(%v2364 init: %v2365) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2369 = stablehlo.broadcast_in_dim %v2368, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2370 = stablehlo.divide %v2369, %v2366 : tensor<32x256x14x14xf32>
    %v2371 = stablehlo.subtract %v2364, %v2370 : tensor<32x256x14x14xf32>
    %v2372 = stablehlo.multiply %v2371, %v2371 : tensor<32x256x14x14xf32>
    %v2373 = stablehlo.reduce(%v2372 init: %v2365) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2374 = stablehlo.broadcast_in_dim %v2373, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2375 = stablehlo.divide %v2374, %v2366 : tensor<32x256x14x14xf32>
    %v2376 = stablehlo.add %v2375, %v2367 : tensor<32x256x14x14xf32>
    %v2377 = stablehlo.rsqrt %v2376 : tensor<32x256x14x14xf32>
    %v2378 = stablehlo.multiply %v2371, %v2377 : tensor<32x256x14x14xf32>
    %v2379 = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2380 = stablehlo.multiply %v2379, %v2363 : tensor<32x256x14x14xf32>
    %v2381 = stablehlo.reduce(%v2380 init: %v2365) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2382 = stablehlo.broadcast_in_dim %v2381, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2383 = stablehlo.multiply %v2378, %v2380 : tensor<32x256x14x14xf32>
    %v2384 = stablehlo.reduce(%v2383 init: %v2365) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2385 = stablehlo.broadcast_in_dim %v2384, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2386 = stablehlo.multiply %v2380, %v2366 : tensor<32x256x14x14xf32>
    %v2387 = stablehlo.subtract %v2386, %v2382 : tensor<32x256x14x14xf32>
    %v2388 = stablehlo.multiply %v2378, %v2385 : tensor<32x256x14x14xf32>
    %v2389 = stablehlo.subtract %v2387, %v2388 : tensor<32x256x14x14xf32>
    %v2390 = stablehlo.divide %v2377, %v2366 : tensor<32x256x14x14xf32>
    %v2391 = stablehlo.multiply %v2390, %v2389 : tensor<32x256x14x14xf32>
    %v2392 = stablehlo.reshape %v2391 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2393 = stablehlo.reshape %v2392 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2394 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2395 = stablehlo.pad %v2393, %v2394, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %v2396 = stablehlo.transpose %d3Wp, dims = [1, 0, 2, 3] : (tensor<256x128x3x3xf32>) -> tensor<128x256x3x3xf32>
    %v2397 = stablehlo.reverse %v2396, dims = [2, 3] : tensor<128x256x3x3xf32>
    %v2398 = stablehlo.convolution(%v2395, %v2397)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x28x28xf32>, tensor<128x256x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2399 = stablehlo.reshape %v2398 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2400 = stablehlo.add %v2362, %v2399 : tensor<32x100352xf32>
    %v2401 = stablehlo.reshape %v440 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2402 = stablehlo.reshape %v2355 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2403 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2404 = stablehlo.pad %v2402, %v2403, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %v2405 = stablehlo.transpose %v2401, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2406 = stablehlo.transpose %v2404, dims = [1, 0, 2, 3] : (tensor<32x256x28x28xf32>) -> tensor<256x32x28x28xf32>
    %v2407 = stablehlo.convolution(%v2405, %v2406)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<256x32x28x28xf32>) -> tensor<128x256x3x3xf32>
    %v2408 = stablehlo.transpose %v2407, dims = [1, 0, 2, 3] : (tensor<128x256x3x3xf32>) -> tensor<256x128x3x3xf32>
    %v2409 = stablehlo.constant dense<0.003125> : tensor<256x128x3x3xf32>
    %v2410 = stablehlo.multiply %v2408, %v2409 : tensor<256x128x3x3xf32>
    %v2411 = stablehlo.subtract %d3W1, %v2410 : tensor<256x128x3x3xf32>
    %v2412 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2413 = stablehlo.reshape %v445 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2414 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2415 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2416 = stablehlo.reduce(%v2413 init: %v2412) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2417 = stablehlo.broadcast_in_dim %v2416, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2418 = stablehlo.divide %v2417, %v2414 : tensor<32x256x14x14xf32>
    %v2419 = stablehlo.subtract %v2413, %v2418 : tensor<32x256x14x14xf32>
    %v2420 = stablehlo.multiply %v2419, %v2419 : tensor<32x256x14x14xf32>
    %v2421 = stablehlo.reduce(%v2420 init: %v2412) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2422 = stablehlo.broadcast_in_dim %v2421, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2423 = stablehlo.divide %v2422, %v2414 : tensor<32x256x14x14xf32>
    %v2424 = stablehlo.add %v2423, %v2415 : tensor<32x256x14x14xf32>
    %v2425 = stablehlo.rsqrt %v2424 : tensor<32x256x14x14xf32>
    %v2426 = stablehlo.multiply %v2419, %v2425 : tensor<32x256x14x14xf32>
    %v2427 = stablehlo.reshape %v2325 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2428 = stablehlo.multiply %v2427, %v2426 : tensor<32x256x14x14xf32>
    %v2429 = stablehlo.reduce(%v2428 init: %v2412) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2430 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2431 = stablehlo.multiply %v2429, %v2430 : tensor<256xf32>
    %v2432 = stablehlo.subtract %d3g1, %v2431 : tensor<256xf32>
    %v2433 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2434 = stablehlo.reshape %v2325 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2435 = stablehlo.reduce(%v2434 init: %v2433) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2436 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2437 = stablehlo.multiply %v2435, %v2436 : tensor<256xf32>
    %v2438 = stablehlo.subtract %d3bt1, %v2437 : tensor<256xf32>
    %v2439 = stablehlo.reshape %v467 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2440 = stablehlo.reshape %v2317 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2441 = stablehlo.transpose %v2439, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2442 = stablehlo.transpose %v2440, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2443 = stablehlo.convolution(%v2441, %v2442)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2444 = stablehlo.transpose %v2443, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2445 = stablehlo.constant dense<0.003125> : tensor<256x256x3x3xf32>
    %v2446 = stablehlo.multiply %v2444, %v2445 : tensor<256x256x3x3xf32>
    %v2447 = stablehlo.subtract %d3W2, %v2446 : tensor<256x256x3x3xf32>
    %v2448 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2449 = stablehlo.reshape %v472 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2450 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2451 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2452 = stablehlo.reduce(%v2449 init: %v2448) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2453 = stablehlo.broadcast_in_dim %v2452, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2454 = stablehlo.divide %v2453, %v2450 : tensor<32x256x14x14xf32>
    %v2455 = stablehlo.subtract %v2449, %v2454 : tensor<32x256x14x14xf32>
    %v2456 = stablehlo.multiply %v2455, %v2455 : tensor<32x256x14x14xf32>
    %v2457 = stablehlo.reduce(%v2456 init: %v2448) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2458 = stablehlo.broadcast_in_dim %v2457, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2459 = stablehlo.divide %v2458, %v2450 : tensor<32x256x14x14xf32>
    %v2460 = stablehlo.add %v2459, %v2451 : tensor<32x256x14x14xf32>
    %v2461 = stablehlo.rsqrt %v2460 : tensor<32x256x14x14xf32>
    %v2462 = stablehlo.multiply %v2455, %v2461 : tensor<32x256x14x14xf32>
    %v2463 = stablehlo.reshape %v2287 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2464 = stablehlo.multiply %v2463, %v2462 : tensor<32x256x14x14xf32>
    %v2465 = stablehlo.reduce(%v2464 init: %v2448) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2466 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2467 = stablehlo.multiply %v2465, %v2466 : tensor<256xf32>
    %v2468 = stablehlo.subtract %d3g2, %v2467 : tensor<256xf32>
    %v2469 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2470 = stablehlo.reshape %v2287 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2471 = stablehlo.reduce(%v2470 init: %v2469) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2472 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2473 = stablehlo.multiply %v2471, %v2472 : tensor<256xf32>
    %v2474 = stablehlo.subtract %d3bt2, %v2473 : tensor<256xf32>
    %v2475 = stablehlo.reshape %v440 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2476 = stablehlo.reshape %v2392 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2477 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2478 = stablehlo.pad %v2476, %v2477, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %v2479 = stablehlo.transpose %v2475, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2480 = stablehlo.transpose %v2478, dims = [1, 0, 2, 3] : (tensor<32x256x28x28xf32>) -> tensor<256x32x28x28xf32>
    %v2481 = stablehlo.convolution(%v2479, %v2480)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<256x32x28x28xf32>) -> tensor<128x256x3x3xf32>
    %v2482 = stablehlo.transpose %v2481, dims = [1, 0, 2, 3] : (tensor<128x256x3x3xf32>) -> tensor<256x128x3x3xf32>
    %v2483 = stablehlo.constant dense<0.003125> : tensor<256x128x3x3xf32>
    %v2484 = stablehlo.multiply %v2482, %v2483 : tensor<256x128x3x3xf32>
    %v2485 = stablehlo.subtract %d3Wp, %v2484 : tensor<256x128x3x3xf32>
    %v2486 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2487 = stablehlo.reshape %v497 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2488 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2489 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2490 = stablehlo.reduce(%v2487 init: %v2486) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2491 = stablehlo.broadcast_in_dim %v2490, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2492 = stablehlo.divide %v2491, %v2488 : tensor<32x256x14x14xf32>
    %v2493 = stablehlo.subtract %v2487, %v2492 : tensor<32x256x14x14xf32>
    %v2494 = stablehlo.multiply %v2493, %v2493 : tensor<32x256x14x14xf32>
    %v2495 = stablehlo.reduce(%v2494 init: %v2486) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2496 = stablehlo.broadcast_in_dim %v2495, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2497 = stablehlo.divide %v2496, %v2488 : tensor<32x256x14x14xf32>
    %v2498 = stablehlo.add %v2497, %v2489 : tensor<32x256x14x14xf32>
    %v2499 = stablehlo.rsqrt %v2498 : tensor<32x256x14x14xf32>
    %v2500 = stablehlo.multiply %v2493, %v2499 : tensor<32x256x14x14xf32>
    %v2501 = stablehlo.reshape %v2287 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2502 = stablehlo.multiply %v2501, %v2500 : tensor<32x256x14x14xf32>
    %v2503 = stablehlo.reduce(%v2502 init: %v2486) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2504 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2505 = stablehlo.multiply %v2503, %v2504 : tensor<256xf32>
    %v2506 = stablehlo.subtract %d3gp, %v2505 : tensor<256xf32>
    %v2507 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2508 = stablehlo.reshape %v2287 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2509 = stablehlo.reduce(%v2508 init: %v2507) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2510 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2511 = stablehlo.multiply %v2509, %v2510 : tensor<256xf32>
    %v2512 = stablehlo.subtract %d3btp, %v2511 : tensor<256xf32>
    %v2513 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v2514 = stablehlo.compare GT, %v438, %v2513 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v2515 = stablehlo.select %v2514, %v2400, %v2513 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v2516 = stablehlo.reshape %v2515 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2517 = stablehlo.reshape %v417 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2518 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2519 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v2520 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2521 = stablehlo.reduce(%v2517 init: %v2518) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2522 = stablehlo.broadcast_in_dim %v2521, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2523 = stablehlo.divide %v2522, %v2519 : tensor<32x128x28x28xf32>
    %v2524 = stablehlo.subtract %v2517, %v2523 : tensor<32x128x28x28xf32>
    %v2525 = stablehlo.multiply %v2524, %v2524 : tensor<32x128x28x28xf32>
    %v2526 = stablehlo.reduce(%v2525 init: %v2518) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2527 = stablehlo.broadcast_in_dim %v2526, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2528 = stablehlo.divide %v2527, %v2519 : tensor<32x128x28x28xf32>
    %v2529 = stablehlo.add %v2528, %v2520 : tensor<32x128x28x28xf32>
    %v2530 = stablehlo.rsqrt %v2529 : tensor<32x128x28x28xf32>
    %v2531 = stablehlo.multiply %v2524, %v2530 : tensor<32x128x28x28xf32>
    %v2532 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2533 = stablehlo.multiply %v2532, %v2516 : tensor<32x128x28x28xf32>
    %v2534 = stablehlo.reduce(%v2533 init: %v2518) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2535 = stablehlo.broadcast_in_dim %v2534, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2536 = stablehlo.multiply %v2531, %v2533 : tensor<32x128x28x28xf32>
    %v2537 = stablehlo.reduce(%v2536 init: %v2518) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2538 = stablehlo.broadcast_in_dim %v2537, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2539 = stablehlo.multiply %v2533, %v2519 : tensor<32x128x28x28xf32>
    %v2540 = stablehlo.subtract %v2539, %v2535 : tensor<32x128x28x28xf32>
    %v2541 = stablehlo.multiply %v2531, %v2538 : tensor<32x128x28x28xf32>
    %v2542 = stablehlo.subtract %v2540, %v2541 : tensor<32x128x28x28xf32>
    %v2543 = stablehlo.divide %v2530, %v2519 : tensor<32x128x28x28xf32>
    %v2544 = stablehlo.multiply %v2543, %v2542 : tensor<32x128x28x28xf32>
    %v2545 = stablehlo.reshape %v2544 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2546 = stablehlo.reshape %v2545 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2547 = stablehlo.transpose %s2b2W2, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2548 = stablehlo.reverse %v2547, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2549 = stablehlo.convolution(%v2546, %v2548)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2550 = stablehlo.reshape %v2549 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2551 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v2552 = stablehlo.compare GT, %v410, %v2551 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v2553 = stablehlo.select %v2552, %v2550, %v2551 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v2554 = stablehlo.reshape %v2553 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2555 = stablehlo.reshape %v390 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2556 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2557 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v2558 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2559 = stablehlo.reduce(%v2555 init: %v2556) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2560 = stablehlo.broadcast_in_dim %v2559, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2561 = stablehlo.divide %v2560, %v2557 : tensor<32x128x28x28xf32>
    %v2562 = stablehlo.subtract %v2555, %v2561 : tensor<32x128x28x28xf32>
    %v2563 = stablehlo.multiply %v2562, %v2562 : tensor<32x128x28x28xf32>
    %v2564 = stablehlo.reduce(%v2563 init: %v2556) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2565 = stablehlo.broadcast_in_dim %v2564, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2566 = stablehlo.divide %v2565, %v2557 : tensor<32x128x28x28xf32>
    %v2567 = stablehlo.add %v2566, %v2558 : tensor<32x128x28x28xf32>
    %v2568 = stablehlo.rsqrt %v2567 : tensor<32x128x28x28xf32>
    %v2569 = stablehlo.multiply %v2562, %v2568 : tensor<32x128x28x28xf32>
    %v2570 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2571 = stablehlo.multiply %v2570, %v2554 : tensor<32x128x28x28xf32>
    %v2572 = stablehlo.reduce(%v2571 init: %v2556) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2573 = stablehlo.broadcast_in_dim %v2572, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2574 = stablehlo.multiply %v2569, %v2571 : tensor<32x128x28x28xf32>
    %v2575 = stablehlo.reduce(%v2574 init: %v2556) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2576 = stablehlo.broadcast_in_dim %v2575, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2577 = stablehlo.multiply %v2571, %v2557 : tensor<32x128x28x28xf32>
    %v2578 = stablehlo.subtract %v2577, %v2573 : tensor<32x128x28x28xf32>
    %v2579 = stablehlo.multiply %v2569, %v2576 : tensor<32x128x28x28xf32>
    %v2580 = stablehlo.subtract %v2578, %v2579 : tensor<32x128x28x28xf32>
    %v2581 = stablehlo.divide %v2568, %v2557 : tensor<32x128x28x28xf32>
    %v2582 = stablehlo.multiply %v2581, %v2580 : tensor<32x128x28x28xf32>
    %v2583 = stablehlo.reshape %v2582 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2584 = stablehlo.reshape %v2583 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2585 = stablehlo.transpose %s2b2W1, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2586 = stablehlo.reverse %v2585, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2587 = stablehlo.convolution(%v2584, %v2586)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2588 = stablehlo.reshape %v2587 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2589 = stablehlo.add %v2588, %v2515 : tensor<32x100352xf32>
    %v2590 = stablehlo.reshape %v385 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2591 = stablehlo.reshape %v2583 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2592 = stablehlo.transpose %v2590, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2593 = stablehlo.transpose %v2591, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2594 = stablehlo.convolution(%v2592, %v2593)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2595 = stablehlo.transpose %v2594, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2596 = stablehlo.constant dense<0.003125> : tensor<128x128x3x3xf32>
    %v2597 = stablehlo.multiply %v2595, %v2596 : tensor<128x128x3x3xf32>
    %v2598 = stablehlo.subtract %s2b2W1, %v2597 : tensor<128x128x3x3xf32>
    %v2599 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2600 = stablehlo.reshape %v390 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2601 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v2602 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2603 = stablehlo.reduce(%v2600 init: %v2599) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2604 = stablehlo.broadcast_in_dim %v2603, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2605 = stablehlo.divide %v2604, %v2601 : tensor<32x128x28x28xf32>
    %v2606 = stablehlo.subtract %v2600, %v2605 : tensor<32x128x28x28xf32>
    %v2607 = stablehlo.multiply %v2606, %v2606 : tensor<32x128x28x28xf32>
    %v2608 = stablehlo.reduce(%v2607 init: %v2599) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2609 = stablehlo.broadcast_in_dim %v2608, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2610 = stablehlo.divide %v2609, %v2601 : tensor<32x128x28x28xf32>
    %v2611 = stablehlo.add %v2610, %v2602 : tensor<32x128x28x28xf32>
    %v2612 = stablehlo.rsqrt %v2611 : tensor<32x128x28x28xf32>
    %v2613 = stablehlo.multiply %v2606, %v2612 : tensor<32x128x28x28xf32>
    %v2614 = stablehlo.reshape %v2553 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2615 = stablehlo.multiply %v2614, %v2613 : tensor<32x128x28x28xf32>
    %v2616 = stablehlo.reduce(%v2615 init: %v2599) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2617 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v2618 = stablehlo.multiply %v2616, %v2617 : tensor<128xf32>
    %v2619 = stablehlo.subtract %s2b2g1, %v2618 : tensor<128xf32>
    %v2620 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2621 = stablehlo.reshape %v2553 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2622 = stablehlo.reduce(%v2621 init: %v2620) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2623 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v2624 = stablehlo.multiply %v2622, %v2623 : tensor<128xf32>
    %v2625 = stablehlo.subtract %s2b2bt1, %v2624 : tensor<128xf32>
    %v2626 = stablehlo.reshape %v412 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2627 = stablehlo.reshape %v2545 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2628 = stablehlo.transpose %v2626, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2629 = stablehlo.transpose %v2627, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2630 = stablehlo.convolution(%v2628, %v2629)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2631 = stablehlo.transpose %v2630, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2632 = stablehlo.constant dense<0.003125> : tensor<128x128x3x3xf32>
    %v2633 = stablehlo.multiply %v2631, %v2632 : tensor<128x128x3x3xf32>
    %v2634 = stablehlo.subtract %s2b2W2, %v2633 : tensor<128x128x3x3xf32>
    %v2635 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2636 = stablehlo.reshape %v417 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2637 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v2638 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2639 = stablehlo.reduce(%v2636 init: %v2635) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2640 = stablehlo.broadcast_in_dim %v2639, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2641 = stablehlo.divide %v2640, %v2637 : tensor<32x128x28x28xf32>
    %v2642 = stablehlo.subtract %v2636, %v2641 : tensor<32x128x28x28xf32>
    %v2643 = stablehlo.multiply %v2642, %v2642 : tensor<32x128x28x28xf32>
    %v2644 = stablehlo.reduce(%v2643 init: %v2635) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2645 = stablehlo.broadcast_in_dim %v2644, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2646 = stablehlo.divide %v2645, %v2637 : tensor<32x128x28x28xf32>
    %v2647 = stablehlo.add %v2646, %v2638 : tensor<32x128x28x28xf32>
    %v2648 = stablehlo.rsqrt %v2647 : tensor<32x128x28x28xf32>
    %v2649 = stablehlo.multiply %v2642, %v2648 : tensor<32x128x28x28xf32>
    %v2650 = stablehlo.reshape %v2515 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2651 = stablehlo.multiply %v2650, %v2649 : tensor<32x128x28x28xf32>
    %v2652 = stablehlo.reduce(%v2651 init: %v2635) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2653 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v2654 = stablehlo.multiply %v2652, %v2653 : tensor<128xf32>
    %v2655 = stablehlo.subtract %s2b2g2, %v2654 : tensor<128xf32>
    %v2656 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2657 = stablehlo.reshape %v2515 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2658 = stablehlo.reduce(%v2657 init: %v2656) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2659 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v2660 = stablehlo.multiply %v2658, %v2659 : tensor<128xf32>
    %v2661 = stablehlo.subtract %s2b2bt2, %v2660 : tensor<128xf32>
    %v2662 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v2663 = stablehlo.compare GT, %v383, %v2662 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v2664 = stablehlo.select %v2663, %v2589, %v2662 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v2665 = stablehlo.reshape %v2664 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2666 = stablehlo.reshape %v362 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2667 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2668 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v2669 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2670 = stablehlo.reduce(%v2666 init: %v2667) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2671 = stablehlo.broadcast_in_dim %v2670, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2672 = stablehlo.divide %v2671, %v2668 : tensor<32x128x28x28xf32>
    %v2673 = stablehlo.subtract %v2666, %v2672 : tensor<32x128x28x28xf32>
    %v2674 = stablehlo.multiply %v2673, %v2673 : tensor<32x128x28x28xf32>
    %v2675 = stablehlo.reduce(%v2674 init: %v2667) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2676 = stablehlo.broadcast_in_dim %v2675, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2677 = stablehlo.divide %v2676, %v2668 : tensor<32x128x28x28xf32>
    %v2678 = stablehlo.add %v2677, %v2669 : tensor<32x128x28x28xf32>
    %v2679 = stablehlo.rsqrt %v2678 : tensor<32x128x28x28xf32>
    %v2680 = stablehlo.multiply %v2673, %v2679 : tensor<32x128x28x28xf32>
    %v2681 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2682 = stablehlo.multiply %v2681, %v2665 : tensor<32x128x28x28xf32>
    %v2683 = stablehlo.reduce(%v2682 init: %v2667) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2684 = stablehlo.broadcast_in_dim %v2683, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2685 = stablehlo.multiply %v2680, %v2682 : tensor<32x128x28x28xf32>
    %v2686 = stablehlo.reduce(%v2685 init: %v2667) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2687 = stablehlo.broadcast_in_dim %v2686, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2688 = stablehlo.multiply %v2682, %v2668 : tensor<32x128x28x28xf32>
    %v2689 = stablehlo.subtract %v2688, %v2684 : tensor<32x128x28x28xf32>
    %v2690 = stablehlo.multiply %v2680, %v2687 : tensor<32x128x28x28xf32>
    %v2691 = stablehlo.subtract %v2689, %v2690 : tensor<32x128x28x28xf32>
    %v2692 = stablehlo.divide %v2679, %v2668 : tensor<32x128x28x28xf32>
    %v2693 = stablehlo.multiply %v2692, %v2691 : tensor<32x128x28x28xf32>
    %v2694 = stablehlo.reshape %v2693 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2695 = stablehlo.reshape %v2694 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2696 = stablehlo.transpose %s2b1W2, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2697 = stablehlo.reverse %v2696, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2698 = stablehlo.convolution(%v2695, %v2697)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2699 = stablehlo.reshape %v2698 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2700 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v2701 = stablehlo.compare GT, %v355, %v2700 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v2702 = stablehlo.select %v2701, %v2699, %v2700 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v2703 = stablehlo.reshape %v2702 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2704 = stablehlo.reshape %v335 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2705 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2706 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v2707 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2708 = stablehlo.reduce(%v2704 init: %v2705) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2709 = stablehlo.broadcast_in_dim %v2708, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2710 = stablehlo.divide %v2709, %v2706 : tensor<32x128x28x28xf32>
    %v2711 = stablehlo.subtract %v2704, %v2710 : tensor<32x128x28x28xf32>
    %v2712 = stablehlo.multiply %v2711, %v2711 : tensor<32x128x28x28xf32>
    %v2713 = stablehlo.reduce(%v2712 init: %v2705) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2714 = stablehlo.broadcast_in_dim %v2713, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2715 = stablehlo.divide %v2714, %v2706 : tensor<32x128x28x28xf32>
    %v2716 = stablehlo.add %v2715, %v2707 : tensor<32x128x28x28xf32>
    %v2717 = stablehlo.rsqrt %v2716 : tensor<32x128x28x28xf32>
    %v2718 = stablehlo.multiply %v2711, %v2717 : tensor<32x128x28x28xf32>
    %v2719 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2720 = stablehlo.multiply %v2719, %v2703 : tensor<32x128x28x28xf32>
    %v2721 = stablehlo.reduce(%v2720 init: %v2705) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2722 = stablehlo.broadcast_in_dim %v2721, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2723 = stablehlo.multiply %v2718, %v2720 : tensor<32x128x28x28xf32>
    %v2724 = stablehlo.reduce(%v2723 init: %v2705) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2725 = stablehlo.broadcast_in_dim %v2724, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2726 = stablehlo.multiply %v2720, %v2706 : tensor<32x128x28x28xf32>
    %v2727 = stablehlo.subtract %v2726, %v2722 : tensor<32x128x28x28xf32>
    %v2728 = stablehlo.multiply %v2718, %v2725 : tensor<32x128x28x28xf32>
    %v2729 = stablehlo.subtract %v2727, %v2728 : tensor<32x128x28x28xf32>
    %v2730 = stablehlo.divide %v2717, %v2706 : tensor<32x128x28x28xf32>
    %v2731 = stablehlo.multiply %v2730, %v2729 : tensor<32x128x28x28xf32>
    %v2732 = stablehlo.reshape %v2731 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2733 = stablehlo.reshape %v2732 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2734 = stablehlo.transpose %s2b1W1, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2735 = stablehlo.reverse %v2734, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2736 = stablehlo.convolution(%v2733, %v2735)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2737 = stablehlo.reshape %v2736 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2738 = stablehlo.add %v2737, %v2664 : tensor<32x100352xf32>
    %v2739 = stablehlo.reshape %v330 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2740 = stablehlo.reshape %v2732 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2741 = stablehlo.transpose %v2739, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2742 = stablehlo.transpose %v2740, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2743 = stablehlo.convolution(%v2741, %v2742)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2744 = stablehlo.transpose %v2743, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2745 = stablehlo.constant dense<0.003125> : tensor<128x128x3x3xf32>
    %v2746 = stablehlo.multiply %v2744, %v2745 : tensor<128x128x3x3xf32>
    %v2747 = stablehlo.subtract %s2b1W1, %v2746 : tensor<128x128x3x3xf32>
    %v2748 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2749 = stablehlo.reshape %v335 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2750 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v2751 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2752 = stablehlo.reduce(%v2749 init: %v2748) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2753 = stablehlo.broadcast_in_dim %v2752, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2754 = stablehlo.divide %v2753, %v2750 : tensor<32x128x28x28xf32>
    %v2755 = stablehlo.subtract %v2749, %v2754 : tensor<32x128x28x28xf32>
    %v2756 = stablehlo.multiply %v2755, %v2755 : tensor<32x128x28x28xf32>
    %v2757 = stablehlo.reduce(%v2756 init: %v2748) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2758 = stablehlo.broadcast_in_dim %v2757, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2759 = stablehlo.divide %v2758, %v2750 : tensor<32x128x28x28xf32>
    %v2760 = stablehlo.add %v2759, %v2751 : tensor<32x128x28x28xf32>
    %v2761 = stablehlo.rsqrt %v2760 : tensor<32x128x28x28xf32>
    %v2762 = stablehlo.multiply %v2755, %v2761 : tensor<32x128x28x28xf32>
    %v2763 = stablehlo.reshape %v2702 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2764 = stablehlo.multiply %v2763, %v2762 : tensor<32x128x28x28xf32>
    %v2765 = stablehlo.reduce(%v2764 init: %v2748) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2766 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v2767 = stablehlo.multiply %v2765, %v2766 : tensor<128xf32>
    %v2768 = stablehlo.subtract %s2b1g1, %v2767 : tensor<128xf32>
    %v2769 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2770 = stablehlo.reshape %v2702 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2771 = stablehlo.reduce(%v2770 init: %v2769) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2772 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v2773 = stablehlo.multiply %v2771, %v2772 : tensor<128xf32>
    %v2774 = stablehlo.subtract %s2b1bt1, %v2773 : tensor<128xf32>
    %v2775 = stablehlo.reshape %v357 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2776 = stablehlo.reshape %v2694 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2777 = stablehlo.transpose %v2775, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2778 = stablehlo.transpose %v2776, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2779 = stablehlo.convolution(%v2777, %v2778)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2780 = stablehlo.transpose %v2779, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2781 = stablehlo.constant dense<0.003125> : tensor<128x128x3x3xf32>
    %v2782 = stablehlo.multiply %v2780, %v2781 : tensor<128x128x3x3xf32>
    %v2783 = stablehlo.subtract %s2b1W2, %v2782 : tensor<128x128x3x3xf32>
    %v2784 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2785 = stablehlo.reshape %v362 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2786 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v2787 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2788 = stablehlo.reduce(%v2785 init: %v2784) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2789 = stablehlo.broadcast_in_dim %v2788, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2790 = stablehlo.divide %v2789, %v2786 : tensor<32x128x28x28xf32>
    %v2791 = stablehlo.subtract %v2785, %v2790 : tensor<32x128x28x28xf32>
    %v2792 = stablehlo.multiply %v2791, %v2791 : tensor<32x128x28x28xf32>
    %v2793 = stablehlo.reduce(%v2792 init: %v2784) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2794 = stablehlo.broadcast_in_dim %v2793, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2795 = stablehlo.divide %v2794, %v2786 : tensor<32x128x28x28xf32>
    %v2796 = stablehlo.add %v2795, %v2787 : tensor<32x128x28x28xf32>
    %v2797 = stablehlo.rsqrt %v2796 : tensor<32x128x28x28xf32>
    %v2798 = stablehlo.multiply %v2791, %v2797 : tensor<32x128x28x28xf32>
    %v2799 = stablehlo.reshape %v2664 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2800 = stablehlo.multiply %v2799, %v2798 : tensor<32x128x28x28xf32>
    %v2801 = stablehlo.reduce(%v2800 init: %v2784) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2802 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v2803 = stablehlo.multiply %v2801, %v2802 : tensor<128xf32>
    %v2804 = stablehlo.subtract %s2b1g2, %v2803 : tensor<128xf32>
    %v2805 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2806 = stablehlo.reshape %v2664 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2807 = stablehlo.reduce(%v2806 init: %v2805) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2808 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v2809 = stablehlo.multiply %v2807, %v2808 : tensor<128xf32>
    %v2810 = stablehlo.subtract %s2b1bt2, %v2809 : tensor<128xf32>
    %v2811 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v2812 = stablehlo.compare GT, %v328, %v2811 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v2813 = stablehlo.select %v2812, %v2738, %v2811 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v2814 = stablehlo.reshape %v2813 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2815 = stablehlo.reshape %v307 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2816 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2817 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v2818 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2819 = stablehlo.reduce(%v2815 init: %v2816) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2820 = stablehlo.broadcast_in_dim %v2819, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2821 = stablehlo.divide %v2820, %v2817 : tensor<32x128x28x28xf32>
    %v2822 = stablehlo.subtract %v2815, %v2821 : tensor<32x128x28x28xf32>
    %v2823 = stablehlo.multiply %v2822, %v2822 : tensor<32x128x28x28xf32>
    %v2824 = stablehlo.reduce(%v2823 init: %v2816) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2825 = stablehlo.broadcast_in_dim %v2824, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2826 = stablehlo.divide %v2825, %v2817 : tensor<32x128x28x28xf32>
    %v2827 = stablehlo.add %v2826, %v2818 : tensor<32x128x28x28xf32>
    %v2828 = stablehlo.rsqrt %v2827 : tensor<32x128x28x28xf32>
    %v2829 = stablehlo.multiply %v2822, %v2828 : tensor<32x128x28x28xf32>
    %v2830 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2831 = stablehlo.multiply %v2830, %v2814 : tensor<32x128x28x28xf32>
    %v2832 = stablehlo.reduce(%v2831 init: %v2816) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2833 = stablehlo.broadcast_in_dim %v2832, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2834 = stablehlo.multiply %v2829, %v2831 : tensor<32x128x28x28xf32>
    %v2835 = stablehlo.reduce(%v2834 init: %v2816) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2836 = stablehlo.broadcast_in_dim %v2835, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2837 = stablehlo.multiply %v2831, %v2817 : tensor<32x128x28x28xf32>
    %v2838 = stablehlo.subtract %v2837, %v2833 : tensor<32x128x28x28xf32>
    %v2839 = stablehlo.multiply %v2829, %v2836 : tensor<32x128x28x28xf32>
    %v2840 = stablehlo.subtract %v2838, %v2839 : tensor<32x128x28x28xf32>
    %v2841 = stablehlo.divide %v2828, %v2817 : tensor<32x128x28x28xf32>
    %v2842 = stablehlo.multiply %v2841, %v2840 : tensor<32x128x28x28xf32>
    %v2843 = stablehlo.reshape %v2842 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2844 = stablehlo.reshape %v2843 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2845 = stablehlo.transpose %s2b0W2, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2846 = stablehlo.reverse %v2845, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2847 = stablehlo.convolution(%v2844, %v2846)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2848 = stablehlo.reshape %v2847 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2849 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v2850 = stablehlo.compare GT, %v300, %v2849 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v2851 = stablehlo.select %v2850, %v2848, %v2849 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v2852 = stablehlo.reshape %v2851 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2853 = stablehlo.reshape %v280 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2854 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2855 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v2856 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2857 = stablehlo.reduce(%v2853 init: %v2854) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2858 = stablehlo.broadcast_in_dim %v2857, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2859 = stablehlo.divide %v2858, %v2855 : tensor<32x128x28x28xf32>
    %v2860 = stablehlo.subtract %v2853, %v2859 : tensor<32x128x28x28xf32>
    %v2861 = stablehlo.multiply %v2860, %v2860 : tensor<32x128x28x28xf32>
    %v2862 = stablehlo.reduce(%v2861 init: %v2854) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2863 = stablehlo.broadcast_in_dim %v2862, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2864 = stablehlo.divide %v2863, %v2855 : tensor<32x128x28x28xf32>
    %v2865 = stablehlo.add %v2864, %v2856 : tensor<32x128x28x28xf32>
    %v2866 = stablehlo.rsqrt %v2865 : tensor<32x128x28x28xf32>
    %v2867 = stablehlo.multiply %v2860, %v2866 : tensor<32x128x28x28xf32>
    %v2868 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2869 = stablehlo.multiply %v2868, %v2852 : tensor<32x128x28x28xf32>
    %v2870 = stablehlo.reduce(%v2869 init: %v2854) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2871 = stablehlo.broadcast_in_dim %v2870, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2872 = stablehlo.multiply %v2867, %v2869 : tensor<32x128x28x28xf32>
    %v2873 = stablehlo.reduce(%v2872 init: %v2854) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2874 = stablehlo.broadcast_in_dim %v2873, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2875 = stablehlo.multiply %v2869, %v2855 : tensor<32x128x28x28xf32>
    %v2876 = stablehlo.subtract %v2875, %v2871 : tensor<32x128x28x28xf32>
    %v2877 = stablehlo.multiply %v2867, %v2874 : tensor<32x128x28x28xf32>
    %v2878 = stablehlo.subtract %v2876, %v2877 : tensor<32x128x28x28xf32>
    %v2879 = stablehlo.divide %v2866, %v2855 : tensor<32x128x28x28xf32>
    %v2880 = stablehlo.multiply %v2879, %v2878 : tensor<32x128x28x28xf32>
    %v2881 = stablehlo.reshape %v2880 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2882 = stablehlo.reshape %v2881 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2883 = stablehlo.transpose %s2b0W1, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2884 = stablehlo.reverse %v2883, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2885 = stablehlo.convolution(%v2882, %v2884)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2886 = stablehlo.reshape %v2885 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2887 = stablehlo.add %v2886, %v2813 : tensor<32x100352xf32>
    %v2888 = stablehlo.reshape %v275 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2889 = stablehlo.reshape %v2881 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2890 = stablehlo.transpose %v2888, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2891 = stablehlo.transpose %v2889, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2892 = stablehlo.convolution(%v2890, %v2891)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2893 = stablehlo.transpose %v2892, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2894 = stablehlo.constant dense<0.003125> : tensor<128x128x3x3xf32>
    %v2895 = stablehlo.multiply %v2893, %v2894 : tensor<128x128x3x3xf32>
    %v2896 = stablehlo.subtract %s2b0W1, %v2895 : tensor<128x128x3x3xf32>
    %v2897 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2898 = stablehlo.reshape %v280 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2899 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v2900 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2901 = stablehlo.reduce(%v2898 init: %v2897) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2902 = stablehlo.broadcast_in_dim %v2901, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2903 = stablehlo.divide %v2902, %v2899 : tensor<32x128x28x28xf32>
    %v2904 = stablehlo.subtract %v2898, %v2903 : tensor<32x128x28x28xf32>
    %v2905 = stablehlo.multiply %v2904, %v2904 : tensor<32x128x28x28xf32>
    %v2906 = stablehlo.reduce(%v2905 init: %v2897) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2907 = stablehlo.broadcast_in_dim %v2906, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2908 = stablehlo.divide %v2907, %v2899 : tensor<32x128x28x28xf32>
    %v2909 = stablehlo.add %v2908, %v2900 : tensor<32x128x28x28xf32>
    %v2910 = stablehlo.rsqrt %v2909 : tensor<32x128x28x28xf32>
    %v2911 = stablehlo.multiply %v2904, %v2910 : tensor<32x128x28x28xf32>
    %v2912 = stablehlo.reshape %v2851 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2913 = stablehlo.multiply %v2912, %v2911 : tensor<32x128x28x28xf32>
    %v2914 = stablehlo.reduce(%v2913 init: %v2897) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2915 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v2916 = stablehlo.multiply %v2914, %v2915 : tensor<128xf32>
    %v2917 = stablehlo.subtract %s2b0g1, %v2916 : tensor<128xf32>
    %v2918 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2919 = stablehlo.reshape %v2851 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2920 = stablehlo.reduce(%v2919 init: %v2918) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2921 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v2922 = stablehlo.multiply %v2920, %v2921 : tensor<128xf32>
    %v2923 = stablehlo.subtract %s2b0bt1, %v2922 : tensor<128xf32>
    %v2924 = stablehlo.reshape %v302 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2925 = stablehlo.reshape %v2843 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2926 = stablehlo.transpose %v2924, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2927 = stablehlo.transpose %v2925, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2928 = stablehlo.convolution(%v2926, %v2927)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2929 = stablehlo.transpose %v2928, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2930 = stablehlo.constant dense<0.003125> : tensor<128x128x3x3xf32>
    %v2931 = stablehlo.multiply %v2929, %v2930 : tensor<128x128x3x3xf32>
    %v2932 = stablehlo.subtract %s2b0W2, %v2931 : tensor<128x128x3x3xf32>
    %v2933 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2934 = stablehlo.reshape %v307 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2935 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v2936 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2937 = stablehlo.reduce(%v2934 init: %v2933) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2938 = stablehlo.broadcast_in_dim %v2937, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2939 = stablehlo.divide %v2938, %v2935 : tensor<32x128x28x28xf32>
    %v2940 = stablehlo.subtract %v2934, %v2939 : tensor<32x128x28x28xf32>
    %v2941 = stablehlo.multiply %v2940, %v2940 : tensor<32x128x28x28xf32>
    %v2942 = stablehlo.reduce(%v2941 init: %v2933) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2943 = stablehlo.broadcast_in_dim %v2942, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2944 = stablehlo.divide %v2943, %v2935 : tensor<32x128x28x28xf32>
    %v2945 = stablehlo.add %v2944, %v2936 : tensor<32x128x28x28xf32>
    %v2946 = stablehlo.rsqrt %v2945 : tensor<32x128x28x28xf32>
    %v2947 = stablehlo.multiply %v2940, %v2946 : tensor<32x128x28x28xf32>
    %v2948 = stablehlo.reshape %v2813 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2949 = stablehlo.multiply %v2948, %v2947 : tensor<32x128x28x28xf32>
    %v2950 = stablehlo.reduce(%v2949 init: %v2933) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2951 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v2952 = stablehlo.multiply %v2950, %v2951 : tensor<128xf32>
    %v2953 = stablehlo.subtract %s2b0g2, %v2952 : tensor<128xf32>
    %v2954 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2955 = stablehlo.reshape %v2813 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2956 = stablehlo.reduce(%v2955 init: %v2954) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2957 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v2958 = stablehlo.multiply %v2956, %v2957 : tensor<128xf32>
    %v2959 = stablehlo.subtract %s2b0bt2, %v2958 : tensor<128xf32>
    %v2960 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v2961 = stablehlo.compare GT, %v273, %v2960 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v2962 = stablehlo.select %v2961, %v2887, %v2960 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v2963 = stablehlo.reshape %v2962 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2964 = stablehlo.reshape %v227 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2965 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2966 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v2967 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2968 = stablehlo.reduce(%v2964 init: %v2965) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2969 = stablehlo.broadcast_in_dim %v2968, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2970 = stablehlo.divide %v2969, %v2966 : tensor<32x128x28x28xf32>
    %v2971 = stablehlo.subtract %v2964, %v2970 : tensor<32x128x28x28xf32>
    %v2972 = stablehlo.multiply %v2971, %v2971 : tensor<32x128x28x28xf32>
    %v2973 = stablehlo.reduce(%v2972 init: %v2965) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2974 = stablehlo.broadcast_in_dim %v2973, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2975 = stablehlo.divide %v2974, %v2966 : tensor<32x128x28x28xf32>
    %v2976 = stablehlo.add %v2975, %v2967 : tensor<32x128x28x28xf32>
    %v2977 = stablehlo.rsqrt %v2976 : tensor<32x128x28x28xf32>
    %v2978 = stablehlo.multiply %v2971, %v2977 : tensor<32x128x28x28xf32>
    %v2979 = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2980 = stablehlo.multiply %v2979, %v2963 : tensor<32x128x28x28xf32>
    %v2981 = stablehlo.reduce(%v2980 init: %v2965) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2982 = stablehlo.broadcast_in_dim %v2981, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2983 = stablehlo.multiply %v2978, %v2980 : tensor<32x128x28x28xf32>
    %v2984 = stablehlo.reduce(%v2983 init: %v2965) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2985 = stablehlo.broadcast_in_dim %v2984, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2986 = stablehlo.multiply %v2980, %v2966 : tensor<32x128x28x28xf32>
    %v2987 = stablehlo.subtract %v2986, %v2982 : tensor<32x128x28x28xf32>
    %v2988 = stablehlo.multiply %v2978, %v2985 : tensor<32x128x28x28xf32>
    %v2989 = stablehlo.subtract %v2987, %v2988 : tensor<32x128x28x28xf32>
    %v2990 = stablehlo.divide %v2977, %v2966 : tensor<32x128x28x28xf32>
    %v2991 = stablehlo.multiply %v2990, %v2989 : tensor<32x128x28x28xf32>
    %v2992 = stablehlo.reshape %v2991 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2993 = stablehlo.reshape %v2992 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2994 = stablehlo.transpose %d2W2, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2995 = stablehlo.reverse %v2994, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2996 = stablehlo.convolution(%v2993, %v2995)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2997 = stablehlo.reshape %v2996 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2998 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v2999 = stablehlo.compare GT, %v220, %v2998 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v3000 = stablehlo.select %v2999, %v2997, %v2998 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v3001 = stablehlo.reshape %v3000 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3002 = stablehlo.reshape %v200 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3003 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3004 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v3005 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3006 = stablehlo.reduce(%v3002 init: %v3003) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3007 = stablehlo.broadcast_in_dim %v3006, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3008 = stablehlo.divide %v3007, %v3004 : tensor<32x128x28x28xf32>
    %v3009 = stablehlo.subtract %v3002, %v3008 : tensor<32x128x28x28xf32>
    %v3010 = stablehlo.multiply %v3009, %v3009 : tensor<32x128x28x28xf32>
    %v3011 = stablehlo.reduce(%v3010 init: %v3003) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3012 = stablehlo.broadcast_in_dim %v3011, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3013 = stablehlo.divide %v3012, %v3004 : tensor<32x128x28x28xf32>
    %v3014 = stablehlo.add %v3013, %v3005 : tensor<32x128x28x28xf32>
    %v3015 = stablehlo.rsqrt %v3014 : tensor<32x128x28x28xf32>
    %v3016 = stablehlo.multiply %v3009, %v3015 : tensor<32x128x28x28xf32>
    %v3017 = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3018 = stablehlo.multiply %v3017, %v3001 : tensor<32x128x28x28xf32>
    %v3019 = stablehlo.reduce(%v3018 init: %v3003) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3020 = stablehlo.broadcast_in_dim %v3019, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3021 = stablehlo.multiply %v3016, %v3018 : tensor<32x128x28x28xf32>
    %v3022 = stablehlo.reduce(%v3021 init: %v3003) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3023 = stablehlo.broadcast_in_dim %v3022, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3024 = stablehlo.multiply %v3018, %v3004 : tensor<32x128x28x28xf32>
    %v3025 = stablehlo.subtract %v3024, %v3020 : tensor<32x128x28x28xf32>
    %v3026 = stablehlo.multiply %v3016, %v3023 : tensor<32x128x28x28xf32>
    %v3027 = stablehlo.subtract %v3025, %v3026 : tensor<32x128x28x28xf32>
    %v3028 = stablehlo.divide %v3015, %v3004 : tensor<32x128x28x28xf32>
    %v3029 = stablehlo.multiply %v3028, %v3027 : tensor<32x128x28x28xf32>
    %v3030 = stablehlo.reshape %v3029 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3031 = stablehlo.reshape %v3030 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3032 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3033 = stablehlo.pad %v3031, %v3032, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %v3034 = stablehlo.transpose %d2W1, dims = [1, 0, 2, 3] : (tensor<128x64x3x3xf32>) -> tensor<64x128x3x3xf32>
    %v3035 = stablehlo.reverse %v3034, dims = [2, 3] : tensor<64x128x3x3xf32>
    %v3036 = stablehlo.convolution(%v3033, %v3035)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x56x56xf32>, tensor<64x128x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3037 = stablehlo.reshape %v3036 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3038 = stablehlo.reshape %v2962 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3039 = stablehlo.reshape %v252 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3040 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3041 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v3042 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3043 = stablehlo.reduce(%v3039 init: %v3040) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3044 = stablehlo.broadcast_in_dim %v3043, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3045 = stablehlo.divide %v3044, %v3041 : tensor<32x128x28x28xf32>
    %v3046 = stablehlo.subtract %v3039, %v3045 : tensor<32x128x28x28xf32>
    %v3047 = stablehlo.multiply %v3046, %v3046 : tensor<32x128x28x28xf32>
    %v3048 = stablehlo.reduce(%v3047 init: %v3040) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3049 = stablehlo.broadcast_in_dim %v3048, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3050 = stablehlo.divide %v3049, %v3041 : tensor<32x128x28x28xf32>
    %v3051 = stablehlo.add %v3050, %v3042 : tensor<32x128x28x28xf32>
    %v3052 = stablehlo.rsqrt %v3051 : tensor<32x128x28x28xf32>
    %v3053 = stablehlo.multiply %v3046, %v3052 : tensor<32x128x28x28xf32>
    %v3054 = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3055 = stablehlo.multiply %v3054, %v3038 : tensor<32x128x28x28xf32>
    %v3056 = stablehlo.reduce(%v3055 init: %v3040) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3057 = stablehlo.broadcast_in_dim %v3056, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3058 = stablehlo.multiply %v3053, %v3055 : tensor<32x128x28x28xf32>
    %v3059 = stablehlo.reduce(%v3058 init: %v3040) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3060 = stablehlo.broadcast_in_dim %v3059, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3061 = stablehlo.multiply %v3055, %v3041 : tensor<32x128x28x28xf32>
    %v3062 = stablehlo.subtract %v3061, %v3057 : tensor<32x128x28x28xf32>
    %v3063 = stablehlo.multiply %v3053, %v3060 : tensor<32x128x28x28xf32>
    %v3064 = stablehlo.subtract %v3062, %v3063 : tensor<32x128x28x28xf32>
    %v3065 = stablehlo.divide %v3052, %v3041 : tensor<32x128x28x28xf32>
    %v3066 = stablehlo.multiply %v3065, %v3064 : tensor<32x128x28x28xf32>
    %v3067 = stablehlo.reshape %v3066 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3068 = stablehlo.reshape %v3067 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3069 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3070 = stablehlo.pad %v3068, %v3069, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %v3071 = stablehlo.transpose %d2Wp, dims = [1, 0, 2, 3] : (tensor<128x64x3x3xf32>) -> tensor<64x128x3x3xf32>
    %v3072 = stablehlo.reverse %v3071, dims = [2, 3] : tensor<64x128x3x3xf32>
    %v3073 = stablehlo.convolution(%v3070, %v3072)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x56x56xf32>, tensor<64x128x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3074 = stablehlo.reshape %v3073 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3075 = stablehlo.add %v3037, %v3074 : tensor<32x200704xf32>
    %v3076 = stablehlo.reshape %v195 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3077 = stablehlo.reshape %v3030 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3078 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3079 = stablehlo.pad %v3077, %v3078, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %v3080 = stablehlo.transpose %v3076, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3081 = stablehlo.transpose %v3079, dims = [1, 0, 2, 3] : (tensor<32x128x56x56xf32>) -> tensor<128x32x56x56xf32>
    %v3082 = stablehlo.convolution(%v3080, %v3081)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<128x32x56x56xf32>) -> tensor<64x128x3x3xf32>
    %v3083 = stablehlo.transpose %v3082, dims = [1, 0, 2, 3] : (tensor<64x128x3x3xf32>) -> tensor<128x64x3x3xf32>
    %v3084 = stablehlo.constant dense<0.003125> : tensor<128x64x3x3xf32>
    %v3085 = stablehlo.multiply %v3083, %v3084 : tensor<128x64x3x3xf32>
    %v3086 = stablehlo.subtract %d2W1, %v3085 : tensor<128x64x3x3xf32>
    %v3087 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3088 = stablehlo.reshape %v200 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3089 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v3090 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3091 = stablehlo.reduce(%v3088 init: %v3087) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3092 = stablehlo.broadcast_in_dim %v3091, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3093 = stablehlo.divide %v3092, %v3089 : tensor<32x128x28x28xf32>
    %v3094 = stablehlo.subtract %v3088, %v3093 : tensor<32x128x28x28xf32>
    %v3095 = stablehlo.multiply %v3094, %v3094 : tensor<32x128x28x28xf32>
    %v3096 = stablehlo.reduce(%v3095 init: %v3087) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3097 = stablehlo.broadcast_in_dim %v3096, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3098 = stablehlo.divide %v3097, %v3089 : tensor<32x128x28x28xf32>
    %v3099 = stablehlo.add %v3098, %v3090 : tensor<32x128x28x28xf32>
    %v3100 = stablehlo.rsqrt %v3099 : tensor<32x128x28x28xf32>
    %v3101 = stablehlo.multiply %v3094, %v3100 : tensor<32x128x28x28xf32>
    %v3102 = stablehlo.reshape %v3000 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3103 = stablehlo.multiply %v3102, %v3101 : tensor<32x128x28x28xf32>
    %v3104 = stablehlo.reduce(%v3103 init: %v3087) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3105 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v3106 = stablehlo.multiply %v3104, %v3105 : tensor<128xf32>
    %v3107 = stablehlo.subtract %d2g1, %v3106 : tensor<128xf32>
    %v3108 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3109 = stablehlo.reshape %v3000 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3110 = stablehlo.reduce(%v3109 init: %v3108) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3111 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v3112 = stablehlo.multiply %v3110, %v3111 : tensor<128xf32>
    %v3113 = stablehlo.subtract %d2bt1, %v3112 : tensor<128xf32>
    %v3114 = stablehlo.reshape %v222 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3115 = stablehlo.reshape %v2992 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3116 = stablehlo.transpose %v3114, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3117 = stablehlo.transpose %v3115, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3118 = stablehlo.convolution(%v3116, %v3117)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v3119 = stablehlo.transpose %v3118, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3120 = stablehlo.constant dense<0.003125> : tensor<128x128x3x3xf32>
    %v3121 = stablehlo.multiply %v3119, %v3120 : tensor<128x128x3x3xf32>
    %v3122 = stablehlo.subtract %d2W2, %v3121 : tensor<128x128x3x3xf32>
    %v3123 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3124 = stablehlo.reshape %v227 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3125 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v3126 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3127 = stablehlo.reduce(%v3124 init: %v3123) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3128 = stablehlo.broadcast_in_dim %v3127, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3129 = stablehlo.divide %v3128, %v3125 : tensor<32x128x28x28xf32>
    %v3130 = stablehlo.subtract %v3124, %v3129 : tensor<32x128x28x28xf32>
    %v3131 = stablehlo.multiply %v3130, %v3130 : tensor<32x128x28x28xf32>
    %v3132 = stablehlo.reduce(%v3131 init: %v3123) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3133 = stablehlo.broadcast_in_dim %v3132, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3134 = stablehlo.divide %v3133, %v3125 : tensor<32x128x28x28xf32>
    %v3135 = stablehlo.add %v3134, %v3126 : tensor<32x128x28x28xf32>
    %v3136 = stablehlo.rsqrt %v3135 : tensor<32x128x28x28xf32>
    %v3137 = stablehlo.multiply %v3130, %v3136 : tensor<32x128x28x28xf32>
    %v3138 = stablehlo.reshape %v2962 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3139 = stablehlo.multiply %v3138, %v3137 : tensor<32x128x28x28xf32>
    %v3140 = stablehlo.reduce(%v3139 init: %v3123) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3141 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v3142 = stablehlo.multiply %v3140, %v3141 : tensor<128xf32>
    %v3143 = stablehlo.subtract %d2g2, %v3142 : tensor<128xf32>
    %v3144 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3145 = stablehlo.reshape %v2962 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3146 = stablehlo.reduce(%v3145 init: %v3144) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3147 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v3148 = stablehlo.multiply %v3146, %v3147 : tensor<128xf32>
    %v3149 = stablehlo.subtract %d2bt2, %v3148 : tensor<128xf32>
    %v3150 = stablehlo.reshape %v195 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3151 = stablehlo.reshape %v3067 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3152 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3153 = stablehlo.pad %v3151, %v3152, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %v3154 = stablehlo.transpose %v3150, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3155 = stablehlo.transpose %v3153, dims = [1, 0, 2, 3] : (tensor<32x128x56x56xf32>) -> tensor<128x32x56x56xf32>
    %v3156 = stablehlo.convolution(%v3154, %v3155)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<128x32x56x56xf32>) -> tensor<64x128x3x3xf32>
    %v3157 = stablehlo.transpose %v3156, dims = [1, 0, 2, 3] : (tensor<64x128x3x3xf32>) -> tensor<128x64x3x3xf32>
    %v3158 = stablehlo.constant dense<0.003125> : tensor<128x64x3x3xf32>
    %v3159 = stablehlo.multiply %v3157, %v3158 : tensor<128x64x3x3xf32>
    %v3160 = stablehlo.subtract %d2Wp, %v3159 : tensor<128x64x3x3xf32>
    %v3161 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3162 = stablehlo.reshape %v252 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3163 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v3164 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3165 = stablehlo.reduce(%v3162 init: %v3161) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3166 = stablehlo.broadcast_in_dim %v3165, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3167 = stablehlo.divide %v3166, %v3163 : tensor<32x128x28x28xf32>
    %v3168 = stablehlo.subtract %v3162, %v3167 : tensor<32x128x28x28xf32>
    %v3169 = stablehlo.multiply %v3168, %v3168 : tensor<32x128x28x28xf32>
    %v3170 = stablehlo.reduce(%v3169 init: %v3161) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3171 = stablehlo.broadcast_in_dim %v3170, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3172 = stablehlo.divide %v3171, %v3163 : tensor<32x128x28x28xf32>
    %v3173 = stablehlo.add %v3172, %v3164 : tensor<32x128x28x28xf32>
    %v3174 = stablehlo.rsqrt %v3173 : tensor<32x128x28x28xf32>
    %v3175 = stablehlo.multiply %v3168, %v3174 : tensor<32x128x28x28xf32>
    %v3176 = stablehlo.reshape %v2962 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3177 = stablehlo.multiply %v3176, %v3175 : tensor<32x128x28x28xf32>
    %v3178 = stablehlo.reduce(%v3177 init: %v3161) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3179 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v3180 = stablehlo.multiply %v3178, %v3179 : tensor<128xf32>
    %v3181 = stablehlo.subtract %d2gp, %v3180 : tensor<128xf32>
    %v3182 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3183 = stablehlo.reshape %v2962 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3184 = stablehlo.reduce(%v3183 init: %v3182) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3185 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v3186 = stablehlo.multiply %v3184, %v3185 : tensor<128xf32>
    %v3187 = stablehlo.subtract %d2btp, %v3186 : tensor<128xf32>
    %v3188 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v3189 = stablehlo.compare GT, %v193, %v3188 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v3190 = stablehlo.select %v3189, %v3075, %v3188 : tensor<32x200704xi1>, tensor<32x200704xf32>
    %v3191 = stablehlo.reshape %v3190 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3192 = stablehlo.reshape %v172 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3193 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3194 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v3195 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3196 = stablehlo.reduce(%v3192 init: %v3193) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3197 = stablehlo.broadcast_in_dim %v3196, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3198 = stablehlo.divide %v3197, %v3194 : tensor<32x64x56x56xf32>
    %v3199 = stablehlo.subtract %v3192, %v3198 : tensor<32x64x56x56xf32>
    %v3200 = stablehlo.multiply %v3199, %v3199 : tensor<32x64x56x56xf32>
    %v3201 = stablehlo.reduce(%v3200 init: %v3193) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3202 = stablehlo.broadcast_in_dim %v3201, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3203 = stablehlo.divide %v3202, %v3194 : tensor<32x64x56x56xf32>
    %v3204 = stablehlo.add %v3203, %v3195 : tensor<32x64x56x56xf32>
    %v3205 = stablehlo.rsqrt %v3204 : tensor<32x64x56x56xf32>
    %v3206 = stablehlo.multiply %v3199, %v3205 : tensor<32x64x56x56xf32>
    %v3207 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3208 = stablehlo.multiply %v3207, %v3191 : tensor<32x64x56x56xf32>
    %v3209 = stablehlo.reduce(%v3208 init: %v3193) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3210 = stablehlo.broadcast_in_dim %v3209, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3211 = stablehlo.multiply %v3206, %v3208 : tensor<32x64x56x56xf32>
    %v3212 = stablehlo.reduce(%v3211 init: %v3193) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3213 = stablehlo.broadcast_in_dim %v3212, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3214 = stablehlo.multiply %v3208, %v3194 : tensor<32x64x56x56xf32>
    %v3215 = stablehlo.subtract %v3214, %v3210 : tensor<32x64x56x56xf32>
    %v3216 = stablehlo.multiply %v3206, %v3213 : tensor<32x64x56x56xf32>
    %v3217 = stablehlo.subtract %v3215, %v3216 : tensor<32x64x56x56xf32>
    %v3218 = stablehlo.divide %v3205, %v3194 : tensor<32x64x56x56xf32>
    %v3219 = stablehlo.multiply %v3218, %v3217 : tensor<32x64x56x56xf32>
    %v3220 = stablehlo.reshape %v3219 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3221 = stablehlo.reshape %v3220 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3222 = stablehlo.transpose %s1b2W2, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3223 = stablehlo.reverse %v3222, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3224 = stablehlo.convolution(%v3221, %v3223)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3225 = stablehlo.reshape %v3224 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3226 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v3227 = stablehlo.compare GT, %v165, %v3226 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v3228 = stablehlo.select %v3227, %v3225, %v3226 : tensor<32x200704xi1>, tensor<32x200704xf32>
    %v3229 = stablehlo.reshape %v3228 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3230 = stablehlo.reshape %v145 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3231 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3232 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v3233 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3234 = stablehlo.reduce(%v3230 init: %v3231) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3235 = stablehlo.broadcast_in_dim %v3234, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3236 = stablehlo.divide %v3235, %v3232 : tensor<32x64x56x56xf32>
    %v3237 = stablehlo.subtract %v3230, %v3236 : tensor<32x64x56x56xf32>
    %v3238 = stablehlo.multiply %v3237, %v3237 : tensor<32x64x56x56xf32>
    %v3239 = stablehlo.reduce(%v3238 init: %v3231) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3240 = stablehlo.broadcast_in_dim %v3239, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3241 = stablehlo.divide %v3240, %v3232 : tensor<32x64x56x56xf32>
    %v3242 = stablehlo.add %v3241, %v3233 : tensor<32x64x56x56xf32>
    %v3243 = stablehlo.rsqrt %v3242 : tensor<32x64x56x56xf32>
    %v3244 = stablehlo.multiply %v3237, %v3243 : tensor<32x64x56x56xf32>
    %v3245 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3246 = stablehlo.multiply %v3245, %v3229 : tensor<32x64x56x56xf32>
    %v3247 = stablehlo.reduce(%v3246 init: %v3231) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3248 = stablehlo.broadcast_in_dim %v3247, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3249 = stablehlo.multiply %v3244, %v3246 : tensor<32x64x56x56xf32>
    %v3250 = stablehlo.reduce(%v3249 init: %v3231) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3251 = stablehlo.broadcast_in_dim %v3250, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3252 = stablehlo.multiply %v3246, %v3232 : tensor<32x64x56x56xf32>
    %v3253 = stablehlo.subtract %v3252, %v3248 : tensor<32x64x56x56xf32>
    %v3254 = stablehlo.multiply %v3244, %v3251 : tensor<32x64x56x56xf32>
    %v3255 = stablehlo.subtract %v3253, %v3254 : tensor<32x64x56x56xf32>
    %v3256 = stablehlo.divide %v3243, %v3232 : tensor<32x64x56x56xf32>
    %v3257 = stablehlo.multiply %v3256, %v3255 : tensor<32x64x56x56xf32>
    %v3258 = stablehlo.reshape %v3257 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3259 = stablehlo.reshape %v3258 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3260 = stablehlo.transpose %s1b2W1, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3261 = stablehlo.reverse %v3260, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3262 = stablehlo.convolution(%v3259, %v3261)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3263 = stablehlo.reshape %v3262 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3264 = stablehlo.add %v3263, %v3190 : tensor<32x200704xf32>
    %v3265 = stablehlo.reshape %v140 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3266 = stablehlo.reshape %v3258 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3267 = stablehlo.transpose %v3265, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3268 = stablehlo.transpose %v3266, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3269 = stablehlo.convolution(%v3267, %v3268)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3270 = stablehlo.transpose %v3269, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3271 = stablehlo.constant dense<0.003125> : tensor<64x64x3x3xf32>
    %v3272 = stablehlo.multiply %v3270, %v3271 : tensor<64x64x3x3xf32>
    %v3273 = stablehlo.subtract %s1b2W1, %v3272 : tensor<64x64x3x3xf32>
    %v3274 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3275 = stablehlo.reshape %v145 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3276 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v3277 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3278 = stablehlo.reduce(%v3275 init: %v3274) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3279 = stablehlo.broadcast_in_dim %v3278, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3280 = stablehlo.divide %v3279, %v3276 : tensor<32x64x56x56xf32>
    %v3281 = stablehlo.subtract %v3275, %v3280 : tensor<32x64x56x56xf32>
    %v3282 = stablehlo.multiply %v3281, %v3281 : tensor<32x64x56x56xf32>
    %v3283 = stablehlo.reduce(%v3282 init: %v3274) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3284 = stablehlo.broadcast_in_dim %v3283, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3285 = stablehlo.divide %v3284, %v3276 : tensor<32x64x56x56xf32>
    %v3286 = stablehlo.add %v3285, %v3277 : tensor<32x64x56x56xf32>
    %v3287 = stablehlo.rsqrt %v3286 : tensor<32x64x56x56xf32>
    %v3288 = stablehlo.multiply %v3281, %v3287 : tensor<32x64x56x56xf32>
    %v3289 = stablehlo.reshape %v3228 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3290 = stablehlo.multiply %v3289, %v3288 : tensor<32x64x56x56xf32>
    %v3291 = stablehlo.reduce(%v3290 init: %v3274) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3292 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3293 = stablehlo.multiply %v3291, %v3292 : tensor<64xf32>
    %v3294 = stablehlo.subtract %s1b2g1, %v3293 : tensor<64xf32>
    %v3295 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3296 = stablehlo.reshape %v3228 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3297 = stablehlo.reduce(%v3296 init: %v3295) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3298 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3299 = stablehlo.multiply %v3297, %v3298 : tensor<64xf32>
    %v3300 = stablehlo.subtract %s1b2bt1, %v3299 : tensor<64xf32>
    %v3301 = stablehlo.reshape %v167 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3302 = stablehlo.reshape %v3220 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3303 = stablehlo.transpose %v3301, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3304 = stablehlo.transpose %v3302, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3305 = stablehlo.convolution(%v3303, %v3304)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3306 = stablehlo.transpose %v3305, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3307 = stablehlo.constant dense<0.003125> : tensor<64x64x3x3xf32>
    %v3308 = stablehlo.multiply %v3306, %v3307 : tensor<64x64x3x3xf32>
    %v3309 = stablehlo.subtract %s1b2W2, %v3308 : tensor<64x64x3x3xf32>
    %v3310 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3311 = stablehlo.reshape %v172 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3312 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v3313 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3314 = stablehlo.reduce(%v3311 init: %v3310) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3315 = stablehlo.broadcast_in_dim %v3314, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3316 = stablehlo.divide %v3315, %v3312 : tensor<32x64x56x56xf32>
    %v3317 = stablehlo.subtract %v3311, %v3316 : tensor<32x64x56x56xf32>
    %v3318 = stablehlo.multiply %v3317, %v3317 : tensor<32x64x56x56xf32>
    %v3319 = stablehlo.reduce(%v3318 init: %v3310) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3320 = stablehlo.broadcast_in_dim %v3319, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3321 = stablehlo.divide %v3320, %v3312 : tensor<32x64x56x56xf32>
    %v3322 = stablehlo.add %v3321, %v3313 : tensor<32x64x56x56xf32>
    %v3323 = stablehlo.rsqrt %v3322 : tensor<32x64x56x56xf32>
    %v3324 = stablehlo.multiply %v3317, %v3323 : tensor<32x64x56x56xf32>
    %v3325 = stablehlo.reshape %v3190 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3326 = stablehlo.multiply %v3325, %v3324 : tensor<32x64x56x56xf32>
    %v3327 = stablehlo.reduce(%v3326 init: %v3310) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3328 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3329 = stablehlo.multiply %v3327, %v3328 : tensor<64xf32>
    %v3330 = stablehlo.subtract %s1b2g2, %v3329 : tensor<64xf32>
    %v3331 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3332 = stablehlo.reshape %v3190 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3333 = stablehlo.reduce(%v3332 init: %v3331) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3334 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3335 = stablehlo.multiply %v3333, %v3334 : tensor<64xf32>
    %v3336 = stablehlo.subtract %s1b2bt2, %v3335 : tensor<64xf32>
    %v3337 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v3338 = stablehlo.compare GT, %v138, %v3337 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v3339 = stablehlo.select %v3338, %v3264, %v3337 : tensor<32x200704xi1>, tensor<32x200704xf32>
    %v3340 = stablehlo.reshape %v3339 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3341 = stablehlo.reshape %v117 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3342 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3343 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v3344 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3345 = stablehlo.reduce(%v3341 init: %v3342) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3346 = stablehlo.broadcast_in_dim %v3345, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3347 = stablehlo.divide %v3346, %v3343 : tensor<32x64x56x56xf32>
    %v3348 = stablehlo.subtract %v3341, %v3347 : tensor<32x64x56x56xf32>
    %v3349 = stablehlo.multiply %v3348, %v3348 : tensor<32x64x56x56xf32>
    %v3350 = stablehlo.reduce(%v3349 init: %v3342) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3351 = stablehlo.broadcast_in_dim %v3350, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3352 = stablehlo.divide %v3351, %v3343 : tensor<32x64x56x56xf32>
    %v3353 = stablehlo.add %v3352, %v3344 : tensor<32x64x56x56xf32>
    %v3354 = stablehlo.rsqrt %v3353 : tensor<32x64x56x56xf32>
    %v3355 = stablehlo.multiply %v3348, %v3354 : tensor<32x64x56x56xf32>
    %v3356 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3357 = stablehlo.multiply %v3356, %v3340 : tensor<32x64x56x56xf32>
    %v3358 = stablehlo.reduce(%v3357 init: %v3342) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3359 = stablehlo.broadcast_in_dim %v3358, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3360 = stablehlo.multiply %v3355, %v3357 : tensor<32x64x56x56xf32>
    %v3361 = stablehlo.reduce(%v3360 init: %v3342) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3362 = stablehlo.broadcast_in_dim %v3361, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3363 = stablehlo.multiply %v3357, %v3343 : tensor<32x64x56x56xf32>
    %v3364 = stablehlo.subtract %v3363, %v3359 : tensor<32x64x56x56xf32>
    %v3365 = stablehlo.multiply %v3355, %v3362 : tensor<32x64x56x56xf32>
    %v3366 = stablehlo.subtract %v3364, %v3365 : tensor<32x64x56x56xf32>
    %v3367 = stablehlo.divide %v3354, %v3343 : tensor<32x64x56x56xf32>
    %v3368 = stablehlo.multiply %v3367, %v3366 : tensor<32x64x56x56xf32>
    %v3369 = stablehlo.reshape %v3368 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3370 = stablehlo.reshape %v3369 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3371 = stablehlo.transpose %s1b1W2, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3372 = stablehlo.reverse %v3371, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3373 = stablehlo.convolution(%v3370, %v3372)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3374 = stablehlo.reshape %v3373 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3375 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v3376 = stablehlo.compare GT, %v110, %v3375 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v3377 = stablehlo.select %v3376, %v3374, %v3375 : tensor<32x200704xi1>, tensor<32x200704xf32>
    %v3378 = stablehlo.reshape %v3377 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3379 = stablehlo.reshape %v90 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3380 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3381 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v3382 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3383 = stablehlo.reduce(%v3379 init: %v3380) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3384 = stablehlo.broadcast_in_dim %v3383, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3385 = stablehlo.divide %v3384, %v3381 : tensor<32x64x56x56xf32>
    %v3386 = stablehlo.subtract %v3379, %v3385 : tensor<32x64x56x56xf32>
    %v3387 = stablehlo.multiply %v3386, %v3386 : tensor<32x64x56x56xf32>
    %v3388 = stablehlo.reduce(%v3387 init: %v3380) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3389 = stablehlo.broadcast_in_dim %v3388, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3390 = stablehlo.divide %v3389, %v3381 : tensor<32x64x56x56xf32>
    %v3391 = stablehlo.add %v3390, %v3382 : tensor<32x64x56x56xf32>
    %v3392 = stablehlo.rsqrt %v3391 : tensor<32x64x56x56xf32>
    %v3393 = stablehlo.multiply %v3386, %v3392 : tensor<32x64x56x56xf32>
    %v3394 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3395 = stablehlo.multiply %v3394, %v3378 : tensor<32x64x56x56xf32>
    %v3396 = stablehlo.reduce(%v3395 init: %v3380) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3397 = stablehlo.broadcast_in_dim %v3396, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3398 = stablehlo.multiply %v3393, %v3395 : tensor<32x64x56x56xf32>
    %v3399 = stablehlo.reduce(%v3398 init: %v3380) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3400 = stablehlo.broadcast_in_dim %v3399, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3401 = stablehlo.multiply %v3395, %v3381 : tensor<32x64x56x56xf32>
    %v3402 = stablehlo.subtract %v3401, %v3397 : tensor<32x64x56x56xf32>
    %v3403 = stablehlo.multiply %v3393, %v3400 : tensor<32x64x56x56xf32>
    %v3404 = stablehlo.subtract %v3402, %v3403 : tensor<32x64x56x56xf32>
    %v3405 = stablehlo.divide %v3392, %v3381 : tensor<32x64x56x56xf32>
    %v3406 = stablehlo.multiply %v3405, %v3404 : tensor<32x64x56x56xf32>
    %v3407 = stablehlo.reshape %v3406 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3408 = stablehlo.reshape %v3407 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3409 = stablehlo.transpose %s1b1W1, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3410 = stablehlo.reverse %v3409, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3411 = stablehlo.convolution(%v3408, %v3410)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3412 = stablehlo.reshape %v3411 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3413 = stablehlo.add %v3412, %v3339 : tensor<32x200704xf32>
    %v3414 = stablehlo.reshape %v85 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3415 = stablehlo.reshape %v3407 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3416 = stablehlo.transpose %v3414, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3417 = stablehlo.transpose %v3415, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3418 = stablehlo.convolution(%v3416, %v3417)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3419 = stablehlo.transpose %v3418, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3420 = stablehlo.constant dense<0.003125> : tensor<64x64x3x3xf32>
    %v3421 = stablehlo.multiply %v3419, %v3420 : tensor<64x64x3x3xf32>
    %v3422 = stablehlo.subtract %s1b1W1, %v3421 : tensor<64x64x3x3xf32>
    %v3423 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3424 = stablehlo.reshape %v90 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3425 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v3426 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3427 = stablehlo.reduce(%v3424 init: %v3423) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3428 = stablehlo.broadcast_in_dim %v3427, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3429 = stablehlo.divide %v3428, %v3425 : tensor<32x64x56x56xf32>
    %v3430 = stablehlo.subtract %v3424, %v3429 : tensor<32x64x56x56xf32>
    %v3431 = stablehlo.multiply %v3430, %v3430 : tensor<32x64x56x56xf32>
    %v3432 = stablehlo.reduce(%v3431 init: %v3423) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3433 = stablehlo.broadcast_in_dim %v3432, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3434 = stablehlo.divide %v3433, %v3425 : tensor<32x64x56x56xf32>
    %v3435 = stablehlo.add %v3434, %v3426 : tensor<32x64x56x56xf32>
    %v3436 = stablehlo.rsqrt %v3435 : tensor<32x64x56x56xf32>
    %v3437 = stablehlo.multiply %v3430, %v3436 : tensor<32x64x56x56xf32>
    %v3438 = stablehlo.reshape %v3377 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3439 = stablehlo.multiply %v3438, %v3437 : tensor<32x64x56x56xf32>
    %v3440 = stablehlo.reduce(%v3439 init: %v3423) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3441 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3442 = stablehlo.multiply %v3440, %v3441 : tensor<64xf32>
    %v3443 = stablehlo.subtract %s1b1g1, %v3442 : tensor<64xf32>
    %v3444 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3445 = stablehlo.reshape %v3377 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3446 = stablehlo.reduce(%v3445 init: %v3444) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3447 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3448 = stablehlo.multiply %v3446, %v3447 : tensor<64xf32>
    %v3449 = stablehlo.subtract %s1b1bt1, %v3448 : tensor<64xf32>
    %v3450 = stablehlo.reshape %v112 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3451 = stablehlo.reshape %v3369 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3452 = stablehlo.transpose %v3450, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3453 = stablehlo.transpose %v3451, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3454 = stablehlo.convolution(%v3452, %v3453)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3455 = stablehlo.transpose %v3454, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3456 = stablehlo.constant dense<0.003125> : tensor<64x64x3x3xf32>
    %v3457 = stablehlo.multiply %v3455, %v3456 : tensor<64x64x3x3xf32>
    %v3458 = stablehlo.subtract %s1b1W2, %v3457 : tensor<64x64x3x3xf32>
    %v3459 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3460 = stablehlo.reshape %v117 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3461 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v3462 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3463 = stablehlo.reduce(%v3460 init: %v3459) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3464 = stablehlo.broadcast_in_dim %v3463, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3465 = stablehlo.divide %v3464, %v3461 : tensor<32x64x56x56xf32>
    %v3466 = stablehlo.subtract %v3460, %v3465 : tensor<32x64x56x56xf32>
    %v3467 = stablehlo.multiply %v3466, %v3466 : tensor<32x64x56x56xf32>
    %v3468 = stablehlo.reduce(%v3467 init: %v3459) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3469 = stablehlo.broadcast_in_dim %v3468, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3470 = stablehlo.divide %v3469, %v3461 : tensor<32x64x56x56xf32>
    %v3471 = stablehlo.add %v3470, %v3462 : tensor<32x64x56x56xf32>
    %v3472 = stablehlo.rsqrt %v3471 : tensor<32x64x56x56xf32>
    %v3473 = stablehlo.multiply %v3466, %v3472 : tensor<32x64x56x56xf32>
    %v3474 = stablehlo.reshape %v3339 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3475 = stablehlo.multiply %v3474, %v3473 : tensor<32x64x56x56xf32>
    %v3476 = stablehlo.reduce(%v3475 init: %v3459) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3477 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3478 = stablehlo.multiply %v3476, %v3477 : tensor<64xf32>
    %v3479 = stablehlo.subtract %s1b1g2, %v3478 : tensor<64xf32>
    %v3480 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3481 = stablehlo.reshape %v3339 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3482 = stablehlo.reduce(%v3481 init: %v3480) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3483 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3484 = stablehlo.multiply %v3482, %v3483 : tensor<64xf32>
    %v3485 = stablehlo.subtract %s1b1bt2, %v3484 : tensor<64xf32>
    %v3486 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v3487 = stablehlo.compare GT, %v83, %v3486 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v3488 = stablehlo.select %v3487, %v3413, %v3486 : tensor<32x200704xi1>, tensor<32x200704xf32>
    %v3489 = stablehlo.reshape %v3488 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3490 = stablehlo.reshape %v62 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3491 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3492 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v3493 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3494 = stablehlo.reduce(%v3490 init: %v3491) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3495 = stablehlo.broadcast_in_dim %v3494, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3496 = stablehlo.divide %v3495, %v3492 : tensor<32x64x56x56xf32>
    %v3497 = stablehlo.subtract %v3490, %v3496 : tensor<32x64x56x56xf32>
    %v3498 = stablehlo.multiply %v3497, %v3497 : tensor<32x64x56x56xf32>
    %v3499 = stablehlo.reduce(%v3498 init: %v3491) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3500 = stablehlo.broadcast_in_dim %v3499, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3501 = stablehlo.divide %v3500, %v3492 : tensor<32x64x56x56xf32>
    %v3502 = stablehlo.add %v3501, %v3493 : tensor<32x64x56x56xf32>
    %v3503 = stablehlo.rsqrt %v3502 : tensor<32x64x56x56xf32>
    %v3504 = stablehlo.multiply %v3497, %v3503 : tensor<32x64x56x56xf32>
    %v3505 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3506 = stablehlo.multiply %v3505, %v3489 : tensor<32x64x56x56xf32>
    %v3507 = stablehlo.reduce(%v3506 init: %v3491) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3508 = stablehlo.broadcast_in_dim %v3507, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3509 = stablehlo.multiply %v3504, %v3506 : tensor<32x64x56x56xf32>
    %v3510 = stablehlo.reduce(%v3509 init: %v3491) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3511 = stablehlo.broadcast_in_dim %v3510, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3512 = stablehlo.multiply %v3506, %v3492 : tensor<32x64x56x56xf32>
    %v3513 = stablehlo.subtract %v3512, %v3508 : tensor<32x64x56x56xf32>
    %v3514 = stablehlo.multiply %v3504, %v3511 : tensor<32x64x56x56xf32>
    %v3515 = stablehlo.subtract %v3513, %v3514 : tensor<32x64x56x56xf32>
    %v3516 = stablehlo.divide %v3503, %v3492 : tensor<32x64x56x56xf32>
    %v3517 = stablehlo.multiply %v3516, %v3515 : tensor<32x64x56x56xf32>
    %v3518 = stablehlo.reshape %v3517 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3519 = stablehlo.reshape %v3518 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3520 = stablehlo.transpose %s1b0W2, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3521 = stablehlo.reverse %v3520, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3522 = stablehlo.convolution(%v3519, %v3521)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3523 = stablehlo.reshape %v3522 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3524 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v3525 = stablehlo.compare GT, %v55, %v3524 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v3526 = stablehlo.select %v3525, %v3523, %v3524 : tensor<32x200704xi1>, tensor<32x200704xf32>
    %v3527 = stablehlo.reshape %v3526 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3528 = stablehlo.reshape %v35 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3529 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3530 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v3531 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3532 = stablehlo.reduce(%v3528 init: %v3529) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3533 = stablehlo.broadcast_in_dim %v3532, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3534 = stablehlo.divide %v3533, %v3530 : tensor<32x64x56x56xf32>
    %v3535 = stablehlo.subtract %v3528, %v3534 : tensor<32x64x56x56xf32>
    %v3536 = stablehlo.multiply %v3535, %v3535 : tensor<32x64x56x56xf32>
    %v3537 = stablehlo.reduce(%v3536 init: %v3529) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3538 = stablehlo.broadcast_in_dim %v3537, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3539 = stablehlo.divide %v3538, %v3530 : tensor<32x64x56x56xf32>
    %v3540 = stablehlo.add %v3539, %v3531 : tensor<32x64x56x56xf32>
    %v3541 = stablehlo.rsqrt %v3540 : tensor<32x64x56x56xf32>
    %v3542 = stablehlo.multiply %v3535, %v3541 : tensor<32x64x56x56xf32>
    %v3543 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3544 = stablehlo.multiply %v3543, %v3527 : tensor<32x64x56x56xf32>
    %v3545 = stablehlo.reduce(%v3544 init: %v3529) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3546 = stablehlo.broadcast_in_dim %v3545, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3547 = stablehlo.multiply %v3542, %v3544 : tensor<32x64x56x56xf32>
    %v3548 = stablehlo.reduce(%v3547 init: %v3529) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3549 = stablehlo.broadcast_in_dim %v3548, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3550 = stablehlo.multiply %v3544, %v3530 : tensor<32x64x56x56xf32>
    %v3551 = stablehlo.subtract %v3550, %v3546 : tensor<32x64x56x56xf32>
    %v3552 = stablehlo.multiply %v3542, %v3549 : tensor<32x64x56x56xf32>
    %v3553 = stablehlo.subtract %v3551, %v3552 : tensor<32x64x56x56xf32>
    %v3554 = stablehlo.divide %v3541, %v3530 : tensor<32x64x56x56xf32>
    %v3555 = stablehlo.multiply %v3554, %v3553 : tensor<32x64x56x56xf32>
    %v3556 = stablehlo.reshape %v3555 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3557 = stablehlo.reshape %v3556 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3558 = stablehlo.transpose %s1b0W1, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3559 = stablehlo.reverse %v3558, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3560 = stablehlo.convolution(%v3557, %v3559)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3561 = stablehlo.reshape %v3560 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3562 = stablehlo.add %v3561, %v3488 : tensor<32x200704xf32>
    %v3563 = stablehlo.reshape %v30 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3564 = stablehlo.reshape %v3556 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3565 = stablehlo.transpose %v3563, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3566 = stablehlo.transpose %v3564, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3567 = stablehlo.convolution(%v3565, %v3566)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3568 = stablehlo.transpose %v3567, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3569 = stablehlo.constant dense<0.003125> : tensor<64x64x3x3xf32>
    %v3570 = stablehlo.multiply %v3568, %v3569 : tensor<64x64x3x3xf32>
    %v3571 = stablehlo.subtract %s1b0W1, %v3570 : tensor<64x64x3x3xf32>
    %v3572 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3573 = stablehlo.reshape %v35 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3574 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v3575 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3576 = stablehlo.reduce(%v3573 init: %v3572) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3577 = stablehlo.broadcast_in_dim %v3576, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3578 = stablehlo.divide %v3577, %v3574 : tensor<32x64x56x56xf32>
    %v3579 = stablehlo.subtract %v3573, %v3578 : tensor<32x64x56x56xf32>
    %v3580 = stablehlo.multiply %v3579, %v3579 : tensor<32x64x56x56xf32>
    %v3581 = stablehlo.reduce(%v3580 init: %v3572) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3582 = stablehlo.broadcast_in_dim %v3581, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3583 = stablehlo.divide %v3582, %v3574 : tensor<32x64x56x56xf32>
    %v3584 = stablehlo.add %v3583, %v3575 : tensor<32x64x56x56xf32>
    %v3585 = stablehlo.rsqrt %v3584 : tensor<32x64x56x56xf32>
    %v3586 = stablehlo.multiply %v3579, %v3585 : tensor<32x64x56x56xf32>
    %v3587 = stablehlo.reshape %v3526 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3588 = stablehlo.multiply %v3587, %v3586 : tensor<32x64x56x56xf32>
    %v3589 = stablehlo.reduce(%v3588 init: %v3572) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3590 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3591 = stablehlo.multiply %v3589, %v3590 : tensor<64xf32>
    %v3592 = stablehlo.subtract %s1b0g1, %v3591 : tensor<64xf32>
    %v3593 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3594 = stablehlo.reshape %v3526 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3595 = stablehlo.reduce(%v3594 init: %v3593) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3596 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3597 = stablehlo.multiply %v3595, %v3596 : tensor<64xf32>
    %v3598 = stablehlo.subtract %s1b0bt1, %v3597 : tensor<64xf32>
    %v3599 = stablehlo.reshape %v57 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3600 = stablehlo.reshape %v3518 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3601 = stablehlo.transpose %v3599, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3602 = stablehlo.transpose %v3600, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3603 = stablehlo.convolution(%v3601, %v3602)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3604 = stablehlo.transpose %v3603, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3605 = stablehlo.constant dense<0.003125> : tensor<64x64x3x3xf32>
    %v3606 = stablehlo.multiply %v3604, %v3605 : tensor<64x64x3x3xf32>
    %v3607 = stablehlo.subtract %s1b0W2, %v3606 : tensor<64x64x3x3xf32>
    %v3608 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3609 = stablehlo.reshape %v62 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3610 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v3611 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3612 = stablehlo.reduce(%v3609 init: %v3608) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3613 = stablehlo.broadcast_in_dim %v3612, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3614 = stablehlo.divide %v3613, %v3610 : tensor<32x64x56x56xf32>
    %v3615 = stablehlo.subtract %v3609, %v3614 : tensor<32x64x56x56xf32>
    %v3616 = stablehlo.multiply %v3615, %v3615 : tensor<32x64x56x56xf32>
    %v3617 = stablehlo.reduce(%v3616 init: %v3608) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3618 = stablehlo.broadcast_in_dim %v3617, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3619 = stablehlo.divide %v3618, %v3610 : tensor<32x64x56x56xf32>
    %v3620 = stablehlo.add %v3619, %v3611 : tensor<32x64x56x56xf32>
    %v3621 = stablehlo.rsqrt %v3620 : tensor<32x64x56x56xf32>
    %v3622 = stablehlo.multiply %v3615, %v3621 : tensor<32x64x56x56xf32>
    %v3623 = stablehlo.reshape %v3488 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3624 = stablehlo.multiply %v3623, %v3622 : tensor<32x64x56x56xf32>
    %v3625 = stablehlo.reduce(%v3624 init: %v3608) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3626 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3627 = stablehlo.multiply %v3625, %v3626 : tensor<64xf32>
    %v3628 = stablehlo.subtract %s1b0g2, %v3627 : tensor<64xf32>
    %v3629 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3630 = stablehlo.reshape %v3488 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3631 = stablehlo.reduce(%v3630 init: %v3629) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3632 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3633 = stablehlo.multiply %v3631, %v3632 : tensor<64xf32>
    %v3634 = stablehlo.subtract %s1b0bt2, %v3633 : tensor<64xf32>
    %v3635 = stablehlo.reshape %v26 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3636 = stablehlo.reshape %v3562 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3637 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3638 = "stablehlo.select_and_scatter"(%v3635, %v3636, %v3637) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<32x64x112x112xf32>, tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x112x112xf32>
    %v3639 = stablehlo.reshape %v3638 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v3640 = stablehlo.constant dense<0.0> : tensor<32x802816xf32>
    %v3641 = stablehlo.compare GT, %v24, %v3640 : (tensor<32x802816xf32>, tensor<32x802816xf32>) -> tensor<32x802816xi1>
    %v3642 = stablehlo.select %v3641, %v3639, %v3640 : tensor<32x802816xi1>, tensor<32x802816xf32>
    %v3643 = stablehlo.reshape %v3642 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3644 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3645 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3646 = stablehlo.constant dense<12544.0> : tensor<32x64x112x112xf32>
    %v3647 = stablehlo.constant dense<1.0e-05> : tensor<32x64x112x112xf32>
    %v3648 = stablehlo.reduce(%v3644 init: %v3645) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3649 = stablehlo.broadcast_in_dim %v3648, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v3650 = stablehlo.divide %v3649, %v3646 : tensor<32x64x112x112xf32>
    %v3651 = stablehlo.subtract %v3644, %v3650 : tensor<32x64x112x112xf32>
    %v3652 = stablehlo.multiply %v3651, %v3651 : tensor<32x64x112x112xf32>
    %v3653 = stablehlo.reduce(%v3652 init: %v3645) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3654 = stablehlo.broadcast_in_dim %v3653, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v3655 = stablehlo.divide %v3654, %v3646 : tensor<32x64x112x112xf32>
    %v3656 = stablehlo.add %v3655, %v3647 : tensor<32x64x112x112xf32>
    %v3657 = stablehlo.rsqrt %v3656 : tensor<32x64x112x112xf32>
    %v3658 = stablehlo.multiply %v3651, %v3657 : tensor<32x64x112x112xf32>
    %v3659 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3660 = stablehlo.multiply %v3659, %v3643 : tensor<32x64x112x112xf32>
    %v3661 = stablehlo.reduce(%v3660 init: %v3645) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3662 = stablehlo.broadcast_in_dim %v3661, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v3663 = stablehlo.multiply %v3658, %v3660 : tensor<32x64x112x112xf32>
    %v3664 = stablehlo.reduce(%v3663 init: %v3645) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3665 = stablehlo.broadcast_in_dim %v3664, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v3666 = stablehlo.multiply %v3660, %v3646 : tensor<32x64x112x112xf32>
    %v3667 = stablehlo.subtract %v3666, %v3662 : tensor<32x64x112x112xf32>
    %v3668 = stablehlo.multiply %v3658, %v3665 : tensor<32x64x112x112xf32>
    %v3669 = stablehlo.subtract %v3667, %v3668 : tensor<32x64x112x112xf32>
    %v3670 = stablehlo.divide %v3657, %v3646 : tensor<32x64x112x112xf32>
    %v3671 = stablehlo.multiply %v3670, %v3669 : tensor<32x64x112x112xf32>
    %v3672 = stablehlo.reshape %v3671 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v3673 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v3674 = stablehlo.reshape %v3672 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3675 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3676 = stablehlo.pad %v3674, %v3675, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64x224x224xf32>
    %v3677 = stablehlo.transpose %v3673, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %v3678 = stablehlo.transpose %v3676, dims = [1, 0, 2, 3] : (tensor<32x64x224x224xf32>) -> tensor<64x32x224x224xf32>
    %v3679 = stablehlo.convolution(%v3677, %v3678)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<64x32x224x224xf32>) -> tensor<3x64x7x7xf32>
    %v3680 = stablehlo.transpose %v3679, dims = [1, 0, 2, 3] : (tensor<3x64x7x7xf32>) -> tensor<64x3x7x7xf32>
    %v3681 = stablehlo.constant dense<0.003125> : tensor<64x3x7x7xf32>
    %v3682 = stablehlo.multiply %v3680, %v3681 : tensor<64x3x7x7xf32>
    %v3683 = stablehlo.subtract %sW, %v3682 : tensor<64x3x7x7xf32>
    %v3684 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3685 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3686 = stablehlo.constant dense<12544.0> : tensor<32x64x112x112xf32>
    %v3687 = stablehlo.constant dense<1.0e-05> : tensor<32x64x112x112xf32>
    %v3688 = stablehlo.reduce(%v3685 init: %v3684) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3689 = stablehlo.broadcast_in_dim %v3688, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v3690 = stablehlo.divide %v3689, %v3686 : tensor<32x64x112x112xf32>
    %v3691 = stablehlo.subtract %v3685, %v3690 : tensor<32x64x112x112xf32>
    %v3692 = stablehlo.multiply %v3691, %v3691 : tensor<32x64x112x112xf32>
    %v3693 = stablehlo.reduce(%v3692 init: %v3684) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3694 = stablehlo.broadcast_in_dim %v3693, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v3695 = stablehlo.divide %v3694, %v3686 : tensor<32x64x112x112xf32>
    %v3696 = stablehlo.add %v3695, %v3687 : tensor<32x64x112x112xf32>
    %v3697 = stablehlo.rsqrt %v3696 : tensor<32x64x112x112xf32>
    %v3698 = stablehlo.multiply %v3691, %v3697 : tensor<32x64x112x112xf32>
    %v3699 = stablehlo.reshape %v3642 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3700 = stablehlo.multiply %v3699, %v3698 : tensor<32x64x112x112xf32>
    %v3701 = stablehlo.reduce(%v3700 init: %v3684) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3702 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3703 = stablehlo.multiply %v3701, %v3702 : tensor<64xf32>
    %v3704 = stablehlo.subtract %sg, %v3703 : tensor<64xf32>
    %v3705 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3706 = stablehlo.reshape %v3642 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3707 = stablehlo.reduce(%v3706 init: %v3705) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3708 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3709 = stablehlo.multiply %v3707, %v3708 : tensor<64xf32>
    %v3710 = stablehlo.subtract %sbt, %v3709 : tensor<64xf32>
    return %v3683, %v3704, %v3710, %v3571, %v3592, %v3598, %v3607, %v3628, %v3634, %v3422, %v3443, %v3449, %v3458, %v3479, %v3485, %v3273, %v3294, %v3300, %v3309, %v3330, %v3336, %v3086, %v3107, %v3113, %v3122, %v3143, %v3149, %v3160, %v3181, %v3187, %v2896, %v2917, %v2923, %v2932, %v2953, %v2959, %v2747, %v2768, %v2774, %v2783, %v2804, %v2810, %v2598, %v2619, %v2625, %v2634, %v2655, %v2661, %v2411, %v2432, %v2438, %v2447, %v2468, %v2474, %v2485, %v2506, %v2512, %v2221, %v2242, %v2248, %v2257, %v2278, %v2284, %v2072, %v2093, %v2099, %v2108, %v2129, %v2135, %v1923, %v1944, %v1950, %v1959, %v1980, %v1986, %v1774, %v1795, %v1801, %v1810, %v1831, %v1837, %v1625, %v1646, %v1652, %v1661, %v1682, %v1688, %v1438, %v1459, %v1465, %v1474, %v1495, %v1501, %v1512, %v1533, %v1539, %v1248, %v1269, %v1275, %v1284, %v1305, %v1311, %v1099, %v1120, %v1126, %v1135, %v1156, %v1162, %v1008, %v1013 : tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>
  }
}
