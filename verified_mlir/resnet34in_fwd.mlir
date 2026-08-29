module @m {
  func.func @resnet34in_fwd(%x: tensor<256x150528xf32>, %sW: tensor<64x3x7x7xf32>, %sg: tensor<64xf32>, %sbt: tensor<64xf32>, %s1b0W1: tensor<64x64x3x3xf32>, %s1b0g1: tensor<64xf32>, %s1b0bt1: tensor<64xf32>, %s1b0W2: tensor<64x64x3x3xf32>, %s1b0g2: tensor<64xf32>, %s1b0bt2: tensor<64xf32>, %s1b1W1: tensor<64x64x3x3xf32>, %s1b1g1: tensor<64xf32>, %s1b1bt1: tensor<64xf32>, %s1b1W2: tensor<64x64x3x3xf32>, %s1b1g2: tensor<64xf32>, %s1b1bt2: tensor<64xf32>, %s1b2W1: tensor<64x64x3x3xf32>, %s1b2g1: tensor<64xf32>, %s1b2bt1: tensor<64xf32>, %s1b2W2: tensor<64x64x3x3xf32>, %s1b2g2: tensor<64xf32>, %s1b2bt2: tensor<64xf32>, %d2W1: tensor<128x64x3x3xf32>, %d2g1: tensor<128xf32>, %d2bt1: tensor<128xf32>, %d2W2: tensor<128x128x3x3xf32>, %d2g2: tensor<128xf32>, %d2bt2: tensor<128xf32>, %d2Wp: tensor<128x64x1x1xf32>, %d2gp: tensor<128xf32>, %d2btp: tensor<128xf32>, %s2b0W1: tensor<128x128x3x3xf32>, %s2b0g1: tensor<128xf32>, %s2b0bt1: tensor<128xf32>, %s2b0W2: tensor<128x128x3x3xf32>, %s2b0g2: tensor<128xf32>, %s2b0bt2: tensor<128xf32>, %s2b1W1: tensor<128x128x3x3xf32>, %s2b1g1: tensor<128xf32>, %s2b1bt1: tensor<128xf32>, %s2b1W2: tensor<128x128x3x3xf32>, %s2b1g2: tensor<128xf32>, %s2b1bt2: tensor<128xf32>, %s2b2W1: tensor<128x128x3x3xf32>, %s2b2g1: tensor<128xf32>, %s2b2bt1: tensor<128xf32>, %s2b2W2: tensor<128x128x3x3xf32>, %s2b2g2: tensor<128xf32>, %s2b2bt2: tensor<128xf32>, %d3W1: tensor<256x128x3x3xf32>, %d3g1: tensor<256xf32>, %d3bt1: tensor<256xf32>, %d3W2: tensor<256x256x3x3xf32>, %d3g2: tensor<256xf32>, %d3bt2: tensor<256xf32>, %d3Wp: tensor<256x128x1x1xf32>, %d3gp: tensor<256xf32>, %d3btp: tensor<256xf32>, %s3b0W1: tensor<256x256x3x3xf32>, %s3b0g1: tensor<256xf32>, %s3b0bt1: tensor<256xf32>, %s3b0W2: tensor<256x256x3x3xf32>, %s3b0g2: tensor<256xf32>, %s3b0bt2: tensor<256xf32>, %s3b1W1: tensor<256x256x3x3xf32>, %s3b1g1: tensor<256xf32>, %s3b1bt1: tensor<256xf32>, %s3b1W2: tensor<256x256x3x3xf32>, %s3b1g2: tensor<256xf32>, %s3b1bt2: tensor<256xf32>, %s3b2W1: tensor<256x256x3x3xf32>, %s3b2g1: tensor<256xf32>, %s3b2bt1: tensor<256xf32>, %s3b2W2: tensor<256x256x3x3xf32>, %s3b2g2: tensor<256xf32>, %s3b2bt2: tensor<256xf32>, %s3b3W1: tensor<256x256x3x3xf32>, %s3b3g1: tensor<256xf32>, %s3b3bt1: tensor<256xf32>, %s3b3W2: tensor<256x256x3x3xf32>, %s3b3g2: tensor<256xf32>, %s3b3bt2: tensor<256xf32>, %s3b4W1: tensor<256x256x3x3xf32>, %s3b4g1: tensor<256xf32>, %s3b4bt1: tensor<256xf32>, %s3b4W2: tensor<256x256x3x3xf32>, %s3b4g2: tensor<256xf32>, %s3b4bt2: tensor<256xf32>, %d4W1: tensor<512x256x3x3xf32>, %d4g1: tensor<512xf32>, %d4bt1: tensor<512xf32>, %d4W2: tensor<512x512x3x3xf32>, %d4g2: tensor<512xf32>, %d4bt2: tensor<512xf32>, %d4Wp: tensor<512x256x1x1xf32>, %d4gp: tensor<512xf32>, %d4btp: tensor<512xf32>, %s4b0W1: tensor<512x512x3x3xf32>, %s4b0g1: tensor<512xf32>, %s4b0bt1: tensor<512xf32>, %s4b0W2: tensor<512x512x3x3xf32>, %s4b0g2: tensor<512xf32>, %s4b0bt2: tensor<512xf32>, %s4b1W1: tensor<512x512x3x3xf32>, %s4b1g1: tensor<512xf32>, %s4b1bt1: tensor<512xf32>, %s4b1W2: tensor<512x512x3x3xf32>, %s4b1g2: tensor<512xf32>, %s4b1bt2: tensor<512xf32>, %Wd: tensor<512x1000xf32>, %bd: tensor<1000xf32>) -> tensor<256x1000xf32> {
    // ── ResNet-34 forward: every line is pretty(verified AST node) ──
    // §2l step B: the conv biases are gone from the signature (BN removes them; He et al.'s
    // `.convBn` has none). The proven conv ops still take a bias operand, so it is bound to a
    // zero constant here — same op, `bias = 0`, and `x + 0.0` is exact.
    %zb64 = stablehlo.constant dense<0.0> : tensor<64xf32>
    %zb128 = stablehlo.constant dense<0.0> : tensor<128xf32>
    %zb256 = stablehlo.constant dense<0.0> : tensor<256xf32>
    %zb512 = stablehlo.constant dense<0.0> : tensor<512xf32>
    %v0 = stablehlo.reshape %x : (tensor<256x150528xf32>) -> tensor<256x3x224x224xf32>
    %v1 = stablehlo.convolution(%v0, %sW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<256x64x112x112xf32>
    %v2 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<256x64x112x112xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<256x64x112x112xf32>) -> tensor<256x802816xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v6 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7 = stablehlo.constant dense<12544.0> : tensor<256x64x112x112xf32>
    %v8 = stablehlo.constant dense<1.0e-05> : tensor<256x64x112x112xf32>
    %v9 = stablehlo.reduce(%v5 init: %v6) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<256x64xf32>
    %v10 = stablehlo.broadcast_in_dim %v9, dims = [0, 1] : (tensor<256x64xf32>) -> tensor<256x64x112x112xf32>
    %v11 = stablehlo.divide %v10, %v7 : tensor<256x64x112x112xf32>
    %v12 = stablehlo.subtract %v5, %v11 : tensor<256x64x112x112xf32>
    %v13 = stablehlo.multiply %v12, %v12 : tensor<256x64x112x112xf32>
    %v14 = stablehlo.reduce(%v13 init: %v6) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<256x64xf32>
    %v15 = stablehlo.broadcast_in_dim %v14, dims = [0, 1] : (tensor<256x64xf32>) -> tensor<256x64x112x112xf32>
    %v16 = stablehlo.divide %v15, %v7 : tensor<256x64x112x112xf32>
    %v17 = stablehlo.add %v16, %v8 : tensor<256x64x112x112xf32>
    %v18 = stablehlo.rsqrt %v17 : tensor<256x64x112x112xf32>
    %v19 = stablehlo.multiply %v12, %v18 : tensor<256x64x112x112xf32>
    %v20 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v21 = stablehlo.broadcast_in_dim %sbt, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v22 = stablehlo.multiply %v19, %v20 : tensor<256x64x112x112xf32>
    %v23 = stablehlo.add %v22, %v21 : tensor<256x64x112x112xf32>
    %v24 = stablehlo.reshape %v23 : (tensor<256x64x112x112xf32>) -> tensor<256x802816xf32>
    %v25 = stablehlo.reshape %v24 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v26 = stablehlo.constant dense<0.0> : tensor<256x64x112x112xf32>
    %v27 = stablehlo.maximum %v25, %v26 : tensor<256x64x112x112xf32>
    %v28 = stablehlo.reshape %v27 : (tensor<256x64x112x112xf32>) -> tensor<256x802816xf32>
    %v29 = stablehlo.reshape %v28 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v30 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v31 = "stablehlo.reduce_window"(%v29, %v30) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>} : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<256x64x56x56xf32>
    %v32 = stablehlo.reshape %v31 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v33 = stablehlo.reshape %v32 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v34 = stablehlo.convolution(%v33, %s1b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v35 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v36 = stablehlo.add %v34, %v35 : tensor<256x64x56x56xf32>
    %v37 = stablehlo.reshape %v36 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v38 = stablehlo.reshape %v37 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v39 = stablehlo.constant dense<0.0> : tensor<f32>
    %v40 = stablehlo.constant dense<3136.0> : tensor<256x64x56x56xf32>
    %v41 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v42 = stablehlo.reduce(%v38 init: %v39) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<256x64xf32>
    %v43 = stablehlo.broadcast_in_dim %v42, dims = [0, 1] : (tensor<256x64xf32>) -> tensor<256x64x56x56xf32>
    %v44 = stablehlo.divide %v43, %v40 : tensor<256x64x56x56xf32>
    %v45 = stablehlo.subtract %v38, %v44 : tensor<256x64x56x56xf32>
    %v46 = stablehlo.multiply %v45, %v45 : tensor<256x64x56x56xf32>
    %v47 = stablehlo.reduce(%v46 init: %v39) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<256x64xf32>
    %v48 = stablehlo.broadcast_in_dim %v47, dims = [0, 1] : (tensor<256x64xf32>) -> tensor<256x64x56x56xf32>
    %v49 = stablehlo.divide %v48, %v40 : tensor<256x64x56x56xf32>
    %v50 = stablehlo.add %v49, %v41 : tensor<256x64x56x56xf32>
    %v51 = stablehlo.rsqrt %v50 : tensor<256x64x56x56xf32>
    %v52 = stablehlo.multiply %v45, %v51 : tensor<256x64x56x56xf32>
    %v53 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v54 = stablehlo.broadcast_in_dim %s1b0bt1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v55 = stablehlo.multiply %v52, %v53 : tensor<256x64x56x56xf32>
    %v56 = stablehlo.add %v55, %v54 : tensor<256x64x56x56xf32>
    %v57 = stablehlo.reshape %v56 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v59 = stablehlo.constant dense<0.0> : tensor<256x64x56x56xf32>
    %v60 = stablehlo.maximum %v58, %v59 : tensor<256x64x56x56xf32>
    %v61 = stablehlo.reshape %v60 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v62 = stablehlo.reshape %v61 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v63 = stablehlo.convolution(%v62, %s1b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v64 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v65 = stablehlo.add %v63, %v64 : tensor<256x64x56x56xf32>
    %v66 = stablehlo.reshape %v65 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v67 = stablehlo.reshape %v66 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v68 = stablehlo.constant dense<0.0> : tensor<f32>
    %v69 = stablehlo.constant dense<3136.0> : tensor<256x64x56x56xf32>
    %v70 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v71 = stablehlo.reduce(%v67 init: %v68) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<256x64xf32>
    %v72 = stablehlo.broadcast_in_dim %v71, dims = [0, 1] : (tensor<256x64xf32>) -> tensor<256x64x56x56xf32>
    %v73 = stablehlo.divide %v72, %v69 : tensor<256x64x56x56xf32>
    %v74 = stablehlo.subtract %v67, %v73 : tensor<256x64x56x56xf32>
    %v75 = stablehlo.multiply %v74, %v74 : tensor<256x64x56x56xf32>
    %v76 = stablehlo.reduce(%v75 init: %v68) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<256x64xf32>
    %v77 = stablehlo.broadcast_in_dim %v76, dims = [0, 1] : (tensor<256x64xf32>) -> tensor<256x64x56x56xf32>
    %v78 = stablehlo.divide %v77, %v69 : tensor<256x64x56x56xf32>
    %v79 = stablehlo.add %v78, %v70 : tensor<256x64x56x56xf32>
    %v80 = stablehlo.rsqrt %v79 : tensor<256x64x56x56xf32>
    %v81 = stablehlo.multiply %v74, %v80 : tensor<256x64x56x56xf32>
    %v82 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v83 = stablehlo.broadcast_in_dim %s1b0bt2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v84 = stablehlo.multiply %v81, %v82 : tensor<256x64x56x56xf32>
    %v85 = stablehlo.add %v84, %v83 : tensor<256x64x56x56xf32>
    %v86 = stablehlo.reshape %v85 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v87 = stablehlo.reshape %v86 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v88 = stablehlo.reshape %v32 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v89 = stablehlo.add %v87, %v88 : tensor<256x64x56x56xf32>
    %v90 = stablehlo.reshape %v89 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v91 = stablehlo.reshape %v90 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v92 = stablehlo.constant dense<0.0> : tensor<256x64x56x56xf32>
    %v93 = stablehlo.maximum %v91, %v92 : tensor<256x64x56x56xf32>
    %v94 = stablehlo.reshape %v93 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v95 = stablehlo.reshape %v94 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v96 = stablehlo.convolution(%v95, %s1b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v97 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v98 = stablehlo.add %v96, %v97 : tensor<256x64x56x56xf32>
    %v99 = stablehlo.reshape %v98 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v100 = stablehlo.reshape %v99 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v101 = stablehlo.constant dense<0.0> : tensor<f32>
    %v102 = stablehlo.constant dense<3136.0> : tensor<256x64x56x56xf32>
    %v103 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v104 = stablehlo.reduce(%v100 init: %v101) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<256x64xf32>
    %v105 = stablehlo.broadcast_in_dim %v104, dims = [0, 1] : (tensor<256x64xf32>) -> tensor<256x64x56x56xf32>
    %v106 = stablehlo.divide %v105, %v102 : tensor<256x64x56x56xf32>
    %v107 = stablehlo.subtract %v100, %v106 : tensor<256x64x56x56xf32>
    %v108 = stablehlo.multiply %v107, %v107 : tensor<256x64x56x56xf32>
    %v109 = stablehlo.reduce(%v108 init: %v101) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<256x64xf32>
    %v110 = stablehlo.broadcast_in_dim %v109, dims = [0, 1] : (tensor<256x64xf32>) -> tensor<256x64x56x56xf32>
    %v111 = stablehlo.divide %v110, %v102 : tensor<256x64x56x56xf32>
    %v112 = stablehlo.add %v111, %v103 : tensor<256x64x56x56xf32>
    %v113 = stablehlo.rsqrt %v112 : tensor<256x64x56x56xf32>
    %v114 = stablehlo.multiply %v107, %v113 : tensor<256x64x56x56xf32>
    %v115 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v116 = stablehlo.broadcast_in_dim %s1b1bt1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v117 = stablehlo.multiply %v114, %v115 : tensor<256x64x56x56xf32>
    %v118 = stablehlo.add %v117, %v116 : tensor<256x64x56x56xf32>
    %v119 = stablehlo.reshape %v118 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v120 = stablehlo.reshape %v119 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v121 = stablehlo.constant dense<0.0> : tensor<256x64x56x56xf32>
    %v122 = stablehlo.maximum %v120, %v121 : tensor<256x64x56x56xf32>
    %v123 = stablehlo.reshape %v122 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v124 = stablehlo.reshape %v123 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v125 = stablehlo.convolution(%v124, %s1b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v126 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v127 = stablehlo.add %v125, %v126 : tensor<256x64x56x56xf32>
    %v128 = stablehlo.reshape %v127 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v129 = stablehlo.reshape %v128 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v130 = stablehlo.constant dense<0.0> : tensor<f32>
    %v131 = stablehlo.constant dense<3136.0> : tensor<256x64x56x56xf32>
    %v132 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v133 = stablehlo.reduce(%v129 init: %v130) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<256x64xf32>
    %v134 = stablehlo.broadcast_in_dim %v133, dims = [0, 1] : (tensor<256x64xf32>) -> tensor<256x64x56x56xf32>
    %v135 = stablehlo.divide %v134, %v131 : tensor<256x64x56x56xf32>
    %v136 = stablehlo.subtract %v129, %v135 : tensor<256x64x56x56xf32>
    %v137 = stablehlo.multiply %v136, %v136 : tensor<256x64x56x56xf32>
    %v138 = stablehlo.reduce(%v137 init: %v130) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<256x64xf32>
    %v139 = stablehlo.broadcast_in_dim %v138, dims = [0, 1] : (tensor<256x64xf32>) -> tensor<256x64x56x56xf32>
    %v140 = stablehlo.divide %v139, %v131 : tensor<256x64x56x56xf32>
    %v141 = stablehlo.add %v140, %v132 : tensor<256x64x56x56xf32>
    %v142 = stablehlo.rsqrt %v141 : tensor<256x64x56x56xf32>
    %v143 = stablehlo.multiply %v136, %v142 : tensor<256x64x56x56xf32>
    %v144 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v145 = stablehlo.broadcast_in_dim %s1b1bt2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v146 = stablehlo.multiply %v143, %v144 : tensor<256x64x56x56xf32>
    %v147 = stablehlo.add %v146, %v145 : tensor<256x64x56x56xf32>
    %v148 = stablehlo.reshape %v147 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v149 = stablehlo.reshape %v148 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v150 = stablehlo.reshape %v94 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v151 = stablehlo.add %v149, %v150 : tensor<256x64x56x56xf32>
    %v152 = stablehlo.reshape %v151 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v153 = stablehlo.reshape %v152 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v154 = stablehlo.constant dense<0.0> : tensor<256x64x56x56xf32>
    %v155 = stablehlo.maximum %v153, %v154 : tensor<256x64x56x56xf32>
    %v156 = stablehlo.reshape %v155 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v157 = stablehlo.reshape %v156 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v158 = stablehlo.convolution(%v157, %s1b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v159 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v160 = stablehlo.add %v158, %v159 : tensor<256x64x56x56xf32>
    %v161 = stablehlo.reshape %v160 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v162 = stablehlo.reshape %v161 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v163 = stablehlo.constant dense<0.0> : tensor<f32>
    %v164 = stablehlo.constant dense<3136.0> : tensor<256x64x56x56xf32>
    %v165 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v166 = stablehlo.reduce(%v162 init: %v163) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<256x64xf32>
    %v167 = stablehlo.broadcast_in_dim %v166, dims = [0, 1] : (tensor<256x64xf32>) -> tensor<256x64x56x56xf32>
    %v168 = stablehlo.divide %v167, %v164 : tensor<256x64x56x56xf32>
    %v169 = stablehlo.subtract %v162, %v168 : tensor<256x64x56x56xf32>
    %v170 = stablehlo.multiply %v169, %v169 : tensor<256x64x56x56xf32>
    %v171 = stablehlo.reduce(%v170 init: %v163) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<256x64xf32>
    %v172 = stablehlo.broadcast_in_dim %v171, dims = [0, 1] : (tensor<256x64xf32>) -> tensor<256x64x56x56xf32>
    %v173 = stablehlo.divide %v172, %v164 : tensor<256x64x56x56xf32>
    %v174 = stablehlo.add %v173, %v165 : tensor<256x64x56x56xf32>
    %v175 = stablehlo.rsqrt %v174 : tensor<256x64x56x56xf32>
    %v176 = stablehlo.multiply %v169, %v175 : tensor<256x64x56x56xf32>
    %v177 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v178 = stablehlo.broadcast_in_dim %s1b2bt1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v179 = stablehlo.multiply %v176, %v177 : tensor<256x64x56x56xf32>
    %v180 = stablehlo.add %v179, %v178 : tensor<256x64x56x56xf32>
    %v181 = stablehlo.reshape %v180 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v182 = stablehlo.reshape %v181 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v183 = stablehlo.constant dense<0.0> : tensor<256x64x56x56xf32>
    %v184 = stablehlo.maximum %v182, %v183 : tensor<256x64x56x56xf32>
    %v185 = stablehlo.reshape %v184 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v186 = stablehlo.reshape %v185 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v187 = stablehlo.convolution(%v186, %s1b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v188 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v189 = stablehlo.add %v187, %v188 : tensor<256x64x56x56xf32>
    %v190 = stablehlo.reshape %v189 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v191 = stablehlo.reshape %v190 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v192 = stablehlo.constant dense<0.0> : tensor<f32>
    %v193 = stablehlo.constant dense<3136.0> : tensor<256x64x56x56xf32>
    %v194 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v195 = stablehlo.reduce(%v191 init: %v192) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<256x64xf32>
    %v196 = stablehlo.broadcast_in_dim %v195, dims = [0, 1] : (tensor<256x64xf32>) -> tensor<256x64x56x56xf32>
    %v197 = stablehlo.divide %v196, %v193 : tensor<256x64x56x56xf32>
    %v198 = stablehlo.subtract %v191, %v197 : tensor<256x64x56x56xf32>
    %v199 = stablehlo.multiply %v198, %v198 : tensor<256x64x56x56xf32>
    %v200 = stablehlo.reduce(%v199 init: %v192) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<256x64xf32>
    %v201 = stablehlo.broadcast_in_dim %v200, dims = [0, 1] : (tensor<256x64xf32>) -> tensor<256x64x56x56xf32>
    %v202 = stablehlo.divide %v201, %v193 : tensor<256x64x56x56xf32>
    %v203 = stablehlo.add %v202, %v194 : tensor<256x64x56x56xf32>
    %v204 = stablehlo.rsqrt %v203 : tensor<256x64x56x56xf32>
    %v205 = stablehlo.multiply %v198, %v204 : tensor<256x64x56x56xf32>
    %v206 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v207 = stablehlo.broadcast_in_dim %s1b2bt2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v208 = stablehlo.multiply %v205, %v206 : tensor<256x64x56x56xf32>
    %v209 = stablehlo.add %v208, %v207 : tensor<256x64x56x56xf32>
    %v210 = stablehlo.reshape %v209 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v211 = stablehlo.reshape %v210 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v212 = stablehlo.reshape %v156 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v213 = stablehlo.add %v211, %v212 : tensor<256x64x56x56xf32>
    %v214 = stablehlo.reshape %v213 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v215 = stablehlo.reshape %v214 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v216 = stablehlo.constant dense<0.0> : tensor<256x64x56x56xf32>
    %v217 = stablehlo.maximum %v215, %v216 : tensor<256x64x56x56xf32>
    %v218 = stablehlo.reshape %v217 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v219 = stablehlo.reshape %v218 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v220 = stablehlo.convolution(%v219, %d2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<128x64x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v221 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v222 = stablehlo.add %v220, %v221 : tensor<256x128x28x28xf32>
    %v223 = stablehlo.reshape %v222 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v224 = stablehlo.reshape %v223 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v225 = stablehlo.constant dense<0.0> : tensor<f32>
    %v226 = stablehlo.constant dense<784.0> : tensor<256x128x28x28xf32>
    %v227 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v228 = stablehlo.reduce(%v224 init: %v225) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128xf32>
    %v229 = stablehlo.broadcast_in_dim %v228, dims = [0, 1] : (tensor<256x128xf32>) -> tensor<256x128x28x28xf32>
    %v230 = stablehlo.divide %v229, %v226 : tensor<256x128x28x28xf32>
    %v231 = stablehlo.subtract %v224, %v230 : tensor<256x128x28x28xf32>
    %v232 = stablehlo.multiply %v231, %v231 : tensor<256x128x28x28xf32>
    %v233 = stablehlo.reduce(%v232 init: %v225) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128xf32>
    %v234 = stablehlo.broadcast_in_dim %v233, dims = [0, 1] : (tensor<256x128xf32>) -> tensor<256x128x28x28xf32>
    %v235 = stablehlo.divide %v234, %v226 : tensor<256x128x28x28xf32>
    %v236 = stablehlo.add %v235, %v227 : tensor<256x128x28x28xf32>
    %v237 = stablehlo.rsqrt %v236 : tensor<256x128x28x28xf32>
    %v238 = stablehlo.multiply %v231, %v237 : tensor<256x128x28x28xf32>
    %v239 = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v240 = stablehlo.broadcast_in_dim %d2bt1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v241 = stablehlo.multiply %v238, %v239 : tensor<256x128x28x28xf32>
    %v242 = stablehlo.add %v241, %v240 : tensor<256x128x28x28xf32>
    %v243 = stablehlo.reshape %v242 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v244 = stablehlo.reshape %v243 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v245 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v246 = stablehlo.maximum %v244, %v245 : tensor<256x128x28x28xf32>
    %v247 = stablehlo.reshape %v246 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v248 = stablehlo.reshape %v247 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v249 = stablehlo.convolution(%v248, %d2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v250 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v251 = stablehlo.add %v249, %v250 : tensor<256x128x28x28xf32>
    %v252 = stablehlo.reshape %v251 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v253 = stablehlo.reshape %v252 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v254 = stablehlo.constant dense<0.0> : tensor<f32>
    %v255 = stablehlo.constant dense<784.0> : tensor<256x128x28x28xf32>
    %v256 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v257 = stablehlo.reduce(%v253 init: %v254) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128xf32>
    %v258 = stablehlo.broadcast_in_dim %v257, dims = [0, 1] : (tensor<256x128xf32>) -> tensor<256x128x28x28xf32>
    %v259 = stablehlo.divide %v258, %v255 : tensor<256x128x28x28xf32>
    %v260 = stablehlo.subtract %v253, %v259 : tensor<256x128x28x28xf32>
    %v261 = stablehlo.multiply %v260, %v260 : tensor<256x128x28x28xf32>
    %v262 = stablehlo.reduce(%v261 init: %v254) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128xf32>
    %v263 = stablehlo.broadcast_in_dim %v262, dims = [0, 1] : (tensor<256x128xf32>) -> tensor<256x128x28x28xf32>
    %v264 = stablehlo.divide %v263, %v255 : tensor<256x128x28x28xf32>
    %v265 = stablehlo.add %v264, %v256 : tensor<256x128x28x28xf32>
    %v266 = stablehlo.rsqrt %v265 : tensor<256x128x28x28xf32>
    %v267 = stablehlo.multiply %v260, %v266 : tensor<256x128x28x28xf32>
    %v268 = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v269 = stablehlo.broadcast_in_dim %d2bt2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v270 = stablehlo.multiply %v267, %v268 : tensor<256x128x28x28xf32>
    %v271 = stablehlo.add %v270, %v269 : tensor<256x128x28x28xf32>
    %v272 = stablehlo.reshape %v271 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v273 = stablehlo.reshape %v218 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v274 = stablehlo.convolution(%v273, %d2Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<128x64x1x1xf32>) -> tensor<256x128x28x28xf32>
    %v275 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v276 = stablehlo.add %v274, %v275 : tensor<256x128x28x28xf32>
    %v277 = stablehlo.reshape %v276 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v278 = stablehlo.reshape %v277 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v279 = stablehlo.constant dense<0.0> : tensor<f32>
    %v280 = stablehlo.constant dense<784.0> : tensor<256x128x28x28xf32>
    %v281 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v282 = stablehlo.reduce(%v278 init: %v279) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128xf32>
    %v283 = stablehlo.broadcast_in_dim %v282, dims = [0, 1] : (tensor<256x128xf32>) -> tensor<256x128x28x28xf32>
    %v284 = stablehlo.divide %v283, %v280 : tensor<256x128x28x28xf32>
    %v285 = stablehlo.subtract %v278, %v284 : tensor<256x128x28x28xf32>
    %v286 = stablehlo.multiply %v285, %v285 : tensor<256x128x28x28xf32>
    %v287 = stablehlo.reduce(%v286 init: %v279) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128xf32>
    %v288 = stablehlo.broadcast_in_dim %v287, dims = [0, 1] : (tensor<256x128xf32>) -> tensor<256x128x28x28xf32>
    %v289 = stablehlo.divide %v288, %v280 : tensor<256x128x28x28xf32>
    %v290 = stablehlo.add %v289, %v281 : tensor<256x128x28x28xf32>
    %v291 = stablehlo.rsqrt %v290 : tensor<256x128x28x28xf32>
    %v292 = stablehlo.multiply %v285, %v291 : tensor<256x128x28x28xf32>
    %v293 = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v294 = stablehlo.broadcast_in_dim %d2btp, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v295 = stablehlo.multiply %v292, %v293 : tensor<256x128x28x28xf32>
    %v296 = stablehlo.add %v295, %v294 : tensor<256x128x28x28xf32>
    %v297 = stablehlo.reshape %v296 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v298 = stablehlo.reshape %v272 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v299 = stablehlo.reshape %v297 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v300 = stablehlo.add %v298, %v299 : tensor<256x128x28x28xf32>
    %v301 = stablehlo.reshape %v300 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v302 = stablehlo.reshape %v301 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v303 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v304 = stablehlo.maximum %v302, %v303 : tensor<256x128x28x28xf32>
    %v305 = stablehlo.reshape %v304 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v306 = stablehlo.reshape %v305 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v307 = stablehlo.convolution(%v306, %s2b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v308 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v309 = stablehlo.add %v307, %v308 : tensor<256x128x28x28xf32>
    %v310 = stablehlo.reshape %v309 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v311 = stablehlo.reshape %v310 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v312 = stablehlo.constant dense<0.0> : tensor<f32>
    %v313 = stablehlo.constant dense<784.0> : tensor<256x128x28x28xf32>
    %v314 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v315 = stablehlo.reduce(%v311 init: %v312) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128xf32>
    %v316 = stablehlo.broadcast_in_dim %v315, dims = [0, 1] : (tensor<256x128xf32>) -> tensor<256x128x28x28xf32>
    %v317 = stablehlo.divide %v316, %v313 : tensor<256x128x28x28xf32>
    %v318 = stablehlo.subtract %v311, %v317 : tensor<256x128x28x28xf32>
    %v319 = stablehlo.multiply %v318, %v318 : tensor<256x128x28x28xf32>
    %v320 = stablehlo.reduce(%v319 init: %v312) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128xf32>
    %v321 = stablehlo.broadcast_in_dim %v320, dims = [0, 1] : (tensor<256x128xf32>) -> tensor<256x128x28x28xf32>
    %v322 = stablehlo.divide %v321, %v313 : tensor<256x128x28x28xf32>
    %v323 = stablehlo.add %v322, %v314 : tensor<256x128x28x28xf32>
    %v324 = stablehlo.rsqrt %v323 : tensor<256x128x28x28xf32>
    %v325 = stablehlo.multiply %v318, %v324 : tensor<256x128x28x28xf32>
    %v326 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v327 = stablehlo.broadcast_in_dim %s2b0bt1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v328 = stablehlo.multiply %v325, %v326 : tensor<256x128x28x28xf32>
    %v329 = stablehlo.add %v328, %v327 : tensor<256x128x28x28xf32>
    %v330 = stablehlo.reshape %v329 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v331 = stablehlo.reshape %v330 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v332 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v333 = stablehlo.maximum %v331, %v332 : tensor<256x128x28x28xf32>
    %v334 = stablehlo.reshape %v333 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v335 = stablehlo.reshape %v334 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v336 = stablehlo.convolution(%v335, %s2b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v337 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v338 = stablehlo.add %v336, %v337 : tensor<256x128x28x28xf32>
    %v339 = stablehlo.reshape %v338 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v340 = stablehlo.reshape %v339 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v341 = stablehlo.constant dense<0.0> : tensor<f32>
    %v342 = stablehlo.constant dense<784.0> : tensor<256x128x28x28xf32>
    %v343 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v344 = stablehlo.reduce(%v340 init: %v341) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128xf32>
    %v345 = stablehlo.broadcast_in_dim %v344, dims = [0, 1] : (tensor<256x128xf32>) -> tensor<256x128x28x28xf32>
    %v346 = stablehlo.divide %v345, %v342 : tensor<256x128x28x28xf32>
    %v347 = stablehlo.subtract %v340, %v346 : tensor<256x128x28x28xf32>
    %v348 = stablehlo.multiply %v347, %v347 : tensor<256x128x28x28xf32>
    %v349 = stablehlo.reduce(%v348 init: %v341) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128xf32>
    %v350 = stablehlo.broadcast_in_dim %v349, dims = [0, 1] : (tensor<256x128xf32>) -> tensor<256x128x28x28xf32>
    %v351 = stablehlo.divide %v350, %v342 : tensor<256x128x28x28xf32>
    %v352 = stablehlo.add %v351, %v343 : tensor<256x128x28x28xf32>
    %v353 = stablehlo.rsqrt %v352 : tensor<256x128x28x28xf32>
    %v354 = stablehlo.multiply %v347, %v353 : tensor<256x128x28x28xf32>
    %v355 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v356 = stablehlo.broadcast_in_dim %s2b0bt2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v357 = stablehlo.multiply %v354, %v355 : tensor<256x128x28x28xf32>
    %v358 = stablehlo.add %v357, %v356 : tensor<256x128x28x28xf32>
    %v359 = stablehlo.reshape %v358 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v360 = stablehlo.reshape %v359 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v361 = stablehlo.reshape %v305 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v362 = stablehlo.add %v360, %v361 : tensor<256x128x28x28xf32>
    %v363 = stablehlo.reshape %v362 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v364 = stablehlo.reshape %v363 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v365 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v366 = stablehlo.maximum %v364, %v365 : tensor<256x128x28x28xf32>
    %v367 = stablehlo.reshape %v366 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v368 = stablehlo.reshape %v367 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v369 = stablehlo.convolution(%v368, %s2b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v370 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v371 = stablehlo.add %v369, %v370 : tensor<256x128x28x28xf32>
    %v372 = stablehlo.reshape %v371 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v373 = stablehlo.reshape %v372 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v374 = stablehlo.constant dense<0.0> : tensor<f32>
    %v375 = stablehlo.constant dense<784.0> : tensor<256x128x28x28xf32>
    %v376 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v377 = stablehlo.reduce(%v373 init: %v374) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128xf32>
    %v378 = stablehlo.broadcast_in_dim %v377, dims = [0, 1] : (tensor<256x128xf32>) -> tensor<256x128x28x28xf32>
    %v379 = stablehlo.divide %v378, %v375 : tensor<256x128x28x28xf32>
    %v380 = stablehlo.subtract %v373, %v379 : tensor<256x128x28x28xf32>
    %v381 = stablehlo.multiply %v380, %v380 : tensor<256x128x28x28xf32>
    %v382 = stablehlo.reduce(%v381 init: %v374) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128xf32>
    %v383 = stablehlo.broadcast_in_dim %v382, dims = [0, 1] : (tensor<256x128xf32>) -> tensor<256x128x28x28xf32>
    %v384 = stablehlo.divide %v383, %v375 : tensor<256x128x28x28xf32>
    %v385 = stablehlo.add %v384, %v376 : tensor<256x128x28x28xf32>
    %v386 = stablehlo.rsqrt %v385 : tensor<256x128x28x28xf32>
    %v387 = stablehlo.multiply %v380, %v386 : tensor<256x128x28x28xf32>
    %v388 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v389 = stablehlo.broadcast_in_dim %s2b1bt1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v390 = stablehlo.multiply %v387, %v388 : tensor<256x128x28x28xf32>
    %v391 = stablehlo.add %v390, %v389 : tensor<256x128x28x28xf32>
    %v392 = stablehlo.reshape %v391 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v393 = stablehlo.reshape %v392 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v394 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v395 = stablehlo.maximum %v393, %v394 : tensor<256x128x28x28xf32>
    %v396 = stablehlo.reshape %v395 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v397 = stablehlo.reshape %v396 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v398 = stablehlo.convolution(%v397, %s2b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v399 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v400 = stablehlo.add %v398, %v399 : tensor<256x128x28x28xf32>
    %v401 = stablehlo.reshape %v400 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v402 = stablehlo.reshape %v401 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v403 = stablehlo.constant dense<0.0> : tensor<f32>
    %v404 = stablehlo.constant dense<784.0> : tensor<256x128x28x28xf32>
    %v405 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v406 = stablehlo.reduce(%v402 init: %v403) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128xf32>
    %v407 = stablehlo.broadcast_in_dim %v406, dims = [0, 1] : (tensor<256x128xf32>) -> tensor<256x128x28x28xf32>
    %v408 = stablehlo.divide %v407, %v404 : tensor<256x128x28x28xf32>
    %v409 = stablehlo.subtract %v402, %v408 : tensor<256x128x28x28xf32>
    %v410 = stablehlo.multiply %v409, %v409 : tensor<256x128x28x28xf32>
    %v411 = stablehlo.reduce(%v410 init: %v403) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128xf32>
    %v412 = stablehlo.broadcast_in_dim %v411, dims = [0, 1] : (tensor<256x128xf32>) -> tensor<256x128x28x28xf32>
    %v413 = stablehlo.divide %v412, %v404 : tensor<256x128x28x28xf32>
    %v414 = stablehlo.add %v413, %v405 : tensor<256x128x28x28xf32>
    %v415 = stablehlo.rsqrt %v414 : tensor<256x128x28x28xf32>
    %v416 = stablehlo.multiply %v409, %v415 : tensor<256x128x28x28xf32>
    %v417 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v418 = stablehlo.broadcast_in_dim %s2b1bt2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v419 = stablehlo.multiply %v416, %v417 : tensor<256x128x28x28xf32>
    %v420 = stablehlo.add %v419, %v418 : tensor<256x128x28x28xf32>
    %v421 = stablehlo.reshape %v420 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v422 = stablehlo.reshape %v421 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v423 = stablehlo.reshape %v367 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v424 = stablehlo.add %v422, %v423 : tensor<256x128x28x28xf32>
    %v425 = stablehlo.reshape %v424 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v426 = stablehlo.reshape %v425 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v427 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v428 = stablehlo.maximum %v426, %v427 : tensor<256x128x28x28xf32>
    %v429 = stablehlo.reshape %v428 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v430 = stablehlo.reshape %v429 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v431 = stablehlo.convolution(%v430, %s2b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v432 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v433 = stablehlo.add %v431, %v432 : tensor<256x128x28x28xf32>
    %v434 = stablehlo.reshape %v433 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v435 = stablehlo.reshape %v434 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v436 = stablehlo.constant dense<0.0> : tensor<f32>
    %v437 = stablehlo.constant dense<784.0> : tensor<256x128x28x28xf32>
    %v438 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v439 = stablehlo.reduce(%v435 init: %v436) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128xf32>
    %v440 = stablehlo.broadcast_in_dim %v439, dims = [0, 1] : (tensor<256x128xf32>) -> tensor<256x128x28x28xf32>
    %v441 = stablehlo.divide %v440, %v437 : tensor<256x128x28x28xf32>
    %v442 = stablehlo.subtract %v435, %v441 : tensor<256x128x28x28xf32>
    %v443 = stablehlo.multiply %v442, %v442 : tensor<256x128x28x28xf32>
    %v444 = stablehlo.reduce(%v443 init: %v436) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128xf32>
    %v445 = stablehlo.broadcast_in_dim %v444, dims = [0, 1] : (tensor<256x128xf32>) -> tensor<256x128x28x28xf32>
    %v446 = stablehlo.divide %v445, %v437 : tensor<256x128x28x28xf32>
    %v447 = stablehlo.add %v446, %v438 : tensor<256x128x28x28xf32>
    %v448 = stablehlo.rsqrt %v447 : tensor<256x128x28x28xf32>
    %v449 = stablehlo.multiply %v442, %v448 : tensor<256x128x28x28xf32>
    %v450 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v451 = stablehlo.broadcast_in_dim %s2b2bt1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v452 = stablehlo.multiply %v449, %v450 : tensor<256x128x28x28xf32>
    %v453 = stablehlo.add %v452, %v451 : tensor<256x128x28x28xf32>
    %v454 = stablehlo.reshape %v453 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v455 = stablehlo.reshape %v454 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v456 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v457 = stablehlo.maximum %v455, %v456 : tensor<256x128x28x28xf32>
    %v458 = stablehlo.reshape %v457 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v459 = stablehlo.reshape %v458 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v460 = stablehlo.convolution(%v459, %s2b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v461 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v462 = stablehlo.add %v460, %v461 : tensor<256x128x28x28xf32>
    %v463 = stablehlo.reshape %v462 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v464 = stablehlo.reshape %v463 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v465 = stablehlo.constant dense<0.0> : tensor<f32>
    %v466 = stablehlo.constant dense<784.0> : tensor<256x128x28x28xf32>
    %v467 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v468 = stablehlo.reduce(%v464 init: %v465) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128xf32>
    %v469 = stablehlo.broadcast_in_dim %v468, dims = [0, 1] : (tensor<256x128xf32>) -> tensor<256x128x28x28xf32>
    %v470 = stablehlo.divide %v469, %v466 : tensor<256x128x28x28xf32>
    %v471 = stablehlo.subtract %v464, %v470 : tensor<256x128x28x28xf32>
    %v472 = stablehlo.multiply %v471, %v471 : tensor<256x128x28x28xf32>
    %v473 = stablehlo.reduce(%v472 init: %v465) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128xf32>
    %v474 = stablehlo.broadcast_in_dim %v473, dims = [0, 1] : (tensor<256x128xf32>) -> tensor<256x128x28x28xf32>
    %v475 = stablehlo.divide %v474, %v466 : tensor<256x128x28x28xf32>
    %v476 = stablehlo.add %v475, %v467 : tensor<256x128x28x28xf32>
    %v477 = stablehlo.rsqrt %v476 : tensor<256x128x28x28xf32>
    %v478 = stablehlo.multiply %v471, %v477 : tensor<256x128x28x28xf32>
    %v479 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v480 = stablehlo.broadcast_in_dim %s2b2bt2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v481 = stablehlo.multiply %v478, %v479 : tensor<256x128x28x28xf32>
    %v482 = stablehlo.add %v481, %v480 : tensor<256x128x28x28xf32>
    %v483 = stablehlo.reshape %v482 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v484 = stablehlo.reshape %v483 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v485 = stablehlo.reshape %v429 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v486 = stablehlo.add %v484, %v485 : tensor<256x128x28x28xf32>
    %v487 = stablehlo.reshape %v486 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v488 = stablehlo.reshape %v487 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v489 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v490 = stablehlo.maximum %v488, %v489 : tensor<256x128x28x28xf32>
    %v491 = stablehlo.reshape %v490 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v492 = stablehlo.reshape %v491 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v493 = stablehlo.convolution(%v492, %d3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<256x128x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v494 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v495 = stablehlo.add %v493, %v494 : tensor<256x256x14x14xf32>
    %v496 = stablehlo.reshape %v495 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v497 = stablehlo.reshape %v496 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v498 = stablehlo.constant dense<0.0> : tensor<f32>
    %v499 = stablehlo.constant dense<196.0> : tensor<256x256x14x14xf32>
    %v500 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v501 = stablehlo.reduce(%v497 init: %v498) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v502 = stablehlo.broadcast_in_dim %v501, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v503 = stablehlo.divide %v502, %v499 : tensor<256x256x14x14xf32>
    %v504 = stablehlo.subtract %v497, %v503 : tensor<256x256x14x14xf32>
    %v505 = stablehlo.multiply %v504, %v504 : tensor<256x256x14x14xf32>
    %v506 = stablehlo.reduce(%v505 init: %v498) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v507 = stablehlo.broadcast_in_dim %v506, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v508 = stablehlo.divide %v507, %v499 : tensor<256x256x14x14xf32>
    %v509 = stablehlo.add %v508, %v500 : tensor<256x256x14x14xf32>
    %v510 = stablehlo.rsqrt %v509 : tensor<256x256x14x14xf32>
    %v511 = stablehlo.multiply %v504, %v510 : tensor<256x256x14x14xf32>
    %v512 = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v513 = stablehlo.broadcast_in_dim %d3bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v514 = stablehlo.multiply %v511, %v512 : tensor<256x256x14x14xf32>
    %v515 = stablehlo.add %v514, %v513 : tensor<256x256x14x14xf32>
    %v516 = stablehlo.reshape %v515 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v517 = stablehlo.reshape %v516 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v518 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v519 = stablehlo.maximum %v517, %v518 : tensor<256x256x14x14xf32>
    %v520 = stablehlo.reshape %v519 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v521 = stablehlo.reshape %v520 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v522 = stablehlo.convolution(%v521, %d3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v523 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v524 = stablehlo.add %v522, %v523 : tensor<256x256x14x14xf32>
    %v525 = stablehlo.reshape %v524 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v526 = stablehlo.reshape %v525 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v527 = stablehlo.constant dense<0.0> : tensor<f32>
    %v528 = stablehlo.constant dense<196.0> : tensor<256x256x14x14xf32>
    %v529 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v530 = stablehlo.reduce(%v526 init: %v527) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v531 = stablehlo.broadcast_in_dim %v530, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v532 = stablehlo.divide %v531, %v528 : tensor<256x256x14x14xf32>
    %v533 = stablehlo.subtract %v526, %v532 : tensor<256x256x14x14xf32>
    %v534 = stablehlo.multiply %v533, %v533 : tensor<256x256x14x14xf32>
    %v535 = stablehlo.reduce(%v534 init: %v527) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v536 = stablehlo.broadcast_in_dim %v535, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v537 = stablehlo.divide %v536, %v528 : tensor<256x256x14x14xf32>
    %v538 = stablehlo.add %v537, %v529 : tensor<256x256x14x14xf32>
    %v539 = stablehlo.rsqrt %v538 : tensor<256x256x14x14xf32>
    %v540 = stablehlo.multiply %v533, %v539 : tensor<256x256x14x14xf32>
    %v541 = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v542 = stablehlo.broadcast_in_dim %d3bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v543 = stablehlo.multiply %v540, %v541 : tensor<256x256x14x14xf32>
    %v544 = stablehlo.add %v543, %v542 : tensor<256x256x14x14xf32>
    %v545 = stablehlo.reshape %v544 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v546 = stablehlo.reshape %v491 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v547 = stablehlo.convolution(%v546, %d3Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<256x128x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v548 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v549 = stablehlo.add %v547, %v548 : tensor<256x256x14x14xf32>
    %v550 = stablehlo.reshape %v549 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v551 = stablehlo.reshape %v550 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v552 = stablehlo.constant dense<0.0> : tensor<f32>
    %v553 = stablehlo.constant dense<196.0> : tensor<256x256x14x14xf32>
    %v554 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v555 = stablehlo.reduce(%v551 init: %v552) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v556 = stablehlo.broadcast_in_dim %v555, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v557 = stablehlo.divide %v556, %v553 : tensor<256x256x14x14xf32>
    %v558 = stablehlo.subtract %v551, %v557 : tensor<256x256x14x14xf32>
    %v559 = stablehlo.multiply %v558, %v558 : tensor<256x256x14x14xf32>
    %v560 = stablehlo.reduce(%v559 init: %v552) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v561 = stablehlo.broadcast_in_dim %v560, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v562 = stablehlo.divide %v561, %v553 : tensor<256x256x14x14xf32>
    %v563 = stablehlo.add %v562, %v554 : tensor<256x256x14x14xf32>
    %v564 = stablehlo.rsqrt %v563 : tensor<256x256x14x14xf32>
    %v565 = stablehlo.multiply %v558, %v564 : tensor<256x256x14x14xf32>
    %v566 = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v567 = stablehlo.broadcast_in_dim %d3btp, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v568 = stablehlo.multiply %v565, %v566 : tensor<256x256x14x14xf32>
    %v569 = stablehlo.add %v568, %v567 : tensor<256x256x14x14xf32>
    %v570 = stablehlo.reshape %v569 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v571 = stablehlo.reshape %v545 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v572 = stablehlo.reshape %v570 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v573 = stablehlo.add %v571, %v572 : tensor<256x256x14x14xf32>
    %v574 = stablehlo.reshape %v573 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v575 = stablehlo.reshape %v574 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v576 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v577 = stablehlo.maximum %v575, %v576 : tensor<256x256x14x14xf32>
    %v578 = stablehlo.reshape %v577 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v579 = stablehlo.reshape %v578 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v580 = stablehlo.convolution(%v579, %s3b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v581 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v582 = stablehlo.add %v580, %v581 : tensor<256x256x14x14xf32>
    %v583 = stablehlo.reshape %v582 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v584 = stablehlo.reshape %v583 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v585 = stablehlo.constant dense<0.0> : tensor<f32>
    %v586 = stablehlo.constant dense<196.0> : tensor<256x256x14x14xf32>
    %v587 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v588 = stablehlo.reduce(%v584 init: %v585) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v589 = stablehlo.broadcast_in_dim %v588, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v590 = stablehlo.divide %v589, %v586 : tensor<256x256x14x14xf32>
    %v591 = stablehlo.subtract %v584, %v590 : tensor<256x256x14x14xf32>
    %v592 = stablehlo.multiply %v591, %v591 : tensor<256x256x14x14xf32>
    %v593 = stablehlo.reduce(%v592 init: %v585) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v594 = stablehlo.broadcast_in_dim %v593, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v595 = stablehlo.divide %v594, %v586 : tensor<256x256x14x14xf32>
    %v596 = stablehlo.add %v595, %v587 : tensor<256x256x14x14xf32>
    %v597 = stablehlo.rsqrt %v596 : tensor<256x256x14x14xf32>
    %v598 = stablehlo.multiply %v591, %v597 : tensor<256x256x14x14xf32>
    %v599 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v600 = stablehlo.broadcast_in_dim %s3b0bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v601 = stablehlo.multiply %v598, %v599 : tensor<256x256x14x14xf32>
    %v602 = stablehlo.add %v601, %v600 : tensor<256x256x14x14xf32>
    %v603 = stablehlo.reshape %v602 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v604 = stablehlo.reshape %v603 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v605 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v606 = stablehlo.maximum %v604, %v605 : tensor<256x256x14x14xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v608 = stablehlo.reshape %v607 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v609 = stablehlo.convolution(%v608, %s3b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v610 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v611 = stablehlo.add %v609, %v610 : tensor<256x256x14x14xf32>
    %v612 = stablehlo.reshape %v611 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v613 = stablehlo.reshape %v612 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v614 = stablehlo.constant dense<0.0> : tensor<f32>
    %v615 = stablehlo.constant dense<196.0> : tensor<256x256x14x14xf32>
    %v616 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v617 = stablehlo.reduce(%v613 init: %v614) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v618 = stablehlo.broadcast_in_dim %v617, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v619 = stablehlo.divide %v618, %v615 : tensor<256x256x14x14xf32>
    %v620 = stablehlo.subtract %v613, %v619 : tensor<256x256x14x14xf32>
    %v621 = stablehlo.multiply %v620, %v620 : tensor<256x256x14x14xf32>
    %v622 = stablehlo.reduce(%v621 init: %v614) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v623 = stablehlo.broadcast_in_dim %v622, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v624 = stablehlo.divide %v623, %v615 : tensor<256x256x14x14xf32>
    %v625 = stablehlo.add %v624, %v616 : tensor<256x256x14x14xf32>
    %v626 = stablehlo.rsqrt %v625 : tensor<256x256x14x14xf32>
    %v627 = stablehlo.multiply %v620, %v626 : tensor<256x256x14x14xf32>
    %v628 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v629 = stablehlo.broadcast_in_dim %s3b0bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v630 = stablehlo.multiply %v627, %v628 : tensor<256x256x14x14xf32>
    %v631 = stablehlo.add %v630, %v629 : tensor<256x256x14x14xf32>
    %v632 = stablehlo.reshape %v631 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v633 = stablehlo.reshape %v632 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v634 = stablehlo.reshape %v578 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v635 = stablehlo.add %v633, %v634 : tensor<256x256x14x14xf32>
    %v636 = stablehlo.reshape %v635 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v637 = stablehlo.reshape %v636 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v638 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v639 = stablehlo.maximum %v637, %v638 : tensor<256x256x14x14xf32>
    %v640 = stablehlo.reshape %v639 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v641 = stablehlo.reshape %v640 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v642 = stablehlo.convolution(%v641, %s3b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v643 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v644 = stablehlo.add %v642, %v643 : tensor<256x256x14x14xf32>
    %v645 = stablehlo.reshape %v644 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v646 = stablehlo.reshape %v645 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v647 = stablehlo.constant dense<0.0> : tensor<f32>
    %v648 = stablehlo.constant dense<196.0> : tensor<256x256x14x14xf32>
    %v649 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v650 = stablehlo.reduce(%v646 init: %v647) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v651 = stablehlo.broadcast_in_dim %v650, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v652 = stablehlo.divide %v651, %v648 : tensor<256x256x14x14xf32>
    %v653 = stablehlo.subtract %v646, %v652 : tensor<256x256x14x14xf32>
    %v654 = stablehlo.multiply %v653, %v653 : tensor<256x256x14x14xf32>
    %v655 = stablehlo.reduce(%v654 init: %v647) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v656 = stablehlo.broadcast_in_dim %v655, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v657 = stablehlo.divide %v656, %v648 : tensor<256x256x14x14xf32>
    %v658 = stablehlo.add %v657, %v649 : tensor<256x256x14x14xf32>
    %v659 = stablehlo.rsqrt %v658 : tensor<256x256x14x14xf32>
    %v660 = stablehlo.multiply %v653, %v659 : tensor<256x256x14x14xf32>
    %v661 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v662 = stablehlo.broadcast_in_dim %s3b1bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v663 = stablehlo.multiply %v660, %v661 : tensor<256x256x14x14xf32>
    %v664 = stablehlo.add %v663, %v662 : tensor<256x256x14x14xf32>
    %v665 = stablehlo.reshape %v664 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v666 = stablehlo.reshape %v665 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v667 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v668 = stablehlo.maximum %v666, %v667 : tensor<256x256x14x14xf32>
    %v669 = stablehlo.reshape %v668 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v670 = stablehlo.reshape %v669 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v671 = stablehlo.convolution(%v670, %s3b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v672 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v673 = stablehlo.add %v671, %v672 : tensor<256x256x14x14xf32>
    %v674 = stablehlo.reshape %v673 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v675 = stablehlo.reshape %v674 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v676 = stablehlo.constant dense<0.0> : tensor<f32>
    %v677 = stablehlo.constant dense<196.0> : tensor<256x256x14x14xf32>
    %v678 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v679 = stablehlo.reduce(%v675 init: %v676) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v680 = stablehlo.broadcast_in_dim %v679, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v681 = stablehlo.divide %v680, %v677 : tensor<256x256x14x14xf32>
    %v682 = stablehlo.subtract %v675, %v681 : tensor<256x256x14x14xf32>
    %v683 = stablehlo.multiply %v682, %v682 : tensor<256x256x14x14xf32>
    %v684 = stablehlo.reduce(%v683 init: %v676) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v685 = stablehlo.broadcast_in_dim %v684, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v686 = stablehlo.divide %v685, %v677 : tensor<256x256x14x14xf32>
    %v687 = stablehlo.add %v686, %v678 : tensor<256x256x14x14xf32>
    %v688 = stablehlo.rsqrt %v687 : tensor<256x256x14x14xf32>
    %v689 = stablehlo.multiply %v682, %v688 : tensor<256x256x14x14xf32>
    %v690 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v691 = stablehlo.broadcast_in_dim %s3b1bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v692 = stablehlo.multiply %v689, %v690 : tensor<256x256x14x14xf32>
    %v693 = stablehlo.add %v692, %v691 : tensor<256x256x14x14xf32>
    %v694 = stablehlo.reshape %v693 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v695 = stablehlo.reshape %v694 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v696 = stablehlo.reshape %v640 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v697 = stablehlo.add %v695, %v696 : tensor<256x256x14x14xf32>
    %v698 = stablehlo.reshape %v697 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v699 = stablehlo.reshape %v698 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v700 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v701 = stablehlo.maximum %v699, %v700 : tensor<256x256x14x14xf32>
    %v702 = stablehlo.reshape %v701 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v703 = stablehlo.reshape %v702 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v704 = stablehlo.convolution(%v703, %s3b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v705 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v706 = stablehlo.add %v704, %v705 : tensor<256x256x14x14xf32>
    %v707 = stablehlo.reshape %v706 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v708 = stablehlo.reshape %v707 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v709 = stablehlo.constant dense<0.0> : tensor<f32>
    %v710 = stablehlo.constant dense<196.0> : tensor<256x256x14x14xf32>
    %v711 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v712 = stablehlo.reduce(%v708 init: %v709) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v713 = stablehlo.broadcast_in_dim %v712, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v714 = stablehlo.divide %v713, %v710 : tensor<256x256x14x14xf32>
    %v715 = stablehlo.subtract %v708, %v714 : tensor<256x256x14x14xf32>
    %v716 = stablehlo.multiply %v715, %v715 : tensor<256x256x14x14xf32>
    %v717 = stablehlo.reduce(%v716 init: %v709) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v718 = stablehlo.broadcast_in_dim %v717, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v719 = stablehlo.divide %v718, %v710 : tensor<256x256x14x14xf32>
    %v720 = stablehlo.add %v719, %v711 : tensor<256x256x14x14xf32>
    %v721 = stablehlo.rsqrt %v720 : tensor<256x256x14x14xf32>
    %v722 = stablehlo.multiply %v715, %v721 : tensor<256x256x14x14xf32>
    %v723 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v724 = stablehlo.broadcast_in_dim %s3b2bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v725 = stablehlo.multiply %v722, %v723 : tensor<256x256x14x14xf32>
    %v726 = stablehlo.add %v725, %v724 : tensor<256x256x14x14xf32>
    %v727 = stablehlo.reshape %v726 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v728 = stablehlo.reshape %v727 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v729 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v730 = stablehlo.maximum %v728, %v729 : tensor<256x256x14x14xf32>
    %v731 = stablehlo.reshape %v730 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v732 = stablehlo.reshape %v731 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v733 = stablehlo.convolution(%v732, %s3b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v734 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v735 = stablehlo.add %v733, %v734 : tensor<256x256x14x14xf32>
    %v736 = stablehlo.reshape %v735 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v737 = stablehlo.reshape %v736 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v738 = stablehlo.constant dense<0.0> : tensor<f32>
    %v739 = stablehlo.constant dense<196.0> : tensor<256x256x14x14xf32>
    %v740 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v741 = stablehlo.reduce(%v737 init: %v738) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v742 = stablehlo.broadcast_in_dim %v741, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v743 = stablehlo.divide %v742, %v739 : tensor<256x256x14x14xf32>
    %v744 = stablehlo.subtract %v737, %v743 : tensor<256x256x14x14xf32>
    %v745 = stablehlo.multiply %v744, %v744 : tensor<256x256x14x14xf32>
    %v746 = stablehlo.reduce(%v745 init: %v738) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v747 = stablehlo.broadcast_in_dim %v746, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v748 = stablehlo.divide %v747, %v739 : tensor<256x256x14x14xf32>
    %v749 = stablehlo.add %v748, %v740 : tensor<256x256x14x14xf32>
    %v750 = stablehlo.rsqrt %v749 : tensor<256x256x14x14xf32>
    %v751 = stablehlo.multiply %v744, %v750 : tensor<256x256x14x14xf32>
    %v752 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v753 = stablehlo.broadcast_in_dim %s3b2bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v754 = stablehlo.multiply %v751, %v752 : tensor<256x256x14x14xf32>
    %v755 = stablehlo.add %v754, %v753 : tensor<256x256x14x14xf32>
    %v756 = stablehlo.reshape %v755 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v757 = stablehlo.reshape %v756 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v758 = stablehlo.reshape %v702 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v759 = stablehlo.add %v757, %v758 : tensor<256x256x14x14xf32>
    %v760 = stablehlo.reshape %v759 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v761 = stablehlo.reshape %v760 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v762 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v763 = stablehlo.maximum %v761, %v762 : tensor<256x256x14x14xf32>
    %v764 = stablehlo.reshape %v763 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v765 = stablehlo.reshape %v764 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v766 = stablehlo.convolution(%v765, %s3b3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v767 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v768 = stablehlo.add %v766, %v767 : tensor<256x256x14x14xf32>
    %v769 = stablehlo.reshape %v768 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v770 = stablehlo.reshape %v769 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v771 = stablehlo.constant dense<0.0> : tensor<f32>
    %v772 = stablehlo.constant dense<196.0> : tensor<256x256x14x14xf32>
    %v773 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v774 = stablehlo.reduce(%v770 init: %v771) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v775 = stablehlo.broadcast_in_dim %v774, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v776 = stablehlo.divide %v775, %v772 : tensor<256x256x14x14xf32>
    %v777 = stablehlo.subtract %v770, %v776 : tensor<256x256x14x14xf32>
    %v778 = stablehlo.multiply %v777, %v777 : tensor<256x256x14x14xf32>
    %v779 = stablehlo.reduce(%v778 init: %v771) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v780 = stablehlo.broadcast_in_dim %v779, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v781 = stablehlo.divide %v780, %v772 : tensor<256x256x14x14xf32>
    %v782 = stablehlo.add %v781, %v773 : tensor<256x256x14x14xf32>
    %v783 = stablehlo.rsqrt %v782 : tensor<256x256x14x14xf32>
    %v784 = stablehlo.multiply %v777, %v783 : tensor<256x256x14x14xf32>
    %v785 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v786 = stablehlo.broadcast_in_dim %s3b3bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v787 = stablehlo.multiply %v784, %v785 : tensor<256x256x14x14xf32>
    %v788 = stablehlo.add %v787, %v786 : tensor<256x256x14x14xf32>
    %v789 = stablehlo.reshape %v788 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v790 = stablehlo.reshape %v789 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v791 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v792 = stablehlo.maximum %v790, %v791 : tensor<256x256x14x14xf32>
    %v793 = stablehlo.reshape %v792 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v794 = stablehlo.reshape %v793 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v795 = stablehlo.convolution(%v794, %s3b3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v796 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v797 = stablehlo.add %v795, %v796 : tensor<256x256x14x14xf32>
    %v798 = stablehlo.reshape %v797 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v799 = stablehlo.reshape %v798 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v800 = stablehlo.constant dense<0.0> : tensor<f32>
    %v801 = stablehlo.constant dense<196.0> : tensor<256x256x14x14xf32>
    %v802 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v803 = stablehlo.reduce(%v799 init: %v800) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v804 = stablehlo.broadcast_in_dim %v803, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v805 = stablehlo.divide %v804, %v801 : tensor<256x256x14x14xf32>
    %v806 = stablehlo.subtract %v799, %v805 : tensor<256x256x14x14xf32>
    %v807 = stablehlo.multiply %v806, %v806 : tensor<256x256x14x14xf32>
    %v808 = stablehlo.reduce(%v807 init: %v800) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v809 = stablehlo.broadcast_in_dim %v808, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v810 = stablehlo.divide %v809, %v801 : tensor<256x256x14x14xf32>
    %v811 = stablehlo.add %v810, %v802 : tensor<256x256x14x14xf32>
    %v812 = stablehlo.rsqrt %v811 : tensor<256x256x14x14xf32>
    %v813 = stablehlo.multiply %v806, %v812 : tensor<256x256x14x14xf32>
    %v814 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v815 = stablehlo.broadcast_in_dim %s3b3bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v816 = stablehlo.multiply %v813, %v814 : tensor<256x256x14x14xf32>
    %v817 = stablehlo.add %v816, %v815 : tensor<256x256x14x14xf32>
    %v818 = stablehlo.reshape %v817 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v819 = stablehlo.reshape %v818 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v820 = stablehlo.reshape %v764 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v821 = stablehlo.add %v819, %v820 : tensor<256x256x14x14xf32>
    %v822 = stablehlo.reshape %v821 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v823 = stablehlo.reshape %v822 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v824 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v825 = stablehlo.maximum %v823, %v824 : tensor<256x256x14x14xf32>
    %v826 = stablehlo.reshape %v825 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v827 = stablehlo.reshape %v826 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v828 = stablehlo.convolution(%v827, %s3b4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v829 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v830 = stablehlo.add %v828, %v829 : tensor<256x256x14x14xf32>
    %v831 = stablehlo.reshape %v830 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v832 = stablehlo.reshape %v831 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v833 = stablehlo.constant dense<0.0> : tensor<f32>
    %v834 = stablehlo.constant dense<196.0> : tensor<256x256x14x14xf32>
    %v835 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v836 = stablehlo.reduce(%v832 init: %v833) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v837 = stablehlo.broadcast_in_dim %v836, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v838 = stablehlo.divide %v837, %v834 : tensor<256x256x14x14xf32>
    %v839 = stablehlo.subtract %v832, %v838 : tensor<256x256x14x14xf32>
    %v840 = stablehlo.multiply %v839, %v839 : tensor<256x256x14x14xf32>
    %v841 = stablehlo.reduce(%v840 init: %v833) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v842 = stablehlo.broadcast_in_dim %v841, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v843 = stablehlo.divide %v842, %v834 : tensor<256x256x14x14xf32>
    %v844 = stablehlo.add %v843, %v835 : tensor<256x256x14x14xf32>
    %v845 = stablehlo.rsqrt %v844 : tensor<256x256x14x14xf32>
    %v846 = stablehlo.multiply %v839, %v845 : tensor<256x256x14x14xf32>
    %v847 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v848 = stablehlo.broadcast_in_dim %s3b4bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v849 = stablehlo.multiply %v846, %v847 : tensor<256x256x14x14xf32>
    %v850 = stablehlo.add %v849, %v848 : tensor<256x256x14x14xf32>
    %v851 = stablehlo.reshape %v850 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v852 = stablehlo.reshape %v851 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v853 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v854 = stablehlo.maximum %v852, %v853 : tensor<256x256x14x14xf32>
    %v855 = stablehlo.reshape %v854 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v856 = stablehlo.reshape %v855 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v857 = stablehlo.convolution(%v856, %s3b4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v858 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v859 = stablehlo.add %v857, %v858 : tensor<256x256x14x14xf32>
    %v860 = stablehlo.reshape %v859 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v861 = stablehlo.reshape %v860 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v862 = stablehlo.constant dense<0.0> : tensor<f32>
    %v863 = stablehlo.constant dense<196.0> : tensor<256x256x14x14xf32>
    %v864 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v865 = stablehlo.reduce(%v861 init: %v862) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v866 = stablehlo.broadcast_in_dim %v865, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v867 = stablehlo.divide %v866, %v863 : tensor<256x256x14x14xf32>
    %v868 = stablehlo.subtract %v861, %v867 : tensor<256x256x14x14xf32>
    %v869 = stablehlo.multiply %v868, %v868 : tensor<256x256x14x14xf32>
    %v870 = stablehlo.reduce(%v869 init: %v862) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v871 = stablehlo.broadcast_in_dim %v870, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v872 = stablehlo.divide %v871, %v863 : tensor<256x256x14x14xf32>
    %v873 = stablehlo.add %v872, %v864 : tensor<256x256x14x14xf32>
    %v874 = stablehlo.rsqrt %v873 : tensor<256x256x14x14xf32>
    %v875 = stablehlo.multiply %v868, %v874 : tensor<256x256x14x14xf32>
    %v876 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v877 = stablehlo.broadcast_in_dim %s3b4bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v878 = stablehlo.multiply %v875, %v876 : tensor<256x256x14x14xf32>
    %v879 = stablehlo.add %v878, %v877 : tensor<256x256x14x14xf32>
    %v880 = stablehlo.reshape %v879 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v881 = stablehlo.reshape %v880 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v882 = stablehlo.reshape %v826 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v883 = stablehlo.add %v881, %v882 : tensor<256x256x14x14xf32>
    %v884 = stablehlo.reshape %v883 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v885 = stablehlo.reshape %v884 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v886 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v887 = stablehlo.maximum %v885, %v886 : tensor<256x256x14x14xf32>
    %v888 = stablehlo.reshape %v887 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v889 = stablehlo.reshape %v888 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v890 = stablehlo.convolution(%v889, %d4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<512x256x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v891 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v892 = stablehlo.add %v890, %v891 : tensor<256x512x7x7xf32>
    %v893 = stablehlo.reshape %v892 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v894 = stablehlo.reshape %v893 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v895 = stablehlo.constant dense<0.0> : tensor<f32>
    %v896 = stablehlo.constant dense<49.0> : tensor<256x512x7x7xf32>
    %v897 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v898 = stablehlo.reduce(%v894 init: %v895) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512xf32>
    %v899 = stablehlo.broadcast_in_dim %v898, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512x7x7xf32>
    %v900 = stablehlo.divide %v899, %v896 : tensor<256x512x7x7xf32>
    %v901 = stablehlo.subtract %v894, %v900 : tensor<256x512x7x7xf32>
    %v902 = stablehlo.multiply %v901, %v901 : tensor<256x512x7x7xf32>
    %v903 = stablehlo.reduce(%v902 init: %v895) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512xf32>
    %v904 = stablehlo.broadcast_in_dim %v903, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512x7x7xf32>
    %v905 = stablehlo.divide %v904, %v896 : tensor<256x512x7x7xf32>
    %v906 = stablehlo.add %v905, %v897 : tensor<256x512x7x7xf32>
    %v907 = stablehlo.rsqrt %v906 : tensor<256x512x7x7xf32>
    %v908 = stablehlo.multiply %v901, %v907 : tensor<256x512x7x7xf32>
    %v909 = stablehlo.broadcast_in_dim %d4g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v910 = stablehlo.broadcast_in_dim %d4bt1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v911 = stablehlo.multiply %v908, %v909 : tensor<256x512x7x7xf32>
    %v912 = stablehlo.add %v911, %v910 : tensor<256x512x7x7xf32>
    %v913 = stablehlo.reshape %v912 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v914 = stablehlo.reshape %v913 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v915 = stablehlo.constant dense<0.0> : tensor<256x512x7x7xf32>
    %v916 = stablehlo.maximum %v914, %v915 : tensor<256x512x7x7xf32>
    %v917 = stablehlo.reshape %v916 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v918 = stablehlo.reshape %v917 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v919 = stablehlo.convolution(%v918, %d4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v920 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v921 = stablehlo.add %v919, %v920 : tensor<256x512x7x7xf32>
    %v922 = stablehlo.reshape %v921 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v923 = stablehlo.reshape %v922 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v924 = stablehlo.constant dense<0.0> : tensor<f32>
    %v925 = stablehlo.constant dense<49.0> : tensor<256x512x7x7xf32>
    %v926 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v927 = stablehlo.reduce(%v923 init: %v924) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512xf32>
    %v928 = stablehlo.broadcast_in_dim %v927, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512x7x7xf32>
    %v929 = stablehlo.divide %v928, %v925 : tensor<256x512x7x7xf32>
    %v930 = stablehlo.subtract %v923, %v929 : tensor<256x512x7x7xf32>
    %v931 = stablehlo.multiply %v930, %v930 : tensor<256x512x7x7xf32>
    %v932 = stablehlo.reduce(%v931 init: %v924) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512xf32>
    %v933 = stablehlo.broadcast_in_dim %v932, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512x7x7xf32>
    %v934 = stablehlo.divide %v933, %v925 : tensor<256x512x7x7xf32>
    %v935 = stablehlo.add %v934, %v926 : tensor<256x512x7x7xf32>
    %v936 = stablehlo.rsqrt %v935 : tensor<256x512x7x7xf32>
    %v937 = stablehlo.multiply %v930, %v936 : tensor<256x512x7x7xf32>
    %v938 = stablehlo.broadcast_in_dim %d4g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v939 = stablehlo.broadcast_in_dim %d4bt2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v940 = stablehlo.multiply %v937, %v938 : tensor<256x512x7x7xf32>
    %v941 = stablehlo.add %v940, %v939 : tensor<256x512x7x7xf32>
    %v942 = stablehlo.reshape %v941 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v943 = stablehlo.reshape %v888 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v944 = stablehlo.convolution(%v943, %d4Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<512x256x1x1xf32>) -> tensor<256x512x7x7xf32>
    %v945 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v946 = stablehlo.add %v944, %v945 : tensor<256x512x7x7xf32>
    %v947 = stablehlo.reshape %v946 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v948 = stablehlo.reshape %v947 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v949 = stablehlo.constant dense<0.0> : tensor<f32>
    %v950 = stablehlo.constant dense<49.0> : tensor<256x512x7x7xf32>
    %v951 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v952 = stablehlo.reduce(%v948 init: %v949) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512xf32>
    %v953 = stablehlo.broadcast_in_dim %v952, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512x7x7xf32>
    %v954 = stablehlo.divide %v953, %v950 : tensor<256x512x7x7xf32>
    %v955 = stablehlo.subtract %v948, %v954 : tensor<256x512x7x7xf32>
    %v956 = stablehlo.multiply %v955, %v955 : tensor<256x512x7x7xf32>
    %v957 = stablehlo.reduce(%v956 init: %v949) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512xf32>
    %v958 = stablehlo.broadcast_in_dim %v957, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512x7x7xf32>
    %v959 = stablehlo.divide %v958, %v950 : tensor<256x512x7x7xf32>
    %v960 = stablehlo.add %v959, %v951 : tensor<256x512x7x7xf32>
    %v961 = stablehlo.rsqrt %v960 : tensor<256x512x7x7xf32>
    %v962 = stablehlo.multiply %v955, %v961 : tensor<256x512x7x7xf32>
    %v963 = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v964 = stablehlo.broadcast_in_dim %d4btp, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v965 = stablehlo.multiply %v962, %v963 : tensor<256x512x7x7xf32>
    %v966 = stablehlo.add %v965, %v964 : tensor<256x512x7x7xf32>
    %v967 = stablehlo.reshape %v966 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v968 = stablehlo.reshape %v942 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v969 = stablehlo.reshape %v967 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v970 = stablehlo.add %v968, %v969 : tensor<256x512x7x7xf32>
    %v971 = stablehlo.reshape %v970 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v972 = stablehlo.reshape %v971 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v973 = stablehlo.constant dense<0.0> : tensor<256x512x7x7xf32>
    %v974 = stablehlo.maximum %v972, %v973 : tensor<256x512x7x7xf32>
    %v975 = stablehlo.reshape %v974 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v976 = stablehlo.reshape %v975 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v977 = stablehlo.convolution(%v976, %s4b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v978 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v979 = stablehlo.add %v977, %v978 : tensor<256x512x7x7xf32>
    %v980 = stablehlo.reshape %v979 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v981 = stablehlo.reshape %v980 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v982 = stablehlo.constant dense<0.0> : tensor<f32>
    %v983 = stablehlo.constant dense<49.0> : tensor<256x512x7x7xf32>
    %v984 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v985 = stablehlo.reduce(%v981 init: %v982) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512xf32>
    %v986 = stablehlo.broadcast_in_dim %v985, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512x7x7xf32>
    %v987 = stablehlo.divide %v986, %v983 : tensor<256x512x7x7xf32>
    %v988 = stablehlo.subtract %v981, %v987 : tensor<256x512x7x7xf32>
    %v989 = stablehlo.multiply %v988, %v988 : tensor<256x512x7x7xf32>
    %v990 = stablehlo.reduce(%v989 init: %v982) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512xf32>
    %v991 = stablehlo.broadcast_in_dim %v990, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512x7x7xf32>
    %v992 = stablehlo.divide %v991, %v983 : tensor<256x512x7x7xf32>
    %v993 = stablehlo.add %v992, %v984 : tensor<256x512x7x7xf32>
    %v994 = stablehlo.rsqrt %v993 : tensor<256x512x7x7xf32>
    %v995 = stablehlo.multiply %v988, %v994 : tensor<256x512x7x7xf32>
    %v996 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v997 = stablehlo.broadcast_in_dim %s4b0bt1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v998 = stablehlo.multiply %v995, %v996 : tensor<256x512x7x7xf32>
    %v999 = stablehlo.add %v998, %v997 : tensor<256x512x7x7xf32>
    %v1000 = stablehlo.reshape %v999 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1001 = stablehlo.reshape %v1000 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1002 = stablehlo.constant dense<0.0> : tensor<256x512x7x7xf32>
    %v1003 = stablehlo.maximum %v1001, %v1002 : tensor<256x512x7x7xf32>
    %v1004 = stablehlo.reshape %v1003 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1005 = stablehlo.reshape %v1004 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1006 = stablehlo.convolution(%v1005, %s4b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v1007 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1008 = stablehlo.add %v1006, %v1007 : tensor<256x512x7x7xf32>
    %v1009 = stablehlo.reshape %v1008 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1010 = stablehlo.reshape %v1009 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1011 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1012 = stablehlo.constant dense<49.0> : tensor<256x512x7x7xf32>
    %v1013 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1014 = stablehlo.reduce(%v1010 init: %v1011) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512xf32>
    %v1015 = stablehlo.broadcast_in_dim %v1014, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512x7x7xf32>
    %v1016 = stablehlo.divide %v1015, %v1012 : tensor<256x512x7x7xf32>
    %v1017 = stablehlo.subtract %v1010, %v1016 : tensor<256x512x7x7xf32>
    %v1018 = stablehlo.multiply %v1017, %v1017 : tensor<256x512x7x7xf32>
    %v1019 = stablehlo.reduce(%v1018 init: %v1011) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512xf32>
    %v1020 = stablehlo.broadcast_in_dim %v1019, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512x7x7xf32>
    %v1021 = stablehlo.divide %v1020, %v1012 : tensor<256x512x7x7xf32>
    %v1022 = stablehlo.add %v1021, %v1013 : tensor<256x512x7x7xf32>
    %v1023 = stablehlo.rsqrt %v1022 : tensor<256x512x7x7xf32>
    %v1024 = stablehlo.multiply %v1017, %v1023 : tensor<256x512x7x7xf32>
    %v1025 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1026 = stablehlo.broadcast_in_dim %s4b0bt2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1027 = stablehlo.multiply %v1024, %v1025 : tensor<256x512x7x7xf32>
    %v1028 = stablehlo.add %v1027, %v1026 : tensor<256x512x7x7xf32>
    %v1029 = stablehlo.reshape %v1028 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1030 = stablehlo.reshape %v1029 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1031 = stablehlo.reshape %v975 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1032 = stablehlo.add %v1030, %v1031 : tensor<256x512x7x7xf32>
    %v1033 = stablehlo.reshape %v1032 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1034 = stablehlo.reshape %v1033 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1035 = stablehlo.constant dense<0.0> : tensor<256x512x7x7xf32>
    %v1036 = stablehlo.maximum %v1034, %v1035 : tensor<256x512x7x7xf32>
    %v1037 = stablehlo.reshape %v1036 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1038 = stablehlo.reshape %v1037 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1039 = stablehlo.convolution(%v1038, %s4b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v1040 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1041 = stablehlo.add %v1039, %v1040 : tensor<256x512x7x7xf32>
    %v1042 = stablehlo.reshape %v1041 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1043 = stablehlo.reshape %v1042 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1044 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1045 = stablehlo.constant dense<49.0> : tensor<256x512x7x7xf32>
    %v1046 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1047 = stablehlo.reduce(%v1043 init: %v1044) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512xf32>
    %v1048 = stablehlo.broadcast_in_dim %v1047, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512x7x7xf32>
    %v1049 = stablehlo.divide %v1048, %v1045 : tensor<256x512x7x7xf32>
    %v1050 = stablehlo.subtract %v1043, %v1049 : tensor<256x512x7x7xf32>
    %v1051 = stablehlo.multiply %v1050, %v1050 : tensor<256x512x7x7xf32>
    %v1052 = stablehlo.reduce(%v1051 init: %v1044) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512xf32>
    %v1053 = stablehlo.broadcast_in_dim %v1052, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512x7x7xf32>
    %v1054 = stablehlo.divide %v1053, %v1045 : tensor<256x512x7x7xf32>
    %v1055 = stablehlo.add %v1054, %v1046 : tensor<256x512x7x7xf32>
    %v1056 = stablehlo.rsqrt %v1055 : tensor<256x512x7x7xf32>
    %v1057 = stablehlo.multiply %v1050, %v1056 : tensor<256x512x7x7xf32>
    %v1058 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1059 = stablehlo.broadcast_in_dim %s4b1bt1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1060 = stablehlo.multiply %v1057, %v1058 : tensor<256x512x7x7xf32>
    %v1061 = stablehlo.add %v1060, %v1059 : tensor<256x512x7x7xf32>
    %v1062 = stablehlo.reshape %v1061 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1063 = stablehlo.reshape %v1062 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1064 = stablehlo.constant dense<0.0> : tensor<256x512x7x7xf32>
    %v1065 = stablehlo.maximum %v1063, %v1064 : tensor<256x512x7x7xf32>
    %v1066 = stablehlo.reshape %v1065 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1067 = stablehlo.reshape %v1066 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1068 = stablehlo.convolution(%v1067, %s4b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v1069 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1070 = stablehlo.add %v1068, %v1069 : tensor<256x512x7x7xf32>
    %v1071 = stablehlo.reshape %v1070 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1072 = stablehlo.reshape %v1071 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1073 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1074 = stablehlo.constant dense<49.0> : tensor<256x512x7x7xf32>
    %v1075 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1076 = stablehlo.reduce(%v1072 init: %v1073) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512xf32>
    %v1077 = stablehlo.broadcast_in_dim %v1076, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512x7x7xf32>
    %v1078 = stablehlo.divide %v1077, %v1074 : tensor<256x512x7x7xf32>
    %v1079 = stablehlo.subtract %v1072, %v1078 : tensor<256x512x7x7xf32>
    %v1080 = stablehlo.multiply %v1079, %v1079 : tensor<256x512x7x7xf32>
    %v1081 = stablehlo.reduce(%v1080 init: %v1073) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512xf32>
    %v1082 = stablehlo.broadcast_in_dim %v1081, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512x7x7xf32>
    %v1083 = stablehlo.divide %v1082, %v1074 : tensor<256x512x7x7xf32>
    %v1084 = stablehlo.add %v1083, %v1075 : tensor<256x512x7x7xf32>
    %v1085 = stablehlo.rsqrt %v1084 : tensor<256x512x7x7xf32>
    %v1086 = stablehlo.multiply %v1079, %v1085 : tensor<256x512x7x7xf32>
    %v1087 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1088 = stablehlo.broadcast_in_dim %s4b1bt2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1089 = stablehlo.multiply %v1086, %v1087 : tensor<256x512x7x7xf32>
    %v1090 = stablehlo.add %v1089, %v1088 : tensor<256x512x7x7xf32>
    %v1091 = stablehlo.reshape %v1090 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1092 = stablehlo.reshape %v1091 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1093 = stablehlo.reshape %v1037 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1094 = stablehlo.add %v1092, %v1093 : tensor<256x512x7x7xf32>
    %v1095 = stablehlo.reshape %v1094 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1096 = stablehlo.reshape %v1095 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1097 = stablehlo.constant dense<0.0> : tensor<256x512x7x7xf32>
    %v1098 = stablehlo.maximum %v1096, %v1097 : tensor<256x512x7x7xf32>
    %v1099 = stablehlo.reshape %v1098 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1100 = stablehlo.reshape %v1099 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1101 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1102 = stablehlo.reduce(%v1100 init: %v1101) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512xf32>
    %v1103 = stablehlo.constant dense<49.0> : tensor<256x512xf32>
    %v1104 = stablehlo.divide %v1102, %v1103 : tensor<256x512xf32>
    %v1105 = stablehlo.dot_general %v1104, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x512xf32>, tensor<512x1000xf32>) -> tensor<256x1000xf32>
    %v1106 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<1000xf32>) -> tensor<256x1000xf32>
    %v1107 = stablehlo.add %v1105, %v1106 : tensor<256x1000xf32>
    return %v1107 : tensor<256x1000xf32>
  }
}
