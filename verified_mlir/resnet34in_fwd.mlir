module @m {
  func.func @resnet34in_fwd(%x: tensor<256x150528xf32>, %sW: tensor<64x3x7x7xf32>, %sg: tensor<64xf32>, %sbt: tensor<64xf32>, %s1b0W1: tensor<64x64x3x3xf32>, %s1b0g1: tensor<64xf32>, %s1b0bt1: tensor<64xf32>, %s1b0W2: tensor<64x64x3x3xf32>, %s1b0g2: tensor<64xf32>, %s1b0bt2: tensor<64xf32>, %s1b1W1: tensor<64x64x3x3xf32>, %s1b1g1: tensor<64xf32>, %s1b1bt1: tensor<64xf32>, %s1b1W2: tensor<64x64x3x3xf32>, %s1b1g2: tensor<64xf32>, %s1b1bt2: tensor<64xf32>, %s1b2W1: tensor<64x64x3x3xf32>, %s1b2g1: tensor<64xf32>, %s1b2bt1: tensor<64xf32>, %s1b2W2: tensor<64x64x3x3xf32>, %s1b2g2: tensor<64xf32>, %s1b2bt2: tensor<64xf32>, %d2W1: tensor<128x64x3x3xf32>, %d2g1: tensor<128xf32>, %d2bt1: tensor<128xf32>, %d2W2: tensor<128x128x3x3xf32>, %d2g2: tensor<128xf32>, %d2bt2: tensor<128xf32>, %d2Wp: tensor<128x64x3x3xf32>, %d2gp: tensor<128xf32>, %d2btp: tensor<128xf32>, %s2b0W1: tensor<128x128x3x3xf32>, %s2b0g1: tensor<128xf32>, %s2b0bt1: tensor<128xf32>, %s2b0W2: tensor<128x128x3x3xf32>, %s2b0g2: tensor<128xf32>, %s2b0bt2: tensor<128xf32>, %s2b1W1: tensor<128x128x3x3xf32>, %s2b1g1: tensor<128xf32>, %s2b1bt1: tensor<128xf32>, %s2b1W2: tensor<128x128x3x3xf32>, %s2b1g2: tensor<128xf32>, %s2b1bt2: tensor<128xf32>, %s2b2W1: tensor<128x128x3x3xf32>, %s2b2g1: tensor<128xf32>, %s2b2bt1: tensor<128xf32>, %s2b2W2: tensor<128x128x3x3xf32>, %s2b2g2: tensor<128xf32>, %s2b2bt2: tensor<128xf32>, %d3W1: tensor<256x128x3x3xf32>, %d3g1: tensor<256xf32>, %d3bt1: tensor<256xf32>, %d3W2: tensor<256x256x3x3xf32>, %d3g2: tensor<256xf32>, %d3bt2: tensor<256xf32>, %d3Wp: tensor<256x128x3x3xf32>, %d3gp: tensor<256xf32>, %d3btp: tensor<256xf32>, %s3b0W1: tensor<256x256x3x3xf32>, %s3b0g1: tensor<256xf32>, %s3b0bt1: tensor<256xf32>, %s3b0W2: tensor<256x256x3x3xf32>, %s3b0g2: tensor<256xf32>, %s3b0bt2: tensor<256xf32>, %s3b1W1: tensor<256x256x3x3xf32>, %s3b1g1: tensor<256xf32>, %s3b1bt1: tensor<256xf32>, %s3b1W2: tensor<256x256x3x3xf32>, %s3b1g2: tensor<256xf32>, %s3b1bt2: tensor<256xf32>, %s3b2W1: tensor<256x256x3x3xf32>, %s3b2g1: tensor<256xf32>, %s3b2bt1: tensor<256xf32>, %s3b2W2: tensor<256x256x3x3xf32>, %s3b2g2: tensor<256xf32>, %s3b2bt2: tensor<256xf32>, %s3b3W1: tensor<256x256x3x3xf32>, %s3b3g1: tensor<256xf32>, %s3b3bt1: tensor<256xf32>, %s3b3W2: tensor<256x256x3x3xf32>, %s3b3g2: tensor<256xf32>, %s3b3bt2: tensor<256xf32>, %s3b4W1: tensor<256x256x3x3xf32>, %s3b4g1: tensor<256xf32>, %s3b4bt1: tensor<256xf32>, %s3b4W2: tensor<256x256x3x3xf32>, %s3b4g2: tensor<256xf32>, %s3b4bt2: tensor<256xf32>, %d4W1: tensor<512x256x3x3xf32>, %d4g1: tensor<512xf32>, %d4bt1: tensor<512xf32>, %d4W2: tensor<512x512x3x3xf32>, %d4g2: tensor<512xf32>, %d4bt2: tensor<512xf32>, %d4Wp: tensor<512x256x3x3xf32>, %d4gp: tensor<512xf32>, %d4btp: tensor<512xf32>, %s4b0W1: tensor<512x512x3x3xf32>, %s4b0g1: tensor<512xf32>, %s4b0bt1: tensor<512xf32>, %s4b0W2: tensor<512x512x3x3xf32>, %s4b0g2: tensor<512xf32>, %s4b0bt2: tensor<512xf32>, %s4b1W1: tensor<512x512x3x3xf32>, %s4b1g1: tensor<512xf32>, %s4b1bt1: tensor<512xf32>, %s4b1W2: tensor<512x512x3x3xf32>, %s4b1g2: tensor<512xf32>, %s4b1bt2: tensor<512xf32>, %Wd: tensor<512x1000xf32>, %bd: tensor<1000xf32>) -> tensor<256x1000xf32> {
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
    %v25 = stablehlo.constant dense<0.0> : tensor<256x802816xf32>
    %v26 = stablehlo.maximum %v24, %v25 : tensor<256x802816xf32>
    %v27 = stablehlo.reshape %v26 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v28 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v29 = "stablehlo.reduce_window"(%v27, %v28) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<256x64x56x56xf32>
    %v30 = stablehlo.reshape %v29 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v32 = stablehlo.convolution(%v31, %s1b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v33 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v34 = stablehlo.add %v32, %v33 : tensor<256x64x56x56xf32>
    %v35 = stablehlo.reshape %v34 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v36 = stablehlo.reshape %v35 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v37 = stablehlo.constant dense<0.0> : tensor<f32>
    %v38 = stablehlo.constant dense<3136.0> : tensor<256x64x56x56xf32>
    %v39 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v40 = stablehlo.reduce(%v36 init: %v37) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<256x64xf32>
    %v41 = stablehlo.broadcast_in_dim %v40, dims = [0, 1] : (tensor<256x64xf32>) -> tensor<256x64x56x56xf32>
    %v42 = stablehlo.divide %v41, %v38 : tensor<256x64x56x56xf32>
    %v43 = stablehlo.subtract %v36, %v42 : tensor<256x64x56x56xf32>
    %v44 = stablehlo.multiply %v43, %v43 : tensor<256x64x56x56xf32>
    %v45 = stablehlo.reduce(%v44 init: %v37) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<256x64xf32>
    %v46 = stablehlo.broadcast_in_dim %v45, dims = [0, 1] : (tensor<256x64xf32>) -> tensor<256x64x56x56xf32>
    %v47 = stablehlo.divide %v46, %v38 : tensor<256x64x56x56xf32>
    %v48 = stablehlo.add %v47, %v39 : tensor<256x64x56x56xf32>
    %v49 = stablehlo.rsqrt %v48 : tensor<256x64x56x56xf32>
    %v50 = stablehlo.multiply %v43, %v49 : tensor<256x64x56x56xf32>
    %v51 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v52 = stablehlo.broadcast_in_dim %s1b0bt1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v53 = stablehlo.multiply %v50, %v51 : tensor<256x64x56x56xf32>
    %v54 = stablehlo.add %v53, %v52 : tensor<256x64x56x56xf32>
    %v55 = stablehlo.reshape %v54 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v56 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v57 = stablehlo.maximum %v55, %v56 : tensor<256x200704xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v59 = stablehlo.convolution(%v58, %s1b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v60 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v61 = stablehlo.add %v59, %v60 : tensor<256x64x56x56xf32>
    %v62 = stablehlo.reshape %v61 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v63 = stablehlo.reshape %v62 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v64 = stablehlo.constant dense<0.0> : tensor<f32>
    %v65 = stablehlo.constant dense<3136.0> : tensor<256x64x56x56xf32>
    %v66 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v67 = stablehlo.reduce(%v63 init: %v64) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<256x64xf32>
    %v68 = stablehlo.broadcast_in_dim %v67, dims = [0, 1] : (tensor<256x64xf32>) -> tensor<256x64x56x56xf32>
    %v69 = stablehlo.divide %v68, %v65 : tensor<256x64x56x56xf32>
    %v70 = stablehlo.subtract %v63, %v69 : tensor<256x64x56x56xf32>
    %v71 = stablehlo.multiply %v70, %v70 : tensor<256x64x56x56xf32>
    %v72 = stablehlo.reduce(%v71 init: %v64) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<256x64xf32>
    %v73 = stablehlo.broadcast_in_dim %v72, dims = [0, 1] : (tensor<256x64xf32>) -> tensor<256x64x56x56xf32>
    %v74 = stablehlo.divide %v73, %v65 : tensor<256x64x56x56xf32>
    %v75 = stablehlo.add %v74, %v66 : tensor<256x64x56x56xf32>
    %v76 = stablehlo.rsqrt %v75 : tensor<256x64x56x56xf32>
    %v77 = stablehlo.multiply %v70, %v76 : tensor<256x64x56x56xf32>
    %v78 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v79 = stablehlo.broadcast_in_dim %s1b0bt2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v80 = stablehlo.multiply %v77, %v78 : tensor<256x64x56x56xf32>
    %v81 = stablehlo.add %v80, %v79 : tensor<256x64x56x56xf32>
    %v82 = stablehlo.reshape %v81 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v83 = stablehlo.add %v82, %v30 : tensor<256x200704xf32>
    %v84 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v85 = stablehlo.maximum %v83, %v84 : tensor<256x200704xf32>
    %v86 = stablehlo.reshape %v85 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v87 = stablehlo.convolution(%v86, %s1b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v88 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v89 = stablehlo.add %v87, %v88 : tensor<256x64x56x56xf32>
    %v90 = stablehlo.reshape %v89 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v91 = stablehlo.reshape %v90 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v92 = stablehlo.constant dense<0.0> : tensor<f32>
    %v93 = stablehlo.constant dense<3136.0> : tensor<256x64x56x56xf32>
    %v94 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v95 = stablehlo.reduce(%v91 init: %v92) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<256x64xf32>
    %v96 = stablehlo.broadcast_in_dim %v95, dims = [0, 1] : (tensor<256x64xf32>) -> tensor<256x64x56x56xf32>
    %v97 = stablehlo.divide %v96, %v93 : tensor<256x64x56x56xf32>
    %v98 = stablehlo.subtract %v91, %v97 : tensor<256x64x56x56xf32>
    %v99 = stablehlo.multiply %v98, %v98 : tensor<256x64x56x56xf32>
    %v100 = stablehlo.reduce(%v99 init: %v92) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<256x64xf32>
    %v101 = stablehlo.broadcast_in_dim %v100, dims = [0, 1] : (tensor<256x64xf32>) -> tensor<256x64x56x56xf32>
    %v102 = stablehlo.divide %v101, %v93 : tensor<256x64x56x56xf32>
    %v103 = stablehlo.add %v102, %v94 : tensor<256x64x56x56xf32>
    %v104 = stablehlo.rsqrt %v103 : tensor<256x64x56x56xf32>
    %v105 = stablehlo.multiply %v98, %v104 : tensor<256x64x56x56xf32>
    %v106 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v107 = stablehlo.broadcast_in_dim %s1b1bt1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v108 = stablehlo.multiply %v105, %v106 : tensor<256x64x56x56xf32>
    %v109 = stablehlo.add %v108, %v107 : tensor<256x64x56x56xf32>
    %v110 = stablehlo.reshape %v109 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v111 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v112 = stablehlo.maximum %v110, %v111 : tensor<256x200704xf32>
    %v113 = stablehlo.reshape %v112 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v114 = stablehlo.convolution(%v113, %s1b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v115 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v116 = stablehlo.add %v114, %v115 : tensor<256x64x56x56xf32>
    %v117 = stablehlo.reshape %v116 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v118 = stablehlo.reshape %v117 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v119 = stablehlo.constant dense<0.0> : tensor<f32>
    %v120 = stablehlo.constant dense<3136.0> : tensor<256x64x56x56xf32>
    %v121 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v122 = stablehlo.reduce(%v118 init: %v119) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<256x64xf32>
    %v123 = stablehlo.broadcast_in_dim %v122, dims = [0, 1] : (tensor<256x64xf32>) -> tensor<256x64x56x56xf32>
    %v124 = stablehlo.divide %v123, %v120 : tensor<256x64x56x56xf32>
    %v125 = stablehlo.subtract %v118, %v124 : tensor<256x64x56x56xf32>
    %v126 = stablehlo.multiply %v125, %v125 : tensor<256x64x56x56xf32>
    %v127 = stablehlo.reduce(%v126 init: %v119) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<256x64xf32>
    %v128 = stablehlo.broadcast_in_dim %v127, dims = [0, 1] : (tensor<256x64xf32>) -> tensor<256x64x56x56xf32>
    %v129 = stablehlo.divide %v128, %v120 : tensor<256x64x56x56xf32>
    %v130 = stablehlo.add %v129, %v121 : tensor<256x64x56x56xf32>
    %v131 = stablehlo.rsqrt %v130 : tensor<256x64x56x56xf32>
    %v132 = stablehlo.multiply %v125, %v131 : tensor<256x64x56x56xf32>
    %v133 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v134 = stablehlo.broadcast_in_dim %s1b1bt2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v135 = stablehlo.multiply %v132, %v133 : tensor<256x64x56x56xf32>
    %v136 = stablehlo.add %v135, %v134 : tensor<256x64x56x56xf32>
    %v137 = stablehlo.reshape %v136 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v138 = stablehlo.add %v137, %v85 : tensor<256x200704xf32>
    %v139 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v140 = stablehlo.maximum %v138, %v139 : tensor<256x200704xf32>
    %v141 = stablehlo.reshape %v140 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v142 = stablehlo.convolution(%v141, %s1b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v143 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v144 = stablehlo.add %v142, %v143 : tensor<256x64x56x56xf32>
    %v145 = stablehlo.reshape %v144 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v146 = stablehlo.reshape %v145 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v147 = stablehlo.constant dense<0.0> : tensor<f32>
    %v148 = stablehlo.constant dense<3136.0> : tensor<256x64x56x56xf32>
    %v149 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v150 = stablehlo.reduce(%v146 init: %v147) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<256x64xf32>
    %v151 = stablehlo.broadcast_in_dim %v150, dims = [0, 1] : (tensor<256x64xf32>) -> tensor<256x64x56x56xf32>
    %v152 = stablehlo.divide %v151, %v148 : tensor<256x64x56x56xf32>
    %v153 = stablehlo.subtract %v146, %v152 : tensor<256x64x56x56xf32>
    %v154 = stablehlo.multiply %v153, %v153 : tensor<256x64x56x56xf32>
    %v155 = stablehlo.reduce(%v154 init: %v147) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<256x64xf32>
    %v156 = stablehlo.broadcast_in_dim %v155, dims = [0, 1] : (tensor<256x64xf32>) -> tensor<256x64x56x56xf32>
    %v157 = stablehlo.divide %v156, %v148 : tensor<256x64x56x56xf32>
    %v158 = stablehlo.add %v157, %v149 : tensor<256x64x56x56xf32>
    %v159 = stablehlo.rsqrt %v158 : tensor<256x64x56x56xf32>
    %v160 = stablehlo.multiply %v153, %v159 : tensor<256x64x56x56xf32>
    %v161 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v162 = stablehlo.broadcast_in_dim %s1b2bt1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v163 = stablehlo.multiply %v160, %v161 : tensor<256x64x56x56xf32>
    %v164 = stablehlo.add %v163, %v162 : tensor<256x64x56x56xf32>
    %v165 = stablehlo.reshape %v164 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v166 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v167 = stablehlo.maximum %v165, %v166 : tensor<256x200704xf32>
    %v168 = stablehlo.reshape %v167 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v169 = stablehlo.convolution(%v168, %s1b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v170 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v171 = stablehlo.add %v169, %v170 : tensor<256x64x56x56xf32>
    %v172 = stablehlo.reshape %v171 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v173 = stablehlo.reshape %v172 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v174 = stablehlo.constant dense<0.0> : tensor<f32>
    %v175 = stablehlo.constant dense<3136.0> : tensor<256x64x56x56xf32>
    %v176 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v177 = stablehlo.reduce(%v173 init: %v174) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<256x64xf32>
    %v178 = stablehlo.broadcast_in_dim %v177, dims = [0, 1] : (tensor<256x64xf32>) -> tensor<256x64x56x56xf32>
    %v179 = stablehlo.divide %v178, %v175 : tensor<256x64x56x56xf32>
    %v180 = stablehlo.subtract %v173, %v179 : tensor<256x64x56x56xf32>
    %v181 = stablehlo.multiply %v180, %v180 : tensor<256x64x56x56xf32>
    %v182 = stablehlo.reduce(%v181 init: %v174) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<256x64xf32>
    %v183 = stablehlo.broadcast_in_dim %v182, dims = [0, 1] : (tensor<256x64xf32>) -> tensor<256x64x56x56xf32>
    %v184 = stablehlo.divide %v183, %v175 : tensor<256x64x56x56xf32>
    %v185 = stablehlo.add %v184, %v176 : tensor<256x64x56x56xf32>
    %v186 = stablehlo.rsqrt %v185 : tensor<256x64x56x56xf32>
    %v187 = stablehlo.multiply %v180, %v186 : tensor<256x64x56x56xf32>
    %v188 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v189 = stablehlo.broadcast_in_dim %s1b2bt2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v190 = stablehlo.multiply %v187, %v188 : tensor<256x64x56x56xf32>
    %v191 = stablehlo.add %v190, %v189 : tensor<256x64x56x56xf32>
    %v192 = stablehlo.reshape %v191 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v193 = stablehlo.add %v192, %v140 : tensor<256x200704xf32>
    %v194 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v195 = stablehlo.maximum %v193, %v194 : tensor<256x200704xf32>
    %v196 = stablehlo.reshape %v195 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v197 = stablehlo.convolution(%v196, %d2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<128x64x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v198 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v199 = stablehlo.add %v197, %v198 : tensor<256x128x28x28xf32>
    %v200 = stablehlo.reshape %v199 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v201 = stablehlo.reshape %v200 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v202 = stablehlo.constant dense<0.0> : tensor<f32>
    %v203 = stablehlo.constant dense<784.0> : tensor<256x128x28x28xf32>
    %v204 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v205 = stablehlo.reduce(%v201 init: %v202) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128xf32>
    %v206 = stablehlo.broadcast_in_dim %v205, dims = [0, 1] : (tensor<256x128xf32>) -> tensor<256x128x28x28xf32>
    %v207 = stablehlo.divide %v206, %v203 : tensor<256x128x28x28xf32>
    %v208 = stablehlo.subtract %v201, %v207 : tensor<256x128x28x28xf32>
    %v209 = stablehlo.multiply %v208, %v208 : tensor<256x128x28x28xf32>
    %v210 = stablehlo.reduce(%v209 init: %v202) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128xf32>
    %v211 = stablehlo.broadcast_in_dim %v210, dims = [0, 1] : (tensor<256x128xf32>) -> tensor<256x128x28x28xf32>
    %v212 = stablehlo.divide %v211, %v203 : tensor<256x128x28x28xf32>
    %v213 = stablehlo.add %v212, %v204 : tensor<256x128x28x28xf32>
    %v214 = stablehlo.rsqrt %v213 : tensor<256x128x28x28xf32>
    %v215 = stablehlo.multiply %v208, %v214 : tensor<256x128x28x28xf32>
    %v216 = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v217 = stablehlo.broadcast_in_dim %d2bt1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v218 = stablehlo.multiply %v215, %v216 : tensor<256x128x28x28xf32>
    %v219 = stablehlo.add %v218, %v217 : tensor<256x128x28x28xf32>
    %v220 = stablehlo.reshape %v219 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v221 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v222 = stablehlo.maximum %v220, %v221 : tensor<256x100352xf32>
    %v223 = stablehlo.reshape %v222 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v224 = stablehlo.convolution(%v223, %d2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v225 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v226 = stablehlo.add %v224, %v225 : tensor<256x128x28x28xf32>
    %v227 = stablehlo.reshape %v226 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v228 = stablehlo.reshape %v227 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v229 = stablehlo.constant dense<0.0> : tensor<f32>
    %v230 = stablehlo.constant dense<784.0> : tensor<256x128x28x28xf32>
    %v231 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v232 = stablehlo.reduce(%v228 init: %v229) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128xf32>
    %v233 = stablehlo.broadcast_in_dim %v232, dims = [0, 1] : (tensor<256x128xf32>) -> tensor<256x128x28x28xf32>
    %v234 = stablehlo.divide %v233, %v230 : tensor<256x128x28x28xf32>
    %v235 = stablehlo.subtract %v228, %v234 : tensor<256x128x28x28xf32>
    %v236 = stablehlo.multiply %v235, %v235 : tensor<256x128x28x28xf32>
    %v237 = stablehlo.reduce(%v236 init: %v229) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128xf32>
    %v238 = stablehlo.broadcast_in_dim %v237, dims = [0, 1] : (tensor<256x128xf32>) -> tensor<256x128x28x28xf32>
    %v239 = stablehlo.divide %v238, %v230 : tensor<256x128x28x28xf32>
    %v240 = stablehlo.add %v239, %v231 : tensor<256x128x28x28xf32>
    %v241 = stablehlo.rsqrt %v240 : tensor<256x128x28x28xf32>
    %v242 = stablehlo.multiply %v235, %v241 : tensor<256x128x28x28xf32>
    %v243 = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v244 = stablehlo.broadcast_in_dim %d2bt2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v245 = stablehlo.multiply %v242, %v243 : tensor<256x128x28x28xf32>
    %v246 = stablehlo.add %v245, %v244 : tensor<256x128x28x28xf32>
    %v247 = stablehlo.reshape %v246 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v248 = stablehlo.reshape %v195 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v249 = stablehlo.convolution(%v248, %d2Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<128x64x3x3xf32>) -> tensor<256x128x28x28xf32>
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
    %v268 = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v269 = stablehlo.broadcast_in_dim %d2btp, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v270 = stablehlo.multiply %v267, %v268 : tensor<256x128x28x28xf32>
    %v271 = stablehlo.add %v270, %v269 : tensor<256x128x28x28xf32>
    %v272 = stablehlo.reshape %v271 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v273 = stablehlo.add %v247, %v272 : tensor<256x100352xf32>
    %v274 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v275 = stablehlo.maximum %v273, %v274 : tensor<256x100352xf32>
    %v276 = stablehlo.reshape %v275 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v277 = stablehlo.convolution(%v276, %s2b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v278 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v279 = stablehlo.add %v277, %v278 : tensor<256x128x28x28xf32>
    %v280 = stablehlo.reshape %v279 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v281 = stablehlo.reshape %v280 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v282 = stablehlo.constant dense<0.0> : tensor<f32>
    %v283 = stablehlo.constant dense<784.0> : tensor<256x128x28x28xf32>
    %v284 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v285 = stablehlo.reduce(%v281 init: %v282) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128xf32>
    %v286 = stablehlo.broadcast_in_dim %v285, dims = [0, 1] : (tensor<256x128xf32>) -> tensor<256x128x28x28xf32>
    %v287 = stablehlo.divide %v286, %v283 : tensor<256x128x28x28xf32>
    %v288 = stablehlo.subtract %v281, %v287 : tensor<256x128x28x28xf32>
    %v289 = stablehlo.multiply %v288, %v288 : tensor<256x128x28x28xf32>
    %v290 = stablehlo.reduce(%v289 init: %v282) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128xf32>
    %v291 = stablehlo.broadcast_in_dim %v290, dims = [0, 1] : (tensor<256x128xf32>) -> tensor<256x128x28x28xf32>
    %v292 = stablehlo.divide %v291, %v283 : tensor<256x128x28x28xf32>
    %v293 = stablehlo.add %v292, %v284 : tensor<256x128x28x28xf32>
    %v294 = stablehlo.rsqrt %v293 : tensor<256x128x28x28xf32>
    %v295 = stablehlo.multiply %v288, %v294 : tensor<256x128x28x28xf32>
    %v296 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v297 = stablehlo.broadcast_in_dim %s2b0bt1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v298 = stablehlo.multiply %v295, %v296 : tensor<256x128x28x28xf32>
    %v299 = stablehlo.add %v298, %v297 : tensor<256x128x28x28xf32>
    %v300 = stablehlo.reshape %v299 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v301 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v302 = stablehlo.maximum %v300, %v301 : tensor<256x100352xf32>
    %v303 = stablehlo.reshape %v302 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v304 = stablehlo.convolution(%v303, %s2b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v305 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v306 = stablehlo.add %v304, %v305 : tensor<256x128x28x28xf32>
    %v307 = stablehlo.reshape %v306 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v308 = stablehlo.reshape %v307 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v309 = stablehlo.constant dense<0.0> : tensor<f32>
    %v310 = stablehlo.constant dense<784.0> : tensor<256x128x28x28xf32>
    %v311 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v312 = stablehlo.reduce(%v308 init: %v309) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128xf32>
    %v313 = stablehlo.broadcast_in_dim %v312, dims = [0, 1] : (tensor<256x128xf32>) -> tensor<256x128x28x28xf32>
    %v314 = stablehlo.divide %v313, %v310 : tensor<256x128x28x28xf32>
    %v315 = stablehlo.subtract %v308, %v314 : tensor<256x128x28x28xf32>
    %v316 = stablehlo.multiply %v315, %v315 : tensor<256x128x28x28xf32>
    %v317 = stablehlo.reduce(%v316 init: %v309) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128xf32>
    %v318 = stablehlo.broadcast_in_dim %v317, dims = [0, 1] : (tensor<256x128xf32>) -> tensor<256x128x28x28xf32>
    %v319 = stablehlo.divide %v318, %v310 : tensor<256x128x28x28xf32>
    %v320 = stablehlo.add %v319, %v311 : tensor<256x128x28x28xf32>
    %v321 = stablehlo.rsqrt %v320 : tensor<256x128x28x28xf32>
    %v322 = stablehlo.multiply %v315, %v321 : tensor<256x128x28x28xf32>
    %v323 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v324 = stablehlo.broadcast_in_dim %s2b0bt2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v325 = stablehlo.multiply %v322, %v323 : tensor<256x128x28x28xf32>
    %v326 = stablehlo.add %v325, %v324 : tensor<256x128x28x28xf32>
    %v327 = stablehlo.reshape %v326 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v328 = stablehlo.add %v327, %v275 : tensor<256x100352xf32>
    %v329 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v330 = stablehlo.maximum %v328, %v329 : tensor<256x100352xf32>
    %v331 = stablehlo.reshape %v330 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v332 = stablehlo.convolution(%v331, %s2b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v333 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v334 = stablehlo.add %v332, %v333 : tensor<256x128x28x28xf32>
    %v335 = stablehlo.reshape %v334 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v336 = stablehlo.reshape %v335 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v337 = stablehlo.constant dense<0.0> : tensor<f32>
    %v338 = stablehlo.constant dense<784.0> : tensor<256x128x28x28xf32>
    %v339 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v340 = stablehlo.reduce(%v336 init: %v337) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128xf32>
    %v341 = stablehlo.broadcast_in_dim %v340, dims = [0, 1] : (tensor<256x128xf32>) -> tensor<256x128x28x28xf32>
    %v342 = stablehlo.divide %v341, %v338 : tensor<256x128x28x28xf32>
    %v343 = stablehlo.subtract %v336, %v342 : tensor<256x128x28x28xf32>
    %v344 = stablehlo.multiply %v343, %v343 : tensor<256x128x28x28xf32>
    %v345 = stablehlo.reduce(%v344 init: %v337) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128xf32>
    %v346 = stablehlo.broadcast_in_dim %v345, dims = [0, 1] : (tensor<256x128xf32>) -> tensor<256x128x28x28xf32>
    %v347 = stablehlo.divide %v346, %v338 : tensor<256x128x28x28xf32>
    %v348 = stablehlo.add %v347, %v339 : tensor<256x128x28x28xf32>
    %v349 = stablehlo.rsqrt %v348 : tensor<256x128x28x28xf32>
    %v350 = stablehlo.multiply %v343, %v349 : tensor<256x128x28x28xf32>
    %v351 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v352 = stablehlo.broadcast_in_dim %s2b1bt1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v353 = stablehlo.multiply %v350, %v351 : tensor<256x128x28x28xf32>
    %v354 = stablehlo.add %v353, %v352 : tensor<256x128x28x28xf32>
    %v355 = stablehlo.reshape %v354 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v356 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v357 = stablehlo.maximum %v355, %v356 : tensor<256x100352xf32>
    %v358 = stablehlo.reshape %v357 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v359 = stablehlo.convolution(%v358, %s2b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v360 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v361 = stablehlo.add %v359, %v360 : tensor<256x128x28x28xf32>
    %v362 = stablehlo.reshape %v361 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v363 = stablehlo.reshape %v362 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v364 = stablehlo.constant dense<0.0> : tensor<f32>
    %v365 = stablehlo.constant dense<784.0> : tensor<256x128x28x28xf32>
    %v366 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v367 = stablehlo.reduce(%v363 init: %v364) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128xf32>
    %v368 = stablehlo.broadcast_in_dim %v367, dims = [0, 1] : (tensor<256x128xf32>) -> tensor<256x128x28x28xf32>
    %v369 = stablehlo.divide %v368, %v365 : tensor<256x128x28x28xf32>
    %v370 = stablehlo.subtract %v363, %v369 : tensor<256x128x28x28xf32>
    %v371 = stablehlo.multiply %v370, %v370 : tensor<256x128x28x28xf32>
    %v372 = stablehlo.reduce(%v371 init: %v364) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128xf32>
    %v373 = stablehlo.broadcast_in_dim %v372, dims = [0, 1] : (tensor<256x128xf32>) -> tensor<256x128x28x28xf32>
    %v374 = stablehlo.divide %v373, %v365 : tensor<256x128x28x28xf32>
    %v375 = stablehlo.add %v374, %v366 : tensor<256x128x28x28xf32>
    %v376 = stablehlo.rsqrt %v375 : tensor<256x128x28x28xf32>
    %v377 = stablehlo.multiply %v370, %v376 : tensor<256x128x28x28xf32>
    %v378 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v379 = stablehlo.broadcast_in_dim %s2b1bt2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v380 = stablehlo.multiply %v377, %v378 : tensor<256x128x28x28xf32>
    %v381 = stablehlo.add %v380, %v379 : tensor<256x128x28x28xf32>
    %v382 = stablehlo.reshape %v381 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v383 = stablehlo.add %v382, %v330 : tensor<256x100352xf32>
    %v384 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v385 = stablehlo.maximum %v383, %v384 : tensor<256x100352xf32>
    %v386 = stablehlo.reshape %v385 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v387 = stablehlo.convolution(%v386, %s2b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v388 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v389 = stablehlo.add %v387, %v388 : tensor<256x128x28x28xf32>
    %v390 = stablehlo.reshape %v389 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v391 = stablehlo.reshape %v390 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v392 = stablehlo.constant dense<0.0> : tensor<f32>
    %v393 = stablehlo.constant dense<784.0> : tensor<256x128x28x28xf32>
    %v394 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v395 = stablehlo.reduce(%v391 init: %v392) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128xf32>
    %v396 = stablehlo.broadcast_in_dim %v395, dims = [0, 1] : (tensor<256x128xf32>) -> tensor<256x128x28x28xf32>
    %v397 = stablehlo.divide %v396, %v393 : tensor<256x128x28x28xf32>
    %v398 = stablehlo.subtract %v391, %v397 : tensor<256x128x28x28xf32>
    %v399 = stablehlo.multiply %v398, %v398 : tensor<256x128x28x28xf32>
    %v400 = stablehlo.reduce(%v399 init: %v392) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128xf32>
    %v401 = stablehlo.broadcast_in_dim %v400, dims = [0, 1] : (tensor<256x128xf32>) -> tensor<256x128x28x28xf32>
    %v402 = stablehlo.divide %v401, %v393 : tensor<256x128x28x28xf32>
    %v403 = stablehlo.add %v402, %v394 : tensor<256x128x28x28xf32>
    %v404 = stablehlo.rsqrt %v403 : tensor<256x128x28x28xf32>
    %v405 = stablehlo.multiply %v398, %v404 : tensor<256x128x28x28xf32>
    %v406 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v407 = stablehlo.broadcast_in_dim %s2b2bt1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v408 = stablehlo.multiply %v405, %v406 : tensor<256x128x28x28xf32>
    %v409 = stablehlo.add %v408, %v407 : tensor<256x128x28x28xf32>
    %v410 = stablehlo.reshape %v409 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v411 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v412 = stablehlo.maximum %v410, %v411 : tensor<256x100352xf32>
    %v413 = stablehlo.reshape %v412 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v414 = stablehlo.convolution(%v413, %s2b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v415 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v416 = stablehlo.add %v414, %v415 : tensor<256x128x28x28xf32>
    %v417 = stablehlo.reshape %v416 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v418 = stablehlo.reshape %v417 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v419 = stablehlo.constant dense<0.0> : tensor<f32>
    %v420 = stablehlo.constant dense<784.0> : tensor<256x128x28x28xf32>
    %v421 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v422 = stablehlo.reduce(%v418 init: %v419) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128xf32>
    %v423 = stablehlo.broadcast_in_dim %v422, dims = [0, 1] : (tensor<256x128xf32>) -> tensor<256x128x28x28xf32>
    %v424 = stablehlo.divide %v423, %v420 : tensor<256x128x28x28xf32>
    %v425 = stablehlo.subtract %v418, %v424 : tensor<256x128x28x28xf32>
    %v426 = stablehlo.multiply %v425, %v425 : tensor<256x128x28x28xf32>
    %v427 = stablehlo.reduce(%v426 init: %v419) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128xf32>
    %v428 = stablehlo.broadcast_in_dim %v427, dims = [0, 1] : (tensor<256x128xf32>) -> tensor<256x128x28x28xf32>
    %v429 = stablehlo.divide %v428, %v420 : tensor<256x128x28x28xf32>
    %v430 = stablehlo.add %v429, %v421 : tensor<256x128x28x28xf32>
    %v431 = stablehlo.rsqrt %v430 : tensor<256x128x28x28xf32>
    %v432 = stablehlo.multiply %v425, %v431 : tensor<256x128x28x28xf32>
    %v433 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v434 = stablehlo.broadcast_in_dim %s2b2bt2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v435 = stablehlo.multiply %v432, %v433 : tensor<256x128x28x28xf32>
    %v436 = stablehlo.add %v435, %v434 : tensor<256x128x28x28xf32>
    %v437 = stablehlo.reshape %v436 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v438 = stablehlo.add %v437, %v385 : tensor<256x100352xf32>
    %v439 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v440 = stablehlo.maximum %v438, %v439 : tensor<256x100352xf32>
    %v441 = stablehlo.reshape %v440 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v442 = stablehlo.convolution(%v441, %d3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<256x128x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v443 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v444 = stablehlo.add %v442, %v443 : tensor<256x256x14x14xf32>
    %v445 = stablehlo.reshape %v444 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v446 = stablehlo.reshape %v445 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v447 = stablehlo.constant dense<0.0> : tensor<f32>
    %v448 = stablehlo.constant dense<196.0> : tensor<256x256x14x14xf32>
    %v449 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v450 = stablehlo.reduce(%v446 init: %v447) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v451 = stablehlo.broadcast_in_dim %v450, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v452 = stablehlo.divide %v451, %v448 : tensor<256x256x14x14xf32>
    %v453 = stablehlo.subtract %v446, %v452 : tensor<256x256x14x14xf32>
    %v454 = stablehlo.multiply %v453, %v453 : tensor<256x256x14x14xf32>
    %v455 = stablehlo.reduce(%v454 init: %v447) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v456 = stablehlo.broadcast_in_dim %v455, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v457 = stablehlo.divide %v456, %v448 : tensor<256x256x14x14xf32>
    %v458 = stablehlo.add %v457, %v449 : tensor<256x256x14x14xf32>
    %v459 = stablehlo.rsqrt %v458 : tensor<256x256x14x14xf32>
    %v460 = stablehlo.multiply %v453, %v459 : tensor<256x256x14x14xf32>
    %v461 = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v462 = stablehlo.broadcast_in_dim %d3bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v463 = stablehlo.multiply %v460, %v461 : tensor<256x256x14x14xf32>
    %v464 = stablehlo.add %v463, %v462 : tensor<256x256x14x14xf32>
    %v465 = stablehlo.reshape %v464 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v466 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v467 = stablehlo.maximum %v465, %v466 : tensor<256x50176xf32>
    %v468 = stablehlo.reshape %v467 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v469 = stablehlo.convolution(%v468, %d3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v470 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v471 = stablehlo.add %v469, %v470 : tensor<256x256x14x14xf32>
    %v472 = stablehlo.reshape %v471 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v473 = stablehlo.reshape %v472 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v474 = stablehlo.constant dense<0.0> : tensor<f32>
    %v475 = stablehlo.constant dense<196.0> : tensor<256x256x14x14xf32>
    %v476 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v477 = stablehlo.reduce(%v473 init: %v474) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v478 = stablehlo.broadcast_in_dim %v477, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v479 = stablehlo.divide %v478, %v475 : tensor<256x256x14x14xf32>
    %v480 = stablehlo.subtract %v473, %v479 : tensor<256x256x14x14xf32>
    %v481 = stablehlo.multiply %v480, %v480 : tensor<256x256x14x14xf32>
    %v482 = stablehlo.reduce(%v481 init: %v474) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v483 = stablehlo.broadcast_in_dim %v482, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v484 = stablehlo.divide %v483, %v475 : tensor<256x256x14x14xf32>
    %v485 = stablehlo.add %v484, %v476 : tensor<256x256x14x14xf32>
    %v486 = stablehlo.rsqrt %v485 : tensor<256x256x14x14xf32>
    %v487 = stablehlo.multiply %v480, %v486 : tensor<256x256x14x14xf32>
    %v488 = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v489 = stablehlo.broadcast_in_dim %d3bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v490 = stablehlo.multiply %v487, %v488 : tensor<256x256x14x14xf32>
    %v491 = stablehlo.add %v490, %v489 : tensor<256x256x14x14xf32>
    %v492 = stablehlo.reshape %v491 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v493 = stablehlo.reshape %v440 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v494 = stablehlo.convolution(%v493, %d3Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<256x128x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v495 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v496 = stablehlo.add %v494, %v495 : tensor<256x256x14x14xf32>
    %v497 = stablehlo.reshape %v496 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v498 = stablehlo.reshape %v497 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v499 = stablehlo.constant dense<0.0> : tensor<f32>
    %v500 = stablehlo.constant dense<196.0> : tensor<256x256x14x14xf32>
    %v501 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v502 = stablehlo.reduce(%v498 init: %v499) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v503 = stablehlo.broadcast_in_dim %v502, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v504 = stablehlo.divide %v503, %v500 : tensor<256x256x14x14xf32>
    %v505 = stablehlo.subtract %v498, %v504 : tensor<256x256x14x14xf32>
    %v506 = stablehlo.multiply %v505, %v505 : tensor<256x256x14x14xf32>
    %v507 = stablehlo.reduce(%v506 init: %v499) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v508 = stablehlo.broadcast_in_dim %v507, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v509 = stablehlo.divide %v508, %v500 : tensor<256x256x14x14xf32>
    %v510 = stablehlo.add %v509, %v501 : tensor<256x256x14x14xf32>
    %v511 = stablehlo.rsqrt %v510 : tensor<256x256x14x14xf32>
    %v512 = stablehlo.multiply %v505, %v511 : tensor<256x256x14x14xf32>
    %v513 = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v514 = stablehlo.broadcast_in_dim %d3btp, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v515 = stablehlo.multiply %v512, %v513 : tensor<256x256x14x14xf32>
    %v516 = stablehlo.add %v515, %v514 : tensor<256x256x14x14xf32>
    %v517 = stablehlo.reshape %v516 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v518 = stablehlo.add %v492, %v517 : tensor<256x50176xf32>
    %v519 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v520 = stablehlo.maximum %v518, %v519 : tensor<256x50176xf32>
    %v521 = stablehlo.reshape %v520 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v522 = stablehlo.convolution(%v521, %s3b0W1)
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
    %v541 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v542 = stablehlo.broadcast_in_dim %s3b0bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v543 = stablehlo.multiply %v540, %v541 : tensor<256x256x14x14xf32>
    %v544 = stablehlo.add %v543, %v542 : tensor<256x256x14x14xf32>
    %v545 = stablehlo.reshape %v544 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v546 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v547 = stablehlo.maximum %v545, %v546 : tensor<256x50176xf32>
    %v548 = stablehlo.reshape %v547 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v549 = stablehlo.convolution(%v548, %s3b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v550 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v551 = stablehlo.add %v549, %v550 : tensor<256x256x14x14xf32>
    %v552 = stablehlo.reshape %v551 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v553 = stablehlo.reshape %v552 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v554 = stablehlo.constant dense<0.0> : tensor<f32>
    %v555 = stablehlo.constant dense<196.0> : tensor<256x256x14x14xf32>
    %v556 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v557 = stablehlo.reduce(%v553 init: %v554) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v558 = stablehlo.broadcast_in_dim %v557, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v559 = stablehlo.divide %v558, %v555 : tensor<256x256x14x14xf32>
    %v560 = stablehlo.subtract %v553, %v559 : tensor<256x256x14x14xf32>
    %v561 = stablehlo.multiply %v560, %v560 : tensor<256x256x14x14xf32>
    %v562 = stablehlo.reduce(%v561 init: %v554) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v563 = stablehlo.broadcast_in_dim %v562, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v564 = stablehlo.divide %v563, %v555 : tensor<256x256x14x14xf32>
    %v565 = stablehlo.add %v564, %v556 : tensor<256x256x14x14xf32>
    %v566 = stablehlo.rsqrt %v565 : tensor<256x256x14x14xf32>
    %v567 = stablehlo.multiply %v560, %v566 : tensor<256x256x14x14xf32>
    %v568 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v569 = stablehlo.broadcast_in_dim %s3b0bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v570 = stablehlo.multiply %v567, %v568 : tensor<256x256x14x14xf32>
    %v571 = stablehlo.add %v570, %v569 : tensor<256x256x14x14xf32>
    %v572 = stablehlo.reshape %v571 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v573 = stablehlo.add %v572, %v520 : tensor<256x50176xf32>
    %v574 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v575 = stablehlo.maximum %v573, %v574 : tensor<256x50176xf32>
    %v576 = stablehlo.reshape %v575 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v577 = stablehlo.convolution(%v576, %s3b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v578 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v579 = stablehlo.add %v577, %v578 : tensor<256x256x14x14xf32>
    %v580 = stablehlo.reshape %v579 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v581 = stablehlo.reshape %v580 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v582 = stablehlo.constant dense<0.0> : tensor<f32>
    %v583 = stablehlo.constant dense<196.0> : tensor<256x256x14x14xf32>
    %v584 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v585 = stablehlo.reduce(%v581 init: %v582) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v586 = stablehlo.broadcast_in_dim %v585, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v587 = stablehlo.divide %v586, %v583 : tensor<256x256x14x14xf32>
    %v588 = stablehlo.subtract %v581, %v587 : tensor<256x256x14x14xf32>
    %v589 = stablehlo.multiply %v588, %v588 : tensor<256x256x14x14xf32>
    %v590 = stablehlo.reduce(%v589 init: %v582) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v591 = stablehlo.broadcast_in_dim %v590, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v592 = stablehlo.divide %v591, %v583 : tensor<256x256x14x14xf32>
    %v593 = stablehlo.add %v592, %v584 : tensor<256x256x14x14xf32>
    %v594 = stablehlo.rsqrt %v593 : tensor<256x256x14x14xf32>
    %v595 = stablehlo.multiply %v588, %v594 : tensor<256x256x14x14xf32>
    %v596 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v597 = stablehlo.broadcast_in_dim %s3b1bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v598 = stablehlo.multiply %v595, %v596 : tensor<256x256x14x14xf32>
    %v599 = stablehlo.add %v598, %v597 : tensor<256x256x14x14xf32>
    %v600 = stablehlo.reshape %v599 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v601 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v602 = stablehlo.maximum %v600, %v601 : tensor<256x50176xf32>
    %v603 = stablehlo.reshape %v602 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v604 = stablehlo.convolution(%v603, %s3b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v605 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v606 = stablehlo.add %v604, %v605 : tensor<256x256x14x14xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v608 = stablehlo.reshape %v607 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v609 = stablehlo.constant dense<0.0> : tensor<f32>
    %v610 = stablehlo.constant dense<196.0> : tensor<256x256x14x14xf32>
    %v611 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v612 = stablehlo.reduce(%v608 init: %v609) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v613 = stablehlo.broadcast_in_dim %v612, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v614 = stablehlo.divide %v613, %v610 : tensor<256x256x14x14xf32>
    %v615 = stablehlo.subtract %v608, %v614 : tensor<256x256x14x14xf32>
    %v616 = stablehlo.multiply %v615, %v615 : tensor<256x256x14x14xf32>
    %v617 = stablehlo.reduce(%v616 init: %v609) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v618 = stablehlo.broadcast_in_dim %v617, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v619 = stablehlo.divide %v618, %v610 : tensor<256x256x14x14xf32>
    %v620 = stablehlo.add %v619, %v611 : tensor<256x256x14x14xf32>
    %v621 = stablehlo.rsqrt %v620 : tensor<256x256x14x14xf32>
    %v622 = stablehlo.multiply %v615, %v621 : tensor<256x256x14x14xf32>
    %v623 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v624 = stablehlo.broadcast_in_dim %s3b1bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v625 = stablehlo.multiply %v622, %v623 : tensor<256x256x14x14xf32>
    %v626 = stablehlo.add %v625, %v624 : tensor<256x256x14x14xf32>
    %v627 = stablehlo.reshape %v626 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v628 = stablehlo.add %v627, %v575 : tensor<256x50176xf32>
    %v629 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v630 = stablehlo.maximum %v628, %v629 : tensor<256x50176xf32>
    %v631 = stablehlo.reshape %v630 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v632 = stablehlo.convolution(%v631, %s3b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v633 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v634 = stablehlo.add %v632, %v633 : tensor<256x256x14x14xf32>
    %v635 = stablehlo.reshape %v634 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v636 = stablehlo.reshape %v635 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v637 = stablehlo.constant dense<0.0> : tensor<f32>
    %v638 = stablehlo.constant dense<196.0> : tensor<256x256x14x14xf32>
    %v639 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v640 = stablehlo.reduce(%v636 init: %v637) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v641 = stablehlo.broadcast_in_dim %v640, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v642 = stablehlo.divide %v641, %v638 : tensor<256x256x14x14xf32>
    %v643 = stablehlo.subtract %v636, %v642 : tensor<256x256x14x14xf32>
    %v644 = stablehlo.multiply %v643, %v643 : tensor<256x256x14x14xf32>
    %v645 = stablehlo.reduce(%v644 init: %v637) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v646 = stablehlo.broadcast_in_dim %v645, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v647 = stablehlo.divide %v646, %v638 : tensor<256x256x14x14xf32>
    %v648 = stablehlo.add %v647, %v639 : tensor<256x256x14x14xf32>
    %v649 = stablehlo.rsqrt %v648 : tensor<256x256x14x14xf32>
    %v650 = stablehlo.multiply %v643, %v649 : tensor<256x256x14x14xf32>
    %v651 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v652 = stablehlo.broadcast_in_dim %s3b2bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v653 = stablehlo.multiply %v650, %v651 : tensor<256x256x14x14xf32>
    %v654 = stablehlo.add %v653, %v652 : tensor<256x256x14x14xf32>
    %v655 = stablehlo.reshape %v654 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v656 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v657 = stablehlo.maximum %v655, %v656 : tensor<256x50176xf32>
    %v658 = stablehlo.reshape %v657 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v659 = stablehlo.convolution(%v658, %s3b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v660 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v661 = stablehlo.add %v659, %v660 : tensor<256x256x14x14xf32>
    %v662 = stablehlo.reshape %v661 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v663 = stablehlo.reshape %v662 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v664 = stablehlo.constant dense<0.0> : tensor<f32>
    %v665 = stablehlo.constant dense<196.0> : tensor<256x256x14x14xf32>
    %v666 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v667 = stablehlo.reduce(%v663 init: %v664) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v668 = stablehlo.broadcast_in_dim %v667, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v669 = stablehlo.divide %v668, %v665 : tensor<256x256x14x14xf32>
    %v670 = stablehlo.subtract %v663, %v669 : tensor<256x256x14x14xf32>
    %v671 = stablehlo.multiply %v670, %v670 : tensor<256x256x14x14xf32>
    %v672 = stablehlo.reduce(%v671 init: %v664) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v673 = stablehlo.broadcast_in_dim %v672, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v674 = stablehlo.divide %v673, %v665 : tensor<256x256x14x14xf32>
    %v675 = stablehlo.add %v674, %v666 : tensor<256x256x14x14xf32>
    %v676 = stablehlo.rsqrt %v675 : tensor<256x256x14x14xf32>
    %v677 = stablehlo.multiply %v670, %v676 : tensor<256x256x14x14xf32>
    %v678 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v679 = stablehlo.broadcast_in_dim %s3b2bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v680 = stablehlo.multiply %v677, %v678 : tensor<256x256x14x14xf32>
    %v681 = stablehlo.add %v680, %v679 : tensor<256x256x14x14xf32>
    %v682 = stablehlo.reshape %v681 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v683 = stablehlo.add %v682, %v630 : tensor<256x50176xf32>
    %v684 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v685 = stablehlo.maximum %v683, %v684 : tensor<256x50176xf32>
    %v686 = stablehlo.reshape %v685 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v687 = stablehlo.convolution(%v686, %s3b3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v688 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v689 = stablehlo.add %v687, %v688 : tensor<256x256x14x14xf32>
    %v690 = stablehlo.reshape %v689 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v691 = stablehlo.reshape %v690 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v692 = stablehlo.constant dense<0.0> : tensor<f32>
    %v693 = stablehlo.constant dense<196.0> : tensor<256x256x14x14xf32>
    %v694 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v695 = stablehlo.reduce(%v691 init: %v692) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v696 = stablehlo.broadcast_in_dim %v695, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v697 = stablehlo.divide %v696, %v693 : tensor<256x256x14x14xf32>
    %v698 = stablehlo.subtract %v691, %v697 : tensor<256x256x14x14xf32>
    %v699 = stablehlo.multiply %v698, %v698 : tensor<256x256x14x14xf32>
    %v700 = stablehlo.reduce(%v699 init: %v692) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v701 = stablehlo.broadcast_in_dim %v700, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v702 = stablehlo.divide %v701, %v693 : tensor<256x256x14x14xf32>
    %v703 = stablehlo.add %v702, %v694 : tensor<256x256x14x14xf32>
    %v704 = stablehlo.rsqrt %v703 : tensor<256x256x14x14xf32>
    %v705 = stablehlo.multiply %v698, %v704 : tensor<256x256x14x14xf32>
    %v706 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v707 = stablehlo.broadcast_in_dim %s3b3bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v708 = stablehlo.multiply %v705, %v706 : tensor<256x256x14x14xf32>
    %v709 = stablehlo.add %v708, %v707 : tensor<256x256x14x14xf32>
    %v710 = stablehlo.reshape %v709 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v711 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v712 = stablehlo.maximum %v710, %v711 : tensor<256x50176xf32>
    %v713 = stablehlo.reshape %v712 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v714 = stablehlo.convolution(%v713, %s3b3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v715 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v716 = stablehlo.add %v714, %v715 : tensor<256x256x14x14xf32>
    %v717 = stablehlo.reshape %v716 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v718 = stablehlo.reshape %v717 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v719 = stablehlo.constant dense<0.0> : tensor<f32>
    %v720 = stablehlo.constant dense<196.0> : tensor<256x256x14x14xf32>
    %v721 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v722 = stablehlo.reduce(%v718 init: %v719) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v723 = stablehlo.broadcast_in_dim %v722, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v724 = stablehlo.divide %v723, %v720 : tensor<256x256x14x14xf32>
    %v725 = stablehlo.subtract %v718, %v724 : tensor<256x256x14x14xf32>
    %v726 = stablehlo.multiply %v725, %v725 : tensor<256x256x14x14xf32>
    %v727 = stablehlo.reduce(%v726 init: %v719) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v728 = stablehlo.broadcast_in_dim %v727, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v729 = stablehlo.divide %v728, %v720 : tensor<256x256x14x14xf32>
    %v730 = stablehlo.add %v729, %v721 : tensor<256x256x14x14xf32>
    %v731 = stablehlo.rsqrt %v730 : tensor<256x256x14x14xf32>
    %v732 = stablehlo.multiply %v725, %v731 : tensor<256x256x14x14xf32>
    %v733 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v734 = stablehlo.broadcast_in_dim %s3b3bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v735 = stablehlo.multiply %v732, %v733 : tensor<256x256x14x14xf32>
    %v736 = stablehlo.add %v735, %v734 : tensor<256x256x14x14xf32>
    %v737 = stablehlo.reshape %v736 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v738 = stablehlo.add %v737, %v685 : tensor<256x50176xf32>
    %v739 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v740 = stablehlo.maximum %v738, %v739 : tensor<256x50176xf32>
    %v741 = stablehlo.reshape %v740 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v742 = stablehlo.convolution(%v741, %s3b4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v743 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v744 = stablehlo.add %v742, %v743 : tensor<256x256x14x14xf32>
    %v745 = stablehlo.reshape %v744 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v746 = stablehlo.reshape %v745 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v747 = stablehlo.constant dense<0.0> : tensor<f32>
    %v748 = stablehlo.constant dense<196.0> : tensor<256x256x14x14xf32>
    %v749 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v750 = stablehlo.reduce(%v746 init: %v747) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v751 = stablehlo.broadcast_in_dim %v750, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v752 = stablehlo.divide %v751, %v748 : tensor<256x256x14x14xf32>
    %v753 = stablehlo.subtract %v746, %v752 : tensor<256x256x14x14xf32>
    %v754 = stablehlo.multiply %v753, %v753 : tensor<256x256x14x14xf32>
    %v755 = stablehlo.reduce(%v754 init: %v747) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v756 = stablehlo.broadcast_in_dim %v755, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v757 = stablehlo.divide %v756, %v748 : tensor<256x256x14x14xf32>
    %v758 = stablehlo.add %v757, %v749 : tensor<256x256x14x14xf32>
    %v759 = stablehlo.rsqrt %v758 : tensor<256x256x14x14xf32>
    %v760 = stablehlo.multiply %v753, %v759 : tensor<256x256x14x14xf32>
    %v761 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v762 = stablehlo.broadcast_in_dim %s3b4bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v763 = stablehlo.multiply %v760, %v761 : tensor<256x256x14x14xf32>
    %v764 = stablehlo.add %v763, %v762 : tensor<256x256x14x14xf32>
    %v765 = stablehlo.reshape %v764 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v766 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v767 = stablehlo.maximum %v765, %v766 : tensor<256x50176xf32>
    %v768 = stablehlo.reshape %v767 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v769 = stablehlo.convolution(%v768, %s3b4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v770 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v771 = stablehlo.add %v769, %v770 : tensor<256x256x14x14xf32>
    %v772 = stablehlo.reshape %v771 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v773 = stablehlo.reshape %v772 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v774 = stablehlo.constant dense<0.0> : tensor<f32>
    %v775 = stablehlo.constant dense<196.0> : tensor<256x256x14x14xf32>
    %v776 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v777 = stablehlo.reduce(%v773 init: %v774) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v778 = stablehlo.broadcast_in_dim %v777, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v779 = stablehlo.divide %v778, %v775 : tensor<256x256x14x14xf32>
    %v780 = stablehlo.subtract %v773, %v779 : tensor<256x256x14x14xf32>
    %v781 = stablehlo.multiply %v780, %v780 : tensor<256x256x14x14xf32>
    %v782 = stablehlo.reduce(%v781 init: %v774) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256xf32>
    %v783 = stablehlo.broadcast_in_dim %v782, dims = [0, 1] : (tensor<256x256xf32>) -> tensor<256x256x14x14xf32>
    %v784 = stablehlo.divide %v783, %v775 : tensor<256x256x14x14xf32>
    %v785 = stablehlo.add %v784, %v776 : tensor<256x256x14x14xf32>
    %v786 = stablehlo.rsqrt %v785 : tensor<256x256x14x14xf32>
    %v787 = stablehlo.multiply %v780, %v786 : tensor<256x256x14x14xf32>
    %v788 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v789 = stablehlo.broadcast_in_dim %s3b4bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v790 = stablehlo.multiply %v787, %v788 : tensor<256x256x14x14xf32>
    %v791 = stablehlo.add %v790, %v789 : tensor<256x256x14x14xf32>
    %v792 = stablehlo.reshape %v791 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v793 = stablehlo.add %v792, %v740 : tensor<256x50176xf32>
    %v794 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v795 = stablehlo.maximum %v793, %v794 : tensor<256x50176xf32>
    %v796 = stablehlo.reshape %v795 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v797 = stablehlo.convolution(%v796, %d4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<512x256x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v798 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v799 = stablehlo.add %v797, %v798 : tensor<256x512x7x7xf32>
    %v800 = stablehlo.reshape %v799 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v801 = stablehlo.reshape %v800 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v802 = stablehlo.constant dense<0.0> : tensor<f32>
    %v803 = stablehlo.constant dense<49.0> : tensor<256x512x7x7xf32>
    %v804 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v805 = stablehlo.reduce(%v801 init: %v802) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512xf32>
    %v806 = stablehlo.broadcast_in_dim %v805, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512x7x7xf32>
    %v807 = stablehlo.divide %v806, %v803 : tensor<256x512x7x7xf32>
    %v808 = stablehlo.subtract %v801, %v807 : tensor<256x512x7x7xf32>
    %v809 = stablehlo.multiply %v808, %v808 : tensor<256x512x7x7xf32>
    %v810 = stablehlo.reduce(%v809 init: %v802) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512xf32>
    %v811 = stablehlo.broadcast_in_dim %v810, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512x7x7xf32>
    %v812 = stablehlo.divide %v811, %v803 : tensor<256x512x7x7xf32>
    %v813 = stablehlo.add %v812, %v804 : tensor<256x512x7x7xf32>
    %v814 = stablehlo.rsqrt %v813 : tensor<256x512x7x7xf32>
    %v815 = stablehlo.multiply %v808, %v814 : tensor<256x512x7x7xf32>
    %v816 = stablehlo.broadcast_in_dim %d4g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v817 = stablehlo.broadcast_in_dim %d4bt1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v818 = stablehlo.multiply %v815, %v816 : tensor<256x512x7x7xf32>
    %v819 = stablehlo.add %v818, %v817 : tensor<256x512x7x7xf32>
    %v820 = stablehlo.reshape %v819 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v821 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v822 = stablehlo.maximum %v820, %v821 : tensor<256x25088xf32>
    %v823 = stablehlo.reshape %v822 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v824 = stablehlo.convolution(%v823, %d4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v825 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v826 = stablehlo.add %v824, %v825 : tensor<256x512x7x7xf32>
    %v827 = stablehlo.reshape %v826 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v828 = stablehlo.reshape %v827 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v829 = stablehlo.constant dense<0.0> : tensor<f32>
    %v830 = stablehlo.constant dense<49.0> : tensor<256x512x7x7xf32>
    %v831 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v832 = stablehlo.reduce(%v828 init: %v829) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512xf32>
    %v833 = stablehlo.broadcast_in_dim %v832, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512x7x7xf32>
    %v834 = stablehlo.divide %v833, %v830 : tensor<256x512x7x7xf32>
    %v835 = stablehlo.subtract %v828, %v834 : tensor<256x512x7x7xf32>
    %v836 = stablehlo.multiply %v835, %v835 : tensor<256x512x7x7xf32>
    %v837 = stablehlo.reduce(%v836 init: %v829) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512xf32>
    %v838 = stablehlo.broadcast_in_dim %v837, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512x7x7xf32>
    %v839 = stablehlo.divide %v838, %v830 : tensor<256x512x7x7xf32>
    %v840 = stablehlo.add %v839, %v831 : tensor<256x512x7x7xf32>
    %v841 = stablehlo.rsqrt %v840 : tensor<256x512x7x7xf32>
    %v842 = stablehlo.multiply %v835, %v841 : tensor<256x512x7x7xf32>
    %v843 = stablehlo.broadcast_in_dim %d4g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v844 = stablehlo.broadcast_in_dim %d4bt2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v845 = stablehlo.multiply %v842, %v843 : tensor<256x512x7x7xf32>
    %v846 = stablehlo.add %v845, %v844 : tensor<256x512x7x7xf32>
    %v847 = stablehlo.reshape %v846 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v848 = stablehlo.reshape %v795 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v849 = stablehlo.convolution(%v848, %d4Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<512x256x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v850 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v851 = stablehlo.add %v849, %v850 : tensor<256x512x7x7xf32>
    %v852 = stablehlo.reshape %v851 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v853 = stablehlo.reshape %v852 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v854 = stablehlo.constant dense<0.0> : tensor<f32>
    %v855 = stablehlo.constant dense<49.0> : tensor<256x512x7x7xf32>
    %v856 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v857 = stablehlo.reduce(%v853 init: %v854) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512xf32>
    %v858 = stablehlo.broadcast_in_dim %v857, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512x7x7xf32>
    %v859 = stablehlo.divide %v858, %v855 : tensor<256x512x7x7xf32>
    %v860 = stablehlo.subtract %v853, %v859 : tensor<256x512x7x7xf32>
    %v861 = stablehlo.multiply %v860, %v860 : tensor<256x512x7x7xf32>
    %v862 = stablehlo.reduce(%v861 init: %v854) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512xf32>
    %v863 = stablehlo.broadcast_in_dim %v862, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512x7x7xf32>
    %v864 = stablehlo.divide %v863, %v855 : tensor<256x512x7x7xf32>
    %v865 = stablehlo.add %v864, %v856 : tensor<256x512x7x7xf32>
    %v866 = stablehlo.rsqrt %v865 : tensor<256x512x7x7xf32>
    %v867 = stablehlo.multiply %v860, %v866 : tensor<256x512x7x7xf32>
    %v868 = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v869 = stablehlo.broadcast_in_dim %d4btp, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v870 = stablehlo.multiply %v867, %v868 : tensor<256x512x7x7xf32>
    %v871 = stablehlo.add %v870, %v869 : tensor<256x512x7x7xf32>
    %v872 = stablehlo.reshape %v871 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v873 = stablehlo.add %v847, %v872 : tensor<256x25088xf32>
    %v874 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v875 = stablehlo.maximum %v873, %v874 : tensor<256x25088xf32>
    %v876 = stablehlo.reshape %v875 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v877 = stablehlo.convolution(%v876, %s4b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v878 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v879 = stablehlo.add %v877, %v878 : tensor<256x512x7x7xf32>
    %v880 = stablehlo.reshape %v879 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v881 = stablehlo.reshape %v880 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v882 = stablehlo.constant dense<0.0> : tensor<f32>
    %v883 = stablehlo.constant dense<49.0> : tensor<256x512x7x7xf32>
    %v884 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v885 = stablehlo.reduce(%v881 init: %v882) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512xf32>
    %v886 = stablehlo.broadcast_in_dim %v885, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512x7x7xf32>
    %v887 = stablehlo.divide %v886, %v883 : tensor<256x512x7x7xf32>
    %v888 = stablehlo.subtract %v881, %v887 : tensor<256x512x7x7xf32>
    %v889 = stablehlo.multiply %v888, %v888 : tensor<256x512x7x7xf32>
    %v890 = stablehlo.reduce(%v889 init: %v882) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512xf32>
    %v891 = stablehlo.broadcast_in_dim %v890, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512x7x7xf32>
    %v892 = stablehlo.divide %v891, %v883 : tensor<256x512x7x7xf32>
    %v893 = stablehlo.add %v892, %v884 : tensor<256x512x7x7xf32>
    %v894 = stablehlo.rsqrt %v893 : tensor<256x512x7x7xf32>
    %v895 = stablehlo.multiply %v888, %v894 : tensor<256x512x7x7xf32>
    %v896 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v897 = stablehlo.broadcast_in_dim %s4b0bt1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v898 = stablehlo.multiply %v895, %v896 : tensor<256x512x7x7xf32>
    %v899 = stablehlo.add %v898, %v897 : tensor<256x512x7x7xf32>
    %v900 = stablehlo.reshape %v899 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v901 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v902 = stablehlo.maximum %v900, %v901 : tensor<256x25088xf32>
    %v903 = stablehlo.reshape %v902 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v904 = stablehlo.convolution(%v903, %s4b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v905 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v906 = stablehlo.add %v904, %v905 : tensor<256x512x7x7xf32>
    %v907 = stablehlo.reshape %v906 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v908 = stablehlo.reshape %v907 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v909 = stablehlo.constant dense<0.0> : tensor<f32>
    %v910 = stablehlo.constant dense<49.0> : tensor<256x512x7x7xf32>
    %v911 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v912 = stablehlo.reduce(%v908 init: %v909) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512xf32>
    %v913 = stablehlo.broadcast_in_dim %v912, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512x7x7xf32>
    %v914 = stablehlo.divide %v913, %v910 : tensor<256x512x7x7xf32>
    %v915 = stablehlo.subtract %v908, %v914 : tensor<256x512x7x7xf32>
    %v916 = stablehlo.multiply %v915, %v915 : tensor<256x512x7x7xf32>
    %v917 = stablehlo.reduce(%v916 init: %v909) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512xf32>
    %v918 = stablehlo.broadcast_in_dim %v917, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512x7x7xf32>
    %v919 = stablehlo.divide %v918, %v910 : tensor<256x512x7x7xf32>
    %v920 = stablehlo.add %v919, %v911 : tensor<256x512x7x7xf32>
    %v921 = stablehlo.rsqrt %v920 : tensor<256x512x7x7xf32>
    %v922 = stablehlo.multiply %v915, %v921 : tensor<256x512x7x7xf32>
    %v923 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v924 = stablehlo.broadcast_in_dim %s4b0bt2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v925 = stablehlo.multiply %v922, %v923 : tensor<256x512x7x7xf32>
    %v926 = stablehlo.add %v925, %v924 : tensor<256x512x7x7xf32>
    %v927 = stablehlo.reshape %v926 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v928 = stablehlo.add %v927, %v875 : tensor<256x25088xf32>
    %v929 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v930 = stablehlo.maximum %v928, %v929 : tensor<256x25088xf32>
    %v931 = stablehlo.reshape %v930 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v932 = stablehlo.convolution(%v931, %s4b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v933 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v934 = stablehlo.add %v932, %v933 : tensor<256x512x7x7xf32>
    %v935 = stablehlo.reshape %v934 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v936 = stablehlo.reshape %v935 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v937 = stablehlo.constant dense<0.0> : tensor<f32>
    %v938 = stablehlo.constant dense<49.0> : tensor<256x512x7x7xf32>
    %v939 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v940 = stablehlo.reduce(%v936 init: %v937) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512xf32>
    %v941 = stablehlo.broadcast_in_dim %v940, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512x7x7xf32>
    %v942 = stablehlo.divide %v941, %v938 : tensor<256x512x7x7xf32>
    %v943 = stablehlo.subtract %v936, %v942 : tensor<256x512x7x7xf32>
    %v944 = stablehlo.multiply %v943, %v943 : tensor<256x512x7x7xf32>
    %v945 = stablehlo.reduce(%v944 init: %v937) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512xf32>
    %v946 = stablehlo.broadcast_in_dim %v945, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512x7x7xf32>
    %v947 = stablehlo.divide %v946, %v938 : tensor<256x512x7x7xf32>
    %v948 = stablehlo.add %v947, %v939 : tensor<256x512x7x7xf32>
    %v949 = stablehlo.rsqrt %v948 : tensor<256x512x7x7xf32>
    %v950 = stablehlo.multiply %v943, %v949 : tensor<256x512x7x7xf32>
    %v951 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v952 = stablehlo.broadcast_in_dim %s4b1bt1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v953 = stablehlo.multiply %v950, %v951 : tensor<256x512x7x7xf32>
    %v954 = stablehlo.add %v953, %v952 : tensor<256x512x7x7xf32>
    %v955 = stablehlo.reshape %v954 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v956 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v957 = stablehlo.maximum %v955, %v956 : tensor<256x25088xf32>
    %v958 = stablehlo.reshape %v957 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v959 = stablehlo.convolution(%v958, %s4b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v960 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v961 = stablehlo.add %v959, %v960 : tensor<256x512x7x7xf32>
    %v962 = stablehlo.reshape %v961 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v963 = stablehlo.reshape %v962 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v964 = stablehlo.constant dense<0.0> : tensor<f32>
    %v965 = stablehlo.constant dense<49.0> : tensor<256x512x7x7xf32>
    %v966 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v967 = stablehlo.reduce(%v963 init: %v964) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512xf32>
    %v968 = stablehlo.broadcast_in_dim %v967, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512x7x7xf32>
    %v969 = stablehlo.divide %v968, %v965 : tensor<256x512x7x7xf32>
    %v970 = stablehlo.subtract %v963, %v969 : tensor<256x512x7x7xf32>
    %v971 = stablehlo.multiply %v970, %v970 : tensor<256x512x7x7xf32>
    %v972 = stablehlo.reduce(%v971 init: %v964) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512xf32>
    %v973 = stablehlo.broadcast_in_dim %v972, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512x7x7xf32>
    %v974 = stablehlo.divide %v973, %v965 : tensor<256x512x7x7xf32>
    %v975 = stablehlo.add %v974, %v966 : tensor<256x512x7x7xf32>
    %v976 = stablehlo.rsqrt %v975 : tensor<256x512x7x7xf32>
    %v977 = stablehlo.multiply %v970, %v976 : tensor<256x512x7x7xf32>
    %v978 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v979 = stablehlo.broadcast_in_dim %s4b1bt2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v980 = stablehlo.multiply %v977, %v978 : tensor<256x512x7x7xf32>
    %v981 = stablehlo.add %v980, %v979 : tensor<256x512x7x7xf32>
    %v982 = stablehlo.reshape %v981 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v983 = stablehlo.add %v982, %v930 : tensor<256x25088xf32>
    %v984 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v985 = stablehlo.maximum %v983, %v984 : tensor<256x25088xf32>
    %v986 = stablehlo.reshape %v985 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v987 = stablehlo.constant dense<0.0> : tensor<f32>
    %v988 = stablehlo.reduce(%v986 init: %v987) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512xf32>
    %v989 = stablehlo.constant dense<49.0> : tensor<256x512xf32>
    %v990 = stablehlo.divide %v988, %v989 : tensor<256x512xf32>
    %v991 = stablehlo.dot_general %v990, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x512xf32>, tensor<512x1000xf32>) -> tensor<256x1000xf32>
    %v992 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<1000xf32>) -> tensor<256x1000xf32>
    %v993 = stablehlo.add %v991, %v992 : tensor<256x1000xf32>
    return %v993 : tensor<256x1000xf32>
  }
}
