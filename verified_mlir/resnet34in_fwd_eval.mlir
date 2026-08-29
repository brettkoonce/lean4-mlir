module @m {
  func.func @resnet34in_fwd_eval(%x: tensor<256x150528xf32>, %sW: tensor<64x3x7x7xf32>, %sg: tensor<64xf32>, %sbt: tensor<64xf32>, %s1b0W1: tensor<64x64x3x3xf32>, %s1b0g1: tensor<64xf32>, %s1b0bt1: tensor<64xf32>, %s1b0W2: tensor<64x64x3x3xf32>, %s1b0g2: tensor<64xf32>, %s1b0bt2: tensor<64xf32>, %s1b1W1: tensor<64x64x3x3xf32>, %s1b1g1: tensor<64xf32>, %s1b1bt1: tensor<64xf32>, %s1b1W2: tensor<64x64x3x3xf32>, %s1b1g2: tensor<64xf32>, %s1b1bt2: tensor<64xf32>, %s1b2W1: tensor<64x64x3x3xf32>, %s1b2g1: tensor<64xf32>, %s1b2bt1: tensor<64xf32>, %s1b2W2: tensor<64x64x3x3xf32>, %s1b2g2: tensor<64xf32>, %s1b2bt2: tensor<64xf32>, %d2W1: tensor<128x64x3x3xf32>, %d2g1: tensor<128xf32>, %d2bt1: tensor<128xf32>, %d2W2: tensor<128x128x3x3xf32>, %d2g2: tensor<128xf32>, %d2bt2: tensor<128xf32>, %d2Wp: tensor<128x64x1x1xf32>, %d2gp: tensor<128xf32>, %d2btp: tensor<128xf32>, %s2b0W1: tensor<128x128x3x3xf32>, %s2b0g1: tensor<128xf32>, %s2b0bt1: tensor<128xf32>, %s2b0W2: tensor<128x128x3x3xf32>, %s2b0g2: tensor<128xf32>, %s2b0bt2: tensor<128xf32>, %s2b1W1: tensor<128x128x3x3xf32>, %s2b1g1: tensor<128xf32>, %s2b1bt1: tensor<128xf32>, %s2b1W2: tensor<128x128x3x3xf32>, %s2b1g2: tensor<128xf32>, %s2b1bt2: tensor<128xf32>, %s2b2W1: tensor<128x128x3x3xf32>, %s2b2g1: tensor<128xf32>, %s2b2bt1: tensor<128xf32>, %s2b2W2: tensor<128x128x3x3xf32>, %s2b2g2: tensor<128xf32>, %s2b2bt2: tensor<128xf32>, %d3W1: tensor<256x128x3x3xf32>, %d3g1: tensor<256xf32>, %d3bt1: tensor<256xf32>, %d3W2: tensor<256x256x3x3xf32>, %d3g2: tensor<256xf32>, %d3bt2: tensor<256xf32>, %d3Wp: tensor<256x128x1x1xf32>, %d3gp: tensor<256xf32>, %d3btp: tensor<256xf32>, %s3b0W1: tensor<256x256x3x3xf32>, %s3b0g1: tensor<256xf32>, %s3b0bt1: tensor<256xf32>, %s3b0W2: tensor<256x256x3x3xf32>, %s3b0g2: tensor<256xf32>, %s3b0bt2: tensor<256xf32>, %s3b1W1: tensor<256x256x3x3xf32>, %s3b1g1: tensor<256xf32>, %s3b1bt1: tensor<256xf32>, %s3b1W2: tensor<256x256x3x3xf32>, %s3b1g2: tensor<256xf32>, %s3b1bt2: tensor<256xf32>, %s3b2W1: tensor<256x256x3x3xf32>, %s3b2g1: tensor<256xf32>, %s3b2bt1: tensor<256xf32>, %s3b2W2: tensor<256x256x3x3xf32>, %s3b2g2: tensor<256xf32>, %s3b2bt2: tensor<256xf32>, %s3b3W1: tensor<256x256x3x3xf32>, %s3b3g1: tensor<256xf32>, %s3b3bt1: tensor<256xf32>, %s3b3W2: tensor<256x256x3x3xf32>, %s3b3g2: tensor<256xf32>, %s3b3bt2: tensor<256xf32>, %s3b4W1: tensor<256x256x3x3xf32>, %s3b4g1: tensor<256xf32>, %s3b4bt1: tensor<256xf32>, %s3b4W2: tensor<256x256x3x3xf32>, %s3b4g2: tensor<256xf32>, %s3b4bt2: tensor<256xf32>, %d4W1: tensor<512x256x3x3xf32>, %d4g1: tensor<512xf32>, %d4bt1: tensor<512xf32>, %d4W2: tensor<512x512x3x3xf32>, %d4g2: tensor<512xf32>, %d4bt2: tensor<512xf32>, %d4Wp: tensor<512x256x1x1xf32>, %d4gp: tensor<512xf32>, %d4btp: tensor<512xf32>, %s4b0W1: tensor<512x512x3x3xf32>, %s4b0g1: tensor<512xf32>, %s4b0bt1: tensor<512xf32>, %s4b0W2: tensor<512x512x3x3xf32>, %s4b0g2: tensor<512xf32>, %s4b0bt2: tensor<512xf32>, %s4b1W1: tensor<512x512x3x3xf32>, %s4b1g1: tensor<512xf32>, %s4b1bt1: tensor<512xf32>, %s4b1W2: tensor<512x512x3x3xf32>, %s4b1g2: tensor<512xf32>, %s4b1bt2: tensor<512xf32>, %Wd: tensor<512x1000xf32>, %bd: tensor<1000xf32>, %stnmu: tensor<64xf32>, %stnvar: tensor<64xf32>, %s1b0n1mu: tensor<64xf32>, %s1b0n1var: tensor<64xf32>, %s1b0n2mu: tensor<64xf32>, %s1b0n2var: tensor<64xf32>, %s1b1n1mu: tensor<64xf32>, %s1b1n1var: tensor<64xf32>, %s1b1n2mu: tensor<64xf32>, %s1b1n2var: tensor<64xf32>, %s1b2n1mu: tensor<64xf32>, %s1b2n1var: tensor<64xf32>, %s1b2n2mu: tensor<64xf32>, %s1b2n2var: tensor<64xf32>, %d2n1mu: tensor<128xf32>, %d2n1var: tensor<128xf32>, %d2n2mu: tensor<128xf32>, %d2n2var: tensor<128xf32>, %d2npmu: tensor<128xf32>, %d2npvar: tensor<128xf32>, %s2b0n1mu: tensor<128xf32>, %s2b0n1var: tensor<128xf32>, %s2b0n2mu: tensor<128xf32>, %s2b0n2var: tensor<128xf32>, %s2b1n1mu: tensor<128xf32>, %s2b1n1var: tensor<128xf32>, %s2b1n2mu: tensor<128xf32>, %s2b1n2var: tensor<128xf32>, %s2b2n1mu: tensor<128xf32>, %s2b2n1var: tensor<128xf32>, %s2b2n2mu: tensor<128xf32>, %s2b2n2var: tensor<128xf32>, %d3n1mu: tensor<256xf32>, %d3n1var: tensor<256xf32>, %d3n2mu: tensor<256xf32>, %d3n2var: tensor<256xf32>, %d3npmu: tensor<256xf32>, %d3npvar: tensor<256xf32>, %s3b0n1mu: tensor<256xf32>, %s3b0n1var: tensor<256xf32>, %s3b0n2mu: tensor<256xf32>, %s3b0n2var: tensor<256xf32>, %s3b1n1mu: tensor<256xf32>, %s3b1n1var: tensor<256xf32>, %s3b1n2mu: tensor<256xf32>, %s3b1n2var: tensor<256xf32>, %s3b2n1mu: tensor<256xf32>, %s3b2n1var: tensor<256xf32>, %s3b2n2mu: tensor<256xf32>, %s3b2n2var: tensor<256xf32>, %s3b3n1mu: tensor<256xf32>, %s3b3n1var: tensor<256xf32>, %s3b3n2mu: tensor<256xf32>, %s3b3n2var: tensor<256xf32>, %s3b4n1mu: tensor<256xf32>, %s3b4n1var: tensor<256xf32>, %s3b4n2mu: tensor<256xf32>, %s3b4n2var: tensor<256xf32>, %d4n1mu: tensor<512xf32>, %d4n1var: tensor<512xf32>, %d4n2mu: tensor<512xf32>, %d4n2var: tensor<512xf32>, %d4npmu: tensor<512xf32>, %d4npvar: tensor<512xf32>, %s4b0n1mu: tensor<512xf32>, %s4b0n1var: tensor<512xf32>, %s4b0n2mu: tensor<512xf32>, %s4b0n2var: tensor<512xf32>, %s4b1n1mu: tensor<512xf32>, %s4b1n1var: tensor<512xf32>, %s4b1n2mu: tensor<512xf32>, %s4b1n2var: tensor<512xf32>) -> tensor<256x1000xf32> {
    // ── ResNet-34 eval forward (running-stats BN): every line is pretty(verified AST node) ──
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
    %v6 = stablehlo.broadcast_in_dim %stnmu, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v7 = stablehlo.subtract %v5, %v6 : tensor<256x64x112x112xf32>
    %v8 = stablehlo.broadcast_in_dim %stnvar, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v9 = stablehlo.constant dense<1.0e-05> : tensor<256x64x112x112xf32>
    %v10 = stablehlo.add %v8, %v9 : tensor<256x64x112x112xf32>
    %v11 = stablehlo.rsqrt %v10 : tensor<256x64x112x112xf32>
    %v12 = stablehlo.multiply %v7, %v11 : tensor<256x64x112x112xf32>
    %v13 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v14 = stablehlo.broadcast_in_dim %sbt, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v15 = stablehlo.multiply %v12, %v13 : tensor<256x64x112x112xf32>
    %v16 = stablehlo.add %v15, %v14 : tensor<256x64x112x112xf32>
    %v17 = stablehlo.reshape %v16 : (tensor<256x64x112x112xf32>) -> tensor<256x802816xf32>
    %v18 = stablehlo.reshape %v17 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v19 = stablehlo.constant dense<0.0> : tensor<256x64x112x112xf32>
    %v20 = stablehlo.maximum %v18, %v19 : tensor<256x64x112x112xf32>
    %v21 = stablehlo.reshape %v20 : (tensor<256x64x112x112xf32>) -> tensor<256x802816xf32>
    %v22 = stablehlo.reshape %v21 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v23 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v24 = "stablehlo.reduce_window"(%v22, %v23) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>} : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<256x64x56x56xf32>
    %v25 = stablehlo.reshape %v24 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v26 = stablehlo.reshape %v25 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v27 = stablehlo.convolution(%v26, %s1b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v28 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v29 = stablehlo.add %v27, %v28 : tensor<256x64x56x56xf32>
    %v30 = stablehlo.reshape %v29 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v32 = stablehlo.broadcast_in_dim %s1b0n1mu, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v33 = stablehlo.subtract %v31, %v32 : tensor<256x64x56x56xf32>
    %v34 = stablehlo.broadcast_in_dim %s1b0n1var, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v35 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v36 = stablehlo.add %v34, %v35 : tensor<256x64x56x56xf32>
    %v37 = stablehlo.rsqrt %v36 : tensor<256x64x56x56xf32>
    %v38 = stablehlo.multiply %v33, %v37 : tensor<256x64x56x56xf32>
    %v39 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v40 = stablehlo.broadcast_in_dim %s1b0bt1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v41 = stablehlo.multiply %v38, %v39 : tensor<256x64x56x56xf32>
    %v42 = stablehlo.add %v41, %v40 : tensor<256x64x56x56xf32>
    %v43 = stablehlo.reshape %v42 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v44 = stablehlo.reshape %v43 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v45 = stablehlo.constant dense<0.0> : tensor<256x64x56x56xf32>
    %v46 = stablehlo.maximum %v44, %v45 : tensor<256x64x56x56xf32>
    %v47 = stablehlo.reshape %v46 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v48 = stablehlo.reshape %v47 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v49 = stablehlo.convolution(%v48, %s1b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v50 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v51 = stablehlo.add %v49, %v50 : tensor<256x64x56x56xf32>
    %v52 = stablehlo.reshape %v51 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v53 = stablehlo.reshape %v52 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v54 = stablehlo.broadcast_in_dim %s1b0n2mu, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v55 = stablehlo.subtract %v53, %v54 : tensor<256x64x56x56xf32>
    %v56 = stablehlo.broadcast_in_dim %s1b0n2var, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v57 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v58 = stablehlo.add %v56, %v57 : tensor<256x64x56x56xf32>
    %v59 = stablehlo.rsqrt %v58 : tensor<256x64x56x56xf32>
    %v60 = stablehlo.multiply %v55, %v59 : tensor<256x64x56x56xf32>
    %v61 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v62 = stablehlo.broadcast_in_dim %s1b0bt2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v63 = stablehlo.multiply %v60, %v61 : tensor<256x64x56x56xf32>
    %v64 = stablehlo.add %v63, %v62 : tensor<256x64x56x56xf32>
    %v65 = stablehlo.reshape %v64 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v66 = stablehlo.reshape %v65 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v67 = stablehlo.reshape %v25 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v68 = stablehlo.add %v66, %v67 : tensor<256x64x56x56xf32>
    %v69 = stablehlo.reshape %v68 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v70 = stablehlo.reshape %v69 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v71 = stablehlo.constant dense<0.0> : tensor<256x64x56x56xf32>
    %v72 = stablehlo.maximum %v70, %v71 : tensor<256x64x56x56xf32>
    %v73 = stablehlo.reshape %v72 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v74 = stablehlo.reshape %v73 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v75 = stablehlo.convolution(%v74, %s1b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v76 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v77 = stablehlo.add %v75, %v76 : tensor<256x64x56x56xf32>
    %v78 = stablehlo.reshape %v77 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v79 = stablehlo.reshape %v78 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v80 = stablehlo.broadcast_in_dim %s1b1n1mu, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v81 = stablehlo.subtract %v79, %v80 : tensor<256x64x56x56xf32>
    %v82 = stablehlo.broadcast_in_dim %s1b1n1var, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v83 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v84 = stablehlo.add %v82, %v83 : tensor<256x64x56x56xf32>
    %v85 = stablehlo.rsqrt %v84 : tensor<256x64x56x56xf32>
    %v86 = stablehlo.multiply %v81, %v85 : tensor<256x64x56x56xf32>
    %v87 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v88 = stablehlo.broadcast_in_dim %s1b1bt1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v89 = stablehlo.multiply %v86, %v87 : tensor<256x64x56x56xf32>
    %v90 = stablehlo.add %v89, %v88 : tensor<256x64x56x56xf32>
    %v91 = stablehlo.reshape %v90 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v92 = stablehlo.reshape %v91 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v93 = stablehlo.constant dense<0.0> : tensor<256x64x56x56xf32>
    %v94 = stablehlo.maximum %v92, %v93 : tensor<256x64x56x56xf32>
    %v95 = stablehlo.reshape %v94 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v96 = stablehlo.reshape %v95 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v97 = stablehlo.convolution(%v96, %s1b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v98 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v99 = stablehlo.add %v97, %v98 : tensor<256x64x56x56xf32>
    %v100 = stablehlo.reshape %v99 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v101 = stablehlo.reshape %v100 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v102 = stablehlo.broadcast_in_dim %s1b1n2mu, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v103 = stablehlo.subtract %v101, %v102 : tensor<256x64x56x56xf32>
    %v104 = stablehlo.broadcast_in_dim %s1b1n2var, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v105 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v106 = stablehlo.add %v104, %v105 : tensor<256x64x56x56xf32>
    %v107 = stablehlo.rsqrt %v106 : tensor<256x64x56x56xf32>
    %v108 = stablehlo.multiply %v103, %v107 : tensor<256x64x56x56xf32>
    %v109 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v110 = stablehlo.broadcast_in_dim %s1b1bt2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v111 = stablehlo.multiply %v108, %v109 : tensor<256x64x56x56xf32>
    %v112 = stablehlo.add %v111, %v110 : tensor<256x64x56x56xf32>
    %v113 = stablehlo.reshape %v112 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v114 = stablehlo.reshape %v113 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v115 = stablehlo.reshape %v73 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v116 = stablehlo.add %v114, %v115 : tensor<256x64x56x56xf32>
    %v117 = stablehlo.reshape %v116 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v118 = stablehlo.reshape %v117 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v119 = stablehlo.constant dense<0.0> : tensor<256x64x56x56xf32>
    %v120 = stablehlo.maximum %v118, %v119 : tensor<256x64x56x56xf32>
    %v121 = stablehlo.reshape %v120 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v122 = stablehlo.reshape %v121 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v123 = stablehlo.convolution(%v122, %s1b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v124 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v125 = stablehlo.add %v123, %v124 : tensor<256x64x56x56xf32>
    %v126 = stablehlo.reshape %v125 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v127 = stablehlo.reshape %v126 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v128 = stablehlo.broadcast_in_dim %s1b2n1mu, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v129 = stablehlo.subtract %v127, %v128 : tensor<256x64x56x56xf32>
    %v130 = stablehlo.broadcast_in_dim %s1b2n1var, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v131 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v132 = stablehlo.add %v130, %v131 : tensor<256x64x56x56xf32>
    %v133 = stablehlo.rsqrt %v132 : tensor<256x64x56x56xf32>
    %v134 = stablehlo.multiply %v129, %v133 : tensor<256x64x56x56xf32>
    %v135 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v136 = stablehlo.broadcast_in_dim %s1b2bt1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v137 = stablehlo.multiply %v134, %v135 : tensor<256x64x56x56xf32>
    %v138 = stablehlo.add %v137, %v136 : tensor<256x64x56x56xf32>
    %v139 = stablehlo.reshape %v138 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v140 = stablehlo.reshape %v139 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v141 = stablehlo.constant dense<0.0> : tensor<256x64x56x56xf32>
    %v142 = stablehlo.maximum %v140, %v141 : tensor<256x64x56x56xf32>
    %v143 = stablehlo.reshape %v142 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v144 = stablehlo.reshape %v143 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v145 = stablehlo.convolution(%v144, %s1b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v146 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v147 = stablehlo.add %v145, %v146 : tensor<256x64x56x56xf32>
    %v148 = stablehlo.reshape %v147 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v149 = stablehlo.reshape %v148 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v150 = stablehlo.broadcast_in_dim %s1b2n2mu, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v151 = stablehlo.subtract %v149, %v150 : tensor<256x64x56x56xf32>
    %v152 = stablehlo.broadcast_in_dim %s1b2n2var, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v153 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v154 = stablehlo.add %v152, %v153 : tensor<256x64x56x56xf32>
    %v155 = stablehlo.rsqrt %v154 : tensor<256x64x56x56xf32>
    %v156 = stablehlo.multiply %v151, %v155 : tensor<256x64x56x56xf32>
    %v157 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v158 = stablehlo.broadcast_in_dim %s1b2bt2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v159 = stablehlo.multiply %v156, %v157 : tensor<256x64x56x56xf32>
    %v160 = stablehlo.add %v159, %v158 : tensor<256x64x56x56xf32>
    %v161 = stablehlo.reshape %v160 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v162 = stablehlo.reshape %v161 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v163 = stablehlo.reshape %v121 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v164 = stablehlo.add %v162, %v163 : tensor<256x64x56x56xf32>
    %v165 = stablehlo.reshape %v164 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v166 = stablehlo.reshape %v165 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v167 = stablehlo.constant dense<0.0> : tensor<256x64x56x56xf32>
    %v168 = stablehlo.maximum %v166, %v167 : tensor<256x64x56x56xf32>
    %v169 = stablehlo.reshape %v168 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v170 = stablehlo.reshape %v169 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v171 = stablehlo.convolution(%v170, %d2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<128x64x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v172 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v173 = stablehlo.add %v171, %v172 : tensor<256x128x28x28xf32>
    %v174 = stablehlo.reshape %v173 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v175 = stablehlo.reshape %v174 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v176 = stablehlo.broadcast_in_dim %d2n1mu, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v177 = stablehlo.subtract %v175, %v176 : tensor<256x128x28x28xf32>
    %v178 = stablehlo.broadcast_in_dim %d2n1var, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v179 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v180 = stablehlo.add %v178, %v179 : tensor<256x128x28x28xf32>
    %v181 = stablehlo.rsqrt %v180 : tensor<256x128x28x28xf32>
    %v182 = stablehlo.multiply %v177, %v181 : tensor<256x128x28x28xf32>
    %v183 = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v184 = stablehlo.broadcast_in_dim %d2bt1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v185 = stablehlo.multiply %v182, %v183 : tensor<256x128x28x28xf32>
    %v186 = stablehlo.add %v185, %v184 : tensor<256x128x28x28xf32>
    %v187 = stablehlo.reshape %v186 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v188 = stablehlo.reshape %v187 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v189 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v190 = stablehlo.maximum %v188, %v189 : tensor<256x128x28x28xf32>
    %v191 = stablehlo.reshape %v190 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v192 = stablehlo.reshape %v191 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v193 = stablehlo.convolution(%v192, %d2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v194 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v195 = stablehlo.add %v193, %v194 : tensor<256x128x28x28xf32>
    %v196 = stablehlo.reshape %v195 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v197 = stablehlo.reshape %v196 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v198 = stablehlo.broadcast_in_dim %d2n2mu, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v199 = stablehlo.subtract %v197, %v198 : tensor<256x128x28x28xf32>
    %v200 = stablehlo.broadcast_in_dim %d2n2var, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v201 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v202 = stablehlo.add %v200, %v201 : tensor<256x128x28x28xf32>
    %v203 = stablehlo.rsqrt %v202 : tensor<256x128x28x28xf32>
    %v204 = stablehlo.multiply %v199, %v203 : tensor<256x128x28x28xf32>
    %v205 = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v206 = stablehlo.broadcast_in_dim %d2bt2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v207 = stablehlo.multiply %v204, %v205 : tensor<256x128x28x28xf32>
    %v208 = stablehlo.add %v207, %v206 : tensor<256x128x28x28xf32>
    %v209 = stablehlo.reshape %v208 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v210 = stablehlo.reshape %v169 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v211 = stablehlo.convolution(%v210, %d2Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<128x64x1x1xf32>) -> tensor<256x128x28x28xf32>
    %v212 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v213 = stablehlo.add %v211, %v212 : tensor<256x128x28x28xf32>
    %v214 = stablehlo.reshape %v213 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v215 = stablehlo.reshape %v214 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v216 = stablehlo.broadcast_in_dim %d2npmu, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v217 = stablehlo.subtract %v215, %v216 : tensor<256x128x28x28xf32>
    %v218 = stablehlo.broadcast_in_dim %d2npvar, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v219 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v220 = stablehlo.add %v218, %v219 : tensor<256x128x28x28xf32>
    %v221 = stablehlo.rsqrt %v220 : tensor<256x128x28x28xf32>
    %v222 = stablehlo.multiply %v217, %v221 : tensor<256x128x28x28xf32>
    %v223 = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v224 = stablehlo.broadcast_in_dim %d2btp, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v225 = stablehlo.multiply %v222, %v223 : tensor<256x128x28x28xf32>
    %v226 = stablehlo.add %v225, %v224 : tensor<256x128x28x28xf32>
    %v227 = stablehlo.reshape %v226 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v228 = stablehlo.reshape %v209 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v229 = stablehlo.reshape %v227 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v230 = stablehlo.add %v228, %v229 : tensor<256x128x28x28xf32>
    %v231 = stablehlo.reshape %v230 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v232 = stablehlo.reshape %v231 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v233 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v234 = stablehlo.maximum %v232, %v233 : tensor<256x128x28x28xf32>
    %v235 = stablehlo.reshape %v234 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v236 = stablehlo.reshape %v235 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v237 = stablehlo.convolution(%v236, %s2b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v238 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v239 = stablehlo.add %v237, %v238 : tensor<256x128x28x28xf32>
    %v240 = stablehlo.reshape %v239 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v241 = stablehlo.reshape %v240 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v242 = stablehlo.broadcast_in_dim %s2b0n1mu, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v243 = stablehlo.subtract %v241, %v242 : tensor<256x128x28x28xf32>
    %v244 = stablehlo.broadcast_in_dim %s2b0n1var, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v245 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v246 = stablehlo.add %v244, %v245 : tensor<256x128x28x28xf32>
    %v247 = stablehlo.rsqrt %v246 : tensor<256x128x28x28xf32>
    %v248 = stablehlo.multiply %v243, %v247 : tensor<256x128x28x28xf32>
    %v249 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v250 = stablehlo.broadcast_in_dim %s2b0bt1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v251 = stablehlo.multiply %v248, %v249 : tensor<256x128x28x28xf32>
    %v252 = stablehlo.add %v251, %v250 : tensor<256x128x28x28xf32>
    %v253 = stablehlo.reshape %v252 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v254 = stablehlo.reshape %v253 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v255 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v256 = stablehlo.maximum %v254, %v255 : tensor<256x128x28x28xf32>
    %v257 = stablehlo.reshape %v256 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v258 = stablehlo.reshape %v257 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v259 = stablehlo.convolution(%v258, %s2b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v260 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v261 = stablehlo.add %v259, %v260 : tensor<256x128x28x28xf32>
    %v262 = stablehlo.reshape %v261 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v263 = stablehlo.reshape %v262 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v264 = stablehlo.broadcast_in_dim %s2b0n2mu, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v265 = stablehlo.subtract %v263, %v264 : tensor<256x128x28x28xf32>
    %v266 = stablehlo.broadcast_in_dim %s2b0n2var, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v267 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v268 = stablehlo.add %v266, %v267 : tensor<256x128x28x28xf32>
    %v269 = stablehlo.rsqrt %v268 : tensor<256x128x28x28xf32>
    %v270 = stablehlo.multiply %v265, %v269 : tensor<256x128x28x28xf32>
    %v271 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v272 = stablehlo.broadcast_in_dim %s2b0bt2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v273 = stablehlo.multiply %v270, %v271 : tensor<256x128x28x28xf32>
    %v274 = stablehlo.add %v273, %v272 : tensor<256x128x28x28xf32>
    %v275 = stablehlo.reshape %v274 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v276 = stablehlo.reshape %v275 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v277 = stablehlo.reshape %v235 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v278 = stablehlo.add %v276, %v277 : tensor<256x128x28x28xf32>
    %v279 = stablehlo.reshape %v278 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v280 = stablehlo.reshape %v279 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v281 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v282 = stablehlo.maximum %v280, %v281 : tensor<256x128x28x28xf32>
    %v283 = stablehlo.reshape %v282 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v284 = stablehlo.reshape %v283 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v285 = stablehlo.convolution(%v284, %s2b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v286 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v287 = stablehlo.add %v285, %v286 : tensor<256x128x28x28xf32>
    %v288 = stablehlo.reshape %v287 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v289 = stablehlo.reshape %v288 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v290 = stablehlo.broadcast_in_dim %s2b1n1mu, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v291 = stablehlo.subtract %v289, %v290 : tensor<256x128x28x28xf32>
    %v292 = stablehlo.broadcast_in_dim %s2b1n1var, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v293 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v294 = stablehlo.add %v292, %v293 : tensor<256x128x28x28xf32>
    %v295 = stablehlo.rsqrt %v294 : tensor<256x128x28x28xf32>
    %v296 = stablehlo.multiply %v291, %v295 : tensor<256x128x28x28xf32>
    %v297 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v298 = stablehlo.broadcast_in_dim %s2b1bt1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v299 = stablehlo.multiply %v296, %v297 : tensor<256x128x28x28xf32>
    %v300 = stablehlo.add %v299, %v298 : tensor<256x128x28x28xf32>
    %v301 = stablehlo.reshape %v300 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v302 = stablehlo.reshape %v301 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v303 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v304 = stablehlo.maximum %v302, %v303 : tensor<256x128x28x28xf32>
    %v305 = stablehlo.reshape %v304 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v306 = stablehlo.reshape %v305 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v307 = stablehlo.convolution(%v306, %s2b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v308 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v309 = stablehlo.add %v307, %v308 : tensor<256x128x28x28xf32>
    %v310 = stablehlo.reshape %v309 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v311 = stablehlo.reshape %v310 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v312 = stablehlo.broadcast_in_dim %s2b1n2mu, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v313 = stablehlo.subtract %v311, %v312 : tensor<256x128x28x28xf32>
    %v314 = stablehlo.broadcast_in_dim %s2b1n2var, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v315 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v316 = stablehlo.add %v314, %v315 : tensor<256x128x28x28xf32>
    %v317 = stablehlo.rsqrt %v316 : tensor<256x128x28x28xf32>
    %v318 = stablehlo.multiply %v313, %v317 : tensor<256x128x28x28xf32>
    %v319 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v320 = stablehlo.broadcast_in_dim %s2b1bt2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v321 = stablehlo.multiply %v318, %v319 : tensor<256x128x28x28xf32>
    %v322 = stablehlo.add %v321, %v320 : tensor<256x128x28x28xf32>
    %v323 = stablehlo.reshape %v322 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v324 = stablehlo.reshape %v323 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v325 = stablehlo.reshape %v283 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v326 = stablehlo.add %v324, %v325 : tensor<256x128x28x28xf32>
    %v327 = stablehlo.reshape %v326 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v328 = stablehlo.reshape %v327 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v329 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v330 = stablehlo.maximum %v328, %v329 : tensor<256x128x28x28xf32>
    %v331 = stablehlo.reshape %v330 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v332 = stablehlo.reshape %v331 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v333 = stablehlo.convolution(%v332, %s2b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v334 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v335 = stablehlo.add %v333, %v334 : tensor<256x128x28x28xf32>
    %v336 = stablehlo.reshape %v335 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v337 = stablehlo.reshape %v336 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v338 = stablehlo.broadcast_in_dim %s2b2n1mu, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v339 = stablehlo.subtract %v337, %v338 : tensor<256x128x28x28xf32>
    %v340 = stablehlo.broadcast_in_dim %s2b2n1var, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v341 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v342 = stablehlo.add %v340, %v341 : tensor<256x128x28x28xf32>
    %v343 = stablehlo.rsqrt %v342 : tensor<256x128x28x28xf32>
    %v344 = stablehlo.multiply %v339, %v343 : tensor<256x128x28x28xf32>
    %v345 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v346 = stablehlo.broadcast_in_dim %s2b2bt1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v347 = stablehlo.multiply %v344, %v345 : tensor<256x128x28x28xf32>
    %v348 = stablehlo.add %v347, %v346 : tensor<256x128x28x28xf32>
    %v349 = stablehlo.reshape %v348 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v350 = stablehlo.reshape %v349 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v351 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v352 = stablehlo.maximum %v350, %v351 : tensor<256x128x28x28xf32>
    %v353 = stablehlo.reshape %v352 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v354 = stablehlo.reshape %v353 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v355 = stablehlo.convolution(%v354, %s2b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v356 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v357 = stablehlo.add %v355, %v356 : tensor<256x128x28x28xf32>
    %v358 = stablehlo.reshape %v357 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v359 = stablehlo.reshape %v358 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v360 = stablehlo.broadcast_in_dim %s2b2n2mu, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v361 = stablehlo.subtract %v359, %v360 : tensor<256x128x28x28xf32>
    %v362 = stablehlo.broadcast_in_dim %s2b2n2var, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v363 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v364 = stablehlo.add %v362, %v363 : tensor<256x128x28x28xf32>
    %v365 = stablehlo.rsqrt %v364 : tensor<256x128x28x28xf32>
    %v366 = stablehlo.multiply %v361, %v365 : tensor<256x128x28x28xf32>
    %v367 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v368 = stablehlo.broadcast_in_dim %s2b2bt2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v369 = stablehlo.multiply %v366, %v367 : tensor<256x128x28x28xf32>
    %v370 = stablehlo.add %v369, %v368 : tensor<256x128x28x28xf32>
    %v371 = stablehlo.reshape %v370 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v372 = stablehlo.reshape %v371 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v373 = stablehlo.reshape %v331 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v374 = stablehlo.add %v372, %v373 : tensor<256x128x28x28xf32>
    %v375 = stablehlo.reshape %v374 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v376 = stablehlo.reshape %v375 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v377 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v378 = stablehlo.maximum %v376, %v377 : tensor<256x128x28x28xf32>
    %v379 = stablehlo.reshape %v378 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v380 = stablehlo.reshape %v379 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v381 = stablehlo.convolution(%v380, %d3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<256x128x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v382 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v383 = stablehlo.add %v381, %v382 : tensor<256x256x14x14xf32>
    %v384 = stablehlo.reshape %v383 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v385 = stablehlo.reshape %v384 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v386 = stablehlo.broadcast_in_dim %d3n1mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v387 = stablehlo.subtract %v385, %v386 : tensor<256x256x14x14xf32>
    %v388 = stablehlo.broadcast_in_dim %d3n1var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v389 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v390 = stablehlo.add %v388, %v389 : tensor<256x256x14x14xf32>
    %v391 = stablehlo.rsqrt %v390 : tensor<256x256x14x14xf32>
    %v392 = stablehlo.multiply %v387, %v391 : tensor<256x256x14x14xf32>
    %v393 = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v394 = stablehlo.broadcast_in_dim %d3bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v395 = stablehlo.multiply %v392, %v393 : tensor<256x256x14x14xf32>
    %v396 = stablehlo.add %v395, %v394 : tensor<256x256x14x14xf32>
    %v397 = stablehlo.reshape %v396 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v398 = stablehlo.reshape %v397 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v399 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v400 = stablehlo.maximum %v398, %v399 : tensor<256x256x14x14xf32>
    %v401 = stablehlo.reshape %v400 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v402 = stablehlo.reshape %v401 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v403 = stablehlo.convolution(%v402, %d3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v404 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v405 = stablehlo.add %v403, %v404 : tensor<256x256x14x14xf32>
    %v406 = stablehlo.reshape %v405 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v407 = stablehlo.reshape %v406 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v408 = stablehlo.broadcast_in_dim %d3n2mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v409 = stablehlo.subtract %v407, %v408 : tensor<256x256x14x14xf32>
    %v410 = stablehlo.broadcast_in_dim %d3n2var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v411 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v412 = stablehlo.add %v410, %v411 : tensor<256x256x14x14xf32>
    %v413 = stablehlo.rsqrt %v412 : tensor<256x256x14x14xf32>
    %v414 = stablehlo.multiply %v409, %v413 : tensor<256x256x14x14xf32>
    %v415 = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v416 = stablehlo.broadcast_in_dim %d3bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v417 = stablehlo.multiply %v414, %v415 : tensor<256x256x14x14xf32>
    %v418 = stablehlo.add %v417, %v416 : tensor<256x256x14x14xf32>
    %v419 = stablehlo.reshape %v418 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v420 = stablehlo.reshape %v379 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v421 = stablehlo.convolution(%v420, %d3Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<256x128x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v422 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v423 = stablehlo.add %v421, %v422 : tensor<256x256x14x14xf32>
    %v424 = stablehlo.reshape %v423 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v425 = stablehlo.reshape %v424 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v426 = stablehlo.broadcast_in_dim %d3npmu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v427 = stablehlo.subtract %v425, %v426 : tensor<256x256x14x14xf32>
    %v428 = stablehlo.broadcast_in_dim %d3npvar, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v429 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v430 = stablehlo.add %v428, %v429 : tensor<256x256x14x14xf32>
    %v431 = stablehlo.rsqrt %v430 : tensor<256x256x14x14xf32>
    %v432 = stablehlo.multiply %v427, %v431 : tensor<256x256x14x14xf32>
    %v433 = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v434 = stablehlo.broadcast_in_dim %d3btp, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v435 = stablehlo.multiply %v432, %v433 : tensor<256x256x14x14xf32>
    %v436 = stablehlo.add %v435, %v434 : tensor<256x256x14x14xf32>
    %v437 = stablehlo.reshape %v436 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v438 = stablehlo.reshape %v419 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v439 = stablehlo.reshape %v437 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v440 = stablehlo.add %v438, %v439 : tensor<256x256x14x14xf32>
    %v441 = stablehlo.reshape %v440 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v442 = stablehlo.reshape %v441 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v443 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v444 = stablehlo.maximum %v442, %v443 : tensor<256x256x14x14xf32>
    %v445 = stablehlo.reshape %v444 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v446 = stablehlo.reshape %v445 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v447 = stablehlo.convolution(%v446, %s3b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v448 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v449 = stablehlo.add %v447, %v448 : tensor<256x256x14x14xf32>
    %v450 = stablehlo.reshape %v449 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v451 = stablehlo.reshape %v450 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v452 = stablehlo.broadcast_in_dim %s3b0n1mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v453 = stablehlo.subtract %v451, %v452 : tensor<256x256x14x14xf32>
    %v454 = stablehlo.broadcast_in_dim %s3b0n1var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v455 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v456 = stablehlo.add %v454, %v455 : tensor<256x256x14x14xf32>
    %v457 = stablehlo.rsqrt %v456 : tensor<256x256x14x14xf32>
    %v458 = stablehlo.multiply %v453, %v457 : tensor<256x256x14x14xf32>
    %v459 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v460 = stablehlo.broadcast_in_dim %s3b0bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v461 = stablehlo.multiply %v458, %v459 : tensor<256x256x14x14xf32>
    %v462 = stablehlo.add %v461, %v460 : tensor<256x256x14x14xf32>
    %v463 = stablehlo.reshape %v462 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v464 = stablehlo.reshape %v463 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v465 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v466 = stablehlo.maximum %v464, %v465 : tensor<256x256x14x14xf32>
    %v467 = stablehlo.reshape %v466 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v468 = stablehlo.reshape %v467 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v469 = stablehlo.convolution(%v468, %s3b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v470 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v471 = stablehlo.add %v469, %v470 : tensor<256x256x14x14xf32>
    %v472 = stablehlo.reshape %v471 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v473 = stablehlo.reshape %v472 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v474 = stablehlo.broadcast_in_dim %s3b0n2mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v475 = stablehlo.subtract %v473, %v474 : tensor<256x256x14x14xf32>
    %v476 = stablehlo.broadcast_in_dim %s3b0n2var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v477 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v478 = stablehlo.add %v476, %v477 : tensor<256x256x14x14xf32>
    %v479 = stablehlo.rsqrt %v478 : tensor<256x256x14x14xf32>
    %v480 = stablehlo.multiply %v475, %v479 : tensor<256x256x14x14xf32>
    %v481 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v482 = stablehlo.broadcast_in_dim %s3b0bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v483 = stablehlo.multiply %v480, %v481 : tensor<256x256x14x14xf32>
    %v484 = stablehlo.add %v483, %v482 : tensor<256x256x14x14xf32>
    %v485 = stablehlo.reshape %v484 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v486 = stablehlo.reshape %v485 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v487 = stablehlo.reshape %v445 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v488 = stablehlo.add %v486, %v487 : tensor<256x256x14x14xf32>
    %v489 = stablehlo.reshape %v488 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v490 = stablehlo.reshape %v489 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v491 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v492 = stablehlo.maximum %v490, %v491 : tensor<256x256x14x14xf32>
    %v493 = stablehlo.reshape %v492 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v494 = stablehlo.reshape %v493 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v495 = stablehlo.convolution(%v494, %s3b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v496 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v497 = stablehlo.add %v495, %v496 : tensor<256x256x14x14xf32>
    %v498 = stablehlo.reshape %v497 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v499 = stablehlo.reshape %v498 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v500 = stablehlo.broadcast_in_dim %s3b1n1mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v501 = stablehlo.subtract %v499, %v500 : tensor<256x256x14x14xf32>
    %v502 = stablehlo.broadcast_in_dim %s3b1n1var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v503 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v504 = stablehlo.add %v502, %v503 : tensor<256x256x14x14xf32>
    %v505 = stablehlo.rsqrt %v504 : tensor<256x256x14x14xf32>
    %v506 = stablehlo.multiply %v501, %v505 : tensor<256x256x14x14xf32>
    %v507 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v508 = stablehlo.broadcast_in_dim %s3b1bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v509 = stablehlo.multiply %v506, %v507 : tensor<256x256x14x14xf32>
    %v510 = stablehlo.add %v509, %v508 : tensor<256x256x14x14xf32>
    %v511 = stablehlo.reshape %v510 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v512 = stablehlo.reshape %v511 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v513 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v514 = stablehlo.maximum %v512, %v513 : tensor<256x256x14x14xf32>
    %v515 = stablehlo.reshape %v514 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v516 = stablehlo.reshape %v515 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v517 = stablehlo.convolution(%v516, %s3b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v518 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v519 = stablehlo.add %v517, %v518 : tensor<256x256x14x14xf32>
    %v520 = stablehlo.reshape %v519 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v521 = stablehlo.reshape %v520 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v522 = stablehlo.broadcast_in_dim %s3b1n2mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v523 = stablehlo.subtract %v521, %v522 : tensor<256x256x14x14xf32>
    %v524 = stablehlo.broadcast_in_dim %s3b1n2var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v525 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v526 = stablehlo.add %v524, %v525 : tensor<256x256x14x14xf32>
    %v527 = stablehlo.rsqrt %v526 : tensor<256x256x14x14xf32>
    %v528 = stablehlo.multiply %v523, %v527 : tensor<256x256x14x14xf32>
    %v529 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v530 = stablehlo.broadcast_in_dim %s3b1bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v531 = stablehlo.multiply %v528, %v529 : tensor<256x256x14x14xf32>
    %v532 = stablehlo.add %v531, %v530 : tensor<256x256x14x14xf32>
    %v533 = stablehlo.reshape %v532 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v534 = stablehlo.reshape %v533 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v535 = stablehlo.reshape %v493 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v536 = stablehlo.add %v534, %v535 : tensor<256x256x14x14xf32>
    %v537 = stablehlo.reshape %v536 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v538 = stablehlo.reshape %v537 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v539 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v540 = stablehlo.maximum %v538, %v539 : tensor<256x256x14x14xf32>
    %v541 = stablehlo.reshape %v540 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v542 = stablehlo.reshape %v541 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v543 = stablehlo.convolution(%v542, %s3b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v544 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v545 = stablehlo.add %v543, %v544 : tensor<256x256x14x14xf32>
    %v546 = stablehlo.reshape %v545 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v547 = stablehlo.reshape %v546 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v548 = stablehlo.broadcast_in_dim %s3b2n1mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v549 = stablehlo.subtract %v547, %v548 : tensor<256x256x14x14xf32>
    %v550 = stablehlo.broadcast_in_dim %s3b2n1var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v551 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v552 = stablehlo.add %v550, %v551 : tensor<256x256x14x14xf32>
    %v553 = stablehlo.rsqrt %v552 : tensor<256x256x14x14xf32>
    %v554 = stablehlo.multiply %v549, %v553 : tensor<256x256x14x14xf32>
    %v555 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v556 = stablehlo.broadcast_in_dim %s3b2bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v557 = stablehlo.multiply %v554, %v555 : tensor<256x256x14x14xf32>
    %v558 = stablehlo.add %v557, %v556 : tensor<256x256x14x14xf32>
    %v559 = stablehlo.reshape %v558 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v560 = stablehlo.reshape %v559 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v561 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v562 = stablehlo.maximum %v560, %v561 : tensor<256x256x14x14xf32>
    %v563 = stablehlo.reshape %v562 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v564 = stablehlo.reshape %v563 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v565 = stablehlo.convolution(%v564, %s3b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v566 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v567 = stablehlo.add %v565, %v566 : tensor<256x256x14x14xf32>
    %v568 = stablehlo.reshape %v567 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v569 = stablehlo.reshape %v568 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v570 = stablehlo.broadcast_in_dim %s3b2n2mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v571 = stablehlo.subtract %v569, %v570 : tensor<256x256x14x14xf32>
    %v572 = stablehlo.broadcast_in_dim %s3b2n2var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v573 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v574 = stablehlo.add %v572, %v573 : tensor<256x256x14x14xf32>
    %v575 = stablehlo.rsqrt %v574 : tensor<256x256x14x14xf32>
    %v576 = stablehlo.multiply %v571, %v575 : tensor<256x256x14x14xf32>
    %v577 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v578 = stablehlo.broadcast_in_dim %s3b2bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v579 = stablehlo.multiply %v576, %v577 : tensor<256x256x14x14xf32>
    %v580 = stablehlo.add %v579, %v578 : tensor<256x256x14x14xf32>
    %v581 = stablehlo.reshape %v580 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v582 = stablehlo.reshape %v581 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v583 = stablehlo.reshape %v541 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v584 = stablehlo.add %v582, %v583 : tensor<256x256x14x14xf32>
    %v585 = stablehlo.reshape %v584 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v586 = stablehlo.reshape %v585 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v587 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v588 = stablehlo.maximum %v586, %v587 : tensor<256x256x14x14xf32>
    %v589 = stablehlo.reshape %v588 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v590 = stablehlo.reshape %v589 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v591 = stablehlo.convolution(%v590, %s3b3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v592 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v593 = stablehlo.add %v591, %v592 : tensor<256x256x14x14xf32>
    %v594 = stablehlo.reshape %v593 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v595 = stablehlo.reshape %v594 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v596 = stablehlo.broadcast_in_dim %s3b3n1mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v597 = stablehlo.subtract %v595, %v596 : tensor<256x256x14x14xf32>
    %v598 = stablehlo.broadcast_in_dim %s3b3n1var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v599 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v600 = stablehlo.add %v598, %v599 : tensor<256x256x14x14xf32>
    %v601 = stablehlo.rsqrt %v600 : tensor<256x256x14x14xf32>
    %v602 = stablehlo.multiply %v597, %v601 : tensor<256x256x14x14xf32>
    %v603 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v604 = stablehlo.broadcast_in_dim %s3b3bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v605 = stablehlo.multiply %v602, %v603 : tensor<256x256x14x14xf32>
    %v606 = stablehlo.add %v605, %v604 : tensor<256x256x14x14xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v608 = stablehlo.reshape %v607 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v609 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v610 = stablehlo.maximum %v608, %v609 : tensor<256x256x14x14xf32>
    %v611 = stablehlo.reshape %v610 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v612 = stablehlo.reshape %v611 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v613 = stablehlo.convolution(%v612, %s3b3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v614 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v615 = stablehlo.add %v613, %v614 : tensor<256x256x14x14xf32>
    %v616 = stablehlo.reshape %v615 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v617 = stablehlo.reshape %v616 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v618 = stablehlo.broadcast_in_dim %s3b3n2mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v619 = stablehlo.subtract %v617, %v618 : tensor<256x256x14x14xf32>
    %v620 = stablehlo.broadcast_in_dim %s3b3n2var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v621 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v622 = stablehlo.add %v620, %v621 : tensor<256x256x14x14xf32>
    %v623 = stablehlo.rsqrt %v622 : tensor<256x256x14x14xf32>
    %v624 = stablehlo.multiply %v619, %v623 : tensor<256x256x14x14xf32>
    %v625 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v626 = stablehlo.broadcast_in_dim %s3b3bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v627 = stablehlo.multiply %v624, %v625 : tensor<256x256x14x14xf32>
    %v628 = stablehlo.add %v627, %v626 : tensor<256x256x14x14xf32>
    %v629 = stablehlo.reshape %v628 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v630 = stablehlo.reshape %v629 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v631 = stablehlo.reshape %v589 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v632 = stablehlo.add %v630, %v631 : tensor<256x256x14x14xf32>
    %v633 = stablehlo.reshape %v632 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v634 = stablehlo.reshape %v633 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v635 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v636 = stablehlo.maximum %v634, %v635 : tensor<256x256x14x14xf32>
    %v637 = stablehlo.reshape %v636 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v638 = stablehlo.reshape %v637 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v639 = stablehlo.convolution(%v638, %s3b4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v640 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v641 = stablehlo.add %v639, %v640 : tensor<256x256x14x14xf32>
    %v642 = stablehlo.reshape %v641 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v643 = stablehlo.reshape %v642 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v644 = stablehlo.broadcast_in_dim %s3b4n1mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v645 = stablehlo.subtract %v643, %v644 : tensor<256x256x14x14xf32>
    %v646 = stablehlo.broadcast_in_dim %s3b4n1var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v647 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v648 = stablehlo.add %v646, %v647 : tensor<256x256x14x14xf32>
    %v649 = stablehlo.rsqrt %v648 : tensor<256x256x14x14xf32>
    %v650 = stablehlo.multiply %v645, %v649 : tensor<256x256x14x14xf32>
    %v651 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v652 = stablehlo.broadcast_in_dim %s3b4bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v653 = stablehlo.multiply %v650, %v651 : tensor<256x256x14x14xf32>
    %v654 = stablehlo.add %v653, %v652 : tensor<256x256x14x14xf32>
    %v655 = stablehlo.reshape %v654 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v656 = stablehlo.reshape %v655 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v657 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v658 = stablehlo.maximum %v656, %v657 : tensor<256x256x14x14xf32>
    %v659 = stablehlo.reshape %v658 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v660 = stablehlo.reshape %v659 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v661 = stablehlo.convolution(%v660, %s3b4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v662 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v663 = stablehlo.add %v661, %v662 : tensor<256x256x14x14xf32>
    %v664 = stablehlo.reshape %v663 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v665 = stablehlo.reshape %v664 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v666 = stablehlo.broadcast_in_dim %s3b4n2mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v667 = stablehlo.subtract %v665, %v666 : tensor<256x256x14x14xf32>
    %v668 = stablehlo.broadcast_in_dim %s3b4n2var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v669 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v670 = stablehlo.add %v668, %v669 : tensor<256x256x14x14xf32>
    %v671 = stablehlo.rsqrt %v670 : tensor<256x256x14x14xf32>
    %v672 = stablehlo.multiply %v667, %v671 : tensor<256x256x14x14xf32>
    %v673 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v674 = stablehlo.broadcast_in_dim %s3b4bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v675 = stablehlo.multiply %v672, %v673 : tensor<256x256x14x14xf32>
    %v676 = stablehlo.add %v675, %v674 : tensor<256x256x14x14xf32>
    %v677 = stablehlo.reshape %v676 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v678 = stablehlo.reshape %v677 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v679 = stablehlo.reshape %v637 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v680 = stablehlo.add %v678, %v679 : tensor<256x256x14x14xf32>
    %v681 = stablehlo.reshape %v680 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v682 = stablehlo.reshape %v681 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v683 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v684 = stablehlo.maximum %v682, %v683 : tensor<256x256x14x14xf32>
    %v685 = stablehlo.reshape %v684 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v686 = stablehlo.reshape %v685 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v687 = stablehlo.convolution(%v686, %d4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<512x256x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v688 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v689 = stablehlo.add %v687, %v688 : tensor<256x512x7x7xf32>
    %v690 = stablehlo.reshape %v689 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v691 = stablehlo.reshape %v690 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v692 = stablehlo.broadcast_in_dim %d4n1mu, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v693 = stablehlo.subtract %v691, %v692 : tensor<256x512x7x7xf32>
    %v694 = stablehlo.broadcast_in_dim %d4n1var, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v695 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v696 = stablehlo.add %v694, %v695 : tensor<256x512x7x7xf32>
    %v697 = stablehlo.rsqrt %v696 : tensor<256x512x7x7xf32>
    %v698 = stablehlo.multiply %v693, %v697 : tensor<256x512x7x7xf32>
    %v699 = stablehlo.broadcast_in_dim %d4g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v700 = stablehlo.broadcast_in_dim %d4bt1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v701 = stablehlo.multiply %v698, %v699 : tensor<256x512x7x7xf32>
    %v702 = stablehlo.add %v701, %v700 : tensor<256x512x7x7xf32>
    %v703 = stablehlo.reshape %v702 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v704 = stablehlo.reshape %v703 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v705 = stablehlo.constant dense<0.0> : tensor<256x512x7x7xf32>
    %v706 = stablehlo.maximum %v704, %v705 : tensor<256x512x7x7xf32>
    %v707 = stablehlo.reshape %v706 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v708 = stablehlo.reshape %v707 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v709 = stablehlo.convolution(%v708, %d4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v710 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v711 = stablehlo.add %v709, %v710 : tensor<256x512x7x7xf32>
    %v712 = stablehlo.reshape %v711 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v713 = stablehlo.reshape %v712 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v714 = stablehlo.broadcast_in_dim %d4n2mu, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v715 = stablehlo.subtract %v713, %v714 : tensor<256x512x7x7xf32>
    %v716 = stablehlo.broadcast_in_dim %d4n2var, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v717 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v718 = stablehlo.add %v716, %v717 : tensor<256x512x7x7xf32>
    %v719 = stablehlo.rsqrt %v718 : tensor<256x512x7x7xf32>
    %v720 = stablehlo.multiply %v715, %v719 : tensor<256x512x7x7xf32>
    %v721 = stablehlo.broadcast_in_dim %d4g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v722 = stablehlo.broadcast_in_dim %d4bt2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v723 = stablehlo.multiply %v720, %v721 : tensor<256x512x7x7xf32>
    %v724 = stablehlo.add %v723, %v722 : tensor<256x512x7x7xf32>
    %v725 = stablehlo.reshape %v724 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v726 = stablehlo.reshape %v685 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v727 = stablehlo.convolution(%v726, %d4Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<512x256x1x1xf32>) -> tensor<256x512x7x7xf32>
    %v728 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v729 = stablehlo.add %v727, %v728 : tensor<256x512x7x7xf32>
    %v730 = stablehlo.reshape %v729 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v731 = stablehlo.reshape %v730 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v732 = stablehlo.broadcast_in_dim %d4npmu, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v733 = stablehlo.subtract %v731, %v732 : tensor<256x512x7x7xf32>
    %v734 = stablehlo.broadcast_in_dim %d4npvar, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v735 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v736 = stablehlo.add %v734, %v735 : tensor<256x512x7x7xf32>
    %v737 = stablehlo.rsqrt %v736 : tensor<256x512x7x7xf32>
    %v738 = stablehlo.multiply %v733, %v737 : tensor<256x512x7x7xf32>
    %v739 = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v740 = stablehlo.broadcast_in_dim %d4btp, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v741 = stablehlo.multiply %v738, %v739 : tensor<256x512x7x7xf32>
    %v742 = stablehlo.add %v741, %v740 : tensor<256x512x7x7xf32>
    %v743 = stablehlo.reshape %v742 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v744 = stablehlo.reshape %v725 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v745 = stablehlo.reshape %v743 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v746 = stablehlo.add %v744, %v745 : tensor<256x512x7x7xf32>
    %v747 = stablehlo.reshape %v746 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v748 = stablehlo.reshape %v747 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v749 = stablehlo.constant dense<0.0> : tensor<256x512x7x7xf32>
    %v750 = stablehlo.maximum %v748, %v749 : tensor<256x512x7x7xf32>
    %v751 = stablehlo.reshape %v750 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v752 = stablehlo.reshape %v751 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v753 = stablehlo.convolution(%v752, %s4b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v754 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v755 = stablehlo.add %v753, %v754 : tensor<256x512x7x7xf32>
    %v756 = stablehlo.reshape %v755 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v757 = stablehlo.reshape %v756 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v758 = stablehlo.broadcast_in_dim %s4b0n1mu, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v759 = stablehlo.subtract %v757, %v758 : tensor<256x512x7x7xf32>
    %v760 = stablehlo.broadcast_in_dim %s4b0n1var, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v761 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v762 = stablehlo.add %v760, %v761 : tensor<256x512x7x7xf32>
    %v763 = stablehlo.rsqrt %v762 : tensor<256x512x7x7xf32>
    %v764 = stablehlo.multiply %v759, %v763 : tensor<256x512x7x7xf32>
    %v765 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v766 = stablehlo.broadcast_in_dim %s4b0bt1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v767 = stablehlo.multiply %v764, %v765 : tensor<256x512x7x7xf32>
    %v768 = stablehlo.add %v767, %v766 : tensor<256x512x7x7xf32>
    %v769 = stablehlo.reshape %v768 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v770 = stablehlo.reshape %v769 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v771 = stablehlo.constant dense<0.0> : tensor<256x512x7x7xf32>
    %v772 = stablehlo.maximum %v770, %v771 : tensor<256x512x7x7xf32>
    %v773 = stablehlo.reshape %v772 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v774 = stablehlo.reshape %v773 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v775 = stablehlo.convolution(%v774, %s4b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v776 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v777 = stablehlo.add %v775, %v776 : tensor<256x512x7x7xf32>
    %v778 = stablehlo.reshape %v777 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v779 = stablehlo.reshape %v778 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v780 = stablehlo.broadcast_in_dim %s4b0n2mu, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v781 = stablehlo.subtract %v779, %v780 : tensor<256x512x7x7xf32>
    %v782 = stablehlo.broadcast_in_dim %s4b0n2var, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v783 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v784 = stablehlo.add %v782, %v783 : tensor<256x512x7x7xf32>
    %v785 = stablehlo.rsqrt %v784 : tensor<256x512x7x7xf32>
    %v786 = stablehlo.multiply %v781, %v785 : tensor<256x512x7x7xf32>
    %v787 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v788 = stablehlo.broadcast_in_dim %s4b0bt2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v789 = stablehlo.multiply %v786, %v787 : tensor<256x512x7x7xf32>
    %v790 = stablehlo.add %v789, %v788 : tensor<256x512x7x7xf32>
    %v791 = stablehlo.reshape %v790 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v792 = stablehlo.reshape %v791 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v793 = stablehlo.reshape %v751 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v794 = stablehlo.add %v792, %v793 : tensor<256x512x7x7xf32>
    %v795 = stablehlo.reshape %v794 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v796 = stablehlo.reshape %v795 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v797 = stablehlo.constant dense<0.0> : tensor<256x512x7x7xf32>
    %v798 = stablehlo.maximum %v796, %v797 : tensor<256x512x7x7xf32>
    %v799 = stablehlo.reshape %v798 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v800 = stablehlo.reshape %v799 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v801 = stablehlo.convolution(%v800, %s4b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v802 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v803 = stablehlo.add %v801, %v802 : tensor<256x512x7x7xf32>
    %v804 = stablehlo.reshape %v803 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v805 = stablehlo.reshape %v804 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v806 = stablehlo.broadcast_in_dim %s4b1n1mu, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v807 = stablehlo.subtract %v805, %v806 : tensor<256x512x7x7xf32>
    %v808 = stablehlo.broadcast_in_dim %s4b1n1var, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v809 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v810 = stablehlo.add %v808, %v809 : tensor<256x512x7x7xf32>
    %v811 = stablehlo.rsqrt %v810 : tensor<256x512x7x7xf32>
    %v812 = stablehlo.multiply %v807, %v811 : tensor<256x512x7x7xf32>
    %v813 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v814 = stablehlo.broadcast_in_dim %s4b1bt1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v815 = stablehlo.multiply %v812, %v813 : tensor<256x512x7x7xf32>
    %v816 = stablehlo.add %v815, %v814 : tensor<256x512x7x7xf32>
    %v817 = stablehlo.reshape %v816 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v818 = stablehlo.reshape %v817 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v819 = stablehlo.constant dense<0.0> : tensor<256x512x7x7xf32>
    %v820 = stablehlo.maximum %v818, %v819 : tensor<256x512x7x7xf32>
    %v821 = stablehlo.reshape %v820 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v822 = stablehlo.reshape %v821 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v823 = stablehlo.convolution(%v822, %s4b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v824 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v825 = stablehlo.add %v823, %v824 : tensor<256x512x7x7xf32>
    %v826 = stablehlo.reshape %v825 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v827 = stablehlo.reshape %v826 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v828 = stablehlo.broadcast_in_dim %s4b1n2mu, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v829 = stablehlo.subtract %v827, %v828 : tensor<256x512x7x7xf32>
    %v830 = stablehlo.broadcast_in_dim %s4b1n2var, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v831 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v832 = stablehlo.add %v830, %v831 : tensor<256x512x7x7xf32>
    %v833 = stablehlo.rsqrt %v832 : tensor<256x512x7x7xf32>
    %v834 = stablehlo.multiply %v829, %v833 : tensor<256x512x7x7xf32>
    %v835 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v836 = stablehlo.broadcast_in_dim %s4b1bt2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v837 = stablehlo.multiply %v834, %v835 : tensor<256x512x7x7xf32>
    %v838 = stablehlo.add %v837, %v836 : tensor<256x512x7x7xf32>
    %v839 = stablehlo.reshape %v838 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v840 = stablehlo.reshape %v839 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v841 = stablehlo.reshape %v799 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v842 = stablehlo.add %v840, %v841 : tensor<256x512x7x7xf32>
    %v843 = stablehlo.reshape %v842 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v844 = stablehlo.reshape %v843 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v845 = stablehlo.constant dense<0.0> : tensor<256x512x7x7xf32>
    %v846 = stablehlo.maximum %v844, %v845 : tensor<256x512x7x7xf32>
    %v847 = stablehlo.reshape %v846 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v848 = stablehlo.reshape %v847 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v849 = stablehlo.constant dense<0.0> : tensor<f32>
    %v850 = stablehlo.reduce(%v848 init: %v849) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512xf32>
    %v851 = stablehlo.constant dense<49.0> : tensor<256x512xf32>
    %v852 = stablehlo.divide %v850, %v851 : tensor<256x512xf32>
    %v853 = stablehlo.dot_general %v852, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x512xf32>, tensor<512x1000xf32>) -> tensor<256x1000xf32>
    %v854 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<1000xf32>) -> tensor<256x1000xf32>
    %v855 = stablehlo.add %v853, %v854 : tensor<256x1000xf32>
    return %v855 : tensor<256x1000xf32>
  }
}
