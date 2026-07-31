module @m {
  func.func @resnet34_fwd_eval(%x: tensor<32x150528xf32>, %sW: tensor<64x3x7x7xf32>, %sg: tensor<64xf32>, %sbt: tensor<64xf32>, %s1b0W1: tensor<64x64x3x3xf32>, %s1b0g1: tensor<64xf32>, %s1b0bt1: tensor<64xf32>, %s1b0W2: tensor<64x64x3x3xf32>, %s1b0g2: tensor<64xf32>, %s1b0bt2: tensor<64xf32>, %s1b1W1: tensor<64x64x3x3xf32>, %s1b1g1: tensor<64xf32>, %s1b1bt1: tensor<64xf32>, %s1b1W2: tensor<64x64x3x3xf32>, %s1b1g2: tensor<64xf32>, %s1b1bt2: tensor<64xf32>, %s1b2W1: tensor<64x64x3x3xf32>, %s1b2g1: tensor<64xf32>, %s1b2bt1: tensor<64xf32>, %s1b2W2: tensor<64x64x3x3xf32>, %s1b2g2: tensor<64xf32>, %s1b2bt2: tensor<64xf32>, %d2W1: tensor<128x64x3x3xf32>, %d2g1: tensor<128xf32>, %d2bt1: tensor<128xf32>, %d2W2: tensor<128x128x3x3xf32>, %d2g2: tensor<128xf32>, %d2bt2: tensor<128xf32>, %d2Wp: tensor<128x64x3x3xf32>, %d2gp: tensor<128xf32>, %d2btp: tensor<128xf32>, %s2b0W1: tensor<128x128x3x3xf32>, %s2b0g1: tensor<128xf32>, %s2b0bt1: tensor<128xf32>, %s2b0W2: tensor<128x128x3x3xf32>, %s2b0g2: tensor<128xf32>, %s2b0bt2: tensor<128xf32>, %s2b1W1: tensor<128x128x3x3xf32>, %s2b1g1: tensor<128xf32>, %s2b1bt1: tensor<128xf32>, %s2b1W2: tensor<128x128x3x3xf32>, %s2b1g2: tensor<128xf32>, %s2b1bt2: tensor<128xf32>, %s2b2W1: tensor<128x128x3x3xf32>, %s2b2g1: tensor<128xf32>, %s2b2bt1: tensor<128xf32>, %s2b2W2: tensor<128x128x3x3xf32>, %s2b2g2: tensor<128xf32>, %s2b2bt2: tensor<128xf32>, %d3W1: tensor<256x128x3x3xf32>, %d3g1: tensor<256xf32>, %d3bt1: tensor<256xf32>, %d3W2: tensor<256x256x3x3xf32>, %d3g2: tensor<256xf32>, %d3bt2: tensor<256xf32>, %d3Wp: tensor<256x128x3x3xf32>, %d3gp: tensor<256xf32>, %d3btp: tensor<256xf32>, %s3b0W1: tensor<256x256x3x3xf32>, %s3b0g1: tensor<256xf32>, %s3b0bt1: tensor<256xf32>, %s3b0W2: tensor<256x256x3x3xf32>, %s3b0g2: tensor<256xf32>, %s3b0bt2: tensor<256xf32>, %s3b1W1: tensor<256x256x3x3xf32>, %s3b1g1: tensor<256xf32>, %s3b1bt1: tensor<256xf32>, %s3b1W2: tensor<256x256x3x3xf32>, %s3b1g2: tensor<256xf32>, %s3b1bt2: tensor<256xf32>, %s3b2W1: tensor<256x256x3x3xf32>, %s3b2g1: tensor<256xf32>, %s3b2bt1: tensor<256xf32>, %s3b2W2: tensor<256x256x3x3xf32>, %s3b2g2: tensor<256xf32>, %s3b2bt2: tensor<256xf32>, %s3b3W1: tensor<256x256x3x3xf32>, %s3b3g1: tensor<256xf32>, %s3b3bt1: tensor<256xf32>, %s3b3W2: tensor<256x256x3x3xf32>, %s3b3g2: tensor<256xf32>, %s3b3bt2: tensor<256xf32>, %s3b4W1: tensor<256x256x3x3xf32>, %s3b4g1: tensor<256xf32>, %s3b4bt1: tensor<256xf32>, %s3b4W2: tensor<256x256x3x3xf32>, %s3b4g2: tensor<256xf32>, %s3b4bt2: tensor<256xf32>, %d4W1: tensor<512x256x3x3xf32>, %d4g1: tensor<512xf32>, %d4bt1: tensor<512xf32>, %d4W2: tensor<512x512x3x3xf32>, %d4g2: tensor<512xf32>, %d4bt2: tensor<512xf32>, %d4Wp: tensor<512x256x3x3xf32>, %d4gp: tensor<512xf32>, %d4btp: tensor<512xf32>, %s4b0W1: tensor<512x512x3x3xf32>, %s4b0g1: tensor<512xf32>, %s4b0bt1: tensor<512xf32>, %s4b0W2: tensor<512x512x3x3xf32>, %s4b0g2: tensor<512xf32>, %s4b0bt2: tensor<512xf32>, %s4b1W1: tensor<512x512x3x3xf32>, %s4b1g1: tensor<512xf32>, %s4b1bt1: tensor<512xf32>, %s4b1W2: tensor<512x512x3x3xf32>, %s4b1g2: tensor<512xf32>, %s4b1bt2: tensor<512xf32>, %Wd: tensor<512x10xf32>, %bd: tensor<10xf32>, %stnmu: tensor<64xf32>, %stnvar: tensor<64xf32>, %s1b0n1mu: tensor<64xf32>, %s1b0n1var: tensor<64xf32>, %s1b0n2mu: tensor<64xf32>, %s1b0n2var: tensor<64xf32>, %s1b1n1mu: tensor<64xf32>, %s1b1n1var: tensor<64xf32>, %s1b1n2mu: tensor<64xf32>, %s1b1n2var: tensor<64xf32>, %s1b2n1mu: tensor<64xf32>, %s1b2n1var: tensor<64xf32>, %s1b2n2mu: tensor<64xf32>, %s1b2n2var: tensor<64xf32>, %d2n1mu: tensor<128xf32>, %d2n1var: tensor<128xf32>, %d2n2mu: tensor<128xf32>, %d2n2var: tensor<128xf32>, %d2npmu: tensor<128xf32>, %d2npvar: tensor<128xf32>, %s2b0n1mu: tensor<128xf32>, %s2b0n1var: tensor<128xf32>, %s2b0n2mu: tensor<128xf32>, %s2b0n2var: tensor<128xf32>, %s2b1n1mu: tensor<128xf32>, %s2b1n1var: tensor<128xf32>, %s2b1n2mu: tensor<128xf32>, %s2b1n2var: tensor<128xf32>, %s2b2n1mu: tensor<128xf32>, %s2b2n1var: tensor<128xf32>, %s2b2n2mu: tensor<128xf32>, %s2b2n2var: tensor<128xf32>, %d3n1mu: tensor<256xf32>, %d3n1var: tensor<256xf32>, %d3n2mu: tensor<256xf32>, %d3n2var: tensor<256xf32>, %d3npmu: tensor<256xf32>, %d3npvar: tensor<256xf32>, %s3b0n1mu: tensor<256xf32>, %s3b0n1var: tensor<256xf32>, %s3b0n2mu: tensor<256xf32>, %s3b0n2var: tensor<256xf32>, %s3b1n1mu: tensor<256xf32>, %s3b1n1var: tensor<256xf32>, %s3b1n2mu: tensor<256xf32>, %s3b1n2var: tensor<256xf32>, %s3b2n1mu: tensor<256xf32>, %s3b2n1var: tensor<256xf32>, %s3b2n2mu: tensor<256xf32>, %s3b2n2var: tensor<256xf32>, %s3b3n1mu: tensor<256xf32>, %s3b3n1var: tensor<256xf32>, %s3b3n2mu: tensor<256xf32>, %s3b3n2var: tensor<256xf32>, %s3b4n1mu: tensor<256xf32>, %s3b4n1var: tensor<256xf32>, %s3b4n2mu: tensor<256xf32>, %s3b4n2var: tensor<256xf32>, %d4n1mu: tensor<512xf32>, %d4n1var: tensor<512xf32>, %d4n2mu: tensor<512xf32>, %d4n2var: tensor<512xf32>, %d4npmu: tensor<512xf32>, %d4npvar: tensor<512xf32>, %s4b0n1mu: tensor<512xf32>, %s4b0n1var: tensor<512xf32>, %s4b0n2mu: tensor<512xf32>, %s4b0n2var: tensor<512xf32>, %s4b1n1mu: tensor<512xf32>, %s4b1n1var: tensor<512xf32>, %s4b1n2mu: tensor<512xf32>, %s4b1n2var: tensor<512xf32>) -> tensor<32x10xf32> {
    // ── ResNet-34 eval forward (running-stats BN): every line is pretty(verified AST node) ──
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
    %v6 = stablehlo.broadcast_in_dim %stnmu, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v7 = stablehlo.subtract %v5, %v6 : tensor<32x64x112x112xf32>
    %v8 = stablehlo.broadcast_in_dim %stnvar, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v9 = stablehlo.constant dense<1.0e-05> : tensor<32x64x112x112xf32>
    %v10 = stablehlo.add %v8, %v9 : tensor<32x64x112x112xf32>
    %v11 = stablehlo.rsqrt %v10 : tensor<32x64x112x112xf32>
    %v12 = stablehlo.multiply %v7, %v11 : tensor<32x64x112x112xf32>
    %v13 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v14 = stablehlo.broadcast_in_dim %sbt, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v15 = stablehlo.multiply %v12, %v13 : tensor<32x64x112x112xf32>
    %v16 = stablehlo.add %v15, %v14 : tensor<32x64x112x112xf32>
    %v17 = stablehlo.reshape %v16 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v18 = stablehlo.constant dense<0.0> : tensor<32x802816xf32>
    %v19 = stablehlo.maximum %v17, %v18 : tensor<32x802816xf32>
    %v20 = stablehlo.reshape %v19 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v21 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v22 = "stablehlo.reduce_window"(%v20, %v21) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64x56x56xf32>
    %v23 = stablehlo.reshape %v22 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v24 = stablehlo.reshape %v23 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v25 = stablehlo.convolution(%v24, %s1b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v26 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v27 = stablehlo.add %v25, %v26 : tensor<32x64x56x56xf32>
    %v28 = stablehlo.reshape %v27 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v29 = stablehlo.reshape %v28 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v30 = stablehlo.broadcast_in_dim %s1b0n1mu, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v31 = stablehlo.subtract %v29, %v30 : tensor<32x64x56x56xf32>
    %v32 = stablehlo.broadcast_in_dim %s1b0n1var, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v33 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v34 = stablehlo.add %v32, %v33 : tensor<32x64x56x56xf32>
    %v35 = stablehlo.rsqrt %v34 : tensor<32x64x56x56xf32>
    %v36 = stablehlo.multiply %v31, %v35 : tensor<32x64x56x56xf32>
    %v37 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v38 = stablehlo.broadcast_in_dim %s1b0bt1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v39 = stablehlo.multiply %v36, %v37 : tensor<32x64x56x56xf32>
    %v40 = stablehlo.add %v39, %v38 : tensor<32x64x56x56xf32>
    %v41 = stablehlo.reshape %v40 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v42 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v43 = stablehlo.maximum %v41, %v42 : tensor<32x200704xf32>
    %v44 = stablehlo.reshape %v43 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v45 = stablehlo.convolution(%v44, %s1b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v46 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v47 = stablehlo.add %v45, %v46 : tensor<32x64x56x56xf32>
    %v48 = stablehlo.reshape %v47 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v49 = stablehlo.reshape %v48 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v50 = stablehlo.broadcast_in_dim %s1b0n2mu, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v51 = stablehlo.subtract %v49, %v50 : tensor<32x64x56x56xf32>
    %v52 = stablehlo.broadcast_in_dim %s1b0n2var, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v53 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v54 = stablehlo.add %v52, %v53 : tensor<32x64x56x56xf32>
    %v55 = stablehlo.rsqrt %v54 : tensor<32x64x56x56xf32>
    %v56 = stablehlo.multiply %v51, %v55 : tensor<32x64x56x56xf32>
    %v57 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v58 = stablehlo.broadcast_in_dim %s1b0bt2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v59 = stablehlo.multiply %v56, %v57 : tensor<32x64x56x56xf32>
    %v60 = stablehlo.add %v59, %v58 : tensor<32x64x56x56xf32>
    %v61 = stablehlo.reshape %v60 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v62 = stablehlo.add %v61, %v23 : tensor<32x200704xf32>
    %v63 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v64 = stablehlo.maximum %v62, %v63 : tensor<32x200704xf32>
    %v65 = stablehlo.reshape %v64 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v66 = stablehlo.convolution(%v65, %s1b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v67 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v68 = stablehlo.add %v66, %v67 : tensor<32x64x56x56xf32>
    %v69 = stablehlo.reshape %v68 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v70 = stablehlo.reshape %v69 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v71 = stablehlo.broadcast_in_dim %s1b1n1mu, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v72 = stablehlo.subtract %v70, %v71 : tensor<32x64x56x56xf32>
    %v73 = stablehlo.broadcast_in_dim %s1b1n1var, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v74 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v75 = stablehlo.add %v73, %v74 : tensor<32x64x56x56xf32>
    %v76 = stablehlo.rsqrt %v75 : tensor<32x64x56x56xf32>
    %v77 = stablehlo.multiply %v72, %v76 : tensor<32x64x56x56xf32>
    %v78 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v79 = stablehlo.broadcast_in_dim %s1b1bt1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v80 = stablehlo.multiply %v77, %v78 : tensor<32x64x56x56xf32>
    %v81 = stablehlo.add %v80, %v79 : tensor<32x64x56x56xf32>
    %v82 = stablehlo.reshape %v81 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v83 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v84 = stablehlo.maximum %v82, %v83 : tensor<32x200704xf32>
    %v85 = stablehlo.reshape %v84 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v86 = stablehlo.convolution(%v85, %s1b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v87 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v88 = stablehlo.add %v86, %v87 : tensor<32x64x56x56xf32>
    %v89 = stablehlo.reshape %v88 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v90 = stablehlo.reshape %v89 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v91 = stablehlo.broadcast_in_dim %s1b1n2mu, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v92 = stablehlo.subtract %v90, %v91 : tensor<32x64x56x56xf32>
    %v93 = stablehlo.broadcast_in_dim %s1b1n2var, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v94 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v95 = stablehlo.add %v93, %v94 : tensor<32x64x56x56xf32>
    %v96 = stablehlo.rsqrt %v95 : tensor<32x64x56x56xf32>
    %v97 = stablehlo.multiply %v92, %v96 : tensor<32x64x56x56xf32>
    %v98 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v99 = stablehlo.broadcast_in_dim %s1b1bt2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v100 = stablehlo.multiply %v97, %v98 : tensor<32x64x56x56xf32>
    %v101 = stablehlo.add %v100, %v99 : tensor<32x64x56x56xf32>
    %v102 = stablehlo.reshape %v101 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v103 = stablehlo.add %v102, %v64 : tensor<32x200704xf32>
    %v104 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v105 = stablehlo.maximum %v103, %v104 : tensor<32x200704xf32>
    %v106 = stablehlo.reshape %v105 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v107 = stablehlo.convolution(%v106, %s1b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v108 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v109 = stablehlo.add %v107, %v108 : tensor<32x64x56x56xf32>
    %v110 = stablehlo.reshape %v109 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v111 = stablehlo.reshape %v110 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v112 = stablehlo.broadcast_in_dim %s1b2n1mu, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v113 = stablehlo.subtract %v111, %v112 : tensor<32x64x56x56xf32>
    %v114 = stablehlo.broadcast_in_dim %s1b2n1var, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v115 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v116 = stablehlo.add %v114, %v115 : tensor<32x64x56x56xf32>
    %v117 = stablehlo.rsqrt %v116 : tensor<32x64x56x56xf32>
    %v118 = stablehlo.multiply %v113, %v117 : tensor<32x64x56x56xf32>
    %v119 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v120 = stablehlo.broadcast_in_dim %s1b2bt1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v121 = stablehlo.multiply %v118, %v119 : tensor<32x64x56x56xf32>
    %v122 = stablehlo.add %v121, %v120 : tensor<32x64x56x56xf32>
    %v123 = stablehlo.reshape %v122 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v124 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v125 = stablehlo.maximum %v123, %v124 : tensor<32x200704xf32>
    %v126 = stablehlo.reshape %v125 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v127 = stablehlo.convolution(%v126, %s1b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v128 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v129 = stablehlo.add %v127, %v128 : tensor<32x64x56x56xf32>
    %v130 = stablehlo.reshape %v129 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v131 = stablehlo.reshape %v130 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v132 = stablehlo.broadcast_in_dim %s1b2n2mu, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v133 = stablehlo.subtract %v131, %v132 : tensor<32x64x56x56xf32>
    %v134 = stablehlo.broadcast_in_dim %s1b2n2var, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v135 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v136 = stablehlo.add %v134, %v135 : tensor<32x64x56x56xf32>
    %v137 = stablehlo.rsqrt %v136 : tensor<32x64x56x56xf32>
    %v138 = stablehlo.multiply %v133, %v137 : tensor<32x64x56x56xf32>
    %v139 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v140 = stablehlo.broadcast_in_dim %s1b2bt2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v141 = stablehlo.multiply %v138, %v139 : tensor<32x64x56x56xf32>
    %v142 = stablehlo.add %v141, %v140 : tensor<32x64x56x56xf32>
    %v143 = stablehlo.reshape %v142 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v144 = stablehlo.add %v143, %v105 : tensor<32x200704xf32>
    %v145 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v146 = stablehlo.maximum %v144, %v145 : tensor<32x200704xf32>
    %v147 = stablehlo.reshape %v146 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v148 = stablehlo.convolution(%v147, %d2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<128x64x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v149 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v150 = stablehlo.add %v148, %v149 : tensor<32x128x28x28xf32>
    %v151 = stablehlo.reshape %v150 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v152 = stablehlo.reshape %v151 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v153 = stablehlo.broadcast_in_dim %d2n1mu, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v154 = stablehlo.subtract %v152, %v153 : tensor<32x128x28x28xf32>
    %v155 = stablehlo.broadcast_in_dim %d2n1var, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v156 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v157 = stablehlo.add %v155, %v156 : tensor<32x128x28x28xf32>
    %v158 = stablehlo.rsqrt %v157 : tensor<32x128x28x28xf32>
    %v159 = stablehlo.multiply %v154, %v158 : tensor<32x128x28x28xf32>
    %v160 = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v161 = stablehlo.broadcast_in_dim %d2bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v162 = stablehlo.multiply %v159, %v160 : tensor<32x128x28x28xf32>
    %v163 = stablehlo.add %v162, %v161 : tensor<32x128x28x28xf32>
    %v164 = stablehlo.reshape %v163 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v165 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v166 = stablehlo.maximum %v164, %v165 : tensor<32x100352xf32>
    %v167 = stablehlo.reshape %v166 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v168 = stablehlo.convolution(%v167, %d2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v169 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v170 = stablehlo.add %v168, %v169 : tensor<32x128x28x28xf32>
    %v171 = stablehlo.reshape %v170 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v172 = stablehlo.reshape %v171 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v173 = stablehlo.broadcast_in_dim %d2n2mu, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v174 = stablehlo.subtract %v172, %v173 : tensor<32x128x28x28xf32>
    %v175 = stablehlo.broadcast_in_dim %d2n2var, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v176 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v177 = stablehlo.add %v175, %v176 : tensor<32x128x28x28xf32>
    %v178 = stablehlo.rsqrt %v177 : tensor<32x128x28x28xf32>
    %v179 = stablehlo.multiply %v174, %v178 : tensor<32x128x28x28xf32>
    %v180 = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v181 = stablehlo.broadcast_in_dim %d2bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v182 = stablehlo.multiply %v179, %v180 : tensor<32x128x28x28xf32>
    %v183 = stablehlo.add %v182, %v181 : tensor<32x128x28x28xf32>
    %v184 = stablehlo.reshape %v183 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v185 = stablehlo.reshape %v146 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v186 = stablehlo.convolution(%v185, %d2Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<128x64x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v187 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v188 = stablehlo.add %v186, %v187 : tensor<32x128x28x28xf32>
    %v189 = stablehlo.reshape %v188 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v190 = stablehlo.reshape %v189 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v191 = stablehlo.broadcast_in_dim %d2npmu, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v192 = stablehlo.subtract %v190, %v191 : tensor<32x128x28x28xf32>
    %v193 = stablehlo.broadcast_in_dim %d2npvar, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v194 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v195 = stablehlo.add %v193, %v194 : tensor<32x128x28x28xf32>
    %v196 = stablehlo.rsqrt %v195 : tensor<32x128x28x28xf32>
    %v197 = stablehlo.multiply %v192, %v196 : tensor<32x128x28x28xf32>
    %v198 = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v199 = stablehlo.broadcast_in_dim %d2btp, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v200 = stablehlo.multiply %v197, %v198 : tensor<32x128x28x28xf32>
    %v201 = stablehlo.add %v200, %v199 : tensor<32x128x28x28xf32>
    %v202 = stablehlo.reshape %v201 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v203 = stablehlo.add %v184, %v202 : tensor<32x100352xf32>
    %v204 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v205 = stablehlo.maximum %v203, %v204 : tensor<32x100352xf32>
    %v206 = stablehlo.reshape %v205 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v207 = stablehlo.convolution(%v206, %s2b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v208 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v209 = stablehlo.add %v207, %v208 : tensor<32x128x28x28xf32>
    %v210 = stablehlo.reshape %v209 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v211 = stablehlo.reshape %v210 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v212 = stablehlo.broadcast_in_dim %s2b0n1mu, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v213 = stablehlo.subtract %v211, %v212 : tensor<32x128x28x28xf32>
    %v214 = stablehlo.broadcast_in_dim %s2b0n1var, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v215 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v216 = stablehlo.add %v214, %v215 : tensor<32x128x28x28xf32>
    %v217 = stablehlo.rsqrt %v216 : tensor<32x128x28x28xf32>
    %v218 = stablehlo.multiply %v213, %v217 : tensor<32x128x28x28xf32>
    %v219 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v220 = stablehlo.broadcast_in_dim %s2b0bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v221 = stablehlo.multiply %v218, %v219 : tensor<32x128x28x28xf32>
    %v222 = stablehlo.add %v221, %v220 : tensor<32x128x28x28xf32>
    %v223 = stablehlo.reshape %v222 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v224 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v225 = stablehlo.maximum %v223, %v224 : tensor<32x100352xf32>
    %v226 = stablehlo.reshape %v225 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v227 = stablehlo.convolution(%v226, %s2b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v228 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v229 = stablehlo.add %v227, %v228 : tensor<32x128x28x28xf32>
    %v230 = stablehlo.reshape %v229 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v231 = stablehlo.reshape %v230 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v232 = stablehlo.broadcast_in_dim %s2b0n2mu, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v233 = stablehlo.subtract %v231, %v232 : tensor<32x128x28x28xf32>
    %v234 = stablehlo.broadcast_in_dim %s2b0n2var, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v235 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v236 = stablehlo.add %v234, %v235 : tensor<32x128x28x28xf32>
    %v237 = stablehlo.rsqrt %v236 : tensor<32x128x28x28xf32>
    %v238 = stablehlo.multiply %v233, %v237 : tensor<32x128x28x28xf32>
    %v239 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v240 = stablehlo.broadcast_in_dim %s2b0bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v241 = stablehlo.multiply %v238, %v239 : tensor<32x128x28x28xf32>
    %v242 = stablehlo.add %v241, %v240 : tensor<32x128x28x28xf32>
    %v243 = stablehlo.reshape %v242 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v244 = stablehlo.add %v243, %v205 : tensor<32x100352xf32>
    %v245 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v246 = stablehlo.maximum %v244, %v245 : tensor<32x100352xf32>
    %v247 = stablehlo.reshape %v246 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v248 = stablehlo.convolution(%v247, %s2b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v249 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v250 = stablehlo.add %v248, %v249 : tensor<32x128x28x28xf32>
    %v251 = stablehlo.reshape %v250 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v252 = stablehlo.reshape %v251 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v253 = stablehlo.broadcast_in_dim %s2b1n1mu, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v254 = stablehlo.subtract %v252, %v253 : tensor<32x128x28x28xf32>
    %v255 = stablehlo.broadcast_in_dim %s2b1n1var, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v256 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v257 = stablehlo.add %v255, %v256 : tensor<32x128x28x28xf32>
    %v258 = stablehlo.rsqrt %v257 : tensor<32x128x28x28xf32>
    %v259 = stablehlo.multiply %v254, %v258 : tensor<32x128x28x28xf32>
    %v260 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v261 = stablehlo.broadcast_in_dim %s2b1bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v262 = stablehlo.multiply %v259, %v260 : tensor<32x128x28x28xf32>
    %v263 = stablehlo.add %v262, %v261 : tensor<32x128x28x28xf32>
    %v264 = stablehlo.reshape %v263 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v265 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v266 = stablehlo.maximum %v264, %v265 : tensor<32x100352xf32>
    %v267 = stablehlo.reshape %v266 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v268 = stablehlo.convolution(%v267, %s2b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v269 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v270 = stablehlo.add %v268, %v269 : tensor<32x128x28x28xf32>
    %v271 = stablehlo.reshape %v270 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v272 = stablehlo.reshape %v271 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v273 = stablehlo.broadcast_in_dim %s2b1n2mu, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v274 = stablehlo.subtract %v272, %v273 : tensor<32x128x28x28xf32>
    %v275 = stablehlo.broadcast_in_dim %s2b1n2var, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v276 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v277 = stablehlo.add %v275, %v276 : tensor<32x128x28x28xf32>
    %v278 = stablehlo.rsqrt %v277 : tensor<32x128x28x28xf32>
    %v279 = stablehlo.multiply %v274, %v278 : tensor<32x128x28x28xf32>
    %v280 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v281 = stablehlo.broadcast_in_dim %s2b1bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v282 = stablehlo.multiply %v279, %v280 : tensor<32x128x28x28xf32>
    %v283 = stablehlo.add %v282, %v281 : tensor<32x128x28x28xf32>
    %v284 = stablehlo.reshape %v283 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v285 = stablehlo.add %v284, %v246 : tensor<32x100352xf32>
    %v286 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v287 = stablehlo.maximum %v285, %v286 : tensor<32x100352xf32>
    %v288 = stablehlo.reshape %v287 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v289 = stablehlo.convolution(%v288, %s2b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v290 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v291 = stablehlo.add %v289, %v290 : tensor<32x128x28x28xf32>
    %v292 = stablehlo.reshape %v291 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v293 = stablehlo.reshape %v292 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v294 = stablehlo.broadcast_in_dim %s2b2n1mu, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v295 = stablehlo.subtract %v293, %v294 : tensor<32x128x28x28xf32>
    %v296 = stablehlo.broadcast_in_dim %s2b2n1var, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v297 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v298 = stablehlo.add %v296, %v297 : tensor<32x128x28x28xf32>
    %v299 = stablehlo.rsqrt %v298 : tensor<32x128x28x28xf32>
    %v300 = stablehlo.multiply %v295, %v299 : tensor<32x128x28x28xf32>
    %v301 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v302 = stablehlo.broadcast_in_dim %s2b2bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v303 = stablehlo.multiply %v300, %v301 : tensor<32x128x28x28xf32>
    %v304 = stablehlo.add %v303, %v302 : tensor<32x128x28x28xf32>
    %v305 = stablehlo.reshape %v304 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v306 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v307 = stablehlo.maximum %v305, %v306 : tensor<32x100352xf32>
    %v308 = stablehlo.reshape %v307 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v309 = stablehlo.convolution(%v308, %s2b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v310 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v311 = stablehlo.add %v309, %v310 : tensor<32x128x28x28xf32>
    %v312 = stablehlo.reshape %v311 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v313 = stablehlo.reshape %v312 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v314 = stablehlo.broadcast_in_dim %s2b2n2mu, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v315 = stablehlo.subtract %v313, %v314 : tensor<32x128x28x28xf32>
    %v316 = stablehlo.broadcast_in_dim %s2b2n2var, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v317 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v318 = stablehlo.add %v316, %v317 : tensor<32x128x28x28xf32>
    %v319 = stablehlo.rsqrt %v318 : tensor<32x128x28x28xf32>
    %v320 = stablehlo.multiply %v315, %v319 : tensor<32x128x28x28xf32>
    %v321 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v322 = stablehlo.broadcast_in_dim %s2b2bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v323 = stablehlo.multiply %v320, %v321 : tensor<32x128x28x28xf32>
    %v324 = stablehlo.add %v323, %v322 : tensor<32x128x28x28xf32>
    %v325 = stablehlo.reshape %v324 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v326 = stablehlo.add %v325, %v287 : tensor<32x100352xf32>
    %v327 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v328 = stablehlo.maximum %v326, %v327 : tensor<32x100352xf32>
    %v329 = stablehlo.reshape %v328 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v330 = stablehlo.convolution(%v329, %d3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<256x128x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v331 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v332 = stablehlo.add %v330, %v331 : tensor<32x256x14x14xf32>
    %v333 = stablehlo.reshape %v332 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v334 = stablehlo.reshape %v333 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v335 = stablehlo.broadcast_in_dim %d3n1mu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v336 = stablehlo.subtract %v334, %v335 : tensor<32x256x14x14xf32>
    %v337 = stablehlo.broadcast_in_dim %d3n1var, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v338 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v339 = stablehlo.add %v337, %v338 : tensor<32x256x14x14xf32>
    %v340 = stablehlo.rsqrt %v339 : tensor<32x256x14x14xf32>
    %v341 = stablehlo.multiply %v336, %v340 : tensor<32x256x14x14xf32>
    %v342 = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v343 = stablehlo.broadcast_in_dim %d3bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v344 = stablehlo.multiply %v341, %v342 : tensor<32x256x14x14xf32>
    %v345 = stablehlo.add %v344, %v343 : tensor<32x256x14x14xf32>
    %v346 = stablehlo.reshape %v345 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v347 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v348 = stablehlo.maximum %v346, %v347 : tensor<32x50176xf32>
    %v349 = stablehlo.reshape %v348 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v350 = stablehlo.convolution(%v349, %d3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v351 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v352 = stablehlo.add %v350, %v351 : tensor<32x256x14x14xf32>
    %v353 = stablehlo.reshape %v352 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v354 = stablehlo.reshape %v353 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v355 = stablehlo.broadcast_in_dim %d3n2mu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v356 = stablehlo.subtract %v354, %v355 : tensor<32x256x14x14xf32>
    %v357 = stablehlo.broadcast_in_dim %d3n2var, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v358 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v359 = stablehlo.add %v357, %v358 : tensor<32x256x14x14xf32>
    %v360 = stablehlo.rsqrt %v359 : tensor<32x256x14x14xf32>
    %v361 = stablehlo.multiply %v356, %v360 : tensor<32x256x14x14xf32>
    %v362 = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v363 = stablehlo.broadcast_in_dim %d3bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v364 = stablehlo.multiply %v361, %v362 : tensor<32x256x14x14xf32>
    %v365 = stablehlo.add %v364, %v363 : tensor<32x256x14x14xf32>
    %v366 = stablehlo.reshape %v365 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v367 = stablehlo.reshape %v328 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v368 = stablehlo.convolution(%v367, %d3Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<256x128x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v369 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v370 = stablehlo.add %v368, %v369 : tensor<32x256x14x14xf32>
    %v371 = stablehlo.reshape %v370 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v372 = stablehlo.reshape %v371 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v373 = stablehlo.broadcast_in_dim %d3npmu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v374 = stablehlo.subtract %v372, %v373 : tensor<32x256x14x14xf32>
    %v375 = stablehlo.broadcast_in_dim %d3npvar, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v376 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v377 = stablehlo.add %v375, %v376 : tensor<32x256x14x14xf32>
    %v378 = stablehlo.rsqrt %v377 : tensor<32x256x14x14xf32>
    %v379 = stablehlo.multiply %v374, %v378 : tensor<32x256x14x14xf32>
    %v380 = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v381 = stablehlo.broadcast_in_dim %d3btp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v382 = stablehlo.multiply %v379, %v380 : tensor<32x256x14x14xf32>
    %v383 = stablehlo.add %v382, %v381 : tensor<32x256x14x14xf32>
    %v384 = stablehlo.reshape %v383 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v385 = stablehlo.add %v366, %v384 : tensor<32x50176xf32>
    %v386 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v387 = stablehlo.maximum %v385, %v386 : tensor<32x50176xf32>
    %v388 = stablehlo.reshape %v387 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v389 = stablehlo.convolution(%v388, %s3b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v390 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v391 = stablehlo.add %v389, %v390 : tensor<32x256x14x14xf32>
    %v392 = stablehlo.reshape %v391 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v393 = stablehlo.reshape %v392 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v394 = stablehlo.broadcast_in_dim %s3b0n1mu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v395 = stablehlo.subtract %v393, %v394 : tensor<32x256x14x14xf32>
    %v396 = stablehlo.broadcast_in_dim %s3b0n1var, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v397 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v398 = stablehlo.add %v396, %v397 : tensor<32x256x14x14xf32>
    %v399 = stablehlo.rsqrt %v398 : tensor<32x256x14x14xf32>
    %v400 = stablehlo.multiply %v395, %v399 : tensor<32x256x14x14xf32>
    %v401 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v402 = stablehlo.broadcast_in_dim %s3b0bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v403 = stablehlo.multiply %v400, %v401 : tensor<32x256x14x14xf32>
    %v404 = stablehlo.add %v403, %v402 : tensor<32x256x14x14xf32>
    %v405 = stablehlo.reshape %v404 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v406 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v407 = stablehlo.maximum %v405, %v406 : tensor<32x50176xf32>
    %v408 = stablehlo.reshape %v407 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v409 = stablehlo.convolution(%v408, %s3b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v410 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v411 = stablehlo.add %v409, %v410 : tensor<32x256x14x14xf32>
    %v412 = stablehlo.reshape %v411 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v413 = stablehlo.reshape %v412 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v414 = stablehlo.broadcast_in_dim %s3b0n2mu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v415 = stablehlo.subtract %v413, %v414 : tensor<32x256x14x14xf32>
    %v416 = stablehlo.broadcast_in_dim %s3b0n2var, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v417 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v418 = stablehlo.add %v416, %v417 : tensor<32x256x14x14xf32>
    %v419 = stablehlo.rsqrt %v418 : tensor<32x256x14x14xf32>
    %v420 = stablehlo.multiply %v415, %v419 : tensor<32x256x14x14xf32>
    %v421 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v422 = stablehlo.broadcast_in_dim %s3b0bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v423 = stablehlo.multiply %v420, %v421 : tensor<32x256x14x14xf32>
    %v424 = stablehlo.add %v423, %v422 : tensor<32x256x14x14xf32>
    %v425 = stablehlo.reshape %v424 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v426 = stablehlo.add %v425, %v387 : tensor<32x50176xf32>
    %v427 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v428 = stablehlo.maximum %v426, %v427 : tensor<32x50176xf32>
    %v429 = stablehlo.reshape %v428 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v430 = stablehlo.convolution(%v429, %s3b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v431 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v432 = stablehlo.add %v430, %v431 : tensor<32x256x14x14xf32>
    %v433 = stablehlo.reshape %v432 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v434 = stablehlo.reshape %v433 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v435 = stablehlo.broadcast_in_dim %s3b1n1mu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v436 = stablehlo.subtract %v434, %v435 : tensor<32x256x14x14xf32>
    %v437 = stablehlo.broadcast_in_dim %s3b1n1var, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v438 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v439 = stablehlo.add %v437, %v438 : tensor<32x256x14x14xf32>
    %v440 = stablehlo.rsqrt %v439 : tensor<32x256x14x14xf32>
    %v441 = stablehlo.multiply %v436, %v440 : tensor<32x256x14x14xf32>
    %v442 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v443 = stablehlo.broadcast_in_dim %s3b1bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v444 = stablehlo.multiply %v441, %v442 : tensor<32x256x14x14xf32>
    %v445 = stablehlo.add %v444, %v443 : tensor<32x256x14x14xf32>
    %v446 = stablehlo.reshape %v445 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v447 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v448 = stablehlo.maximum %v446, %v447 : tensor<32x50176xf32>
    %v449 = stablehlo.reshape %v448 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v450 = stablehlo.convolution(%v449, %s3b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v451 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v452 = stablehlo.add %v450, %v451 : tensor<32x256x14x14xf32>
    %v453 = stablehlo.reshape %v452 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v454 = stablehlo.reshape %v453 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v455 = stablehlo.broadcast_in_dim %s3b1n2mu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v456 = stablehlo.subtract %v454, %v455 : tensor<32x256x14x14xf32>
    %v457 = stablehlo.broadcast_in_dim %s3b1n2var, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v458 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v459 = stablehlo.add %v457, %v458 : tensor<32x256x14x14xf32>
    %v460 = stablehlo.rsqrt %v459 : tensor<32x256x14x14xf32>
    %v461 = stablehlo.multiply %v456, %v460 : tensor<32x256x14x14xf32>
    %v462 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v463 = stablehlo.broadcast_in_dim %s3b1bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v464 = stablehlo.multiply %v461, %v462 : tensor<32x256x14x14xf32>
    %v465 = stablehlo.add %v464, %v463 : tensor<32x256x14x14xf32>
    %v466 = stablehlo.reshape %v465 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v467 = stablehlo.add %v466, %v428 : tensor<32x50176xf32>
    %v468 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v469 = stablehlo.maximum %v467, %v468 : tensor<32x50176xf32>
    %v470 = stablehlo.reshape %v469 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v471 = stablehlo.convolution(%v470, %s3b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v472 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v473 = stablehlo.add %v471, %v472 : tensor<32x256x14x14xf32>
    %v474 = stablehlo.reshape %v473 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v475 = stablehlo.reshape %v474 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v476 = stablehlo.broadcast_in_dim %s3b2n1mu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v477 = stablehlo.subtract %v475, %v476 : tensor<32x256x14x14xf32>
    %v478 = stablehlo.broadcast_in_dim %s3b2n1var, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v479 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v480 = stablehlo.add %v478, %v479 : tensor<32x256x14x14xf32>
    %v481 = stablehlo.rsqrt %v480 : tensor<32x256x14x14xf32>
    %v482 = stablehlo.multiply %v477, %v481 : tensor<32x256x14x14xf32>
    %v483 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v484 = stablehlo.broadcast_in_dim %s3b2bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v485 = stablehlo.multiply %v482, %v483 : tensor<32x256x14x14xf32>
    %v486 = stablehlo.add %v485, %v484 : tensor<32x256x14x14xf32>
    %v487 = stablehlo.reshape %v486 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v488 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v489 = stablehlo.maximum %v487, %v488 : tensor<32x50176xf32>
    %v490 = stablehlo.reshape %v489 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v491 = stablehlo.convolution(%v490, %s3b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v492 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v493 = stablehlo.add %v491, %v492 : tensor<32x256x14x14xf32>
    %v494 = stablehlo.reshape %v493 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v495 = stablehlo.reshape %v494 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v496 = stablehlo.broadcast_in_dim %s3b2n2mu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v497 = stablehlo.subtract %v495, %v496 : tensor<32x256x14x14xf32>
    %v498 = stablehlo.broadcast_in_dim %s3b2n2var, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v499 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v500 = stablehlo.add %v498, %v499 : tensor<32x256x14x14xf32>
    %v501 = stablehlo.rsqrt %v500 : tensor<32x256x14x14xf32>
    %v502 = stablehlo.multiply %v497, %v501 : tensor<32x256x14x14xf32>
    %v503 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v504 = stablehlo.broadcast_in_dim %s3b2bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v505 = stablehlo.multiply %v502, %v503 : tensor<32x256x14x14xf32>
    %v506 = stablehlo.add %v505, %v504 : tensor<32x256x14x14xf32>
    %v507 = stablehlo.reshape %v506 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v508 = stablehlo.add %v507, %v469 : tensor<32x50176xf32>
    %v509 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v510 = stablehlo.maximum %v508, %v509 : tensor<32x50176xf32>
    %v511 = stablehlo.reshape %v510 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v512 = stablehlo.convolution(%v511, %s3b3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v513 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v514 = stablehlo.add %v512, %v513 : tensor<32x256x14x14xf32>
    %v515 = stablehlo.reshape %v514 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v516 = stablehlo.reshape %v515 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v517 = stablehlo.broadcast_in_dim %s3b3n1mu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v518 = stablehlo.subtract %v516, %v517 : tensor<32x256x14x14xf32>
    %v519 = stablehlo.broadcast_in_dim %s3b3n1var, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v520 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v521 = stablehlo.add %v519, %v520 : tensor<32x256x14x14xf32>
    %v522 = stablehlo.rsqrt %v521 : tensor<32x256x14x14xf32>
    %v523 = stablehlo.multiply %v518, %v522 : tensor<32x256x14x14xf32>
    %v524 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v525 = stablehlo.broadcast_in_dim %s3b3bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v526 = stablehlo.multiply %v523, %v524 : tensor<32x256x14x14xf32>
    %v527 = stablehlo.add %v526, %v525 : tensor<32x256x14x14xf32>
    %v528 = stablehlo.reshape %v527 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v529 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v530 = stablehlo.maximum %v528, %v529 : tensor<32x50176xf32>
    %v531 = stablehlo.reshape %v530 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v532 = stablehlo.convolution(%v531, %s3b3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v533 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v534 = stablehlo.add %v532, %v533 : tensor<32x256x14x14xf32>
    %v535 = stablehlo.reshape %v534 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v536 = stablehlo.reshape %v535 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v537 = stablehlo.broadcast_in_dim %s3b3n2mu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v538 = stablehlo.subtract %v536, %v537 : tensor<32x256x14x14xf32>
    %v539 = stablehlo.broadcast_in_dim %s3b3n2var, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v540 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v541 = stablehlo.add %v539, %v540 : tensor<32x256x14x14xf32>
    %v542 = stablehlo.rsqrt %v541 : tensor<32x256x14x14xf32>
    %v543 = stablehlo.multiply %v538, %v542 : tensor<32x256x14x14xf32>
    %v544 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v545 = stablehlo.broadcast_in_dim %s3b3bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v546 = stablehlo.multiply %v543, %v544 : tensor<32x256x14x14xf32>
    %v547 = stablehlo.add %v546, %v545 : tensor<32x256x14x14xf32>
    %v548 = stablehlo.reshape %v547 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v549 = stablehlo.add %v548, %v510 : tensor<32x50176xf32>
    %v550 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v551 = stablehlo.maximum %v549, %v550 : tensor<32x50176xf32>
    %v552 = stablehlo.reshape %v551 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v553 = stablehlo.convolution(%v552, %s3b4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v554 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v555 = stablehlo.add %v553, %v554 : tensor<32x256x14x14xf32>
    %v556 = stablehlo.reshape %v555 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v557 = stablehlo.reshape %v556 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v558 = stablehlo.broadcast_in_dim %s3b4n1mu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v559 = stablehlo.subtract %v557, %v558 : tensor<32x256x14x14xf32>
    %v560 = stablehlo.broadcast_in_dim %s3b4n1var, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v561 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v562 = stablehlo.add %v560, %v561 : tensor<32x256x14x14xf32>
    %v563 = stablehlo.rsqrt %v562 : tensor<32x256x14x14xf32>
    %v564 = stablehlo.multiply %v559, %v563 : tensor<32x256x14x14xf32>
    %v565 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v566 = stablehlo.broadcast_in_dim %s3b4bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v567 = stablehlo.multiply %v564, %v565 : tensor<32x256x14x14xf32>
    %v568 = stablehlo.add %v567, %v566 : tensor<32x256x14x14xf32>
    %v569 = stablehlo.reshape %v568 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v570 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v571 = stablehlo.maximum %v569, %v570 : tensor<32x50176xf32>
    %v572 = stablehlo.reshape %v571 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v573 = stablehlo.convolution(%v572, %s3b4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v574 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v575 = stablehlo.add %v573, %v574 : tensor<32x256x14x14xf32>
    %v576 = stablehlo.reshape %v575 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v577 = stablehlo.reshape %v576 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v578 = stablehlo.broadcast_in_dim %s3b4n2mu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v579 = stablehlo.subtract %v577, %v578 : tensor<32x256x14x14xf32>
    %v580 = stablehlo.broadcast_in_dim %s3b4n2var, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v581 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v582 = stablehlo.add %v580, %v581 : tensor<32x256x14x14xf32>
    %v583 = stablehlo.rsqrt %v582 : tensor<32x256x14x14xf32>
    %v584 = stablehlo.multiply %v579, %v583 : tensor<32x256x14x14xf32>
    %v585 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v586 = stablehlo.broadcast_in_dim %s3b4bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v587 = stablehlo.multiply %v584, %v585 : tensor<32x256x14x14xf32>
    %v588 = stablehlo.add %v587, %v586 : tensor<32x256x14x14xf32>
    %v589 = stablehlo.reshape %v588 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v590 = stablehlo.add %v589, %v551 : tensor<32x50176xf32>
    %v591 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v592 = stablehlo.maximum %v590, %v591 : tensor<32x50176xf32>
    %v593 = stablehlo.reshape %v592 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v594 = stablehlo.convolution(%v593, %d4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<512x256x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v595 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v596 = stablehlo.add %v594, %v595 : tensor<32x512x7x7xf32>
    %v597 = stablehlo.reshape %v596 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v598 = stablehlo.reshape %v597 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v599 = stablehlo.broadcast_in_dim %d4n1mu, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v600 = stablehlo.subtract %v598, %v599 : tensor<32x512x7x7xf32>
    %v601 = stablehlo.broadcast_in_dim %d4n1var, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v602 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v603 = stablehlo.add %v601, %v602 : tensor<32x512x7x7xf32>
    %v604 = stablehlo.rsqrt %v603 : tensor<32x512x7x7xf32>
    %v605 = stablehlo.multiply %v600, %v604 : tensor<32x512x7x7xf32>
    %v606 = stablehlo.broadcast_in_dim %d4g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v607 = stablehlo.broadcast_in_dim %d4bt1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v608 = stablehlo.multiply %v605, %v606 : tensor<32x512x7x7xf32>
    %v609 = stablehlo.add %v608, %v607 : tensor<32x512x7x7xf32>
    %v610 = stablehlo.reshape %v609 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v611 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v612 = stablehlo.maximum %v610, %v611 : tensor<32x25088xf32>
    %v613 = stablehlo.reshape %v612 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v614 = stablehlo.convolution(%v613, %d4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v615 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v616 = stablehlo.add %v614, %v615 : tensor<32x512x7x7xf32>
    %v617 = stablehlo.reshape %v616 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v618 = stablehlo.reshape %v617 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v619 = stablehlo.broadcast_in_dim %d4n2mu, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v620 = stablehlo.subtract %v618, %v619 : tensor<32x512x7x7xf32>
    %v621 = stablehlo.broadcast_in_dim %d4n2var, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v622 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v623 = stablehlo.add %v621, %v622 : tensor<32x512x7x7xf32>
    %v624 = stablehlo.rsqrt %v623 : tensor<32x512x7x7xf32>
    %v625 = stablehlo.multiply %v620, %v624 : tensor<32x512x7x7xf32>
    %v626 = stablehlo.broadcast_in_dim %d4g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v627 = stablehlo.broadcast_in_dim %d4bt2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v628 = stablehlo.multiply %v625, %v626 : tensor<32x512x7x7xf32>
    %v629 = stablehlo.add %v628, %v627 : tensor<32x512x7x7xf32>
    %v630 = stablehlo.reshape %v629 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v631 = stablehlo.reshape %v592 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v632 = stablehlo.convolution(%v631, %d4Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<512x256x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v633 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v634 = stablehlo.add %v632, %v633 : tensor<32x512x7x7xf32>
    %v635 = stablehlo.reshape %v634 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v636 = stablehlo.reshape %v635 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v637 = stablehlo.broadcast_in_dim %d4npmu, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v638 = stablehlo.subtract %v636, %v637 : tensor<32x512x7x7xf32>
    %v639 = stablehlo.broadcast_in_dim %d4npvar, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v640 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v641 = stablehlo.add %v639, %v640 : tensor<32x512x7x7xf32>
    %v642 = stablehlo.rsqrt %v641 : tensor<32x512x7x7xf32>
    %v643 = stablehlo.multiply %v638, %v642 : tensor<32x512x7x7xf32>
    %v644 = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v645 = stablehlo.broadcast_in_dim %d4btp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v646 = stablehlo.multiply %v643, %v644 : tensor<32x512x7x7xf32>
    %v647 = stablehlo.add %v646, %v645 : tensor<32x512x7x7xf32>
    %v648 = stablehlo.reshape %v647 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v649 = stablehlo.add %v630, %v648 : tensor<32x25088xf32>
    %v650 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v651 = stablehlo.maximum %v649, %v650 : tensor<32x25088xf32>
    %v652 = stablehlo.reshape %v651 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v653 = stablehlo.convolution(%v652, %s4b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v654 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v655 = stablehlo.add %v653, %v654 : tensor<32x512x7x7xf32>
    %v656 = stablehlo.reshape %v655 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v657 = stablehlo.reshape %v656 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v658 = stablehlo.broadcast_in_dim %s4b0n1mu, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v659 = stablehlo.subtract %v657, %v658 : tensor<32x512x7x7xf32>
    %v660 = stablehlo.broadcast_in_dim %s4b0n1var, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v661 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v662 = stablehlo.add %v660, %v661 : tensor<32x512x7x7xf32>
    %v663 = stablehlo.rsqrt %v662 : tensor<32x512x7x7xf32>
    %v664 = stablehlo.multiply %v659, %v663 : tensor<32x512x7x7xf32>
    %v665 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v666 = stablehlo.broadcast_in_dim %s4b0bt1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v667 = stablehlo.multiply %v664, %v665 : tensor<32x512x7x7xf32>
    %v668 = stablehlo.add %v667, %v666 : tensor<32x512x7x7xf32>
    %v669 = stablehlo.reshape %v668 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v670 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v671 = stablehlo.maximum %v669, %v670 : tensor<32x25088xf32>
    %v672 = stablehlo.reshape %v671 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v673 = stablehlo.convolution(%v672, %s4b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v674 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v675 = stablehlo.add %v673, %v674 : tensor<32x512x7x7xf32>
    %v676 = stablehlo.reshape %v675 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v677 = stablehlo.reshape %v676 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v678 = stablehlo.broadcast_in_dim %s4b0n2mu, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v679 = stablehlo.subtract %v677, %v678 : tensor<32x512x7x7xf32>
    %v680 = stablehlo.broadcast_in_dim %s4b0n2var, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v681 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v682 = stablehlo.add %v680, %v681 : tensor<32x512x7x7xf32>
    %v683 = stablehlo.rsqrt %v682 : tensor<32x512x7x7xf32>
    %v684 = stablehlo.multiply %v679, %v683 : tensor<32x512x7x7xf32>
    %v685 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v686 = stablehlo.broadcast_in_dim %s4b0bt2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v687 = stablehlo.multiply %v684, %v685 : tensor<32x512x7x7xf32>
    %v688 = stablehlo.add %v687, %v686 : tensor<32x512x7x7xf32>
    %v689 = stablehlo.reshape %v688 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v690 = stablehlo.add %v689, %v651 : tensor<32x25088xf32>
    %v691 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v692 = stablehlo.maximum %v690, %v691 : tensor<32x25088xf32>
    %v693 = stablehlo.reshape %v692 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v694 = stablehlo.convolution(%v693, %s4b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v695 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v696 = stablehlo.add %v694, %v695 : tensor<32x512x7x7xf32>
    %v697 = stablehlo.reshape %v696 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v698 = stablehlo.reshape %v697 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v699 = stablehlo.broadcast_in_dim %s4b1n1mu, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v700 = stablehlo.subtract %v698, %v699 : tensor<32x512x7x7xf32>
    %v701 = stablehlo.broadcast_in_dim %s4b1n1var, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v702 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v703 = stablehlo.add %v701, %v702 : tensor<32x512x7x7xf32>
    %v704 = stablehlo.rsqrt %v703 : tensor<32x512x7x7xf32>
    %v705 = stablehlo.multiply %v700, %v704 : tensor<32x512x7x7xf32>
    %v706 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v707 = stablehlo.broadcast_in_dim %s4b1bt1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v708 = stablehlo.multiply %v705, %v706 : tensor<32x512x7x7xf32>
    %v709 = stablehlo.add %v708, %v707 : tensor<32x512x7x7xf32>
    %v710 = stablehlo.reshape %v709 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v711 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v712 = stablehlo.maximum %v710, %v711 : tensor<32x25088xf32>
    %v713 = stablehlo.reshape %v712 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v714 = stablehlo.convolution(%v713, %s4b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v715 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v716 = stablehlo.add %v714, %v715 : tensor<32x512x7x7xf32>
    %v717 = stablehlo.reshape %v716 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v718 = stablehlo.reshape %v717 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v719 = stablehlo.broadcast_in_dim %s4b1n2mu, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v720 = stablehlo.subtract %v718, %v719 : tensor<32x512x7x7xf32>
    %v721 = stablehlo.broadcast_in_dim %s4b1n2var, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v722 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v723 = stablehlo.add %v721, %v722 : tensor<32x512x7x7xf32>
    %v724 = stablehlo.rsqrt %v723 : tensor<32x512x7x7xf32>
    %v725 = stablehlo.multiply %v720, %v724 : tensor<32x512x7x7xf32>
    %v726 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v727 = stablehlo.broadcast_in_dim %s4b1bt2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v728 = stablehlo.multiply %v725, %v726 : tensor<32x512x7x7xf32>
    %v729 = stablehlo.add %v728, %v727 : tensor<32x512x7x7xf32>
    %v730 = stablehlo.reshape %v729 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v731 = stablehlo.add %v730, %v692 : tensor<32x25088xf32>
    %v732 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v733 = stablehlo.maximum %v731, %v732 : tensor<32x25088xf32>
    %v734 = stablehlo.reshape %v733 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v735 = stablehlo.constant dense<0.0> : tensor<f32>
    %v736 = stablehlo.reduce(%v734 init: %v735) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v737 = stablehlo.constant dense<49.0> : tensor<32x512xf32>
    %v738 = stablehlo.divide %v736, %v737 : tensor<32x512xf32>
    %v739 = stablehlo.dot_general %v738, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x512xf32>, tensor<512x10xf32>) -> tensor<32x10xf32>
    %v740 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v741 = stablehlo.add %v739, %v740 : tensor<32x10xf32>
    return %v741 : tensor<32x10xf32>
  }
}
