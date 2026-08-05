module @m {
  func.func @resnet50in_fwd_eval(%x: tensor<256x150528xf32>, %sW: tensor<64x3x7x7xf32>, %sg: tensor<64xf32>, %sbt: tensor<64xf32>, %s1b0W1: tensor<64x64x1x1xf32>, %s1b0g1: tensor<64xf32>, %s1b0bt1: tensor<64xf32>, %s1b0W2: tensor<64x64x3x3xf32>, %s1b0g2: tensor<64xf32>, %s1b0bt2: tensor<64xf32>, %s1b0W3: tensor<256x64x1x1xf32>, %s1b0g3: tensor<256xf32>, %s1b0bt3: tensor<256xf32>, %s1b0Wp: tensor<256x64x1x1xf32>, %s1b0gp: tensor<256xf32>, %s1b0btp: tensor<256xf32>, %s1b1W1: tensor<64x256x1x1xf32>, %s1b1g1: tensor<64xf32>, %s1b1bt1: tensor<64xf32>, %s1b1W2: tensor<64x64x3x3xf32>, %s1b1g2: tensor<64xf32>, %s1b1bt2: tensor<64xf32>, %s1b1W3: tensor<256x64x1x1xf32>, %s1b1g3: tensor<256xf32>, %s1b1bt3: tensor<256xf32>, %s1b2W1: tensor<64x256x1x1xf32>, %s1b2g1: tensor<64xf32>, %s1b2bt1: tensor<64xf32>, %s1b2W2: tensor<64x64x3x3xf32>, %s1b2g2: tensor<64xf32>, %s1b2bt2: tensor<64xf32>, %s1b2W3: tensor<256x64x1x1xf32>, %s1b2g3: tensor<256xf32>, %s1b2bt3: tensor<256xf32>, %s2b0W1: tensor<128x256x1x1xf32>, %s2b0g1: tensor<128xf32>, %s2b0bt1: tensor<128xf32>, %s2b0W2: tensor<128x128x3x3xf32>, %s2b0g2: tensor<128xf32>, %s2b0bt2: tensor<128xf32>, %s2b0W3: tensor<512x128x1x1xf32>, %s2b0g3: tensor<512xf32>, %s2b0bt3: tensor<512xf32>, %s2b0Wp: tensor<512x256x1x1xf32>, %s2b0gp: tensor<512xf32>, %s2b0btp: tensor<512xf32>, %s2b1W1: tensor<128x512x1x1xf32>, %s2b1g1: tensor<128xf32>, %s2b1bt1: tensor<128xf32>, %s2b1W2: tensor<128x128x3x3xf32>, %s2b1g2: tensor<128xf32>, %s2b1bt2: tensor<128xf32>, %s2b1W3: tensor<512x128x1x1xf32>, %s2b1g3: tensor<512xf32>, %s2b1bt3: tensor<512xf32>, %s2b2W1: tensor<128x512x1x1xf32>, %s2b2g1: tensor<128xf32>, %s2b2bt1: tensor<128xf32>, %s2b2W2: tensor<128x128x3x3xf32>, %s2b2g2: tensor<128xf32>, %s2b2bt2: tensor<128xf32>, %s2b2W3: tensor<512x128x1x1xf32>, %s2b2g3: tensor<512xf32>, %s2b2bt3: tensor<512xf32>, %s2b3W1: tensor<128x512x1x1xf32>, %s2b3g1: tensor<128xf32>, %s2b3bt1: tensor<128xf32>, %s2b3W2: tensor<128x128x3x3xf32>, %s2b3g2: tensor<128xf32>, %s2b3bt2: tensor<128xf32>, %s2b3W3: tensor<512x128x1x1xf32>, %s2b3g3: tensor<512xf32>, %s2b3bt3: tensor<512xf32>, %s3b0W1: tensor<256x512x1x1xf32>, %s3b0g1: tensor<256xf32>, %s3b0bt1: tensor<256xf32>, %s3b0W2: tensor<256x256x3x3xf32>, %s3b0g2: tensor<256xf32>, %s3b0bt2: tensor<256xf32>, %s3b0W3: tensor<1024x256x1x1xf32>, %s3b0g3: tensor<1024xf32>, %s3b0bt3: tensor<1024xf32>, %s3b0Wp: tensor<1024x512x1x1xf32>, %s3b0gp: tensor<1024xf32>, %s3b0btp: tensor<1024xf32>, %s3b1W1: tensor<256x1024x1x1xf32>, %s3b1g1: tensor<256xf32>, %s3b1bt1: tensor<256xf32>, %s3b1W2: tensor<256x256x3x3xf32>, %s3b1g2: tensor<256xf32>, %s3b1bt2: tensor<256xf32>, %s3b1W3: tensor<1024x256x1x1xf32>, %s3b1g3: tensor<1024xf32>, %s3b1bt3: tensor<1024xf32>, %s3b2W1: tensor<256x1024x1x1xf32>, %s3b2g1: tensor<256xf32>, %s3b2bt1: tensor<256xf32>, %s3b2W2: tensor<256x256x3x3xf32>, %s3b2g2: tensor<256xf32>, %s3b2bt2: tensor<256xf32>, %s3b2W3: tensor<1024x256x1x1xf32>, %s3b2g3: tensor<1024xf32>, %s3b2bt3: tensor<1024xf32>, %s3b3W1: tensor<256x1024x1x1xf32>, %s3b3g1: tensor<256xf32>, %s3b3bt1: tensor<256xf32>, %s3b3W2: tensor<256x256x3x3xf32>, %s3b3g2: tensor<256xf32>, %s3b3bt2: tensor<256xf32>, %s3b3W3: tensor<1024x256x1x1xf32>, %s3b3g3: tensor<1024xf32>, %s3b3bt3: tensor<1024xf32>, %s3b4W1: tensor<256x1024x1x1xf32>, %s3b4g1: tensor<256xf32>, %s3b4bt1: tensor<256xf32>, %s3b4W2: tensor<256x256x3x3xf32>, %s3b4g2: tensor<256xf32>, %s3b4bt2: tensor<256xf32>, %s3b4W3: tensor<1024x256x1x1xf32>, %s3b4g3: tensor<1024xf32>, %s3b4bt3: tensor<1024xf32>, %s3b5W1: tensor<256x1024x1x1xf32>, %s3b5g1: tensor<256xf32>, %s3b5bt1: tensor<256xf32>, %s3b5W2: tensor<256x256x3x3xf32>, %s3b5g2: tensor<256xf32>, %s3b5bt2: tensor<256xf32>, %s3b5W3: tensor<1024x256x1x1xf32>, %s3b5g3: tensor<1024xf32>, %s3b5bt3: tensor<1024xf32>, %s4b0W1: tensor<512x1024x1x1xf32>, %s4b0g1: tensor<512xf32>, %s4b0bt1: tensor<512xf32>, %s4b0W2: tensor<512x512x3x3xf32>, %s4b0g2: tensor<512xf32>, %s4b0bt2: tensor<512xf32>, %s4b0W3: tensor<2048x512x1x1xf32>, %s4b0g3: tensor<2048xf32>, %s4b0bt3: tensor<2048xf32>, %s4b0Wp: tensor<2048x1024x1x1xf32>, %s4b0gp: tensor<2048xf32>, %s4b0btp: tensor<2048xf32>, %s4b1W1: tensor<512x2048x1x1xf32>, %s4b1g1: tensor<512xf32>, %s4b1bt1: tensor<512xf32>, %s4b1W2: tensor<512x512x3x3xf32>, %s4b1g2: tensor<512xf32>, %s4b1bt2: tensor<512xf32>, %s4b1W3: tensor<2048x512x1x1xf32>, %s4b1g3: tensor<2048xf32>, %s4b1bt3: tensor<2048xf32>, %s4b2W1: tensor<512x2048x1x1xf32>, %s4b2g1: tensor<512xf32>, %s4b2bt1: tensor<512xf32>, %s4b2W2: tensor<512x512x3x3xf32>, %s4b2g2: tensor<512xf32>, %s4b2bt2: tensor<512xf32>, %s4b2W3: tensor<2048x512x1x1xf32>, %s4b2g3: tensor<2048xf32>, %s4b2bt3: tensor<2048xf32>, %Wd: tensor<2048x1000xf32>, %bd: tensor<1000xf32>, %stnmu: tensor<64xf32>, %stnvar: tensor<64xf32>, %s1b0n1mu: tensor<64xf32>, %s1b0n1var: tensor<64xf32>, %s1b0n2mu: tensor<64xf32>, %s1b0n2var: tensor<64xf32>, %s1b0n3mu: tensor<256xf32>, %s1b0n3var: tensor<256xf32>, %s1b0npmu: tensor<256xf32>, %s1b0npvar: tensor<256xf32>, %s1b1n1mu: tensor<64xf32>, %s1b1n1var: tensor<64xf32>, %s1b1n2mu: tensor<64xf32>, %s1b1n2var: tensor<64xf32>, %s1b1n3mu: tensor<256xf32>, %s1b1n3var: tensor<256xf32>, %s1b2n1mu: tensor<64xf32>, %s1b2n1var: tensor<64xf32>, %s1b2n2mu: tensor<64xf32>, %s1b2n2var: tensor<64xf32>, %s1b2n3mu: tensor<256xf32>, %s1b2n3var: tensor<256xf32>, %s2b0n1mu: tensor<128xf32>, %s2b0n1var: tensor<128xf32>, %s2b0n2mu: tensor<128xf32>, %s2b0n2var: tensor<128xf32>, %s2b0n3mu: tensor<512xf32>, %s2b0n3var: tensor<512xf32>, %s2b0npmu: tensor<512xf32>, %s2b0npvar: tensor<512xf32>, %s2b1n1mu: tensor<128xf32>, %s2b1n1var: tensor<128xf32>, %s2b1n2mu: tensor<128xf32>, %s2b1n2var: tensor<128xf32>, %s2b1n3mu: tensor<512xf32>, %s2b1n3var: tensor<512xf32>, %s2b2n1mu: tensor<128xf32>, %s2b2n1var: tensor<128xf32>, %s2b2n2mu: tensor<128xf32>, %s2b2n2var: tensor<128xf32>, %s2b2n3mu: tensor<512xf32>, %s2b2n3var: tensor<512xf32>, %s2b3n1mu: tensor<128xf32>, %s2b3n1var: tensor<128xf32>, %s2b3n2mu: tensor<128xf32>, %s2b3n2var: tensor<128xf32>, %s2b3n3mu: tensor<512xf32>, %s2b3n3var: tensor<512xf32>, %s3b0n1mu: tensor<256xf32>, %s3b0n1var: tensor<256xf32>, %s3b0n2mu: tensor<256xf32>, %s3b0n2var: tensor<256xf32>, %s3b0n3mu: tensor<1024xf32>, %s3b0n3var: tensor<1024xf32>, %s3b0npmu: tensor<1024xf32>, %s3b0npvar: tensor<1024xf32>, %s3b1n1mu: tensor<256xf32>, %s3b1n1var: tensor<256xf32>, %s3b1n2mu: tensor<256xf32>, %s3b1n2var: tensor<256xf32>, %s3b1n3mu: tensor<1024xf32>, %s3b1n3var: tensor<1024xf32>, %s3b2n1mu: tensor<256xf32>, %s3b2n1var: tensor<256xf32>, %s3b2n2mu: tensor<256xf32>, %s3b2n2var: tensor<256xf32>, %s3b2n3mu: tensor<1024xf32>, %s3b2n3var: tensor<1024xf32>, %s3b3n1mu: tensor<256xf32>, %s3b3n1var: tensor<256xf32>, %s3b3n2mu: tensor<256xf32>, %s3b3n2var: tensor<256xf32>, %s3b3n3mu: tensor<1024xf32>, %s3b3n3var: tensor<1024xf32>, %s3b4n1mu: tensor<256xf32>, %s3b4n1var: tensor<256xf32>, %s3b4n2mu: tensor<256xf32>, %s3b4n2var: tensor<256xf32>, %s3b4n3mu: tensor<1024xf32>, %s3b4n3var: tensor<1024xf32>, %s3b5n1mu: tensor<256xf32>, %s3b5n1var: tensor<256xf32>, %s3b5n2mu: tensor<256xf32>, %s3b5n2var: tensor<256xf32>, %s3b5n3mu: tensor<1024xf32>, %s3b5n3var: tensor<1024xf32>, %s4b0n1mu: tensor<512xf32>, %s4b0n1var: tensor<512xf32>, %s4b0n2mu: tensor<512xf32>, %s4b0n2var: tensor<512xf32>, %s4b0n3mu: tensor<2048xf32>, %s4b0n3var: tensor<2048xf32>, %s4b0npmu: tensor<2048xf32>, %s4b0npvar: tensor<2048xf32>, %s4b1n1mu: tensor<512xf32>, %s4b1n1var: tensor<512xf32>, %s4b1n2mu: tensor<512xf32>, %s4b1n2var: tensor<512xf32>, %s4b1n3mu: tensor<2048xf32>, %s4b1n3var: tensor<2048xf32>, %s4b2n1mu: tensor<512xf32>, %s4b2n1var: tensor<512xf32>, %s4b2n2mu: tensor<512xf32>, %s4b2n2var: tensor<512xf32>, %s4b2n3mu: tensor<2048xf32>, %s4b2n3var: tensor<2048xf32>) -> tensor<256x1000xf32> {
    // ── ResNet-50 eval forward (running-stats BN): every line is pretty(verified AST node) ──
    // §2l step B: the conv biases are gone from the signature (BN removes them; He et al.'s
    // `.convBn` has none). The proven conv ops still take a bias operand, so it is bound to a
    // zero constant here — same op, `bias = 0`, and `x + 0.0` is exact.
    %zb64 = stablehlo.constant dense<0.0> : tensor<64xf32>
    %zb128 = stablehlo.constant dense<0.0> : tensor<128xf32>
    %zb256 = stablehlo.constant dense<0.0> : tensor<256xf32>
    %zb512 = stablehlo.constant dense<0.0> : tensor<512xf32>
    %zb1024 = stablehlo.constant dense<0.0> : tensor<1024xf32>
    %zb2048 = stablehlo.constant dense<0.0> : tensor<2048xf32>
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
    %v18 = stablehlo.constant dense<0.0> : tensor<256x802816xf32>
    %v19 = stablehlo.maximum %v17, %v18 : tensor<256x802816xf32>
    %v20 = stablehlo.reshape %v19 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v21 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v22 = "stablehlo.reduce_window"(%v20, %v21) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>} : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<256x64x56x56xf32>
    %v23 = stablehlo.reshape %v22 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v24 = stablehlo.reshape %v23 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v25 = stablehlo.convolution(%v24, %s1b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x1x1xf32>) -> tensor<256x64x56x56xf32>
    %v26 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v27 = stablehlo.add %v25, %v26 : tensor<256x64x56x56xf32>
    %v28 = stablehlo.reshape %v27 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v29 = stablehlo.reshape %v28 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v30 = stablehlo.broadcast_in_dim %s1b0n1mu, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v31 = stablehlo.subtract %v29, %v30 : tensor<256x64x56x56xf32>
    %v32 = stablehlo.broadcast_in_dim %s1b0n1var, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v33 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v34 = stablehlo.add %v32, %v33 : tensor<256x64x56x56xf32>
    %v35 = stablehlo.rsqrt %v34 : tensor<256x64x56x56xf32>
    %v36 = stablehlo.multiply %v31, %v35 : tensor<256x64x56x56xf32>
    %v37 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v38 = stablehlo.broadcast_in_dim %s1b0bt1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v39 = stablehlo.multiply %v36, %v37 : tensor<256x64x56x56xf32>
    %v40 = stablehlo.add %v39, %v38 : tensor<256x64x56x56xf32>
    %v41 = stablehlo.reshape %v40 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v42 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v43 = stablehlo.maximum %v41, %v42 : tensor<256x200704xf32>
    %v44 = stablehlo.reshape %v43 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v45 = stablehlo.convolution(%v44, %s1b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v46 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v47 = stablehlo.add %v45, %v46 : tensor<256x64x56x56xf32>
    %v48 = stablehlo.reshape %v47 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v49 = stablehlo.reshape %v48 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v50 = stablehlo.broadcast_in_dim %s1b0n2mu, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v51 = stablehlo.subtract %v49, %v50 : tensor<256x64x56x56xf32>
    %v52 = stablehlo.broadcast_in_dim %s1b0n2var, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v53 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v54 = stablehlo.add %v52, %v53 : tensor<256x64x56x56xf32>
    %v55 = stablehlo.rsqrt %v54 : tensor<256x64x56x56xf32>
    %v56 = stablehlo.multiply %v51, %v55 : tensor<256x64x56x56xf32>
    %v57 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v58 = stablehlo.broadcast_in_dim %s1b0bt2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v59 = stablehlo.multiply %v56, %v57 : tensor<256x64x56x56xf32>
    %v60 = stablehlo.add %v59, %v58 : tensor<256x64x56x56xf32>
    %v61 = stablehlo.reshape %v60 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v62 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v63 = stablehlo.maximum %v61, %v62 : tensor<256x200704xf32>
    %v64 = stablehlo.reshape %v63 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v65 = stablehlo.convolution(%v64, %s1b0W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<256x256x56x56xf32>
    %v66 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v67 = stablehlo.add %v65, %v66 : tensor<256x256x56x56xf32>
    %v68 = stablehlo.reshape %v67 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v69 = stablehlo.reshape %v68 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v70 = stablehlo.broadcast_in_dim %s1b0n3mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v71 = stablehlo.subtract %v69, %v70 : tensor<256x256x56x56xf32>
    %v72 = stablehlo.broadcast_in_dim %s1b0n3var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v73 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v74 = stablehlo.add %v72, %v73 : tensor<256x256x56x56xf32>
    %v75 = stablehlo.rsqrt %v74 : tensor<256x256x56x56xf32>
    %v76 = stablehlo.multiply %v71, %v75 : tensor<256x256x56x56xf32>
    %v77 = stablehlo.broadcast_in_dim %s1b0g3, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v78 = stablehlo.broadcast_in_dim %s1b0bt3, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v79 = stablehlo.multiply %v76, %v77 : tensor<256x256x56x56xf32>
    %v80 = stablehlo.add %v79, %v78 : tensor<256x256x56x56xf32>
    %v81 = stablehlo.reshape %v80 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v82 = stablehlo.reshape %v23 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v83 = stablehlo.convolution(%v82, %s1b0Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<256x256x56x56xf32>
    %v84 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v85 = stablehlo.add %v83, %v84 : tensor<256x256x56x56xf32>
    %v86 = stablehlo.reshape %v85 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v87 = stablehlo.reshape %v86 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v88 = stablehlo.broadcast_in_dim %s1b0npmu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v89 = stablehlo.subtract %v87, %v88 : tensor<256x256x56x56xf32>
    %v90 = stablehlo.broadcast_in_dim %s1b0npvar, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v91 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v92 = stablehlo.add %v90, %v91 : tensor<256x256x56x56xf32>
    %v93 = stablehlo.rsqrt %v92 : tensor<256x256x56x56xf32>
    %v94 = stablehlo.multiply %v89, %v93 : tensor<256x256x56x56xf32>
    %v95 = stablehlo.broadcast_in_dim %s1b0gp, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v96 = stablehlo.broadcast_in_dim %s1b0btp, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v97 = stablehlo.multiply %v94, %v95 : tensor<256x256x56x56xf32>
    %v98 = stablehlo.add %v97, %v96 : tensor<256x256x56x56xf32>
    %v99 = stablehlo.reshape %v98 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v100 = stablehlo.add %v81, %v99 : tensor<256x802816xf32>
    %v101 = stablehlo.constant dense<0.0> : tensor<256x802816xf32>
    %v102 = stablehlo.maximum %v100, %v101 : tensor<256x802816xf32>
    %v103 = stablehlo.reshape %v102 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v104 = stablehlo.convolution(%v103, %s1b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tensor<256x64x56x56xf32>
    %v105 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v106 = stablehlo.add %v104, %v105 : tensor<256x64x56x56xf32>
    %v107 = stablehlo.reshape %v106 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v108 = stablehlo.reshape %v107 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v109 = stablehlo.broadcast_in_dim %s1b1n1mu, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v110 = stablehlo.subtract %v108, %v109 : tensor<256x64x56x56xf32>
    %v111 = stablehlo.broadcast_in_dim %s1b1n1var, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v112 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v113 = stablehlo.add %v111, %v112 : tensor<256x64x56x56xf32>
    %v114 = stablehlo.rsqrt %v113 : tensor<256x64x56x56xf32>
    %v115 = stablehlo.multiply %v110, %v114 : tensor<256x64x56x56xf32>
    %v116 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v117 = stablehlo.broadcast_in_dim %s1b1bt1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v118 = stablehlo.multiply %v115, %v116 : tensor<256x64x56x56xf32>
    %v119 = stablehlo.add %v118, %v117 : tensor<256x64x56x56xf32>
    %v120 = stablehlo.reshape %v119 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v121 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v122 = stablehlo.maximum %v120, %v121 : tensor<256x200704xf32>
    %v123 = stablehlo.reshape %v122 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v124 = stablehlo.convolution(%v123, %s1b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v125 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v126 = stablehlo.add %v124, %v125 : tensor<256x64x56x56xf32>
    %v127 = stablehlo.reshape %v126 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v128 = stablehlo.reshape %v127 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v129 = stablehlo.broadcast_in_dim %s1b1n2mu, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v130 = stablehlo.subtract %v128, %v129 : tensor<256x64x56x56xf32>
    %v131 = stablehlo.broadcast_in_dim %s1b1n2var, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v132 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v133 = stablehlo.add %v131, %v132 : tensor<256x64x56x56xf32>
    %v134 = stablehlo.rsqrt %v133 : tensor<256x64x56x56xf32>
    %v135 = stablehlo.multiply %v130, %v134 : tensor<256x64x56x56xf32>
    %v136 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v137 = stablehlo.broadcast_in_dim %s1b1bt2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v138 = stablehlo.multiply %v135, %v136 : tensor<256x64x56x56xf32>
    %v139 = stablehlo.add %v138, %v137 : tensor<256x64x56x56xf32>
    %v140 = stablehlo.reshape %v139 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v141 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v142 = stablehlo.maximum %v140, %v141 : tensor<256x200704xf32>
    %v143 = stablehlo.reshape %v142 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v144 = stablehlo.convolution(%v143, %s1b1W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<256x256x56x56xf32>
    %v145 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v146 = stablehlo.add %v144, %v145 : tensor<256x256x56x56xf32>
    %v147 = stablehlo.reshape %v146 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v148 = stablehlo.reshape %v147 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v149 = stablehlo.broadcast_in_dim %s1b1n3mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v150 = stablehlo.subtract %v148, %v149 : tensor<256x256x56x56xf32>
    %v151 = stablehlo.broadcast_in_dim %s1b1n3var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v152 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v153 = stablehlo.add %v151, %v152 : tensor<256x256x56x56xf32>
    %v154 = stablehlo.rsqrt %v153 : tensor<256x256x56x56xf32>
    %v155 = stablehlo.multiply %v150, %v154 : tensor<256x256x56x56xf32>
    %v156 = stablehlo.broadcast_in_dim %s1b1g3, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v157 = stablehlo.broadcast_in_dim %s1b1bt3, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v158 = stablehlo.multiply %v155, %v156 : tensor<256x256x56x56xf32>
    %v159 = stablehlo.add %v158, %v157 : tensor<256x256x56x56xf32>
    %v160 = stablehlo.reshape %v159 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v161 = stablehlo.add %v160, %v102 : tensor<256x802816xf32>
    %v162 = stablehlo.constant dense<0.0> : tensor<256x802816xf32>
    %v163 = stablehlo.maximum %v161, %v162 : tensor<256x802816xf32>
    %v164 = stablehlo.reshape %v163 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v165 = stablehlo.convolution(%v164, %s1b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tensor<256x64x56x56xf32>
    %v166 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v167 = stablehlo.add %v165, %v166 : tensor<256x64x56x56xf32>
    %v168 = stablehlo.reshape %v167 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v169 = stablehlo.reshape %v168 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v170 = stablehlo.broadcast_in_dim %s1b2n1mu, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v171 = stablehlo.subtract %v169, %v170 : tensor<256x64x56x56xf32>
    %v172 = stablehlo.broadcast_in_dim %s1b2n1var, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v173 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v174 = stablehlo.add %v172, %v173 : tensor<256x64x56x56xf32>
    %v175 = stablehlo.rsqrt %v174 : tensor<256x64x56x56xf32>
    %v176 = stablehlo.multiply %v171, %v175 : tensor<256x64x56x56xf32>
    %v177 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v178 = stablehlo.broadcast_in_dim %s1b2bt1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v179 = stablehlo.multiply %v176, %v177 : tensor<256x64x56x56xf32>
    %v180 = stablehlo.add %v179, %v178 : tensor<256x64x56x56xf32>
    %v181 = stablehlo.reshape %v180 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v182 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v183 = stablehlo.maximum %v181, %v182 : tensor<256x200704xf32>
    %v184 = stablehlo.reshape %v183 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v185 = stablehlo.convolution(%v184, %s1b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v186 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v187 = stablehlo.add %v185, %v186 : tensor<256x64x56x56xf32>
    %v188 = stablehlo.reshape %v187 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v189 = stablehlo.reshape %v188 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v190 = stablehlo.broadcast_in_dim %s1b2n2mu, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v191 = stablehlo.subtract %v189, %v190 : tensor<256x64x56x56xf32>
    %v192 = stablehlo.broadcast_in_dim %s1b2n2var, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v193 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v194 = stablehlo.add %v192, %v193 : tensor<256x64x56x56xf32>
    %v195 = stablehlo.rsqrt %v194 : tensor<256x64x56x56xf32>
    %v196 = stablehlo.multiply %v191, %v195 : tensor<256x64x56x56xf32>
    %v197 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v198 = stablehlo.broadcast_in_dim %s1b2bt2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v199 = stablehlo.multiply %v196, %v197 : tensor<256x64x56x56xf32>
    %v200 = stablehlo.add %v199, %v198 : tensor<256x64x56x56xf32>
    %v201 = stablehlo.reshape %v200 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v202 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v203 = stablehlo.maximum %v201, %v202 : tensor<256x200704xf32>
    %v204 = stablehlo.reshape %v203 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v205 = stablehlo.convolution(%v204, %s1b2W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<256x256x56x56xf32>
    %v206 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v207 = stablehlo.add %v205, %v206 : tensor<256x256x56x56xf32>
    %v208 = stablehlo.reshape %v207 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v209 = stablehlo.reshape %v208 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v210 = stablehlo.broadcast_in_dim %s1b2n3mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v211 = stablehlo.subtract %v209, %v210 : tensor<256x256x56x56xf32>
    %v212 = stablehlo.broadcast_in_dim %s1b2n3var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v213 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v214 = stablehlo.add %v212, %v213 : tensor<256x256x56x56xf32>
    %v215 = stablehlo.rsqrt %v214 : tensor<256x256x56x56xf32>
    %v216 = stablehlo.multiply %v211, %v215 : tensor<256x256x56x56xf32>
    %v217 = stablehlo.broadcast_in_dim %s1b2g3, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v218 = stablehlo.broadcast_in_dim %s1b2bt3, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v219 = stablehlo.multiply %v216, %v217 : tensor<256x256x56x56xf32>
    %v220 = stablehlo.add %v219, %v218 : tensor<256x256x56x56xf32>
    %v221 = stablehlo.reshape %v220 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v222 = stablehlo.add %v221, %v163 : tensor<256x802816xf32>
    %v223 = stablehlo.constant dense<0.0> : tensor<256x802816xf32>
    %v224 = stablehlo.maximum %v222, %v223 : tensor<256x802816xf32>
    %v225 = stablehlo.reshape %v224 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v226 = stablehlo.convolution(%v225, %s2b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<128x256x1x1xf32>) -> tensor<256x128x56x56xf32>
    %v227 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v228 = stablehlo.add %v226, %v227 : tensor<256x128x56x56xf32>
    %v229 = stablehlo.reshape %v228 : (tensor<256x128x56x56xf32>) -> tensor<256x401408xf32>
    %v230 = stablehlo.reshape %v229 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v231 = stablehlo.broadcast_in_dim %s2b0n1mu, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v232 = stablehlo.subtract %v230, %v231 : tensor<256x128x56x56xf32>
    %v233 = stablehlo.broadcast_in_dim %s2b0n1var, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v234 = stablehlo.constant dense<1.0e-05> : tensor<256x128x56x56xf32>
    %v235 = stablehlo.add %v233, %v234 : tensor<256x128x56x56xf32>
    %v236 = stablehlo.rsqrt %v235 : tensor<256x128x56x56xf32>
    %v237 = stablehlo.multiply %v232, %v236 : tensor<256x128x56x56xf32>
    %v238 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v239 = stablehlo.broadcast_in_dim %s2b0bt1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v240 = stablehlo.multiply %v237, %v238 : tensor<256x128x56x56xf32>
    %v241 = stablehlo.add %v240, %v239 : tensor<256x128x56x56xf32>
    %v242 = stablehlo.reshape %v241 : (tensor<256x128x56x56xf32>) -> tensor<256x401408xf32>
    %v243 = stablehlo.constant dense<0.0> : tensor<256x401408xf32>
    %v244 = stablehlo.maximum %v242, %v243 : tensor<256x401408xf32>
    %v245 = stablehlo.reshape %v244 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v246 = stablehlo.convolution(%v245, %s2b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x56x56xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v247 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v248 = stablehlo.add %v246, %v247 : tensor<256x128x28x28xf32>
    %v249 = stablehlo.reshape %v248 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v250 = stablehlo.reshape %v249 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v251 = stablehlo.broadcast_in_dim %s2b0n2mu, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v252 = stablehlo.subtract %v250, %v251 : tensor<256x128x28x28xf32>
    %v253 = stablehlo.broadcast_in_dim %s2b0n2var, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v254 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v255 = stablehlo.add %v253, %v254 : tensor<256x128x28x28xf32>
    %v256 = stablehlo.rsqrt %v255 : tensor<256x128x28x28xf32>
    %v257 = stablehlo.multiply %v252, %v256 : tensor<256x128x28x28xf32>
    %v258 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v259 = stablehlo.broadcast_in_dim %s2b0bt2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v260 = stablehlo.multiply %v257, %v258 : tensor<256x128x28x28xf32>
    %v261 = stablehlo.add %v260, %v259 : tensor<256x128x28x28xf32>
    %v262 = stablehlo.reshape %v261 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v263 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v264 = stablehlo.maximum %v262, %v263 : tensor<256x100352xf32>
    %v265 = stablehlo.reshape %v264 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v266 = stablehlo.convolution(%v265, %s2b0W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<256x512x28x28xf32>
    %v267 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v268 = stablehlo.add %v266, %v267 : tensor<256x512x28x28xf32>
    %v269 = stablehlo.reshape %v268 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v270 = stablehlo.reshape %v269 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v271 = stablehlo.broadcast_in_dim %s2b0n3mu, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v272 = stablehlo.subtract %v270, %v271 : tensor<256x512x28x28xf32>
    %v273 = stablehlo.broadcast_in_dim %s2b0n3var, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v274 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v275 = stablehlo.add %v273, %v274 : tensor<256x512x28x28xf32>
    %v276 = stablehlo.rsqrt %v275 : tensor<256x512x28x28xf32>
    %v277 = stablehlo.multiply %v272, %v276 : tensor<256x512x28x28xf32>
    %v278 = stablehlo.broadcast_in_dim %s2b0g3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v279 = stablehlo.broadcast_in_dim %s2b0bt3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v280 = stablehlo.multiply %v277, %v278 : tensor<256x512x28x28xf32>
    %v281 = stablehlo.add %v280, %v279 : tensor<256x512x28x28xf32>
    %v282 = stablehlo.reshape %v281 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v283 = stablehlo.reshape %v224 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v284 = stablehlo.convolution(%v283, %s2b0Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<512x256x1x1xf32>) -> tensor<256x512x28x28xf32>
    %v285 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v286 = stablehlo.add %v284, %v285 : tensor<256x512x28x28xf32>
    %v287 = stablehlo.reshape %v286 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v288 = stablehlo.reshape %v287 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v289 = stablehlo.broadcast_in_dim %s2b0npmu, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v290 = stablehlo.subtract %v288, %v289 : tensor<256x512x28x28xf32>
    %v291 = stablehlo.broadcast_in_dim %s2b0npvar, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v292 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v293 = stablehlo.add %v291, %v292 : tensor<256x512x28x28xf32>
    %v294 = stablehlo.rsqrt %v293 : tensor<256x512x28x28xf32>
    %v295 = stablehlo.multiply %v290, %v294 : tensor<256x512x28x28xf32>
    %v296 = stablehlo.broadcast_in_dim %s2b0gp, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v297 = stablehlo.broadcast_in_dim %s2b0btp, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v298 = stablehlo.multiply %v295, %v296 : tensor<256x512x28x28xf32>
    %v299 = stablehlo.add %v298, %v297 : tensor<256x512x28x28xf32>
    %v300 = stablehlo.reshape %v299 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v301 = stablehlo.add %v282, %v300 : tensor<256x401408xf32>
    %v302 = stablehlo.constant dense<0.0> : tensor<256x401408xf32>
    %v303 = stablehlo.maximum %v301, %v302 : tensor<256x401408xf32>
    %v304 = stablehlo.reshape %v303 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v305 = stablehlo.convolution(%v304, %s2b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<256x128x28x28xf32>
    %v306 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v307 = stablehlo.add %v305, %v306 : tensor<256x128x28x28xf32>
    %v308 = stablehlo.reshape %v307 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v309 = stablehlo.reshape %v308 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v310 = stablehlo.broadcast_in_dim %s2b1n1mu, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v311 = stablehlo.subtract %v309, %v310 : tensor<256x128x28x28xf32>
    %v312 = stablehlo.broadcast_in_dim %s2b1n1var, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v313 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v314 = stablehlo.add %v312, %v313 : tensor<256x128x28x28xf32>
    %v315 = stablehlo.rsqrt %v314 : tensor<256x128x28x28xf32>
    %v316 = stablehlo.multiply %v311, %v315 : tensor<256x128x28x28xf32>
    %v317 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v318 = stablehlo.broadcast_in_dim %s2b1bt1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v319 = stablehlo.multiply %v316, %v317 : tensor<256x128x28x28xf32>
    %v320 = stablehlo.add %v319, %v318 : tensor<256x128x28x28xf32>
    %v321 = stablehlo.reshape %v320 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v322 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v323 = stablehlo.maximum %v321, %v322 : tensor<256x100352xf32>
    %v324 = stablehlo.reshape %v323 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v325 = stablehlo.convolution(%v324, %s2b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v326 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v327 = stablehlo.add %v325, %v326 : tensor<256x128x28x28xf32>
    %v328 = stablehlo.reshape %v327 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v329 = stablehlo.reshape %v328 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v330 = stablehlo.broadcast_in_dim %s2b1n2mu, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v331 = stablehlo.subtract %v329, %v330 : tensor<256x128x28x28xf32>
    %v332 = stablehlo.broadcast_in_dim %s2b1n2var, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v333 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v334 = stablehlo.add %v332, %v333 : tensor<256x128x28x28xf32>
    %v335 = stablehlo.rsqrt %v334 : tensor<256x128x28x28xf32>
    %v336 = stablehlo.multiply %v331, %v335 : tensor<256x128x28x28xf32>
    %v337 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v338 = stablehlo.broadcast_in_dim %s2b1bt2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v339 = stablehlo.multiply %v336, %v337 : tensor<256x128x28x28xf32>
    %v340 = stablehlo.add %v339, %v338 : tensor<256x128x28x28xf32>
    %v341 = stablehlo.reshape %v340 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v342 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v343 = stablehlo.maximum %v341, %v342 : tensor<256x100352xf32>
    %v344 = stablehlo.reshape %v343 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v345 = stablehlo.convolution(%v344, %s2b1W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<256x512x28x28xf32>
    %v346 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v347 = stablehlo.add %v345, %v346 : tensor<256x512x28x28xf32>
    %v348 = stablehlo.reshape %v347 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v349 = stablehlo.reshape %v348 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v350 = stablehlo.broadcast_in_dim %s2b1n3mu, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v351 = stablehlo.subtract %v349, %v350 : tensor<256x512x28x28xf32>
    %v352 = stablehlo.broadcast_in_dim %s2b1n3var, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v353 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v354 = stablehlo.add %v352, %v353 : tensor<256x512x28x28xf32>
    %v355 = stablehlo.rsqrt %v354 : tensor<256x512x28x28xf32>
    %v356 = stablehlo.multiply %v351, %v355 : tensor<256x512x28x28xf32>
    %v357 = stablehlo.broadcast_in_dim %s2b1g3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v358 = stablehlo.broadcast_in_dim %s2b1bt3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v359 = stablehlo.multiply %v356, %v357 : tensor<256x512x28x28xf32>
    %v360 = stablehlo.add %v359, %v358 : tensor<256x512x28x28xf32>
    %v361 = stablehlo.reshape %v360 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v362 = stablehlo.add %v361, %v303 : tensor<256x401408xf32>
    %v363 = stablehlo.constant dense<0.0> : tensor<256x401408xf32>
    %v364 = stablehlo.maximum %v362, %v363 : tensor<256x401408xf32>
    %v365 = stablehlo.reshape %v364 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v366 = stablehlo.convolution(%v365, %s2b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<256x128x28x28xf32>
    %v367 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v368 = stablehlo.add %v366, %v367 : tensor<256x128x28x28xf32>
    %v369 = stablehlo.reshape %v368 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v370 = stablehlo.reshape %v369 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v371 = stablehlo.broadcast_in_dim %s2b2n1mu, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v372 = stablehlo.subtract %v370, %v371 : tensor<256x128x28x28xf32>
    %v373 = stablehlo.broadcast_in_dim %s2b2n1var, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v374 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v375 = stablehlo.add %v373, %v374 : tensor<256x128x28x28xf32>
    %v376 = stablehlo.rsqrt %v375 : tensor<256x128x28x28xf32>
    %v377 = stablehlo.multiply %v372, %v376 : tensor<256x128x28x28xf32>
    %v378 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v379 = stablehlo.broadcast_in_dim %s2b2bt1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v380 = stablehlo.multiply %v377, %v378 : tensor<256x128x28x28xf32>
    %v381 = stablehlo.add %v380, %v379 : tensor<256x128x28x28xf32>
    %v382 = stablehlo.reshape %v381 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v383 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v384 = stablehlo.maximum %v382, %v383 : tensor<256x100352xf32>
    %v385 = stablehlo.reshape %v384 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v386 = stablehlo.convolution(%v385, %s2b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v387 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v388 = stablehlo.add %v386, %v387 : tensor<256x128x28x28xf32>
    %v389 = stablehlo.reshape %v388 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v390 = stablehlo.reshape %v389 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v391 = stablehlo.broadcast_in_dim %s2b2n2mu, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v392 = stablehlo.subtract %v390, %v391 : tensor<256x128x28x28xf32>
    %v393 = stablehlo.broadcast_in_dim %s2b2n2var, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v394 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v395 = stablehlo.add %v393, %v394 : tensor<256x128x28x28xf32>
    %v396 = stablehlo.rsqrt %v395 : tensor<256x128x28x28xf32>
    %v397 = stablehlo.multiply %v392, %v396 : tensor<256x128x28x28xf32>
    %v398 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v399 = stablehlo.broadcast_in_dim %s2b2bt2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v400 = stablehlo.multiply %v397, %v398 : tensor<256x128x28x28xf32>
    %v401 = stablehlo.add %v400, %v399 : tensor<256x128x28x28xf32>
    %v402 = stablehlo.reshape %v401 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v403 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v404 = stablehlo.maximum %v402, %v403 : tensor<256x100352xf32>
    %v405 = stablehlo.reshape %v404 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v406 = stablehlo.convolution(%v405, %s2b2W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<256x512x28x28xf32>
    %v407 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v408 = stablehlo.add %v406, %v407 : tensor<256x512x28x28xf32>
    %v409 = stablehlo.reshape %v408 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v410 = stablehlo.reshape %v409 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v411 = stablehlo.broadcast_in_dim %s2b2n3mu, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v412 = stablehlo.subtract %v410, %v411 : tensor<256x512x28x28xf32>
    %v413 = stablehlo.broadcast_in_dim %s2b2n3var, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v414 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v415 = stablehlo.add %v413, %v414 : tensor<256x512x28x28xf32>
    %v416 = stablehlo.rsqrt %v415 : tensor<256x512x28x28xf32>
    %v417 = stablehlo.multiply %v412, %v416 : tensor<256x512x28x28xf32>
    %v418 = stablehlo.broadcast_in_dim %s2b2g3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v419 = stablehlo.broadcast_in_dim %s2b2bt3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v420 = stablehlo.multiply %v417, %v418 : tensor<256x512x28x28xf32>
    %v421 = stablehlo.add %v420, %v419 : tensor<256x512x28x28xf32>
    %v422 = stablehlo.reshape %v421 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v423 = stablehlo.add %v422, %v364 : tensor<256x401408xf32>
    %v424 = stablehlo.constant dense<0.0> : tensor<256x401408xf32>
    %v425 = stablehlo.maximum %v423, %v424 : tensor<256x401408xf32>
    %v426 = stablehlo.reshape %v425 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v427 = stablehlo.convolution(%v426, %s2b3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<256x128x28x28xf32>
    %v428 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v429 = stablehlo.add %v427, %v428 : tensor<256x128x28x28xf32>
    %v430 = stablehlo.reshape %v429 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v431 = stablehlo.reshape %v430 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v432 = stablehlo.broadcast_in_dim %s2b3n1mu, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v433 = stablehlo.subtract %v431, %v432 : tensor<256x128x28x28xf32>
    %v434 = stablehlo.broadcast_in_dim %s2b3n1var, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v435 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v436 = stablehlo.add %v434, %v435 : tensor<256x128x28x28xf32>
    %v437 = stablehlo.rsqrt %v436 : tensor<256x128x28x28xf32>
    %v438 = stablehlo.multiply %v433, %v437 : tensor<256x128x28x28xf32>
    %v439 = stablehlo.broadcast_in_dim %s2b3g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v440 = stablehlo.broadcast_in_dim %s2b3bt1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v441 = stablehlo.multiply %v438, %v439 : tensor<256x128x28x28xf32>
    %v442 = stablehlo.add %v441, %v440 : tensor<256x128x28x28xf32>
    %v443 = stablehlo.reshape %v442 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v444 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v445 = stablehlo.maximum %v443, %v444 : tensor<256x100352xf32>
    %v446 = stablehlo.reshape %v445 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v447 = stablehlo.convolution(%v446, %s2b3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v448 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v449 = stablehlo.add %v447, %v448 : tensor<256x128x28x28xf32>
    %v450 = stablehlo.reshape %v449 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v451 = stablehlo.reshape %v450 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v452 = stablehlo.broadcast_in_dim %s2b3n2mu, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v453 = stablehlo.subtract %v451, %v452 : tensor<256x128x28x28xf32>
    %v454 = stablehlo.broadcast_in_dim %s2b3n2var, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v455 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v456 = stablehlo.add %v454, %v455 : tensor<256x128x28x28xf32>
    %v457 = stablehlo.rsqrt %v456 : tensor<256x128x28x28xf32>
    %v458 = stablehlo.multiply %v453, %v457 : tensor<256x128x28x28xf32>
    %v459 = stablehlo.broadcast_in_dim %s2b3g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v460 = stablehlo.broadcast_in_dim %s2b3bt2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v461 = stablehlo.multiply %v458, %v459 : tensor<256x128x28x28xf32>
    %v462 = stablehlo.add %v461, %v460 : tensor<256x128x28x28xf32>
    %v463 = stablehlo.reshape %v462 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v464 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v465 = stablehlo.maximum %v463, %v464 : tensor<256x100352xf32>
    %v466 = stablehlo.reshape %v465 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v467 = stablehlo.convolution(%v466, %s2b3W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<256x512x28x28xf32>
    %v468 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v469 = stablehlo.add %v467, %v468 : tensor<256x512x28x28xf32>
    %v470 = stablehlo.reshape %v469 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v471 = stablehlo.reshape %v470 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v472 = stablehlo.broadcast_in_dim %s2b3n3mu, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v473 = stablehlo.subtract %v471, %v472 : tensor<256x512x28x28xf32>
    %v474 = stablehlo.broadcast_in_dim %s2b3n3var, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v475 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v476 = stablehlo.add %v474, %v475 : tensor<256x512x28x28xf32>
    %v477 = stablehlo.rsqrt %v476 : tensor<256x512x28x28xf32>
    %v478 = stablehlo.multiply %v473, %v477 : tensor<256x512x28x28xf32>
    %v479 = stablehlo.broadcast_in_dim %s2b3g3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v480 = stablehlo.broadcast_in_dim %s2b3bt3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v481 = stablehlo.multiply %v478, %v479 : tensor<256x512x28x28xf32>
    %v482 = stablehlo.add %v481, %v480 : tensor<256x512x28x28xf32>
    %v483 = stablehlo.reshape %v482 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v484 = stablehlo.add %v483, %v425 : tensor<256x401408xf32>
    %v485 = stablehlo.constant dense<0.0> : tensor<256x401408xf32>
    %v486 = stablehlo.maximum %v484, %v485 : tensor<256x401408xf32>
    %v487 = stablehlo.reshape %v486 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v488 = stablehlo.convolution(%v487, %s3b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x28x28xf32>, tensor<256x512x1x1xf32>) -> tensor<256x256x28x28xf32>
    %v489 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v490 = stablehlo.add %v488, %v489 : tensor<256x256x28x28xf32>
    %v491 = stablehlo.reshape %v490 : (tensor<256x256x28x28xf32>) -> tensor<256x200704xf32>
    %v492 = stablehlo.reshape %v491 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v493 = stablehlo.broadcast_in_dim %s3b0n1mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v494 = stablehlo.subtract %v492, %v493 : tensor<256x256x28x28xf32>
    %v495 = stablehlo.broadcast_in_dim %s3b0n1var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v496 = stablehlo.constant dense<1.0e-05> : tensor<256x256x28x28xf32>
    %v497 = stablehlo.add %v495, %v496 : tensor<256x256x28x28xf32>
    %v498 = stablehlo.rsqrt %v497 : tensor<256x256x28x28xf32>
    %v499 = stablehlo.multiply %v494, %v498 : tensor<256x256x28x28xf32>
    %v500 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v501 = stablehlo.broadcast_in_dim %s3b0bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v502 = stablehlo.multiply %v499, %v500 : tensor<256x256x28x28xf32>
    %v503 = stablehlo.add %v502, %v501 : tensor<256x256x28x28xf32>
    %v504 = stablehlo.reshape %v503 : (tensor<256x256x28x28xf32>) -> tensor<256x200704xf32>
    %v505 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v506 = stablehlo.maximum %v504, %v505 : tensor<256x200704xf32>
    %v507 = stablehlo.reshape %v506 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v508 = stablehlo.convolution(%v507, %s3b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x28x28xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v509 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v510 = stablehlo.add %v508, %v509 : tensor<256x256x14x14xf32>
    %v511 = stablehlo.reshape %v510 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v512 = stablehlo.reshape %v511 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v513 = stablehlo.broadcast_in_dim %s3b0n2mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v514 = stablehlo.subtract %v512, %v513 : tensor<256x256x14x14xf32>
    %v515 = stablehlo.broadcast_in_dim %s3b0n2var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v516 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v517 = stablehlo.add %v515, %v516 : tensor<256x256x14x14xf32>
    %v518 = stablehlo.rsqrt %v517 : tensor<256x256x14x14xf32>
    %v519 = stablehlo.multiply %v514, %v518 : tensor<256x256x14x14xf32>
    %v520 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v521 = stablehlo.broadcast_in_dim %s3b0bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v522 = stablehlo.multiply %v519, %v520 : tensor<256x256x14x14xf32>
    %v523 = stablehlo.add %v522, %v521 : tensor<256x256x14x14xf32>
    %v524 = stablehlo.reshape %v523 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v525 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v526 = stablehlo.maximum %v524, %v525 : tensor<256x50176xf32>
    %v527 = stablehlo.reshape %v526 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v528 = stablehlo.convolution(%v527, %s3b0W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v529 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v530 = stablehlo.add %v528, %v529 : tensor<256x1024x14x14xf32>
    %v531 = stablehlo.reshape %v530 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v532 = stablehlo.reshape %v531 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v533 = stablehlo.broadcast_in_dim %s3b0n3mu, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v534 = stablehlo.subtract %v532, %v533 : tensor<256x1024x14x14xf32>
    %v535 = stablehlo.broadcast_in_dim %s3b0n3var, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v536 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v537 = stablehlo.add %v535, %v536 : tensor<256x1024x14x14xf32>
    %v538 = stablehlo.rsqrt %v537 : tensor<256x1024x14x14xf32>
    %v539 = stablehlo.multiply %v534, %v538 : tensor<256x1024x14x14xf32>
    %v540 = stablehlo.broadcast_in_dim %s3b0g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v541 = stablehlo.broadcast_in_dim %s3b0bt3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v542 = stablehlo.multiply %v539, %v540 : tensor<256x1024x14x14xf32>
    %v543 = stablehlo.add %v542, %v541 : tensor<256x1024x14x14xf32>
    %v544 = stablehlo.reshape %v543 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v545 = stablehlo.reshape %v486 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v546 = stablehlo.convolution(%v545, %s3b0Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x28x28xf32>, tensor<1024x512x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v547 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v548 = stablehlo.add %v546, %v547 : tensor<256x1024x14x14xf32>
    %v549 = stablehlo.reshape %v548 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v550 = stablehlo.reshape %v549 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v551 = stablehlo.broadcast_in_dim %s3b0npmu, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v552 = stablehlo.subtract %v550, %v551 : tensor<256x1024x14x14xf32>
    %v553 = stablehlo.broadcast_in_dim %s3b0npvar, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v554 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v555 = stablehlo.add %v553, %v554 : tensor<256x1024x14x14xf32>
    %v556 = stablehlo.rsqrt %v555 : tensor<256x1024x14x14xf32>
    %v557 = stablehlo.multiply %v552, %v556 : tensor<256x1024x14x14xf32>
    %v558 = stablehlo.broadcast_in_dim %s3b0gp, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v559 = stablehlo.broadcast_in_dim %s3b0btp, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v560 = stablehlo.multiply %v557, %v558 : tensor<256x1024x14x14xf32>
    %v561 = stablehlo.add %v560, %v559 : tensor<256x1024x14x14xf32>
    %v562 = stablehlo.reshape %v561 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v563 = stablehlo.add %v544, %v562 : tensor<256x200704xf32>
    %v564 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v565 = stablehlo.maximum %v563, %v564 : tensor<256x200704xf32>
    %v566 = stablehlo.reshape %v565 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v567 = stablehlo.convolution(%v566, %s3b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v568 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v569 = stablehlo.add %v567, %v568 : tensor<256x256x14x14xf32>
    %v570 = stablehlo.reshape %v569 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v571 = stablehlo.reshape %v570 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v572 = stablehlo.broadcast_in_dim %s3b1n1mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v573 = stablehlo.subtract %v571, %v572 : tensor<256x256x14x14xf32>
    %v574 = stablehlo.broadcast_in_dim %s3b1n1var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v575 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v576 = stablehlo.add %v574, %v575 : tensor<256x256x14x14xf32>
    %v577 = stablehlo.rsqrt %v576 : tensor<256x256x14x14xf32>
    %v578 = stablehlo.multiply %v573, %v577 : tensor<256x256x14x14xf32>
    %v579 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v580 = stablehlo.broadcast_in_dim %s3b1bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v581 = stablehlo.multiply %v578, %v579 : tensor<256x256x14x14xf32>
    %v582 = stablehlo.add %v581, %v580 : tensor<256x256x14x14xf32>
    %v583 = stablehlo.reshape %v582 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v584 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v585 = stablehlo.maximum %v583, %v584 : tensor<256x50176xf32>
    %v586 = stablehlo.reshape %v585 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v587 = stablehlo.convolution(%v586, %s3b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v588 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v589 = stablehlo.add %v587, %v588 : tensor<256x256x14x14xf32>
    %v590 = stablehlo.reshape %v589 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v591 = stablehlo.reshape %v590 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v592 = stablehlo.broadcast_in_dim %s3b1n2mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v593 = stablehlo.subtract %v591, %v592 : tensor<256x256x14x14xf32>
    %v594 = stablehlo.broadcast_in_dim %s3b1n2var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v595 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v596 = stablehlo.add %v594, %v595 : tensor<256x256x14x14xf32>
    %v597 = stablehlo.rsqrt %v596 : tensor<256x256x14x14xf32>
    %v598 = stablehlo.multiply %v593, %v597 : tensor<256x256x14x14xf32>
    %v599 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v600 = stablehlo.broadcast_in_dim %s3b1bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v601 = stablehlo.multiply %v598, %v599 : tensor<256x256x14x14xf32>
    %v602 = stablehlo.add %v601, %v600 : tensor<256x256x14x14xf32>
    %v603 = stablehlo.reshape %v602 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v604 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v605 = stablehlo.maximum %v603, %v604 : tensor<256x50176xf32>
    %v606 = stablehlo.reshape %v605 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v607 = stablehlo.convolution(%v606, %s3b1W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v608 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v609 = stablehlo.add %v607, %v608 : tensor<256x1024x14x14xf32>
    %v610 = stablehlo.reshape %v609 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v611 = stablehlo.reshape %v610 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v612 = stablehlo.broadcast_in_dim %s3b1n3mu, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v613 = stablehlo.subtract %v611, %v612 : tensor<256x1024x14x14xf32>
    %v614 = stablehlo.broadcast_in_dim %s3b1n3var, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v615 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v616 = stablehlo.add %v614, %v615 : tensor<256x1024x14x14xf32>
    %v617 = stablehlo.rsqrt %v616 : tensor<256x1024x14x14xf32>
    %v618 = stablehlo.multiply %v613, %v617 : tensor<256x1024x14x14xf32>
    %v619 = stablehlo.broadcast_in_dim %s3b1g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v620 = stablehlo.broadcast_in_dim %s3b1bt3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v621 = stablehlo.multiply %v618, %v619 : tensor<256x1024x14x14xf32>
    %v622 = stablehlo.add %v621, %v620 : tensor<256x1024x14x14xf32>
    %v623 = stablehlo.reshape %v622 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v624 = stablehlo.add %v623, %v565 : tensor<256x200704xf32>
    %v625 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v626 = stablehlo.maximum %v624, %v625 : tensor<256x200704xf32>
    %v627 = stablehlo.reshape %v626 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v628 = stablehlo.convolution(%v627, %s3b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v629 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v630 = stablehlo.add %v628, %v629 : tensor<256x256x14x14xf32>
    %v631 = stablehlo.reshape %v630 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v632 = stablehlo.reshape %v631 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v633 = stablehlo.broadcast_in_dim %s3b2n1mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v634 = stablehlo.subtract %v632, %v633 : tensor<256x256x14x14xf32>
    %v635 = stablehlo.broadcast_in_dim %s3b2n1var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v636 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v637 = stablehlo.add %v635, %v636 : tensor<256x256x14x14xf32>
    %v638 = stablehlo.rsqrt %v637 : tensor<256x256x14x14xf32>
    %v639 = stablehlo.multiply %v634, %v638 : tensor<256x256x14x14xf32>
    %v640 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v641 = stablehlo.broadcast_in_dim %s3b2bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v642 = stablehlo.multiply %v639, %v640 : tensor<256x256x14x14xf32>
    %v643 = stablehlo.add %v642, %v641 : tensor<256x256x14x14xf32>
    %v644 = stablehlo.reshape %v643 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v645 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v646 = stablehlo.maximum %v644, %v645 : tensor<256x50176xf32>
    %v647 = stablehlo.reshape %v646 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v648 = stablehlo.convolution(%v647, %s3b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v649 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v650 = stablehlo.add %v648, %v649 : tensor<256x256x14x14xf32>
    %v651 = stablehlo.reshape %v650 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v652 = stablehlo.reshape %v651 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v653 = stablehlo.broadcast_in_dim %s3b2n2mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v654 = stablehlo.subtract %v652, %v653 : tensor<256x256x14x14xf32>
    %v655 = stablehlo.broadcast_in_dim %s3b2n2var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v656 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v657 = stablehlo.add %v655, %v656 : tensor<256x256x14x14xf32>
    %v658 = stablehlo.rsqrt %v657 : tensor<256x256x14x14xf32>
    %v659 = stablehlo.multiply %v654, %v658 : tensor<256x256x14x14xf32>
    %v660 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v661 = stablehlo.broadcast_in_dim %s3b2bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v662 = stablehlo.multiply %v659, %v660 : tensor<256x256x14x14xf32>
    %v663 = stablehlo.add %v662, %v661 : tensor<256x256x14x14xf32>
    %v664 = stablehlo.reshape %v663 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v665 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v666 = stablehlo.maximum %v664, %v665 : tensor<256x50176xf32>
    %v667 = stablehlo.reshape %v666 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v668 = stablehlo.convolution(%v667, %s3b2W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v669 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v670 = stablehlo.add %v668, %v669 : tensor<256x1024x14x14xf32>
    %v671 = stablehlo.reshape %v670 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v672 = stablehlo.reshape %v671 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v673 = stablehlo.broadcast_in_dim %s3b2n3mu, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v674 = stablehlo.subtract %v672, %v673 : tensor<256x1024x14x14xf32>
    %v675 = stablehlo.broadcast_in_dim %s3b2n3var, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v676 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v677 = stablehlo.add %v675, %v676 : tensor<256x1024x14x14xf32>
    %v678 = stablehlo.rsqrt %v677 : tensor<256x1024x14x14xf32>
    %v679 = stablehlo.multiply %v674, %v678 : tensor<256x1024x14x14xf32>
    %v680 = stablehlo.broadcast_in_dim %s3b2g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v681 = stablehlo.broadcast_in_dim %s3b2bt3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v682 = stablehlo.multiply %v679, %v680 : tensor<256x1024x14x14xf32>
    %v683 = stablehlo.add %v682, %v681 : tensor<256x1024x14x14xf32>
    %v684 = stablehlo.reshape %v683 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v685 = stablehlo.add %v684, %v626 : tensor<256x200704xf32>
    %v686 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v687 = stablehlo.maximum %v685, %v686 : tensor<256x200704xf32>
    %v688 = stablehlo.reshape %v687 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v689 = stablehlo.convolution(%v688, %s3b3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v690 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v691 = stablehlo.add %v689, %v690 : tensor<256x256x14x14xf32>
    %v692 = stablehlo.reshape %v691 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v693 = stablehlo.reshape %v692 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v694 = stablehlo.broadcast_in_dim %s3b3n1mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v695 = stablehlo.subtract %v693, %v694 : tensor<256x256x14x14xf32>
    %v696 = stablehlo.broadcast_in_dim %s3b3n1var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v697 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v698 = stablehlo.add %v696, %v697 : tensor<256x256x14x14xf32>
    %v699 = stablehlo.rsqrt %v698 : tensor<256x256x14x14xf32>
    %v700 = stablehlo.multiply %v695, %v699 : tensor<256x256x14x14xf32>
    %v701 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v702 = stablehlo.broadcast_in_dim %s3b3bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v703 = stablehlo.multiply %v700, %v701 : tensor<256x256x14x14xf32>
    %v704 = stablehlo.add %v703, %v702 : tensor<256x256x14x14xf32>
    %v705 = stablehlo.reshape %v704 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v706 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v707 = stablehlo.maximum %v705, %v706 : tensor<256x50176xf32>
    %v708 = stablehlo.reshape %v707 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v709 = stablehlo.convolution(%v708, %s3b3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v710 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v711 = stablehlo.add %v709, %v710 : tensor<256x256x14x14xf32>
    %v712 = stablehlo.reshape %v711 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v713 = stablehlo.reshape %v712 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v714 = stablehlo.broadcast_in_dim %s3b3n2mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v715 = stablehlo.subtract %v713, %v714 : tensor<256x256x14x14xf32>
    %v716 = stablehlo.broadcast_in_dim %s3b3n2var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v717 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v718 = stablehlo.add %v716, %v717 : tensor<256x256x14x14xf32>
    %v719 = stablehlo.rsqrt %v718 : tensor<256x256x14x14xf32>
    %v720 = stablehlo.multiply %v715, %v719 : tensor<256x256x14x14xf32>
    %v721 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v722 = stablehlo.broadcast_in_dim %s3b3bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v723 = stablehlo.multiply %v720, %v721 : tensor<256x256x14x14xf32>
    %v724 = stablehlo.add %v723, %v722 : tensor<256x256x14x14xf32>
    %v725 = stablehlo.reshape %v724 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v726 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v727 = stablehlo.maximum %v725, %v726 : tensor<256x50176xf32>
    %v728 = stablehlo.reshape %v727 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v729 = stablehlo.convolution(%v728, %s3b3W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v730 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v731 = stablehlo.add %v729, %v730 : tensor<256x1024x14x14xf32>
    %v732 = stablehlo.reshape %v731 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v733 = stablehlo.reshape %v732 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v734 = stablehlo.broadcast_in_dim %s3b3n3mu, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v735 = stablehlo.subtract %v733, %v734 : tensor<256x1024x14x14xf32>
    %v736 = stablehlo.broadcast_in_dim %s3b3n3var, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v737 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v738 = stablehlo.add %v736, %v737 : tensor<256x1024x14x14xf32>
    %v739 = stablehlo.rsqrt %v738 : tensor<256x1024x14x14xf32>
    %v740 = stablehlo.multiply %v735, %v739 : tensor<256x1024x14x14xf32>
    %v741 = stablehlo.broadcast_in_dim %s3b3g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v742 = stablehlo.broadcast_in_dim %s3b3bt3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v743 = stablehlo.multiply %v740, %v741 : tensor<256x1024x14x14xf32>
    %v744 = stablehlo.add %v743, %v742 : tensor<256x1024x14x14xf32>
    %v745 = stablehlo.reshape %v744 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v746 = stablehlo.add %v745, %v687 : tensor<256x200704xf32>
    %v747 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v748 = stablehlo.maximum %v746, %v747 : tensor<256x200704xf32>
    %v749 = stablehlo.reshape %v748 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v750 = stablehlo.convolution(%v749, %s3b4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v751 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v752 = stablehlo.add %v750, %v751 : tensor<256x256x14x14xf32>
    %v753 = stablehlo.reshape %v752 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v754 = stablehlo.reshape %v753 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v755 = stablehlo.broadcast_in_dim %s3b4n1mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v756 = stablehlo.subtract %v754, %v755 : tensor<256x256x14x14xf32>
    %v757 = stablehlo.broadcast_in_dim %s3b4n1var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v758 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v759 = stablehlo.add %v757, %v758 : tensor<256x256x14x14xf32>
    %v760 = stablehlo.rsqrt %v759 : tensor<256x256x14x14xf32>
    %v761 = stablehlo.multiply %v756, %v760 : tensor<256x256x14x14xf32>
    %v762 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v763 = stablehlo.broadcast_in_dim %s3b4bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v764 = stablehlo.multiply %v761, %v762 : tensor<256x256x14x14xf32>
    %v765 = stablehlo.add %v764, %v763 : tensor<256x256x14x14xf32>
    %v766 = stablehlo.reshape %v765 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v767 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v768 = stablehlo.maximum %v766, %v767 : tensor<256x50176xf32>
    %v769 = stablehlo.reshape %v768 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v770 = stablehlo.convolution(%v769, %s3b4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v771 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v772 = stablehlo.add %v770, %v771 : tensor<256x256x14x14xf32>
    %v773 = stablehlo.reshape %v772 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v774 = stablehlo.reshape %v773 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v775 = stablehlo.broadcast_in_dim %s3b4n2mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v776 = stablehlo.subtract %v774, %v775 : tensor<256x256x14x14xf32>
    %v777 = stablehlo.broadcast_in_dim %s3b4n2var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v778 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v779 = stablehlo.add %v777, %v778 : tensor<256x256x14x14xf32>
    %v780 = stablehlo.rsqrt %v779 : tensor<256x256x14x14xf32>
    %v781 = stablehlo.multiply %v776, %v780 : tensor<256x256x14x14xf32>
    %v782 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v783 = stablehlo.broadcast_in_dim %s3b4bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v784 = stablehlo.multiply %v781, %v782 : tensor<256x256x14x14xf32>
    %v785 = stablehlo.add %v784, %v783 : tensor<256x256x14x14xf32>
    %v786 = stablehlo.reshape %v785 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v787 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v788 = stablehlo.maximum %v786, %v787 : tensor<256x50176xf32>
    %v789 = stablehlo.reshape %v788 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v790 = stablehlo.convolution(%v789, %s3b4W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v791 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v792 = stablehlo.add %v790, %v791 : tensor<256x1024x14x14xf32>
    %v793 = stablehlo.reshape %v792 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v794 = stablehlo.reshape %v793 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v795 = stablehlo.broadcast_in_dim %s3b4n3mu, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v796 = stablehlo.subtract %v794, %v795 : tensor<256x1024x14x14xf32>
    %v797 = stablehlo.broadcast_in_dim %s3b4n3var, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v798 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v799 = stablehlo.add %v797, %v798 : tensor<256x1024x14x14xf32>
    %v800 = stablehlo.rsqrt %v799 : tensor<256x1024x14x14xf32>
    %v801 = stablehlo.multiply %v796, %v800 : tensor<256x1024x14x14xf32>
    %v802 = stablehlo.broadcast_in_dim %s3b4g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v803 = stablehlo.broadcast_in_dim %s3b4bt3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v804 = stablehlo.multiply %v801, %v802 : tensor<256x1024x14x14xf32>
    %v805 = stablehlo.add %v804, %v803 : tensor<256x1024x14x14xf32>
    %v806 = stablehlo.reshape %v805 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v807 = stablehlo.add %v806, %v748 : tensor<256x200704xf32>
    %v808 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v809 = stablehlo.maximum %v807, %v808 : tensor<256x200704xf32>
    %v810 = stablehlo.reshape %v809 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v811 = stablehlo.convolution(%v810, %s3b5W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v812 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v813 = stablehlo.add %v811, %v812 : tensor<256x256x14x14xf32>
    %v814 = stablehlo.reshape %v813 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v815 = stablehlo.reshape %v814 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v816 = stablehlo.broadcast_in_dim %s3b5n1mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v817 = stablehlo.subtract %v815, %v816 : tensor<256x256x14x14xf32>
    %v818 = stablehlo.broadcast_in_dim %s3b5n1var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v819 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v820 = stablehlo.add %v818, %v819 : tensor<256x256x14x14xf32>
    %v821 = stablehlo.rsqrt %v820 : tensor<256x256x14x14xf32>
    %v822 = stablehlo.multiply %v817, %v821 : tensor<256x256x14x14xf32>
    %v823 = stablehlo.broadcast_in_dim %s3b5g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v824 = stablehlo.broadcast_in_dim %s3b5bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v825 = stablehlo.multiply %v822, %v823 : tensor<256x256x14x14xf32>
    %v826 = stablehlo.add %v825, %v824 : tensor<256x256x14x14xf32>
    %v827 = stablehlo.reshape %v826 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v828 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v829 = stablehlo.maximum %v827, %v828 : tensor<256x50176xf32>
    %v830 = stablehlo.reshape %v829 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v831 = stablehlo.convolution(%v830, %s3b5W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v832 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v833 = stablehlo.add %v831, %v832 : tensor<256x256x14x14xf32>
    %v834 = stablehlo.reshape %v833 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v835 = stablehlo.reshape %v834 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v836 = stablehlo.broadcast_in_dim %s3b5n2mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v837 = stablehlo.subtract %v835, %v836 : tensor<256x256x14x14xf32>
    %v838 = stablehlo.broadcast_in_dim %s3b5n2var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v839 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v840 = stablehlo.add %v838, %v839 : tensor<256x256x14x14xf32>
    %v841 = stablehlo.rsqrt %v840 : tensor<256x256x14x14xf32>
    %v842 = stablehlo.multiply %v837, %v841 : tensor<256x256x14x14xf32>
    %v843 = stablehlo.broadcast_in_dim %s3b5g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v844 = stablehlo.broadcast_in_dim %s3b5bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v845 = stablehlo.multiply %v842, %v843 : tensor<256x256x14x14xf32>
    %v846 = stablehlo.add %v845, %v844 : tensor<256x256x14x14xf32>
    %v847 = stablehlo.reshape %v846 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v848 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v849 = stablehlo.maximum %v847, %v848 : tensor<256x50176xf32>
    %v850 = stablehlo.reshape %v849 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v851 = stablehlo.convolution(%v850, %s3b5W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v852 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v853 = stablehlo.add %v851, %v852 : tensor<256x1024x14x14xf32>
    %v854 = stablehlo.reshape %v853 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v855 = stablehlo.reshape %v854 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v856 = stablehlo.broadcast_in_dim %s3b5n3mu, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v857 = stablehlo.subtract %v855, %v856 : tensor<256x1024x14x14xf32>
    %v858 = stablehlo.broadcast_in_dim %s3b5n3var, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v859 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v860 = stablehlo.add %v858, %v859 : tensor<256x1024x14x14xf32>
    %v861 = stablehlo.rsqrt %v860 : tensor<256x1024x14x14xf32>
    %v862 = stablehlo.multiply %v857, %v861 : tensor<256x1024x14x14xf32>
    %v863 = stablehlo.broadcast_in_dim %s3b5g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v864 = stablehlo.broadcast_in_dim %s3b5bt3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v865 = stablehlo.multiply %v862, %v863 : tensor<256x1024x14x14xf32>
    %v866 = stablehlo.add %v865, %v864 : tensor<256x1024x14x14xf32>
    %v867 = stablehlo.reshape %v866 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v868 = stablehlo.add %v867, %v809 : tensor<256x200704xf32>
    %v869 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v870 = stablehlo.maximum %v868, %v869 : tensor<256x200704xf32>
    %v871 = stablehlo.reshape %v870 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v872 = stablehlo.convolution(%v871, %s4b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<512x1024x1x1xf32>) -> tensor<256x512x14x14xf32>
    %v873 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v874 = stablehlo.add %v872, %v873 : tensor<256x512x14x14xf32>
    %v875 = stablehlo.reshape %v874 : (tensor<256x512x14x14xf32>) -> tensor<256x100352xf32>
    %v876 = stablehlo.reshape %v875 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v877 = stablehlo.broadcast_in_dim %s4b0n1mu, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v878 = stablehlo.subtract %v876, %v877 : tensor<256x512x14x14xf32>
    %v879 = stablehlo.broadcast_in_dim %s4b0n1var, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v880 = stablehlo.constant dense<1.0e-05> : tensor<256x512x14x14xf32>
    %v881 = stablehlo.add %v879, %v880 : tensor<256x512x14x14xf32>
    %v882 = stablehlo.rsqrt %v881 : tensor<256x512x14x14xf32>
    %v883 = stablehlo.multiply %v878, %v882 : tensor<256x512x14x14xf32>
    %v884 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v885 = stablehlo.broadcast_in_dim %s4b0bt1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v886 = stablehlo.multiply %v883, %v884 : tensor<256x512x14x14xf32>
    %v887 = stablehlo.add %v886, %v885 : tensor<256x512x14x14xf32>
    %v888 = stablehlo.reshape %v887 : (tensor<256x512x14x14xf32>) -> tensor<256x100352xf32>
    %v889 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v890 = stablehlo.maximum %v888, %v889 : tensor<256x100352xf32>
    %v891 = stablehlo.reshape %v890 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v892 = stablehlo.convolution(%v891, %s4b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x14x14xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v893 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v894 = stablehlo.add %v892, %v893 : tensor<256x512x7x7xf32>
    %v895 = stablehlo.reshape %v894 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v896 = stablehlo.reshape %v895 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v897 = stablehlo.broadcast_in_dim %s4b0n2mu, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v898 = stablehlo.subtract %v896, %v897 : tensor<256x512x7x7xf32>
    %v899 = stablehlo.broadcast_in_dim %s4b0n2var, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v900 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v901 = stablehlo.add %v899, %v900 : tensor<256x512x7x7xf32>
    %v902 = stablehlo.rsqrt %v901 : tensor<256x512x7x7xf32>
    %v903 = stablehlo.multiply %v898, %v902 : tensor<256x512x7x7xf32>
    %v904 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v905 = stablehlo.broadcast_in_dim %s4b0bt2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v906 = stablehlo.multiply %v903, %v904 : tensor<256x512x7x7xf32>
    %v907 = stablehlo.add %v906, %v905 : tensor<256x512x7x7xf32>
    %v908 = stablehlo.reshape %v907 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v909 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v910 = stablehlo.maximum %v908, %v909 : tensor<256x25088xf32>
    %v911 = stablehlo.reshape %v910 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v912 = stablehlo.convolution(%v911, %s4b0W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tensor<256x2048x7x7xf32>
    %v913 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v914 = stablehlo.add %v912, %v913 : tensor<256x2048x7x7xf32>
    %v915 = stablehlo.reshape %v914 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v916 = stablehlo.reshape %v915 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v917 = stablehlo.broadcast_in_dim %s4b0n3mu, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v918 = stablehlo.subtract %v916, %v917 : tensor<256x2048x7x7xf32>
    %v919 = stablehlo.broadcast_in_dim %s4b0n3var, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v920 = stablehlo.constant dense<1.0e-05> : tensor<256x2048x7x7xf32>
    %v921 = stablehlo.add %v919, %v920 : tensor<256x2048x7x7xf32>
    %v922 = stablehlo.rsqrt %v921 : tensor<256x2048x7x7xf32>
    %v923 = stablehlo.multiply %v918, %v922 : tensor<256x2048x7x7xf32>
    %v924 = stablehlo.broadcast_in_dim %s4b0g3, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v925 = stablehlo.broadcast_in_dim %s4b0bt3, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v926 = stablehlo.multiply %v923, %v924 : tensor<256x2048x7x7xf32>
    %v927 = stablehlo.add %v926, %v925 : tensor<256x2048x7x7xf32>
    %v928 = stablehlo.reshape %v927 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v929 = stablehlo.reshape %v870 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v930 = stablehlo.convolution(%v929, %s4b0Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<2048x1024x1x1xf32>) -> tensor<256x2048x7x7xf32>
    %v931 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v932 = stablehlo.add %v930, %v931 : tensor<256x2048x7x7xf32>
    %v933 = stablehlo.reshape %v932 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v934 = stablehlo.reshape %v933 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v935 = stablehlo.broadcast_in_dim %s4b0npmu, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v936 = stablehlo.subtract %v934, %v935 : tensor<256x2048x7x7xf32>
    %v937 = stablehlo.broadcast_in_dim %s4b0npvar, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v938 = stablehlo.constant dense<1.0e-05> : tensor<256x2048x7x7xf32>
    %v939 = stablehlo.add %v937, %v938 : tensor<256x2048x7x7xf32>
    %v940 = stablehlo.rsqrt %v939 : tensor<256x2048x7x7xf32>
    %v941 = stablehlo.multiply %v936, %v940 : tensor<256x2048x7x7xf32>
    %v942 = stablehlo.broadcast_in_dim %s4b0gp, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v943 = stablehlo.broadcast_in_dim %s4b0btp, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v944 = stablehlo.multiply %v941, %v942 : tensor<256x2048x7x7xf32>
    %v945 = stablehlo.add %v944, %v943 : tensor<256x2048x7x7xf32>
    %v946 = stablehlo.reshape %v945 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v947 = stablehlo.add %v928, %v946 : tensor<256x100352xf32>
    %v948 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v949 = stablehlo.maximum %v947, %v948 : tensor<256x100352xf32>
    %v950 = stablehlo.reshape %v949 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v951 = stablehlo.convolution(%v950, %s4b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x2048x7x7xf32>, tensor<512x2048x1x1xf32>) -> tensor<256x512x7x7xf32>
    %v952 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v953 = stablehlo.add %v951, %v952 : tensor<256x512x7x7xf32>
    %v954 = stablehlo.reshape %v953 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v955 = stablehlo.reshape %v954 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v956 = stablehlo.broadcast_in_dim %s4b1n1mu, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v957 = stablehlo.subtract %v955, %v956 : tensor<256x512x7x7xf32>
    %v958 = stablehlo.broadcast_in_dim %s4b1n1var, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v959 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v960 = stablehlo.add %v958, %v959 : tensor<256x512x7x7xf32>
    %v961 = stablehlo.rsqrt %v960 : tensor<256x512x7x7xf32>
    %v962 = stablehlo.multiply %v957, %v961 : tensor<256x512x7x7xf32>
    %v963 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v964 = stablehlo.broadcast_in_dim %s4b1bt1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v965 = stablehlo.multiply %v962, %v963 : tensor<256x512x7x7xf32>
    %v966 = stablehlo.add %v965, %v964 : tensor<256x512x7x7xf32>
    %v967 = stablehlo.reshape %v966 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v968 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v969 = stablehlo.maximum %v967, %v968 : tensor<256x25088xf32>
    %v970 = stablehlo.reshape %v969 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v971 = stablehlo.convolution(%v970, %s4b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v972 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v973 = stablehlo.add %v971, %v972 : tensor<256x512x7x7xf32>
    %v974 = stablehlo.reshape %v973 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v975 = stablehlo.reshape %v974 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v976 = stablehlo.broadcast_in_dim %s4b1n2mu, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v977 = stablehlo.subtract %v975, %v976 : tensor<256x512x7x7xf32>
    %v978 = stablehlo.broadcast_in_dim %s4b1n2var, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v979 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v980 = stablehlo.add %v978, %v979 : tensor<256x512x7x7xf32>
    %v981 = stablehlo.rsqrt %v980 : tensor<256x512x7x7xf32>
    %v982 = stablehlo.multiply %v977, %v981 : tensor<256x512x7x7xf32>
    %v983 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v984 = stablehlo.broadcast_in_dim %s4b1bt2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v985 = stablehlo.multiply %v982, %v983 : tensor<256x512x7x7xf32>
    %v986 = stablehlo.add %v985, %v984 : tensor<256x512x7x7xf32>
    %v987 = stablehlo.reshape %v986 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v988 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v989 = stablehlo.maximum %v987, %v988 : tensor<256x25088xf32>
    %v990 = stablehlo.reshape %v989 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v991 = stablehlo.convolution(%v990, %s4b1W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tensor<256x2048x7x7xf32>
    %v992 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v993 = stablehlo.add %v991, %v992 : tensor<256x2048x7x7xf32>
    %v994 = stablehlo.reshape %v993 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v995 = stablehlo.reshape %v994 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v996 = stablehlo.broadcast_in_dim %s4b1n3mu, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v997 = stablehlo.subtract %v995, %v996 : tensor<256x2048x7x7xf32>
    %v998 = stablehlo.broadcast_in_dim %s4b1n3var, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v999 = stablehlo.constant dense<1.0e-05> : tensor<256x2048x7x7xf32>
    %v1000 = stablehlo.add %v998, %v999 : tensor<256x2048x7x7xf32>
    %v1001 = stablehlo.rsqrt %v1000 : tensor<256x2048x7x7xf32>
    %v1002 = stablehlo.multiply %v997, %v1001 : tensor<256x2048x7x7xf32>
    %v1003 = stablehlo.broadcast_in_dim %s4b1g3, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1004 = stablehlo.broadcast_in_dim %s4b1bt3, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1005 = stablehlo.multiply %v1002, %v1003 : tensor<256x2048x7x7xf32>
    %v1006 = stablehlo.add %v1005, %v1004 : tensor<256x2048x7x7xf32>
    %v1007 = stablehlo.reshape %v1006 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1008 = stablehlo.add %v1007, %v949 : tensor<256x100352xf32>
    %v1009 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v1010 = stablehlo.maximum %v1008, %v1009 : tensor<256x100352xf32>
    %v1011 = stablehlo.reshape %v1010 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1012 = stablehlo.convolution(%v1011, %s4b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x2048x7x7xf32>, tensor<512x2048x1x1xf32>) -> tensor<256x512x7x7xf32>
    %v1013 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1014 = stablehlo.add %v1012, %v1013 : tensor<256x512x7x7xf32>
    %v1015 = stablehlo.reshape %v1014 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1016 = stablehlo.reshape %v1015 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1017 = stablehlo.broadcast_in_dim %s4b2n1mu, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1018 = stablehlo.subtract %v1016, %v1017 : tensor<256x512x7x7xf32>
    %v1019 = stablehlo.broadcast_in_dim %s4b2n1var, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1020 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1021 = stablehlo.add %v1019, %v1020 : tensor<256x512x7x7xf32>
    %v1022 = stablehlo.rsqrt %v1021 : tensor<256x512x7x7xf32>
    %v1023 = stablehlo.multiply %v1018, %v1022 : tensor<256x512x7x7xf32>
    %v1024 = stablehlo.broadcast_in_dim %s4b2g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1025 = stablehlo.broadcast_in_dim %s4b2bt1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1026 = stablehlo.multiply %v1023, %v1024 : tensor<256x512x7x7xf32>
    %v1027 = stablehlo.add %v1026, %v1025 : tensor<256x512x7x7xf32>
    %v1028 = stablehlo.reshape %v1027 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1029 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v1030 = stablehlo.maximum %v1028, %v1029 : tensor<256x25088xf32>
    %v1031 = stablehlo.reshape %v1030 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1032 = stablehlo.convolution(%v1031, %s4b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v1033 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1034 = stablehlo.add %v1032, %v1033 : tensor<256x512x7x7xf32>
    %v1035 = stablehlo.reshape %v1034 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1036 = stablehlo.reshape %v1035 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1037 = stablehlo.broadcast_in_dim %s4b2n2mu, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1038 = stablehlo.subtract %v1036, %v1037 : tensor<256x512x7x7xf32>
    %v1039 = stablehlo.broadcast_in_dim %s4b2n2var, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1040 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1041 = stablehlo.add %v1039, %v1040 : tensor<256x512x7x7xf32>
    %v1042 = stablehlo.rsqrt %v1041 : tensor<256x512x7x7xf32>
    %v1043 = stablehlo.multiply %v1038, %v1042 : tensor<256x512x7x7xf32>
    %v1044 = stablehlo.broadcast_in_dim %s4b2g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1045 = stablehlo.broadcast_in_dim %s4b2bt2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1046 = stablehlo.multiply %v1043, %v1044 : tensor<256x512x7x7xf32>
    %v1047 = stablehlo.add %v1046, %v1045 : tensor<256x512x7x7xf32>
    %v1048 = stablehlo.reshape %v1047 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1049 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v1050 = stablehlo.maximum %v1048, %v1049 : tensor<256x25088xf32>
    %v1051 = stablehlo.reshape %v1050 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1052 = stablehlo.convolution(%v1051, %s4b2W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tensor<256x2048x7x7xf32>
    %v1053 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1054 = stablehlo.add %v1052, %v1053 : tensor<256x2048x7x7xf32>
    %v1055 = stablehlo.reshape %v1054 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1056 = stablehlo.reshape %v1055 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1057 = stablehlo.broadcast_in_dim %s4b2n3mu, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1058 = stablehlo.subtract %v1056, %v1057 : tensor<256x2048x7x7xf32>
    %v1059 = stablehlo.broadcast_in_dim %s4b2n3var, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1060 = stablehlo.constant dense<1.0e-05> : tensor<256x2048x7x7xf32>
    %v1061 = stablehlo.add %v1059, %v1060 : tensor<256x2048x7x7xf32>
    %v1062 = stablehlo.rsqrt %v1061 : tensor<256x2048x7x7xf32>
    %v1063 = stablehlo.multiply %v1058, %v1062 : tensor<256x2048x7x7xf32>
    %v1064 = stablehlo.broadcast_in_dim %s4b2g3, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1065 = stablehlo.broadcast_in_dim %s4b2bt3, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1066 = stablehlo.multiply %v1063, %v1064 : tensor<256x2048x7x7xf32>
    %v1067 = stablehlo.add %v1066, %v1065 : tensor<256x2048x7x7xf32>
    %v1068 = stablehlo.reshape %v1067 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1069 = stablehlo.add %v1068, %v1010 : tensor<256x100352xf32>
    %v1070 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v1071 = stablehlo.maximum %v1069, %v1070 : tensor<256x100352xf32>
    %v1072 = stablehlo.reshape %v1071 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1073 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1074 = stablehlo.reduce(%v1072 init: %v1073) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<256x2048xf32>
    %v1075 = stablehlo.constant dense<49.0> : tensor<256x2048xf32>
    %v1076 = stablehlo.divide %v1074, %v1075 : tensor<256x2048xf32>
    %v1077 = stablehlo.dot_general %v1076, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x2048xf32>, tensor<2048x1000xf32>) -> tensor<256x1000xf32>
    %v1078 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<1000xf32>) -> tensor<256x1000xf32>
    %v1079 = stablehlo.add %v1077, %v1078 : tensor<256x1000xf32>
    return %v1079 : tensor<256x1000xf32>
  }
}
