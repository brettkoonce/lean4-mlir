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
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x1x1xf32>) -> tensor<256x64x56x56xf32>
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
    %v67 = stablehlo.constant dense<0.0> : tensor<256x64x56x56xf32>
    %v68 = stablehlo.maximum %v66, %v67 : tensor<256x64x56x56xf32>
    %v69 = stablehlo.reshape %v68 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v70 = stablehlo.reshape %v69 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v71 = stablehlo.convolution(%v70, %s1b0W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<256x256x56x56xf32>
    %v72 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v73 = stablehlo.add %v71, %v72 : tensor<256x256x56x56xf32>
    %v74 = stablehlo.reshape %v73 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v75 = stablehlo.reshape %v74 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v76 = stablehlo.broadcast_in_dim %s1b0n3mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v77 = stablehlo.subtract %v75, %v76 : tensor<256x256x56x56xf32>
    %v78 = stablehlo.broadcast_in_dim %s1b0n3var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v79 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v80 = stablehlo.add %v78, %v79 : tensor<256x256x56x56xf32>
    %v81 = stablehlo.rsqrt %v80 : tensor<256x256x56x56xf32>
    %v82 = stablehlo.multiply %v77, %v81 : tensor<256x256x56x56xf32>
    %v83 = stablehlo.broadcast_in_dim %s1b0g3, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v84 = stablehlo.broadcast_in_dim %s1b0bt3, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v85 = stablehlo.multiply %v82, %v83 : tensor<256x256x56x56xf32>
    %v86 = stablehlo.add %v85, %v84 : tensor<256x256x56x56xf32>
    %v87 = stablehlo.reshape %v86 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v88 = stablehlo.reshape %v25 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v89 = stablehlo.convolution(%v88, %s1b0Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<256x256x56x56xf32>
    %v90 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v91 = stablehlo.add %v89, %v90 : tensor<256x256x56x56xf32>
    %v92 = stablehlo.reshape %v91 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v93 = stablehlo.reshape %v92 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v94 = stablehlo.broadcast_in_dim %s1b0npmu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v95 = stablehlo.subtract %v93, %v94 : tensor<256x256x56x56xf32>
    %v96 = stablehlo.broadcast_in_dim %s1b0npvar, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v97 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v98 = stablehlo.add %v96, %v97 : tensor<256x256x56x56xf32>
    %v99 = stablehlo.rsqrt %v98 : tensor<256x256x56x56xf32>
    %v100 = stablehlo.multiply %v95, %v99 : tensor<256x256x56x56xf32>
    %v101 = stablehlo.broadcast_in_dim %s1b0gp, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v102 = stablehlo.broadcast_in_dim %s1b0btp, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v103 = stablehlo.multiply %v100, %v101 : tensor<256x256x56x56xf32>
    %v104 = stablehlo.add %v103, %v102 : tensor<256x256x56x56xf32>
    %v105 = stablehlo.reshape %v104 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v106 = stablehlo.reshape %v87 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v107 = stablehlo.reshape %v105 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v108 = stablehlo.add %v106, %v107 : tensor<256x256x56x56xf32>
    %v109 = stablehlo.reshape %v108 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v110 = stablehlo.reshape %v109 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v111 = stablehlo.constant dense<0.0> : tensor<256x256x56x56xf32>
    %v112 = stablehlo.maximum %v110, %v111 : tensor<256x256x56x56xf32>
    %v113 = stablehlo.reshape %v112 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v114 = stablehlo.reshape %v113 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v115 = stablehlo.convolution(%v114, %s1b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tensor<256x64x56x56xf32>
    %v116 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v117 = stablehlo.add %v115, %v116 : tensor<256x64x56x56xf32>
    %v118 = stablehlo.reshape %v117 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v119 = stablehlo.reshape %v118 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v120 = stablehlo.broadcast_in_dim %s1b1n1mu, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v121 = stablehlo.subtract %v119, %v120 : tensor<256x64x56x56xf32>
    %v122 = stablehlo.broadcast_in_dim %s1b1n1var, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v123 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v124 = stablehlo.add %v122, %v123 : tensor<256x64x56x56xf32>
    %v125 = stablehlo.rsqrt %v124 : tensor<256x64x56x56xf32>
    %v126 = stablehlo.multiply %v121, %v125 : tensor<256x64x56x56xf32>
    %v127 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v128 = stablehlo.broadcast_in_dim %s1b1bt1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v129 = stablehlo.multiply %v126, %v127 : tensor<256x64x56x56xf32>
    %v130 = stablehlo.add %v129, %v128 : tensor<256x64x56x56xf32>
    %v131 = stablehlo.reshape %v130 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v132 = stablehlo.reshape %v131 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v133 = stablehlo.constant dense<0.0> : tensor<256x64x56x56xf32>
    %v134 = stablehlo.maximum %v132, %v133 : tensor<256x64x56x56xf32>
    %v135 = stablehlo.reshape %v134 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v136 = stablehlo.reshape %v135 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v137 = stablehlo.convolution(%v136, %s1b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v138 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v139 = stablehlo.add %v137, %v138 : tensor<256x64x56x56xf32>
    %v140 = stablehlo.reshape %v139 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v141 = stablehlo.reshape %v140 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v142 = stablehlo.broadcast_in_dim %s1b1n2mu, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v143 = stablehlo.subtract %v141, %v142 : tensor<256x64x56x56xf32>
    %v144 = stablehlo.broadcast_in_dim %s1b1n2var, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v145 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v146 = stablehlo.add %v144, %v145 : tensor<256x64x56x56xf32>
    %v147 = stablehlo.rsqrt %v146 : tensor<256x64x56x56xf32>
    %v148 = stablehlo.multiply %v143, %v147 : tensor<256x64x56x56xf32>
    %v149 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v150 = stablehlo.broadcast_in_dim %s1b1bt2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v151 = stablehlo.multiply %v148, %v149 : tensor<256x64x56x56xf32>
    %v152 = stablehlo.add %v151, %v150 : tensor<256x64x56x56xf32>
    %v153 = stablehlo.reshape %v152 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v154 = stablehlo.reshape %v153 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v155 = stablehlo.constant dense<0.0> : tensor<256x64x56x56xf32>
    %v156 = stablehlo.maximum %v154, %v155 : tensor<256x64x56x56xf32>
    %v157 = stablehlo.reshape %v156 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v158 = stablehlo.reshape %v157 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v159 = stablehlo.convolution(%v158, %s1b1W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<256x256x56x56xf32>
    %v160 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v161 = stablehlo.add %v159, %v160 : tensor<256x256x56x56xf32>
    %v162 = stablehlo.reshape %v161 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v163 = stablehlo.reshape %v162 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v164 = stablehlo.broadcast_in_dim %s1b1n3mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v165 = stablehlo.subtract %v163, %v164 : tensor<256x256x56x56xf32>
    %v166 = stablehlo.broadcast_in_dim %s1b1n3var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v167 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v168 = stablehlo.add %v166, %v167 : tensor<256x256x56x56xf32>
    %v169 = stablehlo.rsqrt %v168 : tensor<256x256x56x56xf32>
    %v170 = stablehlo.multiply %v165, %v169 : tensor<256x256x56x56xf32>
    %v171 = stablehlo.broadcast_in_dim %s1b1g3, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v172 = stablehlo.broadcast_in_dim %s1b1bt3, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v173 = stablehlo.multiply %v170, %v171 : tensor<256x256x56x56xf32>
    %v174 = stablehlo.add %v173, %v172 : tensor<256x256x56x56xf32>
    %v175 = stablehlo.reshape %v174 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v176 = stablehlo.reshape %v175 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v177 = stablehlo.reshape %v113 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v178 = stablehlo.add %v176, %v177 : tensor<256x256x56x56xf32>
    %v179 = stablehlo.reshape %v178 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v180 = stablehlo.reshape %v179 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v181 = stablehlo.constant dense<0.0> : tensor<256x256x56x56xf32>
    %v182 = stablehlo.maximum %v180, %v181 : tensor<256x256x56x56xf32>
    %v183 = stablehlo.reshape %v182 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v184 = stablehlo.reshape %v183 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v185 = stablehlo.convolution(%v184, %s1b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tensor<256x64x56x56xf32>
    %v186 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v187 = stablehlo.add %v185, %v186 : tensor<256x64x56x56xf32>
    %v188 = stablehlo.reshape %v187 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v189 = stablehlo.reshape %v188 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v190 = stablehlo.broadcast_in_dim %s1b2n1mu, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v191 = stablehlo.subtract %v189, %v190 : tensor<256x64x56x56xf32>
    %v192 = stablehlo.broadcast_in_dim %s1b2n1var, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v193 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v194 = stablehlo.add %v192, %v193 : tensor<256x64x56x56xf32>
    %v195 = stablehlo.rsqrt %v194 : tensor<256x64x56x56xf32>
    %v196 = stablehlo.multiply %v191, %v195 : tensor<256x64x56x56xf32>
    %v197 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v198 = stablehlo.broadcast_in_dim %s1b2bt1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v199 = stablehlo.multiply %v196, %v197 : tensor<256x64x56x56xf32>
    %v200 = stablehlo.add %v199, %v198 : tensor<256x64x56x56xf32>
    %v201 = stablehlo.reshape %v200 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v202 = stablehlo.reshape %v201 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v203 = stablehlo.constant dense<0.0> : tensor<256x64x56x56xf32>
    %v204 = stablehlo.maximum %v202, %v203 : tensor<256x64x56x56xf32>
    %v205 = stablehlo.reshape %v204 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v206 = stablehlo.reshape %v205 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v207 = stablehlo.convolution(%v206, %s1b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v208 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v209 = stablehlo.add %v207, %v208 : tensor<256x64x56x56xf32>
    %v210 = stablehlo.reshape %v209 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v211 = stablehlo.reshape %v210 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v212 = stablehlo.broadcast_in_dim %s1b2n2mu, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v213 = stablehlo.subtract %v211, %v212 : tensor<256x64x56x56xf32>
    %v214 = stablehlo.broadcast_in_dim %s1b2n2var, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v215 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v216 = stablehlo.add %v214, %v215 : tensor<256x64x56x56xf32>
    %v217 = stablehlo.rsqrt %v216 : tensor<256x64x56x56xf32>
    %v218 = stablehlo.multiply %v213, %v217 : tensor<256x64x56x56xf32>
    %v219 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v220 = stablehlo.broadcast_in_dim %s1b2bt2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v221 = stablehlo.multiply %v218, %v219 : tensor<256x64x56x56xf32>
    %v222 = stablehlo.add %v221, %v220 : tensor<256x64x56x56xf32>
    %v223 = stablehlo.reshape %v222 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v224 = stablehlo.reshape %v223 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v225 = stablehlo.constant dense<0.0> : tensor<256x64x56x56xf32>
    %v226 = stablehlo.maximum %v224, %v225 : tensor<256x64x56x56xf32>
    %v227 = stablehlo.reshape %v226 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v228 = stablehlo.reshape %v227 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v229 = stablehlo.convolution(%v228, %s1b2W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<256x256x56x56xf32>
    %v230 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v231 = stablehlo.add %v229, %v230 : tensor<256x256x56x56xf32>
    %v232 = stablehlo.reshape %v231 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v233 = stablehlo.reshape %v232 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v234 = stablehlo.broadcast_in_dim %s1b2n3mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v235 = stablehlo.subtract %v233, %v234 : tensor<256x256x56x56xf32>
    %v236 = stablehlo.broadcast_in_dim %s1b2n3var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v237 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v238 = stablehlo.add %v236, %v237 : tensor<256x256x56x56xf32>
    %v239 = stablehlo.rsqrt %v238 : tensor<256x256x56x56xf32>
    %v240 = stablehlo.multiply %v235, %v239 : tensor<256x256x56x56xf32>
    %v241 = stablehlo.broadcast_in_dim %s1b2g3, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v242 = stablehlo.broadcast_in_dim %s1b2bt3, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v243 = stablehlo.multiply %v240, %v241 : tensor<256x256x56x56xf32>
    %v244 = stablehlo.add %v243, %v242 : tensor<256x256x56x56xf32>
    %v245 = stablehlo.reshape %v244 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v246 = stablehlo.reshape %v245 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v247 = stablehlo.reshape %v183 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v248 = stablehlo.add %v246, %v247 : tensor<256x256x56x56xf32>
    %v249 = stablehlo.reshape %v248 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v250 = stablehlo.reshape %v249 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v251 = stablehlo.constant dense<0.0> : tensor<256x256x56x56xf32>
    %v252 = stablehlo.maximum %v250, %v251 : tensor<256x256x56x56xf32>
    %v253 = stablehlo.reshape %v252 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v254 = stablehlo.reshape %v253 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v255 = stablehlo.convolution(%v254, %s2b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<128x256x1x1xf32>) -> tensor<256x128x56x56xf32>
    %v256 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v257 = stablehlo.add %v255, %v256 : tensor<256x128x56x56xf32>
    %v258 = stablehlo.reshape %v257 : (tensor<256x128x56x56xf32>) -> tensor<256x401408xf32>
    %v259 = stablehlo.reshape %v258 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v260 = stablehlo.broadcast_in_dim %s2b0n1mu, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v261 = stablehlo.subtract %v259, %v260 : tensor<256x128x56x56xf32>
    %v262 = stablehlo.broadcast_in_dim %s2b0n1var, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v263 = stablehlo.constant dense<1.0e-05> : tensor<256x128x56x56xf32>
    %v264 = stablehlo.add %v262, %v263 : tensor<256x128x56x56xf32>
    %v265 = stablehlo.rsqrt %v264 : tensor<256x128x56x56xf32>
    %v266 = stablehlo.multiply %v261, %v265 : tensor<256x128x56x56xf32>
    %v267 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v268 = stablehlo.broadcast_in_dim %s2b0bt1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v269 = stablehlo.multiply %v266, %v267 : tensor<256x128x56x56xf32>
    %v270 = stablehlo.add %v269, %v268 : tensor<256x128x56x56xf32>
    %v271 = stablehlo.reshape %v270 : (tensor<256x128x56x56xf32>) -> tensor<256x401408xf32>
    %v272 = stablehlo.reshape %v271 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v273 = stablehlo.constant dense<0.0> : tensor<256x128x56x56xf32>
    %v274 = stablehlo.maximum %v272, %v273 : tensor<256x128x56x56xf32>
    %v275 = stablehlo.reshape %v274 : (tensor<256x128x56x56xf32>) -> tensor<256x401408xf32>
    %v276 = stablehlo.reshape %v275 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v277 = stablehlo.convolution(%v276, %s2b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x56x56xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v278 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v279 = stablehlo.add %v277, %v278 : tensor<256x128x28x28xf32>
    %v280 = stablehlo.reshape %v279 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v281 = stablehlo.reshape %v280 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v282 = stablehlo.broadcast_in_dim %s2b0n2mu, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v283 = stablehlo.subtract %v281, %v282 : tensor<256x128x28x28xf32>
    %v284 = stablehlo.broadcast_in_dim %s2b0n2var, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v285 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v286 = stablehlo.add %v284, %v285 : tensor<256x128x28x28xf32>
    %v287 = stablehlo.rsqrt %v286 : tensor<256x128x28x28xf32>
    %v288 = stablehlo.multiply %v283, %v287 : tensor<256x128x28x28xf32>
    %v289 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v290 = stablehlo.broadcast_in_dim %s2b0bt2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v291 = stablehlo.multiply %v288, %v289 : tensor<256x128x28x28xf32>
    %v292 = stablehlo.add %v291, %v290 : tensor<256x128x28x28xf32>
    %v293 = stablehlo.reshape %v292 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v294 = stablehlo.reshape %v293 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v295 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v296 = stablehlo.maximum %v294, %v295 : tensor<256x128x28x28xf32>
    %v297 = stablehlo.reshape %v296 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v298 = stablehlo.reshape %v297 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v299 = stablehlo.convolution(%v298, %s2b0W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<256x512x28x28xf32>
    %v300 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v301 = stablehlo.add %v299, %v300 : tensor<256x512x28x28xf32>
    %v302 = stablehlo.reshape %v301 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v303 = stablehlo.reshape %v302 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v304 = stablehlo.broadcast_in_dim %s2b0n3mu, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v305 = stablehlo.subtract %v303, %v304 : tensor<256x512x28x28xf32>
    %v306 = stablehlo.broadcast_in_dim %s2b0n3var, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v307 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v308 = stablehlo.add %v306, %v307 : tensor<256x512x28x28xf32>
    %v309 = stablehlo.rsqrt %v308 : tensor<256x512x28x28xf32>
    %v310 = stablehlo.multiply %v305, %v309 : tensor<256x512x28x28xf32>
    %v311 = stablehlo.broadcast_in_dim %s2b0g3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v312 = stablehlo.broadcast_in_dim %s2b0bt3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v313 = stablehlo.multiply %v310, %v311 : tensor<256x512x28x28xf32>
    %v314 = stablehlo.add %v313, %v312 : tensor<256x512x28x28xf32>
    %v315 = stablehlo.reshape %v314 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v316 = stablehlo.reshape %v253 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v317 = stablehlo.convolution(%v316, %s2b0Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<512x256x1x1xf32>) -> tensor<256x512x28x28xf32>
    %v318 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v319 = stablehlo.add %v317, %v318 : tensor<256x512x28x28xf32>
    %v320 = stablehlo.reshape %v319 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v321 = stablehlo.reshape %v320 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v322 = stablehlo.broadcast_in_dim %s2b0npmu, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v323 = stablehlo.subtract %v321, %v322 : tensor<256x512x28x28xf32>
    %v324 = stablehlo.broadcast_in_dim %s2b0npvar, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v325 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v326 = stablehlo.add %v324, %v325 : tensor<256x512x28x28xf32>
    %v327 = stablehlo.rsqrt %v326 : tensor<256x512x28x28xf32>
    %v328 = stablehlo.multiply %v323, %v327 : tensor<256x512x28x28xf32>
    %v329 = stablehlo.broadcast_in_dim %s2b0gp, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v330 = stablehlo.broadcast_in_dim %s2b0btp, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v331 = stablehlo.multiply %v328, %v329 : tensor<256x512x28x28xf32>
    %v332 = stablehlo.add %v331, %v330 : tensor<256x512x28x28xf32>
    %v333 = stablehlo.reshape %v332 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v334 = stablehlo.reshape %v315 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v335 = stablehlo.reshape %v333 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v336 = stablehlo.add %v334, %v335 : tensor<256x512x28x28xf32>
    %v337 = stablehlo.reshape %v336 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v338 = stablehlo.reshape %v337 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v339 = stablehlo.constant dense<0.0> : tensor<256x512x28x28xf32>
    %v340 = stablehlo.maximum %v338, %v339 : tensor<256x512x28x28xf32>
    %v341 = stablehlo.reshape %v340 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v342 = stablehlo.reshape %v341 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v343 = stablehlo.convolution(%v342, %s2b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<256x128x28x28xf32>
    %v344 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v345 = stablehlo.add %v343, %v344 : tensor<256x128x28x28xf32>
    %v346 = stablehlo.reshape %v345 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v347 = stablehlo.reshape %v346 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v348 = stablehlo.broadcast_in_dim %s2b1n1mu, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v349 = stablehlo.subtract %v347, %v348 : tensor<256x128x28x28xf32>
    %v350 = stablehlo.broadcast_in_dim %s2b1n1var, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v351 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v352 = stablehlo.add %v350, %v351 : tensor<256x128x28x28xf32>
    %v353 = stablehlo.rsqrt %v352 : tensor<256x128x28x28xf32>
    %v354 = stablehlo.multiply %v349, %v353 : tensor<256x128x28x28xf32>
    %v355 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v356 = stablehlo.broadcast_in_dim %s2b1bt1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v357 = stablehlo.multiply %v354, %v355 : tensor<256x128x28x28xf32>
    %v358 = stablehlo.add %v357, %v356 : tensor<256x128x28x28xf32>
    %v359 = stablehlo.reshape %v358 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v360 = stablehlo.reshape %v359 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v361 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v362 = stablehlo.maximum %v360, %v361 : tensor<256x128x28x28xf32>
    %v363 = stablehlo.reshape %v362 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v364 = stablehlo.reshape %v363 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v365 = stablehlo.convolution(%v364, %s2b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v366 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v367 = stablehlo.add %v365, %v366 : tensor<256x128x28x28xf32>
    %v368 = stablehlo.reshape %v367 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v369 = stablehlo.reshape %v368 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v370 = stablehlo.broadcast_in_dim %s2b1n2mu, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v371 = stablehlo.subtract %v369, %v370 : tensor<256x128x28x28xf32>
    %v372 = stablehlo.broadcast_in_dim %s2b1n2var, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v373 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v374 = stablehlo.add %v372, %v373 : tensor<256x128x28x28xf32>
    %v375 = stablehlo.rsqrt %v374 : tensor<256x128x28x28xf32>
    %v376 = stablehlo.multiply %v371, %v375 : tensor<256x128x28x28xf32>
    %v377 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v378 = stablehlo.broadcast_in_dim %s2b1bt2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v379 = stablehlo.multiply %v376, %v377 : tensor<256x128x28x28xf32>
    %v380 = stablehlo.add %v379, %v378 : tensor<256x128x28x28xf32>
    %v381 = stablehlo.reshape %v380 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v382 = stablehlo.reshape %v381 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v383 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v384 = stablehlo.maximum %v382, %v383 : tensor<256x128x28x28xf32>
    %v385 = stablehlo.reshape %v384 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v386 = stablehlo.reshape %v385 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v387 = stablehlo.convolution(%v386, %s2b1W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<256x512x28x28xf32>
    %v388 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v389 = stablehlo.add %v387, %v388 : tensor<256x512x28x28xf32>
    %v390 = stablehlo.reshape %v389 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v391 = stablehlo.reshape %v390 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v392 = stablehlo.broadcast_in_dim %s2b1n3mu, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v393 = stablehlo.subtract %v391, %v392 : tensor<256x512x28x28xf32>
    %v394 = stablehlo.broadcast_in_dim %s2b1n3var, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v395 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v396 = stablehlo.add %v394, %v395 : tensor<256x512x28x28xf32>
    %v397 = stablehlo.rsqrt %v396 : tensor<256x512x28x28xf32>
    %v398 = stablehlo.multiply %v393, %v397 : tensor<256x512x28x28xf32>
    %v399 = stablehlo.broadcast_in_dim %s2b1g3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v400 = stablehlo.broadcast_in_dim %s2b1bt3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v401 = stablehlo.multiply %v398, %v399 : tensor<256x512x28x28xf32>
    %v402 = stablehlo.add %v401, %v400 : tensor<256x512x28x28xf32>
    %v403 = stablehlo.reshape %v402 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v404 = stablehlo.reshape %v403 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v405 = stablehlo.reshape %v341 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v406 = stablehlo.add %v404, %v405 : tensor<256x512x28x28xf32>
    %v407 = stablehlo.reshape %v406 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v408 = stablehlo.reshape %v407 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v409 = stablehlo.constant dense<0.0> : tensor<256x512x28x28xf32>
    %v410 = stablehlo.maximum %v408, %v409 : tensor<256x512x28x28xf32>
    %v411 = stablehlo.reshape %v410 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v412 = stablehlo.reshape %v411 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v413 = stablehlo.convolution(%v412, %s2b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<256x128x28x28xf32>
    %v414 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v415 = stablehlo.add %v413, %v414 : tensor<256x128x28x28xf32>
    %v416 = stablehlo.reshape %v415 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v417 = stablehlo.reshape %v416 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v418 = stablehlo.broadcast_in_dim %s2b2n1mu, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v419 = stablehlo.subtract %v417, %v418 : tensor<256x128x28x28xf32>
    %v420 = stablehlo.broadcast_in_dim %s2b2n1var, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v421 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v422 = stablehlo.add %v420, %v421 : tensor<256x128x28x28xf32>
    %v423 = stablehlo.rsqrt %v422 : tensor<256x128x28x28xf32>
    %v424 = stablehlo.multiply %v419, %v423 : tensor<256x128x28x28xf32>
    %v425 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v426 = stablehlo.broadcast_in_dim %s2b2bt1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v427 = stablehlo.multiply %v424, %v425 : tensor<256x128x28x28xf32>
    %v428 = stablehlo.add %v427, %v426 : tensor<256x128x28x28xf32>
    %v429 = stablehlo.reshape %v428 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v430 = stablehlo.reshape %v429 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v431 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v432 = stablehlo.maximum %v430, %v431 : tensor<256x128x28x28xf32>
    %v433 = stablehlo.reshape %v432 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v434 = stablehlo.reshape %v433 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v435 = stablehlo.convolution(%v434, %s2b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v436 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v437 = stablehlo.add %v435, %v436 : tensor<256x128x28x28xf32>
    %v438 = stablehlo.reshape %v437 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v439 = stablehlo.reshape %v438 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v440 = stablehlo.broadcast_in_dim %s2b2n2mu, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v441 = stablehlo.subtract %v439, %v440 : tensor<256x128x28x28xf32>
    %v442 = stablehlo.broadcast_in_dim %s2b2n2var, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v443 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v444 = stablehlo.add %v442, %v443 : tensor<256x128x28x28xf32>
    %v445 = stablehlo.rsqrt %v444 : tensor<256x128x28x28xf32>
    %v446 = stablehlo.multiply %v441, %v445 : tensor<256x128x28x28xf32>
    %v447 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v448 = stablehlo.broadcast_in_dim %s2b2bt2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v449 = stablehlo.multiply %v446, %v447 : tensor<256x128x28x28xf32>
    %v450 = stablehlo.add %v449, %v448 : tensor<256x128x28x28xf32>
    %v451 = stablehlo.reshape %v450 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v452 = stablehlo.reshape %v451 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v453 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v454 = stablehlo.maximum %v452, %v453 : tensor<256x128x28x28xf32>
    %v455 = stablehlo.reshape %v454 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v456 = stablehlo.reshape %v455 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v457 = stablehlo.convolution(%v456, %s2b2W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<256x512x28x28xf32>
    %v458 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v459 = stablehlo.add %v457, %v458 : tensor<256x512x28x28xf32>
    %v460 = stablehlo.reshape %v459 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v461 = stablehlo.reshape %v460 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v462 = stablehlo.broadcast_in_dim %s2b2n3mu, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v463 = stablehlo.subtract %v461, %v462 : tensor<256x512x28x28xf32>
    %v464 = stablehlo.broadcast_in_dim %s2b2n3var, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v465 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v466 = stablehlo.add %v464, %v465 : tensor<256x512x28x28xf32>
    %v467 = stablehlo.rsqrt %v466 : tensor<256x512x28x28xf32>
    %v468 = stablehlo.multiply %v463, %v467 : tensor<256x512x28x28xf32>
    %v469 = stablehlo.broadcast_in_dim %s2b2g3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v470 = stablehlo.broadcast_in_dim %s2b2bt3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v471 = stablehlo.multiply %v468, %v469 : tensor<256x512x28x28xf32>
    %v472 = stablehlo.add %v471, %v470 : tensor<256x512x28x28xf32>
    %v473 = stablehlo.reshape %v472 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v474 = stablehlo.reshape %v473 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v475 = stablehlo.reshape %v411 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v476 = stablehlo.add %v474, %v475 : tensor<256x512x28x28xf32>
    %v477 = stablehlo.reshape %v476 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v478 = stablehlo.reshape %v477 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v479 = stablehlo.constant dense<0.0> : tensor<256x512x28x28xf32>
    %v480 = stablehlo.maximum %v478, %v479 : tensor<256x512x28x28xf32>
    %v481 = stablehlo.reshape %v480 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v482 = stablehlo.reshape %v481 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v483 = stablehlo.convolution(%v482, %s2b3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<256x128x28x28xf32>
    %v484 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v485 = stablehlo.add %v483, %v484 : tensor<256x128x28x28xf32>
    %v486 = stablehlo.reshape %v485 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v487 = stablehlo.reshape %v486 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v488 = stablehlo.broadcast_in_dim %s2b3n1mu, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v489 = stablehlo.subtract %v487, %v488 : tensor<256x128x28x28xf32>
    %v490 = stablehlo.broadcast_in_dim %s2b3n1var, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v491 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v492 = stablehlo.add %v490, %v491 : tensor<256x128x28x28xf32>
    %v493 = stablehlo.rsqrt %v492 : tensor<256x128x28x28xf32>
    %v494 = stablehlo.multiply %v489, %v493 : tensor<256x128x28x28xf32>
    %v495 = stablehlo.broadcast_in_dim %s2b3g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v496 = stablehlo.broadcast_in_dim %s2b3bt1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v497 = stablehlo.multiply %v494, %v495 : tensor<256x128x28x28xf32>
    %v498 = stablehlo.add %v497, %v496 : tensor<256x128x28x28xf32>
    %v499 = stablehlo.reshape %v498 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v500 = stablehlo.reshape %v499 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v501 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v502 = stablehlo.maximum %v500, %v501 : tensor<256x128x28x28xf32>
    %v503 = stablehlo.reshape %v502 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v504 = stablehlo.reshape %v503 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v505 = stablehlo.convolution(%v504, %s2b3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v506 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v507 = stablehlo.add %v505, %v506 : tensor<256x128x28x28xf32>
    %v508 = stablehlo.reshape %v507 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v509 = stablehlo.reshape %v508 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v510 = stablehlo.broadcast_in_dim %s2b3n2mu, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v511 = stablehlo.subtract %v509, %v510 : tensor<256x128x28x28xf32>
    %v512 = stablehlo.broadcast_in_dim %s2b3n2var, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v513 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v514 = stablehlo.add %v512, %v513 : tensor<256x128x28x28xf32>
    %v515 = stablehlo.rsqrt %v514 : tensor<256x128x28x28xf32>
    %v516 = stablehlo.multiply %v511, %v515 : tensor<256x128x28x28xf32>
    %v517 = stablehlo.broadcast_in_dim %s2b3g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v518 = stablehlo.broadcast_in_dim %s2b3bt2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v519 = stablehlo.multiply %v516, %v517 : tensor<256x128x28x28xf32>
    %v520 = stablehlo.add %v519, %v518 : tensor<256x128x28x28xf32>
    %v521 = stablehlo.reshape %v520 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v522 = stablehlo.reshape %v521 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v523 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v524 = stablehlo.maximum %v522, %v523 : tensor<256x128x28x28xf32>
    %v525 = stablehlo.reshape %v524 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v526 = stablehlo.reshape %v525 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v527 = stablehlo.convolution(%v526, %s2b3W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<256x512x28x28xf32>
    %v528 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v529 = stablehlo.add %v527, %v528 : tensor<256x512x28x28xf32>
    %v530 = stablehlo.reshape %v529 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v531 = stablehlo.reshape %v530 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v532 = stablehlo.broadcast_in_dim %s2b3n3mu, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v533 = stablehlo.subtract %v531, %v532 : tensor<256x512x28x28xf32>
    %v534 = stablehlo.broadcast_in_dim %s2b3n3var, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v535 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v536 = stablehlo.add %v534, %v535 : tensor<256x512x28x28xf32>
    %v537 = stablehlo.rsqrt %v536 : tensor<256x512x28x28xf32>
    %v538 = stablehlo.multiply %v533, %v537 : tensor<256x512x28x28xf32>
    %v539 = stablehlo.broadcast_in_dim %s2b3g3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v540 = stablehlo.broadcast_in_dim %s2b3bt3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v541 = stablehlo.multiply %v538, %v539 : tensor<256x512x28x28xf32>
    %v542 = stablehlo.add %v541, %v540 : tensor<256x512x28x28xf32>
    %v543 = stablehlo.reshape %v542 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v544 = stablehlo.reshape %v543 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v545 = stablehlo.reshape %v481 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v546 = stablehlo.add %v544, %v545 : tensor<256x512x28x28xf32>
    %v547 = stablehlo.reshape %v546 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v548 = stablehlo.reshape %v547 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v549 = stablehlo.constant dense<0.0> : tensor<256x512x28x28xf32>
    %v550 = stablehlo.maximum %v548, %v549 : tensor<256x512x28x28xf32>
    %v551 = stablehlo.reshape %v550 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v552 = stablehlo.reshape %v551 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v553 = stablehlo.convolution(%v552, %s3b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x28x28xf32>, tensor<256x512x1x1xf32>) -> tensor<256x256x28x28xf32>
    %v554 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v555 = stablehlo.add %v553, %v554 : tensor<256x256x28x28xf32>
    %v556 = stablehlo.reshape %v555 : (tensor<256x256x28x28xf32>) -> tensor<256x200704xf32>
    %v557 = stablehlo.reshape %v556 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v558 = stablehlo.broadcast_in_dim %s3b0n1mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v559 = stablehlo.subtract %v557, %v558 : tensor<256x256x28x28xf32>
    %v560 = stablehlo.broadcast_in_dim %s3b0n1var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v561 = stablehlo.constant dense<1.0e-05> : tensor<256x256x28x28xf32>
    %v562 = stablehlo.add %v560, %v561 : tensor<256x256x28x28xf32>
    %v563 = stablehlo.rsqrt %v562 : tensor<256x256x28x28xf32>
    %v564 = stablehlo.multiply %v559, %v563 : tensor<256x256x28x28xf32>
    %v565 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v566 = stablehlo.broadcast_in_dim %s3b0bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v567 = stablehlo.multiply %v564, %v565 : tensor<256x256x28x28xf32>
    %v568 = stablehlo.add %v567, %v566 : tensor<256x256x28x28xf32>
    %v569 = stablehlo.reshape %v568 : (tensor<256x256x28x28xf32>) -> tensor<256x200704xf32>
    %v570 = stablehlo.reshape %v569 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v571 = stablehlo.constant dense<0.0> : tensor<256x256x28x28xf32>
    %v572 = stablehlo.maximum %v570, %v571 : tensor<256x256x28x28xf32>
    %v573 = stablehlo.reshape %v572 : (tensor<256x256x28x28xf32>) -> tensor<256x200704xf32>
    %v574 = stablehlo.reshape %v573 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v575 = stablehlo.convolution(%v574, %s3b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x28x28xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v576 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v577 = stablehlo.add %v575, %v576 : tensor<256x256x14x14xf32>
    %v578 = stablehlo.reshape %v577 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v579 = stablehlo.reshape %v578 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v580 = stablehlo.broadcast_in_dim %s3b0n2mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v581 = stablehlo.subtract %v579, %v580 : tensor<256x256x14x14xf32>
    %v582 = stablehlo.broadcast_in_dim %s3b0n2var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v583 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v584 = stablehlo.add %v582, %v583 : tensor<256x256x14x14xf32>
    %v585 = stablehlo.rsqrt %v584 : tensor<256x256x14x14xf32>
    %v586 = stablehlo.multiply %v581, %v585 : tensor<256x256x14x14xf32>
    %v587 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v588 = stablehlo.broadcast_in_dim %s3b0bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v589 = stablehlo.multiply %v586, %v587 : tensor<256x256x14x14xf32>
    %v590 = stablehlo.add %v589, %v588 : tensor<256x256x14x14xf32>
    %v591 = stablehlo.reshape %v590 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v592 = stablehlo.reshape %v591 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v593 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v594 = stablehlo.maximum %v592, %v593 : tensor<256x256x14x14xf32>
    %v595 = stablehlo.reshape %v594 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v596 = stablehlo.reshape %v595 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v597 = stablehlo.convolution(%v596, %s3b0W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v598 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v599 = stablehlo.add %v597, %v598 : tensor<256x1024x14x14xf32>
    %v600 = stablehlo.reshape %v599 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v601 = stablehlo.reshape %v600 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v602 = stablehlo.broadcast_in_dim %s3b0n3mu, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v603 = stablehlo.subtract %v601, %v602 : tensor<256x1024x14x14xf32>
    %v604 = stablehlo.broadcast_in_dim %s3b0n3var, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v605 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v606 = stablehlo.add %v604, %v605 : tensor<256x1024x14x14xf32>
    %v607 = stablehlo.rsqrt %v606 : tensor<256x1024x14x14xf32>
    %v608 = stablehlo.multiply %v603, %v607 : tensor<256x1024x14x14xf32>
    %v609 = stablehlo.broadcast_in_dim %s3b0g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v610 = stablehlo.broadcast_in_dim %s3b0bt3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v611 = stablehlo.multiply %v608, %v609 : tensor<256x1024x14x14xf32>
    %v612 = stablehlo.add %v611, %v610 : tensor<256x1024x14x14xf32>
    %v613 = stablehlo.reshape %v612 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v614 = stablehlo.reshape %v551 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v615 = stablehlo.convolution(%v614, %s3b0Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x28x28xf32>, tensor<1024x512x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v616 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v617 = stablehlo.add %v615, %v616 : tensor<256x1024x14x14xf32>
    %v618 = stablehlo.reshape %v617 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v619 = stablehlo.reshape %v618 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v620 = stablehlo.broadcast_in_dim %s3b0npmu, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v621 = stablehlo.subtract %v619, %v620 : tensor<256x1024x14x14xf32>
    %v622 = stablehlo.broadcast_in_dim %s3b0npvar, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v623 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v624 = stablehlo.add %v622, %v623 : tensor<256x1024x14x14xf32>
    %v625 = stablehlo.rsqrt %v624 : tensor<256x1024x14x14xf32>
    %v626 = stablehlo.multiply %v621, %v625 : tensor<256x1024x14x14xf32>
    %v627 = stablehlo.broadcast_in_dim %s3b0gp, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v628 = stablehlo.broadcast_in_dim %s3b0btp, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v629 = stablehlo.multiply %v626, %v627 : tensor<256x1024x14x14xf32>
    %v630 = stablehlo.add %v629, %v628 : tensor<256x1024x14x14xf32>
    %v631 = stablehlo.reshape %v630 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v632 = stablehlo.reshape %v613 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v633 = stablehlo.reshape %v631 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v634 = stablehlo.add %v632, %v633 : tensor<256x1024x14x14xf32>
    %v635 = stablehlo.reshape %v634 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v636 = stablehlo.reshape %v635 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v637 = stablehlo.constant dense<0.0> : tensor<256x1024x14x14xf32>
    %v638 = stablehlo.maximum %v636, %v637 : tensor<256x1024x14x14xf32>
    %v639 = stablehlo.reshape %v638 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v640 = stablehlo.reshape %v639 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v641 = stablehlo.convolution(%v640, %s3b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v642 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v643 = stablehlo.add %v641, %v642 : tensor<256x256x14x14xf32>
    %v644 = stablehlo.reshape %v643 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v645 = stablehlo.reshape %v644 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v646 = stablehlo.broadcast_in_dim %s3b1n1mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v647 = stablehlo.subtract %v645, %v646 : tensor<256x256x14x14xf32>
    %v648 = stablehlo.broadcast_in_dim %s3b1n1var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v649 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v650 = stablehlo.add %v648, %v649 : tensor<256x256x14x14xf32>
    %v651 = stablehlo.rsqrt %v650 : tensor<256x256x14x14xf32>
    %v652 = stablehlo.multiply %v647, %v651 : tensor<256x256x14x14xf32>
    %v653 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v654 = stablehlo.broadcast_in_dim %s3b1bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v655 = stablehlo.multiply %v652, %v653 : tensor<256x256x14x14xf32>
    %v656 = stablehlo.add %v655, %v654 : tensor<256x256x14x14xf32>
    %v657 = stablehlo.reshape %v656 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v658 = stablehlo.reshape %v657 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v659 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v660 = stablehlo.maximum %v658, %v659 : tensor<256x256x14x14xf32>
    %v661 = stablehlo.reshape %v660 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v662 = stablehlo.reshape %v661 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v663 = stablehlo.convolution(%v662, %s3b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v664 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v665 = stablehlo.add %v663, %v664 : tensor<256x256x14x14xf32>
    %v666 = stablehlo.reshape %v665 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v667 = stablehlo.reshape %v666 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v668 = stablehlo.broadcast_in_dim %s3b1n2mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v669 = stablehlo.subtract %v667, %v668 : tensor<256x256x14x14xf32>
    %v670 = stablehlo.broadcast_in_dim %s3b1n2var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v671 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v672 = stablehlo.add %v670, %v671 : tensor<256x256x14x14xf32>
    %v673 = stablehlo.rsqrt %v672 : tensor<256x256x14x14xf32>
    %v674 = stablehlo.multiply %v669, %v673 : tensor<256x256x14x14xf32>
    %v675 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v676 = stablehlo.broadcast_in_dim %s3b1bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v677 = stablehlo.multiply %v674, %v675 : tensor<256x256x14x14xf32>
    %v678 = stablehlo.add %v677, %v676 : tensor<256x256x14x14xf32>
    %v679 = stablehlo.reshape %v678 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v680 = stablehlo.reshape %v679 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v681 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v682 = stablehlo.maximum %v680, %v681 : tensor<256x256x14x14xf32>
    %v683 = stablehlo.reshape %v682 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v684 = stablehlo.reshape %v683 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v685 = stablehlo.convolution(%v684, %s3b1W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v686 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v687 = stablehlo.add %v685, %v686 : tensor<256x1024x14x14xf32>
    %v688 = stablehlo.reshape %v687 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v689 = stablehlo.reshape %v688 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v690 = stablehlo.broadcast_in_dim %s3b1n3mu, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v691 = stablehlo.subtract %v689, %v690 : tensor<256x1024x14x14xf32>
    %v692 = stablehlo.broadcast_in_dim %s3b1n3var, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v693 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v694 = stablehlo.add %v692, %v693 : tensor<256x1024x14x14xf32>
    %v695 = stablehlo.rsqrt %v694 : tensor<256x1024x14x14xf32>
    %v696 = stablehlo.multiply %v691, %v695 : tensor<256x1024x14x14xf32>
    %v697 = stablehlo.broadcast_in_dim %s3b1g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v698 = stablehlo.broadcast_in_dim %s3b1bt3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v699 = stablehlo.multiply %v696, %v697 : tensor<256x1024x14x14xf32>
    %v700 = stablehlo.add %v699, %v698 : tensor<256x1024x14x14xf32>
    %v701 = stablehlo.reshape %v700 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v702 = stablehlo.reshape %v701 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v703 = stablehlo.reshape %v639 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v704 = stablehlo.add %v702, %v703 : tensor<256x1024x14x14xf32>
    %v705 = stablehlo.reshape %v704 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v706 = stablehlo.reshape %v705 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v707 = stablehlo.constant dense<0.0> : tensor<256x1024x14x14xf32>
    %v708 = stablehlo.maximum %v706, %v707 : tensor<256x1024x14x14xf32>
    %v709 = stablehlo.reshape %v708 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v710 = stablehlo.reshape %v709 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v711 = stablehlo.convolution(%v710, %s3b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v712 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v713 = stablehlo.add %v711, %v712 : tensor<256x256x14x14xf32>
    %v714 = stablehlo.reshape %v713 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v715 = stablehlo.reshape %v714 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v716 = stablehlo.broadcast_in_dim %s3b2n1mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v717 = stablehlo.subtract %v715, %v716 : tensor<256x256x14x14xf32>
    %v718 = stablehlo.broadcast_in_dim %s3b2n1var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v719 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v720 = stablehlo.add %v718, %v719 : tensor<256x256x14x14xf32>
    %v721 = stablehlo.rsqrt %v720 : tensor<256x256x14x14xf32>
    %v722 = stablehlo.multiply %v717, %v721 : tensor<256x256x14x14xf32>
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
    %v738 = stablehlo.broadcast_in_dim %s3b2n2mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v739 = stablehlo.subtract %v737, %v738 : tensor<256x256x14x14xf32>
    %v740 = stablehlo.broadcast_in_dim %s3b2n2var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v741 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v742 = stablehlo.add %v740, %v741 : tensor<256x256x14x14xf32>
    %v743 = stablehlo.rsqrt %v742 : tensor<256x256x14x14xf32>
    %v744 = stablehlo.multiply %v739, %v743 : tensor<256x256x14x14xf32>
    %v745 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v746 = stablehlo.broadcast_in_dim %s3b2bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v747 = stablehlo.multiply %v744, %v745 : tensor<256x256x14x14xf32>
    %v748 = stablehlo.add %v747, %v746 : tensor<256x256x14x14xf32>
    %v749 = stablehlo.reshape %v748 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v750 = stablehlo.reshape %v749 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v751 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v752 = stablehlo.maximum %v750, %v751 : tensor<256x256x14x14xf32>
    %v753 = stablehlo.reshape %v752 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v754 = stablehlo.reshape %v753 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v755 = stablehlo.convolution(%v754, %s3b2W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v756 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v757 = stablehlo.add %v755, %v756 : tensor<256x1024x14x14xf32>
    %v758 = stablehlo.reshape %v757 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v759 = stablehlo.reshape %v758 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v760 = stablehlo.broadcast_in_dim %s3b2n3mu, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v761 = stablehlo.subtract %v759, %v760 : tensor<256x1024x14x14xf32>
    %v762 = stablehlo.broadcast_in_dim %s3b2n3var, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v763 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v764 = stablehlo.add %v762, %v763 : tensor<256x1024x14x14xf32>
    %v765 = stablehlo.rsqrt %v764 : tensor<256x1024x14x14xf32>
    %v766 = stablehlo.multiply %v761, %v765 : tensor<256x1024x14x14xf32>
    %v767 = stablehlo.broadcast_in_dim %s3b2g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v768 = stablehlo.broadcast_in_dim %s3b2bt3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v769 = stablehlo.multiply %v766, %v767 : tensor<256x1024x14x14xf32>
    %v770 = stablehlo.add %v769, %v768 : tensor<256x1024x14x14xf32>
    %v771 = stablehlo.reshape %v770 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v772 = stablehlo.reshape %v771 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v773 = stablehlo.reshape %v709 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v774 = stablehlo.add %v772, %v773 : tensor<256x1024x14x14xf32>
    %v775 = stablehlo.reshape %v774 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v776 = stablehlo.reshape %v775 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v777 = stablehlo.constant dense<0.0> : tensor<256x1024x14x14xf32>
    %v778 = stablehlo.maximum %v776, %v777 : tensor<256x1024x14x14xf32>
    %v779 = stablehlo.reshape %v778 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v780 = stablehlo.reshape %v779 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v781 = stablehlo.convolution(%v780, %s3b3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v782 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v783 = stablehlo.add %v781, %v782 : tensor<256x256x14x14xf32>
    %v784 = stablehlo.reshape %v783 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v785 = stablehlo.reshape %v784 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v786 = stablehlo.broadcast_in_dim %s3b3n1mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v787 = stablehlo.subtract %v785, %v786 : tensor<256x256x14x14xf32>
    %v788 = stablehlo.broadcast_in_dim %s3b3n1var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v789 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v790 = stablehlo.add %v788, %v789 : tensor<256x256x14x14xf32>
    %v791 = stablehlo.rsqrt %v790 : tensor<256x256x14x14xf32>
    %v792 = stablehlo.multiply %v787, %v791 : tensor<256x256x14x14xf32>
    %v793 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v794 = stablehlo.broadcast_in_dim %s3b3bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v795 = stablehlo.multiply %v792, %v793 : tensor<256x256x14x14xf32>
    %v796 = stablehlo.add %v795, %v794 : tensor<256x256x14x14xf32>
    %v797 = stablehlo.reshape %v796 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v798 = stablehlo.reshape %v797 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v799 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v800 = stablehlo.maximum %v798, %v799 : tensor<256x256x14x14xf32>
    %v801 = stablehlo.reshape %v800 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v802 = stablehlo.reshape %v801 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v803 = stablehlo.convolution(%v802, %s3b3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v804 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v805 = stablehlo.add %v803, %v804 : tensor<256x256x14x14xf32>
    %v806 = stablehlo.reshape %v805 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v807 = stablehlo.reshape %v806 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v808 = stablehlo.broadcast_in_dim %s3b3n2mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v809 = stablehlo.subtract %v807, %v808 : tensor<256x256x14x14xf32>
    %v810 = stablehlo.broadcast_in_dim %s3b3n2var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v811 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v812 = stablehlo.add %v810, %v811 : tensor<256x256x14x14xf32>
    %v813 = stablehlo.rsqrt %v812 : tensor<256x256x14x14xf32>
    %v814 = stablehlo.multiply %v809, %v813 : tensor<256x256x14x14xf32>
    %v815 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v816 = stablehlo.broadcast_in_dim %s3b3bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v817 = stablehlo.multiply %v814, %v815 : tensor<256x256x14x14xf32>
    %v818 = stablehlo.add %v817, %v816 : tensor<256x256x14x14xf32>
    %v819 = stablehlo.reshape %v818 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v820 = stablehlo.reshape %v819 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v821 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v822 = stablehlo.maximum %v820, %v821 : tensor<256x256x14x14xf32>
    %v823 = stablehlo.reshape %v822 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v824 = stablehlo.reshape %v823 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v825 = stablehlo.convolution(%v824, %s3b3W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v826 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v827 = stablehlo.add %v825, %v826 : tensor<256x1024x14x14xf32>
    %v828 = stablehlo.reshape %v827 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v829 = stablehlo.reshape %v828 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v830 = stablehlo.broadcast_in_dim %s3b3n3mu, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v831 = stablehlo.subtract %v829, %v830 : tensor<256x1024x14x14xf32>
    %v832 = stablehlo.broadcast_in_dim %s3b3n3var, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v833 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v834 = stablehlo.add %v832, %v833 : tensor<256x1024x14x14xf32>
    %v835 = stablehlo.rsqrt %v834 : tensor<256x1024x14x14xf32>
    %v836 = stablehlo.multiply %v831, %v835 : tensor<256x1024x14x14xf32>
    %v837 = stablehlo.broadcast_in_dim %s3b3g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v838 = stablehlo.broadcast_in_dim %s3b3bt3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v839 = stablehlo.multiply %v836, %v837 : tensor<256x1024x14x14xf32>
    %v840 = stablehlo.add %v839, %v838 : tensor<256x1024x14x14xf32>
    %v841 = stablehlo.reshape %v840 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v842 = stablehlo.reshape %v841 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v843 = stablehlo.reshape %v779 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v844 = stablehlo.add %v842, %v843 : tensor<256x1024x14x14xf32>
    %v845 = stablehlo.reshape %v844 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v846 = stablehlo.reshape %v845 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v847 = stablehlo.constant dense<0.0> : tensor<256x1024x14x14xf32>
    %v848 = stablehlo.maximum %v846, %v847 : tensor<256x1024x14x14xf32>
    %v849 = stablehlo.reshape %v848 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v850 = stablehlo.reshape %v849 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v851 = stablehlo.convolution(%v850, %s3b4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v852 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v853 = stablehlo.add %v851, %v852 : tensor<256x256x14x14xf32>
    %v854 = stablehlo.reshape %v853 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v855 = stablehlo.reshape %v854 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v856 = stablehlo.broadcast_in_dim %s3b4n1mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v857 = stablehlo.subtract %v855, %v856 : tensor<256x256x14x14xf32>
    %v858 = stablehlo.broadcast_in_dim %s3b4n1var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v859 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v860 = stablehlo.add %v858, %v859 : tensor<256x256x14x14xf32>
    %v861 = stablehlo.rsqrt %v860 : tensor<256x256x14x14xf32>
    %v862 = stablehlo.multiply %v857, %v861 : tensor<256x256x14x14xf32>
    %v863 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v864 = stablehlo.broadcast_in_dim %s3b4bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v865 = stablehlo.multiply %v862, %v863 : tensor<256x256x14x14xf32>
    %v866 = stablehlo.add %v865, %v864 : tensor<256x256x14x14xf32>
    %v867 = stablehlo.reshape %v866 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v868 = stablehlo.reshape %v867 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v869 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v870 = stablehlo.maximum %v868, %v869 : tensor<256x256x14x14xf32>
    %v871 = stablehlo.reshape %v870 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v872 = stablehlo.reshape %v871 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v873 = stablehlo.convolution(%v872, %s3b4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v874 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v875 = stablehlo.add %v873, %v874 : tensor<256x256x14x14xf32>
    %v876 = stablehlo.reshape %v875 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v877 = stablehlo.reshape %v876 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v878 = stablehlo.broadcast_in_dim %s3b4n2mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v879 = stablehlo.subtract %v877, %v878 : tensor<256x256x14x14xf32>
    %v880 = stablehlo.broadcast_in_dim %s3b4n2var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v881 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v882 = stablehlo.add %v880, %v881 : tensor<256x256x14x14xf32>
    %v883 = stablehlo.rsqrt %v882 : tensor<256x256x14x14xf32>
    %v884 = stablehlo.multiply %v879, %v883 : tensor<256x256x14x14xf32>
    %v885 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v886 = stablehlo.broadcast_in_dim %s3b4bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v887 = stablehlo.multiply %v884, %v885 : tensor<256x256x14x14xf32>
    %v888 = stablehlo.add %v887, %v886 : tensor<256x256x14x14xf32>
    %v889 = stablehlo.reshape %v888 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v890 = stablehlo.reshape %v889 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v891 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v892 = stablehlo.maximum %v890, %v891 : tensor<256x256x14x14xf32>
    %v893 = stablehlo.reshape %v892 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v894 = stablehlo.reshape %v893 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v895 = stablehlo.convolution(%v894, %s3b4W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v896 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v897 = stablehlo.add %v895, %v896 : tensor<256x1024x14x14xf32>
    %v898 = stablehlo.reshape %v897 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v899 = stablehlo.reshape %v898 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v900 = stablehlo.broadcast_in_dim %s3b4n3mu, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v901 = stablehlo.subtract %v899, %v900 : tensor<256x1024x14x14xf32>
    %v902 = stablehlo.broadcast_in_dim %s3b4n3var, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v903 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v904 = stablehlo.add %v902, %v903 : tensor<256x1024x14x14xf32>
    %v905 = stablehlo.rsqrt %v904 : tensor<256x1024x14x14xf32>
    %v906 = stablehlo.multiply %v901, %v905 : tensor<256x1024x14x14xf32>
    %v907 = stablehlo.broadcast_in_dim %s3b4g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v908 = stablehlo.broadcast_in_dim %s3b4bt3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v909 = stablehlo.multiply %v906, %v907 : tensor<256x1024x14x14xf32>
    %v910 = stablehlo.add %v909, %v908 : tensor<256x1024x14x14xf32>
    %v911 = stablehlo.reshape %v910 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v912 = stablehlo.reshape %v911 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v913 = stablehlo.reshape %v849 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v914 = stablehlo.add %v912, %v913 : tensor<256x1024x14x14xf32>
    %v915 = stablehlo.reshape %v914 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v916 = stablehlo.reshape %v915 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v917 = stablehlo.constant dense<0.0> : tensor<256x1024x14x14xf32>
    %v918 = stablehlo.maximum %v916, %v917 : tensor<256x1024x14x14xf32>
    %v919 = stablehlo.reshape %v918 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v920 = stablehlo.reshape %v919 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v921 = stablehlo.convolution(%v920, %s3b5W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v922 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v923 = stablehlo.add %v921, %v922 : tensor<256x256x14x14xf32>
    %v924 = stablehlo.reshape %v923 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v925 = stablehlo.reshape %v924 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v926 = stablehlo.broadcast_in_dim %s3b5n1mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v927 = stablehlo.subtract %v925, %v926 : tensor<256x256x14x14xf32>
    %v928 = stablehlo.broadcast_in_dim %s3b5n1var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v929 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v930 = stablehlo.add %v928, %v929 : tensor<256x256x14x14xf32>
    %v931 = stablehlo.rsqrt %v930 : tensor<256x256x14x14xf32>
    %v932 = stablehlo.multiply %v927, %v931 : tensor<256x256x14x14xf32>
    %v933 = stablehlo.broadcast_in_dim %s3b5g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v934 = stablehlo.broadcast_in_dim %s3b5bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v935 = stablehlo.multiply %v932, %v933 : tensor<256x256x14x14xf32>
    %v936 = stablehlo.add %v935, %v934 : tensor<256x256x14x14xf32>
    %v937 = stablehlo.reshape %v936 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v938 = stablehlo.reshape %v937 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v939 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v940 = stablehlo.maximum %v938, %v939 : tensor<256x256x14x14xf32>
    %v941 = stablehlo.reshape %v940 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v942 = stablehlo.reshape %v941 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v943 = stablehlo.convolution(%v942, %s3b5W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v944 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v945 = stablehlo.add %v943, %v944 : tensor<256x256x14x14xf32>
    %v946 = stablehlo.reshape %v945 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v947 = stablehlo.reshape %v946 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v948 = stablehlo.broadcast_in_dim %s3b5n2mu, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v949 = stablehlo.subtract %v947, %v948 : tensor<256x256x14x14xf32>
    %v950 = stablehlo.broadcast_in_dim %s3b5n2var, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v951 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v952 = stablehlo.add %v950, %v951 : tensor<256x256x14x14xf32>
    %v953 = stablehlo.rsqrt %v952 : tensor<256x256x14x14xf32>
    %v954 = stablehlo.multiply %v949, %v953 : tensor<256x256x14x14xf32>
    %v955 = stablehlo.broadcast_in_dim %s3b5g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v956 = stablehlo.broadcast_in_dim %s3b5bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v957 = stablehlo.multiply %v954, %v955 : tensor<256x256x14x14xf32>
    %v958 = stablehlo.add %v957, %v956 : tensor<256x256x14x14xf32>
    %v959 = stablehlo.reshape %v958 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v960 = stablehlo.reshape %v959 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v961 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v962 = stablehlo.maximum %v960, %v961 : tensor<256x256x14x14xf32>
    %v963 = stablehlo.reshape %v962 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v964 = stablehlo.reshape %v963 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v965 = stablehlo.convolution(%v964, %s3b5W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v966 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v967 = stablehlo.add %v965, %v966 : tensor<256x1024x14x14xf32>
    %v968 = stablehlo.reshape %v967 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v969 = stablehlo.reshape %v968 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v970 = stablehlo.broadcast_in_dim %s3b5n3mu, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v971 = stablehlo.subtract %v969, %v970 : tensor<256x1024x14x14xf32>
    %v972 = stablehlo.broadcast_in_dim %s3b5n3var, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v973 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v974 = stablehlo.add %v972, %v973 : tensor<256x1024x14x14xf32>
    %v975 = stablehlo.rsqrt %v974 : tensor<256x1024x14x14xf32>
    %v976 = stablehlo.multiply %v971, %v975 : tensor<256x1024x14x14xf32>
    %v977 = stablehlo.broadcast_in_dim %s3b5g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v978 = stablehlo.broadcast_in_dim %s3b5bt3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v979 = stablehlo.multiply %v976, %v977 : tensor<256x1024x14x14xf32>
    %v980 = stablehlo.add %v979, %v978 : tensor<256x1024x14x14xf32>
    %v981 = stablehlo.reshape %v980 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v982 = stablehlo.reshape %v981 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v983 = stablehlo.reshape %v919 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v984 = stablehlo.add %v982, %v983 : tensor<256x1024x14x14xf32>
    %v985 = stablehlo.reshape %v984 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v986 = stablehlo.reshape %v985 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v987 = stablehlo.constant dense<0.0> : tensor<256x1024x14x14xf32>
    %v988 = stablehlo.maximum %v986, %v987 : tensor<256x1024x14x14xf32>
    %v989 = stablehlo.reshape %v988 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v990 = stablehlo.reshape %v989 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v991 = stablehlo.convolution(%v990, %s4b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<512x1024x1x1xf32>) -> tensor<256x512x14x14xf32>
    %v992 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v993 = stablehlo.add %v991, %v992 : tensor<256x512x14x14xf32>
    %v994 = stablehlo.reshape %v993 : (tensor<256x512x14x14xf32>) -> tensor<256x100352xf32>
    %v995 = stablehlo.reshape %v994 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v996 = stablehlo.broadcast_in_dim %s4b0n1mu, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v997 = stablehlo.subtract %v995, %v996 : tensor<256x512x14x14xf32>
    %v998 = stablehlo.broadcast_in_dim %s4b0n1var, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v999 = stablehlo.constant dense<1.0e-05> : tensor<256x512x14x14xf32>
    %v1000 = stablehlo.add %v998, %v999 : tensor<256x512x14x14xf32>
    %v1001 = stablehlo.rsqrt %v1000 : tensor<256x512x14x14xf32>
    %v1002 = stablehlo.multiply %v997, %v1001 : tensor<256x512x14x14xf32>
    %v1003 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v1004 = stablehlo.broadcast_in_dim %s4b0bt1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v1005 = stablehlo.multiply %v1002, %v1003 : tensor<256x512x14x14xf32>
    %v1006 = stablehlo.add %v1005, %v1004 : tensor<256x512x14x14xf32>
    %v1007 = stablehlo.reshape %v1006 : (tensor<256x512x14x14xf32>) -> tensor<256x100352xf32>
    %v1008 = stablehlo.reshape %v1007 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v1009 = stablehlo.constant dense<0.0> : tensor<256x512x14x14xf32>
    %v1010 = stablehlo.maximum %v1008, %v1009 : tensor<256x512x14x14xf32>
    %v1011 = stablehlo.reshape %v1010 : (tensor<256x512x14x14xf32>) -> tensor<256x100352xf32>
    %v1012 = stablehlo.reshape %v1011 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v1013 = stablehlo.convolution(%v1012, %s4b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x14x14xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v1014 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1015 = stablehlo.add %v1013, %v1014 : tensor<256x512x7x7xf32>
    %v1016 = stablehlo.reshape %v1015 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1017 = stablehlo.reshape %v1016 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1018 = stablehlo.broadcast_in_dim %s4b0n2mu, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1019 = stablehlo.subtract %v1017, %v1018 : tensor<256x512x7x7xf32>
    %v1020 = stablehlo.broadcast_in_dim %s4b0n2var, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1021 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1022 = stablehlo.add %v1020, %v1021 : tensor<256x512x7x7xf32>
    %v1023 = stablehlo.rsqrt %v1022 : tensor<256x512x7x7xf32>
    %v1024 = stablehlo.multiply %v1019, %v1023 : tensor<256x512x7x7xf32>
    %v1025 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1026 = stablehlo.broadcast_in_dim %s4b0bt2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1027 = stablehlo.multiply %v1024, %v1025 : tensor<256x512x7x7xf32>
    %v1028 = stablehlo.add %v1027, %v1026 : tensor<256x512x7x7xf32>
    %v1029 = stablehlo.reshape %v1028 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1030 = stablehlo.reshape %v1029 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1031 = stablehlo.constant dense<0.0> : tensor<256x512x7x7xf32>
    %v1032 = stablehlo.maximum %v1030, %v1031 : tensor<256x512x7x7xf32>
    %v1033 = stablehlo.reshape %v1032 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1034 = stablehlo.reshape %v1033 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1035 = stablehlo.convolution(%v1034, %s4b0W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tensor<256x2048x7x7xf32>
    %v1036 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1037 = stablehlo.add %v1035, %v1036 : tensor<256x2048x7x7xf32>
    %v1038 = stablehlo.reshape %v1037 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1039 = stablehlo.reshape %v1038 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1040 = stablehlo.broadcast_in_dim %s4b0n3mu, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1041 = stablehlo.subtract %v1039, %v1040 : tensor<256x2048x7x7xf32>
    %v1042 = stablehlo.broadcast_in_dim %s4b0n3var, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1043 = stablehlo.constant dense<1.0e-05> : tensor<256x2048x7x7xf32>
    %v1044 = stablehlo.add %v1042, %v1043 : tensor<256x2048x7x7xf32>
    %v1045 = stablehlo.rsqrt %v1044 : tensor<256x2048x7x7xf32>
    %v1046 = stablehlo.multiply %v1041, %v1045 : tensor<256x2048x7x7xf32>
    %v1047 = stablehlo.broadcast_in_dim %s4b0g3, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1048 = stablehlo.broadcast_in_dim %s4b0bt3, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1049 = stablehlo.multiply %v1046, %v1047 : tensor<256x2048x7x7xf32>
    %v1050 = stablehlo.add %v1049, %v1048 : tensor<256x2048x7x7xf32>
    %v1051 = stablehlo.reshape %v1050 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1052 = stablehlo.reshape %v989 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v1053 = stablehlo.convolution(%v1052, %s4b0Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<2048x1024x1x1xf32>) -> tensor<256x2048x7x7xf32>
    %v1054 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1055 = stablehlo.add %v1053, %v1054 : tensor<256x2048x7x7xf32>
    %v1056 = stablehlo.reshape %v1055 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1057 = stablehlo.reshape %v1056 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1058 = stablehlo.broadcast_in_dim %s4b0npmu, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1059 = stablehlo.subtract %v1057, %v1058 : tensor<256x2048x7x7xf32>
    %v1060 = stablehlo.broadcast_in_dim %s4b0npvar, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1061 = stablehlo.constant dense<1.0e-05> : tensor<256x2048x7x7xf32>
    %v1062 = stablehlo.add %v1060, %v1061 : tensor<256x2048x7x7xf32>
    %v1063 = stablehlo.rsqrt %v1062 : tensor<256x2048x7x7xf32>
    %v1064 = stablehlo.multiply %v1059, %v1063 : tensor<256x2048x7x7xf32>
    %v1065 = stablehlo.broadcast_in_dim %s4b0gp, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1066 = stablehlo.broadcast_in_dim %s4b0btp, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1067 = stablehlo.multiply %v1064, %v1065 : tensor<256x2048x7x7xf32>
    %v1068 = stablehlo.add %v1067, %v1066 : tensor<256x2048x7x7xf32>
    %v1069 = stablehlo.reshape %v1068 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1070 = stablehlo.reshape %v1051 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1071 = stablehlo.reshape %v1069 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1072 = stablehlo.add %v1070, %v1071 : tensor<256x2048x7x7xf32>
    %v1073 = stablehlo.reshape %v1072 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1074 = stablehlo.reshape %v1073 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1075 = stablehlo.constant dense<0.0> : tensor<256x2048x7x7xf32>
    %v1076 = stablehlo.maximum %v1074, %v1075 : tensor<256x2048x7x7xf32>
    %v1077 = stablehlo.reshape %v1076 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1078 = stablehlo.reshape %v1077 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1079 = stablehlo.convolution(%v1078, %s4b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x2048x7x7xf32>, tensor<512x2048x1x1xf32>) -> tensor<256x512x7x7xf32>
    %v1080 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1081 = stablehlo.add %v1079, %v1080 : tensor<256x512x7x7xf32>
    %v1082 = stablehlo.reshape %v1081 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1083 = stablehlo.reshape %v1082 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1084 = stablehlo.broadcast_in_dim %s4b1n1mu, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1085 = stablehlo.subtract %v1083, %v1084 : tensor<256x512x7x7xf32>
    %v1086 = stablehlo.broadcast_in_dim %s4b1n1var, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1087 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1088 = stablehlo.add %v1086, %v1087 : tensor<256x512x7x7xf32>
    %v1089 = stablehlo.rsqrt %v1088 : tensor<256x512x7x7xf32>
    %v1090 = stablehlo.multiply %v1085, %v1089 : tensor<256x512x7x7xf32>
    %v1091 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1092 = stablehlo.broadcast_in_dim %s4b1bt1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1093 = stablehlo.multiply %v1090, %v1091 : tensor<256x512x7x7xf32>
    %v1094 = stablehlo.add %v1093, %v1092 : tensor<256x512x7x7xf32>
    %v1095 = stablehlo.reshape %v1094 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1096 = stablehlo.reshape %v1095 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1097 = stablehlo.constant dense<0.0> : tensor<256x512x7x7xf32>
    %v1098 = stablehlo.maximum %v1096, %v1097 : tensor<256x512x7x7xf32>
    %v1099 = stablehlo.reshape %v1098 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1100 = stablehlo.reshape %v1099 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1101 = stablehlo.convolution(%v1100, %s4b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v1102 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1103 = stablehlo.add %v1101, %v1102 : tensor<256x512x7x7xf32>
    %v1104 = stablehlo.reshape %v1103 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1105 = stablehlo.reshape %v1104 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1106 = stablehlo.broadcast_in_dim %s4b1n2mu, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1107 = stablehlo.subtract %v1105, %v1106 : tensor<256x512x7x7xf32>
    %v1108 = stablehlo.broadcast_in_dim %s4b1n2var, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1109 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1110 = stablehlo.add %v1108, %v1109 : tensor<256x512x7x7xf32>
    %v1111 = stablehlo.rsqrt %v1110 : tensor<256x512x7x7xf32>
    %v1112 = stablehlo.multiply %v1107, %v1111 : tensor<256x512x7x7xf32>
    %v1113 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1114 = stablehlo.broadcast_in_dim %s4b1bt2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1115 = stablehlo.multiply %v1112, %v1113 : tensor<256x512x7x7xf32>
    %v1116 = stablehlo.add %v1115, %v1114 : tensor<256x512x7x7xf32>
    %v1117 = stablehlo.reshape %v1116 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1118 = stablehlo.reshape %v1117 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1119 = stablehlo.constant dense<0.0> : tensor<256x512x7x7xf32>
    %v1120 = stablehlo.maximum %v1118, %v1119 : tensor<256x512x7x7xf32>
    %v1121 = stablehlo.reshape %v1120 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1122 = stablehlo.reshape %v1121 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1123 = stablehlo.convolution(%v1122, %s4b1W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tensor<256x2048x7x7xf32>
    %v1124 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1125 = stablehlo.add %v1123, %v1124 : tensor<256x2048x7x7xf32>
    %v1126 = stablehlo.reshape %v1125 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1127 = stablehlo.reshape %v1126 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1128 = stablehlo.broadcast_in_dim %s4b1n3mu, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1129 = stablehlo.subtract %v1127, %v1128 : tensor<256x2048x7x7xf32>
    %v1130 = stablehlo.broadcast_in_dim %s4b1n3var, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1131 = stablehlo.constant dense<1.0e-05> : tensor<256x2048x7x7xf32>
    %v1132 = stablehlo.add %v1130, %v1131 : tensor<256x2048x7x7xf32>
    %v1133 = stablehlo.rsqrt %v1132 : tensor<256x2048x7x7xf32>
    %v1134 = stablehlo.multiply %v1129, %v1133 : tensor<256x2048x7x7xf32>
    %v1135 = stablehlo.broadcast_in_dim %s4b1g3, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1136 = stablehlo.broadcast_in_dim %s4b1bt3, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1137 = stablehlo.multiply %v1134, %v1135 : tensor<256x2048x7x7xf32>
    %v1138 = stablehlo.add %v1137, %v1136 : tensor<256x2048x7x7xf32>
    %v1139 = stablehlo.reshape %v1138 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1140 = stablehlo.reshape %v1139 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1141 = stablehlo.reshape %v1077 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1142 = stablehlo.add %v1140, %v1141 : tensor<256x2048x7x7xf32>
    %v1143 = stablehlo.reshape %v1142 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1144 = stablehlo.reshape %v1143 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1145 = stablehlo.constant dense<0.0> : tensor<256x2048x7x7xf32>
    %v1146 = stablehlo.maximum %v1144, %v1145 : tensor<256x2048x7x7xf32>
    %v1147 = stablehlo.reshape %v1146 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1148 = stablehlo.reshape %v1147 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1149 = stablehlo.convolution(%v1148, %s4b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x2048x7x7xf32>, tensor<512x2048x1x1xf32>) -> tensor<256x512x7x7xf32>
    %v1150 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1151 = stablehlo.add %v1149, %v1150 : tensor<256x512x7x7xf32>
    %v1152 = stablehlo.reshape %v1151 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1153 = stablehlo.reshape %v1152 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1154 = stablehlo.broadcast_in_dim %s4b2n1mu, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1155 = stablehlo.subtract %v1153, %v1154 : tensor<256x512x7x7xf32>
    %v1156 = stablehlo.broadcast_in_dim %s4b2n1var, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1157 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1158 = stablehlo.add %v1156, %v1157 : tensor<256x512x7x7xf32>
    %v1159 = stablehlo.rsqrt %v1158 : tensor<256x512x7x7xf32>
    %v1160 = stablehlo.multiply %v1155, %v1159 : tensor<256x512x7x7xf32>
    %v1161 = stablehlo.broadcast_in_dim %s4b2g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1162 = stablehlo.broadcast_in_dim %s4b2bt1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1163 = stablehlo.multiply %v1160, %v1161 : tensor<256x512x7x7xf32>
    %v1164 = stablehlo.add %v1163, %v1162 : tensor<256x512x7x7xf32>
    %v1165 = stablehlo.reshape %v1164 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1166 = stablehlo.reshape %v1165 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1167 = stablehlo.constant dense<0.0> : tensor<256x512x7x7xf32>
    %v1168 = stablehlo.maximum %v1166, %v1167 : tensor<256x512x7x7xf32>
    %v1169 = stablehlo.reshape %v1168 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1170 = stablehlo.reshape %v1169 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1171 = stablehlo.convolution(%v1170, %s4b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v1172 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1173 = stablehlo.add %v1171, %v1172 : tensor<256x512x7x7xf32>
    %v1174 = stablehlo.reshape %v1173 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1175 = stablehlo.reshape %v1174 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1176 = stablehlo.broadcast_in_dim %s4b2n2mu, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1177 = stablehlo.subtract %v1175, %v1176 : tensor<256x512x7x7xf32>
    %v1178 = stablehlo.broadcast_in_dim %s4b2n2var, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1179 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1180 = stablehlo.add %v1178, %v1179 : tensor<256x512x7x7xf32>
    %v1181 = stablehlo.rsqrt %v1180 : tensor<256x512x7x7xf32>
    %v1182 = stablehlo.multiply %v1177, %v1181 : tensor<256x512x7x7xf32>
    %v1183 = stablehlo.broadcast_in_dim %s4b2g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1184 = stablehlo.broadcast_in_dim %s4b2bt2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1185 = stablehlo.multiply %v1182, %v1183 : tensor<256x512x7x7xf32>
    %v1186 = stablehlo.add %v1185, %v1184 : tensor<256x512x7x7xf32>
    %v1187 = stablehlo.reshape %v1186 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1188 = stablehlo.reshape %v1187 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1189 = stablehlo.constant dense<0.0> : tensor<256x512x7x7xf32>
    %v1190 = stablehlo.maximum %v1188, %v1189 : tensor<256x512x7x7xf32>
    %v1191 = stablehlo.reshape %v1190 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1192 = stablehlo.reshape %v1191 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1193 = stablehlo.convolution(%v1192, %s4b2W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tensor<256x2048x7x7xf32>
    %v1194 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1195 = stablehlo.add %v1193, %v1194 : tensor<256x2048x7x7xf32>
    %v1196 = stablehlo.reshape %v1195 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1197 = stablehlo.reshape %v1196 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1198 = stablehlo.broadcast_in_dim %s4b2n3mu, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1199 = stablehlo.subtract %v1197, %v1198 : tensor<256x2048x7x7xf32>
    %v1200 = stablehlo.broadcast_in_dim %s4b2n3var, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1201 = stablehlo.constant dense<1.0e-05> : tensor<256x2048x7x7xf32>
    %v1202 = stablehlo.add %v1200, %v1201 : tensor<256x2048x7x7xf32>
    %v1203 = stablehlo.rsqrt %v1202 : tensor<256x2048x7x7xf32>
    %v1204 = stablehlo.multiply %v1199, %v1203 : tensor<256x2048x7x7xf32>
    %v1205 = stablehlo.broadcast_in_dim %s4b2g3, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1206 = stablehlo.broadcast_in_dim %s4b2bt3, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1207 = stablehlo.multiply %v1204, %v1205 : tensor<256x2048x7x7xf32>
    %v1208 = stablehlo.add %v1207, %v1206 : tensor<256x2048x7x7xf32>
    %v1209 = stablehlo.reshape %v1208 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1210 = stablehlo.reshape %v1209 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1211 = stablehlo.reshape %v1147 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1212 = stablehlo.add %v1210, %v1211 : tensor<256x2048x7x7xf32>
    %v1213 = stablehlo.reshape %v1212 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1214 = stablehlo.reshape %v1213 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1215 = stablehlo.constant dense<0.0> : tensor<256x2048x7x7xf32>
    %v1216 = stablehlo.maximum %v1214, %v1215 : tensor<256x2048x7x7xf32>
    %v1217 = stablehlo.reshape %v1216 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1218 = stablehlo.reshape %v1217 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1219 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1220 = stablehlo.reduce(%v1218 init: %v1219) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<256x2048xf32>
    %v1221 = stablehlo.constant dense<49.0> : tensor<256x2048xf32>
    %v1222 = stablehlo.divide %v1220, %v1221 : tensor<256x2048xf32>
    %v1223 = stablehlo.dot_general %v1222, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x2048xf32>, tensor<2048x1000xf32>) -> tensor<256x1000xf32>
    %v1224 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<1000xf32>) -> tensor<256x1000xf32>
    %v1225 = stablehlo.add %v1223, %v1224 : tensor<256x1000xf32>
    return %v1225 : tensor<256x1000xf32>
  }
}
