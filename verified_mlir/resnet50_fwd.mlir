module @m {
  func.func @resnet50_fwd(%x: tensor<32x150528xf32>, %sW: tensor<64x3x7x7xf32>, %sg: tensor<64xf32>, %sbt: tensor<64xf32>, %s1b0W1: tensor<64x64x1x1xf32>, %s1b0g1: tensor<64xf32>, %s1b0bt1: tensor<64xf32>, %s1b0W2: tensor<64x64x3x3xf32>, %s1b0g2: tensor<64xf32>, %s1b0bt2: tensor<64xf32>, %s1b0W3: tensor<256x64x1x1xf32>, %s1b0g3: tensor<256xf32>, %s1b0bt3: tensor<256xf32>, %s1b0Wp: tensor<256x64x1x1xf32>, %s1b0gp: tensor<256xf32>, %s1b0btp: tensor<256xf32>, %s1b1W1: tensor<64x256x1x1xf32>, %s1b1g1: tensor<64xf32>, %s1b1bt1: tensor<64xf32>, %s1b1W2: tensor<64x64x3x3xf32>, %s1b1g2: tensor<64xf32>, %s1b1bt2: tensor<64xf32>, %s1b1W3: tensor<256x64x1x1xf32>, %s1b1g3: tensor<256xf32>, %s1b1bt3: tensor<256xf32>, %s1b2W1: tensor<64x256x1x1xf32>, %s1b2g1: tensor<64xf32>, %s1b2bt1: tensor<64xf32>, %s1b2W2: tensor<64x64x3x3xf32>, %s1b2g2: tensor<64xf32>, %s1b2bt2: tensor<64xf32>, %s1b2W3: tensor<256x64x1x1xf32>, %s1b2g3: tensor<256xf32>, %s1b2bt3: tensor<256xf32>, %s2b0W1: tensor<128x256x1x1xf32>, %s2b0g1: tensor<128xf32>, %s2b0bt1: tensor<128xf32>, %s2b0W2: tensor<128x128x3x3xf32>, %s2b0g2: tensor<128xf32>, %s2b0bt2: tensor<128xf32>, %s2b0W3: tensor<512x128x1x1xf32>, %s2b0g3: tensor<512xf32>, %s2b0bt3: tensor<512xf32>, %s2b0Wp: tensor<512x256x1x1xf32>, %s2b0gp: tensor<512xf32>, %s2b0btp: tensor<512xf32>, %s2b1W1: tensor<128x512x1x1xf32>, %s2b1g1: tensor<128xf32>, %s2b1bt1: tensor<128xf32>, %s2b1W2: tensor<128x128x3x3xf32>, %s2b1g2: tensor<128xf32>, %s2b1bt2: tensor<128xf32>, %s2b1W3: tensor<512x128x1x1xf32>, %s2b1g3: tensor<512xf32>, %s2b1bt3: tensor<512xf32>, %s2b2W1: tensor<128x512x1x1xf32>, %s2b2g1: tensor<128xf32>, %s2b2bt1: tensor<128xf32>, %s2b2W2: tensor<128x128x3x3xf32>, %s2b2g2: tensor<128xf32>, %s2b2bt2: tensor<128xf32>, %s2b2W3: tensor<512x128x1x1xf32>, %s2b2g3: tensor<512xf32>, %s2b2bt3: tensor<512xf32>, %s2b3W1: tensor<128x512x1x1xf32>, %s2b3g1: tensor<128xf32>, %s2b3bt1: tensor<128xf32>, %s2b3W2: tensor<128x128x3x3xf32>, %s2b3g2: tensor<128xf32>, %s2b3bt2: tensor<128xf32>, %s2b3W3: tensor<512x128x1x1xf32>, %s2b3g3: tensor<512xf32>, %s2b3bt3: tensor<512xf32>, %s3b0W1: tensor<256x512x1x1xf32>, %s3b0g1: tensor<256xf32>, %s3b0bt1: tensor<256xf32>, %s3b0W2: tensor<256x256x3x3xf32>, %s3b0g2: tensor<256xf32>, %s3b0bt2: tensor<256xf32>, %s3b0W3: tensor<1024x256x1x1xf32>, %s3b0g3: tensor<1024xf32>, %s3b0bt3: tensor<1024xf32>, %s3b0Wp: tensor<1024x512x1x1xf32>, %s3b0gp: tensor<1024xf32>, %s3b0btp: tensor<1024xf32>, %s3b1W1: tensor<256x1024x1x1xf32>, %s3b1g1: tensor<256xf32>, %s3b1bt1: tensor<256xf32>, %s3b1W2: tensor<256x256x3x3xf32>, %s3b1g2: tensor<256xf32>, %s3b1bt2: tensor<256xf32>, %s3b1W3: tensor<1024x256x1x1xf32>, %s3b1g3: tensor<1024xf32>, %s3b1bt3: tensor<1024xf32>, %s3b2W1: tensor<256x1024x1x1xf32>, %s3b2g1: tensor<256xf32>, %s3b2bt1: tensor<256xf32>, %s3b2W2: tensor<256x256x3x3xf32>, %s3b2g2: tensor<256xf32>, %s3b2bt2: tensor<256xf32>, %s3b2W3: tensor<1024x256x1x1xf32>, %s3b2g3: tensor<1024xf32>, %s3b2bt3: tensor<1024xf32>, %s3b3W1: tensor<256x1024x1x1xf32>, %s3b3g1: tensor<256xf32>, %s3b3bt1: tensor<256xf32>, %s3b3W2: tensor<256x256x3x3xf32>, %s3b3g2: tensor<256xf32>, %s3b3bt2: tensor<256xf32>, %s3b3W3: tensor<1024x256x1x1xf32>, %s3b3g3: tensor<1024xf32>, %s3b3bt3: tensor<1024xf32>, %s3b4W1: tensor<256x1024x1x1xf32>, %s3b4g1: tensor<256xf32>, %s3b4bt1: tensor<256xf32>, %s3b4W2: tensor<256x256x3x3xf32>, %s3b4g2: tensor<256xf32>, %s3b4bt2: tensor<256xf32>, %s3b4W3: tensor<1024x256x1x1xf32>, %s3b4g3: tensor<1024xf32>, %s3b4bt3: tensor<1024xf32>, %s3b5W1: tensor<256x1024x1x1xf32>, %s3b5g1: tensor<256xf32>, %s3b5bt1: tensor<256xf32>, %s3b5W2: tensor<256x256x3x3xf32>, %s3b5g2: tensor<256xf32>, %s3b5bt2: tensor<256xf32>, %s3b5W3: tensor<1024x256x1x1xf32>, %s3b5g3: tensor<1024xf32>, %s3b5bt3: tensor<1024xf32>, %s4b0W1: tensor<512x1024x1x1xf32>, %s4b0g1: tensor<512xf32>, %s4b0bt1: tensor<512xf32>, %s4b0W2: tensor<512x512x3x3xf32>, %s4b0g2: tensor<512xf32>, %s4b0bt2: tensor<512xf32>, %s4b0W3: tensor<2048x512x1x1xf32>, %s4b0g3: tensor<2048xf32>, %s4b0bt3: tensor<2048xf32>, %s4b0Wp: tensor<2048x1024x1x1xf32>, %s4b0gp: tensor<2048xf32>, %s4b0btp: tensor<2048xf32>, %s4b1W1: tensor<512x2048x1x1xf32>, %s4b1g1: tensor<512xf32>, %s4b1bt1: tensor<512xf32>, %s4b1W2: tensor<512x512x3x3xf32>, %s4b1g2: tensor<512xf32>, %s4b1bt2: tensor<512xf32>, %s4b1W3: tensor<2048x512x1x1xf32>, %s4b1g3: tensor<2048xf32>, %s4b1bt3: tensor<2048xf32>, %s4b2W1: tensor<512x2048x1x1xf32>, %s4b2g1: tensor<512xf32>, %s4b2bt1: tensor<512xf32>, %s4b2W2: tensor<512x512x3x3xf32>, %s4b2g2: tensor<512xf32>, %s4b2bt2: tensor<512xf32>, %s4b2W3: tensor<2048x512x1x1xf32>, %s4b2g3: tensor<2048xf32>, %s4b2bt3: tensor<2048xf32>, %Wd: tensor<2048x10xf32>, %bd: tensor<10xf32>) -> tensor<32x10xf32> {
    // ── ResNet-50 forward: every line is pretty(verified AST node) ──
    // §2l step B: the conv biases are gone from the signature (BN removes them; He et al.'s
    // `.convBn` has none). The proven conv ops still take a bias operand, so it is bound to a
    // zero constant here — same op, `bias = 0`, and `x + 0.0` is exact.
    %zb64 = stablehlo.constant dense<0.0> : tensor<64xf32>
    %zb128 = stablehlo.constant dense<0.0> : tensor<128xf32>
    %zb256 = stablehlo.constant dense<0.0> : tensor<256xf32>
    %zb512 = stablehlo.constant dense<0.0> : tensor<512xf32>
    %zb1024 = stablehlo.constant dense<0.0> : tensor<1024xf32>
    %zb2048 = stablehlo.constant dense<0.0> : tensor<2048xf32>
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
    %v7 = stablehlo.constant dense<401408.0> : tensor<32x64x112x112xf32>
    %v8 = stablehlo.constant dense<1.0e-05> : tensor<32x64x112x112xf32>
    %v9 = stablehlo.reduce(%v5 init: %v6) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v10 = stablehlo.broadcast_in_dim %v9, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v11 = stablehlo.divide %v10, %v7 : tensor<32x64x112x112xf32>
    %v12 = stablehlo.subtract %v5, %v11 : tensor<32x64x112x112xf32>
    %v13 = stablehlo.multiply %v12, %v12 : tensor<32x64x112x112xf32>
    %v14 = stablehlo.reduce(%v13 init: %v6) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v15 = stablehlo.broadcast_in_dim %v14, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
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
    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>} : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64x56x56xf32>
    %v30 = stablehlo.reshape %v29 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v32 = stablehlo.convolution(%v31, %s1b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x1x1xf32>) -> tensor<32x64x56x56xf32>
    %v33 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v34 = stablehlo.add %v32, %v33 : tensor<32x64x56x56xf32>
    %v35 = stablehlo.reshape %v34 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v36 = stablehlo.reshape %v35 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v37 = stablehlo.constant dense<0.0> : tensor<f32>
    %v38 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v39 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v40 = stablehlo.reduce(%v36 init: %v37) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v41 = stablehlo.broadcast_in_dim %v40, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v42 = stablehlo.divide %v41, %v38 : tensor<32x64x56x56xf32>
    %v43 = stablehlo.subtract %v36, %v42 : tensor<32x64x56x56xf32>
    %v44 = stablehlo.multiply %v43, %v43 : tensor<32x64x56x56xf32>
    %v45 = stablehlo.reduce(%v44 init: %v37) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v46 = stablehlo.broadcast_in_dim %v45, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
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
    %v65 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v66 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v67 = stablehlo.reduce(%v63 init: %v64) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v68 = stablehlo.broadcast_in_dim %v67, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v69 = stablehlo.divide %v68, %v65 : tensor<32x64x56x56xf32>
    %v70 = stablehlo.subtract %v63, %v69 : tensor<32x64x56x56xf32>
    %v71 = stablehlo.multiply %v70, %v70 : tensor<32x64x56x56xf32>
    %v72 = stablehlo.reduce(%v71 init: %v64) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v73 = stablehlo.broadcast_in_dim %v72, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v74 = stablehlo.divide %v73, %v65 : tensor<32x64x56x56xf32>
    %v75 = stablehlo.add %v74, %v66 : tensor<32x64x56x56xf32>
    %v76 = stablehlo.rsqrt %v75 : tensor<32x64x56x56xf32>
    %v77 = stablehlo.multiply %v70, %v76 : tensor<32x64x56x56xf32>
    %v78 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v79 = stablehlo.broadcast_in_dim %s1b0bt2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v80 = stablehlo.multiply %v77, %v78 : tensor<32x64x56x56xf32>
    %v81 = stablehlo.add %v80, %v79 : tensor<32x64x56x56xf32>
    %v82 = stablehlo.reshape %v81 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v83 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v84 = stablehlo.maximum %v82, %v83 : tensor<32x200704xf32>
    %v85 = stablehlo.reshape %v84 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v86 = stablehlo.convolution(%v85, %s1b0W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<32x256x56x56xf32>
    %v87 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x56x56xf32>
    %v88 = stablehlo.add %v86, %v87 : tensor<32x256x56x56xf32>
    %v89 = stablehlo.reshape %v88 : (tensor<32x256x56x56xf32>) -> tensor<32x802816xf32>
    %v90 = stablehlo.reshape %v89 : (tensor<32x802816xf32>) -> tensor<32x256x56x56xf32>
    %v91 = stablehlo.constant dense<0.0> : tensor<f32>
    %v92 = stablehlo.constant dense<100352.0> : tensor<32x256x56x56xf32>
    %v93 = stablehlo.constant dense<1.0e-05> : tensor<32x256x56x56xf32>
    %v94 = stablehlo.reduce(%v90 init: %v91) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v95 = stablehlo.broadcast_in_dim %v94, dims = [1] : (tensor<256xf32>) -> tensor<32x256x56x56xf32>
    %v96 = stablehlo.divide %v95, %v92 : tensor<32x256x56x56xf32>
    %v97 = stablehlo.subtract %v90, %v96 : tensor<32x256x56x56xf32>
    %v98 = stablehlo.multiply %v97, %v97 : tensor<32x256x56x56xf32>
    %v99 = stablehlo.reduce(%v98 init: %v91) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v100 = stablehlo.broadcast_in_dim %v99, dims = [1] : (tensor<256xf32>) -> tensor<32x256x56x56xf32>
    %v101 = stablehlo.divide %v100, %v92 : tensor<32x256x56x56xf32>
    %v102 = stablehlo.add %v101, %v93 : tensor<32x256x56x56xf32>
    %v103 = stablehlo.rsqrt %v102 : tensor<32x256x56x56xf32>
    %v104 = stablehlo.multiply %v97, %v103 : tensor<32x256x56x56xf32>
    %v105 = stablehlo.broadcast_in_dim %s1b0g3, dims = [1] : (tensor<256xf32>) -> tensor<32x256x56x56xf32>
    %v106 = stablehlo.broadcast_in_dim %s1b0bt3, dims = [1] : (tensor<256xf32>) -> tensor<32x256x56x56xf32>
    %v107 = stablehlo.multiply %v104, %v105 : tensor<32x256x56x56xf32>
    %v108 = stablehlo.add %v107, %v106 : tensor<32x256x56x56xf32>
    %v109 = stablehlo.reshape %v108 : (tensor<32x256x56x56xf32>) -> tensor<32x802816xf32>
    %v110 = stablehlo.reshape %v30 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v111 = stablehlo.convolution(%v110, %s1b0Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<32x256x56x56xf32>
    %v112 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x56x56xf32>
    %v113 = stablehlo.add %v111, %v112 : tensor<32x256x56x56xf32>
    %v114 = stablehlo.reshape %v113 : (tensor<32x256x56x56xf32>) -> tensor<32x802816xf32>
    %v115 = stablehlo.reshape %v114 : (tensor<32x802816xf32>) -> tensor<32x256x56x56xf32>
    %v116 = stablehlo.constant dense<0.0> : tensor<f32>
    %v117 = stablehlo.constant dense<100352.0> : tensor<32x256x56x56xf32>
    %v118 = stablehlo.constant dense<1.0e-05> : tensor<32x256x56x56xf32>
    %v119 = stablehlo.reduce(%v115 init: %v116) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v120 = stablehlo.broadcast_in_dim %v119, dims = [1] : (tensor<256xf32>) -> tensor<32x256x56x56xf32>
    %v121 = stablehlo.divide %v120, %v117 : tensor<32x256x56x56xf32>
    %v122 = stablehlo.subtract %v115, %v121 : tensor<32x256x56x56xf32>
    %v123 = stablehlo.multiply %v122, %v122 : tensor<32x256x56x56xf32>
    %v124 = stablehlo.reduce(%v123 init: %v116) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v125 = stablehlo.broadcast_in_dim %v124, dims = [1] : (tensor<256xf32>) -> tensor<32x256x56x56xf32>
    %v126 = stablehlo.divide %v125, %v117 : tensor<32x256x56x56xf32>
    %v127 = stablehlo.add %v126, %v118 : tensor<32x256x56x56xf32>
    %v128 = stablehlo.rsqrt %v127 : tensor<32x256x56x56xf32>
    %v129 = stablehlo.multiply %v122, %v128 : tensor<32x256x56x56xf32>
    %v130 = stablehlo.broadcast_in_dim %s1b0gp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x56x56xf32>
    %v131 = stablehlo.broadcast_in_dim %s1b0btp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x56x56xf32>
    %v132 = stablehlo.multiply %v129, %v130 : tensor<32x256x56x56xf32>
    %v133 = stablehlo.add %v132, %v131 : tensor<32x256x56x56xf32>
    %v134 = stablehlo.reshape %v133 : (tensor<32x256x56x56xf32>) -> tensor<32x802816xf32>
    %v135 = stablehlo.add %v109, %v134 : tensor<32x802816xf32>
    %v136 = stablehlo.constant dense<0.0> : tensor<32x802816xf32>
    %v137 = stablehlo.maximum %v135, %v136 : tensor<32x802816xf32>
    %v138 = stablehlo.reshape %v137 : (tensor<32x802816xf32>) -> tensor<32x256x56x56xf32>
    %v139 = stablehlo.convolution(%v138, %s1b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tensor<32x64x56x56xf32>
    %v140 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v141 = stablehlo.add %v139, %v140 : tensor<32x64x56x56xf32>
    %v142 = stablehlo.reshape %v141 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v143 = stablehlo.reshape %v142 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v144 = stablehlo.constant dense<0.0> : tensor<f32>
    %v145 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v146 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v147 = stablehlo.reduce(%v143 init: %v144) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v148 = stablehlo.broadcast_in_dim %v147, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v149 = stablehlo.divide %v148, %v145 : tensor<32x64x56x56xf32>
    %v150 = stablehlo.subtract %v143, %v149 : tensor<32x64x56x56xf32>
    %v151 = stablehlo.multiply %v150, %v150 : tensor<32x64x56x56xf32>
    %v152 = stablehlo.reduce(%v151 init: %v144) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v153 = stablehlo.broadcast_in_dim %v152, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v154 = stablehlo.divide %v153, %v145 : tensor<32x64x56x56xf32>
    %v155 = stablehlo.add %v154, %v146 : tensor<32x64x56x56xf32>
    %v156 = stablehlo.rsqrt %v155 : tensor<32x64x56x56xf32>
    %v157 = stablehlo.multiply %v150, %v156 : tensor<32x64x56x56xf32>
    %v158 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v159 = stablehlo.broadcast_in_dim %s1b1bt1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v160 = stablehlo.multiply %v157, %v158 : tensor<32x64x56x56xf32>
    %v161 = stablehlo.add %v160, %v159 : tensor<32x64x56x56xf32>
    %v162 = stablehlo.reshape %v161 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v163 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v164 = stablehlo.maximum %v162, %v163 : tensor<32x200704xf32>
    %v165 = stablehlo.reshape %v164 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v166 = stablehlo.convolution(%v165, %s1b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v167 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v168 = stablehlo.add %v166, %v167 : tensor<32x64x56x56xf32>
    %v169 = stablehlo.reshape %v168 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v170 = stablehlo.reshape %v169 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v171 = stablehlo.constant dense<0.0> : tensor<f32>
    %v172 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v173 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v174 = stablehlo.reduce(%v170 init: %v171) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v175 = stablehlo.broadcast_in_dim %v174, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v176 = stablehlo.divide %v175, %v172 : tensor<32x64x56x56xf32>
    %v177 = stablehlo.subtract %v170, %v176 : tensor<32x64x56x56xf32>
    %v178 = stablehlo.multiply %v177, %v177 : tensor<32x64x56x56xf32>
    %v179 = stablehlo.reduce(%v178 init: %v171) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v180 = stablehlo.broadcast_in_dim %v179, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v181 = stablehlo.divide %v180, %v172 : tensor<32x64x56x56xf32>
    %v182 = stablehlo.add %v181, %v173 : tensor<32x64x56x56xf32>
    %v183 = stablehlo.rsqrt %v182 : tensor<32x64x56x56xf32>
    %v184 = stablehlo.multiply %v177, %v183 : tensor<32x64x56x56xf32>
    %v185 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v186 = stablehlo.broadcast_in_dim %s1b1bt2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v187 = stablehlo.multiply %v184, %v185 : tensor<32x64x56x56xf32>
    %v188 = stablehlo.add %v187, %v186 : tensor<32x64x56x56xf32>
    %v189 = stablehlo.reshape %v188 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v190 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v191 = stablehlo.maximum %v189, %v190 : tensor<32x200704xf32>
    %v192 = stablehlo.reshape %v191 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v193 = stablehlo.convolution(%v192, %s1b1W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<32x256x56x56xf32>
    %v194 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x56x56xf32>
    %v195 = stablehlo.add %v193, %v194 : tensor<32x256x56x56xf32>
    %v196 = stablehlo.reshape %v195 : (tensor<32x256x56x56xf32>) -> tensor<32x802816xf32>
    %v197 = stablehlo.reshape %v196 : (tensor<32x802816xf32>) -> tensor<32x256x56x56xf32>
    %v198 = stablehlo.constant dense<0.0> : tensor<f32>
    %v199 = stablehlo.constant dense<100352.0> : tensor<32x256x56x56xf32>
    %v200 = stablehlo.constant dense<1.0e-05> : tensor<32x256x56x56xf32>
    %v201 = stablehlo.reduce(%v197 init: %v198) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v202 = stablehlo.broadcast_in_dim %v201, dims = [1] : (tensor<256xf32>) -> tensor<32x256x56x56xf32>
    %v203 = stablehlo.divide %v202, %v199 : tensor<32x256x56x56xf32>
    %v204 = stablehlo.subtract %v197, %v203 : tensor<32x256x56x56xf32>
    %v205 = stablehlo.multiply %v204, %v204 : tensor<32x256x56x56xf32>
    %v206 = stablehlo.reduce(%v205 init: %v198) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v207 = stablehlo.broadcast_in_dim %v206, dims = [1] : (tensor<256xf32>) -> tensor<32x256x56x56xf32>
    %v208 = stablehlo.divide %v207, %v199 : tensor<32x256x56x56xf32>
    %v209 = stablehlo.add %v208, %v200 : tensor<32x256x56x56xf32>
    %v210 = stablehlo.rsqrt %v209 : tensor<32x256x56x56xf32>
    %v211 = stablehlo.multiply %v204, %v210 : tensor<32x256x56x56xf32>
    %v212 = stablehlo.broadcast_in_dim %s1b1g3, dims = [1] : (tensor<256xf32>) -> tensor<32x256x56x56xf32>
    %v213 = stablehlo.broadcast_in_dim %s1b1bt3, dims = [1] : (tensor<256xf32>) -> tensor<32x256x56x56xf32>
    %v214 = stablehlo.multiply %v211, %v212 : tensor<32x256x56x56xf32>
    %v215 = stablehlo.add %v214, %v213 : tensor<32x256x56x56xf32>
    %v216 = stablehlo.reshape %v215 : (tensor<32x256x56x56xf32>) -> tensor<32x802816xf32>
    %v217 = stablehlo.add %v216, %v137 : tensor<32x802816xf32>
    %v218 = stablehlo.constant dense<0.0> : tensor<32x802816xf32>
    %v219 = stablehlo.maximum %v217, %v218 : tensor<32x802816xf32>
    %v220 = stablehlo.reshape %v219 : (tensor<32x802816xf32>) -> tensor<32x256x56x56xf32>
    %v221 = stablehlo.convolution(%v220, %s1b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tensor<32x64x56x56xf32>
    %v222 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v223 = stablehlo.add %v221, %v222 : tensor<32x64x56x56xf32>
    %v224 = stablehlo.reshape %v223 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v225 = stablehlo.reshape %v224 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v226 = stablehlo.constant dense<0.0> : tensor<f32>
    %v227 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v228 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v229 = stablehlo.reduce(%v225 init: %v226) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v230 = stablehlo.broadcast_in_dim %v229, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v231 = stablehlo.divide %v230, %v227 : tensor<32x64x56x56xf32>
    %v232 = stablehlo.subtract %v225, %v231 : tensor<32x64x56x56xf32>
    %v233 = stablehlo.multiply %v232, %v232 : tensor<32x64x56x56xf32>
    %v234 = stablehlo.reduce(%v233 init: %v226) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v235 = stablehlo.broadcast_in_dim %v234, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v236 = stablehlo.divide %v235, %v227 : tensor<32x64x56x56xf32>
    %v237 = stablehlo.add %v236, %v228 : tensor<32x64x56x56xf32>
    %v238 = stablehlo.rsqrt %v237 : tensor<32x64x56x56xf32>
    %v239 = stablehlo.multiply %v232, %v238 : tensor<32x64x56x56xf32>
    %v240 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v241 = stablehlo.broadcast_in_dim %s1b2bt1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v242 = stablehlo.multiply %v239, %v240 : tensor<32x64x56x56xf32>
    %v243 = stablehlo.add %v242, %v241 : tensor<32x64x56x56xf32>
    %v244 = stablehlo.reshape %v243 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v245 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v246 = stablehlo.maximum %v244, %v245 : tensor<32x200704xf32>
    %v247 = stablehlo.reshape %v246 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v248 = stablehlo.convolution(%v247, %s1b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v249 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v250 = stablehlo.add %v248, %v249 : tensor<32x64x56x56xf32>
    %v251 = stablehlo.reshape %v250 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v252 = stablehlo.reshape %v251 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v253 = stablehlo.constant dense<0.0> : tensor<f32>
    %v254 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v255 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v256 = stablehlo.reduce(%v252 init: %v253) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v257 = stablehlo.broadcast_in_dim %v256, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v258 = stablehlo.divide %v257, %v254 : tensor<32x64x56x56xf32>
    %v259 = stablehlo.subtract %v252, %v258 : tensor<32x64x56x56xf32>
    %v260 = stablehlo.multiply %v259, %v259 : tensor<32x64x56x56xf32>
    %v261 = stablehlo.reduce(%v260 init: %v253) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v262 = stablehlo.broadcast_in_dim %v261, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v263 = stablehlo.divide %v262, %v254 : tensor<32x64x56x56xf32>
    %v264 = stablehlo.add %v263, %v255 : tensor<32x64x56x56xf32>
    %v265 = stablehlo.rsqrt %v264 : tensor<32x64x56x56xf32>
    %v266 = stablehlo.multiply %v259, %v265 : tensor<32x64x56x56xf32>
    %v267 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v268 = stablehlo.broadcast_in_dim %s1b2bt2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v269 = stablehlo.multiply %v266, %v267 : tensor<32x64x56x56xf32>
    %v270 = stablehlo.add %v269, %v268 : tensor<32x64x56x56xf32>
    %v271 = stablehlo.reshape %v270 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v272 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v273 = stablehlo.maximum %v271, %v272 : tensor<32x200704xf32>
    %v274 = stablehlo.reshape %v273 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v275 = stablehlo.convolution(%v274, %s1b2W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<32x256x56x56xf32>
    %v276 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x56x56xf32>
    %v277 = stablehlo.add %v275, %v276 : tensor<32x256x56x56xf32>
    %v278 = stablehlo.reshape %v277 : (tensor<32x256x56x56xf32>) -> tensor<32x802816xf32>
    %v279 = stablehlo.reshape %v278 : (tensor<32x802816xf32>) -> tensor<32x256x56x56xf32>
    %v280 = stablehlo.constant dense<0.0> : tensor<f32>
    %v281 = stablehlo.constant dense<100352.0> : tensor<32x256x56x56xf32>
    %v282 = stablehlo.constant dense<1.0e-05> : tensor<32x256x56x56xf32>
    %v283 = stablehlo.reduce(%v279 init: %v280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v284 = stablehlo.broadcast_in_dim %v283, dims = [1] : (tensor<256xf32>) -> tensor<32x256x56x56xf32>
    %v285 = stablehlo.divide %v284, %v281 : tensor<32x256x56x56xf32>
    %v286 = stablehlo.subtract %v279, %v285 : tensor<32x256x56x56xf32>
    %v287 = stablehlo.multiply %v286, %v286 : tensor<32x256x56x56xf32>
    %v288 = stablehlo.reduce(%v287 init: %v280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v289 = stablehlo.broadcast_in_dim %v288, dims = [1] : (tensor<256xf32>) -> tensor<32x256x56x56xf32>
    %v290 = stablehlo.divide %v289, %v281 : tensor<32x256x56x56xf32>
    %v291 = stablehlo.add %v290, %v282 : tensor<32x256x56x56xf32>
    %v292 = stablehlo.rsqrt %v291 : tensor<32x256x56x56xf32>
    %v293 = stablehlo.multiply %v286, %v292 : tensor<32x256x56x56xf32>
    %v294 = stablehlo.broadcast_in_dim %s1b2g3, dims = [1] : (tensor<256xf32>) -> tensor<32x256x56x56xf32>
    %v295 = stablehlo.broadcast_in_dim %s1b2bt3, dims = [1] : (tensor<256xf32>) -> tensor<32x256x56x56xf32>
    %v296 = stablehlo.multiply %v293, %v294 : tensor<32x256x56x56xf32>
    %v297 = stablehlo.add %v296, %v295 : tensor<32x256x56x56xf32>
    %v298 = stablehlo.reshape %v297 : (tensor<32x256x56x56xf32>) -> tensor<32x802816xf32>
    %v299 = stablehlo.add %v298, %v219 : tensor<32x802816xf32>
    %v300 = stablehlo.constant dense<0.0> : tensor<32x802816xf32>
    %v301 = stablehlo.maximum %v299, %v300 : tensor<32x802816xf32>
    %v302 = stablehlo.reshape %v301 : (tensor<32x802816xf32>) -> tensor<32x256x56x56xf32>
    %v303 = stablehlo.convolution(%v302, %s2b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x56x56xf32>, tensor<128x256x1x1xf32>) -> tensor<32x128x56x56xf32>
    %v304 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v305 = stablehlo.add %v303, %v304 : tensor<32x128x56x56xf32>
    %v306 = stablehlo.reshape %v305 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v307 = stablehlo.reshape %v306 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v308 = stablehlo.constant dense<0.0> : tensor<f32>
    %v309 = stablehlo.constant dense<100352.0> : tensor<32x128x56x56xf32>
    %v310 = stablehlo.constant dense<1.0e-05> : tensor<32x128x56x56xf32>
    %v311 = stablehlo.reduce(%v307 init: %v308) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v312 = stablehlo.broadcast_in_dim %v311, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v313 = stablehlo.divide %v312, %v309 : tensor<32x128x56x56xf32>
    %v314 = stablehlo.subtract %v307, %v313 : tensor<32x128x56x56xf32>
    %v315 = stablehlo.multiply %v314, %v314 : tensor<32x128x56x56xf32>
    %v316 = stablehlo.reduce(%v315 init: %v308) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v317 = stablehlo.broadcast_in_dim %v316, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v318 = stablehlo.divide %v317, %v309 : tensor<32x128x56x56xf32>
    %v319 = stablehlo.add %v318, %v310 : tensor<32x128x56x56xf32>
    %v320 = stablehlo.rsqrt %v319 : tensor<32x128x56x56xf32>
    %v321 = stablehlo.multiply %v314, %v320 : tensor<32x128x56x56xf32>
    %v322 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v323 = stablehlo.broadcast_in_dim %s2b0bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v324 = stablehlo.multiply %v321, %v322 : tensor<32x128x56x56xf32>
    %v325 = stablehlo.add %v324, %v323 : tensor<32x128x56x56xf32>
    %v326 = stablehlo.reshape %v325 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v327 = stablehlo.constant dense<0.0> : tensor<32x401408xf32>
    %v328 = stablehlo.maximum %v326, %v327 : tensor<32x401408xf32>
    %v329 = stablehlo.reshape %v328 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v330 = stablehlo.convolution(%v329, %s2b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x56x56xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v331 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v332 = stablehlo.add %v330, %v331 : tensor<32x128x28x28xf32>
    %v333 = stablehlo.reshape %v332 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v334 = stablehlo.reshape %v333 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v335 = stablehlo.constant dense<0.0> : tensor<f32>
    %v336 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v337 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v338 = stablehlo.reduce(%v334 init: %v335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v339 = stablehlo.broadcast_in_dim %v338, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v340 = stablehlo.divide %v339, %v336 : tensor<32x128x28x28xf32>
    %v341 = stablehlo.subtract %v334, %v340 : tensor<32x128x28x28xf32>
    %v342 = stablehlo.multiply %v341, %v341 : tensor<32x128x28x28xf32>
    %v343 = stablehlo.reduce(%v342 init: %v335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v344 = stablehlo.broadcast_in_dim %v343, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v345 = stablehlo.divide %v344, %v336 : tensor<32x128x28x28xf32>
    %v346 = stablehlo.add %v345, %v337 : tensor<32x128x28x28xf32>
    %v347 = stablehlo.rsqrt %v346 : tensor<32x128x28x28xf32>
    %v348 = stablehlo.multiply %v341, %v347 : tensor<32x128x28x28xf32>
    %v349 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v350 = stablehlo.broadcast_in_dim %s2b0bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v351 = stablehlo.multiply %v348, %v349 : tensor<32x128x28x28xf32>
    %v352 = stablehlo.add %v351, %v350 : tensor<32x128x28x28xf32>
    %v353 = stablehlo.reshape %v352 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v354 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v355 = stablehlo.maximum %v353, %v354 : tensor<32x100352xf32>
    %v356 = stablehlo.reshape %v355 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v357 = stablehlo.convolution(%v356, %s2b0W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<32x512x28x28xf32>
    %v358 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x28x28xf32>
    %v359 = stablehlo.add %v357, %v358 : tensor<32x512x28x28xf32>
    %v360 = stablehlo.reshape %v359 : (tensor<32x512x28x28xf32>) -> tensor<32x401408xf32>
    %v361 = stablehlo.reshape %v360 : (tensor<32x401408xf32>) -> tensor<32x512x28x28xf32>
    %v362 = stablehlo.constant dense<0.0> : tensor<f32>
    %v363 = stablehlo.constant dense<25088.0> : tensor<32x512x28x28xf32>
    %v364 = stablehlo.constant dense<1.0e-05> : tensor<32x512x28x28xf32>
    %v365 = stablehlo.reduce(%v361 init: %v362) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v366 = stablehlo.broadcast_in_dim %v365, dims = [1] : (tensor<512xf32>) -> tensor<32x512x28x28xf32>
    %v367 = stablehlo.divide %v366, %v363 : tensor<32x512x28x28xf32>
    %v368 = stablehlo.subtract %v361, %v367 : tensor<32x512x28x28xf32>
    %v369 = stablehlo.multiply %v368, %v368 : tensor<32x512x28x28xf32>
    %v370 = stablehlo.reduce(%v369 init: %v362) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v371 = stablehlo.broadcast_in_dim %v370, dims = [1] : (tensor<512xf32>) -> tensor<32x512x28x28xf32>
    %v372 = stablehlo.divide %v371, %v363 : tensor<32x512x28x28xf32>
    %v373 = stablehlo.add %v372, %v364 : tensor<32x512x28x28xf32>
    %v374 = stablehlo.rsqrt %v373 : tensor<32x512x28x28xf32>
    %v375 = stablehlo.multiply %v368, %v374 : tensor<32x512x28x28xf32>
    %v376 = stablehlo.broadcast_in_dim %s2b0g3, dims = [1] : (tensor<512xf32>) -> tensor<32x512x28x28xf32>
    %v377 = stablehlo.broadcast_in_dim %s2b0bt3, dims = [1] : (tensor<512xf32>) -> tensor<32x512x28x28xf32>
    %v378 = stablehlo.multiply %v375, %v376 : tensor<32x512x28x28xf32>
    %v379 = stablehlo.add %v378, %v377 : tensor<32x512x28x28xf32>
    %v380 = stablehlo.reshape %v379 : (tensor<32x512x28x28xf32>) -> tensor<32x401408xf32>
    %v381 = stablehlo.reshape %v301 : (tensor<32x802816xf32>) -> tensor<32x256x56x56xf32>
    %v382 = stablehlo.convolution(%v381, %s2b0Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x56x56xf32>, tensor<512x256x1x1xf32>) -> tensor<32x512x28x28xf32>
    %v383 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x28x28xf32>
    %v384 = stablehlo.add %v382, %v383 : tensor<32x512x28x28xf32>
    %v385 = stablehlo.reshape %v384 : (tensor<32x512x28x28xf32>) -> tensor<32x401408xf32>
    %v386 = stablehlo.reshape %v385 : (tensor<32x401408xf32>) -> tensor<32x512x28x28xf32>
    %v387 = stablehlo.constant dense<0.0> : tensor<f32>
    %v388 = stablehlo.constant dense<25088.0> : tensor<32x512x28x28xf32>
    %v389 = stablehlo.constant dense<1.0e-05> : tensor<32x512x28x28xf32>
    %v390 = stablehlo.reduce(%v386 init: %v387) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v391 = stablehlo.broadcast_in_dim %v390, dims = [1] : (tensor<512xf32>) -> tensor<32x512x28x28xf32>
    %v392 = stablehlo.divide %v391, %v388 : tensor<32x512x28x28xf32>
    %v393 = stablehlo.subtract %v386, %v392 : tensor<32x512x28x28xf32>
    %v394 = stablehlo.multiply %v393, %v393 : tensor<32x512x28x28xf32>
    %v395 = stablehlo.reduce(%v394 init: %v387) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v396 = stablehlo.broadcast_in_dim %v395, dims = [1] : (tensor<512xf32>) -> tensor<32x512x28x28xf32>
    %v397 = stablehlo.divide %v396, %v388 : tensor<32x512x28x28xf32>
    %v398 = stablehlo.add %v397, %v389 : tensor<32x512x28x28xf32>
    %v399 = stablehlo.rsqrt %v398 : tensor<32x512x28x28xf32>
    %v400 = stablehlo.multiply %v393, %v399 : tensor<32x512x28x28xf32>
    %v401 = stablehlo.broadcast_in_dim %s2b0gp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x28x28xf32>
    %v402 = stablehlo.broadcast_in_dim %s2b0btp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x28x28xf32>
    %v403 = stablehlo.multiply %v400, %v401 : tensor<32x512x28x28xf32>
    %v404 = stablehlo.add %v403, %v402 : tensor<32x512x28x28xf32>
    %v405 = stablehlo.reshape %v404 : (tensor<32x512x28x28xf32>) -> tensor<32x401408xf32>
    %v406 = stablehlo.add %v380, %v405 : tensor<32x401408xf32>
    %v407 = stablehlo.constant dense<0.0> : tensor<32x401408xf32>
    %v408 = stablehlo.maximum %v406, %v407 : tensor<32x401408xf32>
    %v409 = stablehlo.reshape %v408 : (tensor<32x401408xf32>) -> tensor<32x512x28x28xf32>
    %v410 = stablehlo.convolution(%v409, %s2b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<32x128x28x28xf32>
    %v411 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v412 = stablehlo.add %v410, %v411 : tensor<32x128x28x28xf32>
    %v413 = stablehlo.reshape %v412 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v414 = stablehlo.reshape %v413 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v415 = stablehlo.constant dense<0.0> : tensor<f32>
    %v416 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v417 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v418 = stablehlo.reduce(%v414 init: %v415) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v419 = stablehlo.broadcast_in_dim %v418, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v420 = stablehlo.divide %v419, %v416 : tensor<32x128x28x28xf32>
    %v421 = stablehlo.subtract %v414, %v420 : tensor<32x128x28x28xf32>
    %v422 = stablehlo.multiply %v421, %v421 : tensor<32x128x28x28xf32>
    %v423 = stablehlo.reduce(%v422 init: %v415) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v424 = stablehlo.broadcast_in_dim %v423, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v425 = stablehlo.divide %v424, %v416 : tensor<32x128x28x28xf32>
    %v426 = stablehlo.add %v425, %v417 : tensor<32x128x28x28xf32>
    %v427 = stablehlo.rsqrt %v426 : tensor<32x128x28x28xf32>
    %v428 = stablehlo.multiply %v421, %v427 : tensor<32x128x28x28xf32>
    %v429 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v430 = stablehlo.broadcast_in_dim %s2b1bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v431 = stablehlo.multiply %v428, %v429 : tensor<32x128x28x28xf32>
    %v432 = stablehlo.add %v431, %v430 : tensor<32x128x28x28xf32>
    %v433 = stablehlo.reshape %v432 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v434 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v435 = stablehlo.maximum %v433, %v434 : tensor<32x100352xf32>
    %v436 = stablehlo.reshape %v435 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v437 = stablehlo.convolution(%v436, %s2b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v438 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v439 = stablehlo.add %v437, %v438 : tensor<32x128x28x28xf32>
    %v440 = stablehlo.reshape %v439 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v441 = stablehlo.reshape %v440 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v442 = stablehlo.constant dense<0.0> : tensor<f32>
    %v443 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v444 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v445 = stablehlo.reduce(%v441 init: %v442) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v446 = stablehlo.broadcast_in_dim %v445, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v447 = stablehlo.divide %v446, %v443 : tensor<32x128x28x28xf32>
    %v448 = stablehlo.subtract %v441, %v447 : tensor<32x128x28x28xf32>
    %v449 = stablehlo.multiply %v448, %v448 : tensor<32x128x28x28xf32>
    %v450 = stablehlo.reduce(%v449 init: %v442) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v451 = stablehlo.broadcast_in_dim %v450, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v452 = stablehlo.divide %v451, %v443 : tensor<32x128x28x28xf32>
    %v453 = stablehlo.add %v452, %v444 : tensor<32x128x28x28xf32>
    %v454 = stablehlo.rsqrt %v453 : tensor<32x128x28x28xf32>
    %v455 = stablehlo.multiply %v448, %v454 : tensor<32x128x28x28xf32>
    %v456 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v457 = stablehlo.broadcast_in_dim %s2b1bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v458 = stablehlo.multiply %v455, %v456 : tensor<32x128x28x28xf32>
    %v459 = stablehlo.add %v458, %v457 : tensor<32x128x28x28xf32>
    %v460 = stablehlo.reshape %v459 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v461 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v462 = stablehlo.maximum %v460, %v461 : tensor<32x100352xf32>
    %v463 = stablehlo.reshape %v462 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v464 = stablehlo.convolution(%v463, %s2b1W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<32x512x28x28xf32>
    %v465 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x28x28xf32>
    %v466 = stablehlo.add %v464, %v465 : tensor<32x512x28x28xf32>
    %v467 = stablehlo.reshape %v466 : (tensor<32x512x28x28xf32>) -> tensor<32x401408xf32>
    %v468 = stablehlo.reshape %v467 : (tensor<32x401408xf32>) -> tensor<32x512x28x28xf32>
    %v469 = stablehlo.constant dense<0.0> : tensor<f32>
    %v470 = stablehlo.constant dense<25088.0> : tensor<32x512x28x28xf32>
    %v471 = stablehlo.constant dense<1.0e-05> : tensor<32x512x28x28xf32>
    %v472 = stablehlo.reduce(%v468 init: %v469) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v473 = stablehlo.broadcast_in_dim %v472, dims = [1] : (tensor<512xf32>) -> tensor<32x512x28x28xf32>
    %v474 = stablehlo.divide %v473, %v470 : tensor<32x512x28x28xf32>
    %v475 = stablehlo.subtract %v468, %v474 : tensor<32x512x28x28xf32>
    %v476 = stablehlo.multiply %v475, %v475 : tensor<32x512x28x28xf32>
    %v477 = stablehlo.reduce(%v476 init: %v469) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v478 = stablehlo.broadcast_in_dim %v477, dims = [1] : (tensor<512xf32>) -> tensor<32x512x28x28xf32>
    %v479 = stablehlo.divide %v478, %v470 : tensor<32x512x28x28xf32>
    %v480 = stablehlo.add %v479, %v471 : tensor<32x512x28x28xf32>
    %v481 = stablehlo.rsqrt %v480 : tensor<32x512x28x28xf32>
    %v482 = stablehlo.multiply %v475, %v481 : tensor<32x512x28x28xf32>
    %v483 = stablehlo.broadcast_in_dim %s2b1g3, dims = [1] : (tensor<512xf32>) -> tensor<32x512x28x28xf32>
    %v484 = stablehlo.broadcast_in_dim %s2b1bt3, dims = [1] : (tensor<512xf32>) -> tensor<32x512x28x28xf32>
    %v485 = stablehlo.multiply %v482, %v483 : tensor<32x512x28x28xf32>
    %v486 = stablehlo.add %v485, %v484 : tensor<32x512x28x28xf32>
    %v487 = stablehlo.reshape %v486 : (tensor<32x512x28x28xf32>) -> tensor<32x401408xf32>
    %v488 = stablehlo.add %v487, %v408 : tensor<32x401408xf32>
    %v489 = stablehlo.constant dense<0.0> : tensor<32x401408xf32>
    %v490 = stablehlo.maximum %v488, %v489 : tensor<32x401408xf32>
    %v491 = stablehlo.reshape %v490 : (tensor<32x401408xf32>) -> tensor<32x512x28x28xf32>
    %v492 = stablehlo.convolution(%v491, %s2b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<32x128x28x28xf32>
    %v493 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v494 = stablehlo.add %v492, %v493 : tensor<32x128x28x28xf32>
    %v495 = stablehlo.reshape %v494 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v496 = stablehlo.reshape %v495 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v497 = stablehlo.constant dense<0.0> : tensor<f32>
    %v498 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v499 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v500 = stablehlo.reduce(%v496 init: %v497) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v501 = stablehlo.broadcast_in_dim %v500, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v502 = stablehlo.divide %v501, %v498 : tensor<32x128x28x28xf32>
    %v503 = stablehlo.subtract %v496, %v502 : tensor<32x128x28x28xf32>
    %v504 = stablehlo.multiply %v503, %v503 : tensor<32x128x28x28xf32>
    %v505 = stablehlo.reduce(%v504 init: %v497) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v506 = stablehlo.broadcast_in_dim %v505, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v507 = stablehlo.divide %v506, %v498 : tensor<32x128x28x28xf32>
    %v508 = stablehlo.add %v507, %v499 : tensor<32x128x28x28xf32>
    %v509 = stablehlo.rsqrt %v508 : tensor<32x128x28x28xf32>
    %v510 = stablehlo.multiply %v503, %v509 : tensor<32x128x28x28xf32>
    %v511 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v512 = stablehlo.broadcast_in_dim %s2b2bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v513 = stablehlo.multiply %v510, %v511 : tensor<32x128x28x28xf32>
    %v514 = stablehlo.add %v513, %v512 : tensor<32x128x28x28xf32>
    %v515 = stablehlo.reshape %v514 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v516 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v517 = stablehlo.maximum %v515, %v516 : tensor<32x100352xf32>
    %v518 = stablehlo.reshape %v517 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v519 = stablehlo.convolution(%v518, %s2b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v520 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v521 = stablehlo.add %v519, %v520 : tensor<32x128x28x28xf32>
    %v522 = stablehlo.reshape %v521 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v523 = stablehlo.reshape %v522 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v524 = stablehlo.constant dense<0.0> : tensor<f32>
    %v525 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v526 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v527 = stablehlo.reduce(%v523 init: %v524) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v528 = stablehlo.broadcast_in_dim %v527, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v529 = stablehlo.divide %v528, %v525 : tensor<32x128x28x28xf32>
    %v530 = stablehlo.subtract %v523, %v529 : tensor<32x128x28x28xf32>
    %v531 = stablehlo.multiply %v530, %v530 : tensor<32x128x28x28xf32>
    %v532 = stablehlo.reduce(%v531 init: %v524) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v533 = stablehlo.broadcast_in_dim %v532, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v534 = stablehlo.divide %v533, %v525 : tensor<32x128x28x28xf32>
    %v535 = stablehlo.add %v534, %v526 : tensor<32x128x28x28xf32>
    %v536 = stablehlo.rsqrt %v535 : tensor<32x128x28x28xf32>
    %v537 = stablehlo.multiply %v530, %v536 : tensor<32x128x28x28xf32>
    %v538 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v539 = stablehlo.broadcast_in_dim %s2b2bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v540 = stablehlo.multiply %v537, %v538 : tensor<32x128x28x28xf32>
    %v541 = stablehlo.add %v540, %v539 : tensor<32x128x28x28xf32>
    %v542 = stablehlo.reshape %v541 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v543 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v544 = stablehlo.maximum %v542, %v543 : tensor<32x100352xf32>
    %v545 = stablehlo.reshape %v544 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v546 = stablehlo.convolution(%v545, %s2b2W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<32x512x28x28xf32>
    %v547 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x28x28xf32>
    %v548 = stablehlo.add %v546, %v547 : tensor<32x512x28x28xf32>
    %v549 = stablehlo.reshape %v548 : (tensor<32x512x28x28xf32>) -> tensor<32x401408xf32>
    %v550 = stablehlo.reshape %v549 : (tensor<32x401408xf32>) -> tensor<32x512x28x28xf32>
    %v551 = stablehlo.constant dense<0.0> : tensor<f32>
    %v552 = stablehlo.constant dense<25088.0> : tensor<32x512x28x28xf32>
    %v553 = stablehlo.constant dense<1.0e-05> : tensor<32x512x28x28xf32>
    %v554 = stablehlo.reduce(%v550 init: %v551) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v555 = stablehlo.broadcast_in_dim %v554, dims = [1] : (tensor<512xf32>) -> tensor<32x512x28x28xf32>
    %v556 = stablehlo.divide %v555, %v552 : tensor<32x512x28x28xf32>
    %v557 = stablehlo.subtract %v550, %v556 : tensor<32x512x28x28xf32>
    %v558 = stablehlo.multiply %v557, %v557 : tensor<32x512x28x28xf32>
    %v559 = stablehlo.reduce(%v558 init: %v551) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v560 = stablehlo.broadcast_in_dim %v559, dims = [1] : (tensor<512xf32>) -> tensor<32x512x28x28xf32>
    %v561 = stablehlo.divide %v560, %v552 : tensor<32x512x28x28xf32>
    %v562 = stablehlo.add %v561, %v553 : tensor<32x512x28x28xf32>
    %v563 = stablehlo.rsqrt %v562 : tensor<32x512x28x28xf32>
    %v564 = stablehlo.multiply %v557, %v563 : tensor<32x512x28x28xf32>
    %v565 = stablehlo.broadcast_in_dim %s2b2g3, dims = [1] : (tensor<512xf32>) -> tensor<32x512x28x28xf32>
    %v566 = stablehlo.broadcast_in_dim %s2b2bt3, dims = [1] : (tensor<512xf32>) -> tensor<32x512x28x28xf32>
    %v567 = stablehlo.multiply %v564, %v565 : tensor<32x512x28x28xf32>
    %v568 = stablehlo.add %v567, %v566 : tensor<32x512x28x28xf32>
    %v569 = stablehlo.reshape %v568 : (tensor<32x512x28x28xf32>) -> tensor<32x401408xf32>
    %v570 = stablehlo.add %v569, %v490 : tensor<32x401408xf32>
    %v571 = stablehlo.constant dense<0.0> : tensor<32x401408xf32>
    %v572 = stablehlo.maximum %v570, %v571 : tensor<32x401408xf32>
    %v573 = stablehlo.reshape %v572 : (tensor<32x401408xf32>) -> tensor<32x512x28x28xf32>
    %v574 = stablehlo.convolution(%v573, %s2b3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<32x128x28x28xf32>
    %v575 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v576 = stablehlo.add %v574, %v575 : tensor<32x128x28x28xf32>
    %v577 = stablehlo.reshape %v576 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v578 = stablehlo.reshape %v577 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v579 = stablehlo.constant dense<0.0> : tensor<f32>
    %v580 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v581 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v582 = stablehlo.reduce(%v578 init: %v579) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v583 = stablehlo.broadcast_in_dim %v582, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v584 = stablehlo.divide %v583, %v580 : tensor<32x128x28x28xf32>
    %v585 = stablehlo.subtract %v578, %v584 : tensor<32x128x28x28xf32>
    %v586 = stablehlo.multiply %v585, %v585 : tensor<32x128x28x28xf32>
    %v587 = stablehlo.reduce(%v586 init: %v579) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v588 = stablehlo.broadcast_in_dim %v587, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v589 = stablehlo.divide %v588, %v580 : tensor<32x128x28x28xf32>
    %v590 = stablehlo.add %v589, %v581 : tensor<32x128x28x28xf32>
    %v591 = stablehlo.rsqrt %v590 : tensor<32x128x28x28xf32>
    %v592 = stablehlo.multiply %v585, %v591 : tensor<32x128x28x28xf32>
    %v593 = stablehlo.broadcast_in_dim %s2b3g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v594 = stablehlo.broadcast_in_dim %s2b3bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v595 = stablehlo.multiply %v592, %v593 : tensor<32x128x28x28xf32>
    %v596 = stablehlo.add %v595, %v594 : tensor<32x128x28x28xf32>
    %v597 = stablehlo.reshape %v596 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v598 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v599 = stablehlo.maximum %v597, %v598 : tensor<32x100352xf32>
    %v600 = stablehlo.reshape %v599 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v601 = stablehlo.convolution(%v600, %s2b3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v602 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v603 = stablehlo.add %v601, %v602 : tensor<32x128x28x28xf32>
    %v604 = stablehlo.reshape %v603 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v605 = stablehlo.reshape %v604 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v606 = stablehlo.constant dense<0.0> : tensor<f32>
    %v607 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v608 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v609 = stablehlo.reduce(%v605 init: %v606) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v610 = stablehlo.broadcast_in_dim %v609, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v611 = stablehlo.divide %v610, %v607 : tensor<32x128x28x28xf32>
    %v612 = stablehlo.subtract %v605, %v611 : tensor<32x128x28x28xf32>
    %v613 = stablehlo.multiply %v612, %v612 : tensor<32x128x28x28xf32>
    %v614 = stablehlo.reduce(%v613 init: %v606) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v615 = stablehlo.broadcast_in_dim %v614, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v616 = stablehlo.divide %v615, %v607 : tensor<32x128x28x28xf32>
    %v617 = stablehlo.add %v616, %v608 : tensor<32x128x28x28xf32>
    %v618 = stablehlo.rsqrt %v617 : tensor<32x128x28x28xf32>
    %v619 = stablehlo.multiply %v612, %v618 : tensor<32x128x28x28xf32>
    %v620 = stablehlo.broadcast_in_dim %s2b3g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v621 = stablehlo.broadcast_in_dim %s2b3bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v622 = stablehlo.multiply %v619, %v620 : tensor<32x128x28x28xf32>
    %v623 = stablehlo.add %v622, %v621 : tensor<32x128x28x28xf32>
    %v624 = stablehlo.reshape %v623 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v625 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v626 = stablehlo.maximum %v624, %v625 : tensor<32x100352xf32>
    %v627 = stablehlo.reshape %v626 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v628 = stablehlo.convolution(%v627, %s2b3W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<32x512x28x28xf32>
    %v629 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x28x28xf32>
    %v630 = stablehlo.add %v628, %v629 : tensor<32x512x28x28xf32>
    %v631 = stablehlo.reshape %v630 : (tensor<32x512x28x28xf32>) -> tensor<32x401408xf32>
    %v632 = stablehlo.reshape %v631 : (tensor<32x401408xf32>) -> tensor<32x512x28x28xf32>
    %v633 = stablehlo.constant dense<0.0> : tensor<f32>
    %v634 = stablehlo.constant dense<25088.0> : tensor<32x512x28x28xf32>
    %v635 = stablehlo.constant dense<1.0e-05> : tensor<32x512x28x28xf32>
    %v636 = stablehlo.reduce(%v632 init: %v633) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v637 = stablehlo.broadcast_in_dim %v636, dims = [1] : (tensor<512xf32>) -> tensor<32x512x28x28xf32>
    %v638 = stablehlo.divide %v637, %v634 : tensor<32x512x28x28xf32>
    %v639 = stablehlo.subtract %v632, %v638 : tensor<32x512x28x28xf32>
    %v640 = stablehlo.multiply %v639, %v639 : tensor<32x512x28x28xf32>
    %v641 = stablehlo.reduce(%v640 init: %v633) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v642 = stablehlo.broadcast_in_dim %v641, dims = [1] : (tensor<512xf32>) -> tensor<32x512x28x28xf32>
    %v643 = stablehlo.divide %v642, %v634 : tensor<32x512x28x28xf32>
    %v644 = stablehlo.add %v643, %v635 : tensor<32x512x28x28xf32>
    %v645 = stablehlo.rsqrt %v644 : tensor<32x512x28x28xf32>
    %v646 = stablehlo.multiply %v639, %v645 : tensor<32x512x28x28xf32>
    %v647 = stablehlo.broadcast_in_dim %s2b3g3, dims = [1] : (tensor<512xf32>) -> tensor<32x512x28x28xf32>
    %v648 = stablehlo.broadcast_in_dim %s2b3bt3, dims = [1] : (tensor<512xf32>) -> tensor<32x512x28x28xf32>
    %v649 = stablehlo.multiply %v646, %v647 : tensor<32x512x28x28xf32>
    %v650 = stablehlo.add %v649, %v648 : tensor<32x512x28x28xf32>
    %v651 = stablehlo.reshape %v650 : (tensor<32x512x28x28xf32>) -> tensor<32x401408xf32>
    %v652 = stablehlo.add %v651, %v572 : tensor<32x401408xf32>
    %v653 = stablehlo.constant dense<0.0> : tensor<32x401408xf32>
    %v654 = stablehlo.maximum %v652, %v653 : tensor<32x401408xf32>
    %v655 = stablehlo.reshape %v654 : (tensor<32x401408xf32>) -> tensor<32x512x28x28xf32>
    %v656 = stablehlo.convolution(%v655, %s3b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x28x28xf32>, tensor<256x512x1x1xf32>) -> tensor<32x256x28x28xf32>
    %v657 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x28x28xf32>
    %v658 = stablehlo.add %v656, %v657 : tensor<32x256x28x28xf32>
    %v659 = stablehlo.reshape %v658 : (tensor<32x256x28x28xf32>) -> tensor<32x200704xf32>
    %v660 = stablehlo.reshape %v659 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v661 = stablehlo.constant dense<0.0> : tensor<f32>
    %v662 = stablehlo.constant dense<25088.0> : tensor<32x256x28x28xf32>
    %v663 = stablehlo.constant dense<1.0e-05> : tensor<32x256x28x28xf32>
    %v664 = stablehlo.reduce(%v660 init: %v661) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v665 = stablehlo.broadcast_in_dim %v664, dims = [1] : (tensor<256xf32>) -> tensor<32x256x28x28xf32>
    %v666 = stablehlo.divide %v665, %v662 : tensor<32x256x28x28xf32>
    %v667 = stablehlo.subtract %v660, %v666 : tensor<32x256x28x28xf32>
    %v668 = stablehlo.multiply %v667, %v667 : tensor<32x256x28x28xf32>
    %v669 = stablehlo.reduce(%v668 init: %v661) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v670 = stablehlo.broadcast_in_dim %v669, dims = [1] : (tensor<256xf32>) -> tensor<32x256x28x28xf32>
    %v671 = stablehlo.divide %v670, %v662 : tensor<32x256x28x28xf32>
    %v672 = stablehlo.add %v671, %v663 : tensor<32x256x28x28xf32>
    %v673 = stablehlo.rsqrt %v672 : tensor<32x256x28x28xf32>
    %v674 = stablehlo.multiply %v667, %v673 : tensor<32x256x28x28xf32>
    %v675 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x28x28xf32>
    %v676 = stablehlo.broadcast_in_dim %s3b0bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x28x28xf32>
    %v677 = stablehlo.multiply %v674, %v675 : tensor<32x256x28x28xf32>
    %v678 = stablehlo.add %v677, %v676 : tensor<32x256x28x28xf32>
    %v679 = stablehlo.reshape %v678 : (tensor<32x256x28x28xf32>) -> tensor<32x200704xf32>
    %v680 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v681 = stablehlo.maximum %v679, %v680 : tensor<32x200704xf32>
    %v682 = stablehlo.reshape %v681 : (tensor<32x200704xf32>) -> tensor<32x256x28x28xf32>
    %v683 = stablehlo.convolution(%v682, %s3b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x28x28xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v684 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v685 = stablehlo.add %v683, %v684 : tensor<32x256x14x14xf32>
    %v686 = stablehlo.reshape %v685 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v687 = stablehlo.reshape %v686 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v688 = stablehlo.constant dense<0.0> : tensor<f32>
    %v689 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v690 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v691 = stablehlo.reduce(%v687 init: %v688) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v692 = stablehlo.broadcast_in_dim %v691, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v693 = stablehlo.divide %v692, %v689 : tensor<32x256x14x14xf32>
    %v694 = stablehlo.subtract %v687, %v693 : tensor<32x256x14x14xf32>
    %v695 = stablehlo.multiply %v694, %v694 : tensor<32x256x14x14xf32>
    %v696 = stablehlo.reduce(%v695 init: %v688) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v697 = stablehlo.broadcast_in_dim %v696, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v698 = stablehlo.divide %v697, %v689 : tensor<32x256x14x14xf32>
    %v699 = stablehlo.add %v698, %v690 : tensor<32x256x14x14xf32>
    %v700 = stablehlo.rsqrt %v699 : tensor<32x256x14x14xf32>
    %v701 = stablehlo.multiply %v694, %v700 : tensor<32x256x14x14xf32>
    %v702 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v703 = stablehlo.broadcast_in_dim %s3b0bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v704 = stablehlo.multiply %v701, %v702 : tensor<32x256x14x14xf32>
    %v705 = stablehlo.add %v704, %v703 : tensor<32x256x14x14xf32>
    %v706 = stablehlo.reshape %v705 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v707 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v708 = stablehlo.maximum %v706, %v707 : tensor<32x50176xf32>
    %v709 = stablehlo.reshape %v708 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v710 = stablehlo.convolution(%v709, %s3b0W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x14x14xf32>
    %v711 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x14x14xf32>
    %v712 = stablehlo.add %v710, %v711 : tensor<32x1024x14x14xf32>
    %v713 = stablehlo.reshape %v712 : (tensor<32x1024x14x14xf32>) -> tensor<32x200704xf32>
    %v714 = stablehlo.reshape %v713 : (tensor<32x200704xf32>) -> tensor<32x1024x14x14xf32>
    %v715 = stablehlo.constant dense<0.0> : tensor<f32>
    %v716 = stablehlo.constant dense<6272.0> : tensor<32x1024x14x14xf32>
    %v717 = stablehlo.constant dense<1.0e-05> : tensor<32x1024x14x14xf32>
    %v718 = stablehlo.reduce(%v714 init: %v715) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v719 = stablehlo.broadcast_in_dim %v718, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x14x14xf32>
    %v720 = stablehlo.divide %v719, %v716 : tensor<32x1024x14x14xf32>
    %v721 = stablehlo.subtract %v714, %v720 : tensor<32x1024x14x14xf32>
    %v722 = stablehlo.multiply %v721, %v721 : tensor<32x1024x14x14xf32>
    %v723 = stablehlo.reduce(%v722 init: %v715) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v724 = stablehlo.broadcast_in_dim %v723, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x14x14xf32>
    %v725 = stablehlo.divide %v724, %v716 : tensor<32x1024x14x14xf32>
    %v726 = stablehlo.add %v725, %v717 : tensor<32x1024x14x14xf32>
    %v727 = stablehlo.rsqrt %v726 : tensor<32x1024x14x14xf32>
    %v728 = stablehlo.multiply %v721, %v727 : tensor<32x1024x14x14xf32>
    %v729 = stablehlo.broadcast_in_dim %s3b0g3, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x14x14xf32>
    %v730 = stablehlo.broadcast_in_dim %s3b0bt3, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x14x14xf32>
    %v731 = stablehlo.multiply %v728, %v729 : tensor<32x1024x14x14xf32>
    %v732 = stablehlo.add %v731, %v730 : tensor<32x1024x14x14xf32>
    %v733 = stablehlo.reshape %v732 : (tensor<32x1024x14x14xf32>) -> tensor<32x200704xf32>
    %v734 = stablehlo.reshape %v654 : (tensor<32x401408xf32>) -> tensor<32x512x28x28xf32>
    %v735 = stablehlo.convolution(%v734, %s3b0Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x28x28xf32>, tensor<1024x512x1x1xf32>) -> tensor<32x1024x14x14xf32>
    %v736 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x14x14xf32>
    %v737 = stablehlo.add %v735, %v736 : tensor<32x1024x14x14xf32>
    %v738 = stablehlo.reshape %v737 : (tensor<32x1024x14x14xf32>) -> tensor<32x200704xf32>
    %v739 = stablehlo.reshape %v738 : (tensor<32x200704xf32>) -> tensor<32x1024x14x14xf32>
    %v740 = stablehlo.constant dense<0.0> : tensor<f32>
    %v741 = stablehlo.constant dense<6272.0> : tensor<32x1024x14x14xf32>
    %v742 = stablehlo.constant dense<1.0e-05> : tensor<32x1024x14x14xf32>
    %v743 = stablehlo.reduce(%v739 init: %v740) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v744 = stablehlo.broadcast_in_dim %v743, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x14x14xf32>
    %v745 = stablehlo.divide %v744, %v741 : tensor<32x1024x14x14xf32>
    %v746 = stablehlo.subtract %v739, %v745 : tensor<32x1024x14x14xf32>
    %v747 = stablehlo.multiply %v746, %v746 : tensor<32x1024x14x14xf32>
    %v748 = stablehlo.reduce(%v747 init: %v740) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v749 = stablehlo.broadcast_in_dim %v748, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x14x14xf32>
    %v750 = stablehlo.divide %v749, %v741 : tensor<32x1024x14x14xf32>
    %v751 = stablehlo.add %v750, %v742 : tensor<32x1024x14x14xf32>
    %v752 = stablehlo.rsqrt %v751 : tensor<32x1024x14x14xf32>
    %v753 = stablehlo.multiply %v746, %v752 : tensor<32x1024x14x14xf32>
    %v754 = stablehlo.broadcast_in_dim %s3b0gp, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x14x14xf32>
    %v755 = stablehlo.broadcast_in_dim %s3b0btp, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x14x14xf32>
    %v756 = stablehlo.multiply %v753, %v754 : tensor<32x1024x14x14xf32>
    %v757 = stablehlo.add %v756, %v755 : tensor<32x1024x14x14xf32>
    %v758 = stablehlo.reshape %v757 : (tensor<32x1024x14x14xf32>) -> tensor<32x200704xf32>
    %v759 = stablehlo.add %v733, %v758 : tensor<32x200704xf32>
    %v760 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v761 = stablehlo.maximum %v759, %v760 : tensor<32x200704xf32>
    %v762 = stablehlo.reshape %v761 : (tensor<32x200704xf32>) -> tensor<32x1024x14x14xf32>
    %v763 = stablehlo.convolution(%v762, %s3b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x14x14xf32>
    %v764 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v765 = stablehlo.add %v763, %v764 : tensor<32x256x14x14xf32>
    %v766 = stablehlo.reshape %v765 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v767 = stablehlo.reshape %v766 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v768 = stablehlo.constant dense<0.0> : tensor<f32>
    %v769 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v770 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v771 = stablehlo.reduce(%v767 init: %v768) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v772 = stablehlo.broadcast_in_dim %v771, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v773 = stablehlo.divide %v772, %v769 : tensor<32x256x14x14xf32>
    %v774 = stablehlo.subtract %v767, %v773 : tensor<32x256x14x14xf32>
    %v775 = stablehlo.multiply %v774, %v774 : tensor<32x256x14x14xf32>
    %v776 = stablehlo.reduce(%v775 init: %v768) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v777 = stablehlo.broadcast_in_dim %v776, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v778 = stablehlo.divide %v777, %v769 : tensor<32x256x14x14xf32>
    %v779 = stablehlo.add %v778, %v770 : tensor<32x256x14x14xf32>
    %v780 = stablehlo.rsqrt %v779 : tensor<32x256x14x14xf32>
    %v781 = stablehlo.multiply %v774, %v780 : tensor<32x256x14x14xf32>
    %v782 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v783 = stablehlo.broadcast_in_dim %s3b1bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v784 = stablehlo.multiply %v781, %v782 : tensor<32x256x14x14xf32>
    %v785 = stablehlo.add %v784, %v783 : tensor<32x256x14x14xf32>
    %v786 = stablehlo.reshape %v785 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v787 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v788 = stablehlo.maximum %v786, %v787 : tensor<32x50176xf32>
    %v789 = stablehlo.reshape %v788 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v790 = stablehlo.convolution(%v789, %s3b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v791 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v792 = stablehlo.add %v790, %v791 : tensor<32x256x14x14xf32>
    %v793 = stablehlo.reshape %v792 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v794 = stablehlo.reshape %v793 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v795 = stablehlo.constant dense<0.0> : tensor<f32>
    %v796 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v797 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v798 = stablehlo.reduce(%v794 init: %v795) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v799 = stablehlo.broadcast_in_dim %v798, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v800 = stablehlo.divide %v799, %v796 : tensor<32x256x14x14xf32>
    %v801 = stablehlo.subtract %v794, %v800 : tensor<32x256x14x14xf32>
    %v802 = stablehlo.multiply %v801, %v801 : tensor<32x256x14x14xf32>
    %v803 = stablehlo.reduce(%v802 init: %v795) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v804 = stablehlo.broadcast_in_dim %v803, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v805 = stablehlo.divide %v804, %v796 : tensor<32x256x14x14xf32>
    %v806 = stablehlo.add %v805, %v797 : tensor<32x256x14x14xf32>
    %v807 = stablehlo.rsqrt %v806 : tensor<32x256x14x14xf32>
    %v808 = stablehlo.multiply %v801, %v807 : tensor<32x256x14x14xf32>
    %v809 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v810 = stablehlo.broadcast_in_dim %s3b1bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v811 = stablehlo.multiply %v808, %v809 : tensor<32x256x14x14xf32>
    %v812 = stablehlo.add %v811, %v810 : tensor<32x256x14x14xf32>
    %v813 = stablehlo.reshape %v812 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v814 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v815 = stablehlo.maximum %v813, %v814 : tensor<32x50176xf32>
    %v816 = stablehlo.reshape %v815 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v817 = stablehlo.convolution(%v816, %s3b1W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x14x14xf32>
    %v818 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x14x14xf32>
    %v819 = stablehlo.add %v817, %v818 : tensor<32x1024x14x14xf32>
    %v820 = stablehlo.reshape %v819 : (tensor<32x1024x14x14xf32>) -> tensor<32x200704xf32>
    %v821 = stablehlo.reshape %v820 : (tensor<32x200704xf32>) -> tensor<32x1024x14x14xf32>
    %v822 = stablehlo.constant dense<0.0> : tensor<f32>
    %v823 = stablehlo.constant dense<6272.0> : tensor<32x1024x14x14xf32>
    %v824 = stablehlo.constant dense<1.0e-05> : tensor<32x1024x14x14xf32>
    %v825 = stablehlo.reduce(%v821 init: %v822) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v826 = stablehlo.broadcast_in_dim %v825, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x14x14xf32>
    %v827 = stablehlo.divide %v826, %v823 : tensor<32x1024x14x14xf32>
    %v828 = stablehlo.subtract %v821, %v827 : tensor<32x1024x14x14xf32>
    %v829 = stablehlo.multiply %v828, %v828 : tensor<32x1024x14x14xf32>
    %v830 = stablehlo.reduce(%v829 init: %v822) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v831 = stablehlo.broadcast_in_dim %v830, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x14x14xf32>
    %v832 = stablehlo.divide %v831, %v823 : tensor<32x1024x14x14xf32>
    %v833 = stablehlo.add %v832, %v824 : tensor<32x1024x14x14xf32>
    %v834 = stablehlo.rsqrt %v833 : tensor<32x1024x14x14xf32>
    %v835 = stablehlo.multiply %v828, %v834 : tensor<32x1024x14x14xf32>
    %v836 = stablehlo.broadcast_in_dim %s3b1g3, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x14x14xf32>
    %v837 = stablehlo.broadcast_in_dim %s3b1bt3, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x14x14xf32>
    %v838 = stablehlo.multiply %v835, %v836 : tensor<32x1024x14x14xf32>
    %v839 = stablehlo.add %v838, %v837 : tensor<32x1024x14x14xf32>
    %v840 = stablehlo.reshape %v839 : (tensor<32x1024x14x14xf32>) -> tensor<32x200704xf32>
    %v841 = stablehlo.add %v840, %v761 : tensor<32x200704xf32>
    %v842 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v843 = stablehlo.maximum %v841, %v842 : tensor<32x200704xf32>
    %v844 = stablehlo.reshape %v843 : (tensor<32x200704xf32>) -> tensor<32x1024x14x14xf32>
    %v845 = stablehlo.convolution(%v844, %s3b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x14x14xf32>
    %v846 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v847 = stablehlo.add %v845, %v846 : tensor<32x256x14x14xf32>
    %v848 = stablehlo.reshape %v847 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v849 = stablehlo.reshape %v848 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v850 = stablehlo.constant dense<0.0> : tensor<f32>
    %v851 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v852 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v853 = stablehlo.reduce(%v849 init: %v850) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v854 = stablehlo.broadcast_in_dim %v853, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v855 = stablehlo.divide %v854, %v851 : tensor<32x256x14x14xf32>
    %v856 = stablehlo.subtract %v849, %v855 : tensor<32x256x14x14xf32>
    %v857 = stablehlo.multiply %v856, %v856 : tensor<32x256x14x14xf32>
    %v858 = stablehlo.reduce(%v857 init: %v850) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v859 = stablehlo.broadcast_in_dim %v858, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v860 = stablehlo.divide %v859, %v851 : tensor<32x256x14x14xf32>
    %v861 = stablehlo.add %v860, %v852 : tensor<32x256x14x14xf32>
    %v862 = stablehlo.rsqrt %v861 : tensor<32x256x14x14xf32>
    %v863 = stablehlo.multiply %v856, %v862 : tensor<32x256x14x14xf32>
    %v864 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v865 = stablehlo.broadcast_in_dim %s3b2bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v866 = stablehlo.multiply %v863, %v864 : tensor<32x256x14x14xf32>
    %v867 = stablehlo.add %v866, %v865 : tensor<32x256x14x14xf32>
    %v868 = stablehlo.reshape %v867 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v869 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v870 = stablehlo.maximum %v868, %v869 : tensor<32x50176xf32>
    %v871 = stablehlo.reshape %v870 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v872 = stablehlo.convolution(%v871, %s3b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v873 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v874 = stablehlo.add %v872, %v873 : tensor<32x256x14x14xf32>
    %v875 = stablehlo.reshape %v874 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v876 = stablehlo.reshape %v875 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v877 = stablehlo.constant dense<0.0> : tensor<f32>
    %v878 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v879 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v880 = stablehlo.reduce(%v876 init: %v877) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v881 = stablehlo.broadcast_in_dim %v880, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v882 = stablehlo.divide %v881, %v878 : tensor<32x256x14x14xf32>
    %v883 = stablehlo.subtract %v876, %v882 : tensor<32x256x14x14xf32>
    %v884 = stablehlo.multiply %v883, %v883 : tensor<32x256x14x14xf32>
    %v885 = stablehlo.reduce(%v884 init: %v877) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v886 = stablehlo.broadcast_in_dim %v885, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v887 = stablehlo.divide %v886, %v878 : tensor<32x256x14x14xf32>
    %v888 = stablehlo.add %v887, %v879 : tensor<32x256x14x14xf32>
    %v889 = stablehlo.rsqrt %v888 : tensor<32x256x14x14xf32>
    %v890 = stablehlo.multiply %v883, %v889 : tensor<32x256x14x14xf32>
    %v891 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v892 = stablehlo.broadcast_in_dim %s3b2bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v893 = stablehlo.multiply %v890, %v891 : tensor<32x256x14x14xf32>
    %v894 = stablehlo.add %v893, %v892 : tensor<32x256x14x14xf32>
    %v895 = stablehlo.reshape %v894 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v896 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v897 = stablehlo.maximum %v895, %v896 : tensor<32x50176xf32>
    %v898 = stablehlo.reshape %v897 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v899 = stablehlo.convolution(%v898, %s3b2W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x14x14xf32>
    %v900 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x14x14xf32>
    %v901 = stablehlo.add %v899, %v900 : tensor<32x1024x14x14xf32>
    %v902 = stablehlo.reshape %v901 : (tensor<32x1024x14x14xf32>) -> tensor<32x200704xf32>
    %v903 = stablehlo.reshape %v902 : (tensor<32x200704xf32>) -> tensor<32x1024x14x14xf32>
    %v904 = stablehlo.constant dense<0.0> : tensor<f32>
    %v905 = stablehlo.constant dense<6272.0> : tensor<32x1024x14x14xf32>
    %v906 = stablehlo.constant dense<1.0e-05> : tensor<32x1024x14x14xf32>
    %v907 = stablehlo.reduce(%v903 init: %v904) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v908 = stablehlo.broadcast_in_dim %v907, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x14x14xf32>
    %v909 = stablehlo.divide %v908, %v905 : tensor<32x1024x14x14xf32>
    %v910 = stablehlo.subtract %v903, %v909 : tensor<32x1024x14x14xf32>
    %v911 = stablehlo.multiply %v910, %v910 : tensor<32x1024x14x14xf32>
    %v912 = stablehlo.reduce(%v911 init: %v904) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v913 = stablehlo.broadcast_in_dim %v912, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x14x14xf32>
    %v914 = stablehlo.divide %v913, %v905 : tensor<32x1024x14x14xf32>
    %v915 = stablehlo.add %v914, %v906 : tensor<32x1024x14x14xf32>
    %v916 = stablehlo.rsqrt %v915 : tensor<32x1024x14x14xf32>
    %v917 = stablehlo.multiply %v910, %v916 : tensor<32x1024x14x14xf32>
    %v918 = stablehlo.broadcast_in_dim %s3b2g3, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x14x14xf32>
    %v919 = stablehlo.broadcast_in_dim %s3b2bt3, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x14x14xf32>
    %v920 = stablehlo.multiply %v917, %v918 : tensor<32x1024x14x14xf32>
    %v921 = stablehlo.add %v920, %v919 : tensor<32x1024x14x14xf32>
    %v922 = stablehlo.reshape %v921 : (tensor<32x1024x14x14xf32>) -> tensor<32x200704xf32>
    %v923 = stablehlo.add %v922, %v843 : tensor<32x200704xf32>
    %v924 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v925 = stablehlo.maximum %v923, %v924 : tensor<32x200704xf32>
    %v926 = stablehlo.reshape %v925 : (tensor<32x200704xf32>) -> tensor<32x1024x14x14xf32>
    %v927 = stablehlo.convolution(%v926, %s3b3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x14x14xf32>
    %v928 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v929 = stablehlo.add %v927, %v928 : tensor<32x256x14x14xf32>
    %v930 = stablehlo.reshape %v929 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v931 = stablehlo.reshape %v930 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v932 = stablehlo.constant dense<0.0> : tensor<f32>
    %v933 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v934 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v935 = stablehlo.reduce(%v931 init: %v932) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v936 = stablehlo.broadcast_in_dim %v935, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v937 = stablehlo.divide %v936, %v933 : tensor<32x256x14x14xf32>
    %v938 = stablehlo.subtract %v931, %v937 : tensor<32x256x14x14xf32>
    %v939 = stablehlo.multiply %v938, %v938 : tensor<32x256x14x14xf32>
    %v940 = stablehlo.reduce(%v939 init: %v932) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v941 = stablehlo.broadcast_in_dim %v940, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v942 = stablehlo.divide %v941, %v933 : tensor<32x256x14x14xf32>
    %v943 = stablehlo.add %v942, %v934 : tensor<32x256x14x14xf32>
    %v944 = stablehlo.rsqrt %v943 : tensor<32x256x14x14xf32>
    %v945 = stablehlo.multiply %v938, %v944 : tensor<32x256x14x14xf32>
    %v946 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v947 = stablehlo.broadcast_in_dim %s3b3bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v948 = stablehlo.multiply %v945, %v946 : tensor<32x256x14x14xf32>
    %v949 = stablehlo.add %v948, %v947 : tensor<32x256x14x14xf32>
    %v950 = stablehlo.reshape %v949 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v951 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v952 = stablehlo.maximum %v950, %v951 : tensor<32x50176xf32>
    %v953 = stablehlo.reshape %v952 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v954 = stablehlo.convolution(%v953, %s3b3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v955 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v956 = stablehlo.add %v954, %v955 : tensor<32x256x14x14xf32>
    %v957 = stablehlo.reshape %v956 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v958 = stablehlo.reshape %v957 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v959 = stablehlo.constant dense<0.0> : tensor<f32>
    %v960 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v961 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v962 = stablehlo.reduce(%v958 init: %v959) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v963 = stablehlo.broadcast_in_dim %v962, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v964 = stablehlo.divide %v963, %v960 : tensor<32x256x14x14xf32>
    %v965 = stablehlo.subtract %v958, %v964 : tensor<32x256x14x14xf32>
    %v966 = stablehlo.multiply %v965, %v965 : tensor<32x256x14x14xf32>
    %v967 = stablehlo.reduce(%v966 init: %v959) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v968 = stablehlo.broadcast_in_dim %v967, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v969 = stablehlo.divide %v968, %v960 : tensor<32x256x14x14xf32>
    %v970 = stablehlo.add %v969, %v961 : tensor<32x256x14x14xf32>
    %v971 = stablehlo.rsqrt %v970 : tensor<32x256x14x14xf32>
    %v972 = stablehlo.multiply %v965, %v971 : tensor<32x256x14x14xf32>
    %v973 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v974 = stablehlo.broadcast_in_dim %s3b3bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v975 = stablehlo.multiply %v972, %v973 : tensor<32x256x14x14xf32>
    %v976 = stablehlo.add %v975, %v974 : tensor<32x256x14x14xf32>
    %v977 = stablehlo.reshape %v976 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v978 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v979 = stablehlo.maximum %v977, %v978 : tensor<32x50176xf32>
    %v980 = stablehlo.reshape %v979 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v981 = stablehlo.convolution(%v980, %s3b3W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x14x14xf32>
    %v982 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x14x14xf32>
    %v983 = stablehlo.add %v981, %v982 : tensor<32x1024x14x14xf32>
    %v984 = stablehlo.reshape %v983 : (tensor<32x1024x14x14xf32>) -> tensor<32x200704xf32>
    %v985 = stablehlo.reshape %v984 : (tensor<32x200704xf32>) -> tensor<32x1024x14x14xf32>
    %v986 = stablehlo.constant dense<0.0> : tensor<f32>
    %v987 = stablehlo.constant dense<6272.0> : tensor<32x1024x14x14xf32>
    %v988 = stablehlo.constant dense<1.0e-05> : tensor<32x1024x14x14xf32>
    %v989 = stablehlo.reduce(%v985 init: %v986) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v990 = stablehlo.broadcast_in_dim %v989, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x14x14xf32>
    %v991 = stablehlo.divide %v990, %v987 : tensor<32x1024x14x14xf32>
    %v992 = stablehlo.subtract %v985, %v991 : tensor<32x1024x14x14xf32>
    %v993 = stablehlo.multiply %v992, %v992 : tensor<32x1024x14x14xf32>
    %v994 = stablehlo.reduce(%v993 init: %v986) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v995 = stablehlo.broadcast_in_dim %v994, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x14x14xf32>
    %v996 = stablehlo.divide %v995, %v987 : tensor<32x1024x14x14xf32>
    %v997 = stablehlo.add %v996, %v988 : tensor<32x1024x14x14xf32>
    %v998 = stablehlo.rsqrt %v997 : tensor<32x1024x14x14xf32>
    %v999 = stablehlo.multiply %v992, %v998 : tensor<32x1024x14x14xf32>
    %v1000 = stablehlo.broadcast_in_dim %s3b3g3, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x14x14xf32>
    %v1001 = stablehlo.broadcast_in_dim %s3b3bt3, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x14x14xf32>
    %v1002 = stablehlo.multiply %v999, %v1000 : tensor<32x1024x14x14xf32>
    %v1003 = stablehlo.add %v1002, %v1001 : tensor<32x1024x14x14xf32>
    %v1004 = stablehlo.reshape %v1003 : (tensor<32x1024x14x14xf32>) -> tensor<32x200704xf32>
    %v1005 = stablehlo.add %v1004, %v925 : tensor<32x200704xf32>
    %v1006 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v1007 = stablehlo.maximum %v1005, %v1006 : tensor<32x200704xf32>
    %v1008 = stablehlo.reshape %v1007 : (tensor<32x200704xf32>) -> tensor<32x1024x14x14xf32>
    %v1009 = stablehlo.convolution(%v1008, %s3b4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x14x14xf32>
    %v1010 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1011 = stablehlo.add %v1009, %v1010 : tensor<32x256x14x14xf32>
    %v1012 = stablehlo.reshape %v1011 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1013 = stablehlo.reshape %v1012 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1014 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1015 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1016 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1017 = stablehlo.reduce(%v1013 init: %v1014) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1018 = stablehlo.broadcast_in_dim %v1017, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1019 = stablehlo.divide %v1018, %v1015 : tensor<32x256x14x14xf32>
    %v1020 = stablehlo.subtract %v1013, %v1019 : tensor<32x256x14x14xf32>
    %v1021 = stablehlo.multiply %v1020, %v1020 : tensor<32x256x14x14xf32>
    %v1022 = stablehlo.reduce(%v1021 init: %v1014) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1023 = stablehlo.broadcast_in_dim %v1022, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1024 = stablehlo.divide %v1023, %v1015 : tensor<32x256x14x14xf32>
    %v1025 = stablehlo.add %v1024, %v1016 : tensor<32x256x14x14xf32>
    %v1026 = stablehlo.rsqrt %v1025 : tensor<32x256x14x14xf32>
    %v1027 = stablehlo.multiply %v1020, %v1026 : tensor<32x256x14x14xf32>
    %v1028 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1029 = stablehlo.broadcast_in_dim %s3b4bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1030 = stablehlo.multiply %v1027, %v1028 : tensor<32x256x14x14xf32>
    %v1031 = stablehlo.add %v1030, %v1029 : tensor<32x256x14x14xf32>
    %v1032 = stablehlo.reshape %v1031 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1033 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1034 = stablehlo.maximum %v1032, %v1033 : tensor<32x50176xf32>
    %v1035 = stablehlo.reshape %v1034 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1036 = stablehlo.convolution(%v1035, %s3b4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1037 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1038 = stablehlo.add %v1036, %v1037 : tensor<32x256x14x14xf32>
    %v1039 = stablehlo.reshape %v1038 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1040 = stablehlo.reshape %v1039 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1041 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1042 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1043 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1044 = stablehlo.reduce(%v1040 init: %v1041) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1045 = stablehlo.broadcast_in_dim %v1044, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1046 = stablehlo.divide %v1045, %v1042 : tensor<32x256x14x14xf32>
    %v1047 = stablehlo.subtract %v1040, %v1046 : tensor<32x256x14x14xf32>
    %v1048 = stablehlo.multiply %v1047, %v1047 : tensor<32x256x14x14xf32>
    %v1049 = stablehlo.reduce(%v1048 init: %v1041) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1050 = stablehlo.broadcast_in_dim %v1049, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1051 = stablehlo.divide %v1050, %v1042 : tensor<32x256x14x14xf32>
    %v1052 = stablehlo.add %v1051, %v1043 : tensor<32x256x14x14xf32>
    %v1053 = stablehlo.rsqrt %v1052 : tensor<32x256x14x14xf32>
    %v1054 = stablehlo.multiply %v1047, %v1053 : tensor<32x256x14x14xf32>
    %v1055 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1056 = stablehlo.broadcast_in_dim %s3b4bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1057 = stablehlo.multiply %v1054, %v1055 : tensor<32x256x14x14xf32>
    %v1058 = stablehlo.add %v1057, %v1056 : tensor<32x256x14x14xf32>
    %v1059 = stablehlo.reshape %v1058 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1060 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1061 = stablehlo.maximum %v1059, %v1060 : tensor<32x50176xf32>
    %v1062 = stablehlo.reshape %v1061 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1063 = stablehlo.convolution(%v1062, %s3b4W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x14x14xf32>
    %v1064 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x14x14xf32>
    %v1065 = stablehlo.add %v1063, %v1064 : tensor<32x1024x14x14xf32>
    %v1066 = stablehlo.reshape %v1065 : (tensor<32x1024x14x14xf32>) -> tensor<32x200704xf32>
    %v1067 = stablehlo.reshape %v1066 : (tensor<32x200704xf32>) -> tensor<32x1024x14x14xf32>
    %v1068 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1069 = stablehlo.constant dense<6272.0> : tensor<32x1024x14x14xf32>
    %v1070 = stablehlo.constant dense<1.0e-05> : tensor<32x1024x14x14xf32>
    %v1071 = stablehlo.reduce(%v1067 init: %v1068) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1072 = stablehlo.broadcast_in_dim %v1071, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x14x14xf32>
    %v1073 = stablehlo.divide %v1072, %v1069 : tensor<32x1024x14x14xf32>
    %v1074 = stablehlo.subtract %v1067, %v1073 : tensor<32x1024x14x14xf32>
    %v1075 = stablehlo.multiply %v1074, %v1074 : tensor<32x1024x14x14xf32>
    %v1076 = stablehlo.reduce(%v1075 init: %v1068) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1077 = stablehlo.broadcast_in_dim %v1076, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x14x14xf32>
    %v1078 = stablehlo.divide %v1077, %v1069 : tensor<32x1024x14x14xf32>
    %v1079 = stablehlo.add %v1078, %v1070 : tensor<32x1024x14x14xf32>
    %v1080 = stablehlo.rsqrt %v1079 : tensor<32x1024x14x14xf32>
    %v1081 = stablehlo.multiply %v1074, %v1080 : tensor<32x1024x14x14xf32>
    %v1082 = stablehlo.broadcast_in_dim %s3b4g3, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x14x14xf32>
    %v1083 = stablehlo.broadcast_in_dim %s3b4bt3, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x14x14xf32>
    %v1084 = stablehlo.multiply %v1081, %v1082 : tensor<32x1024x14x14xf32>
    %v1085 = stablehlo.add %v1084, %v1083 : tensor<32x1024x14x14xf32>
    %v1086 = stablehlo.reshape %v1085 : (tensor<32x1024x14x14xf32>) -> tensor<32x200704xf32>
    %v1087 = stablehlo.add %v1086, %v1007 : tensor<32x200704xf32>
    %v1088 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v1089 = stablehlo.maximum %v1087, %v1088 : tensor<32x200704xf32>
    %v1090 = stablehlo.reshape %v1089 : (tensor<32x200704xf32>) -> tensor<32x1024x14x14xf32>
    %v1091 = stablehlo.convolution(%v1090, %s3b5W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x14x14xf32>
    %v1092 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1093 = stablehlo.add %v1091, %v1092 : tensor<32x256x14x14xf32>
    %v1094 = stablehlo.reshape %v1093 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1095 = stablehlo.reshape %v1094 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1096 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1097 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1098 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1099 = stablehlo.reduce(%v1095 init: %v1096) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1100 = stablehlo.broadcast_in_dim %v1099, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1101 = stablehlo.divide %v1100, %v1097 : tensor<32x256x14x14xf32>
    %v1102 = stablehlo.subtract %v1095, %v1101 : tensor<32x256x14x14xf32>
    %v1103 = stablehlo.multiply %v1102, %v1102 : tensor<32x256x14x14xf32>
    %v1104 = stablehlo.reduce(%v1103 init: %v1096) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1105 = stablehlo.broadcast_in_dim %v1104, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1106 = stablehlo.divide %v1105, %v1097 : tensor<32x256x14x14xf32>
    %v1107 = stablehlo.add %v1106, %v1098 : tensor<32x256x14x14xf32>
    %v1108 = stablehlo.rsqrt %v1107 : tensor<32x256x14x14xf32>
    %v1109 = stablehlo.multiply %v1102, %v1108 : tensor<32x256x14x14xf32>
    %v1110 = stablehlo.broadcast_in_dim %s3b5g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1111 = stablehlo.broadcast_in_dim %s3b5bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1112 = stablehlo.multiply %v1109, %v1110 : tensor<32x256x14x14xf32>
    %v1113 = stablehlo.add %v1112, %v1111 : tensor<32x256x14x14xf32>
    %v1114 = stablehlo.reshape %v1113 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1115 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1116 = stablehlo.maximum %v1114, %v1115 : tensor<32x50176xf32>
    %v1117 = stablehlo.reshape %v1116 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1118 = stablehlo.convolution(%v1117, %s3b5W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1119 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1120 = stablehlo.add %v1118, %v1119 : tensor<32x256x14x14xf32>
    %v1121 = stablehlo.reshape %v1120 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1122 = stablehlo.reshape %v1121 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1123 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1124 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1125 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1126 = stablehlo.reduce(%v1122 init: %v1123) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1127 = stablehlo.broadcast_in_dim %v1126, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1128 = stablehlo.divide %v1127, %v1124 : tensor<32x256x14x14xf32>
    %v1129 = stablehlo.subtract %v1122, %v1128 : tensor<32x256x14x14xf32>
    %v1130 = stablehlo.multiply %v1129, %v1129 : tensor<32x256x14x14xf32>
    %v1131 = stablehlo.reduce(%v1130 init: %v1123) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1132 = stablehlo.broadcast_in_dim %v1131, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1133 = stablehlo.divide %v1132, %v1124 : tensor<32x256x14x14xf32>
    %v1134 = stablehlo.add %v1133, %v1125 : tensor<32x256x14x14xf32>
    %v1135 = stablehlo.rsqrt %v1134 : tensor<32x256x14x14xf32>
    %v1136 = stablehlo.multiply %v1129, %v1135 : tensor<32x256x14x14xf32>
    %v1137 = stablehlo.broadcast_in_dim %s3b5g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1138 = stablehlo.broadcast_in_dim %s3b5bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1139 = stablehlo.multiply %v1136, %v1137 : tensor<32x256x14x14xf32>
    %v1140 = stablehlo.add %v1139, %v1138 : tensor<32x256x14x14xf32>
    %v1141 = stablehlo.reshape %v1140 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1142 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1143 = stablehlo.maximum %v1141, %v1142 : tensor<32x50176xf32>
    %v1144 = stablehlo.reshape %v1143 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1145 = stablehlo.convolution(%v1144, %s3b5W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x14x14xf32>
    %v1146 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x14x14xf32>
    %v1147 = stablehlo.add %v1145, %v1146 : tensor<32x1024x14x14xf32>
    %v1148 = stablehlo.reshape %v1147 : (tensor<32x1024x14x14xf32>) -> tensor<32x200704xf32>
    %v1149 = stablehlo.reshape %v1148 : (tensor<32x200704xf32>) -> tensor<32x1024x14x14xf32>
    %v1150 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1151 = stablehlo.constant dense<6272.0> : tensor<32x1024x14x14xf32>
    %v1152 = stablehlo.constant dense<1.0e-05> : tensor<32x1024x14x14xf32>
    %v1153 = stablehlo.reduce(%v1149 init: %v1150) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1154 = stablehlo.broadcast_in_dim %v1153, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x14x14xf32>
    %v1155 = stablehlo.divide %v1154, %v1151 : tensor<32x1024x14x14xf32>
    %v1156 = stablehlo.subtract %v1149, %v1155 : tensor<32x1024x14x14xf32>
    %v1157 = stablehlo.multiply %v1156, %v1156 : tensor<32x1024x14x14xf32>
    %v1158 = stablehlo.reduce(%v1157 init: %v1150) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1159 = stablehlo.broadcast_in_dim %v1158, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x14x14xf32>
    %v1160 = stablehlo.divide %v1159, %v1151 : tensor<32x1024x14x14xf32>
    %v1161 = stablehlo.add %v1160, %v1152 : tensor<32x1024x14x14xf32>
    %v1162 = stablehlo.rsqrt %v1161 : tensor<32x1024x14x14xf32>
    %v1163 = stablehlo.multiply %v1156, %v1162 : tensor<32x1024x14x14xf32>
    %v1164 = stablehlo.broadcast_in_dim %s3b5g3, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x14x14xf32>
    %v1165 = stablehlo.broadcast_in_dim %s3b5bt3, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x14x14xf32>
    %v1166 = stablehlo.multiply %v1163, %v1164 : tensor<32x1024x14x14xf32>
    %v1167 = stablehlo.add %v1166, %v1165 : tensor<32x1024x14x14xf32>
    %v1168 = stablehlo.reshape %v1167 : (tensor<32x1024x14x14xf32>) -> tensor<32x200704xf32>
    %v1169 = stablehlo.add %v1168, %v1089 : tensor<32x200704xf32>
    %v1170 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v1171 = stablehlo.maximum %v1169, %v1170 : tensor<32x200704xf32>
    %v1172 = stablehlo.reshape %v1171 : (tensor<32x200704xf32>) -> tensor<32x1024x14x14xf32>
    %v1173 = stablehlo.convolution(%v1172, %s4b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x14x14xf32>, tensor<512x1024x1x1xf32>) -> tensor<32x512x14x14xf32>
    %v1174 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1175 = stablehlo.add %v1173, %v1174 : tensor<32x512x14x14xf32>
    %v1176 = stablehlo.reshape %v1175 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1177 = stablehlo.reshape %v1176 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1178 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1179 = stablehlo.constant dense<6272.0> : tensor<32x512x14x14xf32>
    %v1180 = stablehlo.constant dense<1.0e-05> : tensor<32x512x14x14xf32>
    %v1181 = stablehlo.reduce(%v1177 init: %v1178) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v1182 = stablehlo.broadcast_in_dim %v1181, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1183 = stablehlo.divide %v1182, %v1179 : tensor<32x512x14x14xf32>
    %v1184 = stablehlo.subtract %v1177, %v1183 : tensor<32x512x14x14xf32>
    %v1185 = stablehlo.multiply %v1184, %v1184 : tensor<32x512x14x14xf32>
    %v1186 = stablehlo.reduce(%v1185 init: %v1178) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v1187 = stablehlo.broadcast_in_dim %v1186, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1188 = stablehlo.divide %v1187, %v1179 : tensor<32x512x14x14xf32>
    %v1189 = stablehlo.add %v1188, %v1180 : tensor<32x512x14x14xf32>
    %v1190 = stablehlo.rsqrt %v1189 : tensor<32x512x14x14xf32>
    %v1191 = stablehlo.multiply %v1184, %v1190 : tensor<32x512x14x14xf32>
    %v1192 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1193 = stablehlo.broadcast_in_dim %s4b0bt1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x14x14xf32>
    %v1194 = stablehlo.multiply %v1191, %v1192 : tensor<32x512x14x14xf32>
    %v1195 = stablehlo.add %v1194, %v1193 : tensor<32x512x14x14xf32>
    %v1196 = stablehlo.reshape %v1195 : (tensor<32x512x14x14xf32>) -> tensor<32x100352xf32>
    %v1197 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v1198 = stablehlo.maximum %v1196, %v1197 : tensor<32x100352xf32>
    %v1199 = stablehlo.reshape %v1198 : (tensor<32x100352xf32>) -> tensor<32x512x14x14xf32>
    %v1200 = stablehlo.convolution(%v1199, %s4b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1201 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1202 = stablehlo.add %v1200, %v1201 : tensor<32x512x7x7xf32>
    %v1203 = stablehlo.reshape %v1202 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1204 = stablehlo.reshape %v1203 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1205 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1206 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1207 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1208 = stablehlo.reduce(%v1204 init: %v1205) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1209 = stablehlo.broadcast_in_dim %v1208, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1210 = stablehlo.divide %v1209, %v1206 : tensor<32x512x7x7xf32>
    %v1211 = stablehlo.subtract %v1204, %v1210 : tensor<32x512x7x7xf32>
    %v1212 = stablehlo.multiply %v1211, %v1211 : tensor<32x512x7x7xf32>
    %v1213 = stablehlo.reduce(%v1212 init: %v1205) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1214 = stablehlo.broadcast_in_dim %v1213, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1215 = stablehlo.divide %v1214, %v1206 : tensor<32x512x7x7xf32>
    %v1216 = stablehlo.add %v1215, %v1207 : tensor<32x512x7x7xf32>
    %v1217 = stablehlo.rsqrt %v1216 : tensor<32x512x7x7xf32>
    %v1218 = stablehlo.multiply %v1211, %v1217 : tensor<32x512x7x7xf32>
    %v1219 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1220 = stablehlo.broadcast_in_dim %s4b0bt2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1221 = stablehlo.multiply %v1218, %v1219 : tensor<32x512x7x7xf32>
    %v1222 = stablehlo.add %v1221, %v1220 : tensor<32x512x7x7xf32>
    %v1223 = stablehlo.reshape %v1222 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1224 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v1225 = stablehlo.maximum %v1223, %v1224 : tensor<32x25088xf32>
    %v1226 = stablehlo.reshape %v1225 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1227 = stablehlo.convolution(%v1226, %s4b0W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x7x7xf32>
    %v1228 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x7x7xf32>
    %v1229 = stablehlo.add %v1227, %v1228 : tensor<32x2048x7x7xf32>
    %v1230 = stablehlo.reshape %v1229 : (tensor<32x2048x7x7xf32>) -> tensor<32x100352xf32>
    %v1231 = stablehlo.reshape %v1230 : (tensor<32x100352xf32>) -> tensor<32x2048x7x7xf32>
    %v1232 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1233 = stablehlo.constant dense<1568.0> : tensor<32x2048x7x7xf32>
    %v1234 = stablehlo.constant dense<1.0e-05> : tensor<32x2048x7x7xf32>
    %v1235 = stablehlo.reduce(%v1231 init: %v1232) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1236 = stablehlo.broadcast_in_dim %v1235, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x7x7xf32>
    %v1237 = stablehlo.divide %v1236, %v1233 : tensor<32x2048x7x7xf32>
    %v1238 = stablehlo.subtract %v1231, %v1237 : tensor<32x2048x7x7xf32>
    %v1239 = stablehlo.multiply %v1238, %v1238 : tensor<32x2048x7x7xf32>
    %v1240 = stablehlo.reduce(%v1239 init: %v1232) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1241 = stablehlo.broadcast_in_dim %v1240, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x7x7xf32>
    %v1242 = stablehlo.divide %v1241, %v1233 : tensor<32x2048x7x7xf32>
    %v1243 = stablehlo.add %v1242, %v1234 : tensor<32x2048x7x7xf32>
    %v1244 = stablehlo.rsqrt %v1243 : tensor<32x2048x7x7xf32>
    %v1245 = stablehlo.multiply %v1238, %v1244 : tensor<32x2048x7x7xf32>
    %v1246 = stablehlo.broadcast_in_dim %s4b0g3, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x7x7xf32>
    %v1247 = stablehlo.broadcast_in_dim %s4b0bt3, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x7x7xf32>
    %v1248 = stablehlo.multiply %v1245, %v1246 : tensor<32x2048x7x7xf32>
    %v1249 = stablehlo.add %v1248, %v1247 : tensor<32x2048x7x7xf32>
    %v1250 = stablehlo.reshape %v1249 : (tensor<32x2048x7x7xf32>) -> tensor<32x100352xf32>
    %v1251 = stablehlo.reshape %v1171 : (tensor<32x200704xf32>) -> tensor<32x1024x14x14xf32>
    %v1252 = stablehlo.convolution(%v1251, %s4b0Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x14x14xf32>, tensor<2048x1024x1x1xf32>) -> tensor<32x2048x7x7xf32>
    %v1253 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x7x7xf32>
    %v1254 = stablehlo.add %v1252, %v1253 : tensor<32x2048x7x7xf32>
    %v1255 = stablehlo.reshape %v1254 : (tensor<32x2048x7x7xf32>) -> tensor<32x100352xf32>
    %v1256 = stablehlo.reshape %v1255 : (tensor<32x100352xf32>) -> tensor<32x2048x7x7xf32>
    %v1257 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1258 = stablehlo.constant dense<1568.0> : tensor<32x2048x7x7xf32>
    %v1259 = stablehlo.constant dense<1.0e-05> : tensor<32x2048x7x7xf32>
    %v1260 = stablehlo.reduce(%v1256 init: %v1257) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1261 = stablehlo.broadcast_in_dim %v1260, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x7x7xf32>
    %v1262 = stablehlo.divide %v1261, %v1258 : tensor<32x2048x7x7xf32>
    %v1263 = stablehlo.subtract %v1256, %v1262 : tensor<32x2048x7x7xf32>
    %v1264 = stablehlo.multiply %v1263, %v1263 : tensor<32x2048x7x7xf32>
    %v1265 = stablehlo.reduce(%v1264 init: %v1257) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1266 = stablehlo.broadcast_in_dim %v1265, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x7x7xf32>
    %v1267 = stablehlo.divide %v1266, %v1258 : tensor<32x2048x7x7xf32>
    %v1268 = stablehlo.add %v1267, %v1259 : tensor<32x2048x7x7xf32>
    %v1269 = stablehlo.rsqrt %v1268 : tensor<32x2048x7x7xf32>
    %v1270 = stablehlo.multiply %v1263, %v1269 : tensor<32x2048x7x7xf32>
    %v1271 = stablehlo.broadcast_in_dim %s4b0gp, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x7x7xf32>
    %v1272 = stablehlo.broadcast_in_dim %s4b0btp, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x7x7xf32>
    %v1273 = stablehlo.multiply %v1270, %v1271 : tensor<32x2048x7x7xf32>
    %v1274 = stablehlo.add %v1273, %v1272 : tensor<32x2048x7x7xf32>
    %v1275 = stablehlo.reshape %v1274 : (tensor<32x2048x7x7xf32>) -> tensor<32x100352xf32>
    %v1276 = stablehlo.add %v1250, %v1275 : tensor<32x100352xf32>
    %v1277 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v1278 = stablehlo.maximum %v1276, %v1277 : tensor<32x100352xf32>
    %v1279 = stablehlo.reshape %v1278 : (tensor<32x100352xf32>) -> tensor<32x2048x7x7xf32>
    %v1280 = stablehlo.convolution(%v1279, %s4b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x7x7xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x7x7xf32>
    %v1281 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1282 = stablehlo.add %v1280, %v1281 : tensor<32x512x7x7xf32>
    %v1283 = stablehlo.reshape %v1282 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1284 = stablehlo.reshape %v1283 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1285 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1286 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1287 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1288 = stablehlo.reduce(%v1284 init: %v1285) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1289 = stablehlo.broadcast_in_dim %v1288, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1290 = stablehlo.divide %v1289, %v1286 : tensor<32x512x7x7xf32>
    %v1291 = stablehlo.subtract %v1284, %v1290 : tensor<32x512x7x7xf32>
    %v1292 = stablehlo.multiply %v1291, %v1291 : tensor<32x512x7x7xf32>
    %v1293 = stablehlo.reduce(%v1292 init: %v1285) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1294 = stablehlo.broadcast_in_dim %v1293, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1295 = stablehlo.divide %v1294, %v1286 : tensor<32x512x7x7xf32>
    %v1296 = stablehlo.add %v1295, %v1287 : tensor<32x512x7x7xf32>
    %v1297 = stablehlo.rsqrt %v1296 : tensor<32x512x7x7xf32>
    %v1298 = stablehlo.multiply %v1291, %v1297 : tensor<32x512x7x7xf32>
    %v1299 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1300 = stablehlo.broadcast_in_dim %s4b1bt1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1301 = stablehlo.multiply %v1298, %v1299 : tensor<32x512x7x7xf32>
    %v1302 = stablehlo.add %v1301, %v1300 : tensor<32x512x7x7xf32>
    %v1303 = stablehlo.reshape %v1302 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1304 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v1305 = stablehlo.maximum %v1303, %v1304 : tensor<32x25088xf32>
    %v1306 = stablehlo.reshape %v1305 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1307 = stablehlo.convolution(%v1306, %s4b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1308 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1309 = stablehlo.add %v1307, %v1308 : tensor<32x512x7x7xf32>
    %v1310 = stablehlo.reshape %v1309 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1311 = stablehlo.reshape %v1310 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1312 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1313 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1314 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1315 = stablehlo.reduce(%v1311 init: %v1312) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1316 = stablehlo.broadcast_in_dim %v1315, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1317 = stablehlo.divide %v1316, %v1313 : tensor<32x512x7x7xf32>
    %v1318 = stablehlo.subtract %v1311, %v1317 : tensor<32x512x7x7xf32>
    %v1319 = stablehlo.multiply %v1318, %v1318 : tensor<32x512x7x7xf32>
    %v1320 = stablehlo.reduce(%v1319 init: %v1312) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1321 = stablehlo.broadcast_in_dim %v1320, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1322 = stablehlo.divide %v1321, %v1313 : tensor<32x512x7x7xf32>
    %v1323 = stablehlo.add %v1322, %v1314 : tensor<32x512x7x7xf32>
    %v1324 = stablehlo.rsqrt %v1323 : tensor<32x512x7x7xf32>
    %v1325 = stablehlo.multiply %v1318, %v1324 : tensor<32x512x7x7xf32>
    %v1326 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1327 = stablehlo.broadcast_in_dim %s4b1bt2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1328 = stablehlo.multiply %v1325, %v1326 : tensor<32x512x7x7xf32>
    %v1329 = stablehlo.add %v1328, %v1327 : tensor<32x512x7x7xf32>
    %v1330 = stablehlo.reshape %v1329 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1331 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v1332 = stablehlo.maximum %v1330, %v1331 : tensor<32x25088xf32>
    %v1333 = stablehlo.reshape %v1332 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1334 = stablehlo.convolution(%v1333, %s4b1W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x7x7xf32>
    %v1335 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x7x7xf32>
    %v1336 = stablehlo.add %v1334, %v1335 : tensor<32x2048x7x7xf32>
    %v1337 = stablehlo.reshape %v1336 : (tensor<32x2048x7x7xf32>) -> tensor<32x100352xf32>
    %v1338 = stablehlo.reshape %v1337 : (tensor<32x100352xf32>) -> tensor<32x2048x7x7xf32>
    %v1339 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1340 = stablehlo.constant dense<1568.0> : tensor<32x2048x7x7xf32>
    %v1341 = stablehlo.constant dense<1.0e-05> : tensor<32x2048x7x7xf32>
    %v1342 = stablehlo.reduce(%v1338 init: %v1339) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1343 = stablehlo.broadcast_in_dim %v1342, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x7x7xf32>
    %v1344 = stablehlo.divide %v1343, %v1340 : tensor<32x2048x7x7xf32>
    %v1345 = stablehlo.subtract %v1338, %v1344 : tensor<32x2048x7x7xf32>
    %v1346 = stablehlo.multiply %v1345, %v1345 : tensor<32x2048x7x7xf32>
    %v1347 = stablehlo.reduce(%v1346 init: %v1339) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1348 = stablehlo.broadcast_in_dim %v1347, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x7x7xf32>
    %v1349 = stablehlo.divide %v1348, %v1340 : tensor<32x2048x7x7xf32>
    %v1350 = stablehlo.add %v1349, %v1341 : tensor<32x2048x7x7xf32>
    %v1351 = stablehlo.rsqrt %v1350 : tensor<32x2048x7x7xf32>
    %v1352 = stablehlo.multiply %v1345, %v1351 : tensor<32x2048x7x7xf32>
    %v1353 = stablehlo.broadcast_in_dim %s4b1g3, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x7x7xf32>
    %v1354 = stablehlo.broadcast_in_dim %s4b1bt3, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x7x7xf32>
    %v1355 = stablehlo.multiply %v1352, %v1353 : tensor<32x2048x7x7xf32>
    %v1356 = stablehlo.add %v1355, %v1354 : tensor<32x2048x7x7xf32>
    %v1357 = stablehlo.reshape %v1356 : (tensor<32x2048x7x7xf32>) -> tensor<32x100352xf32>
    %v1358 = stablehlo.add %v1357, %v1278 : tensor<32x100352xf32>
    %v1359 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v1360 = stablehlo.maximum %v1358, %v1359 : tensor<32x100352xf32>
    %v1361 = stablehlo.reshape %v1360 : (tensor<32x100352xf32>) -> tensor<32x2048x7x7xf32>
    %v1362 = stablehlo.convolution(%v1361, %s4b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x2048x7x7xf32>, tensor<512x2048x1x1xf32>) -> tensor<32x512x7x7xf32>
    %v1363 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1364 = stablehlo.add %v1362, %v1363 : tensor<32x512x7x7xf32>
    %v1365 = stablehlo.reshape %v1364 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1366 = stablehlo.reshape %v1365 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1367 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1368 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1369 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1370 = stablehlo.reduce(%v1366 init: %v1367) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1371 = stablehlo.broadcast_in_dim %v1370, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1372 = stablehlo.divide %v1371, %v1368 : tensor<32x512x7x7xf32>
    %v1373 = stablehlo.subtract %v1366, %v1372 : tensor<32x512x7x7xf32>
    %v1374 = stablehlo.multiply %v1373, %v1373 : tensor<32x512x7x7xf32>
    %v1375 = stablehlo.reduce(%v1374 init: %v1367) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1376 = stablehlo.broadcast_in_dim %v1375, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1377 = stablehlo.divide %v1376, %v1368 : tensor<32x512x7x7xf32>
    %v1378 = stablehlo.add %v1377, %v1369 : tensor<32x512x7x7xf32>
    %v1379 = stablehlo.rsqrt %v1378 : tensor<32x512x7x7xf32>
    %v1380 = stablehlo.multiply %v1373, %v1379 : tensor<32x512x7x7xf32>
    %v1381 = stablehlo.broadcast_in_dim %s4b2g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1382 = stablehlo.broadcast_in_dim %s4b2bt1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1383 = stablehlo.multiply %v1380, %v1381 : tensor<32x512x7x7xf32>
    %v1384 = stablehlo.add %v1383, %v1382 : tensor<32x512x7x7xf32>
    %v1385 = stablehlo.reshape %v1384 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1386 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v1387 = stablehlo.maximum %v1385, %v1386 : tensor<32x25088xf32>
    %v1388 = stablehlo.reshape %v1387 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1389 = stablehlo.convolution(%v1388, %s4b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1390 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1391 = stablehlo.add %v1389, %v1390 : tensor<32x512x7x7xf32>
    %v1392 = stablehlo.reshape %v1391 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1393 = stablehlo.reshape %v1392 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1394 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1395 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1396 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1397 = stablehlo.reduce(%v1393 init: %v1394) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1398 = stablehlo.broadcast_in_dim %v1397, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1399 = stablehlo.divide %v1398, %v1395 : tensor<32x512x7x7xf32>
    %v1400 = stablehlo.subtract %v1393, %v1399 : tensor<32x512x7x7xf32>
    %v1401 = stablehlo.multiply %v1400, %v1400 : tensor<32x512x7x7xf32>
    %v1402 = stablehlo.reduce(%v1401 init: %v1394) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1403 = stablehlo.broadcast_in_dim %v1402, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1404 = stablehlo.divide %v1403, %v1395 : tensor<32x512x7x7xf32>
    %v1405 = stablehlo.add %v1404, %v1396 : tensor<32x512x7x7xf32>
    %v1406 = stablehlo.rsqrt %v1405 : tensor<32x512x7x7xf32>
    %v1407 = stablehlo.multiply %v1400, %v1406 : tensor<32x512x7x7xf32>
    %v1408 = stablehlo.broadcast_in_dim %s4b2g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1409 = stablehlo.broadcast_in_dim %s4b2bt2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1410 = stablehlo.multiply %v1407, %v1408 : tensor<32x512x7x7xf32>
    %v1411 = stablehlo.add %v1410, %v1409 : tensor<32x512x7x7xf32>
    %v1412 = stablehlo.reshape %v1411 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1413 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v1414 = stablehlo.maximum %v1412, %v1413 : tensor<32x25088xf32>
    %v1415 = stablehlo.reshape %v1414 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1416 = stablehlo.convolution(%v1415, %s4b2W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tensor<32x2048x7x7xf32>
    %v1417 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x7x7xf32>
    %v1418 = stablehlo.add %v1416, %v1417 : tensor<32x2048x7x7xf32>
    %v1419 = stablehlo.reshape %v1418 : (tensor<32x2048x7x7xf32>) -> tensor<32x100352xf32>
    %v1420 = stablehlo.reshape %v1419 : (tensor<32x100352xf32>) -> tensor<32x2048x7x7xf32>
    %v1421 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1422 = stablehlo.constant dense<1568.0> : tensor<32x2048x7x7xf32>
    %v1423 = stablehlo.constant dense<1.0e-05> : tensor<32x2048x7x7xf32>
    %v1424 = stablehlo.reduce(%v1420 init: %v1421) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1425 = stablehlo.broadcast_in_dim %v1424, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x7x7xf32>
    %v1426 = stablehlo.divide %v1425, %v1422 : tensor<32x2048x7x7xf32>
    %v1427 = stablehlo.subtract %v1420, %v1426 : tensor<32x2048x7x7xf32>
    %v1428 = stablehlo.multiply %v1427, %v1427 : tensor<32x2048x7x7xf32>
    %v1429 = stablehlo.reduce(%v1428 init: %v1421) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1430 = stablehlo.broadcast_in_dim %v1429, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x7x7xf32>
    %v1431 = stablehlo.divide %v1430, %v1422 : tensor<32x2048x7x7xf32>
    %v1432 = stablehlo.add %v1431, %v1423 : tensor<32x2048x7x7xf32>
    %v1433 = stablehlo.rsqrt %v1432 : tensor<32x2048x7x7xf32>
    %v1434 = stablehlo.multiply %v1427, %v1433 : tensor<32x2048x7x7xf32>
    %v1435 = stablehlo.broadcast_in_dim %s4b2g3, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x7x7xf32>
    %v1436 = stablehlo.broadcast_in_dim %s4b2bt3, dims = [1] : (tensor<2048xf32>) -> tensor<32x2048x7x7xf32>
    %v1437 = stablehlo.multiply %v1434, %v1435 : tensor<32x2048x7x7xf32>
    %v1438 = stablehlo.add %v1437, %v1436 : tensor<32x2048x7x7xf32>
    %v1439 = stablehlo.reshape %v1438 : (tensor<32x2048x7x7xf32>) -> tensor<32x100352xf32>
    %v1440 = stablehlo.add %v1439, %v1360 : tensor<32x100352xf32>
    %v1441 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v1442 = stablehlo.maximum %v1440, %v1441 : tensor<32x100352xf32>
    %v1443 = stablehlo.reshape %v1442 : (tensor<32x100352xf32>) -> tensor<32x2048x7x7xf32>
    %v1444 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1445 = stablehlo.reduce(%v1443 init: %v1444) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x2048x7x7xf32>, tensor<f32>) -> tensor<32x2048xf32>
    %v1446 = stablehlo.constant dense<49.0> : tensor<32x2048xf32>
    %v1447 = stablehlo.divide %v1445, %v1446 : tensor<32x2048xf32>
    %v1448 = stablehlo.dot_general %v1447, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x2048xf32>, tensor<2048x10xf32>) -> tensor<32x10xf32>
    %v1449 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v1450 = stablehlo.add %v1448, %v1449 : tensor<32x10xf32>
    return %v1450 : tensor<32x10xf32>
  }
}
