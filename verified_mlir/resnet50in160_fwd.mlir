module @m {
  func.func @resnet50in160_fwd(%x: tensor<64x76800xf32>, %sW: tensor<64x3x7x7xf32>, %sg: tensor<64xf32>, %sbt: tensor<64xf32>, %s1b0W1: tensor<64x64x1x1xf32>, %s1b0g1: tensor<64xf32>, %s1b0bt1: tensor<64xf32>, %s1b0W2: tensor<64x64x3x3xf32>, %s1b0g2: tensor<64xf32>, %s1b0bt2: tensor<64xf32>, %s1b0W3: tensor<256x64x1x1xf32>, %s1b0g3: tensor<256xf32>, %s1b0bt3: tensor<256xf32>, %s1b0Wp: tensor<256x64x1x1xf32>, %s1b0gp: tensor<256xf32>, %s1b0btp: tensor<256xf32>, %s1b1W1: tensor<64x256x1x1xf32>, %s1b1g1: tensor<64xf32>, %s1b1bt1: tensor<64xf32>, %s1b1W2: tensor<64x64x3x3xf32>, %s1b1g2: tensor<64xf32>, %s1b1bt2: tensor<64xf32>, %s1b1W3: tensor<256x64x1x1xf32>, %s1b1g3: tensor<256xf32>, %s1b1bt3: tensor<256xf32>, %s1b2W1: tensor<64x256x1x1xf32>, %s1b2g1: tensor<64xf32>, %s1b2bt1: tensor<64xf32>, %s1b2W2: tensor<64x64x3x3xf32>, %s1b2g2: tensor<64xf32>, %s1b2bt2: tensor<64xf32>, %s1b2W3: tensor<256x64x1x1xf32>, %s1b2g3: tensor<256xf32>, %s1b2bt3: tensor<256xf32>, %s2b0W1: tensor<128x256x1x1xf32>, %s2b0g1: tensor<128xf32>, %s2b0bt1: tensor<128xf32>, %s2b0W2: tensor<128x128x3x3xf32>, %s2b0g2: tensor<128xf32>, %s2b0bt2: tensor<128xf32>, %s2b0W3: tensor<512x128x1x1xf32>, %s2b0g3: tensor<512xf32>, %s2b0bt3: tensor<512xf32>, %s2b0Wp: tensor<512x256x1x1xf32>, %s2b0gp: tensor<512xf32>, %s2b0btp: tensor<512xf32>, %s2b1W1: tensor<128x512x1x1xf32>, %s2b1g1: tensor<128xf32>, %s2b1bt1: tensor<128xf32>, %s2b1W2: tensor<128x128x3x3xf32>, %s2b1g2: tensor<128xf32>, %s2b1bt2: tensor<128xf32>, %s2b1W3: tensor<512x128x1x1xf32>, %s2b1g3: tensor<512xf32>, %s2b1bt3: tensor<512xf32>, %s2b2W1: tensor<128x512x1x1xf32>, %s2b2g1: tensor<128xf32>, %s2b2bt1: tensor<128xf32>, %s2b2W2: tensor<128x128x3x3xf32>, %s2b2g2: tensor<128xf32>, %s2b2bt2: tensor<128xf32>, %s2b2W3: tensor<512x128x1x1xf32>, %s2b2g3: tensor<512xf32>, %s2b2bt3: tensor<512xf32>, %s2b3W1: tensor<128x512x1x1xf32>, %s2b3g1: tensor<128xf32>, %s2b3bt1: tensor<128xf32>, %s2b3W2: tensor<128x128x3x3xf32>, %s2b3g2: tensor<128xf32>, %s2b3bt2: tensor<128xf32>, %s2b3W3: tensor<512x128x1x1xf32>, %s2b3g3: tensor<512xf32>, %s2b3bt3: tensor<512xf32>, %s3b0W1: tensor<256x512x1x1xf32>, %s3b0g1: tensor<256xf32>, %s3b0bt1: tensor<256xf32>, %s3b0W2: tensor<256x256x3x3xf32>, %s3b0g2: tensor<256xf32>, %s3b0bt2: tensor<256xf32>, %s3b0W3: tensor<1024x256x1x1xf32>, %s3b0g3: tensor<1024xf32>, %s3b0bt3: tensor<1024xf32>, %s3b0Wp: tensor<1024x512x1x1xf32>, %s3b0gp: tensor<1024xf32>, %s3b0btp: tensor<1024xf32>, %s3b1W1: tensor<256x1024x1x1xf32>, %s3b1g1: tensor<256xf32>, %s3b1bt1: tensor<256xf32>, %s3b1W2: tensor<256x256x3x3xf32>, %s3b1g2: tensor<256xf32>, %s3b1bt2: tensor<256xf32>, %s3b1W3: tensor<1024x256x1x1xf32>, %s3b1g3: tensor<1024xf32>, %s3b1bt3: tensor<1024xf32>, %s3b2W1: tensor<256x1024x1x1xf32>, %s3b2g1: tensor<256xf32>, %s3b2bt1: tensor<256xf32>, %s3b2W2: tensor<256x256x3x3xf32>, %s3b2g2: tensor<256xf32>, %s3b2bt2: tensor<256xf32>, %s3b2W3: tensor<1024x256x1x1xf32>, %s3b2g3: tensor<1024xf32>, %s3b2bt3: tensor<1024xf32>, %s3b3W1: tensor<256x1024x1x1xf32>, %s3b3g1: tensor<256xf32>, %s3b3bt1: tensor<256xf32>, %s3b3W2: tensor<256x256x3x3xf32>, %s3b3g2: tensor<256xf32>, %s3b3bt2: tensor<256xf32>, %s3b3W3: tensor<1024x256x1x1xf32>, %s3b3g3: tensor<1024xf32>, %s3b3bt3: tensor<1024xf32>, %s3b4W1: tensor<256x1024x1x1xf32>, %s3b4g1: tensor<256xf32>, %s3b4bt1: tensor<256xf32>, %s3b4W2: tensor<256x256x3x3xf32>, %s3b4g2: tensor<256xf32>, %s3b4bt2: tensor<256xf32>, %s3b4W3: tensor<1024x256x1x1xf32>, %s3b4g3: tensor<1024xf32>, %s3b4bt3: tensor<1024xf32>, %s3b5W1: tensor<256x1024x1x1xf32>, %s3b5g1: tensor<256xf32>, %s3b5bt1: tensor<256xf32>, %s3b5W2: tensor<256x256x3x3xf32>, %s3b5g2: tensor<256xf32>, %s3b5bt2: tensor<256xf32>, %s3b5W3: tensor<1024x256x1x1xf32>, %s3b5g3: tensor<1024xf32>, %s3b5bt3: tensor<1024xf32>, %s4b0W1: tensor<512x1024x1x1xf32>, %s4b0g1: tensor<512xf32>, %s4b0bt1: tensor<512xf32>, %s4b0W2: tensor<512x512x3x3xf32>, %s4b0g2: tensor<512xf32>, %s4b0bt2: tensor<512xf32>, %s4b0W3: tensor<2048x512x1x1xf32>, %s4b0g3: tensor<2048xf32>, %s4b0bt3: tensor<2048xf32>, %s4b0Wp: tensor<2048x1024x1x1xf32>, %s4b0gp: tensor<2048xf32>, %s4b0btp: tensor<2048xf32>, %s4b1W1: tensor<512x2048x1x1xf32>, %s4b1g1: tensor<512xf32>, %s4b1bt1: tensor<512xf32>, %s4b1W2: tensor<512x512x3x3xf32>, %s4b1g2: tensor<512xf32>, %s4b1bt2: tensor<512xf32>, %s4b1W3: tensor<2048x512x1x1xf32>, %s4b1g3: tensor<2048xf32>, %s4b1bt3: tensor<2048xf32>, %s4b2W1: tensor<512x2048x1x1xf32>, %s4b2g1: tensor<512xf32>, %s4b2bt1: tensor<512xf32>, %s4b2W2: tensor<512x512x3x3xf32>, %s4b2g2: tensor<512xf32>, %s4b2bt2: tensor<512xf32>, %s4b2W3: tensor<2048x512x1x1xf32>, %s4b2g3: tensor<2048xf32>, %s4b2bt3: tensor<2048xf32>, %Wd: tensor<2048x1000xf32>, %bd: tensor<1000xf32>) -> tensor<64x1000xf32> {
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
    %v0 = stablehlo.reshape %x : (tensor<64x76800xf32>) -> tensor<64x3x160x160xf32>
    %v1 = stablehlo.convolution(%v0, %sW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x3x160x160xf32>, tensor<64x3x7x7xf32>) -> tensor<64x64x80x80xf32>
    %v2 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x80x80xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<64x64x80x80xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<64x64x80x80xf32>) -> tensor<64x409600xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<64x409600xf32>) -> tensor<64x64x80x80xf32>
    %v6 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7 = stablehlo.constant dense<409600.0> : tensor<64x64x80x80xf32>
    %v8 = stablehlo.constant dense<1.0e-05> : tensor<64x64x80x80xf32>
    %v9 = stablehlo.reduce(%v5 init: %v6) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x80x80xf32>, tensor<f32>) -> tensor<64xf32>
    %v10 = stablehlo.broadcast_in_dim %v9, dims = [1] : (tensor<64xf32>) -> tensor<64x64x80x80xf32>
    %v11 = stablehlo.divide %v10, %v7 : tensor<64x64x80x80xf32>
    %v12 = stablehlo.subtract %v5, %v11 : tensor<64x64x80x80xf32>
    %v13 = stablehlo.multiply %v12, %v12 : tensor<64x64x80x80xf32>
    %v14 = stablehlo.reduce(%v13 init: %v6) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x80x80xf32>, tensor<f32>) -> tensor<64xf32>
    %v15 = stablehlo.broadcast_in_dim %v14, dims = [1] : (tensor<64xf32>) -> tensor<64x64x80x80xf32>
    %v16 = stablehlo.divide %v15, %v7 : tensor<64x64x80x80xf32>
    %v17 = stablehlo.add %v16, %v8 : tensor<64x64x80x80xf32>
    %v18 = stablehlo.rsqrt %v17 : tensor<64x64x80x80xf32>
    %v19 = stablehlo.multiply %v12, %v18 : tensor<64x64x80x80xf32>
    %v20 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<64x64x80x80xf32>
    %v21 = stablehlo.broadcast_in_dim %sbt, dims = [1] : (tensor<64xf32>) -> tensor<64x64x80x80xf32>
    %v22 = stablehlo.multiply %v19, %v20 : tensor<64x64x80x80xf32>
    %v23 = stablehlo.add %v22, %v21 : tensor<64x64x80x80xf32>
    %v24 = stablehlo.reshape %v23 : (tensor<64x64x80x80xf32>) -> tensor<64x409600xf32>
    %v25 = stablehlo.reshape %v24 : (tensor<64x409600xf32>) -> tensor<64x64x80x80xf32>
    %v26 = stablehlo.constant dense<0.0> : tensor<64x64x80x80xf32>
    %v27 = stablehlo.maximum %v25, %v26 : tensor<64x64x80x80xf32>
    %v28 = stablehlo.reshape %v27 : (tensor<64x64x80x80xf32>) -> tensor<64x409600xf32>
    %v29 = stablehlo.reshape %v28 : (tensor<64x409600xf32>) -> tensor<64x64x80x80xf32>
    %v30 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v31 = "stablehlo.reduce_window"(%v29, %v30) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>} : (tensor<64x64x80x80xf32>, tensor<f32>) -> tensor<64x64x40x40xf32>
    %v32 = stablehlo.reshape %v31 : (tensor<64x64x40x40xf32>) -> tensor<64x102400xf32>
    %v33 = stablehlo.reshape %v32 : (tensor<64x102400xf32>) -> tensor<64x64x40x40xf32>
    %v34 = stablehlo.convolution(%v33, %s1b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x40x40xf32>, tensor<64x64x1x1xf32>) -> tensor<64x64x40x40xf32>
    %v35 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x40x40xf32>
    %v36 = stablehlo.add %v34, %v35 : tensor<64x64x40x40xf32>
    %v37 = stablehlo.reshape %v36 : (tensor<64x64x40x40xf32>) -> tensor<64x102400xf32>
    %v38 = stablehlo.reshape %v37 : (tensor<64x102400xf32>) -> tensor<64x64x40x40xf32>
    %v39 = stablehlo.constant dense<0.0> : tensor<f32>
    %v40 = stablehlo.constant dense<102400.0> : tensor<64x64x40x40xf32>
    %v41 = stablehlo.constant dense<1.0e-05> : tensor<64x64x40x40xf32>
    %v42 = stablehlo.reduce(%v38 init: %v39) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x40x40xf32>, tensor<f32>) -> tensor<64xf32>
    %v43 = stablehlo.broadcast_in_dim %v42, dims = [1] : (tensor<64xf32>) -> tensor<64x64x40x40xf32>
    %v44 = stablehlo.divide %v43, %v40 : tensor<64x64x40x40xf32>
    %v45 = stablehlo.subtract %v38, %v44 : tensor<64x64x40x40xf32>
    %v46 = stablehlo.multiply %v45, %v45 : tensor<64x64x40x40xf32>
    %v47 = stablehlo.reduce(%v46 init: %v39) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x40x40xf32>, tensor<f32>) -> tensor<64xf32>
    %v48 = stablehlo.broadcast_in_dim %v47, dims = [1] : (tensor<64xf32>) -> tensor<64x64x40x40xf32>
    %v49 = stablehlo.divide %v48, %v40 : tensor<64x64x40x40xf32>
    %v50 = stablehlo.add %v49, %v41 : tensor<64x64x40x40xf32>
    %v51 = stablehlo.rsqrt %v50 : tensor<64x64x40x40xf32>
    %v52 = stablehlo.multiply %v45, %v51 : tensor<64x64x40x40xf32>
    %v53 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x40x40xf32>
    %v54 = stablehlo.broadcast_in_dim %s1b0bt1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x40x40xf32>
    %v55 = stablehlo.multiply %v52, %v53 : tensor<64x64x40x40xf32>
    %v56 = stablehlo.add %v55, %v54 : tensor<64x64x40x40xf32>
    %v57 = stablehlo.reshape %v56 : (tensor<64x64x40x40xf32>) -> tensor<64x102400xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<64x102400xf32>) -> tensor<64x64x40x40xf32>
    %v59 = stablehlo.constant dense<0.0> : tensor<64x64x40x40xf32>
    %v60 = stablehlo.maximum %v58, %v59 : tensor<64x64x40x40xf32>
    %v61 = stablehlo.reshape %v60 : (tensor<64x64x40x40xf32>) -> tensor<64x102400xf32>
    %v62 = stablehlo.reshape %v61 : (tensor<64x102400xf32>) -> tensor<64x64x40x40xf32>
    %v63 = stablehlo.convolution(%v62, %s1b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x40x40xf32>, tensor<64x64x3x3xf32>) -> tensor<64x64x40x40xf32>
    %v64 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x40x40xf32>
    %v65 = stablehlo.add %v63, %v64 : tensor<64x64x40x40xf32>
    %v66 = stablehlo.reshape %v65 : (tensor<64x64x40x40xf32>) -> tensor<64x102400xf32>
    %v67 = stablehlo.reshape %v66 : (tensor<64x102400xf32>) -> tensor<64x64x40x40xf32>
    %v68 = stablehlo.constant dense<0.0> : tensor<f32>
    %v69 = stablehlo.constant dense<102400.0> : tensor<64x64x40x40xf32>
    %v70 = stablehlo.constant dense<1.0e-05> : tensor<64x64x40x40xf32>
    %v71 = stablehlo.reduce(%v67 init: %v68) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x40x40xf32>, tensor<f32>) -> tensor<64xf32>
    %v72 = stablehlo.broadcast_in_dim %v71, dims = [1] : (tensor<64xf32>) -> tensor<64x64x40x40xf32>
    %v73 = stablehlo.divide %v72, %v69 : tensor<64x64x40x40xf32>
    %v74 = stablehlo.subtract %v67, %v73 : tensor<64x64x40x40xf32>
    %v75 = stablehlo.multiply %v74, %v74 : tensor<64x64x40x40xf32>
    %v76 = stablehlo.reduce(%v75 init: %v68) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x40x40xf32>, tensor<f32>) -> tensor<64xf32>
    %v77 = stablehlo.broadcast_in_dim %v76, dims = [1] : (tensor<64xf32>) -> tensor<64x64x40x40xf32>
    %v78 = stablehlo.divide %v77, %v69 : tensor<64x64x40x40xf32>
    %v79 = stablehlo.add %v78, %v70 : tensor<64x64x40x40xf32>
    %v80 = stablehlo.rsqrt %v79 : tensor<64x64x40x40xf32>
    %v81 = stablehlo.multiply %v74, %v80 : tensor<64x64x40x40xf32>
    %v82 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x40x40xf32>
    %v83 = stablehlo.broadcast_in_dim %s1b0bt2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x40x40xf32>
    %v84 = stablehlo.multiply %v81, %v82 : tensor<64x64x40x40xf32>
    %v85 = stablehlo.add %v84, %v83 : tensor<64x64x40x40xf32>
    %v86 = stablehlo.reshape %v85 : (tensor<64x64x40x40xf32>) -> tensor<64x102400xf32>
    %v87 = stablehlo.reshape %v86 : (tensor<64x102400xf32>) -> tensor<64x64x40x40xf32>
    %v88 = stablehlo.constant dense<0.0> : tensor<64x64x40x40xf32>
    %v89 = stablehlo.maximum %v87, %v88 : tensor<64x64x40x40xf32>
    %v90 = stablehlo.reshape %v89 : (tensor<64x64x40x40xf32>) -> tensor<64x102400xf32>
    %v91 = stablehlo.reshape %v90 : (tensor<64x102400xf32>) -> tensor<64x64x40x40xf32>
    %v92 = stablehlo.convolution(%v91, %s1b0W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x40x40xf32>, tensor<256x64x1x1xf32>) -> tensor<64x256x40x40xf32>
    %v93 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x40x40xf32>
    %v94 = stablehlo.add %v92, %v93 : tensor<64x256x40x40xf32>
    %v95 = stablehlo.reshape %v94 : (tensor<64x256x40x40xf32>) -> tensor<64x409600xf32>
    %v96 = stablehlo.reshape %v95 : (tensor<64x409600xf32>) -> tensor<64x256x40x40xf32>
    %v97 = stablehlo.constant dense<0.0> : tensor<f32>
    %v98 = stablehlo.constant dense<102400.0> : tensor<64x256x40x40xf32>
    %v99 = stablehlo.constant dense<1.0e-05> : tensor<64x256x40x40xf32>
    %v100 = stablehlo.reduce(%v96 init: %v97) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x40x40xf32>, tensor<f32>) -> tensor<256xf32>
    %v101 = stablehlo.broadcast_in_dim %v100, dims = [1] : (tensor<256xf32>) -> tensor<64x256x40x40xf32>
    %v102 = stablehlo.divide %v101, %v98 : tensor<64x256x40x40xf32>
    %v103 = stablehlo.subtract %v96, %v102 : tensor<64x256x40x40xf32>
    %v104 = stablehlo.multiply %v103, %v103 : tensor<64x256x40x40xf32>
    %v105 = stablehlo.reduce(%v104 init: %v97) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x40x40xf32>, tensor<f32>) -> tensor<256xf32>
    %v106 = stablehlo.broadcast_in_dim %v105, dims = [1] : (tensor<256xf32>) -> tensor<64x256x40x40xf32>
    %v107 = stablehlo.divide %v106, %v98 : tensor<64x256x40x40xf32>
    %v108 = stablehlo.add %v107, %v99 : tensor<64x256x40x40xf32>
    %v109 = stablehlo.rsqrt %v108 : tensor<64x256x40x40xf32>
    %v110 = stablehlo.multiply %v103, %v109 : tensor<64x256x40x40xf32>
    %v111 = stablehlo.broadcast_in_dim %s1b0g3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x40x40xf32>
    %v112 = stablehlo.broadcast_in_dim %s1b0bt3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x40x40xf32>
    %v113 = stablehlo.multiply %v110, %v111 : tensor<64x256x40x40xf32>
    %v114 = stablehlo.add %v113, %v112 : tensor<64x256x40x40xf32>
    %v115 = stablehlo.reshape %v114 : (tensor<64x256x40x40xf32>) -> tensor<64x409600xf32>
    %v116 = stablehlo.reshape %v32 : (tensor<64x102400xf32>) -> tensor<64x64x40x40xf32>
    %v117 = stablehlo.convolution(%v116, %s1b0Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x40x40xf32>, tensor<256x64x1x1xf32>) -> tensor<64x256x40x40xf32>
    %v118 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x40x40xf32>
    %v119 = stablehlo.add %v117, %v118 : tensor<64x256x40x40xf32>
    %v120 = stablehlo.reshape %v119 : (tensor<64x256x40x40xf32>) -> tensor<64x409600xf32>
    %v121 = stablehlo.reshape %v120 : (tensor<64x409600xf32>) -> tensor<64x256x40x40xf32>
    %v122 = stablehlo.constant dense<0.0> : tensor<f32>
    %v123 = stablehlo.constant dense<102400.0> : tensor<64x256x40x40xf32>
    %v124 = stablehlo.constant dense<1.0e-05> : tensor<64x256x40x40xf32>
    %v125 = stablehlo.reduce(%v121 init: %v122) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x40x40xf32>, tensor<f32>) -> tensor<256xf32>
    %v126 = stablehlo.broadcast_in_dim %v125, dims = [1] : (tensor<256xf32>) -> tensor<64x256x40x40xf32>
    %v127 = stablehlo.divide %v126, %v123 : tensor<64x256x40x40xf32>
    %v128 = stablehlo.subtract %v121, %v127 : tensor<64x256x40x40xf32>
    %v129 = stablehlo.multiply %v128, %v128 : tensor<64x256x40x40xf32>
    %v130 = stablehlo.reduce(%v129 init: %v122) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x40x40xf32>, tensor<f32>) -> tensor<256xf32>
    %v131 = stablehlo.broadcast_in_dim %v130, dims = [1] : (tensor<256xf32>) -> tensor<64x256x40x40xf32>
    %v132 = stablehlo.divide %v131, %v123 : tensor<64x256x40x40xf32>
    %v133 = stablehlo.add %v132, %v124 : tensor<64x256x40x40xf32>
    %v134 = stablehlo.rsqrt %v133 : tensor<64x256x40x40xf32>
    %v135 = stablehlo.multiply %v128, %v134 : tensor<64x256x40x40xf32>
    %v136 = stablehlo.broadcast_in_dim %s1b0gp, dims = [1] : (tensor<256xf32>) -> tensor<64x256x40x40xf32>
    %v137 = stablehlo.broadcast_in_dim %s1b0btp, dims = [1] : (tensor<256xf32>) -> tensor<64x256x40x40xf32>
    %v138 = stablehlo.multiply %v135, %v136 : tensor<64x256x40x40xf32>
    %v139 = stablehlo.add %v138, %v137 : tensor<64x256x40x40xf32>
    %v140 = stablehlo.reshape %v139 : (tensor<64x256x40x40xf32>) -> tensor<64x409600xf32>
    %v141 = stablehlo.reshape %v115 : (tensor<64x409600xf32>) -> tensor<64x256x40x40xf32>
    %v142 = stablehlo.reshape %v140 : (tensor<64x409600xf32>) -> tensor<64x256x40x40xf32>
    %v143 = stablehlo.add %v141, %v142 : tensor<64x256x40x40xf32>
    %v144 = stablehlo.reshape %v143 : (tensor<64x256x40x40xf32>) -> tensor<64x409600xf32>
    %v145 = stablehlo.reshape %v144 : (tensor<64x409600xf32>) -> tensor<64x256x40x40xf32>
    %v146 = stablehlo.constant dense<0.0> : tensor<64x256x40x40xf32>
    %v147 = stablehlo.maximum %v145, %v146 : tensor<64x256x40x40xf32>
    %v148 = stablehlo.reshape %v147 : (tensor<64x256x40x40xf32>) -> tensor<64x409600xf32>
    %v149 = stablehlo.reshape %v148 : (tensor<64x409600xf32>) -> tensor<64x256x40x40xf32>
    %v150 = stablehlo.convolution(%v149, %s1b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x40x40xf32>, tensor<64x256x1x1xf32>) -> tensor<64x64x40x40xf32>
    %v151 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x40x40xf32>
    %v152 = stablehlo.add %v150, %v151 : tensor<64x64x40x40xf32>
    %v153 = stablehlo.reshape %v152 : (tensor<64x64x40x40xf32>) -> tensor<64x102400xf32>
    %v154 = stablehlo.reshape %v153 : (tensor<64x102400xf32>) -> tensor<64x64x40x40xf32>
    %v155 = stablehlo.constant dense<0.0> : tensor<f32>
    %v156 = stablehlo.constant dense<102400.0> : tensor<64x64x40x40xf32>
    %v157 = stablehlo.constant dense<1.0e-05> : tensor<64x64x40x40xf32>
    %v158 = stablehlo.reduce(%v154 init: %v155) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x40x40xf32>, tensor<f32>) -> tensor<64xf32>
    %v159 = stablehlo.broadcast_in_dim %v158, dims = [1] : (tensor<64xf32>) -> tensor<64x64x40x40xf32>
    %v160 = stablehlo.divide %v159, %v156 : tensor<64x64x40x40xf32>
    %v161 = stablehlo.subtract %v154, %v160 : tensor<64x64x40x40xf32>
    %v162 = stablehlo.multiply %v161, %v161 : tensor<64x64x40x40xf32>
    %v163 = stablehlo.reduce(%v162 init: %v155) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x40x40xf32>, tensor<f32>) -> tensor<64xf32>
    %v164 = stablehlo.broadcast_in_dim %v163, dims = [1] : (tensor<64xf32>) -> tensor<64x64x40x40xf32>
    %v165 = stablehlo.divide %v164, %v156 : tensor<64x64x40x40xf32>
    %v166 = stablehlo.add %v165, %v157 : tensor<64x64x40x40xf32>
    %v167 = stablehlo.rsqrt %v166 : tensor<64x64x40x40xf32>
    %v168 = stablehlo.multiply %v161, %v167 : tensor<64x64x40x40xf32>
    %v169 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x40x40xf32>
    %v170 = stablehlo.broadcast_in_dim %s1b1bt1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x40x40xf32>
    %v171 = stablehlo.multiply %v168, %v169 : tensor<64x64x40x40xf32>
    %v172 = stablehlo.add %v171, %v170 : tensor<64x64x40x40xf32>
    %v173 = stablehlo.reshape %v172 : (tensor<64x64x40x40xf32>) -> tensor<64x102400xf32>
    %v174 = stablehlo.reshape %v173 : (tensor<64x102400xf32>) -> tensor<64x64x40x40xf32>
    %v175 = stablehlo.constant dense<0.0> : tensor<64x64x40x40xf32>
    %v176 = stablehlo.maximum %v174, %v175 : tensor<64x64x40x40xf32>
    %v177 = stablehlo.reshape %v176 : (tensor<64x64x40x40xf32>) -> tensor<64x102400xf32>
    %v178 = stablehlo.reshape %v177 : (tensor<64x102400xf32>) -> tensor<64x64x40x40xf32>
    %v179 = stablehlo.convolution(%v178, %s1b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x40x40xf32>, tensor<64x64x3x3xf32>) -> tensor<64x64x40x40xf32>
    %v180 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x40x40xf32>
    %v181 = stablehlo.add %v179, %v180 : tensor<64x64x40x40xf32>
    %v182 = stablehlo.reshape %v181 : (tensor<64x64x40x40xf32>) -> tensor<64x102400xf32>
    %v183 = stablehlo.reshape %v182 : (tensor<64x102400xf32>) -> tensor<64x64x40x40xf32>
    %v184 = stablehlo.constant dense<0.0> : tensor<f32>
    %v185 = stablehlo.constant dense<102400.0> : tensor<64x64x40x40xf32>
    %v186 = stablehlo.constant dense<1.0e-05> : tensor<64x64x40x40xf32>
    %v187 = stablehlo.reduce(%v183 init: %v184) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x40x40xf32>, tensor<f32>) -> tensor<64xf32>
    %v188 = stablehlo.broadcast_in_dim %v187, dims = [1] : (tensor<64xf32>) -> tensor<64x64x40x40xf32>
    %v189 = stablehlo.divide %v188, %v185 : tensor<64x64x40x40xf32>
    %v190 = stablehlo.subtract %v183, %v189 : tensor<64x64x40x40xf32>
    %v191 = stablehlo.multiply %v190, %v190 : tensor<64x64x40x40xf32>
    %v192 = stablehlo.reduce(%v191 init: %v184) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x40x40xf32>, tensor<f32>) -> tensor<64xf32>
    %v193 = stablehlo.broadcast_in_dim %v192, dims = [1] : (tensor<64xf32>) -> tensor<64x64x40x40xf32>
    %v194 = stablehlo.divide %v193, %v185 : tensor<64x64x40x40xf32>
    %v195 = stablehlo.add %v194, %v186 : tensor<64x64x40x40xf32>
    %v196 = stablehlo.rsqrt %v195 : tensor<64x64x40x40xf32>
    %v197 = stablehlo.multiply %v190, %v196 : tensor<64x64x40x40xf32>
    %v198 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x40x40xf32>
    %v199 = stablehlo.broadcast_in_dim %s1b1bt2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x40x40xf32>
    %v200 = stablehlo.multiply %v197, %v198 : tensor<64x64x40x40xf32>
    %v201 = stablehlo.add %v200, %v199 : tensor<64x64x40x40xf32>
    %v202 = stablehlo.reshape %v201 : (tensor<64x64x40x40xf32>) -> tensor<64x102400xf32>
    %v203 = stablehlo.reshape %v202 : (tensor<64x102400xf32>) -> tensor<64x64x40x40xf32>
    %v204 = stablehlo.constant dense<0.0> : tensor<64x64x40x40xf32>
    %v205 = stablehlo.maximum %v203, %v204 : tensor<64x64x40x40xf32>
    %v206 = stablehlo.reshape %v205 : (tensor<64x64x40x40xf32>) -> tensor<64x102400xf32>
    %v207 = stablehlo.reshape %v206 : (tensor<64x102400xf32>) -> tensor<64x64x40x40xf32>
    %v208 = stablehlo.convolution(%v207, %s1b1W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x40x40xf32>, tensor<256x64x1x1xf32>) -> tensor<64x256x40x40xf32>
    %v209 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x40x40xf32>
    %v210 = stablehlo.add %v208, %v209 : tensor<64x256x40x40xf32>
    %v211 = stablehlo.reshape %v210 : (tensor<64x256x40x40xf32>) -> tensor<64x409600xf32>
    %v212 = stablehlo.reshape %v211 : (tensor<64x409600xf32>) -> tensor<64x256x40x40xf32>
    %v213 = stablehlo.constant dense<0.0> : tensor<f32>
    %v214 = stablehlo.constant dense<102400.0> : tensor<64x256x40x40xf32>
    %v215 = stablehlo.constant dense<1.0e-05> : tensor<64x256x40x40xf32>
    %v216 = stablehlo.reduce(%v212 init: %v213) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x40x40xf32>, tensor<f32>) -> tensor<256xf32>
    %v217 = stablehlo.broadcast_in_dim %v216, dims = [1] : (tensor<256xf32>) -> tensor<64x256x40x40xf32>
    %v218 = stablehlo.divide %v217, %v214 : tensor<64x256x40x40xf32>
    %v219 = stablehlo.subtract %v212, %v218 : tensor<64x256x40x40xf32>
    %v220 = stablehlo.multiply %v219, %v219 : tensor<64x256x40x40xf32>
    %v221 = stablehlo.reduce(%v220 init: %v213) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x40x40xf32>, tensor<f32>) -> tensor<256xf32>
    %v222 = stablehlo.broadcast_in_dim %v221, dims = [1] : (tensor<256xf32>) -> tensor<64x256x40x40xf32>
    %v223 = stablehlo.divide %v222, %v214 : tensor<64x256x40x40xf32>
    %v224 = stablehlo.add %v223, %v215 : tensor<64x256x40x40xf32>
    %v225 = stablehlo.rsqrt %v224 : tensor<64x256x40x40xf32>
    %v226 = stablehlo.multiply %v219, %v225 : tensor<64x256x40x40xf32>
    %v227 = stablehlo.broadcast_in_dim %s1b1g3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x40x40xf32>
    %v228 = stablehlo.broadcast_in_dim %s1b1bt3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x40x40xf32>
    %v229 = stablehlo.multiply %v226, %v227 : tensor<64x256x40x40xf32>
    %v230 = stablehlo.add %v229, %v228 : tensor<64x256x40x40xf32>
    %v231 = stablehlo.reshape %v230 : (tensor<64x256x40x40xf32>) -> tensor<64x409600xf32>
    %v232 = stablehlo.reshape %v231 : (tensor<64x409600xf32>) -> tensor<64x256x40x40xf32>
    %v233 = stablehlo.reshape %v148 : (tensor<64x409600xf32>) -> tensor<64x256x40x40xf32>
    %v234 = stablehlo.add %v232, %v233 : tensor<64x256x40x40xf32>
    %v235 = stablehlo.reshape %v234 : (tensor<64x256x40x40xf32>) -> tensor<64x409600xf32>
    %v236 = stablehlo.reshape %v235 : (tensor<64x409600xf32>) -> tensor<64x256x40x40xf32>
    %v237 = stablehlo.constant dense<0.0> : tensor<64x256x40x40xf32>
    %v238 = stablehlo.maximum %v236, %v237 : tensor<64x256x40x40xf32>
    %v239 = stablehlo.reshape %v238 : (tensor<64x256x40x40xf32>) -> tensor<64x409600xf32>
    %v240 = stablehlo.reshape %v239 : (tensor<64x409600xf32>) -> tensor<64x256x40x40xf32>
    %v241 = stablehlo.convolution(%v240, %s1b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x40x40xf32>, tensor<64x256x1x1xf32>) -> tensor<64x64x40x40xf32>
    %v242 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x40x40xf32>
    %v243 = stablehlo.add %v241, %v242 : tensor<64x64x40x40xf32>
    %v244 = stablehlo.reshape %v243 : (tensor<64x64x40x40xf32>) -> tensor<64x102400xf32>
    %v245 = stablehlo.reshape %v244 : (tensor<64x102400xf32>) -> tensor<64x64x40x40xf32>
    %v246 = stablehlo.constant dense<0.0> : tensor<f32>
    %v247 = stablehlo.constant dense<102400.0> : tensor<64x64x40x40xf32>
    %v248 = stablehlo.constant dense<1.0e-05> : tensor<64x64x40x40xf32>
    %v249 = stablehlo.reduce(%v245 init: %v246) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x40x40xf32>, tensor<f32>) -> tensor<64xf32>
    %v250 = stablehlo.broadcast_in_dim %v249, dims = [1] : (tensor<64xf32>) -> tensor<64x64x40x40xf32>
    %v251 = stablehlo.divide %v250, %v247 : tensor<64x64x40x40xf32>
    %v252 = stablehlo.subtract %v245, %v251 : tensor<64x64x40x40xf32>
    %v253 = stablehlo.multiply %v252, %v252 : tensor<64x64x40x40xf32>
    %v254 = stablehlo.reduce(%v253 init: %v246) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x40x40xf32>, tensor<f32>) -> tensor<64xf32>
    %v255 = stablehlo.broadcast_in_dim %v254, dims = [1] : (tensor<64xf32>) -> tensor<64x64x40x40xf32>
    %v256 = stablehlo.divide %v255, %v247 : tensor<64x64x40x40xf32>
    %v257 = stablehlo.add %v256, %v248 : tensor<64x64x40x40xf32>
    %v258 = stablehlo.rsqrt %v257 : tensor<64x64x40x40xf32>
    %v259 = stablehlo.multiply %v252, %v258 : tensor<64x64x40x40xf32>
    %v260 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x40x40xf32>
    %v261 = stablehlo.broadcast_in_dim %s1b2bt1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x40x40xf32>
    %v262 = stablehlo.multiply %v259, %v260 : tensor<64x64x40x40xf32>
    %v263 = stablehlo.add %v262, %v261 : tensor<64x64x40x40xf32>
    %v264 = stablehlo.reshape %v263 : (tensor<64x64x40x40xf32>) -> tensor<64x102400xf32>
    %v265 = stablehlo.reshape %v264 : (tensor<64x102400xf32>) -> tensor<64x64x40x40xf32>
    %v266 = stablehlo.constant dense<0.0> : tensor<64x64x40x40xf32>
    %v267 = stablehlo.maximum %v265, %v266 : tensor<64x64x40x40xf32>
    %v268 = stablehlo.reshape %v267 : (tensor<64x64x40x40xf32>) -> tensor<64x102400xf32>
    %v269 = stablehlo.reshape %v268 : (tensor<64x102400xf32>) -> tensor<64x64x40x40xf32>
    %v270 = stablehlo.convolution(%v269, %s1b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x40x40xf32>, tensor<64x64x3x3xf32>) -> tensor<64x64x40x40xf32>
    %v271 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x40x40xf32>
    %v272 = stablehlo.add %v270, %v271 : tensor<64x64x40x40xf32>
    %v273 = stablehlo.reshape %v272 : (tensor<64x64x40x40xf32>) -> tensor<64x102400xf32>
    %v274 = stablehlo.reshape %v273 : (tensor<64x102400xf32>) -> tensor<64x64x40x40xf32>
    %v275 = stablehlo.constant dense<0.0> : tensor<f32>
    %v276 = stablehlo.constant dense<102400.0> : tensor<64x64x40x40xf32>
    %v277 = stablehlo.constant dense<1.0e-05> : tensor<64x64x40x40xf32>
    %v278 = stablehlo.reduce(%v274 init: %v275) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x40x40xf32>, tensor<f32>) -> tensor<64xf32>
    %v279 = stablehlo.broadcast_in_dim %v278, dims = [1] : (tensor<64xf32>) -> tensor<64x64x40x40xf32>
    %v280 = stablehlo.divide %v279, %v276 : tensor<64x64x40x40xf32>
    %v281 = stablehlo.subtract %v274, %v280 : tensor<64x64x40x40xf32>
    %v282 = stablehlo.multiply %v281, %v281 : tensor<64x64x40x40xf32>
    %v283 = stablehlo.reduce(%v282 init: %v275) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x40x40xf32>, tensor<f32>) -> tensor<64xf32>
    %v284 = stablehlo.broadcast_in_dim %v283, dims = [1] : (tensor<64xf32>) -> tensor<64x64x40x40xf32>
    %v285 = stablehlo.divide %v284, %v276 : tensor<64x64x40x40xf32>
    %v286 = stablehlo.add %v285, %v277 : tensor<64x64x40x40xf32>
    %v287 = stablehlo.rsqrt %v286 : tensor<64x64x40x40xf32>
    %v288 = stablehlo.multiply %v281, %v287 : tensor<64x64x40x40xf32>
    %v289 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x40x40xf32>
    %v290 = stablehlo.broadcast_in_dim %s1b2bt2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x40x40xf32>
    %v291 = stablehlo.multiply %v288, %v289 : tensor<64x64x40x40xf32>
    %v292 = stablehlo.add %v291, %v290 : tensor<64x64x40x40xf32>
    %v293 = stablehlo.reshape %v292 : (tensor<64x64x40x40xf32>) -> tensor<64x102400xf32>
    %v294 = stablehlo.reshape %v293 : (tensor<64x102400xf32>) -> tensor<64x64x40x40xf32>
    %v295 = stablehlo.constant dense<0.0> : tensor<64x64x40x40xf32>
    %v296 = stablehlo.maximum %v294, %v295 : tensor<64x64x40x40xf32>
    %v297 = stablehlo.reshape %v296 : (tensor<64x64x40x40xf32>) -> tensor<64x102400xf32>
    %v298 = stablehlo.reshape %v297 : (tensor<64x102400xf32>) -> tensor<64x64x40x40xf32>
    %v299 = stablehlo.convolution(%v298, %s1b2W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x40x40xf32>, tensor<256x64x1x1xf32>) -> tensor<64x256x40x40xf32>
    %v300 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x40x40xf32>
    %v301 = stablehlo.add %v299, %v300 : tensor<64x256x40x40xf32>
    %v302 = stablehlo.reshape %v301 : (tensor<64x256x40x40xf32>) -> tensor<64x409600xf32>
    %v303 = stablehlo.reshape %v302 : (tensor<64x409600xf32>) -> tensor<64x256x40x40xf32>
    %v304 = stablehlo.constant dense<0.0> : tensor<f32>
    %v305 = stablehlo.constant dense<102400.0> : tensor<64x256x40x40xf32>
    %v306 = stablehlo.constant dense<1.0e-05> : tensor<64x256x40x40xf32>
    %v307 = stablehlo.reduce(%v303 init: %v304) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x40x40xf32>, tensor<f32>) -> tensor<256xf32>
    %v308 = stablehlo.broadcast_in_dim %v307, dims = [1] : (tensor<256xf32>) -> tensor<64x256x40x40xf32>
    %v309 = stablehlo.divide %v308, %v305 : tensor<64x256x40x40xf32>
    %v310 = stablehlo.subtract %v303, %v309 : tensor<64x256x40x40xf32>
    %v311 = stablehlo.multiply %v310, %v310 : tensor<64x256x40x40xf32>
    %v312 = stablehlo.reduce(%v311 init: %v304) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x40x40xf32>, tensor<f32>) -> tensor<256xf32>
    %v313 = stablehlo.broadcast_in_dim %v312, dims = [1] : (tensor<256xf32>) -> tensor<64x256x40x40xf32>
    %v314 = stablehlo.divide %v313, %v305 : tensor<64x256x40x40xf32>
    %v315 = stablehlo.add %v314, %v306 : tensor<64x256x40x40xf32>
    %v316 = stablehlo.rsqrt %v315 : tensor<64x256x40x40xf32>
    %v317 = stablehlo.multiply %v310, %v316 : tensor<64x256x40x40xf32>
    %v318 = stablehlo.broadcast_in_dim %s1b2g3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x40x40xf32>
    %v319 = stablehlo.broadcast_in_dim %s1b2bt3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x40x40xf32>
    %v320 = stablehlo.multiply %v317, %v318 : tensor<64x256x40x40xf32>
    %v321 = stablehlo.add %v320, %v319 : tensor<64x256x40x40xf32>
    %v322 = stablehlo.reshape %v321 : (tensor<64x256x40x40xf32>) -> tensor<64x409600xf32>
    %v323 = stablehlo.reshape %v322 : (tensor<64x409600xf32>) -> tensor<64x256x40x40xf32>
    %v324 = stablehlo.reshape %v239 : (tensor<64x409600xf32>) -> tensor<64x256x40x40xf32>
    %v325 = stablehlo.add %v323, %v324 : tensor<64x256x40x40xf32>
    %v326 = stablehlo.reshape %v325 : (tensor<64x256x40x40xf32>) -> tensor<64x409600xf32>
    %v327 = stablehlo.reshape %v326 : (tensor<64x409600xf32>) -> tensor<64x256x40x40xf32>
    %v328 = stablehlo.constant dense<0.0> : tensor<64x256x40x40xf32>
    %v329 = stablehlo.maximum %v327, %v328 : tensor<64x256x40x40xf32>
    %v330 = stablehlo.reshape %v329 : (tensor<64x256x40x40xf32>) -> tensor<64x409600xf32>
    %v331 = stablehlo.reshape %v330 : (tensor<64x409600xf32>) -> tensor<64x256x40x40xf32>
    %v332 = stablehlo.convolution(%v331, %s2b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x40x40xf32>, tensor<128x256x1x1xf32>) -> tensor<64x128x40x40xf32>
    %v333 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x40x40xf32>
    %v334 = stablehlo.add %v332, %v333 : tensor<64x128x40x40xf32>
    %v335 = stablehlo.reshape %v334 : (tensor<64x128x40x40xf32>) -> tensor<64x204800xf32>
    %v336 = stablehlo.reshape %v335 : (tensor<64x204800xf32>) -> tensor<64x128x40x40xf32>
    %v337 = stablehlo.constant dense<0.0> : tensor<f32>
    %v338 = stablehlo.constant dense<102400.0> : tensor<64x128x40x40xf32>
    %v339 = stablehlo.constant dense<1.0e-05> : tensor<64x128x40x40xf32>
    %v340 = stablehlo.reduce(%v336 init: %v337) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x40x40xf32>, tensor<f32>) -> tensor<128xf32>
    %v341 = stablehlo.broadcast_in_dim %v340, dims = [1] : (tensor<128xf32>) -> tensor<64x128x40x40xf32>
    %v342 = stablehlo.divide %v341, %v338 : tensor<64x128x40x40xf32>
    %v343 = stablehlo.subtract %v336, %v342 : tensor<64x128x40x40xf32>
    %v344 = stablehlo.multiply %v343, %v343 : tensor<64x128x40x40xf32>
    %v345 = stablehlo.reduce(%v344 init: %v337) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x40x40xf32>, tensor<f32>) -> tensor<128xf32>
    %v346 = stablehlo.broadcast_in_dim %v345, dims = [1] : (tensor<128xf32>) -> tensor<64x128x40x40xf32>
    %v347 = stablehlo.divide %v346, %v338 : tensor<64x128x40x40xf32>
    %v348 = stablehlo.add %v347, %v339 : tensor<64x128x40x40xf32>
    %v349 = stablehlo.rsqrt %v348 : tensor<64x128x40x40xf32>
    %v350 = stablehlo.multiply %v343, %v349 : tensor<64x128x40x40xf32>
    %v351 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x40x40xf32>
    %v352 = stablehlo.broadcast_in_dim %s2b0bt1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x40x40xf32>
    %v353 = stablehlo.multiply %v350, %v351 : tensor<64x128x40x40xf32>
    %v354 = stablehlo.add %v353, %v352 : tensor<64x128x40x40xf32>
    %v355 = stablehlo.reshape %v354 : (tensor<64x128x40x40xf32>) -> tensor<64x204800xf32>
    %v356 = stablehlo.reshape %v355 : (tensor<64x204800xf32>) -> tensor<64x128x40x40xf32>
    %v357 = stablehlo.constant dense<0.0> : tensor<64x128x40x40xf32>
    %v358 = stablehlo.maximum %v356, %v357 : tensor<64x128x40x40xf32>
    %v359 = stablehlo.reshape %v358 : (tensor<64x128x40x40xf32>) -> tensor<64x204800xf32>
    %v360 = stablehlo.reshape %v359 : (tensor<64x204800xf32>) -> tensor<64x128x40x40xf32>
    %v361 = stablehlo.convolution(%v360, %s2b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x40x40xf32>, tensor<128x128x3x3xf32>) -> tensor<64x128x20x20xf32>
    %v362 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x20x20xf32>
    %v363 = stablehlo.add %v361, %v362 : tensor<64x128x20x20xf32>
    %v364 = stablehlo.reshape %v363 : (tensor<64x128x20x20xf32>) -> tensor<64x51200xf32>
    %v365 = stablehlo.reshape %v364 : (tensor<64x51200xf32>) -> tensor<64x128x20x20xf32>
    %v366 = stablehlo.constant dense<0.0> : tensor<f32>
    %v367 = stablehlo.constant dense<25600.0> : tensor<64x128x20x20xf32>
    %v368 = stablehlo.constant dense<1.0e-05> : tensor<64x128x20x20xf32>
    %v369 = stablehlo.reduce(%v365 init: %v366) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x20x20xf32>, tensor<f32>) -> tensor<128xf32>
    %v370 = stablehlo.broadcast_in_dim %v369, dims = [1] : (tensor<128xf32>) -> tensor<64x128x20x20xf32>
    %v371 = stablehlo.divide %v370, %v367 : tensor<64x128x20x20xf32>
    %v372 = stablehlo.subtract %v365, %v371 : tensor<64x128x20x20xf32>
    %v373 = stablehlo.multiply %v372, %v372 : tensor<64x128x20x20xf32>
    %v374 = stablehlo.reduce(%v373 init: %v366) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x20x20xf32>, tensor<f32>) -> tensor<128xf32>
    %v375 = stablehlo.broadcast_in_dim %v374, dims = [1] : (tensor<128xf32>) -> tensor<64x128x20x20xf32>
    %v376 = stablehlo.divide %v375, %v367 : tensor<64x128x20x20xf32>
    %v377 = stablehlo.add %v376, %v368 : tensor<64x128x20x20xf32>
    %v378 = stablehlo.rsqrt %v377 : tensor<64x128x20x20xf32>
    %v379 = stablehlo.multiply %v372, %v378 : tensor<64x128x20x20xf32>
    %v380 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x20x20xf32>
    %v381 = stablehlo.broadcast_in_dim %s2b0bt2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x20x20xf32>
    %v382 = stablehlo.multiply %v379, %v380 : tensor<64x128x20x20xf32>
    %v383 = stablehlo.add %v382, %v381 : tensor<64x128x20x20xf32>
    %v384 = stablehlo.reshape %v383 : (tensor<64x128x20x20xf32>) -> tensor<64x51200xf32>
    %v385 = stablehlo.reshape %v384 : (tensor<64x51200xf32>) -> tensor<64x128x20x20xf32>
    %v386 = stablehlo.constant dense<0.0> : tensor<64x128x20x20xf32>
    %v387 = stablehlo.maximum %v385, %v386 : tensor<64x128x20x20xf32>
    %v388 = stablehlo.reshape %v387 : (tensor<64x128x20x20xf32>) -> tensor<64x51200xf32>
    %v389 = stablehlo.reshape %v388 : (tensor<64x51200xf32>) -> tensor<64x128x20x20xf32>
    %v390 = stablehlo.convolution(%v389, %s2b0W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x20x20xf32>, tensor<512x128x1x1xf32>) -> tensor<64x512x20x20xf32>
    %v391 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x20x20xf32>
    %v392 = stablehlo.add %v390, %v391 : tensor<64x512x20x20xf32>
    %v393 = stablehlo.reshape %v392 : (tensor<64x512x20x20xf32>) -> tensor<64x204800xf32>
    %v394 = stablehlo.reshape %v393 : (tensor<64x204800xf32>) -> tensor<64x512x20x20xf32>
    %v395 = stablehlo.constant dense<0.0> : tensor<f32>
    %v396 = stablehlo.constant dense<25600.0> : tensor<64x512x20x20xf32>
    %v397 = stablehlo.constant dense<1.0e-05> : tensor<64x512x20x20xf32>
    %v398 = stablehlo.reduce(%v394 init: %v395) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x20x20xf32>, tensor<f32>) -> tensor<512xf32>
    %v399 = stablehlo.broadcast_in_dim %v398, dims = [1] : (tensor<512xf32>) -> tensor<64x512x20x20xf32>
    %v400 = stablehlo.divide %v399, %v396 : tensor<64x512x20x20xf32>
    %v401 = stablehlo.subtract %v394, %v400 : tensor<64x512x20x20xf32>
    %v402 = stablehlo.multiply %v401, %v401 : tensor<64x512x20x20xf32>
    %v403 = stablehlo.reduce(%v402 init: %v395) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x20x20xf32>, tensor<f32>) -> tensor<512xf32>
    %v404 = stablehlo.broadcast_in_dim %v403, dims = [1] : (tensor<512xf32>) -> tensor<64x512x20x20xf32>
    %v405 = stablehlo.divide %v404, %v396 : tensor<64x512x20x20xf32>
    %v406 = stablehlo.add %v405, %v397 : tensor<64x512x20x20xf32>
    %v407 = stablehlo.rsqrt %v406 : tensor<64x512x20x20xf32>
    %v408 = stablehlo.multiply %v401, %v407 : tensor<64x512x20x20xf32>
    %v409 = stablehlo.broadcast_in_dim %s2b0g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x20x20xf32>
    %v410 = stablehlo.broadcast_in_dim %s2b0bt3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x20x20xf32>
    %v411 = stablehlo.multiply %v408, %v409 : tensor<64x512x20x20xf32>
    %v412 = stablehlo.add %v411, %v410 : tensor<64x512x20x20xf32>
    %v413 = stablehlo.reshape %v412 : (tensor<64x512x20x20xf32>) -> tensor<64x204800xf32>
    %v414 = stablehlo.reshape %v330 : (tensor<64x409600xf32>) -> tensor<64x256x40x40xf32>
    %v415 = stablehlo.convolution(%v414, %s2b0Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x40x40xf32>, tensor<512x256x1x1xf32>) -> tensor<64x512x20x20xf32>
    %v416 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x20x20xf32>
    %v417 = stablehlo.add %v415, %v416 : tensor<64x512x20x20xf32>
    %v418 = stablehlo.reshape %v417 : (tensor<64x512x20x20xf32>) -> tensor<64x204800xf32>
    %v419 = stablehlo.reshape %v418 : (tensor<64x204800xf32>) -> tensor<64x512x20x20xf32>
    %v420 = stablehlo.constant dense<0.0> : tensor<f32>
    %v421 = stablehlo.constant dense<25600.0> : tensor<64x512x20x20xf32>
    %v422 = stablehlo.constant dense<1.0e-05> : tensor<64x512x20x20xf32>
    %v423 = stablehlo.reduce(%v419 init: %v420) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x20x20xf32>, tensor<f32>) -> tensor<512xf32>
    %v424 = stablehlo.broadcast_in_dim %v423, dims = [1] : (tensor<512xf32>) -> tensor<64x512x20x20xf32>
    %v425 = stablehlo.divide %v424, %v421 : tensor<64x512x20x20xf32>
    %v426 = stablehlo.subtract %v419, %v425 : tensor<64x512x20x20xf32>
    %v427 = stablehlo.multiply %v426, %v426 : tensor<64x512x20x20xf32>
    %v428 = stablehlo.reduce(%v427 init: %v420) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x20x20xf32>, tensor<f32>) -> tensor<512xf32>
    %v429 = stablehlo.broadcast_in_dim %v428, dims = [1] : (tensor<512xf32>) -> tensor<64x512x20x20xf32>
    %v430 = stablehlo.divide %v429, %v421 : tensor<64x512x20x20xf32>
    %v431 = stablehlo.add %v430, %v422 : tensor<64x512x20x20xf32>
    %v432 = stablehlo.rsqrt %v431 : tensor<64x512x20x20xf32>
    %v433 = stablehlo.multiply %v426, %v432 : tensor<64x512x20x20xf32>
    %v434 = stablehlo.broadcast_in_dim %s2b0gp, dims = [1] : (tensor<512xf32>) -> tensor<64x512x20x20xf32>
    %v435 = stablehlo.broadcast_in_dim %s2b0btp, dims = [1] : (tensor<512xf32>) -> tensor<64x512x20x20xf32>
    %v436 = stablehlo.multiply %v433, %v434 : tensor<64x512x20x20xf32>
    %v437 = stablehlo.add %v436, %v435 : tensor<64x512x20x20xf32>
    %v438 = stablehlo.reshape %v437 : (tensor<64x512x20x20xf32>) -> tensor<64x204800xf32>
    %v439 = stablehlo.reshape %v413 : (tensor<64x204800xf32>) -> tensor<64x512x20x20xf32>
    %v440 = stablehlo.reshape %v438 : (tensor<64x204800xf32>) -> tensor<64x512x20x20xf32>
    %v441 = stablehlo.add %v439, %v440 : tensor<64x512x20x20xf32>
    %v442 = stablehlo.reshape %v441 : (tensor<64x512x20x20xf32>) -> tensor<64x204800xf32>
    %v443 = stablehlo.reshape %v442 : (tensor<64x204800xf32>) -> tensor<64x512x20x20xf32>
    %v444 = stablehlo.constant dense<0.0> : tensor<64x512x20x20xf32>
    %v445 = stablehlo.maximum %v443, %v444 : tensor<64x512x20x20xf32>
    %v446 = stablehlo.reshape %v445 : (tensor<64x512x20x20xf32>) -> tensor<64x204800xf32>
    %v447 = stablehlo.reshape %v446 : (tensor<64x204800xf32>) -> tensor<64x512x20x20xf32>
    %v448 = stablehlo.convolution(%v447, %s2b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x20x20xf32>, tensor<128x512x1x1xf32>) -> tensor<64x128x20x20xf32>
    %v449 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x20x20xf32>
    %v450 = stablehlo.add %v448, %v449 : tensor<64x128x20x20xf32>
    %v451 = stablehlo.reshape %v450 : (tensor<64x128x20x20xf32>) -> tensor<64x51200xf32>
    %v452 = stablehlo.reshape %v451 : (tensor<64x51200xf32>) -> tensor<64x128x20x20xf32>
    %v453 = stablehlo.constant dense<0.0> : tensor<f32>
    %v454 = stablehlo.constant dense<25600.0> : tensor<64x128x20x20xf32>
    %v455 = stablehlo.constant dense<1.0e-05> : tensor<64x128x20x20xf32>
    %v456 = stablehlo.reduce(%v452 init: %v453) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x20x20xf32>, tensor<f32>) -> tensor<128xf32>
    %v457 = stablehlo.broadcast_in_dim %v456, dims = [1] : (tensor<128xf32>) -> tensor<64x128x20x20xf32>
    %v458 = stablehlo.divide %v457, %v454 : tensor<64x128x20x20xf32>
    %v459 = stablehlo.subtract %v452, %v458 : tensor<64x128x20x20xf32>
    %v460 = stablehlo.multiply %v459, %v459 : tensor<64x128x20x20xf32>
    %v461 = stablehlo.reduce(%v460 init: %v453) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x20x20xf32>, tensor<f32>) -> tensor<128xf32>
    %v462 = stablehlo.broadcast_in_dim %v461, dims = [1] : (tensor<128xf32>) -> tensor<64x128x20x20xf32>
    %v463 = stablehlo.divide %v462, %v454 : tensor<64x128x20x20xf32>
    %v464 = stablehlo.add %v463, %v455 : tensor<64x128x20x20xf32>
    %v465 = stablehlo.rsqrt %v464 : tensor<64x128x20x20xf32>
    %v466 = stablehlo.multiply %v459, %v465 : tensor<64x128x20x20xf32>
    %v467 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x20x20xf32>
    %v468 = stablehlo.broadcast_in_dim %s2b1bt1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x20x20xf32>
    %v469 = stablehlo.multiply %v466, %v467 : tensor<64x128x20x20xf32>
    %v470 = stablehlo.add %v469, %v468 : tensor<64x128x20x20xf32>
    %v471 = stablehlo.reshape %v470 : (tensor<64x128x20x20xf32>) -> tensor<64x51200xf32>
    %v472 = stablehlo.reshape %v471 : (tensor<64x51200xf32>) -> tensor<64x128x20x20xf32>
    %v473 = stablehlo.constant dense<0.0> : tensor<64x128x20x20xf32>
    %v474 = stablehlo.maximum %v472, %v473 : tensor<64x128x20x20xf32>
    %v475 = stablehlo.reshape %v474 : (tensor<64x128x20x20xf32>) -> tensor<64x51200xf32>
    %v476 = stablehlo.reshape %v475 : (tensor<64x51200xf32>) -> tensor<64x128x20x20xf32>
    %v477 = stablehlo.convolution(%v476, %s2b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x20x20xf32>, tensor<128x128x3x3xf32>) -> tensor<64x128x20x20xf32>
    %v478 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x20x20xf32>
    %v479 = stablehlo.add %v477, %v478 : tensor<64x128x20x20xf32>
    %v480 = stablehlo.reshape %v479 : (tensor<64x128x20x20xf32>) -> tensor<64x51200xf32>
    %v481 = stablehlo.reshape %v480 : (tensor<64x51200xf32>) -> tensor<64x128x20x20xf32>
    %v482 = stablehlo.constant dense<0.0> : tensor<f32>
    %v483 = stablehlo.constant dense<25600.0> : tensor<64x128x20x20xf32>
    %v484 = stablehlo.constant dense<1.0e-05> : tensor<64x128x20x20xf32>
    %v485 = stablehlo.reduce(%v481 init: %v482) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x20x20xf32>, tensor<f32>) -> tensor<128xf32>
    %v486 = stablehlo.broadcast_in_dim %v485, dims = [1] : (tensor<128xf32>) -> tensor<64x128x20x20xf32>
    %v487 = stablehlo.divide %v486, %v483 : tensor<64x128x20x20xf32>
    %v488 = stablehlo.subtract %v481, %v487 : tensor<64x128x20x20xf32>
    %v489 = stablehlo.multiply %v488, %v488 : tensor<64x128x20x20xf32>
    %v490 = stablehlo.reduce(%v489 init: %v482) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x20x20xf32>, tensor<f32>) -> tensor<128xf32>
    %v491 = stablehlo.broadcast_in_dim %v490, dims = [1] : (tensor<128xf32>) -> tensor<64x128x20x20xf32>
    %v492 = stablehlo.divide %v491, %v483 : tensor<64x128x20x20xf32>
    %v493 = stablehlo.add %v492, %v484 : tensor<64x128x20x20xf32>
    %v494 = stablehlo.rsqrt %v493 : tensor<64x128x20x20xf32>
    %v495 = stablehlo.multiply %v488, %v494 : tensor<64x128x20x20xf32>
    %v496 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x20x20xf32>
    %v497 = stablehlo.broadcast_in_dim %s2b1bt2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x20x20xf32>
    %v498 = stablehlo.multiply %v495, %v496 : tensor<64x128x20x20xf32>
    %v499 = stablehlo.add %v498, %v497 : tensor<64x128x20x20xf32>
    %v500 = stablehlo.reshape %v499 : (tensor<64x128x20x20xf32>) -> tensor<64x51200xf32>
    %v501 = stablehlo.reshape %v500 : (tensor<64x51200xf32>) -> tensor<64x128x20x20xf32>
    %v502 = stablehlo.constant dense<0.0> : tensor<64x128x20x20xf32>
    %v503 = stablehlo.maximum %v501, %v502 : tensor<64x128x20x20xf32>
    %v504 = stablehlo.reshape %v503 : (tensor<64x128x20x20xf32>) -> tensor<64x51200xf32>
    %v505 = stablehlo.reshape %v504 : (tensor<64x51200xf32>) -> tensor<64x128x20x20xf32>
    %v506 = stablehlo.convolution(%v505, %s2b1W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x20x20xf32>, tensor<512x128x1x1xf32>) -> tensor<64x512x20x20xf32>
    %v507 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x20x20xf32>
    %v508 = stablehlo.add %v506, %v507 : tensor<64x512x20x20xf32>
    %v509 = stablehlo.reshape %v508 : (tensor<64x512x20x20xf32>) -> tensor<64x204800xf32>
    %v510 = stablehlo.reshape %v509 : (tensor<64x204800xf32>) -> tensor<64x512x20x20xf32>
    %v511 = stablehlo.constant dense<0.0> : tensor<f32>
    %v512 = stablehlo.constant dense<25600.0> : tensor<64x512x20x20xf32>
    %v513 = stablehlo.constant dense<1.0e-05> : tensor<64x512x20x20xf32>
    %v514 = stablehlo.reduce(%v510 init: %v511) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x20x20xf32>, tensor<f32>) -> tensor<512xf32>
    %v515 = stablehlo.broadcast_in_dim %v514, dims = [1] : (tensor<512xf32>) -> tensor<64x512x20x20xf32>
    %v516 = stablehlo.divide %v515, %v512 : tensor<64x512x20x20xf32>
    %v517 = stablehlo.subtract %v510, %v516 : tensor<64x512x20x20xf32>
    %v518 = stablehlo.multiply %v517, %v517 : tensor<64x512x20x20xf32>
    %v519 = stablehlo.reduce(%v518 init: %v511) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x20x20xf32>, tensor<f32>) -> tensor<512xf32>
    %v520 = stablehlo.broadcast_in_dim %v519, dims = [1] : (tensor<512xf32>) -> tensor<64x512x20x20xf32>
    %v521 = stablehlo.divide %v520, %v512 : tensor<64x512x20x20xf32>
    %v522 = stablehlo.add %v521, %v513 : tensor<64x512x20x20xf32>
    %v523 = stablehlo.rsqrt %v522 : tensor<64x512x20x20xf32>
    %v524 = stablehlo.multiply %v517, %v523 : tensor<64x512x20x20xf32>
    %v525 = stablehlo.broadcast_in_dim %s2b1g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x20x20xf32>
    %v526 = stablehlo.broadcast_in_dim %s2b1bt3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x20x20xf32>
    %v527 = stablehlo.multiply %v524, %v525 : tensor<64x512x20x20xf32>
    %v528 = stablehlo.add %v527, %v526 : tensor<64x512x20x20xf32>
    %v529 = stablehlo.reshape %v528 : (tensor<64x512x20x20xf32>) -> tensor<64x204800xf32>
    %v530 = stablehlo.reshape %v529 : (tensor<64x204800xf32>) -> tensor<64x512x20x20xf32>
    %v531 = stablehlo.reshape %v446 : (tensor<64x204800xf32>) -> tensor<64x512x20x20xf32>
    %v532 = stablehlo.add %v530, %v531 : tensor<64x512x20x20xf32>
    %v533 = stablehlo.reshape %v532 : (tensor<64x512x20x20xf32>) -> tensor<64x204800xf32>
    %v534 = stablehlo.reshape %v533 : (tensor<64x204800xf32>) -> tensor<64x512x20x20xf32>
    %v535 = stablehlo.constant dense<0.0> : tensor<64x512x20x20xf32>
    %v536 = stablehlo.maximum %v534, %v535 : tensor<64x512x20x20xf32>
    %v537 = stablehlo.reshape %v536 : (tensor<64x512x20x20xf32>) -> tensor<64x204800xf32>
    %v538 = stablehlo.reshape %v537 : (tensor<64x204800xf32>) -> tensor<64x512x20x20xf32>
    %v539 = stablehlo.convolution(%v538, %s2b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x20x20xf32>, tensor<128x512x1x1xf32>) -> tensor<64x128x20x20xf32>
    %v540 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x20x20xf32>
    %v541 = stablehlo.add %v539, %v540 : tensor<64x128x20x20xf32>
    %v542 = stablehlo.reshape %v541 : (tensor<64x128x20x20xf32>) -> tensor<64x51200xf32>
    %v543 = stablehlo.reshape %v542 : (tensor<64x51200xf32>) -> tensor<64x128x20x20xf32>
    %v544 = stablehlo.constant dense<0.0> : tensor<f32>
    %v545 = stablehlo.constant dense<25600.0> : tensor<64x128x20x20xf32>
    %v546 = stablehlo.constant dense<1.0e-05> : tensor<64x128x20x20xf32>
    %v547 = stablehlo.reduce(%v543 init: %v544) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x20x20xf32>, tensor<f32>) -> tensor<128xf32>
    %v548 = stablehlo.broadcast_in_dim %v547, dims = [1] : (tensor<128xf32>) -> tensor<64x128x20x20xf32>
    %v549 = stablehlo.divide %v548, %v545 : tensor<64x128x20x20xf32>
    %v550 = stablehlo.subtract %v543, %v549 : tensor<64x128x20x20xf32>
    %v551 = stablehlo.multiply %v550, %v550 : tensor<64x128x20x20xf32>
    %v552 = stablehlo.reduce(%v551 init: %v544) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x20x20xf32>, tensor<f32>) -> tensor<128xf32>
    %v553 = stablehlo.broadcast_in_dim %v552, dims = [1] : (tensor<128xf32>) -> tensor<64x128x20x20xf32>
    %v554 = stablehlo.divide %v553, %v545 : tensor<64x128x20x20xf32>
    %v555 = stablehlo.add %v554, %v546 : tensor<64x128x20x20xf32>
    %v556 = stablehlo.rsqrt %v555 : tensor<64x128x20x20xf32>
    %v557 = stablehlo.multiply %v550, %v556 : tensor<64x128x20x20xf32>
    %v558 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x20x20xf32>
    %v559 = stablehlo.broadcast_in_dim %s2b2bt1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x20x20xf32>
    %v560 = stablehlo.multiply %v557, %v558 : tensor<64x128x20x20xf32>
    %v561 = stablehlo.add %v560, %v559 : tensor<64x128x20x20xf32>
    %v562 = stablehlo.reshape %v561 : (tensor<64x128x20x20xf32>) -> tensor<64x51200xf32>
    %v563 = stablehlo.reshape %v562 : (tensor<64x51200xf32>) -> tensor<64x128x20x20xf32>
    %v564 = stablehlo.constant dense<0.0> : tensor<64x128x20x20xf32>
    %v565 = stablehlo.maximum %v563, %v564 : tensor<64x128x20x20xf32>
    %v566 = stablehlo.reshape %v565 : (tensor<64x128x20x20xf32>) -> tensor<64x51200xf32>
    %v567 = stablehlo.reshape %v566 : (tensor<64x51200xf32>) -> tensor<64x128x20x20xf32>
    %v568 = stablehlo.convolution(%v567, %s2b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x20x20xf32>, tensor<128x128x3x3xf32>) -> tensor<64x128x20x20xf32>
    %v569 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x20x20xf32>
    %v570 = stablehlo.add %v568, %v569 : tensor<64x128x20x20xf32>
    %v571 = stablehlo.reshape %v570 : (tensor<64x128x20x20xf32>) -> tensor<64x51200xf32>
    %v572 = stablehlo.reshape %v571 : (tensor<64x51200xf32>) -> tensor<64x128x20x20xf32>
    %v573 = stablehlo.constant dense<0.0> : tensor<f32>
    %v574 = stablehlo.constant dense<25600.0> : tensor<64x128x20x20xf32>
    %v575 = stablehlo.constant dense<1.0e-05> : tensor<64x128x20x20xf32>
    %v576 = stablehlo.reduce(%v572 init: %v573) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x20x20xf32>, tensor<f32>) -> tensor<128xf32>
    %v577 = stablehlo.broadcast_in_dim %v576, dims = [1] : (tensor<128xf32>) -> tensor<64x128x20x20xf32>
    %v578 = stablehlo.divide %v577, %v574 : tensor<64x128x20x20xf32>
    %v579 = stablehlo.subtract %v572, %v578 : tensor<64x128x20x20xf32>
    %v580 = stablehlo.multiply %v579, %v579 : tensor<64x128x20x20xf32>
    %v581 = stablehlo.reduce(%v580 init: %v573) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x20x20xf32>, tensor<f32>) -> tensor<128xf32>
    %v582 = stablehlo.broadcast_in_dim %v581, dims = [1] : (tensor<128xf32>) -> tensor<64x128x20x20xf32>
    %v583 = stablehlo.divide %v582, %v574 : tensor<64x128x20x20xf32>
    %v584 = stablehlo.add %v583, %v575 : tensor<64x128x20x20xf32>
    %v585 = stablehlo.rsqrt %v584 : tensor<64x128x20x20xf32>
    %v586 = stablehlo.multiply %v579, %v585 : tensor<64x128x20x20xf32>
    %v587 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x20x20xf32>
    %v588 = stablehlo.broadcast_in_dim %s2b2bt2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x20x20xf32>
    %v589 = stablehlo.multiply %v586, %v587 : tensor<64x128x20x20xf32>
    %v590 = stablehlo.add %v589, %v588 : tensor<64x128x20x20xf32>
    %v591 = stablehlo.reshape %v590 : (tensor<64x128x20x20xf32>) -> tensor<64x51200xf32>
    %v592 = stablehlo.reshape %v591 : (tensor<64x51200xf32>) -> tensor<64x128x20x20xf32>
    %v593 = stablehlo.constant dense<0.0> : tensor<64x128x20x20xf32>
    %v594 = stablehlo.maximum %v592, %v593 : tensor<64x128x20x20xf32>
    %v595 = stablehlo.reshape %v594 : (tensor<64x128x20x20xf32>) -> tensor<64x51200xf32>
    %v596 = stablehlo.reshape %v595 : (tensor<64x51200xf32>) -> tensor<64x128x20x20xf32>
    %v597 = stablehlo.convolution(%v596, %s2b2W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x20x20xf32>, tensor<512x128x1x1xf32>) -> tensor<64x512x20x20xf32>
    %v598 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x20x20xf32>
    %v599 = stablehlo.add %v597, %v598 : tensor<64x512x20x20xf32>
    %v600 = stablehlo.reshape %v599 : (tensor<64x512x20x20xf32>) -> tensor<64x204800xf32>
    %v601 = stablehlo.reshape %v600 : (tensor<64x204800xf32>) -> tensor<64x512x20x20xf32>
    %v602 = stablehlo.constant dense<0.0> : tensor<f32>
    %v603 = stablehlo.constant dense<25600.0> : tensor<64x512x20x20xf32>
    %v604 = stablehlo.constant dense<1.0e-05> : tensor<64x512x20x20xf32>
    %v605 = stablehlo.reduce(%v601 init: %v602) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x20x20xf32>, tensor<f32>) -> tensor<512xf32>
    %v606 = stablehlo.broadcast_in_dim %v605, dims = [1] : (tensor<512xf32>) -> tensor<64x512x20x20xf32>
    %v607 = stablehlo.divide %v606, %v603 : tensor<64x512x20x20xf32>
    %v608 = stablehlo.subtract %v601, %v607 : tensor<64x512x20x20xf32>
    %v609 = stablehlo.multiply %v608, %v608 : tensor<64x512x20x20xf32>
    %v610 = stablehlo.reduce(%v609 init: %v602) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x20x20xf32>, tensor<f32>) -> tensor<512xf32>
    %v611 = stablehlo.broadcast_in_dim %v610, dims = [1] : (tensor<512xf32>) -> tensor<64x512x20x20xf32>
    %v612 = stablehlo.divide %v611, %v603 : tensor<64x512x20x20xf32>
    %v613 = stablehlo.add %v612, %v604 : tensor<64x512x20x20xf32>
    %v614 = stablehlo.rsqrt %v613 : tensor<64x512x20x20xf32>
    %v615 = stablehlo.multiply %v608, %v614 : tensor<64x512x20x20xf32>
    %v616 = stablehlo.broadcast_in_dim %s2b2g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x20x20xf32>
    %v617 = stablehlo.broadcast_in_dim %s2b2bt3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x20x20xf32>
    %v618 = stablehlo.multiply %v615, %v616 : tensor<64x512x20x20xf32>
    %v619 = stablehlo.add %v618, %v617 : tensor<64x512x20x20xf32>
    %v620 = stablehlo.reshape %v619 : (tensor<64x512x20x20xf32>) -> tensor<64x204800xf32>
    %v621 = stablehlo.reshape %v620 : (tensor<64x204800xf32>) -> tensor<64x512x20x20xf32>
    %v622 = stablehlo.reshape %v537 : (tensor<64x204800xf32>) -> tensor<64x512x20x20xf32>
    %v623 = stablehlo.add %v621, %v622 : tensor<64x512x20x20xf32>
    %v624 = stablehlo.reshape %v623 : (tensor<64x512x20x20xf32>) -> tensor<64x204800xf32>
    %v625 = stablehlo.reshape %v624 : (tensor<64x204800xf32>) -> tensor<64x512x20x20xf32>
    %v626 = stablehlo.constant dense<0.0> : tensor<64x512x20x20xf32>
    %v627 = stablehlo.maximum %v625, %v626 : tensor<64x512x20x20xf32>
    %v628 = stablehlo.reshape %v627 : (tensor<64x512x20x20xf32>) -> tensor<64x204800xf32>
    %v629 = stablehlo.reshape %v628 : (tensor<64x204800xf32>) -> tensor<64x512x20x20xf32>
    %v630 = stablehlo.convolution(%v629, %s2b3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x20x20xf32>, tensor<128x512x1x1xf32>) -> tensor<64x128x20x20xf32>
    %v631 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x20x20xf32>
    %v632 = stablehlo.add %v630, %v631 : tensor<64x128x20x20xf32>
    %v633 = stablehlo.reshape %v632 : (tensor<64x128x20x20xf32>) -> tensor<64x51200xf32>
    %v634 = stablehlo.reshape %v633 : (tensor<64x51200xf32>) -> tensor<64x128x20x20xf32>
    %v635 = stablehlo.constant dense<0.0> : tensor<f32>
    %v636 = stablehlo.constant dense<25600.0> : tensor<64x128x20x20xf32>
    %v637 = stablehlo.constant dense<1.0e-05> : tensor<64x128x20x20xf32>
    %v638 = stablehlo.reduce(%v634 init: %v635) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x20x20xf32>, tensor<f32>) -> tensor<128xf32>
    %v639 = stablehlo.broadcast_in_dim %v638, dims = [1] : (tensor<128xf32>) -> tensor<64x128x20x20xf32>
    %v640 = stablehlo.divide %v639, %v636 : tensor<64x128x20x20xf32>
    %v641 = stablehlo.subtract %v634, %v640 : tensor<64x128x20x20xf32>
    %v642 = stablehlo.multiply %v641, %v641 : tensor<64x128x20x20xf32>
    %v643 = stablehlo.reduce(%v642 init: %v635) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x20x20xf32>, tensor<f32>) -> tensor<128xf32>
    %v644 = stablehlo.broadcast_in_dim %v643, dims = [1] : (tensor<128xf32>) -> tensor<64x128x20x20xf32>
    %v645 = stablehlo.divide %v644, %v636 : tensor<64x128x20x20xf32>
    %v646 = stablehlo.add %v645, %v637 : tensor<64x128x20x20xf32>
    %v647 = stablehlo.rsqrt %v646 : tensor<64x128x20x20xf32>
    %v648 = stablehlo.multiply %v641, %v647 : tensor<64x128x20x20xf32>
    %v649 = stablehlo.broadcast_in_dim %s2b3g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x20x20xf32>
    %v650 = stablehlo.broadcast_in_dim %s2b3bt1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x20x20xf32>
    %v651 = stablehlo.multiply %v648, %v649 : tensor<64x128x20x20xf32>
    %v652 = stablehlo.add %v651, %v650 : tensor<64x128x20x20xf32>
    %v653 = stablehlo.reshape %v652 : (tensor<64x128x20x20xf32>) -> tensor<64x51200xf32>
    %v654 = stablehlo.reshape %v653 : (tensor<64x51200xf32>) -> tensor<64x128x20x20xf32>
    %v655 = stablehlo.constant dense<0.0> : tensor<64x128x20x20xf32>
    %v656 = stablehlo.maximum %v654, %v655 : tensor<64x128x20x20xf32>
    %v657 = stablehlo.reshape %v656 : (tensor<64x128x20x20xf32>) -> tensor<64x51200xf32>
    %v658 = stablehlo.reshape %v657 : (tensor<64x51200xf32>) -> tensor<64x128x20x20xf32>
    %v659 = stablehlo.convolution(%v658, %s2b3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x20x20xf32>, tensor<128x128x3x3xf32>) -> tensor<64x128x20x20xf32>
    %v660 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x20x20xf32>
    %v661 = stablehlo.add %v659, %v660 : tensor<64x128x20x20xf32>
    %v662 = stablehlo.reshape %v661 : (tensor<64x128x20x20xf32>) -> tensor<64x51200xf32>
    %v663 = stablehlo.reshape %v662 : (tensor<64x51200xf32>) -> tensor<64x128x20x20xf32>
    %v664 = stablehlo.constant dense<0.0> : tensor<f32>
    %v665 = stablehlo.constant dense<25600.0> : tensor<64x128x20x20xf32>
    %v666 = stablehlo.constant dense<1.0e-05> : tensor<64x128x20x20xf32>
    %v667 = stablehlo.reduce(%v663 init: %v664) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x20x20xf32>, tensor<f32>) -> tensor<128xf32>
    %v668 = stablehlo.broadcast_in_dim %v667, dims = [1] : (tensor<128xf32>) -> tensor<64x128x20x20xf32>
    %v669 = stablehlo.divide %v668, %v665 : tensor<64x128x20x20xf32>
    %v670 = stablehlo.subtract %v663, %v669 : tensor<64x128x20x20xf32>
    %v671 = stablehlo.multiply %v670, %v670 : tensor<64x128x20x20xf32>
    %v672 = stablehlo.reduce(%v671 init: %v664) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x20x20xf32>, tensor<f32>) -> tensor<128xf32>
    %v673 = stablehlo.broadcast_in_dim %v672, dims = [1] : (tensor<128xf32>) -> tensor<64x128x20x20xf32>
    %v674 = stablehlo.divide %v673, %v665 : tensor<64x128x20x20xf32>
    %v675 = stablehlo.add %v674, %v666 : tensor<64x128x20x20xf32>
    %v676 = stablehlo.rsqrt %v675 : tensor<64x128x20x20xf32>
    %v677 = stablehlo.multiply %v670, %v676 : tensor<64x128x20x20xf32>
    %v678 = stablehlo.broadcast_in_dim %s2b3g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x20x20xf32>
    %v679 = stablehlo.broadcast_in_dim %s2b3bt2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x20x20xf32>
    %v680 = stablehlo.multiply %v677, %v678 : tensor<64x128x20x20xf32>
    %v681 = stablehlo.add %v680, %v679 : tensor<64x128x20x20xf32>
    %v682 = stablehlo.reshape %v681 : (tensor<64x128x20x20xf32>) -> tensor<64x51200xf32>
    %v683 = stablehlo.reshape %v682 : (tensor<64x51200xf32>) -> tensor<64x128x20x20xf32>
    %v684 = stablehlo.constant dense<0.0> : tensor<64x128x20x20xf32>
    %v685 = stablehlo.maximum %v683, %v684 : tensor<64x128x20x20xf32>
    %v686 = stablehlo.reshape %v685 : (tensor<64x128x20x20xf32>) -> tensor<64x51200xf32>
    %v687 = stablehlo.reshape %v686 : (tensor<64x51200xf32>) -> tensor<64x128x20x20xf32>
    %v688 = stablehlo.convolution(%v687, %s2b3W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x20x20xf32>, tensor<512x128x1x1xf32>) -> tensor<64x512x20x20xf32>
    %v689 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x20x20xf32>
    %v690 = stablehlo.add %v688, %v689 : tensor<64x512x20x20xf32>
    %v691 = stablehlo.reshape %v690 : (tensor<64x512x20x20xf32>) -> tensor<64x204800xf32>
    %v692 = stablehlo.reshape %v691 : (tensor<64x204800xf32>) -> tensor<64x512x20x20xf32>
    %v693 = stablehlo.constant dense<0.0> : tensor<f32>
    %v694 = stablehlo.constant dense<25600.0> : tensor<64x512x20x20xf32>
    %v695 = stablehlo.constant dense<1.0e-05> : tensor<64x512x20x20xf32>
    %v696 = stablehlo.reduce(%v692 init: %v693) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x20x20xf32>, tensor<f32>) -> tensor<512xf32>
    %v697 = stablehlo.broadcast_in_dim %v696, dims = [1] : (tensor<512xf32>) -> tensor<64x512x20x20xf32>
    %v698 = stablehlo.divide %v697, %v694 : tensor<64x512x20x20xf32>
    %v699 = stablehlo.subtract %v692, %v698 : tensor<64x512x20x20xf32>
    %v700 = stablehlo.multiply %v699, %v699 : tensor<64x512x20x20xf32>
    %v701 = stablehlo.reduce(%v700 init: %v693) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x20x20xf32>, tensor<f32>) -> tensor<512xf32>
    %v702 = stablehlo.broadcast_in_dim %v701, dims = [1] : (tensor<512xf32>) -> tensor<64x512x20x20xf32>
    %v703 = stablehlo.divide %v702, %v694 : tensor<64x512x20x20xf32>
    %v704 = stablehlo.add %v703, %v695 : tensor<64x512x20x20xf32>
    %v705 = stablehlo.rsqrt %v704 : tensor<64x512x20x20xf32>
    %v706 = stablehlo.multiply %v699, %v705 : tensor<64x512x20x20xf32>
    %v707 = stablehlo.broadcast_in_dim %s2b3g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x20x20xf32>
    %v708 = stablehlo.broadcast_in_dim %s2b3bt3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x20x20xf32>
    %v709 = stablehlo.multiply %v706, %v707 : tensor<64x512x20x20xf32>
    %v710 = stablehlo.add %v709, %v708 : tensor<64x512x20x20xf32>
    %v711 = stablehlo.reshape %v710 : (tensor<64x512x20x20xf32>) -> tensor<64x204800xf32>
    %v712 = stablehlo.reshape %v711 : (tensor<64x204800xf32>) -> tensor<64x512x20x20xf32>
    %v713 = stablehlo.reshape %v628 : (tensor<64x204800xf32>) -> tensor<64x512x20x20xf32>
    %v714 = stablehlo.add %v712, %v713 : tensor<64x512x20x20xf32>
    %v715 = stablehlo.reshape %v714 : (tensor<64x512x20x20xf32>) -> tensor<64x204800xf32>
    %v716 = stablehlo.reshape %v715 : (tensor<64x204800xf32>) -> tensor<64x512x20x20xf32>
    %v717 = stablehlo.constant dense<0.0> : tensor<64x512x20x20xf32>
    %v718 = stablehlo.maximum %v716, %v717 : tensor<64x512x20x20xf32>
    %v719 = stablehlo.reshape %v718 : (tensor<64x512x20x20xf32>) -> tensor<64x204800xf32>
    %v720 = stablehlo.reshape %v719 : (tensor<64x204800xf32>) -> tensor<64x512x20x20xf32>
    %v721 = stablehlo.convolution(%v720, %s3b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x20x20xf32>, tensor<256x512x1x1xf32>) -> tensor<64x256x20x20xf32>
    %v722 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x20x20xf32>
    %v723 = stablehlo.add %v721, %v722 : tensor<64x256x20x20xf32>
    %v724 = stablehlo.reshape %v723 : (tensor<64x256x20x20xf32>) -> tensor<64x102400xf32>
    %v725 = stablehlo.reshape %v724 : (tensor<64x102400xf32>) -> tensor<64x256x20x20xf32>
    %v726 = stablehlo.constant dense<0.0> : tensor<f32>
    %v727 = stablehlo.constant dense<25600.0> : tensor<64x256x20x20xf32>
    %v728 = stablehlo.constant dense<1.0e-05> : tensor<64x256x20x20xf32>
    %v729 = stablehlo.reduce(%v725 init: %v726) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x20x20xf32>, tensor<f32>) -> tensor<256xf32>
    %v730 = stablehlo.broadcast_in_dim %v729, dims = [1] : (tensor<256xf32>) -> tensor<64x256x20x20xf32>
    %v731 = stablehlo.divide %v730, %v727 : tensor<64x256x20x20xf32>
    %v732 = stablehlo.subtract %v725, %v731 : tensor<64x256x20x20xf32>
    %v733 = stablehlo.multiply %v732, %v732 : tensor<64x256x20x20xf32>
    %v734 = stablehlo.reduce(%v733 init: %v726) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x20x20xf32>, tensor<f32>) -> tensor<256xf32>
    %v735 = stablehlo.broadcast_in_dim %v734, dims = [1] : (tensor<256xf32>) -> tensor<64x256x20x20xf32>
    %v736 = stablehlo.divide %v735, %v727 : tensor<64x256x20x20xf32>
    %v737 = stablehlo.add %v736, %v728 : tensor<64x256x20x20xf32>
    %v738 = stablehlo.rsqrt %v737 : tensor<64x256x20x20xf32>
    %v739 = stablehlo.multiply %v732, %v738 : tensor<64x256x20x20xf32>
    %v740 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x20x20xf32>
    %v741 = stablehlo.broadcast_in_dim %s3b0bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x20x20xf32>
    %v742 = stablehlo.multiply %v739, %v740 : tensor<64x256x20x20xf32>
    %v743 = stablehlo.add %v742, %v741 : tensor<64x256x20x20xf32>
    %v744 = stablehlo.reshape %v743 : (tensor<64x256x20x20xf32>) -> tensor<64x102400xf32>
    %v745 = stablehlo.reshape %v744 : (tensor<64x102400xf32>) -> tensor<64x256x20x20xf32>
    %v746 = stablehlo.constant dense<0.0> : tensor<64x256x20x20xf32>
    %v747 = stablehlo.maximum %v745, %v746 : tensor<64x256x20x20xf32>
    %v748 = stablehlo.reshape %v747 : (tensor<64x256x20x20xf32>) -> tensor<64x102400xf32>
    %v749 = stablehlo.reshape %v748 : (tensor<64x102400xf32>) -> tensor<64x256x20x20xf32>
    %v750 = stablehlo.convolution(%v749, %s3b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x20x20xf32>, tensor<256x256x3x3xf32>) -> tensor<64x256x10x10xf32>
    %v751 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v752 = stablehlo.add %v750, %v751 : tensor<64x256x10x10xf32>
    %v753 = stablehlo.reshape %v752 : (tensor<64x256x10x10xf32>) -> tensor<64x25600xf32>
    %v754 = stablehlo.reshape %v753 : (tensor<64x25600xf32>) -> tensor<64x256x10x10xf32>
    %v755 = stablehlo.constant dense<0.0> : tensor<f32>
    %v756 = stablehlo.constant dense<6400.0> : tensor<64x256x10x10xf32>
    %v757 = stablehlo.constant dense<1.0e-05> : tensor<64x256x10x10xf32>
    %v758 = stablehlo.reduce(%v754 init: %v755) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x10x10xf32>, tensor<f32>) -> tensor<256xf32>
    %v759 = stablehlo.broadcast_in_dim %v758, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v760 = stablehlo.divide %v759, %v756 : tensor<64x256x10x10xf32>
    %v761 = stablehlo.subtract %v754, %v760 : tensor<64x256x10x10xf32>
    %v762 = stablehlo.multiply %v761, %v761 : tensor<64x256x10x10xf32>
    %v763 = stablehlo.reduce(%v762 init: %v755) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x10x10xf32>, tensor<f32>) -> tensor<256xf32>
    %v764 = stablehlo.broadcast_in_dim %v763, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v765 = stablehlo.divide %v764, %v756 : tensor<64x256x10x10xf32>
    %v766 = stablehlo.add %v765, %v757 : tensor<64x256x10x10xf32>
    %v767 = stablehlo.rsqrt %v766 : tensor<64x256x10x10xf32>
    %v768 = stablehlo.multiply %v761, %v767 : tensor<64x256x10x10xf32>
    %v769 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v770 = stablehlo.broadcast_in_dim %s3b0bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v771 = stablehlo.multiply %v768, %v769 : tensor<64x256x10x10xf32>
    %v772 = stablehlo.add %v771, %v770 : tensor<64x256x10x10xf32>
    %v773 = stablehlo.reshape %v772 : (tensor<64x256x10x10xf32>) -> tensor<64x25600xf32>
    %v774 = stablehlo.reshape %v773 : (tensor<64x25600xf32>) -> tensor<64x256x10x10xf32>
    %v775 = stablehlo.constant dense<0.0> : tensor<64x256x10x10xf32>
    %v776 = stablehlo.maximum %v774, %v775 : tensor<64x256x10x10xf32>
    %v777 = stablehlo.reshape %v776 : (tensor<64x256x10x10xf32>) -> tensor<64x25600xf32>
    %v778 = stablehlo.reshape %v777 : (tensor<64x25600xf32>) -> tensor<64x256x10x10xf32>
    %v779 = stablehlo.convolution(%v778, %s3b0W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x10x10xf32>, tensor<1024x256x1x1xf32>) -> tensor<64x1024x10x10xf32>
    %v780 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x10x10xf32>
    %v781 = stablehlo.add %v779, %v780 : tensor<64x1024x10x10xf32>
    %v782 = stablehlo.reshape %v781 : (tensor<64x1024x10x10xf32>) -> tensor<64x102400xf32>
    %v783 = stablehlo.reshape %v782 : (tensor<64x102400xf32>) -> tensor<64x1024x10x10xf32>
    %v784 = stablehlo.constant dense<0.0> : tensor<f32>
    %v785 = stablehlo.constant dense<6400.0> : tensor<64x1024x10x10xf32>
    %v786 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x10x10xf32>
    %v787 = stablehlo.reduce(%v783 init: %v784) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x10x10xf32>, tensor<f32>) -> tensor<1024xf32>
    %v788 = stablehlo.broadcast_in_dim %v787, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x10x10xf32>
    %v789 = stablehlo.divide %v788, %v785 : tensor<64x1024x10x10xf32>
    %v790 = stablehlo.subtract %v783, %v789 : tensor<64x1024x10x10xf32>
    %v791 = stablehlo.multiply %v790, %v790 : tensor<64x1024x10x10xf32>
    %v792 = stablehlo.reduce(%v791 init: %v784) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x10x10xf32>, tensor<f32>) -> tensor<1024xf32>
    %v793 = stablehlo.broadcast_in_dim %v792, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x10x10xf32>
    %v794 = stablehlo.divide %v793, %v785 : tensor<64x1024x10x10xf32>
    %v795 = stablehlo.add %v794, %v786 : tensor<64x1024x10x10xf32>
    %v796 = stablehlo.rsqrt %v795 : tensor<64x1024x10x10xf32>
    %v797 = stablehlo.multiply %v790, %v796 : tensor<64x1024x10x10xf32>
    %v798 = stablehlo.broadcast_in_dim %s3b0g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x10x10xf32>
    %v799 = stablehlo.broadcast_in_dim %s3b0bt3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x10x10xf32>
    %v800 = stablehlo.multiply %v797, %v798 : tensor<64x1024x10x10xf32>
    %v801 = stablehlo.add %v800, %v799 : tensor<64x1024x10x10xf32>
    %v802 = stablehlo.reshape %v801 : (tensor<64x1024x10x10xf32>) -> tensor<64x102400xf32>
    %v803 = stablehlo.reshape %v719 : (tensor<64x204800xf32>) -> tensor<64x512x20x20xf32>
    %v804 = stablehlo.convolution(%v803, %s3b0Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x20x20xf32>, tensor<1024x512x1x1xf32>) -> tensor<64x1024x10x10xf32>
    %v805 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x10x10xf32>
    %v806 = stablehlo.add %v804, %v805 : tensor<64x1024x10x10xf32>
    %v807 = stablehlo.reshape %v806 : (tensor<64x1024x10x10xf32>) -> tensor<64x102400xf32>
    %v808 = stablehlo.reshape %v807 : (tensor<64x102400xf32>) -> tensor<64x1024x10x10xf32>
    %v809 = stablehlo.constant dense<0.0> : tensor<f32>
    %v810 = stablehlo.constant dense<6400.0> : tensor<64x1024x10x10xf32>
    %v811 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x10x10xf32>
    %v812 = stablehlo.reduce(%v808 init: %v809) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x10x10xf32>, tensor<f32>) -> tensor<1024xf32>
    %v813 = stablehlo.broadcast_in_dim %v812, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x10x10xf32>
    %v814 = stablehlo.divide %v813, %v810 : tensor<64x1024x10x10xf32>
    %v815 = stablehlo.subtract %v808, %v814 : tensor<64x1024x10x10xf32>
    %v816 = stablehlo.multiply %v815, %v815 : tensor<64x1024x10x10xf32>
    %v817 = stablehlo.reduce(%v816 init: %v809) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x10x10xf32>, tensor<f32>) -> tensor<1024xf32>
    %v818 = stablehlo.broadcast_in_dim %v817, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x10x10xf32>
    %v819 = stablehlo.divide %v818, %v810 : tensor<64x1024x10x10xf32>
    %v820 = stablehlo.add %v819, %v811 : tensor<64x1024x10x10xf32>
    %v821 = stablehlo.rsqrt %v820 : tensor<64x1024x10x10xf32>
    %v822 = stablehlo.multiply %v815, %v821 : tensor<64x1024x10x10xf32>
    %v823 = stablehlo.broadcast_in_dim %s3b0gp, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x10x10xf32>
    %v824 = stablehlo.broadcast_in_dim %s3b0btp, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x10x10xf32>
    %v825 = stablehlo.multiply %v822, %v823 : tensor<64x1024x10x10xf32>
    %v826 = stablehlo.add %v825, %v824 : tensor<64x1024x10x10xf32>
    %v827 = stablehlo.reshape %v826 : (tensor<64x1024x10x10xf32>) -> tensor<64x102400xf32>
    %v828 = stablehlo.reshape %v802 : (tensor<64x102400xf32>) -> tensor<64x1024x10x10xf32>
    %v829 = stablehlo.reshape %v827 : (tensor<64x102400xf32>) -> tensor<64x1024x10x10xf32>
    %v830 = stablehlo.add %v828, %v829 : tensor<64x1024x10x10xf32>
    %v831 = stablehlo.reshape %v830 : (tensor<64x1024x10x10xf32>) -> tensor<64x102400xf32>
    %v832 = stablehlo.reshape %v831 : (tensor<64x102400xf32>) -> tensor<64x1024x10x10xf32>
    %v833 = stablehlo.constant dense<0.0> : tensor<64x1024x10x10xf32>
    %v834 = stablehlo.maximum %v832, %v833 : tensor<64x1024x10x10xf32>
    %v835 = stablehlo.reshape %v834 : (tensor<64x1024x10x10xf32>) -> tensor<64x102400xf32>
    %v836 = stablehlo.reshape %v835 : (tensor<64x102400xf32>) -> tensor<64x1024x10x10xf32>
    %v837 = stablehlo.convolution(%v836, %s3b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x10x10xf32>, tensor<256x1024x1x1xf32>) -> tensor<64x256x10x10xf32>
    %v838 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v839 = stablehlo.add %v837, %v838 : tensor<64x256x10x10xf32>
    %v840 = stablehlo.reshape %v839 : (tensor<64x256x10x10xf32>) -> tensor<64x25600xf32>
    %v841 = stablehlo.reshape %v840 : (tensor<64x25600xf32>) -> tensor<64x256x10x10xf32>
    %v842 = stablehlo.constant dense<0.0> : tensor<f32>
    %v843 = stablehlo.constant dense<6400.0> : tensor<64x256x10x10xf32>
    %v844 = stablehlo.constant dense<1.0e-05> : tensor<64x256x10x10xf32>
    %v845 = stablehlo.reduce(%v841 init: %v842) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x10x10xf32>, tensor<f32>) -> tensor<256xf32>
    %v846 = stablehlo.broadcast_in_dim %v845, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v847 = stablehlo.divide %v846, %v843 : tensor<64x256x10x10xf32>
    %v848 = stablehlo.subtract %v841, %v847 : tensor<64x256x10x10xf32>
    %v849 = stablehlo.multiply %v848, %v848 : tensor<64x256x10x10xf32>
    %v850 = stablehlo.reduce(%v849 init: %v842) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x10x10xf32>, tensor<f32>) -> tensor<256xf32>
    %v851 = stablehlo.broadcast_in_dim %v850, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v852 = stablehlo.divide %v851, %v843 : tensor<64x256x10x10xf32>
    %v853 = stablehlo.add %v852, %v844 : tensor<64x256x10x10xf32>
    %v854 = stablehlo.rsqrt %v853 : tensor<64x256x10x10xf32>
    %v855 = stablehlo.multiply %v848, %v854 : tensor<64x256x10x10xf32>
    %v856 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v857 = stablehlo.broadcast_in_dim %s3b1bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v858 = stablehlo.multiply %v855, %v856 : tensor<64x256x10x10xf32>
    %v859 = stablehlo.add %v858, %v857 : tensor<64x256x10x10xf32>
    %v860 = stablehlo.reshape %v859 : (tensor<64x256x10x10xf32>) -> tensor<64x25600xf32>
    %v861 = stablehlo.reshape %v860 : (tensor<64x25600xf32>) -> tensor<64x256x10x10xf32>
    %v862 = stablehlo.constant dense<0.0> : tensor<64x256x10x10xf32>
    %v863 = stablehlo.maximum %v861, %v862 : tensor<64x256x10x10xf32>
    %v864 = stablehlo.reshape %v863 : (tensor<64x256x10x10xf32>) -> tensor<64x25600xf32>
    %v865 = stablehlo.reshape %v864 : (tensor<64x25600xf32>) -> tensor<64x256x10x10xf32>
    %v866 = stablehlo.convolution(%v865, %s3b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x10x10xf32>, tensor<256x256x3x3xf32>) -> tensor<64x256x10x10xf32>
    %v867 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v868 = stablehlo.add %v866, %v867 : tensor<64x256x10x10xf32>
    %v869 = stablehlo.reshape %v868 : (tensor<64x256x10x10xf32>) -> tensor<64x25600xf32>
    %v870 = stablehlo.reshape %v869 : (tensor<64x25600xf32>) -> tensor<64x256x10x10xf32>
    %v871 = stablehlo.constant dense<0.0> : tensor<f32>
    %v872 = stablehlo.constant dense<6400.0> : tensor<64x256x10x10xf32>
    %v873 = stablehlo.constant dense<1.0e-05> : tensor<64x256x10x10xf32>
    %v874 = stablehlo.reduce(%v870 init: %v871) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x10x10xf32>, tensor<f32>) -> tensor<256xf32>
    %v875 = stablehlo.broadcast_in_dim %v874, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v876 = stablehlo.divide %v875, %v872 : tensor<64x256x10x10xf32>
    %v877 = stablehlo.subtract %v870, %v876 : tensor<64x256x10x10xf32>
    %v878 = stablehlo.multiply %v877, %v877 : tensor<64x256x10x10xf32>
    %v879 = stablehlo.reduce(%v878 init: %v871) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x10x10xf32>, tensor<f32>) -> tensor<256xf32>
    %v880 = stablehlo.broadcast_in_dim %v879, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v881 = stablehlo.divide %v880, %v872 : tensor<64x256x10x10xf32>
    %v882 = stablehlo.add %v881, %v873 : tensor<64x256x10x10xf32>
    %v883 = stablehlo.rsqrt %v882 : tensor<64x256x10x10xf32>
    %v884 = stablehlo.multiply %v877, %v883 : tensor<64x256x10x10xf32>
    %v885 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v886 = stablehlo.broadcast_in_dim %s3b1bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v887 = stablehlo.multiply %v884, %v885 : tensor<64x256x10x10xf32>
    %v888 = stablehlo.add %v887, %v886 : tensor<64x256x10x10xf32>
    %v889 = stablehlo.reshape %v888 : (tensor<64x256x10x10xf32>) -> tensor<64x25600xf32>
    %v890 = stablehlo.reshape %v889 : (tensor<64x25600xf32>) -> tensor<64x256x10x10xf32>
    %v891 = stablehlo.constant dense<0.0> : tensor<64x256x10x10xf32>
    %v892 = stablehlo.maximum %v890, %v891 : tensor<64x256x10x10xf32>
    %v893 = stablehlo.reshape %v892 : (tensor<64x256x10x10xf32>) -> tensor<64x25600xf32>
    %v894 = stablehlo.reshape %v893 : (tensor<64x25600xf32>) -> tensor<64x256x10x10xf32>
    %v895 = stablehlo.convolution(%v894, %s3b1W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x10x10xf32>, tensor<1024x256x1x1xf32>) -> tensor<64x1024x10x10xf32>
    %v896 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x10x10xf32>
    %v897 = stablehlo.add %v895, %v896 : tensor<64x1024x10x10xf32>
    %v898 = stablehlo.reshape %v897 : (tensor<64x1024x10x10xf32>) -> tensor<64x102400xf32>
    %v899 = stablehlo.reshape %v898 : (tensor<64x102400xf32>) -> tensor<64x1024x10x10xf32>
    %v900 = stablehlo.constant dense<0.0> : tensor<f32>
    %v901 = stablehlo.constant dense<6400.0> : tensor<64x1024x10x10xf32>
    %v902 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x10x10xf32>
    %v903 = stablehlo.reduce(%v899 init: %v900) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x10x10xf32>, tensor<f32>) -> tensor<1024xf32>
    %v904 = stablehlo.broadcast_in_dim %v903, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x10x10xf32>
    %v905 = stablehlo.divide %v904, %v901 : tensor<64x1024x10x10xf32>
    %v906 = stablehlo.subtract %v899, %v905 : tensor<64x1024x10x10xf32>
    %v907 = stablehlo.multiply %v906, %v906 : tensor<64x1024x10x10xf32>
    %v908 = stablehlo.reduce(%v907 init: %v900) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x10x10xf32>, tensor<f32>) -> tensor<1024xf32>
    %v909 = stablehlo.broadcast_in_dim %v908, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x10x10xf32>
    %v910 = stablehlo.divide %v909, %v901 : tensor<64x1024x10x10xf32>
    %v911 = stablehlo.add %v910, %v902 : tensor<64x1024x10x10xf32>
    %v912 = stablehlo.rsqrt %v911 : tensor<64x1024x10x10xf32>
    %v913 = stablehlo.multiply %v906, %v912 : tensor<64x1024x10x10xf32>
    %v914 = stablehlo.broadcast_in_dim %s3b1g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x10x10xf32>
    %v915 = stablehlo.broadcast_in_dim %s3b1bt3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x10x10xf32>
    %v916 = stablehlo.multiply %v913, %v914 : tensor<64x1024x10x10xf32>
    %v917 = stablehlo.add %v916, %v915 : tensor<64x1024x10x10xf32>
    %v918 = stablehlo.reshape %v917 : (tensor<64x1024x10x10xf32>) -> tensor<64x102400xf32>
    %v919 = stablehlo.reshape %v918 : (tensor<64x102400xf32>) -> tensor<64x1024x10x10xf32>
    %v920 = stablehlo.reshape %v835 : (tensor<64x102400xf32>) -> tensor<64x1024x10x10xf32>
    %v921 = stablehlo.add %v919, %v920 : tensor<64x1024x10x10xf32>
    %v922 = stablehlo.reshape %v921 : (tensor<64x1024x10x10xf32>) -> tensor<64x102400xf32>
    %v923 = stablehlo.reshape %v922 : (tensor<64x102400xf32>) -> tensor<64x1024x10x10xf32>
    %v924 = stablehlo.constant dense<0.0> : tensor<64x1024x10x10xf32>
    %v925 = stablehlo.maximum %v923, %v924 : tensor<64x1024x10x10xf32>
    %v926 = stablehlo.reshape %v925 : (tensor<64x1024x10x10xf32>) -> tensor<64x102400xf32>
    %v927 = stablehlo.reshape %v926 : (tensor<64x102400xf32>) -> tensor<64x1024x10x10xf32>
    %v928 = stablehlo.convolution(%v927, %s3b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x10x10xf32>, tensor<256x1024x1x1xf32>) -> tensor<64x256x10x10xf32>
    %v929 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v930 = stablehlo.add %v928, %v929 : tensor<64x256x10x10xf32>
    %v931 = stablehlo.reshape %v930 : (tensor<64x256x10x10xf32>) -> tensor<64x25600xf32>
    %v932 = stablehlo.reshape %v931 : (tensor<64x25600xf32>) -> tensor<64x256x10x10xf32>
    %v933 = stablehlo.constant dense<0.0> : tensor<f32>
    %v934 = stablehlo.constant dense<6400.0> : tensor<64x256x10x10xf32>
    %v935 = stablehlo.constant dense<1.0e-05> : tensor<64x256x10x10xf32>
    %v936 = stablehlo.reduce(%v932 init: %v933) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x10x10xf32>, tensor<f32>) -> tensor<256xf32>
    %v937 = stablehlo.broadcast_in_dim %v936, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v938 = stablehlo.divide %v937, %v934 : tensor<64x256x10x10xf32>
    %v939 = stablehlo.subtract %v932, %v938 : tensor<64x256x10x10xf32>
    %v940 = stablehlo.multiply %v939, %v939 : tensor<64x256x10x10xf32>
    %v941 = stablehlo.reduce(%v940 init: %v933) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x10x10xf32>, tensor<f32>) -> tensor<256xf32>
    %v942 = stablehlo.broadcast_in_dim %v941, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v943 = stablehlo.divide %v942, %v934 : tensor<64x256x10x10xf32>
    %v944 = stablehlo.add %v943, %v935 : tensor<64x256x10x10xf32>
    %v945 = stablehlo.rsqrt %v944 : tensor<64x256x10x10xf32>
    %v946 = stablehlo.multiply %v939, %v945 : tensor<64x256x10x10xf32>
    %v947 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v948 = stablehlo.broadcast_in_dim %s3b2bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v949 = stablehlo.multiply %v946, %v947 : tensor<64x256x10x10xf32>
    %v950 = stablehlo.add %v949, %v948 : tensor<64x256x10x10xf32>
    %v951 = stablehlo.reshape %v950 : (tensor<64x256x10x10xf32>) -> tensor<64x25600xf32>
    %v952 = stablehlo.reshape %v951 : (tensor<64x25600xf32>) -> tensor<64x256x10x10xf32>
    %v953 = stablehlo.constant dense<0.0> : tensor<64x256x10x10xf32>
    %v954 = stablehlo.maximum %v952, %v953 : tensor<64x256x10x10xf32>
    %v955 = stablehlo.reshape %v954 : (tensor<64x256x10x10xf32>) -> tensor<64x25600xf32>
    %v956 = stablehlo.reshape %v955 : (tensor<64x25600xf32>) -> tensor<64x256x10x10xf32>
    %v957 = stablehlo.convolution(%v956, %s3b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x10x10xf32>, tensor<256x256x3x3xf32>) -> tensor<64x256x10x10xf32>
    %v958 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v959 = stablehlo.add %v957, %v958 : tensor<64x256x10x10xf32>
    %v960 = stablehlo.reshape %v959 : (tensor<64x256x10x10xf32>) -> tensor<64x25600xf32>
    %v961 = stablehlo.reshape %v960 : (tensor<64x25600xf32>) -> tensor<64x256x10x10xf32>
    %v962 = stablehlo.constant dense<0.0> : tensor<f32>
    %v963 = stablehlo.constant dense<6400.0> : tensor<64x256x10x10xf32>
    %v964 = stablehlo.constant dense<1.0e-05> : tensor<64x256x10x10xf32>
    %v965 = stablehlo.reduce(%v961 init: %v962) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x10x10xf32>, tensor<f32>) -> tensor<256xf32>
    %v966 = stablehlo.broadcast_in_dim %v965, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v967 = stablehlo.divide %v966, %v963 : tensor<64x256x10x10xf32>
    %v968 = stablehlo.subtract %v961, %v967 : tensor<64x256x10x10xf32>
    %v969 = stablehlo.multiply %v968, %v968 : tensor<64x256x10x10xf32>
    %v970 = stablehlo.reduce(%v969 init: %v962) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x10x10xf32>, tensor<f32>) -> tensor<256xf32>
    %v971 = stablehlo.broadcast_in_dim %v970, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v972 = stablehlo.divide %v971, %v963 : tensor<64x256x10x10xf32>
    %v973 = stablehlo.add %v972, %v964 : tensor<64x256x10x10xf32>
    %v974 = stablehlo.rsqrt %v973 : tensor<64x256x10x10xf32>
    %v975 = stablehlo.multiply %v968, %v974 : tensor<64x256x10x10xf32>
    %v976 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v977 = stablehlo.broadcast_in_dim %s3b2bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v978 = stablehlo.multiply %v975, %v976 : tensor<64x256x10x10xf32>
    %v979 = stablehlo.add %v978, %v977 : tensor<64x256x10x10xf32>
    %v980 = stablehlo.reshape %v979 : (tensor<64x256x10x10xf32>) -> tensor<64x25600xf32>
    %v981 = stablehlo.reshape %v980 : (tensor<64x25600xf32>) -> tensor<64x256x10x10xf32>
    %v982 = stablehlo.constant dense<0.0> : tensor<64x256x10x10xf32>
    %v983 = stablehlo.maximum %v981, %v982 : tensor<64x256x10x10xf32>
    %v984 = stablehlo.reshape %v983 : (tensor<64x256x10x10xf32>) -> tensor<64x25600xf32>
    %v985 = stablehlo.reshape %v984 : (tensor<64x25600xf32>) -> tensor<64x256x10x10xf32>
    %v986 = stablehlo.convolution(%v985, %s3b2W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x10x10xf32>, tensor<1024x256x1x1xf32>) -> tensor<64x1024x10x10xf32>
    %v987 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x10x10xf32>
    %v988 = stablehlo.add %v986, %v987 : tensor<64x1024x10x10xf32>
    %v989 = stablehlo.reshape %v988 : (tensor<64x1024x10x10xf32>) -> tensor<64x102400xf32>
    %v990 = stablehlo.reshape %v989 : (tensor<64x102400xf32>) -> tensor<64x1024x10x10xf32>
    %v991 = stablehlo.constant dense<0.0> : tensor<f32>
    %v992 = stablehlo.constant dense<6400.0> : tensor<64x1024x10x10xf32>
    %v993 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x10x10xf32>
    %v994 = stablehlo.reduce(%v990 init: %v991) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x10x10xf32>, tensor<f32>) -> tensor<1024xf32>
    %v995 = stablehlo.broadcast_in_dim %v994, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x10x10xf32>
    %v996 = stablehlo.divide %v995, %v992 : tensor<64x1024x10x10xf32>
    %v997 = stablehlo.subtract %v990, %v996 : tensor<64x1024x10x10xf32>
    %v998 = stablehlo.multiply %v997, %v997 : tensor<64x1024x10x10xf32>
    %v999 = stablehlo.reduce(%v998 init: %v991) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x10x10xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1000 = stablehlo.broadcast_in_dim %v999, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x10x10xf32>
    %v1001 = stablehlo.divide %v1000, %v992 : tensor<64x1024x10x10xf32>
    %v1002 = stablehlo.add %v1001, %v993 : tensor<64x1024x10x10xf32>
    %v1003 = stablehlo.rsqrt %v1002 : tensor<64x1024x10x10xf32>
    %v1004 = stablehlo.multiply %v997, %v1003 : tensor<64x1024x10x10xf32>
    %v1005 = stablehlo.broadcast_in_dim %s3b2g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x10x10xf32>
    %v1006 = stablehlo.broadcast_in_dim %s3b2bt3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x10x10xf32>
    %v1007 = stablehlo.multiply %v1004, %v1005 : tensor<64x1024x10x10xf32>
    %v1008 = stablehlo.add %v1007, %v1006 : tensor<64x1024x10x10xf32>
    %v1009 = stablehlo.reshape %v1008 : (tensor<64x1024x10x10xf32>) -> tensor<64x102400xf32>
    %v1010 = stablehlo.reshape %v1009 : (tensor<64x102400xf32>) -> tensor<64x1024x10x10xf32>
    %v1011 = stablehlo.reshape %v926 : (tensor<64x102400xf32>) -> tensor<64x1024x10x10xf32>
    %v1012 = stablehlo.add %v1010, %v1011 : tensor<64x1024x10x10xf32>
    %v1013 = stablehlo.reshape %v1012 : (tensor<64x1024x10x10xf32>) -> tensor<64x102400xf32>
    %v1014 = stablehlo.reshape %v1013 : (tensor<64x102400xf32>) -> tensor<64x1024x10x10xf32>
    %v1015 = stablehlo.constant dense<0.0> : tensor<64x1024x10x10xf32>
    %v1016 = stablehlo.maximum %v1014, %v1015 : tensor<64x1024x10x10xf32>
    %v1017 = stablehlo.reshape %v1016 : (tensor<64x1024x10x10xf32>) -> tensor<64x102400xf32>
    %v1018 = stablehlo.reshape %v1017 : (tensor<64x102400xf32>) -> tensor<64x1024x10x10xf32>
    %v1019 = stablehlo.convolution(%v1018, %s3b3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x10x10xf32>, tensor<256x1024x1x1xf32>) -> tensor<64x256x10x10xf32>
    %v1020 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v1021 = stablehlo.add %v1019, %v1020 : tensor<64x256x10x10xf32>
    %v1022 = stablehlo.reshape %v1021 : (tensor<64x256x10x10xf32>) -> tensor<64x25600xf32>
    %v1023 = stablehlo.reshape %v1022 : (tensor<64x25600xf32>) -> tensor<64x256x10x10xf32>
    %v1024 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1025 = stablehlo.constant dense<6400.0> : tensor<64x256x10x10xf32>
    %v1026 = stablehlo.constant dense<1.0e-05> : tensor<64x256x10x10xf32>
    %v1027 = stablehlo.reduce(%v1023 init: %v1024) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x10x10xf32>, tensor<f32>) -> tensor<256xf32>
    %v1028 = stablehlo.broadcast_in_dim %v1027, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v1029 = stablehlo.divide %v1028, %v1025 : tensor<64x256x10x10xf32>
    %v1030 = stablehlo.subtract %v1023, %v1029 : tensor<64x256x10x10xf32>
    %v1031 = stablehlo.multiply %v1030, %v1030 : tensor<64x256x10x10xf32>
    %v1032 = stablehlo.reduce(%v1031 init: %v1024) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x10x10xf32>, tensor<f32>) -> tensor<256xf32>
    %v1033 = stablehlo.broadcast_in_dim %v1032, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v1034 = stablehlo.divide %v1033, %v1025 : tensor<64x256x10x10xf32>
    %v1035 = stablehlo.add %v1034, %v1026 : tensor<64x256x10x10xf32>
    %v1036 = stablehlo.rsqrt %v1035 : tensor<64x256x10x10xf32>
    %v1037 = stablehlo.multiply %v1030, %v1036 : tensor<64x256x10x10xf32>
    %v1038 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v1039 = stablehlo.broadcast_in_dim %s3b3bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v1040 = stablehlo.multiply %v1037, %v1038 : tensor<64x256x10x10xf32>
    %v1041 = stablehlo.add %v1040, %v1039 : tensor<64x256x10x10xf32>
    %v1042 = stablehlo.reshape %v1041 : (tensor<64x256x10x10xf32>) -> tensor<64x25600xf32>
    %v1043 = stablehlo.reshape %v1042 : (tensor<64x25600xf32>) -> tensor<64x256x10x10xf32>
    %v1044 = stablehlo.constant dense<0.0> : tensor<64x256x10x10xf32>
    %v1045 = stablehlo.maximum %v1043, %v1044 : tensor<64x256x10x10xf32>
    %v1046 = stablehlo.reshape %v1045 : (tensor<64x256x10x10xf32>) -> tensor<64x25600xf32>
    %v1047 = stablehlo.reshape %v1046 : (tensor<64x25600xf32>) -> tensor<64x256x10x10xf32>
    %v1048 = stablehlo.convolution(%v1047, %s3b3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x10x10xf32>, tensor<256x256x3x3xf32>) -> tensor<64x256x10x10xf32>
    %v1049 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v1050 = stablehlo.add %v1048, %v1049 : tensor<64x256x10x10xf32>
    %v1051 = stablehlo.reshape %v1050 : (tensor<64x256x10x10xf32>) -> tensor<64x25600xf32>
    %v1052 = stablehlo.reshape %v1051 : (tensor<64x25600xf32>) -> tensor<64x256x10x10xf32>
    %v1053 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1054 = stablehlo.constant dense<6400.0> : tensor<64x256x10x10xf32>
    %v1055 = stablehlo.constant dense<1.0e-05> : tensor<64x256x10x10xf32>
    %v1056 = stablehlo.reduce(%v1052 init: %v1053) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x10x10xf32>, tensor<f32>) -> tensor<256xf32>
    %v1057 = stablehlo.broadcast_in_dim %v1056, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v1058 = stablehlo.divide %v1057, %v1054 : tensor<64x256x10x10xf32>
    %v1059 = stablehlo.subtract %v1052, %v1058 : tensor<64x256x10x10xf32>
    %v1060 = stablehlo.multiply %v1059, %v1059 : tensor<64x256x10x10xf32>
    %v1061 = stablehlo.reduce(%v1060 init: %v1053) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x10x10xf32>, tensor<f32>) -> tensor<256xf32>
    %v1062 = stablehlo.broadcast_in_dim %v1061, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v1063 = stablehlo.divide %v1062, %v1054 : tensor<64x256x10x10xf32>
    %v1064 = stablehlo.add %v1063, %v1055 : tensor<64x256x10x10xf32>
    %v1065 = stablehlo.rsqrt %v1064 : tensor<64x256x10x10xf32>
    %v1066 = stablehlo.multiply %v1059, %v1065 : tensor<64x256x10x10xf32>
    %v1067 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v1068 = stablehlo.broadcast_in_dim %s3b3bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v1069 = stablehlo.multiply %v1066, %v1067 : tensor<64x256x10x10xf32>
    %v1070 = stablehlo.add %v1069, %v1068 : tensor<64x256x10x10xf32>
    %v1071 = stablehlo.reshape %v1070 : (tensor<64x256x10x10xf32>) -> tensor<64x25600xf32>
    %v1072 = stablehlo.reshape %v1071 : (tensor<64x25600xf32>) -> tensor<64x256x10x10xf32>
    %v1073 = stablehlo.constant dense<0.0> : tensor<64x256x10x10xf32>
    %v1074 = stablehlo.maximum %v1072, %v1073 : tensor<64x256x10x10xf32>
    %v1075 = stablehlo.reshape %v1074 : (tensor<64x256x10x10xf32>) -> tensor<64x25600xf32>
    %v1076 = stablehlo.reshape %v1075 : (tensor<64x25600xf32>) -> tensor<64x256x10x10xf32>
    %v1077 = stablehlo.convolution(%v1076, %s3b3W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x10x10xf32>, tensor<1024x256x1x1xf32>) -> tensor<64x1024x10x10xf32>
    %v1078 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x10x10xf32>
    %v1079 = stablehlo.add %v1077, %v1078 : tensor<64x1024x10x10xf32>
    %v1080 = stablehlo.reshape %v1079 : (tensor<64x1024x10x10xf32>) -> tensor<64x102400xf32>
    %v1081 = stablehlo.reshape %v1080 : (tensor<64x102400xf32>) -> tensor<64x1024x10x10xf32>
    %v1082 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1083 = stablehlo.constant dense<6400.0> : tensor<64x1024x10x10xf32>
    %v1084 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x10x10xf32>
    %v1085 = stablehlo.reduce(%v1081 init: %v1082) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x10x10xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1086 = stablehlo.broadcast_in_dim %v1085, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x10x10xf32>
    %v1087 = stablehlo.divide %v1086, %v1083 : tensor<64x1024x10x10xf32>
    %v1088 = stablehlo.subtract %v1081, %v1087 : tensor<64x1024x10x10xf32>
    %v1089 = stablehlo.multiply %v1088, %v1088 : tensor<64x1024x10x10xf32>
    %v1090 = stablehlo.reduce(%v1089 init: %v1082) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x10x10xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1091 = stablehlo.broadcast_in_dim %v1090, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x10x10xf32>
    %v1092 = stablehlo.divide %v1091, %v1083 : tensor<64x1024x10x10xf32>
    %v1093 = stablehlo.add %v1092, %v1084 : tensor<64x1024x10x10xf32>
    %v1094 = stablehlo.rsqrt %v1093 : tensor<64x1024x10x10xf32>
    %v1095 = stablehlo.multiply %v1088, %v1094 : tensor<64x1024x10x10xf32>
    %v1096 = stablehlo.broadcast_in_dim %s3b3g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x10x10xf32>
    %v1097 = stablehlo.broadcast_in_dim %s3b3bt3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x10x10xf32>
    %v1098 = stablehlo.multiply %v1095, %v1096 : tensor<64x1024x10x10xf32>
    %v1099 = stablehlo.add %v1098, %v1097 : tensor<64x1024x10x10xf32>
    %v1100 = stablehlo.reshape %v1099 : (tensor<64x1024x10x10xf32>) -> tensor<64x102400xf32>
    %v1101 = stablehlo.reshape %v1100 : (tensor<64x102400xf32>) -> tensor<64x1024x10x10xf32>
    %v1102 = stablehlo.reshape %v1017 : (tensor<64x102400xf32>) -> tensor<64x1024x10x10xf32>
    %v1103 = stablehlo.add %v1101, %v1102 : tensor<64x1024x10x10xf32>
    %v1104 = stablehlo.reshape %v1103 : (tensor<64x1024x10x10xf32>) -> tensor<64x102400xf32>
    %v1105 = stablehlo.reshape %v1104 : (tensor<64x102400xf32>) -> tensor<64x1024x10x10xf32>
    %v1106 = stablehlo.constant dense<0.0> : tensor<64x1024x10x10xf32>
    %v1107 = stablehlo.maximum %v1105, %v1106 : tensor<64x1024x10x10xf32>
    %v1108 = stablehlo.reshape %v1107 : (tensor<64x1024x10x10xf32>) -> tensor<64x102400xf32>
    %v1109 = stablehlo.reshape %v1108 : (tensor<64x102400xf32>) -> tensor<64x1024x10x10xf32>
    %v1110 = stablehlo.convolution(%v1109, %s3b4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x10x10xf32>, tensor<256x1024x1x1xf32>) -> tensor<64x256x10x10xf32>
    %v1111 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v1112 = stablehlo.add %v1110, %v1111 : tensor<64x256x10x10xf32>
    %v1113 = stablehlo.reshape %v1112 : (tensor<64x256x10x10xf32>) -> tensor<64x25600xf32>
    %v1114 = stablehlo.reshape %v1113 : (tensor<64x25600xf32>) -> tensor<64x256x10x10xf32>
    %v1115 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1116 = stablehlo.constant dense<6400.0> : tensor<64x256x10x10xf32>
    %v1117 = stablehlo.constant dense<1.0e-05> : tensor<64x256x10x10xf32>
    %v1118 = stablehlo.reduce(%v1114 init: %v1115) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x10x10xf32>, tensor<f32>) -> tensor<256xf32>
    %v1119 = stablehlo.broadcast_in_dim %v1118, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v1120 = stablehlo.divide %v1119, %v1116 : tensor<64x256x10x10xf32>
    %v1121 = stablehlo.subtract %v1114, %v1120 : tensor<64x256x10x10xf32>
    %v1122 = stablehlo.multiply %v1121, %v1121 : tensor<64x256x10x10xf32>
    %v1123 = stablehlo.reduce(%v1122 init: %v1115) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x10x10xf32>, tensor<f32>) -> tensor<256xf32>
    %v1124 = stablehlo.broadcast_in_dim %v1123, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v1125 = stablehlo.divide %v1124, %v1116 : tensor<64x256x10x10xf32>
    %v1126 = stablehlo.add %v1125, %v1117 : tensor<64x256x10x10xf32>
    %v1127 = stablehlo.rsqrt %v1126 : tensor<64x256x10x10xf32>
    %v1128 = stablehlo.multiply %v1121, %v1127 : tensor<64x256x10x10xf32>
    %v1129 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v1130 = stablehlo.broadcast_in_dim %s3b4bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v1131 = stablehlo.multiply %v1128, %v1129 : tensor<64x256x10x10xf32>
    %v1132 = stablehlo.add %v1131, %v1130 : tensor<64x256x10x10xf32>
    %v1133 = stablehlo.reshape %v1132 : (tensor<64x256x10x10xf32>) -> tensor<64x25600xf32>
    %v1134 = stablehlo.reshape %v1133 : (tensor<64x25600xf32>) -> tensor<64x256x10x10xf32>
    %v1135 = stablehlo.constant dense<0.0> : tensor<64x256x10x10xf32>
    %v1136 = stablehlo.maximum %v1134, %v1135 : tensor<64x256x10x10xf32>
    %v1137 = stablehlo.reshape %v1136 : (tensor<64x256x10x10xf32>) -> tensor<64x25600xf32>
    %v1138 = stablehlo.reshape %v1137 : (tensor<64x25600xf32>) -> tensor<64x256x10x10xf32>
    %v1139 = stablehlo.convolution(%v1138, %s3b4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x10x10xf32>, tensor<256x256x3x3xf32>) -> tensor<64x256x10x10xf32>
    %v1140 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v1141 = stablehlo.add %v1139, %v1140 : tensor<64x256x10x10xf32>
    %v1142 = stablehlo.reshape %v1141 : (tensor<64x256x10x10xf32>) -> tensor<64x25600xf32>
    %v1143 = stablehlo.reshape %v1142 : (tensor<64x25600xf32>) -> tensor<64x256x10x10xf32>
    %v1144 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1145 = stablehlo.constant dense<6400.0> : tensor<64x256x10x10xf32>
    %v1146 = stablehlo.constant dense<1.0e-05> : tensor<64x256x10x10xf32>
    %v1147 = stablehlo.reduce(%v1143 init: %v1144) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x10x10xf32>, tensor<f32>) -> tensor<256xf32>
    %v1148 = stablehlo.broadcast_in_dim %v1147, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v1149 = stablehlo.divide %v1148, %v1145 : tensor<64x256x10x10xf32>
    %v1150 = stablehlo.subtract %v1143, %v1149 : tensor<64x256x10x10xf32>
    %v1151 = stablehlo.multiply %v1150, %v1150 : tensor<64x256x10x10xf32>
    %v1152 = stablehlo.reduce(%v1151 init: %v1144) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x10x10xf32>, tensor<f32>) -> tensor<256xf32>
    %v1153 = stablehlo.broadcast_in_dim %v1152, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v1154 = stablehlo.divide %v1153, %v1145 : tensor<64x256x10x10xf32>
    %v1155 = stablehlo.add %v1154, %v1146 : tensor<64x256x10x10xf32>
    %v1156 = stablehlo.rsqrt %v1155 : tensor<64x256x10x10xf32>
    %v1157 = stablehlo.multiply %v1150, %v1156 : tensor<64x256x10x10xf32>
    %v1158 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v1159 = stablehlo.broadcast_in_dim %s3b4bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v1160 = stablehlo.multiply %v1157, %v1158 : tensor<64x256x10x10xf32>
    %v1161 = stablehlo.add %v1160, %v1159 : tensor<64x256x10x10xf32>
    %v1162 = stablehlo.reshape %v1161 : (tensor<64x256x10x10xf32>) -> tensor<64x25600xf32>
    %v1163 = stablehlo.reshape %v1162 : (tensor<64x25600xf32>) -> tensor<64x256x10x10xf32>
    %v1164 = stablehlo.constant dense<0.0> : tensor<64x256x10x10xf32>
    %v1165 = stablehlo.maximum %v1163, %v1164 : tensor<64x256x10x10xf32>
    %v1166 = stablehlo.reshape %v1165 : (tensor<64x256x10x10xf32>) -> tensor<64x25600xf32>
    %v1167 = stablehlo.reshape %v1166 : (tensor<64x25600xf32>) -> tensor<64x256x10x10xf32>
    %v1168 = stablehlo.convolution(%v1167, %s3b4W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x10x10xf32>, tensor<1024x256x1x1xf32>) -> tensor<64x1024x10x10xf32>
    %v1169 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x10x10xf32>
    %v1170 = stablehlo.add %v1168, %v1169 : tensor<64x1024x10x10xf32>
    %v1171 = stablehlo.reshape %v1170 : (tensor<64x1024x10x10xf32>) -> tensor<64x102400xf32>
    %v1172 = stablehlo.reshape %v1171 : (tensor<64x102400xf32>) -> tensor<64x1024x10x10xf32>
    %v1173 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1174 = stablehlo.constant dense<6400.0> : tensor<64x1024x10x10xf32>
    %v1175 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x10x10xf32>
    %v1176 = stablehlo.reduce(%v1172 init: %v1173) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x10x10xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1177 = stablehlo.broadcast_in_dim %v1176, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x10x10xf32>
    %v1178 = stablehlo.divide %v1177, %v1174 : tensor<64x1024x10x10xf32>
    %v1179 = stablehlo.subtract %v1172, %v1178 : tensor<64x1024x10x10xf32>
    %v1180 = stablehlo.multiply %v1179, %v1179 : tensor<64x1024x10x10xf32>
    %v1181 = stablehlo.reduce(%v1180 init: %v1173) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x10x10xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1182 = stablehlo.broadcast_in_dim %v1181, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x10x10xf32>
    %v1183 = stablehlo.divide %v1182, %v1174 : tensor<64x1024x10x10xf32>
    %v1184 = stablehlo.add %v1183, %v1175 : tensor<64x1024x10x10xf32>
    %v1185 = stablehlo.rsqrt %v1184 : tensor<64x1024x10x10xf32>
    %v1186 = stablehlo.multiply %v1179, %v1185 : tensor<64x1024x10x10xf32>
    %v1187 = stablehlo.broadcast_in_dim %s3b4g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x10x10xf32>
    %v1188 = stablehlo.broadcast_in_dim %s3b4bt3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x10x10xf32>
    %v1189 = stablehlo.multiply %v1186, %v1187 : tensor<64x1024x10x10xf32>
    %v1190 = stablehlo.add %v1189, %v1188 : tensor<64x1024x10x10xf32>
    %v1191 = stablehlo.reshape %v1190 : (tensor<64x1024x10x10xf32>) -> tensor<64x102400xf32>
    %v1192 = stablehlo.reshape %v1191 : (tensor<64x102400xf32>) -> tensor<64x1024x10x10xf32>
    %v1193 = stablehlo.reshape %v1108 : (tensor<64x102400xf32>) -> tensor<64x1024x10x10xf32>
    %v1194 = stablehlo.add %v1192, %v1193 : tensor<64x1024x10x10xf32>
    %v1195 = stablehlo.reshape %v1194 : (tensor<64x1024x10x10xf32>) -> tensor<64x102400xf32>
    %v1196 = stablehlo.reshape %v1195 : (tensor<64x102400xf32>) -> tensor<64x1024x10x10xf32>
    %v1197 = stablehlo.constant dense<0.0> : tensor<64x1024x10x10xf32>
    %v1198 = stablehlo.maximum %v1196, %v1197 : tensor<64x1024x10x10xf32>
    %v1199 = stablehlo.reshape %v1198 : (tensor<64x1024x10x10xf32>) -> tensor<64x102400xf32>
    %v1200 = stablehlo.reshape %v1199 : (tensor<64x102400xf32>) -> tensor<64x1024x10x10xf32>
    %v1201 = stablehlo.convolution(%v1200, %s3b5W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x10x10xf32>, tensor<256x1024x1x1xf32>) -> tensor<64x256x10x10xf32>
    %v1202 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v1203 = stablehlo.add %v1201, %v1202 : tensor<64x256x10x10xf32>
    %v1204 = stablehlo.reshape %v1203 : (tensor<64x256x10x10xf32>) -> tensor<64x25600xf32>
    %v1205 = stablehlo.reshape %v1204 : (tensor<64x25600xf32>) -> tensor<64x256x10x10xf32>
    %v1206 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1207 = stablehlo.constant dense<6400.0> : tensor<64x256x10x10xf32>
    %v1208 = stablehlo.constant dense<1.0e-05> : tensor<64x256x10x10xf32>
    %v1209 = stablehlo.reduce(%v1205 init: %v1206) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x10x10xf32>, tensor<f32>) -> tensor<256xf32>
    %v1210 = stablehlo.broadcast_in_dim %v1209, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v1211 = stablehlo.divide %v1210, %v1207 : tensor<64x256x10x10xf32>
    %v1212 = stablehlo.subtract %v1205, %v1211 : tensor<64x256x10x10xf32>
    %v1213 = stablehlo.multiply %v1212, %v1212 : tensor<64x256x10x10xf32>
    %v1214 = stablehlo.reduce(%v1213 init: %v1206) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x10x10xf32>, tensor<f32>) -> tensor<256xf32>
    %v1215 = stablehlo.broadcast_in_dim %v1214, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v1216 = stablehlo.divide %v1215, %v1207 : tensor<64x256x10x10xf32>
    %v1217 = stablehlo.add %v1216, %v1208 : tensor<64x256x10x10xf32>
    %v1218 = stablehlo.rsqrt %v1217 : tensor<64x256x10x10xf32>
    %v1219 = stablehlo.multiply %v1212, %v1218 : tensor<64x256x10x10xf32>
    %v1220 = stablehlo.broadcast_in_dim %s3b5g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v1221 = stablehlo.broadcast_in_dim %s3b5bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v1222 = stablehlo.multiply %v1219, %v1220 : tensor<64x256x10x10xf32>
    %v1223 = stablehlo.add %v1222, %v1221 : tensor<64x256x10x10xf32>
    %v1224 = stablehlo.reshape %v1223 : (tensor<64x256x10x10xf32>) -> tensor<64x25600xf32>
    %v1225 = stablehlo.reshape %v1224 : (tensor<64x25600xf32>) -> tensor<64x256x10x10xf32>
    %v1226 = stablehlo.constant dense<0.0> : tensor<64x256x10x10xf32>
    %v1227 = stablehlo.maximum %v1225, %v1226 : tensor<64x256x10x10xf32>
    %v1228 = stablehlo.reshape %v1227 : (tensor<64x256x10x10xf32>) -> tensor<64x25600xf32>
    %v1229 = stablehlo.reshape %v1228 : (tensor<64x25600xf32>) -> tensor<64x256x10x10xf32>
    %v1230 = stablehlo.convolution(%v1229, %s3b5W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x10x10xf32>, tensor<256x256x3x3xf32>) -> tensor<64x256x10x10xf32>
    %v1231 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v1232 = stablehlo.add %v1230, %v1231 : tensor<64x256x10x10xf32>
    %v1233 = stablehlo.reshape %v1232 : (tensor<64x256x10x10xf32>) -> tensor<64x25600xf32>
    %v1234 = stablehlo.reshape %v1233 : (tensor<64x25600xf32>) -> tensor<64x256x10x10xf32>
    %v1235 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1236 = stablehlo.constant dense<6400.0> : tensor<64x256x10x10xf32>
    %v1237 = stablehlo.constant dense<1.0e-05> : tensor<64x256x10x10xf32>
    %v1238 = stablehlo.reduce(%v1234 init: %v1235) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x10x10xf32>, tensor<f32>) -> tensor<256xf32>
    %v1239 = stablehlo.broadcast_in_dim %v1238, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v1240 = stablehlo.divide %v1239, %v1236 : tensor<64x256x10x10xf32>
    %v1241 = stablehlo.subtract %v1234, %v1240 : tensor<64x256x10x10xf32>
    %v1242 = stablehlo.multiply %v1241, %v1241 : tensor<64x256x10x10xf32>
    %v1243 = stablehlo.reduce(%v1242 init: %v1235) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x10x10xf32>, tensor<f32>) -> tensor<256xf32>
    %v1244 = stablehlo.broadcast_in_dim %v1243, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v1245 = stablehlo.divide %v1244, %v1236 : tensor<64x256x10x10xf32>
    %v1246 = stablehlo.add %v1245, %v1237 : tensor<64x256x10x10xf32>
    %v1247 = stablehlo.rsqrt %v1246 : tensor<64x256x10x10xf32>
    %v1248 = stablehlo.multiply %v1241, %v1247 : tensor<64x256x10x10xf32>
    %v1249 = stablehlo.broadcast_in_dim %s3b5g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v1250 = stablehlo.broadcast_in_dim %s3b5bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x10x10xf32>
    %v1251 = stablehlo.multiply %v1248, %v1249 : tensor<64x256x10x10xf32>
    %v1252 = stablehlo.add %v1251, %v1250 : tensor<64x256x10x10xf32>
    %v1253 = stablehlo.reshape %v1252 : (tensor<64x256x10x10xf32>) -> tensor<64x25600xf32>
    %v1254 = stablehlo.reshape %v1253 : (tensor<64x25600xf32>) -> tensor<64x256x10x10xf32>
    %v1255 = stablehlo.constant dense<0.0> : tensor<64x256x10x10xf32>
    %v1256 = stablehlo.maximum %v1254, %v1255 : tensor<64x256x10x10xf32>
    %v1257 = stablehlo.reshape %v1256 : (tensor<64x256x10x10xf32>) -> tensor<64x25600xf32>
    %v1258 = stablehlo.reshape %v1257 : (tensor<64x25600xf32>) -> tensor<64x256x10x10xf32>
    %v1259 = stablehlo.convolution(%v1258, %s3b5W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x10x10xf32>, tensor<1024x256x1x1xf32>) -> tensor<64x1024x10x10xf32>
    %v1260 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x10x10xf32>
    %v1261 = stablehlo.add %v1259, %v1260 : tensor<64x1024x10x10xf32>
    %v1262 = stablehlo.reshape %v1261 : (tensor<64x1024x10x10xf32>) -> tensor<64x102400xf32>
    %v1263 = stablehlo.reshape %v1262 : (tensor<64x102400xf32>) -> tensor<64x1024x10x10xf32>
    %v1264 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1265 = stablehlo.constant dense<6400.0> : tensor<64x1024x10x10xf32>
    %v1266 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x10x10xf32>
    %v1267 = stablehlo.reduce(%v1263 init: %v1264) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x10x10xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1268 = stablehlo.broadcast_in_dim %v1267, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x10x10xf32>
    %v1269 = stablehlo.divide %v1268, %v1265 : tensor<64x1024x10x10xf32>
    %v1270 = stablehlo.subtract %v1263, %v1269 : tensor<64x1024x10x10xf32>
    %v1271 = stablehlo.multiply %v1270, %v1270 : tensor<64x1024x10x10xf32>
    %v1272 = stablehlo.reduce(%v1271 init: %v1264) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x10x10xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1273 = stablehlo.broadcast_in_dim %v1272, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x10x10xf32>
    %v1274 = stablehlo.divide %v1273, %v1265 : tensor<64x1024x10x10xf32>
    %v1275 = stablehlo.add %v1274, %v1266 : tensor<64x1024x10x10xf32>
    %v1276 = stablehlo.rsqrt %v1275 : tensor<64x1024x10x10xf32>
    %v1277 = stablehlo.multiply %v1270, %v1276 : tensor<64x1024x10x10xf32>
    %v1278 = stablehlo.broadcast_in_dim %s3b5g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x10x10xf32>
    %v1279 = stablehlo.broadcast_in_dim %s3b5bt3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x10x10xf32>
    %v1280 = stablehlo.multiply %v1277, %v1278 : tensor<64x1024x10x10xf32>
    %v1281 = stablehlo.add %v1280, %v1279 : tensor<64x1024x10x10xf32>
    %v1282 = stablehlo.reshape %v1281 : (tensor<64x1024x10x10xf32>) -> tensor<64x102400xf32>
    %v1283 = stablehlo.reshape %v1282 : (tensor<64x102400xf32>) -> tensor<64x1024x10x10xf32>
    %v1284 = stablehlo.reshape %v1199 : (tensor<64x102400xf32>) -> tensor<64x1024x10x10xf32>
    %v1285 = stablehlo.add %v1283, %v1284 : tensor<64x1024x10x10xf32>
    %v1286 = stablehlo.reshape %v1285 : (tensor<64x1024x10x10xf32>) -> tensor<64x102400xf32>
    %v1287 = stablehlo.reshape %v1286 : (tensor<64x102400xf32>) -> tensor<64x1024x10x10xf32>
    %v1288 = stablehlo.constant dense<0.0> : tensor<64x1024x10x10xf32>
    %v1289 = stablehlo.maximum %v1287, %v1288 : tensor<64x1024x10x10xf32>
    %v1290 = stablehlo.reshape %v1289 : (tensor<64x1024x10x10xf32>) -> tensor<64x102400xf32>
    %v1291 = stablehlo.reshape %v1290 : (tensor<64x102400xf32>) -> tensor<64x1024x10x10xf32>
    %v1292 = stablehlo.convolution(%v1291, %s4b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x10x10xf32>, tensor<512x1024x1x1xf32>) -> tensor<64x512x10x10xf32>
    %v1293 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x10x10xf32>
    %v1294 = stablehlo.add %v1292, %v1293 : tensor<64x512x10x10xf32>
    %v1295 = stablehlo.reshape %v1294 : (tensor<64x512x10x10xf32>) -> tensor<64x51200xf32>
    %v1296 = stablehlo.reshape %v1295 : (tensor<64x51200xf32>) -> tensor<64x512x10x10xf32>
    %v1297 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1298 = stablehlo.constant dense<6400.0> : tensor<64x512x10x10xf32>
    %v1299 = stablehlo.constant dense<1.0e-05> : tensor<64x512x10x10xf32>
    %v1300 = stablehlo.reduce(%v1296 init: %v1297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x10x10xf32>, tensor<f32>) -> tensor<512xf32>
    %v1301 = stablehlo.broadcast_in_dim %v1300, dims = [1] : (tensor<512xf32>) -> tensor<64x512x10x10xf32>
    %v1302 = stablehlo.divide %v1301, %v1298 : tensor<64x512x10x10xf32>
    %v1303 = stablehlo.subtract %v1296, %v1302 : tensor<64x512x10x10xf32>
    %v1304 = stablehlo.multiply %v1303, %v1303 : tensor<64x512x10x10xf32>
    %v1305 = stablehlo.reduce(%v1304 init: %v1297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x10x10xf32>, tensor<f32>) -> tensor<512xf32>
    %v1306 = stablehlo.broadcast_in_dim %v1305, dims = [1] : (tensor<512xf32>) -> tensor<64x512x10x10xf32>
    %v1307 = stablehlo.divide %v1306, %v1298 : tensor<64x512x10x10xf32>
    %v1308 = stablehlo.add %v1307, %v1299 : tensor<64x512x10x10xf32>
    %v1309 = stablehlo.rsqrt %v1308 : tensor<64x512x10x10xf32>
    %v1310 = stablehlo.multiply %v1303, %v1309 : tensor<64x512x10x10xf32>
    %v1311 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x10x10xf32>
    %v1312 = stablehlo.broadcast_in_dim %s4b0bt1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x10x10xf32>
    %v1313 = stablehlo.multiply %v1310, %v1311 : tensor<64x512x10x10xf32>
    %v1314 = stablehlo.add %v1313, %v1312 : tensor<64x512x10x10xf32>
    %v1315 = stablehlo.reshape %v1314 : (tensor<64x512x10x10xf32>) -> tensor<64x51200xf32>
    %v1316 = stablehlo.reshape %v1315 : (tensor<64x51200xf32>) -> tensor<64x512x10x10xf32>
    %v1317 = stablehlo.constant dense<0.0> : tensor<64x512x10x10xf32>
    %v1318 = stablehlo.maximum %v1316, %v1317 : tensor<64x512x10x10xf32>
    %v1319 = stablehlo.reshape %v1318 : (tensor<64x512x10x10xf32>) -> tensor<64x51200xf32>
    %v1320 = stablehlo.reshape %v1319 : (tensor<64x51200xf32>) -> tensor<64x512x10x10xf32>
    %v1321 = stablehlo.convolution(%v1320, %s4b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x10x10xf32>, tensor<512x512x3x3xf32>) -> tensor<64x512x5x5xf32>
    %v1322 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x5x5xf32>
    %v1323 = stablehlo.add %v1321, %v1322 : tensor<64x512x5x5xf32>
    %v1324 = stablehlo.reshape %v1323 : (tensor<64x512x5x5xf32>) -> tensor<64x12800xf32>
    %v1325 = stablehlo.reshape %v1324 : (tensor<64x12800xf32>) -> tensor<64x512x5x5xf32>
    %v1326 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1327 = stablehlo.constant dense<1600.0> : tensor<64x512x5x5xf32>
    %v1328 = stablehlo.constant dense<1.0e-05> : tensor<64x512x5x5xf32>
    %v1329 = stablehlo.reduce(%v1325 init: %v1326) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x5x5xf32>, tensor<f32>) -> tensor<512xf32>
    %v1330 = stablehlo.broadcast_in_dim %v1329, dims = [1] : (tensor<512xf32>) -> tensor<64x512x5x5xf32>
    %v1331 = stablehlo.divide %v1330, %v1327 : tensor<64x512x5x5xf32>
    %v1332 = stablehlo.subtract %v1325, %v1331 : tensor<64x512x5x5xf32>
    %v1333 = stablehlo.multiply %v1332, %v1332 : tensor<64x512x5x5xf32>
    %v1334 = stablehlo.reduce(%v1333 init: %v1326) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x5x5xf32>, tensor<f32>) -> tensor<512xf32>
    %v1335 = stablehlo.broadcast_in_dim %v1334, dims = [1] : (tensor<512xf32>) -> tensor<64x512x5x5xf32>
    %v1336 = stablehlo.divide %v1335, %v1327 : tensor<64x512x5x5xf32>
    %v1337 = stablehlo.add %v1336, %v1328 : tensor<64x512x5x5xf32>
    %v1338 = stablehlo.rsqrt %v1337 : tensor<64x512x5x5xf32>
    %v1339 = stablehlo.multiply %v1332, %v1338 : tensor<64x512x5x5xf32>
    %v1340 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x5x5xf32>
    %v1341 = stablehlo.broadcast_in_dim %s4b0bt2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x5x5xf32>
    %v1342 = stablehlo.multiply %v1339, %v1340 : tensor<64x512x5x5xf32>
    %v1343 = stablehlo.add %v1342, %v1341 : tensor<64x512x5x5xf32>
    %v1344 = stablehlo.reshape %v1343 : (tensor<64x512x5x5xf32>) -> tensor<64x12800xf32>
    %v1345 = stablehlo.reshape %v1344 : (tensor<64x12800xf32>) -> tensor<64x512x5x5xf32>
    %v1346 = stablehlo.constant dense<0.0> : tensor<64x512x5x5xf32>
    %v1347 = stablehlo.maximum %v1345, %v1346 : tensor<64x512x5x5xf32>
    %v1348 = stablehlo.reshape %v1347 : (tensor<64x512x5x5xf32>) -> tensor<64x12800xf32>
    %v1349 = stablehlo.reshape %v1348 : (tensor<64x12800xf32>) -> tensor<64x512x5x5xf32>
    %v1350 = stablehlo.convolution(%v1349, %s4b0W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x5x5xf32>, tensor<2048x512x1x1xf32>) -> tensor<64x2048x5x5xf32>
    %v1351 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x5x5xf32>
    %v1352 = stablehlo.add %v1350, %v1351 : tensor<64x2048x5x5xf32>
    %v1353 = stablehlo.reshape %v1352 : (tensor<64x2048x5x5xf32>) -> tensor<64x51200xf32>
    %v1354 = stablehlo.reshape %v1353 : (tensor<64x51200xf32>) -> tensor<64x2048x5x5xf32>
    %v1355 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1356 = stablehlo.constant dense<1600.0> : tensor<64x2048x5x5xf32>
    %v1357 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x5x5xf32>
    %v1358 = stablehlo.reduce(%v1354 init: %v1355) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x5x5xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1359 = stablehlo.broadcast_in_dim %v1358, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x5x5xf32>
    %v1360 = stablehlo.divide %v1359, %v1356 : tensor<64x2048x5x5xf32>
    %v1361 = stablehlo.subtract %v1354, %v1360 : tensor<64x2048x5x5xf32>
    %v1362 = stablehlo.multiply %v1361, %v1361 : tensor<64x2048x5x5xf32>
    %v1363 = stablehlo.reduce(%v1362 init: %v1355) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x5x5xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1364 = stablehlo.broadcast_in_dim %v1363, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x5x5xf32>
    %v1365 = stablehlo.divide %v1364, %v1356 : tensor<64x2048x5x5xf32>
    %v1366 = stablehlo.add %v1365, %v1357 : tensor<64x2048x5x5xf32>
    %v1367 = stablehlo.rsqrt %v1366 : tensor<64x2048x5x5xf32>
    %v1368 = stablehlo.multiply %v1361, %v1367 : tensor<64x2048x5x5xf32>
    %v1369 = stablehlo.broadcast_in_dim %s4b0g3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x5x5xf32>
    %v1370 = stablehlo.broadcast_in_dim %s4b0bt3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x5x5xf32>
    %v1371 = stablehlo.multiply %v1368, %v1369 : tensor<64x2048x5x5xf32>
    %v1372 = stablehlo.add %v1371, %v1370 : tensor<64x2048x5x5xf32>
    %v1373 = stablehlo.reshape %v1372 : (tensor<64x2048x5x5xf32>) -> tensor<64x51200xf32>
    %v1374 = stablehlo.reshape %v1290 : (tensor<64x102400xf32>) -> tensor<64x1024x10x10xf32>
    %v1375 = stablehlo.convolution(%v1374, %s4b0Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x10x10xf32>, tensor<2048x1024x1x1xf32>) -> tensor<64x2048x5x5xf32>
    %v1376 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x5x5xf32>
    %v1377 = stablehlo.add %v1375, %v1376 : tensor<64x2048x5x5xf32>
    %v1378 = stablehlo.reshape %v1377 : (tensor<64x2048x5x5xf32>) -> tensor<64x51200xf32>
    %v1379 = stablehlo.reshape %v1378 : (tensor<64x51200xf32>) -> tensor<64x2048x5x5xf32>
    %v1380 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1381 = stablehlo.constant dense<1600.0> : tensor<64x2048x5x5xf32>
    %v1382 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x5x5xf32>
    %v1383 = stablehlo.reduce(%v1379 init: %v1380) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x5x5xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1384 = stablehlo.broadcast_in_dim %v1383, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x5x5xf32>
    %v1385 = stablehlo.divide %v1384, %v1381 : tensor<64x2048x5x5xf32>
    %v1386 = stablehlo.subtract %v1379, %v1385 : tensor<64x2048x5x5xf32>
    %v1387 = stablehlo.multiply %v1386, %v1386 : tensor<64x2048x5x5xf32>
    %v1388 = stablehlo.reduce(%v1387 init: %v1380) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x5x5xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1389 = stablehlo.broadcast_in_dim %v1388, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x5x5xf32>
    %v1390 = stablehlo.divide %v1389, %v1381 : tensor<64x2048x5x5xf32>
    %v1391 = stablehlo.add %v1390, %v1382 : tensor<64x2048x5x5xf32>
    %v1392 = stablehlo.rsqrt %v1391 : tensor<64x2048x5x5xf32>
    %v1393 = stablehlo.multiply %v1386, %v1392 : tensor<64x2048x5x5xf32>
    %v1394 = stablehlo.broadcast_in_dim %s4b0gp, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x5x5xf32>
    %v1395 = stablehlo.broadcast_in_dim %s4b0btp, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x5x5xf32>
    %v1396 = stablehlo.multiply %v1393, %v1394 : tensor<64x2048x5x5xf32>
    %v1397 = stablehlo.add %v1396, %v1395 : tensor<64x2048x5x5xf32>
    %v1398 = stablehlo.reshape %v1397 : (tensor<64x2048x5x5xf32>) -> tensor<64x51200xf32>
    %v1399 = stablehlo.reshape %v1373 : (tensor<64x51200xf32>) -> tensor<64x2048x5x5xf32>
    %v1400 = stablehlo.reshape %v1398 : (tensor<64x51200xf32>) -> tensor<64x2048x5x5xf32>
    %v1401 = stablehlo.add %v1399, %v1400 : tensor<64x2048x5x5xf32>
    %v1402 = stablehlo.reshape %v1401 : (tensor<64x2048x5x5xf32>) -> tensor<64x51200xf32>
    %v1403 = stablehlo.reshape %v1402 : (tensor<64x51200xf32>) -> tensor<64x2048x5x5xf32>
    %v1404 = stablehlo.constant dense<0.0> : tensor<64x2048x5x5xf32>
    %v1405 = stablehlo.maximum %v1403, %v1404 : tensor<64x2048x5x5xf32>
    %v1406 = stablehlo.reshape %v1405 : (tensor<64x2048x5x5xf32>) -> tensor<64x51200xf32>
    %v1407 = stablehlo.reshape %v1406 : (tensor<64x51200xf32>) -> tensor<64x2048x5x5xf32>
    %v1408 = stablehlo.convolution(%v1407, %s4b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x2048x5x5xf32>, tensor<512x2048x1x1xf32>) -> tensor<64x512x5x5xf32>
    %v1409 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x5x5xf32>
    %v1410 = stablehlo.add %v1408, %v1409 : tensor<64x512x5x5xf32>
    %v1411 = stablehlo.reshape %v1410 : (tensor<64x512x5x5xf32>) -> tensor<64x12800xf32>
    %v1412 = stablehlo.reshape %v1411 : (tensor<64x12800xf32>) -> tensor<64x512x5x5xf32>
    %v1413 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1414 = stablehlo.constant dense<1600.0> : tensor<64x512x5x5xf32>
    %v1415 = stablehlo.constant dense<1.0e-05> : tensor<64x512x5x5xf32>
    %v1416 = stablehlo.reduce(%v1412 init: %v1413) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x5x5xf32>, tensor<f32>) -> tensor<512xf32>
    %v1417 = stablehlo.broadcast_in_dim %v1416, dims = [1] : (tensor<512xf32>) -> tensor<64x512x5x5xf32>
    %v1418 = stablehlo.divide %v1417, %v1414 : tensor<64x512x5x5xf32>
    %v1419 = stablehlo.subtract %v1412, %v1418 : tensor<64x512x5x5xf32>
    %v1420 = stablehlo.multiply %v1419, %v1419 : tensor<64x512x5x5xf32>
    %v1421 = stablehlo.reduce(%v1420 init: %v1413) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x5x5xf32>, tensor<f32>) -> tensor<512xf32>
    %v1422 = stablehlo.broadcast_in_dim %v1421, dims = [1] : (tensor<512xf32>) -> tensor<64x512x5x5xf32>
    %v1423 = stablehlo.divide %v1422, %v1414 : tensor<64x512x5x5xf32>
    %v1424 = stablehlo.add %v1423, %v1415 : tensor<64x512x5x5xf32>
    %v1425 = stablehlo.rsqrt %v1424 : tensor<64x512x5x5xf32>
    %v1426 = stablehlo.multiply %v1419, %v1425 : tensor<64x512x5x5xf32>
    %v1427 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x5x5xf32>
    %v1428 = stablehlo.broadcast_in_dim %s4b1bt1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x5x5xf32>
    %v1429 = stablehlo.multiply %v1426, %v1427 : tensor<64x512x5x5xf32>
    %v1430 = stablehlo.add %v1429, %v1428 : tensor<64x512x5x5xf32>
    %v1431 = stablehlo.reshape %v1430 : (tensor<64x512x5x5xf32>) -> tensor<64x12800xf32>
    %v1432 = stablehlo.reshape %v1431 : (tensor<64x12800xf32>) -> tensor<64x512x5x5xf32>
    %v1433 = stablehlo.constant dense<0.0> : tensor<64x512x5x5xf32>
    %v1434 = stablehlo.maximum %v1432, %v1433 : tensor<64x512x5x5xf32>
    %v1435 = stablehlo.reshape %v1434 : (tensor<64x512x5x5xf32>) -> tensor<64x12800xf32>
    %v1436 = stablehlo.reshape %v1435 : (tensor<64x12800xf32>) -> tensor<64x512x5x5xf32>
    %v1437 = stablehlo.convolution(%v1436, %s4b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x5x5xf32>, tensor<512x512x3x3xf32>) -> tensor<64x512x5x5xf32>
    %v1438 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x5x5xf32>
    %v1439 = stablehlo.add %v1437, %v1438 : tensor<64x512x5x5xf32>
    %v1440 = stablehlo.reshape %v1439 : (tensor<64x512x5x5xf32>) -> tensor<64x12800xf32>
    %v1441 = stablehlo.reshape %v1440 : (tensor<64x12800xf32>) -> tensor<64x512x5x5xf32>
    %v1442 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1443 = stablehlo.constant dense<1600.0> : tensor<64x512x5x5xf32>
    %v1444 = stablehlo.constant dense<1.0e-05> : tensor<64x512x5x5xf32>
    %v1445 = stablehlo.reduce(%v1441 init: %v1442) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x5x5xf32>, tensor<f32>) -> tensor<512xf32>
    %v1446 = stablehlo.broadcast_in_dim %v1445, dims = [1] : (tensor<512xf32>) -> tensor<64x512x5x5xf32>
    %v1447 = stablehlo.divide %v1446, %v1443 : tensor<64x512x5x5xf32>
    %v1448 = stablehlo.subtract %v1441, %v1447 : tensor<64x512x5x5xf32>
    %v1449 = stablehlo.multiply %v1448, %v1448 : tensor<64x512x5x5xf32>
    %v1450 = stablehlo.reduce(%v1449 init: %v1442) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x5x5xf32>, tensor<f32>) -> tensor<512xf32>
    %v1451 = stablehlo.broadcast_in_dim %v1450, dims = [1] : (tensor<512xf32>) -> tensor<64x512x5x5xf32>
    %v1452 = stablehlo.divide %v1451, %v1443 : tensor<64x512x5x5xf32>
    %v1453 = stablehlo.add %v1452, %v1444 : tensor<64x512x5x5xf32>
    %v1454 = stablehlo.rsqrt %v1453 : tensor<64x512x5x5xf32>
    %v1455 = stablehlo.multiply %v1448, %v1454 : tensor<64x512x5x5xf32>
    %v1456 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x5x5xf32>
    %v1457 = stablehlo.broadcast_in_dim %s4b1bt2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x5x5xf32>
    %v1458 = stablehlo.multiply %v1455, %v1456 : tensor<64x512x5x5xf32>
    %v1459 = stablehlo.add %v1458, %v1457 : tensor<64x512x5x5xf32>
    %v1460 = stablehlo.reshape %v1459 : (tensor<64x512x5x5xf32>) -> tensor<64x12800xf32>
    %v1461 = stablehlo.reshape %v1460 : (tensor<64x12800xf32>) -> tensor<64x512x5x5xf32>
    %v1462 = stablehlo.constant dense<0.0> : tensor<64x512x5x5xf32>
    %v1463 = stablehlo.maximum %v1461, %v1462 : tensor<64x512x5x5xf32>
    %v1464 = stablehlo.reshape %v1463 : (tensor<64x512x5x5xf32>) -> tensor<64x12800xf32>
    %v1465 = stablehlo.reshape %v1464 : (tensor<64x12800xf32>) -> tensor<64x512x5x5xf32>
    %v1466 = stablehlo.convolution(%v1465, %s4b1W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x5x5xf32>, tensor<2048x512x1x1xf32>) -> tensor<64x2048x5x5xf32>
    %v1467 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x5x5xf32>
    %v1468 = stablehlo.add %v1466, %v1467 : tensor<64x2048x5x5xf32>
    %v1469 = stablehlo.reshape %v1468 : (tensor<64x2048x5x5xf32>) -> tensor<64x51200xf32>
    %v1470 = stablehlo.reshape %v1469 : (tensor<64x51200xf32>) -> tensor<64x2048x5x5xf32>
    %v1471 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1472 = stablehlo.constant dense<1600.0> : tensor<64x2048x5x5xf32>
    %v1473 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x5x5xf32>
    %v1474 = stablehlo.reduce(%v1470 init: %v1471) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x5x5xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1475 = stablehlo.broadcast_in_dim %v1474, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x5x5xf32>
    %v1476 = stablehlo.divide %v1475, %v1472 : tensor<64x2048x5x5xf32>
    %v1477 = stablehlo.subtract %v1470, %v1476 : tensor<64x2048x5x5xf32>
    %v1478 = stablehlo.multiply %v1477, %v1477 : tensor<64x2048x5x5xf32>
    %v1479 = stablehlo.reduce(%v1478 init: %v1471) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x5x5xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1480 = stablehlo.broadcast_in_dim %v1479, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x5x5xf32>
    %v1481 = stablehlo.divide %v1480, %v1472 : tensor<64x2048x5x5xf32>
    %v1482 = stablehlo.add %v1481, %v1473 : tensor<64x2048x5x5xf32>
    %v1483 = stablehlo.rsqrt %v1482 : tensor<64x2048x5x5xf32>
    %v1484 = stablehlo.multiply %v1477, %v1483 : tensor<64x2048x5x5xf32>
    %v1485 = stablehlo.broadcast_in_dim %s4b1g3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x5x5xf32>
    %v1486 = stablehlo.broadcast_in_dim %s4b1bt3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x5x5xf32>
    %v1487 = stablehlo.multiply %v1484, %v1485 : tensor<64x2048x5x5xf32>
    %v1488 = stablehlo.add %v1487, %v1486 : tensor<64x2048x5x5xf32>
    %v1489 = stablehlo.reshape %v1488 : (tensor<64x2048x5x5xf32>) -> tensor<64x51200xf32>
    %v1490 = stablehlo.reshape %v1489 : (tensor<64x51200xf32>) -> tensor<64x2048x5x5xf32>
    %v1491 = stablehlo.reshape %v1406 : (tensor<64x51200xf32>) -> tensor<64x2048x5x5xf32>
    %v1492 = stablehlo.add %v1490, %v1491 : tensor<64x2048x5x5xf32>
    %v1493 = stablehlo.reshape %v1492 : (tensor<64x2048x5x5xf32>) -> tensor<64x51200xf32>
    %v1494 = stablehlo.reshape %v1493 : (tensor<64x51200xf32>) -> tensor<64x2048x5x5xf32>
    %v1495 = stablehlo.constant dense<0.0> : tensor<64x2048x5x5xf32>
    %v1496 = stablehlo.maximum %v1494, %v1495 : tensor<64x2048x5x5xf32>
    %v1497 = stablehlo.reshape %v1496 : (tensor<64x2048x5x5xf32>) -> tensor<64x51200xf32>
    %v1498 = stablehlo.reshape %v1497 : (tensor<64x51200xf32>) -> tensor<64x2048x5x5xf32>
    %v1499 = stablehlo.convolution(%v1498, %s4b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x2048x5x5xf32>, tensor<512x2048x1x1xf32>) -> tensor<64x512x5x5xf32>
    %v1500 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x5x5xf32>
    %v1501 = stablehlo.add %v1499, %v1500 : tensor<64x512x5x5xf32>
    %v1502 = stablehlo.reshape %v1501 : (tensor<64x512x5x5xf32>) -> tensor<64x12800xf32>
    %v1503 = stablehlo.reshape %v1502 : (tensor<64x12800xf32>) -> tensor<64x512x5x5xf32>
    %v1504 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1505 = stablehlo.constant dense<1600.0> : tensor<64x512x5x5xf32>
    %v1506 = stablehlo.constant dense<1.0e-05> : tensor<64x512x5x5xf32>
    %v1507 = stablehlo.reduce(%v1503 init: %v1504) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x5x5xf32>, tensor<f32>) -> tensor<512xf32>
    %v1508 = stablehlo.broadcast_in_dim %v1507, dims = [1] : (tensor<512xf32>) -> tensor<64x512x5x5xf32>
    %v1509 = stablehlo.divide %v1508, %v1505 : tensor<64x512x5x5xf32>
    %v1510 = stablehlo.subtract %v1503, %v1509 : tensor<64x512x5x5xf32>
    %v1511 = stablehlo.multiply %v1510, %v1510 : tensor<64x512x5x5xf32>
    %v1512 = stablehlo.reduce(%v1511 init: %v1504) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x5x5xf32>, tensor<f32>) -> tensor<512xf32>
    %v1513 = stablehlo.broadcast_in_dim %v1512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x5x5xf32>
    %v1514 = stablehlo.divide %v1513, %v1505 : tensor<64x512x5x5xf32>
    %v1515 = stablehlo.add %v1514, %v1506 : tensor<64x512x5x5xf32>
    %v1516 = stablehlo.rsqrt %v1515 : tensor<64x512x5x5xf32>
    %v1517 = stablehlo.multiply %v1510, %v1516 : tensor<64x512x5x5xf32>
    %v1518 = stablehlo.broadcast_in_dim %s4b2g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x5x5xf32>
    %v1519 = stablehlo.broadcast_in_dim %s4b2bt1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x5x5xf32>
    %v1520 = stablehlo.multiply %v1517, %v1518 : tensor<64x512x5x5xf32>
    %v1521 = stablehlo.add %v1520, %v1519 : tensor<64x512x5x5xf32>
    %v1522 = stablehlo.reshape %v1521 : (tensor<64x512x5x5xf32>) -> tensor<64x12800xf32>
    %v1523 = stablehlo.reshape %v1522 : (tensor<64x12800xf32>) -> tensor<64x512x5x5xf32>
    %v1524 = stablehlo.constant dense<0.0> : tensor<64x512x5x5xf32>
    %v1525 = stablehlo.maximum %v1523, %v1524 : tensor<64x512x5x5xf32>
    %v1526 = stablehlo.reshape %v1525 : (tensor<64x512x5x5xf32>) -> tensor<64x12800xf32>
    %v1527 = stablehlo.reshape %v1526 : (tensor<64x12800xf32>) -> tensor<64x512x5x5xf32>
    %v1528 = stablehlo.convolution(%v1527, %s4b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x5x5xf32>, tensor<512x512x3x3xf32>) -> tensor<64x512x5x5xf32>
    %v1529 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x5x5xf32>
    %v1530 = stablehlo.add %v1528, %v1529 : tensor<64x512x5x5xf32>
    %v1531 = stablehlo.reshape %v1530 : (tensor<64x512x5x5xf32>) -> tensor<64x12800xf32>
    %v1532 = stablehlo.reshape %v1531 : (tensor<64x12800xf32>) -> tensor<64x512x5x5xf32>
    %v1533 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1534 = stablehlo.constant dense<1600.0> : tensor<64x512x5x5xf32>
    %v1535 = stablehlo.constant dense<1.0e-05> : tensor<64x512x5x5xf32>
    %v1536 = stablehlo.reduce(%v1532 init: %v1533) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x5x5xf32>, tensor<f32>) -> tensor<512xf32>
    %v1537 = stablehlo.broadcast_in_dim %v1536, dims = [1] : (tensor<512xf32>) -> tensor<64x512x5x5xf32>
    %v1538 = stablehlo.divide %v1537, %v1534 : tensor<64x512x5x5xf32>
    %v1539 = stablehlo.subtract %v1532, %v1538 : tensor<64x512x5x5xf32>
    %v1540 = stablehlo.multiply %v1539, %v1539 : tensor<64x512x5x5xf32>
    %v1541 = stablehlo.reduce(%v1540 init: %v1533) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x5x5xf32>, tensor<f32>) -> tensor<512xf32>
    %v1542 = stablehlo.broadcast_in_dim %v1541, dims = [1] : (tensor<512xf32>) -> tensor<64x512x5x5xf32>
    %v1543 = stablehlo.divide %v1542, %v1534 : tensor<64x512x5x5xf32>
    %v1544 = stablehlo.add %v1543, %v1535 : tensor<64x512x5x5xf32>
    %v1545 = stablehlo.rsqrt %v1544 : tensor<64x512x5x5xf32>
    %v1546 = stablehlo.multiply %v1539, %v1545 : tensor<64x512x5x5xf32>
    %v1547 = stablehlo.broadcast_in_dim %s4b2g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x5x5xf32>
    %v1548 = stablehlo.broadcast_in_dim %s4b2bt2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x5x5xf32>
    %v1549 = stablehlo.multiply %v1546, %v1547 : tensor<64x512x5x5xf32>
    %v1550 = stablehlo.add %v1549, %v1548 : tensor<64x512x5x5xf32>
    %v1551 = stablehlo.reshape %v1550 : (tensor<64x512x5x5xf32>) -> tensor<64x12800xf32>
    %v1552 = stablehlo.reshape %v1551 : (tensor<64x12800xf32>) -> tensor<64x512x5x5xf32>
    %v1553 = stablehlo.constant dense<0.0> : tensor<64x512x5x5xf32>
    %v1554 = stablehlo.maximum %v1552, %v1553 : tensor<64x512x5x5xf32>
    %v1555 = stablehlo.reshape %v1554 : (tensor<64x512x5x5xf32>) -> tensor<64x12800xf32>
    %v1556 = stablehlo.reshape %v1555 : (tensor<64x12800xf32>) -> tensor<64x512x5x5xf32>
    %v1557 = stablehlo.convolution(%v1556, %s4b2W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x5x5xf32>, tensor<2048x512x1x1xf32>) -> tensor<64x2048x5x5xf32>
    %v1558 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x5x5xf32>
    %v1559 = stablehlo.add %v1557, %v1558 : tensor<64x2048x5x5xf32>
    %v1560 = stablehlo.reshape %v1559 : (tensor<64x2048x5x5xf32>) -> tensor<64x51200xf32>
    %v1561 = stablehlo.reshape %v1560 : (tensor<64x51200xf32>) -> tensor<64x2048x5x5xf32>
    %v1562 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1563 = stablehlo.constant dense<1600.0> : tensor<64x2048x5x5xf32>
    %v1564 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x5x5xf32>
    %v1565 = stablehlo.reduce(%v1561 init: %v1562) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x5x5xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1566 = stablehlo.broadcast_in_dim %v1565, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x5x5xf32>
    %v1567 = stablehlo.divide %v1566, %v1563 : tensor<64x2048x5x5xf32>
    %v1568 = stablehlo.subtract %v1561, %v1567 : tensor<64x2048x5x5xf32>
    %v1569 = stablehlo.multiply %v1568, %v1568 : tensor<64x2048x5x5xf32>
    %v1570 = stablehlo.reduce(%v1569 init: %v1562) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x5x5xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1571 = stablehlo.broadcast_in_dim %v1570, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x5x5xf32>
    %v1572 = stablehlo.divide %v1571, %v1563 : tensor<64x2048x5x5xf32>
    %v1573 = stablehlo.add %v1572, %v1564 : tensor<64x2048x5x5xf32>
    %v1574 = stablehlo.rsqrt %v1573 : tensor<64x2048x5x5xf32>
    %v1575 = stablehlo.multiply %v1568, %v1574 : tensor<64x2048x5x5xf32>
    %v1576 = stablehlo.broadcast_in_dim %s4b2g3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x5x5xf32>
    %v1577 = stablehlo.broadcast_in_dim %s4b2bt3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x5x5xf32>
    %v1578 = stablehlo.multiply %v1575, %v1576 : tensor<64x2048x5x5xf32>
    %v1579 = stablehlo.add %v1578, %v1577 : tensor<64x2048x5x5xf32>
    %v1580 = stablehlo.reshape %v1579 : (tensor<64x2048x5x5xf32>) -> tensor<64x51200xf32>
    %v1581 = stablehlo.reshape %v1580 : (tensor<64x51200xf32>) -> tensor<64x2048x5x5xf32>
    %v1582 = stablehlo.reshape %v1497 : (tensor<64x51200xf32>) -> tensor<64x2048x5x5xf32>
    %v1583 = stablehlo.add %v1581, %v1582 : tensor<64x2048x5x5xf32>
    %v1584 = stablehlo.reshape %v1583 : (tensor<64x2048x5x5xf32>) -> tensor<64x51200xf32>
    %v1585 = stablehlo.reshape %v1584 : (tensor<64x51200xf32>) -> tensor<64x2048x5x5xf32>
    %v1586 = stablehlo.constant dense<0.0> : tensor<64x2048x5x5xf32>
    %v1587 = stablehlo.maximum %v1585, %v1586 : tensor<64x2048x5x5xf32>
    %v1588 = stablehlo.reshape %v1587 : (tensor<64x2048x5x5xf32>) -> tensor<64x51200xf32>
    %v1589 = stablehlo.reshape %v1588 : (tensor<64x51200xf32>) -> tensor<64x2048x5x5xf32>
    %v1590 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1591 = stablehlo.reduce(%v1589 init: %v1590) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x2048x5x5xf32>, tensor<f32>) -> tensor<64x2048xf32>
    %v1592 = stablehlo.constant dense<25.0> : tensor<64x2048xf32>
    %v1593 = stablehlo.divide %v1591, %v1592 : tensor<64x2048xf32>
    %v1594 = stablehlo.dot_general %v1593, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x2048xf32>, tensor<2048x1000xf32>) -> tensor<64x1000xf32>
    %v1595 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<1000xf32>) -> tensor<64x1000xf32>
    %v1596 = stablehlo.add %v1594, %v1595 : tensor<64x1000xf32>
    return %v1596 : tensor<64x1000xf32>
  }
}
