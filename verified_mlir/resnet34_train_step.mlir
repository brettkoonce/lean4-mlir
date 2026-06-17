module @m {
  func.func @resnet34_train_step(%x: tensor<32x150528xf32>, %sW: tensor<64x3x7x7xf32>, %sbi: tensor<64xf32>, %sg: tensor<64xf32>, %sbt: tensor<64xf32>, %s1b0W1: tensor<64x64x3x3xf32>, %s1b0b1: tensor<64xf32>, %s1b0g1: tensor<64xf32>, %s1b0bt1: tensor<64xf32>, %s1b0W2: tensor<64x64x3x3xf32>, %s1b0b2: tensor<64xf32>, %s1b0g2: tensor<64xf32>, %s1b0bt2: tensor<64xf32>, %s1b1W1: tensor<64x64x3x3xf32>, %s1b1b1: tensor<64xf32>, %s1b1g1: tensor<64xf32>, %s1b1bt1: tensor<64xf32>, %s1b1W2: tensor<64x64x3x3xf32>, %s1b1b2: tensor<64xf32>, %s1b1g2: tensor<64xf32>, %s1b1bt2: tensor<64xf32>, %s1b2W1: tensor<64x64x3x3xf32>, %s1b2b1: tensor<64xf32>, %s1b2g1: tensor<64xf32>, %s1b2bt1: tensor<64xf32>, %s1b2W2: tensor<64x64x3x3xf32>, %s1b2b2: tensor<64xf32>, %s1b2g2: tensor<64xf32>, %s1b2bt2: tensor<64xf32>, %d2W1: tensor<128x64x3x3xf32>, %d2b1: tensor<128xf32>, %d2g1: tensor<128xf32>, %d2bt1: tensor<128xf32>, %d2W2: tensor<128x128x3x3xf32>, %d2b2: tensor<128xf32>, %d2g2: tensor<128xf32>, %d2bt2: tensor<128xf32>, %d2Wp: tensor<128x64x3x3xf32>, %d2bp: tensor<128xf32>, %d2gp: tensor<128xf32>, %d2btp: tensor<128xf32>, %s2b0W1: tensor<128x128x3x3xf32>, %s2b0b1: tensor<128xf32>, %s2b0g1: tensor<128xf32>, %s2b0bt1: tensor<128xf32>, %s2b0W2: tensor<128x128x3x3xf32>, %s2b0b2: tensor<128xf32>, %s2b0g2: tensor<128xf32>, %s2b0bt2: tensor<128xf32>, %s2b1W1: tensor<128x128x3x3xf32>, %s2b1b1: tensor<128xf32>, %s2b1g1: tensor<128xf32>, %s2b1bt1: tensor<128xf32>, %s2b1W2: tensor<128x128x3x3xf32>, %s2b1b2: tensor<128xf32>, %s2b1g2: tensor<128xf32>, %s2b1bt2: tensor<128xf32>, %s2b2W1: tensor<128x128x3x3xf32>, %s2b2b1: tensor<128xf32>, %s2b2g1: tensor<128xf32>, %s2b2bt1: tensor<128xf32>, %s2b2W2: tensor<128x128x3x3xf32>, %s2b2b2: tensor<128xf32>, %s2b2g2: tensor<128xf32>, %s2b2bt2: tensor<128xf32>, %d3W1: tensor<256x128x3x3xf32>, %d3b1: tensor<256xf32>, %d3g1: tensor<256xf32>, %d3bt1: tensor<256xf32>, %d3W2: tensor<256x256x3x3xf32>, %d3b2: tensor<256xf32>, %d3g2: tensor<256xf32>, %d3bt2: tensor<256xf32>, %d3Wp: tensor<256x128x3x3xf32>, %d3bp: tensor<256xf32>, %d3gp: tensor<256xf32>, %d3btp: tensor<256xf32>, %s3b0W1: tensor<256x256x3x3xf32>, %s3b0b1: tensor<256xf32>, %s3b0g1: tensor<256xf32>, %s3b0bt1: tensor<256xf32>, %s3b0W2: tensor<256x256x3x3xf32>, %s3b0b2: tensor<256xf32>, %s3b0g2: tensor<256xf32>, %s3b0bt2: tensor<256xf32>, %s3b1W1: tensor<256x256x3x3xf32>, %s3b1b1: tensor<256xf32>, %s3b1g1: tensor<256xf32>, %s3b1bt1: tensor<256xf32>, %s3b1W2: tensor<256x256x3x3xf32>, %s3b1b2: tensor<256xf32>, %s3b1g2: tensor<256xf32>, %s3b1bt2: tensor<256xf32>, %s3b2W1: tensor<256x256x3x3xf32>, %s3b2b1: tensor<256xf32>, %s3b2g1: tensor<256xf32>, %s3b2bt1: tensor<256xf32>, %s3b2W2: tensor<256x256x3x3xf32>, %s3b2b2: tensor<256xf32>, %s3b2g2: tensor<256xf32>, %s3b2bt2: tensor<256xf32>, %s3b3W1: tensor<256x256x3x3xf32>, %s3b3b1: tensor<256xf32>, %s3b3g1: tensor<256xf32>, %s3b3bt1: tensor<256xf32>, %s3b3W2: tensor<256x256x3x3xf32>, %s3b3b2: tensor<256xf32>, %s3b3g2: tensor<256xf32>, %s3b3bt2: tensor<256xf32>, %s3b4W1: tensor<256x256x3x3xf32>, %s3b4b1: tensor<256xf32>, %s3b4g1: tensor<256xf32>, %s3b4bt1: tensor<256xf32>, %s3b4W2: tensor<256x256x3x3xf32>, %s3b4b2: tensor<256xf32>, %s3b4g2: tensor<256xf32>, %s3b4bt2: tensor<256xf32>, %d4W1: tensor<512x256x3x3xf32>, %d4b1: tensor<512xf32>, %d4g1: tensor<512xf32>, %d4bt1: tensor<512xf32>, %d4W2: tensor<512x512x3x3xf32>, %d4b2: tensor<512xf32>, %d4g2: tensor<512xf32>, %d4bt2: tensor<512xf32>, %d4Wp: tensor<512x256x3x3xf32>, %d4bp: tensor<512xf32>, %d4gp: tensor<512xf32>, %d4btp: tensor<512xf32>, %s4b0W1: tensor<512x512x3x3xf32>, %s4b0b1: tensor<512xf32>, %s4b0g1: tensor<512xf32>, %s4b0bt1: tensor<512xf32>, %s4b0W2: tensor<512x512x3x3xf32>, %s4b0b2: tensor<512xf32>, %s4b0g2: tensor<512xf32>, %s4b0bt2: tensor<512xf32>, %s4b1W1: tensor<512x512x3x3xf32>, %s4b1b1: tensor<512xf32>, %s4b1g1: tensor<512xf32>, %s4b1bt1: tensor<512xf32>, %s4b1W2: tensor<512x512x3x3xf32>, %s4b1b2: tensor<512xf32>, %s4b1g2: tensor<512xf32>, %s4b1bt2: tensor<512xf32>, %Wd: tensor<512x10xf32>, %bd: tensor<10xf32>, %onehot: tensor<32x10xf32>) -> (tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>) {
    // ── ResNet-34 train step: every line is pretty(verified AST node) ──
    %v0 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v1 = stablehlo.convolution(%v0, %sW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<32x64x112x112xf32>
    %v2 = stablehlo.broadcast_in_dim %sbi, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
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
    %v33 = stablehlo.broadcast_in_dim %s1b0b1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
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
    %v60 = stablehlo.broadcast_in_dim %s1b0b2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
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
    %v88 = stablehlo.broadcast_in_dim %s1b1b1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
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
    %v115 = stablehlo.broadcast_in_dim %s1b1b2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
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
    %v143 = stablehlo.broadcast_in_dim %s1b2b1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
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
    %v170 = stablehlo.broadcast_in_dim %s1b2b2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
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
    %v198 = stablehlo.broadcast_in_dim %d2b1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
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
    %v225 = stablehlo.broadcast_in_dim %d2b2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
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
    %v250 = stablehlo.broadcast_in_dim %d2bp, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
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
    %v278 = stablehlo.broadcast_in_dim %s2b0b1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
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
    %v305 = stablehlo.broadcast_in_dim %s2b0b2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
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
    %v333 = stablehlo.broadcast_in_dim %s2b1b1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
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
    %v360 = stablehlo.broadcast_in_dim %s2b1b2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
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
    %v388 = stablehlo.broadcast_in_dim %s2b2b1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
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
    %v415 = stablehlo.broadcast_in_dim %s2b2b2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
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
    %v443 = stablehlo.broadcast_in_dim %d3b1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
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
    %v470 = stablehlo.broadcast_in_dim %d3b2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
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
    %v495 = stablehlo.broadcast_in_dim %d3bp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
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
    %v523 = stablehlo.broadcast_in_dim %s3b0b1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
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
    %v550 = stablehlo.broadcast_in_dim %s3b0b2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
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
    %v578 = stablehlo.broadcast_in_dim %s3b1b1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
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
    %v605 = stablehlo.broadcast_in_dim %s3b1b2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
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
    %v633 = stablehlo.broadcast_in_dim %s3b2b1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
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
    %v660 = stablehlo.broadcast_in_dim %s3b2b2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
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
    %v688 = stablehlo.broadcast_in_dim %s3b3b1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
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
    %v715 = stablehlo.broadcast_in_dim %s3b3b2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
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
    %v743 = stablehlo.broadcast_in_dim %s3b4b1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
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
    %v770 = stablehlo.broadcast_in_dim %s3b4b2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
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
    %v798 = stablehlo.broadcast_in_dim %d4b1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
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
    %v825 = stablehlo.broadcast_in_dim %d4b2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
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
    %v850 = stablehlo.broadcast_in_dim %d4bp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
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
    %v878 = stablehlo.broadcast_in_dim %s4b0b1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
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
    %v905 = stablehlo.broadcast_in_dim %s4b0b2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
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
    %v933 = stablehlo.broadcast_in_dim %s4b1b1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
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
    %v960 = stablehlo.broadcast_in_dim %s4b1b2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
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
    %v1100 = stablehlo.reshape %v1084 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1101 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1102 = stablehlo.reduce(%v1100 init: %v1101) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1103 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1104 = stablehlo.multiply %v1102, %v1103 : tensor<512xf32>
    %v1105 = stablehlo.subtract %s4b1b1, %v1104 : tensor<512xf32>
    %v1106 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1107 = stablehlo.reshape %v935 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1108 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1109 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1110 = stablehlo.reduce(%v1107 init: %v1106) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1111 = stablehlo.broadcast_in_dim %v1110, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1112 = stablehlo.divide %v1111, %v1108 : tensor<32x512x7x7xf32>
    %v1113 = stablehlo.subtract %v1107, %v1112 : tensor<32x512x7x7xf32>
    %v1114 = stablehlo.multiply %v1113, %v1113 : tensor<32x512x7x7xf32>
    %v1115 = stablehlo.reduce(%v1114 init: %v1106) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1116 = stablehlo.broadcast_in_dim %v1115, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1117 = stablehlo.divide %v1116, %v1108 : tensor<32x512x7x7xf32>
    %v1118 = stablehlo.add %v1117, %v1109 : tensor<32x512x7x7xf32>
    %v1119 = stablehlo.rsqrt %v1118 : tensor<32x512x7x7xf32>
    %v1120 = stablehlo.multiply %v1113, %v1119 : tensor<32x512x7x7xf32>
    %v1121 = stablehlo.reshape %v1054 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1122 = stablehlo.multiply %v1121, %v1120 : tensor<32x512x7x7xf32>
    %v1123 = stablehlo.reduce(%v1122 init: %v1106) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1124 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1125 = stablehlo.multiply %v1123, %v1124 : tensor<512xf32>
    %v1126 = stablehlo.subtract %s4b1g1, %v1125 : tensor<512xf32>
    %v1127 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1128 = stablehlo.reshape %v1054 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1129 = stablehlo.reduce(%v1128 init: %v1127) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1130 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1131 = stablehlo.multiply %v1129, %v1130 : tensor<512xf32>
    %v1132 = stablehlo.subtract %s4b1bt1, %v1131 : tensor<512xf32>
    %v1133 = stablehlo.reshape %v957 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1134 = stablehlo.reshape %v1046 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1135 = stablehlo.transpose %v1133, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1136 = stablehlo.transpose %v1134, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1137 = stablehlo.convolution(%v1135, %v1136)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1138 = stablehlo.transpose %v1137, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1139 = stablehlo.constant dense<0.003125> : tensor<512x512x3x3xf32>
    %v1140 = stablehlo.multiply %v1138, %v1139 : tensor<512x512x3x3xf32>
    %v1141 = stablehlo.subtract %s4b1W2, %v1140 : tensor<512x512x3x3xf32>
    %v1142 = stablehlo.reshape %v1046 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1143 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1144 = stablehlo.reduce(%v1142 init: %v1143) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1145 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1146 = stablehlo.multiply %v1144, %v1145 : tensor<512xf32>
    %v1147 = stablehlo.subtract %s4b1b2, %v1146 : tensor<512xf32>
    %v1148 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1149 = stablehlo.reshape %v962 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1150 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1151 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1152 = stablehlo.reduce(%v1149 init: %v1148) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1153 = stablehlo.broadcast_in_dim %v1152, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1154 = stablehlo.divide %v1153, %v1150 : tensor<32x512x7x7xf32>
    %v1155 = stablehlo.subtract %v1149, %v1154 : tensor<32x512x7x7xf32>
    %v1156 = stablehlo.multiply %v1155, %v1155 : tensor<32x512x7x7xf32>
    %v1157 = stablehlo.reduce(%v1156 init: %v1148) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1158 = stablehlo.broadcast_in_dim %v1157, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1159 = stablehlo.divide %v1158, %v1150 : tensor<32x512x7x7xf32>
    %v1160 = stablehlo.add %v1159, %v1151 : tensor<32x512x7x7xf32>
    %v1161 = stablehlo.rsqrt %v1160 : tensor<32x512x7x7xf32>
    %v1162 = stablehlo.multiply %v1155, %v1161 : tensor<32x512x7x7xf32>
    %v1163 = stablehlo.reshape %v1016 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1164 = stablehlo.multiply %v1163, %v1162 : tensor<32x512x7x7xf32>
    %v1165 = stablehlo.reduce(%v1164 init: %v1148) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1166 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1167 = stablehlo.multiply %v1165, %v1166 : tensor<512xf32>
    %v1168 = stablehlo.subtract %s4b1g2, %v1167 : tensor<512xf32>
    %v1169 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1170 = stablehlo.reshape %v1016 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1171 = stablehlo.reduce(%v1170 init: %v1169) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1172 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1173 = stablehlo.multiply %v1171, %v1172 : tensor<512xf32>
    %v1174 = stablehlo.subtract %s4b1bt2, %v1173 : tensor<512xf32>
    %v1175 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v1176 = stablehlo.compare GT, %v928, %v1175 : (tensor<32x25088xf32>, tensor<32x25088xf32>) -> tensor<32x25088xi1>
    %v1177 = stablehlo.select %v1176, %v1090, %v1175 : tensor<32x25088xi1>, tensor<32x25088xf32>
    %v1178 = stablehlo.reshape %v1177 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1179 = stablehlo.reshape %v907 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1180 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1181 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1182 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1183 = stablehlo.reduce(%v1179 init: %v1180) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1184 = stablehlo.broadcast_in_dim %v1183, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1185 = stablehlo.divide %v1184, %v1181 : tensor<32x512x7x7xf32>
    %v1186 = stablehlo.subtract %v1179, %v1185 : tensor<32x512x7x7xf32>
    %v1187 = stablehlo.multiply %v1186, %v1186 : tensor<32x512x7x7xf32>
    %v1188 = stablehlo.reduce(%v1187 init: %v1180) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1189 = stablehlo.broadcast_in_dim %v1188, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1190 = stablehlo.divide %v1189, %v1181 : tensor<32x512x7x7xf32>
    %v1191 = stablehlo.add %v1190, %v1182 : tensor<32x512x7x7xf32>
    %v1192 = stablehlo.rsqrt %v1191 : tensor<32x512x7x7xf32>
    %v1193 = stablehlo.multiply %v1186, %v1192 : tensor<32x512x7x7xf32>
    %v1194 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1195 = stablehlo.multiply %v1194, %v1178 : tensor<32x512x7x7xf32>
    %v1196 = stablehlo.reduce(%v1195 init: %v1180) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1197 = stablehlo.broadcast_in_dim %v1196, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1198 = stablehlo.multiply %v1193, %v1195 : tensor<32x512x7x7xf32>
    %v1199 = stablehlo.reduce(%v1198 init: %v1180) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1200 = stablehlo.broadcast_in_dim %v1199, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1201 = stablehlo.multiply %v1195, %v1181 : tensor<32x512x7x7xf32>
    %v1202 = stablehlo.subtract %v1201, %v1197 : tensor<32x512x7x7xf32>
    %v1203 = stablehlo.multiply %v1193, %v1200 : tensor<32x512x7x7xf32>
    %v1204 = stablehlo.subtract %v1202, %v1203 : tensor<32x512x7x7xf32>
    %v1205 = stablehlo.divide %v1192, %v1181 : tensor<32x512x7x7xf32>
    %v1206 = stablehlo.multiply %v1205, %v1204 : tensor<32x512x7x7xf32>
    %v1207 = stablehlo.reshape %v1206 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1208 = stablehlo.reshape %v1207 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1209 = stablehlo.transpose %s4b0W2, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1210 = stablehlo.reverse %v1209, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1211 = stablehlo.convolution(%v1208, %v1210)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1212 = stablehlo.reshape %v1211 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1213 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v1214 = stablehlo.compare GT, %v900, %v1213 : (tensor<32x25088xf32>, tensor<32x25088xf32>) -> tensor<32x25088xi1>
    %v1215 = stablehlo.select %v1214, %v1212, %v1213 : tensor<32x25088xi1>, tensor<32x25088xf32>
    %v1216 = stablehlo.reshape %v1215 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1217 = stablehlo.reshape %v880 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1218 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1219 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1220 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1221 = stablehlo.reduce(%v1217 init: %v1218) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1222 = stablehlo.broadcast_in_dim %v1221, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1223 = stablehlo.divide %v1222, %v1219 : tensor<32x512x7x7xf32>
    %v1224 = stablehlo.subtract %v1217, %v1223 : tensor<32x512x7x7xf32>
    %v1225 = stablehlo.multiply %v1224, %v1224 : tensor<32x512x7x7xf32>
    %v1226 = stablehlo.reduce(%v1225 init: %v1218) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1227 = stablehlo.broadcast_in_dim %v1226, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1228 = stablehlo.divide %v1227, %v1219 : tensor<32x512x7x7xf32>
    %v1229 = stablehlo.add %v1228, %v1220 : tensor<32x512x7x7xf32>
    %v1230 = stablehlo.rsqrt %v1229 : tensor<32x512x7x7xf32>
    %v1231 = stablehlo.multiply %v1224, %v1230 : tensor<32x512x7x7xf32>
    %v1232 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1233 = stablehlo.multiply %v1232, %v1216 : tensor<32x512x7x7xf32>
    %v1234 = stablehlo.reduce(%v1233 init: %v1218) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1235 = stablehlo.broadcast_in_dim %v1234, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1236 = stablehlo.multiply %v1231, %v1233 : tensor<32x512x7x7xf32>
    %v1237 = stablehlo.reduce(%v1236 init: %v1218) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1238 = stablehlo.broadcast_in_dim %v1237, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1239 = stablehlo.multiply %v1233, %v1219 : tensor<32x512x7x7xf32>
    %v1240 = stablehlo.subtract %v1239, %v1235 : tensor<32x512x7x7xf32>
    %v1241 = stablehlo.multiply %v1231, %v1238 : tensor<32x512x7x7xf32>
    %v1242 = stablehlo.subtract %v1240, %v1241 : tensor<32x512x7x7xf32>
    %v1243 = stablehlo.divide %v1230, %v1219 : tensor<32x512x7x7xf32>
    %v1244 = stablehlo.multiply %v1243, %v1242 : tensor<32x512x7x7xf32>
    %v1245 = stablehlo.reshape %v1244 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1246 = stablehlo.reshape %v1245 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1247 = stablehlo.transpose %s4b0W1, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1248 = stablehlo.reverse %v1247, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1249 = stablehlo.convolution(%v1246, %v1248)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1250 = stablehlo.reshape %v1249 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1251 = stablehlo.add %v1250, %v1177 : tensor<32x25088xf32>
    %v1252 = stablehlo.reshape %v875 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1253 = stablehlo.reshape %v1245 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1254 = stablehlo.transpose %v1252, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1255 = stablehlo.transpose %v1253, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1256 = stablehlo.convolution(%v1254, %v1255)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1257 = stablehlo.transpose %v1256, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1258 = stablehlo.constant dense<0.003125> : tensor<512x512x3x3xf32>
    %v1259 = stablehlo.multiply %v1257, %v1258 : tensor<512x512x3x3xf32>
    %v1260 = stablehlo.subtract %s4b0W1, %v1259 : tensor<512x512x3x3xf32>
    %v1261 = stablehlo.reshape %v1245 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1262 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1263 = stablehlo.reduce(%v1261 init: %v1262) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1264 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1265 = stablehlo.multiply %v1263, %v1264 : tensor<512xf32>
    %v1266 = stablehlo.subtract %s4b0b1, %v1265 : tensor<512xf32>
    %v1267 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1268 = stablehlo.reshape %v880 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1269 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1270 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1271 = stablehlo.reduce(%v1268 init: %v1267) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1272 = stablehlo.broadcast_in_dim %v1271, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1273 = stablehlo.divide %v1272, %v1269 : tensor<32x512x7x7xf32>
    %v1274 = stablehlo.subtract %v1268, %v1273 : tensor<32x512x7x7xf32>
    %v1275 = stablehlo.multiply %v1274, %v1274 : tensor<32x512x7x7xf32>
    %v1276 = stablehlo.reduce(%v1275 init: %v1267) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1277 = stablehlo.broadcast_in_dim %v1276, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1278 = stablehlo.divide %v1277, %v1269 : tensor<32x512x7x7xf32>
    %v1279 = stablehlo.add %v1278, %v1270 : tensor<32x512x7x7xf32>
    %v1280 = stablehlo.rsqrt %v1279 : tensor<32x512x7x7xf32>
    %v1281 = stablehlo.multiply %v1274, %v1280 : tensor<32x512x7x7xf32>
    %v1282 = stablehlo.reshape %v1215 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1283 = stablehlo.multiply %v1282, %v1281 : tensor<32x512x7x7xf32>
    %v1284 = stablehlo.reduce(%v1283 init: %v1267) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1285 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1286 = stablehlo.multiply %v1284, %v1285 : tensor<512xf32>
    %v1287 = stablehlo.subtract %s4b0g1, %v1286 : tensor<512xf32>
    %v1288 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1289 = stablehlo.reshape %v1215 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1290 = stablehlo.reduce(%v1289 init: %v1288) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1291 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1292 = stablehlo.multiply %v1290, %v1291 : tensor<512xf32>
    %v1293 = stablehlo.subtract %s4b0bt1, %v1292 : tensor<512xf32>
    %v1294 = stablehlo.reshape %v902 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1295 = stablehlo.reshape %v1207 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1296 = stablehlo.transpose %v1294, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1297 = stablehlo.transpose %v1295, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1298 = stablehlo.convolution(%v1296, %v1297)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1299 = stablehlo.transpose %v1298, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1300 = stablehlo.constant dense<0.003125> : tensor<512x512x3x3xf32>
    %v1301 = stablehlo.multiply %v1299, %v1300 : tensor<512x512x3x3xf32>
    %v1302 = stablehlo.subtract %s4b0W2, %v1301 : tensor<512x512x3x3xf32>
    %v1303 = stablehlo.reshape %v1207 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1304 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1305 = stablehlo.reduce(%v1303 init: %v1304) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1306 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1307 = stablehlo.multiply %v1305, %v1306 : tensor<512xf32>
    %v1308 = stablehlo.subtract %s4b0b2, %v1307 : tensor<512xf32>
    %v1309 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1310 = stablehlo.reshape %v907 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1311 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1312 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1313 = stablehlo.reduce(%v1310 init: %v1309) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1314 = stablehlo.broadcast_in_dim %v1313, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1315 = stablehlo.divide %v1314, %v1311 : tensor<32x512x7x7xf32>
    %v1316 = stablehlo.subtract %v1310, %v1315 : tensor<32x512x7x7xf32>
    %v1317 = stablehlo.multiply %v1316, %v1316 : tensor<32x512x7x7xf32>
    %v1318 = stablehlo.reduce(%v1317 init: %v1309) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1319 = stablehlo.broadcast_in_dim %v1318, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1320 = stablehlo.divide %v1319, %v1311 : tensor<32x512x7x7xf32>
    %v1321 = stablehlo.add %v1320, %v1312 : tensor<32x512x7x7xf32>
    %v1322 = stablehlo.rsqrt %v1321 : tensor<32x512x7x7xf32>
    %v1323 = stablehlo.multiply %v1316, %v1322 : tensor<32x512x7x7xf32>
    %v1324 = stablehlo.reshape %v1177 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1325 = stablehlo.multiply %v1324, %v1323 : tensor<32x512x7x7xf32>
    %v1326 = stablehlo.reduce(%v1325 init: %v1309) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1327 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1328 = stablehlo.multiply %v1326, %v1327 : tensor<512xf32>
    %v1329 = stablehlo.subtract %s4b0g2, %v1328 : tensor<512xf32>
    %v1330 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1331 = stablehlo.reshape %v1177 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1332 = stablehlo.reduce(%v1331 init: %v1330) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1333 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1334 = stablehlo.multiply %v1332, %v1333 : tensor<512xf32>
    %v1335 = stablehlo.subtract %s4b0bt2, %v1334 : tensor<512xf32>
    %v1336 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v1337 = stablehlo.compare GT, %v873, %v1336 : (tensor<32x25088xf32>, tensor<32x25088xf32>) -> tensor<32x25088xi1>
    %v1338 = stablehlo.select %v1337, %v1251, %v1336 : tensor<32x25088xi1>, tensor<32x25088xf32>
    %v1339 = stablehlo.reshape %v1338 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1340 = stablehlo.reshape %v827 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1341 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1342 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1343 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1344 = stablehlo.reduce(%v1340 init: %v1341) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1345 = stablehlo.broadcast_in_dim %v1344, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1346 = stablehlo.divide %v1345, %v1342 : tensor<32x512x7x7xf32>
    %v1347 = stablehlo.subtract %v1340, %v1346 : tensor<32x512x7x7xf32>
    %v1348 = stablehlo.multiply %v1347, %v1347 : tensor<32x512x7x7xf32>
    %v1349 = stablehlo.reduce(%v1348 init: %v1341) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1350 = stablehlo.broadcast_in_dim %v1349, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1351 = stablehlo.divide %v1350, %v1342 : tensor<32x512x7x7xf32>
    %v1352 = stablehlo.add %v1351, %v1343 : tensor<32x512x7x7xf32>
    %v1353 = stablehlo.rsqrt %v1352 : tensor<32x512x7x7xf32>
    %v1354 = stablehlo.multiply %v1347, %v1353 : tensor<32x512x7x7xf32>
    %v1355 = stablehlo.broadcast_in_dim %d4g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1356 = stablehlo.multiply %v1355, %v1339 : tensor<32x512x7x7xf32>
    %v1357 = stablehlo.reduce(%v1356 init: %v1341) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1358 = stablehlo.broadcast_in_dim %v1357, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1359 = stablehlo.multiply %v1354, %v1356 : tensor<32x512x7x7xf32>
    %v1360 = stablehlo.reduce(%v1359 init: %v1341) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1361 = stablehlo.broadcast_in_dim %v1360, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1362 = stablehlo.multiply %v1356, %v1342 : tensor<32x512x7x7xf32>
    %v1363 = stablehlo.subtract %v1362, %v1358 : tensor<32x512x7x7xf32>
    %v1364 = stablehlo.multiply %v1354, %v1361 : tensor<32x512x7x7xf32>
    %v1365 = stablehlo.subtract %v1363, %v1364 : tensor<32x512x7x7xf32>
    %v1366 = stablehlo.divide %v1353, %v1342 : tensor<32x512x7x7xf32>
    %v1367 = stablehlo.multiply %v1366, %v1365 : tensor<32x512x7x7xf32>
    %v1368 = stablehlo.reshape %v1367 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1369 = stablehlo.reshape %v1368 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1370 = stablehlo.transpose %d4W2, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1371 = stablehlo.reverse %v1370, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1372 = stablehlo.convolution(%v1369, %v1371)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1373 = stablehlo.reshape %v1372 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1374 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v1375 = stablehlo.compare GT, %v820, %v1374 : (tensor<32x25088xf32>, tensor<32x25088xf32>) -> tensor<32x25088xi1>
    %v1376 = stablehlo.select %v1375, %v1373, %v1374 : tensor<32x25088xi1>, tensor<32x25088xf32>
    %v1377 = stablehlo.reshape %v1376 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1378 = stablehlo.reshape %v800 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1379 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1380 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1381 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1382 = stablehlo.reduce(%v1378 init: %v1379) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1383 = stablehlo.broadcast_in_dim %v1382, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1384 = stablehlo.divide %v1383, %v1380 : tensor<32x512x7x7xf32>
    %v1385 = stablehlo.subtract %v1378, %v1384 : tensor<32x512x7x7xf32>
    %v1386 = stablehlo.multiply %v1385, %v1385 : tensor<32x512x7x7xf32>
    %v1387 = stablehlo.reduce(%v1386 init: %v1379) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1388 = stablehlo.broadcast_in_dim %v1387, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1389 = stablehlo.divide %v1388, %v1380 : tensor<32x512x7x7xf32>
    %v1390 = stablehlo.add %v1389, %v1381 : tensor<32x512x7x7xf32>
    %v1391 = stablehlo.rsqrt %v1390 : tensor<32x512x7x7xf32>
    %v1392 = stablehlo.multiply %v1385, %v1391 : tensor<32x512x7x7xf32>
    %v1393 = stablehlo.broadcast_in_dim %d4g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1394 = stablehlo.multiply %v1393, %v1377 : tensor<32x512x7x7xf32>
    %v1395 = stablehlo.reduce(%v1394 init: %v1379) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1396 = stablehlo.broadcast_in_dim %v1395, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1397 = stablehlo.multiply %v1392, %v1394 : tensor<32x512x7x7xf32>
    %v1398 = stablehlo.reduce(%v1397 init: %v1379) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1399 = stablehlo.broadcast_in_dim %v1398, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1400 = stablehlo.multiply %v1394, %v1380 : tensor<32x512x7x7xf32>
    %v1401 = stablehlo.subtract %v1400, %v1396 : tensor<32x512x7x7xf32>
    %v1402 = stablehlo.multiply %v1392, %v1399 : tensor<32x512x7x7xf32>
    %v1403 = stablehlo.subtract %v1401, %v1402 : tensor<32x512x7x7xf32>
    %v1404 = stablehlo.divide %v1391, %v1380 : tensor<32x512x7x7xf32>
    %v1405 = stablehlo.multiply %v1404, %v1403 : tensor<32x512x7x7xf32>
    %v1406 = stablehlo.reshape %v1405 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1407 = stablehlo.reshape %v1406 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1408 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1409 = stablehlo.pad %v1407, %v1408, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %v1410 = stablehlo.transpose %d4W1, dims = [1, 0, 2, 3] : (tensor<512x256x3x3xf32>) -> tensor<256x512x3x3xf32>
    %v1411 = stablehlo.reverse %v1410, dims = [2, 3] : tensor<256x512x3x3xf32>
    %v1412 = stablehlo.convolution(%v1409, %v1411)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<256x512x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1413 = stablehlo.reshape %v1412 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1414 = stablehlo.reshape %v1338 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1415 = stablehlo.reshape %v852 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1416 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1417 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1418 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1419 = stablehlo.reduce(%v1415 init: %v1416) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1420 = stablehlo.broadcast_in_dim %v1419, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1421 = stablehlo.divide %v1420, %v1417 : tensor<32x512x7x7xf32>
    %v1422 = stablehlo.subtract %v1415, %v1421 : tensor<32x512x7x7xf32>
    %v1423 = stablehlo.multiply %v1422, %v1422 : tensor<32x512x7x7xf32>
    %v1424 = stablehlo.reduce(%v1423 init: %v1416) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1425 = stablehlo.broadcast_in_dim %v1424, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1426 = stablehlo.divide %v1425, %v1417 : tensor<32x512x7x7xf32>
    %v1427 = stablehlo.add %v1426, %v1418 : tensor<32x512x7x7xf32>
    %v1428 = stablehlo.rsqrt %v1427 : tensor<32x512x7x7xf32>
    %v1429 = stablehlo.multiply %v1422, %v1428 : tensor<32x512x7x7xf32>
    %v1430 = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1431 = stablehlo.multiply %v1430, %v1414 : tensor<32x512x7x7xf32>
    %v1432 = stablehlo.reduce(%v1431 init: %v1416) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1433 = stablehlo.broadcast_in_dim %v1432, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1434 = stablehlo.multiply %v1429, %v1431 : tensor<32x512x7x7xf32>
    %v1435 = stablehlo.reduce(%v1434 init: %v1416) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1436 = stablehlo.broadcast_in_dim %v1435, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1437 = stablehlo.multiply %v1431, %v1417 : tensor<32x512x7x7xf32>
    %v1438 = stablehlo.subtract %v1437, %v1433 : tensor<32x512x7x7xf32>
    %v1439 = stablehlo.multiply %v1429, %v1436 : tensor<32x512x7x7xf32>
    %v1440 = stablehlo.subtract %v1438, %v1439 : tensor<32x512x7x7xf32>
    %v1441 = stablehlo.divide %v1428, %v1417 : tensor<32x512x7x7xf32>
    %v1442 = stablehlo.multiply %v1441, %v1440 : tensor<32x512x7x7xf32>
    %v1443 = stablehlo.reshape %v1442 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1444 = stablehlo.reshape %v1443 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1445 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1446 = stablehlo.pad %v1444, %v1445, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %v1447 = stablehlo.transpose %d4Wp, dims = [1, 0, 2, 3] : (tensor<512x256x3x3xf32>) -> tensor<256x512x3x3xf32>
    %v1448 = stablehlo.reverse %v1447, dims = [2, 3] : tensor<256x512x3x3xf32>
    %v1449 = stablehlo.convolution(%v1446, %v1448)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<256x512x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1450 = stablehlo.reshape %v1449 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1451 = stablehlo.add %v1413, %v1450 : tensor<32x50176xf32>
    %v1452 = stablehlo.reshape %v795 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1453 = stablehlo.reshape %v1406 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1454 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1455 = stablehlo.pad %v1453, %v1454, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %v1456 = stablehlo.transpose %v1452, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1457 = stablehlo.transpose %v1455, dims = [1, 0, 2, 3] : (tensor<32x512x14x14xf32>) -> tensor<512x32x14x14xf32>
    %v1458 = stablehlo.convolution(%v1456, %v1457)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<512x32x14x14xf32>) -> tensor<256x512x3x3xf32>
    %v1459 = stablehlo.transpose %v1458, dims = [1, 0, 2, 3] : (tensor<256x512x3x3xf32>) -> tensor<512x256x3x3xf32>
    %v1460 = stablehlo.constant dense<0.003125> : tensor<512x256x3x3xf32>
    %v1461 = stablehlo.multiply %v1459, %v1460 : tensor<512x256x3x3xf32>
    %v1462 = stablehlo.subtract %d4W1, %v1461 : tensor<512x256x3x3xf32>
    %v1463 = stablehlo.reshape %v1406 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1464 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1465 = stablehlo.reduce(%v1463 init: %v1464) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1466 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1467 = stablehlo.multiply %v1465, %v1466 : tensor<512xf32>
    %v1468 = stablehlo.subtract %d4b1, %v1467 : tensor<512xf32>
    %v1469 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1470 = stablehlo.reshape %v800 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1471 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1472 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1473 = stablehlo.reduce(%v1470 init: %v1469) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1474 = stablehlo.broadcast_in_dim %v1473, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1475 = stablehlo.divide %v1474, %v1471 : tensor<32x512x7x7xf32>
    %v1476 = stablehlo.subtract %v1470, %v1475 : tensor<32x512x7x7xf32>
    %v1477 = stablehlo.multiply %v1476, %v1476 : tensor<32x512x7x7xf32>
    %v1478 = stablehlo.reduce(%v1477 init: %v1469) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1479 = stablehlo.broadcast_in_dim %v1478, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1480 = stablehlo.divide %v1479, %v1471 : tensor<32x512x7x7xf32>
    %v1481 = stablehlo.add %v1480, %v1472 : tensor<32x512x7x7xf32>
    %v1482 = stablehlo.rsqrt %v1481 : tensor<32x512x7x7xf32>
    %v1483 = stablehlo.multiply %v1476, %v1482 : tensor<32x512x7x7xf32>
    %v1484 = stablehlo.reshape %v1376 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1485 = stablehlo.multiply %v1484, %v1483 : tensor<32x512x7x7xf32>
    %v1486 = stablehlo.reduce(%v1485 init: %v1469) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1487 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1488 = stablehlo.multiply %v1486, %v1487 : tensor<512xf32>
    %v1489 = stablehlo.subtract %d4g1, %v1488 : tensor<512xf32>
    %v1490 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1491 = stablehlo.reshape %v1376 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1492 = stablehlo.reduce(%v1491 init: %v1490) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1493 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1494 = stablehlo.multiply %v1492, %v1493 : tensor<512xf32>
    %v1495 = stablehlo.subtract %d4bt1, %v1494 : tensor<512xf32>
    %v1496 = stablehlo.reshape %v822 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1497 = stablehlo.reshape %v1368 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1498 = stablehlo.transpose %v1496, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1499 = stablehlo.transpose %v1497, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1500 = stablehlo.convolution(%v1498, %v1499)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1501 = stablehlo.transpose %v1500, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1502 = stablehlo.constant dense<0.003125> : tensor<512x512x3x3xf32>
    %v1503 = stablehlo.multiply %v1501, %v1502 : tensor<512x512x3x3xf32>
    %v1504 = stablehlo.subtract %d4W2, %v1503 : tensor<512x512x3x3xf32>
    %v1505 = stablehlo.reshape %v1368 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1506 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1507 = stablehlo.reduce(%v1505 init: %v1506) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1508 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1509 = stablehlo.multiply %v1507, %v1508 : tensor<512xf32>
    %v1510 = stablehlo.subtract %d4b2, %v1509 : tensor<512xf32>
    %v1511 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1512 = stablehlo.reshape %v827 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1513 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1514 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1515 = stablehlo.reduce(%v1512 init: %v1511) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1516 = stablehlo.broadcast_in_dim %v1515, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1517 = stablehlo.divide %v1516, %v1513 : tensor<32x512x7x7xf32>
    %v1518 = stablehlo.subtract %v1512, %v1517 : tensor<32x512x7x7xf32>
    %v1519 = stablehlo.multiply %v1518, %v1518 : tensor<32x512x7x7xf32>
    %v1520 = stablehlo.reduce(%v1519 init: %v1511) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1521 = stablehlo.broadcast_in_dim %v1520, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1522 = stablehlo.divide %v1521, %v1513 : tensor<32x512x7x7xf32>
    %v1523 = stablehlo.add %v1522, %v1514 : tensor<32x512x7x7xf32>
    %v1524 = stablehlo.rsqrt %v1523 : tensor<32x512x7x7xf32>
    %v1525 = stablehlo.multiply %v1518, %v1524 : tensor<32x512x7x7xf32>
    %v1526 = stablehlo.reshape %v1338 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1527 = stablehlo.multiply %v1526, %v1525 : tensor<32x512x7x7xf32>
    %v1528 = stablehlo.reduce(%v1527 init: %v1511) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1529 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1530 = stablehlo.multiply %v1528, %v1529 : tensor<512xf32>
    %v1531 = stablehlo.subtract %d4g2, %v1530 : tensor<512xf32>
    %v1532 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1533 = stablehlo.reshape %v1338 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1534 = stablehlo.reduce(%v1533 init: %v1532) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1535 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1536 = stablehlo.multiply %v1534, %v1535 : tensor<512xf32>
    %v1537 = stablehlo.subtract %d4bt2, %v1536 : tensor<512xf32>
    %v1538 = stablehlo.reshape %v795 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1539 = stablehlo.reshape %v1443 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1540 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1541 = stablehlo.pad %v1539, %v1540, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %v1542 = stablehlo.transpose %v1538, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1543 = stablehlo.transpose %v1541, dims = [1, 0, 2, 3] : (tensor<32x512x14x14xf32>) -> tensor<512x32x14x14xf32>
    %v1544 = stablehlo.convolution(%v1542, %v1543)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<512x32x14x14xf32>) -> tensor<256x512x3x3xf32>
    %v1545 = stablehlo.transpose %v1544, dims = [1, 0, 2, 3] : (tensor<256x512x3x3xf32>) -> tensor<512x256x3x3xf32>
    %v1546 = stablehlo.constant dense<0.003125> : tensor<512x256x3x3xf32>
    %v1547 = stablehlo.multiply %v1545, %v1546 : tensor<512x256x3x3xf32>
    %v1548 = stablehlo.subtract %d4Wp, %v1547 : tensor<512x256x3x3xf32>
    %v1549 = stablehlo.reshape %v1443 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1550 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1551 = stablehlo.reduce(%v1549 init: %v1550) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1552 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1553 = stablehlo.multiply %v1551, %v1552 : tensor<512xf32>
    %v1554 = stablehlo.subtract %d4bp, %v1553 : tensor<512xf32>
    %v1555 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1556 = stablehlo.reshape %v852 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1557 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1558 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1559 = stablehlo.reduce(%v1556 init: %v1555) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1560 = stablehlo.broadcast_in_dim %v1559, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1561 = stablehlo.divide %v1560, %v1557 : tensor<32x512x7x7xf32>
    %v1562 = stablehlo.subtract %v1556, %v1561 : tensor<32x512x7x7xf32>
    %v1563 = stablehlo.multiply %v1562, %v1562 : tensor<32x512x7x7xf32>
    %v1564 = stablehlo.reduce(%v1563 init: %v1555) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1565 = stablehlo.broadcast_in_dim %v1564, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1566 = stablehlo.divide %v1565, %v1557 : tensor<32x512x7x7xf32>
    %v1567 = stablehlo.add %v1566, %v1558 : tensor<32x512x7x7xf32>
    %v1568 = stablehlo.rsqrt %v1567 : tensor<32x512x7x7xf32>
    %v1569 = stablehlo.multiply %v1562, %v1568 : tensor<32x512x7x7xf32>
    %v1570 = stablehlo.reshape %v1338 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1571 = stablehlo.multiply %v1570, %v1569 : tensor<32x512x7x7xf32>
    %v1572 = stablehlo.reduce(%v1571 init: %v1555) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1573 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1574 = stablehlo.multiply %v1572, %v1573 : tensor<512xf32>
    %v1575 = stablehlo.subtract %d4gp, %v1574 : tensor<512xf32>
    %v1576 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1577 = stablehlo.reshape %v1338 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1578 = stablehlo.reduce(%v1577 init: %v1576) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1579 = stablehlo.constant dense<0.003125> : tensor<512xf32>
    %v1580 = stablehlo.multiply %v1578, %v1579 : tensor<512xf32>
    %v1581 = stablehlo.subtract %d4btp, %v1580 : tensor<512xf32>
    %v1582 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1583 = stablehlo.compare GT, %v793, %v1582 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v1584 = stablehlo.select %v1583, %v1451, %v1582 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v1585 = stablehlo.reshape %v1584 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1586 = stablehlo.reshape %v772 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1587 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1588 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v1589 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1590 = stablehlo.reduce(%v1586 init: %v1587) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1591 = stablehlo.broadcast_in_dim %v1590, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1592 = stablehlo.divide %v1591, %v1588 : tensor<32x256x14x14xf32>
    %v1593 = stablehlo.subtract %v1586, %v1592 : tensor<32x256x14x14xf32>
    %v1594 = stablehlo.multiply %v1593, %v1593 : tensor<32x256x14x14xf32>
    %v1595 = stablehlo.reduce(%v1594 init: %v1587) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1596 = stablehlo.broadcast_in_dim %v1595, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1597 = stablehlo.divide %v1596, %v1588 : tensor<32x256x14x14xf32>
    %v1598 = stablehlo.add %v1597, %v1589 : tensor<32x256x14x14xf32>
    %v1599 = stablehlo.rsqrt %v1598 : tensor<32x256x14x14xf32>
    %v1600 = stablehlo.multiply %v1593, %v1599 : tensor<32x256x14x14xf32>
    %v1601 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1602 = stablehlo.multiply %v1601, %v1585 : tensor<32x256x14x14xf32>
    %v1603 = stablehlo.reduce(%v1602 init: %v1587) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1604 = stablehlo.broadcast_in_dim %v1603, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1605 = stablehlo.multiply %v1600, %v1602 : tensor<32x256x14x14xf32>
    %v1606 = stablehlo.reduce(%v1605 init: %v1587) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1607 = stablehlo.broadcast_in_dim %v1606, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1608 = stablehlo.multiply %v1602, %v1588 : tensor<32x256x14x14xf32>
    %v1609 = stablehlo.subtract %v1608, %v1604 : tensor<32x256x14x14xf32>
    %v1610 = stablehlo.multiply %v1600, %v1607 : tensor<32x256x14x14xf32>
    %v1611 = stablehlo.subtract %v1609, %v1610 : tensor<32x256x14x14xf32>
    %v1612 = stablehlo.divide %v1599, %v1588 : tensor<32x256x14x14xf32>
    %v1613 = stablehlo.multiply %v1612, %v1611 : tensor<32x256x14x14xf32>
    %v1614 = stablehlo.reshape %v1613 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1615 = stablehlo.reshape %v1614 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1616 = stablehlo.transpose %s3b4W2, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1617 = stablehlo.reverse %v1616, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1618 = stablehlo.convolution(%v1615, %v1617)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1619 = stablehlo.reshape %v1618 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1620 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1621 = stablehlo.compare GT, %v765, %v1620 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v1622 = stablehlo.select %v1621, %v1619, %v1620 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v1623 = stablehlo.reshape %v1622 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1624 = stablehlo.reshape %v745 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1625 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1626 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v1627 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1628 = stablehlo.reduce(%v1624 init: %v1625) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1629 = stablehlo.broadcast_in_dim %v1628, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1630 = stablehlo.divide %v1629, %v1626 : tensor<32x256x14x14xf32>
    %v1631 = stablehlo.subtract %v1624, %v1630 : tensor<32x256x14x14xf32>
    %v1632 = stablehlo.multiply %v1631, %v1631 : tensor<32x256x14x14xf32>
    %v1633 = stablehlo.reduce(%v1632 init: %v1625) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1634 = stablehlo.broadcast_in_dim %v1633, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1635 = stablehlo.divide %v1634, %v1626 : tensor<32x256x14x14xf32>
    %v1636 = stablehlo.add %v1635, %v1627 : tensor<32x256x14x14xf32>
    %v1637 = stablehlo.rsqrt %v1636 : tensor<32x256x14x14xf32>
    %v1638 = stablehlo.multiply %v1631, %v1637 : tensor<32x256x14x14xf32>
    %v1639 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1640 = stablehlo.multiply %v1639, %v1623 : tensor<32x256x14x14xf32>
    %v1641 = stablehlo.reduce(%v1640 init: %v1625) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1642 = stablehlo.broadcast_in_dim %v1641, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1643 = stablehlo.multiply %v1638, %v1640 : tensor<32x256x14x14xf32>
    %v1644 = stablehlo.reduce(%v1643 init: %v1625) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1645 = stablehlo.broadcast_in_dim %v1644, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1646 = stablehlo.multiply %v1640, %v1626 : tensor<32x256x14x14xf32>
    %v1647 = stablehlo.subtract %v1646, %v1642 : tensor<32x256x14x14xf32>
    %v1648 = stablehlo.multiply %v1638, %v1645 : tensor<32x256x14x14xf32>
    %v1649 = stablehlo.subtract %v1647, %v1648 : tensor<32x256x14x14xf32>
    %v1650 = stablehlo.divide %v1637, %v1626 : tensor<32x256x14x14xf32>
    %v1651 = stablehlo.multiply %v1650, %v1649 : tensor<32x256x14x14xf32>
    %v1652 = stablehlo.reshape %v1651 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1653 = stablehlo.reshape %v1652 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1654 = stablehlo.transpose %s3b4W1, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1655 = stablehlo.reverse %v1654, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1656 = stablehlo.convolution(%v1653, %v1655)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1657 = stablehlo.reshape %v1656 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1658 = stablehlo.add %v1657, %v1584 : tensor<32x50176xf32>
    %v1659 = stablehlo.reshape %v740 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1660 = stablehlo.reshape %v1652 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1661 = stablehlo.transpose %v1659, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1662 = stablehlo.transpose %v1660, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1663 = stablehlo.convolution(%v1661, %v1662)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1664 = stablehlo.transpose %v1663, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1665 = stablehlo.constant dense<0.003125> : tensor<256x256x3x3xf32>
    %v1666 = stablehlo.multiply %v1664, %v1665 : tensor<256x256x3x3xf32>
    %v1667 = stablehlo.subtract %s3b4W1, %v1666 : tensor<256x256x3x3xf32>
    %v1668 = stablehlo.reshape %v1652 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1669 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1670 = stablehlo.reduce(%v1668 init: %v1669) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1671 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v1672 = stablehlo.multiply %v1670, %v1671 : tensor<256xf32>
    %v1673 = stablehlo.subtract %s3b4b1, %v1672 : tensor<256xf32>
    %v1674 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1675 = stablehlo.reshape %v745 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1676 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v1677 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1678 = stablehlo.reduce(%v1675 init: %v1674) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1679 = stablehlo.broadcast_in_dim %v1678, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1680 = stablehlo.divide %v1679, %v1676 : tensor<32x256x14x14xf32>
    %v1681 = stablehlo.subtract %v1675, %v1680 : tensor<32x256x14x14xf32>
    %v1682 = stablehlo.multiply %v1681, %v1681 : tensor<32x256x14x14xf32>
    %v1683 = stablehlo.reduce(%v1682 init: %v1674) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1684 = stablehlo.broadcast_in_dim %v1683, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1685 = stablehlo.divide %v1684, %v1676 : tensor<32x256x14x14xf32>
    %v1686 = stablehlo.add %v1685, %v1677 : tensor<32x256x14x14xf32>
    %v1687 = stablehlo.rsqrt %v1686 : tensor<32x256x14x14xf32>
    %v1688 = stablehlo.multiply %v1681, %v1687 : tensor<32x256x14x14xf32>
    %v1689 = stablehlo.reshape %v1622 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1690 = stablehlo.multiply %v1689, %v1688 : tensor<32x256x14x14xf32>
    %v1691 = stablehlo.reduce(%v1690 init: %v1674) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1692 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v1693 = stablehlo.multiply %v1691, %v1692 : tensor<256xf32>
    %v1694 = stablehlo.subtract %s3b4g1, %v1693 : tensor<256xf32>
    %v1695 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1696 = stablehlo.reshape %v1622 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1697 = stablehlo.reduce(%v1696 init: %v1695) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1698 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v1699 = stablehlo.multiply %v1697, %v1698 : tensor<256xf32>
    %v1700 = stablehlo.subtract %s3b4bt1, %v1699 : tensor<256xf32>
    %v1701 = stablehlo.reshape %v767 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1702 = stablehlo.reshape %v1614 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1703 = stablehlo.transpose %v1701, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1704 = stablehlo.transpose %v1702, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1705 = stablehlo.convolution(%v1703, %v1704)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1706 = stablehlo.transpose %v1705, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1707 = stablehlo.constant dense<0.003125> : tensor<256x256x3x3xf32>
    %v1708 = stablehlo.multiply %v1706, %v1707 : tensor<256x256x3x3xf32>
    %v1709 = stablehlo.subtract %s3b4W2, %v1708 : tensor<256x256x3x3xf32>
    %v1710 = stablehlo.reshape %v1614 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1711 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1712 = stablehlo.reduce(%v1710 init: %v1711) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1713 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v1714 = stablehlo.multiply %v1712, %v1713 : tensor<256xf32>
    %v1715 = stablehlo.subtract %s3b4b2, %v1714 : tensor<256xf32>
    %v1716 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1717 = stablehlo.reshape %v772 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1718 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v1719 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1720 = stablehlo.reduce(%v1717 init: %v1716) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1721 = stablehlo.broadcast_in_dim %v1720, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1722 = stablehlo.divide %v1721, %v1718 : tensor<32x256x14x14xf32>
    %v1723 = stablehlo.subtract %v1717, %v1722 : tensor<32x256x14x14xf32>
    %v1724 = stablehlo.multiply %v1723, %v1723 : tensor<32x256x14x14xf32>
    %v1725 = stablehlo.reduce(%v1724 init: %v1716) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1726 = stablehlo.broadcast_in_dim %v1725, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1727 = stablehlo.divide %v1726, %v1718 : tensor<32x256x14x14xf32>
    %v1728 = stablehlo.add %v1727, %v1719 : tensor<32x256x14x14xf32>
    %v1729 = stablehlo.rsqrt %v1728 : tensor<32x256x14x14xf32>
    %v1730 = stablehlo.multiply %v1723, %v1729 : tensor<32x256x14x14xf32>
    %v1731 = stablehlo.reshape %v1584 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1732 = stablehlo.multiply %v1731, %v1730 : tensor<32x256x14x14xf32>
    %v1733 = stablehlo.reduce(%v1732 init: %v1716) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1734 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v1735 = stablehlo.multiply %v1733, %v1734 : tensor<256xf32>
    %v1736 = stablehlo.subtract %s3b4g2, %v1735 : tensor<256xf32>
    %v1737 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1738 = stablehlo.reshape %v1584 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1739 = stablehlo.reduce(%v1738 init: %v1737) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1740 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v1741 = stablehlo.multiply %v1739, %v1740 : tensor<256xf32>
    %v1742 = stablehlo.subtract %s3b4bt2, %v1741 : tensor<256xf32>
    %v1743 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1744 = stablehlo.compare GT, %v738, %v1743 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v1745 = stablehlo.select %v1744, %v1658, %v1743 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v1746 = stablehlo.reshape %v1745 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1747 = stablehlo.reshape %v717 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1748 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1749 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v1750 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1751 = stablehlo.reduce(%v1747 init: %v1748) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1752 = stablehlo.broadcast_in_dim %v1751, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1753 = stablehlo.divide %v1752, %v1749 : tensor<32x256x14x14xf32>
    %v1754 = stablehlo.subtract %v1747, %v1753 : tensor<32x256x14x14xf32>
    %v1755 = stablehlo.multiply %v1754, %v1754 : tensor<32x256x14x14xf32>
    %v1756 = stablehlo.reduce(%v1755 init: %v1748) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1757 = stablehlo.broadcast_in_dim %v1756, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1758 = stablehlo.divide %v1757, %v1749 : tensor<32x256x14x14xf32>
    %v1759 = stablehlo.add %v1758, %v1750 : tensor<32x256x14x14xf32>
    %v1760 = stablehlo.rsqrt %v1759 : tensor<32x256x14x14xf32>
    %v1761 = stablehlo.multiply %v1754, %v1760 : tensor<32x256x14x14xf32>
    %v1762 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1763 = stablehlo.multiply %v1762, %v1746 : tensor<32x256x14x14xf32>
    %v1764 = stablehlo.reduce(%v1763 init: %v1748) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1765 = stablehlo.broadcast_in_dim %v1764, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1766 = stablehlo.multiply %v1761, %v1763 : tensor<32x256x14x14xf32>
    %v1767 = stablehlo.reduce(%v1766 init: %v1748) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1768 = stablehlo.broadcast_in_dim %v1767, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1769 = stablehlo.multiply %v1763, %v1749 : tensor<32x256x14x14xf32>
    %v1770 = stablehlo.subtract %v1769, %v1765 : tensor<32x256x14x14xf32>
    %v1771 = stablehlo.multiply %v1761, %v1768 : tensor<32x256x14x14xf32>
    %v1772 = stablehlo.subtract %v1770, %v1771 : tensor<32x256x14x14xf32>
    %v1773 = stablehlo.divide %v1760, %v1749 : tensor<32x256x14x14xf32>
    %v1774 = stablehlo.multiply %v1773, %v1772 : tensor<32x256x14x14xf32>
    %v1775 = stablehlo.reshape %v1774 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1776 = stablehlo.reshape %v1775 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1777 = stablehlo.transpose %s3b3W2, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1778 = stablehlo.reverse %v1777, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1779 = stablehlo.convolution(%v1776, %v1778)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1780 = stablehlo.reshape %v1779 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1781 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1782 = stablehlo.compare GT, %v710, %v1781 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v1783 = stablehlo.select %v1782, %v1780, %v1781 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v1784 = stablehlo.reshape %v1783 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1785 = stablehlo.reshape %v690 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1786 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1787 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v1788 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1789 = stablehlo.reduce(%v1785 init: %v1786) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1790 = stablehlo.broadcast_in_dim %v1789, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1791 = stablehlo.divide %v1790, %v1787 : tensor<32x256x14x14xf32>
    %v1792 = stablehlo.subtract %v1785, %v1791 : tensor<32x256x14x14xf32>
    %v1793 = stablehlo.multiply %v1792, %v1792 : tensor<32x256x14x14xf32>
    %v1794 = stablehlo.reduce(%v1793 init: %v1786) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1795 = stablehlo.broadcast_in_dim %v1794, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1796 = stablehlo.divide %v1795, %v1787 : tensor<32x256x14x14xf32>
    %v1797 = stablehlo.add %v1796, %v1788 : tensor<32x256x14x14xf32>
    %v1798 = stablehlo.rsqrt %v1797 : tensor<32x256x14x14xf32>
    %v1799 = stablehlo.multiply %v1792, %v1798 : tensor<32x256x14x14xf32>
    %v1800 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1801 = stablehlo.multiply %v1800, %v1784 : tensor<32x256x14x14xf32>
    %v1802 = stablehlo.reduce(%v1801 init: %v1786) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1803 = stablehlo.broadcast_in_dim %v1802, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1804 = stablehlo.multiply %v1799, %v1801 : tensor<32x256x14x14xf32>
    %v1805 = stablehlo.reduce(%v1804 init: %v1786) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1806 = stablehlo.broadcast_in_dim %v1805, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1807 = stablehlo.multiply %v1801, %v1787 : tensor<32x256x14x14xf32>
    %v1808 = stablehlo.subtract %v1807, %v1803 : tensor<32x256x14x14xf32>
    %v1809 = stablehlo.multiply %v1799, %v1806 : tensor<32x256x14x14xf32>
    %v1810 = stablehlo.subtract %v1808, %v1809 : tensor<32x256x14x14xf32>
    %v1811 = stablehlo.divide %v1798, %v1787 : tensor<32x256x14x14xf32>
    %v1812 = stablehlo.multiply %v1811, %v1810 : tensor<32x256x14x14xf32>
    %v1813 = stablehlo.reshape %v1812 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1814 = stablehlo.reshape %v1813 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1815 = stablehlo.transpose %s3b3W1, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1816 = stablehlo.reverse %v1815, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1817 = stablehlo.convolution(%v1814, %v1816)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1818 = stablehlo.reshape %v1817 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1819 = stablehlo.add %v1818, %v1745 : tensor<32x50176xf32>
    %v1820 = stablehlo.reshape %v685 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1821 = stablehlo.reshape %v1813 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1822 = stablehlo.transpose %v1820, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1823 = stablehlo.transpose %v1821, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1824 = stablehlo.convolution(%v1822, %v1823)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1825 = stablehlo.transpose %v1824, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1826 = stablehlo.constant dense<0.003125> : tensor<256x256x3x3xf32>
    %v1827 = stablehlo.multiply %v1825, %v1826 : tensor<256x256x3x3xf32>
    %v1828 = stablehlo.subtract %s3b3W1, %v1827 : tensor<256x256x3x3xf32>
    %v1829 = stablehlo.reshape %v1813 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1830 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1831 = stablehlo.reduce(%v1829 init: %v1830) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1832 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v1833 = stablehlo.multiply %v1831, %v1832 : tensor<256xf32>
    %v1834 = stablehlo.subtract %s3b3b1, %v1833 : tensor<256xf32>
    %v1835 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1836 = stablehlo.reshape %v690 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1837 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v1838 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1839 = stablehlo.reduce(%v1836 init: %v1835) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1840 = stablehlo.broadcast_in_dim %v1839, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1841 = stablehlo.divide %v1840, %v1837 : tensor<32x256x14x14xf32>
    %v1842 = stablehlo.subtract %v1836, %v1841 : tensor<32x256x14x14xf32>
    %v1843 = stablehlo.multiply %v1842, %v1842 : tensor<32x256x14x14xf32>
    %v1844 = stablehlo.reduce(%v1843 init: %v1835) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1845 = stablehlo.broadcast_in_dim %v1844, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1846 = stablehlo.divide %v1845, %v1837 : tensor<32x256x14x14xf32>
    %v1847 = stablehlo.add %v1846, %v1838 : tensor<32x256x14x14xf32>
    %v1848 = stablehlo.rsqrt %v1847 : tensor<32x256x14x14xf32>
    %v1849 = stablehlo.multiply %v1842, %v1848 : tensor<32x256x14x14xf32>
    %v1850 = stablehlo.reshape %v1783 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1851 = stablehlo.multiply %v1850, %v1849 : tensor<32x256x14x14xf32>
    %v1852 = stablehlo.reduce(%v1851 init: %v1835) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1853 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v1854 = stablehlo.multiply %v1852, %v1853 : tensor<256xf32>
    %v1855 = stablehlo.subtract %s3b3g1, %v1854 : tensor<256xf32>
    %v1856 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1857 = stablehlo.reshape %v1783 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1858 = stablehlo.reduce(%v1857 init: %v1856) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1859 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v1860 = stablehlo.multiply %v1858, %v1859 : tensor<256xf32>
    %v1861 = stablehlo.subtract %s3b3bt1, %v1860 : tensor<256xf32>
    %v1862 = stablehlo.reshape %v712 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1863 = stablehlo.reshape %v1775 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1864 = stablehlo.transpose %v1862, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1865 = stablehlo.transpose %v1863, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1866 = stablehlo.convolution(%v1864, %v1865)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1867 = stablehlo.transpose %v1866, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1868 = stablehlo.constant dense<0.003125> : tensor<256x256x3x3xf32>
    %v1869 = stablehlo.multiply %v1867, %v1868 : tensor<256x256x3x3xf32>
    %v1870 = stablehlo.subtract %s3b3W2, %v1869 : tensor<256x256x3x3xf32>
    %v1871 = stablehlo.reshape %v1775 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1872 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1873 = stablehlo.reduce(%v1871 init: %v1872) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1874 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v1875 = stablehlo.multiply %v1873, %v1874 : tensor<256xf32>
    %v1876 = stablehlo.subtract %s3b3b2, %v1875 : tensor<256xf32>
    %v1877 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1878 = stablehlo.reshape %v717 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1879 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v1880 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1881 = stablehlo.reduce(%v1878 init: %v1877) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1882 = stablehlo.broadcast_in_dim %v1881, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1883 = stablehlo.divide %v1882, %v1879 : tensor<32x256x14x14xf32>
    %v1884 = stablehlo.subtract %v1878, %v1883 : tensor<32x256x14x14xf32>
    %v1885 = stablehlo.multiply %v1884, %v1884 : tensor<32x256x14x14xf32>
    %v1886 = stablehlo.reduce(%v1885 init: %v1877) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1887 = stablehlo.broadcast_in_dim %v1886, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1888 = stablehlo.divide %v1887, %v1879 : tensor<32x256x14x14xf32>
    %v1889 = stablehlo.add %v1888, %v1880 : tensor<32x256x14x14xf32>
    %v1890 = stablehlo.rsqrt %v1889 : tensor<32x256x14x14xf32>
    %v1891 = stablehlo.multiply %v1884, %v1890 : tensor<32x256x14x14xf32>
    %v1892 = stablehlo.reshape %v1745 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1893 = stablehlo.multiply %v1892, %v1891 : tensor<32x256x14x14xf32>
    %v1894 = stablehlo.reduce(%v1893 init: %v1877) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1895 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v1896 = stablehlo.multiply %v1894, %v1895 : tensor<256xf32>
    %v1897 = stablehlo.subtract %s3b3g2, %v1896 : tensor<256xf32>
    %v1898 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1899 = stablehlo.reshape %v1745 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1900 = stablehlo.reduce(%v1899 init: %v1898) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1901 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v1902 = stablehlo.multiply %v1900, %v1901 : tensor<256xf32>
    %v1903 = stablehlo.subtract %s3b3bt2, %v1902 : tensor<256xf32>
    %v1904 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1905 = stablehlo.compare GT, %v683, %v1904 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v1906 = stablehlo.select %v1905, %v1819, %v1904 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v1907 = stablehlo.reshape %v1906 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1908 = stablehlo.reshape %v662 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1909 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1910 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v1911 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1912 = stablehlo.reduce(%v1908 init: %v1909) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1913 = stablehlo.broadcast_in_dim %v1912, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1914 = stablehlo.divide %v1913, %v1910 : tensor<32x256x14x14xf32>
    %v1915 = stablehlo.subtract %v1908, %v1914 : tensor<32x256x14x14xf32>
    %v1916 = stablehlo.multiply %v1915, %v1915 : tensor<32x256x14x14xf32>
    %v1917 = stablehlo.reduce(%v1916 init: %v1909) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1918 = stablehlo.broadcast_in_dim %v1917, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1919 = stablehlo.divide %v1918, %v1910 : tensor<32x256x14x14xf32>
    %v1920 = stablehlo.add %v1919, %v1911 : tensor<32x256x14x14xf32>
    %v1921 = stablehlo.rsqrt %v1920 : tensor<32x256x14x14xf32>
    %v1922 = stablehlo.multiply %v1915, %v1921 : tensor<32x256x14x14xf32>
    %v1923 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1924 = stablehlo.multiply %v1923, %v1907 : tensor<32x256x14x14xf32>
    %v1925 = stablehlo.reduce(%v1924 init: %v1909) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1926 = stablehlo.broadcast_in_dim %v1925, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1927 = stablehlo.multiply %v1922, %v1924 : tensor<32x256x14x14xf32>
    %v1928 = stablehlo.reduce(%v1927 init: %v1909) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1929 = stablehlo.broadcast_in_dim %v1928, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1930 = stablehlo.multiply %v1924, %v1910 : tensor<32x256x14x14xf32>
    %v1931 = stablehlo.subtract %v1930, %v1926 : tensor<32x256x14x14xf32>
    %v1932 = stablehlo.multiply %v1922, %v1929 : tensor<32x256x14x14xf32>
    %v1933 = stablehlo.subtract %v1931, %v1932 : tensor<32x256x14x14xf32>
    %v1934 = stablehlo.divide %v1921, %v1910 : tensor<32x256x14x14xf32>
    %v1935 = stablehlo.multiply %v1934, %v1933 : tensor<32x256x14x14xf32>
    %v1936 = stablehlo.reshape %v1935 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1937 = stablehlo.reshape %v1936 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1938 = stablehlo.transpose %s3b2W2, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1939 = stablehlo.reverse %v1938, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1940 = stablehlo.convolution(%v1937, %v1939)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1941 = stablehlo.reshape %v1940 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1942 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1943 = stablehlo.compare GT, %v655, %v1942 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v1944 = stablehlo.select %v1943, %v1941, %v1942 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v1945 = stablehlo.reshape %v1944 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1946 = stablehlo.reshape %v635 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1947 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1948 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v1949 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1950 = stablehlo.reduce(%v1946 init: %v1947) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1951 = stablehlo.broadcast_in_dim %v1950, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1952 = stablehlo.divide %v1951, %v1948 : tensor<32x256x14x14xf32>
    %v1953 = stablehlo.subtract %v1946, %v1952 : tensor<32x256x14x14xf32>
    %v1954 = stablehlo.multiply %v1953, %v1953 : tensor<32x256x14x14xf32>
    %v1955 = stablehlo.reduce(%v1954 init: %v1947) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1956 = stablehlo.broadcast_in_dim %v1955, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1957 = stablehlo.divide %v1956, %v1948 : tensor<32x256x14x14xf32>
    %v1958 = stablehlo.add %v1957, %v1949 : tensor<32x256x14x14xf32>
    %v1959 = stablehlo.rsqrt %v1958 : tensor<32x256x14x14xf32>
    %v1960 = stablehlo.multiply %v1953, %v1959 : tensor<32x256x14x14xf32>
    %v1961 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1962 = stablehlo.multiply %v1961, %v1945 : tensor<32x256x14x14xf32>
    %v1963 = stablehlo.reduce(%v1962 init: %v1947) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1964 = stablehlo.broadcast_in_dim %v1963, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1965 = stablehlo.multiply %v1960, %v1962 : tensor<32x256x14x14xf32>
    %v1966 = stablehlo.reduce(%v1965 init: %v1947) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v1967 = stablehlo.broadcast_in_dim %v1966, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v1968 = stablehlo.multiply %v1962, %v1948 : tensor<32x256x14x14xf32>
    %v1969 = stablehlo.subtract %v1968, %v1964 : tensor<32x256x14x14xf32>
    %v1970 = stablehlo.multiply %v1960, %v1967 : tensor<32x256x14x14xf32>
    %v1971 = stablehlo.subtract %v1969, %v1970 : tensor<32x256x14x14xf32>
    %v1972 = stablehlo.divide %v1959, %v1948 : tensor<32x256x14x14xf32>
    %v1973 = stablehlo.multiply %v1972, %v1971 : tensor<32x256x14x14xf32>
    %v1974 = stablehlo.reshape %v1973 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1975 = stablehlo.reshape %v1974 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1976 = stablehlo.transpose %s3b2W1, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1977 = stablehlo.reverse %v1976, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1978 = stablehlo.convolution(%v1975, %v1977)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1979 = stablehlo.reshape %v1978 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1980 = stablehlo.add %v1979, %v1906 : tensor<32x50176xf32>
    %v1981 = stablehlo.reshape %v630 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1982 = stablehlo.reshape %v1974 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1983 = stablehlo.transpose %v1981, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1984 = stablehlo.transpose %v1982, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1985 = stablehlo.convolution(%v1983, %v1984)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1986 = stablehlo.transpose %v1985, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1987 = stablehlo.constant dense<0.003125> : tensor<256x256x3x3xf32>
    %v1988 = stablehlo.multiply %v1986, %v1987 : tensor<256x256x3x3xf32>
    %v1989 = stablehlo.subtract %s3b2W1, %v1988 : tensor<256x256x3x3xf32>
    %v1990 = stablehlo.reshape %v1974 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1991 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1992 = stablehlo.reduce(%v1990 init: %v1991) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1993 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v1994 = stablehlo.multiply %v1992, %v1993 : tensor<256xf32>
    %v1995 = stablehlo.subtract %s3b2b1, %v1994 : tensor<256xf32>
    %v1996 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1997 = stablehlo.reshape %v635 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1998 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v1999 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2000 = stablehlo.reduce(%v1997 init: %v1996) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2001 = stablehlo.broadcast_in_dim %v2000, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2002 = stablehlo.divide %v2001, %v1998 : tensor<32x256x14x14xf32>
    %v2003 = stablehlo.subtract %v1997, %v2002 : tensor<32x256x14x14xf32>
    %v2004 = stablehlo.multiply %v2003, %v2003 : tensor<32x256x14x14xf32>
    %v2005 = stablehlo.reduce(%v2004 init: %v1996) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2006 = stablehlo.broadcast_in_dim %v2005, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2007 = stablehlo.divide %v2006, %v1998 : tensor<32x256x14x14xf32>
    %v2008 = stablehlo.add %v2007, %v1999 : tensor<32x256x14x14xf32>
    %v2009 = stablehlo.rsqrt %v2008 : tensor<32x256x14x14xf32>
    %v2010 = stablehlo.multiply %v2003, %v2009 : tensor<32x256x14x14xf32>
    %v2011 = stablehlo.reshape %v1944 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2012 = stablehlo.multiply %v2011, %v2010 : tensor<32x256x14x14xf32>
    %v2013 = stablehlo.reduce(%v2012 init: %v1996) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2014 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2015 = stablehlo.multiply %v2013, %v2014 : tensor<256xf32>
    %v2016 = stablehlo.subtract %s3b2g1, %v2015 : tensor<256xf32>
    %v2017 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2018 = stablehlo.reshape %v1944 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2019 = stablehlo.reduce(%v2018 init: %v2017) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2020 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2021 = stablehlo.multiply %v2019, %v2020 : tensor<256xf32>
    %v2022 = stablehlo.subtract %s3b2bt1, %v2021 : tensor<256xf32>
    %v2023 = stablehlo.reshape %v657 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2024 = stablehlo.reshape %v1936 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2025 = stablehlo.transpose %v2023, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2026 = stablehlo.transpose %v2024, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2027 = stablehlo.convolution(%v2025, %v2026)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2028 = stablehlo.transpose %v2027, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2029 = stablehlo.constant dense<0.003125> : tensor<256x256x3x3xf32>
    %v2030 = stablehlo.multiply %v2028, %v2029 : tensor<256x256x3x3xf32>
    %v2031 = stablehlo.subtract %s3b2W2, %v2030 : tensor<256x256x3x3xf32>
    %v2032 = stablehlo.reshape %v1936 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2033 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2034 = stablehlo.reduce(%v2032 init: %v2033) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2035 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2036 = stablehlo.multiply %v2034, %v2035 : tensor<256xf32>
    %v2037 = stablehlo.subtract %s3b2b2, %v2036 : tensor<256xf32>
    %v2038 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2039 = stablehlo.reshape %v662 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2040 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2041 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2042 = stablehlo.reduce(%v2039 init: %v2038) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2043 = stablehlo.broadcast_in_dim %v2042, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2044 = stablehlo.divide %v2043, %v2040 : tensor<32x256x14x14xf32>
    %v2045 = stablehlo.subtract %v2039, %v2044 : tensor<32x256x14x14xf32>
    %v2046 = stablehlo.multiply %v2045, %v2045 : tensor<32x256x14x14xf32>
    %v2047 = stablehlo.reduce(%v2046 init: %v2038) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2048 = stablehlo.broadcast_in_dim %v2047, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2049 = stablehlo.divide %v2048, %v2040 : tensor<32x256x14x14xf32>
    %v2050 = stablehlo.add %v2049, %v2041 : tensor<32x256x14x14xf32>
    %v2051 = stablehlo.rsqrt %v2050 : tensor<32x256x14x14xf32>
    %v2052 = stablehlo.multiply %v2045, %v2051 : tensor<32x256x14x14xf32>
    %v2053 = stablehlo.reshape %v1906 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2054 = stablehlo.multiply %v2053, %v2052 : tensor<32x256x14x14xf32>
    %v2055 = stablehlo.reduce(%v2054 init: %v2038) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2056 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2057 = stablehlo.multiply %v2055, %v2056 : tensor<256xf32>
    %v2058 = stablehlo.subtract %s3b2g2, %v2057 : tensor<256xf32>
    %v2059 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2060 = stablehlo.reshape %v1906 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2061 = stablehlo.reduce(%v2060 init: %v2059) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2062 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2063 = stablehlo.multiply %v2061, %v2062 : tensor<256xf32>
    %v2064 = stablehlo.subtract %s3b2bt2, %v2063 : tensor<256xf32>
    %v2065 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v2066 = stablehlo.compare GT, %v628, %v2065 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v2067 = stablehlo.select %v2066, %v1980, %v2065 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v2068 = stablehlo.reshape %v2067 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2069 = stablehlo.reshape %v607 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2070 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2071 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2072 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2073 = stablehlo.reduce(%v2069 init: %v2070) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2074 = stablehlo.broadcast_in_dim %v2073, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2075 = stablehlo.divide %v2074, %v2071 : tensor<32x256x14x14xf32>
    %v2076 = stablehlo.subtract %v2069, %v2075 : tensor<32x256x14x14xf32>
    %v2077 = stablehlo.multiply %v2076, %v2076 : tensor<32x256x14x14xf32>
    %v2078 = stablehlo.reduce(%v2077 init: %v2070) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2079 = stablehlo.broadcast_in_dim %v2078, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2080 = stablehlo.divide %v2079, %v2071 : tensor<32x256x14x14xf32>
    %v2081 = stablehlo.add %v2080, %v2072 : tensor<32x256x14x14xf32>
    %v2082 = stablehlo.rsqrt %v2081 : tensor<32x256x14x14xf32>
    %v2083 = stablehlo.multiply %v2076, %v2082 : tensor<32x256x14x14xf32>
    %v2084 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2085 = stablehlo.multiply %v2084, %v2068 : tensor<32x256x14x14xf32>
    %v2086 = stablehlo.reduce(%v2085 init: %v2070) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2087 = stablehlo.broadcast_in_dim %v2086, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2088 = stablehlo.multiply %v2083, %v2085 : tensor<32x256x14x14xf32>
    %v2089 = stablehlo.reduce(%v2088 init: %v2070) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2090 = stablehlo.broadcast_in_dim %v2089, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2091 = stablehlo.multiply %v2085, %v2071 : tensor<32x256x14x14xf32>
    %v2092 = stablehlo.subtract %v2091, %v2087 : tensor<32x256x14x14xf32>
    %v2093 = stablehlo.multiply %v2083, %v2090 : tensor<32x256x14x14xf32>
    %v2094 = stablehlo.subtract %v2092, %v2093 : tensor<32x256x14x14xf32>
    %v2095 = stablehlo.divide %v2082, %v2071 : tensor<32x256x14x14xf32>
    %v2096 = stablehlo.multiply %v2095, %v2094 : tensor<32x256x14x14xf32>
    %v2097 = stablehlo.reshape %v2096 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2098 = stablehlo.reshape %v2097 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2099 = stablehlo.transpose %s3b1W2, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2100 = stablehlo.reverse %v2099, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2101 = stablehlo.convolution(%v2098, %v2100)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2102 = stablehlo.reshape %v2101 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2103 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v2104 = stablehlo.compare GT, %v600, %v2103 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v2105 = stablehlo.select %v2104, %v2102, %v2103 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v2106 = stablehlo.reshape %v2105 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2107 = stablehlo.reshape %v580 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2108 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2109 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2110 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2111 = stablehlo.reduce(%v2107 init: %v2108) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2112 = stablehlo.broadcast_in_dim %v2111, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2113 = stablehlo.divide %v2112, %v2109 : tensor<32x256x14x14xf32>
    %v2114 = stablehlo.subtract %v2107, %v2113 : tensor<32x256x14x14xf32>
    %v2115 = stablehlo.multiply %v2114, %v2114 : tensor<32x256x14x14xf32>
    %v2116 = stablehlo.reduce(%v2115 init: %v2108) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2117 = stablehlo.broadcast_in_dim %v2116, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2118 = stablehlo.divide %v2117, %v2109 : tensor<32x256x14x14xf32>
    %v2119 = stablehlo.add %v2118, %v2110 : tensor<32x256x14x14xf32>
    %v2120 = stablehlo.rsqrt %v2119 : tensor<32x256x14x14xf32>
    %v2121 = stablehlo.multiply %v2114, %v2120 : tensor<32x256x14x14xf32>
    %v2122 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2123 = stablehlo.multiply %v2122, %v2106 : tensor<32x256x14x14xf32>
    %v2124 = stablehlo.reduce(%v2123 init: %v2108) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2125 = stablehlo.broadcast_in_dim %v2124, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2126 = stablehlo.multiply %v2121, %v2123 : tensor<32x256x14x14xf32>
    %v2127 = stablehlo.reduce(%v2126 init: %v2108) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2128 = stablehlo.broadcast_in_dim %v2127, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2129 = stablehlo.multiply %v2123, %v2109 : tensor<32x256x14x14xf32>
    %v2130 = stablehlo.subtract %v2129, %v2125 : tensor<32x256x14x14xf32>
    %v2131 = stablehlo.multiply %v2121, %v2128 : tensor<32x256x14x14xf32>
    %v2132 = stablehlo.subtract %v2130, %v2131 : tensor<32x256x14x14xf32>
    %v2133 = stablehlo.divide %v2120, %v2109 : tensor<32x256x14x14xf32>
    %v2134 = stablehlo.multiply %v2133, %v2132 : tensor<32x256x14x14xf32>
    %v2135 = stablehlo.reshape %v2134 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2136 = stablehlo.reshape %v2135 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2137 = stablehlo.transpose %s3b1W1, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2138 = stablehlo.reverse %v2137, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2139 = stablehlo.convolution(%v2136, %v2138)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2140 = stablehlo.reshape %v2139 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2141 = stablehlo.add %v2140, %v2067 : tensor<32x50176xf32>
    %v2142 = stablehlo.reshape %v575 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2143 = stablehlo.reshape %v2135 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2144 = stablehlo.transpose %v2142, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2145 = stablehlo.transpose %v2143, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2146 = stablehlo.convolution(%v2144, %v2145)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2147 = stablehlo.transpose %v2146, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2148 = stablehlo.constant dense<0.003125> : tensor<256x256x3x3xf32>
    %v2149 = stablehlo.multiply %v2147, %v2148 : tensor<256x256x3x3xf32>
    %v2150 = stablehlo.subtract %s3b1W1, %v2149 : tensor<256x256x3x3xf32>
    %v2151 = stablehlo.reshape %v2135 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2152 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2153 = stablehlo.reduce(%v2151 init: %v2152) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2154 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2155 = stablehlo.multiply %v2153, %v2154 : tensor<256xf32>
    %v2156 = stablehlo.subtract %s3b1b1, %v2155 : tensor<256xf32>
    %v2157 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2158 = stablehlo.reshape %v580 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2159 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2160 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2161 = stablehlo.reduce(%v2158 init: %v2157) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2162 = stablehlo.broadcast_in_dim %v2161, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2163 = stablehlo.divide %v2162, %v2159 : tensor<32x256x14x14xf32>
    %v2164 = stablehlo.subtract %v2158, %v2163 : tensor<32x256x14x14xf32>
    %v2165 = stablehlo.multiply %v2164, %v2164 : tensor<32x256x14x14xf32>
    %v2166 = stablehlo.reduce(%v2165 init: %v2157) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2167 = stablehlo.broadcast_in_dim %v2166, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2168 = stablehlo.divide %v2167, %v2159 : tensor<32x256x14x14xf32>
    %v2169 = stablehlo.add %v2168, %v2160 : tensor<32x256x14x14xf32>
    %v2170 = stablehlo.rsqrt %v2169 : tensor<32x256x14x14xf32>
    %v2171 = stablehlo.multiply %v2164, %v2170 : tensor<32x256x14x14xf32>
    %v2172 = stablehlo.reshape %v2105 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2173 = stablehlo.multiply %v2172, %v2171 : tensor<32x256x14x14xf32>
    %v2174 = stablehlo.reduce(%v2173 init: %v2157) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2175 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2176 = stablehlo.multiply %v2174, %v2175 : tensor<256xf32>
    %v2177 = stablehlo.subtract %s3b1g1, %v2176 : tensor<256xf32>
    %v2178 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2179 = stablehlo.reshape %v2105 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2180 = stablehlo.reduce(%v2179 init: %v2178) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2181 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2182 = stablehlo.multiply %v2180, %v2181 : tensor<256xf32>
    %v2183 = stablehlo.subtract %s3b1bt1, %v2182 : tensor<256xf32>
    %v2184 = stablehlo.reshape %v602 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2185 = stablehlo.reshape %v2097 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2186 = stablehlo.transpose %v2184, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2187 = stablehlo.transpose %v2185, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2188 = stablehlo.convolution(%v2186, %v2187)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2189 = stablehlo.transpose %v2188, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2190 = stablehlo.constant dense<0.003125> : tensor<256x256x3x3xf32>
    %v2191 = stablehlo.multiply %v2189, %v2190 : tensor<256x256x3x3xf32>
    %v2192 = stablehlo.subtract %s3b1W2, %v2191 : tensor<256x256x3x3xf32>
    %v2193 = stablehlo.reshape %v2097 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2194 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2195 = stablehlo.reduce(%v2193 init: %v2194) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2196 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2197 = stablehlo.multiply %v2195, %v2196 : tensor<256xf32>
    %v2198 = stablehlo.subtract %s3b1b2, %v2197 : tensor<256xf32>
    %v2199 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2200 = stablehlo.reshape %v607 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2201 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2202 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2203 = stablehlo.reduce(%v2200 init: %v2199) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2204 = stablehlo.broadcast_in_dim %v2203, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2205 = stablehlo.divide %v2204, %v2201 : tensor<32x256x14x14xf32>
    %v2206 = stablehlo.subtract %v2200, %v2205 : tensor<32x256x14x14xf32>
    %v2207 = stablehlo.multiply %v2206, %v2206 : tensor<32x256x14x14xf32>
    %v2208 = stablehlo.reduce(%v2207 init: %v2199) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2209 = stablehlo.broadcast_in_dim %v2208, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2210 = stablehlo.divide %v2209, %v2201 : tensor<32x256x14x14xf32>
    %v2211 = stablehlo.add %v2210, %v2202 : tensor<32x256x14x14xf32>
    %v2212 = stablehlo.rsqrt %v2211 : tensor<32x256x14x14xf32>
    %v2213 = stablehlo.multiply %v2206, %v2212 : tensor<32x256x14x14xf32>
    %v2214 = stablehlo.reshape %v2067 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2215 = stablehlo.multiply %v2214, %v2213 : tensor<32x256x14x14xf32>
    %v2216 = stablehlo.reduce(%v2215 init: %v2199) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2217 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2218 = stablehlo.multiply %v2216, %v2217 : tensor<256xf32>
    %v2219 = stablehlo.subtract %s3b1g2, %v2218 : tensor<256xf32>
    %v2220 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2221 = stablehlo.reshape %v2067 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2222 = stablehlo.reduce(%v2221 init: %v2220) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2223 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2224 = stablehlo.multiply %v2222, %v2223 : tensor<256xf32>
    %v2225 = stablehlo.subtract %s3b1bt2, %v2224 : tensor<256xf32>
    %v2226 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v2227 = stablehlo.compare GT, %v573, %v2226 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v2228 = stablehlo.select %v2227, %v2141, %v2226 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v2229 = stablehlo.reshape %v2228 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2230 = stablehlo.reshape %v552 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2231 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2232 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2233 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2234 = stablehlo.reduce(%v2230 init: %v2231) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2235 = stablehlo.broadcast_in_dim %v2234, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2236 = stablehlo.divide %v2235, %v2232 : tensor<32x256x14x14xf32>
    %v2237 = stablehlo.subtract %v2230, %v2236 : tensor<32x256x14x14xf32>
    %v2238 = stablehlo.multiply %v2237, %v2237 : tensor<32x256x14x14xf32>
    %v2239 = stablehlo.reduce(%v2238 init: %v2231) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2240 = stablehlo.broadcast_in_dim %v2239, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2241 = stablehlo.divide %v2240, %v2232 : tensor<32x256x14x14xf32>
    %v2242 = stablehlo.add %v2241, %v2233 : tensor<32x256x14x14xf32>
    %v2243 = stablehlo.rsqrt %v2242 : tensor<32x256x14x14xf32>
    %v2244 = stablehlo.multiply %v2237, %v2243 : tensor<32x256x14x14xf32>
    %v2245 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2246 = stablehlo.multiply %v2245, %v2229 : tensor<32x256x14x14xf32>
    %v2247 = stablehlo.reduce(%v2246 init: %v2231) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2248 = stablehlo.broadcast_in_dim %v2247, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2249 = stablehlo.multiply %v2244, %v2246 : tensor<32x256x14x14xf32>
    %v2250 = stablehlo.reduce(%v2249 init: %v2231) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2251 = stablehlo.broadcast_in_dim %v2250, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2252 = stablehlo.multiply %v2246, %v2232 : tensor<32x256x14x14xf32>
    %v2253 = stablehlo.subtract %v2252, %v2248 : tensor<32x256x14x14xf32>
    %v2254 = stablehlo.multiply %v2244, %v2251 : tensor<32x256x14x14xf32>
    %v2255 = stablehlo.subtract %v2253, %v2254 : tensor<32x256x14x14xf32>
    %v2256 = stablehlo.divide %v2243, %v2232 : tensor<32x256x14x14xf32>
    %v2257 = stablehlo.multiply %v2256, %v2255 : tensor<32x256x14x14xf32>
    %v2258 = stablehlo.reshape %v2257 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2259 = stablehlo.reshape %v2258 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2260 = stablehlo.transpose %s3b0W2, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2261 = stablehlo.reverse %v2260, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2262 = stablehlo.convolution(%v2259, %v2261)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2263 = stablehlo.reshape %v2262 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2264 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v2265 = stablehlo.compare GT, %v545, %v2264 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v2266 = stablehlo.select %v2265, %v2263, %v2264 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v2267 = stablehlo.reshape %v2266 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2268 = stablehlo.reshape %v525 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2269 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2270 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2271 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2272 = stablehlo.reduce(%v2268 init: %v2269) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2273 = stablehlo.broadcast_in_dim %v2272, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2274 = stablehlo.divide %v2273, %v2270 : tensor<32x256x14x14xf32>
    %v2275 = stablehlo.subtract %v2268, %v2274 : tensor<32x256x14x14xf32>
    %v2276 = stablehlo.multiply %v2275, %v2275 : tensor<32x256x14x14xf32>
    %v2277 = stablehlo.reduce(%v2276 init: %v2269) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2278 = stablehlo.broadcast_in_dim %v2277, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2279 = stablehlo.divide %v2278, %v2270 : tensor<32x256x14x14xf32>
    %v2280 = stablehlo.add %v2279, %v2271 : tensor<32x256x14x14xf32>
    %v2281 = stablehlo.rsqrt %v2280 : tensor<32x256x14x14xf32>
    %v2282 = stablehlo.multiply %v2275, %v2281 : tensor<32x256x14x14xf32>
    %v2283 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2284 = stablehlo.multiply %v2283, %v2267 : tensor<32x256x14x14xf32>
    %v2285 = stablehlo.reduce(%v2284 init: %v2269) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2286 = stablehlo.broadcast_in_dim %v2285, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2287 = stablehlo.multiply %v2282, %v2284 : tensor<32x256x14x14xf32>
    %v2288 = stablehlo.reduce(%v2287 init: %v2269) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2289 = stablehlo.broadcast_in_dim %v2288, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2290 = stablehlo.multiply %v2284, %v2270 : tensor<32x256x14x14xf32>
    %v2291 = stablehlo.subtract %v2290, %v2286 : tensor<32x256x14x14xf32>
    %v2292 = stablehlo.multiply %v2282, %v2289 : tensor<32x256x14x14xf32>
    %v2293 = stablehlo.subtract %v2291, %v2292 : tensor<32x256x14x14xf32>
    %v2294 = stablehlo.divide %v2281, %v2270 : tensor<32x256x14x14xf32>
    %v2295 = stablehlo.multiply %v2294, %v2293 : tensor<32x256x14x14xf32>
    %v2296 = stablehlo.reshape %v2295 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2297 = stablehlo.reshape %v2296 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2298 = stablehlo.transpose %s3b0W1, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2299 = stablehlo.reverse %v2298, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2300 = stablehlo.convolution(%v2297, %v2299)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2301 = stablehlo.reshape %v2300 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2302 = stablehlo.add %v2301, %v2228 : tensor<32x50176xf32>
    %v2303 = stablehlo.reshape %v520 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2304 = stablehlo.reshape %v2296 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2305 = stablehlo.transpose %v2303, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2306 = stablehlo.transpose %v2304, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2307 = stablehlo.convolution(%v2305, %v2306)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2308 = stablehlo.transpose %v2307, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2309 = stablehlo.constant dense<0.003125> : tensor<256x256x3x3xf32>
    %v2310 = stablehlo.multiply %v2308, %v2309 : tensor<256x256x3x3xf32>
    %v2311 = stablehlo.subtract %s3b0W1, %v2310 : tensor<256x256x3x3xf32>
    %v2312 = stablehlo.reshape %v2296 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2313 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2314 = stablehlo.reduce(%v2312 init: %v2313) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2315 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2316 = stablehlo.multiply %v2314, %v2315 : tensor<256xf32>
    %v2317 = stablehlo.subtract %s3b0b1, %v2316 : tensor<256xf32>
    %v2318 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2319 = stablehlo.reshape %v525 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2320 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2321 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2322 = stablehlo.reduce(%v2319 init: %v2318) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2323 = stablehlo.broadcast_in_dim %v2322, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2324 = stablehlo.divide %v2323, %v2320 : tensor<32x256x14x14xf32>
    %v2325 = stablehlo.subtract %v2319, %v2324 : tensor<32x256x14x14xf32>
    %v2326 = stablehlo.multiply %v2325, %v2325 : tensor<32x256x14x14xf32>
    %v2327 = stablehlo.reduce(%v2326 init: %v2318) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2328 = stablehlo.broadcast_in_dim %v2327, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2329 = stablehlo.divide %v2328, %v2320 : tensor<32x256x14x14xf32>
    %v2330 = stablehlo.add %v2329, %v2321 : tensor<32x256x14x14xf32>
    %v2331 = stablehlo.rsqrt %v2330 : tensor<32x256x14x14xf32>
    %v2332 = stablehlo.multiply %v2325, %v2331 : tensor<32x256x14x14xf32>
    %v2333 = stablehlo.reshape %v2266 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2334 = stablehlo.multiply %v2333, %v2332 : tensor<32x256x14x14xf32>
    %v2335 = stablehlo.reduce(%v2334 init: %v2318) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2336 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2337 = stablehlo.multiply %v2335, %v2336 : tensor<256xf32>
    %v2338 = stablehlo.subtract %s3b0g1, %v2337 : tensor<256xf32>
    %v2339 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2340 = stablehlo.reshape %v2266 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2341 = stablehlo.reduce(%v2340 init: %v2339) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2342 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2343 = stablehlo.multiply %v2341, %v2342 : tensor<256xf32>
    %v2344 = stablehlo.subtract %s3b0bt1, %v2343 : tensor<256xf32>
    %v2345 = stablehlo.reshape %v547 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2346 = stablehlo.reshape %v2258 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2347 = stablehlo.transpose %v2345, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2348 = stablehlo.transpose %v2346, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2349 = stablehlo.convolution(%v2347, %v2348)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2350 = stablehlo.transpose %v2349, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2351 = stablehlo.constant dense<0.003125> : tensor<256x256x3x3xf32>
    %v2352 = stablehlo.multiply %v2350, %v2351 : tensor<256x256x3x3xf32>
    %v2353 = stablehlo.subtract %s3b0W2, %v2352 : tensor<256x256x3x3xf32>
    %v2354 = stablehlo.reshape %v2258 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2355 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2356 = stablehlo.reduce(%v2354 init: %v2355) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2357 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2358 = stablehlo.multiply %v2356, %v2357 : tensor<256xf32>
    %v2359 = stablehlo.subtract %s3b0b2, %v2358 : tensor<256xf32>
    %v2360 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2361 = stablehlo.reshape %v552 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2362 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2363 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2364 = stablehlo.reduce(%v2361 init: %v2360) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2365 = stablehlo.broadcast_in_dim %v2364, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2366 = stablehlo.divide %v2365, %v2362 : tensor<32x256x14x14xf32>
    %v2367 = stablehlo.subtract %v2361, %v2366 : tensor<32x256x14x14xf32>
    %v2368 = stablehlo.multiply %v2367, %v2367 : tensor<32x256x14x14xf32>
    %v2369 = stablehlo.reduce(%v2368 init: %v2360) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2370 = stablehlo.broadcast_in_dim %v2369, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2371 = stablehlo.divide %v2370, %v2362 : tensor<32x256x14x14xf32>
    %v2372 = stablehlo.add %v2371, %v2363 : tensor<32x256x14x14xf32>
    %v2373 = stablehlo.rsqrt %v2372 : tensor<32x256x14x14xf32>
    %v2374 = stablehlo.multiply %v2367, %v2373 : tensor<32x256x14x14xf32>
    %v2375 = stablehlo.reshape %v2228 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2376 = stablehlo.multiply %v2375, %v2374 : tensor<32x256x14x14xf32>
    %v2377 = stablehlo.reduce(%v2376 init: %v2360) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2378 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2379 = stablehlo.multiply %v2377, %v2378 : tensor<256xf32>
    %v2380 = stablehlo.subtract %s3b0g2, %v2379 : tensor<256xf32>
    %v2381 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2382 = stablehlo.reshape %v2228 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2383 = stablehlo.reduce(%v2382 init: %v2381) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2384 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2385 = stablehlo.multiply %v2383, %v2384 : tensor<256xf32>
    %v2386 = stablehlo.subtract %s3b0bt2, %v2385 : tensor<256xf32>
    %v2387 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v2388 = stablehlo.compare GT, %v518, %v2387 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v2389 = stablehlo.select %v2388, %v2302, %v2387 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v2390 = stablehlo.reshape %v2389 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2391 = stablehlo.reshape %v472 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2392 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2393 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2394 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2395 = stablehlo.reduce(%v2391 init: %v2392) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2396 = stablehlo.broadcast_in_dim %v2395, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2397 = stablehlo.divide %v2396, %v2393 : tensor<32x256x14x14xf32>
    %v2398 = stablehlo.subtract %v2391, %v2397 : tensor<32x256x14x14xf32>
    %v2399 = stablehlo.multiply %v2398, %v2398 : tensor<32x256x14x14xf32>
    %v2400 = stablehlo.reduce(%v2399 init: %v2392) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2401 = stablehlo.broadcast_in_dim %v2400, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2402 = stablehlo.divide %v2401, %v2393 : tensor<32x256x14x14xf32>
    %v2403 = stablehlo.add %v2402, %v2394 : tensor<32x256x14x14xf32>
    %v2404 = stablehlo.rsqrt %v2403 : tensor<32x256x14x14xf32>
    %v2405 = stablehlo.multiply %v2398, %v2404 : tensor<32x256x14x14xf32>
    %v2406 = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2407 = stablehlo.multiply %v2406, %v2390 : tensor<32x256x14x14xf32>
    %v2408 = stablehlo.reduce(%v2407 init: %v2392) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2409 = stablehlo.broadcast_in_dim %v2408, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2410 = stablehlo.multiply %v2405, %v2407 : tensor<32x256x14x14xf32>
    %v2411 = stablehlo.reduce(%v2410 init: %v2392) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2412 = stablehlo.broadcast_in_dim %v2411, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2413 = stablehlo.multiply %v2407, %v2393 : tensor<32x256x14x14xf32>
    %v2414 = stablehlo.subtract %v2413, %v2409 : tensor<32x256x14x14xf32>
    %v2415 = stablehlo.multiply %v2405, %v2412 : tensor<32x256x14x14xf32>
    %v2416 = stablehlo.subtract %v2414, %v2415 : tensor<32x256x14x14xf32>
    %v2417 = stablehlo.divide %v2404, %v2393 : tensor<32x256x14x14xf32>
    %v2418 = stablehlo.multiply %v2417, %v2416 : tensor<32x256x14x14xf32>
    %v2419 = stablehlo.reshape %v2418 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2420 = stablehlo.reshape %v2419 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2421 = stablehlo.transpose %d3W2, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2422 = stablehlo.reverse %v2421, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2423 = stablehlo.convolution(%v2420, %v2422)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2424 = stablehlo.reshape %v2423 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2425 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v2426 = stablehlo.compare GT, %v465, %v2425 : (tensor<32x50176xf32>, tensor<32x50176xf32>) -> tensor<32x50176xi1>
    %v2427 = stablehlo.select %v2426, %v2424, %v2425 : tensor<32x50176xi1>, tensor<32x50176xf32>
    %v2428 = stablehlo.reshape %v2427 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2429 = stablehlo.reshape %v445 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2430 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2431 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2432 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2433 = stablehlo.reduce(%v2429 init: %v2430) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2434 = stablehlo.broadcast_in_dim %v2433, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2435 = stablehlo.divide %v2434, %v2431 : tensor<32x256x14x14xf32>
    %v2436 = stablehlo.subtract %v2429, %v2435 : tensor<32x256x14x14xf32>
    %v2437 = stablehlo.multiply %v2436, %v2436 : tensor<32x256x14x14xf32>
    %v2438 = stablehlo.reduce(%v2437 init: %v2430) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2439 = stablehlo.broadcast_in_dim %v2438, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2440 = stablehlo.divide %v2439, %v2431 : tensor<32x256x14x14xf32>
    %v2441 = stablehlo.add %v2440, %v2432 : tensor<32x256x14x14xf32>
    %v2442 = stablehlo.rsqrt %v2441 : tensor<32x256x14x14xf32>
    %v2443 = stablehlo.multiply %v2436, %v2442 : tensor<32x256x14x14xf32>
    %v2444 = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2445 = stablehlo.multiply %v2444, %v2428 : tensor<32x256x14x14xf32>
    %v2446 = stablehlo.reduce(%v2445 init: %v2430) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2447 = stablehlo.broadcast_in_dim %v2446, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2448 = stablehlo.multiply %v2443, %v2445 : tensor<32x256x14x14xf32>
    %v2449 = stablehlo.reduce(%v2448 init: %v2430) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2450 = stablehlo.broadcast_in_dim %v2449, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2451 = stablehlo.multiply %v2445, %v2431 : tensor<32x256x14x14xf32>
    %v2452 = stablehlo.subtract %v2451, %v2447 : tensor<32x256x14x14xf32>
    %v2453 = stablehlo.multiply %v2443, %v2450 : tensor<32x256x14x14xf32>
    %v2454 = stablehlo.subtract %v2452, %v2453 : tensor<32x256x14x14xf32>
    %v2455 = stablehlo.divide %v2442, %v2431 : tensor<32x256x14x14xf32>
    %v2456 = stablehlo.multiply %v2455, %v2454 : tensor<32x256x14x14xf32>
    %v2457 = stablehlo.reshape %v2456 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2458 = stablehlo.reshape %v2457 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2459 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2460 = stablehlo.pad %v2458, %v2459, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %v2461 = stablehlo.transpose %d3W1, dims = [1, 0, 2, 3] : (tensor<256x128x3x3xf32>) -> tensor<128x256x3x3xf32>
    %v2462 = stablehlo.reverse %v2461, dims = [2, 3] : tensor<128x256x3x3xf32>
    %v2463 = stablehlo.convolution(%v2460, %v2462)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x28x28xf32>, tensor<128x256x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2464 = stablehlo.reshape %v2463 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2465 = stablehlo.reshape %v2389 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2466 = stablehlo.reshape %v497 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2467 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2468 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2469 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2470 = stablehlo.reduce(%v2466 init: %v2467) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2471 = stablehlo.broadcast_in_dim %v2470, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2472 = stablehlo.divide %v2471, %v2468 : tensor<32x256x14x14xf32>
    %v2473 = stablehlo.subtract %v2466, %v2472 : tensor<32x256x14x14xf32>
    %v2474 = stablehlo.multiply %v2473, %v2473 : tensor<32x256x14x14xf32>
    %v2475 = stablehlo.reduce(%v2474 init: %v2467) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2476 = stablehlo.broadcast_in_dim %v2475, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2477 = stablehlo.divide %v2476, %v2468 : tensor<32x256x14x14xf32>
    %v2478 = stablehlo.add %v2477, %v2469 : tensor<32x256x14x14xf32>
    %v2479 = stablehlo.rsqrt %v2478 : tensor<32x256x14x14xf32>
    %v2480 = stablehlo.multiply %v2473, %v2479 : tensor<32x256x14x14xf32>
    %v2481 = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2482 = stablehlo.multiply %v2481, %v2465 : tensor<32x256x14x14xf32>
    %v2483 = stablehlo.reduce(%v2482 init: %v2467) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2484 = stablehlo.broadcast_in_dim %v2483, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2485 = stablehlo.multiply %v2480, %v2482 : tensor<32x256x14x14xf32>
    %v2486 = stablehlo.reduce(%v2485 init: %v2467) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2487 = stablehlo.broadcast_in_dim %v2486, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2488 = stablehlo.multiply %v2482, %v2468 : tensor<32x256x14x14xf32>
    %v2489 = stablehlo.subtract %v2488, %v2484 : tensor<32x256x14x14xf32>
    %v2490 = stablehlo.multiply %v2480, %v2487 : tensor<32x256x14x14xf32>
    %v2491 = stablehlo.subtract %v2489, %v2490 : tensor<32x256x14x14xf32>
    %v2492 = stablehlo.divide %v2479, %v2468 : tensor<32x256x14x14xf32>
    %v2493 = stablehlo.multiply %v2492, %v2491 : tensor<32x256x14x14xf32>
    %v2494 = stablehlo.reshape %v2493 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2495 = stablehlo.reshape %v2494 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2496 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2497 = stablehlo.pad %v2495, %v2496, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %v2498 = stablehlo.transpose %d3Wp, dims = [1, 0, 2, 3] : (tensor<256x128x3x3xf32>) -> tensor<128x256x3x3xf32>
    %v2499 = stablehlo.reverse %v2498, dims = [2, 3] : tensor<128x256x3x3xf32>
    %v2500 = stablehlo.convolution(%v2497, %v2499)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x28x28xf32>, tensor<128x256x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2501 = stablehlo.reshape %v2500 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2502 = stablehlo.add %v2464, %v2501 : tensor<32x100352xf32>
    %v2503 = stablehlo.reshape %v440 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2504 = stablehlo.reshape %v2457 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2505 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2506 = stablehlo.pad %v2504, %v2505, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %v2507 = stablehlo.transpose %v2503, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2508 = stablehlo.transpose %v2506, dims = [1, 0, 2, 3] : (tensor<32x256x28x28xf32>) -> tensor<256x32x28x28xf32>
    %v2509 = stablehlo.convolution(%v2507, %v2508)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<256x32x28x28xf32>) -> tensor<128x256x3x3xf32>
    %v2510 = stablehlo.transpose %v2509, dims = [1, 0, 2, 3] : (tensor<128x256x3x3xf32>) -> tensor<256x128x3x3xf32>
    %v2511 = stablehlo.constant dense<0.003125> : tensor<256x128x3x3xf32>
    %v2512 = stablehlo.multiply %v2510, %v2511 : tensor<256x128x3x3xf32>
    %v2513 = stablehlo.subtract %d3W1, %v2512 : tensor<256x128x3x3xf32>
    %v2514 = stablehlo.reshape %v2457 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2515 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2516 = stablehlo.reduce(%v2514 init: %v2515) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2517 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2518 = stablehlo.multiply %v2516, %v2517 : tensor<256xf32>
    %v2519 = stablehlo.subtract %d3b1, %v2518 : tensor<256xf32>
    %v2520 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2521 = stablehlo.reshape %v445 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2522 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2523 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2524 = stablehlo.reduce(%v2521 init: %v2520) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2525 = stablehlo.broadcast_in_dim %v2524, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2526 = stablehlo.divide %v2525, %v2522 : tensor<32x256x14x14xf32>
    %v2527 = stablehlo.subtract %v2521, %v2526 : tensor<32x256x14x14xf32>
    %v2528 = stablehlo.multiply %v2527, %v2527 : tensor<32x256x14x14xf32>
    %v2529 = stablehlo.reduce(%v2528 init: %v2520) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2530 = stablehlo.broadcast_in_dim %v2529, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2531 = stablehlo.divide %v2530, %v2522 : tensor<32x256x14x14xf32>
    %v2532 = stablehlo.add %v2531, %v2523 : tensor<32x256x14x14xf32>
    %v2533 = stablehlo.rsqrt %v2532 : tensor<32x256x14x14xf32>
    %v2534 = stablehlo.multiply %v2527, %v2533 : tensor<32x256x14x14xf32>
    %v2535 = stablehlo.reshape %v2427 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2536 = stablehlo.multiply %v2535, %v2534 : tensor<32x256x14x14xf32>
    %v2537 = stablehlo.reduce(%v2536 init: %v2520) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2538 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2539 = stablehlo.multiply %v2537, %v2538 : tensor<256xf32>
    %v2540 = stablehlo.subtract %d3g1, %v2539 : tensor<256xf32>
    %v2541 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2542 = stablehlo.reshape %v2427 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2543 = stablehlo.reduce(%v2542 init: %v2541) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2544 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2545 = stablehlo.multiply %v2543, %v2544 : tensor<256xf32>
    %v2546 = stablehlo.subtract %d3bt1, %v2545 : tensor<256xf32>
    %v2547 = stablehlo.reshape %v467 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2548 = stablehlo.reshape %v2419 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2549 = stablehlo.transpose %v2547, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2550 = stablehlo.transpose %v2548, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2551 = stablehlo.convolution(%v2549, %v2550)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2552 = stablehlo.transpose %v2551, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2553 = stablehlo.constant dense<0.003125> : tensor<256x256x3x3xf32>
    %v2554 = stablehlo.multiply %v2552, %v2553 : tensor<256x256x3x3xf32>
    %v2555 = stablehlo.subtract %d3W2, %v2554 : tensor<256x256x3x3xf32>
    %v2556 = stablehlo.reshape %v2419 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2557 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2558 = stablehlo.reduce(%v2556 init: %v2557) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2559 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2560 = stablehlo.multiply %v2558, %v2559 : tensor<256xf32>
    %v2561 = stablehlo.subtract %d3b2, %v2560 : tensor<256xf32>
    %v2562 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2563 = stablehlo.reshape %v472 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2564 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2565 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2566 = stablehlo.reduce(%v2563 init: %v2562) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2567 = stablehlo.broadcast_in_dim %v2566, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2568 = stablehlo.divide %v2567, %v2564 : tensor<32x256x14x14xf32>
    %v2569 = stablehlo.subtract %v2563, %v2568 : tensor<32x256x14x14xf32>
    %v2570 = stablehlo.multiply %v2569, %v2569 : tensor<32x256x14x14xf32>
    %v2571 = stablehlo.reduce(%v2570 init: %v2562) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2572 = stablehlo.broadcast_in_dim %v2571, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2573 = stablehlo.divide %v2572, %v2564 : tensor<32x256x14x14xf32>
    %v2574 = stablehlo.add %v2573, %v2565 : tensor<32x256x14x14xf32>
    %v2575 = stablehlo.rsqrt %v2574 : tensor<32x256x14x14xf32>
    %v2576 = stablehlo.multiply %v2569, %v2575 : tensor<32x256x14x14xf32>
    %v2577 = stablehlo.reshape %v2389 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2578 = stablehlo.multiply %v2577, %v2576 : tensor<32x256x14x14xf32>
    %v2579 = stablehlo.reduce(%v2578 init: %v2562) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2580 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2581 = stablehlo.multiply %v2579, %v2580 : tensor<256xf32>
    %v2582 = stablehlo.subtract %d3g2, %v2581 : tensor<256xf32>
    %v2583 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2584 = stablehlo.reshape %v2389 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2585 = stablehlo.reduce(%v2584 init: %v2583) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2586 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2587 = stablehlo.multiply %v2585, %v2586 : tensor<256xf32>
    %v2588 = stablehlo.subtract %d3bt2, %v2587 : tensor<256xf32>
    %v2589 = stablehlo.reshape %v440 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2590 = stablehlo.reshape %v2494 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2591 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2592 = stablehlo.pad %v2590, %v2591, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %v2593 = stablehlo.transpose %v2589, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2594 = stablehlo.transpose %v2592, dims = [1, 0, 2, 3] : (tensor<32x256x28x28xf32>) -> tensor<256x32x28x28xf32>
    %v2595 = stablehlo.convolution(%v2593, %v2594)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<256x32x28x28xf32>) -> tensor<128x256x3x3xf32>
    %v2596 = stablehlo.transpose %v2595, dims = [1, 0, 2, 3] : (tensor<128x256x3x3xf32>) -> tensor<256x128x3x3xf32>
    %v2597 = stablehlo.constant dense<0.003125> : tensor<256x128x3x3xf32>
    %v2598 = stablehlo.multiply %v2596, %v2597 : tensor<256x128x3x3xf32>
    %v2599 = stablehlo.subtract %d3Wp, %v2598 : tensor<256x128x3x3xf32>
    %v2600 = stablehlo.reshape %v2494 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2601 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2602 = stablehlo.reduce(%v2600 init: %v2601) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2603 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2604 = stablehlo.multiply %v2602, %v2603 : tensor<256xf32>
    %v2605 = stablehlo.subtract %d3bp, %v2604 : tensor<256xf32>
    %v2606 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2607 = stablehlo.reshape %v497 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2608 = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %v2609 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2610 = stablehlo.reduce(%v2607 init: %v2606) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2611 = stablehlo.broadcast_in_dim %v2610, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2612 = stablehlo.divide %v2611, %v2608 : tensor<32x256x14x14xf32>
    %v2613 = stablehlo.subtract %v2607, %v2612 : tensor<32x256x14x14xf32>
    %v2614 = stablehlo.multiply %v2613, %v2613 : tensor<32x256x14x14xf32>
    %v2615 = stablehlo.reduce(%v2614 init: %v2606) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %v2616 = stablehlo.broadcast_in_dim %v2615, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %v2617 = stablehlo.divide %v2616, %v2608 : tensor<32x256x14x14xf32>
    %v2618 = stablehlo.add %v2617, %v2609 : tensor<32x256x14x14xf32>
    %v2619 = stablehlo.rsqrt %v2618 : tensor<32x256x14x14xf32>
    %v2620 = stablehlo.multiply %v2613, %v2619 : tensor<32x256x14x14xf32>
    %v2621 = stablehlo.reshape %v2389 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2622 = stablehlo.multiply %v2621, %v2620 : tensor<32x256x14x14xf32>
    %v2623 = stablehlo.reduce(%v2622 init: %v2606) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2624 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2625 = stablehlo.multiply %v2623, %v2624 : tensor<256xf32>
    %v2626 = stablehlo.subtract %d3gp, %v2625 : tensor<256xf32>
    %v2627 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2628 = stablehlo.reshape %v2389 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2629 = stablehlo.reduce(%v2628 init: %v2627) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2630 = stablehlo.constant dense<0.003125> : tensor<256xf32>
    %v2631 = stablehlo.multiply %v2629, %v2630 : tensor<256xf32>
    %v2632 = stablehlo.subtract %d3btp, %v2631 : tensor<256xf32>
    %v2633 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v2634 = stablehlo.compare GT, %v438, %v2633 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v2635 = stablehlo.select %v2634, %v2502, %v2633 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v2636 = stablehlo.reshape %v2635 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2637 = stablehlo.reshape %v417 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2638 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2639 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v2640 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2641 = stablehlo.reduce(%v2637 init: %v2638) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2642 = stablehlo.broadcast_in_dim %v2641, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2643 = stablehlo.divide %v2642, %v2639 : tensor<32x128x28x28xf32>
    %v2644 = stablehlo.subtract %v2637, %v2643 : tensor<32x128x28x28xf32>
    %v2645 = stablehlo.multiply %v2644, %v2644 : tensor<32x128x28x28xf32>
    %v2646 = stablehlo.reduce(%v2645 init: %v2638) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2647 = stablehlo.broadcast_in_dim %v2646, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2648 = stablehlo.divide %v2647, %v2639 : tensor<32x128x28x28xf32>
    %v2649 = stablehlo.add %v2648, %v2640 : tensor<32x128x28x28xf32>
    %v2650 = stablehlo.rsqrt %v2649 : tensor<32x128x28x28xf32>
    %v2651 = stablehlo.multiply %v2644, %v2650 : tensor<32x128x28x28xf32>
    %v2652 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2653 = stablehlo.multiply %v2652, %v2636 : tensor<32x128x28x28xf32>
    %v2654 = stablehlo.reduce(%v2653 init: %v2638) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2655 = stablehlo.broadcast_in_dim %v2654, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2656 = stablehlo.multiply %v2651, %v2653 : tensor<32x128x28x28xf32>
    %v2657 = stablehlo.reduce(%v2656 init: %v2638) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2658 = stablehlo.broadcast_in_dim %v2657, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2659 = stablehlo.multiply %v2653, %v2639 : tensor<32x128x28x28xf32>
    %v2660 = stablehlo.subtract %v2659, %v2655 : tensor<32x128x28x28xf32>
    %v2661 = stablehlo.multiply %v2651, %v2658 : tensor<32x128x28x28xf32>
    %v2662 = stablehlo.subtract %v2660, %v2661 : tensor<32x128x28x28xf32>
    %v2663 = stablehlo.divide %v2650, %v2639 : tensor<32x128x28x28xf32>
    %v2664 = stablehlo.multiply %v2663, %v2662 : tensor<32x128x28x28xf32>
    %v2665 = stablehlo.reshape %v2664 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2666 = stablehlo.reshape %v2665 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2667 = stablehlo.transpose %s2b2W2, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2668 = stablehlo.reverse %v2667, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2669 = stablehlo.convolution(%v2666, %v2668)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2670 = stablehlo.reshape %v2669 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2671 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v2672 = stablehlo.compare GT, %v410, %v2671 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v2673 = stablehlo.select %v2672, %v2670, %v2671 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v2674 = stablehlo.reshape %v2673 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2675 = stablehlo.reshape %v390 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2676 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2677 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v2678 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2679 = stablehlo.reduce(%v2675 init: %v2676) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2680 = stablehlo.broadcast_in_dim %v2679, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2681 = stablehlo.divide %v2680, %v2677 : tensor<32x128x28x28xf32>
    %v2682 = stablehlo.subtract %v2675, %v2681 : tensor<32x128x28x28xf32>
    %v2683 = stablehlo.multiply %v2682, %v2682 : tensor<32x128x28x28xf32>
    %v2684 = stablehlo.reduce(%v2683 init: %v2676) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2685 = stablehlo.broadcast_in_dim %v2684, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2686 = stablehlo.divide %v2685, %v2677 : tensor<32x128x28x28xf32>
    %v2687 = stablehlo.add %v2686, %v2678 : tensor<32x128x28x28xf32>
    %v2688 = stablehlo.rsqrt %v2687 : tensor<32x128x28x28xf32>
    %v2689 = stablehlo.multiply %v2682, %v2688 : tensor<32x128x28x28xf32>
    %v2690 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2691 = stablehlo.multiply %v2690, %v2674 : tensor<32x128x28x28xf32>
    %v2692 = stablehlo.reduce(%v2691 init: %v2676) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2693 = stablehlo.broadcast_in_dim %v2692, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2694 = stablehlo.multiply %v2689, %v2691 : tensor<32x128x28x28xf32>
    %v2695 = stablehlo.reduce(%v2694 init: %v2676) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2696 = stablehlo.broadcast_in_dim %v2695, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2697 = stablehlo.multiply %v2691, %v2677 : tensor<32x128x28x28xf32>
    %v2698 = stablehlo.subtract %v2697, %v2693 : tensor<32x128x28x28xf32>
    %v2699 = stablehlo.multiply %v2689, %v2696 : tensor<32x128x28x28xf32>
    %v2700 = stablehlo.subtract %v2698, %v2699 : tensor<32x128x28x28xf32>
    %v2701 = stablehlo.divide %v2688, %v2677 : tensor<32x128x28x28xf32>
    %v2702 = stablehlo.multiply %v2701, %v2700 : tensor<32x128x28x28xf32>
    %v2703 = stablehlo.reshape %v2702 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2704 = stablehlo.reshape %v2703 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2705 = stablehlo.transpose %s2b2W1, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2706 = stablehlo.reverse %v2705, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2707 = stablehlo.convolution(%v2704, %v2706)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2708 = stablehlo.reshape %v2707 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2709 = stablehlo.add %v2708, %v2635 : tensor<32x100352xf32>
    %v2710 = stablehlo.reshape %v385 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2711 = stablehlo.reshape %v2703 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2712 = stablehlo.transpose %v2710, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2713 = stablehlo.transpose %v2711, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2714 = stablehlo.convolution(%v2712, %v2713)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2715 = stablehlo.transpose %v2714, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2716 = stablehlo.constant dense<0.003125> : tensor<128x128x3x3xf32>
    %v2717 = stablehlo.multiply %v2715, %v2716 : tensor<128x128x3x3xf32>
    %v2718 = stablehlo.subtract %s2b2W1, %v2717 : tensor<128x128x3x3xf32>
    %v2719 = stablehlo.reshape %v2703 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2720 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2721 = stablehlo.reduce(%v2719 init: %v2720) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2722 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v2723 = stablehlo.multiply %v2721, %v2722 : tensor<128xf32>
    %v2724 = stablehlo.subtract %s2b2b1, %v2723 : tensor<128xf32>
    %v2725 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2726 = stablehlo.reshape %v390 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2727 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v2728 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2729 = stablehlo.reduce(%v2726 init: %v2725) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2730 = stablehlo.broadcast_in_dim %v2729, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2731 = stablehlo.divide %v2730, %v2727 : tensor<32x128x28x28xf32>
    %v2732 = stablehlo.subtract %v2726, %v2731 : tensor<32x128x28x28xf32>
    %v2733 = stablehlo.multiply %v2732, %v2732 : tensor<32x128x28x28xf32>
    %v2734 = stablehlo.reduce(%v2733 init: %v2725) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2735 = stablehlo.broadcast_in_dim %v2734, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2736 = stablehlo.divide %v2735, %v2727 : tensor<32x128x28x28xf32>
    %v2737 = stablehlo.add %v2736, %v2728 : tensor<32x128x28x28xf32>
    %v2738 = stablehlo.rsqrt %v2737 : tensor<32x128x28x28xf32>
    %v2739 = stablehlo.multiply %v2732, %v2738 : tensor<32x128x28x28xf32>
    %v2740 = stablehlo.reshape %v2673 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2741 = stablehlo.multiply %v2740, %v2739 : tensor<32x128x28x28xf32>
    %v2742 = stablehlo.reduce(%v2741 init: %v2725) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2743 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v2744 = stablehlo.multiply %v2742, %v2743 : tensor<128xf32>
    %v2745 = stablehlo.subtract %s2b2g1, %v2744 : tensor<128xf32>
    %v2746 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2747 = stablehlo.reshape %v2673 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2748 = stablehlo.reduce(%v2747 init: %v2746) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2749 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v2750 = stablehlo.multiply %v2748, %v2749 : tensor<128xf32>
    %v2751 = stablehlo.subtract %s2b2bt1, %v2750 : tensor<128xf32>
    %v2752 = stablehlo.reshape %v412 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2753 = stablehlo.reshape %v2665 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2754 = stablehlo.transpose %v2752, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2755 = stablehlo.transpose %v2753, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2756 = stablehlo.convolution(%v2754, %v2755)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2757 = stablehlo.transpose %v2756, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2758 = stablehlo.constant dense<0.003125> : tensor<128x128x3x3xf32>
    %v2759 = stablehlo.multiply %v2757, %v2758 : tensor<128x128x3x3xf32>
    %v2760 = stablehlo.subtract %s2b2W2, %v2759 : tensor<128x128x3x3xf32>
    %v2761 = stablehlo.reshape %v2665 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2762 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2763 = stablehlo.reduce(%v2761 init: %v2762) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2764 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v2765 = stablehlo.multiply %v2763, %v2764 : tensor<128xf32>
    %v2766 = stablehlo.subtract %s2b2b2, %v2765 : tensor<128xf32>
    %v2767 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2768 = stablehlo.reshape %v417 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2769 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v2770 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2771 = stablehlo.reduce(%v2768 init: %v2767) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2772 = stablehlo.broadcast_in_dim %v2771, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2773 = stablehlo.divide %v2772, %v2769 : tensor<32x128x28x28xf32>
    %v2774 = stablehlo.subtract %v2768, %v2773 : tensor<32x128x28x28xf32>
    %v2775 = stablehlo.multiply %v2774, %v2774 : tensor<32x128x28x28xf32>
    %v2776 = stablehlo.reduce(%v2775 init: %v2767) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2777 = stablehlo.broadcast_in_dim %v2776, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2778 = stablehlo.divide %v2777, %v2769 : tensor<32x128x28x28xf32>
    %v2779 = stablehlo.add %v2778, %v2770 : tensor<32x128x28x28xf32>
    %v2780 = stablehlo.rsqrt %v2779 : tensor<32x128x28x28xf32>
    %v2781 = stablehlo.multiply %v2774, %v2780 : tensor<32x128x28x28xf32>
    %v2782 = stablehlo.reshape %v2635 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2783 = stablehlo.multiply %v2782, %v2781 : tensor<32x128x28x28xf32>
    %v2784 = stablehlo.reduce(%v2783 init: %v2767) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2785 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v2786 = stablehlo.multiply %v2784, %v2785 : tensor<128xf32>
    %v2787 = stablehlo.subtract %s2b2g2, %v2786 : tensor<128xf32>
    %v2788 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2789 = stablehlo.reshape %v2635 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2790 = stablehlo.reduce(%v2789 init: %v2788) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2791 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v2792 = stablehlo.multiply %v2790, %v2791 : tensor<128xf32>
    %v2793 = stablehlo.subtract %s2b2bt2, %v2792 : tensor<128xf32>
    %v2794 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v2795 = stablehlo.compare GT, %v383, %v2794 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v2796 = stablehlo.select %v2795, %v2709, %v2794 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v2797 = stablehlo.reshape %v2796 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2798 = stablehlo.reshape %v362 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2799 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2800 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v2801 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2802 = stablehlo.reduce(%v2798 init: %v2799) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2803 = stablehlo.broadcast_in_dim %v2802, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2804 = stablehlo.divide %v2803, %v2800 : tensor<32x128x28x28xf32>
    %v2805 = stablehlo.subtract %v2798, %v2804 : tensor<32x128x28x28xf32>
    %v2806 = stablehlo.multiply %v2805, %v2805 : tensor<32x128x28x28xf32>
    %v2807 = stablehlo.reduce(%v2806 init: %v2799) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2808 = stablehlo.broadcast_in_dim %v2807, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2809 = stablehlo.divide %v2808, %v2800 : tensor<32x128x28x28xf32>
    %v2810 = stablehlo.add %v2809, %v2801 : tensor<32x128x28x28xf32>
    %v2811 = stablehlo.rsqrt %v2810 : tensor<32x128x28x28xf32>
    %v2812 = stablehlo.multiply %v2805, %v2811 : tensor<32x128x28x28xf32>
    %v2813 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2814 = stablehlo.multiply %v2813, %v2797 : tensor<32x128x28x28xf32>
    %v2815 = stablehlo.reduce(%v2814 init: %v2799) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2816 = stablehlo.broadcast_in_dim %v2815, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2817 = stablehlo.multiply %v2812, %v2814 : tensor<32x128x28x28xf32>
    %v2818 = stablehlo.reduce(%v2817 init: %v2799) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2819 = stablehlo.broadcast_in_dim %v2818, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2820 = stablehlo.multiply %v2814, %v2800 : tensor<32x128x28x28xf32>
    %v2821 = stablehlo.subtract %v2820, %v2816 : tensor<32x128x28x28xf32>
    %v2822 = stablehlo.multiply %v2812, %v2819 : tensor<32x128x28x28xf32>
    %v2823 = stablehlo.subtract %v2821, %v2822 : tensor<32x128x28x28xf32>
    %v2824 = stablehlo.divide %v2811, %v2800 : tensor<32x128x28x28xf32>
    %v2825 = stablehlo.multiply %v2824, %v2823 : tensor<32x128x28x28xf32>
    %v2826 = stablehlo.reshape %v2825 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2827 = stablehlo.reshape %v2826 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2828 = stablehlo.transpose %s2b1W2, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2829 = stablehlo.reverse %v2828, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2830 = stablehlo.convolution(%v2827, %v2829)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2831 = stablehlo.reshape %v2830 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2832 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v2833 = stablehlo.compare GT, %v355, %v2832 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v2834 = stablehlo.select %v2833, %v2831, %v2832 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v2835 = stablehlo.reshape %v2834 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2836 = stablehlo.reshape %v335 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2837 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2838 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v2839 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2840 = stablehlo.reduce(%v2836 init: %v2837) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2841 = stablehlo.broadcast_in_dim %v2840, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2842 = stablehlo.divide %v2841, %v2838 : tensor<32x128x28x28xf32>
    %v2843 = stablehlo.subtract %v2836, %v2842 : tensor<32x128x28x28xf32>
    %v2844 = stablehlo.multiply %v2843, %v2843 : tensor<32x128x28x28xf32>
    %v2845 = stablehlo.reduce(%v2844 init: %v2837) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2846 = stablehlo.broadcast_in_dim %v2845, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2847 = stablehlo.divide %v2846, %v2838 : tensor<32x128x28x28xf32>
    %v2848 = stablehlo.add %v2847, %v2839 : tensor<32x128x28x28xf32>
    %v2849 = stablehlo.rsqrt %v2848 : tensor<32x128x28x28xf32>
    %v2850 = stablehlo.multiply %v2843, %v2849 : tensor<32x128x28x28xf32>
    %v2851 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2852 = stablehlo.multiply %v2851, %v2835 : tensor<32x128x28x28xf32>
    %v2853 = stablehlo.reduce(%v2852 init: %v2837) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2854 = stablehlo.broadcast_in_dim %v2853, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2855 = stablehlo.multiply %v2850, %v2852 : tensor<32x128x28x28xf32>
    %v2856 = stablehlo.reduce(%v2855 init: %v2837) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2857 = stablehlo.broadcast_in_dim %v2856, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2858 = stablehlo.multiply %v2852, %v2838 : tensor<32x128x28x28xf32>
    %v2859 = stablehlo.subtract %v2858, %v2854 : tensor<32x128x28x28xf32>
    %v2860 = stablehlo.multiply %v2850, %v2857 : tensor<32x128x28x28xf32>
    %v2861 = stablehlo.subtract %v2859, %v2860 : tensor<32x128x28x28xf32>
    %v2862 = stablehlo.divide %v2849, %v2838 : tensor<32x128x28x28xf32>
    %v2863 = stablehlo.multiply %v2862, %v2861 : tensor<32x128x28x28xf32>
    %v2864 = stablehlo.reshape %v2863 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2865 = stablehlo.reshape %v2864 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2866 = stablehlo.transpose %s2b1W1, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2867 = stablehlo.reverse %v2866, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2868 = stablehlo.convolution(%v2865, %v2867)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2869 = stablehlo.reshape %v2868 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2870 = stablehlo.add %v2869, %v2796 : tensor<32x100352xf32>
    %v2871 = stablehlo.reshape %v330 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2872 = stablehlo.reshape %v2864 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2873 = stablehlo.transpose %v2871, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2874 = stablehlo.transpose %v2872, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2875 = stablehlo.convolution(%v2873, %v2874)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2876 = stablehlo.transpose %v2875, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2877 = stablehlo.constant dense<0.003125> : tensor<128x128x3x3xf32>
    %v2878 = stablehlo.multiply %v2876, %v2877 : tensor<128x128x3x3xf32>
    %v2879 = stablehlo.subtract %s2b1W1, %v2878 : tensor<128x128x3x3xf32>
    %v2880 = stablehlo.reshape %v2864 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2881 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2882 = stablehlo.reduce(%v2880 init: %v2881) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2883 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v2884 = stablehlo.multiply %v2882, %v2883 : tensor<128xf32>
    %v2885 = stablehlo.subtract %s2b1b1, %v2884 : tensor<128xf32>
    %v2886 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2887 = stablehlo.reshape %v335 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2888 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v2889 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2890 = stablehlo.reduce(%v2887 init: %v2886) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2891 = stablehlo.broadcast_in_dim %v2890, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2892 = stablehlo.divide %v2891, %v2888 : tensor<32x128x28x28xf32>
    %v2893 = stablehlo.subtract %v2887, %v2892 : tensor<32x128x28x28xf32>
    %v2894 = stablehlo.multiply %v2893, %v2893 : tensor<32x128x28x28xf32>
    %v2895 = stablehlo.reduce(%v2894 init: %v2886) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2896 = stablehlo.broadcast_in_dim %v2895, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2897 = stablehlo.divide %v2896, %v2888 : tensor<32x128x28x28xf32>
    %v2898 = stablehlo.add %v2897, %v2889 : tensor<32x128x28x28xf32>
    %v2899 = stablehlo.rsqrt %v2898 : tensor<32x128x28x28xf32>
    %v2900 = stablehlo.multiply %v2893, %v2899 : tensor<32x128x28x28xf32>
    %v2901 = stablehlo.reshape %v2834 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2902 = stablehlo.multiply %v2901, %v2900 : tensor<32x128x28x28xf32>
    %v2903 = stablehlo.reduce(%v2902 init: %v2886) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2904 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v2905 = stablehlo.multiply %v2903, %v2904 : tensor<128xf32>
    %v2906 = stablehlo.subtract %s2b1g1, %v2905 : tensor<128xf32>
    %v2907 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2908 = stablehlo.reshape %v2834 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2909 = stablehlo.reduce(%v2908 init: %v2907) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2910 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v2911 = stablehlo.multiply %v2909, %v2910 : tensor<128xf32>
    %v2912 = stablehlo.subtract %s2b1bt1, %v2911 : tensor<128xf32>
    %v2913 = stablehlo.reshape %v357 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2914 = stablehlo.reshape %v2826 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2915 = stablehlo.transpose %v2913, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2916 = stablehlo.transpose %v2914, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2917 = stablehlo.convolution(%v2915, %v2916)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2918 = stablehlo.transpose %v2917, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2919 = stablehlo.constant dense<0.003125> : tensor<128x128x3x3xf32>
    %v2920 = stablehlo.multiply %v2918, %v2919 : tensor<128x128x3x3xf32>
    %v2921 = stablehlo.subtract %s2b1W2, %v2920 : tensor<128x128x3x3xf32>
    %v2922 = stablehlo.reshape %v2826 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2923 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2924 = stablehlo.reduce(%v2922 init: %v2923) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2925 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v2926 = stablehlo.multiply %v2924, %v2925 : tensor<128xf32>
    %v2927 = stablehlo.subtract %s2b1b2, %v2926 : tensor<128xf32>
    %v2928 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2929 = stablehlo.reshape %v362 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2930 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v2931 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2932 = stablehlo.reduce(%v2929 init: %v2928) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2933 = stablehlo.broadcast_in_dim %v2932, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2934 = stablehlo.divide %v2933, %v2930 : tensor<32x128x28x28xf32>
    %v2935 = stablehlo.subtract %v2929, %v2934 : tensor<32x128x28x28xf32>
    %v2936 = stablehlo.multiply %v2935, %v2935 : tensor<32x128x28x28xf32>
    %v2937 = stablehlo.reduce(%v2936 init: %v2928) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2938 = stablehlo.broadcast_in_dim %v2937, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2939 = stablehlo.divide %v2938, %v2930 : tensor<32x128x28x28xf32>
    %v2940 = stablehlo.add %v2939, %v2931 : tensor<32x128x28x28xf32>
    %v2941 = stablehlo.rsqrt %v2940 : tensor<32x128x28x28xf32>
    %v2942 = stablehlo.multiply %v2935, %v2941 : tensor<32x128x28x28xf32>
    %v2943 = stablehlo.reshape %v2796 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2944 = stablehlo.multiply %v2943, %v2942 : tensor<32x128x28x28xf32>
    %v2945 = stablehlo.reduce(%v2944 init: %v2928) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2946 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v2947 = stablehlo.multiply %v2945, %v2946 : tensor<128xf32>
    %v2948 = stablehlo.subtract %s2b1g2, %v2947 : tensor<128xf32>
    %v2949 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2950 = stablehlo.reshape %v2796 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2951 = stablehlo.reduce(%v2950 init: %v2949) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2952 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v2953 = stablehlo.multiply %v2951, %v2952 : tensor<128xf32>
    %v2954 = stablehlo.subtract %s2b1bt2, %v2953 : tensor<128xf32>
    %v2955 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v2956 = stablehlo.compare GT, %v328, %v2955 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v2957 = stablehlo.select %v2956, %v2870, %v2955 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v2958 = stablehlo.reshape %v2957 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2959 = stablehlo.reshape %v307 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2960 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2961 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v2962 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2963 = stablehlo.reduce(%v2959 init: %v2960) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2964 = stablehlo.broadcast_in_dim %v2963, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2965 = stablehlo.divide %v2964, %v2961 : tensor<32x128x28x28xf32>
    %v2966 = stablehlo.subtract %v2959, %v2965 : tensor<32x128x28x28xf32>
    %v2967 = stablehlo.multiply %v2966, %v2966 : tensor<32x128x28x28xf32>
    %v2968 = stablehlo.reduce(%v2967 init: %v2960) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2969 = stablehlo.broadcast_in_dim %v2968, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2970 = stablehlo.divide %v2969, %v2961 : tensor<32x128x28x28xf32>
    %v2971 = stablehlo.add %v2970, %v2962 : tensor<32x128x28x28xf32>
    %v2972 = stablehlo.rsqrt %v2971 : tensor<32x128x28x28xf32>
    %v2973 = stablehlo.multiply %v2966, %v2972 : tensor<32x128x28x28xf32>
    %v2974 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2975 = stablehlo.multiply %v2974, %v2958 : tensor<32x128x28x28xf32>
    %v2976 = stablehlo.reduce(%v2975 init: %v2960) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2977 = stablehlo.broadcast_in_dim %v2976, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2978 = stablehlo.multiply %v2973, %v2975 : tensor<32x128x28x28xf32>
    %v2979 = stablehlo.reduce(%v2978 init: %v2960) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v2980 = stablehlo.broadcast_in_dim %v2979, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v2981 = stablehlo.multiply %v2975, %v2961 : tensor<32x128x28x28xf32>
    %v2982 = stablehlo.subtract %v2981, %v2977 : tensor<32x128x28x28xf32>
    %v2983 = stablehlo.multiply %v2973, %v2980 : tensor<32x128x28x28xf32>
    %v2984 = stablehlo.subtract %v2982, %v2983 : tensor<32x128x28x28xf32>
    %v2985 = stablehlo.divide %v2972, %v2961 : tensor<32x128x28x28xf32>
    %v2986 = stablehlo.multiply %v2985, %v2984 : tensor<32x128x28x28xf32>
    %v2987 = stablehlo.reshape %v2986 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2988 = stablehlo.reshape %v2987 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2989 = stablehlo.transpose %s2b0W2, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2990 = stablehlo.reverse %v2989, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2991 = stablehlo.convolution(%v2988, %v2990)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2992 = stablehlo.reshape %v2991 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2993 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v2994 = stablehlo.compare GT, %v300, %v2993 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v2995 = stablehlo.select %v2994, %v2992, %v2993 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v2996 = stablehlo.reshape %v2995 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2997 = stablehlo.reshape %v280 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2998 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2999 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v3000 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3001 = stablehlo.reduce(%v2997 init: %v2998) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3002 = stablehlo.broadcast_in_dim %v3001, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3003 = stablehlo.divide %v3002, %v2999 : tensor<32x128x28x28xf32>
    %v3004 = stablehlo.subtract %v2997, %v3003 : tensor<32x128x28x28xf32>
    %v3005 = stablehlo.multiply %v3004, %v3004 : tensor<32x128x28x28xf32>
    %v3006 = stablehlo.reduce(%v3005 init: %v2998) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3007 = stablehlo.broadcast_in_dim %v3006, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3008 = stablehlo.divide %v3007, %v2999 : tensor<32x128x28x28xf32>
    %v3009 = stablehlo.add %v3008, %v3000 : tensor<32x128x28x28xf32>
    %v3010 = stablehlo.rsqrt %v3009 : tensor<32x128x28x28xf32>
    %v3011 = stablehlo.multiply %v3004, %v3010 : tensor<32x128x28x28xf32>
    %v3012 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3013 = stablehlo.multiply %v3012, %v2996 : tensor<32x128x28x28xf32>
    %v3014 = stablehlo.reduce(%v3013 init: %v2998) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3015 = stablehlo.broadcast_in_dim %v3014, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3016 = stablehlo.multiply %v3011, %v3013 : tensor<32x128x28x28xf32>
    %v3017 = stablehlo.reduce(%v3016 init: %v2998) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3018 = stablehlo.broadcast_in_dim %v3017, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3019 = stablehlo.multiply %v3013, %v2999 : tensor<32x128x28x28xf32>
    %v3020 = stablehlo.subtract %v3019, %v3015 : tensor<32x128x28x28xf32>
    %v3021 = stablehlo.multiply %v3011, %v3018 : tensor<32x128x28x28xf32>
    %v3022 = stablehlo.subtract %v3020, %v3021 : tensor<32x128x28x28xf32>
    %v3023 = stablehlo.divide %v3010, %v2999 : tensor<32x128x28x28xf32>
    %v3024 = stablehlo.multiply %v3023, %v3022 : tensor<32x128x28x28xf32>
    %v3025 = stablehlo.reshape %v3024 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3026 = stablehlo.reshape %v3025 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3027 = stablehlo.transpose %s2b0W1, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3028 = stablehlo.reverse %v3027, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3029 = stablehlo.convolution(%v3026, %v3028)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v3030 = stablehlo.reshape %v3029 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3031 = stablehlo.add %v3030, %v2957 : tensor<32x100352xf32>
    %v3032 = stablehlo.reshape %v275 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3033 = stablehlo.reshape %v3025 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3034 = stablehlo.transpose %v3032, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3035 = stablehlo.transpose %v3033, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3036 = stablehlo.convolution(%v3034, %v3035)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v3037 = stablehlo.transpose %v3036, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3038 = stablehlo.constant dense<0.003125> : tensor<128x128x3x3xf32>
    %v3039 = stablehlo.multiply %v3037, %v3038 : tensor<128x128x3x3xf32>
    %v3040 = stablehlo.subtract %s2b0W1, %v3039 : tensor<128x128x3x3xf32>
    %v3041 = stablehlo.reshape %v3025 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3042 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3043 = stablehlo.reduce(%v3041 init: %v3042) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3044 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v3045 = stablehlo.multiply %v3043, %v3044 : tensor<128xf32>
    %v3046 = stablehlo.subtract %s2b0b1, %v3045 : tensor<128xf32>
    %v3047 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3048 = stablehlo.reshape %v280 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3049 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v3050 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3051 = stablehlo.reduce(%v3048 init: %v3047) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3052 = stablehlo.broadcast_in_dim %v3051, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3053 = stablehlo.divide %v3052, %v3049 : tensor<32x128x28x28xf32>
    %v3054 = stablehlo.subtract %v3048, %v3053 : tensor<32x128x28x28xf32>
    %v3055 = stablehlo.multiply %v3054, %v3054 : tensor<32x128x28x28xf32>
    %v3056 = stablehlo.reduce(%v3055 init: %v3047) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3057 = stablehlo.broadcast_in_dim %v3056, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3058 = stablehlo.divide %v3057, %v3049 : tensor<32x128x28x28xf32>
    %v3059 = stablehlo.add %v3058, %v3050 : tensor<32x128x28x28xf32>
    %v3060 = stablehlo.rsqrt %v3059 : tensor<32x128x28x28xf32>
    %v3061 = stablehlo.multiply %v3054, %v3060 : tensor<32x128x28x28xf32>
    %v3062 = stablehlo.reshape %v2995 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3063 = stablehlo.multiply %v3062, %v3061 : tensor<32x128x28x28xf32>
    %v3064 = stablehlo.reduce(%v3063 init: %v3047) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3065 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v3066 = stablehlo.multiply %v3064, %v3065 : tensor<128xf32>
    %v3067 = stablehlo.subtract %s2b0g1, %v3066 : tensor<128xf32>
    %v3068 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3069 = stablehlo.reshape %v2995 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3070 = stablehlo.reduce(%v3069 init: %v3068) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3071 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v3072 = stablehlo.multiply %v3070, %v3071 : tensor<128xf32>
    %v3073 = stablehlo.subtract %s2b0bt1, %v3072 : tensor<128xf32>
    %v3074 = stablehlo.reshape %v302 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3075 = stablehlo.reshape %v2987 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3076 = stablehlo.transpose %v3074, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3077 = stablehlo.transpose %v3075, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3078 = stablehlo.convolution(%v3076, %v3077)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v3079 = stablehlo.transpose %v3078, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3080 = stablehlo.constant dense<0.003125> : tensor<128x128x3x3xf32>
    %v3081 = stablehlo.multiply %v3079, %v3080 : tensor<128x128x3x3xf32>
    %v3082 = stablehlo.subtract %s2b0W2, %v3081 : tensor<128x128x3x3xf32>
    %v3083 = stablehlo.reshape %v2987 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3084 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3085 = stablehlo.reduce(%v3083 init: %v3084) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3086 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v3087 = stablehlo.multiply %v3085, %v3086 : tensor<128xf32>
    %v3088 = stablehlo.subtract %s2b0b2, %v3087 : tensor<128xf32>
    %v3089 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3090 = stablehlo.reshape %v307 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3091 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v3092 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3093 = stablehlo.reduce(%v3090 init: %v3089) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3094 = stablehlo.broadcast_in_dim %v3093, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3095 = stablehlo.divide %v3094, %v3091 : tensor<32x128x28x28xf32>
    %v3096 = stablehlo.subtract %v3090, %v3095 : tensor<32x128x28x28xf32>
    %v3097 = stablehlo.multiply %v3096, %v3096 : tensor<32x128x28x28xf32>
    %v3098 = stablehlo.reduce(%v3097 init: %v3089) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3099 = stablehlo.broadcast_in_dim %v3098, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3100 = stablehlo.divide %v3099, %v3091 : tensor<32x128x28x28xf32>
    %v3101 = stablehlo.add %v3100, %v3092 : tensor<32x128x28x28xf32>
    %v3102 = stablehlo.rsqrt %v3101 : tensor<32x128x28x28xf32>
    %v3103 = stablehlo.multiply %v3096, %v3102 : tensor<32x128x28x28xf32>
    %v3104 = stablehlo.reshape %v2957 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3105 = stablehlo.multiply %v3104, %v3103 : tensor<32x128x28x28xf32>
    %v3106 = stablehlo.reduce(%v3105 init: %v3089) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3107 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v3108 = stablehlo.multiply %v3106, %v3107 : tensor<128xf32>
    %v3109 = stablehlo.subtract %s2b0g2, %v3108 : tensor<128xf32>
    %v3110 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3111 = stablehlo.reshape %v2957 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3112 = stablehlo.reduce(%v3111 init: %v3110) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3113 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v3114 = stablehlo.multiply %v3112, %v3113 : tensor<128xf32>
    %v3115 = stablehlo.subtract %s2b0bt2, %v3114 : tensor<128xf32>
    %v3116 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v3117 = stablehlo.compare GT, %v273, %v3116 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v3118 = stablehlo.select %v3117, %v3031, %v3116 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v3119 = stablehlo.reshape %v3118 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3120 = stablehlo.reshape %v227 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3121 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3122 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v3123 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3124 = stablehlo.reduce(%v3120 init: %v3121) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3125 = stablehlo.broadcast_in_dim %v3124, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3126 = stablehlo.divide %v3125, %v3122 : tensor<32x128x28x28xf32>
    %v3127 = stablehlo.subtract %v3120, %v3126 : tensor<32x128x28x28xf32>
    %v3128 = stablehlo.multiply %v3127, %v3127 : tensor<32x128x28x28xf32>
    %v3129 = stablehlo.reduce(%v3128 init: %v3121) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3130 = stablehlo.broadcast_in_dim %v3129, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3131 = stablehlo.divide %v3130, %v3122 : tensor<32x128x28x28xf32>
    %v3132 = stablehlo.add %v3131, %v3123 : tensor<32x128x28x28xf32>
    %v3133 = stablehlo.rsqrt %v3132 : tensor<32x128x28x28xf32>
    %v3134 = stablehlo.multiply %v3127, %v3133 : tensor<32x128x28x28xf32>
    %v3135 = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3136 = stablehlo.multiply %v3135, %v3119 : tensor<32x128x28x28xf32>
    %v3137 = stablehlo.reduce(%v3136 init: %v3121) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3138 = stablehlo.broadcast_in_dim %v3137, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3139 = stablehlo.multiply %v3134, %v3136 : tensor<32x128x28x28xf32>
    %v3140 = stablehlo.reduce(%v3139 init: %v3121) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3141 = stablehlo.broadcast_in_dim %v3140, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3142 = stablehlo.multiply %v3136, %v3122 : tensor<32x128x28x28xf32>
    %v3143 = stablehlo.subtract %v3142, %v3138 : tensor<32x128x28x28xf32>
    %v3144 = stablehlo.multiply %v3134, %v3141 : tensor<32x128x28x28xf32>
    %v3145 = stablehlo.subtract %v3143, %v3144 : tensor<32x128x28x28xf32>
    %v3146 = stablehlo.divide %v3133, %v3122 : tensor<32x128x28x28xf32>
    %v3147 = stablehlo.multiply %v3146, %v3145 : tensor<32x128x28x28xf32>
    %v3148 = stablehlo.reshape %v3147 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3149 = stablehlo.reshape %v3148 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3150 = stablehlo.transpose %d2W2, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3151 = stablehlo.reverse %v3150, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3152 = stablehlo.convolution(%v3149, %v3151)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v3153 = stablehlo.reshape %v3152 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3154 = stablehlo.constant dense<0.0> : tensor<32x100352xf32>
    %v3155 = stablehlo.compare GT, %v220, %v3154 : (tensor<32x100352xf32>, tensor<32x100352xf32>) -> tensor<32x100352xi1>
    %v3156 = stablehlo.select %v3155, %v3153, %v3154 : tensor<32x100352xi1>, tensor<32x100352xf32>
    %v3157 = stablehlo.reshape %v3156 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3158 = stablehlo.reshape %v200 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3159 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3160 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v3161 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3162 = stablehlo.reduce(%v3158 init: %v3159) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3163 = stablehlo.broadcast_in_dim %v3162, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3164 = stablehlo.divide %v3163, %v3160 : tensor<32x128x28x28xf32>
    %v3165 = stablehlo.subtract %v3158, %v3164 : tensor<32x128x28x28xf32>
    %v3166 = stablehlo.multiply %v3165, %v3165 : tensor<32x128x28x28xf32>
    %v3167 = stablehlo.reduce(%v3166 init: %v3159) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3168 = stablehlo.broadcast_in_dim %v3167, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3169 = stablehlo.divide %v3168, %v3160 : tensor<32x128x28x28xf32>
    %v3170 = stablehlo.add %v3169, %v3161 : tensor<32x128x28x28xf32>
    %v3171 = stablehlo.rsqrt %v3170 : tensor<32x128x28x28xf32>
    %v3172 = stablehlo.multiply %v3165, %v3171 : tensor<32x128x28x28xf32>
    %v3173 = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3174 = stablehlo.multiply %v3173, %v3157 : tensor<32x128x28x28xf32>
    %v3175 = stablehlo.reduce(%v3174 init: %v3159) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3176 = stablehlo.broadcast_in_dim %v3175, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3177 = stablehlo.multiply %v3172, %v3174 : tensor<32x128x28x28xf32>
    %v3178 = stablehlo.reduce(%v3177 init: %v3159) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3179 = stablehlo.broadcast_in_dim %v3178, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3180 = stablehlo.multiply %v3174, %v3160 : tensor<32x128x28x28xf32>
    %v3181 = stablehlo.subtract %v3180, %v3176 : tensor<32x128x28x28xf32>
    %v3182 = stablehlo.multiply %v3172, %v3179 : tensor<32x128x28x28xf32>
    %v3183 = stablehlo.subtract %v3181, %v3182 : tensor<32x128x28x28xf32>
    %v3184 = stablehlo.divide %v3171, %v3160 : tensor<32x128x28x28xf32>
    %v3185 = stablehlo.multiply %v3184, %v3183 : tensor<32x128x28x28xf32>
    %v3186 = stablehlo.reshape %v3185 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3187 = stablehlo.reshape %v3186 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3188 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3189 = stablehlo.pad %v3187, %v3188, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %v3190 = stablehlo.transpose %d2W1, dims = [1, 0, 2, 3] : (tensor<128x64x3x3xf32>) -> tensor<64x128x3x3xf32>
    %v3191 = stablehlo.reverse %v3190, dims = [2, 3] : tensor<64x128x3x3xf32>
    %v3192 = stablehlo.convolution(%v3189, %v3191)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x56x56xf32>, tensor<64x128x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3193 = stablehlo.reshape %v3192 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3194 = stablehlo.reshape %v3118 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3195 = stablehlo.reshape %v252 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3196 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3197 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v3198 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3199 = stablehlo.reduce(%v3195 init: %v3196) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3200 = stablehlo.broadcast_in_dim %v3199, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3201 = stablehlo.divide %v3200, %v3197 : tensor<32x128x28x28xf32>
    %v3202 = stablehlo.subtract %v3195, %v3201 : tensor<32x128x28x28xf32>
    %v3203 = stablehlo.multiply %v3202, %v3202 : tensor<32x128x28x28xf32>
    %v3204 = stablehlo.reduce(%v3203 init: %v3196) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3205 = stablehlo.broadcast_in_dim %v3204, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3206 = stablehlo.divide %v3205, %v3197 : tensor<32x128x28x28xf32>
    %v3207 = stablehlo.add %v3206, %v3198 : tensor<32x128x28x28xf32>
    %v3208 = stablehlo.rsqrt %v3207 : tensor<32x128x28x28xf32>
    %v3209 = stablehlo.multiply %v3202, %v3208 : tensor<32x128x28x28xf32>
    %v3210 = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3211 = stablehlo.multiply %v3210, %v3194 : tensor<32x128x28x28xf32>
    %v3212 = stablehlo.reduce(%v3211 init: %v3196) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3213 = stablehlo.broadcast_in_dim %v3212, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3214 = stablehlo.multiply %v3209, %v3211 : tensor<32x128x28x28xf32>
    %v3215 = stablehlo.reduce(%v3214 init: %v3196) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3216 = stablehlo.broadcast_in_dim %v3215, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3217 = stablehlo.multiply %v3211, %v3197 : tensor<32x128x28x28xf32>
    %v3218 = stablehlo.subtract %v3217, %v3213 : tensor<32x128x28x28xf32>
    %v3219 = stablehlo.multiply %v3209, %v3216 : tensor<32x128x28x28xf32>
    %v3220 = stablehlo.subtract %v3218, %v3219 : tensor<32x128x28x28xf32>
    %v3221 = stablehlo.divide %v3208, %v3197 : tensor<32x128x28x28xf32>
    %v3222 = stablehlo.multiply %v3221, %v3220 : tensor<32x128x28x28xf32>
    %v3223 = stablehlo.reshape %v3222 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3224 = stablehlo.reshape %v3223 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3225 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3226 = stablehlo.pad %v3224, %v3225, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %v3227 = stablehlo.transpose %d2Wp, dims = [1, 0, 2, 3] : (tensor<128x64x3x3xf32>) -> tensor<64x128x3x3xf32>
    %v3228 = stablehlo.reverse %v3227, dims = [2, 3] : tensor<64x128x3x3xf32>
    %v3229 = stablehlo.convolution(%v3226, %v3228)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x56x56xf32>, tensor<64x128x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3230 = stablehlo.reshape %v3229 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3231 = stablehlo.add %v3193, %v3230 : tensor<32x200704xf32>
    %v3232 = stablehlo.reshape %v195 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3233 = stablehlo.reshape %v3186 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3234 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3235 = stablehlo.pad %v3233, %v3234, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %v3236 = stablehlo.transpose %v3232, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3237 = stablehlo.transpose %v3235, dims = [1, 0, 2, 3] : (tensor<32x128x56x56xf32>) -> tensor<128x32x56x56xf32>
    %v3238 = stablehlo.convolution(%v3236, %v3237)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<128x32x56x56xf32>) -> tensor<64x128x3x3xf32>
    %v3239 = stablehlo.transpose %v3238, dims = [1, 0, 2, 3] : (tensor<64x128x3x3xf32>) -> tensor<128x64x3x3xf32>
    %v3240 = stablehlo.constant dense<0.003125> : tensor<128x64x3x3xf32>
    %v3241 = stablehlo.multiply %v3239, %v3240 : tensor<128x64x3x3xf32>
    %v3242 = stablehlo.subtract %d2W1, %v3241 : tensor<128x64x3x3xf32>
    %v3243 = stablehlo.reshape %v3186 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3244 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3245 = stablehlo.reduce(%v3243 init: %v3244) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3246 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v3247 = stablehlo.multiply %v3245, %v3246 : tensor<128xf32>
    %v3248 = stablehlo.subtract %d2b1, %v3247 : tensor<128xf32>
    %v3249 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3250 = stablehlo.reshape %v200 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3251 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v3252 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3253 = stablehlo.reduce(%v3250 init: %v3249) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3254 = stablehlo.broadcast_in_dim %v3253, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3255 = stablehlo.divide %v3254, %v3251 : tensor<32x128x28x28xf32>
    %v3256 = stablehlo.subtract %v3250, %v3255 : tensor<32x128x28x28xf32>
    %v3257 = stablehlo.multiply %v3256, %v3256 : tensor<32x128x28x28xf32>
    %v3258 = stablehlo.reduce(%v3257 init: %v3249) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3259 = stablehlo.broadcast_in_dim %v3258, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3260 = stablehlo.divide %v3259, %v3251 : tensor<32x128x28x28xf32>
    %v3261 = stablehlo.add %v3260, %v3252 : tensor<32x128x28x28xf32>
    %v3262 = stablehlo.rsqrt %v3261 : tensor<32x128x28x28xf32>
    %v3263 = stablehlo.multiply %v3256, %v3262 : tensor<32x128x28x28xf32>
    %v3264 = stablehlo.reshape %v3156 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3265 = stablehlo.multiply %v3264, %v3263 : tensor<32x128x28x28xf32>
    %v3266 = stablehlo.reduce(%v3265 init: %v3249) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3267 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v3268 = stablehlo.multiply %v3266, %v3267 : tensor<128xf32>
    %v3269 = stablehlo.subtract %d2g1, %v3268 : tensor<128xf32>
    %v3270 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3271 = stablehlo.reshape %v3156 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3272 = stablehlo.reduce(%v3271 init: %v3270) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3273 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v3274 = stablehlo.multiply %v3272, %v3273 : tensor<128xf32>
    %v3275 = stablehlo.subtract %d2bt1, %v3274 : tensor<128xf32>
    %v3276 = stablehlo.reshape %v222 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3277 = stablehlo.reshape %v3148 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3278 = stablehlo.transpose %v3276, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3279 = stablehlo.transpose %v3277, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3280 = stablehlo.convolution(%v3278, %v3279)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v3281 = stablehlo.transpose %v3280, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3282 = stablehlo.constant dense<0.003125> : tensor<128x128x3x3xf32>
    %v3283 = stablehlo.multiply %v3281, %v3282 : tensor<128x128x3x3xf32>
    %v3284 = stablehlo.subtract %d2W2, %v3283 : tensor<128x128x3x3xf32>
    %v3285 = stablehlo.reshape %v3148 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3286 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3287 = stablehlo.reduce(%v3285 init: %v3286) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3288 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v3289 = stablehlo.multiply %v3287, %v3288 : tensor<128xf32>
    %v3290 = stablehlo.subtract %d2b2, %v3289 : tensor<128xf32>
    %v3291 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3292 = stablehlo.reshape %v227 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3293 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v3294 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3295 = stablehlo.reduce(%v3292 init: %v3291) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3296 = stablehlo.broadcast_in_dim %v3295, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3297 = stablehlo.divide %v3296, %v3293 : tensor<32x128x28x28xf32>
    %v3298 = stablehlo.subtract %v3292, %v3297 : tensor<32x128x28x28xf32>
    %v3299 = stablehlo.multiply %v3298, %v3298 : tensor<32x128x28x28xf32>
    %v3300 = stablehlo.reduce(%v3299 init: %v3291) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3301 = stablehlo.broadcast_in_dim %v3300, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3302 = stablehlo.divide %v3301, %v3293 : tensor<32x128x28x28xf32>
    %v3303 = stablehlo.add %v3302, %v3294 : tensor<32x128x28x28xf32>
    %v3304 = stablehlo.rsqrt %v3303 : tensor<32x128x28x28xf32>
    %v3305 = stablehlo.multiply %v3298, %v3304 : tensor<32x128x28x28xf32>
    %v3306 = stablehlo.reshape %v3118 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3307 = stablehlo.multiply %v3306, %v3305 : tensor<32x128x28x28xf32>
    %v3308 = stablehlo.reduce(%v3307 init: %v3291) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3309 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v3310 = stablehlo.multiply %v3308, %v3309 : tensor<128xf32>
    %v3311 = stablehlo.subtract %d2g2, %v3310 : tensor<128xf32>
    %v3312 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3313 = stablehlo.reshape %v3118 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3314 = stablehlo.reduce(%v3313 init: %v3312) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3315 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v3316 = stablehlo.multiply %v3314, %v3315 : tensor<128xf32>
    %v3317 = stablehlo.subtract %d2bt2, %v3316 : tensor<128xf32>
    %v3318 = stablehlo.reshape %v195 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3319 = stablehlo.reshape %v3223 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3320 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3321 = stablehlo.pad %v3319, %v3320, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %v3322 = stablehlo.transpose %v3318, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3323 = stablehlo.transpose %v3321, dims = [1, 0, 2, 3] : (tensor<32x128x56x56xf32>) -> tensor<128x32x56x56xf32>
    %v3324 = stablehlo.convolution(%v3322, %v3323)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<128x32x56x56xf32>) -> tensor<64x128x3x3xf32>
    %v3325 = stablehlo.transpose %v3324, dims = [1, 0, 2, 3] : (tensor<64x128x3x3xf32>) -> tensor<128x64x3x3xf32>
    %v3326 = stablehlo.constant dense<0.003125> : tensor<128x64x3x3xf32>
    %v3327 = stablehlo.multiply %v3325, %v3326 : tensor<128x64x3x3xf32>
    %v3328 = stablehlo.subtract %d2Wp, %v3327 : tensor<128x64x3x3xf32>
    %v3329 = stablehlo.reshape %v3223 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3330 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3331 = stablehlo.reduce(%v3329 init: %v3330) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3332 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v3333 = stablehlo.multiply %v3331, %v3332 : tensor<128xf32>
    %v3334 = stablehlo.subtract %d2bp, %v3333 : tensor<128xf32>
    %v3335 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3336 = stablehlo.reshape %v252 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3337 = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %v3338 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3339 = stablehlo.reduce(%v3336 init: %v3335) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3340 = stablehlo.broadcast_in_dim %v3339, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3341 = stablehlo.divide %v3340, %v3337 : tensor<32x128x28x28xf32>
    %v3342 = stablehlo.subtract %v3336, %v3341 : tensor<32x128x28x28xf32>
    %v3343 = stablehlo.multiply %v3342, %v3342 : tensor<32x128x28x28xf32>
    %v3344 = stablehlo.reduce(%v3343 init: %v3335) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %v3345 = stablehlo.broadcast_in_dim %v3344, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %v3346 = stablehlo.divide %v3345, %v3337 : tensor<32x128x28x28xf32>
    %v3347 = stablehlo.add %v3346, %v3338 : tensor<32x128x28x28xf32>
    %v3348 = stablehlo.rsqrt %v3347 : tensor<32x128x28x28xf32>
    %v3349 = stablehlo.multiply %v3342, %v3348 : tensor<32x128x28x28xf32>
    %v3350 = stablehlo.reshape %v3118 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3351 = stablehlo.multiply %v3350, %v3349 : tensor<32x128x28x28xf32>
    %v3352 = stablehlo.reduce(%v3351 init: %v3335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3353 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v3354 = stablehlo.multiply %v3352, %v3353 : tensor<128xf32>
    %v3355 = stablehlo.subtract %d2gp, %v3354 : tensor<128xf32>
    %v3356 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3357 = stablehlo.reshape %v3118 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3358 = stablehlo.reduce(%v3357 init: %v3356) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3359 = stablehlo.constant dense<0.003125> : tensor<128xf32>
    %v3360 = stablehlo.multiply %v3358, %v3359 : tensor<128xf32>
    %v3361 = stablehlo.subtract %d2btp, %v3360 : tensor<128xf32>
    %v3362 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v3363 = stablehlo.compare GT, %v193, %v3362 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v3364 = stablehlo.select %v3363, %v3231, %v3362 : tensor<32x200704xi1>, tensor<32x200704xf32>
    %v3365 = stablehlo.reshape %v3364 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3366 = stablehlo.reshape %v172 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3367 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3368 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v3369 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3370 = stablehlo.reduce(%v3366 init: %v3367) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3371 = stablehlo.broadcast_in_dim %v3370, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3372 = stablehlo.divide %v3371, %v3368 : tensor<32x64x56x56xf32>
    %v3373 = stablehlo.subtract %v3366, %v3372 : tensor<32x64x56x56xf32>
    %v3374 = stablehlo.multiply %v3373, %v3373 : tensor<32x64x56x56xf32>
    %v3375 = stablehlo.reduce(%v3374 init: %v3367) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3376 = stablehlo.broadcast_in_dim %v3375, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3377 = stablehlo.divide %v3376, %v3368 : tensor<32x64x56x56xf32>
    %v3378 = stablehlo.add %v3377, %v3369 : tensor<32x64x56x56xf32>
    %v3379 = stablehlo.rsqrt %v3378 : tensor<32x64x56x56xf32>
    %v3380 = stablehlo.multiply %v3373, %v3379 : tensor<32x64x56x56xf32>
    %v3381 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3382 = stablehlo.multiply %v3381, %v3365 : tensor<32x64x56x56xf32>
    %v3383 = stablehlo.reduce(%v3382 init: %v3367) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3384 = stablehlo.broadcast_in_dim %v3383, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3385 = stablehlo.multiply %v3380, %v3382 : tensor<32x64x56x56xf32>
    %v3386 = stablehlo.reduce(%v3385 init: %v3367) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3387 = stablehlo.broadcast_in_dim %v3386, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3388 = stablehlo.multiply %v3382, %v3368 : tensor<32x64x56x56xf32>
    %v3389 = stablehlo.subtract %v3388, %v3384 : tensor<32x64x56x56xf32>
    %v3390 = stablehlo.multiply %v3380, %v3387 : tensor<32x64x56x56xf32>
    %v3391 = stablehlo.subtract %v3389, %v3390 : tensor<32x64x56x56xf32>
    %v3392 = stablehlo.divide %v3379, %v3368 : tensor<32x64x56x56xf32>
    %v3393 = stablehlo.multiply %v3392, %v3391 : tensor<32x64x56x56xf32>
    %v3394 = stablehlo.reshape %v3393 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3395 = stablehlo.reshape %v3394 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3396 = stablehlo.transpose %s1b2W2, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3397 = stablehlo.reverse %v3396, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3398 = stablehlo.convolution(%v3395, %v3397)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3399 = stablehlo.reshape %v3398 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3400 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v3401 = stablehlo.compare GT, %v165, %v3400 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v3402 = stablehlo.select %v3401, %v3399, %v3400 : tensor<32x200704xi1>, tensor<32x200704xf32>
    %v3403 = stablehlo.reshape %v3402 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3404 = stablehlo.reshape %v145 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3405 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3406 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v3407 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3408 = stablehlo.reduce(%v3404 init: %v3405) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3409 = stablehlo.broadcast_in_dim %v3408, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3410 = stablehlo.divide %v3409, %v3406 : tensor<32x64x56x56xf32>
    %v3411 = stablehlo.subtract %v3404, %v3410 : tensor<32x64x56x56xf32>
    %v3412 = stablehlo.multiply %v3411, %v3411 : tensor<32x64x56x56xf32>
    %v3413 = stablehlo.reduce(%v3412 init: %v3405) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3414 = stablehlo.broadcast_in_dim %v3413, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3415 = stablehlo.divide %v3414, %v3406 : tensor<32x64x56x56xf32>
    %v3416 = stablehlo.add %v3415, %v3407 : tensor<32x64x56x56xf32>
    %v3417 = stablehlo.rsqrt %v3416 : tensor<32x64x56x56xf32>
    %v3418 = stablehlo.multiply %v3411, %v3417 : tensor<32x64x56x56xf32>
    %v3419 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3420 = stablehlo.multiply %v3419, %v3403 : tensor<32x64x56x56xf32>
    %v3421 = stablehlo.reduce(%v3420 init: %v3405) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3422 = stablehlo.broadcast_in_dim %v3421, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3423 = stablehlo.multiply %v3418, %v3420 : tensor<32x64x56x56xf32>
    %v3424 = stablehlo.reduce(%v3423 init: %v3405) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3425 = stablehlo.broadcast_in_dim %v3424, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3426 = stablehlo.multiply %v3420, %v3406 : tensor<32x64x56x56xf32>
    %v3427 = stablehlo.subtract %v3426, %v3422 : tensor<32x64x56x56xf32>
    %v3428 = stablehlo.multiply %v3418, %v3425 : tensor<32x64x56x56xf32>
    %v3429 = stablehlo.subtract %v3427, %v3428 : tensor<32x64x56x56xf32>
    %v3430 = stablehlo.divide %v3417, %v3406 : tensor<32x64x56x56xf32>
    %v3431 = stablehlo.multiply %v3430, %v3429 : tensor<32x64x56x56xf32>
    %v3432 = stablehlo.reshape %v3431 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3433 = stablehlo.reshape %v3432 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3434 = stablehlo.transpose %s1b2W1, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3435 = stablehlo.reverse %v3434, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3436 = stablehlo.convolution(%v3433, %v3435)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3437 = stablehlo.reshape %v3436 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3438 = stablehlo.add %v3437, %v3364 : tensor<32x200704xf32>
    %v3439 = stablehlo.reshape %v140 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3440 = stablehlo.reshape %v3432 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3441 = stablehlo.transpose %v3439, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3442 = stablehlo.transpose %v3440, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3443 = stablehlo.convolution(%v3441, %v3442)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3444 = stablehlo.transpose %v3443, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3445 = stablehlo.constant dense<0.003125> : tensor<64x64x3x3xf32>
    %v3446 = stablehlo.multiply %v3444, %v3445 : tensor<64x64x3x3xf32>
    %v3447 = stablehlo.subtract %s1b2W1, %v3446 : tensor<64x64x3x3xf32>
    %v3448 = stablehlo.reshape %v3432 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3449 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3450 = stablehlo.reduce(%v3448 init: %v3449) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3451 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3452 = stablehlo.multiply %v3450, %v3451 : tensor<64xf32>
    %v3453 = stablehlo.subtract %s1b2b1, %v3452 : tensor<64xf32>
    %v3454 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3455 = stablehlo.reshape %v145 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3456 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v3457 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3458 = stablehlo.reduce(%v3455 init: %v3454) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3459 = stablehlo.broadcast_in_dim %v3458, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3460 = stablehlo.divide %v3459, %v3456 : tensor<32x64x56x56xf32>
    %v3461 = stablehlo.subtract %v3455, %v3460 : tensor<32x64x56x56xf32>
    %v3462 = stablehlo.multiply %v3461, %v3461 : tensor<32x64x56x56xf32>
    %v3463 = stablehlo.reduce(%v3462 init: %v3454) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3464 = stablehlo.broadcast_in_dim %v3463, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3465 = stablehlo.divide %v3464, %v3456 : tensor<32x64x56x56xf32>
    %v3466 = stablehlo.add %v3465, %v3457 : tensor<32x64x56x56xf32>
    %v3467 = stablehlo.rsqrt %v3466 : tensor<32x64x56x56xf32>
    %v3468 = stablehlo.multiply %v3461, %v3467 : tensor<32x64x56x56xf32>
    %v3469 = stablehlo.reshape %v3402 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3470 = stablehlo.multiply %v3469, %v3468 : tensor<32x64x56x56xf32>
    %v3471 = stablehlo.reduce(%v3470 init: %v3454) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3472 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3473 = stablehlo.multiply %v3471, %v3472 : tensor<64xf32>
    %v3474 = stablehlo.subtract %s1b2g1, %v3473 : tensor<64xf32>
    %v3475 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3476 = stablehlo.reshape %v3402 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3477 = stablehlo.reduce(%v3476 init: %v3475) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3478 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3479 = stablehlo.multiply %v3477, %v3478 : tensor<64xf32>
    %v3480 = stablehlo.subtract %s1b2bt1, %v3479 : tensor<64xf32>
    %v3481 = stablehlo.reshape %v167 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3482 = stablehlo.reshape %v3394 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3483 = stablehlo.transpose %v3481, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3484 = stablehlo.transpose %v3482, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3485 = stablehlo.convolution(%v3483, %v3484)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3486 = stablehlo.transpose %v3485, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3487 = stablehlo.constant dense<0.003125> : tensor<64x64x3x3xf32>
    %v3488 = stablehlo.multiply %v3486, %v3487 : tensor<64x64x3x3xf32>
    %v3489 = stablehlo.subtract %s1b2W2, %v3488 : tensor<64x64x3x3xf32>
    %v3490 = stablehlo.reshape %v3394 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3491 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3492 = stablehlo.reduce(%v3490 init: %v3491) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3493 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3494 = stablehlo.multiply %v3492, %v3493 : tensor<64xf32>
    %v3495 = stablehlo.subtract %s1b2b2, %v3494 : tensor<64xf32>
    %v3496 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3497 = stablehlo.reshape %v172 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3498 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v3499 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3500 = stablehlo.reduce(%v3497 init: %v3496) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3501 = stablehlo.broadcast_in_dim %v3500, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3502 = stablehlo.divide %v3501, %v3498 : tensor<32x64x56x56xf32>
    %v3503 = stablehlo.subtract %v3497, %v3502 : tensor<32x64x56x56xf32>
    %v3504 = stablehlo.multiply %v3503, %v3503 : tensor<32x64x56x56xf32>
    %v3505 = stablehlo.reduce(%v3504 init: %v3496) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3506 = stablehlo.broadcast_in_dim %v3505, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3507 = stablehlo.divide %v3506, %v3498 : tensor<32x64x56x56xf32>
    %v3508 = stablehlo.add %v3507, %v3499 : tensor<32x64x56x56xf32>
    %v3509 = stablehlo.rsqrt %v3508 : tensor<32x64x56x56xf32>
    %v3510 = stablehlo.multiply %v3503, %v3509 : tensor<32x64x56x56xf32>
    %v3511 = stablehlo.reshape %v3364 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3512 = stablehlo.multiply %v3511, %v3510 : tensor<32x64x56x56xf32>
    %v3513 = stablehlo.reduce(%v3512 init: %v3496) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3514 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3515 = stablehlo.multiply %v3513, %v3514 : tensor<64xf32>
    %v3516 = stablehlo.subtract %s1b2g2, %v3515 : tensor<64xf32>
    %v3517 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3518 = stablehlo.reshape %v3364 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3519 = stablehlo.reduce(%v3518 init: %v3517) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3520 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3521 = stablehlo.multiply %v3519, %v3520 : tensor<64xf32>
    %v3522 = stablehlo.subtract %s1b2bt2, %v3521 : tensor<64xf32>
    %v3523 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v3524 = stablehlo.compare GT, %v138, %v3523 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v3525 = stablehlo.select %v3524, %v3438, %v3523 : tensor<32x200704xi1>, tensor<32x200704xf32>
    %v3526 = stablehlo.reshape %v3525 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3527 = stablehlo.reshape %v117 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3528 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3529 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v3530 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3531 = stablehlo.reduce(%v3527 init: %v3528) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3532 = stablehlo.broadcast_in_dim %v3531, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3533 = stablehlo.divide %v3532, %v3529 : tensor<32x64x56x56xf32>
    %v3534 = stablehlo.subtract %v3527, %v3533 : tensor<32x64x56x56xf32>
    %v3535 = stablehlo.multiply %v3534, %v3534 : tensor<32x64x56x56xf32>
    %v3536 = stablehlo.reduce(%v3535 init: %v3528) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3537 = stablehlo.broadcast_in_dim %v3536, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3538 = stablehlo.divide %v3537, %v3529 : tensor<32x64x56x56xf32>
    %v3539 = stablehlo.add %v3538, %v3530 : tensor<32x64x56x56xf32>
    %v3540 = stablehlo.rsqrt %v3539 : tensor<32x64x56x56xf32>
    %v3541 = stablehlo.multiply %v3534, %v3540 : tensor<32x64x56x56xf32>
    %v3542 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3543 = stablehlo.multiply %v3542, %v3526 : tensor<32x64x56x56xf32>
    %v3544 = stablehlo.reduce(%v3543 init: %v3528) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3545 = stablehlo.broadcast_in_dim %v3544, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3546 = stablehlo.multiply %v3541, %v3543 : tensor<32x64x56x56xf32>
    %v3547 = stablehlo.reduce(%v3546 init: %v3528) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3548 = stablehlo.broadcast_in_dim %v3547, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3549 = stablehlo.multiply %v3543, %v3529 : tensor<32x64x56x56xf32>
    %v3550 = stablehlo.subtract %v3549, %v3545 : tensor<32x64x56x56xf32>
    %v3551 = stablehlo.multiply %v3541, %v3548 : tensor<32x64x56x56xf32>
    %v3552 = stablehlo.subtract %v3550, %v3551 : tensor<32x64x56x56xf32>
    %v3553 = stablehlo.divide %v3540, %v3529 : tensor<32x64x56x56xf32>
    %v3554 = stablehlo.multiply %v3553, %v3552 : tensor<32x64x56x56xf32>
    %v3555 = stablehlo.reshape %v3554 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3556 = stablehlo.reshape %v3555 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3557 = stablehlo.transpose %s1b1W2, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3558 = stablehlo.reverse %v3557, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3559 = stablehlo.convolution(%v3556, %v3558)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3560 = stablehlo.reshape %v3559 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3561 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v3562 = stablehlo.compare GT, %v110, %v3561 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v3563 = stablehlo.select %v3562, %v3560, %v3561 : tensor<32x200704xi1>, tensor<32x200704xf32>
    %v3564 = stablehlo.reshape %v3563 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3565 = stablehlo.reshape %v90 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3566 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3567 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v3568 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3569 = stablehlo.reduce(%v3565 init: %v3566) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3570 = stablehlo.broadcast_in_dim %v3569, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3571 = stablehlo.divide %v3570, %v3567 : tensor<32x64x56x56xf32>
    %v3572 = stablehlo.subtract %v3565, %v3571 : tensor<32x64x56x56xf32>
    %v3573 = stablehlo.multiply %v3572, %v3572 : tensor<32x64x56x56xf32>
    %v3574 = stablehlo.reduce(%v3573 init: %v3566) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3575 = stablehlo.broadcast_in_dim %v3574, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3576 = stablehlo.divide %v3575, %v3567 : tensor<32x64x56x56xf32>
    %v3577 = stablehlo.add %v3576, %v3568 : tensor<32x64x56x56xf32>
    %v3578 = stablehlo.rsqrt %v3577 : tensor<32x64x56x56xf32>
    %v3579 = stablehlo.multiply %v3572, %v3578 : tensor<32x64x56x56xf32>
    %v3580 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3581 = stablehlo.multiply %v3580, %v3564 : tensor<32x64x56x56xf32>
    %v3582 = stablehlo.reduce(%v3581 init: %v3566) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3583 = stablehlo.broadcast_in_dim %v3582, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3584 = stablehlo.multiply %v3579, %v3581 : tensor<32x64x56x56xf32>
    %v3585 = stablehlo.reduce(%v3584 init: %v3566) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3586 = stablehlo.broadcast_in_dim %v3585, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3587 = stablehlo.multiply %v3581, %v3567 : tensor<32x64x56x56xf32>
    %v3588 = stablehlo.subtract %v3587, %v3583 : tensor<32x64x56x56xf32>
    %v3589 = stablehlo.multiply %v3579, %v3586 : tensor<32x64x56x56xf32>
    %v3590 = stablehlo.subtract %v3588, %v3589 : tensor<32x64x56x56xf32>
    %v3591 = stablehlo.divide %v3578, %v3567 : tensor<32x64x56x56xf32>
    %v3592 = stablehlo.multiply %v3591, %v3590 : tensor<32x64x56x56xf32>
    %v3593 = stablehlo.reshape %v3592 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3594 = stablehlo.reshape %v3593 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3595 = stablehlo.transpose %s1b1W1, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3596 = stablehlo.reverse %v3595, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3597 = stablehlo.convolution(%v3594, %v3596)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3598 = stablehlo.reshape %v3597 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3599 = stablehlo.add %v3598, %v3525 : tensor<32x200704xf32>
    %v3600 = stablehlo.reshape %v85 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3601 = stablehlo.reshape %v3593 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3602 = stablehlo.transpose %v3600, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3603 = stablehlo.transpose %v3601, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3604 = stablehlo.convolution(%v3602, %v3603)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3605 = stablehlo.transpose %v3604, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3606 = stablehlo.constant dense<0.003125> : tensor<64x64x3x3xf32>
    %v3607 = stablehlo.multiply %v3605, %v3606 : tensor<64x64x3x3xf32>
    %v3608 = stablehlo.subtract %s1b1W1, %v3607 : tensor<64x64x3x3xf32>
    %v3609 = stablehlo.reshape %v3593 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3610 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3611 = stablehlo.reduce(%v3609 init: %v3610) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3612 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3613 = stablehlo.multiply %v3611, %v3612 : tensor<64xf32>
    %v3614 = stablehlo.subtract %s1b1b1, %v3613 : tensor<64xf32>
    %v3615 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3616 = stablehlo.reshape %v90 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3617 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v3618 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3619 = stablehlo.reduce(%v3616 init: %v3615) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3620 = stablehlo.broadcast_in_dim %v3619, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3621 = stablehlo.divide %v3620, %v3617 : tensor<32x64x56x56xf32>
    %v3622 = stablehlo.subtract %v3616, %v3621 : tensor<32x64x56x56xf32>
    %v3623 = stablehlo.multiply %v3622, %v3622 : tensor<32x64x56x56xf32>
    %v3624 = stablehlo.reduce(%v3623 init: %v3615) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3625 = stablehlo.broadcast_in_dim %v3624, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3626 = stablehlo.divide %v3625, %v3617 : tensor<32x64x56x56xf32>
    %v3627 = stablehlo.add %v3626, %v3618 : tensor<32x64x56x56xf32>
    %v3628 = stablehlo.rsqrt %v3627 : tensor<32x64x56x56xf32>
    %v3629 = stablehlo.multiply %v3622, %v3628 : tensor<32x64x56x56xf32>
    %v3630 = stablehlo.reshape %v3563 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3631 = stablehlo.multiply %v3630, %v3629 : tensor<32x64x56x56xf32>
    %v3632 = stablehlo.reduce(%v3631 init: %v3615) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3633 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3634 = stablehlo.multiply %v3632, %v3633 : tensor<64xf32>
    %v3635 = stablehlo.subtract %s1b1g1, %v3634 : tensor<64xf32>
    %v3636 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3637 = stablehlo.reshape %v3563 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3638 = stablehlo.reduce(%v3637 init: %v3636) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3639 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3640 = stablehlo.multiply %v3638, %v3639 : tensor<64xf32>
    %v3641 = stablehlo.subtract %s1b1bt1, %v3640 : tensor<64xf32>
    %v3642 = stablehlo.reshape %v112 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3643 = stablehlo.reshape %v3555 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3644 = stablehlo.transpose %v3642, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3645 = stablehlo.transpose %v3643, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3646 = stablehlo.convolution(%v3644, %v3645)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3647 = stablehlo.transpose %v3646, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3648 = stablehlo.constant dense<0.003125> : tensor<64x64x3x3xf32>
    %v3649 = stablehlo.multiply %v3647, %v3648 : tensor<64x64x3x3xf32>
    %v3650 = stablehlo.subtract %s1b1W2, %v3649 : tensor<64x64x3x3xf32>
    %v3651 = stablehlo.reshape %v3555 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3652 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3653 = stablehlo.reduce(%v3651 init: %v3652) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3654 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3655 = stablehlo.multiply %v3653, %v3654 : tensor<64xf32>
    %v3656 = stablehlo.subtract %s1b1b2, %v3655 : tensor<64xf32>
    %v3657 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3658 = stablehlo.reshape %v117 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3659 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v3660 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3661 = stablehlo.reduce(%v3658 init: %v3657) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3662 = stablehlo.broadcast_in_dim %v3661, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3663 = stablehlo.divide %v3662, %v3659 : tensor<32x64x56x56xf32>
    %v3664 = stablehlo.subtract %v3658, %v3663 : tensor<32x64x56x56xf32>
    %v3665 = stablehlo.multiply %v3664, %v3664 : tensor<32x64x56x56xf32>
    %v3666 = stablehlo.reduce(%v3665 init: %v3657) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3667 = stablehlo.broadcast_in_dim %v3666, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3668 = stablehlo.divide %v3667, %v3659 : tensor<32x64x56x56xf32>
    %v3669 = stablehlo.add %v3668, %v3660 : tensor<32x64x56x56xf32>
    %v3670 = stablehlo.rsqrt %v3669 : tensor<32x64x56x56xf32>
    %v3671 = stablehlo.multiply %v3664, %v3670 : tensor<32x64x56x56xf32>
    %v3672 = stablehlo.reshape %v3525 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3673 = stablehlo.multiply %v3672, %v3671 : tensor<32x64x56x56xf32>
    %v3674 = stablehlo.reduce(%v3673 init: %v3657) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3675 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3676 = stablehlo.multiply %v3674, %v3675 : tensor<64xf32>
    %v3677 = stablehlo.subtract %s1b1g2, %v3676 : tensor<64xf32>
    %v3678 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3679 = stablehlo.reshape %v3525 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3680 = stablehlo.reduce(%v3679 init: %v3678) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3681 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3682 = stablehlo.multiply %v3680, %v3681 : tensor<64xf32>
    %v3683 = stablehlo.subtract %s1b1bt2, %v3682 : tensor<64xf32>
    %v3684 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v3685 = stablehlo.compare GT, %v83, %v3684 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v3686 = stablehlo.select %v3685, %v3599, %v3684 : tensor<32x200704xi1>, tensor<32x200704xf32>
    %v3687 = stablehlo.reshape %v3686 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3688 = stablehlo.reshape %v62 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3689 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3690 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v3691 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3692 = stablehlo.reduce(%v3688 init: %v3689) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3693 = stablehlo.broadcast_in_dim %v3692, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3694 = stablehlo.divide %v3693, %v3690 : tensor<32x64x56x56xf32>
    %v3695 = stablehlo.subtract %v3688, %v3694 : tensor<32x64x56x56xf32>
    %v3696 = stablehlo.multiply %v3695, %v3695 : tensor<32x64x56x56xf32>
    %v3697 = stablehlo.reduce(%v3696 init: %v3689) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3698 = stablehlo.broadcast_in_dim %v3697, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3699 = stablehlo.divide %v3698, %v3690 : tensor<32x64x56x56xf32>
    %v3700 = stablehlo.add %v3699, %v3691 : tensor<32x64x56x56xf32>
    %v3701 = stablehlo.rsqrt %v3700 : tensor<32x64x56x56xf32>
    %v3702 = stablehlo.multiply %v3695, %v3701 : tensor<32x64x56x56xf32>
    %v3703 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3704 = stablehlo.multiply %v3703, %v3687 : tensor<32x64x56x56xf32>
    %v3705 = stablehlo.reduce(%v3704 init: %v3689) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3706 = stablehlo.broadcast_in_dim %v3705, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3707 = stablehlo.multiply %v3702, %v3704 : tensor<32x64x56x56xf32>
    %v3708 = stablehlo.reduce(%v3707 init: %v3689) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3709 = stablehlo.broadcast_in_dim %v3708, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3710 = stablehlo.multiply %v3704, %v3690 : tensor<32x64x56x56xf32>
    %v3711 = stablehlo.subtract %v3710, %v3706 : tensor<32x64x56x56xf32>
    %v3712 = stablehlo.multiply %v3702, %v3709 : tensor<32x64x56x56xf32>
    %v3713 = stablehlo.subtract %v3711, %v3712 : tensor<32x64x56x56xf32>
    %v3714 = stablehlo.divide %v3701, %v3690 : tensor<32x64x56x56xf32>
    %v3715 = stablehlo.multiply %v3714, %v3713 : tensor<32x64x56x56xf32>
    %v3716 = stablehlo.reshape %v3715 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3717 = stablehlo.reshape %v3716 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3718 = stablehlo.transpose %s1b0W2, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3719 = stablehlo.reverse %v3718, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3720 = stablehlo.convolution(%v3717, %v3719)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3721 = stablehlo.reshape %v3720 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3722 = stablehlo.constant dense<0.0> : tensor<32x200704xf32>
    %v3723 = stablehlo.compare GT, %v55, %v3722 : (tensor<32x200704xf32>, tensor<32x200704xf32>) -> tensor<32x200704xi1>
    %v3724 = stablehlo.select %v3723, %v3721, %v3722 : tensor<32x200704xi1>, tensor<32x200704xf32>
    %v3725 = stablehlo.reshape %v3724 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3726 = stablehlo.reshape %v35 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3727 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3728 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v3729 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3730 = stablehlo.reduce(%v3726 init: %v3727) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3731 = stablehlo.broadcast_in_dim %v3730, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3732 = stablehlo.divide %v3731, %v3728 : tensor<32x64x56x56xf32>
    %v3733 = stablehlo.subtract %v3726, %v3732 : tensor<32x64x56x56xf32>
    %v3734 = stablehlo.multiply %v3733, %v3733 : tensor<32x64x56x56xf32>
    %v3735 = stablehlo.reduce(%v3734 init: %v3727) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3736 = stablehlo.broadcast_in_dim %v3735, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3737 = stablehlo.divide %v3736, %v3728 : tensor<32x64x56x56xf32>
    %v3738 = stablehlo.add %v3737, %v3729 : tensor<32x64x56x56xf32>
    %v3739 = stablehlo.rsqrt %v3738 : tensor<32x64x56x56xf32>
    %v3740 = stablehlo.multiply %v3733, %v3739 : tensor<32x64x56x56xf32>
    %v3741 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3742 = stablehlo.multiply %v3741, %v3725 : tensor<32x64x56x56xf32>
    %v3743 = stablehlo.reduce(%v3742 init: %v3727) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3744 = stablehlo.broadcast_in_dim %v3743, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3745 = stablehlo.multiply %v3740, %v3742 : tensor<32x64x56x56xf32>
    %v3746 = stablehlo.reduce(%v3745 init: %v3727) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3747 = stablehlo.broadcast_in_dim %v3746, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3748 = stablehlo.multiply %v3742, %v3728 : tensor<32x64x56x56xf32>
    %v3749 = stablehlo.subtract %v3748, %v3744 : tensor<32x64x56x56xf32>
    %v3750 = stablehlo.multiply %v3740, %v3747 : tensor<32x64x56x56xf32>
    %v3751 = stablehlo.subtract %v3749, %v3750 : tensor<32x64x56x56xf32>
    %v3752 = stablehlo.divide %v3739, %v3728 : tensor<32x64x56x56xf32>
    %v3753 = stablehlo.multiply %v3752, %v3751 : tensor<32x64x56x56xf32>
    %v3754 = stablehlo.reshape %v3753 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3755 = stablehlo.reshape %v3754 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3756 = stablehlo.transpose %s1b0W1, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3757 = stablehlo.reverse %v3756, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3758 = stablehlo.convolution(%v3755, %v3757)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3759 = stablehlo.reshape %v3758 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3760 = stablehlo.add %v3759, %v3686 : tensor<32x200704xf32>
    %v3761 = stablehlo.reshape %v30 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3762 = stablehlo.reshape %v3754 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3763 = stablehlo.transpose %v3761, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3764 = stablehlo.transpose %v3762, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3765 = stablehlo.convolution(%v3763, %v3764)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3766 = stablehlo.transpose %v3765, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3767 = stablehlo.constant dense<0.003125> : tensor<64x64x3x3xf32>
    %v3768 = stablehlo.multiply %v3766, %v3767 : tensor<64x64x3x3xf32>
    %v3769 = stablehlo.subtract %s1b0W1, %v3768 : tensor<64x64x3x3xf32>
    %v3770 = stablehlo.reshape %v3754 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3771 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3772 = stablehlo.reduce(%v3770 init: %v3771) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3773 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3774 = stablehlo.multiply %v3772, %v3773 : tensor<64xf32>
    %v3775 = stablehlo.subtract %s1b0b1, %v3774 : tensor<64xf32>
    %v3776 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3777 = stablehlo.reshape %v35 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3778 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v3779 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3780 = stablehlo.reduce(%v3777 init: %v3776) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3781 = stablehlo.broadcast_in_dim %v3780, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3782 = stablehlo.divide %v3781, %v3778 : tensor<32x64x56x56xf32>
    %v3783 = stablehlo.subtract %v3777, %v3782 : tensor<32x64x56x56xf32>
    %v3784 = stablehlo.multiply %v3783, %v3783 : tensor<32x64x56x56xf32>
    %v3785 = stablehlo.reduce(%v3784 init: %v3776) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3786 = stablehlo.broadcast_in_dim %v3785, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3787 = stablehlo.divide %v3786, %v3778 : tensor<32x64x56x56xf32>
    %v3788 = stablehlo.add %v3787, %v3779 : tensor<32x64x56x56xf32>
    %v3789 = stablehlo.rsqrt %v3788 : tensor<32x64x56x56xf32>
    %v3790 = stablehlo.multiply %v3783, %v3789 : tensor<32x64x56x56xf32>
    %v3791 = stablehlo.reshape %v3724 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3792 = stablehlo.multiply %v3791, %v3790 : tensor<32x64x56x56xf32>
    %v3793 = stablehlo.reduce(%v3792 init: %v3776) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3794 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3795 = stablehlo.multiply %v3793, %v3794 : tensor<64xf32>
    %v3796 = stablehlo.subtract %s1b0g1, %v3795 : tensor<64xf32>
    %v3797 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3798 = stablehlo.reshape %v3724 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3799 = stablehlo.reduce(%v3798 init: %v3797) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3800 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3801 = stablehlo.multiply %v3799, %v3800 : tensor<64xf32>
    %v3802 = stablehlo.subtract %s1b0bt1, %v3801 : tensor<64xf32>
    %v3803 = stablehlo.reshape %v57 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3804 = stablehlo.reshape %v3716 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3805 = stablehlo.transpose %v3803, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3806 = stablehlo.transpose %v3804, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3807 = stablehlo.convolution(%v3805, %v3806)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3808 = stablehlo.transpose %v3807, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3809 = stablehlo.constant dense<0.003125> : tensor<64x64x3x3xf32>
    %v3810 = stablehlo.multiply %v3808, %v3809 : tensor<64x64x3x3xf32>
    %v3811 = stablehlo.subtract %s1b0W2, %v3810 : tensor<64x64x3x3xf32>
    %v3812 = stablehlo.reshape %v3716 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3813 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3814 = stablehlo.reduce(%v3812 init: %v3813) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3815 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3816 = stablehlo.multiply %v3814, %v3815 : tensor<64xf32>
    %v3817 = stablehlo.subtract %s1b0b2, %v3816 : tensor<64xf32>
    %v3818 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3819 = stablehlo.reshape %v62 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3820 = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %v3821 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3822 = stablehlo.reduce(%v3819 init: %v3818) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3823 = stablehlo.broadcast_in_dim %v3822, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3824 = stablehlo.divide %v3823, %v3820 : tensor<32x64x56x56xf32>
    %v3825 = stablehlo.subtract %v3819, %v3824 : tensor<32x64x56x56xf32>
    %v3826 = stablehlo.multiply %v3825, %v3825 : tensor<32x64x56x56xf32>
    %v3827 = stablehlo.reduce(%v3826 init: %v3818) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3828 = stablehlo.broadcast_in_dim %v3827, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %v3829 = stablehlo.divide %v3828, %v3820 : tensor<32x64x56x56xf32>
    %v3830 = stablehlo.add %v3829, %v3821 : tensor<32x64x56x56xf32>
    %v3831 = stablehlo.rsqrt %v3830 : tensor<32x64x56x56xf32>
    %v3832 = stablehlo.multiply %v3825, %v3831 : tensor<32x64x56x56xf32>
    %v3833 = stablehlo.reshape %v3686 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3834 = stablehlo.multiply %v3833, %v3832 : tensor<32x64x56x56xf32>
    %v3835 = stablehlo.reduce(%v3834 init: %v3818) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3836 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3837 = stablehlo.multiply %v3835, %v3836 : tensor<64xf32>
    %v3838 = stablehlo.subtract %s1b0g2, %v3837 : tensor<64xf32>
    %v3839 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3840 = stablehlo.reshape %v3686 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3841 = stablehlo.reduce(%v3840 init: %v3839) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3842 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3843 = stablehlo.multiply %v3841, %v3842 : tensor<64xf32>
    %v3844 = stablehlo.subtract %s1b0bt2, %v3843 : tensor<64xf32>
    %v3845 = stablehlo.reshape %v26 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3846 = stablehlo.reshape %v3760 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3847 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3848 = "stablehlo.select_and_scatter"(%v3845, %v3846, %v3847) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<32x64x112x112xf32>, tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x112x112xf32>
    %v3849 = stablehlo.reshape %v3848 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v3850 = stablehlo.constant dense<0.0> : tensor<32x802816xf32>
    %v3851 = stablehlo.compare GT, %v24, %v3850 : (tensor<32x802816xf32>, tensor<32x802816xf32>) -> tensor<32x802816xi1>
    %v3852 = stablehlo.select %v3851, %v3849, %v3850 : tensor<32x802816xi1>, tensor<32x802816xf32>
    %v3853 = stablehlo.reshape %v3852 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3854 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3855 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3856 = stablehlo.constant dense<12544.0> : tensor<32x64x112x112xf32>
    %v3857 = stablehlo.constant dense<1.0e-05> : tensor<32x64x112x112xf32>
    %v3858 = stablehlo.reduce(%v3854 init: %v3855) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3859 = stablehlo.broadcast_in_dim %v3858, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v3860 = stablehlo.divide %v3859, %v3856 : tensor<32x64x112x112xf32>
    %v3861 = stablehlo.subtract %v3854, %v3860 : tensor<32x64x112x112xf32>
    %v3862 = stablehlo.multiply %v3861, %v3861 : tensor<32x64x112x112xf32>
    %v3863 = stablehlo.reduce(%v3862 init: %v3855) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3864 = stablehlo.broadcast_in_dim %v3863, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v3865 = stablehlo.divide %v3864, %v3856 : tensor<32x64x112x112xf32>
    %v3866 = stablehlo.add %v3865, %v3857 : tensor<32x64x112x112xf32>
    %v3867 = stablehlo.rsqrt %v3866 : tensor<32x64x112x112xf32>
    %v3868 = stablehlo.multiply %v3861, %v3867 : tensor<32x64x112x112xf32>
    %v3869 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3870 = stablehlo.multiply %v3869, %v3853 : tensor<32x64x112x112xf32>
    %v3871 = stablehlo.reduce(%v3870 init: %v3855) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3872 = stablehlo.broadcast_in_dim %v3871, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v3873 = stablehlo.multiply %v3868, %v3870 : tensor<32x64x112x112xf32>
    %v3874 = stablehlo.reduce(%v3873 init: %v3855) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3875 = stablehlo.broadcast_in_dim %v3874, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v3876 = stablehlo.multiply %v3870, %v3856 : tensor<32x64x112x112xf32>
    %v3877 = stablehlo.subtract %v3876, %v3872 : tensor<32x64x112x112xf32>
    %v3878 = stablehlo.multiply %v3868, %v3875 : tensor<32x64x112x112xf32>
    %v3879 = stablehlo.subtract %v3877, %v3878 : tensor<32x64x112x112xf32>
    %v3880 = stablehlo.divide %v3867, %v3856 : tensor<32x64x112x112xf32>
    %v3881 = stablehlo.multiply %v3880, %v3879 : tensor<32x64x112x112xf32>
    %v3882 = stablehlo.reshape %v3881 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v3883 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v3884 = stablehlo.reshape %v3882 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3885 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3886 = stablehlo.pad %v3884, %v3885, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64x224x224xf32>
    %v3887 = stablehlo.transpose %v3883, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %v3888 = stablehlo.transpose %v3886, dims = [1, 0, 2, 3] : (tensor<32x64x224x224xf32>) -> tensor<64x32x224x224xf32>
    %v3889 = stablehlo.convolution(%v3887, %v3888)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<64x32x224x224xf32>) -> tensor<3x64x7x7xf32>
    %v3890 = stablehlo.transpose %v3889, dims = [1, 0, 2, 3] : (tensor<3x64x7x7xf32>) -> tensor<64x3x7x7xf32>
    %v3891 = stablehlo.constant dense<0.003125> : tensor<64x3x7x7xf32>
    %v3892 = stablehlo.multiply %v3890, %v3891 : tensor<64x3x7x7xf32>
    %v3893 = stablehlo.subtract %sW, %v3892 : tensor<64x3x7x7xf32>
    %v3894 = stablehlo.reshape %v3882 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3895 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3896 = stablehlo.reduce(%v3894 init: %v3895) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3897 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3898 = stablehlo.multiply %v3896, %v3897 : tensor<64xf32>
    %v3899 = stablehlo.subtract %sbi, %v3898 : tensor<64xf32>
    %v3900 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3901 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3902 = stablehlo.constant dense<12544.0> : tensor<32x64x112x112xf32>
    %v3903 = stablehlo.constant dense<1.0e-05> : tensor<32x64x112x112xf32>
    %v3904 = stablehlo.reduce(%v3901 init: %v3900) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3905 = stablehlo.broadcast_in_dim %v3904, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v3906 = stablehlo.divide %v3905, %v3902 : tensor<32x64x112x112xf32>
    %v3907 = stablehlo.subtract %v3901, %v3906 : tensor<32x64x112x112xf32>
    %v3908 = stablehlo.multiply %v3907, %v3907 : tensor<32x64x112x112xf32>
    %v3909 = stablehlo.reduce(%v3908 init: %v3900) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %v3910 = stablehlo.broadcast_in_dim %v3909, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %v3911 = stablehlo.divide %v3910, %v3902 : tensor<32x64x112x112xf32>
    %v3912 = stablehlo.add %v3911, %v3903 : tensor<32x64x112x112xf32>
    %v3913 = stablehlo.rsqrt %v3912 : tensor<32x64x112x112xf32>
    %v3914 = stablehlo.multiply %v3907, %v3913 : tensor<32x64x112x112xf32>
    %v3915 = stablehlo.reshape %v3852 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3916 = stablehlo.multiply %v3915, %v3914 : tensor<32x64x112x112xf32>
    %v3917 = stablehlo.reduce(%v3916 init: %v3900) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3918 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3919 = stablehlo.multiply %v3917, %v3918 : tensor<64xf32>
    %v3920 = stablehlo.subtract %sg, %v3919 : tensor<64xf32>
    %v3921 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3922 = stablehlo.reshape %v3852 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3923 = stablehlo.reduce(%v3922 init: %v3921) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3924 = stablehlo.constant dense<0.003125> : tensor<64xf32>
    %v3925 = stablehlo.multiply %v3923, %v3924 : tensor<64xf32>
    %v3926 = stablehlo.subtract %sbt, %v3925 : tensor<64xf32>
    return %v3893, %v3899, %v3920, %v3926, %v3769, %v3775, %v3796, %v3802, %v3811, %v3817, %v3838, %v3844, %v3608, %v3614, %v3635, %v3641, %v3650, %v3656, %v3677, %v3683, %v3447, %v3453, %v3474, %v3480, %v3489, %v3495, %v3516, %v3522, %v3242, %v3248, %v3269, %v3275, %v3284, %v3290, %v3311, %v3317, %v3328, %v3334, %v3355, %v3361, %v3040, %v3046, %v3067, %v3073, %v3082, %v3088, %v3109, %v3115, %v2879, %v2885, %v2906, %v2912, %v2921, %v2927, %v2948, %v2954, %v2718, %v2724, %v2745, %v2751, %v2760, %v2766, %v2787, %v2793, %v2513, %v2519, %v2540, %v2546, %v2555, %v2561, %v2582, %v2588, %v2599, %v2605, %v2626, %v2632, %v2311, %v2317, %v2338, %v2344, %v2353, %v2359, %v2380, %v2386, %v2150, %v2156, %v2177, %v2183, %v2192, %v2198, %v2219, %v2225, %v1989, %v1995, %v2016, %v2022, %v2031, %v2037, %v2058, %v2064, %v1828, %v1834, %v1855, %v1861, %v1870, %v1876, %v1897, %v1903, %v1667, %v1673, %v1694, %v1700, %v1709, %v1715, %v1736, %v1742, %v1462, %v1468, %v1489, %v1495, %v1504, %v1510, %v1531, %v1537, %v1548, %v1554, %v1575, %v1581, %v1260, %v1266, %v1287, %v1293, %v1302, %v1308, %v1329, %v1335, %v1099, %v1105, %v1126, %v1132, %v1141, %v1147, %v1168, %v1174, %v1008, %v1013 : tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>
  }
}
