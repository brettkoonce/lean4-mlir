module @m {
  func.func @mnv4in_fwd(%x: tensor<64x150528xf32>, %sW: tensor<32x3x3x3xf32>, %sg: tensor<32xf32>, %sbt: tensor<32xf32>, %f0cW: tensor<128x32x3x3xf32>, %f0cg: tensor<128xf32>, %f0cbt: tensor<128xf32>, %f0pW: tensor<48x128x1x1xf32>, %f0pg: tensor<48xf32>, %f0pbt: tensor<48xf32>, %u1qW: tensor<48x1x3x3xf32>, %u1qg: tensor<48xf32>, %u1qbt: tensor<48xf32>, %u1eW: tensor<192x48x1x1xf32>, %u1eg: tensor<192xf32>, %u1ebt: tensor<192xf32>, %u1dW: tensor<192x1x5x5xf32>, %u1dg: tensor<192xf32>, %u1dbt: tensor<192xf32>, %u1pW: tensor<80x192x1x1xf32>, %u1pg: tensor<80xf32>, %u1pbt: tensor<80xf32>, %u2qW: tensor<80x1x3x3xf32>, %u2qg: tensor<80xf32>, %u2qbt: tensor<80xf32>, %u2eW: tensor<160x80x1x1xf32>, %u2eg: tensor<160xf32>, %u2ebt: tensor<160xf32>, %u2dW: tensor<160x1x3x3xf32>, %u2dg: tensor<160xf32>, %u2dbt: tensor<160xf32>, %u2pW: tensor<80x160x1x1xf32>, %u2pg: tensor<80xf32>, %u2pbt: tensor<80xf32>, %u3eW: tensor<480x80x1x1xf32>, %u3eg: tensor<480xf32>, %u3ebt: tensor<480xf32>, %u3dW: tensor<480x1x3x3xf32>, %u3dg: tensor<480xf32>, %u3dbt: tensor<480xf32>, %u3pW: tensor<160x480x1x1xf32>, %u3pg: tensor<160xf32>, %u3pbt: tensor<160xf32>, %u4qW: tensor<160x1x3x3xf32>, %u4qg: tensor<160xf32>, %u4qbt: tensor<160xf32>, %u4eW: tensor<640x160x1x1xf32>, %u4eg: tensor<640xf32>, %u4ebt: tensor<640xf32>, %u4dW: tensor<640x1x3x3xf32>, %u4dg: tensor<640xf32>, %u4dbt: tensor<640xf32>, %u4pW: tensor<160x640x1x1xf32>, %u4pg: tensor<160xf32>, %u4pbt: tensor<160xf32>, %u5qW: tensor<160x1x3x3xf32>, %u5qg: tensor<160xf32>, %u5qbt: tensor<160xf32>, %u5eW: tensor<640x160x1x1xf32>, %u5eg: tensor<640xf32>, %u5ebt: tensor<640xf32>, %u5dW: tensor<640x1x5x5xf32>, %u5dg: tensor<640xf32>, %u5dbt: tensor<640xf32>, %u5pW: tensor<160x640x1x1xf32>, %u5pg: tensor<160xf32>, %u5pbt: tensor<160xf32>, %u6qW: tensor<160x1x5x5xf32>, %u6qg: tensor<160xf32>, %u6qbt: tensor<160xf32>, %u6eW: tensor<640x160x1x1xf32>, %u6eg: tensor<640xf32>, %u6ebt: tensor<640xf32>, %u6pW: tensor<160x640x1x1xf32>, %u6pg: tensor<160xf32>, %u6pbt: tensor<160xf32>, %u7eW: tensor<640x160x1x1xf32>, %u7eg: tensor<640xf32>, %u7ebt: tensor<640xf32>, %u7dW: tensor<640x1x3x3xf32>, %u7dg: tensor<640xf32>, %u7dbt: tensor<640xf32>, %u7pW: tensor<160x640x1x1xf32>, %u7pg: tensor<160xf32>, %u7pbt: tensor<160xf32>, %u8qW: tensor<160x1x3x3xf32>, %u8qg: tensor<160xf32>, %u8qbt: tensor<160xf32>, %u8eW: tensor<640x160x1x1xf32>, %u8eg: tensor<640xf32>, %u8ebt: tensor<640xf32>, %u8pW: tensor<160x640x1x1xf32>, %u8pg: tensor<160xf32>, %u8pbt: tensor<160xf32>, %u9eW: tensor<640x160x1x1xf32>, %u9eg: tensor<640xf32>, %u9ebt: tensor<640xf32>, %u9pW: tensor<160x640x1x1xf32>, %u9pg: tensor<160xf32>, %u9pbt: tensor<160xf32>, %u10qW: tensor<160x1x3x3xf32>, %u10qg: tensor<160xf32>, %u10qbt: tensor<160xf32>, %u10eW: tensor<640x160x1x1xf32>, %u10eg: tensor<640xf32>, %u10ebt: tensor<640xf32>, %u10dW: tensor<640x1x3x3xf32>, %u10dg: tensor<640xf32>, %u10dbt: tensor<640xf32>, %u10pW: tensor<160x640x1x1xf32>, %u10pg: tensor<160xf32>, %u10pbt: tensor<160xf32>, %u11qW: tensor<160x1x5x5xf32>, %u11qg: tensor<160xf32>, %u11qbt: tensor<160xf32>, %u11eW: tensor<960x160x1x1xf32>, %u11eg: tensor<960xf32>, %u11ebt: tensor<960xf32>, %u11dW: tensor<960x1x5x5xf32>, %u11dg: tensor<960xf32>, %u11dbt: tensor<960xf32>, %u11pW: tensor<256x960x1x1xf32>, %u11pg: tensor<256xf32>, %u11pbt: tensor<256xf32>, %u12qW: tensor<256x1x5x5xf32>, %u12qg: tensor<256xf32>, %u12qbt: tensor<256xf32>, %u12eW: tensor<1024x256x1x1xf32>, %u12eg: tensor<1024xf32>, %u12ebt: tensor<1024xf32>, %u12dW: tensor<1024x1x5x5xf32>, %u12dg: tensor<1024xf32>, %u12dbt: tensor<1024xf32>, %u12pW: tensor<256x1024x1x1xf32>, %u12pg: tensor<256xf32>, %u12pbt: tensor<256xf32>, %u13eW: tensor<1024x256x1x1xf32>, %u13eg: tensor<1024xf32>, %u13ebt: tensor<1024xf32>, %u13dW: tensor<1024x1x3x3xf32>, %u13dg: tensor<1024xf32>, %u13dbt: tensor<1024xf32>, %u13pW: tensor<256x1024x1x1xf32>, %u13pg: tensor<256xf32>, %u13pbt: tensor<256xf32>, %u14qW: tensor<256x1x3x3xf32>, %u14qg: tensor<256xf32>, %u14qbt: tensor<256xf32>, %u14eW: tensor<1024x256x1x1xf32>, %u14eg: tensor<1024xf32>, %u14ebt: tensor<1024xf32>, %u14pW: tensor<256x1024x1x1xf32>, %u14pg: tensor<256xf32>, %u14pbt: tensor<256xf32>, %hW: tensor<1280x256x1x1xf32>, %hg: tensor<1280xf32>, %hbt: tensor<1280xf32>, %Wd: tensor<1280x1000xf32>, %bd: tensor<1000xf32>) -> tensor<64x1000xf32> {
    // ── MobileNetV4-Conv-S forward: every line is pretty(verified AST node) ──
    // §2l step B: the conv biases are gone from the signature (BN removes them; He et al.'s
    // `.convBn` has none). The proven conv ops still take a bias operand, so it is bound to a
    // zero constant here — same op, `bias = 0`, and `x + 0.0` is exact.
    %zb32 = stablehlo.constant dense<0.0> : tensor<32xf32>
    %zb48 = stablehlo.constant dense<0.0> : tensor<48xf32>
    %zb80 = stablehlo.constant dense<0.0> : tensor<80xf32>
    %zb128 = stablehlo.constant dense<0.0> : tensor<128xf32>
    %zb160 = stablehlo.constant dense<0.0> : tensor<160xf32>
    %zb192 = stablehlo.constant dense<0.0> : tensor<192xf32>
    %zb256 = stablehlo.constant dense<0.0> : tensor<256xf32>
    %zb480 = stablehlo.constant dense<0.0> : tensor<480xf32>
    %zb640 = stablehlo.constant dense<0.0> : tensor<640xf32>
    %zb960 = stablehlo.constant dense<0.0> : tensor<960xf32>
    %zb1024 = stablehlo.constant dense<0.0> : tensor<1024xf32>
    %zb1280 = stablehlo.constant dense<0.0> : tensor<1280xf32>
    %v0 = stablehlo.reshape %x : (tensor<64x150528xf32>) -> tensor<64x3x224x224xf32>
    %v1 = stablehlo.convolution(%v0, %sW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x3x224x224xf32>, tensor<32x3x3x3xf32>) -> tensor<64x32x112x112xf32>
    %v2 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<64x32x112x112xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<64x32x112x112xf32>) -> tensor<64x401408xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<64x401408xf32>) -> tensor<64x32x112x112xf32>
    %v6 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7 = stablehlo.constant dense<802816.0> : tensor<64x32x112x112xf32>
    %v8 = stablehlo.constant dense<1.0e-5> : tensor<64x32x112x112xf32>
    %v9 = stablehlo.reduce(%v5 init: %v6) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v10 = stablehlo.broadcast_in_dim %v9, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v11 = stablehlo.divide %v10, %v7 : tensor<64x32x112x112xf32>
    %v12 = stablehlo.subtract %v5, %v11 : tensor<64x32x112x112xf32>
    %v13 = stablehlo.multiply %v12, %v12 : tensor<64x32x112x112xf32>
    %v14 = stablehlo.reduce(%v13 init: %v6) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v15 = stablehlo.broadcast_in_dim %v14, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v16 = stablehlo.divide %v15, %v7 : tensor<64x32x112x112xf32>
    %v17 = stablehlo.add %v16, %v8 : tensor<64x32x112x112xf32>
    %v18 = stablehlo.rsqrt %v17 : tensor<64x32x112x112xf32>
    %v19 = stablehlo.multiply %v12, %v18 : tensor<64x32x112x112xf32>
    %v20 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v21 = stablehlo.broadcast_in_dim %sbt, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v22 = stablehlo.multiply %v19, %v20 : tensor<64x32x112x112xf32>
    %v23 = stablehlo.add %v22, %v21 : tensor<64x32x112x112xf32>
    %v24 = stablehlo.reshape %v23 : (tensor<64x32x112x112xf32>) -> tensor<64x401408xf32>
    %v25 = stablehlo.constant dense<0.0> : tensor<64x401408xf32>
    %v26 = stablehlo.maximum %v24, %v25 : tensor<64x401408xf32>
    %v27 = stablehlo.reshape %v26 : (tensor<64x401408xf32>) -> tensor<64x32x112x112xf32>
    %v28 = stablehlo.convolution(%v27, %f0cW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x112x112xf32>, tensor<128x32x3x3xf32>) -> tensor<64x128x56x56xf32>
    %v29 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v30 = stablehlo.add %v28, %v29 : tensor<64x128x56x56xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v32 = stablehlo.reshape %v31 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v33 = stablehlo.constant dense<0.0> : tensor<f32>
    %v34 = stablehlo.constant dense<200704.0> : tensor<64x128x56x56xf32>
    %v35 = stablehlo.constant dense<1.0e-5> : tensor<64x128x56x56xf32>
    %v36 = stablehlo.reduce(%v32 init: %v33) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v37 = stablehlo.broadcast_in_dim %v36, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v38 = stablehlo.divide %v37, %v34 : tensor<64x128x56x56xf32>
    %v39 = stablehlo.subtract %v32, %v38 : tensor<64x128x56x56xf32>
    %v40 = stablehlo.multiply %v39, %v39 : tensor<64x128x56x56xf32>
    %v41 = stablehlo.reduce(%v40 init: %v33) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v42 = stablehlo.broadcast_in_dim %v41, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v43 = stablehlo.divide %v42, %v34 : tensor<64x128x56x56xf32>
    %v44 = stablehlo.add %v43, %v35 : tensor<64x128x56x56xf32>
    %v45 = stablehlo.rsqrt %v44 : tensor<64x128x56x56xf32>
    %v46 = stablehlo.multiply %v39, %v45 : tensor<64x128x56x56xf32>
    %v47 = stablehlo.broadcast_in_dim %f0cg, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v48 = stablehlo.broadcast_in_dim %f0cbt, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v49 = stablehlo.multiply %v46, %v47 : tensor<64x128x56x56xf32>
    %v50 = stablehlo.add %v49, %v48 : tensor<64x128x56x56xf32>
    %v51 = stablehlo.reshape %v50 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v52 = stablehlo.logistic %v51 : tensor<64x401408xf32>
    %v53 = stablehlo.multiply %v51, %v52 : tensor<64x401408xf32>
    %v54 = stablehlo.reshape %v53 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v55 = stablehlo.convolution(%v54, %f0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x56x56xf32>, tensor<48x128x1x1xf32>) -> tensor<64x48x56x56xf32>
    %v56 = stablehlo.broadcast_in_dim %zb48, dims = [1] : (tensor<48xf32>) -> tensor<64x48x56x56xf32>
    %v57 = stablehlo.add %v55, %v56 : tensor<64x48x56x56xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<64x48x56x56xf32>) -> tensor<64x150528xf32>
    %v59 = stablehlo.reshape %v58 : (tensor<64x150528xf32>) -> tensor<64x48x56x56xf32>
    %v60 = stablehlo.constant dense<0.0> : tensor<f32>
    %v61 = stablehlo.constant dense<200704.0> : tensor<64x48x56x56xf32>
    %v62 = stablehlo.constant dense<1.0e-5> : tensor<64x48x56x56xf32>
    %v63 = stablehlo.reduce(%v59 init: %v60) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x48x56x56xf32>, tensor<f32>) -> tensor<48xf32>
    %v64 = stablehlo.broadcast_in_dim %v63, dims = [1] : (tensor<48xf32>) -> tensor<64x48x56x56xf32>
    %v65 = stablehlo.divide %v64, %v61 : tensor<64x48x56x56xf32>
    %v66 = stablehlo.subtract %v59, %v65 : tensor<64x48x56x56xf32>
    %v67 = stablehlo.multiply %v66, %v66 : tensor<64x48x56x56xf32>
    %v68 = stablehlo.reduce(%v67 init: %v60) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x48x56x56xf32>, tensor<f32>) -> tensor<48xf32>
    %v69 = stablehlo.broadcast_in_dim %v68, dims = [1] : (tensor<48xf32>) -> tensor<64x48x56x56xf32>
    %v70 = stablehlo.divide %v69, %v61 : tensor<64x48x56x56xf32>
    %v71 = stablehlo.add %v70, %v62 : tensor<64x48x56x56xf32>
    %v72 = stablehlo.rsqrt %v71 : tensor<64x48x56x56xf32>
    %v73 = stablehlo.multiply %v66, %v72 : tensor<64x48x56x56xf32>
    %v74 = stablehlo.broadcast_in_dim %f0pg, dims = [1] : (tensor<48xf32>) -> tensor<64x48x56x56xf32>
    %v75 = stablehlo.broadcast_in_dim %f0pbt, dims = [1] : (tensor<48xf32>) -> tensor<64x48x56x56xf32>
    %v76 = stablehlo.multiply %v73, %v74 : tensor<64x48x56x56xf32>
    %v77 = stablehlo.add %v76, %v75 : tensor<64x48x56x56xf32>
    %v78 = stablehlo.reshape %v77 : (tensor<64x48x56x56xf32>) -> tensor<64x150528xf32>
    %v79 = stablehlo.reshape %v78 : (tensor<64x150528xf32>) -> tensor<64x48x56x56xf32>
    %v80 = stablehlo.convolution(%v79, %u1qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 48 : i64} : (tensor<64x48x56x56xf32>, tensor<48x1x3x3xf32>) -> tensor<64x48x28x28xf32>
    %v81 = stablehlo.broadcast_in_dim %zb48, dims = [1] : (tensor<48xf32>) -> tensor<64x48x28x28xf32>
    %v82 = stablehlo.add %v80, %v81 : tensor<64x48x28x28xf32>
    %v83 = stablehlo.reshape %v82 : (tensor<64x48x28x28xf32>) -> tensor<64x37632xf32>
    %v84 = stablehlo.reshape %v83 : (tensor<64x37632xf32>) -> tensor<64x48x28x28xf32>
    %v85 = stablehlo.constant dense<0.0> : tensor<f32>
    %v86 = stablehlo.constant dense<50176.0> : tensor<64x48x28x28xf32>
    %v87 = stablehlo.constant dense<1.0e-5> : tensor<64x48x28x28xf32>
    %v88 = stablehlo.reduce(%v84 init: %v85) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x48x28x28xf32>, tensor<f32>) -> tensor<48xf32>
    %v89 = stablehlo.broadcast_in_dim %v88, dims = [1] : (tensor<48xf32>) -> tensor<64x48x28x28xf32>
    %v90 = stablehlo.divide %v89, %v86 : tensor<64x48x28x28xf32>
    %v91 = stablehlo.subtract %v84, %v90 : tensor<64x48x28x28xf32>
    %v92 = stablehlo.multiply %v91, %v91 : tensor<64x48x28x28xf32>
    %v93 = stablehlo.reduce(%v92 init: %v85) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x48x28x28xf32>, tensor<f32>) -> tensor<48xf32>
    %v94 = stablehlo.broadcast_in_dim %v93, dims = [1] : (tensor<48xf32>) -> tensor<64x48x28x28xf32>
    %v95 = stablehlo.divide %v94, %v86 : tensor<64x48x28x28xf32>
    %v96 = stablehlo.add %v95, %v87 : tensor<64x48x28x28xf32>
    %v97 = stablehlo.rsqrt %v96 : tensor<64x48x28x28xf32>
    %v98 = stablehlo.multiply %v91, %v97 : tensor<64x48x28x28xf32>
    %v99 = stablehlo.broadcast_in_dim %u1qg, dims = [1] : (tensor<48xf32>) -> tensor<64x48x28x28xf32>
    %v100 = stablehlo.broadcast_in_dim %u1qbt, dims = [1] : (tensor<48xf32>) -> tensor<64x48x28x28xf32>
    %v101 = stablehlo.multiply %v98, %v99 : tensor<64x48x28x28xf32>
    %v102 = stablehlo.add %v101, %v100 : tensor<64x48x28x28xf32>
    %v103 = stablehlo.reshape %v102 : (tensor<64x48x28x28xf32>) -> tensor<64x37632xf32>
    %v104 = stablehlo.constant dense<0.0> : tensor<64x37632xf32>
    %v105 = stablehlo.maximum %v103, %v104 : tensor<64x37632xf32>
    %v106 = stablehlo.reshape %v105 : (tensor<64x37632xf32>) -> tensor<64x48x28x28xf32>
    %v107 = stablehlo.convolution(%v106, %u1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x48x28x28xf32>, tensor<192x48x1x1xf32>) -> tensor<64x192x28x28xf32>
    %v108 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v109 = stablehlo.add %v107, %v108 : tensor<64x192x28x28xf32>
    %v110 = stablehlo.reshape %v109 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v111 = stablehlo.reshape %v110 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v112 = stablehlo.constant dense<0.0> : tensor<f32>
    %v113 = stablehlo.constant dense<50176.0> : tensor<64x192x28x28xf32>
    %v114 = stablehlo.constant dense<1.0e-5> : tensor<64x192x28x28xf32>
    %v115 = stablehlo.reduce(%v111 init: %v112) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v116 = stablehlo.broadcast_in_dim %v115, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v117 = stablehlo.divide %v116, %v113 : tensor<64x192x28x28xf32>
    %v118 = stablehlo.subtract %v111, %v117 : tensor<64x192x28x28xf32>
    %v119 = stablehlo.multiply %v118, %v118 : tensor<64x192x28x28xf32>
    %v120 = stablehlo.reduce(%v119 init: %v112) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v121 = stablehlo.broadcast_in_dim %v120, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v122 = stablehlo.divide %v121, %v113 : tensor<64x192x28x28xf32>
    %v123 = stablehlo.add %v122, %v114 : tensor<64x192x28x28xf32>
    %v124 = stablehlo.rsqrt %v123 : tensor<64x192x28x28xf32>
    %v125 = stablehlo.multiply %v118, %v124 : tensor<64x192x28x28xf32>
    %v126 = stablehlo.broadcast_in_dim %u1eg, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v127 = stablehlo.broadcast_in_dim %u1ebt, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v128 = stablehlo.multiply %v125, %v126 : tensor<64x192x28x28xf32>
    %v129 = stablehlo.add %v128, %v127 : tensor<64x192x28x28xf32>
    %v130 = stablehlo.reshape %v129 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v131 = stablehlo.constant dense<0.0> : tensor<64x150528xf32>
    %v132 = stablehlo.maximum %v130, %v131 : tensor<64x150528xf32>
    %v133 = stablehlo.reshape %v132 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v134 = stablehlo.convolution(%v133, %u1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<64x192x28x28xf32>, tensor<192x1x5x5xf32>) -> tensor<64x192x28x28xf32>
    %v135 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v136 = stablehlo.add %v134, %v135 : tensor<64x192x28x28xf32>
    %v137 = stablehlo.reshape %v136 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v138 = stablehlo.reshape %v137 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v139 = stablehlo.constant dense<0.0> : tensor<f32>
    %v140 = stablehlo.constant dense<50176.0> : tensor<64x192x28x28xf32>
    %v141 = stablehlo.constant dense<1.0e-5> : tensor<64x192x28x28xf32>
    %v142 = stablehlo.reduce(%v138 init: %v139) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v143 = stablehlo.broadcast_in_dim %v142, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v144 = stablehlo.divide %v143, %v140 : tensor<64x192x28x28xf32>
    %v145 = stablehlo.subtract %v138, %v144 : tensor<64x192x28x28xf32>
    %v146 = stablehlo.multiply %v145, %v145 : tensor<64x192x28x28xf32>
    %v147 = stablehlo.reduce(%v146 init: %v139) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v148 = stablehlo.broadcast_in_dim %v147, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v149 = stablehlo.divide %v148, %v140 : tensor<64x192x28x28xf32>
    %v150 = stablehlo.add %v149, %v141 : tensor<64x192x28x28xf32>
    %v151 = stablehlo.rsqrt %v150 : tensor<64x192x28x28xf32>
    %v152 = stablehlo.multiply %v145, %v151 : tensor<64x192x28x28xf32>
    %v153 = stablehlo.broadcast_in_dim %u1dg, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v154 = stablehlo.broadcast_in_dim %u1dbt, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v155 = stablehlo.multiply %v152, %v153 : tensor<64x192x28x28xf32>
    %v156 = stablehlo.add %v155, %v154 : tensor<64x192x28x28xf32>
    %v157 = stablehlo.reshape %v156 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v158 = stablehlo.constant dense<0.0> : tensor<64x150528xf32>
    %v159 = stablehlo.maximum %v157, %v158 : tensor<64x150528xf32>
    %v160 = stablehlo.reshape %v159 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v161 = stablehlo.convolution(%v160, %u1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x192x28x28xf32>, tensor<80x192x1x1xf32>) -> tensor<64x80x28x28xf32>
    %v162 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v163 = stablehlo.add %v161, %v162 : tensor<64x80x28x28xf32>
    %v164 = stablehlo.reshape %v163 : (tensor<64x80x28x28xf32>) -> tensor<64x62720xf32>
    %v165 = stablehlo.reshape %v164 : (tensor<64x62720xf32>) -> tensor<64x80x28x28xf32>
    %v166 = stablehlo.constant dense<0.0> : tensor<f32>
    %v167 = stablehlo.constant dense<50176.0> : tensor<64x80x28x28xf32>
    %v168 = stablehlo.constant dense<1.0e-5> : tensor<64x80x28x28xf32>
    %v169 = stablehlo.reduce(%v165 init: %v166) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x80x28x28xf32>, tensor<f32>) -> tensor<80xf32>
    %v170 = stablehlo.broadcast_in_dim %v169, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v171 = stablehlo.divide %v170, %v167 : tensor<64x80x28x28xf32>
    %v172 = stablehlo.subtract %v165, %v171 : tensor<64x80x28x28xf32>
    %v173 = stablehlo.multiply %v172, %v172 : tensor<64x80x28x28xf32>
    %v174 = stablehlo.reduce(%v173 init: %v166) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x80x28x28xf32>, tensor<f32>) -> tensor<80xf32>
    %v175 = stablehlo.broadcast_in_dim %v174, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v176 = stablehlo.divide %v175, %v167 : tensor<64x80x28x28xf32>
    %v177 = stablehlo.add %v176, %v168 : tensor<64x80x28x28xf32>
    %v178 = stablehlo.rsqrt %v177 : tensor<64x80x28x28xf32>
    %v179 = stablehlo.multiply %v172, %v178 : tensor<64x80x28x28xf32>
    %v180 = stablehlo.broadcast_in_dim %u1pg, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v181 = stablehlo.broadcast_in_dim %u1pbt, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v182 = stablehlo.multiply %v179, %v180 : tensor<64x80x28x28xf32>
    %v183 = stablehlo.add %v182, %v181 : tensor<64x80x28x28xf32>
    %v184 = stablehlo.reshape %v183 : (tensor<64x80x28x28xf32>) -> tensor<64x62720xf32>
    %v185 = stablehlo.reshape %v184 : (tensor<64x62720xf32>) -> tensor<64x80x28x28xf32>
    %v186 = stablehlo.convolution(%v185, %u2qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 80 : i64} : (tensor<64x80x28x28xf32>, tensor<80x1x3x3xf32>) -> tensor<64x80x28x28xf32>
    %v187 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v188 = stablehlo.add %v186, %v187 : tensor<64x80x28x28xf32>
    %v189 = stablehlo.reshape %v188 : (tensor<64x80x28x28xf32>) -> tensor<64x62720xf32>
    %v190 = stablehlo.reshape %v189 : (tensor<64x62720xf32>) -> tensor<64x80x28x28xf32>
    %v191 = stablehlo.constant dense<0.0> : tensor<f32>
    %v192 = stablehlo.constant dense<50176.0> : tensor<64x80x28x28xf32>
    %v193 = stablehlo.constant dense<1.0e-5> : tensor<64x80x28x28xf32>
    %v194 = stablehlo.reduce(%v190 init: %v191) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x80x28x28xf32>, tensor<f32>) -> tensor<80xf32>
    %v195 = stablehlo.broadcast_in_dim %v194, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v196 = stablehlo.divide %v195, %v192 : tensor<64x80x28x28xf32>
    %v197 = stablehlo.subtract %v190, %v196 : tensor<64x80x28x28xf32>
    %v198 = stablehlo.multiply %v197, %v197 : tensor<64x80x28x28xf32>
    %v199 = stablehlo.reduce(%v198 init: %v191) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x80x28x28xf32>, tensor<f32>) -> tensor<80xf32>
    %v200 = stablehlo.broadcast_in_dim %v199, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v201 = stablehlo.divide %v200, %v192 : tensor<64x80x28x28xf32>
    %v202 = stablehlo.add %v201, %v193 : tensor<64x80x28x28xf32>
    %v203 = stablehlo.rsqrt %v202 : tensor<64x80x28x28xf32>
    %v204 = stablehlo.multiply %v197, %v203 : tensor<64x80x28x28xf32>
    %v205 = stablehlo.broadcast_in_dim %u2qg, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v206 = stablehlo.broadcast_in_dim %u2qbt, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v207 = stablehlo.multiply %v204, %v205 : tensor<64x80x28x28xf32>
    %v208 = stablehlo.add %v207, %v206 : tensor<64x80x28x28xf32>
    %v209 = stablehlo.reshape %v208 : (tensor<64x80x28x28xf32>) -> tensor<64x62720xf32>
    %v210 = stablehlo.constant dense<0.0> : tensor<64x62720xf32>
    %v211 = stablehlo.maximum %v209, %v210 : tensor<64x62720xf32>
    %v212 = stablehlo.reshape %v211 : (tensor<64x62720xf32>) -> tensor<64x80x28x28xf32>
    %v213 = stablehlo.convolution(%v212, %u2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x80x28x28xf32>, tensor<160x80x1x1xf32>) -> tensor<64x160x28x28xf32>
    %v214 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x28x28xf32>
    %v215 = stablehlo.add %v213, %v214 : tensor<64x160x28x28xf32>
    %v216 = stablehlo.reshape %v215 : (tensor<64x160x28x28xf32>) -> tensor<64x125440xf32>
    %v217 = stablehlo.reshape %v216 : (tensor<64x125440xf32>) -> tensor<64x160x28x28xf32>
    %v218 = stablehlo.constant dense<0.0> : tensor<f32>
    %v219 = stablehlo.constant dense<50176.0> : tensor<64x160x28x28xf32>
    %v220 = stablehlo.constant dense<1.0e-5> : tensor<64x160x28x28xf32>
    %v221 = stablehlo.reduce(%v217 init: %v218) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x160x28x28xf32>, tensor<f32>) -> tensor<160xf32>
    %v222 = stablehlo.broadcast_in_dim %v221, dims = [1] : (tensor<160xf32>) -> tensor<64x160x28x28xf32>
    %v223 = stablehlo.divide %v222, %v219 : tensor<64x160x28x28xf32>
    %v224 = stablehlo.subtract %v217, %v223 : tensor<64x160x28x28xf32>
    %v225 = stablehlo.multiply %v224, %v224 : tensor<64x160x28x28xf32>
    %v226 = stablehlo.reduce(%v225 init: %v218) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x160x28x28xf32>, tensor<f32>) -> tensor<160xf32>
    %v227 = stablehlo.broadcast_in_dim %v226, dims = [1] : (tensor<160xf32>) -> tensor<64x160x28x28xf32>
    %v228 = stablehlo.divide %v227, %v219 : tensor<64x160x28x28xf32>
    %v229 = stablehlo.add %v228, %v220 : tensor<64x160x28x28xf32>
    %v230 = stablehlo.rsqrt %v229 : tensor<64x160x28x28xf32>
    %v231 = stablehlo.multiply %v224, %v230 : tensor<64x160x28x28xf32>
    %v232 = stablehlo.broadcast_in_dim %u2eg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x28x28xf32>
    %v233 = stablehlo.broadcast_in_dim %u2ebt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x28x28xf32>
    %v234 = stablehlo.multiply %v231, %v232 : tensor<64x160x28x28xf32>
    %v235 = stablehlo.add %v234, %v233 : tensor<64x160x28x28xf32>
    %v236 = stablehlo.reshape %v235 : (tensor<64x160x28x28xf32>) -> tensor<64x125440xf32>
    %v237 = stablehlo.constant dense<0.0> : tensor<64x125440xf32>
    %v238 = stablehlo.maximum %v236, %v237 : tensor<64x125440xf32>
    %v239 = stablehlo.reshape %v238 : (tensor<64x125440xf32>) -> tensor<64x160x28x28xf32>
    %v240 = stablehlo.convolution(%v239, %u2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<64x160x28x28xf32>, tensor<160x1x3x3xf32>) -> tensor<64x160x28x28xf32>
    %v241 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x28x28xf32>
    %v242 = stablehlo.add %v240, %v241 : tensor<64x160x28x28xf32>
    %v243 = stablehlo.reshape %v242 : (tensor<64x160x28x28xf32>) -> tensor<64x125440xf32>
    %v244 = stablehlo.reshape %v243 : (tensor<64x125440xf32>) -> tensor<64x160x28x28xf32>
    %v245 = stablehlo.constant dense<0.0> : tensor<f32>
    %v246 = stablehlo.constant dense<50176.0> : tensor<64x160x28x28xf32>
    %v247 = stablehlo.constant dense<1.0e-5> : tensor<64x160x28x28xf32>
    %v248 = stablehlo.reduce(%v244 init: %v245) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x160x28x28xf32>, tensor<f32>) -> tensor<160xf32>
    %v249 = stablehlo.broadcast_in_dim %v248, dims = [1] : (tensor<160xf32>) -> tensor<64x160x28x28xf32>
    %v250 = stablehlo.divide %v249, %v246 : tensor<64x160x28x28xf32>
    %v251 = stablehlo.subtract %v244, %v250 : tensor<64x160x28x28xf32>
    %v252 = stablehlo.multiply %v251, %v251 : tensor<64x160x28x28xf32>
    %v253 = stablehlo.reduce(%v252 init: %v245) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x160x28x28xf32>, tensor<f32>) -> tensor<160xf32>
    %v254 = stablehlo.broadcast_in_dim %v253, dims = [1] : (tensor<160xf32>) -> tensor<64x160x28x28xf32>
    %v255 = stablehlo.divide %v254, %v246 : tensor<64x160x28x28xf32>
    %v256 = stablehlo.add %v255, %v247 : tensor<64x160x28x28xf32>
    %v257 = stablehlo.rsqrt %v256 : tensor<64x160x28x28xf32>
    %v258 = stablehlo.multiply %v251, %v257 : tensor<64x160x28x28xf32>
    %v259 = stablehlo.broadcast_in_dim %u2dg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x28x28xf32>
    %v260 = stablehlo.broadcast_in_dim %u2dbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x28x28xf32>
    %v261 = stablehlo.multiply %v258, %v259 : tensor<64x160x28x28xf32>
    %v262 = stablehlo.add %v261, %v260 : tensor<64x160x28x28xf32>
    %v263 = stablehlo.reshape %v262 : (tensor<64x160x28x28xf32>) -> tensor<64x125440xf32>
    %v264 = stablehlo.constant dense<0.0> : tensor<64x125440xf32>
    %v265 = stablehlo.maximum %v263, %v264 : tensor<64x125440xf32>
    %v266 = stablehlo.reshape %v265 : (tensor<64x125440xf32>) -> tensor<64x160x28x28xf32>
    %v267 = stablehlo.convolution(%v266, %u2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x160x28x28xf32>, tensor<80x160x1x1xf32>) -> tensor<64x80x28x28xf32>
    %v268 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v269 = stablehlo.add %v267, %v268 : tensor<64x80x28x28xf32>
    %v270 = stablehlo.reshape %v269 : (tensor<64x80x28x28xf32>) -> tensor<64x62720xf32>
    %v271 = stablehlo.reshape %v270 : (tensor<64x62720xf32>) -> tensor<64x80x28x28xf32>
    %v272 = stablehlo.constant dense<0.0> : tensor<f32>
    %v273 = stablehlo.constant dense<50176.0> : tensor<64x80x28x28xf32>
    %v274 = stablehlo.constant dense<1.0e-5> : tensor<64x80x28x28xf32>
    %v275 = stablehlo.reduce(%v271 init: %v272) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x80x28x28xf32>, tensor<f32>) -> tensor<80xf32>
    %v276 = stablehlo.broadcast_in_dim %v275, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v277 = stablehlo.divide %v276, %v273 : tensor<64x80x28x28xf32>
    %v278 = stablehlo.subtract %v271, %v277 : tensor<64x80x28x28xf32>
    %v279 = stablehlo.multiply %v278, %v278 : tensor<64x80x28x28xf32>
    %v280 = stablehlo.reduce(%v279 init: %v272) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x80x28x28xf32>, tensor<f32>) -> tensor<80xf32>
    %v281 = stablehlo.broadcast_in_dim %v280, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v282 = stablehlo.divide %v281, %v273 : tensor<64x80x28x28xf32>
    %v283 = stablehlo.add %v282, %v274 : tensor<64x80x28x28xf32>
    %v284 = stablehlo.rsqrt %v283 : tensor<64x80x28x28xf32>
    %v285 = stablehlo.multiply %v278, %v284 : tensor<64x80x28x28xf32>
    %v286 = stablehlo.broadcast_in_dim %u2pg, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v287 = stablehlo.broadcast_in_dim %u2pbt, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v288 = stablehlo.multiply %v285, %v286 : tensor<64x80x28x28xf32>
    %v289 = stablehlo.add %v288, %v287 : tensor<64x80x28x28xf32>
    %v290 = stablehlo.reshape %v289 : (tensor<64x80x28x28xf32>) -> tensor<64x62720xf32>
    %v291 = stablehlo.add %v290, %v184 : tensor<64x62720xf32>
    %v292 = stablehlo.reshape %v291 : (tensor<64x62720xf32>) -> tensor<64x80x28x28xf32>
    %v293 = stablehlo.convolution(%v292, %u3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x80x28x28xf32>, tensor<480x80x1x1xf32>) -> tensor<64x480x28x28xf32>
    %v294 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<64x480x28x28xf32>
    %v295 = stablehlo.add %v293, %v294 : tensor<64x480x28x28xf32>
    %v296 = stablehlo.reshape %v295 : (tensor<64x480x28x28xf32>) -> tensor<64x376320xf32>
    %v297 = stablehlo.reshape %v296 : (tensor<64x376320xf32>) -> tensor<64x480x28x28xf32>
    %v298 = stablehlo.constant dense<0.0> : tensor<f32>
    %v299 = stablehlo.constant dense<50176.0> : tensor<64x480x28x28xf32>
    %v300 = stablehlo.constant dense<1.0e-5> : tensor<64x480x28x28xf32>
    %v301 = stablehlo.reduce(%v297 init: %v298) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x28x28xf32>, tensor<f32>) -> tensor<480xf32>
    %v302 = stablehlo.broadcast_in_dim %v301, dims = [1] : (tensor<480xf32>) -> tensor<64x480x28x28xf32>
    %v303 = stablehlo.divide %v302, %v299 : tensor<64x480x28x28xf32>
    %v304 = stablehlo.subtract %v297, %v303 : tensor<64x480x28x28xf32>
    %v305 = stablehlo.multiply %v304, %v304 : tensor<64x480x28x28xf32>
    %v306 = stablehlo.reduce(%v305 init: %v298) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x28x28xf32>, tensor<f32>) -> tensor<480xf32>
    %v307 = stablehlo.broadcast_in_dim %v306, dims = [1] : (tensor<480xf32>) -> tensor<64x480x28x28xf32>
    %v308 = stablehlo.divide %v307, %v299 : tensor<64x480x28x28xf32>
    %v309 = stablehlo.add %v308, %v300 : tensor<64x480x28x28xf32>
    %v310 = stablehlo.rsqrt %v309 : tensor<64x480x28x28xf32>
    %v311 = stablehlo.multiply %v304, %v310 : tensor<64x480x28x28xf32>
    %v312 = stablehlo.broadcast_in_dim %u3eg, dims = [1] : (tensor<480xf32>) -> tensor<64x480x28x28xf32>
    %v313 = stablehlo.broadcast_in_dim %u3ebt, dims = [1] : (tensor<480xf32>) -> tensor<64x480x28x28xf32>
    %v314 = stablehlo.multiply %v311, %v312 : tensor<64x480x28x28xf32>
    %v315 = stablehlo.add %v314, %v313 : tensor<64x480x28x28xf32>
    %v316 = stablehlo.reshape %v315 : (tensor<64x480x28x28xf32>) -> tensor<64x376320xf32>
    %v317 = stablehlo.constant dense<0.0> : tensor<64x376320xf32>
    %v318 = stablehlo.maximum %v316, %v317 : tensor<64x376320xf32>
    %v319 = stablehlo.reshape %v318 : (tensor<64x376320xf32>) -> tensor<64x480x28x28xf32>
    %v320 = stablehlo.convolution(%v319, %u3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<64x480x28x28xf32>, tensor<480x1x3x3xf32>) -> tensor<64x480x14x14xf32>
    %v321 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v322 = stablehlo.add %v320, %v321 : tensor<64x480x14x14xf32>
    %v323 = stablehlo.reshape %v322 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v324 = stablehlo.reshape %v323 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v325 = stablehlo.constant dense<0.0> : tensor<f32>
    %v326 = stablehlo.constant dense<12544.0> : tensor<64x480x14x14xf32>
    %v327 = stablehlo.constant dense<1.0e-5> : tensor<64x480x14x14xf32>
    %v328 = stablehlo.reduce(%v324 init: %v325) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v329 = stablehlo.broadcast_in_dim %v328, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v330 = stablehlo.divide %v329, %v326 : tensor<64x480x14x14xf32>
    %v331 = stablehlo.subtract %v324, %v330 : tensor<64x480x14x14xf32>
    %v332 = stablehlo.multiply %v331, %v331 : tensor<64x480x14x14xf32>
    %v333 = stablehlo.reduce(%v332 init: %v325) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v334 = stablehlo.broadcast_in_dim %v333, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v335 = stablehlo.divide %v334, %v326 : tensor<64x480x14x14xf32>
    %v336 = stablehlo.add %v335, %v327 : tensor<64x480x14x14xf32>
    %v337 = stablehlo.rsqrt %v336 : tensor<64x480x14x14xf32>
    %v338 = stablehlo.multiply %v331, %v337 : tensor<64x480x14x14xf32>
    %v339 = stablehlo.broadcast_in_dim %u3dg, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v340 = stablehlo.broadcast_in_dim %u3dbt, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v341 = stablehlo.multiply %v338, %v339 : tensor<64x480x14x14xf32>
    %v342 = stablehlo.add %v341, %v340 : tensor<64x480x14x14xf32>
    %v343 = stablehlo.reshape %v342 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v344 = stablehlo.constant dense<0.0> : tensor<64x94080xf32>
    %v345 = stablehlo.maximum %v343, %v344 : tensor<64x94080xf32>
    %v346 = stablehlo.reshape %v345 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v347 = stablehlo.convolution(%v346, %u3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x480x14x14xf32>, tensor<160x480x1x1xf32>) -> tensor<64x160x14x14xf32>
    %v348 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v349 = stablehlo.add %v347, %v348 : tensor<64x160x14x14xf32>
    %v350 = stablehlo.reshape %v349 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v351 = stablehlo.reshape %v350 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v352 = stablehlo.constant dense<0.0> : tensor<f32>
    %v353 = stablehlo.constant dense<12544.0> : tensor<64x160x14x14xf32>
    %v354 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v355 = stablehlo.reduce(%v351 init: %v352) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v356 = stablehlo.broadcast_in_dim %v355, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v357 = stablehlo.divide %v356, %v353 : tensor<64x160x14x14xf32>
    %v358 = stablehlo.subtract %v351, %v357 : tensor<64x160x14x14xf32>
    %v359 = stablehlo.multiply %v358, %v358 : tensor<64x160x14x14xf32>
    %v360 = stablehlo.reduce(%v359 init: %v352) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v361 = stablehlo.broadcast_in_dim %v360, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v362 = stablehlo.divide %v361, %v353 : tensor<64x160x14x14xf32>
    %v363 = stablehlo.add %v362, %v354 : tensor<64x160x14x14xf32>
    %v364 = stablehlo.rsqrt %v363 : tensor<64x160x14x14xf32>
    %v365 = stablehlo.multiply %v358, %v364 : tensor<64x160x14x14xf32>
    %v366 = stablehlo.broadcast_in_dim %u3pg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v367 = stablehlo.broadcast_in_dim %u3pbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v368 = stablehlo.multiply %v365, %v366 : tensor<64x160x14x14xf32>
    %v369 = stablehlo.add %v368, %v367 : tensor<64x160x14x14xf32>
    %v370 = stablehlo.reshape %v369 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v371 = stablehlo.reshape %v370 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v372 = stablehlo.convolution(%v371, %u4qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<64x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<64x160x14x14xf32>
    %v373 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v374 = stablehlo.add %v372, %v373 : tensor<64x160x14x14xf32>
    %v375 = stablehlo.reshape %v374 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v376 = stablehlo.reshape %v375 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v377 = stablehlo.constant dense<0.0> : tensor<f32>
    %v378 = stablehlo.constant dense<12544.0> : tensor<64x160x14x14xf32>
    %v379 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v380 = stablehlo.reduce(%v376 init: %v377) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v381 = stablehlo.broadcast_in_dim %v380, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v382 = stablehlo.divide %v381, %v378 : tensor<64x160x14x14xf32>
    %v383 = stablehlo.subtract %v376, %v382 : tensor<64x160x14x14xf32>
    %v384 = stablehlo.multiply %v383, %v383 : tensor<64x160x14x14xf32>
    %v385 = stablehlo.reduce(%v384 init: %v377) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v386 = stablehlo.broadcast_in_dim %v385, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v387 = stablehlo.divide %v386, %v378 : tensor<64x160x14x14xf32>
    %v388 = stablehlo.add %v387, %v379 : tensor<64x160x14x14xf32>
    %v389 = stablehlo.rsqrt %v388 : tensor<64x160x14x14xf32>
    %v390 = stablehlo.multiply %v383, %v389 : tensor<64x160x14x14xf32>
    %v391 = stablehlo.broadcast_in_dim %u4qg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v392 = stablehlo.broadcast_in_dim %u4qbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v393 = stablehlo.multiply %v390, %v391 : tensor<64x160x14x14xf32>
    %v394 = stablehlo.add %v393, %v392 : tensor<64x160x14x14xf32>
    %v395 = stablehlo.reshape %v394 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v396 = stablehlo.constant dense<0.0> : tensor<64x31360xf32>
    %v397 = stablehlo.maximum %v395, %v396 : tensor<64x31360xf32>
    %v398 = stablehlo.reshape %v397 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v399 = stablehlo.convolution(%v398, %u4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<64x640x14x14xf32>
    %v400 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v401 = stablehlo.add %v399, %v400 : tensor<64x640x14x14xf32>
    %v402 = stablehlo.reshape %v401 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v403 = stablehlo.reshape %v402 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v404 = stablehlo.constant dense<0.0> : tensor<f32>
    %v405 = stablehlo.constant dense<12544.0> : tensor<64x640x14x14xf32>
    %v406 = stablehlo.constant dense<1.0e-5> : tensor<64x640x14x14xf32>
    %v407 = stablehlo.reduce(%v403 init: %v404) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v408 = stablehlo.broadcast_in_dim %v407, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v409 = stablehlo.divide %v408, %v405 : tensor<64x640x14x14xf32>
    %v410 = stablehlo.subtract %v403, %v409 : tensor<64x640x14x14xf32>
    %v411 = stablehlo.multiply %v410, %v410 : tensor<64x640x14x14xf32>
    %v412 = stablehlo.reduce(%v411 init: %v404) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v413 = stablehlo.broadcast_in_dim %v412, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v414 = stablehlo.divide %v413, %v405 : tensor<64x640x14x14xf32>
    %v415 = stablehlo.add %v414, %v406 : tensor<64x640x14x14xf32>
    %v416 = stablehlo.rsqrt %v415 : tensor<64x640x14x14xf32>
    %v417 = stablehlo.multiply %v410, %v416 : tensor<64x640x14x14xf32>
    %v418 = stablehlo.broadcast_in_dim %u4eg, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v419 = stablehlo.broadcast_in_dim %u4ebt, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v420 = stablehlo.multiply %v417, %v418 : tensor<64x640x14x14xf32>
    %v421 = stablehlo.add %v420, %v419 : tensor<64x640x14x14xf32>
    %v422 = stablehlo.reshape %v421 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v423 = stablehlo.constant dense<0.0> : tensor<64x125440xf32>
    %v424 = stablehlo.maximum %v422, %v423 : tensor<64x125440xf32>
    %v425 = stablehlo.reshape %v424 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v426 = stablehlo.convolution(%v425, %u4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 640 : i64} : (tensor<64x640x14x14xf32>, tensor<640x1x3x3xf32>) -> tensor<64x640x14x14xf32>
    %v427 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v428 = stablehlo.add %v426, %v427 : tensor<64x640x14x14xf32>
    %v429 = stablehlo.reshape %v428 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v430 = stablehlo.reshape %v429 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v431 = stablehlo.constant dense<0.0> : tensor<f32>
    %v432 = stablehlo.constant dense<12544.0> : tensor<64x640x14x14xf32>
    %v433 = stablehlo.constant dense<1.0e-5> : tensor<64x640x14x14xf32>
    %v434 = stablehlo.reduce(%v430 init: %v431) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v435 = stablehlo.broadcast_in_dim %v434, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v436 = stablehlo.divide %v435, %v432 : tensor<64x640x14x14xf32>
    %v437 = stablehlo.subtract %v430, %v436 : tensor<64x640x14x14xf32>
    %v438 = stablehlo.multiply %v437, %v437 : tensor<64x640x14x14xf32>
    %v439 = stablehlo.reduce(%v438 init: %v431) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v440 = stablehlo.broadcast_in_dim %v439, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v441 = stablehlo.divide %v440, %v432 : tensor<64x640x14x14xf32>
    %v442 = stablehlo.add %v441, %v433 : tensor<64x640x14x14xf32>
    %v443 = stablehlo.rsqrt %v442 : tensor<64x640x14x14xf32>
    %v444 = stablehlo.multiply %v437, %v443 : tensor<64x640x14x14xf32>
    %v445 = stablehlo.broadcast_in_dim %u4dg, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v446 = stablehlo.broadcast_in_dim %u4dbt, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v447 = stablehlo.multiply %v444, %v445 : tensor<64x640x14x14xf32>
    %v448 = stablehlo.add %v447, %v446 : tensor<64x640x14x14xf32>
    %v449 = stablehlo.reshape %v448 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v450 = stablehlo.constant dense<0.0> : tensor<64x125440xf32>
    %v451 = stablehlo.maximum %v449, %v450 : tensor<64x125440xf32>
    %v452 = stablehlo.reshape %v451 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v453 = stablehlo.convolution(%v452, %u4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<64x160x14x14xf32>
    %v454 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v455 = stablehlo.add %v453, %v454 : tensor<64x160x14x14xf32>
    %v456 = stablehlo.reshape %v455 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v457 = stablehlo.reshape %v456 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v458 = stablehlo.constant dense<0.0> : tensor<f32>
    %v459 = stablehlo.constant dense<12544.0> : tensor<64x160x14x14xf32>
    %v460 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v461 = stablehlo.reduce(%v457 init: %v458) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v462 = stablehlo.broadcast_in_dim %v461, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v463 = stablehlo.divide %v462, %v459 : tensor<64x160x14x14xf32>
    %v464 = stablehlo.subtract %v457, %v463 : tensor<64x160x14x14xf32>
    %v465 = stablehlo.multiply %v464, %v464 : tensor<64x160x14x14xf32>
    %v466 = stablehlo.reduce(%v465 init: %v458) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v467 = stablehlo.broadcast_in_dim %v466, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v468 = stablehlo.divide %v467, %v459 : tensor<64x160x14x14xf32>
    %v469 = stablehlo.add %v468, %v460 : tensor<64x160x14x14xf32>
    %v470 = stablehlo.rsqrt %v469 : tensor<64x160x14x14xf32>
    %v471 = stablehlo.multiply %v464, %v470 : tensor<64x160x14x14xf32>
    %v472 = stablehlo.broadcast_in_dim %u4pg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v473 = stablehlo.broadcast_in_dim %u4pbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v474 = stablehlo.multiply %v471, %v472 : tensor<64x160x14x14xf32>
    %v475 = stablehlo.add %v474, %v473 : tensor<64x160x14x14xf32>
    %v476 = stablehlo.reshape %v475 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v477 = stablehlo.add %v476, %v370 : tensor<64x31360xf32>
    %v478 = stablehlo.reshape %v477 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v479 = stablehlo.convolution(%v478, %u5qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<64x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<64x160x14x14xf32>
    %v480 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v481 = stablehlo.add %v479, %v480 : tensor<64x160x14x14xf32>
    %v482 = stablehlo.reshape %v481 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v483 = stablehlo.reshape %v482 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v484 = stablehlo.constant dense<0.0> : tensor<f32>
    %v485 = stablehlo.constant dense<12544.0> : tensor<64x160x14x14xf32>
    %v486 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v487 = stablehlo.reduce(%v483 init: %v484) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v488 = stablehlo.broadcast_in_dim %v487, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v489 = stablehlo.divide %v488, %v485 : tensor<64x160x14x14xf32>
    %v490 = stablehlo.subtract %v483, %v489 : tensor<64x160x14x14xf32>
    %v491 = stablehlo.multiply %v490, %v490 : tensor<64x160x14x14xf32>
    %v492 = stablehlo.reduce(%v491 init: %v484) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v493 = stablehlo.broadcast_in_dim %v492, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v494 = stablehlo.divide %v493, %v485 : tensor<64x160x14x14xf32>
    %v495 = stablehlo.add %v494, %v486 : tensor<64x160x14x14xf32>
    %v496 = stablehlo.rsqrt %v495 : tensor<64x160x14x14xf32>
    %v497 = stablehlo.multiply %v490, %v496 : tensor<64x160x14x14xf32>
    %v498 = stablehlo.broadcast_in_dim %u5qg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v499 = stablehlo.broadcast_in_dim %u5qbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v500 = stablehlo.multiply %v497, %v498 : tensor<64x160x14x14xf32>
    %v501 = stablehlo.add %v500, %v499 : tensor<64x160x14x14xf32>
    %v502 = stablehlo.reshape %v501 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v503 = stablehlo.constant dense<0.0> : tensor<64x31360xf32>
    %v504 = stablehlo.maximum %v502, %v503 : tensor<64x31360xf32>
    %v505 = stablehlo.reshape %v504 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v506 = stablehlo.convolution(%v505, %u5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<64x640x14x14xf32>
    %v507 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v508 = stablehlo.add %v506, %v507 : tensor<64x640x14x14xf32>
    %v509 = stablehlo.reshape %v508 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v510 = stablehlo.reshape %v509 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v511 = stablehlo.constant dense<0.0> : tensor<f32>
    %v512 = stablehlo.constant dense<12544.0> : tensor<64x640x14x14xf32>
    %v513 = stablehlo.constant dense<1.0e-5> : tensor<64x640x14x14xf32>
    %v514 = stablehlo.reduce(%v510 init: %v511) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v515 = stablehlo.broadcast_in_dim %v514, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v516 = stablehlo.divide %v515, %v512 : tensor<64x640x14x14xf32>
    %v517 = stablehlo.subtract %v510, %v516 : tensor<64x640x14x14xf32>
    %v518 = stablehlo.multiply %v517, %v517 : tensor<64x640x14x14xf32>
    %v519 = stablehlo.reduce(%v518 init: %v511) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v520 = stablehlo.broadcast_in_dim %v519, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v521 = stablehlo.divide %v520, %v512 : tensor<64x640x14x14xf32>
    %v522 = stablehlo.add %v521, %v513 : tensor<64x640x14x14xf32>
    %v523 = stablehlo.rsqrt %v522 : tensor<64x640x14x14xf32>
    %v524 = stablehlo.multiply %v517, %v523 : tensor<64x640x14x14xf32>
    %v525 = stablehlo.broadcast_in_dim %u5eg, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v526 = stablehlo.broadcast_in_dim %u5ebt, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v527 = stablehlo.multiply %v524, %v525 : tensor<64x640x14x14xf32>
    %v528 = stablehlo.add %v527, %v526 : tensor<64x640x14x14xf32>
    %v529 = stablehlo.reshape %v528 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v530 = stablehlo.constant dense<0.0> : tensor<64x125440xf32>
    %v531 = stablehlo.maximum %v529, %v530 : tensor<64x125440xf32>
    %v532 = stablehlo.reshape %v531 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v533 = stablehlo.convolution(%v532, %u5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 640 : i64} : (tensor<64x640x14x14xf32>, tensor<640x1x5x5xf32>) -> tensor<64x640x14x14xf32>
    %v534 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v535 = stablehlo.add %v533, %v534 : tensor<64x640x14x14xf32>
    %v536 = stablehlo.reshape %v535 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v537 = stablehlo.reshape %v536 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v538 = stablehlo.constant dense<0.0> : tensor<f32>
    %v539 = stablehlo.constant dense<12544.0> : tensor<64x640x14x14xf32>
    %v540 = stablehlo.constant dense<1.0e-5> : tensor<64x640x14x14xf32>
    %v541 = stablehlo.reduce(%v537 init: %v538) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v542 = stablehlo.broadcast_in_dim %v541, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v543 = stablehlo.divide %v542, %v539 : tensor<64x640x14x14xf32>
    %v544 = stablehlo.subtract %v537, %v543 : tensor<64x640x14x14xf32>
    %v545 = stablehlo.multiply %v544, %v544 : tensor<64x640x14x14xf32>
    %v546 = stablehlo.reduce(%v545 init: %v538) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v547 = stablehlo.broadcast_in_dim %v546, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v548 = stablehlo.divide %v547, %v539 : tensor<64x640x14x14xf32>
    %v549 = stablehlo.add %v548, %v540 : tensor<64x640x14x14xf32>
    %v550 = stablehlo.rsqrt %v549 : tensor<64x640x14x14xf32>
    %v551 = stablehlo.multiply %v544, %v550 : tensor<64x640x14x14xf32>
    %v552 = stablehlo.broadcast_in_dim %u5dg, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v553 = stablehlo.broadcast_in_dim %u5dbt, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v554 = stablehlo.multiply %v551, %v552 : tensor<64x640x14x14xf32>
    %v555 = stablehlo.add %v554, %v553 : tensor<64x640x14x14xf32>
    %v556 = stablehlo.reshape %v555 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v557 = stablehlo.constant dense<0.0> : tensor<64x125440xf32>
    %v558 = stablehlo.maximum %v556, %v557 : tensor<64x125440xf32>
    %v559 = stablehlo.reshape %v558 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v560 = stablehlo.convolution(%v559, %u5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<64x160x14x14xf32>
    %v561 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v562 = stablehlo.add %v560, %v561 : tensor<64x160x14x14xf32>
    %v563 = stablehlo.reshape %v562 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v564 = stablehlo.reshape %v563 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v565 = stablehlo.constant dense<0.0> : tensor<f32>
    %v566 = stablehlo.constant dense<12544.0> : tensor<64x160x14x14xf32>
    %v567 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v568 = stablehlo.reduce(%v564 init: %v565) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v569 = stablehlo.broadcast_in_dim %v568, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v570 = stablehlo.divide %v569, %v566 : tensor<64x160x14x14xf32>
    %v571 = stablehlo.subtract %v564, %v570 : tensor<64x160x14x14xf32>
    %v572 = stablehlo.multiply %v571, %v571 : tensor<64x160x14x14xf32>
    %v573 = stablehlo.reduce(%v572 init: %v565) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v574 = stablehlo.broadcast_in_dim %v573, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v575 = stablehlo.divide %v574, %v566 : tensor<64x160x14x14xf32>
    %v576 = stablehlo.add %v575, %v567 : tensor<64x160x14x14xf32>
    %v577 = stablehlo.rsqrt %v576 : tensor<64x160x14x14xf32>
    %v578 = stablehlo.multiply %v571, %v577 : tensor<64x160x14x14xf32>
    %v579 = stablehlo.broadcast_in_dim %u5pg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v580 = stablehlo.broadcast_in_dim %u5pbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v581 = stablehlo.multiply %v578, %v579 : tensor<64x160x14x14xf32>
    %v582 = stablehlo.add %v581, %v580 : tensor<64x160x14x14xf32>
    %v583 = stablehlo.reshape %v582 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v584 = stablehlo.add %v583, %v477 : tensor<64x31360xf32>
    %v585 = stablehlo.reshape %v584 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v586 = stablehlo.convolution(%v585, %u6qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<64x160x14x14xf32>, tensor<160x1x5x5xf32>) -> tensor<64x160x14x14xf32>
    %v587 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v588 = stablehlo.add %v586, %v587 : tensor<64x160x14x14xf32>
    %v589 = stablehlo.reshape %v588 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v590 = stablehlo.reshape %v589 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v591 = stablehlo.constant dense<0.0> : tensor<f32>
    %v592 = stablehlo.constant dense<12544.0> : tensor<64x160x14x14xf32>
    %v593 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v594 = stablehlo.reduce(%v590 init: %v591) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v595 = stablehlo.broadcast_in_dim %v594, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v596 = stablehlo.divide %v595, %v592 : tensor<64x160x14x14xf32>
    %v597 = stablehlo.subtract %v590, %v596 : tensor<64x160x14x14xf32>
    %v598 = stablehlo.multiply %v597, %v597 : tensor<64x160x14x14xf32>
    %v599 = stablehlo.reduce(%v598 init: %v591) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v600 = stablehlo.broadcast_in_dim %v599, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v601 = stablehlo.divide %v600, %v592 : tensor<64x160x14x14xf32>
    %v602 = stablehlo.add %v601, %v593 : tensor<64x160x14x14xf32>
    %v603 = stablehlo.rsqrt %v602 : tensor<64x160x14x14xf32>
    %v604 = stablehlo.multiply %v597, %v603 : tensor<64x160x14x14xf32>
    %v605 = stablehlo.broadcast_in_dim %u6qg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v606 = stablehlo.broadcast_in_dim %u6qbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v607 = stablehlo.multiply %v604, %v605 : tensor<64x160x14x14xf32>
    %v608 = stablehlo.add %v607, %v606 : tensor<64x160x14x14xf32>
    %v609 = stablehlo.reshape %v608 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v610 = stablehlo.constant dense<0.0> : tensor<64x31360xf32>
    %v611 = stablehlo.maximum %v609, %v610 : tensor<64x31360xf32>
    %v612 = stablehlo.reshape %v611 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v613 = stablehlo.convolution(%v612, %u6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<64x640x14x14xf32>
    %v614 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v615 = stablehlo.add %v613, %v614 : tensor<64x640x14x14xf32>
    %v616 = stablehlo.reshape %v615 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v617 = stablehlo.reshape %v616 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v618 = stablehlo.constant dense<0.0> : tensor<f32>
    %v619 = stablehlo.constant dense<12544.0> : tensor<64x640x14x14xf32>
    %v620 = stablehlo.constant dense<1.0e-5> : tensor<64x640x14x14xf32>
    %v621 = stablehlo.reduce(%v617 init: %v618) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v622 = stablehlo.broadcast_in_dim %v621, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v623 = stablehlo.divide %v622, %v619 : tensor<64x640x14x14xf32>
    %v624 = stablehlo.subtract %v617, %v623 : tensor<64x640x14x14xf32>
    %v625 = stablehlo.multiply %v624, %v624 : tensor<64x640x14x14xf32>
    %v626 = stablehlo.reduce(%v625 init: %v618) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v627 = stablehlo.broadcast_in_dim %v626, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v628 = stablehlo.divide %v627, %v619 : tensor<64x640x14x14xf32>
    %v629 = stablehlo.add %v628, %v620 : tensor<64x640x14x14xf32>
    %v630 = stablehlo.rsqrt %v629 : tensor<64x640x14x14xf32>
    %v631 = stablehlo.multiply %v624, %v630 : tensor<64x640x14x14xf32>
    %v632 = stablehlo.broadcast_in_dim %u6eg, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v633 = stablehlo.broadcast_in_dim %u6ebt, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v634 = stablehlo.multiply %v631, %v632 : tensor<64x640x14x14xf32>
    %v635 = stablehlo.add %v634, %v633 : tensor<64x640x14x14xf32>
    %v636 = stablehlo.reshape %v635 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v637 = stablehlo.constant dense<0.0> : tensor<64x125440xf32>
    %v638 = stablehlo.maximum %v636, %v637 : tensor<64x125440xf32>
    %v639 = stablehlo.reshape %v638 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v640 = stablehlo.convolution(%v639, %u6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<64x160x14x14xf32>
    %v641 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v642 = stablehlo.add %v640, %v641 : tensor<64x160x14x14xf32>
    %v643 = stablehlo.reshape %v642 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v644 = stablehlo.reshape %v643 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v645 = stablehlo.constant dense<0.0> : tensor<f32>
    %v646 = stablehlo.constant dense<12544.0> : tensor<64x160x14x14xf32>
    %v647 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v648 = stablehlo.reduce(%v644 init: %v645) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v649 = stablehlo.broadcast_in_dim %v648, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v650 = stablehlo.divide %v649, %v646 : tensor<64x160x14x14xf32>
    %v651 = stablehlo.subtract %v644, %v650 : tensor<64x160x14x14xf32>
    %v652 = stablehlo.multiply %v651, %v651 : tensor<64x160x14x14xf32>
    %v653 = stablehlo.reduce(%v652 init: %v645) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v654 = stablehlo.broadcast_in_dim %v653, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v655 = stablehlo.divide %v654, %v646 : tensor<64x160x14x14xf32>
    %v656 = stablehlo.add %v655, %v647 : tensor<64x160x14x14xf32>
    %v657 = stablehlo.rsqrt %v656 : tensor<64x160x14x14xf32>
    %v658 = stablehlo.multiply %v651, %v657 : tensor<64x160x14x14xf32>
    %v659 = stablehlo.broadcast_in_dim %u6pg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v660 = stablehlo.broadcast_in_dim %u6pbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v661 = stablehlo.multiply %v658, %v659 : tensor<64x160x14x14xf32>
    %v662 = stablehlo.add %v661, %v660 : tensor<64x160x14x14xf32>
    %v663 = stablehlo.reshape %v662 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v664 = stablehlo.add %v663, %v584 : tensor<64x31360xf32>
    %v665 = stablehlo.reshape %v664 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v666 = stablehlo.convolution(%v665, %u7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<64x640x14x14xf32>
    %v667 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v668 = stablehlo.add %v666, %v667 : tensor<64x640x14x14xf32>
    %v669 = stablehlo.reshape %v668 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v670 = stablehlo.reshape %v669 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v671 = stablehlo.constant dense<0.0> : tensor<f32>
    %v672 = stablehlo.constant dense<12544.0> : tensor<64x640x14x14xf32>
    %v673 = stablehlo.constant dense<1.0e-5> : tensor<64x640x14x14xf32>
    %v674 = stablehlo.reduce(%v670 init: %v671) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v675 = stablehlo.broadcast_in_dim %v674, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v676 = stablehlo.divide %v675, %v672 : tensor<64x640x14x14xf32>
    %v677 = stablehlo.subtract %v670, %v676 : tensor<64x640x14x14xf32>
    %v678 = stablehlo.multiply %v677, %v677 : tensor<64x640x14x14xf32>
    %v679 = stablehlo.reduce(%v678 init: %v671) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v680 = stablehlo.broadcast_in_dim %v679, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v681 = stablehlo.divide %v680, %v672 : tensor<64x640x14x14xf32>
    %v682 = stablehlo.add %v681, %v673 : tensor<64x640x14x14xf32>
    %v683 = stablehlo.rsqrt %v682 : tensor<64x640x14x14xf32>
    %v684 = stablehlo.multiply %v677, %v683 : tensor<64x640x14x14xf32>
    %v685 = stablehlo.broadcast_in_dim %u7eg, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v686 = stablehlo.broadcast_in_dim %u7ebt, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v687 = stablehlo.multiply %v684, %v685 : tensor<64x640x14x14xf32>
    %v688 = stablehlo.add %v687, %v686 : tensor<64x640x14x14xf32>
    %v689 = stablehlo.reshape %v688 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v690 = stablehlo.constant dense<0.0> : tensor<64x125440xf32>
    %v691 = stablehlo.maximum %v689, %v690 : tensor<64x125440xf32>
    %v692 = stablehlo.reshape %v691 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v693 = stablehlo.convolution(%v692, %u7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 640 : i64} : (tensor<64x640x14x14xf32>, tensor<640x1x3x3xf32>) -> tensor<64x640x14x14xf32>
    %v694 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v695 = stablehlo.add %v693, %v694 : tensor<64x640x14x14xf32>
    %v696 = stablehlo.reshape %v695 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v697 = stablehlo.reshape %v696 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v698 = stablehlo.constant dense<0.0> : tensor<f32>
    %v699 = stablehlo.constant dense<12544.0> : tensor<64x640x14x14xf32>
    %v700 = stablehlo.constant dense<1.0e-5> : tensor<64x640x14x14xf32>
    %v701 = stablehlo.reduce(%v697 init: %v698) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v702 = stablehlo.broadcast_in_dim %v701, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v703 = stablehlo.divide %v702, %v699 : tensor<64x640x14x14xf32>
    %v704 = stablehlo.subtract %v697, %v703 : tensor<64x640x14x14xf32>
    %v705 = stablehlo.multiply %v704, %v704 : tensor<64x640x14x14xf32>
    %v706 = stablehlo.reduce(%v705 init: %v698) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v707 = stablehlo.broadcast_in_dim %v706, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v708 = stablehlo.divide %v707, %v699 : tensor<64x640x14x14xf32>
    %v709 = stablehlo.add %v708, %v700 : tensor<64x640x14x14xf32>
    %v710 = stablehlo.rsqrt %v709 : tensor<64x640x14x14xf32>
    %v711 = stablehlo.multiply %v704, %v710 : tensor<64x640x14x14xf32>
    %v712 = stablehlo.broadcast_in_dim %u7dg, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v713 = stablehlo.broadcast_in_dim %u7dbt, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v714 = stablehlo.multiply %v711, %v712 : tensor<64x640x14x14xf32>
    %v715 = stablehlo.add %v714, %v713 : tensor<64x640x14x14xf32>
    %v716 = stablehlo.reshape %v715 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v717 = stablehlo.constant dense<0.0> : tensor<64x125440xf32>
    %v718 = stablehlo.maximum %v716, %v717 : tensor<64x125440xf32>
    %v719 = stablehlo.reshape %v718 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v720 = stablehlo.convolution(%v719, %u7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<64x160x14x14xf32>
    %v721 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v722 = stablehlo.add %v720, %v721 : tensor<64x160x14x14xf32>
    %v723 = stablehlo.reshape %v722 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v724 = stablehlo.reshape %v723 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v725 = stablehlo.constant dense<0.0> : tensor<f32>
    %v726 = stablehlo.constant dense<12544.0> : tensor<64x160x14x14xf32>
    %v727 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v728 = stablehlo.reduce(%v724 init: %v725) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v729 = stablehlo.broadcast_in_dim %v728, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v730 = stablehlo.divide %v729, %v726 : tensor<64x160x14x14xf32>
    %v731 = stablehlo.subtract %v724, %v730 : tensor<64x160x14x14xf32>
    %v732 = stablehlo.multiply %v731, %v731 : tensor<64x160x14x14xf32>
    %v733 = stablehlo.reduce(%v732 init: %v725) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v734 = stablehlo.broadcast_in_dim %v733, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v735 = stablehlo.divide %v734, %v726 : tensor<64x160x14x14xf32>
    %v736 = stablehlo.add %v735, %v727 : tensor<64x160x14x14xf32>
    %v737 = stablehlo.rsqrt %v736 : tensor<64x160x14x14xf32>
    %v738 = stablehlo.multiply %v731, %v737 : tensor<64x160x14x14xf32>
    %v739 = stablehlo.broadcast_in_dim %u7pg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v740 = stablehlo.broadcast_in_dim %u7pbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v741 = stablehlo.multiply %v738, %v739 : tensor<64x160x14x14xf32>
    %v742 = stablehlo.add %v741, %v740 : tensor<64x160x14x14xf32>
    %v743 = stablehlo.reshape %v742 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v744 = stablehlo.add %v743, %v664 : tensor<64x31360xf32>
    %v745 = stablehlo.reshape %v744 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v746 = stablehlo.convolution(%v745, %u8qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<64x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<64x160x14x14xf32>
    %v747 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v748 = stablehlo.add %v746, %v747 : tensor<64x160x14x14xf32>
    %v749 = stablehlo.reshape %v748 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v750 = stablehlo.reshape %v749 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v751 = stablehlo.constant dense<0.0> : tensor<f32>
    %v752 = stablehlo.constant dense<12544.0> : tensor<64x160x14x14xf32>
    %v753 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v754 = stablehlo.reduce(%v750 init: %v751) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v755 = stablehlo.broadcast_in_dim %v754, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v756 = stablehlo.divide %v755, %v752 : tensor<64x160x14x14xf32>
    %v757 = stablehlo.subtract %v750, %v756 : tensor<64x160x14x14xf32>
    %v758 = stablehlo.multiply %v757, %v757 : tensor<64x160x14x14xf32>
    %v759 = stablehlo.reduce(%v758 init: %v751) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v760 = stablehlo.broadcast_in_dim %v759, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v761 = stablehlo.divide %v760, %v752 : tensor<64x160x14x14xf32>
    %v762 = stablehlo.add %v761, %v753 : tensor<64x160x14x14xf32>
    %v763 = stablehlo.rsqrt %v762 : tensor<64x160x14x14xf32>
    %v764 = stablehlo.multiply %v757, %v763 : tensor<64x160x14x14xf32>
    %v765 = stablehlo.broadcast_in_dim %u8qg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v766 = stablehlo.broadcast_in_dim %u8qbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v767 = stablehlo.multiply %v764, %v765 : tensor<64x160x14x14xf32>
    %v768 = stablehlo.add %v767, %v766 : tensor<64x160x14x14xf32>
    %v769 = stablehlo.reshape %v768 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v770 = stablehlo.constant dense<0.0> : tensor<64x31360xf32>
    %v771 = stablehlo.maximum %v769, %v770 : tensor<64x31360xf32>
    %v772 = stablehlo.reshape %v771 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v773 = stablehlo.convolution(%v772, %u8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<64x640x14x14xf32>
    %v774 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v775 = stablehlo.add %v773, %v774 : tensor<64x640x14x14xf32>
    %v776 = stablehlo.reshape %v775 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v777 = stablehlo.reshape %v776 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v778 = stablehlo.constant dense<0.0> : tensor<f32>
    %v779 = stablehlo.constant dense<12544.0> : tensor<64x640x14x14xf32>
    %v780 = stablehlo.constant dense<1.0e-5> : tensor<64x640x14x14xf32>
    %v781 = stablehlo.reduce(%v777 init: %v778) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v782 = stablehlo.broadcast_in_dim %v781, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v783 = stablehlo.divide %v782, %v779 : tensor<64x640x14x14xf32>
    %v784 = stablehlo.subtract %v777, %v783 : tensor<64x640x14x14xf32>
    %v785 = stablehlo.multiply %v784, %v784 : tensor<64x640x14x14xf32>
    %v786 = stablehlo.reduce(%v785 init: %v778) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v787 = stablehlo.broadcast_in_dim %v786, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v788 = stablehlo.divide %v787, %v779 : tensor<64x640x14x14xf32>
    %v789 = stablehlo.add %v788, %v780 : tensor<64x640x14x14xf32>
    %v790 = stablehlo.rsqrt %v789 : tensor<64x640x14x14xf32>
    %v791 = stablehlo.multiply %v784, %v790 : tensor<64x640x14x14xf32>
    %v792 = stablehlo.broadcast_in_dim %u8eg, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v793 = stablehlo.broadcast_in_dim %u8ebt, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v794 = stablehlo.multiply %v791, %v792 : tensor<64x640x14x14xf32>
    %v795 = stablehlo.add %v794, %v793 : tensor<64x640x14x14xf32>
    %v796 = stablehlo.reshape %v795 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v797 = stablehlo.constant dense<0.0> : tensor<64x125440xf32>
    %v798 = stablehlo.maximum %v796, %v797 : tensor<64x125440xf32>
    %v799 = stablehlo.reshape %v798 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v800 = stablehlo.convolution(%v799, %u8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<64x160x14x14xf32>
    %v801 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v802 = stablehlo.add %v800, %v801 : tensor<64x160x14x14xf32>
    %v803 = stablehlo.reshape %v802 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v804 = stablehlo.reshape %v803 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v805 = stablehlo.constant dense<0.0> : tensor<f32>
    %v806 = stablehlo.constant dense<12544.0> : tensor<64x160x14x14xf32>
    %v807 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v808 = stablehlo.reduce(%v804 init: %v805) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v809 = stablehlo.broadcast_in_dim %v808, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v810 = stablehlo.divide %v809, %v806 : tensor<64x160x14x14xf32>
    %v811 = stablehlo.subtract %v804, %v810 : tensor<64x160x14x14xf32>
    %v812 = stablehlo.multiply %v811, %v811 : tensor<64x160x14x14xf32>
    %v813 = stablehlo.reduce(%v812 init: %v805) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v814 = stablehlo.broadcast_in_dim %v813, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v815 = stablehlo.divide %v814, %v806 : tensor<64x160x14x14xf32>
    %v816 = stablehlo.add %v815, %v807 : tensor<64x160x14x14xf32>
    %v817 = stablehlo.rsqrt %v816 : tensor<64x160x14x14xf32>
    %v818 = stablehlo.multiply %v811, %v817 : tensor<64x160x14x14xf32>
    %v819 = stablehlo.broadcast_in_dim %u8pg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v820 = stablehlo.broadcast_in_dim %u8pbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v821 = stablehlo.multiply %v818, %v819 : tensor<64x160x14x14xf32>
    %v822 = stablehlo.add %v821, %v820 : tensor<64x160x14x14xf32>
    %v823 = stablehlo.reshape %v822 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v824 = stablehlo.add %v823, %v744 : tensor<64x31360xf32>
    %v825 = stablehlo.reshape %v824 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v826 = stablehlo.convolution(%v825, %u9eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<64x640x14x14xf32>
    %v827 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v828 = stablehlo.add %v826, %v827 : tensor<64x640x14x14xf32>
    %v829 = stablehlo.reshape %v828 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v830 = stablehlo.reshape %v829 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v831 = stablehlo.constant dense<0.0> : tensor<f32>
    %v832 = stablehlo.constant dense<12544.0> : tensor<64x640x14x14xf32>
    %v833 = stablehlo.constant dense<1.0e-5> : tensor<64x640x14x14xf32>
    %v834 = stablehlo.reduce(%v830 init: %v831) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v835 = stablehlo.broadcast_in_dim %v834, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v836 = stablehlo.divide %v835, %v832 : tensor<64x640x14x14xf32>
    %v837 = stablehlo.subtract %v830, %v836 : tensor<64x640x14x14xf32>
    %v838 = stablehlo.multiply %v837, %v837 : tensor<64x640x14x14xf32>
    %v839 = stablehlo.reduce(%v838 init: %v831) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v840 = stablehlo.broadcast_in_dim %v839, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v841 = stablehlo.divide %v840, %v832 : tensor<64x640x14x14xf32>
    %v842 = stablehlo.add %v841, %v833 : tensor<64x640x14x14xf32>
    %v843 = stablehlo.rsqrt %v842 : tensor<64x640x14x14xf32>
    %v844 = stablehlo.multiply %v837, %v843 : tensor<64x640x14x14xf32>
    %v845 = stablehlo.broadcast_in_dim %u9eg, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v846 = stablehlo.broadcast_in_dim %u9ebt, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v847 = stablehlo.multiply %v844, %v845 : tensor<64x640x14x14xf32>
    %v848 = stablehlo.add %v847, %v846 : tensor<64x640x14x14xf32>
    %v849 = stablehlo.reshape %v848 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v850 = stablehlo.constant dense<0.0> : tensor<64x125440xf32>
    %v851 = stablehlo.maximum %v849, %v850 : tensor<64x125440xf32>
    %v852 = stablehlo.reshape %v851 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v853 = stablehlo.convolution(%v852, %u9pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<64x160x14x14xf32>
    %v854 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v855 = stablehlo.add %v853, %v854 : tensor<64x160x14x14xf32>
    %v856 = stablehlo.reshape %v855 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v857 = stablehlo.reshape %v856 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v858 = stablehlo.constant dense<0.0> : tensor<f32>
    %v859 = stablehlo.constant dense<12544.0> : tensor<64x160x14x14xf32>
    %v860 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v861 = stablehlo.reduce(%v857 init: %v858) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v862 = stablehlo.broadcast_in_dim %v861, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v863 = stablehlo.divide %v862, %v859 : tensor<64x160x14x14xf32>
    %v864 = stablehlo.subtract %v857, %v863 : tensor<64x160x14x14xf32>
    %v865 = stablehlo.multiply %v864, %v864 : tensor<64x160x14x14xf32>
    %v866 = stablehlo.reduce(%v865 init: %v858) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v867 = stablehlo.broadcast_in_dim %v866, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v868 = stablehlo.divide %v867, %v859 : tensor<64x160x14x14xf32>
    %v869 = stablehlo.add %v868, %v860 : tensor<64x160x14x14xf32>
    %v870 = stablehlo.rsqrt %v869 : tensor<64x160x14x14xf32>
    %v871 = stablehlo.multiply %v864, %v870 : tensor<64x160x14x14xf32>
    %v872 = stablehlo.broadcast_in_dim %u9pg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v873 = stablehlo.broadcast_in_dim %u9pbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v874 = stablehlo.multiply %v871, %v872 : tensor<64x160x14x14xf32>
    %v875 = stablehlo.add %v874, %v873 : tensor<64x160x14x14xf32>
    %v876 = stablehlo.reshape %v875 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v877 = stablehlo.add %v876, %v824 : tensor<64x31360xf32>
    %v878 = stablehlo.reshape %v877 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v879 = stablehlo.convolution(%v878, %u10qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<64x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<64x160x14x14xf32>
    %v880 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v881 = stablehlo.add %v879, %v880 : tensor<64x160x14x14xf32>
    %v882 = stablehlo.reshape %v881 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v883 = stablehlo.reshape %v882 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v884 = stablehlo.constant dense<0.0> : tensor<f32>
    %v885 = stablehlo.constant dense<12544.0> : tensor<64x160x14x14xf32>
    %v886 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v887 = stablehlo.reduce(%v883 init: %v884) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v888 = stablehlo.broadcast_in_dim %v887, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v889 = stablehlo.divide %v888, %v885 : tensor<64x160x14x14xf32>
    %v890 = stablehlo.subtract %v883, %v889 : tensor<64x160x14x14xf32>
    %v891 = stablehlo.multiply %v890, %v890 : tensor<64x160x14x14xf32>
    %v892 = stablehlo.reduce(%v891 init: %v884) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v893 = stablehlo.broadcast_in_dim %v892, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v894 = stablehlo.divide %v893, %v885 : tensor<64x160x14x14xf32>
    %v895 = stablehlo.add %v894, %v886 : tensor<64x160x14x14xf32>
    %v896 = stablehlo.rsqrt %v895 : tensor<64x160x14x14xf32>
    %v897 = stablehlo.multiply %v890, %v896 : tensor<64x160x14x14xf32>
    %v898 = stablehlo.broadcast_in_dim %u10qg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v899 = stablehlo.broadcast_in_dim %u10qbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v900 = stablehlo.multiply %v897, %v898 : tensor<64x160x14x14xf32>
    %v901 = stablehlo.add %v900, %v899 : tensor<64x160x14x14xf32>
    %v902 = stablehlo.reshape %v901 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v903 = stablehlo.constant dense<0.0> : tensor<64x31360xf32>
    %v904 = stablehlo.maximum %v902, %v903 : tensor<64x31360xf32>
    %v905 = stablehlo.reshape %v904 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v906 = stablehlo.convolution(%v905, %u10eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<64x640x14x14xf32>
    %v907 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v908 = stablehlo.add %v906, %v907 : tensor<64x640x14x14xf32>
    %v909 = stablehlo.reshape %v908 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v910 = stablehlo.reshape %v909 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v911 = stablehlo.constant dense<0.0> : tensor<f32>
    %v912 = stablehlo.constant dense<12544.0> : tensor<64x640x14x14xf32>
    %v913 = stablehlo.constant dense<1.0e-5> : tensor<64x640x14x14xf32>
    %v914 = stablehlo.reduce(%v910 init: %v911) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v915 = stablehlo.broadcast_in_dim %v914, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v916 = stablehlo.divide %v915, %v912 : tensor<64x640x14x14xf32>
    %v917 = stablehlo.subtract %v910, %v916 : tensor<64x640x14x14xf32>
    %v918 = stablehlo.multiply %v917, %v917 : tensor<64x640x14x14xf32>
    %v919 = stablehlo.reduce(%v918 init: %v911) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v920 = stablehlo.broadcast_in_dim %v919, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v921 = stablehlo.divide %v920, %v912 : tensor<64x640x14x14xf32>
    %v922 = stablehlo.add %v921, %v913 : tensor<64x640x14x14xf32>
    %v923 = stablehlo.rsqrt %v922 : tensor<64x640x14x14xf32>
    %v924 = stablehlo.multiply %v917, %v923 : tensor<64x640x14x14xf32>
    %v925 = stablehlo.broadcast_in_dim %u10eg, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v926 = stablehlo.broadcast_in_dim %u10ebt, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v927 = stablehlo.multiply %v924, %v925 : tensor<64x640x14x14xf32>
    %v928 = stablehlo.add %v927, %v926 : tensor<64x640x14x14xf32>
    %v929 = stablehlo.reshape %v928 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v930 = stablehlo.constant dense<0.0> : tensor<64x125440xf32>
    %v931 = stablehlo.maximum %v929, %v930 : tensor<64x125440xf32>
    %v932 = stablehlo.reshape %v931 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v933 = stablehlo.convolution(%v932, %u10dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 640 : i64} : (tensor<64x640x14x14xf32>, tensor<640x1x3x3xf32>) -> tensor<64x640x14x14xf32>
    %v934 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v935 = stablehlo.add %v933, %v934 : tensor<64x640x14x14xf32>
    %v936 = stablehlo.reshape %v935 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v937 = stablehlo.reshape %v936 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v938 = stablehlo.constant dense<0.0> : tensor<f32>
    %v939 = stablehlo.constant dense<12544.0> : tensor<64x640x14x14xf32>
    %v940 = stablehlo.constant dense<1.0e-5> : tensor<64x640x14x14xf32>
    %v941 = stablehlo.reduce(%v937 init: %v938) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v942 = stablehlo.broadcast_in_dim %v941, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v943 = stablehlo.divide %v942, %v939 : tensor<64x640x14x14xf32>
    %v944 = stablehlo.subtract %v937, %v943 : tensor<64x640x14x14xf32>
    %v945 = stablehlo.multiply %v944, %v944 : tensor<64x640x14x14xf32>
    %v946 = stablehlo.reduce(%v945 init: %v938) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v947 = stablehlo.broadcast_in_dim %v946, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v948 = stablehlo.divide %v947, %v939 : tensor<64x640x14x14xf32>
    %v949 = stablehlo.add %v948, %v940 : tensor<64x640x14x14xf32>
    %v950 = stablehlo.rsqrt %v949 : tensor<64x640x14x14xf32>
    %v951 = stablehlo.multiply %v944, %v950 : tensor<64x640x14x14xf32>
    %v952 = stablehlo.broadcast_in_dim %u10dg, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v953 = stablehlo.broadcast_in_dim %u10dbt, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v954 = stablehlo.multiply %v951, %v952 : tensor<64x640x14x14xf32>
    %v955 = stablehlo.add %v954, %v953 : tensor<64x640x14x14xf32>
    %v956 = stablehlo.reshape %v955 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v957 = stablehlo.constant dense<0.0> : tensor<64x125440xf32>
    %v958 = stablehlo.maximum %v956, %v957 : tensor<64x125440xf32>
    %v959 = stablehlo.reshape %v958 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v960 = stablehlo.convolution(%v959, %u10pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<64x160x14x14xf32>
    %v961 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v962 = stablehlo.add %v960, %v961 : tensor<64x160x14x14xf32>
    %v963 = stablehlo.reshape %v962 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v964 = stablehlo.reshape %v963 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v965 = stablehlo.constant dense<0.0> : tensor<f32>
    %v966 = stablehlo.constant dense<12544.0> : tensor<64x160x14x14xf32>
    %v967 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v968 = stablehlo.reduce(%v964 init: %v965) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v969 = stablehlo.broadcast_in_dim %v968, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v970 = stablehlo.divide %v969, %v966 : tensor<64x160x14x14xf32>
    %v971 = stablehlo.subtract %v964, %v970 : tensor<64x160x14x14xf32>
    %v972 = stablehlo.multiply %v971, %v971 : tensor<64x160x14x14xf32>
    %v973 = stablehlo.reduce(%v972 init: %v965) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v974 = stablehlo.broadcast_in_dim %v973, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v975 = stablehlo.divide %v974, %v966 : tensor<64x160x14x14xf32>
    %v976 = stablehlo.add %v975, %v967 : tensor<64x160x14x14xf32>
    %v977 = stablehlo.rsqrt %v976 : tensor<64x160x14x14xf32>
    %v978 = stablehlo.multiply %v971, %v977 : tensor<64x160x14x14xf32>
    %v979 = stablehlo.broadcast_in_dim %u10pg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v980 = stablehlo.broadcast_in_dim %u10pbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v981 = stablehlo.multiply %v978, %v979 : tensor<64x160x14x14xf32>
    %v982 = stablehlo.add %v981, %v980 : tensor<64x160x14x14xf32>
    %v983 = stablehlo.reshape %v982 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v984 = stablehlo.add %v983, %v877 : tensor<64x31360xf32>
    %v985 = stablehlo.reshape %v984 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v986 = stablehlo.convolution(%v985, %u11qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<64x160x14x14xf32>, tensor<160x1x5x5xf32>) -> tensor<64x160x7x7xf32>
    %v987 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x7x7xf32>
    %v988 = stablehlo.add %v986, %v987 : tensor<64x160x7x7xf32>
    %v989 = stablehlo.reshape %v988 : (tensor<64x160x7x7xf32>) -> tensor<64x7840xf32>
    %v990 = stablehlo.reshape %v989 : (tensor<64x7840xf32>) -> tensor<64x160x7x7xf32>
    %v991 = stablehlo.constant dense<0.0> : tensor<f32>
    %v992 = stablehlo.constant dense<3136.0> : tensor<64x160x7x7xf32>
    %v993 = stablehlo.constant dense<1.0e-5> : tensor<64x160x7x7xf32>
    %v994 = stablehlo.reduce(%v990 init: %v991) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v995 = stablehlo.broadcast_in_dim %v994, dims = [1] : (tensor<160xf32>) -> tensor<64x160x7x7xf32>
    %v996 = stablehlo.divide %v995, %v992 : tensor<64x160x7x7xf32>
    %v997 = stablehlo.subtract %v990, %v996 : tensor<64x160x7x7xf32>
    %v998 = stablehlo.multiply %v997, %v997 : tensor<64x160x7x7xf32>
    %v999 = stablehlo.reduce(%v998 init: %v991) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v1000 = stablehlo.broadcast_in_dim %v999, dims = [1] : (tensor<160xf32>) -> tensor<64x160x7x7xf32>
    %v1001 = stablehlo.divide %v1000, %v992 : tensor<64x160x7x7xf32>
    %v1002 = stablehlo.add %v1001, %v993 : tensor<64x160x7x7xf32>
    %v1003 = stablehlo.rsqrt %v1002 : tensor<64x160x7x7xf32>
    %v1004 = stablehlo.multiply %v997, %v1003 : tensor<64x160x7x7xf32>
    %v1005 = stablehlo.broadcast_in_dim %u11qg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x7x7xf32>
    %v1006 = stablehlo.broadcast_in_dim %u11qbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x7x7xf32>
    %v1007 = stablehlo.multiply %v1004, %v1005 : tensor<64x160x7x7xf32>
    %v1008 = stablehlo.add %v1007, %v1006 : tensor<64x160x7x7xf32>
    %v1009 = stablehlo.reshape %v1008 : (tensor<64x160x7x7xf32>) -> tensor<64x7840xf32>
    %v1010 = stablehlo.constant dense<0.0> : tensor<64x7840xf32>
    %v1011 = stablehlo.maximum %v1009, %v1010 : tensor<64x7840xf32>
    %v1012 = stablehlo.reshape %v1011 : (tensor<64x7840xf32>) -> tensor<64x160x7x7xf32>
    %v1013 = stablehlo.convolution(%v1012, %u11eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<64x960x7x7xf32>
    %v1014 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1015 = stablehlo.add %v1013, %v1014 : tensor<64x960x7x7xf32>
    %v1016 = stablehlo.reshape %v1015 : (tensor<64x960x7x7xf32>) -> tensor<64x47040xf32>
    %v1017 = stablehlo.reshape %v1016 : (tensor<64x47040xf32>) -> tensor<64x960x7x7xf32>
    %v1018 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1019 = stablehlo.constant dense<3136.0> : tensor<64x960x7x7xf32>
    %v1020 = stablehlo.constant dense<1.0e-5> : tensor<64x960x7x7xf32>
    %v1021 = stablehlo.reduce(%v1017 init: %v1018) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1022 = stablehlo.broadcast_in_dim %v1021, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1023 = stablehlo.divide %v1022, %v1019 : tensor<64x960x7x7xf32>
    %v1024 = stablehlo.subtract %v1017, %v1023 : tensor<64x960x7x7xf32>
    %v1025 = stablehlo.multiply %v1024, %v1024 : tensor<64x960x7x7xf32>
    %v1026 = stablehlo.reduce(%v1025 init: %v1018) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1027 = stablehlo.broadcast_in_dim %v1026, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1028 = stablehlo.divide %v1027, %v1019 : tensor<64x960x7x7xf32>
    %v1029 = stablehlo.add %v1028, %v1020 : tensor<64x960x7x7xf32>
    %v1030 = stablehlo.rsqrt %v1029 : tensor<64x960x7x7xf32>
    %v1031 = stablehlo.multiply %v1024, %v1030 : tensor<64x960x7x7xf32>
    %v1032 = stablehlo.broadcast_in_dim %u11eg, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1033 = stablehlo.broadcast_in_dim %u11ebt, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1034 = stablehlo.multiply %v1031, %v1032 : tensor<64x960x7x7xf32>
    %v1035 = stablehlo.add %v1034, %v1033 : tensor<64x960x7x7xf32>
    %v1036 = stablehlo.reshape %v1035 : (tensor<64x960x7x7xf32>) -> tensor<64x47040xf32>
    %v1037 = stablehlo.constant dense<0.0> : tensor<64x47040xf32>
    %v1038 = stablehlo.maximum %v1036, %v1037 : tensor<64x47040xf32>
    %v1039 = stablehlo.reshape %v1038 : (tensor<64x47040xf32>) -> tensor<64x960x7x7xf32>
    %v1040 = stablehlo.convolution(%v1039, %u11dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<64x960x7x7xf32>, tensor<960x1x5x5xf32>) -> tensor<64x960x7x7xf32>
    %v1041 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1042 = stablehlo.add %v1040, %v1041 : tensor<64x960x7x7xf32>
    %v1043 = stablehlo.reshape %v1042 : (tensor<64x960x7x7xf32>) -> tensor<64x47040xf32>
    %v1044 = stablehlo.reshape %v1043 : (tensor<64x47040xf32>) -> tensor<64x960x7x7xf32>
    %v1045 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1046 = stablehlo.constant dense<3136.0> : tensor<64x960x7x7xf32>
    %v1047 = stablehlo.constant dense<1.0e-5> : tensor<64x960x7x7xf32>
    %v1048 = stablehlo.reduce(%v1044 init: %v1045) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1049 = stablehlo.broadcast_in_dim %v1048, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1050 = stablehlo.divide %v1049, %v1046 : tensor<64x960x7x7xf32>
    %v1051 = stablehlo.subtract %v1044, %v1050 : tensor<64x960x7x7xf32>
    %v1052 = stablehlo.multiply %v1051, %v1051 : tensor<64x960x7x7xf32>
    %v1053 = stablehlo.reduce(%v1052 init: %v1045) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1054 = stablehlo.broadcast_in_dim %v1053, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1055 = stablehlo.divide %v1054, %v1046 : tensor<64x960x7x7xf32>
    %v1056 = stablehlo.add %v1055, %v1047 : tensor<64x960x7x7xf32>
    %v1057 = stablehlo.rsqrt %v1056 : tensor<64x960x7x7xf32>
    %v1058 = stablehlo.multiply %v1051, %v1057 : tensor<64x960x7x7xf32>
    %v1059 = stablehlo.broadcast_in_dim %u11dg, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1060 = stablehlo.broadcast_in_dim %u11dbt, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1061 = stablehlo.multiply %v1058, %v1059 : tensor<64x960x7x7xf32>
    %v1062 = stablehlo.add %v1061, %v1060 : tensor<64x960x7x7xf32>
    %v1063 = stablehlo.reshape %v1062 : (tensor<64x960x7x7xf32>) -> tensor<64x47040xf32>
    %v1064 = stablehlo.constant dense<0.0> : tensor<64x47040xf32>
    %v1065 = stablehlo.maximum %v1063, %v1064 : tensor<64x47040xf32>
    %v1066 = stablehlo.reshape %v1065 : (tensor<64x47040xf32>) -> tensor<64x960x7x7xf32>
    %v1067 = stablehlo.convolution(%v1066, %u11pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x960x7x7xf32>, tensor<256x960x1x1xf32>) -> tensor<64x256x7x7xf32>
    %v1068 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1069 = stablehlo.add %v1067, %v1068 : tensor<64x256x7x7xf32>
    %v1070 = stablehlo.reshape %v1069 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1071 = stablehlo.reshape %v1070 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1072 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1073 = stablehlo.constant dense<3136.0> : tensor<64x256x7x7xf32>
    %v1074 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v1075 = stablehlo.reduce(%v1071 init: %v1072) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1076 = stablehlo.broadcast_in_dim %v1075, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1077 = stablehlo.divide %v1076, %v1073 : tensor<64x256x7x7xf32>
    %v1078 = stablehlo.subtract %v1071, %v1077 : tensor<64x256x7x7xf32>
    %v1079 = stablehlo.multiply %v1078, %v1078 : tensor<64x256x7x7xf32>
    %v1080 = stablehlo.reduce(%v1079 init: %v1072) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1081 = stablehlo.broadcast_in_dim %v1080, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1082 = stablehlo.divide %v1081, %v1073 : tensor<64x256x7x7xf32>
    %v1083 = stablehlo.add %v1082, %v1074 : tensor<64x256x7x7xf32>
    %v1084 = stablehlo.rsqrt %v1083 : tensor<64x256x7x7xf32>
    %v1085 = stablehlo.multiply %v1078, %v1084 : tensor<64x256x7x7xf32>
    %v1086 = stablehlo.broadcast_in_dim %u11pg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1087 = stablehlo.broadcast_in_dim %u11pbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1088 = stablehlo.multiply %v1085, %v1086 : tensor<64x256x7x7xf32>
    %v1089 = stablehlo.add %v1088, %v1087 : tensor<64x256x7x7xf32>
    %v1090 = stablehlo.reshape %v1089 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1091 = stablehlo.reshape %v1090 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1092 = stablehlo.convolution(%v1091, %u12qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<64x256x7x7xf32>, tensor<256x1x5x5xf32>) -> tensor<64x256x7x7xf32>
    %v1093 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1094 = stablehlo.add %v1092, %v1093 : tensor<64x256x7x7xf32>
    %v1095 = stablehlo.reshape %v1094 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1096 = stablehlo.reshape %v1095 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1097 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1098 = stablehlo.constant dense<3136.0> : tensor<64x256x7x7xf32>
    %v1099 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v1100 = stablehlo.reduce(%v1096 init: %v1097) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1101 = stablehlo.broadcast_in_dim %v1100, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1102 = stablehlo.divide %v1101, %v1098 : tensor<64x256x7x7xf32>
    %v1103 = stablehlo.subtract %v1096, %v1102 : tensor<64x256x7x7xf32>
    %v1104 = stablehlo.multiply %v1103, %v1103 : tensor<64x256x7x7xf32>
    %v1105 = stablehlo.reduce(%v1104 init: %v1097) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1106 = stablehlo.broadcast_in_dim %v1105, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1107 = stablehlo.divide %v1106, %v1098 : tensor<64x256x7x7xf32>
    %v1108 = stablehlo.add %v1107, %v1099 : tensor<64x256x7x7xf32>
    %v1109 = stablehlo.rsqrt %v1108 : tensor<64x256x7x7xf32>
    %v1110 = stablehlo.multiply %v1103, %v1109 : tensor<64x256x7x7xf32>
    %v1111 = stablehlo.broadcast_in_dim %u12qg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1112 = stablehlo.broadcast_in_dim %u12qbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1113 = stablehlo.multiply %v1110, %v1111 : tensor<64x256x7x7xf32>
    %v1114 = stablehlo.add %v1113, %v1112 : tensor<64x256x7x7xf32>
    %v1115 = stablehlo.reshape %v1114 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1116 = stablehlo.constant dense<0.0> : tensor<64x12544xf32>
    %v1117 = stablehlo.maximum %v1115, %v1116 : tensor<64x12544xf32>
    %v1118 = stablehlo.reshape %v1117 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1119 = stablehlo.convolution(%v1118, %u12eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<64x1024x7x7xf32>
    %v1120 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1121 = stablehlo.add %v1119, %v1120 : tensor<64x1024x7x7xf32>
    %v1122 = stablehlo.reshape %v1121 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1123 = stablehlo.reshape %v1122 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1124 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1125 = stablehlo.constant dense<3136.0> : tensor<64x1024x7x7xf32>
    %v1126 = stablehlo.constant dense<1.0e-5> : tensor<64x1024x7x7xf32>
    %v1127 = stablehlo.reduce(%v1123 init: %v1124) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1128 = stablehlo.broadcast_in_dim %v1127, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1129 = stablehlo.divide %v1128, %v1125 : tensor<64x1024x7x7xf32>
    %v1130 = stablehlo.subtract %v1123, %v1129 : tensor<64x1024x7x7xf32>
    %v1131 = stablehlo.multiply %v1130, %v1130 : tensor<64x1024x7x7xf32>
    %v1132 = stablehlo.reduce(%v1131 init: %v1124) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1133 = stablehlo.broadcast_in_dim %v1132, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1134 = stablehlo.divide %v1133, %v1125 : tensor<64x1024x7x7xf32>
    %v1135 = stablehlo.add %v1134, %v1126 : tensor<64x1024x7x7xf32>
    %v1136 = stablehlo.rsqrt %v1135 : tensor<64x1024x7x7xf32>
    %v1137 = stablehlo.multiply %v1130, %v1136 : tensor<64x1024x7x7xf32>
    %v1138 = stablehlo.broadcast_in_dim %u12eg, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1139 = stablehlo.broadcast_in_dim %u12ebt, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1140 = stablehlo.multiply %v1137, %v1138 : tensor<64x1024x7x7xf32>
    %v1141 = stablehlo.add %v1140, %v1139 : tensor<64x1024x7x7xf32>
    %v1142 = stablehlo.reshape %v1141 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1143 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1144 = stablehlo.maximum %v1142, %v1143 : tensor<64x50176xf32>
    %v1145 = stablehlo.reshape %v1144 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1146 = stablehlo.convolution(%v1145, %u12dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<64x1024x7x7xf32>, tensor<1024x1x5x5xf32>) -> tensor<64x1024x7x7xf32>
    %v1147 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1148 = stablehlo.add %v1146, %v1147 : tensor<64x1024x7x7xf32>
    %v1149 = stablehlo.reshape %v1148 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1150 = stablehlo.reshape %v1149 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1151 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1152 = stablehlo.constant dense<3136.0> : tensor<64x1024x7x7xf32>
    %v1153 = stablehlo.constant dense<1.0e-5> : tensor<64x1024x7x7xf32>
    %v1154 = stablehlo.reduce(%v1150 init: %v1151) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1155 = stablehlo.broadcast_in_dim %v1154, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1156 = stablehlo.divide %v1155, %v1152 : tensor<64x1024x7x7xf32>
    %v1157 = stablehlo.subtract %v1150, %v1156 : tensor<64x1024x7x7xf32>
    %v1158 = stablehlo.multiply %v1157, %v1157 : tensor<64x1024x7x7xf32>
    %v1159 = stablehlo.reduce(%v1158 init: %v1151) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1160 = stablehlo.broadcast_in_dim %v1159, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1161 = stablehlo.divide %v1160, %v1152 : tensor<64x1024x7x7xf32>
    %v1162 = stablehlo.add %v1161, %v1153 : tensor<64x1024x7x7xf32>
    %v1163 = stablehlo.rsqrt %v1162 : tensor<64x1024x7x7xf32>
    %v1164 = stablehlo.multiply %v1157, %v1163 : tensor<64x1024x7x7xf32>
    %v1165 = stablehlo.broadcast_in_dim %u12dg, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1166 = stablehlo.broadcast_in_dim %u12dbt, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1167 = stablehlo.multiply %v1164, %v1165 : tensor<64x1024x7x7xf32>
    %v1168 = stablehlo.add %v1167, %v1166 : tensor<64x1024x7x7xf32>
    %v1169 = stablehlo.reshape %v1168 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1170 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1171 = stablehlo.maximum %v1169, %v1170 : tensor<64x50176xf32>
    %v1172 = stablehlo.reshape %v1171 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1173 = stablehlo.convolution(%v1172, %u12pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<64x256x7x7xf32>
    %v1174 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1175 = stablehlo.add %v1173, %v1174 : tensor<64x256x7x7xf32>
    %v1176 = stablehlo.reshape %v1175 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1177 = stablehlo.reshape %v1176 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1178 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1179 = stablehlo.constant dense<3136.0> : tensor<64x256x7x7xf32>
    %v1180 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v1181 = stablehlo.reduce(%v1177 init: %v1178) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1182 = stablehlo.broadcast_in_dim %v1181, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1183 = stablehlo.divide %v1182, %v1179 : tensor<64x256x7x7xf32>
    %v1184 = stablehlo.subtract %v1177, %v1183 : tensor<64x256x7x7xf32>
    %v1185 = stablehlo.multiply %v1184, %v1184 : tensor<64x256x7x7xf32>
    %v1186 = stablehlo.reduce(%v1185 init: %v1178) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1187 = stablehlo.broadcast_in_dim %v1186, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1188 = stablehlo.divide %v1187, %v1179 : tensor<64x256x7x7xf32>
    %v1189 = stablehlo.add %v1188, %v1180 : tensor<64x256x7x7xf32>
    %v1190 = stablehlo.rsqrt %v1189 : tensor<64x256x7x7xf32>
    %v1191 = stablehlo.multiply %v1184, %v1190 : tensor<64x256x7x7xf32>
    %v1192 = stablehlo.broadcast_in_dim %u12pg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1193 = stablehlo.broadcast_in_dim %u12pbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1194 = stablehlo.multiply %v1191, %v1192 : tensor<64x256x7x7xf32>
    %v1195 = stablehlo.add %v1194, %v1193 : tensor<64x256x7x7xf32>
    %v1196 = stablehlo.reshape %v1195 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1197 = stablehlo.add %v1196, %v1090 : tensor<64x12544xf32>
    %v1198 = stablehlo.reshape %v1197 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1199 = stablehlo.convolution(%v1198, %u13eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<64x1024x7x7xf32>
    %v1200 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1201 = stablehlo.add %v1199, %v1200 : tensor<64x1024x7x7xf32>
    %v1202 = stablehlo.reshape %v1201 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1203 = stablehlo.reshape %v1202 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1204 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1205 = stablehlo.constant dense<3136.0> : tensor<64x1024x7x7xf32>
    %v1206 = stablehlo.constant dense<1.0e-5> : tensor<64x1024x7x7xf32>
    %v1207 = stablehlo.reduce(%v1203 init: %v1204) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1208 = stablehlo.broadcast_in_dim %v1207, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1209 = stablehlo.divide %v1208, %v1205 : tensor<64x1024x7x7xf32>
    %v1210 = stablehlo.subtract %v1203, %v1209 : tensor<64x1024x7x7xf32>
    %v1211 = stablehlo.multiply %v1210, %v1210 : tensor<64x1024x7x7xf32>
    %v1212 = stablehlo.reduce(%v1211 init: %v1204) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1213 = stablehlo.broadcast_in_dim %v1212, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1214 = stablehlo.divide %v1213, %v1205 : tensor<64x1024x7x7xf32>
    %v1215 = stablehlo.add %v1214, %v1206 : tensor<64x1024x7x7xf32>
    %v1216 = stablehlo.rsqrt %v1215 : tensor<64x1024x7x7xf32>
    %v1217 = stablehlo.multiply %v1210, %v1216 : tensor<64x1024x7x7xf32>
    %v1218 = stablehlo.broadcast_in_dim %u13eg, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1219 = stablehlo.broadcast_in_dim %u13ebt, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1220 = stablehlo.multiply %v1217, %v1218 : tensor<64x1024x7x7xf32>
    %v1221 = stablehlo.add %v1220, %v1219 : tensor<64x1024x7x7xf32>
    %v1222 = stablehlo.reshape %v1221 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1223 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1224 = stablehlo.maximum %v1222, %v1223 : tensor<64x50176xf32>
    %v1225 = stablehlo.reshape %v1224 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1226 = stablehlo.convolution(%v1225, %u13dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<64x1024x7x7xf32>, tensor<1024x1x3x3xf32>) -> tensor<64x1024x7x7xf32>
    %v1227 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1228 = stablehlo.add %v1226, %v1227 : tensor<64x1024x7x7xf32>
    %v1229 = stablehlo.reshape %v1228 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1230 = stablehlo.reshape %v1229 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1231 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1232 = stablehlo.constant dense<3136.0> : tensor<64x1024x7x7xf32>
    %v1233 = stablehlo.constant dense<1.0e-5> : tensor<64x1024x7x7xf32>
    %v1234 = stablehlo.reduce(%v1230 init: %v1231) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1235 = stablehlo.broadcast_in_dim %v1234, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1236 = stablehlo.divide %v1235, %v1232 : tensor<64x1024x7x7xf32>
    %v1237 = stablehlo.subtract %v1230, %v1236 : tensor<64x1024x7x7xf32>
    %v1238 = stablehlo.multiply %v1237, %v1237 : tensor<64x1024x7x7xf32>
    %v1239 = stablehlo.reduce(%v1238 init: %v1231) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1240 = stablehlo.broadcast_in_dim %v1239, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1241 = stablehlo.divide %v1240, %v1232 : tensor<64x1024x7x7xf32>
    %v1242 = stablehlo.add %v1241, %v1233 : tensor<64x1024x7x7xf32>
    %v1243 = stablehlo.rsqrt %v1242 : tensor<64x1024x7x7xf32>
    %v1244 = stablehlo.multiply %v1237, %v1243 : tensor<64x1024x7x7xf32>
    %v1245 = stablehlo.broadcast_in_dim %u13dg, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1246 = stablehlo.broadcast_in_dim %u13dbt, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1247 = stablehlo.multiply %v1244, %v1245 : tensor<64x1024x7x7xf32>
    %v1248 = stablehlo.add %v1247, %v1246 : tensor<64x1024x7x7xf32>
    %v1249 = stablehlo.reshape %v1248 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1250 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1251 = stablehlo.maximum %v1249, %v1250 : tensor<64x50176xf32>
    %v1252 = stablehlo.reshape %v1251 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1253 = stablehlo.convolution(%v1252, %u13pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<64x256x7x7xf32>
    %v1254 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1255 = stablehlo.add %v1253, %v1254 : tensor<64x256x7x7xf32>
    %v1256 = stablehlo.reshape %v1255 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1257 = stablehlo.reshape %v1256 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1258 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1259 = stablehlo.constant dense<3136.0> : tensor<64x256x7x7xf32>
    %v1260 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v1261 = stablehlo.reduce(%v1257 init: %v1258) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1262 = stablehlo.broadcast_in_dim %v1261, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1263 = stablehlo.divide %v1262, %v1259 : tensor<64x256x7x7xf32>
    %v1264 = stablehlo.subtract %v1257, %v1263 : tensor<64x256x7x7xf32>
    %v1265 = stablehlo.multiply %v1264, %v1264 : tensor<64x256x7x7xf32>
    %v1266 = stablehlo.reduce(%v1265 init: %v1258) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1267 = stablehlo.broadcast_in_dim %v1266, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1268 = stablehlo.divide %v1267, %v1259 : tensor<64x256x7x7xf32>
    %v1269 = stablehlo.add %v1268, %v1260 : tensor<64x256x7x7xf32>
    %v1270 = stablehlo.rsqrt %v1269 : tensor<64x256x7x7xf32>
    %v1271 = stablehlo.multiply %v1264, %v1270 : tensor<64x256x7x7xf32>
    %v1272 = stablehlo.broadcast_in_dim %u13pg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1273 = stablehlo.broadcast_in_dim %u13pbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1274 = stablehlo.multiply %v1271, %v1272 : tensor<64x256x7x7xf32>
    %v1275 = stablehlo.add %v1274, %v1273 : tensor<64x256x7x7xf32>
    %v1276 = stablehlo.reshape %v1275 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1277 = stablehlo.add %v1276, %v1197 : tensor<64x12544xf32>
    %v1278 = stablehlo.reshape %v1277 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1279 = stablehlo.convolution(%v1278, %u14qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<64x256x7x7xf32>, tensor<256x1x3x3xf32>) -> tensor<64x256x7x7xf32>
    %v1280 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1281 = stablehlo.add %v1279, %v1280 : tensor<64x256x7x7xf32>
    %v1282 = stablehlo.reshape %v1281 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1283 = stablehlo.reshape %v1282 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1284 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1285 = stablehlo.constant dense<3136.0> : tensor<64x256x7x7xf32>
    %v1286 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v1287 = stablehlo.reduce(%v1283 init: %v1284) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1288 = stablehlo.broadcast_in_dim %v1287, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1289 = stablehlo.divide %v1288, %v1285 : tensor<64x256x7x7xf32>
    %v1290 = stablehlo.subtract %v1283, %v1289 : tensor<64x256x7x7xf32>
    %v1291 = stablehlo.multiply %v1290, %v1290 : tensor<64x256x7x7xf32>
    %v1292 = stablehlo.reduce(%v1291 init: %v1284) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1293 = stablehlo.broadcast_in_dim %v1292, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1294 = stablehlo.divide %v1293, %v1285 : tensor<64x256x7x7xf32>
    %v1295 = stablehlo.add %v1294, %v1286 : tensor<64x256x7x7xf32>
    %v1296 = stablehlo.rsqrt %v1295 : tensor<64x256x7x7xf32>
    %v1297 = stablehlo.multiply %v1290, %v1296 : tensor<64x256x7x7xf32>
    %v1298 = stablehlo.broadcast_in_dim %u14qg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1299 = stablehlo.broadcast_in_dim %u14qbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1300 = stablehlo.multiply %v1297, %v1298 : tensor<64x256x7x7xf32>
    %v1301 = stablehlo.add %v1300, %v1299 : tensor<64x256x7x7xf32>
    %v1302 = stablehlo.reshape %v1301 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1303 = stablehlo.constant dense<0.0> : tensor<64x12544xf32>
    %v1304 = stablehlo.maximum %v1302, %v1303 : tensor<64x12544xf32>
    %v1305 = stablehlo.reshape %v1304 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1306 = stablehlo.convolution(%v1305, %u14eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<64x1024x7x7xf32>
    %v1307 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1308 = stablehlo.add %v1306, %v1307 : tensor<64x1024x7x7xf32>
    %v1309 = stablehlo.reshape %v1308 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1310 = stablehlo.reshape %v1309 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1311 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1312 = stablehlo.constant dense<3136.0> : tensor<64x1024x7x7xf32>
    %v1313 = stablehlo.constant dense<1.0e-5> : tensor<64x1024x7x7xf32>
    %v1314 = stablehlo.reduce(%v1310 init: %v1311) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1315 = stablehlo.broadcast_in_dim %v1314, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1316 = stablehlo.divide %v1315, %v1312 : tensor<64x1024x7x7xf32>
    %v1317 = stablehlo.subtract %v1310, %v1316 : tensor<64x1024x7x7xf32>
    %v1318 = stablehlo.multiply %v1317, %v1317 : tensor<64x1024x7x7xf32>
    %v1319 = stablehlo.reduce(%v1318 init: %v1311) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1320 = stablehlo.broadcast_in_dim %v1319, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1321 = stablehlo.divide %v1320, %v1312 : tensor<64x1024x7x7xf32>
    %v1322 = stablehlo.add %v1321, %v1313 : tensor<64x1024x7x7xf32>
    %v1323 = stablehlo.rsqrt %v1322 : tensor<64x1024x7x7xf32>
    %v1324 = stablehlo.multiply %v1317, %v1323 : tensor<64x1024x7x7xf32>
    %v1325 = stablehlo.broadcast_in_dim %u14eg, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1326 = stablehlo.broadcast_in_dim %u14ebt, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1327 = stablehlo.multiply %v1324, %v1325 : tensor<64x1024x7x7xf32>
    %v1328 = stablehlo.add %v1327, %v1326 : tensor<64x1024x7x7xf32>
    %v1329 = stablehlo.reshape %v1328 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1330 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1331 = stablehlo.maximum %v1329, %v1330 : tensor<64x50176xf32>
    %v1332 = stablehlo.reshape %v1331 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1333 = stablehlo.convolution(%v1332, %u14pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<64x256x7x7xf32>
    %v1334 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1335 = stablehlo.add %v1333, %v1334 : tensor<64x256x7x7xf32>
    %v1336 = stablehlo.reshape %v1335 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1337 = stablehlo.reshape %v1336 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1338 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1339 = stablehlo.constant dense<3136.0> : tensor<64x256x7x7xf32>
    %v1340 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v1341 = stablehlo.reduce(%v1337 init: %v1338) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1342 = stablehlo.broadcast_in_dim %v1341, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1343 = stablehlo.divide %v1342, %v1339 : tensor<64x256x7x7xf32>
    %v1344 = stablehlo.subtract %v1337, %v1343 : tensor<64x256x7x7xf32>
    %v1345 = stablehlo.multiply %v1344, %v1344 : tensor<64x256x7x7xf32>
    %v1346 = stablehlo.reduce(%v1345 init: %v1338) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1347 = stablehlo.broadcast_in_dim %v1346, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1348 = stablehlo.divide %v1347, %v1339 : tensor<64x256x7x7xf32>
    %v1349 = stablehlo.add %v1348, %v1340 : tensor<64x256x7x7xf32>
    %v1350 = stablehlo.rsqrt %v1349 : tensor<64x256x7x7xf32>
    %v1351 = stablehlo.multiply %v1344, %v1350 : tensor<64x256x7x7xf32>
    %v1352 = stablehlo.broadcast_in_dim %u14pg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1353 = stablehlo.broadcast_in_dim %u14pbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1354 = stablehlo.multiply %v1351, %v1352 : tensor<64x256x7x7xf32>
    %v1355 = stablehlo.add %v1354, %v1353 : tensor<64x256x7x7xf32>
    %v1356 = stablehlo.reshape %v1355 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1357 = stablehlo.add %v1356, %v1277 : tensor<64x12544xf32>
    %v1358 = stablehlo.reshape %v1357 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1359 = stablehlo.convolution(%v1358, %hW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x7x7xf32>, tensor<1280x256x1x1xf32>) -> tensor<64x1280x7x7xf32>
    %v1360 = stablehlo.broadcast_in_dim %zb1280, dims = [1] : (tensor<1280xf32>) -> tensor<64x1280x7x7xf32>
    %v1361 = stablehlo.add %v1359, %v1360 : tensor<64x1280x7x7xf32>
    %v1362 = stablehlo.reshape %v1361 : (tensor<64x1280x7x7xf32>) -> tensor<64x62720xf32>
    %v1363 = stablehlo.reshape %v1362 : (tensor<64x62720xf32>) -> tensor<64x1280x7x7xf32>
    %v1364 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1365 = stablehlo.constant dense<3136.0> : tensor<64x1280x7x7xf32>
    %v1366 = stablehlo.constant dense<1.0e-5> : tensor<64x1280x7x7xf32>
    %v1367 = stablehlo.reduce(%v1363 init: %v1364) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1368 = stablehlo.broadcast_in_dim %v1367, dims = [1] : (tensor<1280xf32>) -> tensor<64x1280x7x7xf32>
    %v1369 = stablehlo.divide %v1368, %v1365 : tensor<64x1280x7x7xf32>
    %v1370 = stablehlo.subtract %v1363, %v1369 : tensor<64x1280x7x7xf32>
    %v1371 = stablehlo.multiply %v1370, %v1370 : tensor<64x1280x7x7xf32>
    %v1372 = stablehlo.reduce(%v1371 init: %v1364) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1373 = stablehlo.broadcast_in_dim %v1372, dims = [1] : (tensor<1280xf32>) -> tensor<64x1280x7x7xf32>
    %v1374 = stablehlo.divide %v1373, %v1365 : tensor<64x1280x7x7xf32>
    %v1375 = stablehlo.add %v1374, %v1366 : tensor<64x1280x7x7xf32>
    %v1376 = stablehlo.rsqrt %v1375 : tensor<64x1280x7x7xf32>
    %v1377 = stablehlo.multiply %v1370, %v1376 : tensor<64x1280x7x7xf32>
    %v1378 = stablehlo.broadcast_in_dim %hg, dims = [1] : (tensor<1280xf32>) -> tensor<64x1280x7x7xf32>
    %v1379 = stablehlo.broadcast_in_dim %hbt, dims = [1] : (tensor<1280xf32>) -> tensor<64x1280x7x7xf32>
    %v1380 = stablehlo.multiply %v1377, %v1378 : tensor<64x1280x7x7xf32>
    %v1381 = stablehlo.add %v1380, %v1379 : tensor<64x1280x7x7xf32>
    %v1382 = stablehlo.reshape %v1381 : (tensor<64x1280x7x7xf32>) -> tensor<64x62720xf32>
    %v1383 = stablehlo.constant dense<0.0> : tensor<64x62720xf32>
    %v1384 = stablehlo.maximum %v1382, %v1383 : tensor<64x62720xf32>
    %v1385 = stablehlo.reshape %v1384 : (tensor<64x62720xf32>) -> tensor<64x1280x7x7xf32>
    %v1386 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1387 = stablehlo.reduce(%v1385 init: %v1386) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x1280x7x7xf32>, tensor<f32>) -> tensor<64x1280xf32>
    %v1388 = stablehlo.constant dense<49.0> : tensor<64x1280xf32>
    %v1389 = stablehlo.divide %v1387, %v1388 : tensor<64x1280xf32>
    %v1390 = stablehlo.dot_general %v1389, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x1280xf32>, tensor<1280x1000xf32>) -> tensor<64x1000xf32>
    %v1391 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<1000xf32>) -> tensor<64x1000xf32>
    %v1392 = stablehlo.add %v1390, %v1391 : tensor<64x1000xf32>
    return %v1392 : tensor<64x1000xf32>
  }
}
