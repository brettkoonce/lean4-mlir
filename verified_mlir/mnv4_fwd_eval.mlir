module @m {
  func.func @mnv4_fwd_eval(%x: tensor<32x150528xf32>, %sW: tensor<32x3x3x3xf32>, %sg: tensor<32xf32>, %sbt: tensor<32xf32>, %f0cW: tensor<128x32x3x3xf32>, %f0cg: tensor<128xf32>, %f0cbt: tensor<128xf32>, %f0pW: tensor<48x128x1x1xf32>, %f0pg: tensor<48xf32>, %f0pbt: tensor<48xf32>, %u1qW: tensor<48x1x3x3xf32>, %u1qg: tensor<48xf32>, %u1qbt: tensor<48xf32>, %u1eW: tensor<192x48x1x1xf32>, %u1eg: tensor<192xf32>, %u1ebt: tensor<192xf32>, %u1dW: tensor<192x1x5x5xf32>, %u1dg: tensor<192xf32>, %u1dbt: tensor<192xf32>, %u1pW: tensor<80x192x1x1xf32>, %u1pg: tensor<80xf32>, %u1pbt: tensor<80xf32>, %u2qW: tensor<80x1x3x3xf32>, %u2qg: tensor<80xf32>, %u2qbt: tensor<80xf32>, %u2eW: tensor<160x80x1x1xf32>, %u2eg: tensor<160xf32>, %u2ebt: tensor<160xf32>, %u2dW: tensor<160x1x3x3xf32>, %u2dg: tensor<160xf32>, %u2dbt: tensor<160xf32>, %u2pW: tensor<80x160x1x1xf32>, %u2pg: tensor<80xf32>, %u2pbt: tensor<80xf32>, %u3eW: tensor<480x80x1x1xf32>, %u3eg: tensor<480xf32>, %u3ebt: tensor<480xf32>, %u3dW: tensor<480x1x3x3xf32>, %u3dg: tensor<480xf32>, %u3dbt: tensor<480xf32>, %u3pW: tensor<160x480x1x1xf32>, %u3pg: tensor<160xf32>, %u3pbt: tensor<160xf32>, %u4qW: tensor<160x1x3x3xf32>, %u4qg: tensor<160xf32>, %u4qbt: tensor<160xf32>, %u4eW: tensor<640x160x1x1xf32>, %u4eg: tensor<640xf32>, %u4ebt: tensor<640xf32>, %u4dW: tensor<640x1x3x3xf32>, %u4dg: tensor<640xf32>, %u4dbt: tensor<640xf32>, %u4pW: tensor<160x640x1x1xf32>, %u4pg: tensor<160xf32>, %u4pbt: tensor<160xf32>, %u5qW: tensor<160x1x3x3xf32>, %u5qg: tensor<160xf32>, %u5qbt: tensor<160xf32>, %u5eW: tensor<640x160x1x1xf32>, %u5eg: tensor<640xf32>, %u5ebt: tensor<640xf32>, %u5dW: tensor<640x1x5x5xf32>, %u5dg: tensor<640xf32>, %u5dbt: tensor<640xf32>, %u5pW: tensor<160x640x1x1xf32>, %u5pg: tensor<160xf32>, %u5pbt: tensor<160xf32>, %u6qW: tensor<160x1x5x5xf32>, %u6qg: tensor<160xf32>, %u6qbt: tensor<160xf32>, %u6eW: tensor<640x160x1x1xf32>, %u6eg: tensor<640xf32>, %u6ebt: tensor<640xf32>, %u6pW: tensor<160x640x1x1xf32>, %u6pg: tensor<160xf32>, %u6pbt: tensor<160xf32>, %u7eW: tensor<640x160x1x1xf32>, %u7eg: tensor<640xf32>, %u7ebt: tensor<640xf32>, %u7dW: tensor<640x1x3x3xf32>, %u7dg: tensor<640xf32>, %u7dbt: tensor<640xf32>, %u7pW: tensor<160x640x1x1xf32>, %u7pg: tensor<160xf32>, %u7pbt: tensor<160xf32>, %u8qW: tensor<160x1x3x3xf32>, %u8qg: tensor<160xf32>, %u8qbt: tensor<160xf32>, %u8eW: tensor<640x160x1x1xf32>, %u8eg: tensor<640xf32>, %u8ebt: tensor<640xf32>, %u8pW: tensor<160x640x1x1xf32>, %u8pg: tensor<160xf32>, %u8pbt: tensor<160xf32>, %u9eW: tensor<640x160x1x1xf32>, %u9eg: tensor<640xf32>, %u9ebt: tensor<640xf32>, %u9pW: tensor<160x640x1x1xf32>, %u9pg: tensor<160xf32>, %u9pbt: tensor<160xf32>, %u10qW: tensor<160x1x3x3xf32>, %u10qg: tensor<160xf32>, %u10qbt: tensor<160xf32>, %u10eW: tensor<640x160x1x1xf32>, %u10eg: tensor<640xf32>, %u10ebt: tensor<640xf32>, %u10dW: tensor<640x1x3x3xf32>, %u10dg: tensor<640xf32>, %u10dbt: tensor<640xf32>, %u10pW: tensor<160x640x1x1xf32>, %u10pg: tensor<160xf32>, %u10pbt: tensor<160xf32>, %u11qW: tensor<160x1x5x5xf32>, %u11qg: tensor<160xf32>, %u11qbt: tensor<160xf32>, %u11eW: tensor<960x160x1x1xf32>, %u11eg: tensor<960xf32>, %u11ebt: tensor<960xf32>, %u11dW: tensor<960x1x5x5xf32>, %u11dg: tensor<960xf32>, %u11dbt: tensor<960xf32>, %u11pW: tensor<256x960x1x1xf32>, %u11pg: tensor<256xf32>, %u11pbt: tensor<256xf32>, %u12qW: tensor<256x1x5x5xf32>, %u12qg: tensor<256xf32>, %u12qbt: tensor<256xf32>, %u12eW: tensor<1024x256x1x1xf32>, %u12eg: tensor<1024xf32>, %u12ebt: tensor<1024xf32>, %u12dW: tensor<1024x1x5x5xf32>, %u12dg: tensor<1024xf32>, %u12dbt: tensor<1024xf32>, %u12pW: tensor<256x1024x1x1xf32>, %u12pg: tensor<256xf32>, %u12pbt: tensor<256xf32>, %u13eW: tensor<1024x256x1x1xf32>, %u13eg: tensor<1024xf32>, %u13ebt: tensor<1024xf32>, %u13dW: tensor<1024x1x3x3xf32>, %u13dg: tensor<1024xf32>, %u13dbt: tensor<1024xf32>, %u13pW: tensor<256x1024x1x1xf32>, %u13pg: tensor<256xf32>, %u13pbt: tensor<256xf32>, %u14qW: tensor<256x1x3x3xf32>, %u14qg: tensor<256xf32>, %u14qbt: tensor<256xf32>, %u14eW: tensor<1024x256x1x1xf32>, %u14eg: tensor<1024xf32>, %u14ebt: tensor<1024xf32>, %u14pW: tensor<256x1024x1x1xf32>, %u14pg: tensor<256xf32>, %u14pbt: tensor<256xf32>, %hW: tensor<1280x256x1x1xf32>, %hg: tensor<1280xf32>, %hbt: tensor<1280xf32>, %Wd: tensor<1280x10xf32>, %bd: tensor<10xf32>, %stnmu: tensor<32xf32>, %stnvar: tensor<32xf32>, %f0cnmu: tensor<128xf32>, %f0cnvar: tensor<128xf32>, %f0pnmu: tensor<48xf32>, %f0pnvar: tensor<48xf32>, %u1qnmu: tensor<48xf32>, %u1qnvar: tensor<48xf32>, %u1enmu: tensor<192xf32>, %u1envar: tensor<192xf32>, %u1dnmu: tensor<192xf32>, %u1dnvar: tensor<192xf32>, %u1pnmu: tensor<80xf32>, %u1pnvar: tensor<80xf32>, %u2qnmu: tensor<80xf32>, %u2qnvar: tensor<80xf32>, %u2enmu: tensor<160xf32>, %u2envar: tensor<160xf32>, %u2dnmu: tensor<160xf32>, %u2dnvar: tensor<160xf32>, %u2pnmu: tensor<80xf32>, %u2pnvar: tensor<80xf32>, %u3enmu: tensor<480xf32>, %u3envar: tensor<480xf32>, %u3dnmu: tensor<480xf32>, %u3dnvar: tensor<480xf32>, %u3pnmu: tensor<160xf32>, %u3pnvar: tensor<160xf32>, %u4qnmu: tensor<160xf32>, %u4qnvar: tensor<160xf32>, %u4enmu: tensor<640xf32>, %u4envar: tensor<640xf32>, %u4dnmu: tensor<640xf32>, %u4dnvar: tensor<640xf32>, %u4pnmu: tensor<160xf32>, %u4pnvar: tensor<160xf32>, %u5qnmu: tensor<160xf32>, %u5qnvar: tensor<160xf32>, %u5enmu: tensor<640xf32>, %u5envar: tensor<640xf32>, %u5dnmu: tensor<640xf32>, %u5dnvar: tensor<640xf32>, %u5pnmu: tensor<160xf32>, %u5pnvar: tensor<160xf32>, %u6qnmu: tensor<160xf32>, %u6qnvar: tensor<160xf32>, %u6enmu: tensor<640xf32>, %u6envar: tensor<640xf32>, %u6pnmu: tensor<160xf32>, %u6pnvar: tensor<160xf32>, %u7enmu: tensor<640xf32>, %u7envar: tensor<640xf32>, %u7dnmu: tensor<640xf32>, %u7dnvar: tensor<640xf32>, %u7pnmu: tensor<160xf32>, %u7pnvar: tensor<160xf32>, %u8qnmu: tensor<160xf32>, %u8qnvar: tensor<160xf32>, %u8enmu: tensor<640xf32>, %u8envar: tensor<640xf32>, %u8pnmu: tensor<160xf32>, %u8pnvar: tensor<160xf32>, %u9enmu: tensor<640xf32>, %u9envar: tensor<640xf32>, %u9pnmu: tensor<160xf32>, %u9pnvar: tensor<160xf32>, %u10qnmu: tensor<160xf32>, %u10qnvar: tensor<160xf32>, %u10enmu: tensor<640xf32>, %u10envar: tensor<640xf32>, %u10dnmu: tensor<640xf32>, %u10dnvar: tensor<640xf32>, %u10pnmu: tensor<160xf32>, %u10pnvar: tensor<160xf32>, %u11qnmu: tensor<160xf32>, %u11qnvar: tensor<160xf32>, %u11enmu: tensor<960xf32>, %u11envar: tensor<960xf32>, %u11dnmu: tensor<960xf32>, %u11dnvar: tensor<960xf32>, %u11pnmu: tensor<256xf32>, %u11pnvar: tensor<256xf32>, %u12qnmu: tensor<256xf32>, %u12qnvar: tensor<256xf32>, %u12enmu: tensor<1024xf32>, %u12envar: tensor<1024xf32>, %u12dnmu: tensor<1024xf32>, %u12dnvar: tensor<1024xf32>, %u12pnmu: tensor<256xf32>, %u12pnvar: tensor<256xf32>, %u13enmu: tensor<1024xf32>, %u13envar: tensor<1024xf32>, %u13dnmu: tensor<1024xf32>, %u13dnvar: tensor<1024xf32>, %u13pnmu: tensor<256xf32>, %u13pnvar: tensor<256xf32>, %u14qnmu: tensor<256xf32>, %u14qnvar: tensor<256xf32>, %u14enmu: tensor<1024xf32>, %u14envar: tensor<1024xf32>, %u14pnmu: tensor<256xf32>, %u14pnvar: tensor<256xf32>, %hnmu: tensor<1280xf32>, %hnvar: tensor<1280xf32>) -> tensor<32x10xf32> {
    // ── MobileNetV4-Conv-S eval forward (running-stats BN): every line is pretty(AST node) ──
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
    %v0 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v1 = stablehlo.convolution(%v0, %sW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3x224x224xf32>, tensor<32x3x3x3xf32>) -> tensor<32x32x112x112xf32>
    %v2 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<32x32x112x112xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6 = stablehlo.broadcast_in_dim %stnmu, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v7 = stablehlo.subtract %v5, %v6 : tensor<32x32x112x112xf32>
    %v8 = stablehlo.broadcast_in_dim %stnvar, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v9 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v10 = stablehlo.add %v8, %v9 : tensor<32x32x112x112xf32>
    %v11 = stablehlo.rsqrt %v10 : tensor<32x32x112x112xf32>
    %v12 = stablehlo.multiply %v7, %v11 : tensor<32x32x112x112xf32>
    %v13 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v14 = stablehlo.broadcast_in_dim %sbt, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v15 = stablehlo.multiply %v12, %v13 : tensor<32x32x112x112xf32>
    %v16 = stablehlo.add %v15, %v14 : tensor<32x32x112x112xf32>
    %v17 = stablehlo.reshape %v16 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v18 = stablehlo.constant dense<0.0> : tensor<32x401408xf32>
    %v19 = stablehlo.maximum %v17, %v18 : tensor<32x401408xf32>
    %v20 = stablehlo.reshape %v19 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v21 = stablehlo.convolution(%v20, %f0cW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<128x32x3x3xf32>) -> tensor<32x128x56x56xf32>
    %v22 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v23 = stablehlo.add %v21, %v22 : tensor<32x128x56x56xf32>
    %v24 = stablehlo.reshape %v23 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v25 = stablehlo.reshape %v24 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v26 = stablehlo.broadcast_in_dim %f0cnmu, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v27 = stablehlo.subtract %v25, %v26 : tensor<32x128x56x56xf32>
    %v28 = stablehlo.broadcast_in_dim %f0cnvar, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v29 = stablehlo.constant dense<1.0e-5> : tensor<32x128x56x56xf32>
    %v30 = stablehlo.add %v28, %v29 : tensor<32x128x56x56xf32>
    %v31 = stablehlo.rsqrt %v30 : tensor<32x128x56x56xf32>
    %v32 = stablehlo.multiply %v27, %v31 : tensor<32x128x56x56xf32>
    %v33 = stablehlo.broadcast_in_dim %f0cg, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v34 = stablehlo.broadcast_in_dim %f0cbt, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v35 = stablehlo.multiply %v32, %v33 : tensor<32x128x56x56xf32>
    %v36 = stablehlo.add %v35, %v34 : tensor<32x128x56x56xf32>
    %v37 = stablehlo.reshape %v36 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v38 = stablehlo.logistic %v37 : tensor<32x401408xf32>
    %v39 = stablehlo.multiply %v37, %v38 : tensor<32x401408xf32>
    %v40 = stablehlo.reshape %v39 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v41 = stablehlo.convolution(%v40, %f0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x56x56xf32>, tensor<48x128x1x1xf32>) -> tensor<32x48x56x56xf32>
    %v42 = stablehlo.broadcast_in_dim %zb48, dims = [1] : (tensor<48xf32>) -> tensor<32x48x56x56xf32>
    %v43 = stablehlo.add %v41, %v42 : tensor<32x48x56x56xf32>
    %v44 = stablehlo.reshape %v43 : (tensor<32x48x56x56xf32>) -> tensor<32x150528xf32>
    %v45 = stablehlo.reshape %v44 : (tensor<32x150528xf32>) -> tensor<32x48x56x56xf32>
    %v46 = stablehlo.broadcast_in_dim %f0pnmu, dims = [1] : (tensor<48xf32>) -> tensor<32x48x56x56xf32>
    %v47 = stablehlo.subtract %v45, %v46 : tensor<32x48x56x56xf32>
    %v48 = stablehlo.broadcast_in_dim %f0pnvar, dims = [1] : (tensor<48xf32>) -> tensor<32x48x56x56xf32>
    %v49 = stablehlo.constant dense<1.0e-5> : tensor<32x48x56x56xf32>
    %v50 = stablehlo.add %v48, %v49 : tensor<32x48x56x56xf32>
    %v51 = stablehlo.rsqrt %v50 : tensor<32x48x56x56xf32>
    %v52 = stablehlo.multiply %v47, %v51 : tensor<32x48x56x56xf32>
    %v53 = stablehlo.broadcast_in_dim %f0pg, dims = [1] : (tensor<48xf32>) -> tensor<32x48x56x56xf32>
    %v54 = stablehlo.broadcast_in_dim %f0pbt, dims = [1] : (tensor<48xf32>) -> tensor<32x48x56x56xf32>
    %v55 = stablehlo.multiply %v52, %v53 : tensor<32x48x56x56xf32>
    %v56 = stablehlo.add %v55, %v54 : tensor<32x48x56x56xf32>
    %v57 = stablehlo.reshape %v56 : (tensor<32x48x56x56xf32>) -> tensor<32x150528xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<32x150528xf32>) -> tensor<32x48x56x56xf32>
    %v59 = stablehlo.convolution(%v58, %u1qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 48 : i64} : (tensor<32x48x56x56xf32>, tensor<48x1x3x3xf32>) -> tensor<32x48x28x28xf32>
    %v60 = stablehlo.broadcast_in_dim %zb48, dims = [1] : (tensor<48xf32>) -> tensor<32x48x28x28xf32>
    %v61 = stablehlo.add %v59, %v60 : tensor<32x48x28x28xf32>
    %v62 = stablehlo.reshape %v61 : (tensor<32x48x28x28xf32>) -> tensor<32x37632xf32>
    %v63 = stablehlo.reshape %v62 : (tensor<32x37632xf32>) -> tensor<32x48x28x28xf32>
    %v64 = stablehlo.broadcast_in_dim %u1qnmu, dims = [1] : (tensor<48xf32>) -> tensor<32x48x28x28xf32>
    %v65 = stablehlo.subtract %v63, %v64 : tensor<32x48x28x28xf32>
    %v66 = stablehlo.broadcast_in_dim %u1qnvar, dims = [1] : (tensor<48xf32>) -> tensor<32x48x28x28xf32>
    %v67 = stablehlo.constant dense<1.0e-5> : tensor<32x48x28x28xf32>
    %v68 = stablehlo.add %v66, %v67 : tensor<32x48x28x28xf32>
    %v69 = stablehlo.rsqrt %v68 : tensor<32x48x28x28xf32>
    %v70 = stablehlo.multiply %v65, %v69 : tensor<32x48x28x28xf32>
    %v71 = stablehlo.broadcast_in_dim %u1qg, dims = [1] : (tensor<48xf32>) -> tensor<32x48x28x28xf32>
    %v72 = stablehlo.broadcast_in_dim %u1qbt, dims = [1] : (tensor<48xf32>) -> tensor<32x48x28x28xf32>
    %v73 = stablehlo.multiply %v70, %v71 : tensor<32x48x28x28xf32>
    %v74 = stablehlo.add %v73, %v72 : tensor<32x48x28x28xf32>
    %v75 = stablehlo.reshape %v74 : (tensor<32x48x28x28xf32>) -> tensor<32x37632xf32>
    %v76 = stablehlo.constant dense<0.0> : tensor<32x37632xf32>
    %v77 = stablehlo.maximum %v75, %v76 : tensor<32x37632xf32>
    %v78 = stablehlo.reshape %v77 : (tensor<32x37632xf32>) -> tensor<32x48x28x28xf32>
    %v79 = stablehlo.convolution(%v78, %u1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x48x28x28xf32>, tensor<192x48x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v80 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v81 = stablehlo.add %v79, %v80 : tensor<32x192x28x28xf32>
    %v82 = stablehlo.reshape %v81 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v83 = stablehlo.reshape %v82 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v84 = stablehlo.broadcast_in_dim %u1enmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v85 = stablehlo.subtract %v83, %v84 : tensor<32x192x28x28xf32>
    %v86 = stablehlo.broadcast_in_dim %u1envar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v87 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v88 = stablehlo.add %v86, %v87 : tensor<32x192x28x28xf32>
    %v89 = stablehlo.rsqrt %v88 : tensor<32x192x28x28xf32>
    %v90 = stablehlo.multiply %v85, %v89 : tensor<32x192x28x28xf32>
    %v91 = stablehlo.broadcast_in_dim %u1eg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v92 = stablehlo.broadcast_in_dim %u1ebt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v93 = stablehlo.multiply %v90, %v91 : tensor<32x192x28x28xf32>
    %v94 = stablehlo.add %v93, %v92 : tensor<32x192x28x28xf32>
    %v95 = stablehlo.reshape %v94 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v96 = stablehlo.constant dense<0.0> : tensor<32x150528xf32>
    %v97 = stablehlo.maximum %v95, %v96 : tensor<32x150528xf32>
    %v98 = stablehlo.reshape %v97 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v99 = stablehlo.convolution(%v98, %u1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x5x5xf32>) -> tensor<32x192x28x28xf32>
    %v100 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v101 = stablehlo.add %v99, %v100 : tensor<32x192x28x28xf32>
    %v102 = stablehlo.reshape %v101 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v103 = stablehlo.reshape %v102 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v104 = stablehlo.broadcast_in_dim %u1dnmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v105 = stablehlo.subtract %v103, %v104 : tensor<32x192x28x28xf32>
    %v106 = stablehlo.broadcast_in_dim %u1dnvar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v107 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v108 = stablehlo.add %v106, %v107 : tensor<32x192x28x28xf32>
    %v109 = stablehlo.rsqrt %v108 : tensor<32x192x28x28xf32>
    %v110 = stablehlo.multiply %v105, %v109 : tensor<32x192x28x28xf32>
    %v111 = stablehlo.broadcast_in_dim %u1dg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v112 = stablehlo.broadcast_in_dim %u1dbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v113 = stablehlo.multiply %v110, %v111 : tensor<32x192x28x28xf32>
    %v114 = stablehlo.add %v113, %v112 : tensor<32x192x28x28xf32>
    %v115 = stablehlo.reshape %v114 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v116 = stablehlo.constant dense<0.0> : tensor<32x150528xf32>
    %v117 = stablehlo.maximum %v115, %v116 : tensor<32x150528xf32>
    %v118 = stablehlo.reshape %v117 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v119 = stablehlo.convolution(%v118, %u1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<80x192x1x1xf32>) -> tensor<32x80x28x28xf32>
    %v120 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v121 = stablehlo.add %v119, %v120 : tensor<32x80x28x28xf32>
    %v122 = stablehlo.reshape %v121 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v123 = stablehlo.reshape %v122 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v124 = stablehlo.broadcast_in_dim %u1pnmu, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v125 = stablehlo.subtract %v123, %v124 : tensor<32x80x28x28xf32>
    %v126 = stablehlo.broadcast_in_dim %u1pnvar, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v127 = stablehlo.constant dense<1.0e-5> : tensor<32x80x28x28xf32>
    %v128 = stablehlo.add %v126, %v127 : tensor<32x80x28x28xf32>
    %v129 = stablehlo.rsqrt %v128 : tensor<32x80x28x28xf32>
    %v130 = stablehlo.multiply %v125, %v129 : tensor<32x80x28x28xf32>
    %v131 = stablehlo.broadcast_in_dim %u1pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v132 = stablehlo.broadcast_in_dim %u1pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v133 = stablehlo.multiply %v130, %v131 : tensor<32x80x28x28xf32>
    %v134 = stablehlo.add %v133, %v132 : tensor<32x80x28x28xf32>
    %v135 = stablehlo.reshape %v134 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v136 = stablehlo.reshape %v135 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v137 = stablehlo.convolution(%v136, %u2qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 80 : i64} : (tensor<32x80x28x28xf32>, tensor<80x1x3x3xf32>) -> tensor<32x80x28x28xf32>
    %v138 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v139 = stablehlo.add %v137, %v138 : tensor<32x80x28x28xf32>
    %v140 = stablehlo.reshape %v139 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v141 = stablehlo.reshape %v140 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v142 = stablehlo.broadcast_in_dim %u2qnmu, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v143 = stablehlo.subtract %v141, %v142 : tensor<32x80x28x28xf32>
    %v144 = stablehlo.broadcast_in_dim %u2qnvar, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v145 = stablehlo.constant dense<1.0e-5> : tensor<32x80x28x28xf32>
    %v146 = stablehlo.add %v144, %v145 : tensor<32x80x28x28xf32>
    %v147 = stablehlo.rsqrt %v146 : tensor<32x80x28x28xf32>
    %v148 = stablehlo.multiply %v143, %v147 : tensor<32x80x28x28xf32>
    %v149 = stablehlo.broadcast_in_dim %u2qg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v150 = stablehlo.broadcast_in_dim %u2qbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v151 = stablehlo.multiply %v148, %v149 : tensor<32x80x28x28xf32>
    %v152 = stablehlo.add %v151, %v150 : tensor<32x80x28x28xf32>
    %v153 = stablehlo.reshape %v152 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v154 = stablehlo.constant dense<0.0> : tensor<32x62720xf32>
    %v155 = stablehlo.maximum %v153, %v154 : tensor<32x62720xf32>
    %v156 = stablehlo.reshape %v155 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v157 = stablehlo.convolution(%v156, %u2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x28x28xf32>, tensor<160x80x1x1xf32>) -> tensor<32x160x28x28xf32>
    %v158 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v159 = stablehlo.add %v157, %v158 : tensor<32x160x28x28xf32>
    %v160 = stablehlo.reshape %v159 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v161 = stablehlo.reshape %v160 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v162 = stablehlo.broadcast_in_dim %u2enmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v163 = stablehlo.subtract %v161, %v162 : tensor<32x160x28x28xf32>
    %v164 = stablehlo.broadcast_in_dim %u2envar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v165 = stablehlo.constant dense<1.0e-5> : tensor<32x160x28x28xf32>
    %v166 = stablehlo.add %v164, %v165 : tensor<32x160x28x28xf32>
    %v167 = stablehlo.rsqrt %v166 : tensor<32x160x28x28xf32>
    %v168 = stablehlo.multiply %v163, %v167 : tensor<32x160x28x28xf32>
    %v169 = stablehlo.broadcast_in_dim %u2eg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v170 = stablehlo.broadcast_in_dim %u2ebt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v171 = stablehlo.multiply %v168, %v169 : tensor<32x160x28x28xf32>
    %v172 = stablehlo.add %v171, %v170 : tensor<32x160x28x28xf32>
    %v173 = stablehlo.reshape %v172 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v174 = stablehlo.constant dense<0.0> : tensor<32x125440xf32>
    %v175 = stablehlo.maximum %v173, %v174 : tensor<32x125440xf32>
    %v176 = stablehlo.reshape %v175 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v177 = stablehlo.convolution(%v176, %u2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x28x28xf32>, tensor<160x1x3x3xf32>) -> tensor<32x160x28x28xf32>
    %v178 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v179 = stablehlo.add %v177, %v178 : tensor<32x160x28x28xf32>
    %v180 = stablehlo.reshape %v179 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v181 = stablehlo.reshape %v180 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v182 = stablehlo.broadcast_in_dim %u2dnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v183 = stablehlo.subtract %v181, %v182 : tensor<32x160x28x28xf32>
    %v184 = stablehlo.broadcast_in_dim %u2dnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v185 = stablehlo.constant dense<1.0e-5> : tensor<32x160x28x28xf32>
    %v186 = stablehlo.add %v184, %v185 : tensor<32x160x28x28xf32>
    %v187 = stablehlo.rsqrt %v186 : tensor<32x160x28x28xf32>
    %v188 = stablehlo.multiply %v183, %v187 : tensor<32x160x28x28xf32>
    %v189 = stablehlo.broadcast_in_dim %u2dg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v190 = stablehlo.broadcast_in_dim %u2dbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v191 = stablehlo.multiply %v188, %v189 : tensor<32x160x28x28xf32>
    %v192 = stablehlo.add %v191, %v190 : tensor<32x160x28x28xf32>
    %v193 = stablehlo.reshape %v192 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v194 = stablehlo.constant dense<0.0> : tensor<32x125440xf32>
    %v195 = stablehlo.maximum %v193, %v194 : tensor<32x125440xf32>
    %v196 = stablehlo.reshape %v195 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v197 = stablehlo.convolution(%v196, %u2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x28x28xf32>, tensor<80x160x1x1xf32>) -> tensor<32x80x28x28xf32>
    %v198 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v199 = stablehlo.add %v197, %v198 : tensor<32x80x28x28xf32>
    %v200 = stablehlo.reshape %v199 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v201 = stablehlo.reshape %v200 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v202 = stablehlo.broadcast_in_dim %u2pnmu, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v203 = stablehlo.subtract %v201, %v202 : tensor<32x80x28x28xf32>
    %v204 = stablehlo.broadcast_in_dim %u2pnvar, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v205 = stablehlo.constant dense<1.0e-5> : tensor<32x80x28x28xf32>
    %v206 = stablehlo.add %v204, %v205 : tensor<32x80x28x28xf32>
    %v207 = stablehlo.rsqrt %v206 : tensor<32x80x28x28xf32>
    %v208 = stablehlo.multiply %v203, %v207 : tensor<32x80x28x28xf32>
    %v209 = stablehlo.broadcast_in_dim %u2pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v210 = stablehlo.broadcast_in_dim %u2pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v211 = stablehlo.multiply %v208, %v209 : tensor<32x80x28x28xf32>
    %v212 = stablehlo.add %v211, %v210 : tensor<32x80x28x28xf32>
    %v213 = stablehlo.reshape %v212 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v214 = stablehlo.add %v213, %v135 : tensor<32x62720xf32>
    %v215 = stablehlo.reshape %v214 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v216 = stablehlo.convolution(%v215, %u3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x28x28xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x28x28xf32>
    %v217 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x28x28xf32>
    %v218 = stablehlo.add %v216, %v217 : tensor<32x480x28x28xf32>
    %v219 = stablehlo.reshape %v218 : (tensor<32x480x28x28xf32>) -> tensor<32x376320xf32>
    %v220 = stablehlo.reshape %v219 : (tensor<32x376320xf32>) -> tensor<32x480x28x28xf32>
    %v221 = stablehlo.broadcast_in_dim %u3enmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x28x28xf32>
    %v222 = stablehlo.subtract %v220, %v221 : tensor<32x480x28x28xf32>
    %v223 = stablehlo.broadcast_in_dim %u3envar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x28x28xf32>
    %v224 = stablehlo.constant dense<1.0e-5> : tensor<32x480x28x28xf32>
    %v225 = stablehlo.add %v223, %v224 : tensor<32x480x28x28xf32>
    %v226 = stablehlo.rsqrt %v225 : tensor<32x480x28x28xf32>
    %v227 = stablehlo.multiply %v222, %v226 : tensor<32x480x28x28xf32>
    %v228 = stablehlo.broadcast_in_dim %u3eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x28x28xf32>
    %v229 = stablehlo.broadcast_in_dim %u3ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x28x28xf32>
    %v230 = stablehlo.multiply %v227, %v228 : tensor<32x480x28x28xf32>
    %v231 = stablehlo.add %v230, %v229 : tensor<32x480x28x28xf32>
    %v232 = stablehlo.reshape %v231 : (tensor<32x480x28x28xf32>) -> tensor<32x376320xf32>
    %v233 = stablehlo.constant dense<0.0> : tensor<32x376320xf32>
    %v234 = stablehlo.maximum %v232, %v233 : tensor<32x376320xf32>
    %v235 = stablehlo.reshape %v234 : (tensor<32x376320xf32>) -> tensor<32x480x28x28xf32>
    %v236 = stablehlo.convolution(%v235, %u3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x28x28xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %v237 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v238 = stablehlo.add %v236, %v237 : tensor<32x480x14x14xf32>
    %v239 = stablehlo.reshape %v238 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v240 = stablehlo.reshape %v239 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v241 = stablehlo.broadcast_in_dim %u3dnmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v242 = stablehlo.subtract %v240, %v241 : tensor<32x480x14x14xf32>
    %v243 = stablehlo.broadcast_in_dim %u3dnvar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v244 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v245 = stablehlo.add %v243, %v244 : tensor<32x480x14x14xf32>
    %v246 = stablehlo.rsqrt %v245 : tensor<32x480x14x14xf32>
    %v247 = stablehlo.multiply %v242, %v246 : tensor<32x480x14x14xf32>
    %v248 = stablehlo.broadcast_in_dim %u3dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v249 = stablehlo.broadcast_in_dim %u3dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v250 = stablehlo.multiply %v247, %v248 : tensor<32x480x14x14xf32>
    %v251 = stablehlo.add %v250, %v249 : tensor<32x480x14x14xf32>
    %v252 = stablehlo.reshape %v251 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v253 = stablehlo.constant dense<0.0> : tensor<32x94080xf32>
    %v254 = stablehlo.maximum %v252, %v253 : tensor<32x94080xf32>
    %v255 = stablehlo.reshape %v254 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v256 = stablehlo.convolution(%v255, %u3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<160x480x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v257 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v258 = stablehlo.add %v256, %v257 : tensor<32x160x14x14xf32>
    %v259 = stablehlo.reshape %v258 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v260 = stablehlo.reshape %v259 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v261 = stablehlo.broadcast_in_dim %u3pnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v262 = stablehlo.subtract %v260, %v261 : tensor<32x160x14x14xf32>
    %v263 = stablehlo.broadcast_in_dim %u3pnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v264 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v265 = stablehlo.add %v263, %v264 : tensor<32x160x14x14xf32>
    %v266 = stablehlo.rsqrt %v265 : tensor<32x160x14x14xf32>
    %v267 = stablehlo.multiply %v262, %v266 : tensor<32x160x14x14xf32>
    %v268 = stablehlo.broadcast_in_dim %u3pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v269 = stablehlo.broadcast_in_dim %u3pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v270 = stablehlo.multiply %v267, %v268 : tensor<32x160x14x14xf32>
    %v271 = stablehlo.add %v270, %v269 : tensor<32x160x14x14xf32>
    %v272 = stablehlo.reshape %v271 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v273 = stablehlo.reshape %v272 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v274 = stablehlo.convolution(%v273, %u4qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<32x160x14x14xf32>
    %v275 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v276 = stablehlo.add %v274, %v275 : tensor<32x160x14x14xf32>
    %v277 = stablehlo.reshape %v276 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v278 = stablehlo.reshape %v277 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v279 = stablehlo.broadcast_in_dim %u4qnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v280 = stablehlo.subtract %v278, %v279 : tensor<32x160x14x14xf32>
    %v281 = stablehlo.broadcast_in_dim %u4qnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v282 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v283 = stablehlo.add %v281, %v282 : tensor<32x160x14x14xf32>
    %v284 = stablehlo.rsqrt %v283 : tensor<32x160x14x14xf32>
    %v285 = stablehlo.multiply %v280, %v284 : tensor<32x160x14x14xf32>
    %v286 = stablehlo.broadcast_in_dim %u4qg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v287 = stablehlo.broadcast_in_dim %u4qbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v288 = stablehlo.multiply %v285, %v286 : tensor<32x160x14x14xf32>
    %v289 = stablehlo.add %v288, %v287 : tensor<32x160x14x14xf32>
    %v290 = stablehlo.reshape %v289 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v291 = stablehlo.constant dense<0.0> : tensor<32x31360xf32>
    %v292 = stablehlo.maximum %v290, %v291 : tensor<32x31360xf32>
    %v293 = stablehlo.reshape %v292 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v294 = stablehlo.convolution(%v293, %u4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<32x640x14x14xf32>
    %v295 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v296 = stablehlo.add %v294, %v295 : tensor<32x640x14x14xf32>
    %v297 = stablehlo.reshape %v296 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v298 = stablehlo.reshape %v297 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v299 = stablehlo.broadcast_in_dim %u4enmu, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v300 = stablehlo.subtract %v298, %v299 : tensor<32x640x14x14xf32>
    %v301 = stablehlo.broadcast_in_dim %u4envar, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v302 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v303 = stablehlo.add %v301, %v302 : tensor<32x640x14x14xf32>
    %v304 = stablehlo.rsqrt %v303 : tensor<32x640x14x14xf32>
    %v305 = stablehlo.multiply %v300, %v304 : tensor<32x640x14x14xf32>
    %v306 = stablehlo.broadcast_in_dim %u4eg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v307 = stablehlo.broadcast_in_dim %u4ebt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v308 = stablehlo.multiply %v305, %v306 : tensor<32x640x14x14xf32>
    %v309 = stablehlo.add %v308, %v307 : tensor<32x640x14x14xf32>
    %v310 = stablehlo.reshape %v309 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v311 = stablehlo.constant dense<0.0> : tensor<32x125440xf32>
    %v312 = stablehlo.maximum %v310, %v311 : tensor<32x125440xf32>
    %v313 = stablehlo.reshape %v312 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v314 = stablehlo.convolution(%v313, %u4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 640 : i64} : (tensor<32x640x14x14xf32>, tensor<640x1x3x3xf32>) -> tensor<32x640x14x14xf32>
    %v315 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v316 = stablehlo.add %v314, %v315 : tensor<32x640x14x14xf32>
    %v317 = stablehlo.reshape %v316 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v318 = stablehlo.reshape %v317 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v319 = stablehlo.broadcast_in_dim %u4dnmu, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v320 = stablehlo.subtract %v318, %v319 : tensor<32x640x14x14xf32>
    %v321 = stablehlo.broadcast_in_dim %u4dnvar, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v322 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v323 = stablehlo.add %v321, %v322 : tensor<32x640x14x14xf32>
    %v324 = stablehlo.rsqrt %v323 : tensor<32x640x14x14xf32>
    %v325 = stablehlo.multiply %v320, %v324 : tensor<32x640x14x14xf32>
    %v326 = stablehlo.broadcast_in_dim %u4dg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v327 = stablehlo.broadcast_in_dim %u4dbt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v328 = stablehlo.multiply %v325, %v326 : tensor<32x640x14x14xf32>
    %v329 = stablehlo.add %v328, %v327 : tensor<32x640x14x14xf32>
    %v330 = stablehlo.reshape %v329 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v331 = stablehlo.constant dense<0.0> : tensor<32x125440xf32>
    %v332 = stablehlo.maximum %v330, %v331 : tensor<32x125440xf32>
    %v333 = stablehlo.reshape %v332 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v334 = stablehlo.convolution(%v333, %u4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v335 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v336 = stablehlo.add %v334, %v335 : tensor<32x160x14x14xf32>
    %v337 = stablehlo.reshape %v336 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v338 = stablehlo.reshape %v337 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v339 = stablehlo.broadcast_in_dim %u4pnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v340 = stablehlo.subtract %v338, %v339 : tensor<32x160x14x14xf32>
    %v341 = stablehlo.broadcast_in_dim %u4pnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v342 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v343 = stablehlo.add %v341, %v342 : tensor<32x160x14x14xf32>
    %v344 = stablehlo.rsqrt %v343 : tensor<32x160x14x14xf32>
    %v345 = stablehlo.multiply %v340, %v344 : tensor<32x160x14x14xf32>
    %v346 = stablehlo.broadcast_in_dim %u4pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v347 = stablehlo.broadcast_in_dim %u4pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v348 = stablehlo.multiply %v345, %v346 : tensor<32x160x14x14xf32>
    %v349 = stablehlo.add %v348, %v347 : tensor<32x160x14x14xf32>
    %v350 = stablehlo.reshape %v349 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v351 = stablehlo.add %v350, %v272 : tensor<32x31360xf32>
    %v352 = stablehlo.reshape %v351 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v353 = stablehlo.convolution(%v352, %u5qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<32x160x14x14xf32>
    %v354 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v355 = stablehlo.add %v353, %v354 : tensor<32x160x14x14xf32>
    %v356 = stablehlo.reshape %v355 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v357 = stablehlo.reshape %v356 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v358 = stablehlo.broadcast_in_dim %u5qnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v359 = stablehlo.subtract %v357, %v358 : tensor<32x160x14x14xf32>
    %v360 = stablehlo.broadcast_in_dim %u5qnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v361 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v362 = stablehlo.add %v360, %v361 : tensor<32x160x14x14xf32>
    %v363 = stablehlo.rsqrt %v362 : tensor<32x160x14x14xf32>
    %v364 = stablehlo.multiply %v359, %v363 : tensor<32x160x14x14xf32>
    %v365 = stablehlo.broadcast_in_dim %u5qg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v366 = stablehlo.broadcast_in_dim %u5qbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v367 = stablehlo.multiply %v364, %v365 : tensor<32x160x14x14xf32>
    %v368 = stablehlo.add %v367, %v366 : tensor<32x160x14x14xf32>
    %v369 = stablehlo.reshape %v368 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v370 = stablehlo.constant dense<0.0> : tensor<32x31360xf32>
    %v371 = stablehlo.maximum %v369, %v370 : tensor<32x31360xf32>
    %v372 = stablehlo.reshape %v371 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v373 = stablehlo.convolution(%v372, %u5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<32x640x14x14xf32>
    %v374 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v375 = stablehlo.add %v373, %v374 : tensor<32x640x14x14xf32>
    %v376 = stablehlo.reshape %v375 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v377 = stablehlo.reshape %v376 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v378 = stablehlo.broadcast_in_dim %u5enmu, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v379 = stablehlo.subtract %v377, %v378 : tensor<32x640x14x14xf32>
    %v380 = stablehlo.broadcast_in_dim %u5envar, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v381 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v382 = stablehlo.add %v380, %v381 : tensor<32x640x14x14xf32>
    %v383 = stablehlo.rsqrt %v382 : tensor<32x640x14x14xf32>
    %v384 = stablehlo.multiply %v379, %v383 : tensor<32x640x14x14xf32>
    %v385 = stablehlo.broadcast_in_dim %u5eg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v386 = stablehlo.broadcast_in_dim %u5ebt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v387 = stablehlo.multiply %v384, %v385 : tensor<32x640x14x14xf32>
    %v388 = stablehlo.add %v387, %v386 : tensor<32x640x14x14xf32>
    %v389 = stablehlo.reshape %v388 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v390 = stablehlo.constant dense<0.0> : tensor<32x125440xf32>
    %v391 = stablehlo.maximum %v389, %v390 : tensor<32x125440xf32>
    %v392 = stablehlo.reshape %v391 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v393 = stablehlo.convolution(%v392, %u5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 640 : i64} : (tensor<32x640x14x14xf32>, tensor<640x1x5x5xf32>) -> tensor<32x640x14x14xf32>
    %v394 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v395 = stablehlo.add %v393, %v394 : tensor<32x640x14x14xf32>
    %v396 = stablehlo.reshape %v395 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v397 = stablehlo.reshape %v396 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v398 = stablehlo.broadcast_in_dim %u5dnmu, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v399 = stablehlo.subtract %v397, %v398 : tensor<32x640x14x14xf32>
    %v400 = stablehlo.broadcast_in_dim %u5dnvar, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v401 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v402 = stablehlo.add %v400, %v401 : tensor<32x640x14x14xf32>
    %v403 = stablehlo.rsqrt %v402 : tensor<32x640x14x14xf32>
    %v404 = stablehlo.multiply %v399, %v403 : tensor<32x640x14x14xf32>
    %v405 = stablehlo.broadcast_in_dim %u5dg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v406 = stablehlo.broadcast_in_dim %u5dbt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v407 = stablehlo.multiply %v404, %v405 : tensor<32x640x14x14xf32>
    %v408 = stablehlo.add %v407, %v406 : tensor<32x640x14x14xf32>
    %v409 = stablehlo.reshape %v408 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v410 = stablehlo.constant dense<0.0> : tensor<32x125440xf32>
    %v411 = stablehlo.maximum %v409, %v410 : tensor<32x125440xf32>
    %v412 = stablehlo.reshape %v411 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v413 = stablehlo.convolution(%v412, %u5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v414 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v415 = stablehlo.add %v413, %v414 : tensor<32x160x14x14xf32>
    %v416 = stablehlo.reshape %v415 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v417 = stablehlo.reshape %v416 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v418 = stablehlo.broadcast_in_dim %u5pnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v419 = stablehlo.subtract %v417, %v418 : tensor<32x160x14x14xf32>
    %v420 = stablehlo.broadcast_in_dim %u5pnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v421 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v422 = stablehlo.add %v420, %v421 : tensor<32x160x14x14xf32>
    %v423 = stablehlo.rsqrt %v422 : tensor<32x160x14x14xf32>
    %v424 = stablehlo.multiply %v419, %v423 : tensor<32x160x14x14xf32>
    %v425 = stablehlo.broadcast_in_dim %u5pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v426 = stablehlo.broadcast_in_dim %u5pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v427 = stablehlo.multiply %v424, %v425 : tensor<32x160x14x14xf32>
    %v428 = stablehlo.add %v427, %v426 : tensor<32x160x14x14xf32>
    %v429 = stablehlo.reshape %v428 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v430 = stablehlo.add %v429, %v351 : tensor<32x31360xf32>
    %v431 = stablehlo.reshape %v430 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v432 = stablehlo.convolution(%v431, %u6qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x14x14xf32>, tensor<160x1x5x5xf32>) -> tensor<32x160x14x14xf32>
    %v433 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v434 = stablehlo.add %v432, %v433 : tensor<32x160x14x14xf32>
    %v435 = stablehlo.reshape %v434 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v436 = stablehlo.reshape %v435 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v437 = stablehlo.broadcast_in_dim %u6qnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v438 = stablehlo.subtract %v436, %v437 : tensor<32x160x14x14xf32>
    %v439 = stablehlo.broadcast_in_dim %u6qnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v440 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v441 = stablehlo.add %v439, %v440 : tensor<32x160x14x14xf32>
    %v442 = stablehlo.rsqrt %v441 : tensor<32x160x14x14xf32>
    %v443 = stablehlo.multiply %v438, %v442 : tensor<32x160x14x14xf32>
    %v444 = stablehlo.broadcast_in_dim %u6qg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v445 = stablehlo.broadcast_in_dim %u6qbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v446 = stablehlo.multiply %v443, %v444 : tensor<32x160x14x14xf32>
    %v447 = stablehlo.add %v446, %v445 : tensor<32x160x14x14xf32>
    %v448 = stablehlo.reshape %v447 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v449 = stablehlo.constant dense<0.0> : tensor<32x31360xf32>
    %v450 = stablehlo.maximum %v448, %v449 : tensor<32x31360xf32>
    %v451 = stablehlo.reshape %v450 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v452 = stablehlo.convolution(%v451, %u6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<32x640x14x14xf32>
    %v453 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v454 = stablehlo.add %v452, %v453 : tensor<32x640x14x14xf32>
    %v455 = stablehlo.reshape %v454 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v456 = stablehlo.reshape %v455 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v457 = stablehlo.broadcast_in_dim %u6enmu, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v458 = stablehlo.subtract %v456, %v457 : tensor<32x640x14x14xf32>
    %v459 = stablehlo.broadcast_in_dim %u6envar, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v460 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v461 = stablehlo.add %v459, %v460 : tensor<32x640x14x14xf32>
    %v462 = stablehlo.rsqrt %v461 : tensor<32x640x14x14xf32>
    %v463 = stablehlo.multiply %v458, %v462 : tensor<32x640x14x14xf32>
    %v464 = stablehlo.broadcast_in_dim %u6eg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v465 = stablehlo.broadcast_in_dim %u6ebt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v466 = stablehlo.multiply %v463, %v464 : tensor<32x640x14x14xf32>
    %v467 = stablehlo.add %v466, %v465 : tensor<32x640x14x14xf32>
    %v468 = stablehlo.reshape %v467 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v469 = stablehlo.constant dense<0.0> : tensor<32x125440xf32>
    %v470 = stablehlo.maximum %v468, %v469 : tensor<32x125440xf32>
    %v471 = stablehlo.reshape %v470 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v472 = stablehlo.convolution(%v471, %u6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v473 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v474 = stablehlo.add %v472, %v473 : tensor<32x160x14x14xf32>
    %v475 = stablehlo.reshape %v474 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v476 = stablehlo.reshape %v475 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v477 = stablehlo.broadcast_in_dim %u6pnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v478 = stablehlo.subtract %v476, %v477 : tensor<32x160x14x14xf32>
    %v479 = stablehlo.broadcast_in_dim %u6pnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v480 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v481 = stablehlo.add %v479, %v480 : tensor<32x160x14x14xf32>
    %v482 = stablehlo.rsqrt %v481 : tensor<32x160x14x14xf32>
    %v483 = stablehlo.multiply %v478, %v482 : tensor<32x160x14x14xf32>
    %v484 = stablehlo.broadcast_in_dim %u6pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v485 = stablehlo.broadcast_in_dim %u6pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v486 = stablehlo.multiply %v483, %v484 : tensor<32x160x14x14xf32>
    %v487 = stablehlo.add %v486, %v485 : tensor<32x160x14x14xf32>
    %v488 = stablehlo.reshape %v487 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v489 = stablehlo.add %v488, %v430 : tensor<32x31360xf32>
    %v490 = stablehlo.reshape %v489 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v491 = stablehlo.convolution(%v490, %u7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<32x640x14x14xf32>
    %v492 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v493 = stablehlo.add %v491, %v492 : tensor<32x640x14x14xf32>
    %v494 = stablehlo.reshape %v493 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v495 = stablehlo.reshape %v494 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v496 = stablehlo.broadcast_in_dim %u7enmu, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v497 = stablehlo.subtract %v495, %v496 : tensor<32x640x14x14xf32>
    %v498 = stablehlo.broadcast_in_dim %u7envar, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v499 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v500 = stablehlo.add %v498, %v499 : tensor<32x640x14x14xf32>
    %v501 = stablehlo.rsqrt %v500 : tensor<32x640x14x14xf32>
    %v502 = stablehlo.multiply %v497, %v501 : tensor<32x640x14x14xf32>
    %v503 = stablehlo.broadcast_in_dim %u7eg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v504 = stablehlo.broadcast_in_dim %u7ebt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v505 = stablehlo.multiply %v502, %v503 : tensor<32x640x14x14xf32>
    %v506 = stablehlo.add %v505, %v504 : tensor<32x640x14x14xf32>
    %v507 = stablehlo.reshape %v506 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v508 = stablehlo.constant dense<0.0> : tensor<32x125440xf32>
    %v509 = stablehlo.maximum %v507, %v508 : tensor<32x125440xf32>
    %v510 = stablehlo.reshape %v509 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v511 = stablehlo.convolution(%v510, %u7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 640 : i64} : (tensor<32x640x14x14xf32>, tensor<640x1x3x3xf32>) -> tensor<32x640x14x14xf32>
    %v512 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v513 = stablehlo.add %v511, %v512 : tensor<32x640x14x14xf32>
    %v514 = stablehlo.reshape %v513 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v515 = stablehlo.reshape %v514 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v516 = stablehlo.broadcast_in_dim %u7dnmu, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v517 = stablehlo.subtract %v515, %v516 : tensor<32x640x14x14xf32>
    %v518 = stablehlo.broadcast_in_dim %u7dnvar, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v519 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v520 = stablehlo.add %v518, %v519 : tensor<32x640x14x14xf32>
    %v521 = stablehlo.rsqrt %v520 : tensor<32x640x14x14xf32>
    %v522 = stablehlo.multiply %v517, %v521 : tensor<32x640x14x14xf32>
    %v523 = stablehlo.broadcast_in_dim %u7dg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v524 = stablehlo.broadcast_in_dim %u7dbt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v525 = stablehlo.multiply %v522, %v523 : tensor<32x640x14x14xf32>
    %v526 = stablehlo.add %v525, %v524 : tensor<32x640x14x14xf32>
    %v527 = stablehlo.reshape %v526 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v528 = stablehlo.constant dense<0.0> : tensor<32x125440xf32>
    %v529 = stablehlo.maximum %v527, %v528 : tensor<32x125440xf32>
    %v530 = stablehlo.reshape %v529 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v531 = stablehlo.convolution(%v530, %u7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v532 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v533 = stablehlo.add %v531, %v532 : tensor<32x160x14x14xf32>
    %v534 = stablehlo.reshape %v533 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v535 = stablehlo.reshape %v534 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v536 = stablehlo.broadcast_in_dim %u7pnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v537 = stablehlo.subtract %v535, %v536 : tensor<32x160x14x14xf32>
    %v538 = stablehlo.broadcast_in_dim %u7pnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v539 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v540 = stablehlo.add %v538, %v539 : tensor<32x160x14x14xf32>
    %v541 = stablehlo.rsqrt %v540 : tensor<32x160x14x14xf32>
    %v542 = stablehlo.multiply %v537, %v541 : tensor<32x160x14x14xf32>
    %v543 = stablehlo.broadcast_in_dim %u7pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v544 = stablehlo.broadcast_in_dim %u7pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v545 = stablehlo.multiply %v542, %v543 : tensor<32x160x14x14xf32>
    %v546 = stablehlo.add %v545, %v544 : tensor<32x160x14x14xf32>
    %v547 = stablehlo.reshape %v546 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v548 = stablehlo.add %v547, %v489 : tensor<32x31360xf32>
    %v549 = stablehlo.reshape %v548 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v550 = stablehlo.convolution(%v549, %u8qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<32x160x14x14xf32>
    %v551 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v552 = stablehlo.add %v550, %v551 : tensor<32x160x14x14xf32>
    %v553 = stablehlo.reshape %v552 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v554 = stablehlo.reshape %v553 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v555 = stablehlo.broadcast_in_dim %u8qnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v556 = stablehlo.subtract %v554, %v555 : tensor<32x160x14x14xf32>
    %v557 = stablehlo.broadcast_in_dim %u8qnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v558 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v559 = stablehlo.add %v557, %v558 : tensor<32x160x14x14xf32>
    %v560 = stablehlo.rsqrt %v559 : tensor<32x160x14x14xf32>
    %v561 = stablehlo.multiply %v556, %v560 : tensor<32x160x14x14xf32>
    %v562 = stablehlo.broadcast_in_dim %u8qg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v563 = stablehlo.broadcast_in_dim %u8qbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v564 = stablehlo.multiply %v561, %v562 : tensor<32x160x14x14xf32>
    %v565 = stablehlo.add %v564, %v563 : tensor<32x160x14x14xf32>
    %v566 = stablehlo.reshape %v565 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v567 = stablehlo.constant dense<0.0> : tensor<32x31360xf32>
    %v568 = stablehlo.maximum %v566, %v567 : tensor<32x31360xf32>
    %v569 = stablehlo.reshape %v568 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v570 = stablehlo.convolution(%v569, %u8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<32x640x14x14xf32>
    %v571 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v572 = stablehlo.add %v570, %v571 : tensor<32x640x14x14xf32>
    %v573 = stablehlo.reshape %v572 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v574 = stablehlo.reshape %v573 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v575 = stablehlo.broadcast_in_dim %u8enmu, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v576 = stablehlo.subtract %v574, %v575 : tensor<32x640x14x14xf32>
    %v577 = stablehlo.broadcast_in_dim %u8envar, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v578 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v579 = stablehlo.add %v577, %v578 : tensor<32x640x14x14xf32>
    %v580 = stablehlo.rsqrt %v579 : tensor<32x640x14x14xf32>
    %v581 = stablehlo.multiply %v576, %v580 : tensor<32x640x14x14xf32>
    %v582 = stablehlo.broadcast_in_dim %u8eg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v583 = stablehlo.broadcast_in_dim %u8ebt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v584 = stablehlo.multiply %v581, %v582 : tensor<32x640x14x14xf32>
    %v585 = stablehlo.add %v584, %v583 : tensor<32x640x14x14xf32>
    %v586 = stablehlo.reshape %v585 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v587 = stablehlo.constant dense<0.0> : tensor<32x125440xf32>
    %v588 = stablehlo.maximum %v586, %v587 : tensor<32x125440xf32>
    %v589 = stablehlo.reshape %v588 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v590 = stablehlo.convolution(%v589, %u8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v591 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v592 = stablehlo.add %v590, %v591 : tensor<32x160x14x14xf32>
    %v593 = stablehlo.reshape %v592 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v594 = stablehlo.reshape %v593 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v595 = stablehlo.broadcast_in_dim %u8pnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v596 = stablehlo.subtract %v594, %v595 : tensor<32x160x14x14xf32>
    %v597 = stablehlo.broadcast_in_dim %u8pnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v598 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v599 = stablehlo.add %v597, %v598 : tensor<32x160x14x14xf32>
    %v600 = stablehlo.rsqrt %v599 : tensor<32x160x14x14xf32>
    %v601 = stablehlo.multiply %v596, %v600 : tensor<32x160x14x14xf32>
    %v602 = stablehlo.broadcast_in_dim %u8pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v603 = stablehlo.broadcast_in_dim %u8pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v604 = stablehlo.multiply %v601, %v602 : tensor<32x160x14x14xf32>
    %v605 = stablehlo.add %v604, %v603 : tensor<32x160x14x14xf32>
    %v606 = stablehlo.reshape %v605 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v607 = stablehlo.add %v606, %v548 : tensor<32x31360xf32>
    %v608 = stablehlo.reshape %v607 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v609 = stablehlo.convolution(%v608, %u9eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<32x640x14x14xf32>
    %v610 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v611 = stablehlo.add %v609, %v610 : tensor<32x640x14x14xf32>
    %v612 = stablehlo.reshape %v611 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v613 = stablehlo.reshape %v612 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v614 = stablehlo.broadcast_in_dim %u9enmu, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v615 = stablehlo.subtract %v613, %v614 : tensor<32x640x14x14xf32>
    %v616 = stablehlo.broadcast_in_dim %u9envar, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v617 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v618 = stablehlo.add %v616, %v617 : tensor<32x640x14x14xf32>
    %v619 = stablehlo.rsqrt %v618 : tensor<32x640x14x14xf32>
    %v620 = stablehlo.multiply %v615, %v619 : tensor<32x640x14x14xf32>
    %v621 = stablehlo.broadcast_in_dim %u9eg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v622 = stablehlo.broadcast_in_dim %u9ebt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v623 = stablehlo.multiply %v620, %v621 : tensor<32x640x14x14xf32>
    %v624 = stablehlo.add %v623, %v622 : tensor<32x640x14x14xf32>
    %v625 = stablehlo.reshape %v624 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v626 = stablehlo.constant dense<0.0> : tensor<32x125440xf32>
    %v627 = stablehlo.maximum %v625, %v626 : tensor<32x125440xf32>
    %v628 = stablehlo.reshape %v627 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v629 = stablehlo.convolution(%v628, %u9pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v630 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v631 = stablehlo.add %v629, %v630 : tensor<32x160x14x14xf32>
    %v632 = stablehlo.reshape %v631 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v633 = stablehlo.reshape %v632 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v634 = stablehlo.broadcast_in_dim %u9pnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v635 = stablehlo.subtract %v633, %v634 : tensor<32x160x14x14xf32>
    %v636 = stablehlo.broadcast_in_dim %u9pnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v637 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v638 = stablehlo.add %v636, %v637 : tensor<32x160x14x14xf32>
    %v639 = stablehlo.rsqrt %v638 : tensor<32x160x14x14xf32>
    %v640 = stablehlo.multiply %v635, %v639 : tensor<32x160x14x14xf32>
    %v641 = stablehlo.broadcast_in_dim %u9pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v642 = stablehlo.broadcast_in_dim %u9pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v643 = stablehlo.multiply %v640, %v641 : tensor<32x160x14x14xf32>
    %v644 = stablehlo.add %v643, %v642 : tensor<32x160x14x14xf32>
    %v645 = stablehlo.reshape %v644 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v646 = stablehlo.add %v645, %v607 : tensor<32x31360xf32>
    %v647 = stablehlo.reshape %v646 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v648 = stablehlo.convolution(%v647, %u10qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<32x160x14x14xf32>
    %v649 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v650 = stablehlo.add %v648, %v649 : tensor<32x160x14x14xf32>
    %v651 = stablehlo.reshape %v650 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v652 = stablehlo.reshape %v651 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v653 = stablehlo.broadcast_in_dim %u10qnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v654 = stablehlo.subtract %v652, %v653 : tensor<32x160x14x14xf32>
    %v655 = stablehlo.broadcast_in_dim %u10qnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v656 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v657 = stablehlo.add %v655, %v656 : tensor<32x160x14x14xf32>
    %v658 = stablehlo.rsqrt %v657 : tensor<32x160x14x14xf32>
    %v659 = stablehlo.multiply %v654, %v658 : tensor<32x160x14x14xf32>
    %v660 = stablehlo.broadcast_in_dim %u10qg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v661 = stablehlo.broadcast_in_dim %u10qbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v662 = stablehlo.multiply %v659, %v660 : tensor<32x160x14x14xf32>
    %v663 = stablehlo.add %v662, %v661 : tensor<32x160x14x14xf32>
    %v664 = stablehlo.reshape %v663 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v665 = stablehlo.constant dense<0.0> : tensor<32x31360xf32>
    %v666 = stablehlo.maximum %v664, %v665 : tensor<32x31360xf32>
    %v667 = stablehlo.reshape %v666 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v668 = stablehlo.convolution(%v667, %u10eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<32x640x14x14xf32>
    %v669 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v670 = stablehlo.add %v668, %v669 : tensor<32x640x14x14xf32>
    %v671 = stablehlo.reshape %v670 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v672 = stablehlo.reshape %v671 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v673 = stablehlo.broadcast_in_dim %u10enmu, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v674 = stablehlo.subtract %v672, %v673 : tensor<32x640x14x14xf32>
    %v675 = stablehlo.broadcast_in_dim %u10envar, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v676 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v677 = stablehlo.add %v675, %v676 : tensor<32x640x14x14xf32>
    %v678 = stablehlo.rsqrt %v677 : tensor<32x640x14x14xf32>
    %v679 = stablehlo.multiply %v674, %v678 : tensor<32x640x14x14xf32>
    %v680 = stablehlo.broadcast_in_dim %u10eg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v681 = stablehlo.broadcast_in_dim %u10ebt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v682 = stablehlo.multiply %v679, %v680 : tensor<32x640x14x14xf32>
    %v683 = stablehlo.add %v682, %v681 : tensor<32x640x14x14xf32>
    %v684 = stablehlo.reshape %v683 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v685 = stablehlo.constant dense<0.0> : tensor<32x125440xf32>
    %v686 = stablehlo.maximum %v684, %v685 : tensor<32x125440xf32>
    %v687 = stablehlo.reshape %v686 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v688 = stablehlo.convolution(%v687, %u10dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 640 : i64} : (tensor<32x640x14x14xf32>, tensor<640x1x3x3xf32>) -> tensor<32x640x14x14xf32>
    %v689 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v690 = stablehlo.add %v688, %v689 : tensor<32x640x14x14xf32>
    %v691 = stablehlo.reshape %v690 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v692 = stablehlo.reshape %v691 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v693 = stablehlo.broadcast_in_dim %u10dnmu, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v694 = stablehlo.subtract %v692, %v693 : tensor<32x640x14x14xf32>
    %v695 = stablehlo.broadcast_in_dim %u10dnvar, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v696 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v697 = stablehlo.add %v695, %v696 : tensor<32x640x14x14xf32>
    %v698 = stablehlo.rsqrt %v697 : tensor<32x640x14x14xf32>
    %v699 = stablehlo.multiply %v694, %v698 : tensor<32x640x14x14xf32>
    %v700 = stablehlo.broadcast_in_dim %u10dg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v701 = stablehlo.broadcast_in_dim %u10dbt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v702 = stablehlo.multiply %v699, %v700 : tensor<32x640x14x14xf32>
    %v703 = stablehlo.add %v702, %v701 : tensor<32x640x14x14xf32>
    %v704 = stablehlo.reshape %v703 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v705 = stablehlo.constant dense<0.0> : tensor<32x125440xf32>
    %v706 = stablehlo.maximum %v704, %v705 : tensor<32x125440xf32>
    %v707 = stablehlo.reshape %v706 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v708 = stablehlo.convolution(%v707, %u10pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v709 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v710 = stablehlo.add %v708, %v709 : tensor<32x160x14x14xf32>
    %v711 = stablehlo.reshape %v710 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v712 = stablehlo.reshape %v711 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v713 = stablehlo.broadcast_in_dim %u10pnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v714 = stablehlo.subtract %v712, %v713 : tensor<32x160x14x14xf32>
    %v715 = stablehlo.broadcast_in_dim %u10pnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v716 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v717 = stablehlo.add %v715, %v716 : tensor<32x160x14x14xf32>
    %v718 = stablehlo.rsqrt %v717 : tensor<32x160x14x14xf32>
    %v719 = stablehlo.multiply %v714, %v718 : tensor<32x160x14x14xf32>
    %v720 = stablehlo.broadcast_in_dim %u10pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v721 = stablehlo.broadcast_in_dim %u10pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v722 = stablehlo.multiply %v719, %v720 : tensor<32x160x14x14xf32>
    %v723 = stablehlo.add %v722, %v721 : tensor<32x160x14x14xf32>
    %v724 = stablehlo.reshape %v723 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v725 = stablehlo.add %v724, %v646 : tensor<32x31360xf32>
    %v726 = stablehlo.reshape %v725 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v727 = stablehlo.convolution(%v726, %u11qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x14x14xf32>, tensor<160x1x5x5xf32>) -> tensor<32x160x7x7xf32>
    %v728 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v729 = stablehlo.add %v727, %v728 : tensor<32x160x7x7xf32>
    %v730 = stablehlo.reshape %v729 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v731 = stablehlo.reshape %v730 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v732 = stablehlo.broadcast_in_dim %u11qnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v733 = stablehlo.subtract %v731, %v732 : tensor<32x160x7x7xf32>
    %v734 = stablehlo.broadcast_in_dim %u11qnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v735 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v736 = stablehlo.add %v734, %v735 : tensor<32x160x7x7xf32>
    %v737 = stablehlo.rsqrt %v736 : tensor<32x160x7x7xf32>
    %v738 = stablehlo.multiply %v733, %v737 : tensor<32x160x7x7xf32>
    %v739 = stablehlo.broadcast_in_dim %u11qg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v740 = stablehlo.broadcast_in_dim %u11qbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v741 = stablehlo.multiply %v738, %v739 : tensor<32x160x7x7xf32>
    %v742 = stablehlo.add %v741, %v740 : tensor<32x160x7x7xf32>
    %v743 = stablehlo.reshape %v742 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v744 = stablehlo.constant dense<0.0> : tensor<32x7840xf32>
    %v745 = stablehlo.maximum %v743, %v744 : tensor<32x7840xf32>
    %v746 = stablehlo.reshape %v745 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v747 = stablehlo.convolution(%v746, %u11eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v748 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v749 = stablehlo.add %v747, %v748 : tensor<32x960x7x7xf32>
    %v750 = stablehlo.reshape %v749 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v751 = stablehlo.reshape %v750 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v752 = stablehlo.broadcast_in_dim %u11enmu, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v753 = stablehlo.subtract %v751, %v752 : tensor<32x960x7x7xf32>
    %v754 = stablehlo.broadcast_in_dim %u11envar, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v755 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v756 = stablehlo.add %v754, %v755 : tensor<32x960x7x7xf32>
    %v757 = stablehlo.rsqrt %v756 : tensor<32x960x7x7xf32>
    %v758 = stablehlo.multiply %v753, %v757 : tensor<32x960x7x7xf32>
    %v759 = stablehlo.broadcast_in_dim %u11eg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v760 = stablehlo.broadcast_in_dim %u11ebt, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v761 = stablehlo.multiply %v758, %v759 : tensor<32x960x7x7xf32>
    %v762 = stablehlo.add %v761, %v760 : tensor<32x960x7x7xf32>
    %v763 = stablehlo.reshape %v762 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v764 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v765 = stablehlo.maximum %v763, %v764 : tensor<32x47040xf32>
    %v766 = stablehlo.reshape %v765 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v767 = stablehlo.convolution(%v766, %u11dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x5x5xf32>) -> tensor<32x960x7x7xf32>
    %v768 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v769 = stablehlo.add %v767, %v768 : tensor<32x960x7x7xf32>
    %v770 = stablehlo.reshape %v769 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v771 = stablehlo.reshape %v770 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v772 = stablehlo.broadcast_in_dim %u11dnmu, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v773 = stablehlo.subtract %v771, %v772 : tensor<32x960x7x7xf32>
    %v774 = stablehlo.broadcast_in_dim %u11dnvar, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v775 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v776 = stablehlo.add %v774, %v775 : tensor<32x960x7x7xf32>
    %v777 = stablehlo.rsqrt %v776 : tensor<32x960x7x7xf32>
    %v778 = stablehlo.multiply %v773, %v777 : tensor<32x960x7x7xf32>
    %v779 = stablehlo.broadcast_in_dim %u11dg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v780 = stablehlo.broadcast_in_dim %u11dbt, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v781 = stablehlo.multiply %v778, %v779 : tensor<32x960x7x7xf32>
    %v782 = stablehlo.add %v781, %v780 : tensor<32x960x7x7xf32>
    %v783 = stablehlo.reshape %v782 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v784 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v785 = stablehlo.maximum %v783, %v784 : tensor<32x47040xf32>
    %v786 = stablehlo.reshape %v785 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v787 = stablehlo.convolution(%v786, %u11pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<256x960x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v788 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v789 = stablehlo.add %v787, %v788 : tensor<32x256x7x7xf32>
    %v790 = stablehlo.reshape %v789 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v791 = stablehlo.reshape %v790 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v792 = stablehlo.broadcast_in_dim %u11pnmu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v793 = stablehlo.subtract %v791, %v792 : tensor<32x256x7x7xf32>
    %v794 = stablehlo.broadcast_in_dim %u11pnvar, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v795 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v796 = stablehlo.add %v794, %v795 : tensor<32x256x7x7xf32>
    %v797 = stablehlo.rsqrt %v796 : tensor<32x256x7x7xf32>
    %v798 = stablehlo.multiply %v793, %v797 : tensor<32x256x7x7xf32>
    %v799 = stablehlo.broadcast_in_dim %u11pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v800 = stablehlo.broadcast_in_dim %u11pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v801 = stablehlo.multiply %v798, %v799 : tensor<32x256x7x7xf32>
    %v802 = stablehlo.add %v801, %v800 : tensor<32x256x7x7xf32>
    %v803 = stablehlo.reshape %v802 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v804 = stablehlo.reshape %v803 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v805 = stablehlo.convolution(%v804, %u12qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x7x7xf32>, tensor<256x1x5x5xf32>) -> tensor<32x256x7x7xf32>
    %v806 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v807 = stablehlo.add %v805, %v806 : tensor<32x256x7x7xf32>
    %v808 = stablehlo.reshape %v807 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v809 = stablehlo.reshape %v808 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v810 = stablehlo.broadcast_in_dim %u12qnmu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v811 = stablehlo.subtract %v809, %v810 : tensor<32x256x7x7xf32>
    %v812 = stablehlo.broadcast_in_dim %u12qnvar, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v813 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v814 = stablehlo.add %v812, %v813 : tensor<32x256x7x7xf32>
    %v815 = stablehlo.rsqrt %v814 : tensor<32x256x7x7xf32>
    %v816 = stablehlo.multiply %v811, %v815 : tensor<32x256x7x7xf32>
    %v817 = stablehlo.broadcast_in_dim %u12qg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v818 = stablehlo.broadcast_in_dim %u12qbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v819 = stablehlo.multiply %v816, %v817 : tensor<32x256x7x7xf32>
    %v820 = stablehlo.add %v819, %v818 : tensor<32x256x7x7xf32>
    %v821 = stablehlo.reshape %v820 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v822 = stablehlo.constant dense<0.0> : tensor<32x12544xf32>
    %v823 = stablehlo.maximum %v821, %v822 : tensor<32x12544xf32>
    %v824 = stablehlo.reshape %v823 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v825 = stablehlo.convolution(%v824, %u12eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v826 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v827 = stablehlo.add %v825, %v826 : tensor<32x1024x7x7xf32>
    %v828 = stablehlo.reshape %v827 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v829 = stablehlo.reshape %v828 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v830 = stablehlo.broadcast_in_dim %u12enmu, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v831 = stablehlo.subtract %v829, %v830 : tensor<32x1024x7x7xf32>
    %v832 = stablehlo.broadcast_in_dim %u12envar, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v833 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v834 = stablehlo.add %v832, %v833 : tensor<32x1024x7x7xf32>
    %v835 = stablehlo.rsqrt %v834 : tensor<32x1024x7x7xf32>
    %v836 = stablehlo.multiply %v831, %v835 : tensor<32x1024x7x7xf32>
    %v837 = stablehlo.broadcast_in_dim %u12eg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v838 = stablehlo.broadcast_in_dim %u12ebt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v839 = stablehlo.multiply %v836, %v837 : tensor<32x1024x7x7xf32>
    %v840 = stablehlo.add %v839, %v838 : tensor<32x1024x7x7xf32>
    %v841 = stablehlo.reshape %v840 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v842 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v843 = stablehlo.maximum %v841, %v842 : tensor<32x50176xf32>
    %v844 = stablehlo.reshape %v843 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v845 = stablehlo.convolution(%v844, %u12dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<32x1024x7x7xf32>, tensor<1024x1x5x5xf32>) -> tensor<32x1024x7x7xf32>
    %v846 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v847 = stablehlo.add %v845, %v846 : tensor<32x1024x7x7xf32>
    %v848 = stablehlo.reshape %v847 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v849 = stablehlo.reshape %v848 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v850 = stablehlo.broadcast_in_dim %u12dnmu, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v851 = stablehlo.subtract %v849, %v850 : tensor<32x1024x7x7xf32>
    %v852 = stablehlo.broadcast_in_dim %u12dnvar, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v853 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v854 = stablehlo.add %v852, %v853 : tensor<32x1024x7x7xf32>
    %v855 = stablehlo.rsqrt %v854 : tensor<32x1024x7x7xf32>
    %v856 = stablehlo.multiply %v851, %v855 : tensor<32x1024x7x7xf32>
    %v857 = stablehlo.broadcast_in_dim %u12dg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v858 = stablehlo.broadcast_in_dim %u12dbt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v859 = stablehlo.multiply %v856, %v857 : tensor<32x1024x7x7xf32>
    %v860 = stablehlo.add %v859, %v858 : tensor<32x1024x7x7xf32>
    %v861 = stablehlo.reshape %v860 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v862 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v863 = stablehlo.maximum %v861, %v862 : tensor<32x50176xf32>
    %v864 = stablehlo.reshape %v863 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v865 = stablehlo.convolution(%v864, %u12pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v866 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v867 = stablehlo.add %v865, %v866 : tensor<32x256x7x7xf32>
    %v868 = stablehlo.reshape %v867 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v869 = stablehlo.reshape %v868 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v870 = stablehlo.broadcast_in_dim %u12pnmu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v871 = stablehlo.subtract %v869, %v870 : tensor<32x256x7x7xf32>
    %v872 = stablehlo.broadcast_in_dim %u12pnvar, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v873 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v874 = stablehlo.add %v872, %v873 : tensor<32x256x7x7xf32>
    %v875 = stablehlo.rsqrt %v874 : tensor<32x256x7x7xf32>
    %v876 = stablehlo.multiply %v871, %v875 : tensor<32x256x7x7xf32>
    %v877 = stablehlo.broadcast_in_dim %u12pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v878 = stablehlo.broadcast_in_dim %u12pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v879 = stablehlo.multiply %v876, %v877 : tensor<32x256x7x7xf32>
    %v880 = stablehlo.add %v879, %v878 : tensor<32x256x7x7xf32>
    %v881 = stablehlo.reshape %v880 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v882 = stablehlo.add %v881, %v803 : tensor<32x12544xf32>
    %v883 = stablehlo.reshape %v882 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v884 = stablehlo.convolution(%v883, %u13eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v885 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v886 = stablehlo.add %v884, %v885 : tensor<32x1024x7x7xf32>
    %v887 = stablehlo.reshape %v886 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v888 = stablehlo.reshape %v887 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v889 = stablehlo.broadcast_in_dim %u13enmu, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v890 = stablehlo.subtract %v888, %v889 : tensor<32x1024x7x7xf32>
    %v891 = stablehlo.broadcast_in_dim %u13envar, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v892 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v893 = stablehlo.add %v891, %v892 : tensor<32x1024x7x7xf32>
    %v894 = stablehlo.rsqrt %v893 : tensor<32x1024x7x7xf32>
    %v895 = stablehlo.multiply %v890, %v894 : tensor<32x1024x7x7xf32>
    %v896 = stablehlo.broadcast_in_dim %u13eg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v897 = stablehlo.broadcast_in_dim %u13ebt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v898 = stablehlo.multiply %v895, %v896 : tensor<32x1024x7x7xf32>
    %v899 = stablehlo.add %v898, %v897 : tensor<32x1024x7x7xf32>
    %v900 = stablehlo.reshape %v899 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v901 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v902 = stablehlo.maximum %v900, %v901 : tensor<32x50176xf32>
    %v903 = stablehlo.reshape %v902 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v904 = stablehlo.convolution(%v903, %u13dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<32x1024x7x7xf32>, tensor<1024x1x3x3xf32>) -> tensor<32x1024x7x7xf32>
    %v905 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v906 = stablehlo.add %v904, %v905 : tensor<32x1024x7x7xf32>
    %v907 = stablehlo.reshape %v906 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v908 = stablehlo.reshape %v907 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v909 = stablehlo.broadcast_in_dim %u13dnmu, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v910 = stablehlo.subtract %v908, %v909 : tensor<32x1024x7x7xf32>
    %v911 = stablehlo.broadcast_in_dim %u13dnvar, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v912 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v913 = stablehlo.add %v911, %v912 : tensor<32x1024x7x7xf32>
    %v914 = stablehlo.rsqrt %v913 : tensor<32x1024x7x7xf32>
    %v915 = stablehlo.multiply %v910, %v914 : tensor<32x1024x7x7xf32>
    %v916 = stablehlo.broadcast_in_dim %u13dg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v917 = stablehlo.broadcast_in_dim %u13dbt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v918 = stablehlo.multiply %v915, %v916 : tensor<32x1024x7x7xf32>
    %v919 = stablehlo.add %v918, %v917 : tensor<32x1024x7x7xf32>
    %v920 = stablehlo.reshape %v919 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v921 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v922 = stablehlo.maximum %v920, %v921 : tensor<32x50176xf32>
    %v923 = stablehlo.reshape %v922 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v924 = stablehlo.convolution(%v923, %u13pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v925 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v926 = stablehlo.add %v924, %v925 : tensor<32x256x7x7xf32>
    %v927 = stablehlo.reshape %v926 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v928 = stablehlo.reshape %v927 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v929 = stablehlo.broadcast_in_dim %u13pnmu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v930 = stablehlo.subtract %v928, %v929 : tensor<32x256x7x7xf32>
    %v931 = stablehlo.broadcast_in_dim %u13pnvar, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v932 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v933 = stablehlo.add %v931, %v932 : tensor<32x256x7x7xf32>
    %v934 = stablehlo.rsqrt %v933 : tensor<32x256x7x7xf32>
    %v935 = stablehlo.multiply %v930, %v934 : tensor<32x256x7x7xf32>
    %v936 = stablehlo.broadcast_in_dim %u13pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v937 = stablehlo.broadcast_in_dim %u13pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v938 = stablehlo.multiply %v935, %v936 : tensor<32x256x7x7xf32>
    %v939 = stablehlo.add %v938, %v937 : tensor<32x256x7x7xf32>
    %v940 = stablehlo.reshape %v939 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v941 = stablehlo.add %v940, %v882 : tensor<32x12544xf32>
    %v942 = stablehlo.reshape %v941 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v943 = stablehlo.convolution(%v942, %u14qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x7x7xf32>, tensor<256x1x3x3xf32>) -> tensor<32x256x7x7xf32>
    %v944 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v945 = stablehlo.add %v943, %v944 : tensor<32x256x7x7xf32>
    %v946 = stablehlo.reshape %v945 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v947 = stablehlo.reshape %v946 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v948 = stablehlo.broadcast_in_dim %u14qnmu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v949 = stablehlo.subtract %v947, %v948 : tensor<32x256x7x7xf32>
    %v950 = stablehlo.broadcast_in_dim %u14qnvar, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v951 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v952 = stablehlo.add %v950, %v951 : tensor<32x256x7x7xf32>
    %v953 = stablehlo.rsqrt %v952 : tensor<32x256x7x7xf32>
    %v954 = stablehlo.multiply %v949, %v953 : tensor<32x256x7x7xf32>
    %v955 = stablehlo.broadcast_in_dim %u14qg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v956 = stablehlo.broadcast_in_dim %u14qbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v957 = stablehlo.multiply %v954, %v955 : tensor<32x256x7x7xf32>
    %v958 = stablehlo.add %v957, %v956 : tensor<32x256x7x7xf32>
    %v959 = stablehlo.reshape %v958 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v960 = stablehlo.constant dense<0.0> : tensor<32x12544xf32>
    %v961 = stablehlo.maximum %v959, %v960 : tensor<32x12544xf32>
    %v962 = stablehlo.reshape %v961 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v963 = stablehlo.convolution(%v962, %u14eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v964 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v965 = stablehlo.add %v963, %v964 : tensor<32x1024x7x7xf32>
    %v966 = stablehlo.reshape %v965 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v967 = stablehlo.reshape %v966 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v968 = stablehlo.broadcast_in_dim %u14enmu, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v969 = stablehlo.subtract %v967, %v968 : tensor<32x1024x7x7xf32>
    %v970 = stablehlo.broadcast_in_dim %u14envar, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v971 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v972 = stablehlo.add %v970, %v971 : tensor<32x1024x7x7xf32>
    %v973 = stablehlo.rsqrt %v972 : tensor<32x1024x7x7xf32>
    %v974 = stablehlo.multiply %v969, %v973 : tensor<32x1024x7x7xf32>
    %v975 = stablehlo.broadcast_in_dim %u14eg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v976 = stablehlo.broadcast_in_dim %u14ebt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v977 = stablehlo.multiply %v974, %v975 : tensor<32x1024x7x7xf32>
    %v978 = stablehlo.add %v977, %v976 : tensor<32x1024x7x7xf32>
    %v979 = stablehlo.reshape %v978 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v980 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v981 = stablehlo.maximum %v979, %v980 : tensor<32x50176xf32>
    %v982 = stablehlo.reshape %v981 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v983 = stablehlo.convolution(%v982, %u14pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v984 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v985 = stablehlo.add %v983, %v984 : tensor<32x256x7x7xf32>
    %v986 = stablehlo.reshape %v985 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v987 = stablehlo.reshape %v986 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v988 = stablehlo.broadcast_in_dim %u14pnmu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v989 = stablehlo.subtract %v987, %v988 : tensor<32x256x7x7xf32>
    %v990 = stablehlo.broadcast_in_dim %u14pnvar, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v991 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v992 = stablehlo.add %v990, %v991 : tensor<32x256x7x7xf32>
    %v993 = stablehlo.rsqrt %v992 : tensor<32x256x7x7xf32>
    %v994 = stablehlo.multiply %v989, %v993 : tensor<32x256x7x7xf32>
    %v995 = stablehlo.broadcast_in_dim %u14pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v996 = stablehlo.broadcast_in_dim %u14pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v997 = stablehlo.multiply %v994, %v995 : tensor<32x256x7x7xf32>
    %v998 = stablehlo.add %v997, %v996 : tensor<32x256x7x7xf32>
    %v999 = stablehlo.reshape %v998 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1000 = stablehlo.add %v999, %v941 : tensor<32x12544xf32>
    %v1001 = stablehlo.reshape %v1000 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1002 = stablehlo.convolution(%v1001, %hW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<1280x256x1x1xf32>) -> tensor<32x1280x7x7xf32>
    %v1003 = stablehlo.broadcast_in_dim %zb1280, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1004 = stablehlo.add %v1002, %v1003 : tensor<32x1280x7x7xf32>
    %v1005 = stablehlo.reshape %v1004 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1006 = stablehlo.reshape %v1005 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1007 = stablehlo.broadcast_in_dim %hnmu, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1008 = stablehlo.subtract %v1006, %v1007 : tensor<32x1280x7x7xf32>
    %v1009 = stablehlo.broadcast_in_dim %hnvar, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1010 = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %v1011 = stablehlo.add %v1009, %v1010 : tensor<32x1280x7x7xf32>
    %v1012 = stablehlo.rsqrt %v1011 : tensor<32x1280x7x7xf32>
    %v1013 = stablehlo.multiply %v1008, %v1012 : tensor<32x1280x7x7xf32>
    %v1014 = stablehlo.broadcast_in_dim %hg, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1015 = stablehlo.broadcast_in_dim %hbt, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1016 = stablehlo.multiply %v1013, %v1014 : tensor<32x1280x7x7xf32>
    %v1017 = stablehlo.add %v1016, %v1015 : tensor<32x1280x7x7xf32>
    %v1018 = stablehlo.reshape %v1017 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1019 = stablehlo.constant dense<0.0> : tensor<32x62720xf32>
    %v1020 = stablehlo.maximum %v1018, %v1019 : tensor<32x62720xf32>
    %v1021 = stablehlo.reshape %v1020 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1022 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1023 = stablehlo.reduce(%v1021 init: %v1022) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v1024 = stablehlo.constant dense<49.0> : tensor<32x1280xf32>
    %v1025 = stablehlo.divide %v1023, %v1024 : tensor<32x1280xf32>
    %v1026 = stablehlo.dot_general %v1025, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1280xf32>, tensor<1280x10xf32>) -> tensor<32x10xf32>
    %v1027 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v1028 = stablehlo.add %v1026, %v1027 : tensor<32x10xf32>
    return %v1028 : tensor<32x10xf32>
  }
}
