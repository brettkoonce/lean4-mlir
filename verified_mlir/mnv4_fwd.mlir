module @m {
  func.func @mnv4_fwd(%x: tensor<32x150528xf32>, %sW: tensor<32x3x3x3xf32>, %sg: tensor<32xf32>, %sbt: tensor<32xf32>, %f0cW: tensor<128x32x3x3xf32>, %f0cg: tensor<128xf32>, %f0cbt: tensor<128xf32>, %f0pW: tensor<48x128x1x1xf32>, %f0pg: tensor<48xf32>, %f0pbt: tensor<48xf32>, %u1qW: tensor<48x1x3x3xf32>, %u1qg: tensor<48xf32>, %u1qbt: tensor<48xf32>, %u1eW: tensor<192x48x1x1xf32>, %u1eg: tensor<192xf32>, %u1ebt: tensor<192xf32>, %u1dW: tensor<192x1x5x5xf32>, %u1dg: tensor<192xf32>, %u1dbt: tensor<192xf32>, %u1pW: tensor<80x192x1x1xf32>, %u1pg: tensor<80xf32>, %u1pbt: tensor<80xf32>, %u2qW: tensor<80x1x3x3xf32>, %u2qg: tensor<80xf32>, %u2qbt: tensor<80xf32>, %u2eW: tensor<160x80x1x1xf32>, %u2eg: tensor<160xf32>, %u2ebt: tensor<160xf32>, %u2dW: tensor<160x1x3x3xf32>, %u2dg: tensor<160xf32>, %u2dbt: tensor<160xf32>, %u2pW: tensor<80x160x1x1xf32>, %u2pg: tensor<80xf32>, %u2pbt: tensor<80xf32>, %u3qW: tensor<80x1x3x3xf32>, %u3qg: tensor<80xf32>, %u3qbt: tensor<80xf32>, %u3eW: tensor<480x80x1x1xf32>, %u3eg: tensor<480xf32>, %u3ebt: tensor<480xf32>, %u3dW: tensor<480x1x5x5xf32>, %u3dg: tensor<480xf32>, %u3dbt: tensor<480xf32>, %u3pW: tensor<160x480x1x1xf32>, %u3pg: tensor<160xf32>, %u3pbt: tensor<160xf32>, %u4qW: tensor<160x1x3x3xf32>, %u4qg: tensor<160xf32>, %u4qbt: tensor<160xf32>, %u4eW: tensor<640x160x1x1xf32>, %u4eg: tensor<640xf32>, %u4ebt: tensor<640xf32>, %u4dW: tensor<640x1x3x3xf32>, %u4dg: tensor<640xf32>, %u4dbt: tensor<640xf32>, %u4pW: tensor<160x640x1x1xf32>, %u4pg: tensor<160xf32>, %u4pbt: tensor<160xf32>, %u5qW: tensor<160x1x3x3xf32>, %u5qg: tensor<160xf32>, %u5qbt: tensor<160xf32>, %u5eW: tensor<640x160x1x1xf32>, %u5eg: tensor<640xf32>, %u5ebt: tensor<640xf32>, %u5dW: tensor<640x1x3x3xf32>, %u5dg: tensor<640xf32>, %u5dbt: tensor<640xf32>, %u5pW: tensor<160x640x1x1xf32>, %u5pg: tensor<160xf32>, %u5pbt: tensor<160xf32>, %u6qW: tensor<160x1x3x3xf32>, %u6qg: tensor<160xf32>, %u6qbt: tensor<160xf32>, %u6eW: tensor<640x160x1x1xf32>, %u6eg: tensor<640xf32>, %u6ebt: tensor<640xf32>, %u6dW: tensor<640x1x5x5xf32>, %u6dg: tensor<640xf32>, %u6dbt: tensor<640xf32>, %u6pW: tensor<160x640x1x1xf32>, %u6pg: tensor<160xf32>, %u6pbt: tensor<160xf32>, %u7qW: tensor<160x1x3x3xf32>, %u7qg: tensor<160xf32>, %u7qbt: tensor<160xf32>, %u7eW: tensor<640x160x1x1xf32>, %u7eg: tensor<640xf32>, %u7ebt: tensor<640xf32>, %u7dW: tensor<640x1x3x3xf32>, %u7dg: tensor<640xf32>, %u7dbt: tensor<640xf32>, %u7pW: tensor<160x640x1x1xf32>, %u7pg: tensor<160xf32>, %u7pbt: tensor<160xf32>, %u8qW: tensor<160x1x3x3xf32>, %u8qg: tensor<160xf32>, %u8qbt: tensor<160xf32>, %u8eW: tensor<640x160x1x1xf32>, %u8eg: tensor<640xf32>, %u8ebt: tensor<640xf32>, %u8pW: tensor<160x640x1x1xf32>, %u8pg: tensor<160xf32>, %u8pbt: tensor<160xf32>, %u9eW: tensor<320x160x1x1xf32>, %u9eg: tensor<320xf32>, %u9ebt: tensor<320xf32>, %u9pW: tensor<160x320x1x1xf32>, %u9pg: tensor<160xf32>, %u9pbt: tensor<160xf32>, %u10qW: tensor<160x1x3x3xf32>, %u10qg: tensor<160xf32>, %u10qbt: tensor<160xf32>, %u10eW: tensor<640x160x1x1xf32>, %u10eg: tensor<640xf32>, %u10ebt: tensor<640xf32>, %u10pW: tensor<160x640x1x1xf32>, %u10pg: tensor<160xf32>, %u10pbt: tensor<160xf32>, %u11qW: tensor<160x1x5x5xf32>, %u11qg: tensor<160xf32>, %u11qbt: tensor<160xf32>, %u11eW: tensor<960x160x1x1xf32>, %u11eg: tensor<960xf32>, %u11ebt: tensor<960xf32>, %u11dW: tensor<960x1x5x5xf32>, %u11dg: tensor<960xf32>, %u11dbt: tensor<960xf32>, %u11pW: tensor<256x960x1x1xf32>, %u11pg: tensor<256xf32>, %u11pbt: tensor<256xf32>, %u12qW: tensor<256x1x5x5xf32>, %u12qg: tensor<256xf32>, %u12qbt: tensor<256xf32>, %u12eW: tensor<1024x256x1x1xf32>, %u12eg: tensor<1024xf32>, %u12ebt: tensor<1024xf32>, %u12dW: tensor<1024x1x5x5xf32>, %u12dg: tensor<1024xf32>, %u12dbt: tensor<1024xf32>, %u12pW: tensor<256x1024x1x1xf32>, %u12pg: tensor<256xf32>, %u12pbt: tensor<256xf32>, %u13qW: tensor<256x1x3x3xf32>, %u13qg: tensor<256xf32>, %u13qbt: tensor<256xf32>, %u13eW: tensor<1024x256x1x1xf32>, %u13eg: tensor<1024xf32>, %u13ebt: tensor<1024xf32>, %u13dW: tensor<1024x1x5x5xf32>, %u13dg: tensor<1024xf32>, %u13dbt: tensor<1024xf32>, %u13pW: tensor<256x1024x1x1xf32>, %u13pg: tensor<256xf32>, %u13pbt: tensor<256xf32>, %u14qW: tensor<256x1x3x3xf32>, %u14qg: tensor<256xf32>, %u14qbt: tensor<256xf32>, %u14eW: tensor<1024x256x1x1xf32>, %u14eg: tensor<1024xf32>, %u14ebt: tensor<1024xf32>, %u14dW: tensor<1024x1x5x5xf32>, %u14dg: tensor<1024xf32>, %u14dbt: tensor<1024xf32>, %u14pW: tensor<256x1024x1x1xf32>, %u14pg: tensor<256xf32>, %u14pbt: tensor<256xf32>, %u15eW: tensor<1024x256x1x1xf32>, %u15eg: tensor<1024xf32>, %u15ebt: tensor<1024xf32>, %u15pW: tensor<256x1024x1x1xf32>, %u15pg: tensor<256xf32>, %u15pbt: tensor<256xf32>, %u16qW: tensor<256x1x3x3xf32>, %u16qg: tensor<256xf32>, %u16qbt: tensor<256xf32>, %u16eW: tensor<1024x256x1x1xf32>, %u16eg: tensor<1024xf32>, %u16ebt: tensor<1024xf32>, %u16pW: tensor<256x1024x1x1xf32>, %u16pg: tensor<256xf32>, %u16pbt: tensor<256xf32>, %u17qW: tensor<256x1x3x3xf32>, %u17qg: tensor<256xf32>, %u17qbt: tensor<256xf32>, %u17eW: tensor<512x256x1x1xf32>, %u17eg: tensor<512xf32>, %u17ebt: tensor<512xf32>, %u17dW: tensor<512x1x5x5xf32>, %u17dg: tensor<512xf32>, %u17dbt: tensor<512xf32>, %u17pW: tensor<256x512x1x1xf32>, %u17pg: tensor<256xf32>, %u17pbt: tensor<256xf32>, %u18qW: tensor<256x1x5x5xf32>, %u18qg: tensor<256xf32>, %u18qbt: tensor<256xf32>, %u18eW: tensor<1024x256x1x1xf32>, %u18eg: tensor<1024xf32>, %u18ebt: tensor<1024xf32>, %u18dW: tensor<1024x1x5x5xf32>, %u18dg: tensor<1024xf32>, %u18dbt: tensor<1024xf32>, %u18pW: tensor<256x1024x1x1xf32>, %u18pg: tensor<256xf32>, %u18pbt: tensor<256xf32>, %u19eW: tensor<1024x256x1x1xf32>, %u19eg: tensor<1024xf32>, %u19ebt: tensor<1024xf32>, %u19pW: tensor<256x1024x1x1xf32>, %u19pg: tensor<256xf32>, %u19pbt: tensor<256xf32>, %u20eW: tensor<1024x256x1x1xf32>, %u20eg: tensor<1024xf32>, %u20ebt: tensor<1024xf32>, %u20pW: tensor<256x1024x1x1xf32>, %u20pg: tensor<256xf32>, %u20pbt: tensor<256xf32>, %u21qW: tensor<256x1x5x5xf32>, %u21qg: tensor<256xf32>, %u21qbt: tensor<256xf32>, %u21eW: tensor<512x256x1x1xf32>, %u21eg: tensor<512xf32>, %u21ebt: tensor<512xf32>, %u21pW: tensor<256x512x1x1xf32>, %u21pg: tensor<256xf32>, %u21pbt: tensor<256xf32>, %h1W: tensor<960x256x1x1xf32>, %h1g: tensor<960xf32>, %h1bt: tensor<960xf32>, %hW: tensor<1280x960x1x1xf32>, %hg: tensor<1280xf32>, %hbt: tensor<1280xf32>, %Wd: tensor<1280x10xf32>, %bd: tensor<10xf32>) -> tensor<32x10xf32> {
    // ── MobileNetV4-Conv-M forward: every line is pretty(verified AST node) ──
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
    %zb320 = stablehlo.constant dense<0.0> : tensor<320xf32>
    %zb480 = stablehlo.constant dense<0.0> : tensor<480xf32>
    %zb512 = stablehlo.constant dense<0.0> : tensor<512xf32>
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
    %v6 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v8 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v9 = stablehlo.reduce(%v5 init: %v6) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v10 = stablehlo.broadcast_in_dim %v9, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v11 = stablehlo.divide %v10, %v7 : tensor<32x32x112x112xf32>
    %v12 = stablehlo.subtract %v5, %v11 : tensor<32x32x112x112xf32>
    %v13 = stablehlo.multiply %v12, %v12 : tensor<32x32x112x112xf32>
    %v14 = stablehlo.reduce(%v13 init: %v6) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v15 = stablehlo.broadcast_in_dim %v14, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v16 = stablehlo.divide %v15, %v7 : tensor<32x32x112x112xf32>
    %v17 = stablehlo.add %v16, %v8 : tensor<32x32x112x112xf32>
    %v18 = stablehlo.rsqrt %v17 : tensor<32x32x112x112xf32>
    %v19 = stablehlo.multiply %v12, %v18 : tensor<32x32x112x112xf32>
    %v20 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v21 = stablehlo.broadcast_in_dim %sbt, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v22 = stablehlo.multiply %v19, %v20 : tensor<32x32x112x112xf32>
    %v23 = stablehlo.add %v22, %v21 : tensor<32x32x112x112xf32>
    %v24 = stablehlo.reshape %v23 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v25 = stablehlo.reshape %v24 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v26 = stablehlo.constant dense<0.0> : tensor<32x32x112x112xf32>
    %v27 = stablehlo.maximum %v25, %v26 : tensor<32x32x112x112xf32>
    %v28 = stablehlo.reshape %v27 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v29 = stablehlo.reshape %v28 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v30 = stablehlo.convolution(%v29, %f0cW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<128x32x3x3xf32>) -> tensor<32x128x56x56xf32>
    %v31 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v32 = stablehlo.add %v30, %v31 : tensor<32x128x56x56xf32>
    %v33 = stablehlo.reshape %v32 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v34 = stablehlo.reshape %v33 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v35 = stablehlo.constant dense<0.0> : tensor<f32>
    %v36 = stablehlo.constant dense<100352.0> : tensor<32x128x56x56xf32>
    %v37 = stablehlo.constant dense<1.0e-5> : tensor<32x128x56x56xf32>
    %v38 = stablehlo.reduce(%v34 init: %v35) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v39 = stablehlo.broadcast_in_dim %v38, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v40 = stablehlo.divide %v39, %v36 : tensor<32x128x56x56xf32>
    %v41 = stablehlo.subtract %v34, %v40 : tensor<32x128x56x56xf32>
    %v42 = stablehlo.multiply %v41, %v41 : tensor<32x128x56x56xf32>
    %v43 = stablehlo.reduce(%v42 init: %v35) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v44 = stablehlo.broadcast_in_dim %v43, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v45 = stablehlo.divide %v44, %v36 : tensor<32x128x56x56xf32>
    %v46 = stablehlo.add %v45, %v37 : tensor<32x128x56x56xf32>
    %v47 = stablehlo.rsqrt %v46 : tensor<32x128x56x56xf32>
    %v48 = stablehlo.multiply %v41, %v47 : tensor<32x128x56x56xf32>
    %v49 = stablehlo.broadcast_in_dim %f0cg, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v50 = stablehlo.broadcast_in_dim %f0cbt, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v51 = stablehlo.multiply %v48, %v49 : tensor<32x128x56x56xf32>
    %v52 = stablehlo.add %v51, %v50 : tensor<32x128x56x56xf32>
    %v53 = stablehlo.reshape %v52 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v54 = stablehlo.reshape %v53 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v55 = stablehlo.logistic %v54 : tensor<32x32x112x112xf32>
    %v56 = stablehlo.multiply %v54, %v55 : tensor<32x32x112x112xf32>
    %v57 = stablehlo.reshape %v56 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v59 = stablehlo.convolution(%v58, %f0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x56x56xf32>, tensor<48x128x1x1xf32>) -> tensor<32x48x56x56xf32>
    %v60 = stablehlo.broadcast_in_dim %zb48, dims = [1] : (tensor<48xf32>) -> tensor<32x48x56x56xf32>
    %v61 = stablehlo.add %v59, %v60 : tensor<32x48x56x56xf32>
    %v62 = stablehlo.reshape %v61 : (tensor<32x48x56x56xf32>) -> tensor<32x150528xf32>
    %v63 = stablehlo.reshape %v62 : (tensor<32x150528xf32>) -> tensor<32x48x56x56xf32>
    %v64 = stablehlo.constant dense<0.0> : tensor<f32>
    %v65 = stablehlo.constant dense<100352.0> : tensor<32x48x56x56xf32>
    %v66 = stablehlo.constant dense<1.0e-5> : tensor<32x48x56x56xf32>
    %v67 = stablehlo.reduce(%v63 init: %v64) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x48x56x56xf32>, tensor<f32>) -> tensor<48xf32>
    %v68 = stablehlo.broadcast_in_dim %v67, dims = [1] : (tensor<48xf32>) -> tensor<32x48x56x56xf32>
    %v69 = stablehlo.divide %v68, %v65 : tensor<32x48x56x56xf32>
    %v70 = stablehlo.subtract %v63, %v69 : tensor<32x48x56x56xf32>
    %v71 = stablehlo.multiply %v70, %v70 : tensor<32x48x56x56xf32>
    %v72 = stablehlo.reduce(%v71 init: %v64) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x48x56x56xf32>, tensor<f32>) -> tensor<48xf32>
    %v73 = stablehlo.broadcast_in_dim %v72, dims = [1] : (tensor<48xf32>) -> tensor<32x48x56x56xf32>
    %v74 = stablehlo.divide %v73, %v65 : tensor<32x48x56x56xf32>
    %v75 = stablehlo.add %v74, %v66 : tensor<32x48x56x56xf32>
    %v76 = stablehlo.rsqrt %v75 : tensor<32x48x56x56xf32>
    %v77 = stablehlo.multiply %v70, %v76 : tensor<32x48x56x56xf32>
    %v78 = stablehlo.broadcast_in_dim %f0pg, dims = [1] : (tensor<48xf32>) -> tensor<32x48x56x56xf32>
    %v79 = stablehlo.broadcast_in_dim %f0pbt, dims = [1] : (tensor<48xf32>) -> tensor<32x48x56x56xf32>
    %v80 = stablehlo.multiply %v77, %v78 : tensor<32x48x56x56xf32>
    %v81 = stablehlo.add %v80, %v79 : tensor<32x48x56x56xf32>
    %v82 = stablehlo.reshape %v81 : (tensor<32x48x56x56xf32>) -> tensor<32x150528xf32>
    %v83 = stablehlo.reshape %v82 : (tensor<32x150528xf32>) -> tensor<32x48x56x56xf32>
    %v84 = stablehlo.convolution(%v83, %u1qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 48 : i64} : (tensor<32x48x56x56xf32>, tensor<48x1x3x3xf32>) -> tensor<32x48x28x28xf32>
    %v85 = stablehlo.broadcast_in_dim %zb48, dims = [1] : (tensor<48xf32>) -> tensor<32x48x28x28xf32>
    %v86 = stablehlo.add %v84, %v85 : tensor<32x48x28x28xf32>
    %v87 = stablehlo.reshape %v86 : (tensor<32x48x28x28xf32>) -> tensor<32x37632xf32>
    %v88 = stablehlo.reshape %v87 : (tensor<32x37632xf32>) -> tensor<32x48x28x28xf32>
    %v89 = stablehlo.constant dense<0.0> : tensor<f32>
    %v90 = stablehlo.constant dense<25088.0> : tensor<32x48x28x28xf32>
    %v91 = stablehlo.constant dense<1.0e-5> : tensor<32x48x28x28xf32>
    %v92 = stablehlo.reduce(%v88 init: %v89) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x48x28x28xf32>, tensor<f32>) -> tensor<48xf32>
    %v93 = stablehlo.broadcast_in_dim %v92, dims = [1] : (tensor<48xf32>) -> tensor<32x48x28x28xf32>
    %v94 = stablehlo.divide %v93, %v90 : tensor<32x48x28x28xf32>
    %v95 = stablehlo.subtract %v88, %v94 : tensor<32x48x28x28xf32>
    %v96 = stablehlo.multiply %v95, %v95 : tensor<32x48x28x28xf32>
    %v97 = stablehlo.reduce(%v96 init: %v89) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x48x28x28xf32>, tensor<f32>) -> tensor<48xf32>
    %v98 = stablehlo.broadcast_in_dim %v97, dims = [1] : (tensor<48xf32>) -> tensor<32x48x28x28xf32>
    %v99 = stablehlo.divide %v98, %v90 : tensor<32x48x28x28xf32>
    %v100 = stablehlo.add %v99, %v91 : tensor<32x48x28x28xf32>
    %v101 = stablehlo.rsqrt %v100 : tensor<32x48x28x28xf32>
    %v102 = stablehlo.multiply %v95, %v101 : tensor<32x48x28x28xf32>
    %v103 = stablehlo.broadcast_in_dim %u1qg, dims = [1] : (tensor<48xf32>) -> tensor<32x48x28x28xf32>
    %v104 = stablehlo.broadcast_in_dim %u1qbt, dims = [1] : (tensor<48xf32>) -> tensor<32x48x28x28xf32>
    %v105 = stablehlo.multiply %v102, %v103 : tensor<32x48x28x28xf32>
    %v106 = stablehlo.add %v105, %v104 : tensor<32x48x28x28xf32>
    %v107 = stablehlo.reshape %v106 : (tensor<32x48x28x28xf32>) -> tensor<32x37632xf32>
    %v108 = stablehlo.reshape %v107 : (tensor<32x37632xf32>) -> tensor<32x48x28x28xf32>
    %v109 = stablehlo.constant dense<0.0> : tensor<32x48x28x28xf32>
    %v110 = stablehlo.maximum %v108, %v109 : tensor<32x48x28x28xf32>
    %v111 = stablehlo.reshape %v110 : (tensor<32x48x28x28xf32>) -> tensor<32x37632xf32>
    %v112 = stablehlo.reshape %v111 : (tensor<32x37632xf32>) -> tensor<32x48x28x28xf32>
    %v113 = stablehlo.convolution(%v112, %u1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x48x28x28xf32>, tensor<192x48x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v114 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v115 = stablehlo.add %v113, %v114 : tensor<32x192x28x28xf32>
    %v116 = stablehlo.reshape %v115 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v117 = stablehlo.reshape %v116 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v118 = stablehlo.constant dense<0.0> : tensor<f32>
    %v119 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v120 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v121 = stablehlo.reduce(%v117 init: %v118) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v122 = stablehlo.broadcast_in_dim %v121, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v123 = stablehlo.divide %v122, %v119 : tensor<32x192x28x28xf32>
    %v124 = stablehlo.subtract %v117, %v123 : tensor<32x192x28x28xf32>
    %v125 = stablehlo.multiply %v124, %v124 : tensor<32x192x28x28xf32>
    %v126 = stablehlo.reduce(%v125 init: %v118) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v127 = stablehlo.broadcast_in_dim %v126, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v128 = stablehlo.divide %v127, %v119 : tensor<32x192x28x28xf32>
    %v129 = stablehlo.add %v128, %v120 : tensor<32x192x28x28xf32>
    %v130 = stablehlo.rsqrt %v129 : tensor<32x192x28x28xf32>
    %v131 = stablehlo.multiply %v124, %v130 : tensor<32x192x28x28xf32>
    %v132 = stablehlo.broadcast_in_dim %u1eg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v133 = stablehlo.broadcast_in_dim %u1ebt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v134 = stablehlo.multiply %v131, %v132 : tensor<32x192x28x28xf32>
    %v135 = stablehlo.add %v134, %v133 : tensor<32x192x28x28xf32>
    %v136 = stablehlo.reshape %v135 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v137 = stablehlo.reshape %v136 : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v138 = stablehlo.constant dense<0.0> : tensor<32x3x224x224xf32>
    %v139 = stablehlo.maximum %v137, %v138 : tensor<32x3x224x224xf32>
    %v140 = stablehlo.reshape %v139 : (tensor<32x3x224x224xf32>) -> tensor<32x150528xf32>
    %v141 = stablehlo.reshape %v140 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v142 = stablehlo.convolution(%v141, %u1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x5x5xf32>) -> tensor<32x192x28x28xf32>
    %v143 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v144 = stablehlo.add %v142, %v143 : tensor<32x192x28x28xf32>
    %v145 = stablehlo.reshape %v144 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v146 = stablehlo.reshape %v145 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v147 = stablehlo.constant dense<0.0> : tensor<f32>
    %v148 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v149 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v150 = stablehlo.reduce(%v146 init: %v147) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v151 = stablehlo.broadcast_in_dim %v150, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v152 = stablehlo.divide %v151, %v148 : tensor<32x192x28x28xf32>
    %v153 = stablehlo.subtract %v146, %v152 : tensor<32x192x28x28xf32>
    %v154 = stablehlo.multiply %v153, %v153 : tensor<32x192x28x28xf32>
    %v155 = stablehlo.reduce(%v154 init: %v147) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v156 = stablehlo.broadcast_in_dim %v155, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v157 = stablehlo.divide %v156, %v148 : tensor<32x192x28x28xf32>
    %v158 = stablehlo.add %v157, %v149 : tensor<32x192x28x28xf32>
    %v159 = stablehlo.rsqrt %v158 : tensor<32x192x28x28xf32>
    %v160 = stablehlo.multiply %v153, %v159 : tensor<32x192x28x28xf32>
    %v161 = stablehlo.broadcast_in_dim %u1dg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v162 = stablehlo.broadcast_in_dim %u1dbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v163 = stablehlo.multiply %v160, %v161 : tensor<32x192x28x28xf32>
    %v164 = stablehlo.add %v163, %v162 : tensor<32x192x28x28xf32>
    %v165 = stablehlo.reshape %v164 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v166 = stablehlo.reshape %v165 : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v167 = stablehlo.constant dense<0.0> : tensor<32x3x224x224xf32>
    %v168 = stablehlo.maximum %v166, %v167 : tensor<32x3x224x224xf32>
    %v169 = stablehlo.reshape %v168 : (tensor<32x3x224x224xf32>) -> tensor<32x150528xf32>
    %v170 = stablehlo.reshape %v169 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v171 = stablehlo.convolution(%v170, %u1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<80x192x1x1xf32>) -> tensor<32x80x28x28xf32>
    %v172 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v173 = stablehlo.add %v171, %v172 : tensor<32x80x28x28xf32>
    %v174 = stablehlo.reshape %v173 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v175 = stablehlo.reshape %v174 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v176 = stablehlo.constant dense<0.0> : tensor<f32>
    %v177 = stablehlo.constant dense<25088.0> : tensor<32x80x28x28xf32>
    %v178 = stablehlo.constant dense<1.0e-5> : tensor<32x80x28x28xf32>
    %v179 = stablehlo.reduce(%v175 init: %v176) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x28x28xf32>, tensor<f32>) -> tensor<80xf32>
    %v180 = stablehlo.broadcast_in_dim %v179, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v181 = stablehlo.divide %v180, %v177 : tensor<32x80x28x28xf32>
    %v182 = stablehlo.subtract %v175, %v181 : tensor<32x80x28x28xf32>
    %v183 = stablehlo.multiply %v182, %v182 : tensor<32x80x28x28xf32>
    %v184 = stablehlo.reduce(%v183 init: %v176) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x28x28xf32>, tensor<f32>) -> tensor<80xf32>
    %v185 = stablehlo.broadcast_in_dim %v184, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v186 = stablehlo.divide %v185, %v177 : tensor<32x80x28x28xf32>
    %v187 = stablehlo.add %v186, %v178 : tensor<32x80x28x28xf32>
    %v188 = stablehlo.rsqrt %v187 : tensor<32x80x28x28xf32>
    %v189 = stablehlo.multiply %v182, %v188 : tensor<32x80x28x28xf32>
    %v190 = stablehlo.broadcast_in_dim %u1pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v191 = stablehlo.broadcast_in_dim %u1pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v192 = stablehlo.multiply %v189, %v190 : tensor<32x80x28x28xf32>
    %v193 = stablehlo.add %v192, %v191 : tensor<32x80x28x28xf32>
    %v194 = stablehlo.reshape %v193 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v195 = stablehlo.reshape %v194 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v196 = stablehlo.convolution(%v195, %u2qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 80 : i64} : (tensor<32x80x28x28xf32>, tensor<80x1x3x3xf32>) -> tensor<32x80x28x28xf32>
    %v197 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v198 = stablehlo.add %v196, %v197 : tensor<32x80x28x28xf32>
    %v199 = stablehlo.reshape %v198 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v200 = stablehlo.reshape %v199 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v201 = stablehlo.constant dense<0.0> : tensor<f32>
    %v202 = stablehlo.constant dense<25088.0> : tensor<32x80x28x28xf32>
    %v203 = stablehlo.constant dense<1.0e-5> : tensor<32x80x28x28xf32>
    %v204 = stablehlo.reduce(%v200 init: %v201) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x28x28xf32>, tensor<f32>) -> tensor<80xf32>
    %v205 = stablehlo.broadcast_in_dim %v204, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v206 = stablehlo.divide %v205, %v202 : tensor<32x80x28x28xf32>
    %v207 = stablehlo.subtract %v200, %v206 : tensor<32x80x28x28xf32>
    %v208 = stablehlo.multiply %v207, %v207 : tensor<32x80x28x28xf32>
    %v209 = stablehlo.reduce(%v208 init: %v201) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x28x28xf32>, tensor<f32>) -> tensor<80xf32>
    %v210 = stablehlo.broadcast_in_dim %v209, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v211 = stablehlo.divide %v210, %v202 : tensor<32x80x28x28xf32>
    %v212 = stablehlo.add %v211, %v203 : tensor<32x80x28x28xf32>
    %v213 = stablehlo.rsqrt %v212 : tensor<32x80x28x28xf32>
    %v214 = stablehlo.multiply %v207, %v213 : tensor<32x80x28x28xf32>
    %v215 = stablehlo.broadcast_in_dim %u2qg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v216 = stablehlo.broadcast_in_dim %u2qbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v217 = stablehlo.multiply %v214, %v215 : tensor<32x80x28x28xf32>
    %v218 = stablehlo.add %v217, %v216 : tensor<32x80x28x28xf32>
    %v219 = stablehlo.reshape %v218 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v220 = stablehlo.reshape %v219 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v221 = stablehlo.constant dense<0.0> : tensor<32x80x28x28xf32>
    %v222 = stablehlo.maximum %v220, %v221 : tensor<32x80x28x28xf32>
    %v223 = stablehlo.reshape %v222 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v224 = stablehlo.reshape %v223 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v225 = stablehlo.convolution(%v224, %u2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x28x28xf32>, tensor<160x80x1x1xf32>) -> tensor<32x160x28x28xf32>
    %v226 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v227 = stablehlo.add %v225, %v226 : tensor<32x160x28x28xf32>
    %v228 = stablehlo.reshape %v227 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v229 = stablehlo.reshape %v228 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v230 = stablehlo.constant dense<0.0> : tensor<f32>
    %v231 = stablehlo.constant dense<25088.0> : tensor<32x160x28x28xf32>
    %v232 = stablehlo.constant dense<1.0e-5> : tensor<32x160x28x28xf32>
    %v233 = stablehlo.reduce(%v229 init: %v230) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x28x28xf32>, tensor<f32>) -> tensor<160xf32>
    %v234 = stablehlo.broadcast_in_dim %v233, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v235 = stablehlo.divide %v234, %v231 : tensor<32x160x28x28xf32>
    %v236 = stablehlo.subtract %v229, %v235 : tensor<32x160x28x28xf32>
    %v237 = stablehlo.multiply %v236, %v236 : tensor<32x160x28x28xf32>
    %v238 = stablehlo.reduce(%v237 init: %v230) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x28x28xf32>, tensor<f32>) -> tensor<160xf32>
    %v239 = stablehlo.broadcast_in_dim %v238, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v240 = stablehlo.divide %v239, %v231 : tensor<32x160x28x28xf32>
    %v241 = stablehlo.add %v240, %v232 : tensor<32x160x28x28xf32>
    %v242 = stablehlo.rsqrt %v241 : tensor<32x160x28x28xf32>
    %v243 = stablehlo.multiply %v236, %v242 : tensor<32x160x28x28xf32>
    %v244 = stablehlo.broadcast_in_dim %u2eg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v245 = stablehlo.broadcast_in_dim %u2ebt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v246 = stablehlo.multiply %v243, %v244 : tensor<32x160x28x28xf32>
    %v247 = stablehlo.add %v246, %v245 : tensor<32x160x28x28xf32>
    %v248 = stablehlo.reshape %v247 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v249 = stablehlo.reshape %v248 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v250 = stablehlo.constant dense<0.0> : tensor<32x160x28x28xf32>
    %v251 = stablehlo.maximum %v249, %v250 : tensor<32x160x28x28xf32>
    %v252 = stablehlo.reshape %v251 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v253 = stablehlo.reshape %v252 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v254 = stablehlo.convolution(%v253, %u2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x28x28xf32>, tensor<160x1x3x3xf32>) -> tensor<32x160x28x28xf32>
    %v255 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v256 = stablehlo.add %v254, %v255 : tensor<32x160x28x28xf32>
    %v257 = stablehlo.reshape %v256 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v258 = stablehlo.reshape %v257 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v259 = stablehlo.constant dense<0.0> : tensor<f32>
    %v260 = stablehlo.constant dense<25088.0> : tensor<32x160x28x28xf32>
    %v261 = stablehlo.constant dense<1.0e-5> : tensor<32x160x28x28xf32>
    %v262 = stablehlo.reduce(%v258 init: %v259) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x28x28xf32>, tensor<f32>) -> tensor<160xf32>
    %v263 = stablehlo.broadcast_in_dim %v262, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v264 = stablehlo.divide %v263, %v260 : tensor<32x160x28x28xf32>
    %v265 = stablehlo.subtract %v258, %v264 : tensor<32x160x28x28xf32>
    %v266 = stablehlo.multiply %v265, %v265 : tensor<32x160x28x28xf32>
    %v267 = stablehlo.reduce(%v266 init: %v259) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x28x28xf32>, tensor<f32>) -> tensor<160xf32>
    %v268 = stablehlo.broadcast_in_dim %v267, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v269 = stablehlo.divide %v268, %v260 : tensor<32x160x28x28xf32>
    %v270 = stablehlo.add %v269, %v261 : tensor<32x160x28x28xf32>
    %v271 = stablehlo.rsqrt %v270 : tensor<32x160x28x28xf32>
    %v272 = stablehlo.multiply %v265, %v271 : tensor<32x160x28x28xf32>
    %v273 = stablehlo.broadcast_in_dim %u2dg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v274 = stablehlo.broadcast_in_dim %u2dbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v275 = stablehlo.multiply %v272, %v273 : tensor<32x160x28x28xf32>
    %v276 = stablehlo.add %v275, %v274 : tensor<32x160x28x28xf32>
    %v277 = stablehlo.reshape %v276 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v278 = stablehlo.reshape %v277 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v279 = stablehlo.constant dense<0.0> : tensor<32x160x28x28xf32>
    %v280 = stablehlo.maximum %v278, %v279 : tensor<32x160x28x28xf32>
    %v281 = stablehlo.reshape %v280 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v282 = stablehlo.reshape %v281 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v283 = stablehlo.convolution(%v282, %u2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x28x28xf32>, tensor<80x160x1x1xf32>) -> tensor<32x80x28x28xf32>
    %v284 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v285 = stablehlo.add %v283, %v284 : tensor<32x80x28x28xf32>
    %v286 = stablehlo.reshape %v285 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v287 = stablehlo.reshape %v286 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v288 = stablehlo.constant dense<0.0> : tensor<f32>
    %v289 = stablehlo.constant dense<25088.0> : tensor<32x80x28x28xf32>
    %v290 = stablehlo.constant dense<1.0e-5> : tensor<32x80x28x28xf32>
    %v291 = stablehlo.reduce(%v287 init: %v288) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x28x28xf32>, tensor<f32>) -> tensor<80xf32>
    %v292 = stablehlo.broadcast_in_dim %v291, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v293 = stablehlo.divide %v292, %v289 : tensor<32x80x28x28xf32>
    %v294 = stablehlo.subtract %v287, %v293 : tensor<32x80x28x28xf32>
    %v295 = stablehlo.multiply %v294, %v294 : tensor<32x80x28x28xf32>
    %v296 = stablehlo.reduce(%v295 init: %v288) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x28x28xf32>, tensor<f32>) -> tensor<80xf32>
    %v297 = stablehlo.broadcast_in_dim %v296, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v298 = stablehlo.divide %v297, %v289 : tensor<32x80x28x28xf32>
    %v299 = stablehlo.add %v298, %v290 : tensor<32x80x28x28xf32>
    %v300 = stablehlo.rsqrt %v299 : tensor<32x80x28x28xf32>
    %v301 = stablehlo.multiply %v294, %v300 : tensor<32x80x28x28xf32>
    %v302 = stablehlo.broadcast_in_dim %u2pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v303 = stablehlo.broadcast_in_dim %u2pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v304 = stablehlo.multiply %v301, %v302 : tensor<32x80x28x28xf32>
    %v305 = stablehlo.add %v304, %v303 : tensor<32x80x28x28xf32>
    %v306 = stablehlo.reshape %v305 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v307 = stablehlo.reshape %v306 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v308 = stablehlo.reshape %v194 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v309 = stablehlo.add %v307, %v308 : tensor<32x80x28x28xf32>
    %v310 = stablehlo.reshape %v309 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v311 = stablehlo.reshape %v310 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v312 = stablehlo.convolution(%v311, %u3qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 80 : i64} : (tensor<32x80x28x28xf32>, tensor<80x1x3x3xf32>) -> tensor<32x80x14x14xf32>
    %v313 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v314 = stablehlo.add %v312, %v313 : tensor<32x80x14x14xf32>
    %v315 = stablehlo.reshape %v314 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v316 = stablehlo.reshape %v315 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v317 = stablehlo.constant dense<0.0> : tensor<f32>
    %v318 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v319 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v320 = stablehlo.reduce(%v316 init: %v317) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v321 = stablehlo.broadcast_in_dim %v320, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v322 = stablehlo.divide %v321, %v318 : tensor<32x80x14x14xf32>
    %v323 = stablehlo.subtract %v316, %v322 : tensor<32x80x14x14xf32>
    %v324 = stablehlo.multiply %v323, %v323 : tensor<32x80x14x14xf32>
    %v325 = stablehlo.reduce(%v324 init: %v317) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v326 = stablehlo.broadcast_in_dim %v325, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v327 = stablehlo.divide %v326, %v318 : tensor<32x80x14x14xf32>
    %v328 = stablehlo.add %v327, %v319 : tensor<32x80x14x14xf32>
    %v329 = stablehlo.rsqrt %v328 : tensor<32x80x14x14xf32>
    %v330 = stablehlo.multiply %v323, %v329 : tensor<32x80x14x14xf32>
    %v331 = stablehlo.broadcast_in_dim %u3qg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v332 = stablehlo.broadcast_in_dim %u3qbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v333 = stablehlo.multiply %v330, %v331 : tensor<32x80x14x14xf32>
    %v334 = stablehlo.add %v333, %v332 : tensor<32x80x14x14xf32>
    %v335 = stablehlo.reshape %v334 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v336 = stablehlo.reshape %v335 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v337 = stablehlo.constant dense<0.0> : tensor<32x80x14x14xf32>
    %v338 = stablehlo.maximum %v336, %v337 : tensor<32x80x14x14xf32>
    %v339 = stablehlo.reshape %v338 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v340 = stablehlo.reshape %v339 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v341 = stablehlo.convolution(%v340, %u3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v342 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v343 = stablehlo.add %v341, %v342 : tensor<32x480x14x14xf32>
    %v344 = stablehlo.reshape %v343 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v345 = stablehlo.reshape %v344 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v346 = stablehlo.constant dense<0.0> : tensor<f32>
    %v347 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v348 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v349 = stablehlo.reduce(%v345 init: %v346) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v350 = stablehlo.broadcast_in_dim %v349, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v351 = stablehlo.divide %v350, %v347 : tensor<32x480x14x14xf32>
    %v352 = stablehlo.subtract %v345, %v351 : tensor<32x480x14x14xf32>
    %v353 = stablehlo.multiply %v352, %v352 : tensor<32x480x14x14xf32>
    %v354 = stablehlo.reduce(%v353 init: %v346) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v355 = stablehlo.broadcast_in_dim %v354, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v356 = stablehlo.divide %v355, %v347 : tensor<32x480x14x14xf32>
    %v357 = stablehlo.add %v356, %v348 : tensor<32x480x14x14xf32>
    %v358 = stablehlo.rsqrt %v357 : tensor<32x480x14x14xf32>
    %v359 = stablehlo.multiply %v352, %v358 : tensor<32x480x14x14xf32>
    %v360 = stablehlo.broadcast_in_dim %u3eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v361 = stablehlo.broadcast_in_dim %u3ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v362 = stablehlo.multiply %v359, %v360 : tensor<32x480x14x14xf32>
    %v363 = stablehlo.add %v362, %v361 : tensor<32x480x14x14xf32>
    %v364 = stablehlo.reshape %v363 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v365 = stablehlo.reshape %v364 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v366 = stablehlo.constant dense<0.0> : tensor<32x480x14x14xf32>
    %v367 = stablehlo.maximum %v365, %v366 : tensor<32x480x14x14xf32>
    %v368 = stablehlo.reshape %v367 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v369 = stablehlo.reshape %v368 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v370 = stablehlo.convolution(%v369, %u3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x5x5xf32>) -> tensor<32x480x14x14xf32>
    %v371 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v372 = stablehlo.add %v370, %v371 : tensor<32x480x14x14xf32>
    %v373 = stablehlo.reshape %v372 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v374 = stablehlo.reshape %v373 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v375 = stablehlo.constant dense<0.0> : tensor<f32>
    %v376 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v377 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v378 = stablehlo.reduce(%v374 init: %v375) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v379 = stablehlo.broadcast_in_dim %v378, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v380 = stablehlo.divide %v379, %v376 : tensor<32x480x14x14xf32>
    %v381 = stablehlo.subtract %v374, %v380 : tensor<32x480x14x14xf32>
    %v382 = stablehlo.multiply %v381, %v381 : tensor<32x480x14x14xf32>
    %v383 = stablehlo.reduce(%v382 init: %v375) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v384 = stablehlo.broadcast_in_dim %v383, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v385 = stablehlo.divide %v384, %v376 : tensor<32x480x14x14xf32>
    %v386 = stablehlo.add %v385, %v377 : tensor<32x480x14x14xf32>
    %v387 = stablehlo.rsqrt %v386 : tensor<32x480x14x14xf32>
    %v388 = stablehlo.multiply %v381, %v387 : tensor<32x480x14x14xf32>
    %v389 = stablehlo.broadcast_in_dim %u3dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v390 = stablehlo.broadcast_in_dim %u3dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v391 = stablehlo.multiply %v388, %v389 : tensor<32x480x14x14xf32>
    %v392 = stablehlo.add %v391, %v390 : tensor<32x480x14x14xf32>
    %v393 = stablehlo.reshape %v392 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v394 = stablehlo.reshape %v393 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v395 = stablehlo.constant dense<0.0> : tensor<32x480x14x14xf32>
    %v396 = stablehlo.maximum %v394, %v395 : tensor<32x480x14x14xf32>
    %v397 = stablehlo.reshape %v396 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v398 = stablehlo.reshape %v397 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v399 = stablehlo.convolution(%v398, %u3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<160x480x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v400 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v401 = stablehlo.add %v399, %v400 : tensor<32x160x14x14xf32>
    %v402 = stablehlo.reshape %v401 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v403 = stablehlo.reshape %v402 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v404 = stablehlo.constant dense<0.0> : tensor<f32>
    %v405 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v406 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v407 = stablehlo.reduce(%v403 init: %v404) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v408 = stablehlo.broadcast_in_dim %v407, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v409 = stablehlo.divide %v408, %v405 : tensor<32x160x14x14xf32>
    %v410 = stablehlo.subtract %v403, %v409 : tensor<32x160x14x14xf32>
    %v411 = stablehlo.multiply %v410, %v410 : tensor<32x160x14x14xf32>
    %v412 = stablehlo.reduce(%v411 init: %v404) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v413 = stablehlo.broadcast_in_dim %v412, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v414 = stablehlo.divide %v413, %v405 : tensor<32x160x14x14xf32>
    %v415 = stablehlo.add %v414, %v406 : tensor<32x160x14x14xf32>
    %v416 = stablehlo.rsqrt %v415 : tensor<32x160x14x14xf32>
    %v417 = stablehlo.multiply %v410, %v416 : tensor<32x160x14x14xf32>
    %v418 = stablehlo.broadcast_in_dim %u3pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v419 = stablehlo.broadcast_in_dim %u3pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v420 = stablehlo.multiply %v417, %v418 : tensor<32x160x14x14xf32>
    %v421 = stablehlo.add %v420, %v419 : tensor<32x160x14x14xf32>
    %v422 = stablehlo.reshape %v421 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v423 = stablehlo.reshape %v422 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v424 = stablehlo.convolution(%v423, %u4qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<32x160x14x14xf32>
    %v425 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v426 = stablehlo.add %v424, %v425 : tensor<32x160x14x14xf32>
    %v427 = stablehlo.reshape %v426 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v428 = stablehlo.reshape %v427 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v429 = stablehlo.constant dense<0.0> : tensor<f32>
    %v430 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v431 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v432 = stablehlo.reduce(%v428 init: %v429) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v433 = stablehlo.broadcast_in_dim %v432, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v434 = stablehlo.divide %v433, %v430 : tensor<32x160x14x14xf32>
    %v435 = stablehlo.subtract %v428, %v434 : tensor<32x160x14x14xf32>
    %v436 = stablehlo.multiply %v435, %v435 : tensor<32x160x14x14xf32>
    %v437 = stablehlo.reduce(%v436 init: %v429) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v438 = stablehlo.broadcast_in_dim %v437, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v439 = stablehlo.divide %v438, %v430 : tensor<32x160x14x14xf32>
    %v440 = stablehlo.add %v439, %v431 : tensor<32x160x14x14xf32>
    %v441 = stablehlo.rsqrt %v440 : tensor<32x160x14x14xf32>
    %v442 = stablehlo.multiply %v435, %v441 : tensor<32x160x14x14xf32>
    %v443 = stablehlo.broadcast_in_dim %u4qg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v444 = stablehlo.broadcast_in_dim %u4qbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v445 = stablehlo.multiply %v442, %v443 : tensor<32x160x14x14xf32>
    %v446 = stablehlo.add %v445, %v444 : tensor<32x160x14x14xf32>
    %v447 = stablehlo.reshape %v446 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v448 = stablehlo.reshape %v447 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v449 = stablehlo.constant dense<0.0> : tensor<32x160x14x14xf32>
    %v450 = stablehlo.maximum %v448, %v449 : tensor<32x160x14x14xf32>
    %v451 = stablehlo.reshape %v450 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v452 = stablehlo.reshape %v451 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v453 = stablehlo.convolution(%v452, %u4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<32x640x14x14xf32>
    %v454 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v455 = stablehlo.add %v453, %v454 : tensor<32x640x14x14xf32>
    %v456 = stablehlo.reshape %v455 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v457 = stablehlo.reshape %v456 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v458 = stablehlo.constant dense<0.0> : tensor<f32>
    %v459 = stablehlo.constant dense<6272.0> : tensor<32x640x14x14xf32>
    %v460 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v461 = stablehlo.reduce(%v457 init: %v458) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v462 = stablehlo.broadcast_in_dim %v461, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v463 = stablehlo.divide %v462, %v459 : tensor<32x640x14x14xf32>
    %v464 = stablehlo.subtract %v457, %v463 : tensor<32x640x14x14xf32>
    %v465 = stablehlo.multiply %v464, %v464 : tensor<32x640x14x14xf32>
    %v466 = stablehlo.reduce(%v465 init: %v458) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v467 = stablehlo.broadcast_in_dim %v466, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v468 = stablehlo.divide %v467, %v459 : tensor<32x640x14x14xf32>
    %v469 = stablehlo.add %v468, %v460 : tensor<32x640x14x14xf32>
    %v470 = stablehlo.rsqrt %v469 : tensor<32x640x14x14xf32>
    %v471 = stablehlo.multiply %v464, %v470 : tensor<32x640x14x14xf32>
    %v472 = stablehlo.broadcast_in_dim %u4eg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v473 = stablehlo.broadcast_in_dim %u4ebt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v474 = stablehlo.multiply %v471, %v472 : tensor<32x640x14x14xf32>
    %v475 = stablehlo.add %v474, %v473 : tensor<32x640x14x14xf32>
    %v476 = stablehlo.reshape %v475 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v477 = stablehlo.reshape %v476 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v478 = stablehlo.constant dense<0.0> : tensor<32x160x28x28xf32>
    %v479 = stablehlo.maximum %v477, %v478 : tensor<32x160x28x28xf32>
    %v480 = stablehlo.reshape %v479 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v481 = stablehlo.reshape %v480 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v482 = stablehlo.convolution(%v481, %u4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 640 : i64} : (tensor<32x640x14x14xf32>, tensor<640x1x3x3xf32>) -> tensor<32x640x14x14xf32>
    %v483 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v484 = stablehlo.add %v482, %v483 : tensor<32x640x14x14xf32>
    %v485 = stablehlo.reshape %v484 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v486 = stablehlo.reshape %v485 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v487 = stablehlo.constant dense<0.0> : tensor<f32>
    %v488 = stablehlo.constant dense<6272.0> : tensor<32x640x14x14xf32>
    %v489 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v490 = stablehlo.reduce(%v486 init: %v487) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v491 = stablehlo.broadcast_in_dim %v490, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v492 = stablehlo.divide %v491, %v488 : tensor<32x640x14x14xf32>
    %v493 = stablehlo.subtract %v486, %v492 : tensor<32x640x14x14xf32>
    %v494 = stablehlo.multiply %v493, %v493 : tensor<32x640x14x14xf32>
    %v495 = stablehlo.reduce(%v494 init: %v487) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v496 = stablehlo.broadcast_in_dim %v495, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v497 = stablehlo.divide %v496, %v488 : tensor<32x640x14x14xf32>
    %v498 = stablehlo.add %v497, %v489 : tensor<32x640x14x14xf32>
    %v499 = stablehlo.rsqrt %v498 : tensor<32x640x14x14xf32>
    %v500 = stablehlo.multiply %v493, %v499 : tensor<32x640x14x14xf32>
    %v501 = stablehlo.broadcast_in_dim %u4dg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v502 = stablehlo.broadcast_in_dim %u4dbt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v503 = stablehlo.multiply %v500, %v501 : tensor<32x640x14x14xf32>
    %v504 = stablehlo.add %v503, %v502 : tensor<32x640x14x14xf32>
    %v505 = stablehlo.reshape %v504 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v506 = stablehlo.reshape %v505 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v507 = stablehlo.constant dense<0.0> : tensor<32x160x28x28xf32>
    %v508 = stablehlo.maximum %v506, %v507 : tensor<32x160x28x28xf32>
    %v509 = stablehlo.reshape %v508 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v510 = stablehlo.reshape %v509 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v511 = stablehlo.convolution(%v510, %u4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v512 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v513 = stablehlo.add %v511, %v512 : tensor<32x160x14x14xf32>
    %v514 = stablehlo.reshape %v513 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v515 = stablehlo.reshape %v514 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v516 = stablehlo.constant dense<0.0> : tensor<f32>
    %v517 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v518 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v519 = stablehlo.reduce(%v515 init: %v516) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v520 = stablehlo.broadcast_in_dim %v519, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v521 = stablehlo.divide %v520, %v517 : tensor<32x160x14x14xf32>
    %v522 = stablehlo.subtract %v515, %v521 : tensor<32x160x14x14xf32>
    %v523 = stablehlo.multiply %v522, %v522 : tensor<32x160x14x14xf32>
    %v524 = stablehlo.reduce(%v523 init: %v516) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v525 = stablehlo.broadcast_in_dim %v524, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v526 = stablehlo.divide %v525, %v517 : tensor<32x160x14x14xf32>
    %v527 = stablehlo.add %v526, %v518 : tensor<32x160x14x14xf32>
    %v528 = stablehlo.rsqrt %v527 : tensor<32x160x14x14xf32>
    %v529 = stablehlo.multiply %v522, %v528 : tensor<32x160x14x14xf32>
    %v530 = stablehlo.broadcast_in_dim %u4pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v531 = stablehlo.broadcast_in_dim %u4pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v532 = stablehlo.multiply %v529, %v530 : tensor<32x160x14x14xf32>
    %v533 = stablehlo.add %v532, %v531 : tensor<32x160x14x14xf32>
    %v534 = stablehlo.reshape %v533 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v535 = stablehlo.reshape %v534 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v536 = stablehlo.reshape %v422 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v537 = stablehlo.add %v535, %v536 : tensor<32x160x14x14xf32>
    %v538 = stablehlo.reshape %v537 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v539 = stablehlo.reshape %v538 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v540 = stablehlo.convolution(%v539, %u5qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<32x160x14x14xf32>
    %v541 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v542 = stablehlo.add %v540, %v541 : tensor<32x160x14x14xf32>
    %v543 = stablehlo.reshape %v542 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v544 = stablehlo.reshape %v543 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v545 = stablehlo.constant dense<0.0> : tensor<f32>
    %v546 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v547 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v548 = stablehlo.reduce(%v544 init: %v545) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v549 = stablehlo.broadcast_in_dim %v548, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v550 = stablehlo.divide %v549, %v546 : tensor<32x160x14x14xf32>
    %v551 = stablehlo.subtract %v544, %v550 : tensor<32x160x14x14xf32>
    %v552 = stablehlo.multiply %v551, %v551 : tensor<32x160x14x14xf32>
    %v553 = stablehlo.reduce(%v552 init: %v545) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v554 = stablehlo.broadcast_in_dim %v553, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v555 = stablehlo.divide %v554, %v546 : tensor<32x160x14x14xf32>
    %v556 = stablehlo.add %v555, %v547 : tensor<32x160x14x14xf32>
    %v557 = stablehlo.rsqrt %v556 : tensor<32x160x14x14xf32>
    %v558 = stablehlo.multiply %v551, %v557 : tensor<32x160x14x14xf32>
    %v559 = stablehlo.broadcast_in_dim %u5qg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v560 = stablehlo.broadcast_in_dim %u5qbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v561 = stablehlo.multiply %v558, %v559 : tensor<32x160x14x14xf32>
    %v562 = stablehlo.add %v561, %v560 : tensor<32x160x14x14xf32>
    %v563 = stablehlo.reshape %v562 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v564 = stablehlo.reshape %v563 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v565 = stablehlo.constant dense<0.0> : tensor<32x160x14x14xf32>
    %v566 = stablehlo.maximum %v564, %v565 : tensor<32x160x14x14xf32>
    %v567 = stablehlo.reshape %v566 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v568 = stablehlo.reshape %v567 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v569 = stablehlo.convolution(%v568, %u5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<32x640x14x14xf32>
    %v570 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v571 = stablehlo.add %v569, %v570 : tensor<32x640x14x14xf32>
    %v572 = stablehlo.reshape %v571 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v573 = stablehlo.reshape %v572 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v574 = stablehlo.constant dense<0.0> : tensor<f32>
    %v575 = stablehlo.constant dense<6272.0> : tensor<32x640x14x14xf32>
    %v576 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v577 = stablehlo.reduce(%v573 init: %v574) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v578 = stablehlo.broadcast_in_dim %v577, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v579 = stablehlo.divide %v578, %v575 : tensor<32x640x14x14xf32>
    %v580 = stablehlo.subtract %v573, %v579 : tensor<32x640x14x14xf32>
    %v581 = stablehlo.multiply %v580, %v580 : tensor<32x640x14x14xf32>
    %v582 = stablehlo.reduce(%v581 init: %v574) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v583 = stablehlo.broadcast_in_dim %v582, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v584 = stablehlo.divide %v583, %v575 : tensor<32x640x14x14xf32>
    %v585 = stablehlo.add %v584, %v576 : tensor<32x640x14x14xf32>
    %v586 = stablehlo.rsqrt %v585 : tensor<32x640x14x14xf32>
    %v587 = stablehlo.multiply %v580, %v586 : tensor<32x640x14x14xf32>
    %v588 = stablehlo.broadcast_in_dim %u5eg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v589 = stablehlo.broadcast_in_dim %u5ebt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v590 = stablehlo.multiply %v587, %v588 : tensor<32x640x14x14xf32>
    %v591 = stablehlo.add %v590, %v589 : tensor<32x640x14x14xf32>
    %v592 = stablehlo.reshape %v591 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v593 = stablehlo.reshape %v592 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v594 = stablehlo.constant dense<0.0> : tensor<32x160x28x28xf32>
    %v595 = stablehlo.maximum %v593, %v594 : tensor<32x160x28x28xf32>
    %v596 = stablehlo.reshape %v595 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v597 = stablehlo.reshape %v596 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v598 = stablehlo.convolution(%v597, %u5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 640 : i64} : (tensor<32x640x14x14xf32>, tensor<640x1x3x3xf32>) -> tensor<32x640x14x14xf32>
    %v599 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v600 = stablehlo.add %v598, %v599 : tensor<32x640x14x14xf32>
    %v601 = stablehlo.reshape %v600 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v602 = stablehlo.reshape %v601 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v603 = stablehlo.constant dense<0.0> : tensor<f32>
    %v604 = stablehlo.constant dense<6272.0> : tensor<32x640x14x14xf32>
    %v605 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v606 = stablehlo.reduce(%v602 init: %v603) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v607 = stablehlo.broadcast_in_dim %v606, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v608 = stablehlo.divide %v607, %v604 : tensor<32x640x14x14xf32>
    %v609 = stablehlo.subtract %v602, %v608 : tensor<32x640x14x14xf32>
    %v610 = stablehlo.multiply %v609, %v609 : tensor<32x640x14x14xf32>
    %v611 = stablehlo.reduce(%v610 init: %v603) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v612 = stablehlo.broadcast_in_dim %v611, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v613 = stablehlo.divide %v612, %v604 : tensor<32x640x14x14xf32>
    %v614 = stablehlo.add %v613, %v605 : tensor<32x640x14x14xf32>
    %v615 = stablehlo.rsqrt %v614 : tensor<32x640x14x14xf32>
    %v616 = stablehlo.multiply %v609, %v615 : tensor<32x640x14x14xf32>
    %v617 = stablehlo.broadcast_in_dim %u5dg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v618 = stablehlo.broadcast_in_dim %u5dbt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v619 = stablehlo.multiply %v616, %v617 : tensor<32x640x14x14xf32>
    %v620 = stablehlo.add %v619, %v618 : tensor<32x640x14x14xf32>
    %v621 = stablehlo.reshape %v620 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v622 = stablehlo.reshape %v621 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v623 = stablehlo.constant dense<0.0> : tensor<32x160x28x28xf32>
    %v624 = stablehlo.maximum %v622, %v623 : tensor<32x160x28x28xf32>
    %v625 = stablehlo.reshape %v624 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v626 = stablehlo.reshape %v625 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v627 = stablehlo.convolution(%v626, %u5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v628 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v629 = stablehlo.add %v627, %v628 : tensor<32x160x14x14xf32>
    %v630 = stablehlo.reshape %v629 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v631 = stablehlo.reshape %v630 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v632 = stablehlo.constant dense<0.0> : tensor<f32>
    %v633 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v634 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v635 = stablehlo.reduce(%v631 init: %v632) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v636 = stablehlo.broadcast_in_dim %v635, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v637 = stablehlo.divide %v636, %v633 : tensor<32x160x14x14xf32>
    %v638 = stablehlo.subtract %v631, %v637 : tensor<32x160x14x14xf32>
    %v639 = stablehlo.multiply %v638, %v638 : tensor<32x160x14x14xf32>
    %v640 = stablehlo.reduce(%v639 init: %v632) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v641 = stablehlo.broadcast_in_dim %v640, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v642 = stablehlo.divide %v641, %v633 : tensor<32x160x14x14xf32>
    %v643 = stablehlo.add %v642, %v634 : tensor<32x160x14x14xf32>
    %v644 = stablehlo.rsqrt %v643 : tensor<32x160x14x14xf32>
    %v645 = stablehlo.multiply %v638, %v644 : tensor<32x160x14x14xf32>
    %v646 = stablehlo.broadcast_in_dim %u5pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v647 = stablehlo.broadcast_in_dim %u5pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v648 = stablehlo.multiply %v645, %v646 : tensor<32x160x14x14xf32>
    %v649 = stablehlo.add %v648, %v647 : tensor<32x160x14x14xf32>
    %v650 = stablehlo.reshape %v649 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v651 = stablehlo.reshape %v650 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v652 = stablehlo.reshape %v538 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v653 = stablehlo.add %v651, %v652 : tensor<32x160x14x14xf32>
    %v654 = stablehlo.reshape %v653 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v655 = stablehlo.reshape %v654 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v656 = stablehlo.convolution(%v655, %u6qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<32x160x14x14xf32>
    %v657 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v658 = stablehlo.add %v656, %v657 : tensor<32x160x14x14xf32>
    %v659 = stablehlo.reshape %v658 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v660 = stablehlo.reshape %v659 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v661 = stablehlo.constant dense<0.0> : tensor<f32>
    %v662 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v663 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v664 = stablehlo.reduce(%v660 init: %v661) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v665 = stablehlo.broadcast_in_dim %v664, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v666 = stablehlo.divide %v665, %v662 : tensor<32x160x14x14xf32>
    %v667 = stablehlo.subtract %v660, %v666 : tensor<32x160x14x14xf32>
    %v668 = stablehlo.multiply %v667, %v667 : tensor<32x160x14x14xf32>
    %v669 = stablehlo.reduce(%v668 init: %v661) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v670 = stablehlo.broadcast_in_dim %v669, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v671 = stablehlo.divide %v670, %v662 : tensor<32x160x14x14xf32>
    %v672 = stablehlo.add %v671, %v663 : tensor<32x160x14x14xf32>
    %v673 = stablehlo.rsqrt %v672 : tensor<32x160x14x14xf32>
    %v674 = stablehlo.multiply %v667, %v673 : tensor<32x160x14x14xf32>
    %v675 = stablehlo.broadcast_in_dim %u6qg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v676 = stablehlo.broadcast_in_dim %u6qbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v677 = stablehlo.multiply %v674, %v675 : tensor<32x160x14x14xf32>
    %v678 = stablehlo.add %v677, %v676 : tensor<32x160x14x14xf32>
    %v679 = stablehlo.reshape %v678 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v680 = stablehlo.reshape %v679 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v681 = stablehlo.constant dense<0.0> : tensor<32x160x14x14xf32>
    %v682 = stablehlo.maximum %v680, %v681 : tensor<32x160x14x14xf32>
    %v683 = stablehlo.reshape %v682 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v684 = stablehlo.reshape %v683 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v685 = stablehlo.convolution(%v684, %u6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<32x640x14x14xf32>
    %v686 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v687 = stablehlo.add %v685, %v686 : tensor<32x640x14x14xf32>
    %v688 = stablehlo.reshape %v687 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v689 = stablehlo.reshape %v688 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v690 = stablehlo.constant dense<0.0> : tensor<f32>
    %v691 = stablehlo.constant dense<6272.0> : tensor<32x640x14x14xf32>
    %v692 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v693 = stablehlo.reduce(%v689 init: %v690) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v694 = stablehlo.broadcast_in_dim %v693, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v695 = stablehlo.divide %v694, %v691 : tensor<32x640x14x14xf32>
    %v696 = stablehlo.subtract %v689, %v695 : tensor<32x640x14x14xf32>
    %v697 = stablehlo.multiply %v696, %v696 : tensor<32x640x14x14xf32>
    %v698 = stablehlo.reduce(%v697 init: %v690) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v699 = stablehlo.broadcast_in_dim %v698, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v700 = stablehlo.divide %v699, %v691 : tensor<32x640x14x14xf32>
    %v701 = stablehlo.add %v700, %v692 : tensor<32x640x14x14xf32>
    %v702 = stablehlo.rsqrt %v701 : tensor<32x640x14x14xf32>
    %v703 = stablehlo.multiply %v696, %v702 : tensor<32x640x14x14xf32>
    %v704 = stablehlo.broadcast_in_dim %u6eg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v705 = stablehlo.broadcast_in_dim %u6ebt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v706 = stablehlo.multiply %v703, %v704 : tensor<32x640x14x14xf32>
    %v707 = stablehlo.add %v706, %v705 : tensor<32x640x14x14xf32>
    %v708 = stablehlo.reshape %v707 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v709 = stablehlo.reshape %v708 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v710 = stablehlo.constant dense<0.0> : tensor<32x160x28x28xf32>
    %v711 = stablehlo.maximum %v709, %v710 : tensor<32x160x28x28xf32>
    %v712 = stablehlo.reshape %v711 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v713 = stablehlo.reshape %v712 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v714 = stablehlo.convolution(%v713, %u6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 640 : i64} : (tensor<32x640x14x14xf32>, tensor<640x1x5x5xf32>) -> tensor<32x640x14x14xf32>
    %v715 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v716 = stablehlo.add %v714, %v715 : tensor<32x640x14x14xf32>
    %v717 = stablehlo.reshape %v716 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v718 = stablehlo.reshape %v717 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v719 = stablehlo.constant dense<0.0> : tensor<f32>
    %v720 = stablehlo.constant dense<6272.0> : tensor<32x640x14x14xf32>
    %v721 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v722 = stablehlo.reduce(%v718 init: %v719) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v723 = stablehlo.broadcast_in_dim %v722, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v724 = stablehlo.divide %v723, %v720 : tensor<32x640x14x14xf32>
    %v725 = stablehlo.subtract %v718, %v724 : tensor<32x640x14x14xf32>
    %v726 = stablehlo.multiply %v725, %v725 : tensor<32x640x14x14xf32>
    %v727 = stablehlo.reduce(%v726 init: %v719) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v728 = stablehlo.broadcast_in_dim %v727, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v729 = stablehlo.divide %v728, %v720 : tensor<32x640x14x14xf32>
    %v730 = stablehlo.add %v729, %v721 : tensor<32x640x14x14xf32>
    %v731 = stablehlo.rsqrt %v730 : tensor<32x640x14x14xf32>
    %v732 = stablehlo.multiply %v725, %v731 : tensor<32x640x14x14xf32>
    %v733 = stablehlo.broadcast_in_dim %u6dg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v734 = stablehlo.broadcast_in_dim %u6dbt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v735 = stablehlo.multiply %v732, %v733 : tensor<32x640x14x14xf32>
    %v736 = stablehlo.add %v735, %v734 : tensor<32x640x14x14xf32>
    %v737 = stablehlo.reshape %v736 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v738 = stablehlo.reshape %v737 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v739 = stablehlo.constant dense<0.0> : tensor<32x160x28x28xf32>
    %v740 = stablehlo.maximum %v738, %v739 : tensor<32x160x28x28xf32>
    %v741 = stablehlo.reshape %v740 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v742 = stablehlo.reshape %v741 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v743 = stablehlo.convolution(%v742, %u6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v744 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v745 = stablehlo.add %v743, %v744 : tensor<32x160x14x14xf32>
    %v746 = stablehlo.reshape %v745 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v747 = stablehlo.reshape %v746 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v748 = stablehlo.constant dense<0.0> : tensor<f32>
    %v749 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v750 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v751 = stablehlo.reduce(%v747 init: %v748) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v752 = stablehlo.broadcast_in_dim %v751, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v753 = stablehlo.divide %v752, %v749 : tensor<32x160x14x14xf32>
    %v754 = stablehlo.subtract %v747, %v753 : tensor<32x160x14x14xf32>
    %v755 = stablehlo.multiply %v754, %v754 : tensor<32x160x14x14xf32>
    %v756 = stablehlo.reduce(%v755 init: %v748) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v757 = stablehlo.broadcast_in_dim %v756, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v758 = stablehlo.divide %v757, %v749 : tensor<32x160x14x14xf32>
    %v759 = stablehlo.add %v758, %v750 : tensor<32x160x14x14xf32>
    %v760 = stablehlo.rsqrt %v759 : tensor<32x160x14x14xf32>
    %v761 = stablehlo.multiply %v754, %v760 : tensor<32x160x14x14xf32>
    %v762 = stablehlo.broadcast_in_dim %u6pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v763 = stablehlo.broadcast_in_dim %u6pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v764 = stablehlo.multiply %v761, %v762 : tensor<32x160x14x14xf32>
    %v765 = stablehlo.add %v764, %v763 : tensor<32x160x14x14xf32>
    %v766 = stablehlo.reshape %v765 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v767 = stablehlo.reshape %v766 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v768 = stablehlo.reshape %v654 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v769 = stablehlo.add %v767, %v768 : tensor<32x160x14x14xf32>
    %v770 = stablehlo.reshape %v769 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v771 = stablehlo.reshape %v770 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v772 = stablehlo.convolution(%v771, %u7qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<32x160x14x14xf32>
    %v773 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v774 = stablehlo.add %v772, %v773 : tensor<32x160x14x14xf32>
    %v775 = stablehlo.reshape %v774 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v776 = stablehlo.reshape %v775 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v777 = stablehlo.constant dense<0.0> : tensor<f32>
    %v778 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v779 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v780 = stablehlo.reduce(%v776 init: %v777) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v781 = stablehlo.broadcast_in_dim %v780, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v782 = stablehlo.divide %v781, %v778 : tensor<32x160x14x14xf32>
    %v783 = stablehlo.subtract %v776, %v782 : tensor<32x160x14x14xf32>
    %v784 = stablehlo.multiply %v783, %v783 : tensor<32x160x14x14xf32>
    %v785 = stablehlo.reduce(%v784 init: %v777) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v786 = stablehlo.broadcast_in_dim %v785, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v787 = stablehlo.divide %v786, %v778 : tensor<32x160x14x14xf32>
    %v788 = stablehlo.add %v787, %v779 : tensor<32x160x14x14xf32>
    %v789 = stablehlo.rsqrt %v788 : tensor<32x160x14x14xf32>
    %v790 = stablehlo.multiply %v783, %v789 : tensor<32x160x14x14xf32>
    %v791 = stablehlo.broadcast_in_dim %u7qg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v792 = stablehlo.broadcast_in_dim %u7qbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v793 = stablehlo.multiply %v790, %v791 : tensor<32x160x14x14xf32>
    %v794 = stablehlo.add %v793, %v792 : tensor<32x160x14x14xf32>
    %v795 = stablehlo.reshape %v794 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v796 = stablehlo.reshape %v795 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v797 = stablehlo.constant dense<0.0> : tensor<32x160x14x14xf32>
    %v798 = stablehlo.maximum %v796, %v797 : tensor<32x160x14x14xf32>
    %v799 = stablehlo.reshape %v798 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v800 = stablehlo.reshape %v799 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v801 = stablehlo.convolution(%v800, %u7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<32x640x14x14xf32>
    %v802 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v803 = stablehlo.add %v801, %v802 : tensor<32x640x14x14xf32>
    %v804 = stablehlo.reshape %v803 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v805 = stablehlo.reshape %v804 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v806 = stablehlo.constant dense<0.0> : tensor<f32>
    %v807 = stablehlo.constant dense<6272.0> : tensor<32x640x14x14xf32>
    %v808 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v809 = stablehlo.reduce(%v805 init: %v806) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v810 = stablehlo.broadcast_in_dim %v809, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v811 = stablehlo.divide %v810, %v807 : tensor<32x640x14x14xf32>
    %v812 = stablehlo.subtract %v805, %v811 : tensor<32x640x14x14xf32>
    %v813 = stablehlo.multiply %v812, %v812 : tensor<32x640x14x14xf32>
    %v814 = stablehlo.reduce(%v813 init: %v806) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v815 = stablehlo.broadcast_in_dim %v814, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v816 = stablehlo.divide %v815, %v807 : tensor<32x640x14x14xf32>
    %v817 = stablehlo.add %v816, %v808 : tensor<32x640x14x14xf32>
    %v818 = stablehlo.rsqrt %v817 : tensor<32x640x14x14xf32>
    %v819 = stablehlo.multiply %v812, %v818 : tensor<32x640x14x14xf32>
    %v820 = stablehlo.broadcast_in_dim %u7eg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v821 = stablehlo.broadcast_in_dim %u7ebt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v822 = stablehlo.multiply %v819, %v820 : tensor<32x640x14x14xf32>
    %v823 = stablehlo.add %v822, %v821 : tensor<32x640x14x14xf32>
    %v824 = stablehlo.reshape %v823 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v825 = stablehlo.reshape %v824 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v826 = stablehlo.constant dense<0.0> : tensor<32x160x28x28xf32>
    %v827 = stablehlo.maximum %v825, %v826 : tensor<32x160x28x28xf32>
    %v828 = stablehlo.reshape %v827 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v829 = stablehlo.reshape %v828 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v830 = stablehlo.convolution(%v829, %u7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 640 : i64} : (tensor<32x640x14x14xf32>, tensor<640x1x3x3xf32>) -> tensor<32x640x14x14xf32>
    %v831 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v832 = stablehlo.add %v830, %v831 : tensor<32x640x14x14xf32>
    %v833 = stablehlo.reshape %v832 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v834 = stablehlo.reshape %v833 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v835 = stablehlo.constant dense<0.0> : tensor<f32>
    %v836 = stablehlo.constant dense<6272.0> : tensor<32x640x14x14xf32>
    %v837 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v838 = stablehlo.reduce(%v834 init: %v835) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v839 = stablehlo.broadcast_in_dim %v838, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v840 = stablehlo.divide %v839, %v836 : tensor<32x640x14x14xf32>
    %v841 = stablehlo.subtract %v834, %v840 : tensor<32x640x14x14xf32>
    %v842 = stablehlo.multiply %v841, %v841 : tensor<32x640x14x14xf32>
    %v843 = stablehlo.reduce(%v842 init: %v835) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v844 = stablehlo.broadcast_in_dim %v843, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v845 = stablehlo.divide %v844, %v836 : tensor<32x640x14x14xf32>
    %v846 = stablehlo.add %v845, %v837 : tensor<32x640x14x14xf32>
    %v847 = stablehlo.rsqrt %v846 : tensor<32x640x14x14xf32>
    %v848 = stablehlo.multiply %v841, %v847 : tensor<32x640x14x14xf32>
    %v849 = stablehlo.broadcast_in_dim %u7dg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v850 = stablehlo.broadcast_in_dim %u7dbt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v851 = stablehlo.multiply %v848, %v849 : tensor<32x640x14x14xf32>
    %v852 = stablehlo.add %v851, %v850 : tensor<32x640x14x14xf32>
    %v853 = stablehlo.reshape %v852 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v854 = stablehlo.reshape %v853 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v855 = stablehlo.constant dense<0.0> : tensor<32x160x28x28xf32>
    %v856 = stablehlo.maximum %v854, %v855 : tensor<32x160x28x28xf32>
    %v857 = stablehlo.reshape %v856 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v858 = stablehlo.reshape %v857 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v859 = stablehlo.convolution(%v858, %u7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v860 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v861 = stablehlo.add %v859, %v860 : tensor<32x160x14x14xf32>
    %v862 = stablehlo.reshape %v861 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v863 = stablehlo.reshape %v862 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v864 = stablehlo.constant dense<0.0> : tensor<f32>
    %v865 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v866 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v867 = stablehlo.reduce(%v863 init: %v864) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v868 = stablehlo.broadcast_in_dim %v867, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v869 = stablehlo.divide %v868, %v865 : tensor<32x160x14x14xf32>
    %v870 = stablehlo.subtract %v863, %v869 : tensor<32x160x14x14xf32>
    %v871 = stablehlo.multiply %v870, %v870 : tensor<32x160x14x14xf32>
    %v872 = stablehlo.reduce(%v871 init: %v864) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v873 = stablehlo.broadcast_in_dim %v872, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v874 = stablehlo.divide %v873, %v865 : tensor<32x160x14x14xf32>
    %v875 = stablehlo.add %v874, %v866 : tensor<32x160x14x14xf32>
    %v876 = stablehlo.rsqrt %v875 : tensor<32x160x14x14xf32>
    %v877 = stablehlo.multiply %v870, %v876 : tensor<32x160x14x14xf32>
    %v878 = stablehlo.broadcast_in_dim %u7pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v879 = stablehlo.broadcast_in_dim %u7pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v880 = stablehlo.multiply %v877, %v878 : tensor<32x160x14x14xf32>
    %v881 = stablehlo.add %v880, %v879 : tensor<32x160x14x14xf32>
    %v882 = stablehlo.reshape %v881 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v883 = stablehlo.reshape %v882 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v884 = stablehlo.reshape %v770 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v885 = stablehlo.add %v883, %v884 : tensor<32x160x14x14xf32>
    %v886 = stablehlo.reshape %v885 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v887 = stablehlo.reshape %v886 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v888 = stablehlo.convolution(%v887, %u8qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<32x160x14x14xf32>
    %v889 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v890 = stablehlo.add %v888, %v889 : tensor<32x160x14x14xf32>
    %v891 = stablehlo.reshape %v890 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v892 = stablehlo.reshape %v891 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v893 = stablehlo.constant dense<0.0> : tensor<f32>
    %v894 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v895 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v896 = stablehlo.reduce(%v892 init: %v893) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v897 = stablehlo.broadcast_in_dim %v896, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v898 = stablehlo.divide %v897, %v894 : tensor<32x160x14x14xf32>
    %v899 = stablehlo.subtract %v892, %v898 : tensor<32x160x14x14xf32>
    %v900 = stablehlo.multiply %v899, %v899 : tensor<32x160x14x14xf32>
    %v901 = stablehlo.reduce(%v900 init: %v893) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v902 = stablehlo.broadcast_in_dim %v901, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v903 = stablehlo.divide %v902, %v894 : tensor<32x160x14x14xf32>
    %v904 = stablehlo.add %v903, %v895 : tensor<32x160x14x14xf32>
    %v905 = stablehlo.rsqrt %v904 : tensor<32x160x14x14xf32>
    %v906 = stablehlo.multiply %v899, %v905 : tensor<32x160x14x14xf32>
    %v907 = stablehlo.broadcast_in_dim %u8qg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v908 = stablehlo.broadcast_in_dim %u8qbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v909 = stablehlo.multiply %v906, %v907 : tensor<32x160x14x14xf32>
    %v910 = stablehlo.add %v909, %v908 : tensor<32x160x14x14xf32>
    %v911 = stablehlo.reshape %v910 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v912 = stablehlo.reshape %v911 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v913 = stablehlo.constant dense<0.0> : tensor<32x160x14x14xf32>
    %v914 = stablehlo.maximum %v912, %v913 : tensor<32x160x14x14xf32>
    %v915 = stablehlo.reshape %v914 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v916 = stablehlo.reshape %v915 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v917 = stablehlo.convolution(%v916, %u8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<32x640x14x14xf32>
    %v918 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v919 = stablehlo.add %v917, %v918 : tensor<32x640x14x14xf32>
    %v920 = stablehlo.reshape %v919 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v921 = stablehlo.reshape %v920 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v922 = stablehlo.constant dense<0.0> : tensor<f32>
    %v923 = stablehlo.constant dense<6272.0> : tensor<32x640x14x14xf32>
    %v924 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v925 = stablehlo.reduce(%v921 init: %v922) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v926 = stablehlo.broadcast_in_dim %v925, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v927 = stablehlo.divide %v926, %v923 : tensor<32x640x14x14xf32>
    %v928 = stablehlo.subtract %v921, %v927 : tensor<32x640x14x14xf32>
    %v929 = stablehlo.multiply %v928, %v928 : tensor<32x640x14x14xf32>
    %v930 = stablehlo.reduce(%v929 init: %v922) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v931 = stablehlo.broadcast_in_dim %v930, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v932 = stablehlo.divide %v931, %v923 : tensor<32x640x14x14xf32>
    %v933 = stablehlo.add %v932, %v924 : tensor<32x640x14x14xf32>
    %v934 = stablehlo.rsqrt %v933 : tensor<32x640x14x14xf32>
    %v935 = stablehlo.multiply %v928, %v934 : tensor<32x640x14x14xf32>
    %v936 = stablehlo.broadcast_in_dim %u8eg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v937 = stablehlo.broadcast_in_dim %u8ebt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v938 = stablehlo.multiply %v935, %v936 : tensor<32x640x14x14xf32>
    %v939 = stablehlo.add %v938, %v937 : tensor<32x640x14x14xf32>
    %v940 = stablehlo.reshape %v939 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v941 = stablehlo.reshape %v940 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v942 = stablehlo.constant dense<0.0> : tensor<32x160x28x28xf32>
    %v943 = stablehlo.maximum %v941, %v942 : tensor<32x160x28x28xf32>
    %v944 = stablehlo.reshape %v943 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v945 = stablehlo.reshape %v944 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v946 = stablehlo.convolution(%v945, %u8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v947 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v948 = stablehlo.add %v946, %v947 : tensor<32x160x14x14xf32>
    %v949 = stablehlo.reshape %v948 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v950 = stablehlo.reshape %v949 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v951 = stablehlo.constant dense<0.0> : tensor<f32>
    %v952 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v953 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v954 = stablehlo.reduce(%v950 init: %v951) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v955 = stablehlo.broadcast_in_dim %v954, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v956 = stablehlo.divide %v955, %v952 : tensor<32x160x14x14xf32>
    %v957 = stablehlo.subtract %v950, %v956 : tensor<32x160x14x14xf32>
    %v958 = stablehlo.multiply %v957, %v957 : tensor<32x160x14x14xf32>
    %v959 = stablehlo.reduce(%v958 init: %v951) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v960 = stablehlo.broadcast_in_dim %v959, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v961 = stablehlo.divide %v960, %v952 : tensor<32x160x14x14xf32>
    %v962 = stablehlo.add %v961, %v953 : tensor<32x160x14x14xf32>
    %v963 = stablehlo.rsqrt %v962 : tensor<32x160x14x14xf32>
    %v964 = stablehlo.multiply %v957, %v963 : tensor<32x160x14x14xf32>
    %v965 = stablehlo.broadcast_in_dim %u8pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v966 = stablehlo.broadcast_in_dim %u8pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v967 = stablehlo.multiply %v964, %v965 : tensor<32x160x14x14xf32>
    %v968 = stablehlo.add %v967, %v966 : tensor<32x160x14x14xf32>
    %v969 = stablehlo.reshape %v968 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v970 = stablehlo.reshape %v969 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v971 = stablehlo.reshape %v886 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v972 = stablehlo.add %v970, %v971 : tensor<32x160x14x14xf32>
    %v973 = stablehlo.reshape %v972 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v974 = stablehlo.reshape %v973 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v975 = stablehlo.convolution(%v974, %u9eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x14x14xf32>, tensor<320x160x1x1xf32>) -> tensor<32x320x14x14xf32>
    %v976 = stablehlo.broadcast_in_dim %zb320, dims = [1] : (tensor<320xf32>) -> tensor<32x320x14x14xf32>
    %v977 = stablehlo.add %v975, %v976 : tensor<32x320x14x14xf32>
    %v978 = stablehlo.reshape %v977 : (tensor<32x320x14x14xf32>) -> tensor<32x62720xf32>
    %v979 = stablehlo.reshape %v978 : (tensor<32x62720xf32>) -> tensor<32x320x14x14xf32>
    %v980 = stablehlo.constant dense<0.0> : tensor<f32>
    %v981 = stablehlo.constant dense<6272.0> : tensor<32x320x14x14xf32>
    %v982 = stablehlo.constant dense<1.0e-5> : tensor<32x320x14x14xf32>
    %v983 = stablehlo.reduce(%v979 init: %v980) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x14x14xf32>, tensor<f32>) -> tensor<320xf32>
    %v984 = stablehlo.broadcast_in_dim %v983, dims = [1] : (tensor<320xf32>) -> tensor<32x320x14x14xf32>
    %v985 = stablehlo.divide %v984, %v981 : tensor<32x320x14x14xf32>
    %v986 = stablehlo.subtract %v979, %v985 : tensor<32x320x14x14xf32>
    %v987 = stablehlo.multiply %v986, %v986 : tensor<32x320x14x14xf32>
    %v988 = stablehlo.reduce(%v987 init: %v980) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x14x14xf32>, tensor<f32>) -> tensor<320xf32>
    %v989 = stablehlo.broadcast_in_dim %v988, dims = [1] : (tensor<320xf32>) -> tensor<32x320x14x14xf32>
    %v990 = stablehlo.divide %v989, %v981 : tensor<32x320x14x14xf32>
    %v991 = stablehlo.add %v990, %v982 : tensor<32x320x14x14xf32>
    %v992 = stablehlo.rsqrt %v991 : tensor<32x320x14x14xf32>
    %v993 = stablehlo.multiply %v986, %v992 : tensor<32x320x14x14xf32>
    %v994 = stablehlo.broadcast_in_dim %u9eg, dims = [1] : (tensor<320xf32>) -> tensor<32x320x14x14xf32>
    %v995 = stablehlo.broadcast_in_dim %u9ebt, dims = [1] : (tensor<320xf32>) -> tensor<32x320x14x14xf32>
    %v996 = stablehlo.multiply %v993, %v994 : tensor<32x320x14x14xf32>
    %v997 = stablehlo.add %v996, %v995 : tensor<32x320x14x14xf32>
    %v998 = stablehlo.reshape %v997 : (tensor<32x320x14x14xf32>) -> tensor<32x62720xf32>
    %v999 = stablehlo.reshape %v998 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v1000 = stablehlo.constant dense<0.0> : tensor<32x80x28x28xf32>
    %v1001 = stablehlo.maximum %v999, %v1000 : tensor<32x80x28x28xf32>
    %v1002 = stablehlo.reshape %v1001 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v1003 = stablehlo.reshape %v1002 : (tensor<32x62720xf32>) -> tensor<32x320x14x14xf32>
    %v1004 = stablehlo.convolution(%v1003, %u9pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x320x14x14xf32>, tensor<160x320x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v1005 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v1006 = stablehlo.add %v1004, %v1005 : tensor<32x160x14x14xf32>
    %v1007 = stablehlo.reshape %v1006 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v1008 = stablehlo.reshape %v1007 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v1009 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1010 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v1011 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v1012 = stablehlo.reduce(%v1008 init: %v1009) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v1013 = stablehlo.broadcast_in_dim %v1012, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v1014 = stablehlo.divide %v1013, %v1010 : tensor<32x160x14x14xf32>
    %v1015 = stablehlo.subtract %v1008, %v1014 : tensor<32x160x14x14xf32>
    %v1016 = stablehlo.multiply %v1015, %v1015 : tensor<32x160x14x14xf32>
    %v1017 = stablehlo.reduce(%v1016 init: %v1009) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v1018 = stablehlo.broadcast_in_dim %v1017, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v1019 = stablehlo.divide %v1018, %v1010 : tensor<32x160x14x14xf32>
    %v1020 = stablehlo.add %v1019, %v1011 : tensor<32x160x14x14xf32>
    %v1021 = stablehlo.rsqrt %v1020 : tensor<32x160x14x14xf32>
    %v1022 = stablehlo.multiply %v1015, %v1021 : tensor<32x160x14x14xf32>
    %v1023 = stablehlo.broadcast_in_dim %u9pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v1024 = stablehlo.broadcast_in_dim %u9pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v1025 = stablehlo.multiply %v1022, %v1023 : tensor<32x160x14x14xf32>
    %v1026 = stablehlo.add %v1025, %v1024 : tensor<32x160x14x14xf32>
    %v1027 = stablehlo.reshape %v1026 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v1028 = stablehlo.reshape %v1027 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v1029 = stablehlo.reshape %v973 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v1030 = stablehlo.add %v1028, %v1029 : tensor<32x160x14x14xf32>
    %v1031 = stablehlo.reshape %v1030 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v1032 = stablehlo.reshape %v1031 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v1033 = stablehlo.convolution(%v1032, %u10qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<32x160x14x14xf32>
    %v1034 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v1035 = stablehlo.add %v1033, %v1034 : tensor<32x160x14x14xf32>
    %v1036 = stablehlo.reshape %v1035 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v1037 = stablehlo.reshape %v1036 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v1038 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1039 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v1040 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v1041 = stablehlo.reduce(%v1037 init: %v1038) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v1042 = stablehlo.broadcast_in_dim %v1041, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v1043 = stablehlo.divide %v1042, %v1039 : tensor<32x160x14x14xf32>
    %v1044 = stablehlo.subtract %v1037, %v1043 : tensor<32x160x14x14xf32>
    %v1045 = stablehlo.multiply %v1044, %v1044 : tensor<32x160x14x14xf32>
    %v1046 = stablehlo.reduce(%v1045 init: %v1038) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v1047 = stablehlo.broadcast_in_dim %v1046, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v1048 = stablehlo.divide %v1047, %v1039 : tensor<32x160x14x14xf32>
    %v1049 = stablehlo.add %v1048, %v1040 : tensor<32x160x14x14xf32>
    %v1050 = stablehlo.rsqrt %v1049 : tensor<32x160x14x14xf32>
    %v1051 = stablehlo.multiply %v1044, %v1050 : tensor<32x160x14x14xf32>
    %v1052 = stablehlo.broadcast_in_dim %u10qg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v1053 = stablehlo.broadcast_in_dim %u10qbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v1054 = stablehlo.multiply %v1051, %v1052 : tensor<32x160x14x14xf32>
    %v1055 = stablehlo.add %v1054, %v1053 : tensor<32x160x14x14xf32>
    %v1056 = stablehlo.reshape %v1055 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v1057 = stablehlo.reshape %v1056 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v1058 = stablehlo.constant dense<0.0> : tensor<32x160x14x14xf32>
    %v1059 = stablehlo.maximum %v1057, %v1058 : tensor<32x160x14x14xf32>
    %v1060 = stablehlo.reshape %v1059 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v1061 = stablehlo.reshape %v1060 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v1062 = stablehlo.convolution(%v1061, %u10eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<32x640x14x14xf32>
    %v1063 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v1064 = stablehlo.add %v1062, %v1063 : tensor<32x640x14x14xf32>
    %v1065 = stablehlo.reshape %v1064 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v1066 = stablehlo.reshape %v1065 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v1067 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1068 = stablehlo.constant dense<6272.0> : tensor<32x640x14x14xf32>
    %v1069 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v1070 = stablehlo.reduce(%v1066 init: %v1067) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v1071 = stablehlo.broadcast_in_dim %v1070, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v1072 = stablehlo.divide %v1071, %v1068 : tensor<32x640x14x14xf32>
    %v1073 = stablehlo.subtract %v1066, %v1072 : tensor<32x640x14x14xf32>
    %v1074 = stablehlo.multiply %v1073, %v1073 : tensor<32x640x14x14xf32>
    %v1075 = stablehlo.reduce(%v1074 init: %v1067) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v1076 = stablehlo.broadcast_in_dim %v1075, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v1077 = stablehlo.divide %v1076, %v1068 : tensor<32x640x14x14xf32>
    %v1078 = stablehlo.add %v1077, %v1069 : tensor<32x640x14x14xf32>
    %v1079 = stablehlo.rsqrt %v1078 : tensor<32x640x14x14xf32>
    %v1080 = stablehlo.multiply %v1073, %v1079 : tensor<32x640x14x14xf32>
    %v1081 = stablehlo.broadcast_in_dim %u10eg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v1082 = stablehlo.broadcast_in_dim %u10ebt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v1083 = stablehlo.multiply %v1080, %v1081 : tensor<32x640x14x14xf32>
    %v1084 = stablehlo.add %v1083, %v1082 : tensor<32x640x14x14xf32>
    %v1085 = stablehlo.reshape %v1084 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v1086 = stablehlo.reshape %v1085 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v1087 = stablehlo.constant dense<0.0> : tensor<32x160x28x28xf32>
    %v1088 = stablehlo.maximum %v1086, %v1087 : tensor<32x160x28x28xf32>
    %v1089 = stablehlo.reshape %v1088 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v1090 = stablehlo.reshape %v1089 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v1091 = stablehlo.convolution(%v1090, %u10pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v1092 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v1093 = stablehlo.add %v1091, %v1092 : tensor<32x160x14x14xf32>
    %v1094 = stablehlo.reshape %v1093 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v1095 = stablehlo.reshape %v1094 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v1096 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1097 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v1098 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v1099 = stablehlo.reduce(%v1095 init: %v1096) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v1100 = stablehlo.broadcast_in_dim %v1099, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v1101 = stablehlo.divide %v1100, %v1097 : tensor<32x160x14x14xf32>
    %v1102 = stablehlo.subtract %v1095, %v1101 : tensor<32x160x14x14xf32>
    %v1103 = stablehlo.multiply %v1102, %v1102 : tensor<32x160x14x14xf32>
    %v1104 = stablehlo.reduce(%v1103 init: %v1096) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v1105 = stablehlo.broadcast_in_dim %v1104, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v1106 = stablehlo.divide %v1105, %v1097 : tensor<32x160x14x14xf32>
    %v1107 = stablehlo.add %v1106, %v1098 : tensor<32x160x14x14xf32>
    %v1108 = stablehlo.rsqrt %v1107 : tensor<32x160x14x14xf32>
    %v1109 = stablehlo.multiply %v1102, %v1108 : tensor<32x160x14x14xf32>
    %v1110 = stablehlo.broadcast_in_dim %u10pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v1111 = stablehlo.broadcast_in_dim %u10pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v1112 = stablehlo.multiply %v1109, %v1110 : tensor<32x160x14x14xf32>
    %v1113 = stablehlo.add %v1112, %v1111 : tensor<32x160x14x14xf32>
    %v1114 = stablehlo.reshape %v1113 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v1115 = stablehlo.reshape %v1114 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v1116 = stablehlo.reshape %v1031 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v1117 = stablehlo.add %v1115, %v1116 : tensor<32x160x14x14xf32>
    %v1118 = stablehlo.reshape %v1117 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v1119 = stablehlo.reshape %v1118 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v1120 = stablehlo.convolution(%v1119, %u11qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x14x14xf32>, tensor<160x1x5x5xf32>) -> tensor<32x160x7x7xf32>
    %v1121 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1122 = stablehlo.add %v1120, %v1121 : tensor<32x160x7x7xf32>
    %v1123 = stablehlo.reshape %v1122 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1124 = stablehlo.reshape %v1123 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1125 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1126 = stablehlo.constant dense<1568.0> : tensor<32x160x7x7xf32>
    %v1127 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v1128 = stablehlo.reduce(%v1124 init: %v1125) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v1129 = stablehlo.broadcast_in_dim %v1128, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1130 = stablehlo.divide %v1129, %v1126 : tensor<32x160x7x7xf32>
    %v1131 = stablehlo.subtract %v1124, %v1130 : tensor<32x160x7x7xf32>
    %v1132 = stablehlo.multiply %v1131, %v1131 : tensor<32x160x7x7xf32>
    %v1133 = stablehlo.reduce(%v1132 init: %v1125) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v1134 = stablehlo.broadcast_in_dim %v1133, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1135 = stablehlo.divide %v1134, %v1126 : tensor<32x160x7x7xf32>
    %v1136 = stablehlo.add %v1135, %v1127 : tensor<32x160x7x7xf32>
    %v1137 = stablehlo.rsqrt %v1136 : tensor<32x160x7x7xf32>
    %v1138 = stablehlo.multiply %v1131, %v1137 : tensor<32x160x7x7xf32>
    %v1139 = stablehlo.broadcast_in_dim %u11qg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1140 = stablehlo.broadcast_in_dim %u11qbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1141 = stablehlo.multiply %v1138, %v1139 : tensor<32x160x7x7xf32>
    %v1142 = stablehlo.add %v1141, %v1140 : tensor<32x160x7x7xf32>
    %v1143 = stablehlo.reshape %v1142 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1144 = stablehlo.reshape %v1143 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1145 = stablehlo.constant dense<0.0> : tensor<32x160x7x7xf32>
    %v1146 = stablehlo.maximum %v1144, %v1145 : tensor<32x160x7x7xf32>
    %v1147 = stablehlo.reshape %v1146 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1148 = stablehlo.reshape %v1147 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1149 = stablehlo.convolution(%v1148, %u11eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v1150 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1151 = stablehlo.add %v1149, %v1150 : tensor<32x960x7x7xf32>
    %v1152 = stablehlo.reshape %v1151 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1153 = stablehlo.reshape %v1152 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1154 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1155 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v1156 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1157 = stablehlo.reduce(%v1153 init: %v1154) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1158 = stablehlo.broadcast_in_dim %v1157, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1159 = stablehlo.divide %v1158, %v1155 : tensor<32x960x7x7xf32>
    %v1160 = stablehlo.subtract %v1153, %v1159 : tensor<32x960x7x7xf32>
    %v1161 = stablehlo.multiply %v1160, %v1160 : tensor<32x960x7x7xf32>
    %v1162 = stablehlo.reduce(%v1161 init: %v1154) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1163 = stablehlo.broadcast_in_dim %v1162, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1164 = stablehlo.divide %v1163, %v1155 : tensor<32x960x7x7xf32>
    %v1165 = stablehlo.add %v1164, %v1156 : tensor<32x960x7x7xf32>
    %v1166 = stablehlo.rsqrt %v1165 : tensor<32x960x7x7xf32>
    %v1167 = stablehlo.multiply %v1160, %v1166 : tensor<32x960x7x7xf32>
    %v1168 = stablehlo.broadcast_in_dim %u11eg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1169 = stablehlo.broadcast_in_dim %u11ebt, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1170 = stablehlo.multiply %v1167, %v1168 : tensor<32x960x7x7xf32>
    %v1171 = stablehlo.add %v1170, %v1169 : tensor<32x960x7x7xf32>
    %v1172 = stablehlo.reshape %v1171 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1173 = stablehlo.reshape %v1172 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1174 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v1175 = stablehlo.maximum %v1173, %v1174 : tensor<32x960x7x7xf32>
    %v1176 = stablehlo.reshape %v1175 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1177 = stablehlo.reshape %v1176 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1178 = stablehlo.convolution(%v1177, %u11dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x5x5xf32>) -> tensor<32x960x7x7xf32>
    %v1179 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1180 = stablehlo.add %v1178, %v1179 : tensor<32x960x7x7xf32>
    %v1181 = stablehlo.reshape %v1180 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1182 = stablehlo.reshape %v1181 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1183 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1184 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v1185 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1186 = stablehlo.reduce(%v1182 init: %v1183) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1187 = stablehlo.broadcast_in_dim %v1186, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1188 = stablehlo.divide %v1187, %v1184 : tensor<32x960x7x7xf32>
    %v1189 = stablehlo.subtract %v1182, %v1188 : tensor<32x960x7x7xf32>
    %v1190 = stablehlo.multiply %v1189, %v1189 : tensor<32x960x7x7xf32>
    %v1191 = stablehlo.reduce(%v1190 init: %v1183) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1192 = stablehlo.broadcast_in_dim %v1191, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1193 = stablehlo.divide %v1192, %v1184 : tensor<32x960x7x7xf32>
    %v1194 = stablehlo.add %v1193, %v1185 : tensor<32x960x7x7xf32>
    %v1195 = stablehlo.rsqrt %v1194 : tensor<32x960x7x7xf32>
    %v1196 = stablehlo.multiply %v1189, %v1195 : tensor<32x960x7x7xf32>
    %v1197 = stablehlo.broadcast_in_dim %u11dg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1198 = stablehlo.broadcast_in_dim %u11dbt, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1199 = stablehlo.multiply %v1196, %v1197 : tensor<32x960x7x7xf32>
    %v1200 = stablehlo.add %v1199, %v1198 : tensor<32x960x7x7xf32>
    %v1201 = stablehlo.reshape %v1200 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1202 = stablehlo.reshape %v1201 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1203 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v1204 = stablehlo.maximum %v1202, %v1203 : tensor<32x960x7x7xf32>
    %v1205 = stablehlo.reshape %v1204 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1206 = stablehlo.reshape %v1205 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1207 = stablehlo.convolution(%v1206, %u11pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<256x960x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1208 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1209 = stablehlo.add %v1207, %v1208 : tensor<32x256x7x7xf32>
    %v1210 = stablehlo.reshape %v1209 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1211 = stablehlo.reshape %v1210 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1212 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1213 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1214 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1215 = stablehlo.reduce(%v1211 init: %v1212) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1216 = stablehlo.broadcast_in_dim %v1215, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1217 = stablehlo.divide %v1216, %v1213 : tensor<32x256x7x7xf32>
    %v1218 = stablehlo.subtract %v1211, %v1217 : tensor<32x256x7x7xf32>
    %v1219 = stablehlo.multiply %v1218, %v1218 : tensor<32x256x7x7xf32>
    %v1220 = stablehlo.reduce(%v1219 init: %v1212) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1221 = stablehlo.broadcast_in_dim %v1220, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1222 = stablehlo.divide %v1221, %v1213 : tensor<32x256x7x7xf32>
    %v1223 = stablehlo.add %v1222, %v1214 : tensor<32x256x7x7xf32>
    %v1224 = stablehlo.rsqrt %v1223 : tensor<32x256x7x7xf32>
    %v1225 = stablehlo.multiply %v1218, %v1224 : tensor<32x256x7x7xf32>
    %v1226 = stablehlo.broadcast_in_dim %u11pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1227 = stablehlo.broadcast_in_dim %u11pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1228 = stablehlo.multiply %v1225, %v1226 : tensor<32x256x7x7xf32>
    %v1229 = stablehlo.add %v1228, %v1227 : tensor<32x256x7x7xf32>
    %v1230 = stablehlo.reshape %v1229 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1231 = stablehlo.reshape %v1230 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1232 = stablehlo.convolution(%v1231, %u12qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x7x7xf32>, tensor<256x1x5x5xf32>) -> tensor<32x256x7x7xf32>
    %v1233 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1234 = stablehlo.add %v1232, %v1233 : tensor<32x256x7x7xf32>
    %v1235 = stablehlo.reshape %v1234 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1236 = stablehlo.reshape %v1235 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1237 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1238 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1239 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1240 = stablehlo.reduce(%v1236 init: %v1237) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1241 = stablehlo.broadcast_in_dim %v1240, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1242 = stablehlo.divide %v1241, %v1238 : tensor<32x256x7x7xf32>
    %v1243 = stablehlo.subtract %v1236, %v1242 : tensor<32x256x7x7xf32>
    %v1244 = stablehlo.multiply %v1243, %v1243 : tensor<32x256x7x7xf32>
    %v1245 = stablehlo.reduce(%v1244 init: %v1237) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1246 = stablehlo.broadcast_in_dim %v1245, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1247 = stablehlo.divide %v1246, %v1238 : tensor<32x256x7x7xf32>
    %v1248 = stablehlo.add %v1247, %v1239 : tensor<32x256x7x7xf32>
    %v1249 = stablehlo.rsqrt %v1248 : tensor<32x256x7x7xf32>
    %v1250 = stablehlo.multiply %v1243, %v1249 : tensor<32x256x7x7xf32>
    %v1251 = stablehlo.broadcast_in_dim %u12qg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1252 = stablehlo.broadcast_in_dim %u12qbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1253 = stablehlo.multiply %v1250, %v1251 : tensor<32x256x7x7xf32>
    %v1254 = stablehlo.add %v1253, %v1252 : tensor<32x256x7x7xf32>
    %v1255 = stablehlo.reshape %v1254 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1256 = stablehlo.reshape %v1255 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1257 = stablehlo.constant dense<0.0> : tensor<32x256x7x7xf32>
    %v1258 = stablehlo.maximum %v1256, %v1257 : tensor<32x256x7x7xf32>
    %v1259 = stablehlo.reshape %v1258 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1260 = stablehlo.reshape %v1259 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1261 = stablehlo.convolution(%v1260, %u12eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v1262 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1263 = stablehlo.add %v1261, %v1262 : tensor<32x1024x7x7xf32>
    %v1264 = stablehlo.reshape %v1263 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1265 = stablehlo.reshape %v1264 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1266 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1267 = stablehlo.constant dense<1568.0> : tensor<32x1024x7x7xf32>
    %v1268 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1269 = stablehlo.reduce(%v1265 init: %v1266) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1270 = stablehlo.broadcast_in_dim %v1269, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1271 = stablehlo.divide %v1270, %v1267 : tensor<32x1024x7x7xf32>
    %v1272 = stablehlo.subtract %v1265, %v1271 : tensor<32x1024x7x7xf32>
    %v1273 = stablehlo.multiply %v1272, %v1272 : tensor<32x1024x7x7xf32>
    %v1274 = stablehlo.reduce(%v1273 init: %v1266) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1275 = stablehlo.broadcast_in_dim %v1274, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1276 = stablehlo.divide %v1275, %v1267 : tensor<32x1024x7x7xf32>
    %v1277 = stablehlo.add %v1276, %v1268 : tensor<32x1024x7x7xf32>
    %v1278 = stablehlo.rsqrt %v1277 : tensor<32x1024x7x7xf32>
    %v1279 = stablehlo.multiply %v1272, %v1278 : tensor<32x1024x7x7xf32>
    %v1280 = stablehlo.broadcast_in_dim %u12eg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1281 = stablehlo.broadcast_in_dim %u12ebt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1282 = stablehlo.multiply %v1279, %v1280 : tensor<32x1024x7x7xf32>
    %v1283 = stablehlo.add %v1282, %v1281 : tensor<32x1024x7x7xf32>
    %v1284 = stablehlo.reshape %v1283 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1285 = stablehlo.reshape %v1284 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1286 = stablehlo.constant dense<0.0> : tensor<32x1024x7x7xf32>
    %v1287 = stablehlo.maximum %v1285, %v1286 : tensor<32x1024x7x7xf32>
    %v1288 = stablehlo.reshape %v1287 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1289 = stablehlo.reshape %v1288 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1290 = stablehlo.convolution(%v1289, %u12dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<32x1024x7x7xf32>, tensor<1024x1x5x5xf32>) -> tensor<32x1024x7x7xf32>
    %v1291 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1292 = stablehlo.add %v1290, %v1291 : tensor<32x1024x7x7xf32>
    %v1293 = stablehlo.reshape %v1292 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1294 = stablehlo.reshape %v1293 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1295 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1296 = stablehlo.constant dense<1568.0> : tensor<32x1024x7x7xf32>
    %v1297 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1298 = stablehlo.reduce(%v1294 init: %v1295) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1299 = stablehlo.broadcast_in_dim %v1298, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1300 = stablehlo.divide %v1299, %v1296 : tensor<32x1024x7x7xf32>
    %v1301 = stablehlo.subtract %v1294, %v1300 : tensor<32x1024x7x7xf32>
    %v1302 = stablehlo.multiply %v1301, %v1301 : tensor<32x1024x7x7xf32>
    %v1303 = stablehlo.reduce(%v1302 init: %v1295) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1304 = stablehlo.broadcast_in_dim %v1303, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1305 = stablehlo.divide %v1304, %v1296 : tensor<32x1024x7x7xf32>
    %v1306 = stablehlo.add %v1305, %v1297 : tensor<32x1024x7x7xf32>
    %v1307 = stablehlo.rsqrt %v1306 : tensor<32x1024x7x7xf32>
    %v1308 = stablehlo.multiply %v1301, %v1307 : tensor<32x1024x7x7xf32>
    %v1309 = stablehlo.broadcast_in_dim %u12dg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1310 = stablehlo.broadcast_in_dim %u12dbt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1311 = stablehlo.multiply %v1308, %v1309 : tensor<32x1024x7x7xf32>
    %v1312 = stablehlo.add %v1311, %v1310 : tensor<32x1024x7x7xf32>
    %v1313 = stablehlo.reshape %v1312 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1314 = stablehlo.reshape %v1313 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1315 = stablehlo.constant dense<0.0> : tensor<32x1024x7x7xf32>
    %v1316 = stablehlo.maximum %v1314, %v1315 : tensor<32x1024x7x7xf32>
    %v1317 = stablehlo.reshape %v1316 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1318 = stablehlo.reshape %v1317 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1319 = stablehlo.convolution(%v1318, %u12pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1320 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1321 = stablehlo.add %v1319, %v1320 : tensor<32x256x7x7xf32>
    %v1322 = stablehlo.reshape %v1321 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1323 = stablehlo.reshape %v1322 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1324 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1325 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1326 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1327 = stablehlo.reduce(%v1323 init: %v1324) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1328 = stablehlo.broadcast_in_dim %v1327, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1329 = stablehlo.divide %v1328, %v1325 : tensor<32x256x7x7xf32>
    %v1330 = stablehlo.subtract %v1323, %v1329 : tensor<32x256x7x7xf32>
    %v1331 = stablehlo.multiply %v1330, %v1330 : tensor<32x256x7x7xf32>
    %v1332 = stablehlo.reduce(%v1331 init: %v1324) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1333 = stablehlo.broadcast_in_dim %v1332, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1334 = stablehlo.divide %v1333, %v1325 : tensor<32x256x7x7xf32>
    %v1335 = stablehlo.add %v1334, %v1326 : tensor<32x256x7x7xf32>
    %v1336 = stablehlo.rsqrt %v1335 : tensor<32x256x7x7xf32>
    %v1337 = stablehlo.multiply %v1330, %v1336 : tensor<32x256x7x7xf32>
    %v1338 = stablehlo.broadcast_in_dim %u12pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1339 = stablehlo.broadcast_in_dim %u12pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1340 = stablehlo.multiply %v1337, %v1338 : tensor<32x256x7x7xf32>
    %v1341 = stablehlo.add %v1340, %v1339 : tensor<32x256x7x7xf32>
    %v1342 = stablehlo.reshape %v1341 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1343 = stablehlo.reshape %v1342 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1344 = stablehlo.reshape %v1230 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1345 = stablehlo.add %v1343, %v1344 : tensor<32x256x7x7xf32>
    %v1346 = stablehlo.reshape %v1345 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1347 = stablehlo.reshape %v1346 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1348 = stablehlo.convolution(%v1347, %u13qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x7x7xf32>, tensor<256x1x3x3xf32>) -> tensor<32x256x7x7xf32>
    %v1349 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1350 = stablehlo.add %v1348, %v1349 : tensor<32x256x7x7xf32>
    %v1351 = stablehlo.reshape %v1350 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1352 = stablehlo.reshape %v1351 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1353 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1354 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1355 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1356 = stablehlo.reduce(%v1352 init: %v1353) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1357 = stablehlo.broadcast_in_dim %v1356, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1358 = stablehlo.divide %v1357, %v1354 : tensor<32x256x7x7xf32>
    %v1359 = stablehlo.subtract %v1352, %v1358 : tensor<32x256x7x7xf32>
    %v1360 = stablehlo.multiply %v1359, %v1359 : tensor<32x256x7x7xf32>
    %v1361 = stablehlo.reduce(%v1360 init: %v1353) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1362 = stablehlo.broadcast_in_dim %v1361, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1363 = stablehlo.divide %v1362, %v1354 : tensor<32x256x7x7xf32>
    %v1364 = stablehlo.add %v1363, %v1355 : tensor<32x256x7x7xf32>
    %v1365 = stablehlo.rsqrt %v1364 : tensor<32x256x7x7xf32>
    %v1366 = stablehlo.multiply %v1359, %v1365 : tensor<32x256x7x7xf32>
    %v1367 = stablehlo.broadcast_in_dim %u13qg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1368 = stablehlo.broadcast_in_dim %u13qbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1369 = stablehlo.multiply %v1366, %v1367 : tensor<32x256x7x7xf32>
    %v1370 = stablehlo.add %v1369, %v1368 : tensor<32x256x7x7xf32>
    %v1371 = stablehlo.reshape %v1370 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1372 = stablehlo.reshape %v1371 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1373 = stablehlo.constant dense<0.0> : tensor<32x256x7x7xf32>
    %v1374 = stablehlo.maximum %v1372, %v1373 : tensor<32x256x7x7xf32>
    %v1375 = stablehlo.reshape %v1374 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1376 = stablehlo.reshape %v1375 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1377 = stablehlo.convolution(%v1376, %u13eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v1378 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1379 = stablehlo.add %v1377, %v1378 : tensor<32x1024x7x7xf32>
    %v1380 = stablehlo.reshape %v1379 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1381 = stablehlo.reshape %v1380 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1382 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1383 = stablehlo.constant dense<1568.0> : tensor<32x1024x7x7xf32>
    %v1384 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1385 = stablehlo.reduce(%v1381 init: %v1382) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1386 = stablehlo.broadcast_in_dim %v1385, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1387 = stablehlo.divide %v1386, %v1383 : tensor<32x1024x7x7xf32>
    %v1388 = stablehlo.subtract %v1381, %v1387 : tensor<32x1024x7x7xf32>
    %v1389 = stablehlo.multiply %v1388, %v1388 : tensor<32x1024x7x7xf32>
    %v1390 = stablehlo.reduce(%v1389 init: %v1382) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1391 = stablehlo.broadcast_in_dim %v1390, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1392 = stablehlo.divide %v1391, %v1383 : tensor<32x1024x7x7xf32>
    %v1393 = stablehlo.add %v1392, %v1384 : tensor<32x1024x7x7xf32>
    %v1394 = stablehlo.rsqrt %v1393 : tensor<32x1024x7x7xf32>
    %v1395 = stablehlo.multiply %v1388, %v1394 : tensor<32x1024x7x7xf32>
    %v1396 = stablehlo.broadcast_in_dim %u13eg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1397 = stablehlo.broadcast_in_dim %u13ebt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1398 = stablehlo.multiply %v1395, %v1396 : tensor<32x1024x7x7xf32>
    %v1399 = stablehlo.add %v1398, %v1397 : tensor<32x1024x7x7xf32>
    %v1400 = stablehlo.reshape %v1399 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1401 = stablehlo.reshape %v1400 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1402 = stablehlo.constant dense<0.0> : tensor<32x1024x7x7xf32>
    %v1403 = stablehlo.maximum %v1401, %v1402 : tensor<32x1024x7x7xf32>
    %v1404 = stablehlo.reshape %v1403 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1405 = stablehlo.reshape %v1404 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1406 = stablehlo.convolution(%v1405, %u13dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<32x1024x7x7xf32>, tensor<1024x1x5x5xf32>) -> tensor<32x1024x7x7xf32>
    %v1407 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1408 = stablehlo.add %v1406, %v1407 : tensor<32x1024x7x7xf32>
    %v1409 = stablehlo.reshape %v1408 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1410 = stablehlo.reshape %v1409 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1411 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1412 = stablehlo.constant dense<1568.0> : tensor<32x1024x7x7xf32>
    %v1413 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1414 = stablehlo.reduce(%v1410 init: %v1411) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1415 = stablehlo.broadcast_in_dim %v1414, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1416 = stablehlo.divide %v1415, %v1412 : tensor<32x1024x7x7xf32>
    %v1417 = stablehlo.subtract %v1410, %v1416 : tensor<32x1024x7x7xf32>
    %v1418 = stablehlo.multiply %v1417, %v1417 : tensor<32x1024x7x7xf32>
    %v1419 = stablehlo.reduce(%v1418 init: %v1411) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1420 = stablehlo.broadcast_in_dim %v1419, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1421 = stablehlo.divide %v1420, %v1412 : tensor<32x1024x7x7xf32>
    %v1422 = stablehlo.add %v1421, %v1413 : tensor<32x1024x7x7xf32>
    %v1423 = stablehlo.rsqrt %v1422 : tensor<32x1024x7x7xf32>
    %v1424 = stablehlo.multiply %v1417, %v1423 : tensor<32x1024x7x7xf32>
    %v1425 = stablehlo.broadcast_in_dim %u13dg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1426 = stablehlo.broadcast_in_dim %u13dbt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1427 = stablehlo.multiply %v1424, %v1425 : tensor<32x1024x7x7xf32>
    %v1428 = stablehlo.add %v1427, %v1426 : tensor<32x1024x7x7xf32>
    %v1429 = stablehlo.reshape %v1428 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1430 = stablehlo.reshape %v1429 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1431 = stablehlo.constant dense<0.0> : tensor<32x1024x7x7xf32>
    %v1432 = stablehlo.maximum %v1430, %v1431 : tensor<32x1024x7x7xf32>
    %v1433 = stablehlo.reshape %v1432 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1434 = stablehlo.reshape %v1433 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1435 = stablehlo.convolution(%v1434, %u13pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1436 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1437 = stablehlo.add %v1435, %v1436 : tensor<32x256x7x7xf32>
    %v1438 = stablehlo.reshape %v1437 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1439 = stablehlo.reshape %v1438 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1440 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1441 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1442 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1443 = stablehlo.reduce(%v1439 init: %v1440) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1444 = stablehlo.broadcast_in_dim %v1443, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1445 = stablehlo.divide %v1444, %v1441 : tensor<32x256x7x7xf32>
    %v1446 = stablehlo.subtract %v1439, %v1445 : tensor<32x256x7x7xf32>
    %v1447 = stablehlo.multiply %v1446, %v1446 : tensor<32x256x7x7xf32>
    %v1448 = stablehlo.reduce(%v1447 init: %v1440) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1449 = stablehlo.broadcast_in_dim %v1448, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1450 = stablehlo.divide %v1449, %v1441 : tensor<32x256x7x7xf32>
    %v1451 = stablehlo.add %v1450, %v1442 : tensor<32x256x7x7xf32>
    %v1452 = stablehlo.rsqrt %v1451 : tensor<32x256x7x7xf32>
    %v1453 = stablehlo.multiply %v1446, %v1452 : tensor<32x256x7x7xf32>
    %v1454 = stablehlo.broadcast_in_dim %u13pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1455 = stablehlo.broadcast_in_dim %u13pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1456 = stablehlo.multiply %v1453, %v1454 : tensor<32x256x7x7xf32>
    %v1457 = stablehlo.add %v1456, %v1455 : tensor<32x256x7x7xf32>
    %v1458 = stablehlo.reshape %v1457 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1459 = stablehlo.reshape %v1458 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1460 = stablehlo.reshape %v1346 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1461 = stablehlo.add %v1459, %v1460 : tensor<32x256x7x7xf32>
    %v1462 = stablehlo.reshape %v1461 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1463 = stablehlo.reshape %v1462 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1464 = stablehlo.convolution(%v1463, %u14qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x7x7xf32>, tensor<256x1x3x3xf32>) -> tensor<32x256x7x7xf32>
    %v1465 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1466 = stablehlo.add %v1464, %v1465 : tensor<32x256x7x7xf32>
    %v1467 = stablehlo.reshape %v1466 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1468 = stablehlo.reshape %v1467 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1469 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1470 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1471 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1472 = stablehlo.reduce(%v1468 init: %v1469) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1473 = stablehlo.broadcast_in_dim %v1472, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1474 = stablehlo.divide %v1473, %v1470 : tensor<32x256x7x7xf32>
    %v1475 = stablehlo.subtract %v1468, %v1474 : tensor<32x256x7x7xf32>
    %v1476 = stablehlo.multiply %v1475, %v1475 : tensor<32x256x7x7xf32>
    %v1477 = stablehlo.reduce(%v1476 init: %v1469) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1478 = stablehlo.broadcast_in_dim %v1477, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1479 = stablehlo.divide %v1478, %v1470 : tensor<32x256x7x7xf32>
    %v1480 = stablehlo.add %v1479, %v1471 : tensor<32x256x7x7xf32>
    %v1481 = stablehlo.rsqrt %v1480 : tensor<32x256x7x7xf32>
    %v1482 = stablehlo.multiply %v1475, %v1481 : tensor<32x256x7x7xf32>
    %v1483 = stablehlo.broadcast_in_dim %u14qg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1484 = stablehlo.broadcast_in_dim %u14qbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1485 = stablehlo.multiply %v1482, %v1483 : tensor<32x256x7x7xf32>
    %v1486 = stablehlo.add %v1485, %v1484 : tensor<32x256x7x7xf32>
    %v1487 = stablehlo.reshape %v1486 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1488 = stablehlo.reshape %v1487 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1489 = stablehlo.constant dense<0.0> : tensor<32x256x7x7xf32>
    %v1490 = stablehlo.maximum %v1488, %v1489 : tensor<32x256x7x7xf32>
    %v1491 = stablehlo.reshape %v1490 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1492 = stablehlo.reshape %v1491 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1493 = stablehlo.convolution(%v1492, %u14eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v1494 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1495 = stablehlo.add %v1493, %v1494 : tensor<32x1024x7x7xf32>
    %v1496 = stablehlo.reshape %v1495 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1497 = stablehlo.reshape %v1496 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1498 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1499 = stablehlo.constant dense<1568.0> : tensor<32x1024x7x7xf32>
    %v1500 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1501 = stablehlo.reduce(%v1497 init: %v1498) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1502 = stablehlo.broadcast_in_dim %v1501, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1503 = stablehlo.divide %v1502, %v1499 : tensor<32x1024x7x7xf32>
    %v1504 = stablehlo.subtract %v1497, %v1503 : tensor<32x1024x7x7xf32>
    %v1505 = stablehlo.multiply %v1504, %v1504 : tensor<32x1024x7x7xf32>
    %v1506 = stablehlo.reduce(%v1505 init: %v1498) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1507 = stablehlo.broadcast_in_dim %v1506, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1508 = stablehlo.divide %v1507, %v1499 : tensor<32x1024x7x7xf32>
    %v1509 = stablehlo.add %v1508, %v1500 : tensor<32x1024x7x7xf32>
    %v1510 = stablehlo.rsqrt %v1509 : tensor<32x1024x7x7xf32>
    %v1511 = stablehlo.multiply %v1504, %v1510 : tensor<32x1024x7x7xf32>
    %v1512 = stablehlo.broadcast_in_dim %u14eg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1513 = stablehlo.broadcast_in_dim %u14ebt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1514 = stablehlo.multiply %v1511, %v1512 : tensor<32x1024x7x7xf32>
    %v1515 = stablehlo.add %v1514, %v1513 : tensor<32x1024x7x7xf32>
    %v1516 = stablehlo.reshape %v1515 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1517 = stablehlo.reshape %v1516 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1518 = stablehlo.constant dense<0.0> : tensor<32x1024x7x7xf32>
    %v1519 = stablehlo.maximum %v1517, %v1518 : tensor<32x1024x7x7xf32>
    %v1520 = stablehlo.reshape %v1519 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1521 = stablehlo.reshape %v1520 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1522 = stablehlo.convolution(%v1521, %u14dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<32x1024x7x7xf32>, tensor<1024x1x5x5xf32>) -> tensor<32x1024x7x7xf32>
    %v1523 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1524 = stablehlo.add %v1522, %v1523 : tensor<32x1024x7x7xf32>
    %v1525 = stablehlo.reshape %v1524 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1526 = stablehlo.reshape %v1525 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1527 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1528 = stablehlo.constant dense<1568.0> : tensor<32x1024x7x7xf32>
    %v1529 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1530 = stablehlo.reduce(%v1526 init: %v1527) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1531 = stablehlo.broadcast_in_dim %v1530, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1532 = stablehlo.divide %v1531, %v1528 : tensor<32x1024x7x7xf32>
    %v1533 = stablehlo.subtract %v1526, %v1532 : tensor<32x1024x7x7xf32>
    %v1534 = stablehlo.multiply %v1533, %v1533 : tensor<32x1024x7x7xf32>
    %v1535 = stablehlo.reduce(%v1534 init: %v1527) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1536 = stablehlo.broadcast_in_dim %v1535, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1537 = stablehlo.divide %v1536, %v1528 : tensor<32x1024x7x7xf32>
    %v1538 = stablehlo.add %v1537, %v1529 : tensor<32x1024x7x7xf32>
    %v1539 = stablehlo.rsqrt %v1538 : tensor<32x1024x7x7xf32>
    %v1540 = stablehlo.multiply %v1533, %v1539 : tensor<32x1024x7x7xf32>
    %v1541 = stablehlo.broadcast_in_dim %u14dg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1542 = stablehlo.broadcast_in_dim %u14dbt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1543 = stablehlo.multiply %v1540, %v1541 : tensor<32x1024x7x7xf32>
    %v1544 = stablehlo.add %v1543, %v1542 : tensor<32x1024x7x7xf32>
    %v1545 = stablehlo.reshape %v1544 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1546 = stablehlo.reshape %v1545 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1547 = stablehlo.constant dense<0.0> : tensor<32x1024x7x7xf32>
    %v1548 = stablehlo.maximum %v1546, %v1547 : tensor<32x1024x7x7xf32>
    %v1549 = stablehlo.reshape %v1548 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1550 = stablehlo.reshape %v1549 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1551 = stablehlo.convolution(%v1550, %u14pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1552 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1553 = stablehlo.add %v1551, %v1552 : tensor<32x256x7x7xf32>
    %v1554 = stablehlo.reshape %v1553 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1555 = stablehlo.reshape %v1554 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1556 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1557 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1558 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1559 = stablehlo.reduce(%v1555 init: %v1556) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1560 = stablehlo.broadcast_in_dim %v1559, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1561 = stablehlo.divide %v1560, %v1557 : tensor<32x256x7x7xf32>
    %v1562 = stablehlo.subtract %v1555, %v1561 : tensor<32x256x7x7xf32>
    %v1563 = stablehlo.multiply %v1562, %v1562 : tensor<32x256x7x7xf32>
    %v1564 = stablehlo.reduce(%v1563 init: %v1556) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1565 = stablehlo.broadcast_in_dim %v1564, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1566 = stablehlo.divide %v1565, %v1557 : tensor<32x256x7x7xf32>
    %v1567 = stablehlo.add %v1566, %v1558 : tensor<32x256x7x7xf32>
    %v1568 = stablehlo.rsqrt %v1567 : tensor<32x256x7x7xf32>
    %v1569 = stablehlo.multiply %v1562, %v1568 : tensor<32x256x7x7xf32>
    %v1570 = stablehlo.broadcast_in_dim %u14pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1571 = stablehlo.broadcast_in_dim %u14pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1572 = stablehlo.multiply %v1569, %v1570 : tensor<32x256x7x7xf32>
    %v1573 = stablehlo.add %v1572, %v1571 : tensor<32x256x7x7xf32>
    %v1574 = stablehlo.reshape %v1573 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1575 = stablehlo.reshape %v1574 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1576 = stablehlo.reshape %v1462 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1577 = stablehlo.add %v1575, %v1576 : tensor<32x256x7x7xf32>
    %v1578 = stablehlo.reshape %v1577 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1579 = stablehlo.reshape %v1578 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1580 = stablehlo.convolution(%v1579, %u15eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v1581 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1582 = stablehlo.add %v1580, %v1581 : tensor<32x1024x7x7xf32>
    %v1583 = stablehlo.reshape %v1582 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1584 = stablehlo.reshape %v1583 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1585 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1586 = stablehlo.constant dense<1568.0> : tensor<32x1024x7x7xf32>
    %v1587 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1588 = stablehlo.reduce(%v1584 init: %v1585) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1589 = stablehlo.broadcast_in_dim %v1588, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1590 = stablehlo.divide %v1589, %v1586 : tensor<32x1024x7x7xf32>
    %v1591 = stablehlo.subtract %v1584, %v1590 : tensor<32x1024x7x7xf32>
    %v1592 = stablehlo.multiply %v1591, %v1591 : tensor<32x1024x7x7xf32>
    %v1593 = stablehlo.reduce(%v1592 init: %v1585) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1594 = stablehlo.broadcast_in_dim %v1593, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1595 = stablehlo.divide %v1594, %v1586 : tensor<32x1024x7x7xf32>
    %v1596 = stablehlo.add %v1595, %v1587 : tensor<32x1024x7x7xf32>
    %v1597 = stablehlo.rsqrt %v1596 : tensor<32x1024x7x7xf32>
    %v1598 = stablehlo.multiply %v1591, %v1597 : tensor<32x1024x7x7xf32>
    %v1599 = stablehlo.broadcast_in_dim %u15eg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1600 = stablehlo.broadcast_in_dim %u15ebt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1601 = stablehlo.multiply %v1598, %v1599 : tensor<32x1024x7x7xf32>
    %v1602 = stablehlo.add %v1601, %v1600 : tensor<32x1024x7x7xf32>
    %v1603 = stablehlo.reshape %v1602 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1604 = stablehlo.reshape %v1603 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1605 = stablehlo.constant dense<0.0> : tensor<32x1024x7x7xf32>
    %v1606 = stablehlo.maximum %v1604, %v1605 : tensor<32x1024x7x7xf32>
    %v1607 = stablehlo.reshape %v1606 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1608 = stablehlo.reshape %v1607 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1609 = stablehlo.convolution(%v1608, %u15pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1610 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1611 = stablehlo.add %v1609, %v1610 : tensor<32x256x7x7xf32>
    %v1612 = stablehlo.reshape %v1611 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1613 = stablehlo.reshape %v1612 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1614 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1615 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1616 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1617 = stablehlo.reduce(%v1613 init: %v1614) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1618 = stablehlo.broadcast_in_dim %v1617, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1619 = stablehlo.divide %v1618, %v1615 : tensor<32x256x7x7xf32>
    %v1620 = stablehlo.subtract %v1613, %v1619 : tensor<32x256x7x7xf32>
    %v1621 = stablehlo.multiply %v1620, %v1620 : tensor<32x256x7x7xf32>
    %v1622 = stablehlo.reduce(%v1621 init: %v1614) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1623 = stablehlo.broadcast_in_dim %v1622, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1624 = stablehlo.divide %v1623, %v1615 : tensor<32x256x7x7xf32>
    %v1625 = stablehlo.add %v1624, %v1616 : tensor<32x256x7x7xf32>
    %v1626 = stablehlo.rsqrt %v1625 : tensor<32x256x7x7xf32>
    %v1627 = stablehlo.multiply %v1620, %v1626 : tensor<32x256x7x7xf32>
    %v1628 = stablehlo.broadcast_in_dim %u15pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1629 = stablehlo.broadcast_in_dim %u15pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1630 = stablehlo.multiply %v1627, %v1628 : tensor<32x256x7x7xf32>
    %v1631 = stablehlo.add %v1630, %v1629 : tensor<32x256x7x7xf32>
    %v1632 = stablehlo.reshape %v1631 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1633 = stablehlo.reshape %v1632 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1634 = stablehlo.reshape %v1578 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1635 = stablehlo.add %v1633, %v1634 : tensor<32x256x7x7xf32>
    %v1636 = stablehlo.reshape %v1635 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1637 = stablehlo.reshape %v1636 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1638 = stablehlo.convolution(%v1637, %u16qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x7x7xf32>, tensor<256x1x3x3xf32>) -> tensor<32x256x7x7xf32>
    %v1639 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1640 = stablehlo.add %v1638, %v1639 : tensor<32x256x7x7xf32>
    %v1641 = stablehlo.reshape %v1640 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1642 = stablehlo.reshape %v1641 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1643 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1644 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1645 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1646 = stablehlo.reduce(%v1642 init: %v1643) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1647 = stablehlo.broadcast_in_dim %v1646, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1648 = stablehlo.divide %v1647, %v1644 : tensor<32x256x7x7xf32>
    %v1649 = stablehlo.subtract %v1642, %v1648 : tensor<32x256x7x7xf32>
    %v1650 = stablehlo.multiply %v1649, %v1649 : tensor<32x256x7x7xf32>
    %v1651 = stablehlo.reduce(%v1650 init: %v1643) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1652 = stablehlo.broadcast_in_dim %v1651, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1653 = stablehlo.divide %v1652, %v1644 : tensor<32x256x7x7xf32>
    %v1654 = stablehlo.add %v1653, %v1645 : tensor<32x256x7x7xf32>
    %v1655 = stablehlo.rsqrt %v1654 : tensor<32x256x7x7xf32>
    %v1656 = stablehlo.multiply %v1649, %v1655 : tensor<32x256x7x7xf32>
    %v1657 = stablehlo.broadcast_in_dim %u16qg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1658 = stablehlo.broadcast_in_dim %u16qbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1659 = stablehlo.multiply %v1656, %v1657 : tensor<32x256x7x7xf32>
    %v1660 = stablehlo.add %v1659, %v1658 : tensor<32x256x7x7xf32>
    %v1661 = stablehlo.reshape %v1660 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1662 = stablehlo.reshape %v1661 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1663 = stablehlo.constant dense<0.0> : tensor<32x256x7x7xf32>
    %v1664 = stablehlo.maximum %v1662, %v1663 : tensor<32x256x7x7xf32>
    %v1665 = stablehlo.reshape %v1664 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1666 = stablehlo.reshape %v1665 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1667 = stablehlo.convolution(%v1666, %u16eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v1668 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1669 = stablehlo.add %v1667, %v1668 : tensor<32x1024x7x7xf32>
    %v1670 = stablehlo.reshape %v1669 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1671 = stablehlo.reshape %v1670 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1672 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1673 = stablehlo.constant dense<1568.0> : tensor<32x1024x7x7xf32>
    %v1674 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1675 = stablehlo.reduce(%v1671 init: %v1672) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1676 = stablehlo.broadcast_in_dim %v1675, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1677 = stablehlo.divide %v1676, %v1673 : tensor<32x1024x7x7xf32>
    %v1678 = stablehlo.subtract %v1671, %v1677 : tensor<32x1024x7x7xf32>
    %v1679 = stablehlo.multiply %v1678, %v1678 : tensor<32x1024x7x7xf32>
    %v1680 = stablehlo.reduce(%v1679 init: %v1672) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1681 = stablehlo.broadcast_in_dim %v1680, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1682 = stablehlo.divide %v1681, %v1673 : tensor<32x1024x7x7xf32>
    %v1683 = stablehlo.add %v1682, %v1674 : tensor<32x1024x7x7xf32>
    %v1684 = stablehlo.rsqrt %v1683 : tensor<32x1024x7x7xf32>
    %v1685 = stablehlo.multiply %v1678, %v1684 : tensor<32x1024x7x7xf32>
    %v1686 = stablehlo.broadcast_in_dim %u16eg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1687 = stablehlo.broadcast_in_dim %u16ebt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1688 = stablehlo.multiply %v1685, %v1686 : tensor<32x1024x7x7xf32>
    %v1689 = stablehlo.add %v1688, %v1687 : tensor<32x1024x7x7xf32>
    %v1690 = stablehlo.reshape %v1689 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1691 = stablehlo.reshape %v1690 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1692 = stablehlo.constant dense<0.0> : tensor<32x1024x7x7xf32>
    %v1693 = stablehlo.maximum %v1691, %v1692 : tensor<32x1024x7x7xf32>
    %v1694 = stablehlo.reshape %v1693 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1695 = stablehlo.reshape %v1694 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1696 = stablehlo.convolution(%v1695, %u16pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1697 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1698 = stablehlo.add %v1696, %v1697 : tensor<32x256x7x7xf32>
    %v1699 = stablehlo.reshape %v1698 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1700 = stablehlo.reshape %v1699 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1701 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1702 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1703 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1704 = stablehlo.reduce(%v1700 init: %v1701) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1705 = stablehlo.broadcast_in_dim %v1704, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1706 = stablehlo.divide %v1705, %v1702 : tensor<32x256x7x7xf32>
    %v1707 = stablehlo.subtract %v1700, %v1706 : tensor<32x256x7x7xf32>
    %v1708 = stablehlo.multiply %v1707, %v1707 : tensor<32x256x7x7xf32>
    %v1709 = stablehlo.reduce(%v1708 init: %v1701) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1710 = stablehlo.broadcast_in_dim %v1709, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1711 = stablehlo.divide %v1710, %v1702 : tensor<32x256x7x7xf32>
    %v1712 = stablehlo.add %v1711, %v1703 : tensor<32x256x7x7xf32>
    %v1713 = stablehlo.rsqrt %v1712 : tensor<32x256x7x7xf32>
    %v1714 = stablehlo.multiply %v1707, %v1713 : tensor<32x256x7x7xf32>
    %v1715 = stablehlo.broadcast_in_dim %u16pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1716 = stablehlo.broadcast_in_dim %u16pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1717 = stablehlo.multiply %v1714, %v1715 : tensor<32x256x7x7xf32>
    %v1718 = stablehlo.add %v1717, %v1716 : tensor<32x256x7x7xf32>
    %v1719 = stablehlo.reshape %v1718 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1720 = stablehlo.reshape %v1719 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1721 = stablehlo.reshape %v1636 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1722 = stablehlo.add %v1720, %v1721 : tensor<32x256x7x7xf32>
    %v1723 = stablehlo.reshape %v1722 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1724 = stablehlo.reshape %v1723 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1725 = stablehlo.convolution(%v1724, %u17qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x7x7xf32>, tensor<256x1x3x3xf32>) -> tensor<32x256x7x7xf32>
    %v1726 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1727 = stablehlo.add %v1725, %v1726 : tensor<32x256x7x7xf32>
    %v1728 = stablehlo.reshape %v1727 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1729 = stablehlo.reshape %v1728 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1730 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1731 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1732 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1733 = stablehlo.reduce(%v1729 init: %v1730) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1734 = stablehlo.broadcast_in_dim %v1733, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1735 = stablehlo.divide %v1734, %v1731 : tensor<32x256x7x7xf32>
    %v1736 = stablehlo.subtract %v1729, %v1735 : tensor<32x256x7x7xf32>
    %v1737 = stablehlo.multiply %v1736, %v1736 : tensor<32x256x7x7xf32>
    %v1738 = stablehlo.reduce(%v1737 init: %v1730) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1739 = stablehlo.broadcast_in_dim %v1738, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1740 = stablehlo.divide %v1739, %v1731 : tensor<32x256x7x7xf32>
    %v1741 = stablehlo.add %v1740, %v1732 : tensor<32x256x7x7xf32>
    %v1742 = stablehlo.rsqrt %v1741 : tensor<32x256x7x7xf32>
    %v1743 = stablehlo.multiply %v1736, %v1742 : tensor<32x256x7x7xf32>
    %v1744 = stablehlo.broadcast_in_dim %u17qg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1745 = stablehlo.broadcast_in_dim %u17qbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1746 = stablehlo.multiply %v1743, %v1744 : tensor<32x256x7x7xf32>
    %v1747 = stablehlo.add %v1746, %v1745 : tensor<32x256x7x7xf32>
    %v1748 = stablehlo.reshape %v1747 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1749 = stablehlo.reshape %v1748 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1750 = stablehlo.constant dense<0.0> : tensor<32x256x7x7xf32>
    %v1751 = stablehlo.maximum %v1749, %v1750 : tensor<32x256x7x7xf32>
    %v1752 = stablehlo.reshape %v1751 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1753 = stablehlo.reshape %v1752 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1754 = stablehlo.convolution(%v1753, %u17eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<512x256x1x1xf32>) -> tensor<32x512x7x7xf32>
    %v1755 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1756 = stablehlo.add %v1754, %v1755 : tensor<32x512x7x7xf32>
    %v1757 = stablehlo.reshape %v1756 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1758 = stablehlo.reshape %v1757 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1759 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1760 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1761 = stablehlo.constant dense<1.0e-5> : tensor<32x512x7x7xf32>
    %v1762 = stablehlo.reduce(%v1758 init: %v1759) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1763 = stablehlo.broadcast_in_dim %v1762, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1764 = stablehlo.divide %v1763, %v1760 : tensor<32x512x7x7xf32>
    %v1765 = stablehlo.subtract %v1758, %v1764 : tensor<32x512x7x7xf32>
    %v1766 = stablehlo.multiply %v1765, %v1765 : tensor<32x512x7x7xf32>
    %v1767 = stablehlo.reduce(%v1766 init: %v1759) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1768 = stablehlo.broadcast_in_dim %v1767, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1769 = stablehlo.divide %v1768, %v1760 : tensor<32x512x7x7xf32>
    %v1770 = stablehlo.add %v1769, %v1761 : tensor<32x512x7x7xf32>
    %v1771 = stablehlo.rsqrt %v1770 : tensor<32x512x7x7xf32>
    %v1772 = stablehlo.multiply %v1765, %v1771 : tensor<32x512x7x7xf32>
    %v1773 = stablehlo.broadcast_in_dim %u17eg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1774 = stablehlo.broadcast_in_dim %u17ebt, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1775 = stablehlo.multiply %v1772, %v1773 : tensor<32x512x7x7xf32>
    %v1776 = stablehlo.add %v1775, %v1774 : tensor<32x512x7x7xf32>
    %v1777 = stablehlo.reshape %v1776 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1778 = stablehlo.reshape %v1777 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1779 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1780 = stablehlo.maximum %v1778, %v1779 : tensor<32x512x7x7xf32>
    %v1781 = stablehlo.reshape %v1780 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1782 = stablehlo.reshape %v1781 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1783 = stablehlo.convolution(%v1782, %u17dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x7x7xf32>, tensor<512x1x5x5xf32>) -> tensor<32x512x7x7xf32>
    %v1784 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1785 = stablehlo.add %v1783, %v1784 : tensor<32x512x7x7xf32>
    %v1786 = stablehlo.reshape %v1785 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1787 = stablehlo.reshape %v1786 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1788 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1789 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1790 = stablehlo.constant dense<1.0e-5> : tensor<32x512x7x7xf32>
    %v1791 = stablehlo.reduce(%v1787 init: %v1788) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1792 = stablehlo.broadcast_in_dim %v1791, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1793 = stablehlo.divide %v1792, %v1789 : tensor<32x512x7x7xf32>
    %v1794 = stablehlo.subtract %v1787, %v1793 : tensor<32x512x7x7xf32>
    %v1795 = stablehlo.multiply %v1794, %v1794 : tensor<32x512x7x7xf32>
    %v1796 = stablehlo.reduce(%v1795 init: %v1788) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1797 = stablehlo.broadcast_in_dim %v1796, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1798 = stablehlo.divide %v1797, %v1789 : tensor<32x512x7x7xf32>
    %v1799 = stablehlo.add %v1798, %v1790 : tensor<32x512x7x7xf32>
    %v1800 = stablehlo.rsqrt %v1799 : tensor<32x512x7x7xf32>
    %v1801 = stablehlo.multiply %v1794, %v1800 : tensor<32x512x7x7xf32>
    %v1802 = stablehlo.broadcast_in_dim %u17dg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1803 = stablehlo.broadcast_in_dim %u17dbt, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1804 = stablehlo.multiply %v1801, %v1802 : tensor<32x512x7x7xf32>
    %v1805 = stablehlo.add %v1804, %v1803 : tensor<32x512x7x7xf32>
    %v1806 = stablehlo.reshape %v1805 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1807 = stablehlo.reshape %v1806 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1808 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1809 = stablehlo.maximum %v1807, %v1808 : tensor<32x512x7x7xf32>
    %v1810 = stablehlo.reshape %v1809 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1811 = stablehlo.reshape %v1810 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1812 = stablehlo.convolution(%v1811, %u17pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<256x512x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1813 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1814 = stablehlo.add %v1812, %v1813 : tensor<32x256x7x7xf32>
    %v1815 = stablehlo.reshape %v1814 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1816 = stablehlo.reshape %v1815 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1817 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1818 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1819 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1820 = stablehlo.reduce(%v1816 init: %v1817) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1821 = stablehlo.broadcast_in_dim %v1820, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1822 = stablehlo.divide %v1821, %v1818 : tensor<32x256x7x7xf32>
    %v1823 = stablehlo.subtract %v1816, %v1822 : tensor<32x256x7x7xf32>
    %v1824 = stablehlo.multiply %v1823, %v1823 : tensor<32x256x7x7xf32>
    %v1825 = stablehlo.reduce(%v1824 init: %v1817) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1826 = stablehlo.broadcast_in_dim %v1825, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1827 = stablehlo.divide %v1826, %v1818 : tensor<32x256x7x7xf32>
    %v1828 = stablehlo.add %v1827, %v1819 : tensor<32x256x7x7xf32>
    %v1829 = stablehlo.rsqrt %v1828 : tensor<32x256x7x7xf32>
    %v1830 = stablehlo.multiply %v1823, %v1829 : tensor<32x256x7x7xf32>
    %v1831 = stablehlo.broadcast_in_dim %u17pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1832 = stablehlo.broadcast_in_dim %u17pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1833 = stablehlo.multiply %v1830, %v1831 : tensor<32x256x7x7xf32>
    %v1834 = stablehlo.add %v1833, %v1832 : tensor<32x256x7x7xf32>
    %v1835 = stablehlo.reshape %v1834 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1836 = stablehlo.reshape %v1835 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1837 = stablehlo.reshape %v1723 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1838 = stablehlo.add %v1836, %v1837 : tensor<32x256x7x7xf32>
    %v1839 = stablehlo.reshape %v1838 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1840 = stablehlo.reshape %v1839 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1841 = stablehlo.convolution(%v1840, %u18qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x7x7xf32>, tensor<256x1x5x5xf32>) -> tensor<32x256x7x7xf32>
    %v1842 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1843 = stablehlo.add %v1841, %v1842 : tensor<32x256x7x7xf32>
    %v1844 = stablehlo.reshape %v1843 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1845 = stablehlo.reshape %v1844 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1846 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1847 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1848 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1849 = stablehlo.reduce(%v1845 init: %v1846) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1850 = stablehlo.broadcast_in_dim %v1849, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1851 = stablehlo.divide %v1850, %v1847 : tensor<32x256x7x7xf32>
    %v1852 = stablehlo.subtract %v1845, %v1851 : tensor<32x256x7x7xf32>
    %v1853 = stablehlo.multiply %v1852, %v1852 : tensor<32x256x7x7xf32>
    %v1854 = stablehlo.reduce(%v1853 init: %v1846) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1855 = stablehlo.broadcast_in_dim %v1854, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1856 = stablehlo.divide %v1855, %v1847 : tensor<32x256x7x7xf32>
    %v1857 = stablehlo.add %v1856, %v1848 : tensor<32x256x7x7xf32>
    %v1858 = stablehlo.rsqrt %v1857 : tensor<32x256x7x7xf32>
    %v1859 = stablehlo.multiply %v1852, %v1858 : tensor<32x256x7x7xf32>
    %v1860 = stablehlo.broadcast_in_dim %u18qg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1861 = stablehlo.broadcast_in_dim %u18qbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1862 = stablehlo.multiply %v1859, %v1860 : tensor<32x256x7x7xf32>
    %v1863 = stablehlo.add %v1862, %v1861 : tensor<32x256x7x7xf32>
    %v1864 = stablehlo.reshape %v1863 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1865 = stablehlo.reshape %v1864 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1866 = stablehlo.constant dense<0.0> : tensor<32x256x7x7xf32>
    %v1867 = stablehlo.maximum %v1865, %v1866 : tensor<32x256x7x7xf32>
    %v1868 = stablehlo.reshape %v1867 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1869 = stablehlo.reshape %v1868 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1870 = stablehlo.convolution(%v1869, %u18eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v1871 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1872 = stablehlo.add %v1870, %v1871 : tensor<32x1024x7x7xf32>
    %v1873 = stablehlo.reshape %v1872 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1874 = stablehlo.reshape %v1873 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1875 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1876 = stablehlo.constant dense<1568.0> : tensor<32x1024x7x7xf32>
    %v1877 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1878 = stablehlo.reduce(%v1874 init: %v1875) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1879 = stablehlo.broadcast_in_dim %v1878, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1880 = stablehlo.divide %v1879, %v1876 : tensor<32x1024x7x7xf32>
    %v1881 = stablehlo.subtract %v1874, %v1880 : tensor<32x1024x7x7xf32>
    %v1882 = stablehlo.multiply %v1881, %v1881 : tensor<32x1024x7x7xf32>
    %v1883 = stablehlo.reduce(%v1882 init: %v1875) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1884 = stablehlo.broadcast_in_dim %v1883, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1885 = stablehlo.divide %v1884, %v1876 : tensor<32x1024x7x7xf32>
    %v1886 = stablehlo.add %v1885, %v1877 : tensor<32x1024x7x7xf32>
    %v1887 = stablehlo.rsqrt %v1886 : tensor<32x1024x7x7xf32>
    %v1888 = stablehlo.multiply %v1881, %v1887 : tensor<32x1024x7x7xf32>
    %v1889 = stablehlo.broadcast_in_dim %u18eg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1890 = stablehlo.broadcast_in_dim %u18ebt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1891 = stablehlo.multiply %v1888, %v1889 : tensor<32x1024x7x7xf32>
    %v1892 = stablehlo.add %v1891, %v1890 : tensor<32x1024x7x7xf32>
    %v1893 = stablehlo.reshape %v1892 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1894 = stablehlo.reshape %v1893 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1895 = stablehlo.constant dense<0.0> : tensor<32x1024x7x7xf32>
    %v1896 = stablehlo.maximum %v1894, %v1895 : tensor<32x1024x7x7xf32>
    %v1897 = stablehlo.reshape %v1896 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1898 = stablehlo.reshape %v1897 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1899 = stablehlo.convolution(%v1898, %u18dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<32x1024x7x7xf32>, tensor<1024x1x5x5xf32>) -> tensor<32x1024x7x7xf32>
    %v1900 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1901 = stablehlo.add %v1899, %v1900 : tensor<32x1024x7x7xf32>
    %v1902 = stablehlo.reshape %v1901 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1903 = stablehlo.reshape %v1902 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1904 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1905 = stablehlo.constant dense<1568.0> : tensor<32x1024x7x7xf32>
    %v1906 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1907 = stablehlo.reduce(%v1903 init: %v1904) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1908 = stablehlo.broadcast_in_dim %v1907, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1909 = stablehlo.divide %v1908, %v1905 : tensor<32x1024x7x7xf32>
    %v1910 = stablehlo.subtract %v1903, %v1909 : tensor<32x1024x7x7xf32>
    %v1911 = stablehlo.multiply %v1910, %v1910 : tensor<32x1024x7x7xf32>
    %v1912 = stablehlo.reduce(%v1911 init: %v1904) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1913 = stablehlo.broadcast_in_dim %v1912, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1914 = stablehlo.divide %v1913, %v1905 : tensor<32x1024x7x7xf32>
    %v1915 = stablehlo.add %v1914, %v1906 : tensor<32x1024x7x7xf32>
    %v1916 = stablehlo.rsqrt %v1915 : tensor<32x1024x7x7xf32>
    %v1917 = stablehlo.multiply %v1910, %v1916 : tensor<32x1024x7x7xf32>
    %v1918 = stablehlo.broadcast_in_dim %u18dg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1919 = stablehlo.broadcast_in_dim %u18dbt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1920 = stablehlo.multiply %v1917, %v1918 : tensor<32x1024x7x7xf32>
    %v1921 = stablehlo.add %v1920, %v1919 : tensor<32x1024x7x7xf32>
    %v1922 = stablehlo.reshape %v1921 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1923 = stablehlo.reshape %v1922 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1924 = stablehlo.constant dense<0.0> : tensor<32x1024x7x7xf32>
    %v1925 = stablehlo.maximum %v1923, %v1924 : tensor<32x1024x7x7xf32>
    %v1926 = stablehlo.reshape %v1925 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1927 = stablehlo.reshape %v1926 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1928 = stablehlo.convolution(%v1927, %u18pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1929 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1930 = stablehlo.add %v1928, %v1929 : tensor<32x256x7x7xf32>
    %v1931 = stablehlo.reshape %v1930 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1932 = stablehlo.reshape %v1931 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1933 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1934 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1935 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1936 = stablehlo.reduce(%v1932 init: %v1933) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1937 = stablehlo.broadcast_in_dim %v1936, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1938 = stablehlo.divide %v1937, %v1934 : tensor<32x256x7x7xf32>
    %v1939 = stablehlo.subtract %v1932, %v1938 : tensor<32x256x7x7xf32>
    %v1940 = stablehlo.multiply %v1939, %v1939 : tensor<32x256x7x7xf32>
    %v1941 = stablehlo.reduce(%v1940 init: %v1933) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1942 = stablehlo.broadcast_in_dim %v1941, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1943 = stablehlo.divide %v1942, %v1934 : tensor<32x256x7x7xf32>
    %v1944 = stablehlo.add %v1943, %v1935 : tensor<32x256x7x7xf32>
    %v1945 = stablehlo.rsqrt %v1944 : tensor<32x256x7x7xf32>
    %v1946 = stablehlo.multiply %v1939, %v1945 : tensor<32x256x7x7xf32>
    %v1947 = stablehlo.broadcast_in_dim %u18pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1948 = stablehlo.broadcast_in_dim %u18pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1949 = stablehlo.multiply %v1946, %v1947 : tensor<32x256x7x7xf32>
    %v1950 = stablehlo.add %v1949, %v1948 : tensor<32x256x7x7xf32>
    %v1951 = stablehlo.reshape %v1950 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1952 = stablehlo.reshape %v1951 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1953 = stablehlo.reshape %v1839 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1954 = stablehlo.add %v1952, %v1953 : tensor<32x256x7x7xf32>
    %v1955 = stablehlo.reshape %v1954 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1956 = stablehlo.reshape %v1955 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1957 = stablehlo.convolution(%v1956, %u19eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v1958 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1959 = stablehlo.add %v1957, %v1958 : tensor<32x1024x7x7xf32>
    %v1960 = stablehlo.reshape %v1959 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1961 = stablehlo.reshape %v1960 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1962 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1963 = stablehlo.constant dense<1568.0> : tensor<32x1024x7x7xf32>
    %v1964 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1965 = stablehlo.reduce(%v1961 init: %v1962) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1966 = stablehlo.broadcast_in_dim %v1965, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1967 = stablehlo.divide %v1966, %v1963 : tensor<32x1024x7x7xf32>
    %v1968 = stablehlo.subtract %v1961, %v1967 : tensor<32x1024x7x7xf32>
    %v1969 = stablehlo.multiply %v1968, %v1968 : tensor<32x1024x7x7xf32>
    %v1970 = stablehlo.reduce(%v1969 init: %v1962) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1971 = stablehlo.broadcast_in_dim %v1970, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1972 = stablehlo.divide %v1971, %v1963 : tensor<32x1024x7x7xf32>
    %v1973 = stablehlo.add %v1972, %v1964 : tensor<32x1024x7x7xf32>
    %v1974 = stablehlo.rsqrt %v1973 : tensor<32x1024x7x7xf32>
    %v1975 = stablehlo.multiply %v1968, %v1974 : tensor<32x1024x7x7xf32>
    %v1976 = stablehlo.broadcast_in_dim %u19eg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1977 = stablehlo.broadcast_in_dim %u19ebt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1978 = stablehlo.multiply %v1975, %v1976 : tensor<32x1024x7x7xf32>
    %v1979 = stablehlo.add %v1978, %v1977 : tensor<32x1024x7x7xf32>
    %v1980 = stablehlo.reshape %v1979 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1981 = stablehlo.reshape %v1980 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1982 = stablehlo.constant dense<0.0> : tensor<32x1024x7x7xf32>
    %v1983 = stablehlo.maximum %v1981, %v1982 : tensor<32x1024x7x7xf32>
    %v1984 = stablehlo.reshape %v1983 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1985 = stablehlo.reshape %v1984 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1986 = stablehlo.convolution(%v1985, %u19pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1987 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1988 = stablehlo.add %v1986, %v1987 : tensor<32x256x7x7xf32>
    %v1989 = stablehlo.reshape %v1988 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1990 = stablehlo.reshape %v1989 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1991 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1992 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1993 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1994 = stablehlo.reduce(%v1990 init: %v1991) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1995 = stablehlo.broadcast_in_dim %v1994, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1996 = stablehlo.divide %v1995, %v1992 : tensor<32x256x7x7xf32>
    %v1997 = stablehlo.subtract %v1990, %v1996 : tensor<32x256x7x7xf32>
    %v1998 = stablehlo.multiply %v1997, %v1997 : tensor<32x256x7x7xf32>
    %v1999 = stablehlo.reduce(%v1998 init: %v1991) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v2000 = stablehlo.broadcast_in_dim %v1999, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v2001 = stablehlo.divide %v2000, %v1992 : tensor<32x256x7x7xf32>
    %v2002 = stablehlo.add %v2001, %v1993 : tensor<32x256x7x7xf32>
    %v2003 = stablehlo.rsqrt %v2002 : tensor<32x256x7x7xf32>
    %v2004 = stablehlo.multiply %v1997, %v2003 : tensor<32x256x7x7xf32>
    %v2005 = stablehlo.broadcast_in_dim %u19pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v2006 = stablehlo.broadcast_in_dim %u19pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v2007 = stablehlo.multiply %v2004, %v2005 : tensor<32x256x7x7xf32>
    %v2008 = stablehlo.add %v2007, %v2006 : tensor<32x256x7x7xf32>
    %v2009 = stablehlo.reshape %v2008 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v2010 = stablehlo.reshape %v2009 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v2011 = stablehlo.reshape %v1955 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v2012 = stablehlo.add %v2010, %v2011 : tensor<32x256x7x7xf32>
    %v2013 = stablehlo.reshape %v2012 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v2014 = stablehlo.reshape %v2013 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v2015 = stablehlo.convolution(%v2014, %u20eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v2016 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v2017 = stablehlo.add %v2015, %v2016 : tensor<32x1024x7x7xf32>
    %v2018 = stablehlo.reshape %v2017 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v2019 = stablehlo.reshape %v2018 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2020 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2021 = stablehlo.constant dense<1568.0> : tensor<32x1024x7x7xf32>
    %v2022 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v2023 = stablehlo.reduce(%v2019 init: %v2020) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2024 = stablehlo.broadcast_in_dim %v2023, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v2025 = stablehlo.divide %v2024, %v2021 : tensor<32x1024x7x7xf32>
    %v2026 = stablehlo.subtract %v2019, %v2025 : tensor<32x1024x7x7xf32>
    %v2027 = stablehlo.multiply %v2026, %v2026 : tensor<32x1024x7x7xf32>
    %v2028 = stablehlo.reduce(%v2027 init: %v2020) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2029 = stablehlo.broadcast_in_dim %v2028, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v2030 = stablehlo.divide %v2029, %v2021 : tensor<32x1024x7x7xf32>
    %v2031 = stablehlo.add %v2030, %v2022 : tensor<32x1024x7x7xf32>
    %v2032 = stablehlo.rsqrt %v2031 : tensor<32x1024x7x7xf32>
    %v2033 = stablehlo.multiply %v2026, %v2032 : tensor<32x1024x7x7xf32>
    %v2034 = stablehlo.broadcast_in_dim %u20eg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v2035 = stablehlo.broadcast_in_dim %u20ebt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v2036 = stablehlo.multiply %v2033, %v2034 : tensor<32x1024x7x7xf32>
    %v2037 = stablehlo.add %v2036, %v2035 : tensor<32x1024x7x7xf32>
    %v2038 = stablehlo.reshape %v2037 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v2039 = stablehlo.reshape %v2038 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2040 = stablehlo.constant dense<0.0> : tensor<32x1024x7x7xf32>
    %v2041 = stablehlo.maximum %v2039, %v2040 : tensor<32x1024x7x7xf32>
    %v2042 = stablehlo.reshape %v2041 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v2043 = stablehlo.reshape %v2042 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v2044 = stablehlo.convolution(%v2043, %u20pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v2045 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v2046 = stablehlo.add %v2044, %v2045 : tensor<32x256x7x7xf32>
    %v2047 = stablehlo.reshape %v2046 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v2048 = stablehlo.reshape %v2047 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v2049 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2050 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v2051 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v2052 = stablehlo.reduce(%v2048 init: %v2049) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v2053 = stablehlo.broadcast_in_dim %v2052, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v2054 = stablehlo.divide %v2053, %v2050 : tensor<32x256x7x7xf32>
    %v2055 = stablehlo.subtract %v2048, %v2054 : tensor<32x256x7x7xf32>
    %v2056 = stablehlo.multiply %v2055, %v2055 : tensor<32x256x7x7xf32>
    %v2057 = stablehlo.reduce(%v2056 init: %v2049) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v2058 = stablehlo.broadcast_in_dim %v2057, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v2059 = stablehlo.divide %v2058, %v2050 : tensor<32x256x7x7xf32>
    %v2060 = stablehlo.add %v2059, %v2051 : tensor<32x256x7x7xf32>
    %v2061 = stablehlo.rsqrt %v2060 : tensor<32x256x7x7xf32>
    %v2062 = stablehlo.multiply %v2055, %v2061 : tensor<32x256x7x7xf32>
    %v2063 = stablehlo.broadcast_in_dim %u20pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v2064 = stablehlo.broadcast_in_dim %u20pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v2065 = stablehlo.multiply %v2062, %v2063 : tensor<32x256x7x7xf32>
    %v2066 = stablehlo.add %v2065, %v2064 : tensor<32x256x7x7xf32>
    %v2067 = stablehlo.reshape %v2066 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v2068 = stablehlo.reshape %v2067 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v2069 = stablehlo.reshape %v2013 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v2070 = stablehlo.add %v2068, %v2069 : tensor<32x256x7x7xf32>
    %v2071 = stablehlo.reshape %v2070 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v2072 = stablehlo.reshape %v2071 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v2073 = stablehlo.convolution(%v2072, %u21qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x7x7xf32>, tensor<256x1x5x5xf32>) -> tensor<32x256x7x7xf32>
    %v2074 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v2075 = stablehlo.add %v2073, %v2074 : tensor<32x256x7x7xf32>
    %v2076 = stablehlo.reshape %v2075 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v2077 = stablehlo.reshape %v2076 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v2078 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2079 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v2080 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v2081 = stablehlo.reduce(%v2077 init: %v2078) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v2082 = stablehlo.broadcast_in_dim %v2081, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v2083 = stablehlo.divide %v2082, %v2079 : tensor<32x256x7x7xf32>
    %v2084 = stablehlo.subtract %v2077, %v2083 : tensor<32x256x7x7xf32>
    %v2085 = stablehlo.multiply %v2084, %v2084 : tensor<32x256x7x7xf32>
    %v2086 = stablehlo.reduce(%v2085 init: %v2078) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v2087 = stablehlo.broadcast_in_dim %v2086, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v2088 = stablehlo.divide %v2087, %v2079 : tensor<32x256x7x7xf32>
    %v2089 = stablehlo.add %v2088, %v2080 : tensor<32x256x7x7xf32>
    %v2090 = stablehlo.rsqrt %v2089 : tensor<32x256x7x7xf32>
    %v2091 = stablehlo.multiply %v2084, %v2090 : tensor<32x256x7x7xf32>
    %v2092 = stablehlo.broadcast_in_dim %u21qg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v2093 = stablehlo.broadcast_in_dim %u21qbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v2094 = stablehlo.multiply %v2091, %v2092 : tensor<32x256x7x7xf32>
    %v2095 = stablehlo.add %v2094, %v2093 : tensor<32x256x7x7xf32>
    %v2096 = stablehlo.reshape %v2095 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v2097 = stablehlo.reshape %v2096 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v2098 = stablehlo.constant dense<0.0> : tensor<32x256x7x7xf32>
    %v2099 = stablehlo.maximum %v2097, %v2098 : tensor<32x256x7x7xf32>
    %v2100 = stablehlo.reshape %v2099 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v2101 = stablehlo.reshape %v2100 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v2102 = stablehlo.convolution(%v2101, %u21eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<512x256x1x1xf32>) -> tensor<32x512x7x7xf32>
    %v2103 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v2104 = stablehlo.add %v2102, %v2103 : tensor<32x512x7x7xf32>
    %v2105 = stablehlo.reshape %v2104 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v2106 = stablehlo.reshape %v2105 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v2107 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2108 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v2109 = stablehlo.constant dense<1.0e-5> : tensor<32x512x7x7xf32>
    %v2110 = stablehlo.reduce(%v2106 init: %v2107) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2111 = stablehlo.broadcast_in_dim %v2110, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v2112 = stablehlo.divide %v2111, %v2108 : tensor<32x512x7x7xf32>
    %v2113 = stablehlo.subtract %v2106, %v2112 : tensor<32x512x7x7xf32>
    %v2114 = stablehlo.multiply %v2113, %v2113 : tensor<32x512x7x7xf32>
    %v2115 = stablehlo.reduce(%v2114 init: %v2107) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2116 = stablehlo.broadcast_in_dim %v2115, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v2117 = stablehlo.divide %v2116, %v2108 : tensor<32x512x7x7xf32>
    %v2118 = stablehlo.add %v2117, %v2109 : tensor<32x512x7x7xf32>
    %v2119 = stablehlo.rsqrt %v2118 : tensor<32x512x7x7xf32>
    %v2120 = stablehlo.multiply %v2113, %v2119 : tensor<32x512x7x7xf32>
    %v2121 = stablehlo.broadcast_in_dim %u21eg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v2122 = stablehlo.broadcast_in_dim %u21ebt, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v2123 = stablehlo.multiply %v2120, %v2121 : tensor<32x512x7x7xf32>
    %v2124 = stablehlo.add %v2123, %v2122 : tensor<32x512x7x7xf32>
    %v2125 = stablehlo.reshape %v2124 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v2126 = stablehlo.reshape %v2125 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v2127 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v2128 = stablehlo.maximum %v2126, %v2127 : tensor<32x512x7x7xf32>
    %v2129 = stablehlo.reshape %v2128 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v2130 = stablehlo.reshape %v2129 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v2131 = stablehlo.convolution(%v2130, %u21pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<256x512x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v2132 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v2133 = stablehlo.add %v2131, %v2132 : tensor<32x256x7x7xf32>
    %v2134 = stablehlo.reshape %v2133 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v2135 = stablehlo.reshape %v2134 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v2136 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2137 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v2138 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v2139 = stablehlo.reduce(%v2135 init: %v2136) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v2140 = stablehlo.broadcast_in_dim %v2139, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v2141 = stablehlo.divide %v2140, %v2137 : tensor<32x256x7x7xf32>
    %v2142 = stablehlo.subtract %v2135, %v2141 : tensor<32x256x7x7xf32>
    %v2143 = stablehlo.multiply %v2142, %v2142 : tensor<32x256x7x7xf32>
    %v2144 = stablehlo.reduce(%v2143 init: %v2136) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v2145 = stablehlo.broadcast_in_dim %v2144, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v2146 = stablehlo.divide %v2145, %v2137 : tensor<32x256x7x7xf32>
    %v2147 = stablehlo.add %v2146, %v2138 : tensor<32x256x7x7xf32>
    %v2148 = stablehlo.rsqrt %v2147 : tensor<32x256x7x7xf32>
    %v2149 = stablehlo.multiply %v2142, %v2148 : tensor<32x256x7x7xf32>
    %v2150 = stablehlo.broadcast_in_dim %u21pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v2151 = stablehlo.broadcast_in_dim %u21pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v2152 = stablehlo.multiply %v2149, %v2150 : tensor<32x256x7x7xf32>
    %v2153 = stablehlo.add %v2152, %v2151 : tensor<32x256x7x7xf32>
    %v2154 = stablehlo.reshape %v2153 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v2155 = stablehlo.reshape %v2154 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v2156 = stablehlo.reshape %v2071 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v2157 = stablehlo.add %v2155, %v2156 : tensor<32x256x7x7xf32>
    %v2158 = stablehlo.reshape %v2157 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v2159 = stablehlo.reshape %v2158 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v2160 = stablehlo.convolution(%v2159, %h1W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<960x256x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v2161 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2162 = stablehlo.add %v2160, %v2161 : tensor<32x960x7x7xf32>
    %v2163 = stablehlo.reshape %v2162 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v2164 = stablehlo.reshape %v2163 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2165 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2166 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v2167 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v2168 = stablehlo.reduce(%v2164 init: %v2165) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2169 = stablehlo.broadcast_in_dim %v2168, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2170 = stablehlo.divide %v2169, %v2166 : tensor<32x960x7x7xf32>
    %v2171 = stablehlo.subtract %v2164, %v2170 : tensor<32x960x7x7xf32>
    %v2172 = stablehlo.multiply %v2171, %v2171 : tensor<32x960x7x7xf32>
    %v2173 = stablehlo.reduce(%v2172 init: %v2165) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2174 = stablehlo.broadcast_in_dim %v2173, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2175 = stablehlo.divide %v2174, %v2166 : tensor<32x960x7x7xf32>
    %v2176 = stablehlo.add %v2175, %v2167 : tensor<32x960x7x7xf32>
    %v2177 = stablehlo.rsqrt %v2176 : tensor<32x960x7x7xf32>
    %v2178 = stablehlo.multiply %v2171, %v2177 : tensor<32x960x7x7xf32>
    %v2179 = stablehlo.broadcast_in_dim %h1g, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2180 = stablehlo.broadcast_in_dim %h1bt, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2181 = stablehlo.multiply %v2178, %v2179 : tensor<32x960x7x7xf32>
    %v2182 = stablehlo.add %v2181, %v2180 : tensor<32x960x7x7xf32>
    %v2183 = stablehlo.reshape %v2182 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v2184 = stablehlo.reshape %v2183 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2185 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v2186 = stablehlo.maximum %v2184, %v2185 : tensor<32x960x7x7xf32>
    %v2187 = stablehlo.reshape %v2186 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v2188 = stablehlo.reshape %v2187 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2189 = stablehlo.convolution(%v2188, %hW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<1280x960x1x1xf32>) -> tensor<32x1280x7x7xf32>
    %v2190 = stablehlo.broadcast_in_dim %zb1280, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v2191 = stablehlo.add %v2189, %v2190 : tensor<32x1280x7x7xf32>
    %v2192 = stablehlo.reshape %v2191 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v2193 = stablehlo.reshape %v2192 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v2194 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2195 = stablehlo.constant dense<1568.0> : tensor<32x1280x7x7xf32>
    %v2196 = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %v2197 = stablehlo.reduce(%v2193 init: %v2194) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v2198 = stablehlo.broadcast_in_dim %v2197, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v2199 = stablehlo.divide %v2198, %v2195 : tensor<32x1280x7x7xf32>
    %v2200 = stablehlo.subtract %v2193, %v2199 : tensor<32x1280x7x7xf32>
    %v2201 = stablehlo.multiply %v2200, %v2200 : tensor<32x1280x7x7xf32>
    %v2202 = stablehlo.reduce(%v2201 init: %v2194) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v2203 = stablehlo.broadcast_in_dim %v2202, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v2204 = stablehlo.divide %v2203, %v2195 : tensor<32x1280x7x7xf32>
    %v2205 = stablehlo.add %v2204, %v2196 : tensor<32x1280x7x7xf32>
    %v2206 = stablehlo.rsqrt %v2205 : tensor<32x1280x7x7xf32>
    %v2207 = stablehlo.multiply %v2200, %v2206 : tensor<32x1280x7x7xf32>
    %v2208 = stablehlo.broadcast_in_dim %hg, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v2209 = stablehlo.broadcast_in_dim %hbt, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v2210 = stablehlo.multiply %v2207, %v2208 : tensor<32x1280x7x7xf32>
    %v2211 = stablehlo.add %v2210, %v2209 : tensor<32x1280x7x7xf32>
    %v2212 = stablehlo.reshape %v2211 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v2213 = stablehlo.reshape %v2212 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v2214 = stablehlo.constant dense<0.0> : tensor<32x80x28x28xf32>
    %v2215 = stablehlo.maximum %v2213, %v2214 : tensor<32x80x28x28xf32>
    %v2216 = stablehlo.reshape %v2215 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v2217 = stablehlo.reshape %v2216 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v2218 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2219 = stablehlo.reduce(%v2217 init: %v2218) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v2220 = stablehlo.constant dense<49.0> : tensor<32x1280xf32>
    %v2221 = stablehlo.divide %v2219, %v2220 : tensor<32x1280xf32>
    %v2222 = stablehlo.dot_general %v2221, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1280xf32>, tensor<1280x10xf32>) -> tensor<32x10xf32>
    %v2223 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v2224 = stablehlo.add %v2222, %v2223 : tensor<32x10xf32>
    return %v2224 : tensor<32x10xf32>
  }
}
