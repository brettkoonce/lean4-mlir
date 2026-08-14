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
    %v25 = stablehlo.constant dense<0.0> : tensor<32x401408xf32>
    %v26 = stablehlo.maximum %v24, %v25 : tensor<32x401408xf32>
    %v27 = stablehlo.reshape %v26 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v28 = stablehlo.convolution(%v27, %f0cW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<128x32x3x3xf32>) -> tensor<32x128x56x56xf32>
    %v29 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v30 = stablehlo.add %v28, %v29 : tensor<32x128x56x56xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v32 = stablehlo.reshape %v31 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v33 = stablehlo.constant dense<0.0> : tensor<f32>
    %v34 = stablehlo.constant dense<100352.0> : tensor<32x128x56x56xf32>
    %v35 = stablehlo.constant dense<1.0e-5> : tensor<32x128x56x56xf32>
    %v36 = stablehlo.reduce(%v32 init: %v33) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v37 = stablehlo.broadcast_in_dim %v36, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v38 = stablehlo.divide %v37, %v34 : tensor<32x128x56x56xf32>
    %v39 = stablehlo.subtract %v32, %v38 : tensor<32x128x56x56xf32>
    %v40 = stablehlo.multiply %v39, %v39 : tensor<32x128x56x56xf32>
    %v41 = stablehlo.reduce(%v40 init: %v33) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v42 = stablehlo.broadcast_in_dim %v41, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v43 = stablehlo.divide %v42, %v34 : tensor<32x128x56x56xf32>
    %v44 = stablehlo.add %v43, %v35 : tensor<32x128x56x56xf32>
    %v45 = stablehlo.rsqrt %v44 : tensor<32x128x56x56xf32>
    %v46 = stablehlo.multiply %v39, %v45 : tensor<32x128x56x56xf32>
    %v47 = stablehlo.broadcast_in_dim %f0cg, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v48 = stablehlo.broadcast_in_dim %f0cbt, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v49 = stablehlo.multiply %v46, %v47 : tensor<32x128x56x56xf32>
    %v50 = stablehlo.add %v49, %v48 : tensor<32x128x56x56xf32>
    %v51 = stablehlo.reshape %v50 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v52 = stablehlo.logistic %v51 : tensor<32x401408xf32>
    %v53 = stablehlo.multiply %v51, %v52 : tensor<32x401408xf32>
    %v54 = stablehlo.reshape %v53 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v55 = stablehlo.convolution(%v54, %f0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x56x56xf32>, tensor<48x128x1x1xf32>) -> tensor<32x48x56x56xf32>
    %v56 = stablehlo.broadcast_in_dim %zb48, dims = [1] : (tensor<48xf32>) -> tensor<32x48x56x56xf32>
    %v57 = stablehlo.add %v55, %v56 : tensor<32x48x56x56xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<32x48x56x56xf32>) -> tensor<32x150528xf32>
    %v59 = stablehlo.reshape %v58 : (tensor<32x150528xf32>) -> tensor<32x48x56x56xf32>
    %v60 = stablehlo.constant dense<0.0> : tensor<f32>
    %v61 = stablehlo.constant dense<100352.0> : tensor<32x48x56x56xf32>
    %v62 = stablehlo.constant dense<1.0e-5> : tensor<32x48x56x56xf32>
    %v63 = stablehlo.reduce(%v59 init: %v60) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x48x56x56xf32>, tensor<f32>) -> tensor<48xf32>
    %v64 = stablehlo.broadcast_in_dim %v63, dims = [1] : (tensor<48xf32>) -> tensor<32x48x56x56xf32>
    %v65 = stablehlo.divide %v64, %v61 : tensor<32x48x56x56xf32>
    %v66 = stablehlo.subtract %v59, %v65 : tensor<32x48x56x56xf32>
    %v67 = stablehlo.multiply %v66, %v66 : tensor<32x48x56x56xf32>
    %v68 = stablehlo.reduce(%v67 init: %v60) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x48x56x56xf32>, tensor<f32>) -> tensor<48xf32>
    %v69 = stablehlo.broadcast_in_dim %v68, dims = [1] : (tensor<48xf32>) -> tensor<32x48x56x56xf32>
    %v70 = stablehlo.divide %v69, %v61 : tensor<32x48x56x56xf32>
    %v71 = stablehlo.add %v70, %v62 : tensor<32x48x56x56xf32>
    %v72 = stablehlo.rsqrt %v71 : tensor<32x48x56x56xf32>
    %v73 = stablehlo.multiply %v66, %v72 : tensor<32x48x56x56xf32>
    %v74 = stablehlo.broadcast_in_dim %f0pg, dims = [1] : (tensor<48xf32>) -> tensor<32x48x56x56xf32>
    %v75 = stablehlo.broadcast_in_dim %f0pbt, dims = [1] : (tensor<48xf32>) -> tensor<32x48x56x56xf32>
    %v76 = stablehlo.multiply %v73, %v74 : tensor<32x48x56x56xf32>
    %v77 = stablehlo.add %v76, %v75 : tensor<32x48x56x56xf32>
    %v78 = stablehlo.reshape %v77 : (tensor<32x48x56x56xf32>) -> tensor<32x150528xf32>
    %v79 = stablehlo.reshape %v78 : (tensor<32x150528xf32>) -> tensor<32x48x56x56xf32>
    %v80 = stablehlo.convolution(%v79, %u1qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 48 : i64} : (tensor<32x48x56x56xf32>, tensor<48x1x3x3xf32>) -> tensor<32x48x28x28xf32>
    %v81 = stablehlo.broadcast_in_dim %zb48, dims = [1] : (tensor<48xf32>) -> tensor<32x48x28x28xf32>
    %v82 = stablehlo.add %v80, %v81 : tensor<32x48x28x28xf32>
    %v83 = stablehlo.reshape %v82 : (tensor<32x48x28x28xf32>) -> tensor<32x37632xf32>
    %v84 = stablehlo.reshape %v83 : (tensor<32x37632xf32>) -> tensor<32x48x28x28xf32>
    %v85 = stablehlo.constant dense<0.0> : tensor<f32>
    %v86 = stablehlo.constant dense<25088.0> : tensor<32x48x28x28xf32>
    %v87 = stablehlo.constant dense<1.0e-5> : tensor<32x48x28x28xf32>
    %v88 = stablehlo.reduce(%v84 init: %v85) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x48x28x28xf32>, tensor<f32>) -> tensor<48xf32>
    %v89 = stablehlo.broadcast_in_dim %v88, dims = [1] : (tensor<48xf32>) -> tensor<32x48x28x28xf32>
    %v90 = stablehlo.divide %v89, %v86 : tensor<32x48x28x28xf32>
    %v91 = stablehlo.subtract %v84, %v90 : tensor<32x48x28x28xf32>
    %v92 = stablehlo.multiply %v91, %v91 : tensor<32x48x28x28xf32>
    %v93 = stablehlo.reduce(%v92 init: %v85) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x48x28x28xf32>, tensor<f32>) -> tensor<48xf32>
    %v94 = stablehlo.broadcast_in_dim %v93, dims = [1] : (tensor<48xf32>) -> tensor<32x48x28x28xf32>
    %v95 = stablehlo.divide %v94, %v86 : tensor<32x48x28x28xf32>
    %v96 = stablehlo.add %v95, %v87 : tensor<32x48x28x28xf32>
    %v97 = stablehlo.rsqrt %v96 : tensor<32x48x28x28xf32>
    %v98 = stablehlo.multiply %v91, %v97 : tensor<32x48x28x28xf32>
    %v99 = stablehlo.broadcast_in_dim %u1qg, dims = [1] : (tensor<48xf32>) -> tensor<32x48x28x28xf32>
    %v100 = stablehlo.broadcast_in_dim %u1qbt, dims = [1] : (tensor<48xf32>) -> tensor<32x48x28x28xf32>
    %v101 = stablehlo.multiply %v98, %v99 : tensor<32x48x28x28xf32>
    %v102 = stablehlo.add %v101, %v100 : tensor<32x48x28x28xf32>
    %v103 = stablehlo.reshape %v102 : (tensor<32x48x28x28xf32>) -> tensor<32x37632xf32>
    %v104 = stablehlo.constant dense<0.0> : tensor<32x37632xf32>
    %v105 = stablehlo.maximum %v103, %v104 : tensor<32x37632xf32>
    %v106 = stablehlo.reshape %v105 : (tensor<32x37632xf32>) -> tensor<32x48x28x28xf32>
    %v107 = stablehlo.convolution(%v106, %u1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x48x28x28xf32>, tensor<192x48x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v108 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v109 = stablehlo.add %v107, %v108 : tensor<32x192x28x28xf32>
    %v110 = stablehlo.reshape %v109 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v111 = stablehlo.reshape %v110 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v112 = stablehlo.constant dense<0.0> : tensor<f32>
    %v113 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v114 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v115 = stablehlo.reduce(%v111 init: %v112) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v116 = stablehlo.broadcast_in_dim %v115, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v117 = stablehlo.divide %v116, %v113 : tensor<32x192x28x28xf32>
    %v118 = stablehlo.subtract %v111, %v117 : tensor<32x192x28x28xf32>
    %v119 = stablehlo.multiply %v118, %v118 : tensor<32x192x28x28xf32>
    %v120 = stablehlo.reduce(%v119 init: %v112) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v121 = stablehlo.broadcast_in_dim %v120, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v122 = stablehlo.divide %v121, %v113 : tensor<32x192x28x28xf32>
    %v123 = stablehlo.add %v122, %v114 : tensor<32x192x28x28xf32>
    %v124 = stablehlo.rsqrt %v123 : tensor<32x192x28x28xf32>
    %v125 = stablehlo.multiply %v118, %v124 : tensor<32x192x28x28xf32>
    %v126 = stablehlo.broadcast_in_dim %u1eg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v127 = stablehlo.broadcast_in_dim %u1ebt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v128 = stablehlo.multiply %v125, %v126 : tensor<32x192x28x28xf32>
    %v129 = stablehlo.add %v128, %v127 : tensor<32x192x28x28xf32>
    %v130 = stablehlo.reshape %v129 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v131 = stablehlo.constant dense<0.0> : tensor<32x150528xf32>
    %v132 = stablehlo.maximum %v130, %v131 : tensor<32x150528xf32>
    %v133 = stablehlo.reshape %v132 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v134 = stablehlo.convolution(%v133, %u1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x5x5xf32>) -> tensor<32x192x28x28xf32>
    %v135 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v136 = stablehlo.add %v134, %v135 : tensor<32x192x28x28xf32>
    %v137 = stablehlo.reshape %v136 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v138 = stablehlo.reshape %v137 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v139 = stablehlo.constant dense<0.0> : tensor<f32>
    %v140 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v141 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v142 = stablehlo.reduce(%v138 init: %v139) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v143 = stablehlo.broadcast_in_dim %v142, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v144 = stablehlo.divide %v143, %v140 : tensor<32x192x28x28xf32>
    %v145 = stablehlo.subtract %v138, %v144 : tensor<32x192x28x28xf32>
    %v146 = stablehlo.multiply %v145, %v145 : tensor<32x192x28x28xf32>
    %v147 = stablehlo.reduce(%v146 init: %v139) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v148 = stablehlo.broadcast_in_dim %v147, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v149 = stablehlo.divide %v148, %v140 : tensor<32x192x28x28xf32>
    %v150 = stablehlo.add %v149, %v141 : tensor<32x192x28x28xf32>
    %v151 = stablehlo.rsqrt %v150 : tensor<32x192x28x28xf32>
    %v152 = stablehlo.multiply %v145, %v151 : tensor<32x192x28x28xf32>
    %v153 = stablehlo.broadcast_in_dim %u1dg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v154 = stablehlo.broadcast_in_dim %u1dbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v155 = stablehlo.multiply %v152, %v153 : tensor<32x192x28x28xf32>
    %v156 = stablehlo.add %v155, %v154 : tensor<32x192x28x28xf32>
    %v157 = stablehlo.reshape %v156 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v158 = stablehlo.constant dense<0.0> : tensor<32x150528xf32>
    %v159 = stablehlo.maximum %v157, %v158 : tensor<32x150528xf32>
    %v160 = stablehlo.reshape %v159 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v161 = stablehlo.convolution(%v160, %u1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<80x192x1x1xf32>) -> tensor<32x80x28x28xf32>
    %v162 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v163 = stablehlo.add %v161, %v162 : tensor<32x80x28x28xf32>
    %v164 = stablehlo.reshape %v163 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v165 = stablehlo.reshape %v164 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v166 = stablehlo.constant dense<0.0> : tensor<f32>
    %v167 = stablehlo.constant dense<25088.0> : tensor<32x80x28x28xf32>
    %v168 = stablehlo.constant dense<1.0e-5> : tensor<32x80x28x28xf32>
    %v169 = stablehlo.reduce(%v165 init: %v166) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x28x28xf32>, tensor<f32>) -> tensor<80xf32>
    %v170 = stablehlo.broadcast_in_dim %v169, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v171 = stablehlo.divide %v170, %v167 : tensor<32x80x28x28xf32>
    %v172 = stablehlo.subtract %v165, %v171 : tensor<32x80x28x28xf32>
    %v173 = stablehlo.multiply %v172, %v172 : tensor<32x80x28x28xf32>
    %v174 = stablehlo.reduce(%v173 init: %v166) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x28x28xf32>, tensor<f32>) -> tensor<80xf32>
    %v175 = stablehlo.broadcast_in_dim %v174, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v176 = stablehlo.divide %v175, %v167 : tensor<32x80x28x28xf32>
    %v177 = stablehlo.add %v176, %v168 : tensor<32x80x28x28xf32>
    %v178 = stablehlo.rsqrt %v177 : tensor<32x80x28x28xf32>
    %v179 = stablehlo.multiply %v172, %v178 : tensor<32x80x28x28xf32>
    %v180 = stablehlo.broadcast_in_dim %u1pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v181 = stablehlo.broadcast_in_dim %u1pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v182 = stablehlo.multiply %v179, %v180 : tensor<32x80x28x28xf32>
    %v183 = stablehlo.add %v182, %v181 : tensor<32x80x28x28xf32>
    %v184 = stablehlo.reshape %v183 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v185 = stablehlo.reshape %v184 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v186 = stablehlo.convolution(%v185, %u2qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 80 : i64} : (tensor<32x80x28x28xf32>, tensor<80x1x3x3xf32>) -> tensor<32x80x28x28xf32>
    %v187 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v188 = stablehlo.add %v186, %v187 : tensor<32x80x28x28xf32>
    %v189 = stablehlo.reshape %v188 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v190 = stablehlo.reshape %v189 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v191 = stablehlo.constant dense<0.0> : tensor<f32>
    %v192 = stablehlo.constant dense<25088.0> : tensor<32x80x28x28xf32>
    %v193 = stablehlo.constant dense<1.0e-5> : tensor<32x80x28x28xf32>
    %v194 = stablehlo.reduce(%v190 init: %v191) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x28x28xf32>, tensor<f32>) -> tensor<80xf32>
    %v195 = stablehlo.broadcast_in_dim %v194, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v196 = stablehlo.divide %v195, %v192 : tensor<32x80x28x28xf32>
    %v197 = stablehlo.subtract %v190, %v196 : tensor<32x80x28x28xf32>
    %v198 = stablehlo.multiply %v197, %v197 : tensor<32x80x28x28xf32>
    %v199 = stablehlo.reduce(%v198 init: %v191) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x28x28xf32>, tensor<f32>) -> tensor<80xf32>
    %v200 = stablehlo.broadcast_in_dim %v199, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v201 = stablehlo.divide %v200, %v192 : tensor<32x80x28x28xf32>
    %v202 = stablehlo.add %v201, %v193 : tensor<32x80x28x28xf32>
    %v203 = stablehlo.rsqrt %v202 : tensor<32x80x28x28xf32>
    %v204 = stablehlo.multiply %v197, %v203 : tensor<32x80x28x28xf32>
    %v205 = stablehlo.broadcast_in_dim %u2qg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v206 = stablehlo.broadcast_in_dim %u2qbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v207 = stablehlo.multiply %v204, %v205 : tensor<32x80x28x28xf32>
    %v208 = stablehlo.add %v207, %v206 : tensor<32x80x28x28xf32>
    %v209 = stablehlo.reshape %v208 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v210 = stablehlo.constant dense<0.0> : tensor<32x62720xf32>
    %v211 = stablehlo.maximum %v209, %v210 : tensor<32x62720xf32>
    %v212 = stablehlo.reshape %v211 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v213 = stablehlo.convolution(%v212, %u2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x28x28xf32>, tensor<160x80x1x1xf32>) -> tensor<32x160x28x28xf32>
    %v214 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v215 = stablehlo.add %v213, %v214 : tensor<32x160x28x28xf32>
    %v216 = stablehlo.reshape %v215 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v217 = stablehlo.reshape %v216 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v218 = stablehlo.constant dense<0.0> : tensor<f32>
    %v219 = stablehlo.constant dense<25088.0> : tensor<32x160x28x28xf32>
    %v220 = stablehlo.constant dense<1.0e-5> : tensor<32x160x28x28xf32>
    %v221 = stablehlo.reduce(%v217 init: %v218) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x28x28xf32>, tensor<f32>) -> tensor<160xf32>
    %v222 = stablehlo.broadcast_in_dim %v221, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v223 = stablehlo.divide %v222, %v219 : tensor<32x160x28x28xf32>
    %v224 = stablehlo.subtract %v217, %v223 : tensor<32x160x28x28xf32>
    %v225 = stablehlo.multiply %v224, %v224 : tensor<32x160x28x28xf32>
    %v226 = stablehlo.reduce(%v225 init: %v218) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x28x28xf32>, tensor<f32>) -> tensor<160xf32>
    %v227 = stablehlo.broadcast_in_dim %v226, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v228 = stablehlo.divide %v227, %v219 : tensor<32x160x28x28xf32>
    %v229 = stablehlo.add %v228, %v220 : tensor<32x160x28x28xf32>
    %v230 = stablehlo.rsqrt %v229 : tensor<32x160x28x28xf32>
    %v231 = stablehlo.multiply %v224, %v230 : tensor<32x160x28x28xf32>
    %v232 = stablehlo.broadcast_in_dim %u2eg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v233 = stablehlo.broadcast_in_dim %u2ebt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v234 = stablehlo.multiply %v231, %v232 : tensor<32x160x28x28xf32>
    %v235 = stablehlo.add %v234, %v233 : tensor<32x160x28x28xf32>
    %v236 = stablehlo.reshape %v235 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v237 = stablehlo.constant dense<0.0> : tensor<32x125440xf32>
    %v238 = stablehlo.maximum %v236, %v237 : tensor<32x125440xf32>
    %v239 = stablehlo.reshape %v238 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v240 = stablehlo.convolution(%v239, %u2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x28x28xf32>, tensor<160x1x3x3xf32>) -> tensor<32x160x28x28xf32>
    %v241 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v242 = stablehlo.add %v240, %v241 : tensor<32x160x28x28xf32>
    %v243 = stablehlo.reshape %v242 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v244 = stablehlo.reshape %v243 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v245 = stablehlo.constant dense<0.0> : tensor<f32>
    %v246 = stablehlo.constant dense<25088.0> : tensor<32x160x28x28xf32>
    %v247 = stablehlo.constant dense<1.0e-5> : tensor<32x160x28x28xf32>
    %v248 = stablehlo.reduce(%v244 init: %v245) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x28x28xf32>, tensor<f32>) -> tensor<160xf32>
    %v249 = stablehlo.broadcast_in_dim %v248, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v250 = stablehlo.divide %v249, %v246 : tensor<32x160x28x28xf32>
    %v251 = stablehlo.subtract %v244, %v250 : tensor<32x160x28x28xf32>
    %v252 = stablehlo.multiply %v251, %v251 : tensor<32x160x28x28xf32>
    %v253 = stablehlo.reduce(%v252 init: %v245) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x28x28xf32>, tensor<f32>) -> tensor<160xf32>
    %v254 = stablehlo.broadcast_in_dim %v253, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v255 = stablehlo.divide %v254, %v246 : tensor<32x160x28x28xf32>
    %v256 = stablehlo.add %v255, %v247 : tensor<32x160x28x28xf32>
    %v257 = stablehlo.rsqrt %v256 : tensor<32x160x28x28xf32>
    %v258 = stablehlo.multiply %v251, %v257 : tensor<32x160x28x28xf32>
    %v259 = stablehlo.broadcast_in_dim %u2dg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v260 = stablehlo.broadcast_in_dim %u2dbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v261 = stablehlo.multiply %v258, %v259 : tensor<32x160x28x28xf32>
    %v262 = stablehlo.add %v261, %v260 : tensor<32x160x28x28xf32>
    %v263 = stablehlo.reshape %v262 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v264 = stablehlo.constant dense<0.0> : tensor<32x125440xf32>
    %v265 = stablehlo.maximum %v263, %v264 : tensor<32x125440xf32>
    %v266 = stablehlo.reshape %v265 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v267 = stablehlo.convolution(%v266, %u2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x28x28xf32>, tensor<80x160x1x1xf32>) -> tensor<32x80x28x28xf32>
    %v268 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v269 = stablehlo.add %v267, %v268 : tensor<32x80x28x28xf32>
    %v270 = stablehlo.reshape %v269 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v271 = stablehlo.reshape %v270 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v272 = stablehlo.constant dense<0.0> : tensor<f32>
    %v273 = stablehlo.constant dense<25088.0> : tensor<32x80x28x28xf32>
    %v274 = stablehlo.constant dense<1.0e-5> : tensor<32x80x28x28xf32>
    %v275 = stablehlo.reduce(%v271 init: %v272) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x28x28xf32>, tensor<f32>) -> tensor<80xf32>
    %v276 = stablehlo.broadcast_in_dim %v275, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v277 = stablehlo.divide %v276, %v273 : tensor<32x80x28x28xf32>
    %v278 = stablehlo.subtract %v271, %v277 : tensor<32x80x28x28xf32>
    %v279 = stablehlo.multiply %v278, %v278 : tensor<32x80x28x28xf32>
    %v280 = stablehlo.reduce(%v279 init: %v272) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x28x28xf32>, tensor<f32>) -> tensor<80xf32>
    %v281 = stablehlo.broadcast_in_dim %v280, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v282 = stablehlo.divide %v281, %v273 : tensor<32x80x28x28xf32>
    %v283 = stablehlo.add %v282, %v274 : tensor<32x80x28x28xf32>
    %v284 = stablehlo.rsqrt %v283 : tensor<32x80x28x28xf32>
    %v285 = stablehlo.multiply %v278, %v284 : tensor<32x80x28x28xf32>
    %v286 = stablehlo.broadcast_in_dim %u2pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v287 = stablehlo.broadcast_in_dim %u2pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v288 = stablehlo.multiply %v285, %v286 : tensor<32x80x28x28xf32>
    %v289 = stablehlo.add %v288, %v287 : tensor<32x80x28x28xf32>
    %v290 = stablehlo.reshape %v289 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v291 = stablehlo.add %v290, %v184 : tensor<32x62720xf32>
    %v292 = stablehlo.reshape %v291 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v293 = stablehlo.convolution(%v292, %u3qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 80 : i64} : (tensor<32x80x28x28xf32>, tensor<80x1x3x3xf32>) -> tensor<32x80x14x14xf32>
    %v294 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v295 = stablehlo.add %v293, %v294 : tensor<32x80x14x14xf32>
    %v296 = stablehlo.reshape %v295 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v297 = stablehlo.reshape %v296 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v298 = stablehlo.constant dense<0.0> : tensor<f32>
    %v299 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v300 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v301 = stablehlo.reduce(%v297 init: %v298) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v302 = stablehlo.broadcast_in_dim %v301, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v303 = stablehlo.divide %v302, %v299 : tensor<32x80x14x14xf32>
    %v304 = stablehlo.subtract %v297, %v303 : tensor<32x80x14x14xf32>
    %v305 = stablehlo.multiply %v304, %v304 : tensor<32x80x14x14xf32>
    %v306 = stablehlo.reduce(%v305 init: %v298) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v307 = stablehlo.broadcast_in_dim %v306, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v308 = stablehlo.divide %v307, %v299 : tensor<32x80x14x14xf32>
    %v309 = stablehlo.add %v308, %v300 : tensor<32x80x14x14xf32>
    %v310 = stablehlo.rsqrt %v309 : tensor<32x80x14x14xf32>
    %v311 = stablehlo.multiply %v304, %v310 : tensor<32x80x14x14xf32>
    %v312 = stablehlo.broadcast_in_dim %u3qg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v313 = stablehlo.broadcast_in_dim %u3qbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v314 = stablehlo.multiply %v311, %v312 : tensor<32x80x14x14xf32>
    %v315 = stablehlo.add %v314, %v313 : tensor<32x80x14x14xf32>
    %v316 = stablehlo.reshape %v315 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v317 = stablehlo.constant dense<0.0> : tensor<32x15680xf32>
    %v318 = stablehlo.maximum %v316, %v317 : tensor<32x15680xf32>
    %v319 = stablehlo.reshape %v318 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v320 = stablehlo.convolution(%v319, %u3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v321 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v322 = stablehlo.add %v320, %v321 : tensor<32x480x14x14xf32>
    %v323 = stablehlo.reshape %v322 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v324 = stablehlo.reshape %v323 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v325 = stablehlo.constant dense<0.0> : tensor<f32>
    %v326 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v327 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v328 = stablehlo.reduce(%v324 init: %v325) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v329 = stablehlo.broadcast_in_dim %v328, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v330 = stablehlo.divide %v329, %v326 : tensor<32x480x14x14xf32>
    %v331 = stablehlo.subtract %v324, %v330 : tensor<32x480x14x14xf32>
    %v332 = stablehlo.multiply %v331, %v331 : tensor<32x480x14x14xf32>
    %v333 = stablehlo.reduce(%v332 init: %v325) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v334 = stablehlo.broadcast_in_dim %v333, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v335 = stablehlo.divide %v334, %v326 : tensor<32x480x14x14xf32>
    %v336 = stablehlo.add %v335, %v327 : tensor<32x480x14x14xf32>
    %v337 = stablehlo.rsqrt %v336 : tensor<32x480x14x14xf32>
    %v338 = stablehlo.multiply %v331, %v337 : tensor<32x480x14x14xf32>
    %v339 = stablehlo.broadcast_in_dim %u3eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v340 = stablehlo.broadcast_in_dim %u3ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v341 = stablehlo.multiply %v338, %v339 : tensor<32x480x14x14xf32>
    %v342 = stablehlo.add %v341, %v340 : tensor<32x480x14x14xf32>
    %v343 = stablehlo.reshape %v342 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v344 = stablehlo.constant dense<0.0> : tensor<32x94080xf32>
    %v345 = stablehlo.maximum %v343, %v344 : tensor<32x94080xf32>
    %v346 = stablehlo.reshape %v345 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v347 = stablehlo.convolution(%v346, %u3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x5x5xf32>) -> tensor<32x480x14x14xf32>
    %v348 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v349 = stablehlo.add %v347, %v348 : tensor<32x480x14x14xf32>
    %v350 = stablehlo.reshape %v349 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v351 = stablehlo.reshape %v350 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v352 = stablehlo.constant dense<0.0> : tensor<f32>
    %v353 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v354 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v355 = stablehlo.reduce(%v351 init: %v352) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v356 = stablehlo.broadcast_in_dim %v355, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v357 = stablehlo.divide %v356, %v353 : tensor<32x480x14x14xf32>
    %v358 = stablehlo.subtract %v351, %v357 : tensor<32x480x14x14xf32>
    %v359 = stablehlo.multiply %v358, %v358 : tensor<32x480x14x14xf32>
    %v360 = stablehlo.reduce(%v359 init: %v352) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v361 = stablehlo.broadcast_in_dim %v360, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v362 = stablehlo.divide %v361, %v353 : tensor<32x480x14x14xf32>
    %v363 = stablehlo.add %v362, %v354 : tensor<32x480x14x14xf32>
    %v364 = stablehlo.rsqrt %v363 : tensor<32x480x14x14xf32>
    %v365 = stablehlo.multiply %v358, %v364 : tensor<32x480x14x14xf32>
    %v366 = stablehlo.broadcast_in_dim %u3dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v367 = stablehlo.broadcast_in_dim %u3dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v368 = stablehlo.multiply %v365, %v366 : tensor<32x480x14x14xf32>
    %v369 = stablehlo.add %v368, %v367 : tensor<32x480x14x14xf32>
    %v370 = stablehlo.reshape %v369 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v371 = stablehlo.constant dense<0.0> : tensor<32x94080xf32>
    %v372 = stablehlo.maximum %v370, %v371 : tensor<32x94080xf32>
    %v373 = stablehlo.reshape %v372 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v374 = stablehlo.convolution(%v373, %u3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<160x480x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v375 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v376 = stablehlo.add %v374, %v375 : tensor<32x160x14x14xf32>
    %v377 = stablehlo.reshape %v376 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v378 = stablehlo.reshape %v377 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v379 = stablehlo.constant dense<0.0> : tensor<f32>
    %v380 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v381 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v382 = stablehlo.reduce(%v378 init: %v379) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v383 = stablehlo.broadcast_in_dim %v382, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v384 = stablehlo.divide %v383, %v380 : tensor<32x160x14x14xf32>
    %v385 = stablehlo.subtract %v378, %v384 : tensor<32x160x14x14xf32>
    %v386 = stablehlo.multiply %v385, %v385 : tensor<32x160x14x14xf32>
    %v387 = stablehlo.reduce(%v386 init: %v379) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v388 = stablehlo.broadcast_in_dim %v387, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v389 = stablehlo.divide %v388, %v380 : tensor<32x160x14x14xf32>
    %v390 = stablehlo.add %v389, %v381 : tensor<32x160x14x14xf32>
    %v391 = stablehlo.rsqrt %v390 : tensor<32x160x14x14xf32>
    %v392 = stablehlo.multiply %v385, %v391 : tensor<32x160x14x14xf32>
    %v393 = stablehlo.broadcast_in_dim %u3pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v394 = stablehlo.broadcast_in_dim %u3pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v395 = stablehlo.multiply %v392, %v393 : tensor<32x160x14x14xf32>
    %v396 = stablehlo.add %v395, %v394 : tensor<32x160x14x14xf32>
    %v397 = stablehlo.reshape %v396 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v398 = stablehlo.reshape %v397 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v399 = stablehlo.convolution(%v398, %u4qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<32x160x14x14xf32>
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
    %v418 = stablehlo.broadcast_in_dim %u4qg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v419 = stablehlo.broadcast_in_dim %u4qbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v420 = stablehlo.multiply %v417, %v418 : tensor<32x160x14x14xf32>
    %v421 = stablehlo.add %v420, %v419 : tensor<32x160x14x14xf32>
    %v422 = stablehlo.reshape %v421 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v423 = stablehlo.constant dense<0.0> : tensor<32x31360xf32>
    %v424 = stablehlo.maximum %v422, %v423 : tensor<32x31360xf32>
    %v425 = stablehlo.reshape %v424 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v426 = stablehlo.convolution(%v425, %u4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<32x640x14x14xf32>
    %v427 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v428 = stablehlo.add %v426, %v427 : tensor<32x640x14x14xf32>
    %v429 = stablehlo.reshape %v428 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v430 = stablehlo.reshape %v429 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v431 = stablehlo.constant dense<0.0> : tensor<f32>
    %v432 = stablehlo.constant dense<6272.0> : tensor<32x640x14x14xf32>
    %v433 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v434 = stablehlo.reduce(%v430 init: %v431) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v435 = stablehlo.broadcast_in_dim %v434, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v436 = stablehlo.divide %v435, %v432 : tensor<32x640x14x14xf32>
    %v437 = stablehlo.subtract %v430, %v436 : tensor<32x640x14x14xf32>
    %v438 = stablehlo.multiply %v437, %v437 : tensor<32x640x14x14xf32>
    %v439 = stablehlo.reduce(%v438 init: %v431) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v440 = stablehlo.broadcast_in_dim %v439, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v441 = stablehlo.divide %v440, %v432 : tensor<32x640x14x14xf32>
    %v442 = stablehlo.add %v441, %v433 : tensor<32x640x14x14xf32>
    %v443 = stablehlo.rsqrt %v442 : tensor<32x640x14x14xf32>
    %v444 = stablehlo.multiply %v437, %v443 : tensor<32x640x14x14xf32>
    %v445 = stablehlo.broadcast_in_dim %u4eg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v446 = stablehlo.broadcast_in_dim %u4ebt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v447 = stablehlo.multiply %v444, %v445 : tensor<32x640x14x14xf32>
    %v448 = stablehlo.add %v447, %v446 : tensor<32x640x14x14xf32>
    %v449 = stablehlo.reshape %v448 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v450 = stablehlo.constant dense<0.0> : tensor<32x125440xf32>
    %v451 = stablehlo.maximum %v449, %v450 : tensor<32x125440xf32>
    %v452 = stablehlo.reshape %v451 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v453 = stablehlo.convolution(%v452, %u4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 640 : i64} : (tensor<32x640x14x14xf32>, tensor<640x1x3x3xf32>) -> tensor<32x640x14x14xf32>
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
    %v472 = stablehlo.broadcast_in_dim %u4dg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v473 = stablehlo.broadcast_in_dim %u4dbt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v474 = stablehlo.multiply %v471, %v472 : tensor<32x640x14x14xf32>
    %v475 = stablehlo.add %v474, %v473 : tensor<32x640x14x14xf32>
    %v476 = stablehlo.reshape %v475 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v477 = stablehlo.constant dense<0.0> : tensor<32x125440xf32>
    %v478 = stablehlo.maximum %v476, %v477 : tensor<32x125440xf32>
    %v479 = stablehlo.reshape %v478 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v480 = stablehlo.convolution(%v479, %u4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v481 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v482 = stablehlo.add %v480, %v481 : tensor<32x160x14x14xf32>
    %v483 = stablehlo.reshape %v482 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v484 = stablehlo.reshape %v483 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v485 = stablehlo.constant dense<0.0> : tensor<f32>
    %v486 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v487 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v488 = stablehlo.reduce(%v484 init: %v485) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v489 = stablehlo.broadcast_in_dim %v488, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v490 = stablehlo.divide %v489, %v486 : tensor<32x160x14x14xf32>
    %v491 = stablehlo.subtract %v484, %v490 : tensor<32x160x14x14xf32>
    %v492 = stablehlo.multiply %v491, %v491 : tensor<32x160x14x14xf32>
    %v493 = stablehlo.reduce(%v492 init: %v485) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v494 = stablehlo.broadcast_in_dim %v493, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v495 = stablehlo.divide %v494, %v486 : tensor<32x160x14x14xf32>
    %v496 = stablehlo.add %v495, %v487 : tensor<32x160x14x14xf32>
    %v497 = stablehlo.rsqrt %v496 : tensor<32x160x14x14xf32>
    %v498 = stablehlo.multiply %v491, %v497 : tensor<32x160x14x14xf32>
    %v499 = stablehlo.broadcast_in_dim %u4pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v500 = stablehlo.broadcast_in_dim %u4pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v501 = stablehlo.multiply %v498, %v499 : tensor<32x160x14x14xf32>
    %v502 = stablehlo.add %v501, %v500 : tensor<32x160x14x14xf32>
    %v503 = stablehlo.reshape %v502 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v504 = stablehlo.add %v503, %v397 : tensor<32x31360xf32>
    %v505 = stablehlo.reshape %v504 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v506 = stablehlo.convolution(%v505, %u5qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<32x160x14x14xf32>
    %v507 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v508 = stablehlo.add %v506, %v507 : tensor<32x160x14x14xf32>
    %v509 = stablehlo.reshape %v508 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v510 = stablehlo.reshape %v509 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v511 = stablehlo.constant dense<0.0> : tensor<f32>
    %v512 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v513 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v514 = stablehlo.reduce(%v510 init: %v511) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v515 = stablehlo.broadcast_in_dim %v514, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v516 = stablehlo.divide %v515, %v512 : tensor<32x160x14x14xf32>
    %v517 = stablehlo.subtract %v510, %v516 : tensor<32x160x14x14xf32>
    %v518 = stablehlo.multiply %v517, %v517 : tensor<32x160x14x14xf32>
    %v519 = stablehlo.reduce(%v518 init: %v511) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v520 = stablehlo.broadcast_in_dim %v519, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v521 = stablehlo.divide %v520, %v512 : tensor<32x160x14x14xf32>
    %v522 = stablehlo.add %v521, %v513 : tensor<32x160x14x14xf32>
    %v523 = stablehlo.rsqrt %v522 : tensor<32x160x14x14xf32>
    %v524 = stablehlo.multiply %v517, %v523 : tensor<32x160x14x14xf32>
    %v525 = stablehlo.broadcast_in_dim %u5qg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v526 = stablehlo.broadcast_in_dim %u5qbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v527 = stablehlo.multiply %v524, %v525 : tensor<32x160x14x14xf32>
    %v528 = stablehlo.add %v527, %v526 : tensor<32x160x14x14xf32>
    %v529 = stablehlo.reshape %v528 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v530 = stablehlo.constant dense<0.0> : tensor<32x31360xf32>
    %v531 = stablehlo.maximum %v529, %v530 : tensor<32x31360xf32>
    %v532 = stablehlo.reshape %v531 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v533 = stablehlo.convolution(%v532, %u5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<32x640x14x14xf32>
    %v534 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v535 = stablehlo.add %v533, %v534 : tensor<32x640x14x14xf32>
    %v536 = stablehlo.reshape %v535 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v537 = stablehlo.reshape %v536 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v538 = stablehlo.constant dense<0.0> : tensor<f32>
    %v539 = stablehlo.constant dense<6272.0> : tensor<32x640x14x14xf32>
    %v540 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v541 = stablehlo.reduce(%v537 init: %v538) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v542 = stablehlo.broadcast_in_dim %v541, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v543 = stablehlo.divide %v542, %v539 : tensor<32x640x14x14xf32>
    %v544 = stablehlo.subtract %v537, %v543 : tensor<32x640x14x14xf32>
    %v545 = stablehlo.multiply %v544, %v544 : tensor<32x640x14x14xf32>
    %v546 = stablehlo.reduce(%v545 init: %v538) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v547 = stablehlo.broadcast_in_dim %v546, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v548 = stablehlo.divide %v547, %v539 : tensor<32x640x14x14xf32>
    %v549 = stablehlo.add %v548, %v540 : tensor<32x640x14x14xf32>
    %v550 = stablehlo.rsqrt %v549 : tensor<32x640x14x14xf32>
    %v551 = stablehlo.multiply %v544, %v550 : tensor<32x640x14x14xf32>
    %v552 = stablehlo.broadcast_in_dim %u5eg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v553 = stablehlo.broadcast_in_dim %u5ebt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v554 = stablehlo.multiply %v551, %v552 : tensor<32x640x14x14xf32>
    %v555 = stablehlo.add %v554, %v553 : tensor<32x640x14x14xf32>
    %v556 = stablehlo.reshape %v555 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v557 = stablehlo.constant dense<0.0> : tensor<32x125440xf32>
    %v558 = stablehlo.maximum %v556, %v557 : tensor<32x125440xf32>
    %v559 = stablehlo.reshape %v558 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v560 = stablehlo.convolution(%v559, %u5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 640 : i64} : (tensor<32x640x14x14xf32>, tensor<640x1x3x3xf32>) -> tensor<32x640x14x14xf32>
    %v561 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v562 = stablehlo.add %v560, %v561 : tensor<32x640x14x14xf32>
    %v563 = stablehlo.reshape %v562 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v564 = stablehlo.reshape %v563 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v565 = stablehlo.constant dense<0.0> : tensor<f32>
    %v566 = stablehlo.constant dense<6272.0> : tensor<32x640x14x14xf32>
    %v567 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v568 = stablehlo.reduce(%v564 init: %v565) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v569 = stablehlo.broadcast_in_dim %v568, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v570 = stablehlo.divide %v569, %v566 : tensor<32x640x14x14xf32>
    %v571 = stablehlo.subtract %v564, %v570 : tensor<32x640x14x14xf32>
    %v572 = stablehlo.multiply %v571, %v571 : tensor<32x640x14x14xf32>
    %v573 = stablehlo.reduce(%v572 init: %v565) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v574 = stablehlo.broadcast_in_dim %v573, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v575 = stablehlo.divide %v574, %v566 : tensor<32x640x14x14xf32>
    %v576 = stablehlo.add %v575, %v567 : tensor<32x640x14x14xf32>
    %v577 = stablehlo.rsqrt %v576 : tensor<32x640x14x14xf32>
    %v578 = stablehlo.multiply %v571, %v577 : tensor<32x640x14x14xf32>
    %v579 = stablehlo.broadcast_in_dim %u5dg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v580 = stablehlo.broadcast_in_dim %u5dbt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v581 = stablehlo.multiply %v578, %v579 : tensor<32x640x14x14xf32>
    %v582 = stablehlo.add %v581, %v580 : tensor<32x640x14x14xf32>
    %v583 = stablehlo.reshape %v582 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v584 = stablehlo.constant dense<0.0> : tensor<32x125440xf32>
    %v585 = stablehlo.maximum %v583, %v584 : tensor<32x125440xf32>
    %v586 = stablehlo.reshape %v585 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v587 = stablehlo.convolution(%v586, %u5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v588 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v589 = stablehlo.add %v587, %v588 : tensor<32x160x14x14xf32>
    %v590 = stablehlo.reshape %v589 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v591 = stablehlo.reshape %v590 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v592 = stablehlo.constant dense<0.0> : tensor<f32>
    %v593 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v594 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v595 = stablehlo.reduce(%v591 init: %v592) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v596 = stablehlo.broadcast_in_dim %v595, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v597 = stablehlo.divide %v596, %v593 : tensor<32x160x14x14xf32>
    %v598 = stablehlo.subtract %v591, %v597 : tensor<32x160x14x14xf32>
    %v599 = stablehlo.multiply %v598, %v598 : tensor<32x160x14x14xf32>
    %v600 = stablehlo.reduce(%v599 init: %v592) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v601 = stablehlo.broadcast_in_dim %v600, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v602 = stablehlo.divide %v601, %v593 : tensor<32x160x14x14xf32>
    %v603 = stablehlo.add %v602, %v594 : tensor<32x160x14x14xf32>
    %v604 = stablehlo.rsqrt %v603 : tensor<32x160x14x14xf32>
    %v605 = stablehlo.multiply %v598, %v604 : tensor<32x160x14x14xf32>
    %v606 = stablehlo.broadcast_in_dim %u5pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v607 = stablehlo.broadcast_in_dim %u5pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v608 = stablehlo.multiply %v605, %v606 : tensor<32x160x14x14xf32>
    %v609 = stablehlo.add %v608, %v607 : tensor<32x160x14x14xf32>
    %v610 = stablehlo.reshape %v609 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v611 = stablehlo.add %v610, %v504 : tensor<32x31360xf32>
    %v612 = stablehlo.reshape %v611 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v613 = stablehlo.convolution(%v612, %u6qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<32x160x14x14xf32>
    %v614 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v615 = stablehlo.add %v613, %v614 : tensor<32x160x14x14xf32>
    %v616 = stablehlo.reshape %v615 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v617 = stablehlo.reshape %v616 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v618 = stablehlo.constant dense<0.0> : tensor<f32>
    %v619 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v620 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v621 = stablehlo.reduce(%v617 init: %v618) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v622 = stablehlo.broadcast_in_dim %v621, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v623 = stablehlo.divide %v622, %v619 : tensor<32x160x14x14xf32>
    %v624 = stablehlo.subtract %v617, %v623 : tensor<32x160x14x14xf32>
    %v625 = stablehlo.multiply %v624, %v624 : tensor<32x160x14x14xf32>
    %v626 = stablehlo.reduce(%v625 init: %v618) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v627 = stablehlo.broadcast_in_dim %v626, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v628 = stablehlo.divide %v627, %v619 : tensor<32x160x14x14xf32>
    %v629 = stablehlo.add %v628, %v620 : tensor<32x160x14x14xf32>
    %v630 = stablehlo.rsqrt %v629 : tensor<32x160x14x14xf32>
    %v631 = stablehlo.multiply %v624, %v630 : tensor<32x160x14x14xf32>
    %v632 = stablehlo.broadcast_in_dim %u6qg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v633 = stablehlo.broadcast_in_dim %u6qbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v634 = stablehlo.multiply %v631, %v632 : tensor<32x160x14x14xf32>
    %v635 = stablehlo.add %v634, %v633 : tensor<32x160x14x14xf32>
    %v636 = stablehlo.reshape %v635 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v637 = stablehlo.constant dense<0.0> : tensor<32x31360xf32>
    %v638 = stablehlo.maximum %v636, %v637 : tensor<32x31360xf32>
    %v639 = stablehlo.reshape %v638 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v640 = stablehlo.convolution(%v639, %u6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<32x640x14x14xf32>
    %v641 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v642 = stablehlo.add %v640, %v641 : tensor<32x640x14x14xf32>
    %v643 = stablehlo.reshape %v642 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v644 = stablehlo.reshape %v643 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v645 = stablehlo.constant dense<0.0> : tensor<f32>
    %v646 = stablehlo.constant dense<6272.0> : tensor<32x640x14x14xf32>
    %v647 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v648 = stablehlo.reduce(%v644 init: %v645) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v649 = stablehlo.broadcast_in_dim %v648, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v650 = stablehlo.divide %v649, %v646 : tensor<32x640x14x14xf32>
    %v651 = stablehlo.subtract %v644, %v650 : tensor<32x640x14x14xf32>
    %v652 = stablehlo.multiply %v651, %v651 : tensor<32x640x14x14xf32>
    %v653 = stablehlo.reduce(%v652 init: %v645) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v654 = stablehlo.broadcast_in_dim %v653, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v655 = stablehlo.divide %v654, %v646 : tensor<32x640x14x14xf32>
    %v656 = stablehlo.add %v655, %v647 : tensor<32x640x14x14xf32>
    %v657 = stablehlo.rsqrt %v656 : tensor<32x640x14x14xf32>
    %v658 = stablehlo.multiply %v651, %v657 : tensor<32x640x14x14xf32>
    %v659 = stablehlo.broadcast_in_dim %u6eg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v660 = stablehlo.broadcast_in_dim %u6ebt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v661 = stablehlo.multiply %v658, %v659 : tensor<32x640x14x14xf32>
    %v662 = stablehlo.add %v661, %v660 : tensor<32x640x14x14xf32>
    %v663 = stablehlo.reshape %v662 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v664 = stablehlo.constant dense<0.0> : tensor<32x125440xf32>
    %v665 = stablehlo.maximum %v663, %v664 : tensor<32x125440xf32>
    %v666 = stablehlo.reshape %v665 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v667 = stablehlo.convolution(%v666, %u6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 640 : i64} : (tensor<32x640x14x14xf32>, tensor<640x1x5x5xf32>) -> tensor<32x640x14x14xf32>
    %v668 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v669 = stablehlo.add %v667, %v668 : tensor<32x640x14x14xf32>
    %v670 = stablehlo.reshape %v669 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v671 = stablehlo.reshape %v670 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v672 = stablehlo.constant dense<0.0> : tensor<f32>
    %v673 = stablehlo.constant dense<6272.0> : tensor<32x640x14x14xf32>
    %v674 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v675 = stablehlo.reduce(%v671 init: %v672) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v676 = stablehlo.broadcast_in_dim %v675, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v677 = stablehlo.divide %v676, %v673 : tensor<32x640x14x14xf32>
    %v678 = stablehlo.subtract %v671, %v677 : tensor<32x640x14x14xf32>
    %v679 = stablehlo.multiply %v678, %v678 : tensor<32x640x14x14xf32>
    %v680 = stablehlo.reduce(%v679 init: %v672) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v681 = stablehlo.broadcast_in_dim %v680, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v682 = stablehlo.divide %v681, %v673 : tensor<32x640x14x14xf32>
    %v683 = stablehlo.add %v682, %v674 : tensor<32x640x14x14xf32>
    %v684 = stablehlo.rsqrt %v683 : tensor<32x640x14x14xf32>
    %v685 = stablehlo.multiply %v678, %v684 : tensor<32x640x14x14xf32>
    %v686 = stablehlo.broadcast_in_dim %u6dg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v687 = stablehlo.broadcast_in_dim %u6dbt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v688 = stablehlo.multiply %v685, %v686 : tensor<32x640x14x14xf32>
    %v689 = stablehlo.add %v688, %v687 : tensor<32x640x14x14xf32>
    %v690 = stablehlo.reshape %v689 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v691 = stablehlo.constant dense<0.0> : tensor<32x125440xf32>
    %v692 = stablehlo.maximum %v690, %v691 : tensor<32x125440xf32>
    %v693 = stablehlo.reshape %v692 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v694 = stablehlo.convolution(%v693, %u6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v695 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v696 = stablehlo.add %v694, %v695 : tensor<32x160x14x14xf32>
    %v697 = stablehlo.reshape %v696 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v698 = stablehlo.reshape %v697 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v699 = stablehlo.constant dense<0.0> : tensor<f32>
    %v700 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v701 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v702 = stablehlo.reduce(%v698 init: %v699) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v703 = stablehlo.broadcast_in_dim %v702, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v704 = stablehlo.divide %v703, %v700 : tensor<32x160x14x14xf32>
    %v705 = stablehlo.subtract %v698, %v704 : tensor<32x160x14x14xf32>
    %v706 = stablehlo.multiply %v705, %v705 : tensor<32x160x14x14xf32>
    %v707 = stablehlo.reduce(%v706 init: %v699) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v708 = stablehlo.broadcast_in_dim %v707, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v709 = stablehlo.divide %v708, %v700 : tensor<32x160x14x14xf32>
    %v710 = stablehlo.add %v709, %v701 : tensor<32x160x14x14xf32>
    %v711 = stablehlo.rsqrt %v710 : tensor<32x160x14x14xf32>
    %v712 = stablehlo.multiply %v705, %v711 : tensor<32x160x14x14xf32>
    %v713 = stablehlo.broadcast_in_dim %u6pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v714 = stablehlo.broadcast_in_dim %u6pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v715 = stablehlo.multiply %v712, %v713 : tensor<32x160x14x14xf32>
    %v716 = stablehlo.add %v715, %v714 : tensor<32x160x14x14xf32>
    %v717 = stablehlo.reshape %v716 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v718 = stablehlo.add %v717, %v611 : tensor<32x31360xf32>
    %v719 = stablehlo.reshape %v718 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v720 = stablehlo.convolution(%v719, %u7qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<32x160x14x14xf32>
    %v721 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v722 = stablehlo.add %v720, %v721 : tensor<32x160x14x14xf32>
    %v723 = stablehlo.reshape %v722 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v724 = stablehlo.reshape %v723 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v725 = stablehlo.constant dense<0.0> : tensor<f32>
    %v726 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v727 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v728 = stablehlo.reduce(%v724 init: %v725) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v729 = stablehlo.broadcast_in_dim %v728, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v730 = stablehlo.divide %v729, %v726 : tensor<32x160x14x14xf32>
    %v731 = stablehlo.subtract %v724, %v730 : tensor<32x160x14x14xf32>
    %v732 = stablehlo.multiply %v731, %v731 : tensor<32x160x14x14xf32>
    %v733 = stablehlo.reduce(%v732 init: %v725) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v734 = stablehlo.broadcast_in_dim %v733, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v735 = stablehlo.divide %v734, %v726 : tensor<32x160x14x14xf32>
    %v736 = stablehlo.add %v735, %v727 : tensor<32x160x14x14xf32>
    %v737 = stablehlo.rsqrt %v736 : tensor<32x160x14x14xf32>
    %v738 = stablehlo.multiply %v731, %v737 : tensor<32x160x14x14xf32>
    %v739 = stablehlo.broadcast_in_dim %u7qg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v740 = stablehlo.broadcast_in_dim %u7qbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v741 = stablehlo.multiply %v738, %v739 : tensor<32x160x14x14xf32>
    %v742 = stablehlo.add %v741, %v740 : tensor<32x160x14x14xf32>
    %v743 = stablehlo.reshape %v742 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v744 = stablehlo.constant dense<0.0> : tensor<32x31360xf32>
    %v745 = stablehlo.maximum %v743, %v744 : tensor<32x31360xf32>
    %v746 = stablehlo.reshape %v745 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v747 = stablehlo.convolution(%v746, %u7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<32x640x14x14xf32>
    %v748 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v749 = stablehlo.add %v747, %v748 : tensor<32x640x14x14xf32>
    %v750 = stablehlo.reshape %v749 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v751 = stablehlo.reshape %v750 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v752 = stablehlo.constant dense<0.0> : tensor<f32>
    %v753 = stablehlo.constant dense<6272.0> : tensor<32x640x14x14xf32>
    %v754 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v755 = stablehlo.reduce(%v751 init: %v752) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v756 = stablehlo.broadcast_in_dim %v755, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v757 = stablehlo.divide %v756, %v753 : tensor<32x640x14x14xf32>
    %v758 = stablehlo.subtract %v751, %v757 : tensor<32x640x14x14xf32>
    %v759 = stablehlo.multiply %v758, %v758 : tensor<32x640x14x14xf32>
    %v760 = stablehlo.reduce(%v759 init: %v752) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v761 = stablehlo.broadcast_in_dim %v760, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v762 = stablehlo.divide %v761, %v753 : tensor<32x640x14x14xf32>
    %v763 = stablehlo.add %v762, %v754 : tensor<32x640x14x14xf32>
    %v764 = stablehlo.rsqrt %v763 : tensor<32x640x14x14xf32>
    %v765 = stablehlo.multiply %v758, %v764 : tensor<32x640x14x14xf32>
    %v766 = stablehlo.broadcast_in_dim %u7eg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v767 = stablehlo.broadcast_in_dim %u7ebt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v768 = stablehlo.multiply %v765, %v766 : tensor<32x640x14x14xf32>
    %v769 = stablehlo.add %v768, %v767 : tensor<32x640x14x14xf32>
    %v770 = stablehlo.reshape %v769 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v771 = stablehlo.constant dense<0.0> : tensor<32x125440xf32>
    %v772 = stablehlo.maximum %v770, %v771 : tensor<32x125440xf32>
    %v773 = stablehlo.reshape %v772 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v774 = stablehlo.convolution(%v773, %u7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 640 : i64} : (tensor<32x640x14x14xf32>, tensor<640x1x3x3xf32>) -> tensor<32x640x14x14xf32>
    %v775 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v776 = stablehlo.add %v774, %v775 : tensor<32x640x14x14xf32>
    %v777 = stablehlo.reshape %v776 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v778 = stablehlo.reshape %v777 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v779 = stablehlo.constant dense<0.0> : tensor<f32>
    %v780 = stablehlo.constant dense<6272.0> : tensor<32x640x14x14xf32>
    %v781 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v782 = stablehlo.reduce(%v778 init: %v779) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v783 = stablehlo.broadcast_in_dim %v782, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v784 = stablehlo.divide %v783, %v780 : tensor<32x640x14x14xf32>
    %v785 = stablehlo.subtract %v778, %v784 : tensor<32x640x14x14xf32>
    %v786 = stablehlo.multiply %v785, %v785 : tensor<32x640x14x14xf32>
    %v787 = stablehlo.reduce(%v786 init: %v779) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v788 = stablehlo.broadcast_in_dim %v787, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v789 = stablehlo.divide %v788, %v780 : tensor<32x640x14x14xf32>
    %v790 = stablehlo.add %v789, %v781 : tensor<32x640x14x14xf32>
    %v791 = stablehlo.rsqrt %v790 : tensor<32x640x14x14xf32>
    %v792 = stablehlo.multiply %v785, %v791 : tensor<32x640x14x14xf32>
    %v793 = stablehlo.broadcast_in_dim %u7dg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v794 = stablehlo.broadcast_in_dim %u7dbt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v795 = stablehlo.multiply %v792, %v793 : tensor<32x640x14x14xf32>
    %v796 = stablehlo.add %v795, %v794 : tensor<32x640x14x14xf32>
    %v797 = stablehlo.reshape %v796 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v798 = stablehlo.constant dense<0.0> : tensor<32x125440xf32>
    %v799 = stablehlo.maximum %v797, %v798 : tensor<32x125440xf32>
    %v800 = stablehlo.reshape %v799 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v801 = stablehlo.convolution(%v800, %u7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v802 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v803 = stablehlo.add %v801, %v802 : tensor<32x160x14x14xf32>
    %v804 = stablehlo.reshape %v803 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v805 = stablehlo.reshape %v804 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v806 = stablehlo.constant dense<0.0> : tensor<f32>
    %v807 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v808 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v809 = stablehlo.reduce(%v805 init: %v806) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v810 = stablehlo.broadcast_in_dim %v809, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v811 = stablehlo.divide %v810, %v807 : tensor<32x160x14x14xf32>
    %v812 = stablehlo.subtract %v805, %v811 : tensor<32x160x14x14xf32>
    %v813 = stablehlo.multiply %v812, %v812 : tensor<32x160x14x14xf32>
    %v814 = stablehlo.reduce(%v813 init: %v806) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v815 = stablehlo.broadcast_in_dim %v814, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v816 = stablehlo.divide %v815, %v807 : tensor<32x160x14x14xf32>
    %v817 = stablehlo.add %v816, %v808 : tensor<32x160x14x14xf32>
    %v818 = stablehlo.rsqrt %v817 : tensor<32x160x14x14xf32>
    %v819 = stablehlo.multiply %v812, %v818 : tensor<32x160x14x14xf32>
    %v820 = stablehlo.broadcast_in_dim %u7pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v821 = stablehlo.broadcast_in_dim %u7pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v822 = stablehlo.multiply %v819, %v820 : tensor<32x160x14x14xf32>
    %v823 = stablehlo.add %v822, %v821 : tensor<32x160x14x14xf32>
    %v824 = stablehlo.reshape %v823 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v825 = stablehlo.add %v824, %v718 : tensor<32x31360xf32>
    %v826 = stablehlo.reshape %v825 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v827 = stablehlo.convolution(%v826, %u8qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<32x160x14x14xf32>
    %v828 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v829 = stablehlo.add %v827, %v828 : tensor<32x160x14x14xf32>
    %v830 = stablehlo.reshape %v829 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v831 = stablehlo.reshape %v830 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v832 = stablehlo.constant dense<0.0> : tensor<f32>
    %v833 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v834 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v835 = stablehlo.reduce(%v831 init: %v832) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v836 = stablehlo.broadcast_in_dim %v835, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v837 = stablehlo.divide %v836, %v833 : tensor<32x160x14x14xf32>
    %v838 = stablehlo.subtract %v831, %v837 : tensor<32x160x14x14xf32>
    %v839 = stablehlo.multiply %v838, %v838 : tensor<32x160x14x14xf32>
    %v840 = stablehlo.reduce(%v839 init: %v832) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v841 = stablehlo.broadcast_in_dim %v840, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v842 = stablehlo.divide %v841, %v833 : tensor<32x160x14x14xf32>
    %v843 = stablehlo.add %v842, %v834 : tensor<32x160x14x14xf32>
    %v844 = stablehlo.rsqrt %v843 : tensor<32x160x14x14xf32>
    %v845 = stablehlo.multiply %v838, %v844 : tensor<32x160x14x14xf32>
    %v846 = stablehlo.broadcast_in_dim %u8qg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v847 = stablehlo.broadcast_in_dim %u8qbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v848 = stablehlo.multiply %v845, %v846 : tensor<32x160x14x14xf32>
    %v849 = stablehlo.add %v848, %v847 : tensor<32x160x14x14xf32>
    %v850 = stablehlo.reshape %v849 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v851 = stablehlo.constant dense<0.0> : tensor<32x31360xf32>
    %v852 = stablehlo.maximum %v850, %v851 : tensor<32x31360xf32>
    %v853 = stablehlo.reshape %v852 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v854 = stablehlo.convolution(%v853, %u8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<32x640x14x14xf32>
    %v855 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v856 = stablehlo.add %v854, %v855 : tensor<32x640x14x14xf32>
    %v857 = stablehlo.reshape %v856 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v858 = stablehlo.reshape %v857 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v859 = stablehlo.constant dense<0.0> : tensor<f32>
    %v860 = stablehlo.constant dense<6272.0> : tensor<32x640x14x14xf32>
    %v861 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v862 = stablehlo.reduce(%v858 init: %v859) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v863 = stablehlo.broadcast_in_dim %v862, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v864 = stablehlo.divide %v863, %v860 : tensor<32x640x14x14xf32>
    %v865 = stablehlo.subtract %v858, %v864 : tensor<32x640x14x14xf32>
    %v866 = stablehlo.multiply %v865, %v865 : tensor<32x640x14x14xf32>
    %v867 = stablehlo.reduce(%v866 init: %v859) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v868 = stablehlo.broadcast_in_dim %v867, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v869 = stablehlo.divide %v868, %v860 : tensor<32x640x14x14xf32>
    %v870 = stablehlo.add %v869, %v861 : tensor<32x640x14x14xf32>
    %v871 = stablehlo.rsqrt %v870 : tensor<32x640x14x14xf32>
    %v872 = stablehlo.multiply %v865, %v871 : tensor<32x640x14x14xf32>
    %v873 = stablehlo.broadcast_in_dim %u8eg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v874 = stablehlo.broadcast_in_dim %u8ebt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v875 = stablehlo.multiply %v872, %v873 : tensor<32x640x14x14xf32>
    %v876 = stablehlo.add %v875, %v874 : tensor<32x640x14x14xf32>
    %v877 = stablehlo.reshape %v876 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v878 = stablehlo.constant dense<0.0> : tensor<32x125440xf32>
    %v879 = stablehlo.maximum %v877, %v878 : tensor<32x125440xf32>
    %v880 = stablehlo.reshape %v879 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v881 = stablehlo.convolution(%v880, %u8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v882 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v883 = stablehlo.add %v881, %v882 : tensor<32x160x14x14xf32>
    %v884 = stablehlo.reshape %v883 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v885 = stablehlo.reshape %v884 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v886 = stablehlo.constant dense<0.0> : tensor<f32>
    %v887 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v888 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v889 = stablehlo.reduce(%v885 init: %v886) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v890 = stablehlo.broadcast_in_dim %v889, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v891 = stablehlo.divide %v890, %v887 : tensor<32x160x14x14xf32>
    %v892 = stablehlo.subtract %v885, %v891 : tensor<32x160x14x14xf32>
    %v893 = stablehlo.multiply %v892, %v892 : tensor<32x160x14x14xf32>
    %v894 = stablehlo.reduce(%v893 init: %v886) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v895 = stablehlo.broadcast_in_dim %v894, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v896 = stablehlo.divide %v895, %v887 : tensor<32x160x14x14xf32>
    %v897 = stablehlo.add %v896, %v888 : tensor<32x160x14x14xf32>
    %v898 = stablehlo.rsqrt %v897 : tensor<32x160x14x14xf32>
    %v899 = stablehlo.multiply %v892, %v898 : tensor<32x160x14x14xf32>
    %v900 = stablehlo.broadcast_in_dim %u8pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v901 = stablehlo.broadcast_in_dim %u8pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v902 = stablehlo.multiply %v899, %v900 : tensor<32x160x14x14xf32>
    %v903 = stablehlo.add %v902, %v901 : tensor<32x160x14x14xf32>
    %v904 = stablehlo.reshape %v903 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v905 = stablehlo.add %v904, %v825 : tensor<32x31360xf32>
    %v906 = stablehlo.reshape %v905 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v907 = stablehlo.convolution(%v906, %u9eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x14x14xf32>, tensor<320x160x1x1xf32>) -> tensor<32x320x14x14xf32>
    %v908 = stablehlo.broadcast_in_dim %zb320, dims = [1] : (tensor<320xf32>) -> tensor<32x320x14x14xf32>
    %v909 = stablehlo.add %v907, %v908 : tensor<32x320x14x14xf32>
    %v910 = stablehlo.reshape %v909 : (tensor<32x320x14x14xf32>) -> tensor<32x62720xf32>
    %v911 = stablehlo.reshape %v910 : (tensor<32x62720xf32>) -> tensor<32x320x14x14xf32>
    %v912 = stablehlo.constant dense<0.0> : tensor<f32>
    %v913 = stablehlo.constant dense<6272.0> : tensor<32x320x14x14xf32>
    %v914 = stablehlo.constant dense<1.0e-5> : tensor<32x320x14x14xf32>
    %v915 = stablehlo.reduce(%v911 init: %v912) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x14x14xf32>, tensor<f32>) -> tensor<320xf32>
    %v916 = stablehlo.broadcast_in_dim %v915, dims = [1] : (tensor<320xf32>) -> tensor<32x320x14x14xf32>
    %v917 = stablehlo.divide %v916, %v913 : tensor<32x320x14x14xf32>
    %v918 = stablehlo.subtract %v911, %v917 : tensor<32x320x14x14xf32>
    %v919 = stablehlo.multiply %v918, %v918 : tensor<32x320x14x14xf32>
    %v920 = stablehlo.reduce(%v919 init: %v912) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x14x14xf32>, tensor<f32>) -> tensor<320xf32>
    %v921 = stablehlo.broadcast_in_dim %v920, dims = [1] : (tensor<320xf32>) -> tensor<32x320x14x14xf32>
    %v922 = stablehlo.divide %v921, %v913 : tensor<32x320x14x14xf32>
    %v923 = stablehlo.add %v922, %v914 : tensor<32x320x14x14xf32>
    %v924 = stablehlo.rsqrt %v923 : tensor<32x320x14x14xf32>
    %v925 = stablehlo.multiply %v918, %v924 : tensor<32x320x14x14xf32>
    %v926 = stablehlo.broadcast_in_dim %u9eg, dims = [1] : (tensor<320xf32>) -> tensor<32x320x14x14xf32>
    %v927 = stablehlo.broadcast_in_dim %u9ebt, dims = [1] : (tensor<320xf32>) -> tensor<32x320x14x14xf32>
    %v928 = stablehlo.multiply %v925, %v926 : tensor<32x320x14x14xf32>
    %v929 = stablehlo.add %v928, %v927 : tensor<32x320x14x14xf32>
    %v930 = stablehlo.reshape %v929 : (tensor<32x320x14x14xf32>) -> tensor<32x62720xf32>
    %v931 = stablehlo.constant dense<0.0> : tensor<32x62720xf32>
    %v932 = stablehlo.maximum %v930, %v931 : tensor<32x62720xf32>
    %v933 = stablehlo.reshape %v932 : (tensor<32x62720xf32>) -> tensor<32x320x14x14xf32>
    %v934 = stablehlo.convolution(%v933, %u9pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x320x14x14xf32>, tensor<160x320x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v935 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v936 = stablehlo.add %v934, %v935 : tensor<32x160x14x14xf32>
    %v937 = stablehlo.reshape %v936 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v938 = stablehlo.reshape %v937 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v939 = stablehlo.constant dense<0.0> : tensor<f32>
    %v940 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v941 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v942 = stablehlo.reduce(%v938 init: %v939) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v943 = stablehlo.broadcast_in_dim %v942, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v944 = stablehlo.divide %v943, %v940 : tensor<32x160x14x14xf32>
    %v945 = stablehlo.subtract %v938, %v944 : tensor<32x160x14x14xf32>
    %v946 = stablehlo.multiply %v945, %v945 : tensor<32x160x14x14xf32>
    %v947 = stablehlo.reduce(%v946 init: %v939) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v948 = stablehlo.broadcast_in_dim %v947, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v949 = stablehlo.divide %v948, %v940 : tensor<32x160x14x14xf32>
    %v950 = stablehlo.add %v949, %v941 : tensor<32x160x14x14xf32>
    %v951 = stablehlo.rsqrt %v950 : tensor<32x160x14x14xf32>
    %v952 = stablehlo.multiply %v945, %v951 : tensor<32x160x14x14xf32>
    %v953 = stablehlo.broadcast_in_dim %u9pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v954 = stablehlo.broadcast_in_dim %u9pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v955 = stablehlo.multiply %v952, %v953 : tensor<32x160x14x14xf32>
    %v956 = stablehlo.add %v955, %v954 : tensor<32x160x14x14xf32>
    %v957 = stablehlo.reshape %v956 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v958 = stablehlo.add %v957, %v905 : tensor<32x31360xf32>
    %v959 = stablehlo.reshape %v958 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v960 = stablehlo.convolution(%v959, %u10qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<32x160x14x14xf32>
    %v961 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v962 = stablehlo.add %v960, %v961 : tensor<32x160x14x14xf32>
    %v963 = stablehlo.reshape %v962 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v964 = stablehlo.reshape %v963 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v965 = stablehlo.constant dense<0.0> : tensor<f32>
    %v966 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v967 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v968 = stablehlo.reduce(%v964 init: %v965) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v969 = stablehlo.broadcast_in_dim %v968, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v970 = stablehlo.divide %v969, %v966 : tensor<32x160x14x14xf32>
    %v971 = stablehlo.subtract %v964, %v970 : tensor<32x160x14x14xf32>
    %v972 = stablehlo.multiply %v971, %v971 : tensor<32x160x14x14xf32>
    %v973 = stablehlo.reduce(%v972 init: %v965) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v974 = stablehlo.broadcast_in_dim %v973, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v975 = stablehlo.divide %v974, %v966 : tensor<32x160x14x14xf32>
    %v976 = stablehlo.add %v975, %v967 : tensor<32x160x14x14xf32>
    %v977 = stablehlo.rsqrt %v976 : tensor<32x160x14x14xf32>
    %v978 = stablehlo.multiply %v971, %v977 : tensor<32x160x14x14xf32>
    %v979 = stablehlo.broadcast_in_dim %u10qg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v980 = stablehlo.broadcast_in_dim %u10qbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v981 = stablehlo.multiply %v978, %v979 : tensor<32x160x14x14xf32>
    %v982 = stablehlo.add %v981, %v980 : tensor<32x160x14x14xf32>
    %v983 = stablehlo.reshape %v982 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v984 = stablehlo.constant dense<0.0> : tensor<32x31360xf32>
    %v985 = stablehlo.maximum %v983, %v984 : tensor<32x31360xf32>
    %v986 = stablehlo.reshape %v985 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v987 = stablehlo.convolution(%v986, %u10eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<32x640x14x14xf32>
    %v988 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v989 = stablehlo.add %v987, %v988 : tensor<32x640x14x14xf32>
    %v990 = stablehlo.reshape %v989 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v991 = stablehlo.reshape %v990 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v992 = stablehlo.constant dense<0.0> : tensor<f32>
    %v993 = stablehlo.constant dense<6272.0> : tensor<32x640x14x14xf32>
    %v994 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v995 = stablehlo.reduce(%v991 init: %v992) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v996 = stablehlo.broadcast_in_dim %v995, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v997 = stablehlo.divide %v996, %v993 : tensor<32x640x14x14xf32>
    %v998 = stablehlo.subtract %v991, %v997 : tensor<32x640x14x14xf32>
    %v999 = stablehlo.multiply %v998, %v998 : tensor<32x640x14x14xf32>
    %v1000 = stablehlo.reduce(%v999 init: %v992) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v1001 = stablehlo.broadcast_in_dim %v1000, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v1002 = stablehlo.divide %v1001, %v993 : tensor<32x640x14x14xf32>
    %v1003 = stablehlo.add %v1002, %v994 : tensor<32x640x14x14xf32>
    %v1004 = stablehlo.rsqrt %v1003 : tensor<32x640x14x14xf32>
    %v1005 = stablehlo.multiply %v998, %v1004 : tensor<32x640x14x14xf32>
    %v1006 = stablehlo.broadcast_in_dim %u10eg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v1007 = stablehlo.broadcast_in_dim %u10ebt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v1008 = stablehlo.multiply %v1005, %v1006 : tensor<32x640x14x14xf32>
    %v1009 = stablehlo.add %v1008, %v1007 : tensor<32x640x14x14xf32>
    %v1010 = stablehlo.reshape %v1009 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v1011 = stablehlo.constant dense<0.0> : tensor<32x125440xf32>
    %v1012 = stablehlo.maximum %v1010, %v1011 : tensor<32x125440xf32>
    %v1013 = stablehlo.reshape %v1012 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v1014 = stablehlo.convolution(%v1013, %u10pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v1015 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v1016 = stablehlo.add %v1014, %v1015 : tensor<32x160x14x14xf32>
    %v1017 = stablehlo.reshape %v1016 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v1018 = stablehlo.reshape %v1017 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v1019 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1020 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v1021 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v1022 = stablehlo.reduce(%v1018 init: %v1019) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v1023 = stablehlo.broadcast_in_dim %v1022, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v1024 = stablehlo.divide %v1023, %v1020 : tensor<32x160x14x14xf32>
    %v1025 = stablehlo.subtract %v1018, %v1024 : tensor<32x160x14x14xf32>
    %v1026 = stablehlo.multiply %v1025, %v1025 : tensor<32x160x14x14xf32>
    %v1027 = stablehlo.reduce(%v1026 init: %v1019) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v1028 = stablehlo.broadcast_in_dim %v1027, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v1029 = stablehlo.divide %v1028, %v1020 : tensor<32x160x14x14xf32>
    %v1030 = stablehlo.add %v1029, %v1021 : tensor<32x160x14x14xf32>
    %v1031 = stablehlo.rsqrt %v1030 : tensor<32x160x14x14xf32>
    %v1032 = stablehlo.multiply %v1025, %v1031 : tensor<32x160x14x14xf32>
    %v1033 = stablehlo.broadcast_in_dim %u10pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v1034 = stablehlo.broadcast_in_dim %u10pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v1035 = stablehlo.multiply %v1032, %v1033 : tensor<32x160x14x14xf32>
    %v1036 = stablehlo.add %v1035, %v1034 : tensor<32x160x14x14xf32>
    %v1037 = stablehlo.reshape %v1036 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v1038 = stablehlo.add %v1037, %v958 : tensor<32x31360xf32>
    %v1039 = stablehlo.reshape %v1038 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v1040 = stablehlo.convolution(%v1039, %u11qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x14x14xf32>, tensor<160x1x5x5xf32>) -> tensor<32x160x7x7xf32>
    %v1041 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1042 = stablehlo.add %v1040, %v1041 : tensor<32x160x7x7xf32>
    %v1043 = stablehlo.reshape %v1042 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1044 = stablehlo.reshape %v1043 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1045 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1046 = stablehlo.constant dense<1568.0> : tensor<32x160x7x7xf32>
    %v1047 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v1048 = stablehlo.reduce(%v1044 init: %v1045) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v1049 = stablehlo.broadcast_in_dim %v1048, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1050 = stablehlo.divide %v1049, %v1046 : tensor<32x160x7x7xf32>
    %v1051 = stablehlo.subtract %v1044, %v1050 : tensor<32x160x7x7xf32>
    %v1052 = stablehlo.multiply %v1051, %v1051 : tensor<32x160x7x7xf32>
    %v1053 = stablehlo.reduce(%v1052 init: %v1045) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x7x7xf32>, tensor<f32>) -> tensor<160xf32>
    %v1054 = stablehlo.broadcast_in_dim %v1053, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1055 = stablehlo.divide %v1054, %v1046 : tensor<32x160x7x7xf32>
    %v1056 = stablehlo.add %v1055, %v1047 : tensor<32x160x7x7xf32>
    %v1057 = stablehlo.rsqrt %v1056 : tensor<32x160x7x7xf32>
    %v1058 = stablehlo.multiply %v1051, %v1057 : tensor<32x160x7x7xf32>
    %v1059 = stablehlo.broadcast_in_dim %u11qg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1060 = stablehlo.broadcast_in_dim %u11qbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v1061 = stablehlo.multiply %v1058, %v1059 : tensor<32x160x7x7xf32>
    %v1062 = stablehlo.add %v1061, %v1060 : tensor<32x160x7x7xf32>
    %v1063 = stablehlo.reshape %v1062 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v1064 = stablehlo.constant dense<0.0> : tensor<32x7840xf32>
    %v1065 = stablehlo.maximum %v1063, %v1064 : tensor<32x7840xf32>
    %v1066 = stablehlo.reshape %v1065 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v1067 = stablehlo.convolution(%v1066, %u11eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v1068 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1069 = stablehlo.add %v1067, %v1068 : tensor<32x960x7x7xf32>
    %v1070 = stablehlo.reshape %v1069 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1071 = stablehlo.reshape %v1070 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1072 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1073 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v1074 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1075 = stablehlo.reduce(%v1071 init: %v1072) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1076 = stablehlo.broadcast_in_dim %v1075, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1077 = stablehlo.divide %v1076, %v1073 : tensor<32x960x7x7xf32>
    %v1078 = stablehlo.subtract %v1071, %v1077 : tensor<32x960x7x7xf32>
    %v1079 = stablehlo.multiply %v1078, %v1078 : tensor<32x960x7x7xf32>
    %v1080 = stablehlo.reduce(%v1079 init: %v1072) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1081 = stablehlo.broadcast_in_dim %v1080, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1082 = stablehlo.divide %v1081, %v1073 : tensor<32x960x7x7xf32>
    %v1083 = stablehlo.add %v1082, %v1074 : tensor<32x960x7x7xf32>
    %v1084 = stablehlo.rsqrt %v1083 : tensor<32x960x7x7xf32>
    %v1085 = stablehlo.multiply %v1078, %v1084 : tensor<32x960x7x7xf32>
    %v1086 = stablehlo.broadcast_in_dim %u11eg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1087 = stablehlo.broadcast_in_dim %u11ebt, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1088 = stablehlo.multiply %v1085, %v1086 : tensor<32x960x7x7xf32>
    %v1089 = stablehlo.add %v1088, %v1087 : tensor<32x960x7x7xf32>
    %v1090 = stablehlo.reshape %v1089 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1091 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v1092 = stablehlo.maximum %v1090, %v1091 : tensor<32x47040xf32>
    %v1093 = stablehlo.reshape %v1092 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1094 = stablehlo.convolution(%v1093, %u11dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x5x5xf32>) -> tensor<32x960x7x7xf32>
    %v1095 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1096 = stablehlo.add %v1094, %v1095 : tensor<32x960x7x7xf32>
    %v1097 = stablehlo.reshape %v1096 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1098 = stablehlo.reshape %v1097 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1099 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1100 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v1101 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1102 = stablehlo.reduce(%v1098 init: %v1099) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1103 = stablehlo.broadcast_in_dim %v1102, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1104 = stablehlo.divide %v1103, %v1100 : tensor<32x960x7x7xf32>
    %v1105 = stablehlo.subtract %v1098, %v1104 : tensor<32x960x7x7xf32>
    %v1106 = stablehlo.multiply %v1105, %v1105 : tensor<32x960x7x7xf32>
    %v1107 = stablehlo.reduce(%v1106 init: %v1099) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1108 = stablehlo.broadcast_in_dim %v1107, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1109 = stablehlo.divide %v1108, %v1100 : tensor<32x960x7x7xf32>
    %v1110 = stablehlo.add %v1109, %v1101 : tensor<32x960x7x7xf32>
    %v1111 = stablehlo.rsqrt %v1110 : tensor<32x960x7x7xf32>
    %v1112 = stablehlo.multiply %v1105, %v1111 : tensor<32x960x7x7xf32>
    %v1113 = stablehlo.broadcast_in_dim %u11dg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1114 = stablehlo.broadcast_in_dim %u11dbt, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1115 = stablehlo.multiply %v1112, %v1113 : tensor<32x960x7x7xf32>
    %v1116 = stablehlo.add %v1115, %v1114 : tensor<32x960x7x7xf32>
    %v1117 = stablehlo.reshape %v1116 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1118 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v1119 = stablehlo.maximum %v1117, %v1118 : tensor<32x47040xf32>
    %v1120 = stablehlo.reshape %v1119 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1121 = stablehlo.convolution(%v1120, %u11pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<256x960x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1122 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1123 = stablehlo.add %v1121, %v1122 : tensor<32x256x7x7xf32>
    %v1124 = stablehlo.reshape %v1123 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1125 = stablehlo.reshape %v1124 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1126 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1127 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1128 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1129 = stablehlo.reduce(%v1125 init: %v1126) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1130 = stablehlo.broadcast_in_dim %v1129, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1131 = stablehlo.divide %v1130, %v1127 : tensor<32x256x7x7xf32>
    %v1132 = stablehlo.subtract %v1125, %v1131 : tensor<32x256x7x7xf32>
    %v1133 = stablehlo.multiply %v1132, %v1132 : tensor<32x256x7x7xf32>
    %v1134 = stablehlo.reduce(%v1133 init: %v1126) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1135 = stablehlo.broadcast_in_dim %v1134, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1136 = stablehlo.divide %v1135, %v1127 : tensor<32x256x7x7xf32>
    %v1137 = stablehlo.add %v1136, %v1128 : tensor<32x256x7x7xf32>
    %v1138 = stablehlo.rsqrt %v1137 : tensor<32x256x7x7xf32>
    %v1139 = stablehlo.multiply %v1132, %v1138 : tensor<32x256x7x7xf32>
    %v1140 = stablehlo.broadcast_in_dim %u11pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1141 = stablehlo.broadcast_in_dim %u11pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1142 = stablehlo.multiply %v1139, %v1140 : tensor<32x256x7x7xf32>
    %v1143 = stablehlo.add %v1142, %v1141 : tensor<32x256x7x7xf32>
    %v1144 = stablehlo.reshape %v1143 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1145 = stablehlo.reshape %v1144 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1146 = stablehlo.convolution(%v1145, %u12qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x7x7xf32>, tensor<256x1x5x5xf32>) -> tensor<32x256x7x7xf32>
    %v1147 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1148 = stablehlo.add %v1146, %v1147 : tensor<32x256x7x7xf32>
    %v1149 = stablehlo.reshape %v1148 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1150 = stablehlo.reshape %v1149 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1151 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1152 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1153 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1154 = stablehlo.reduce(%v1150 init: %v1151) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1155 = stablehlo.broadcast_in_dim %v1154, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1156 = stablehlo.divide %v1155, %v1152 : tensor<32x256x7x7xf32>
    %v1157 = stablehlo.subtract %v1150, %v1156 : tensor<32x256x7x7xf32>
    %v1158 = stablehlo.multiply %v1157, %v1157 : tensor<32x256x7x7xf32>
    %v1159 = stablehlo.reduce(%v1158 init: %v1151) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1160 = stablehlo.broadcast_in_dim %v1159, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1161 = stablehlo.divide %v1160, %v1152 : tensor<32x256x7x7xf32>
    %v1162 = stablehlo.add %v1161, %v1153 : tensor<32x256x7x7xf32>
    %v1163 = stablehlo.rsqrt %v1162 : tensor<32x256x7x7xf32>
    %v1164 = stablehlo.multiply %v1157, %v1163 : tensor<32x256x7x7xf32>
    %v1165 = stablehlo.broadcast_in_dim %u12qg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1166 = stablehlo.broadcast_in_dim %u12qbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1167 = stablehlo.multiply %v1164, %v1165 : tensor<32x256x7x7xf32>
    %v1168 = stablehlo.add %v1167, %v1166 : tensor<32x256x7x7xf32>
    %v1169 = stablehlo.reshape %v1168 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1170 = stablehlo.constant dense<0.0> : tensor<32x12544xf32>
    %v1171 = stablehlo.maximum %v1169, %v1170 : tensor<32x12544xf32>
    %v1172 = stablehlo.reshape %v1171 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1173 = stablehlo.convolution(%v1172, %u12eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v1174 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1175 = stablehlo.add %v1173, %v1174 : tensor<32x1024x7x7xf32>
    %v1176 = stablehlo.reshape %v1175 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1177 = stablehlo.reshape %v1176 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1178 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1179 = stablehlo.constant dense<1568.0> : tensor<32x1024x7x7xf32>
    %v1180 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1181 = stablehlo.reduce(%v1177 init: %v1178) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1182 = stablehlo.broadcast_in_dim %v1181, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1183 = stablehlo.divide %v1182, %v1179 : tensor<32x1024x7x7xf32>
    %v1184 = stablehlo.subtract %v1177, %v1183 : tensor<32x1024x7x7xf32>
    %v1185 = stablehlo.multiply %v1184, %v1184 : tensor<32x1024x7x7xf32>
    %v1186 = stablehlo.reduce(%v1185 init: %v1178) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1187 = stablehlo.broadcast_in_dim %v1186, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1188 = stablehlo.divide %v1187, %v1179 : tensor<32x1024x7x7xf32>
    %v1189 = stablehlo.add %v1188, %v1180 : tensor<32x1024x7x7xf32>
    %v1190 = stablehlo.rsqrt %v1189 : tensor<32x1024x7x7xf32>
    %v1191 = stablehlo.multiply %v1184, %v1190 : tensor<32x1024x7x7xf32>
    %v1192 = stablehlo.broadcast_in_dim %u12eg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1193 = stablehlo.broadcast_in_dim %u12ebt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1194 = stablehlo.multiply %v1191, %v1192 : tensor<32x1024x7x7xf32>
    %v1195 = stablehlo.add %v1194, %v1193 : tensor<32x1024x7x7xf32>
    %v1196 = stablehlo.reshape %v1195 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1197 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1198 = stablehlo.maximum %v1196, %v1197 : tensor<32x50176xf32>
    %v1199 = stablehlo.reshape %v1198 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1200 = stablehlo.convolution(%v1199, %u12dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<32x1024x7x7xf32>, tensor<1024x1x5x5xf32>) -> tensor<32x1024x7x7xf32>
    %v1201 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1202 = stablehlo.add %v1200, %v1201 : tensor<32x1024x7x7xf32>
    %v1203 = stablehlo.reshape %v1202 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1204 = stablehlo.reshape %v1203 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1205 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1206 = stablehlo.constant dense<1568.0> : tensor<32x1024x7x7xf32>
    %v1207 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1208 = stablehlo.reduce(%v1204 init: %v1205) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1209 = stablehlo.broadcast_in_dim %v1208, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1210 = stablehlo.divide %v1209, %v1206 : tensor<32x1024x7x7xf32>
    %v1211 = stablehlo.subtract %v1204, %v1210 : tensor<32x1024x7x7xf32>
    %v1212 = stablehlo.multiply %v1211, %v1211 : tensor<32x1024x7x7xf32>
    %v1213 = stablehlo.reduce(%v1212 init: %v1205) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1214 = stablehlo.broadcast_in_dim %v1213, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1215 = stablehlo.divide %v1214, %v1206 : tensor<32x1024x7x7xf32>
    %v1216 = stablehlo.add %v1215, %v1207 : tensor<32x1024x7x7xf32>
    %v1217 = stablehlo.rsqrt %v1216 : tensor<32x1024x7x7xf32>
    %v1218 = stablehlo.multiply %v1211, %v1217 : tensor<32x1024x7x7xf32>
    %v1219 = stablehlo.broadcast_in_dim %u12dg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1220 = stablehlo.broadcast_in_dim %u12dbt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1221 = stablehlo.multiply %v1218, %v1219 : tensor<32x1024x7x7xf32>
    %v1222 = stablehlo.add %v1221, %v1220 : tensor<32x1024x7x7xf32>
    %v1223 = stablehlo.reshape %v1222 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1224 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1225 = stablehlo.maximum %v1223, %v1224 : tensor<32x50176xf32>
    %v1226 = stablehlo.reshape %v1225 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1227 = stablehlo.convolution(%v1226, %u12pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1228 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1229 = stablehlo.add %v1227, %v1228 : tensor<32x256x7x7xf32>
    %v1230 = stablehlo.reshape %v1229 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1231 = stablehlo.reshape %v1230 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1232 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1233 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1234 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1235 = stablehlo.reduce(%v1231 init: %v1232) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1236 = stablehlo.broadcast_in_dim %v1235, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1237 = stablehlo.divide %v1236, %v1233 : tensor<32x256x7x7xf32>
    %v1238 = stablehlo.subtract %v1231, %v1237 : tensor<32x256x7x7xf32>
    %v1239 = stablehlo.multiply %v1238, %v1238 : tensor<32x256x7x7xf32>
    %v1240 = stablehlo.reduce(%v1239 init: %v1232) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1241 = stablehlo.broadcast_in_dim %v1240, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1242 = stablehlo.divide %v1241, %v1233 : tensor<32x256x7x7xf32>
    %v1243 = stablehlo.add %v1242, %v1234 : tensor<32x256x7x7xf32>
    %v1244 = stablehlo.rsqrt %v1243 : tensor<32x256x7x7xf32>
    %v1245 = stablehlo.multiply %v1238, %v1244 : tensor<32x256x7x7xf32>
    %v1246 = stablehlo.broadcast_in_dim %u12pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1247 = stablehlo.broadcast_in_dim %u12pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1248 = stablehlo.multiply %v1245, %v1246 : tensor<32x256x7x7xf32>
    %v1249 = stablehlo.add %v1248, %v1247 : tensor<32x256x7x7xf32>
    %v1250 = stablehlo.reshape %v1249 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1251 = stablehlo.add %v1250, %v1144 : tensor<32x12544xf32>
    %v1252 = stablehlo.reshape %v1251 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1253 = stablehlo.convolution(%v1252, %u13qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x7x7xf32>, tensor<256x1x3x3xf32>) -> tensor<32x256x7x7xf32>
    %v1254 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1255 = stablehlo.add %v1253, %v1254 : tensor<32x256x7x7xf32>
    %v1256 = stablehlo.reshape %v1255 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1257 = stablehlo.reshape %v1256 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1258 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1259 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1260 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1261 = stablehlo.reduce(%v1257 init: %v1258) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1262 = stablehlo.broadcast_in_dim %v1261, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1263 = stablehlo.divide %v1262, %v1259 : tensor<32x256x7x7xf32>
    %v1264 = stablehlo.subtract %v1257, %v1263 : tensor<32x256x7x7xf32>
    %v1265 = stablehlo.multiply %v1264, %v1264 : tensor<32x256x7x7xf32>
    %v1266 = stablehlo.reduce(%v1265 init: %v1258) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1267 = stablehlo.broadcast_in_dim %v1266, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1268 = stablehlo.divide %v1267, %v1259 : tensor<32x256x7x7xf32>
    %v1269 = stablehlo.add %v1268, %v1260 : tensor<32x256x7x7xf32>
    %v1270 = stablehlo.rsqrt %v1269 : tensor<32x256x7x7xf32>
    %v1271 = stablehlo.multiply %v1264, %v1270 : tensor<32x256x7x7xf32>
    %v1272 = stablehlo.broadcast_in_dim %u13qg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1273 = stablehlo.broadcast_in_dim %u13qbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1274 = stablehlo.multiply %v1271, %v1272 : tensor<32x256x7x7xf32>
    %v1275 = stablehlo.add %v1274, %v1273 : tensor<32x256x7x7xf32>
    %v1276 = stablehlo.reshape %v1275 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1277 = stablehlo.constant dense<0.0> : tensor<32x12544xf32>
    %v1278 = stablehlo.maximum %v1276, %v1277 : tensor<32x12544xf32>
    %v1279 = stablehlo.reshape %v1278 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1280 = stablehlo.convolution(%v1279, %u13eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v1281 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1282 = stablehlo.add %v1280, %v1281 : tensor<32x1024x7x7xf32>
    %v1283 = stablehlo.reshape %v1282 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1284 = stablehlo.reshape %v1283 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1285 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1286 = stablehlo.constant dense<1568.0> : tensor<32x1024x7x7xf32>
    %v1287 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1288 = stablehlo.reduce(%v1284 init: %v1285) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1289 = stablehlo.broadcast_in_dim %v1288, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1290 = stablehlo.divide %v1289, %v1286 : tensor<32x1024x7x7xf32>
    %v1291 = stablehlo.subtract %v1284, %v1290 : tensor<32x1024x7x7xf32>
    %v1292 = stablehlo.multiply %v1291, %v1291 : tensor<32x1024x7x7xf32>
    %v1293 = stablehlo.reduce(%v1292 init: %v1285) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1294 = stablehlo.broadcast_in_dim %v1293, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1295 = stablehlo.divide %v1294, %v1286 : tensor<32x1024x7x7xf32>
    %v1296 = stablehlo.add %v1295, %v1287 : tensor<32x1024x7x7xf32>
    %v1297 = stablehlo.rsqrt %v1296 : tensor<32x1024x7x7xf32>
    %v1298 = stablehlo.multiply %v1291, %v1297 : tensor<32x1024x7x7xf32>
    %v1299 = stablehlo.broadcast_in_dim %u13eg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1300 = stablehlo.broadcast_in_dim %u13ebt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1301 = stablehlo.multiply %v1298, %v1299 : tensor<32x1024x7x7xf32>
    %v1302 = stablehlo.add %v1301, %v1300 : tensor<32x1024x7x7xf32>
    %v1303 = stablehlo.reshape %v1302 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1304 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1305 = stablehlo.maximum %v1303, %v1304 : tensor<32x50176xf32>
    %v1306 = stablehlo.reshape %v1305 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1307 = stablehlo.convolution(%v1306, %u13dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<32x1024x7x7xf32>, tensor<1024x1x5x5xf32>) -> tensor<32x1024x7x7xf32>
    %v1308 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1309 = stablehlo.add %v1307, %v1308 : tensor<32x1024x7x7xf32>
    %v1310 = stablehlo.reshape %v1309 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1311 = stablehlo.reshape %v1310 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1312 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1313 = stablehlo.constant dense<1568.0> : tensor<32x1024x7x7xf32>
    %v1314 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1315 = stablehlo.reduce(%v1311 init: %v1312) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1316 = stablehlo.broadcast_in_dim %v1315, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1317 = stablehlo.divide %v1316, %v1313 : tensor<32x1024x7x7xf32>
    %v1318 = stablehlo.subtract %v1311, %v1317 : tensor<32x1024x7x7xf32>
    %v1319 = stablehlo.multiply %v1318, %v1318 : tensor<32x1024x7x7xf32>
    %v1320 = stablehlo.reduce(%v1319 init: %v1312) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1321 = stablehlo.broadcast_in_dim %v1320, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1322 = stablehlo.divide %v1321, %v1313 : tensor<32x1024x7x7xf32>
    %v1323 = stablehlo.add %v1322, %v1314 : tensor<32x1024x7x7xf32>
    %v1324 = stablehlo.rsqrt %v1323 : tensor<32x1024x7x7xf32>
    %v1325 = stablehlo.multiply %v1318, %v1324 : tensor<32x1024x7x7xf32>
    %v1326 = stablehlo.broadcast_in_dim %u13dg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1327 = stablehlo.broadcast_in_dim %u13dbt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1328 = stablehlo.multiply %v1325, %v1326 : tensor<32x1024x7x7xf32>
    %v1329 = stablehlo.add %v1328, %v1327 : tensor<32x1024x7x7xf32>
    %v1330 = stablehlo.reshape %v1329 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1331 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1332 = stablehlo.maximum %v1330, %v1331 : tensor<32x50176xf32>
    %v1333 = stablehlo.reshape %v1332 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1334 = stablehlo.convolution(%v1333, %u13pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1335 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1336 = stablehlo.add %v1334, %v1335 : tensor<32x256x7x7xf32>
    %v1337 = stablehlo.reshape %v1336 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1338 = stablehlo.reshape %v1337 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1339 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1340 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1341 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1342 = stablehlo.reduce(%v1338 init: %v1339) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1343 = stablehlo.broadcast_in_dim %v1342, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1344 = stablehlo.divide %v1343, %v1340 : tensor<32x256x7x7xf32>
    %v1345 = stablehlo.subtract %v1338, %v1344 : tensor<32x256x7x7xf32>
    %v1346 = stablehlo.multiply %v1345, %v1345 : tensor<32x256x7x7xf32>
    %v1347 = stablehlo.reduce(%v1346 init: %v1339) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1348 = stablehlo.broadcast_in_dim %v1347, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1349 = stablehlo.divide %v1348, %v1340 : tensor<32x256x7x7xf32>
    %v1350 = stablehlo.add %v1349, %v1341 : tensor<32x256x7x7xf32>
    %v1351 = stablehlo.rsqrt %v1350 : tensor<32x256x7x7xf32>
    %v1352 = stablehlo.multiply %v1345, %v1351 : tensor<32x256x7x7xf32>
    %v1353 = stablehlo.broadcast_in_dim %u13pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1354 = stablehlo.broadcast_in_dim %u13pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1355 = stablehlo.multiply %v1352, %v1353 : tensor<32x256x7x7xf32>
    %v1356 = stablehlo.add %v1355, %v1354 : tensor<32x256x7x7xf32>
    %v1357 = stablehlo.reshape %v1356 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1358 = stablehlo.add %v1357, %v1251 : tensor<32x12544xf32>
    %v1359 = stablehlo.reshape %v1358 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1360 = stablehlo.convolution(%v1359, %u14qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x7x7xf32>, tensor<256x1x3x3xf32>) -> tensor<32x256x7x7xf32>
    %v1361 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1362 = stablehlo.add %v1360, %v1361 : tensor<32x256x7x7xf32>
    %v1363 = stablehlo.reshape %v1362 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1364 = stablehlo.reshape %v1363 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1365 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1366 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1367 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1368 = stablehlo.reduce(%v1364 init: %v1365) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1369 = stablehlo.broadcast_in_dim %v1368, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1370 = stablehlo.divide %v1369, %v1366 : tensor<32x256x7x7xf32>
    %v1371 = stablehlo.subtract %v1364, %v1370 : tensor<32x256x7x7xf32>
    %v1372 = stablehlo.multiply %v1371, %v1371 : tensor<32x256x7x7xf32>
    %v1373 = stablehlo.reduce(%v1372 init: %v1365) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1374 = stablehlo.broadcast_in_dim %v1373, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1375 = stablehlo.divide %v1374, %v1366 : tensor<32x256x7x7xf32>
    %v1376 = stablehlo.add %v1375, %v1367 : tensor<32x256x7x7xf32>
    %v1377 = stablehlo.rsqrt %v1376 : tensor<32x256x7x7xf32>
    %v1378 = stablehlo.multiply %v1371, %v1377 : tensor<32x256x7x7xf32>
    %v1379 = stablehlo.broadcast_in_dim %u14qg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1380 = stablehlo.broadcast_in_dim %u14qbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1381 = stablehlo.multiply %v1378, %v1379 : tensor<32x256x7x7xf32>
    %v1382 = stablehlo.add %v1381, %v1380 : tensor<32x256x7x7xf32>
    %v1383 = stablehlo.reshape %v1382 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1384 = stablehlo.constant dense<0.0> : tensor<32x12544xf32>
    %v1385 = stablehlo.maximum %v1383, %v1384 : tensor<32x12544xf32>
    %v1386 = stablehlo.reshape %v1385 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1387 = stablehlo.convolution(%v1386, %u14eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v1388 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1389 = stablehlo.add %v1387, %v1388 : tensor<32x1024x7x7xf32>
    %v1390 = stablehlo.reshape %v1389 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1391 = stablehlo.reshape %v1390 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1392 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1393 = stablehlo.constant dense<1568.0> : tensor<32x1024x7x7xf32>
    %v1394 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1395 = stablehlo.reduce(%v1391 init: %v1392) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1396 = stablehlo.broadcast_in_dim %v1395, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1397 = stablehlo.divide %v1396, %v1393 : tensor<32x1024x7x7xf32>
    %v1398 = stablehlo.subtract %v1391, %v1397 : tensor<32x1024x7x7xf32>
    %v1399 = stablehlo.multiply %v1398, %v1398 : tensor<32x1024x7x7xf32>
    %v1400 = stablehlo.reduce(%v1399 init: %v1392) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1401 = stablehlo.broadcast_in_dim %v1400, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1402 = stablehlo.divide %v1401, %v1393 : tensor<32x1024x7x7xf32>
    %v1403 = stablehlo.add %v1402, %v1394 : tensor<32x1024x7x7xf32>
    %v1404 = stablehlo.rsqrt %v1403 : tensor<32x1024x7x7xf32>
    %v1405 = stablehlo.multiply %v1398, %v1404 : tensor<32x1024x7x7xf32>
    %v1406 = stablehlo.broadcast_in_dim %u14eg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1407 = stablehlo.broadcast_in_dim %u14ebt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1408 = stablehlo.multiply %v1405, %v1406 : tensor<32x1024x7x7xf32>
    %v1409 = stablehlo.add %v1408, %v1407 : tensor<32x1024x7x7xf32>
    %v1410 = stablehlo.reshape %v1409 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1411 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1412 = stablehlo.maximum %v1410, %v1411 : tensor<32x50176xf32>
    %v1413 = stablehlo.reshape %v1412 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1414 = stablehlo.convolution(%v1413, %u14dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<32x1024x7x7xf32>, tensor<1024x1x5x5xf32>) -> tensor<32x1024x7x7xf32>
    %v1415 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1416 = stablehlo.add %v1414, %v1415 : tensor<32x1024x7x7xf32>
    %v1417 = stablehlo.reshape %v1416 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1418 = stablehlo.reshape %v1417 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1419 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1420 = stablehlo.constant dense<1568.0> : tensor<32x1024x7x7xf32>
    %v1421 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1422 = stablehlo.reduce(%v1418 init: %v1419) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1423 = stablehlo.broadcast_in_dim %v1422, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1424 = stablehlo.divide %v1423, %v1420 : tensor<32x1024x7x7xf32>
    %v1425 = stablehlo.subtract %v1418, %v1424 : tensor<32x1024x7x7xf32>
    %v1426 = stablehlo.multiply %v1425, %v1425 : tensor<32x1024x7x7xf32>
    %v1427 = stablehlo.reduce(%v1426 init: %v1419) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1428 = stablehlo.broadcast_in_dim %v1427, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1429 = stablehlo.divide %v1428, %v1420 : tensor<32x1024x7x7xf32>
    %v1430 = stablehlo.add %v1429, %v1421 : tensor<32x1024x7x7xf32>
    %v1431 = stablehlo.rsqrt %v1430 : tensor<32x1024x7x7xf32>
    %v1432 = stablehlo.multiply %v1425, %v1431 : tensor<32x1024x7x7xf32>
    %v1433 = stablehlo.broadcast_in_dim %u14dg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1434 = stablehlo.broadcast_in_dim %u14dbt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1435 = stablehlo.multiply %v1432, %v1433 : tensor<32x1024x7x7xf32>
    %v1436 = stablehlo.add %v1435, %v1434 : tensor<32x1024x7x7xf32>
    %v1437 = stablehlo.reshape %v1436 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1438 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1439 = stablehlo.maximum %v1437, %v1438 : tensor<32x50176xf32>
    %v1440 = stablehlo.reshape %v1439 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1441 = stablehlo.convolution(%v1440, %u14pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1442 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1443 = stablehlo.add %v1441, %v1442 : tensor<32x256x7x7xf32>
    %v1444 = stablehlo.reshape %v1443 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1445 = stablehlo.reshape %v1444 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1446 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1447 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1448 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1449 = stablehlo.reduce(%v1445 init: %v1446) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1450 = stablehlo.broadcast_in_dim %v1449, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1451 = stablehlo.divide %v1450, %v1447 : tensor<32x256x7x7xf32>
    %v1452 = stablehlo.subtract %v1445, %v1451 : tensor<32x256x7x7xf32>
    %v1453 = stablehlo.multiply %v1452, %v1452 : tensor<32x256x7x7xf32>
    %v1454 = stablehlo.reduce(%v1453 init: %v1446) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1455 = stablehlo.broadcast_in_dim %v1454, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1456 = stablehlo.divide %v1455, %v1447 : tensor<32x256x7x7xf32>
    %v1457 = stablehlo.add %v1456, %v1448 : tensor<32x256x7x7xf32>
    %v1458 = stablehlo.rsqrt %v1457 : tensor<32x256x7x7xf32>
    %v1459 = stablehlo.multiply %v1452, %v1458 : tensor<32x256x7x7xf32>
    %v1460 = stablehlo.broadcast_in_dim %u14pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1461 = stablehlo.broadcast_in_dim %u14pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1462 = stablehlo.multiply %v1459, %v1460 : tensor<32x256x7x7xf32>
    %v1463 = stablehlo.add %v1462, %v1461 : tensor<32x256x7x7xf32>
    %v1464 = stablehlo.reshape %v1463 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1465 = stablehlo.add %v1464, %v1358 : tensor<32x12544xf32>
    %v1466 = stablehlo.reshape %v1465 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1467 = stablehlo.convolution(%v1466, %u15eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v1468 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1469 = stablehlo.add %v1467, %v1468 : tensor<32x1024x7x7xf32>
    %v1470 = stablehlo.reshape %v1469 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1471 = stablehlo.reshape %v1470 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1472 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1473 = stablehlo.constant dense<1568.0> : tensor<32x1024x7x7xf32>
    %v1474 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1475 = stablehlo.reduce(%v1471 init: %v1472) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1476 = stablehlo.broadcast_in_dim %v1475, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1477 = stablehlo.divide %v1476, %v1473 : tensor<32x1024x7x7xf32>
    %v1478 = stablehlo.subtract %v1471, %v1477 : tensor<32x1024x7x7xf32>
    %v1479 = stablehlo.multiply %v1478, %v1478 : tensor<32x1024x7x7xf32>
    %v1480 = stablehlo.reduce(%v1479 init: %v1472) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1481 = stablehlo.broadcast_in_dim %v1480, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1482 = stablehlo.divide %v1481, %v1473 : tensor<32x1024x7x7xf32>
    %v1483 = stablehlo.add %v1482, %v1474 : tensor<32x1024x7x7xf32>
    %v1484 = stablehlo.rsqrt %v1483 : tensor<32x1024x7x7xf32>
    %v1485 = stablehlo.multiply %v1478, %v1484 : tensor<32x1024x7x7xf32>
    %v1486 = stablehlo.broadcast_in_dim %u15eg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1487 = stablehlo.broadcast_in_dim %u15ebt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1488 = stablehlo.multiply %v1485, %v1486 : tensor<32x1024x7x7xf32>
    %v1489 = stablehlo.add %v1488, %v1487 : tensor<32x1024x7x7xf32>
    %v1490 = stablehlo.reshape %v1489 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1491 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1492 = stablehlo.maximum %v1490, %v1491 : tensor<32x50176xf32>
    %v1493 = stablehlo.reshape %v1492 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1494 = stablehlo.convolution(%v1493, %u15pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1495 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1496 = stablehlo.add %v1494, %v1495 : tensor<32x256x7x7xf32>
    %v1497 = stablehlo.reshape %v1496 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1498 = stablehlo.reshape %v1497 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1499 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1500 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1501 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1502 = stablehlo.reduce(%v1498 init: %v1499) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1503 = stablehlo.broadcast_in_dim %v1502, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1504 = stablehlo.divide %v1503, %v1500 : tensor<32x256x7x7xf32>
    %v1505 = stablehlo.subtract %v1498, %v1504 : tensor<32x256x7x7xf32>
    %v1506 = stablehlo.multiply %v1505, %v1505 : tensor<32x256x7x7xf32>
    %v1507 = stablehlo.reduce(%v1506 init: %v1499) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1508 = stablehlo.broadcast_in_dim %v1507, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1509 = stablehlo.divide %v1508, %v1500 : tensor<32x256x7x7xf32>
    %v1510 = stablehlo.add %v1509, %v1501 : tensor<32x256x7x7xf32>
    %v1511 = stablehlo.rsqrt %v1510 : tensor<32x256x7x7xf32>
    %v1512 = stablehlo.multiply %v1505, %v1511 : tensor<32x256x7x7xf32>
    %v1513 = stablehlo.broadcast_in_dim %u15pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1514 = stablehlo.broadcast_in_dim %u15pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1515 = stablehlo.multiply %v1512, %v1513 : tensor<32x256x7x7xf32>
    %v1516 = stablehlo.add %v1515, %v1514 : tensor<32x256x7x7xf32>
    %v1517 = stablehlo.reshape %v1516 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1518 = stablehlo.add %v1517, %v1465 : tensor<32x12544xf32>
    %v1519 = stablehlo.reshape %v1518 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1520 = stablehlo.convolution(%v1519, %u16qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x7x7xf32>, tensor<256x1x3x3xf32>) -> tensor<32x256x7x7xf32>
    %v1521 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1522 = stablehlo.add %v1520, %v1521 : tensor<32x256x7x7xf32>
    %v1523 = stablehlo.reshape %v1522 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1524 = stablehlo.reshape %v1523 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1525 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1526 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1527 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1528 = stablehlo.reduce(%v1524 init: %v1525) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1529 = stablehlo.broadcast_in_dim %v1528, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1530 = stablehlo.divide %v1529, %v1526 : tensor<32x256x7x7xf32>
    %v1531 = stablehlo.subtract %v1524, %v1530 : tensor<32x256x7x7xf32>
    %v1532 = stablehlo.multiply %v1531, %v1531 : tensor<32x256x7x7xf32>
    %v1533 = stablehlo.reduce(%v1532 init: %v1525) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1534 = stablehlo.broadcast_in_dim %v1533, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1535 = stablehlo.divide %v1534, %v1526 : tensor<32x256x7x7xf32>
    %v1536 = stablehlo.add %v1535, %v1527 : tensor<32x256x7x7xf32>
    %v1537 = stablehlo.rsqrt %v1536 : tensor<32x256x7x7xf32>
    %v1538 = stablehlo.multiply %v1531, %v1537 : tensor<32x256x7x7xf32>
    %v1539 = stablehlo.broadcast_in_dim %u16qg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1540 = stablehlo.broadcast_in_dim %u16qbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1541 = stablehlo.multiply %v1538, %v1539 : tensor<32x256x7x7xf32>
    %v1542 = stablehlo.add %v1541, %v1540 : tensor<32x256x7x7xf32>
    %v1543 = stablehlo.reshape %v1542 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1544 = stablehlo.constant dense<0.0> : tensor<32x12544xf32>
    %v1545 = stablehlo.maximum %v1543, %v1544 : tensor<32x12544xf32>
    %v1546 = stablehlo.reshape %v1545 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1547 = stablehlo.convolution(%v1546, %u16eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v1548 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1549 = stablehlo.add %v1547, %v1548 : tensor<32x1024x7x7xf32>
    %v1550 = stablehlo.reshape %v1549 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1551 = stablehlo.reshape %v1550 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1552 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1553 = stablehlo.constant dense<1568.0> : tensor<32x1024x7x7xf32>
    %v1554 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1555 = stablehlo.reduce(%v1551 init: %v1552) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1556 = stablehlo.broadcast_in_dim %v1555, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1557 = stablehlo.divide %v1556, %v1553 : tensor<32x1024x7x7xf32>
    %v1558 = stablehlo.subtract %v1551, %v1557 : tensor<32x1024x7x7xf32>
    %v1559 = stablehlo.multiply %v1558, %v1558 : tensor<32x1024x7x7xf32>
    %v1560 = stablehlo.reduce(%v1559 init: %v1552) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1561 = stablehlo.broadcast_in_dim %v1560, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1562 = stablehlo.divide %v1561, %v1553 : tensor<32x1024x7x7xf32>
    %v1563 = stablehlo.add %v1562, %v1554 : tensor<32x1024x7x7xf32>
    %v1564 = stablehlo.rsqrt %v1563 : tensor<32x1024x7x7xf32>
    %v1565 = stablehlo.multiply %v1558, %v1564 : tensor<32x1024x7x7xf32>
    %v1566 = stablehlo.broadcast_in_dim %u16eg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1567 = stablehlo.broadcast_in_dim %u16ebt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1568 = stablehlo.multiply %v1565, %v1566 : tensor<32x1024x7x7xf32>
    %v1569 = stablehlo.add %v1568, %v1567 : tensor<32x1024x7x7xf32>
    %v1570 = stablehlo.reshape %v1569 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1571 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1572 = stablehlo.maximum %v1570, %v1571 : tensor<32x50176xf32>
    %v1573 = stablehlo.reshape %v1572 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1574 = stablehlo.convolution(%v1573, %u16pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1575 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1576 = stablehlo.add %v1574, %v1575 : tensor<32x256x7x7xf32>
    %v1577 = stablehlo.reshape %v1576 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1578 = stablehlo.reshape %v1577 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1579 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1580 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1581 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1582 = stablehlo.reduce(%v1578 init: %v1579) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1583 = stablehlo.broadcast_in_dim %v1582, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1584 = stablehlo.divide %v1583, %v1580 : tensor<32x256x7x7xf32>
    %v1585 = stablehlo.subtract %v1578, %v1584 : tensor<32x256x7x7xf32>
    %v1586 = stablehlo.multiply %v1585, %v1585 : tensor<32x256x7x7xf32>
    %v1587 = stablehlo.reduce(%v1586 init: %v1579) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1588 = stablehlo.broadcast_in_dim %v1587, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1589 = stablehlo.divide %v1588, %v1580 : tensor<32x256x7x7xf32>
    %v1590 = stablehlo.add %v1589, %v1581 : tensor<32x256x7x7xf32>
    %v1591 = stablehlo.rsqrt %v1590 : tensor<32x256x7x7xf32>
    %v1592 = stablehlo.multiply %v1585, %v1591 : tensor<32x256x7x7xf32>
    %v1593 = stablehlo.broadcast_in_dim %u16pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1594 = stablehlo.broadcast_in_dim %u16pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1595 = stablehlo.multiply %v1592, %v1593 : tensor<32x256x7x7xf32>
    %v1596 = stablehlo.add %v1595, %v1594 : tensor<32x256x7x7xf32>
    %v1597 = stablehlo.reshape %v1596 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1598 = stablehlo.add %v1597, %v1518 : tensor<32x12544xf32>
    %v1599 = stablehlo.reshape %v1598 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1600 = stablehlo.convolution(%v1599, %u17qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x7x7xf32>, tensor<256x1x3x3xf32>) -> tensor<32x256x7x7xf32>
    %v1601 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1602 = stablehlo.add %v1600, %v1601 : tensor<32x256x7x7xf32>
    %v1603 = stablehlo.reshape %v1602 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1604 = stablehlo.reshape %v1603 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1605 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1606 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1607 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1608 = stablehlo.reduce(%v1604 init: %v1605) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1609 = stablehlo.broadcast_in_dim %v1608, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1610 = stablehlo.divide %v1609, %v1606 : tensor<32x256x7x7xf32>
    %v1611 = stablehlo.subtract %v1604, %v1610 : tensor<32x256x7x7xf32>
    %v1612 = stablehlo.multiply %v1611, %v1611 : tensor<32x256x7x7xf32>
    %v1613 = stablehlo.reduce(%v1612 init: %v1605) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1614 = stablehlo.broadcast_in_dim %v1613, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1615 = stablehlo.divide %v1614, %v1606 : tensor<32x256x7x7xf32>
    %v1616 = stablehlo.add %v1615, %v1607 : tensor<32x256x7x7xf32>
    %v1617 = stablehlo.rsqrt %v1616 : tensor<32x256x7x7xf32>
    %v1618 = stablehlo.multiply %v1611, %v1617 : tensor<32x256x7x7xf32>
    %v1619 = stablehlo.broadcast_in_dim %u17qg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1620 = stablehlo.broadcast_in_dim %u17qbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1621 = stablehlo.multiply %v1618, %v1619 : tensor<32x256x7x7xf32>
    %v1622 = stablehlo.add %v1621, %v1620 : tensor<32x256x7x7xf32>
    %v1623 = stablehlo.reshape %v1622 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1624 = stablehlo.constant dense<0.0> : tensor<32x12544xf32>
    %v1625 = stablehlo.maximum %v1623, %v1624 : tensor<32x12544xf32>
    %v1626 = stablehlo.reshape %v1625 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1627 = stablehlo.convolution(%v1626, %u17eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<512x256x1x1xf32>) -> tensor<32x512x7x7xf32>
    %v1628 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1629 = stablehlo.add %v1627, %v1628 : tensor<32x512x7x7xf32>
    %v1630 = stablehlo.reshape %v1629 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1631 = stablehlo.reshape %v1630 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1632 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1633 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1634 = stablehlo.constant dense<1.0e-5> : tensor<32x512x7x7xf32>
    %v1635 = stablehlo.reduce(%v1631 init: %v1632) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1636 = stablehlo.broadcast_in_dim %v1635, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1637 = stablehlo.divide %v1636, %v1633 : tensor<32x512x7x7xf32>
    %v1638 = stablehlo.subtract %v1631, %v1637 : tensor<32x512x7x7xf32>
    %v1639 = stablehlo.multiply %v1638, %v1638 : tensor<32x512x7x7xf32>
    %v1640 = stablehlo.reduce(%v1639 init: %v1632) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1641 = stablehlo.broadcast_in_dim %v1640, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1642 = stablehlo.divide %v1641, %v1633 : tensor<32x512x7x7xf32>
    %v1643 = stablehlo.add %v1642, %v1634 : tensor<32x512x7x7xf32>
    %v1644 = stablehlo.rsqrt %v1643 : tensor<32x512x7x7xf32>
    %v1645 = stablehlo.multiply %v1638, %v1644 : tensor<32x512x7x7xf32>
    %v1646 = stablehlo.broadcast_in_dim %u17eg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1647 = stablehlo.broadcast_in_dim %u17ebt, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1648 = stablehlo.multiply %v1645, %v1646 : tensor<32x512x7x7xf32>
    %v1649 = stablehlo.add %v1648, %v1647 : tensor<32x512x7x7xf32>
    %v1650 = stablehlo.reshape %v1649 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1651 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v1652 = stablehlo.maximum %v1650, %v1651 : tensor<32x25088xf32>
    %v1653 = stablehlo.reshape %v1652 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1654 = stablehlo.convolution(%v1653, %u17dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x7x7xf32>, tensor<512x1x5x5xf32>) -> tensor<32x512x7x7xf32>
    %v1655 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1656 = stablehlo.add %v1654, %v1655 : tensor<32x512x7x7xf32>
    %v1657 = stablehlo.reshape %v1656 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1658 = stablehlo.reshape %v1657 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1659 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1660 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1661 = stablehlo.constant dense<1.0e-5> : tensor<32x512x7x7xf32>
    %v1662 = stablehlo.reduce(%v1658 init: %v1659) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1663 = stablehlo.broadcast_in_dim %v1662, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1664 = stablehlo.divide %v1663, %v1660 : tensor<32x512x7x7xf32>
    %v1665 = stablehlo.subtract %v1658, %v1664 : tensor<32x512x7x7xf32>
    %v1666 = stablehlo.multiply %v1665, %v1665 : tensor<32x512x7x7xf32>
    %v1667 = stablehlo.reduce(%v1666 init: %v1659) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1668 = stablehlo.broadcast_in_dim %v1667, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1669 = stablehlo.divide %v1668, %v1660 : tensor<32x512x7x7xf32>
    %v1670 = stablehlo.add %v1669, %v1661 : tensor<32x512x7x7xf32>
    %v1671 = stablehlo.rsqrt %v1670 : tensor<32x512x7x7xf32>
    %v1672 = stablehlo.multiply %v1665, %v1671 : tensor<32x512x7x7xf32>
    %v1673 = stablehlo.broadcast_in_dim %u17dg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1674 = stablehlo.broadcast_in_dim %u17dbt, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1675 = stablehlo.multiply %v1672, %v1673 : tensor<32x512x7x7xf32>
    %v1676 = stablehlo.add %v1675, %v1674 : tensor<32x512x7x7xf32>
    %v1677 = stablehlo.reshape %v1676 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1678 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v1679 = stablehlo.maximum %v1677, %v1678 : tensor<32x25088xf32>
    %v1680 = stablehlo.reshape %v1679 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1681 = stablehlo.convolution(%v1680, %u17pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<256x512x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1682 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1683 = stablehlo.add %v1681, %v1682 : tensor<32x256x7x7xf32>
    %v1684 = stablehlo.reshape %v1683 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1685 = stablehlo.reshape %v1684 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1686 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1687 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1688 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1689 = stablehlo.reduce(%v1685 init: %v1686) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1690 = stablehlo.broadcast_in_dim %v1689, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1691 = stablehlo.divide %v1690, %v1687 : tensor<32x256x7x7xf32>
    %v1692 = stablehlo.subtract %v1685, %v1691 : tensor<32x256x7x7xf32>
    %v1693 = stablehlo.multiply %v1692, %v1692 : tensor<32x256x7x7xf32>
    %v1694 = stablehlo.reduce(%v1693 init: %v1686) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1695 = stablehlo.broadcast_in_dim %v1694, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1696 = stablehlo.divide %v1695, %v1687 : tensor<32x256x7x7xf32>
    %v1697 = stablehlo.add %v1696, %v1688 : tensor<32x256x7x7xf32>
    %v1698 = stablehlo.rsqrt %v1697 : tensor<32x256x7x7xf32>
    %v1699 = stablehlo.multiply %v1692, %v1698 : tensor<32x256x7x7xf32>
    %v1700 = stablehlo.broadcast_in_dim %u17pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1701 = stablehlo.broadcast_in_dim %u17pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1702 = stablehlo.multiply %v1699, %v1700 : tensor<32x256x7x7xf32>
    %v1703 = stablehlo.add %v1702, %v1701 : tensor<32x256x7x7xf32>
    %v1704 = stablehlo.reshape %v1703 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1705 = stablehlo.add %v1704, %v1598 : tensor<32x12544xf32>
    %v1706 = stablehlo.reshape %v1705 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1707 = stablehlo.convolution(%v1706, %u18qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x7x7xf32>, tensor<256x1x5x5xf32>) -> tensor<32x256x7x7xf32>
    %v1708 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1709 = stablehlo.add %v1707, %v1708 : tensor<32x256x7x7xf32>
    %v1710 = stablehlo.reshape %v1709 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1711 = stablehlo.reshape %v1710 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1712 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1713 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1714 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1715 = stablehlo.reduce(%v1711 init: %v1712) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1716 = stablehlo.broadcast_in_dim %v1715, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1717 = stablehlo.divide %v1716, %v1713 : tensor<32x256x7x7xf32>
    %v1718 = stablehlo.subtract %v1711, %v1717 : tensor<32x256x7x7xf32>
    %v1719 = stablehlo.multiply %v1718, %v1718 : tensor<32x256x7x7xf32>
    %v1720 = stablehlo.reduce(%v1719 init: %v1712) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1721 = stablehlo.broadcast_in_dim %v1720, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1722 = stablehlo.divide %v1721, %v1713 : tensor<32x256x7x7xf32>
    %v1723 = stablehlo.add %v1722, %v1714 : tensor<32x256x7x7xf32>
    %v1724 = stablehlo.rsqrt %v1723 : tensor<32x256x7x7xf32>
    %v1725 = stablehlo.multiply %v1718, %v1724 : tensor<32x256x7x7xf32>
    %v1726 = stablehlo.broadcast_in_dim %u18qg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1727 = stablehlo.broadcast_in_dim %u18qbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1728 = stablehlo.multiply %v1725, %v1726 : tensor<32x256x7x7xf32>
    %v1729 = stablehlo.add %v1728, %v1727 : tensor<32x256x7x7xf32>
    %v1730 = stablehlo.reshape %v1729 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1731 = stablehlo.constant dense<0.0> : tensor<32x12544xf32>
    %v1732 = stablehlo.maximum %v1730, %v1731 : tensor<32x12544xf32>
    %v1733 = stablehlo.reshape %v1732 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1734 = stablehlo.convolution(%v1733, %u18eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v1735 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1736 = stablehlo.add %v1734, %v1735 : tensor<32x1024x7x7xf32>
    %v1737 = stablehlo.reshape %v1736 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1738 = stablehlo.reshape %v1737 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1739 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1740 = stablehlo.constant dense<1568.0> : tensor<32x1024x7x7xf32>
    %v1741 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1742 = stablehlo.reduce(%v1738 init: %v1739) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1743 = stablehlo.broadcast_in_dim %v1742, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1744 = stablehlo.divide %v1743, %v1740 : tensor<32x1024x7x7xf32>
    %v1745 = stablehlo.subtract %v1738, %v1744 : tensor<32x1024x7x7xf32>
    %v1746 = stablehlo.multiply %v1745, %v1745 : tensor<32x1024x7x7xf32>
    %v1747 = stablehlo.reduce(%v1746 init: %v1739) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1748 = stablehlo.broadcast_in_dim %v1747, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1749 = stablehlo.divide %v1748, %v1740 : tensor<32x1024x7x7xf32>
    %v1750 = stablehlo.add %v1749, %v1741 : tensor<32x1024x7x7xf32>
    %v1751 = stablehlo.rsqrt %v1750 : tensor<32x1024x7x7xf32>
    %v1752 = stablehlo.multiply %v1745, %v1751 : tensor<32x1024x7x7xf32>
    %v1753 = stablehlo.broadcast_in_dim %u18eg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1754 = stablehlo.broadcast_in_dim %u18ebt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1755 = stablehlo.multiply %v1752, %v1753 : tensor<32x1024x7x7xf32>
    %v1756 = stablehlo.add %v1755, %v1754 : tensor<32x1024x7x7xf32>
    %v1757 = stablehlo.reshape %v1756 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1758 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1759 = stablehlo.maximum %v1757, %v1758 : tensor<32x50176xf32>
    %v1760 = stablehlo.reshape %v1759 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1761 = stablehlo.convolution(%v1760, %u18dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<32x1024x7x7xf32>, tensor<1024x1x5x5xf32>) -> tensor<32x1024x7x7xf32>
    %v1762 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1763 = stablehlo.add %v1761, %v1762 : tensor<32x1024x7x7xf32>
    %v1764 = stablehlo.reshape %v1763 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1765 = stablehlo.reshape %v1764 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1766 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1767 = stablehlo.constant dense<1568.0> : tensor<32x1024x7x7xf32>
    %v1768 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1769 = stablehlo.reduce(%v1765 init: %v1766) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1770 = stablehlo.broadcast_in_dim %v1769, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1771 = stablehlo.divide %v1770, %v1767 : tensor<32x1024x7x7xf32>
    %v1772 = stablehlo.subtract %v1765, %v1771 : tensor<32x1024x7x7xf32>
    %v1773 = stablehlo.multiply %v1772, %v1772 : tensor<32x1024x7x7xf32>
    %v1774 = stablehlo.reduce(%v1773 init: %v1766) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1775 = stablehlo.broadcast_in_dim %v1774, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1776 = stablehlo.divide %v1775, %v1767 : tensor<32x1024x7x7xf32>
    %v1777 = stablehlo.add %v1776, %v1768 : tensor<32x1024x7x7xf32>
    %v1778 = stablehlo.rsqrt %v1777 : tensor<32x1024x7x7xf32>
    %v1779 = stablehlo.multiply %v1772, %v1778 : tensor<32x1024x7x7xf32>
    %v1780 = stablehlo.broadcast_in_dim %u18dg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1781 = stablehlo.broadcast_in_dim %u18dbt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1782 = stablehlo.multiply %v1779, %v1780 : tensor<32x1024x7x7xf32>
    %v1783 = stablehlo.add %v1782, %v1781 : tensor<32x1024x7x7xf32>
    %v1784 = stablehlo.reshape %v1783 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1785 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1786 = stablehlo.maximum %v1784, %v1785 : tensor<32x50176xf32>
    %v1787 = stablehlo.reshape %v1786 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1788 = stablehlo.convolution(%v1787, %u18pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1789 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1790 = stablehlo.add %v1788, %v1789 : tensor<32x256x7x7xf32>
    %v1791 = stablehlo.reshape %v1790 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1792 = stablehlo.reshape %v1791 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1793 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1794 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1795 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1796 = stablehlo.reduce(%v1792 init: %v1793) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1797 = stablehlo.broadcast_in_dim %v1796, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1798 = stablehlo.divide %v1797, %v1794 : tensor<32x256x7x7xf32>
    %v1799 = stablehlo.subtract %v1792, %v1798 : tensor<32x256x7x7xf32>
    %v1800 = stablehlo.multiply %v1799, %v1799 : tensor<32x256x7x7xf32>
    %v1801 = stablehlo.reduce(%v1800 init: %v1793) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1802 = stablehlo.broadcast_in_dim %v1801, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1803 = stablehlo.divide %v1802, %v1794 : tensor<32x256x7x7xf32>
    %v1804 = stablehlo.add %v1803, %v1795 : tensor<32x256x7x7xf32>
    %v1805 = stablehlo.rsqrt %v1804 : tensor<32x256x7x7xf32>
    %v1806 = stablehlo.multiply %v1799, %v1805 : tensor<32x256x7x7xf32>
    %v1807 = stablehlo.broadcast_in_dim %u18pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1808 = stablehlo.broadcast_in_dim %u18pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1809 = stablehlo.multiply %v1806, %v1807 : tensor<32x256x7x7xf32>
    %v1810 = stablehlo.add %v1809, %v1808 : tensor<32x256x7x7xf32>
    %v1811 = stablehlo.reshape %v1810 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1812 = stablehlo.add %v1811, %v1705 : tensor<32x12544xf32>
    %v1813 = stablehlo.reshape %v1812 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1814 = stablehlo.convolution(%v1813, %u19eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v1815 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1816 = stablehlo.add %v1814, %v1815 : tensor<32x1024x7x7xf32>
    %v1817 = stablehlo.reshape %v1816 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1818 = stablehlo.reshape %v1817 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1819 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1820 = stablehlo.constant dense<1568.0> : tensor<32x1024x7x7xf32>
    %v1821 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1822 = stablehlo.reduce(%v1818 init: %v1819) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1823 = stablehlo.broadcast_in_dim %v1822, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1824 = stablehlo.divide %v1823, %v1820 : tensor<32x1024x7x7xf32>
    %v1825 = stablehlo.subtract %v1818, %v1824 : tensor<32x1024x7x7xf32>
    %v1826 = stablehlo.multiply %v1825, %v1825 : tensor<32x1024x7x7xf32>
    %v1827 = stablehlo.reduce(%v1826 init: %v1819) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1828 = stablehlo.broadcast_in_dim %v1827, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1829 = stablehlo.divide %v1828, %v1820 : tensor<32x1024x7x7xf32>
    %v1830 = stablehlo.add %v1829, %v1821 : tensor<32x1024x7x7xf32>
    %v1831 = stablehlo.rsqrt %v1830 : tensor<32x1024x7x7xf32>
    %v1832 = stablehlo.multiply %v1825, %v1831 : tensor<32x1024x7x7xf32>
    %v1833 = stablehlo.broadcast_in_dim %u19eg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1834 = stablehlo.broadcast_in_dim %u19ebt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1835 = stablehlo.multiply %v1832, %v1833 : tensor<32x1024x7x7xf32>
    %v1836 = stablehlo.add %v1835, %v1834 : tensor<32x1024x7x7xf32>
    %v1837 = stablehlo.reshape %v1836 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1838 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1839 = stablehlo.maximum %v1837, %v1838 : tensor<32x50176xf32>
    %v1840 = stablehlo.reshape %v1839 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1841 = stablehlo.convolution(%v1840, %u19pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x7x7xf32>
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
    %v1860 = stablehlo.broadcast_in_dim %u19pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1861 = stablehlo.broadcast_in_dim %u19pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1862 = stablehlo.multiply %v1859, %v1860 : tensor<32x256x7x7xf32>
    %v1863 = stablehlo.add %v1862, %v1861 : tensor<32x256x7x7xf32>
    %v1864 = stablehlo.reshape %v1863 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1865 = stablehlo.add %v1864, %v1812 : tensor<32x12544xf32>
    %v1866 = stablehlo.reshape %v1865 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1867 = stablehlo.convolution(%v1866, %u20eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v1868 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1869 = stablehlo.add %v1867, %v1868 : tensor<32x1024x7x7xf32>
    %v1870 = stablehlo.reshape %v1869 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1871 = stablehlo.reshape %v1870 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1872 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1873 = stablehlo.constant dense<1568.0> : tensor<32x1024x7x7xf32>
    %v1874 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1875 = stablehlo.reduce(%v1871 init: %v1872) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1876 = stablehlo.broadcast_in_dim %v1875, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1877 = stablehlo.divide %v1876, %v1873 : tensor<32x1024x7x7xf32>
    %v1878 = stablehlo.subtract %v1871, %v1877 : tensor<32x1024x7x7xf32>
    %v1879 = stablehlo.multiply %v1878, %v1878 : tensor<32x1024x7x7xf32>
    %v1880 = stablehlo.reduce(%v1879 init: %v1872) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1881 = stablehlo.broadcast_in_dim %v1880, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1882 = stablehlo.divide %v1881, %v1873 : tensor<32x1024x7x7xf32>
    %v1883 = stablehlo.add %v1882, %v1874 : tensor<32x1024x7x7xf32>
    %v1884 = stablehlo.rsqrt %v1883 : tensor<32x1024x7x7xf32>
    %v1885 = stablehlo.multiply %v1878, %v1884 : tensor<32x1024x7x7xf32>
    %v1886 = stablehlo.broadcast_in_dim %u20eg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1887 = stablehlo.broadcast_in_dim %u20ebt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1888 = stablehlo.multiply %v1885, %v1886 : tensor<32x1024x7x7xf32>
    %v1889 = stablehlo.add %v1888, %v1887 : tensor<32x1024x7x7xf32>
    %v1890 = stablehlo.reshape %v1889 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1891 = stablehlo.constant dense<0.0> : tensor<32x50176xf32>
    %v1892 = stablehlo.maximum %v1890, %v1891 : tensor<32x50176xf32>
    %v1893 = stablehlo.reshape %v1892 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1894 = stablehlo.convolution(%v1893, %u20pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1895 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1896 = stablehlo.add %v1894, %v1895 : tensor<32x256x7x7xf32>
    %v1897 = stablehlo.reshape %v1896 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1898 = stablehlo.reshape %v1897 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1899 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1900 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1901 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1902 = stablehlo.reduce(%v1898 init: %v1899) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1903 = stablehlo.broadcast_in_dim %v1902, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1904 = stablehlo.divide %v1903, %v1900 : tensor<32x256x7x7xf32>
    %v1905 = stablehlo.subtract %v1898, %v1904 : tensor<32x256x7x7xf32>
    %v1906 = stablehlo.multiply %v1905, %v1905 : tensor<32x256x7x7xf32>
    %v1907 = stablehlo.reduce(%v1906 init: %v1899) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1908 = stablehlo.broadcast_in_dim %v1907, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1909 = stablehlo.divide %v1908, %v1900 : tensor<32x256x7x7xf32>
    %v1910 = stablehlo.add %v1909, %v1901 : tensor<32x256x7x7xf32>
    %v1911 = stablehlo.rsqrt %v1910 : tensor<32x256x7x7xf32>
    %v1912 = stablehlo.multiply %v1905, %v1911 : tensor<32x256x7x7xf32>
    %v1913 = stablehlo.broadcast_in_dim %u20pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1914 = stablehlo.broadcast_in_dim %u20pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1915 = stablehlo.multiply %v1912, %v1913 : tensor<32x256x7x7xf32>
    %v1916 = stablehlo.add %v1915, %v1914 : tensor<32x256x7x7xf32>
    %v1917 = stablehlo.reshape %v1916 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1918 = stablehlo.add %v1917, %v1865 : tensor<32x12544xf32>
    %v1919 = stablehlo.reshape %v1918 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1920 = stablehlo.convolution(%v1919, %u21qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x7x7xf32>, tensor<256x1x5x5xf32>) -> tensor<32x256x7x7xf32>
    %v1921 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1922 = stablehlo.add %v1920, %v1921 : tensor<32x256x7x7xf32>
    %v1923 = stablehlo.reshape %v1922 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1924 = stablehlo.reshape %v1923 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1925 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1926 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1927 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1928 = stablehlo.reduce(%v1924 init: %v1925) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1929 = stablehlo.broadcast_in_dim %v1928, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1930 = stablehlo.divide %v1929, %v1926 : tensor<32x256x7x7xf32>
    %v1931 = stablehlo.subtract %v1924, %v1930 : tensor<32x256x7x7xf32>
    %v1932 = stablehlo.multiply %v1931, %v1931 : tensor<32x256x7x7xf32>
    %v1933 = stablehlo.reduce(%v1932 init: %v1925) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1934 = stablehlo.broadcast_in_dim %v1933, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1935 = stablehlo.divide %v1934, %v1926 : tensor<32x256x7x7xf32>
    %v1936 = stablehlo.add %v1935, %v1927 : tensor<32x256x7x7xf32>
    %v1937 = stablehlo.rsqrt %v1936 : tensor<32x256x7x7xf32>
    %v1938 = stablehlo.multiply %v1931, %v1937 : tensor<32x256x7x7xf32>
    %v1939 = stablehlo.broadcast_in_dim %u21qg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1940 = stablehlo.broadcast_in_dim %u21qbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1941 = stablehlo.multiply %v1938, %v1939 : tensor<32x256x7x7xf32>
    %v1942 = stablehlo.add %v1941, %v1940 : tensor<32x256x7x7xf32>
    %v1943 = stablehlo.reshape %v1942 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1944 = stablehlo.constant dense<0.0> : tensor<32x12544xf32>
    %v1945 = stablehlo.maximum %v1943, %v1944 : tensor<32x12544xf32>
    %v1946 = stablehlo.reshape %v1945 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1947 = stablehlo.convolution(%v1946, %u21eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<512x256x1x1xf32>) -> tensor<32x512x7x7xf32>
    %v1948 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1949 = stablehlo.add %v1947, %v1948 : tensor<32x512x7x7xf32>
    %v1950 = stablehlo.reshape %v1949 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1951 = stablehlo.reshape %v1950 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1952 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1953 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1954 = stablehlo.constant dense<1.0e-5> : tensor<32x512x7x7xf32>
    %v1955 = stablehlo.reduce(%v1951 init: %v1952) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1956 = stablehlo.broadcast_in_dim %v1955, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1957 = stablehlo.divide %v1956, %v1953 : tensor<32x512x7x7xf32>
    %v1958 = stablehlo.subtract %v1951, %v1957 : tensor<32x512x7x7xf32>
    %v1959 = stablehlo.multiply %v1958, %v1958 : tensor<32x512x7x7xf32>
    %v1960 = stablehlo.reduce(%v1959 init: %v1952) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1961 = stablehlo.broadcast_in_dim %v1960, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1962 = stablehlo.divide %v1961, %v1953 : tensor<32x512x7x7xf32>
    %v1963 = stablehlo.add %v1962, %v1954 : tensor<32x512x7x7xf32>
    %v1964 = stablehlo.rsqrt %v1963 : tensor<32x512x7x7xf32>
    %v1965 = stablehlo.multiply %v1958, %v1964 : tensor<32x512x7x7xf32>
    %v1966 = stablehlo.broadcast_in_dim %u21eg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1967 = stablehlo.broadcast_in_dim %u21ebt, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1968 = stablehlo.multiply %v1965, %v1966 : tensor<32x512x7x7xf32>
    %v1969 = stablehlo.add %v1968, %v1967 : tensor<32x512x7x7xf32>
    %v1970 = stablehlo.reshape %v1969 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1971 = stablehlo.constant dense<0.0> : tensor<32x25088xf32>
    %v1972 = stablehlo.maximum %v1970, %v1971 : tensor<32x25088xf32>
    %v1973 = stablehlo.reshape %v1972 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1974 = stablehlo.convolution(%v1973, %u21pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<256x512x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1975 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1976 = stablehlo.add %v1974, %v1975 : tensor<32x256x7x7xf32>
    %v1977 = stablehlo.reshape %v1976 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1978 = stablehlo.reshape %v1977 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1979 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1980 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1981 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1982 = stablehlo.reduce(%v1978 init: %v1979) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1983 = stablehlo.broadcast_in_dim %v1982, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1984 = stablehlo.divide %v1983, %v1980 : tensor<32x256x7x7xf32>
    %v1985 = stablehlo.subtract %v1978, %v1984 : tensor<32x256x7x7xf32>
    %v1986 = stablehlo.multiply %v1985, %v1985 : tensor<32x256x7x7xf32>
    %v1987 = stablehlo.reduce(%v1986 init: %v1979) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1988 = stablehlo.broadcast_in_dim %v1987, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1989 = stablehlo.divide %v1988, %v1980 : tensor<32x256x7x7xf32>
    %v1990 = stablehlo.add %v1989, %v1981 : tensor<32x256x7x7xf32>
    %v1991 = stablehlo.rsqrt %v1990 : tensor<32x256x7x7xf32>
    %v1992 = stablehlo.multiply %v1985, %v1991 : tensor<32x256x7x7xf32>
    %v1993 = stablehlo.broadcast_in_dim %u21pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1994 = stablehlo.broadcast_in_dim %u21pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1995 = stablehlo.multiply %v1992, %v1993 : tensor<32x256x7x7xf32>
    %v1996 = stablehlo.add %v1995, %v1994 : tensor<32x256x7x7xf32>
    %v1997 = stablehlo.reshape %v1996 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1998 = stablehlo.add %v1997, %v1918 : tensor<32x12544xf32>
    %v1999 = stablehlo.reshape %v1998 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v2000 = stablehlo.convolution(%v1999, %h1W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<960x256x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v2001 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2002 = stablehlo.add %v2000, %v2001 : tensor<32x960x7x7xf32>
    %v2003 = stablehlo.reshape %v2002 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v2004 = stablehlo.reshape %v2003 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2005 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2006 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v2007 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v2008 = stablehlo.reduce(%v2004 init: %v2005) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2009 = stablehlo.broadcast_in_dim %v2008, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2010 = stablehlo.divide %v2009, %v2006 : tensor<32x960x7x7xf32>
    %v2011 = stablehlo.subtract %v2004, %v2010 : tensor<32x960x7x7xf32>
    %v2012 = stablehlo.multiply %v2011, %v2011 : tensor<32x960x7x7xf32>
    %v2013 = stablehlo.reduce(%v2012 init: %v2005) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2014 = stablehlo.broadcast_in_dim %v2013, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2015 = stablehlo.divide %v2014, %v2006 : tensor<32x960x7x7xf32>
    %v2016 = stablehlo.add %v2015, %v2007 : tensor<32x960x7x7xf32>
    %v2017 = stablehlo.rsqrt %v2016 : tensor<32x960x7x7xf32>
    %v2018 = stablehlo.multiply %v2011, %v2017 : tensor<32x960x7x7xf32>
    %v2019 = stablehlo.broadcast_in_dim %h1g, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2020 = stablehlo.broadcast_in_dim %h1bt, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2021 = stablehlo.multiply %v2018, %v2019 : tensor<32x960x7x7xf32>
    %v2022 = stablehlo.add %v2021, %v2020 : tensor<32x960x7x7xf32>
    %v2023 = stablehlo.reshape %v2022 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v2024 = stablehlo.constant dense<0.0> : tensor<32x47040xf32>
    %v2025 = stablehlo.maximum %v2023, %v2024 : tensor<32x47040xf32>
    %v2026 = stablehlo.reshape %v2025 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2027 = stablehlo.convolution(%v2026, %hW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<1280x960x1x1xf32>) -> tensor<32x1280x7x7xf32>
    %v2028 = stablehlo.broadcast_in_dim %zb1280, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v2029 = stablehlo.add %v2027, %v2028 : tensor<32x1280x7x7xf32>
    %v2030 = stablehlo.reshape %v2029 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v2031 = stablehlo.reshape %v2030 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v2032 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2033 = stablehlo.constant dense<1568.0> : tensor<32x1280x7x7xf32>
    %v2034 = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %v2035 = stablehlo.reduce(%v2031 init: %v2032) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v2036 = stablehlo.broadcast_in_dim %v2035, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v2037 = stablehlo.divide %v2036, %v2033 : tensor<32x1280x7x7xf32>
    %v2038 = stablehlo.subtract %v2031, %v2037 : tensor<32x1280x7x7xf32>
    %v2039 = stablehlo.multiply %v2038, %v2038 : tensor<32x1280x7x7xf32>
    %v2040 = stablehlo.reduce(%v2039 init: %v2032) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v2041 = stablehlo.broadcast_in_dim %v2040, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v2042 = stablehlo.divide %v2041, %v2033 : tensor<32x1280x7x7xf32>
    %v2043 = stablehlo.add %v2042, %v2034 : tensor<32x1280x7x7xf32>
    %v2044 = stablehlo.rsqrt %v2043 : tensor<32x1280x7x7xf32>
    %v2045 = stablehlo.multiply %v2038, %v2044 : tensor<32x1280x7x7xf32>
    %v2046 = stablehlo.broadcast_in_dim %hg, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v2047 = stablehlo.broadcast_in_dim %hbt, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v2048 = stablehlo.multiply %v2045, %v2046 : tensor<32x1280x7x7xf32>
    %v2049 = stablehlo.add %v2048, %v2047 : tensor<32x1280x7x7xf32>
    %v2050 = stablehlo.reshape %v2049 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v2051 = stablehlo.constant dense<0.0> : tensor<32x62720xf32>
    %v2052 = stablehlo.maximum %v2050, %v2051 : tensor<32x62720xf32>
    %v2053 = stablehlo.reshape %v2052 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v2054 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2055 = stablehlo.reduce(%v2053 init: %v2054) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v2056 = stablehlo.constant dense<49.0> : tensor<32x1280xf32>
    %v2057 = stablehlo.divide %v2055, %v2056 : tensor<32x1280xf32>
    %v2058 = stablehlo.dot_general %v2057, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1280xf32>, tensor<1280x10xf32>) -> tensor<32x10xf32>
    %v2059 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v2060 = stablehlo.add %v2058, %v2059 : tensor<32x10xf32>
    return %v2060 : tensor<32x10xf32>
  }
}
