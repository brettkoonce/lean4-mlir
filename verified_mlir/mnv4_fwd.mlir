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
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
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
    %v54 = stablehlo.reshape %v53 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v55 = stablehlo.constant dense<0.0> : tensor<32x128x56x56xf32>
    %v56 = stablehlo.maximum %v54, %v55 : tensor<32x128x56x56xf32>
    %v57 = stablehlo.reshape %v56 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
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
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 48 : i64} : (tensor<32x48x56x56xf32>, tensor<48x1x3x3xf32>) -> tensor<32x48x56x56xf32>
    %v85 = stablehlo.broadcast_in_dim %zb48, dims = [1] : (tensor<48xf32>) -> tensor<32x48x56x56xf32>
    %v86 = stablehlo.add %v84, %v85 : tensor<32x48x56x56xf32>
    %v87 = stablehlo.reshape %v86 : (tensor<32x48x56x56xf32>) -> tensor<32x150528xf32>
    %v88 = stablehlo.reshape %v87 : (tensor<32x150528xf32>) -> tensor<32x48x56x56xf32>
    %v89 = stablehlo.constant dense<0.0> : tensor<f32>
    %v90 = stablehlo.constant dense<100352.0> : tensor<32x48x56x56xf32>
    %v91 = stablehlo.constant dense<1.0e-5> : tensor<32x48x56x56xf32>
    %v92 = stablehlo.reduce(%v88 init: %v89) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x48x56x56xf32>, tensor<f32>) -> tensor<48xf32>
    %v93 = stablehlo.broadcast_in_dim %v92, dims = [1] : (tensor<48xf32>) -> tensor<32x48x56x56xf32>
    %v94 = stablehlo.divide %v93, %v90 : tensor<32x48x56x56xf32>
    %v95 = stablehlo.subtract %v88, %v94 : tensor<32x48x56x56xf32>
    %v96 = stablehlo.multiply %v95, %v95 : tensor<32x48x56x56xf32>
    %v97 = stablehlo.reduce(%v96 init: %v89) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x48x56x56xf32>, tensor<f32>) -> tensor<48xf32>
    %v98 = stablehlo.broadcast_in_dim %v97, dims = [1] : (tensor<48xf32>) -> tensor<32x48x56x56xf32>
    %v99 = stablehlo.divide %v98, %v90 : tensor<32x48x56x56xf32>
    %v100 = stablehlo.add %v99, %v91 : tensor<32x48x56x56xf32>
    %v101 = stablehlo.rsqrt %v100 : tensor<32x48x56x56xf32>
    %v102 = stablehlo.multiply %v95, %v101 : tensor<32x48x56x56xf32>
    %v103 = stablehlo.broadcast_in_dim %u1qg, dims = [1] : (tensor<48xf32>) -> tensor<32x48x56x56xf32>
    %v104 = stablehlo.broadcast_in_dim %u1qbt, dims = [1] : (tensor<48xf32>) -> tensor<32x48x56x56xf32>
    %v105 = stablehlo.multiply %v102, %v103 : tensor<32x48x56x56xf32>
    %v106 = stablehlo.add %v105, %v104 : tensor<32x48x56x56xf32>
    %v107 = stablehlo.reshape %v106 : (tensor<32x48x56x56xf32>) -> tensor<32x150528xf32>
    %v108 = stablehlo.reshape %v107 : (tensor<32x150528xf32>) -> tensor<32x48x56x56xf32>
    %v109 = stablehlo.convolution(%v108, %u1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x48x56x56xf32>, tensor<192x48x1x1xf32>) -> tensor<32x192x56x56xf32>
    %v110 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x56x56xf32>
    %v111 = stablehlo.add %v109, %v110 : tensor<32x192x56x56xf32>
    %v112 = stablehlo.reshape %v111 : (tensor<32x192x56x56xf32>) -> tensor<32x602112xf32>
    %v113 = stablehlo.reshape %v112 : (tensor<32x602112xf32>) -> tensor<32x192x56x56xf32>
    %v114 = stablehlo.constant dense<0.0> : tensor<f32>
    %v115 = stablehlo.constant dense<100352.0> : tensor<32x192x56x56xf32>
    %v116 = stablehlo.constant dense<1.0e-5> : tensor<32x192x56x56xf32>
    %v117 = stablehlo.reduce(%v113 init: %v114) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x56x56xf32>, tensor<f32>) -> tensor<192xf32>
    %v118 = stablehlo.broadcast_in_dim %v117, dims = [1] : (tensor<192xf32>) -> tensor<32x192x56x56xf32>
    %v119 = stablehlo.divide %v118, %v115 : tensor<32x192x56x56xf32>
    %v120 = stablehlo.subtract %v113, %v119 : tensor<32x192x56x56xf32>
    %v121 = stablehlo.multiply %v120, %v120 : tensor<32x192x56x56xf32>
    %v122 = stablehlo.reduce(%v121 init: %v114) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x56x56xf32>, tensor<f32>) -> tensor<192xf32>
    %v123 = stablehlo.broadcast_in_dim %v122, dims = [1] : (tensor<192xf32>) -> tensor<32x192x56x56xf32>
    %v124 = stablehlo.divide %v123, %v115 : tensor<32x192x56x56xf32>
    %v125 = stablehlo.add %v124, %v116 : tensor<32x192x56x56xf32>
    %v126 = stablehlo.rsqrt %v125 : tensor<32x192x56x56xf32>
    %v127 = stablehlo.multiply %v120, %v126 : tensor<32x192x56x56xf32>
    %v128 = stablehlo.broadcast_in_dim %u1eg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x56x56xf32>
    %v129 = stablehlo.broadcast_in_dim %u1ebt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x56x56xf32>
    %v130 = stablehlo.multiply %v127, %v128 : tensor<32x192x56x56xf32>
    %v131 = stablehlo.add %v130, %v129 : tensor<32x192x56x56xf32>
    %v132 = stablehlo.reshape %v131 : (tensor<32x192x56x56xf32>) -> tensor<32x602112xf32>
    %v133 = stablehlo.reshape %v132 : (tensor<32x602112xf32>) -> tensor<32x192x56x56xf32>
    %v134 = stablehlo.constant dense<0.0> : tensor<32x192x56x56xf32>
    %v135 = stablehlo.maximum %v133, %v134 : tensor<32x192x56x56xf32>
    %v136 = stablehlo.reshape %v135 : (tensor<32x192x56x56xf32>) -> tensor<32x602112xf32>
    %v137 = stablehlo.reshape %v136 : (tensor<32x602112xf32>) -> tensor<32x192x56x56xf32>
    %v138 = stablehlo.convolution(%v137, %u1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x56x56xf32>, tensor<192x1x5x5xf32>) -> tensor<32x192x28x28xf32>
    %v139 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v140 = stablehlo.add %v138, %v139 : tensor<32x192x28x28xf32>
    %v141 = stablehlo.reshape %v140 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v142 = stablehlo.reshape %v141 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v143 = stablehlo.constant dense<0.0> : tensor<f32>
    %v144 = stablehlo.constant dense<25088.0> : tensor<32x192x28x28xf32>
    %v145 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v146 = stablehlo.reduce(%v142 init: %v143) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v147 = stablehlo.broadcast_in_dim %v146, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v148 = stablehlo.divide %v147, %v144 : tensor<32x192x28x28xf32>
    %v149 = stablehlo.subtract %v142, %v148 : tensor<32x192x28x28xf32>
    %v150 = stablehlo.multiply %v149, %v149 : tensor<32x192x28x28xf32>
    %v151 = stablehlo.reduce(%v150 init: %v143) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v152 = stablehlo.broadcast_in_dim %v151, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v153 = stablehlo.divide %v152, %v144 : tensor<32x192x28x28xf32>
    %v154 = stablehlo.add %v153, %v145 : tensor<32x192x28x28xf32>
    %v155 = stablehlo.rsqrt %v154 : tensor<32x192x28x28xf32>
    %v156 = stablehlo.multiply %v149, %v155 : tensor<32x192x28x28xf32>
    %v157 = stablehlo.broadcast_in_dim %u1dg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v158 = stablehlo.broadcast_in_dim %u1dbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v159 = stablehlo.multiply %v156, %v157 : tensor<32x192x28x28xf32>
    %v160 = stablehlo.add %v159, %v158 : tensor<32x192x28x28xf32>
    %v161 = stablehlo.reshape %v160 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v162 = stablehlo.reshape %v161 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v163 = stablehlo.constant dense<0.0> : tensor<32x192x28x28xf32>
    %v164 = stablehlo.maximum %v162, %v163 : tensor<32x192x28x28xf32>
    %v165 = stablehlo.reshape %v164 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v166 = stablehlo.reshape %v165 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v167 = stablehlo.convolution(%v166, %u1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<80x192x1x1xf32>) -> tensor<32x80x28x28xf32>
    %v168 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v169 = stablehlo.add %v167, %v168 : tensor<32x80x28x28xf32>
    %v170 = stablehlo.reshape %v169 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v171 = stablehlo.reshape %v170 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v172 = stablehlo.constant dense<0.0> : tensor<f32>
    %v173 = stablehlo.constant dense<25088.0> : tensor<32x80x28x28xf32>
    %v174 = stablehlo.constant dense<1.0e-5> : tensor<32x80x28x28xf32>
    %v175 = stablehlo.reduce(%v171 init: %v172) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x28x28xf32>, tensor<f32>) -> tensor<80xf32>
    %v176 = stablehlo.broadcast_in_dim %v175, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v177 = stablehlo.divide %v176, %v173 : tensor<32x80x28x28xf32>
    %v178 = stablehlo.subtract %v171, %v177 : tensor<32x80x28x28xf32>
    %v179 = stablehlo.multiply %v178, %v178 : tensor<32x80x28x28xf32>
    %v180 = stablehlo.reduce(%v179 init: %v172) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x28x28xf32>, tensor<f32>) -> tensor<80xf32>
    %v181 = stablehlo.broadcast_in_dim %v180, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v182 = stablehlo.divide %v181, %v173 : tensor<32x80x28x28xf32>
    %v183 = stablehlo.add %v182, %v174 : tensor<32x80x28x28xf32>
    %v184 = stablehlo.rsqrt %v183 : tensor<32x80x28x28xf32>
    %v185 = stablehlo.multiply %v178, %v184 : tensor<32x80x28x28xf32>
    %v186 = stablehlo.broadcast_in_dim %u1pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v187 = stablehlo.broadcast_in_dim %u1pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v188 = stablehlo.multiply %v185, %v186 : tensor<32x80x28x28xf32>
    %v189 = stablehlo.add %v188, %v187 : tensor<32x80x28x28xf32>
    %v190 = stablehlo.reshape %v189 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v191 = stablehlo.reshape %v190 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v192 = stablehlo.convolution(%v191, %u2qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 80 : i64} : (tensor<32x80x28x28xf32>, tensor<80x1x3x3xf32>) -> tensor<32x80x28x28xf32>
    %v193 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v194 = stablehlo.add %v192, %v193 : tensor<32x80x28x28xf32>
    %v195 = stablehlo.reshape %v194 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v196 = stablehlo.reshape %v195 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v197 = stablehlo.constant dense<0.0> : tensor<f32>
    %v198 = stablehlo.constant dense<25088.0> : tensor<32x80x28x28xf32>
    %v199 = stablehlo.constant dense<1.0e-5> : tensor<32x80x28x28xf32>
    %v200 = stablehlo.reduce(%v196 init: %v197) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x28x28xf32>, tensor<f32>) -> tensor<80xf32>
    %v201 = stablehlo.broadcast_in_dim %v200, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v202 = stablehlo.divide %v201, %v198 : tensor<32x80x28x28xf32>
    %v203 = stablehlo.subtract %v196, %v202 : tensor<32x80x28x28xf32>
    %v204 = stablehlo.multiply %v203, %v203 : tensor<32x80x28x28xf32>
    %v205 = stablehlo.reduce(%v204 init: %v197) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x28x28xf32>, tensor<f32>) -> tensor<80xf32>
    %v206 = stablehlo.broadcast_in_dim %v205, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v207 = stablehlo.divide %v206, %v198 : tensor<32x80x28x28xf32>
    %v208 = stablehlo.add %v207, %v199 : tensor<32x80x28x28xf32>
    %v209 = stablehlo.rsqrt %v208 : tensor<32x80x28x28xf32>
    %v210 = stablehlo.multiply %v203, %v209 : tensor<32x80x28x28xf32>
    %v211 = stablehlo.broadcast_in_dim %u2qg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v212 = stablehlo.broadcast_in_dim %u2qbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v213 = stablehlo.multiply %v210, %v211 : tensor<32x80x28x28xf32>
    %v214 = stablehlo.add %v213, %v212 : tensor<32x80x28x28xf32>
    %v215 = stablehlo.reshape %v214 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v216 = stablehlo.reshape %v215 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v217 = stablehlo.convolution(%v216, %u2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x28x28xf32>, tensor<160x80x1x1xf32>) -> tensor<32x160x28x28xf32>
    %v218 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v219 = stablehlo.add %v217, %v218 : tensor<32x160x28x28xf32>
    %v220 = stablehlo.reshape %v219 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v221 = stablehlo.reshape %v220 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v222 = stablehlo.constant dense<0.0> : tensor<f32>
    %v223 = stablehlo.constant dense<25088.0> : tensor<32x160x28x28xf32>
    %v224 = stablehlo.constant dense<1.0e-5> : tensor<32x160x28x28xf32>
    %v225 = stablehlo.reduce(%v221 init: %v222) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x28x28xf32>, tensor<f32>) -> tensor<160xf32>
    %v226 = stablehlo.broadcast_in_dim %v225, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v227 = stablehlo.divide %v226, %v223 : tensor<32x160x28x28xf32>
    %v228 = stablehlo.subtract %v221, %v227 : tensor<32x160x28x28xf32>
    %v229 = stablehlo.multiply %v228, %v228 : tensor<32x160x28x28xf32>
    %v230 = stablehlo.reduce(%v229 init: %v222) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x28x28xf32>, tensor<f32>) -> tensor<160xf32>
    %v231 = stablehlo.broadcast_in_dim %v230, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v232 = stablehlo.divide %v231, %v223 : tensor<32x160x28x28xf32>
    %v233 = stablehlo.add %v232, %v224 : tensor<32x160x28x28xf32>
    %v234 = stablehlo.rsqrt %v233 : tensor<32x160x28x28xf32>
    %v235 = stablehlo.multiply %v228, %v234 : tensor<32x160x28x28xf32>
    %v236 = stablehlo.broadcast_in_dim %u2eg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v237 = stablehlo.broadcast_in_dim %u2ebt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v238 = stablehlo.multiply %v235, %v236 : tensor<32x160x28x28xf32>
    %v239 = stablehlo.add %v238, %v237 : tensor<32x160x28x28xf32>
    %v240 = stablehlo.reshape %v239 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v241 = stablehlo.reshape %v240 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v242 = stablehlo.constant dense<0.0> : tensor<32x160x28x28xf32>
    %v243 = stablehlo.maximum %v241, %v242 : tensor<32x160x28x28xf32>
    %v244 = stablehlo.reshape %v243 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v245 = stablehlo.reshape %v244 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v246 = stablehlo.convolution(%v245, %u2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x28x28xf32>, tensor<160x1x3x3xf32>) -> tensor<32x160x28x28xf32>
    %v247 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v248 = stablehlo.add %v246, %v247 : tensor<32x160x28x28xf32>
    %v249 = stablehlo.reshape %v248 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v250 = stablehlo.reshape %v249 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v251 = stablehlo.constant dense<0.0> : tensor<f32>
    %v252 = stablehlo.constant dense<25088.0> : tensor<32x160x28x28xf32>
    %v253 = stablehlo.constant dense<1.0e-5> : tensor<32x160x28x28xf32>
    %v254 = stablehlo.reduce(%v250 init: %v251) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x28x28xf32>, tensor<f32>) -> tensor<160xf32>
    %v255 = stablehlo.broadcast_in_dim %v254, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v256 = stablehlo.divide %v255, %v252 : tensor<32x160x28x28xf32>
    %v257 = stablehlo.subtract %v250, %v256 : tensor<32x160x28x28xf32>
    %v258 = stablehlo.multiply %v257, %v257 : tensor<32x160x28x28xf32>
    %v259 = stablehlo.reduce(%v258 init: %v251) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x28x28xf32>, tensor<f32>) -> tensor<160xf32>
    %v260 = stablehlo.broadcast_in_dim %v259, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v261 = stablehlo.divide %v260, %v252 : tensor<32x160x28x28xf32>
    %v262 = stablehlo.add %v261, %v253 : tensor<32x160x28x28xf32>
    %v263 = stablehlo.rsqrt %v262 : tensor<32x160x28x28xf32>
    %v264 = stablehlo.multiply %v257, %v263 : tensor<32x160x28x28xf32>
    %v265 = stablehlo.broadcast_in_dim %u2dg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v266 = stablehlo.broadcast_in_dim %u2dbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v267 = stablehlo.multiply %v264, %v265 : tensor<32x160x28x28xf32>
    %v268 = stablehlo.add %v267, %v266 : tensor<32x160x28x28xf32>
    %v269 = stablehlo.reshape %v268 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v270 = stablehlo.reshape %v269 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v271 = stablehlo.constant dense<0.0> : tensor<32x160x28x28xf32>
    %v272 = stablehlo.maximum %v270, %v271 : tensor<32x160x28x28xf32>
    %v273 = stablehlo.reshape %v272 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v274 = stablehlo.reshape %v273 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v275 = stablehlo.convolution(%v274, %u2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x28x28xf32>, tensor<80x160x1x1xf32>) -> tensor<32x80x28x28xf32>
    %v276 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v277 = stablehlo.add %v275, %v276 : tensor<32x80x28x28xf32>
    %v278 = stablehlo.reshape %v277 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v279 = stablehlo.reshape %v278 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v280 = stablehlo.constant dense<0.0> : tensor<f32>
    %v281 = stablehlo.constant dense<25088.0> : tensor<32x80x28x28xf32>
    %v282 = stablehlo.constant dense<1.0e-5> : tensor<32x80x28x28xf32>
    %v283 = stablehlo.reduce(%v279 init: %v280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x28x28xf32>, tensor<f32>) -> tensor<80xf32>
    %v284 = stablehlo.broadcast_in_dim %v283, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v285 = stablehlo.divide %v284, %v281 : tensor<32x80x28x28xf32>
    %v286 = stablehlo.subtract %v279, %v285 : tensor<32x80x28x28xf32>
    %v287 = stablehlo.multiply %v286, %v286 : tensor<32x80x28x28xf32>
    %v288 = stablehlo.reduce(%v287 init: %v280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x28x28xf32>, tensor<f32>) -> tensor<80xf32>
    %v289 = stablehlo.broadcast_in_dim %v288, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v290 = stablehlo.divide %v289, %v281 : tensor<32x80x28x28xf32>
    %v291 = stablehlo.add %v290, %v282 : tensor<32x80x28x28xf32>
    %v292 = stablehlo.rsqrt %v291 : tensor<32x80x28x28xf32>
    %v293 = stablehlo.multiply %v286, %v292 : tensor<32x80x28x28xf32>
    %v294 = stablehlo.broadcast_in_dim %u2pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v295 = stablehlo.broadcast_in_dim %u2pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v296 = stablehlo.multiply %v293, %v294 : tensor<32x80x28x28xf32>
    %v297 = stablehlo.add %v296, %v295 : tensor<32x80x28x28xf32>
    %v298 = stablehlo.reshape %v297 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v299 = stablehlo.reshape %v298 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v300 = stablehlo.reshape %v190 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v301 = stablehlo.add %v299, %v300 : tensor<32x80x28x28xf32>
    %v302 = stablehlo.reshape %v301 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v303 = stablehlo.reshape %v302 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v304 = stablehlo.convolution(%v303, %u3qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 80 : i64} : (tensor<32x80x28x28xf32>, tensor<80x1x3x3xf32>) -> tensor<32x80x28x28xf32>
    %v305 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v306 = stablehlo.add %v304, %v305 : tensor<32x80x28x28xf32>
    %v307 = stablehlo.reshape %v306 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v308 = stablehlo.reshape %v307 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v309 = stablehlo.constant dense<0.0> : tensor<f32>
    %v310 = stablehlo.constant dense<25088.0> : tensor<32x80x28x28xf32>
    %v311 = stablehlo.constant dense<1.0e-5> : tensor<32x80x28x28xf32>
    %v312 = stablehlo.reduce(%v308 init: %v309) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x28x28xf32>, tensor<f32>) -> tensor<80xf32>
    %v313 = stablehlo.broadcast_in_dim %v312, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v314 = stablehlo.divide %v313, %v310 : tensor<32x80x28x28xf32>
    %v315 = stablehlo.subtract %v308, %v314 : tensor<32x80x28x28xf32>
    %v316 = stablehlo.multiply %v315, %v315 : tensor<32x80x28x28xf32>
    %v317 = stablehlo.reduce(%v316 init: %v309) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x28x28xf32>, tensor<f32>) -> tensor<80xf32>
    %v318 = stablehlo.broadcast_in_dim %v317, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v319 = stablehlo.divide %v318, %v310 : tensor<32x80x28x28xf32>
    %v320 = stablehlo.add %v319, %v311 : tensor<32x80x28x28xf32>
    %v321 = stablehlo.rsqrt %v320 : tensor<32x80x28x28xf32>
    %v322 = stablehlo.multiply %v315, %v321 : tensor<32x80x28x28xf32>
    %v323 = stablehlo.broadcast_in_dim %u3qg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v324 = stablehlo.broadcast_in_dim %u3qbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v325 = stablehlo.multiply %v322, %v323 : tensor<32x80x28x28xf32>
    %v326 = stablehlo.add %v325, %v324 : tensor<32x80x28x28xf32>
    %v327 = stablehlo.reshape %v326 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v328 = stablehlo.reshape %v327 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v329 = stablehlo.convolution(%v328, %u3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x28x28xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x28x28xf32>
    %v330 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x28x28xf32>
    %v331 = stablehlo.add %v329, %v330 : tensor<32x480x28x28xf32>
    %v332 = stablehlo.reshape %v331 : (tensor<32x480x28x28xf32>) -> tensor<32x376320xf32>
    %v333 = stablehlo.reshape %v332 : (tensor<32x376320xf32>) -> tensor<32x480x28x28xf32>
    %v334 = stablehlo.constant dense<0.0> : tensor<f32>
    %v335 = stablehlo.constant dense<25088.0> : tensor<32x480x28x28xf32>
    %v336 = stablehlo.constant dense<1.0e-5> : tensor<32x480x28x28xf32>
    %v337 = stablehlo.reduce(%v333 init: %v334) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x28x28xf32>, tensor<f32>) -> tensor<480xf32>
    %v338 = stablehlo.broadcast_in_dim %v337, dims = [1] : (tensor<480xf32>) -> tensor<32x480x28x28xf32>
    %v339 = stablehlo.divide %v338, %v335 : tensor<32x480x28x28xf32>
    %v340 = stablehlo.subtract %v333, %v339 : tensor<32x480x28x28xf32>
    %v341 = stablehlo.multiply %v340, %v340 : tensor<32x480x28x28xf32>
    %v342 = stablehlo.reduce(%v341 init: %v334) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x28x28xf32>, tensor<f32>) -> tensor<480xf32>
    %v343 = stablehlo.broadcast_in_dim %v342, dims = [1] : (tensor<480xf32>) -> tensor<32x480x28x28xf32>
    %v344 = stablehlo.divide %v343, %v335 : tensor<32x480x28x28xf32>
    %v345 = stablehlo.add %v344, %v336 : tensor<32x480x28x28xf32>
    %v346 = stablehlo.rsqrt %v345 : tensor<32x480x28x28xf32>
    %v347 = stablehlo.multiply %v340, %v346 : tensor<32x480x28x28xf32>
    %v348 = stablehlo.broadcast_in_dim %u3eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x28x28xf32>
    %v349 = stablehlo.broadcast_in_dim %u3ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x28x28xf32>
    %v350 = stablehlo.multiply %v347, %v348 : tensor<32x480x28x28xf32>
    %v351 = stablehlo.add %v350, %v349 : tensor<32x480x28x28xf32>
    %v352 = stablehlo.reshape %v351 : (tensor<32x480x28x28xf32>) -> tensor<32x376320xf32>
    %v353 = stablehlo.reshape %v352 : (tensor<32x376320xf32>) -> tensor<32x480x28x28xf32>
    %v354 = stablehlo.constant dense<0.0> : tensor<32x480x28x28xf32>
    %v355 = stablehlo.maximum %v353, %v354 : tensor<32x480x28x28xf32>
    %v356 = stablehlo.reshape %v355 : (tensor<32x480x28x28xf32>) -> tensor<32x376320xf32>
    %v357 = stablehlo.reshape %v356 : (tensor<32x376320xf32>) -> tensor<32x480x28x28xf32>
    %v358 = stablehlo.convolution(%v357, %u3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x28x28xf32>, tensor<480x1x5x5xf32>) -> tensor<32x480x14x14xf32>
    %v359 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v360 = stablehlo.add %v358, %v359 : tensor<32x480x14x14xf32>
    %v361 = stablehlo.reshape %v360 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v362 = stablehlo.reshape %v361 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v363 = stablehlo.constant dense<0.0> : tensor<f32>
    %v364 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v365 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v366 = stablehlo.reduce(%v362 init: %v363) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v367 = stablehlo.broadcast_in_dim %v366, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v368 = stablehlo.divide %v367, %v364 : tensor<32x480x14x14xf32>
    %v369 = stablehlo.subtract %v362, %v368 : tensor<32x480x14x14xf32>
    %v370 = stablehlo.multiply %v369, %v369 : tensor<32x480x14x14xf32>
    %v371 = stablehlo.reduce(%v370 init: %v363) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v372 = stablehlo.broadcast_in_dim %v371, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v373 = stablehlo.divide %v372, %v364 : tensor<32x480x14x14xf32>
    %v374 = stablehlo.add %v373, %v365 : tensor<32x480x14x14xf32>
    %v375 = stablehlo.rsqrt %v374 : tensor<32x480x14x14xf32>
    %v376 = stablehlo.multiply %v369, %v375 : tensor<32x480x14x14xf32>
    %v377 = stablehlo.broadcast_in_dim %u3dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v378 = stablehlo.broadcast_in_dim %u3dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v379 = stablehlo.multiply %v376, %v377 : tensor<32x480x14x14xf32>
    %v380 = stablehlo.add %v379, %v378 : tensor<32x480x14x14xf32>
    %v381 = stablehlo.reshape %v380 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v382 = stablehlo.reshape %v381 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v383 = stablehlo.constant dense<0.0> : tensor<32x480x14x14xf32>
    %v384 = stablehlo.maximum %v382, %v383 : tensor<32x480x14x14xf32>
    %v385 = stablehlo.reshape %v384 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v386 = stablehlo.reshape %v385 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v387 = stablehlo.convolution(%v386, %u3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<160x480x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v388 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v389 = stablehlo.add %v387, %v388 : tensor<32x160x14x14xf32>
    %v390 = stablehlo.reshape %v389 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v391 = stablehlo.reshape %v390 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v392 = stablehlo.constant dense<0.0> : tensor<f32>
    %v393 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v394 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v395 = stablehlo.reduce(%v391 init: %v392) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v396 = stablehlo.broadcast_in_dim %v395, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v397 = stablehlo.divide %v396, %v393 : tensor<32x160x14x14xf32>
    %v398 = stablehlo.subtract %v391, %v397 : tensor<32x160x14x14xf32>
    %v399 = stablehlo.multiply %v398, %v398 : tensor<32x160x14x14xf32>
    %v400 = stablehlo.reduce(%v399 init: %v392) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v401 = stablehlo.broadcast_in_dim %v400, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v402 = stablehlo.divide %v401, %v393 : tensor<32x160x14x14xf32>
    %v403 = stablehlo.add %v402, %v394 : tensor<32x160x14x14xf32>
    %v404 = stablehlo.rsqrt %v403 : tensor<32x160x14x14xf32>
    %v405 = stablehlo.multiply %v398, %v404 : tensor<32x160x14x14xf32>
    %v406 = stablehlo.broadcast_in_dim %u3pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v407 = stablehlo.broadcast_in_dim %u3pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v408 = stablehlo.multiply %v405, %v406 : tensor<32x160x14x14xf32>
    %v409 = stablehlo.add %v408, %v407 : tensor<32x160x14x14xf32>
    %v410 = stablehlo.reshape %v409 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v411 = stablehlo.reshape %v410 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v412 = stablehlo.convolution(%v411, %u4qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<32x160x14x14xf32>
    %v413 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v414 = stablehlo.add %v412, %v413 : tensor<32x160x14x14xf32>
    %v415 = stablehlo.reshape %v414 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v416 = stablehlo.reshape %v415 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v417 = stablehlo.constant dense<0.0> : tensor<f32>
    %v418 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v419 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v420 = stablehlo.reduce(%v416 init: %v417) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v421 = stablehlo.broadcast_in_dim %v420, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v422 = stablehlo.divide %v421, %v418 : tensor<32x160x14x14xf32>
    %v423 = stablehlo.subtract %v416, %v422 : tensor<32x160x14x14xf32>
    %v424 = stablehlo.multiply %v423, %v423 : tensor<32x160x14x14xf32>
    %v425 = stablehlo.reduce(%v424 init: %v417) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v426 = stablehlo.broadcast_in_dim %v425, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v427 = stablehlo.divide %v426, %v418 : tensor<32x160x14x14xf32>
    %v428 = stablehlo.add %v427, %v419 : tensor<32x160x14x14xf32>
    %v429 = stablehlo.rsqrt %v428 : tensor<32x160x14x14xf32>
    %v430 = stablehlo.multiply %v423, %v429 : tensor<32x160x14x14xf32>
    %v431 = stablehlo.broadcast_in_dim %u4qg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v432 = stablehlo.broadcast_in_dim %u4qbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v433 = stablehlo.multiply %v430, %v431 : tensor<32x160x14x14xf32>
    %v434 = stablehlo.add %v433, %v432 : tensor<32x160x14x14xf32>
    %v435 = stablehlo.reshape %v434 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v436 = stablehlo.reshape %v435 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v437 = stablehlo.convolution(%v436, %u4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<32x640x14x14xf32>
    %v438 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v439 = stablehlo.add %v437, %v438 : tensor<32x640x14x14xf32>
    %v440 = stablehlo.reshape %v439 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v441 = stablehlo.reshape %v440 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v442 = stablehlo.constant dense<0.0> : tensor<f32>
    %v443 = stablehlo.constant dense<6272.0> : tensor<32x640x14x14xf32>
    %v444 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v445 = stablehlo.reduce(%v441 init: %v442) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v446 = stablehlo.broadcast_in_dim %v445, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v447 = stablehlo.divide %v446, %v443 : tensor<32x640x14x14xf32>
    %v448 = stablehlo.subtract %v441, %v447 : tensor<32x640x14x14xf32>
    %v449 = stablehlo.multiply %v448, %v448 : tensor<32x640x14x14xf32>
    %v450 = stablehlo.reduce(%v449 init: %v442) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v451 = stablehlo.broadcast_in_dim %v450, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v452 = stablehlo.divide %v451, %v443 : tensor<32x640x14x14xf32>
    %v453 = stablehlo.add %v452, %v444 : tensor<32x640x14x14xf32>
    %v454 = stablehlo.rsqrt %v453 : tensor<32x640x14x14xf32>
    %v455 = stablehlo.multiply %v448, %v454 : tensor<32x640x14x14xf32>
    %v456 = stablehlo.broadcast_in_dim %u4eg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v457 = stablehlo.broadcast_in_dim %u4ebt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v458 = stablehlo.multiply %v455, %v456 : tensor<32x640x14x14xf32>
    %v459 = stablehlo.add %v458, %v457 : tensor<32x640x14x14xf32>
    %v460 = stablehlo.reshape %v459 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v461 = stablehlo.reshape %v460 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v462 = stablehlo.constant dense<0.0> : tensor<32x640x14x14xf32>
    %v463 = stablehlo.maximum %v461, %v462 : tensor<32x640x14x14xf32>
    %v464 = stablehlo.reshape %v463 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v465 = stablehlo.reshape %v464 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v466 = stablehlo.convolution(%v465, %u4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 640 : i64} : (tensor<32x640x14x14xf32>, tensor<640x1x3x3xf32>) -> tensor<32x640x14x14xf32>
    %v467 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v468 = stablehlo.add %v466, %v467 : tensor<32x640x14x14xf32>
    %v469 = stablehlo.reshape %v468 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v470 = stablehlo.reshape %v469 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v471 = stablehlo.constant dense<0.0> : tensor<f32>
    %v472 = stablehlo.constant dense<6272.0> : tensor<32x640x14x14xf32>
    %v473 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v474 = stablehlo.reduce(%v470 init: %v471) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v475 = stablehlo.broadcast_in_dim %v474, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v476 = stablehlo.divide %v475, %v472 : tensor<32x640x14x14xf32>
    %v477 = stablehlo.subtract %v470, %v476 : tensor<32x640x14x14xf32>
    %v478 = stablehlo.multiply %v477, %v477 : tensor<32x640x14x14xf32>
    %v479 = stablehlo.reduce(%v478 init: %v471) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v480 = stablehlo.broadcast_in_dim %v479, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v481 = stablehlo.divide %v480, %v472 : tensor<32x640x14x14xf32>
    %v482 = stablehlo.add %v481, %v473 : tensor<32x640x14x14xf32>
    %v483 = stablehlo.rsqrt %v482 : tensor<32x640x14x14xf32>
    %v484 = stablehlo.multiply %v477, %v483 : tensor<32x640x14x14xf32>
    %v485 = stablehlo.broadcast_in_dim %u4dg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v486 = stablehlo.broadcast_in_dim %u4dbt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v487 = stablehlo.multiply %v484, %v485 : tensor<32x640x14x14xf32>
    %v488 = stablehlo.add %v487, %v486 : tensor<32x640x14x14xf32>
    %v489 = stablehlo.reshape %v488 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v490 = stablehlo.reshape %v489 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v491 = stablehlo.constant dense<0.0> : tensor<32x640x14x14xf32>
    %v492 = stablehlo.maximum %v490, %v491 : tensor<32x640x14x14xf32>
    %v493 = stablehlo.reshape %v492 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v494 = stablehlo.reshape %v493 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v495 = stablehlo.convolution(%v494, %u4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v496 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v497 = stablehlo.add %v495, %v496 : tensor<32x160x14x14xf32>
    %v498 = stablehlo.reshape %v497 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v499 = stablehlo.reshape %v498 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v500 = stablehlo.constant dense<0.0> : tensor<f32>
    %v501 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v502 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v503 = stablehlo.reduce(%v499 init: %v500) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v504 = stablehlo.broadcast_in_dim %v503, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v505 = stablehlo.divide %v504, %v501 : tensor<32x160x14x14xf32>
    %v506 = stablehlo.subtract %v499, %v505 : tensor<32x160x14x14xf32>
    %v507 = stablehlo.multiply %v506, %v506 : tensor<32x160x14x14xf32>
    %v508 = stablehlo.reduce(%v507 init: %v500) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v509 = stablehlo.broadcast_in_dim %v508, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v510 = stablehlo.divide %v509, %v501 : tensor<32x160x14x14xf32>
    %v511 = stablehlo.add %v510, %v502 : tensor<32x160x14x14xf32>
    %v512 = stablehlo.rsqrt %v511 : tensor<32x160x14x14xf32>
    %v513 = stablehlo.multiply %v506, %v512 : tensor<32x160x14x14xf32>
    %v514 = stablehlo.broadcast_in_dim %u4pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v515 = stablehlo.broadcast_in_dim %u4pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v516 = stablehlo.multiply %v513, %v514 : tensor<32x160x14x14xf32>
    %v517 = stablehlo.add %v516, %v515 : tensor<32x160x14x14xf32>
    %v518 = stablehlo.reshape %v517 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v519 = stablehlo.reshape %v518 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v520 = stablehlo.reshape %v410 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v521 = stablehlo.add %v519, %v520 : tensor<32x160x14x14xf32>
    %v522 = stablehlo.reshape %v521 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v523 = stablehlo.reshape %v522 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v524 = stablehlo.convolution(%v523, %u5qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<32x160x14x14xf32>
    %v525 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v526 = stablehlo.add %v524, %v525 : tensor<32x160x14x14xf32>
    %v527 = stablehlo.reshape %v526 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v528 = stablehlo.reshape %v527 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v529 = stablehlo.constant dense<0.0> : tensor<f32>
    %v530 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v531 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v532 = stablehlo.reduce(%v528 init: %v529) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v533 = stablehlo.broadcast_in_dim %v532, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v534 = stablehlo.divide %v533, %v530 : tensor<32x160x14x14xf32>
    %v535 = stablehlo.subtract %v528, %v534 : tensor<32x160x14x14xf32>
    %v536 = stablehlo.multiply %v535, %v535 : tensor<32x160x14x14xf32>
    %v537 = stablehlo.reduce(%v536 init: %v529) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v538 = stablehlo.broadcast_in_dim %v537, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v539 = stablehlo.divide %v538, %v530 : tensor<32x160x14x14xf32>
    %v540 = stablehlo.add %v539, %v531 : tensor<32x160x14x14xf32>
    %v541 = stablehlo.rsqrt %v540 : tensor<32x160x14x14xf32>
    %v542 = stablehlo.multiply %v535, %v541 : tensor<32x160x14x14xf32>
    %v543 = stablehlo.broadcast_in_dim %u5qg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v544 = stablehlo.broadcast_in_dim %u5qbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v545 = stablehlo.multiply %v542, %v543 : tensor<32x160x14x14xf32>
    %v546 = stablehlo.add %v545, %v544 : tensor<32x160x14x14xf32>
    %v547 = stablehlo.reshape %v546 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v548 = stablehlo.reshape %v547 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v549 = stablehlo.convolution(%v548, %u5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<32x640x14x14xf32>
    %v550 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v551 = stablehlo.add %v549, %v550 : tensor<32x640x14x14xf32>
    %v552 = stablehlo.reshape %v551 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v553 = stablehlo.reshape %v552 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v554 = stablehlo.constant dense<0.0> : tensor<f32>
    %v555 = stablehlo.constant dense<6272.0> : tensor<32x640x14x14xf32>
    %v556 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v557 = stablehlo.reduce(%v553 init: %v554) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v558 = stablehlo.broadcast_in_dim %v557, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v559 = stablehlo.divide %v558, %v555 : tensor<32x640x14x14xf32>
    %v560 = stablehlo.subtract %v553, %v559 : tensor<32x640x14x14xf32>
    %v561 = stablehlo.multiply %v560, %v560 : tensor<32x640x14x14xf32>
    %v562 = stablehlo.reduce(%v561 init: %v554) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v563 = stablehlo.broadcast_in_dim %v562, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v564 = stablehlo.divide %v563, %v555 : tensor<32x640x14x14xf32>
    %v565 = stablehlo.add %v564, %v556 : tensor<32x640x14x14xf32>
    %v566 = stablehlo.rsqrt %v565 : tensor<32x640x14x14xf32>
    %v567 = stablehlo.multiply %v560, %v566 : tensor<32x640x14x14xf32>
    %v568 = stablehlo.broadcast_in_dim %u5eg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v569 = stablehlo.broadcast_in_dim %u5ebt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v570 = stablehlo.multiply %v567, %v568 : tensor<32x640x14x14xf32>
    %v571 = stablehlo.add %v570, %v569 : tensor<32x640x14x14xf32>
    %v572 = stablehlo.reshape %v571 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v573 = stablehlo.reshape %v572 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v574 = stablehlo.constant dense<0.0> : tensor<32x640x14x14xf32>
    %v575 = stablehlo.maximum %v573, %v574 : tensor<32x640x14x14xf32>
    %v576 = stablehlo.reshape %v575 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v577 = stablehlo.reshape %v576 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v578 = stablehlo.convolution(%v577, %u5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 640 : i64} : (tensor<32x640x14x14xf32>, tensor<640x1x3x3xf32>) -> tensor<32x640x14x14xf32>
    %v579 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v580 = stablehlo.add %v578, %v579 : tensor<32x640x14x14xf32>
    %v581 = stablehlo.reshape %v580 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v582 = stablehlo.reshape %v581 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v583 = stablehlo.constant dense<0.0> : tensor<f32>
    %v584 = stablehlo.constant dense<6272.0> : tensor<32x640x14x14xf32>
    %v585 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v586 = stablehlo.reduce(%v582 init: %v583) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v587 = stablehlo.broadcast_in_dim %v586, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v588 = stablehlo.divide %v587, %v584 : tensor<32x640x14x14xf32>
    %v589 = stablehlo.subtract %v582, %v588 : tensor<32x640x14x14xf32>
    %v590 = stablehlo.multiply %v589, %v589 : tensor<32x640x14x14xf32>
    %v591 = stablehlo.reduce(%v590 init: %v583) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v592 = stablehlo.broadcast_in_dim %v591, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v593 = stablehlo.divide %v592, %v584 : tensor<32x640x14x14xf32>
    %v594 = stablehlo.add %v593, %v585 : tensor<32x640x14x14xf32>
    %v595 = stablehlo.rsqrt %v594 : tensor<32x640x14x14xf32>
    %v596 = stablehlo.multiply %v589, %v595 : tensor<32x640x14x14xf32>
    %v597 = stablehlo.broadcast_in_dim %u5dg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v598 = stablehlo.broadcast_in_dim %u5dbt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v599 = stablehlo.multiply %v596, %v597 : tensor<32x640x14x14xf32>
    %v600 = stablehlo.add %v599, %v598 : tensor<32x640x14x14xf32>
    %v601 = stablehlo.reshape %v600 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v602 = stablehlo.reshape %v601 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v603 = stablehlo.constant dense<0.0> : tensor<32x640x14x14xf32>
    %v604 = stablehlo.maximum %v602, %v603 : tensor<32x640x14x14xf32>
    %v605 = stablehlo.reshape %v604 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v606 = stablehlo.reshape %v605 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v607 = stablehlo.convolution(%v606, %u5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v608 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v609 = stablehlo.add %v607, %v608 : tensor<32x160x14x14xf32>
    %v610 = stablehlo.reshape %v609 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v611 = stablehlo.reshape %v610 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v612 = stablehlo.constant dense<0.0> : tensor<f32>
    %v613 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v614 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v615 = stablehlo.reduce(%v611 init: %v612) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v616 = stablehlo.broadcast_in_dim %v615, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v617 = stablehlo.divide %v616, %v613 : tensor<32x160x14x14xf32>
    %v618 = stablehlo.subtract %v611, %v617 : tensor<32x160x14x14xf32>
    %v619 = stablehlo.multiply %v618, %v618 : tensor<32x160x14x14xf32>
    %v620 = stablehlo.reduce(%v619 init: %v612) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v621 = stablehlo.broadcast_in_dim %v620, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v622 = stablehlo.divide %v621, %v613 : tensor<32x160x14x14xf32>
    %v623 = stablehlo.add %v622, %v614 : tensor<32x160x14x14xf32>
    %v624 = stablehlo.rsqrt %v623 : tensor<32x160x14x14xf32>
    %v625 = stablehlo.multiply %v618, %v624 : tensor<32x160x14x14xf32>
    %v626 = stablehlo.broadcast_in_dim %u5pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v627 = stablehlo.broadcast_in_dim %u5pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v628 = stablehlo.multiply %v625, %v626 : tensor<32x160x14x14xf32>
    %v629 = stablehlo.add %v628, %v627 : tensor<32x160x14x14xf32>
    %v630 = stablehlo.reshape %v629 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v631 = stablehlo.reshape %v630 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v632 = stablehlo.reshape %v522 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v633 = stablehlo.add %v631, %v632 : tensor<32x160x14x14xf32>
    %v634 = stablehlo.reshape %v633 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v635 = stablehlo.reshape %v634 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v636 = stablehlo.convolution(%v635, %u6qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<32x160x14x14xf32>
    %v637 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v638 = stablehlo.add %v636, %v637 : tensor<32x160x14x14xf32>
    %v639 = stablehlo.reshape %v638 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v640 = stablehlo.reshape %v639 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v641 = stablehlo.constant dense<0.0> : tensor<f32>
    %v642 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v643 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v644 = stablehlo.reduce(%v640 init: %v641) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v645 = stablehlo.broadcast_in_dim %v644, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v646 = stablehlo.divide %v645, %v642 : tensor<32x160x14x14xf32>
    %v647 = stablehlo.subtract %v640, %v646 : tensor<32x160x14x14xf32>
    %v648 = stablehlo.multiply %v647, %v647 : tensor<32x160x14x14xf32>
    %v649 = stablehlo.reduce(%v648 init: %v641) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v650 = stablehlo.broadcast_in_dim %v649, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v651 = stablehlo.divide %v650, %v642 : tensor<32x160x14x14xf32>
    %v652 = stablehlo.add %v651, %v643 : tensor<32x160x14x14xf32>
    %v653 = stablehlo.rsqrt %v652 : tensor<32x160x14x14xf32>
    %v654 = stablehlo.multiply %v647, %v653 : tensor<32x160x14x14xf32>
    %v655 = stablehlo.broadcast_in_dim %u6qg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v656 = stablehlo.broadcast_in_dim %u6qbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v657 = stablehlo.multiply %v654, %v655 : tensor<32x160x14x14xf32>
    %v658 = stablehlo.add %v657, %v656 : tensor<32x160x14x14xf32>
    %v659 = stablehlo.reshape %v658 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v660 = stablehlo.reshape %v659 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v661 = stablehlo.convolution(%v660, %u6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<32x640x14x14xf32>
    %v662 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v663 = stablehlo.add %v661, %v662 : tensor<32x640x14x14xf32>
    %v664 = stablehlo.reshape %v663 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v665 = stablehlo.reshape %v664 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v666 = stablehlo.constant dense<0.0> : tensor<f32>
    %v667 = stablehlo.constant dense<6272.0> : tensor<32x640x14x14xf32>
    %v668 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v669 = stablehlo.reduce(%v665 init: %v666) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v670 = stablehlo.broadcast_in_dim %v669, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v671 = stablehlo.divide %v670, %v667 : tensor<32x640x14x14xf32>
    %v672 = stablehlo.subtract %v665, %v671 : tensor<32x640x14x14xf32>
    %v673 = stablehlo.multiply %v672, %v672 : tensor<32x640x14x14xf32>
    %v674 = stablehlo.reduce(%v673 init: %v666) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v675 = stablehlo.broadcast_in_dim %v674, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v676 = stablehlo.divide %v675, %v667 : tensor<32x640x14x14xf32>
    %v677 = stablehlo.add %v676, %v668 : tensor<32x640x14x14xf32>
    %v678 = stablehlo.rsqrt %v677 : tensor<32x640x14x14xf32>
    %v679 = stablehlo.multiply %v672, %v678 : tensor<32x640x14x14xf32>
    %v680 = stablehlo.broadcast_in_dim %u6eg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v681 = stablehlo.broadcast_in_dim %u6ebt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v682 = stablehlo.multiply %v679, %v680 : tensor<32x640x14x14xf32>
    %v683 = stablehlo.add %v682, %v681 : tensor<32x640x14x14xf32>
    %v684 = stablehlo.reshape %v683 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v685 = stablehlo.reshape %v684 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v686 = stablehlo.constant dense<0.0> : tensor<32x640x14x14xf32>
    %v687 = stablehlo.maximum %v685, %v686 : tensor<32x640x14x14xf32>
    %v688 = stablehlo.reshape %v687 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v689 = stablehlo.reshape %v688 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v690 = stablehlo.convolution(%v689, %u6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 640 : i64} : (tensor<32x640x14x14xf32>, tensor<640x1x5x5xf32>) -> tensor<32x640x14x14xf32>
    %v691 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v692 = stablehlo.add %v690, %v691 : tensor<32x640x14x14xf32>
    %v693 = stablehlo.reshape %v692 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v694 = stablehlo.reshape %v693 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v695 = stablehlo.constant dense<0.0> : tensor<f32>
    %v696 = stablehlo.constant dense<6272.0> : tensor<32x640x14x14xf32>
    %v697 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v698 = stablehlo.reduce(%v694 init: %v695) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v699 = stablehlo.broadcast_in_dim %v698, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v700 = stablehlo.divide %v699, %v696 : tensor<32x640x14x14xf32>
    %v701 = stablehlo.subtract %v694, %v700 : tensor<32x640x14x14xf32>
    %v702 = stablehlo.multiply %v701, %v701 : tensor<32x640x14x14xf32>
    %v703 = stablehlo.reduce(%v702 init: %v695) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v704 = stablehlo.broadcast_in_dim %v703, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v705 = stablehlo.divide %v704, %v696 : tensor<32x640x14x14xf32>
    %v706 = stablehlo.add %v705, %v697 : tensor<32x640x14x14xf32>
    %v707 = stablehlo.rsqrt %v706 : tensor<32x640x14x14xf32>
    %v708 = stablehlo.multiply %v701, %v707 : tensor<32x640x14x14xf32>
    %v709 = stablehlo.broadcast_in_dim %u6dg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v710 = stablehlo.broadcast_in_dim %u6dbt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v711 = stablehlo.multiply %v708, %v709 : tensor<32x640x14x14xf32>
    %v712 = stablehlo.add %v711, %v710 : tensor<32x640x14x14xf32>
    %v713 = stablehlo.reshape %v712 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v714 = stablehlo.reshape %v713 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v715 = stablehlo.constant dense<0.0> : tensor<32x640x14x14xf32>
    %v716 = stablehlo.maximum %v714, %v715 : tensor<32x640x14x14xf32>
    %v717 = stablehlo.reshape %v716 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v718 = stablehlo.reshape %v717 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v719 = stablehlo.convolution(%v718, %u6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v720 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v721 = stablehlo.add %v719, %v720 : tensor<32x160x14x14xf32>
    %v722 = stablehlo.reshape %v721 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v723 = stablehlo.reshape %v722 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v724 = stablehlo.constant dense<0.0> : tensor<f32>
    %v725 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v726 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v727 = stablehlo.reduce(%v723 init: %v724) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v728 = stablehlo.broadcast_in_dim %v727, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v729 = stablehlo.divide %v728, %v725 : tensor<32x160x14x14xf32>
    %v730 = stablehlo.subtract %v723, %v729 : tensor<32x160x14x14xf32>
    %v731 = stablehlo.multiply %v730, %v730 : tensor<32x160x14x14xf32>
    %v732 = stablehlo.reduce(%v731 init: %v724) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v733 = stablehlo.broadcast_in_dim %v732, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v734 = stablehlo.divide %v733, %v725 : tensor<32x160x14x14xf32>
    %v735 = stablehlo.add %v734, %v726 : tensor<32x160x14x14xf32>
    %v736 = stablehlo.rsqrt %v735 : tensor<32x160x14x14xf32>
    %v737 = stablehlo.multiply %v730, %v736 : tensor<32x160x14x14xf32>
    %v738 = stablehlo.broadcast_in_dim %u6pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v739 = stablehlo.broadcast_in_dim %u6pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v740 = stablehlo.multiply %v737, %v738 : tensor<32x160x14x14xf32>
    %v741 = stablehlo.add %v740, %v739 : tensor<32x160x14x14xf32>
    %v742 = stablehlo.reshape %v741 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v743 = stablehlo.reshape %v742 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v744 = stablehlo.reshape %v634 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v745 = stablehlo.add %v743, %v744 : tensor<32x160x14x14xf32>
    %v746 = stablehlo.reshape %v745 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v747 = stablehlo.reshape %v746 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v748 = stablehlo.convolution(%v747, %u7qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<32x160x14x14xf32>
    %v749 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v750 = stablehlo.add %v748, %v749 : tensor<32x160x14x14xf32>
    %v751 = stablehlo.reshape %v750 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v752 = stablehlo.reshape %v751 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v753 = stablehlo.constant dense<0.0> : tensor<f32>
    %v754 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v755 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v756 = stablehlo.reduce(%v752 init: %v753) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v757 = stablehlo.broadcast_in_dim %v756, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v758 = stablehlo.divide %v757, %v754 : tensor<32x160x14x14xf32>
    %v759 = stablehlo.subtract %v752, %v758 : tensor<32x160x14x14xf32>
    %v760 = stablehlo.multiply %v759, %v759 : tensor<32x160x14x14xf32>
    %v761 = stablehlo.reduce(%v760 init: %v753) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v762 = stablehlo.broadcast_in_dim %v761, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v763 = stablehlo.divide %v762, %v754 : tensor<32x160x14x14xf32>
    %v764 = stablehlo.add %v763, %v755 : tensor<32x160x14x14xf32>
    %v765 = stablehlo.rsqrt %v764 : tensor<32x160x14x14xf32>
    %v766 = stablehlo.multiply %v759, %v765 : tensor<32x160x14x14xf32>
    %v767 = stablehlo.broadcast_in_dim %u7qg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v768 = stablehlo.broadcast_in_dim %u7qbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v769 = stablehlo.multiply %v766, %v767 : tensor<32x160x14x14xf32>
    %v770 = stablehlo.add %v769, %v768 : tensor<32x160x14x14xf32>
    %v771 = stablehlo.reshape %v770 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v772 = stablehlo.reshape %v771 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v773 = stablehlo.convolution(%v772, %u7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<32x640x14x14xf32>
    %v774 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v775 = stablehlo.add %v773, %v774 : tensor<32x640x14x14xf32>
    %v776 = stablehlo.reshape %v775 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v777 = stablehlo.reshape %v776 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v778 = stablehlo.constant dense<0.0> : tensor<f32>
    %v779 = stablehlo.constant dense<6272.0> : tensor<32x640x14x14xf32>
    %v780 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v781 = stablehlo.reduce(%v777 init: %v778) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v782 = stablehlo.broadcast_in_dim %v781, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v783 = stablehlo.divide %v782, %v779 : tensor<32x640x14x14xf32>
    %v784 = stablehlo.subtract %v777, %v783 : tensor<32x640x14x14xf32>
    %v785 = stablehlo.multiply %v784, %v784 : tensor<32x640x14x14xf32>
    %v786 = stablehlo.reduce(%v785 init: %v778) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v787 = stablehlo.broadcast_in_dim %v786, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v788 = stablehlo.divide %v787, %v779 : tensor<32x640x14x14xf32>
    %v789 = stablehlo.add %v788, %v780 : tensor<32x640x14x14xf32>
    %v790 = stablehlo.rsqrt %v789 : tensor<32x640x14x14xf32>
    %v791 = stablehlo.multiply %v784, %v790 : tensor<32x640x14x14xf32>
    %v792 = stablehlo.broadcast_in_dim %u7eg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v793 = stablehlo.broadcast_in_dim %u7ebt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v794 = stablehlo.multiply %v791, %v792 : tensor<32x640x14x14xf32>
    %v795 = stablehlo.add %v794, %v793 : tensor<32x640x14x14xf32>
    %v796 = stablehlo.reshape %v795 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v797 = stablehlo.reshape %v796 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v798 = stablehlo.constant dense<0.0> : tensor<32x640x14x14xf32>
    %v799 = stablehlo.maximum %v797, %v798 : tensor<32x640x14x14xf32>
    %v800 = stablehlo.reshape %v799 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v801 = stablehlo.reshape %v800 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v802 = stablehlo.convolution(%v801, %u7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 640 : i64} : (tensor<32x640x14x14xf32>, tensor<640x1x3x3xf32>) -> tensor<32x640x14x14xf32>
    %v803 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v804 = stablehlo.add %v802, %v803 : tensor<32x640x14x14xf32>
    %v805 = stablehlo.reshape %v804 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v806 = stablehlo.reshape %v805 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v807 = stablehlo.constant dense<0.0> : tensor<f32>
    %v808 = stablehlo.constant dense<6272.0> : tensor<32x640x14x14xf32>
    %v809 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v810 = stablehlo.reduce(%v806 init: %v807) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v811 = stablehlo.broadcast_in_dim %v810, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v812 = stablehlo.divide %v811, %v808 : tensor<32x640x14x14xf32>
    %v813 = stablehlo.subtract %v806, %v812 : tensor<32x640x14x14xf32>
    %v814 = stablehlo.multiply %v813, %v813 : tensor<32x640x14x14xf32>
    %v815 = stablehlo.reduce(%v814 init: %v807) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v816 = stablehlo.broadcast_in_dim %v815, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v817 = stablehlo.divide %v816, %v808 : tensor<32x640x14x14xf32>
    %v818 = stablehlo.add %v817, %v809 : tensor<32x640x14x14xf32>
    %v819 = stablehlo.rsqrt %v818 : tensor<32x640x14x14xf32>
    %v820 = stablehlo.multiply %v813, %v819 : tensor<32x640x14x14xf32>
    %v821 = stablehlo.broadcast_in_dim %u7dg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v822 = stablehlo.broadcast_in_dim %u7dbt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v823 = stablehlo.multiply %v820, %v821 : tensor<32x640x14x14xf32>
    %v824 = stablehlo.add %v823, %v822 : tensor<32x640x14x14xf32>
    %v825 = stablehlo.reshape %v824 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v826 = stablehlo.reshape %v825 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v827 = stablehlo.constant dense<0.0> : tensor<32x640x14x14xf32>
    %v828 = stablehlo.maximum %v826, %v827 : tensor<32x640x14x14xf32>
    %v829 = stablehlo.reshape %v828 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v830 = stablehlo.reshape %v829 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v831 = stablehlo.convolution(%v830, %u7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v832 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v833 = stablehlo.add %v831, %v832 : tensor<32x160x14x14xf32>
    %v834 = stablehlo.reshape %v833 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v835 = stablehlo.reshape %v834 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v836 = stablehlo.constant dense<0.0> : tensor<f32>
    %v837 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v838 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v839 = stablehlo.reduce(%v835 init: %v836) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v840 = stablehlo.broadcast_in_dim %v839, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v841 = stablehlo.divide %v840, %v837 : tensor<32x160x14x14xf32>
    %v842 = stablehlo.subtract %v835, %v841 : tensor<32x160x14x14xf32>
    %v843 = stablehlo.multiply %v842, %v842 : tensor<32x160x14x14xf32>
    %v844 = stablehlo.reduce(%v843 init: %v836) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v845 = stablehlo.broadcast_in_dim %v844, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v846 = stablehlo.divide %v845, %v837 : tensor<32x160x14x14xf32>
    %v847 = stablehlo.add %v846, %v838 : tensor<32x160x14x14xf32>
    %v848 = stablehlo.rsqrt %v847 : tensor<32x160x14x14xf32>
    %v849 = stablehlo.multiply %v842, %v848 : tensor<32x160x14x14xf32>
    %v850 = stablehlo.broadcast_in_dim %u7pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v851 = stablehlo.broadcast_in_dim %u7pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v852 = stablehlo.multiply %v849, %v850 : tensor<32x160x14x14xf32>
    %v853 = stablehlo.add %v852, %v851 : tensor<32x160x14x14xf32>
    %v854 = stablehlo.reshape %v853 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v855 = stablehlo.reshape %v854 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v856 = stablehlo.reshape %v746 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v857 = stablehlo.add %v855, %v856 : tensor<32x160x14x14xf32>
    %v858 = stablehlo.reshape %v857 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v859 = stablehlo.reshape %v858 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v860 = stablehlo.convolution(%v859, %u8qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<32x160x14x14xf32>
    %v861 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v862 = stablehlo.add %v860, %v861 : tensor<32x160x14x14xf32>
    %v863 = stablehlo.reshape %v862 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v864 = stablehlo.reshape %v863 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v865 = stablehlo.constant dense<0.0> : tensor<f32>
    %v866 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v867 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v868 = stablehlo.reduce(%v864 init: %v865) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v869 = stablehlo.broadcast_in_dim %v868, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v870 = stablehlo.divide %v869, %v866 : tensor<32x160x14x14xf32>
    %v871 = stablehlo.subtract %v864, %v870 : tensor<32x160x14x14xf32>
    %v872 = stablehlo.multiply %v871, %v871 : tensor<32x160x14x14xf32>
    %v873 = stablehlo.reduce(%v872 init: %v865) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v874 = stablehlo.broadcast_in_dim %v873, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v875 = stablehlo.divide %v874, %v866 : tensor<32x160x14x14xf32>
    %v876 = stablehlo.add %v875, %v867 : tensor<32x160x14x14xf32>
    %v877 = stablehlo.rsqrt %v876 : tensor<32x160x14x14xf32>
    %v878 = stablehlo.multiply %v871, %v877 : tensor<32x160x14x14xf32>
    %v879 = stablehlo.broadcast_in_dim %u8qg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v880 = stablehlo.broadcast_in_dim %u8qbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v881 = stablehlo.multiply %v878, %v879 : tensor<32x160x14x14xf32>
    %v882 = stablehlo.add %v881, %v880 : tensor<32x160x14x14xf32>
    %v883 = stablehlo.reshape %v882 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v884 = stablehlo.reshape %v883 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v885 = stablehlo.convolution(%v884, %u8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<32x640x14x14xf32>
    %v886 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v887 = stablehlo.add %v885, %v886 : tensor<32x640x14x14xf32>
    %v888 = stablehlo.reshape %v887 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v889 = stablehlo.reshape %v888 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v890 = stablehlo.constant dense<0.0> : tensor<f32>
    %v891 = stablehlo.constant dense<6272.0> : tensor<32x640x14x14xf32>
    %v892 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v893 = stablehlo.reduce(%v889 init: %v890) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v894 = stablehlo.broadcast_in_dim %v893, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v895 = stablehlo.divide %v894, %v891 : tensor<32x640x14x14xf32>
    %v896 = stablehlo.subtract %v889, %v895 : tensor<32x640x14x14xf32>
    %v897 = stablehlo.multiply %v896, %v896 : tensor<32x640x14x14xf32>
    %v898 = stablehlo.reduce(%v897 init: %v890) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v899 = stablehlo.broadcast_in_dim %v898, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v900 = stablehlo.divide %v899, %v891 : tensor<32x640x14x14xf32>
    %v901 = stablehlo.add %v900, %v892 : tensor<32x640x14x14xf32>
    %v902 = stablehlo.rsqrt %v901 : tensor<32x640x14x14xf32>
    %v903 = stablehlo.multiply %v896, %v902 : tensor<32x640x14x14xf32>
    %v904 = stablehlo.broadcast_in_dim %u8eg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v905 = stablehlo.broadcast_in_dim %u8ebt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v906 = stablehlo.multiply %v903, %v904 : tensor<32x640x14x14xf32>
    %v907 = stablehlo.add %v906, %v905 : tensor<32x640x14x14xf32>
    %v908 = stablehlo.reshape %v907 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v909 = stablehlo.reshape %v908 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v910 = stablehlo.constant dense<0.0> : tensor<32x640x14x14xf32>
    %v911 = stablehlo.maximum %v909, %v910 : tensor<32x640x14x14xf32>
    %v912 = stablehlo.reshape %v911 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v913 = stablehlo.reshape %v912 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v914 = stablehlo.convolution(%v913, %u8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v915 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v916 = stablehlo.add %v914, %v915 : tensor<32x160x14x14xf32>
    %v917 = stablehlo.reshape %v916 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v918 = stablehlo.reshape %v917 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v919 = stablehlo.constant dense<0.0> : tensor<f32>
    %v920 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v921 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v922 = stablehlo.reduce(%v918 init: %v919) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v923 = stablehlo.broadcast_in_dim %v922, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v924 = stablehlo.divide %v923, %v920 : tensor<32x160x14x14xf32>
    %v925 = stablehlo.subtract %v918, %v924 : tensor<32x160x14x14xf32>
    %v926 = stablehlo.multiply %v925, %v925 : tensor<32x160x14x14xf32>
    %v927 = stablehlo.reduce(%v926 init: %v919) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v928 = stablehlo.broadcast_in_dim %v927, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v929 = stablehlo.divide %v928, %v920 : tensor<32x160x14x14xf32>
    %v930 = stablehlo.add %v929, %v921 : tensor<32x160x14x14xf32>
    %v931 = stablehlo.rsqrt %v930 : tensor<32x160x14x14xf32>
    %v932 = stablehlo.multiply %v925, %v931 : tensor<32x160x14x14xf32>
    %v933 = stablehlo.broadcast_in_dim %u8pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v934 = stablehlo.broadcast_in_dim %u8pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v935 = stablehlo.multiply %v932, %v933 : tensor<32x160x14x14xf32>
    %v936 = stablehlo.add %v935, %v934 : tensor<32x160x14x14xf32>
    %v937 = stablehlo.reshape %v936 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v938 = stablehlo.reshape %v937 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v939 = stablehlo.reshape %v858 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v940 = stablehlo.add %v938, %v939 : tensor<32x160x14x14xf32>
    %v941 = stablehlo.reshape %v940 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v942 = stablehlo.reshape %v941 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v943 = stablehlo.convolution(%v942, %u9eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x14x14xf32>, tensor<320x160x1x1xf32>) -> tensor<32x320x14x14xf32>
    %v944 = stablehlo.broadcast_in_dim %zb320, dims = [1] : (tensor<320xf32>) -> tensor<32x320x14x14xf32>
    %v945 = stablehlo.add %v943, %v944 : tensor<32x320x14x14xf32>
    %v946 = stablehlo.reshape %v945 : (tensor<32x320x14x14xf32>) -> tensor<32x62720xf32>
    %v947 = stablehlo.reshape %v946 : (tensor<32x62720xf32>) -> tensor<32x320x14x14xf32>
    %v948 = stablehlo.constant dense<0.0> : tensor<f32>
    %v949 = stablehlo.constant dense<6272.0> : tensor<32x320x14x14xf32>
    %v950 = stablehlo.constant dense<1.0e-5> : tensor<32x320x14x14xf32>
    %v951 = stablehlo.reduce(%v947 init: %v948) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x14x14xf32>, tensor<f32>) -> tensor<320xf32>
    %v952 = stablehlo.broadcast_in_dim %v951, dims = [1] : (tensor<320xf32>) -> tensor<32x320x14x14xf32>
    %v953 = stablehlo.divide %v952, %v949 : tensor<32x320x14x14xf32>
    %v954 = stablehlo.subtract %v947, %v953 : tensor<32x320x14x14xf32>
    %v955 = stablehlo.multiply %v954, %v954 : tensor<32x320x14x14xf32>
    %v956 = stablehlo.reduce(%v955 init: %v948) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x14x14xf32>, tensor<f32>) -> tensor<320xf32>
    %v957 = stablehlo.broadcast_in_dim %v956, dims = [1] : (tensor<320xf32>) -> tensor<32x320x14x14xf32>
    %v958 = stablehlo.divide %v957, %v949 : tensor<32x320x14x14xf32>
    %v959 = stablehlo.add %v958, %v950 : tensor<32x320x14x14xf32>
    %v960 = stablehlo.rsqrt %v959 : tensor<32x320x14x14xf32>
    %v961 = stablehlo.multiply %v954, %v960 : tensor<32x320x14x14xf32>
    %v962 = stablehlo.broadcast_in_dim %u9eg, dims = [1] : (tensor<320xf32>) -> tensor<32x320x14x14xf32>
    %v963 = stablehlo.broadcast_in_dim %u9ebt, dims = [1] : (tensor<320xf32>) -> tensor<32x320x14x14xf32>
    %v964 = stablehlo.multiply %v961, %v962 : tensor<32x320x14x14xf32>
    %v965 = stablehlo.add %v964, %v963 : tensor<32x320x14x14xf32>
    %v966 = stablehlo.reshape %v965 : (tensor<32x320x14x14xf32>) -> tensor<32x62720xf32>
    %v967 = stablehlo.reshape %v966 : (tensor<32x62720xf32>) -> tensor<32x320x14x14xf32>
    %v968 = stablehlo.constant dense<0.0> : tensor<32x320x14x14xf32>
    %v969 = stablehlo.maximum %v967, %v968 : tensor<32x320x14x14xf32>
    %v970 = stablehlo.reshape %v969 : (tensor<32x320x14x14xf32>) -> tensor<32x62720xf32>
    %v971 = stablehlo.reshape %v970 : (tensor<32x62720xf32>) -> tensor<32x320x14x14xf32>
    %v972 = stablehlo.convolution(%v971, %u9pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x320x14x14xf32>, tensor<160x320x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v973 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v974 = stablehlo.add %v972, %v973 : tensor<32x160x14x14xf32>
    %v975 = stablehlo.reshape %v974 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v976 = stablehlo.reshape %v975 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v977 = stablehlo.constant dense<0.0> : tensor<f32>
    %v978 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v979 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v980 = stablehlo.reduce(%v976 init: %v977) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v981 = stablehlo.broadcast_in_dim %v980, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v982 = stablehlo.divide %v981, %v978 : tensor<32x160x14x14xf32>
    %v983 = stablehlo.subtract %v976, %v982 : tensor<32x160x14x14xf32>
    %v984 = stablehlo.multiply %v983, %v983 : tensor<32x160x14x14xf32>
    %v985 = stablehlo.reduce(%v984 init: %v977) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v986 = stablehlo.broadcast_in_dim %v985, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v987 = stablehlo.divide %v986, %v978 : tensor<32x160x14x14xf32>
    %v988 = stablehlo.add %v987, %v979 : tensor<32x160x14x14xf32>
    %v989 = stablehlo.rsqrt %v988 : tensor<32x160x14x14xf32>
    %v990 = stablehlo.multiply %v983, %v989 : tensor<32x160x14x14xf32>
    %v991 = stablehlo.broadcast_in_dim %u9pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v992 = stablehlo.broadcast_in_dim %u9pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v993 = stablehlo.multiply %v990, %v991 : tensor<32x160x14x14xf32>
    %v994 = stablehlo.add %v993, %v992 : tensor<32x160x14x14xf32>
    %v995 = stablehlo.reshape %v994 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v996 = stablehlo.reshape %v995 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v997 = stablehlo.reshape %v941 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v998 = stablehlo.add %v996, %v997 : tensor<32x160x14x14xf32>
    %v999 = stablehlo.reshape %v998 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v1000 = stablehlo.reshape %v999 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v1001 = stablehlo.convolution(%v1000, %u10qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<32x160x14x14xf32>
    %v1002 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v1003 = stablehlo.add %v1001, %v1002 : tensor<32x160x14x14xf32>
    %v1004 = stablehlo.reshape %v1003 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v1005 = stablehlo.reshape %v1004 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v1006 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1007 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v1008 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v1009 = stablehlo.reduce(%v1005 init: %v1006) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v1010 = stablehlo.broadcast_in_dim %v1009, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v1011 = stablehlo.divide %v1010, %v1007 : tensor<32x160x14x14xf32>
    %v1012 = stablehlo.subtract %v1005, %v1011 : tensor<32x160x14x14xf32>
    %v1013 = stablehlo.multiply %v1012, %v1012 : tensor<32x160x14x14xf32>
    %v1014 = stablehlo.reduce(%v1013 init: %v1006) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v1015 = stablehlo.broadcast_in_dim %v1014, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v1016 = stablehlo.divide %v1015, %v1007 : tensor<32x160x14x14xf32>
    %v1017 = stablehlo.add %v1016, %v1008 : tensor<32x160x14x14xf32>
    %v1018 = stablehlo.rsqrt %v1017 : tensor<32x160x14x14xf32>
    %v1019 = stablehlo.multiply %v1012, %v1018 : tensor<32x160x14x14xf32>
    %v1020 = stablehlo.broadcast_in_dim %u10qg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v1021 = stablehlo.broadcast_in_dim %u10qbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v1022 = stablehlo.multiply %v1019, %v1020 : tensor<32x160x14x14xf32>
    %v1023 = stablehlo.add %v1022, %v1021 : tensor<32x160x14x14xf32>
    %v1024 = stablehlo.reshape %v1023 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v1025 = stablehlo.reshape %v1024 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v1026 = stablehlo.convolution(%v1025, %u10eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<32x640x14x14xf32>
    %v1027 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v1028 = stablehlo.add %v1026, %v1027 : tensor<32x640x14x14xf32>
    %v1029 = stablehlo.reshape %v1028 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v1030 = stablehlo.reshape %v1029 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v1031 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1032 = stablehlo.constant dense<6272.0> : tensor<32x640x14x14xf32>
    %v1033 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v1034 = stablehlo.reduce(%v1030 init: %v1031) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v1035 = stablehlo.broadcast_in_dim %v1034, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v1036 = stablehlo.divide %v1035, %v1032 : tensor<32x640x14x14xf32>
    %v1037 = stablehlo.subtract %v1030, %v1036 : tensor<32x640x14x14xf32>
    %v1038 = stablehlo.multiply %v1037, %v1037 : tensor<32x640x14x14xf32>
    %v1039 = stablehlo.reduce(%v1038 init: %v1031) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x640x14x14xf32>, tensor<f32>) -> tensor<640xf32>
    %v1040 = stablehlo.broadcast_in_dim %v1039, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v1041 = stablehlo.divide %v1040, %v1032 : tensor<32x640x14x14xf32>
    %v1042 = stablehlo.add %v1041, %v1033 : tensor<32x640x14x14xf32>
    %v1043 = stablehlo.rsqrt %v1042 : tensor<32x640x14x14xf32>
    %v1044 = stablehlo.multiply %v1037, %v1043 : tensor<32x640x14x14xf32>
    %v1045 = stablehlo.broadcast_in_dim %u10eg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v1046 = stablehlo.broadcast_in_dim %u10ebt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v1047 = stablehlo.multiply %v1044, %v1045 : tensor<32x640x14x14xf32>
    %v1048 = stablehlo.add %v1047, %v1046 : tensor<32x640x14x14xf32>
    %v1049 = stablehlo.reshape %v1048 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v1050 = stablehlo.reshape %v1049 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v1051 = stablehlo.constant dense<0.0> : tensor<32x640x14x14xf32>
    %v1052 = stablehlo.maximum %v1050, %v1051 : tensor<32x640x14x14xf32>
    %v1053 = stablehlo.reshape %v1052 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v1054 = stablehlo.reshape %v1053 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v1055 = stablehlo.convolution(%v1054, %u10pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v1056 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v1057 = stablehlo.add %v1055, %v1056 : tensor<32x160x14x14xf32>
    %v1058 = stablehlo.reshape %v1057 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v1059 = stablehlo.reshape %v1058 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v1060 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1061 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v1062 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v1063 = stablehlo.reduce(%v1059 init: %v1060) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v1064 = stablehlo.broadcast_in_dim %v1063, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v1065 = stablehlo.divide %v1064, %v1061 : tensor<32x160x14x14xf32>
    %v1066 = stablehlo.subtract %v1059, %v1065 : tensor<32x160x14x14xf32>
    %v1067 = stablehlo.multiply %v1066, %v1066 : tensor<32x160x14x14xf32>
    %v1068 = stablehlo.reduce(%v1067 init: %v1060) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v1069 = stablehlo.broadcast_in_dim %v1068, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v1070 = stablehlo.divide %v1069, %v1061 : tensor<32x160x14x14xf32>
    %v1071 = stablehlo.add %v1070, %v1062 : tensor<32x160x14x14xf32>
    %v1072 = stablehlo.rsqrt %v1071 : tensor<32x160x14x14xf32>
    %v1073 = stablehlo.multiply %v1066, %v1072 : tensor<32x160x14x14xf32>
    %v1074 = stablehlo.broadcast_in_dim %u10pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v1075 = stablehlo.broadcast_in_dim %u10pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v1076 = stablehlo.multiply %v1073, %v1074 : tensor<32x160x14x14xf32>
    %v1077 = stablehlo.add %v1076, %v1075 : tensor<32x160x14x14xf32>
    %v1078 = stablehlo.reshape %v1077 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v1079 = stablehlo.reshape %v1078 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v1080 = stablehlo.reshape %v999 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v1081 = stablehlo.add %v1079, %v1080 : tensor<32x160x14x14xf32>
    %v1082 = stablehlo.reshape %v1081 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v1083 = stablehlo.reshape %v1082 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v1084 = stablehlo.convolution(%v1083, %u11qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x14x14xf32>, tensor<160x1x5x5xf32>) -> tensor<32x160x14x14xf32>
    %v1085 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v1086 = stablehlo.add %v1084, %v1085 : tensor<32x160x14x14xf32>
    %v1087 = stablehlo.reshape %v1086 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v1088 = stablehlo.reshape %v1087 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v1089 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1090 = stablehlo.constant dense<6272.0> : tensor<32x160x14x14xf32>
    %v1091 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v1092 = stablehlo.reduce(%v1088 init: %v1089) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v1093 = stablehlo.broadcast_in_dim %v1092, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v1094 = stablehlo.divide %v1093, %v1090 : tensor<32x160x14x14xf32>
    %v1095 = stablehlo.subtract %v1088, %v1094 : tensor<32x160x14x14xf32>
    %v1096 = stablehlo.multiply %v1095, %v1095 : tensor<32x160x14x14xf32>
    %v1097 = stablehlo.reduce(%v1096 init: %v1089) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x160x14x14xf32>, tensor<f32>) -> tensor<160xf32>
    %v1098 = stablehlo.broadcast_in_dim %v1097, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v1099 = stablehlo.divide %v1098, %v1090 : tensor<32x160x14x14xf32>
    %v1100 = stablehlo.add %v1099, %v1091 : tensor<32x160x14x14xf32>
    %v1101 = stablehlo.rsqrt %v1100 : tensor<32x160x14x14xf32>
    %v1102 = stablehlo.multiply %v1095, %v1101 : tensor<32x160x14x14xf32>
    %v1103 = stablehlo.broadcast_in_dim %u11qg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v1104 = stablehlo.broadcast_in_dim %u11qbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v1105 = stablehlo.multiply %v1102, %v1103 : tensor<32x160x14x14xf32>
    %v1106 = stablehlo.add %v1105, %v1104 : tensor<32x160x14x14xf32>
    %v1107 = stablehlo.reshape %v1106 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v1108 = stablehlo.reshape %v1107 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v1109 = stablehlo.convolution(%v1108, %u11eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x14x14xf32>, tensor<960x160x1x1xf32>) -> tensor<32x960x14x14xf32>
    %v1110 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x14x14xf32>
    %v1111 = stablehlo.add %v1109, %v1110 : tensor<32x960x14x14xf32>
    %v1112 = stablehlo.reshape %v1111 : (tensor<32x960x14x14xf32>) -> tensor<32x188160xf32>
    %v1113 = stablehlo.reshape %v1112 : (tensor<32x188160xf32>) -> tensor<32x960x14x14xf32>
    %v1114 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1115 = stablehlo.constant dense<6272.0> : tensor<32x960x14x14xf32>
    %v1116 = stablehlo.constant dense<1.0e-5> : tensor<32x960x14x14xf32>
    %v1117 = stablehlo.reduce(%v1113 init: %v1114) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x14x14xf32>, tensor<f32>) -> tensor<960xf32>
    %v1118 = stablehlo.broadcast_in_dim %v1117, dims = [1] : (tensor<960xf32>) -> tensor<32x960x14x14xf32>
    %v1119 = stablehlo.divide %v1118, %v1115 : tensor<32x960x14x14xf32>
    %v1120 = stablehlo.subtract %v1113, %v1119 : tensor<32x960x14x14xf32>
    %v1121 = stablehlo.multiply %v1120, %v1120 : tensor<32x960x14x14xf32>
    %v1122 = stablehlo.reduce(%v1121 init: %v1114) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x14x14xf32>, tensor<f32>) -> tensor<960xf32>
    %v1123 = stablehlo.broadcast_in_dim %v1122, dims = [1] : (tensor<960xf32>) -> tensor<32x960x14x14xf32>
    %v1124 = stablehlo.divide %v1123, %v1115 : tensor<32x960x14x14xf32>
    %v1125 = stablehlo.add %v1124, %v1116 : tensor<32x960x14x14xf32>
    %v1126 = stablehlo.rsqrt %v1125 : tensor<32x960x14x14xf32>
    %v1127 = stablehlo.multiply %v1120, %v1126 : tensor<32x960x14x14xf32>
    %v1128 = stablehlo.broadcast_in_dim %u11eg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x14x14xf32>
    %v1129 = stablehlo.broadcast_in_dim %u11ebt, dims = [1] : (tensor<960xf32>) -> tensor<32x960x14x14xf32>
    %v1130 = stablehlo.multiply %v1127, %v1128 : tensor<32x960x14x14xf32>
    %v1131 = stablehlo.add %v1130, %v1129 : tensor<32x960x14x14xf32>
    %v1132 = stablehlo.reshape %v1131 : (tensor<32x960x14x14xf32>) -> tensor<32x188160xf32>
    %v1133 = stablehlo.reshape %v1132 : (tensor<32x188160xf32>) -> tensor<32x960x14x14xf32>
    %v1134 = stablehlo.constant dense<0.0> : tensor<32x960x14x14xf32>
    %v1135 = stablehlo.maximum %v1133, %v1134 : tensor<32x960x14x14xf32>
    %v1136 = stablehlo.reshape %v1135 : (tensor<32x960x14x14xf32>) -> tensor<32x188160xf32>
    %v1137 = stablehlo.reshape %v1136 : (tensor<32x188160xf32>) -> tensor<32x960x14x14xf32>
    %v1138 = stablehlo.convolution(%v1137, %u11dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x14x14xf32>, tensor<960x1x5x5xf32>) -> tensor<32x960x7x7xf32>
    %v1139 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1140 = stablehlo.add %v1138, %v1139 : tensor<32x960x7x7xf32>
    %v1141 = stablehlo.reshape %v1140 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1142 = stablehlo.reshape %v1141 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1143 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1144 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v1145 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1146 = stablehlo.reduce(%v1142 init: %v1143) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1147 = stablehlo.broadcast_in_dim %v1146, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1148 = stablehlo.divide %v1147, %v1144 : tensor<32x960x7x7xf32>
    %v1149 = stablehlo.subtract %v1142, %v1148 : tensor<32x960x7x7xf32>
    %v1150 = stablehlo.multiply %v1149, %v1149 : tensor<32x960x7x7xf32>
    %v1151 = stablehlo.reduce(%v1150 init: %v1143) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v1152 = stablehlo.broadcast_in_dim %v1151, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1153 = stablehlo.divide %v1152, %v1144 : tensor<32x960x7x7xf32>
    %v1154 = stablehlo.add %v1153, %v1145 : tensor<32x960x7x7xf32>
    %v1155 = stablehlo.rsqrt %v1154 : tensor<32x960x7x7xf32>
    %v1156 = stablehlo.multiply %v1149, %v1155 : tensor<32x960x7x7xf32>
    %v1157 = stablehlo.broadcast_in_dim %u11dg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1158 = stablehlo.broadcast_in_dim %u11dbt, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1159 = stablehlo.multiply %v1156, %v1157 : tensor<32x960x7x7xf32>
    %v1160 = stablehlo.add %v1159, %v1158 : tensor<32x960x7x7xf32>
    %v1161 = stablehlo.reshape %v1160 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1162 = stablehlo.reshape %v1161 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1163 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v1164 = stablehlo.maximum %v1162, %v1163 : tensor<32x960x7x7xf32>
    %v1165 = stablehlo.reshape %v1164 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1166 = stablehlo.reshape %v1165 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1167 = stablehlo.convolution(%v1166, %u11pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<256x960x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1168 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1169 = stablehlo.add %v1167, %v1168 : tensor<32x256x7x7xf32>
    %v1170 = stablehlo.reshape %v1169 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1171 = stablehlo.reshape %v1170 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1172 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1173 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1174 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1175 = stablehlo.reduce(%v1171 init: %v1172) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1176 = stablehlo.broadcast_in_dim %v1175, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1177 = stablehlo.divide %v1176, %v1173 : tensor<32x256x7x7xf32>
    %v1178 = stablehlo.subtract %v1171, %v1177 : tensor<32x256x7x7xf32>
    %v1179 = stablehlo.multiply %v1178, %v1178 : tensor<32x256x7x7xf32>
    %v1180 = stablehlo.reduce(%v1179 init: %v1172) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1181 = stablehlo.broadcast_in_dim %v1180, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1182 = stablehlo.divide %v1181, %v1173 : tensor<32x256x7x7xf32>
    %v1183 = stablehlo.add %v1182, %v1174 : tensor<32x256x7x7xf32>
    %v1184 = stablehlo.rsqrt %v1183 : tensor<32x256x7x7xf32>
    %v1185 = stablehlo.multiply %v1178, %v1184 : tensor<32x256x7x7xf32>
    %v1186 = stablehlo.broadcast_in_dim %u11pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1187 = stablehlo.broadcast_in_dim %u11pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1188 = stablehlo.multiply %v1185, %v1186 : tensor<32x256x7x7xf32>
    %v1189 = stablehlo.add %v1188, %v1187 : tensor<32x256x7x7xf32>
    %v1190 = stablehlo.reshape %v1189 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1191 = stablehlo.reshape %v1190 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1192 = stablehlo.convolution(%v1191, %u12qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x7x7xf32>, tensor<256x1x5x5xf32>) -> tensor<32x256x7x7xf32>
    %v1193 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1194 = stablehlo.add %v1192, %v1193 : tensor<32x256x7x7xf32>
    %v1195 = stablehlo.reshape %v1194 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1196 = stablehlo.reshape %v1195 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1197 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1198 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1199 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1200 = stablehlo.reduce(%v1196 init: %v1197) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1201 = stablehlo.broadcast_in_dim %v1200, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1202 = stablehlo.divide %v1201, %v1198 : tensor<32x256x7x7xf32>
    %v1203 = stablehlo.subtract %v1196, %v1202 : tensor<32x256x7x7xf32>
    %v1204 = stablehlo.multiply %v1203, %v1203 : tensor<32x256x7x7xf32>
    %v1205 = stablehlo.reduce(%v1204 init: %v1197) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1206 = stablehlo.broadcast_in_dim %v1205, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1207 = stablehlo.divide %v1206, %v1198 : tensor<32x256x7x7xf32>
    %v1208 = stablehlo.add %v1207, %v1199 : tensor<32x256x7x7xf32>
    %v1209 = stablehlo.rsqrt %v1208 : tensor<32x256x7x7xf32>
    %v1210 = stablehlo.multiply %v1203, %v1209 : tensor<32x256x7x7xf32>
    %v1211 = stablehlo.broadcast_in_dim %u12qg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1212 = stablehlo.broadcast_in_dim %u12qbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1213 = stablehlo.multiply %v1210, %v1211 : tensor<32x256x7x7xf32>
    %v1214 = stablehlo.add %v1213, %v1212 : tensor<32x256x7x7xf32>
    %v1215 = stablehlo.reshape %v1214 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1216 = stablehlo.reshape %v1215 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1217 = stablehlo.convolution(%v1216, %u12eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v1218 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1219 = stablehlo.add %v1217, %v1218 : tensor<32x1024x7x7xf32>
    %v1220 = stablehlo.reshape %v1219 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1221 = stablehlo.reshape %v1220 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1222 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1223 = stablehlo.constant dense<1568.0> : tensor<32x1024x7x7xf32>
    %v1224 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1225 = stablehlo.reduce(%v1221 init: %v1222) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1226 = stablehlo.broadcast_in_dim %v1225, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1227 = stablehlo.divide %v1226, %v1223 : tensor<32x1024x7x7xf32>
    %v1228 = stablehlo.subtract %v1221, %v1227 : tensor<32x1024x7x7xf32>
    %v1229 = stablehlo.multiply %v1228, %v1228 : tensor<32x1024x7x7xf32>
    %v1230 = stablehlo.reduce(%v1229 init: %v1222) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1231 = stablehlo.broadcast_in_dim %v1230, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1232 = stablehlo.divide %v1231, %v1223 : tensor<32x1024x7x7xf32>
    %v1233 = stablehlo.add %v1232, %v1224 : tensor<32x1024x7x7xf32>
    %v1234 = stablehlo.rsqrt %v1233 : tensor<32x1024x7x7xf32>
    %v1235 = stablehlo.multiply %v1228, %v1234 : tensor<32x1024x7x7xf32>
    %v1236 = stablehlo.broadcast_in_dim %u12eg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1237 = stablehlo.broadcast_in_dim %u12ebt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1238 = stablehlo.multiply %v1235, %v1236 : tensor<32x1024x7x7xf32>
    %v1239 = stablehlo.add %v1238, %v1237 : tensor<32x1024x7x7xf32>
    %v1240 = stablehlo.reshape %v1239 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1241 = stablehlo.reshape %v1240 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1242 = stablehlo.constant dense<0.0> : tensor<32x1024x7x7xf32>
    %v1243 = stablehlo.maximum %v1241, %v1242 : tensor<32x1024x7x7xf32>
    %v1244 = stablehlo.reshape %v1243 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1245 = stablehlo.reshape %v1244 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1246 = stablehlo.convolution(%v1245, %u12dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<32x1024x7x7xf32>, tensor<1024x1x5x5xf32>) -> tensor<32x1024x7x7xf32>
    %v1247 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1248 = stablehlo.add %v1246, %v1247 : tensor<32x1024x7x7xf32>
    %v1249 = stablehlo.reshape %v1248 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1250 = stablehlo.reshape %v1249 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1251 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1252 = stablehlo.constant dense<1568.0> : tensor<32x1024x7x7xf32>
    %v1253 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1254 = stablehlo.reduce(%v1250 init: %v1251) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1255 = stablehlo.broadcast_in_dim %v1254, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1256 = stablehlo.divide %v1255, %v1252 : tensor<32x1024x7x7xf32>
    %v1257 = stablehlo.subtract %v1250, %v1256 : tensor<32x1024x7x7xf32>
    %v1258 = stablehlo.multiply %v1257, %v1257 : tensor<32x1024x7x7xf32>
    %v1259 = stablehlo.reduce(%v1258 init: %v1251) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1260 = stablehlo.broadcast_in_dim %v1259, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1261 = stablehlo.divide %v1260, %v1252 : tensor<32x1024x7x7xf32>
    %v1262 = stablehlo.add %v1261, %v1253 : tensor<32x1024x7x7xf32>
    %v1263 = stablehlo.rsqrt %v1262 : tensor<32x1024x7x7xf32>
    %v1264 = stablehlo.multiply %v1257, %v1263 : tensor<32x1024x7x7xf32>
    %v1265 = stablehlo.broadcast_in_dim %u12dg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1266 = stablehlo.broadcast_in_dim %u12dbt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1267 = stablehlo.multiply %v1264, %v1265 : tensor<32x1024x7x7xf32>
    %v1268 = stablehlo.add %v1267, %v1266 : tensor<32x1024x7x7xf32>
    %v1269 = stablehlo.reshape %v1268 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1270 = stablehlo.reshape %v1269 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1271 = stablehlo.constant dense<0.0> : tensor<32x1024x7x7xf32>
    %v1272 = stablehlo.maximum %v1270, %v1271 : tensor<32x1024x7x7xf32>
    %v1273 = stablehlo.reshape %v1272 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1274 = stablehlo.reshape %v1273 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1275 = stablehlo.convolution(%v1274, %u12pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1276 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1277 = stablehlo.add %v1275, %v1276 : tensor<32x256x7x7xf32>
    %v1278 = stablehlo.reshape %v1277 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1279 = stablehlo.reshape %v1278 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1280 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1281 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1282 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1283 = stablehlo.reduce(%v1279 init: %v1280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1284 = stablehlo.broadcast_in_dim %v1283, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1285 = stablehlo.divide %v1284, %v1281 : tensor<32x256x7x7xf32>
    %v1286 = stablehlo.subtract %v1279, %v1285 : tensor<32x256x7x7xf32>
    %v1287 = stablehlo.multiply %v1286, %v1286 : tensor<32x256x7x7xf32>
    %v1288 = stablehlo.reduce(%v1287 init: %v1280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1289 = stablehlo.broadcast_in_dim %v1288, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1290 = stablehlo.divide %v1289, %v1281 : tensor<32x256x7x7xf32>
    %v1291 = stablehlo.add %v1290, %v1282 : tensor<32x256x7x7xf32>
    %v1292 = stablehlo.rsqrt %v1291 : tensor<32x256x7x7xf32>
    %v1293 = stablehlo.multiply %v1286, %v1292 : tensor<32x256x7x7xf32>
    %v1294 = stablehlo.broadcast_in_dim %u12pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1295 = stablehlo.broadcast_in_dim %u12pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1296 = stablehlo.multiply %v1293, %v1294 : tensor<32x256x7x7xf32>
    %v1297 = stablehlo.add %v1296, %v1295 : tensor<32x256x7x7xf32>
    %v1298 = stablehlo.reshape %v1297 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1299 = stablehlo.reshape %v1298 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1300 = stablehlo.reshape %v1190 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1301 = stablehlo.add %v1299, %v1300 : tensor<32x256x7x7xf32>
    %v1302 = stablehlo.reshape %v1301 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1303 = stablehlo.reshape %v1302 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1304 = stablehlo.convolution(%v1303, %u13qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x7x7xf32>, tensor<256x1x3x3xf32>) -> tensor<32x256x7x7xf32>
    %v1305 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1306 = stablehlo.add %v1304, %v1305 : tensor<32x256x7x7xf32>
    %v1307 = stablehlo.reshape %v1306 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1308 = stablehlo.reshape %v1307 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1309 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1310 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1311 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1312 = stablehlo.reduce(%v1308 init: %v1309) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1313 = stablehlo.broadcast_in_dim %v1312, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1314 = stablehlo.divide %v1313, %v1310 : tensor<32x256x7x7xf32>
    %v1315 = stablehlo.subtract %v1308, %v1314 : tensor<32x256x7x7xf32>
    %v1316 = stablehlo.multiply %v1315, %v1315 : tensor<32x256x7x7xf32>
    %v1317 = stablehlo.reduce(%v1316 init: %v1309) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1318 = stablehlo.broadcast_in_dim %v1317, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1319 = stablehlo.divide %v1318, %v1310 : tensor<32x256x7x7xf32>
    %v1320 = stablehlo.add %v1319, %v1311 : tensor<32x256x7x7xf32>
    %v1321 = stablehlo.rsqrt %v1320 : tensor<32x256x7x7xf32>
    %v1322 = stablehlo.multiply %v1315, %v1321 : tensor<32x256x7x7xf32>
    %v1323 = stablehlo.broadcast_in_dim %u13qg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1324 = stablehlo.broadcast_in_dim %u13qbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1325 = stablehlo.multiply %v1322, %v1323 : tensor<32x256x7x7xf32>
    %v1326 = stablehlo.add %v1325, %v1324 : tensor<32x256x7x7xf32>
    %v1327 = stablehlo.reshape %v1326 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1328 = stablehlo.reshape %v1327 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1329 = stablehlo.convolution(%v1328, %u13eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v1330 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1331 = stablehlo.add %v1329, %v1330 : tensor<32x1024x7x7xf32>
    %v1332 = stablehlo.reshape %v1331 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1333 = stablehlo.reshape %v1332 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1334 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1335 = stablehlo.constant dense<1568.0> : tensor<32x1024x7x7xf32>
    %v1336 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1337 = stablehlo.reduce(%v1333 init: %v1334) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1338 = stablehlo.broadcast_in_dim %v1337, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1339 = stablehlo.divide %v1338, %v1335 : tensor<32x1024x7x7xf32>
    %v1340 = stablehlo.subtract %v1333, %v1339 : tensor<32x1024x7x7xf32>
    %v1341 = stablehlo.multiply %v1340, %v1340 : tensor<32x1024x7x7xf32>
    %v1342 = stablehlo.reduce(%v1341 init: %v1334) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1343 = stablehlo.broadcast_in_dim %v1342, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1344 = stablehlo.divide %v1343, %v1335 : tensor<32x1024x7x7xf32>
    %v1345 = stablehlo.add %v1344, %v1336 : tensor<32x1024x7x7xf32>
    %v1346 = stablehlo.rsqrt %v1345 : tensor<32x1024x7x7xf32>
    %v1347 = stablehlo.multiply %v1340, %v1346 : tensor<32x1024x7x7xf32>
    %v1348 = stablehlo.broadcast_in_dim %u13eg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1349 = stablehlo.broadcast_in_dim %u13ebt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1350 = stablehlo.multiply %v1347, %v1348 : tensor<32x1024x7x7xf32>
    %v1351 = stablehlo.add %v1350, %v1349 : tensor<32x1024x7x7xf32>
    %v1352 = stablehlo.reshape %v1351 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1353 = stablehlo.reshape %v1352 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1354 = stablehlo.constant dense<0.0> : tensor<32x1024x7x7xf32>
    %v1355 = stablehlo.maximum %v1353, %v1354 : tensor<32x1024x7x7xf32>
    %v1356 = stablehlo.reshape %v1355 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1357 = stablehlo.reshape %v1356 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1358 = stablehlo.convolution(%v1357, %u13dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<32x1024x7x7xf32>, tensor<1024x1x5x5xf32>) -> tensor<32x1024x7x7xf32>
    %v1359 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1360 = stablehlo.add %v1358, %v1359 : tensor<32x1024x7x7xf32>
    %v1361 = stablehlo.reshape %v1360 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1362 = stablehlo.reshape %v1361 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1363 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1364 = stablehlo.constant dense<1568.0> : tensor<32x1024x7x7xf32>
    %v1365 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1366 = stablehlo.reduce(%v1362 init: %v1363) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1367 = stablehlo.broadcast_in_dim %v1366, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1368 = stablehlo.divide %v1367, %v1364 : tensor<32x1024x7x7xf32>
    %v1369 = stablehlo.subtract %v1362, %v1368 : tensor<32x1024x7x7xf32>
    %v1370 = stablehlo.multiply %v1369, %v1369 : tensor<32x1024x7x7xf32>
    %v1371 = stablehlo.reduce(%v1370 init: %v1363) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1372 = stablehlo.broadcast_in_dim %v1371, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1373 = stablehlo.divide %v1372, %v1364 : tensor<32x1024x7x7xf32>
    %v1374 = stablehlo.add %v1373, %v1365 : tensor<32x1024x7x7xf32>
    %v1375 = stablehlo.rsqrt %v1374 : tensor<32x1024x7x7xf32>
    %v1376 = stablehlo.multiply %v1369, %v1375 : tensor<32x1024x7x7xf32>
    %v1377 = stablehlo.broadcast_in_dim %u13dg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1378 = stablehlo.broadcast_in_dim %u13dbt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1379 = stablehlo.multiply %v1376, %v1377 : tensor<32x1024x7x7xf32>
    %v1380 = stablehlo.add %v1379, %v1378 : tensor<32x1024x7x7xf32>
    %v1381 = stablehlo.reshape %v1380 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1382 = stablehlo.reshape %v1381 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1383 = stablehlo.constant dense<0.0> : tensor<32x1024x7x7xf32>
    %v1384 = stablehlo.maximum %v1382, %v1383 : tensor<32x1024x7x7xf32>
    %v1385 = stablehlo.reshape %v1384 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1386 = stablehlo.reshape %v1385 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1387 = stablehlo.convolution(%v1386, %u13pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1388 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1389 = stablehlo.add %v1387, %v1388 : tensor<32x256x7x7xf32>
    %v1390 = stablehlo.reshape %v1389 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1391 = stablehlo.reshape %v1390 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1392 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1393 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1394 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1395 = stablehlo.reduce(%v1391 init: %v1392) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1396 = stablehlo.broadcast_in_dim %v1395, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1397 = stablehlo.divide %v1396, %v1393 : tensor<32x256x7x7xf32>
    %v1398 = stablehlo.subtract %v1391, %v1397 : tensor<32x256x7x7xf32>
    %v1399 = stablehlo.multiply %v1398, %v1398 : tensor<32x256x7x7xf32>
    %v1400 = stablehlo.reduce(%v1399 init: %v1392) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1401 = stablehlo.broadcast_in_dim %v1400, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1402 = stablehlo.divide %v1401, %v1393 : tensor<32x256x7x7xf32>
    %v1403 = stablehlo.add %v1402, %v1394 : tensor<32x256x7x7xf32>
    %v1404 = stablehlo.rsqrt %v1403 : tensor<32x256x7x7xf32>
    %v1405 = stablehlo.multiply %v1398, %v1404 : tensor<32x256x7x7xf32>
    %v1406 = stablehlo.broadcast_in_dim %u13pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1407 = stablehlo.broadcast_in_dim %u13pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1408 = stablehlo.multiply %v1405, %v1406 : tensor<32x256x7x7xf32>
    %v1409 = stablehlo.add %v1408, %v1407 : tensor<32x256x7x7xf32>
    %v1410 = stablehlo.reshape %v1409 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1411 = stablehlo.reshape %v1410 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1412 = stablehlo.reshape %v1302 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1413 = stablehlo.add %v1411, %v1412 : tensor<32x256x7x7xf32>
    %v1414 = stablehlo.reshape %v1413 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1415 = stablehlo.reshape %v1414 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1416 = stablehlo.convolution(%v1415, %u14qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x7x7xf32>, tensor<256x1x3x3xf32>) -> tensor<32x256x7x7xf32>
    %v1417 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1418 = stablehlo.add %v1416, %v1417 : tensor<32x256x7x7xf32>
    %v1419 = stablehlo.reshape %v1418 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1420 = stablehlo.reshape %v1419 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1421 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1422 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1423 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1424 = stablehlo.reduce(%v1420 init: %v1421) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1425 = stablehlo.broadcast_in_dim %v1424, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1426 = stablehlo.divide %v1425, %v1422 : tensor<32x256x7x7xf32>
    %v1427 = stablehlo.subtract %v1420, %v1426 : tensor<32x256x7x7xf32>
    %v1428 = stablehlo.multiply %v1427, %v1427 : tensor<32x256x7x7xf32>
    %v1429 = stablehlo.reduce(%v1428 init: %v1421) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1430 = stablehlo.broadcast_in_dim %v1429, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1431 = stablehlo.divide %v1430, %v1422 : tensor<32x256x7x7xf32>
    %v1432 = stablehlo.add %v1431, %v1423 : tensor<32x256x7x7xf32>
    %v1433 = stablehlo.rsqrt %v1432 : tensor<32x256x7x7xf32>
    %v1434 = stablehlo.multiply %v1427, %v1433 : tensor<32x256x7x7xf32>
    %v1435 = stablehlo.broadcast_in_dim %u14qg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1436 = stablehlo.broadcast_in_dim %u14qbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1437 = stablehlo.multiply %v1434, %v1435 : tensor<32x256x7x7xf32>
    %v1438 = stablehlo.add %v1437, %v1436 : tensor<32x256x7x7xf32>
    %v1439 = stablehlo.reshape %v1438 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1440 = stablehlo.reshape %v1439 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1441 = stablehlo.convolution(%v1440, %u14eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v1442 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1443 = stablehlo.add %v1441, %v1442 : tensor<32x1024x7x7xf32>
    %v1444 = stablehlo.reshape %v1443 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1445 = stablehlo.reshape %v1444 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1446 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1447 = stablehlo.constant dense<1568.0> : tensor<32x1024x7x7xf32>
    %v1448 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1449 = stablehlo.reduce(%v1445 init: %v1446) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1450 = stablehlo.broadcast_in_dim %v1449, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1451 = stablehlo.divide %v1450, %v1447 : tensor<32x1024x7x7xf32>
    %v1452 = stablehlo.subtract %v1445, %v1451 : tensor<32x1024x7x7xf32>
    %v1453 = stablehlo.multiply %v1452, %v1452 : tensor<32x1024x7x7xf32>
    %v1454 = stablehlo.reduce(%v1453 init: %v1446) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1455 = stablehlo.broadcast_in_dim %v1454, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1456 = stablehlo.divide %v1455, %v1447 : tensor<32x1024x7x7xf32>
    %v1457 = stablehlo.add %v1456, %v1448 : tensor<32x1024x7x7xf32>
    %v1458 = stablehlo.rsqrt %v1457 : tensor<32x1024x7x7xf32>
    %v1459 = stablehlo.multiply %v1452, %v1458 : tensor<32x1024x7x7xf32>
    %v1460 = stablehlo.broadcast_in_dim %u14eg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1461 = stablehlo.broadcast_in_dim %u14ebt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1462 = stablehlo.multiply %v1459, %v1460 : tensor<32x1024x7x7xf32>
    %v1463 = stablehlo.add %v1462, %v1461 : tensor<32x1024x7x7xf32>
    %v1464 = stablehlo.reshape %v1463 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1465 = stablehlo.reshape %v1464 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1466 = stablehlo.constant dense<0.0> : tensor<32x1024x7x7xf32>
    %v1467 = stablehlo.maximum %v1465, %v1466 : tensor<32x1024x7x7xf32>
    %v1468 = stablehlo.reshape %v1467 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1469 = stablehlo.reshape %v1468 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1470 = stablehlo.convolution(%v1469, %u14dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<32x1024x7x7xf32>, tensor<1024x1x5x5xf32>) -> tensor<32x1024x7x7xf32>
    %v1471 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1472 = stablehlo.add %v1470, %v1471 : tensor<32x1024x7x7xf32>
    %v1473 = stablehlo.reshape %v1472 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1474 = stablehlo.reshape %v1473 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1475 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1476 = stablehlo.constant dense<1568.0> : tensor<32x1024x7x7xf32>
    %v1477 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1478 = stablehlo.reduce(%v1474 init: %v1475) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1479 = stablehlo.broadcast_in_dim %v1478, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1480 = stablehlo.divide %v1479, %v1476 : tensor<32x1024x7x7xf32>
    %v1481 = stablehlo.subtract %v1474, %v1480 : tensor<32x1024x7x7xf32>
    %v1482 = stablehlo.multiply %v1481, %v1481 : tensor<32x1024x7x7xf32>
    %v1483 = stablehlo.reduce(%v1482 init: %v1475) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1484 = stablehlo.broadcast_in_dim %v1483, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1485 = stablehlo.divide %v1484, %v1476 : tensor<32x1024x7x7xf32>
    %v1486 = stablehlo.add %v1485, %v1477 : tensor<32x1024x7x7xf32>
    %v1487 = stablehlo.rsqrt %v1486 : tensor<32x1024x7x7xf32>
    %v1488 = stablehlo.multiply %v1481, %v1487 : tensor<32x1024x7x7xf32>
    %v1489 = stablehlo.broadcast_in_dim %u14dg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1490 = stablehlo.broadcast_in_dim %u14dbt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1491 = stablehlo.multiply %v1488, %v1489 : tensor<32x1024x7x7xf32>
    %v1492 = stablehlo.add %v1491, %v1490 : tensor<32x1024x7x7xf32>
    %v1493 = stablehlo.reshape %v1492 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1494 = stablehlo.reshape %v1493 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1495 = stablehlo.constant dense<0.0> : tensor<32x1024x7x7xf32>
    %v1496 = stablehlo.maximum %v1494, %v1495 : tensor<32x1024x7x7xf32>
    %v1497 = stablehlo.reshape %v1496 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1498 = stablehlo.reshape %v1497 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1499 = stablehlo.convolution(%v1498, %u14pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1500 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1501 = stablehlo.add %v1499, %v1500 : tensor<32x256x7x7xf32>
    %v1502 = stablehlo.reshape %v1501 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1503 = stablehlo.reshape %v1502 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1504 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1505 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1506 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1507 = stablehlo.reduce(%v1503 init: %v1504) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1508 = stablehlo.broadcast_in_dim %v1507, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1509 = stablehlo.divide %v1508, %v1505 : tensor<32x256x7x7xf32>
    %v1510 = stablehlo.subtract %v1503, %v1509 : tensor<32x256x7x7xf32>
    %v1511 = stablehlo.multiply %v1510, %v1510 : tensor<32x256x7x7xf32>
    %v1512 = stablehlo.reduce(%v1511 init: %v1504) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1513 = stablehlo.broadcast_in_dim %v1512, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1514 = stablehlo.divide %v1513, %v1505 : tensor<32x256x7x7xf32>
    %v1515 = stablehlo.add %v1514, %v1506 : tensor<32x256x7x7xf32>
    %v1516 = stablehlo.rsqrt %v1515 : tensor<32x256x7x7xf32>
    %v1517 = stablehlo.multiply %v1510, %v1516 : tensor<32x256x7x7xf32>
    %v1518 = stablehlo.broadcast_in_dim %u14pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1519 = stablehlo.broadcast_in_dim %u14pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1520 = stablehlo.multiply %v1517, %v1518 : tensor<32x256x7x7xf32>
    %v1521 = stablehlo.add %v1520, %v1519 : tensor<32x256x7x7xf32>
    %v1522 = stablehlo.reshape %v1521 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1523 = stablehlo.reshape %v1522 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1524 = stablehlo.reshape %v1414 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1525 = stablehlo.add %v1523, %v1524 : tensor<32x256x7x7xf32>
    %v1526 = stablehlo.reshape %v1525 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1527 = stablehlo.reshape %v1526 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1528 = stablehlo.convolution(%v1527, %u15eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v1529 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1530 = stablehlo.add %v1528, %v1529 : tensor<32x1024x7x7xf32>
    %v1531 = stablehlo.reshape %v1530 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1532 = stablehlo.reshape %v1531 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1533 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1534 = stablehlo.constant dense<1568.0> : tensor<32x1024x7x7xf32>
    %v1535 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1536 = stablehlo.reduce(%v1532 init: %v1533) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1537 = stablehlo.broadcast_in_dim %v1536, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1538 = stablehlo.divide %v1537, %v1534 : tensor<32x1024x7x7xf32>
    %v1539 = stablehlo.subtract %v1532, %v1538 : tensor<32x1024x7x7xf32>
    %v1540 = stablehlo.multiply %v1539, %v1539 : tensor<32x1024x7x7xf32>
    %v1541 = stablehlo.reduce(%v1540 init: %v1533) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1542 = stablehlo.broadcast_in_dim %v1541, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1543 = stablehlo.divide %v1542, %v1534 : tensor<32x1024x7x7xf32>
    %v1544 = stablehlo.add %v1543, %v1535 : tensor<32x1024x7x7xf32>
    %v1545 = stablehlo.rsqrt %v1544 : tensor<32x1024x7x7xf32>
    %v1546 = stablehlo.multiply %v1539, %v1545 : tensor<32x1024x7x7xf32>
    %v1547 = stablehlo.broadcast_in_dim %u15eg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1548 = stablehlo.broadcast_in_dim %u15ebt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1549 = stablehlo.multiply %v1546, %v1547 : tensor<32x1024x7x7xf32>
    %v1550 = stablehlo.add %v1549, %v1548 : tensor<32x1024x7x7xf32>
    %v1551 = stablehlo.reshape %v1550 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1552 = stablehlo.reshape %v1551 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1553 = stablehlo.constant dense<0.0> : tensor<32x1024x7x7xf32>
    %v1554 = stablehlo.maximum %v1552, %v1553 : tensor<32x1024x7x7xf32>
    %v1555 = stablehlo.reshape %v1554 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1556 = stablehlo.reshape %v1555 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1557 = stablehlo.convolution(%v1556, %u15pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1558 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1559 = stablehlo.add %v1557, %v1558 : tensor<32x256x7x7xf32>
    %v1560 = stablehlo.reshape %v1559 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1561 = stablehlo.reshape %v1560 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1562 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1563 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1564 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1565 = stablehlo.reduce(%v1561 init: %v1562) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1566 = stablehlo.broadcast_in_dim %v1565, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1567 = stablehlo.divide %v1566, %v1563 : tensor<32x256x7x7xf32>
    %v1568 = stablehlo.subtract %v1561, %v1567 : tensor<32x256x7x7xf32>
    %v1569 = stablehlo.multiply %v1568, %v1568 : tensor<32x256x7x7xf32>
    %v1570 = stablehlo.reduce(%v1569 init: %v1562) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1571 = stablehlo.broadcast_in_dim %v1570, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1572 = stablehlo.divide %v1571, %v1563 : tensor<32x256x7x7xf32>
    %v1573 = stablehlo.add %v1572, %v1564 : tensor<32x256x7x7xf32>
    %v1574 = stablehlo.rsqrt %v1573 : tensor<32x256x7x7xf32>
    %v1575 = stablehlo.multiply %v1568, %v1574 : tensor<32x256x7x7xf32>
    %v1576 = stablehlo.broadcast_in_dim %u15pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1577 = stablehlo.broadcast_in_dim %u15pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1578 = stablehlo.multiply %v1575, %v1576 : tensor<32x256x7x7xf32>
    %v1579 = stablehlo.add %v1578, %v1577 : tensor<32x256x7x7xf32>
    %v1580 = stablehlo.reshape %v1579 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1581 = stablehlo.reshape %v1580 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1582 = stablehlo.reshape %v1526 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1583 = stablehlo.add %v1581, %v1582 : tensor<32x256x7x7xf32>
    %v1584 = stablehlo.reshape %v1583 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1585 = stablehlo.reshape %v1584 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1586 = stablehlo.convolution(%v1585, %u16qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x7x7xf32>, tensor<256x1x3x3xf32>) -> tensor<32x256x7x7xf32>
    %v1587 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1588 = stablehlo.add %v1586, %v1587 : tensor<32x256x7x7xf32>
    %v1589 = stablehlo.reshape %v1588 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1590 = stablehlo.reshape %v1589 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1591 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1592 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1593 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1594 = stablehlo.reduce(%v1590 init: %v1591) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1595 = stablehlo.broadcast_in_dim %v1594, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1596 = stablehlo.divide %v1595, %v1592 : tensor<32x256x7x7xf32>
    %v1597 = stablehlo.subtract %v1590, %v1596 : tensor<32x256x7x7xf32>
    %v1598 = stablehlo.multiply %v1597, %v1597 : tensor<32x256x7x7xf32>
    %v1599 = stablehlo.reduce(%v1598 init: %v1591) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1600 = stablehlo.broadcast_in_dim %v1599, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1601 = stablehlo.divide %v1600, %v1592 : tensor<32x256x7x7xf32>
    %v1602 = stablehlo.add %v1601, %v1593 : tensor<32x256x7x7xf32>
    %v1603 = stablehlo.rsqrt %v1602 : tensor<32x256x7x7xf32>
    %v1604 = stablehlo.multiply %v1597, %v1603 : tensor<32x256x7x7xf32>
    %v1605 = stablehlo.broadcast_in_dim %u16qg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1606 = stablehlo.broadcast_in_dim %u16qbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1607 = stablehlo.multiply %v1604, %v1605 : tensor<32x256x7x7xf32>
    %v1608 = stablehlo.add %v1607, %v1606 : tensor<32x256x7x7xf32>
    %v1609 = stablehlo.reshape %v1608 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1610 = stablehlo.reshape %v1609 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1611 = stablehlo.convolution(%v1610, %u16eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v1612 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1613 = stablehlo.add %v1611, %v1612 : tensor<32x1024x7x7xf32>
    %v1614 = stablehlo.reshape %v1613 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1615 = stablehlo.reshape %v1614 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1616 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1617 = stablehlo.constant dense<1568.0> : tensor<32x1024x7x7xf32>
    %v1618 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1619 = stablehlo.reduce(%v1615 init: %v1616) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1620 = stablehlo.broadcast_in_dim %v1619, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1621 = stablehlo.divide %v1620, %v1617 : tensor<32x1024x7x7xf32>
    %v1622 = stablehlo.subtract %v1615, %v1621 : tensor<32x1024x7x7xf32>
    %v1623 = stablehlo.multiply %v1622, %v1622 : tensor<32x1024x7x7xf32>
    %v1624 = stablehlo.reduce(%v1623 init: %v1616) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1625 = stablehlo.broadcast_in_dim %v1624, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1626 = stablehlo.divide %v1625, %v1617 : tensor<32x1024x7x7xf32>
    %v1627 = stablehlo.add %v1626, %v1618 : tensor<32x1024x7x7xf32>
    %v1628 = stablehlo.rsqrt %v1627 : tensor<32x1024x7x7xf32>
    %v1629 = stablehlo.multiply %v1622, %v1628 : tensor<32x1024x7x7xf32>
    %v1630 = stablehlo.broadcast_in_dim %u16eg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1631 = stablehlo.broadcast_in_dim %u16ebt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1632 = stablehlo.multiply %v1629, %v1630 : tensor<32x1024x7x7xf32>
    %v1633 = stablehlo.add %v1632, %v1631 : tensor<32x1024x7x7xf32>
    %v1634 = stablehlo.reshape %v1633 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1635 = stablehlo.reshape %v1634 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1636 = stablehlo.constant dense<0.0> : tensor<32x1024x7x7xf32>
    %v1637 = stablehlo.maximum %v1635, %v1636 : tensor<32x1024x7x7xf32>
    %v1638 = stablehlo.reshape %v1637 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1639 = stablehlo.reshape %v1638 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1640 = stablehlo.convolution(%v1639, %u16pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1641 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1642 = stablehlo.add %v1640, %v1641 : tensor<32x256x7x7xf32>
    %v1643 = stablehlo.reshape %v1642 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1644 = stablehlo.reshape %v1643 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1645 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1646 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1647 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1648 = stablehlo.reduce(%v1644 init: %v1645) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1649 = stablehlo.broadcast_in_dim %v1648, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1650 = stablehlo.divide %v1649, %v1646 : tensor<32x256x7x7xf32>
    %v1651 = stablehlo.subtract %v1644, %v1650 : tensor<32x256x7x7xf32>
    %v1652 = stablehlo.multiply %v1651, %v1651 : tensor<32x256x7x7xf32>
    %v1653 = stablehlo.reduce(%v1652 init: %v1645) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1654 = stablehlo.broadcast_in_dim %v1653, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1655 = stablehlo.divide %v1654, %v1646 : tensor<32x256x7x7xf32>
    %v1656 = stablehlo.add %v1655, %v1647 : tensor<32x256x7x7xf32>
    %v1657 = stablehlo.rsqrt %v1656 : tensor<32x256x7x7xf32>
    %v1658 = stablehlo.multiply %v1651, %v1657 : tensor<32x256x7x7xf32>
    %v1659 = stablehlo.broadcast_in_dim %u16pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1660 = stablehlo.broadcast_in_dim %u16pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1661 = stablehlo.multiply %v1658, %v1659 : tensor<32x256x7x7xf32>
    %v1662 = stablehlo.add %v1661, %v1660 : tensor<32x256x7x7xf32>
    %v1663 = stablehlo.reshape %v1662 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1664 = stablehlo.reshape %v1663 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1665 = stablehlo.reshape %v1584 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1666 = stablehlo.add %v1664, %v1665 : tensor<32x256x7x7xf32>
    %v1667 = stablehlo.reshape %v1666 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1668 = stablehlo.reshape %v1667 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1669 = stablehlo.convolution(%v1668, %u17qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x7x7xf32>, tensor<256x1x3x3xf32>) -> tensor<32x256x7x7xf32>
    %v1670 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1671 = stablehlo.add %v1669, %v1670 : tensor<32x256x7x7xf32>
    %v1672 = stablehlo.reshape %v1671 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1673 = stablehlo.reshape %v1672 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1674 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1675 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1676 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1677 = stablehlo.reduce(%v1673 init: %v1674) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1678 = stablehlo.broadcast_in_dim %v1677, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1679 = stablehlo.divide %v1678, %v1675 : tensor<32x256x7x7xf32>
    %v1680 = stablehlo.subtract %v1673, %v1679 : tensor<32x256x7x7xf32>
    %v1681 = stablehlo.multiply %v1680, %v1680 : tensor<32x256x7x7xf32>
    %v1682 = stablehlo.reduce(%v1681 init: %v1674) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1683 = stablehlo.broadcast_in_dim %v1682, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1684 = stablehlo.divide %v1683, %v1675 : tensor<32x256x7x7xf32>
    %v1685 = stablehlo.add %v1684, %v1676 : tensor<32x256x7x7xf32>
    %v1686 = stablehlo.rsqrt %v1685 : tensor<32x256x7x7xf32>
    %v1687 = stablehlo.multiply %v1680, %v1686 : tensor<32x256x7x7xf32>
    %v1688 = stablehlo.broadcast_in_dim %u17qg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1689 = stablehlo.broadcast_in_dim %u17qbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1690 = stablehlo.multiply %v1687, %v1688 : tensor<32x256x7x7xf32>
    %v1691 = stablehlo.add %v1690, %v1689 : tensor<32x256x7x7xf32>
    %v1692 = stablehlo.reshape %v1691 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1693 = stablehlo.reshape %v1692 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1694 = stablehlo.convolution(%v1693, %u17eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<512x256x1x1xf32>) -> tensor<32x512x7x7xf32>
    %v1695 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1696 = stablehlo.add %v1694, %v1695 : tensor<32x512x7x7xf32>
    %v1697 = stablehlo.reshape %v1696 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1698 = stablehlo.reshape %v1697 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1699 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1700 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1701 = stablehlo.constant dense<1.0e-5> : tensor<32x512x7x7xf32>
    %v1702 = stablehlo.reduce(%v1698 init: %v1699) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1703 = stablehlo.broadcast_in_dim %v1702, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1704 = stablehlo.divide %v1703, %v1700 : tensor<32x512x7x7xf32>
    %v1705 = stablehlo.subtract %v1698, %v1704 : tensor<32x512x7x7xf32>
    %v1706 = stablehlo.multiply %v1705, %v1705 : tensor<32x512x7x7xf32>
    %v1707 = stablehlo.reduce(%v1706 init: %v1699) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1708 = stablehlo.broadcast_in_dim %v1707, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1709 = stablehlo.divide %v1708, %v1700 : tensor<32x512x7x7xf32>
    %v1710 = stablehlo.add %v1709, %v1701 : tensor<32x512x7x7xf32>
    %v1711 = stablehlo.rsqrt %v1710 : tensor<32x512x7x7xf32>
    %v1712 = stablehlo.multiply %v1705, %v1711 : tensor<32x512x7x7xf32>
    %v1713 = stablehlo.broadcast_in_dim %u17eg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1714 = stablehlo.broadcast_in_dim %u17ebt, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1715 = stablehlo.multiply %v1712, %v1713 : tensor<32x512x7x7xf32>
    %v1716 = stablehlo.add %v1715, %v1714 : tensor<32x512x7x7xf32>
    %v1717 = stablehlo.reshape %v1716 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1718 = stablehlo.reshape %v1717 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1719 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1720 = stablehlo.maximum %v1718, %v1719 : tensor<32x512x7x7xf32>
    %v1721 = stablehlo.reshape %v1720 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1722 = stablehlo.reshape %v1721 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1723 = stablehlo.convolution(%v1722, %u17dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x7x7xf32>, tensor<512x1x5x5xf32>) -> tensor<32x512x7x7xf32>
    %v1724 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1725 = stablehlo.add %v1723, %v1724 : tensor<32x512x7x7xf32>
    %v1726 = stablehlo.reshape %v1725 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1727 = stablehlo.reshape %v1726 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1728 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1729 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1730 = stablehlo.constant dense<1.0e-5> : tensor<32x512x7x7xf32>
    %v1731 = stablehlo.reduce(%v1727 init: %v1728) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1732 = stablehlo.broadcast_in_dim %v1731, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1733 = stablehlo.divide %v1732, %v1729 : tensor<32x512x7x7xf32>
    %v1734 = stablehlo.subtract %v1727, %v1733 : tensor<32x512x7x7xf32>
    %v1735 = stablehlo.multiply %v1734, %v1734 : tensor<32x512x7x7xf32>
    %v1736 = stablehlo.reduce(%v1735 init: %v1728) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1737 = stablehlo.broadcast_in_dim %v1736, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1738 = stablehlo.divide %v1737, %v1729 : tensor<32x512x7x7xf32>
    %v1739 = stablehlo.add %v1738, %v1730 : tensor<32x512x7x7xf32>
    %v1740 = stablehlo.rsqrt %v1739 : tensor<32x512x7x7xf32>
    %v1741 = stablehlo.multiply %v1734, %v1740 : tensor<32x512x7x7xf32>
    %v1742 = stablehlo.broadcast_in_dim %u17dg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1743 = stablehlo.broadcast_in_dim %u17dbt, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1744 = stablehlo.multiply %v1741, %v1742 : tensor<32x512x7x7xf32>
    %v1745 = stablehlo.add %v1744, %v1743 : tensor<32x512x7x7xf32>
    %v1746 = stablehlo.reshape %v1745 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1747 = stablehlo.reshape %v1746 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1748 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1749 = stablehlo.maximum %v1747, %v1748 : tensor<32x512x7x7xf32>
    %v1750 = stablehlo.reshape %v1749 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1751 = stablehlo.reshape %v1750 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1752 = stablehlo.convolution(%v1751, %u17pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<256x512x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1753 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1754 = stablehlo.add %v1752, %v1753 : tensor<32x256x7x7xf32>
    %v1755 = stablehlo.reshape %v1754 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1756 = stablehlo.reshape %v1755 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1757 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1758 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1759 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1760 = stablehlo.reduce(%v1756 init: %v1757) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1761 = stablehlo.broadcast_in_dim %v1760, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1762 = stablehlo.divide %v1761, %v1758 : tensor<32x256x7x7xf32>
    %v1763 = stablehlo.subtract %v1756, %v1762 : tensor<32x256x7x7xf32>
    %v1764 = stablehlo.multiply %v1763, %v1763 : tensor<32x256x7x7xf32>
    %v1765 = stablehlo.reduce(%v1764 init: %v1757) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1766 = stablehlo.broadcast_in_dim %v1765, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1767 = stablehlo.divide %v1766, %v1758 : tensor<32x256x7x7xf32>
    %v1768 = stablehlo.add %v1767, %v1759 : tensor<32x256x7x7xf32>
    %v1769 = stablehlo.rsqrt %v1768 : tensor<32x256x7x7xf32>
    %v1770 = stablehlo.multiply %v1763, %v1769 : tensor<32x256x7x7xf32>
    %v1771 = stablehlo.broadcast_in_dim %u17pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1772 = stablehlo.broadcast_in_dim %u17pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1773 = stablehlo.multiply %v1770, %v1771 : tensor<32x256x7x7xf32>
    %v1774 = stablehlo.add %v1773, %v1772 : tensor<32x256x7x7xf32>
    %v1775 = stablehlo.reshape %v1774 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1776 = stablehlo.reshape %v1775 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1777 = stablehlo.reshape %v1667 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1778 = stablehlo.add %v1776, %v1777 : tensor<32x256x7x7xf32>
    %v1779 = stablehlo.reshape %v1778 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1780 = stablehlo.reshape %v1779 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1781 = stablehlo.convolution(%v1780, %u18qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x7x7xf32>, tensor<256x1x5x5xf32>) -> tensor<32x256x7x7xf32>
    %v1782 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1783 = stablehlo.add %v1781, %v1782 : tensor<32x256x7x7xf32>
    %v1784 = stablehlo.reshape %v1783 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1785 = stablehlo.reshape %v1784 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1786 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1787 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1788 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1789 = stablehlo.reduce(%v1785 init: %v1786) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1790 = stablehlo.broadcast_in_dim %v1789, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1791 = stablehlo.divide %v1790, %v1787 : tensor<32x256x7x7xf32>
    %v1792 = stablehlo.subtract %v1785, %v1791 : tensor<32x256x7x7xf32>
    %v1793 = stablehlo.multiply %v1792, %v1792 : tensor<32x256x7x7xf32>
    %v1794 = stablehlo.reduce(%v1793 init: %v1786) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1795 = stablehlo.broadcast_in_dim %v1794, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1796 = stablehlo.divide %v1795, %v1787 : tensor<32x256x7x7xf32>
    %v1797 = stablehlo.add %v1796, %v1788 : tensor<32x256x7x7xf32>
    %v1798 = stablehlo.rsqrt %v1797 : tensor<32x256x7x7xf32>
    %v1799 = stablehlo.multiply %v1792, %v1798 : tensor<32x256x7x7xf32>
    %v1800 = stablehlo.broadcast_in_dim %u18qg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1801 = stablehlo.broadcast_in_dim %u18qbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1802 = stablehlo.multiply %v1799, %v1800 : tensor<32x256x7x7xf32>
    %v1803 = stablehlo.add %v1802, %v1801 : tensor<32x256x7x7xf32>
    %v1804 = stablehlo.reshape %v1803 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1805 = stablehlo.reshape %v1804 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1806 = stablehlo.convolution(%v1805, %u18eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v1807 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1808 = stablehlo.add %v1806, %v1807 : tensor<32x1024x7x7xf32>
    %v1809 = stablehlo.reshape %v1808 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1810 = stablehlo.reshape %v1809 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1811 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1812 = stablehlo.constant dense<1568.0> : tensor<32x1024x7x7xf32>
    %v1813 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1814 = stablehlo.reduce(%v1810 init: %v1811) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1815 = stablehlo.broadcast_in_dim %v1814, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1816 = stablehlo.divide %v1815, %v1812 : tensor<32x1024x7x7xf32>
    %v1817 = stablehlo.subtract %v1810, %v1816 : tensor<32x1024x7x7xf32>
    %v1818 = stablehlo.multiply %v1817, %v1817 : tensor<32x1024x7x7xf32>
    %v1819 = stablehlo.reduce(%v1818 init: %v1811) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1820 = stablehlo.broadcast_in_dim %v1819, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1821 = stablehlo.divide %v1820, %v1812 : tensor<32x1024x7x7xf32>
    %v1822 = stablehlo.add %v1821, %v1813 : tensor<32x1024x7x7xf32>
    %v1823 = stablehlo.rsqrt %v1822 : tensor<32x1024x7x7xf32>
    %v1824 = stablehlo.multiply %v1817, %v1823 : tensor<32x1024x7x7xf32>
    %v1825 = stablehlo.broadcast_in_dim %u18eg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1826 = stablehlo.broadcast_in_dim %u18ebt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1827 = stablehlo.multiply %v1824, %v1825 : tensor<32x1024x7x7xf32>
    %v1828 = stablehlo.add %v1827, %v1826 : tensor<32x1024x7x7xf32>
    %v1829 = stablehlo.reshape %v1828 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1830 = stablehlo.reshape %v1829 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1831 = stablehlo.constant dense<0.0> : tensor<32x1024x7x7xf32>
    %v1832 = stablehlo.maximum %v1830, %v1831 : tensor<32x1024x7x7xf32>
    %v1833 = stablehlo.reshape %v1832 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1834 = stablehlo.reshape %v1833 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1835 = stablehlo.convolution(%v1834, %u18dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<32x1024x7x7xf32>, tensor<1024x1x5x5xf32>) -> tensor<32x1024x7x7xf32>
    %v1836 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1837 = stablehlo.add %v1835, %v1836 : tensor<32x1024x7x7xf32>
    %v1838 = stablehlo.reshape %v1837 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1839 = stablehlo.reshape %v1838 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1840 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1841 = stablehlo.constant dense<1568.0> : tensor<32x1024x7x7xf32>
    %v1842 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1843 = stablehlo.reduce(%v1839 init: %v1840) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1844 = stablehlo.broadcast_in_dim %v1843, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1845 = stablehlo.divide %v1844, %v1841 : tensor<32x1024x7x7xf32>
    %v1846 = stablehlo.subtract %v1839, %v1845 : tensor<32x1024x7x7xf32>
    %v1847 = stablehlo.multiply %v1846, %v1846 : tensor<32x1024x7x7xf32>
    %v1848 = stablehlo.reduce(%v1847 init: %v1840) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1849 = stablehlo.broadcast_in_dim %v1848, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1850 = stablehlo.divide %v1849, %v1841 : tensor<32x1024x7x7xf32>
    %v1851 = stablehlo.add %v1850, %v1842 : tensor<32x1024x7x7xf32>
    %v1852 = stablehlo.rsqrt %v1851 : tensor<32x1024x7x7xf32>
    %v1853 = stablehlo.multiply %v1846, %v1852 : tensor<32x1024x7x7xf32>
    %v1854 = stablehlo.broadcast_in_dim %u18dg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1855 = stablehlo.broadcast_in_dim %u18dbt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1856 = stablehlo.multiply %v1853, %v1854 : tensor<32x1024x7x7xf32>
    %v1857 = stablehlo.add %v1856, %v1855 : tensor<32x1024x7x7xf32>
    %v1858 = stablehlo.reshape %v1857 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1859 = stablehlo.reshape %v1858 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1860 = stablehlo.constant dense<0.0> : tensor<32x1024x7x7xf32>
    %v1861 = stablehlo.maximum %v1859, %v1860 : tensor<32x1024x7x7xf32>
    %v1862 = stablehlo.reshape %v1861 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1863 = stablehlo.reshape %v1862 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1864 = stablehlo.convolution(%v1863, %u18pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1865 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1866 = stablehlo.add %v1864, %v1865 : tensor<32x256x7x7xf32>
    %v1867 = stablehlo.reshape %v1866 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1868 = stablehlo.reshape %v1867 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1869 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1870 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1871 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1872 = stablehlo.reduce(%v1868 init: %v1869) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1873 = stablehlo.broadcast_in_dim %v1872, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1874 = stablehlo.divide %v1873, %v1870 : tensor<32x256x7x7xf32>
    %v1875 = stablehlo.subtract %v1868, %v1874 : tensor<32x256x7x7xf32>
    %v1876 = stablehlo.multiply %v1875, %v1875 : tensor<32x256x7x7xf32>
    %v1877 = stablehlo.reduce(%v1876 init: %v1869) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1878 = stablehlo.broadcast_in_dim %v1877, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1879 = stablehlo.divide %v1878, %v1870 : tensor<32x256x7x7xf32>
    %v1880 = stablehlo.add %v1879, %v1871 : tensor<32x256x7x7xf32>
    %v1881 = stablehlo.rsqrt %v1880 : tensor<32x256x7x7xf32>
    %v1882 = stablehlo.multiply %v1875, %v1881 : tensor<32x256x7x7xf32>
    %v1883 = stablehlo.broadcast_in_dim %u18pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1884 = stablehlo.broadcast_in_dim %u18pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1885 = stablehlo.multiply %v1882, %v1883 : tensor<32x256x7x7xf32>
    %v1886 = stablehlo.add %v1885, %v1884 : tensor<32x256x7x7xf32>
    %v1887 = stablehlo.reshape %v1886 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1888 = stablehlo.reshape %v1887 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1889 = stablehlo.reshape %v1779 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1890 = stablehlo.add %v1888, %v1889 : tensor<32x256x7x7xf32>
    %v1891 = stablehlo.reshape %v1890 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1892 = stablehlo.reshape %v1891 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1893 = stablehlo.convolution(%v1892, %u19eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v1894 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1895 = stablehlo.add %v1893, %v1894 : tensor<32x1024x7x7xf32>
    %v1896 = stablehlo.reshape %v1895 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1897 = stablehlo.reshape %v1896 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1898 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1899 = stablehlo.constant dense<1568.0> : tensor<32x1024x7x7xf32>
    %v1900 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1901 = stablehlo.reduce(%v1897 init: %v1898) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1902 = stablehlo.broadcast_in_dim %v1901, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1903 = stablehlo.divide %v1902, %v1899 : tensor<32x1024x7x7xf32>
    %v1904 = stablehlo.subtract %v1897, %v1903 : tensor<32x1024x7x7xf32>
    %v1905 = stablehlo.multiply %v1904, %v1904 : tensor<32x1024x7x7xf32>
    %v1906 = stablehlo.reduce(%v1905 init: %v1898) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1907 = stablehlo.broadcast_in_dim %v1906, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1908 = stablehlo.divide %v1907, %v1899 : tensor<32x1024x7x7xf32>
    %v1909 = stablehlo.add %v1908, %v1900 : tensor<32x1024x7x7xf32>
    %v1910 = stablehlo.rsqrt %v1909 : tensor<32x1024x7x7xf32>
    %v1911 = stablehlo.multiply %v1904, %v1910 : tensor<32x1024x7x7xf32>
    %v1912 = stablehlo.broadcast_in_dim %u19eg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1913 = stablehlo.broadcast_in_dim %u19ebt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1914 = stablehlo.multiply %v1911, %v1912 : tensor<32x1024x7x7xf32>
    %v1915 = stablehlo.add %v1914, %v1913 : tensor<32x1024x7x7xf32>
    %v1916 = stablehlo.reshape %v1915 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1917 = stablehlo.reshape %v1916 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1918 = stablehlo.constant dense<0.0> : tensor<32x1024x7x7xf32>
    %v1919 = stablehlo.maximum %v1917, %v1918 : tensor<32x1024x7x7xf32>
    %v1920 = stablehlo.reshape %v1919 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1921 = stablehlo.reshape %v1920 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1922 = stablehlo.convolution(%v1921, %u19pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1923 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1924 = stablehlo.add %v1922, %v1923 : tensor<32x256x7x7xf32>
    %v1925 = stablehlo.reshape %v1924 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1926 = stablehlo.reshape %v1925 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1927 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1928 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1929 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1930 = stablehlo.reduce(%v1926 init: %v1927) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1931 = stablehlo.broadcast_in_dim %v1930, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1932 = stablehlo.divide %v1931, %v1928 : tensor<32x256x7x7xf32>
    %v1933 = stablehlo.subtract %v1926, %v1932 : tensor<32x256x7x7xf32>
    %v1934 = stablehlo.multiply %v1933, %v1933 : tensor<32x256x7x7xf32>
    %v1935 = stablehlo.reduce(%v1934 init: %v1927) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1936 = stablehlo.broadcast_in_dim %v1935, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1937 = stablehlo.divide %v1936, %v1928 : tensor<32x256x7x7xf32>
    %v1938 = stablehlo.add %v1937, %v1929 : tensor<32x256x7x7xf32>
    %v1939 = stablehlo.rsqrt %v1938 : tensor<32x256x7x7xf32>
    %v1940 = stablehlo.multiply %v1933, %v1939 : tensor<32x256x7x7xf32>
    %v1941 = stablehlo.broadcast_in_dim %u19pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1942 = stablehlo.broadcast_in_dim %u19pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1943 = stablehlo.multiply %v1940, %v1941 : tensor<32x256x7x7xf32>
    %v1944 = stablehlo.add %v1943, %v1942 : tensor<32x256x7x7xf32>
    %v1945 = stablehlo.reshape %v1944 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1946 = stablehlo.reshape %v1945 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1947 = stablehlo.reshape %v1891 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1948 = stablehlo.add %v1946, %v1947 : tensor<32x256x7x7xf32>
    %v1949 = stablehlo.reshape %v1948 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1950 = stablehlo.reshape %v1949 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1951 = stablehlo.convolution(%v1950, %u20eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v1952 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1953 = stablehlo.add %v1951, %v1952 : tensor<32x1024x7x7xf32>
    %v1954 = stablehlo.reshape %v1953 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1955 = stablehlo.reshape %v1954 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1956 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1957 = stablehlo.constant dense<1568.0> : tensor<32x1024x7x7xf32>
    %v1958 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1959 = stablehlo.reduce(%v1955 init: %v1956) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1960 = stablehlo.broadcast_in_dim %v1959, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1961 = stablehlo.divide %v1960, %v1957 : tensor<32x1024x7x7xf32>
    %v1962 = stablehlo.subtract %v1955, %v1961 : tensor<32x1024x7x7xf32>
    %v1963 = stablehlo.multiply %v1962, %v1962 : tensor<32x1024x7x7xf32>
    %v1964 = stablehlo.reduce(%v1963 init: %v1956) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1024x7x7xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1965 = stablehlo.broadcast_in_dim %v1964, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1966 = stablehlo.divide %v1965, %v1957 : tensor<32x1024x7x7xf32>
    %v1967 = stablehlo.add %v1966, %v1958 : tensor<32x1024x7x7xf32>
    %v1968 = stablehlo.rsqrt %v1967 : tensor<32x1024x7x7xf32>
    %v1969 = stablehlo.multiply %v1962, %v1968 : tensor<32x1024x7x7xf32>
    %v1970 = stablehlo.broadcast_in_dim %u20eg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1971 = stablehlo.broadcast_in_dim %u20ebt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1972 = stablehlo.multiply %v1969, %v1970 : tensor<32x1024x7x7xf32>
    %v1973 = stablehlo.add %v1972, %v1971 : tensor<32x1024x7x7xf32>
    %v1974 = stablehlo.reshape %v1973 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1975 = stablehlo.reshape %v1974 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1976 = stablehlo.constant dense<0.0> : tensor<32x1024x7x7xf32>
    %v1977 = stablehlo.maximum %v1975, %v1976 : tensor<32x1024x7x7xf32>
    %v1978 = stablehlo.reshape %v1977 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1979 = stablehlo.reshape %v1978 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1980 = stablehlo.convolution(%v1979, %u20pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1981 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1982 = stablehlo.add %v1980, %v1981 : tensor<32x256x7x7xf32>
    %v1983 = stablehlo.reshape %v1982 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1984 = stablehlo.reshape %v1983 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1985 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1986 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v1987 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1988 = stablehlo.reduce(%v1984 init: %v1985) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1989 = stablehlo.broadcast_in_dim %v1988, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1990 = stablehlo.divide %v1989, %v1986 : tensor<32x256x7x7xf32>
    %v1991 = stablehlo.subtract %v1984, %v1990 : tensor<32x256x7x7xf32>
    %v1992 = stablehlo.multiply %v1991, %v1991 : tensor<32x256x7x7xf32>
    %v1993 = stablehlo.reduce(%v1992 init: %v1985) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v1994 = stablehlo.broadcast_in_dim %v1993, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1995 = stablehlo.divide %v1994, %v1986 : tensor<32x256x7x7xf32>
    %v1996 = stablehlo.add %v1995, %v1987 : tensor<32x256x7x7xf32>
    %v1997 = stablehlo.rsqrt %v1996 : tensor<32x256x7x7xf32>
    %v1998 = stablehlo.multiply %v1991, %v1997 : tensor<32x256x7x7xf32>
    %v1999 = stablehlo.broadcast_in_dim %u20pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v2000 = stablehlo.broadcast_in_dim %u20pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v2001 = stablehlo.multiply %v1998, %v1999 : tensor<32x256x7x7xf32>
    %v2002 = stablehlo.add %v2001, %v2000 : tensor<32x256x7x7xf32>
    %v2003 = stablehlo.reshape %v2002 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v2004 = stablehlo.reshape %v2003 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v2005 = stablehlo.reshape %v1949 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v2006 = stablehlo.add %v2004, %v2005 : tensor<32x256x7x7xf32>
    %v2007 = stablehlo.reshape %v2006 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v2008 = stablehlo.reshape %v2007 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v2009 = stablehlo.convolution(%v2008, %u21qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x7x7xf32>, tensor<256x1x5x5xf32>) -> tensor<32x256x7x7xf32>
    %v2010 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v2011 = stablehlo.add %v2009, %v2010 : tensor<32x256x7x7xf32>
    %v2012 = stablehlo.reshape %v2011 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v2013 = stablehlo.reshape %v2012 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v2014 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2015 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v2016 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v2017 = stablehlo.reduce(%v2013 init: %v2014) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v2018 = stablehlo.broadcast_in_dim %v2017, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v2019 = stablehlo.divide %v2018, %v2015 : tensor<32x256x7x7xf32>
    %v2020 = stablehlo.subtract %v2013, %v2019 : tensor<32x256x7x7xf32>
    %v2021 = stablehlo.multiply %v2020, %v2020 : tensor<32x256x7x7xf32>
    %v2022 = stablehlo.reduce(%v2021 init: %v2014) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v2023 = stablehlo.broadcast_in_dim %v2022, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v2024 = stablehlo.divide %v2023, %v2015 : tensor<32x256x7x7xf32>
    %v2025 = stablehlo.add %v2024, %v2016 : tensor<32x256x7x7xf32>
    %v2026 = stablehlo.rsqrt %v2025 : tensor<32x256x7x7xf32>
    %v2027 = stablehlo.multiply %v2020, %v2026 : tensor<32x256x7x7xf32>
    %v2028 = stablehlo.broadcast_in_dim %u21qg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v2029 = stablehlo.broadcast_in_dim %u21qbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v2030 = stablehlo.multiply %v2027, %v2028 : tensor<32x256x7x7xf32>
    %v2031 = stablehlo.add %v2030, %v2029 : tensor<32x256x7x7xf32>
    %v2032 = stablehlo.reshape %v2031 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v2033 = stablehlo.reshape %v2032 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v2034 = stablehlo.convolution(%v2033, %u21eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<512x256x1x1xf32>) -> tensor<32x512x7x7xf32>
    %v2035 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v2036 = stablehlo.add %v2034, %v2035 : tensor<32x512x7x7xf32>
    %v2037 = stablehlo.reshape %v2036 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v2038 = stablehlo.reshape %v2037 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v2039 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2040 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v2041 = stablehlo.constant dense<1.0e-5> : tensor<32x512x7x7xf32>
    %v2042 = stablehlo.reduce(%v2038 init: %v2039) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2043 = stablehlo.broadcast_in_dim %v2042, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v2044 = stablehlo.divide %v2043, %v2040 : tensor<32x512x7x7xf32>
    %v2045 = stablehlo.subtract %v2038, %v2044 : tensor<32x512x7x7xf32>
    %v2046 = stablehlo.multiply %v2045, %v2045 : tensor<32x512x7x7xf32>
    %v2047 = stablehlo.reduce(%v2046 init: %v2039) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2048 = stablehlo.broadcast_in_dim %v2047, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v2049 = stablehlo.divide %v2048, %v2040 : tensor<32x512x7x7xf32>
    %v2050 = stablehlo.add %v2049, %v2041 : tensor<32x512x7x7xf32>
    %v2051 = stablehlo.rsqrt %v2050 : tensor<32x512x7x7xf32>
    %v2052 = stablehlo.multiply %v2045, %v2051 : tensor<32x512x7x7xf32>
    %v2053 = stablehlo.broadcast_in_dim %u21eg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v2054 = stablehlo.broadcast_in_dim %u21ebt, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v2055 = stablehlo.multiply %v2052, %v2053 : tensor<32x512x7x7xf32>
    %v2056 = stablehlo.add %v2055, %v2054 : tensor<32x512x7x7xf32>
    %v2057 = stablehlo.reshape %v2056 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v2058 = stablehlo.reshape %v2057 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v2059 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v2060 = stablehlo.maximum %v2058, %v2059 : tensor<32x512x7x7xf32>
    %v2061 = stablehlo.reshape %v2060 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v2062 = stablehlo.reshape %v2061 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v2063 = stablehlo.convolution(%v2062, %u21pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<256x512x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v2064 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v2065 = stablehlo.add %v2063, %v2064 : tensor<32x256x7x7xf32>
    %v2066 = stablehlo.reshape %v2065 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v2067 = stablehlo.reshape %v2066 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v2068 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2069 = stablehlo.constant dense<1568.0> : tensor<32x256x7x7xf32>
    %v2070 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v2071 = stablehlo.reduce(%v2067 init: %v2068) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v2072 = stablehlo.broadcast_in_dim %v2071, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v2073 = stablehlo.divide %v2072, %v2069 : tensor<32x256x7x7xf32>
    %v2074 = stablehlo.subtract %v2067, %v2073 : tensor<32x256x7x7xf32>
    %v2075 = stablehlo.multiply %v2074, %v2074 : tensor<32x256x7x7xf32>
    %v2076 = stablehlo.reduce(%v2075 init: %v2068) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x7x7xf32>, tensor<f32>) -> tensor<256xf32>
    %v2077 = stablehlo.broadcast_in_dim %v2076, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v2078 = stablehlo.divide %v2077, %v2069 : tensor<32x256x7x7xf32>
    %v2079 = stablehlo.add %v2078, %v2070 : tensor<32x256x7x7xf32>
    %v2080 = stablehlo.rsqrt %v2079 : tensor<32x256x7x7xf32>
    %v2081 = stablehlo.multiply %v2074, %v2080 : tensor<32x256x7x7xf32>
    %v2082 = stablehlo.broadcast_in_dim %u21pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v2083 = stablehlo.broadcast_in_dim %u21pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v2084 = stablehlo.multiply %v2081, %v2082 : tensor<32x256x7x7xf32>
    %v2085 = stablehlo.add %v2084, %v2083 : tensor<32x256x7x7xf32>
    %v2086 = stablehlo.reshape %v2085 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v2087 = stablehlo.reshape %v2086 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v2088 = stablehlo.reshape %v2007 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v2089 = stablehlo.add %v2087, %v2088 : tensor<32x256x7x7xf32>
    %v2090 = stablehlo.reshape %v2089 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v2091 = stablehlo.reshape %v2090 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v2092 = stablehlo.convolution(%v2091, %h1W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<960x256x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v2093 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2094 = stablehlo.add %v2092, %v2093 : tensor<32x960x7x7xf32>
    %v2095 = stablehlo.reshape %v2094 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v2096 = stablehlo.reshape %v2095 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2097 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2098 = stablehlo.constant dense<1568.0> : tensor<32x960x7x7xf32>
    %v2099 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v2100 = stablehlo.reduce(%v2096 init: %v2097) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2101 = stablehlo.broadcast_in_dim %v2100, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2102 = stablehlo.divide %v2101, %v2098 : tensor<32x960x7x7xf32>
    %v2103 = stablehlo.subtract %v2096, %v2102 : tensor<32x960x7x7xf32>
    %v2104 = stablehlo.multiply %v2103, %v2103 : tensor<32x960x7x7xf32>
    %v2105 = stablehlo.reduce(%v2104 init: %v2097) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<960xf32>
    %v2106 = stablehlo.broadcast_in_dim %v2105, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2107 = stablehlo.divide %v2106, %v2098 : tensor<32x960x7x7xf32>
    %v2108 = stablehlo.add %v2107, %v2099 : tensor<32x960x7x7xf32>
    %v2109 = stablehlo.rsqrt %v2108 : tensor<32x960x7x7xf32>
    %v2110 = stablehlo.multiply %v2103, %v2109 : tensor<32x960x7x7xf32>
    %v2111 = stablehlo.broadcast_in_dim %h1g, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2112 = stablehlo.broadcast_in_dim %h1bt, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v2113 = stablehlo.multiply %v2110, %v2111 : tensor<32x960x7x7xf32>
    %v2114 = stablehlo.add %v2113, %v2112 : tensor<32x960x7x7xf32>
    %v2115 = stablehlo.reshape %v2114 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v2116 = stablehlo.reshape %v2115 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2117 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v2118 = stablehlo.maximum %v2116, %v2117 : tensor<32x960x7x7xf32>
    %v2119 = stablehlo.reshape %v2118 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v2120 = stablehlo.reshape %v2119 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v2121 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2122 = stablehlo.reduce(%v2120 init: %v2121) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x960x7x7xf32>, tensor<f32>) -> tensor<32x960xf32>
    %v2123 = stablehlo.constant dense<49.0> : tensor<32x960xf32>
    %v2124 = stablehlo.divide %v2122, %v2123 : tensor<32x960xf32>
    %v2125 = stablehlo.reshape %v2124 : (tensor<32x960xf32>) -> tensor<32x960x1x1xf32>
    %v2126 = stablehlo.convolution(%v2125, %hW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x1x1xf32>, tensor<1280x960x1x1xf32>) -> tensor<32x1280x1x1xf32>
    %v2127 = stablehlo.broadcast_in_dim %zb1280, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x1x1xf32>
    %v2128 = stablehlo.add %v2126, %v2127 : tensor<32x1280x1x1xf32>
    %v2129 = stablehlo.reshape %v2128 : (tensor<32x1280x1x1xf32>) -> tensor<32x1280xf32>
    %v2130 = stablehlo.reshape %v2129 : (tensor<32x1280xf32>) -> tensor<32x1280x1x1xf32>
    %v2131 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2132 = stablehlo.constant dense<32.0> : tensor<32x1280x1x1xf32>
    %v2133 = stablehlo.constant dense<1.0e-5> : tensor<32x1280x1x1xf32>
    %v2134 = stablehlo.reduce(%v2130 init: %v2131) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x1x1xf32>, tensor<f32>) -> tensor<1280xf32>
    %v2135 = stablehlo.broadcast_in_dim %v2134, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x1x1xf32>
    %v2136 = stablehlo.divide %v2135, %v2132 : tensor<32x1280x1x1xf32>
    %v2137 = stablehlo.subtract %v2130, %v2136 : tensor<32x1280x1x1xf32>
    %v2138 = stablehlo.multiply %v2137, %v2137 : tensor<32x1280x1x1xf32>
    %v2139 = stablehlo.reduce(%v2138 init: %v2131) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x1x1xf32>, tensor<f32>) -> tensor<1280xf32>
    %v2140 = stablehlo.broadcast_in_dim %v2139, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x1x1xf32>
    %v2141 = stablehlo.divide %v2140, %v2132 : tensor<32x1280x1x1xf32>
    %v2142 = stablehlo.add %v2141, %v2133 : tensor<32x1280x1x1xf32>
    %v2143 = stablehlo.rsqrt %v2142 : tensor<32x1280x1x1xf32>
    %v2144 = stablehlo.multiply %v2137, %v2143 : tensor<32x1280x1x1xf32>
    %v2145 = stablehlo.broadcast_in_dim %hg, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x1x1xf32>
    %v2146 = stablehlo.broadcast_in_dim %hbt, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x1x1xf32>
    %v2147 = stablehlo.multiply %v2144, %v2145 : tensor<32x1280x1x1xf32>
    %v2148 = stablehlo.add %v2147, %v2146 : tensor<32x1280x1x1xf32>
    %v2149 = stablehlo.reshape %v2148 : (tensor<32x1280x1x1xf32>) -> tensor<32x1280xf32>
    %v2150 = stablehlo.reshape %v2149 : (tensor<32x1280xf32>) -> tensor<32x1280x1x1xf32>
    %v2151 = stablehlo.constant dense<0.0> : tensor<32x1280x1x1xf32>
    %v2152 = stablehlo.maximum %v2150, %v2151 : tensor<32x1280x1x1xf32>
    %v2153 = stablehlo.reshape %v2152 : (tensor<32x1280x1x1xf32>) -> tensor<32x1280xf32>
    %v2154 = stablehlo.dot_general %v2153, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1280xf32>, tensor<1280x10xf32>) -> tensor<32x10xf32>
    %v2155 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v2156 = stablehlo.add %v2154, %v2155 : tensor<32x10xf32>
    return %v2156 : tensor<32x10xf32>
  }
}
