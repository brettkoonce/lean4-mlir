module @m {
  func.func @mnv4_fwd_eval(%x: tensor<32x150528xf32>, %sW: tensor<32x3x3x3xf32>, %sg: tensor<32xf32>, %sbt: tensor<32xf32>, %f0cW: tensor<128x32x3x3xf32>, %f0cg: tensor<128xf32>, %f0cbt: tensor<128xf32>, %f0pW: tensor<48x128x1x1xf32>, %f0pg: tensor<48xf32>, %f0pbt: tensor<48xf32>, %u1qW: tensor<48x1x3x3xf32>, %u1qg: tensor<48xf32>, %u1qbt: tensor<48xf32>, %u1eW: tensor<192x48x1x1xf32>, %u1eg: tensor<192xf32>, %u1ebt: tensor<192xf32>, %u1dW: tensor<192x1x5x5xf32>, %u1dg: tensor<192xf32>, %u1dbt: tensor<192xf32>, %u1pW: tensor<80x192x1x1xf32>, %u1pg: tensor<80xf32>, %u1pbt: tensor<80xf32>, %u2qW: tensor<80x1x3x3xf32>, %u2qg: tensor<80xf32>, %u2qbt: tensor<80xf32>, %u2eW: tensor<160x80x1x1xf32>, %u2eg: tensor<160xf32>, %u2ebt: tensor<160xf32>, %u2dW: tensor<160x1x3x3xf32>, %u2dg: tensor<160xf32>, %u2dbt: tensor<160xf32>, %u2pW: tensor<80x160x1x1xf32>, %u2pg: tensor<80xf32>, %u2pbt: tensor<80xf32>, %u3qW: tensor<80x1x3x3xf32>, %u3qg: tensor<80xf32>, %u3qbt: tensor<80xf32>, %u3eW: tensor<480x80x1x1xf32>, %u3eg: tensor<480xf32>, %u3ebt: tensor<480xf32>, %u3dW: tensor<480x1x5x5xf32>, %u3dg: tensor<480xf32>, %u3dbt: tensor<480xf32>, %u3pW: tensor<160x480x1x1xf32>, %u3pg: tensor<160xf32>, %u3pbt: tensor<160xf32>, %u4qW: tensor<160x1x3x3xf32>, %u4qg: tensor<160xf32>, %u4qbt: tensor<160xf32>, %u4eW: tensor<640x160x1x1xf32>, %u4eg: tensor<640xf32>, %u4ebt: tensor<640xf32>, %u4dW: tensor<640x1x3x3xf32>, %u4dg: tensor<640xf32>, %u4dbt: tensor<640xf32>, %u4pW: tensor<160x640x1x1xf32>, %u4pg: tensor<160xf32>, %u4pbt: tensor<160xf32>, %u5qW: tensor<160x1x3x3xf32>, %u5qg: tensor<160xf32>, %u5qbt: tensor<160xf32>, %u5eW: tensor<640x160x1x1xf32>, %u5eg: tensor<640xf32>, %u5ebt: tensor<640xf32>, %u5dW: tensor<640x1x3x3xf32>, %u5dg: tensor<640xf32>, %u5dbt: tensor<640xf32>, %u5pW: tensor<160x640x1x1xf32>, %u5pg: tensor<160xf32>, %u5pbt: tensor<160xf32>, %u6qW: tensor<160x1x3x3xf32>, %u6qg: tensor<160xf32>, %u6qbt: tensor<160xf32>, %u6eW: tensor<640x160x1x1xf32>, %u6eg: tensor<640xf32>, %u6ebt: tensor<640xf32>, %u6dW: tensor<640x1x5x5xf32>, %u6dg: tensor<640xf32>, %u6dbt: tensor<640xf32>, %u6pW: tensor<160x640x1x1xf32>, %u6pg: tensor<160xf32>, %u6pbt: tensor<160xf32>, %u7qW: tensor<160x1x3x3xf32>, %u7qg: tensor<160xf32>, %u7qbt: tensor<160xf32>, %u7eW: tensor<640x160x1x1xf32>, %u7eg: tensor<640xf32>, %u7ebt: tensor<640xf32>, %u7dW: tensor<640x1x3x3xf32>, %u7dg: tensor<640xf32>, %u7dbt: tensor<640xf32>, %u7pW: tensor<160x640x1x1xf32>, %u7pg: tensor<160xf32>, %u7pbt: tensor<160xf32>, %u8qW: tensor<160x1x3x3xf32>, %u8qg: tensor<160xf32>, %u8qbt: tensor<160xf32>, %u8eW: tensor<640x160x1x1xf32>, %u8eg: tensor<640xf32>, %u8ebt: tensor<640xf32>, %u8pW: tensor<160x640x1x1xf32>, %u8pg: tensor<160xf32>, %u8pbt: tensor<160xf32>, %u9eW: tensor<320x160x1x1xf32>, %u9eg: tensor<320xf32>, %u9ebt: tensor<320xf32>, %u9pW: tensor<160x320x1x1xf32>, %u9pg: tensor<160xf32>, %u9pbt: tensor<160xf32>, %u10qW: tensor<160x1x3x3xf32>, %u10qg: tensor<160xf32>, %u10qbt: tensor<160xf32>, %u10eW: tensor<640x160x1x1xf32>, %u10eg: tensor<640xf32>, %u10ebt: tensor<640xf32>, %u10pW: tensor<160x640x1x1xf32>, %u10pg: tensor<160xf32>, %u10pbt: tensor<160xf32>, %u11qW: tensor<160x1x5x5xf32>, %u11qg: tensor<160xf32>, %u11qbt: tensor<160xf32>, %u11eW: tensor<960x160x1x1xf32>, %u11eg: tensor<960xf32>, %u11ebt: tensor<960xf32>, %u11dW: tensor<960x1x5x5xf32>, %u11dg: tensor<960xf32>, %u11dbt: tensor<960xf32>, %u11pW: tensor<256x960x1x1xf32>, %u11pg: tensor<256xf32>, %u11pbt: tensor<256xf32>, %u12qW: tensor<256x1x5x5xf32>, %u12qg: tensor<256xf32>, %u12qbt: tensor<256xf32>, %u12eW: tensor<1024x256x1x1xf32>, %u12eg: tensor<1024xf32>, %u12ebt: tensor<1024xf32>, %u12dW: tensor<1024x1x5x5xf32>, %u12dg: tensor<1024xf32>, %u12dbt: tensor<1024xf32>, %u12pW: tensor<256x1024x1x1xf32>, %u12pg: tensor<256xf32>, %u12pbt: tensor<256xf32>, %u13qW: tensor<256x1x3x3xf32>, %u13qg: tensor<256xf32>, %u13qbt: tensor<256xf32>, %u13eW: tensor<1024x256x1x1xf32>, %u13eg: tensor<1024xf32>, %u13ebt: tensor<1024xf32>, %u13dW: tensor<1024x1x5x5xf32>, %u13dg: tensor<1024xf32>, %u13dbt: tensor<1024xf32>, %u13pW: tensor<256x1024x1x1xf32>, %u13pg: tensor<256xf32>, %u13pbt: tensor<256xf32>, %u14qW: tensor<256x1x3x3xf32>, %u14qg: tensor<256xf32>, %u14qbt: tensor<256xf32>, %u14eW: tensor<1024x256x1x1xf32>, %u14eg: tensor<1024xf32>, %u14ebt: tensor<1024xf32>, %u14dW: tensor<1024x1x5x5xf32>, %u14dg: tensor<1024xf32>, %u14dbt: tensor<1024xf32>, %u14pW: tensor<256x1024x1x1xf32>, %u14pg: tensor<256xf32>, %u14pbt: tensor<256xf32>, %u15eW: tensor<1024x256x1x1xf32>, %u15eg: tensor<1024xf32>, %u15ebt: tensor<1024xf32>, %u15pW: tensor<256x1024x1x1xf32>, %u15pg: tensor<256xf32>, %u15pbt: tensor<256xf32>, %u16qW: tensor<256x1x3x3xf32>, %u16qg: tensor<256xf32>, %u16qbt: tensor<256xf32>, %u16eW: tensor<1024x256x1x1xf32>, %u16eg: tensor<1024xf32>, %u16ebt: tensor<1024xf32>, %u16pW: tensor<256x1024x1x1xf32>, %u16pg: tensor<256xf32>, %u16pbt: tensor<256xf32>, %u17qW: tensor<256x1x3x3xf32>, %u17qg: tensor<256xf32>, %u17qbt: tensor<256xf32>, %u17eW: tensor<512x256x1x1xf32>, %u17eg: tensor<512xf32>, %u17ebt: tensor<512xf32>, %u17dW: tensor<512x1x5x5xf32>, %u17dg: tensor<512xf32>, %u17dbt: tensor<512xf32>, %u17pW: tensor<256x512x1x1xf32>, %u17pg: tensor<256xf32>, %u17pbt: tensor<256xf32>, %u18qW: tensor<256x1x5x5xf32>, %u18qg: tensor<256xf32>, %u18qbt: tensor<256xf32>, %u18eW: tensor<1024x256x1x1xf32>, %u18eg: tensor<1024xf32>, %u18ebt: tensor<1024xf32>, %u18dW: tensor<1024x1x5x5xf32>, %u18dg: tensor<1024xf32>, %u18dbt: tensor<1024xf32>, %u18pW: tensor<256x1024x1x1xf32>, %u18pg: tensor<256xf32>, %u18pbt: tensor<256xf32>, %u19eW: tensor<1024x256x1x1xf32>, %u19eg: tensor<1024xf32>, %u19ebt: tensor<1024xf32>, %u19pW: tensor<256x1024x1x1xf32>, %u19pg: tensor<256xf32>, %u19pbt: tensor<256xf32>, %u20eW: tensor<1024x256x1x1xf32>, %u20eg: tensor<1024xf32>, %u20ebt: tensor<1024xf32>, %u20pW: tensor<256x1024x1x1xf32>, %u20pg: tensor<256xf32>, %u20pbt: tensor<256xf32>, %u21qW: tensor<256x1x5x5xf32>, %u21qg: tensor<256xf32>, %u21qbt: tensor<256xf32>, %u21eW: tensor<512x256x1x1xf32>, %u21eg: tensor<512xf32>, %u21ebt: tensor<512xf32>, %u21pW: tensor<256x512x1x1xf32>, %u21pg: tensor<256xf32>, %u21pbt: tensor<256xf32>, %h1W: tensor<960x256x1x1xf32>, %h1g: tensor<960xf32>, %h1bt: tensor<960xf32>, %hW: tensor<1280x960x1x1xf32>, %hg: tensor<1280xf32>, %hbt: tensor<1280xf32>, %Wd: tensor<1280x10xf32>, %bd: tensor<10xf32>, %stnmu: tensor<32xf32>, %stnvar: tensor<32xf32>, %f0cnmu: tensor<128xf32>, %f0cnvar: tensor<128xf32>, %f0pnmu: tensor<48xf32>, %f0pnvar: tensor<48xf32>, %u1qnmu: tensor<48xf32>, %u1qnvar: tensor<48xf32>, %u1enmu: tensor<192xf32>, %u1envar: tensor<192xf32>, %u1dnmu: tensor<192xf32>, %u1dnvar: tensor<192xf32>, %u1pnmu: tensor<80xf32>, %u1pnvar: tensor<80xf32>, %u2qnmu: tensor<80xf32>, %u2qnvar: tensor<80xf32>, %u2enmu: tensor<160xf32>, %u2envar: tensor<160xf32>, %u2dnmu: tensor<160xf32>, %u2dnvar: tensor<160xf32>, %u2pnmu: tensor<80xf32>, %u2pnvar: tensor<80xf32>, %u3qnmu: tensor<80xf32>, %u3qnvar: tensor<80xf32>, %u3enmu: tensor<480xf32>, %u3envar: tensor<480xf32>, %u3dnmu: tensor<480xf32>, %u3dnvar: tensor<480xf32>, %u3pnmu: tensor<160xf32>, %u3pnvar: tensor<160xf32>, %u4qnmu: tensor<160xf32>, %u4qnvar: tensor<160xf32>, %u4enmu: tensor<640xf32>, %u4envar: tensor<640xf32>, %u4dnmu: tensor<640xf32>, %u4dnvar: tensor<640xf32>, %u4pnmu: tensor<160xf32>, %u4pnvar: tensor<160xf32>, %u5qnmu: tensor<160xf32>, %u5qnvar: tensor<160xf32>, %u5enmu: tensor<640xf32>, %u5envar: tensor<640xf32>, %u5dnmu: tensor<640xf32>, %u5dnvar: tensor<640xf32>, %u5pnmu: tensor<160xf32>, %u5pnvar: tensor<160xf32>, %u6qnmu: tensor<160xf32>, %u6qnvar: tensor<160xf32>, %u6enmu: tensor<640xf32>, %u6envar: tensor<640xf32>, %u6dnmu: tensor<640xf32>, %u6dnvar: tensor<640xf32>, %u6pnmu: tensor<160xf32>, %u6pnvar: tensor<160xf32>, %u7qnmu: tensor<160xf32>, %u7qnvar: tensor<160xf32>, %u7enmu: tensor<640xf32>, %u7envar: tensor<640xf32>, %u7dnmu: tensor<640xf32>, %u7dnvar: tensor<640xf32>, %u7pnmu: tensor<160xf32>, %u7pnvar: tensor<160xf32>, %u8qnmu: tensor<160xf32>, %u8qnvar: tensor<160xf32>, %u8enmu: tensor<640xf32>, %u8envar: tensor<640xf32>, %u8pnmu: tensor<160xf32>, %u8pnvar: tensor<160xf32>, %u9enmu: tensor<320xf32>, %u9envar: tensor<320xf32>, %u9pnmu: tensor<160xf32>, %u9pnvar: tensor<160xf32>, %u10qnmu: tensor<160xf32>, %u10qnvar: tensor<160xf32>, %u10enmu: tensor<640xf32>, %u10envar: tensor<640xf32>, %u10pnmu: tensor<160xf32>, %u10pnvar: tensor<160xf32>, %u11qnmu: tensor<160xf32>, %u11qnvar: tensor<160xf32>, %u11enmu: tensor<960xf32>, %u11envar: tensor<960xf32>, %u11dnmu: tensor<960xf32>, %u11dnvar: tensor<960xf32>, %u11pnmu: tensor<256xf32>, %u11pnvar: tensor<256xf32>, %u12qnmu: tensor<256xf32>, %u12qnvar: tensor<256xf32>, %u12enmu: tensor<1024xf32>, %u12envar: tensor<1024xf32>, %u12dnmu: tensor<1024xf32>, %u12dnvar: tensor<1024xf32>, %u12pnmu: tensor<256xf32>, %u12pnvar: tensor<256xf32>, %u13qnmu: tensor<256xf32>, %u13qnvar: tensor<256xf32>, %u13enmu: tensor<1024xf32>, %u13envar: tensor<1024xf32>, %u13dnmu: tensor<1024xf32>, %u13dnvar: tensor<1024xf32>, %u13pnmu: tensor<256xf32>, %u13pnvar: tensor<256xf32>, %u14qnmu: tensor<256xf32>, %u14qnvar: tensor<256xf32>, %u14enmu: tensor<1024xf32>, %u14envar: tensor<1024xf32>, %u14dnmu: tensor<1024xf32>, %u14dnvar: tensor<1024xf32>, %u14pnmu: tensor<256xf32>, %u14pnvar: tensor<256xf32>, %u15enmu: tensor<1024xf32>, %u15envar: tensor<1024xf32>, %u15pnmu: tensor<256xf32>, %u15pnvar: tensor<256xf32>, %u16qnmu: tensor<256xf32>, %u16qnvar: tensor<256xf32>, %u16enmu: tensor<1024xf32>, %u16envar: tensor<1024xf32>, %u16pnmu: tensor<256xf32>, %u16pnvar: tensor<256xf32>, %u17qnmu: tensor<256xf32>, %u17qnvar: tensor<256xf32>, %u17enmu: tensor<512xf32>, %u17envar: tensor<512xf32>, %u17dnmu: tensor<512xf32>, %u17dnvar: tensor<512xf32>, %u17pnmu: tensor<256xf32>, %u17pnvar: tensor<256xf32>, %u18qnmu: tensor<256xf32>, %u18qnvar: tensor<256xf32>, %u18enmu: tensor<1024xf32>, %u18envar: tensor<1024xf32>, %u18dnmu: tensor<1024xf32>, %u18dnvar: tensor<1024xf32>, %u18pnmu: tensor<256xf32>, %u18pnvar: tensor<256xf32>, %u19enmu: tensor<1024xf32>, %u19envar: tensor<1024xf32>, %u19pnmu: tensor<256xf32>, %u19pnvar: tensor<256xf32>, %u20enmu: tensor<1024xf32>, %u20envar: tensor<1024xf32>, %u20pnmu: tensor<256xf32>, %u20pnvar: tensor<256xf32>, %u21qnmu: tensor<256xf32>, %u21qnvar: tensor<256xf32>, %u21enmu: tensor<512xf32>, %u21envar: tensor<512xf32>, %u21pnmu: tensor<256xf32>, %u21pnvar: tensor<256xf32>, %h1nmu: tensor<960xf32>, %h1nvar: tensor<960xf32>, %hnmu: tensor<1280xf32>, %hnvar: tensor<1280xf32>) -> tensor<32x10xf32> {
    // ── MobileNetV4-Conv-M eval forward (running-stats BN): every line is pretty(AST node) ──
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
    %v18 = stablehlo.reshape %v17 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v19 = stablehlo.constant dense<0.0> : tensor<32x32x112x112xf32>
    %v20 = stablehlo.maximum %v18, %v19 : tensor<32x32x112x112xf32>
    %v21 = stablehlo.reshape %v20 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v22 = stablehlo.reshape %v21 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v23 = stablehlo.convolution(%v22, %f0cW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<128x32x3x3xf32>) -> tensor<32x128x56x56xf32>
    %v24 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v25 = stablehlo.add %v23, %v24 : tensor<32x128x56x56xf32>
    %v26 = stablehlo.reshape %v25 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v27 = stablehlo.reshape %v26 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v28 = stablehlo.broadcast_in_dim %f0cnmu, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v29 = stablehlo.subtract %v27, %v28 : tensor<32x128x56x56xf32>
    %v30 = stablehlo.broadcast_in_dim %f0cnvar, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v31 = stablehlo.constant dense<1.0e-5> : tensor<32x128x56x56xf32>
    %v32 = stablehlo.add %v30, %v31 : tensor<32x128x56x56xf32>
    %v33 = stablehlo.rsqrt %v32 : tensor<32x128x56x56xf32>
    %v34 = stablehlo.multiply %v29, %v33 : tensor<32x128x56x56xf32>
    %v35 = stablehlo.broadcast_in_dim %f0cg, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v36 = stablehlo.broadcast_in_dim %f0cbt, dims = [1] : (tensor<128xf32>) -> tensor<32x128x56x56xf32>
    %v37 = stablehlo.multiply %v34, %v35 : tensor<32x128x56x56xf32>
    %v38 = stablehlo.add %v37, %v36 : tensor<32x128x56x56xf32>
    %v39 = stablehlo.reshape %v38 : (tensor<32x128x56x56xf32>) -> tensor<32x401408xf32>
    %v40 = stablehlo.reshape %v39 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v41 = stablehlo.logistic %v40 : tensor<32x32x112x112xf32>
    %v42 = stablehlo.multiply %v40, %v41 : tensor<32x32x112x112xf32>
    %v43 = stablehlo.reshape %v42 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v44 = stablehlo.reshape %v43 : (tensor<32x401408xf32>) -> tensor<32x128x56x56xf32>
    %v45 = stablehlo.convolution(%v44, %f0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x56x56xf32>, tensor<48x128x1x1xf32>) -> tensor<32x48x56x56xf32>
    %v46 = stablehlo.broadcast_in_dim %zb48, dims = [1] : (tensor<48xf32>) -> tensor<32x48x56x56xf32>
    %v47 = stablehlo.add %v45, %v46 : tensor<32x48x56x56xf32>
    %v48 = stablehlo.reshape %v47 : (tensor<32x48x56x56xf32>) -> tensor<32x150528xf32>
    %v49 = stablehlo.reshape %v48 : (tensor<32x150528xf32>) -> tensor<32x48x56x56xf32>
    %v50 = stablehlo.broadcast_in_dim %f0pnmu, dims = [1] : (tensor<48xf32>) -> tensor<32x48x56x56xf32>
    %v51 = stablehlo.subtract %v49, %v50 : tensor<32x48x56x56xf32>
    %v52 = stablehlo.broadcast_in_dim %f0pnvar, dims = [1] : (tensor<48xf32>) -> tensor<32x48x56x56xf32>
    %v53 = stablehlo.constant dense<1.0e-5> : tensor<32x48x56x56xf32>
    %v54 = stablehlo.add %v52, %v53 : tensor<32x48x56x56xf32>
    %v55 = stablehlo.rsqrt %v54 : tensor<32x48x56x56xf32>
    %v56 = stablehlo.multiply %v51, %v55 : tensor<32x48x56x56xf32>
    %v57 = stablehlo.broadcast_in_dim %f0pg, dims = [1] : (tensor<48xf32>) -> tensor<32x48x56x56xf32>
    %v58 = stablehlo.broadcast_in_dim %f0pbt, dims = [1] : (tensor<48xf32>) -> tensor<32x48x56x56xf32>
    %v59 = stablehlo.multiply %v56, %v57 : tensor<32x48x56x56xf32>
    %v60 = stablehlo.add %v59, %v58 : tensor<32x48x56x56xf32>
    %v61 = stablehlo.reshape %v60 : (tensor<32x48x56x56xf32>) -> tensor<32x150528xf32>
    %v62 = stablehlo.reshape %v61 : (tensor<32x150528xf32>) -> tensor<32x48x56x56xf32>
    %v63 = stablehlo.convolution(%v62, %u1qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 48 : i64} : (tensor<32x48x56x56xf32>, tensor<48x1x3x3xf32>) -> tensor<32x48x28x28xf32>
    %v64 = stablehlo.broadcast_in_dim %zb48, dims = [1] : (tensor<48xf32>) -> tensor<32x48x28x28xf32>
    %v65 = stablehlo.add %v63, %v64 : tensor<32x48x28x28xf32>
    %v66 = stablehlo.reshape %v65 : (tensor<32x48x28x28xf32>) -> tensor<32x37632xf32>
    %v67 = stablehlo.reshape %v66 : (tensor<32x37632xf32>) -> tensor<32x48x28x28xf32>
    %v68 = stablehlo.broadcast_in_dim %u1qnmu, dims = [1] : (tensor<48xf32>) -> tensor<32x48x28x28xf32>
    %v69 = stablehlo.subtract %v67, %v68 : tensor<32x48x28x28xf32>
    %v70 = stablehlo.broadcast_in_dim %u1qnvar, dims = [1] : (tensor<48xf32>) -> tensor<32x48x28x28xf32>
    %v71 = stablehlo.constant dense<1.0e-5> : tensor<32x48x28x28xf32>
    %v72 = stablehlo.add %v70, %v71 : tensor<32x48x28x28xf32>
    %v73 = stablehlo.rsqrt %v72 : tensor<32x48x28x28xf32>
    %v74 = stablehlo.multiply %v69, %v73 : tensor<32x48x28x28xf32>
    %v75 = stablehlo.broadcast_in_dim %u1qg, dims = [1] : (tensor<48xf32>) -> tensor<32x48x28x28xf32>
    %v76 = stablehlo.broadcast_in_dim %u1qbt, dims = [1] : (tensor<48xf32>) -> tensor<32x48x28x28xf32>
    %v77 = stablehlo.multiply %v74, %v75 : tensor<32x48x28x28xf32>
    %v78 = stablehlo.add %v77, %v76 : tensor<32x48x28x28xf32>
    %v79 = stablehlo.reshape %v78 : (tensor<32x48x28x28xf32>) -> tensor<32x37632xf32>
    %v80 = stablehlo.reshape %v79 : (tensor<32x37632xf32>) -> tensor<32x48x28x28xf32>
    %v81 = stablehlo.constant dense<0.0> : tensor<32x48x28x28xf32>
    %v82 = stablehlo.maximum %v80, %v81 : tensor<32x48x28x28xf32>
    %v83 = stablehlo.reshape %v82 : (tensor<32x48x28x28xf32>) -> tensor<32x37632xf32>
    %v84 = stablehlo.reshape %v83 : (tensor<32x37632xf32>) -> tensor<32x48x28x28xf32>
    %v85 = stablehlo.convolution(%v84, %u1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x48x28x28xf32>, tensor<192x48x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v86 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v87 = stablehlo.add %v85, %v86 : tensor<32x192x28x28xf32>
    %v88 = stablehlo.reshape %v87 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v89 = stablehlo.reshape %v88 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v90 = stablehlo.broadcast_in_dim %u1enmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v91 = stablehlo.subtract %v89, %v90 : tensor<32x192x28x28xf32>
    %v92 = stablehlo.broadcast_in_dim %u1envar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v93 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v94 = stablehlo.add %v92, %v93 : tensor<32x192x28x28xf32>
    %v95 = stablehlo.rsqrt %v94 : tensor<32x192x28x28xf32>
    %v96 = stablehlo.multiply %v91, %v95 : tensor<32x192x28x28xf32>
    %v97 = stablehlo.broadcast_in_dim %u1eg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v98 = stablehlo.broadcast_in_dim %u1ebt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v99 = stablehlo.multiply %v96, %v97 : tensor<32x192x28x28xf32>
    %v100 = stablehlo.add %v99, %v98 : tensor<32x192x28x28xf32>
    %v101 = stablehlo.reshape %v100 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v102 = stablehlo.reshape %v101 : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v103 = stablehlo.constant dense<0.0> : tensor<32x3x224x224xf32>
    %v104 = stablehlo.maximum %v102, %v103 : tensor<32x3x224x224xf32>
    %v105 = stablehlo.reshape %v104 : (tensor<32x3x224x224xf32>) -> tensor<32x150528xf32>
    %v106 = stablehlo.reshape %v105 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v107 = stablehlo.convolution(%v106, %u1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x5x5xf32>) -> tensor<32x192x28x28xf32>
    %v108 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v109 = stablehlo.add %v107, %v108 : tensor<32x192x28x28xf32>
    %v110 = stablehlo.reshape %v109 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v111 = stablehlo.reshape %v110 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v112 = stablehlo.broadcast_in_dim %u1dnmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v113 = stablehlo.subtract %v111, %v112 : tensor<32x192x28x28xf32>
    %v114 = stablehlo.broadcast_in_dim %u1dnvar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v115 = stablehlo.constant dense<1.0e-5> : tensor<32x192x28x28xf32>
    %v116 = stablehlo.add %v114, %v115 : tensor<32x192x28x28xf32>
    %v117 = stablehlo.rsqrt %v116 : tensor<32x192x28x28xf32>
    %v118 = stablehlo.multiply %v113, %v117 : tensor<32x192x28x28xf32>
    %v119 = stablehlo.broadcast_in_dim %u1dg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v120 = stablehlo.broadcast_in_dim %u1dbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v121 = stablehlo.multiply %v118, %v119 : tensor<32x192x28x28xf32>
    %v122 = stablehlo.add %v121, %v120 : tensor<32x192x28x28xf32>
    %v123 = stablehlo.reshape %v122 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v124 = stablehlo.reshape %v123 : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v125 = stablehlo.constant dense<0.0> : tensor<32x3x224x224xf32>
    %v126 = stablehlo.maximum %v124, %v125 : tensor<32x3x224x224xf32>
    %v127 = stablehlo.reshape %v126 : (tensor<32x3x224x224xf32>) -> tensor<32x150528xf32>
    %v128 = stablehlo.reshape %v127 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v129 = stablehlo.convolution(%v128, %u1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<80x192x1x1xf32>) -> tensor<32x80x28x28xf32>
    %v130 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v131 = stablehlo.add %v129, %v130 : tensor<32x80x28x28xf32>
    %v132 = stablehlo.reshape %v131 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v133 = stablehlo.reshape %v132 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v134 = stablehlo.broadcast_in_dim %u1pnmu, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v135 = stablehlo.subtract %v133, %v134 : tensor<32x80x28x28xf32>
    %v136 = stablehlo.broadcast_in_dim %u1pnvar, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v137 = stablehlo.constant dense<1.0e-5> : tensor<32x80x28x28xf32>
    %v138 = stablehlo.add %v136, %v137 : tensor<32x80x28x28xf32>
    %v139 = stablehlo.rsqrt %v138 : tensor<32x80x28x28xf32>
    %v140 = stablehlo.multiply %v135, %v139 : tensor<32x80x28x28xf32>
    %v141 = stablehlo.broadcast_in_dim %u1pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v142 = stablehlo.broadcast_in_dim %u1pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v143 = stablehlo.multiply %v140, %v141 : tensor<32x80x28x28xf32>
    %v144 = stablehlo.add %v143, %v142 : tensor<32x80x28x28xf32>
    %v145 = stablehlo.reshape %v144 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v146 = stablehlo.reshape %v145 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v147 = stablehlo.convolution(%v146, %u2qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 80 : i64} : (tensor<32x80x28x28xf32>, tensor<80x1x3x3xf32>) -> tensor<32x80x28x28xf32>
    %v148 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v149 = stablehlo.add %v147, %v148 : tensor<32x80x28x28xf32>
    %v150 = stablehlo.reshape %v149 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v151 = stablehlo.reshape %v150 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v152 = stablehlo.broadcast_in_dim %u2qnmu, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v153 = stablehlo.subtract %v151, %v152 : tensor<32x80x28x28xf32>
    %v154 = stablehlo.broadcast_in_dim %u2qnvar, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v155 = stablehlo.constant dense<1.0e-5> : tensor<32x80x28x28xf32>
    %v156 = stablehlo.add %v154, %v155 : tensor<32x80x28x28xf32>
    %v157 = stablehlo.rsqrt %v156 : tensor<32x80x28x28xf32>
    %v158 = stablehlo.multiply %v153, %v157 : tensor<32x80x28x28xf32>
    %v159 = stablehlo.broadcast_in_dim %u2qg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v160 = stablehlo.broadcast_in_dim %u2qbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v161 = stablehlo.multiply %v158, %v159 : tensor<32x80x28x28xf32>
    %v162 = stablehlo.add %v161, %v160 : tensor<32x80x28x28xf32>
    %v163 = stablehlo.reshape %v162 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v164 = stablehlo.reshape %v163 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v165 = stablehlo.constant dense<0.0> : tensor<32x80x28x28xf32>
    %v166 = stablehlo.maximum %v164, %v165 : tensor<32x80x28x28xf32>
    %v167 = stablehlo.reshape %v166 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v168 = stablehlo.reshape %v167 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v169 = stablehlo.convolution(%v168, %u2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x28x28xf32>, tensor<160x80x1x1xf32>) -> tensor<32x160x28x28xf32>
    %v170 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v171 = stablehlo.add %v169, %v170 : tensor<32x160x28x28xf32>
    %v172 = stablehlo.reshape %v171 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v173 = stablehlo.reshape %v172 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v174 = stablehlo.broadcast_in_dim %u2enmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v175 = stablehlo.subtract %v173, %v174 : tensor<32x160x28x28xf32>
    %v176 = stablehlo.broadcast_in_dim %u2envar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v177 = stablehlo.constant dense<1.0e-5> : tensor<32x160x28x28xf32>
    %v178 = stablehlo.add %v176, %v177 : tensor<32x160x28x28xf32>
    %v179 = stablehlo.rsqrt %v178 : tensor<32x160x28x28xf32>
    %v180 = stablehlo.multiply %v175, %v179 : tensor<32x160x28x28xf32>
    %v181 = stablehlo.broadcast_in_dim %u2eg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v182 = stablehlo.broadcast_in_dim %u2ebt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v183 = stablehlo.multiply %v180, %v181 : tensor<32x160x28x28xf32>
    %v184 = stablehlo.add %v183, %v182 : tensor<32x160x28x28xf32>
    %v185 = stablehlo.reshape %v184 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v186 = stablehlo.reshape %v185 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v187 = stablehlo.constant dense<0.0> : tensor<32x160x28x28xf32>
    %v188 = stablehlo.maximum %v186, %v187 : tensor<32x160x28x28xf32>
    %v189 = stablehlo.reshape %v188 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v190 = stablehlo.reshape %v189 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v191 = stablehlo.convolution(%v190, %u2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x28x28xf32>, tensor<160x1x3x3xf32>) -> tensor<32x160x28x28xf32>
    %v192 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v193 = stablehlo.add %v191, %v192 : tensor<32x160x28x28xf32>
    %v194 = stablehlo.reshape %v193 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v195 = stablehlo.reshape %v194 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v196 = stablehlo.broadcast_in_dim %u2dnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v197 = stablehlo.subtract %v195, %v196 : tensor<32x160x28x28xf32>
    %v198 = stablehlo.broadcast_in_dim %u2dnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v199 = stablehlo.constant dense<1.0e-5> : tensor<32x160x28x28xf32>
    %v200 = stablehlo.add %v198, %v199 : tensor<32x160x28x28xf32>
    %v201 = stablehlo.rsqrt %v200 : tensor<32x160x28x28xf32>
    %v202 = stablehlo.multiply %v197, %v201 : tensor<32x160x28x28xf32>
    %v203 = stablehlo.broadcast_in_dim %u2dg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v204 = stablehlo.broadcast_in_dim %u2dbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x28x28xf32>
    %v205 = stablehlo.multiply %v202, %v203 : tensor<32x160x28x28xf32>
    %v206 = stablehlo.add %v205, %v204 : tensor<32x160x28x28xf32>
    %v207 = stablehlo.reshape %v206 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v208 = stablehlo.reshape %v207 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v209 = stablehlo.constant dense<0.0> : tensor<32x160x28x28xf32>
    %v210 = stablehlo.maximum %v208, %v209 : tensor<32x160x28x28xf32>
    %v211 = stablehlo.reshape %v210 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v212 = stablehlo.reshape %v211 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v213 = stablehlo.convolution(%v212, %u2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x28x28xf32>, tensor<80x160x1x1xf32>) -> tensor<32x80x28x28xf32>
    %v214 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v215 = stablehlo.add %v213, %v214 : tensor<32x80x28x28xf32>
    %v216 = stablehlo.reshape %v215 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v217 = stablehlo.reshape %v216 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v218 = stablehlo.broadcast_in_dim %u2pnmu, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v219 = stablehlo.subtract %v217, %v218 : tensor<32x80x28x28xf32>
    %v220 = stablehlo.broadcast_in_dim %u2pnvar, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v221 = stablehlo.constant dense<1.0e-5> : tensor<32x80x28x28xf32>
    %v222 = stablehlo.add %v220, %v221 : tensor<32x80x28x28xf32>
    %v223 = stablehlo.rsqrt %v222 : tensor<32x80x28x28xf32>
    %v224 = stablehlo.multiply %v219, %v223 : tensor<32x80x28x28xf32>
    %v225 = stablehlo.broadcast_in_dim %u2pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v226 = stablehlo.broadcast_in_dim %u2pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x28x28xf32>
    %v227 = stablehlo.multiply %v224, %v225 : tensor<32x80x28x28xf32>
    %v228 = stablehlo.add %v227, %v226 : tensor<32x80x28x28xf32>
    %v229 = stablehlo.reshape %v228 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v230 = stablehlo.reshape %v229 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v231 = stablehlo.reshape %v145 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v232 = stablehlo.add %v230, %v231 : tensor<32x80x28x28xf32>
    %v233 = stablehlo.reshape %v232 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v234 = stablehlo.reshape %v233 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v235 = stablehlo.convolution(%v234, %u3qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 80 : i64} : (tensor<32x80x28x28xf32>, tensor<80x1x3x3xf32>) -> tensor<32x80x14x14xf32>
    %v236 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v237 = stablehlo.add %v235, %v236 : tensor<32x80x14x14xf32>
    %v238 = stablehlo.reshape %v237 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v239 = stablehlo.reshape %v238 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v240 = stablehlo.broadcast_in_dim %u3qnmu, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v241 = stablehlo.subtract %v239, %v240 : tensor<32x80x14x14xf32>
    %v242 = stablehlo.broadcast_in_dim %u3qnvar, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v243 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v244 = stablehlo.add %v242, %v243 : tensor<32x80x14x14xf32>
    %v245 = stablehlo.rsqrt %v244 : tensor<32x80x14x14xf32>
    %v246 = stablehlo.multiply %v241, %v245 : tensor<32x80x14x14xf32>
    %v247 = stablehlo.broadcast_in_dim %u3qg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v248 = stablehlo.broadcast_in_dim %u3qbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v249 = stablehlo.multiply %v246, %v247 : tensor<32x80x14x14xf32>
    %v250 = stablehlo.add %v249, %v248 : tensor<32x80x14x14xf32>
    %v251 = stablehlo.reshape %v250 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v252 = stablehlo.reshape %v251 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v253 = stablehlo.constant dense<0.0> : tensor<32x80x14x14xf32>
    %v254 = stablehlo.maximum %v252, %v253 : tensor<32x80x14x14xf32>
    %v255 = stablehlo.reshape %v254 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v256 = stablehlo.reshape %v255 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v257 = stablehlo.convolution(%v256, %u3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v258 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v259 = stablehlo.add %v257, %v258 : tensor<32x480x14x14xf32>
    %v260 = stablehlo.reshape %v259 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v261 = stablehlo.reshape %v260 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v262 = stablehlo.broadcast_in_dim %u3enmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v263 = stablehlo.subtract %v261, %v262 : tensor<32x480x14x14xf32>
    %v264 = stablehlo.broadcast_in_dim %u3envar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v265 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v266 = stablehlo.add %v264, %v265 : tensor<32x480x14x14xf32>
    %v267 = stablehlo.rsqrt %v266 : tensor<32x480x14x14xf32>
    %v268 = stablehlo.multiply %v263, %v267 : tensor<32x480x14x14xf32>
    %v269 = stablehlo.broadcast_in_dim %u3eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v270 = stablehlo.broadcast_in_dim %u3ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v271 = stablehlo.multiply %v268, %v269 : tensor<32x480x14x14xf32>
    %v272 = stablehlo.add %v271, %v270 : tensor<32x480x14x14xf32>
    %v273 = stablehlo.reshape %v272 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v274 = stablehlo.reshape %v273 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v275 = stablehlo.constant dense<0.0> : tensor<32x480x14x14xf32>
    %v276 = stablehlo.maximum %v274, %v275 : tensor<32x480x14x14xf32>
    %v277 = stablehlo.reshape %v276 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v278 = stablehlo.reshape %v277 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v279 = stablehlo.convolution(%v278, %u3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x5x5xf32>) -> tensor<32x480x14x14xf32>
    %v280 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v281 = stablehlo.add %v279, %v280 : tensor<32x480x14x14xf32>
    %v282 = stablehlo.reshape %v281 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v283 = stablehlo.reshape %v282 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v284 = stablehlo.broadcast_in_dim %u3dnmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v285 = stablehlo.subtract %v283, %v284 : tensor<32x480x14x14xf32>
    %v286 = stablehlo.broadcast_in_dim %u3dnvar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v287 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v288 = stablehlo.add %v286, %v287 : tensor<32x480x14x14xf32>
    %v289 = stablehlo.rsqrt %v288 : tensor<32x480x14x14xf32>
    %v290 = stablehlo.multiply %v285, %v289 : tensor<32x480x14x14xf32>
    %v291 = stablehlo.broadcast_in_dim %u3dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v292 = stablehlo.broadcast_in_dim %u3dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v293 = stablehlo.multiply %v290, %v291 : tensor<32x480x14x14xf32>
    %v294 = stablehlo.add %v293, %v292 : tensor<32x480x14x14xf32>
    %v295 = stablehlo.reshape %v294 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v296 = stablehlo.reshape %v295 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v297 = stablehlo.constant dense<0.0> : tensor<32x480x14x14xf32>
    %v298 = stablehlo.maximum %v296, %v297 : tensor<32x480x14x14xf32>
    %v299 = stablehlo.reshape %v298 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v300 = stablehlo.reshape %v299 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v301 = stablehlo.convolution(%v300, %u3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<160x480x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v302 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v303 = stablehlo.add %v301, %v302 : tensor<32x160x14x14xf32>
    %v304 = stablehlo.reshape %v303 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v305 = stablehlo.reshape %v304 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v306 = stablehlo.broadcast_in_dim %u3pnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v307 = stablehlo.subtract %v305, %v306 : tensor<32x160x14x14xf32>
    %v308 = stablehlo.broadcast_in_dim %u3pnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v309 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v310 = stablehlo.add %v308, %v309 : tensor<32x160x14x14xf32>
    %v311 = stablehlo.rsqrt %v310 : tensor<32x160x14x14xf32>
    %v312 = stablehlo.multiply %v307, %v311 : tensor<32x160x14x14xf32>
    %v313 = stablehlo.broadcast_in_dim %u3pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v314 = stablehlo.broadcast_in_dim %u3pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v315 = stablehlo.multiply %v312, %v313 : tensor<32x160x14x14xf32>
    %v316 = stablehlo.add %v315, %v314 : tensor<32x160x14x14xf32>
    %v317 = stablehlo.reshape %v316 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v318 = stablehlo.reshape %v317 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v319 = stablehlo.convolution(%v318, %u4qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<32x160x14x14xf32>
    %v320 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v321 = stablehlo.add %v319, %v320 : tensor<32x160x14x14xf32>
    %v322 = stablehlo.reshape %v321 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v323 = stablehlo.reshape %v322 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v324 = stablehlo.broadcast_in_dim %u4qnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v325 = stablehlo.subtract %v323, %v324 : tensor<32x160x14x14xf32>
    %v326 = stablehlo.broadcast_in_dim %u4qnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v327 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v328 = stablehlo.add %v326, %v327 : tensor<32x160x14x14xf32>
    %v329 = stablehlo.rsqrt %v328 : tensor<32x160x14x14xf32>
    %v330 = stablehlo.multiply %v325, %v329 : tensor<32x160x14x14xf32>
    %v331 = stablehlo.broadcast_in_dim %u4qg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v332 = stablehlo.broadcast_in_dim %u4qbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v333 = stablehlo.multiply %v330, %v331 : tensor<32x160x14x14xf32>
    %v334 = stablehlo.add %v333, %v332 : tensor<32x160x14x14xf32>
    %v335 = stablehlo.reshape %v334 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v336 = stablehlo.reshape %v335 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v337 = stablehlo.constant dense<0.0> : tensor<32x160x14x14xf32>
    %v338 = stablehlo.maximum %v336, %v337 : tensor<32x160x14x14xf32>
    %v339 = stablehlo.reshape %v338 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v340 = stablehlo.reshape %v339 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v341 = stablehlo.convolution(%v340, %u4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<32x640x14x14xf32>
    %v342 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v343 = stablehlo.add %v341, %v342 : tensor<32x640x14x14xf32>
    %v344 = stablehlo.reshape %v343 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v345 = stablehlo.reshape %v344 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v346 = stablehlo.broadcast_in_dim %u4enmu, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v347 = stablehlo.subtract %v345, %v346 : tensor<32x640x14x14xf32>
    %v348 = stablehlo.broadcast_in_dim %u4envar, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v349 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v350 = stablehlo.add %v348, %v349 : tensor<32x640x14x14xf32>
    %v351 = stablehlo.rsqrt %v350 : tensor<32x640x14x14xf32>
    %v352 = stablehlo.multiply %v347, %v351 : tensor<32x640x14x14xf32>
    %v353 = stablehlo.broadcast_in_dim %u4eg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v354 = stablehlo.broadcast_in_dim %u4ebt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v355 = stablehlo.multiply %v352, %v353 : tensor<32x640x14x14xf32>
    %v356 = stablehlo.add %v355, %v354 : tensor<32x640x14x14xf32>
    %v357 = stablehlo.reshape %v356 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v358 = stablehlo.reshape %v357 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v359 = stablehlo.constant dense<0.0> : tensor<32x160x28x28xf32>
    %v360 = stablehlo.maximum %v358, %v359 : tensor<32x160x28x28xf32>
    %v361 = stablehlo.reshape %v360 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v362 = stablehlo.reshape %v361 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v363 = stablehlo.convolution(%v362, %u4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 640 : i64} : (tensor<32x640x14x14xf32>, tensor<640x1x3x3xf32>) -> tensor<32x640x14x14xf32>
    %v364 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v365 = stablehlo.add %v363, %v364 : tensor<32x640x14x14xf32>
    %v366 = stablehlo.reshape %v365 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v367 = stablehlo.reshape %v366 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v368 = stablehlo.broadcast_in_dim %u4dnmu, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v369 = stablehlo.subtract %v367, %v368 : tensor<32x640x14x14xf32>
    %v370 = stablehlo.broadcast_in_dim %u4dnvar, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v371 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v372 = stablehlo.add %v370, %v371 : tensor<32x640x14x14xf32>
    %v373 = stablehlo.rsqrt %v372 : tensor<32x640x14x14xf32>
    %v374 = stablehlo.multiply %v369, %v373 : tensor<32x640x14x14xf32>
    %v375 = stablehlo.broadcast_in_dim %u4dg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v376 = stablehlo.broadcast_in_dim %u4dbt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v377 = stablehlo.multiply %v374, %v375 : tensor<32x640x14x14xf32>
    %v378 = stablehlo.add %v377, %v376 : tensor<32x640x14x14xf32>
    %v379 = stablehlo.reshape %v378 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v380 = stablehlo.reshape %v379 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v381 = stablehlo.constant dense<0.0> : tensor<32x160x28x28xf32>
    %v382 = stablehlo.maximum %v380, %v381 : tensor<32x160x28x28xf32>
    %v383 = stablehlo.reshape %v382 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v384 = stablehlo.reshape %v383 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v385 = stablehlo.convolution(%v384, %u4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v386 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v387 = stablehlo.add %v385, %v386 : tensor<32x160x14x14xf32>
    %v388 = stablehlo.reshape %v387 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v389 = stablehlo.reshape %v388 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v390 = stablehlo.broadcast_in_dim %u4pnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v391 = stablehlo.subtract %v389, %v390 : tensor<32x160x14x14xf32>
    %v392 = stablehlo.broadcast_in_dim %u4pnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v393 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v394 = stablehlo.add %v392, %v393 : tensor<32x160x14x14xf32>
    %v395 = stablehlo.rsqrt %v394 : tensor<32x160x14x14xf32>
    %v396 = stablehlo.multiply %v391, %v395 : tensor<32x160x14x14xf32>
    %v397 = stablehlo.broadcast_in_dim %u4pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v398 = stablehlo.broadcast_in_dim %u4pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v399 = stablehlo.multiply %v396, %v397 : tensor<32x160x14x14xf32>
    %v400 = stablehlo.add %v399, %v398 : tensor<32x160x14x14xf32>
    %v401 = stablehlo.reshape %v400 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v402 = stablehlo.reshape %v401 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v403 = stablehlo.reshape %v317 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v404 = stablehlo.add %v402, %v403 : tensor<32x160x14x14xf32>
    %v405 = stablehlo.reshape %v404 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v406 = stablehlo.reshape %v405 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v407 = stablehlo.convolution(%v406, %u5qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<32x160x14x14xf32>
    %v408 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v409 = stablehlo.add %v407, %v408 : tensor<32x160x14x14xf32>
    %v410 = stablehlo.reshape %v409 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v411 = stablehlo.reshape %v410 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v412 = stablehlo.broadcast_in_dim %u5qnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v413 = stablehlo.subtract %v411, %v412 : tensor<32x160x14x14xf32>
    %v414 = stablehlo.broadcast_in_dim %u5qnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v415 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v416 = stablehlo.add %v414, %v415 : tensor<32x160x14x14xf32>
    %v417 = stablehlo.rsqrt %v416 : tensor<32x160x14x14xf32>
    %v418 = stablehlo.multiply %v413, %v417 : tensor<32x160x14x14xf32>
    %v419 = stablehlo.broadcast_in_dim %u5qg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v420 = stablehlo.broadcast_in_dim %u5qbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v421 = stablehlo.multiply %v418, %v419 : tensor<32x160x14x14xf32>
    %v422 = stablehlo.add %v421, %v420 : tensor<32x160x14x14xf32>
    %v423 = stablehlo.reshape %v422 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v424 = stablehlo.reshape %v423 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v425 = stablehlo.constant dense<0.0> : tensor<32x160x14x14xf32>
    %v426 = stablehlo.maximum %v424, %v425 : tensor<32x160x14x14xf32>
    %v427 = stablehlo.reshape %v426 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v428 = stablehlo.reshape %v427 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v429 = stablehlo.convolution(%v428, %u5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<32x640x14x14xf32>
    %v430 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v431 = stablehlo.add %v429, %v430 : tensor<32x640x14x14xf32>
    %v432 = stablehlo.reshape %v431 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v433 = stablehlo.reshape %v432 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v434 = stablehlo.broadcast_in_dim %u5enmu, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v435 = stablehlo.subtract %v433, %v434 : tensor<32x640x14x14xf32>
    %v436 = stablehlo.broadcast_in_dim %u5envar, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v437 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v438 = stablehlo.add %v436, %v437 : tensor<32x640x14x14xf32>
    %v439 = stablehlo.rsqrt %v438 : tensor<32x640x14x14xf32>
    %v440 = stablehlo.multiply %v435, %v439 : tensor<32x640x14x14xf32>
    %v441 = stablehlo.broadcast_in_dim %u5eg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v442 = stablehlo.broadcast_in_dim %u5ebt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v443 = stablehlo.multiply %v440, %v441 : tensor<32x640x14x14xf32>
    %v444 = stablehlo.add %v443, %v442 : tensor<32x640x14x14xf32>
    %v445 = stablehlo.reshape %v444 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v446 = stablehlo.reshape %v445 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v447 = stablehlo.constant dense<0.0> : tensor<32x160x28x28xf32>
    %v448 = stablehlo.maximum %v446, %v447 : tensor<32x160x28x28xf32>
    %v449 = stablehlo.reshape %v448 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v450 = stablehlo.reshape %v449 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v451 = stablehlo.convolution(%v450, %u5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 640 : i64} : (tensor<32x640x14x14xf32>, tensor<640x1x3x3xf32>) -> tensor<32x640x14x14xf32>
    %v452 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v453 = stablehlo.add %v451, %v452 : tensor<32x640x14x14xf32>
    %v454 = stablehlo.reshape %v453 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v455 = stablehlo.reshape %v454 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v456 = stablehlo.broadcast_in_dim %u5dnmu, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v457 = stablehlo.subtract %v455, %v456 : tensor<32x640x14x14xf32>
    %v458 = stablehlo.broadcast_in_dim %u5dnvar, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v459 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v460 = stablehlo.add %v458, %v459 : tensor<32x640x14x14xf32>
    %v461 = stablehlo.rsqrt %v460 : tensor<32x640x14x14xf32>
    %v462 = stablehlo.multiply %v457, %v461 : tensor<32x640x14x14xf32>
    %v463 = stablehlo.broadcast_in_dim %u5dg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v464 = stablehlo.broadcast_in_dim %u5dbt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v465 = stablehlo.multiply %v462, %v463 : tensor<32x640x14x14xf32>
    %v466 = stablehlo.add %v465, %v464 : tensor<32x640x14x14xf32>
    %v467 = stablehlo.reshape %v466 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v468 = stablehlo.reshape %v467 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v469 = stablehlo.constant dense<0.0> : tensor<32x160x28x28xf32>
    %v470 = stablehlo.maximum %v468, %v469 : tensor<32x160x28x28xf32>
    %v471 = stablehlo.reshape %v470 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v472 = stablehlo.reshape %v471 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v473 = stablehlo.convolution(%v472, %u5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v474 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v475 = stablehlo.add %v473, %v474 : tensor<32x160x14x14xf32>
    %v476 = stablehlo.reshape %v475 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v477 = stablehlo.reshape %v476 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v478 = stablehlo.broadcast_in_dim %u5pnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v479 = stablehlo.subtract %v477, %v478 : tensor<32x160x14x14xf32>
    %v480 = stablehlo.broadcast_in_dim %u5pnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v481 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v482 = stablehlo.add %v480, %v481 : tensor<32x160x14x14xf32>
    %v483 = stablehlo.rsqrt %v482 : tensor<32x160x14x14xf32>
    %v484 = stablehlo.multiply %v479, %v483 : tensor<32x160x14x14xf32>
    %v485 = stablehlo.broadcast_in_dim %u5pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v486 = stablehlo.broadcast_in_dim %u5pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v487 = stablehlo.multiply %v484, %v485 : tensor<32x160x14x14xf32>
    %v488 = stablehlo.add %v487, %v486 : tensor<32x160x14x14xf32>
    %v489 = stablehlo.reshape %v488 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v490 = stablehlo.reshape %v489 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v491 = stablehlo.reshape %v405 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v492 = stablehlo.add %v490, %v491 : tensor<32x160x14x14xf32>
    %v493 = stablehlo.reshape %v492 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v494 = stablehlo.reshape %v493 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v495 = stablehlo.convolution(%v494, %u6qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<32x160x14x14xf32>
    %v496 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v497 = stablehlo.add %v495, %v496 : tensor<32x160x14x14xf32>
    %v498 = stablehlo.reshape %v497 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v499 = stablehlo.reshape %v498 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v500 = stablehlo.broadcast_in_dim %u6qnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v501 = stablehlo.subtract %v499, %v500 : tensor<32x160x14x14xf32>
    %v502 = stablehlo.broadcast_in_dim %u6qnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v503 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v504 = stablehlo.add %v502, %v503 : tensor<32x160x14x14xf32>
    %v505 = stablehlo.rsqrt %v504 : tensor<32x160x14x14xf32>
    %v506 = stablehlo.multiply %v501, %v505 : tensor<32x160x14x14xf32>
    %v507 = stablehlo.broadcast_in_dim %u6qg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v508 = stablehlo.broadcast_in_dim %u6qbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v509 = stablehlo.multiply %v506, %v507 : tensor<32x160x14x14xf32>
    %v510 = stablehlo.add %v509, %v508 : tensor<32x160x14x14xf32>
    %v511 = stablehlo.reshape %v510 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v512 = stablehlo.reshape %v511 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v513 = stablehlo.constant dense<0.0> : tensor<32x160x14x14xf32>
    %v514 = stablehlo.maximum %v512, %v513 : tensor<32x160x14x14xf32>
    %v515 = stablehlo.reshape %v514 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v516 = stablehlo.reshape %v515 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v517 = stablehlo.convolution(%v516, %u6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<32x640x14x14xf32>
    %v518 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v519 = stablehlo.add %v517, %v518 : tensor<32x640x14x14xf32>
    %v520 = stablehlo.reshape %v519 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v521 = stablehlo.reshape %v520 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v522 = stablehlo.broadcast_in_dim %u6enmu, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v523 = stablehlo.subtract %v521, %v522 : tensor<32x640x14x14xf32>
    %v524 = stablehlo.broadcast_in_dim %u6envar, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v525 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v526 = stablehlo.add %v524, %v525 : tensor<32x640x14x14xf32>
    %v527 = stablehlo.rsqrt %v526 : tensor<32x640x14x14xf32>
    %v528 = stablehlo.multiply %v523, %v527 : tensor<32x640x14x14xf32>
    %v529 = stablehlo.broadcast_in_dim %u6eg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v530 = stablehlo.broadcast_in_dim %u6ebt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v531 = stablehlo.multiply %v528, %v529 : tensor<32x640x14x14xf32>
    %v532 = stablehlo.add %v531, %v530 : tensor<32x640x14x14xf32>
    %v533 = stablehlo.reshape %v532 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v534 = stablehlo.reshape %v533 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v535 = stablehlo.constant dense<0.0> : tensor<32x160x28x28xf32>
    %v536 = stablehlo.maximum %v534, %v535 : tensor<32x160x28x28xf32>
    %v537 = stablehlo.reshape %v536 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v538 = stablehlo.reshape %v537 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v539 = stablehlo.convolution(%v538, %u6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 640 : i64} : (tensor<32x640x14x14xf32>, tensor<640x1x5x5xf32>) -> tensor<32x640x14x14xf32>
    %v540 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v541 = stablehlo.add %v539, %v540 : tensor<32x640x14x14xf32>
    %v542 = stablehlo.reshape %v541 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v543 = stablehlo.reshape %v542 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v544 = stablehlo.broadcast_in_dim %u6dnmu, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v545 = stablehlo.subtract %v543, %v544 : tensor<32x640x14x14xf32>
    %v546 = stablehlo.broadcast_in_dim %u6dnvar, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v547 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v548 = stablehlo.add %v546, %v547 : tensor<32x640x14x14xf32>
    %v549 = stablehlo.rsqrt %v548 : tensor<32x640x14x14xf32>
    %v550 = stablehlo.multiply %v545, %v549 : tensor<32x640x14x14xf32>
    %v551 = stablehlo.broadcast_in_dim %u6dg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v552 = stablehlo.broadcast_in_dim %u6dbt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v553 = stablehlo.multiply %v550, %v551 : tensor<32x640x14x14xf32>
    %v554 = stablehlo.add %v553, %v552 : tensor<32x640x14x14xf32>
    %v555 = stablehlo.reshape %v554 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v556 = stablehlo.reshape %v555 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v557 = stablehlo.constant dense<0.0> : tensor<32x160x28x28xf32>
    %v558 = stablehlo.maximum %v556, %v557 : tensor<32x160x28x28xf32>
    %v559 = stablehlo.reshape %v558 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v560 = stablehlo.reshape %v559 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v561 = stablehlo.convolution(%v560, %u6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v562 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v563 = stablehlo.add %v561, %v562 : tensor<32x160x14x14xf32>
    %v564 = stablehlo.reshape %v563 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v565 = stablehlo.reshape %v564 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v566 = stablehlo.broadcast_in_dim %u6pnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v567 = stablehlo.subtract %v565, %v566 : tensor<32x160x14x14xf32>
    %v568 = stablehlo.broadcast_in_dim %u6pnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v569 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v570 = stablehlo.add %v568, %v569 : tensor<32x160x14x14xf32>
    %v571 = stablehlo.rsqrt %v570 : tensor<32x160x14x14xf32>
    %v572 = stablehlo.multiply %v567, %v571 : tensor<32x160x14x14xf32>
    %v573 = stablehlo.broadcast_in_dim %u6pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v574 = stablehlo.broadcast_in_dim %u6pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v575 = stablehlo.multiply %v572, %v573 : tensor<32x160x14x14xf32>
    %v576 = stablehlo.add %v575, %v574 : tensor<32x160x14x14xf32>
    %v577 = stablehlo.reshape %v576 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v578 = stablehlo.reshape %v577 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v579 = stablehlo.reshape %v493 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v580 = stablehlo.add %v578, %v579 : tensor<32x160x14x14xf32>
    %v581 = stablehlo.reshape %v580 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v582 = stablehlo.reshape %v581 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v583 = stablehlo.convolution(%v582, %u7qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<32x160x14x14xf32>
    %v584 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v585 = stablehlo.add %v583, %v584 : tensor<32x160x14x14xf32>
    %v586 = stablehlo.reshape %v585 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v587 = stablehlo.reshape %v586 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v588 = stablehlo.broadcast_in_dim %u7qnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v589 = stablehlo.subtract %v587, %v588 : tensor<32x160x14x14xf32>
    %v590 = stablehlo.broadcast_in_dim %u7qnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v591 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v592 = stablehlo.add %v590, %v591 : tensor<32x160x14x14xf32>
    %v593 = stablehlo.rsqrt %v592 : tensor<32x160x14x14xf32>
    %v594 = stablehlo.multiply %v589, %v593 : tensor<32x160x14x14xf32>
    %v595 = stablehlo.broadcast_in_dim %u7qg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v596 = stablehlo.broadcast_in_dim %u7qbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v597 = stablehlo.multiply %v594, %v595 : tensor<32x160x14x14xf32>
    %v598 = stablehlo.add %v597, %v596 : tensor<32x160x14x14xf32>
    %v599 = stablehlo.reshape %v598 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v600 = stablehlo.reshape %v599 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v601 = stablehlo.constant dense<0.0> : tensor<32x160x14x14xf32>
    %v602 = stablehlo.maximum %v600, %v601 : tensor<32x160x14x14xf32>
    %v603 = stablehlo.reshape %v602 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v604 = stablehlo.reshape %v603 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v605 = stablehlo.convolution(%v604, %u7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<32x640x14x14xf32>
    %v606 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v607 = stablehlo.add %v605, %v606 : tensor<32x640x14x14xf32>
    %v608 = stablehlo.reshape %v607 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v609 = stablehlo.reshape %v608 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v610 = stablehlo.broadcast_in_dim %u7enmu, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v611 = stablehlo.subtract %v609, %v610 : tensor<32x640x14x14xf32>
    %v612 = stablehlo.broadcast_in_dim %u7envar, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v613 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v614 = stablehlo.add %v612, %v613 : tensor<32x640x14x14xf32>
    %v615 = stablehlo.rsqrt %v614 : tensor<32x640x14x14xf32>
    %v616 = stablehlo.multiply %v611, %v615 : tensor<32x640x14x14xf32>
    %v617 = stablehlo.broadcast_in_dim %u7eg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v618 = stablehlo.broadcast_in_dim %u7ebt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v619 = stablehlo.multiply %v616, %v617 : tensor<32x640x14x14xf32>
    %v620 = stablehlo.add %v619, %v618 : tensor<32x640x14x14xf32>
    %v621 = stablehlo.reshape %v620 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v622 = stablehlo.reshape %v621 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v623 = stablehlo.constant dense<0.0> : tensor<32x160x28x28xf32>
    %v624 = stablehlo.maximum %v622, %v623 : tensor<32x160x28x28xf32>
    %v625 = stablehlo.reshape %v624 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v626 = stablehlo.reshape %v625 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v627 = stablehlo.convolution(%v626, %u7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 640 : i64} : (tensor<32x640x14x14xf32>, tensor<640x1x3x3xf32>) -> tensor<32x640x14x14xf32>
    %v628 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v629 = stablehlo.add %v627, %v628 : tensor<32x640x14x14xf32>
    %v630 = stablehlo.reshape %v629 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v631 = stablehlo.reshape %v630 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v632 = stablehlo.broadcast_in_dim %u7dnmu, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v633 = stablehlo.subtract %v631, %v632 : tensor<32x640x14x14xf32>
    %v634 = stablehlo.broadcast_in_dim %u7dnvar, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v635 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v636 = stablehlo.add %v634, %v635 : tensor<32x640x14x14xf32>
    %v637 = stablehlo.rsqrt %v636 : tensor<32x640x14x14xf32>
    %v638 = stablehlo.multiply %v633, %v637 : tensor<32x640x14x14xf32>
    %v639 = stablehlo.broadcast_in_dim %u7dg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v640 = stablehlo.broadcast_in_dim %u7dbt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v641 = stablehlo.multiply %v638, %v639 : tensor<32x640x14x14xf32>
    %v642 = stablehlo.add %v641, %v640 : tensor<32x640x14x14xf32>
    %v643 = stablehlo.reshape %v642 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v644 = stablehlo.reshape %v643 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v645 = stablehlo.constant dense<0.0> : tensor<32x160x28x28xf32>
    %v646 = stablehlo.maximum %v644, %v645 : tensor<32x160x28x28xf32>
    %v647 = stablehlo.reshape %v646 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v648 = stablehlo.reshape %v647 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v649 = stablehlo.convolution(%v648, %u7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v650 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v651 = stablehlo.add %v649, %v650 : tensor<32x160x14x14xf32>
    %v652 = stablehlo.reshape %v651 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v653 = stablehlo.reshape %v652 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v654 = stablehlo.broadcast_in_dim %u7pnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v655 = stablehlo.subtract %v653, %v654 : tensor<32x160x14x14xf32>
    %v656 = stablehlo.broadcast_in_dim %u7pnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v657 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v658 = stablehlo.add %v656, %v657 : tensor<32x160x14x14xf32>
    %v659 = stablehlo.rsqrt %v658 : tensor<32x160x14x14xf32>
    %v660 = stablehlo.multiply %v655, %v659 : tensor<32x160x14x14xf32>
    %v661 = stablehlo.broadcast_in_dim %u7pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v662 = stablehlo.broadcast_in_dim %u7pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v663 = stablehlo.multiply %v660, %v661 : tensor<32x160x14x14xf32>
    %v664 = stablehlo.add %v663, %v662 : tensor<32x160x14x14xf32>
    %v665 = stablehlo.reshape %v664 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v666 = stablehlo.reshape %v665 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v667 = stablehlo.reshape %v581 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v668 = stablehlo.add %v666, %v667 : tensor<32x160x14x14xf32>
    %v669 = stablehlo.reshape %v668 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v670 = stablehlo.reshape %v669 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v671 = stablehlo.convolution(%v670, %u8qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<32x160x14x14xf32>
    %v672 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v673 = stablehlo.add %v671, %v672 : tensor<32x160x14x14xf32>
    %v674 = stablehlo.reshape %v673 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v675 = stablehlo.reshape %v674 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v676 = stablehlo.broadcast_in_dim %u8qnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v677 = stablehlo.subtract %v675, %v676 : tensor<32x160x14x14xf32>
    %v678 = stablehlo.broadcast_in_dim %u8qnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v679 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v680 = stablehlo.add %v678, %v679 : tensor<32x160x14x14xf32>
    %v681 = stablehlo.rsqrt %v680 : tensor<32x160x14x14xf32>
    %v682 = stablehlo.multiply %v677, %v681 : tensor<32x160x14x14xf32>
    %v683 = stablehlo.broadcast_in_dim %u8qg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v684 = stablehlo.broadcast_in_dim %u8qbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v685 = stablehlo.multiply %v682, %v683 : tensor<32x160x14x14xf32>
    %v686 = stablehlo.add %v685, %v684 : tensor<32x160x14x14xf32>
    %v687 = stablehlo.reshape %v686 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v688 = stablehlo.reshape %v687 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v689 = stablehlo.constant dense<0.0> : tensor<32x160x14x14xf32>
    %v690 = stablehlo.maximum %v688, %v689 : tensor<32x160x14x14xf32>
    %v691 = stablehlo.reshape %v690 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v692 = stablehlo.reshape %v691 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v693 = stablehlo.convolution(%v692, %u8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<32x640x14x14xf32>
    %v694 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v695 = stablehlo.add %v693, %v694 : tensor<32x640x14x14xf32>
    %v696 = stablehlo.reshape %v695 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v697 = stablehlo.reshape %v696 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v698 = stablehlo.broadcast_in_dim %u8enmu, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v699 = stablehlo.subtract %v697, %v698 : tensor<32x640x14x14xf32>
    %v700 = stablehlo.broadcast_in_dim %u8envar, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v701 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v702 = stablehlo.add %v700, %v701 : tensor<32x640x14x14xf32>
    %v703 = stablehlo.rsqrt %v702 : tensor<32x640x14x14xf32>
    %v704 = stablehlo.multiply %v699, %v703 : tensor<32x640x14x14xf32>
    %v705 = stablehlo.broadcast_in_dim %u8eg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v706 = stablehlo.broadcast_in_dim %u8ebt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v707 = stablehlo.multiply %v704, %v705 : tensor<32x640x14x14xf32>
    %v708 = stablehlo.add %v707, %v706 : tensor<32x640x14x14xf32>
    %v709 = stablehlo.reshape %v708 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v710 = stablehlo.reshape %v709 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v711 = stablehlo.constant dense<0.0> : tensor<32x160x28x28xf32>
    %v712 = stablehlo.maximum %v710, %v711 : tensor<32x160x28x28xf32>
    %v713 = stablehlo.reshape %v712 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v714 = stablehlo.reshape %v713 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v715 = stablehlo.convolution(%v714, %u8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v716 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v717 = stablehlo.add %v715, %v716 : tensor<32x160x14x14xf32>
    %v718 = stablehlo.reshape %v717 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v719 = stablehlo.reshape %v718 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v720 = stablehlo.broadcast_in_dim %u8pnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v721 = stablehlo.subtract %v719, %v720 : tensor<32x160x14x14xf32>
    %v722 = stablehlo.broadcast_in_dim %u8pnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v723 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v724 = stablehlo.add %v722, %v723 : tensor<32x160x14x14xf32>
    %v725 = stablehlo.rsqrt %v724 : tensor<32x160x14x14xf32>
    %v726 = stablehlo.multiply %v721, %v725 : tensor<32x160x14x14xf32>
    %v727 = stablehlo.broadcast_in_dim %u8pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v728 = stablehlo.broadcast_in_dim %u8pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v729 = stablehlo.multiply %v726, %v727 : tensor<32x160x14x14xf32>
    %v730 = stablehlo.add %v729, %v728 : tensor<32x160x14x14xf32>
    %v731 = stablehlo.reshape %v730 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v732 = stablehlo.reshape %v731 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v733 = stablehlo.reshape %v669 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v734 = stablehlo.add %v732, %v733 : tensor<32x160x14x14xf32>
    %v735 = stablehlo.reshape %v734 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v736 = stablehlo.reshape %v735 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v737 = stablehlo.convolution(%v736, %u9eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x14x14xf32>, tensor<320x160x1x1xf32>) -> tensor<32x320x14x14xf32>
    %v738 = stablehlo.broadcast_in_dim %zb320, dims = [1] : (tensor<320xf32>) -> tensor<32x320x14x14xf32>
    %v739 = stablehlo.add %v737, %v738 : tensor<32x320x14x14xf32>
    %v740 = stablehlo.reshape %v739 : (tensor<32x320x14x14xf32>) -> tensor<32x62720xf32>
    %v741 = stablehlo.reshape %v740 : (tensor<32x62720xf32>) -> tensor<32x320x14x14xf32>
    %v742 = stablehlo.broadcast_in_dim %u9enmu, dims = [1] : (tensor<320xf32>) -> tensor<32x320x14x14xf32>
    %v743 = stablehlo.subtract %v741, %v742 : tensor<32x320x14x14xf32>
    %v744 = stablehlo.broadcast_in_dim %u9envar, dims = [1] : (tensor<320xf32>) -> tensor<32x320x14x14xf32>
    %v745 = stablehlo.constant dense<1.0e-5> : tensor<32x320x14x14xf32>
    %v746 = stablehlo.add %v744, %v745 : tensor<32x320x14x14xf32>
    %v747 = stablehlo.rsqrt %v746 : tensor<32x320x14x14xf32>
    %v748 = stablehlo.multiply %v743, %v747 : tensor<32x320x14x14xf32>
    %v749 = stablehlo.broadcast_in_dim %u9eg, dims = [1] : (tensor<320xf32>) -> tensor<32x320x14x14xf32>
    %v750 = stablehlo.broadcast_in_dim %u9ebt, dims = [1] : (tensor<320xf32>) -> tensor<32x320x14x14xf32>
    %v751 = stablehlo.multiply %v748, %v749 : tensor<32x320x14x14xf32>
    %v752 = stablehlo.add %v751, %v750 : tensor<32x320x14x14xf32>
    %v753 = stablehlo.reshape %v752 : (tensor<32x320x14x14xf32>) -> tensor<32x62720xf32>
    %v754 = stablehlo.reshape %v753 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v755 = stablehlo.constant dense<0.0> : tensor<32x80x28x28xf32>
    %v756 = stablehlo.maximum %v754, %v755 : tensor<32x80x28x28xf32>
    %v757 = stablehlo.reshape %v756 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v758 = stablehlo.reshape %v757 : (tensor<32x62720xf32>) -> tensor<32x320x14x14xf32>
    %v759 = stablehlo.convolution(%v758, %u9pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x320x14x14xf32>, tensor<160x320x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v760 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v761 = stablehlo.add %v759, %v760 : tensor<32x160x14x14xf32>
    %v762 = stablehlo.reshape %v761 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v763 = stablehlo.reshape %v762 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v764 = stablehlo.broadcast_in_dim %u9pnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v765 = stablehlo.subtract %v763, %v764 : tensor<32x160x14x14xf32>
    %v766 = stablehlo.broadcast_in_dim %u9pnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v767 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v768 = stablehlo.add %v766, %v767 : tensor<32x160x14x14xf32>
    %v769 = stablehlo.rsqrt %v768 : tensor<32x160x14x14xf32>
    %v770 = stablehlo.multiply %v765, %v769 : tensor<32x160x14x14xf32>
    %v771 = stablehlo.broadcast_in_dim %u9pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v772 = stablehlo.broadcast_in_dim %u9pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v773 = stablehlo.multiply %v770, %v771 : tensor<32x160x14x14xf32>
    %v774 = stablehlo.add %v773, %v772 : tensor<32x160x14x14xf32>
    %v775 = stablehlo.reshape %v774 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v776 = stablehlo.reshape %v775 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v777 = stablehlo.reshape %v735 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v778 = stablehlo.add %v776, %v777 : tensor<32x160x14x14xf32>
    %v779 = stablehlo.reshape %v778 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v780 = stablehlo.reshape %v779 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v781 = stablehlo.convolution(%v780, %u10qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<32x160x14x14xf32>
    %v782 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v783 = stablehlo.add %v781, %v782 : tensor<32x160x14x14xf32>
    %v784 = stablehlo.reshape %v783 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v785 = stablehlo.reshape %v784 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v786 = stablehlo.broadcast_in_dim %u10qnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v787 = stablehlo.subtract %v785, %v786 : tensor<32x160x14x14xf32>
    %v788 = stablehlo.broadcast_in_dim %u10qnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v789 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v790 = stablehlo.add %v788, %v789 : tensor<32x160x14x14xf32>
    %v791 = stablehlo.rsqrt %v790 : tensor<32x160x14x14xf32>
    %v792 = stablehlo.multiply %v787, %v791 : tensor<32x160x14x14xf32>
    %v793 = stablehlo.broadcast_in_dim %u10qg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v794 = stablehlo.broadcast_in_dim %u10qbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v795 = stablehlo.multiply %v792, %v793 : tensor<32x160x14x14xf32>
    %v796 = stablehlo.add %v795, %v794 : tensor<32x160x14x14xf32>
    %v797 = stablehlo.reshape %v796 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v798 = stablehlo.reshape %v797 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v799 = stablehlo.constant dense<0.0> : tensor<32x160x14x14xf32>
    %v800 = stablehlo.maximum %v798, %v799 : tensor<32x160x14x14xf32>
    %v801 = stablehlo.reshape %v800 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v802 = stablehlo.reshape %v801 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v803 = stablehlo.convolution(%v802, %u10eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<32x640x14x14xf32>
    %v804 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v805 = stablehlo.add %v803, %v804 : tensor<32x640x14x14xf32>
    %v806 = stablehlo.reshape %v805 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v807 = stablehlo.reshape %v806 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v808 = stablehlo.broadcast_in_dim %u10enmu, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v809 = stablehlo.subtract %v807, %v808 : tensor<32x640x14x14xf32>
    %v810 = stablehlo.broadcast_in_dim %u10envar, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v811 = stablehlo.constant dense<1.0e-5> : tensor<32x640x14x14xf32>
    %v812 = stablehlo.add %v810, %v811 : tensor<32x640x14x14xf32>
    %v813 = stablehlo.rsqrt %v812 : tensor<32x640x14x14xf32>
    %v814 = stablehlo.multiply %v809, %v813 : tensor<32x640x14x14xf32>
    %v815 = stablehlo.broadcast_in_dim %u10eg, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v816 = stablehlo.broadcast_in_dim %u10ebt, dims = [1] : (tensor<640xf32>) -> tensor<32x640x14x14xf32>
    %v817 = stablehlo.multiply %v814, %v815 : tensor<32x640x14x14xf32>
    %v818 = stablehlo.add %v817, %v816 : tensor<32x640x14x14xf32>
    %v819 = stablehlo.reshape %v818 : (tensor<32x640x14x14xf32>) -> tensor<32x125440xf32>
    %v820 = stablehlo.reshape %v819 : (tensor<32x125440xf32>) -> tensor<32x160x28x28xf32>
    %v821 = stablehlo.constant dense<0.0> : tensor<32x160x28x28xf32>
    %v822 = stablehlo.maximum %v820, %v821 : tensor<32x160x28x28xf32>
    %v823 = stablehlo.reshape %v822 : (tensor<32x160x28x28xf32>) -> tensor<32x125440xf32>
    %v824 = stablehlo.reshape %v823 : (tensor<32x125440xf32>) -> tensor<32x640x14x14xf32>
    %v825 = stablehlo.convolution(%v824, %u10pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<32x160x14x14xf32>
    %v826 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v827 = stablehlo.add %v825, %v826 : tensor<32x160x14x14xf32>
    %v828 = stablehlo.reshape %v827 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v829 = stablehlo.reshape %v828 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v830 = stablehlo.broadcast_in_dim %u10pnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v831 = stablehlo.subtract %v829, %v830 : tensor<32x160x14x14xf32>
    %v832 = stablehlo.broadcast_in_dim %u10pnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v833 = stablehlo.constant dense<1.0e-5> : tensor<32x160x14x14xf32>
    %v834 = stablehlo.add %v832, %v833 : tensor<32x160x14x14xf32>
    %v835 = stablehlo.rsqrt %v834 : tensor<32x160x14x14xf32>
    %v836 = stablehlo.multiply %v831, %v835 : tensor<32x160x14x14xf32>
    %v837 = stablehlo.broadcast_in_dim %u10pg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v838 = stablehlo.broadcast_in_dim %u10pbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x14x14xf32>
    %v839 = stablehlo.multiply %v836, %v837 : tensor<32x160x14x14xf32>
    %v840 = stablehlo.add %v839, %v838 : tensor<32x160x14x14xf32>
    %v841 = stablehlo.reshape %v840 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v842 = stablehlo.reshape %v841 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v843 = stablehlo.reshape %v779 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v844 = stablehlo.add %v842, %v843 : tensor<32x160x14x14xf32>
    %v845 = stablehlo.reshape %v844 : (tensor<32x160x14x14xf32>) -> tensor<32x31360xf32>
    %v846 = stablehlo.reshape %v845 : (tensor<32x31360xf32>) -> tensor<32x160x14x14xf32>
    %v847 = stablehlo.convolution(%v846, %u11qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<32x160x14x14xf32>, tensor<160x1x5x5xf32>) -> tensor<32x160x7x7xf32>
    %v848 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v849 = stablehlo.add %v847, %v848 : tensor<32x160x7x7xf32>
    %v850 = stablehlo.reshape %v849 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v851 = stablehlo.reshape %v850 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v852 = stablehlo.broadcast_in_dim %u11qnmu, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v853 = stablehlo.subtract %v851, %v852 : tensor<32x160x7x7xf32>
    %v854 = stablehlo.broadcast_in_dim %u11qnvar, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v855 = stablehlo.constant dense<1.0e-5> : tensor<32x160x7x7xf32>
    %v856 = stablehlo.add %v854, %v855 : tensor<32x160x7x7xf32>
    %v857 = stablehlo.rsqrt %v856 : tensor<32x160x7x7xf32>
    %v858 = stablehlo.multiply %v853, %v857 : tensor<32x160x7x7xf32>
    %v859 = stablehlo.broadcast_in_dim %u11qg, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v860 = stablehlo.broadcast_in_dim %u11qbt, dims = [1] : (tensor<160xf32>) -> tensor<32x160x7x7xf32>
    %v861 = stablehlo.multiply %v858, %v859 : tensor<32x160x7x7xf32>
    %v862 = stablehlo.add %v861, %v860 : tensor<32x160x7x7xf32>
    %v863 = stablehlo.reshape %v862 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v864 = stablehlo.reshape %v863 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v865 = stablehlo.constant dense<0.0> : tensor<32x160x7x7xf32>
    %v866 = stablehlo.maximum %v864, %v865 : tensor<32x160x7x7xf32>
    %v867 = stablehlo.reshape %v866 : (tensor<32x160x7x7xf32>) -> tensor<32x7840xf32>
    %v868 = stablehlo.reshape %v867 : (tensor<32x7840xf32>) -> tensor<32x160x7x7xf32>
    %v869 = stablehlo.convolution(%v868, %u11eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v870 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v871 = stablehlo.add %v869, %v870 : tensor<32x960x7x7xf32>
    %v872 = stablehlo.reshape %v871 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v873 = stablehlo.reshape %v872 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v874 = stablehlo.broadcast_in_dim %u11enmu, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v875 = stablehlo.subtract %v873, %v874 : tensor<32x960x7x7xf32>
    %v876 = stablehlo.broadcast_in_dim %u11envar, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v877 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v878 = stablehlo.add %v876, %v877 : tensor<32x960x7x7xf32>
    %v879 = stablehlo.rsqrt %v878 : tensor<32x960x7x7xf32>
    %v880 = stablehlo.multiply %v875, %v879 : tensor<32x960x7x7xf32>
    %v881 = stablehlo.broadcast_in_dim %u11eg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v882 = stablehlo.broadcast_in_dim %u11ebt, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v883 = stablehlo.multiply %v880, %v881 : tensor<32x960x7x7xf32>
    %v884 = stablehlo.add %v883, %v882 : tensor<32x960x7x7xf32>
    %v885 = stablehlo.reshape %v884 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v886 = stablehlo.reshape %v885 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v887 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v888 = stablehlo.maximum %v886, %v887 : tensor<32x960x7x7xf32>
    %v889 = stablehlo.reshape %v888 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v890 = stablehlo.reshape %v889 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v891 = stablehlo.convolution(%v890, %u11dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<32x960x7x7xf32>, tensor<960x1x5x5xf32>) -> tensor<32x960x7x7xf32>
    %v892 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v893 = stablehlo.add %v891, %v892 : tensor<32x960x7x7xf32>
    %v894 = stablehlo.reshape %v893 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v895 = stablehlo.reshape %v894 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v896 = stablehlo.broadcast_in_dim %u11dnmu, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v897 = stablehlo.subtract %v895, %v896 : tensor<32x960x7x7xf32>
    %v898 = stablehlo.broadcast_in_dim %u11dnvar, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v899 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v900 = stablehlo.add %v898, %v899 : tensor<32x960x7x7xf32>
    %v901 = stablehlo.rsqrt %v900 : tensor<32x960x7x7xf32>
    %v902 = stablehlo.multiply %v897, %v901 : tensor<32x960x7x7xf32>
    %v903 = stablehlo.broadcast_in_dim %u11dg, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v904 = stablehlo.broadcast_in_dim %u11dbt, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v905 = stablehlo.multiply %v902, %v903 : tensor<32x960x7x7xf32>
    %v906 = stablehlo.add %v905, %v904 : tensor<32x960x7x7xf32>
    %v907 = stablehlo.reshape %v906 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v908 = stablehlo.reshape %v907 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v909 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v910 = stablehlo.maximum %v908, %v909 : tensor<32x960x7x7xf32>
    %v911 = stablehlo.reshape %v910 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v912 = stablehlo.reshape %v911 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v913 = stablehlo.convolution(%v912, %u11pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<256x960x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v914 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v915 = stablehlo.add %v913, %v914 : tensor<32x256x7x7xf32>
    %v916 = stablehlo.reshape %v915 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v917 = stablehlo.reshape %v916 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v918 = stablehlo.broadcast_in_dim %u11pnmu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v919 = stablehlo.subtract %v917, %v918 : tensor<32x256x7x7xf32>
    %v920 = stablehlo.broadcast_in_dim %u11pnvar, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v921 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v922 = stablehlo.add %v920, %v921 : tensor<32x256x7x7xf32>
    %v923 = stablehlo.rsqrt %v922 : tensor<32x256x7x7xf32>
    %v924 = stablehlo.multiply %v919, %v923 : tensor<32x256x7x7xf32>
    %v925 = stablehlo.broadcast_in_dim %u11pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v926 = stablehlo.broadcast_in_dim %u11pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v927 = stablehlo.multiply %v924, %v925 : tensor<32x256x7x7xf32>
    %v928 = stablehlo.add %v927, %v926 : tensor<32x256x7x7xf32>
    %v929 = stablehlo.reshape %v928 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v930 = stablehlo.reshape %v929 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v931 = stablehlo.convolution(%v930, %u12qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x7x7xf32>, tensor<256x1x5x5xf32>) -> tensor<32x256x7x7xf32>
    %v932 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v933 = stablehlo.add %v931, %v932 : tensor<32x256x7x7xf32>
    %v934 = stablehlo.reshape %v933 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v935 = stablehlo.reshape %v934 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v936 = stablehlo.broadcast_in_dim %u12qnmu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v937 = stablehlo.subtract %v935, %v936 : tensor<32x256x7x7xf32>
    %v938 = stablehlo.broadcast_in_dim %u12qnvar, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v939 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v940 = stablehlo.add %v938, %v939 : tensor<32x256x7x7xf32>
    %v941 = stablehlo.rsqrt %v940 : tensor<32x256x7x7xf32>
    %v942 = stablehlo.multiply %v937, %v941 : tensor<32x256x7x7xf32>
    %v943 = stablehlo.broadcast_in_dim %u12qg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v944 = stablehlo.broadcast_in_dim %u12qbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v945 = stablehlo.multiply %v942, %v943 : tensor<32x256x7x7xf32>
    %v946 = stablehlo.add %v945, %v944 : tensor<32x256x7x7xf32>
    %v947 = stablehlo.reshape %v946 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v948 = stablehlo.reshape %v947 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v949 = stablehlo.constant dense<0.0> : tensor<32x256x7x7xf32>
    %v950 = stablehlo.maximum %v948, %v949 : tensor<32x256x7x7xf32>
    %v951 = stablehlo.reshape %v950 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v952 = stablehlo.reshape %v951 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v953 = stablehlo.convolution(%v952, %u12eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v954 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v955 = stablehlo.add %v953, %v954 : tensor<32x1024x7x7xf32>
    %v956 = stablehlo.reshape %v955 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v957 = stablehlo.reshape %v956 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v958 = stablehlo.broadcast_in_dim %u12enmu, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v959 = stablehlo.subtract %v957, %v958 : tensor<32x1024x7x7xf32>
    %v960 = stablehlo.broadcast_in_dim %u12envar, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v961 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v962 = stablehlo.add %v960, %v961 : tensor<32x1024x7x7xf32>
    %v963 = stablehlo.rsqrt %v962 : tensor<32x1024x7x7xf32>
    %v964 = stablehlo.multiply %v959, %v963 : tensor<32x1024x7x7xf32>
    %v965 = stablehlo.broadcast_in_dim %u12eg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v966 = stablehlo.broadcast_in_dim %u12ebt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v967 = stablehlo.multiply %v964, %v965 : tensor<32x1024x7x7xf32>
    %v968 = stablehlo.add %v967, %v966 : tensor<32x1024x7x7xf32>
    %v969 = stablehlo.reshape %v968 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v970 = stablehlo.reshape %v969 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v971 = stablehlo.constant dense<0.0> : tensor<32x1024x7x7xf32>
    %v972 = stablehlo.maximum %v970, %v971 : tensor<32x1024x7x7xf32>
    %v973 = stablehlo.reshape %v972 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v974 = stablehlo.reshape %v973 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v975 = stablehlo.convolution(%v974, %u12dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<32x1024x7x7xf32>, tensor<1024x1x5x5xf32>) -> tensor<32x1024x7x7xf32>
    %v976 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v977 = stablehlo.add %v975, %v976 : tensor<32x1024x7x7xf32>
    %v978 = stablehlo.reshape %v977 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v979 = stablehlo.reshape %v978 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v980 = stablehlo.broadcast_in_dim %u12dnmu, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v981 = stablehlo.subtract %v979, %v980 : tensor<32x1024x7x7xf32>
    %v982 = stablehlo.broadcast_in_dim %u12dnvar, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v983 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v984 = stablehlo.add %v982, %v983 : tensor<32x1024x7x7xf32>
    %v985 = stablehlo.rsqrt %v984 : tensor<32x1024x7x7xf32>
    %v986 = stablehlo.multiply %v981, %v985 : tensor<32x1024x7x7xf32>
    %v987 = stablehlo.broadcast_in_dim %u12dg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v988 = stablehlo.broadcast_in_dim %u12dbt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v989 = stablehlo.multiply %v986, %v987 : tensor<32x1024x7x7xf32>
    %v990 = stablehlo.add %v989, %v988 : tensor<32x1024x7x7xf32>
    %v991 = stablehlo.reshape %v990 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v992 = stablehlo.reshape %v991 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v993 = stablehlo.constant dense<0.0> : tensor<32x1024x7x7xf32>
    %v994 = stablehlo.maximum %v992, %v993 : tensor<32x1024x7x7xf32>
    %v995 = stablehlo.reshape %v994 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v996 = stablehlo.reshape %v995 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v997 = stablehlo.convolution(%v996, %u12pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v998 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v999 = stablehlo.add %v997, %v998 : tensor<32x256x7x7xf32>
    %v1000 = stablehlo.reshape %v999 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1001 = stablehlo.reshape %v1000 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1002 = stablehlo.broadcast_in_dim %u12pnmu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1003 = stablehlo.subtract %v1001, %v1002 : tensor<32x256x7x7xf32>
    %v1004 = stablehlo.broadcast_in_dim %u12pnvar, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1005 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1006 = stablehlo.add %v1004, %v1005 : tensor<32x256x7x7xf32>
    %v1007 = stablehlo.rsqrt %v1006 : tensor<32x256x7x7xf32>
    %v1008 = stablehlo.multiply %v1003, %v1007 : tensor<32x256x7x7xf32>
    %v1009 = stablehlo.broadcast_in_dim %u12pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1010 = stablehlo.broadcast_in_dim %u12pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1011 = stablehlo.multiply %v1008, %v1009 : tensor<32x256x7x7xf32>
    %v1012 = stablehlo.add %v1011, %v1010 : tensor<32x256x7x7xf32>
    %v1013 = stablehlo.reshape %v1012 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1014 = stablehlo.reshape %v1013 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1015 = stablehlo.reshape %v929 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1016 = stablehlo.add %v1014, %v1015 : tensor<32x256x7x7xf32>
    %v1017 = stablehlo.reshape %v1016 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1018 = stablehlo.reshape %v1017 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1019 = stablehlo.convolution(%v1018, %u13qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x7x7xf32>, tensor<256x1x3x3xf32>) -> tensor<32x256x7x7xf32>
    %v1020 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1021 = stablehlo.add %v1019, %v1020 : tensor<32x256x7x7xf32>
    %v1022 = stablehlo.reshape %v1021 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1023 = stablehlo.reshape %v1022 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1024 = stablehlo.broadcast_in_dim %u13qnmu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1025 = stablehlo.subtract %v1023, %v1024 : tensor<32x256x7x7xf32>
    %v1026 = stablehlo.broadcast_in_dim %u13qnvar, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1027 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1028 = stablehlo.add %v1026, %v1027 : tensor<32x256x7x7xf32>
    %v1029 = stablehlo.rsqrt %v1028 : tensor<32x256x7x7xf32>
    %v1030 = stablehlo.multiply %v1025, %v1029 : tensor<32x256x7x7xf32>
    %v1031 = stablehlo.broadcast_in_dim %u13qg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1032 = stablehlo.broadcast_in_dim %u13qbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1033 = stablehlo.multiply %v1030, %v1031 : tensor<32x256x7x7xf32>
    %v1034 = stablehlo.add %v1033, %v1032 : tensor<32x256x7x7xf32>
    %v1035 = stablehlo.reshape %v1034 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1036 = stablehlo.reshape %v1035 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1037 = stablehlo.constant dense<0.0> : tensor<32x256x7x7xf32>
    %v1038 = stablehlo.maximum %v1036, %v1037 : tensor<32x256x7x7xf32>
    %v1039 = stablehlo.reshape %v1038 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1040 = stablehlo.reshape %v1039 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1041 = stablehlo.convolution(%v1040, %u13eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v1042 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1043 = stablehlo.add %v1041, %v1042 : tensor<32x1024x7x7xf32>
    %v1044 = stablehlo.reshape %v1043 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1045 = stablehlo.reshape %v1044 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1046 = stablehlo.broadcast_in_dim %u13enmu, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1047 = stablehlo.subtract %v1045, %v1046 : tensor<32x1024x7x7xf32>
    %v1048 = stablehlo.broadcast_in_dim %u13envar, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1049 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1050 = stablehlo.add %v1048, %v1049 : tensor<32x1024x7x7xf32>
    %v1051 = stablehlo.rsqrt %v1050 : tensor<32x1024x7x7xf32>
    %v1052 = stablehlo.multiply %v1047, %v1051 : tensor<32x1024x7x7xf32>
    %v1053 = stablehlo.broadcast_in_dim %u13eg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1054 = stablehlo.broadcast_in_dim %u13ebt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1055 = stablehlo.multiply %v1052, %v1053 : tensor<32x1024x7x7xf32>
    %v1056 = stablehlo.add %v1055, %v1054 : tensor<32x1024x7x7xf32>
    %v1057 = stablehlo.reshape %v1056 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1058 = stablehlo.reshape %v1057 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1059 = stablehlo.constant dense<0.0> : tensor<32x1024x7x7xf32>
    %v1060 = stablehlo.maximum %v1058, %v1059 : tensor<32x1024x7x7xf32>
    %v1061 = stablehlo.reshape %v1060 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1062 = stablehlo.reshape %v1061 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1063 = stablehlo.convolution(%v1062, %u13dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<32x1024x7x7xf32>, tensor<1024x1x5x5xf32>) -> tensor<32x1024x7x7xf32>
    %v1064 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1065 = stablehlo.add %v1063, %v1064 : tensor<32x1024x7x7xf32>
    %v1066 = stablehlo.reshape %v1065 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1067 = stablehlo.reshape %v1066 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1068 = stablehlo.broadcast_in_dim %u13dnmu, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1069 = stablehlo.subtract %v1067, %v1068 : tensor<32x1024x7x7xf32>
    %v1070 = stablehlo.broadcast_in_dim %u13dnvar, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1071 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1072 = stablehlo.add %v1070, %v1071 : tensor<32x1024x7x7xf32>
    %v1073 = stablehlo.rsqrt %v1072 : tensor<32x1024x7x7xf32>
    %v1074 = stablehlo.multiply %v1069, %v1073 : tensor<32x1024x7x7xf32>
    %v1075 = stablehlo.broadcast_in_dim %u13dg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1076 = stablehlo.broadcast_in_dim %u13dbt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1077 = stablehlo.multiply %v1074, %v1075 : tensor<32x1024x7x7xf32>
    %v1078 = stablehlo.add %v1077, %v1076 : tensor<32x1024x7x7xf32>
    %v1079 = stablehlo.reshape %v1078 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1080 = stablehlo.reshape %v1079 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1081 = stablehlo.constant dense<0.0> : tensor<32x1024x7x7xf32>
    %v1082 = stablehlo.maximum %v1080, %v1081 : tensor<32x1024x7x7xf32>
    %v1083 = stablehlo.reshape %v1082 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1084 = stablehlo.reshape %v1083 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1085 = stablehlo.convolution(%v1084, %u13pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1086 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1087 = stablehlo.add %v1085, %v1086 : tensor<32x256x7x7xf32>
    %v1088 = stablehlo.reshape %v1087 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1089 = stablehlo.reshape %v1088 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1090 = stablehlo.broadcast_in_dim %u13pnmu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1091 = stablehlo.subtract %v1089, %v1090 : tensor<32x256x7x7xf32>
    %v1092 = stablehlo.broadcast_in_dim %u13pnvar, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1093 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1094 = stablehlo.add %v1092, %v1093 : tensor<32x256x7x7xf32>
    %v1095 = stablehlo.rsqrt %v1094 : tensor<32x256x7x7xf32>
    %v1096 = stablehlo.multiply %v1091, %v1095 : tensor<32x256x7x7xf32>
    %v1097 = stablehlo.broadcast_in_dim %u13pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1098 = stablehlo.broadcast_in_dim %u13pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1099 = stablehlo.multiply %v1096, %v1097 : tensor<32x256x7x7xf32>
    %v1100 = stablehlo.add %v1099, %v1098 : tensor<32x256x7x7xf32>
    %v1101 = stablehlo.reshape %v1100 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1102 = stablehlo.reshape %v1101 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1103 = stablehlo.reshape %v1017 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1104 = stablehlo.add %v1102, %v1103 : tensor<32x256x7x7xf32>
    %v1105 = stablehlo.reshape %v1104 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1106 = stablehlo.reshape %v1105 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1107 = stablehlo.convolution(%v1106, %u14qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x7x7xf32>, tensor<256x1x3x3xf32>) -> tensor<32x256x7x7xf32>
    %v1108 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1109 = stablehlo.add %v1107, %v1108 : tensor<32x256x7x7xf32>
    %v1110 = stablehlo.reshape %v1109 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1111 = stablehlo.reshape %v1110 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1112 = stablehlo.broadcast_in_dim %u14qnmu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1113 = stablehlo.subtract %v1111, %v1112 : tensor<32x256x7x7xf32>
    %v1114 = stablehlo.broadcast_in_dim %u14qnvar, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1115 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1116 = stablehlo.add %v1114, %v1115 : tensor<32x256x7x7xf32>
    %v1117 = stablehlo.rsqrt %v1116 : tensor<32x256x7x7xf32>
    %v1118 = stablehlo.multiply %v1113, %v1117 : tensor<32x256x7x7xf32>
    %v1119 = stablehlo.broadcast_in_dim %u14qg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1120 = stablehlo.broadcast_in_dim %u14qbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1121 = stablehlo.multiply %v1118, %v1119 : tensor<32x256x7x7xf32>
    %v1122 = stablehlo.add %v1121, %v1120 : tensor<32x256x7x7xf32>
    %v1123 = stablehlo.reshape %v1122 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1124 = stablehlo.reshape %v1123 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1125 = stablehlo.constant dense<0.0> : tensor<32x256x7x7xf32>
    %v1126 = stablehlo.maximum %v1124, %v1125 : tensor<32x256x7x7xf32>
    %v1127 = stablehlo.reshape %v1126 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1128 = stablehlo.reshape %v1127 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1129 = stablehlo.convolution(%v1128, %u14eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v1130 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1131 = stablehlo.add %v1129, %v1130 : tensor<32x1024x7x7xf32>
    %v1132 = stablehlo.reshape %v1131 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1133 = stablehlo.reshape %v1132 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1134 = stablehlo.broadcast_in_dim %u14enmu, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1135 = stablehlo.subtract %v1133, %v1134 : tensor<32x1024x7x7xf32>
    %v1136 = stablehlo.broadcast_in_dim %u14envar, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1137 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1138 = stablehlo.add %v1136, %v1137 : tensor<32x1024x7x7xf32>
    %v1139 = stablehlo.rsqrt %v1138 : tensor<32x1024x7x7xf32>
    %v1140 = stablehlo.multiply %v1135, %v1139 : tensor<32x1024x7x7xf32>
    %v1141 = stablehlo.broadcast_in_dim %u14eg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1142 = stablehlo.broadcast_in_dim %u14ebt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1143 = stablehlo.multiply %v1140, %v1141 : tensor<32x1024x7x7xf32>
    %v1144 = stablehlo.add %v1143, %v1142 : tensor<32x1024x7x7xf32>
    %v1145 = stablehlo.reshape %v1144 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1146 = stablehlo.reshape %v1145 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1147 = stablehlo.constant dense<0.0> : tensor<32x1024x7x7xf32>
    %v1148 = stablehlo.maximum %v1146, %v1147 : tensor<32x1024x7x7xf32>
    %v1149 = stablehlo.reshape %v1148 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1150 = stablehlo.reshape %v1149 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1151 = stablehlo.convolution(%v1150, %u14dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<32x1024x7x7xf32>, tensor<1024x1x5x5xf32>) -> tensor<32x1024x7x7xf32>
    %v1152 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1153 = stablehlo.add %v1151, %v1152 : tensor<32x1024x7x7xf32>
    %v1154 = stablehlo.reshape %v1153 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1155 = stablehlo.reshape %v1154 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1156 = stablehlo.broadcast_in_dim %u14dnmu, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1157 = stablehlo.subtract %v1155, %v1156 : tensor<32x1024x7x7xf32>
    %v1158 = stablehlo.broadcast_in_dim %u14dnvar, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1159 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1160 = stablehlo.add %v1158, %v1159 : tensor<32x1024x7x7xf32>
    %v1161 = stablehlo.rsqrt %v1160 : tensor<32x1024x7x7xf32>
    %v1162 = stablehlo.multiply %v1157, %v1161 : tensor<32x1024x7x7xf32>
    %v1163 = stablehlo.broadcast_in_dim %u14dg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1164 = stablehlo.broadcast_in_dim %u14dbt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1165 = stablehlo.multiply %v1162, %v1163 : tensor<32x1024x7x7xf32>
    %v1166 = stablehlo.add %v1165, %v1164 : tensor<32x1024x7x7xf32>
    %v1167 = stablehlo.reshape %v1166 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1168 = stablehlo.reshape %v1167 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1169 = stablehlo.constant dense<0.0> : tensor<32x1024x7x7xf32>
    %v1170 = stablehlo.maximum %v1168, %v1169 : tensor<32x1024x7x7xf32>
    %v1171 = stablehlo.reshape %v1170 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1172 = stablehlo.reshape %v1171 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1173 = stablehlo.convolution(%v1172, %u14pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1174 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1175 = stablehlo.add %v1173, %v1174 : tensor<32x256x7x7xf32>
    %v1176 = stablehlo.reshape %v1175 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1177 = stablehlo.reshape %v1176 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1178 = stablehlo.broadcast_in_dim %u14pnmu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1179 = stablehlo.subtract %v1177, %v1178 : tensor<32x256x7x7xf32>
    %v1180 = stablehlo.broadcast_in_dim %u14pnvar, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1181 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1182 = stablehlo.add %v1180, %v1181 : tensor<32x256x7x7xf32>
    %v1183 = stablehlo.rsqrt %v1182 : tensor<32x256x7x7xf32>
    %v1184 = stablehlo.multiply %v1179, %v1183 : tensor<32x256x7x7xf32>
    %v1185 = stablehlo.broadcast_in_dim %u14pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1186 = stablehlo.broadcast_in_dim %u14pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1187 = stablehlo.multiply %v1184, %v1185 : tensor<32x256x7x7xf32>
    %v1188 = stablehlo.add %v1187, %v1186 : tensor<32x256x7x7xf32>
    %v1189 = stablehlo.reshape %v1188 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1190 = stablehlo.reshape %v1189 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1191 = stablehlo.reshape %v1105 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1192 = stablehlo.add %v1190, %v1191 : tensor<32x256x7x7xf32>
    %v1193 = stablehlo.reshape %v1192 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1194 = stablehlo.reshape %v1193 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1195 = stablehlo.convolution(%v1194, %u15eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v1196 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1197 = stablehlo.add %v1195, %v1196 : tensor<32x1024x7x7xf32>
    %v1198 = stablehlo.reshape %v1197 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1199 = stablehlo.reshape %v1198 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1200 = stablehlo.broadcast_in_dim %u15enmu, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1201 = stablehlo.subtract %v1199, %v1200 : tensor<32x1024x7x7xf32>
    %v1202 = stablehlo.broadcast_in_dim %u15envar, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1203 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1204 = stablehlo.add %v1202, %v1203 : tensor<32x1024x7x7xf32>
    %v1205 = stablehlo.rsqrt %v1204 : tensor<32x1024x7x7xf32>
    %v1206 = stablehlo.multiply %v1201, %v1205 : tensor<32x1024x7x7xf32>
    %v1207 = stablehlo.broadcast_in_dim %u15eg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1208 = stablehlo.broadcast_in_dim %u15ebt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1209 = stablehlo.multiply %v1206, %v1207 : tensor<32x1024x7x7xf32>
    %v1210 = stablehlo.add %v1209, %v1208 : tensor<32x1024x7x7xf32>
    %v1211 = stablehlo.reshape %v1210 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1212 = stablehlo.reshape %v1211 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1213 = stablehlo.constant dense<0.0> : tensor<32x1024x7x7xf32>
    %v1214 = stablehlo.maximum %v1212, %v1213 : tensor<32x1024x7x7xf32>
    %v1215 = stablehlo.reshape %v1214 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1216 = stablehlo.reshape %v1215 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1217 = stablehlo.convolution(%v1216, %u15pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1218 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1219 = stablehlo.add %v1217, %v1218 : tensor<32x256x7x7xf32>
    %v1220 = stablehlo.reshape %v1219 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1221 = stablehlo.reshape %v1220 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1222 = stablehlo.broadcast_in_dim %u15pnmu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1223 = stablehlo.subtract %v1221, %v1222 : tensor<32x256x7x7xf32>
    %v1224 = stablehlo.broadcast_in_dim %u15pnvar, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1225 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1226 = stablehlo.add %v1224, %v1225 : tensor<32x256x7x7xf32>
    %v1227 = stablehlo.rsqrt %v1226 : tensor<32x256x7x7xf32>
    %v1228 = stablehlo.multiply %v1223, %v1227 : tensor<32x256x7x7xf32>
    %v1229 = stablehlo.broadcast_in_dim %u15pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1230 = stablehlo.broadcast_in_dim %u15pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1231 = stablehlo.multiply %v1228, %v1229 : tensor<32x256x7x7xf32>
    %v1232 = stablehlo.add %v1231, %v1230 : tensor<32x256x7x7xf32>
    %v1233 = stablehlo.reshape %v1232 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1234 = stablehlo.reshape %v1233 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1235 = stablehlo.reshape %v1193 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1236 = stablehlo.add %v1234, %v1235 : tensor<32x256x7x7xf32>
    %v1237 = stablehlo.reshape %v1236 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1238 = stablehlo.reshape %v1237 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1239 = stablehlo.convolution(%v1238, %u16qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x7x7xf32>, tensor<256x1x3x3xf32>) -> tensor<32x256x7x7xf32>
    %v1240 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1241 = stablehlo.add %v1239, %v1240 : tensor<32x256x7x7xf32>
    %v1242 = stablehlo.reshape %v1241 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1243 = stablehlo.reshape %v1242 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1244 = stablehlo.broadcast_in_dim %u16qnmu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1245 = stablehlo.subtract %v1243, %v1244 : tensor<32x256x7x7xf32>
    %v1246 = stablehlo.broadcast_in_dim %u16qnvar, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1247 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1248 = stablehlo.add %v1246, %v1247 : tensor<32x256x7x7xf32>
    %v1249 = stablehlo.rsqrt %v1248 : tensor<32x256x7x7xf32>
    %v1250 = stablehlo.multiply %v1245, %v1249 : tensor<32x256x7x7xf32>
    %v1251 = stablehlo.broadcast_in_dim %u16qg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1252 = stablehlo.broadcast_in_dim %u16qbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1253 = stablehlo.multiply %v1250, %v1251 : tensor<32x256x7x7xf32>
    %v1254 = stablehlo.add %v1253, %v1252 : tensor<32x256x7x7xf32>
    %v1255 = stablehlo.reshape %v1254 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1256 = stablehlo.reshape %v1255 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1257 = stablehlo.constant dense<0.0> : tensor<32x256x7x7xf32>
    %v1258 = stablehlo.maximum %v1256, %v1257 : tensor<32x256x7x7xf32>
    %v1259 = stablehlo.reshape %v1258 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1260 = stablehlo.reshape %v1259 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1261 = stablehlo.convolution(%v1260, %u16eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v1262 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1263 = stablehlo.add %v1261, %v1262 : tensor<32x1024x7x7xf32>
    %v1264 = stablehlo.reshape %v1263 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1265 = stablehlo.reshape %v1264 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1266 = stablehlo.broadcast_in_dim %u16enmu, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1267 = stablehlo.subtract %v1265, %v1266 : tensor<32x1024x7x7xf32>
    %v1268 = stablehlo.broadcast_in_dim %u16envar, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1269 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1270 = stablehlo.add %v1268, %v1269 : tensor<32x1024x7x7xf32>
    %v1271 = stablehlo.rsqrt %v1270 : tensor<32x1024x7x7xf32>
    %v1272 = stablehlo.multiply %v1267, %v1271 : tensor<32x1024x7x7xf32>
    %v1273 = stablehlo.broadcast_in_dim %u16eg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1274 = stablehlo.broadcast_in_dim %u16ebt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1275 = stablehlo.multiply %v1272, %v1273 : tensor<32x1024x7x7xf32>
    %v1276 = stablehlo.add %v1275, %v1274 : tensor<32x1024x7x7xf32>
    %v1277 = stablehlo.reshape %v1276 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1278 = stablehlo.reshape %v1277 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1279 = stablehlo.constant dense<0.0> : tensor<32x1024x7x7xf32>
    %v1280 = stablehlo.maximum %v1278, %v1279 : tensor<32x1024x7x7xf32>
    %v1281 = stablehlo.reshape %v1280 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1282 = stablehlo.reshape %v1281 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1283 = stablehlo.convolution(%v1282, %u16pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1284 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1285 = stablehlo.add %v1283, %v1284 : tensor<32x256x7x7xf32>
    %v1286 = stablehlo.reshape %v1285 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1287 = stablehlo.reshape %v1286 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1288 = stablehlo.broadcast_in_dim %u16pnmu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1289 = stablehlo.subtract %v1287, %v1288 : tensor<32x256x7x7xf32>
    %v1290 = stablehlo.broadcast_in_dim %u16pnvar, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1291 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1292 = stablehlo.add %v1290, %v1291 : tensor<32x256x7x7xf32>
    %v1293 = stablehlo.rsqrt %v1292 : tensor<32x256x7x7xf32>
    %v1294 = stablehlo.multiply %v1289, %v1293 : tensor<32x256x7x7xf32>
    %v1295 = stablehlo.broadcast_in_dim %u16pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1296 = stablehlo.broadcast_in_dim %u16pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1297 = stablehlo.multiply %v1294, %v1295 : tensor<32x256x7x7xf32>
    %v1298 = stablehlo.add %v1297, %v1296 : tensor<32x256x7x7xf32>
    %v1299 = stablehlo.reshape %v1298 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1300 = stablehlo.reshape %v1299 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1301 = stablehlo.reshape %v1237 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1302 = stablehlo.add %v1300, %v1301 : tensor<32x256x7x7xf32>
    %v1303 = stablehlo.reshape %v1302 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1304 = stablehlo.reshape %v1303 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1305 = stablehlo.convolution(%v1304, %u17qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x7x7xf32>, tensor<256x1x3x3xf32>) -> tensor<32x256x7x7xf32>
    %v1306 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1307 = stablehlo.add %v1305, %v1306 : tensor<32x256x7x7xf32>
    %v1308 = stablehlo.reshape %v1307 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1309 = stablehlo.reshape %v1308 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1310 = stablehlo.broadcast_in_dim %u17qnmu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1311 = stablehlo.subtract %v1309, %v1310 : tensor<32x256x7x7xf32>
    %v1312 = stablehlo.broadcast_in_dim %u17qnvar, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1313 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1314 = stablehlo.add %v1312, %v1313 : tensor<32x256x7x7xf32>
    %v1315 = stablehlo.rsqrt %v1314 : tensor<32x256x7x7xf32>
    %v1316 = stablehlo.multiply %v1311, %v1315 : tensor<32x256x7x7xf32>
    %v1317 = stablehlo.broadcast_in_dim %u17qg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1318 = stablehlo.broadcast_in_dim %u17qbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1319 = stablehlo.multiply %v1316, %v1317 : tensor<32x256x7x7xf32>
    %v1320 = stablehlo.add %v1319, %v1318 : tensor<32x256x7x7xf32>
    %v1321 = stablehlo.reshape %v1320 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1322 = stablehlo.reshape %v1321 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1323 = stablehlo.constant dense<0.0> : tensor<32x256x7x7xf32>
    %v1324 = stablehlo.maximum %v1322, %v1323 : tensor<32x256x7x7xf32>
    %v1325 = stablehlo.reshape %v1324 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1326 = stablehlo.reshape %v1325 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1327 = stablehlo.convolution(%v1326, %u17eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<512x256x1x1xf32>) -> tensor<32x512x7x7xf32>
    %v1328 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1329 = stablehlo.add %v1327, %v1328 : tensor<32x512x7x7xf32>
    %v1330 = stablehlo.reshape %v1329 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1331 = stablehlo.reshape %v1330 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1332 = stablehlo.broadcast_in_dim %u17enmu, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1333 = stablehlo.subtract %v1331, %v1332 : tensor<32x512x7x7xf32>
    %v1334 = stablehlo.broadcast_in_dim %u17envar, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1335 = stablehlo.constant dense<1.0e-5> : tensor<32x512x7x7xf32>
    %v1336 = stablehlo.add %v1334, %v1335 : tensor<32x512x7x7xf32>
    %v1337 = stablehlo.rsqrt %v1336 : tensor<32x512x7x7xf32>
    %v1338 = stablehlo.multiply %v1333, %v1337 : tensor<32x512x7x7xf32>
    %v1339 = stablehlo.broadcast_in_dim %u17eg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1340 = stablehlo.broadcast_in_dim %u17ebt, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1341 = stablehlo.multiply %v1338, %v1339 : tensor<32x512x7x7xf32>
    %v1342 = stablehlo.add %v1341, %v1340 : tensor<32x512x7x7xf32>
    %v1343 = stablehlo.reshape %v1342 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1344 = stablehlo.reshape %v1343 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1345 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1346 = stablehlo.maximum %v1344, %v1345 : tensor<32x512x7x7xf32>
    %v1347 = stablehlo.reshape %v1346 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1348 = stablehlo.reshape %v1347 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1349 = stablehlo.convolution(%v1348, %u17dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<32x512x7x7xf32>, tensor<512x1x5x5xf32>) -> tensor<32x512x7x7xf32>
    %v1350 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1351 = stablehlo.add %v1349, %v1350 : tensor<32x512x7x7xf32>
    %v1352 = stablehlo.reshape %v1351 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1353 = stablehlo.reshape %v1352 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1354 = stablehlo.broadcast_in_dim %u17dnmu, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1355 = stablehlo.subtract %v1353, %v1354 : tensor<32x512x7x7xf32>
    %v1356 = stablehlo.broadcast_in_dim %u17dnvar, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1357 = stablehlo.constant dense<1.0e-5> : tensor<32x512x7x7xf32>
    %v1358 = stablehlo.add %v1356, %v1357 : tensor<32x512x7x7xf32>
    %v1359 = stablehlo.rsqrt %v1358 : tensor<32x512x7x7xf32>
    %v1360 = stablehlo.multiply %v1355, %v1359 : tensor<32x512x7x7xf32>
    %v1361 = stablehlo.broadcast_in_dim %u17dg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1362 = stablehlo.broadcast_in_dim %u17dbt, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1363 = stablehlo.multiply %v1360, %v1361 : tensor<32x512x7x7xf32>
    %v1364 = stablehlo.add %v1363, %v1362 : tensor<32x512x7x7xf32>
    %v1365 = stablehlo.reshape %v1364 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1366 = stablehlo.reshape %v1365 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1367 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1368 = stablehlo.maximum %v1366, %v1367 : tensor<32x512x7x7xf32>
    %v1369 = stablehlo.reshape %v1368 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1370 = stablehlo.reshape %v1369 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1371 = stablehlo.convolution(%v1370, %u17pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<256x512x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1372 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1373 = stablehlo.add %v1371, %v1372 : tensor<32x256x7x7xf32>
    %v1374 = stablehlo.reshape %v1373 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1375 = stablehlo.reshape %v1374 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1376 = stablehlo.broadcast_in_dim %u17pnmu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1377 = stablehlo.subtract %v1375, %v1376 : tensor<32x256x7x7xf32>
    %v1378 = stablehlo.broadcast_in_dim %u17pnvar, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1379 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1380 = stablehlo.add %v1378, %v1379 : tensor<32x256x7x7xf32>
    %v1381 = stablehlo.rsqrt %v1380 : tensor<32x256x7x7xf32>
    %v1382 = stablehlo.multiply %v1377, %v1381 : tensor<32x256x7x7xf32>
    %v1383 = stablehlo.broadcast_in_dim %u17pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1384 = stablehlo.broadcast_in_dim %u17pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1385 = stablehlo.multiply %v1382, %v1383 : tensor<32x256x7x7xf32>
    %v1386 = stablehlo.add %v1385, %v1384 : tensor<32x256x7x7xf32>
    %v1387 = stablehlo.reshape %v1386 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1388 = stablehlo.reshape %v1387 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1389 = stablehlo.reshape %v1303 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1390 = stablehlo.add %v1388, %v1389 : tensor<32x256x7x7xf32>
    %v1391 = stablehlo.reshape %v1390 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1392 = stablehlo.reshape %v1391 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1393 = stablehlo.convolution(%v1392, %u18qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x7x7xf32>, tensor<256x1x5x5xf32>) -> tensor<32x256x7x7xf32>
    %v1394 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1395 = stablehlo.add %v1393, %v1394 : tensor<32x256x7x7xf32>
    %v1396 = stablehlo.reshape %v1395 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1397 = stablehlo.reshape %v1396 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1398 = stablehlo.broadcast_in_dim %u18qnmu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1399 = stablehlo.subtract %v1397, %v1398 : tensor<32x256x7x7xf32>
    %v1400 = stablehlo.broadcast_in_dim %u18qnvar, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1401 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1402 = stablehlo.add %v1400, %v1401 : tensor<32x256x7x7xf32>
    %v1403 = stablehlo.rsqrt %v1402 : tensor<32x256x7x7xf32>
    %v1404 = stablehlo.multiply %v1399, %v1403 : tensor<32x256x7x7xf32>
    %v1405 = stablehlo.broadcast_in_dim %u18qg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1406 = stablehlo.broadcast_in_dim %u18qbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1407 = stablehlo.multiply %v1404, %v1405 : tensor<32x256x7x7xf32>
    %v1408 = stablehlo.add %v1407, %v1406 : tensor<32x256x7x7xf32>
    %v1409 = stablehlo.reshape %v1408 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1410 = stablehlo.reshape %v1409 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1411 = stablehlo.constant dense<0.0> : tensor<32x256x7x7xf32>
    %v1412 = stablehlo.maximum %v1410, %v1411 : tensor<32x256x7x7xf32>
    %v1413 = stablehlo.reshape %v1412 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1414 = stablehlo.reshape %v1413 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1415 = stablehlo.convolution(%v1414, %u18eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v1416 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1417 = stablehlo.add %v1415, %v1416 : tensor<32x1024x7x7xf32>
    %v1418 = stablehlo.reshape %v1417 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1419 = stablehlo.reshape %v1418 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1420 = stablehlo.broadcast_in_dim %u18enmu, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1421 = stablehlo.subtract %v1419, %v1420 : tensor<32x1024x7x7xf32>
    %v1422 = stablehlo.broadcast_in_dim %u18envar, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1423 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1424 = stablehlo.add %v1422, %v1423 : tensor<32x1024x7x7xf32>
    %v1425 = stablehlo.rsqrt %v1424 : tensor<32x1024x7x7xf32>
    %v1426 = stablehlo.multiply %v1421, %v1425 : tensor<32x1024x7x7xf32>
    %v1427 = stablehlo.broadcast_in_dim %u18eg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1428 = stablehlo.broadcast_in_dim %u18ebt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1429 = stablehlo.multiply %v1426, %v1427 : tensor<32x1024x7x7xf32>
    %v1430 = stablehlo.add %v1429, %v1428 : tensor<32x1024x7x7xf32>
    %v1431 = stablehlo.reshape %v1430 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1432 = stablehlo.reshape %v1431 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1433 = stablehlo.constant dense<0.0> : tensor<32x1024x7x7xf32>
    %v1434 = stablehlo.maximum %v1432, %v1433 : tensor<32x1024x7x7xf32>
    %v1435 = stablehlo.reshape %v1434 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1436 = stablehlo.reshape %v1435 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1437 = stablehlo.convolution(%v1436, %u18dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<32x1024x7x7xf32>, tensor<1024x1x5x5xf32>) -> tensor<32x1024x7x7xf32>
    %v1438 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1439 = stablehlo.add %v1437, %v1438 : tensor<32x1024x7x7xf32>
    %v1440 = stablehlo.reshape %v1439 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1441 = stablehlo.reshape %v1440 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1442 = stablehlo.broadcast_in_dim %u18dnmu, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1443 = stablehlo.subtract %v1441, %v1442 : tensor<32x1024x7x7xf32>
    %v1444 = stablehlo.broadcast_in_dim %u18dnvar, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1445 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1446 = stablehlo.add %v1444, %v1445 : tensor<32x1024x7x7xf32>
    %v1447 = stablehlo.rsqrt %v1446 : tensor<32x1024x7x7xf32>
    %v1448 = stablehlo.multiply %v1443, %v1447 : tensor<32x1024x7x7xf32>
    %v1449 = stablehlo.broadcast_in_dim %u18dg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1450 = stablehlo.broadcast_in_dim %u18dbt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1451 = stablehlo.multiply %v1448, %v1449 : tensor<32x1024x7x7xf32>
    %v1452 = stablehlo.add %v1451, %v1450 : tensor<32x1024x7x7xf32>
    %v1453 = stablehlo.reshape %v1452 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1454 = stablehlo.reshape %v1453 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1455 = stablehlo.constant dense<0.0> : tensor<32x1024x7x7xf32>
    %v1456 = stablehlo.maximum %v1454, %v1455 : tensor<32x1024x7x7xf32>
    %v1457 = stablehlo.reshape %v1456 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1458 = stablehlo.reshape %v1457 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1459 = stablehlo.convolution(%v1458, %u18pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1460 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1461 = stablehlo.add %v1459, %v1460 : tensor<32x256x7x7xf32>
    %v1462 = stablehlo.reshape %v1461 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1463 = stablehlo.reshape %v1462 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1464 = stablehlo.broadcast_in_dim %u18pnmu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1465 = stablehlo.subtract %v1463, %v1464 : tensor<32x256x7x7xf32>
    %v1466 = stablehlo.broadcast_in_dim %u18pnvar, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1467 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1468 = stablehlo.add %v1466, %v1467 : tensor<32x256x7x7xf32>
    %v1469 = stablehlo.rsqrt %v1468 : tensor<32x256x7x7xf32>
    %v1470 = stablehlo.multiply %v1465, %v1469 : tensor<32x256x7x7xf32>
    %v1471 = stablehlo.broadcast_in_dim %u18pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1472 = stablehlo.broadcast_in_dim %u18pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1473 = stablehlo.multiply %v1470, %v1471 : tensor<32x256x7x7xf32>
    %v1474 = stablehlo.add %v1473, %v1472 : tensor<32x256x7x7xf32>
    %v1475 = stablehlo.reshape %v1474 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1476 = stablehlo.reshape %v1475 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1477 = stablehlo.reshape %v1391 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1478 = stablehlo.add %v1476, %v1477 : tensor<32x256x7x7xf32>
    %v1479 = stablehlo.reshape %v1478 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1480 = stablehlo.reshape %v1479 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1481 = stablehlo.convolution(%v1480, %u19eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v1482 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1483 = stablehlo.add %v1481, %v1482 : tensor<32x1024x7x7xf32>
    %v1484 = stablehlo.reshape %v1483 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1485 = stablehlo.reshape %v1484 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1486 = stablehlo.broadcast_in_dim %u19enmu, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1487 = stablehlo.subtract %v1485, %v1486 : tensor<32x1024x7x7xf32>
    %v1488 = stablehlo.broadcast_in_dim %u19envar, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1489 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1490 = stablehlo.add %v1488, %v1489 : tensor<32x1024x7x7xf32>
    %v1491 = stablehlo.rsqrt %v1490 : tensor<32x1024x7x7xf32>
    %v1492 = stablehlo.multiply %v1487, %v1491 : tensor<32x1024x7x7xf32>
    %v1493 = stablehlo.broadcast_in_dim %u19eg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1494 = stablehlo.broadcast_in_dim %u19ebt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1495 = stablehlo.multiply %v1492, %v1493 : tensor<32x1024x7x7xf32>
    %v1496 = stablehlo.add %v1495, %v1494 : tensor<32x1024x7x7xf32>
    %v1497 = stablehlo.reshape %v1496 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1498 = stablehlo.reshape %v1497 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1499 = stablehlo.constant dense<0.0> : tensor<32x1024x7x7xf32>
    %v1500 = stablehlo.maximum %v1498, %v1499 : tensor<32x1024x7x7xf32>
    %v1501 = stablehlo.reshape %v1500 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1502 = stablehlo.reshape %v1501 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1503 = stablehlo.convolution(%v1502, %u19pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1504 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1505 = stablehlo.add %v1503, %v1504 : tensor<32x256x7x7xf32>
    %v1506 = stablehlo.reshape %v1505 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1507 = stablehlo.reshape %v1506 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1508 = stablehlo.broadcast_in_dim %u19pnmu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1509 = stablehlo.subtract %v1507, %v1508 : tensor<32x256x7x7xf32>
    %v1510 = stablehlo.broadcast_in_dim %u19pnvar, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1511 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1512 = stablehlo.add %v1510, %v1511 : tensor<32x256x7x7xf32>
    %v1513 = stablehlo.rsqrt %v1512 : tensor<32x256x7x7xf32>
    %v1514 = stablehlo.multiply %v1509, %v1513 : tensor<32x256x7x7xf32>
    %v1515 = stablehlo.broadcast_in_dim %u19pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1516 = stablehlo.broadcast_in_dim %u19pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1517 = stablehlo.multiply %v1514, %v1515 : tensor<32x256x7x7xf32>
    %v1518 = stablehlo.add %v1517, %v1516 : tensor<32x256x7x7xf32>
    %v1519 = stablehlo.reshape %v1518 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1520 = stablehlo.reshape %v1519 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1521 = stablehlo.reshape %v1479 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1522 = stablehlo.add %v1520, %v1521 : tensor<32x256x7x7xf32>
    %v1523 = stablehlo.reshape %v1522 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1524 = stablehlo.reshape %v1523 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1525 = stablehlo.convolution(%v1524, %u20eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<32x1024x7x7xf32>
    %v1526 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1527 = stablehlo.add %v1525, %v1526 : tensor<32x1024x7x7xf32>
    %v1528 = stablehlo.reshape %v1527 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1529 = stablehlo.reshape %v1528 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1530 = stablehlo.broadcast_in_dim %u20enmu, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1531 = stablehlo.subtract %v1529, %v1530 : tensor<32x1024x7x7xf32>
    %v1532 = stablehlo.broadcast_in_dim %u20envar, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1533 = stablehlo.constant dense<1.0e-5> : tensor<32x1024x7x7xf32>
    %v1534 = stablehlo.add %v1532, %v1533 : tensor<32x1024x7x7xf32>
    %v1535 = stablehlo.rsqrt %v1534 : tensor<32x1024x7x7xf32>
    %v1536 = stablehlo.multiply %v1531, %v1535 : tensor<32x1024x7x7xf32>
    %v1537 = stablehlo.broadcast_in_dim %u20eg, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1538 = stablehlo.broadcast_in_dim %u20ebt, dims = [1] : (tensor<1024xf32>) -> tensor<32x1024x7x7xf32>
    %v1539 = stablehlo.multiply %v1536, %v1537 : tensor<32x1024x7x7xf32>
    %v1540 = stablehlo.add %v1539, %v1538 : tensor<32x1024x7x7xf32>
    %v1541 = stablehlo.reshape %v1540 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1542 = stablehlo.reshape %v1541 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1543 = stablehlo.constant dense<0.0> : tensor<32x1024x7x7xf32>
    %v1544 = stablehlo.maximum %v1542, %v1543 : tensor<32x1024x7x7xf32>
    %v1545 = stablehlo.reshape %v1544 : (tensor<32x1024x7x7xf32>) -> tensor<32x50176xf32>
    %v1546 = stablehlo.reshape %v1545 : (tensor<32x50176xf32>) -> tensor<32x1024x7x7xf32>
    %v1547 = stablehlo.convolution(%v1546, %u20pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1548 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1549 = stablehlo.add %v1547, %v1548 : tensor<32x256x7x7xf32>
    %v1550 = stablehlo.reshape %v1549 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1551 = stablehlo.reshape %v1550 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1552 = stablehlo.broadcast_in_dim %u20pnmu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1553 = stablehlo.subtract %v1551, %v1552 : tensor<32x256x7x7xf32>
    %v1554 = stablehlo.broadcast_in_dim %u20pnvar, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1555 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1556 = stablehlo.add %v1554, %v1555 : tensor<32x256x7x7xf32>
    %v1557 = stablehlo.rsqrt %v1556 : tensor<32x256x7x7xf32>
    %v1558 = stablehlo.multiply %v1553, %v1557 : tensor<32x256x7x7xf32>
    %v1559 = stablehlo.broadcast_in_dim %u20pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1560 = stablehlo.broadcast_in_dim %u20pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1561 = stablehlo.multiply %v1558, %v1559 : tensor<32x256x7x7xf32>
    %v1562 = stablehlo.add %v1561, %v1560 : tensor<32x256x7x7xf32>
    %v1563 = stablehlo.reshape %v1562 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1564 = stablehlo.reshape %v1563 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1565 = stablehlo.reshape %v1523 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1566 = stablehlo.add %v1564, %v1565 : tensor<32x256x7x7xf32>
    %v1567 = stablehlo.reshape %v1566 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1568 = stablehlo.reshape %v1567 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1569 = stablehlo.convolution(%v1568, %u21qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<32x256x7x7xf32>, tensor<256x1x5x5xf32>) -> tensor<32x256x7x7xf32>
    %v1570 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1571 = stablehlo.add %v1569, %v1570 : tensor<32x256x7x7xf32>
    %v1572 = stablehlo.reshape %v1571 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1573 = stablehlo.reshape %v1572 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1574 = stablehlo.broadcast_in_dim %u21qnmu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1575 = stablehlo.subtract %v1573, %v1574 : tensor<32x256x7x7xf32>
    %v1576 = stablehlo.broadcast_in_dim %u21qnvar, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1577 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1578 = stablehlo.add %v1576, %v1577 : tensor<32x256x7x7xf32>
    %v1579 = stablehlo.rsqrt %v1578 : tensor<32x256x7x7xf32>
    %v1580 = stablehlo.multiply %v1575, %v1579 : tensor<32x256x7x7xf32>
    %v1581 = stablehlo.broadcast_in_dim %u21qg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1582 = stablehlo.broadcast_in_dim %u21qbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1583 = stablehlo.multiply %v1580, %v1581 : tensor<32x256x7x7xf32>
    %v1584 = stablehlo.add %v1583, %v1582 : tensor<32x256x7x7xf32>
    %v1585 = stablehlo.reshape %v1584 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1586 = stablehlo.reshape %v1585 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1587 = stablehlo.constant dense<0.0> : tensor<32x256x7x7xf32>
    %v1588 = stablehlo.maximum %v1586, %v1587 : tensor<32x256x7x7xf32>
    %v1589 = stablehlo.reshape %v1588 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1590 = stablehlo.reshape %v1589 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1591 = stablehlo.convolution(%v1590, %u21eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<512x256x1x1xf32>) -> tensor<32x512x7x7xf32>
    %v1592 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1593 = stablehlo.add %v1591, %v1592 : tensor<32x512x7x7xf32>
    %v1594 = stablehlo.reshape %v1593 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1595 = stablehlo.reshape %v1594 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1596 = stablehlo.broadcast_in_dim %u21enmu, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1597 = stablehlo.subtract %v1595, %v1596 : tensor<32x512x7x7xf32>
    %v1598 = stablehlo.broadcast_in_dim %u21envar, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1599 = stablehlo.constant dense<1.0e-5> : tensor<32x512x7x7xf32>
    %v1600 = stablehlo.add %v1598, %v1599 : tensor<32x512x7x7xf32>
    %v1601 = stablehlo.rsqrt %v1600 : tensor<32x512x7x7xf32>
    %v1602 = stablehlo.multiply %v1597, %v1601 : tensor<32x512x7x7xf32>
    %v1603 = stablehlo.broadcast_in_dim %u21eg, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1604 = stablehlo.broadcast_in_dim %u21ebt, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1605 = stablehlo.multiply %v1602, %v1603 : tensor<32x512x7x7xf32>
    %v1606 = stablehlo.add %v1605, %v1604 : tensor<32x512x7x7xf32>
    %v1607 = stablehlo.reshape %v1606 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1608 = stablehlo.reshape %v1607 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1609 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1610 = stablehlo.maximum %v1608, %v1609 : tensor<32x512x7x7xf32>
    %v1611 = stablehlo.reshape %v1610 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1612 = stablehlo.reshape %v1611 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1613 = stablehlo.convolution(%v1612, %u21pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<256x512x1x1xf32>) -> tensor<32x256x7x7xf32>
    %v1614 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1615 = stablehlo.add %v1613, %v1614 : tensor<32x256x7x7xf32>
    %v1616 = stablehlo.reshape %v1615 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1617 = stablehlo.reshape %v1616 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1618 = stablehlo.broadcast_in_dim %u21pnmu, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1619 = stablehlo.subtract %v1617, %v1618 : tensor<32x256x7x7xf32>
    %v1620 = stablehlo.broadcast_in_dim %u21pnvar, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1621 = stablehlo.constant dense<1.0e-5> : tensor<32x256x7x7xf32>
    %v1622 = stablehlo.add %v1620, %v1621 : tensor<32x256x7x7xf32>
    %v1623 = stablehlo.rsqrt %v1622 : tensor<32x256x7x7xf32>
    %v1624 = stablehlo.multiply %v1619, %v1623 : tensor<32x256x7x7xf32>
    %v1625 = stablehlo.broadcast_in_dim %u21pg, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1626 = stablehlo.broadcast_in_dim %u21pbt, dims = [1] : (tensor<256xf32>) -> tensor<32x256x7x7xf32>
    %v1627 = stablehlo.multiply %v1624, %v1625 : tensor<32x256x7x7xf32>
    %v1628 = stablehlo.add %v1627, %v1626 : tensor<32x256x7x7xf32>
    %v1629 = stablehlo.reshape %v1628 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1630 = stablehlo.reshape %v1629 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1631 = stablehlo.reshape %v1567 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1632 = stablehlo.add %v1630, %v1631 : tensor<32x256x7x7xf32>
    %v1633 = stablehlo.reshape %v1632 : (tensor<32x256x7x7xf32>) -> tensor<32x12544xf32>
    %v1634 = stablehlo.reshape %v1633 : (tensor<32x12544xf32>) -> tensor<32x256x7x7xf32>
    %v1635 = stablehlo.convolution(%v1634, %h1W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x7x7xf32>, tensor<960x256x1x1xf32>) -> tensor<32x960x7x7xf32>
    %v1636 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1637 = stablehlo.add %v1635, %v1636 : tensor<32x960x7x7xf32>
    %v1638 = stablehlo.reshape %v1637 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1639 = stablehlo.reshape %v1638 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1640 = stablehlo.broadcast_in_dim %h1nmu, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1641 = stablehlo.subtract %v1639, %v1640 : tensor<32x960x7x7xf32>
    %v1642 = stablehlo.broadcast_in_dim %h1nvar, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1643 = stablehlo.constant dense<1.0e-5> : tensor<32x960x7x7xf32>
    %v1644 = stablehlo.add %v1642, %v1643 : tensor<32x960x7x7xf32>
    %v1645 = stablehlo.rsqrt %v1644 : tensor<32x960x7x7xf32>
    %v1646 = stablehlo.multiply %v1641, %v1645 : tensor<32x960x7x7xf32>
    %v1647 = stablehlo.broadcast_in_dim %h1g, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1648 = stablehlo.broadcast_in_dim %h1bt, dims = [1] : (tensor<960xf32>) -> tensor<32x960x7x7xf32>
    %v1649 = stablehlo.multiply %v1646, %v1647 : tensor<32x960x7x7xf32>
    %v1650 = stablehlo.add %v1649, %v1648 : tensor<32x960x7x7xf32>
    %v1651 = stablehlo.reshape %v1650 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1652 = stablehlo.reshape %v1651 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1653 = stablehlo.constant dense<0.0> : tensor<32x960x7x7xf32>
    %v1654 = stablehlo.maximum %v1652, %v1653 : tensor<32x960x7x7xf32>
    %v1655 = stablehlo.reshape %v1654 : (tensor<32x960x7x7xf32>) -> tensor<32x47040xf32>
    %v1656 = stablehlo.reshape %v1655 : (tensor<32x47040xf32>) -> tensor<32x960x7x7xf32>
    %v1657 = stablehlo.convolution(%v1656, %hW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x960x7x7xf32>, tensor<1280x960x1x1xf32>) -> tensor<32x1280x7x7xf32>
    %v1658 = stablehlo.broadcast_in_dim %zb1280, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1659 = stablehlo.add %v1657, %v1658 : tensor<32x1280x7x7xf32>
    %v1660 = stablehlo.reshape %v1659 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1661 = stablehlo.reshape %v1660 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1662 = stablehlo.broadcast_in_dim %hnmu, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1663 = stablehlo.subtract %v1661, %v1662 : tensor<32x1280x7x7xf32>
    %v1664 = stablehlo.broadcast_in_dim %hnvar, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1665 = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %v1666 = stablehlo.add %v1664, %v1665 : tensor<32x1280x7x7xf32>
    %v1667 = stablehlo.rsqrt %v1666 : tensor<32x1280x7x7xf32>
    %v1668 = stablehlo.multiply %v1663, %v1667 : tensor<32x1280x7x7xf32>
    %v1669 = stablehlo.broadcast_in_dim %hg, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1670 = stablehlo.broadcast_in_dim %hbt, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1671 = stablehlo.multiply %v1668, %v1669 : tensor<32x1280x7x7xf32>
    %v1672 = stablehlo.add %v1671, %v1670 : tensor<32x1280x7x7xf32>
    %v1673 = stablehlo.reshape %v1672 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1674 = stablehlo.reshape %v1673 : (tensor<32x62720xf32>) -> tensor<32x80x28x28xf32>
    %v1675 = stablehlo.constant dense<0.0> : tensor<32x80x28x28xf32>
    %v1676 = stablehlo.maximum %v1674, %v1675 : tensor<32x80x28x28xf32>
    %v1677 = stablehlo.reshape %v1676 : (tensor<32x80x28x28xf32>) -> tensor<32x62720xf32>
    %v1678 = stablehlo.reshape %v1677 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1679 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1680 = stablehlo.reduce(%v1678 init: %v1679) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v1681 = stablehlo.constant dense<49.0> : tensor<32x1280xf32>
    %v1682 = stablehlo.divide %v1680, %v1681 : tensor<32x1280xf32>
    %v1683 = stablehlo.dot_general %v1682, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1280xf32>, tensor<1280x10xf32>) -> tensor<32x10xf32>
    %v1684 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v1685 = stablehlo.add %v1683, %v1684 : tensor<32x10xf32>
    return %v1685 : tensor<32x10xf32>
  }
}
