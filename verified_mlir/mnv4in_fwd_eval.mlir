module @m {
  func.func @mnv4in_fwd_eval(%x: tensor<64x150528xf32>, %sW: tensor<32x3x3x3xf32>, %sg: tensor<32xf32>, %sbt: tensor<32xf32>, %f0cW: tensor<128x32x3x3xf32>, %f0cg: tensor<128xf32>, %f0cbt: tensor<128xf32>, %f0pW: tensor<48x128x1x1xf32>, %f0pg: tensor<48xf32>, %f0pbt: tensor<48xf32>, %u1qW: tensor<48x1x3x3xf32>, %u1qg: tensor<48xf32>, %u1qbt: tensor<48xf32>, %u1eW: tensor<192x48x1x1xf32>, %u1eg: tensor<192xf32>, %u1ebt: tensor<192xf32>, %u1dW: tensor<192x1x5x5xf32>, %u1dg: tensor<192xf32>, %u1dbt: tensor<192xf32>, %u1pW: tensor<80x192x1x1xf32>, %u1pg: tensor<80xf32>, %u1pbt: tensor<80xf32>, %u2qW: tensor<80x1x3x3xf32>, %u2qg: tensor<80xf32>, %u2qbt: tensor<80xf32>, %u2eW: tensor<160x80x1x1xf32>, %u2eg: tensor<160xf32>, %u2ebt: tensor<160xf32>, %u2dW: tensor<160x1x3x3xf32>, %u2dg: tensor<160xf32>, %u2dbt: tensor<160xf32>, %u2pW: tensor<80x160x1x1xf32>, %u2pg: tensor<80xf32>, %u2pbt: tensor<80xf32>, %u3qW: tensor<80x1x3x3xf32>, %u3qg: tensor<80xf32>, %u3qbt: tensor<80xf32>, %u3eW: tensor<480x80x1x1xf32>, %u3eg: tensor<480xf32>, %u3ebt: tensor<480xf32>, %u3dW: tensor<480x1x5x5xf32>, %u3dg: tensor<480xf32>, %u3dbt: tensor<480xf32>, %u3pW: tensor<160x480x1x1xf32>, %u3pg: tensor<160xf32>, %u3pbt: tensor<160xf32>, %u4qW: tensor<160x1x3x3xf32>, %u4qg: tensor<160xf32>, %u4qbt: tensor<160xf32>, %u4eW: tensor<640x160x1x1xf32>, %u4eg: tensor<640xf32>, %u4ebt: tensor<640xf32>, %u4dW: tensor<640x1x3x3xf32>, %u4dg: tensor<640xf32>, %u4dbt: tensor<640xf32>, %u4pW: tensor<160x640x1x1xf32>, %u4pg: tensor<160xf32>, %u4pbt: tensor<160xf32>, %u5qW: tensor<160x1x3x3xf32>, %u5qg: tensor<160xf32>, %u5qbt: tensor<160xf32>, %u5eW: tensor<640x160x1x1xf32>, %u5eg: tensor<640xf32>, %u5ebt: tensor<640xf32>, %u5dW: tensor<640x1x3x3xf32>, %u5dg: tensor<640xf32>, %u5dbt: tensor<640xf32>, %u5pW: tensor<160x640x1x1xf32>, %u5pg: tensor<160xf32>, %u5pbt: tensor<160xf32>, %u6qW: tensor<160x1x3x3xf32>, %u6qg: tensor<160xf32>, %u6qbt: tensor<160xf32>, %u6eW: tensor<640x160x1x1xf32>, %u6eg: tensor<640xf32>, %u6ebt: tensor<640xf32>, %u6dW: tensor<640x1x5x5xf32>, %u6dg: tensor<640xf32>, %u6dbt: tensor<640xf32>, %u6pW: tensor<160x640x1x1xf32>, %u6pg: tensor<160xf32>, %u6pbt: tensor<160xf32>, %u7qW: tensor<160x1x3x3xf32>, %u7qg: tensor<160xf32>, %u7qbt: tensor<160xf32>, %u7eW: tensor<640x160x1x1xf32>, %u7eg: tensor<640xf32>, %u7ebt: tensor<640xf32>, %u7dW: tensor<640x1x3x3xf32>, %u7dg: tensor<640xf32>, %u7dbt: tensor<640xf32>, %u7pW: tensor<160x640x1x1xf32>, %u7pg: tensor<160xf32>, %u7pbt: tensor<160xf32>, %u8qW: tensor<160x1x3x3xf32>, %u8qg: tensor<160xf32>, %u8qbt: tensor<160xf32>, %u8eW: tensor<640x160x1x1xf32>, %u8eg: tensor<640xf32>, %u8ebt: tensor<640xf32>, %u8pW: tensor<160x640x1x1xf32>, %u8pg: tensor<160xf32>, %u8pbt: tensor<160xf32>, %u9eW: tensor<320x160x1x1xf32>, %u9eg: tensor<320xf32>, %u9ebt: tensor<320xf32>, %u9pW: tensor<160x320x1x1xf32>, %u9pg: tensor<160xf32>, %u9pbt: tensor<160xf32>, %u10qW: tensor<160x1x3x3xf32>, %u10qg: tensor<160xf32>, %u10qbt: tensor<160xf32>, %u10eW: tensor<640x160x1x1xf32>, %u10eg: tensor<640xf32>, %u10ebt: tensor<640xf32>, %u10pW: tensor<160x640x1x1xf32>, %u10pg: tensor<160xf32>, %u10pbt: tensor<160xf32>, %u11qW: tensor<160x1x5x5xf32>, %u11qg: tensor<160xf32>, %u11qbt: tensor<160xf32>, %u11eW: tensor<960x160x1x1xf32>, %u11eg: tensor<960xf32>, %u11ebt: tensor<960xf32>, %u11dW: tensor<960x1x5x5xf32>, %u11dg: tensor<960xf32>, %u11dbt: tensor<960xf32>, %u11pW: tensor<256x960x1x1xf32>, %u11pg: tensor<256xf32>, %u11pbt: tensor<256xf32>, %u12qW: tensor<256x1x5x5xf32>, %u12qg: tensor<256xf32>, %u12qbt: tensor<256xf32>, %u12eW: tensor<1024x256x1x1xf32>, %u12eg: tensor<1024xf32>, %u12ebt: tensor<1024xf32>, %u12dW: tensor<1024x1x5x5xf32>, %u12dg: tensor<1024xf32>, %u12dbt: tensor<1024xf32>, %u12pW: tensor<256x1024x1x1xf32>, %u12pg: tensor<256xf32>, %u12pbt: tensor<256xf32>, %u13qW: tensor<256x1x3x3xf32>, %u13qg: tensor<256xf32>, %u13qbt: tensor<256xf32>, %u13eW: tensor<1024x256x1x1xf32>, %u13eg: tensor<1024xf32>, %u13ebt: tensor<1024xf32>, %u13dW: tensor<1024x1x5x5xf32>, %u13dg: tensor<1024xf32>, %u13dbt: tensor<1024xf32>, %u13pW: tensor<256x1024x1x1xf32>, %u13pg: tensor<256xf32>, %u13pbt: tensor<256xf32>, %u14qW: tensor<256x1x3x3xf32>, %u14qg: tensor<256xf32>, %u14qbt: tensor<256xf32>, %u14eW: tensor<1024x256x1x1xf32>, %u14eg: tensor<1024xf32>, %u14ebt: tensor<1024xf32>, %u14dW: tensor<1024x1x5x5xf32>, %u14dg: tensor<1024xf32>, %u14dbt: tensor<1024xf32>, %u14pW: tensor<256x1024x1x1xf32>, %u14pg: tensor<256xf32>, %u14pbt: tensor<256xf32>, %u15eW: tensor<1024x256x1x1xf32>, %u15eg: tensor<1024xf32>, %u15ebt: tensor<1024xf32>, %u15pW: tensor<256x1024x1x1xf32>, %u15pg: tensor<256xf32>, %u15pbt: tensor<256xf32>, %u16qW: tensor<256x1x3x3xf32>, %u16qg: tensor<256xf32>, %u16qbt: tensor<256xf32>, %u16eW: tensor<1024x256x1x1xf32>, %u16eg: tensor<1024xf32>, %u16ebt: tensor<1024xf32>, %u16pW: tensor<256x1024x1x1xf32>, %u16pg: tensor<256xf32>, %u16pbt: tensor<256xf32>, %u17qW: tensor<256x1x3x3xf32>, %u17qg: tensor<256xf32>, %u17qbt: tensor<256xf32>, %u17eW: tensor<512x256x1x1xf32>, %u17eg: tensor<512xf32>, %u17ebt: tensor<512xf32>, %u17dW: tensor<512x1x5x5xf32>, %u17dg: tensor<512xf32>, %u17dbt: tensor<512xf32>, %u17pW: tensor<256x512x1x1xf32>, %u17pg: tensor<256xf32>, %u17pbt: tensor<256xf32>, %u18qW: tensor<256x1x5x5xf32>, %u18qg: tensor<256xf32>, %u18qbt: tensor<256xf32>, %u18eW: tensor<1024x256x1x1xf32>, %u18eg: tensor<1024xf32>, %u18ebt: tensor<1024xf32>, %u18dW: tensor<1024x1x5x5xf32>, %u18dg: tensor<1024xf32>, %u18dbt: tensor<1024xf32>, %u18pW: tensor<256x1024x1x1xf32>, %u18pg: tensor<256xf32>, %u18pbt: tensor<256xf32>, %u19eW: tensor<1024x256x1x1xf32>, %u19eg: tensor<1024xf32>, %u19ebt: tensor<1024xf32>, %u19pW: tensor<256x1024x1x1xf32>, %u19pg: tensor<256xf32>, %u19pbt: tensor<256xf32>, %u20eW: tensor<1024x256x1x1xf32>, %u20eg: tensor<1024xf32>, %u20ebt: tensor<1024xf32>, %u20pW: tensor<256x1024x1x1xf32>, %u20pg: tensor<256xf32>, %u20pbt: tensor<256xf32>, %u21qW: tensor<256x1x5x5xf32>, %u21qg: tensor<256xf32>, %u21qbt: tensor<256xf32>, %u21eW: tensor<512x256x1x1xf32>, %u21eg: tensor<512xf32>, %u21ebt: tensor<512xf32>, %u21pW: tensor<256x512x1x1xf32>, %u21pg: tensor<256xf32>, %u21pbt: tensor<256xf32>, %h1W: tensor<960x256x1x1xf32>, %h1g: tensor<960xf32>, %h1bt: tensor<960xf32>, %hW: tensor<1280x960x1x1xf32>, %hg: tensor<1280xf32>, %hbt: tensor<1280xf32>, %Wd: tensor<1280x1000xf32>, %bd: tensor<1000xf32>, %stnmu: tensor<32xf32>, %stnvar: tensor<32xf32>, %f0cnmu: tensor<128xf32>, %f0cnvar: tensor<128xf32>, %f0pnmu: tensor<48xf32>, %f0pnvar: tensor<48xf32>, %u1qnmu: tensor<48xf32>, %u1qnvar: tensor<48xf32>, %u1enmu: tensor<192xf32>, %u1envar: tensor<192xf32>, %u1dnmu: tensor<192xf32>, %u1dnvar: tensor<192xf32>, %u1pnmu: tensor<80xf32>, %u1pnvar: tensor<80xf32>, %u2qnmu: tensor<80xf32>, %u2qnvar: tensor<80xf32>, %u2enmu: tensor<160xf32>, %u2envar: tensor<160xf32>, %u2dnmu: tensor<160xf32>, %u2dnvar: tensor<160xf32>, %u2pnmu: tensor<80xf32>, %u2pnvar: tensor<80xf32>, %u3qnmu: tensor<80xf32>, %u3qnvar: tensor<80xf32>, %u3enmu: tensor<480xf32>, %u3envar: tensor<480xf32>, %u3dnmu: tensor<480xf32>, %u3dnvar: tensor<480xf32>, %u3pnmu: tensor<160xf32>, %u3pnvar: tensor<160xf32>, %u4qnmu: tensor<160xf32>, %u4qnvar: tensor<160xf32>, %u4enmu: tensor<640xf32>, %u4envar: tensor<640xf32>, %u4dnmu: tensor<640xf32>, %u4dnvar: tensor<640xf32>, %u4pnmu: tensor<160xf32>, %u4pnvar: tensor<160xf32>, %u5qnmu: tensor<160xf32>, %u5qnvar: tensor<160xf32>, %u5enmu: tensor<640xf32>, %u5envar: tensor<640xf32>, %u5dnmu: tensor<640xf32>, %u5dnvar: tensor<640xf32>, %u5pnmu: tensor<160xf32>, %u5pnvar: tensor<160xf32>, %u6qnmu: tensor<160xf32>, %u6qnvar: tensor<160xf32>, %u6enmu: tensor<640xf32>, %u6envar: tensor<640xf32>, %u6dnmu: tensor<640xf32>, %u6dnvar: tensor<640xf32>, %u6pnmu: tensor<160xf32>, %u6pnvar: tensor<160xf32>, %u7qnmu: tensor<160xf32>, %u7qnvar: tensor<160xf32>, %u7enmu: tensor<640xf32>, %u7envar: tensor<640xf32>, %u7dnmu: tensor<640xf32>, %u7dnvar: tensor<640xf32>, %u7pnmu: tensor<160xf32>, %u7pnvar: tensor<160xf32>, %u8qnmu: tensor<160xf32>, %u8qnvar: tensor<160xf32>, %u8enmu: tensor<640xf32>, %u8envar: tensor<640xf32>, %u8pnmu: tensor<160xf32>, %u8pnvar: tensor<160xf32>, %u9enmu: tensor<320xf32>, %u9envar: tensor<320xf32>, %u9pnmu: tensor<160xf32>, %u9pnvar: tensor<160xf32>, %u10qnmu: tensor<160xf32>, %u10qnvar: tensor<160xf32>, %u10enmu: tensor<640xf32>, %u10envar: tensor<640xf32>, %u10pnmu: tensor<160xf32>, %u10pnvar: tensor<160xf32>, %u11qnmu: tensor<160xf32>, %u11qnvar: tensor<160xf32>, %u11enmu: tensor<960xf32>, %u11envar: tensor<960xf32>, %u11dnmu: tensor<960xf32>, %u11dnvar: tensor<960xf32>, %u11pnmu: tensor<256xf32>, %u11pnvar: tensor<256xf32>, %u12qnmu: tensor<256xf32>, %u12qnvar: tensor<256xf32>, %u12enmu: tensor<1024xf32>, %u12envar: tensor<1024xf32>, %u12dnmu: tensor<1024xf32>, %u12dnvar: tensor<1024xf32>, %u12pnmu: tensor<256xf32>, %u12pnvar: tensor<256xf32>, %u13qnmu: tensor<256xf32>, %u13qnvar: tensor<256xf32>, %u13enmu: tensor<1024xf32>, %u13envar: tensor<1024xf32>, %u13dnmu: tensor<1024xf32>, %u13dnvar: tensor<1024xf32>, %u13pnmu: tensor<256xf32>, %u13pnvar: tensor<256xf32>, %u14qnmu: tensor<256xf32>, %u14qnvar: tensor<256xf32>, %u14enmu: tensor<1024xf32>, %u14envar: tensor<1024xf32>, %u14dnmu: tensor<1024xf32>, %u14dnvar: tensor<1024xf32>, %u14pnmu: tensor<256xf32>, %u14pnvar: tensor<256xf32>, %u15enmu: tensor<1024xf32>, %u15envar: tensor<1024xf32>, %u15pnmu: tensor<256xf32>, %u15pnvar: tensor<256xf32>, %u16qnmu: tensor<256xf32>, %u16qnvar: tensor<256xf32>, %u16enmu: tensor<1024xf32>, %u16envar: tensor<1024xf32>, %u16pnmu: tensor<256xf32>, %u16pnvar: tensor<256xf32>, %u17qnmu: tensor<256xf32>, %u17qnvar: tensor<256xf32>, %u17enmu: tensor<512xf32>, %u17envar: tensor<512xf32>, %u17dnmu: tensor<512xf32>, %u17dnvar: tensor<512xf32>, %u17pnmu: tensor<256xf32>, %u17pnvar: tensor<256xf32>, %u18qnmu: tensor<256xf32>, %u18qnvar: tensor<256xf32>, %u18enmu: tensor<1024xf32>, %u18envar: tensor<1024xf32>, %u18dnmu: tensor<1024xf32>, %u18dnvar: tensor<1024xf32>, %u18pnmu: tensor<256xf32>, %u18pnvar: tensor<256xf32>, %u19enmu: tensor<1024xf32>, %u19envar: tensor<1024xf32>, %u19pnmu: tensor<256xf32>, %u19pnvar: tensor<256xf32>, %u20enmu: tensor<1024xf32>, %u20envar: tensor<1024xf32>, %u20pnmu: tensor<256xf32>, %u20pnvar: tensor<256xf32>, %u21qnmu: tensor<256xf32>, %u21qnvar: tensor<256xf32>, %u21enmu: tensor<512xf32>, %u21envar: tensor<512xf32>, %u21pnmu: tensor<256xf32>, %u21pnvar: tensor<256xf32>, %h1nmu: tensor<960xf32>, %h1nvar: tensor<960xf32>, %hnmu: tensor<1280xf32>, %hnvar: tensor<1280xf32>) -> tensor<64x1000xf32> {
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
    %v0 = stablehlo.reshape %x : (tensor<64x150528xf32>) -> tensor<64x3x224x224xf32>
    %v1 = stablehlo.convolution(%v0, %sW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x3x224x224xf32>, tensor<32x3x3x3xf32>) -> tensor<64x32x112x112xf32>
    %v2 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<64x32x112x112xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<64x32x112x112xf32>) -> tensor<64x401408xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<64x401408xf32>) -> tensor<64x32x112x112xf32>
    %v6 = stablehlo.broadcast_in_dim %stnmu, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v7 = stablehlo.subtract %v5, %v6 : tensor<64x32x112x112xf32>
    %v8 = stablehlo.broadcast_in_dim %stnvar, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v9 = stablehlo.constant dense<1.0e-5> : tensor<64x32x112x112xf32>
    %v10 = stablehlo.add %v8, %v9 : tensor<64x32x112x112xf32>
    %v11 = stablehlo.rsqrt %v10 : tensor<64x32x112x112xf32>
    %v12 = stablehlo.multiply %v7, %v11 : tensor<64x32x112x112xf32>
    %v13 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v14 = stablehlo.broadcast_in_dim %sbt, dims = [1] : (tensor<32xf32>) -> tensor<64x32x112x112xf32>
    %v15 = stablehlo.multiply %v12, %v13 : tensor<64x32x112x112xf32>
    %v16 = stablehlo.add %v15, %v14 : tensor<64x32x112x112xf32>
    %v17 = stablehlo.reshape %v16 : (tensor<64x32x112x112xf32>) -> tensor<64x401408xf32>
    %v18 = stablehlo.constant dense<0.0> : tensor<64x401408xf32>
    %v19 = stablehlo.maximum %v17, %v18 : tensor<64x401408xf32>
    %v20 = stablehlo.reshape %v19 : (tensor<64x401408xf32>) -> tensor<64x32x112x112xf32>
    %v21 = stablehlo.convolution(%v20, %f0cW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x112x112xf32>, tensor<128x32x3x3xf32>) -> tensor<64x128x56x56xf32>
    %v22 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v23 = stablehlo.add %v21, %v22 : tensor<64x128x56x56xf32>
    %v24 = stablehlo.reshape %v23 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v25 = stablehlo.reshape %v24 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v26 = stablehlo.broadcast_in_dim %f0cnmu, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v27 = stablehlo.subtract %v25, %v26 : tensor<64x128x56x56xf32>
    %v28 = stablehlo.broadcast_in_dim %f0cnvar, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v29 = stablehlo.constant dense<1.0e-5> : tensor<64x128x56x56xf32>
    %v30 = stablehlo.add %v28, %v29 : tensor<64x128x56x56xf32>
    %v31 = stablehlo.rsqrt %v30 : tensor<64x128x56x56xf32>
    %v32 = stablehlo.multiply %v27, %v31 : tensor<64x128x56x56xf32>
    %v33 = stablehlo.broadcast_in_dim %f0cg, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v34 = stablehlo.broadcast_in_dim %f0cbt, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v35 = stablehlo.multiply %v32, %v33 : tensor<64x128x56x56xf32>
    %v36 = stablehlo.add %v35, %v34 : tensor<64x128x56x56xf32>
    %v37 = stablehlo.reshape %v36 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v38 = stablehlo.logistic %v37 : tensor<64x401408xf32>
    %v39 = stablehlo.multiply %v37, %v38 : tensor<64x401408xf32>
    %v40 = stablehlo.reshape %v39 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v41 = stablehlo.convolution(%v40, %f0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x56x56xf32>, tensor<48x128x1x1xf32>) -> tensor<64x48x56x56xf32>
    %v42 = stablehlo.broadcast_in_dim %zb48, dims = [1] : (tensor<48xf32>) -> tensor<64x48x56x56xf32>
    %v43 = stablehlo.add %v41, %v42 : tensor<64x48x56x56xf32>
    %v44 = stablehlo.reshape %v43 : (tensor<64x48x56x56xf32>) -> tensor<64x150528xf32>
    %v45 = stablehlo.reshape %v44 : (tensor<64x150528xf32>) -> tensor<64x48x56x56xf32>
    %v46 = stablehlo.broadcast_in_dim %f0pnmu, dims = [1] : (tensor<48xf32>) -> tensor<64x48x56x56xf32>
    %v47 = stablehlo.subtract %v45, %v46 : tensor<64x48x56x56xf32>
    %v48 = stablehlo.broadcast_in_dim %f0pnvar, dims = [1] : (tensor<48xf32>) -> tensor<64x48x56x56xf32>
    %v49 = stablehlo.constant dense<1.0e-5> : tensor<64x48x56x56xf32>
    %v50 = stablehlo.add %v48, %v49 : tensor<64x48x56x56xf32>
    %v51 = stablehlo.rsqrt %v50 : tensor<64x48x56x56xf32>
    %v52 = stablehlo.multiply %v47, %v51 : tensor<64x48x56x56xf32>
    %v53 = stablehlo.broadcast_in_dim %f0pg, dims = [1] : (tensor<48xf32>) -> tensor<64x48x56x56xf32>
    %v54 = stablehlo.broadcast_in_dim %f0pbt, dims = [1] : (tensor<48xf32>) -> tensor<64x48x56x56xf32>
    %v55 = stablehlo.multiply %v52, %v53 : tensor<64x48x56x56xf32>
    %v56 = stablehlo.add %v55, %v54 : tensor<64x48x56x56xf32>
    %v57 = stablehlo.reshape %v56 : (tensor<64x48x56x56xf32>) -> tensor<64x150528xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<64x150528xf32>) -> tensor<64x48x56x56xf32>
    %v59 = stablehlo.convolution(%v58, %u1qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 48 : i64} : (tensor<64x48x56x56xf32>, tensor<48x1x3x3xf32>) -> tensor<64x48x28x28xf32>
    %v60 = stablehlo.broadcast_in_dim %zb48, dims = [1] : (tensor<48xf32>) -> tensor<64x48x28x28xf32>
    %v61 = stablehlo.add %v59, %v60 : tensor<64x48x28x28xf32>
    %v62 = stablehlo.reshape %v61 : (tensor<64x48x28x28xf32>) -> tensor<64x37632xf32>
    %v63 = stablehlo.reshape %v62 : (tensor<64x37632xf32>) -> tensor<64x48x28x28xf32>
    %v64 = stablehlo.broadcast_in_dim %u1qnmu, dims = [1] : (tensor<48xf32>) -> tensor<64x48x28x28xf32>
    %v65 = stablehlo.subtract %v63, %v64 : tensor<64x48x28x28xf32>
    %v66 = stablehlo.broadcast_in_dim %u1qnvar, dims = [1] : (tensor<48xf32>) -> tensor<64x48x28x28xf32>
    %v67 = stablehlo.constant dense<1.0e-5> : tensor<64x48x28x28xf32>
    %v68 = stablehlo.add %v66, %v67 : tensor<64x48x28x28xf32>
    %v69 = stablehlo.rsqrt %v68 : tensor<64x48x28x28xf32>
    %v70 = stablehlo.multiply %v65, %v69 : tensor<64x48x28x28xf32>
    %v71 = stablehlo.broadcast_in_dim %u1qg, dims = [1] : (tensor<48xf32>) -> tensor<64x48x28x28xf32>
    %v72 = stablehlo.broadcast_in_dim %u1qbt, dims = [1] : (tensor<48xf32>) -> tensor<64x48x28x28xf32>
    %v73 = stablehlo.multiply %v70, %v71 : tensor<64x48x28x28xf32>
    %v74 = stablehlo.add %v73, %v72 : tensor<64x48x28x28xf32>
    %v75 = stablehlo.reshape %v74 : (tensor<64x48x28x28xf32>) -> tensor<64x37632xf32>
    %v76 = stablehlo.constant dense<0.0> : tensor<64x37632xf32>
    %v77 = stablehlo.maximum %v75, %v76 : tensor<64x37632xf32>
    %v78 = stablehlo.reshape %v77 : (tensor<64x37632xf32>) -> tensor<64x48x28x28xf32>
    %v79 = stablehlo.convolution(%v78, %u1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x48x28x28xf32>, tensor<192x48x1x1xf32>) -> tensor<64x192x28x28xf32>
    %v80 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v81 = stablehlo.add %v79, %v80 : tensor<64x192x28x28xf32>
    %v82 = stablehlo.reshape %v81 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v83 = stablehlo.reshape %v82 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v84 = stablehlo.broadcast_in_dim %u1enmu, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v85 = stablehlo.subtract %v83, %v84 : tensor<64x192x28x28xf32>
    %v86 = stablehlo.broadcast_in_dim %u1envar, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v87 = stablehlo.constant dense<1.0e-5> : tensor<64x192x28x28xf32>
    %v88 = stablehlo.add %v86, %v87 : tensor<64x192x28x28xf32>
    %v89 = stablehlo.rsqrt %v88 : tensor<64x192x28x28xf32>
    %v90 = stablehlo.multiply %v85, %v89 : tensor<64x192x28x28xf32>
    %v91 = stablehlo.broadcast_in_dim %u1eg, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v92 = stablehlo.broadcast_in_dim %u1ebt, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v93 = stablehlo.multiply %v90, %v91 : tensor<64x192x28x28xf32>
    %v94 = stablehlo.add %v93, %v92 : tensor<64x192x28x28xf32>
    %v95 = stablehlo.reshape %v94 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v96 = stablehlo.constant dense<0.0> : tensor<64x150528xf32>
    %v97 = stablehlo.maximum %v95, %v96 : tensor<64x150528xf32>
    %v98 = stablehlo.reshape %v97 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v99 = stablehlo.convolution(%v98, %u1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<64x192x28x28xf32>, tensor<192x1x5x5xf32>) -> tensor<64x192x28x28xf32>
    %v100 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v101 = stablehlo.add %v99, %v100 : tensor<64x192x28x28xf32>
    %v102 = stablehlo.reshape %v101 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v103 = stablehlo.reshape %v102 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v104 = stablehlo.broadcast_in_dim %u1dnmu, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v105 = stablehlo.subtract %v103, %v104 : tensor<64x192x28x28xf32>
    %v106 = stablehlo.broadcast_in_dim %u1dnvar, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v107 = stablehlo.constant dense<1.0e-5> : tensor<64x192x28x28xf32>
    %v108 = stablehlo.add %v106, %v107 : tensor<64x192x28x28xf32>
    %v109 = stablehlo.rsqrt %v108 : tensor<64x192x28x28xf32>
    %v110 = stablehlo.multiply %v105, %v109 : tensor<64x192x28x28xf32>
    %v111 = stablehlo.broadcast_in_dim %u1dg, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v112 = stablehlo.broadcast_in_dim %u1dbt, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v113 = stablehlo.multiply %v110, %v111 : tensor<64x192x28x28xf32>
    %v114 = stablehlo.add %v113, %v112 : tensor<64x192x28x28xf32>
    %v115 = stablehlo.reshape %v114 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v116 = stablehlo.constant dense<0.0> : tensor<64x150528xf32>
    %v117 = stablehlo.maximum %v115, %v116 : tensor<64x150528xf32>
    %v118 = stablehlo.reshape %v117 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v119 = stablehlo.convolution(%v118, %u1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x192x28x28xf32>, tensor<80x192x1x1xf32>) -> tensor<64x80x28x28xf32>
    %v120 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v121 = stablehlo.add %v119, %v120 : tensor<64x80x28x28xf32>
    %v122 = stablehlo.reshape %v121 : (tensor<64x80x28x28xf32>) -> tensor<64x62720xf32>
    %v123 = stablehlo.reshape %v122 : (tensor<64x62720xf32>) -> tensor<64x80x28x28xf32>
    %v124 = stablehlo.broadcast_in_dim %u1pnmu, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v125 = stablehlo.subtract %v123, %v124 : tensor<64x80x28x28xf32>
    %v126 = stablehlo.broadcast_in_dim %u1pnvar, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v127 = stablehlo.constant dense<1.0e-5> : tensor<64x80x28x28xf32>
    %v128 = stablehlo.add %v126, %v127 : tensor<64x80x28x28xf32>
    %v129 = stablehlo.rsqrt %v128 : tensor<64x80x28x28xf32>
    %v130 = stablehlo.multiply %v125, %v129 : tensor<64x80x28x28xf32>
    %v131 = stablehlo.broadcast_in_dim %u1pg, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v132 = stablehlo.broadcast_in_dim %u1pbt, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v133 = stablehlo.multiply %v130, %v131 : tensor<64x80x28x28xf32>
    %v134 = stablehlo.add %v133, %v132 : tensor<64x80x28x28xf32>
    %v135 = stablehlo.reshape %v134 : (tensor<64x80x28x28xf32>) -> tensor<64x62720xf32>
    %v136 = stablehlo.reshape %v135 : (tensor<64x62720xf32>) -> tensor<64x80x28x28xf32>
    %v137 = stablehlo.convolution(%v136, %u2qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 80 : i64} : (tensor<64x80x28x28xf32>, tensor<80x1x3x3xf32>) -> tensor<64x80x28x28xf32>
    %v138 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v139 = stablehlo.add %v137, %v138 : tensor<64x80x28x28xf32>
    %v140 = stablehlo.reshape %v139 : (tensor<64x80x28x28xf32>) -> tensor<64x62720xf32>
    %v141 = stablehlo.reshape %v140 : (tensor<64x62720xf32>) -> tensor<64x80x28x28xf32>
    %v142 = stablehlo.broadcast_in_dim %u2qnmu, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v143 = stablehlo.subtract %v141, %v142 : tensor<64x80x28x28xf32>
    %v144 = stablehlo.broadcast_in_dim %u2qnvar, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v145 = stablehlo.constant dense<1.0e-5> : tensor<64x80x28x28xf32>
    %v146 = stablehlo.add %v144, %v145 : tensor<64x80x28x28xf32>
    %v147 = stablehlo.rsqrt %v146 : tensor<64x80x28x28xf32>
    %v148 = stablehlo.multiply %v143, %v147 : tensor<64x80x28x28xf32>
    %v149 = stablehlo.broadcast_in_dim %u2qg, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v150 = stablehlo.broadcast_in_dim %u2qbt, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v151 = stablehlo.multiply %v148, %v149 : tensor<64x80x28x28xf32>
    %v152 = stablehlo.add %v151, %v150 : tensor<64x80x28x28xf32>
    %v153 = stablehlo.reshape %v152 : (tensor<64x80x28x28xf32>) -> tensor<64x62720xf32>
    %v154 = stablehlo.constant dense<0.0> : tensor<64x62720xf32>
    %v155 = stablehlo.maximum %v153, %v154 : tensor<64x62720xf32>
    %v156 = stablehlo.reshape %v155 : (tensor<64x62720xf32>) -> tensor<64x80x28x28xf32>
    %v157 = stablehlo.convolution(%v156, %u2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x80x28x28xf32>, tensor<160x80x1x1xf32>) -> tensor<64x160x28x28xf32>
    %v158 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x28x28xf32>
    %v159 = stablehlo.add %v157, %v158 : tensor<64x160x28x28xf32>
    %v160 = stablehlo.reshape %v159 : (tensor<64x160x28x28xf32>) -> tensor<64x125440xf32>
    %v161 = stablehlo.reshape %v160 : (tensor<64x125440xf32>) -> tensor<64x160x28x28xf32>
    %v162 = stablehlo.broadcast_in_dim %u2enmu, dims = [1] : (tensor<160xf32>) -> tensor<64x160x28x28xf32>
    %v163 = stablehlo.subtract %v161, %v162 : tensor<64x160x28x28xf32>
    %v164 = stablehlo.broadcast_in_dim %u2envar, dims = [1] : (tensor<160xf32>) -> tensor<64x160x28x28xf32>
    %v165 = stablehlo.constant dense<1.0e-5> : tensor<64x160x28x28xf32>
    %v166 = stablehlo.add %v164, %v165 : tensor<64x160x28x28xf32>
    %v167 = stablehlo.rsqrt %v166 : tensor<64x160x28x28xf32>
    %v168 = stablehlo.multiply %v163, %v167 : tensor<64x160x28x28xf32>
    %v169 = stablehlo.broadcast_in_dim %u2eg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x28x28xf32>
    %v170 = stablehlo.broadcast_in_dim %u2ebt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x28x28xf32>
    %v171 = stablehlo.multiply %v168, %v169 : tensor<64x160x28x28xf32>
    %v172 = stablehlo.add %v171, %v170 : tensor<64x160x28x28xf32>
    %v173 = stablehlo.reshape %v172 : (tensor<64x160x28x28xf32>) -> tensor<64x125440xf32>
    %v174 = stablehlo.constant dense<0.0> : tensor<64x125440xf32>
    %v175 = stablehlo.maximum %v173, %v174 : tensor<64x125440xf32>
    %v176 = stablehlo.reshape %v175 : (tensor<64x125440xf32>) -> tensor<64x160x28x28xf32>
    %v177 = stablehlo.convolution(%v176, %u2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<64x160x28x28xf32>, tensor<160x1x3x3xf32>) -> tensor<64x160x28x28xf32>
    %v178 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x28x28xf32>
    %v179 = stablehlo.add %v177, %v178 : tensor<64x160x28x28xf32>
    %v180 = stablehlo.reshape %v179 : (tensor<64x160x28x28xf32>) -> tensor<64x125440xf32>
    %v181 = stablehlo.reshape %v180 : (tensor<64x125440xf32>) -> tensor<64x160x28x28xf32>
    %v182 = stablehlo.broadcast_in_dim %u2dnmu, dims = [1] : (tensor<160xf32>) -> tensor<64x160x28x28xf32>
    %v183 = stablehlo.subtract %v181, %v182 : tensor<64x160x28x28xf32>
    %v184 = stablehlo.broadcast_in_dim %u2dnvar, dims = [1] : (tensor<160xf32>) -> tensor<64x160x28x28xf32>
    %v185 = stablehlo.constant dense<1.0e-5> : tensor<64x160x28x28xf32>
    %v186 = stablehlo.add %v184, %v185 : tensor<64x160x28x28xf32>
    %v187 = stablehlo.rsqrt %v186 : tensor<64x160x28x28xf32>
    %v188 = stablehlo.multiply %v183, %v187 : tensor<64x160x28x28xf32>
    %v189 = stablehlo.broadcast_in_dim %u2dg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x28x28xf32>
    %v190 = stablehlo.broadcast_in_dim %u2dbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x28x28xf32>
    %v191 = stablehlo.multiply %v188, %v189 : tensor<64x160x28x28xf32>
    %v192 = stablehlo.add %v191, %v190 : tensor<64x160x28x28xf32>
    %v193 = stablehlo.reshape %v192 : (tensor<64x160x28x28xf32>) -> tensor<64x125440xf32>
    %v194 = stablehlo.constant dense<0.0> : tensor<64x125440xf32>
    %v195 = stablehlo.maximum %v193, %v194 : tensor<64x125440xf32>
    %v196 = stablehlo.reshape %v195 : (tensor<64x125440xf32>) -> tensor<64x160x28x28xf32>
    %v197 = stablehlo.convolution(%v196, %u2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x160x28x28xf32>, tensor<80x160x1x1xf32>) -> tensor<64x80x28x28xf32>
    %v198 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v199 = stablehlo.add %v197, %v198 : tensor<64x80x28x28xf32>
    %v200 = stablehlo.reshape %v199 : (tensor<64x80x28x28xf32>) -> tensor<64x62720xf32>
    %v201 = stablehlo.reshape %v200 : (tensor<64x62720xf32>) -> tensor<64x80x28x28xf32>
    %v202 = stablehlo.broadcast_in_dim %u2pnmu, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v203 = stablehlo.subtract %v201, %v202 : tensor<64x80x28x28xf32>
    %v204 = stablehlo.broadcast_in_dim %u2pnvar, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v205 = stablehlo.constant dense<1.0e-5> : tensor<64x80x28x28xf32>
    %v206 = stablehlo.add %v204, %v205 : tensor<64x80x28x28xf32>
    %v207 = stablehlo.rsqrt %v206 : tensor<64x80x28x28xf32>
    %v208 = stablehlo.multiply %v203, %v207 : tensor<64x80x28x28xf32>
    %v209 = stablehlo.broadcast_in_dim %u2pg, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v210 = stablehlo.broadcast_in_dim %u2pbt, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v211 = stablehlo.multiply %v208, %v209 : tensor<64x80x28x28xf32>
    %v212 = stablehlo.add %v211, %v210 : tensor<64x80x28x28xf32>
    %v213 = stablehlo.reshape %v212 : (tensor<64x80x28x28xf32>) -> tensor<64x62720xf32>
    %v214 = stablehlo.add %v213, %v135 : tensor<64x62720xf32>
    %v215 = stablehlo.reshape %v214 : (tensor<64x62720xf32>) -> tensor<64x80x28x28xf32>
    %v216 = stablehlo.convolution(%v215, %u3qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 80 : i64} : (tensor<64x80x28x28xf32>, tensor<80x1x3x3xf32>) -> tensor<64x80x14x14xf32>
    %v217 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v218 = stablehlo.add %v216, %v217 : tensor<64x80x14x14xf32>
    %v219 = stablehlo.reshape %v218 : (tensor<64x80x14x14xf32>) -> tensor<64x15680xf32>
    %v220 = stablehlo.reshape %v219 : (tensor<64x15680xf32>) -> tensor<64x80x14x14xf32>
    %v221 = stablehlo.broadcast_in_dim %u3qnmu, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v222 = stablehlo.subtract %v220, %v221 : tensor<64x80x14x14xf32>
    %v223 = stablehlo.broadcast_in_dim %u3qnvar, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v224 = stablehlo.constant dense<1.0e-5> : tensor<64x80x14x14xf32>
    %v225 = stablehlo.add %v223, %v224 : tensor<64x80x14x14xf32>
    %v226 = stablehlo.rsqrt %v225 : tensor<64x80x14x14xf32>
    %v227 = stablehlo.multiply %v222, %v226 : tensor<64x80x14x14xf32>
    %v228 = stablehlo.broadcast_in_dim %u3qg, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v229 = stablehlo.broadcast_in_dim %u3qbt, dims = [1] : (tensor<80xf32>) -> tensor<64x80x14x14xf32>
    %v230 = stablehlo.multiply %v227, %v228 : tensor<64x80x14x14xf32>
    %v231 = stablehlo.add %v230, %v229 : tensor<64x80x14x14xf32>
    %v232 = stablehlo.reshape %v231 : (tensor<64x80x14x14xf32>) -> tensor<64x15680xf32>
    %v233 = stablehlo.constant dense<0.0> : tensor<64x15680xf32>
    %v234 = stablehlo.maximum %v232, %v233 : tensor<64x15680xf32>
    %v235 = stablehlo.reshape %v234 : (tensor<64x15680xf32>) -> tensor<64x80x14x14xf32>
    %v236 = stablehlo.convolution(%v235, %u3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<64x480x14x14xf32>
    %v237 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v238 = stablehlo.add %v236, %v237 : tensor<64x480x14x14xf32>
    %v239 = stablehlo.reshape %v238 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v240 = stablehlo.reshape %v239 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v241 = stablehlo.broadcast_in_dim %u3enmu, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v242 = stablehlo.subtract %v240, %v241 : tensor<64x480x14x14xf32>
    %v243 = stablehlo.broadcast_in_dim %u3envar, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v244 = stablehlo.constant dense<1.0e-5> : tensor<64x480x14x14xf32>
    %v245 = stablehlo.add %v243, %v244 : tensor<64x480x14x14xf32>
    %v246 = stablehlo.rsqrt %v245 : tensor<64x480x14x14xf32>
    %v247 = stablehlo.multiply %v242, %v246 : tensor<64x480x14x14xf32>
    %v248 = stablehlo.broadcast_in_dim %u3eg, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v249 = stablehlo.broadcast_in_dim %u3ebt, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v250 = stablehlo.multiply %v247, %v248 : tensor<64x480x14x14xf32>
    %v251 = stablehlo.add %v250, %v249 : tensor<64x480x14x14xf32>
    %v252 = stablehlo.reshape %v251 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v253 = stablehlo.constant dense<0.0> : tensor<64x94080xf32>
    %v254 = stablehlo.maximum %v252, %v253 : tensor<64x94080xf32>
    %v255 = stablehlo.reshape %v254 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v256 = stablehlo.convolution(%v255, %u3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<64x480x14x14xf32>, tensor<480x1x5x5xf32>) -> tensor<64x480x14x14xf32>
    %v257 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v258 = stablehlo.add %v256, %v257 : tensor<64x480x14x14xf32>
    %v259 = stablehlo.reshape %v258 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v260 = stablehlo.reshape %v259 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v261 = stablehlo.broadcast_in_dim %u3dnmu, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v262 = stablehlo.subtract %v260, %v261 : tensor<64x480x14x14xf32>
    %v263 = stablehlo.broadcast_in_dim %u3dnvar, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v264 = stablehlo.constant dense<1.0e-5> : tensor<64x480x14x14xf32>
    %v265 = stablehlo.add %v263, %v264 : tensor<64x480x14x14xf32>
    %v266 = stablehlo.rsqrt %v265 : tensor<64x480x14x14xf32>
    %v267 = stablehlo.multiply %v262, %v266 : tensor<64x480x14x14xf32>
    %v268 = stablehlo.broadcast_in_dim %u3dg, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v269 = stablehlo.broadcast_in_dim %u3dbt, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v270 = stablehlo.multiply %v267, %v268 : tensor<64x480x14x14xf32>
    %v271 = stablehlo.add %v270, %v269 : tensor<64x480x14x14xf32>
    %v272 = stablehlo.reshape %v271 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v273 = stablehlo.constant dense<0.0> : tensor<64x94080xf32>
    %v274 = stablehlo.maximum %v272, %v273 : tensor<64x94080xf32>
    %v275 = stablehlo.reshape %v274 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v276 = stablehlo.convolution(%v275, %u3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x480x14x14xf32>, tensor<160x480x1x1xf32>) -> tensor<64x160x14x14xf32>
    %v277 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v278 = stablehlo.add %v276, %v277 : tensor<64x160x14x14xf32>
    %v279 = stablehlo.reshape %v278 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v280 = stablehlo.reshape %v279 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v281 = stablehlo.broadcast_in_dim %u3pnmu, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v282 = stablehlo.subtract %v280, %v281 : tensor<64x160x14x14xf32>
    %v283 = stablehlo.broadcast_in_dim %u3pnvar, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v284 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v285 = stablehlo.add %v283, %v284 : tensor<64x160x14x14xf32>
    %v286 = stablehlo.rsqrt %v285 : tensor<64x160x14x14xf32>
    %v287 = stablehlo.multiply %v282, %v286 : tensor<64x160x14x14xf32>
    %v288 = stablehlo.broadcast_in_dim %u3pg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v289 = stablehlo.broadcast_in_dim %u3pbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v290 = stablehlo.multiply %v287, %v288 : tensor<64x160x14x14xf32>
    %v291 = stablehlo.add %v290, %v289 : tensor<64x160x14x14xf32>
    %v292 = stablehlo.reshape %v291 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v293 = stablehlo.reshape %v292 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v294 = stablehlo.convolution(%v293, %u4qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<64x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<64x160x14x14xf32>
    %v295 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v296 = stablehlo.add %v294, %v295 : tensor<64x160x14x14xf32>
    %v297 = stablehlo.reshape %v296 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v298 = stablehlo.reshape %v297 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v299 = stablehlo.broadcast_in_dim %u4qnmu, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v300 = stablehlo.subtract %v298, %v299 : tensor<64x160x14x14xf32>
    %v301 = stablehlo.broadcast_in_dim %u4qnvar, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v302 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v303 = stablehlo.add %v301, %v302 : tensor<64x160x14x14xf32>
    %v304 = stablehlo.rsqrt %v303 : tensor<64x160x14x14xf32>
    %v305 = stablehlo.multiply %v300, %v304 : tensor<64x160x14x14xf32>
    %v306 = stablehlo.broadcast_in_dim %u4qg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v307 = stablehlo.broadcast_in_dim %u4qbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v308 = stablehlo.multiply %v305, %v306 : tensor<64x160x14x14xf32>
    %v309 = stablehlo.add %v308, %v307 : tensor<64x160x14x14xf32>
    %v310 = stablehlo.reshape %v309 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v311 = stablehlo.constant dense<0.0> : tensor<64x31360xf32>
    %v312 = stablehlo.maximum %v310, %v311 : tensor<64x31360xf32>
    %v313 = stablehlo.reshape %v312 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v314 = stablehlo.convolution(%v313, %u4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<64x640x14x14xf32>
    %v315 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v316 = stablehlo.add %v314, %v315 : tensor<64x640x14x14xf32>
    %v317 = stablehlo.reshape %v316 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v318 = stablehlo.reshape %v317 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v319 = stablehlo.broadcast_in_dim %u4enmu, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v320 = stablehlo.subtract %v318, %v319 : tensor<64x640x14x14xf32>
    %v321 = stablehlo.broadcast_in_dim %u4envar, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v322 = stablehlo.constant dense<1.0e-5> : tensor<64x640x14x14xf32>
    %v323 = stablehlo.add %v321, %v322 : tensor<64x640x14x14xf32>
    %v324 = stablehlo.rsqrt %v323 : tensor<64x640x14x14xf32>
    %v325 = stablehlo.multiply %v320, %v324 : tensor<64x640x14x14xf32>
    %v326 = stablehlo.broadcast_in_dim %u4eg, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v327 = stablehlo.broadcast_in_dim %u4ebt, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v328 = stablehlo.multiply %v325, %v326 : tensor<64x640x14x14xf32>
    %v329 = stablehlo.add %v328, %v327 : tensor<64x640x14x14xf32>
    %v330 = stablehlo.reshape %v329 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v331 = stablehlo.constant dense<0.0> : tensor<64x125440xf32>
    %v332 = stablehlo.maximum %v330, %v331 : tensor<64x125440xf32>
    %v333 = stablehlo.reshape %v332 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v334 = stablehlo.convolution(%v333, %u4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 640 : i64} : (tensor<64x640x14x14xf32>, tensor<640x1x3x3xf32>) -> tensor<64x640x14x14xf32>
    %v335 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v336 = stablehlo.add %v334, %v335 : tensor<64x640x14x14xf32>
    %v337 = stablehlo.reshape %v336 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v338 = stablehlo.reshape %v337 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v339 = stablehlo.broadcast_in_dim %u4dnmu, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v340 = stablehlo.subtract %v338, %v339 : tensor<64x640x14x14xf32>
    %v341 = stablehlo.broadcast_in_dim %u4dnvar, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v342 = stablehlo.constant dense<1.0e-5> : tensor<64x640x14x14xf32>
    %v343 = stablehlo.add %v341, %v342 : tensor<64x640x14x14xf32>
    %v344 = stablehlo.rsqrt %v343 : tensor<64x640x14x14xf32>
    %v345 = stablehlo.multiply %v340, %v344 : tensor<64x640x14x14xf32>
    %v346 = stablehlo.broadcast_in_dim %u4dg, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v347 = stablehlo.broadcast_in_dim %u4dbt, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v348 = stablehlo.multiply %v345, %v346 : tensor<64x640x14x14xf32>
    %v349 = stablehlo.add %v348, %v347 : tensor<64x640x14x14xf32>
    %v350 = stablehlo.reshape %v349 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v351 = stablehlo.constant dense<0.0> : tensor<64x125440xf32>
    %v352 = stablehlo.maximum %v350, %v351 : tensor<64x125440xf32>
    %v353 = stablehlo.reshape %v352 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v354 = stablehlo.convolution(%v353, %u4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<64x160x14x14xf32>
    %v355 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v356 = stablehlo.add %v354, %v355 : tensor<64x160x14x14xf32>
    %v357 = stablehlo.reshape %v356 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v358 = stablehlo.reshape %v357 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v359 = stablehlo.broadcast_in_dim %u4pnmu, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v360 = stablehlo.subtract %v358, %v359 : tensor<64x160x14x14xf32>
    %v361 = stablehlo.broadcast_in_dim %u4pnvar, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v362 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v363 = stablehlo.add %v361, %v362 : tensor<64x160x14x14xf32>
    %v364 = stablehlo.rsqrt %v363 : tensor<64x160x14x14xf32>
    %v365 = stablehlo.multiply %v360, %v364 : tensor<64x160x14x14xf32>
    %v366 = stablehlo.broadcast_in_dim %u4pg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v367 = stablehlo.broadcast_in_dim %u4pbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v368 = stablehlo.multiply %v365, %v366 : tensor<64x160x14x14xf32>
    %v369 = stablehlo.add %v368, %v367 : tensor<64x160x14x14xf32>
    %v370 = stablehlo.reshape %v369 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v371 = stablehlo.add %v370, %v292 : tensor<64x31360xf32>
    %v372 = stablehlo.reshape %v371 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v373 = stablehlo.convolution(%v372, %u5qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<64x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<64x160x14x14xf32>
    %v374 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v375 = stablehlo.add %v373, %v374 : tensor<64x160x14x14xf32>
    %v376 = stablehlo.reshape %v375 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v377 = stablehlo.reshape %v376 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v378 = stablehlo.broadcast_in_dim %u5qnmu, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v379 = stablehlo.subtract %v377, %v378 : tensor<64x160x14x14xf32>
    %v380 = stablehlo.broadcast_in_dim %u5qnvar, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v381 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v382 = stablehlo.add %v380, %v381 : tensor<64x160x14x14xf32>
    %v383 = stablehlo.rsqrt %v382 : tensor<64x160x14x14xf32>
    %v384 = stablehlo.multiply %v379, %v383 : tensor<64x160x14x14xf32>
    %v385 = stablehlo.broadcast_in_dim %u5qg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v386 = stablehlo.broadcast_in_dim %u5qbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v387 = stablehlo.multiply %v384, %v385 : tensor<64x160x14x14xf32>
    %v388 = stablehlo.add %v387, %v386 : tensor<64x160x14x14xf32>
    %v389 = stablehlo.reshape %v388 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v390 = stablehlo.constant dense<0.0> : tensor<64x31360xf32>
    %v391 = stablehlo.maximum %v389, %v390 : tensor<64x31360xf32>
    %v392 = stablehlo.reshape %v391 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v393 = stablehlo.convolution(%v392, %u5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<64x640x14x14xf32>
    %v394 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v395 = stablehlo.add %v393, %v394 : tensor<64x640x14x14xf32>
    %v396 = stablehlo.reshape %v395 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v397 = stablehlo.reshape %v396 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v398 = stablehlo.broadcast_in_dim %u5enmu, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v399 = stablehlo.subtract %v397, %v398 : tensor<64x640x14x14xf32>
    %v400 = stablehlo.broadcast_in_dim %u5envar, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v401 = stablehlo.constant dense<1.0e-5> : tensor<64x640x14x14xf32>
    %v402 = stablehlo.add %v400, %v401 : tensor<64x640x14x14xf32>
    %v403 = stablehlo.rsqrt %v402 : tensor<64x640x14x14xf32>
    %v404 = stablehlo.multiply %v399, %v403 : tensor<64x640x14x14xf32>
    %v405 = stablehlo.broadcast_in_dim %u5eg, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v406 = stablehlo.broadcast_in_dim %u5ebt, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v407 = stablehlo.multiply %v404, %v405 : tensor<64x640x14x14xf32>
    %v408 = stablehlo.add %v407, %v406 : tensor<64x640x14x14xf32>
    %v409 = stablehlo.reshape %v408 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v410 = stablehlo.constant dense<0.0> : tensor<64x125440xf32>
    %v411 = stablehlo.maximum %v409, %v410 : tensor<64x125440xf32>
    %v412 = stablehlo.reshape %v411 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v413 = stablehlo.convolution(%v412, %u5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 640 : i64} : (tensor<64x640x14x14xf32>, tensor<640x1x3x3xf32>) -> tensor<64x640x14x14xf32>
    %v414 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v415 = stablehlo.add %v413, %v414 : tensor<64x640x14x14xf32>
    %v416 = stablehlo.reshape %v415 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v417 = stablehlo.reshape %v416 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v418 = stablehlo.broadcast_in_dim %u5dnmu, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v419 = stablehlo.subtract %v417, %v418 : tensor<64x640x14x14xf32>
    %v420 = stablehlo.broadcast_in_dim %u5dnvar, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v421 = stablehlo.constant dense<1.0e-5> : tensor<64x640x14x14xf32>
    %v422 = stablehlo.add %v420, %v421 : tensor<64x640x14x14xf32>
    %v423 = stablehlo.rsqrt %v422 : tensor<64x640x14x14xf32>
    %v424 = stablehlo.multiply %v419, %v423 : tensor<64x640x14x14xf32>
    %v425 = stablehlo.broadcast_in_dim %u5dg, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v426 = stablehlo.broadcast_in_dim %u5dbt, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v427 = stablehlo.multiply %v424, %v425 : tensor<64x640x14x14xf32>
    %v428 = stablehlo.add %v427, %v426 : tensor<64x640x14x14xf32>
    %v429 = stablehlo.reshape %v428 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v430 = stablehlo.constant dense<0.0> : tensor<64x125440xf32>
    %v431 = stablehlo.maximum %v429, %v430 : tensor<64x125440xf32>
    %v432 = stablehlo.reshape %v431 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v433 = stablehlo.convolution(%v432, %u5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<64x160x14x14xf32>
    %v434 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v435 = stablehlo.add %v433, %v434 : tensor<64x160x14x14xf32>
    %v436 = stablehlo.reshape %v435 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v437 = stablehlo.reshape %v436 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v438 = stablehlo.broadcast_in_dim %u5pnmu, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v439 = stablehlo.subtract %v437, %v438 : tensor<64x160x14x14xf32>
    %v440 = stablehlo.broadcast_in_dim %u5pnvar, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v441 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v442 = stablehlo.add %v440, %v441 : tensor<64x160x14x14xf32>
    %v443 = stablehlo.rsqrt %v442 : tensor<64x160x14x14xf32>
    %v444 = stablehlo.multiply %v439, %v443 : tensor<64x160x14x14xf32>
    %v445 = stablehlo.broadcast_in_dim %u5pg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v446 = stablehlo.broadcast_in_dim %u5pbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v447 = stablehlo.multiply %v444, %v445 : tensor<64x160x14x14xf32>
    %v448 = stablehlo.add %v447, %v446 : tensor<64x160x14x14xf32>
    %v449 = stablehlo.reshape %v448 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v450 = stablehlo.add %v449, %v371 : tensor<64x31360xf32>
    %v451 = stablehlo.reshape %v450 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v452 = stablehlo.convolution(%v451, %u6qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<64x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<64x160x14x14xf32>
    %v453 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v454 = stablehlo.add %v452, %v453 : tensor<64x160x14x14xf32>
    %v455 = stablehlo.reshape %v454 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v456 = stablehlo.reshape %v455 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v457 = stablehlo.broadcast_in_dim %u6qnmu, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v458 = stablehlo.subtract %v456, %v457 : tensor<64x160x14x14xf32>
    %v459 = stablehlo.broadcast_in_dim %u6qnvar, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v460 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v461 = stablehlo.add %v459, %v460 : tensor<64x160x14x14xf32>
    %v462 = stablehlo.rsqrt %v461 : tensor<64x160x14x14xf32>
    %v463 = stablehlo.multiply %v458, %v462 : tensor<64x160x14x14xf32>
    %v464 = stablehlo.broadcast_in_dim %u6qg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v465 = stablehlo.broadcast_in_dim %u6qbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v466 = stablehlo.multiply %v463, %v464 : tensor<64x160x14x14xf32>
    %v467 = stablehlo.add %v466, %v465 : tensor<64x160x14x14xf32>
    %v468 = stablehlo.reshape %v467 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v469 = stablehlo.constant dense<0.0> : tensor<64x31360xf32>
    %v470 = stablehlo.maximum %v468, %v469 : tensor<64x31360xf32>
    %v471 = stablehlo.reshape %v470 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v472 = stablehlo.convolution(%v471, %u6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<64x640x14x14xf32>
    %v473 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v474 = stablehlo.add %v472, %v473 : tensor<64x640x14x14xf32>
    %v475 = stablehlo.reshape %v474 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v476 = stablehlo.reshape %v475 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v477 = stablehlo.broadcast_in_dim %u6enmu, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v478 = stablehlo.subtract %v476, %v477 : tensor<64x640x14x14xf32>
    %v479 = stablehlo.broadcast_in_dim %u6envar, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v480 = stablehlo.constant dense<1.0e-5> : tensor<64x640x14x14xf32>
    %v481 = stablehlo.add %v479, %v480 : tensor<64x640x14x14xf32>
    %v482 = stablehlo.rsqrt %v481 : tensor<64x640x14x14xf32>
    %v483 = stablehlo.multiply %v478, %v482 : tensor<64x640x14x14xf32>
    %v484 = stablehlo.broadcast_in_dim %u6eg, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v485 = stablehlo.broadcast_in_dim %u6ebt, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v486 = stablehlo.multiply %v483, %v484 : tensor<64x640x14x14xf32>
    %v487 = stablehlo.add %v486, %v485 : tensor<64x640x14x14xf32>
    %v488 = stablehlo.reshape %v487 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v489 = stablehlo.constant dense<0.0> : tensor<64x125440xf32>
    %v490 = stablehlo.maximum %v488, %v489 : tensor<64x125440xf32>
    %v491 = stablehlo.reshape %v490 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v492 = stablehlo.convolution(%v491, %u6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 640 : i64} : (tensor<64x640x14x14xf32>, tensor<640x1x5x5xf32>) -> tensor<64x640x14x14xf32>
    %v493 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v494 = stablehlo.add %v492, %v493 : tensor<64x640x14x14xf32>
    %v495 = stablehlo.reshape %v494 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v496 = stablehlo.reshape %v495 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v497 = stablehlo.broadcast_in_dim %u6dnmu, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v498 = stablehlo.subtract %v496, %v497 : tensor<64x640x14x14xf32>
    %v499 = stablehlo.broadcast_in_dim %u6dnvar, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v500 = stablehlo.constant dense<1.0e-5> : tensor<64x640x14x14xf32>
    %v501 = stablehlo.add %v499, %v500 : tensor<64x640x14x14xf32>
    %v502 = stablehlo.rsqrt %v501 : tensor<64x640x14x14xf32>
    %v503 = stablehlo.multiply %v498, %v502 : tensor<64x640x14x14xf32>
    %v504 = stablehlo.broadcast_in_dim %u6dg, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v505 = stablehlo.broadcast_in_dim %u6dbt, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v506 = stablehlo.multiply %v503, %v504 : tensor<64x640x14x14xf32>
    %v507 = stablehlo.add %v506, %v505 : tensor<64x640x14x14xf32>
    %v508 = stablehlo.reshape %v507 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v509 = stablehlo.constant dense<0.0> : tensor<64x125440xf32>
    %v510 = stablehlo.maximum %v508, %v509 : tensor<64x125440xf32>
    %v511 = stablehlo.reshape %v510 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v512 = stablehlo.convolution(%v511, %u6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<64x160x14x14xf32>
    %v513 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v514 = stablehlo.add %v512, %v513 : tensor<64x160x14x14xf32>
    %v515 = stablehlo.reshape %v514 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v516 = stablehlo.reshape %v515 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v517 = stablehlo.broadcast_in_dim %u6pnmu, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v518 = stablehlo.subtract %v516, %v517 : tensor<64x160x14x14xf32>
    %v519 = stablehlo.broadcast_in_dim %u6pnvar, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v520 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v521 = stablehlo.add %v519, %v520 : tensor<64x160x14x14xf32>
    %v522 = stablehlo.rsqrt %v521 : tensor<64x160x14x14xf32>
    %v523 = stablehlo.multiply %v518, %v522 : tensor<64x160x14x14xf32>
    %v524 = stablehlo.broadcast_in_dim %u6pg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v525 = stablehlo.broadcast_in_dim %u6pbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v526 = stablehlo.multiply %v523, %v524 : tensor<64x160x14x14xf32>
    %v527 = stablehlo.add %v526, %v525 : tensor<64x160x14x14xf32>
    %v528 = stablehlo.reshape %v527 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v529 = stablehlo.add %v528, %v450 : tensor<64x31360xf32>
    %v530 = stablehlo.reshape %v529 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v531 = stablehlo.convolution(%v530, %u7qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<64x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<64x160x14x14xf32>
    %v532 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v533 = stablehlo.add %v531, %v532 : tensor<64x160x14x14xf32>
    %v534 = stablehlo.reshape %v533 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v535 = stablehlo.reshape %v534 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v536 = stablehlo.broadcast_in_dim %u7qnmu, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v537 = stablehlo.subtract %v535, %v536 : tensor<64x160x14x14xf32>
    %v538 = stablehlo.broadcast_in_dim %u7qnvar, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v539 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v540 = stablehlo.add %v538, %v539 : tensor<64x160x14x14xf32>
    %v541 = stablehlo.rsqrt %v540 : tensor<64x160x14x14xf32>
    %v542 = stablehlo.multiply %v537, %v541 : tensor<64x160x14x14xf32>
    %v543 = stablehlo.broadcast_in_dim %u7qg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v544 = stablehlo.broadcast_in_dim %u7qbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v545 = stablehlo.multiply %v542, %v543 : tensor<64x160x14x14xf32>
    %v546 = stablehlo.add %v545, %v544 : tensor<64x160x14x14xf32>
    %v547 = stablehlo.reshape %v546 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v548 = stablehlo.constant dense<0.0> : tensor<64x31360xf32>
    %v549 = stablehlo.maximum %v547, %v548 : tensor<64x31360xf32>
    %v550 = stablehlo.reshape %v549 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v551 = stablehlo.convolution(%v550, %u7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<64x640x14x14xf32>
    %v552 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v553 = stablehlo.add %v551, %v552 : tensor<64x640x14x14xf32>
    %v554 = stablehlo.reshape %v553 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v555 = stablehlo.reshape %v554 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v556 = stablehlo.broadcast_in_dim %u7enmu, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v557 = stablehlo.subtract %v555, %v556 : tensor<64x640x14x14xf32>
    %v558 = stablehlo.broadcast_in_dim %u7envar, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v559 = stablehlo.constant dense<1.0e-5> : tensor<64x640x14x14xf32>
    %v560 = stablehlo.add %v558, %v559 : tensor<64x640x14x14xf32>
    %v561 = stablehlo.rsqrt %v560 : tensor<64x640x14x14xf32>
    %v562 = stablehlo.multiply %v557, %v561 : tensor<64x640x14x14xf32>
    %v563 = stablehlo.broadcast_in_dim %u7eg, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v564 = stablehlo.broadcast_in_dim %u7ebt, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v565 = stablehlo.multiply %v562, %v563 : tensor<64x640x14x14xf32>
    %v566 = stablehlo.add %v565, %v564 : tensor<64x640x14x14xf32>
    %v567 = stablehlo.reshape %v566 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v568 = stablehlo.constant dense<0.0> : tensor<64x125440xf32>
    %v569 = stablehlo.maximum %v567, %v568 : tensor<64x125440xf32>
    %v570 = stablehlo.reshape %v569 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v571 = stablehlo.convolution(%v570, %u7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 640 : i64} : (tensor<64x640x14x14xf32>, tensor<640x1x3x3xf32>) -> tensor<64x640x14x14xf32>
    %v572 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v573 = stablehlo.add %v571, %v572 : tensor<64x640x14x14xf32>
    %v574 = stablehlo.reshape %v573 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v575 = stablehlo.reshape %v574 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v576 = stablehlo.broadcast_in_dim %u7dnmu, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v577 = stablehlo.subtract %v575, %v576 : tensor<64x640x14x14xf32>
    %v578 = stablehlo.broadcast_in_dim %u7dnvar, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v579 = stablehlo.constant dense<1.0e-5> : tensor<64x640x14x14xf32>
    %v580 = stablehlo.add %v578, %v579 : tensor<64x640x14x14xf32>
    %v581 = stablehlo.rsqrt %v580 : tensor<64x640x14x14xf32>
    %v582 = stablehlo.multiply %v577, %v581 : tensor<64x640x14x14xf32>
    %v583 = stablehlo.broadcast_in_dim %u7dg, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v584 = stablehlo.broadcast_in_dim %u7dbt, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v585 = stablehlo.multiply %v582, %v583 : tensor<64x640x14x14xf32>
    %v586 = stablehlo.add %v585, %v584 : tensor<64x640x14x14xf32>
    %v587 = stablehlo.reshape %v586 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v588 = stablehlo.constant dense<0.0> : tensor<64x125440xf32>
    %v589 = stablehlo.maximum %v587, %v588 : tensor<64x125440xf32>
    %v590 = stablehlo.reshape %v589 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v591 = stablehlo.convolution(%v590, %u7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<64x160x14x14xf32>
    %v592 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v593 = stablehlo.add %v591, %v592 : tensor<64x160x14x14xf32>
    %v594 = stablehlo.reshape %v593 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v595 = stablehlo.reshape %v594 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v596 = stablehlo.broadcast_in_dim %u7pnmu, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v597 = stablehlo.subtract %v595, %v596 : tensor<64x160x14x14xf32>
    %v598 = stablehlo.broadcast_in_dim %u7pnvar, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v599 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v600 = stablehlo.add %v598, %v599 : tensor<64x160x14x14xf32>
    %v601 = stablehlo.rsqrt %v600 : tensor<64x160x14x14xf32>
    %v602 = stablehlo.multiply %v597, %v601 : tensor<64x160x14x14xf32>
    %v603 = stablehlo.broadcast_in_dim %u7pg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v604 = stablehlo.broadcast_in_dim %u7pbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v605 = stablehlo.multiply %v602, %v603 : tensor<64x160x14x14xf32>
    %v606 = stablehlo.add %v605, %v604 : tensor<64x160x14x14xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v608 = stablehlo.add %v607, %v529 : tensor<64x31360xf32>
    %v609 = stablehlo.reshape %v608 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v610 = stablehlo.convolution(%v609, %u8qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<64x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<64x160x14x14xf32>
    %v611 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v612 = stablehlo.add %v610, %v611 : tensor<64x160x14x14xf32>
    %v613 = stablehlo.reshape %v612 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v614 = stablehlo.reshape %v613 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v615 = stablehlo.broadcast_in_dim %u8qnmu, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v616 = stablehlo.subtract %v614, %v615 : tensor<64x160x14x14xf32>
    %v617 = stablehlo.broadcast_in_dim %u8qnvar, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v618 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v619 = stablehlo.add %v617, %v618 : tensor<64x160x14x14xf32>
    %v620 = stablehlo.rsqrt %v619 : tensor<64x160x14x14xf32>
    %v621 = stablehlo.multiply %v616, %v620 : tensor<64x160x14x14xf32>
    %v622 = stablehlo.broadcast_in_dim %u8qg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v623 = stablehlo.broadcast_in_dim %u8qbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v624 = stablehlo.multiply %v621, %v622 : tensor<64x160x14x14xf32>
    %v625 = stablehlo.add %v624, %v623 : tensor<64x160x14x14xf32>
    %v626 = stablehlo.reshape %v625 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v627 = stablehlo.constant dense<0.0> : tensor<64x31360xf32>
    %v628 = stablehlo.maximum %v626, %v627 : tensor<64x31360xf32>
    %v629 = stablehlo.reshape %v628 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v630 = stablehlo.convolution(%v629, %u8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<64x640x14x14xf32>
    %v631 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v632 = stablehlo.add %v630, %v631 : tensor<64x640x14x14xf32>
    %v633 = stablehlo.reshape %v632 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v634 = stablehlo.reshape %v633 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v635 = stablehlo.broadcast_in_dim %u8enmu, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v636 = stablehlo.subtract %v634, %v635 : tensor<64x640x14x14xf32>
    %v637 = stablehlo.broadcast_in_dim %u8envar, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v638 = stablehlo.constant dense<1.0e-5> : tensor<64x640x14x14xf32>
    %v639 = stablehlo.add %v637, %v638 : tensor<64x640x14x14xf32>
    %v640 = stablehlo.rsqrt %v639 : tensor<64x640x14x14xf32>
    %v641 = stablehlo.multiply %v636, %v640 : tensor<64x640x14x14xf32>
    %v642 = stablehlo.broadcast_in_dim %u8eg, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v643 = stablehlo.broadcast_in_dim %u8ebt, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v644 = stablehlo.multiply %v641, %v642 : tensor<64x640x14x14xf32>
    %v645 = stablehlo.add %v644, %v643 : tensor<64x640x14x14xf32>
    %v646 = stablehlo.reshape %v645 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v647 = stablehlo.constant dense<0.0> : tensor<64x125440xf32>
    %v648 = stablehlo.maximum %v646, %v647 : tensor<64x125440xf32>
    %v649 = stablehlo.reshape %v648 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v650 = stablehlo.convolution(%v649, %u8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<64x160x14x14xf32>
    %v651 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v652 = stablehlo.add %v650, %v651 : tensor<64x160x14x14xf32>
    %v653 = stablehlo.reshape %v652 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v654 = stablehlo.reshape %v653 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v655 = stablehlo.broadcast_in_dim %u8pnmu, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v656 = stablehlo.subtract %v654, %v655 : tensor<64x160x14x14xf32>
    %v657 = stablehlo.broadcast_in_dim %u8pnvar, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v658 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v659 = stablehlo.add %v657, %v658 : tensor<64x160x14x14xf32>
    %v660 = stablehlo.rsqrt %v659 : tensor<64x160x14x14xf32>
    %v661 = stablehlo.multiply %v656, %v660 : tensor<64x160x14x14xf32>
    %v662 = stablehlo.broadcast_in_dim %u8pg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v663 = stablehlo.broadcast_in_dim %u8pbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v664 = stablehlo.multiply %v661, %v662 : tensor<64x160x14x14xf32>
    %v665 = stablehlo.add %v664, %v663 : tensor<64x160x14x14xf32>
    %v666 = stablehlo.reshape %v665 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v667 = stablehlo.add %v666, %v608 : tensor<64x31360xf32>
    %v668 = stablehlo.reshape %v667 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v669 = stablehlo.convolution(%v668, %u9eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x160x14x14xf32>, tensor<320x160x1x1xf32>) -> tensor<64x320x14x14xf32>
    %v670 = stablehlo.broadcast_in_dim %zb320, dims = [1] : (tensor<320xf32>) -> tensor<64x320x14x14xf32>
    %v671 = stablehlo.add %v669, %v670 : tensor<64x320x14x14xf32>
    %v672 = stablehlo.reshape %v671 : (tensor<64x320x14x14xf32>) -> tensor<64x62720xf32>
    %v673 = stablehlo.reshape %v672 : (tensor<64x62720xf32>) -> tensor<64x320x14x14xf32>
    %v674 = stablehlo.broadcast_in_dim %u9enmu, dims = [1] : (tensor<320xf32>) -> tensor<64x320x14x14xf32>
    %v675 = stablehlo.subtract %v673, %v674 : tensor<64x320x14x14xf32>
    %v676 = stablehlo.broadcast_in_dim %u9envar, dims = [1] : (tensor<320xf32>) -> tensor<64x320x14x14xf32>
    %v677 = stablehlo.constant dense<1.0e-5> : tensor<64x320x14x14xf32>
    %v678 = stablehlo.add %v676, %v677 : tensor<64x320x14x14xf32>
    %v679 = stablehlo.rsqrt %v678 : tensor<64x320x14x14xf32>
    %v680 = stablehlo.multiply %v675, %v679 : tensor<64x320x14x14xf32>
    %v681 = stablehlo.broadcast_in_dim %u9eg, dims = [1] : (tensor<320xf32>) -> tensor<64x320x14x14xf32>
    %v682 = stablehlo.broadcast_in_dim %u9ebt, dims = [1] : (tensor<320xf32>) -> tensor<64x320x14x14xf32>
    %v683 = stablehlo.multiply %v680, %v681 : tensor<64x320x14x14xf32>
    %v684 = stablehlo.add %v683, %v682 : tensor<64x320x14x14xf32>
    %v685 = stablehlo.reshape %v684 : (tensor<64x320x14x14xf32>) -> tensor<64x62720xf32>
    %v686 = stablehlo.constant dense<0.0> : tensor<64x62720xf32>
    %v687 = stablehlo.maximum %v685, %v686 : tensor<64x62720xf32>
    %v688 = stablehlo.reshape %v687 : (tensor<64x62720xf32>) -> tensor<64x320x14x14xf32>
    %v689 = stablehlo.convolution(%v688, %u9pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x320x14x14xf32>, tensor<160x320x1x1xf32>) -> tensor<64x160x14x14xf32>
    %v690 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v691 = stablehlo.add %v689, %v690 : tensor<64x160x14x14xf32>
    %v692 = stablehlo.reshape %v691 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v693 = stablehlo.reshape %v692 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v694 = stablehlo.broadcast_in_dim %u9pnmu, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v695 = stablehlo.subtract %v693, %v694 : tensor<64x160x14x14xf32>
    %v696 = stablehlo.broadcast_in_dim %u9pnvar, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v697 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v698 = stablehlo.add %v696, %v697 : tensor<64x160x14x14xf32>
    %v699 = stablehlo.rsqrt %v698 : tensor<64x160x14x14xf32>
    %v700 = stablehlo.multiply %v695, %v699 : tensor<64x160x14x14xf32>
    %v701 = stablehlo.broadcast_in_dim %u9pg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v702 = stablehlo.broadcast_in_dim %u9pbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v703 = stablehlo.multiply %v700, %v701 : tensor<64x160x14x14xf32>
    %v704 = stablehlo.add %v703, %v702 : tensor<64x160x14x14xf32>
    %v705 = stablehlo.reshape %v704 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v706 = stablehlo.add %v705, %v667 : tensor<64x31360xf32>
    %v707 = stablehlo.reshape %v706 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v708 = stablehlo.convolution(%v707, %u10qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<64x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<64x160x14x14xf32>
    %v709 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v710 = stablehlo.add %v708, %v709 : tensor<64x160x14x14xf32>
    %v711 = stablehlo.reshape %v710 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v712 = stablehlo.reshape %v711 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v713 = stablehlo.broadcast_in_dim %u10qnmu, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v714 = stablehlo.subtract %v712, %v713 : tensor<64x160x14x14xf32>
    %v715 = stablehlo.broadcast_in_dim %u10qnvar, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v716 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v717 = stablehlo.add %v715, %v716 : tensor<64x160x14x14xf32>
    %v718 = stablehlo.rsqrt %v717 : tensor<64x160x14x14xf32>
    %v719 = stablehlo.multiply %v714, %v718 : tensor<64x160x14x14xf32>
    %v720 = stablehlo.broadcast_in_dim %u10qg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v721 = stablehlo.broadcast_in_dim %u10qbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v722 = stablehlo.multiply %v719, %v720 : tensor<64x160x14x14xf32>
    %v723 = stablehlo.add %v722, %v721 : tensor<64x160x14x14xf32>
    %v724 = stablehlo.reshape %v723 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v725 = stablehlo.constant dense<0.0> : tensor<64x31360xf32>
    %v726 = stablehlo.maximum %v724, %v725 : tensor<64x31360xf32>
    %v727 = stablehlo.reshape %v726 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v728 = stablehlo.convolution(%v727, %u10eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<64x640x14x14xf32>
    %v729 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v730 = stablehlo.add %v728, %v729 : tensor<64x640x14x14xf32>
    %v731 = stablehlo.reshape %v730 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v732 = stablehlo.reshape %v731 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v733 = stablehlo.broadcast_in_dim %u10enmu, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v734 = stablehlo.subtract %v732, %v733 : tensor<64x640x14x14xf32>
    %v735 = stablehlo.broadcast_in_dim %u10envar, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v736 = stablehlo.constant dense<1.0e-5> : tensor<64x640x14x14xf32>
    %v737 = stablehlo.add %v735, %v736 : tensor<64x640x14x14xf32>
    %v738 = stablehlo.rsqrt %v737 : tensor<64x640x14x14xf32>
    %v739 = stablehlo.multiply %v734, %v738 : tensor<64x640x14x14xf32>
    %v740 = stablehlo.broadcast_in_dim %u10eg, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v741 = stablehlo.broadcast_in_dim %u10ebt, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v742 = stablehlo.multiply %v739, %v740 : tensor<64x640x14x14xf32>
    %v743 = stablehlo.add %v742, %v741 : tensor<64x640x14x14xf32>
    %v744 = stablehlo.reshape %v743 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v745 = stablehlo.constant dense<0.0> : tensor<64x125440xf32>
    %v746 = stablehlo.maximum %v744, %v745 : tensor<64x125440xf32>
    %v747 = stablehlo.reshape %v746 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v748 = stablehlo.convolution(%v747, %u10pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<64x160x14x14xf32>
    %v749 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v750 = stablehlo.add %v748, %v749 : tensor<64x160x14x14xf32>
    %v751 = stablehlo.reshape %v750 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v752 = stablehlo.reshape %v751 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v753 = stablehlo.broadcast_in_dim %u10pnmu, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v754 = stablehlo.subtract %v752, %v753 : tensor<64x160x14x14xf32>
    %v755 = stablehlo.broadcast_in_dim %u10pnvar, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v756 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v757 = stablehlo.add %v755, %v756 : tensor<64x160x14x14xf32>
    %v758 = stablehlo.rsqrt %v757 : tensor<64x160x14x14xf32>
    %v759 = stablehlo.multiply %v754, %v758 : tensor<64x160x14x14xf32>
    %v760 = stablehlo.broadcast_in_dim %u10pg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v761 = stablehlo.broadcast_in_dim %u10pbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v762 = stablehlo.multiply %v759, %v760 : tensor<64x160x14x14xf32>
    %v763 = stablehlo.add %v762, %v761 : tensor<64x160x14x14xf32>
    %v764 = stablehlo.reshape %v763 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v765 = stablehlo.add %v764, %v706 : tensor<64x31360xf32>
    %v766 = stablehlo.reshape %v765 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v767 = stablehlo.convolution(%v766, %u11qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<64x160x14x14xf32>, tensor<160x1x5x5xf32>) -> tensor<64x160x7x7xf32>
    %v768 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x7x7xf32>
    %v769 = stablehlo.add %v767, %v768 : tensor<64x160x7x7xf32>
    %v770 = stablehlo.reshape %v769 : (tensor<64x160x7x7xf32>) -> tensor<64x7840xf32>
    %v771 = stablehlo.reshape %v770 : (tensor<64x7840xf32>) -> tensor<64x160x7x7xf32>
    %v772 = stablehlo.broadcast_in_dim %u11qnmu, dims = [1] : (tensor<160xf32>) -> tensor<64x160x7x7xf32>
    %v773 = stablehlo.subtract %v771, %v772 : tensor<64x160x7x7xf32>
    %v774 = stablehlo.broadcast_in_dim %u11qnvar, dims = [1] : (tensor<160xf32>) -> tensor<64x160x7x7xf32>
    %v775 = stablehlo.constant dense<1.0e-5> : tensor<64x160x7x7xf32>
    %v776 = stablehlo.add %v774, %v775 : tensor<64x160x7x7xf32>
    %v777 = stablehlo.rsqrt %v776 : tensor<64x160x7x7xf32>
    %v778 = stablehlo.multiply %v773, %v777 : tensor<64x160x7x7xf32>
    %v779 = stablehlo.broadcast_in_dim %u11qg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x7x7xf32>
    %v780 = stablehlo.broadcast_in_dim %u11qbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x7x7xf32>
    %v781 = stablehlo.multiply %v778, %v779 : tensor<64x160x7x7xf32>
    %v782 = stablehlo.add %v781, %v780 : tensor<64x160x7x7xf32>
    %v783 = stablehlo.reshape %v782 : (tensor<64x160x7x7xf32>) -> tensor<64x7840xf32>
    %v784 = stablehlo.constant dense<0.0> : tensor<64x7840xf32>
    %v785 = stablehlo.maximum %v783, %v784 : tensor<64x7840xf32>
    %v786 = stablehlo.reshape %v785 : (tensor<64x7840xf32>) -> tensor<64x160x7x7xf32>
    %v787 = stablehlo.convolution(%v786, %u11eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x160x7x7xf32>, tensor<960x160x1x1xf32>) -> tensor<64x960x7x7xf32>
    %v788 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v789 = stablehlo.add %v787, %v788 : tensor<64x960x7x7xf32>
    %v790 = stablehlo.reshape %v789 : (tensor<64x960x7x7xf32>) -> tensor<64x47040xf32>
    %v791 = stablehlo.reshape %v790 : (tensor<64x47040xf32>) -> tensor<64x960x7x7xf32>
    %v792 = stablehlo.broadcast_in_dim %u11enmu, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v793 = stablehlo.subtract %v791, %v792 : tensor<64x960x7x7xf32>
    %v794 = stablehlo.broadcast_in_dim %u11envar, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v795 = stablehlo.constant dense<1.0e-5> : tensor<64x960x7x7xf32>
    %v796 = stablehlo.add %v794, %v795 : tensor<64x960x7x7xf32>
    %v797 = stablehlo.rsqrt %v796 : tensor<64x960x7x7xf32>
    %v798 = stablehlo.multiply %v793, %v797 : tensor<64x960x7x7xf32>
    %v799 = stablehlo.broadcast_in_dim %u11eg, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v800 = stablehlo.broadcast_in_dim %u11ebt, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v801 = stablehlo.multiply %v798, %v799 : tensor<64x960x7x7xf32>
    %v802 = stablehlo.add %v801, %v800 : tensor<64x960x7x7xf32>
    %v803 = stablehlo.reshape %v802 : (tensor<64x960x7x7xf32>) -> tensor<64x47040xf32>
    %v804 = stablehlo.constant dense<0.0> : tensor<64x47040xf32>
    %v805 = stablehlo.maximum %v803, %v804 : tensor<64x47040xf32>
    %v806 = stablehlo.reshape %v805 : (tensor<64x47040xf32>) -> tensor<64x960x7x7xf32>
    %v807 = stablehlo.convolution(%v806, %u11dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<64x960x7x7xf32>, tensor<960x1x5x5xf32>) -> tensor<64x960x7x7xf32>
    %v808 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v809 = stablehlo.add %v807, %v808 : tensor<64x960x7x7xf32>
    %v810 = stablehlo.reshape %v809 : (tensor<64x960x7x7xf32>) -> tensor<64x47040xf32>
    %v811 = stablehlo.reshape %v810 : (tensor<64x47040xf32>) -> tensor<64x960x7x7xf32>
    %v812 = stablehlo.broadcast_in_dim %u11dnmu, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v813 = stablehlo.subtract %v811, %v812 : tensor<64x960x7x7xf32>
    %v814 = stablehlo.broadcast_in_dim %u11dnvar, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v815 = stablehlo.constant dense<1.0e-5> : tensor<64x960x7x7xf32>
    %v816 = stablehlo.add %v814, %v815 : tensor<64x960x7x7xf32>
    %v817 = stablehlo.rsqrt %v816 : tensor<64x960x7x7xf32>
    %v818 = stablehlo.multiply %v813, %v817 : tensor<64x960x7x7xf32>
    %v819 = stablehlo.broadcast_in_dim %u11dg, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v820 = stablehlo.broadcast_in_dim %u11dbt, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v821 = stablehlo.multiply %v818, %v819 : tensor<64x960x7x7xf32>
    %v822 = stablehlo.add %v821, %v820 : tensor<64x960x7x7xf32>
    %v823 = stablehlo.reshape %v822 : (tensor<64x960x7x7xf32>) -> tensor<64x47040xf32>
    %v824 = stablehlo.constant dense<0.0> : tensor<64x47040xf32>
    %v825 = stablehlo.maximum %v823, %v824 : tensor<64x47040xf32>
    %v826 = stablehlo.reshape %v825 : (tensor<64x47040xf32>) -> tensor<64x960x7x7xf32>
    %v827 = stablehlo.convolution(%v826, %u11pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x960x7x7xf32>, tensor<256x960x1x1xf32>) -> tensor<64x256x7x7xf32>
    %v828 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v829 = stablehlo.add %v827, %v828 : tensor<64x256x7x7xf32>
    %v830 = stablehlo.reshape %v829 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v831 = stablehlo.reshape %v830 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v832 = stablehlo.broadcast_in_dim %u11pnmu, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v833 = stablehlo.subtract %v831, %v832 : tensor<64x256x7x7xf32>
    %v834 = stablehlo.broadcast_in_dim %u11pnvar, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v835 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v836 = stablehlo.add %v834, %v835 : tensor<64x256x7x7xf32>
    %v837 = stablehlo.rsqrt %v836 : tensor<64x256x7x7xf32>
    %v838 = stablehlo.multiply %v833, %v837 : tensor<64x256x7x7xf32>
    %v839 = stablehlo.broadcast_in_dim %u11pg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v840 = stablehlo.broadcast_in_dim %u11pbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v841 = stablehlo.multiply %v838, %v839 : tensor<64x256x7x7xf32>
    %v842 = stablehlo.add %v841, %v840 : tensor<64x256x7x7xf32>
    %v843 = stablehlo.reshape %v842 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v844 = stablehlo.reshape %v843 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v845 = stablehlo.convolution(%v844, %u12qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<64x256x7x7xf32>, tensor<256x1x5x5xf32>) -> tensor<64x256x7x7xf32>
    %v846 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v847 = stablehlo.add %v845, %v846 : tensor<64x256x7x7xf32>
    %v848 = stablehlo.reshape %v847 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v849 = stablehlo.reshape %v848 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v850 = stablehlo.broadcast_in_dim %u12qnmu, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v851 = stablehlo.subtract %v849, %v850 : tensor<64x256x7x7xf32>
    %v852 = stablehlo.broadcast_in_dim %u12qnvar, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v853 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v854 = stablehlo.add %v852, %v853 : tensor<64x256x7x7xf32>
    %v855 = stablehlo.rsqrt %v854 : tensor<64x256x7x7xf32>
    %v856 = stablehlo.multiply %v851, %v855 : tensor<64x256x7x7xf32>
    %v857 = stablehlo.broadcast_in_dim %u12qg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v858 = stablehlo.broadcast_in_dim %u12qbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v859 = stablehlo.multiply %v856, %v857 : tensor<64x256x7x7xf32>
    %v860 = stablehlo.add %v859, %v858 : tensor<64x256x7x7xf32>
    %v861 = stablehlo.reshape %v860 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v862 = stablehlo.constant dense<0.0> : tensor<64x12544xf32>
    %v863 = stablehlo.maximum %v861, %v862 : tensor<64x12544xf32>
    %v864 = stablehlo.reshape %v863 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v865 = stablehlo.convolution(%v864, %u12eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<64x1024x7x7xf32>
    %v866 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v867 = stablehlo.add %v865, %v866 : tensor<64x1024x7x7xf32>
    %v868 = stablehlo.reshape %v867 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v869 = stablehlo.reshape %v868 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v870 = stablehlo.broadcast_in_dim %u12enmu, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v871 = stablehlo.subtract %v869, %v870 : tensor<64x1024x7x7xf32>
    %v872 = stablehlo.broadcast_in_dim %u12envar, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v873 = stablehlo.constant dense<1.0e-5> : tensor<64x1024x7x7xf32>
    %v874 = stablehlo.add %v872, %v873 : tensor<64x1024x7x7xf32>
    %v875 = stablehlo.rsqrt %v874 : tensor<64x1024x7x7xf32>
    %v876 = stablehlo.multiply %v871, %v875 : tensor<64x1024x7x7xf32>
    %v877 = stablehlo.broadcast_in_dim %u12eg, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v878 = stablehlo.broadcast_in_dim %u12ebt, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v879 = stablehlo.multiply %v876, %v877 : tensor<64x1024x7x7xf32>
    %v880 = stablehlo.add %v879, %v878 : tensor<64x1024x7x7xf32>
    %v881 = stablehlo.reshape %v880 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v882 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v883 = stablehlo.maximum %v881, %v882 : tensor<64x50176xf32>
    %v884 = stablehlo.reshape %v883 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v885 = stablehlo.convolution(%v884, %u12dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<64x1024x7x7xf32>, tensor<1024x1x5x5xf32>) -> tensor<64x1024x7x7xf32>
    %v886 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v887 = stablehlo.add %v885, %v886 : tensor<64x1024x7x7xf32>
    %v888 = stablehlo.reshape %v887 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v889 = stablehlo.reshape %v888 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v890 = stablehlo.broadcast_in_dim %u12dnmu, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v891 = stablehlo.subtract %v889, %v890 : tensor<64x1024x7x7xf32>
    %v892 = stablehlo.broadcast_in_dim %u12dnvar, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v893 = stablehlo.constant dense<1.0e-5> : tensor<64x1024x7x7xf32>
    %v894 = stablehlo.add %v892, %v893 : tensor<64x1024x7x7xf32>
    %v895 = stablehlo.rsqrt %v894 : tensor<64x1024x7x7xf32>
    %v896 = stablehlo.multiply %v891, %v895 : tensor<64x1024x7x7xf32>
    %v897 = stablehlo.broadcast_in_dim %u12dg, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v898 = stablehlo.broadcast_in_dim %u12dbt, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v899 = stablehlo.multiply %v896, %v897 : tensor<64x1024x7x7xf32>
    %v900 = stablehlo.add %v899, %v898 : tensor<64x1024x7x7xf32>
    %v901 = stablehlo.reshape %v900 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v902 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v903 = stablehlo.maximum %v901, %v902 : tensor<64x50176xf32>
    %v904 = stablehlo.reshape %v903 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v905 = stablehlo.convolution(%v904, %u12pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<64x256x7x7xf32>
    %v906 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v907 = stablehlo.add %v905, %v906 : tensor<64x256x7x7xf32>
    %v908 = stablehlo.reshape %v907 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v909 = stablehlo.reshape %v908 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v910 = stablehlo.broadcast_in_dim %u12pnmu, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v911 = stablehlo.subtract %v909, %v910 : tensor<64x256x7x7xf32>
    %v912 = stablehlo.broadcast_in_dim %u12pnvar, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v913 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v914 = stablehlo.add %v912, %v913 : tensor<64x256x7x7xf32>
    %v915 = stablehlo.rsqrt %v914 : tensor<64x256x7x7xf32>
    %v916 = stablehlo.multiply %v911, %v915 : tensor<64x256x7x7xf32>
    %v917 = stablehlo.broadcast_in_dim %u12pg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v918 = stablehlo.broadcast_in_dim %u12pbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v919 = stablehlo.multiply %v916, %v917 : tensor<64x256x7x7xf32>
    %v920 = stablehlo.add %v919, %v918 : tensor<64x256x7x7xf32>
    %v921 = stablehlo.reshape %v920 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v922 = stablehlo.add %v921, %v843 : tensor<64x12544xf32>
    %v923 = stablehlo.reshape %v922 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v924 = stablehlo.convolution(%v923, %u13qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<64x256x7x7xf32>, tensor<256x1x3x3xf32>) -> tensor<64x256x7x7xf32>
    %v925 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v926 = stablehlo.add %v924, %v925 : tensor<64x256x7x7xf32>
    %v927 = stablehlo.reshape %v926 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v928 = stablehlo.reshape %v927 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v929 = stablehlo.broadcast_in_dim %u13qnmu, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v930 = stablehlo.subtract %v928, %v929 : tensor<64x256x7x7xf32>
    %v931 = stablehlo.broadcast_in_dim %u13qnvar, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v932 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v933 = stablehlo.add %v931, %v932 : tensor<64x256x7x7xf32>
    %v934 = stablehlo.rsqrt %v933 : tensor<64x256x7x7xf32>
    %v935 = stablehlo.multiply %v930, %v934 : tensor<64x256x7x7xf32>
    %v936 = stablehlo.broadcast_in_dim %u13qg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v937 = stablehlo.broadcast_in_dim %u13qbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v938 = stablehlo.multiply %v935, %v936 : tensor<64x256x7x7xf32>
    %v939 = stablehlo.add %v938, %v937 : tensor<64x256x7x7xf32>
    %v940 = stablehlo.reshape %v939 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v941 = stablehlo.constant dense<0.0> : tensor<64x12544xf32>
    %v942 = stablehlo.maximum %v940, %v941 : tensor<64x12544xf32>
    %v943 = stablehlo.reshape %v942 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v944 = stablehlo.convolution(%v943, %u13eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<64x1024x7x7xf32>
    %v945 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v946 = stablehlo.add %v944, %v945 : tensor<64x1024x7x7xf32>
    %v947 = stablehlo.reshape %v946 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v948 = stablehlo.reshape %v947 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v949 = stablehlo.broadcast_in_dim %u13enmu, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v950 = stablehlo.subtract %v948, %v949 : tensor<64x1024x7x7xf32>
    %v951 = stablehlo.broadcast_in_dim %u13envar, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v952 = stablehlo.constant dense<1.0e-5> : tensor<64x1024x7x7xf32>
    %v953 = stablehlo.add %v951, %v952 : tensor<64x1024x7x7xf32>
    %v954 = stablehlo.rsqrt %v953 : tensor<64x1024x7x7xf32>
    %v955 = stablehlo.multiply %v950, %v954 : tensor<64x1024x7x7xf32>
    %v956 = stablehlo.broadcast_in_dim %u13eg, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v957 = stablehlo.broadcast_in_dim %u13ebt, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v958 = stablehlo.multiply %v955, %v956 : tensor<64x1024x7x7xf32>
    %v959 = stablehlo.add %v958, %v957 : tensor<64x1024x7x7xf32>
    %v960 = stablehlo.reshape %v959 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v961 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v962 = stablehlo.maximum %v960, %v961 : tensor<64x50176xf32>
    %v963 = stablehlo.reshape %v962 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v964 = stablehlo.convolution(%v963, %u13dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<64x1024x7x7xf32>, tensor<1024x1x5x5xf32>) -> tensor<64x1024x7x7xf32>
    %v965 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v966 = stablehlo.add %v964, %v965 : tensor<64x1024x7x7xf32>
    %v967 = stablehlo.reshape %v966 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v968 = stablehlo.reshape %v967 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v969 = stablehlo.broadcast_in_dim %u13dnmu, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v970 = stablehlo.subtract %v968, %v969 : tensor<64x1024x7x7xf32>
    %v971 = stablehlo.broadcast_in_dim %u13dnvar, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v972 = stablehlo.constant dense<1.0e-5> : tensor<64x1024x7x7xf32>
    %v973 = stablehlo.add %v971, %v972 : tensor<64x1024x7x7xf32>
    %v974 = stablehlo.rsqrt %v973 : tensor<64x1024x7x7xf32>
    %v975 = stablehlo.multiply %v970, %v974 : tensor<64x1024x7x7xf32>
    %v976 = stablehlo.broadcast_in_dim %u13dg, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v977 = stablehlo.broadcast_in_dim %u13dbt, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v978 = stablehlo.multiply %v975, %v976 : tensor<64x1024x7x7xf32>
    %v979 = stablehlo.add %v978, %v977 : tensor<64x1024x7x7xf32>
    %v980 = stablehlo.reshape %v979 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v981 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v982 = stablehlo.maximum %v980, %v981 : tensor<64x50176xf32>
    %v983 = stablehlo.reshape %v982 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v984 = stablehlo.convolution(%v983, %u13pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<64x256x7x7xf32>
    %v985 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v986 = stablehlo.add %v984, %v985 : tensor<64x256x7x7xf32>
    %v987 = stablehlo.reshape %v986 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v988 = stablehlo.reshape %v987 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v989 = stablehlo.broadcast_in_dim %u13pnmu, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v990 = stablehlo.subtract %v988, %v989 : tensor<64x256x7x7xf32>
    %v991 = stablehlo.broadcast_in_dim %u13pnvar, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v992 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v993 = stablehlo.add %v991, %v992 : tensor<64x256x7x7xf32>
    %v994 = stablehlo.rsqrt %v993 : tensor<64x256x7x7xf32>
    %v995 = stablehlo.multiply %v990, %v994 : tensor<64x256x7x7xf32>
    %v996 = stablehlo.broadcast_in_dim %u13pg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v997 = stablehlo.broadcast_in_dim %u13pbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v998 = stablehlo.multiply %v995, %v996 : tensor<64x256x7x7xf32>
    %v999 = stablehlo.add %v998, %v997 : tensor<64x256x7x7xf32>
    %v1000 = stablehlo.reshape %v999 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1001 = stablehlo.add %v1000, %v922 : tensor<64x12544xf32>
    %v1002 = stablehlo.reshape %v1001 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1003 = stablehlo.convolution(%v1002, %u14qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<64x256x7x7xf32>, tensor<256x1x3x3xf32>) -> tensor<64x256x7x7xf32>
    %v1004 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1005 = stablehlo.add %v1003, %v1004 : tensor<64x256x7x7xf32>
    %v1006 = stablehlo.reshape %v1005 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1007 = stablehlo.reshape %v1006 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1008 = stablehlo.broadcast_in_dim %u14qnmu, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1009 = stablehlo.subtract %v1007, %v1008 : tensor<64x256x7x7xf32>
    %v1010 = stablehlo.broadcast_in_dim %u14qnvar, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1011 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v1012 = stablehlo.add %v1010, %v1011 : tensor<64x256x7x7xf32>
    %v1013 = stablehlo.rsqrt %v1012 : tensor<64x256x7x7xf32>
    %v1014 = stablehlo.multiply %v1009, %v1013 : tensor<64x256x7x7xf32>
    %v1015 = stablehlo.broadcast_in_dim %u14qg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1016 = stablehlo.broadcast_in_dim %u14qbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1017 = stablehlo.multiply %v1014, %v1015 : tensor<64x256x7x7xf32>
    %v1018 = stablehlo.add %v1017, %v1016 : tensor<64x256x7x7xf32>
    %v1019 = stablehlo.reshape %v1018 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1020 = stablehlo.constant dense<0.0> : tensor<64x12544xf32>
    %v1021 = stablehlo.maximum %v1019, %v1020 : tensor<64x12544xf32>
    %v1022 = stablehlo.reshape %v1021 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1023 = stablehlo.convolution(%v1022, %u14eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<64x1024x7x7xf32>
    %v1024 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1025 = stablehlo.add %v1023, %v1024 : tensor<64x1024x7x7xf32>
    %v1026 = stablehlo.reshape %v1025 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1027 = stablehlo.reshape %v1026 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1028 = stablehlo.broadcast_in_dim %u14enmu, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1029 = stablehlo.subtract %v1027, %v1028 : tensor<64x1024x7x7xf32>
    %v1030 = stablehlo.broadcast_in_dim %u14envar, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1031 = stablehlo.constant dense<1.0e-5> : tensor<64x1024x7x7xf32>
    %v1032 = stablehlo.add %v1030, %v1031 : tensor<64x1024x7x7xf32>
    %v1033 = stablehlo.rsqrt %v1032 : tensor<64x1024x7x7xf32>
    %v1034 = stablehlo.multiply %v1029, %v1033 : tensor<64x1024x7x7xf32>
    %v1035 = stablehlo.broadcast_in_dim %u14eg, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1036 = stablehlo.broadcast_in_dim %u14ebt, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1037 = stablehlo.multiply %v1034, %v1035 : tensor<64x1024x7x7xf32>
    %v1038 = stablehlo.add %v1037, %v1036 : tensor<64x1024x7x7xf32>
    %v1039 = stablehlo.reshape %v1038 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1040 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1041 = stablehlo.maximum %v1039, %v1040 : tensor<64x50176xf32>
    %v1042 = stablehlo.reshape %v1041 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1043 = stablehlo.convolution(%v1042, %u14dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<64x1024x7x7xf32>, tensor<1024x1x5x5xf32>) -> tensor<64x1024x7x7xf32>
    %v1044 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1045 = stablehlo.add %v1043, %v1044 : tensor<64x1024x7x7xf32>
    %v1046 = stablehlo.reshape %v1045 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1047 = stablehlo.reshape %v1046 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1048 = stablehlo.broadcast_in_dim %u14dnmu, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1049 = stablehlo.subtract %v1047, %v1048 : tensor<64x1024x7x7xf32>
    %v1050 = stablehlo.broadcast_in_dim %u14dnvar, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1051 = stablehlo.constant dense<1.0e-5> : tensor<64x1024x7x7xf32>
    %v1052 = stablehlo.add %v1050, %v1051 : tensor<64x1024x7x7xf32>
    %v1053 = stablehlo.rsqrt %v1052 : tensor<64x1024x7x7xf32>
    %v1054 = stablehlo.multiply %v1049, %v1053 : tensor<64x1024x7x7xf32>
    %v1055 = stablehlo.broadcast_in_dim %u14dg, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1056 = stablehlo.broadcast_in_dim %u14dbt, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1057 = stablehlo.multiply %v1054, %v1055 : tensor<64x1024x7x7xf32>
    %v1058 = stablehlo.add %v1057, %v1056 : tensor<64x1024x7x7xf32>
    %v1059 = stablehlo.reshape %v1058 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1060 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1061 = stablehlo.maximum %v1059, %v1060 : tensor<64x50176xf32>
    %v1062 = stablehlo.reshape %v1061 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1063 = stablehlo.convolution(%v1062, %u14pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<64x256x7x7xf32>
    %v1064 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1065 = stablehlo.add %v1063, %v1064 : tensor<64x256x7x7xf32>
    %v1066 = stablehlo.reshape %v1065 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1067 = stablehlo.reshape %v1066 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1068 = stablehlo.broadcast_in_dim %u14pnmu, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1069 = stablehlo.subtract %v1067, %v1068 : tensor<64x256x7x7xf32>
    %v1070 = stablehlo.broadcast_in_dim %u14pnvar, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1071 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v1072 = stablehlo.add %v1070, %v1071 : tensor<64x256x7x7xf32>
    %v1073 = stablehlo.rsqrt %v1072 : tensor<64x256x7x7xf32>
    %v1074 = stablehlo.multiply %v1069, %v1073 : tensor<64x256x7x7xf32>
    %v1075 = stablehlo.broadcast_in_dim %u14pg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1076 = stablehlo.broadcast_in_dim %u14pbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1077 = stablehlo.multiply %v1074, %v1075 : tensor<64x256x7x7xf32>
    %v1078 = stablehlo.add %v1077, %v1076 : tensor<64x256x7x7xf32>
    %v1079 = stablehlo.reshape %v1078 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1080 = stablehlo.add %v1079, %v1001 : tensor<64x12544xf32>
    %v1081 = stablehlo.reshape %v1080 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1082 = stablehlo.convolution(%v1081, %u15eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<64x1024x7x7xf32>
    %v1083 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1084 = stablehlo.add %v1082, %v1083 : tensor<64x1024x7x7xf32>
    %v1085 = stablehlo.reshape %v1084 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1086 = stablehlo.reshape %v1085 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1087 = stablehlo.broadcast_in_dim %u15enmu, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1088 = stablehlo.subtract %v1086, %v1087 : tensor<64x1024x7x7xf32>
    %v1089 = stablehlo.broadcast_in_dim %u15envar, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1090 = stablehlo.constant dense<1.0e-5> : tensor<64x1024x7x7xf32>
    %v1091 = stablehlo.add %v1089, %v1090 : tensor<64x1024x7x7xf32>
    %v1092 = stablehlo.rsqrt %v1091 : tensor<64x1024x7x7xf32>
    %v1093 = stablehlo.multiply %v1088, %v1092 : tensor<64x1024x7x7xf32>
    %v1094 = stablehlo.broadcast_in_dim %u15eg, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1095 = stablehlo.broadcast_in_dim %u15ebt, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1096 = stablehlo.multiply %v1093, %v1094 : tensor<64x1024x7x7xf32>
    %v1097 = stablehlo.add %v1096, %v1095 : tensor<64x1024x7x7xf32>
    %v1098 = stablehlo.reshape %v1097 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1099 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1100 = stablehlo.maximum %v1098, %v1099 : tensor<64x50176xf32>
    %v1101 = stablehlo.reshape %v1100 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1102 = stablehlo.convolution(%v1101, %u15pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<64x256x7x7xf32>
    %v1103 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1104 = stablehlo.add %v1102, %v1103 : tensor<64x256x7x7xf32>
    %v1105 = stablehlo.reshape %v1104 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1106 = stablehlo.reshape %v1105 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1107 = stablehlo.broadcast_in_dim %u15pnmu, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1108 = stablehlo.subtract %v1106, %v1107 : tensor<64x256x7x7xf32>
    %v1109 = stablehlo.broadcast_in_dim %u15pnvar, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1110 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v1111 = stablehlo.add %v1109, %v1110 : tensor<64x256x7x7xf32>
    %v1112 = stablehlo.rsqrt %v1111 : tensor<64x256x7x7xf32>
    %v1113 = stablehlo.multiply %v1108, %v1112 : tensor<64x256x7x7xf32>
    %v1114 = stablehlo.broadcast_in_dim %u15pg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1115 = stablehlo.broadcast_in_dim %u15pbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1116 = stablehlo.multiply %v1113, %v1114 : tensor<64x256x7x7xf32>
    %v1117 = stablehlo.add %v1116, %v1115 : tensor<64x256x7x7xf32>
    %v1118 = stablehlo.reshape %v1117 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1119 = stablehlo.add %v1118, %v1080 : tensor<64x12544xf32>
    %v1120 = stablehlo.reshape %v1119 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1121 = stablehlo.convolution(%v1120, %u16qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<64x256x7x7xf32>, tensor<256x1x3x3xf32>) -> tensor<64x256x7x7xf32>
    %v1122 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1123 = stablehlo.add %v1121, %v1122 : tensor<64x256x7x7xf32>
    %v1124 = stablehlo.reshape %v1123 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1125 = stablehlo.reshape %v1124 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1126 = stablehlo.broadcast_in_dim %u16qnmu, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1127 = stablehlo.subtract %v1125, %v1126 : tensor<64x256x7x7xf32>
    %v1128 = stablehlo.broadcast_in_dim %u16qnvar, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1129 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v1130 = stablehlo.add %v1128, %v1129 : tensor<64x256x7x7xf32>
    %v1131 = stablehlo.rsqrt %v1130 : tensor<64x256x7x7xf32>
    %v1132 = stablehlo.multiply %v1127, %v1131 : tensor<64x256x7x7xf32>
    %v1133 = stablehlo.broadcast_in_dim %u16qg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1134 = stablehlo.broadcast_in_dim %u16qbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1135 = stablehlo.multiply %v1132, %v1133 : tensor<64x256x7x7xf32>
    %v1136 = stablehlo.add %v1135, %v1134 : tensor<64x256x7x7xf32>
    %v1137 = stablehlo.reshape %v1136 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1138 = stablehlo.constant dense<0.0> : tensor<64x12544xf32>
    %v1139 = stablehlo.maximum %v1137, %v1138 : tensor<64x12544xf32>
    %v1140 = stablehlo.reshape %v1139 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1141 = stablehlo.convolution(%v1140, %u16eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<64x1024x7x7xf32>
    %v1142 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1143 = stablehlo.add %v1141, %v1142 : tensor<64x1024x7x7xf32>
    %v1144 = stablehlo.reshape %v1143 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1145 = stablehlo.reshape %v1144 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1146 = stablehlo.broadcast_in_dim %u16enmu, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1147 = stablehlo.subtract %v1145, %v1146 : tensor<64x1024x7x7xf32>
    %v1148 = stablehlo.broadcast_in_dim %u16envar, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1149 = stablehlo.constant dense<1.0e-5> : tensor<64x1024x7x7xf32>
    %v1150 = stablehlo.add %v1148, %v1149 : tensor<64x1024x7x7xf32>
    %v1151 = stablehlo.rsqrt %v1150 : tensor<64x1024x7x7xf32>
    %v1152 = stablehlo.multiply %v1147, %v1151 : tensor<64x1024x7x7xf32>
    %v1153 = stablehlo.broadcast_in_dim %u16eg, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1154 = stablehlo.broadcast_in_dim %u16ebt, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1155 = stablehlo.multiply %v1152, %v1153 : tensor<64x1024x7x7xf32>
    %v1156 = stablehlo.add %v1155, %v1154 : tensor<64x1024x7x7xf32>
    %v1157 = stablehlo.reshape %v1156 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1158 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1159 = stablehlo.maximum %v1157, %v1158 : tensor<64x50176xf32>
    %v1160 = stablehlo.reshape %v1159 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1161 = stablehlo.convolution(%v1160, %u16pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<64x256x7x7xf32>
    %v1162 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1163 = stablehlo.add %v1161, %v1162 : tensor<64x256x7x7xf32>
    %v1164 = stablehlo.reshape %v1163 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1165 = stablehlo.reshape %v1164 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1166 = stablehlo.broadcast_in_dim %u16pnmu, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1167 = stablehlo.subtract %v1165, %v1166 : tensor<64x256x7x7xf32>
    %v1168 = stablehlo.broadcast_in_dim %u16pnvar, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1169 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v1170 = stablehlo.add %v1168, %v1169 : tensor<64x256x7x7xf32>
    %v1171 = stablehlo.rsqrt %v1170 : tensor<64x256x7x7xf32>
    %v1172 = stablehlo.multiply %v1167, %v1171 : tensor<64x256x7x7xf32>
    %v1173 = stablehlo.broadcast_in_dim %u16pg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1174 = stablehlo.broadcast_in_dim %u16pbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1175 = stablehlo.multiply %v1172, %v1173 : tensor<64x256x7x7xf32>
    %v1176 = stablehlo.add %v1175, %v1174 : tensor<64x256x7x7xf32>
    %v1177 = stablehlo.reshape %v1176 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1178 = stablehlo.add %v1177, %v1119 : tensor<64x12544xf32>
    %v1179 = stablehlo.reshape %v1178 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1180 = stablehlo.convolution(%v1179, %u17qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<64x256x7x7xf32>, tensor<256x1x3x3xf32>) -> tensor<64x256x7x7xf32>
    %v1181 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1182 = stablehlo.add %v1180, %v1181 : tensor<64x256x7x7xf32>
    %v1183 = stablehlo.reshape %v1182 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1184 = stablehlo.reshape %v1183 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1185 = stablehlo.broadcast_in_dim %u17qnmu, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1186 = stablehlo.subtract %v1184, %v1185 : tensor<64x256x7x7xf32>
    %v1187 = stablehlo.broadcast_in_dim %u17qnvar, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1188 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v1189 = stablehlo.add %v1187, %v1188 : tensor<64x256x7x7xf32>
    %v1190 = stablehlo.rsqrt %v1189 : tensor<64x256x7x7xf32>
    %v1191 = stablehlo.multiply %v1186, %v1190 : tensor<64x256x7x7xf32>
    %v1192 = stablehlo.broadcast_in_dim %u17qg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1193 = stablehlo.broadcast_in_dim %u17qbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1194 = stablehlo.multiply %v1191, %v1192 : tensor<64x256x7x7xf32>
    %v1195 = stablehlo.add %v1194, %v1193 : tensor<64x256x7x7xf32>
    %v1196 = stablehlo.reshape %v1195 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1197 = stablehlo.constant dense<0.0> : tensor<64x12544xf32>
    %v1198 = stablehlo.maximum %v1196, %v1197 : tensor<64x12544xf32>
    %v1199 = stablehlo.reshape %v1198 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1200 = stablehlo.convolution(%v1199, %u17eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x7x7xf32>, tensor<512x256x1x1xf32>) -> tensor<64x512x7x7xf32>
    %v1201 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1202 = stablehlo.add %v1200, %v1201 : tensor<64x512x7x7xf32>
    %v1203 = stablehlo.reshape %v1202 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1204 = stablehlo.reshape %v1203 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1205 = stablehlo.broadcast_in_dim %u17enmu, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1206 = stablehlo.subtract %v1204, %v1205 : tensor<64x512x7x7xf32>
    %v1207 = stablehlo.broadcast_in_dim %u17envar, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1208 = stablehlo.constant dense<1.0e-5> : tensor<64x512x7x7xf32>
    %v1209 = stablehlo.add %v1207, %v1208 : tensor<64x512x7x7xf32>
    %v1210 = stablehlo.rsqrt %v1209 : tensor<64x512x7x7xf32>
    %v1211 = stablehlo.multiply %v1206, %v1210 : tensor<64x512x7x7xf32>
    %v1212 = stablehlo.broadcast_in_dim %u17eg, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1213 = stablehlo.broadcast_in_dim %u17ebt, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1214 = stablehlo.multiply %v1211, %v1212 : tensor<64x512x7x7xf32>
    %v1215 = stablehlo.add %v1214, %v1213 : tensor<64x512x7x7xf32>
    %v1216 = stablehlo.reshape %v1215 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1217 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v1218 = stablehlo.maximum %v1216, %v1217 : tensor<64x25088xf32>
    %v1219 = stablehlo.reshape %v1218 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1220 = stablehlo.convolution(%v1219, %u17dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<64x512x7x7xf32>, tensor<512x1x5x5xf32>) -> tensor<64x512x7x7xf32>
    %v1221 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1222 = stablehlo.add %v1220, %v1221 : tensor<64x512x7x7xf32>
    %v1223 = stablehlo.reshape %v1222 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1224 = stablehlo.reshape %v1223 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1225 = stablehlo.broadcast_in_dim %u17dnmu, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1226 = stablehlo.subtract %v1224, %v1225 : tensor<64x512x7x7xf32>
    %v1227 = stablehlo.broadcast_in_dim %u17dnvar, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1228 = stablehlo.constant dense<1.0e-5> : tensor<64x512x7x7xf32>
    %v1229 = stablehlo.add %v1227, %v1228 : tensor<64x512x7x7xf32>
    %v1230 = stablehlo.rsqrt %v1229 : tensor<64x512x7x7xf32>
    %v1231 = stablehlo.multiply %v1226, %v1230 : tensor<64x512x7x7xf32>
    %v1232 = stablehlo.broadcast_in_dim %u17dg, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1233 = stablehlo.broadcast_in_dim %u17dbt, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1234 = stablehlo.multiply %v1231, %v1232 : tensor<64x512x7x7xf32>
    %v1235 = stablehlo.add %v1234, %v1233 : tensor<64x512x7x7xf32>
    %v1236 = stablehlo.reshape %v1235 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1237 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v1238 = stablehlo.maximum %v1236, %v1237 : tensor<64x25088xf32>
    %v1239 = stablehlo.reshape %v1238 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1240 = stablehlo.convolution(%v1239, %u17pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xf32>, tensor<256x512x1x1xf32>) -> tensor<64x256x7x7xf32>
    %v1241 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1242 = stablehlo.add %v1240, %v1241 : tensor<64x256x7x7xf32>
    %v1243 = stablehlo.reshape %v1242 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1244 = stablehlo.reshape %v1243 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1245 = stablehlo.broadcast_in_dim %u17pnmu, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1246 = stablehlo.subtract %v1244, %v1245 : tensor<64x256x7x7xf32>
    %v1247 = stablehlo.broadcast_in_dim %u17pnvar, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1248 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v1249 = stablehlo.add %v1247, %v1248 : tensor<64x256x7x7xf32>
    %v1250 = stablehlo.rsqrt %v1249 : tensor<64x256x7x7xf32>
    %v1251 = stablehlo.multiply %v1246, %v1250 : tensor<64x256x7x7xf32>
    %v1252 = stablehlo.broadcast_in_dim %u17pg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1253 = stablehlo.broadcast_in_dim %u17pbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1254 = stablehlo.multiply %v1251, %v1252 : tensor<64x256x7x7xf32>
    %v1255 = stablehlo.add %v1254, %v1253 : tensor<64x256x7x7xf32>
    %v1256 = stablehlo.reshape %v1255 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1257 = stablehlo.add %v1256, %v1178 : tensor<64x12544xf32>
    %v1258 = stablehlo.reshape %v1257 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1259 = stablehlo.convolution(%v1258, %u18qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<64x256x7x7xf32>, tensor<256x1x5x5xf32>) -> tensor<64x256x7x7xf32>
    %v1260 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1261 = stablehlo.add %v1259, %v1260 : tensor<64x256x7x7xf32>
    %v1262 = stablehlo.reshape %v1261 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1263 = stablehlo.reshape %v1262 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1264 = stablehlo.broadcast_in_dim %u18qnmu, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1265 = stablehlo.subtract %v1263, %v1264 : tensor<64x256x7x7xf32>
    %v1266 = stablehlo.broadcast_in_dim %u18qnvar, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1267 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v1268 = stablehlo.add %v1266, %v1267 : tensor<64x256x7x7xf32>
    %v1269 = stablehlo.rsqrt %v1268 : tensor<64x256x7x7xf32>
    %v1270 = stablehlo.multiply %v1265, %v1269 : tensor<64x256x7x7xf32>
    %v1271 = stablehlo.broadcast_in_dim %u18qg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1272 = stablehlo.broadcast_in_dim %u18qbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1273 = stablehlo.multiply %v1270, %v1271 : tensor<64x256x7x7xf32>
    %v1274 = stablehlo.add %v1273, %v1272 : tensor<64x256x7x7xf32>
    %v1275 = stablehlo.reshape %v1274 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1276 = stablehlo.constant dense<0.0> : tensor<64x12544xf32>
    %v1277 = stablehlo.maximum %v1275, %v1276 : tensor<64x12544xf32>
    %v1278 = stablehlo.reshape %v1277 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1279 = stablehlo.convolution(%v1278, %u18eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<64x1024x7x7xf32>
    %v1280 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1281 = stablehlo.add %v1279, %v1280 : tensor<64x1024x7x7xf32>
    %v1282 = stablehlo.reshape %v1281 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1283 = stablehlo.reshape %v1282 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1284 = stablehlo.broadcast_in_dim %u18enmu, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1285 = stablehlo.subtract %v1283, %v1284 : tensor<64x1024x7x7xf32>
    %v1286 = stablehlo.broadcast_in_dim %u18envar, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1287 = stablehlo.constant dense<1.0e-5> : tensor<64x1024x7x7xf32>
    %v1288 = stablehlo.add %v1286, %v1287 : tensor<64x1024x7x7xf32>
    %v1289 = stablehlo.rsqrt %v1288 : tensor<64x1024x7x7xf32>
    %v1290 = stablehlo.multiply %v1285, %v1289 : tensor<64x1024x7x7xf32>
    %v1291 = stablehlo.broadcast_in_dim %u18eg, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1292 = stablehlo.broadcast_in_dim %u18ebt, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1293 = stablehlo.multiply %v1290, %v1291 : tensor<64x1024x7x7xf32>
    %v1294 = stablehlo.add %v1293, %v1292 : tensor<64x1024x7x7xf32>
    %v1295 = stablehlo.reshape %v1294 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1296 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1297 = stablehlo.maximum %v1295, %v1296 : tensor<64x50176xf32>
    %v1298 = stablehlo.reshape %v1297 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1299 = stablehlo.convolution(%v1298, %u18dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<64x1024x7x7xf32>, tensor<1024x1x5x5xf32>) -> tensor<64x1024x7x7xf32>
    %v1300 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1301 = stablehlo.add %v1299, %v1300 : tensor<64x1024x7x7xf32>
    %v1302 = stablehlo.reshape %v1301 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1303 = stablehlo.reshape %v1302 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1304 = stablehlo.broadcast_in_dim %u18dnmu, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1305 = stablehlo.subtract %v1303, %v1304 : tensor<64x1024x7x7xf32>
    %v1306 = stablehlo.broadcast_in_dim %u18dnvar, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1307 = stablehlo.constant dense<1.0e-5> : tensor<64x1024x7x7xf32>
    %v1308 = stablehlo.add %v1306, %v1307 : tensor<64x1024x7x7xf32>
    %v1309 = stablehlo.rsqrt %v1308 : tensor<64x1024x7x7xf32>
    %v1310 = stablehlo.multiply %v1305, %v1309 : tensor<64x1024x7x7xf32>
    %v1311 = stablehlo.broadcast_in_dim %u18dg, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1312 = stablehlo.broadcast_in_dim %u18dbt, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1313 = stablehlo.multiply %v1310, %v1311 : tensor<64x1024x7x7xf32>
    %v1314 = stablehlo.add %v1313, %v1312 : tensor<64x1024x7x7xf32>
    %v1315 = stablehlo.reshape %v1314 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1316 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1317 = stablehlo.maximum %v1315, %v1316 : tensor<64x50176xf32>
    %v1318 = stablehlo.reshape %v1317 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1319 = stablehlo.convolution(%v1318, %u18pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<64x256x7x7xf32>
    %v1320 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1321 = stablehlo.add %v1319, %v1320 : tensor<64x256x7x7xf32>
    %v1322 = stablehlo.reshape %v1321 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1323 = stablehlo.reshape %v1322 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1324 = stablehlo.broadcast_in_dim %u18pnmu, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1325 = stablehlo.subtract %v1323, %v1324 : tensor<64x256x7x7xf32>
    %v1326 = stablehlo.broadcast_in_dim %u18pnvar, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1327 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v1328 = stablehlo.add %v1326, %v1327 : tensor<64x256x7x7xf32>
    %v1329 = stablehlo.rsqrt %v1328 : tensor<64x256x7x7xf32>
    %v1330 = stablehlo.multiply %v1325, %v1329 : tensor<64x256x7x7xf32>
    %v1331 = stablehlo.broadcast_in_dim %u18pg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1332 = stablehlo.broadcast_in_dim %u18pbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1333 = stablehlo.multiply %v1330, %v1331 : tensor<64x256x7x7xf32>
    %v1334 = stablehlo.add %v1333, %v1332 : tensor<64x256x7x7xf32>
    %v1335 = stablehlo.reshape %v1334 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1336 = stablehlo.add %v1335, %v1257 : tensor<64x12544xf32>
    %v1337 = stablehlo.reshape %v1336 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1338 = stablehlo.convolution(%v1337, %u19eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<64x1024x7x7xf32>
    %v1339 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1340 = stablehlo.add %v1338, %v1339 : tensor<64x1024x7x7xf32>
    %v1341 = stablehlo.reshape %v1340 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1342 = stablehlo.reshape %v1341 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1343 = stablehlo.broadcast_in_dim %u19enmu, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1344 = stablehlo.subtract %v1342, %v1343 : tensor<64x1024x7x7xf32>
    %v1345 = stablehlo.broadcast_in_dim %u19envar, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1346 = stablehlo.constant dense<1.0e-5> : tensor<64x1024x7x7xf32>
    %v1347 = stablehlo.add %v1345, %v1346 : tensor<64x1024x7x7xf32>
    %v1348 = stablehlo.rsqrt %v1347 : tensor<64x1024x7x7xf32>
    %v1349 = stablehlo.multiply %v1344, %v1348 : tensor<64x1024x7x7xf32>
    %v1350 = stablehlo.broadcast_in_dim %u19eg, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1351 = stablehlo.broadcast_in_dim %u19ebt, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1352 = stablehlo.multiply %v1349, %v1350 : tensor<64x1024x7x7xf32>
    %v1353 = stablehlo.add %v1352, %v1351 : tensor<64x1024x7x7xf32>
    %v1354 = stablehlo.reshape %v1353 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1355 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1356 = stablehlo.maximum %v1354, %v1355 : tensor<64x50176xf32>
    %v1357 = stablehlo.reshape %v1356 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1358 = stablehlo.convolution(%v1357, %u19pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<64x256x7x7xf32>
    %v1359 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1360 = stablehlo.add %v1358, %v1359 : tensor<64x256x7x7xf32>
    %v1361 = stablehlo.reshape %v1360 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1362 = stablehlo.reshape %v1361 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1363 = stablehlo.broadcast_in_dim %u19pnmu, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1364 = stablehlo.subtract %v1362, %v1363 : tensor<64x256x7x7xf32>
    %v1365 = stablehlo.broadcast_in_dim %u19pnvar, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1366 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v1367 = stablehlo.add %v1365, %v1366 : tensor<64x256x7x7xf32>
    %v1368 = stablehlo.rsqrt %v1367 : tensor<64x256x7x7xf32>
    %v1369 = stablehlo.multiply %v1364, %v1368 : tensor<64x256x7x7xf32>
    %v1370 = stablehlo.broadcast_in_dim %u19pg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1371 = stablehlo.broadcast_in_dim %u19pbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1372 = stablehlo.multiply %v1369, %v1370 : tensor<64x256x7x7xf32>
    %v1373 = stablehlo.add %v1372, %v1371 : tensor<64x256x7x7xf32>
    %v1374 = stablehlo.reshape %v1373 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1375 = stablehlo.add %v1374, %v1336 : tensor<64x12544xf32>
    %v1376 = stablehlo.reshape %v1375 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1377 = stablehlo.convolution(%v1376, %u20eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<64x1024x7x7xf32>
    %v1378 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1379 = stablehlo.add %v1377, %v1378 : tensor<64x1024x7x7xf32>
    %v1380 = stablehlo.reshape %v1379 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1381 = stablehlo.reshape %v1380 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1382 = stablehlo.broadcast_in_dim %u20enmu, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1383 = stablehlo.subtract %v1381, %v1382 : tensor<64x1024x7x7xf32>
    %v1384 = stablehlo.broadcast_in_dim %u20envar, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1385 = stablehlo.constant dense<1.0e-5> : tensor<64x1024x7x7xf32>
    %v1386 = stablehlo.add %v1384, %v1385 : tensor<64x1024x7x7xf32>
    %v1387 = stablehlo.rsqrt %v1386 : tensor<64x1024x7x7xf32>
    %v1388 = stablehlo.multiply %v1383, %v1387 : tensor<64x1024x7x7xf32>
    %v1389 = stablehlo.broadcast_in_dim %u20eg, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1390 = stablehlo.broadcast_in_dim %u20ebt, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1391 = stablehlo.multiply %v1388, %v1389 : tensor<64x1024x7x7xf32>
    %v1392 = stablehlo.add %v1391, %v1390 : tensor<64x1024x7x7xf32>
    %v1393 = stablehlo.reshape %v1392 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1394 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1395 = stablehlo.maximum %v1393, %v1394 : tensor<64x50176xf32>
    %v1396 = stablehlo.reshape %v1395 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1397 = stablehlo.convolution(%v1396, %u20pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<64x256x7x7xf32>
    %v1398 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1399 = stablehlo.add %v1397, %v1398 : tensor<64x256x7x7xf32>
    %v1400 = stablehlo.reshape %v1399 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1401 = stablehlo.reshape %v1400 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1402 = stablehlo.broadcast_in_dim %u20pnmu, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1403 = stablehlo.subtract %v1401, %v1402 : tensor<64x256x7x7xf32>
    %v1404 = stablehlo.broadcast_in_dim %u20pnvar, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1405 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v1406 = stablehlo.add %v1404, %v1405 : tensor<64x256x7x7xf32>
    %v1407 = stablehlo.rsqrt %v1406 : tensor<64x256x7x7xf32>
    %v1408 = stablehlo.multiply %v1403, %v1407 : tensor<64x256x7x7xf32>
    %v1409 = stablehlo.broadcast_in_dim %u20pg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1410 = stablehlo.broadcast_in_dim %u20pbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1411 = stablehlo.multiply %v1408, %v1409 : tensor<64x256x7x7xf32>
    %v1412 = stablehlo.add %v1411, %v1410 : tensor<64x256x7x7xf32>
    %v1413 = stablehlo.reshape %v1412 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1414 = stablehlo.add %v1413, %v1375 : tensor<64x12544xf32>
    %v1415 = stablehlo.reshape %v1414 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1416 = stablehlo.convolution(%v1415, %u21qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<64x256x7x7xf32>, tensor<256x1x5x5xf32>) -> tensor<64x256x7x7xf32>
    %v1417 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1418 = stablehlo.add %v1416, %v1417 : tensor<64x256x7x7xf32>
    %v1419 = stablehlo.reshape %v1418 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1420 = stablehlo.reshape %v1419 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1421 = stablehlo.broadcast_in_dim %u21qnmu, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1422 = stablehlo.subtract %v1420, %v1421 : tensor<64x256x7x7xf32>
    %v1423 = stablehlo.broadcast_in_dim %u21qnvar, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1424 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v1425 = stablehlo.add %v1423, %v1424 : tensor<64x256x7x7xf32>
    %v1426 = stablehlo.rsqrt %v1425 : tensor<64x256x7x7xf32>
    %v1427 = stablehlo.multiply %v1422, %v1426 : tensor<64x256x7x7xf32>
    %v1428 = stablehlo.broadcast_in_dim %u21qg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1429 = stablehlo.broadcast_in_dim %u21qbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1430 = stablehlo.multiply %v1427, %v1428 : tensor<64x256x7x7xf32>
    %v1431 = stablehlo.add %v1430, %v1429 : tensor<64x256x7x7xf32>
    %v1432 = stablehlo.reshape %v1431 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1433 = stablehlo.constant dense<0.0> : tensor<64x12544xf32>
    %v1434 = stablehlo.maximum %v1432, %v1433 : tensor<64x12544xf32>
    %v1435 = stablehlo.reshape %v1434 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1436 = stablehlo.convolution(%v1435, %u21eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x7x7xf32>, tensor<512x256x1x1xf32>) -> tensor<64x512x7x7xf32>
    %v1437 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1438 = stablehlo.add %v1436, %v1437 : tensor<64x512x7x7xf32>
    %v1439 = stablehlo.reshape %v1438 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1440 = stablehlo.reshape %v1439 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1441 = stablehlo.broadcast_in_dim %u21enmu, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1442 = stablehlo.subtract %v1440, %v1441 : tensor<64x512x7x7xf32>
    %v1443 = stablehlo.broadcast_in_dim %u21envar, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1444 = stablehlo.constant dense<1.0e-5> : tensor<64x512x7x7xf32>
    %v1445 = stablehlo.add %v1443, %v1444 : tensor<64x512x7x7xf32>
    %v1446 = stablehlo.rsqrt %v1445 : tensor<64x512x7x7xf32>
    %v1447 = stablehlo.multiply %v1442, %v1446 : tensor<64x512x7x7xf32>
    %v1448 = stablehlo.broadcast_in_dim %u21eg, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1449 = stablehlo.broadcast_in_dim %u21ebt, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1450 = stablehlo.multiply %v1447, %v1448 : tensor<64x512x7x7xf32>
    %v1451 = stablehlo.add %v1450, %v1449 : tensor<64x512x7x7xf32>
    %v1452 = stablehlo.reshape %v1451 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1453 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v1454 = stablehlo.maximum %v1452, %v1453 : tensor<64x25088xf32>
    %v1455 = stablehlo.reshape %v1454 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1456 = stablehlo.convolution(%v1455, %u21pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xf32>, tensor<256x512x1x1xf32>) -> tensor<64x256x7x7xf32>
    %v1457 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1458 = stablehlo.add %v1456, %v1457 : tensor<64x256x7x7xf32>
    %v1459 = stablehlo.reshape %v1458 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1460 = stablehlo.reshape %v1459 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1461 = stablehlo.broadcast_in_dim %u21pnmu, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1462 = stablehlo.subtract %v1460, %v1461 : tensor<64x256x7x7xf32>
    %v1463 = stablehlo.broadcast_in_dim %u21pnvar, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1464 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v1465 = stablehlo.add %v1463, %v1464 : tensor<64x256x7x7xf32>
    %v1466 = stablehlo.rsqrt %v1465 : tensor<64x256x7x7xf32>
    %v1467 = stablehlo.multiply %v1462, %v1466 : tensor<64x256x7x7xf32>
    %v1468 = stablehlo.broadcast_in_dim %u21pg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1469 = stablehlo.broadcast_in_dim %u21pbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1470 = stablehlo.multiply %v1467, %v1468 : tensor<64x256x7x7xf32>
    %v1471 = stablehlo.add %v1470, %v1469 : tensor<64x256x7x7xf32>
    %v1472 = stablehlo.reshape %v1471 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1473 = stablehlo.add %v1472, %v1414 : tensor<64x12544xf32>
    %v1474 = stablehlo.reshape %v1473 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1475 = stablehlo.convolution(%v1474, %h1W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x7x7xf32>, tensor<960x256x1x1xf32>) -> tensor<64x960x7x7xf32>
    %v1476 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1477 = stablehlo.add %v1475, %v1476 : tensor<64x960x7x7xf32>
    %v1478 = stablehlo.reshape %v1477 : (tensor<64x960x7x7xf32>) -> tensor<64x47040xf32>
    %v1479 = stablehlo.reshape %v1478 : (tensor<64x47040xf32>) -> tensor<64x960x7x7xf32>
    %v1480 = stablehlo.broadcast_in_dim %h1nmu, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1481 = stablehlo.subtract %v1479, %v1480 : tensor<64x960x7x7xf32>
    %v1482 = stablehlo.broadcast_in_dim %h1nvar, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1483 = stablehlo.constant dense<1.0e-5> : tensor<64x960x7x7xf32>
    %v1484 = stablehlo.add %v1482, %v1483 : tensor<64x960x7x7xf32>
    %v1485 = stablehlo.rsqrt %v1484 : tensor<64x960x7x7xf32>
    %v1486 = stablehlo.multiply %v1481, %v1485 : tensor<64x960x7x7xf32>
    %v1487 = stablehlo.broadcast_in_dim %h1g, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1488 = stablehlo.broadcast_in_dim %h1bt, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1489 = stablehlo.multiply %v1486, %v1487 : tensor<64x960x7x7xf32>
    %v1490 = stablehlo.add %v1489, %v1488 : tensor<64x960x7x7xf32>
    %v1491 = stablehlo.reshape %v1490 : (tensor<64x960x7x7xf32>) -> tensor<64x47040xf32>
    %v1492 = stablehlo.constant dense<0.0> : tensor<64x47040xf32>
    %v1493 = stablehlo.maximum %v1491, %v1492 : tensor<64x47040xf32>
    %v1494 = stablehlo.reshape %v1493 : (tensor<64x47040xf32>) -> tensor<64x960x7x7xf32>
    %v1495 = stablehlo.convolution(%v1494, %hW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x960x7x7xf32>, tensor<1280x960x1x1xf32>) -> tensor<64x1280x7x7xf32>
    %v1496 = stablehlo.broadcast_in_dim %zb1280, dims = [1] : (tensor<1280xf32>) -> tensor<64x1280x7x7xf32>
    %v1497 = stablehlo.add %v1495, %v1496 : tensor<64x1280x7x7xf32>
    %v1498 = stablehlo.reshape %v1497 : (tensor<64x1280x7x7xf32>) -> tensor<64x62720xf32>
    %v1499 = stablehlo.reshape %v1498 : (tensor<64x62720xf32>) -> tensor<64x1280x7x7xf32>
    %v1500 = stablehlo.broadcast_in_dim %hnmu, dims = [1] : (tensor<1280xf32>) -> tensor<64x1280x7x7xf32>
    %v1501 = stablehlo.subtract %v1499, %v1500 : tensor<64x1280x7x7xf32>
    %v1502 = stablehlo.broadcast_in_dim %hnvar, dims = [1] : (tensor<1280xf32>) -> tensor<64x1280x7x7xf32>
    %v1503 = stablehlo.constant dense<1.0e-5> : tensor<64x1280x7x7xf32>
    %v1504 = stablehlo.add %v1502, %v1503 : tensor<64x1280x7x7xf32>
    %v1505 = stablehlo.rsqrt %v1504 : tensor<64x1280x7x7xf32>
    %v1506 = stablehlo.multiply %v1501, %v1505 : tensor<64x1280x7x7xf32>
    %v1507 = stablehlo.broadcast_in_dim %hg, dims = [1] : (tensor<1280xf32>) -> tensor<64x1280x7x7xf32>
    %v1508 = stablehlo.broadcast_in_dim %hbt, dims = [1] : (tensor<1280xf32>) -> tensor<64x1280x7x7xf32>
    %v1509 = stablehlo.multiply %v1506, %v1507 : tensor<64x1280x7x7xf32>
    %v1510 = stablehlo.add %v1509, %v1508 : tensor<64x1280x7x7xf32>
    %v1511 = stablehlo.reshape %v1510 : (tensor<64x1280x7x7xf32>) -> tensor<64x62720xf32>
    %v1512 = stablehlo.constant dense<0.0> : tensor<64x62720xf32>
    %v1513 = stablehlo.maximum %v1511, %v1512 : tensor<64x62720xf32>
    %v1514 = stablehlo.reshape %v1513 : (tensor<64x62720xf32>) -> tensor<64x1280x7x7xf32>
    %v1515 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1516 = stablehlo.reduce(%v1514 init: %v1515) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x1280x7x7xf32>, tensor<f32>) -> tensor<64x1280xf32>
    %v1517 = stablehlo.constant dense<49.0> : tensor<64x1280xf32>
    %v1518 = stablehlo.divide %v1516, %v1517 : tensor<64x1280xf32>
    %v1519 = stablehlo.dot_general %v1518, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x1280xf32>, tensor<1280x1000xf32>) -> tensor<64x1000xf32>
    %v1520 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<1000xf32>) -> tensor<64x1000xf32>
    %v1521 = stablehlo.add %v1519, %v1520 : tensor<64x1000xf32>
    return %v1521 : tensor<64x1000xf32>
  }
}
