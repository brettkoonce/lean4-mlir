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
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
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
    %v18 = stablehlo.reshape %v17 : (tensor<64x401408xf32>) -> tensor<64x32x112x112xf32>
    %v19 = stablehlo.constant dense<0.0> : tensor<64x32x112x112xf32>
    %v20 = stablehlo.maximum %v18, %v19 : tensor<64x32x112x112xf32>
    %v21 = stablehlo.reshape %v20 : (tensor<64x32x112x112xf32>) -> tensor<64x401408xf32>
    %v22 = stablehlo.reshape %v21 : (tensor<64x401408xf32>) -> tensor<64x32x112x112xf32>
    %v23 = stablehlo.convolution(%v22, %f0cW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x112x112xf32>, tensor<128x32x3x3xf32>) -> tensor<64x128x56x56xf32>
    %v24 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v25 = stablehlo.add %v23, %v24 : tensor<64x128x56x56xf32>
    %v26 = stablehlo.reshape %v25 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v27 = stablehlo.reshape %v26 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v28 = stablehlo.broadcast_in_dim %f0cnmu, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v29 = stablehlo.subtract %v27, %v28 : tensor<64x128x56x56xf32>
    %v30 = stablehlo.broadcast_in_dim %f0cnvar, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v31 = stablehlo.constant dense<1.0e-5> : tensor<64x128x56x56xf32>
    %v32 = stablehlo.add %v30, %v31 : tensor<64x128x56x56xf32>
    %v33 = stablehlo.rsqrt %v32 : tensor<64x128x56x56xf32>
    %v34 = stablehlo.multiply %v29, %v33 : tensor<64x128x56x56xf32>
    %v35 = stablehlo.broadcast_in_dim %f0cg, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v36 = stablehlo.broadcast_in_dim %f0cbt, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v37 = stablehlo.multiply %v34, %v35 : tensor<64x128x56x56xf32>
    %v38 = stablehlo.add %v37, %v36 : tensor<64x128x56x56xf32>
    %v39 = stablehlo.reshape %v38 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v40 = stablehlo.reshape %v39 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v41 = stablehlo.constant dense<0.0> : tensor<64x128x56x56xf32>
    %v42 = stablehlo.maximum %v40, %v41 : tensor<64x128x56x56xf32>
    %v43 = stablehlo.reshape %v42 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v44 = stablehlo.reshape %v43 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v45 = stablehlo.convolution(%v44, %f0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x56x56xf32>, tensor<48x128x1x1xf32>) -> tensor<64x48x56x56xf32>
    %v46 = stablehlo.broadcast_in_dim %zb48, dims = [1] : (tensor<48xf32>) -> tensor<64x48x56x56xf32>
    %v47 = stablehlo.add %v45, %v46 : tensor<64x48x56x56xf32>
    %v48 = stablehlo.reshape %v47 : (tensor<64x48x56x56xf32>) -> tensor<64x150528xf32>
    %v49 = stablehlo.reshape %v48 : (tensor<64x150528xf32>) -> tensor<64x48x56x56xf32>
    %v50 = stablehlo.broadcast_in_dim %f0pnmu, dims = [1] : (tensor<48xf32>) -> tensor<64x48x56x56xf32>
    %v51 = stablehlo.subtract %v49, %v50 : tensor<64x48x56x56xf32>
    %v52 = stablehlo.broadcast_in_dim %f0pnvar, dims = [1] : (tensor<48xf32>) -> tensor<64x48x56x56xf32>
    %v53 = stablehlo.constant dense<1.0e-5> : tensor<64x48x56x56xf32>
    %v54 = stablehlo.add %v52, %v53 : tensor<64x48x56x56xf32>
    %v55 = stablehlo.rsqrt %v54 : tensor<64x48x56x56xf32>
    %v56 = stablehlo.multiply %v51, %v55 : tensor<64x48x56x56xf32>
    %v57 = stablehlo.broadcast_in_dim %f0pg, dims = [1] : (tensor<48xf32>) -> tensor<64x48x56x56xf32>
    %v58 = stablehlo.broadcast_in_dim %f0pbt, dims = [1] : (tensor<48xf32>) -> tensor<64x48x56x56xf32>
    %v59 = stablehlo.multiply %v56, %v57 : tensor<64x48x56x56xf32>
    %v60 = stablehlo.add %v59, %v58 : tensor<64x48x56x56xf32>
    %v61 = stablehlo.reshape %v60 : (tensor<64x48x56x56xf32>) -> tensor<64x150528xf32>
    %v62 = stablehlo.reshape %v61 : (tensor<64x150528xf32>) -> tensor<64x48x56x56xf32>
    %v63 = stablehlo.convolution(%v62, %u1qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 48 : i64} : (tensor<64x48x56x56xf32>, tensor<48x1x3x3xf32>) -> tensor<64x48x56x56xf32>
    %v64 = stablehlo.broadcast_in_dim %zb48, dims = [1] : (tensor<48xf32>) -> tensor<64x48x56x56xf32>
    %v65 = stablehlo.add %v63, %v64 : tensor<64x48x56x56xf32>
    %v66 = stablehlo.reshape %v65 : (tensor<64x48x56x56xf32>) -> tensor<64x150528xf32>
    %v67 = stablehlo.reshape %v66 : (tensor<64x150528xf32>) -> tensor<64x48x56x56xf32>
    %v68 = stablehlo.broadcast_in_dim %u1qnmu, dims = [1] : (tensor<48xf32>) -> tensor<64x48x56x56xf32>
    %v69 = stablehlo.subtract %v67, %v68 : tensor<64x48x56x56xf32>
    %v70 = stablehlo.broadcast_in_dim %u1qnvar, dims = [1] : (tensor<48xf32>) -> tensor<64x48x56x56xf32>
    %v71 = stablehlo.constant dense<1.0e-5> : tensor<64x48x56x56xf32>
    %v72 = stablehlo.add %v70, %v71 : tensor<64x48x56x56xf32>
    %v73 = stablehlo.rsqrt %v72 : tensor<64x48x56x56xf32>
    %v74 = stablehlo.multiply %v69, %v73 : tensor<64x48x56x56xf32>
    %v75 = stablehlo.broadcast_in_dim %u1qg, dims = [1] : (tensor<48xf32>) -> tensor<64x48x56x56xf32>
    %v76 = stablehlo.broadcast_in_dim %u1qbt, dims = [1] : (tensor<48xf32>) -> tensor<64x48x56x56xf32>
    %v77 = stablehlo.multiply %v74, %v75 : tensor<64x48x56x56xf32>
    %v78 = stablehlo.add %v77, %v76 : tensor<64x48x56x56xf32>
    %v79 = stablehlo.reshape %v78 : (tensor<64x48x56x56xf32>) -> tensor<64x150528xf32>
    %v80 = stablehlo.reshape %v79 : (tensor<64x150528xf32>) -> tensor<64x48x56x56xf32>
    %v81 = stablehlo.convolution(%v80, %u1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x48x56x56xf32>, tensor<192x48x1x1xf32>) -> tensor<64x192x56x56xf32>
    %v82 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<64x192x56x56xf32>
    %v83 = stablehlo.add %v81, %v82 : tensor<64x192x56x56xf32>
    %v84 = stablehlo.reshape %v83 : (tensor<64x192x56x56xf32>) -> tensor<64x602112xf32>
    %v85 = stablehlo.reshape %v84 : (tensor<64x602112xf32>) -> tensor<64x192x56x56xf32>
    %v86 = stablehlo.broadcast_in_dim %u1enmu, dims = [1] : (tensor<192xf32>) -> tensor<64x192x56x56xf32>
    %v87 = stablehlo.subtract %v85, %v86 : tensor<64x192x56x56xf32>
    %v88 = stablehlo.broadcast_in_dim %u1envar, dims = [1] : (tensor<192xf32>) -> tensor<64x192x56x56xf32>
    %v89 = stablehlo.constant dense<1.0e-5> : tensor<64x192x56x56xf32>
    %v90 = stablehlo.add %v88, %v89 : tensor<64x192x56x56xf32>
    %v91 = stablehlo.rsqrt %v90 : tensor<64x192x56x56xf32>
    %v92 = stablehlo.multiply %v87, %v91 : tensor<64x192x56x56xf32>
    %v93 = stablehlo.broadcast_in_dim %u1eg, dims = [1] : (tensor<192xf32>) -> tensor<64x192x56x56xf32>
    %v94 = stablehlo.broadcast_in_dim %u1ebt, dims = [1] : (tensor<192xf32>) -> tensor<64x192x56x56xf32>
    %v95 = stablehlo.multiply %v92, %v93 : tensor<64x192x56x56xf32>
    %v96 = stablehlo.add %v95, %v94 : tensor<64x192x56x56xf32>
    %v97 = stablehlo.reshape %v96 : (tensor<64x192x56x56xf32>) -> tensor<64x602112xf32>
    %v98 = stablehlo.reshape %v97 : (tensor<64x602112xf32>) -> tensor<64x192x56x56xf32>
    %v99 = stablehlo.constant dense<0.0> : tensor<64x192x56x56xf32>
    %v100 = stablehlo.maximum %v98, %v99 : tensor<64x192x56x56xf32>
    %v101 = stablehlo.reshape %v100 : (tensor<64x192x56x56xf32>) -> tensor<64x602112xf32>
    %v102 = stablehlo.reshape %v101 : (tensor<64x602112xf32>) -> tensor<64x192x56x56xf32>
    %v103 = stablehlo.convolution(%v102, %u1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<64x192x56x56xf32>, tensor<192x1x5x5xf32>) -> tensor<64x192x28x28xf32>
    %v104 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v105 = stablehlo.add %v103, %v104 : tensor<64x192x28x28xf32>
    %v106 = stablehlo.reshape %v105 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v107 = stablehlo.reshape %v106 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v108 = stablehlo.broadcast_in_dim %u1dnmu, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v109 = stablehlo.subtract %v107, %v108 : tensor<64x192x28x28xf32>
    %v110 = stablehlo.broadcast_in_dim %u1dnvar, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v111 = stablehlo.constant dense<1.0e-5> : tensor<64x192x28x28xf32>
    %v112 = stablehlo.add %v110, %v111 : tensor<64x192x28x28xf32>
    %v113 = stablehlo.rsqrt %v112 : tensor<64x192x28x28xf32>
    %v114 = stablehlo.multiply %v109, %v113 : tensor<64x192x28x28xf32>
    %v115 = stablehlo.broadcast_in_dim %u1dg, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v116 = stablehlo.broadcast_in_dim %u1dbt, dims = [1] : (tensor<192xf32>) -> tensor<64x192x28x28xf32>
    %v117 = stablehlo.multiply %v114, %v115 : tensor<64x192x28x28xf32>
    %v118 = stablehlo.add %v117, %v116 : tensor<64x192x28x28xf32>
    %v119 = stablehlo.reshape %v118 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v120 = stablehlo.reshape %v119 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v121 = stablehlo.constant dense<0.0> : tensor<64x192x28x28xf32>
    %v122 = stablehlo.maximum %v120, %v121 : tensor<64x192x28x28xf32>
    %v123 = stablehlo.reshape %v122 : (tensor<64x192x28x28xf32>) -> tensor<64x150528xf32>
    %v124 = stablehlo.reshape %v123 : (tensor<64x150528xf32>) -> tensor<64x192x28x28xf32>
    %v125 = stablehlo.convolution(%v124, %u1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x192x28x28xf32>, tensor<80x192x1x1xf32>) -> tensor<64x80x28x28xf32>
    %v126 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v127 = stablehlo.add %v125, %v126 : tensor<64x80x28x28xf32>
    %v128 = stablehlo.reshape %v127 : (tensor<64x80x28x28xf32>) -> tensor<64x62720xf32>
    %v129 = stablehlo.reshape %v128 : (tensor<64x62720xf32>) -> tensor<64x80x28x28xf32>
    %v130 = stablehlo.broadcast_in_dim %u1pnmu, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v131 = stablehlo.subtract %v129, %v130 : tensor<64x80x28x28xf32>
    %v132 = stablehlo.broadcast_in_dim %u1pnvar, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v133 = stablehlo.constant dense<1.0e-5> : tensor<64x80x28x28xf32>
    %v134 = stablehlo.add %v132, %v133 : tensor<64x80x28x28xf32>
    %v135 = stablehlo.rsqrt %v134 : tensor<64x80x28x28xf32>
    %v136 = stablehlo.multiply %v131, %v135 : tensor<64x80x28x28xf32>
    %v137 = stablehlo.broadcast_in_dim %u1pg, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v138 = stablehlo.broadcast_in_dim %u1pbt, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v139 = stablehlo.multiply %v136, %v137 : tensor<64x80x28x28xf32>
    %v140 = stablehlo.add %v139, %v138 : tensor<64x80x28x28xf32>
    %v141 = stablehlo.reshape %v140 : (tensor<64x80x28x28xf32>) -> tensor<64x62720xf32>
    %v142 = stablehlo.reshape %v141 : (tensor<64x62720xf32>) -> tensor<64x80x28x28xf32>
    %v143 = stablehlo.convolution(%v142, %u2qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 80 : i64} : (tensor<64x80x28x28xf32>, tensor<80x1x3x3xf32>) -> tensor<64x80x28x28xf32>
    %v144 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v145 = stablehlo.add %v143, %v144 : tensor<64x80x28x28xf32>
    %v146 = stablehlo.reshape %v145 : (tensor<64x80x28x28xf32>) -> tensor<64x62720xf32>
    %v147 = stablehlo.reshape %v146 : (tensor<64x62720xf32>) -> tensor<64x80x28x28xf32>
    %v148 = stablehlo.broadcast_in_dim %u2qnmu, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v149 = stablehlo.subtract %v147, %v148 : tensor<64x80x28x28xf32>
    %v150 = stablehlo.broadcast_in_dim %u2qnvar, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v151 = stablehlo.constant dense<1.0e-5> : tensor<64x80x28x28xf32>
    %v152 = stablehlo.add %v150, %v151 : tensor<64x80x28x28xf32>
    %v153 = stablehlo.rsqrt %v152 : tensor<64x80x28x28xf32>
    %v154 = stablehlo.multiply %v149, %v153 : tensor<64x80x28x28xf32>
    %v155 = stablehlo.broadcast_in_dim %u2qg, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v156 = stablehlo.broadcast_in_dim %u2qbt, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v157 = stablehlo.multiply %v154, %v155 : tensor<64x80x28x28xf32>
    %v158 = stablehlo.add %v157, %v156 : tensor<64x80x28x28xf32>
    %v159 = stablehlo.reshape %v158 : (tensor<64x80x28x28xf32>) -> tensor<64x62720xf32>
    %v160 = stablehlo.reshape %v159 : (tensor<64x62720xf32>) -> tensor<64x80x28x28xf32>
    %v161 = stablehlo.convolution(%v160, %u2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x80x28x28xf32>, tensor<160x80x1x1xf32>) -> tensor<64x160x28x28xf32>
    %v162 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x28x28xf32>
    %v163 = stablehlo.add %v161, %v162 : tensor<64x160x28x28xf32>
    %v164 = stablehlo.reshape %v163 : (tensor<64x160x28x28xf32>) -> tensor<64x125440xf32>
    %v165 = stablehlo.reshape %v164 : (tensor<64x125440xf32>) -> tensor<64x160x28x28xf32>
    %v166 = stablehlo.broadcast_in_dim %u2enmu, dims = [1] : (tensor<160xf32>) -> tensor<64x160x28x28xf32>
    %v167 = stablehlo.subtract %v165, %v166 : tensor<64x160x28x28xf32>
    %v168 = stablehlo.broadcast_in_dim %u2envar, dims = [1] : (tensor<160xf32>) -> tensor<64x160x28x28xf32>
    %v169 = stablehlo.constant dense<1.0e-5> : tensor<64x160x28x28xf32>
    %v170 = stablehlo.add %v168, %v169 : tensor<64x160x28x28xf32>
    %v171 = stablehlo.rsqrt %v170 : tensor<64x160x28x28xf32>
    %v172 = stablehlo.multiply %v167, %v171 : tensor<64x160x28x28xf32>
    %v173 = stablehlo.broadcast_in_dim %u2eg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x28x28xf32>
    %v174 = stablehlo.broadcast_in_dim %u2ebt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x28x28xf32>
    %v175 = stablehlo.multiply %v172, %v173 : tensor<64x160x28x28xf32>
    %v176 = stablehlo.add %v175, %v174 : tensor<64x160x28x28xf32>
    %v177 = stablehlo.reshape %v176 : (tensor<64x160x28x28xf32>) -> tensor<64x125440xf32>
    %v178 = stablehlo.reshape %v177 : (tensor<64x125440xf32>) -> tensor<64x160x28x28xf32>
    %v179 = stablehlo.constant dense<0.0> : tensor<64x160x28x28xf32>
    %v180 = stablehlo.maximum %v178, %v179 : tensor<64x160x28x28xf32>
    %v181 = stablehlo.reshape %v180 : (tensor<64x160x28x28xf32>) -> tensor<64x125440xf32>
    %v182 = stablehlo.reshape %v181 : (tensor<64x125440xf32>) -> tensor<64x160x28x28xf32>
    %v183 = stablehlo.convolution(%v182, %u2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<64x160x28x28xf32>, tensor<160x1x3x3xf32>) -> tensor<64x160x28x28xf32>
    %v184 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x28x28xf32>
    %v185 = stablehlo.add %v183, %v184 : tensor<64x160x28x28xf32>
    %v186 = stablehlo.reshape %v185 : (tensor<64x160x28x28xf32>) -> tensor<64x125440xf32>
    %v187 = stablehlo.reshape %v186 : (tensor<64x125440xf32>) -> tensor<64x160x28x28xf32>
    %v188 = stablehlo.broadcast_in_dim %u2dnmu, dims = [1] : (tensor<160xf32>) -> tensor<64x160x28x28xf32>
    %v189 = stablehlo.subtract %v187, %v188 : tensor<64x160x28x28xf32>
    %v190 = stablehlo.broadcast_in_dim %u2dnvar, dims = [1] : (tensor<160xf32>) -> tensor<64x160x28x28xf32>
    %v191 = stablehlo.constant dense<1.0e-5> : tensor<64x160x28x28xf32>
    %v192 = stablehlo.add %v190, %v191 : tensor<64x160x28x28xf32>
    %v193 = stablehlo.rsqrt %v192 : tensor<64x160x28x28xf32>
    %v194 = stablehlo.multiply %v189, %v193 : tensor<64x160x28x28xf32>
    %v195 = stablehlo.broadcast_in_dim %u2dg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x28x28xf32>
    %v196 = stablehlo.broadcast_in_dim %u2dbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x28x28xf32>
    %v197 = stablehlo.multiply %v194, %v195 : tensor<64x160x28x28xf32>
    %v198 = stablehlo.add %v197, %v196 : tensor<64x160x28x28xf32>
    %v199 = stablehlo.reshape %v198 : (tensor<64x160x28x28xf32>) -> tensor<64x125440xf32>
    %v200 = stablehlo.reshape %v199 : (tensor<64x125440xf32>) -> tensor<64x160x28x28xf32>
    %v201 = stablehlo.constant dense<0.0> : tensor<64x160x28x28xf32>
    %v202 = stablehlo.maximum %v200, %v201 : tensor<64x160x28x28xf32>
    %v203 = stablehlo.reshape %v202 : (tensor<64x160x28x28xf32>) -> tensor<64x125440xf32>
    %v204 = stablehlo.reshape %v203 : (tensor<64x125440xf32>) -> tensor<64x160x28x28xf32>
    %v205 = stablehlo.convolution(%v204, %u2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x160x28x28xf32>, tensor<80x160x1x1xf32>) -> tensor<64x80x28x28xf32>
    %v206 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v207 = stablehlo.add %v205, %v206 : tensor<64x80x28x28xf32>
    %v208 = stablehlo.reshape %v207 : (tensor<64x80x28x28xf32>) -> tensor<64x62720xf32>
    %v209 = stablehlo.reshape %v208 : (tensor<64x62720xf32>) -> tensor<64x80x28x28xf32>
    %v210 = stablehlo.broadcast_in_dim %u2pnmu, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v211 = stablehlo.subtract %v209, %v210 : tensor<64x80x28x28xf32>
    %v212 = stablehlo.broadcast_in_dim %u2pnvar, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v213 = stablehlo.constant dense<1.0e-5> : tensor<64x80x28x28xf32>
    %v214 = stablehlo.add %v212, %v213 : tensor<64x80x28x28xf32>
    %v215 = stablehlo.rsqrt %v214 : tensor<64x80x28x28xf32>
    %v216 = stablehlo.multiply %v211, %v215 : tensor<64x80x28x28xf32>
    %v217 = stablehlo.broadcast_in_dim %u2pg, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v218 = stablehlo.broadcast_in_dim %u2pbt, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v219 = stablehlo.multiply %v216, %v217 : tensor<64x80x28x28xf32>
    %v220 = stablehlo.add %v219, %v218 : tensor<64x80x28x28xf32>
    %v221 = stablehlo.reshape %v220 : (tensor<64x80x28x28xf32>) -> tensor<64x62720xf32>
    %v222 = stablehlo.reshape %v221 : (tensor<64x62720xf32>) -> tensor<64x80x28x28xf32>
    %v223 = stablehlo.reshape %v141 : (tensor<64x62720xf32>) -> tensor<64x80x28x28xf32>
    %v224 = stablehlo.add %v222, %v223 : tensor<64x80x28x28xf32>
    %v225 = stablehlo.reshape %v224 : (tensor<64x80x28x28xf32>) -> tensor<64x62720xf32>
    %v226 = stablehlo.reshape %v225 : (tensor<64x62720xf32>) -> tensor<64x80x28x28xf32>
    %v227 = stablehlo.convolution(%v226, %u3qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 80 : i64} : (tensor<64x80x28x28xf32>, tensor<80x1x3x3xf32>) -> tensor<64x80x28x28xf32>
    %v228 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v229 = stablehlo.add %v227, %v228 : tensor<64x80x28x28xf32>
    %v230 = stablehlo.reshape %v229 : (tensor<64x80x28x28xf32>) -> tensor<64x62720xf32>
    %v231 = stablehlo.reshape %v230 : (tensor<64x62720xf32>) -> tensor<64x80x28x28xf32>
    %v232 = stablehlo.broadcast_in_dim %u3qnmu, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v233 = stablehlo.subtract %v231, %v232 : tensor<64x80x28x28xf32>
    %v234 = stablehlo.broadcast_in_dim %u3qnvar, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v235 = stablehlo.constant dense<1.0e-5> : tensor<64x80x28x28xf32>
    %v236 = stablehlo.add %v234, %v235 : tensor<64x80x28x28xf32>
    %v237 = stablehlo.rsqrt %v236 : tensor<64x80x28x28xf32>
    %v238 = stablehlo.multiply %v233, %v237 : tensor<64x80x28x28xf32>
    %v239 = stablehlo.broadcast_in_dim %u3qg, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v240 = stablehlo.broadcast_in_dim %u3qbt, dims = [1] : (tensor<80xf32>) -> tensor<64x80x28x28xf32>
    %v241 = stablehlo.multiply %v238, %v239 : tensor<64x80x28x28xf32>
    %v242 = stablehlo.add %v241, %v240 : tensor<64x80x28x28xf32>
    %v243 = stablehlo.reshape %v242 : (tensor<64x80x28x28xf32>) -> tensor<64x62720xf32>
    %v244 = stablehlo.reshape %v243 : (tensor<64x62720xf32>) -> tensor<64x80x28x28xf32>
    %v245 = stablehlo.convolution(%v244, %u3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x80x28x28xf32>, tensor<480x80x1x1xf32>) -> tensor<64x480x28x28xf32>
    %v246 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<64x480x28x28xf32>
    %v247 = stablehlo.add %v245, %v246 : tensor<64x480x28x28xf32>
    %v248 = stablehlo.reshape %v247 : (tensor<64x480x28x28xf32>) -> tensor<64x376320xf32>
    %v249 = stablehlo.reshape %v248 : (tensor<64x376320xf32>) -> tensor<64x480x28x28xf32>
    %v250 = stablehlo.broadcast_in_dim %u3enmu, dims = [1] : (tensor<480xf32>) -> tensor<64x480x28x28xf32>
    %v251 = stablehlo.subtract %v249, %v250 : tensor<64x480x28x28xf32>
    %v252 = stablehlo.broadcast_in_dim %u3envar, dims = [1] : (tensor<480xf32>) -> tensor<64x480x28x28xf32>
    %v253 = stablehlo.constant dense<1.0e-5> : tensor<64x480x28x28xf32>
    %v254 = stablehlo.add %v252, %v253 : tensor<64x480x28x28xf32>
    %v255 = stablehlo.rsqrt %v254 : tensor<64x480x28x28xf32>
    %v256 = stablehlo.multiply %v251, %v255 : tensor<64x480x28x28xf32>
    %v257 = stablehlo.broadcast_in_dim %u3eg, dims = [1] : (tensor<480xf32>) -> tensor<64x480x28x28xf32>
    %v258 = stablehlo.broadcast_in_dim %u3ebt, dims = [1] : (tensor<480xf32>) -> tensor<64x480x28x28xf32>
    %v259 = stablehlo.multiply %v256, %v257 : tensor<64x480x28x28xf32>
    %v260 = stablehlo.add %v259, %v258 : tensor<64x480x28x28xf32>
    %v261 = stablehlo.reshape %v260 : (tensor<64x480x28x28xf32>) -> tensor<64x376320xf32>
    %v262 = stablehlo.reshape %v261 : (tensor<64x376320xf32>) -> tensor<64x480x28x28xf32>
    %v263 = stablehlo.constant dense<0.0> : tensor<64x480x28x28xf32>
    %v264 = stablehlo.maximum %v262, %v263 : tensor<64x480x28x28xf32>
    %v265 = stablehlo.reshape %v264 : (tensor<64x480x28x28xf32>) -> tensor<64x376320xf32>
    %v266 = stablehlo.reshape %v265 : (tensor<64x376320xf32>) -> tensor<64x480x28x28xf32>
    %v267 = stablehlo.convolution(%v266, %u3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<64x480x28x28xf32>, tensor<480x1x5x5xf32>) -> tensor<64x480x14x14xf32>
    %v268 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v269 = stablehlo.add %v267, %v268 : tensor<64x480x14x14xf32>
    %v270 = stablehlo.reshape %v269 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v271 = stablehlo.reshape %v270 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v272 = stablehlo.broadcast_in_dim %u3dnmu, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v273 = stablehlo.subtract %v271, %v272 : tensor<64x480x14x14xf32>
    %v274 = stablehlo.broadcast_in_dim %u3dnvar, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v275 = stablehlo.constant dense<1.0e-5> : tensor<64x480x14x14xf32>
    %v276 = stablehlo.add %v274, %v275 : tensor<64x480x14x14xf32>
    %v277 = stablehlo.rsqrt %v276 : tensor<64x480x14x14xf32>
    %v278 = stablehlo.multiply %v273, %v277 : tensor<64x480x14x14xf32>
    %v279 = stablehlo.broadcast_in_dim %u3dg, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v280 = stablehlo.broadcast_in_dim %u3dbt, dims = [1] : (tensor<480xf32>) -> tensor<64x480x14x14xf32>
    %v281 = stablehlo.multiply %v278, %v279 : tensor<64x480x14x14xf32>
    %v282 = stablehlo.add %v281, %v280 : tensor<64x480x14x14xf32>
    %v283 = stablehlo.reshape %v282 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v284 = stablehlo.reshape %v283 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v285 = stablehlo.constant dense<0.0> : tensor<64x480x14x14xf32>
    %v286 = stablehlo.maximum %v284, %v285 : tensor<64x480x14x14xf32>
    %v287 = stablehlo.reshape %v286 : (tensor<64x480x14x14xf32>) -> tensor<64x94080xf32>
    %v288 = stablehlo.reshape %v287 : (tensor<64x94080xf32>) -> tensor<64x480x14x14xf32>
    %v289 = stablehlo.convolution(%v288, %u3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x480x14x14xf32>, tensor<160x480x1x1xf32>) -> tensor<64x160x14x14xf32>
    %v290 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v291 = stablehlo.add %v289, %v290 : tensor<64x160x14x14xf32>
    %v292 = stablehlo.reshape %v291 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v293 = stablehlo.reshape %v292 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v294 = stablehlo.broadcast_in_dim %u3pnmu, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v295 = stablehlo.subtract %v293, %v294 : tensor<64x160x14x14xf32>
    %v296 = stablehlo.broadcast_in_dim %u3pnvar, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v297 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v298 = stablehlo.add %v296, %v297 : tensor<64x160x14x14xf32>
    %v299 = stablehlo.rsqrt %v298 : tensor<64x160x14x14xf32>
    %v300 = stablehlo.multiply %v295, %v299 : tensor<64x160x14x14xf32>
    %v301 = stablehlo.broadcast_in_dim %u3pg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v302 = stablehlo.broadcast_in_dim %u3pbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v303 = stablehlo.multiply %v300, %v301 : tensor<64x160x14x14xf32>
    %v304 = stablehlo.add %v303, %v302 : tensor<64x160x14x14xf32>
    %v305 = stablehlo.reshape %v304 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v306 = stablehlo.reshape %v305 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v307 = stablehlo.convolution(%v306, %u4qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<64x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<64x160x14x14xf32>
    %v308 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v309 = stablehlo.add %v307, %v308 : tensor<64x160x14x14xf32>
    %v310 = stablehlo.reshape %v309 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v311 = stablehlo.reshape %v310 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v312 = stablehlo.broadcast_in_dim %u4qnmu, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v313 = stablehlo.subtract %v311, %v312 : tensor<64x160x14x14xf32>
    %v314 = stablehlo.broadcast_in_dim %u4qnvar, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v315 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v316 = stablehlo.add %v314, %v315 : tensor<64x160x14x14xf32>
    %v317 = stablehlo.rsqrt %v316 : tensor<64x160x14x14xf32>
    %v318 = stablehlo.multiply %v313, %v317 : tensor<64x160x14x14xf32>
    %v319 = stablehlo.broadcast_in_dim %u4qg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v320 = stablehlo.broadcast_in_dim %u4qbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v321 = stablehlo.multiply %v318, %v319 : tensor<64x160x14x14xf32>
    %v322 = stablehlo.add %v321, %v320 : tensor<64x160x14x14xf32>
    %v323 = stablehlo.reshape %v322 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v324 = stablehlo.reshape %v323 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v325 = stablehlo.convolution(%v324, %u4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<64x640x14x14xf32>
    %v326 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v327 = stablehlo.add %v325, %v326 : tensor<64x640x14x14xf32>
    %v328 = stablehlo.reshape %v327 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v329 = stablehlo.reshape %v328 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v330 = stablehlo.broadcast_in_dim %u4enmu, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v331 = stablehlo.subtract %v329, %v330 : tensor<64x640x14x14xf32>
    %v332 = stablehlo.broadcast_in_dim %u4envar, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v333 = stablehlo.constant dense<1.0e-5> : tensor<64x640x14x14xf32>
    %v334 = stablehlo.add %v332, %v333 : tensor<64x640x14x14xf32>
    %v335 = stablehlo.rsqrt %v334 : tensor<64x640x14x14xf32>
    %v336 = stablehlo.multiply %v331, %v335 : tensor<64x640x14x14xf32>
    %v337 = stablehlo.broadcast_in_dim %u4eg, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v338 = stablehlo.broadcast_in_dim %u4ebt, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v339 = stablehlo.multiply %v336, %v337 : tensor<64x640x14x14xf32>
    %v340 = stablehlo.add %v339, %v338 : tensor<64x640x14x14xf32>
    %v341 = stablehlo.reshape %v340 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v342 = stablehlo.reshape %v341 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v343 = stablehlo.constant dense<0.0> : tensor<64x640x14x14xf32>
    %v344 = stablehlo.maximum %v342, %v343 : tensor<64x640x14x14xf32>
    %v345 = stablehlo.reshape %v344 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v346 = stablehlo.reshape %v345 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v347 = stablehlo.convolution(%v346, %u4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 640 : i64} : (tensor<64x640x14x14xf32>, tensor<640x1x3x3xf32>) -> tensor<64x640x14x14xf32>
    %v348 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v349 = stablehlo.add %v347, %v348 : tensor<64x640x14x14xf32>
    %v350 = stablehlo.reshape %v349 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v351 = stablehlo.reshape %v350 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v352 = stablehlo.broadcast_in_dim %u4dnmu, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v353 = stablehlo.subtract %v351, %v352 : tensor<64x640x14x14xf32>
    %v354 = stablehlo.broadcast_in_dim %u4dnvar, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v355 = stablehlo.constant dense<1.0e-5> : tensor<64x640x14x14xf32>
    %v356 = stablehlo.add %v354, %v355 : tensor<64x640x14x14xf32>
    %v357 = stablehlo.rsqrt %v356 : tensor<64x640x14x14xf32>
    %v358 = stablehlo.multiply %v353, %v357 : tensor<64x640x14x14xf32>
    %v359 = stablehlo.broadcast_in_dim %u4dg, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v360 = stablehlo.broadcast_in_dim %u4dbt, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v361 = stablehlo.multiply %v358, %v359 : tensor<64x640x14x14xf32>
    %v362 = stablehlo.add %v361, %v360 : tensor<64x640x14x14xf32>
    %v363 = stablehlo.reshape %v362 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v364 = stablehlo.reshape %v363 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v365 = stablehlo.constant dense<0.0> : tensor<64x640x14x14xf32>
    %v366 = stablehlo.maximum %v364, %v365 : tensor<64x640x14x14xf32>
    %v367 = stablehlo.reshape %v366 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v368 = stablehlo.reshape %v367 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v369 = stablehlo.convolution(%v368, %u4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<64x160x14x14xf32>
    %v370 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v371 = stablehlo.add %v369, %v370 : tensor<64x160x14x14xf32>
    %v372 = stablehlo.reshape %v371 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v373 = stablehlo.reshape %v372 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v374 = stablehlo.broadcast_in_dim %u4pnmu, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v375 = stablehlo.subtract %v373, %v374 : tensor<64x160x14x14xf32>
    %v376 = stablehlo.broadcast_in_dim %u4pnvar, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v377 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v378 = stablehlo.add %v376, %v377 : tensor<64x160x14x14xf32>
    %v379 = stablehlo.rsqrt %v378 : tensor<64x160x14x14xf32>
    %v380 = stablehlo.multiply %v375, %v379 : tensor<64x160x14x14xf32>
    %v381 = stablehlo.broadcast_in_dim %u4pg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v382 = stablehlo.broadcast_in_dim %u4pbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v383 = stablehlo.multiply %v380, %v381 : tensor<64x160x14x14xf32>
    %v384 = stablehlo.add %v383, %v382 : tensor<64x160x14x14xf32>
    %v385 = stablehlo.reshape %v384 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v386 = stablehlo.reshape %v385 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v387 = stablehlo.reshape %v305 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v388 = stablehlo.add %v386, %v387 : tensor<64x160x14x14xf32>
    %v389 = stablehlo.reshape %v388 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v390 = stablehlo.reshape %v389 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v391 = stablehlo.convolution(%v390, %u5qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<64x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<64x160x14x14xf32>
    %v392 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v393 = stablehlo.add %v391, %v392 : tensor<64x160x14x14xf32>
    %v394 = stablehlo.reshape %v393 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v395 = stablehlo.reshape %v394 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v396 = stablehlo.broadcast_in_dim %u5qnmu, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v397 = stablehlo.subtract %v395, %v396 : tensor<64x160x14x14xf32>
    %v398 = stablehlo.broadcast_in_dim %u5qnvar, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v399 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v400 = stablehlo.add %v398, %v399 : tensor<64x160x14x14xf32>
    %v401 = stablehlo.rsqrt %v400 : tensor<64x160x14x14xf32>
    %v402 = stablehlo.multiply %v397, %v401 : tensor<64x160x14x14xf32>
    %v403 = stablehlo.broadcast_in_dim %u5qg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v404 = stablehlo.broadcast_in_dim %u5qbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v405 = stablehlo.multiply %v402, %v403 : tensor<64x160x14x14xf32>
    %v406 = stablehlo.add %v405, %v404 : tensor<64x160x14x14xf32>
    %v407 = stablehlo.reshape %v406 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v408 = stablehlo.reshape %v407 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v409 = stablehlo.convolution(%v408, %u5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<64x640x14x14xf32>
    %v410 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v411 = stablehlo.add %v409, %v410 : tensor<64x640x14x14xf32>
    %v412 = stablehlo.reshape %v411 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v413 = stablehlo.reshape %v412 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v414 = stablehlo.broadcast_in_dim %u5enmu, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v415 = stablehlo.subtract %v413, %v414 : tensor<64x640x14x14xf32>
    %v416 = stablehlo.broadcast_in_dim %u5envar, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v417 = stablehlo.constant dense<1.0e-5> : tensor<64x640x14x14xf32>
    %v418 = stablehlo.add %v416, %v417 : tensor<64x640x14x14xf32>
    %v419 = stablehlo.rsqrt %v418 : tensor<64x640x14x14xf32>
    %v420 = stablehlo.multiply %v415, %v419 : tensor<64x640x14x14xf32>
    %v421 = stablehlo.broadcast_in_dim %u5eg, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v422 = stablehlo.broadcast_in_dim %u5ebt, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v423 = stablehlo.multiply %v420, %v421 : tensor<64x640x14x14xf32>
    %v424 = stablehlo.add %v423, %v422 : tensor<64x640x14x14xf32>
    %v425 = stablehlo.reshape %v424 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v426 = stablehlo.reshape %v425 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v427 = stablehlo.constant dense<0.0> : tensor<64x640x14x14xf32>
    %v428 = stablehlo.maximum %v426, %v427 : tensor<64x640x14x14xf32>
    %v429 = stablehlo.reshape %v428 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v430 = stablehlo.reshape %v429 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v431 = stablehlo.convolution(%v430, %u5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 640 : i64} : (tensor<64x640x14x14xf32>, tensor<640x1x3x3xf32>) -> tensor<64x640x14x14xf32>
    %v432 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v433 = stablehlo.add %v431, %v432 : tensor<64x640x14x14xf32>
    %v434 = stablehlo.reshape %v433 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v435 = stablehlo.reshape %v434 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v436 = stablehlo.broadcast_in_dim %u5dnmu, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v437 = stablehlo.subtract %v435, %v436 : tensor<64x640x14x14xf32>
    %v438 = stablehlo.broadcast_in_dim %u5dnvar, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v439 = stablehlo.constant dense<1.0e-5> : tensor<64x640x14x14xf32>
    %v440 = stablehlo.add %v438, %v439 : tensor<64x640x14x14xf32>
    %v441 = stablehlo.rsqrt %v440 : tensor<64x640x14x14xf32>
    %v442 = stablehlo.multiply %v437, %v441 : tensor<64x640x14x14xf32>
    %v443 = stablehlo.broadcast_in_dim %u5dg, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v444 = stablehlo.broadcast_in_dim %u5dbt, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v445 = stablehlo.multiply %v442, %v443 : tensor<64x640x14x14xf32>
    %v446 = stablehlo.add %v445, %v444 : tensor<64x640x14x14xf32>
    %v447 = stablehlo.reshape %v446 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v448 = stablehlo.reshape %v447 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v449 = stablehlo.constant dense<0.0> : tensor<64x640x14x14xf32>
    %v450 = stablehlo.maximum %v448, %v449 : tensor<64x640x14x14xf32>
    %v451 = stablehlo.reshape %v450 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v452 = stablehlo.reshape %v451 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v453 = stablehlo.convolution(%v452, %u5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<64x160x14x14xf32>
    %v454 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v455 = stablehlo.add %v453, %v454 : tensor<64x160x14x14xf32>
    %v456 = stablehlo.reshape %v455 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v457 = stablehlo.reshape %v456 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v458 = stablehlo.broadcast_in_dim %u5pnmu, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v459 = stablehlo.subtract %v457, %v458 : tensor<64x160x14x14xf32>
    %v460 = stablehlo.broadcast_in_dim %u5pnvar, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v461 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v462 = stablehlo.add %v460, %v461 : tensor<64x160x14x14xf32>
    %v463 = stablehlo.rsqrt %v462 : tensor<64x160x14x14xf32>
    %v464 = stablehlo.multiply %v459, %v463 : tensor<64x160x14x14xf32>
    %v465 = stablehlo.broadcast_in_dim %u5pg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v466 = stablehlo.broadcast_in_dim %u5pbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v467 = stablehlo.multiply %v464, %v465 : tensor<64x160x14x14xf32>
    %v468 = stablehlo.add %v467, %v466 : tensor<64x160x14x14xf32>
    %v469 = stablehlo.reshape %v468 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v470 = stablehlo.reshape %v469 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v471 = stablehlo.reshape %v389 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v472 = stablehlo.add %v470, %v471 : tensor<64x160x14x14xf32>
    %v473 = stablehlo.reshape %v472 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v474 = stablehlo.reshape %v473 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v475 = stablehlo.convolution(%v474, %u6qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<64x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<64x160x14x14xf32>
    %v476 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v477 = stablehlo.add %v475, %v476 : tensor<64x160x14x14xf32>
    %v478 = stablehlo.reshape %v477 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v479 = stablehlo.reshape %v478 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v480 = stablehlo.broadcast_in_dim %u6qnmu, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v481 = stablehlo.subtract %v479, %v480 : tensor<64x160x14x14xf32>
    %v482 = stablehlo.broadcast_in_dim %u6qnvar, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v483 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v484 = stablehlo.add %v482, %v483 : tensor<64x160x14x14xf32>
    %v485 = stablehlo.rsqrt %v484 : tensor<64x160x14x14xf32>
    %v486 = stablehlo.multiply %v481, %v485 : tensor<64x160x14x14xf32>
    %v487 = stablehlo.broadcast_in_dim %u6qg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v488 = stablehlo.broadcast_in_dim %u6qbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v489 = stablehlo.multiply %v486, %v487 : tensor<64x160x14x14xf32>
    %v490 = stablehlo.add %v489, %v488 : tensor<64x160x14x14xf32>
    %v491 = stablehlo.reshape %v490 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v492 = stablehlo.reshape %v491 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v493 = stablehlo.convolution(%v492, %u6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<64x640x14x14xf32>
    %v494 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v495 = stablehlo.add %v493, %v494 : tensor<64x640x14x14xf32>
    %v496 = stablehlo.reshape %v495 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v497 = stablehlo.reshape %v496 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v498 = stablehlo.broadcast_in_dim %u6enmu, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v499 = stablehlo.subtract %v497, %v498 : tensor<64x640x14x14xf32>
    %v500 = stablehlo.broadcast_in_dim %u6envar, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v501 = stablehlo.constant dense<1.0e-5> : tensor<64x640x14x14xf32>
    %v502 = stablehlo.add %v500, %v501 : tensor<64x640x14x14xf32>
    %v503 = stablehlo.rsqrt %v502 : tensor<64x640x14x14xf32>
    %v504 = stablehlo.multiply %v499, %v503 : tensor<64x640x14x14xf32>
    %v505 = stablehlo.broadcast_in_dim %u6eg, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v506 = stablehlo.broadcast_in_dim %u6ebt, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v507 = stablehlo.multiply %v504, %v505 : tensor<64x640x14x14xf32>
    %v508 = stablehlo.add %v507, %v506 : tensor<64x640x14x14xf32>
    %v509 = stablehlo.reshape %v508 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v510 = stablehlo.reshape %v509 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v511 = stablehlo.constant dense<0.0> : tensor<64x640x14x14xf32>
    %v512 = stablehlo.maximum %v510, %v511 : tensor<64x640x14x14xf32>
    %v513 = stablehlo.reshape %v512 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v514 = stablehlo.reshape %v513 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v515 = stablehlo.convolution(%v514, %u6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 640 : i64} : (tensor<64x640x14x14xf32>, tensor<640x1x5x5xf32>) -> tensor<64x640x14x14xf32>
    %v516 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v517 = stablehlo.add %v515, %v516 : tensor<64x640x14x14xf32>
    %v518 = stablehlo.reshape %v517 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v519 = stablehlo.reshape %v518 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v520 = stablehlo.broadcast_in_dim %u6dnmu, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v521 = stablehlo.subtract %v519, %v520 : tensor<64x640x14x14xf32>
    %v522 = stablehlo.broadcast_in_dim %u6dnvar, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v523 = stablehlo.constant dense<1.0e-5> : tensor<64x640x14x14xf32>
    %v524 = stablehlo.add %v522, %v523 : tensor<64x640x14x14xf32>
    %v525 = stablehlo.rsqrt %v524 : tensor<64x640x14x14xf32>
    %v526 = stablehlo.multiply %v521, %v525 : tensor<64x640x14x14xf32>
    %v527 = stablehlo.broadcast_in_dim %u6dg, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v528 = stablehlo.broadcast_in_dim %u6dbt, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v529 = stablehlo.multiply %v526, %v527 : tensor<64x640x14x14xf32>
    %v530 = stablehlo.add %v529, %v528 : tensor<64x640x14x14xf32>
    %v531 = stablehlo.reshape %v530 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v532 = stablehlo.reshape %v531 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v533 = stablehlo.constant dense<0.0> : tensor<64x640x14x14xf32>
    %v534 = stablehlo.maximum %v532, %v533 : tensor<64x640x14x14xf32>
    %v535 = stablehlo.reshape %v534 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v536 = stablehlo.reshape %v535 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v537 = stablehlo.convolution(%v536, %u6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<64x160x14x14xf32>
    %v538 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v539 = stablehlo.add %v537, %v538 : tensor<64x160x14x14xf32>
    %v540 = stablehlo.reshape %v539 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v541 = stablehlo.reshape %v540 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v542 = stablehlo.broadcast_in_dim %u6pnmu, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v543 = stablehlo.subtract %v541, %v542 : tensor<64x160x14x14xf32>
    %v544 = stablehlo.broadcast_in_dim %u6pnvar, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v545 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v546 = stablehlo.add %v544, %v545 : tensor<64x160x14x14xf32>
    %v547 = stablehlo.rsqrt %v546 : tensor<64x160x14x14xf32>
    %v548 = stablehlo.multiply %v543, %v547 : tensor<64x160x14x14xf32>
    %v549 = stablehlo.broadcast_in_dim %u6pg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v550 = stablehlo.broadcast_in_dim %u6pbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v551 = stablehlo.multiply %v548, %v549 : tensor<64x160x14x14xf32>
    %v552 = stablehlo.add %v551, %v550 : tensor<64x160x14x14xf32>
    %v553 = stablehlo.reshape %v552 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v554 = stablehlo.reshape %v553 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v555 = stablehlo.reshape %v473 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v556 = stablehlo.add %v554, %v555 : tensor<64x160x14x14xf32>
    %v557 = stablehlo.reshape %v556 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v558 = stablehlo.reshape %v557 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v559 = stablehlo.convolution(%v558, %u7qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<64x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<64x160x14x14xf32>
    %v560 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v561 = stablehlo.add %v559, %v560 : tensor<64x160x14x14xf32>
    %v562 = stablehlo.reshape %v561 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v563 = stablehlo.reshape %v562 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v564 = stablehlo.broadcast_in_dim %u7qnmu, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v565 = stablehlo.subtract %v563, %v564 : tensor<64x160x14x14xf32>
    %v566 = stablehlo.broadcast_in_dim %u7qnvar, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v567 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v568 = stablehlo.add %v566, %v567 : tensor<64x160x14x14xf32>
    %v569 = stablehlo.rsqrt %v568 : tensor<64x160x14x14xf32>
    %v570 = stablehlo.multiply %v565, %v569 : tensor<64x160x14x14xf32>
    %v571 = stablehlo.broadcast_in_dim %u7qg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v572 = stablehlo.broadcast_in_dim %u7qbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v573 = stablehlo.multiply %v570, %v571 : tensor<64x160x14x14xf32>
    %v574 = stablehlo.add %v573, %v572 : tensor<64x160x14x14xf32>
    %v575 = stablehlo.reshape %v574 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v576 = stablehlo.reshape %v575 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v577 = stablehlo.convolution(%v576, %u7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<64x640x14x14xf32>
    %v578 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v579 = stablehlo.add %v577, %v578 : tensor<64x640x14x14xf32>
    %v580 = stablehlo.reshape %v579 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v581 = stablehlo.reshape %v580 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v582 = stablehlo.broadcast_in_dim %u7enmu, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v583 = stablehlo.subtract %v581, %v582 : tensor<64x640x14x14xf32>
    %v584 = stablehlo.broadcast_in_dim %u7envar, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v585 = stablehlo.constant dense<1.0e-5> : tensor<64x640x14x14xf32>
    %v586 = stablehlo.add %v584, %v585 : tensor<64x640x14x14xf32>
    %v587 = stablehlo.rsqrt %v586 : tensor<64x640x14x14xf32>
    %v588 = stablehlo.multiply %v583, %v587 : tensor<64x640x14x14xf32>
    %v589 = stablehlo.broadcast_in_dim %u7eg, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v590 = stablehlo.broadcast_in_dim %u7ebt, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v591 = stablehlo.multiply %v588, %v589 : tensor<64x640x14x14xf32>
    %v592 = stablehlo.add %v591, %v590 : tensor<64x640x14x14xf32>
    %v593 = stablehlo.reshape %v592 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v594 = stablehlo.reshape %v593 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v595 = stablehlo.constant dense<0.0> : tensor<64x640x14x14xf32>
    %v596 = stablehlo.maximum %v594, %v595 : tensor<64x640x14x14xf32>
    %v597 = stablehlo.reshape %v596 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v598 = stablehlo.reshape %v597 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v599 = stablehlo.convolution(%v598, %u7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 640 : i64} : (tensor<64x640x14x14xf32>, tensor<640x1x3x3xf32>) -> tensor<64x640x14x14xf32>
    %v600 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v601 = stablehlo.add %v599, %v600 : tensor<64x640x14x14xf32>
    %v602 = stablehlo.reshape %v601 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v603 = stablehlo.reshape %v602 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v604 = stablehlo.broadcast_in_dim %u7dnmu, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v605 = stablehlo.subtract %v603, %v604 : tensor<64x640x14x14xf32>
    %v606 = stablehlo.broadcast_in_dim %u7dnvar, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v607 = stablehlo.constant dense<1.0e-5> : tensor<64x640x14x14xf32>
    %v608 = stablehlo.add %v606, %v607 : tensor<64x640x14x14xf32>
    %v609 = stablehlo.rsqrt %v608 : tensor<64x640x14x14xf32>
    %v610 = stablehlo.multiply %v605, %v609 : tensor<64x640x14x14xf32>
    %v611 = stablehlo.broadcast_in_dim %u7dg, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v612 = stablehlo.broadcast_in_dim %u7dbt, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v613 = stablehlo.multiply %v610, %v611 : tensor<64x640x14x14xf32>
    %v614 = stablehlo.add %v613, %v612 : tensor<64x640x14x14xf32>
    %v615 = stablehlo.reshape %v614 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v616 = stablehlo.reshape %v615 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v617 = stablehlo.constant dense<0.0> : tensor<64x640x14x14xf32>
    %v618 = stablehlo.maximum %v616, %v617 : tensor<64x640x14x14xf32>
    %v619 = stablehlo.reshape %v618 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v620 = stablehlo.reshape %v619 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v621 = stablehlo.convolution(%v620, %u7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<64x160x14x14xf32>
    %v622 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v623 = stablehlo.add %v621, %v622 : tensor<64x160x14x14xf32>
    %v624 = stablehlo.reshape %v623 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v625 = stablehlo.reshape %v624 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v626 = stablehlo.broadcast_in_dim %u7pnmu, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v627 = stablehlo.subtract %v625, %v626 : tensor<64x160x14x14xf32>
    %v628 = stablehlo.broadcast_in_dim %u7pnvar, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v629 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v630 = stablehlo.add %v628, %v629 : tensor<64x160x14x14xf32>
    %v631 = stablehlo.rsqrt %v630 : tensor<64x160x14x14xf32>
    %v632 = stablehlo.multiply %v627, %v631 : tensor<64x160x14x14xf32>
    %v633 = stablehlo.broadcast_in_dim %u7pg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v634 = stablehlo.broadcast_in_dim %u7pbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v635 = stablehlo.multiply %v632, %v633 : tensor<64x160x14x14xf32>
    %v636 = stablehlo.add %v635, %v634 : tensor<64x160x14x14xf32>
    %v637 = stablehlo.reshape %v636 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v638 = stablehlo.reshape %v637 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v639 = stablehlo.reshape %v557 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v640 = stablehlo.add %v638, %v639 : tensor<64x160x14x14xf32>
    %v641 = stablehlo.reshape %v640 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v642 = stablehlo.reshape %v641 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v643 = stablehlo.convolution(%v642, %u8qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<64x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<64x160x14x14xf32>
    %v644 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v645 = stablehlo.add %v643, %v644 : tensor<64x160x14x14xf32>
    %v646 = stablehlo.reshape %v645 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v647 = stablehlo.reshape %v646 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v648 = stablehlo.broadcast_in_dim %u8qnmu, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v649 = stablehlo.subtract %v647, %v648 : tensor<64x160x14x14xf32>
    %v650 = stablehlo.broadcast_in_dim %u8qnvar, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v651 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v652 = stablehlo.add %v650, %v651 : tensor<64x160x14x14xf32>
    %v653 = stablehlo.rsqrt %v652 : tensor<64x160x14x14xf32>
    %v654 = stablehlo.multiply %v649, %v653 : tensor<64x160x14x14xf32>
    %v655 = stablehlo.broadcast_in_dim %u8qg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v656 = stablehlo.broadcast_in_dim %u8qbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v657 = stablehlo.multiply %v654, %v655 : tensor<64x160x14x14xf32>
    %v658 = stablehlo.add %v657, %v656 : tensor<64x160x14x14xf32>
    %v659 = stablehlo.reshape %v658 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v660 = stablehlo.reshape %v659 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v661 = stablehlo.convolution(%v660, %u8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<64x640x14x14xf32>
    %v662 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v663 = stablehlo.add %v661, %v662 : tensor<64x640x14x14xf32>
    %v664 = stablehlo.reshape %v663 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v665 = stablehlo.reshape %v664 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v666 = stablehlo.broadcast_in_dim %u8enmu, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v667 = stablehlo.subtract %v665, %v666 : tensor<64x640x14x14xf32>
    %v668 = stablehlo.broadcast_in_dim %u8envar, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v669 = stablehlo.constant dense<1.0e-5> : tensor<64x640x14x14xf32>
    %v670 = stablehlo.add %v668, %v669 : tensor<64x640x14x14xf32>
    %v671 = stablehlo.rsqrt %v670 : tensor<64x640x14x14xf32>
    %v672 = stablehlo.multiply %v667, %v671 : tensor<64x640x14x14xf32>
    %v673 = stablehlo.broadcast_in_dim %u8eg, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v674 = stablehlo.broadcast_in_dim %u8ebt, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v675 = stablehlo.multiply %v672, %v673 : tensor<64x640x14x14xf32>
    %v676 = stablehlo.add %v675, %v674 : tensor<64x640x14x14xf32>
    %v677 = stablehlo.reshape %v676 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v678 = stablehlo.reshape %v677 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v679 = stablehlo.constant dense<0.0> : tensor<64x640x14x14xf32>
    %v680 = stablehlo.maximum %v678, %v679 : tensor<64x640x14x14xf32>
    %v681 = stablehlo.reshape %v680 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v682 = stablehlo.reshape %v681 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v683 = stablehlo.convolution(%v682, %u8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<64x160x14x14xf32>
    %v684 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v685 = stablehlo.add %v683, %v684 : tensor<64x160x14x14xf32>
    %v686 = stablehlo.reshape %v685 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v687 = stablehlo.reshape %v686 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v688 = stablehlo.broadcast_in_dim %u8pnmu, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v689 = stablehlo.subtract %v687, %v688 : tensor<64x160x14x14xf32>
    %v690 = stablehlo.broadcast_in_dim %u8pnvar, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v691 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v692 = stablehlo.add %v690, %v691 : tensor<64x160x14x14xf32>
    %v693 = stablehlo.rsqrt %v692 : tensor<64x160x14x14xf32>
    %v694 = stablehlo.multiply %v689, %v693 : tensor<64x160x14x14xf32>
    %v695 = stablehlo.broadcast_in_dim %u8pg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v696 = stablehlo.broadcast_in_dim %u8pbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v697 = stablehlo.multiply %v694, %v695 : tensor<64x160x14x14xf32>
    %v698 = stablehlo.add %v697, %v696 : tensor<64x160x14x14xf32>
    %v699 = stablehlo.reshape %v698 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v700 = stablehlo.reshape %v699 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v701 = stablehlo.reshape %v641 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v702 = stablehlo.add %v700, %v701 : tensor<64x160x14x14xf32>
    %v703 = stablehlo.reshape %v702 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v704 = stablehlo.reshape %v703 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v705 = stablehlo.convolution(%v704, %u9eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x160x14x14xf32>, tensor<320x160x1x1xf32>) -> tensor<64x320x14x14xf32>
    %v706 = stablehlo.broadcast_in_dim %zb320, dims = [1] : (tensor<320xf32>) -> tensor<64x320x14x14xf32>
    %v707 = stablehlo.add %v705, %v706 : tensor<64x320x14x14xf32>
    %v708 = stablehlo.reshape %v707 : (tensor<64x320x14x14xf32>) -> tensor<64x62720xf32>
    %v709 = stablehlo.reshape %v708 : (tensor<64x62720xf32>) -> tensor<64x320x14x14xf32>
    %v710 = stablehlo.broadcast_in_dim %u9enmu, dims = [1] : (tensor<320xf32>) -> tensor<64x320x14x14xf32>
    %v711 = stablehlo.subtract %v709, %v710 : tensor<64x320x14x14xf32>
    %v712 = stablehlo.broadcast_in_dim %u9envar, dims = [1] : (tensor<320xf32>) -> tensor<64x320x14x14xf32>
    %v713 = stablehlo.constant dense<1.0e-5> : tensor<64x320x14x14xf32>
    %v714 = stablehlo.add %v712, %v713 : tensor<64x320x14x14xf32>
    %v715 = stablehlo.rsqrt %v714 : tensor<64x320x14x14xf32>
    %v716 = stablehlo.multiply %v711, %v715 : tensor<64x320x14x14xf32>
    %v717 = stablehlo.broadcast_in_dim %u9eg, dims = [1] : (tensor<320xf32>) -> tensor<64x320x14x14xf32>
    %v718 = stablehlo.broadcast_in_dim %u9ebt, dims = [1] : (tensor<320xf32>) -> tensor<64x320x14x14xf32>
    %v719 = stablehlo.multiply %v716, %v717 : tensor<64x320x14x14xf32>
    %v720 = stablehlo.add %v719, %v718 : tensor<64x320x14x14xf32>
    %v721 = stablehlo.reshape %v720 : (tensor<64x320x14x14xf32>) -> tensor<64x62720xf32>
    %v722 = stablehlo.reshape %v721 : (tensor<64x62720xf32>) -> tensor<64x320x14x14xf32>
    %v723 = stablehlo.constant dense<0.0> : tensor<64x320x14x14xf32>
    %v724 = stablehlo.maximum %v722, %v723 : tensor<64x320x14x14xf32>
    %v725 = stablehlo.reshape %v724 : (tensor<64x320x14x14xf32>) -> tensor<64x62720xf32>
    %v726 = stablehlo.reshape %v725 : (tensor<64x62720xf32>) -> tensor<64x320x14x14xf32>
    %v727 = stablehlo.convolution(%v726, %u9pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x320x14x14xf32>, tensor<160x320x1x1xf32>) -> tensor<64x160x14x14xf32>
    %v728 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v729 = stablehlo.add %v727, %v728 : tensor<64x160x14x14xf32>
    %v730 = stablehlo.reshape %v729 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v731 = stablehlo.reshape %v730 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v732 = stablehlo.broadcast_in_dim %u9pnmu, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v733 = stablehlo.subtract %v731, %v732 : tensor<64x160x14x14xf32>
    %v734 = stablehlo.broadcast_in_dim %u9pnvar, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v735 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v736 = stablehlo.add %v734, %v735 : tensor<64x160x14x14xf32>
    %v737 = stablehlo.rsqrt %v736 : tensor<64x160x14x14xf32>
    %v738 = stablehlo.multiply %v733, %v737 : tensor<64x160x14x14xf32>
    %v739 = stablehlo.broadcast_in_dim %u9pg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v740 = stablehlo.broadcast_in_dim %u9pbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v741 = stablehlo.multiply %v738, %v739 : tensor<64x160x14x14xf32>
    %v742 = stablehlo.add %v741, %v740 : tensor<64x160x14x14xf32>
    %v743 = stablehlo.reshape %v742 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v744 = stablehlo.reshape %v743 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v745 = stablehlo.reshape %v703 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v746 = stablehlo.add %v744, %v745 : tensor<64x160x14x14xf32>
    %v747 = stablehlo.reshape %v746 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v748 = stablehlo.reshape %v747 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v749 = stablehlo.convolution(%v748, %u10qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<64x160x14x14xf32>, tensor<160x1x3x3xf32>) -> tensor<64x160x14x14xf32>
    %v750 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v751 = stablehlo.add %v749, %v750 : tensor<64x160x14x14xf32>
    %v752 = stablehlo.reshape %v751 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v753 = stablehlo.reshape %v752 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v754 = stablehlo.broadcast_in_dim %u10qnmu, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v755 = stablehlo.subtract %v753, %v754 : tensor<64x160x14x14xf32>
    %v756 = stablehlo.broadcast_in_dim %u10qnvar, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v757 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v758 = stablehlo.add %v756, %v757 : tensor<64x160x14x14xf32>
    %v759 = stablehlo.rsqrt %v758 : tensor<64x160x14x14xf32>
    %v760 = stablehlo.multiply %v755, %v759 : tensor<64x160x14x14xf32>
    %v761 = stablehlo.broadcast_in_dim %u10qg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v762 = stablehlo.broadcast_in_dim %u10qbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v763 = stablehlo.multiply %v760, %v761 : tensor<64x160x14x14xf32>
    %v764 = stablehlo.add %v763, %v762 : tensor<64x160x14x14xf32>
    %v765 = stablehlo.reshape %v764 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v766 = stablehlo.reshape %v765 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v767 = stablehlo.convolution(%v766, %u10eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x160x14x14xf32>, tensor<640x160x1x1xf32>) -> tensor<64x640x14x14xf32>
    %v768 = stablehlo.broadcast_in_dim %zb640, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v769 = stablehlo.add %v767, %v768 : tensor<64x640x14x14xf32>
    %v770 = stablehlo.reshape %v769 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v771 = stablehlo.reshape %v770 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v772 = stablehlo.broadcast_in_dim %u10enmu, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v773 = stablehlo.subtract %v771, %v772 : tensor<64x640x14x14xf32>
    %v774 = stablehlo.broadcast_in_dim %u10envar, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v775 = stablehlo.constant dense<1.0e-5> : tensor<64x640x14x14xf32>
    %v776 = stablehlo.add %v774, %v775 : tensor<64x640x14x14xf32>
    %v777 = stablehlo.rsqrt %v776 : tensor<64x640x14x14xf32>
    %v778 = stablehlo.multiply %v773, %v777 : tensor<64x640x14x14xf32>
    %v779 = stablehlo.broadcast_in_dim %u10eg, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v780 = stablehlo.broadcast_in_dim %u10ebt, dims = [1] : (tensor<640xf32>) -> tensor<64x640x14x14xf32>
    %v781 = stablehlo.multiply %v778, %v779 : tensor<64x640x14x14xf32>
    %v782 = stablehlo.add %v781, %v780 : tensor<64x640x14x14xf32>
    %v783 = stablehlo.reshape %v782 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v784 = stablehlo.reshape %v783 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v785 = stablehlo.constant dense<0.0> : tensor<64x640x14x14xf32>
    %v786 = stablehlo.maximum %v784, %v785 : tensor<64x640x14x14xf32>
    %v787 = stablehlo.reshape %v786 : (tensor<64x640x14x14xf32>) -> tensor<64x125440xf32>
    %v788 = stablehlo.reshape %v787 : (tensor<64x125440xf32>) -> tensor<64x640x14x14xf32>
    %v789 = stablehlo.convolution(%v788, %u10pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x640x14x14xf32>, tensor<160x640x1x1xf32>) -> tensor<64x160x14x14xf32>
    %v790 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v791 = stablehlo.add %v789, %v790 : tensor<64x160x14x14xf32>
    %v792 = stablehlo.reshape %v791 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v793 = stablehlo.reshape %v792 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v794 = stablehlo.broadcast_in_dim %u10pnmu, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v795 = stablehlo.subtract %v793, %v794 : tensor<64x160x14x14xf32>
    %v796 = stablehlo.broadcast_in_dim %u10pnvar, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v797 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v798 = stablehlo.add %v796, %v797 : tensor<64x160x14x14xf32>
    %v799 = stablehlo.rsqrt %v798 : tensor<64x160x14x14xf32>
    %v800 = stablehlo.multiply %v795, %v799 : tensor<64x160x14x14xf32>
    %v801 = stablehlo.broadcast_in_dim %u10pg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v802 = stablehlo.broadcast_in_dim %u10pbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v803 = stablehlo.multiply %v800, %v801 : tensor<64x160x14x14xf32>
    %v804 = stablehlo.add %v803, %v802 : tensor<64x160x14x14xf32>
    %v805 = stablehlo.reshape %v804 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v806 = stablehlo.reshape %v805 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v807 = stablehlo.reshape %v747 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v808 = stablehlo.add %v806, %v807 : tensor<64x160x14x14xf32>
    %v809 = stablehlo.reshape %v808 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v810 = stablehlo.reshape %v809 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v811 = stablehlo.convolution(%v810, %u11qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 160 : i64} : (tensor<64x160x14x14xf32>, tensor<160x1x5x5xf32>) -> tensor<64x160x14x14xf32>
    %v812 = stablehlo.broadcast_in_dim %zb160, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v813 = stablehlo.add %v811, %v812 : tensor<64x160x14x14xf32>
    %v814 = stablehlo.reshape %v813 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v815 = stablehlo.reshape %v814 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v816 = stablehlo.broadcast_in_dim %u11qnmu, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v817 = stablehlo.subtract %v815, %v816 : tensor<64x160x14x14xf32>
    %v818 = stablehlo.broadcast_in_dim %u11qnvar, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v819 = stablehlo.constant dense<1.0e-5> : tensor<64x160x14x14xf32>
    %v820 = stablehlo.add %v818, %v819 : tensor<64x160x14x14xf32>
    %v821 = stablehlo.rsqrt %v820 : tensor<64x160x14x14xf32>
    %v822 = stablehlo.multiply %v817, %v821 : tensor<64x160x14x14xf32>
    %v823 = stablehlo.broadcast_in_dim %u11qg, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v824 = stablehlo.broadcast_in_dim %u11qbt, dims = [1] : (tensor<160xf32>) -> tensor<64x160x14x14xf32>
    %v825 = stablehlo.multiply %v822, %v823 : tensor<64x160x14x14xf32>
    %v826 = stablehlo.add %v825, %v824 : tensor<64x160x14x14xf32>
    %v827 = stablehlo.reshape %v826 : (tensor<64x160x14x14xf32>) -> tensor<64x31360xf32>
    %v828 = stablehlo.reshape %v827 : (tensor<64x31360xf32>) -> tensor<64x160x14x14xf32>
    %v829 = stablehlo.convolution(%v828, %u11eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x160x14x14xf32>, tensor<960x160x1x1xf32>) -> tensor<64x960x14x14xf32>
    %v830 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<64x960x14x14xf32>
    %v831 = stablehlo.add %v829, %v830 : tensor<64x960x14x14xf32>
    %v832 = stablehlo.reshape %v831 : (tensor<64x960x14x14xf32>) -> tensor<64x188160xf32>
    %v833 = stablehlo.reshape %v832 : (tensor<64x188160xf32>) -> tensor<64x960x14x14xf32>
    %v834 = stablehlo.broadcast_in_dim %u11enmu, dims = [1] : (tensor<960xf32>) -> tensor<64x960x14x14xf32>
    %v835 = stablehlo.subtract %v833, %v834 : tensor<64x960x14x14xf32>
    %v836 = stablehlo.broadcast_in_dim %u11envar, dims = [1] : (tensor<960xf32>) -> tensor<64x960x14x14xf32>
    %v837 = stablehlo.constant dense<1.0e-5> : tensor<64x960x14x14xf32>
    %v838 = stablehlo.add %v836, %v837 : tensor<64x960x14x14xf32>
    %v839 = stablehlo.rsqrt %v838 : tensor<64x960x14x14xf32>
    %v840 = stablehlo.multiply %v835, %v839 : tensor<64x960x14x14xf32>
    %v841 = stablehlo.broadcast_in_dim %u11eg, dims = [1] : (tensor<960xf32>) -> tensor<64x960x14x14xf32>
    %v842 = stablehlo.broadcast_in_dim %u11ebt, dims = [1] : (tensor<960xf32>) -> tensor<64x960x14x14xf32>
    %v843 = stablehlo.multiply %v840, %v841 : tensor<64x960x14x14xf32>
    %v844 = stablehlo.add %v843, %v842 : tensor<64x960x14x14xf32>
    %v845 = stablehlo.reshape %v844 : (tensor<64x960x14x14xf32>) -> tensor<64x188160xf32>
    %v846 = stablehlo.reshape %v845 : (tensor<64x188160xf32>) -> tensor<64x960x14x14xf32>
    %v847 = stablehlo.constant dense<0.0> : tensor<64x960x14x14xf32>
    %v848 = stablehlo.maximum %v846, %v847 : tensor<64x960x14x14xf32>
    %v849 = stablehlo.reshape %v848 : (tensor<64x960x14x14xf32>) -> tensor<64x188160xf32>
    %v850 = stablehlo.reshape %v849 : (tensor<64x188160xf32>) -> tensor<64x960x14x14xf32>
    %v851 = stablehlo.convolution(%v850, %u11dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 960 : i64} : (tensor<64x960x14x14xf32>, tensor<960x1x5x5xf32>) -> tensor<64x960x7x7xf32>
    %v852 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v853 = stablehlo.add %v851, %v852 : tensor<64x960x7x7xf32>
    %v854 = stablehlo.reshape %v853 : (tensor<64x960x7x7xf32>) -> tensor<64x47040xf32>
    %v855 = stablehlo.reshape %v854 : (tensor<64x47040xf32>) -> tensor<64x960x7x7xf32>
    %v856 = stablehlo.broadcast_in_dim %u11dnmu, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v857 = stablehlo.subtract %v855, %v856 : tensor<64x960x7x7xf32>
    %v858 = stablehlo.broadcast_in_dim %u11dnvar, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v859 = stablehlo.constant dense<1.0e-5> : tensor<64x960x7x7xf32>
    %v860 = stablehlo.add %v858, %v859 : tensor<64x960x7x7xf32>
    %v861 = stablehlo.rsqrt %v860 : tensor<64x960x7x7xf32>
    %v862 = stablehlo.multiply %v857, %v861 : tensor<64x960x7x7xf32>
    %v863 = stablehlo.broadcast_in_dim %u11dg, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v864 = stablehlo.broadcast_in_dim %u11dbt, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v865 = stablehlo.multiply %v862, %v863 : tensor<64x960x7x7xf32>
    %v866 = stablehlo.add %v865, %v864 : tensor<64x960x7x7xf32>
    %v867 = stablehlo.reshape %v866 : (tensor<64x960x7x7xf32>) -> tensor<64x47040xf32>
    %v868 = stablehlo.reshape %v867 : (tensor<64x47040xf32>) -> tensor<64x960x7x7xf32>
    %v869 = stablehlo.constant dense<0.0> : tensor<64x960x7x7xf32>
    %v870 = stablehlo.maximum %v868, %v869 : tensor<64x960x7x7xf32>
    %v871 = stablehlo.reshape %v870 : (tensor<64x960x7x7xf32>) -> tensor<64x47040xf32>
    %v872 = stablehlo.reshape %v871 : (tensor<64x47040xf32>) -> tensor<64x960x7x7xf32>
    %v873 = stablehlo.convolution(%v872, %u11pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x960x7x7xf32>, tensor<256x960x1x1xf32>) -> tensor<64x256x7x7xf32>
    %v874 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v875 = stablehlo.add %v873, %v874 : tensor<64x256x7x7xf32>
    %v876 = stablehlo.reshape %v875 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v877 = stablehlo.reshape %v876 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v878 = stablehlo.broadcast_in_dim %u11pnmu, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v879 = stablehlo.subtract %v877, %v878 : tensor<64x256x7x7xf32>
    %v880 = stablehlo.broadcast_in_dim %u11pnvar, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v881 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v882 = stablehlo.add %v880, %v881 : tensor<64x256x7x7xf32>
    %v883 = stablehlo.rsqrt %v882 : tensor<64x256x7x7xf32>
    %v884 = stablehlo.multiply %v879, %v883 : tensor<64x256x7x7xf32>
    %v885 = stablehlo.broadcast_in_dim %u11pg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v886 = stablehlo.broadcast_in_dim %u11pbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v887 = stablehlo.multiply %v884, %v885 : tensor<64x256x7x7xf32>
    %v888 = stablehlo.add %v887, %v886 : tensor<64x256x7x7xf32>
    %v889 = stablehlo.reshape %v888 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v890 = stablehlo.reshape %v889 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v891 = stablehlo.convolution(%v890, %u12qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<64x256x7x7xf32>, tensor<256x1x5x5xf32>) -> tensor<64x256x7x7xf32>
    %v892 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v893 = stablehlo.add %v891, %v892 : tensor<64x256x7x7xf32>
    %v894 = stablehlo.reshape %v893 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v895 = stablehlo.reshape %v894 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v896 = stablehlo.broadcast_in_dim %u12qnmu, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v897 = stablehlo.subtract %v895, %v896 : tensor<64x256x7x7xf32>
    %v898 = stablehlo.broadcast_in_dim %u12qnvar, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v899 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v900 = stablehlo.add %v898, %v899 : tensor<64x256x7x7xf32>
    %v901 = stablehlo.rsqrt %v900 : tensor<64x256x7x7xf32>
    %v902 = stablehlo.multiply %v897, %v901 : tensor<64x256x7x7xf32>
    %v903 = stablehlo.broadcast_in_dim %u12qg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v904 = stablehlo.broadcast_in_dim %u12qbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v905 = stablehlo.multiply %v902, %v903 : tensor<64x256x7x7xf32>
    %v906 = stablehlo.add %v905, %v904 : tensor<64x256x7x7xf32>
    %v907 = stablehlo.reshape %v906 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v908 = stablehlo.reshape %v907 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v909 = stablehlo.convolution(%v908, %u12eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<64x1024x7x7xf32>
    %v910 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v911 = stablehlo.add %v909, %v910 : tensor<64x1024x7x7xf32>
    %v912 = stablehlo.reshape %v911 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v913 = stablehlo.reshape %v912 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v914 = stablehlo.broadcast_in_dim %u12enmu, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v915 = stablehlo.subtract %v913, %v914 : tensor<64x1024x7x7xf32>
    %v916 = stablehlo.broadcast_in_dim %u12envar, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v917 = stablehlo.constant dense<1.0e-5> : tensor<64x1024x7x7xf32>
    %v918 = stablehlo.add %v916, %v917 : tensor<64x1024x7x7xf32>
    %v919 = stablehlo.rsqrt %v918 : tensor<64x1024x7x7xf32>
    %v920 = stablehlo.multiply %v915, %v919 : tensor<64x1024x7x7xf32>
    %v921 = stablehlo.broadcast_in_dim %u12eg, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v922 = stablehlo.broadcast_in_dim %u12ebt, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v923 = stablehlo.multiply %v920, %v921 : tensor<64x1024x7x7xf32>
    %v924 = stablehlo.add %v923, %v922 : tensor<64x1024x7x7xf32>
    %v925 = stablehlo.reshape %v924 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v926 = stablehlo.reshape %v925 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v927 = stablehlo.constant dense<0.0> : tensor<64x1024x7x7xf32>
    %v928 = stablehlo.maximum %v926, %v927 : tensor<64x1024x7x7xf32>
    %v929 = stablehlo.reshape %v928 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v930 = stablehlo.reshape %v929 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v931 = stablehlo.convolution(%v930, %u12dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<64x1024x7x7xf32>, tensor<1024x1x5x5xf32>) -> tensor<64x1024x7x7xf32>
    %v932 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v933 = stablehlo.add %v931, %v932 : tensor<64x1024x7x7xf32>
    %v934 = stablehlo.reshape %v933 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v935 = stablehlo.reshape %v934 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v936 = stablehlo.broadcast_in_dim %u12dnmu, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v937 = stablehlo.subtract %v935, %v936 : tensor<64x1024x7x7xf32>
    %v938 = stablehlo.broadcast_in_dim %u12dnvar, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v939 = stablehlo.constant dense<1.0e-5> : tensor<64x1024x7x7xf32>
    %v940 = stablehlo.add %v938, %v939 : tensor<64x1024x7x7xf32>
    %v941 = stablehlo.rsqrt %v940 : tensor<64x1024x7x7xf32>
    %v942 = stablehlo.multiply %v937, %v941 : tensor<64x1024x7x7xf32>
    %v943 = stablehlo.broadcast_in_dim %u12dg, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v944 = stablehlo.broadcast_in_dim %u12dbt, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v945 = stablehlo.multiply %v942, %v943 : tensor<64x1024x7x7xf32>
    %v946 = stablehlo.add %v945, %v944 : tensor<64x1024x7x7xf32>
    %v947 = stablehlo.reshape %v946 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v948 = stablehlo.reshape %v947 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v949 = stablehlo.constant dense<0.0> : tensor<64x1024x7x7xf32>
    %v950 = stablehlo.maximum %v948, %v949 : tensor<64x1024x7x7xf32>
    %v951 = stablehlo.reshape %v950 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v952 = stablehlo.reshape %v951 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v953 = stablehlo.convolution(%v952, %u12pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<64x256x7x7xf32>
    %v954 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v955 = stablehlo.add %v953, %v954 : tensor<64x256x7x7xf32>
    %v956 = stablehlo.reshape %v955 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v957 = stablehlo.reshape %v956 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v958 = stablehlo.broadcast_in_dim %u12pnmu, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v959 = stablehlo.subtract %v957, %v958 : tensor<64x256x7x7xf32>
    %v960 = stablehlo.broadcast_in_dim %u12pnvar, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v961 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v962 = stablehlo.add %v960, %v961 : tensor<64x256x7x7xf32>
    %v963 = stablehlo.rsqrt %v962 : tensor<64x256x7x7xf32>
    %v964 = stablehlo.multiply %v959, %v963 : tensor<64x256x7x7xf32>
    %v965 = stablehlo.broadcast_in_dim %u12pg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v966 = stablehlo.broadcast_in_dim %u12pbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v967 = stablehlo.multiply %v964, %v965 : tensor<64x256x7x7xf32>
    %v968 = stablehlo.add %v967, %v966 : tensor<64x256x7x7xf32>
    %v969 = stablehlo.reshape %v968 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v970 = stablehlo.reshape %v969 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v971 = stablehlo.reshape %v889 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v972 = stablehlo.add %v970, %v971 : tensor<64x256x7x7xf32>
    %v973 = stablehlo.reshape %v972 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v974 = stablehlo.reshape %v973 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v975 = stablehlo.convolution(%v974, %u13qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<64x256x7x7xf32>, tensor<256x1x3x3xf32>) -> tensor<64x256x7x7xf32>
    %v976 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v977 = stablehlo.add %v975, %v976 : tensor<64x256x7x7xf32>
    %v978 = stablehlo.reshape %v977 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v979 = stablehlo.reshape %v978 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v980 = stablehlo.broadcast_in_dim %u13qnmu, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v981 = stablehlo.subtract %v979, %v980 : tensor<64x256x7x7xf32>
    %v982 = stablehlo.broadcast_in_dim %u13qnvar, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v983 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v984 = stablehlo.add %v982, %v983 : tensor<64x256x7x7xf32>
    %v985 = stablehlo.rsqrt %v984 : tensor<64x256x7x7xf32>
    %v986 = stablehlo.multiply %v981, %v985 : tensor<64x256x7x7xf32>
    %v987 = stablehlo.broadcast_in_dim %u13qg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v988 = stablehlo.broadcast_in_dim %u13qbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v989 = stablehlo.multiply %v986, %v987 : tensor<64x256x7x7xf32>
    %v990 = stablehlo.add %v989, %v988 : tensor<64x256x7x7xf32>
    %v991 = stablehlo.reshape %v990 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v992 = stablehlo.reshape %v991 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v993 = stablehlo.convolution(%v992, %u13eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<64x1024x7x7xf32>
    %v994 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v995 = stablehlo.add %v993, %v994 : tensor<64x1024x7x7xf32>
    %v996 = stablehlo.reshape %v995 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v997 = stablehlo.reshape %v996 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v998 = stablehlo.broadcast_in_dim %u13enmu, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v999 = stablehlo.subtract %v997, %v998 : tensor<64x1024x7x7xf32>
    %v1000 = stablehlo.broadcast_in_dim %u13envar, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1001 = stablehlo.constant dense<1.0e-5> : tensor<64x1024x7x7xf32>
    %v1002 = stablehlo.add %v1000, %v1001 : tensor<64x1024x7x7xf32>
    %v1003 = stablehlo.rsqrt %v1002 : tensor<64x1024x7x7xf32>
    %v1004 = stablehlo.multiply %v999, %v1003 : tensor<64x1024x7x7xf32>
    %v1005 = stablehlo.broadcast_in_dim %u13eg, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1006 = stablehlo.broadcast_in_dim %u13ebt, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1007 = stablehlo.multiply %v1004, %v1005 : tensor<64x1024x7x7xf32>
    %v1008 = stablehlo.add %v1007, %v1006 : tensor<64x1024x7x7xf32>
    %v1009 = stablehlo.reshape %v1008 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1010 = stablehlo.reshape %v1009 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1011 = stablehlo.constant dense<0.0> : tensor<64x1024x7x7xf32>
    %v1012 = stablehlo.maximum %v1010, %v1011 : tensor<64x1024x7x7xf32>
    %v1013 = stablehlo.reshape %v1012 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1014 = stablehlo.reshape %v1013 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1015 = stablehlo.convolution(%v1014, %u13dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<64x1024x7x7xf32>, tensor<1024x1x5x5xf32>) -> tensor<64x1024x7x7xf32>
    %v1016 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1017 = stablehlo.add %v1015, %v1016 : tensor<64x1024x7x7xf32>
    %v1018 = stablehlo.reshape %v1017 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1019 = stablehlo.reshape %v1018 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1020 = stablehlo.broadcast_in_dim %u13dnmu, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1021 = stablehlo.subtract %v1019, %v1020 : tensor<64x1024x7x7xf32>
    %v1022 = stablehlo.broadcast_in_dim %u13dnvar, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1023 = stablehlo.constant dense<1.0e-5> : tensor<64x1024x7x7xf32>
    %v1024 = stablehlo.add %v1022, %v1023 : tensor<64x1024x7x7xf32>
    %v1025 = stablehlo.rsqrt %v1024 : tensor<64x1024x7x7xf32>
    %v1026 = stablehlo.multiply %v1021, %v1025 : tensor<64x1024x7x7xf32>
    %v1027 = stablehlo.broadcast_in_dim %u13dg, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1028 = stablehlo.broadcast_in_dim %u13dbt, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1029 = stablehlo.multiply %v1026, %v1027 : tensor<64x1024x7x7xf32>
    %v1030 = stablehlo.add %v1029, %v1028 : tensor<64x1024x7x7xf32>
    %v1031 = stablehlo.reshape %v1030 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1032 = stablehlo.reshape %v1031 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1033 = stablehlo.constant dense<0.0> : tensor<64x1024x7x7xf32>
    %v1034 = stablehlo.maximum %v1032, %v1033 : tensor<64x1024x7x7xf32>
    %v1035 = stablehlo.reshape %v1034 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1036 = stablehlo.reshape %v1035 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1037 = stablehlo.convolution(%v1036, %u13pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<64x256x7x7xf32>
    %v1038 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1039 = stablehlo.add %v1037, %v1038 : tensor<64x256x7x7xf32>
    %v1040 = stablehlo.reshape %v1039 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1041 = stablehlo.reshape %v1040 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1042 = stablehlo.broadcast_in_dim %u13pnmu, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1043 = stablehlo.subtract %v1041, %v1042 : tensor<64x256x7x7xf32>
    %v1044 = stablehlo.broadcast_in_dim %u13pnvar, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1045 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v1046 = stablehlo.add %v1044, %v1045 : tensor<64x256x7x7xf32>
    %v1047 = stablehlo.rsqrt %v1046 : tensor<64x256x7x7xf32>
    %v1048 = stablehlo.multiply %v1043, %v1047 : tensor<64x256x7x7xf32>
    %v1049 = stablehlo.broadcast_in_dim %u13pg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1050 = stablehlo.broadcast_in_dim %u13pbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1051 = stablehlo.multiply %v1048, %v1049 : tensor<64x256x7x7xf32>
    %v1052 = stablehlo.add %v1051, %v1050 : tensor<64x256x7x7xf32>
    %v1053 = stablehlo.reshape %v1052 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1054 = stablehlo.reshape %v1053 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1055 = stablehlo.reshape %v973 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1056 = stablehlo.add %v1054, %v1055 : tensor<64x256x7x7xf32>
    %v1057 = stablehlo.reshape %v1056 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1058 = stablehlo.reshape %v1057 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1059 = stablehlo.convolution(%v1058, %u14qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<64x256x7x7xf32>, tensor<256x1x3x3xf32>) -> tensor<64x256x7x7xf32>
    %v1060 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1061 = stablehlo.add %v1059, %v1060 : tensor<64x256x7x7xf32>
    %v1062 = stablehlo.reshape %v1061 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1063 = stablehlo.reshape %v1062 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1064 = stablehlo.broadcast_in_dim %u14qnmu, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1065 = stablehlo.subtract %v1063, %v1064 : tensor<64x256x7x7xf32>
    %v1066 = stablehlo.broadcast_in_dim %u14qnvar, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1067 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v1068 = stablehlo.add %v1066, %v1067 : tensor<64x256x7x7xf32>
    %v1069 = stablehlo.rsqrt %v1068 : tensor<64x256x7x7xf32>
    %v1070 = stablehlo.multiply %v1065, %v1069 : tensor<64x256x7x7xf32>
    %v1071 = stablehlo.broadcast_in_dim %u14qg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1072 = stablehlo.broadcast_in_dim %u14qbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1073 = stablehlo.multiply %v1070, %v1071 : tensor<64x256x7x7xf32>
    %v1074 = stablehlo.add %v1073, %v1072 : tensor<64x256x7x7xf32>
    %v1075 = stablehlo.reshape %v1074 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1076 = stablehlo.reshape %v1075 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1077 = stablehlo.convolution(%v1076, %u14eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<64x1024x7x7xf32>
    %v1078 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1079 = stablehlo.add %v1077, %v1078 : tensor<64x1024x7x7xf32>
    %v1080 = stablehlo.reshape %v1079 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1081 = stablehlo.reshape %v1080 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1082 = stablehlo.broadcast_in_dim %u14enmu, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1083 = stablehlo.subtract %v1081, %v1082 : tensor<64x1024x7x7xf32>
    %v1084 = stablehlo.broadcast_in_dim %u14envar, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1085 = stablehlo.constant dense<1.0e-5> : tensor<64x1024x7x7xf32>
    %v1086 = stablehlo.add %v1084, %v1085 : tensor<64x1024x7x7xf32>
    %v1087 = stablehlo.rsqrt %v1086 : tensor<64x1024x7x7xf32>
    %v1088 = stablehlo.multiply %v1083, %v1087 : tensor<64x1024x7x7xf32>
    %v1089 = stablehlo.broadcast_in_dim %u14eg, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1090 = stablehlo.broadcast_in_dim %u14ebt, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1091 = stablehlo.multiply %v1088, %v1089 : tensor<64x1024x7x7xf32>
    %v1092 = stablehlo.add %v1091, %v1090 : tensor<64x1024x7x7xf32>
    %v1093 = stablehlo.reshape %v1092 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1094 = stablehlo.reshape %v1093 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1095 = stablehlo.constant dense<0.0> : tensor<64x1024x7x7xf32>
    %v1096 = stablehlo.maximum %v1094, %v1095 : tensor<64x1024x7x7xf32>
    %v1097 = stablehlo.reshape %v1096 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1098 = stablehlo.reshape %v1097 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1099 = stablehlo.convolution(%v1098, %u14dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<64x1024x7x7xf32>, tensor<1024x1x5x5xf32>) -> tensor<64x1024x7x7xf32>
    %v1100 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1101 = stablehlo.add %v1099, %v1100 : tensor<64x1024x7x7xf32>
    %v1102 = stablehlo.reshape %v1101 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1103 = stablehlo.reshape %v1102 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1104 = stablehlo.broadcast_in_dim %u14dnmu, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1105 = stablehlo.subtract %v1103, %v1104 : tensor<64x1024x7x7xf32>
    %v1106 = stablehlo.broadcast_in_dim %u14dnvar, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1107 = stablehlo.constant dense<1.0e-5> : tensor<64x1024x7x7xf32>
    %v1108 = stablehlo.add %v1106, %v1107 : tensor<64x1024x7x7xf32>
    %v1109 = stablehlo.rsqrt %v1108 : tensor<64x1024x7x7xf32>
    %v1110 = stablehlo.multiply %v1105, %v1109 : tensor<64x1024x7x7xf32>
    %v1111 = stablehlo.broadcast_in_dim %u14dg, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1112 = stablehlo.broadcast_in_dim %u14dbt, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1113 = stablehlo.multiply %v1110, %v1111 : tensor<64x1024x7x7xf32>
    %v1114 = stablehlo.add %v1113, %v1112 : tensor<64x1024x7x7xf32>
    %v1115 = stablehlo.reshape %v1114 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1116 = stablehlo.reshape %v1115 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1117 = stablehlo.constant dense<0.0> : tensor<64x1024x7x7xf32>
    %v1118 = stablehlo.maximum %v1116, %v1117 : tensor<64x1024x7x7xf32>
    %v1119 = stablehlo.reshape %v1118 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1120 = stablehlo.reshape %v1119 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1121 = stablehlo.convolution(%v1120, %u14pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<64x256x7x7xf32>
    %v1122 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1123 = stablehlo.add %v1121, %v1122 : tensor<64x256x7x7xf32>
    %v1124 = stablehlo.reshape %v1123 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1125 = stablehlo.reshape %v1124 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1126 = stablehlo.broadcast_in_dim %u14pnmu, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1127 = stablehlo.subtract %v1125, %v1126 : tensor<64x256x7x7xf32>
    %v1128 = stablehlo.broadcast_in_dim %u14pnvar, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1129 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v1130 = stablehlo.add %v1128, %v1129 : tensor<64x256x7x7xf32>
    %v1131 = stablehlo.rsqrt %v1130 : tensor<64x256x7x7xf32>
    %v1132 = stablehlo.multiply %v1127, %v1131 : tensor<64x256x7x7xf32>
    %v1133 = stablehlo.broadcast_in_dim %u14pg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1134 = stablehlo.broadcast_in_dim %u14pbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1135 = stablehlo.multiply %v1132, %v1133 : tensor<64x256x7x7xf32>
    %v1136 = stablehlo.add %v1135, %v1134 : tensor<64x256x7x7xf32>
    %v1137 = stablehlo.reshape %v1136 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1138 = stablehlo.reshape %v1137 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1139 = stablehlo.reshape %v1057 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1140 = stablehlo.add %v1138, %v1139 : tensor<64x256x7x7xf32>
    %v1141 = stablehlo.reshape %v1140 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1142 = stablehlo.reshape %v1141 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1143 = stablehlo.convolution(%v1142, %u15eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<64x1024x7x7xf32>
    %v1144 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1145 = stablehlo.add %v1143, %v1144 : tensor<64x1024x7x7xf32>
    %v1146 = stablehlo.reshape %v1145 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1147 = stablehlo.reshape %v1146 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1148 = stablehlo.broadcast_in_dim %u15enmu, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1149 = stablehlo.subtract %v1147, %v1148 : tensor<64x1024x7x7xf32>
    %v1150 = stablehlo.broadcast_in_dim %u15envar, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1151 = stablehlo.constant dense<1.0e-5> : tensor<64x1024x7x7xf32>
    %v1152 = stablehlo.add %v1150, %v1151 : tensor<64x1024x7x7xf32>
    %v1153 = stablehlo.rsqrt %v1152 : tensor<64x1024x7x7xf32>
    %v1154 = stablehlo.multiply %v1149, %v1153 : tensor<64x1024x7x7xf32>
    %v1155 = stablehlo.broadcast_in_dim %u15eg, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1156 = stablehlo.broadcast_in_dim %u15ebt, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1157 = stablehlo.multiply %v1154, %v1155 : tensor<64x1024x7x7xf32>
    %v1158 = stablehlo.add %v1157, %v1156 : tensor<64x1024x7x7xf32>
    %v1159 = stablehlo.reshape %v1158 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1160 = stablehlo.reshape %v1159 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1161 = stablehlo.constant dense<0.0> : tensor<64x1024x7x7xf32>
    %v1162 = stablehlo.maximum %v1160, %v1161 : tensor<64x1024x7x7xf32>
    %v1163 = stablehlo.reshape %v1162 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1164 = stablehlo.reshape %v1163 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1165 = stablehlo.convolution(%v1164, %u15pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<64x256x7x7xf32>
    %v1166 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1167 = stablehlo.add %v1165, %v1166 : tensor<64x256x7x7xf32>
    %v1168 = stablehlo.reshape %v1167 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1169 = stablehlo.reshape %v1168 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1170 = stablehlo.broadcast_in_dim %u15pnmu, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1171 = stablehlo.subtract %v1169, %v1170 : tensor<64x256x7x7xf32>
    %v1172 = stablehlo.broadcast_in_dim %u15pnvar, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1173 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v1174 = stablehlo.add %v1172, %v1173 : tensor<64x256x7x7xf32>
    %v1175 = stablehlo.rsqrt %v1174 : tensor<64x256x7x7xf32>
    %v1176 = stablehlo.multiply %v1171, %v1175 : tensor<64x256x7x7xf32>
    %v1177 = stablehlo.broadcast_in_dim %u15pg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1178 = stablehlo.broadcast_in_dim %u15pbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1179 = stablehlo.multiply %v1176, %v1177 : tensor<64x256x7x7xf32>
    %v1180 = stablehlo.add %v1179, %v1178 : tensor<64x256x7x7xf32>
    %v1181 = stablehlo.reshape %v1180 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1182 = stablehlo.reshape %v1181 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1183 = stablehlo.reshape %v1141 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1184 = stablehlo.add %v1182, %v1183 : tensor<64x256x7x7xf32>
    %v1185 = stablehlo.reshape %v1184 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1186 = stablehlo.reshape %v1185 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1187 = stablehlo.convolution(%v1186, %u16qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<64x256x7x7xf32>, tensor<256x1x3x3xf32>) -> tensor<64x256x7x7xf32>
    %v1188 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1189 = stablehlo.add %v1187, %v1188 : tensor<64x256x7x7xf32>
    %v1190 = stablehlo.reshape %v1189 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1191 = stablehlo.reshape %v1190 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1192 = stablehlo.broadcast_in_dim %u16qnmu, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1193 = stablehlo.subtract %v1191, %v1192 : tensor<64x256x7x7xf32>
    %v1194 = stablehlo.broadcast_in_dim %u16qnvar, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1195 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v1196 = stablehlo.add %v1194, %v1195 : tensor<64x256x7x7xf32>
    %v1197 = stablehlo.rsqrt %v1196 : tensor<64x256x7x7xf32>
    %v1198 = stablehlo.multiply %v1193, %v1197 : tensor<64x256x7x7xf32>
    %v1199 = stablehlo.broadcast_in_dim %u16qg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1200 = stablehlo.broadcast_in_dim %u16qbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1201 = stablehlo.multiply %v1198, %v1199 : tensor<64x256x7x7xf32>
    %v1202 = stablehlo.add %v1201, %v1200 : tensor<64x256x7x7xf32>
    %v1203 = stablehlo.reshape %v1202 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1204 = stablehlo.reshape %v1203 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1205 = stablehlo.convolution(%v1204, %u16eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<64x1024x7x7xf32>
    %v1206 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1207 = stablehlo.add %v1205, %v1206 : tensor<64x1024x7x7xf32>
    %v1208 = stablehlo.reshape %v1207 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1209 = stablehlo.reshape %v1208 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1210 = stablehlo.broadcast_in_dim %u16enmu, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1211 = stablehlo.subtract %v1209, %v1210 : tensor<64x1024x7x7xf32>
    %v1212 = stablehlo.broadcast_in_dim %u16envar, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1213 = stablehlo.constant dense<1.0e-5> : tensor<64x1024x7x7xf32>
    %v1214 = stablehlo.add %v1212, %v1213 : tensor<64x1024x7x7xf32>
    %v1215 = stablehlo.rsqrt %v1214 : tensor<64x1024x7x7xf32>
    %v1216 = stablehlo.multiply %v1211, %v1215 : tensor<64x1024x7x7xf32>
    %v1217 = stablehlo.broadcast_in_dim %u16eg, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1218 = stablehlo.broadcast_in_dim %u16ebt, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1219 = stablehlo.multiply %v1216, %v1217 : tensor<64x1024x7x7xf32>
    %v1220 = stablehlo.add %v1219, %v1218 : tensor<64x1024x7x7xf32>
    %v1221 = stablehlo.reshape %v1220 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1222 = stablehlo.reshape %v1221 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1223 = stablehlo.constant dense<0.0> : tensor<64x1024x7x7xf32>
    %v1224 = stablehlo.maximum %v1222, %v1223 : tensor<64x1024x7x7xf32>
    %v1225 = stablehlo.reshape %v1224 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1226 = stablehlo.reshape %v1225 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1227 = stablehlo.convolution(%v1226, %u16pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<64x256x7x7xf32>
    %v1228 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1229 = stablehlo.add %v1227, %v1228 : tensor<64x256x7x7xf32>
    %v1230 = stablehlo.reshape %v1229 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1231 = stablehlo.reshape %v1230 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1232 = stablehlo.broadcast_in_dim %u16pnmu, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1233 = stablehlo.subtract %v1231, %v1232 : tensor<64x256x7x7xf32>
    %v1234 = stablehlo.broadcast_in_dim %u16pnvar, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1235 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v1236 = stablehlo.add %v1234, %v1235 : tensor<64x256x7x7xf32>
    %v1237 = stablehlo.rsqrt %v1236 : tensor<64x256x7x7xf32>
    %v1238 = stablehlo.multiply %v1233, %v1237 : tensor<64x256x7x7xf32>
    %v1239 = stablehlo.broadcast_in_dim %u16pg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1240 = stablehlo.broadcast_in_dim %u16pbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1241 = stablehlo.multiply %v1238, %v1239 : tensor<64x256x7x7xf32>
    %v1242 = stablehlo.add %v1241, %v1240 : tensor<64x256x7x7xf32>
    %v1243 = stablehlo.reshape %v1242 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1244 = stablehlo.reshape %v1243 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1245 = stablehlo.reshape %v1185 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1246 = stablehlo.add %v1244, %v1245 : tensor<64x256x7x7xf32>
    %v1247 = stablehlo.reshape %v1246 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1248 = stablehlo.reshape %v1247 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1249 = stablehlo.convolution(%v1248, %u17qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<64x256x7x7xf32>, tensor<256x1x3x3xf32>) -> tensor<64x256x7x7xf32>
    %v1250 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1251 = stablehlo.add %v1249, %v1250 : tensor<64x256x7x7xf32>
    %v1252 = stablehlo.reshape %v1251 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1253 = stablehlo.reshape %v1252 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1254 = stablehlo.broadcast_in_dim %u17qnmu, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1255 = stablehlo.subtract %v1253, %v1254 : tensor<64x256x7x7xf32>
    %v1256 = stablehlo.broadcast_in_dim %u17qnvar, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1257 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v1258 = stablehlo.add %v1256, %v1257 : tensor<64x256x7x7xf32>
    %v1259 = stablehlo.rsqrt %v1258 : tensor<64x256x7x7xf32>
    %v1260 = stablehlo.multiply %v1255, %v1259 : tensor<64x256x7x7xf32>
    %v1261 = stablehlo.broadcast_in_dim %u17qg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1262 = stablehlo.broadcast_in_dim %u17qbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1263 = stablehlo.multiply %v1260, %v1261 : tensor<64x256x7x7xf32>
    %v1264 = stablehlo.add %v1263, %v1262 : tensor<64x256x7x7xf32>
    %v1265 = stablehlo.reshape %v1264 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1266 = stablehlo.reshape %v1265 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1267 = stablehlo.convolution(%v1266, %u17eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x7x7xf32>, tensor<512x256x1x1xf32>) -> tensor<64x512x7x7xf32>
    %v1268 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1269 = stablehlo.add %v1267, %v1268 : tensor<64x512x7x7xf32>
    %v1270 = stablehlo.reshape %v1269 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1271 = stablehlo.reshape %v1270 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1272 = stablehlo.broadcast_in_dim %u17enmu, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1273 = stablehlo.subtract %v1271, %v1272 : tensor<64x512x7x7xf32>
    %v1274 = stablehlo.broadcast_in_dim %u17envar, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1275 = stablehlo.constant dense<1.0e-5> : tensor<64x512x7x7xf32>
    %v1276 = stablehlo.add %v1274, %v1275 : tensor<64x512x7x7xf32>
    %v1277 = stablehlo.rsqrt %v1276 : tensor<64x512x7x7xf32>
    %v1278 = stablehlo.multiply %v1273, %v1277 : tensor<64x512x7x7xf32>
    %v1279 = stablehlo.broadcast_in_dim %u17eg, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1280 = stablehlo.broadcast_in_dim %u17ebt, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1281 = stablehlo.multiply %v1278, %v1279 : tensor<64x512x7x7xf32>
    %v1282 = stablehlo.add %v1281, %v1280 : tensor<64x512x7x7xf32>
    %v1283 = stablehlo.reshape %v1282 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1284 = stablehlo.reshape %v1283 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1285 = stablehlo.constant dense<0.0> : tensor<64x512x7x7xf32>
    %v1286 = stablehlo.maximum %v1284, %v1285 : tensor<64x512x7x7xf32>
    %v1287 = stablehlo.reshape %v1286 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1288 = stablehlo.reshape %v1287 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1289 = stablehlo.convolution(%v1288, %u17dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 512 : i64} : (tensor<64x512x7x7xf32>, tensor<512x1x5x5xf32>) -> tensor<64x512x7x7xf32>
    %v1290 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1291 = stablehlo.add %v1289, %v1290 : tensor<64x512x7x7xf32>
    %v1292 = stablehlo.reshape %v1291 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1293 = stablehlo.reshape %v1292 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1294 = stablehlo.broadcast_in_dim %u17dnmu, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1295 = stablehlo.subtract %v1293, %v1294 : tensor<64x512x7x7xf32>
    %v1296 = stablehlo.broadcast_in_dim %u17dnvar, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1297 = stablehlo.constant dense<1.0e-5> : tensor<64x512x7x7xf32>
    %v1298 = stablehlo.add %v1296, %v1297 : tensor<64x512x7x7xf32>
    %v1299 = stablehlo.rsqrt %v1298 : tensor<64x512x7x7xf32>
    %v1300 = stablehlo.multiply %v1295, %v1299 : tensor<64x512x7x7xf32>
    %v1301 = stablehlo.broadcast_in_dim %u17dg, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1302 = stablehlo.broadcast_in_dim %u17dbt, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1303 = stablehlo.multiply %v1300, %v1301 : tensor<64x512x7x7xf32>
    %v1304 = stablehlo.add %v1303, %v1302 : tensor<64x512x7x7xf32>
    %v1305 = stablehlo.reshape %v1304 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1306 = stablehlo.reshape %v1305 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1307 = stablehlo.constant dense<0.0> : tensor<64x512x7x7xf32>
    %v1308 = stablehlo.maximum %v1306, %v1307 : tensor<64x512x7x7xf32>
    %v1309 = stablehlo.reshape %v1308 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1310 = stablehlo.reshape %v1309 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1311 = stablehlo.convolution(%v1310, %u17pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xf32>, tensor<256x512x1x1xf32>) -> tensor<64x256x7x7xf32>
    %v1312 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1313 = stablehlo.add %v1311, %v1312 : tensor<64x256x7x7xf32>
    %v1314 = stablehlo.reshape %v1313 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1315 = stablehlo.reshape %v1314 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1316 = stablehlo.broadcast_in_dim %u17pnmu, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1317 = stablehlo.subtract %v1315, %v1316 : tensor<64x256x7x7xf32>
    %v1318 = stablehlo.broadcast_in_dim %u17pnvar, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1319 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v1320 = stablehlo.add %v1318, %v1319 : tensor<64x256x7x7xf32>
    %v1321 = stablehlo.rsqrt %v1320 : tensor<64x256x7x7xf32>
    %v1322 = stablehlo.multiply %v1317, %v1321 : tensor<64x256x7x7xf32>
    %v1323 = stablehlo.broadcast_in_dim %u17pg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1324 = stablehlo.broadcast_in_dim %u17pbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1325 = stablehlo.multiply %v1322, %v1323 : tensor<64x256x7x7xf32>
    %v1326 = stablehlo.add %v1325, %v1324 : tensor<64x256x7x7xf32>
    %v1327 = stablehlo.reshape %v1326 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1328 = stablehlo.reshape %v1327 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1329 = stablehlo.reshape %v1247 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1330 = stablehlo.add %v1328, %v1329 : tensor<64x256x7x7xf32>
    %v1331 = stablehlo.reshape %v1330 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1332 = stablehlo.reshape %v1331 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1333 = stablehlo.convolution(%v1332, %u18qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<64x256x7x7xf32>, tensor<256x1x5x5xf32>) -> tensor<64x256x7x7xf32>
    %v1334 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1335 = stablehlo.add %v1333, %v1334 : tensor<64x256x7x7xf32>
    %v1336 = stablehlo.reshape %v1335 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1337 = stablehlo.reshape %v1336 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1338 = stablehlo.broadcast_in_dim %u18qnmu, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1339 = stablehlo.subtract %v1337, %v1338 : tensor<64x256x7x7xf32>
    %v1340 = stablehlo.broadcast_in_dim %u18qnvar, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1341 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v1342 = stablehlo.add %v1340, %v1341 : tensor<64x256x7x7xf32>
    %v1343 = stablehlo.rsqrt %v1342 : tensor<64x256x7x7xf32>
    %v1344 = stablehlo.multiply %v1339, %v1343 : tensor<64x256x7x7xf32>
    %v1345 = stablehlo.broadcast_in_dim %u18qg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1346 = stablehlo.broadcast_in_dim %u18qbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1347 = stablehlo.multiply %v1344, %v1345 : tensor<64x256x7x7xf32>
    %v1348 = stablehlo.add %v1347, %v1346 : tensor<64x256x7x7xf32>
    %v1349 = stablehlo.reshape %v1348 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1350 = stablehlo.reshape %v1349 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1351 = stablehlo.convolution(%v1350, %u18eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<64x1024x7x7xf32>
    %v1352 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1353 = stablehlo.add %v1351, %v1352 : tensor<64x1024x7x7xf32>
    %v1354 = stablehlo.reshape %v1353 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1355 = stablehlo.reshape %v1354 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1356 = stablehlo.broadcast_in_dim %u18enmu, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1357 = stablehlo.subtract %v1355, %v1356 : tensor<64x1024x7x7xf32>
    %v1358 = stablehlo.broadcast_in_dim %u18envar, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1359 = stablehlo.constant dense<1.0e-5> : tensor<64x1024x7x7xf32>
    %v1360 = stablehlo.add %v1358, %v1359 : tensor<64x1024x7x7xf32>
    %v1361 = stablehlo.rsqrt %v1360 : tensor<64x1024x7x7xf32>
    %v1362 = stablehlo.multiply %v1357, %v1361 : tensor<64x1024x7x7xf32>
    %v1363 = stablehlo.broadcast_in_dim %u18eg, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1364 = stablehlo.broadcast_in_dim %u18ebt, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1365 = stablehlo.multiply %v1362, %v1363 : tensor<64x1024x7x7xf32>
    %v1366 = stablehlo.add %v1365, %v1364 : tensor<64x1024x7x7xf32>
    %v1367 = stablehlo.reshape %v1366 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1368 = stablehlo.reshape %v1367 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1369 = stablehlo.constant dense<0.0> : tensor<64x1024x7x7xf32>
    %v1370 = stablehlo.maximum %v1368, %v1369 : tensor<64x1024x7x7xf32>
    %v1371 = stablehlo.reshape %v1370 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1372 = stablehlo.reshape %v1371 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1373 = stablehlo.convolution(%v1372, %u18dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1024 : i64} : (tensor<64x1024x7x7xf32>, tensor<1024x1x5x5xf32>) -> tensor<64x1024x7x7xf32>
    %v1374 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1375 = stablehlo.add %v1373, %v1374 : tensor<64x1024x7x7xf32>
    %v1376 = stablehlo.reshape %v1375 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1377 = stablehlo.reshape %v1376 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1378 = stablehlo.broadcast_in_dim %u18dnmu, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1379 = stablehlo.subtract %v1377, %v1378 : tensor<64x1024x7x7xf32>
    %v1380 = stablehlo.broadcast_in_dim %u18dnvar, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1381 = stablehlo.constant dense<1.0e-5> : tensor<64x1024x7x7xf32>
    %v1382 = stablehlo.add %v1380, %v1381 : tensor<64x1024x7x7xf32>
    %v1383 = stablehlo.rsqrt %v1382 : tensor<64x1024x7x7xf32>
    %v1384 = stablehlo.multiply %v1379, %v1383 : tensor<64x1024x7x7xf32>
    %v1385 = stablehlo.broadcast_in_dim %u18dg, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1386 = stablehlo.broadcast_in_dim %u18dbt, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1387 = stablehlo.multiply %v1384, %v1385 : tensor<64x1024x7x7xf32>
    %v1388 = stablehlo.add %v1387, %v1386 : tensor<64x1024x7x7xf32>
    %v1389 = stablehlo.reshape %v1388 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1390 = stablehlo.reshape %v1389 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1391 = stablehlo.constant dense<0.0> : tensor<64x1024x7x7xf32>
    %v1392 = stablehlo.maximum %v1390, %v1391 : tensor<64x1024x7x7xf32>
    %v1393 = stablehlo.reshape %v1392 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1394 = stablehlo.reshape %v1393 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1395 = stablehlo.convolution(%v1394, %u18pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<64x256x7x7xf32>
    %v1396 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1397 = stablehlo.add %v1395, %v1396 : tensor<64x256x7x7xf32>
    %v1398 = stablehlo.reshape %v1397 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1399 = stablehlo.reshape %v1398 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1400 = stablehlo.broadcast_in_dim %u18pnmu, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1401 = stablehlo.subtract %v1399, %v1400 : tensor<64x256x7x7xf32>
    %v1402 = stablehlo.broadcast_in_dim %u18pnvar, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1403 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v1404 = stablehlo.add %v1402, %v1403 : tensor<64x256x7x7xf32>
    %v1405 = stablehlo.rsqrt %v1404 : tensor<64x256x7x7xf32>
    %v1406 = stablehlo.multiply %v1401, %v1405 : tensor<64x256x7x7xf32>
    %v1407 = stablehlo.broadcast_in_dim %u18pg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1408 = stablehlo.broadcast_in_dim %u18pbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1409 = stablehlo.multiply %v1406, %v1407 : tensor<64x256x7x7xf32>
    %v1410 = stablehlo.add %v1409, %v1408 : tensor<64x256x7x7xf32>
    %v1411 = stablehlo.reshape %v1410 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1412 = stablehlo.reshape %v1411 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1413 = stablehlo.reshape %v1331 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1414 = stablehlo.add %v1412, %v1413 : tensor<64x256x7x7xf32>
    %v1415 = stablehlo.reshape %v1414 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1416 = stablehlo.reshape %v1415 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1417 = stablehlo.convolution(%v1416, %u19eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<64x1024x7x7xf32>
    %v1418 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1419 = stablehlo.add %v1417, %v1418 : tensor<64x1024x7x7xf32>
    %v1420 = stablehlo.reshape %v1419 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1421 = stablehlo.reshape %v1420 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1422 = stablehlo.broadcast_in_dim %u19enmu, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1423 = stablehlo.subtract %v1421, %v1422 : tensor<64x1024x7x7xf32>
    %v1424 = stablehlo.broadcast_in_dim %u19envar, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1425 = stablehlo.constant dense<1.0e-5> : tensor<64x1024x7x7xf32>
    %v1426 = stablehlo.add %v1424, %v1425 : tensor<64x1024x7x7xf32>
    %v1427 = stablehlo.rsqrt %v1426 : tensor<64x1024x7x7xf32>
    %v1428 = stablehlo.multiply %v1423, %v1427 : tensor<64x1024x7x7xf32>
    %v1429 = stablehlo.broadcast_in_dim %u19eg, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1430 = stablehlo.broadcast_in_dim %u19ebt, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1431 = stablehlo.multiply %v1428, %v1429 : tensor<64x1024x7x7xf32>
    %v1432 = stablehlo.add %v1431, %v1430 : tensor<64x1024x7x7xf32>
    %v1433 = stablehlo.reshape %v1432 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1434 = stablehlo.reshape %v1433 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1435 = stablehlo.constant dense<0.0> : tensor<64x1024x7x7xf32>
    %v1436 = stablehlo.maximum %v1434, %v1435 : tensor<64x1024x7x7xf32>
    %v1437 = stablehlo.reshape %v1436 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1438 = stablehlo.reshape %v1437 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1439 = stablehlo.convolution(%v1438, %u19pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<64x256x7x7xf32>
    %v1440 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1441 = stablehlo.add %v1439, %v1440 : tensor<64x256x7x7xf32>
    %v1442 = stablehlo.reshape %v1441 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1443 = stablehlo.reshape %v1442 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1444 = stablehlo.broadcast_in_dim %u19pnmu, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1445 = stablehlo.subtract %v1443, %v1444 : tensor<64x256x7x7xf32>
    %v1446 = stablehlo.broadcast_in_dim %u19pnvar, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1447 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v1448 = stablehlo.add %v1446, %v1447 : tensor<64x256x7x7xf32>
    %v1449 = stablehlo.rsqrt %v1448 : tensor<64x256x7x7xf32>
    %v1450 = stablehlo.multiply %v1445, %v1449 : tensor<64x256x7x7xf32>
    %v1451 = stablehlo.broadcast_in_dim %u19pg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1452 = stablehlo.broadcast_in_dim %u19pbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1453 = stablehlo.multiply %v1450, %v1451 : tensor<64x256x7x7xf32>
    %v1454 = stablehlo.add %v1453, %v1452 : tensor<64x256x7x7xf32>
    %v1455 = stablehlo.reshape %v1454 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1456 = stablehlo.reshape %v1455 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1457 = stablehlo.reshape %v1415 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1458 = stablehlo.add %v1456, %v1457 : tensor<64x256x7x7xf32>
    %v1459 = stablehlo.reshape %v1458 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1460 = stablehlo.reshape %v1459 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1461 = stablehlo.convolution(%v1460, %u20eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x7x7xf32>, tensor<1024x256x1x1xf32>) -> tensor<64x1024x7x7xf32>
    %v1462 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1463 = stablehlo.add %v1461, %v1462 : tensor<64x1024x7x7xf32>
    %v1464 = stablehlo.reshape %v1463 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1465 = stablehlo.reshape %v1464 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1466 = stablehlo.broadcast_in_dim %u20enmu, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1467 = stablehlo.subtract %v1465, %v1466 : tensor<64x1024x7x7xf32>
    %v1468 = stablehlo.broadcast_in_dim %u20envar, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1469 = stablehlo.constant dense<1.0e-5> : tensor<64x1024x7x7xf32>
    %v1470 = stablehlo.add %v1468, %v1469 : tensor<64x1024x7x7xf32>
    %v1471 = stablehlo.rsqrt %v1470 : tensor<64x1024x7x7xf32>
    %v1472 = stablehlo.multiply %v1467, %v1471 : tensor<64x1024x7x7xf32>
    %v1473 = stablehlo.broadcast_in_dim %u20eg, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1474 = stablehlo.broadcast_in_dim %u20ebt, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x7x7xf32>
    %v1475 = stablehlo.multiply %v1472, %v1473 : tensor<64x1024x7x7xf32>
    %v1476 = stablehlo.add %v1475, %v1474 : tensor<64x1024x7x7xf32>
    %v1477 = stablehlo.reshape %v1476 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1478 = stablehlo.reshape %v1477 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1479 = stablehlo.constant dense<0.0> : tensor<64x1024x7x7xf32>
    %v1480 = stablehlo.maximum %v1478, %v1479 : tensor<64x1024x7x7xf32>
    %v1481 = stablehlo.reshape %v1480 : (tensor<64x1024x7x7xf32>) -> tensor<64x50176xf32>
    %v1482 = stablehlo.reshape %v1481 : (tensor<64x50176xf32>) -> tensor<64x1024x7x7xf32>
    %v1483 = stablehlo.convolution(%v1482, %u20pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x7x7xf32>, tensor<256x1024x1x1xf32>) -> tensor<64x256x7x7xf32>
    %v1484 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1485 = stablehlo.add %v1483, %v1484 : tensor<64x256x7x7xf32>
    %v1486 = stablehlo.reshape %v1485 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1487 = stablehlo.reshape %v1486 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1488 = stablehlo.broadcast_in_dim %u20pnmu, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1489 = stablehlo.subtract %v1487, %v1488 : tensor<64x256x7x7xf32>
    %v1490 = stablehlo.broadcast_in_dim %u20pnvar, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1491 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v1492 = stablehlo.add %v1490, %v1491 : tensor<64x256x7x7xf32>
    %v1493 = stablehlo.rsqrt %v1492 : tensor<64x256x7x7xf32>
    %v1494 = stablehlo.multiply %v1489, %v1493 : tensor<64x256x7x7xf32>
    %v1495 = stablehlo.broadcast_in_dim %u20pg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1496 = stablehlo.broadcast_in_dim %u20pbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1497 = stablehlo.multiply %v1494, %v1495 : tensor<64x256x7x7xf32>
    %v1498 = stablehlo.add %v1497, %v1496 : tensor<64x256x7x7xf32>
    %v1499 = stablehlo.reshape %v1498 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1500 = stablehlo.reshape %v1499 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1501 = stablehlo.reshape %v1459 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1502 = stablehlo.add %v1500, %v1501 : tensor<64x256x7x7xf32>
    %v1503 = stablehlo.reshape %v1502 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1504 = stablehlo.reshape %v1503 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1505 = stablehlo.convolution(%v1504, %u21qW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 256 : i64} : (tensor<64x256x7x7xf32>, tensor<256x1x5x5xf32>) -> tensor<64x256x7x7xf32>
    %v1506 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1507 = stablehlo.add %v1505, %v1506 : tensor<64x256x7x7xf32>
    %v1508 = stablehlo.reshape %v1507 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1509 = stablehlo.reshape %v1508 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1510 = stablehlo.broadcast_in_dim %u21qnmu, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1511 = stablehlo.subtract %v1509, %v1510 : tensor<64x256x7x7xf32>
    %v1512 = stablehlo.broadcast_in_dim %u21qnvar, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1513 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v1514 = stablehlo.add %v1512, %v1513 : tensor<64x256x7x7xf32>
    %v1515 = stablehlo.rsqrt %v1514 : tensor<64x256x7x7xf32>
    %v1516 = stablehlo.multiply %v1511, %v1515 : tensor<64x256x7x7xf32>
    %v1517 = stablehlo.broadcast_in_dim %u21qg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1518 = stablehlo.broadcast_in_dim %u21qbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1519 = stablehlo.multiply %v1516, %v1517 : tensor<64x256x7x7xf32>
    %v1520 = stablehlo.add %v1519, %v1518 : tensor<64x256x7x7xf32>
    %v1521 = stablehlo.reshape %v1520 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1522 = stablehlo.reshape %v1521 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1523 = stablehlo.convolution(%v1522, %u21eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x7x7xf32>, tensor<512x256x1x1xf32>) -> tensor<64x512x7x7xf32>
    %v1524 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1525 = stablehlo.add %v1523, %v1524 : tensor<64x512x7x7xf32>
    %v1526 = stablehlo.reshape %v1525 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1527 = stablehlo.reshape %v1526 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1528 = stablehlo.broadcast_in_dim %u21enmu, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1529 = stablehlo.subtract %v1527, %v1528 : tensor<64x512x7x7xf32>
    %v1530 = stablehlo.broadcast_in_dim %u21envar, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1531 = stablehlo.constant dense<1.0e-5> : tensor<64x512x7x7xf32>
    %v1532 = stablehlo.add %v1530, %v1531 : tensor<64x512x7x7xf32>
    %v1533 = stablehlo.rsqrt %v1532 : tensor<64x512x7x7xf32>
    %v1534 = stablehlo.multiply %v1529, %v1533 : tensor<64x512x7x7xf32>
    %v1535 = stablehlo.broadcast_in_dim %u21eg, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1536 = stablehlo.broadcast_in_dim %u21ebt, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1537 = stablehlo.multiply %v1534, %v1535 : tensor<64x512x7x7xf32>
    %v1538 = stablehlo.add %v1537, %v1536 : tensor<64x512x7x7xf32>
    %v1539 = stablehlo.reshape %v1538 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1540 = stablehlo.reshape %v1539 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1541 = stablehlo.constant dense<0.0> : tensor<64x512x7x7xf32>
    %v1542 = stablehlo.maximum %v1540, %v1541 : tensor<64x512x7x7xf32>
    %v1543 = stablehlo.reshape %v1542 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1544 = stablehlo.reshape %v1543 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1545 = stablehlo.convolution(%v1544, %u21pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xf32>, tensor<256x512x1x1xf32>) -> tensor<64x256x7x7xf32>
    %v1546 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1547 = stablehlo.add %v1545, %v1546 : tensor<64x256x7x7xf32>
    %v1548 = stablehlo.reshape %v1547 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1549 = stablehlo.reshape %v1548 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1550 = stablehlo.broadcast_in_dim %u21pnmu, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1551 = stablehlo.subtract %v1549, %v1550 : tensor<64x256x7x7xf32>
    %v1552 = stablehlo.broadcast_in_dim %u21pnvar, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1553 = stablehlo.constant dense<1.0e-5> : tensor<64x256x7x7xf32>
    %v1554 = stablehlo.add %v1552, %v1553 : tensor<64x256x7x7xf32>
    %v1555 = stablehlo.rsqrt %v1554 : tensor<64x256x7x7xf32>
    %v1556 = stablehlo.multiply %v1551, %v1555 : tensor<64x256x7x7xf32>
    %v1557 = stablehlo.broadcast_in_dim %u21pg, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1558 = stablehlo.broadcast_in_dim %u21pbt, dims = [1] : (tensor<256xf32>) -> tensor<64x256x7x7xf32>
    %v1559 = stablehlo.multiply %v1556, %v1557 : tensor<64x256x7x7xf32>
    %v1560 = stablehlo.add %v1559, %v1558 : tensor<64x256x7x7xf32>
    %v1561 = stablehlo.reshape %v1560 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1562 = stablehlo.reshape %v1561 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1563 = stablehlo.reshape %v1503 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1564 = stablehlo.add %v1562, %v1563 : tensor<64x256x7x7xf32>
    %v1565 = stablehlo.reshape %v1564 : (tensor<64x256x7x7xf32>) -> tensor<64x12544xf32>
    %v1566 = stablehlo.reshape %v1565 : (tensor<64x12544xf32>) -> tensor<64x256x7x7xf32>
    %v1567 = stablehlo.convolution(%v1566, %h1W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x7x7xf32>, tensor<960x256x1x1xf32>) -> tensor<64x960x7x7xf32>
    %v1568 = stablehlo.broadcast_in_dim %zb960, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1569 = stablehlo.add %v1567, %v1568 : tensor<64x960x7x7xf32>
    %v1570 = stablehlo.reshape %v1569 : (tensor<64x960x7x7xf32>) -> tensor<64x47040xf32>
    %v1571 = stablehlo.reshape %v1570 : (tensor<64x47040xf32>) -> tensor<64x960x7x7xf32>
    %v1572 = stablehlo.broadcast_in_dim %h1nmu, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1573 = stablehlo.subtract %v1571, %v1572 : tensor<64x960x7x7xf32>
    %v1574 = stablehlo.broadcast_in_dim %h1nvar, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1575 = stablehlo.constant dense<1.0e-5> : tensor<64x960x7x7xf32>
    %v1576 = stablehlo.add %v1574, %v1575 : tensor<64x960x7x7xf32>
    %v1577 = stablehlo.rsqrt %v1576 : tensor<64x960x7x7xf32>
    %v1578 = stablehlo.multiply %v1573, %v1577 : tensor<64x960x7x7xf32>
    %v1579 = stablehlo.broadcast_in_dim %h1g, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1580 = stablehlo.broadcast_in_dim %h1bt, dims = [1] : (tensor<960xf32>) -> tensor<64x960x7x7xf32>
    %v1581 = stablehlo.multiply %v1578, %v1579 : tensor<64x960x7x7xf32>
    %v1582 = stablehlo.add %v1581, %v1580 : tensor<64x960x7x7xf32>
    %v1583 = stablehlo.reshape %v1582 : (tensor<64x960x7x7xf32>) -> tensor<64x47040xf32>
    %v1584 = stablehlo.reshape %v1583 : (tensor<64x47040xf32>) -> tensor<64x960x7x7xf32>
    %v1585 = stablehlo.constant dense<0.0> : tensor<64x960x7x7xf32>
    %v1586 = stablehlo.maximum %v1584, %v1585 : tensor<64x960x7x7xf32>
    %v1587 = stablehlo.reshape %v1586 : (tensor<64x960x7x7xf32>) -> tensor<64x47040xf32>
    %v1588 = stablehlo.reshape %v1587 : (tensor<64x47040xf32>) -> tensor<64x960x7x7xf32>
    %v1589 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1590 = stablehlo.reduce(%v1588 init: %v1589) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x960x7x7xf32>, tensor<f32>) -> tensor<64x960xf32>
    %v1591 = stablehlo.constant dense<49.0> : tensor<64x960xf32>
    %v1592 = stablehlo.divide %v1590, %v1591 : tensor<64x960xf32>
    %v1593 = stablehlo.reshape %v1592 : (tensor<64x960xf32>) -> tensor<64x960x1x1xf32>
    %v1594 = stablehlo.convolution(%v1593, %hW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x960x1x1xf32>, tensor<1280x960x1x1xf32>) -> tensor<64x1280x1x1xf32>
    %v1595 = stablehlo.broadcast_in_dim %zb1280, dims = [1] : (tensor<1280xf32>) -> tensor<64x1280x1x1xf32>
    %v1596 = stablehlo.add %v1594, %v1595 : tensor<64x1280x1x1xf32>
    %v1597 = stablehlo.reshape %v1596 : (tensor<64x1280x1x1xf32>) -> tensor<64x1280xf32>
    %v1598 = stablehlo.reshape %v1597 : (tensor<64x1280xf32>) -> tensor<64x1280x1x1xf32>
    %v1599 = stablehlo.broadcast_in_dim %hnmu, dims = [1] : (tensor<1280xf32>) -> tensor<64x1280x1x1xf32>
    %v1600 = stablehlo.subtract %v1598, %v1599 : tensor<64x1280x1x1xf32>
    %v1601 = stablehlo.broadcast_in_dim %hnvar, dims = [1] : (tensor<1280xf32>) -> tensor<64x1280x1x1xf32>
    %v1602 = stablehlo.constant dense<1.0e-5> : tensor<64x1280x1x1xf32>
    %v1603 = stablehlo.add %v1601, %v1602 : tensor<64x1280x1x1xf32>
    %v1604 = stablehlo.rsqrt %v1603 : tensor<64x1280x1x1xf32>
    %v1605 = stablehlo.multiply %v1600, %v1604 : tensor<64x1280x1x1xf32>
    %v1606 = stablehlo.broadcast_in_dim %hg, dims = [1] : (tensor<1280xf32>) -> tensor<64x1280x1x1xf32>
    %v1607 = stablehlo.broadcast_in_dim %hbt, dims = [1] : (tensor<1280xf32>) -> tensor<64x1280x1x1xf32>
    %v1608 = stablehlo.multiply %v1605, %v1606 : tensor<64x1280x1x1xf32>
    %v1609 = stablehlo.add %v1608, %v1607 : tensor<64x1280x1x1xf32>
    %v1610 = stablehlo.reshape %v1609 : (tensor<64x1280x1x1xf32>) -> tensor<64x1280xf32>
    %v1611 = stablehlo.reshape %v1610 : (tensor<64x1280xf32>) -> tensor<64x1280x1x1xf32>
    %v1612 = stablehlo.constant dense<0.0> : tensor<64x1280x1x1xf32>
    %v1613 = stablehlo.maximum %v1611, %v1612 : tensor<64x1280x1x1xf32>
    %v1614 = stablehlo.reshape %v1613 : (tensor<64x1280x1x1xf32>) -> tensor<64x1280xf32>
    %v1615 = stablehlo.dot_general %v1614, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x1280xf32>, tensor<1280x1000xf32>) -> tensor<64x1000xf32>
    %v1616 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<1000xf32>) -> tensor<64x1000xf32>
    %v1617 = stablehlo.add %v1615, %v1616 : tensor<64x1000xf32>
    return %v1617 : tensor<64x1000xf32>
  }
}
