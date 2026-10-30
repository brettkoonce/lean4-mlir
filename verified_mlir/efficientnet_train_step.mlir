module @m {
  func.func @efficientnet_train_step(%x: tensor<32x150528xf32>, %sW: tensor<32x3x3x3xf32>, %sg: tensor<32xf32>, %sbt: tensor<32xf32>, %b1dW: tensor<32x1x3x3xf32>, %b1dg: tensor<32xf32>, %b1dbt: tensor<32xf32>, %b1zW1: tensor<32x8xf32>, %b1zb1: tensor<8xf32>, %b1zW2: tensor<8x32xf32>, %b1zb2: tensor<32xf32>, %b1pW: tensor<16x32x1x1xf32>, %b1pg: tensor<16xf32>, %b1pbt: tensor<16xf32>, %b2eW: tensor<96x16x1x1xf32>, %b2eg: tensor<96xf32>, %b2ebt: tensor<96xf32>, %b2dW: tensor<96x1x3x3xf32>, %b2dg: tensor<96xf32>, %b2dbt: tensor<96xf32>, %b2zW1: tensor<96x4xf32>, %b2zb1: tensor<4xf32>, %b2zW2: tensor<4x96xf32>, %b2zb2: tensor<96xf32>, %b2pW: tensor<24x96x1x1xf32>, %b2pg: tensor<24xf32>, %b2pbt: tensor<24xf32>, %b3eW: tensor<144x24x1x1xf32>, %b3eg: tensor<144xf32>, %b3ebt: tensor<144xf32>, %b3dW: tensor<144x1x3x3xf32>, %b3dg: tensor<144xf32>, %b3dbt: tensor<144xf32>, %b3zW1: tensor<144x6xf32>, %b3zb1: tensor<6xf32>, %b3zW2: tensor<6x144xf32>, %b3zb2: tensor<144xf32>, %b3pW: tensor<24x144x1x1xf32>, %b3pg: tensor<24xf32>, %b3pbt: tensor<24xf32>, %b4eW: tensor<144x24x1x1xf32>, %b4eg: tensor<144xf32>, %b4ebt: tensor<144xf32>, %b4dW: tensor<144x1x5x5xf32>, %b4dg: tensor<144xf32>, %b4dbt: tensor<144xf32>, %b4zW1: tensor<144x6xf32>, %b4zb1: tensor<6xf32>, %b4zW2: tensor<6x144xf32>, %b4zb2: tensor<144xf32>, %b4pW: tensor<40x144x1x1xf32>, %b4pg: tensor<40xf32>, %b4pbt: tensor<40xf32>, %b5eW: tensor<240x40x1x1xf32>, %b5eg: tensor<240xf32>, %b5ebt: tensor<240xf32>, %b5dW: tensor<240x1x5x5xf32>, %b5dg: tensor<240xf32>, %b5dbt: tensor<240xf32>, %b5zW1: tensor<240x10xf32>, %b5zb1: tensor<10xf32>, %b5zW2: tensor<10x240xf32>, %b5zb2: tensor<240xf32>, %b5pW: tensor<40x240x1x1xf32>, %b5pg: tensor<40xf32>, %b5pbt: tensor<40xf32>, %b6eW: tensor<240x40x1x1xf32>, %b6eg: tensor<240xf32>, %b6ebt: tensor<240xf32>, %b6dW: tensor<240x1x3x3xf32>, %b6dg: tensor<240xf32>, %b6dbt: tensor<240xf32>, %b6zW1: tensor<240x10xf32>, %b6zb1: tensor<10xf32>, %b6zW2: tensor<10x240xf32>, %b6zb2: tensor<240xf32>, %b6pW: tensor<80x240x1x1xf32>, %b6pg: tensor<80xf32>, %b6pbt: tensor<80xf32>, %b7eW: tensor<480x80x1x1xf32>, %b7eg: tensor<480xf32>, %b7ebt: tensor<480xf32>, %b7dW: tensor<480x1x3x3xf32>, %b7dg: tensor<480xf32>, %b7dbt: tensor<480xf32>, %b7zW1: tensor<480x20xf32>, %b7zb1: tensor<20xf32>, %b7zW2: tensor<20x480xf32>, %b7zb2: tensor<480xf32>, %b7pW: tensor<80x480x1x1xf32>, %b7pg: tensor<80xf32>, %b7pbt: tensor<80xf32>, %b8eW: tensor<480x80x1x1xf32>, %b8eg: tensor<480xf32>, %b8ebt: tensor<480xf32>, %b8dW: tensor<480x1x3x3xf32>, %b8dg: tensor<480xf32>, %b8dbt: tensor<480xf32>, %b8zW1: tensor<480x20xf32>, %b8zb1: tensor<20xf32>, %b8zW2: tensor<20x480xf32>, %b8zb2: tensor<480xf32>, %b8pW: tensor<80x480x1x1xf32>, %b8pg: tensor<80xf32>, %b8pbt: tensor<80xf32>, %b9eW: tensor<480x80x1x1xf32>, %b9eg: tensor<480xf32>, %b9ebt: tensor<480xf32>, %b9dW: tensor<480x1x5x5xf32>, %b9dg: tensor<480xf32>, %b9dbt: tensor<480xf32>, %b9zW1: tensor<480x20xf32>, %b9zb1: tensor<20xf32>, %b9zW2: tensor<20x480xf32>, %b9zb2: tensor<480xf32>, %b9pW: tensor<112x480x1x1xf32>, %b9pg: tensor<112xf32>, %b9pbt: tensor<112xf32>, %b10eW: tensor<672x112x1x1xf32>, %b10eg: tensor<672xf32>, %b10ebt: tensor<672xf32>, %b10dW: tensor<672x1x5x5xf32>, %b10dg: tensor<672xf32>, %b10dbt: tensor<672xf32>, %b10zW1: tensor<672x28xf32>, %b10zb1: tensor<28xf32>, %b10zW2: tensor<28x672xf32>, %b10zb2: tensor<672xf32>, %b10pW: tensor<112x672x1x1xf32>, %b10pg: tensor<112xf32>, %b10pbt: tensor<112xf32>, %b11eW: tensor<672x112x1x1xf32>, %b11eg: tensor<672xf32>, %b11ebt: tensor<672xf32>, %b11dW: tensor<672x1x5x5xf32>, %b11dg: tensor<672xf32>, %b11dbt: tensor<672xf32>, %b11zW1: tensor<672x28xf32>, %b11zb1: tensor<28xf32>, %b11zW2: tensor<28x672xf32>, %b11zb2: tensor<672xf32>, %b11pW: tensor<112x672x1x1xf32>, %b11pg: tensor<112xf32>, %b11pbt: tensor<112xf32>, %b12eW: tensor<672x112x1x1xf32>, %b12eg: tensor<672xf32>, %b12ebt: tensor<672xf32>, %b12dW: tensor<672x1x5x5xf32>, %b12dg: tensor<672xf32>, %b12dbt: tensor<672xf32>, %b12zW1: tensor<672x28xf32>, %b12zb1: tensor<28xf32>, %b12zW2: tensor<28x672xf32>, %b12zb2: tensor<672xf32>, %b12pW: tensor<192x672x1x1xf32>, %b12pg: tensor<192xf32>, %b12pbt: tensor<192xf32>, %b13eW: tensor<1152x192x1x1xf32>, %b13eg: tensor<1152xf32>, %b13ebt: tensor<1152xf32>, %b13dW: tensor<1152x1x5x5xf32>, %b13dg: tensor<1152xf32>, %b13dbt: tensor<1152xf32>, %b13zW1: tensor<1152x48xf32>, %b13zb1: tensor<48xf32>, %b13zW2: tensor<48x1152xf32>, %b13zb2: tensor<1152xf32>, %b13pW: tensor<192x1152x1x1xf32>, %b13pg: tensor<192xf32>, %b13pbt: tensor<192xf32>, %b14eW: tensor<1152x192x1x1xf32>, %b14eg: tensor<1152xf32>, %b14ebt: tensor<1152xf32>, %b14dW: tensor<1152x1x5x5xf32>, %b14dg: tensor<1152xf32>, %b14dbt: tensor<1152xf32>, %b14zW1: tensor<1152x48xf32>, %b14zb1: tensor<48xf32>, %b14zW2: tensor<48x1152xf32>, %b14zb2: tensor<1152xf32>, %b14pW: tensor<192x1152x1x1xf32>, %b14pg: tensor<192xf32>, %b14pbt: tensor<192xf32>, %b15eW: tensor<1152x192x1x1xf32>, %b15eg: tensor<1152xf32>, %b15ebt: tensor<1152xf32>, %b15dW: tensor<1152x1x5x5xf32>, %b15dg: tensor<1152xf32>, %b15dbt: tensor<1152xf32>, %b15zW1: tensor<1152x48xf32>, %b15zb1: tensor<48xf32>, %b15zW2: tensor<48x1152xf32>, %b15zb2: tensor<1152xf32>, %b15pW: tensor<192x1152x1x1xf32>, %b15pg: tensor<192xf32>, %b15pbt: tensor<192xf32>, %b16eW: tensor<1152x192x1x1xf32>, %b16eg: tensor<1152xf32>, %b16ebt: tensor<1152xf32>, %b16dW: tensor<1152x1x3x3xf32>, %b16dg: tensor<1152xf32>, %b16dbt: tensor<1152xf32>, %b16zW1: tensor<1152x48xf32>, %b16zb1: tensor<48xf32>, %b16zW2: tensor<48x1152xf32>, %b16zb2: tensor<1152xf32>, %b16pW: tensor<320x1152x1x1xf32>, %b16pg: tensor<320xf32>, %b16pbt: tensor<320xf32>, %hW: tensor<1280x320x1x1xf32>, %hg: tensor<1280xf32>, %hbt: tensor<1280xf32>, %Wd: tensor<1280x10xf32>, %bd: tensor<10xf32>, %onehot: tensor<32x10xf32>) -> (tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x8xf32>, tensor<8xf32>, tensor<8x32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x4xf32>, tensor<4xf32>, tensor<4x96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x6xf32>, tensor<6xf32>, tensor<6x144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x5x5xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x6xf32>, tensor<6xf32>, tensor<6x144xf32>, tensor<144xf32>, tensor<40x144x1x1xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240x40x1x1xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x1x5x5xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x10xf32>, tensor<10xf32>, tensor<10x240xf32>, tensor<240xf32>, tensor<40x240x1x1xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240x40x1x1xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x1x3x3xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x10xf32>, tensor<10xf32>, tensor<10x240xf32>, tensor<240xf32>, tensor<80x240x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x3x3xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<80x480x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x3x3xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<80x480x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x5x5xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<112x480x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<112x672x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<112x672x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<192x672x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x3x3xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<320x1152x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x10xf32>, tensor<10xf32>) {
    // ── EfficientNet-B0 (16-MBConv) train step: every op is pretty(verified AST node) except the %zb zero-bias constants ──
    // The conv biases are not in the signature (BN removes them; He et al.'s
    // `.convBn` has none). The proven conv ops still take a bias operand, so it is bound to a
    // zero constant here — same op, `bias = 0`, and `x + 0.0` is exact.
    %zb16 = stablehlo.constant dense<0.0> : tensor<16xf32>
    %zb24 = stablehlo.constant dense<0.0> : tensor<24xf32>
    %zb32 = stablehlo.constant dense<0.0> : tensor<32xf32>
    %zb40 = stablehlo.constant dense<0.0> : tensor<40xf32>
    %zb80 = stablehlo.constant dense<0.0> : tensor<80xf32>
    %zb96 = stablehlo.constant dense<0.0> : tensor<96xf32>
    %zb112 = stablehlo.constant dense<0.0> : tensor<112xf32>
    %zb144 = stablehlo.constant dense<0.0> : tensor<144xf32>
    %zb192 = stablehlo.constant dense<0.0> : tensor<192xf32>
    %zb240 = stablehlo.constant dense<0.0> : tensor<240xf32>
    %zb320 = stablehlo.constant dense<0.0> : tensor<320xf32>
    %zb480 = stablehlo.constant dense<0.0> : tensor<480xf32>
    %zb672 = stablehlo.constant dense<0.0> : tensor<672xf32>
    %zb1152 = stablehlo.constant dense<0.0> : tensor<1152xf32>
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
    %v26 = stablehlo.logistic %v25 : tensor<32x32x112x112xf32>
    %v27 = stablehlo.multiply %v25, %v26 : tensor<32x32x112x112xf32>
    %v28 = stablehlo.reshape %v27 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v29 = stablehlo.reshape %v28 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v30 = stablehlo.convolution(%v29, %b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 32 : i64} : (tensor<32x32x112x112xf32>, tensor<32x1x3x3xf32>) -> tensor<32x32x112x112xf32>
    %v31 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v32 = stablehlo.add %v30, %v31 : tensor<32x32x112x112xf32>
    %v33 = stablehlo.reshape %v32 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v34 = stablehlo.reshape %v33 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v35 = stablehlo.constant dense<0.0> : tensor<f32>
    %v36 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v37 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v38 = stablehlo.reduce(%v34 init: %v35) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v39 = stablehlo.broadcast_in_dim %v38, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v40 = stablehlo.divide %v39, %v36 : tensor<32x32x112x112xf32>
    %v41 = stablehlo.subtract %v34, %v40 : tensor<32x32x112x112xf32>
    %v42 = stablehlo.multiply %v41, %v41 : tensor<32x32x112x112xf32>
    %v43 = stablehlo.reduce(%v42 init: %v35) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v44 = stablehlo.broadcast_in_dim %v43, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v45 = stablehlo.divide %v44, %v36 : tensor<32x32x112x112xf32>
    %v46 = stablehlo.add %v45, %v37 : tensor<32x32x112x112xf32>
    %v47 = stablehlo.rsqrt %v46 : tensor<32x32x112x112xf32>
    %v48 = stablehlo.multiply %v41, %v47 : tensor<32x32x112x112xf32>
    %v49 = stablehlo.broadcast_in_dim %b1dg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v50 = stablehlo.broadcast_in_dim %b1dbt, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v51 = stablehlo.multiply %v48, %v49 : tensor<32x32x112x112xf32>
    %v52 = stablehlo.add %v51, %v50 : tensor<32x32x112x112xf32>
    %v53 = stablehlo.reshape %v52 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v54 = stablehlo.reshape %v53 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v55 = stablehlo.logistic %v54 : tensor<32x32x112x112xf32>
    %v56 = stablehlo.multiply %v54, %v55 : tensor<32x32x112x112xf32>
    %v57 = stablehlo.reshape %v56 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v59 = stablehlo.constant dense<0.0> : tensor<f32>
    %v60 = stablehlo.reduce(%v58 init: %v59) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v61 = stablehlo.constant dense<12544.0> : tensor<32x32xf32>
    %v62 = stablehlo.divide %v60, %v61 : tensor<32x32xf32>
    %v63 = stablehlo.dot_general %v62, %b1zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x8xf32>) -> tensor<32x8xf32>
    %v64 = stablehlo.broadcast_in_dim %b1zb1, dims = [1] : (tensor<8xf32>) -> tensor<32x8xf32>
    %v65 = stablehlo.add %v63, %v64 : tensor<32x8xf32>
    %v66 = stablehlo.logistic %v65 : tensor<32x8xf32>
    %v67 = stablehlo.multiply %v65, %v66 : tensor<32x8xf32>
    %v68 = stablehlo.dot_general %v67, %b1zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x8xf32>, tensor<8x32xf32>) -> tensor<32x32xf32>
    %v69 = stablehlo.broadcast_in_dim %b1zb2, dims = [1] : (tensor<32xf32>) -> tensor<32x32xf32>
    %v70 = stablehlo.add %v68, %v69 : tensor<32x32xf32>
    %v71 = stablehlo.logistic %v70 : tensor<32x32xf32>
    %v72 = stablehlo.broadcast_in_dim %v71, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v73 = stablehlo.multiply %v58, %v72 : tensor<32x32x112x112xf32>
    %v74 = stablehlo.reshape %v73 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v75 = stablehlo.reshape %v74 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v76 = stablehlo.convolution(%v75, %b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<16x32x1x1xf32>) -> tensor<32x16x112x112xf32>
    %v77 = stablehlo.broadcast_in_dim %zb16, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v78 = stablehlo.add %v76, %v77 : tensor<32x16x112x112xf32>
    %v79 = stablehlo.reshape %v78 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v80 = stablehlo.reshape %v79 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v81 = stablehlo.constant dense<0.0> : tensor<f32>
    %v82 = stablehlo.constant dense<401408.0> : tensor<32x16x112x112xf32>
    %v83 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v84 = stablehlo.reduce(%v80 init: %v81) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v85 = stablehlo.broadcast_in_dim %v84, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v86 = stablehlo.divide %v85, %v82 : tensor<32x16x112x112xf32>
    %v87 = stablehlo.subtract %v80, %v86 : tensor<32x16x112x112xf32>
    %v88 = stablehlo.multiply %v87, %v87 : tensor<32x16x112x112xf32>
    %v89 = stablehlo.reduce(%v88 init: %v81) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v90 = stablehlo.broadcast_in_dim %v89, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v91 = stablehlo.divide %v90, %v82 : tensor<32x16x112x112xf32>
    %v92 = stablehlo.add %v91, %v83 : tensor<32x16x112x112xf32>
    %v93 = stablehlo.rsqrt %v92 : tensor<32x16x112x112xf32>
    %v94 = stablehlo.multiply %v87, %v93 : tensor<32x16x112x112xf32>
    %v95 = stablehlo.broadcast_in_dim %b1pg, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v96 = stablehlo.broadcast_in_dim %b1pbt, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v97 = stablehlo.multiply %v94, %v95 : tensor<32x16x112x112xf32>
    %v98 = stablehlo.add %v97, %v96 : tensor<32x16x112x112xf32>
    %v99 = stablehlo.reshape %v98 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v100 = stablehlo.reshape %v99 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v101 = stablehlo.convolution(%v100, %b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x16x112x112xf32>, tensor<96x16x1x1xf32>) -> tensor<32x96x112x112xf32>
    %v102 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v103 = stablehlo.add %v101, %v102 : tensor<32x96x112x112xf32>
    %v104 = stablehlo.reshape %v103 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v105 = stablehlo.reshape %v104 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v106 = stablehlo.constant dense<0.0> : tensor<f32>
    %v107 = stablehlo.constant dense<401408.0> : tensor<32x96x112x112xf32>
    %v108 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v109 = stablehlo.reduce(%v105 init: %v106) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v110 = stablehlo.broadcast_in_dim %v109, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v111 = stablehlo.divide %v110, %v107 : tensor<32x96x112x112xf32>
    %v112 = stablehlo.subtract %v105, %v111 : tensor<32x96x112x112xf32>
    %v113 = stablehlo.multiply %v112, %v112 : tensor<32x96x112x112xf32>
    %v114 = stablehlo.reduce(%v113 init: %v106) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v115 = stablehlo.broadcast_in_dim %v114, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v116 = stablehlo.divide %v115, %v107 : tensor<32x96x112x112xf32>
    %v117 = stablehlo.add %v116, %v108 : tensor<32x96x112x112xf32>
    %v118 = stablehlo.rsqrt %v117 : tensor<32x96x112x112xf32>
    %v119 = stablehlo.multiply %v112, %v118 : tensor<32x96x112x112xf32>
    %v120 = stablehlo.broadcast_in_dim %b2eg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v121 = stablehlo.broadcast_in_dim %b2ebt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v122 = stablehlo.multiply %v119, %v120 : tensor<32x96x112x112xf32>
    %v123 = stablehlo.add %v122, %v121 : tensor<32x96x112x112xf32>
    %v124 = stablehlo.reshape %v123 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v125 = stablehlo.reshape %v124 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v126 = stablehlo.logistic %v125 : tensor<32x96x112x112xf32>
    %v127 = stablehlo.multiply %v125, %v126 : tensor<32x96x112x112xf32>
    %v128 = stablehlo.reshape %v127 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v129 = stablehlo.reshape %v128 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v130 = stablehlo.convolution(%v129, %b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x112x112xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x56x56xf32>
    %v131 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v132 = stablehlo.add %v130, %v131 : tensor<32x96x56x56xf32>
    %v133 = stablehlo.reshape %v132 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v134 = stablehlo.reshape %v133 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v135 = stablehlo.constant dense<0.0> : tensor<f32>
    %v136 = stablehlo.constant dense<100352.0> : tensor<32x96x56x56xf32>
    %v137 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v138 = stablehlo.reduce(%v134 init: %v135) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v139 = stablehlo.broadcast_in_dim %v138, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v140 = stablehlo.divide %v139, %v136 : tensor<32x96x56x56xf32>
    %v141 = stablehlo.subtract %v134, %v140 : tensor<32x96x56x56xf32>
    %v142 = stablehlo.multiply %v141, %v141 : tensor<32x96x56x56xf32>
    %v143 = stablehlo.reduce(%v142 init: %v135) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v144 = stablehlo.broadcast_in_dim %v143, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v145 = stablehlo.divide %v144, %v136 : tensor<32x96x56x56xf32>
    %v146 = stablehlo.add %v145, %v137 : tensor<32x96x56x56xf32>
    %v147 = stablehlo.rsqrt %v146 : tensor<32x96x56x56xf32>
    %v148 = stablehlo.multiply %v141, %v147 : tensor<32x96x56x56xf32>
    %v149 = stablehlo.broadcast_in_dim %b2dg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v150 = stablehlo.broadcast_in_dim %b2dbt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v151 = stablehlo.multiply %v148, %v149 : tensor<32x96x56x56xf32>
    %v152 = stablehlo.add %v151, %v150 : tensor<32x96x56x56xf32>
    %v153 = stablehlo.reshape %v152 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v154 = stablehlo.reshape %v153 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v155 = stablehlo.logistic %v154 : tensor<32x96x56x56xf32>
    %v156 = stablehlo.multiply %v154, %v155 : tensor<32x96x56x56xf32>
    %v157 = stablehlo.reshape %v156 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v158 = stablehlo.reshape %v157 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v159 = stablehlo.constant dense<0.0> : tensor<f32>
    %v160 = stablehlo.reduce(%v158 init: %v159) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v161 = stablehlo.constant dense<3136.0> : tensor<32x96xf32>
    %v162 = stablehlo.divide %v160, %v161 : tensor<32x96xf32>
    %v163 = stablehlo.dot_general %v162, %b2zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<96x4xf32>) -> tensor<32x4xf32>
    %v164 = stablehlo.broadcast_in_dim %b2zb1, dims = [1] : (tensor<4xf32>) -> tensor<32x4xf32>
    %v165 = stablehlo.add %v163, %v164 : tensor<32x4xf32>
    %v166 = stablehlo.logistic %v165 : tensor<32x4xf32>
    %v167 = stablehlo.multiply %v165, %v166 : tensor<32x4xf32>
    %v168 = stablehlo.dot_general %v167, %b2zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<4x96xf32>) -> tensor<32x96xf32>
    %v169 = stablehlo.broadcast_in_dim %b2zb2, dims = [1] : (tensor<96xf32>) -> tensor<32x96xf32>
    %v170 = stablehlo.add %v168, %v169 : tensor<32x96xf32>
    %v171 = stablehlo.logistic %v170 : tensor<32x96xf32>
    %v172 = stablehlo.broadcast_in_dim %v171, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v173 = stablehlo.multiply %v158, %v172 : tensor<32x96x56x56xf32>
    %v174 = stablehlo.reshape %v173 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v175 = stablehlo.reshape %v174 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v176 = stablehlo.convolution(%v175, %b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<24x96x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v177 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v178 = stablehlo.add %v176, %v177 : tensor<32x24x56x56xf32>
    %v179 = stablehlo.reshape %v178 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v180 = stablehlo.reshape %v179 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v181 = stablehlo.constant dense<0.0> : tensor<f32>
    %v182 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v183 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v184 = stablehlo.reduce(%v180 init: %v181) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v185 = stablehlo.broadcast_in_dim %v184, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v186 = stablehlo.divide %v185, %v182 : tensor<32x24x56x56xf32>
    %v187 = stablehlo.subtract %v180, %v186 : tensor<32x24x56x56xf32>
    %v188 = stablehlo.multiply %v187, %v187 : tensor<32x24x56x56xf32>
    %v189 = stablehlo.reduce(%v188 init: %v181) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v190 = stablehlo.broadcast_in_dim %v189, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v191 = stablehlo.divide %v190, %v182 : tensor<32x24x56x56xf32>
    %v192 = stablehlo.add %v191, %v183 : tensor<32x24x56x56xf32>
    %v193 = stablehlo.rsqrt %v192 : tensor<32x24x56x56xf32>
    %v194 = stablehlo.multiply %v187, %v193 : tensor<32x24x56x56xf32>
    %v195 = stablehlo.broadcast_in_dim %b2pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v196 = stablehlo.broadcast_in_dim %b2pbt, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v197 = stablehlo.multiply %v194, %v195 : tensor<32x24x56x56xf32>
    %v198 = stablehlo.add %v197, %v196 : tensor<32x24x56x56xf32>
    %v199 = stablehlo.reshape %v198 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v200 = stablehlo.reshape %v199 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v201 = stablehlo.convolution(%v200, %b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v202 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v203 = stablehlo.add %v201, %v202 : tensor<32x144x56x56xf32>
    %v204 = stablehlo.reshape %v203 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v205 = stablehlo.reshape %v204 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v206 = stablehlo.constant dense<0.0> : tensor<f32>
    %v207 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v208 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v209 = stablehlo.reduce(%v205 init: %v206) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v210 = stablehlo.broadcast_in_dim %v209, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v211 = stablehlo.divide %v210, %v207 : tensor<32x144x56x56xf32>
    %v212 = stablehlo.subtract %v205, %v211 : tensor<32x144x56x56xf32>
    %v213 = stablehlo.multiply %v212, %v212 : tensor<32x144x56x56xf32>
    %v214 = stablehlo.reduce(%v213 init: %v206) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v215 = stablehlo.broadcast_in_dim %v214, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v216 = stablehlo.divide %v215, %v207 : tensor<32x144x56x56xf32>
    %v217 = stablehlo.add %v216, %v208 : tensor<32x144x56x56xf32>
    %v218 = stablehlo.rsqrt %v217 : tensor<32x144x56x56xf32>
    %v219 = stablehlo.multiply %v212, %v218 : tensor<32x144x56x56xf32>
    %v220 = stablehlo.broadcast_in_dim %b3eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v221 = stablehlo.broadcast_in_dim %b3ebt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v222 = stablehlo.multiply %v219, %v220 : tensor<32x144x56x56xf32>
    %v223 = stablehlo.add %v222, %v221 : tensor<32x144x56x56xf32>
    %v224 = stablehlo.reshape %v223 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v225 = stablehlo.reshape %v224 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v226 = stablehlo.logistic %v225 : tensor<32x144x56x56xf32>
    %v227 = stablehlo.multiply %v225, %v226 : tensor<32x144x56x56xf32>
    %v228 = stablehlo.reshape %v227 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v229 = stablehlo.reshape %v228 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v230 = stablehlo.convolution(%v229, %b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x56x56xf32>
    %v231 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v232 = stablehlo.add %v230, %v231 : tensor<32x144x56x56xf32>
    %v233 = stablehlo.reshape %v232 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v234 = stablehlo.reshape %v233 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v235 = stablehlo.constant dense<0.0> : tensor<f32>
    %v236 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v237 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v238 = stablehlo.reduce(%v234 init: %v235) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v239 = stablehlo.broadcast_in_dim %v238, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v240 = stablehlo.divide %v239, %v236 : tensor<32x144x56x56xf32>
    %v241 = stablehlo.subtract %v234, %v240 : tensor<32x144x56x56xf32>
    %v242 = stablehlo.multiply %v241, %v241 : tensor<32x144x56x56xf32>
    %v243 = stablehlo.reduce(%v242 init: %v235) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v244 = stablehlo.broadcast_in_dim %v243, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v245 = stablehlo.divide %v244, %v236 : tensor<32x144x56x56xf32>
    %v246 = stablehlo.add %v245, %v237 : tensor<32x144x56x56xf32>
    %v247 = stablehlo.rsqrt %v246 : tensor<32x144x56x56xf32>
    %v248 = stablehlo.multiply %v241, %v247 : tensor<32x144x56x56xf32>
    %v249 = stablehlo.broadcast_in_dim %b3dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v250 = stablehlo.broadcast_in_dim %b3dbt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v251 = stablehlo.multiply %v248, %v249 : tensor<32x144x56x56xf32>
    %v252 = stablehlo.add %v251, %v250 : tensor<32x144x56x56xf32>
    %v253 = stablehlo.reshape %v252 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v254 = stablehlo.reshape %v253 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v255 = stablehlo.logistic %v254 : tensor<32x144x56x56xf32>
    %v256 = stablehlo.multiply %v254, %v255 : tensor<32x144x56x56xf32>
    %v257 = stablehlo.reshape %v256 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v258 = stablehlo.reshape %v257 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v259 = stablehlo.constant dense<0.0> : tensor<f32>
    %v260 = stablehlo.reduce(%v258 init: %v259) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v261 = stablehlo.constant dense<3136.0> : tensor<32x144xf32>
    %v262 = stablehlo.divide %v260, %v261 : tensor<32x144xf32>
    %v263 = stablehlo.dot_general %v262, %b3zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v264 = stablehlo.broadcast_in_dim %b3zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v265 = stablehlo.add %v263, %v264 : tensor<32x6xf32>
    %v266 = stablehlo.logistic %v265 : tensor<32x6xf32>
    %v267 = stablehlo.multiply %v265, %v266 : tensor<32x6xf32>
    %v268 = stablehlo.dot_general %v267, %b3zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v269 = stablehlo.broadcast_in_dim %b3zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v270 = stablehlo.add %v268, %v269 : tensor<32x144xf32>
    %v271 = stablehlo.logistic %v270 : tensor<32x144xf32>
    %v272 = stablehlo.broadcast_in_dim %v271, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v273 = stablehlo.multiply %v258, %v272 : tensor<32x144x56x56xf32>
    %v274 = stablehlo.reshape %v273 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v275 = stablehlo.reshape %v274 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v276 = stablehlo.convolution(%v275, %b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v277 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v278 = stablehlo.add %v276, %v277 : tensor<32x24x56x56xf32>
    %v279 = stablehlo.reshape %v278 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v280 = stablehlo.reshape %v279 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v281 = stablehlo.constant dense<0.0> : tensor<f32>
    %v282 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v283 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v284 = stablehlo.reduce(%v280 init: %v281) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v285 = stablehlo.broadcast_in_dim %v284, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v286 = stablehlo.divide %v285, %v282 : tensor<32x24x56x56xf32>
    %v287 = stablehlo.subtract %v280, %v286 : tensor<32x24x56x56xf32>
    %v288 = stablehlo.multiply %v287, %v287 : tensor<32x24x56x56xf32>
    %v289 = stablehlo.reduce(%v288 init: %v281) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v290 = stablehlo.broadcast_in_dim %v289, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v291 = stablehlo.divide %v290, %v282 : tensor<32x24x56x56xf32>
    %v292 = stablehlo.add %v291, %v283 : tensor<32x24x56x56xf32>
    %v293 = stablehlo.rsqrt %v292 : tensor<32x24x56x56xf32>
    %v294 = stablehlo.multiply %v287, %v293 : tensor<32x24x56x56xf32>
    %v295 = stablehlo.broadcast_in_dim %b3pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v296 = stablehlo.broadcast_in_dim %b3pbt, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v297 = stablehlo.multiply %v294, %v295 : tensor<32x24x56x56xf32>
    %v298 = stablehlo.add %v297, %v296 : tensor<32x24x56x56xf32>
    %v299 = stablehlo.reshape %v298 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v300 = stablehlo.reshape %v299 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v301 = stablehlo.reshape %v199 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v302 = stablehlo.add %v300, %v301 : tensor<32x24x56x56xf32>
    %v303 = stablehlo.reshape %v302 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v304 = stablehlo.reshape %v303 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v305 = stablehlo.convolution(%v304, %b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v306 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v307 = stablehlo.add %v305, %v306 : tensor<32x144x56x56xf32>
    %v308 = stablehlo.reshape %v307 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v309 = stablehlo.reshape %v308 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v310 = stablehlo.constant dense<0.0> : tensor<f32>
    %v311 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v312 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v313 = stablehlo.reduce(%v309 init: %v310) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v314 = stablehlo.broadcast_in_dim %v313, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v315 = stablehlo.divide %v314, %v311 : tensor<32x144x56x56xf32>
    %v316 = stablehlo.subtract %v309, %v315 : tensor<32x144x56x56xf32>
    %v317 = stablehlo.multiply %v316, %v316 : tensor<32x144x56x56xf32>
    %v318 = stablehlo.reduce(%v317 init: %v310) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v319 = stablehlo.broadcast_in_dim %v318, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v320 = stablehlo.divide %v319, %v311 : tensor<32x144x56x56xf32>
    %v321 = stablehlo.add %v320, %v312 : tensor<32x144x56x56xf32>
    %v322 = stablehlo.rsqrt %v321 : tensor<32x144x56x56xf32>
    %v323 = stablehlo.multiply %v316, %v322 : tensor<32x144x56x56xf32>
    %v324 = stablehlo.broadcast_in_dim %b4eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v325 = stablehlo.broadcast_in_dim %b4ebt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v326 = stablehlo.multiply %v323, %v324 : tensor<32x144x56x56xf32>
    %v327 = stablehlo.add %v326, %v325 : tensor<32x144x56x56xf32>
    %v328 = stablehlo.reshape %v327 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v329 = stablehlo.reshape %v328 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v330 = stablehlo.logistic %v329 : tensor<32x144x56x56xf32>
    %v331 = stablehlo.multiply %v329, %v330 : tensor<32x144x56x56xf32>
    %v332 = stablehlo.reshape %v331 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v333 = stablehlo.reshape %v332 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v334 = stablehlo.convolution(%v333, %b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 2], [1, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x5x5xf32>) -> tensor<32x144x28x28xf32>
    %v335 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v336 = stablehlo.add %v334, %v335 : tensor<32x144x28x28xf32>
    %v337 = stablehlo.reshape %v336 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v338 = stablehlo.reshape %v337 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v339 = stablehlo.constant dense<0.0> : tensor<f32>
    %v340 = stablehlo.constant dense<25088.0> : tensor<32x144x28x28xf32>
    %v341 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v342 = stablehlo.reduce(%v338 init: %v339) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v343 = stablehlo.broadcast_in_dim %v342, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v344 = stablehlo.divide %v343, %v340 : tensor<32x144x28x28xf32>
    %v345 = stablehlo.subtract %v338, %v344 : tensor<32x144x28x28xf32>
    %v346 = stablehlo.multiply %v345, %v345 : tensor<32x144x28x28xf32>
    %v347 = stablehlo.reduce(%v346 init: %v339) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v348 = stablehlo.broadcast_in_dim %v347, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v349 = stablehlo.divide %v348, %v340 : tensor<32x144x28x28xf32>
    %v350 = stablehlo.add %v349, %v341 : tensor<32x144x28x28xf32>
    %v351 = stablehlo.rsqrt %v350 : tensor<32x144x28x28xf32>
    %v352 = stablehlo.multiply %v345, %v351 : tensor<32x144x28x28xf32>
    %v353 = stablehlo.broadcast_in_dim %b4dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v354 = stablehlo.broadcast_in_dim %b4dbt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v355 = stablehlo.multiply %v352, %v353 : tensor<32x144x28x28xf32>
    %v356 = stablehlo.add %v355, %v354 : tensor<32x144x28x28xf32>
    %v357 = stablehlo.reshape %v356 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v358 = stablehlo.reshape %v357 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v359 = stablehlo.logistic %v358 : tensor<32x144x28x28xf32>
    %v360 = stablehlo.multiply %v358, %v359 : tensor<32x144x28x28xf32>
    %v361 = stablehlo.reshape %v360 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v362 = stablehlo.reshape %v361 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v363 = stablehlo.constant dense<0.0> : tensor<f32>
    %v364 = stablehlo.reduce(%v362 init: %v363) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v365 = stablehlo.constant dense<784.0> : tensor<32x144xf32>
    %v366 = stablehlo.divide %v364, %v365 : tensor<32x144xf32>
    %v367 = stablehlo.dot_general %v366, %b4zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v368 = stablehlo.broadcast_in_dim %b4zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v369 = stablehlo.add %v367, %v368 : tensor<32x6xf32>
    %v370 = stablehlo.logistic %v369 : tensor<32x6xf32>
    %v371 = stablehlo.multiply %v369, %v370 : tensor<32x6xf32>
    %v372 = stablehlo.dot_general %v371, %b4zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v373 = stablehlo.broadcast_in_dim %b4zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v374 = stablehlo.add %v372, %v373 : tensor<32x144xf32>
    %v375 = stablehlo.logistic %v374 : tensor<32x144xf32>
    %v376 = stablehlo.broadcast_in_dim %v375, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v377 = stablehlo.multiply %v362, %v376 : tensor<32x144x28x28xf32>
    %v378 = stablehlo.reshape %v377 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v379 = stablehlo.reshape %v378 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v380 = stablehlo.convolution(%v379, %b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x28x28xf32>, tensor<40x144x1x1xf32>) -> tensor<32x40x28x28xf32>
    %v381 = stablehlo.broadcast_in_dim %zb40, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v382 = stablehlo.add %v380, %v381 : tensor<32x40x28x28xf32>
    %v383 = stablehlo.reshape %v382 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v384 = stablehlo.reshape %v383 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v385 = stablehlo.constant dense<0.0> : tensor<f32>
    %v386 = stablehlo.constant dense<25088.0> : tensor<32x40x28x28xf32>
    %v387 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v388 = stablehlo.reduce(%v384 init: %v385) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v389 = stablehlo.broadcast_in_dim %v388, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v390 = stablehlo.divide %v389, %v386 : tensor<32x40x28x28xf32>
    %v391 = stablehlo.subtract %v384, %v390 : tensor<32x40x28x28xf32>
    %v392 = stablehlo.multiply %v391, %v391 : tensor<32x40x28x28xf32>
    %v393 = stablehlo.reduce(%v392 init: %v385) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v394 = stablehlo.broadcast_in_dim %v393, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v395 = stablehlo.divide %v394, %v386 : tensor<32x40x28x28xf32>
    %v396 = stablehlo.add %v395, %v387 : tensor<32x40x28x28xf32>
    %v397 = stablehlo.rsqrt %v396 : tensor<32x40x28x28xf32>
    %v398 = stablehlo.multiply %v391, %v397 : tensor<32x40x28x28xf32>
    %v399 = stablehlo.broadcast_in_dim %b4pg, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v400 = stablehlo.broadcast_in_dim %b4pbt, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v401 = stablehlo.multiply %v398, %v399 : tensor<32x40x28x28xf32>
    %v402 = stablehlo.add %v401, %v400 : tensor<32x40x28x28xf32>
    %v403 = stablehlo.reshape %v402 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v404 = stablehlo.reshape %v403 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v405 = stablehlo.convolution(%v404, %b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<32x240x28x28xf32>
    %v406 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v407 = stablehlo.add %v405, %v406 : tensor<32x240x28x28xf32>
    %v408 = stablehlo.reshape %v407 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v409 = stablehlo.reshape %v408 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v410 = stablehlo.constant dense<0.0> : tensor<f32>
    %v411 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v412 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v413 = stablehlo.reduce(%v409 init: %v410) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v414 = stablehlo.broadcast_in_dim %v413, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v415 = stablehlo.divide %v414, %v411 : tensor<32x240x28x28xf32>
    %v416 = stablehlo.subtract %v409, %v415 : tensor<32x240x28x28xf32>
    %v417 = stablehlo.multiply %v416, %v416 : tensor<32x240x28x28xf32>
    %v418 = stablehlo.reduce(%v417 init: %v410) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v419 = stablehlo.broadcast_in_dim %v418, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v420 = stablehlo.divide %v419, %v411 : tensor<32x240x28x28xf32>
    %v421 = stablehlo.add %v420, %v412 : tensor<32x240x28x28xf32>
    %v422 = stablehlo.rsqrt %v421 : tensor<32x240x28x28xf32>
    %v423 = stablehlo.multiply %v416, %v422 : tensor<32x240x28x28xf32>
    %v424 = stablehlo.broadcast_in_dim %b5eg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v425 = stablehlo.broadcast_in_dim %b5ebt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v426 = stablehlo.multiply %v423, %v424 : tensor<32x240x28x28xf32>
    %v427 = stablehlo.add %v426, %v425 : tensor<32x240x28x28xf32>
    %v428 = stablehlo.reshape %v427 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v429 = stablehlo.reshape %v428 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v430 = stablehlo.logistic %v429 : tensor<32x240x28x28xf32>
    %v431 = stablehlo.multiply %v429, %v430 : tensor<32x240x28x28xf32>
    %v432 = stablehlo.reshape %v431 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v433 = stablehlo.reshape %v432 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v434 = stablehlo.convolution(%v433, %b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x5x5xf32>) -> tensor<32x240x28x28xf32>
    %v435 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v436 = stablehlo.add %v434, %v435 : tensor<32x240x28x28xf32>
    %v437 = stablehlo.reshape %v436 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v438 = stablehlo.reshape %v437 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v439 = stablehlo.constant dense<0.0> : tensor<f32>
    %v440 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v441 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v442 = stablehlo.reduce(%v438 init: %v439) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v443 = stablehlo.broadcast_in_dim %v442, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v444 = stablehlo.divide %v443, %v440 : tensor<32x240x28x28xf32>
    %v445 = stablehlo.subtract %v438, %v444 : tensor<32x240x28x28xf32>
    %v446 = stablehlo.multiply %v445, %v445 : tensor<32x240x28x28xf32>
    %v447 = stablehlo.reduce(%v446 init: %v439) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v448 = stablehlo.broadcast_in_dim %v447, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v449 = stablehlo.divide %v448, %v440 : tensor<32x240x28x28xf32>
    %v450 = stablehlo.add %v449, %v441 : tensor<32x240x28x28xf32>
    %v451 = stablehlo.rsqrt %v450 : tensor<32x240x28x28xf32>
    %v452 = stablehlo.multiply %v445, %v451 : tensor<32x240x28x28xf32>
    %v453 = stablehlo.broadcast_in_dim %b5dg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v454 = stablehlo.broadcast_in_dim %b5dbt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v455 = stablehlo.multiply %v452, %v453 : tensor<32x240x28x28xf32>
    %v456 = stablehlo.add %v455, %v454 : tensor<32x240x28x28xf32>
    %v457 = stablehlo.reshape %v456 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v458 = stablehlo.reshape %v457 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v459 = stablehlo.logistic %v458 : tensor<32x240x28x28xf32>
    %v460 = stablehlo.multiply %v458, %v459 : tensor<32x240x28x28xf32>
    %v461 = stablehlo.reshape %v460 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v462 = stablehlo.reshape %v461 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v463 = stablehlo.constant dense<0.0> : tensor<f32>
    %v464 = stablehlo.reduce(%v462 init: %v463) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v465 = stablehlo.constant dense<784.0> : tensor<32x240xf32>
    %v466 = stablehlo.divide %v464, %v465 : tensor<32x240xf32>
    %v467 = stablehlo.dot_general %v466, %b5zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v468 = stablehlo.broadcast_in_dim %b5zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v469 = stablehlo.add %v467, %v468 : tensor<32x10xf32>
    %v470 = stablehlo.logistic %v469 : tensor<32x10xf32>
    %v471 = stablehlo.multiply %v469, %v470 : tensor<32x10xf32>
    %v472 = stablehlo.dot_general %v471, %b5zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v473 = stablehlo.broadcast_in_dim %b5zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v474 = stablehlo.add %v472, %v473 : tensor<32x240xf32>
    %v475 = stablehlo.logistic %v474 : tensor<32x240xf32>
    %v476 = stablehlo.broadcast_in_dim %v475, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x28x28xf32>
    %v477 = stablehlo.multiply %v462, %v476 : tensor<32x240x28x28xf32>
    %v478 = stablehlo.reshape %v477 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v479 = stablehlo.reshape %v478 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v480 = stablehlo.convolution(%v479, %b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x28x28xf32>, tensor<40x240x1x1xf32>) -> tensor<32x40x28x28xf32>
    %v481 = stablehlo.broadcast_in_dim %zb40, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v482 = stablehlo.add %v480, %v481 : tensor<32x40x28x28xf32>
    %v483 = stablehlo.reshape %v482 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v484 = stablehlo.reshape %v483 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v485 = stablehlo.constant dense<0.0> : tensor<f32>
    %v486 = stablehlo.constant dense<25088.0> : tensor<32x40x28x28xf32>
    %v487 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v488 = stablehlo.reduce(%v484 init: %v485) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v489 = stablehlo.broadcast_in_dim %v488, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v490 = stablehlo.divide %v489, %v486 : tensor<32x40x28x28xf32>
    %v491 = stablehlo.subtract %v484, %v490 : tensor<32x40x28x28xf32>
    %v492 = stablehlo.multiply %v491, %v491 : tensor<32x40x28x28xf32>
    %v493 = stablehlo.reduce(%v492 init: %v485) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v494 = stablehlo.broadcast_in_dim %v493, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v495 = stablehlo.divide %v494, %v486 : tensor<32x40x28x28xf32>
    %v496 = stablehlo.add %v495, %v487 : tensor<32x40x28x28xf32>
    %v497 = stablehlo.rsqrt %v496 : tensor<32x40x28x28xf32>
    %v498 = stablehlo.multiply %v491, %v497 : tensor<32x40x28x28xf32>
    %v499 = stablehlo.broadcast_in_dim %b5pg, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v500 = stablehlo.broadcast_in_dim %b5pbt, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v501 = stablehlo.multiply %v498, %v499 : tensor<32x40x28x28xf32>
    %v502 = stablehlo.add %v501, %v500 : tensor<32x40x28x28xf32>
    %v503 = stablehlo.reshape %v502 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v504 = stablehlo.reshape %v503 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v505 = stablehlo.reshape %v403 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v506 = stablehlo.add %v504, %v505 : tensor<32x40x28x28xf32>
    %v507 = stablehlo.reshape %v506 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v508 = stablehlo.reshape %v507 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v509 = stablehlo.convolution(%v508, %b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<32x240x28x28xf32>
    %v510 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v511 = stablehlo.add %v509, %v510 : tensor<32x240x28x28xf32>
    %v512 = stablehlo.reshape %v511 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v513 = stablehlo.reshape %v512 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v514 = stablehlo.constant dense<0.0> : tensor<f32>
    %v515 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v516 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v517 = stablehlo.reduce(%v513 init: %v514) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v518 = stablehlo.broadcast_in_dim %v517, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v519 = stablehlo.divide %v518, %v515 : tensor<32x240x28x28xf32>
    %v520 = stablehlo.subtract %v513, %v519 : tensor<32x240x28x28xf32>
    %v521 = stablehlo.multiply %v520, %v520 : tensor<32x240x28x28xf32>
    %v522 = stablehlo.reduce(%v521 init: %v514) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v523 = stablehlo.broadcast_in_dim %v522, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v524 = stablehlo.divide %v523, %v515 : tensor<32x240x28x28xf32>
    %v525 = stablehlo.add %v524, %v516 : tensor<32x240x28x28xf32>
    %v526 = stablehlo.rsqrt %v525 : tensor<32x240x28x28xf32>
    %v527 = stablehlo.multiply %v520, %v526 : tensor<32x240x28x28xf32>
    %v528 = stablehlo.broadcast_in_dim %b6eg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v529 = stablehlo.broadcast_in_dim %b6ebt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v530 = stablehlo.multiply %v527, %v528 : tensor<32x240x28x28xf32>
    %v531 = stablehlo.add %v530, %v529 : tensor<32x240x28x28xf32>
    %v532 = stablehlo.reshape %v531 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v533 = stablehlo.reshape %v532 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v534 = stablehlo.logistic %v533 : tensor<32x240x28x28xf32>
    %v535 = stablehlo.multiply %v533, %v534 : tensor<32x240x28x28xf32>
    %v536 = stablehlo.reshape %v535 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v537 = stablehlo.reshape %v536 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v538 = stablehlo.convolution(%v537, %b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x3x3xf32>) -> tensor<32x240x14x14xf32>
    %v539 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v540 = stablehlo.add %v538, %v539 : tensor<32x240x14x14xf32>
    %v541 = stablehlo.reshape %v540 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v542 = stablehlo.reshape %v541 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v543 = stablehlo.constant dense<0.0> : tensor<f32>
    %v544 = stablehlo.constant dense<6272.0> : tensor<32x240x14x14xf32>
    %v545 = stablehlo.constant dense<1.0e-5> : tensor<32x240x14x14xf32>
    %v546 = stablehlo.reduce(%v542 init: %v543) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v547 = stablehlo.broadcast_in_dim %v546, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v548 = stablehlo.divide %v547, %v544 : tensor<32x240x14x14xf32>
    %v549 = stablehlo.subtract %v542, %v548 : tensor<32x240x14x14xf32>
    %v550 = stablehlo.multiply %v549, %v549 : tensor<32x240x14x14xf32>
    %v551 = stablehlo.reduce(%v550 init: %v543) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v552 = stablehlo.broadcast_in_dim %v551, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v553 = stablehlo.divide %v552, %v544 : tensor<32x240x14x14xf32>
    %v554 = stablehlo.add %v553, %v545 : tensor<32x240x14x14xf32>
    %v555 = stablehlo.rsqrt %v554 : tensor<32x240x14x14xf32>
    %v556 = stablehlo.multiply %v549, %v555 : tensor<32x240x14x14xf32>
    %v557 = stablehlo.broadcast_in_dim %b6dg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v558 = stablehlo.broadcast_in_dim %b6dbt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v559 = stablehlo.multiply %v556, %v557 : tensor<32x240x14x14xf32>
    %v560 = stablehlo.add %v559, %v558 : tensor<32x240x14x14xf32>
    %v561 = stablehlo.reshape %v560 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v562 = stablehlo.reshape %v561 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v563 = stablehlo.logistic %v562 : tensor<32x240x14x14xf32>
    %v564 = stablehlo.multiply %v562, %v563 : tensor<32x240x14x14xf32>
    %v565 = stablehlo.reshape %v564 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v566 = stablehlo.reshape %v565 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v567 = stablehlo.constant dense<0.0> : tensor<f32>
    %v568 = stablehlo.reduce(%v566 init: %v567) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v569 = stablehlo.constant dense<196.0> : tensor<32x240xf32>
    %v570 = stablehlo.divide %v568, %v569 : tensor<32x240xf32>
    %v571 = stablehlo.dot_general %v570, %b6zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v572 = stablehlo.broadcast_in_dim %b6zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v573 = stablehlo.add %v571, %v572 : tensor<32x10xf32>
    %v574 = stablehlo.logistic %v573 : tensor<32x10xf32>
    %v575 = stablehlo.multiply %v573, %v574 : tensor<32x10xf32>
    %v576 = stablehlo.dot_general %v575, %b6zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v577 = stablehlo.broadcast_in_dim %b6zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v578 = stablehlo.add %v576, %v577 : tensor<32x240xf32>
    %v579 = stablehlo.logistic %v578 : tensor<32x240xf32>
    %v580 = stablehlo.broadcast_in_dim %v579, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x14x14xf32>
    %v581 = stablehlo.multiply %v566, %v580 : tensor<32x240x14x14xf32>
    %v582 = stablehlo.reshape %v581 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v583 = stablehlo.reshape %v582 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v584 = stablehlo.convolution(%v583, %b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x14x14xf32>, tensor<80x240x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v585 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v586 = stablehlo.add %v584, %v585 : tensor<32x80x14x14xf32>
    %v587 = stablehlo.reshape %v586 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v588 = stablehlo.reshape %v587 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v589 = stablehlo.constant dense<0.0> : tensor<f32>
    %v590 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v591 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v592 = stablehlo.reduce(%v588 init: %v589) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v593 = stablehlo.broadcast_in_dim %v592, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v594 = stablehlo.divide %v593, %v590 : tensor<32x80x14x14xf32>
    %v595 = stablehlo.subtract %v588, %v594 : tensor<32x80x14x14xf32>
    %v596 = stablehlo.multiply %v595, %v595 : tensor<32x80x14x14xf32>
    %v597 = stablehlo.reduce(%v596 init: %v589) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v598 = stablehlo.broadcast_in_dim %v597, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v599 = stablehlo.divide %v598, %v590 : tensor<32x80x14x14xf32>
    %v600 = stablehlo.add %v599, %v591 : tensor<32x80x14x14xf32>
    %v601 = stablehlo.rsqrt %v600 : tensor<32x80x14x14xf32>
    %v602 = stablehlo.multiply %v595, %v601 : tensor<32x80x14x14xf32>
    %v603 = stablehlo.broadcast_in_dim %b6pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v604 = stablehlo.broadcast_in_dim %b6pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v605 = stablehlo.multiply %v602, %v603 : tensor<32x80x14x14xf32>
    %v606 = stablehlo.add %v605, %v604 : tensor<32x80x14x14xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v608 = stablehlo.reshape %v607 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v609 = stablehlo.convolution(%v608, %b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v610 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v611 = stablehlo.add %v609, %v610 : tensor<32x480x14x14xf32>
    %v612 = stablehlo.reshape %v611 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v613 = stablehlo.reshape %v612 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v614 = stablehlo.constant dense<0.0> : tensor<f32>
    %v615 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v616 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v617 = stablehlo.reduce(%v613 init: %v614) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v618 = stablehlo.broadcast_in_dim %v617, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v619 = stablehlo.divide %v618, %v615 : tensor<32x480x14x14xf32>
    %v620 = stablehlo.subtract %v613, %v619 : tensor<32x480x14x14xf32>
    %v621 = stablehlo.multiply %v620, %v620 : tensor<32x480x14x14xf32>
    %v622 = stablehlo.reduce(%v621 init: %v614) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v623 = stablehlo.broadcast_in_dim %v622, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v624 = stablehlo.divide %v623, %v615 : tensor<32x480x14x14xf32>
    %v625 = stablehlo.add %v624, %v616 : tensor<32x480x14x14xf32>
    %v626 = stablehlo.rsqrt %v625 : tensor<32x480x14x14xf32>
    %v627 = stablehlo.multiply %v620, %v626 : tensor<32x480x14x14xf32>
    %v628 = stablehlo.broadcast_in_dim %b7eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v629 = stablehlo.broadcast_in_dim %b7ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v630 = stablehlo.multiply %v627, %v628 : tensor<32x480x14x14xf32>
    %v631 = stablehlo.add %v630, %v629 : tensor<32x480x14x14xf32>
    %v632 = stablehlo.reshape %v631 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v633 = stablehlo.reshape %v632 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v634 = stablehlo.logistic %v633 : tensor<32x480x14x14xf32>
    %v635 = stablehlo.multiply %v633, %v634 : tensor<32x480x14x14xf32>
    %v636 = stablehlo.reshape %v635 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v637 = stablehlo.reshape %v636 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v638 = stablehlo.convolution(%v637, %b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %v639 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v640 = stablehlo.add %v638, %v639 : tensor<32x480x14x14xf32>
    %v641 = stablehlo.reshape %v640 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v642 = stablehlo.reshape %v641 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v643 = stablehlo.constant dense<0.0> : tensor<f32>
    %v644 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v645 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v646 = stablehlo.reduce(%v642 init: %v643) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v647 = stablehlo.broadcast_in_dim %v646, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v648 = stablehlo.divide %v647, %v644 : tensor<32x480x14x14xf32>
    %v649 = stablehlo.subtract %v642, %v648 : tensor<32x480x14x14xf32>
    %v650 = stablehlo.multiply %v649, %v649 : tensor<32x480x14x14xf32>
    %v651 = stablehlo.reduce(%v650 init: %v643) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v652 = stablehlo.broadcast_in_dim %v651, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v653 = stablehlo.divide %v652, %v644 : tensor<32x480x14x14xf32>
    %v654 = stablehlo.add %v653, %v645 : tensor<32x480x14x14xf32>
    %v655 = stablehlo.rsqrt %v654 : tensor<32x480x14x14xf32>
    %v656 = stablehlo.multiply %v649, %v655 : tensor<32x480x14x14xf32>
    %v657 = stablehlo.broadcast_in_dim %b7dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v658 = stablehlo.broadcast_in_dim %b7dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v659 = stablehlo.multiply %v656, %v657 : tensor<32x480x14x14xf32>
    %v660 = stablehlo.add %v659, %v658 : tensor<32x480x14x14xf32>
    %v661 = stablehlo.reshape %v660 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v662 = stablehlo.reshape %v661 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v663 = stablehlo.logistic %v662 : tensor<32x480x14x14xf32>
    %v664 = stablehlo.multiply %v662, %v663 : tensor<32x480x14x14xf32>
    %v665 = stablehlo.reshape %v664 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v666 = stablehlo.reshape %v665 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v667 = stablehlo.constant dense<0.0> : tensor<f32>
    %v668 = stablehlo.reduce(%v666 init: %v667) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v669 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v670 = stablehlo.divide %v668, %v669 : tensor<32x480xf32>
    %v671 = stablehlo.dot_general %v670, %b7zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v672 = stablehlo.broadcast_in_dim %b7zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v673 = stablehlo.add %v671, %v672 : tensor<32x20xf32>
    %v674 = stablehlo.logistic %v673 : tensor<32x20xf32>
    %v675 = stablehlo.multiply %v673, %v674 : tensor<32x20xf32>
    %v676 = stablehlo.dot_general %v675, %b7zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v677 = stablehlo.broadcast_in_dim %b7zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v678 = stablehlo.add %v676, %v677 : tensor<32x480xf32>
    %v679 = stablehlo.logistic %v678 : tensor<32x480xf32>
    %v680 = stablehlo.broadcast_in_dim %v679, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v681 = stablehlo.multiply %v666, %v680 : tensor<32x480x14x14xf32>
    %v682 = stablehlo.reshape %v681 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v683 = stablehlo.reshape %v682 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v684 = stablehlo.convolution(%v683, %b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v685 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v686 = stablehlo.add %v684, %v685 : tensor<32x80x14x14xf32>
    %v687 = stablehlo.reshape %v686 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v688 = stablehlo.reshape %v687 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v689 = stablehlo.constant dense<0.0> : tensor<f32>
    %v690 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v691 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v692 = stablehlo.reduce(%v688 init: %v689) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v693 = stablehlo.broadcast_in_dim %v692, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v694 = stablehlo.divide %v693, %v690 : tensor<32x80x14x14xf32>
    %v695 = stablehlo.subtract %v688, %v694 : tensor<32x80x14x14xf32>
    %v696 = stablehlo.multiply %v695, %v695 : tensor<32x80x14x14xf32>
    %v697 = stablehlo.reduce(%v696 init: %v689) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v698 = stablehlo.broadcast_in_dim %v697, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v699 = stablehlo.divide %v698, %v690 : tensor<32x80x14x14xf32>
    %v700 = stablehlo.add %v699, %v691 : tensor<32x80x14x14xf32>
    %v701 = stablehlo.rsqrt %v700 : tensor<32x80x14x14xf32>
    %v702 = stablehlo.multiply %v695, %v701 : tensor<32x80x14x14xf32>
    %v703 = stablehlo.broadcast_in_dim %b7pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v704 = stablehlo.broadcast_in_dim %b7pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v705 = stablehlo.multiply %v702, %v703 : tensor<32x80x14x14xf32>
    %v706 = stablehlo.add %v705, %v704 : tensor<32x80x14x14xf32>
    %v707 = stablehlo.reshape %v706 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v708 = stablehlo.reshape %v707 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v709 = stablehlo.reshape %v607 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v710 = stablehlo.add %v708, %v709 : tensor<32x80x14x14xf32>
    %v711 = stablehlo.reshape %v710 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v712 = stablehlo.reshape %v711 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v713 = stablehlo.convolution(%v712, %b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v714 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v715 = stablehlo.add %v713, %v714 : tensor<32x480x14x14xf32>
    %v716 = stablehlo.reshape %v715 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v717 = stablehlo.reshape %v716 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v718 = stablehlo.constant dense<0.0> : tensor<f32>
    %v719 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v720 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v721 = stablehlo.reduce(%v717 init: %v718) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v722 = stablehlo.broadcast_in_dim %v721, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v723 = stablehlo.divide %v722, %v719 : tensor<32x480x14x14xf32>
    %v724 = stablehlo.subtract %v717, %v723 : tensor<32x480x14x14xf32>
    %v725 = stablehlo.multiply %v724, %v724 : tensor<32x480x14x14xf32>
    %v726 = stablehlo.reduce(%v725 init: %v718) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v727 = stablehlo.broadcast_in_dim %v726, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v728 = stablehlo.divide %v727, %v719 : tensor<32x480x14x14xf32>
    %v729 = stablehlo.add %v728, %v720 : tensor<32x480x14x14xf32>
    %v730 = stablehlo.rsqrt %v729 : tensor<32x480x14x14xf32>
    %v731 = stablehlo.multiply %v724, %v730 : tensor<32x480x14x14xf32>
    %v732 = stablehlo.broadcast_in_dim %b8eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v733 = stablehlo.broadcast_in_dim %b8ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v734 = stablehlo.multiply %v731, %v732 : tensor<32x480x14x14xf32>
    %v735 = stablehlo.add %v734, %v733 : tensor<32x480x14x14xf32>
    %v736 = stablehlo.reshape %v735 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v737 = stablehlo.reshape %v736 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v738 = stablehlo.logistic %v737 : tensor<32x480x14x14xf32>
    %v739 = stablehlo.multiply %v737, %v738 : tensor<32x480x14x14xf32>
    %v740 = stablehlo.reshape %v739 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v741 = stablehlo.reshape %v740 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v742 = stablehlo.convolution(%v741, %b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %v743 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v744 = stablehlo.add %v742, %v743 : tensor<32x480x14x14xf32>
    %v745 = stablehlo.reshape %v744 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v746 = stablehlo.reshape %v745 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v747 = stablehlo.constant dense<0.0> : tensor<f32>
    %v748 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v749 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v750 = stablehlo.reduce(%v746 init: %v747) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v751 = stablehlo.broadcast_in_dim %v750, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v752 = stablehlo.divide %v751, %v748 : tensor<32x480x14x14xf32>
    %v753 = stablehlo.subtract %v746, %v752 : tensor<32x480x14x14xf32>
    %v754 = stablehlo.multiply %v753, %v753 : tensor<32x480x14x14xf32>
    %v755 = stablehlo.reduce(%v754 init: %v747) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v756 = stablehlo.broadcast_in_dim %v755, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v757 = stablehlo.divide %v756, %v748 : tensor<32x480x14x14xf32>
    %v758 = stablehlo.add %v757, %v749 : tensor<32x480x14x14xf32>
    %v759 = stablehlo.rsqrt %v758 : tensor<32x480x14x14xf32>
    %v760 = stablehlo.multiply %v753, %v759 : tensor<32x480x14x14xf32>
    %v761 = stablehlo.broadcast_in_dim %b8dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v762 = stablehlo.broadcast_in_dim %b8dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v763 = stablehlo.multiply %v760, %v761 : tensor<32x480x14x14xf32>
    %v764 = stablehlo.add %v763, %v762 : tensor<32x480x14x14xf32>
    %v765 = stablehlo.reshape %v764 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v766 = stablehlo.reshape %v765 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v767 = stablehlo.logistic %v766 : tensor<32x480x14x14xf32>
    %v768 = stablehlo.multiply %v766, %v767 : tensor<32x480x14x14xf32>
    %v769 = stablehlo.reshape %v768 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v770 = stablehlo.reshape %v769 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v771 = stablehlo.constant dense<0.0> : tensor<f32>
    %v772 = stablehlo.reduce(%v770 init: %v771) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v773 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v774 = stablehlo.divide %v772, %v773 : tensor<32x480xf32>
    %v775 = stablehlo.dot_general %v774, %b8zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v776 = stablehlo.broadcast_in_dim %b8zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v777 = stablehlo.add %v775, %v776 : tensor<32x20xf32>
    %v778 = stablehlo.logistic %v777 : tensor<32x20xf32>
    %v779 = stablehlo.multiply %v777, %v778 : tensor<32x20xf32>
    %v780 = stablehlo.dot_general %v779, %b8zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v781 = stablehlo.broadcast_in_dim %b8zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v782 = stablehlo.add %v780, %v781 : tensor<32x480xf32>
    %v783 = stablehlo.logistic %v782 : tensor<32x480xf32>
    %v784 = stablehlo.broadcast_in_dim %v783, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v785 = stablehlo.multiply %v770, %v784 : tensor<32x480x14x14xf32>
    %v786 = stablehlo.reshape %v785 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v787 = stablehlo.reshape %v786 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v788 = stablehlo.convolution(%v787, %b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v789 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v790 = stablehlo.add %v788, %v789 : tensor<32x80x14x14xf32>
    %v791 = stablehlo.reshape %v790 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v792 = stablehlo.reshape %v791 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v793 = stablehlo.constant dense<0.0> : tensor<f32>
    %v794 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v795 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v796 = stablehlo.reduce(%v792 init: %v793) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v797 = stablehlo.broadcast_in_dim %v796, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v798 = stablehlo.divide %v797, %v794 : tensor<32x80x14x14xf32>
    %v799 = stablehlo.subtract %v792, %v798 : tensor<32x80x14x14xf32>
    %v800 = stablehlo.multiply %v799, %v799 : tensor<32x80x14x14xf32>
    %v801 = stablehlo.reduce(%v800 init: %v793) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v802 = stablehlo.broadcast_in_dim %v801, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v803 = stablehlo.divide %v802, %v794 : tensor<32x80x14x14xf32>
    %v804 = stablehlo.add %v803, %v795 : tensor<32x80x14x14xf32>
    %v805 = stablehlo.rsqrt %v804 : tensor<32x80x14x14xf32>
    %v806 = stablehlo.multiply %v799, %v805 : tensor<32x80x14x14xf32>
    %v807 = stablehlo.broadcast_in_dim %b8pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v808 = stablehlo.broadcast_in_dim %b8pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v809 = stablehlo.multiply %v806, %v807 : tensor<32x80x14x14xf32>
    %v810 = stablehlo.add %v809, %v808 : tensor<32x80x14x14xf32>
    %v811 = stablehlo.reshape %v810 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v812 = stablehlo.reshape %v811 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v813 = stablehlo.reshape %v711 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v814 = stablehlo.add %v812, %v813 : tensor<32x80x14x14xf32>
    %v815 = stablehlo.reshape %v814 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v816 = stablehlo.reshape %v815 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v817 = stablehlo.convolution(%v816, %b9eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v818 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v819 = stablehlo.add %v817, %v818 : tensor<32x480x14x14xf32>
    %v820 = stablehlo.reshape %v819 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v821 = stablehlo.reshape %v820 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v822 = stablehlo.constant dense<0.0> : tensor<f32>
    %v823 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v824 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v825 = stablehlo.reduce(%v821 init: %v822) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v826 = stablehlo.broadcast_in_dim %v825, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v827 = stablehlo.divide %v826, %v823 : tensor<32x480x14x14xf32>
    %v828 = stablehlo.subtract %v821, %v827 : tensor<32x480x14x14xf32>
    %v829 = stablehlo.multiply %v828, %v828 : tensor<32x480x14x14xf32>
    %v830 = stablehlo.reduce(%v829 init: %v822) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v831 = stablehlo.broadcast_in_dim %v830, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v832 = stablehlo.divide %v831, %v823 : tensor<32x480x14x14xf32>
    %v833 = stablehlo.add %v832, %v824 : tensor<32x480x14x14xf32>
    %v834 = stablehlo.rsqrt %v833 : tensor<32x480x14x14xf32>
    %v835 = stablehlo.multiply %v828, %v834 : tensor<32x480x14x14xf32>
    %v836 = stablehlo.broadcast_in_dim %b9eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v837 = stablehlo.broadcast_in_dim %b9ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v838 = stablehlo.multiply %v835, %v836 : tensor<32x480x14x14xf32>
    %v839 = stablehlo.add %v838, %v837 : tensor<32x480x14x14xf32>
    %v840 = stablehlo.reshape %v839 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v841 = stablehlo.reshape %v840 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v842 = stablehlo.logistic %v841 : tensor<32x480x14x14xf32>
    %v843 = stablehlo.multiply %v841, %v842 : tensor<32x480x14x14xf32>
    %v844 = stablehlo.reshape %v843 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v845 = stablehlo.reshape %v844 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v846 = stablehlo.convolution(%v845, %b9dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x5x5xf32>) -> tensor<32x480x14x14xf32>
    %v847 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v848 = stablehlo.add %v846, %v847 : tensor<32x480x14x14xf32>
    %v849 = stablehlo.reshape %v848 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v850 = stablehlo.reshape %v849 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v851 = stablehlo.constant dense<0.0> : tensor<f32>
    %v852 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v853 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v854 = stablehlo.reduce(%v850 init: %v851) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v855 = stablehlo.broadcast_in_dim %v854, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v856 = stablehlo.divide %v855, %v852 : tensor<32x480x14x14xf32>
    %v857 = stablehlo.subtract %v850, %v856 : tensor<32x480x14x14xf32>
    %v858 = stablehlo.multiply %v857, %v857 : tensor<32x480x14x14xf32>
    %v859 = stablehlo.reduce(%v858 init: %v851) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v860 = stablehlo.broadcast_in_dim %v859, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v861 = stablehlo.divide %v860, %v852 : tensor<32x480x14x14xf32>
    %v862 = stablehlo.add %v861, %v853 : tensor<32x480x14x14xf32>
    %v863 = stablehlo.rsqrt %v862 : tensor<32x480x14x14xf32>
    %v864 = stablehlo.multiply %v857, %v863 : tensor<32x480x14x14xf32>
    %v865 = stablehlo.broadcast_in_dim %b9dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v866 = stablehlo.broadcast_in_dim %b9dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v867 = stablehlo.multiply %v864, %v865 : tensor<32x480x14x14xf32>
    %v868 = stablehlo.add %v867, %v866 : tensor<32x480x14x14xf32>
    %v869 = stablehlo.reshape %v868 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v870 = stablehlo.reshape %v869 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v871 = stablehlo.logistic %v870 : tensor<32x480x14x14xf32>
    %v872 = stablehlo.multiply %v870, %v871 : tensor<32x480x14x14xf32>
    %v873 = stablehlo.reshape %v872 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v874 = stablehlo.reshape %v873 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v875 = stablehlo.constant dense<0.0> : tensor<f32>
    %v876 = stablehlo.reduce(%v874 init: %v875) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v877 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v878 = stablehlo.divide %v876, %v877 : tensor<32x480xf32>
    %v879 = stablehlo.dot_general %v878, %b9zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v880 = stablehlo.broadcast_in_dim %b9zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v881 = stablehlo.add %v879, %v880 : tensor<32x20xf32>
    %v882 = stablehlo.logistic %v881 : tensor<32x20xf32>
    %v883 = stablehlo.multiply %v881, %v882 : tensor<32x20xf32>
    %v884 = stablehlo.dot_general %v883, %b9zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v885 = stablehlo.broadcast_in_dim %b9zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v886 = stablehlo.add %v884, %v885 : tensor<32x480xf32>
    %v887 = stablehlo.logistic %v886 : tensor<32x480xf32>
    %v888 = stablehlo.broadcast_in_dim %v887, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v889 = stablehlo.multiply %v874, %v888 : tensor<32x480x14x14xf32>
    %v890 = stablehlo.reshape %v889 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v891 = stablehlo.reshape %v890 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v892 = stablehlo.convolution(%v891, %b9pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<112x480x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v893 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v894 = stablehlo.add %v892, %v893 : tensor<32x112x14x14xf32>
    %v895 = stablehlo.reshape %v894 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v896 = stablehlo.reshape %v895 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v897 = stablehlo.constant dense<0.0> : tensor<f32>
    %v898 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v899 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v900 = stablehlo.reduce(%v896 init: %v897) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v901 = stablehlo.broadcast_in_dim %v900, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v902 = stablehlo.divide %v901, %v898 : tensor<32x112x14x14xf32>
    %v903 = stablehlo.subtract %v896, %v902 : tensor<32x112x14x14xf32>
    %v904 = stablehlo.multiply %v903, %v903 : tensor<32x112x14x14xf32>
    %v905 = stablehlo.reduce(%v904 init: %v897) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v906 = stablehlo.broadcast_in_dim %v905, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v907 = stablehlo.divide %v906, %v898 : tensor<32x112x14x14xf32>
    %v908 = stablehlo.add %v907, %v899 : tensor<32x112x14x14xf32>
    %v909 = stablehlo.rsqrt %v908 : tensor<32x112x14x14xf32>
    %v910 = stablehlo.multiply %v903, %v909 : tensor<32x112x14x14xf32>
    %v911 = stablehlo.broadcast_in_dim %b9pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v912 = stablehlo.broadcast_in_dim %b9pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v913 = stablehlo.multiply %v910, %v911 : tensor<32x112x14x14xf32>
    %v914 = stablehlo.add %v913, %v912 : tensor<32x112x14x14xf32>
    %v915 = stablehlo.reshape %v914 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v916 = stablehlo.reshape %v915 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v917 = stablehlo.convolution(%v916, %b10eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v918 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v919 = stablehlo.add %v917, %v918 : tensor<32x672x14x14xf32>
    %v920 = stablehlo.reshape %v919 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v921 = stablehlo.reshape %v920 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v922 = stablehlo.constant dense<0.0> : tensor<f32>
    %v923 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v924 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v925 = stablehlo.reduce(%v921 init: %v922) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v926 = stablehlo.broadcast_in_dim %v925, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v927 = stablehlo.divide %v926, %v923 : tensor<32x672x14x14xf32>
    %v928 = stablehlo.subtract %v921, %v927 : tensor<32x672x14x14xf32>
    %v929 = stablehlo.multiply %v928, %v928 : tensor<32x672x14x14xf32>
    %v930 = stablehlo.reduce(%v929 init: %v922) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v931 = stablehlo.broadcast_in_dim %v930, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v932 = stablehlo.divide %v931, %v923 : tensor<32x672x14x14xf32>
    %v933 = stablehlo.add %v932, %v924 : tensor<32x672x14x14xf32>
    %v934 = stablehlo.rsqrt %v933 : tensor<32x672x14x14xf32>
    %v935 = stablehlo.multiply %v928, %v934 : tensor<32x672x14x14xf32>
    %v936 = stablehlo.broadcast_in_dim %b10eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v937 = stablehlo.broadcast_in_dim %b10ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v938 = stablehlo.multiply %v935, %v936 : tensor<32x672x14x14xf32>
    %v939 = stablehlo.add %v938, %v937 : tensor<32x672x14x14xf32>
    %v940 = stablehlo.reshape %v939 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v941 = stablehlo.reshape %v940 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v942 = stablehlo.logistic %v941 : tensor<32x672x14x14xf32>
    %v943 = stablehlo.multiply %v941, %v942 : tensor<32x672x14x14xf32>
    %v944 = stablehlo.reshape %v943 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v945 = stablehlo.reshape %v944 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v946 = stablehlo.convolution(%v945, %b10dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %v947 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v948 = stablehlo.add %v946, %v947 : tensor<32x672x14x14xf32>
    %v949 = stablehlo.reshape %v948 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v950 = stablehlo.reshape %v949 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v951 = stablehlo.constant dense<0.0> : tensor<f32>
    %v952 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v953 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v954 = stablehlo.reduce(%v950 init: %v951) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v955 = stablehlo.broadcast_in_dim %v954, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v956 = stablehlo.divide %v955, %v952 : tensor<32x672x14x14xf32>
    %v957 = stablehlo.subtract %v950, %v956 : tensor<32x672x14x14xf32>
    %v958 = stablehlo.multiply %v957, %v957 : tensor<32x672x14x14xf32>
    %v959 = stablehlo.reduce(%v958 init: %v951) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v960 = stablehlo.broadcast_in_dim %v959, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v961 = stablehlo.divide %v960, %v952 : tensor<32x672x14x14xf32>
    %v962 = stablehlo.add %v961, %v953 : tensor<32x672x14x14xf32>
    %v963 = stablehlo.rsqrt %v962 : tensor<32x672x14x14xf32>
    %v964 = stablehlo.multiply %v957, %v963 : tensor<32x672x14x14xf32>
    %v965 = stablehlo.broadcast_in_dim %b10dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v966 = stablehlo.broadcast_in_dim %b10dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v967 = stablehlo.multiply %v964, %v965 : tensor<32x672x14x14xf32>
    %v968 = stablehlo.add %v967, %v966 : tensor<32x672x14x14xf32>
    %v969 = stablehlo.reshape %v968 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v970 = stablehlo.reshape %v969 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v971 = stablehlo.logistic %v970 : tensor<32x672x14x14xf32>
    %v972 = stablehlo.multiply %v970, %v971 : tensor<32x672x14x14xf32>
    %v973 = stablehlo.reshape %v972 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v974 = stablehlo.reshape %v973 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v975 = stablehlo.constant dense<0.0> : tensor<f32>
    %v976 = stablehlo.reduce(%v974 init: %v975) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v977 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v978 = stablehlo.divide %v976, %v977 : tensor<32x672xf32>
    %v979 = stablehlo.dot_general %v978, %b10zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v980 = stablehlo.broadcast_in_dim %b10zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v981 = stablehlo.add %v979, %v980 : tensor<32x28xf32>
    %v982 = stablehlo.logistic %v981 : tensor<32x28xf32>
    %v983 = stablehlo.multiply %v981, %v982 : tensor<32x28xf32>
    %v984 = stablehlo.dot_general %v983, %b10zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v985 = stablehlo.broadcast_in_dim %b10zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v986 = stablehlo.add %v984, %v985 : tensor<32x672xf32>
    %v987 = stablehlo.logistic %v986 : tensor<32x672xf32>
    %v988 = stablehlo.broadcast_in_dim %v987, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v989 = stablehlo.multiply %v974, %v988 : tensor<32x672x14x14xf32>
    %v990 = stablehlo.reshape %v989 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v991 = stablehlo.reshape %v990 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v992 = stablehlo.convolution(%v991, %b10pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v993 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v994 = stablehlo.add %v992, %v993 : tensor<32x112x14x14xf32>
    %v995 = stablehlo.reshape %v994 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v996 = stablehlo.reshape %v995 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v997 = stablehlo.constant dense<0.0> : tensor<f32>
    %v998 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v999 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v1000 = stablehlo.reduce(%v996 init: %v997) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1001 = stablehlo.broadcast_in_dim %v1000, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1002 = stablehlo.divide %v1001, %v998 : tensor<32x112x14x14xf32>
    %v1003 = stablehlo.subtract %v996, %v1002 : tensor<32x112x14x14xf32>
    %v1004 = stablehlo.multiply %v1003, %v1003 : tensor<32x112x14x14xf32>
    %v1005 = stablehlo.reduce(%v1004 init: %v997) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1006 = stablehlo.broadcast_in_dim %v1005, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1007 = stablehlo.divide %v1006, %v998 : tensor<32x112x14x14xf32>
    %v1008 = stablehlo.add %v1007, %v999 : tensor<32x112x14x14xf32>
    %v1009 = stablehlo.rsqrt %v1008 : tensor<32x112x14x14xf32>
    %v1010 = stablehlo.multiply %v1003, %v1009 : tensor<32x112x14x14xf32>
    %v1011 = stablehlo.broadcast_in_dim %b10pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1012 = stablehlo.broadcast_in_dim %b10pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1013 = stablehlo.multiply %v1010, %v1011 : tensor<32x112x14x14xf32>
    %v1014 = stablehlo.add %v1013, %v1012 : tensor<32x112x14x14xf32>
    %v1015 = stablehlo.reshape %v1014 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1016 = stablehlo.reshape %v1015 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1017 = stablehlo.reshape %v915 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1018 = stablehlo.add %v1016, %v1017 : tensor<32x112x14x14xf32>
    %v1019 = stablehlo.reshape %v1018 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1020 = stablehlo.reshape %v1019 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1021 = stablehlo.convolution(%v1020, %b11eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v1022 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1023 = stablehlo.add %v1021, %v1022 : tensor<32x672x14x14xf32>
    %v1024 = stablehlo.reshape %v1023 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1025 = stablehlo.reshape %v1024 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1026 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1027 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v1028 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v1029 = stablehlo.reduce(%v1025 init: %v1026) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1030 = stablehlo.broadcast_in_dim %v1029, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1031 = stablehlo.divide %v1030, %v1027 : tensor<32x672x14x14xf32>
    %v1032 = stablehlo.subtract %v1025, %v1031 : tensor<32x672x14x14xf32>
    %v1033 = stablehlo.multiply %v1032, %v1032 : tensor<32x672x14x14xf32>
    %v1034 = stablehlo.reduce(%v1033 init: %v1026) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1035 = stablehlo.broadcast_in_dim %v1034, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1036 = stablehlo.divide %v1035, %v1027 : tensor<32x672x14x14xf32>
    %v1037 = stablehlo.add %v1036, %v1028 : tensor<32x672x14x14xf32>
    %v1038 = stablehlo.rsqrt %v1037 : tensor<32x672x14x14xf32>
    %v1039 = stablehlo.multiply %v1032, %v1038 : tensor<32x672x14x14xf32>
    %v1040 = stablehlo.broadcast_in_dim %b11eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1041 = stablehlo.broadcast_in_dim %b11ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1042 = stablehlo.multiply %v1039, %v1040 : tensor<32x672x14x14xf32>
    %v1043 = stablehlo.add %v1042, %v1041 : tensor<32x672x14x14xf32>
    %v1044 = stablehlo.reshape %v1043 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1045 = stablehlo.reshape %v1044 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1046 = stablehlo.logistic %v1045 : tensor<32x672x14x14xf32>
    %v1047 = stablehlo.multiply %v1045, %v1046 : tensor<32x672x14x14xf32>
    %v1048 = stablehlo.reshape %v1047 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1049 = stablehlo.reshape %v1048 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1050 = stablehlo.convolution(%v1049, %b11dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %v1051 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1052 = stablehlo.add %v1050, %v1051 : tensor<32x672x14x14xf32>
    %v1053 = stablehlo.reshape %v1052 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1054 = stablehlo.reshape %v1053 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1055 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1056 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v1057 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v1058 = stablehlo.reduce(%v1054 init: %v1055) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1059 = stablehlo.broadcast_in_dim %v1058, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1060 = stablehlo.divide %v1059, %v1056 : tensor<32x672x14x14xf32>
    %v1061 = stablehlo.subtract %v1054, %v1060 : tensor<32x672x14x14xf32>
    %v1062 = stablehlo.multiply %v1061, %v1061 : tensor<32x672x14x14xf32>
    %v1063 = stablehlo.reduce(%v1062 init: %v1055) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1064 = stablehlo.broadcast_in_dim %v1063, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1065 = stablehlo.divide %v1064, %v1056 : tensor<32x672x14x14xf32>
    %v1066 = stablehlo.add %v1065, %v1057 : tensor<32x672x14x14xf32>
    %v1067 = stablehlo.rsqrt %v1066 : tensor<32x672x14x14xf32>
    %v1068 = stablehlo.multiply %v1061, %v1067 : tensor<32x672x14x14xf32>
    %v1069 = stablehlo.broadcast_in_dim %b11dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1070 = stablehlo.broadcast_in_dim %b11dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1071 = stablehlo.multiply %v1068, %v1069 : tensor<32x672x14x14xf32>
    %v1072 = stablehlo.add %v1071, %v1070 : tensor<32x672x14x14xf32>
    %v1073 = stablehlo.reshape %v1072 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1074 = stablehlo.reshape %v1073 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1075 = stablehlo.logistic %v1074 : tensor<32x672x14x14xf32>
    %v1076 = stablehlo.multiply %v1074, %v1075 : tensor<32x672x14x14xf32>
    %v1077 = stablehlo.reshape %v1076 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1078 = stablehlo.reshape %v1077 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1079 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1080 = stablehlo.reduce(%v1078 init: %v1079) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v1081 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v1082 = stablehlo.divide %v1080, %v1081 : tensor<32x672xf32>
    %v1083 = stablehlo.dot_general %v1082, %b11zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v1084 = stablehlo.broadcast_in_dim %b11zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v1085 = stablehlo.add %v1083, %v1084 : tensor<32x28xf32>
    %v1086 = stablehlo.logistic %v1085 : tensor<32x28xf32>
    %v1087 = stablehlo.multiply %v1085, %v1086 : tensor<32x28xf32>
    %v1088 = stablehlo.dot_general %v1087, %b11zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v1089 = stablehlo.broadcast_in_dim %b11zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v1090 = stablehlo.add %v1088, %v1089 : tensor<32x672xf32>
    %v1091 = stablehlo.logistic %v1090 : tensor<32x672xf32>
    %v1092 = stablehlo.broadcast_in_dim %v1091, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v1093 = stablehlo.multiply %v1078, %v1092 : tensor<32x672x14x14xf32>
    %v1094 = stablehlo.reshape %v1093 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1095 = stablehlo.reshape %v1094 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1096 = stablehlo.convolution(%v1095, %b11pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v1097 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1098 = stablehlo.add %v1096, %v1097 : tensor<32x112x14x14xf32>
    %v1099 = stablehlo.reshape %v1098 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1100 = stablehlo.reshape %v1099 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1101 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1102 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v1103 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v1104 = stablehlo.reduce(%v1100 init: %v1101) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1105 = stablehlo.broadcast_in_dim %v1104, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1106 = stablehlo.divide %v1105, %v1102 : tensor<32x112x14x14xf32>
    %v1107 = stablehlo.subtract %v1100, %v1106 : tensor<32x112x14x14xf32>
    %v1108 = stablehlo.multiply %v1107, %v1107 : tensor<32x112x14x14xf32>
    %v1109 = stablehlo.reduce(%v1108 init: %v1101) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1110 = stablehlo.broadcast_in_dim %v1109, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1111 = stablehlo.divide %v1110, %v1102 : tensor<32x112x14x14xf32>
    %v1112 = stablehlo.add %v1111, %v1103 : tensor<32x112x14x14xf32>
    %v1113 = stablehlo.rsqrt %v1112 : tensor<32x112x14x14xf32>
    %v1114 = stablehlo.multiply %v1107, %v1113 : tensor<32x112x14x14xf32>
    %v1115 = stablehlo.broadcast_in_dim %b11pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1116 = stablehlo.broadcast_in_dim %b11pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1117 = stablehlo.multiply %v1114, %v1115 : tensor<32x112x14x14xf32>
    %v1118 = stablehlo.add %v1117, %v1116 : tensor<32x112x14x14xf32>
    %v1119 = stablehlo.reshape %v1118 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1120 = stablehlo.reshape %v1119 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1121 = stablehlo.reshape %v1019 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1122 = stablehlo.add %v1120, %v1121 : tensor<32x112x14x14xf32>
    %v1123 = stablehlo.reshape %v1122 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1124 = stablehlo.reshape %v1123 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1125 = stablehlo.convolution(%v1124, %b12eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v1126 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1127 = stablehlo.add %v1125, %v1126 : tensor<32x672x14x14xf32>
    %v1128 = stablehlo.reshape %v1127 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1129 = stablehlo.reshape %v1128 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1130 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1131 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v1132 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v1133 = stablehlo.reduce(%v1129 init: %v1130) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1134 = stablehlo.broadcast_in_dim %v1133, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1135 = stablehlo.divide %v1134, %v1131 : tensor<32x672x14x14xf32>
    %v1136 = stablehlo.subtract %v1129, %v1135 : tensor<32x672x14x14xf32>
    %v1137 = stablehlo.multiply %v1136, %v1136 : tensor<32x672x14x14xf32>
    %v1138 = stablehlo.reduce(%v1137 init: %v1130) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1139 = stablehlo.broadcast_in_dim %v1138, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1140 = stablehlo.divide %v1139, %v1131 : tensor<32x672x14x14xf32>
    %v1141 = stablehlo.add %v1140, %v1132 : tensor<32x672x14x14xf32>
    %v1142 = stablehlo.rsqrt %v1141 : tensor<32x672x14x14xf32>
    %v1143 = stablehlo.multiply %v1136, %v1142 : tensor<32x672x14x14xf32>
    %v1144 = stablehlo.broadcast_in_dim %b12eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1145 = stablehlo.broadcast_in_dim %b12ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1146 = stablehlo.multiply %v1143, %v1144 : tensor<32x672x14x14xf32>
    %v1147 = stablehlo.add %v1146, %v1145 : tensor<32x672x14x14xf32>
    %v1148 = stablehlo.reshape %v1147 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1149 = stablehlo.reshape %v1148 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1150 = stablehlo.logistic %v1149 : tensor<32x672x14x14xf32>
    %v1151 = stablehlo.multiply %v1149, %v1150 : tensor<32x672x14x14xf32>
    %v1152 = stablehlo.reshape %v1151 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1153 = stablehlo.reshape %v1152 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1154 = stablehlo.convolution(%v1153, %b12dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 2], [1, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x7x7xf32>
    %v1155 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1156 = stablehlo.add %v1154, %v1155 : tensor<32x672x7x7xf32>
    %v1157 = stablehlo.reshape %v1156 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v1158 = stablehlo.reshape %v1157 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1159 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1160 = stablehlo.constant dense<1568.0> : tensor<32x672x7x7xf32>
    %v1161 = stablehlo.constant dense<1.0e-5> : tensor<32x672x7x7xf32>
    %v1162 = stablehlo.reduce(%v1158 init: %v1159) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v1163 = stablehlo.broadcast_in_dim %v1162, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1164 = stablehlo.divide %v1163, %v1160 : tensor<32x672x7x7xf32>
    %v1165 = stablehlo.subtract %v1158, %v1164 : tensor<32x672x7x7xf32>
    %v1166 = stablehlo.multiply %v1165, %v1165 : tensor<32x672x7x7xf32>
    %v1167 = stablehlo.reduce(%v1166 init: %v1159) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v1168 = stablehlo.broadcast_in_dim %v1167, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1169 = stablehlo.divide %v1168, %v1160 : tensor<32x672x7x7xf32>
    %v1170 = stablehlo.add %v1169, %v1161 : tensor<32x672x7x7xf32>
    %v1171 = stablehlo.rsqrt %v1170 : tensor<32x672x7x7xf32>
    %v1172 = stablehlo.multiply %v1165, %v1171 : tensor<32x672x7x7xf32>
    %v1173 = stablehlo.broadcast_in_dim %b12dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1174 = stablehlo.broadcast_in_dim %b12dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1175 = stablehlo.multiply %v1172, %v1173 : tensor<32x672x7x7xf32>
    %v1176 = stablehlo.add %v1175, %v1174 : tensor<32x672x7x7xf32>
    %v1177 = stablehlo.reshape %v1176 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v1178 = stablehlo.reshape %v1177 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1179 = stablehlo.logistic %v1178 : tensor<32x672x7x7xf32>
    %v1180 = stablehlo.multiply %v1178, %v1179 : tensor<32x672x7x7xf32>
    %v1181 = stablehlo.reshape %v1180 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v1182 = stablehlo.reshape %v1181 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1183 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1184 = stablehlo.reduce(%v1182 init: %v1183) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v1185 = stablehlo.constant dense<49.0> : tensor<32x672xf32>
    %v1186 = stablehlo.divide %v1184, %v1185 : tensor<32x672xf32>
    %v1187 = stablehlo.dot_general %v1186, %b12zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v1188 = stablehlo.broadcast_in_dim %b12zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v1189 = stablehlo.add %v1187, %v1188 : tensor<32x28xf32>
    %v1190 = stablehlo.logistic %v1189 : tensor<32x28xf32>
    %v1191 = stablehlo.multiply %v1189, %v1190 : tensor<32x28xf32>
    %v1192 = stablehlo.dot_general %v1191, %b12zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v1193 = stablehlo.broadcast_in_dim %b12zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v1194 = stablehlo.add %v1192, %v1193 : tensor<32x672xf32>
    %v1195 = stablehlo.logistic %v1194 : tensor<32x672xf32>
    %v1196 = stablehlo.broadcast_in_dim %v1195, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x7x7xf32>
    %v1197 = stablehlo.multiply %v1182, %v1196 : tensor<32x672x7x7xf32>
    %v1198 = stablehlo.reshape %v1197 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v1199 = stablehlo.reshape %v1198 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1200 = stablehlo.convolution(%v1199, %b12pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x7x7xf32>, tensor<192x672x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1201 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1202 = stablehlo.add %v1200, %v1201 : tensor<32x192x7x7xf32>
    %v1203 = stablehlo.reshape %v1202 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1204 = stablehlo.reshape %v1203 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1205 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1206 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v1207 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1208 = stablehlo.reduce(%v1204 init: %v1205) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1209 = stablehlo.broadcast_in_dim %v1208, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1210 = stablehlo.divide %v1209, %v1206 : tensor<32x192x7x7xf32>
    %v1211 = stablehlo.subtract %v1204, %v1210 : tensor<32x192x7x7xf32>
    %v1212 = stablehlo.multiply %v1211, %v1211 : tensor<32x192x7x7xf32>
    %v1213 = stablehlo.reduce(%v1212 init: %v1205) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1214 = stablehlo.broadcast_in_dim %v1213, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1215 = stablehlo.divide %v1214, %v1206 : tensor<32x192x7x7xf32>
    %v1216 = stablehlo.add %v1215, %v1207 : tensor<32x192x7x7xf32>
    %v1217 = stablehlo.rsqrt %v1216 : tensor<32x192x7x7xf32>
    %v1218 = stablehlo.multiply %v1211, %v1217 : tensor<32x192x7x7xf32>
    %v1219 = stablehlo.broadcast_in_dim %b12pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1220 = stablehlo.broadcast_in_dim %b12pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1221 = stablehlo.multiply %v1218, %v1219 : tensor<32x192x7x7xf32>
    %v1222 = stablehlo.add %v1221, %v1220 : tensor<32x192x7x7xf32>
    %v1223 = stablehlo.reshape %v1222 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1224 = stablehlo.reshape %v1223 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1225 = stablehlo.convolution(%v1224, %b13eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1226 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1227 = stablehlo.add %v1225, %v1226 : tensor<32x1152x7x7xf32>
    %v1228 = stablehlo.reshape %v1227 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1229 = stablehlo.reshape %v1228 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1230 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1231 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1232 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1233 = stablehlo.reduce(%v1229 init: %v1230) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1234 = stablehlo.broadcast_in_dim %v1233, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1235 = stablehlo.divide %v1234, %v1231 : tensor<32x1152x7x7xf32>
    %v1236 = stablehlo.subtract %v1229, %v1235 : tensor<32x1152x7x7xf32>
    %v1237 = stablehlo.multiply %v1236, %v1236 : tensor<32x1152x7x7xf32>
    %v1238 = stablehlo.reduce(%v1237 init: %v1230) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1239 = stablehlo.broadcast_in_dim %v1238, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1240 = stablehlo.divide %v1239, %v1231 : tensor<32x1152x7x7xf32>
    %v1241 = stablehlo.add %v1240, %v1232 : tensor<32x1152x7x7xf32>
    %v1242 = stablehlo.rsqrt %v1241 : tensor<32x1152x7x7xf32>
    %v1243 = stablehlo.multiply %v1236, %v1242 : tensor<32x1152x7x7xf32>
    %v1244 = stablehlo.broadcast_in_dim %b13eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1245 = stablehlo.broadcast_in_dim %b13ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1246 = stablehlo.multiply %v1243, %v1244 : tensor<32x1152x7x7xf32>
    %v1247 = stablehlo.add %v1246, %v1245 : tensor<32x1152x7x7xf32>
    %v1248 = stablehlo.reshape %v1247 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1249 = stablehlo.reshape %v1248 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1250 = stablehlo.logistic %v1249 : tensor<32x1152x7x7xf32>
    %v1251 = stablehlo.multiply %v1249, %v1250 : tensor<32x1152x7x7xf32>
    %v1252 = stablehlo.reshape %v1251 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1253 = stablehlo.reshape %v1252 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1254 = stablehlo.convolution(%v1253, %b13dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v1255 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1256 = stablehlo.add %v1254, %v1255 : tensor<32x1152x7x7xf32>
    %v1257 = stablehlo.reshape %v1256 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1258 = stablehlo.reshape %v1257 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1259 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1260 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1261 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1262 = stablehlo.reduce(%v1258 init: %v1259) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1263 = stablehlo.broadcast_in_dim %v1262, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1264 = stablehlo.divide %v1263, %v1260 : tensor<32x1152x7x7xf32>
    %v1265 = stablehlo.subtract %v1258, %v1264 : tensor<32x1152x7x7xf32>
    %v1266 = stablehlo.multiply %v1265, %v1265 : tensor<32x1152x7x7xf32>
    %v1267 = stablehlo.reduce(%v1266 init: %v1259) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1268 = stablehlo.broadcast_in_dim %v1267, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1269 = stablehlo.divide %v1268, %v1260 : tensor<32x1152x7x7xf32>
    %v1270 = stablehlo.add %v1269, %v1261 : tensor<32x1152x7x7xf32>
    %v1271 = stablehlo.rsqrt %v1270 : tensor<32x1152x7x7xf32>
    %v1272 = stablehlo.multiply %v1265, %v1271 : tensor<32x1152x7x7xf32>
    %v1273 = stablehlo.broadcast_in_dim %b13dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1274 = stablehlo.broadcast_in_dim %b13dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1275 = stablehlo.multiply %v1272, %v1273 : tensor<32x1152x7x7xf32>
    %v1276 = stablehlo.add %v1275, %v1274 : tensor<32x1152x7x7xf32>
    %v1277 = stablehlo.reshape %v1276 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1278 = stablehlo.reshape %v1277 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1279 = stablehlo.logistic %v1278 : tensor<32x1152x7x7xf32>
    %v1280 = stablehlo.multiply %v1278, %v1279 : tensor<32x1152x7x7xf32>
    %v1281 = stablehlo.reshape %v1280 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1282 = stablehlo.reshape %v1281 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1283 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1284 = stablehlo.reduce(%v1282 init: %v1283) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1285 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1286 = stablehlo.divide %v1284, %v1285 : tensor<32x1152xf32>
    %v1287 = stablehlo.dot_general %v1286, %b13zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1288 = stablehlo.broadcast_in_dim %b13zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1289 = stablehlo.add %v1287, %v1288 : tensor<32x48xf32>
    %v1290 = stablehlo.logistic %v1289 : tensor<32x48xf32>
    %v1291 = stablehlo.multiply %v1289, %v1290 : tensor<32x48xf32>
    %v1292 = stablehlo.dot_general %v1291, %b13zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1293 = stablehlo.broadcast_in_dim %b13zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1294 = stablehlo.add %v1292, %v1293 : tensor<32x1152xf32>
    %v1295 = stablehlo.logistic %v1294 : tensor<32x1152xf32>
    %v1296 = stablehlo.broadcast_in_dim %v1295, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1297 = stablehlo.multiply %v1282, %v1296 : tensor<32x1152x7x7xf32>
    %v1298 = stablehlo.reshape %v1297 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1299 = stablehlo.reshape %v1298 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1300 = stablehlo.convolution(%v1299, %b13pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1301 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1302 = stablehlo.add %v1300, %v1301 : tensor<32x192x7x7xf32>
    %v1303 = stablehlo.reshape %v1302 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1304 = stablehlo.reshape %v1303 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1305 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1306 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v1307 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1308 = stablehlo.reduce(%v1304 init: %v1305) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1309 = stablehlo.broadcast_in_dim %v1308, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1310 = stablehlo.divide %v1309, %v1306 : tensor<32x192x7x7xf32>
    %v1311 = stablehlo.subtract %v1304, %v1310 : tensor<32x192x7x7xf32>
    %v1312 = stablehlo.multiply %v1311, %v1311 : tensor<32x192x7x7xf32>
    %v1313 = stablehlo.reduce(%v1312 init: %v1305) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1314 = stablehlo.broadcast_in_dim %v1313, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1315 = stablehlo.divide %v1314, %v1306 : tensor<32x192x7x7xf32>
    %v1316 = stablehlo.add %v1315, %v1307 : tensor<32x192x7x7xf32>
    %v1317 = stablehlo.rsqrt %v1316 : tensor<32x192x7x7xf32>
    %v1318 = stablehlo.multiply %v1311, %v1317 : tensor<32x192x7x7xf32>
    %v1319 = stablehlo.broadcast_in_dim %b13pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1320 = stablehlo.broadcast_in_dim %b13pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1321 = stablehlo.multiply %v1318, %v1319 : tensor<32x192x7x7xf32>
    %v1322 = stablehlo.add %v1321, %v1320 : tensor<32x192x7x7xf32>
    %v1323 = stablehlo.reshape %v1322 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1324 = stablehlo.reshape %v1323 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1325 = stablehlo.reshape %v1223 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1326 = stablehlo.add %v1324, %v1325 : tensor<32x192x7x7xf32>
    %v1327 = stablehlo.reshape %v1326 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1328 = stablehlo.reshape %v1327 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1329 = stablehlo.convolution(%v1328, %b14eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1330 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1331 = stablehlo.add %v1329, %v1330 : tensor<32x1152x7x7xf32>
    %v1332 = stablehlo.reshape %v1331 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1333 = stablehlo.reshape %v1332 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1334 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1335 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1336 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1337 = stablehlo.reduce(%v1333 init: %v1334) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1338 = stablehlo.broadcast_in_dim %v1337, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1339 = stablehlo.divide %v1338, %v1335 : tensor<32x1152x7x7xf32>
    %v1340 = stablehlo.subtract %v1333, %v1339 : tensor<32x1152x7x7xf32>
    %v1341 = stablehlo.multiply %v1340, %v1340 : tensor<32x1152x7x7xf32>
    %v1342 = stablehlo.reduce(%v1341 init: %v1334) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1343 = stablehlo.broadcast_in_dim %v1342, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1344 = stablehlo.divide %v1343, %v1335 : tensor<32x1152x7x7xf32>
    %v1345 = stablehlo.add %v1344, %v1336 : tensor<32x1152x7x7xf32>
    %v1346 = stablehlo.rsqrt %v1345 : tensor<32x1152x7x7xf32>
    %v1347 = stablehlo.multiply %v1340, %v1346 : tensor<32x1152x7x7xf32>
    %v1348 = stablehlo.broadcast_in_dim %b14eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1349 = stablehlo.broadcast_in_dim %b14ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1350 = stablehlo.multiply %v1347, %v1348 : tensor<32x1152x7x7xf32>
    %v1351 = stablehlo.add %v1350, %v1349 : tensor<32x1152x7x7xf32>
    %v1352 = stablehlo.reshape %v1351 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1353 = stablehlo.reshape %v1352 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1354 = stablehlo.logistic %v1353 : tensor<32x1152x7x7xf32>
    %v1355 = stablehlo.multiply %v1353, %v1354 : tensor<32x1152x7x7xf32>
    %v1356 = stablehlo.reshape %v1355 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1357 = stablehlo.reshape %v1356 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1358 = stablehlo.convolution(%v1357, %b14dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v1359 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1360 = stablehlo.add %v1358, %v1359 : tensor<32x1152x7x7xf32>
    %v1361 = stablehlo.reshape %v1360 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1362 = stablehlo.reshape %v1361 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1363 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1364 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1365 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1366 = stablehlo.reduce(%v1362 init: %v1363) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1367 = stablehlo.broadcast_in_dim %v1366, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1368 = stablehlo.divide %v1367, %v1364 : tensor<32x1152x7x7xf32>
    %v1369 = stablehlo.subtract %v1362, %v1368 : tensor<32x1152x7x7xf32>
    %v1370 = stablehlo.multiply %v1369, %v1369 : tensor<32x1152x7x7xf32>
    %v1371 = stablehlo.reduce(%v1370 init: %v1363) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1372 = stablehlo.broadcast_in_dim %v1371, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1373 = stablehlo.divide %v1372, %v1364 : tensor<32x1152x7x7xf32>
    %v1374 = stablehlo.add %v1373, %v1365 : tensor<32x1152x7x7xf32>
    %v1375 = stablehlo.rsqrt %v1374 : tensor<32x1152x7x7xf32>
    %v1376 = stablehlo.multiply %v1369, %v1375 : tensor<32x1152x7x7xf32>
    %v1377 = stablehlo.broadcast_in_dim %b14dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1378 = stablehlo.broadcast_in_dim %b14dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1379 = stablehlo.multiply %v1376, %v1377 : tensor<32x1152x7x7xf32>
    %v1380 = stablehlo.add %v1379, %v1378 : tensor<32x1152x7x7xf32>
    %v1381 = stablehlo.reshape %v1380 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1382 = stablehlo.reshape %v1381 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1383 = stablehlo.logistic %v1382 : tensor<32x1152x7x7xf32>
    %v1384 = stablehlo.multiply %v1382, %v1383 : tensor<32x1152x7x7xf32>
    %v1385 = stablehlo.reshape %v1384 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1386 = stablehlo.reshape %v1385 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1387 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1388 = stablehlo.reduce(%v1386 init: %v1387) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1389 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1390 = stablehlo.divide %v1388, %v1389 : tensor<32x1152xf32>
    %v1391 = stablehlo.dot_general %v1390, %b14zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1392 = stablehlo.broadcast_in_dim %b14zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1393 = stablehlo.add %v1391, %v1392 : tensor<32x48xf32>
    %v1394 = stablehlo.logistic %v1393 : tensor<32x48xf32>
    %v1395 = stablehlo.multiply %v1393, %v1394 : tensor<32x48xf32>
    %v1396 = stablehlo.dot_general %v1395, %b14zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1397 = stablehlo.broadcast_in_dim %b14zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1398 = stablehlo.add %v1396, %v1397 : tensor<32x1152xf32>
    %v1399 = stablehlo.logistic %v1398 : tensor<32x1152xf32>
    %v1400 = stablehlo.broadcast_in_dim %v1399, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1401 = stablehlo.multiply %v1386, %v1400 : tensor<32x1152x7x7xf32>
    %v1402 = stablehlo.reshape %v1401 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1403 = stablehlo.reshape %v1402 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1404 = stablehlo.convolution(%v1403, %b14pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1405 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1406 = stablehlo.add %v1404, %v1405 : tensor<32x192x7x7xf32>
    %v1407 = stablehlo.reshape %v1406 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1408 = stablehlo.reshape %v1407 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1409 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1410 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v1411 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1412 = stablehlo.reduce(%v1408 init: %v1409) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1413 = stablehlo.broadcast_in_dim %v1412, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1414 = stablehlo.divide %v1413, %v1410 : tensor<32x192x7x7xf32>
    %v1415 = stablehlo.subtract %v1408, %v1414 : tensor<32x192x7x7xf32>
    %v1416 = stablehlo.multiply %v1415, %v1415 : tensor<32x192x7x7xf32>
    %v1417 = stablehlo.reduce(%v1416 init: %v1409) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1418 = stablehlo.broadcast_in_dim %v1417, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1419 = stablehlo.divide %v1418, %v1410 : tensor<32x192x7x7xf32>
    %v1420 = stablehlo.add %v1419, %v1411 : tensor<32x192x7x7xf32>
    %v1421 = stablehlo.rsqrt %v1420 : tensor<32x192x7x7xf32>
    %v1422 = stablehlo.multiply %v1415, %v1421 : tensor<32x192x7x7xf32>
    %v1423 = stablehlo.broadcast_in_dim %b14pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1424 = stablehlo.broadcast_in_dim %b14pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1425 = stablehlo.multiply %v1422, %v1423 : tensor<32x192x7x7xf32>
    %v1426 = stablehlo.add %v1425, %v1424 : tensor<32x192x7x7xf32>
    %v1427 = stablehlo.reshape %v1426 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1428 = stablehlo.reshape %v1427 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1429 = stablehlo.reshape %v1327 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1430 = stablehlo.add %v1428, %v1429 : tensor<32x192x7x7xf32>
    %v1431 = stablehlo.reshape %v1430 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1432 = stablehlo.reshape %v1431 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1433 = stablehlo.convolution(%v1432, %b15eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1434 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1435 = stablehlo.add %v1433, %v1434 : tensor<32x1152x7x7xf32>
    %v1436 = stablehlo.reshape %v1435 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1437 = stablehlo.reshape %v1436 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1438 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1439 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1440 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1441 = stablehlo.reduce(%v1437 init: %v1438) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1442 = stablehlo.broadcast_in_dim %v1441, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1443 = stablehlo.divide %v1442, %v1439 : tensor<32x1152x7x7xf32>
    %v1444 = stablehlo.subtract %v1437, %v1443 : tensor<32x1152x7x7xf32>
    %v1445 = stablehlo.multiply %v1444, %v1444 : tensor<32x1152x7x7xf32>
    %v1446 = stablehlo.reduce(%v1445 init: %v1438) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1447 = stablehlo.broadcast_in_dim %v1446, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1448 = stablehlo.divide %v1447, %v1439 : tensor<32x1152x7x7xf32>
    %v1449 = stablehlo.add %v1448, %v1440 : tensor<32x1152x7x7xf32>
    %v1450 = stablehlo.rsqrt %v1449 : tensor<32x1152x7x7xf32>
    %v1451 = stablehlo.multiply %v1444, %v1450 : tensor<32x1152x7x7xf32>
    %v1452 = stablehlo.broadcast_in_dim %b15eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1453 = stablehlo.broadcast_in_dim %b15ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1454 = stablehlo.multiply %v1451, %v1452 : tensor<32x1152x7x7xf32>
    %v1455 = stablehlo.add %v1454, %v1453 : tensor<32x1152x7x7xf32>
    %v1456 = stablehlo.reshape %v1455 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1457 = stablehlo.reshape %v1456 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1458 = stablehlo.logistic %v1457 : tensor<32x1152x7x7xf32>
    %v1459 = stablehlo.multiply %v1457, %v1458 : tensor<32x1152x7x7xf32>
    %v1460 = stablehlo.reshape %v1459 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1461 = stablehlo.reshape %v1460 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1462 = stablehlo.convolution(%v1461, %b15dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v1463 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1464 = stablehlo.add %v1462, %v1463 : tensor<32x1152x7x7xf32>
    %v1465 = stablehlo.reshape %v1464 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1466 = stablehlo.reshape %v1465 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1467 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1468 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1469 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1470 = stablehlo.reduce(%v1466 init: %v1467) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1471 = stablehlo.broadcast_in_dim %v1470, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1472 = stablehlo.divide %v1471, %v1468 : tensor<32x1152x7x7xf32>
    %v1473 = stablehlo.subtract %v1466, %v1472 : tensor<32x1152x7x7xf32>
    %v1474 = stablehlo.multiply %v1473, %v1473 : tensor<32x1152x7x7xf32>
    %v1475 = stablehlo.reduce(%v1474 init: %v1467) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1476 = stablehlo.broadcast_in_dim %v1475, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1477 = stablehlo.divide %v1476, %v1468 : tensor<32x1152x7x7xf32>
    %v1478 = stablehlo.add %v1477, %v1469 : tensor<32x1152x7x7xf32>
    %v1479 = stablehlo.rsqrt %v1478 : tensor<32x1152x7x7xf32>
    %v1480 = stablehlo.multiply %v1473, %v1479 : tensor<32x1152x7x7xf32>
    %v1481 = stablehlo.broadcast_in_dim %b15dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1482 = stablehlo.broadcast_in_dim %b15dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1483 = stablehlo.multiply %v1480, %v1481 : tensor<32x1152x7x7xf32>
    %v1484 = stablehlo.add %v1483, %v1482 : tensor<32x1152x7x7xf32>
    %v1485 = stablehlo.reshape %v1484 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1486 = stablehlo.reshape %v1485 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1487 = stablehlo.logistic %v1486 : tensor<32x1152x7x7xf32>
    %v1488 = stablehlo.multiply %v1486, %v1487 : tensor<32x1152x7x7xf32>
    %v1489 = stablehlo.reshape %v1488 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1490 = stablehlo.reshape %v1489 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1491 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1492 = stablehlo.reduce(%v1490 init: %v1491) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1493 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1494 = stablehlo.divide %v1492, %v1493 : tensor<32x1152xf32>
    %v1495 = stablehlo.dot_general %v1494, %b15zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1496 = stablehlo.broadcast_in_dim %b15zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1497 = stablehlo.add %v1495, %v1496 : tensor<32x48xf32>
    %v1498 = stablehlo.logistic %v1497 : tensor<32x48xf32>
    %v1499 = stablehlo.multiply %v1497, %v1498 : tensor<32x48xf32>
    %v1500 = stablehlo.dot_general %v1499, %b15zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1501 = stablehlo.broadcast_in_dim %b15zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1502 = stablehlo.add %v1500, %v1501 : tensor<32x1152xf32>
    %v1503 = stablehlo.logistic %v1502 : tensor<32x1152xf32>
    %v1504 = stablehlo.broadcast_in_dim %v1503, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1505 = stablehlo.multiply %v1490, %v1504 : tensor<32x1152x7x7xf32>
    %v1506 = stablehlo.reshape %v1505 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1507 = stablehlo.reshape %v1506 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1508 = stablehlo.convolution(%v1507, %b15pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1509 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1510 = stablehlo.add %v1508, %v1509 : tensor<32x192x7x7xf32>
    %v1511 = stablehlo.reshape %v1510 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1512 = stablehlo.reshape %v1511 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1513 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1514 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v1515 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1516 = stablehlo.reduce(%v1512 init: %v1513) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1517 = stablehlo.broadcast_in_dim %v1516, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1518 = stablehlo.divide %v1517, %v1514 : tensor<32x192x7x7xf32>
    %v1519 = stablehlo.subtract %v1512, %v1518 : tensor<32x192x7x7xf32>
    %v1520 = stablehlo.multiply %v1519, %v1519 : tensor<32x192x7x7xf32>
    %v1521 = stablehlo.reduce(%v1520 init: %v1513) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1522 = stablehlo.broadcast_in_dim %v1521, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1523 = stablehlo.divide %v1522, %v1514 : tensor<32x192x7x7xf32>
    %v1524 = stablehlo.add %v1523, %v1515 : tensor<32x192x7x7xf32>
    %v1525 = stablehlo.rsqrt %v1524 : tensor<32x192x7x7xf32>
    %v1526 = stablehlo.multiply %v1519, %v1525 : tensor<32x192x7x7xf32>
    %v1527 = stablehlo.broadcast_in_dim %b15pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1528 = stablehlo.broadcast_in_dim %b15pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1529 = stablehlo.multiply %v1526, %v1527 : tensor<32x192x7x7xf32>
    %v1530 = stablehlo.add %v1529, %v1528 : tensor<32x192x7x7xf32>
    %v1531 = stablehlo.reshape %v1530 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1532 = stablehlo.reshape %v1531 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1533 = stablehlo.reshape %v1431 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1534 = stablehlo.add %v1532, %v1533 : tensor<32x192x7x7xf32>
    %v1535 = stablehlo.reshape %v1534 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1536 = stablehlo.reshape %v1535 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1537 = stablehlo.convolution(%v1536, %b16eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1538 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1539 = stablehlo.add %v1537, %v1538 : tensor<32x1152x7x7xf32>
    %v1540 = stablehlo.reshape %v1539 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1541 = stablehlo.reshape %v1540 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1542 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1543 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1544 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1545 = stablehlo.reduce(%v1541 init: %v1542) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1546 = stablehlo.broadcast_in_dim %v1545, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1547 = stablehlo.divide %v1546, %v1543 : tensor<32x1152x7x7xf32>
    %v1548 = stablehlo.subtract %v1541, %v1547 : tensor<32x1152x7x7xf32>
    %v1549 = stablehlo.multiply %v1548, %v1548 : tensor<32x1152x7x7xf32>
    %v1550 = stablehlo.reduce(%v1549 init: %v1542) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1551 = stablehlo.broadcast_in_dim %v1550, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1552 = stablehlo.divide %v1551, %v1543 : tensor<32x1152x7x7xf32>
    %v1553 = stablehlo.add %v1552, %v1544 : tensor<32x1152x7x7xf32>
    %v1554 = stablehlo.rsqrt %v1553 : tensor<32x1152x7x7xf32>
    %v1555 = stablehlo.multiply %v1548, %v1554 : tensor<32x1152x7x7xf32>
    %v1556 = stablehlo.broadcast_in_dim %b16eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1557 = stablehlo.broadcast_in_dim %b16ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1558 = stablehlo.multiply %v1555, %v1556 : tensor<32x1152x7x7xf32>
    %v1559 = stablehlo.add %v1558, %v1557 : tensor<32x1152x7x7xf32>
    %v1560 = stablehlo.reshape %v1559 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1561 = stablehlo.reshape %v1560 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1562 = stablehlo.logistic %v1561 : tensor<32x1152x7x7xf32>
    %v1563 = stablehlo.multiply %v1561, %v1562 : tensor<32x1152x7x7xf32>
    %v1564 = stablehlo.reshape %v1563 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1565 = stablehlo.reshape %v1564 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1566 = stablehlo.convolution(%v1565, %b16dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x3x3xf32>) -> tensor<32x1152x7x7xf32>
    %v1567 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1568 = stablehlo.add %v1566, %v1567 : tensor<32x1152x7x7xf32>
    %v1569 = stablehlo.reshape %v1568 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1570 = stablehlo.reshape %v1569 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1571 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1572 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1573 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1574 = stablehlo.reduce(%v1570 init: %v1571) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1575 = stablehlo.broadcast_in_dim %v1574, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1576 = stablehlo.divide %v1575, %v1572 : tensor<32x1152x7x7xf32>
    %v1577 = stablehlo.subtract %v1570, %v1576 : tensor<32x1152x7x7xf32>
    %v1578 = stablehlo.multiply %v1577, %v1577 : tensor<32x1152x7x7xf32>
    %v1579 = stablehlo.reduce(%v1578 init: %v1571) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1580 = stablehlo.broadcast_in_dim %v1579, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1581 = stablehlo.divide %v1580, %v1572 : tensor<32x1152x7x7xf32>
    %v1582 = stablehlo.add %v1581, %v1573 : tensor<32x1152x7x7xf32>
    %v1583 = stablehlo.rsqrt %v1582 : tensor<32x1152x7x7xf32>
    %v1584 = stablehlo.multiply %v1577, %v1583 : tensor<32x1152x7x7xf32>
    %v1585 = stablehlo.broadcast_in_dim %b16dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1586 = stablehlo.broadcast_in_dim %b16dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1587 = stablehlo.multiply %v1584, %v1585 : tensor<32x1152x7x7xf32>
    %v1588 = stablehlo.add %v1587, %v1586 : tensor<32x1152x7x7xf32>
    %v1589 = stablehlo.reshape %v1588 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1590 = stablehlo.reshape %v1589 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1591 = stablehlo.logistic %v1590 : tensor<32x1152x7x7xf32>
    %v1592 = stablehlo.multiply %v1590, %v1591 : tensor<32x1152x7x7xf32>
    %v1593 = stablehlo.reshape %v1592 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1594 = stablehlo.reshape %v1593 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1595 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1596 = stablehlo.reduce(%v1594 init: %v1595) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1597 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1598 = stablehlo.divide %v1596, %v1597 : tensor<32x1152xf32>
    %v1599 = stablehlo.dot_general %v1598, %b16zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1600 = stablehlo.broadcast_in_dim %b16zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1601 = stablehlo.add %v1599, %v1600 : tensor<32x48xf32>
    %v1602 = stablehlo.logistic %v1601 : tensor<32x48xf32>
    %v1603 = stablehlo.multiply %v1601, %v1602 : tensor<32x48xf32>
    %v1604 = stablehlo.dot_general %v1603, %b16zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1605 = stablehlo.broadcast_in_dim %b16zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1606 = stablehlo.add %v1604, %v1605 : tensor<32x1152xf32>
    %v1607 = stablehlo.logistic %v1606 : tensor<32x1152xf32>
    %v1608 = stablehlo.broadcast_in_dim %v1607, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1609 = stablehlo.multiply %v1594, %v1608 : tensor<32x1152x7x7xf32>
    %v1610 = stablehlo.reshape %v1609 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1611 = stablehlo.reshape %v1610 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1612 = stablehlo.convolution(%v1611, %b16pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<320x1152x1x1xf32>) -> tensor<32x320x7x7xf32>
    %v1613 = stablehlo.broadcast_in_dim %zb320, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1614 = stablehlo.add %v1612, %v1613 : tensor<32x320x7x7xf32>
    %v1615 = stablehlo.reshape %v1614 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1616 = stablehlo.reshape %v1615 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1617 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1618 = stablehlo.constant dense<1568.0> : tensor<32x320x7x7xf32>
    %v1619 = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %v1620 = stablehlo.reduce(%v1616 init: %v1617) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1621 = stablehlo.broadcast_in_dim %v1620, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1622 = stablehlo.divide %v1621, %v1618 : tensor<32x320x7x7xf32>
    %v1623 = stablehlo.subtract %v1616, %v1622 : tensor<32x320x7x7xf32>
    %v1624 = stablehlo.multiply %v1623, %v1623 : tensor<32x320x7x7xf32>
    %v1625 = stablehlo.reduce(%v1624 init: %v1617) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1626 = stablehlo.broadcast_in_dim %v1625, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1627 = stablehlo.divide %v1626, %v1618 : tensor<32x320x7x7xf32>
    %v1628 = stablehlo.add %v1627, %v1619 : tensor<32x320x7x7xf32>
    %v1629 = stablehlo.rsqrt %v1628 : tensor<32x320x7x7xf32>
    %v1630 = stablehlo.multiply %v1623, %v1629 : tensor<32x320x7x7xf32>
    %v1631 = stablehlo.broadcast_in_dim %b16pg, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1632 = stablehlo.broadcast_in_dim %b16pbt, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1633 = stablehlo.multiply %v1630, %v1631 : tensor<32x320x7x7xf32>
    %v1634 = stablehlo.add %v1633, %v1632 : tensor<32x320x7x7xf32>
    %v1635 = stablehlo.reshape %v1634 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1636 = stablehlo.reshape %v1635 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1637 = stablehlo.convolution(%v1636, %hW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x320x7x7xf32>, tensor<1280x320x1x1xf32>) -> tensor<32x1280x7x7xf32>
    %v1638 = stablehlo.broadcast_in_dim %zb1280, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1639 = stablehlo.add %v1637, %v1638 : tensor<32x1280x7x7xf32>
    %v1640 = stablehlo.reshape %v1639 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1641 = stablehlo.reshape %v1640 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1642 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1643 = stablehlo.constant dense<1568.0> : tensor<32x1280x7x7xf32>
    %v1644 = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %v1645 = stablehlo.reduce(%v1641 init: %v1642) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1646 = stablehlo.broadcast_in_dim %v1645, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1647 = stablehlo.divide %v1646, %v1643 : tensor<32x1280x7x7xf32>
    %v1648 = stablehlo.subtract %v1641, %v1647 : tensor<32x1280x7x7xf32>
    %v1649 = stablehlo.multiply %v1648, %v1648 : tensor<32x1280x7x7xf32>
    %v1650 = stablehlo.reduce(%v1649 init: %v1642) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1651 = stablehlo.broadcast_in_dim %v1650, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1652 = stablehlo.divide %v1651, %v1643 : tensor<32x1280x7x7xf32>
    %v1653 = stablehlo.add %v1652, %v1644 : tensor<32x1280x7x7xf32>
    %v1654 = stablehlo.rsqrt %v1653 : tensor<32x1280x7x7xf32>
    %v1655 = stablehlo.multiply %v1648, %v1654 : tensor<32x1280x7x7xf32>
    %v1656 = stablehlo.broadcast_in_dim %hg, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1657 = stablehlo.broadcast_in_dim %hbt, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1658 = stablehlo.multiply %v1655, %v1656 : tensor<32x1280x7x7xf32>
    %v1659 = stablehlo.add %v1658, %v1657 : tensor<32x1280x7x7xf32>
    %v1660 = stablehlo.reshape %v1659 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1661 = stablehlo.reshape %v1660 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1662 = stablehlo.logistic %v1661 : tensor<32x1280x7x7xf32>
    %v1663 = stablehlo.multiply %v1661, %v1662 : tensor<32x1280x7x7xf32>
    %v1664 = stablehlo.reshape %v1663 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1665 = stablehlo.reshape %v1664 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1666 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1667 = stablehlo.reduce(%v1665 init: %v1666) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v1668 = stablehlo.constant dense<49.0> : tensor<32x1280xf32>
    %v1669 = stablehlo.divide %v1667, %v1668 : tensor<32x1280xf32>
    %v1670 = stablehlo.dot_general %v1669, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1280xf32>, tensor<1280x10xf32>) -> tensor<32x10xf32>
    %v1671 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v1672 = stablehlo.add %v1670, %v1671 : tensor<32x10xf32>
    %v1673 = stablehlo.reshape %v1672 : (tensor<32x10xf32>) -> tensor<32x1x10xf32>
    %v1674 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1675 = stablehlo.exponential %v1673 : tensor<32x1x10xf32>
    %v1676 = stablehlo.reduce(%v1675 init: %v1674) applies stablehlo.add across dimensions = [2] : (tensor<32x1x10xf32>, tensor<f32>) -> tensor<32x1xf32>
    %v1677 = stablehlo.broadcast_in_dim %v1676, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x10xf32>
    %v1678 = stablehlo.divide %v1675, %v1677 : tensor<32x1x10xf32>
    %v1679 = stablehlo.reshape %v1678 : (tensor<32x1x10xf32>) -> tensor<32x10xf32>
    %v1680 = stablehlo.subtract %v1679, %onehot : tensor<32x10xf32>
    %v1681 = stablehlo.reshape %v1680 : (tensor<32x10xf32>) -> tensor<32x1x10xf32>
    %v1682 = stablehlo.dot_general %v1681, %Wd, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x10xf32>, tensor<1280x10xf32>) -> tensor<32x1x1280xf32>
    %v1683 = stablehlo.reshape %v1682 : (tensor<32x1x1280xf32>) -> tensor<32x1280xf32>
    %v1684 = stablehlo.dot_general %v1669, %v1680, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1280xf32>, tensor<32x10xf32>) -> tensor<1280x10xf32>
    %v1685 = stablehlo.constant dense<0.05> : tensor<1280x10xf32>
    %v1686 = stablehlo.multiply %v1684, %v1685 : tensor<1280x10xf32>
    %v1687 = stablehlo.subtract %Wd, %v1686 : tensor<1280x10xf32>
    %v1688 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1689 = stablehlo.reduce(%v1680 init: %v1688) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1690 = stablehlo.constant dense<0.05> : tensor<10xf32>
    %v1691 = stablehlo.multiply %v1689, %v1690 : tensor<10xf32>
    %v1692 = stablehlo.subtract %bd, %v1691 : tensor<10xf32>
    %v1693 = stablehlo.broadcast_in_dim %v1683, dims = [0, 1] : (tensor<32x1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1694 = stablehlo.constant dense<49.0> : tensor<32x1280x7x7xf32>
    %v1695 = stablehlo.divide %v1693, %v1694 : tensor<32x1280x7x7xf32>
    %v1696 = stablehlo.reshape %v1695 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1697 = stablehlo.reshape %v1696 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1698 = stablehlo.reshape %v1660 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1699 = stablehlo.logistic %v1698 : tensor<32x1280x7x7xf32>
    %v1700 = stablehlo.constant dense<1.0> : tensor<32x1280x7x7xf32>
    %v1701 = stablehlo.subtract %v1700, %v1699 : tensor<32x1280x7x7xf32>
    %v1702 = stablehlo.multiply %v1698, %v1701 : tensor<32x1280x7x7xf32>
    %v1703 = stablehlo.add %v1700, %v1702 : tensor<32x1280x7x7xf32>
    %v1704 = stablehlo.multiply %v1699, %v1703 : tensor<32x1280x7x7xf32>
    %v1705 = stablehlo.multiply %v1697, %v1704 : tensor<32x1280x7x7xf32>
    %v1706 = stablehlo.reshape %v1705 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1707 = stablehlo.reshape %v1640 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1708 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1709 = stablehlo.constant dense<1568.0> : tensor<32x1280x7x7xf32>
    %v1710 = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %v1711 = stablehlo.reduce(%v1707 init: %v1708) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1712 = stablehlo.broadcast_in_dim %v1711, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1713 = stablehlo.divide %v1712, %v1709 : tensor<32x1280x7x7xf32>
    %v1714 = stablehlo.subtract %v1707, %v1713 : tensor<32x1280x7x7xf32>
    %v1715 = stablehlo.multiply %v1714, %v1714 : tensor<32x1280x7x7xf32>
    %v1716 = stablehlo.reduce(%v1715 init: %v1708) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1717 = stablehlo.broadcast_in_dim %v1716, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1718 = stablehlo.divide %v1717, %v1709 : tensor<32x1280x7x7xf32>
    %v1719 = stablehlo.add %v1718, %v1710 : tensor<32x1280x7x7xf32>
    %v1720 = stablehlo.rsqrt %v1719 : tensor<32x1280x7x7xf32>
    %v1721 = stablehlo.multiply %v1714, %v1720 : tensor<32x1280x7x7xf32>
    %v1722 = stablehlo.broadcast_in_dim %hg, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1723 = stablehlo.reshape %v1706 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1724 = stablehlo.multiply %v1722, %v1723 : tensor<32x1280x7x7xf32>
    %v1725 = stablehlo.reduce(%v1724 init: %v1708) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1726 = stablehlo.broadcast_in_dim %v1725, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1727 = stablehlo.multiply %v1721, %v1724 : tensor<32x1280x7x7xf32>
    %v1728 = stablehlo.reduce(%v1727 init: %v1708) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1729 = stablehlo.broadcast_in_dim %v1728, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1730 = stablehlo.multiply %v1724, %v1709 : tensor<32x1280x7x7xf32>
    %v1731 = stablehlo.subtract %v1730, %v1726 : tensor<32x1280x7x7xf32>
    %v1732 = stablehlo.multiply %v1721, %v1729 : tensor<32x1280x7x7xf32>
    %v1733 = stablehlo.subtract %v1731, %v1732 : tensor<32x1280x7x7xf32>
    %v1734 = stablehlo.divide %v1720, %v1709 : tensor<32x1280x7x7xf32>
    %v1735 = stablehlo.multiply %v1734, %v1733 : tensor<32x1280x7x7xf32>
    %v1736 = stablehlo.reshape %v1735 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1737 = stablehlo.reshape %v1736 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1738 = stablehlo.reverse %hW, dims = [2, 3] : tensor<1280x320x1x1xf32>
    %v1739 = stablehlo.transpose %v1738, dims = [1, 0, 2, 3] : (tensor<1280x320x1x1xf32>) -> tensor<320x1280x1x1xf32>
    %v1740 = stablehlo.convolution(%v1737, %v1739)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1280x7x7xf32>, tensor<320x1280x1x1xf32>) -> tensor<32x320x7x7xf32>
    %v1741 = stablehlo.reshape %v1740 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1742 = stablehlo.reshape %v1640 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1743 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1744 = stablehlo.constant dense<1568.0> : tensor<32x1280x7x7xf32>
    %v1745 = stablehlo.reduce(%v1742 init: %v1743) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1746 = stablehlo.broadcast_in_dim %v1745, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1747 = stablehlo.divide %v1746, %v1744 : tensor<32x1280x7x7xf32>
    %v1748 = stablehlo.subtract %v1742, %v1747 : tensor<32x1280x7x7xf32>
    %v1749 = stablehlo.multiply %v1748, %v1748 : tensor<32x1280x7x7xf32>
    %v1750 = stablehlo.reduce(%v1749 init: %v1743) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1751 = stablehlo.broadcast_in_dim %v1750, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1752 = stablehlo.divide %v1751, %v1744 : tensor<32x1280x7x7xf32>
    %v1753 = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %v1754 = stablehlo.add %v1752, %v1753 : tensor<32x1280x7x7xf32>
    %v1755 = stablehlo.rsqrt %v1754 : tensor<32x1280x7x7xf32>
    %v1756 = stablehlo.multiply %v1748, %v1755 : tensor<32x1280x7x7xf32>
    %v1757 = stablehlo.reshape %v1706 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1758 = stablehlo.multiply %v1757, %v1756 : tensor<32x1280x7x7xf32>
    %v1759 = stablehlo.reduce(%v1758 init: %v1743) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1760 = stablehlo.constant dense<0.05> : tensor<1280xf32>
    %v1761 = stablehlo.multiply %v1759, %v1760 : tensor<1280xf32>
    %v1762 = stablehlo.subtract %hg, %v1761 : tensor<1280xf32>
    %v1763 = stablehlo.reshape %v1706 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1764 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1765 = stablehlo.reduce(%v1763 init: %v1764) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1766 = stablehlo.constant dense<0.05> : tensor<1280xf32>
    %v1767 = stablehlo.multiply %v1765, %v1766 : tensor<1280xf32>
    %v1768 = stablehlo.subtract %hbt, %v1767 : tensor<1280xf32>
    %v1769 = stablehlo.reshape %v1635 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1770 = stablehlo.reshape %v1736 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1771 = stablehlo.transpose %v1769, dims = [1, 0, 2, 3] : (tensor<32x320x7x7xf32>) -> tensor<320x32x7x7xf32>
    %v1772 = stablehlo.transpose %v1770, dims = [1, 0, 2, 3] : (tensor<32x1280x7x7xf32>) -> tensor<1280x32x7x7xf32>
    %v1773 = stablehlo.convolution(%v1771, %v1772)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<320x32x7x7xf32>, tensor<1280x32x7x7xf32>) -> tensor<320x1280x1x1xf32>
    %v1774 = stablehlo.transpose %v1773, dims = [1, 0, 2, 3] : (tensor<320x1280x1x1xf32>) -> tensor<1280x320x1x1xf32>
    %v1775 = stablehlo.constant dense<0.05> : tensor<1280x320x1x1xf32>
    %v1776 = stablehlo.multiply %v1774, %v1775 : tensor<1280x320x1x1xf32>
    %v1777 = stablehlo.subtract %hW, %v1776 : tensor<1280x320x1x1xf32>
    %v1778 = stablehlo.reshape %v1615 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1779 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1780 = stablehlo.constant dense<1568.0> : tensor<32x320x7x7xf32>
    %v1781 = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %v1782 = stablehlo.reduce(%v1778 init: %v1779) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1783 = stablehlo.broadcast_in_dim %v1782, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1784 = stablehlo.divide %v1783, %v1780 : tensor<32x320x7x7xf32>
    %v1785 = stablehlo.subtract %v1778, %v1784 : tensor<32x320x7x7xf32>
    %v1786 = stablehlo.multiply %v1785, %v1785 : tensor<32x320x7x7xf32>
    %v1787 = stablehlo.reduce(%v1786 init: %v1779) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1788 = stablehlo.broadcast_in_dim %v1787, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1789 = stablehlo.divide %v1788, %v1780 : tensor<32x320x7x7xf32>
    %v1790 = stablehlo.add %v1789, %v1781 : tensor<32x320x7x7xf32>
    %v1791 = stablehlo.rsqrt %v1790 : tensor<32x320x7x7xf32>
    %v1792 = stablehlo.multiply %v1785, %v1791 : tensor<32x320x7x7xf32>
    %v1793 = stablehlo.broadcast_in_dim %b16pg, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1794 = stablehlo.reshape %v1741 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1795 = stablehlo.multiply %v1793, %v1794 : tensor<32x320x7x7xf32>
    %v1796 = stablehlo.reduce(%v1795 init: %v1779) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1797 = stablehlo.broadcast_in_dim %v1796, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1798 = stablehlo.multiply %v1792, %v1795 : tensor<32x320x7x7xf32>
    %v1799 = stablehlo.reduce(%v1798 init: %v1779) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1800 = stablehlo.broadcast_in_dim %v1799, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1801 = stablehlo.multiply %v1795, %v1780 : tensor<32x320x7x7xf32>
    %v1802 = stablehlo.subtract %v1801, %v1797 : tensor<32x320x7x7xf32>
    %v1803 = stablehlo.multiply %v1792, %v1800 : tensor<32x320x7x7xf32>
    %v1804 = stablehlo.subtract %v1802, %v1803 : tensor<32x320x7x7xf32>
    %v1805 = stablehlo.divide %v1791, %v1780 : tensor<32x320x7x7xf32>
    %v1806 = stablehlo.multiply %v1805, %v1804 : tensor<32x320x7x7xf32>
    %v1807 = stablehlo.reshape %v1806 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1808 = stablehlo.reshape %v1807 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1809 = stablehlo.reverse %b16pW, dims = [2, 3] : tensor<320x1152x1x1xf32>
    %v1810 = stablehlo.transpose %v1809, dims = [1, 0, 2, 3] : (tensor<320x1152x1x1xf32>) -> tensor<1152x320x1x1xf32>
    %v1811 = stablehlo.convolution(%v1808, %v1810)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x320x7x7xf32>, tensor<1152x320x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1812 = stablehlo.reshape %v1811 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1813 = stablehlo.reshape %v1615 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1814 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1815 = stablehlo.constant dense<1568.0> : tensor<32x320x7x7xf32>
    %v1816 = stablehlo.reduce(%v1813 init: %v1814) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1817 = stablehlo.broadcast_in_dim %v1816, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1818 = stablehlo.divide %v1817, %v1815 : tensor<32x320x7x7xf32>
    %v1819 = stablehlo.subtract %v1813, %v1818 : tensor<32x320x7x7xf32>
    %v1820 = stablehlo.multiply %v1819, %v1819 : tensor<32x320x7x7xf32>
    %v1821 = stablehlo.reduce(%v1820 init: %v1814) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1822 = stablehlo.broadcast_in_dim %v1821, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1823 = stablehlo.divide %v1822, %v1815 : tensor<32x320x7x7xf32>
    %v1824 = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %v1825 = stablehlo.add %v1823, %v1824 : tensor<32x320x7x7xf32>
    %v1826 = stablehlo.rsqrt %v1825 : tensor<32x320x7x7xf32>
    %v1827 = stablehlo.multiply %v1819, %v1826 : tensor<32x320x7x7xf32>
    %v1828 = stablehlo.reshape %v1741 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1829 = stablehlo.multiply %v1828, %v1827 : tensor<32x320x7x7xf32>
    %v1830 = stablehlo.reduce(%v1829 init: %v1814) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1831 = stablehlo.constant dense<0.05> : tensor<320xf32>
    %v1832 = stablehlo.multiply %v1830, %v1831 : tensor<320xf32>
    %v1833 = stablehlo.subtract %b16pg, %v1832 : tensor<320xf32>
    %v1834 = stablehlo.reshape %v1741 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1835 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1836 = stablehlo.reduce(%v1834 init: %v1835) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1837 = stablehlo.constant dense<0.05> : tensor<320xf32>
    %v1838 = stablehlo.multiply %v1836, %v1837 : tensor<320xf32>
    %v1839 = stablehlo.subtract %b16pbt, %v1838 : tensor<320xf32>
    %v1840 = stablehlo.reshape %v1610 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1841 = stablehlo.reshape %v1807 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1842 = stablehlo.transpose %v1840, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v1843 = stablehlo.transpose %v1841, dims = [1, 0, 2, 3] : (tensor<32x320x7x7xf32>) -> tensor<320x32x7x7xf32>
    %v1844 = stablehlo.convolution(%v1842, %v1843)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<320x32x7x7xf32>) -> tensor<1152x320x1x1xf32>
    %v1845 = stablehlo.transpose %v1844, dims = [1, 0, 2, 3] : (tensor<1152x320x1x1xf32>) -> tensor<320x1152x1x1xf32>
    %v1846 = stablehlo.constant dense<0.05> : tensor<320x1152x1x1xf32>
    %v1847 = stablehlo.multiply %v1845, %v1846 : tensor<320x1152x1x1xf32>
    %v1848 = stablehlo.subtract %b16pW, %v1847 : tensor<320x1152x1x1xf32>
    %v1849 = stablehlo.reshape %v1593 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1850 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1851 = stablehlo.reduce(%v1849 init: %v1850) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1852 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1853 = stablehlo.divide %v1851, %v1852 : tensor<32x1152xf32>
    %v1854 = stablehlo.dot_general %v1853, %b16zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1855 = stablehlo.broadcast_in_dim %b16zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1856 = stablehlo.add %v1854, %v1855 : tensor<32x48xf32>
    %v1857 = stablehlo.logistic %v1856 : tensor<32x48xf32>
    %v1858 = stablehlo.multiply %v1856, %v1857 : tensor<32x48xf32>
    %v1859 = stablehlo.dot_general %v1858, %b16zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1860 = stablehlo.broadcast_in_dim %b16zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1861 = stablehlo.add %v1859, %v1860 : tensor<32x1152xf32>
    %v1862 = stablehlo.logistic %v1861 : tensor<32x1152xf32>
    %v1863 = stablehlo.reshape %v1812 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1864 = stablehlo.broadcast_in_dim %v1862, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1865 = stablehlo.multiply %v1864, %v1863 : tensor<32x1152x7x7xf32>
    %v1866 = stablehlo.multiply %v1849, %v1863 : tensor<32x1152x7x7xf32>
    %v1867 = stablehlo.reduce(%v1866 init: %v1850) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1868 = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %v1869 = stablehlo.subtract %v1868, %v1862 : tensor<32x1152xf32>
    %v1870 = stablehlo.multiply %v1862, %v1869 : tensor<32x1152xf32>
    %v1871 = stablehlo.multiply %v1867, %v1870 : tensor<32x1152xf32>
    %v1872 = stablehlo.dot_general %v1871, %b16zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<48x1152xf32>) -> tensor<32x48xf32>
    %v1873 = stablehlo.logistic %v1856 : tensor<32x48xf32>
    %v1874 = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %v1875 = stablehlo.subtract %v1874, %v1873 : tensor<32x48xf32>
    %v1876 = stablehlo.multiply %v1856, %v1875 : tensor<32x48xf32>
    %v1877 = stablehlo.add %v1874, %v1876 : tensor<32x48xf32>
    %v1878 = stablehlo.multiply %v1873, %v1877 : tensor<32x48xf32>
    %v1879 = stablehlo.multiply %v1872, %v1878 : tensor<32x48xf32>
    %v1880 = stablehlo.dot_general %v1879, %b16zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<1152x48xf32>) -> tensor<32x1152xf32>
    %v1881 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1882 = stablehlo.divide %v1880, %v1881 : tensor<32x1152xf32>
    %v1883 = stablehlo.broadcast_in_dim %v1882, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1884 = stablehlo.add %v1865, %v1883 : tensor<32x1152x7x7xf32>
    %v1885 = stablehlo.reshape %v1884 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1886 = stablehlo.reshape %v1593 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1887 = stablehlo.reshape %v1812 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1888 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1889 = stablehlo.multiply %v1886, %v1887 : tensor<32x1152x7x7xf32>
    %v1890 = stablehlo.reduce(%v1889 init: %v1888) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1891 = stablehlo.logistic %v1606 : tensor<32x1152xf32>
    %v1892 = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %v1893 = stablehlo.subtract %v1892, %v1891 : tensor<32x1152xf32>
    %v1894 = stablehlo.multiply %v1891, %v1893 : tensor<32x1152xf32>
    %v1895 = stablehlo.multiply %v1890, %v1894 : tensor<32x1152xf32>
    %v1896 = stablehlo.dot_general %v1603, %v1895, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<32x1152xf32>) -> tensor<48x1152xf32>
    %v1897 = stablehlo.constant dense<0.05> : tensor<48x1152xf32>
    %v1898 = stablehlo.multiply %v1896, %v1897 : tensor<48x1152xf32>
    %v1899 = stablehlo.subtract %b16zW2, %v1898 : tensor<48x1152xf32>
    %v1900 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1901 = stablehlo.reduce(%v1895 init: %v1900) applies stablehlo.add across dimensions = [0] : (tensor<32x1152xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1902 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v1903 = stablehlo.multiply %v1901, %v1902 : tensor<1152xf32>
    %v1904 = stablehlo.subtract %b16zb2, %v1903 : tensor<1152xf32>
    %v1905 = stablehlo.reshape %v1895 : (tensor<32x1152xf32>) -> tensor<32x1x1152xf32>
    %v1906 = stablehlo.dot_general %v1905, %b16zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x1152xf32>, tensor<48x1152xf32>) -> tensor<32x1x48xf32>
    %v1907 = stablehlo.reshape %v1906 : (tensor<32x1x48xf32>) -> tensor<32x48xf32>
    %v1908 = stablehlo.logistic %v1601 : tensor<32x48xf32>
    %v1909 = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %v1910 = stablehlo.subtract %v1909, %v1908 : tensor<32x48xf32>
    %v1911 = stablehlo.multiply %v1601, %v1910 : tensor<32x48xf32>
    %v1912 = stablehlo.add %v1909, %v1911 : tensor<32x48xf32>
    %v1913 = stablehlo.multiply %v1908, %v1912 : tensor<32x48xf32>
    %v1914 = stablehlo.multiply %v1907, %v1913 : tensor<32x48xf32>
    %v1915 = stablehlo.dot_general %v1598, %v1914, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<32x48xf32>) -> tensor<1152x48xf32>
    %v1916 = stablehlo.constant dense<0.05> : tensor<1152x48xf32>
    %v1917 = stablehlo.multiply %v1915, %v1916 : tensor<1152x48xf32>
    %v1918 = stablehlo.subtract %b16zW1, %v1917 : tensor<1152x48xf32>
    %v1919 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1920 = stablehlo.reduce(%v1914 init: %v1919) applies stablehlo.add across dimensions = [0] : (tensor<32x48xf32>, tensor<f32>) -> tensor<48xf32>
    %v1921 = stablehlo.constant dense<0.05> : tensor<48xf32>
    %v1922 = stablehlo.multiply %v1920, %v1921 : tensor<48xf32>
    %v1923 = stablehlo.subtract %b16zb1, %v1922 : tensor<48xf32>
    %v1924 = stablehlo.reshape %v1885 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1925 = stablehlo.reshape %v1589 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1926 = stablehlo.logistic %v1925 : tensor<32x1152x7x7xf32>
    %v1927 = stablehlo.constant dense<1.0> : tensor<32x1152x7x7xf32>
    %v1928 = stablehlo.subtract %v1927, %v1926 : tensor<32x1152x7x7xf32>
    %v1929 = stablehlo.multiply %v1925, %v1928 : tensor<32x1152x7x7xf32>
    %v1930 = stablehlo.add %v1927, %v1929 : tensor<32x1152x7x7xf32>
    %v1931 = stablehlo.multiply %v1926, %v1930 : tensor<32x1152x7x7xf32>
    %v1932 = stablehlo.multiply %v1924, %v1931 : tensor<32x1152x7x7xf32>
    %v1933 = stablehlo.reshape %v1932 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1934 = stablehlo.reshape %v1569 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1935 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1936 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1937 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1938 = stablehlo.reduce(%v1934 init: %v1935) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1939 = stablehlo.broadcast_in_dim %v1938, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1940 = stablehlo.divide %v1939, %v1936 : tensor<32x1152x7x7xf32>
    %v1941 = stablehlo.subtract %v1934, %v1940 : tensor<32x1152x7x7xf32>
    %v1942 = stablehlo.multiply %v1941, %v1941 : tensor<32x1152x7x7xf32>
    %v1943 = stablehlo.reduce(%v1942 init: %v1935) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1944 = stablehlo.broadcast_in_dim %v1943, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1945 = stablehlo.divide %v1944, %v1936 : tensor<32x1152x7x7xf32>
    %v1946 = stablehlo.add %v1945, %v1937 : tensor<32x1152x7x7xf32>
    %v1947 = stablehlo.rsqrt %v1946 : tensor<32x1152x7x7xf32>
    %v1948 = stablehlo.multiply %v1941, %v1947 : tensor<32x1152x7x7xf32>
    %v1949 = stablehlo.broadcast_in_dim %b16dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1950 = stablehlo.reshape %v1933 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1951 = stablehlo.multiply %v1949, %v1950 : tensor<32x1152x7x7xf32>
    %v1952 = stablehlo.reduce(%v1951 init: %v1935) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1953 = stablehlo.broadcast_in_dim %v1952, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1954 = stablehlo.multiply %v1948, %v1951 : tensor<32x1152x7x7xf32>
    %v1955 = stablehlo.reduce(%v1954 init: %v1935) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1956 = stablehlo.broadcast_in_dim %v1955, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1957 = stablehlo.multiply %v1951, %v1936 : tensor<32x1152x7x7xf32>
    %v1958 = stablehlo.subtract %v1957, %v1953 : tensor<32x1152x7x7xf32>
    %v1959 = stablehlo.multiply %v1948, %v1956 : tensor<32x1152x7x7xf32>
    %v1960 = stablehlo.subtract %v1958, %v1959 : tensor<32x1152x7x7xf32>
    %v1961 = stablehlo.divide %v1947, %v1936 : tensor<32x1152x7x7xf32>
    %v1962 = stablehlo.multiply %v1961, %v1960 : tensor<32x1152x7x7xf32>
    %v1963 = stablehlo.reshape %v1962 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1964 = stablehlo.reshape %v1963 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1965 = stablehlo.reverse %b16dW, dims = [2, 3] : tensor<1152x1x3x3xf32>
    %v1966 = stablehlo.convolution(%v1964, %v1965)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x3x3xf32>) -> tensor<32x1152x7x7xf32>
    %v1967 = stablehlo.reshape %v1966 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1968 = stablehlo.reshape %v1569 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1969 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1970 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1971 = stablehlo.reduce(%v1968 init: %v1969) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1972 = stablehlo.broadcast_in_dim %v1971, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1973 = stablehlo.divide %v1972, %v1970 : tensor<32x1152x7x7xf32>
    %v1974 = stablehlo.subtract %v1968, %v1973 : tensor<32x1152x7x7xf32>
    %v1975 = stablehlo.multiply %v1974, %v1974 : tensor<32x1152x7x7xf32>
    %v1976 = stablehlo.reduce(%v1975 init: %v1969) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1977 = stablehlo.broadcast_in_dim %v1976, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1978 = stablehlo.divide %v1977, %v1970 : tensor<32x1152x7x7xf32>
    %v1979 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1980 = stablehlo.add %v1978, %v1979 : tensor<32x1152x7x7xf32>
    %v1981 = stablehlo.rsqrt %v1980 : tensor<32x1152x7x7xf32>
    %v1982 = stablehlo.multiply %v1974, %v1981 : tensor<32x1152x7x7xf32>
    %v1983 = stablehlo.reshape %v1933 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1984 = stablehlo.multiply %v1983, %v1982 : tensor<32x1152x7x7xf32>
    %v1985 = stablehlo.reduce(%v1984 init: %v1969) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1986 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v1987 = stablehlo.multiply %v1985, %v1986 : tensor<1152xf32>
    %v1988 = stablehlo.subtract %b16dg, %v1987 : tensor<1152xf32>
    %v1989 = stablehlo.reshape %v1933 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1990 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1991 = stablehlo.reduce(%v1989 init: %v1990) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1992 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v1993 = stablehlo.multiply %v1991, %v1992 : tensor<1152xf32>
    %v1994 = stablehlo.subtract %b16dbt, %v1993 : tensor<1152xf32>
    %v1995 = stablehlo.reshape %v1564 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1996 = stablehlo.reshape %v1963 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1997 = stablehlo.transpose %v1995, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v1998 = stablehlo.transpose %v1996, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v1999 = stablehlo.convolution(%v1997, %v1998)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1152 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<1x1152x3x3xf32>
    %v2000 = stablehlo.reshape %v1999 : (tensor<1x1152x3x3xf32>) -> tensor<1152x1x3x3xf32>
    %v2001 = stablehlo.constant dense<0.05> : tensor<1152x1x3x3xf32>
    %v2002 = stablehlo.multiply %v2000, %v2001 : tensor<1152x1x3x3xf32>
    %v2003 = stablehlo.subtract %b16dW, %v2002 : tensor<1152x1x3x3xf32>
    %v2004 = stablehlo.reshape %v1967 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2005 = stablehlo.reshape %v1560 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2006 = stablehlo.logistic %v2005 : tensor<32x1152x7x7xf32>
    %v2007 = stablehlo.constant dense<1.0> : tensor<32x1152x7x7xf32>
    %v2008 = stablehlo.subtract %v2007, %v2006 : tensor<32x1152x7x7xf32>
    %v2009 = stablehlo.multiply %v2005, %v2008 : tensor<32x1152x7x7xf32>
    %v2010 = stablehlo.add %v2007, %v2009 : tensor<32x1152x7x7xf32>
    %v2011 = stablehlo.multiply %v2006, %v2010 : tensor<32x1152x7x7xf32>
    %v2012 = stablehlo.multiply %v2004, %v2011 : tensor<32x1152x7x7xf32>
    %v2013 = stablehlo.reshape %v2012 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2014 = stablehlo.reshape %v1540 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2015 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2016 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2017 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2018 = stablehlo.reduce(%v2014 init: %v2015) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2019 = stablehlo.broadcast_in_dim %v2018, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2020 = stablehlo.divide %v2019, %v2016 : tensor<32x1152x7x7xf32>
    %v2021 = stablehlo.subtract %v2014, %v2020 : tensor<32x1152x7x7xf32>
    %v2022 = stablehlo.multiply %v2021, %v2021 : tensor<32x1152x7x7xf32>
    %v2023 = stablehlo.reduce(%v2022 init: %v2015) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2024 = stablehlo.broadcast_in_dim %v2023, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2025 = stablehlo.divide %v2024, %v2016 : tensor<32x1152x7x7xf32>
    %v2026 = stablehlo.add %v2025, %v2017 : tensor<32x1152x7x7xf32>
    %v2027 = stablehlo.rsqrt %v2026 : tensor<32x1152x7x7xf32>
    %v2028 = stablehlo.multiply %v2021, %v2027 : tensor<32x1152x7x7xf32>
    %v2029 = stablehlo.broadcast_in_dim %b16eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2030 = stablehlo.reshape %v2013 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2031 = stablehlo.multiply %v2029, %v2030 : tensor<32x1152x7x7xf32>
    %v2032 = stablehlo.reduce(%v2031 init: %v2015) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2033 = stablehlo.broadcast_in_dim %v2032, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2034 = stablehlo.multiply %v2028, %v2031 : tensor<32x1152x7x7xf32>
    %v2035 = stablehlo.reduce(%v2034 init: %v2015) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2036 = stablehlo.broadcast_in_dim %v2035, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2037 = stablehlo.multiply %v2031, %v2016 : tensor<32x1152x7x7xf32>
    %v2038 = stablehlo.subtract %v2037, %v2033 : tensor<32x1152x7x7xf32>
    %v2039 = stablehlo.multiply %v2028, %v2036 : tensor<32x1152x7x7xf32>
    %v2040 = stablehlo.subtract %v2038, %v2039 : tensor<32x1152x7x7xf32>
    %v2041 = stablehlo.divide %v2027, %v2016 : tensor<32x1152x7x7xf32>
    %v2042 = stablehlo.multiply %v2041, %v2040 : tensor<32x1152x7x7xf32>
    %v2043 = stablehlo.reshape %v2042 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2044 = stablehlo.reshape %v2043 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2045 = stablehlo.reverse %b16eW, dims = [2, 3] : tensor<1152x192x1x1xf32>
    %v2046 = stablehlo.transpose %v2045, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %v2047 = stablehlo.convolution(%v2044, %v2046)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v2048 = stablehlo.reshape %v2047 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v2049 = stablehlo.reshape %v1540 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2050 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2051 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2052 = stablehlo.reduce(%v2049 init: %v2050) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2053 = stablehlo.broadcast_in_dim %v2052, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2054 = stablehlo.divide %v2053, %v2051 : tensor<32x1152x7x7xf32>
    %v2055 = stablehlo.subtract %v2049, %v2054 : tensor<32x1152x7x7xf32>
    %v2056 = stablehlo.multiply %v2055, %v2055 : tensor<32x1152x7x7xf32>
    %v2057 = stablehlo.reduce(%v2056 init: %v2050) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2058 = stablehlo.broadcast_in_dim %v2057, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2059 = stablehlo.divide %v2058, %v2051 : tensor<32x1152x7x7xf32>
    %v2060 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2061 = stablehlo.add %v2059, %v2060 : tensor<32x1152x7x7xf32>
    %v2062 = stablehlo.rsqrt %v2061 : tensor<32x1152x7x7xf32>
    %v2063 = stablehlo.multiply %v2055, %v2062 : tensor<32x1152x7x7xf32>
    %v2064 = stablehlo.reshape %v2013 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2065 = stablehlo.multiply %v2064, %v2063 : tensor<32x1152x7x7xf32>
    %v2066 = stablehlo.reduce(%v2065 init: %v2050) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2067 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2068 = stablehlo.multiply %v2066, %v2067 : tensor<1152xf32>
    %v2069 = stablehlo.subtract %b16eg, %v2068 : tensor<1152xf32>
    %v2070 = stablehlo.reshape %v2013 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2071 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2072 = stablehlo.reduce(%v2070 init: %v2071) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2073 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2074 = stablehlo.multiply %v2072, %v2073 : tensor<1152xf32>
    %v2075 = stablehlo.subtract %b16ebt, %v2074 : tensor<1152xf32>
    %v2076 = stablehlo.reshape %v1535 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2077 = stablehlo.reshape %v2043 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2078 = stablehlo.transpose %v2076, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %v2079 = stablehlo.transpose %v2077, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2080 = stablehlo.convolution(%v2078, %v2079)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<192x1152x1x1xf32>
    %v2081 = stablehlo.transpose %v2080, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %v2082 = stablehlo.constant dense<0.05> : tensor<1152x192x1x1xf32>
    %v2083 = stablehlo.multiply %v2081, %v2082 : tensor<1152x192x1x1xf32>
    %v2084 = stablehlo.subtract %b16eW, %v2083 : tensor<1152x192x1x1xf32>
    %v2085 = stablehlo.reshape %v1511 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2086 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2087 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v2088 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v2089 = stablehlo.reduce(%v2085 init: %v2086) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2090 = stablehlo.broadcast_in_dim %v2089, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2091 = stablehlo.divide %v2090, %v2087 : tensor<32x192x7x7xf32>
    %v2092 = stablehlo.subtract %v2085, %v2091 : tensor<32x192x7x7xf32>
    %v2093 = stablehlo.multiply %v2092, %v2092 : tensor<32x192x7x7xf32>
    %v2094 = stablehlo.reduce(%v2093 init: %v2086) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2095 = stablehlo.broadcast_in_dim %v2094, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2096 = stablehlo.divide %v2095, %v2087 : tensor<32x192x7x7xf32>
    %v2097 = stablehlo.add %v2096, %v2088 : tensor<32x192x7x7xf32>
    %v2098 = stablehlo.rsqrt %v2097 : tensor<32x192x7x7xf32>
    %v2099 = stablehlo.multiply %v2092, %v2098 : tensor<32x192x7x7xf32>
    %v2100 = stablehlo.broadcast_in_dim %b15pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2101 = stablehlo.reshape %v2048 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2102 = stablehlo.multiply %v2100, %v2101 : tensor<32x192x7x7xf32>
    %v2103 = stablehlo.reduce(%v2102 init: %v2086) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2104 = stablehlo.broadcast_in_dim %v2103, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2105 = stablehlo.multiply %v2099, %v2102 : tensor<32x192x7x7xf32>
    %v2106 = stablehlo.reduce(%v2105 init: %v2086) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2107 = stablehlo.broadcast_in_dim %v2106, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2108 = stablehlo.multiply %v2102, %v2087 : tensor<32x192x7x7xf32>
    %v2109 = stablehlo.subtract %v2108, %v2104 : tensor<32x192x7x7xf32>
    %v2110 = stablehlo.multiply %v2099, %v2107 : tensor<32x192x7x7xf32>
    %v2111 = stablehlo.subtract %v2109, %v2110 : tensor<32x192x7x7xf32>
    %v2112 = stablehlo.divide %v2098, %v2087 : tensor<32x192x7x7xf32>
    %v2113 = stablehlo.multiply %v2112, %v2111 : tensor<32x192x7x7xf32>
    %v2114 = stablehlo.reshape %v2113 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v2115 = stablehlo.reshape %v2114 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2116 = stablehlo.reverse %b15pW, dims = [2, 3] : tensor<192x1152x1x1xf32>
    %v2117 = stablehlo.transpose %v2116, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %v2118 = stablehlo.convolution(%v2115, %v2117)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v2119 = stablehlo.reshape %v2118 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2120 = stablehlo.reshape %v1511 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2121 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2122 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v2123 = stablehlo.reduce(%v2120 init: %v2121) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2124 = stablehlo.broadcast_in_dim %v2123, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2125 = stablehlo.divide %v2124, %v2122 : tensor<32x192x7x7xf32>
    %v2126 = stablehlo.subtract %v2120, %v2125 : tensor<32x192x7x7xf32>
    %v2127 = stablehlo.multiply %v2126, %v2126 : tensor<32x192x7x7xf32>
    %v2128 = stablehlo.reduce(%v2127 init: %v2121) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2129 = stablehlo.broadcast_in_dim %v2128, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2130 = stablehlo.divide %v2129, %v2122 : tensor<32x192x7x7xf32>
    %v2131 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v2132 = stablehlo.add %v2130, %v2131 : tensor<32x192x7x7xf32>
    %v2133 = stablehlo.rsqrt %v2132 : tensor<32x192x7x7xf32>
    %v2134 = stablehlo.multiply %v2126, %v2133 : tensor<32x192x7x7xf32>
    %v2135 = stablehlo.reshape %v2048 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2136 = stablehlo.multiply %v2135, %v2134 : tensor<32x192x7x7xf32>
    %v2137 = stablehlo.reduce(%v2136 init: %v2121) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2138 = stablehlo.constant dense<0.05> : tensor<192xf32>
    %v2139 = stablehlo.multiply %v2137, %v2138 : tensor<192xf32>
    %v2140 = stablehlo.subtract %b15pg, %v2139 : tensor<192xf32>
    %v2141 = stablehlo.reshape %v2048 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2142 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2143 = stablehlo.reduce(%v2141 init: %v2142) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2144 = stablehlo.constant dense<0.05> : tensor<192xf32>
    %v2145 = stablehlo.multiply %v2143, %v2144 : tensor<192xf32>
    %v2146 = stablehlo.subtract %b15pbt, %v2145 : tensor<192xf32>
    %v2147 = stablehlo.reshape %v1506 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2148 = stablehlo.reshape %v2114 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2149 = stablehlo.transpose %v2147, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2150 = stablehlo.transpose %v2148, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %v2151 = stablehlo.convolution(%v2149, %v2150)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<192x32x7x7xf32>) -> tensor<1152x192x1x1xf32>
    %v2152 = stablehlo.transpose %v2151, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %v2153 = stablehlo.constant dense<0.05> : tensor<192x1152x1x1xf32>
    %v2154 = stablehlo.multiply %v2152, %v2153 : tensor<192x1152x1x1xf32>
    %v2155 = stablehlo.subtract %b15pW, %v2154 : tensor<192x1152x1x1xf32>
    %v2156 = stablehlo.reshape %v1489 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2157 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2158 = stablehlo.reduce(%v2156 init: %v2157) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2159 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v2160 = stablehlo.divide %v2158, %v2159 : tensor<32x1152xf32>
    %v2161 = stablehlo.dot_general %v2160, %b15zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v2162 = stablehlo.broadcast_in_dim %b15zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v2163 = stablehlo.add %v2161, %v2162 : tensor<32x48xf32>
    %v2164 = stablehlo.logistic %v2163 : tensor<32x48xf32>
    %v2165 = stablehlo.multiply %v2163, %v2164 : tensor<32x48xf32>
    %v2166 = stablehlo.dot_general %v2165, %b15zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v2167 = stablehlo.broadcast_in_dim %b15zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v2168 = stablehlo.add %v2166, %v2167 : tensor<32x1152xf32>
    %v2169 = stablehlo.logistic %v2168 : tensor<32x1152xf32>
    %v2170 = stablehlo.reshape %v2119 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2171 = stablehlo.broadcast_in_dim %v2169, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2172 = stablehlo.multiply %v2171, %v2170 : tensor<32x1152x7x7xf32>
    %v2173 = stablehlo.multiply %v2156, %v2170 : tensor<32x1152x7x7xf32>
    %v2174 = stablehlo.reduce(%v2173 init: %v2157) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2175 = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %v2176 = stablehlo.subtract %v2175, %v2169 : tensor<32x1152xf32>
    %v2177 = stablehlo.multiply %v2169, %v2176 : tensor<32x1152xf32>
    %v2178 = stablehlo.multiply %v2174, %v2177 : tensor<32x1152xf32>
    %v2179 = stablehlo.dot_general %v2178, %b15zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<48x1152xf32>) -> tensor<32x48xf32>
    %v2180 = stablehlo.logistic %v2163 : tensor<32x48xf32>
    %v2181 = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %v2182 = stablehlo.subtract %v2181, %v2180 : tensor<32x48xf32>
    %v2183 = stablehlo.multiply %v2163, %v2182 : tensor<32x48xf32>
    %v2184 = stablehlo.add %v2181, %v2183 : tensor<32x48xf32>
    %v2185 = stablehlo.multiply %v2180, %v2184 : tensor<32x48xf32>
    %v2186 = stablehlo.multiply %v2179, %v2185 : tensor<32x48xf32>
    %v2187 = stablehlo.dot_general %v2186, %b15zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<1152x48xf32>) -> tensor<32x1152xf32>
    %v2188 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v2189 = stablehlo.divide %v2187, %v2188 : tensor<32x1152xf32>
    %v2190 = stablehlo.broadcast_in_dim %v2189, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2191 = stablehlo.add %v2172, %v2190 : tensor<32x1152x7x7xf32>
    %v2192 = stablehlo.reshape %v2191 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2193 = stablehlo.reshape %v1489 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2194 = stablehlo.reshape %v2119 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2195 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2196 = stablehlo.multiply %v2193, %v2194 : tensor<32x1152x7x7xf32>
    %v2197 = stablehlo.reduce(%v2196 init: %v2195) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2198 = stablehlo.logistic %v1502 : tensor<32x1152xf32>
    %v2199 = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %v2200 = stablehlo.subtract %v2199, %v2198 : tensor<32x1152xf32>
    %v2201 = stablehlo.multiply %v2198, %v2200 : tensor<32x1152xf32>
    %v2202 = stablehlo.multiply %v2197, %v2201 : tensor<32x1152xf32>
    %v2203 = stablehlo.dot_general %v1499, %v2202, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<32x1152xf32>) -> tensor<48x1152xf32>
    %v2204 = stablehlo.constant dense<0.05> : tensor<48x1152xf32>
    %v2205 = stablehlo.multiply %v2203, %v2204 : tensor<48x1152xf32>
    %v2206 = stablehlo.subtract %b15zW2, %v2205 : tensor<48x1152xf32>
    %v2207 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2208 = stablehlo.reduce(%v2202 init: %v2207) applies stablehlo.add across dimensions = [0] : (tensor<32x1152xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2209 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2210 = stablehlo.multiply %v2208, %v2209 : tensor<1152xf32>
    %v2211 = stablehlo.subtract %b15zb2, %v2210 : tensor<1152xf32>
    %v2212 = stablehlo.reshape %v2202 : (tensor<32x1152xf32>) -> tensor<32x1x1152xf32>
    %v2213 = stablehlo.dot_general %v2212, %b15zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x1152xf32>, tensor<48x1152xf32>) -> tensor<32x1x48xf32>
    %v2214 = stablehlo.reshape %v2213 : (tensor<32x1x48xf32>) -> tensor<32x48xf32>
    %v2215 = stablehlo.logistic %v1497 : tensor<32x48xf32>
    %v2216 = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %v2217 = stablehlo.subtract %v2216, %v2215 : tensor<32x48xf32>
    %v2218 = stablehlo.multiply %v1497, %v2217 : tensor<32x48xf32>
    %v2219 = stablehlo.add %v2216, %v2218 : tensor<32x48xf32>
    %v2220 = stablehlo.multiply %v2215, %v2219 : tensor<32x48xf32>
    %v2221 = stablehlo.multiply %v2214, %v2220 : tensor<32x48xf32>
    %v2222 = stablehlo.dot_general %v1494, %v2221, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<32x48xf32>) -> tensor<1152x48xf32>
    %v2223 = stablehlo.constant dense<0.05> : tensor<1152x48xf32>
    %v2224 = stablehlo.multiply %v2222, %v2223 : tensor<1152x48xf32>
    %v2225 = stablehlo.subtract %b15zW1, %v2224 : tensor<1152x48xf32>
    %v2226 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2227 = stablehlo.reduce(%v2221 init: %v2226) applies stablehlo.add across dimensions = [0] : (tensor<32x48xf32>, tensor<f32>) -> tensor<48xf32>
    %v2228 = stablehlo.constant dense<0.05> : tensor<48xf32>
    %v2229 = stablehlo.multiply %v2227, %v2228 : tensor<48xf32>
    %v2230 = stablehlo.subtract %b15zb1, %v2229 : tensor<48xf32>
    %v2231 = stablehlo.reshape %v2192 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2232 = stablehlo.reshape %v1485 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2233 = stablehlo.logistic %v2232 : tensor<32x1152x7x7xf32>
    %v2234 = stablehlo.constant dense<1.0> : tensor<32x1152x7x7xf32>
    %v2235 = stablehlo.subtract %v2234, %v2233 : tensor<32x1152x7x7xf32>
    %v2236 = stablehlo.multiply %v2232, %v2235 : tensor<32x1152x7x7xf32>
    %v2237 = stablehlo.add %v2234, %v2236 : tensor<32x1152x7x7xf32>
    %v2238 = stablehlo.multiply %v2233, %v2237 : tensor<32x1152x7x7xf32>
    %v2239 = stablehlo.multiply %v2231, %v2238 : tensor<32x1152x7x7xf32>
    %v2240 = stablehlo.reshape %v2239 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2241 = stablehlo.reshape %v1465 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2242 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2243 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2244 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2245 = stablehlo.reduce(%v2241 init: %v2242) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2246 = stablehlo.broadcast_in_dim %v2245, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2247 = stablehlo.divide %v2246, %v2243 : tensor<32x1152x7x7xf32>
    %v2248 = stablehlo.subtract %v2241, %v2247 : tensor<32x1152x7x7xf32>
    %v2249 = stablehlo.multiply %v2248, %v2248 : tensor<32x1152x7x7xf32>
    %v2250 = stablehlo.reduce(%v2249 init: %v2242) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2251 = stablehlo.broadcast_in_dim %v2250, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2252 = stablehlo.divide %v2251, %v2243 : tensor<32x1152x7x7xf32>
    %v2253 = stablehlo.add %v2252, %v2244 : tensor<32x1152x7x7xf32>
    %v2254 = stablehlo.rsqrt %v2253 : tensor<32x1152x7x7xf32>
    %v2255 = stablehlo.multiply %v2248, %v2254 : tensor<32x1152x7x7xf32>
    %v2256 = stablehlo.broadcast_in_dim %b15dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2257 = stablehlo.reshape %v2240 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2258 = stablehlo.multiply %v2256, %v2257 : tensor<32x1152x7x7xf32>
    %v2259 = stablehlo.reduce(%v2258 init: %v2242) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2260 = stablehlo.broadcast_in_dim %v2259, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2261 = stablehlo.multiply %v2255, %v2258 : tensor<32x1152x7x7xf32>
    %v2262 = stablehlo.reduce(%v2261 init: %v2242) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2263 = stablehlo.broadcast_in_dim %v2262, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2264 = stablehlo.multiply %v2258, %v2243 : tensor<32x1152x7x7xf32>
    %v2265 = stablehlo.subtract %v2264, %v2260 : tensor<32x1152x7x7xf32>
    %v2266 = stablehlo.multiply %v2255, %v2263 : tensor<32x1152x7x7xf32>
    %v2267 = stablehlo.subtract %v2265, %v2266 : tensor<32x1152x7x7xf32>
    %v2268 = stablehlo.divide %v2254, %v2243 : tensor<32x1152x7x7xf32>
    %v2269 = stablehlo.multiply %v2268, %v2267 : tensor<32x1152x7x7xf32>
    %v2270 = stablehlo.reshape %v2269 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2271 = stablehlo.reshape %v2270 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2272 = stablehlo.reverse %b15dW, dims = [2, 3] : tensor<1152x1x5x5xf32>
    %v2273 = stablehlo.convolution(%v2271, %v2272)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v2274 = stablehlo.reshape %v2273 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2275 = stablehlo.reshape %v1465 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2276 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2277 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2278 = stablehlo.reduce(%v2275 init: %v2276) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2279 = stablehlo.broadcast_in_dim %v2278, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2280 = stablehlo.divide %v2279, %v2277 : tensor<32x1152x7x7xf32>
    %v2281 = stablehlo.subtract %v2275, %v2280 : tensor<32x1152x7x7xf32>
    %v2282 = stablehlo.multiply %v2281, %v2281 : tensor<32x1152x7x7xf32>
    %v2283 = stablehlo.reduce(%v2282 init: %v2276) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2284 = stablehlo.broadcast_in_dim %v2283, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2285 = stablehlo.divide %v2284, %v2277 : tensor<32x1152x7x7xf32>
    %v2286 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2287 = stablehlo.add %v2285, %v2286 : tensor<32x1152x7x7xf32>
    %v2288 = stablehlo.rsqrt %v2287 : tensor<32x1152x7x7xf32>
    %v2289 = stablehlo.multiply %v2281, %v2288 : tensor<32x1152x7x7xf32>
    %v2290 = stablehlo.reshape %v2240 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2291 = stablehlo.multiply %v2290, %v2289 : tensor<32x1152x7x7xf32>
    %v2292 = stablehlo.reduce(%v2291 init: %v2276) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2293 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2294 = stablehlo.multiply %v2292, %v2293 : tensor<1152xf32>
    %v2295 = stablehlo.subtract %b15dg, %v2294 : tensor<1152xf32>
    %v2296 = stablehlo.reshape %v2240 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2297 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2298 = stablehlo.reduce(%v2296 init: %v2297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2299 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2300 = stablehlo.multiply %v2298, %v2299 : tensor<1152xf32>
    %v2301 = stablehlo.subtract %b15dbt, %v2300 : tensor<1152xf32>
    %v2302 = stablehlo.reshape %v1460 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2303 = stablehlo.reshape %v2270 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2304 = stablehlo.transpose %v2302, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2305 = stablehlo.transpose %v2303, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2306 = stablehlo.convolution(%v2304, %v2305)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1152 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<1x1152x5x5xf32>
    %v2307 = stablehlo.reshape %v2306 : (tensor<1x1152x5x5xf32>) -> tensor<1152x1x5x5xf32>
    %v2308 = stablehlo.constant dense<0.05> : tensor<1152x1x5x5xf32>
    %v2309 = stablehlo.multiply %v2307, %v2308 : tensor<1152x1x5x5xf32>
    %v2310 = stablehlo.subtract %b15dW, %v2309 : tensor<1152x1x5x5xf32>
    %v2311 = stablehlo.reshape %v2274 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2312 = stablehlo.reshape %v1456 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2313 = stablehlo.logistic %v2312 : tensor<32x1152x7x7xf32>
    %v2314 = stablehlo.constant dense<1.0> : tensor<32x1152x7x7xf32>
    %v2315 = stablehlo.subtract %v2314, %v2313 : tensor<32x1152x7x7xf32>
    %v2316 = stablehlo.multiply %v2312, %v2315 : tensor<32x1152x7x7xf32>
    %v2317 = stablehlo.add %v2314, %v2316 : tensor<32x1152x7x7xf32>
    %v2318 = stablehlo.multiply %v2313, %v2317 : tensor<32x1152x7x7xf32>
    %v2319 = stablehlo.multiply %v2311, %v2318 : tensor<32x1152x7x7xf32>
    %v2320 = stablehlo.reshape %v2319 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2321 = stablehlo.reshape %v1436 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2322 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2323 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2324 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2325 = stablehlo.reduce(%v2321 init: %v2322) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2326 = stablehlo.broadcast_in_dim %v2325, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2327 = stablehlo.divide %v2326, %v2323 : tensor<32x1152x7x7xf32>
    %v2328 = stablehlo.subtract %v2321, %v2327 : tensor<32x1152x7x7xf32>
    %v2329 = stablehlo.multiply %v2328, %v2328 : tensor<32x1152x7x7xf32>
    %v2330 = stablehlo.reduce(%v2329 init: %v2322) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2331 = stablehlo.broadcast_in_dim %v2330, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2332 = stablehlo.divide %v2331, %v2323 : tensor<32x1152x7x7xf32>
    %v2333 = stablehlo.add %v2332, %v2324 : tensor<32x1152x7x7xf32>
    %v2334 = stablehlo.rsqrt %v2333 : tensor<32x1152x7x7xf32>
    %v2335 = stablehlo.multiply %v2328, %v2334 : tensor<32x1152x7x7xf32>
    %v2336 = stablehlo.broadcast_in_dim %b15eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2337 = stablehlo.reshape %v2320 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2338 = stablehlo.multiply %v2336, %v2337 : tensor<32x1152x7x7xf32>
    %v2339 = stablehlo.reduce(%v2338 init: %v2322) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2340 = stablehlo.broadcast_in_dim %v2339, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2341 = stablehlo.multiply %v2335, %v2338 : tensor<32x1152x7x7xf32>
    %v2342 = stablehlo.reduce(%v2341 init: %v2322) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2343 = stablehlo.broadcast_in_dim %v2342, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2344 = stablehlo.multiply %v2338, %v2323 : tensor<32x1152x7x7xf32>
    %v2345 = stablehlo.subtract %v2344, %v2340 : tensor<32x1152x7x7xf32>
    %v2346 = stablehlo.multiply %v2335, %v2343 : tensor<32x1152x7x7xf32>
    %v2347 = stablehlo.subtract %v2345, %v2346 : tensor<32x1152x7x7xf32>
    %v2348 = stablehlo.divide %v2334, %v2323 : tensor<32x1152x7x7xf32>
    %v2349 = stablehlo.multiply %v2348, %v2347 : tensor<32x1152x7x7xf32>
    %v2350 = stablehlo.reshape %v2349 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2351 = stablehlo.reshape %v2350 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2352 = stablehlo.reverse %b15eW, dims = [2, 3] : tensor<1152x192x1x1xf32>
    %v2353 = stablehlo.transpose %v2352, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %v2354 = stablehlo.convolution(%v2351, %v2353)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v2355 = stablehlo.reshape %v2354 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v2356 = stablehlo.reshape %v1436 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2357 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2358 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2359 = stablehlo.reduce(%v2356 init: %v2357) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2360 = stablehlo.broadcast_in_dim %v2359, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2361 = stablehlo.divide %v2360, %v2358 : tensor<32x1152x7x7xf32>
    %v2362 = stablehlo.subtract %v2356, %v2361 : tensor<32x1152x7x7xf32>
    %v2363 = stablehlo.multiply %v2362, %v2362 : tensor<32x1152x7x7xf32>
    %v2364 = stablehlo.reduce(%v2363 init: %v2357) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2365 = stablehlo.broadcast_in_dim %v2364, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2366 = stablehlo.divide %v2365, %v2358 : tensor<32x1152x7x7xf32>
    %v2367 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2368 = stablehlo.add %v2366, %v2367 : tensor<32x1152x7x7xf32>
    %v2369 = stablehlo.rsqrt %v2368 : tensor<32x1152x7x7xf32>
    %v2370 = stablehlo.multiply %v2362, %v2369 : tensor<32x1152x7x7xf32>
    %v2371 = stablehlo.reshape %v2320 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2372 = stablehlo.multiply %v2371, %v2370 : tensor<32x1152x7x7xf32>
    %v2373 = stablehlo.reduce(%v2372 init: %v2357) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2374 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2375 = stablehlo.multiply %v2373, %v2374 : tensor<1152xf32>
    %v2376 = stablehlo.subtract %b15eg, %v2375 : tensor<1152xf32>
    %v2377 = stablehlo.reshape %v2320 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2378 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2379 = stablehlo.reduce(%v2377 init: %v2378) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2380 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2381 = stablehlo.multiply %v2379, %v2380 : tensor<1152xf32>
    %v2382 = stablehlo.subtract %b15ebt, %v2381 : tensor<1152xf32>
    %v2383 = stablehlo.reshape %v1431 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2384 = stablehlo.reshape %v2350 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2385 = stablehlo.transpose %v2383, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %v2386 = stablehlo.transpose %v2384, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2387 = stablehlo.convolution(%v2385, %v2386)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<192x1152x1x1xf32>
    %v2388 = stablehlo.transpose %v2387, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %v2389 = stablehlo.constant dense<0.05> : tensor<1152x192x1x1xf32>
    %v2390 = stablehlo.multiply %v2388, %v2389 : tensor<1152x192x1x1xf32>
    %v2391 = stablehlo.subtract %b15eW, %v2390 : tensor<1152x192x1x1xf32>
    %v2392 = stablehlo.reshape %v2355 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2393 = stablehlo.reshape %v2048 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2394 = stablehlo.add %v2392, %v2393 : tensor<32x192x7x7xf32>
    %v2395 = stablehlo.reshape %v2394 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v2396 = stablehlo.reshape %v1407 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2397 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2398 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v2399 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v2400 = stablehlo.reduce(%v2396 init: %v2397) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2401 = stablehlo.broadcast_in_dim %v2400, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2402 = stablehlo.divide %v2401, %v2398 : tensor<32x192x7x7xf32>
    %v2403 = stablehlo.subtract %v2396, %v2402 : tensor<32x192x7x7xf32>
    %v2404 = stablehlo.multiply %v2403, %v2403 : tensor<32x192x7x7xf32>
    %v2405 = stablehlo.reduce(%v2404 init: %v2397) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2406 = stablehlo.broadcast_in_dim %v2405, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2407 = stablehlo.divide %v2406, %v2398 : tensor<32x192x7x7xf32>
    %v2408 = stablehlo.add %v2407, %v2399 : tensor<32x192x7x7xf32>
    %v2409 = stablehlo.rsqrt %v2408 : tensor<32x192x7x7xf32>
    %v2410 = stablehlo.multiply %v2403, %v2409 : tensor<32x192x7x7xf32>
    %v2411 = stablehlo.broadcast_in_dim %b14pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2412 = stablehlo.reshape %v2395 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2413 = stablehlo.multiply %v2411, %v2412 : tensor<32x192x7x7xf32>
    %v2414 = stablehlo.reduce(%v2413 init: %v2397) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2415 = stablehlo.broadcast_in_dim %v2414, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2416 = stablehlo.multiply %v2410, %v2413 : tensor<32x192x7x7xf32>
    %v2417 = stablehlo.reduce(%v2416 init: %v2397) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2418 = stablehlo.broadcast_in_dim %v2417, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2419 = stablehlo.multiply %v2413, %v2398 : tensor<32x192x7x7xf32>
    %v2420 = stablehlo.subtract %v2419, %v2415 : tensor<32x192x7x7xf32>
    %v2421 = stablehlo.multiply %v2410, %v2418 : tensor<32x192x7x7xf32>
    %v2422 = stablehlo.subtract %v2420, %v2421 : tensor<32x192x7x7xf32>
    %v2423 = stablehlo.divide %v2409, %v2398 : tensor<32x192x7x7xf32>
    %v2424 = stablehlo.multiply %v2423, %v2422 : tensor<32x192x7x7xf32>
    %v2425 = stablehlo.reshape %v2424 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v2426 = stablehlo.reshape %v2425 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2427 = stablehlo.reverse %b14pW, dims = [2, 3] : tensor<192x1152x1x1xf32>
    %v2428 = stablehlo.transpose %v2427, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %v2429 = stablehlo.convolution(%v2426, %v2428)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v2430 = stablehlo.reshape %v2429 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2431 = stablehlo.reshape %v1407 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2432 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2433 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v2434 = stablehlo.reduce(%v2431 init: %v2432) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2435 = stablehlo.broadcast_in_dim %v2434, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2436 = stablehlo.divide %v2435, %v2433 : tensor<32x192x7x7xf32>
    %v2437 = stablehlo.subtract %v2431, %v2436 : tensor<32x192x7x7xf32>
    %v2438 = stablehlo.multiply %v2437, %v2437 : tensor<32x192x7x7xf32>
    %v2439 = stablehlo.reduce(%v2438 init: %v2432) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2440 = stablehlo.broadcast_in_dim %v2439, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2441 = stablehlo.divide %v2440, %v2433 : tensor<32x192x7x7xf32>
    %v2442 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v2443 = stablehlo.add %v2441, %v2442 : tensor<32x192x7x7xf32>
    %v2444 = stablehlo.rsqrt %v2443 : tensor<32x192x7x7xf32>
    %v2445 = stablehlo.multiply %v2437, %v2444 : tensor<32x192x7x7xf32>
    %v2446 = stablehlo.reshape %v2395 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2447 = stablehlo.multiply %v2446, %v2445 : tensor<32x192x7x7xf32>
    %v2448 = stablehlo.reduce(%v2447 init: %v2432) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2449 = stablehlo.constant dense<0.05> : tensor<192xf32>
    %v2450 = stablehlo.multiply %v2448, %v2449 : tensor<192xf32>
    %v2451 = stablehlo.subtract %b14pg, %v2450 : tensor<192xf32>
    %v2452 = stablehlo.reshape %v2395 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2453 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2454 = stablehlo.reduce(%v2452 init: %v2453) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2455 = stablehlo.constant dense<0.05> : tensor<192xf32>
    %v2456 = stablehlo.multiply %v2454, %v2455 : tensor<192xf32>
    %v2457 = stablehlo.subtract %b14pbt, %v2456 : tensor<192xf32>
    %v2458 = stablehlo.reshape %v1402 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2459 = stablehlo.reshape %v2425 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2460 = stablehlo.transpose %v2458, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2461 = stablehlo.transpose %v2459, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %v2462 = stablehlo.convolution(%v2460, %v2461)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<192x32x7x7xf32>) -> tensor<1152x192x1x1xf32>
    %v2463 = stablehlo.transpose %v2462, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %v2464 = stablehlo.constant dense<0.05> : tensor<192x1152x1x1xf32>
    %v2465 = stablehlo.multiply %v2463, %v2464 : tensor<192x1152x1x1xf32>
    %v2466 = stablehlo.subtract %b14pW, %v2465 : tensor<192x1152x1x1xf32>
    %v2467 = stablehlo.reshape %v1385 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2468 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2469 = stablehlo.reduce(%v2467 init: %v2468) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2470 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v2471 = stablehlo.divide %v2469, %v2470 : tensor<32x1152xf32>
    %v2472 = stablehlo.dot_general %v2471, %b14zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v2473 = stablehlo.broadcast_in_dim %b14zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v2474 = stablehlo.add %v2472, %v2473 : tensor<32x48xf32>
    %v2475 = stablehlo.logistic %v2474 : tensor<32x48xf32>
    %v2476 = stablehlo.multiply %v2474, %v2475 : tensor<32x48xf32>
    %v2477 = stablehlo.dot_general %v2476, %b14zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v2478 = stablehlo.broadcast_in_dim %b14zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v2479 = stablehlo.add %v2477, %v2478 : tensor<32x1152xf32>
    %v2480 = stablehlo.logistic %v2479 : tensor<32x1152xf32>
    %v2481 = stablehlo.reshape %v2430 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2482 = stablehlo.broadcast_in_dim %v2480, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2483 = stablehlo.multiply %v2482, %v2481 : tensor<32x1152x7x7xf32>
    %v2484 = stablehlo.multiply %v2467, %v2481 : tensor<32x1152x7x7xf32>
    %v2485 = stablehlo.reduce(%v2484 init: %v2468) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2486 = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %v2487 = stablehlo.subtract %v2486, %v2480 : tensor<32x1152xf32>
    %v2488 = stablehlo.multiply %v2480, %v2487 : tensor<32x1152xf32>
    %v2489 = stablehlo.multiply %v2485, %v2488 : tensor<32x1152xf32>
    %v2490 = stablehlo.dot_general %v2489, %b14zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<48x1152xf32>) -> tensor<32x48xf32>
    %v2491 = stablehlo.logistic %v2474 : tensor<32x48xf32>
    %v2492 = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %v2493 = stablehlo.subtract %v2492, %v2491 : tensor<32x48xf32>
    %v2494 = stablehlo.multiply %v2474, %v2493 : tensor<32x48xf32>
    %v2495 = stablehlo.add %v2492, %v2494 : tensor<32x48xf32>
    %v2496 = stablehlo.multiply %v2491, %v2495 : tensor<32x48xf32>
    %v2497 = stablehlo.multiply %v2490, %v2496 : tensor<32x48xf32>
    %v2498 = stablehlo.dot_general %v2497, %b14zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<1152x48xf32>) -> tensor<32x1152xf32>
    %v2499 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v2500 = stablehlo.divide %v2498, %v2499 : tensor<32x1152xf32>
    %v2501 = stablehlo.broadcast_in_dim %v2500, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2502 = stablehlo.add %v2483, %v2501 : tensor<32x1152x7x7xf32>
    %v2503 = stablehlo.reshape %v2502 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2504 = stablehlo.reshape %v1385 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2505 = stablehlo.reshape %v2430 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2506 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2507 = stablehlo.multiply %v2504, %v2505 : tensor<32x1152x7x7xf32>
    %v2508 = stablehlo.reduce(%v2507 init: %v2506) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2509 = stablehlo.logistic %v1398 : tensor<32x1152xf32>
    %v2510 = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %v2511 = stablehlo.subtract %v2510, %v2509 : tensor<32x1152xf32>
    %v2512 = stablehlo.multiply %v2509, %v2511 : tensor<32x1152xf32>
    %v2513 = stablehlo.multiply %v2508, %v2512 : tensor<32x1152xf32>
    %v2514 = stablehlo.dot_general %v1395, %v2513, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<32x1152xf32>) -> tensor<48x1152xf32>
    %v2515 = stablehlo.constant dense<0.05> : tensor<48x1152xf32>
    %v2516 = stablehlo.multiply %v2514, %v2515 : tensor<48x1152xf32>
    %v2517 = stablehlo.subtract %b14zW2, %v2516 : tensor<48x1152xf32>
    %v2518 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2519 = stablehlo.reduce(%v2513 init: %v2518) applies stablehlo.add across dimensions = [0] : (tensor<32x1152xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2520 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2521 = stablehlo.multiply %v2519, %v2520 : tensor<1152xf32>
    %v2522 = stablehlo.subtract %b14zb2, %v2521 : tensor<1152xf32>
    %v2523 = stablehlo.reshape %v2513 : (tensor<32x1152xf32>) -> tensor<32x1x1152xf32>
    %v2524 = stablehlo.dot_general %v2523, %b14zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x1152xf32>, tensor<48x1152xf32>) -> tensor<32x1x48xf32>
    %v2525 = stablehlo.reshape %v2524 : (tensor<32x1x48xf32>) -> tensor<32x48xf32>
    %v2526 = stablehlo.logistic %v1393 : tensor<32x48xf32>
    %v2527 = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %v2528 = stablehlo.subtract %v2527, %v2526 : tensor<32x48xf32>
    %v2529 = stablehlo.multiply %v1393, %v2528 : tensor<32x48xf32>
    %v2530 = stablehlo.add %v2527, %v2529 : tensor<32x48xf32>
    %v2531 = stablehlo.multiply %v2526, %v2530 : tensor<32x48xf32>
    %v2532 = stablehlo.multiply %v2525, %v2531 : tensor<32x48xf32>
    %v2533 = stablehlo.dot_general %v1390, %v2532, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<32x48xf32>) -> tensor<1152x48xf32>
    %v2534 = stablehlo.constant dense<0.05> : tensor<1152x48xf32>
    %v2535 = stablehlo.multiply %v2533, %v2534 : tensor<1152x48xf32>
    %v2536 = stablehlo.subtract %b14zW1, %v2535 : tensor<1152x48xf32>
    %v2537 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2538 = stablehlo.reduce(%v2532 init: %v2537) applies stablehlo.add across dimensions = [0] : (tensor<32x48xf32>, tensor<f32>) -> tensor<48xf32>
    %v2539 = stablehlo.constant dense<0.05> : tensor<48xf32>
    %v2540 = stablehlo.multiply %v2538, %v2539 : tensor<48xf32>
    %v2541 = stablehlo.subtract %b14zb1, %v2540 : tensor<48xf32>
    %v2542 = stablehlo.reshape %v2503 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2543 = stablehlo.reshape %v1381 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2544 = stablehlo.logistic %v2543 : tensor<32x1152x7x7xf32>
    %v2545 = stablehlo.constant dense<1.0> : tensor<32x1152x7x7xf32>
    %v2546 = stablehlo.subtract %v2545, %v2544 : tensor<32x1152x7x7xf32>
    %v2547 = stablehlo.multiply %v2543, %v2546 : tensor<32x1152x7x7xf32>
    %v2548 = stablehlo.add %v2545, %v2547 : tensor<32x1152x7x7xf32>
    %v2549 = stablehlo.multiply %v2544, %v2548 : tensor<32x1152x7x7xf32>
    %v2550 = stablehlo.multiply %v2542, %v2549 : tensor<32x1152x7x7xf32>
    %v2551 = stablehlo.reshape %v2550 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2552 = stablehlo.reshape %v1361 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2553 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2554 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2555 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2556 = stablehlo.reduce(%v2552 init: %v2553) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2557 = stablehlo.broadcast_in_dim %v2556, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2558 = stablehlo.divide %v2557, %v2554 : tensor<32x1152x7x7xf32>
    %v2559 = stablehlo.subtract %v2552, %v2558 : tensor<32x1152x7x7xf32>
    %v2560 = stablehlo.multiply %v2559, %v2559 : tensor<32x1152x7x7xf32>
    %v2561 = stablehlo.reduce(%v2560 init: %v2553) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2562 = stablehlo.broadcast_in_dim %v2561, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2563 = stablehlo.divide %v2562, %v2554 : tensor<32x1152x7x7xf32>
    %v2564 = stablehlo.add %v2563, %v2555 : tensor<32x1152x7x7xf32>
    %v2565 = stablehlo.rsqrt %v2564 : tensor<32x1152x7x7xf32>
    %v2566 = stablehlo.multiply %v2559, %v2565 : tensor<32x1152x7x7xf32>
    %v2567 = stablehlo.broadcast_in_dim %b14dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2568 = stablehlo.reshape %v2551 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2569 = stablehlo.multiply %v2567, %v2568 : tensor<32x1152x7x7xf32>
    %v2570 = stablehlo.reduce(%v2569 init: %v2553) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2571 = stablehlo.broadcast_in_dim %v2570, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2572 = stablehlo.multiply %v2566, %v2569 : tensor<32x1152x7x7xf32>
    %v2573 = stablehlo.reduce(%v2572 init: %v2553) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2574 = stablehlo.broadcast_in_dim %v2573, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2575 = stablehlo.multiply %v2569, %v2554 : tensor<32x1152x7x7xf32>
    %v2576 = stablehlo.subtract %v2575, %v2571 : tensor<32x1152x7x7xf32>
    %v2577 = stablehlo.multiply %v2566, %v2574 : tensor<32x1152x7x7xf32>
    %v2578 = stablehlo.subtract %v2576, %v2577 : tensor<32x1152x7x7xf32>
    %v2579 = stablehlo.divide %v2565, %v2554 : tensor<32x1152x7x7xf32>
    %v2580 = stablehlo.multiply %v2579, %v2578 : tensor<32x1152x7x7xf32>
    %v2581 = stablehlo.reshape %v2580 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2582 = stablehlo.reshape %v2581 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2583 = stablehlo.reverse %b14dW, dims = [2, 3] : tensor<1152x1x5x5xf32>
    %v2584 = stablehlo.convolution(%v2582, %v2583)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v2585 = stablehlo.reshape %v2584 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2586 = stablehlo.reshape %v1361 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2587 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2588 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2589 = stablehlo.reduce(%v2586 init: %v2587) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2590 = stablehlo.broadcast_in_dim %v2589, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2591 = stablehlo.divide %v2590, %v2588 : tensor<32x1152x7x7xf32>
    %v2592 = stablehlo.subtract %v2586, %v2591 : tensor<32x1152x7x7xf32>
    %v2593 = stablehlo.multiply %v2592, %v2592 : tensor<32x1152x7x7xf32>
    %v2594 = stablehlo.reduce(%v2593 init: %v2587) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2595 = stablehlo.broadcast_in_dim %v2594, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2596 = stablehlo.divide %v2595, %v2588 : tensor<32x1152x7x7xf32>
    %v2597 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2598 = stablehlo.add %v2596, %v2597 : tensor<32x1152x7x7xf32>
    %v2599 = stablehlo.rsqrt %v2598 : tensor<32x1152x7x7xf32>
    %v2600 = stablehlo.multiply %v2592, %v2599 : tensor<32x1152x7x7xf32>
    %v2601 = stablehlo.reshape %v2551 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2602 = stablehlo.multiply %v2601, %v2600 : tensor<32x1152x7x7xf32>
    %v2603 = stablehlo.reduce(%v2602 init: %v2587) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2604 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2605 = stablehlo.multiply %v2603, %v2604 : tensor<1152xf32>
    %v2606 = stablehlo.subtract %b14dg, %v2605 : tensor<1152xf32>
    %v2607 = stablehlo.reshape %v2551 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2608 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2609 = stablehlo.reduce(%v2607 init: %v2608) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2610 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2611 = stablehlo.multiply %v2609, %v2610 : tensor<1152xf32>
    %v2612 = stablehlo.subtract %b14dbt, %v2611 : tensor<1152xf32>
    %v2613 = stablehlo.reshape %v1356 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2614 = stablehlo.reshape %v2581 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2615 = stablehlo.transpose %v2613, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2616 = stablehlo.transpose %v2614, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2617 = stablehlo.convolution(%v2615, %v2616)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1152 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<1x1152x5x5xf32>
    %v2618 = stablehlo.reshape %v2617 : (tensor<1x1152x5x5xf32>) -> tensor<1152x1x5x5xf32>
    %v2619 = stablehlo.constant dense<0.05> : tensor<1152x1x5x5xf32>
    %v2620 = stablehlo.multiply %v2618, %v2619 : tensor<1152x1x5x5xf32>
    %v2621 = stablehlo.subtract %b14dW, %v2620 : tensor<1152x1x5x5xf32>
    %v2622 = stablehlo.reshape %v2585 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2623 = stablehlo.reshape %v1352 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2624 = stablehlo.logistic %v2623 : tensor<32x1152x7x7xf32>
    %v2625 = stablehlo.constant dense<1.0> : tensor<32x1152x7x7xf32>
    %v2626 = stablehlo.subtract %v2625, %v2624 : tensor<32x1152x7x7xf32>
    %v2627 = stablehlo.multiply %v2623, %v2626 : tensor<32x1152x7x7xf32>
    %v2628 = stablehlo.add %v2625, %v2627 : tensor<32x1152x7x7xf32>
    %v2629 = stablehlo.multiply %v2624, %v2628 : tensor<32x1152x7x7xf32>
    %v2630 = stablehlo.multiply %v2622, %v2629 : tensor<32x1152x7x7xf32>
    %v2631 = stablehlo.reshape %v2630 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2632 = stablehlo.reshape %v1332 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2633 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2634 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2635 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2636 = stablehlo.reduce(%v2632 init: %v2633) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2637 = stablehlo.broadcast_in_dim %v2636, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2638 = stablehlo.divide %v2637, %v2634 : tensor<32x1152x7x7xf32>
    %v2639 = stablehlo.subtract %v2632, %v2638 : tensor<32x1152x7x7xf32>
    %v2640 = stablehlo.multiply %v2639, %v2639 : tensor<32x1152x7x7xf32>
    %v2641 = stablehlo.reduce(%v2640 init: %v2633) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2642 = stablehlo.broadcast_in_dim %v2641, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2643 = stablehlo.divide %v2642, %v2634 : tensor<32x1152x7x7xf32>
    %v2644 = stablehlo.add %v2643, %v2635 : tensor<32x1152x7x7xf32>
    %v2645 = stablehlo.rsqrt %v2644 : tensor<32x1152x7x7xf32>
    %v2646 = stablehlo.multiply %v2639, %v2645 : tensor<32x1152x7x7xf32>
    %v2647 = stablehlo.broadcast_in_dim %b14eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2648 = stablehlo.reshape %v2631 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2649 = stablehlo.multiply %v2647, %v2648 : tensor<32x1152x7x7xf32>
    %v2650 = stablehlo.reduce(%v2649 init: %v2633) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2651 = stablehlo.broadcast_in_dim %v2650, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2652 = stablehlo.multiply %v2646, %v2649 : tensor<32x1152x7x7xf32>
    %v2653 = stablehlo.reduce(%v2652 init: %v2633) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2654 = stablehlo.broadcast_in_dim %v2653, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2655 = stablehlo.multiply %v2649, %v2634 : tensor<32x1152x7x7xf32>
    %v2656 = stablehlo.subtract %v2655, %v2651 : tensor<32x1152x7x7xf32>
    %v2657 = stablehlo.multiply %v2646, %v2654 : tensor<32x1152x7x7xf32>
    %v2658 = stablehlo.subtract %v2656, %v2657 : tensor<32x1152x7x7xf32>
    %v2659 = stablehlo.divide %v2645, %v2634 : tensor<32x1152x7x7xf32>
    %v2660 = stablehlo.multiply %v2659, %v2658 : tensor<32x1152x7x7xf32>
    %v2661 = stablehlo.reshape %v2660 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2662 = stablehlo.reshape %v2661 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2663 = stablehlo.reverse %b14eW, dims = [2, 3] : tensor<1152x192x1x1xf32>
    %v2664 = stablehlo.transpose %v2663, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %v2665 = stablehlo.convolution(%v2662, %v2664)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v2666 = stablehlo.reshape %v2665 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v2667 = stablehlo.reshape %v1332 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2668 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2669 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2670 = stablehlo.reduce(%v2667 init: %v2668) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2671 = stablehlo.broadcast_in_dim %v2670, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2672 = stablehlo.divide %v2671, %v2669 : tensor<32x1152x7x7xf32>
    %v2673 = stablehlo.subtract %v2667, %v2672 : tensor<32x1152x7x7xf32>
    %v2674 = stablehlo.multiply %v2673, %v2673 : tensor<32x1152x7x7xf32>
    %v2675 = stablehlo.reduce(%v2674 init: %v2668) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2676 = stablehlo.broadcast_in_dim %v2675, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2677 = stablehlo.divide %v2676, %v2669 : tensor<32x1152x7x7xf32>
    %v2678 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2679 = stablehlo.add %v2677, %v2678 : tensor<32x1152x7x7xf32>
    %v2680 = stablehlo.rsqrt %v2679 : tensor<32x1152x7x7xf32>
    %v2681 = stablehlo.multiply %v2673, %v2680 : tensor<32x1152x7x7xf32>
    %v2682 = stablehlo.reshape %v2631 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2683 = stablehlo.multiply %v2682, %v2681 : tensor<32x1152x7x7xf32>
    %v2684 = stablehlo.reduce(%v2683 init: %v2668) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2685 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2686 = stablehlo.multiply %v2684, %v2685 : tensor<1152xf32>
    %v2687 = stablehlo.subtract %b14eg, %v2686 : tensor<1152xf32>
    %v2688 = stablehlo.reshape %v2631 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2689 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2690 = stablehlo.reduce(%v2688 init: %v2689) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2691 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2692 = stablehlo.multiply %v2690, %v2691 : tensor<1152xf32>
    %v2693 = stablehlo.subtract %b14ebt, %v2692 : tensor<1152xf32>
    %v2694 = stablehlo.reshape %v1327 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2695 = stablehlo.reshape %v2661 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2696 = stablehlo.transpose %v2694, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %v2697 = stablehlo.transpose %v2695, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2698 = stablehlo.convolution(%v2696, %v2697)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<192x1152x1x1xf32>
    %v2699 = stablehlo.transpose %v2698, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %v2700 = stablehlo.constant dense<0.05> : tensor<1152x192x1x1xf32>
    %v2701 = stablehlo.multiply %v2699, %v2700 : tensor<1152x192x1x1xf32>
    %v2702 = stablehlo.subtract %b14eW, %v2701 : tensor<1152x192x1x1xf32>
    %v2703 = stablehlo.reshape %v2666 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2704 = stablehlo.reshape %v2395 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2705 = stablehlo.add %v2703, %v2704 : tensor<32x192x7x7xf32>
    %v2706 = stablehlo.reshape %v2705 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v2707 = stablehlo.reshape %v1303 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2708 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2709 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v2710 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v2711 = stablehlo.reduce(%v2707 init: %v2708) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2712 = stablehlo.broadcast_in_dim %v2711, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2713 = stablehlo.divide %v2712, %v2709 : tensor<32x192x7x7xf32>
    %v2714 = stablehlo.subtract %v2707, %v2713 : tensor<32x192x7x7xf32>
    %v2715 = stablehlo.multiply %v2714, %v2714 : tensor<32x192x7x7xf32>
    %v2716 = stablehlo.reduce(%v2715 init: %v2708) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2717 = stablehlo.broadcast_in_dim %v2716, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2718 = stablehlo.divide %v2717, %v2709 : tensor<32x192x7x7xf32>
    %v2719 = stablehlo.add %v2718, %v2710 : tensor<32x192x7x7xf32>
    %v2720 = stablehlo.rsqrt %v2719 : tensor<32x192x7x7xf32>
    %v2721 = stablehlo.multiply %v2714, %v2720 : tensor<32x192x7x7xf32>
    %v2722 = stablehlo.broadcast_in_dim %b13pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2723 = stablehlo.reshape %v2706 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2724 = stablehlo.multiply %v2722, %v2723 : tensor<32x192x7x7xf32>
    %v2725 = stablehlo.reduce(%v2724 init: %v2708) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2726 = stablehlo.broadcast_in_dim %v2725, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2727 = stablehlo.multiply %v2721, %v2724 : tensor<32x192x7x7xf32>
    %v2728 = stablehlo.reduce(%v2727 init: %v2708) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2729 = stablehlo.broadcast_in_dim %v2728, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2730 = stablehlo.multiply %v2724, %v2709 : tensor<32x192x7x7xf32>
    %v2731 = stablehlo.subtract %v2730, %v2726 : tensor<32x192x7x7xf32>
    %v2732 = stablehlo.multiply %v2721, %v2729 : tensor<32x192x7x7xf32>
    %v2733 = stablehlo.subtract %v2731, %v2732 : tensor<32x192x7x7xf32>
    %v2734 = stablehlo.divide %v2720, %v2709 : tensor<32x192x7x7xf32>
    %v2735 = stablehlo.multiply %v2734, %v2733 : tensor<32x192x7x7xf32>
    %v2736 = stablehlo.reshape %v2735 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v2737 = stablehlo.reshape %v2736 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2738 = stablehlo.reverse %b13pW, dims = [2, 3] : tensor<192x1152x1x1xf32>
    %v2739 = stablehlo.transpose %v2738, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %v2740 = stablehlo.convolution(%v2737, %v2739)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v2741 = stablehlo.reshape %v2740 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2742 = stablehlo.reshape %v1303 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2743 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2744 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v2745 = stablehlo.reduce(%v2742 init: %v2743) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2746 = stablehlo.broadcast_in_dim %v2745, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2747 = stablehlo.divide %v2746, %v2744 : tensor<32x192x7x7xf32>
    %v2748 = stablehlo.subtract %v2742, %v2747 : tensor<32x192x7x7xf32>
    %v2749 = stablehlo.multiply %v2748, %v2748 : tensor<32x192x7x7xf32>
    %v2750 = stablehlo.reduce(%v2749 init: %v2743) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2751 = stablehlo.broadcast_in_dim %v2750, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2752 = stablehlo.divide %v2751, %v2744 : tensor<32x192x7x7xf32>
    %v2753 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v2754 = stablehlo.add %v2752, %v2753 : tensor<32x192x7x7xf32>
    %v2755 = stablehlo.rsqrt %v2754 : tensor<32x192x7x7xf32>
    %v2756 = stablehlo.multiply %v2748, %v2755 : tensor<32x192x7x7xf32>
    %v2757 = stablehlo.reshape %v2706 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2758 = stablehlo.multiply %v2757, %v2756 : tensor<32x192x7x7xf32>
    %v2759 = stablehlo.reduce(%v2758 init: %v2743) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2760 = stablehlo.constant dense<0.05> : tensor<192xf32>
    %v2761 = stablehlo.multiply %v2759, %v2760 : tensor<192xf32>
    %v2762 = stablehlo.subtract %b13pg, %v2761 : tensor<192xf32>
    %v2763 = stablehlo.reshape %v2706 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2764 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2765 = stablehlo.reduce(%v2763 init: %v2764) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2766 = stablehlo.constant dense<0.05> : tensor<192xf32>
    %v2767 = stablehlo.multiply %v2765, %v2766 : tensor<192xf32>
    %v2768 = stablehlo.subtract %b13pbt, %v2767 : tensor<192xf32>
    %v2769 = stablehlo.reshape %v1298 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2770 = stablehlo.reshape %v2736 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2771 = stablehlo.transpose %v2769, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2772 = stablehlo.transpose %v2770, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %v2773 = stablehlo.convolution(%v2771, %v2772)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<192x32x7x7xf32>) -> tensor<1152x192x1x1xf32>
    %v2774 = stablehlo.transpose %v2773, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %v2775 = stablehlo.constant dense<0.05> : tensor<192x1152x1x1xf32>
    %v2776 = stablehlo.multiply %v2774, %v2775 : tensor<192x1152x1x1xf32>
    %v2777 = stablehlo.subtract %b13pW, %v2776 : tensor<192x1152x1x1xf32>
    %v2778 = stablehlo.reshape %v1281 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2779 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2780 = stablehlo.reduce(%v2778 init: %v2779) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2781 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v2782 = stablehlo.divide %v2780, %v2781 : tensor<32x1152xf32>
    %v2783 = stablehlo.dot_general %v2782, %b13zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v2784 = stablehlo.broadcast_in_dim %b13zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v2785 = stablehlo.add %v2783, %v2784 : tensor<32x48xf32>
    %v2786 = stablehlo.logistic %v2785 : tensor<32x48xf32>
    %v2787 = stablehlo.multiply %v2785, %v2786 : tensor<32x48xf32>
    %v2788 = stablehlo.dot_general %v2787, %b13zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v2789 = stablehlo.broadcast_in_dim %b13zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v2790 = stablehlo.add %v2788, %v2789 : tensor<32x1152xf32>
    %v2791 = stablehlo.logistic %v2790 : tensor<32x1152xf32>
    %v2792 = stablehlo.reshape %v2741 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2793 = stablehlo.broadcast_in_dim %v2791, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2794 = stablehlo.multiply %v2793, %v2792 : tensor<32x1152x7x7xf32>
    %v2795 = stablehlo.multiply %v2778, %v2792 : tensor<32x1152x7x7xf32>
    %v2796 = stablehlo.reduce(%v2795 init: %v2779) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2797 = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %v2798 = stablehlo.subtract %v2797, %v2791 : tensor<32x1152xf32>
    %v2799 = stablehlo.multiply %v2791, %v2798 : tensor<32x1152xf32>
    %v2800 = stablehlo.multiply %v2796, %v2799 : tensor<32x1152xf32>
    %v2801 = stablehlo.dot_general %v2800, %b13zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<48x1152xf32>) -> tensor<32x48xf32>
    %v2802 = stablehlo.logistic %v2785 : tensor<32x48xf32>
    %v2803 = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %v2804 = stablehlo.subtract %v2803, %v2802 : tensor<32x48xf32>
    %v2805 = stablehlo.multiply %v2785, %v2804 : tensor<32x48xf32>
    %v2806 = stablehlo.add %v2803, %v2805 : tensor<32x48xf32>
    %v2807 = stablehlo.multiply %v2802, %v2806 : tensor<32x48xf32>
    %v2808 = stablehlo.multiply %v2801, %v2807 : tensor<32x48xf32>
    %v2809 = stablehlo.dot_general %v2808, %b13zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<1152x48xf32>) -> tensor<32x1152xf32>
    %v2810 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v2811 = stablehlo.divide %v2809, %v2810 : tensor<32x1152xf32>
    %v2812 = stablehlo.broadcast_in_dim %v2811, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2813 = stablehlo.add %v2794, %v2812 : tensor<32x1152x7x7xf32>
    %v2814 = stablehlo.reshape %v2813 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2815 = stablehlo.reshape %v1281 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2816 = stablehlo.reshape %v2741 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2817 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2818 = stablehlo.multiply %v2815, %v2816 : tensor<32x1152x7x7xf32>
    %v2819 = stablehlo.reduce(%v2818 init: %v2817) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2820 = stablehlo.logistic %v1294 : tensor<32x1152xf32>
    %v2821 = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %v2822 = stablehlo.subtract %v2821, %v2820 : tensor<32x1152xf32>
    %v2823 = stablehlo.multiply %v2820, %v2822 : tensor<32x1152xf32>
    %v2824 = stablehlo.multiply %v2819, %v2823 : tensor<32x1152xf32>
    %v2825 = stablehlo.dot_general %v1291, %v2824, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<32x1152xf32>) -> tensor<48x1152xf32>
    %v2826 = stablehlo.constant dense<0.05> : tensor<48x1152xf32>
    %v2827 = stablehlo.multiply %v2825, %v2826 : tensor<48x1152xf32>
    %v2828 = stablehlo.subtract %b13zW2, %v2827 : tensor<48x1152xf32>
    %v2829 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2830 = stablehlo.reduce(%v2824 init: %v2829) applies stablehlo.add across dimensions = [0] : (tensor<32x1152xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2831 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2832 = stablehlo.multiply %v2830, %v2831 : tensor<1152xf32>
    %v2833 = stablehlo.subtract %b13zb2, %v2832 : tensor<1152xf32>
    %v2834 = stablehlo.reshape %v2824 : (tensor<32x1152xf32>) -> tensor<32x1x1152xf32>
    %v2835 = stablehlo.dot_general %v2834, %b13zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x1152xf32>, tensor<48x1152xf32>) -> tensor<32x1x48xf32>
    %v2836 = stablehlo.reshape %v2835 : (tensor<32x1x48xf32>) -> tensor<32x48xf32>
    %v2837 = stablehlo.logistic %v1289 : tensor<32x48xf32>
    %v2838 = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %v2839 = stablehlo.subtract %v2838, %v2837 : tensor<32x48xf32>
    %v2840 = stablehlo.multiply %v1289, %v2839 : tensor<32x48xf32>
    %v2841 = stablehlo.add %v2838, %v2840 : tensor<32x48xf32>
    %v2842 = stablehlo.multiply %v2837, %v2841 : tensor<32x48xf32>
    %v2843 = stablehlo.multiply %v2836, %v2842 : tensor<32x48xf32>
    %v2844 = stablehlo.dot_general %v1286, %v2843, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<32x48xf32>) -> tensor<1152x48xf32>
    %v2845 = stablehlo.constant dense<0.05> : tensor<1152x48xf32>
    %v2846 = stablehlo.multiply %v2844, %v2845 : tensor<1152x48xf32>
    %v2847 = stablehlo.subtract %b13zW1, %v2846 : tensor<1152x48xf32>
    %v2848 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2849 = stablehlo.reduce(%v2843 init: %v2848) applies stablehlo.add across dimensions = [0] : (tensor<32x48xf32>, tensor<f32>) -> tensor<48xf32>
    %v2850 = stablehlo.constant dense<0.05> : tensor<48xf32>
    %v2851 = stablehlo.multiply %v2849, %v2850 : tensor<48xf32>
    %v2852 = stablehlo.subtract %b13zb1, %v2851 : tensor<48xf32>
    %v2853 = stablehlo.reshape %v2814 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2854 = stablehlo.reshape %v1277 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2855 = stablehlo.logistic %v2854 : tensor<32x1152x7x7xf32>
    %v2856 = stablehlo.constant dense<1.0> : tensor<32x1152x7x7xf32>
    %v2857 = stablehlo.subtract %v2856, %v2855 : tensor<32x1152x7x7xf32>
    %v2858 = stablehlo.multiply %v2854, %v2857 : tensor<32x1152x7x7xf32>
    %v2859 = stablehlo.add %v2856, %v2858 : tensor<32x1152x7x7xf32>
    %v2860 = stablehlo.multiply %v2855, %v2859 : tensor<32x1152x7x7xf32>
    %v2861 = stablehlo.multiply %v2853, %v2860 : tensor<32x1152x7x7xf32>
    %v2862 = stablehlo.reshape %v2861 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2863 = stablehlo.reshape %v1257 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2864 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2865 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2866 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2867 = stablehlo.reduce(%v2863 init: %v2864) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2868 = stablehlo.broadcast_in_dim %v2867, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2869 = stablehlo.divide %v2868, %v2865 : tensor<32x1152x7x7xf32>
    %v2870 = stablehlo.subtract %v2863, %v2869 : tensor<32x1152x7x7xf32>
    %v2871 = stablehlo.multiply %v2870, %v2870 : tensor<32x1152x7x7xf32>
    %v2872 = stablehlo.reduce(%v2871 init: %v2864) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2873 = stablehlo.broadcast_in_dim %v2872, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2874 = stablehlo.divide %v2873, %v2865 : tensor<32x1152x7x7xf32>
    %v2875 = stablehlo.add %v2874, %v2866 : tensor<32x1152x7x7xf32>
    %v2876 = stablehlo.rsqrt %v2875 : tensor<32x1152x7x7xf32>
    %v2877 = stablehlo.multiply %v2870, %v2876 : tensor<32x1152x7x7xf32>
    %v2878 = stablehlo.broadcast_in_dim %b13dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2879 = stablehlo.reshape %v2862 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2880 = stablehlo.multiply %v2878, %v2879 : tensor<32x1152x7x7xf32>
    %v2881 = stablehlo.reduce(%v2880 init: %v2864) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2882 = stablehlo.broadcast_in_dim %v2881, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2883 = stablehlo.multiply %v2877, %v2880 : tensor<32x1152x7x7xf32>
    %v2884 = stablehlo.reduce(%v2883 init: %v2864) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2885 = stablehlo.broadcast_in_dim %v2884, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2886 = stablehlo.multiply %v2880, %v2865 : tensor<32x1152x7x7xf32>
    %v2887 = stablehlo.subtract %v2886, %v2882 : tensor<32x1152x7x7xf32>
    %v2888 = stablehlo.multiply %v2877, %v2885 : tensor<32x1152x7x7xf32>
    %v2889 = stablehlo.subtract %v2887, %v2888 : tensor<32x1152x7x7xf32>
    %v2890 = stablehlo.divide %v2876, %v2865 : tensor<32x1152x7x7xf32>
    %v2891 = stablehlo.multiply %v2890, %v2889 : tensor<32x1152x7x7xf32>
    %v2892 = stablehlo.reshape %v2891 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2893 = stablehlo.reshape %v2892 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2894 = stablehlo.reverse %b13dW, dims = [2, 3] : tensor<1152x1x5x5xf32>
    %v2895 = stablehlo.convolution(%v2893, %v2894)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v2896 = stablehlo.reshape %v2895 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2897 = stablehlo.reshape %v1257 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2898 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2899 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2900 = stablehlo.reduce(%v2897 init: %v2898) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2901 = stablehlo.broadcast_in_dim %v2900, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2902 = stablehlo.divide %v2901, %v2899 : tensor<32x1152x7x7xf32>
    %v2903 = stablehlo.subtract %v2897, %v2902 : tensor<32x1152x7x7xf32>
    %v2904 = stablehlo.multiply %v2903, %v2903 : tensor<32x1152x7x7xf32>
    %v2905 = stablehlo.reduce(%v2904 init: %v2898) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2906 = stablehlo.broadcast_in_dim %v2905, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2907 = stablehlo.divide %v2906, %v2899 : tensor<32x1152x7x7xf32>
    %v2908 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2909 = stablehlo.add %v2907, %v2908 : tensor<32x1152x7x7xf32>
    %v2910 = stablehlo.rsqrt %v2909 : tensor<32x1152x7x7xf32>
    %v2911 = stablehlo.multiply %v2903, %v2910 : tensor<32x1152x7x7xf32>
    %v2912 = stablehlo.reshape %v2862 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2913 = stablehlo.multiply %v2912, %v2911 : tensor<32x1152x7x7xf32>
    %v2914 = stablehlo.reduce(%v2913 init: %v2898) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2915 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2916 = stablehlo.multiply %v2914, %v2915 : tensor<1152xf32>
    %v2917 = stablehlo.subtract %b13dg, %v2916 : tensor<1152xf32>
    %v2918 = stablehlo.reshape %v2862 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2919 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2920 = stablehlo.reduce(%v2918 init: %v2919) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2921 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2922 = stablehlo.multiply %v2920, %v2921 : tensor<1152xf32>
    %v2923 = stablehlo.subtract %b13dbt, %v2922 : tensor<1152xf32>
    %v2924 = stablehlo.reshape %v1252 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2925 = stablehlo.reshape %v2892 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2926 = stablehlo.transpose %v2924, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2927 = stablehlo.transpose %v2925, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2928 = stablehlo.convolution(%v2926, %v2927)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1152 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<1x1152x5x5xf32>
    %v2929 = stablehlo.reshape %v2928 : (tensor<1x1152x5x5xf32>) -> tensor<1152x1x5x5xf32>
    %v2930 = stablehlo.constant dense<0.05> : tensor<1152x1x5x5xf32>
    %v2931 = stablehlo.multiply %v2929, %v2930 : tensor<1152x1x5x5xf32>
    %v2932 = stablehlo.subtract %b13dW, %v2931 : tensor<1152x1x5x5xf32>
    %v2933 = stablehlo.reshape %v2896 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2934 = stablehlo.reshape %v1248 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2935 = stablehlo.logistic %v2934 : tensor<32x1152x7x7xf32>
    %v2936 = stablehlo.constant dense<1.0> : tensor<32x1152x7x7xf32>
    %v2937 = stablehlo.subtract %v2936, %v2935 : tensor<32x1152x7x7xf32>
    %v2938 = stablehlo.multiply %v2934, %v2937 : tensor<32x1152x7x7xf32>
    %v2939 = stablehlo.add %v2936, %v2938 : tensor<32x1152x7x7xf32>
    %v2940 = stablehlo.multiply %v2935, %v2939 : tensor<32x1152x7x7xf32>
    %v2941 = stablehlo.multiply %v2933, %v2940 : tensor<32x1152x7x7xf32>
    %v2942 = stablehlo.reshape %v2941 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2943 = stablehlo.reshape %v1228 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2944 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2945 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2946 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2947 = stablehlo.reduce(%v2943 init: %v2944) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2948 = stablehlo.broadcast_in_dim %v2947, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2949 = stablehlo.divide %v2948, %v2945 : tensor<32x1152x7x7xf32>
    %v2950 = stablehlo.subtract %v2943, %v2949 : tensor<32x1152x7x7xf32>
    %v2951 = stablehlo.multiply %v2950, %v2950 : tensor<32x1152x7x7xf32>
    %v2952 = stablehlo.reduce(%v2951 init: %v2944) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2953 = stablehlo.broadcast_in_dim %v2952, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2954 = stablehlo.divide %v2953, %v2945 : tensor<32x1152x7x7xf32>
    %v2955 = stablehlo.add %v2954, %v2946 : tensor<32x1152x7x7xf32>
    %v2956 = stablehlo.rsqrt %v2955 : tensor<32x1152x7x7xf32>
    %v2957 = stablehlo.multiply %v2950, %v2956 : tensor<32x1152x7x7xf32>
    %v2958 = stablehlo.broadcast_in_dim %b13eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2959 = stablehlo.reshape %v2942 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2960 = stablehlo.multiply %v2958, %v2959 : tensor<32x1152x7x7xf32>
    %v2961 = stablehlo.reduce(%v2960 init: %v2944) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2962 = stablehlo.broadcast_in_dim %v2961, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2963 = stablehlo.multiply %v2957, %v2960 : tensor<32x1152x7x7xf32>
    %v2964 = stablehlo.reduce(%v2963 init: %v2944) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2965 = stablehlo.broadcast_in_dim %v2964, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2966 = stablehlo.multiply %v2960, %v2945 : tensor<32x1152x7x7xf32>
    %v2967 = stablehlo.subtract %v2966, %v2962 : tensor<32x1152x7x7xf32>
    %v2968 = stablehlo.multiply %v2957, %v2965 : tensor<32x1152x7x7xf32>
    %v2969 = stablehlo.subtract %v2967, %v2968 : tensor<32x1152x7x7xf32>
    %v2970 = stablehlo.divide %v2956, %v2945 : tensor<32x1152x7x7xf32>
    %v2971 = stablehlo.multiply %v2970, %v2969 : tensor<32x1152x7x7xf32>
    %v2972 = stablehlo.reshape %v2971 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2973 = stablehlo.reshape %v2972 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2974 = stablehlo.reverse %b13eW, dims = [2, 3] : tensor<1152x192x1x1xf32>
    %v2975 = stablehlo.transpose %v2974, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %v2976 = stablehlo.convolution(%v2973, %v2975)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v2977 = stablehlo.reshape %v2976 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v2978 = stablehlo.reshape %v1228 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2979 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2980 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2981 = stablehlo.reduce(%v2978 init: %v2979) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2982 = stablehlo.broadcast_in_dim %v2981, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2983 = stablehlo.divide %v2982, %v2980 : tensor<32x1152x7x7xf32>
    %v2984 = stablehlo.subtract %v2978, %v2983 : tensor<32x1152x7x7xf32>
    %v2985 = stablehlo.multiply %v2984, %v2984 : tensor<32x1152x7x7xf32>
    %v2986 = stablehlo.reduce(%v2985 init: %v2979) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2987 = stablehlo.broadcast_in_dim %v2986, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2988 = stablehlo.divide %v2987, %v2980 : tensor<32x1152x7x7xf32>
    %v2989 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2990 = stablehlo.add %v2988, %v2989 : tensor<32x1152x7x7xf32>
    %v2991 = stablehlo.rsqrt %v2990 : tensor<32x1152x7x7xf32>
    %v2992 = stablehlo.multiply %v2984, %v2991 : tensor<32x1152x7x7xf32>
    %v2993 = stablehlo.reshape %v2942 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2994 = stablehlo.multiply %v2993, %v2992 : tensor<32x1152x7x7xf32>
    %v2995 = stablehlo.reduce(%v2994 init: %v2979) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2996 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2997 = stablehlo.multiply %v2995, %v2996 : tensor<1152xf32>
    %v2998 = stablehlo.subtract %b13eg, %v2997 : tensor<1152xf32>
    %v2999 = stablehlo.reshape %v2942 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3000 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3001 = stablehlo.reduce(%v2999 init: %v3000) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3002 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v3003 = stablehlo.multiply %v3001, %v3002 : tensor<1152xf32>
    %v3004 = stablehlo.subtract %b13ebt, %v3003 : tensor<1152xf32>
    %v3005 = stablehlo.reshape %v1223 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v3006 = stablehlo.reshape %v2972 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3007 = stablehlo.transpose %v3005, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %v3008 = stablehlo.transpose %v3006, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v3009 = stablehlo.convolution(%v3007, %v3008)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<192x1152x1x1xf32>
    %v3010 = stablehlo.transpose %v3009, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %v3011 = stablehlo.constant dense<0.05> : tensor<1152x192x1x1xf32>
    %v3012 = stablehlo.multiply %v3010, %v3011 : tensor<1152x192x1x1xf32>
    %v3013 = stablehlo.subtract %b13eW, %v3012 : tensor<1152x192x1x1xf32>
    %v3014 = stablehlo.reshape %v2977 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v3015 = stablehlo.reshape %v2706 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v3016 = stablehlo.add %v3014, %v3015 : tensor<32x192x7x7xf32>
    %v3017 = stablehlo.reshape %v3016 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v3018 = stablehlo.reshape %v1203 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v3019 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3020 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v3021 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v3022 = stablehlo.reduce(%v3018 init: %v3019) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v3023 = stablehlo.broadcast_in_dim %v3022, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v3024 = stablehlo.divide %v3023, %v3020 : tensor<32x192x7x7xf32>
    %v3025 = stablehlo.subtract %v3018, %v3024 : tensor<32x192x7x7xf32>
    %v3026 = stablehlo.multiply %v3025, %v3025 : tensor<32x192x7x7xf32>
    %v3027 = stablehlo.reduce(%v3026 init: %v3019) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v3028 = stablehlo.broadcast_in_dim %v3027, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v3029 = stablehlo.divide %v3028, %v3020 : tensor<32x192x7x7xf32>
    %v3030 = stablehlo.add %v3029, %v3021 : tensor<32x192x7x7xf32>
    %v3031 = stablehlo.rsqrt %v3030 : tensor<32x192x7x7xf32>
    %v3032 = stablehlo.multiply %v3025, %v3031 : tensor<32x192x7x7xf32>
    %v3033 = stablehlo.broadcast_in_dim %b12pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v3034 = stablehlo.reshape %v3017 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v3035 = stablehlo.multiply %v3033, %v3034 : tensor<32x192x7x7xf32>
    %v3036 = stablehlo.reduce(%v3035 init: %v3019) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v3037 = stablehlo.broadcast_in_dim %v3036, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v3038 = stablehlo.multiply %v3032, %v3035 : tensor<32x192x7x7xf32>
    %v3039 = stablehlo.reduce(%v3038 init: %v3019) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v3040 = stablehlo.broadcast_in_dim %v3039, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v3041 = stablehlo.multiply %v3035, %v3020 : tensor<32x192x7x7xf32>
    %v3042 = stablehlo.subtract %v3041, %v3037 : tensor<32x192x7x7xf32>
    %v3043 = stablehlo.multiply %v3032, %v3040 : tensor<32x192x7x7xf32>
    %v3044 = stablehlo.subtract %v3042, %v3043 : tensor<32x192x7x7xf32>
    %v3045 = stablehlo.divide %v3031, %v3020 : tensor<32x192x7x7xf32>
    %v3046 = stablehlo.multiply %v3045, %v3044 : tensor<32x192x7x7xf32>
    %v3047 = stablehlo.reshape %v3046 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v3048 = stablehlo.reshape %v3047 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v3049 = stablehlo.reverse %b12pW, dims = [2, 3] : tensor<192x672x1x1xf32>
    %v3050 = stablehlo.transpose %v3049, dims = [1, 0, 2, 3] : (tensor<192x672x1x1xf32>) -> tensor<672x192x1x1xf32>
    %v3051 = stablehlo.convolution(%v3048, %v3050)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<672x192x1x1xf32>) -> tensor<32x672x7x7xf32>
    %v3052 = stablehlo.reshape %v3051 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v3053 = stablehlo.reshape %v1203 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v3054 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3055 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v3056 = stablehlo.reduce(%v3053 init: %v3054) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v3057 = stablehlo.broadcast_in_dim %v3056, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v3058 = stablehlo.divide %v3057, %v3055 : tensor<32x192x7x7xf32>
    %v3059 = stablehlo.subtract %v3053, %v3058 : tensor<32x192x7x7xf32>
    %v3060 = stablehlo.multiply %v3059, %v3059 : tensor<32x192x7x7xf32>
    %v3061 = stablehlo.reduce(%v3060 init: %v3054) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v3062 = stablehlo.broadcast_in_dim %v3061, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v3063 = stablehlo.divide %v3062, %v3055 : tensor<32x192x7x7xf32>
    %v3064 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v3065 = stablehlo.add %v3063, %v3064 : tensor<32x192x7x7xf32>
    %v3066 = stablehlo.rsqrt %v3065 : tensor<32x192x7x7xf32>
    %v3067 = stablehlo.multiply %v3059, %v3066 : tensor<32x192x7x7xf32>
    %v3068 = stablehlo.reshape %v3017 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v3069 = stablehlo.multiply %v3068, %v3067 : tensor<32x192x7x7xf32>
    %v3070 = stablehlo.reduce(%v3069 init: %v3054) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v3071 = stablehlo.constant dense<0.05> : tensor<192xf32>
    %v3072 = stablehlo.multiply %v3070, %v3071 : tensor<192xf32>
    %v3073 = stablehlo.subtract %b12pg, %v3072 : tensor<192xf32>
    %v3074 = stablehlo.reshape %v3017 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v3075 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3076 = stablehlo.reduce(%v3074 init: %v3075) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v3077 = stablehlo.constant dense<0.05> : tensor<192xf32>
    %v3078 = stablehlo.multiply %v3076, %v3077 : tensor<192xf32>
    %v3079 = stablehlo.subtract %b12pbt, %v3078 : tensor<192xf32>
    %v3080 = stablehlo.reshape %v1198 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3081 = stablehlo.reshape %v3047 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v3082 = stablehlo.transpose %v3080, dims = [1, 0, 2, 3] : (tensor<32x672x7x7xf32>) -> tensor<672x32x7x7xf32>
    %v3083 = stablehlo.transpose %v3081, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %v3084 = stablehlo.convolution(%v3082, %v3083)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<672x32x7x7xf32>, tensor<192x32x7x7xf32>) -> tensor<672x192x1x1xf32>
    %v3085 = stablehlo.transpose %v3084, dims = [1, 0, 2, 3] : (tensor<672x192x1x1xf32>) -> tensor<192x672x1x1xf32>
    %v3086 = stablehlo.constant dense<0.05> : tensor<192x672x1x1xf32>
    %v3087 = stablehlo.multiply %v3085, %v3086 : tensor<192x672x1x1xf32>
    %v3088 = stablehlo.subtract %b12pW, %v3087 : tensor<192x672x1x1xf32>
    %v3089 = stablehlo.reshape %v1181 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3090 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3091 = stablehlo.reduce(%v3089 init: %v3090) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3092 = stablehlo.constant dense<49.0> : tensor<32x672xf32>
    %v3093 = stablehlo.divide %v3091, %v3092 : tensor<32x672xf32>
    %v3094 = stablehlo.dot_general %v3093, %b12zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v3095 = stablehlo.broadcast_in_dim %b12zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v3096 = stablehlo.add %v3094, %v3095 : tensor<32x28xf32>
    %v3097 = stablehlo.logistic %v3096 : tensor<32x28xf32>
    %v3098 = stablehlo.multiply %v3096, %v3097 : tensor<32x28xf32>
    %v3099 = stablehlo.dot_general %v3098, %b12zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v3100 = stablehlo.broadcast_in_dim %b12zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v3101 = stablehlo.add %v3099, %v3100 : tensor<32x672xf32>
    %v3102 = stablehlo.logistic %v3101 : tensor<32x672xf32>
    %v3103 = stablehlo.reshape %v3052 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3104 = stablehlo.broadcast_in_dim %v3102, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x7x7xf32>
    %v3105 = stablehlo.multiply %v3104, %v3103 : tensor<32x672x7x7xf32>
    %v3106 = stablehlo.multiply %v3089, %v3103 : tensor<32x672x7x7xf32>
    %v3107 = stablehlo.reduce(%v3106 init: %v3090) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3108 = stablehlo.constant dense<1.0> : tensor<32x672xf32>
    %v3109 = stablehlo.subtract %v3108, %v3102 : tensor<32x672xf32>
    %v3110 = stablehlo.multiply %v3102, %v3109 : tensor<32x672xf32>
    %v3111 = stablehlo.multiply %v3107, %v3110 : tensor<32x672xf32>
    %v3112 = stablehlo.dot_general %v3111, %b12zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<28x672xf32>) -> tensor<32x28xf32>
    %v3113 = stablehlo.logistic %v3096 : tensor<32x28xf32>
    %v3114 = stablehlo.constant dense<1.0> : tensor<32x28xf32>
    %v3115 = stablehlo.subtract %v3114, %v3113 : tensor<32x28xf32>
    %v3116 = stablehlo.multiply %v3096, %v3115 : tensor<32x28xf32>
    %v3117 = stablehlo.add %v3114, %v3116 : tensor<32x28xf32>
    %v3118 = stablehlo.multiply %v3113, %v3117 : tensor<32x28xf32>
    %v3119 = stablehlo.multiply %v3112, %v3118 : tensor<32x28xf32>
    %v3120 = stablehlo.dot_general %v3119, %b12zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<672x28xf32>) -> tensor<32x672xf32>
    %v3121 = stablehlo.constant dense<49.0> : tensor<32x672xf32>
    %v3122 = stablehlo.divide %v3120, %v3121 : tensor<32x672xf32>
    %v3123 = stablehlo.broadcast_in_dim %v3122, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x7x7xf32>
    %v3124 = stablehlo.add %v3105, %v3123 : tensor<32x672x7x7xf32>
    %v3125 = stablehlo.reshape %v3124 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v3126 = stablehlo.reshape %v1181 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3127 = stablehlo.reshape %v3052 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3128 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3129 = stablehlo.multiply %v3126, %v3127 : tensor<32x672x7x7xf32>
    %v3130 = stablehlo.reduce(%v3129 init: %v3128) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3131 = stablehlo.logistic %v1194 : tensor<32x672xf32>
    %v3132 = stablehlo.constant dense<1.0> : tensor<32x672xf32>
    %v3133 = stablehlo.subtract %v3132, %v3131 : tensor<32x672xf32>
    %v3134 = stablehlo.multiply %v3131, %v3133 : tensor<32x672xf32>
    %v3135 = stablehlo.multiply %v3130, %v3134 : tensor<32x672xf32>
    %v3136 = stablehlo.dot_general %v1191, %v3135, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<32x672xf32>) -> tensor<28x672xf32>
    %v3137 = stablehlo.constant dense<0.05> : tensor<28x672xf32>
    %v3138 = stablehlo.multiply %v3136, %v3137 : tensor<28x672xf32>
    %v3139 = stablehlo.subtract %b12zW2, %v3138 : tensor<28x672xf32>
    %v3140 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3141 = stablehlo.reduce(%v3135 init: %v3140) applies stablehlo.add across dimensions = [0] : (tensor<32x672xf32>, tensor<f32>) -> tensor<672xf32>
    %v3142 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3143 = stablehlo.multiply %v3141, %v3142 : tensor<672xf32>
    %v3144 = stablehlo.subtract %b12zb2, %v3143 : tensor<672xf32>
    %v3145 = stablehlo.reshape %v3135 : (tensor<32x672xf32>) -> tensor<32x1x672xf32>
    %v3146 = stablehlo.dot_general %v3145, %b12zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x672xf32>, tensor<28x672xf32>) -> tensor<32x1x28xf32>
    %v3147 = stablehlo.reshape %v3146 : (tensor<32x1x28xf32>) -> tensor<32x28xf32>
    %v3148 = stablehlo.logistic %v1189 : tensor<32x28xf32>
    %v3149 = stablehlo.constant dense<1.0> : tensor<32x28xf32>
    %v3150 = stablehlo.subtract %v3149, %v3148 : tensor<32x28xf32>
    %v3151 = stablehlo.multiply %v1189, %v3150 : tensor<32x28xf32>
    %v3152 = stablehlo.add %v3149, %v3151 : tensor<32x28xf32>
    %v3153 = stablehlo.multiply %v3148, %v3152 : tensor<32x28xf32>
    %v3154 = stablehlo.multiply %v3147, %v3153 : tensor<32x28xf32>
    %v3155 = stablehlo.dot_general %v1186, %v3154, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<32x28xf32>) -> tensor<672x28xf32>
    %v3156 = stablehlo.constant dense<0.05> : tensor<672x28xf32>
    %v3157 = stablehlo.multiply %v3155, %v3156 : tensor<672x28xf32>
    %v3158 = stablehlo.subtract %b12zW1, %v3157 : tensor<672x28xf32>
    %v3159 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3160 = stablehlo.reduce(%v3154 init: %v3159) applies stablehlo.add across dimensions = [0] : (tensor<32x28xf32>, tensor<f32>) -> tensor<28xf32>
    %v3161 = stablehlo.constant dense<0.05> : tensor<28xf32>
    %v3162 = stablehlo.multiply %v3160, %v3161 : tensor<28xf32>
    %v3163 = stablehlo.subtract %b12zb1, %v3162 : tensor<28xf32>
    %v3164 = stablehlo.reshape %v3125 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3165 = stablehlo.reshape %v1177 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3166 = stablehlo.logistic %v3165 : tensor<32x672x7x7xf32>
    %v3167 = stablehlo.constant dense<1.0> : tensor<32x672x7x7xf32>
    %v3168 = stablehlo.subtract %v3167, %v3166 : tensor<32x672x7x7xf32>
    %v3169 = stablehlo.multiply %v3165, %v3168 : tensor<32x672x7x7xf32>
    %v3170 = stablehlo.add %v3167, %v3169 : tensor<32x672x7x7xf32>
    %v3171 = stablehlo.multiply %v3166, %v3170 : tensor<32x672x7x7xf32>
    %v3172 = stablehlo.multiply %v3164, %v3171 : tensor<32x672x7x7xf32>
    %v3173 = stablehlo.reshape %v3172 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v3174 = stablehlo.reshape %v1157 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3175 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3176 = stablehlo.constant dense<1568.0> : tensor<32x672x7x7xf32>
    %v3177 = stablehlo.constant dense<1.0e-5> : tensor<32x672x7x7xf32>
    %v3178 = stablehlo.reduce(%v3174 init: %v3175) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v3179 = stablehlo.broadcast_in_dim %v3178, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v3180 = stablehlo.divide %v3179, %v3176 : tensor<32x672x7x7xf32>
    %v3181 = stablehlo.subtract %v3174, %v3180 : tensor<32x672x7x7xf32>
    %v3182 = stablehlo.multiply %v3181, %v3181 : tensor<32x672x7x7xf32>
    %v3183 = stablehlo.reduce(%v3182 init: %v3175) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v3184 = stablehlo.broadcast_in_dim %v3183, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v3185 = stablehlo.divide %v3184, %v3176 : tensor<32x672x7x7xf32>
    %v3186 = stablehlo.add %v3185, %v3177 : tensor<32x672x7x7xf32>
    %v3187 = stablehlo.rsqrt %v3186 : tensor<32x672x7x7xf32>
    %v3188 = stablehlo.multiply %v3181, %v3187 : tensor<32x672x7x7xf32>
    %v3189 = stablehlo.broadcast_in_dim %b12dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v3190 = stablehlo.reshape %v3173 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3191 = stablehlo.multiply %v3189, %v3190 : tensor<32x672x7x7xf32>
    %v3192 = stablehlo.reduce(%v3191 init: %v3175) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v3193 = stablehlo.broadcast_in_dim %v3192, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v3194 = stablehlo.multiply %v3188, %v3191 : tensor<32x672x7x7xf32>
    %v3195 = stablehlo.reduce(%v3194 init: %v3175) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v3196 = stablehlo.broadcast_in_dim %v3195, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v3197 = stablehlo.multiply %v3191, %v3176 : tensor<32x672x7x7xf32>
    %v3198 = stablehlo.subtract %v3197, %v3193 : tensor<32x672x7x7xf32>
    %v3199 = stablehlo.multiply %v3188, %v3196 : tensor<32x672x7x7xf32>
    %v3200 = stablehlo.subtract %v3198, %v3199 : tensor<32x672x7x7xf32>
    %v3201 = stablehlo.divide %v3187, %v3176 : tensor<32x672x7x7xf32>
    %v3202 = stablehlo.multiply %v3201, %v3200 : tensor<32x672x7x7xf32>
    %v3203 = stablehlo.reshape %v3202 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v3204 = stablehlo.reshape %v3203 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3205 = stablehlo.reverse %b12dW, dims = [2, 3] : tensor<672x1x5x5xf32>
    %v3206 = stablehlo.convolution(%v3204, %v3205)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x7x7xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %v3207 = stablehlo.reshape %v3206 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3208 = stablehlo.reshape %v1157 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3209 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3210 = stablehlo.constant dense<1568.0> : tensor<32x672x7x7xf32>
    %v3211 = stablehlo.reduce(%v3208 init: %v3209) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v3212 = stablehlo.broadcast_in_dim %v3211, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v3213 = stablehlo.divide %v3212, %v3210 : tensor<32x672x7x7xf32>
    %v3214 = stablehlo.subtract %v3208, %v3213 : tensor<32x672x7x7xf32>
    %v3215 = stablehlo.multiply %v3214, %v3214 : tensor<32x672x7x7xf32>
    %v3216 = stablehlo.reduce(%v3215 init: %v3209) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v3217 = stablehlo.broadcast_in_dim %v3216, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v3218 = stablehlo.divide %v3217, %v3210 : tensor<32x672x7x7xf32>
    %v3219 = stablehlo.constant dense<1.0e-5> : tensor<32x672x7x7xf32>
    %v3220 = stablehlo.add %v3218, %v3219 : tensor<32x672x7x7xf32>
    %v3221 = stablehlo.rsqrt %v3220 : tensor<32x672x7x7xf32>
    %v3222 = stablehlo.multiply %v3214, %v3221 : tensor<32x672x7x7xf32>
    %v3223 = stablehlo.reshape %v3173 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3224 = stablehlo.multiply %v3223, %v3222 : tensor<32x672x7x7xf32>
    %v3225 = stablehlo.reduce(%v3224 init: %v3209) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v3226 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3227 = stablehlo.multiply %v3225, %v3226 : tensor<672xf32>
    %v3228 = stablehlo.subtract %b12dg, %v3227 : tensor<672xf32>
    %v3229 = stablehlo.reshape %v3173 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3230 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3231 = stablehlo.reduce(%v3229 init: %v3230) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v3232 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3233 = stablehlo.multiply %v3231, %v3232 : tensor<672xf32>
    %v3234 = stablehlo.subtract %b12dbt, %v3233 : tensor<672xf32>
    %v3235 = stablehlo.reshape %v1152 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3236 = stablehlo.reshape %v3203 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3237 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3238 = stablehlo.pad %v3236, %v3237, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672x14x14xf32>
    %v3239 = stablehlo.transpose %v3235, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3240 = stablehlo.transpose %v3238, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3241 = stablehlo.convolution(%v3239, %v3240)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 3], [1, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 672 : i64, feature_group_count = 1 : i64} : (tensor<672x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<1x672x5x5xf32>
    %v3242 = stablehlo.reshape %v3241 : (tensor<1x672x5x5xf32>) -> tensor<672x1x5x5xf32>
    %v3243 = stablehlo.constant dense<0.05> : tensor<672x1x5x5xf32>
    %v3244 = stablehlo.multiply %v3242, %v3243 : tensor<672x1x5x5xf32>
    %v3245 = stablehlo.subtract %b12dW, %v3244 : tensor<672x1x5x5xf32>
    %v3246 = stablehlo.reshape %v3207 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3247 = stablehlo.reshape %v1148 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3248 = stablehlo.logistic %v3247 : tensor<32x672x14x14xf32>
    %v3249 = stablehlo.constant dense<1.0> : tensor<32x672x14x14xf32>
    %v3250 = stablehlo.subtract %v3249, %v3248 : tensor<32x672x14x14xf32>
    %v3251 = stablehlo.multiply %v3247, %v3250 : tensor<32x672x14x14xf32>
    %v3252 = stablehlo.add %v3249, %v3251 : tensor<32x672x14x14xf32>
    %v3253 = stablehlo.multiply %v3248, %v3252 : tensor<32x672x14x14xf32>
    %v3254 = stablehlo.multiply %v3246, %v3253 : tensor<32x672x14x14xf32>
    %v3255 = stablehlo.reshape %v3254 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3256 = stablehlo.reshape %v1128 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3257 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3258 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3259 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3260 = stablehlo.reduce(%v3256 init: %v3257) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3261 = stablehlo.broadcast_in_dim %v3260, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3262 = stablehlo.divide %v3261, %v3258 : tensor<32x672x14x14xf32>
    %v3263 = stablehlo.subtract %v3256, %v3262 : tensor<32x672x14x14xf32>
    %v3264 = stablehlo.multiply %v3263, %v3263 : tensor<32x672x14x14xf32>
    %v3265 = stablehlo.reduce(%v3264 init: %v3257) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3266 = stablehlo.broadcast_in_dim %v3265, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3267 = stablehlo.divide %v3266, %v3258 : tensor<32x672x14x14xf32>
    %v3268 = stablehlo.add %v3267, %v3259 : tensor<32x672x14x14xf32>
    %v3269 = stablehlo.rsqrt %v3268 : tensor<32x672x14x14xf32>
    %v3270 = stablehlo.multiply %v3263, %v3269 : tensor<32x672x14x14xf32>
    %v3271 = stablehlo.broadcast_in_dim %b12eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3272 = stablehlo.reshape %v3255 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3273 = stablehlo.multiply %v3271, %v3272 : tensor<32x672x14x14xf32>
    %v3274 = stablehlo.reduce(%v3273 init: %v3257) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3275 = stablehlo.broadcast_in_dim %v3274, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3276 = stablehlo.multiply %v3270, %v3273 : tensor<32x672x14x14xf32>
    %v3277 = stablehlo.reduce(%v3276 init: %v3257) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3278 = stablehlo.broadcast_in_dim %v3277, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3279 = stablehlo.multiply %v3273, %v3258 : tensor<32x672x14x14xf32>
    %v3280 = stablehlo.subtract %v3279, %v3275 : tensor<32x672x14x14xf32>
    %v3281 = stablehlo.multiply %v3270, %v3278 : tensor<32x672x14x14xf32>
    %v3282 = stablehlo.subtract %v3280, %v3281 : tensor<32x672x14x14xf32>
    %v3283 = stablehlo.divide %v3269, %v3258 : tensor<32x672x14x14xf32>
    %v3284 = stablehlo.multiply %v3283, %v3282 : tensor<32x672x14x14xf32>
    %v3285 = stablehlo.reshape %v3284 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3286 = stablehlo.reshape %v3285 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3287 = stablehlo.reverse %b12eW, dims = [2, 3] : tensor<672x112x1x1xf32>
    %v3288 = stablehlo.transpose %v3287, dims = [1, 0, 2, 3] : (tensor<672x112x1x1xf32>) -> tensor<112x672x1x1xf32>
    %v3289 = stablehlo.convolution(%v3286, %v3288)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v3290 = stablehlo.reshape %v3289 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v3291 = stablehlo.reshape %v1128 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3292 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3293 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3294 = stablehlo.reduce(%v3291 init: %v3292) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3295 = stablehlo.broadcast_in_dim %v3294, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3296 = stablehlo.divide %v3295, %v3293 : tensor<32x672x14x14xf32>
    %v3297 = stablehlo.subtract %v3291, %v3296 : tensor<32x672x14x14xf32>
    %v3298 = stablehlo.multiply %v3297, %v3297 : tensor<32x672x14x14xf32>
    %v3299 = stablehlo.reduce(%v3298 init: %v3292) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3300 = stablehlo.broadcast_in_dim %v3299, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3301 = stablehlo.divide %v3300, %v3293 : tensor<32x672x14x14xf32>
    %v3302 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3303 = stablehlo.add %v3301, %v3302 : tensor<32x672x14x14xf32>
    %v3304 = stablehlo.rsqrt %v3303 : tensor<32x672x14x14xf32>
    %v3305 = stablehlo.multiply %v3297, %v3304 : tensor<32x672x14x14xf32>
    %v3306 = stablehlo.reshape %v3255 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3307 = stablehlo.multiply %v3306, %v3305 : tensor<32x672x14x14xf32>
    %v3308 = stablehlo.reduce(%v3307 init: %v3292) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3309 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3310 = stablehlo.multiply %v3308, %v3309 : tensor<672xf32>
    %v3311 = stablehlo.subtract %b12eg, %v3310 : tensor<672xf32>
    %v3312 = stablehlo.reshape %v3255 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3313 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3314 = stablehlo.reduce(%v3312 init: %v3313) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3315 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3316 = stablehlo.multiply %v3314, %v3315 : tensor<672xf32>
    %v3317 = stablehlo.subtract %b12ebt, %v3316 : tensor<672xf32>
    %v3318 = stablehlo.reshape %v1123 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3319 = stablehlo.reshape %v3285 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3320 = stablehlo.transpose %v3318, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %v3321 = stablehlo.transpose %v3319, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3322 = stablehlo.convolution(%v3320, %v3321)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<112x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<112x672x1x1xf32>
    %v3323 = stablehlo.transpose %v3322, dims = [1, 0, 2, 3] : (tensor<112x672x1x1xf32>) -> tensor<672x112x1x1xf32>
    %v3324 = stablehlo.constant dense<0.05> : tensor<672x112x1x1xf32>
    %v3325 = stablehlo.multiply %v3323, %v3324 : tensor<672x112x1x1xf32>
    %v3326 = stablehlo.subtract %b12eW, %v3325 : tensor<672x112x1x1xf32>
    %v3327 = stablehlo.reshape %v1099 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3328 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3329 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v3330 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v3331 = stablehlo.reduce(%v3327 init: %v3328) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3332 = stablehlo.broadcast_in_dim %v3331, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3333 = stablehlo.divide %v3332, %v3329 : tensor<32x112x14x14xf32>
    %v3334 = stablehlo.subtract %v3327, %v3333 : tensor<32x112x14x14xf32>
    %v3335 = stablehlo.multiply %v3334, %v3334 : tensor<32x112x14x14xf32>
    %v3336 = stablehlo.reduce(%v3335 init: %v3328) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3337 = stablehlo.broadcast_in_dim %v3336, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3338 = stablehlo.divide %v3337, %v3329 : tensor<32x112x14x14xf32>
    %v3339 = stablehlo.add %v3338, %v3330 : tensor<32x112x14x14xf32>
    %v3340 = stablehlo.rsqrt %v3339 : tensor<32x112x14x14xf32>
    %v3341 = stablehlo.multiply %v3334, %v3340 : tensor<32x112x14x14xf32>
    %v3342 = stablehlo.broadcast_in_dim %b11pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3343 = stablehlo.reshape %v3290 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3344 = stablehlo.multiply %v3342, %v3343 : tensor<32x112x14x14xf32>
    %v3345 = stablehlo.reduce(%v3344 init: %v3328) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3346 = stablehlo.broadcast_in_dim %v3345, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3347 = stablehlo.multiply %v3341, %v3344 : tensor<32x112x14x14xf32>
    %v3348 = stablehlo.reduce(%v3347 init: %v3328) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3349 = stablehlo.broadcast_in_dim %v3348, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3350 = stablehlo.multiply %v3344, %v3329 : tensor<32x112x14x14xf32>
    %v3351 = stablehlo.subtract %v3350, %v3346 : tensor<32x112x14x14xf32>
    %v3352 = stablehlo.multiply %v3341, %v3349 : tensor<32x112x14x14xf32>
    %v3353 = stablehlo.subtract %v3351, %v3352 : tensor<32x112x14x14xf32>
    %v3354 = stablehlo.divide %v3340, %v3329 : tensor<32x112x14x14xf32>
    %v3355 = stablehlo.multiply %v3354, %v3353 : tensor<32x112x14x14xf32>
    %v3356 = stablehlo.reshape %v3355 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v3357 = stablehlo.reshape %v3356 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3358 = stablehlo.reverse %b11pW, dims = [2, 3] : tensor<112x672x1x1xf32>
    %v3359 = stablehlo.transpose %v3358, dims = [1, 0, 2, 3] : (tensor<112x672x1x1xf32>) -> tensor<672x112x1x1xf32>
    %v3360 = stablehlo.convolution(%v3357, %v3359)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v3361 = stablehlo.reshape %v3360 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3362 = stablehlo.reshape %v1099 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3363 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3364 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v3365 = stablehlo.reduce(%v3362 init: %v3363) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3366 = stablehlo.broadcast_in_dim %v3365, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3367 = stablehlo.divide %v3366, %v3364 : tensor<32x112x14x14xf32>
    %v3368 = stablehlo.subtract %v3362, %v3367 : tensor<32x112x14x14xf32>
    %v3369 = stablehlo.multiply %v3368, %v3368 : tensor<32x112x14x14xf32>
    %v3370 = stablehlo.reduce(%v3369 init: %v3363) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3371 = stablehlo.broadcast_in_dim %v3370, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3372 = stablehlo.divide %v3371, %v3364 : tensor<32x112x14x14xf32>
    %v3373 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v3374 = stablehlo.add %v3372, %v3373 : tensor<32x112x14x14xf32>
    %v3375 = stablehlo.rsqrt %v3374 : tensor<32x112x14x14xf32>
    %v3376 = stablehlo.multiply %v3368, %v3375 : tensor<32x112x14x14xf32>
    %v3377 = stablehlo.reshape %v3290 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3378 = stablehlo.multiply %v3377, %v3376 : tensor<32x112x14x14xf32>
    %v3379 = stablehlo.reduce(%v3378 init: %v3363) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3380 = stablehlo.constant dense<0.05> : tensor<112xf32>
    %v3381 = stablehlo.multiply %v3379, %v3380 : tensor<112xf32>
    %v3382 = stablehlo.subtract %b11pg, %v3381 : tensor<112xf32>
    %v3383 = stablehlo.reshape %v3290 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3384 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3385 = stablehlo.reduce(%v3383 init: %v3384) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3386 = stablehlo.constant dense<0.05> : tensor<112xf32>
    %v3387 = stablehlo.multiply %v3385, %v3386 : tensor<112xf32>
    %v3388 = stablehlo.subtract %b11pbt, %v3387 : tensor<112xf32>
    %v3389 = stablehlo.reshape %v1094 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3390 = stablehlo.reshape %v3356 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3391 = stablehlo.transpose %v3389, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3392 = stablehlo.transpose %v3390, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %v3393 = stablehlo.convolution(%v3391, %v3392)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<672x32x14x14xf32>, tensor<112x32x14x14xf32>) -> tensor<672x112x1x1xf32>
    %v3394 = stablehlo.transpose %v3393, dims = [1, 0, 2, 3] : (tensor<672x112x1x1xf32>) -> tensor<112x672x1x1xf32>
    %v3395 = stablehlo.constant dense<0.05> : tensor<112x672x1x1xf32>
    %v3396 = stablehlo.multiply %v3394, %v3395 : tensor<112x672x1x1xf32>
    %v3397 = stablehlo.subtract %b11pW, %v3396 : tensor<112x672x1x1xf32>
    %v3398 = stablehlo.reshape %v1077 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3399 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3400 = stablehlo.reduce(%v3398 init: %v3399) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3401 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v3402 = stablehlo.divide %v3400, %v3401 : tensor<32x672xf32>
    %v3403 = stablehlo.dot_general %v3402, %b11zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v3404 = stablehlo.broadcast_in_dim %b11zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v3405 = stablehlo.add %v3403, %v3404 : tensor<32x28xf32>
    %v3406 = stablehlo.logistic %v3405 : tensor<32x28xf32>
    %v3407 = stablehlo.multiply %v3405, %v3406 : tensor<32x28xf32>
    %v3408 = stablehlo.dot_general %v3407, %b11zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v3409 = stablehlo.broadcast_in_dim %b11zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v3410 = stablehlo.add %v3408, %v3409 : tensor<32x672xf32>
    %v3411 = stablehlo.logistic %v3410 : tensor<32x672xf32>
    %v3412 = stablehlo.reshape %v3361 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3413 = stablehlo.broadcast_in_dim %v3411, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v3414 = stablehlo.multiply %v3413, %v3412 : tensor<32x672x14x14xf32>
    %v3415 = stablehlo.multiply %v3398, %v3412 : tensor<32x672x14x14xf32>
    %v3416 = stablehlo.reduce(%v3415 init: %v3399) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3417 = stablehlo.constant dense<1.0> : tensor<32x672xf32>
    %v3418 = stablehlo.subtract %v3417, %v3411 : tensor<32x672xf32>
    %v3419 = stablehlo.multiply %v3411, %v3418 : tensor<32x672xf32>
    %v3420 = stablehlo.multiply %v3416, %v3419 : tensor<32x672xf32>
    %v3421 = stablehlo.dot_general %v3420, %b11zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<28x672xf32>) -> tensor<32x28xf32>
    %v3422 = stablehlo.logistic %v3405 : tensor<32x28xf32>
    %v3423 = stablehlo.constant dense<1.0> : tensor<32x28xf32>
    %v3424 = stablehlo.subtract %v3423, %v3422 : tensor<32x28xf32>
    %v3425 = stablehlo.multiply %v3405, %v3424 : tensor<32x28xf32>
    %v3426 = stablehlo.add %v3423, %v3425 : tensor<32x28xf32>
    %v3427 = stablehlo.multiply %v3422, %v3426 : tensor<32x28xf32>
    %v3428 = stablehlo.multiply %v3421, %v3427 : tensor<32x28xf32>
    %v3429 = stablehlo.dot_general %v3428, %b11zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<672x28xf32>) -> tensor<32x672xf32>
    %v3430 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v3431 = stablehlo.divide %v3429, %v3430 : tensor<32x672xf32>
    %v3432 = stablehlo.broadcast_in_dim %v3431, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v3433 = stablehlo.add %v3414, %v3432 : tensor<32x672x14x14xf32>
    %v3434 = stablehlo.reshape %v3433 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3435 = stablehlo.reshape %v1077 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3436 = stablehlo.reshape %v3361 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3437 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3438 = stablehlo.multiply %v3435, %v3436 : tensor<32x672x14x14xf32>
    %v3439 = stablehlo.reduce(%v3438 init: %v3437) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3440 = stablehlo.logistic %v1090 : tensor<32x672xf32>
    %v3441 = stablehlo.constant dense<1.0> : tensor<32x672xf32>
    %v3442 = stablehlo.subtract %v3441, %v3440 : tensor<32x672xf32>
    %v3443 = stablehlo.multiply %v3440, %v3442 : tensor<32x672xf32>
    %v3444 = stablehlo.multiply %v3439, %v3443 : tensor<32x672xf32>
    %v3445 = stablehlo.dot_general %v1087, %v3444, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<32x672xf32>) -> tensor<28x672xf32>
    %v3446 = stablehlo.constant dense<0.05> : tensor<28x672xf32>
    %v3447 = stablehlo.multiply %v3445, %v3446 : tensor<28x672xf32>
    %v3448 = stablehlo.subtract %b11zW2, %v3447 : tensor<28x672xf32>
    %v3449 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3450 = stablehlo.reduce(%v3444 init: %v3449) applies stablehlo.add across dimensions = [0] : (tensor<32x672xf32>, tensor<f32>) -> tensor<672xf32>
    %v3451 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3452 = stablehlo.multiply %v3450, %v3451 : tensor<672xf32>
    %v3453 = stablehlo.subtract %b11zb2, %v3452 : tensor<672xf32>
    %v3454 = stablehlo.reshape %v3444 : (tensor<32x672xf32>) -> tensor<32x1x672xf32>
    %v3455 = stablehlo.dot_general %v3454, %b11zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x672xf32>, tensor<28x672xf32>) -> tensor<32x1x28xf32>
    %v3456 = stablehlo.reshape %v3455 : (tensor<32x1x28xf32>) -> tensor<32x28xf32>
    %v3457 = stablehlo.logistic %v1085 : tensor<32x28xf32>
    %v3458 = stablehlo.constant dense<1.0> : tensor<32x28xf32>
    %v3459 = stablehlo.subtract %v3458, %v3457 : tensor<32x28xf32>
    %v3460 = stablehlo.multiply %v1085, %v3459 : tensor<32x28xf32>
    %v3461 = stablehlo.add %v3458, %v3460 : tensor<32x28xf32>
    %v3462 = stablehlo.multiply %v3457, %v3461 : tensor<32x28xf32>
    %v3463 = stablehlo.multiply %v3456, %v3462 : tensor<32x28xf32>
    %v3464 = stablehlo.dot_general %v1082, %v3463, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<32x28xf32>) -> tensor<672x28xf32>
    %v3465 = stablehlo.constant dense<0.05> : tensor<672x28xf32>
    %v3466 = stablehlo.multiply %v3464, %v3465 : tensor<672x28xf32>
    %v3467 = stablehlo.subtract %b11zW1, %v3466 : tensor<672x28xf32>
    %v3468 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3469 = stablehlo.reduce(%v3463 init: %v3468) applies stablehlo.add across dimensions = [0] : (tensor<32x28xf32>, tensor<f32>) -> tensor<28xf32>
    %v3470 = stablehlo.constant dense<0.05> : tensor<28xf32>
    %v3471 = stablehlo.multiply %v3469, %v3470 : tensor<28xf32>
    %v3472 = stablehlo.subtract %b11zb1, %v3471 : tensor<28xf32>
    %v3473 = stablehlo.reshape %v3434 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3474 = stablehlo.reshape %v1073 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3475 = stablehlo.logistic %v3474 : tensor<32x672x14x14xf32>
    %v3476 = stablehlo.constant dense<1.0> : tensor<32x672x14x14xf32>
    %v3477 = stablehlo.subtract %v3476, %v3475 : tensor<32x672x14x14xf32>
    %v3478 = stablehlo.multiply %v3474, %v3477 : tensor<32x672x14x14xf32>
    %v3479 = stablehlo.add %v3476, %v3478 : tensor<32x672x14x14xf32>
    %v3480 = stablehlo.multiply %v3475, %v3479 : tensor<32x672x14x14xf32>
    %v3481 = stablehlo.multiply %v3473, %v3480 : tensor<32x672x14x14xf32>
    %v3482 = stablehlo.reshape %v3481 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3483 = stablehlo.reshape %v1053 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3484 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3485 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3486 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3487 = stablehlo.reduce(%v3483 init: %v3484) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3488 = stablehlo.broadcast_in_dim %v3487, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3489 = stablehlo.divide %v3488, %v3485 : tensor<32x672x14x14xf32>
    %v3490 = stablehlo.subtract %v3483, %v3489 : tensor<32x672x14x14xf32>
    %v3491 = stablehlo.multiply %v3490, %v3490 : tensor<32x672x14x14xf32>
    %v3492 = stablehlo.reduce(%v3491 init: %v3484) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3493 = stablehlo.broadcast_in_dim %v3492, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3494 = stablehlo.divide %v3493, %v3485 : tensor<32x672x14x14xf32>
    %v3495 = stablehlo.add %v3494, %v3486 : tensor<32x672x14x14xf32>
    %v3496 = stablehlo.rsqrt %v3495 : tensor<32x672x14x14xf32>
    %v3497 = stablehlo.multiply %v3490, %v3496 : tensor<32x672x14x14xf32>
    %v3498 = stablehlo.broadcast_in_dim %b11dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3499 = stablehlo.reshape %v3482 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3500 = stablehlo.multiply %v3498, %v3499 : tensor<32x672x14x14xf32>
    %v3501 = stablehlo.reduce(%v3500 init: %v3484) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3502 = stablehlo.broadcast_in_dim %v3501, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3503 = stablehlo.multiply %v3497, %v3500 : tensor<32x672x14x14xf32>
    %v3504 = stablehlo.reduce(%v3503 init: %v3484) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3505 = stablehlo.broadcast_in_dim %v3504, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3506 = stablehlo.multiply %v3500, %v3485 : tensor<32x672x14x14xf32>
    %v3507 = stablehlo.subtract %v3506, %v3502 : tensor<32x672x14x14xf32>
    %v3508 = stablehlo.multiply %v3497, %v3505 : tensor<32x672x14x14xf32>
    %v3509 = stablehlo.subtract %v3507, %v3508 : tensor<32x672x14x14xf32>
    %v3510 = stablehlo.divide %v3496, %v3485 : tensor<32x672x14x14xf32>
    %v3511 = stablehlo.multiply %v3510, %v3509 : tensor<32x672x14x14xf32>
    %v3512 = stablehlo.reshape %v3511 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3513 = stablehlo.reshape %v3512 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3514 = stablehlo.reverse %b11dW, dims = [2, 3] : tensor<672x1x5x5xf32>
    %v3515 = stablehlo.convolution(%v3513, %v3514)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %v3516 = stablehlo.reshape %v3515 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3517 = stablehlo.reshape %v1053 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3518 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3519 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3520 = stablehlo.reduce(%v3517 init: %v3518) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3521 = stablehlo.broadcast_in_dim %v3520, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3522 = stablehlo.divide %v3521, %v3519 : tensor<32x672x14x14xf32>
    %v3523 = stablehlo.subtract %v3517, %v3522 : tensor<32x672x14x14xf32>
    %v3524 = stablehlo.multiply %v3523, %v3523 : tensor<32x672x14x14xf32>
    %v3525 = stablehlo.reduce(%v3524 init: %v3518) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3526 = stablehlo.broadcast_in_dim %v3525, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3527 = stablehlo.divide %v3526, %v3519 : tensor<32x672x14x14xf32>
    %v3528 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3529 = stablehlo.add %v3527, %v3528 : tensor<32x672x14x14xf32>
    %v3530 = stablehlo.rsqrt %v3529 : tensor<32x672x14x14xf32>
    %v3531 = stablehlo.multiply %v3523, %v3530 : tensor<32x672x14x14xf32>
    %v3532 = stablehlo.reshape %v3482 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3533 = stablehlo.multiply %v3532, %v3531 : tensor<32x672x14x14xf32>
    %v3534 = stablehlo.reduce(%v3533 init: %v3518) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3535 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3536 = stablehlo.multiply %v3534, %v3535 : tensor<672xf32>
    %v3537 = stablehlo.subtract %b11dg, %v3536 : tensor<672xf32>
    %v3538 = stablehlo.reshape %v3482 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3539 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3540 = stablehlo.reduce(%v3538 init: %v3539) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3541 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3542 = stablehlo.multiply %v3540, %v3541 : tensor<672xf32>
    %v3543 = stablehlo.subtract %b11dbt, %v3542 : tensor<672xf32>
    %v3544 = stablehlo.reshape %v1048 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3545 = stablehlo.reshape %v3512 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3546 = stablehlo.transpose %v3544, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3547 = stablehlo.transpose %v3545, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3548 = stablehlo.convolution(%v3546, %v3547)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 672 : i64, feature_group_count = 1 : i64} : (tensor<672x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<1x672x5x5xf32>
    %v3549 = stablehlo.reshape %v3548 : (tensor<1x672x5x5xf32>) -> tensor<672x1x5x5xf32>
    %v3550 = stablehlo.constant dense<0.05> : tensor<672x1x5x5xf32>
    %v3551 = stablehlo.multiply %v3549, %v3550 : tensor<672x1x5x5xf32>
    %v3552 = stablehlo.subtract %b11dW, %v3551 : tensor<672x1x5x5xf32>
    %v3553 = stablehlo.reshape %v3516 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3554 = stablehlo.reshape %v1044 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3555 = stablehlo.logistic %v3554 : tensor<32x672x14x14xf32>
    %v3556 = stablehlo.constant dense<1.0> : tensor<32x672x14x14xf32>
    %v3557 = stablehlo.subtract %v3556, %v3555 : tensor<32x672x14x14xf32>
    %v3558 = stablehlo.multiply %v3554, %v3557 : tensor<32x672x14x14xf32>
    %v3559 = stablehlo.add %v3556, %v3558 : tensor<32x672x14x14xf32>
    %v3560 = stablehlo.multiply %v3555, %v3559 : tensor<32x672x14x14xf32>
    %v3561 = stablehlo.multiply %v3553, %v3560 : tensor<32x672x14x14xf32>
    %v3562 = stablehlo.reshape %v3561 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3563 = stablehlo.reshape %v1024 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3564 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3565 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3566 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3567 = stablehlo.reduce(%v3563 init: %v3564) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3568 = stablehlo.broadcast_in_dim %v3567, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3569 = stablehlo.divide %v3568, %v3565 : tensor<32x672x14x14xf32>
    %v3570 = stablehlo.subtract %v3563, %v3569 : tensor<32x672x14x14xf32>
    %v3571 = stablehlo.multiply %v3570, %v3570 : tensor<32x672x14x14xf32>
    %v3572 = stablehlo.reduce(%v3571 init: %v3564) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3573 = stablehlo.broadcast_in_dim %v3572, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3574 = stablehlo.divide %v3573, %v3565 : tensor<32x672x14x14xf32>
    %v3575 = stablehlo.add %v3574, %v3566 : tensor<32x672x14x14xf32>
    %v3576 = stablehlo.rsqrt %v3575 : tensor<32x672x14x14xf32>
    %v3577 = stablehlo.multiply %v3570, %v3576 : tensor<32x672x14x14xf32>
    %v3578 = stablehlo.broadcast_in_dim %b11eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3579 = stablehlo.reshape %v3562 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3580 = stablehlo.multiply %v3578, %v3579 : tensor<32x672x14x14xf32>
    %v3581 = stablehlo.reduce(%v3580 init: %v3564) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3582 = stablehlo.broadcast_in_dim %v3581, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3583 = stablehlo.multiply %v3577, %v3580 : tensor<32x672x14x14xf32>
    %v3584 = stablehlo.reduce(%v3583 init: %v3564) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3585 = stablehlo.broadcast_in_dim %v3584, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3586 = stablehlo.multiply %v3580, %v3565 : tensor<32x672x14x14xf32>
    %v3587 = stablehlo.subtract %v3586, %v3582 : tensor<32x672x14x14xf32>
    %v3588 = stablehlo.multiply %v3577, %v3585 : tensor<32x672x14x14xf32>
    %v3589 = stablehlo.subtract %v3587, %v3588 : tensor<32x672x14x14xf32>
    %v3590 = stablehlo.divide %v3576, %v3565 : tensor<32x672x14x14xf32>
    %v3591 = stablehlo.multiply %v3590, %v3589 : tensor<32x672x14x14xf32>
    %v3592 = stablehlo.reshape %v3591 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3593 = stablehlo.reshape %v3592 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3594 = stablehlo.reverse %b11eW, dims = [2, 3] : tensor<672x112x1x1xf32>
    %v3595 = stablehlo.transpose %v3594, dims = [1, 0, 2, 3] : (tensor<672x112x1x1xf32>) -> tensor<112x672x1x1xf32>
    %v3596 = stablehlo.convolution(%v3593, %v3595)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v3597 = stablehlo.reshape %v3596 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v3598 = stablehlo.reshape %v1024 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3599 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3600 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3601 = stablehlo.reduce(%v3598 init: %v3599) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3602 = stablehlo.broadcast_in_dim %v3601, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3603 = stablehlo.divide %v3602, %v3600 : tensor<32x672x14x14xf32>
    %v3604 = stablehlo.subtract %v3598, %v3603 : tensor<32x672x14x14xf32>
    %v3605 = stablehlo.multiply %v3604, %v3604 : tensor<32x672x14x14xf32>
    %v3606 = stablehlo.reduce(%v3605 init: %v3599) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3607 = stablehlo.broadcast_in_dim %v3606, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3608 = stablehlo.divide %v3607, %v3600 : tensor<32x672x14x14xf32>
    %v3609 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3610 = stablehlo.add %v3608, %v3609 : tensor<32x672x14x14xf32>
    %v3611 = stablehlo.rsqrt %v3610 : tensor<32x672x14x14xf32>
    %v3612 = stablehlo.multiply %v3604, %v3611 : tensor<32x672x14x14xf32>
    %v3613 = stablehlo.reshape %v3562 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3614 = stablehlo.multiply %v3613, %v3612 : tensor<32x672x14x14xf32>
    %v3615 = stablehlo.reduce(%v3614 init: %v3599) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3616 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3617 = stablehlo.multiply %v3615, %v3616 : tensor<672xf32>
    %v3618 = stablehlo.subtract %b11eg, %v3617 : tensor<672xf32>
    %v3619 = stablehlo.reshape %v3562 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3620 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3621 = stablehlo.reduce(%v3619 init: %v3620) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3622 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3623 = stablehlo.multiply %v3621, %v3622 : tensor<672xf32>
    %v3624 = stablehlo.subtract %b11ebt, %v3623 : tensor<672xf32>
    %v3625 = stablehlo.reshape %v1019 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3626 = stablehlo.reshape %v3592 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3627 = stablehlo.transpose %v3625, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %v3628 = stablehlo.transpose %v3626, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3629 = stablehlo.convolution(%v3627, %v3628)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<112x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<112x672x1x1xf32>
    %v3630 = stablehlo.transpose %v3629, dims = [1, 0, 2, 3] : (tensor<112x672x1x1xf32>) -> tensor<672x112x1x1xf32>
    %v3631 = stablehlo.constant dense<0.05> : tensor<672x112x1x1xf32>
    %v3632 = stablehlo.multiply %v3630, %v3631 : tensor<672x112x1x1xf32>
    %v3633 = stablehlo.subtract %b11eW, %v3632 : tensor<672x112x1x1xf32>
    %v3634 = stablehlo.reshape %v3597 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3635 = stablehlo.reshape %v3290 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3636 = stablehlo.add %v3634, %v3635 : tensor<32x112x14x14xf32>
    %v3637 = stablehlo.reshape %v3636 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v3638 = stablehlo.reshape %v995 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3639 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3640 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v3641 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v3642 = stablehlo.reduce(%v3638 init: %v3639) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3643 = stablehlo.broadcast_in_dim %v3642, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3644 = stablehlo.divide %v3643, %v3640 : tensor<32x112x14x14xf32>
    %v3645 = stablehlo.subtract %v3638, %v3644 : tensor<32x112x14x14xf32>
    %v3646 = stablehlo.multiply %v3645, %v3645 : tensor<32x112x14x14xf32>
    %v3647 = stablehlo.reduce(%v3646 init: %v3639) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3648 = stablehlo.broadcast_in_dim %v3647, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3649 = stablehlo.divide %v3648, %v3640 : tensor<32x112x14x14xf32>
    %v3650 = stablehlo.add %v3649, %v3641 : tensor<32x112x14x14xf32>
    %v3651 = stablehlo.rsqrt %v3650 : tensor<32x112x14x14xf32>
    %v3652 = stablehlo.multiply %v3645, %v3651 : tensor<32x112x14x14xf32>
    %v3653 = stablehlo.broadcast_in_dim %b10pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3654 = stablehlo.reshape %v3637 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3655 = stablehlo.multiply %v3653, %v3654 : tensor<32x112x14x14xf32>
    %v3656 = stablehlo.reduce(%v3655 init: %v3639) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3657 = stablehlo.broadcast_in_dim %v3656, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3658 = stablehlo.multiply %v3652, %v3655 : tensor<32x112x14x14xf32>
    %v3659 = stablehlo.reduce(%v3658 init: %v3639) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3660 = stablehlo.broadcast_in_dim %v3659, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3661 = stablehlo.multiply %v3655, %v3640 : tensor<32x112x14x14xf32>
    %v3662 = stablehlo.subtract %v3661, %v3657 : tensor<32x112x14x14xf32>
    %v3663 = stablehlo.multiply %v3652, %v3660 : tensor<32x112x14x14xf32>
    %v3664 = stablehlo.subtract %v3662, %v3663 : tensor<32x112x14x14xf32>
    %v3665 = stablehlo.divide %v3651, %v3640 : tensor<32x112x14x14xf32>
    %v3666 = stablehlo.multiply %v3665, %v3664 : tensor<32x112x14x14xf32>
    %v3667 = stablehlo.reshape %v3666 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v3668 = stablehlo.reshape %v3667 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3669 = stablehlo.reverse %b10pW, dims = [2, 3] : tensor<112x672x1x1xf32>
    %v3670 = stablehlo.transpose %v3669, dims = [1, 0, 2, 3] : (tensor<112x672x1x1xf32>) -> tensor<672x112x1x1xf32>
    %v3671 = stablehlo.convolution(%v3668, %v3670)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v3672 = stablehlo.reshape %v3671 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3673 = stablehlo.reshape %v995 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3674 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3675 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v3676 = stablehlo.reduce(%v3673 init: %v3674) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3677 = stablehlo.broadcast_in_dim %v3676, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3678 = stablehlo.divide %v3677, %v3675 : tensor<32x112x14x14xf32>
    %v3679 = stablehlo.subtract %v3673, %v3678 : tensor<32x112x14x14xf32>
    %v3680 = stablehlo.multiply %v3679, %v3679 : tensor<32x112x14x14xf32>
    %v3681 = stablehlo.reduce(%v3680 init: %v3674) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3682 = stablehlo.broadcast_in_dim %v3681, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3683 = stablehlo.divide %v3682, %v3675 : tensor<32x112x14x14xf32>
    %v3684 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v3685 = stablehlo.add %v3683, %v3684 : tensor<32x112x14x14xf32>
    %v3686 = stablehlo.rsqrt %v3685 : tensor<32x112x14x14xf32>
    %v3687 = stablehlo.multiply %v3679, %v3686 : tensor<32x112x14x14xf32>
    %v3688 = stablehlo.reshape %v3637 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3689 = stablehlo.multiply %v3688, %v3687 : tensor<32x112x14x14xf32>
    %v3690 = stablehlo.reduce(%v3689 init: %v3674) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3691 = stablehlo.constant dense<0.05> : tensor<112xf32>
    %v3692 = stablehlo.multiply %v3690, %v3691 : tensor<112xf32>
    %v3693 = stablehlo.subtract %b10pg, %v3692 : tensor<112xf32>
    %v3694 = stablehlo.reshape %v3637 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3695 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3696 = stablehlo.reduce(%v3694 init: %v3695) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3697 = stablehlo.constant dense<0.05> : tensor<112xf32>
    %v3698 = stablehlo.multiply %v3696, %v3697 : tensor<112xf32>
    %v3699 = stablehlo.subtract %b10pbt, %v3698 : tensor<112xf32>
    %v3700 = stablehlo.reshape %v990 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3701 = stablehlo.reshape %v3667 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3702 = stablehlo.transpose %v3700, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3703 = stablehlo.transpose %v3701, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %v3704 = stablehlo.convolution(%v3702, %v3703)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<672x32x14x14xf32>, tensor<112x32x14x14xf32>) -> tensor<672x112x1x1xf32>
    %v3705 = stablehlo.transpose %v3704, dims = [1, 0, 2, 3] : (tensor<672x112x1x1xf32>) -> tensor<112x672x1x1xf32>
    %v3706 = stablehlo.constant dense<0.05> : tensor<112x672x1x1xf32>
    %v3707 = stablehlo.multiply %v3705, %v3706 : tensor<112x672x1x1xf32>
    %v3708 = stablehlo.subtract %b10pW, %v3707 : tensor<112x672x1x1xf32>
    %v3709 = stablehlo.reshape %v973 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3710 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3711 = stablehlo.reduce(%v3709 init: %v3710) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3712 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v3713 = stablehlo.divide %v3711, %v3712 : tensor<32x672xf32>
    %v3714 = stablehlo.dot_general %v3713, %b10zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v3715 = stablehlo.broadcast_in_dim %b10zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v3716 = stablehlo.add %v3714, %v3715 : tensor<32x28xf32>
    %v3717 = stablehlo.logistic %v3716 : tensor<32x28xf32>
    %v3718 = stablehlo.multiply %v3716, %v3717 : tensor<32x28xf32>
    %v3719 = stablehlo.dot_general %v3718, %b10zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v3720 = stablehlo.broadcast_in_dim %b10zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v3721 = stablehlo.add %v3719, %v3720 : tensor<32x672xf32>
    %v3722 = stablehlo.logistic %v3721 : tensor<32x672xf32>
    %v3723 = stablehlo.reshape %v3672 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3724 = stablehlo.broadcast_in_dim %v3722, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v3725 = stablehlo.multiply %v3724, %v3723 : tensor<32x672x14x14xf32>
    %v3726 = stablehlo.multiply %v3709, %v3723 : tensor<32x672x14x14xf32>
    %v3727 = stablehlo.reduce(%v3726 init: %v3710) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3728 = stablehlo.constant dense<1.0> : tensor<32x672xf32>
    %v3729 = stablehlo.subtract %v3728, %v3722 : tensor<32x672xf32>
    %v3730 = stablehlo.multiply %v3722, %v3729 : tensor<32x672xf32>
    %v3731 = stablehlo.multiply %v3727, %v3730 : tensor<32x672xf32>
    %v3732 = stablehlo.dot_general %v3731, %b10zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<28x672xf32>) -> tensor<32x28xf32>
    %v3733 = stablehlo.logistic %v3716 : tensor<32x28xf32>
    %v3734 = stablehlo.constant dense<1.0> : tensor<32x28xf32>
    %v3735 = stablehlo.subtract %v3734, %v3733 : tensor<32x28xf32>
    %v3736 = stablehlo.multiply %v3716, %v3735 : tensor<32x28xf32>
    %v3737 = stablehlo.add %v3734, %v3736 : tensor<32x28xf32>
    %v3738 = stablehlo.multiply %v3733, %v3737 : tensor<32x28xf32>
    %v3739 = stablehlo.multiply %v3732, %v3738 : tensor<32x28xf32>
    %v3740 = stablehlo.dot_general %v3739, %b10zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<672x28xf32>) -> tensor<32x672xf32>
    %v3741 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v3742 = stablehlo.divide %v3740, %v3741 : tensor<32x672xf32>
    %v3743 = stablehlo.broadcast_in_dim %v3742, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v3744 = stablehlo.add %v3725, %v3743 : tensor<32x672x14x14xf32>
    %v3745 = stablehlo.reshape %v3744 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3746 = stablehlo.reshape %v973 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3747 = stablehlo.reshape %v3672 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3748 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3749 = stablehlo.multiply %v3746, %v3747 : tensor<32x672x14x14xf32>
    %v3750 = stablehlo.reduce(%v3749 init: %v3748) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3751 = stablehlo.logistic %v986 : tensor<32x672xf32>
    %v3752 = stablehlo.constant dense<1.0> : tensor<32x672xf32>
    %v3753 = stablehlo.subtract %v3752, %v3751 : tensor<32x672xf32>
    %v3754 = stablehlo.multiply %v3751, %v3753 : tensor<32x672xf32>
    %v3755 = stablehlo.multiply %v3750, %v3754 : tensor<32x672xf32>
    %v3756 = stablehlo.dot_general %v983, %v3755, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<32x672xf32>) -> tensor<28x672xf32>
    %v3757 = stablehlo.constant dense<0.05> : tensor<28x672xf32>
    %v3758 = stablehlo.multiply %v3756, %v3757 : tensor<28x672xf32>
    %v3759 = stablehlo.subtract %b10zW2, %v3758 : tensor<28x672xf32>
    %v3760 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3761 = stablehlo.reduce(%v3755 init: %v3760) applies stablehlo.add across dimensions = [0] : (tensor<32x672xf32>, tensor<f32>) -> tensor<672xf32>
    %v3762 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3763 = stablehlo.multiply %v3761, %v3762 : tensor<672xf32>
    %v3764 = stablehlo.subtract %b10zb2, %v3763 : tensor<672xf32>
    %v3765 = stablehlo.reshape %v3755 : (tensor<32x672xf32>) -> tensor<32x1x672xf32>
    %v3766 = stablehlo.dot_general %v3765, %b10zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x672xf32>, tensor<28x672xf32>) -> tensor<32x1x28xf32>
    %v3767 = stablehlo.reshape %v3766 : (tensor<32x1x28xf32>) -> tensor<32x28xf32>
    %v3768 = stablehlo.logistic %v981 : tensor<32x28xf32>
    %v3769 = stablehlo.constant dense<1.0> : tensor<32x28xf32>
    %v3770 = stablehlo.subtract %v3769, %v3768 : tensor<32x28xf32>
    %v3771 = stablehlo.multiply %v981, %v3770 : tensor<32x28xf32>
    %v3772 = stablehlo.add %v3769, %v3771 : tensor<32x28xf32>
    %v3773 = stablehlo.multiply %v3768, %v3772 : tensor<32x28xf32>
    %v3774 = stablehlo.multiply %v3767, %v3773 : tensor<32x28xf32>
    %v3775 = stablehlo.dot_general %v978, %v3774, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<32x28xf32>) -> tensor<672x28xf32>
    %v3776 = stablehlo.constant dense<0.05> : tensor<672x28xf32>
    %v3777 = stablehlo.multiply %v3775, %v3776 : tensor<672x28xf32>
    %v3778 = stablehlo.subtract %b10zW1, %v3777 : tensor<672x28xf32>
    %v3779 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3780 = stablehlo.reduce(%v3774 init: %v3779) applies stablehlo.add across dimensions = [0] : (tensor<32x28xf32>, tensor<f32>) -> tensor<28xf32>
    %v3781 = stablehlo.constant dense<0.05> : tensor<28xf32>
    %v3782 = stablehlo.multiply %v3780, %v3781 : tensor<28xf32>
    %v3783 = stablehlo.subtract %b10zb1, %v3782 : tensor<28xf32>
    %v3784 = stablehlo.reshape %v3745 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3785 = stablehlo.reshape %v969 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3786 = stablehlo.logistic %v3785 : tensor<32x672x14x14xf32>
    %v3787 = stablehlo.constant dense<1.0> : tensor<32x672x14x14xf32>
    %v3788 = stablehlo.subtract %v3787, %v3786 : tensor<32x672x14x14xf32>
    %v3789 = stablehlo.multiply %v3785, %v3788 : tensor<32x672x14x14xf32>
    %v3790 = stablehlo.add %v3787, %v3789 : tensor<32x672x14x14xf32>
    %v3791 = stablehlo.multiply %v3786, %v3790 : tensor<32x672x14x14xf32>
    %v3792 = stablehlo.multiply %v3784, %v3791 : tensor<32x672x14x14xf32>
    %v3793 = stablehlo.reshape %v3792 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3794 = stablehlo.reshape %v949 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3795 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3796 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3797 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3798 = stablehlo.reduce(%v3794 init: %v3795) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3799 = stablehlo.broadcast_in_dim %v3798, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3800 = stablehlo.divide %v3799, %v3796 : tensor<32x672x14x14xf32>
    %v3801 = stablehlo.subtract %v3794, %v3800 : tensor<32x672x14x14xf32>
    %v3802 = stablehlo.multiply %v3801, %v3801 : tensor<32x672x14x14xf32>
    %v3803 = stablehlo.reduce(%v3802 init: %v3795) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3804 = stablehlo.broadcast_in_dim %v3803, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3805 = stablehlo.divide %v3804, %v3796 : tensor<32x672x14x14xf32>
    %v3806 = stablehlo.add %v3805, %v3797 : tensor<32x672x14x14xf32>
    %v3807 = stablehlo.rsqrt %v3806 : tensor<32x672x14x14xf32>
    %v3808 = stablehlo.multiply %v3801, %v3807 : tensor<32x672x14x14xf32>
    %v3809 = stablehlo.broadcast_in_dim %b10dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3810 = stablehlo.reshape %v3793 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3811 = stablehlo.multiply %v3809, %v3810 : tensor<32x672x14x14xf32>
    %v3812 = stablehlo.reduce(%v3811 init: %v3795) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3813 = stablehlo.broadcast_in_dim %v3812, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3814 = stablehlo.multiply %v3808, %v3811 : tensor<32x672x14x14xf32>
    %v3815 = stablehlo.reduce(%v3814 init: %v3795) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3816 = stablehlo.broadcast_in_dim %v3815, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3817 = stablehlo.multiply %v3811, %v3796 : tensor<32x672x14x14xf32>
    %v3818 = stablehlo.subtract %v3817, %v3813 : tensor<32x672x14x14xf32>
    %v3819 = stablehlo.multiply %v3808, %v3816 : tensor<32x672x14x14xf32>
    %v3820 = stablehlo.subtract %v3818, %v3819 : tensor<32x672x14x14xf32>
    %v3821 = stablehlo.divide %v3807, %v3796 : tensor<32x672x14x14xf32>
    %v3822 = stablehlo.multiply %v3821, %v3820 : tensor<32x672x14x14xf32>
    %v3823 = stablehlo.reshape %v3822 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3824 = stablehlo.reshape %v3823 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3825 = stablehlo.reverse %b10dW, dims = [2, 3] : tensor<672x1x5x5xf32>
    %v3826 = stablehlo.convolution(%v3824, %v3825)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %v3827 = stablehlo.reshape %v3826 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3828 = stablehlo.reshape %v949 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3829 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3830 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3831 = stablehlo.reduce(%v3828 init: %v3829) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3832 = stablehlo.broadcast_in_dim %v3831, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3833 = stablehlo.divide %v3832, %v3830 : tensor<32x672x14x14xf32>
    %v3834 = stablehlo.subtract %v3828, %v3833 : tensor<32x672x14x14xf32>
    %v3835 = stablehlo.multiply %v3834, %v3834 : tensor<32x672x14x14xf32>
    %v3836 = stablehlo.reduce(%v3835 init: %v3829) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3837 = stablehlo.broadcast_in_dim %v3836, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3838 = stablehlo.divide %v3837, %v3830 : tensor<32x672x14x14xf32>
    %v3839 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3840 = stablehlo.add %v3838, %v3839 : tensor<32x672x14x14xf32>
    %v3841 = stablehlo.rsqrt %v3840 : tensor<32x672x14x14xf32>
    %v3842 = stablehlo.multiply %v3834, %v3841 : tensor<32x672x14x14xf32>
    %v3843 = stablehlo.reshape %v3793 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3844 = stablehlo.multiply %v3843, %v3842 : tensor<32x672x14x14xf32>
    %v3845 = stablehlo.reduce(%v3844 init: %v3829) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3846 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3847 = stablehlo.multiply %v3845, %v3846 : tensor<672xf32>
    %v3848 = stablehlo.subtract %b10dg, %v3847 : tensor<672xf32>
    %v3849 = stablehlo.reshape %v3793 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3850 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3851 = stablehlo.reduce(%v3849 init: %v3850) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3852 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3853 = stablehlo.multiply %v3851, %v3852 : tensor<672xf32>
    %v3854 = stablehlo.subtract %b10dbt, %v3853 : tensor<672xf32>
    %v3855 = stablehlo.reshape %v944 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3856 = stablehlo.reshape %v3823 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3857 = stablehlo.transpose %v3855, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3858 = stablehlo.transpose %v3856, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3859 = stablehlo.convolution(%v3857, %v3858)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 672 : i64, feature_group_count = 1 : i64} : (tensor<672x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<1x672x5x5xf32>
    %v3860 = stablehlo.reshape %v3859 : (tensor<1x672x5x5xf32>) -> tensor<672x1x5x5xf32>
    %v3861 = stablehlo.constant dense<0.05> : tensor<672x1x5x5xf32>
    %v3862 = stablehlo.multiply %v3860, %v3861 : tensor<672x1x5x5xf32>
    %v3863 = stablehlo.subtract %b10dW, %v3862 : tensor<672x1x5x5xf32>
    %v3864 = stablehlo.reshape %v3827 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3865 = stablehlo.reshape %v940 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3866 = stablehlo.logistic %v3865 : tensor<32x672x14x14xf32>
    %v3867 = stablehlo.constant dense<1.0> : tensor<32x672x14x14xf32>
    %v3868 = stablehlo.subtract %v3867, %v3866 : tensor<32x672x14x14xf32>
    %v3869 = stablehlo.multiply %v3865, %v3868 : tensor<32x672x14x14xf32>
    %v3870 = stablehlo.add %v3867, %v3869 : tensor<32x672x14x14xf32>
    %v3871 = stablehlo.multiply %v3866, %v3870 : tensor<32x672x14x14xf32>
    %v3872 = stablehlo.multiply %v3864, %v3871 : tensor<32x672x14x14xf32>
    %v3873 = stablehlo.reshape %v3872 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3874 = stablehlo.reshape %v920 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3875 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3876 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3877 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3878 = stablehlo.reduce(%v3874 init: %v3875) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3879 = stablehlo.broadcast_in_dim %v3878, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3880 = stablehlo.divide %v3879, %v3876 : tensor<32x672x14x14xf32>
    %v3881 = stablehlo.subtract %v3874, %v3880 : tensor<32x672x14x14xf32>
    %v3882 = stablehlo.multiply %v3881, %v3881 : tensor<32x672x14x14xf32>
    %v3883 = stablehlo.reduce(%v3882 init: %v3875) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3884 = stablehlo.broadcast_in_dim %v3883, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3885 = stablehlo.divide %v3884, %v3876 : tensor<32x672x14x14xf32>
    %v3886 = stablehlo.add %v3885, %v3877 : tensor<32x672x14x14xf32>
    %v3887 = stablehlo.rsqrt %v3886 : tensor<32x672x14x14xf32>
    %v3888 = stablehlo.multiply %v3881, %v3887 : tensor<32x672x14x14xf32>
    %v3889 = stablehlo.broadcast_in_dim %b10eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3890 = stablehlo.reshape %v3873 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3891 = stablehlo.multiply %v3889, %v3890 : tensor<32x672x14x14xf32>
    %v3892 = stablehlo.reduce(%v3891 init: %v3875) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3893 = stablehlo.broadcast_in_dim %v3892, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3894 = stablehlo.multiply %v3888, %v3891 : tensor<32x672x14x14xf32>
    %v3895 = stablehlo.reduce(%v3894 init: %v3875) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3896 = stablehlo.broadcast_in_dim %v3895, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3897 = stablehlo.multiply %v3891, %v3876 : tensor<32x672x14x14xf32>
    %v3898 = stablehlo.subtract %v3897, %v3893 : tensor<32x672x14x14xf32>
    %v3899 = stablehlo.multiply %v3888, %v3896 : tensor<32x672x14x14xf32>
    %v3900 = stablehlo.subtract %v3898, %v3899 : tensor<32x672x14x14xf32>
    %v3901 = stablehlo.divide %v3887, %v3876 : tensor<32x672x14x14xf32>
    %v3902 = stablehlo.multiply %v3901, %v3900 : tensor<32x672x14x14xf32>
    %v3903 = stablehlo.reshape %v3902 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3904 = stablehlo.reshape %v3903 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3905 = stablehlo.reverse %b10eW, dims = [2, 3] : tensor<672x112x1x1xf32>
    %v3906 = stablehlo.transpose %v3905, dims = [1, 0, 2, 3] : (tensor<672x112x1x1xf32>) -> tensor<112x672x1x1xf32>
    %v3907 = stablehlo.convolution(%v3904, %v3906)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v3908 = stablehlo.reshape %v3907 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v3909 = stablehlo.reshape %v920 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3910 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3911 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3912 = stablehlo.reduce(%v3909 init: %v3910) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3913 = stablehlo.broadcast_in_dim %v3912, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3914 = stablehlo.divide %v3913, %v3911 : tensor<32x672x14x14xf32>
    %v3915 = stablehlo.subtract %v3909, %v3914 : tensor<32x672x14x14xf32>
    %v3916 = stablehlo.multiply %v3915, %v3915 : tensor<32x672x14x14xf32>
    %v3917 = stablehlo.reduce(%v3916 init: %v3910) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3918 = stablehlo.broadcast_in_dim %v3917, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3919 = stablehlo.divide %v3918, %v3911 : tensor<32x672x14x14xf32>
    %v3920 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3921 = stablehlo.add %v3919, %v3920 : tensor<32x672x14x14xf32>
    %v3922 = stablehlo.rsqrt %v3921 : tensor<32x672x14x14xf32>
    %v3923 = stablehlo.multiply %v3915, %v3922 : tensor<32x672x14x14xf32>
    %v3924 = stablehlo.reshape %v3873 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3925 = stablehlo.multiply %v3924, %v3923 : tensor<32x672x14x14xf32>
    %v3926 = stablehlo.reduce(%v3925 init: %v3910) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3927 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3928 = stablehlo.multiply %v3926, %v3927 : tensor<672xf32>
    %v3929 = stablehlo.subtract %b10eg, %v3928 : tensor<672xf32>
    %v3930 = stablehlo.reshape %v3873 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3931 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3932 = stablehlo.reduce(%v3930 init: %v3931) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3933 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3934 = stablehlo.multiply %v3932, %v3933 : tensor<672xf32>
    %v3935 = stablehlo.subtract %b10ebt, %v3934 : tensor<672xf32>
    %v3936 = stablehlo.reshape %v915 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3937 = stablehlo.reshape %v3903 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3938 = stablehlo.transpose %v3936, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %v3939 = stablehlo.transpose %v3937, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3940 = stablehlo.convolution(%v3938, %v3939)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<112x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<112x672x1x1xf32>
    %v3941 = stablehlo.transpose %v3940, dims = [1, 0, 2, 3] : (tensor<112x672x1x1xf32>) -> tensor<672x112x1x1xf32>
    %v3942 = stablehlo.constant dense<0.05> : tensor<672x112x1x1xf32>
    %v3943 = stablehlo.multiply %v3941, %v3942 : tensor<672x112x1x1xf32>
    %v3944 = stablehlo.subtract %b10eW, %v3943 : tensor<672x112x1x1xf32>
    %v3945 = stablehlo.reshape %v3908 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3946 = stablehlo.reshape %v3637 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3947 = stablehlo.add %v3945, %v3946 : tensor<32x112x14x14xf32>
    %v3948 = stablehlo.reshape %v3947 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v3949 = stablehlo.reshape %v895 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3950 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3951 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v3952 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v3953 = stablehlo.reduce(%v3949 init: %v3950) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3954 = stablehlo.broadcast_in_dim %v3953, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3955 = stablehlo.divide %v3954, %v3951 : tensor<32x112x14x14xf32>
    %v3956 = stablehlo.subtract %v3949, %v3955 : tensor<32x112x14x14xf32>
    %v3957 = stablehlo.multiply %v3956, %v3956 : tensor<32x112x14x14xf32>
    %v3958 = stablehlo.reduce(%v3957 init: %v3950) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3959 = stablehlo.broadcast_in_dim %v3958, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3960 = stablehlo.divide %v3959, %v3951 : tensor<32x112x14x14xf32>
    %v3961 = stablehlo.add %v3960, %v3952 : tensor<32x112x14x14xf32>
    %v3962 = stablehlo.rsqrt %v3961 : tensor<32x112x14x14xf32>
    %v3963 = stablehlo.multiply %v3956, %v3962 : tensor<32x112x14x14xf32>
    %v3964 = stablehlo.broadcast_in_dim %b9pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3965 = stablehlo.reshape %v3948 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3966 = stablehlo.multiply %v3964, %v3965 : tensor<32x112x14x14xf32>
    %v3967 = stablehlo.reduce(%v3966 init: %v3950) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3968 = stablehlo.broadcast_in_dim %v3967, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3969 = stablehlo.multiply %v3963, %v3966 : tensor<32x112x14x14xf32>
    %v3970 = stablehlo.reduce(%v3969 init: %v3950) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3971 = stablehlo.broadcast_in_dim %v3970, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3972 = stablehlo.multiply %v3966, %v3951 : tensor<32x112x14x14xf32>
    %v3973 = stablehlo.subtract %v3972, %v3968 : tensor<32x112x14x14xf32>
    %v3974 = stablehlo.multiply %v3963, %v3971 : tensor<32x112x14x14xf32>
    %v3975 = stablehlo.subtract %v3973, %v3974 : tensor<32x112x14x14xf32>
    %v3976 = stablehlo.divide %v3962, %v3951 : tensor<32x112x14x14xf32>
    %v3977 = stablehlo.multiply %v3976, %v3975 : tensor<32x112x14x14xf32>
    %v3978 = stablehlo.reshape %v3977 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v3979 = stablehlo.reshape %v3978 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3980 = stablehlo.reverse %b9pW, dims = [2, 3] : tensor<112x480x1x1xf32>
    %v3981 = stablehlo.transpose %v3980, dims = [1, 0, 2, 3] : (tensor<112x480x1x1xf32>) -> tensor<480x112x1x1xf32>
    %v3982 = stablehlo.convolution(%v3979, %v3981)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<480x112x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v3983 = stablehlo.reshape %v3982 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v3984 = stablehlo.reshape %v895 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3985 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3986 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v3987 = stablehlo.reduce(%v3984 init: %v3985) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3988 = stablehlo.broadcast_in_dim %v3987, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3989 = stablehlo.divide %v3988, %v3986 : tensor<32x112x14x14xf32>
    %v3990 = stablehlo.subtract %v3984, %v3989 : tensor<32x112x14x14xf32>
    %v3991 = stablehlo.multiply %v3990, %v3990 : tensor<32x112x14x14xf32>
    %v3992 = stablehlo.reduce(%v3991 init: %v3985) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3993 = stablehlo.broadcast_in_dim %v3992, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3994 = stablehlo.divide %v3993, %v3986 : tensor<32x112x14x14xf32>
    %v3995 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v3996 = stablehlo.add %v3994, %v3995 : tensor<32x112x14x14xf32>
    %v3997 = stablehlo.rsqrt %v3996 : tensor<32x112x14x14xf32>
    %v3998 = stablehlo.multiply %v3990, %v3997 : tensor<32x112x14x14xf32>
    %v3999 = stablehlo.reshape %v3948 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v4000 = stablehlo.multiply %v3999, %v3998 : tensor<32x112x14x14xf32>
    %v4001 = stablehlo.reduce(%v4000 init: %v3985) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v4002 = stablehlo.constant dense<0.05> : tensor<112xf32>
    %v4003 = stablehlo.multiply %v4001, %v4002 : tensor<112xf32>
    %v4004 = stablehlo.subtract %b9pg, %v4003 : tensor<112xf32>
    %v4005 = stablehlo.reshape %v3948 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v4006 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4007 = stablehlo.reduce(%v4005 init: %v4006) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v4008 = stablehlo.constant dense<0.05> : tensor<112xf32>
    %v4009 = stablehlo.multiply %v4007, %v4008 : tensor<112xf32>
    %v4010 = stablehlo.subtract %b9pbt, %v4009 : tensor<112xf32>
    %v4011 = stablehlo.reshape %v890 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4012 = stablehlo.reshape %v3978 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v4013 = stablehlo.transpose %v4011, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4014 = stablehlo.transpose %v4012, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %v4015 = stablehlo.convolution(%v4013, %v4014)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<112x32x14x14xf32>) -> tensor<480x112x1x1xf32>
    %v4016 = stablehlo.transpose %v4015, dims = [1, 0, 2, 3] : (tensor<480x112x1x1xf32>) -> tensor<112x480x1x1xf32>
    %v4017 = stablehlo.constant dense<0.05> : tensor<112x480x1x1xf32>
    %v4018 = stablehlo.multiply %v4016, %v4017 : tensor<112x480x1x1xf32>
    %v4019 = stablehlo.subtract %b9pW, %v4018 : tensor<112x480x1x1xf32>
    %v4020 = stablehlo.reshape %v873 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4021 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4022 = stablehlo.reduce(%v4020 init: %v4021) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4023 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v4024 = stablehlo.divide %v4022, %v4023 : tensor<32x480xf32>
    %v4025 = stablehlo.dot_general %v4024, %b9zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v4026 = stablehlo.broadcast_in_dim %b9zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v4027 = stablehlo.add %v4025, %v4026 : tensor<32x20xf32>
    %v4028 = stablehlo.logistic %v4027 : tensor<32x20xf32>
    %v4029 = stablehlo.multiply %v4027, %v4028 : tensor<32x20xf32>
    %v4030 = stablehlo.dot_general %v4029, %b9zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v4031 = stablehlo.broadcast_in_dim %b9zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v4032 = stablehlo.add %v4030, %v4031 : tensor<32x480xf32>
    %v4033 = stablehlo.logistic %v4032 : tensor<32x480xf32>
    %v4034 = stablehlo.reshape %v3983 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4035 = stablehlo.broadcast_in_dim %v4033, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v4036 = stablehlo.multiply %v4035, %v4034 : tensor<32x480x14x14xf32>
    %v4037 = stablehlo.multiply %v4020, %v4034 : tensor<32x480x14x14xf32>
    %v4038 = stablehlo.reduce(%v4037 init: %v4021) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4039 = stablehlo.constant dense<1.0> : tensor<32x480xf32>
    %v4040 = stablehlo.subtract %v4039, %v4033 : tensor<32x480xf32>
    %v4041 = stablehlo.multiply %v4033, %v4040 : tensor<32x480xf32>
    %v4042 = stablehlo.multiply %v4038, %v4041 : tensor<32x480xf32>
    %v4043 = stablehlo.dot_general %v4042, %b9zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<20x480xf32>) -> tensor<32x20xf32>
    %v4044 = stablehlo.logistic %v4027 : tensor<32x20xf32>
    %v4045 = stablehlo.constant dense<1.0> : tensor<32x20xf32>
    %v4046 = stablehlo.subtract %v4045, %v4044 : tensor<32x20xf32>
    %v4047 = stablehlo.multiply %v4027, %v4046 : tensor<32x20xf32>
    %v4048 = stablehlo.add %v4045, %v4047 : tensor<32x20xf32>
    %v4049 = stablehlo.multiply %v4044, %v4048 : tensor<32x20xf32>
    %v4050 = stablehlo.multiply %v4043, %v4049 : tensor<32x20xf32>
    %v4051 = stablehlo.dot_general %v4050, %b9zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<480x20xf32>) -> tensor<32x480xf32>
    %v4052 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v4053 = stablehlo.divide %v4051, %v4052 : tensor<32x480xf32>
    %v4054 = stablehlo.broadcast_in_dim %v4053, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v4055 = stablehlo.add %v4036, %v4054 : tensor<32x480x14x14xf32>
    %v4056 = stablehlo.reshape %v4055 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4057 = stablehlo.reshape %v873 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4058 = stablehlo.reshape %v3983 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4059 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4060 = stablehlo.multiply %v4057, %v4058 : tensor<32x480x14x14xf32>
    %v4061 = stablehlo.reduce(%v4060 init: %v4059) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4062 = stablehlo.logistic %v886 : tensor<32x480xf32>
    %v4063 = stablehlo.constant dense<1.0> : tensor<32x480xf32>
    %v4064 = stablehlo.subtract %v4063, %v4062 : tensor<32x480xf32>
    %v4065 = stablehlo.multiply %v4062, %v4064 : tensor<32x480xf32>
    %v4066 = stablehlo.multiply %v4061, %v4065 : tensor<32x480xf32>
    %v4067 = stablehlo.dot_general %v883, %v4066, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<32x480xf32>) -> tensor<20x480xf32>
    %v4068 = stablehlo.constant dense<0.05> : tensor<20x480xf32>
    %v4069 = stablehlo.multiply %v4067, %v4068 : tensor<20x480xf32>
    %v4070 = stablehlo.subtract %b9zW2, %v4069 : tensor<20x480xf32>
    %v4071 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4072 = stablehlo.reduce(%v4066 init: %v4071) applies stablehlo.add across dimensions = [0] : (tensor<32x480xf32>, tensor<f32>) -> tensor<480xf32>
    %v4073 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4074 = stablehlo.multiply %v4072, %v4073 : tensor<480xf32>
    %v4075 = stablehlo.subtract %b9zb2, %v4074 : tensor<480xf32>
    %v4076 = stablehlo.reshape %v4066 : (tensor<32x480xf32>) -> tensor<32x1x480xf32>
    %v4077 = stablehlo.dot_general %v4076, %b9zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x480xf32>, tensor<20x480xf32>) -> tensor<32x1x20xf32>
    %v4078 = stablehlo.reshape %v4077 : (tensor<32x1x20xf32>) -> tensor<32x20xf32>
    %v4079 = stablehlo.logistic %v881 : tensor<32x20xf32>
    %v4080 = stablehlo.constant dense<1.0> : tensor<32x20xf32>
    %v4081 = stablehlo.subtract %v4080, %v4079 : tensor<32x20xf32>
    %v4082 = stablehlo.multiply %v881, %v4081 : tensor<32x20xf32>
    %v4083 = stablehlo.add %v4080, %v4082 : tensor<32x20xf32>
    %v4084 = stablehlo.multiply %v4079, %v4083 : tensor<32x20xf32>
    %v4085 = stablehlo.multiply %v4078, %v4084 : tensor<32x20xf32>
    %v4086 = stablehlo.dot_general %v878, %v4085, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<32x20xf32>) -> tensor<480x20xf32>
    %v4087 = stablehlo.constant dense<0.05> : tensor<480x20xf32>
    %v4088 = stablehlo.multiply %v4086, %v4087 : tensor<480x20xf32>
    %v4089 = stablehlo.subtract %b9zW1, %v4088 : tensor<480x20xf32>
    %v4090 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4091 = stablehlo.reduce(%v4085 init: %v4090) applies stablehlo.add across dimensions = [0] : (tensor<32x20xf32>, tensor<f32>) -> tensor<20xf32>
    %v4092 = stablehlo.constant dense<0.05> : tensor<20xf32>
    %v4093 = stablehlo.multiply %v4091, %v4092 : tensor<20xf32>
    %v4094 = stablehlo.subtract %b9zb1, %v4093 : tensor<20xf32>
    %v4095 = stablehlo.reshape %v4056 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4096 = stablehlo.reshape %v869 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4097 = stablehlo.logistic %v4096 : tensor<32x480x14x14xf32>
    %v4098 = stablehlo.constant dense<1.0> : tensor<32x480x14x14xf32>
    %v4099 = stablehlo.subtract %v4098, %v4097 : tensor<32x480x14x14xf32>
    %v4100 = stablehlo.multiply %v4096, %v4099 : tensor<32x480x14x14xf32>
    %v4101 = stablehlo.add %v4098, %v4100 : tensor<32x480x14x14xf32>
    %v4102 = stablehlo.multiply %v4097, %v4101 : tensor<32x480x14x14xf32>
    %v4103 = stablehlo.multiply %v4095, %v4102 : tensor<32x480x14x14xf32>
    %v4104 = stablehlo.reshape %v4103 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4105 = stablehlo.reshape %v849 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4106 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4107 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4108 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4109 = stablehlo.reduce(%v4105 init: %v4106) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4110 = stablehlo.broadcast_in_dim %v4109, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4111 = stablehlo.divide %v4110, %v4107 : tensor<32x480x14x14xf32>
    %v4112 = stablehlo.subtract %v4105, %v4111 : tensor<32x480x14x14xf32>
    %v4113 = stablehlo.multiply %v4112, %v4112 : tensor<32x480x14x14xf32>
    %v4114 = stablehlo.reduce(%v4113 init: %v4106) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4115 = stablehlo.broadcast_in_dim %v4114, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4116 = stablehlo.divide %v4115, %v4107 : tensor<32x480x14x14xf32>
    %v4117 = stablehlo.add %v4116, %v4108 : tensor<32x480x14x14xf32>
    %v4118 = stablehlo.rsqrt %v4117 : tensor<32x480x14x14xf32>
    %v4119 = stablehlo.multiply %v4112, %v4118 : tensor<32x480x14x14xf32>
    %v4120 = stablehlo.broadcast_in_dim %b9dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4121 = stablehlo.reshape %v4104 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4122 = stablehlo.multiply %v4120, %v4121 : tensor<32x480x14x14xf32>
    %v4123 = stablehlo.reduce(%v4122 init: %v4106) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4124 = stablehlo.broadcast_in_dim %v4123, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4125 = stablehlo.multiply %v4119, %v4122 : tensor<32x480x14x14xf32>
    %v4126 = stablehlo.reduce(%v4125 init: %v4106) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4127 = stablehlo.broadcast_in_dim %v4126, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4128 = stablehlo.multiply %v4122, %v4107 : tensor<32x480x14x14xf32>
    %v4129 = stablehlo.subtract %v4128, %v4124 : tensor<32x480x14x14xf32>
    %v4130 = stablehlo.multiply %v4119, %v4127 : tensor<32x480x14x14xf32>
    %v4131 = stablehlo.subtract %v4129, %v4130 : tensor<32x480x14x14xf32>
    %v4132 = stablehlo.divide %v4118, %v4107 : tensor<32x480x14x14xf32>
    %v4133 = stablehlo.multiply %v4132, %v4131 : tensor<32x480x14x14xf32>
    %v4134 = stablehlo.reshape %v4133 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4135 = stablehlo.reshape %v4134 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4136 = stablehlo.reverse %b9dW, dims = [2, 3] : tensor<480x1x5x5xf32>
    %v4137 = stablehlo.convolution(%v4135, %v4136)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x5x5xf32>) -> tensor<32x480x14x14xf32>
    %v4138 = stablehlo.reshape %v4137 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4139 = stablehlo.reshape %v849 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4140 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4141 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4142 = stablehlo.reduce(%v4139 init: %v4140) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4143 = stablehlo.broadcast_in_dim %v4142, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4144 = stablehlo.divide %v4143, %v4141 : tensor<32x480x14x14xf32>
    %v4145 = stablehlo.subtract %v4139, %v4144 : tensor<32x480x14x14xf32>
    %v4146 = stablehlo.multiply %v4145, %v4145 : tensor<32x480x14x14xf32>
    %v4147 = stablehlo.reduce(%v4146 init: %v4140) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4148 = stablehlo.broadcast_in_dim %v4147, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4149 = stablehlo.divide %v4148, %v4141 : tensor<32x480x14x14xf32>
    %v4150 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4151 = stablehlo.add %v4149, %v4150 : tensor<32x480x14x14xf32>
    %v4152 = stablehlo.rsqrt %v4151 : tensor<32x480x14x14xf32>
    %v4153 = stablehlo.multiply %v4145, %v4152 : tensor<32x480x14x14xf32>
    %v4154 = stablehlo.reshape %v4104 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4155 = stablehlo.multiply %v4154, %v4153 : tensor<32x480x14x14xf32>
    %v4156 = stablehlo.reduce(%v4155 init: %v4140) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4157 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4158 = stablehlo.multiply %v4156, %v4157 : tensor<480xf32>
    %v4159 = stablehlo.subtract %b9dg, %v4158 : tensor<480xf32>
    %v4160 = stablehlo.reshape %v4104 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4161 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4162 = stablehlo.reduce(%v4160 init: %v4161) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4163 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4164 = stablehlo.multiply %v4162, %v4163 : tensor<480xf32>
    %v4165 = stablehlo.subtract %b9dbt, %v4164 : tensor<480xf32>
    %v4166 = stablehlo.reshape %v844 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4167 = stablehlo.reshape %v4134 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4168 = stablehlo.transpose %v4166, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4169 = stablehlo.transpose %v4167, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4170 = stablehlo.convolution(%v4168, %v4169)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 480 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<1x480x5x5xf32>
    %v4171 = stablehlo.reshape %v4170 : (tensor<1x480x5x5xf32>) -> tensor<480x1x5x5xf32>
    %v4172 = stablehlo.constant dense<0.05> : tensor<480x1x5x5xf32>
    %v4173 = stablehlo.multiply %v4171, %v4172 : tensor<480x1x5x5xf32>
    %v4174 = stablehlo.subtract %b9dW, %v4173 : tensor<480x1x5x5xf32>
    %v4175 = stablehlo.reshape %v4138 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4176 = stablehlo.reshape %v840 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4177 = stablehlo.logistic %v4176 : tensor<32x480x14x14xf32>
    %v4178 = stablehlo.constant dense<1.0> : tensor<32x480x14x14xf32>
    %v4179 = stablehlo.subtract %v4178, %v4177 : tensor<32x480x14x14xf32>
    %v4180 = stablehlo.multiply %v4176, %v4179 : tensor<32x480x14x14xf32>
    %v4181 = stablehlo.add %v4178, %v4180 : tensor<32x480x14x14xf32>
    %v4182 = stablehlo.multiply %v4177, %v4181 : tensor<32x480x14x14xf32>
    %v4183 = stablehlo.multiply %v4175, %v4182 : tensor<32x480x14x14xf32>
    %v4184 = stablehlo.reshape %v4183 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4185 = stablehlo.reshape %v820 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4186 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4187 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4188 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4189 = stablehlo.reduce(%v4185 init: %v4186) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4190 = stablehlo.broadcast_in_dim %v4189, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4191 = stablehlo.divide %v4190, %v4187 : tensor<32x480x14x14xf32>
    %v4192 = stablehlo.subtract %v4185, %v4191 : tensor<32x480x14x14xf32>
    %v4193 = stablehlo.multiply %v4192, %v4192 : tensor<32x480x14x14xf32>
    %v4194 = stablehlo.reduce(%v4193 init: %v4186) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4195 = stablehlo.broadcast_in_dim %v4194, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4196 = stablehlo.divide %v4195, %v4187 : tensor<32x480x14x14xf32>
    %v4197 = stablehlo.add %v4196, %v4188 : tensor<32x480x14x14xf32>
    %v4198 = stablehlo.rsqrt %v4197 : tensor<32x480x14x14xf32>
    %v4199 = stablehlo.multiply %v4192, %v4198 : tensor<32x480x14x14xf32>
    %v4200 = stablehlo.broadcast_in_dim %b9eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4201 = stablehlo.reshape %v4184 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4202 = stablehlo.multiply %v4200, %v4201 : tensor<32x480x14x14xf32>
    %v4203 = stablehlo.reduce(%v4202 init: %v4186) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4204 = stablehlo.broadcast_in_dim %v4203, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4205 = stablehlo.multiply %v4199, %v4202 : tensor<32x480x14x14xf32>
    %v4206 = stablehlo.reduce(%v4205 init: %v4186) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4207 = stablehlo.broadcast_in_dim %v4206, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4208 = stablehlo.multiply %v4202, %v4187 : tensor<32x480x14x14xf32>
    %v4209 = stablehlo.subtract %v4208, %v4204 : tensor<32x480x14x14xf32>
    %v4210 = stablehlo.multiply %v4199, %v4207 : tensor<32x480x14x14xf32>
    %v4211 = stablehlo.subtract %v4209, %v4210 : tensor<32x480x14x14xf32>
    %v4212 = stablehlo.divide %v4198, %v4187 : tensor<32x480x14x14xf32>
    %v4213 = stablehlo.multiply %v4212, %v4211 : tensor<32x480x14x14xf32>
    %v4214 = stablehlo.reshape %v4213 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4215 = stablehlo.reshape %v4214 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4216 = stablehlo.reverse %b9eW, dims = [2, 3] : tensor<480x80x1x1xf32>
    %v4217 = stablehlo.transpose %v4216, dims = [1, 0, 2, 3] : (tensor<480x80x1x1xf32>) -> tensor<80x480x1x1xf32>
    %v4218 = stablehlo.convolution(%v4215, %v4217)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v4219 = stablehlo.reshape %v4218 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v4220 = stablehlo.reshape %v820 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4221 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4222 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4223 = stablehlo.reduce(%v4220 init: %v4221) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4224 = stablehlo.broadcast_in_dim %v4223, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4225 = stablehlo.divide %v4224, %v4222 : tensor<32x480x14x14xf32>
    %v4226 = stablehlo.subtract %v4220, %v4225 : tensor<32x480x14x14xf32>
    %v4227 = stablehlo.multiply %v4226, %v4226 : tensor<32x480x14x14xf32>
    %v4228 = stablehlo.reduce(%v4227 init: %v4221) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4229 = stablehlo.broadcast_in_dim %v4228, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4230 = stablehlo.divide %v4229, %v4222 : tensor<32x480x14x14xf32>
    %v4231 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4232 = stablehlo.add %v4230, %v4231 : tensor<32x480x14x14xf32>
    %v4233 = stablehlo.rsqrt %v4232 : tensor<32x480x14x14xf32>
    %v4234 = stablehlo.multiply %v4226, %v4233 : tensor<32x480x14x14xf32>
    %v4235 = stablehlo.reshape %v4184 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4236 = stablehlo.multiply %v4235, %v4234 : tensor<32x480x14x14xf32>
    %v4237 = stablehlo.reduce(%v4236 init: %v4221) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4238 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4239 = stablehlo.multiply %v4237, %v4238 : tensor<480xf32>
    %v4240 = stablehlo.subtract %b9eg, %v4239 : tensor<480xf32>
    %v4241 = stablehlo.reshape %v4184 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4242 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4243 = stablehlo.reduce(%v4241 init: %v4242) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4244 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4245 = stablehlo.multiply %v4243, %v4244 : tensor<480xf32>
    %v4246 = stablehlo.subtract %b9ebt, %v4245 : tensor<480xf32>
    %v4247 = stablehlo.reshape %v815 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4248 = stablehlo.reshape %v4214 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4249 = stablehlo.transpose %v4247, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %v4250 = stablehlo.transpose %v4248, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4251 = stablehlo.convolution(%v4249, %v4250)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<80x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<80x480x1x1xf32>
    %v4252 = stablehlo.transpose %v4251, dims = [1, 0, 2, 3] : (tensor<80x480x1x1xf32>) -> tensor<480x80x1x1xf32>
    %v4253 = stablehlo.constant dense<0.05> : tensor<480x80x1x1xf32>
    %v4254 = stablehlo.multiply %v4252, %v4253 : tensor<480x80x1x1xf32>
    %v4255 = stablehlo.subtract %b9eW, %v4254 : tensor<480x80x1x1xf32>
    %v4256 = stablehlo.reshape %v791 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4257 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4258 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v4259 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v4260 = stablehlo.reduce(%v4256 init: %v4257) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4261 = stablehlo.broadcast_in_dim %v4260, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4262 = stablehlo.divide %v4261, %v4258 : tensor<32x80x14x14xf32>
    %v4263 = stablehlo.subtract %v4256, %v4262 : tensor<32x80x14x14xf32>
    %v4264 = stablehlo.multiply %v4263, %v4263 : tensor<32x80x14x14xf32>
    %v4265 = stablehlo.reduce(%v4264 init: %v4257) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4266 = stablehlo.broadcast_in_dim %v4265, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4267 = stablehlo.divide %v4266, %v4258 : tensor<32x80x14x14xf32>
    %v4268 = stablehlo.add %v4267, %v4259 : tensor<32x80x14x14xf32>
    %v4269 = stablehlo.rsqrt %v4268 : tensor<32x80x14x14xf32>
    %v4270 = stablehlo.multiply %v4263, %v4269 : tensor<32x80x14x14xf32>
    %v4271 = stablehlo.broadcast_in_dim %b8pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4272 = stablehlo.reshape %v4219 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4273 = stablehlo.multiply %v4271, %v4272 : tensor<32x80x14x14xf32>
    %v4274 = stablehlo.reduce(%v4273 init: %v4257) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4275 = stablehlo.broadcast_in_dim %v4274, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4276 = stablehlo.multiply %v4270, %v4273 : tensor<32x80x14x14xf32>
    %v4277 = stablehlo.reduce(%v4276 init: %v4257) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4278 = stablehlo.broadcast_in_dim %v4277, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4279 = stablehlo.multiply %v4273, %v4258 : tensor<32x80x14x14xf32>
    %v4280 = stablehlo.subtract %v4279, %v4275 : tensor<32x80x14x14xf32>
    %v4281 = stablehlo.multiply %v4270, %v4278 : tensor<32x80x14x14xf32>
    %v4282 = stablehlo.subtract %v4280, %v4281 : tensor<32x80x14x14xf32>
    %v4283 = stablehlo.divide %v4269, %v4258 : tensor<32x80x14x14xf32>
    %v4284 = stablehlo.multiply %v4283, %v4282 : tensor<32x80x14x14xf32>
    %v4285 = stablehlo.reshape %v4284 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v4286 = stablehlo.reshape %v4285 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4287 = stablehlo.reverse %b8pW, dims = [2, 3] : tensor<80x480x1x1xf32>
    %v4288 = stablehlo.transpose %v4287, dims = [1, 0, 2, 3] : (tensor<80x480x1x1xf32>) -> tensor<480x80x1x1xf32>
    %v4289 = stablehlo.convolution(%v4286, %v4288)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v4290 = stablehlo.reshape %v4289 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4291 = stablehlo.reshape %v791 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4292 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4293 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v4294 = stablehlo.reduce(%v4291 init: %v4292) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4295 = stablehlo.broadcast_in_dim %v4294, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4296 = stablehlo.divide %v4295, %v4293 : tensor<32x80x14x14xf32>
    %v4297 = stablehlo.subtract %v4291, %v4296 : tensor<32x80x14x14xf32>
    %v4298 = stablehlo.multiply %v4297, %v4297 : tensor<32x80x14x14xf32>
    %v4299 = stablehlo.reduce(%v4298 init: %v4292) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4300 = stablehlo.broadcast_in_dim %v4299, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4301 = stablehlo.divide %v4300, %v4293 : tensor<32x80x14x14xf32>
    %v4302 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v4303 = stablehlo.add %v4301, %v4302 : tensor<32x80x14x14xf32>
    %v4304 = stablehlo.rsqrt %v4303 : tensor<32x80x14x14xf32>
    %v4305 = stablehlo.multiply %v4297, %v4304 : tensor<32x80x14x14xf32>
    %v4306 = stablehlo.reshape %v4219 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4307 = stablehlo.multiply %v4306, %v4305 : tensor<32x80x14x14xf32>
    %v4308 = stablehlo.reduce(%v4307 init: %v4292) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4309 = stablehlo.constant dense<0.05> : tensor<80xf32>
    %v4310 = stablehlo.multiply %v4308, %v4309 : tensor<80xf32>
    %v4311 = stablehlo.subtract %b8pg, %v4310 : tensor<80xf32>
    %v4312 = stablehlo.reshape %v4219 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4313 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4314 = stablehlo.reduce(%v4312 init: %v4313) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4315 = stablehlo.constant dense<0.05> : tensor<80xf32>
    %v4316 = stablehlo.multiply %v4314, %v4315 : tensor<80xf32>
    %v4317 = stablehlo.subtract %b8pbt, %v4316 : tensor<80xf32>
    %v4318 = stablehlo.reshape %v786 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4319 = stablehlo.reshape %v4285 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4320 = stablehlo.transpose %v4318, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4321 = stablehlo.transpose %v4319, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %v4322 = stablehlo.convolution(%v4320, %v4321)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<80x32x14x14xf32>) -> tensor<480x80x1x1xf32>
    %v4323 = stablehlo.transpose %v4322, dims = [1, 0, 2, 3] : (tensor<480x80x1x1xf32>) -> tensor<80x480x1x1xf32>
    %v4324 = stablehlo.constant dense<0.05> : tensor<80x480x1x1xf32>
    %v4325 = stablehlo.multiply %v4323, %v4324 : tensor<80x480x1x1xf32>
    %v4326 = stablehlo.subtract %b8pW, %v4325 : tensor<80x480x1x1xf32>
    %v4327 = stablehlo.reshape %v769 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4328 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4329 = stablehlo.reduce(%v4327 init: %v4328) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4330 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v4331 = stablehlo.divide %v4329, %v4330 : tensor<32x480xf32>
    %v4332 = stablehlo.dot_general %v4331, %b8zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v4333 = stablehlo.broadcast_in_dim %b8zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v4334 = stablehlo.add %v4332, %v4333 : tensor<32x20xf32>
    %v4335 = stablehlo.logistic %v4334 : tensor<32x20xf32>
    %v4336 = stablehlo.multiply %v4334, %v4335 : tensor<32x20xf32>
    %v4337 = stablehlo.dot_general %v4336, %b8zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v4338 = stablehlo.broadcast_in_dim %b8zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v4339 = stablehlo.add %v4337, %v4338 : tensor<32x480xf32>
    %v4340 = stablehlo.logistic %v4339 : tensor<32x480xf32>
    %v4341 = stablehlo.reshape %v4290 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4342 = stablehlo.broadcast_in_dim %v4340, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v4343 = stablehlo.multiply %v4342, %v4341 : tensor<32x480x14x14xf32>
    %v4344 = stablehlo.multiply %v4327, %v4341 : tensor<32x480x14x14xf32>
    %v4345 = stablehlo.reduce(%v4344 init: %v4328) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4346 = stablehlo.constant dense<1.0> : tensor<32x480xf32>
    %v4347 = stablehlo.subtract %v4346, %v4340 : tensor<32x480xf32>
    %v4348 = stablehlo.multiply %v4340, %v4347 : tensor<32x480xf32>
    %v4349 = stablehlo.multiply %v4345, %v4348 : tensor<32x480xf32>
    %v4350 = stablehlo.dot_general %v4349, %b8zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<20x480xf32>) -> tensor<32x20xf32>
    %v4351 = stablehlo.logistic %v4334 : tensor<32x20xf32>
    %v4352 = stablehlo.constant dense<1.0> : tensor<32x20xf32>
    %v4353 = stablehlo.subtract %v4352, %v4351 : tensor<32x20xf32>
    %v4354 = stablehlo.multiply %v4334, %v4353 : tensor<32x20xf32>
    %v4355 = stablehlo.add %v4352, %v4354 : tensor<32x20xf32>
    %v4356 = stablehlo.multiply %v4351, %v4355 : tensor<32x20xf32>
    %v4357 = stablehlo.multiply %v4350, %v4356 : tensor<32x20xf32>
    %v4358 = stablehlo.dot_general %v4357, %b8zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<480x20xf32>) -> tensor<32x480xf32>
    %v4359 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v4360 = stablehlo.divide %v4358, %v4359 : tensor<32x480xf32>
    %v4361 = stablehlo.broadcast_in_dim %v4360, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v4362 = stablehlo.add %v4343, %v4361 : tensor<32x480x14x14xf32>
    %v4363 = stablehlo.reshape %v4362 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4364 = stablehlo.reshape %v769 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4365 = stablehlo.reshape %v4290 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4366 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4367 = stablehlo.multiply %v4364, %v4365 : tensor<32x480x14x14xf32>
    %v4368 = stablehlo.reduce(%v4367 init: %v4366) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4369 = stablehlo.logistic %v782 : tensor<32x480xf32>
    %v4370 = stablehlo.constant dense<1.0> : tensor<32x480xf32>
    %v4371 = stablehlo.subtract %v4370, %v4369 : tensor<32x480xf32>
    %v4372 = stablehlo.multiply %v4369, %v4371 : tensor<32x480xf32>
    %v4373 = stablehlo.multiply %v4368, %v4372 : tensor<32x480xf32>
    %v4374 = stablehlo.dot_general %v779, %v4373, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<32x480xf32>) -> tensor<20x480xf32>
    %v4375 = stablehlo.constant dense<0.05> : tensor<20x480xf32>
    %v4376 = stablehlo.multiply %v4374, %v4375 : tensor<20x480xf32>
    %v4377 = stablehlo.subtract %b8zW2, %v4376 : tensor<20x480xf32>
    %v4378 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4379 = stablehlo.reduce(%v4373 init: %v4378) applies stablehlo.add across dimensions = [0] : (tensor<32x480xf32>, tensor<f32>) -> tensor<480xf32>
    %v4380 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4381 = stablehlo.multiply %v4379, %v4380 : tensor<480xf32>
    %v4382 = stablehlo.subtract %b8zb2, %v4381 : tensor<480xf32>
    %v4383 = stablehlo.reshape %v4373 : (tensor<32x480xf32>) -> tensor<32x1x480xf32>
    %v4384 = stablehlo.dot_general %v4383, %b8zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x480xf32>, tensor<20x480xf32>) -> tensor<32x1x20xf32>
    %v4385 = stablehlo.reshape %v4384 : (tensor<32x1x20xf32>) -> tensor<32x20xf32>
    %v4386 = stablehlo.logistic %v777 : tensor<32x20xf32>
    %v4387 = stablehlo.constant dense<1.0> : tensor<32x20xf32>
    %v4388 = stablehlo.subtract %v4387, %v4386 : tensor<32x20xf32>
    %v4389 = stablehlo.multiply %v777, %v4388 : tensor<32x20xf32>
    %v4390 = stablehlo.add %v4387, %v4389 : tensor<32x20xf32>
    %v4391 = stablehlo.multiply %v4386, %v4390 : tensor<32x20xf32>
    %v4392 = stablehlo.multiply %v4385, %v4391 : tensor<32x20xf32>
    %v4393 = stablehlo.dot_general %v774, %v4392, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<32x20xf32>) -> tensor<480x20xf32>
    %v4394 = stablehlo.constant dense<0.05> : tensor<480x20xf32>
    %v4395 = stablehlo.multiply %v4393, %v4394 : tensor<480x20xf32>
    %v4396 = stablehlo.subtract %b8zW1, %v4395 : tensor<480x20xf32>
    %v4397 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4398 = stablehlo.reduce(%v4392 init: %v4397) applies stablehlo.add across dimensions = [0] : (tensor<32x20xf32>, tensor<f32>) -> tensor<20xf32>
    %v4399 = stablehlo.constant dense<0.05> : tensor<20xf32>
    %v4400 = stablehlo.multiply %v4398, %v4399 : tensor<20xf32>
    %v4401 = stablehlo.subtract %b8zb1, %v4400 : tensor<20xf32>
    %v4402 = stablehlo.reshape %v4363 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4403 = stablehlo.reshape %v765 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4404 = stablehlo.logistic %v4403 : tensor<32x480x14x14xf32>
    %v4405 = stablehlo.constant dense<1.0> : tensor<32x480x14x14xf32>
    %v4406 = stablehlo.subtract %v4405, %v4404 : tensor<32x480x14x14xf32>
    %v4407 = stablehlo.multiply %v4403, %v4406 : tensor<32x480x14x14xf32>
    %v4408 = stablehlo.add %v4405, %v4407 : tensor<32x480x14x14xf32>
    %v4409 = stablehlo.multiply %v4404, %v4408 : tensor<32x480x14x14xf32>
    %v4410 = stablehlo.multiply %v4402, %v4409 : tensor<32x480x14x14xf32>
    %v4411 = stablehlo.reshape %v4410 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4412 = stablehlo.reshape %v745 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4413 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4414 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4415 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4416 = stablehlo.reduce(%v4412 init: %v4413) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4417 = stablehlo.broadcast_in_dim %v4416, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4418 = stablehlo.divide %v4417, %v4414 : tensor<32x480x14x14xf32>
    %v4419 = stablehlo.subtract %v4412, %v4418 : tensor<32x480x14x14xf32>
    %v4420 = stablehlo.multiply %v4419, %v4419 : tensor<32x480x14x14xf32>
    %v4421 = stablehlo.reduce(%v4420 init: %v4413) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4422 = stablehlo.broadcast_in_dim %v4421, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4423 = stablehlo.divide %v4422, %v4414 : tensor<32x480x14x14xf32>
    %v4424 = stablehlo.add %v4423, %v4415 : tensor<32x480x14x14xf32>
    %v4425 = stablehlo.rsqrt %v4424 : tensor<32x480x14x14xf32>
    %v4426 = stablehlo.multiply %v4419, %v4425 : tensor<32x480x14x14xf32>
    %v4427 = stablehlo.broadcast_in_dim %b8dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4428 = stablehlo.reshape %v4411 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4429 = stablehlo.multiply %v4427, %v4428 : tensor<32x480x14x14xf32>
    %v4430 = stablehlo.reduce(%v4429 init: %v4413) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4431 = stablehlo.broadcast_in_dim %v4430, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4432 = stablehlo.multiply %v4426, %v4429 : tensor<32x480x14x14xf32>
    %v4433 = stablehlo.reduce(%v4432 init: %v4413) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4434 = stablehlo.broadcast_in_dim %v4433, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4435 = stablehlo.multiply %v4429, %v4414 : tensor<32x480x14x14xf32>
    %v4436 = stablehlo.subtract %v4435, %v4431 : tensor<32x480x14x14xf32>
    %v4437 = stablehlo.multiply %v4426, %v4434 : tensor<32x480x14x14xf32>
    %v4438 = stablehlo.subtract %v4436, %v4437 : tensor<32x480x14x14xf32>
    %v4439 = stablehlo.divide %v4425, %v4414 : tensor<32x480x14x14xf32>
    %v4440 = stablehlo.multiply %v4439, %v4438 : tensor<32x480x14x14xf32>
    %v4441 = stablehlo.reshape %v4440 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4442 = stablehlo.reshape %v4441 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4443 = stablehlo.reverse %b8dW, dims = [2, 3] : tensor<480x1x3x3xf32>
    %v4444 = stablehlo.convolution(%v4442, %v4443)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %v4445 = stablehlo.reshape %v4444 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4446 = stablehlo.reshape %v745 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4447 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4448 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4449 = stablehlo.reduce(%v4446 init: %v4447) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4450 = stablehlo.broadcast_in_dim %v4449, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4451 = stablehlo.divide %v4450, %v4448 : tensor<32x480x14x14xf32>
    %v4452 = stablehlo.subtract %v4446, %v4451 : tensor<32x480x14x14xf32>
    %v4453 = stablehlo.multiply %v4452, %v4452 : tensor<32x480x14x14xf32>
    %v4454 = stablehlo.reduce(%v4453 init: %v4447) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4455 = stablehlo.broadcast_in_dim %v4454, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4456 = stablehlo.divide %v4455, %v4448 : tensor<32x480x14x14xf32>
    %v4457 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4458 = stablehlo.add %v4456, %v4457 : tensor<32x480x14x14xf32>
    %v4459 = stablehlo.rsqrt %v4458 : tensor<32x480x14x14xf32>
    %v4460 = stablehlo.multiply %v4452, %v4459 : tensor<32x480x14x14xf32>
    %v4461 = stablehlo.reshape %v4411 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4462 = stablehlo.multiply %v4461, %v4460 : tensor<32x480x14x14xf32>
    %v4463 = stablehlo.reduce(%v4462 init: %v4447) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4464 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4465 = stablehlo.multiply %v4463, %v4464 : tensor<480xf32>
    %v4466 = stablehlo.subtract %b8dg, %v4465 : tensor<480xf32>
    %v4467 = stablehlo.reshape %v4411 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4468 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4469 = stablehlo.reduce(%v4467 init: %v4468) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4470 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4471 = stablehlo.multiply %v4469, %v4470 : tensor<480xf32>
    %v4472 = stablehlo.subtract %b8dbt, %v4471 : tensor<480xf32>
    %v4473 = stablehlo.reshape %v740 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4474 = stablehlo.reshape %v4441 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4475 = stablehlo.transpose %v4473, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4476 = stablehlo.transpose %v4474, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4477 = stablehlo.convolution(%v4475, %v4476)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 480 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<1x480x3x3xf32>
    %v4478 = stablehlo.reshape %v4477 : (tensor<1x480x3x3xf32>) -> tensor<480x1x3x3xf32>
    %v4479 = stablehlo.constant dense<0.05> : tensor<480x1x3x3xf32>
    %v4480 = stablehlo.multiply %v4478, %v4479 : tensor<480x1x3x3xf32>
    %v4481 = stablehlo.subtract %b8dW, %v4480 : tensor<480x1x3x3xf32>
    %v4482 = stablehlo.reshape %v4445 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4483 = stablehlo.reshape %v736 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4484 = stablehlo.logistic %v4483 : tensor<32x480x14x14xf32>
    %v4485 = stablehlo.constant dense<1.0> : tensor<32x480x14x14xf32>
    %v4486 = stablehlo.subtract %v4485, %v4484 : tensor<32x480x14x14xf32>
    %v4487 = stablehlo.multiply %v4483, %v4486 : tensor<32x480x14x14xf32>
    %v4488 = stablehlo.add %v4485, %v4487 : tensor<32x480x14x14xf32>
    %v4489 = stablehlo.multiply %v4484, %v4488 : tensor<32x480x14x14xf32>
    %v4490 = stablehlo.multiply %v4482, %v4489 : tensor<32x480x14x14xf32>
    %v4491 = stablehlo.reshape %v4490 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4492 = stablehlo.reshape %v716 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4493 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4494 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4495 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4496 = stablehlo.reduce(%v4492 init: %v4493) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4497 = stablehlo.broadcast_in_dim %v4496, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4498 = stablehlo.divide %v4497, %v4494 : tensor<32x480x14x14xf32>
    %v4499 = stablehlo.subtract %v4492, %v4498 : tensor<32x480x14x14xf32>
    %v4500 = stablehlo.multiply %v4499, %v4499 : tensor<32x480x14x14xf32>
    %v4501 = stablehlo.reduce(%v4500 init: %v4493) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4502 = stablehlo.broadcast_in_dim %v4501, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4503 = stablehlo.divide %v4502, %v4494 : tensor<32x480x14x14xf32>
    %v4504 = stablehlo.add %v4503, %v4495 : tensor<32x480x14x14xf32>
    %v4505 = stablehlo.rsqrt %v4504 : tensor<32x480x14x14xf32>
    %v4506 = stablehlo.multiply %v4499, %v4505 : tensor<32x480x14x14xf32>
    %v4507 = stablehlo.broadcast_in_dim %b8eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4508 = stablehlo.reshape %v4491 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4509 = stablehlo.multiply %v4507, %v4508 : tensor<32x480x14x14xf32>
    %v4510 = stablehlo.reduce(%v4509 init: %v4493) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4511 = stablehlo.broadcast_in_dim %v4510, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4512 = stablehlo.multiply %v4506, %v4509 : tensor<32x480x14x14xf32>
    %v4513 = stablehlo.reduce(%v4512 init: %v4493) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4514 = stablehlo.broadcast_in_dim %v4513, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4515 = stablehlo.multiply %v4509, %v4494 : tensor<32x480x14x14xf32>
    %v4516 = stablehlo.subtract %v4515, %v4511 : tensor<32x480x14x14xf32>
    %v4517 = stablehlo.multiply %v4506, %v4514 : tensor<32x480x14x14xf32>
    %v4518 = stablehlo.subtract %v4516, %v4517 : tensor<32x480x14x14xf32>
    %v4519 = stablehlo.divide %v4505, %v4494 : tensor<32x480x14x14xf32>
    %v4520 = stablehlo.multiply %v4519, %v4518 : tensor<32x480x14x14xf32>
    %v4521 = stablehlo.reshape %v4520 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4522 = stablehlo.reshape %v4521 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4523 = stablehlo.reverse %b8eW, dims = [2, 3] : tensor<480x80x1x1xf32>
    %v4524 = stablehlo.transpose %v4523, dims = [1, 0, 2, 3] : (tensor<480x80x1x1xf32>) -> tensor<80x480x1x1xf32>
    %v4525 = stablehlo.convolution(%v4522, %v4524)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v4526 = stablehlo.reshape %v4525 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v4527 = stablehlo.reshape %v716 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4528 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4529 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4530 = stablehlo.reduce(%v4527 init: %v4528) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4531 = stablehlo.broadcast_in_dim %v4530, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4532 = stablehlo.divide %v4531, %v4529 : tensor<32x480x14x14xf32>
    %v4533 = stablehlo.subtract %v4527, %v4532 : tensor<32x480x14x14xf32>
    %v4534 = stablehlo.multiply %v4533, %v4533 : tensor<32x480x14x14xf32>
    %v4535 = stablehlo.reduce(%v4534 init: %v4528) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4536 = stablehlo.broadcast_in_dim %v4535, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4537 = stablehlo.divide %v4536, %v4529 : tensor<32x480x14x14xf32>
    %v4538 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4539 = stablehlo.add %v4537, %v4538 : tensor<32x480x14x14xf32>
    %v4540 = stablehlo.rsqrt %v4539 : tensor<32x480x14x14xf32>
    %v4541 = stablehlo.multiply %v4533, %v4540 : tensor<32x480x14x14xf32>
    %v4542 = stablehlo.reshape %v4491 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4543 = stablehlo.multiply %v4542, %v4541 : tensor<32x480x14x14xf32>
    %v4544 = stablehlo.reduce(%v4543 init: %v4528) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4545 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4546 = stablehlo.multiply %v4544, %v4545 : tensor<480xf32>
    %v4547 = stablehlo.subtract %b8eg, %v4546 : tensor<480xf32>
    %v4548 = stablehlo.reshape %v4491 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4549 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4550 = stablehlo.reduce(%v4548 init: %v4549) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4551 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4552 = stablehlo.multiply %v4550, %v4551 : tensor<480xf32>
    %v4553 = stablehlo.subtract %b8ebt, %v4552 : tensor<480xf32>
    %v4554 = stablehlo.reshape %v711 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4555 = stablehlo.reshape %v4521 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4556 = stablehlo.transpose %v4554, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %v4557 = stablehlo.transpose %v4555, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4558 = stablehlo.convolution(%v4556, %v4557)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<80x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<80x480x1x1xf32>
    %v4559 = stablehlo.transpose %v4558, dims = [1, 0, 2, 3] : (tensor<80x480x1x1xf32>) -> tensor<480x80x1x1xf32>
    %v4560 = stablehlo.constant dense<0.05> : tensor<480x80x1x1xf32>
    %v4561 = stablehlo.multiply %v4559, %v4560 : tensor<480x80x1x1xf32>
    %v4562 = stablehlo.subtract %b8eW, %v4561 : tensor<480x80x1x1xf32>
    %v4563 = stablehlo.reshape %v4526 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4564 = stablehlo.reshape %v4219 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4565 = stablehlo.add %v4563, %v4564 : tensor<32x80x14x14xf32>
    %v4566 = stablehlo.reshape %v4565 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v4567 = stablehlo.reshape %v687 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4568 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4569 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v4570 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v4571 = stablehlo.reduce(%v4567 init: %v4568) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4572 = stablehlo.broadcast_in_dim %v4571, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4573 = stablehlo.divide %v4572, %v4569 : tensor<32x80x14x14xf32>
    %v4574 = stablehlo.subtract %v4567, %v4573 : tensor<32x80x14x14xf32>
    %v4575 = stablehlo.multiply %v4574, %v4574 : tensor<32x80x14x14xf32>
    %v4576 = stablehlo.reduce(%v4575 init: %v4568) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4577 = stablehlo.broadcast_in_dim %v4576, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4578 = stablehlo.divide %v4577, %v4569 : tensor<32x80x14x14xf32>
    %v4579 = stablehlo.add %v4578, %v4570 : tensor<32x80x14x14xf32>
    %v4580 = stablehlo.rsqrt %v4579 : tensor<32x80x14x14xf32>
    %v4581 = stablehlo.multiply %v4574, %v4580 : tensor<32x80x14x14xf32>
    %v4582 = stablehlo.broadcast_in_dim %b7pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4583 = stablehlo.reshape %v4566 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4584 = stablehlo.multiply %v4582, %v4583 : tensor<32x80x14x14xf32>
    %v4585 = stablehlo.reduce(%v4584 init: %v4568) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4586 = stablehlo.broadcast_in_dim %v4585, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4587 = stablehlo.multiply %v4581, %v4584 : tensor<32x80x14x14xf32>
    %v4588 = stablehlo.reduce(%v4587 init: %v4568) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4589 = stablehlo.broadcast_in_dim %v4588, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4590 = stablehlo.multiply %v4584, %v4569 : tensor<32x80x14x14xf32>
    %v4591 = stablehlo.subtract %v4590, %v4586 : tensor<32x80x14x14xf32>
    %v4592 = stablehlo.multiply %v4581, %v4589 : tensor<32x80x14x14xf32>
    %v4593 = stablehlo.subtract %v4591, %v4592 : tensor<32x80x14x14xf32>
    %v4594 = stablehlo.divide %v4580, %v4569 : tensor<32x80x14x14xf32>
    %v4595 = stablehlo.multiply %v4594, %v4593 : tensor<32x80x14x14xf32>
    %v4596 = stablehlo.reshape %v4595 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v4597 = stablehlo.reshape %v4596 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4598 = stablehlo.reverse %b7pW, dims = [2, 3] : tensor<80x480x1x1xf32>
    %v4599 = stablehlo.transpose %v4598, dims = [1, 0, 2, 3] : (tensor<80x480x1x1xf32>) -> tensor<480x80x1x1xf32>
    %v4600 = stablehlo.convolution(%v4597, %v4599)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v4601 = stablehlo.reshape %v4600 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4602 = stablehlo.reshape %v687 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4603 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4604 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v4605 = stablehlo.reduce(%v4602 init: %v4603) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4606 = stablehlo.broadcast_in_dim %v4605, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4607 = stablehlo.divide %v4606, %v4604 : tensor<32x80x14x14xf32>
    %v4608 = stablehlo.subtract %v4602, %v4607 : tensor<32x80x14x14xf32>
    %v4609 = stablehlo.multiply %v4608, %v4608 : tensor<32x80x14x14xf32>
    %v4610 = stablehlo.reduce(%v4609 init: %v4603) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4611 = stablehlo.broadcast_in_dim %v4610, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4612 = stablehlo.divide %v4611, %v4604 : tensor<32x80x14x14xf32>
    %v4613 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v4614 = stablehlo.add %v4612, %v4613 : tensor<32x80x14x14xf32>
    %v4615 = stablehlo.rsqrt %v4614 : tensor<32x80x14x14xf32>
    %v4616 = stablehlo.multiply %v4608, %v4615 : tensor<32x80x14x14xf32>
    %v4617 = stablehlo.reshape %v4566 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4618 = stablehlo.multiply %v4617, %v4616 : tensor<32x80x14x14xf32>
    %v4619 = stablehlo.reduce(%v4618 init: %v4603) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4620 = stablehlo.constant dense<0.05> : tensor<80xf32>
    %v4621 = stablehlo.multiply %v4619, %v4620 : tensor<80xf32>
    %v4622 = stablehlo.subtract %b7pg, %v4621 : tensor<80xf32>
    %v4623 = stablehlo.reshape %v4566 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4624 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4625 = stablehlo.reduce(%v4623 init: %v4624) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4626 = stablehlo.constant dense<0.05> : tensor<80xf32>
    %v4627 = stablehlo.multiply %v4625, %v4626 : tensor<80xf32>
    %v4628 = stablehlo.subtract %b7pbt, %v4627 : tensor<80xf32>
    %v4629 = stablehlo.reshape %v682 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4630 = stablehlo.reshape %v4596 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4631 = stablehlo.transpose %v4629, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4632 = stablehlo.transpose %v4630, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %v4633 = stablehlo.convolution(%v4631, %v4632)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<80x32x14x14xf32>) -> tensor<480x80x1x1xf32>
    %v4634 = stablehlo.transpose %v4633, dims = [1, 0, 2, 3] : (tensor<480x80x1x1xf32>) -> tensor<80x480x1x1xf32>
    %v4635 = stablehlo.constant dense<0.05> : tensor<80x480x1x1xf32>
    %v4636 = stablehlo.multiply %v4634, %v4635 : tensor<80x480x1x1xf32>
    %v4637 = stablehlo.subtract %b7pW, %v4636 : tensor<80x480x1x1xf32>
    %v4638 = stablehlo.reshape %v665 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4639 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4640 = stablehlo.reduce(%v4638 init: %v4639) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4641 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v4642 = stablehlo.divide %v4640, %v4641 : tensor<32x480xf32>
    %v4643 = stablehlo.dot_general %v4642, %b7zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v4644 = stablehlo.broadcast_in_dim %b7zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v4645 = stablehlo.add %v4643, %v4644 : tensor<32x20xf32>
    %v4646 = stablehlo.logistic %v4645 : tensor<32x20xf32>
    %v4647 = stablehlo.multiply %v4645, %v4646 : tensor<32x20xf32>
    %v4648 = stablehlo.dot_general %v4647, %b7zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v4649 = stablehlo.broadcast_in_dim %b7zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v4650 = stablehlo.add %v4648, %v4649 : tensor<32x480xf32>
    %v4651 = stablehlo.logistic %v4650 : tensor<32x480xf32>
    %v4652 = stablehlo.reshape %v4601 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4653 = stablehlo.broadcast_in_dim %v4651, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v4654 = stablehlo.multiply %v4653, %v4652 : tensor<32x480x14x14xf32>
    %v4655 = stablehlo.multiply %v4638, %v4652 : tensor<32x480x14x14xf32>
    %v4656 = stablehlo.reduce(%v4655 init: %v4639) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4657 = stablehlo.constant dense<1.0> : tensor<32x480xf32>
    %v4658 = stablehlo.subtract %v4657, %v4651 : tensor<32x480xf32>
    %v4659 = stablehlo.multiply %v4651, %v4658 : tensor<32x480xf32>
    %v4660 = stablehlo.multiply %v4656, %v4659 : tensor<32x480xf32>
    %v4661 = stablehlo.dot_general %v4660, %b7zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<20x480xf32>) -> tensor<32x20xf32>
    %v4662 = stablehlo.logistic %v4645 : tensor<32x20xf32>
    %v4663 = stablehlo.constant dense<1.0> : tensor<32x20xf32>
    %v4664 = stablehlo.subtract %v4663, %v4662 : tensor<32x20xf32>
    %v4665 = stablehlo.multiply %v4645, %v4664 : tensor<32x20xf32>
    %v4666 = stablehlo.add %v4663, %v4665 : tensor<32x20xf32>
    %v4667 = stablehlo.multiply %v4662, %v4666 : tensor<32x20xf32>
    %v4668 = stablehlo.multiply %v4661, %v4667 : tensor<32x20xf32>
    %v4669 = stablehlo.dot_general %v4668, %b7zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<480x20xf32>) -> tensor<32x480xf32>
    %v4670 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v4671 = stablehlo.divide %v4669, %v4670 : tensor<32x480xf32>
    %v4672 = stablehlo.broadcast_in_dim %v4671, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v4673 = stablehlo.add %v4654, %v4672 : tensor<32x480x14x14xf32>
    %v4674 = stablehlo.reshape %v4673 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4675 = stablehlo.reshape %v665 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4676 = stablehlo.reshape %v4601 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4677 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4678 = stablehlo.multiply %v4675, %v4676 : tensor<32x480x14x14xf32>
    %v4679 = stablehlo.reduce(%v4678 init: %v4677) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4680 = stablehlo.logistic %v678 : tensor<32x480xf32>
    %v4681 = stablehlo.constant dense<1.0> : tensor<32x480xf32>
    %v4682 = stablehlo.subtract %v4681, %v4680 : tensor<32x480xf32>
    %v4683 = stablehlo.multiply %v4680, %v4682 : tensor<32x480xf32>
    %v4684 = stablehlo.multiply %v4679, %v4683 : tensor<32x480xf32>
    %v4685 = stablehlo.dot_general %v675, %v4684, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<32x480xf32>) -> tensor<20x480xf32>
    %v4686 = stablehlo.constant dense<0.05> : tensor<20x480xf32>
    %v4687 = stablehlo.multiply %v4685, %v4686 : tensor<20x480xf32>
    %v4688 = stablehlo.subtract %b7zW2, %v4687 : tensor<20x480xf32>
    %v4689 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4690 = stablehlo.reduce(%v4684 init: %v4689) applies stablehlo.add across dimensions = [0] : (tensor<32x480xf32>, tensor<f32>) -> tensor<480xf32>
    %v4691 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4692 = stablehlo.multiply %v4690, %v4691 : tensor<480xf32>
    %v4693 = stablehlo.subtract %b7zb2, %v4692 : tensor<480xf32>
    %v4694 = stablehlo.reshape %v4684 : (tensor<32x480xf32>) -> tensor<32x1x480xf32>
    %v4695 = stablehlo.dot_general %v4694, %b7zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x480xf32>, tensor<20x480xf32>) -> tensor<32x1x20xf32>
    %v4696 = stablehlo.reshape %v4695 : (tensor<32x1x20xf32>) -> tensor<32x20xf32>
    %v4697 = stablehlo.logistic %v673 : tensor<32x20xf32>
    %v4698 = stablehlo.constant dense<1.0> : tensor<32x20xf32>
    %v4699 = stablehlo.subtract %v4698, %v4697 : tensor<32x20xf32>
    %v4700 = stablehlo.multiply %v673, %v4699 : tensor<32x20xf32>
    %v4701 = stablehlo.add %v4698, %v4700 : tensor<32x20xf32>
    %v4702 = stablehlo.multiply %v4697, %v4701 : tensor<32x20xf32>
    %v4703 = stablehlo.multiply %v4696, %v4702 : tensor<32x20xf32>
    %v4704 = stablehlo.dot_general %v670, %v4703, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<32x20xf32>) -> tensor<480x20xf32>
    %v4705 = stablehlo.constant dense<0.05> : tensor<480x20xf32>
    %v4706 = stablehlo.multiply %v4704, %v4705 : tensor<480x20xf32>
    %v4707 = stablehlo.subtract %b7zW1, %v4706 : tensor<480x20xf32>
    %v4708 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4709 = stablehlo.reduce(%v4703 init: %v4708) applies stablehlo.add across dimensions = [0] : (tensor<32x20xf32>, tensor<f32>) -> tensor<20xf32>
    %v4710 = stablehlo.constant dense<0.05> : tensor<20xf32>
    %v4711 = stablehlo.multiply %v4709, %v4710 : tensor<20xf32>
    %v4712 = stablehlo.subtract %b7zb1, %v4711 : tensor<20xf32>
    %v4713 = stablehlo.reshape %v4674 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4714 = stablehlo.reshape %v661 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4715 = stablehlo.logistic %v4714 : tensor<32x480x14x14xf32>
    %v4716 = stablehlo.constant dense<1.0> : tensor<32x480x14x14xf32>
    %v4717 = stablehlo.subtract %v4716, %v4715 : tensor<32x480x14x14xf32>
    %v4718 = stablehlo.multiply %v4714, %v4717 : tensor<32x480x14x14xf32>
    %v4719 = stablehlo.add %v4716, %v4718 : tensor<32x480x14x14xf32>
    %v4720 = stablehlo.multiply %v4715, %v4719 : tensor<32x480x14x14xf32>
    %v4721 = stablehlo.multiply %v4713, %v4720 : tensor<32x480x14x14xf32>
    %v4722 = stablehlo.reshape %v4721 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4723 = stablehlo.reshape %v641 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4724 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4725 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4726 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4727 = stablehlo.reduce(%v4723 init: %v4724) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4728 = stablehlo.broadcast_in_dim %v4727, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4729 = stablehlo.divide %v4728, %v4725 : tensor<32x480x14x14xf32>
    %v4730 = stablehlo.subtract %v4723, %v4729 : tensor<32x480x14x14xf32>
    %v4731 = stablehlo.multiply %v4730, %v4730 : tensor<32x480x14x14xf32>
    %v4732 = stablehlo.reduce(%v4731 init: %v4724) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4733 = stablehlo.broadcast_in_dim %v4732, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4734 = stablehlo.divide %v4733, %v4725 : tensor<32x480x14x14xf32>
    %v4735 = stablehlo.add %v4734, %v4726 : tensor<32x480x14x14xf32>
    %v4736 = stablehlo.rsqrt %v4735 : tensor<32x480x14x14xf32>
    %v4737 = stablehlo.multiply %v4730, %v4736 : tensor<32x480x14x14xf32>
    %v4738 = stablehlo.broadcast_in_dim %b7dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4739 = stablehlo.reshape %v4722 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4740 = stablehlo.multiply %v4738, %v4739 : tensor<32x480x14x14xf32>
    %v4741 = stablehlo.reduce(%v4740 init: %v4724) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4742 = stablehlo.broadcast_in_dim %v4741, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4743 = stablehlo.multiply %v4737, %v4740 : tensor<32x480x14x14xf32>
    %v4744 = stablehlo.reduce(%v4743 init: %v4724) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4745 = stablehlo.broadcast_in_dim %v4744, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4746 = stablehlo.multiply %v4740, %v4725 : tensor<32x480x14x14xf32>
    %v4747 = stablehlo.subtract %v4746, %v4742 : tensor<32x480x14x14xf32>
    %v4748 = stablehlo.multiply %v4737, %v4745 : tensor<32x480x14x14xf32>
    %v4749 = stablehlo.subtract %v4747, %v4748 : tensor<32x480x14x14xf32>
    %v4750 = stablehlo.divide %v4736, %v4725 : tensor<32x480x14x14xf32>
    %v4751 = stablehlo.multiply %v4750, %v4749 : tensor<32x480x14x14xf32>
    %v4752 = stablehlo.reshape %v4751 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4753 = stablehlo.reshape %v4752 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4754 = stablehlo.reverse %b7dW, dims = [2, 3] : tensor<480x1x3x3xf32>
    %v4755 = stablehlo.convolution(%v4753, %v4754)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %v4756 = stablehlo.reshape %v4755 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4757 = stablehlo.reshape %v641 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4758 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4759 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4760 = stablehlo.reduce(%v4757 init: %v4758) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4761 = stablehlo.broadcast_in_dim %v4760, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4762 = stablehlo.divide %v4761, %v4759 : tensor<32x480x14x14xf32>
    %v4763 = stablehlo.subtract %v4757, %v4762 : tensor<32x480x14x14xf32>
    %v4764 = stablehlo.multiply %v4763, %v4763 : tensor<32x480x14x14xf32>
    %v4765 = stablehlo.reduce(%v4764 init: %v4758) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4766 = stablehlo.broadcast_in_dim %v4765, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4767 = stablehlo.divide %v4766, %v4759 : tensor<32x480x14x14xf32>
    %v4768 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4769 = stablehlo.add %v4767, %v4768 : tensor<32x480x14x14xf32>
    %v4770 = stablehlo.rsqrt %v4769 : tensor<32x480x14x14xf32>
    %v4771 = stablehlo.multiply %v4763, %v4770 : tensor<32x480x14x14xf32>
    %v4772 = stablehlo.reshape %v4722 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4773 = stablehlo.multiply %v4772, %v4771 : tensor<32x480x14x14xf32>
    %v4774 = stablehlo.reduce(%v4773 init: %v4758) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4775 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4776 = stablehlo.multiply %v4774, %v4775 : tensor<480xf32>
    %v4777 = stablehlo.subtract %b7dg, %v4776 : tensor<480xf32>
    %v4778 = stablehlo.reshape %v4722 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4779 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4780 = stablehlo.reduce(%v4778 init: %v4779) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4781 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4782 = stablehlo.multiply %v4780, %v4781 : tensor<480xf32>
    %v4783 = stablehlo.subtract %b7dbt, %v4782 : tensor<480xf32>
    %v4784 = stablehlo.reshape %v636 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4785 = stablehlo.reshape %v4752 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4786 = stablehlo.transpose %v4784, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4787 = stablehlo.transpose %v4785, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4788 = stablehlo.convolution(%v4786, %v4787)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 480 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<1x480x3x3xf32>
    %v4789 = stablehlo.reshape %v4788 : (tensor<1x480x3x3xf32>) -> tensor<480x1x3x3xf32>
    %v4790 = stablehlo.constant dense<0.05> : tensor<480x1x3x3xf32>
    %v4791 = stablehlo.multiply %v4789, %v4790 : tensor<480x1x3x3xf32>
    %v4792 = stablehlo.subtract %b7dW, %v4791 : tensor<480x1x3x3xf32>
    %v4793 = stablehlo.reshape %v4756 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4794 = stablehlo.reshape %v632 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4795 = stablehlo.logistic %v4794 : tensor<32x480x14x14xf32>
    %v4796 = stablehlo.constant dense<1.0> : tensor<32x480x14x14xf32>
    %v4797 = stablehlo.subtract %v4796, %v4795 : tensor<32x480x14x14xf32>
    %v4798 = stablehlo.multiply %v4794, %v4797 : tensor<32x480x14x14xf32>
    %v4799 = stablehlo.add %v4796, %v4798 : tensor<32x480x14x14xf32>
    %v4800 = stablehlo.multiply %v4795, %v4799 : tensor<32x480x14x14xf32>
    %v4801 = stablehlo.multiply %v4793, %v4800 : tensor<32x480x14x14xf32>
    %v4802 = stablehlo.reshape %v4801 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4803 = stablehlo.reshape %v612 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4804 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4805 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4806 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4807 = stablehlo.reduce(%v4803 init: %v4804) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4808 = stablehlo.broadcast_in_dim %v4807, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4809 = stablehlo.divide %v4808, %v4805 : tensor<32x480x14x14xf32>
    %v4810 = stablehlo.subtract %v4803, %v4809 : tensor<32x480x14x14xf32>
    %v4811 = stablehlo.multiply %v4810, %v4810 : tensor<32x480x14x14xf32>
    %v4812 = stablehlo.reduce(%v4811 init: %v4804) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4813 = stablehlo.broadcast_in_dim %v4812, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4814 = stablehlo.divide %v4813, %v4805 : tensor<32x480x14x14xf32>
    %v4815 = stablehlo.add %v4814, %v4806 : tensor<32x480x14x14xf32>
    %v4816 = stablehlo.rsqrt %v4815 : tensor<32x480x14x14xf32>
    %v4817 = stablehlo.multiply %v4810, %v4816 : tensor<32x480x14x14xf32>
    %v4818 = stablehlo.broadcast_in_dim %b7eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4819 = stablehlo.reshape %v4802 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4820 = stablehlo.multiply %v4818, %v4819 : tensor<32x480x14x14xf32>
    %v4821 = stablehlo.reduce(%v4820 init: %v4804) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4822 = stablehlo.broadcast_in_dim %v4821, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4823 = stablehlo.multiply %v4817, %v4820 : tensor<32x480x14x14xf32>
    %v4824 = stablehlo.reduce(%v4823 init: %v4804) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4825 = stablehlo.broadcast_in_dim %v4824, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4826 = stablehlo.multiply %v4820, %v4805 : tensor<32x480x14x14xf32>
    %v4827 = stablehlo.subtract %v4826, %v4822 : tensor<32x480x14x14xf32>
    %v4828 = stablehlo.multiply %v4817, %v4825 : tensor<32x480x14x14xf32>
    %v4829 = stablehlo.subtract %v4827, %v4828 : tensor<32x480x14x14xf32>
    %v4830 = stablehlo.divide %v4816, %v4805 : tensor<32x480x14x14xf32>
    %v4831 = stablehlo.multiply %v4830, %v4829 : tensor<32x480x14x14xf32>
    %v4832 = stablehlo.reshape %v4831 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4833 = stablehlo.reshape %v4832 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4834 = stablehlo.reverse %b7eW, dims = [2, 3] : tensor<480x80x1x1xf32>
    %v4835 = stablehlo.transpose %v4834, dims = [1, 0, 2, 3] : (tensor<480x80x1x1xf32>) -> tensor<80x480x1x1xf32>
    %v4836 = stablehlo.convolution(%v4833, %v4835)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v4837 = stablehlo.reshape %v4836 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v4838 = stablehlo.reshape %v612 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4839 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4840 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4841 = stablehlo.reduce(%v4838 init: %v4839) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4842 = stablehlo.broadcast_in_dim %v4841, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4843 = stablehlo.divide %v4842, %v4840 : tensor<32x480x14x14xf32>
    %v4844 = stablehlo.subtract %v4838, %v4843 : tensor<32x480x14x14xf32>
    %v4845 = stablehlo.multiply %v4844, %v4844 : tensor<32x480x14x14xf32>
    %v4846 = stablehlo.reduce(%v4845 init: %v4839) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4847 = stablehlo.broadcast_in_dim %v4846, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4848 = stablehlo.divide %v4847, %v4840 : tensor<32x480x14x14xf32>
    %v4849 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4850 = stablehlo.add %v4848, %v4849 : tensor<32x480x14x14xf32>
    %v4851 = stablehlo.rsqrt %v4850 : tensor<32x480x14x14xf32>
    %v4852 = stablehlo.multiply %v4844, %v4851 : tensor<32x480x14x14xf32>
    %v4853 = stablehlo.reshape %v4802 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4854 = stablehlo.multiply %v4853, %v4852 : tensor<32x480x14x14xf32>
    %v4855 = stablehlo.reduce(%v4854 init: %v4839) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4856 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4857 = stablehlo.multiply %v4855, %v4856 : tensor<480xf32>
    %v4858 = stablehlo.subtract %b7eg, %v4857 : tensor<480xf32>
    %v4859 = stablehlo.reshape %v4802 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4860 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4861 = stablehlo.reduce(%v4859 init: %v4860) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4862 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4863 = stablehlo.multiply %v4861, %v4862 : tensor<480xf32>
    %v4864 = stablehlo.subtract %b7ebt, %v4863 : tensor<480xf32>
    %v4865 = stablehlo.reshape %v607 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4866 = stablehlo.reshape %v4832 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4867 = stablehlo.transpose %v4865, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %v4868 = stablehlo.transpose %v4866, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4869 = stablehlo.convolution(%v4867, %v4868)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<80x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<80x480x1x1xf32>
    %v4870 = stablehlo.transpose %v4869, dims = [1, 0, 2, 3] : (tensor<80x480x1x1xf32>) -> tensor<480x80x1x1xf32>
    %v4871 = stablehlo.constant dense<0.05> : tensor<480x80x1x1xf32>
    %v4872 = stablehlo.multiply %v4870, %v4871 : tensor<480x80x1x1xf32>
    %v4873 = stablehlo.subtract %b7eW, %v4872 : tensor<480x80x1x1xf32>
    %v4874 = stablehlo.reshape %v4837 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4875 = stablehlo.reshape %v4566 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4876 = stablehlo.add %v4874, %v4875 : tensor<32x80x14x14xf32>
    %v4877 = stablehlo.reshape %v4876 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v4878 = stablehlo.reshape %v587 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4879 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4880 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v4881 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v4882 = stablehlo.reduce(%v4878 init: %v4879) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4883 = stablehlo.broadcast_in_dim %v4882, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4884 = stablehlo.divide %v4883, %v4880 : tensor<32x80x14x14xf32>
    %v4885 = stablehlo.subtract %v4878, %v4884 : tensor<32x80x14x14xf32>
    %v4886 = stablehlo.multiply %v4885, %v4885 : tensor<32x80x14x14xf32>
    %v4887 = stablehlo.reduce(%v4886 init: %v4879) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4888 = stablehlo.broadcast_in_dim %v4887, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4889 = stablehlo.divide %v4888, %v4880 : tensor<32x80x14x14xf32>
    %v4890 = stablehlo.add %v4889, %v4881 : tensor<32x80x14x14xf32>
    %v4891 = stablehlo.rsqrt %v4890 : tensor<32x80x14x14xf32>
    %v4892 = stablehlo.multiply %v4885, %v4891 : tensor<32x80x14x14xf32>
    %v4893 = stablehlo.broadcast_in_dim %b6pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4894 = stablehlo.reshape %v4877 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4895 = stablehlo.multiply %v4893, %v4894 : tensor<32x80x14x14xf32>
    %v4896 = stablehlo.reduce(%v4895 init: %v4879) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4897 = stablehlo.broadcast_in_dim %v4896, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4898 = stablehlo.multiply %v4892, %v4895 : tensor<32x80x14x14xf32>
    %v4899 = stablehlo.reduce(%v4898 init: %v4879) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4900 = stablehlo.broadcast_in_dim %v4899, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4901 = stablehlo.multiply %v4895, %v4880 : tensor<32x80x14x14xf32>
    %v4902 = stablehlo.subtract %v4901, %v4897 : tensor<32x80x14x14xf32>
    %v4903 = stablehlo.multiply %v4892, %v4900 : tensor<32x80x14x14xf32>
    %v4904 = stablehlo.subtract %v4902, %v4903 : tensor<32x80x14x14xf32>
    %v4905 = stablehlo.divide %v4891, %v4880 : tensor<32x80x14x14xf32>
    %v4906 = stablehlo.multiply %v4905, %v4904 : tensor<32x80x14x14xf32>
    %v4907 = stablehlo.reshape %v4906 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v4908 = stablehlo.reshape %v4907 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4909 = stablehlo.reverse %b6pW, dims = [2, 3] : tensor<80x240x1x1xf32>
    %v4910 = stablehlo.transpose %v4909, dims = [1, 0, 2, 3] : (tensor<80x240x1x1xf32>) -> tensor<240x80x1x1xf32>
    %v4911 = stablehlo.convolution(%v4908, %v4910)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<240x80x1x1xf32>) -> tensor<32x240x14x14xf32>
    %v4912 = stablehlo.reshape %v4911 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v4913 = stablehlo.reshape %v587 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4914 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4915 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v4916 = stablehlo.reduce(%v4913 init: %v4914) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4917 = stablehlo.broadcast_in_dim %v4916, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4918 = stablehlo.divide %v4917, %v4915 : tensor<32x80x14x14xf32>
    %v4919 = stablehlo.subtract %v4913, %v4918 : tensor<32x80x14x14xf32>
    %v4920 = stablehlo.multiply %v4919, %v4919 : tensor<32x80x14x14xf32>
    %v4921 = stablehlo.reduce(%v4920 init: %v4914) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4922 = stablehlo.broadcast_in_dim %v4921, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4923 = stablehlo.divide %v4922, %v4915 : tensor<32x80x14x14xf32>
    %v4924 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v4925 = stablehlo.add %v4923, %v4924 : tensor<32x80x14x14xf32>
    %v4926 = stablehlo.rsqrt %v4925 : tensor<32x80x14x14xf32>
    %v4927 = stablehlo.multiply %v4919, %v4926 : tensor<32x80x14x14xf32>
    %v4928 = stablehlo.reshape %v4877 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4929 = stablehlo.multiply %v4928, %v4927 : tensor<32x80x14x14xf32>
    %v4930 = stablehlo.reduce(%v4929 init: %v4914) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4931 = stablehlo.constant dense<0.05> : tensor<80xf32>
    %v4932 = stablehlo.multiply %v4930, %v4931 : tensor<80xf32>
    %v4933 = stablehlo.subtract %b6pg, %v4932 : tensor<80xf32>
    %v4934 = stablehlo.reshape %v4877 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4935 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4936 = stablehlo.reduce(%v4934 init: %v4935) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4937 = stablehlo.constant dense<0.05> : tensor<80xf32>
    %v4938 = stablehlo.multiply %v4936, %v4937 : tensor<80xf32>
    %v4939 = stablehlo.subtract %b6pbt, %v4938 : tensor<80xf32>
    %v4940 = stablehlo.reshape %v582 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v4941 = stablehlo.reshape %v4907 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4942 = stablehlo.transpose %v4940, dims = [1, 0, 2, 3] : (tensor<32x240x14x14xf32>) -> tensor<240x32x14x14xf32>
    %v4943 = stablehlo.transpose %v4941, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %v4944 = stablehlo.convolution(%v4942, %v4943)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<240x32x14x14xf32>, tensor<80x32x14x14xf32>) -> tensor<240x80x1x1xf32>
    %v4945 = stablehlo.transpose %v4944, dims = [1, 0, 2, 3] : (tensor<240x80x1x1xf32>) -> tensor<80x240x1x1xf32>
    %v4946 = stablehlo.constant dense<0.05> : tensor<80x240x1x1xf32>
    %v4947 = stablehlo.multiply %v4945, %v4946 : tensor<80x240x1x1xf32>
    %v4948 = stablehlo.subtract %b6pW, %v4947 : tensor<80x240x1x1xf32>
    %v4949 = stablehlo.reshape %v565 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v4950 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4951 = stablehlo.reduce(%v4949 init: %v4950) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v4952 = stablehlo.constant dense<196.0> : tensor<32x240xf32>
    %v4953 = stablehlo.divide %v4951, %v4952 : tensor<32x240xf32>
    %v4954 = stablehlo.dot_general %v4953, %b6zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v4955 = stablehlo.broadcast_in_dim %b6zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v4956 = stablehlo.add %v4954, %v4955 : tensor<32x10xf32>
    %v4957 = stablehlo.logistic %v4956 : tensor<32x10xf32>
    %v4958 = stablehlo.multiply %v4956, %v4957 : tensor<32x10xf32>
    %v4959 = stablehlo.dot_general %v4958, %b6zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v4960 = stablehlo.broadcast_in_dim %b6zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v4961 = stablehlo.add %v4959, %v4960 : tensor<32x240xf32>
    %v4962 = stablehlo.logistic %v4961 : tensor<32x240xf32>
    %v4963 = stablehlo.reshape %v4912 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v4964 = stablehlo.broadcast_in_dim %v4962, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x14x14xf32>
    %v4965 = stablehlo.multiply %v4964, %v4963 : tensor<32x240x14x14xf32>
    %v4966 = stablehlo.multiply %v4949, %v4963 : tensor<32x240x14x14xf32>
    %v4967 = stablehlo.reduce(%v4966 init: %v4950) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v4968 = stablehlo.constant dense<1.0> : tensor<32x240xf32>
    %v4969 = stablehlo.subtract %v4968, %v4962 : tensor<32x240xf32>
    %v4970 = stablehlo.multiply %v4962, %v4969 : tensor<32x240xf32>
    %v4971 = stablehlo.multiply %v4967, %v4970 : tensor<32x240xf32>
    %v4972 = stablehlo.dot_general %v4971, %b6zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<10x240xf32>) -> tensor<32x10xf32>
    %v4973 = stablehlo.logistic %v4956 : tensor<32x10xf32>
    %v4974 = stablehlo.constant dense<1.0> : tensor<32x10xf32>
    %v4975 = stablehlo.subtract %v4974, %v4973 : tensor<32x10xf32>
    %v4976 = stablehlo.multiply %v4956, %v4975 : tensor<32x10xf32>
    %v4977 = stablehlo.add %v4974, %v4976 : tensor<32x10xf32>
    %v4978 = stablehlo.multiply %v4973, %v4977 : tensor<32x10xf32>
    %v4979 = stablehlo.multiply %v4972, %v4978 : tensor<32x10xf32>
    %v4980 = stablehlo.dot_general %v4979, %b6zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<240x10xf32>) -> tensor<32x240xf32>
    %v4981 = stablehlo.constant dense<196.0> : tensor<32x240xf32>
    %v4982 = stablehlo.divide %v4980, %v4981 : tensor<32x240xf32>
    %v4983 = stablehlo.broadcast_in_dim %v4982, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x14x14xf32>
    %v4984 = stablehlo.add %v4965, %v4983 : tensor<32x240x14x14xf32>
    %v4985 = stablehlo.reshape %v4984 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v4986 = stablehlo.reshape %v565 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v4987 = stablehlo.reshape %v4912 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v4988 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4989 = stablehlo.multiply %v4986, %v4987 : tensor<32x240x14x14xf32>
    %v4990 = stablehlo.reduce(%v4989 init: %v4988) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v4991 = stablehlo.logistic %v578 : tensor<32x240xf32>
    %v4992 = stablehlo.constant dense<1.0> : tensor<32x240xf32>
    %v4993 = stablehlo.subtract %v4992, %v4991 : tensor<32x240xf32>
    %v4994 = stablehlo.multiply %v4991, %v4993 : tensor<32x240xf32>
    %v4995 = stablehlo.multiply %v4990, %v4994 : tensor<32x240xf32>
    %v4996 = stablehlo.dot_general %v575, %v4995, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<32x240xf32>) -> tensor<10x240xf32>
    %v4997 = stablehlo.constant dense<0.05> : tensor<10x240xf32>
    %v4998 = stablehlo.multiply %v4996, %v4997 : tensor<10x240xf32>
    %v4999 = stablehlo.subtract %b6zW2, %v4998 : tensor<10x240xf32>
    %v5000 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5001 = stablehlo.reduce(%v4995 init: %v5000) applies stablehlo.add across dimensions = [0] : (tensor<32x240xf32>, tensor<f32>) -> tensor<240xf32>
    %v5002 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5003 = stablehlo.multiply %v5001, %v5002 : tensor<240xf32>
    %v5004 = stablehlo.subtract %b6zb2, %v5003 : tensor<240xf32>
    %v5005 = stablehlo.reshape %v4995 : (tensor<32x240xf32>) -> tensor<32x1x240xf32>
    %v5006 = stablehlo.dot_general %v5005, %b6zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x240xf32>, tensor<10x240xf32>) -> tensor<32x1x10xf32>
    %v5007 = stablehlo.reshape %v5006 : (tensor<32x1x10xf32>) -> tensor<32x10xf32>
    %v5008 = stablehlo.logistic %v573 : tensor<32x10xf32>
    %v5009 = stablehlo.constant dense<1.0> : tensor<32x10xf32>
    %v5010 = stablehlo.subtract %v5009, %v5008 : tensor<32x10xf32>
    %v5011 = stablehlo.multiply %v573, %v5010 : tensor<32x10xf32>
    %v5012 = stablehlo.add %v5009, %v5011 : tensor<32x10xf32>
    %v5013 = stablehlo.multiply %v5008, %v5012 : tensor<32x10xf32>
    %v5014 = stablehlo.multiply %v5007, %v5013 : tensor<32x10xf32>
    %v5015 = stablehlo.dot_general %v570, %v5014, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<32x10xf32>) -> tensor<240x10xf32>
    %v5016 = stablehlo.constant dense<0.05> : tensor<240x10xf32>
    %v5017 = stablehlo.multiply %v5015, %v5016 : tensor<240x10xf32>
    %v5018 = stablehlo.subtract %b6zW1, %v5017 : tensor<240x10xf32>
    %v5019 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5020 = stablehlo.reduce(%v5014 init: %v5019) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v5021 = stablehlo.constant dense<0.05> : tensor<10xf32>
    %v5022 = stablehlo.multiply %v5020, %v5021 : tensor<10xf32>
    %v5023 = stablehlo.subtract %b6zb1, %v5022 : tensor<10xf32>
    %v5024 = stablehlo.reshape %v4985 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5025 = stablehlo.reshape %v561 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5026 = stablehlo.logistic %v5025 : tensor<32x240x14x14xf32>
    %v5027 = stablehlo.constant dense<1.0> : tensor<32x240x14x14xf32>
    %v5028 = stablehlo.subtract %v5027, %v5026 : tensor<32x240x14x14xf32>
    %v5029 = stablehlo.multiply %v5025, %v5028 : tensor<32x240x14x14xf32>
    %v5030 = stablehlo.add %v5027, %v5029 : tensor<32x240x14x14xf32>
    %v5031 = stablehlo.multiply %v5026, %v5030 : tensor<32x240x14x14xf32>
    %v5032 = stablehlo.multiply %v5024, %v5031 : tensor<32x240x14x14xf32>
    %v5033 = stablehlo.reshape %v5032 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v5034 = stablehlo.reshape %v541 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5035 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5036 = stablehlo.constant dense<6272.0> : tensor<32x240x14x14xf32>
    %v5037 = stablehlo.constant dense<1.0e-5> : tensor<32x240x14x14xf32>
    %v5038 = stablehlo.reduce(%v5034 init: %v5035) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5039 = stablehlo.broadcast_in_dim %v5038, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v5040 = stablehlo.divide %v5039, %v5036 : tensor<32x240x14x14xf32>
    %v5041 = stablehlo.subtract %v5034, %v5040 : tensor<32x240x14x14xf32>
    %v5042 = stablehlo.multiply %v5041, %v5041 : tensor<32x240x14x14xf32>
    %v5043 = stablehlo.reduce(%v5042 init: %v5035) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5044 = stablehlo.broadcast_in_dim %v5043, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v5045 = stablehlo.divide %v5044, %v5036 : tensor<32x240x14x14xf32>
    %v5046 = stablehlo.add %v5045, %v5037 : tensor<32x240x14x14xf32>
    %v5047 = stablehlo.rsqrt %v5046 : tensor<32x240x14x14xf32>
    %v5048 = stablehlo.multiply %v5041, %v5047 : tensor<32x240x14x14xf32>
    %v5049 = stablehlo.broadcast_in_dim %b6dg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v5050 = stablehlo.reshape %v5033 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5051 = stablehlo.multiply %v5049, %v5050 : tensor<32x240x14x14xf32>
    %v5052 = stablehlo.reduce(%v5051 init: %v5035) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5053 = stablehlo.broadcast_in_dim %v5052, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v5054 = stablehlo.multiply %v5048, %v5051 : tensor<32x240x14x14xf32>
    %v5055 = stablehlo.reduce(%v5054 init: %v5035) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5056 = stablehlo.broadcast_in_dim %v5055, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v5057 = stablehlo.multiply %v5051, %v5036 : tensor<32x240x14x14xf32>
    %v5058 = stablehlo.subtract %v5057, %v5053 : tensor<32x240x14x14xf32>
    %v5059 = stablehlo.multiply %v5048, %v5056 : tensor<32x240x14x14xf32>
    %v5060 = stablehlo.subtract %v5058, %v5059 : tensor<32x240x14x14xf32>
    %v5061 = stablehlo.divide %v5047, %v5036 : tensor<32x240x14x14xf32>
    %v5062 = stablehlo.multiply %v5061, %v5060 : tensor<32x240x14x14xf32>
    %v5063 = stablehlo.reshape %v5062 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v5064 = stablehlo.reshape %v5063 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5065 = stablehlo.reverse %b6dW, dims = [2, 3] : tensor<240x1x3x3xf32>
    %v5066 = stablehlo.convolution(%v5064, %v5065)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 1], [2, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x14x14xf32>, tensor<240x1x3x3xf32>) -> tensor<32x240x28x28xf32>
    %v5067 = stablehlo.reshape %v5066 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5068 = stablehlo.reshape %v541 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5069 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5070 = stablehlo.constant dense<6272.0> : tensor<32x240x14x14xf32>
    %v5071 = stablehlo.reduce(%v5068 init: %v5069) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5072 = stablehlo.broadcast_in_dim %v5071, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v5073 = stablehlo.divide %v5072, %v5070 : tensor<32x240x14x14xf32>
    %v5074 = stablehlo.subtract %v5068, %v5073 : tensor<32x240x14x14xf32>
    %v5075 = stablehlo.multiply %v5074, %v5074 : tensor<32x240x14x14xf32>
    %v5076 = stablehlo.reduce(%v5075 init: %v5069) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5077 = stablehlo.broadcast_in_dim %v5076, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v5078 = stablehlo.divide %v5077, %v5070 : tensor<32x240x14x14xf32>
    %v5079 = stablehlo.constant dense<1.0e-5> : tensor<32x240x14x14xf32>
    %v5080 = stablehlo.add %v5078, %v5079 : tensor<32x240x14x14xf32>
    %v5081 = stablehlo.rsqrt %v5080 : tensor<32x240x14x14xf32>
    %v5082 = stablehlo.multiply %v5074, %v5081 : tensor<32x240x14x14xf32>
    %v5083 = stablehlo.reshape %v5033 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5084 = stablehlo.multiply %v5083, %v5082 : tensor<32x240x14x14xf32>
    %v5085 = stablehlo.reduce(%v5084 init: %v5069) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5086 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5087 = stablehlo.multiply %v5085, %v5086 : tensor<240xf32>
    %v5088 = stablehlo.subtract %b6dg, %v5087 : tensor<240xf32>
    %v5089 = stablehlo.reshape %v5033 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5090 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5091 = stablehlo.reduce(%v5089 init: %v5090) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5092 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5093 = stablehlo.multiply %v5091, %v5092 : tensor<240xf32>
    %v5094 = stablehlo.subtract %b6dbt, %v5093 : tensor<240xf32>
    %v5095 = stablehlo.reshape %v536 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5096 = stablehlo.reshape %v5063 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5097 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5098 = stablehlo.pad %v5096, %v5097, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240x28x28xf32>
    %v5099 = stablehlo.transpose %v5095, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %v5100 = stablehlo.transpose %v5098, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %v5101 = stablehlo.convolution(%v5099, %v5100)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 2], [0, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 240 : i64, feature_group_count = 1 : i64} : (tensor<240x32x28x28xf32>, tensor<240x32x28x28xf32>) -> tensor<1x240x3x3xf32>
    %v5102 = stablehlo.reshape %v5101 : (tensor<1x240x3x3xf32>) -> tensor<240x1x3x3xf32>
    %v5103 = stablehlo.constant dense<0.05> : tensor<240x1x3x3xf32>
    %v5104 = stablehlo.multiply %v5102, %v5103 : tensor<240x1x3x3xf32>
    %v5105 = stablehlo.subtract %b6dW, %v5104 : tensor<240x1x3x3xf32>
    %v5106 = stablehlo.reshape %v5067 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5107 = stablehlo.reshape %v532 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5108 = stablehlo.logistic %v5107 : tensor<32x240x28x28xf32>
    %v5109 = stablehlo.constant dense<1.0> : tensor<32x240x28x28xf32>
    %v5110 = stablehlo.subtract %v5109, %v5108 : tensor<32x240x28x28xf32>
    %v5111 = stablehlo.multiply %v5107, %v5110 : tensor<32x240x28x28xf32>
    %v5112 = stablehlo.add %v5109, %v5111 : tensor<32x240x28x28xf32>
    %v5113 = stablehlo.multiply %v5108, %v5112 : tensor<32x240x28x28xf32>
    %v5114 = stablehlo.multiply %v5106, %v5113 : tensor<32x240x28x28xf32>
    %v5115 = stablehlo.reshape %v5114 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5116 = stablehlo.reshape %v512 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5117 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5118 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v5119 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v5120 = stablehlo.reduce(%v5116 init: %v5117) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5121 = stablehlo.broadcast_in_dim %v5120, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5122 = stablehlo.divide %v5121, %v5118 : tensor<32x240x28x28xf32>
    %v5123 = stablehlo.subtract %v5116, %v5122 : tensor<32x240x28x28xf32>
    %v5124 = stablehlo.multiply %v5123, %v5123 : tensor<32x240x28x28xf32>
    %v5125 = stablehlo.reduce(%v5124 init: %v5117) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5126 = stablehlo.broadcast_in_dim %v5125, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5127 = stablehlo.divide %v5126, %v5118 : tensor<32x240x28x28xf32>
    %v5128 = stablehlo.add %v5127, %v5119 : tensor<32x240x28x28xf32>
    %v5129 = stablehlo.rsqrt %v5128 : tensor<32x240x28x28xf32>
    %v5130 = stablehlo.multiply %v5123, %v5129 : tensor<32x240x28x28xf32>
    %v5131 = stablehlo.broadcast_in_dim %b6eg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5132 = stablehlo.reshape %v5115 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5133 = stablehlo.multiply %v5131, %v5132 : tensor<32x240x28x28xf32>
    %v5134 = stablehlo.reduce(%v5133 init: %v5117) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5135 = stablehlo.broadcast_in_dim %v5134, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5136 = stablehlo.multiply %v5130, %v5133 : tensor<32x240x28x28xf32>
    %v5137 = stablehlo.reduce(%v5136 init: %v5117) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5138 = stablehlo.broadcast_in_dim %v5137, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5139 = stablehlo.multiply %v5133, %v5118 : tensor<32x240x28x28xf32>
    %v5140 = stablehlo.subtract %v5139, %v5135 : tensor<32x240x28x28xf32>
    %v5141 = stablehlo.multiply %v5130, %v5138 : tensor<32x240x28x28xf32>
    %v5142 = stablehlo.subtract %v5140, %v5141 : tensor<32x240x28x28xf32>
    %v5143 = stablehlo.divide %v5129, %v5118 : tensor<32x240x28x28xf32>
    %v5144 = stablehlo.multiply %v5143, %v5142 : tensor<32x240x28x28xf32>
    %v5145 = stablehlo.reshape %v5144 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5146 = stablehlo.reshape %v5145 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5147 = stablehlo.reverse %b6eW, dims = [2, 3] : tensor<240x40x1x1xf32>
    %v5148 = stablehlo.transpose %v5147, dims = [1, 0, 2, 3] : (tensor<240x40x1x1xf32>) -> tensor<40x240x1x1xf32>
    %v5149 = stablehlo.convolution(%v5146, %v5148)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x28x28xf32>, tensor<40x240x1x1xf32>) -> tensor<32x40x28x28xf32>
    %v5150 = stablehlo.reshape %v5149 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v5151 = stablehlo.reshape %v512 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5152 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5153 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v5154 = stablehlo.reduce(%v5151 init: %v5152) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5155 = stablehlo.broadcast_in_dim %v5154, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5156 = stablehlo.divide %v5155, %v5153 : tensor<32x240x28x28xf32>
    %v5157 = stablehlo.subtract %v5151, %v5156 : tensor<32x240x28x28xf32>
    %v5158 = stablehlo.multiply %v5157, %v5157 : tensor<32x240x28x28xf32>
    %v5159 = stablehlo.reduce(%v5158 init: %v5152) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5160 = stablehlo.broadcast_in_dim %v5159, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5161 = stablehlo.divide %v5160, %v5153 : tensor<32x240x28x28xf32>
    %v5162 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v5163 = stablehlo.add %v5161, %v5162 : tensor<32x240x28x28xf32>
    %v5164 = stablehlo.rsqrt %v5163 : tensor<32x240x28x28xf32>
    %v5165 = stablehlo.multiply %v5157, %v5164 : tensor<32x240x28x28xf32>
    %v5166 = stablehlo.reshape %v5115 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5167 = stablehlo.multiply %v5166, %v5165 : tensor<32x240x28x28xf32>
    %v5168 = stablehlo.reduce(%v5167 init: %v5152) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5169 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5170 = stablehlo.multiply %v5168, %v5169 : tensor<240xf32>
    %v5171 = stablehlo.subtract %b6eg, %v5170 : tensor<240xf32>
    %v5172 = stablehlo.reshape %v5115 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5173 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5174 = stablehlo.reduce(%v5172 init: %v5173) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5175 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5176 = stablehlo.multiply %v5174, %v5175 : tensor<240xf32>
    %v5177 = stablehlo.subtract %b6ebt, %v5176 : tensor<240xf32>
    %v5178 = stablehlo.reshape %v507 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5179 = stablehlo.reshape %v5145 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5180 = stablehlo.transpose %v5178, dims = [1, 0, 2, 3] : (tensor<32x40x28x28xf32>) -> tensor<40x32x28x28xf32>
    %v5181 = stablehlo.transpose %v5179, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %v5182 = stablehlo.convolution(%v5180, %v5181)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<40x32x28x28xf32>, tensor<240x32x28x28xf32>) -> tensor<40x240x1x1xf32>
    %v5183 = stablehlo.transpose %v5182, dims = [1, 0, 2, 3] : (tensor<40x240x1x1xf32>) -> tensor<240x40x1x1xf32>
    %v5184 = stablehlo.constant dense<0.05> : tensor<240x40x1x1xf32>
    %v5185 = stablehlo.multiply %v5183, %v5184 : tensor<240x40x1x1xf32>
    %v5186 = stablehlo.subtract %b6eW, %v5185 : tensor<240x40x1x1xf32>
    %v5187 = stablehlo.reshape %v483 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5188 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5189 = stablehlo.constant dense<25088.0> : tensor<32x40x28x28xf32>
    %v5190 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v5191 = stablehlo.reduce(%v5187 init: %v5188) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5192 = stablehlo.broadcast_in_dim %v5191, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5193 = stablehlo.divide %v5192, %v5189 : tensor<32x40x28x28xf32>
    %v5194 = stablehlo.subtract %v5187, %v5193 : tensor<32x40x28x28xf32>
    %v5195 = stablehlo.multiply %v5194, %v5194 : tensor<32x40x28x28xf32>
    %v5196 = stablehlo.reduce(%v5195 init: %v5188) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5197 = stablehlo.broadcast_in_dim %v5196, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5198 = stablehlo.divide %v5197, %v5189 : tensor<32x40x28x28xf32>
    %v5199 = stablehlo.add %v5198, %v5190 : tensor<32x40x28x28xf32>
    %v5200 = stablehlo.rsqrt %v5199 : tensor<32x40x28x28xf32>
    %v5201 = stablehlo.multiply %v5194, %v5200 : tensor<32x40x28x28xf32>
    %v5202 = stablehlo.broadcast_in_dim %b5pg, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5203 = stablehlo.reshape %v5150 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5204 = stablehlo.multiply %v5202, %v5203 : tensor<32x40x28x28xf32>
    %v5205 = stablehlo.reduce(%v5204 init: %v5188) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5206 = stablehlo.broadcast_in_dim %v5205, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5207 = stablehlo.multiply %v5201, %v5204 : tensor<32x40x28x28xf32>
    %v5208 = stablehlo.reduce(%v5207 init: %v5188) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5209 = stablehlo.broadcast_in_dim %v5208, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5210 = stablehlo.multiply %v5204, %v5189 : tensor<32x40x28x28xf32>
    %v5211 = stablehlo.subtract %v5210, %v5206 : tensor<32x40x28x28xf32>
    %v5212 = stablehlo.multiply %v5201, %v5209 : tensor<32x40x28x28xf32>
    %v5213 = stablehlo.subtract %v5211, %v5212 : tensor<32x40x28x28xf32>
    %v5214 = stablehlo.divide %v5200, %v5189 : tensor<32x40x28x28xf32>
    %v5215 = stablehlo.multiply %v5214, %v5213 : tensor<32x40x28x28xf32>
    %v5216 = stablehlo.reshape %v5215 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v5217 = stablehlo.reshape %v5216 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5218 = stablehlo.reverse %b5pW, dims = [2, 3] : tensor<40x240x1x1xf32>
    %v5219 = stablehlo.transpose %v5218, dims = [1, 0, 2, 3] : (tensor<40x240x1x1xf32>) -> tensor<240x40x1x1xf32>
    %v5220 = stablehlo.convolution(%v5217, %v5219)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<32x240x28x28xf32>
    %v5221 = stablehlo.reshape %v5220 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5222 = stablehlo.reshape %v483 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5223 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5224 = stablehlo.constant dense<25088.0> : tensor<32x40x28x28xf32>
    %v5225 = stablehlo.reduce(%v5222 init: %v5223) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5226 = stablehlo.broadcast_in_dim %v5225, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5227 = stablehlo.divide %v5226, %v5224 : tensor<32x40x28x28xf32>
    %v5228 = stablehlo.subtract %v5222, %v5227 : tensor<32x40x28x28xf32>
    %v5229 = stablehlo.multiply %v5228, %v5228 : tensor<32x40x28x28xf32>
    %v5230 = stablehlo.reduce(%v5229 init: %v5223) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5231 = stablehlo.broadcast_in_dim %v5230, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5232 = stablehlo.divide %v5231, %v5224 : tensor<32x40x28x28xf32>
    %v5233 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v5234 = stablehlo.add %v5232, %v5233 : tensor<32x40x28x28xf32>
    %v5235 = stablehlo.rsqrt %v5234 : tensor<32x40x28x28xf32>
    %v5236 = stablehlo.multiply %v5228, %v5235 : tensor<32x40x28x28xf32>
    %v5237 = stablehlo.reshape %v5150 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5238 = stablehlo.multiply %v5237, %v5236 : tensor<32x40x28x28xf32>
    %v5239 = stablehlo.reduce(%v5238 init: %v5223) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5240 = stablehlo.constant dense<0.05> : tensor<40xf32>
    %v5241 = stablehlo.multiply %v5239, %v5240 : tensor<40xf32>
    %v5242 = stablehlo.subtract %b5pg, %v5241 : tensor<40xf32>
    %v5243 = stablehlo.reshape %v5150 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5244 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5245 = stablehlo.reduce(%v5243 init: %v5244) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5246 = stablehlo.constant dense<0.05> : tensor<40xf32>
    %v5247 = stablehlo.multiply %v5245, %v5246 : tensor<40xf32>
    %v5248 = stablehlo.subtract %b5pbt, %v5247 : tensor<40xf32>
    %v5249 = stablehlo.reshape %v478 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5250 = stablehlo.reshape %v5216 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5251 = stablehlo.transpose %v5249, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %v5252 = stablehlo.transpose %v5250, dims = [1, 0, 2, 3] : (tensor<32x40x28x28xf32>) -> tensor<40x32x28x28xf32>
    %v5253 = stablehlo.convolution(%v5251, %v5252)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<240x32x28x28xf32>, tensor<40x32x28x28xf32>) -> tensor<240x40x1x1xf32>
    %v5254 = stablehlo.transpose %v5253, dims = [1, 0, 2, 3] : (tensor<240x40x1x1xf32>) -> tensor<40x240x1x1xf32>
    %v5255 = stablehlo.constant dense<0.05> : tensor<40x240x1x1xf32>
    %v5256 = stablehlo.multiply %v5254, %v5255 : tensor<40x240x1x1xf32>
    %v5257 = stablehlo.subtract %b5pW, %v5256 : tensor<40x240x1x1xf32>
    %v5258 = stablehlo.reshape %v461 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5259 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5260 = stablehlo.reduce(%v5258 init: %v5259) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v5261 = stablehlo.constant dense<784.0> : tensor<32x240xf32>
    %v5262 = stablehlo.divide %v5260, %v5261 : tensor<32x240xf32>
    %v5263 = stablehlo.dot_general %v5262, %b5zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v5264 = stablehlo.broadcast_in_dim %b5zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v5265 = stablehlo.add %v5263, %v5264 : tensor<32x10xf32>
    %v5266 = stablehlo.logistic %v5265 : tensor<32x10xf32>
    %v5267 = stablehlo.multiply %v5265, %v5266 : tensor<32x10xf32>
    %v5268 = stablehlo.dot_general %v5267, %b5zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v5269 = stablehlo.broadcast_in_dim %b5zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v5270 = stablehlo.add %v5268, %v5269 : tensor<32x240xf32>
    %v5271 = stablehlo.logistic %v5270 : tensor<32x240xf32>
    %v5272 = stablehlo.reshape %v5221 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5273 = stablehlo.broadcast_in_dim %v5271, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x28x28xf32>
    %v5274 = stablehlo.multiply %v5273, %v5272 : tensor<32x240x28x28xf32>
    %v5275 = stablehlo.multiply %v5258, %v5272 : tensor<32x240x28x28xf32>
    %v5276 = stablehlo.reduce(%v5275 init: %v5259) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v5277 = stablehlo.constant dense<1.0> : tensor<32x240xf32>
    %v5278 = stablehlo.subtract %v5277, %v5271 : tensor<32x240xf32>
    %v5279 = stablehlo.multiply %v5271, %v5278 : tensor<32x240xf32>
    %v5280 = stablehlo.multiply %v5276, %v5279 : tensor<32x240xf32>
    %v5281 = stablehlo.dot_general %v5280, %b5zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<10x240xf32>) -> tensor<32x10xf32>
    %v5282 = stablehlo.logistic %v5265 : tensor<32x10xf32>
    %v5283 = stablehlo.constant dense<1.0> : tensor<32x10xf32>
    %v5284 = stablehlo.subtract %v5283, %v5282 : tensor<32x10xf32>
    %v5285 = stablehlo.multiply %v5265, %v5284 : tensor<32x10xf32>
    %v5286 = stablehlo.add %v5283, %v5285 : tensor<32x10xf32>
    %v5287 = stablehlo.multiply %v5282, %v5286 : tensor<32x10xf32>
    %v5288 = stablehlo.multiply %v5281, %v5287 : tensor<32x10xf32>
    %v5289 = stablehlo.dot_general %v5288, %b5zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<240x10xf32>) -> tensor<32x240xf32>
    %v5290 = stablehlo.constant dense<784.0> : tensor<32x240xf32>
    %v5291 = stablehlo.divide %v5289, %v5290 : tensor<32x240xf32>
    %v5292 = stablehlo.broadcast_in_dim %v5291, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x28x28xf32>
    %v5293 = stablehlo.add %v5274, %v5292 : tensor<32x240x28x28xf32>
    %v5294 = stablehlo.reshape %v5293 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5295 = stablehlo.reshape %v461 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5296 = stablehlo.reshape %v5221 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5297 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5298 = stablehlo.multiply %v5295, %v5296 : tensor<32x240x28x28xf32>
    %v5299 = stablehlo.reduce(%v5298 init: %v5297) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v5300 = stablehlo.logistic %v474 : tensor<32x240xf32>
    %v5301 = stablehlo.constant dense<1.0> : tensor<32x240xf32>
    %v5302 = stablehlo.subtract %v5301, %v5300 : tensor<32x240xf32>
    %v5303 = stablehlo.multiply %v5300, %v5302 : tensor<32x240xf32>
    %v5304 = stablehlo.multiply %v5299, %v5303 : tensor<32x240xf32>
    %v5305 = stablehlo.dot_general %v471, %v5304, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<32x240xf32>) -> tensor<10x240xf32>
    %v5306 = stablehlo.constant dense<0.05> : tensor<10x240xf32>
    %v5307 = stablehlo.multiply %v5305, %v5306 : tensor<10x240xf32>
    %v5308 = stablehlo.subtract %b5zW2, %v5307 : tensor<10x240xf32>
    %v5309 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5310 = stablehlo.reduce(%v5304 init: %v5309) applies stablehlo.add across dimensions = [0] : (tensor<32x240xf32>, tensor<f32>) -> tensor<240xf32>
    %v5311 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5312 = stablehlo.multiply %v5310, %v5311 : tensor<240xf32>
    %v5313 = stablehlo.subtract %b5zb2, %v5312 : tensor<240xf32>
    %v5314 = stablehlo.reshape %v5304 : (tensor<32x240xf32>) -> tensor<32x1x240xf32>
    %v5315 = stablehlo.dot_general %v5314, %b5zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x240xf32>, tensor<10x240xf32>) -> tensor<32x1x10xf32>
    %v5316 = stablehlo.reshape %v5315 : (tensor<32x1x10xf32>) -> tensor<32x10xf32>
    %v5317 = stablehlo.logistic %v469 : tensor<32x10xf32>
    %v5318 = stablehlo.constant dense<1.0> : tensor<32x10xf32>
    %v5319 = stablehlo.subtract %v5318, %v5317 : tensor<32x10xf32>
    %v5320 = stablehlo.multiply %v469, %v5319 : tensor<32x10xf32>
    %v5321 = stablehlo.add %v5318, %v5320 : tensor<32x10xf32>
    %v5322 = stablehlo.multiply %v5317, %v5321 : tensor<32x10xf32>
    %v5323 = stablehlo.multiply %v5316, %v5322 : tensor<32x10xf32>
    %v5324 = stablehlo.dot_general %v466, %v5323, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<32x10xf32>) -> tensor<240x10xf32>
    %v5325 = stablehlo.constant dense<0.05> : tensor<240x10xf32>
    %v5326 = stablehlo.multiply %v5324, %v5325 : tensor<240x10xf32>
    %v5327 = stablehlo.subtract %b5zW1, %v5326 : tensor<240x10xf32>
    %v5328 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5329 = stablehlo.reduce(%v5323 init: %v5328) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v5330 = stablehlo.constant dense<0.05> : tensor<10xf32>
    %v5331 = stablehlo.multiply %v5329, %v5330 : tensor<10xf32>
    %v5332 = stablehlo.subtract %b5zb1, %v5331 : tensor<10xf32>
    %v5333 = stablehlo.reshape %v5294 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5334 = stablehlo.reshape %v457 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5335 = stablehlo.logistic %v5334 : tensor<32x240x28x28xf32>
    %v5336 = stablehlo.constant dense<1.0> : tensor<32x240x28x28xf32>
    %v5337 = stablehlo.subtract %v5336, %v5335 : tensor<32x240x28x28xf32>
    %v5338 = stablehlo.multiply %v5334, %v5337 : tensor<32x240x28x28xf32>
    %v5339 = stablehlo.add %v5336, %v5338 : tensor<32x240x28x28xf32>
    %v5340 = stablehlo.multiply %v5335, %v5339 : tensor<32x240x28x28xf32>
    %v5341 = stablehlo.multiply %v5333, %v5340 : tensor<32x240x28x28xf32>
    %v5342 = stablehlo.reshape %v5341 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5343 = stablehlo.reshape %v437 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5344 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5345 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v5346 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v5347 = stablehlo.reduce(%v5343 init: %v5344) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5348 = stablehlo.broadcast_in_dim %v5347, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5349 = stablehlo.divide %v5348, %v5345 : tensor<32x240x28x28xf32>
    %v5350 = stablehlo.subtract %v5343, %v5349 : tensor<32x240x28x28xf32>
    %v5351 = stablehlo.multiply %v5350, %v5350 : tensor<32x240x28x28xf32>
    %v5352 = stablehlo.reduce(%v5351 init: %v5344) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5353 = stablehlo.broadcast_in_dim %v5352, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5354 = stablehlo.divide %v5353, %v5345 : tensor<32x240x28x28xf32>
    %v5355 = stablehlo.add %v5354, %v5346 : tensor<32x240x28x28xf32>
    %v5356 = stablehlo.rsqrt %v5355 : tensor<32x240x28x28xf32>
    %v5357 = stablehlo.multiply %v5350, %v5356 : tensor<32x240x28x28xf32>
    %v5358 = stablehlo.broadcast_in_dim %b5dg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5359 = stablehlo.reshape %v5342 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5360 = stablehlo.multiply %v5358, %v5359 : tensor<32x240x28x28xf32>
    %v5361 = stablehlo.reduce(%v5360 init: %v5344) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5362 = stablehlo.broadcast_in_dim %v5361, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5363 = stablehlo.multiply %v5357, %v5360 : tensor<32x240x28x28xf32>
    %v5364 = stablehlo.reduce(%v5363 init: %v5344) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5365 = stablehlo.broadcast_in_dim %v5364, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5366 = stablehlo.multiply %v5360, %v5345 : tensor<32x240x28x28xf32>
    %v5367 = stablehlo.subtract %v5366, %v5362 : tensor<32x240x28x28xf32>
    %v5368 = stablehlo.multiply %v5357, %v5365 : tensor<32x240x28x28xf32>
    %v5369 = stablehlo.subtract %v5367, %v5368 : tensor<32x240x28x28xf32>
    %v5370 = stablehlo.divide %v5356, %v5345 : tensor<32x240x28x28xf32>
    %v5371 = stablehlo.multiply %v5370, %v5369 : tensor<32x240x28x28xf32>
    %v5372 = stablehlo.reshape %v5371 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5373 = stablehlo.reshape %v5372 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5374 = stablehlo.reverse %b5dW, dims = [2, 3] : tensor<240x1x5x5xf32>
    %v5375 = stablehlo.convolution(%v5373, %v5374)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x5x5xf32>) -> tensor<32x240x28x28xf32>
    %v5376 = stablehlo.reshape %v5375 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5377 = stablehlo.reshape %v437 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5378 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5379 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v5380 = stablehlo.reduce(%v5377 init: %v5378) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5381 = stablehlo.broadcast_in_dim %v5380, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5382 = stablehlo.divide %v5381, %v5379 : tensor<32x240x28x28xf32>
    %v5383 = stablehlo.subtract %v5377, %v5382 : tensor<32x240x28x28xf32>
    %v5384 = stablehlo.multiply %v5383, %v5383 : tensor<32x240x28x28xf32>
    %v5385 = stablehlo.reduce(%v5384 init: %v5378) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5386 = stablehlo.broadcast_in_dim %v5385, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5387 = stablehlo.divide %v5386, %v5379 : tensor<32x240x28x28xf32>
    %v5388 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v5389 = stablehlo.add %v5387, %v5388 : tensor<32x240x28x28xf32>
    %v5390 = stablehlo.rsqrt %v5389 : tensor<32x240x28x28xf32>
    %v5391 = stablehlo.multiply %v5383, %v5390 : tensor<32x240x28x28xf32>
    %v5392 = stablehlo.reshape %v5342 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5393 = stablehlo.multiply %v5392, %v5391 : tensor<32x240x28x28xf32>
    %v5394 = stablehlo.reduce(%v5393 init: %v5378) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5395 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5396 = stablehlo.multiply %v5394, %v5395 : tensor<240xf32>
    %v5397 = stablehlo.subtract %b5dg, %v5396 : tensor<240xf32>
    %v5398 = stablehlo.reshape %v5342 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5399 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5400 = stablehlo.reduce(%v5398 init: %v5399) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5401 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5402 = stablehlo.multiply %v5400, %v5401 : tensor<240xf32>
    %v5403 = stablehlo.subtract %b5dbt, %v5402 : tensor<240xf32>
    %v5404 = stablehlo.reshape %v432 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5405 = stablehlo.reshape %v5372 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5406 = stablehlo.transpose %v5404, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %v5407 = stablehlo.transpose %v5405, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %v5408 = stablehlo.convolution(%v5406, %v5407)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 240 : i64, feature_group_count = 1 : i64} : (tensor<240x32x28x28xf32>, tensor<240x32x28x28xf32>) -> tensor<1x240x5x5xf32>
    %v5409 = stablehlo.reshape %v5408 : (tensor<1x240x5x5xf32>) -> tensor<240x1x5x5xf32>
    %v5410 = stablehlo.constant dense<0.05> : tensor<240x1x5x5xf32>
    %v5411 = stablehlo.multiply %v5409, %v5410 : tensor<240x1x5x5xf32>
    %v5412 = stablehlo.subtract %b5dW, %v5411 : tensor<240x1x5x5xf32>
    %v5413 = stablehlo.reshape %v5376 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5414 = stablehlo.reshape %v428 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5415 = stablehlo.logistic %v5414 : tensor<32x240x28x28xf32>
    %v5416 = stablehlo.constant dense<1.0> : tensor<32x240x28x28xf32>
    %v5417 = stablehlo.subtract %v5416, %v5415 : tensor<32x240x28x28xf32>
    %v5418 = stablehlo.multiply %v5414, %v5417 : tensor<32x240x28x28xf32>
    %v5419 = stablehlo.add %v5416, %v5418 : tensor<32x240x28x28xf32>
    %v5420 = stablehlo.multiply %v5415, %v5419 : tensor<32x240x28x28xf32>
    %v5421 = stablehlo.multiply %v5413, %v5420 : tensor<32x240x28x28xf32>
    %v5422 = stablehlo.reshape %v5421 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5423 = stablehlo.reshape %v408 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5424 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5425 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v5426 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v5427 = stablehlo.reduce(%v5423 init: %v5424) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5428 = stablehlo.broadcast_in_dim %v5427, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5429 = stablehlo.divide %v5428, %v5425 : tensor<32x240x28x28xf32>
    %v5430 = stablehlo.subtract %v5423, %v5429 : tensor<32x240x28x28xf32>
    %v5431 = stablehlo.multiply %v5430, %v5430 : tensor<32x240x28x28xf32>
    %v5432 = stablehlo.reduce(%v5431 init: %v5424) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5433 = stablehlo.broadcast_in_dim %v5432, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5434 = stablehlo.divide %v5433, %v5425 : tensor<32x240x28x28xf32>
    %v5435 = stablehlo.add %v5434, %v5426 : tensor<32x240x28x28xf32>
    %v5436 = stablehlo.rsqrt %v5435 : tensor<32x240x28x28xf32>
    %v5437 = stablehlo.multiply %v5430, %v5436 : tensor<32x240x28x28xf32>
    %v5438 = stablehlo.broadcast_in_dim %b5eg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5439 = stablehlo.reshape %v5422 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5440 = stablehlo.multiply %v5438, %v5439 : tensor<32x240x28x28xf32>
    %v5441 = stablehlo.reduce(%v5440 init: %v5424) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5442 = stablehlo.broadcast_in_dim %v5441, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5443 = stablehlo.multiply %v5437, %v5440 : tensor<32x240x28x28xf32>
    %v5444 = stablehlo.reduce(%v5443 init: %v5424) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5445 = stablehlo.broadcast_in_dim %v5444, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5446 = stablehlo.multiply %v5440, %v5425 : tensor<32x240x28x28xf32>
    %v5447 = stablehlo.subtract %v5446, %v5442 : tensor<32x240x28x28xf32>
    %v5448 = stablehlo.multiply %v5437, %v5445 : tensor<32x240x28x28xf32>
    %v5449 = stablehlo.subtract %v5447, %v5448 : tensor<32x240x28x28xf32>
    %v5450 = stablehlo.divide %v5436, %v5425 : tensor<32x240x28x28xf32>
    %v5451 = stablehlo.multiply %v5450, %v5449 : tensor<32x240x28x28xf32>
    %v5452 = stablehlo.reshape %v5451 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5453 = stablehlo.reshape %v5452 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5454 = stablehlo.reverse %b5eW, dims = [2, 3] : tensor<240x40x1x1xf32>
    %v5455 = stablehlo.transpose %v5454, dims = [1, 0, 2, 3] : (tensor<240x40x1x1xf32>) -> tensor<40x240x1x1xf32>
    %v5456 = stablehlo.convolution(%v5453, %v5455)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x28x28xf32>, tensor<40x240x1x1xf32>) -> tensor<32x40x28x28xf32>
    %v5457 = stablehlo.reshape %v5456 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v5458 = stablehlo.reshape %v408 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5459 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5460 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v5461 = stablehlo.reduce(%v5458 init: %v5459) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5462 = stablehlo.broadcast_in_dim %v5461, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5463 = stablehlo.divide %v5462, %v5460 : tensor<32x240x28x28xf32>
    %v5464 = stablehlo.subtract %v5458, %v5463 : tensor<32x240x28x28xf32>
    %v5465 = stablehlo.multiply %v5464, %v5464 : tensor<32x240x28x28xf32>
    %v5466 = stablehlo.reduce(%v5465 init: %v5459) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5467 = stablehlo.broadcast_in_dim %v5466, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5468 = stablehlo.divide %v5467, %v5460 : tensor<32x240x28x28xf32>
    %v5469 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v5470 = stablehlo.add %v5468, %v5469 : tensor<32x240x28x28xf32>
    %v5471 = stablehlo.rsqrt %v5470 : tensor<32x240x28x28xf32>
    %v5472 = stablehlo.multiply %v5464, %v5471 : tensor<32x240x28x28xf32>
    %v5473 = stablehlo.reshape %v5422 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5474 = stablehlo.multiply %v5473, %v5472 : tensor<32x240x28x28xf32>
    %v5475 = stablehlo.reduce(%v5474 init: %v5459) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5476 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5477 = stablehlo.multiply %v5475, %v5476 : tensor<240xf32>
    %v5478 = stablehlo.subtract %b5eg, %v5477 : tensor<240xf32>
    %v5479 = stablehlo.reshape %v5422 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5480 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5481 = stablehlo.reduce(%v5479 init: %v5480) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5482 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5483 = stablehlo.multiply %v5481, %v5482 : tensor<240xf32>
    %v5484 = stablehlo.subtract %b5ebt, %v5483 : tensor<240xf32>
    %v5485 = stablehlo.reshape %v403 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5486 = stablehlo.reshape %v5452 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5487 = stablehlo.transpose %v5485, dims = [1, 0, 2, 3] : (tensor<32x40x28x28xf32>) -> tensor<40x32x28x28xf32>
    %v5488 = stablehlo.transpose %v5486, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %v5489 = stablehlo.convolution(%v5487, %v5488)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<40x32x28x28xf32>, tensor<240x32x28x28xf32>) -> tensor<40x240x1x1xf32>
    %v5490 = stablehlo.transpose %v5489, dims = [1, 0, 2, 3] : (tensor<40x240x1x1xf32>) -> tensor<240x40x1x1xf32>
    %v5491 = stablehlo.constant dense<0.05> : tensor<240x40x1x1xf32>
    %v5492 = stablehlo.multiply %v5490, %v5491 : tensor<240x40x1x1xf32>
    %v5493 = stablehlo.subtract %b5eW, %v5492 : tensor<240x40x1x1xf32>
    %v5494 = stablehlo.reshape %v5457 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5495 = stablehlo.reshape %v5150 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5496 = stablehlo.add %v5494, %v5495 : tensor<32x40x28x28xf32>
    %v5497 = stablehlo.reshape %v5496 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v5498 = stablehlo.reshape %v383 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5499 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5500 = stablehlo.constant dense<25088.0> : tensor<32x40x28x28xf32>
    %v5501 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v5502 = stablehlo.reduce(%v5498 init: %v5499) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5503 = stablehlo.broadcast_in_dim %v5502, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5504 = stablehlo.divide %v5503, %v5500 : tensor<32x40x28x28xf32>
    %v5505 = stablehlo.subtract %v5498, %v5504 : tensor<32x40x28x28xf32>
    %v5506 = stablehlo.multiply %v5505, %v5505 : tensor<32x40x28x28xf32>
    %v5507 = stablehlo.reduce(%v5506 init: %v5499) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5508 = stablehlo.broadcast_in_dim %v5507, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5509 = stablehlo.divide %v5508, %v5500 : tensor<32x40x28x28xf32>
    %v5510 = stablehlo.add %v5509, %v5501 : tensor<32x40x28x28xf32>
    %v5511 = stablehlo.rsqrt %v5510 : tensor<32x40x28x28xf32>
    %v5512 = stablehlo.multiply %v5505, %v5511 : tensor<32x40x28x28xf32>
    %v5513 = stablehlo.broadcast_in_dim %b4pg, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5514 = stablehlo.reshape %v5497 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5515 = stablehlo.multiply %v5513, %v5514 : tensor<32x40x28x28xf32>
    %v5516 = stablehlo.reduce(%v5515 init: %v5499) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5517 = stablehlo.broadcast_in_dim %v5516, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5518 = stablehlo.multiply %v5512, %v5515 : tensor<32x40x28x28xf32>
    %v5519 = stablehlo.reduce(%v5518 init: %v5499) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5520 = stablehlo.broadcast_in_dim %v5519, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5521 = stablehlo.multiply %v5515, %v5500 : tensor<32x40x28x28xf32>
    %v5522 = stablehlo.subtract %v5521, %v5517 : tensor<32x40x28x28xf32>
    %v5523 = stablehlo.multiply %v5512, %v5520 : tensor<32x40x28x28xf32>
    %v5524 = stablehlo.subtract %v5522, %v5523 : tensor<32x40x28x28xf32>
    %v5525 = stablehlo.divide %v5511, %v5500 : tensor<32x40x28x28xf32>
    %v5526 = stablehlo.multiply %v5525, %v5524 : tensor<32x40x28x28xf32>
    %v5527 = stablehlo.reshape %v5526 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v5528 = stablehlo.reshape %v5527 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5529 = stablehlo.reverse %b4pW, dims = [2, 3] : tensor<40x144x1x1xf32>
    %v5530 = stablehlo.transpose %v5529, dims = [1, 0, 2, 3] : (tensor<40x144x1x1xf32>) -> tensor<144x40x1x1xf32>
    %v5531 = stablehlo.convolution(%v5528, %v5530)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<144x40x1x1xf32>) -> tensor<32x144x28x28xf32>
    %v5532 = stablehlo.reshape %v5531 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v5533 = stablehlo.reshape %v383 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5534 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5535 = stablehlo.constant dense<25088.0> : tensor<32x40x28x28xf32>
    %v5536 = stablehlo.reduce(%v5533 init: %v5534) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5537 = stablehlo.broadcast_in_dim %v5536, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5538 = stablehlo.divide %v5537, %v5535 : tensor<32x40x28x28xf32>
    %v5539 = stablehlo.subtract %v5533, %v5538 : tensor<32x40x28x28xf32>
    %v5540 = stablehlo.multiply %v5539, %v5539 : tensor<32x40x28x28xf32>
    %v5541 = stablehlo.reduce(%v5540 init: %v5534) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5542 = stablehlo.broadcast_in_dim %v5541, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5543 = stablehlo.divide %v5542, %v5535 : tensor<32x40x28x28xf32>
    %v5544 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v5545 = stablehlo.add %v5543, %v5544 : tensor<32x40x28x28xf32>
    %v5546 = stablehlo.rsqrt %v5545 : tensor<32x40x28x28xf32>
    %v5547 = stablehlo.multiply %v5539, %v5546 : tensor<32x40x28x28xf32>
    %v5548 = stablehlo.reshape %v5497 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5549 = stablehlo.multiply %v5548, %v5547 : tensor<32x40x28x28xf32>
    %v5550 = stablehlo.reduce(%v5549 init: %v5534) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5551 = stablehlo.constant dense<0.05> : tensor<40xf32>
    %v5552 = stablehlo.multiply %v5550, %v5551 : tensor<40xf32>
    %v5553 = stablehlo.subtract %b4pg, %v5552 : tensor<40xf32>
    %v5554 = stablehlo.reshape %v5497 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5555 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5556 = stablehlo.reduce(%v5554 init: %v5555) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5557 = stablehlo.constant dense<0.05> : tensor<40xf32>
    %v5558 = stablehlo.multiply %v5556, %v5557 : tensor<40xf32>
    %v5559 = stablehlo.subtract %b4pbt, %v5558 : tensor<40xf32>
    %v5560 = stablehlo.reshape %v378 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5561 = stablehlo.reshape %v5527 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5562 = stablehlo.transpose %v5560, dims = [1, 0, 2, 3] : (tensor<32x144x28x28xf32>) -> tensor<144x32x28x28xf32>
    %v5563 = stablehlo.transpose %v5561, dims = [1, 0, 2, 3] : (tensor<32x40x28x28xf32>) -> tensor<40x32x28x28xf32>
    %v5564 = stablehlo.convolution(%v5562, %v5563)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<144x32x28x28xf32>, tensor<40x32x28x28xf32>) -> tensor<144x40x1x1xf32>
    %v5565 = stablehlo.transpose %v5564, dims = [1, 0, 2, 3] : (tensor<144x40x1x1xf32>) -> tensor<40x144x1x1xf32>
    %v5566 = stablehlo.constant dense<0.05> : tensor<40x144x1x1xf32>
    %v5567 = stablehlo.multiply %v5565, %v5566 : tensor<40x144x1x1xf32>
    %v5568 = stablehlo.subtract %b4pW, %v5567 : tensor<40x144x1x1xf32>
    %v5569 = stablehlo.reshape %v361 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5570 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5571 = stablehlo.reduce(%v5569 init: %v5570) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5572 = stablehlo.constant dense<784.0> : tensor<32x144xf32>
    %v5573 = stablehlo.divide %v5571, %v5572 : tensor<32x144xf32>
    %v5574 = stablehlo.dot_general %v5573, %b4zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v5575 = stablehlo.broadcast_in_dim %b4zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v5576 = stablehlo.add %v5574, %v5575 : tensor<32x6xf32>
    %v5577 = stablehlo.logistic %v5576 : tensor<32x6xf32>
    %v5578 = stablehlo.multiply %v5576, %v5577 : tensor<32x6xf32>
    %v5579 = stablehlo.dot_general %v5578, %b4zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v5580 = stablehlo.broadcast_in_dim %b4zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v5581 = stablehlo.add %v5579, %v5580 : tensor<32x144xf32>
    %v5582 = stablehlo.logistic %v5581 : tensor<32x144xf32>
    %v5583 = stablehlo.reshape %v5532 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5584 = stablehlo.broadcast_in_dim %v5582, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v5585 = stablehlo.multiply %v5584, %v5583 : tensor<32x144x28x28xf32>
    %v5586 = stablehlo.multiply %v5569, %v5583 : tensor<32x144x28x28xf32>
    %v5587 = stablehlo.reduce(%v5586 init: %v5570) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5588 = stablehlo.constant dense<1.0> : tensor<32x144xf32>
    %v5589 = stablehlo.subtract %v5588, %v5582 : tensor<32x144xf32>
    %v5590 = stablehlo.multiply %v5582, %v5589 : tensor<32x144xf32>
    %v5591 = stablehlo.multiply %v5587, %v5590 : tensor<32x144xf32>
    %v5592 = stablehlo.dot_general %v5591, %b4zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<6x144xf32>) -> tensor<32x6xf32>
    %v5593 = stablehlo.logistic %v5576 : tensor<32x6xf32>
    %v5594 = stablehlo.constant dense<1.0> : tensor<32x6xf32>
    %v5595 = stablehlo.subtract %v5594, %v5593 : tensor<32x6xf32>
    %v5596 = stablehlo.multiply %v5576, %v5595 : tensor<32x6xf32>
    %v5597 = stablehlo.add %v5594, %v5596 : tensor<32x6xf32>
    %v5598 = stablehlo.multiply %v5593, %v5597 : tensor<32x6xf32>
    %v5599 = stablehlo.multiply %v5592, %v5598 : tensor<32x6xf32>
    %v5600 = stablehlo.dot_general %v5599, %b4zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<144x6xf32>) -> tensor<32x144xf32>
    %v5601 = stablehlo.constant dense<784.0> : tensor<32x144xf32>
    %v5602 = stablehlo.divide %v5600, %v5601 : tensor<32x144xf32>
    %v5603 = stablehlo.broadcast_in_dim %v5602, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v5604 = stablehlo.add %v5585, %v5603 : tensor<32x144x28x28xf32>
    %v5605 = stablehlo.reshape %v5604 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v5606 = stablehlo.reshape %v361 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5607 = stablehlo.reshape %v5532 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5608 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5609 = stablehlo.multiply %v5606, %v5607 : tensor<32x144x28x28xf32>
    %v5610 = stablehlo.reduce(%v5609 init: %v5608) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5611 = stablehlo.logistic %v374 : tensor<32x144xf32>
    %v5612 = stablehlo.constant dense<1.0> : tensor<32x144xf32>
    %v5613 = stablehlo.subtract %v5612, %v5611 : tensor<32x144xf32>
    %v5614 = stablehlo.multiply %v5611, %v5613 : tensor<32x144xf32>
    %v5615 = stablehlo.multiply %v5610, %v5614 : tensor<32x144xf32>
    %v5616 = stablehlo.dot_general %v371, %v5615, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<32x144xf32>) -> tensor<6x144xf32>
    %v5617 = stablehlo.constant dense<0.05> : tensor<6x144xf32>
    %v5618 = stablehlo.multiply %v5616, %v5617 : tensor<6x144xf32>
    %v5619 = stablehlo.subtract %b4zW2, %v5618 : tensor<6x144xf32>
    %v5620 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5621 = stablehlo.reduce(%v5615 init: %v5620) applies stablehlo.add across dimensions = [0] : (tensor<32x144xf32>, tensor<f32>) -> tensor<144xf32>
    %v5622 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v5623 = stablehlo.multiply %v5621, %v5622 : tensor<144xf32>
    %v5624 = stablehlo.subtract %b4zb2, %v5623 : tensor<144xf32>
    %v5625 = stablehlo.reshape %v5615 : (tensor<32x144xf32>) -> tensor<32x1x144xf32>
    %v5626 = stablehlo.dot_general %v5625, %b4zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x144xf32>, tensor<6x144xf32>) -> tensor<32x1x6xf32>
    %v5627 = stablehlo.reshape %v5626 : (tensor<32x1x6xf32>) -> tensor<32x6xf32>
    %v5628 = stablehlo.logistic %v369 : tensor<32x6xf32>
    %v5629 = stablehlo.constant dense<1.0> : tensor<32x6xf32>
    %v5630 = stablehlo.subtract %v5629, %v5628 : tensor<32x6xf32>
    %v5631 = stablehlo.multiply %v369, %v5630 : tensor<32x6xf32>
    %v5632 = stablehlo.add %v5629, %v5631 : tensor<32x6xf32>
    %v5633 = stablehlo.multiply %v5628, %v5632 : tensor<32x6xf32>
    %v5634 = stablehlo.multiply %v5627, %v5633 : tensor<32x6xf32>
    %v5635 = stablehlo.dot_general %v366, %v5634, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<32x6xf32>) -> tensor<144x6xf32>
    %v5636 = stablehlo.constant dense<0.05> : tensor<144x6xf32>
    %v5637 = stablehlo.multiply %v5635, %v5636 : tensor<144x6xf32>
    %v5638 = stablehlo.subtract %b4zW1, %v5637 : tensor<144x6xf32>
    %v5639 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5640 = stablehlo.reduce(%v5634 init: %v5639) applies stablehlo.add across dimensions = [0] : (tensor<32x6xf32>, tensor<f32>) -> tensor<6xf32>
    %v5641 = stablehlo.constant dense<0.05> : tensor<6xf32>
    %v5642 = stablehlo.multiply %v5640, %v5641 : tensor<6xf32>
    %v5643 = stablehlo.subtract %b4zb1, %v5642 : tensor<6xf32>
    %v5644 = stablehlo.reshape %v5605 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5645 = stablehlo.reshape %v357 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5646 = stablehlo.logistic %v5645 : tensor<32x144x28x28xf32>
    %v5647 = stablehlo.constant dense<1.0> : tensor<32x144x28x28xf32>
    %v5648 = stablehlo.subtract %v5647, %v5646 : tensor<32x144x28x28xf32>
    %v5649 = stablehlo.multiply %v5645, %v5648 : tensor<32x144x28x28xf32>
    %v5650 = stablehlo.add %v5647, %v5649 : tensor<32x144x28x28xf32>
    %v5651 = stablehlo.multiply %v5646, %v5650 : tensor<32x144x28x28xf32>
    %v5652 = stablehlo.multiply %v5644, %v5651 : tensor<32x144x28x28xf32>
    %v5653 = stablehlo.reshape %v5652 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v5654 = stablehlo.reshape %v337 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5655 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5656 = stablehlo.constant dense<25088.0> : tensor<32x144x28x28xf32>
    %v5657 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v5658 = stablehlo.reduce(%v5654 init: %v5655) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5659 = stablehlo.broadcast_in_dim %v5658, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5660 = stablehlo.divide %v5659, %v5656 : tensor<32x144x28x28xf32>
    %v5661 = stablehlo.subtract %v5654, %v5660 : tensor<32x144x28x28xf32>
    %v5662 = stablehlo.multiply %v5661, %v5661 : tensor<32x144x28x28xf32>
    %v5663 = stablehlo.reduce(%v5662 init: %v5655) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5664 = stablehlo.broadcast_in_dim %v5663, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5665 = stablehlo.divide %v5664, %v5656 : tensor<32x144x28x28xf32>
    %v5666 = stablehlo.add %v5665, %v5657 : tensor<32x144x28x28xf32>
    %v5667 = stablehlo.rsqrt %v5666 : tensor<32x144x28x28xf32>
    %v5668 = stablehlo.multiply %v5661, %v5667 : tensor<32x144x28x28xf32>
    %v5669 = stablehlo.broadcast_in_dim %b4dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5670 = stablehlo.reshape %v5653 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5671 = stablehlo.multiply %v5669, %v5670 : tensor<32x144x28x28xf32>
    %v5672 = stablehlo.reduce(%v5671 init: %v5655) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5673 = stablehlo.broadcast_in_dim %v5672, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5674 = stablehlo.multiply %v5668, %v5671 : tensor<32x144x28x28xf32>
    %v5675 = stablehlo.reduce(%v5674 init: %v5655) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5676 = stablehlo.broadcast_in_dim %v5675, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5677 = stablehlo.multiply %v5671, %v5656 : tensor<32x144x28x28xf32>
    %v5678 = stablehlo.subtract %v5677, %v5673 : tensor<32x144x28x28xf32>
    %v5679 = stablehlo.multiply %v5668, %v5676 : tensor<32x144x28x28xf32>
    %v5680 = stablehlo.subtract %v5678, %v5679 : tensor<32x144x28x28xf32>
    %v5681 = stablehlo.divide %v5667, %v5656 : tensor<32x144x28x28xf32>
    %v5682 = stablehlo.multiply %v5681, %v5680 : tensor<32x144x28x28xf32>
    %v5683 = stablehlo.reshape %v5682 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v5684 = stablehlo.reshape %v5683 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5685 = stablehlo.reverse %b4dW, dims = [2, 3] : tensor<144x1x5x5xf32>
    %v5686 = stablehlo.convolution(%v5684, %v5685)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x28x28xf32>, tensor<144x1x5x5xf32>) -> tensor<32x144x56x56xf32>
    %v5687 = stablehlo.reshape %v5686 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v5688 = stablehlo.reshape %v337 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5689 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5690 = stablehlo.constant dense<25088.0> : tensor<32x144x28x28xf32>
    %v5691 = stablehlo.reduce(%v5688 init: %v5689) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5692 = stablehlo.broadcast_in_dim %v5691, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5693 = stablehlo.divide %v5692, %v5690 : tensor<32x144x28x28xf32>
    %v5694 = stablehlo.subtract %v5688, %v5693 : tensor<32x144x28x28xf32>
    %v5695 = stablehlo.multiply %v5694, %v5694 : tensor<32x144x28x28xf32>
    %v5696 = stablehlo.reduce(%v5695 init: %v5689) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5697 = stablehlo.broadcast_in_dim %v5696, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5698 = stablehlo.divide %v5697, %v5690 : tensor<32x144x28x28xf32>
    %v5699 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v5700 = stablehlo.add %v5698, %v5699 : tensor<32x144x28x28xf32>
    %v5701 = stablehlo.rsqrt %v5700 : tensor<32x144x28x28xf32>
    %v5702 = stablehlo.multiply %v5694, %v5701 : tensor<32x144x28x28xf32>
    %v5703 = stablehlo.reshape %v5653 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5704 = stablehlo.multiply %v5703, %v5702 : tensor<32x144x28x28xf32>
    %v5705 = stablehlo.reduce(%v5704 init: %v5689) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5706 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v5707 = stablehlo.multiply %v5705, %v5706 : tensor<144xf32>
    %v5708 = stablehlo.subtract %b4dg, %v5707 : tensor<144xf32>
    %v5709 = stablehlo.reshape %v5653 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5710 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5711 = stablehlo.reduce(%v5709 init: %v5710) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5712 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v5713 = stablehlo.multiply %v5711, %v5712 : tensor<144xf32>
    %v5714 = stablehlo.subtract %b4dbt, %v5713 : tensor<144xf32>
    %v5715 = stablehlo.reshape %v332 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5716 = stablehlo.reshape %v5683 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5717 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5718 = stablehlo.pad %v5716, %v5717, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144x56x56xf32>
    %v5719 = stablehlo.transpose %v5715, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v5720 = stablehlo.transpose %v5718, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v5721 = stablehlo.convolution(%v5719, %v5720)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 3], [1, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 144 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<1x144x5x5xf32>
    %v5722 = stablehlo.reshape %v5721 : (tensor<1x144x5x5xf32>) -> tensor<144x1x5x5xf32>
    %v5723 = stablehlo.constant dense<0.05> : tensor<144x1x5x5xf32>
    %v5724 = stablehlo.multiply %v5722, %v5723 : tensor<144x1x5x5xf32>
    %v5725 = stablehlo.subtract %b4dW, %v5724 : tensor<144x1x5x5xf32>
    %v5726 = stablehlo.reshape %v5687 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5727 = stablehlo.reshape %v328 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5728 = stablehlo.logistic %v5727 : tensor<32x144x56x56xf32>
    %v5729 = stablehlo.constant dense<1.0> : tensor<32x144x56x56xf32>
    %v5730 = stablehlo.subtract %v5729, %v5728 : tensor<32x144x56x56xf32>
    %v5731 = stablehlo.multiply %v5727, %v5730 : tensor<32x144x56x56xf32>
    %v5732 = stablehlo.add %v5729, %v5731 : tensor<32x144x56x56xf32>
    %v5733 = stablehlo.multiply %v5728, %v5732 : tensor<32x144x56x56xf32>
    %v5734 = stablehlo.multiply %v5726, %v5733 : tensor<32x144x56x56xf32>
    %v5735 = stablehlo.reshape %v5734 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v5736 = stablehlo.reshape %v308 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5737 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5738 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v5739 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v5740 = stablehlo.reduce(%v5736 init: %v5737) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5741 = stablehlo.broadcast_in_dim %v5740, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5742 = stablehlo.divide %v5741, %v5738 : tensor<32x144x56x56xf32>
    %v5743 = stablehlo.subtract %v5736, %v5742 : tensor<32x144x56x56xf32>
    %v5744 = stablehlo.multiply %v5743, %v5743 : tensor<32x144x56x56xf32>
    %v5745 = stablehlo.reduce(%v5744 init: %v5737) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5746 = stablehlo.broadcast_in_dim %v5745, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5747 = stablehlo.divide %v5746, %v5738 : tensor<32x144x56x56xf32>
    %v5748 = stablehlo.add %v5747, %v5739 : tensor<32x144x56x56xf32>
    %v5749 = stablehlo.rsqrt %v5748 : tensor<32x144x56x56xf32>
    %v5750 = stablehlo.multiply %v5743, %v5749 : tensor<32x144x56x56xf32>
    %v5751 = stablehlo.broadcast_in_dim %b4eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5752 = stablehlo.reshape %v5735 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5753 = stablehlo.multiply %v5751, %v5752 : tensor<32x144x56x56xf32>
    %v5754 = stablehlo.reduce(%v5753 init: %v5737) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5755 = stablehlo.broadcast_in_dim %v5754, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5756 = stablehlo.multiply %v5750, %v5753 : tensor<32x144x56x56xf32>
    %v5757 = stablehlo.reduce(%v5756 init: %v5737) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5758 = stablehlo.broadcast_in_dim %v5757, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5759 = stablehlo.multiply %v5753, %v5738 : tensor<32x144x56x56xf32>
    %v5760 = stablehlo.subtract %v5759, %v5755 : tensor<32x144x56x56xf32>
    %v5761 = stablehlo.multiply %v5750, %v5758 : tensor<32x144x56x56xf32>
    %v5762 = stablehlo.subtract %v5760, %v5761 : tensor<32x144x56x56xf32>
    %v5763 = stablehlo.divide %v5749, %v5738 : tensor<32x144x56x56xf32>
    %v5764 = stablehlo.multiply %v5763, %v5762 : tensor<32x144x56x56xf32>
    %v5765 = stablehlo.reshape %v5764 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v5766 = stablehlo.reshape %v5765 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5767 = stablehlo.reverse %b4eW, dims = [2, 3] : tensor<144x24x1x1xf32>
    %v5768 = stablehlo.transpose %v5767, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %v5769 = stablehlo.convolution(%v5766, %v5768)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v5770 = stablehlo.reshape %v5769 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v5771 = stablehlo.reshape %v308 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5772 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5773 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v5774 = stablehlo.reduce(%v5771 init: %v5772) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5775 = stablehlo.broadcast_in_dim %v5774, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5776 = stablehlo.divide %v5775, %v5773 : tensor<32x144x56x56xf32>
    %v5777 = stablehlo.subtract %v5771, %v5776 : tensor<32x144x56x56xf32>
    %v5778 = stablehlo.multiply %v5777, %v5777 : tensor<32x144x56x56xf32>
    %v5779 = stablehlo.reduce(%v5778 init: %v5772) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5780 = stablehlo.broadcast_in_dim %v5779, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5781 = stablehlo.divide %v5780, %v5773 : tensor<32x144x56x56xf32>
    %v5782 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v5783 = stablehlo.add %v5781, %v5782 : tensor<32x144x56x56xf32>
    %v5784 = stablehlo.rsqrt %v5783 : tensor<32x144x56x56xf32>
    %v5785 = stablehlo.multiply %v5777, %v5784 : tensor<32x144x56x56xf32>
    %v5786 = stablehlo.reshape %v5735 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5787 = stablehlo.multiply %v5786, %v5785 : tensor<32x144x56x56xf32>
    %v5788 = stablehlo.reduce(%v5787 init: %v5772) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5789 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v5790 = stablehlo.multiply %v5788, %v5789 : tensor<144xf32>
    %v5791 = stablehlo.subtract %b4eg, %v5790 : tensor<144xf32>
    %v5792 = stablehlo.reshape %v5735 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5793 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5794 = stablehlo.reduce(%v5792 init: %v5793) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5795 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v5796 = stablehlo.multiply %v5794, %v5795 : tensor<144xf32>
    %v5797 = stablehlo.subtract %b4ebt, %v5796 : tensor<144xf32>
    %v5798 = stablehlo.reshape %v303 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5799 = stablehlo.reshape %v5765 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5800 = stablehlo.transpose %v5798, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v5801 = stablehlo.transpose %v5799, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v5802 = stablehlo.convolution(%v5800, %v5801)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<24x144x1x1xf32>
    %v5803 = stablehlo.transpose %v5802, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %v5804 = stablehlo.constant dense<0.05> : tensor<144x24x1x1xf32>
    %v5805 = stablehlo.multiply %v5803, %v5804 : tensor<144x24x1x1xf32>
    %v5806 = stablehlo.subtract %b4eW, %v5805 : tensor<144x24x1x1xf32>
    %v5807 = stablehlo.reshape %v279 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5808 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5809 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v5810 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v5811 = stablehlo.reduce(%v5807 init: %v5808) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5812 = stablehlo.broadcast_in_dim %v5811, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v5813 = stablehlo.divide %v5812, %v5809 : tensor<32x24x56x56xf32>
    %v5814 = stablehlo.subtract %v5807, %v5813 : tensor<32x24x56x56xf32>
    %v5815 = stablehlo.multiply %v5814, %v5814 : tensor<32x24x56x56xf32>
    %v5816 = stablehlo.reduce(%v5815 init: %v5808) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5817 = stablehlo.broadcast_in_dim %v5816, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v5818 = stablehlo.divide %v5817, %v5809 : tensor<32x24x56x56xf32>
    %v5819 = stablehlo.add %v5818, %v5810 : tensor<32x24x56x56xf32>
    %v5820 = stablehlo.rsqrt %v5819 : tensor<32x24x56x56xf32>
    %v5821 = stablehlo.multiply %v5814, %v5820 : tensor<32x24x56x56xf32>
    %v5822 = stablehlo.broadcast_in_dim %b3pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v5823 = stablehlo.reshape %v5770 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5824 = stablehlo.multiply %v5822, %v5823 : tensor<32x24x56x56xf32>
    %v5825 = stablehlo.reduce(%v5824 init: %v5808) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5826 = stablehlo.broadcast_in_dim %v5825, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v5827 = stablehlo.multiply %v5821, %v5824 : tensor<32x24x56x56xf32>
    %v5828 = stablehlo.reduce(%v5827 init: %v5808) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5829 = stablehlo.broadcast_in_dim %v5828, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v5830 = stablehlo.multiply %v5824, %v5809 : tensor<32x24x56x56xf32>
    %v5831 = stablehlo.subtract %v5830, %v5826 : tensor<32x24x56x56xf32>
    %v5832 = stablehlo.multiply %v5821, %v5829 : tensor<32x24x56x56xf32>
    %v5833 = stablehlo.subtract %v5831, %v5832 : tensor<32x24x56x56xf32>
    %v5834 = stablehlo.divide %v5820, %v5809 : tensor<32x24x56x56xf32>
    %v5835 = stablehlo.multiply %v5834, %v5833 : tensor<32x24x56x56xf32>
    %v5836 = stablehlo.reshape %v5835 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v5837 = stablehlo.reshape %v5836 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5838 = stablehlo.reverse %b3pW, dims = [2, 3] : tensor<24x144x1x1xf32>
    %v5839 = stablehlo.transpose %v5838, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %v5840 = stablehlo.convolution(%v5837, %v5839)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v5841 = stablehlo.reshape %v5840 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v5842 = stablehlo.reshape %v279 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5843 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5844 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v5845 = stablehlo.reduce(%v5842 init: %v5843) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5846 = stablehlo.broadcast_in_dim %v5845, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v5847 = stablehlo.divide %v5846, %v5844 : tensor<32x24x56x56xf32>
    %v5848 = stablehlo.subtract %v5842, %v5847 : tensor<32x24x56x56xf32>
    %v5849 = stablehlo.multiply %v5848, %v5848 : tensor<32x24x56x56xf32>
    %v5850 = stablehlo.reduce(%v5849 init: %v5843) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5851 = stablehlo.broadcast_in_dim %v5850, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v5852 = stablehlo.divide %v5851, %v5844 : tensor<32x24x56x56xf32>
    %v5853 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v5854 = stablehlo.add %v5852, %v5853 : tensor<32x24x56x56xf32>
    %v5855 = stablehlo.rsqrt %v5854 : tensor<32x24x56x56xf32>
    %v5856 = stablehlo.multiply %v5848, %v5855 : tensor<32x24x56x56xf32>
    %v5857 = stablehlo.reshape %v5770 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5858 = stablehlo.multiply %v5857, %v5856 : tensor<32x24x56x56xf32>
    %v5859 = stablehlo.reduce(%v5858 init: %v5843) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5860 = stablehlo.constant dense<0.05> : tensor<24xf32>
    %v5861 = stablehlo.multiply %v5859, %v5860 : tensor<24xf32>
    %v5862 = stablehlo.subtract %b3pg, %v5861 : tensor<24xf32>
    %v5863 = stablehlo.reshape %v5770 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5864 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5865 = stablehlo.reduce(%v5863 init: %v5864) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5866 = stablehlo.constant dense<0.05> : tensor<24xf32>
    %v5867 = stablehlo.multiply %v5865, %v5866 : tensor<24xf32>
    %v5868 = stablehlo.subtract %b3pbt, %v5867 : tensor<24xf32>
    %v5869 = stablehlo.reshape %v274 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5870 = stablehlo.reshape %v5836 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5871 = stablehlo.transpose %v5869, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v5872 = stablehlo.transpose %v5870, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v5873 = stablehlo.convolution(%v5871, %v5872)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<24x32x56x56xf32>) -> tensor<144x24x1x1xf32>
    %v5874 = stablehlo.transpose %v5873, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %v5875 = stablehlo.constant dense<0.05> : tensor<24x144x1x1xf32>
    %v5876 = stablehlo.multiply %v5874, %v5875 : tensor<24x144x1x1xf32>
    %v5877 = stablehlo.subtract %b3pW, %v5876 : tensor<24x144x1x1xf32>
    %v5878 = stablehlo.reshape %v257 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5879 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5880 = stablehlo.reduce(%v5878 init: %v5879) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5881 = stablehlo.constant dense<3136.0> : tensor<32x144xf32>
    %v5882 = stablehlo.divide %v5880, %v5881 : tensor<32x144xf32>
    %v5883 = stablehlo.dot_general %v5882, %b3zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v5884 = stablehlo.broadcast_in_dim %b3zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v5885 = stablehlo.add %v5883, %v5884 : tensor<32x6xf32>
    %v5886 = stablehlo.logistic %v5885 : tensor<32x6xf32>
    %v5887 = stablehlo.multiply %v5885, %v5886 : tensor<32x6xf32>
    %v5888 = stablehlo.dot_general %v5887, %b3zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v5889 = stablehlo.broadcast_in_dim %b3zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v5890 = stablehlo.add %v5888, %v5889 : tensor<32x144xf32>
    %v5891 = stablehlo.logistic %v5890 : tensor<32x144xf32>
    %v5892 = stablehlo.reshape %v5841 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5893 = stablehlo.broadcast_in_dim %v5891, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v5894 = stablehlo.multiply %v5893, %v5892 : tensor<32x144x56x56xf32>
    %v5895 = stablehlo.multiply %v5878, %v5892 : tensor<32x144x56x56xf32>
    %v5896 = stablehlo.reduce(%v5895 init: %v5879) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5897 = stablehlo.constant dense<1.0> : tensor<32x144xf32>
    %v5898 = stablehlo.subtract %v5897, %v5891 : tensor<32x144xf32>
    %v5899 = stablehlo.multiply %v5891, %v5898 : tensor<32x144xf32>
    %v5900 = stablehlo.multiply %v5896, %v5899 : tensor<32x144xf32>
    %v5901 = stablehlo.dot_general %v5900, %b3zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<6x144xf32>) -> tensor<32x6xf32>
    %v5902 = stablehlo.logistic %v5885 : tensor<32x6xf32>
    %v5903 = stablehlo.constant dense<1.0> : tensor<32x6xf32>
    %v5904 = stablehlo.subtract %v5903, %v5902 : tensor<32x6xf32>
    %v5905 = stablehlo.multiply %v5885, %v5904 : tensor<32x6xf32>
    %v5906 = stablehlo.add %v5903, %v5905 : tensor<32x6xf32>
    %v5907 = stablehlo.multiply %v5902, %v5906 : tensor<32x6xf32>
    %v5908 = stablehlo.multiply %v5901, %v5907 : tensor<32x6xf32>
    %v5909 = stablehlo.dot_general %v5908, %b3zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<144x6xf32>) -> tensor<32x144xf32>
    %v5910 = stablehlo.constant dense<3136.0> : tensor<32x144xf32>
    %v5911 = stablehlo.divide %v5909, %v5910 : tensor<32x144xf32>
    %v5912 = stablehlo.broadcast_in_dim %v5911, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v5913 = stablehlo.add %v5894, %v5912 : tensor<32x144x56x56xf32>
    %v5914 = stablehlo.reshape %v5913 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v5915 = stablehlo.reshape %v257 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5916 = stablehlo.reshape %v5841 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5917 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5918 = stablehlo.multiply %v5915, %v5916 : tensor<32x144x56x56xf32>
    %v5919 = stablehlo.reduce(%v5918 init: %v5917) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5920 = stablehlo.logistic %v270 : tensor<32x144xf32>
    %v5921 = stablehlo.constant dense<1.0> : tensor<32x144xf32>
    %v5922 = stablehlo.subtract %v5921, %v5920 : tensor<32x144xf32>
    %v5923 = stablehlo.multiply %v5920, %v5922 : tensor<32x144xf32>
    %v5924 = stablehlo.multiply %v5919, %v5923 : tensor<32x144xf32>
    %v5925 = stablehlo.dot_general %v267, %v5924, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<32x144xf32>) -> tensor<6x144xf32>
    %v5926 = stablehlo.constant dense<0.05> : tensor<6x144xf32>
    %v5927 = stablehlo.multiply %v5925, %v5926 : tensor<6x144xf32>
    %v5928 = stablehlo.subtract %b3zW2, %v5927 : tensor<6x144xf32>
    %v5929 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5930 = stablehlo.reduce(%v5924 init: %v5929) applies stablehlo.add across dimensions = [0] : (tensor<32x144xf32>, tensor<f32>) -> tensor<144xf32>
    %v5931 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v5932 = stablehlo.multiply %v5930, %v5931 : tensor<144xf32>
    %v5933 = stablehlo.subtract %b3zb2, %v5932 : tensor<144xf32>
    %v5934 = stablehlo.reshape %v5924 : (tensor<32x144xf32>) -> tensor<32x1x144xf32>
    %v5935 = stablehlo.dot_general %v5934, %b3zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x144xf32>, tensor<6x144xf32>) -> tensor<32x1x6xf32>
    %v5936 = stablehlo.reshape %v5935 : (tensor<32x1x6xf32>) -> tensor<32x6xf32>
    %v5937 = stablehlo.logistic %v265 : tensor<32x6xf32>
    %v5938 = stablehlo.constant dense<1.0> : tensor<32x6xf32>
    %v5939 = stablehlo.subtract %v5938, %v5937 : tensor<32x6xf32>
    %v5940 = stablehlo.multiply %v265, %v5939 : tensor<32x6xf32>
    %v5941 = stablehlo.add %v5938, %v5940 : tensor<32x6xf32>
    %v5942 = stablehlo.multiply %v5937, %v5941 : tensor<32x6xf32>
    %v5943 = stablehlo.multiply %v5936, %v5942 : tensor<32x6xf32>
    %v5944 = stablehlo.dot_general %v262, %v5943, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<32x6xf32>) -> tensor<144x6xf32>
    %v5945 = stablehlo.constant dense<0.05> : tensor<144x6xf32>
    %v5946 = stablehlo.multiply %v5944, %v5945 : tensor<144x6xf32>
    %v5947 = stablehlo.subtract %b3zW1, %v5946 : tensor<144x6xf32>
    %v5948 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5949 = stablehlo.reduce(%v5943 init: %v5948) applies stablehlo.add across dimensions = [0] : (tensor<32x6xf32>, tensor<f32>) -> tensor<6xf32>
    %v5950 = stablehlo.constant dense<0.05> : tensor<6xf32>
    %v5951 = stablehlo.multiply %v5949, %v5950 : tensor<6xf32>
    %v5952 = stablehlo.subtract %b3zb1, %v5951 : tensor<6xf32>
    %v5953 = stablehlo.reshape %v5914 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5954 = stablehlo.reshape %v253 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5955 = stablehlo.logistic %v5954 : tensor<32x144x56x56xf32>
    %v5956 = stablehlo.constant dense<1.0> : tensor<32x144x56x56xf32>
    %v5957 = stablehlo.subtract %v5956, %v5955 : tensor<32x144x56x56xf32>
    %v5958 = stablehlo.multiply %v5954, %v5957 : tensor<32x144x56x56xf32>
    %v5959 = stablehlo.add %v5956, %v5958 : tensor<32x144x56x56xf32>
    %v5960 = stablehlo.multiply %v5955, %v5959 : tensor<32x144x56x56xf32>
    %v5961 = stablehlo.multiply %v5953, %v5960 : tensor<32x144x56x56xf32>
    %v5962 = stablehlo.reshape %v5961 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v5963 = stablehlo.reshape %v233 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5964 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5965 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v5966 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v5967 = stablehlo.reduce(%v5963 init: %v5964) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5968 = stablehlo.broadcast_in_dim %v5967, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5969 = stablehlo.divide %v5968, %v5965 : tensor<32x144x56x56xf32>
    %v5970 = stablehlo.subtract %v5963, %v5969 : tensor<32x144x56x56xf32>
    %v5971 = stablehlo.multiply %v5970, %v5970 : tensor<32x144x56x56xf32>
    %v5972 = stablehlo.reduce(%v5971 init: %v5964) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5973 = stablehlo.broadcast_in_dim %v5972, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5974 = stablehlo.divide %v5973, %v5965 : tensor<32x144x56x56xf32>
    %v5975 = stablehlo.add %v5974, %v5966 : tensor<32x144x56x56xf32>
    %v5976 = stablehlo.rsqrt %v5975 : tensor<32x144x56x56xf32>
    %v5977 = stablehlo.multiply %v5970, %v5976 : tensor<32x144x56x56xf32>
    %v5978 = stablehlo.broadcast_in_dim %b3dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5979 = stablehlo.reshape %v5962 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5980 = stablehlo.multiply %v5978, %v5979 : tensor<32x144x56x56xf32>
    %v5981 = stablehlo.reduce(%v5980 init: %v5964) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5982 = stablehlo.broadcast_in_dim %v5981, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5983 = stablehlo.multiply %v5977, %v5980 : tensor<32x144x56x56xf32>
    %v5984 = stablehlo.reduce(%v5983 init: %v5964) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5985 = stablehlo.broadcast_in_dim %v5984, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5986 = stablehlo.multiply %v5980, %v5965 : tensor<32x144x56x56xf32>
    %v5987 = stablehlo.subtract %v5986, %v5982 : tensor<32x144x56x56xf32>
    %v5988 = stablehlo.multiply %v5977, %v5985 : tensor<32x144x56x56xf32>
    %v5989 = stablehlo.subtract %v5987, %v5988 : tensor<32x144x56x56xf32>
    %v5990 = stablehlo.divide %v5976, %v5965 : tensor<32x144x56x56xf32>
    %v5991 = stablehlo.multiply %v5990, %v5989 : tensor<32x144x56x56xf32>
    %v5992 = stablehlo.reshape %v5991 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v5993 = stablehlo.reshape %v5992 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5994 = stablehlo.reverse %b3dW, dims = [2, 3] : tensor<144x1x3x3xf32>
    %v5995 = stablehlo.convolution(%v5993, %v5994)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x56x56xf32>
    %v5996 = stablehlo.reshape %v5995 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v5997 = stablehlo.reshape %v233 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5998 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5999 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v6000 = stablehlo.reduce(%v5997 init: %v5998) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6001 = stablehlo.broadcast_in_dim %v6000, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6002 = stablehlo.divide %v6001, %v5999 : tensor<32x144x56x56xf32>
    %v6003 = stablehlo.subtract %v5997, %v6002 : tensor<32x144x56x56xf32>
    %v6004 = stablehlo.multiply %v6003, %v6003 : tensor<32x144x56x56xf32>
    %v6005 = stablehlo.reduce(%v6004 init: %v5998) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6006 = stablehlo.broadcast_in_dim %v6005, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6007 = stablehlo.divide %v6006, %v5999 : tensor<32x144x56x56xf32>
    %v6008 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v6009 = stablehlo.add %v6007, %v6008 : tensor<32x144x56x56xf32>
    %v6010 = stablehlo.rsqrt %v6009 : tensor<32x144x56x56xf32>
    %v6011 = stablehlo.multiply %v6003, %v6010 : tensor<32x144x56x56xf32>
    %v6012 = stablehlo.reshape %v5962 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6013 = stablehlo.multiply %v6012, %v6011 : tensor<32x144x56x56xf32>
    %v6014 = stablehlo.reduce(%v6013 init: %v5998) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6015 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v6016 = stablehlo.multiply %v6014, %v6015 : tensor<144xf32>
    %v6017 = stablehlo.subtract %b3dg, %v6016 : tensor<144xf32>
    %v6018 = stablehlo.reshape %v5962 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6019 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6020 = stablehlo.reduce(%v6018 init: %v6019) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6021 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v6022 = stablehlo.multiply %v6020, %v6021 : tensor<144xf32>
    %v6023 = stablehlo.subtract %b3dbt, %v6022 : tensor<144xf32>
    %v6024 = stablehlo.reshape %v228 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6025 = stablehlo.reshape %v5992 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6026 = stablehlo.transpose %v6024, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v6027 = stablehlo.transpose %v6025, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v6028 = stablehlo.convolution(%v6026, %v6027)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 144 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<1x144x3x3xf32>
    %v6029 = stablehlo.reshape %v6028 : (tensor<1x144x3x3xf32>) -> tensor<144x1x3x3xf32>
    %v6030 = stablehlo.constant dense<0.05> : tensor<144x1x3x3xf32>
    %v6031 = stablehlo.multiply %v6029, %v6030 : tensor<144x1x3x3xf32>
    %v6032 = stablehlo.subtract %b3dW, %v6031 : tensor<144x1x3x3xf32>
    %v6033 = stablehlo.reshape %v5996 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6034 = stablehlo.reshape %v224 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6035 = stablehlo.logistic %v6034 : tensor<32x144x56x56xf32>
    %v6036 = stablehlo.constant dense<1.0> : tensor<32x144x56x56xf32>
    %v6037 = stablehlo.subtract %v6036, %v6035 : tensor<32x144x56x56xf32>
    %v6038 = stablehlo.multiply %v6034, %v6037 : tensor<32x144x56x56xf32>
    %v6039 = stablehlo.add %v6036, %v6038 : tensor<32x144x56x56xf32>
    %v6040 = stablehlo.multiply %v6035, %v6039 : tensor<32x144x56x56xf32>
    %v6041 = stablehlo.multiply %v6033, %v6040 : tensor<32x144x56x56xf32>
    %v6042 = stablehlo.reshape %v6041 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v6043 = stablehlo.reshape %v204 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6044 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6045 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v6046 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v6047 = stablehlo.reduce(%v6043 init: %v6044) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6048 = stablehlo.broadcast_in_dim %v6047, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6049 = stablehlo.divide %v6048, %v6045 : tensor<32x144x56x56xf32>
    %v6050 = stablehlo.subtract %v6043, %v6049 : tensor<32x144x56x56xf32>
    %v6051 = stablehlo.multiply %v6050, %v6050 : tensor<32x144x56x56xf32>
    %v6052 = stablehlo.reduce(%v6051 init: %v6044) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6053 = stablehlo.broadcast_in_dim %v6052, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6054 = stablehlo.divide %v6053, %v6045 : tensor<32x144x56x56xf32>
    %v6055 = stablehlo.add %v6054, %v6046 : tensor<32x144x56x56xf32>
    %v6056 = stablehlo.rsqrt %v6055 : tensor<32x144x56x56xf32>
    %v6057 = stablehlo.multiply %v6050, %v6056 : tensor<32x144x56x56xf32>
    %v6058 = stablehlo.broadcast_in_dim %b3eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6059 = stablehlo.reshape %v6042 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6060 = stablehlo.multiply %v6058, %v6059 : tensor<32x144x56x56xf32>
    %v6061 = stablehlo.reduce(%v6060 init: %v6044) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6062 = stablehlo.broadcast_in_dim %v6061, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6063 = stablehlo.multiply %v6057, %v6060 : tensor<32x144x56x56xf32>
    %v6064 = stablehlo.reduce(%v6063 init: %v6044) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6065 = stablehlo.broadcast_in_dim %v6064, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6066 = stablehlo.multiply %v6060, %v6045 : tensor<32x144x56x56xf32>
    %v6067 = stablehlo.subtract %v6066, %v6062 : tensor<32x144x56x56xf32>
    %v6068 = stablehlo.multiply %v6057, %v6065 : tensor<32x144x56x56xf32>
    %v6069 = stablehlo.subtract %v6067, %v6068 : tensor<32x144x56x56xf32>
    %v6070 = stablehlo.divide %v6056, %v6045 : tensor<32x144x56x56xf32>
    %v6071 = stablehlo.multiply %v6070, %v6069 : tensor<32x144x56x56xf32>
    %v6072 = stablehlo.reshape %v6071 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v6073 = stablehlo.reshape %v6072 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6074 = stablehlo.reverse %b3eW, dims = [2, 3] : tensor<144x24x1x1xf32>
    %v6075 = stablehlo.transpose %v6074, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %v6076 = stablehlo.convolution(%v6073, %v6075)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v6077 = stablehlo.reshape %v6076 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v6078 = stablehlo.reshape %v204 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6079 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6080 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v6081 = stablehlo.reduce(%v6078 init: %v6079) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6082 = stablehlo.broadcast_in_dim %v6081, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6083 = stablehlo.divide %v6082, %v6080 : tensor<32x144x56x56xf32>
    %v6084 = stablehlo.subtract %v6078, %v6083 : tensor<32x144x56x56xf32>
    %v6085 = stablehlo.multiply %v6084, %v6084 : tensor<32x144x56x56xf32>
    %v6086 = stablehlo.reduce(%v6085 init: %v6079) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6087 = stablehlo.broadcast_in_dim %v6086, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6088 = stablehlo.divide %v6087, %v6080 : tensor<32x144x56x56xf32>
    %v6089 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v6090 = stablehlo.add %v6088, %v6089 : tensor<32x144x56x56xf32>
    %v6091 = stablehlo.rsqrt %v6090 : tensor<32x144x56x56xf32>
    %v6092 = stablehlo.multiply %v6084, %v6091 : tensor<32x144x56x56xf32>
    %v6093 = stablehlo.reshape %v6042 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6094 = stablehlo.multiply %v6093, %v6092 : tensor<32x144x56x56xf32>
    %v6095 = stablehlo.reduce(%v6094 init: %v6079) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6096 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v6097 = stablehlo.multiply %v6095, %v6096 : tensor<144xf32>
    %v6098 = stablehlo.subtract %b3eg, %v6097 : tensor<144xf32>
    %v6099 = stablehlo.reshape %v6042 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6100 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6101 = stablehlo.reduce(%v6099 init: %v6100) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6102 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v6103 = stablehlo.multiply %v6101, %v6102 : tensor<144xf32>
    %v6104 = stablehlo.subtract %b3ebt, %v6103 : tensor<144xf32>
    %v6105 = stablehlo.reshape %v199 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6106 = stablehlo.reshape %v6072 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6107 = stablehlo.transpose %v6105, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v6108 = stablehlo.transpose %v6106, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v6109 = stablehlo.convolution(%v6107, %v6108)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<24x144x1x1xf32>
    %v6110 = stablehlo.transpose %v6109, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %v6111 = stablehlo.constant dense<0.05> : tensor<144x24x1x1xf32>
    %v6112 = stablehlo.multiply %v6110, %v6111 : tensor<144x24x1x1xf32>
    %v6113 = stablehlo.subtract %b3eW, %v6112 : tensor<144x24x1x1xf32>
    %v6114 = stablehlo.reshape %v6077 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6115 = stablehlo.reshape %v5770 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6116 = stablehlo.add %v6114, %v6115 : tensor<32x24x56x56xf32>
    %v6117 = stablehlo.reshape %v6116 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v6118 = stablehlo.reshape %v179 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6119 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6120 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v6121 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v6122 = stablehlo.reduce(%v6118 init: %v6119) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6123 = stablehlo.broadcast_in_dim %v6122, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6124 = stablehlo.divide %v6123, %v6120 : tensor<32x24x56x56xf32>
    %v6125 = stablehlo.subtract %v6118, %v6124 : tensor<32x24x56x56xf32>
    %v6126 = stablehlo.multiply %v6125, %v6125 : tensor<32x24x56x56xf32>
    %v6127 = stablehlo.reduce(%v6126 init: %v6119) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6128 = stablehlo.broadcast_in_dim %v6127, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6129 = stablehlo.divide %v6128, %v6120 : tensor<32x24x56x56xf32>
    %v6130 = stablehlo.add %v6129, %v6121 : tensor<32x24x56x56xf32>
    %v6131 = stablehlo.rsqrt %v6130 : tensor<32x24x56x56xf32>
    %v6132 = stablehlo.multiply %v6125, %v6131 : tensor<32x24x56x56xf32>
    %v6133 = stablehlo.broadcast_in_dim %b2pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6134 = stablehlo.reshape %v6117 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6135 = stablehlo.multiply %v6133, %v6134 : tensor<32x24x56x56xf32>
    %v6136 = stablehlo.reduce(%v6135 init: %v6119) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6137 = stablehlo.broadcast_in_dim %v6136, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6138 = stablehlo.multiply %v6132, %v6135 : tensor<32x24x56x56xf32>
    %v6139 = stablehlo.reduce(%v6138 init: %v6119) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6140 = stablehlo.broadcast_in_dim %v6139, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6141 = stablehlo.multiply %v6135, %v6120 : tensor<32x24x56x56xf32>
    %v6142 = stablehlo.subtract %v6141, %v6137 : tensor<32x24x56x56xf32>
    %v6143 = stablehlo.multiply %v6132, %v6140 : tensor<32x24x56x56xf32>
    %v6144 = stablehlo.subtract %v6142, %v6143 : tensor<32x24x56x56xf32>
    %v6145 = stablehlo.divide %v6131, %v6120 : tensor<32x24x56x56xf32>
    %v6146 = stablehlo.multiply %v6145, %v6144 : tensor<32x24x56x56xf32>
    %v6147 = stablehlo.reshape %v6146 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v6148 = stablehlo.reshape %v6147 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6149 = stablehlo.reverse %b2pW, dims = [2, 3] : tensor<24x96x1x1xf32>
    %v6150 = stablehlo.transpose %v6149, dims = [1, 0, 2, 3] : (tensor<24x96x1x1xf32>) -> tensor<96x24x1x1xf32>
    %v6151 = stablehlo.convolution(%v6148, %v6150)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<96x24x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v6152 = stablehlo.reshape %v6151 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v6153 = stablehlo.reshape %v179 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6154 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6155 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v6156 = stablehlo.reduce(%v6153 init: %v6154) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6157 = stablehlo.broadcast_in_dim %v6156, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6158 = stablehlo.divide %v6157, %v6155 : tensor<32x24x56x56xf32>
    %v6159 = stablehlo.subtract %v6153, %v6158 : tensor<32x24x56x56xf32>
    %v6160 = stablehlo.multiply %v6159, %v6159 : tensor<32x24x56x56xf32>
    %v6161 = stablehlo.reduce(%v6160 init: %v6154) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6162 = stablehlo.broadcast_in_dim %v6161, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6163 = stablehlo.divide %v6162, %v6155 : tensor<32x24x56x56xf32>
    %v6164 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v6165 = stablehlo.add %v6163, %v6164 : tensor<32x24x56x56xf32>
    %v6166 = stablehlo.rsqrt %v6165 : tensor<32x24x56x56xf32>
    %v6167 = stablehlo.multiply %v6159, %v6166 : tensor<32x24x56x56xf32>
    %v6168 = stablehlo.reshape %v6117 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6169 = stablehlo.multiply %v6168, %v6167 : tensor<32x24x56x56xf32>
    %v6170 = stablehlo.reduce(%v6169 init: %v6154) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6171 = stablehlo.constant dense<0.05> : tensor<24xf32>
    %v6172 = stablehlo.multiply %v6170, %v6171 : tensor<24xf32>
    %v6173 = stablehlo.subtract %b2pg, %v6172 : tensor<24xf32>
    %v6174 = stablehlo.reshape %v6117 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6175 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6176 = stablehlo.reduce(%v6174 init: %v6175) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6177 = stablehlo.constant dense<0.05> : tensor<24xf32>
    %v6178 = stablehlo.multiply %v6176, %v6177 : tensor<24xf32>
    %v6179 = stablehlo.subtract %b2pbt, %v6178 : tensor<24xf32>
    %v6180 = stablehlo.reshape %v174 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6181 = stablehlo.reshape %v6147 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6182 = stablehlo.transpose %v6180, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v6183 = stablehlo.transpose %v6181, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v6184 = stablehlo.convolution(%v6182, %v6183)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<24x32x56x56xf32>) -> tensor<96x24x1x1xf32>
    %v6185 = stablehlo.transpose %v6184, dims = [1, 0, 2, 3] : (tensor<96x24x1x1xf32>) -> tensor<24x96x1x1xf32>
    %v6186 = stablehlo.constant dense<0.05> : tensor<24x96x1x1xf32>
    %v6187 = stablehlo.multiply %v6185, %v6186 : tensor<24x96x1x1xf32>
    %v6188 = stablehlo.subtract %b2pW, %v6187 : tensor<24x96x1x1xf32>
    %v6189 = stablehlo.reshape %v157 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6190 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6191 = stablehlo.reduce(%v6189 init: %v6190) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v6192 = stablehlo.constant dense<3136.0> : tensor<32x96xf32>
    %v6193 = stablehlo.divide %v6191, %v6192 : tensor<32x96xf32>
    %v6194 = stablehlo.dot_general %v6193, %b2zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<96x4xf32>) -> tensor<32x4xf32>
    %v6195 = stablehlo.broadcast_in_dim %b2zb1, dims = [1] : (tensor<4xf32>) -> tensor<32x4xf32>
    %v6196 = stablehlo.add %v6194, %v6195 : tensor<32x4xf32>
    %v6197 = stablehlo.logistic %v6196 : tensor<32x4xf32>
    %v6198 = stablehlo.multiply %v6196, %v6197 : tensor<32x4xf32>
    %v6199 = stablehlo.dot_general %v6198, %b2zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<4x96xf32>) -> tensor<32x96xf32>
    %v6200 = stablehlo.broadcast_in_dim %b2zb2, dims = [1] : (tensor<96xf32>) -> tensor<32x96xf32>
    %v6201 = stablehlo.add %v6199, %v6200 : tensor<32x96xf32>
    %v6202 = stablehlo.logistic %v6201 : tensor<32x96xf32>
    %v6203 = stablehlo.reshape %v6152 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6204 = stablehlo.broadcast_in_dim %v6202, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v6205 = stablehlo.multiply %v6204, %v6203 : tensor<32x96x56x56xf32>
    %v6206 = stablehlo.multiply %v6189, %v6203 : tensor<32x96x56x56xf32>
    %v6207 = stablehlo.reduce(%v6206 init: %v6190) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v6208 = stablehlo.constant dense<1.0> : tensor<32x96xf32>
    %v6209 = stablehlo.subtract %v6208, %v6202 : tensor<32x96xf32>
    %v6210 = stablehlo.multiply %v6202, %v6209 : tensor<32x96xf32>
    %v6211 = stablehlo.multiply %v6207, %v6210 : tensor<32x96xf32>
    %v6212 = stablehlo.dot_general %v6211, %b2zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<4x96xf32>) -> tensor<32x4xf32>
    %v6213 = stablehlo.logistic %v6196 : tensor<32x4xf32>
    %v6214 = stablehlo.constant dense<1.0> : tensor<32x4xf32>
    %v6215 = stablehlo.subtract %v6214, %v6213 : tensor<32x4xf32>
    %v6216 = stablehlo.multiply %v6196, %v6215 : tensor<32x4xf32>
    %v6217 = stablehlo.add %v6214, %v6216 : tensor<32x4xf32>
    %v6218 = stablehlo.multiply %v6213, %v6217 : tensor<32x4xf32>
    %v6219 = stablehlo.multiply %v6212, %v6218 : tensor<32x4xf32>
    %v6220 = stablehlo.dot_general %v6219, %b2zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<96x4xf32>) -> tensor<32x96xf32>
    %v6221 = stablehlo.constant dense<3136.0> : tensor<32x96xf32>
    %v6222 = stablehlo.divide %v6220, %v6221 : tensor<32x96xf32>
    %v6223 = stablehlo.broadcast_in_dim %v6222, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v6224 = stablehlo.add %v6205, %v6223 : tensor<32x96x56x56xf32>
    %v6225 = stablehlo.reshape %v6224 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v6226 = stablehlo.reshape %v157 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6227 = stablehlo.reshape %v6152 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6228 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6229 = stablehlo.multiply %v6226, %v6227 : tensor<32x96x56x56xf32>
    %v6230 = stablehlo.reduce(%v6229 init: %v6228) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v6231 = stablehlo.logistic %v170 : tensor<32x96xf32>
    %v6232 = stablehlo.constant dense<1.0> : tensor<32x96xf32>
    %v6233 = stablehlo.subtract %v6232, %v6231 : tensor<32x96xf32>
    %v6234 = stablehlo.multiply %v6231, %v6233 : tensor<32x96xf32>
    %v6235 = stablehlo.multiply %v6230, %v6234 : tensor<32x96xf32>
    %v6236 = stablehlo.dot_general %v167, %v6235, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<32x96xf32>) -> tensor<4x96xf32>
    %v6237 = stablehlo.constant dense<0.05> : tensor<4x96xf32>
    %v6238 = stablehlo.multiply %v6236, %v6237 : tensor<4x96xf32>
    %v6239 = stablehlo.subtract %b2zW2, %v6238 : tensor<4x96xf32>
    %v6240 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6241 = stablehlo.reduce(%v6235 init: %v6240) applies stablehlo.add across dimensions = [0] : (tensor<32x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v6242 = stablehlo.constant dense<0.05> : tensor<96xf32>
    %v6243 = stablehlo.multiply %v6241, %v6242 : tensor<96xf32>
    %v6244 = stablehlo.subtract %b2zb2, %v6243 : tensor<96xf32>
    %v6245 = stablehlo.reshape %v6235 : (tensor<32x96xf32>) -> tensor<32x1x96xf32>
    %v6246 = stablehlo.dot_general %v6245, %b2zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x96xf32>, tensor<4x96xf32>) -> tensor<32x1x4xf32>
    %v6247 = stablehlo.reshape %v6246 : (tensor<32x1x4xf32>) -> tensor<32x4xf32>
    %v6248 = stablehlo.logistic %v165 : tensor<32x4xf32>
    %v6249 = stablehlo.constant dense<1.0> : tensor<32x4xf32>
    %v6250 = stablehlo.subtract %v6249, %v6248 : tensor<32x4xf32>
    %v6251 = stablehlo.multiply %v165, %v6250 : tensor<32x4xf32>
    %v6252 = stablehlo.add %v6249, %v6251 : tensor<32x4xf32>
    %v6253 = stablehlo.multiply %v6248, %v6252 : tensor<32x4xf32>
    %v6254 = stablehlo.multiply %v6247, %v6253 : tensor<32x4xf32>
    %v6255 = stablehlo.dot_general %v162, %v6254, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<32x4xf32>) -> tensor<96x4xf32>
    %v6256 = stablehlo.constant dense<0.05> : tensor<96x4xf32>
    %v6257 = stablehlo.multiply %v6255, %v6256 : tensor<96x4xf32>
    %v6258 = stablehlo.subtract %b2zW1, %v6257 : tensor<96x4xf32>
    %v6259 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6260 = stablehlo.reduce(%v6254 init: %v6259) applies stablehlo.add across dimensions = [0] : (tensor<32x4xf32>, tensor<f32>) -> tensor<4xf32>
    %v6261 = stablehlo.constant dense<0.05> : tensor<4xf32>
    %v6262 = stablehlo.multiply %v6260, %v6261 : tensor<4xf32>
    %v6263 = stablehlo.subtract %b2zb1, %v6262 : tensor<4xf32>
    %v6264 = stablehlo.reshape %v6225 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6265 = stablehlo.reshape %v153 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6266 = stablehlo.logistic %v6265 : tensor<32x96x56x56xf32>
    %v6267 = stablehlo.constant dense<1.0> : tensor<32x96x56x56xf32>
    %v6268 = stablehlo.subtract %v6267, %v6266 : tensor<32x96x56x56xf32>
    %v6269 = stablehlo.multiply %v6265, %v6268 : tensor<32x96x56x56xf32>
    %v6270 = stablehlo.add %v6267, %v6269 : tensor<32x96x56x56xf32>
    %v6271 = stablehlo.multiply %v6266, %v6270 : tensor<32x96x56x56xf32>
    %v6272 = stablehlo.multiply %v6264, %v6271 : tensor<32x96x56x56xf32>
    %v6273 = stablehlo.reshape %v6272 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v6274 = stablehlo.reshape %v133 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6275 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6276 = stablehlo.constant dense<100352.0> : tensor<32x96x56x56xf32>
    %v6277 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v6278 = stablehlo.reduce(%v6274 init: %v6275) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6279 = stablehlo.broadcast_in_dim %v6278, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v6280 = stablehlo.divide %v6279, %v6276 : tensor<32x96x56x56xf32>
    %v6281 = stablehlo.subtract %v6274, %v6280 : tensor<32x96x56x56xf32>
    %v6282 = stablehlo.multiply %v6281, %v6281 : tensor<32x96x56x56xf32>
    %v6283 = stablehlo.reduce(%v6282 init: %v6275) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6284 = stablehlo.broadcast_in_dim %v6283, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v6285 = stablehlo.divide %v6284, %v6276 : tensor<32x96x56x56xf32>
    %v6286 = stablehlo.add %v6285, %v6277 : tensor<32x96x56x56xf32>
    %v6287 = stablehlo.rsqrt %v6286 : tensor<32x96x56x56xf32>
    %v6288 = stablehlo.multiply %v6281, %v6287 : tensor<32x96x56x56xf32>
    %v6289 = stablehlo.broadcast_in_dim %b2dg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v6290 = stablehlo.reshape %v6273 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6291 = stablehlo.multiply %v6289, %v6290 : tensor<32x96x56x56xf32>
    %v6292 = stablehlo.reduce(%v6291 init: %v6275) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6293 = stablehlo.broadcast_in_dim %v6292, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v6294 = stablehlo.multiply %v6288, %v6291 : tensor<32x96x56x56xf32>
    %v6295 = stablehlo.reduce(%v6294 init: %v6275) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6296 = stablehlo.broadcast_in_dim %v6295, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v6297 = stablehlo.multiply %v6291, %v6276 : tensor<32x96x56x56xf32>
    %v6298 = stablehlo.subtract %v6297, %v6293 : tensor<32x96x56x56xf32>
    %v6299 = stablehlo.multiply %v6288, %v6296 : tensor<32x96x56x56xf32>
    %v6300 = stablehlo.subtract %v6298, %v6299 : tensor<32x96x56x56xf32>
    %v6301 = stablehlo.divide %v6287, %v6276 : tensor<32x96x56x56xf32>
    %v6302 = stablehlo.multiply %v6301, %v6300 : tensor<32x96x56x56xf32>
    %v6303 = stablehlo.reshape %v6302 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v6304 = stablehlo.reshape %v6303 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6305 = stablehlo.reverse %b2dW, dims = [2, 3] : tensor<96x1x3x3xf32>
    %v6306 = stablehlo.convolution(%v6304, %v6305)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 1], [2, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x112x112xf32>
    %v6307 = stablehlo.reshape %v6306 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v6308 = stablehlo.reshape %v133 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6309 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6310 = stablehlo.constant dense<100352.0> : tensor<32x96x56x56xf32>
    %v6311 = stablehlo.reduce(%v6308 init: %v6309) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6312 = stablehlo.broadcast_in_dim %v6311, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v6313 = stablehlo.divide %v6312, %v6310 : tensor<32x96x56x56xf32>
    %v6314 = stablehlo.subtract %v6308, %v6313 : tensor<32x96x56x56xf32>
    %v6315 = stablehlo.multiply %v6314, %v6314 : tensor<32x96x56x56xf32>
    %v6316 = stablehlo.reduce(%v6315 init: %v6309) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6317 = stablehlo.broadcast_in_dim %v6316, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v6318 = stablehlo.divide %v6317, %v6310 : tensor<32x96x56x56xf32>
    %v6319 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v6320 = stablehlo.add %v6318, %v6319 : tensor<32x96x56x56xf32>
    %v6321 = stablehlo.rsqrt %v6320 : tensor<32x96x56x56xf32>
    %v6322 = stablehlo.multiply %v6314, %v6321 : tensor<32x96x56x56xf32>
    %v6323 = stablehlo.reshape %v6273 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6324 = stablehlo.multiply %v6323, %v6322 : tensor<32x96x56x56xf32>
    %v6325 = stablehlo.reduce(%v6324 init: %v6309) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6326 = stablehlo.constant dense<0.05> : tensor<96xf32>
    %v6327 = stablehlo.multiply %v6325, %v6326 : tensor<96xf32>
    %v6328 = stablehlo.subtract %b2dg, %v6327 : tensor<96xf32>
    %v6329 = stablehlo.reshape %v6273 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6330 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6331 = stablehlo.reduce(%v6329 init: %v6330) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6332 = stablehlo.constant dense<0.05> : tensor<96xf32>
    %v6333 = stablehlo.multiply %v6331, %v6332 : tensor<96xf32>
    %v6334 = stablehlo.subtract %b2dbt, %v6333 : tensor<96xf32>
    %v6335 = stablehlo.reshape %v128 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6336 = stablehlo.reshape %v6303 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6337 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6338 = stablehlo.pad %v6336, %v6337, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96x112x112xf32>
    %v6339 = stablehlo.transpose %v6335, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %v6340 = stablehlo.transpose %v6338, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %v6341 = stablehlo.convolution(%v6339, %v6340)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 2], [0, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x112x112xf32>, tensor<96x32x112x112xf32>) -> tensor<1x96x3x3xf32>
    %v6342 = stablehlo.reshape %v6341 : (tensor<1x96x3x3xf32>) -> tensor<96x1x3x3xf32>
    %v6343 = stablehlo.constant dense<0.05> : tensor<96x1x3x3xf32>
    %v6344 = stablehlo.multiply %v6342, %v6343 : tensor<96x1x3x3xf32>
    %v6345 = stablehlo.subtract %b2dW, %v6344 : tensor<96x1x3x3xf32>
    %v6346 = stablehlo.reshape %v6307 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6347 = stablehlo.reshape %v124 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6348 = stablehlo.logistic %v6347 : tensor<32x96x112x112xf32>
    %v6349 = stablehlo.constant dense<1.0> : tensor<32x96x112x112xf32>
    %v6350 = stablehlo.subtract %v6349, %v6348 : tensor<32x96x112x112xf32>
    %v6351 = stablehlo.multiply %v6347, %v6350 : tensor<32x96x112x112xf32>
    %v6352 = stablehlo.add %v6349, %v6351 : tensor<32x96x112x112xf32>
    %v6353 = stablehlo.multiply %v6348, %v6352 : tensor<32x96x112x112xf32>
    %v6354 = stablehlo.multiply %v6346, %v6353 : tensor<32x96x112x112xf32>
    %v6355 = stablehlo.reshape %v6354 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v6356 = stablehlo.reshape %v104 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6357 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6358 = stablehlo.constant dense<401408.0> : tensor<32x96x112x112xf32>
    %v6359 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v6360 = stablehlo.reduce(%v6356 init: %v6357) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6361 = stablehlo.broadcast_in_dim %v6360, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v6362 = stablehlo.divide %v6361, %v6358 : tensor<32x96x112x112xf32>
    %v6363 = stablehlo.subtract %v6356, %v6362 : tensor<32x96x112x112xf32>
    %v6364 = stablehlo.multiply %v6363, %v6363 : tensor<32x96x112x112xf32>
    %v6365 = stablehlo.reduce(%v6364 init: %v6357) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6366 = stablehlo.broadcast_in_dim %v6365, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v6367 = stablehlo.divide %v6366, %v6358 : tensor<32x96x112x112xf32>
    %v6368 = stablehlo.add %v6367, %v6359 : tensor<32x96x112x112xf32>
    %v6369 = stablehlo.rsqrt %v6368 : tensor<32x96x112x112xf32>
    %v6370 = stablehlo.multiply %v6363, %v6369 : tensor<32x96x112x112xf32>
    %v6371 = stablehlo.broadcast_in_dim %b2eg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v6372 = stablehlo.reshape %v6355 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6373 = stablehlo.multiply %v6371, %v6372 : tensor<32x96x112x112xf32>
    %v6374 = stablehlo.reduce(%v6373 init: %v6357) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6375 = stablehlo.broadcast_in_dim %v6374, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v6376 = stablehlo.multiply %v6370, %v6373 : tensor<32x96x112x112xf32>
    %v6377 = stablehlo.reduce(%v6376 init: %v6357) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6378 = stablehlo.broadcast_in_dim %v6377, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v6379 = stablehlo.multiply %v6373, %v6358 : tensor<32x96x112x112xf32>
    %v6380 = stablehlo.subtract %v6379, %v6375 : tensor<32x96x112x112xf32>
    %v6381 = stablehlo.multiply %v6370, %v6378 : tensor<32x96x112x112xf32>
    %v6382 = stablehlo.subtract %v6380, %v6381 : tensor<32x96x112x112xf32>
    %v6383 = stablehlo.divide %v6369, %v6358 : tensor<32x96x112x112xf32>
    %v6384 = stablehlo.multiply %v6383, %v6382 : tensor<32x96x112x112xf32>
    %v6385 = stablehlo.reshape %v6384 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v6386 = stablehlo.reshape %v6385 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6387 = stablehlo.reverse %b2eW, dims = [2, 3] : tensor<96x16x1x1xf32>
    %v6388 = stablehlo.transpose %v6387, dims = [1, 0, 2, 3] : (tensor<96x16x1x1xf32>) -> tensor<16x96x1x1xf32>
    %v6389 = stablehlo.convolution(%v6386, %v6388)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x112x112xf32>, tensor<16x96x1x1xf32>) -> tensor<32x16x112x112xf32>
    %v6390 = stablehlo.reshape %v6389 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v6391 = stablehlo.reshape %v104 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6392 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6393 = stablehlo.constant dense<401408.0> : tensor<32x96x112x112xf32>
    %v6394 = stablehlo.reduce(%v6391 init: %v6392) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6395 = stablehlo.broadcast_in_dim %v6394, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v6396 = stablehlo.divide %v6395, %v6393 : tensor<32x96x112x112xf32>
    %v6397 = stablehlo.subtract %v6391, %v6396 : tensor<32x96x112x112xf32>
    %v6398 = stablehlo.multiply %v6397, %v6397 : tensor<32x96x112x112xf32>
    %v6399 = stablehlo.reduce(%v6398 init: %v6392) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6400 = stablehlo.broadcast_in_dim %v6399, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v6401 = stablehlo.divide %v6400, %v6393 : tensor<32x96x112x112xf32>
    %v6402 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v6403 = stablehlo.add %v6401, %v6402 : tensor<32x96x112x112xf32>
    %v6404 = stablehlo.rsqrt %v6403 : tensor<32x96x112x112xf32>
    %v6405 = stablehlo.multiply %v6397, %v6404 : tensor<32x96x112x112xf32>
    %v6406 = stablehlo.reshape %v6355 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6407 = stablehlo.multiply %v6406, %v6405 : tensor<32x96x112x112xf32>
    %v6408 = stablehlo.reduce(%v6407 init: %v6392) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6409 = stablehlo.constant dense<0.05> : tensor<96xf32>
    %v6410 = stablehlo.multiply %v6408, %v6409 : tensor<96xf32>
    %v6411 = stablehlo.subtract %b2eg, %v6410 : tensor<96xf32>
    %v6412 = stablehlo.reshape %v6355 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6413 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6414 = stablehlo.reduce(%v6412 init: %v6413) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6415 = stablehlo.constant dense<0.05> : tensor<96xf32>
    %v6416 = stablehlo.multiply %v6414, %v6415 : tensor<96xf32>
    %v6417 = stablehlo.subtract %b2ebt, %v6416 : tensor<96xf32>
    %v6418 = stablehlo.reshape %v99 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6419 = stablehlo.reshape %v6385 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6420 = stablehlo.transpose %v6418, dims = [1, 0, 2, 3] : (tensor<32x16x112x112xf32>) -> tensor<16x32x112x112xf32>
    %v6421 = stablehlo.transpose %v6419, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %v6422 = stablehlo.convolution(%v6420, %v6421)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x32x112x112xf32>, tensor<96x32x112x112xf32>) -> tensor<16x96x1x1xf32>
    %v6423 = stablehlo.transpose %v6422, dims = [1, 0, 2, 3] : (tensor<16x96x1x1xf32>) -> tensor<96x16x1x1xf32>
    %v6424 = stablehlo.constant dense<0.05> : tensor<96x16x1x1xf32>
    %v6425 = stablehlo.multiply %v6423, %v6424 : tensor<96x16x1x1xf32>
    %v6426 = stablehlo.subtract %b2eW, %v6425 : tensor<96x16x1x1xf32>
    %v6427 = stablehlo.reshape %v79 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6428 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6429 = stablehlo.constant dense<401408.0> : tensor<32x16x112x112xf32>
    %v6430 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v6431 = stablehlo.reduce(%v6427 init: %v6428) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6432 = stablehlo.broadcast_in_dim %v6431, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v6433 = stablehlo.divide %v6432, %v6429 : tensor<32x16x112x112xf32>
    %v6434 = stablehlo.subtract %v6427, %v6433 : tensor<32x16x112x112xf32>
    %v6435 = stablehlo.multiply %v6434, %v6434 : tensor<32x16x112x112xf32>
    %v6436 = stablehlo.reduce(%v6435 init: %v6428) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6437 = stablehlo.broadcast_in_dim %v6436, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v6438 = stablehlo.divide %v6437, %v6429 : tensor<32x16x112x112xf32>
    %v6439 = stablehlo.add %v6438, %v6430 : tensor<32x16x112x112xf32>
    %v6440 = stablehlo.rsqrt %v6439 : tensor<32x16x112x112xf32>
    %v6441 = stablehlo.multiply %v6434, %v6440 : tensor<32x16x112x112xf32>
    %v6442 = stablehlo.broadcast_in_dim %b1pg, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v6443 = stablehlo.reshape %v6390 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6444 = stablehlo.multiply %v6442, %v6443 : tensor<32x16x112x112xf32>
    %v6445 = stablehlo.reduce(%v6444 init: %v6428) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6446 = stablehlo.broadcast_in_dim %v6445, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v6447 = stablehlo.multiply %v6441, %v6444 : tensor<32x16x112x112xf32>
    %v6448 = stablehlo.reduce(%v6447 init: %v6428) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6449 = stablehlo.broadcast_in_dim %v6448, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v6450 = stablehlo.multiply %v6444, %v6429 : tensor<32x16x112x112xf32>
    %v6451 = stablehlo.subtract %v6450, %v6446 : tensor<32x16x112x112xf32>
    %v6452 = stablehlo.multiply %v6441, %v6449 : tensor<32x16x112x112xf32>
    %v6453 = stablehlo.subtract %v6451, %v6452 : tensor<32x16x112x112xf32>
    %v6454 = stablehlo.divide %v6440, %v6429 : tensor<32x16x112x112xf32>
    %v6455 = stablehlo.multiply %v6454, %v6453 : tensor<32x16x112x112xf32>
    %v6456 = stablehlo.reshape %v6455 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v6457 = stablehlo.reshape %v6456 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6458 = stablehlo.reverse %b1pW, dims = [2, 3] : tensor<16x32x1x1xf32>
    %v6459 = stablehlo.transpose %v6458, dims = [1, 0, 2, 3] : (tensor<16x32x1x1xf32>) -> tensor<32x16x1x1xf32>
    %v6460 = stablehlo.convolution(%v6457, %v6459)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x16x112x112xf32>, tensor<32x16x1x1xf32>) -> tensor<32x32x112x112xf32>
    %v6461 = stablehlo.reshape %v6460 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v6462 = stablehlo.reshape %v79 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6463 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6464 = stablehlo.constant dense<401408.0> : tensor<32x16x112x112xf32>
    %v6465 = stablehlo.reduce(%v6462 init: %v6463) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6466 = stablehlo.broadcast_in_dim %v6465, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v6467 = stablehlo.divide %v6466, %v6464 : tensor<32x16x112x112xf32>
    %v6468 = stablehlo.subtract %v6462, %v6467 : tensor<32x16x112x112xf32>
    %v6469 = stablehlo.multiply %v6468, %v6468 : tensor<32x16x112x112xf32>
    %v6470 = stablehlo.reduce(%v6469 init: %v6463) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6471 = stablehlo.broadcast_in_dim %v6470, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v6472 = stablehlo.divide %v6471, %v6464 : tensor<32x16x112x112xf32>
    %v6473 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v6474 = stablehlo.add %v6472, %v6473 : tensor<32x16x112x112xf32>
    %v6475 = stablehlo.rsqrt %v6474 : tensor<32x16x112x112xf32>
    %v6476 = stablehlo.multiply %v6468, %v6475 : tensor<32x16x112x112xf32>
    %v6477 = stablehlo.reshape %v6390 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6478 = stablehlo.multiply %v6477, %v6476 : tensor<32x16x112x112xf32>
    %v6479 = stablehlo.reduce(%v6478 init: %v6463) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6480 = stablehlo.constant dense<0.05> : tensor<16xf32>
    %v6481 = stablehlo.multiply %v6479, %v6480 : tensor<16xf32>
    %v6482 = stablehlo.subtract %b1pg, %v6481 : tensor<16xf32>
    %v6483 = stablehlo.reshape %v6390 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6484 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6485 = stablehlo.reduce(%v6483 init: %v6484) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6486 = stablehlo.constant dense<0.05> : tensor<16xf32>
    %v6487 = stablehlo.multiply %v6485, %v6486 : tensor<16xf32>
    %v6488 = stablehlo.subtract %b1pbt, %v6487 : tensor<16xf32>
    %v6489 = stablehlo.reshape %v74 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6490 = stablehlo.reshape %v6456 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6491 = stablehlo.transpose %v6489, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %v6492 = stablehlo.transpose %v6490, dims = [1, 0, 2, 3] : (tensor<32x16x112x112xf32>) -> tensor<16x32x112x112xf32>
    %v6493 = stablehlo.convolution(%v6491, %v6492)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<16x32x112x112xf32>) -> tensor<32x16x1x1xf32>
    %v6494 = stablehlo.transpose %v6493, dims = [1, 0, 2, 3] : (tensor<32x16x1x1xf32>) -> tensor<16x32x1x1xf32>
    %v6495 = stablehlo.constant dense<0.05> : tensor<16x32x1x1xf32>
    %v6496 = stablehlo.multiply %v6494, %v6495 : tensor<16x32x1x1xf32>
    %v6497 = stablehlo.subtract %b1pW, %v6496 : tensor<16x32x1x1xf32>
    %v6498 = stablehlo.reshape %v57 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6499 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6500 = stablehlo.reduce(%v6498 init: %v6499) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v6501 = stablehlo.constant dense<12544.0> : tensor<32x32xf32>
    %v6502 = stablehlo.divide %v6500, %v6501 : tensor<32x32xf32>
    %v6503 = stablehlo.dot_general %v6502, %b1zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x8xf32>) -> tensor<32x8xf32>
    %v6504 = stablehlo.broadcast_in_dim %b1zb1, dims = [1] : (tensor<8xf32>) -> tensor<32x8xf32>
    %v6505 = stablehlo.add %v6503, %v6504 : tensor<32x8xf32>
    %v6506 = stablehlo.logistic %v6505 : tensor<32x8xf32>
    %v6507 = stablehlo.multiply %v6505, %v6506 : tensor<32x8xf32>
    %v6508 = stablehlo.dot_general %v6507, %b1zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x8xf32>, tensor<8x32xf32>) -> tensor<32x32xf32>
    %v6509 = stablehlo.broadcast_in_dim %b1zb2, dims = [1] : (tensor<32xf32>) -> tensor<32x32xf32>
    %v6510 = stablehlo.add %v6508, %v6509 : tensor<32x32xf32>
    %v6511 = stablehlo.logistic %v6510 : tensor<32x32xf32>
    %v6512 = stablehlo.reshape %v6461 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6513 = stablehlo.broadcast_in_dim %v6511, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v6514 = stablehlo.multiply %v6513, %v6512 : tensor<32x32x112x112xf32>
    %v6515 = stablehlo.multiply %v6498, %v6512 : tensor<32x32x112x112xf32>
    %v6516 = stablehlo.reduce(%v6515 init: %v6499) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v6517 = stablehlo.constant dense<1.0> : tensor<32x32xf32>
    %v6518 = stablehlo.subtract %v6517, %v6511 : tensor<32x32xf32>
    %v6519 = stablehlo.multiply %v6511, %v6518 : tensor<32x32xf32>
    %v6520 = stablehlo.multiply %v6516, %v6519 : tensor<32x32xf32>
    %v6521 = stablehlo.dot_general %v6520, %b1zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<8x32xf32>) -> tensor<32x8xf32>
    %v6522 = stablehlo.logistic %v6505 : tensor<32x8xf32>
    %v6523 = stablehlo.constant dense<1.0> : tensor<32x8xf32>
    %v6524 = stablehlo.subtract %v6523, %v6522 : tensor<32x8xf32>
    %v6525 = stablehlo.multiply %v6505, %v6524 : tensor<32x8xf32>
    %v6526 = stablehlo.add %v6523, %v6525 : tensor<32x8xf32>
    %v6527 = stablehlo.multiply %v6522, %v6526 : tensor<32x8xf32>
    %v6528 = stablehlo.multiply %v6521, %v6527 : tensor<32x8xf32>
    %v6529 = stablehlo.dot_general %v6528, %b1zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x8xf32>, tensor<32x8xf32>) -> tensor<32x32xf32>
    %v6530 = stablehlo.constant dense<12544.0> : tensor<32x32xf32>
    %v6531 = stablehlo.divide %v6529, %v6530 : tensor<32x32xf32>
    %v6532 = stablehlo.broadcast_in_dim %v6531, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v6533 = stablehlo.add %v6514, %v6532 : tensor<32x32x112x112xf32>
    %v6534 = stablehlo.reshape %v6533 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v6535 = stablehlo.reshape %v57 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6536 = stablehlo.reshape %v6461 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6537 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6538 = stablehlo.multiply %v6535, %v6536 : tensor<32x32x112x112xf32>
    %v6539 = stablehlo.reduce(%v6538 init: %v6537) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v6540 = stablehlo.logistic %v70 : tensor<32x32xf32>
    %v6541 = stablehlo.constant dense<1.0> : tensor<32x32xf32>
    %v6542 = stablehlo.subtract %v6541, %v6540 : tensor<32x32xf32>
    %v6543 = stablehlo.multiply %v6540, %v6542 : tensor<32x32xf32>
    %v6544 = stablehlo.multiply %v6539, %v6543 : tensor<32x32xf32>
    %v6545 = stablehlo.dot_general %v67, %v6544, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x8xf32>, tensor<32x32xf32>) -> tensor<8x32xf32>
    %v6546 = stablehlo.constant dense<0.05> : tensor<8x32xf32>
    %v6547 = stablehlo.multiply %v6545, %v6546 : tensor<8x32xf32>
    %v6548 = stablehlo.subtract %b1zW2, %v6547 : tensor<8x32xf32>
    %v6549 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6550 = stablehlo.reduce(%v6544 init: %v6549) applies stablehlo.add across dimensions = [0] : (tensor<32x32xf32>, tensor<f32>) -> tensor<32xf32>
    %v6551 = stablehlo.constant dense<0.05> : tensor<32xf32>
    %v6552 = stablehlo.multiply %v6550, %v6551 : tensor<32xf32>
    %v6553 = stablehlo.subtract %b1zb2, %v6552 : tensor<32xf32>
    %v6554 = stablehlo.reshape %v6544 : (tensor<32x32xf32>) -> tensor<32x1x32xf32>
    %v6555 = stablehlo.dot_general %v6554, %b1zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x32xf32>, tensor<8x32xf32>) -> tensor<32x1x8xf32>
    %v6556 = stablehlo.reshape %v6555 : (tensor<32x1x8xf32>) -> tensor<32x8xf32>
    %v6557 = stablehlo.logistic %v65 : tensor<32x8xf32>
    %v6558 = stablehlo.constant dense<1.0> : tensor<32x8xf32>
    %v6559 = stablehlo.subtract %v6558, %v6557 : tensor<32x8xf32>
    %v6560 = stablehlo.multiply %v65, %v6559 : tensor<32x8xf32>
    %v6561 = stablehlo.add %v6558, %v6560 : tensor<32x8xf32>
    %v6562 = stablehlo.multiply %v6557, %v6561 : tensor<32x8xf32>
    %v6563 = stablehlo.multiply %v6556, %v6562 : tensor<32x8xf32>
    %v6564 = stablehlo.dot_general %v62, %v6563, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x8xf32>) -> tensor<32x8xf32>
    %v6565 = stablehlo.constant dense<0.05> : tensor<32x8xf32>
    %v6566 = stablehlo.multiply %v6564, %v6565 : tensor<32x8xf32>
    %v6567 = stablehlo.subtract %b1zW1, %v6566 : tensor<32x8xf32>
    %v6568 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6569 = stablehlo.reduce(%v6563 init: %v6568) applies stablehlo.add across dimensions = [0] : (tensor<32x8xf32>, tensor<f32>) -> tensor<8xf32>
    %v6570 = stablehlo.constant dense<0.05> : tensor<8xf32>
    %v6571 = stablehlo.multiply %v6569, %v6570 : tensor<8xf32>
    %v6572 = stablehlo.subtract %b1zb1, %v6571 : tensor<8xf32>
    %v6573 = stablehlo.reshape %v6534 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6574 = stablehlo.reshape %v53 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6575 = stablehlo.logistic %v6574 : tensor<32x32x112x112xf32>
    %v6576 = stablehlo.constant dense<1.0> : tensor<32x32x112x112xf32>
    %v6577 = stablehlo.subtract %v6576, %v6575 : tensor<32x32x112x112xf32>
    %v6578 = stablehlo.multiply %v6574, %v6577 : tensor<32x32x112x112xf32>
    %v6579 = stablehlo.add %v6576, %v6578 : tensor<32x32x112x112xf32>
    %v6580 = stablehlo.multiply %v6575, %v6579 : tensor<32x32x112x112xf32>
    %v6581 = stablehlo.multiply %v6573, %v6580 : tensor<32x32x112x112xf32>
    %v6582 = stablehlo.reshape %v6581 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v6583 = stablehlo.reshape %v33 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6584 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6585 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v6586 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v6587 = stablehlo.reduce(%v6583 init: %v6584) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6588 = stablehlo.broadcast_in_dim %v6587, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6589 = stablehlo.divide %v6588, %v6585 : tensor<32x32x112x112xf32>
    %v6590 = stablehlo.subtract %v6583, %v6589 : tensor<32x32x112x112xf32>
    %v6591 = stablehlo.multiply %v6590, %v6590 : tensor<32x32x112x112xf32>
    %v6592 = stablehlo.reduce(%v6591 init: %v6584) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6593 = stablehlo.broadcast_in_dim %v6592, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6594 = stablehlo.divide %v6593, %v6585 : tensor<32x32x112x112xf32>
    %v6595 = stablehlo.add %v6594, %v6586 : tensor<32x32x112x112xf32>
    %v6596 = stablehlo.rsqrt %v6595 : tensor<32x32x112x112xf32>
    %v6597 = stablehlo.multiply %v6590, %v6596 : tensor<32x32x112x112xf32>
    %v6598 = stablehlo.broadcast_in_dim %b1dg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6599 = stablehlo.reshape %v6582 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6600 = stablehlo.multiply %v6598, %v6599 : tensor<32x32x112x112xf32>
    %v6601 = stablehlo.reduce(%v6600 init: %v6584) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6602 = stablehlo.broadcast_in_dim %v6601, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6603 = stablehlo.multiply %v6597, %v6600 : tensor<32x32x112x112xf32>
    %v6604 = stablehlo.reduce(%v6603 init: %v6584) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6605 = stablehlo.broadcast_in_dim %v6604, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6606 = stablehlo.multiply %v6600, %v6585 : tensor<32x32x112x112xf32>
    %v6607 = stablehlo.subtract %v6606, %v6602 : tensor<32x32x112x112xf32>
    %v6608 = stablehlo.multiply %v6597, %v6605 : tensor<32x32x112x112xf32>
    %v6609 = stablehlo.subtract %v6607, %v6608 : tensor<32x32x112x112xf32>
    %v6610 = stablehlo.divide %v6596, %v6585 : tensor<32x32x112x112xf32>
    %v6611 = stablehlo.multiply %v6610, %v6609 : tensor<32x32x112x112xf32>
    %v6612 = stablehlo.reshape %v6611 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v6613 = stablehlo.reshape %v6612 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6614 = stablehlo.reverse %b1dW, dims = [2, 3] : tensor<32x1x3x3xf32>
    %v6615 = stablehlo.convolution(%v6613, %v6614)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 32 : i64} : (tensor<32x32x112x112xf32>, tensor<32x1x3x3xf32>) -> tensor<32x32x112x112xf32>
    %v6616 = stablehlo.reshape %v6615 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v6617 = stablehlo.reshape %v33 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6618 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6619 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v6620 = stablehlo.reduce(%v6617 init: %v6618) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6621 = stablehlo.broadcast_in_dim %v6620, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6622 = stablehlo.divide %v6621, %v6619 : tensor<32x32x112x112xf32>
    %v6623 = stablehlo.subtract %v6617, %v6622 : tensor<32x32x112x112xf32>
    %v6624 = stablehlo.multiply %v6623, %v6623 : tensor<32x32x112x112xf32>
    %v6625 = stablehlo.reduce(%v6624 init: %v6618) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6626 = stablehlo.broadcast_in_dim %v6625, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6627 = stablehlo.divide %v6626, %v6619 : tensor<32x32x112x112xf32>
    %v6628 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v6629 = stablehlo.add %v6627, %v6628 : tensor<32x32x112x112xf32>
    %v6630 = stablehlo.rsqrt %v6629 : tensor<32x32x112x112xf32>
    %v6631 = stablehlo.multiply %v6623, %v6630 : tensor<32x32x112x112xf32>
    %v6632 = stablehlo.reshape %v6582 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6633 = stablehlo.multiply %v6632, %v6631 : tensor<32x32x112x112xf32>
    %v6634 = stablehlo.reduce(%v6633 init: %v6618) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6635 = stablehlo.constant dense<0.05> : tensor<32xf32>
    %v6636 = stablehlo.multiply %v6634, %v6635 : tensor<32xf32>
    %v6637 = stablehlo.subtract %b1dg, %v6636 : tensor<32xf32>
    %v6638 = stablehlo.reshape %v6582 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6639 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6640 = stablehlo.reduce(%v6638 init: %v6639) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6641 = stablehlo.constant dense<0.05> : tensor<32xf32>
    %v6642 = stablehlo.multiply %v6640, %v6641 : tensor<32xf32>
    %v6643 = stablehlo.subtract %b1dbt, %v6642 : tensor<32xf32>
    %v6644 = stablehlo.reshape %v28 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6645 = stablehlo.reshape %v6612 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6646 = stablehlo.transpose %v6644, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %v6647 = stablehlo.transpose %v6645, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %v6648 = stablehlo.convolution(%v6646, %v6647)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 32 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<32x32x112x112xf32>) -> tensor<1x32x3x3xf32>
    %v6649 = stablehlo.reshape %v6648 : (tensor<1x32x3x3xf32>) -> tensor<32x1x3x3xf32>
    %v6650 = stablehlo.constant dense<0.05> : tensor<32x1x3x3xf32>
    %v6651 = stablehlo.multiply %v6649, %v6650 : tensor<32x1x3x3xf32>
    %v6652 = stablehlo.subtract %b1dW, %v6651 : tensor<32x1x3x3xf32>
    %v6653 = stablehlo.reshape %v6616 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6654 = stablehlo.reshape %v24 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6655 = stablehlo.logistic %v6654 : tensor<32x32x112x112xf32>
    %v6656 = stablehlo.constant dense<1.0> : tensor<32x32x112x112xf32>
    %v6657 = stablehlo.subtract %v6656, %v6655 : tensor<32x32x112x112xf32>
    %v6658 = stablehlo.multiply %v6654, %v6657 : tensor<32x32x112x112xf32>
    %v6659 = stablehlo.add %v6656, %v6658 : tensor<32x32x112x112xf32>
    %v6660 = stablehlo.multiply %v6655, %v6659 : tensor<32x32x112x112xf32>
    %v6661 = stablehlo.multiply %v6653, %v6660 : tensor<32x32x112x112xf32>
    %v6662 = stablehlo.reshape %v6661 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v6663 = stablehlo.reshape %v4 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6664 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6665 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v6666 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v6667 = stablehlo.reduce(%v6663 init: %v6664) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6668 = stablehlo.broadcast_in_dim %v6667, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6669 = stablehlo.divide %v6668, %v6665 : tensor<32x32x112x112xf32>
    %v6670 = stablehlo.subtract %v6663, %v6669 : tensor<32x32x112x112xf32>
    %v6671 = stablehlo.multiply %v6670, %v6670 : tensor<32x32x112x112xf32>
    %v6672 = stablehlo.reduce(%v6671 init: %v6664) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6673 = stablehlo.broadcast_in_dim %v6672, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6674 = stablehlo.divide %v6673, %v6665 : tensor<32x32x112x112xf32>
    %v6675 = stablehlo.add %v6674, %v6666 : tensor<32x32x112x112xf32>
    %v6676 = stablehlo.rsqrt %v6675 : tensor<32x32x112x112xf32>
    %v6677 = stablehlo.multiply %v6670, %v6676 : tensor<32x32x112x112xf32>
    %v6678 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6679 = stablehlo.reshape %v6662 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6680 = stablehlo.multiply %v6678, %v6679 : tensor<32x32x112x112xf32>
    %v6681 = stablehlo.reduce(%v6680 init: %v6664) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6682 = stablehlo.broadcast_in_dim %v6681, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6683 = stablehlo.multiply %v6677, %v6680 : tensor<32x32x112x112xf32>
    %v6684 = stablehlo.reduce(%v6683 init: %v6664) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6685 = stablehlo.broadcast_in_dim %v6684, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6686 = stablehlo.multiply %v6680, %v6665 : tensor<32x32x112x112xf32>
    %v6687 = stablehlo.subtract %v6686, %v6682 : tensor<32x32x112x112xf32>
    %v6688 = stablehlo.multiply %v6677, %v6685 : tensor<32x32x112x112xf32>
    %v6689 = stablehlo.subtract %v6687, %v6688 : tensor<32x32x112x112xf32>
    %v6690 = stablehlo.divide %v6676, %v6665 : tensor<32x32x112x112xf32>
    %v6691 = stablehlo.multiply %v6690, %v6689 : tensor<32x32x112x112xf32>
    %v6692 = stablehlo.reshape %v6691 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v6693 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v6694 = stablehlo.reshape %v6692 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6695 = stablehlo.transpose %v6693, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %v6697 = stablehlo.transpose %v6694, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %v6698 = stablehlo.convolution(%v6695, %v6697)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<32x32x112x112xf32>) -> tensor<3x32x3x3xf32>
    %v6699 = stablehlo.transpose %v6698, dims = [1, 0, 2, 3] : (tensor<3x32x3x3xf32>) -> tensor<32x3x3x3xf32>
    %v6700 = stablehlo.constant dense<0.05> : tensor<32x3x3x3xf32>
    %v6701 = stablehlo.multiply %v6699, %v6700 : tensor<32x3x3x3xf32>
    %v6702 = stablehlo.subtract %sW, %v6701 : tensor<32x3x3x3xf32>
    %v6703 = stablehlo.reshape %v4 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6704 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6705 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v6706 = stablehlo.reduce(%v6703 init: %v6704) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6707 = stablehlo.broadcast_in_dim %v6706, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6708 = stablehlo.divide %v6707, %v6705 : tensor<32x32x112x112xf32>
    %v6709 = stablehlo.subtract %v6703, %v6708 : tensor<32x32x112x112xf32>
    %v6710 = stablehlo.multiply %v6709, %v6709 : tensor<32x32x112x112xf32>
    %v6711 = stablehlo.reduce(%v6710 init: %v6704) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6712 = stablehlo.broadcast_in_dim %v6711, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6713 = stablehlo.divide %v6712, %v6705 : tensor<32x32x112x112xf32>
    %v6714 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v6715 = stablehlo.add %v6713, %v6714 : tensor<32x32x112x112xf32>
    %v6716 = stablehlo.rsqrt %v6715 : tensor<32x32x112x112xf32>
    %v6717 = stablehlo.multiply %v6709, %v6716 : tensor<32x32x112x112xf32>
    %v6718 = stablehlo.reshape %v6662 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6719 = stablehlo.multiply %v6718, %v6717 : tensor<32x32x112x112xf32>
    %v6720 = stablehlo.reduce(%v6719 init: %v6704) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6721 = stablehlo.constant dense<0.05> : tensor<32xf32>
    %v6722 = stablehlo.multiply %v6720, %v6721 : tensor<32xf32>
    %v6723 = stablehlo.subtract %sg, %v6722 : tensor<32xf32>
    %v6724 = stablehlo.reshape %v6662 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6725 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6726 = stablehlo.reduce(%v6724 init: %v6725) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6727 = stablehlo.constant dense<0.05> : tensor<32xf32>
    %v6728 = stablehlo.multiply %v6726, %v6727 : tensor<32xf32>
    %v6729 = stablehlo.subtract %sbt, %v6728 : tensor<32xf32>
    return %v6702, %v6723, %v6729, %v6652, %v6637, %v6643, %v6567, %v6572, %v6548, %v6553, %v6497, %v6482, %v6488, %v6426, %v6411, %v6417, %v6345, %v6328, %v6334, %v6258, %v6263, %v6239, %v6244, %v6188, %v6173, %v6179, %v6113, %v6098, %v6104, %v6032, %v6017, %v6023, %v5947, %v5952, %v5928, %v5933, %v5877, %v5862, %v5868, %v5806, %v5791, %v5797, %v5725, %v5708, %v5714, %v5638, %v5643, %v5619, %v5624, %v5568, %v5553, %v5559, %v5493, %v5478, %v5484, %v5412, %v5397, %v5403, %v5327, %v5332, %v5308, %v5313, %v5257, %v5242, %v5248, %v5186, %v5171, %v5177, %v5105, %v5088, %v5094, %v5018, %v5023, %v4999, %v5004, %v4948, %v4933, %v4939, %v4873, %v4858, %v4864, %v4792, %v4777, %v4783, %v4707, %v4712, %v4688, %v4693, %v4637, %v4622, %v4628, %v4562, %v4547, %v4553, %v4481, %v4466, %v4472, %v4396, %v4401, %v4377, %v4382, %v4326, %v4311, %v4317, %v4255, %v4240, %v4246, %v4174, %v4159, %v4165, %v4089, %v4094, %v4070, %v4075, %v4019, %v4004, %v4010, %v3944, %v3929, %v3935, %v3863, %v3848, %v3854, %v3778, %v3783, %v3759, %v3764, %v3708, %v3693, %v3699, %v3633, %v3618, %v3624, %v3552, %v3537, %v3543, %v3467, %v3472, %v3448, %v3453, %v3397, %v3382, %v3388, %v3326, %v3311, %v3317, %v3245, %v3228, %v3234, %v3158, %v3163, %v3139, %v3144, %v3088, %v3073, %v3079, %v3013, %v2998, %v3004, %v2932, %v2917, %v2923, %v2847, %v2852, %v2828, %v2833, %v2777, %v2762, %v2768, %v2702, %v2687, %v2693, %v2621, %v2606, %v2612, %v2536, %v2541, %v2517, %v2522, %v2466, %v2451, %v2457, %v2391, %v2376, %v2382, %v2310, %v2295, %v2301, %v2225, %v2230, %v2206, %v2211, %v2155, %v2140, %v2146, %v2084, %v2069, %v2075, %v2003, %v1988, %v1994, %v1918, %v1923, %v1899, %v1904, %v1848, %v1833, %v1839, %v1777, %v1762, %v1768, %v1687, %v1692 : tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x8xf32>, tensor<8xf32>, tensor<8x32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x4xf32>, tensor<4xf32>, tensor<4x96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x6xf32>, tensor<6xf32>, tensor<6x144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x5x5xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x6xf32>, tensor<6xf32>, tensor<6x144xf32>, tensor<144xf32>, tensor<40x144x1x1xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240x40x1x1xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x1x5x5xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x10xf32>, tensor<10xf32>, tensor<10x240xf32>, tensor<240xf32>, tensor<40x240x1x1xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240x40x1x1xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x1x3x3xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x10xf32>, tensor<10xf32>, tensor<10x240xf32>, tensor<240xf32>, tensor<80x240x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x3x3xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<80x480x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x3x3xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<80x480x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x5x5xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<112x480x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<112x672x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<112x672x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<192x672x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x3x3xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<320x1152x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x10xf32>, tensor<10xf32>
  }
}
