module @m {
  func.func @efficientnet_train_step(%x: tensor<32x150528xf32>, %sW: tensor<32x3x3x3xf32>, %sg: tensor<32xf32>, %sbt: tensor<32xf32>, %b1dW: tensor<32x1x3x3xf32>, %b1dg: tensor<32xf32>, %b1dbt: tensor<32xf32>, %b1zW1: tensor<32x8xf32>, %b1zb1: tensor<8xf32>, %b1zW2: tensor<8x32xf32>, %b1zb2: tensor<32xf32>, %b1pW: tensor<16x32x1x1xf32>, %b1pg: tensor<16xf32>, %b1pbt: tensor<16xf32>, %b2eW: tensor<96x16x1x1xf32>, %b2eg: tensor<96xf32>, %b2ebt: tensor<96xf32>, %b2dW: tensor<96x1x3x3xf32>, %b2dg: tensor<96xf32>, %b2dbt: tensor<96xf32>, %b2zW1: tensor<96x4xf32>, %b2zb1: tensor<4xf32>, %b2zW2: tensor<4x96xf32>, %b2zb2: tensor<96xf32>, %b2pW: tensor<24x96x1x1xf32>, %b2pg: tensor<24xf32>, %b2pbt: tensor<24xf32>, %b3eW: tensor<144x24x1x1xf32>, %b3eg: tensor<144xf32>, %b3ebt: tensor<144xf32>, %b3dW: tensor<144x1x3x3xf32>, %b3dg: tensor<144xf32>, %b3dbt: tensor<144xf32>, %b3zW1: tensor<144x6xf32>, %b3zb1: tensor<6xf32>, %b3zW2: tensor<6x144xf32>, %b3zb2: tensor<144xf32>, %b3pW: tensor<24x144x1x1xf32>, %b3pg: tensor<24xf32>, %b3pbt: tensor<24xf32>, %b4eW: tensor<144x24x1x1xf32>, %b4eg: tensor<144xf32>, %b4ebt: tensor<144xf32>, %b4dW: tensor<144x1x5x5xf32>, %b4dg: tensor<144xf32>, %b4dbt: tensor<144xf32>, %b4zW1: tensor<144x6xf32>, %b4zb1: tensor<6xf32>, %b4zW2: tensor<6x144xf32>, %b4zb2: tensor<144xf32>, %b4pW: tensor<40x144x1x1xf32>, %b4pg: tensor<40xf32>, %b4pbt: tensor<40xf32>, %b5eW: tensor<240x40x1x1xf32>, %b5eg: tensor<240xf32>, %b5ebt: tensor<240xf32>, %b5dW: tensor<240x1x5x5xf32>, %b5dg: tensor<240xf32>, %b5dbt: tensor<240xf32>, %b5zW1: tensor<240x10xf32>, %b5zb1: tensor<10xf32>, %b5zW2: tensor<10x240xf32>, %b5zb2: tensor<240xf32>, %b5pW: tensor<40x240x1x1xf32>, %b5pg: tensor<40xf32>, %b5pbt: tensor<40xf32>, %b6eW: tensor<240x40x1x1xf32>, %b6eg: tensor<240xf32>, %b6ebt: tensor<240xf32>, %b6dW: tensor<240x1x3x3xf32>, %b6dg: tensor<240xf32>, %b6dbt: tensor<240xf32>, %b6zW1: tensor<240x10xf32>, %b6zb1: tensor<10xf32>, %b6zW2: tensor<10x240xf32>, %b6zb2: tensor<240xf32>, %b6pW: tensor<80x240x1x1xf32>, %b6pg: tensor<80xf32>, %b6pbt: tensor<80xf32>, %b7eW: tensor<480x80x1x1xf32>, %b7eg: tensor<480xf32>, %b7ebt: tensor<480xf32>, %b7dW: tensor<480x1x3x3xf32>, %b7dg: tensor<480xf32>, %b7dbt: tensor<480xf32>, %b7zW1: tensor<480x20xf32>, %b7zb1: tensor<20xf32>, %b7zW2: tensor<20x480xf32>, %b7zb2: tensor<480xf32>, %b7pW: tensor<80x480x1x1xf32>, %b7pg: tensor<80xf32>, %b7pbt: tensor<80xf32>, %b8eW: tensor<480x80x1x1xf32>, %b8eg: tensor<480xf32>, %b8ebt: tensor<480xf32>, %b8dW: tensor<480x1x3x3xf32>, %b8dg: tensor<480xf32>, %b8dbt: tensor<480xf32>, %b8zW1: tensor<480x20xf32>, %b8zb1: tensor<20xf32>, %b8zW2: tensor<20x480xf32>, %b8zb2: tensor<480xf32>, %b8pW: tensor<80x480x1x1xf32>, %b8pg: tensor<80xf32>, %b8pbt: tensor<80xf32>, %b9eW: tensor<480x80x1x1xf32>, %b9eg: tensor<480xf32>, %b9ebt: tensor<480xf32>, %b9dW: tensor<480x1x5x5xf32>, %b9dg: tensor<480xf32>, %b9dbt: tensor<480xf32>, %b9zW1: tensor<480x20xf32>, %b9zb1: tensor<20xf32>, %b9zW2: tensor<20x480xf32>, %b9zb2: tensor<480xf32>, %b9pW: tensor<112x480x1x1xf32>, %b9pg: tensor<112xf32>, %b9pbt: tensor<112xf32>, %b10eW: tensor<672x112x1x1xf32>, %b10eg: tensor<672xf32>, %b10ebt: tensor<672xf32>, %b10dW: tensor<672x1x5x5xf32>, %b10dg: tensor<672xf32>, %b10dbt: tensor<672xf32>, %b10zW1: tensor<672x28xf32>, %b10zb1: tensor<28xf32>, %b10zW2: tensor<28x672xf32>, %b10zb2: tensor<672xf32>, %b10pW: tensor<112x672x1x1xf32>, %b10pg: tensor<112xf32>, %b10pbt: tensor<112xf32>, %b11eW: tensor<672x112x1x1xf32>, %b11eg: tensor<672xf32>, %b11ebt: tensor<672xf32>, %b11dW: tensor<672x1x5x5xf32>, %b11dg: tensor<672xf32>, %b11dbt: tensor<672xf32>, %b11zW1: tensor<672x28xf32>, %b11zb1: tensor<28xf32>, %b11zW2: tensor<28x672xf32>, %b11zb2: tensor<672xf32>, %b11pW: tensor<112x672x1x1xf32>, %b11pg: tensor<112xf32>, %b11pbt: tensor<112xf32>, %b12eW: tensor<672x112x1x1xf32>, %b12eg: tensor<672xf32>, %b12ebt: tensor<672xf32>, %b12dW: tensor<672x1x5x5xf32>, %b12dg: tensor<672xf32>, %b12dbt: tensor<672xf32>, %b12zW1: tensor<672x28xf32>, %b12zb1: tensor<28xf32>, %b12zW2: tensor<28x672xf32>, %b12zb2: tensor<672xf32>, %b12pW: tensor<192x672x1x1xf32>, %b12pg: tensor<192xf32>, %b12pbt: tensor<192xf32>, %b13eW: tensor<1152x192x1x1xf32>, %b13eg: tensor<1152xf32>, %b13ebt: tensor<1152xf32>, %b13dW: tensor<1152x1x5x5xf32>, %b13dg: tensor<1152xf32>, %b13dbt: tensor<1152xf32>, %b13zW1: tensor<1152x48xf32>, %b13zb1: tensor<48xf32>, %b13zW2: tensor<48x1152xf32>, %b13zb2: tensor<1152xf32>, %b13pW: tensor<192x1152x1x1xf32>, %b13pg: tensor<192xf32>, %b13pbt: tensor<192xf32>, %b14eW: tensor<1152x192x1x1xf32>, %b14eg: tensor<1152xf32>, %b14ebt: tensor<1152xf32>, %b14dW: tensor<1152x1x5x5xf32>, %b14dg: tensor<1152xf32>, %b14dbt: tensor<1152xf32>, %b14zW1: tensor<1152x48xf32>, %b14zb1: tensor<48xf32>, %b14zW2: tensor<48x1152xf32>, %b14zb2: tensor<1152xf32>, %b14pW: tensor<192x1152x1x1xf32>, %b14pg: tensor<192xf32>, %b14pbt: tensor<192xf32>, %b15eW: tensor<1152x192x1x1xf32>, %b15eg: tensor<1152xf32>, %b15ebt: tensor<1152xf32>, %b15dW: tensor<1152x1x5x5xf32>, %b15dg: tensor<1152xf32>, %b15dbt: tensor<1152xf32>, %b15zW1: tensor<1152x48xf32>, %b15zb1: tensor<48xf32>, %b15zW2: tensor<48x1152xf32>, %b15zb2: tensor<1152xf32>, %b15pW: tensor<192x1152x1x1xf32>, %b15pg: tensor<192xf32>, %b15pbt: tensor<192xf32>, %b16eW: tensor<1152x192x1x1xf32>, %b16eg: tensor<1152xf32>, %b16ebt: tensor<1152xf32>, %b16dW: tensor<1152x1x3x3xf32>, %b16dg: tensor<1152xf32>, %b16dbt: tensor<1152xf32>, %b16zW1: tensor<1152x48xf32>, %b16zb1: tensor<48xf32>, %b16zW2: tensor<48x1152xf32>, %b16zb2: tensor<1152xf32>, %b16pW: tensor<320x1152x1x1xf32>, %b16pg: tensor<320xf32>, %b16pbt: tensor<320xf32>, %hW: tensor<1280x320x1x1xf32>, %hg: tensor<1280xf32>, %hbt: tensor<1280xf32>, %Wd: tensor<1280x10xf32>, %bd: tensor<10xf32>, %onehot: tensor<32x10xf32>) -> (tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x8xf32>, tensor<8xf32>, tensor<8x32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x4xf32>, tensor<4xf32>, tensor<4x96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x6xf32>, tensor<6xf32>, tensor<6x144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x5x5xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x6xf32>, tensor<6xf32>, tensor<6x144xf32>, tensor<144xf32>, tensor<40x144x1x1xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240x40x1x1xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x1x5x5xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x10xf32>, tensor<10xf32>, tensor<10x240xf32>, tensor<240xf32>, tensor<40x240x1x1xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240x40x1x1xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x1x3x3xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x10xf32>, tensor<10xf32>, tensor<10x240xf32>, tensor<240xf32>, tensor<80x240x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x3x3xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<80x480x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x3x3xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<80x480x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x5x5xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<112x480x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<112x672x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<112x672x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<192x672x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x3x3xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<320x1152x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x10xf32>, tensor<10xf32>) {
    // ── EfficientNet-B0 (16-MBConv) train step: every line is pretty(verified AST node) ──
    // §2l step B: the conv biases are gone from the signature (BN removes them; He et al.'s
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
    %v71 = stablehlo.reshape %v57 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v72 = stablehlo.constant dense<0.0> : tensor<f32>
    %v73 = stablehlo.reduce(%v71 init: %v72) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v74 = stablehlo.constant dense<12544.0> : tensor<32x32xf32>
    %v75 = stablehlo.divide %v73, %v74 : tensor<32x32xf32>
    %v76 = stablehlo.dot_general %v75, %b1zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x8xf32>) -> tensor<32x8xf32>
    %v77 = stablehlo.broadcast_in_dim %b1zb1, dims = [1] : (tensor<8xf32>) -> tensor<32x8xf32>
    %v78 = stablehlo.add %v76, %v77 : tensor<32x8xf32>
    %v79 = stablehlo.logistic %v78 : tensor<32x8xf32>
    %v80 = stablehlo.multiply %v78, %v79 : tensor<32x8xf32>
    %v81 = stablehlo.dot_general %v80, %b1zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x8xf32>, tensor<8x32xf32>) -> tensor<32x32xf32>
    %v82 = stablehlo.broadcast_in_dim %b1zb2, dims = [1] : (tensor<32xf32>) -> tensor<32x32xf32>
    %v83 = stablehlo.add %v81, %v82 : tensor<32x32xf32>
    %v84 = stablehlo.logistic %v83 : tensor<32x32xf32>
    %v85 = stablehlo.broadcast_in_dim %v84, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v86 = stablehlo.multiply %v71, %v85 : tensor<32x32x112x112xf32>
    %v87 = stablehlo.reshape %v86 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v88 = stablehlo.reshape %v87 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v89 = stablehlo.convolution(%v88, %b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<16x32x1x1xf32>) -> tensor<32x16x112x112xf32>
    %v90 = stablehlo.broadcast_in_dim %zb16, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v91 = stablehlo.add %v89, %v90 : tensor<32x16x112x112xf32>
    %v92 = stablehlo.reshape %v91 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v93 = stablehlo.reshape %v92 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v94 = stablehlo.constant dense<0.0> : tensor<f32>
    %v95 = stablehlo.constant dense<401408.0> : tensor<32x16x112x112xf32>
    %v96 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v97 = stablehlo.reduce(%v93 init: %v94) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v98 = stablehlo.broadcast_in_dim %v97, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v99 = stablehlo.divide %v98, %v95 : tensor<32x16x112x112xf32>
    %v100 = stablehlo.subtract %v93, %v99 : tensor<32x16x112x112xf32>
    %v101 = stablehlo.multiply %v100, %v100 : tensor<32x16x112x112xf32>
    %v102 = stablehlo.reduce(%v101 init: %v94) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v103 = stablehlo.broadcast_in_dim %v102, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v104 = stablehlo.divide %v103, %v95 : tensor<32x16x112x112xf32>
    %v105 = stablehlo.add %v104, %v96 : tensor<32x16x112x112xf32>
    %v106 = stablehlo.rsqrt %v105 : tensor<32x16x112x112xf32>
    %v107 = stablehlo.multiply %v100, %v106 : tensor<32x16x112x112xf32>
    %v108 = stablehlo.broadcast_in_dim %b1pg, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v109 = stablehlo.broadcast_in_dim %b1pbt, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v110 = stablehlo.multiply %v107, %v108 : tensor<32x16x112x112xf32>
    %v111 = stablehlo.add %v110, %v109 : tensor<32x16x112x112xf32>
    %v112 = stablehlo.reshape %v111 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v113 = stablehlo.reshape %v112 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v114 = stablehlo.convolution(%v113, %b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x16x112x112xf32>, tensor<96x16x1x1xf32>) -> tensor<32x96x112x112xf32>
    %v115 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v116 = stablehlo.add %v114, %v115 : tensor<32x96x112x112xf32>
    %v117 = stablehlo.reshape %v116 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v118 = stablehlo.reshape %v117 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v119 = stablehlo.constant dense<0.0> : tensor<f32>
    %v120 = stablehlo.constant dense<401408.0> : tensor<32x96x112x112xf32>
    %v121 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v122 = stablehlo.reduce(%v118 init: %v119) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v123 = stablehlo.broadcast_in_dim %v122, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v124 = stablehlo.divide %v123, %v120 : tensor<32x96x112x112xf32>
    %v125 = stablehlo.subtract %v118, %v124 : tensor<32x96x112x112xf32>
    %v126 = stablehlo.multiply %v125, %v125 : tensor<32x96x112x112xf32>
    %v127 = stablehlo.reduce(%v126 init: %v119) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v128 = stablehlo.broadcast_in_dim %v127, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v129 = stablehlo.divide %v128, %v120 : tensor<32x96x112x112xf32>
    %v130 = stablehlo.add %v129, %v121 : tensor<32x96x112x112xf32>
    %v131 = stablehlo.rsqrt %v130 : tensor<32x96x112x112xf32>
    %v132 = stablehlo.multiply %v125, %v131 : tensor<32x96x112x112xf32>
    %v133 = stablehlo.broadcast_in_dim %b2eg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v134 = stablehlo.broadcast_in_dim %b2ebt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v135 = stablehlo.multiply %v132, %v133 : tensor<32x96x112x112xf32>
    %v136 = stablehlo.add %v135, %v134 : tensor<32x96x112x112xf32>
    %v137 = stablehlo.reshape %v136 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v138 = stablehlo.reshape %v137 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v139 = stablehlo.logistic %v138 : tensor<32x96x112x112xf32>
    %v140 = stablehlo.multiply %v138, %v139 : tensor<32x96x112x112xf32>
    %v141 = stablehlo.reshape %v140 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v142 = stablehlo.reshape %v141 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v143 = stablehlo.convolution(%v142, %b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x112x112xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x56x56xf32>
    %v144 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v145 = stablehlo.add %v143, %v144 : tensor<32x96x56x56xf32>
    %v146 = stablehlo.reshape %v145 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v147 = stablehlo.reshape %v146 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v148 = stablehlo.constant dense<0.0> : tensor<f32>
    %v149 = stablehlo.constant dense<100352.0> : tensor<32x96x56x56xf32>
    %v150 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v151 = stablehlo.reduce(%v147 init: %v148) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v152 = stablehlo.broadcast_in_dim %v151, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v153 = stablehlo.divide %v152, %v149 : tensor<32x96x56x56xf32>
    %v154 = stablehlo.subtract %v147, %v153 : tensor<32x96x56x56xf32>
    %v155 = stablehlo.multiply %v154, %v154 : tensor<32x96x56x56xf32>
    %v156 = stablehlo.reduce(%v155 init: %v148) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v157 = stablehlo.broadcast_in_dim %v156, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v158 = stablehlo.divide %v157, %v149 : tensor<32x96x56x56xf32>
    %v159 = stablehlo.add %v158, %v150 : tensor<32x96x56x56xf32>
    %v160 = stablehlo.rsqrt %v159 : tensor<32x96x56x56xf32>
    %v161 = stablehlo.multiply %v154, %v160 : tensor<32x96x56x56xf32>
    %v162 = stablehlo.broadcast_in_dim %b2dg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v163 = stablehlo.broadcast_in_dim %b2dbt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v164 = stablehlo.multiply %v161, %v162 : tensor<32x96x56x56xf32>
    %v165 = stablehlo.add %v164, %v163 : tensor<32x96x56x56xf32>
    %v166 = stablehlo.reshape %v165 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v167 = stablehlo.reshape %v166 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v168 = stablehlo.logistic %v167 : tensor<32x96x56x56xf32>
    %v169 = stablehlo.multiply %v167, %v168 : tensor<32x96x56x56xf32>
    %v170 = stablehlo.reshape %v169 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v171 = stablehlo.reshape %v170 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v172 = stablehlo.constant dense<0.0> : tensor<f32>
    %v173 = stablehlo.reduce(%v171 init: %v172) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v174 = stablehlo.constant dense<3136.0> : tensor<32x96xf32>
    %v175 = stablehlo.divide %v173, %v174 : tensor<32x96xf32>
    %v176 = stablehlo.dot_general %v175, %b2zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<96x4xf32>) -> tensor<32x4xf32>
    %v177 = stablehlo.broadcast_in_dim %b2zb1, dims = [1] : (tensor<4xf32>) -> tensor<32x4xf32>
    %v178 = stablehlo.add %v176, %v177 : tensor<32x4xf32>
    %v179 = stablehlo.logistic %v178 : tensor<32x4xf32>
    %v180 = stablehlo.multiply %v178, %v179 : tensor<32x4xf32>
    %v181 = stablehlo.dot_general %v180, %b2zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<4x96xf32>) -> tensor<32x96xf32>
    %v182 = stablehlo.broadcast_in_dim %b2zb2, dims = [1] : (tensor<96xf32>) -> tensor<32x96xf32>
    %v183 = stablehlo.add %v181, %v182 : tensor<32x96xf32>
    %v184 = stablehlo.reshape %v170 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v185 = stablehlo.constant dense<0.0> : tensor<f32>
    %v186 = stablehlo.reduce(%v184 init: %v185) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v187 = stablehlo.constant dense<3136.0> : tensor<32x96xf32>
    %v188 = stablehlo.divide %v186, %v187 : tensor<32x96xf32>
    %v189 = stablehlo.dot_general %v188, %b2zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<96x4xf32>) -> tensor<32x4xf32>
    %v190 = stablehlo.broadcast_in_dim %b2zb1, dims = [1] : (tensor<4xf32>) -> tensor<32x4xf32>
    %v191 = stablehlo.add %v189, %v190 : tensor<32x4xf32>
    %v192 = stablehlo.logistic %v191 : tensor<32x4xf32>
    %v193 = stablehlo.multiply %v191, %v192 : tensor<32x4xf32>
    %v194 = stablehlo.dot_general %v193, %b2zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<4x96xf32>) -> tensor<32x96xf32>
    %v195 = stablehlo.broadcast_in_dim %b2zb2, dims = [1] : (tensor<96xf32>) -> tensor<32x96xf32>
    %v196 = stablehlo.add %v194, %v195 : tensor<32x96xf32>
    %v197 = stablehlo.logistic %v196 : tensor<32x96xf32>
    %v198 = stablehlo.broadcast_in_dim %v197, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v199 = stablehlo.multiply %v184, %v198 : tensor<32x96x56x56xf32>
    %v200 = stablehlo.reshape %v199 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v201 = stablehlo.reshape %v200 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v202 = stablehlo.convolution(%v201, %b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<24x96x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v203 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v204 = stablehlo.add %v202, %v203 : tensor<32x24x56x56xf32>
    %v205 = stablehlo.reshape %v204 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v206 = stablehlo.reshape %v205 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v207 = stablehlo.constant dense<0.0> : tensor<f32>
    %v208 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v209 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v210 = stablehlo.reduce(%v206 init: %v207) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v211 = stablehlo.broadcast_in_dim %v210, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v212 = stablehlo.divide %v211, %v208 : tensor<32x24x56x56xf32>
    %v213 = stablehlo.subtract %v206, %v212 : tensor<32x24x56x56xf32>
    %v214 = stablehlo.multiply %v213, %v213 : tensor<32x24x56x56xf32>
    %v215 = stablehlo.reduce(%v214 init: %v207) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v216 = stablehlo.broadcast_in_dim %v215, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v217 = stablehlo.divide %v216, %v208 : tensor<32x24x56x56xf32>
    %v218 = stablehlo.add %v217, %v209 : tensor<32x24x56x56xf32>
    %v219 = stablehlo.rsqrt %v218 : tensor<32x24x56x56xf32>
    %v220 = stablehlo.multiply %v213, %v219 : tensor<32x24x56x56xf32>
    %v221 = stablehlo.broadcast_in_dim %b2pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v222 = stablehlo.broadcast_in_dim %b2pbt, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v223 = stablehlo.multiply %v220, %v221 : tensor<32x24x56x56xf32>
    %v224 = stablehlo.add %v223, %v222 : tensor<32x24x56x56xf32>
    %v225 = stablehlo.reshape %v224 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v226 = stablehlo.reshape %v225 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v227 = stablehlo.convolution(%v226, %b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v228 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v229 = stablehlo.add %v227, %v228 : tensor<32x144x56x56xf32>
    %v230 = stablehlo.reshape %v229 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v231 = stablehlo.reshape %v230 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v232 = stablehlo.constant dense<0.0> : tensor<f32>
    %v233 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v234 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v235 = stablehlo.reduce(%v231 init: %v232) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v236 = stablehlo.broadcast_in_dim %v235, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v237 = stablehlo.divide %v236, %v233 : tensor<32x144x56x56xf32>
    %v238 = stablehlo.subtract %v231, %v237 : tensor<32x144x56x56xf32>
    %v239 = stablehlo.multiply %v238, %v238 : tensor<32x144x56x56xf32>
    %v240 = stablehlo.reduce(%v239 init: %v232) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v241 = stablehlo.broadcast_in_dim %v240, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v242 = stablehlo.divide %v241, %v233 : tensor<32x144x56x56xf32>
    %v243 = stablehlo.add %v242, %v234 : tensor<32x144x56x56xf32>
    %v244 = stablehlo.rsqrt %v243 : tensor<32x144x56x56xf32>
    %v245 = stablehlo.multiply %v238, %v244 : tensor<32x144x56x56xf32>
    %v246 = stablehlo.broadcast_in_dim %b3eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v247 = stablehlo.broadcast_in_dim %b3ebt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v248 = stablehlo.multiply %v245, %v246 : tensor<32x144x56x56xf32>
    %v249 = stablehlo.add %v248, %v247 : tensor<32x144x56x56xf32>
    %v250 = stablehlo.reshape %v249 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v251 = stablehlo.reshape %v250 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v252 = stablehlo.logistic %v251 : tensor<32x144x56x56xf32>
    %v253 = stablehlo.multiply %v251, %v252 : tensor<32x144x56x56xf32>
    %v254 = stablehlo.reshape %v253 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v255 = stablehlo.reshape %v254 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v256 = stablehlo.convolution(%v255, %b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x56x56xf32>
    %v257 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v258 = stablehlo.add %v256, %v257 : tensor<32x144x56x56xf32>
    %v259 = stablehlo.reshape %v258 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v260 = stablehlo.reshape %v259 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v261 = stablehlo.constant dense<0.0> : tensor<f32>
    %v262 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v263 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v264 = stablehlo.reduce(%v260 init: %v261) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v265 = stablehlo.broadcast_in_dim %v264, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v266 = stablehlo.divide %v265, %v262 : tensor<32x144x56x56xf32>
    %v267 = stablehlo.subtract %v260, %v266 : tensor<32x144x56x56xf32>
    %v268 = stablehlo.multiply %v267, %v267 : tensor<32x144x56x56xf32>
    %v269 = stablehlo.reduce(%v268 init: %v261) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v270 = stablehlo.broadcast_in_dim %v269, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v271 = stablehlo.divide %v270, %v262 : tensor<32x144x56x56xf32>
    %v272 = stablehlo.add %v271, %v263 : tensor<32x144x56x56xf32>
    %v273 = stablehlo.rsqrt %v272 : tensor<32x144x56x56xf32>
    %v274 = stablehlo.multiply %v267, %v273 : tensor<32x144x56x56xf32>
    %v275 = stablehlo.broadcast_in_dim %b3dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v276 = stablehlo.broadcast_in_dim %b3dbt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v277 = stablehlo.multiply %v274, %v275 : tensor<32x144x56x56xf32>
    %v278 = stablehlo.add %v277, %v276 : tensor<32x144x56x56xf32>
    %v279 = stablehlo.reshape %v278 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v280 = stablehlo.reshape %v279 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v281 = stablehlo.logistic %v280 : tensor<32x144x56x56xf32>
    %v282 = stablehlo.multiply %v280, %v281 : tensor<32x144x56x56xf32>
    %v283 = stablehlo.reshape %v282 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v284 = stablehlo.reshape %v283 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v285 = stablehlo.constant dense<0.0> : tensor<f32>
    %v286 = stablehlo.reduce(%v284 init: %v285) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v287 = stablehlo.constant dense<3136.0> : tensor<32x144xf32>
    %v288 = stablehlo.divide %v286, %v287 : tensor<32x144xf32>
    %v289 = stablehlo.dot_general %v288, %b3zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v290 = stablehlo.broadcast_in_dim %b3zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v291 = stablehlo.add %v289, %v290 : tensor<32x6xf32>
    %v292 = stablehlo.logistic %v291 : tensor<32x6xf32>
    %v293 = stablehlo.multiply %v291, %v292 : tensor<32x6xf32>
    %v294 = stablehlo.dot_general %v293, %b3zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v295 = stablehlo.broadcast_in_dim %b3zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v296 = stablehlo.add %v294, %v295 : tensor<32x144xf32>
    %v297 = stablehlo.reshape %v283 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v298 = stablehlo.constant dense<0.0> : tensor<f32>
    %v299 = stablehlo.reduce(%v297 init: %v298) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v300 = stablehlo.constant dense<3136.0> : tensor<32x144xf32>
    %v301 = stablehlo.divide %v299, %v300 : tensor<32x144xf32>
    %v302 = stablehlo.dot_general %v301, %b3zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v303 = stablehlo.broadcast_in_dim %b3zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v304 = stablehlo.add %v302, %v303 : tensor<32x6xf32>
    %v305 = stablehlo.logistic %v304 : tensor<32x6xf32>
    %v306 = stablehlo.multiply %v304, %v305 : tensor<32x6xf32>
    %v307 = stablehlo.dot_general %v306, %b3zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v308 = stablehlo.broadcast_in_dim %b3zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v309 = stablehlo.add %v307, %v308 : tensor<32x144xf32>
    %v310 = stablehlo.logistic %v309 : tensor<32x144xf32>
    %v311 = stablehlo.broadcast_in_dim %v310, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v312 = stablehlo.multiply %v297, %v311 : tensor<32x144x56x56xf32>
    %v313 = stablehlo.reshape %v312 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v314 = stablehlo.reshape %v313 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v315 = stablehlo.convolution(%v314, %b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v316 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v317 = stablehlo.add %v315, %v316 : tensor<32x24x56x56xf32>
    %v318 = stablehlo.reshape %v317 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v319 = stablehlo.reshape %v318 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v320 = stablehlo.constant dense<0.0> : tensor<f32>
    %v321 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v322 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v323 = stablehlo.reduce(%v319 init: %v320) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v324 = stablehlo.broadcast_in_dim %v323, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v325 = stablehlo.divide %v324, %v321 : tensor<32x24x56x56xf32>
    %v326 = stablehlo.subtract %v319, %v325 : tensor<32x24x56x56xf32>
    %v327 = stablehlo.multiply %v326, %v326 : tensor<32x24x56x56xf32>
    %v328 = stablehlo.reduce(%v327 init: %v320) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v329 = stablehlo.broadcast_in_dim %v328, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v330 = stablehlo.divide %v329, %v321 : tensor<32x24x56x56xf32>
    %v331 = stablehlo.add %v330, %v322 : tensor<32x24x56x56xf32>
    %v332 = stablehlo.rsqrt %v331 : tensor<32x24x56x56xf32>
    %v333 = stablehlo.multiply %v326, %v332 : tensor<32x24x56x56xf32>
    %v334 = stablehlo.broadcast_in_dim %b3pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v335 = stablehlo.broadcast_in_dim %b3pbt, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v336 = stablehlo.multiply %v333, %v334 : tensor<32x24x56x56xf32>
    %v337 = stablehlo.add %v336, %v335 : tensor<32x24x56x56xf32>
    %v338 = stablehlo.reshape %v337 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v339 = stablehlo.reshape %v338 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v340 = stablehlo.reshape %v225 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v341 = stablehlo.add %v339, %v340 : tensor<32x24x56x56xf32>
    %v342 = stablehlo.reshape %v341 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v343 = stablehlo.reshape %v342 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v344 = stablehlo.convolution(%v343, %b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v345 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v346 = stablehlo.add %v344, %v345 : tensor<32x144x56x56xf32>
    %v347 = stablehlo.reshape %v346 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v348 = stablehlo.reshape %v347 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v349 = stablehlo.constant dense<0.0> : tensor<f32>
    %v350 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v351 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v352 = stablehlo.reduce(%v348 init: %v349) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v353 = stablehlo.broadcast_in_dim %v352, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v354 = stablehlo.divide %v353, %v350 : tensor<32x144x56x56xf32>
    %v355 = stablehlo.subtract %v348, %v354 : tensor<32x144x56x56xf32>
    %v356 = stablehlo.multiply %v355, %v355 : tensor<32x144x56x56xf32>
    %v357 = stablehlo.reduce(%v356 init: %v349) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v358 = stablehlo.broadcast_in_dim %v357, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v359 = stablehlo.divide %v358, %v350 : tensor<32x144x56x56xf32>
    %v360 = stablehlo.add %v359, %v351 : tensor<32x144x56x56xf32>
    %v361 = stablehlo.rsqrt %v360 : tensor<32x144x56x56xf32>
    %v362 = stablehlo.multiply %v355, %v361 : tensor<32x144x56x56xf32>
    %v363 = stablehlo.broadcast_in_dim %b4eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v364 = stablehlo.broadcast_in_dim %b4ebt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v365 = stablehlo.multiply %v362, %v363 : tensor<32x144x56x56xf32>
    %v366 = stablehlo.add %v365, %v364 : tensor<32x144x56x56xf32>
    %v367 = stablehlo.reshape %v366 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v368 = stablehlo.reshape %v367 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v369 = stablehlo.logistic %v368 : tensor<32x144x56x56xf32>
    %v370 = stablehlo.multiply %v368, %v369 : tensor<32x144x56x56xf32>
    %v371 = stablehlo.reshape %v370 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v372 = stablehlo.reshape %v371 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v373 = stablehlo.convolution(%v372, %b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x5x5xf32>) -> tensor<32x144x28x28xf32>
    %v374 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v375 = stablehlo.add %v373, %v374 : tensor<32x144x28x28xf32>
    %v376 = stablehlo.reshape %v375 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v377 = stablehlo.reshape %v376 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v378 = stablehlo.constant dense<0.0> : tensor<f32>
    %v379 = stablehlo.constant dense<25088.0> : tensor<32x144x28x28xf32>
    %v380 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v381 = stablehlo.reduce(%v377 init: %v378) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v382 = stablehlo.broadcast_in_dim %v381, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v383 = stablehlo.divide %v382, %v379 : tensor<32x144x28x28xf32>
    %v384 = stablehlo.subtract %v377, %v383 : tensor<32x144x28x28xf32>
    %v385 = stablehlo.multiply %v384, %v384 : tensor<32x144x28x28xf32>
    %v386 = stablehlo.reduce(%v385 init: %v378) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v387 = stablehlo.broadcast_in_dim %v386, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v388 = stablehlo.divide %v387, %v379 : tensor<32x144x28x28xf32>
    %v389 = stablehlo.add %v388, %v380 : tensor<32x144x28x28xf32>
    %v390 = stablehlo.rsqrt %v389 : tensor<32x144x28x28xf32>
    %v391 = stablehlo.multiply %v384, %v390 : tensor<32x144x28x28xf32>
    %v392 = stablehlo.broadcast_in_dim %b4dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v393 = stablehlo.broadcast_in_dim %b4dbt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v394 = stablehlo.multiply %v391, %v392 : tensor<32x144x28x28xf32>
    %v395 = stablehlo.add %v394, %v393 : tensor<32x144x28x28xf32>
    %v396 = stablehlo.reshape %v395 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v397 = stablehlo.reshape %v396 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v398 = stablehlo.logistic %v397 : tensor<32x144x28x28xf32>
    %v399 = stablehlo.multiply %v397, %v398 : tensor<32x144x28x28xf32>
    %v400 = stablehlo.reshape %v399 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v401 = stablehlo.reshape %v400 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v402 = stablehlo.constant dense<0.0> : tensor<f32>
    %v403 = stablehlo.reduce(%v401 init: %v402) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v404 = stablehlo.constant dense<784.0> : tensor<32x144xf32>
    %v405 = stablehlo.divide %v403, %v404 : tensor<32x144xf32>
    %v406 = stablehlo.dot_general %v405, %b4zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v407 = stablehlo.broadcast_in_dim %b4zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v408 = stablehlo.add %v406, %v407 : tensor<32x6xf32>
    %v409 = stablehlo.logistic %v408 : tensor<32x6xf32>
    %v410 = stablehlo.multiply %v408, %v409 : tensor<32x6xf32>
    %v411 = stablehlo.dot_general %v410, %b4zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v412 = stablehlo.broadcast_in_dim %b4zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v413 = stablehlo.add %v411, %v412 : tensor<32x144xf32>
    %v414 = stablehlo.reshape %v400 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v415 = stablehlo.constant dense<0.0> : tensor<f32>
    %v416 = stablehlo.reduce(%v414 init: %v415) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v417 = stablehlo.constant dense<784.0> : tensor<32x144xf32>
    %v418 = stablehlo.divide %v416, %v417 : tensor<32x144xf32>
    %v419 = stablehlo.dot_general %v418, %b4zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v420 = stablehlo.broadcast_in_dim %b4zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v421 = stablehlo.add %v419, %v420 : tensor<32x6xf32>
    %v422 = stablehlo.logistic %v421 : tensor<32x6xf32>
    %v423 = stablehlo.multiply %v421, %v422 : tensor<32x6xf32>
    %v424 = stablehlo.dot_general %v423, %b4zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v425 = stablehlo.broadcast_in_dim %b4zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v426 = stablehlo.add %v424, %v425 : tensor<32x144xf32>
    %v427 = stablehlo.logistic %v426 : tensor<32x144xf32>
    %v428 = stablehlo.broadcast_in_dim %v427, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v429 = stablehlo.multiply %v414, %v428 : tensor<32x144x28x28xf32>
    %v430 = stablehlo.reshape %v429 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v431 = stablehlo.reshape %v430 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v432 = stablehlo.convolution(%v431, %b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x28x28xf32>, tensor<40x144x1x1xf32>) -> tensor<32x40x28x28xf32>
    %v433 = stablehlo.broadcast_in_dim %zb40, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v434 = stablehlo.add %v432, %v433 : tensor<32x40x28x28xf32>
    %v435 = stablehlo.reshape %v434 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v436 = stablehlo.reshape %v435 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v437 = stablehlo.constant dense<0.0> : tensor<f32>
    %v438 = stablehlo.constant dense<25088.0> : tensor<32x40x28x28xf32>
    %v439 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v440 = stablehlo.reduce(%v436 init: %v437) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v441 = stablehlo.broadcast_in_dim %v440, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v442 = stablehlo.divide %v441, %v438 : tensor<32x40x28x28xf32>
    %v443 = stablehlo.subtract %v436, %v442 : tensor<32x40x28x28xf32>
    %v444 = stablehlo.multiply %v443, %v443 : tensor<32x40x28x28xf32>
    %v445 = stablehlo.reduce(%v444 init: %v437) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v446 = stablehlo.broadcast_in_dim %v445, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v447 = stablehlo.divide %v446, %v438 : tensor<32x40x28x28xf32>
    %v448 = stablehlo.add %v447, %v439 : tensor<32x40x28x28xf32>
    %v449 = stablehlo.rsqrt %v448 : tensor<32x40x28x28xf32>
    %v450 = stablehlo.multiply %v443, %v449 : tensor<32x40x28x28xf32>
    %v451 = stablehlo.broadcast_in_dim %b4pg, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v452 = stablehlo.broadcast_in_dim %b4pbt, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v453 = stablehlo.multiply %v450, %v451 : tensor<32x40x28x28xf32>
    %v454 = stablehlo.add %v453, %v452 : tensor<32x40x28x28xf32>
    %v455 = stablehlo.reshape %v454 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v456 = stablehlo.reshape %v455 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v457 = stablehlo.convolution(%v456, %b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<32x240x28x28xf32>
    %v458 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v459 = stablehlo.add %v457, %v458 : tensor<32x240x28x28xf32>
    %v460 = stablehlo.reshape %v459 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v461 = stablehlo.reshape %v460 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v462 = stablehlo.constant dense<0.0> : tensor<f32>
    %v463 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v464 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v465 = stablehlo.reduce(%v461 init: %v462) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v466 = stablehlo.broadcast_in_dim %v465, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v467 = stablehlo.divide %v466, %v463 : tensor<32x240x28x28xf32>
    %v468 = stablehlo.subtract %v461, %v467 : tensor<32x240x28x28xf32>
    %v469 = stablehlo.multiply %v468, %v468 : tensor<32x240x28x28xf32>
    %v470 = stablehlo.reduce(%v469 init: %v462) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v471 = stablehlo.broadcast_in_dim %v470, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v472 = stablehlo.divide %v471, %v463 : tensor<32x240x28x28xf32>
    %v473 = stablehlo.add %v472, %v464 : tensor<32x240x28x28xf32>
    %v474 = stablehlo.rsqrt %v473 : tensor<32x240x28x28xf32>
    %v475 = stablehlo.multiply %v468, %v474 : tensor<32x240x28x28xf32>
    %v476 = stablehlo.broadcast_in_dim %b5eg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v477 = stablehlo.broadcast_in_dim %b5ebt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v478 = stablehlo.multiply %v475, %v476 : tensor<32x240x28x28xf32>
    %v479 = stablehlo.add %v478, %v477 : tensor<32x240x28x28xf32>
    %v480 = stablehlo.reshape %v479 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v481 = stablehlo.reshape %v480 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v482 = stablehlo.logistic %v481 : tensor<32x240x28x28xf32>
    %v483 = stablehlo.multiply %v481, %v482 : tensor<32x240x28x28xf32>
    %v484 = stablehlo.reshape %v483 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v485 = stablehlo.reshape %v484 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v486 = stablehlo.convolution(%v485, %b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x5x5xf32>) -> tensor<32x240x28x28xf32>
    %v487 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v488 = stablehlo.add %v486, %v487 : tensor<32x240x28x28xf32>
    %v489 = stablehlo.reshape %v488 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v490 = stablehlo.reshape %v489 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v491 = stablehlo.constant dense<0.0> : tensor<f32>
    %v492 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v493 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v494 = stablehlo.reduce(%v490 init: %v491) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v495 = stablehlo.broadcast_in_dim %v494, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v496 = stablehlo.divide %v495, %v492 : tensor<32x240x28x28xf32>
    %v497 = stablehlo.subtract %v490, %v496 : tensor<32x240x28x28xf32>
    %v498 = stablehlo.multiply %v497, %v497 : tensor<32x240x28x28xf32>
    %v499 = stablehlo.reduce(%v498 init: %v491) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v500 = stablehlo.broadcast_in_dim %v499, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v501 = stablehlo.divide %v500, %v492 : tensor<32x240x28x28xf32>
    %v502 = stablehlo.add %v501, %v493 : tensor<32x240x28x28xf32>
    %v503 = stablehlo.rsqrt %v502 : tensor<32x240x28x28xf32>
    %v504 = stablehlo.multiply %v497, %v503 : tensor<32x240x28x28xf32>
    %v505 = stablehlo.broadcast_in_dim %b5dg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v506 = stablehlo.broadcast_in_dim %b5dbt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v507 = stablehlo.multiply %v504, %v505 : tensor<32x240x28x28xf32>
    %v508 = stablehlo.add %v507, %v506 : tensor<32x240x28x28xf32>
    %v509 = stablehlo.reshape %v508 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v510 = stablehlo.reshape %v509 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v511 = stablehlo.logistic %v510 : tensor<32x240x28x28xf32>
    %v512 = stablehlo.multiply %v510, %v511 : tensor<32x240x28x28xf32>
    %v513 = stablehlo.reshape %v512 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v514 = stablehlo.reshape %v513 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v515 = stablehlo.constant dense<0.0> : tensor<f32>
    %v516 = stablehlo.reduce(%v514 init: %v515) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v517 = stablehlo.constant dense<784.0> : tensor<32x240xf32>
    %v518 = stablehlo.divide %v516, %v517 : tensor<32x240xf32>
    %v519 = stablehlo.dot_general %v518, %b5zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v520 = stablehlo.broadcast_in_dim %b5zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v521 = stablehlo.add %v519, %v520 : tensor<32x10xf32>
    %v522 = stablehlo.logistic %v521 : tensor<32x10xf32>
    %v523 = stablehlo.multiply %v521, %v522 : tensor<32x10xf32>
    %v524 = stablehlo.dot_general %v523, %b5zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v525 = stablehlo.broadcast_in_dim %b5zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v526 = stablehlo.add %v524, %v525 : tensor<32x240xf32>
    %v527 = stablehlo.reshape %v513 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v528 = stablehlo.constant dense<0.0> : tensor<f32>
    %v529 = stablehlo.reduce(%v527 init: %v528) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v530 = stablehlo.constant dense<784.0> : tensor<32x240xf32>
    %v531 = stablehlo.divide %v529, %v530 : tensor<32x240xf32>
    %v532 = stablehlo.dot_general %v531, %b5zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v533 = stablehlo.broadcast_in_dim %b5zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v534 = stablehlo.add %v532, %v533 : tensor<32x10xf32>
    %v535 = stablehlo.logistic %v534 : tensor<32x10xf32>
    %v536 = stablehlo.multiply %v534, %v535 : tensor<32x10xf32>
    %v537 = stablehlo.dot_general %v536, %b5zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v538 = stablehlo.broadcast_in_dim %b5zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v539 = stablehlo.add %v537, %v538 : tensor<32x240xf32>
    %v540 = stablehlo.logistic %v539 : tensor<32x240xf32>
    %v541 = stablehlo.broadcast_in_dim %v540, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x28x28xf32>
    %v542 = stablehlo.multiply %v527, %v541 : tensor<32x240x28x28xf32>
    %v543 = stablehlo.reshape %v542 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v544 = stablehlo.reshape %v543 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v545 = stablehlo.convolution(%v544, %b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x28x28xf32>, tensor<40x240x1x1xf32>) -> tensor<32x40x28x28xf32>
    %v546 = stablehlo.broadcast_in_dim %zb40, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v547 = stablehlo.add %v545, %v546 : tensor<32x40x28x28xf32>
    %v548 = stablehlo.reshape %v547 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v549 = stablehlo.reshape %v548 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v550 = stablehlo.constant dense<0.0> : tensor<f32>
    %v551 = stablehlo.constant dense<25088.0> : tensor<32x40x28x28xf32>
    %v552 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v553 = stablehlo.reduce(%v549 init: %v550) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v554 = stablehlo.broadcast_in_dim %v553, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v555 = stablehlo.divide %v554, %v551 : tensor<32x40x28x28xf32>
    %v556 = stablehlo.subtract %v549, %v555 : tensor<32x40x28x28xf32>
    %v557 = stablehlo.multiply %v556, %v556 : tensor<32x40x28x28xf32>
    %v558 = stablehlo.reduce(%v557 init: %v550) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v559 = stablehlo.broadcast_in_dim %v558, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v560 = stablehlo.divide %v559, %v551 : tensor<32x40x28x28xf32>
    %v561 = stablehlo.add %v560, %v552 : tensor<32x40x28x28xf32>
    %v562 = stablehlo.rsqrt %v561 : tensor<32x40x28x28xf32>
    %v563 = stablehlo.multiply %v556, %v562 : tensor<32x40x28x28xf32>
    %v564 = stablehlo.broadcast_in_dim %b5pg, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v565 = stablehlo.broadcast_in_dim %b5pbt, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v566 = stablehlo.multiply %v563, %v564 : tensor<32x40x28x28xf32>
    %v567 = stablehlo.add %v566, %v565 : tensor<32x40x28x28xf32>
    %v568 = stablehlo.reshape %v567 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v569 = stablehlo.reshape %v568 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v570 = stablehlo.reshape %v455 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v571 = stablehlo.add %v569, %v570 : tensor<32x40x28x28xf32>
    %v572 = stablehlo.reshape %v571 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v573 = stablehlo.reshape %v572 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v574 = stablehlo.convolution(%v573, %b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<32x240x28x28xf32>
    %v575 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v576 = stablehlo.add %v574, %v575 : tensor<32x240x28x28xf32>
    %v577 = stablehlo.reshape %v576 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v578 = stablehlo.reshape %v577 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v579 = stablehlo.constant dense<0.0> : tensor<f32>
    %v580 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v581 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v582 = stablehlo.reduce(%v578 init: %v579) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v583 = stablehlo.broadcast_in_dim %v582, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v584 = stablehlo.divide %v583, %v580 : tensor<32x240x28x28xf32>
    %v585 = stablehlo.subtract %v578, %v584 : tensor<32x240x28x28xf32>
    %v586 = stablehlo.multiply %v585, %v585 : tensor<32x240x28x28xf32>
    %v587 = stablehlo.reduce(%v586 init: %v579) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v588 = stablehlo.broadcast_in_dim %v587, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v589 = stablehlo.divide %v588, %v580 : tensor<32x240x28x28xf32>
    %v590 = stablehlo.add %v589, %v581 : tensor<32x240x28x28xf32>
    %v591 = stablehlo.rsqrt %v590 : tensor<32x240x28x28xf32>
    %v592 = stablehlo.multiply %v585, %v591 : tensor<32x240x28x28xf32>
    %v593 = stablehlo.broadcast_in_dim %b6eg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v594 = stablehlo.broadcast_in_dim %b6ebt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v595 = stablehlo.multiply %v592, %v593 : tensor<32x240x28x28xf32>
    %v596 = stablehlo.add %v595, %v594 : tensor<32x240x28x28xf32>
    %v597 = stablehlo.reshape %v596 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v598 = stablehlo.reshape %v597 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v599 = stablehlo.logistic %v598 : tensor<32x240x28x28xf32>
    %v600 = stablehlo.multiply %v598, %v599 : tensor<32x240x28x28xf32>
    %v601 = stablehlo.reshape %v600 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v602 = stablehlo.reshape %v601 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v603 = stablehlo.convolution(%v602, %b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x3x3xf32>) -> tensor<32x240x14x14xf32>
    %v604 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v605 = stablehlo.add %v603, %v604 : tensor<32x240x14x14xf32>
    %v606 = stablehlo.reshape %v605 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v608 = stablehlo.constant dense<0.0> : tensor<f32>
    %v609 = stablehlo.constant dense<6272.0> : tensor<32x240x14x14xf32>
    %v610 = stablehlo.constant dense<1.0e-5> : tensor<32x240x14x14xf32>
    %v611 = stablehlo.reduce(%v607 init: %v608) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v612 = stablehlo.broadcast_in_dim %v611, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v613 = stablehlo.divide %v612, %v609 : tensor<32x240x14x14xf32>
    %v614 = stablehlo.subtract %v607, %v613 : tensor<32x240x14x14xf32>
    %v615 = stablehlo.multiply %v614, %v614 : tensor<32x240x14x14xf32>
    %v616 = stablehlo.reduce(%v615 init: %v608) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v617 = stablehlo.broadcast_in_dim %v616, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v618 = stablehlo.divide %v617, %v609 : tensor<32x240x14x14xf32>
    %v619 = stablehlo.add %v618, %v610 : tensor<32x240x14x14xf32>
    %v620 = stablehlo.rsqrt %v619 : tensor<32x240x14x14xf32>
    %v621 = stablehlo.multiply %v614, %v620 : tensor<32x240x14x14xf32>
    %v622 = stablehlo.broadcast_in_dim %b6dg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v623 = stablehlo.broadcast_in_dim %b6dbt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v624 = stablehlo.multiply %v621, %v622 : tensor<32x240x14x14xf32>
    %v625 = stablehlo.add %v624, %v623 : tensor<32x240x14x14xf32>
    %v626 = stablehlo.reshape %v625 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v627 = stablehlo.reshape %v626 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v628 = stablehlo.logistic %v627 : tensor<32x240x14x14xf32>
    %v629 = stablehlo.multiply %v627, %v628 : tensor<32x240x14x14xf32>
    %v630 = stablehlo.reshape %v629 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v631 = stablehlo.reshape %v630 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v632 = stablehlo.constant dense<0.0> : tensor<f32>
    %v633 = stablehlo.reduce(%v631 init: %v632) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v634 = stablehlo.constant dense<196.0> : tensor<32x240xf32>
    %v635 = stablehlo.divide %v633, %v634 : tensor<32x240xf32>
    %v636 = stablehlo.dot_general %v635, %b6zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v637 = stablehlo.broadcast_in_dim %b6zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v638 = stablehlo.add %v636, %v637 : tensor<32x10xf32>
    %v639 = stablehlo.logistic %v638 : tensor<32x10xf32>
    %v640 = stablehlo.multiply %v638, %v639 : tensor<32x10xf32>
    %v641 = stablehlo.dot_general %v640, %b6zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v642 = stablehlo.broadcast_in_dim %b6zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v643 = stablehlo.add %v641, %v642 : tensor<32x240xf32>
    %v644 = stablehlo.reshape %v630 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v645 = stablehlo.constant dense<0.0> : tensor<f32>
    %v646 = stablehlo.reduce(%v644 init: %v645) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v647 = stablehlo.constant dense<196.0> : tensor<32x240xf32>
    %v648 = stablehlo.divide %v646, %v647 : tensor<32x240xf32>
    %v649 = stablehlo.dot_general %v648, %b6zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v650 = stablehlo.broadcast_in_dim %b6zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v651 = stablehlo.add %v649, %v650 : tensor<32x10xf32>
    %v652 = stablehlo.logistic %v651 : tensor<32x10xf32>
    %v653 = stablehlo.multiply %v651, %v652 : tensor<32x10xf32>
    %v654 = stablehlo.dot_general %v653, %b6zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v655 = stablehlo.broadcast_in_dim %b6zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v656 = stablehlo.add %v654, %v655 : tensor<32x240xf32>
    %v657 = stablehlo.logistic %v656 : tensor<32x240xf32>
    %v658 = stablehlo.broadcast_in_dim %v657, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x14x14xf32>
    %v659 = stablehlo.multiply %v644, %v658 : tensor<32x240x14x14xf32>
    %v660 = stablehlo.reshape %v659 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v661 = stablehlo.reshape %v660 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v662 = stablehlo.convolution(%v661, %b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x14x14xf32>, tensor<80x240x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v663 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v664 = stablehlo.add %v662, %v663 : tensor<32x80x14x14xf32>
    %v665 = stablehlo.reshape %v664 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v666 = stablehlo.reshape %v665 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v667 = stablehlo.constant dense<0.0> : tensor<f32>
    %v668 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v669 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v670 = stablehlo.reduce(%v666 init: %v667) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v671 = stablehlo.broadcast_in_dim %v670, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v672 = stablehlo.divide %v671, %v668 : tensor<32x80x14x14xf32>
    %v673 = stablehlo.subtract %v666, %v672 : tensor<32x80x14x14xf32>
    %v674 = stablehlo.multiply %v673, %v673 : tensor<32x80x14x14xf32>
    %v675 = stablehlo.reduce(%v674 init: %v667) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v676 = stablehlo.broadcast_in_dim %v675, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v677 = stablehlo.divide %v676, %v668 : tensor<32x80x14x14xf32>
    %v678 = stablehlo.add %v677, %v669 : tensor<32x80x14x14xf32>
    %v679 = stablehlo.rsqrt %v678 : tensor<32x80x14x14xf32>
    %v680 = stablehlo.multiply %v673, %v679 : tensor<32x80x14x14xf32>
    %v681 = stablehlo.broadcast_in_dim %b6pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v682 = stablehlo.broadcast_in_dim %b6pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v683 = stablehlo.multiply %v680, %v681 : tensor<32x80x14x14xf32>
    %v684 = stablehlo.add %v683, %v682 : tensor<32x80x14x14xf32>
    %v685 = stablehlo.reshape %v684 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v686 = stablehlo.reshape %v685 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v687 = stablehlo.convolution(%v686, %b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v688 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v689 = stablehlo.add %v687, %v688 : tensor<32x480x14x14xf32>
    %v690 = stablehlo.reshape %v689 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v691 = stablehlo.reshape %v690 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v692 = stablehlo.constant dense<0.0> : tensor<f32>
    %v693 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v694 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v695 = stablehlo.reduce(%v691 init: %v692) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v696 = stablehlo.broadcast_in_dim %v695, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v697 = stablehlo.divide %v696, %v693 : tensor<32x480x14x14xf32>
    %v698 = stablehlo.subtract %v691, %v697 : tensor<32x480x14x14xf32>
    %v699 = stablehlo.multiply %v698, %v698 : tensor<32x480x14x14xf32>
    %v700 = stablehlo.reduce(%v699 init: %v692) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v701 = stablehlo.broadcast_in_dim %v700, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v702 = stablehlo.divide %v701, %v693 : tensor<32x480x14x14xf32>
    %v703 = stablehlo.add %v702, %v694 : tensor<32x480x14x14xf32>
    %v704 = stablehlo.rsqrt %v703 : tensor<32x480x14x14xf32>
    %v705 = stablehlo.multiply %v698, %v704 : tensor<32x480x14x14xf32>
    %v706 = stablehlo.broadcast_in_dim %b7eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v707 = stablehlo.broadcast_in_dim %b7ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v708 = stablehlo.multiply %v705, %v706 : tensor<32x480x14x14xf32>
    %v709 = stablehlo.add %v708, %v707 : tensor<32x480x14x14xf32>
    %v710 = stablehlo.reshape %v709 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v711 = stablehlo.reshape %v710 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v712 = stablehlo.logistic %v711 : tensor<32x480x14x14xf32>
    %v713 = stablehlo.multiply %v711, %v712 : tensor<32x480x14x14xf32>
    %v714 = stablehlo.reshape %v713 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v715 = stablehlo.reshape %v714 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v716 = stablehlo.convolution(%v715, %b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %v717 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v718 = stablehlo.add %v716, %v717 : tensor<32x480x14x14xf32>
    %v719 = stablehlo.reshape %v718 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v720 = stablehlo.reshape %v719 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v721 = stablehlo.constant dense<0.0> : tensor<f32>
    %v722 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v723 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v724 = stablehlo.reduce(%v720 init: %v721) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v725 = stablehlo.broadcast_in_dim %v724, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v726 = stablehlo.divide %v725, %v722 : tensor<32x480x14x14xf32>
    %v727 = stablehlo.subtract %v720, %v726 : tensor<32x480x14x14xf32>
    %v728 = stablehlo.multiply %v727, %v727 : tensor<32x480x14x14xf32>
    %v729 = stablehlo.reduce(%v728 init: %v721) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v730 = stablehlo.broadcast_in_dim %v729, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v731 = stablehlo.divide %v730, %v722 : tensor<32x480x14x14xf32>
    %v732 = stablehlo.add %v731, %v723 : tensor<32x480x14x14xf32>
    %v733 = stablehlo.rsqrt %v732 : tensor<32x480x14x14xf32>
    %v734 = stablehlo.multiply %v727, %v733 : tensor<32x480x14x14xf32>
    %v735 = stablehlo.broadcast_in_dim %b7dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v736 = stablehlo.broadcast_in_dim %b7dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v737 = stablehlo.multiply %v734, %v735 : tensor<32x480x14x14xf32>
    %v738 = stablehlo.add %v737, %v736 : tensor<32x480x14x14xf32>
    %v739 = stablehlo.reshape %v738 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v740 = stablehlo.reshape %v739 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v741 = stablehlo.logistic %v740 : tensor<32x480x14x14xf32>
    %v742 = stablehlo.multiply %v740, %v741 : tensor<32x480x14x14xf32>
    %v743 = stablehlo.reshape %v742 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v744 = stablehlo.reshape %v743 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v745 = stablehlo.constant dense<0.0> : tensor<f32>
    %v746 = stablehlo.reduce(%v744 init: %v745) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v747 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v748 = stablehlo.divide %v746, %v747 : tensor<32x480xf32>
    %v749 = stablehlo.dot_general %v748, %b7zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v750 = stablehlo.broadcast_in_dim %b7zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v751 = stablehlo.add %v749, %v750 : tensor<32x20xf32>
    %v752 = stablehlo.logistic %v751 : tensor<32x20xf32>
    %v753 = stablehlo.multiply %v751, %v752 : tensor<32x20xf32>
    %v754 = stablehlo.dot_general %v753, %b7zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v755 = stablehlo.broadcast_in_dim %b7zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v756 = stablehlo.add %v754, %v755 : tensor<32x480xf32>
    %v757 = stablehlo.reshape %v743 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v758 = stablehlo.constant dense<0.0> : tensor<f32>
    %v759 = stablehlo.reduce(%v757 init: %v758) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v760 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v761 = stablehlo.divide %v759, %v760 : tensor<32x480xf32>
    %v762 = stablehlo.dot_general %v761, %b7zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v763 = stablehlo.broadcast_in_dim %b7zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v764 = stablehlo.add %v762, %v763 : tensor<32x20xf32>
    %v765 = stablehlo.logistic %v764 : tensor<32x20xf32>
    %v766 = stablehlo.multiply %v764, %v765 : tensor<32x20xf32>
    %v767 = stablehlo.dot_general %v766, %b7zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v768 = stablehlo.broadcast_in_dim %b7zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v769 = stablehlo.add %v767, %v768 : tensor<32x480xf32>
    %v770 = stablehlo.logistic %v769 : tensor<32x480xf32>
    %v771 = stablehlo.broadcast_in_dim %v770, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v772 = stablehlo.multiply %v757, %v771 : tensor<32x480x14x14xf32>
    %v773 = stablehlo.reshape %v772 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v774 = stablehlo.reshape %v773 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v775 = stablehlo.convolution(%v774, %b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v776 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v777 = stablehlo.add %v775, %v776 : tensor<32x80x14x14xf32>
    %v778 = stablehlo.reshape %v777 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v779 = stablehlo.reshape %v778 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v780 = stablehlo.constant dense<0.0> : tensor<f32>
    %v781 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v782 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v783 = stablehlo.reduce(%v779 init: %v780) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v784 = stablehlo.broadcast_in_dim %v783, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v785 = stablehlo.divide %v784, %v781 : tensor<32x80x14x14xf32>
    %v786 = stablehlo.subtract %v779, %v785 : tensor<32x80x14x14xf32>
    %v787 = stablehlo.multiply %v786, %v786 : tensor<32x80x14x14xf32>
    %v788 = stablehlo.reduce(%v787 init: %v780) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v789 = stablehlo.broadcast_in_dim %v788, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v790 = stablehlo.divide %v789, %v781 : tensor<32x80x14x14xf32>
    %v791 = stablehlo.add %v790, %v782 : tensor<32x80x14x14xf32>
    %v792 = stablehlo.rsqrt %v791 : tensor<32x80x14x14xf32>
    %v793 = stablehlo.multiply %v786, %v792 : tensor<32x80x14x14xf32>
    %v794 = stablehlo.broadcast_in_dim %b7pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v795 = stablehlo.broadcast_in_dim %b7pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v796 = stablehlo.multiply %v793, %v794 : tensor<32x80x14x14xf32>
    %v797 = stablehlo.add %v796, %v795 : tensor<32x80x14x14xf32>
    %v798 = stablehlo.reshape %v797 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v799 = stablehlo.reshape %v798 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v800 = stablehlo.reshape %v685 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v801 = stablehlo.add %v799, %v800 : tensor<32x80x14x14xf32>
    %v802 = stablehlo.reshape %v801 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v803 = stablehlo.reshape %v802 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v804 = stablehlo.convolution(%v803, %b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v805 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v806 = stablehlo.add %v804, %v805 : tensor<32x480x14x14xf32>
    %v807 = stablehlo.reshape %v806 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v808 = stablehlo.reshape %v807 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v809 = stablehlo.constant dense<0.0> : tensor<f32>
    %v810 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v811 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v812 = stablehlo.reduce(%v808 init: %v809) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v813 = stablehlo.broadcast_in_dim %v812, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v814 = stablehlo.divide %v813, %v810 : tensor<32x480x14x14xf32>
    %v815 = stablehlo.subtract %v808, %v814 : tensor<32x480x14x14xf32>
    %v816 = stablehlo.multiply %v815, %v815 : tensor<32x480x14x14xf32>
    %v817 = stablehlo.reduce(%v816 init: %v809) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v818 = stablehlo.broadcast_in_dim %v817, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v819 = stablehlo.divide %v818, %v810 : tensor<32x480x14x14xf32>
    %v820 = stablehlo.add %v819, %v811 : tensor<32x480x14x14xf32>
    %v821 = stablehlo.rsqrt %v820 : tensor<32x480x14x14xf32>
    %v822 = stablehlo.multiply %v815, %v821 : tensor<32x480x14x14xf32>
    %v823 = stablehlo.broadcast_in_dim %b8eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v824 = stablehlo.broadcast_in_dim %b8ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v825 = stablehlo.multiply %v822, %v823 : tensor<32x480x14x14xf32>
    %v826 = stablehlo.add %v825, %v824 : tensor<32x480x14x14xf32>
    %v827 = stablehlo.reshape %v826 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v828 = stablehlo.reshape %v827 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v829 = stablehlo.logistic %v828 : tensor<32x480x14x14xf32>
    %v830 = stablehlo.multiply %v828, %v829 : tensor<32x480x14x14xf32>
    %v831 = stablehlo.reshape %v830 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v832 = stablehlo.reshape %v831 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v833 = stablehlo.convolution(%v832, %b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %v834 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v835 = stablehlo.add %v833, %v834 : tensor<32x480x14x14xf32>
    %v836 = stablehlo.reshape %v835 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v837 = stablehlo.reshape %v836 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v838 = stablehlo.constant dense<0.0> : tensor<f32>
    %v839 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v840 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v841 = stablehlo.reduce(%v837 init: %v838) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v842 = stablehlo.broadcast_in_dim %v841, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v843 = stablehlo.divide %v842, %v839 : tensor<32x480x14x14xf32>
    %v844 = stablehlo.subtract %v837, %v843 : tensor<32x480x14x14xf32>
    %v845 = stablehlo.multiply %v844, %v844 : tensor<32x480x14x14xf32>
    %v846 = stablehlo.reduce(%v845 init: %v838) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v847 = stablehlo.broadcast_in_dim %v846, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v848 = stablehlo.divide %v847, %v839 : tensor<32x480x14x14xf32>
    %v849 = stablehlo.add %v848, %v840 : tensor<32x480x14x14xf32>
    %v850 = stablehlo.rsqrt %v849 : tensor<32x480x14x14xf32>
    %v851 = stablehlo.multiply %v844, %v850 : tensor<32x480x14x14xf32>
    %v852 = stablehlo.broadcast_in_dim %b8dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v853 = stablehlo.broadcast_in_dim %b8dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v854 = stablehlo.multiply %v851, %v852 : tensor<32x480x14x14xf32>
    %v855 = stablehlo.add %v854, %v853 : tensor<32x480x14x14xf32>
    %v856 = stablehlo.reshape %v855 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v857 = stablehlo.reshape %v856 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v858 = stablehlo.logistic %v857 : tensor<32x480x14x14xf32>
    %v859 = stablehlo.multiply %v857, %v858 : tensor<32x480x14x14xf32>
    %v860 = stablehlo.reshape %v859 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v861 = stablehlo.reshape %v860 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v862 = stablehlo.constant dense<0.0> : tensor<f32>
    %v863 = stablehlo.reduce(%v861 init: %v862) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v864 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v865 = stablehlo.divide %v863, %v864 : tensor<32x480xf32>
    %v866 = stablehlo.dot_general %v865, %b8zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v867 = stablehlo.broadcast_in_dim %b8zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v868 = stablehlo.add %v866, %v867 : tensor<32x20xf32>
    %v869 = stablehlo.logistic %v868 : tensor<32x20xf32>
    %v870 = stablehlo.multiply %v868, %v869 : tensor<32x20xf32>
    %v871 = stablehlo.dot_general %v870, %b8zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v872 = stablehlo.broadcast_in_dim %b8zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v873 = stablehlo.add %v871, %v872 : tensor<32x480xf32>
    %v874 = stablehlo.reshape %v860 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v875 = stablehlo.constant dense<0.0> : tensor<f32>
    %v876 = stablehlo.reduce(%v874 init: %v875) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v877 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v878 = stablehlo.divide %v876, %v877 : tensor<32x480xf32>
    %v879 = stablehlo.dot_general %v878, %b8zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v880 = stablehlo.broadcast_in_dim %b8zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v881 = stablehlo.add %v879, %v880 : tensor<32x20xf32>
    %v882 = stablehlo.logistic %v881 : tensor<32x20xf32>
    %v883 = stablehlo.multiply %v881, %v882 : tensor<32x20xf32>
    %v884 = stablehlo.dot_general %v883, %b8zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v885 = stablehlo.broadcast_in_dim %b8zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v886 = stablehlo.add %v884, %v885 : tensor<32x480xf32>
    %v887 = stablehlo.logistic %v886 : tensor<32x480xf32>
    %v888 = stablehlo.broadcast_in_dim %v887, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v889 = stablehlo.multiply %v874, %v888 : tensor<32x480x14x14xf32>
    %v890 = stablehlo.reshape %v889 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v891 = stablehlo.reshape %v890 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v892 = stablehlo.convolution(%v891, %b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v893 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v894 = stablehlo.add %v892, %v893 : tensor<32x80x14x14xf32>
    %v895 = stablehlo.reshape %v894 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v896 = stablehlo.reshape %v895 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v897 = stablehlo.constant dense<0.0> : tensor<f32>
    %v898 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v899 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v900 = stablehlo.reduce(%v896 init: %v897) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v901 = stablehlo.broadcast_in_dim %v900, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v902 = stablehlo.divide %v901, %v898 : tensor<32x80x14x14xf32>
    %v903 = stablehlo.subtract %v896, %v902 : tensor<32x80x14x14xf32>
    %v904 = stablehlo.multiply %v903, %v903 : tensor<32x80x14x14xf32>
    %v905 = stablehlo.reduce(%v904 init: %v897) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v906 = stablehlo.broadcast_in_dim %v905, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v907 = stablehlo.divide %v906, %v898 : tensor<32x80x14x14xf32>
    %v908 = stablehlo.add %v907, %v899 : tensor<32x80x14x14xf32>
    %v909 = stablehlo.rsqrt %v908 : tensor<32x80x14x14xf32>
    %v910 = stablehlo.multiply %v903, %v909 : tensor<32x80x14x14xf32>
    %v911 = stablehlo.broadcast_in_dim %b8pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v912 = stablehlo.broadcast_in_dim %b8pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v913 = stablehlo.multiply %v910, %v911 : tensor<32x80x14x14xf32>
    %v914 = stablehlo.add %v913, %v912 : tensor<32x80x14x14xf32>
    %v915 = stablehlo.reshape %v914 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v916 = stablehlo.reshape %v915 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v917 = stablehlo.reshape %v802 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v918 = stablehlo.add %v916, %v917 : tensor<32x80x14x14xf32>
    %v919 = stablehlo.reshape %v918 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v920 = stablehlo.reshape %v919 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v921 = stablehlo.convolution(%v920, %b9eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v922 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v923 = stablehlo.add %v921, %v922 : tensor<32x480x14x14xf32>
    %v924 = stablehlo.reshape %v923 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v925 = stablehlo.reshape %v924 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v926 = stablehlo.constant dense<0.0> : tensor<f32>
    %v927 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v928 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v929 = stablehlo.reduce(%v925 init: %v926) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v930 = stablehlo.broadcast_in_dim %v929, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v931 = stablehlo.divide %v930, %v927 : tensor<32x480x14x14xf32>
    %v932 = stablehlo.subtract %v925, %v931 : tensor<32x480x14x14xf32>
    %v933 = stablehlo.multiply %v932, %v932 : tensor<32x480x14x14xf32>
    %v934 = stablehlo.reduce(%v933 init: %v926) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v935 = stablehlo.broadcast_in_dim %v934, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v936 = stablehlo.divide %v935, %v927 : tensor<32x480x14x14xf32>
    %v937 = stablehlo.add %v936, %v928 : tensor<32x480x14x14xf32>
    %v938 = stablehlo.rsqrt %v937 : tensor<32x480x14x14xf32>
    %v939 = stablehlo.multiply %v932, %v938 : tensor<32x480x14x14xf32>
    %v940 = stablehlo.broadcast_in_dim %b9eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v941 = stablehlo.broadcast_in_dim %b9ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v942 = stablehlo.multiply %v939, %v940 : tensor<32x480x14x14xf32>
    %v943 = stablehlo.add %v942, %v941 : tensor<32x480x14x14xf32>
    %v944 = stablehlo.reshape %v943 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v945 = stablehlo.reshape %v944 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v946 = stablehlo.logistic %v945 : tensor<32x480x14x14xf32>
    %v947 = stablehlo.multiply %v945, %v946 : tensor<32x480x14x14xf32>
    %v948 = stablehlo.reshape %v947 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v949 = stablehlo.reshape %v948 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v950 = stablehlo.convolution(%v949, %b9dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x5x5xf32>) -> tensor<32x480x14x14xf32>
    %v951 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v952 = stablehlo.add %v950, %v951 : tensor<32x480x14x14xf32>
    %v953 = stablehlo.reshape %v952 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v954 = stablehlo.reshape %v953 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v955 = stablehlo.constant dense<0.0> : tensor<f32>
    %v956 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v957 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v958 = stablehlo.reduce(%v954 init: %v955) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v959 = stablehlo.broadcast_in_dim %v958, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v960 = stablehlo.divide %v959, %v956 : tensor<32x480x14x14xf32>
    %v961 = stablehlo.subtract %v954, %v960 : tensor<32x480x14x14xf32>
    %v962 = stablehlo.multiply %v961, %v961 : tensor<32x480x14x14xf32>
    %v963 = stablehlo.reduce(%v962 init: %v955) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v964 = stablehlo.broadcast_in_dim %v963, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v965 = stablehlo.divide %v964, %v956 : tensor<32x480x14x14xf32>
    %v966 = stablehlo.add %v965, %v957 : tensor<32x480x14x14xf32>
    %v967 = stablehlo.rsqrt %v966 : tensor<32x480x14x14xf32>
    %v968 = stablehlo.multiply %v961, %v967 : tensor<32x480x14x14xf32>
    %v969 = stablehlo.broadcast_in_dim %b9dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v970 = stablehlo.broadcast_in_dim %b9dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v971 = stablehlo.multiply %v968, %v969 : tensor<32x480x14x14xf32>
    %v972 = stablehlo.add %v971, %v970 : tensor<32x480x14x14xf32>
    %v973 = stablehlo.reshape %v972 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v974 = stablehlo.reshape %v973 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v975 = stablehlo.logistic %v974 : tensor<32x480x14x14xf32>
    %v976 = stablehlo.multiply %v974, %v975 : tensor<32x480x14x14xf32>
    %v977 = stablehlo.reshape %v976 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v978 = stablehlo.reshape %v977 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v979 = stablehlo.constant dense<0.0> : tensor<f32>
    %v980 = stablehlo.reduce(%v978 init: %v979) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v981 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v982 = stablehlo.divide %v980, %v981 : tensor<32x480xf32>
    %v983 = stablehlo.dot_general %v982, %b9zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v984 = stablehlo.broadcast_in_dim %b9zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v985 = stablehlo.add %v983, %v984 : tensor<32x20xf32>
    %v986 = stablehlo.logistic %v985 : tensor<32x20xf32>
    %v987 = stablehlo.multiply %v985, %v986 : tensor<32x20xf32>
    %v988 = stablehlo.dot_general %v987, %b9zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v989 = stablehlo.broadcast_in_dim %b9zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v990 = stablehlo.add %v988, %v989 : tensor<32x480xf32>
    %v991 = stablehlo.reshape %v977 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v992 = stablehlo.constant dense<0.0> : tensor<f32>
    %v993 = stablehlo.reduce(%v991 init: %v992) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v994 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v995 = stablehlo.divide %v993, %v994 : tensor<32x480xf32>
    %v996 = stablehlo.dot_general %v995, %b9zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v997 = stablehlo.broadcast_in_dim %b9zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v998 = stablehlo.add %v996, %v997 : tensor<32x20xf32>
    %v999 = stablehlo.logistic %v998 : tensor<32x20xf32>
    %v1000 = stablehlo.multiply %v998, %v999 : tensor<32x20xf32>
    %v1001 = stablehlo.dot_general %v1000, %b9zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v1002 = stablehlo.broadcast_in_dim %b9zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v1003 = stablehlo.add %v1001, %v1002 : tensor<32x480xf32>
    %v1004 = stablehlo.logistic %v1003 : tensor<32x480xf32>
    %v1005 = stablehlo.broadcast_in_dim %v1004, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v1006 = stablehlo.multiply %v991, %v1005 : tensor<32x480x14x14xf32>
    %v1007 = stablehlo.reshape %v1006 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v1008 = stablehlo.reshape %v1007 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v1009 = stablehlo.convolution(%v1008, %b9pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<112x480x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v1010 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1011 = stablehlo.add %v1009, %v1010 : tensor<32x112x14x14xf32>
    %v1012 = stablehlo.reshape %v1011 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1013 = stablehlo.reshape %v1012 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1014 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1015 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v1016 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v1017 = stablehlo.reduce(%v1013 init: %v1014) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1018 = stablehlo.broadcast_in_dim %v1017, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1019 = stablehlo.divide %v1018, %v1015 : tensor<32x112x14x14xf32>
    %v1020 = stablehlo.subtract %v1013, %v1019 : tensor<32x112x14x14xf32>
    %v1021 = stablehlo.multiply %v1020, %v1020 : tensor<32x112x14x14xf32>
    %v1022 = stablehlo.reduce(%v1021 init: %v1014) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1023 = stablehlo.broadcast_in_dim %v1022, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1024 = stablehlo.divide %v1023, %v1015 : tensor<32x112x14x14xf32>
    %v1025 = stablehlo.add %v1024, %v1016 : tensor<32x112x14x14xf32>
    %v1026 = stablehlo.rsqrt %v1025 : tensor<32x112x14x14xf32>
    %v1027 = stablehlo.multiply %v1020, %v1026 : tensor<32x112x14x14xf32>
    %v1028 = stablehlo.broadcast_in_dim %b9pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1029 = stablehlo.broadcast_in_dim %b9pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1030 = stablehlo.multiply %v1027, %v1028 : tensor<32x112x14x14xf32>
    %v1031 = stablehlo.add %v1030, %v1029 : tensor<32x112x14x14xf32>
    %v1032 = stablehlo.reshape %v1031 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1033 = stablehlo.reshape %v1032 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1034 = stablehlo.convolution(%v1033, %b10eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v1035 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1036 = stablehlo.add %v1034, %v1035 : tensor<32x672x14x14xf32>
    %v1037 = stablehlo.reshape %v1036 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1038 = stablehlo.reshape %v1037 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1039 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1040 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v1041 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v1042 = stablehlo.reduce(%v1038 init: %v1039) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1043 = stablehlo.broadcast_in_dim %v1042, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1044 = stablehlo.divide %v1043, %v1040 : tensor<32x672x14x14xf32>
    %v1045 = stablehlo.subtract %v1038, %v1044 : tensor<32x672x14x14xf32>
    %v1046 = stablehlo.multiply %v1045, %v1045 : tensor<32x672x14x14xf32>
    %v1047 = stablehlo.reduce(%v1046 init: %v1039) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1048 = stablehlo.broadcast_in_dim %v1047, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1049 = stablehlo.divide %v1048, %v1040 : tensor<32x672x14x14xf32>
    %v1050 = stablehlo.add %v1049, %v1041 : tensor<32x672x14x14xf32>
    %v1051 = stablehlo.rsqrt %v1050 : tensor<32x672x14x14xf32>
    %v1052 = stablehlo.multiply %v1045, %v1051 : tensor<32x672x14x14xf32>
    %v1053 = stablehlo.broadcast_in_dim %b10eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1054 = stablehlo.broadcast_in_dim %b10ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1055 = stablehlo.multiply %v1052, %v1053 : tensor<32x672x14x14xf32>
    %v1056 = stablehlo.add %v1055, %v1054 : tensor<32x672x14x14xf32>
    %v1057 = stablehlo.reshape %v1056 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1058 = stablehlo.reshape %v1057 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1059 = stablehlo.logistic %v1058 : tensor<32x672x14x14xf32>
    %v1060 = stablehlo.multiply %v1058, %v1059 : tensor<32x672x14x14xf32>
    %v1061 = stablehlo.reshape %v1060 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1062 = stablehlo.reshape %v1061 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1063 = stablehlo.convolution(%v1062, %b10dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %v1064 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1065 = stablehlo.add %v1063, %v1064 : tensor<32x672x14x14xf32>
    %v1066 = stablehlo.reshape %v1065 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1067 = stablehlo.reshape %v1066 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1068 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1069 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v1070 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v1071 = stablehlo.reduce(%v1067 init: %v1068) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1072 = stablehlo.broadcast_in_dim %v1071, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1073 = stablehlo.divide %v1072, %v1069 : tensor<32x672x14x14xf32>
    %v1074 = stablehlo.subtract %v1067, %v1073 : tensor<32x672x14x14xf32>
    %v1075 = stablehlo.multiply %v1074, %v1074 : tensor<32x672x14x14xf32>
    %v1076 = stablehlo.reduce(%v1075 init: %v1068) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1077 = stablehlo.broadcast_in_dim %v1076, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1078 = stablehlo.divide %v1077, %v1069 : tensor<32x672x14x14xf32>
    %v1079 = stablehlo.add %v1078, %v1070 : tensor<32x672x14x14xf32>
    %v1080 = stablehlo.rsqrt %v1079 : tensor<32x672x14x14xf32>
    %v1081 = stablehlo.multiply %v1074, %v1080 : tensor<32x672x14x14xf32>
    %v1082 = stablehlo.broadcast_in_dim %b10dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1083 = stablehlo.broadcast_in_dim %b10dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1084 = stablehlo.multiply %v1081, %v1082 : tensor<32x672x14x14xf32>
    %v1085 = stablehlo.add %v1084, %v1083 : tensor<32x672x14x14xf32>
    %v1086 = stablehlo.reshape %v1085 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1087 = stablehlo.reshape %v1086 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1088 = stablehlo.logistic %v1087 : tensor<32x672x14x14xf32>
    %v1089 = stablehlo.multiply %v1087, %v1088 : tensor<32x672x14x14xf32>
    %v1090 = stablehlo.reshape %v1089 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1091 = stablehlo.reshape %v1090 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1092 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1093 = stablehlo.reduce(%v1091 init: %v1092) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v1094 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v1095 = stablehlo.divide %v1093, %v1094 : tensor<32x672xf32>
    %v1096 = stablehlo.dot_general %v1095, %b10zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v1097 = stablehlo.broadcast_in_dim %b10zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v1098 = stablehlo.add %v1096, %v1097 : tensor<32x28xf32>
    %v1099 = stablehlo.logistic %v1098 : tensor<32x28xf32>
    %v1100 = stablehlo.multiply %v1098, %v1099 : tensor<32x28xf32>
    %v1101 = stablehlo.dot_general %v1100, %b10zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v1102 = stablehlo.broadcast_in_dim %b10zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v1103 = stablehlo.add %v1101, %v1102 : tensor<32x672xf32>
    %v1104 = stablehlo.reshape %v1090 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1105 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1106 = stablehlo.reduce(%v1104 init: %v1105) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v1107 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v1108 = stablehlo.divide %v1106, %v1107 : tensor<32x672xf32>
    %v1109 = stablehlo.dot_general %v1108, %b10zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v1110 = stablehlo.broadcast_in_dim %b10zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v1111 = stablehlo.add %v1109, %v1110 : tensor<32x28xf32>
    %v1112 = stablehlo.logistic %v1111 : tensor<32x28xf32>
    %v1113 = stablehlo.multiply %v1111, %v1112 : tensor<32x28xf32>
    %v1114 = stablehlo.dot_general %v1113, %b10zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v1115 = stablehlo.broadcast_in_dim %b10zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v1116 = stablehlo.add %v1114, %v1115 : tensor<32x672xf32>
    %v1117 = stablehlo.logistic %v1116 : tensor<32x672xf32>
    %v1118 = stablehlo.broadcast_in_dim %v1117, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v1119 = stablehlo.multiply %v1104, %v1118 : tensor<32x672x14x14xf32>
    %v1120 = stablehlo.reshape %v1119 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1121 = stablehlo.reshape %v1120 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1122 = stablehlo.convolution(%v1121, %b10pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v1123 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1124 = stablehlo.add %v1122, %v1123 : tensor<32x112x14x14xf32>
    %v1125 = stablehlo.reshape %v1124 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1126 = stablehlo.reshape %v1125 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1127 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1128 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v1129 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v1130 = stablehlo.reduce(%v1126 init: %v1127) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1131 = stablehlo.broadcast_in_dim %v1130, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1132 = stablehlo.divide %v1131, %v1128 : tensor<32x112x14x14xf32>
    %v1133 = stablehlo.subtract %v1126, %v1132 : tensor<32x112x14x14xf32>
    %v1134 = stablehlo.multiply %v1133, %v1133 : tensor<32x112x14x14xf32>
    %v1135 = stablehlo.reduce(%v1134 init: %v1127) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1136 = stablehlo.broadcast_in_dim %v1135, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1137 = stablehlo.divide %v1136, %v1128 : tensor<32x112x14x14xf32>
    %v1138 = stablehlo.add %v1137, %v1129 : tensor<32x112x14x14xf32>
    %v1139 = stablehlo.rsqrt %v1138 : tensor<32x112x14x14xf32>
    %v1140 = stablehlo.multiply %v1133, %v1139 : tensor<32x112x14x14xf32>
    %v1141 = stablehlo.broadcast_in_dim %b10pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1142 = stablehlo.broadcast_in_dim %b10pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1143 = stablehlo.multiply %v1140, %v1141 : tensor<32x112x14x14xf32>
    %v1144 = stablehlo.add %v1143, %v1142 : tensor<32x112x14x14xf32>
    %v1145 = stablehlo.reshape %v1144 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1146 = stablehlo.reshape %v1145 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1147 = stablehlo.reshape %v1032 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1148 = stablehlo.add %v1146, %v1147 : tensor<32x112x14x14xf32>
    %v1149 = stablehlo.reshape %v1148 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1150 = stablehlo.reshape %v1149 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1151 = stablehlo.convolution(%v1150, %b11eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v1152 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1153 = stablehlo.add %v1151, %v1152 : tensor<32x672x14x14xf32>
    %v1154 = stablehlo.reshape %v1153 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1155 = stablehlo.reshape %v1154 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1156 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1157 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v1158 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v1159 = stablehlo.reduce(%v1155 init: %v1156) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1160 = stablehlo.broadcast_in_dim %v1159, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1161 = stablehlo.divide %v1160, %v1157 : tensor<32x672x14x14xf32>
    %v1162 = stablehlo.subtract %v1155, %v1161 : tensor<32x672x14x14xf32>
    %v1163 = stablehlo.multiply %v1162, %v1162 : tensor<32x672x14x14xf32>
    %v1164 = stablehlo.reduce(%v1163 init: %v1156) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1165 = stablehlo.broadcast_in_dim %v1164, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1166 = stablehlo.divide %v1165, %v1157 : tensor<32x672x14x14xf32>
    %v1167 = stablehlo.add %v1166, %v1158 : tensor<32x672x14x14xf32>
    %v1168 = stablehlo.rsqrt %v1167 : tensor<32x672x14x14xf32>
    %v1169 = stablehlo.multiply %v1162, %v1168 : tensor<32x672x14x14xf32>
    %v1170 = stablehlo.broadcast_in_dim %b11eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1171 = stablehlo.broadcast_in_dim %b11ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1172 = stablehlo.multiply %v1169, %v1170 : tensor<32x672x14x14xf32>
    %v1173 = stablehlo.add %v1172, %v1171 : tensor<32x672x14x14xf32>
    %v1174 = stablehlo.reshape %v1173 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1175 = stablehlo.reshape %v1174 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1176 = stablehlo.logistic %v1175 : tensor<32x672x14x14xf32>
    %v1177 = stablehlo.multiply %v1175, %v1176 : tensor<32x672x14x14xf32>
    %v1178 = stablehlo.reshape %v1177 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1179 = stablehlo.reshape %v1178 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1180 = stablehlo.convolution(%v1179, %b11dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %v1181 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1182 = stablehlo.add %v1180, %v1181 : tensor<32x672x14x14xf32>
    %v1183 = stablehlo.reshape %v1182 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1184 = stablehlo.reshape %v1183 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1185 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1186 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v1187 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v1188 = stablehlo.reduce(%v1184 init: %v1185) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1189 = stablehlo.broadcast_in_dim %v1188, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1190 = stablehlo.divide %v1189, %v1186 : tensor<32x672x14x14xf32>
    %v1191 = stablehlo.subtract %v1184, %v1190 : tensor<32x672x14x14xf32>
    %v1192 = stablehlo.multiply %v1191, %v1191 : tensor<32x672x14x14xf32>
    %v1193 = stablehlo.reduce(%v1192 init: %v1185) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1194 = stablehlo.broadcast_in_dim %v1193, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1195 = stablehlo.divide %v1194, %v1186 : tensor<32x672x14x14xf32>
    %v1196 = stablehlo.add %v1195, %v1187 : tensor<32x672x14x14xf32>
    %v1197 = stablehlo.rsqrt %v1196 : tensor<32x672x14x14xf32>
    %v1198 = stablehlo.multiply %v1191, %v1197 : tensor<32x672x14x14xf32>
    %v1199 = stablehlo.broadcast_in_dim %b11dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1200 = stablehlo.broadcast_in_dim %b11dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1201 = stablehlo.multiply %v1198, %v1199 : tensor<32x672x14x14xf32>
    %v1202 = stablehlo.add %v1201, %v1200 : tensor<32x672x14x14xf32>
    %v1203 = stablehlo.reshape %v1202 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1204 = stablehlo.reshape %v1203 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1205 = stablehlo.logistic %v1204 : tensor<32x672x14x14xf32>
    %v1206 = stablehlo.multiply %v1204, %v1205 : tensor<32x672x14x14xf32>
    %v1207 = stablehlo.reshape %v1206 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1208 = stablehlo.reshape %v1207 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1209 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1210 = stablehlo.reduce(%v1208 init: %v1209) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v1211 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v1212 = stablehlo.divide %v1210, %v1211 : tensor<32x672xf32>
    %v1213 = stablehlo.dot_general %v1212, %b11zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v1214 = stablehlo.broadcast_in_dim %b11zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v1215 = stablehlo.add %v1213, %v1214 : tensor<32x28xf32>
    %v1216 = stablehlo.logistic %v1215 : tensor<32x28xf32>
    %v1217 = stablehlo.multiply %v1215, %v1216 : tensor<32x28xf32>
    %v1218 = stablehlo.dot_general %v1217, %b11zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v1219 = stablehlo.broadcast_in_dim %b11zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v1220 = stablehlo.add %v1218, %v1219 : tensor<32x672xf32>
    %v1221 = stablehlo.reshape %v1207 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1222 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1223 = stablehlo.reduce(%v1221 init: %v1222) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v1224 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v1225 = stablehlo.divide %v1223, %v1224 : tensor<32x672xf32>
    %v1226 = stablehlo.dot_general %v1225, %b11zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v1227 = stablehlo.broadcast_in_dim %b11zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v1228 = stablehlo.add %v1226, %v1227 : tensor<32x28xf32>
    %v1229 = stablehlo.logistic %v1228 : tensor<32x28xf32>
    %v1230 = stablehlo.multiply %v1228, %v1229 : tensor<32x28xf32>
    %v1231 = stablehlo.dot_general %v1230, %b11zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v1232 = stablehlo.broadcast_in_dim %b11zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v1233 = stablehlo.add %v1231, %v1232 : tensor<32x672xf32>
    %v1234 = stablehlo.logistic %v1233 : tensor<32x672xf32>
    %v1235 = stablehlo.broadcast_in_dim %v1234, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v1236 = stablehlo.multiply %v1221, %v1235 : tensor<32x672x14x14xf32>
    %v1237 = stablehlo.reshape %v1236 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1238 = stablehlo.reshape %v1237 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1239 = stablehlo.convolution(%v1238, %b11pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v1240 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1241 = stablehlo.add %v1239, %v1240 : tensor<32x112x14x14xf32>
    %v1242 = stablehlo.reshape %v1241 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1243 = stablehlo.reshape %v1242 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1244 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1245 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v1246 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v1247 = stablehlo.reduce(%v1243 init: %v1244) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1248 = stablehlo.broadcast_in_dim %v1247, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1249 = stablehlo.divide %v1248, %v1245 : tensor<32x112x14x14xf32>
    %v1250 = stablehlo.subtract %v1243, %v1249 : tensor<32x112x14x14xf32>
    %v1251 = stablehlo.multiply %v1250, %v1250 : tensor<32x112x14x14xf32>
    %v1252 = stablehlo.reduce(%v1251 init: %v1244) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1253 = stablehlo.broadcast_in_dim %v1252, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1254 = stablehlo.divide %v1253, %v1245 : tensor<32x112x14x14xf32>
    %v1255 = stablehlo.add %v1254, %v1246 : tensor<32x112x14x14xf32>
    %v1256 = stablehlo.rsqrt %v1255 : tensor<32x112x14x14xf32>
    %v1257 = stablehlo.multiply %v1250, %v1256 : tensor<32x112x14x14xf32>
    %v1258 = stablehlo.broadcast_in_dim %b11pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1259 = stablehlo.broadcast_in_dim %b11pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1260 = stablehlo.multiply %v1257, %v1258 : tensor<32x112x14x14xf32>
    %v1261 = stablehlo.add %v1260, %v1259 : tensor<32x112x14x14xf32>
    %v1262 = stablehlo.reshape %v1261 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1263 = stablehlo.reshape %v1262 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1264 = stablehlo.reshape %v1149 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1265 = stablehlo.add %v1263, %v1264 : tensor<32x112x14x14xf32>
    %v1266 = stablehlo.reshape %v1265 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1267 = stablehlo.reshape %v1266 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1268 = stablehlo.convolution(%v1267, %b12eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v1269 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1270 = stablehlo.add %v1268, %v1269 : tensor<32x672x14x14xf32>
    %v1271 = stablehlo.reshape %v1270 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1272 = stablehlo.reshape %v1271 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1273 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1274 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v1275 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v1276 = stablehlo.reduce(%v1272 init: %v1273) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1277 = stablehlo.broadcast_in_dim %v1276, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1278 = stablehlo.divide %v1277, %v1274 : tensor<32x672x14x14xf32>
    %v1279 = stablehlo.subtract %v1272, %v1278 : tensor<32x672x14x14xf32>
    %v1280 = stablehlo.multiply %v1279, %v1279 : tensor<32x672x14x14xf32>
    %v1281 = stablehlo.reduce(%v1280 init: %v1273) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1282 = stablehlo.broadcast_in_dim %v1281, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1283 = stablehlo.divide %v1282, %v1274 : tensor<32x672x14x14xf32>
    %v1284 = stablehlo.add %v1283, %v1275 : tensor<32x672x14x14xf32>
    %v1285 = stablehlo.rsqrt %v1284 : tensor<32x672x14x14xf32>
    %v1286 = stablehlo.multiply %v1279, %v1285 : tensor<32x672x14x14xf32>
    %v1287 = stablehlo.broadcast_in_dim %b12eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1288 = stablehlo.broadcast_in_dim %b12ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1289 = stablehlo.multiply %v1286, %v1287 : tensor<32x672x14x14xf32>
    %v1290 = stablehlo.add %v1289, %v1288 : tensor<32x672x14x14xf32>
    %v1291 = stablehlo.reshape %v1290 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1292 = stablehlo.reshape %v1291 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1293 = stablehlo.logistic %v1292 : tensor<32x672x14x14xf32>
    %v1294 = stablehlo.multiply %v1292, %v1293 : tensor<32x672x14x14xf32>
    %v1295 = stablehlo.reshape %v1294 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1296 = stablehlo.reshape %v1295 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1297 = stablehlo.convolution(%v1296, %b12dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x7x7xf32>
    %v1298 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1299 = stablehlo.add %v1297, %v1298 : tensor<32x672x7x7xf32>
    %v1300 = stablehlo.reshape %v1299 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v1301 = stablehlo.reshape %v1300 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1302 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1303 = stablehlo.constant dense<1568.0> : tensor<32x672x7x7xf32>
    %v1304 = stablehlo.constant dense<1.0e-5> : tensor<32x672x7x7xf32>
    %v1305 = stablehlo.reduce(%v1301 init: %v1302) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v1306 = stablehlo.broadcast_in_dim %v1305, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1307 = stablehlo.divide %v1306, %v1303 : tensor<32x672x7x7xf32>
    %v1308 = stablehlo.subtract %v1301, %v1307 : tensor<32x672x7x7xf32>
    %v1309 = stablehlo.multiply %v1308, %v1308 : tensor<32x672x7x7xf32>
    %v1310 = stablehlo.reduce(%v1309 init: %v1302) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v1311 = stablehlo.broadcast_in_dim %v1310, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1312 = stablehlo.divide %v1311, %v1303 : tensor<32x672x7x7xf32>
    %v1313 = stablehlo.add %v1312, %v1304 : tensor<32x672x7x7xf32>
    %v1314 = stablehlo.rsqrt %v1313 : tensor<32x672x7x7xf32>
    %v1315 = stablehlo.multiply %v1308, %v1314 : tensor<32x672x7x7xf32>
    %v1316 = stablehlo.broadcast_in_dim %b12dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1317 = stablehlo.broadcast_in_dim %b12dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1318 = stablehlo.multiply %v1315, %v1316 : tensor<32x672x7x7xf32>
    %v1319 = stablehlo.add %v1318, %v1317 : tensor<32x672x7x7xf32>
    %v1320 = stablehlo.reshape %v1319 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v1321 = stablehlo.reshape %v1320 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1322 = stablehlo.logistic %v1321 : tensor<32x672x7x7xf32>
    %v1323 = stablehlo.multiply %v1321, %v1322 : tensor<32x672x7x7xf32>
    %v1324 = stablehlo.reshape %v1323 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v1325 = stablehlo.reshape %v1324 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1326 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1327 = stablehlo.reduce(%v1325 init: %v1326) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v1328 = stablehlo.constant dense<49.0> : tensor<32x672xf32>
    %v1329 = stablehlo.divide %v1327, %v1328 : tensor<32x672xf32>
    %v1330 = stablehlo.dot_general %v1329, %b12zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v1331 = stablehlo.broadcast_in_dim %b12zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v1332 = stablehlo.add %v1330, %v1331 : tensor<32x28xf32>
    %v1333 = stablehlo.logistic %v1332 : tensor<32x28xf32>
    %v1334 = stablehlo.multiply %v1332, %v1333 : tensor<32x28xf32>
    %v1335 = stablehlo.dot_general %v1334, %b12zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v1336 = stablehlo.broadcast_in_dim %b12zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v1337 = stablehlo.add %v1335, %v1336 : tensor<32x672xf32>
    %v1338 = stablehlo.reshape %v1324 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1339 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1340 = stablehlo.reduce(%v1338 init: %v1339) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v1341 = stablehlo.constant dense<49.0> : tensor<32x672xf32>
    %v1342 = stablehlo.divide %v1340, %v1341 : tensor<32x672xf32>
    %v1343 = stablehlo.dot_general %v1342, %b12zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v1344 = stablehlo.broadcast_in_dim %b12zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v1345 = stablehlo.add %v1343, %v1344 : tensor<32x28xf32>
    %v1346 = stablehlo.logistic %v1345 : tensor<32x28xf32>
    %v1347 = stablehlo.multiply %v1345, %v1346 : tensor<32x28xf32>
    %v1348 = stablehlo.dot_general %v1347, %b12zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v1349 = stablehlo.broadcast_in_dim %b12zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v1350 = stablehlo.add %v1348, %v1349 : tensor<32x672xf32>
    %v1351 = stablehlo.logistic %v1350 : tensor<32x672xf32>
    %v1352 = stablehlo.broadcast_in_dim %v1351, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x7x7xf32>
    %v1353 = stablehlo.multiply %v1338, %v1352 : tensor<32x672x7x7xf32>
    %v1354 = stablehlo.reshape %v1353 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v1355 = stablehlo.reshape %v1354 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1356 = stablehlo.convolution(%v1355, %b12pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x7x7xf32>, tensor<192x672x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1357 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1358 = stablehlo.add %v1356, %v1357 : tensor<32x192x7x7xf32>
    %v1359 = stablehlo.reshape %v1358 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1360 = stablehlo.reshape %v1359 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1361 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1362 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v1363 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1364 = stablehlo.reduce(%v1360 init: %v1361) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1365 = stablehlo.broadcast_in_dim %v1364, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1366 = stablehlo.divide %v1365, %v1362 : tensor<32x192x7x7xf32>
    %v1367 = stablehlo.subtract %v1360, %v1366 : tensor<32x192x7x7xf32>
    %v1368 = stablehlo.multiply %v1367, %v1367 : tensor<32x192x7x7xf32>
    %v1369 = stablehlo.reduce(%v1368 init: %v1361) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1370 = stablehlo.broadcast_in_dim %v1369, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1371 = stablehlo.divide %v1370, %v1362 : tensor<32x192x7x7xf32>
    %v1372 = stablehlo.add %v1371, %v1363 : tensor<32x192x7x7xf32>
    %v1373 = stablehlo.rsqrt %v1372 : tensor<32x192x7x7xf32>
    %v1374 = stablehlo.multiply %v1367, %v1373 : tensor<32x192x7x7xf32>
    %v1375 = stablehlo.broadcast_in_dim %b12pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1376 = stablehlo.broadcast_in_dim %b12pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1377 = stablehlo.multiply %v1374, %v1375 : tensor<32x192x7x7xf32>
    %v1378 = stablehlo.add %v1377, %v1376 : tensor<32x192x7x7xf32>
    %v1379 = stablehlo.reshape %v1378 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1380 = stablehlo.reshape %v1379 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1381 = stablehlo.convolution(%v1380, %b13eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1382 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1383 = stablehlo.add %v1381, %v1382 : tensor<32x1152x7x7xf32>
    %v1384 = stablehlo.reshape %v1383 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1385 = stablehlo.reshape %v1384 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1386 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1387 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1388 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1389 = stablehlo.reduce(%v1385 init: %v1386) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1390 = stablehlo.broadcast_in_dim %v1389, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1391 = stablehlo.divide %v1390, %v1387 : tensor<32x1152x7x7xf32>
    %v1392 = stablehlo.subtract %v1385, %v1391 : tensor<32x1152x7x7xf32>
    %v1393 = stablehlo.multiply %v1392, %v1392 : tensor<32x1152x7x7xf32>
    %v1394 = stablehlo.reduce(%v1393 init: %v1386) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1395 = stablehlo.broadcast_in_dim %v1394, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1396 = stablehlo.divide %v1395, %v1387 : tensor<32x1152x7x7xf32>
    %v1397 = stablehlo.add %v1396, %v1388 : tensor<32x1152x7x7xf32>
    %v1398 = stablehlo.rsqrt %v1397 : tensor<32x1152x7x7xf32>
    %v1399 = stablehlo.multiply %v1392, %v1398 : tensor<32x1152x7x7xf32>
    %v1400 = stablehlo.broadcast_in_dim %b13eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1401 = stablehlo.broadcast_in_dim %b13ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1402 = stablehlo.multiply %v1399, %v1400 : tensor<32x1152x7x7xf32>
    %v1403 = stablehlo.add %v1402, %v1401 : tensor<32x1152x7x7xf32>
    %v1404 = stablehlo.reshape %v1403 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1405 = stablehlo.reshape %v1404 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1406 = stablehlo.logistic %v1405 : tensor<32x1152x7x7xf32>
    %v1407 = stablehlo.multiply %v1405, %v1406 : tensor<32x1152x7x7xf32>
    %v1408 = stablehlo.reshape %v1407 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1409 = stablehlo.reshape %v1408 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1410 = stablehlo.convolution(%v1409, %b13dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v1411 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1412 = stablehlo.add %v1410, %v1411 : tensor<32x1152x7x7xf32>
    %v1413 = stablehlo.reshape %v1412 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1414 = stablehlo.reshape %v1413 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1415 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1416 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1417 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1418 = stablehlo.reduce(%v1414 init: %v1415) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1419 = stablehlo.broadcast_in_dim %v1418, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1420 = stablehlo.divide %v1419, %v1416 : tensor<32x1152x7x7xf32>
    %v1421 = stablehlo.subtract %v1414, %v1420 : tensor<32x1152x7x7xf32>
    %v1422 = stablehlo.multiply %v1421, %v1421 : tensor<32x1152x7x7xf32>
    %v1423 = stablehlo.reduce(%v1422 init: %v1415) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1424 = stablehlo.broadcast_in_dim %v1423, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1425 = stablehlo.divide %v1424, %v1416 : tensor<32x1152x7x7xf32>
    %v1426 = stablehlo.add %v1425, %v1417 : tensor<32x1152x7x7xf32>
    %v1427 = stablehlo.rsqrt %v1426 : tensor<32x1152x7x7xf32>
    %v1428 = stablehlo.multiply %v1421, %v1427 : tensor<32x1152x7x7xf32>
    %v1429 = stablehlo.broadcast_in_dim %b13dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1430 = stablehlo.broadcast_in_dim %b13dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1431 = stablehlo.multiply %v1428, %v1429 : tensor<32x1152x7x7xf32>
    %v1432 = stablehlo.add %v1431, %v1430 : tensor<32x1152x7x7xf32>
    %v1433 = stablehlo.reshape %v1432 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1434 = stablehlo.reshape %v1433 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1435 = stablehlo.logistic %v1434 : tensor<32x1152x7x7xf32>
    %v1436 = stablehlo.multiply %v1434, %v1435 : tensor<32x1152x7x7xf32>
    %v1437 = stablehlo.reshape %v1436 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1438 = stablehlo.reshape %v1437 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1439 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1440 = stablehlo.reduce(%v1438 init: %v1439) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1441 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1442 = stablehlo.divide %v1440, %v1441 : tensor<32x1152xf32>
    %v1443 = stablehlo.dot_general %v1442, %b13zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1444 = stablehlo.broadcast_in_dim %b13zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1445 = stablehlo.add %v1443, %v1444 : tensor<32x48xf32>
    %v1446 = stablehlo.logistic %v1445 : tensor<32x48xf32>
    %v1447 = stablehlo.multiply %v1445, %v1446 : tensor<32x48xf32>
    %v1448 = stablehlo.dot_general %v1447, %b13zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1449 = stablehlo.broadcast_in_dim %b13zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1450 = stablehlo.add %v1448, %v1449 : tensor<32x1152xf32>
    %v1451 = stablehlo.reshape %v1437 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1452 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1453 = stablehlo.reduce(%v1451 init: %v1452) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1454 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1455 = stablehlo.divide %v1453, %v1454 : tensor<32x1152xf32>
    %v1456 = stablehlo.dot_general %v1455, %b13zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1457 = stablehlo.broadcast_in_dim %b13zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1458 = stablehlo.add %v1456, %v1457 : tensor<32x48xf32>
    %v1459 = stablehlo.logistic %v1458 : tensor<32x48xf32>
    %v1460 = stablehlo.multiply %v1458, %v1459 : tensor<32x48xf32>
    %v1461 = stablehlo.dot_general %v1460, %b13zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1462 = stablehlo.broadcast_in_dim %b13zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1463 = stablehlo.add %v1461, %v1462 : tensor<32x1152xf32>
    %v1464 = stablehlo.logistic %v1463 : tensor<32x1152xf32>
    %v1465 = stablehlo.broadcast_in_dim %v1464, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1466 = stablehlo.multiply %v1451, %v1465 : tensor<32x1152x7x7xf32>
    %v1467 = stablehlo.reshape %v1466 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1468 = stablehlo.reshape %v1467 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1469 = stablehlo.convolution(%v1468, %b13pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1470 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1471 = stablehlo.add %v1469, %v1470 : tensor<32x192x7x7xf32>
    %v1472 = stablehlo.reshape %v1471 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1473 = stablehlo.reshape %v1472 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1474 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1475 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v1476 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1477 = stablehlo.reduce(%v1473 init: %v1474) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1478 = stablehlo.broadcast_in_dim %v1477, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1479 = stablehlo.divide %v1478, %v1475 : tensor<32x192x7x7xf32>
    %v1480 = stablehlo.subtract %v1473, %v1479 : tensor<32x192x7x7xf32>
    %v1481 = stablehlo.multiply %v1480, %v1480 : tensor<32x192x7x7xf32>
    %v1482 = stablehlo.reduce(%v1481 init: %v1474) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1483 = stablehlo.broadcast_in_dim %v1482, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1484 = stablehlo.divide %v1483, %v1475 : tensor<32x192x7x7xf32>
    %v1485 = stablehlo.add %v1484, %v1476 : tensor<32x192x7x7xf32>
    %v1486 = stablehlo.rsqrt %v1485 : tensor<32x192x7x7xf32>
    %v1487 = stablehlo.multiply %v1480, %v1486 : tensor<32x192x7x7xf32>
    %v1488 = stablehlo.broadcast_in_dim %b13pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1489 = stablehlo.broadcast_in_dim %b13pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1490 = stablehlo.multiply %v1487, %v1488 : tensor<32x192x7x7xf32>
    %v1491 = stablehlo.add %v1490, %v1489 : tensor<32x192x7x7xf32>
    %v1492 = stablehlo.reshape %v1491 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1493 = stablehlo.reshape %v1492 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1494 = stablehlo.reshape %v1379 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1495 = stablehlo.add %v1493, %v1494 : tensor<32x192x7x7xf32>
    %v1496 = stablehlo.reshape %v1495 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1497 = stablehlo.reshape %v1496 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1498 = stablehlo.convolution(%v1497, %b14eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1499 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1500 = stablehlo.add %v1498, %v1499 : tensor<32x1152x7x7xf32>
    %v1501 = stablehlo.reshape %v1500 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1502 = stablehlo.reshape %v1501 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1503 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1504 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1505 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1506 = stablehlo.reduce(%v1502 init: %v1503) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1507 = stablehlo.broadcast_in_dim %v1506, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1508 = stablehlo.divide %v1507, %v1504 : tensor<32x1152x7x7xf32>
    %v1509 = stablehlo.subtract %v1502, %v1508 : tensor<32x1152x7x7xf32>
    %v1510 = stablehlo.multiply %v1509, %v1509 : tensor<32x1152x7x7xf32>
    %v1511 = stablehlo.reduce(%v1510 init: %v1503) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1512 = stablehlo.broadcast_in_dim %v1511, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1513 = stablehlo.divide %v1512, %v1504 : tensor<32x1152x7x7xf32>
    %v1514 = stablehlo.add %v1513, %v1505 : tensor<32x1152x7x7xf32>
    %v1515 = stablehlo.rsqrt %v1514 : tensor<32x1152x7x7xf32>
    %v1516 = stablehlo.multiply %v1509, %v1515 : tensor<32x1152x7x7xf32>
    %v1517 = stablehlo.broadcast_in_dim %b14eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1518 = stablehlo.broadcast_in_dim %b14ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1519 = stablehlo.multiply %v1516, %v1517 : tensor<32x1152x7x7xf32>
    %v1520 = stablehlo.add %v1519, %v1518 : tensor<32x1152x7x7xf32>
    %v1521 = stablehlo.reshape %v1520 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1522 = stablehlo.reshape %v1521 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1523 = stablehlo.logistic %v1522 : tensor<32x1152x7x7xf32>
    %v1524 = stablehlo.multiply %v1522, %v1523 : tensor<32x1152x7x7xf32>
    %v1525 = stablehlo.reshape %v1524 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1526 = stablehlo.reshape %v1525 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1527 = stablehlo.convolution(%v1526, %b14dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v1528 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1529 = stablehlo.add %v1527, %v1528 : tensor<32x1152x7x7xf32>
    %v1530 = stablehlo.reshape %v1529 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1531 = stablehlo.reshape %v1530 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1532 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1533 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1534 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1535 = stablehlo.reduce(%v1531 init: %v1532) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1536 = stablehlo.broadcast_in_dim %v1535, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1537 = stablehlo.divide %v1536, %v1533 : tensor<32x1152x7x7xf32>
    %v1538 = stablehlo.subtract %v1531, %v1537 : tensor<32x1152x7x7xf32>
    %v1539 = stablehlo.multiply %v1538, %v1538 : tensor<32x1152x7x7xf32>
    %v1540 = stablehlo.reduce(%v1539 init: %v1532) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1541 = stablehlo.broadcast_in_dim %v1540, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1542 = stablehlo.divide %v1541, %v1533 : tensor<32x1152x7x7xf32>
    %v1543 = stablehlo.add %v1542, %v1534 : tensor<32x1152x7x7xf32>
    %v1544 = stablehlo.rsqrt %v1543 : tensor<32x1152x7x7xf32>
    %v1545 = stablehlo.multiply %v1538, %v1544 : tensor<32x1152x7x7xf32>
    %v1546 = stablehlo.broadcast_in_dim %b14dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1547 = stablehlo.broadcast_in_dim %b14dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1548 = stablehlo.multiply %v1545, %v1546 : tensor<32x1152x7x7xf32>
    %v1549 = stablehlo.add %v1548, %v1547 : tensor<32x1152x7x7xf32>
    %v1550 = stablehlo.reshape %v1549 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1551 = stablehlo.reshape %v1550 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1552 = stablehlo.logistic %v1551 : tensor<32x1152x7x7xf32>
    %v1553 = stablehlo.multiply %v1551, %v1552 : tensor<32x1152x7x7xf32>
    %v1554 = stablehlo.reshape %v1553 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1555 = stablehlo.reshape %v1554 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1556 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1557 = stablehlo.reduce(%v1555 init: %v1556) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1558 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1559 = stablehlo.divide %v1557, %v1558 : tensor<32x1152xf32>
    %v1560 = stablehlo.dot_general %v1559, %b14zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1561 = stablehlo.broadcast_in_dim %b14zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1562 = stablehlo.add %v1560, %v1561 : tensor<32x48xf32>
    %v1563 = stablehlo.logistic %v1562 : tensor<32x48xf32>
    %v1564 = stablehlo.multiply %v1562, %v1563 : tensor<32x48xf32>
    %v1565 = stablehlo.dot_general %v1564, %b14zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1566 = stablehlo.broadcast_in_dim %b14zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1567 = stablehlo.add %v1565, %v1566 : tensor<32x1152xf32>
    %v1568 = stablehlo.reshape %v1554 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1569 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1570 = stablehlo.reduce(%v1568 init: %v1569) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1571 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1572 = stablehlo.divide %v1570, %v1571 : tensor<32x1152xf32>
    %v1573 = stablehlo.dot_general %v1572, %b14zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1574 = stablehlo.broadcast_in_dim %b14zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1575 = stablehlo.add %v1573, %v1574 : tensor<32x48xf32>
    %v1576 = stablehlo.logistic %v1575 : tensor<32x48xf32>
    %v1577 = stablehlo.multiply %v1575, %v1576 : tensor<32x48xf32>
    %v1578 = stablehlo.dot_general %v1577, %b14zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1579 = stablehlo.broadcast_in_dim %b14zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1580 = stablehlo.add %v1578, %v1579 : tensor<32x1152xf32>
    %v1581 = stablehlo.logistic %v1580 : tensor<32x1152xf32>
    %v1582 = stablehlo.broadcast_in_dim %v1581, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1583 = stablehlo.multiply %v1568, %v1582 : tensor<32x1152x7x7xf32>
    %v1584 = stablehlo.reshape %v1583 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1585 = stablehlo.reshape %v1584 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1586 = stablehlo.convolution(%v1585, %b14pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1587 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1588 = stablehlo.add %v1586, %v1587 : tensor<32x192x7x7xf32>
    %v1589 = stablehlo.reshape %v1588 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1590 = stablehlo.reshape %v1589 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1591 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1592 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v1593 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1594 = stablehlo.reduce(%v1590 init: %v1591) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1595 = stablehlo.broadcast_in_dim %v1594, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1596 = stablehlo.divide %v1595, %v1592 : tensor<32x192x7x7xf32>
    %v1597 = stablehlo.subtract %v1590, %v1596 : tensor<32x192x7x7xf32>
    %v1598 = stablehlo.multiply %v1597, %v1597 : tensor<32x192x7x7xf32>
    %v1599 = stablehlo.reduce(%v1598 init: %v1591) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1600 = stablehlo.broadcast_in_dim %v1599, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1601 = stablehlo.divide %v1600, %v1592 : tensor<32x192x7x7xf32>
    %v1602 = stablehlo.add %v1601, %v1593 : tensor<32x192x7x7xf32>
    %v1603 = stablehlo.rsqrt %v1602 : tensor<32x192x7x7xf32>
    %v1604 = stablehlo.multiply %v1597, %v1603 : tensor<32x192x7x7xf32>
    %v1605 = stablehlo.broadcast_in_dim %b14pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1606 = stablehlo.broadcast_in_dim %b14pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1607 = stablehlo.multiply %v1604, %v1605 : tensor<32x192x7x7xf32>
    %v1608 = stablehlo.add %v1607, %v1606 : tensor<32x192x7x7xf32>
    %v1609 = stablehlo.reshape %v1608 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1610 = stablehlo.reshape %v1609 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1611 = stablehlo.reshape %v1496 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1612 = stablehlo.add %v1610, %v1611 : tensor<32x192x7x7xf32>
    %v1613 = stablehlo.reshape %v1612 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1614 = stablehlo.reshape %v1613 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1615 = stablehlo.convolution(%v1614, %b15eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1616 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1617 = stablehlo.add %v1615, %v1616 : tensor<32x1152x7x7xf32>
    %v1618 = stablehlo.reshape %v1617 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1619 = stablehlo.reshape %v1618 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1620 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1621 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1622 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1623 = stablehlo.reduce(%v1619 init: %v1620) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1624 = stablehlo.broadcast_in_dim %v1623, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1625 = stablehlo.divide %v1624, %v1621 : tensor<32x1152x7x7xf32>
    %v1626 = stablehlo.subtract %v1619, %v1625 : tensor<32x1152x7x7xf32>
    %v1627 = stablehlo.multiply %v1626, %v1626 : tensor<32x1152x7x7xf32>
    %v1628 = stablehlo.reduce(%v1627 init: %v1620) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1629 = stablehlo.broadcast_in_dim %v1628, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1630 = stablehlo.divide %v1629, %v1621 : tensor<32x1152x7x7xf32>
    %v1631 = stablehlo.add %v1630, %v1622 : tensor<32x1152x7x7xf32>
    %v1632 = stablehlo.rsqrt %v1631 : tensor<32x1152x7x7xf32>
    %v1633 = stablehlo.multiply %v1626, %v1632 : tensor<32x1152x7x7xf32>
    %v1634 = stablehlo.broadcast_in_dim %b15eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1635 = stablehlo.broadcast_in_dim %b15ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1636 = stablehlo.multiply %v1633, %v1634 : tensor<32x1152x7x7xf32>
    %v1637 = stablehlo.add %v1636, %v1635 : tensor<32x1152x7x7xf32>
    %v1638 = stablehlo.reshape %v1637 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1639 = stablehlo.reshape %v1638 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1640 = stablehlo.logistic %v1639 : tensor<32x1152x7x7xf32>
    %v1641 = stablehlo.multiply %v1639, %v1640 : tensor<32x1152x7x7xf32>
    %v1642 = stablehlo.reshape %v1641 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1643 = stablehlo.reshape %v1642 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1644 = stablehlo.convolution(%v1643, %b15dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v1645 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1646 = stablehlo.add %v1644, %v1645 : tensor<32x1152x7x7xf32>
    %v1647 = stablehlo.reshape %v1646 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1648 = stablehlo.reshape %v1647 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1649 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1650 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1651 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1652 = stablehlo.reduce(%v1648 init: %v1649) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1653 = stablehlo.broadcast_in_dim %v1652, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1654 = stablehlo.divide %v1653, %v1650 : tensor<32x1152x7x7xf32>
    %v1655 = stablehlo.subtract %v1648, %v1654 : tensor<32x1152x7x7xf32>
    %v1656 = stablehlo.multiply %v1655, %v1655 : tensor<32x1152x7x7xf32>
    %v1657 = stablehlo.reduce(%v1656 init: %v1649) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1658 = stablehlo.broadcast_in_dim %v1657, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1659 = stablehlo.divide %v1658, %v1650 : tensor<32x1152x7x7xf32>
    %v1660 = stablehlo.add %v1659, %v1651 : tensor<32x1152x7x7xf32>
    %v1661 = stablehlo.rsqrt %v1660 : tensor<32x1152x7x7xf32>
    %v1662 = stablehlo.multiply %v1655, %v1661 : tensor<32x1152x7x7xf32>
    %v1663 = stablehlo.broadcast_in_dim %b15dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1664 = stablehlo.broadcast_in_dim %b15dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1665 = stablehlo.multiply %v1662, %v1663 : tensor<32x1152x7x7xf32>
    %v1666 = stablehlo.add %v1665, %v1664 : tensor<32x1152x7x7xf32>
    %v1667 = stablehlo.reshape %v1666 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1668 = stablehlo.reshape %v1667 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1669 = stablehlo.logistic %v1668 : tensor<32x1152x7x7xf32>
    %v1670 = stablehlo.multiply %v1668, %v1669 : tensor<32x1152x7x7xf32>
    %v1671 = stablehlo.reshape %v1670 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1672 = stablehlo.reshape %v1671 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1673 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1674 = stablehlo.reduce(%v1672 init: %v1673) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1675 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1676 = stablehlo.divide %v1674, %v1675 : tensor<32x1152xf32>
    %v1677 = stablehlo.dot_general %v1676, %b15zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1678 = stablehlo.broadcast_in_dim %b15zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1679 = stablehlo.add %v1677, %v1678 : tensor<32x48xf32>
    %v1680 = stablehlo.logistic %v1679 : tensor<32x48xf32>
    %v1681 = stablehlo.multiply %v1679, %v1680 : tensor<32x48xf32>
    %v1682 = stablehlo.dot_general %v1681, %b15zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1683 = stablehlo.broadcast_in_dim %b15zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1684 = stablehlo.add %v1682, %v1683 : tensor<32x1152xf32>
    %v1685 = stablehlo.reshape %v1671 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1686 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1687 = stablehlo.reduce(%v1685 init: %v1686) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1688 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1689 = stablehlo.divide %v1687, %v1688 : tensor<32x1152xf32>
    %v1690 = stablehlo.dot_general %v1689, %b15zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1691 = stablehlo.broadcast_in_dim %b15zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1692 = stablehlo.add %v1690, %v1691 : tensor<32x48xf32>
    %v1693 = stablehlo.logistic %v1692 : tensor<32x48xf32>
    %v1694 = stablehlo.multiply %v1692, %v1693 : tensor<32x48xf32>
    %v1695 = stablehlo.dot_general %v1694, %b15zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1696 = stablehlo.broadcast_in_dim %b15zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1697 = stablehlo.add %v1695, %v1696 : tensor<32x1152xf32>
    %v1698 = stablehlo.logistic %v1697 : tensor<32x1152xf32>
    %v1699 = stablehlo.broadcast_in_dim %v1698, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1700 = stablehlo.multiply %v1685, %v1699 : tensor<32x1152x7x7xf32>
    %v1701 = stablehlo.reshape %v1700 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1702 = stablehlo.reshape %v1701 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1703 = stablehlo.convolution(%v1702, %b15pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1704 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1705 = stablehlo.add %v1703, %v1704 : tensor<32x192x7x7xf32>
    %v1706 = stablehlo.reshape %v1705 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1707 = stablehlo.reshape %v1706 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1708 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1709 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v1710 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1711 = stablehlo.reduce(%v1707 init: %v1708) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1712 = stablehlo.broadcast_in_dim %v1711, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1713 = stablehlo.divide %v1712, %v1709 : tensor<32x192x7x7xf32>
    %v1714 = stablehlo.subtract %v1707, %v1713 : tensor<32x192x7x7xf32>
    %v1715 = stablehlo.multiply %v1714, %v1714 : tensor<32x192x7x7xf32>
    %v1716 = stablehlo.reduce(%v1715 init: %v1708) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1717 = stablehlo.broadcast_in_dim %v1716, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1718 = stablehlo.divide %v1717, %v1709 : tensor<32x192x7x7xf32>
    %v1719 = stablehlo.add %v1718, %v1710 : tensor<32x192x7x7xf32>
    %v1720 = stablehlo.rsqrt %v1719 : tensor<32x192x7x7xf32>
    %v1721 = stablehlo.multiply %v1714, %v1720 : tensor<32x192x7x7xf32>
    %v1722 = stablehlo.broadcast_in_dim %b15pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1723 = stablehlo.broadcast_in_dim %b15pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1724 = stablehlo.multiply %v1721, %v1722 : tensor<32x192x7x7xf32>
    %v1725 = stablehlo.add %v1724, %v1723 : tensor<32x192x7x7xf32>
    %v1726 = stablehlo.reshape %v1725 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1727 = stablehlo.reshape %v1726 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1728 = stablehlo.reshape %v1613 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1729 = stablehlo.add %v1727, %v1728 : tensor<32x192x7x7xf32>
    %v1730 = stablehlo.reshape %v1729 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1731 = stablehlo.reshape %v1730 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1732 = stablehlo.convolution(%v1731, %b16eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1733 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1734 = stablehlo.add %v1732, %v1733 : tensor<32x1152x7x7xf32>
    %v1735 = stablehlo.reshape %v1734 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1736 = stablehlo.reshape %v1735 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1737 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1738 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1739 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1740 = stablehlo.reduce(%v1736 init: %v1737) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1741 = stablehlo.broadcast_in_dim %v1740, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1742 = stablehlo.divide %v1741, %v1738 : tensor<32x1152x7x7xf32>
    %v1743 = stablehlo.subtract %v1736, %v1742 : tensor<32x1152x7x7xf32>
    %v1744 = stablehlo.multiply %v1743, %v1743 : tensor<32x1152x7x7xf32>
    %v1745 = stablehlo.reduce(%v1744 init: %v1737) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1746 = stablehlo.broadcast_in_dim %v1745, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1747 = stablehlo.divide %v1746, %v1738 : tensor<32x1152x7x7xf32>
    %v1748 = stablehlo.add %v1747, %v1739 : tensor<32x1152x7x7xf32>
    %v1749 = stablehlo.rsqrt %v1748 : tensor<32x1152x7x7xf32>
    %v1750 = stablehlo.multiply %v1743, %v1749 : tensor<32x1152x7x7xf32>
    %v1751 = stablehlo.broadcast_in_dim %b16eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1752 = stablehlo.broadcast_in_dim %b16ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1753 = stablehlo.multiply %v1750, %v1751 : tensor<32x1152x7x7xf32>
    %v1754 = stablehlo.add %v1753, %v1752 : tensor<32x1152x7x7xf32>
    %v1755 = stablehlo.reshape %v1754 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1756 = stablehlo.reshape %v1755 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1757 = stablehlo.logistic %v1756 : tensor<32x1152x7x7xf32>
    %v1758 = stablehlo.multiply %v1756, %v1757 : tensor<32x1152x7x7xf32>
    %v1759 = stablehlo.reshape %v1758 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1760 = stablehlo.reshape %v1759 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1761 = stablehlo.convolution(%v1760, %b16dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x3x3xf32>) -> tensor<32x1152x7x7xf32>
    %v1762 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1763 = stablehlo.add %v1761, %v1762 : tensor<32x1152x7x7xf32>
    %v1764 = stablehlo.reshape %v1763 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1765 = stablehlo.reshape %v1764 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1766 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1767 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1768 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1769 = stablehlo.reduce(%v1765 init: %v1766) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1770 = stablehlo.broadcast_in_dim %v1769, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1771 = stablehlo.divide %v1770, %v1767 : tensor<32x1152x7x7xf32>
    %v1772 = stablehlo.subtract %v1765, %v1771 : tensor<32x1152x7x7xf32>
    %v1773 = stablehlo.multiply %v1772, %v1772 : tensor<32x1152x7x7xf32>
    %v1774 = stablehlo.reduce(%v1773 init: %v1766) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1775 = stablehlo.broadcast_in_dim %v1774, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1776 = stablehlo.divide %v1775, %v1767 : tensor<32x1152x7x7xf32>
    %v1777 = stablehlo.add %v1776, %v1768 : tensor<32x1152x7x7xf32>
    %v1778 = stablehlo.rsqrt %v1777 : tensor<32x1152x7x7xf32>
    %v1779 = stablehlo.multiply %v1772, %v1778 : tensor<32x1152x7x7xf32>
    %v1780 = stablehlo.broadcast_in_dim %b16dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1781 = stablehlo.broadcast_in_dim %b16dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1782 = stablehlo.multiply %v1779, %v1780 : tensor<32x1152x7x7xf32>
    %v1783 = stablehlo.add %v1782, %v1781 : tensor<32x1152x7x7xf32>
    %v1784 = stablehlo.reshape %v1783 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1785 = stablehlo.reshape %v1784 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1786 = stablehlo.logistic %v1785 : tensor<32x1152x7x7xf32>
    %v1787 = stablehlo.multiply %v1785, %v1786 : tensor<32x1152x7x7xf32>
    %v1788 = stablehlo.reshape %v1787 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1789 = stablehlo.reshape %v1788 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1790 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1791 = stablehlo.reduce(%v1789 init: %v1790) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1792 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1793 = stablehlo.divide %v1791, %v1792 : tensor<32x1152xf32>
    %v1794 = stablehlo.dot_general %v1793, %b16zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1795 = stablehlo.broadcast_in_dim %b16zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1796 = stablehlo.add %v1794, %v1795 : tensor<32x48xf32>
    %v1797 = stablehlo.logistic %v1796 : tensor<32x48xf32>
    %v1798 = stablehlo.multiply %v1796, %v1797 : tensor<32x48xf32>
    %v1799 = stablehlo.dot_general %v1798, %b16zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1800 = stablehlo.broadcast_in_dim %b16zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1801 = stablehlo.add %v1799, %v1800 : tensor<32x1152xf32>
    %v1802 = stablehlo.reshape %v1788 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1803 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1804 = stablehlo.reduce(%v1802 init: %v1803) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1805 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1806 = stablehlo.divide %v1804, %v1805 : tensor<32x1152xf32>
    %v1807 = stablehlo.dot_general %v1806, %b16zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1808 = stablehlo.broadcast_in_dim %b16zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1809 = stablehlo.add %v1807, %v1808 : tensor<32x48xf32>
    %v1810 = stablehlo.logistic %v1809 : tensor<32x48xf32>
    %v1811 = stablehlo.multiply %v1809, %v1810 : tensor<32x48xf32>
    %v1812 = stablehlo.dot_general %v1811, %b16zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1813 = stablehlo.broadcast_in_dim %b16zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1814 = stablehlo.add %v1812, %v1813 : tensor<32x1152xf32>
    %v1815 = stablehlo.logistic %v1814 : tensor<32x1152xf32>
    %v1816 = stablehlo.broadcast_in_dim %v1815, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1817 = stablehlo.multiply %v1802, %v1816 : tensor<32x1152x7x7xf32>
    %v1818 = stablehlo.reshape %v1817 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1819 = stablehlo.reshape %v1818 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1820 = stablehlo.convolution(%v1819, %b16pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<320x1152x1x1xf32>) -> tensor<32x320x7x7xf32>
    %v1821 = stablehlo.broadcast_in_dim %zb320, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1822 = stablehlo.add %v1820, %v1821 : tensor<32x320x7x7xf32>
    %v1823 = stablehlo.reshape %v1822 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1824 = stablehlo.reshape %v1823 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1825 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1826 = stablehlo.constant dense<1568.0> : tensor<32x320x7x7xf32>
    %v1827 = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %v1828 = stablehlo.reduce(%v1824 init: %v1825) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1829 = stablehlo.broadcast_in_dim %v1828, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1830 = stablehlo.divide %v1829, %v1826 : tensor<32x320x7x7xf32>
    %v1831 = stablehlo.subtract %v1824, %v1830 : tensor<32x320x7x7xf32>
    %v1832 = stablehlo.multiply %v1831, %v1831 : tensor<32x320x7x7xf32>
    %v1833 = stablehlo.reduce(%v1832 init: %v1825) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1834 = stablehlo.broadcast_in_dim %v1833, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1835 = stablehlo.divide %v1834, %v1826 : tensor<32x320x7x7xf32>
    %v1836 = stablehlo.add %v1835, %v1827 : tensor<32x320x7x7xf32>
    %v1837 = stablehlo.rsqrt %v1836 : tensor<32x320x7x7xf32>
    %v1838 = stablehlo.multiply %v1831, %v1837 : tensor<32x320x7x7xf32>
    %v1839 = stablehlo.broadcast_in_dim %b16pg, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1840 = stablehlo.broadcast_in_dim %b16pbt, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1841 = stablehlo.multiply %v1838, %v1839 : tensor<32x320x7x7xf32>
    %v1842 = stablehlo.add %v1841, %v1840 : tensor<32x320x7x7xf32>
    %v1843 = stablehlo.reshape %v1842 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1844 = stablehlo.reshape %v1843 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1845 = stablehlo.convolution(%v1844, %hW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x320x7x7xf32>, tensor<1280x320x1x1xf32>) -> tensor<32x1280x7x7xf32>
    %v1846 = stablehlo.broadcast_in_dim %zb1280, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1847 = stablehlo.add %v1845, %v1846 : tensor<32x1280x7x7xf32>
    %v1848 = stablehlo.reshape %v1847 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1849 = stablehlo.reshape %v1848 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1850 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1851 = stablehlo.constant dense<1568.0> : tensor<32x1280x7x7xf32>
    %v1852 = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %v1853 = stablehlo.reduce(%v1849 init: %v1850) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1854 = stablehlo.broadcast_in_dim %v1853, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1855 = stablehlo.divide %v1854, %v1851 : tensor<32x1280x7x7xf32>
    %v1856 = stablehlo.subtract %v1849, %v1855 : tensor<32x1280x7x7xf32>
    %v1857 = stablehlo.multiply %v1856, %v1856 : tensor<32x1280x7x7xf32>
    %v1858 = stablehlo.reduce(%v1857 init: %v1850) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1859 = stablehlo.broadcast_in_dim %v1858, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1860 = stablehlo.divide %v1859, %v1851 : tensor<32x1280x7x7xf32>
    %v1861 = stablehlo.add %v1860, %v1852 : tensor<32x1280x7x7xf32>
    %v1862 = stablehlo.rsqrt %v1861 : tensor<32x1280x7x7xf32>
    %v1863 = stablehlo.multiply %v1856, %v1862 : tensor<32x1280x7x7xf32>
    %v1864 = stablehlo.broadcast_in_dim %hg, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1865 = stablehlo.broadcast_in_dim %hbt, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1866 = stablehlo.multiply %v1863, %v1864 : tensor<32x1280x7x7xf32>
    %v1867 = stablehlo.add %v1866, %v1865 : tensor<32x1280x7x7xf32>
    %v1868 = stablehlo.reshape %v1867 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1869 = stablehlo.reshape %v1868 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1870 = stablehlo.logistic %v1869 : tensor<32x1280x7x7xf32>
    %v1871 = stablehlo.multiply %v1869, %v1870 : tensor<32x1280x7x7xf32>
    %v1872 = stablehlo.reshape %v1871 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1873 = stablehlo.reshape %v1872 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1874 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1875 = stablehlo.reduce(%v1873 init: %v1874) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v1876 = stablehlo.constant dense<49.0> : tensor<32x1280xf32>
    %v1877 = stablehlo.divide %v1875, %v1876 : tensor<32x1280xf32>
    %v1878 = stablehlo.dot_general %v1877, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1280xf32>, tensor<1280x10xf32>) -> tensor<32x10xf32>
    %v1879 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v1880 = stablehlo.add %v1878, %v1879 : tensor<32x10xf32>
    %v1881 = stablehlo.reshape %v1880 : (tensor<32x10xf32>) -> tensor<32x1x10xf32>
    %v1882 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1883 = stablehlo.exponential %v1881 : tensor<32x1x10xf32>
    %v1884 = stablehlo.reduce(%v1883 init: %v1882) applies stablehlo.add across dimensions = [2] : (tensor<32x1x10xf32>, tensor<f32>) -> tensor<32x1xf32>
    %v1885 = stablehlo.broadcast_in_dim %v1884, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x10xf32>
    %v1886 = stablehlo.divide %v1883, %v1885 : tensor<32x1x10xf32>
    %v1887 = stablehlo.reshape %v1886 : (tensor<32x1x10xf32>) -> tensor<32x10xf32>
    %v1888 = stablehlo.subtract %v1887, %onehot : tensor<32x10xf32>
    %v1889 = stablehlo.reshape %v1888 : (tensor<32x10xf32>) -> tensor<32x1x10xf32>
    %v1890 = stablehlo.dot_general %v1889, %Wd, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x10xf32>, tensor<1280x10xf32>) -> tensor<32x1x1280xf32>
    %v1891 = stablehlo.reshape %v1890 : (tensor<32x1x1280xf32>) -> tensor<32x1280xf32>
    %v1892 = stablehlo.dot_general %v1877, %v1888, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1280xf32>, tensor<32x10xf32>) -> tensor<1280x10xf32>
    %v1893 = stablehlo.constant dense<0.05> : tensor<1280x10xf32>
    %v1894 = stablehlo.multiply %v1892, %v1893 : tensor<1280x10xf32>
    %v1895 = stablehlo.subtract %Wd, %v1894 : tensor<1280x10xf32>
    %v1896 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1897 = stablehlo.reduce(%v1888 init: %v1896) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1898 = stablehlo.constant dense<0.05> : tensor<10xf32>
    %v1899 = stablehlo.multiply %v1897, %v1898 : tensor<10xf32>
    %v1900 = stablehlo.subtract %bd, %v1899 : tensor<10xf32>
    %v1901 = stablehlo.broadcast_in_dim %v1891, dims = [0, 1] : (tensor<32x1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1902 = stablehlo.constant dense<49.0> : tensor<32x1280x7x7xf32>
    %v1903 = stablehlo.divide %v1901, %v1902 : tensor<32x1280x7x7xf32>
    %v1904 = stablehlo.reshape %v1903 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1905 = stablehlo.reshape %v1904 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1906 = stablehlo.reshape %v1868 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1907 = stablehlo.logistic %v1906 : tensor<32x1280x7x7xf32>
    %v1908 = stablehlo.constant dense<1.0> : tensor<32x1280x7x7xf32>
    %v1909 = stablehlo.subtract %v1908, %v1907 : tensor<32x1280x7x7xf32>
    %v1910 = stablehlo.multiply %v1906, %v1909 : tensor<32x1280x7x7xf32>
    %v1911 = stablehlo.add %v1908, %v1910 : tensor<32x1280x7x7xf32>
    %v1912 = stablehlo.multiply %v1907, %v1911 : tensor<32x1280x7x7xf32>
    %v1913 = stablehlo.multiply %v1905, %v1912 : tensor<32x1280x7x7xf32>
    %v1914 = stablehlo.reshape %v1913 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1915 = stablehlo.reshape %v1848 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1916 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1917 = stablehlo.constant dense<1568.0> : tensor<32x1280x7x7xf32>
    %v1918 = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %v1919 = stablehlo.reduce(%v1915 init: %v1916) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1920 = stablehlo.broadcast_in_dim %v1919, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1921 = stablehlo.divide %v1920, %v1917 : tensor<32x1280x7x7xf32>
    %v1922 = stablehlo.subtract %v1915, %v1921 : tensor<32x1280x7x7xf32>
    %v1923 = stablehlo.multiply %v1922, %v1922 : tensor<32x1280x7x7xf32>
    %v1924 = stablehlo.reduce(%v1923 init: %v1916) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1925 = stablehlo.broadcast_in_dim %v1924, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1926 = stablehlo.divide %v1925, %v1917 : tensor<32x1280x7x7xf32>
    %v1927 = stablehlo.add %v1926, %v1918 : tensor<32x1280x7x7xf32>
    %v1928 = stablehlo.rsqrt %v1927 : tensor<32x1280x7x7xf32>
    %v1929 = stablehlo.multiply %v1922, %v1928 : tensor<32x1280x7x7xf32>
    %v1930 = stablehlo.broadcast_in_dim %hg, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1931 = stablehlo.reshape %v1914 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1932 = stablehlo.multiply %v1930, %v1931 : tensor<32x1280x7x7xf32>
    %v1933 = stablehlo.reduce(%v1932 init: %v1916) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1934 = stablehlo.broadcast_in_dim %v1933, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1935 = stablehlo.multiply %v1929, %v1932 : tensor<32x1280x7x7xf32>
    %v1936 = stablehlo.reduce(%v1935 init: %v1916) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1937 = stablehlo.broadcast_in_dim %v1936, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1938 = stablehlo.multiply %v1932, %v1917 : tensor<32x1280x7x7xf32>
    %v1939 = stablehlo.subtract %v1938, %v1934 : tensor<32x1280x7x7xf32>
    %v1940 = stablehlo.multiply %v1929, %v1937 : tensor<32x1280x7x7xf32>
    %v1941 = stablehlo.subtract %v1939, %v1940 : tensor<32x1280x7x7xf32>
    %v1942 = stablehlo.divide %v1928, %v1917 : tensor<32x1280x7x7xf32>
    %v1943 = stablehlo.multiply %v1942, %v1941 : tensor<32x1280x7x7xf32>
    %v1944 = stablehlo.reshape %v1943 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1945 = stablehlo.reshape %v1944 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1946 = stablehlo.reverse %hW, dims = [2, 3] : tensor<1280x320x1x1xf32>
    %v1947 = stablehlo.transpose %v1946, dims = [1, 0, 2, 3] : (tensor<1280x320x1x1xf32>) -> tensor<320x1280x1x1xf32>
    %v1948 = stablehlo.convolution(%v1945, %v1947)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1280x7x7xf32>, tensor<320x1280x1x1xf32>) -> tensor<32x320x7x7xf32>
    %v1949 = stablehlo.reshape %v1948 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1950 = stablehlo.reshape %v1848 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1951 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1952 = stablehlo.constant dense<1568.0> : tensor<32x1280x7x7xf32>
    %v1953 = stablehlo.reduce(%v1950 init: %v1951) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1954 = stablehlo.broadcast_in_dim %v1953, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1955 = stablehlo.divide %v1954, %v1952 : tensor<32x1280x7x7xf32>
    %v1956 = stablehlo.subtract %v1950, %v1955 : tensor<32x1280x7x7xf32>
    %v1957 = stablehlo.multiply %v1956, %v1956 : tensor<32x1280x7x7xf32>
    %v1958 = stablehlo.reduce(%v1957 init: %v1951) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1959 = stablehlo.broadcast_in_dim %v1958, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1960 = stablehlo.divide %v1959, %v1952 : tensor<32x1280x7x7xf32>
    %v1961 = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %v1962 = stablehlo.add %v1960, %v1961 : tensor<32x1280x7x7xf32>
    %v1963 = stablehlo.rsqrt %v1962 : tensor<32x1280x7x7xf32>
    %v1964 = stablehlo.multiply %v1956, %v1963 : tensor<32x1280x7x7xf32>
    %v1965 = stablehlo.reshape %v1914 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1966 = stablehlo.multiply %v1965, %v1964 : tensor<32x1280x7x7xf32>
    %v1967 = stablehlo.reduce(%v1966 init: %v1951) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1968 = stablehlo.constant dense<0.05> : tensor<1280xf32>
    %v1969 = stablehlo.multiply %v1967, %v1968 : tensor<1280xf32>
    %v1970 = stablehlo.subtract %hg, %v1969 : tensor<1280xf32>
    %v1971 = stablehlo.reshape %v1914 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1972 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1973 = stablehlo.reduce(%v1971 init: %v1972) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1974 = stablehlo.constant dense<0.05> : tensor<1280xf32>
    %v1975 = stablehlo.multiply %v1973, %v1974 : tensor<1280xf32>
    %v1976 = stablehlo.subtract %hbt, %v1975 : tensor<1280xf32>
    %v1977 = stablehlo.reshape %v1843 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1978 = stablehlo.reshape %v1944 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1979 = stablehlo.transpose %v1977, dims = [1, 0, 2, 3] : (tensor<32x320x7x7xf32>) -> tensor<320x32x7x7xf32>
    %v1980 = stablehlo.transpose %v1978, dims = [1, 0, 2, 3] : (tensor<32x1280x7x7xf32>) -> tensor<1280x32x7x7xf32>
    %v1981 = stablehlo.convolution(%v1979, %v1980)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<320x32x7x7xf32>, tensor<1280x32x7x7xf32>) -> tensor<320x1280x1x1xf32>
    %v1982 = stablehlo.transpose %v1981, dims = [1, 0, 2, 3] : (tensor<320x1280x1x1xf32>) -> tensor<1280x320x1x1xf32>
    %v1983 = stablehlo.constant dense<0.05> : tensor<1280x320x1x1xf32>
    %v1984 = stablehlo.multiply %v1982, %v1983 : tensor<1280x320x1x1xf32>
    %v1985 = stablehlo.subtract %hW, %v1984 : tensor<1280x320x1x1xf32>
    %v1986 = stablehlo.reshape %v1823 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1987 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1988 = stablehlo.constant dense<1568.0> : tensor<32x320x7x7xf32>
    %v1989 = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %v1990 = stablehlo.reduce(%v1986 init: %v1987) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1991 = stablehlo.broadcast_in_dim %v1990, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1992 = stablehlo.divide %v1991, %v1988 : tensor<32x320x7x7xf32>
    %v1993 = stablehlo.subtract %v1986, %v1992 : tensor<32x320x7x7xf32>
    %v1994 = stablehlo.multiply %v1993, %v1993 : tensor<32x320x7x7xf32>
    %v1995 = stablehlo.reduce(%v1994 init: %v1987) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1996 = stablehlo.broadcast_in_dim %v1995, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1997 = stablehlo.divide %v1996, %v1988 : tensor<32x320x7x7xf32>
    %v1998 = stablehlo.add %v1997, %v1989 : tensor<32x320x7x7xf32>
    %v1999 = stablehlo.rsqrt %v1998 : tensor<32x320x7x7xf32>
    %v2000 = stablehlo.multiply %v1993, %v1999 : tensor<32x320x7x7xf32>
    %v2001 = stablehlo.broadcast_in_dim %b16pg, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v2002 = stablehlo.reshape %v1949 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v2003 = stablehlo.multiply %v2001, %v2002 : tensor<32x320x7x7xf32>
    %v2004 = stablehlo.reduce(%v2003 init: %v1987) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v2005 = stablehlo.broadcast_in_dim %v2004, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v2006 = stablehlo.multiply %v2000, %v2003 : tensor<32x320x7x7xf32>
    %v2007 = stablehlo.reduce(%v2006 init: %v1987) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v2008 = stablehlo.broadcast_in_dim %v2007, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v2009 = stablehlo.multiply %v2003, %v1988 : tensor<32x320x7x7xf32>
    %v2010 = stablehlo.subtract %v2009, %v2005 : tensor<32x320x7x7xf32>
    %v2011 = stablehlo.multiply %v2000, %v2008 : tensor<32x320x7x7xf32>
    %v2012 = stablehlo.subtract %v2010, %v2011 : tensor<32x320x7x7xf32>
    %v2013 = stablehlo.divide %v1999, %v1988 : tensor<32x320x7x7xf32>
    %v2014 = stablehlo.multiply %v2013, %v2012 : tensor<32x320x7x7xf32>
    %v2015 = stablehlo.reshape %v2014 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v2016 = stablehlo.reshape %v2015 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v2017 = stablehlo.reverse %b16pW, dims = [2, 3] : tensor<320x1152x1x1xf32>
    %v2018 = stablehlo.transpose %v2017, dims = [1, 0, 2, 3] : (tensor<320x1152x1x1xf32>) -> tensor<1152x320x1x1xf32>
    %v2019 = stablehlo.convolution(%v2016, %v2018)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x320x7x7xf32>, tensor<1152x320x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v2020 = stablehlo.reshape %v2019 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2021 = stablehlo.reshape %v1823 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v2022 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2023 = stablehlo.constant dense<1568.0> : tensor<32x320x7x7xf32>
    %v2024 = stablehlo.reduce(%v2021 init: %v2022) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v2025 = stablehlo.broadcast_in_dim %v2024, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v2026 = stablehlo.divide %v2025, %v2023 : tensor<32x320x7x7xf32>
    %v2027 = stablehlo.subtract %v2021, %v2026 : tensor<32x320x7x7xf32>
    %v2028 = stablehlo.multiply %v2027, %v2027 : tensor<32x320x7x7xf32>
    %v2029 = stablehlo.reduce(%v2028 init: %v2022) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v2030 = stablehlo.broadcast_in_dim %v2029, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v2031 = stablehlo.divide %v2030, %v2023 : tensor<32x320x7x7xf32>
    %v2032 = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %v2033 = stablehlo.add %v2031, %v2032 : tensor<32x320x7x7xf32>
    %v2034 = stablehlo.rsqrt %v2033 : tensor<32x320x7x7xf32>
    %v2035 = stablehlo.multiply %v2027, %v2034 : tensor<32x320x7x7xf32>
    %v2036 = stablehlo.reshape %v1949 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v2037 = stablehlo.multiply %v2036, %v2035 : tensor<32x320x7x7xf32>
    %v2038 = stablehlo.reduce(%v2037 init: %v2022) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v2039 = stablehlo.constant dense<0.05> : tensor<320xf32>
    %v2040 = stablehlo.multiply %v2038, %v2039 : tensor<320xf32>
    %v2041 = stablehlo.subtract %b16pg, %v2040 : tensor<320xf32>
    %v2042 = stablehlo.reshape %v1949 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v2043 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2044 = stablehlo.reduce(%v2042 init: %v2043) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v2045 = stablehlo.constant dense<0.05> : tensor<320xf32>
    %v2046 = stablehlo.multiply %v2044, %v2045 : tensor<320xf32>
    %v2047 = stablehlo.subtract %b16pbt, %v2046 : tensor<320xf32>
    %v2048 = stablehlo.reshape %v1818 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2049 = stablehlo.reshape %v2015 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v2050 = stablehlo.transpose %v2048, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2051 = stablehlo.transpose %v2049, dims = [1, 0, 2, 3] : (tensor<32x320x7x7xf32>) -> tensor<320x32x7x7xf32>
    %v2052 = stablehlo.convolution(%v2050, %v2051)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<320x32x7x7xf32>) -> tensor<1152x320x1x1xf32>
    %v2053 = stablehlo.transpose %v2052, dims = [1, 0, 2, 3] : (tensor<1152x320x1x1xf32>) -> tensor<320x1152x1x1xf32>
    %v2054 = stablehlo.constant dense<0.05> : tensor<320x1152x1x1xf32>
    %v2055 = stablehlo.multiply %v2053, %v2054 : tensor<320x1152x1x1xf32>
    %v2056 = stablehlo.subtract %b16pW, %v2055 : tensor<320x1152x1x1xf32>
    %v2057 = stablehlo.reshape %v1788 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2058 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2059 = stablehlo.reduce(%v2057 init: %v2058) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2060 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v2061 = stablehlo.divide %v2059, %v2060 : tensor<32x1152xf32>
    %v2062 = stablehlo.dot_general %v2061, %b16zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v2063 = stablehlo.broadcast_in_dim %b16zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v2064 = stablehlo.add %v2062, %v2063 : tensor<32x48xf32>
    %v2065 = stablehlo.logistic %v2064 : tensor<32x48xf32>
    %v2066 = stablehlo.multiply %v2064, %v2065 : tensor<32x48xf32>
    %v2067 = stablehlo.dot_general %v2066, %b16zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v2068 = stablehlo.broadcast_in_dim %b16zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v2069 = stablehlo.add %v2067, %v2068 : tensor<32x1152xf32>
    %v2070 = stablehlo.logistic %v2069 : tensor<32x1152xf32>
    %v2071 = stablehlo.reshape %v2020 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2072 = stablehlo.broadcast_in_dim %v2070, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2073 = stablehlo.multiply %v2072, %v2071 : tensor<32x1152x7x7xf32>
    %v2074 = stablehlo.multiply %v2057, %v2071 : tensor<32x1152x7x7xf32>
    %v2075 = stablehlo.reduce(%v2074 init: %v2058) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2076 = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %v2077 = stablehlo.subtract %v2076, %v2070 : tensor<32x1152xf32>
    %v2078 = stablehlo.multiply %v2070, %v2077 : tensor<32x1152xf32>
    %v2079 = stablehlo.multiply %v2075, %v2078 : tensor<32x1152xf32>
    %v2080 = stablehlo.dot_general %v2079, %b16zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<48x1152xf32>) -> tensor<32x48xf32>
    %v2081 = stablehlo.logistic %v2064 : tensor<32x48xf32>
    %v2082 = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %v2083 = stablehlo.subtract %v2082, %v2081 : tensor<32x48xf32>
    %v2084 = stablehlo.multiply %v2064, %v2083 : tensor<32x48xf32>
    %v2085 = stablehlo.add %v2082, %v2084 : tensor<32x48xf32>
    %v2086 = stablehlo.multiply %v2081, %v2085 : tensor<32x48xf32>
    %v2087 = stablehlo.multiply %v2080, %v2086 : tensor<32x48xf32>
    %v2088 = stablehlo.dot_general %v2087, %b16zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<1152x48xf32>) -> tensor<32x1152xf32>
    %v2089 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v2090 = stablehlo.divide %v2088, %v2089 : tensor<32x1152xf32>
    %v2091 = stablehlo.broadcast_in_dim %v2090, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2092 = stablehlo.add %v2073, %v2091 : tensor<32x1152x7x7xf32>
    %v2093 = stablehlo.reshape %v2092 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2094 = stablehlo.reshape %v1788 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2095 = stablehlo.reshape %v2020 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2096 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2097 = stablehlo.multiply %v2094, %v2095 : tensor<32x1152x7x7xf32>
    %v2098 = stablehlo.reduce(%v2097 init: %v2096) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2099 = stablehlo.logistic %v1801 : tensor<32x1152xf32>
    %v2100 = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %v2101 = stablehlo.subtract %v2100, %v2099 : tensor<32x1152xf32>
    %v2102 = stablehlo.multiply %v2099, %v2101 : tensor<32x1152xf32>
    %v2103 = stablehlo.multiply %v2098, %v2102 : tensor<32x1152xf32>
    %v2104 = stablehlo.dot_general %v1798, %v2103, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<32x1152xf32>) -> tensor<48x1152xf32>
    %v2105 = stablehlo.constant dense<0.05> : tensor<48x1152xf32>
    %v2106 = stablehlo.multiply %v2104, %v2105 : tensor<48x1152xf32>
    %v2107 = stablehlo.subtract %b16zW2, %v2106 : tensor<48x1152xf32>
    %v2108 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2109 = stablehlo.reduce(%v2103 init: %v2108) applies stablehlo.add across dimensions = [0] : (tensor<32x1152xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2110 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2111 = stablehlo.multiply %v2109, %v2110 : tensor<1152xf32>
    %v2112 = stablehlo.subtract %b16zb2, %v2111 : tensor<1152xf32>
    %v2113 = stablehlo.reshape %v2103 : (tensor<32x1152xf32>) -> tensor<32x1x1152xf32>
    %v2114 = stablehlo.dot_general %v2113, %b16zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x1152xf32>, tensor<48x1152xf32>) -> tensor<32x1x48xf32>
    %v2115 = stablehlo.reshape %v2114 : (tensor<32x1x48xf32>) -> tensor<32x48xf32>
    %v2116 = stablehlo.logistic %v1796 : tensor<32x48xf32>
    %v2117 = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %v2118 = stablehlo.subtract %v2117, %v2116 : tensor<32x48xf32>
    %v2119 = stablehlo.multiply %v1796, %v2118 : tensor<32x48xf32>
    %v2120 = stablehlo.add %v2117, %v2119 : tensor<32x48xf32>
    %v2121 = stablehlo.multiply %v2116, %v2120 : tensor<32x48xf32>
    %v2122 = stablehlo.multiply %v2115, %v2121 : tensor<32x48xf32>
    %v2123 = stablehlo.dot_general %v1793, %v2122, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<32x48xf32>) -> tensor<1152x48xf32>
    %v2124 = stablehlo.constant dense<0.05> : tensor<1152x48xf32>
    %v2125 = stablehlo.multiply %v2123, %v2124 : tensor<1152x48xf32>
    %v2126 = stablehlo.subtract %b16zW1, %v2125 : tensor<1152x48xf32>
    %v2127 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2128 = stablehlo.reduce(%v2122 init: %v2127) applies stablehlo.add across dimensions = [0] : (tensor<32x48xf32>, tensor<f32>) -> tensor<48xf32>
    %v2129 = stablehlo.constant dense<0.05> : tensor<48xf32>
    %v2130 = stablehlo.multiply %v2128, %v2129 : tensor<48xf32>
    %v2131 = stablehlo.subtract %b16zb1, %v2130 : tensor<48xf32>
    %v2132 = stablehlo.reshape %v2093 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2133 = stablehlo.reshape %v1784 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2134 = stablehlo.logistic %v2133 : tensor<32x1152x7x7xf32>
    %v2135 = stablehlo.constant dense<1.0> : tensor<32x1152x7x7xf32>
    %v2136 = stablehlo.subtract %v2135, %v2134 : tensor<32x1152x7x7xf32>
    %v2137 = stablehlo.multiply %v2133, %v2136 : tensor<32x1152x7x7xf32>
    %v2138 = stablehlo.add %v2135, %v2137 : tensor<32x1152x7x7xf32>
    %v2139 = stablehlo.multiply %v2134, %v2138 : tensor<32x1152x7x7xf32>
    %v2140 = stablehlo.multiply %v2132, %v2139 : tensor<32x1152x7x7xf32>
    %v2141 = stablehlo.reshape %v2140 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2142 = stablehlo.reshape %v1764 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2143 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2144 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2145 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2146 = stablehlo.reduce(%v2142 init: %v2143) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2147 = stablehlo.broadcast_in_dim %v2146, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2148 = stablehlo.divide %v2147, %v2144 : tensor<32x1152x7x7xf32>
    %v2149 = stablehlo.subtract %v2142, %v2148 : tensor<32x1152x7x7xf32>
    %v2150 = stablehlo.multiply %v2149, %v2149 : tensor<32x1152x7x7xf32>
    %v2151 = stablehlo.reduce(%v2150 init: %v2143) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2152 = stablehlo.broadcast_in_dim %v2151, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2153 = stablehlo.divide %v2152, %v2144 : tensor<32x1152x7x7xf32>
    %v2154 = stablehlo.add %v2153, %v2145 : tensor<32x1152x7x7xf32>
    %v2155 = stablehlo.rsqrt %v2154 : tensor<32x1152x7x7xf32>
    %v2156 = stablehlo.multiply %v2149, %v2155 : tensor<32x1152x7x7xf32>
    %v2157 = stablehlo.broadcast_in_dim %b16dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2158 = stablehlo.reshape %v2141 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2159 = stablehlo.multiply %v2157, %v2158 : tensor<32x1152x7x7xf32>
    %v2160 = stablehlo.reduce(%v2159 init: %v2143) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2161 = stablehlo.broadcast_in_dim %v2160, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2162 = stablehlo.multiply %v2156, %v2159 : tensor<32x1152x7x7xf32>
    %v2163 = stablehlo.reduce(%v2162 init: %v2143) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2164 = stablehlo.broadcast_in_dim %v2163, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2165 = stablehlo.multiply %v2159, %v2144 : tensor<32x1152x7x7xf32>
    %v2166 = stablehlo.subtract %v2165, %v2161 : tensor<32x1152x7x7xf32>
    %v2167 = stablehlo.multiply %v2156, %v2164 : tensor<32x1152x7x7xf32>
    %v2168 = stablehlo.subtract %v2166, %v2167 : tensor<32x1152x7x7xf32>
    %v2169 = stablehlo.divide %v2155, %v2144 : tensor<32x1152x7x7xf32>
    %v2170 = stablehlo.multiply %v2169, %v2168 : tensor<32x1152x7x7xf32>
    %v2171 = stablehlo.reshape %v2170 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2172 = stablehlo.reshape %v2171 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2173 = stablehlo.reverse %b16dW, dims = [2, 3] : tensor<1152x1x3x3xf32>
    %v2174 = stablehlo.convolution(%v2172, %v2173)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x3x3xf32>) -> tensor<32x1152x7x7xf32>
    %v2175 = stablehlo.reshape %v2174 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2176 = stablehlo.reshape %v1764 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2177 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2178 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2179 = stablehlo.reduce(%v2176 init: %v2177) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2180 = stablehlo.broadcast_in_dim %v2179, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2181 = stablehlo.divide %v2180, %v2178 : tensor<32x1152x7x7xf32>
    %v2182 = stablehlo.subtract %v2176, %v2181 : tensor<32x1152x7x7xf32>
    %v2183 = stablehlo.multiply %v2182, %v2182 : tensor<32x1152x7x7xf32>
    %v2184 = stablehlo.reduce(%v2183 init: %v2177) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2185 = stablehlo.broadcast_in_dim %v2184, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2186 = stablehlo.divide %v2185, %v2178 : tensor<32x1152x7x7xf32>
    %v2187 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2188 = stablehlo.add %v2186, %v2187 : tensor<32x1152x7x7xf32>
    %v2189 = stablehlo.rsqrt %v2188 : tensor<32x1152x7x7xf32>
    %v2190 = stablehlo.multiply %v2182, %v2189 : tensor<32x1152x7x7xf32>
    %v2191 = stablehlo.reshape %v2141 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2192 = stablehlo.multiply %v2191, %v2190 : tensor<32x1152x7x7xf32>
    %v2193 = stablehlo.reduce(%v2192 init: %v2177) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2194 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2195 = stablehlo.multiply %v2193, %v2194 : tensor<1152xf32>
    %v2196 = stablehlo.subtract %b16dg, %v2195 : tensor<1152xf32>
    %v2197 = stablehlo.reshape %v2141 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2198 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2199 = stablehlo.reduce(%v2197 init: %v2198) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2200 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2201 = stablehlo.multiply %v2199, %v2200 : tensor<1152xf32>
    %v2202 = stablehlo.subtract %b16dbt, %v2201 : tensor<1152xf32>
    %v2203 = stablehlo.reshape %v1759 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2204 = stablehlo.reshape %v2171 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2205 = stablehlo.transpose %v2203, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2206 = stablehlo.transpose %v2204, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2207 = stablehlo.convolution(%v2205, %v2206)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1152 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<1x1152x3x3xf32>
    %v2208 = stablehlo.reshape %v2207 : (tensor<1x1152x3x3xf32>) -> tensor<1152x1x3x3xf32>
    %v2209 = stablehlo.constant dense<0.05> : tensor<1152x1x3x3xf32>
    %v2210 = stablehlo.multiply %v2208, %v2209 : tensor<1152x1x3x3xf32>
    %v2211 = stablehlo.subtract %b16dW, %v2210 : tensor<1152x1x3x3xf32>
    %v2212 = stablehlo.reshape %v2175 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2213 = stablehlo.reshape %v1755 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2214 = stablehlo.logistic %v2213 : tensor<32x1152x7x7xf32>
    %v2215 = stablehlo.constant dense<1.0> : tensor<32x1152x7x7xf32>
    %v2216 = stablehlo.subtract %v2215, %v2214 : tensor<32x1152x7x7xf32>
    %v2217 = stablehlo.multiply %v2213, %v2216 : tensor<32x1152x7x7xf32>
    %v2218 = stablehlo.add %v2215, %v2217 : tensor<32x1152x7x7xf32>
    %v2219 = stablehlo.multiply %v2214, %v2218 : tensor<32x1152x7x7xf32>
    %v2220 = stablehlo.multiply %v2212, %v2219 : tensor<32x1152x7x7xf32>
    %v2221 = stablehlo.reshape %v2220 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2222 = stablehlo.reshape %v1735 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2223 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2224 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2225 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2226 = stablehlo.reduce(%v2222 init: %v2223) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2227 = stablehlo.broadcast_in_dim %v2226, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2228 = stablehlo.divide %v2227, %v2224 : tensor<32x1152x7x7xf32>
    %v2229 = stablehlo.subtract %v2222, %v2228 : tensor<32x1152x7x7xf32>
    %v2230 = stablehlo.multiply %v2229, %v2229 : tensor<32x1152x7x7xf32>
    %v2231 = stablehlo.reduce(%v2230 init: %v2223) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2232 = stablehlo.broadcast_in_dim %v2231, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2233 = stablehlo.divide %v2232, %v2224 : tensor<32x1152x7x7xf32>
    %v2234 = stablehlo.add %v2233, %v2225 : tensor<32x1152x7x7xf32>
    %v2235 = stablehlo.rsqrt %v2234 : tensor<32x1152x7x7xf32>
    %v2236 = stablehlo.multiply %v2229, %v2235 : tensor<32x1152x7x7xf32>
    %v2237 = stablehlo.broadcast_in_dim %b16eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2238 = stablehlo.reshape %v2221 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2239 = stablehlo.multiply %v2237, %v2238 : tensor<32x1152x7x7xf32>
    %v2240 = stablehlo.reduce(%v2239 init: %v2223) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2241 = stablehlo.broadcast_in_dim %v2240, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2242 = stablehlo.multiply %v2236, %v2239 : tensor<32x1152x7x7xf32>
    %v2243 = stablehlo.reduce(%v2242 init: %v2223) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2244 = stablehlo.broadcast_in_dim %v2243, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2245 = stablehlo.multiply %v2239, %v2224 : tensor<32x1152x7x7xf32>
    %v2246 = stablehlo.subtract %v2245, %v2241 : tensor<32x1152x7x7xf32>
    %v2247 = stablehlo.multiply %v2236, %v2244 : tensor<32x1152x7x7xf32>
    %v2248 = stablehlo.subtract %v2246, %v2247 : tensor<32x1152x7x7xf32>
    %v2249 = stablehlo.divide %v2235, %v2224 : tensor<32x1152x7x7xf32>
    %v2250 = stablehlo.multiply %v2249, %v2248 : tensor<32x1152x7x7xf32>
    %v2251 = stablehlo.reshape %v2250 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2252 = stablehlo.reshape %v2251 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2253 = stablehlo.reverse %b16eW, dims = [2, 3] : tensor<1152x192x1x1xf32>
    %v2254 = stablehlo.transpose %v2253, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %v2255 = stablehlo.convolution(%v2252, %v2254)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v2256 = stablehlo.reshape %v2255 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v2257 = stablehlo.reshape %v1735 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2258 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2259 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2260 = stablehlo.reduce(%v2257 init: %v2258) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2261 = stablehlo.broadcast_in_dim %v2260, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2262 = stablehlo.divide %v2261, %v2259 : tensor<32x1152x7x7xf32>
    %v2263 = stablehlo.subtract %v2257, %v2262 : tensor<32x1152x7x7xf32>
    %v2264 = stablehlo.multiply %v2263, %v2263 : tensor<32x1152x7x7xf32>
    %v2265 = stablehlo.reduce(%v2264 init: %v2258) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2266 = stablehlo.broadcast_in_dim %v2265, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2267 = stablehlo.divide %v2266, %v2259 : tensor<32x1152x7x7xf32>
    %v2268 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2269 = stablehlo.add %v2267, %v2268 : tensor<32x1152x7x7xf32>
    %v2270 = stablehlo.rsqrt %v2269 : tensor<32x1152x7x7xf32>
    %v2271 = stablehlo.multiply %v2263, %v2270 : tensor<32x1152x7x7xf32>
    %v2272 = stablehlo.reshape %v2221 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2273 = stablehlo.multiply %v2272, %v2271 : tensor<32x1152x7x7xf32>
    %v2274 = stablehlo.reduce(%v2273 init: %v2258) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2275 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2276 = stablehlo.multiply %v2274, %v2275 : tensor<1152xf32>
    %v2277 = stablehlo.subtract %b16eg, %v2276 : tensor<1152xf32>
    %v2278 = stablehlo.reshape %v2221 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2279 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2280 = stablehlo.reduce(%v2278 init: %v2279) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2281 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2282 = stablehlo.multiply %v2280, %v2281 : tensor<1152xf32>
    %v2283 = stablehlo.subtract %b16ebt, %v2282 : tensor<1152xf32>
    %v2284 = stablehlo.reshape %v1730 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2285 = stablehlo.reshape %v2251 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2286 = stablehlo.transpose %v2284, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %v2287 = stablehlo.transpose %v2285, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2288 = stablehlo.convolution(%v2286, %v2287)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<192x1152x1x1xf32>
    %v2289 = stablehlo.transpose %v2288, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %v2290 = stablehlo.constant dense<0.05> : tensor<1152x192x1x1xf32>
    %v2291 = stablehlo.multiply %v2289, %v2290 : tensor<1152x192x1x1xf32>
    %v2292 = stablehlo.subtract %b16eW, %v2291 : tensor<1152x192x1x1xf32>
    %v2293 = stablehlo.reshape %v1706 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2294 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2295 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v2296 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v2297 = stablehlo.reduce(%v2293 init: %v2294) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2298 = stablehlo.broadcast_in_dim %v2297, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2299 = stablehlo.divide %v2298, %v2295 : tensor<32x192x7x7xf32>
    %v2300 = stablehlo.subtract %v2293, %v2299 : tensor<32x192x7x7xf32>
    %v2301 = stablehlo.multiply %v2300, %v2300 : tensor<32x192x7x7xf32>
    %v2302 = stablehlo.reduce(%v2301 init: %v2294) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2303 = stablehlo.broadcast_in_dim %v2302, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2304 = stablehlo.divide %v2303, %v2295 : tensor<32x192x7x7xf32>
    %v2305 = stablehlo.add %v2304, %v2296 : tensor<32x192x7x7xf32>
    %v2306 = stablehlo.rsqrt %v2305 : tensor<32x192x7x7xf32>
    %v2307 = stablehlo.multiply %v2300, %v2306 : tensor<32x192x7x7xf32>
    %v2308 = stablehlo.broadcast_in_dim %b15pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2309 = stablehlo.reshape %v2256 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2310 = stablehlo.multiply %v2308, %v2309 : tensor<32x192x7x7xf32>
    %v2311 = stablehlo.reduce(%v2310 init: %v2294) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2312 = stablehlo.broadcast_in_dim %v2311, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2313 = stablehlo.multiply %v2307, %v2310 : tensor<32x192x7x7xf32>
    %v2314 = stablehlo.reduce(%v2313 init: %v2294) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2315 = stablehlo.broadcast_in_dim %v2314, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2316 = stablehlo.multiply %v2310, %v2295 : tensor<32x192x7x7xf32>
    %v2317 = stablehlo.subtract %v2316, %v2312 : tensor<32x192x7x7xf32>
    %v2318 = stablehlo.multiply %v2307, %v2315 : tensor<32x192x7x7xf32>
    %v2319 = stablehlo.subtract %v2317, %v2318 : tensor<32x192x7x7xf32>
    %v2320 = stablehlo.divide %v2306, %v2295 : tensor<32x192x7x7xf32>
    %v2321 = stablehlo.multiply %v2320, %v2319 : tensor<32x192x7x7xf32>
    %v2322 = stablehlo.reshape %v2321 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v2323 = stablehlo.reshape %v2322 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2324 = stablehlo.reverse %b15pW, dims = [2, 3] : tensor<192x1152x1x1xf32>
    %v2325 = stablehlo.transpose %v2324, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %v2326 = stablehlo.convolution(%v2323, %v2325)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v2327 = stablehlo.reshape %v2326 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2328 = stablehlo.reshape %v1706 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2329 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2330 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v2331 = stablehlo.reduce(%v2328 init: %v2329) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2332 = stablehlo.broadcast_in_dim %v2331, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2333 = stablehlo.divide %v2332, %v2330 : tensor<32x192x7x7xf32>
    %v2334 = stablehlo.subtract %v2328, %v2333 : tensor<32x192x7x7xf32>
    %v2335 = stablehlo.multiply %v2334, %v2334 : tensor<32x192x7x7xf32>
    %v2336 = stablehlo.reduce(%v2335 init: %v2329) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2337 = stablehlo.broadcast_in_dim %v2336, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2338 = stablehlo.divide %v2337, %v2330 : tensor<32x192x7x7xf32>
    %v2339 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v2340 = stablehlo.add %v2338, %v2339 : tensor<32x192x7x7xf32>
    %v2341 = stablehlo.rsqrt %v2340 : tensor<32x192x7x7xf32>
    %v2342 = stablehlo.multiply %v2334, %v2341 : tensor<32x192x7x7xf32>
    %v2343 = stablehlo.reshape %v2256 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2344 = stablehlo.multiply %v2343, %v2342 : tensor<32x192x7x7xf32>
    %v2345 = stablehlo.reduce(%v2344 init: %v2329) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2346 = stablehlo.constant dense<0.05> : tensor<192xf32>
    %v2347 = stablehlo.multiply %v2345, %v2346 : tensor<192xf32>
    %v2348 = stablehlo.subtract %b15pg, %v2347 : tensor<192xf32>
    %v2349 = stablehlo.reshape %v2256 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2350 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2351 = stablehlo.reduce(%v2349 init: %v2350) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2352 = stablehlo.constant dense<0.05> : tensor<192xf32>
    %v2353 = stablehlo.multiply %v2351, %v2352 : tensor<192xf32>
    %v2354 = stablehlo.subtract %b15pbt, %v2353 : tensor<192xf32>
    %v2355 = stablehlo.reshape %v1701 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2356 = stablehlo.reshape %v2322 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2357 = stablehlo.transpose %v2355, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2358 = stablehlo.transpose %v2356, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %v2359 = stablehlo.convolution(%v2357, %v2358)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<192x32x7x7xf32>) -> tensor<1152x192x1x1xf32>
    %v2360 = stablehlo.transpose %v2359, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %v2361 = stablehlo.constant dense<0.05> : tensor<192x1152x1x1xf32>
    %v2362 = stablehlo.multiply %v2360, %v2361 : tensor<192x1152x1x1xf32>
    %v2363 = stablehlo.subtract %b15pW, %v2362 : tensor<192x1152x1x1xf32>
    %v2364 = stablehlo.reshape %v1671 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2365 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2366 = stablehlo.reduce(%v2364 init: %v2365) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2367 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v2368 = stablehlo.divide %v2366, %v2367 : tensor<32x1152xf32>
    %v2369 = stablehlo.dot_general %v2368, %b15zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v2370 = stablehlo.broadcast_in_dim %b15zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v2371 = stablehlo.add %v2369, %v2370 : tensor<32x48xf32>
    %v2372 = stablehlo.logistic %v2371 : tensor<32x48xf32>
    %v2373 = stablehlo.multiply %v2371, %v2372 : tensor<32x48xf32>
    %v2374 = stablehlo.dot_general %v2373, %b15zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v2375 = stablehlo.broadcast_in_dim %b15zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v2376 = stablehlo.add %v2374, %v2375 : tensor<32x1152xf32>
    %v2377 = stablehlo.logistic %v2376 : tensor<32x1152xf32>
    %v2378 = stablehlo.reshape %v2327 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2379 = stablehlo.broadcast_in_dim %v2377, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2380 = stablehlo.multiply %v2379, %v2378 : tensor<32x1152x7x7xf32>
    %v2381 = stablehlo.multiply %v2364, %v2378 : tensor<32x1152x7x7xf32>
    %v2382 = stablehlo.reduce(%v2381 init: %v2365) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2383 = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %v2384 = stablehlo.subtract %v2383, %v2377 : tensor<32x1152xf32>
    %v2385 = stablehlo.multiply %v2377, %v2384 : tensor<32x1152xf32>
    %v2386 = stablehlo.multiply %v2382, %v2385 : tensor<32x1152xf32>
    %v2387 = stablehlo.dot_general %v2386, %b15zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<48x1152xf32>) -> tensor<32x48xf32>
    %v2388 = stablehlo.logistic %v2371 : tensor<32x48xf32>
    %v2389 = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %v2390 = stablehlo.subtract %v2389, %v2388 : tensor<32x48xf32>
    %v2391 = stablehlo.multiply %v2371, %v2390 : tensor<32x48xf32>
    %v2392 = stablehlo.add %v2389, %v2391 : tensor<32x48xf32>
    %v2393 = stablehlo.multiply %v2388, %v2392 : tensor<32x48xf32>
    %v2394 = stablehlo.multiply %v2387, %v2393 : tensor<32x48xf32>
    %v2395 = stablehlo.dot_general %v2394, %b15zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<1152x48xf32>) -> tensor<32x1152xf32>
    %v2396 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v2397 = stablehlo.divide %v2395, %v2396 : tensor<32x1152xf32>
    %v2398 = stablehlo.broadcast_in_dim %v2397, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2399 = stablehlo.add %v2380, %v2398 : tensor<32x1152x7x7xf32>
    %v2400 = stablehlo.reshape %v2399 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2401 = stablehlo.reshape %v1671 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2402 = stablehlo.reshape %v2327 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2403 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2404 = stablehlo.multiply %v2401, %v2402 : tensor<32x1152x7x7xf32>
    %v2405 = stablehlo.reduce(%v2404 init: %v2403) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2406 = stablehlo.logistic %v1684 : tensor<32x1152xf32>
    %v2407 = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %v2408 = stablehlo.subtract %v2407, %v2406 : tensor<32x1152xf32>
    %v2409 = stablehlo.multiply %v2406, %v2408 : tensor<32x1152xf32>
    %v2410 = stablehlo.multiply %v2405, %v2409 : tensor<32x1152xf32>
    %v2411 = stablehlo.dot_general %v1681, %v2410, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<32x1152xf32>) -> tensor<48x1152xf32>
    %v2412 = stablehlo.constant dense<0.05> : tensor<48x1152xf32>
    %v2413 = stablehlo.multiply %v2411, %v2412 : tensor<48x1152xf32>
    %v2414 = stablehlo.subtract %b15zW2, %v2413 : tensor<48x1152xf32>
    %v2415 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2416 = stablehlo.reduce(%v2410 init: %v2415) applies stablehlo.add across dimensions = [0] : (tensor<32x1152xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2417 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2418 = stablehlo.multiply %v2416, %v2417 : tensor<1152xf32>
    %v2419 = stablehlo.subtract %b15zb2, %v2418 : tensor<1152xf32>
    %v2420 = stablehlo.reshape %v2410 : (tensor<32x1152xf32>) -> tensor<32x1x1152xf32>
    %v2421 = stablehlo.dot_general %v2420, %b15zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x1152xf32>, tensor<48x1152xf32>) -> tensor<32x1x48xf32>
    %v2422 = stablehlo.reshape %v2421 : (tensor<32x1x48xf32>) -> tensor<32x48xf32>
    %v2423 = stablehlo.logistic %v1679 : tensor<32x48xf32>
    %v2424 = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %v2425 = stablehlo.subtract %v2424, %v2423 : tensor<32x48xf32>
    %v2426 = stablehlo.multiply %v1679, %v2425 : tensor<32x48xf32>
    %v2427 = stablehlo.add %v2424, %v2426 : tensor<32x48xf32>
    %v2428 = stablehlo.multiply %v2423, %v2427 : tensor<32x48xf32>
    %v2429 = stablehlo.multiply %v2422, %v2428 : tensor<32x48xf32>
    %v2430 = stablehlo.dot_general %v1676, %v2429, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<32x48xf32>) -> tensor<1152x48xf32>
    %v2431 = stablehlo.constant dense<0.05> : tensor<1152x48xf32>
    %v2432 = stablehlo.multiply %v2430, %v2431 : tensor<1152x48xf32>
    %v2433 = stablehlo.subtract %b15zW1, %v2432 : tensor<1152x48xf32>
    %v2434 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2435 = stablehlo.reduce(%v2429 init: %v2434) applies stablehlo.add across dimensions = [0] : (tensor<32x48xf32>, tensor<f32>) -> tensor<48xf32>
    %v2436 = stablehlo.constant dense<0.05> : tensor<48xf32>
    %v2437 = stablehlo.multiply %v2435, %v2436 : tensor<48xf32>
    %v2438 = stablehlo.subtract %b15zb1, %v2437 : tensor<48xf32>
    %v2439 = stablehlo.reshape %v2400 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2440 = stablehlo.reshape %v1667 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2441 = stablehlo.logistic %v2440 : tensor<32x1152x7x7xf32>
    %v2442 = stablehlo.constant dense<1.0> : tensor<32x1152x7x7xf32>
    %v2443 = stablehlo.subtract %v2442, %v2441 : tensor<32x1152x7x7xf32>
    %v2444 = stablehlo.multiply %v2440, %v2443 : tensor<32x1152x7x7xf32>
    %v2445 = stablehlo.add %v2442, %v2444 : tensor<32x1152x7x7xf32>
    %v2446 = stablehlo.multiply %v2441, %v2445 : tensor<32x1152x7x7xf32>
    %v2447 = stablehlo.multiply %v2439, %v2446 : tensor<32x1152x7x7xf32>
    %v2448 = stablehlo.reshape %v2447 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2449 = stablehlo.reshape %v1647 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2450 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2451 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2452 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2453 = stablehlo.reduce(%v2449 init: %v2450) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2454 = stablehlo.broadcast_in_dim %v2453, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2455 = stablehlo.divide %v2454, %v2451 : tensor<32x1152x7x7xf32>
    %v2456 = stablehlo.subtract %v2449, %v2455 : tensor<32x1152x7x7xf32>
    %v2457 = stablehlo.multiply %v2456, %v2456 : tensor<32x1152x7x7xf32>
    %v2458 = stablehlo.reduce(%v2457 init: %v2450) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2459 = stablehlo.broadcast_in_dim %v2458, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2460 = stablehlo.divide %v2459, %v2451 : tensor<32x1152x7x7xf32>
    %v2461 = stablehlo.add %v2460, %v2452 : tensor<32x1152x7x7xf32>
    %v2462 = stablehlo.rsqrt %v2461 : tensor<32x1152x7x7xf32>
    %v2463 = stablehlo.multiply %v2456, %v2462 : tensor<32x1152x7x7xf32>
    %v2464 = stablehlo.broadcast_in_dim %b15dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2465 = stablehlo.reshape %v2448 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2466 = stablehlo.multiply %v2464, %v2465 : tensor<32x1152x7x7xf32>
    %v2467 = stablehlo.reduce(%v2466 init: %v2450) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2468 = stablehlo.broadcast_in_dim %v2467, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2469 = stablehlo.multiply %v2463, %v2466 : tensor<32x1152x7x7xf32>
    %v2470 = stablehlo.reduce(%v2469 init: %v2450) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2471 = stablehlo.broadcast_in_dim %v2470, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2472 = stablehlo.multiply %v2466, %v2451 : tensor<32x1152x7x7xf32>
    %v2473 = stablehlo.subtract %v2472, %v2468 : tensor<32x1152x7x7xf32>
    %v2474 = stablehlo.multiply %v2463, %v2471 : tensor<32x1152x7x7xf32>
    %v2475 = stablehlo.subtract %v2473, %v2474 : tensor<32x1152x7x7xf32>
    %v2476 = stablehlo.divide %v2462, %v2451 : tensor<32x1152x7x7xf32>
    %v2477 = stablehlo.multiply %v2476, %v2475 : tensor<32x1152x7x7xf32>
    %v2478 = stablehlo.reshape %v2477 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2479 = stablehlo.reshape %v2478 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2480 = stablehlo.reverse %b15dW, dims = [2, 3] : tensor<1152x1x5x5xf32>
    %v2481 = stablehlo.convolution(%v2479, %v2480)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v2482 = stablehlo.reshape %v2481 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2483 = stablehlo.reshape %v1647 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2484 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2485 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2486 = stablehlo.reduce(%v2483 init: %v2484) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2487 = stablehlo.broadcast_in_dim %v2486, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2488 = stablehlo.divide %v2487, %v2485 : tensor<32x1152x7x7xf32>
    %v2489 = stablehlo.subtract %v2483, %v2488 : tensor<32x1152x7x7xf32>
    %v2490 = stablehlo.multiply %v2489, %v2489 : tensor<32x1152x7x7xf32>
    %v2491 = stablehlo.reduce(%v2490 init: %v2484) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2492 = stablehlo.broadcast_in_dim %v2491, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2493 = stablehlo.divide %v2492, %v2485 : tensor<32x1152x7x7xf32>
    %v2494 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2495 = stablehlo.add %v2493, %v2494 : tensor<32x1152x7x7xf32>
    %v2496 = stablehlo.rsqrt %v2495 : tensor<32x1152x7x7xf32>
    %v2497 = stablehlo.multiply %v2489, %v2496 : tensor<32x1152x7x7xf32>
    %v2498 = stablehlo.reshape %v2448 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2499 = stablehlo.multiply %v2498, %v2497 : tensor<32x1152x7x7xf32>
    %v2500 = stablehlo.reduce(%v2499 init: %v2484) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2501 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2502 = stablehlo.multiply %v2500, %v2501 : tensor<1152xf32>
    %v2503 = stablehlo.subtract %b15dg, %v2502 : tensor<1152xf32>
    %v2504 = stablehlo.reshape %v2448 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2505 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2506 = stablehlo.reduce(%v2504 init: %v2505) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2507 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2508 = stablehlo.multiply %v2506, %v2507 : tensor<1152xf32>
    %v2509 = stablehlo.subtract %b15dbt, %v2508 : tensor<1152xf32>
    %v2510 = stablehlo.reshape %v1642 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2511 = stablehlo.reshape %v2478 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2512 = stablehlo.transpose %v2510, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2513 = stablehlo.transpose %v2511, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2514 = stablehlo.convolution(%v2512, %v2513)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1152 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<1x1152x5x5xf32>
    %v2515 = stablehlo.reshape %v2514 : (tensor<1x1152x5x5xf32>) -> tensor<1152x1x5x5xf32>
    %v2516 = stablehlo.constant dense<0.05> : tensor<1152x1x5x5xf32>
    %v2517 = stablehlo.multiply %v2515, %v2516 : tensor<1152x1x5x5xf32>
    %v2518 = stablehlo.subtract %b15dW, %v2517 : tensor<1152x1x5x5xf32>
    %v2519 = stablehlo.reshape %v2482 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2520 = stablehlo.reshape %v1638 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2521 = stablehlo.logistic %v2520 : tensor<32x1152x7x7xf32>
    %v2522 = stablehlo.constant dense<1.0> : tensor<32x1152x7x7xf32>
    %v2523 = stablehlo.subtract %v2522, %v2521 : tensor<32x1152x7x7xf32>
    %v2524 = stablehlo.multiply %v2520, %v2523 : tensor<32x1152x7x7xf32>
    %v2525 = stablehlo.add %v2522, %v2524 : tensor<32x1152x7x7xf32>
    %v2526 = stablehlo.multiply %v2521, %v2525 : tensor<32x1152x7x7xf32>
    %v2527 = stablehlo.multiply %v2519, %v2526 : tensor<32x1152x7x7xf32>
    %v2528 = stablehlo.reshape %v2527 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2529 = stablehlo.reshape %v1618 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2530 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2531 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2532 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2533 = stablehlo.reduce(%v2529 init: %v2530) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2534 = stablehlo.broadcast_in_dim %v2533, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2535 = stablehlo.divide %v2534, %v2531 : tensor<32x1152x7x7xf32>
    %v2536 = stablehlo.subtract %v2529, %v2535 : tensor<32x1152x7x7xf32>
    %v2537 = stablehlo.multiply %v2536, %v2536 : tensor<32x1152x7x7xf32>
    %v2538 = stablehlo.reduce(%v2537 init: %v2530) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2539 = stablehlo.broadcast_in_dim %v2538, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2540 = stablehlo.divide %v2539, %v2531 : tensor<32x1152x7x7xf32>
    %v2541 = stablehlo.add %v2540, %v2532 : tensor<32x1152x7x7xf32>
    %v2542 = stablehlo.rsqrt %v2541 : tensor<32x1152x7x7xf32>
    %v2543 = stablehlo.multiply %v2536, %v2542 : tensor<32x1152x7x7xf32>
    %v2544 = stablehlo.broadcast_in_dim %b15eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2545 = stablehlo.reshape %v2528 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2546 = stablehlo.multiply %v2544, %v2545 : tensor<32x1152x7x7xf32>
    %v2547 = stablehlo.reduce(%v2546 init: %v2530) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2548 = stablehlo.broadcast_in_dim %v2547, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2549 = stablehlo.multiply %v2543, %v2546 : tensor<32x1152x7x7xf32>
    %v2550 = stablehlo.reduce(%v2549 init: %v2530) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2551 = stablehlo.broadcast_in_dim %v2550, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2552 = stablehlo.multiply %v2546, %v2531 : tensor<32x1152x7x7xf32>
    %v2553 = stablehlo.subtract %v2552, %v2548 : tensor<32x1152x7x7xf32>
    %v2554 = stablehlo.multiply %v2543, %v2551 : tensor<32x1152x7x7xf32>
    %v2555 = stablehlo.subtract %v2553, %v2554 : tensor<32x1152x7x7xf32>
    %v2556 = stablehlo.divide %v2542, %v2531 : tensor<32x1152x7x7xf32>
    %v2557 = stablehlo.multiply %v2556, %v2555 : tensor<32x1152x7x7xf32>
    %v2558 = stablehlo.reshape %v2557 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2559 = stablehlo.reshape %v2558 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2560 = stablehlo.reverse %b15eW, dims = [2, 3] : tensor<1152x192x1x1xf32>
    %v2561 = stablehlo.transpose %v2560, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %v2562 = stablehlo.convolution(%v2559, %v2561)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v2563 = stablehlo.reshape %v2562 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v2564 = stablehlo.reshape %v1618 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2565 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2566 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2567 = stablehlo.reduce(%v2564 init: %v2565) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2568 = stablehlo.broadcast_in_dim %v2567, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2569 = stablehlo.divide %v2568, %v2566 : tensor<32x1152x7x7xf32>
    %v2570 = stablehlo.subtract %v2564, %v2569 : tensor<32x1152x7x7xf32>
    %v2571 = stablehlo.multiply %v2570, %v2570 : tensor<32x1152x7x7xf32>
    %v2572 = stablehlo.reduce(%v2571 init: %v2565) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2573 = stablehlo.broadcast_in_dim %v2572, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2574 = stablehlo.divide %v2573, %v2566 : tensor<32x1152x7x7xf32>
    %v2575 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2576 = stablehlo.add %v2574, %v2575 : tensor<32x1152x7x7xf32>
    %v2577 = stablehlo.rsqrt %v2576 : tensor<32x1152x7x7xf32>
    %v2578 = stablehlo.multiply %v2570, %v2577 : tensor<32x1152x7x7xf32>
    %v2579 = stablehlo.reshape %v2528 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2580 = stablehlo.multiply %v2579, %v2578 : tensor<32x1152x7x7xf32>
    %v2581 = stablehlo.reduce(%v2580 init: %v2565) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2582 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2583 = stablehlo.multiply %v2581, %v2582 : tensor<1152xf32>
    %v2584 = stablehlo.subtract %b15eg, %v2583 : tensor<1152xf32>
    %v2585 = stablehlo.reshape %v2528 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2586 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2587 = stablehlo.reduce(%v2585 init: %v2586) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2588 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2589 = stablehlo.multiply %v2587, %v2588 : tensor<1152xf32>
    %v2590 = stablehlo.subtract %b15ebt, %v2589 : tensor<1152xf32>
    %v2591 = stablehlo.reshape %v1613 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2592 = stablehlo.reshape %v2558 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2593 = stablehlo.transpose %v2591, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %v2594 = stablehlo.transpose %v2592, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2595 = stablehlo.convolution(%v2593, %v2594)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<192x1152x1x1xf32>
    %v2596 = stablehlo.transpose %v2595, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %v2597 = stablehlo.constant dense<0.05> : tensor<1152x192x1x1xf32>
    %v2598 = stablehlo.multiply %v2596, %v2597 : tensor<1152x192x1x1xf32>
    %v2599 = stablehlo.subtract %b15eW, %v2598 : tensor<1152x192x1x1xf32>
    %v2600 = stablehlo.reshape %v2563 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2601 = stablehlo.reshape %v2256 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2602 = stablehlo.add %v2600, %v2601 : tensor<32x192x7x7xf32>
    %v2603 = stablehlo.reshape %v2602 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v2604 = stablehlo.reshape %v1589 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2605 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2606 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v2607 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v2608 = stablehlo.reduce(%v2604 init: %v2605) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2609 = stablehlo.broadcast_in_dim %v2608, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2610 = stablehlo.divide %v2609, %v2606 : tensor<32x192x7x7xf32>
    %v2611 = stablehlo.subtract %v2604, %v2610 : tensor<32x192x7x7xf32>
    %v2612 = stablehlo.multiply %v2611, %v2611 : tensor<32x192x7x7xf32>
    %v2613 = stablehlo.reduce(%v2612 init: %v2605) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2614 = stablehlo.broadcast_in_dim %v2613, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2615 = stablehlo.divide %v2614, %v2606 : tensor<32x192x7x7xf32>
    %v2616 = stablehlo.add %v2615, %v2607 : tensor<32x192x7x7xf32>
    %v2617 = stablehlo.rsqrt %v2616 : tensor<32x192x7x7xf32>
    %v2618 = stablehlo.multiply %v2611, %v2617 : tensor<32x192x7x7xf32>
    %v2619 = stablehlo.broadcast_in_dim %b14pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2620 = stablehlo.reshape %v2603 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2621 = stablehlo.multiply %v2619, %v2620 : tensor<32x192x7x7xf32>
    %v2622 = stablehlo.reduce(%v2621 init: %v2605) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2623 = stablehlo.broadcast_in_dim %v2622, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2624 = stablehlo.multiply %v2618, %v2621 : tensor<32x192x7x7xf32>
    %v2625 = stablehlo.reduce(%v2624 init: %v2605) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2626 = stablehlo.broadcast_in_dim %v2625, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2627 = stablehlo.multiply %v2621, %v2606 : tensor<32x192x7x7xf32>
    %v2628 = stablehlo.subtract %v2627, %v2623 : tensor<32x192x7x7xf32>
    %v2629 = stablehlo.multiply %v2618, %v2626 : tensor<32x192x7x7xf32>
    %v2630 = stablehlo.subtract %v2628, %v2629 : tensor<32x192x7x7xf32>
    %v2631 = stablehlo.divide %v2617, %v2606 : tensor<32x192x7x7xf32>
    %v2632 = stablehlo.multiply %v2631, %v2630 : tensor<32x192x7x7xf32>
    %v2633 = stablehlo.reshape %v2632 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v2634 = stablehlo.reshape %v2633 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2635 = stablehlo.reverse %b14pW, dims = [2, 3] : tensor<192x1152x1x1xf32>
    %v2636 = stablehlo.transpose %v2635, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %v2637 = stablehlo.convolution(%v2634, %v2636)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v2638 = stablehlo.reshape %v2637 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2639 = stablehlo.reshape %v1589 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2640 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2641 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v2642 = stablehlo.reduce(%v2639 init: %v2640) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2643 = stablehlo.broadcast_in_dim %v2642, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2644 = stablehlo.divide %v2643, %v2641 : tensor<32x192x7x7xf32>
    %v2645 = stablehlo.subtract %v2639, %v2644 : tensor<32x192x7x7xf32>
    %v2646 = stablehlo.multiply %v2645, %v2645 : tensor<32x192x7x7xf32>
    %v2647 = stablehlo.reduce(%v2646 init: %v2640) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2648 = stablehlo.broadcast_in_dim %v2647, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2649 = stablehlo.divide %v2648, %v2641 : tensor<32x192x7x7xf32>
    %v2650 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v2651 = stablehlo.add %v2649, %v2650 : tensor<32x192x7x7xf32>
    %v2652 = stablehlo.rsqrt %v2651 : tensor<32x192x7x7xf32>
    %v2653 = stablehlo.multiply %v2645, %v2652 : tensor<32x192x7x7xf32>
    %v2654 = stablehlo.reshape %v2603 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2655 = stablehlo.multiply %v2654, %v2653 : tensor<32x192x7x7xf32>
    %v2656 = stablehlo.reduce(%v2655 init: %v2640) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2657 = stablehlo.constant dense<0.05> : tensor<192xf32>
    %v2658 = stablehlo.multiply %v2656, %v2657 : tensor<192xf32>
    %v2659 = stablehlo.subtract %b14pg, %v2658 : tensor<192xf32>
    %v2660 = stablehlo.reshape %v2603 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2661 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2662 = stablehlo.reduce(%v2660 init: %v2661) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2663 = stablehlo.constant dense<0.05> : tensor<192xf32>
    %v2664 = stablehlo.multiply %v2662, %v2663 : tensor<192xf32>
    %v2665 = stablehlo.subtract %b14pbt, %v2664 : tensor<192xf32>
    %v2666 = stablehlo.reshape %v1584 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2667 = stablehlo.reshape %v2633 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2668 = stablehlo.transpose %v2666, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2669 = stablehlo.transpose %v2667, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %v2670 = stablehlo.convolution(%v2668, %v2669)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<192x32x7x7xf32>) -> tensor<1152x192x1x1xf32>
    %v2671 = stablehlo.transpose %v2670, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %v2672 = stablehlo.constant dense<0.05> : tensor<192x1152x1x1xf32>
    %v2673 = stablehlo.multiply %v2671, %v2672 : tensor<192x1152x1x1xf32>
    %v2674 = stablehlo.subtract %b14pW, %v2673 : tensor<192x1152x1x1xf32>
    %v2675 = stablehlo.reshape %v1554 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2676 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2677 = stablehlo.reduce(%v2675 init: %v2676) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2678 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v2679 = stablehlo.divide %v2677, %v2678 : tensor<32x1152xf32>
    %v2680 = stablehlo.dot_general %v2679, %b14zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v2681 = stablehlo.broadcast_in_dim %b14zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v2682 = stablehlo.add %v2680, %v2681 : tensor<32x48xf32>
    %v2683 = stablehlo.logistic %v2682 : tensor<32x48xf32>
    %v2684 = stablehlo.multiply %v2682, %v2683 : tensor<32x48xf32>
    %v2685 = stablehlo.dot_general %v2684, %b14zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v2686 = stablehlo.broadcast_in_dim %b14zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v2687 = stablehlo.add %v2685, %v2686 : tensor<32x1152xf32>
    %v2688 = stablehlo.logistic %v2687 : tensor<32x1152xf32>
    %v2689 = stablehlo.reshape %v2638 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2690 = stablehlo.broadcast_in_dim %v2688, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2691 = stablehlo.multiply %v2690, %v2689 : tensor<32x1152x7x7xf32>
    %v2692 = stablehlo.multiply %v2675, %v2689 : tensor<32x1152x7x7xf32>
    %v2693 = stablehlo.reduce(%v2692 init: %v2676) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2694 = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %v2695 = stablehlo.subtract %v2694, %v2688 : tensor<32x1152xf32>
    %v2696 = stablehlo.multiply %v2688, %v2695 : tensor<32x1152xf32>
    %v2697 = stablehlo.multiply %v2693, %v2696 : tensor<32x1152xf32>
    %v2698 = stablehlo.dot_general %v2697, %b14zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<48x1152xf32>) -> tensor<32x48xf32>
    %v2699 = stablehlo.logistic %v2682 : tensor<32x48xf32>
    %v2700 = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %v2701 = stablehlo.subtract %v2700, %v2699 : tensor<32x48xf32>
    %v2702 = stablehlo.multiply %v2682, %v2701 : tensor<32x48xf32>
    %v2703 = stablehlo.add %v2700, %v2702 : tensor<32x48xf32>
    %v2704 = stablehlo.multiply %v2699, %v2703 : tensor<32x48xf32>
    %v2705 = stablehlo.multiply %v2698, %v2704 : tensor<32x48xf32>
    %v2706 = stablehlo.dot_general %v2705, %b14zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<1152x48xf32>) -> tensor<32x1152xf32>
    %v2707 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v2708 = stablehlo.divide %v2706, %v2707 : tensor<32x1152xf32>
    %v2709 = stablehlo.broadcast_in_dim %v2708, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2710 = stablehlo.add %v2691, %v2709 : tensor<32x1152x7x7xf32>
    %v2711 = stablehlo.reshape %v2710 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2712 = stablehlo.reshape %v1554 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2713 = stablehlo.reshape %v2638 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2714 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2715 = stablehlo.multiply %v2712, %v2713 : tensor<32x1152x7x7xf32>
    %v2716 = stablehlo.reduce(%v2715 init: %v2714) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2717 = stablehlo.logistic %v1567 : tensor<32x1152xf32>
    %v2718 = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %v2719 = stablehlo.subtract %v2718, %v2717 : tensor<32x1152xf32>
    %v2720 = stablehlo.multiply %v2717, %v2719 : tensor<32x1152xf32>
    %v2721 = stablehlo.multiply %v2716, %v2720 : tensor<32x1152xf32>
    %v2722 = stablehlo.dot_general %v1564, %v2721, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<32x1152xf32>) -> tensor<48x1152xf32>
    %v2723 = stablehlo.constant dense<0.05> : tensor<48x1152xf32>
    %v2724 = stablehlo.multiply %v2722, %v2723 : tensor<48x1152xf32>
    %v2725 = stablehlo.subtract %b14zW2, %v2724 : tensor<48x1152xf32>
    %v2726 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2727 = stablehlo.reduce(%v2721 init: %v2726) applies stablehlo.add across dimensions = [0] : (tensor<32x1152xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2728 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2729 = stablehlo.multiply %v2727, %v2728 : tensor<1152xf32>
    %v2730 = stablehlo.subtract %b14zb2, %v2729 : tensor<1152xf32>
    %v2731 = stablehlo.reshape %v2721 : (tensor<32x1152xf32>) -> tensor<32x1x1152xf32>
    %v2732 = stablehlo.dot_general %v2731, %b14zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x1152xf32>, tensor<48x1152xf32>) -> tensor<32x1x48xf32>
    %v2733 = stablehlo.reshape %v2732 : (tensor<32x1x48xf32>) -> tensor<32x48xf32>
    %v2734 = stablehlo.logistic %v1562 : tensor<32x48xf32>
    %v2735 = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %v2736 = stablehlo.subtract %v2735, %v2734 : tensor<32x48xf32>
    %v2737 = stablehlo.multiply %v1562, %v2736 : tensor<32x48xf32>
    %v2738 = stablehlo.add %v2735, %v2737 : tensor<32x48xf32>
    %v2739 = stablehlo.multiply %v2734, %v2738 : tensor<32x48xf32>
    %v2740 = stablehlo.multiply %v2733, %v2739 : tensor<32x48xf32>
    %v2741 = stablehlo.dot_general %v1559, %v2740, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<32x48xf32>) -> tensor<1152x48xf32>
    %v2742 = stablehlo.constant dense<0.05> : tensor<1152x48xf32>
    %v2743 = stablehlo.multiply %v2741, %v2742 : tensor<1152x48xf32>
    %v2744 = stablehlo.subtract %b14zW1, %v2743 : tensor<1152x48xf32>
    %v2745 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2746 = stablehlo.reduce(%v2740 init: %v2745) applies stablehlo.add across dimensions = [0] : (tensor<32x48xf32>, tensor<f32>) -> tensor<48xf32>
    %v2747 = stablehlo.constant dense<0.05> : tensor<48xf32>
    %v2748 = stablehlo.multiply %v2746, %v2747 : tensor<48xf32>
    %v2749 = stablehlo.subtract %b14zb1, %v2748 : tensor<48xf32>
    %v2750 = stablehlo.reshape %v2711 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2751 = stablehlo.reshape %v1550 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2752 = stablehlo.logistic %v2751 : tensor<32x1152x7x7xf32>
    %v2753 = stablehlo.constant dense<1.0> : tensor<32x1152x7x7xf32>
    %v2754 = stablehlo.subtract %v2753, %v2752 : tensor<32x1152x7x7xf32>
    %v2755 = stablehlo.multiply %v2751, %v2754 : tensor<32x1152x7x7xf32>
    %v2756 = stablehlo.add %v2753, %v2755 : tensor<32x1152x7x7xf32>
    %v2757 = stablehlo.multiply %v2752, %v2756 : tensor<32x1152x7x7xf32>
    %v2758 = stablehlo.multiply %v2750, %v2757 : tensor<32x1152x7x7xf32>
    %v2759 = stablehlo.reshape %v2758 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2760 = stablehlo.reshape %v1530 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2761 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2762 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2763 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2764 = stablehlo.reduce(%v2760 init: %v2761) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2765 = stablehlo.broadcast_in_dim %v2764, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2766 = stablehlo.divide %v2765, %v2762 : tensor<32x1152x7x7xf32>
    %v2767 = stablehlo.subtract %v2760, %v2766 : tensor<32x1152x7x7xf32>
    %v2768 = stablehlo.multiply %v2767, %v2767 : tensor<32x1152x7x7xf32>
    %v2769 = stablehlo.reduce(%v2768 init: %v2761) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2770 = stablehlo.broadcast_in_dim %v2769, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2771 = stablehlo.divide %v2770, %v2762 : tensor<32x1152x7x7xf32>
    %v2772 = stablehlo.add %v2771, %v2763 : tensor<32x1152x7x7xf32>
    %v2773 = stablehlo.rsqrt %v2772 : tensor<32x1152x7x7xf32>
    %v2774 = stablehlo.multiply %v2767, %v2773 : tensor<32x1152x7x7xf32>
    %v2775 = stablehlo.broadcast_in_dim %b14dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2776 = stablehlo.reshape %v2759 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2777 = stablehlo.multiply %v2775, %v2776 : tensor<32x1152x7x7xf32>
    %v2778 = stablehlo.reduce(%v2777 init: %v2761) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2779 = stablehlo.broadcast_in_dim %v2778, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2780 = stablehlo.multiply %v2774, %v2777 : tensor<32x1152x7x7xf32>
    %v2781 = stablehlo.reduce(%v2780 init: %v2761) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2782 = stablehlo.broadcast_in_dim %v2781, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2783 = stablehlo.multiply %v2777, %v2762 : tensor<32x1152x7x7xf32>
    %v2784 = stablehlo.subtract %v2783, %v2779 : tensor<32x1152x7x7xf32>
    %v2785 = stablehlo.multiply %v2774, %v2782 : tensor<32x1152x7x7xf32>
    %v2786 = stablehlo.subtract %v2784, %v2785 : tensor<32x1152x7x7xf32>
    %v2787 = stablehlo.divide %v2773, %v2762 : tensor<32x1152x7x7xf32>
    %v2788 = stablehlo.multiply %v2787, %v2786 : tensor<32x1152x7x7xf32>
    %v2789 = stablehlo.reshape %v2788 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2790 = stablehlo.reshape %v2789 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2791 = stablehlo.reverse %b14dW, dims = [2, 3] : tensor<1152x1x5x5xf32>
    %v2792 = stablehlo.convolution(%v2790, %v2791)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v2793 = stablehlo.reshape %v2792 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2794 = stablehlo.reshape %v1530 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2795 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2796 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2797 = stablehlo.reduce(%v2794 init: %v2795) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2798 = stablehlo.broadcast_in_dim %v2797, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2799 = stablehlo.divide %v2798, %v2796 : tensor<32x1152x7x7xf32>
    %v2800 = stablehlo.subtract %v2794, %v2799 : tensor<32x1152x7x7xf32>
    %v2801 = stablehlo.multiply %v2800, %v2800 : tensor<32x1152x7x7xf32>
    %v2802 = stablehlo.reduce(%v2801 init: %v2795) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2803 = stablehlo.broadcast_in_dim %v2802, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2804 = stablehlo.divide %v2803, %v2796 : tensor<32x1152x7x7xf32>
    %v2805 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2806 = stablehlo.add %v2804, %v2805 : tensor<32x1152x7x7xf32>
    %v2807 = stablehlo.rsqrt %v2806 : tensor<32x1152x7x7xf32>
    %v2808 = stablehlo.multiply %v2800, %v2807 : tensor<32x1152x7x7xf32>
    %v2809 = stablehlo.reshape %v2759 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2810 = stablehlo.multiply %v2809, %v2808 : tensor<32x1152x7x7xf32>
    %v2811 = stablehlo.reduce(%v2810 init: %v2795) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2812 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2813 = stablehlo.multiply %v2811, %v2812 : tensor<1152xf32>
    %v2814 = stablehlo.subtract %b14dg, %v2813 : tensor<1152xf32>
    %v2815 = stablehlo.reshape %v2759 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2816 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2817 = stablehlo.reduce(%v2815 init: %v2816) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2818 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2819 = stablehlo.multiply %v2817, %v2818 : tensor<1152xf32>
    %v2820 = stablehlo.subtract %b14dbt, %v2819 : tensor<1152xf32>
    %v2821 = stablehlo.reshape %v1525 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2822 = stablehlo.reshape %v2789 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2823 = stablehlo.transpose %v2821, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2824 = stablehlo.transpose %v2822, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2825 = stablehlo.convolution(%v2823, %v2824)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1152 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<1x1152x5x5xf32>
    %v2826 = stablehlo.reshape %v2825 : (tensor<1x1152x5x5xf32>) -> tensor<1152x1x5x5xf32>
    %v2827 = stablehlo.constant dense<0.05> : tensor<1152x1x5x5xf32>
    %v2828 = stablehlo.multiply %v2826, %v2827 : tensor<1152x1x5x5xf32>
    %v2829 = stablehlo.subtract %b14dW, %v2828 : tensor<1152x1x5x5xf32>
    %v2830 = stablehlo.reshape %v2793 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2831 = stablehlo.reshape %v1521 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2832 = stablehlo.logistic %v2831 : tensor<32x1152x7x7xf32>
    %v2833 = stablehlo.constant dense<1.0> : tensor<32x1152x7x7xf32>
    %v2834 = stablehlo.subtract %v2833, %v2832 : tensor<32x1152x7x7xf32>
    %v2835 = stablehlo.multiply %v2831, %v2834 : tensor<32x1152x7x7xf32>
    %v2836 = stablehlo.add %v2833, %v2835 : tensor<32x1152x7x7xf32>
    %v2837 = stablehlo.multiply %v2832, %v2836 : tensor<32x1152x7x7xf32>
    %v2838 = stablehlo.multiply %v2830, %v2837 : tensor<32x1152x7x7xf32>
    %v2839 = stablehlo.reshape %v2838 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2840 = stablehlo.reshape %v1501 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2841 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2842 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2843 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2844 = stablehlo.reduce(%v2840 init: %v2841) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2845 = stablehlo.broadcast_in_dim %v2844, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2846 = stablehlo.divide %v2845, %v2842 : tensor<32x1152x7x7xf32>
    %v2847 = stablehlo.subtract %v2840, %v2846 : tensor<32x1152x7x7xf32>
    %v2848 = stablehlo.multiply %v2847, %v2847 : tensor<32x1152x7x7xf32>
    %v2849 = stablehlo.reduce(%v2848 init: %v2841) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2850 = stablehlo.broadcast_in_dim %v2849, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2851 = stablehlo.divide %v2850, %v2842 : tensor<32x1152x7x7xf32>
    %v2852 = stablehlo.add %v2851, %v2843 : tensor<32x1152x7x7xf32>
    %v2853 = stablehlo.rsqrt %v2852 : tensor<32x1152x7x7xf32>
    %v2854 = stablehlo.multiply %v2847, %v2853 : tensor<32x1152x7x7xf32>
    %v2855 = stablehlo.broadcast_in_dim %b14eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2856 = stablehlo.reshape %v2839 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2857 = stablehlo.multiply %v2855, %v2856 : tensor<32x1152x7x7xf32>
    %v2858 = stablehlo.reduce(%v2857 init: %v2841) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2859 = stablehlo.broadcast_in_dim %v2858, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2860 = stablehlo.multiply %v2854, %v2857 : tensor<32x1152x7x7xf32>
    %v2861 = stablehlo.reduce(%v2860 init: %v2841) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2862 = stablehlo.broadcast_in_dim %v2861, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2863 = stablehlo.multiply %v2857, %v2842 : tensor<32x1152x7x7xf32>
    %v2864 = stablehlo.subtract %v2863, %v2859 : tensor<32x1152x7x7xf32>
    %v2865 = stablehlo.multiply %v2854, %v2862 : tensor<32x1152x7x7xf32>
    %v2866 = stablehlo.subtract %v2864, %v2865 : tensor<32x1152x7x7xf32>
    %v2867 = stablehlo.divide %v2853, %v2842 : tensor<32x1152x7x7xf32>
    %v2868 = stablehlo.multiply %v2867, %v2866 : tensor<32x1152x7x7xf32>
    %v2869 = stablehlo.reshape %v2868 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2870 = stablehlo.reshape %v2869 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2871 = stablehlo.reverse %b14eW, dims = [2, 3] : tensor<1152x192x1x1xf32>
    %v2872 = stablehlo.transpose %v2871, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %v2873 = stablehlo.convolution(%v2870, %v2872)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v2874 = stablehlo.reshape %v2873 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v2875 = stablehlo.reshape %v1501 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2876 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2877 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v2878 = stablehlo.reduce(%v2875 init: %v2876) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2879 = stablehlo.broadcast_in_dim %v2878, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2880 = stablehlo.divide %v2879, %v2877 : tensor<32x1152x7x7xf32>
    %v2881 = stablehlo.subtract %v2875, %v2880 : tensor<32x1152x7x7xf32>
    %v2882 = stablehlo.multiply %v2881, %v2881 : tensor<32x1152x7x7xf32>
    %v2883 = stablehlo.reduce(%v2882 init: %v2876) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2884 = stablehlo.broadcast_in_dim %v2883, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v2885 = stablehlo.divide %v2884, %v2877 : tensor<32x1152x7x7xf32>
    %v2886 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v2887 = stablehlo.add %v2885, %v2886 : tensor<32x1152x7x7xf32>
    %v2888 = stablehlo.rsqrt %v2887 : tensor<32x1152x7x7xf32>
    %v2889 = stablehlo.multiply %v2881, %v2888 : tensor<32x1152x7x7xf32>
    %v2890 = stablehlo.reshape %v2839 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2891 = stablehlo.multiply %v2890, %v2889 : tensor<32x1152x7x7xf32>
    %v2892 = stablehlo.reduce(%v2891 init: %v2876) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2893 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2894 = stablehlo.multiply %v2892, %v2893 : tensor<1152xf32>
    %v2895 = stablehlo.subtract %b14eg, %v2894 : tensor<1152xf32>
    %v2896 = stablehlo.reshape %v2839 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2897 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2898 = stablehlo.reduce(%v2896 init: %v2897) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v2899 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v2900 = stablehlo.multiply %v2898, %v2899 : tensor<1152xf32>
    %v2901 = stablehlo.subtract %b14ebt, %v2900 : tensor<1152xf32>
    %v2902 = stablehlo.reshape %v1496 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2903 = stablehlo.reshape %v2869 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2904 = stablehlo.transpose %v2902, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %v2905 = stablehlo.transpose %v2903, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2906 = stablehlo.convolution(%v2904, %v2905)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<192x1152x1x1xf32>
    %v2907 = stablehlo.transpose %v2906, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %v2908 = stablehlo.constant dense<0.05> : tensor<1152x192x1x1xf32>
    %v2909 = stablehlo.multiply %v2907, %v2908 : tensor<1152x192x1x1xf32>
    %v2910 = stablehlo.subtract %b14eW, %v2909 : tensor<1152x192x1x1xf32>
    %v2911 = stablehlo.reshape %v2874 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2912 = stablehlo.reshape %v2603 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2913 = stablehlo.add %v2911, %v2912 : tensor<32x192x7x7xf32>
    %v2914 = stablehlo.reshape %v2913 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v2915 = stablehlo.reshape %v1472 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2916 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2917 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v2918 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v2919 = stablehlo.reduce(%v2915 init: %v2916) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2920 = stablehlo.broadcast_in_dim %v2919, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2921 = stablehlo.divide %v2920, %v2917 : tensor<32x192x7x7xf32>
    %v2922 = stablehlo.subtract %v2915, %v2921 : tensor<32x192x7x7xf32>
    %v2923 = stablehlo.multiply %v2922, %v2922 : tensor<32x192x7x7xf32>
    %v2924 = stablehlo.reduce(%v2923 init: %v2916) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2925 = stablehlo.broadcast_in_dim %v2924, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2926 = stablehlo.divide %v2925, %v2917 : tensor<32x192x7x7xf32>
    %v2927 = stablehlo.add %v2926, %v2918 : tensor<32x192x7x7xf32>
    %v2928 = stablehlo.rsqrt %v2927 : tensor<32x192x7x7xf32>
    %v2929 = stablehlo.multiply %v2922, %v2928 : tensor<32x192x7x7xf32>
    %v2930 = stablehlo.broadcast_in_dim %b13pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2931 = stablehlo.reshape %v2914 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2932 = stablehlo.multiply %v2930, %v2931 : tensor<32x192x7x7xf32>
    %v2933 = stablehlo.reduce(%v2932 init: %v2916) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2934 = stablehlo.broadcast_in_dim %v2933, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2935 = stablehlo.multiply %v2929, %v2932 : tensor<32x192x7x7xf32>
    %v2936 = stablehlo.reduce(%v2935 init: %v2916) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2937 = stablehlo.broadcast_in_dim %v2936, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2938 = stablehlo.multiply %v2932, %v2917 : tensor<32x192x7x7xf32>
    %v2939 = stablehlo.subtract %v2938, %v2934 : tensor<32x192x7x7xf32>
    %v2940 = stablehlo.multiply %v2929, %v2937 : tensor<32x192x7x7xf32>
    %v2941 = stablehlo.subtract %v2939, %v2940 : tensor<32x192x7x7xf32>
    %v2942 = stablehlo.divide %v2928, %v2917 : tensor<32x192x7x7xf32>
    %v2943 = stablehlo.multiply %v2942, %v2941 : tensor<32x192x7x7xf32>
    %v2944 = stablehlo.reshape %v2943 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v2945 = stablehlo.reshape %v2944 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2946 = stablehlo.reverse %b13pW, dims = [2, 3] : tensor<192x1152x1x1xf32>
    %v2947 = stablehlo.transpose %v2946, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %v2948 = stablehlo.convolution(%v2945, %v2947)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v2949 = stablehlo.reshape %v2948 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v2950 = stablehlo.reshape %v1472 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2951 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2952 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v2953 = stablehlo.reduce(%v2950 init: %v2951) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2954 = stablehlo.broadcast_in_dim %v2953, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2955 = stablehlo.divide %v2954, %v2952 : tensor<32x192x7x7xf32>
    %v2956 = stablehlo.subtract %v2950, %v2955 : tensor<32x192x7x7xf32>
    %v2957 = stablehlo.multiply %v2956, %v2956 : tensor<32x192x7x7xf32>
    %v2958 = stablehlo.reduce(%v2957 init: %v2951) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2959 = stablehlo.broadcast_in_dim %v2958, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v2960 = stablehlo.divide %v2959, %v2952 : tensor<32x192x7x7xf32>
    %v2961 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v2962 = stablehlo.add %v2960, %v2961 : tensor<32x192x7x7xf32>
    %v2963 = stablehlo.rsqrt %v2962 : tensor<32x192x7x7xf32>
    %v2964 = stablehlo.multiply %v2956, %v2963 : tensor<32x192x7x7xf32>
    %v2965 = stablehlo.reshape %v2914 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2966 = stablehlo.multiply %v2965, %v2964 : tensor<32x192x7x7xf32>
    %v2967 = stablehlo.reduce(%v2966 init: %v2951) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2968 = stablehlo.constant dense<0.05> : tensor<192xf32>
    %v2969 = stablehlo.multiply %v2967, %v2968 : tensor<192xf32>
    %v2970 = stablehlo.subtract %b13pg, %v2969 : tensor<192xf32>
    %v2971 = stablehlo.reshape %v2914 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2972 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2973 = stablehlo.reduce(%v2971 init: %v2972) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v2974 = stablehlo.constant dense<0.05> : tensor<192xf32>
    %v2975 = stablehlo.multiply %v2973, %v2974 : tensor<192xf32>
    %v2976 = stablehlo.subtract %b13pbt, %v2975 : tensor<192xf32>
    %v2977 = stablehlo.reshape %v1467 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2978 = stablehlo.reshape %v2944 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v2979 = stablehlo.transpose %v2977, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v2980 = stablehlo.transpose %v2978, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %v2981 = stablehlo.convolution(%v2979, %v2980)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<192x32x7x7xf32>) -> tensor<1152x192x1x1xf32>
    %v2982 = stablehlo.transpose %v2981, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %v2983 = stablehlo.constant dense<0.05> : tensor<192x1152x1x1xf32>
    %v2984 = stablehlo.multiply %v2982, %v2983 : tensor<192x1152x1x1xf32>
    %v2985 = stablehlo.subtract %b13pW, %v2984 : tensor<192x1152x1x1xf32>
    %v2986 = stablehlo.reshape %v1437 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v2987 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2988 = stablehlo.reduce(%v2986 init: %v2987) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v2989 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v2990 = stablehlo.divide %v2988, %v2989 : tensor<32x1152xf32>
    %v2991 = stablehlo.dot_general %v2990, %b13zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v2992 = stablehlo.broadcast_in_dim %b13zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v2993 = stablehlo.add %v2991, %v2992 : tensor<32x48xf32>
    %v2994 = stablehlo.logistic %v2993 : tensor<32x48xf32>
    %v2995 = stablehlo.multiply %v2993, %v2994 : tensor<32x48xf32>
    %v2996 = stablehlo.dot_general %v2995, %b13zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v2997 = stablehlo.broadcast_in_dim %b13zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v2998 = stablehlo.add %v2996, %v2997 : tensor<32x1152xf32>
    %v2999 = stablehlo.logistic %v2998 : tensor<32x1152xf32>
    %v3000 = stablehlo.reshape %v2949 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3001 = stablehlo.broadcast_in_dim %v2999, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v3002 = stablehlo.multiply %v3001, %v3000 : tensor<32x1152x7x7xf32>
    %v3003 = stablehlo.multiply %v2986, %v3000 : tensor<32x1152x7x7xf32>
    %v3004 = stablehlo.reduce(%v3003 init: %v2987) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v3005 = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %v3006 = stablehlo.subtract %v3005, %v2999 : tensor<32x1152xf32>
    %v3007 = stablehlo.multiply %v2999, %v3006 : tensor<32x1152xf32>
    %v3008 = stablehlo.multiply %v3004, %v3007 : tensor<32x1152xf32>
    %v3009 = stablehlo.dot_general %v3008, %b13zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<48x1152xf32>) -> tensor<32x48xf32>
    %v3010 = stablehlo.logistic %v2993 : tensor<32x48xf32>
    %v3011 = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %v3012 = stablehlo.subtract %v3011, %v3010 : tensor<32x48xf32>
    %v3013 = stablehlo.multiply %v2993, %v3012 : tensor<32x48xf32>
    %v3014 = stablehlo.add %v3011, %v3013 : tensor<32x48xf32>
    %v3015 = stablehlo.multiply %v3010, %v3014 : tensor<32x48xf32>
    %v3016 = stablehlo.multiply %v3009, %v3015 : tensor<32x48xf32>
    %v3017 = stablehlo.dot_general %v3016, %b13zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<1152x48xf32>) -> tensor<32x1152xf32>
    %v3018 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v3019 = stablehlo.divide %v3017, %v3018 : tensor<32x1152xf32>
    %v3020 = stablehlo.broadcast_in_dim %v3019, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v3021 = stablehlo.add %v3002, %v3020 : tensor<32x1152x7x7xf32>
    %v3022 = stablehlo.reshape %v3021 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v3023 = stablehlo.reshape %v1437 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3024 = stablehlo.reshape %v2949 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3025 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3026 = stablehlo.multiply %v3023, %v3024 : tensor<32x1152x7x7xf32>
    %v3027 = stablehlo.reduce(%v3026 init: %v3025) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v3028 = stablehlo.logistic %v1450 : tensor<32x1152xf32>
    %v3029 = stablehlo.constant dense<1.0> : tensor<32x1152xf32>
    %v3030 = stablehlo.subtract %v3029, %v3028 : tensor<32x1152xf32>
    %v3031 = stablehlo.multiply %v3028, %v3030 : tensor<32x1152xf32>
    %v3032 = stablehlo.multiply %v3027, %v3031 : tensor<32x1152xf32>
    %v3033 = stablehlo.dot_general %v1447, %v3032, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<32x1152xf32>) -> tensor<48x1152xf32>
    %v3034 = stablehlo.constant dense<0.05> : tensor<48x1152xf32>
    %v3035 = stablehlo.multiply %v3033, %v3034 : tensor<48x1152xf32>
    %v3036 = stablehlo.subtract %b13zW2, %v3035 : tensor<48x1152xf32>
    %v3037 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3038 = stablehlo.reduce(%v3032 init: %v3037) applies stablehlo.add across dimensions = [0] : (tensor<32x1152xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3039 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v3040 = stablehlo.multiply %v3038, %v3039 : tensor<1152xf32>
    %v3041 = stablehlo.subtract %b13zb2, %v3040 : tensor<1152xf32>
    %v3042 = stablehlo.reshape %v3032 : (tensor<32x1152xf32>) -> tensor<32x1x1152xf32>
    %v3043 = stablehlo.dot_general %v3042, %b13zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x1152xf32>, tensor<48x1152xf32>) -> tensor<32x1x48xf32>
    %v3044 = stablehlo.reshape %v3043 : (tensor<32x1x48xf32>) -> tensor<32x48xf32>
    %v3045 = stablehlo.logistic %v1445 : tensor<32x48xf32>
    %v3046 = stablehlo.constant dense<1.0> : tensor<32x48xf32>
    %v3047 = stablehlo.subtract %v3046, %v3045 : tensor<32x48xf32>
    %v3048 = stablehlo.multiply %v1445, %v3047 : tensor<32x48xf32>
    %v3049 = stablehlo.add %v3046, %v3048 : tensor<32x48xf32>
    %v3050 = stablehlo.multiply %v3045, %v3049 : tensor<32x48xf32>
    %v3051 = stablehlo.multiply %v3044, %v3050 : tensor<32x48xf32>
    %v3052 = stablehlo.dot_general %v1442, %v3051, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<32x48xf32>) -> tensor<1152x48xf32>
    %v3053 = stablehlo.constant dense<0.05> : tensor<1152x48xf32>
    %v3054 = stablehlo.multiply %v3052, %v3053 : tensor<1152x48xf32>
    %v3055 = stablehlo.subtract %b13zW1, %v3054 : tensor<1152x48xf32>
    %v3056 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3057 = stablehlo.reduce(%v3051 init: %v3056) applies stablehlo.add across dimensions = [0] : (tensor<32x48xf32>, tensor<f32>) -> tensor<48xf32>
    %v3058 = stablehlo.constant dense<0.05> : tensor<48xf32>
    %v3059 = stablehlo.multiply %v3057, %v3058 : tensor<48xf32>
    %v3060 = stablehlo.subtract %b13zb1, %v3059 : tensor<48xf32>
    %v3061 = stablehlo.reshape %v3022 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3062 = stablehlo.reshape %v1433 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3063 = stablehlo.logistic %v3062 : tensor<32x1152x7x7xf32>
    %v3064 = stablehlo.constant dense<1.0> : tensor<32x1152x7x7xf32>
    %v3065 = stablehlo.subtract %v3064, %v3063 : tensor<32x1152x7x7xf32>
    %v3066 = stablehlo.multiply %v3062, %v3065 : tensor<32x1152x7x7xf32>
    %v3067 = stablehlo.add %v3064, %v3066 : tensor<32x1152x7x7xf32>
    %v3068 = stablehlo.multiply %v3063, %v3067 : tensor<32x1152x7x7xf32>
    %v3069 = stablehlo.multiply %v3061, %v3068 : tensor<32x1152x7x7xf32>
    %v3070 = stablehlo.reshape %v3069 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v3071 = stablehlo.reshape %v1413 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3072 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3073 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v3074 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v3075 = stablehlo.reduce(%v3071 init: %v3072) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3076 = stablehlo.broadcast_in_dim %v3075, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v3077 = stablehlo.divide %v3076, %v3073 : tensor<32x1152x7x7xf32>
    %v3078 = stablehlo.subtract %v3071, %v3077 : tensor<32x1152x7x7xf32>
    %v3079 = stablehlo.multiply %v3078, %v3078 : tensor<32x1152x7x7xf32>
    %v3080 = stablehlo.reduce(%v3079 init: %v3072) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3081 = stablehlo.broadcast_in_dim %v3080, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v3082 = stablehlo.divide %v3081, %v3073 : tensor<32x1152x7x7xf32>
    %v3083 = stablehlo.add %v3082, %v3074 : tensor<32x1152x7x7xf32>
    %v3084 = stablehlo.rsqrt %v3083 : tensor<32x1152x7x7xf32>
    %v3085 = stablehlo.multiply %v3078, %v3084 : tensor<32x1152x7x7xf32>
    %v3086 = stablehlo.broadcast_in_dim %b13dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v3087 = stablehlo.reshape %v3070 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3088 = stablehlo.multiply %v3086, %v3087 : tensor<32x1152x7x7xf32>
    %v3089 = stablehlo.reduce(%v3088 init: %v3072) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3090 = stablehlo.broadcast_in_dim %v3089, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v3091 = stablehlo.multiply %v3085, %v3088 : tensor<32x1152x7x7xf32>
    %v3092 = stablehlo.reduce(%v3091 init: %v3072) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3093 = stablehlo.broadcast_in_dim %v3092, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v3094 = stablehlo.multiply %v3088, %v3073 : tensor<32x1152x7x7xf32>
    %v3095 = stablehlo.subtract %v3094, %v3090 : tensor<32x1152x7x7xf32>
    %v3096 = stablehlo.multiply %v3085, %v3093 : tensor<32x1152x7x7xf32>
    %v3097 = stablehlo.subtract %v3095, %v3096 : tensor<32x1152x7x7xf32>
    %v3098 = stablehlo.divide %v3084, %v3073 : tensor<32x1152x7x7xf32>
    %v3099 = stablehlo.multiply %v3098, %v3097 : tensor<32x1152x7x7xf32>
    %v3100 = stablehlo.reshape %v3099 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v3101 = stablehlo.reshape %v3100 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3102 = stablehlo.reverse %b13dW, dims = [2, 3] : tensor<1152x1x5x5xf32>
    %v3103 = stablehlo.convolution(%v3101, %v3102)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v3104 = stablehlo.reshape %v3103 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v3105 = stablehlo.reshape %v1413 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3106 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3107 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v3108 = stablehlo.reduce(%v3105 init: %v3106) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3109 = stablehlo.broadcast_in_dim %v3108, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v3110 = stablehlo.divide %v3109, %v3107 : tensor<32x1152x7x7xf32>
    %v3111 = stablehlo.subtract %v3105, %v3110 : tensor<32x1152x7x7xf32>
    %v3112 = stablehlo.multiply %v3111, %v3111 : tensor<32x1152x7x7xf32>
    %v3113 = stablehlo.reduce(%v3112 init: %v3106) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3114 = stablehlo.broadcast_in_dim %v3113, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v3115 = stablehlo.divide %v3114, %v3107 : tensor<32x1152x7x7xf32>
    %v3116 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v3117 = stablehlo.add %v3115, %v3116 : tensor<32x1152x7x7xf32>
    %v3118 = stablehlo.rsqrt %v3117 : tensor<32x1152x7x7xf32>
    %v3119 = stablehlo.multiply %v3111, %v3118 : tensor<32x1152x7x7xf32>
    %v3120 = stablehlo.reshape %v3070 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3121 = stablehlo.multiply %v3120, %v3119 : tensor<32x1152x7x7xf32>
    %v3122 = stablehlo.reduce(%v3121 init: %v3106) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3123 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v3124 = stablehlo.multiply %v3122, %v3123 : tensor<1152xf32>
    %v3125 = stablehlo.subtract %b13dg, %v3124 : tensor<1152xf32>
    %v3126 = stablehlo.reshape %v3070 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3127 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3128 = stablehlo.reduce(%v3126 init: %v3127) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3129 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v3130 = stablehlo.multiply %v3128, %v3129 : tensor<1152xf32>
    %v3131 = stablehlo.subtract %b13dbt, %v3130 : tensor<1152xf32>
    %v3132 = stablehlo.reshape %v1408 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3133 = stablehlo.reshape %v3100 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3134 = stablehlo.transpose %v3132, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v3135 = stablehlo.transpose %v3133, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v3136 = stablehlo.convolution(%v3134, %v3135)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1152 : i64, feature_group_count = 1 : i64} : (tensor<1152x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<1x1152x5x5xf32>
    %v3137 = stablehlo.reshape %v3136 : (tensor<1x1152x5x5xf32>) -> tensor<1152x1x5x5xf32>
    %v3138 = stablehlo.constant dense<0.05> : tensor<1152x1x5x5xf32>
    %v3139 = stablehlo.multiply %v3137, %v3138 : tensor<1152x1x5x5xf32>
    %v3140 = stablehlo.subtract %b13dW, %v3139 : tensor<1152x1x5x5xf32>
    %v3141 = stablehlo.reshape %v3104 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3142 = stablehlo.reshape %v1404 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3143 = stablehlo.logistic %v3142 : tensor<32x1152x7x7xf32>
    %v3144 = stablehlo.constant dense<1.0> : tensor<32x1152x7x7xf32>
    %v3145 = stablehlo.subtract %v3144, %v3143 : tensor<32x1152x7x7xf32>
    %v3146 = stablehlo.multiply %v3142, %v3145 : tensor<32x1152x7x7xf32>
    %v3147 = stablehlo.add %v3144, %v3146 : tensor<32x1152x7x7xf32>
    %v3148 = stablehlo.multiply %v3143, %v3147 : tensor<32x1152x7x7xf32>
    %v3149 = stablehlo.multiply %v3141, %v3148 : tensor<32x1152x7x7xf32>
    %v3150 = stablehlo.reshape %v3149 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v3151 = stablehlo.reshape %v1384 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3152 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3153 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v3154 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v3155 = stablehlo.reduce(%v3151 init: %v3152) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3156 = stablehlo.broadcast_in_dim %v3155, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v3157 = stablehlo.divide %v3156, %v3153 : tensor<32x1152x7x7xf32>
    %v3158 = stablehlo.subtract %v3151, %v3157 : tensor<32x1152x7x7xf32>
    %v3159 = stablehlo.multiply %v3158, %v3158 : tensor<32x1152x7x7xf32>
    %v3160 = stablehlo.reduce(%v3159 init: %v3152) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3161 = stablehlo.broadcast_in_dim %v3160, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v3162 = stablehlo.divide %v3161, %v3153 : tensor<32x1152x7x7xf32>
    %v3163 = stablehlo.add %v3162, %v3154 : tensor<32x1152x7x7xf32>
    %v3164 = stablehlo.rsqrt %v3163 : tensor<32x1152x7x7xf32>
    %v3165 = stablehlo.multiply %v3158, %v3164 : tensor<32x1152x7x7xf32>
    %v3166 = stablehlo.broadcast_in_dim %b13eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v3167 = stablehlo.reshape %v3150 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3168 = stablehlo.multiply %v3166, %v3167 : tensor<32x1152x7x7xf32>
    %v3169 = stablehlo.reduce(%v3168 init: %v3152) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3170 = stablehlo.broadcast_in_dim %v3169, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v3171 = stablehlo.multiply %v3165, %v3168 : tensor<32x1152x7x7xf32>
    %v3172 = stablehlo.reduce(%v3171 init: %v3152) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3173 = stablehlo.broadcast_in_dim %v3172, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v3174 = stablehlo.multiply %v3168, %v3153 : tensor<32x1152x7x7xf32>
    %v3175 = stablehlo.subtract %v3174, %v3170 : tensor<32x1152x7x7xf32>
    %v3176 = stablehlo.multiply %v3165, %v3173 : tensor<32x1152x7x7xf32>
    %v3177 = stablehlo.subtract %v3175, %v3176 : tensor<32x1152x7x7xf32>
    %v3178 = stablehlo.divide %v3164, %v3153 : tensor<32x1152x7x7xf32>
    %v3179 = stablehlo.multiply %v3178, %v3177 : tensor<32x1152x7x7xf32>
    %v3180 = stablehlo.reshape %v3179 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v3181 = stablehlo.reshape %v3180 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3182 = stablehlo.reverse %b13eW, dims = [2, 3] : tensor<1152x192x1x1xf32>
    %v3183 = stablehlo.transpose %v3182, dims = [1, 0, 2, 3] : (tensor<1152x192x1x1xf32>) -> tensor<192x1152x1x1xf32>
    %v3184 = stablehlo.convolution(%v3181, %v3183)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v3185 = stablehlo.reshape %v3184 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v3186 = stablehlo.reshape %v1384 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3187 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3188 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v3189 = stablehlo.reduce(%v3186 init: %v3187) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3190 = stablehlo.broadcast_in_dim %v3189, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v3191 = stablehlo.divide %v3190, %v3188 : tensor<32x1152x7x7xf32>
    %v3192 = stablehlo.subtract %v3186, %v3191 : tensor<32x1152x7x7xf32>
    %v3193 = stablehlo.multiply %v3192, %v3192 : tensor<32x1152x7x7xf32>
    %v3194 = stablehlo.reduce(%v3193 init: %v3187) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3195 = stablehlo.broadcast_in_dim %v3194, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v3196 = stablehlo.divide %v3195, %v3188 : tensor<32x1152x7x7xf32>
    %v3197 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v3198 = stablehlo.add %v3196, %v3197 : tensor<32x1152x7x7xf32>
    %v3199 = stablehlo.rsqrt %v3198 : tensor<32x1152x7x7xf32>
    %v3200 = stablehlo.multiply %v3192, %v3199 : tensor<32x1152x7x7xf32>
    %v3201 = stablehlo.reshape %v3150 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3202 = stablehlo.multiply %v3201, %v3200 : tensor<32x1152x7x7xf32>
    %v3203 = stablehlo.reduce(%v3202 init: %v3187) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3204 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v3205 = stablehlo.multiply %v3203, %v3204 : tensor<1152xf32>
    %v3206 = stablehlo.subtract %b13eg, %v3205 : tensor<1152xf32>
    %v3207 = stablehlo.reshape %v3150 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3208 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3209 = stablehlo.reduce(%v3207 init: %v3208) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v3210 = stablehlo.constant dense<0.05> : tensor<1152xf32>
    %v3211 = stablehlo.multiply %v3209, %v3210 : tensor<1152xf32>
    %v3212 = stablehlo.subtract %b13ebt, %v3211 : tensor<1152xf32>
    %v3213 = stablehlo.reshape %v1379 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v3214 = stablehlo.reshape %v3180 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v3215 = stablehlo.transpose %v3213, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %v3216 = stablehlo.transpose %v3214, dims = [1, 0, 2, 3] : (tensor<32x1152x7x7xf32>) -> tensor<1152x32x7x7xf32>
    %v3217 = stablehlo.convolution(%v3215, %v3216)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x7x7xf32>, tensor<1152x32x7x7xf32>) -> tensor<192x1152x1x1xf32>
    %v3218 = stablehlo.transpose %v3217, dims = [1, 0, 2, 3] : (tensor<192x1152x1x1xf32>) -> tensor<1152x192x1x1xf32>
    %v3219 = stablehlo.constant dense<0.05> : tensor<1152x192x1x1xf32>
    %v3220 = stablehlo.multiply %v3218, %v3219 : tensor<1152x192x1x1xf32>
    %v3221 = stablehlo.subtract %b13eW, %v3220 : tensor<1152x192x1x1xf32>
    %v3222 = stablehlo.reshape %v3185 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v3223 = stablehlo.reshape %v2914 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v3224 = stablehlo.add %v3222, %v3223 : tensor<32x192x7x7xf32>
    %v3225 = stablehlo.reshape %v3224 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v3226 = stablehlo.reshape %v1359 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v3227 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3228 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v3229 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v3230 = stablehlo.reduce(%v3226 init: %v3227) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v3231 = stablehlo.broadcast_in_dim %v3230, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v3232 = stablehlo.divide %v3231, %v3228 : tensor<32x192x7x7xf32>
    %v3233 = stablehlo.subtract %v3226, %v3232 : tensor<32x192x7x7xf32>
    %v3234 = stablehlo.multiply %v3233, %v3233 : tensor<32x192x7x7xf32>
    %v3235 = stablehlo.reduce(%v3234 init: %v3227) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v3236 = stablehlo.broadcast_in_dim %v3235, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v3237 = stablehlo.divide %v3236, %v3228 : tensor<32x192x7x7xf32>
    %v3238 = stablehlo.add %v3237, %v3229 : tensor<32x192x7x7xf32>
    %v3239 = stablehlo.rsqrt %v3238 : tensor<32x192x7x7xf32>
    %v3240 = stablehlo.multiply %v3233, %v3239 : tensor<32x192x7x7xf32>
    %v3241 = stablehlo.broadcast_in_dim %b12pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v3242 = stablehlo.reshape %v3225 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v3243 = stablehlo.multiply %v3241, %v3242 : tensor<32x192x7x7xf32>
    %v3244 = stablehlo.reduce(%v3243 init: %v3227) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v3245 = stablehlo.broadcast_in_dim %v3244, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v3246 = stablehlo.multiply %v3240, %v3243 : tensor<32x192x7x7xf32>
    %v3247 = stablehlo.reduce(%v3246 init: %v3227) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v3248 = stablehlo.broadcast_in_dim %v3247, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v3249 = stablehlo.multiply %v3243, %v3228 : tensor<32x192x7x7xf32>
    %v3250 = stablehlo.subtract %v3249, %v3245 : tensor<32x192x7x7xf32>
    %v3251 = stablehlo.multiply %v3240, %v3248 : tensor<32x192x7x7xf32>
    %v3252 = stablehlo.subtract %v3250, %v3251 : tensor<32x192x7x7xf32>
    %v3253 = stablehlo.divide %v3239, %v3228 : tensor<32x192x7x7xf32>
    %v3254 = stablehlo.multiply %v3253, %v3252 : tensor<32x192x7x7xf32>
    %v3255 = stablehlo.reshape %v3254 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v3256 = stablehlo.reshape %v3255 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v3257 = stablehlo.reverse %b12pW, dims = [2, 3] : tensor<192x672x1x1xf32>
    %v3258 = stablehlo.transpose %v3257, dims = [1, 0, 2, 3] : (tensor<192x672x1x1xf32>) -> tensor<672x192x1x1xf32>
    %v3259 = stablehlo.convolution(%v3256, %v3258)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<672x192x1x1xf32>) -> tensor<32x672x7x7xf32>
    %v3260 = stablehlo.reshape %v3259 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v3261 = stablehlo.reshape %v1359 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v3262 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3263 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v3264 = stablehlo.reduce(%v3261 init: %v3262) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v3265 = stablehlo.broadcast_in_dim %v3264, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v3266 = stablehlo.divide %v3265, %v3263 : tensor<32x192x7x7xf32>
    %v3267 = stablehlo.subtract %v3261, %v3266 : tensor<32x192x7x7xf32>
    %v3268 = stablehlo.multiply %v3267, %v3267 : tensor<32x192x7x7xf32>
    %v3269 = stablehlo.reduce(%v3268 init: %v3262) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v3270 = stablehlo.broadcast_in_dim %v3269, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v3271 = stablehlo.divide %v3270, %v3263 : tensor<32x192x7x7xf32>
    %v3272 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v3273 = stablehlo.add %v3271, %v3272 : tensor<32x192x7x7xf32>
    %v3274 = stablehlo.rsqrt %v3273 : tensor<32x192x7x7xf32>
    %v3275 = stablehlo.multiply %v3267, %v3274 : tensor<32x192x7x7xf32>
    %v3276 = stablehlo.reshape %v3225 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v3277 = stablehlo.multiply %v3276, %v3275 : tensor<32x192x7x7xf32>
    %v3278 = stablehlo.reduce(%v3277 init: %v3262) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v3279 = stablehlo.constant dense<0.05> : tensor<192xf32>
    %v3280 = stablehlo.multiply %v3278, %v3279 : tensor<192xf32>
    %v3281 = stablehlo.subtract %b12pg, %v3280 : tensor<192xf32>
    %v3282 = stablehlo.reshape %v3225 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v3283 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3284 = stablehlo.reduce(%v3282 init: %v3283) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v3285 = stablehlo.constant dense<0.05> : tensor<192xf32>
    %v3286 = stablehlo.multiply %v3284, %v3285 : tensor<192xf32>
    %v3287 = stablehlo.subtract %b12pbt, %v3286 : tensor<192xf32>
    %v3288 = stablehlo.reshape %v1354 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3289 = stablehlo.reshape %v3255 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v3290 = stablehlo.transpose %v3288, dims = [1, 0, 2, 3] : (tensor<32x672x7x7xf32>) -> tensor<672x32x7x7xf32>
    %v3291 = stablehlo.transpose %v3289, dims = [1, 0, 2, 3] : (tensor<32x192x7x7xf32>) -> tensor<192x32x7x7xf32>
    %v3292 = stablehlo.convolution(%v3290, %v3291)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<672x32x7x7xf32>, tensor<192x32x7x7xf32>) -> tensor<672x192x1x1xf32>
    %v3293 = stablehlo.transpose %v3292, dims = [1, 0, 2, 3] : (tensor<672x192x1x1xf32>) -> tensor<192x672x1x1xf32>
    %v3294 = stablehlo.constant dense<0.05> : tensor<192x672x1x1xf32>
    %v3295 = stablehlo.multiply %v3293, %v3294 : tensor<192x672x1x1xf32>
    %v3296 = stablehlo.subtract %b12pW, %v3295 : tensor<192x672x1x1xf32>
    %v3297 = stablehlo.reshape %v1324 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3298 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3299 = stablehlo.reduce(%v3297 init: %v3298) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3300 = stablehlo.constant dense<49.0> : tensor<32x672xf32>
    %v3301 = stablehlo.divide %v3299, %v3300 : tensor<32x672xf32>
    %v3302 = stablehlo.dot_general %v3301, %b12zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v3303 = stablehlo.broadcast_in_dim %b12zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v3304 = stablehlo.add %v3302, %v3303 : tensor<32x28xf32>
    %v3305 = stablehlo.logistic %v3304 : tensor<32x28xf32>
    %v3306 = stablehlo.multiply %v3304, %v3305 : tensor<32x28xf32>
    %v3307 = stablehlo.dot_general %v3306, %b12zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v3308 = stablehlo.broadcast_in_dim %b12zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v3309 = stablehlo.add %v3307, %v3308 : tensor<32x672xf32>
    %v3310 = stablehlo.logistic %v3309 : tensor<32x672xf32>
    %v3311 = stablehlo.reshape %v3260 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3312 = stablehlo.broadcast_in_dim %v3310, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x7x7xf32>
    %v3313 = stablehlo.multiply %v3312, %v3311 : tensor<32x672x7x7xf32>
    %v3314 = stablehlo.multiply %v3297, %v3311 : tensor<32x672x7x7xf32>
    %v3315 = stablehlo.reduce(%v3314 init: %v3298) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3316 = stablehlo.constant dense<1.0> : tensor<32x672xf32>
    %v3317 = stablehlo.subtract %v3316, %v3310 : tensor<32x672xf32>
    %v3318 = stablehlo.multiply %v3310, %v3317 : tensor<32x672xf32>
    %v3319 = stablehlo.multiply %v3315, %v3318 : tensor<32x672xf32>
    %v3320 = stablehlo.dot_general %v3319, %b12zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<28x672xf32>) -> tensor<32x28xf32>
    %v3321 = stablehlo.logistic %v3304 : tensor<32x28xf32>
    %v3322 = stablehlo.constant dense<1.0> : tensor<32x28xf32>
    %v3323 = stablehlo.subtract %v3322, %v3321 : tensor<32x28xf32>
    %v3324 = stablehlo.multiply %v3304, %v3323 : tensor<32x28xf32>
    %v3325 = stablehlo.add %v3322, %v3324 : tensor<32x28xf32>
    %v3326 = stablehlo.multiply %v3321, %v3325 : tensor<32x28xf32>
    %v3327 = stablehlo.multiply %v3320, %v3326 : tensor<32x28xf32>
    %v3328 = stablehlo.dot_general %v3327, %b12zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<672x28xf32>) -> tensor<32x672xf32>
    %v3329 = stablehlo.constant dense<49.0> : tensor<32x672xf32>
    %v3330 = stablehlo.divide %v3328, %v3329 : tensor<32x672xf32>
    %v3331 = stablehlo.broadcast_in_dim %v3330, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x7x7xf32>
    %v3332 = stablehlo.add %v3313, %v3331 : tensor<32x672x7x7xf32>
    %v3333 = stablehlo.reshape %v3332 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v3334 = stablehlo.reshape %v1324 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3335 = stablehlo.reshape %v3260 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3336 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3337 = stablehlo.multiply %v3334, %v3335 : tensor<32x672x7x7xf32>
    %v3338 = stablehlo.reduce(%v3337 init: %v3336) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3339 = stablehlo.logistic %v1337 : tensor<32x672xf32>
    %v3340 = stablehlo.constant dense<1.0> : tensor<32x672xf32>
    %v3341 = stablehlo.subtract %v3340, %v3339 : tensor<32x672xf32>
    %v3342 = stablehlo.multiply %v3339, %v3341 : tensor<32x672xf32>
    %v3343 = stablehlo.multiply %v3338, %v3342 : tensor<32x672xf32>
    %v3344 = stablehlo.dot_general %v1334, %v3343, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<32x672xf32>) -> tensor<28x672xf32>
    %v3345 = stablehlo.constant dense<0.05> : tensor<28x672xf32>
    %v3346 = stablehlo.multiply %v3344, %v3345 : tensor<28x672xf32>
    %v3347 = stablehlo.subtract %b12zW2, %v3346 : tensor<28x672xf32>
    %v3348 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3349 = stablehlo.reduce(%v3343 init: %v3348) applies stablehlo.add across dimensions = [0] : (tensor<32x672xf32>, tensor<f32>) -> tensor<672xf32>
    %v3350 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3351 = stablehlo.multiply %v3349, %v3350 : tensor<672xf32>
    %v3352 = stablehlo.subtract %b12zb2, %v3351 : tensor<672xf32>
    %v3353 = stablehlo.reshape %v3343 : (tensor<32x672xf32>) -> tensor<32x1x672xf32>
    %v3354 = stablehlo.dot_general %v3353, %b12zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x672xf32>, tensor<28x672xf32>) -> tensor<32x1x28xf32>
    %v3355 = stablehlo.reshape %v3354 : (tensor<32x1x28xf32>) -> tensor<32x28xf32>
    %v3356 = stablehlo.logistic %v1332 : tensor<32x28xf32>
    %v3357 = stablehlo.constant dense<1.0> : tensor<32x28xf32>
    %v3358 = stablehlo.subtract %v3357, %v3356 : tensor<32x28xf32>
    %v3359 = stablehlo.multiply %v1332, %v3358 : tensor<32x28xf32>
    %v3360 = stablehlo.add %v3357, %v3359 : tensor<32x28xf32>
    %v3361 = stablehlo.multiply %v3356, %v3360 : tensor<32x28xf32>
    %v3362 = stablehlo.multiply %v3355, %v3361 : tensor<32x28xf32>
    %v3363 = stablehlo.dot_general %v1329, %v3362, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<32x28xf32>) -> tensor<672x28xf32>
    %v3364 = stablehlo.constant dense<0.05> : tensor<672x28xf32>
    %v3365 = stablehlo.multiply %v3363, %v3364 : tensor<672x28xf32>
    %v3366 = stablehlo.subtract %b12zW1, %v3365 : tensor<672x28xf32>
    %v3367 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3368 = stablehlo.reduce(%v3362 init: %v3367) applies stablehlo.add across dimensions = [0] : (tensor<32x28xf32>, tensor<f32>) -> tensor<28xf32>
    %v3369 = stablehlo.constant dense<0.05> : tensor<28xf32>
    %v3370 = stablehlo.multiply %v3368, %v3369 : tensor<28xf32>
    %v3371 = stablehlo.subtract %b12zb1, %v3370 : tensor<28xf32>
    %v3372 = stablehlo.reshape %v3333 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3373 = stablehlo.reshape %v1320 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3374 = stablehlo.logistic %v3373 : tensor<32x672x7x7xf32>
    %v3375 = stablehlo.constant dense<1.0> : tensor<32x672x7x7xf32>
    %v3376 = stablehlo.subtract %v3375, %v3374 : tensor<32x672x7x7xf32>
    %v3377 = stablehlo.multiply %v3373, %v3376 : tensor<32x672x7x7xf32>
    %v3378 = stablehlo.add %v3375, %v3377 : tensor<32x672x7x7xf32>
    %v3379 = stablehlo.multiply %v3374, %v3378 : tensor<32x672x7x7xf32>
    %v3380 = stablehlo.multiply %v3372, %v3379 : tensor<32x672x7x7xf32>
    %v3381 = stablehlo.reshape %v3380 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v3382 = stablehlo.reshape %v1300 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3383 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3384 = stablehlo.constant dense<1568.0> : tensor<32x672x7x7xf32>
    %v3385 = stablehlo.constant dense<1.0e-5> : tensor<32x672x7x7xf32>
    %v3386 = stablehlo.reduce(%v3382 init: %v3383) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v3387 = stablehlo.broadcast_in_dim %v3386, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v3388 = stablehlo.divide %v3387, %v3384 : tensor<32x672x7x7xf32>
    %v3389 = stablehlo.subtract %v3382, %v3388 : tensor<32x672x7x7xf32>
    %v3390 = stablehlo.multiply %v3389, %v3389 : tensor<32x672x7x7xf32>
    %v3391 = stablehlo.reduce(%v3390 init: %v3383) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v3392 = stablehlo.broadcast_in_dim %v3391, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v3393 = stablehlo.divide %v3392, %v3384 : tensor<32x672x7x7xf32>
    %v3394 = stablehlo.add %v3393, %v3385 : tensor<32x672x7x7xf32>
    %v3395 = stablehlo.rsqrt %v3394 : tensor<32x672x7x7xf32>
    %v3396 = stablehlo.multiply %v3389, %v3395 : tensor<32x672x7x7xf32>
    %v3397 = stablehlo.broadcast_in_dim %b12dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v3398 = stablehlo.reshape %v3381 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3399 = stablehlo.multiply %v3397, %v3398 : tensor<32x672x7x7xf32>
    %v3400 = stablehlo.reduce(%v3399 init: %v3383) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v3401 = stablehlo.broadcast_in_dim %v3400, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v3402 = stablehlo.multiply %v3396, %v3399 : tensor<32x672x7x7xf32>
    %v3403 = stablehlo.reduce(%v3402 init: %v3383) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v3404 = stablehlo.broadcast_in_dim %v3403, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v3405 = stablehlo.multiply %v3399, %v3384 : tensor<32x672x7x7xf32>
    %v3406 = stablehlo.subtract %v3405, %v3401 : tensor<32x672x7x7xf32>
    %v3407 = stablehlo.multiply %v3396, %v3404 : tensor<32x672x7x7xf32>
    %v3408 = stablehlo.subtract %v3406, %v3407 : tensor<32x672x7x7xf32>
    %v3409 = stablehlo.divide %v3395, %v3384 : tensor<32x672x7x7xf32>
    %v3410 = stablehlo.multiply %v3409, %v3408 : tensor<32x672x7x7xf32>
    %v3411 = stablehlo.reshape %v3410 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v3412 = stablehlo.reshape %v3411 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3413 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3414 = stablehlo.pad %v3412, %v3413, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672x14x14xf32>
    %v3415 = stablehlo.reverse %b12dW, dims = [2, 3] : tensor<672x1x5x5xf32>
    %v3416 = stablehlo.convolution(%v3414, %v3415)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %v3417 = stablehlo.reshape %v3416 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3418 = stablehlo.reshape %v1300 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3419 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3420 = stablehlo.constant dense<1568.0> : tensor<32x672x7x7xf32>
    %v3421 = stablehlo.reduce(%v3418 init: %v3419) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v3422 = stablehlo.broadcast_in_dim %v3421, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v3423 = stablehlo.divide %v3422, %v3420 : tensor<32x672x7x7xf32>
    %v3424 = stablehlo.subtract %v3418, %v3423 : tensor<32x672x7x7xf32>
    %v3425 = stablehlo.multiply %v3424, %v3424 : tensor<32x672x7x7xf32>
    %v3426 = stablehlo.reduce(%v3425 init: %v3419) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v3427 = stablehlo.broadcast_in_dim %v3426, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v3428 = stablehlo.divide %v3427, %v3420 : tensor<32x672x7x7xf32>
    %v3429 = stablehlo.constant dense<1.0e-5> : tensor<32x672x7x7xf32>
    %v3430 = stablehlo.add %v3428, %v3429 : tensor<32x672x7x7xf32>
    %v3431 = stablehlo.rsqrt %v3430 : tensor<32x672x7x7xf32>
    %v3432 = stablehlo.multiply %v3424, %v3431 : tensor<32x672x7x7xf32>
    %v3433 = stablehlo.reshape %v3381 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3434 = stablehlo.multiply %v3433, %v3432 : tensor<32x672x7x7xf32>
    %v3435 = stablehlo.reduce(%v3434 init: %v3419) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v3436 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3437 = stablehlo.multiply %v3435, %v3436 : tensor<672xf32>
    %v3438 = stablehlo.subtract %b12dg, %v3437 : tensor<672xf32>
    %v3439 = stablehlo.reshape %v3381 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3440 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3441 = stablehlo.reduce(%v3439 init: %v3440) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v3442 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3443 = stablehlo.multiply %v3441, %v3442 : tensor<672xf32>
    %v3444 = stablehlo.subtract %b12dbt, %v3443 : tensor<672xf32>
    %v3445 = stablehlo.reshape %v1295 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3446 = stablehlo.reshape %v3411 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3447 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3448 = stablehlo.pad %v3446, %v3447, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672x14x14xf32>
    %v3449 = stablehlo.transpose %v3445, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3450 = stablehlo.transpose %v3448, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3451 = stablehlo.convolution(%v3449, %v3450)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 672 : i64, feature_group_count = 1 : i64} : (tensor<672x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<1x672x5x5xf32>
    %v3452 = stablehlo.reshape %v3451 : (tensor<1x672x5x5xf32>) -> tensor<672x1x5x5xf32>
    %v3453 = stablehlo.constant dense<0.05> : tensor<672x1x5x5xf32>
    %v3454 = stablehlo.multiply %v3452, %v3453 : tensor<672x1x5x5xf32>
    %v3455 = stablehlo.subtract %b12dW, %v3454 : tensor<672x1x5x5xf32>
    %v3456 = stablehlo.reshape %v3417 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3457 = stablehlo.reshape %v1291 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3458 = stablehlo.logistic %v3457 : tensor<32x672x14x14xf32>
    %v3459 = stablehlo.constant dense<1.0> : tensor<32x672x14x14xf32>
    %v3460 = stablehlo.subtract %v3459, %v3458 : tensor<32x672x14x14xf32>
    %v3461 = stablehlo.multiply %v3457, %v3460 : tensor<32x672x14x14xf32>
    %v3462 = stablehlo.add %v3459, %v3461 : tensor<32x672x14x14xf32>
    %v3463 = stablehlo.multiply %v3458, %v3462 : tensor<32x672x14x14xf32>
    %v3464 = stablehlo.multiply %v3456, %v3463 : tensor<32x672x14x14xf32>
    %v3465 = stablehlo.reshape %v3464 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3466 = stablehlo.reshape %v1271 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3467 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3468 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3469 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3470 = stablehlo.reduce(%v3466 init: %v3467) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3471 = stablehlo.broadcast_in_dim %v3470, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3472 = stablehlo.divide %v3471, %v3468 : tensor<32x672x14x14xf32>
    %v3473 = stablehlo.subtract %v3466, %v3472 : tensor<32x672x14x14xf32>
    %v3474 = stablehlo.multiply %v3473, %v3473 : tensor<32x672x14x14xf32>
    %v3475 = stablehlo.reduce(%v3474 init: %v3467) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3476 = stablehlo.broadcast_in_dim %v3475, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3477 = stablehlo.divide %v3476, %v3468 : tensor<32x672x14x14xf32>
    %v3478 = stablehlo.add %v3477, %v3469 : tensor<32x672x14x14xf32>
    %v3479 = stablehlo.rsqrt %v3478 : tensor<32x672x14x14xf32>
    %v3480 = stablehlo.multiply %v3473, %v3479 : tensor<32x672x14x14xf32>
    %v3481 = stablehlo.broadcast_in_dim %b12eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3482 = stablehlo.reshape %v3465 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3483 = stablehlo.multiply %v3481, %v3482 : tensor<32x672x14x14xf32>
    %v3484 = stablehlo.reduce(%v3483 init: %v3467) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3485 = stablehlo.broadcast_in_dim %v3484, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3486 = stablehlo.multiply %v3480, %v3483 : tensor<32x672x14x14xf32>
    %v3487 = stablehlo.reduce(%v3486 init: %v3467) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3488 = stablehlo.broadcast_in_dim %v3487, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3489 = stablehlo.multiply %v3483, %v3468 : tensor<32x672x14x14xf32>
    %v3490 = stablehlo.subtract %v3489, %v3485 : tensor<32x672x14x14xf32>
    %v3491 = stablehlo.multiply %v3480, %v3488 : tensor<32x672x14x14xf32>
    %v3492 = stablehlo.subtract %v3490, %v3491 : tensor<32x672x14x14xf32>
    %v3493 = stablehlo.divide %v3479, %v3468 : tensor<32x672x14x14xf32>
    %v3494 = stablehlo.multiply %v3493, %v3492 : tensor<32x672x14x14xf32>
    %v3495 = stablehlo.reshape %v3494 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3496 = stablehlo.reshape %v3495 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3497 = stablehlo.reverse %b12eW, dims = [2, 3] : tensor<672x112x1x1xf32>
    %v3498 = stablehlo.transpose %v3497, dims = [1, 0, 2, 3] : (tensor<672x112x1x1xf32>) -> tensor<112x672x1x1xf32>
    %v3499 = stablehlo.convolution(%v3496, %v3498)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v3500 = stablehlo.reshape %v3499 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v3501 = stablehlo.reshape %v1271 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3502 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3503 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3504 = stablehlo.reduce(%v3501 init: %v3502) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3505 = stablehlo.broadcast_in_dim %v3504, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3506 = stablehlo.divide %v3505, %v3503 : tensor<32x672x14x14xf32>
    %v3507 = stablehlo.subtract %v3501, %v3506 : tensor<32x672x14x14xf32>
    %v3508 = stablehlo.multiply %v3507, %v3507 : tensor<32x672x14x14xf32>
    %v3509 = stablehlo.reduce(%v3508 init: %v3502) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3510 = stablehlo.broadcast_in_dim %v3509, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3511 = stablehlo.divide %v3510, %v3503 : tensor<32x672x14x14xf32>
    %v3512 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3513 = stablehlo.add %v3511, %v3512 : tensor<32x672x14x14xf32>
    %v3514 = stablehlo.rsqrt %v3513 : tensor<32x672x14x14xf32>
    %v3515 = stablehlo.multiply %v3507, %v3514 : tensor<32x672x14x14xf32>
    %v3516 = stablehlo.reshape %v3465 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3517 = stablehlo.multiply %v3516, %v3515 : tensor<32x672x14x14xf32>
    %v3518 = stablehlo.reduce(%v3517 init: %v3502) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3519 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3520 = stablehlo.multiply %v3518, %v3519 : tensor<672xf32>
    %v3521 = stablehlo.subtract %b12eg, %v3520 : tensor<672xf32>
    %v3522 = stablehlo.reshape %v3465 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3523 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3524 = stablehlo.reduce(%v3522 init: %v3523) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3525 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3526 = stablehlo.multiply %v3524, %v3525 : tensor<672xf32>
    %v3527 = stablehlo.subtract %b12ebt, %v3526 : tensor<672xf32>
    %v3528 = stablehlo.reshape %v1266 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3529 = stablehlo.reshape %v3495 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3530 = stablehlo.transpose %v3528, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %v3531 = stablehlo.transpose %v3529, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3532 = stablehlo.convolution(%v3530, %v3531)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<112x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<112x672x1x1xf32>
    %v3533 = stablehlo.transpose %v3532, dims = [1, 0, 2, 3] : (tensor<112x672x1x1xf32>) -> tensor<672x112x1x1xf32>
    %v3534 = stablehlo.constant dense<0.05> : tensor<672x112x1x1xf32>
    %v3535 = stablehlo.multiply %v3533, %v3534 : tensor<672x112x1x1xf32>
    %v3536 = stablehlo.subtract %b12eW, %v3535 : tensor<672x112x1x1xf32>
    %v3537 = stablehlo.reshape %v1242 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3538 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3539 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v3540 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v3541 = stablehlo.reduce(%v3537 init: %v3538) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3542 = stablehlo.broadcast_in_dim %v3541, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3543 = stablehlo.divide %v3542, %v3539 : tensor<32x112x14x14xf32>
    %v3544 = stablehlo.subtract %v3537, %v3543 : tensor<32x112x14x14xf32>
    %v3545 = stablehlo.multiply %v3544, %v3544 : tensor<32x112x14x14xf32>
    %v3546 = stablehlo.reduce(%v3545 init: %v3538) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3547 = stablehlo.broadcast_in_dim %v3546, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3548 = stablehlo.divide %v3547, %v3539 : tensor<32x112x14x14xf32>
    %v3549 = stablehlo.add %v3548, %v3540 : tensor<32x112x14x14xf32>
    %v3550 = stablehlo.rsqrt %v3549 : tensor<32x112x14x14xf32>
    %v3551 = stablehlo.multiply %v3544, %v3550 : tensor<32x112x14x14xf32>
    %v3552 = stablehlo.broadcast_in_dim %b11pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3553 = stablehlo.reshape %v3500 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3554 = stablehlo.multiply %v3552, %v3553 : tensor<32x112x14x14xf32>
    %v3555 = stablehlo.reduce(%v3554 init: %v3538) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3556 = stablehlo.broadcast_in_dim %v3555, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3557 = stablehlo.multiply %v3551, %v3554 : tensor<32x112x14x14xf32>
    %v3558 = stablehlo.reduce(%v3557 init: %v3538) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3559 = stablehlo.broadcast_in_dim %v3558, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3560 = stablehlo.multiply %v3554, %v3539 : tensor<32x112x14x14xf32>
    %v3561 = stablehlo.subtract %v3560, %v3556 : tensor<32x112x14x14xf32>
    %v3562 = stablehlo.multiply %v3551, %v3559 : tensor<32x112x14x14xf32>
    %v3563 = stablehlo.subtract %v3561, %v3562 : tensor<32x112x14x14xf32>
    %v3564 = stablehlo.divide %v3550, %v3539 : tensor<32x112x14x14xf32>
    %v3565 = stablehlo.multiply %v3564, %v3563 : tensor<32x112x14x14xf32>
    %v3566 = stablehlo.reshape %v3565 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v3567 = stablehlo.reshape %v3566 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3568 = stablehlo.reverse %b11pW, dims = [2, 3] : tensor<112x672x1x1xf32>
    %v3569 = stablehlo.transpose %v3568, dims = [1, 0, 2, 3] : (tensor<112x672x1x1xf32>) -> tensor<672x112x1x1xf32>
    %v3570 = stablehlo.convolution(%v3567, %v3569)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v3571 = stablehlo.reshape %v3570 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3572 = stablehlo.reshape %v1242 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3573 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3574 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v3575 = stablehlo.reduce(%v3572 init: %v3573) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3576 = stablehlo.broadcast_in_dim %v3575, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3577 = stablehlo.divide %v3576, %v3574 : tensor<32x112x14x14xf32>
    %v3578 = stablehlo.subtract %v3572, %v3577 : tensor<32x112x14x14xf32>
    %v3579 = stablehlo.multiply %v3578, %v3578 : tensor<32x112x14x14xf32>
    %v3580 = stablehlo.reduce(%v3579 init: %v3573) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3581 = stablehlo.broadcast_in_dim %v3580, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3582 = stablehlo.divide %v3581, %v3574 : tensor<32x112x14x14xf32>
    %v3583 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v3584 = stablehlo.add %v3582, %v3583 : tensor<32x112x14x14xf32>
    %v3585 = stablehlo.rsqrt %v3584 : tensor<32x112x14x14xf32>
    %v3586 = stablehlo.multiply %v3578, %v3585 : tensor<32x112x14x14xf32>
    %v3587 = stablehlo.reshape %v3500 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3588 = stablehlo.multiply %v3587, %v3586 : tensor<32x112x14x14xf32>
    %v3589 = stablehlo.reduce(%v3588 init: %v3573) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3590 = stablehlo.constant dense<0.05> : tensor<112xf32>
    %v3591 = stablehlo.multiply %v3589, %v3590 : tensor<112xf32>
    %v3592 = stablehlo.subtract %b11pg, %v3591 : tensor<112xf32>
    %v3593 = stablehlo.reshape %v3500 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3594 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3595 = stablehlo.reduce(%v3593 init: %v3594) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3596 = stablehlo.constant dense<0.05> : tensor<112xf32>
    %v3597 = stablehlo.multiply %v3595, %v3596 : tensor<112xf32>
    %v3598 = stablehlo.subtract %b11pbt, %v3597 : tensor<112xf32>
    %v3599 = stablehlo.reshape %v1237 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3600 = stablehlo.reshape %v3566 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3601 = stablehlo.transpose %v3599, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3602 = stablehlo.transpose %v3600, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %v3603 = stablehlo.convolution(%v3601, %v3602)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<672x32x14x14xf32>, tensor<112x32x14x14xf32>) -> tensor<672x112x1x1xf32>
    %v3604 = stablehlo.transpose %v3603, dims = [1, 0, 2, 3] : (tensor<672x112x1x1xf32>) -> tensor<112x672x1x1xf32>
    %v3605 = stablehlo.constant dense<0.05> : tensor<112x672x1x1xf32>
    %v3606 = stablehlo.multiply %v3604, %v3605 : tensor<112x672x1x1xf32>
    %v3607 = stablehlo.subtract %b11pW, %v3606 : tensor<112x672x1x1xf32>
    %v3608 = stablehlo.reshape %v1207 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3609 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3610 = stablehlo.reduce(%v3608 init: %v3609) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3611 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v3612 = stablehlo.divide %v3610, %v3611 : tensor<32x672xf32>
    %v3613 = stablehlo.dot_general %v3612, %b11zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v3614 = stablehlo.broadcast_in_dim %b11zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v3615 = stablehlo.add %v3613, %v3614 : tensor<32x28xf32>
    %v3616 = stablehlo.logistic %v3615 : tensor<32x28xf32>
    %v3617 = stablehlo.multiply %v3615, %v3616 : tensor<32x28xf32>
    %v3618 = stablehlo.dot_general %v3617, %b11zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v3619 = stablehlo.broadcast_in_dim %b11zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v3620 = stablehlo.add %v3618, %v3619 : tensor<32x672xf32>
    %v3621 = stablehlo.logistic %v3620 : tensor<32x672xf32>
    %v3622 = stablehlo.reshape %v3571 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3623 = stablehlo.broadcast_in_dim %v3621, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v3624 = stablehlo.multiply %v3623, %v3622 : tensor<32x672x14x14xf32>
    %v3625 = stablehlo.multiply %v3608, %v3622 : tensor<32x672x14x14xf32>
    %v3626 = stablehlo.reduce(%v3625 init: %v3609) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3627 = stablehlo.constant dense<1.0> : tensor<32x672xf32>
    %v3628 = stablehlo.subtract %v3627, %v3621 : tensor<32x672xf32>
    %v3629 = stablehlo.multiply %v3621, %v3628 : tensor<32x672xf32>
    %v3630 = stablehlo.multiply %v3626, %v3629 : tensor<32x672xf32>
    %v3631 = stablehlo.dot_general %v3630, %b11zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<28x672xf32>) -> tensor<32x28xf32>
    %v3632 = stablehlo.logistic %v3615 : tensor<32x28xf32>
    %v3633 = stablehlo.constant dense<1.0> : tensor<32x28xf32>
    %v3634 = stablehlo.subtract %v3633, %v3632 : tensor<32x28xf32>
    %v3635 = stablehlo.multiply %v3615, %v3634 : tensor<32x28xf32>
    %v3636 = stablehlo.add %v3633, %v3635 : tensor<32x28xf32>
    %v3637 = stablehlo.multiply %v3632, %v3636 : tensor<32x28xf32>
    %v3638 = stablehlo.multiply %v3631, %v3637 : tensor<32x28xf32>
    %v3639 = stablehlo.dot_general %v3638, %b11zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<672x28xf32>) -> tensor<32x672xf32>
    %v3640 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v3641 = stablehlo.divide %v3639, %v3640 : tensor<32x672xf32>
    %v3642 = stablehlo.broadcast_in_dim %v3641, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v3643 = stablehlo.add %v3624, %v3642 : tensor<32x672x14x14xf32>
    %v3644 = stablehlo.reshape %v3643 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3645 = stablehlo.reshape %v1207 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3646 = stablehlo.reshape %v3571 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3647 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3648 = stablehlo.multiply %v3645, %v3646 : tensor<32x672x14x14xf32>
    %v3649 = stablehlo.reduce(%v3648 init: %v3647) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3650 = stablehlo.logistic %v1220 : tensor<32x672xf32>
    %v3651 = stablehlo.constant dense<1.0> : tensor<32x672xf32>
    %v3652 = stablehlo.subtract %v3651, %v3650 : tensor<32x672xf32>
    %v3653 = stablehlo.multiply %v3650, %v3652 : tensor<32x672xf32>
    %v3654 = stablehlo.multiply %v3649, %v3653 : tensor<32x672xf32>
    %v3655 = stablehlo.dot_general %v1217, %v3654, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<32x672xf32>) -> tensor<28x672xf32>
    %v3656 = stablehlo.constant dense<0.05> : tensor<28x672xf32>
    %v3657 = stablehlo.multiply %v3655, %v3656 : tensor<28x672xf32>
    %v3658 = stablehlo.subtract %b11zW2, %v3657 : tensor<28x672xf32>
    %v3659 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3660 = stablehlo.reduce(%v3654 init: %v3659) applies stablehlo.add across dimensions = [0] : (tensor<32x672xf32>, tensor<f32>) -> tensor<672xf32>
    %v3661 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3662 = stablehlo.multiply %v3660, %v3661 : tensor<672xf32>
    %v3663 = stablehlo.subtract %b11zb2, %v3662 : tensor<672xf32>
    %v3664 = stablehlo.reshape %v3654 : (tensor<32x672xf32>) -> tensor<32x1x672xf32>
    %v3665 = stablehlo.dot_general %v3664, %b11zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x672xf32>, tensor<28x672xf32>) -> tensor<32x1x28xf32>
    %v3666 = stablehlo.reshape %v3665 : (tensor<32x1x28xf32>) -> tensor<32x28xf32>
    %v3667 = stablehlo.logistic %v1215 : tensor<32x28xf32>
    %v3668 = stablehlo.constant dense<1.0> : tensor<32x28xf32>
    %v3669 = stablehlo.subtract %v3668, %v3667 : tensor<32x28xf32>
    %v3670 = stablehlo.multiply %v1215, %v3669 : tensor<32x28xf32>
    %v3671 = stablehlo.add %v3668, %v3670 : tensor<32x28xf32>
    %v3672 = stablehlo.multiply %v3667, %v3671 : tensor<32x28xf32>
    %v3673 = stablehlo.multiply %v3666, %v3672 : tensor<32x28xf32>
    %v3674 = stablehlo.dot_general %v1212, %v3673, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<32x28xf32>) -> tensor<672x28xf32>
    %v3675 = stablehlo.constant dense<0.05> : tensor<672x28xf32>
    %v3676 = stablehlo.multiply %v3674, %v3675 : tensor<672x28xf32>
    %v3677 = stablehlo.subtract %b11zW1, %v3676 : tensor<672x28xf32>
    %v3678 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3679 = stablehlo.reduce(%v3673 init: %v3678) applies stablehlo.add across dimensions = [0] : (tensor<32x28xf32>, tensor<f32>) -> tensor<28xf32>
    %v3680 = stablehlo.constant dense<0.05> : tensor<28xf32>
    %v3681 = stablehlo.multiply %v3679, %v3680 : tensor<28xf32>
    %v3682 = stablehlo.subtract %b11zb1, %v3681 : tensor<28xf32>
    %v3683 = stablehlo.reshape %v3644 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3684 = stablehlo.reshape %v1203 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3685 = stablehlo.logistic %v3684 : tensor<32x672x14x14xf32>
    %v3686 = stablehlo.constant dense<1.0> : tensor<32x672x14x14xf32>
    %v3687 = stablehlo.subtract %v3686, %v3685 : tensor<32x672x14x14xf32>
    %v3688 = stablehlo.multiply %v3684, %v3687 : tensor<32x672x14x14xf32>
    %v3689 = stablehlo.add %v3686, %v3688 : tensor<32x672x14x14xf32>
    %v3690 = stablehlo.multiply %v3685, %v3689 : tensor<32x672x14x14xf32>
    %v3691 = stablehlo.multiply %v3683, %v3690 : tensor<32x672x14x14xf32>
    %v3692 = stablehlo.reshape %v3691 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3693 = stablehlo.reshape %v1183 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3694 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3695 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3696 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3697 = stablehlo.reduce(%v3693 init: %v3694) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3698 = stablehlo.broadcast_in_dim %v3697, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3699 = stablehlo.divide %v3698, %v3695 : tensor<32x672x14x14xf32>
    %v3700 = stablehlo.subtract %v3693, %v3699 : tensor<32x672x14x14xf32>
    %v3701 = stablehlo.multiply %v3700, %v3700 : tensor<32x672x14x14xf32>
    %v3702 = stablehlo.reduce(%v3701 init: %v3694) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3703 = stablehlo.broadcast_in_dim %v3702, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3704 = stablehlo.divide %v3703, %v3695 : tensor<32x672x14x14xf32>
    %v3705 = stablehlo.add %v3704, %v3696 : tensor<32x672x14x14xf32>
    %v3706 = stablehlo.rsqrt %v3705 : tensor<32x672x14x14xf32>
    %v3707 = stablehlo.multiply %v3700, %v3706 : tensor<32x672x14x14xf32>
    %v3708 = stablehlo.broadcast_in_dim %b11dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3709 = stablehlo.reshape %v3692 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3710 = stablehlo.multiply %v3708, %v3709 : tensor<32x672x14x14xf32>
    %v3711 = stablehlo.reduce(%v3710 init: %v3694) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3712 = stablehlo.broadcast_in_dim %v3711, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3713 = stablehlo.multiply %v3707, %v3710 : tensor<32x672x14x14xf32>
    %v3714 = stablehlo.reduce(%v3713 init: %v3694) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3715 = stablehlo.broadcast_in_dim %v3714, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3716 = stablehlo.multiply %v3710, %v3695 : tensor<32x672x14x14xf32>
    %v3717 = stablehlo.subtract %v3716, %v3712 : tensor<32x672x14x14xf32>
    %v3718 = stablehlo.multiply %v3707, %v3715 : tensor<32x672x14x14xf32>
    %v3719 = stablehlo.subtract %v3717, %v3718 : tensor<32x672x14x14xf32>
    %v3720 = stablehlo.divide %v3706, %v3695 : tensor<32x672x14x14xf32>
    %v3721 = stablehlo.multiply %v3720, %v3719 : tensor<32x672x14x14xf32>
    %v3722 = stablehlo.reshape %v3721 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3723 = stablehlo.reshape %v3722 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3724 = stablehlo.reverse %b11dW, dims = [2, 3] : tensor<672x1x5x5xf32>
    %v3725 = stablehlo.convolution(%v3723, %v3724)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %v3726 = stablehlo.reshape %v3725 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3727 = stablehlo.reshape %v1183 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3728 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3729 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3730 = stablehlo.reduce(%v3727 init: %v3728) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3731 = stablehlo.broadcast_in_dim %v3730, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3732 = stablehlo.divide %v3731, %v3729 : tensor<32x672x14x14xf32>
    %v3733 = stablehlo.subtract %v3727, %v3732 : tensor<32x672x14x14xf32>
    %v3734 = stablehlo.multiply %v3733, %v3733 : tensor<32x672x14x14xf32>
    %v3735 = stablehlo.reduce(%v3734 init: %v3728) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3736 = stablehlo.broadcast_in_dim %v3735, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3737 = stablehlo.divide %v3736, %v3729 : tensor<32x672x14x14xf32>
    %v3738 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3739 = stablehlo.add %v3737, %v3738 : tensor<32x672x14x14xf32>
    %v3740 = stablehlo.rsqrt %v3739 : tensor<32x672x14x14xf32>
    %v3741 = stablehlo.multiply %v3733, %v3740 : tensor<32x672x14x14xf32>
    %v3742 = stablehlo.reshape %v3692 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3743 = stablehlo.multiply %v3742, %v3741 : tensor<32x672x14x14xf32>
    %v3744 = stablehlo.reduce(%v3743 init: %v3728) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3745 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3746 = stablehlo.multiply %v3744, %v3745 : tensor<672xf32>
    %v3747 = stablehlo.subtract %b11dg, %v3746 : tensor<672xf32>
    %v3748 = stablehlo.reshape %v3692 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3749 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3750 = stablehlo.reduce(%v3748 init: %v3749) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3751 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3752 = stablehlo.multiply %v3750, %v3751 : tensor<672xf32>
    %v3753 = stablehlo.subtract %b11dbt, %v3752 : tensor<672xf32>
    %v3754 = stablehlo.reshape %v1178 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3755 = stablehlo.reshape %v3722 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3756 = stablehlo.transpose %v3754, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3757 = stablehlo.transpose %v3755, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3758 = stablehlo.convolution(%v3756, %v3757)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 672 : i64, feature_group_count = 1 : i64} : (tensor<672x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<1x672x5x5xf32>
    %v3759 = stablehlo.reshape %v3758 : (tensor<1x672x5x5xf32>) -> tensor<672x1x5x5xf32>
    %v3760 = stablehlo.constant dense<0.05> : tensor<672x1x5x5xf32>
    %v3761 = stablehlo.multiply %v3759, %v3760 : tensor<672x1x5x5xf32>
    %v3762 = stablehlo.subtract %b11dW, %v3761 : tensor<672x1x5x5xf32>
    %v3763 = stablehlo.reshape %v3726 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3764 = stablehlo.reshape %v1174 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3765 = stablehlo.logistic %v3764 : tensor<32x672x14x14xf32>
    %v3766 = stablehlo.constant dense<1.0> : tensor<32x672x14x14xf32>
    %v3767 = stablehlo.subtract %v3766, %v3765 : tensor<32x672x14x14xf32>
    %v3768 = stablehlo.multiply %v3764, %v3767 : tensor<32x672x14x14xf32>
    %v3769 = stablehlo.add %v3766, %v3768 : tensor<32x672x14x14xf32>
    %v3770 = stablehlo.multiply %v3765, %v3769 : tensor<32x672x14x14xf32>
    %v3771 = stablehlo.multiply %v3763, %v3770 : tensor<32x672x14x14xf32>
    %v3772 = stablehlo.reshape %v3771 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3773 = stablehlo.reshape %v1154 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3774 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3775 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3776 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3777 = stablehlo.reduce(%v3773 init: %v3774) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3778 = stablehlo.broadcast_in_dim %v3777, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3779 = stablehlo.divide %v3778, %v3775 : tensor<32x672x14x14xf32>
    %v3780 = stablehlo.subtract %v3773, %v3779 : tensor<32x672x14x14xf32>
    %v3781 = stablehlo.multiply %v3780, %v3780 : tensor<32x672x14x14xf32>
    %v3782 = stablehlo.reduce(%v3781 init: %v3774) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3783 = stablehlo.broadcast_in_dim %v3782, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3784 = stablehlo.divide %v3783, %v3775 : tensor<32x672x14x14xf32>
    %v3785 = stablehlo.add %v3784, %v3776 : tensor<32x672x14x14xf32>
    %v3786 = stablehlo.rsqrt %v3785 : tensor<32x672x14x14xf32>
    %v3787 = stablehlo.multiply %v3780, %v3786 : tensor<32x672x14x14xf32>
    %v3788 = stablehlo.broadcast_in_dim %b11eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3789 = stablehlo.reshape %v3772 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3790 = stablehlo.multiply %v3788, %v3789 : tensor<32x672x14x14xf32>
    %v3791 = stablehlo.reduce(%v3790 init: %v3774) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3792 = stablehlo.broadcast_in_dim %v3791, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3793 = stablehlo.multiply %v3787, %v3790 : tensor<32x672x14x14xf32>
    %v3794 = stablehlo.reduce(%v3793 init: %v3774) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3795 = stablehlo.broadcast_in_dim %v3794, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3796 = stablehlo.multiply %v3790, %v3775 : tensor<32x672x14x14xf32>
    %v3797 = stablehlo.subtract %v3796, %v3792 : tensor<32x672x14x14xf32>
    %v3798 = stablehlo.multiply %v3787, %v3795 : tensor<32x672x14x14xf32>
    %v3799 = stablehlo.subtract %v3797, %v3798 : tensor<32x672x14x14xf32>
    %v3800 = stablehlo.divide %v3786, %v3775 : tensor<32x672x14x14xf32>
    %v3801 = stablehlo.multiply %v3800, %v3799 : tensor<32x672x14x14xf32>
    %v3802 = stablehlo.reshape %v3801 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3803 = stablehlo.reshape %v3802 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3804 = stablehlo.reverse %b11eW, dims = [2, 3] : tensor<672x112x1x1xf32>
    %v3805 = stablehlo.transpose %v3804, dims = [1, 0, 2, 3] : (tensor<672x112x1x1xf32>) -> tensor<112x672x1x1xf32>
    %v3806 = stablehlo.convolution(%v3803, %v3805)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v3807 = stablehlo.reshape %v3806 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v3808 = stablehlo.reshape %v1154 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3809 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3810 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3811 = stablehlo.reduce(%v3808 init: %v3809) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3812 = stablehlo.broadcast_in_dim %v3811, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3813 = stablehlo.divide %v3812, %v3810 : tensor<32x672x14x14xf32>
    %v3814 = stablehlo.subtract %v3808, %v3813 : tensor<32x672x14x14xf32>
    %v3815 = stablehlo.multiply %v3814, %v3814 : tensor<32x672x14x14xf32>
    %v3816 = stablehlo.reduce(%v3815 init: %v3809) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3817 = stablehlo.broadcast_in_dim %v3816, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3818 = stablehlo.divide %v3817, %v3810 : tensor<32x672x14x14xf32>
    %v3819 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3820 = stablehlo.add %v3818, %v3819 : tensor<32x672x14x14xf32>
    %v3821 = stablehlo.rsqrt %v3820 : tensor<32x672x14x14xf32>
    %v3822 = stablehlo.multiply %v3814, %v3821 : tensor<32x672x14x14xf32>
    %v3823 = stablehlo.reshape %v3772 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3824 = stablehlo.multiply %v3823, %v3822 : tensor<32x672x14x14xf32>
    %v3825 = stablehlo.reduce(%v3824 init: %v3809) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3826 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3827 = stablehlo.multiply %v3825, %v3826 : tensor<672xf32>
    %v3828 = stablehlo.subtract %b11eg, %v3827 : tensor<672xf32>
    %v3829 = stablehlo.reshape %v3772 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3830 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3831 = stablehlo.reduce(%v3829 init: %v3830) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3832 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3833 = stablehlo.multiply %v3831, %v3832 : tensor<672xf32>
    %v3834 = stablehlo.subtract %b11ebt, %v3833 : tensor<672xf32>
    %v3835 = stablehlo.reshape %v1149 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3836 = stablehlo.reshape %v3802 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3837 = stablehlo.transpose %v3835, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %v3838 = stablehlo.transpose %v3836, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3839 = stablehlo.convolution(%v3837, %v3838)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<112x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<112x672x1x1xf32>
    %v3840 = stablehlo.transpose %v3839, dims = [1, 0, 2, 3] : (tensor<112x672x1x1xf32>) -> tensor<672x112x1x1xf32>
    %v3841 = stablehlo.constant dense<0.05> : tensor<672x112x1x1xf32>
    %v3842 = stablehlo.multiply %v3840, %v3841 : tensor<672x112x1x1xf32>
    %v3843 = stablehlo.subtract %b11eW, %v3842 : tensor<672x112x1x1xf32>
    %v3844 = stablehlo.reshape %v3807 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3845 = stablehlo.reshape %v3500 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3846 = stablehlo.add %v3844, %v3845 : tensor<32x112x14x14xf32>
    %v3847 = stablehlo.reshape %v3846 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v3848 = stablehlo.reshape %v1125 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3849 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3850 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v3851 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v3852 = stablehlo.reduce(%v3848 init: %v3849) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3853 = stablehlo.broadcast_in_dim %v3852, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3854 = stablehlo.divide %v3853, %v3850 : tensor<32x112x14x14xf32>
    %v3855 = stablehlo.subtract %v3848, %v3854 : tensor<32x112x14x14xf32>
    %v3856 = stablehlo.multiply %v3855, %v3855 : tensor<32x112x14x14xf32>
    %v3857 = stablehlo.reduce(%v3856 init: %v3849) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3858 = stablehlo.broadcast_in_dim %v3857, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3859 = stablehlo.divide %v3858, %v3850 : tensor<32x112x14x14xf32>
    %v3860 = stablehlo.add %v3859, %v3851 : tensor<32x112x14x14xf32>
    %v3861 = stablehlo.rsqrt %v3860 : tensor<32x112x14x14xf32>
    %v3862 = stablehlo.multiply %v3855, %v3861 : tensor<32x112x14x14xf32>
    %v3863 = stablehlo.broadcast_in_dim %b10pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3864 = stablehlo.reshape %v3847 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3865 = stablehlo.multiply %v3863, %v3864 : tensor<32x112x14x14xf32>
    %v3866 = stablehlo.reduce(%v3865 init: %v3849) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3867 = stablehlo.broadcast_in_dim %v3866, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3868 = stablehlo.multiply %v3862, %v3865 : tensor<32x112x14x14xf32>
    %v3869 = stablehlo.reduce(%v3868 init: %v3849) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3870 = stablehlo.broadcast_in_dim %v3869, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3871 = stablehlo.multiply %v3865, %v3850 : tensor<32x112x14x14xf32>
    %v3872 = stablehlo.subtract %v3871, %v3867 : tensor<32x112x14x14xf32>
    %v3873 = stablehlo.multiply %v3862, %v3870 : tensor<32x112x14x14xf32>
    %v3874 = stablehlo.subtract %v3872, %v3873 : tensor<32x112x14x14xf32>
    %v3875 = stablehlo.divide %v3861, %v3850 : tensor<32x112x14x14xf32>
    %v3876 = stablehlo.multiply %v3875, %v3874 : tensor<32x112x14x14xf32>
    %v3877 = stablehlo.reshape %v3876 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v3878 = stablehlo.reshape %v3877 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3879 = stablehlo.reverse %b10pW, dims = [2, 3] : tensor<112x672x1x1xf32>
    %v3880 = stablehlo.transpose %v3879, dims = [1, 0, 2, 3] : (tensor<112x672x1x1xf32>) -> tensor<672x112x1x1xf32>
    %v3881 = stablehlo.convolution(%v3878, %v3880)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v3882 = stablehlo.reshape %v3881 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3883 = stablehlo.reshape %v1125 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3884 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3885 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v3886 = stablehlo.reduce(%v3883 init: %v3884) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3887 = stablehlo.broadcast_in_dim %v3886, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3888 = stablehlo.divide %v3887, %v3885 : tensor<32x112x14x14xf32>
    %v3889 = stablehlo.subtract %v3883, %v3888 : tensor<32x112x14x14xf32>
    %v3890 = stablehlo.multiply %v3889, %v3889 : tensor<32x112x14x14xf32>
    %v3891 = stablehlo.reduce(%v3890 init: %v3884) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3892 = stablehlo.broadcast_in_dim %v3891, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3893 = stablehlo.divide %v3892, %v3885 : tensor<32x112x14x14xf32>
    %v3894 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v3895 = stablehlo.add %v3893, %v3894 : tensor<32x112x14x14xf32>
    %v3896 = stablehlo.rsqrt %v3895 : tensor<32x112x14x14xf32>
    %v3897 = stablehlo.multiply %v3889, %v3896 : tensor<32x112x14x14xf32>
    %v3898 = stablehlo.reshape %v3847 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3899 = stablehlo.multiply %v3898, %v3897 : tensor<32x112x14x14xf32>
    %v3900 = stablehlo.reduce(%v3899 init: %v3884) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3901 = stablehlo.constant dense<0.05> : tensor<112xf32>
    %v3902 = stablehlo.multiply %v3900, %v3901 : tensor<112xf32>
    %v3903 = stablehlo.subtract %b10pg, %v3902 : tensor<112xf32>
    %v3904 = stablehlo.reshape %v3847 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3905 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3906 = stablehlo.reduce(%v3904 init: %v3905) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3907 = stablehlo.constant dense<0.05> : tensor<112xf32>
    %v3908 = stablehlo.multiply %v3906, %v3907 : tensor<112xf32>
    %v3909 = stablehlo.subtract %b10pbt, %v3908 : tensor<112xf32>
    %v3910 = stablehlo.reshape %v1120 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3911 = stablehlo.reshape %v3877 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3912 = stablehlo.transpose %v3910, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3913 = stablehlo.transpose %v3911, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %v3914 = stablehlo.convolution(%v3912, %v3913)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<672x32x14x14xf32>, tensor<112x32x14x14xf32>) -> tensor<672x112x1x1xf32>
    %v3915 = stablehlo.transpose %v3914, dims = [1, 0, 2, 3] : (tensor<672x112x1x1xf32>) -> tensor<112x672x1x1xf32>
    %v3916 = stablehlo.constant dense<0.05> : tensor<112x672x1x1xf32>
    %v3917 = stablehlo.multiply %v3915, %v3916 : tensor<112x672x1x1xf32>
    %v3918 = stablehlo.subtract %b10pW, %v3917 : tensor<112x672x1x1xf32>
    %v3919 = stablehlo.reshape %v1090 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3920 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3921 = stablehlo.reduce(%v3919 init: %v3920) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3922 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v3923 = stablehlo.divide %v3921, %v3922 : tensor<32x672xf32>
    %v3924 = stablehlo.dot_general %v3923, %b10zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v3925 = stablehlo.broadcast_in_dim %b10zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v3926 = stablehlo.add %v3924, %v3925 : tensor<32x28xf32>
    %v3927 = stablehlo.logistic %v3926 : tensor<32x28xf32>
    %v3928 = stablehlo.multiply %v3926, %v3927 : tensor<32x28xf32>
    %v3929 = stablehlo.dot_general %v3928, %b10zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v3930 = stablehlo.broadcast_in_dim %b10zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v3931 = stablehlo.add %v3929, %v3930 : tensor<32x672xf32>
    %v3932 = stablehlo.logistic %v3931 : tensor<32x672xf32>
    %v3933 = stablehlo.reshape %v3882 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3934 = stablehlo.broadcast_in_dim %v3932, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v3935 = stablehlo.multiply %v3934, %v3933 : tensor<32x672x14x14xf32>
    %v3936 = stablehlo.multiply %v3919, %v3933 : tensor<32x672x14x14xf32>
    %v3937 = stablehlo.reduce(%v3936 init: %v3920) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3938 = stablehlo.constant dense<1.0> : tensor<32x672xf32>
    %v3939 = stablehlo.subtract %v3938, %v3932 : tensor<32x672xf32>
    %v3940 = stablehlo.multiply %v3932, %v3939 : tensor<32x672xf32>
    %v3941 = stablehlo.multiply %v3937, %v3940 : tensor<32x672xf32>
    %v3942 = stablehlo.dot_general %v3941, %b10zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<28x672xf32>) -> tensor<32x28xf32>
    %v3943 = stablehlo.logistic %v3926 : tensor<32x28xf32>
    %v3944 = stablehlo.constant dense<1.0> : tensor<32x28xf32>
    %v3945 = stablehlo.subtract %v3944, %v3943 : tensor<32x28xf32>
    %v3946 = stablehlo.multiply %v3926, %v3945 : tensor<32x28xf32>
    %v3947 = stablehlo.add %v3944, %v3946 : tensor<32x28xf32>
    %v3948 = stablehlo.multiply %v3943, %v3947 : tensor<32x28xf32>
    %v3949 = stablehlo.multiply %v3942, %v3948 : tensor<32x28xf32>
    %v3950 = stablehlo.dot_general %v3949, %b10zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<672x28xf32>) -> tensor<32x672xf32>
    %v3951 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v3952 = stablehlo.divide %v3950, %v3951 : tensor<32x672xf32>
    %v3953 = stablehlo.broadcast_in_dim %v3952, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v3954 = stablehlo.add %v3935, %v3953 : tensor<32x672x14x14xf32>
    %v3955 = stablehlo.reshape %v3954 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3956 = stablehlo.reshape %v1090 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3957 = stablehlo.reshape %v3882 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3958 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3959 = stablehlo.multiply %v3956, %v3957 : tensor<32x672x14x14xf32>
    %v3960 = stablehlo.reduce(%v3959 init: %v3958) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3961 = stablehlo.logistic %v1103 : tensor<32x672xf32>
    %v3962 = stablehlo.constant dense<1.0> : tensor<32x672xf32>
    %v3963 = stablehlo.subtract %v3962, %v3961 : tensor<32x672xf32>
    %v3964 = stablehlo.multiply %v3961, %v3963 : tensor<32x672xf32>
    %v3965 = stablehlo.multiply %v3960, %v3964 : tensor<32x672xf32>
    %v3966 = stablehlo.dot_general %v1100, %v3965, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<32x672xf32>) -> tensor<28x672xf32>
    %v3967 = stablehlo.constant dense<0.05> : tensor<28x672xf32>
    %v3968 = stablehlo.multiply %v3966, %v3967 : tensor<28x672xf32>
    %v3969 = stablehlo.subtract %b10zW2, %v3968 : tensor<28x672xf32>
    %v3970 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3971 = stablehlo.reduce(%v3965 init: %v3970) applies stablehlo.add across dimensions = [0] : (tensor<32x672xf32>, tensor<f32>) -> tensor<672xf32>
    %v3972 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3973 = stablehlo.multiply %v3971, %v3972 : tensor<672xf32>
    %v3974 = stablehlo.subtract %b10zb2, %v3973 : tensor<672xf32>
    %v3975 = stablehlo.reshape %v3965 : (tensor<32x672xf32>) -> tensor<32x1x672xf32>
    %v3976 = stablehlo.dot_general %v3975, %b10zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x672xf32>, tensor<28x672xf32>) -> tensor<32x1x28xf32>
    %v3977 = stablehlo.reshape %v3976 : (tensor<32x1x28xf32>) -> tensor<32x28xf32>
    %v3978 = stablehlo.logistic %v1098 : tensor<32x28xf32>
    %v3979 = stablehlo.constant dense<1.0> : tensor<32x28xf32>
    %v3980 = stablehlo.subtract %v3979, %v3978 : tensor<32x28xf32>
    %v3981 = stablehlo.multiply %v1098, %v3980 : tensor<32x28xf32>
    %v3982 = stablehlo.add %v3979, %v3981 : tensor<32x28xf32>
    %v3983 = stablehlo.multiply %v3978, %v3982 : tensor<32x28xf32>
    %v3984 = stablehlo.multiply %v3977, %v3983 : tensor<32x28xf32>
    %v3985 = stablehlo.dot_general %v1095, %v3984, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<32x28xf32>) -> tensor<672x28xf32>
    %v3986 = stablehlo.constant dense<0.05> : tensor<672x28xf32>
    %v3987 = stablehlo.multiply %v3985, %v3986 : tensor<672x28xf32>
    %v3988 = stablehlo.subtract %b10zW1, %v3987 : tensor<672x28xf32>
    %v3989 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3990 = stablehlo.reduce(%v3984 init: %v3989) applies stablehlo.add across dimensions = [0] : (tensor<32x28xf32>, tensor<f32>) -> tensor<28xf32>
    %v3991 = stablehlo.constant dense<0.05> : tensor<28xf32>
    %v3992 = stablehlo.multiply %v3990, %v3991 : tensor<28xf32>
    %v3993 = stablehlo.subtract %b10zb1, %v3992 : tensor<28xf32>
    %v3994 = stablehlo.reshape %v3955 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3995 = stablehlo.reshape %v1086 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3996 = stablehlo.logistic %v3995 : tensor<32x672x14x14xf32>
    %v3997 = stablehlo.constant dense<1.0> : tensor<32x672x14x14xf32>
    %v3998 = stablehlo.subtract %v3997, %v3996 : tensor<32x672x14x14xf32>
    %v3999 = stablehlo.multiply %v3995, %v3998 : tensor<32x672x14x14xf32>
    %v4000 = stablehlo.add %v3997, %v3999 : tensor<32x672x14x14xf32>
    %v4001 = stablehlo.multiply %v3996, %v4000 : tensor<32x672x14x14xf32>
    %v4002 = stablehlo.multiply %v3994, %v4001 : tensor<32x672x14x14xf32>
    %v4003 = stablehlo.reshape %v4002 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v4004 = stablehlo.reshape %v1066 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v4005 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4006 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v4007 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v4008 = stablehlo.reduce(%v4004 init: %v4005) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v4009 = stablehlo.broadcast_in_dim %v4008, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v4010 = stablehlo.divide %v4009, %v4006 : tensor<32x672x14x14xf32>
    %v4011 = stablehlo.subtract %v4004, %v4010 : tensor<32x672x14x14xf32>
    %v4012 = stablehlo.multiply %v4011, %v4011 : tensor<32x672x14x14xf32>
    %v4013 = stablehlo.reduce(%v4012 init: %v4005) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v4014 = stablehlo.broadcast_in_dim %v4013, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v4015 = stablehlo.divide %v4014, %v4006 : tensor<32x672x14x14xf32>
    %v4016 = stablehlo.add %v4015, %v4007 : tensor<32x672x14x14xf32>
    %v4017 = stablehlo.rsqrt %v4016 : tensor<32x672x14x14xf32>
    %v4018 = stablehlo.multiply %v4011, %v4017 : tensor<32x672x14x14xf32>
    %v4019 = stablehlo.broadcast_in_dim %b10dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v4020 = stablehlo.reshape %v4003 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v4021 = stablehlo.multiply %v4019, %v4020 : tensor<32x672x14x14xf32>
    %v4022 = stablehlo.reduce(%v4021 init: %v4005) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v4023 = stablehlo.broadcast_in_dim %v4022, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v4024 = stablehlo.multiply %v4018, %v4021 : tensor<32x672x14x14xf32>
    %v4025 = stablehlo.reduce(%v4024 init: %v4005) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v4026 = stablehlo.broadcast_in_dim %v4025, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v4027 = stablehlo.multiply %v4021, %v4006 : tensor<32x672x14x14xf32>
    %v4028 = stablehlo.subtract %v4027, %v4023 : tensor<32x672x14x14xf32>
    %v4029 = stablehlo.multiply %v4018, %v4026 : tensor<32x672x14x14xf32>
    %v4030 = stablehlo.subtract %v4028, %v4029 : tensor<32x672x14x14xf32>
    %v4031 = stablehlo.divide %v4017, %v4006 : tensor<32x672x14x14xf32>
    %v4032 = stablehlo.multiply %v4031, %v4030 : tensor<32x672x14x14xf32>
    %v4033 = stablehlo.reshape %v4032 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v4034 = stablehlo.reshape %v4033 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v4035 = stablehlo.reverse %b10dW, dims = [2, 3] : tensor<672x1x5x5xf32>
    %v4036 = stablehlo.convolution(%v4034, %v4035)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %v4037 = stablehlo.reshape %v4036 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v4038 = stablehlo.reshape %v1066 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v4039 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4040 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v4041 = stablehlo.reduce(%v4038 init: %v4039) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v4042 = stablehlo.broadcast_in_dim %v4041, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v4043 = stablehlo.divide %v4042, %v4040 : tensor<32x672x14x14xf32>
    %v4044 = stablehlo.subtract %v4038, %v4043 : tensor<32x672x14x14xf32>
    %v4045 = stablehlo.multiply %v4044, %v4044 : tensor<32x672x14x14xf32>
    %v4046 = stablehlo.reduce(%v4045 init: %v4039) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v4047 = stablehlo.broadcast_in_dim %v4046, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v4048 = stablehlo.divide %v4047, %v4040 : tensor<32x672x14x14xf32>
    %v4049 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v4050 = stablehlo.add %v4048, %v4049 : tensor<32x672x14x14xf32>
    %v4051 = stablehlo.rsqrt %v4050 : tensor<32x672x14x14xf32>
    %v4052 = stablehlo.multiply %v4044, %v4051 : tensor<32x672x14x14xf32>
    %v4053 = stablehlo.reshape %v4003 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v4054 = stablehlo.multiply %v4053, %v4052 : tensor<32x672x14x14xf32>
    %v4055 = stablehlo.reduce(%v4054 init: %v4039) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v4056 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v4057 = stablehlo.multiply %v4055, %v4056 : tensor<672xf32>
    %v4058 = stablehlo.subtract %b10dg, %v4057 : tensor<672xf32>
    %v4059 = stablehlo.reshape %v4003 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v4060 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4061 = stablehlo.reduce(%v4059 init: %v4060) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v4062 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v4063 = stablehlo.multiply %v4061, %v4062 : tensor<672xf32>
    %v4064 = stablehlo.subtract %b10dbt, %v4063 : tensor<672xf32>
    %v4065 = stablehlo.reshape %v1061 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v4066 = stablehlo.reshape %v4033 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v4067 = stablehlo.transpose %v4065, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v4068 = stablehlo.transpose %v4066, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v4069 = stablehlo.convolution(%v4067, %v4068)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 672 : i64, feature_group_count = 1 : i64} : (tensor<672x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<1x672x5x5xf32>
    %v4070 = stablehlo.reshape %v4069 : (tensor<1x672x5x5xf32>) -> tensor<672x1x5x5xf32>
    %v4071 = stablehlo.constant dense<0.05> : tensor<672x1x5x5xf32>
    %v4072 = stablehlo.multiply %v4070, %v4071 : tensor<672x1x5x5xf32>
    %v4073 = stablehlo.subtract %b10dW, %v4072 : tensor<672x1x5x5xf32>
    %v4074 = stablehlo.reshape %v4037 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v4075 = stablehlo.reshape %v1057 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v4076 = stablehlo.logistic %v4075 : tensor<32x672x14x14xf32>
    %v4077 = stablehlo.constant dense<1.0> : tensor<32x672x14x14xf32>
    %v4078 = stablehlo.subtract %v4077, %v4076 : tensor<32x672x14x14xf32>
    %v4079 = stablehlo.multiply %v4075, %v4078 : tensor<32x672x14x14xf32>
    %v4080 = stablehlo.add %v4077, %v4079 : tensor<32x672x14x14xf32>
    %v4081 = stablehlo.multiply %v4076, %v4080 : tensor<32x672x14x14xf32>
    %v4082 = stablehlo.multiply %v4074, %v4081 : tensor<32x672x14x14xf32>
    %v4083 = stablehlo.reshape %v4082 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v4084 = stablehlo.reshape %v1037 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v4085 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4086 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v4087 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v4088 = stablehlo.reduce(%v4084 init: %v4085) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v4089 = stablehlo.broadcast_in_dim %v4088, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v4090 = stablehlo.divide %v4089, %v4086 : tensor<32x672x14x14xf32>
    %v4091 = stablehlo.subtract %v4084, %v4090 : tensor<32x672x14x14xf32>
    %v4092 = stablehlo.multiply %v4091, %v4091 : tensor<32x672x14x14xf32>
    %v4093 = stablehlo.reduce(%v4092 init: %v4085) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v4094 = stablehlo.broadcast_in_dim %v4093, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v4095 = stablehlo.divide %v4094, %v4086 : tensor<32x672x14x14xf32>
    %v4096 = stablehlo.add %v4095, %v4087 : tensor<32x672x14x14xf32>
    %v4097 = stablehlo.rsqrt %v4096 : tensor<32x672x14x14xf32>
    %v4098 = stablehlo.multiply %v4091, %v4097 : tensor<32x672x14x14xf32>
    %v4099 = stablehlo.broadcast_in_dim %b10eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v4100 = stablehlo.reshape %v4083 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v4101 = stablehlo.multiply %v4099, %v4100 : tensor<32x672x14x14xf32>
    %v4102 = stablehlo.reduce(%v4101 init: %v4085) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v4103 = stablehlo.broadcast_in_dim %v4102, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v4104 = stablehlo.multiply %v4098, %v4101 : tensor<32x672x14x14xf32>
    %v4105 = stablehlo.reduce(%v4104 init: %v4085) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v4106 = stablehlo.broadcast_in_dim %v4105, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v4107 = stablehlo.multiply %v4101, %v4086 : tensor<32x672x14x14xf32>
    %v4108 = stablehlo.subtract %v4107, %v4103 : tensor<32x672x14x14xf32>
    %v4109 = stablehlo.multiply %v4098, %v4106 : tensor<32x672x14x14xf32>
    %v4110 = stablehlo.subtract %v4108, %v4109 : tensor<32x672x14x14xf32>
    %v4111 = stablehlo.divide %v4097, %v4086 : tensor<32x672x14x14xf32>
    %v4112 = stablehlo.multiply %v4111, %v4110 : tensor<32x672x14x14xf32>
    %v4113 = stablehlo.reshape %v4112 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v4114 = stablehlo.reshape %v4113 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v4115 = stablehlo.reverse %b10eW, dims = [2, 3] : tensor<672x112x1x1xf32>
    %v4116 = stablehlo.transpose %v4115, dims = [1, 0, 2, 3] : (tensor<672x112x1x1xf32>) -> tensor<112x672x1x1xf32>
    %v4117 = stablehlo.convolution(%v4114, %v4116)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v4118 = stablehlo.reshape %v4117 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v4119 = stablehlo.reshape %v1037 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v4120 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4121 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v4122 = stablehlo.reduce(%v4119 init: %v4120) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v4123 = stablehlo.broadcast_in_dim %v4122, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v4124 = stablehlo.divide %v4123, %v4121 : tensor<32x672x14x14xf32>
    %v4125 = stablehlo.subtract %v4119, %v4124 : tensor<32x672x14x14xf32>
    %v4126 = stablehlo.multiply %v4125, %v4125 : tensor<32x672x14x14xf32>
    %v4127 = stablehlo.reduce(%v4126 init: %v4120) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v4128 = stablehlo.broadcast_in_dim %v4127, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v4129 = stablehlo.divide %v4128, %v4121 : tensor<32x672x14x14xf32>
    %v4130 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v4131 = stablehlo.add %v4129, %v4130 : tensor<32x672x14x14xf32>
    %v4132 = stablehlo.rsqrt %v4131 : tensor<32x672x14x14xf32>
    %v4133 = stablehlo.multiply %v4125, %v4132 : tensor<32x672x14x14xf32>
    %v4134 = stablehlo.reshape %v4083 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v4135 = stablehlo.multiply %v4134, %v4133 : tensor<32x672x14x14xf32>
    %v4136 = stablehlo.reduce(%v4135 init: %v4120) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v4137 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v4138 = stablehlo.multiply %v4136, %v4137 : tensor<672xf32>
    %v4139 = stablehlo.subtract %b10eg, %v4138 : tensor<672xf32>
    %v4140 = stablehlo.reshape %v4083 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v4141 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4142 = stablehlo.reduce(%v4140 init: %v4141) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v4143 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v4144 = stablehlo.multiply %v4142, %v4143 : tensor<672xf32>
    %v4145 = stablehlo.subtract %b10ebt, %v4144 : tensor<672xf32>
    %v4146 = stablehlo.reshape %v1032 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v4147 = stablehlo.reshape %v4113 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v4148 = stablehlo.transpose %v4146, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %v4149 = stablehlo.transpose %v4147, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v4150 = stablehlo.convolution(%v4148, %v4149)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<112x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<112x672x1x1xf32>
    %v4151 = stablehlo.transpose %v4150, dims = [1, 0, 2, 3] : (tensor<112x672x1x1xf32>) -> tensor<672x112x1x1xf32>
    %v4152 = stablehlo.constant dense<0.05> : tensor<672x112x1x1xf32>
    %v4153 = stablehlo.multiply %v4151, %v4152 : tensor<672x112x1x1xf32>
    %v4154 = stablehlo.subtract %b10eW, %v4153 : tensor<672x112x1x1xf32>
    %v4155 = stablehlo.reshape %v4118 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v4156 = stablehlo.reshape %v3847 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v4157 = stablehlo.add %v4155, %v4156 : tensor<32x112x14x14xf32>
    %v4158 = stablehlo.reshape %v4157 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v4159 = stablehlo.reshape %v1012 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v4160 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4161 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v4162 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v4163 = stablehlo.reduce(%v4159 init: %v4160) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v4164 = stablehlo.broadcast_in_dim %v4163, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v4165 = stablehlo.divide %v4164, %v4161 : tensor<32x112x14x14xf32>
    %v4166 = stablehlo.subtract %v4159, %v4165 : tensor<32x112x14x14xf32>
    %v4167 = stablehlo.multiply %v4166, %v4166 : tensor<32x112x14x14xf32>
    %v4168 = stablehlo.reduce(%v4167 init: %v4160) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v4169 = stablehlo.broadcast_in_dim %v4168, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v4170 = stablehlo.divide %v4169, %v4161 : tensor<32x112x14x14xf32>
    %v4171 = stablehlo.add %v4170, %v4162 : tensor<32x112x14x14xf32>
    %v4172 = stablehlo.rsqrt %v4171 : tensor<32x112x14x14xf32>
    %v4173 = stablehlo.multiply %v4166, %v4172 : tensor<32x112x14x14xf32>
    %v4174 = stablehlo.broadcast_in_dim %b9pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v4175 = stablehlo.reshape %v4158 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v4176 = stablehlo.multiply %v4174, %v4175 : tensor<32x112x14x14xf32>
    %v4177 = stablehlo.reduce(%v4176 init: %v4160) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v4178 = stablehlo.broadcast_in_dim %v4177, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v4179 = stablehlo.multiply %v4173, %v4176 : tensor<32x112x14x14xf32>
    %v4180 = stablehlo.reduce(%v4179 init: %v4160) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v4181 = stablehlo.broadcast_in_dim %v4180, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v4182 = stablehlo.multiply %v4176, %v4161 : tensor<32x112x14x14xf32>
    %v4183 = stablehlo.subtract %v4182, %v4178 : tensor<32x112x14x14xf32>
    %v4184 = stablehlo.multiply %v4173, %v4181 : tensor<32x112x14x14xf32>
    %v4185 = stablehlo.subtract %v4183, %v4184 : tensor<32x112x14x14xf32>
    %v4186 = stablehlo.divide %v4172, %v4161 : tensor<32x112x14x14xf32>
    %v4187 = stablehlo.multiply %v4186, %v4185 : tensor<32x112x14x14xf32>
    %v4188 = stablehlo.reshape %v4187 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v4189 = stablehlo.reshape %v4188 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v4190 = stablehlo.reverse %b9pW, dims = [2, 3] : tensor<112x480x1x1xf32>
    %v4191 = stablehlo.transpose %v4190, dims = [1, 0, 2, 3] : (tensor<112x480x1x1xf32>) -> tensor<480x112x1x1xf32>
    %v4192 = stablehlo.convolution(%v4189, %v4191)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<480x112x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v4193 = stablehlo.reshape %v4192 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4194 = stablehlo.reshape %v1012 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v4195 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4196 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v4197 = stablehlo.reduce(%v4194 init: %v4195) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v4198 = stablehlo.broadcast_in_dim %v4197, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v4199 = stablehlo.divide %v4198, %v4196 : tensor<32x112x14x14xf32>
    %v4200 = stablehlo.subtract %v4194, %v4199 : tensor<32x112x14x14xf32>
    %v4201 = stablehlo.multiply %v4200, %v4200 : tensor<32x112x14x14xf32>
    %v4202 = stablehlo.reduce(%v4201 init: %v4195) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v4203 = stablehlo.broadcast_in_dim %v4202, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v4204 = stablehlo.divide %v4203, %v4196 : tensor<32x112x14x14xf32>
    %v4205 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v4206 = stablehlo.add %v4204, %v4205 : tensor<32x112x14x14xf32>
    %v4207 = stablehlo.rsqrt %v4206 : tensor<32x112x14x14xf32>
    %v4208 = stablehlo.multiply %v4200, %v4207 : tensor<32x112x14x14xf32>
    %v4209 = stablehlo.reshape %v4158 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v4210 = stablehlo.multiply %v4209, %v4208 : tensor<32x112x14x14xf32>
    %v4211 = stablehlo.reduce(%v4210 init: %v4195) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v4212 = stablehlo.constant dense<0.05> : tensor<112xf32>
    %v4213 = stablehlo.multiply %v4211, %v4212 : tensor<112xf32>
    %v4214 = stablehlo.subtract %b9pg, %v4213 : tensor<112xf32>
    %v4215 = stablehlo.reshape %v4158 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v4216 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4217 = stablehlo.reduce(%v4215 init: %v4216) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v4218 = stablehlo.constant dense<0.05> : tensor<112xf32>
    %v4219 = stablehlo.multiply %v4217, %v4218 : tensor<112xf32>
    %v4220 = stablehlo.subtract %b9pbt, %v4219 : tensor<112xf32>
    %v4221 = stablehlo.reshape %v1007 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4222 = stablehlo.reshape %v4188 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v4223 = stablehlo.transpose %v4221, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4224 = stablehlo.transpose %v4222, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %v4225 = stablehlo.convolution(%v4223, %v4224)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<112x32x14x14xf32>) -> tensor<480x112x1x1xf32>
    %v4226 = stablehlo.transpose %v4225, dims = [1, 0, 2, 3] : (tensor<480x112x1x1xf32>) -> tensor<112x480x1x1xf32>
    %v4227 = stablehlo.constant dense<0.05> : tensor<112x480x1x1xf32>
    %v4228 = stablehlo.multiply %v4226, %v4227 : tensor<112x480x1x1xf32>
    %v4229 = stablehlo.subtract %b9pW, %v4228 : tensor<112x480x1x1xf32>
    %v4230 = stablehlo.reshape %v977 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4231 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4232 = stablehlo.reduce(%v4230 init: %v4231) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4233 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v4234 = stablehlo.divide %v4232, %v4233 : tensor<32x480xf32>
    %v4235 = stablehlo.dot_general %v4234, %b9zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v4236 = stablehlo.broadcast_in_dim %b9zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v4237 = stablehlo.add %v4235, %v4236 : tensor<32x20xf32>
    %v4238 = stablehlo.logistic %v4237 : tensor<32x20xf32>
    %v4239 = stablehlo.multiply %v4237, %v4238 : tensor<32x20xf32>
    %v4240 = stablehlo.dot_general %v4239, %b9zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v4241 = stablehlo.broadcast_in_dim %b9zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v4242 = stablehlo.add %v4240, %v4241 : tensor<32x480xf32>
    %v4243 = stablehlo.logistic %v4242 : tensor<32x480xf32>
    %v4244 = stablehlo.reshape %v4193 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4245 = stablehlo.broadcast_in_dim %v4243, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v4246 = stablehlo.multiply %v4245, %v4244 : tensor<32x480x14x14xf32>
    %v4247 = stablehlo.multiply %v4230, %v4244 : tensor<32x480x14x14xf32>
    %v4248 = stablehlo.reduce(%v4247 init: %v4231) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4249 = stablehlo.constant dense<1.0> : tensor<32x480xf32>
    %v4250 = stablehlo.subtract %v4249, %v4243 : tensor<32x480xf32>
    %v4251 = stablehlo.multiply %v4243, %v4250 : tensor<32x480xf32>
    %v4252 = stablehlo.multiply %v4248, %v4251 : tensor<32x480xf32>
    %v4253 = stablehlo.dot_general %v4252, %b9zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<20x480xf32>) -> tensor<32x20xf32>
    %v4254 = stablehlo.logistic %v4237 : tensor<32x20xf32>
    %v4255 = stablehlo.constant dense<1.0> : tensor<32x20xf32>
    %v4256 = stablehlo.subtract %v4255, %v4254 : tensor<32x20xf32>
    %v4257 = stablehlo.multiply %v4237, %v4256 : tensor<32x20xf32>
    %v4258 = stablehlo.add %v4255, %v4257 : tensor<32x20xf32>
    %v4259 = stablehlo.multiply %v4254, %v4258 : tensor<32x20xf32>
    %v4260 = stablehlo.multiply %v4253, %v4259 : tensor<32x20xf32>
    %v4261 = stablehlo.dot_general %v4260, %b9zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<480x20xf32>) -> tensor<32x480xf32>
    %v4262 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v4263 = stablehlo.divide %v4261, %v4262 : tensor<32x480xf32>
    %v4264 = stablehlo.broadcast_in_dim %v4263, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v4265 = stablehlo.add %v4246, %v4264 : tensor<32x480x14x14xf32>
    %v4266 = stablehlo.reshape %v4265 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4267 = stablehlo.reshape %v977 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4268 = stablehlo.reshape %v4193 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4269 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4270 = stablehlo.multiply %v4267, %v4268 : tensor<32x480x14x14xf32>
    %v4271 = stablehlo.reduce(%v4270 init: %v4269) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4272 = stablehlo.logistic %v990 : tensor<32x480xf32>
    %v4273 = stablehlo.constant dense<1.0> : tensor<32x480xf32>
    %v4274 = stablehlo.subtract %v4273, %v4272 : tensor<32x480xf32>
    %v4275 = stablehlo.multiply %v4272, %v4274 : tensor<32x480xf32>
    %v4276 = stablehlo.multiply %v4271, %v4275 : tensor<32x480xf32>
    %v4277 = stablehlo.dot_general %v987, %v4276, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<32x480xf32>) -> tensor<20x480xf32>
    %v4278 = stablehlo.constant dense<0.05> : tensor<20x480xf32>
    %v4279 = stablehlo.multiply %v4277, %v4278 : tensor<20x480xf32>
    %v4280 = stablehlo.subtract %b9zW2, %v4279 : tensor<20x480xf32>
    %v4281 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4282 = stablehlo.reduce(%v4276 init: %v4281) applies stablehlo.add across dimensions = [0] : (tensor<32x480xf32>, tensor<f32>) -> tensor<480xf32>
    %v4283 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4284 = stablehlo.multiply %v4282, %v4283 : tensor<480xf32>
    %v4285 = stablehlo.subtract %b9zb2, %v4284 : tensor<480xf32>
    %v4286 = stablehlo.reshape %v4276 : (tensor<32x480xf32>) -> tensor<32x1x480xf32>
    %v4287 = stablehlo.dot_general %v4286, %b9zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x480xf32>, tensor<20x480xf32>) -> tensor<32x1x20xf32>
    %v4288 = stablehlo.reshape %v4287 : (tensor<32x1x20xf32>) -> tensor<32x20xf32>
    %v4289 = stablehlo.logistic %v985 : tensor<32x20xf32>
    %v4290 = stablehlo.constant dense<1.0> : tensor<32x20xf32>
    %v4291 = stablehlo.subtract %v4290, %v4289 : tensor<32x20xf32>
    %v4292 = stablehlo.multiply %v985, %v4291 : tensor<32x20xf32>
    %v4293 = stablehlo.add %v4290, %v4292 : tensor<32x20xf32>
    %v4294 = stablehlo.multiply %v4289, %v4293 : tensor<32x20xf32>
    %v4295 = stablehlo.multiply %v4288, %v4294 : tensor<32x20xf32>
    %v4296 = stablehlo.dot_general %v982, %v4295, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<32x20xf32>) -> tensor<480x20xf32>
    %v4297 = stablehlo.constant dense<0.05> : tensor<480x20xf32>
    %v4298 = stablehlo.multiply %v4296, %v4297 : tensor<480x20xf32>
    %v4299 = stablehlo.subtract %b9zW1, %v4298 : tensor<480x20xf32>
    %v4300 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4301 = stablehlo.reduce(%v4295 init: %v4300) applies stablehlo.add across dimensions = [0] : (tensor<32x20xf32>, tensor<f32>) -> tensor<20xf32>
    %v4302 = stablehlo.constant dense<0.05> : tensor<20xf32>
    %v4303 = stablehlo.multiply %v4301, %v4302 : tensor<20xf32>
    %v4304 = stablehlo.subtract %b9zb1, %v4303 : tensor<20xf32>
    %v4305 = stablehlo.reshape %v4266 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4306 = stablehlo.reshape %v973 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4307 = stablehlo.logistic %v4306 : tensor<32x480x14x14xf32>
    %v4308 = stablehlo.constant dense<1.0> : tensor<32x480x14x14xf32>
    %v4309 = stablehlo.subtract %v4308, %v4307 : tensor<32x480x14x14xf32>
    %v4310 = stablehlo.multiply %v4306, %v4309 : tensor<32x480x14x14xf32>
    %v4311 = stablehlo.add %v4308, %v4310 : tensor<32x480x14x14xf32>
    %v4312 = stablehlo.multiply %v4307, %v4311 : tensor<32x480x14x14xf32>
    %v4313 = stablehlo.multiply %v4305, %v4312 : tensor<32x480x14x14xf32>
    %v4314 = stablehlo.reshape %v4313 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4315 = stablehlo.reshape %v953 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4316 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4317 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4318 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4319 = stablehlo.reduce(%v4315 init: %v4316) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4320 = stablehlo.broadcast_in_dim %v4319, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4321 = stablehlo.divide %v4320, %v4317 : tensor<32x480x14x14xf32>
    %v4322 = stablehlo.subtract %v4315, %v4321 : tensor<32x480x14x14xf32>
    %v4323 = stablehlo.multiply %v4322, %v4322 : tensor<32x480x14x14xf32>
    %v4324 = stablehlo.reduce(%v4323 init: %v4316) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4325 = stablehlo.broadcast_in_dim %v4324, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4326 = stablehlo.divide %v4325, %v4317 : tensor<32x480x14x14xf32>
    %v4327 = stablehlo.add %v4326, %v4318 : tensor<32x480x14x14xf32>
    %v4328 = stablehlo.rsqrt %v4327 : tensor<32x480x14x14xf32>
    %v4329 = stablehlo.multiply %v4322, %v4328 : tensor<32x480x14x14xf32>
    %v4330 = stablehlo.broadcast_in_dim %b9dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4331 = stablehlo.reshape %v4314 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4332 = stablehlo.multiply %v4330, %v4331 : tensor<32x480x14x14xf32>
    %v4333 = stablehlo.reduce(%v4332 init: %v4316) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4334 = stablehlo.broadcast_in_dim %v4333, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4335 = stablehlo.multiply %v4329, %v4332 : tensor<32x480x14x14xf32>
    %v4336 = stablehlo.reduce(%v4335 init: %v4316) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4337 = stablehlo.broadcast_in_dim %v4336, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4338 = stablehlo.multiply %v4332, %v4317 : tensor<32x480x14x14xf32>
    %v4339 = stablehlo.subtract %v4338, %v4334 : tensor<32x480x14x14xf32>
    %v4340 = stablehlo.multiply %v4329, %v4337 : tensor<32x480x14x14xf32>
    %v4341 = stablehlo.subtract %v4339, %v4340 : tensor<32x480x14x14xf32>
    %v4342 = stablehlo.divide %v4328, %v4317 : tensor<32x480x14x14xf32>
    %v4343 = stablehlo.multiply %v4342, %v4341 : tensor<32x480x14x14xf32>
    %v4344 = stablehlo.reshape %v4343 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4345 = stablehlo.reshape %v4344 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4346 = stablehlo.reverse %b9dW, dims = [2, 3] : tensor<480x1x5x5xf32>
    %v4347 = stablehlo.convolution(%v4345, %v4346)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x5x5xf32>) -> tensor<32x480x14x14xf32>
    %v4348 = stablehlo.reshape %v4347 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4349 = stablehlo.reshape %v953 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4350 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4351 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4352 = stablehlo.reduce(%v4349 init: %v4350) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4353 = stablehlo.broadcast_in_dim %v4352, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4354 = stablehlo.divide %v4353, %v4351 : tensor<32x480x14x14xf32>
    %v4355 = stablehlo.subtract %v4349, %v4354 : tensor<32x480x14x14xf32>
    %v4356 = stablehlo.multiply %v4355, %v4355 : tensor<32x480x14x14xf32>
    %v4357 = stablehlo.reduce(%v4356 init: %v4350) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4358 = stablehlo.broadcast_in_dim %v4357, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4359 = stablehlo.divide %v4358, %v4351 : tensor<32x480x14x14xf32>
    %v4360 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4361 = stablehlo.add %v4359, %v4360 : tensor<32x480x14x14xf32>
    %v4362 = stablehlo.rsqrt %v4361 : tensor<32x480x14x14xf32>
    %v4363 = stablehlo.multiply %v4355, %v4362 : tensor<32x480x14x14xf32>
    %v4364 = stablehlo.reshape %v4314 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4365 = stablehlo.multiply %v4364, %v4363 : tensor<32x480x14x14xf32>
    %v4366 = stablehlo.reduce(%v4365 init: %v4350) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4367 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4368 = stablehlo.multiply %v4366, %v4367 : tensor<480xf32>
    %v4369 = stablehlo.subtract %b9dg, %v4368 : tensor<480xf32>
    %v4370 = stablehlo.reshape %v4314 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4371 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4372 = stablehlo.reduce(%v4370 init: %v4371) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4373 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4374 = stablehlo.multiply %v4372, %v4373 : tensor<480xf32>
    %v4375 = stablehlo.subtract %b9dbt, %v4374 : tensor<480xf32>
    %v4376 = stablehlo.reshape %v948 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4377 = stablehlo.reshape %v4344 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4378 = stablehlo.transpose %v4376, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4379 = stablehlo.transpose %v4377, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4380 = stablehlo.convolution(%v4378, %v4379)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 480 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<1x480x5x5xf32>
    %v4381 = stablehlo.reshape %v4380 : (tensor<1x480x5x5xf32>) -> tensor<480x1x5x5xf32>
    %v4382 = stablehlo.constant dense<0.05> : tensor<480x1x5x5xf32>
    %v4383 = stablehlo.multiply %v4381, %v4382 : tensor<480x1x5x5xf32>
    %v4384 = stablehlo.subtract %b9dW, %v4383 : tensor<480x1x5x5xf32>
    %v4385 = stablehlo.reshape %v4348 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4386 = stablehlo.reshape %v944 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4387 = stablehlo.logistic %v4386 : tensor<32x480x14x14xf32>
    %v4388 = stablehlo.constant dense<1.0> : tensor<32x480x14x14xf32>
    %v4389 = stablehlo.subtract %v4388, %v4387 : tensor<32x480x14x14xf32>
    %v4390 = stablehlo.multiply %v4386, %v4389 : tensor<32x480x14x14xf32>
    %v4391 = stablehlo.add %v4388, %v4390 : tensor<32x480x14x14xf32>
    %v4392 = stablehlo.multiply %v4387, %v4391 : tensor<32x480x14x14xf32>
    %v4393 = stablehlo.multiply %v4385, %v4392 : tensor<32x480x14x14xf32>
    %v4394 = stablehlo.reshape %v4393 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4395 = stablehlo.reshape %v924 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4396 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4397 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4398 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4399 = stablehlo.reduce(%v4395 init: %v4396) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4400 = stablehlo.broadcast_in_dim %v4399, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4401 = stablehlo.divide %v4400, %v4397 : tensor<32x480x14x14xf32>
    %v4402 = stablehlo.subtract %v4395, %v4401 : tensor<32x480x14x14xf32>
    %v4403 = stablehlo.multiply %v4402, %v4402 : tensor<32x480x14x14xf32>
    %v4404 = stablehlo.reduce(%v4403 init: %v4396) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4405 = stablehlo.broadcast_in_dim %v4404, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4406 = stablehlo.divide %v4405, %v4397 : tensor<32x480x14x14xf32>
    %v4407 = stablehlo.add %v4406, %v4398 : tensor<32x480x14x14xf32>
    %v4408 = stablehlo.rsqrt %v4407 : tensor<32x480x14x14xf32>
    %v4409 = stablehlo.multiply %v4402, %v4408 : tensor<32x480x14x14xf32>
    %v4410 = stablehlo.broadcast_in_dim %b9eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4411 = stablehlo.reshape %v4394 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4412 = stablehlo.multiply %v4410, %v4411 : tensor<32x480x14x14xf32>
    %v4413 = stablehlo.reduce(%v4412 init: %v4396) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4414 = stablehlo.broadcast_in_dim %v4413, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4415 = stablehlo.multiply %v4409, %v4412 : tensor<32x480x14x14xf32>
    %v4416 = stablehlo.reduce(%v4415 init: %v4396) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4417 = stablehlo.broadcast_in_dim %v4416, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4418 = stablehlo.multiply %v4412, %v4397 : tensor<32x480x14x14xf32>
    %v4419 = stablehlo.subtract %v4418, %v4414 : tensor<32x480x14x14xf32>
    %v4420 = stablehlo.multiply %v4409, %v4417 : tensor<32x480x14x14xf32>
    %v4421 = stablehlo.subtract %v4419, %v4420 : tensor<32x480x14x14xf32>
    %v4422 = stablehlo.divide %v4408, %v4397 : tensor<32x480x14x14xf32>
    %v4423 = stablehlo.multiply %v4422, %v4421 : tensor<32x480x14x14xf32>
    %v4424 = stablehlo.reshape %v4423 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4425 = stablehlo.reshape %v4424 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4426 = stablehlo.reverse %b9eW, dims = [2, 3] : tensor<480x80x1x1xf32>
    %v4427 = stablehlo.transpose %v4426, dims = [1, 0, 2, 3] : (tensor<480x80x1x1xf32>) -> tensor<80x480x1x1xf32>
    %v4428 = stablehlo.convolution(%v4425, %v4427)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v4429 = stablehlo.reshape %v4428 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v4430 = stablehlo.reshape %v924 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4431 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4432 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4433 = stablehlo.reduce(%v4430 init: %v4431) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4434 = stablehlo.broadcast_in_dim %v4433, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4435 = stablehlo.divide %v4434, %v4432 : tensor<32x480x14x14xf32>
    %v4436 = stablehlo.subtract %v4430, %v4435 : tensor<32x480x14x14xf32>
    %v4437 = stablehlo.multiply %v4436, %v4436 : tensor<32x480x14x14xf32>
    %v4438 = stablehlo.reduce(%v4437 init: %v4431) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4439 = stablehlo.broadcast_in_dim %v4438, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4440 = stablehlo.divide %v4439, %v4432 : tensor<32x480x14x14xf32>
    %v4441 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4442 = stablehlo.add %v4440, %v4441 : tensor<32x480x14x14xf32>
    %v4443 = stablehlo.rsqrt %v4442 : tensor<32x480x14x14xf32>
    %v4444 = stablehlo.multiply %v4436, %v4443 : tensor<32x480x14x14xf32>
    %v4445 = stablehlo.reshape %v4394 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4446 = stablehlo.multiply %v4445, %v4444 : tensor<32x480x14x14xf32>
    %v4447 = stablehlo.reduce(%v4446 init: %v4431) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4448 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4449 = stablehlo.multiply %v4447, %v4448 : tensor<480xf32>
    %v4450 = stablehlo.subtract %b9eg, %v4449 : tensor<480xf32>
    %v4451 = stablehlo.reshape %v4394 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4452 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4453 = stablehlo.reduce(%v4451 init: %v4452) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4454 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4455 = stablehlo.multiply %v4453, %v4454 : tensor<480xf32>
    %v4456 = stablehlo.subtract %b9ebt, %v4455 : tensor<480xf32>
    %v4457 = stablehlo.reshape %v919 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4458 = stablehlo.reshape %v4424 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4459 = stablehlo.transpose %v4457, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %v4460 = stablehlo.transpose %v4458, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4461 = stablehlo.convolution(%v4459, %v4460)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<80x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<80x480x1x1xf32>
    %v4462 = stablehlo.transpose %v4461, dims = [1, 0, 2, 3] : (tensor<80x480x1x1xf32>) -> tensor<480x80x1x1xf32>
    %v4463 = stablehlo.constant dense<0.05> : tensor<480x80x1x1xf32>
    %v4464 = stablehlo.multiply %v4462, %v4463 : tensor<480x80x1x1xf32>
    %v4465 = stablehlo.subtract %b9eW, %v4464 : tensor<480x80x1x1xf32>
    %v4466 = stablehlo.reshape %v895 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4467 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4468 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v4469 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v4470 = stablehlo.reduce(%v4466 init: %v4467) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4471 = stablehlo.broadcast_in_dim %v4470, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4472 = stablehlo.divide %v4471, %v4468 : tensor<32x80x14x14xf32>
    %v4473 = stablehlo.subtract %v4466, %v4472 : tensor<32x80x14x14xf32>
    %v4474 = stablehlo.multiply %v4473, %v4473 : tensor<32x80x14x14xf32>
    %v4475 = stablehlo.reduce(%v4474 init: %v4467) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4476 = stablehlo.broadcast_in_dim %v4475, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4477 = stablehlo.divide %v4476, %v4468 : tensor<32x80x14x14xf32>
    %v4478 = stablehlo.add %v4477, %v4469 : tensor<32x80x14x14xf32>
    %v4479 = stablehlo.rsqrt %v4478 : tensor<32x80x14x14xf32>
    %v4480 = stablehlo.multiply %v4473, %v4479 : tensor<32x80x14x14xf32>
    %v4481 = stablehlo.broadcast_in_dim %b8pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4482 = stablehlo.reshape %v4429 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4483 = stablehlo.multiply %v4481, %v4482 : tensor<32x80x14x14xf32>
    %v4484 = stablehlo.reduce(%v4483 init: %v4467) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4485 = stablehlo.broadcast_in_dim %v4484, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4486 = stablehlo.multiply %v4480, %v4483 : tensor<32x80x14x14xf32>
    %v4487 = stablehlo.reduce(%v4486 init: %v4467) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4488 = stablehlo.broadcast_in_dim %v4487, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4489 = stablehlo.multiply %v4483, %v4468 : tensor<32x80x14x14xf32>
    %v4490 = stablehlo.subtract %v4489, %v4485 : tensor<32x80x14x14xf32>
    %v4491 = stablehlo.multiply %v4480, %v4488 : tensor<32x80x14x14xf32>
    %v4492 = stablehlo.subtract %v4490, %v4491 : tensor<32x80x14x14xf32>
    %v4493 = stablehlo.divide %v4479, %v4468 : tensor<32x80x14x14xf32>
    %v4494 = stablehlo.multiply %v4493, %v4492 : tensor<32x80x14x14xf32>
    %v4495 = stablehlo.reshape %v4494 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v4496 = stablehlo.reshape %v4495 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4497 = stablehlo.reverse %b8pW, dims = [2, 3] : tensor<80x480x1x1xf32>
    %v4498 = stablehlo.transpose %v4497, dims = [1, 0, 2, 3] : (tensor<80x480x1x1xf32>) -> tensor<480x80x1x1xf32>
    %v4499 = stablehlo.convolution(%v4496, %v4498)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v4500 = stablehlo.reshape %v4499 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4501 = stablehlo.reshape %v895 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4502 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4503 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v4504 = stablehlo.reduce(%v4501 init: %v4502) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4505 = stablehlo.broadcast_in_dim %v4504, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4506 = stablehlo.divide %v4505, %v4503 : tensor<32x80x14x14xf32>
    %v4507 = stablehlo.subtract %v4501, %v4506 : tensor<32x80x14x14xf32>
    %v4508 = stablehlo.multiply %v4507, %v4507 : tensor<32x80x14x14xf32>
    %v4509 = stablehlo.reduce(%v4508 init: %v4502) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4510 = stablehlo.broadcast_in_dim %v4509, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4511 = stablehlo.divide %v4510, %v4503 : tensor<32x80x14x14xf32>
    %v4512 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v4513 = stablehlo.add %v4511, %v4512 : tensor<32x80x14x14xf32>
    %v4514 = stablehlo.rsqrt %v4513 : tensor<32x80x14x14xf32>
    %v4515 = stablehlo.multiply %v4507, %v4514 : tensor<32x80x14x14xf32>
    %v4516 = stablehlo.reshape %v4429 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4517 = stablehlo.multiply %v4516, %v4515 : tensor<32x80x14x14xf32>
    %v4518 = stablehlo.reduce(%v4517 init: %v4502) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4519 = stablehlo.constant dense<0.05> : tensor<80xf32>
    %v4520 = stablehlo.multiply %v4518, %v4519 : tensor<80xf32>
    %v4521 = stablehlo.subtract %b8pg, %v4520 : tensor<80xf32>
    %v4522 = stablehlo.reshape %v4429 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4523 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4524 = stablehlo.reduce(%v4522 init: %v4523) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4525 = stablehlo.constant dense<0.05> : tensor<80xf32>
    %v4526 = stablehlo.multiply %v4524, %v4525 : tensor<80xf32>
    %v4527 = stablehlo.subtract %b8pbt, %v4526 : tensor<80xf32>
    %v4528 = stablehlo.reshape %v890 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4529 = stablehlo.reshape %v4495 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4530 = stablehlo.transpose %v4528, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4531 = stablehlo.transpose %v4529, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %v4532 = stablehlo.convolution(%v4530, %v4531)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<80x32x14x14xf32>) -> tensor<480x80x1x1xf32>
    %v4533 = stablehlo.transpose %v4532, dims = [1, 0, 2, 3] : (tensor<480x80x1x1xf32>) -> tensor<80x480x1x1xf32>
    %v4534 = stablehlo.constant dense<0.05> : tensor<80x480x1x1xf32>
    %v4535 = stablehlo.multiply %v4533, %v4534 : tensor<80x480x1x1xf32>
    %v4536 = stablehlo.subtract %b8pW, %v4535 : tensor<80x480x1x1xf32>
    %v4537 = stablehlo.reshape %v860 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4538 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4539 = stablehlo.reduce(%v4537 init: %v4538) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4540 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v4541 = stablehlo.divide %v4539, %v4540 : tensor<32x480xf32>
    %v4542 = stablehlo.dot_general %v4541, %b8zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v4543 = stablehlo.broadcast_in_dim %b8zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v4544 = stablehlo.add %v4542, %v4543 : tensor<32x20xf32>
    %v4545 = stablehlo.logistic %v4544 : tensor<32x20xf32>
    %v4546 = stablehlo.multiply %v4544, %v4545 : tensor<32x20xf32>
    %v4547 = stablehlo.dot_general %v4546, %b8zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v4548 = stablehlo.broadcast_in_dim %b8zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v4549 = stablehlo.add %v4547, %v4548 : tensor<32x480xf32>
    %v4550 = stablehlo.logistic %v4549 : tensor<32x480xf32>
    %v4551 = stablehlo.reshape %v4500 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4552 = stablehlo.broadcast_in_dim %v4550, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v4553 = stablehlo.multiply %v4552, %v4551 : tensor<32x480x14x14xf32>
    %v4554 = stablehlo.multiply %v4537, %v4551 : tensor<32x480x14x14xf32>
    %v4555 = stablehlo.reduce(%v4554 init: %v4538) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4556 = stablehlo.constant dense<1.0> : tensor<32x480xf32>
    %v4557 = stablehlo.subtract %v4556, %v4550 : tensor<32x480xf32>
    %v4558 = stablehlo.multiply %v4550, %v4557 : tensor<32x480xf32>
    %v4559 = stablehlo.multiply %v4555, %v4558 : tensor<32x480xf32>
    %v4560 = stablehlo.dot_general %v4559, %b8zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<20x480xf32>) -> tensor<32x20xf32>
    %v4561 = stablehlo.logistic %v4544 : tensor<32x20xf32>
    %v4562 = stablehlo.constant dense<1.0> : tensor<32x20xf32>
    %v4563 = stablehlo.subtract %v4562, %v4561 : tensor<32x20xf32>
    %v4564 = stablehlo.multiply %v4544, %v4563 : tensor<32x20xf32>
    %v4565 = stablehlo.add %v4562, %v4564 : tensor<32x20xf32>
    %v4566 = stablehlo.multiply %v4561, %v4565 : tensor<32x20xf32>
    %v4567 = stablehlo.multiply %v4560, %v4566 : tensor<32x20xf32>
    %v4568 = stablehlo.dot_general %v4567, %b8zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<480x20xf32>) -> tensor<32x480xf32>
    %v4569 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v4570 = stablehlo.divide %v4568, %v4569 : tensor<32x480xf32>
    %v4571 = stablehlo.broadcast_in_dim %v4570, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v4572 = stablehlo.add %v4553, %v4571 : tensor<32x480x14x14xf32>
    %v4573 = stablehlo.reshape %v4572 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4574 = stablehlo.reshape %v860 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4575 = stablehlo.reshape %v4500 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4576 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4577 = stablehlo.multiply %v4574, %v4575 : tensor<32x480x14x14xf32>
    %v4578 = stablehlo.reduce(%v4577 init: %v4576) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4579 = stablehlo.logistic %v873 : tensor<32x480xf32>
    %v4580 = stablehlo.constant dense<1.0> : tensor<32x480xf32>
    %v4581 = stablehlo.subtract %v4580, %v4579 : tensor<32x480xf32>
    %v4582 = stablehlo.multiply %v4579, %v4581 : tensor<32x480xf32>
    %v4583 = stablehlo.multiply %v4578, %v4582 : tensor<32x480xf32>
    %v4584 = stablehlo.dot_general %v870, %v4583, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<32x480xf32>) -> tensor<20x480xf32>
    %v4585 = stablehlo.constant dense<0.05> : tensor<20x480xf32>
    %v4586 = stablehlo.multiply %v4584, %v4585 : tensor<20x480xf32>
    %v4587 = stablehlo.subtract %b8zW2, %v4586 : tensor<20x480xf32>
    %v4588 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4589 = stablehlo.reduce(%v4583 init: %v4588) applies stablehlo.add across dimensions = [0] : (tensor<32x480xf32>, tensor<f32>) -> tensor<480xf32>
    %v4590 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4591 = stablehlo.multiply %v4589, %v4590 : tensor<480xf32>
    %v4592 = stablehlo.subtract %b8zb2, %v4591 : tensor<480xf32>
    %v4593 = stablehlo.reshape %v4583 : (tensor<32x480xf32>) -> tensor<32x1x480xf32>
    %v4594 = stablehlo.dot_general %v4593, %b8zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x480xf32>, tensor<20x480xf32>) -> tensor<32x1x20xf32>
    %v4595 = stablehlo.reshape %v4594 : (tensor<32x1x20xf32>) -> tensor<32x20xf32>
    %v4596 = stablehlo.logistic %v868 : tensor<32x20xf32>
    %v4597 = stablehlo.constant dense<1.0> : tensor<32x20xf32>
    %v4598 = stablehlo.subtract %v4597, %v4596 : tensor<32x20xf32>
    %v4599 = stablehlo.multiply %v868, %v4598 : tensor<32x20xf32>
    %v4600 = stablehlo.add %v4597, %v4599 : tensor<32x20xf32>
    %v4601 = stablehlo.multiply %v4596, %v4600 : tensor<32x20xf32>
    %v4602 = stablehlo.multiply %v4595, %v4601 : tensor<32x20xf32>
    %v4603 = stablehlo.dot_general %v865, %v4602, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<32x20xf32>) -> tensor<480x20xf32>
    %v4604 = stablehlo.constant dense<0.05> : tensor<480x20xf32>
    %v4605 = stablehlo.multiply %v4603, %v4604 : tensor<480x20xf32>
    %v4606 = stablehlo.subtract %b8zW1, %v4605 : tensor<480x20xf32>
    %v4607 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4608 = stablehlo.reduce(%v4602 init: %v4607) applies stablehlo.add across dimensions = [0] : (tensor<32x20xf32>, tensor<f32>) -> tensor<20xf32>
    %v4609 = stablehlo.constant dense<0.05> : tensor<20xf32>
    %v4610 = stablehlo.multiply %v4608, %v4609 : tensor<20xf32>
    %v4611 = stablehlo.subtract %b8zb1, %v4610 : tensor<20xf32>
    %v4612 = stablehlo.reshape %v4573 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4613 = stablehlo.reshape %v856 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4614 = stablehlo.logistic %v4613 : tensor<32x480x14x14xf32>
    %v4615 = stablehlo.constant dense<1.0> : tensor<32x480x14x14xf32>
    %v4616 = stablehlo.subtract %v4615, %v4614 : tensor<32x480x14x14xf32>
    %v4617 = stablehlo.multiply %v4613, %v4616 : tensor<32x480x14x14xf32>
    %v4618 = stablehlo.add %v4615, %v4617 : tensor<32x480x14x14xf32>
    %v4619 = stablehlo.multiply %v4614, %v4618 : tensor<32x480x14x14xf32>
    %v4620 = stablehlo.multiply %v4612, %v4619 : tensor<32x480x14x14xf32>
    %v4621 = stablehlo.reshape %v4620 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4622 = stablehlo.reshape %v836 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4623 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4624 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4625 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4626 = stablehlo.reduce(%v4622 init: %v4623) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4627 = stablehlo.broadcast_in_dim %v4626, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4628 = stablehlo.divide %v4627, %v4624 : tensor<32x480x14x14xf32>
    %v4629 = stablehlo.subtract %v4622, %v4628 : tensor<32x480x14x14xf32>
    %v4630 = stablehlo.multiply %v4629, %v4629 : tensor<32x480x14x14xf32>
    %v4631 = stablehlo.reduce(%v4630 init: %v4623) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4632 = stablehlo.broadcast_in_dim %v4631, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4633 = stablehlo.divide %v4632, %v4624 : tensor<32x480x14x14xf32>
    %v4634 = stablehlo.add %v4633, %v4625 : tensor<32x480x14x14xf32>
    %v4635 = stablehlo.rsqrt %v4634 : tensor<32x480x14x14xf32>
    %v4636 = stablehlo.multiply %v4629, %v4635 : tensor<32x480x14x14xf32>
    %v4637 = stablehlo.broadcast_in_dim %b8dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4638 = stablehlo.reshape %v4621 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4639 = stablehlo.multiply %v4637, %v4638 : tensor<32x480x14x14xf32>
    %v4640 = stablehlo.reduce(%v4639 init: %v4623) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4641 = stablehlo.broadcast_in_dim %v4640, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4642 = stablehlo.multiply %v4636, %v4639 : tensor<32x480x14x14xf32>
    %v4643 = stablehlo.reduce(%v4642 init: %v4623) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4644 = stablehlo.broadcast_in_dim %v4643, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4645 = stablehlo.multiply %v4639, %v4624 : tensor<32x480x14x14xf32>
    %v4646 = stablehlo.subtract %v4645, %v4641 : tensor<32x480x14x14xf32>
    %v4647 = stablehlo.multiply %v4636, %v4644 : tensor<32x480x14x14xf32>
    %v4648 = stablehlo.subtract %v4646, %v4647 : tensor<32x480x14x14xf32>
    %v4649 = stablehlo.divide %v4635, %v4624 : tensor<32x480x14x14xf32>
    %v4650 = stablehlo.multiply %v4649, %v4648 : tensor<32x480x14x14xf32>
    %v4651 = stablehlo.reshape %v4650 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4652 = stablehlo.reshape %v4651 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4653 = stablehlo.reverse %b8dW, dims = [2, 3] : tensor<480x1x3x3xf32>
    %v4654 = stablehlo.convolution(%v4652, %v4653)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %v4655 = stablehlo.reshape %v4654 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4656 = stablehlo.reshape %v836 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4657 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4658 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4659 = stablehlo.reduce(%v4656 init: %v4657) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4660 = stablehlo.broadcast_in_dim %v4659, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4661 = stablehlo.divide %v4660, %v4658 : tensor<32x480x14x14xf32>
    %v4662 = stablehlo.subtract %v4656, %v4661 : tensor<32x480x14x14xf32>
    %v4663 = stablehlo.multiply %v4662, %v4662 : tensor<32x480x14x14xf32>
    %v4664 = stablehlo.reduce(%v4663 init: %v4657) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4665 = stablehlo.broadcast_in_dim %v4664, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4666 = stablehlo.divide %v4665, %v4658 : tensor<32x480x14x14xf32>
    %v4667 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4668 = stablehlo.add %v4666, %v4667 : tensor<32x480x14x14xf32>
    %v4669 = stablehlo.rsqrt %v4668 : tensor<32x480x14x14xf32>
    %v4670 = stablehlo.multiply %v4662, %v4669 : tensor<32x480x14x14xf32>
    %v4671 = stablehlo.reshape %v4621 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4672 = stablehlo.multiply %v4671, %v4670 : tensor<32x480x14x14xf32>
    %v4673 = stablehlo.reduce(%v4672 init: %v4657) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4674 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4675 = stablehlo.multiply %v4673, %v4674 : tensor<480xf32>
    %v4676 = stablehlo.subtract %b8dg, %v4675 : tensor<480xf32>
    %v4677 = stablehlo.reshape %v4621 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4678 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4679 = stablehlo.reduce(%v4677 init: %v4678) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4680 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4681 = stablehlo.multiply %v4679, %v4680 : tensor<480xf32>
    %v4682 = stablehlo.subtract %b8dbt, %v4681 : tensor<480xf32>
    %v4683 = stablehlo.reshape %v831 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4684 = stablehlo.reshape %v4651 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4685 = stablehlo.transpose %v4683, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4686 = stablehlo.transpose %v4684, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4687 = stablehlo.convolution(%v4685, %v4686)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 480 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<1x480x3x3xf32>
    %v4688 = stablehlo.reshape %v4687 : (tensor<1x480x3x3xf32>) -> tensor<480x1x3x3xf32>
    %v4689 = stablehlo.constant dense<0.05> : tensor<480x1x3x3xf32>
    %v4690 = stablehlo.multiply %v4688, %v4689 : tensor<480x1x3x3xf32>
    %v4691 = stablehlo.subtract %b8dW, %v4690 : tensor<480x1x3x3xf32>
    %v4692 = stablehlo.reshape %v4655 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4693 = stablehlo.reshape %v827 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4694 = stablehlo.logistic %v4693 : tensor<32x480x14x14xf32>
    %v4695 = stablehlo.constant dense<1.0> : tensor<32x480x14x14xf32>
    %v4696 = stablehlo.subtract %v4695, %v4694 : tensor<32x480x14x14xf32>
    %v4697 = stablehlo.multiply %v4693, %v4696 : tensor<32x480x14x14xf32>
    %v4698 = stablehlo.add %v4695, %v4697 : tensor<32x480x14x14xf32>
    %v4699 = stablehlo.multiply %v4694, %v4698 : tensor<32x480x14x14xf32>
    %v4700 = stablehlo.multiply %v4692, %v4699 : tensor<32x480x14x14xf32>
    %v4701 = stablehlo.reshape %v4700 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4702 = stablehlo.reshape %v807 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4703 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4704 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4705 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4706 = stablehlo.reduce(%v4702 init: %v4703) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4707 = stablehlo.broadcast_in_dim %v4706, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4708 = stablehlo.divide %v4707, %v4704 : tensor<32x480x14x14xf32>
    %v4709 = stablehlo.subtract %v4702, %v4708 : tensor<32x480x14x14xf32>
    %v4710 = stablehlo.multiply %v4709, %v4709 : tensor<32x480x14x14xf32>
    %v4711 = stablehlo.reduce(%v4710 init: %v4703) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4712 = stablehlo.broadcast_in_dim %v4711, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4713 = stablehlo.divide %v4712, %v4704 : tensor<32x480x14x14xf32>
    %v4714 = stablehlo.add %v4713, %v4705 : tensor<32x480x14x14xf32>
    %v4715 = stablehlo.rsqrt %v4714 : tensor<32x480x14x14xf32>
    %v4716 = stablehlo.multiply %v4709, %v4715 : tensor<32x480x14x14xf32>
    %v4717 = stablehlo.broadcast_in_dim %b8eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4718 = stablehlo.reshape %v4701 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4719 = stablehlo.multiply %v4717, %v4718 : tensor<32x480x14x14xf32>
    %v4720 = stablehlo.reduce(%v4719 init: %v4703) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4721 = stablehlo.broadcast_in_dim %v4720, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4722 = stablehlo.multiply %v4716, %v4719 : tensor<32x480x14x14xf32>
    %v4723 = stablehlo.reduce(%v4722 init: %v4703) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4724 = stablehlo.broadcast_in_dim %v4723, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4725 = stablehlo.multiply %v4719, %v4704 : tensor<32x480x14x14xf32>
    %v4726 = stablehlo.subtract %v4725, %v4721 : tensor<32x480x14x14xf32>
    %v4727 = stablehlo.multiply %v4716, %v4724 : tensor<32x480x14x14xf32>
    %v4728 = stablehlo.subtract %v4726, %v4727 : tensor<32x480x14x14xf32>
    %v4729 = stablehlo.divide %v4715, %v4704 : tensor<32x480x14x14xf32>
    %v4730 = stablehlo.multiply %v4729, %v4728 : tensor<32x480x14x14xf32>
    %v4731 = stablehlo.reshape %v4730 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4732 = stablehlo.reshape %v4731 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4733 = stablehlo.reverse %b8eW, dims = [2, 3] : tensor<480x80x1x1xf32>
    %v4734 = stablehlo.transpose %v4733, dims = [1, 0, 2, 3] : (tensor<480x80x1x1xf32>) -> tensor<80x480x1x1xf32>
    %v4735 = stablehlo.convolution(%v4732, %v4734)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v4736 = stablehlo.reshape %v4735 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v4737 = stablehlo.reshape %v807 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4738 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4739 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4740 = stablehlo.reduce(%v4737 init: %v4738) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4741 = stablehlo.broadcast_in_dim %v4740, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4742 = stablehlo.divide %v4741, %v4739 : tensor<32x480x14x14xf32>
    %v4743 = stablehlo.subtract %v4737, %v4742 : tensor<32x480x14x14xf32>
    %v4744 = stablehlo.multiply %v4743, %v4743 : tensor<32x480x14x14xf32>
    %v4745 = stablehlo.reduce(%v4744 init: %v4738) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4746 = stablehlo.broadcast_in_dim %v4745, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4747 = stablehlo.divide %v4746, %v4739 : tensor<32x480x14x14xf32>
    %v4748 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4749 = stablehlo.add %v4747, %v4748 : tensor<32x480x14x14xf32>
    %v4750 = stablehlo.rsqrt %v4749 : tensor<32x480x14x14xf32>
    %v4751 = stablehlo.multiply %v4743, %v4750 : tensor<32x480x14x14xf32>
    %v4752 = stablehlo.reshape %v4701 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4753 = stablehlo.multiply %v4752, %v4751 : tensor<32x480x14x14xf32>
    %v4754 = stablehlo.reduce(%v4753 init: %v4738) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4755 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4756 = stablehlo.multiply %v4754, %v4755 : tensor<480xf32>
    %v4757 = stablehlo.subtract %b8eg, %v4756 : tensor<480xf32>
    %v4758 = stablehlo.reshape %v4701 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4759 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4760 = stablehlo.reduce(%v4758 init: %v4759) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4761 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4762 = stablehlo.multiply %v4760, %v4761 : tensor<480xf32>
    %v4763 = stablehlo.subtract %b8ebt, %v4762 : tensor<480xf32>
    %v4764 = stablehlo.reshape %v802 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4765 = stablehlo.reshape %v4731 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4766 = stablehlo.transpose %v4764, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %v4767 = stablehlo.transpose %v4765, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4768 = stablehlo.convolution(%v4766, %v4767)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<80x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<80x480x1x1xf32>
    %v4769 = stablehlo.transpose %v4768, dims = [1, 0, 2, 3] : (tensor<80x480x1x1xf32>) -> tensor<480x80x1x1xf32>
    %v4770 = stablehlo.constant dense<0.05> : tensor<480x80x1x1xf32>
    %v4771 = stablehlo.multiply %v4769, %v4770 : tensor<480x80x1x1xf32>
    %v4772 = stablehlo.subtract %b8eW, %v4771 : tensor<480x80x1x1xf32>
    %v4773 = stablehlo.reshape %v4736 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4774 = stablehlo.reshape %v4429 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4775 = stablehlo.add %v4773, %v4774 : tensor<32x80x14x14xf32>
    %v4776 = stablehlo.reshape %v4775 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v4777 = stablehlo.reshape %v778 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4778 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4779 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v4780 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v4781 = stablehlo.reduce(%v4777 init: %v4778) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4782 = stablehlo.broadcast_in_dim %v4781, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4783 = stablehlo.divide %v4782, %v4779 : tensor<32x80x14x14xf32>
    %v4784 = stablehlo.subtract %v4777, %v4783 : tensor<32x80x14x14xf32>
    %v4785 = stablehlo.multiply %v4784, %v4784 : tensor<32x80x14x14xf32>
    %v4786 = stablehlo.reduce(%v4785 init: %v4778) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4787 = stablehlo.broadcast_in_dim %v4786, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4788 = stablehlo.divide %v4787, %v4779 : tensor<32x80x14x14xf32>
    %v4789 = stablehlo.add %v4788, %v4780 : tensor<32x80x14x14xf32>
    %v4790 = stablehlo.rsqrt %v4789 : tensor<32x80x14x14xf32>
    %v4791 = stablehlo.multiply %v4784, %v4790 : tensor<32x80x14x14xf32>
    %v4792 = stablehlo.broadcast_in_dim %b7pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4793 = stablehlo.reshape %v4776 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4794 = stablehlo.multiply %v4792, %v4793 : tensor<32x80x14x14xf32>
    %v4795 = stablehlo.reduce(%v4794 init: %v4778) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4796 = stablehlo.broadcast_in_dim %v4795, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4797 = stablehlo.multiply %v4791, %v4794 : tensor<32x80x14x14xf32>
    %v4798 = stablehlo.reduce(%v4797 init: %v4778) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4799 = stablehlo.broadcast_in_dim %v4798, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4800 = stablehlo.multiply %v4794, %v4779 : tensor<32x80x14x14xf32>
    %v4801 = stablehlo.subtract %v4800, %v4796 : tensor<32x80x14x14xf32>
    %v4802 = stablehlo.multiply %v4791, %v4799 : tensor<32x80x14x14xf32>
    %v4803 = stablehlo.subtract %v4801, %v4802 : tensor<32x80x14x14xf32>
    %v4804 = stablehlo.divide %v4790, %v4779 : tensor<32x80x14x14xf32>
    %v4805 = stablehlo.multiply %v4804, %v4803 : tensor<32x80x14x14xf32>
    %v4806 = stablehlo.reshape %v4805 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v4807 = stablehlo.reshape %v4806 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4808 = stablehlo.reverse %b7pW, dims = [2, 3] : tensor<80x480x1x1xf32>
    %v4809 = stablehlo.transpose %v4808, dims = [1, 0, 2, 3] : (tensor<80x480x1x1xf32>) -> tensor<480x80x1x1xf32>
    %v4810 = stablehlo.convolution(%v4807, %v4809)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v4811 = stablehlo.reshape %v4810 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4812 = stablehlo.reshape %v778 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4813 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4814 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v4815 = stablehlo.reduce(%v4812 init: %v4813) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4816 = stablehlo.broadcast_in_dim %v4815, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4817 = stablehlo.divide %v4816, %v4814 : tensor<32x80x14x14xf32>
    %v4818 = stablehlo.subtract %v4812, %v4817 : tensor<32x80x14x14xf32>
    %v4819 = stablehlo.multiply %v4818, %v4818 : tensor<32x80x14x14xf32>
    %v4820 = stablehlo.reduce(%v4819 init: %v4813) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4821 = stablehlo.broadcast_in_dim %v4820, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4822 = stablehlo.divide %v4821, %v4814 : tensor<32x80x14x14xf32>
    %v4823 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v4824 = stablehlo.add %v4822, %v4823 : tensor<32x80x14x14xf32>
    %v4825 = stablehlo.rsqrt %v4824 : tensor<32x80x14x14xf32>
    %v4826 = stablehlo.multiply %v4818, %v4825 : tensor<32x80x14x14xf32>
    %v4827 = stablehlo.reshape %v4776 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4828 = stablehlo.multiply %v4827, %v4826 : tensor<32x80x14x14xf32>
    %v4829 = stablehlo.reduce(%v4828 init: %v4813) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4830 = stablehlo.constant dense<0.05> : tensor<80xf32>
    %v4831 = stablehlo.multiply %v4829, %v4830 : tensor<80xf32>
    %v4832 = stablehlo.subtract %b7pg, %v4831 : tensor<80xf32>
    %v4833 = stablehlo.reshape %v4776 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4834 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4835 = stablehlo.reduce(%v4833 init: %v4834) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4836 = stablehlo.constant dense<0.05> : tensor<80xf32>
    %v4837 = stablehlo.multiply %v4835, %v4836 : tensor<80xf32>
    %v4838 = stablehlo.subtract %b7pbt, %v4837 : tensor<80xf32>
    %v4839 = stablehlo.reshape %v773 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4840 = stablehlo.reshape %v4806 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4841 = stablehlo.transpose %v4839, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4842 = stablehlo.transpose %v4840, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %v4843 = stablehlo.convolution(%v4841, %v4842)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<80x32x14x14xf32>) -> tensor<480x80x1x1xf32>
    %v4844 = stablehlo.transpose %v4843, dims = [1, 0, 2, 3] : (tensor<480x80x1x1xf32>) -> tensor<80x480x1x1xf32>
    %v4845 = stablehlo.constant dense<0.05> : tensor<80x480x1x1xf32>
    %v4846 = stablehlo.multiply %v4844, %v4845 : tensor<80x480x1x1xf32>
    %v4847 = stablehlo.subtract %b7pW, %v4846 : tensor<80x480x1x1xf32>
    %v4848 = stablehlo.reshape %v743 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4849 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4850 = stablehlo.reduce(%v4848 init: %v4849) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4851 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v4852 = stablehlo.divide %v4850, %v4851 : tensor<32x480xf32>
    %v4853 = stablehlo.dot_general %v4852, %b7zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v4854 = stablehlo.broadcast_in_dim %b7zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v4855 = stablehlo.add %v4853, %v4854 : tensor<32x20xf32>
    %v4856 = stablehlo.logistic %v4855 : tensor<32x20xf32>
    %v4857 = stablehlo.multiply %v4855, %v4856 : tensor<32x20xf32>
    %v4858 = stablehlo.dot_general %v4857, %b7zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v4859 = stablehlo.broadcast_in_dim %b7zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v4860 = stablehlo.add %v4858, %v4859 : tensor<32x480xf32>
    %v4861 = stablehlo.logistic %v4860 : tensor<32x480xf32>
    %v4862 = stablehlo.reshape %v4811 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4863 = stablehlo.broadcast_in_dim %v4861, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v4864 = stablehlo.multiply %v4863, %v4862 : tensor<32x480x14x14xf32>
    %v4865 = stablehlo.multiply %v4848, %v4862 : tensor<32x480x14x14xf32>
    %v4866 = stablehlo.reduce(%v4865 init: %v4849) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4867 = stablehlo.constant dense<1.0> : tensor<32x480xf32>
    %v4868 = stablehlo.subtract %v4867, %v4861 : tensor<32x480xf32>
    %v4869 = stablehlo.multiply %v4861, %v4868 : tensor<32x480xf32>
    %v4870 = stablehlo.multiply %v4866, %v4869 : tensor<32x480xf32>
    %v4871 = stablehlo.dot_general %v4870, %b7zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<20x480xf32>) -> tensor<32x20xf32>
    %v4872 = stablehlo.logistic %v4855 : tensor<32x20xf32>
    %v4873 = stablehlo.constant dense<1.0> : tensor<32x20xf32>
    %v4874 = stablehlo.subtract %v4873, %v4872 : tensor<32x20xf32>
    %v4875 = stablehlo.multiply %v4855, %v4874 : tensor<32x20xf32>
    %v4876 = stablehlo.add %v4873, %v4875 : tensor<32x20xf32>
    %v4877 = stablehlo.multiply %v4872, %v4876 : tensor<32x20xf32>
    %v4878 = stablehlo.multiply %v4871, %v4877 : tensor<32x20xf32>
    %v4879 = stablehlo.dot_general %v4878, %b7zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<480x20xf32>) -> tensor<32x480xf32>
    %v4880 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v4881 = stablehlo.divide %v4879, %v4880 : tensor<32x480xf32>
    %v4882 = stablehlo.broadcast_in_dim %v4881, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v4883 = stablehlo.add %v4864, %v4882 : tensor<32x480x14x14xf32>
    %v4884 = stablehlo.reshape %v4883 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4885 = stablehlo.reshape %v743 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4886 = stablehlo.reshape %v4811 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4887 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4888 = stablehlo.multiply %v4885, %v4886 : tensor<32x480x14x14xf32>
    %v4889 = stablehlo.reduce(%v4888 init: %v4887) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4890 = stablehlo.logistic %v756 : tensor<32x480xf32>
    %v4891 = stablehlo.constant dense<1.0> : tensor<32x480xf32>
    %v4892 = stablehlo.subtract %v4891, %v4890 : tensor<32x480xf32>
    %v4893 = stablehlo.multiply %v4890, %v4892 : tensor<32x480xf32>
    %v4894 = stablehlo.multiply %v4889, %v4893 : tensor<32x480xf32>
    %v4895 = stablehlo.dot_general %v753, %v4894, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<32x480xf32>) -> tensor<20x480xf32>
    %v4896 = stablehlo.constant dense<0.05> : tensor<20x480xf32>
    %v4897 = stablehlo.multiply %v4895, %v4896 : tensor<20x480xf32>
    %v4898 = stablehlo.subtract %b7zW2, %v4897 : tensor<20x480xf32>
    %v4899 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4900 = stablehlo.reduce(%v4894 init: %v4899) applies stablehlo.add across dimensions = [0] : (tensor<32x480xf32>, tensor<f32>) -> tensor<480xf32>
    %v4901 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4902 = stablehlo.multiply %v4900, %v4901 : tensor<480xf32>
    %v4903 = stablehlo.subtract %b7zb2, %v4902 : tensor<480xf32>
    %v4904 = stablehlo.reshape %v4894 : (tensor<32x480xf32>) -> tensor<32x1x480xf32>
    %v4905 = stablehlo.dot_general %v4904, %b7zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x480xf32>, tensor<20x480xf32>) -> tensor<32x1x20xf32>
    %v4906 = stablehlo.reshape %v4905 : (tensor<32x1x20xf32>) -> tensor<32x20xf32>
    %v4907 = stablehlo.logistic %v751 : tensor<32x20xf32>
    %v4908 = stablehlo.constant dense<1.0> : tensor<32x20xf32>
    %v4909 = stablehlo.subtract %v4908, %v4907 : tensor<32x20xf32>
    %v4910 = stablehlo.multiply %v751, %v4909 : tensor<32x20xf32>
    %v4911 = stablehlo.add %v4908, %v4910 : tensor<32x20xf32>
    %v4912 = stablehlo.multiply %v4907, %v4911 : tensor<32x20xf32>
    %v4913 = stablehlo.multiply %v4906, %v4912 : tensor<32x20xf32>
    %v4914 = stablehlo.dot_general %v748, %v4913, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<32x20xf32>) -> tensor<480x20xf32>
    %v4915 = stablehlo.constant dense<0.05> : tensor<480x20xf32>
    %v4916 = stablehlo.multiply %v4914, %v4915 : tensor<480x20xf32>
    %v4917 = stablehlo.subtract %b7zW1, %v4916 : tensor<480x20xf32>
    %v4918 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4919 = stablehlo.reduce(%v4913 init: %v4918) applies stablehlo.add across dimensions = [0] : (tensor<32x20xf32>, tensor<f32>) -> tensor<20xf32>
    %v4920 = stablehlo.constant dense<0.05> : tensor<20xf32>
    %v4921 = stablehlo.multiply %v4919, %v4920 : tensor<20xf32>
    %v4922 = stablehlo.subtract %b7zb1, %v4921 : tensor<20xf32>
    %v4923 = stablehlo.reshape %v4884 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4924 = stablehlo.reshape %v739 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4925 = stablehlo.logistic %v4924 : tensor<32x480x14x14xf32>
    %v4926 = stablehlo.constant dense<1.0> : tensor<32x480x14x14xf32>
    %v4927 = stablehlo.subtract %v4926, %v4925 : tensor<32x480x14x14xf32>
    %v4928 = stablehlo.multiply %v4924, %v4927 : tensor<32x480x14x14xf32>
    %v4929 = stablehlo.add %v4926, %v4928 : tensor<32x480x14x14xf32>
    %v4930 = stablehlo.multiply %v4925, %v4929 : tensor<32x480x14x14xf32>
    %v4931 = stablehlo.multiply %v4923, %v4930 : tensor<32x480x14x14xf32>
    %v4932 = stablehlo.reshape %v4931 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4933 = stablehlo.reshape %v719 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4934 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4935 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4936 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4937 = stablehlo.reduce(%v4933 init: %v4934) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4938 = stablehlo.broadcast_in_dim %v4937, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4939 = stablehlo.divide %v4938, %v4935 : tensor<32x480x14x14xf32>
    %v4940 = stablehlo.subtract %v4933, %v4939 : tensor<32x480x14x14xf32>
    %v4941 = stablehlo.multiply %v4940, %v4940 : tensor<32x480x14x14xf32>
    %v4942 = stablehlo.reduce(%v4941 init: %v4934) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4943 = stablehlo.broadcast_in_dim %v4942, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4944 = stablehlo.divide %v4943, %v4935 : tensor<32x480x14x14xf32>
    %v4945 = stablehlo.add %v4944, %v4936 : tensor<32x480x14x14xf32>
    %v4946 = stablehlo.rsqrt %v4945 : tensor<32x480x14x14xf32>
    %v4947 = stablehlo.multiply %v4940, %v4946 : tensor<32x480x14x14xf32>
    %v4948 = stablehlo.broadcast_in_dim %b7dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4949 = stablehlo.reshape %v4932 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4950 = stablehlo.multiply %v4948, %v4949 : tensor<32x480x14x14xf32>
    %v4951 = stablehlo.reduce(%v4950 init: %v4934) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4952 = stablehlo.broadcast_in_dim %v4951, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4953 = stablehlo.multiply %v4947, %v4950 : tensor<32x480x14x14xf32>
    %v4954 = stablehlo.reduce(%v4953 init: %v4934) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4955 = stablehlo.broadcast_in_dim %v4954, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4956 = stablehlo.multiply %v4950, %v4935 : tensor<32x480x14x14xf32>
    %v4957 = stablehlo.subtract %v4956, %v4952 : tensor<32x480x14x14xf32>
    %v4958 = stablehlo.multiply %v4947, %v4955 : tensor<32x480x14x14xf32>
    %v4959 = stablehlo.subtract %v4957, %v4958 : tensor<32x480x14x14xf32>
    %v4960 = stablehlo.divide %v4946, %v4935 : tensor<32x480x14x14xf32>
    %v4961 = stablehlo.multiply %v4960, %v4959 : tensor<32x480x14x14xf32>
    %v4962 = stablehlo.reshape %v4961 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4963 = stablehlo.reshape %v4962 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4964 = stablehlo.reverse %b7dW, dims = [2, 3] : tensor<480x1x3x3xf32>
    %v4965 = stablehlo.convolution(%v4963, %v4964)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %v4966 = stablehlo.reshape %v4965 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4967 = stablehlo.reshape %v719 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4968 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4969 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4970 = stablehlo.reduce(%v4967 init: %v4968) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4971 = stablehlo.broadcast_in_dim %v4970, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4972 = stablehlo.divide %v4971, %v4969 : tensor<32x480x14x14xf32>
    %v4973 = stablehlo.subtract %v4967, %v4972 : tensor<32x480x14x14xf32>
    %v4974 = stablehlo.multiply %v4973, %v4973 : tensor<32x480x14x14xf32>
    %v4975 = stablehlo.reduce(%v4974 init: %v4968) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4976 = stablehlo.broadcast_in_dim %v4975, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4977 = stablehlo.divide %v4976, %v4969 : tensor<32x480x14x14xf32>
    %v4978 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4979 = stablehlo.add %v4977, %v4978 : tensor<32x480x14x14xf32>
    %v4980 = stablehlo.rsqrt %v4979 : tensor<32x480x14x14xf32>
    %v4981 = stablehlo.multiply %v4973, %v4980 : tensor<32x480x14x14xf32>
    %v4982 = stablehlo.reshape %v4932 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4983 = stablehlo.multiply %v4982, %v4981 : tensor<32x480x14x14xf32>
    %v4984 = stablehlo.reduce(%v4983 init: %v4968) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4985 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4986 = stablehlo.multiply %v4984, %v4985 : tensor<480xf32>
    %v4987 = stablehlo.subtract %b7dg, %v4986 : tensor<480xf32>
    %v4988 = stablehlo.reshape %v4932 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4989 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4990 = stablehlo.reduce(%v4988 init: %v4989) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4991 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4992 = stablehlo.multiply %v4990, %v4991 : tensor<480xf32>
    %v4993 = stablehlo.subtract %b7dbt, %v4992 : tensor<480xf32>
    %v4994 = stablehlo.reshape %v714 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4995 = stablehlo.reshape %v4962 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4996 = stablehlo.transpose %v4994, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4997 = stablehlo.transpose %v4995, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4998 = stablehlo.convolution(%v4996, %v4997)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 480 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<1x480x3x3xf32>
    %v4999 = stablehlo.reshape %v4998 : (tensor<1x480x3x3xf32>) -> tensor<480x1x3x3xf32>
    %v5000 = stablehlo.constant dense<0.05> : tensor<480x1x3x3xf32>
    %v5001 = stablehlo.multiply %v4999, %v5000 : tensor<480x1x3x3xf32>
    %v5002 = stablehlo.subtract %b7dW, %v5001 : tensor<480x1x3x3xf32>
    %v5003 = stablehlo.reshape %v4966 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v5004 = stablehlo.reshape %v710 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v5005 = stablehlo.logistic %v5004 : tensor<32x480x14x14xf32>
    %v5006 = stablehlo.constant dense<1.0> : tensor<32x480x14x14xf32>
    %v5007 = stablehlo.subtract %v5006, %v5005 : tensor<32x480x14x14xf32>
    %v5008 = stablehlo.multiply %v5004, %v5007 : tensor<32x480x14x14xf32>
    %v5009 = stablehlo.add %v5006, %v5008 : tensor<32x480x14x14xf32>
    %v5010 = stablehlo.multiply %v5005, %v5009 : tensor<32x480x14x14xf32>
    %v5011 = stablehlo.multiply %v5003, %v5010 : tensor<32x480x14x14xf32>
    %v5012 = stablehlo.reshape %v5011 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v5013 = stablehlo.reshape %v690 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v5014 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5015 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v5016 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v5017 = stablehlo.reduce(%v5013 init: %v5014) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v5018 = stablehlo.broadcast_in_dim %v5017, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v5019 = stablehlo.divide %v5018, %v5015 : tensor<32x480x14x14xf32>
    %v5020 = stablehlo.subtract %v5013, %v5019 : tensor<32x480x14x14xf32>
    %v5021 = stablehlo.multiply %v5020, %v5020 : tensor<32x480x14x14xf32>
    %v5022 = stablehlo.reduce(%v5021 init: %v5014) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v5023 = stablehlo.broadcast_in_dim %v5022, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v5024 = stablehlo.divide %v5023, %v5015 : tensor<32x480x14x14xf32>
    %v5025 = stablehlo.add %v5024, %v5016 : tensor<32x480x14x14xf32>
    %v5026 = stablehlo.rsqrt %v5025 : tensor<32x480x14x14xf32>
    %v5027 = stablehlo.multiply %v5020, %v5026 : tensor<32x480x14x14xf32>
    %v5028 = stablehlo.broadcast_in_dim %b7eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v5029 = stablehlo.reshape %v5012 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v5030 = stablehlo.multiply %v5028, %v5029 : tensor<32x480x14x14xf32>
    %v5031 = stablehlo.reduce(%v5030 init: %v5014) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v5032 = stablehlo.broadcast_in_dim %v5031, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v5033 = stablehlo.multiply %v5027, %v5030 : tensor<32x480x14x14xf32>
    %v5034 = stablehlo.reduce(%v5033 init: %v5014) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v5035 = stablehlo.broadcast_in_dim %v5034, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v5036 = stablehlo.multiply %v5030, %v5015 : tensor<32x480x14x14xf32>
    %v5037 = stablehlo.subtract %v5036, %v5032 : tensor<32x480x14x14xf32>
    %v5038 = stablehlo.multiply %v5027, %v5035 : tensor<32x480x14x14xf32>
    %v5039 = stablehlo.subtract %v5037, %v5038 : tensor<32x480x14x14xf32>
    %v5040 = stablehlo.divide %v5026, %v5015 : tensor<32x480x14x14xf32>
    %v5041 = stablehlo.multiply %v5040, %v5039 : tensor<32x480x14x14xf32>
    %v5042 = stablehlo.reshape %v5041 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v5043 = stablehlo.reshape %v5042 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v5044 = stablehlo.reverse %b7eW, dims = [2, 3] : tensor<480x80x1x1xf32>
    %v5045 = stablehlo.transpose %v5044, dims = [1, 0, 2, 3] : (tensor<480x80x1x1xf32>) -> tensor<80x480x1x1xf32>
    %v5046 = stablehlo.convolution(%v5043, %v5045)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v5047 = stablehlo.reshape %v5046 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v5048 = stablehlo.reshape %v690 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v5049 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5050 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v5051 = stablehlo.reduce(%v5048 init: %v5049) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v5052 = stablehlo.broadcast_in_dim %v5051, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v5053 = stablehlo.divide %v5052, %v5050 : tensor<32x480x14x14xf32>
    %v5054 = stablehlo.subtract %v5048, %v5053 : tensor<32x480x14x14xf32>
    %v5055 = stablehlo.multiply %v5054, %v5054 : tensor<32x480x14x14xf32>
    %v5056 = stablehlo.reduce(%v5055 init: %v5049) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v5057 = stablehlo.broadcast_in_dim %v5056, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v5058 = stablehlo.divide %v5057, %v5050 : tensor<32x480x14x14xf32>
    %v5059 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v5060 = stablehlo.add %v5058, %v5059 : tensor<32x480x14x14xf32>
    %v5061 = stablehlo.rsqrt %v5060 : tensor<32x480x14x14xf32>
    %v5062 = stablehlo.multiply %v5054, %v5061 : tensor<32x480x14x14xf32>
    %v5063 = stablehlo.reshape %v5012 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v5064 = stablehlo.multiply %v5063, %v5062 : tensor<32x480x14x14xf32>
    %v5065 = stablehlo.reduce(%v5064 init: %v5049) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v5066 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v5067 = stablehlo.multiply %v5065, %v5066 : tensor<480xf32>
    %v5068 = stablehlo.subtract %b7eg, %v5067 : tensor<480xf32>
    %v5069 = stablehlo.reshape %v5012 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v5070 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5071 = stablehlo.reduce(%v5069 init: %v5070) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v5072 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v5073 = stablehlo.multiply %v5071, %v5072 : tensor<480xf32>
    %v5074 = stablehlo.subtract %b7ebt, %v5073 : tensor<480xf32>
    %v5075 = stablehlo.reshape %v685 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v5076 = stablehlo.reshape %v5042 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v5077 = stablehlo.transpose %v5075, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %v5078 = stablehlo.transpose %v5076, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v5079 = stablehlo.convolution(%v5077, %v5078)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<80x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<80x480x1x1xf32>
    %v5080 = stablehlo.transpose %v5079, dims = [1, 0, 2, 3] : (tensor<80x480x1x1xf32>) -> tensor<480x80x1x1xf32>
    %v5081 = stablehlo.constant dense<0.05> : tensor<480x80x1x1xf32>
    %v5082 = stablehlo.multiply %v5080, %v5081 : tensor<480x80x1x1xf32>
    %v5083 = stablehlo.subtract %b7eW, %v5082 : tensor<480x80x1x1xf32>
    %v5084 = stablehlo.reshape %v5047 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v5085 = stablehlo.reshape %v4776 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v5086 = stablehlo.add %v5084, %v5085 : tensor<32x80x14x14xf32>
    %v5087 = stablehlo.reshape %v5086 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v5088 = stablehlo.reshape %v665 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v5089 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5090 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v5091 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v5092 = stablehlo.reduce(%v5088 init: %v5089) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v5093 = stablehlo.broadcast_in_dim %v5092, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v5094 = stablehlo.divide %v5093, %v5090 : tensor<32x80x14x14xf32>
    %v5095 = stablehlo.subtract %v5088, %v5094 : tensor<32x80x14x14xf32>
    %v5096 = stablehlo.multiply %v5095, %v5095 : tensor<32x80x14x14xf32>
    %v5097 = stablehlo.reduce(%v5096 init: %v5089) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v5098 = stablehlo.broadcast_in_dim %v5097, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v5099 = stablehlo.divide %v5098, %v5090 : tensor<32x80x14x14xf32>
    %v5100 = stablehlo.add %v5099, %v5091 : tensor<32x80x14x14xf32>
    %v5101 = stablehlo.rsqrt %v5100 : tensor<32x80x14x14xf32>
    %v5102 = stablehlo.multiply %v5095, %v5101 : tensor<32x80x14x14xf32>
    %v5103 = stablehlo.broadcast_in_dim %b6pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v5104 = stablehlo.reshape %v5087 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v5105 = stablehlo.multiply %v5103, %v5104 : tensor<32x80x14x14xf32>
    %v5106 = stablehlo.reduce(%v5105 init: %v5089) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v5107 = stablehlo.broadcast_in_dim %v5106, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v5108 = stablehlo.multiply %v5102, %v5105 : tensor<32x80x14x14xf32>
    %v5109 = stablehlo.reduce(%v5108 init: %v5089) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v5110 = stablehlo.broadcast_in_dim %v5109, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v5111 = stablehlo.multiply %v5105, %v5090 : tensor<32x80x14x14xf32>
    %v5112 = stablehlo.subtract %v5111, %v5107 : tensor<32x80x14x14xf32>
    %v5113 = stablehlo.multiply %v5102, %v5110 : tensor<32x80x14x14xf32>
    %v5114 = stablehlo.subtract %v5112, %v5113 : tensor<32x80x14x14xf32>
    %v5115 = stablehlo.divide %v5101, %v5090 : tensor<32x80x14x14xf32>
    %v5116 = stablehlo.multiply %v5115, %v5114 : tensor<32x80x14x14xf32>
    %v5117 = stablehlo.reshape %v5116 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v5118 = stablehlo.reshape %v5117 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v5119 = stablehlo.reverse %b6pW, dims = [2, 3] : tensor<80x240x1x1xf32>
    %v5120 = stablehlo.transpose %v5119, dims = [1, 0, 2, 3] : (tensor<80x240x1x1xf32>) -> tensor<240x80x1x1xf32>
    %v5121 = stablehlo.convolution(%v5118, %v5120)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<240x80x1x1xf32>) -> tensor<32x240x14x14xf32>
    %v5122 = stablehlo.reshape %v5121 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v5123 = stablehlo.reshape %v665 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v5124 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5125 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v5126 = stablehlo.reduce(%v5123 init: %v5124) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v5127 = stablehlo.broadcast_in_dim %v5126, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v5128 = stablehlo.divide %v5127, %v5125 : tensor<32x80x14x14xf32>
    %v5129 = stablehlo.subtract %v5123, %v5128 : tensor<32x80x14x14xf32>
    %v5130 = stablehlo.multiply %v5129, %v5129 : tensor<32x80x14x14xf32>
    %v5131 = stablehlo.reduce(%v5130 init: %v5124) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v5132 = stablehlo.broadcast_in_dim %v5131, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v5133 = stablehlo.divide %v5132, %v5125 : tensor<32x80x14x14xf32>
    %v5134 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v5135 = stablehlo.add %v5133, %v5134 : tensor<32x80x14x14xf32>
    %v5136 = stablehlo.rsqrt %v5135 : tensor<32x80x14x14xf32>
    %v5137 = stablehlo.multiply %v5129, %v5136 : tensor<32x80x14x14xf32>
    %v5138 = stablehlo.reshape %v5087 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v5139 = stablehlo.multiply %v5138, %v5137 : tensor<32x80x14x14xf32>
    %v5140 = stablehlo.reduce(%v5139 init: %v5124) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v5141 = stablehlo.constant dense<0.05> : tensor<80xf32>
    %v5142 = stablehlo.multiply %v5140, %v5141 : tensor<80xf32>
    %v5143 = stablehlo.subtract %b6pg, %v5142 : tensor<80xf32>
    %v5144 = stablehlo.reshape %v5087 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v5145 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5146 = stablehlo.reduce(%v5144 init: %v5145) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v5147 = stablehlo.constant dense<0.05> : tensor<80xf32>
    %v5148 = stablehlo.multiply %v5146, %v5147 : tensor<80xf32>
    %v5149 = stablehlo.subtract %b6pbt, %v5148 : tensor<80xf32>
    %v5150 = stablehlo.reshape %v660 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5151 = stablehlo.reshape %v5117 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v5152 = stablehlo.transpose %v5150, dims = [1, 0, 2, 3] : (tensor<32x240x14x14xf32>) -> tensor<240x32x14x14xf32>
    %v5153 = stablehlo.transpose %v5151, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %v5154 = stablehlo.convolution(%v5152, %v5153)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<240x32x14x14xf32>, tensor<80x32x14x14xf32>) -> tensor<240x80x1x1xf32>
    %v5155 = stablehlo.transpose %v5154, dims = [1, 0, 2, 3] : (tensor<240x80x1x1xf32>) -> tensor<80x240x1x1xf32>
    %v5156 = stablehlo.constant dense<0.05> : tensor<80x240x1x1xf32>
    %v5157 = stablehlo.multiply %v5155, %v5156 : tensor<80x240x1x1xf32>
    %v5158 = stablehlo.subtract %b6pW, %v5157 : tensor<80x240x1x1xf32>
    %v5159 = stablehlo.reshape %v630 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5160 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5161 = stablehlo.reduce(%v5159 init: %v5160) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v5162 = stablehlo.constant dense<196.0> : tensor<32x240xf32>
    %v5163 = stablehlo.divide %v5161, %v5162 : tensor<32x240xf32>
    %v5164 = stablehlo.dot_general %v5163, %b6zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v5165 = stablehlo.broadcast_in_dim %b6zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v5166 = stablehlo.add %v5164, %v5165 : tensor<32x10xf32>
    %v5167 = stablehlo.logistic %v5166 : tensor<32x10xf32>
    %v5168 = stablehlo.multiply %v5166, %v5167 : tensor<32x10xf32>
    %v5169 = stablehlo.dot_general %v5168, %b6zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v5170 = stablehlo.broadcast_in_dim %b6zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v5171 = stablehlo.add %v5169, %v5170 : tensor<32x240xf32>
    %v5172 = stablehlo.logistic %v5171 : tensor<32x240xf32>
    %v5173 = stablehlo.reshape %v5122 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5174 = stablehlo.broadcast_in_dim %v5172, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x14x14xf32>
    %v5175 = stablehlo.multiply %v5174, %v5173 : tensor<32x240x14x14xf32>
    %v5176 = stablehlo.multiply %v5159, %v5173 : tensor<32x240x14x14xf32>
    %v5177 = stablehlo.reduce(%v5176 init: %v5160) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v5178 = stablehlo.constant dense<1.0> : tensor<32x240xf32>
    %v5179 = stablehlo.subtract %v5178, %v5172 : tensor<32x240xf32>
    %v5180 = stablehlo.multiply %v5172, %v5179 : tensor<32x240xf32>
    %v5181 = stablehlo.multiply %v5177, %v5180 : tensor<32x240xf32>
    %v5182 = stablehlo.dot_general %v5181, %b6zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<10x240xf32>) -> tensor<32x10xf32>
    %v5183 = stablehlo.logistic %v5166 : tensor<32x10xf32>
    %v5184 = stablehlo.constant dense<1.0> : tensor<32x10xf32>
    %v5185 = stablehlo.subtract %v5184, %v5183 : tensor<32x10xf32>
    %v5186 = stablehlo.multiply %v5166, %v5185 : tensor<32x10xf32>
    %v5187 = stablehlo.add %v5184, %v5186 : tensor<32x10xf32>
    %v5188 = stablehlo.multiply %v5183, %v5187 : tensor<32x10xf32>
    %v5189 = stablehlo.multiply %v5182, %v5188 : tensor<32x10xf32>
    %v5190 = stablehlo.dot_general %v5189, %b6zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<240x10xf32>) -> tensor<32x240xf32>
    %v5191 = stablehlo.constant dense<196.0> : tensor<32x240xf32>
    %v5192 = stablehlo.divide %v5190, %v5191 : tensor<32x240xf32>
    %v5193 = stablehlo.broadcast_in_dim %v5192, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x14x14xf32>
    %v5194 = stablehlo.add %v5175, %v5193 : tensor<32x240x14x14xf32>
    %v5195 = stablehlo.reshape %v5194 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v5196 = stablehlo.reshape %v630 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5197 = stablehlo.reshape %v5122 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5198 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5199 = stablehlo.multiply %v5196, %v5197 : tensor<32x240x14x14xf32>
    %v5200 = stablehlo.reduce(%v5199 init: %v5198) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v5201 = stablehlo.logistic %v643 : tensor<32x240xf32>
    %v5202 = stablehlo.constant dense<1.0> : tensor<32x240xf32>
    %v5203 = stablehlo.subtract %v5202, %v5201 : tensor<32x240xf32>
    %v5204 = stablehlo.multiply %v5201, %v5203 : tensor<32x240xf32>
    %v5205 = stablehlo.multiply %v5200, %v5204 : tensor<32x240xf32>
    %v5206 = stablehlo.dot_general %v640, %v5205, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<32x240xf32>) -> tensor<10x240xf32>
    %v5207 = stablehlo.constant dense<0.05> : tensor<10x240xf32>
    %v5208 = stablehlo.multiply %v5206, %v5207 : tensor<10x240xf32>
    %v5209 = stablehlo.subtract %b6zW2, %v5208 : tensor<10x240xf32>
    %v5210 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5211 = stablehlo.reduce(%v5205 init: %v5210) applies stablehlo.add across dimensions = [0] : (tensor<32x240xf32>, tensor<f32>) -> tensor<240xf32>
    %v5212 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5213 = stablehlo.multiply %v5211, %v5212 : tensor<240xf32>
    %v5214 = stablehlo.subtract %b6zb2, %v5213 : tensor<240xf32>
    %v5215 = stablehlo.reshape %v5205 : (tensor<32x240xf32>) -> tensor<32x1x240xf32>
    %v5216 = stablehlo.dot_general %v5215, %b6zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x240xf32>, tensor<10x240xf32>) -> tensor<32x1x10xf32>
    %v5217 = stablehlo.reshape %v5216 : (tensor<32x1x10xf32>) -> tensor<32x10xf32>
    %v5218 = stablehlo.logistic %v638 : tensor<32x10xf32>
    %v5219 = stablehlo.constant dense<1.0> : tensor<32x10xf32>
    %v5220 = stablehlo.subtract %v5219, %v5218 : tensor<32x10xf32>
    %v5221 = stablehlo.multiply %v638, %v5220 : tensor<32x10xf32>
    %v5222 = stablehlo.add %v5219, %v5221 : tensor<32x10xf32>
    %v5223 = stablehlo.multiply %v5218, %v5222 : tensor<32x10xf32>
    %v5224 = stablehlo.multiply %v5217, %v5223 : tensor<32x10xf32>
    %v5225 = stablehlo.dot_general %v635, %v5224, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<32x10xf32>) -> tensor<240x10xf32>
    %v5226 = stablehlo.constant dense<0.05> : tensor<240x10xf32>
    %v5227 = stablehlo.multiply %v5225, %v5226 : tensor<240x10xf32>
    %v5228 = stablehlo.subtract %b6zW1, %v5227 : tensor<240x10xf32>
    %v5229 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5230 = stablehlo.reduce(%v5224 init: %v5229) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v5231 = stablehlo.constant dense<0.05> : tensor<10xf32>
    %v5232 = stablehlo.multiply %v5230, %v5231 : tensor<10xf32>
    %v5233 = stablehlo.subtract %b6zb1, %v5232 : tensor<10xf32>
    %v5234 = stablehlo.reshape %v5195 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5235 = stablehlo.reshape %v626 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5236 = stablehlo.logistic %v5235 : tensor<32x240x14x14xf32>
    %v5237 = stablehlo.constant dense<1.0> : tensor<32x240x14x14xf32>
    %v5238 = stablehlo.subtract %v5237, %v5236 : tensor<32x240x14x14xf32>
    %v5239 = stablehlo.multiply %v5235, %v5238 : tensor<32x240x14x14xf32>
    %v5240 = stablehlo.add %v5237, %v5239 : tensor<32x240x14x14xf32>
    %v5241 = stablehlo.multiply %v5236, %v5240 : tensor<32x240x14x14xf32>
    %v5242 = stablehlo.multiply %v5234, %v5241 : tensor<32x240x14x14xf32>
    %v5243 = stablehlo.reshape %v5242 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v5244 = stablehlo.reshape %v606 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5245 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5246 = stablehlo.constant dense<6272.0> : tensor<32x240x14x14xf32>
    %v5247 = stablehlo.constant dense<1.0e-5> : tensor<32x240x14x14xf32>
    %v5248 = stablehlo.reduce(%v5244 init: %v5245) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5249 = stablehlo.broadcast_in_dim %v5248, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v5250 = stablehlo.divide %v5249, %v5246 : tensor<32x240x14x14xf32>
    %v5251 = stablehlo.subtract %v5244, %v5250 : tensor<32x240x14x14xf32>
    %v5252 = stablehlo.multiply %v5251, %v5251 : tensor<32x240x14x14xf32>
    %v5253 = stablehlo.reduce(%v5252 init: %v5245) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5254 = stablehlo.broadcast_in_dim %v5253, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v5255 = stablehlo.divide %v5254, %v5246 : tensor<32x240x14x14xf32>
    %v5256 = stablehlo.add %v5255, %v5247 : tensor<32x240x14x14xf32>
    %v5257 = stablehlo.rsqrt %v5256 : tensor<32x240x14x14xf32>
    %v5258 = stablehlo.multiply %v5251, %v5257 : tensor<32x240x14x14xf32>
    %v5259 = stablehlo.broadcast_in_dim %b6dg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v5260 = stablehlo.reshape %v5243 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5261 = stablehlo.multiply %v5259, %v5260 : tensor<32x240x14x14xf32>
    %v5262 = stablehlo.reduce(%v5261 init: %v5245) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5263 = stablehlo.broadcast_in_dim %v5262, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v5264 = stablehlo.multiply %v5258, %v5261 : tensor<32x240x14x14xf32>
    %v5265 = stablehlo.reduce(%v5264 init: %v5245) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5266 = stablehlo.broadcast_in_dim %v5265, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v5267 = stablehlo.multiply %v5261, %v5246 : tensor<32x240x14x14xf32>
    %v5268 = stablehlo.subtract %v5267, %v5263 : tensor<32x240x14x14xf32>
    %v5269 = stablehlo.multiply %v5258, %v5266 : tensor<32x240x14x14xf32>
    %v5270 = stablehlo.subtract %v5268, %v5269 : tensor<32x240x14x14xf32>
    %v5271 = stablehlo.divide %v5257, %v5246 : tensor<32x240x14x14xf32>
    %v5272 = stablehlo.multiply %v5271, %v5270 : tensor<32x240x14x14xf32>
    %v5273 = stablehlo.reshape %v5272 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v5274 = stablehlo.reshape %v5273 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5275 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5276 = stablehlo.pad %v5274, %v5275, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240x28x28xf32>
    %v5277 = stablehlo.reverse %b6dW, dims = [2, 3] : tensor<240x1x3x3xf32>
    %v5278 = stablehlo.convolution(%v5276, %v5277)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x3x3xf32>) -> tensor<32x240x28x28xf32>
    %v5279 = stablehlo.reshape %v5278 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5280 = stablehlo.reshape %v606 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5281 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5282 = stablehlo.constant dense<6272.0> : tensor<32x240x14x14xf32>
    %v5283 = stablehlo.reduce(%v5280 init: %v5281) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5284 = stablehlo.broadcast_in_dim %v5283, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v5285 = stablehlo.divide %v5284, %v5282 : tensor<32x240x14x14xf32>
    %v5286 = stablehlo.subtract %v5280, %v5285 : tensor<32x240x14x14xf32>
    %v5287 = stablehlo.multiply %v5286, %v5286 : tensor<32x240x14x14xf32>
    %v5288 = stablehlo.reduce(%v5287 init: %v5281) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5289 = stablehlo.broadcast_in_dim %v5288, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v5290 = stablehlo.divide %v5289, %v5282 : tensor<32x240x14x14xf32>
    %v5291 = stablehlo.constant dense<1.0e-5> : tensor<32x240x14x14xf32>
    %v5292 = stablehlo.add %v5290, %v5291 : tensor<32x240x14x14xf32>
    %v5293 = stablehlo.rsqrt %v5292 : tensor<32x240x14x14xf32>
    %v5294 = stablehlo.multiply %v5286, %v5293 : tensor<32x240x14x14xf32>
    %v5295 = stablehlo.reshape %v5243 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5296 = stablehlo.multiply %v5295, %v5294 : tensor<32x240x14x14xf32>
    %v5297 = stablehlo.reduce(%v5296 init: %v5281) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5298 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5299 = stablehlo.multiply %v5297, %v5298 : tensor<240xf32>
    %v5300 = stablehlo.subtract %b6dg, %v5299 : tensor<240xf32>
    %v5301 = stablehlo.reshape %v5243 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5302 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5303 = stablehlo.reduce(%v5301 init: %v5302) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5304 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5305 = stablehlo.multiply %v5303, %v5304 : tensor<240xf32>
    %v5306 = stablehlo.subtract %b6dbt, %v5305 : tensor<240xf32>
    %v5307 = stablehlo.reshape %v601 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5308 = stablehlo.reshape %v5273 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5309 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5310 = stablehlo.pad %v5308, %v5309, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240x28x28xf32>
    %v5311 = stablehlo.transpose %v5307, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %v5312 = stablehlo.transpose %v5310, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %v5313 = stablehlo.convolution(%v5311, %v5312)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 240 : i64, feature_group_count = 1 : i64} : (tensor<240x32x28x28xf32>, tensor<240x32x28x28xf32>) -> tensor<1x240x3x3xf32>
    %v5314 = stablehlo.reshape %v5313 : (tensor<1x240x3x3xf32>) -> tensor<240x1x3x3xf32>
    %v5315 = stablehlo.constant dense<0.05> : tensor<240x1x3x3xf32>
    %v5316 = stablehlo.multiply %v5314, %v5315 : tensor<240x1x3x3xf32>
    %v5317 = stablehlo.subtract %b6dW, %v5316 : tensor<240x1x3x3xf32>
    %v5318 = stablehlo.reshape %v5279 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5319 = stablehlo.reshape %v597 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5320 = stablehlo.logistic %v5319 : tensor<32x240x28x28xf32>
    %v5321 = stablehlo.constant dense<1.0> : tensor<32x240x28x28xf32>
    %v5322 = stablehlo.subtract %v5321, %v5320 : tensor<32x240x28x28xf32>
    %v5323 = stablehlo.multiply %v5319, %v5322 : tensor<32x240x28x28xf32>
    %v5324 = stablehlo.add %v5321, %v5323 : tensor<32x240x28x28xf32>
    %v5325 = stablehlo.multiply %v5320, %v5324 : tensor<32x240x28x28xf32>
    %v5326 = stablehlo.multiply %v5318, %v5325 : tensor<32x240x28x28xf32>
    %v5327 = stablehlo.reshape %v5326 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5328 = stablehlo.reshape %v577 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5329 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5330 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v5331 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v5332 = stablehlo.reduce(%v5328 init: %v5329) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5333 = stablehlo.broadcast_in_dim %v5332, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5334 = stablehlo.divide %v5333, %v5330 : tensor<32x240x28x28xf32>
    %v5335 = stablehlo.subtract %v5328, %v5334 : tensor<32x240x28x28xf32>
    %v5336 = stablehlo.multiply %v5335, %v5335 : tensor<32x240x28x28xf32>
    %v5337 = stablehlo.reduce(%v5336 init: %v5329) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5338 = stablehlo.broadcast_in_dim %v5337, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5339 = stablehlo.divide %v5338, %v5330 : tensor<32x240x28x28xf32>
    %v5340 = stablehlo.add %v5339, %v5331 : tensor<32x240x28x28xf32>
    %v5341 = stablehlo.rsqrt %v5340 : tensor<32x240x28x28xf32>
    %v5342 = stablehlo.multiply %v5335, %v5341 : tensor<32x240x28x28xf32>
    %v5343 = stablehlo.broadcast_in_dim %b6eg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5344 = stablehlo.reshape %v5327 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5345 = stablehlo.multiply %v5343, %v5344 : tensor<32x240x28x28xf32>
    %v5346 = stablehlo.reduce(%v5345 init: %v5329) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5347 = stablehlo.broadcast_in_dim %v5346, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5348 = stablehlo.multiply %v5342, %v5345 : tensor<32x240x28x28xf32>
    %v5349 = stablehlo.reduce(%v5348 init: %v5329) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5350 = stablehlo.broadcast_in_dim %v5349, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5351 = stablehlo.multiply %v5345, %v5330 : tensor<32x240x28x28xf32>
    %v5352 = stablehlo.subtract %v5351, %v5347 : tensor<32x240x28x28xf32>
    %v5353 = stablehlo.multiply %v5342, %v5350 : tensor<32x240x28x28xf32>
    %v5354 = stablehlo.subtract %v5352, %v5353 : tensor<32x240x28x28xf32>
    %v5355 = stablehlo.divide %v5341, %v5330 : tensor<32x240x28x28xf32>
    %v5356 = stablehlo.multiply %v5355, %v5354 : tensor<32x240x28x28xf32>
    %v5357 = stablehlo.reshape %v5356 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5358 = stablehlo.reshape %v5357 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5359 = stablehlo.reverse %b6eW, dims = [2, 3] : tensor<240x40x1x1xf32>
    %v5360 = stablehlo.transpose %v5359, dims = [1, 0, 2, 3] : (tensor<240x40x1x1xf32>) -> tensor<40x240x1x1xf32>
    %v5361 = stablehlo.convolution(%v5358, %v5360)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x28x28xf32>, tensor<40x240x1x1xf32>) -> tensor<32x40x28x28xf32>
    %v5362 = stablehlo.reshape %v5361 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v5363 = stablehlo.reshape %v577 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5364 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5365 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v5366 = stablehlo.reduce(%v5363 init: %v5364) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5367 = stablehlo.broadcast_in_dim %v5366, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5368 = stablehlo.divide %v5367, %v5365 : tensor<32x240x28x28xf32>
    %v5369 = stablehlo.subtract %v5363, %v5368 : tensor<32x240x28x28xf32>
    %v5370 = stablehlo.multiply %v5369, %v5369 : tensor<32x240x28x28xf32>
    %v5371 = stablehlo.reduce(%v5370 init: %v5364) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5372 = stablehlo.broadcast_in_dim %v5371, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5373 = stablehlo.divide %v5372, %v5365 : tensor<32x240x28x28xf32>
    %v5374 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v5375 = stablehlo.add %v5373, %v5374 : tensor<32x240x28x28xf32>
    %v5376 = stablehlo.rsqrt %v5375 : tensor<32x240x28x28xf32>
    %v5377 = stablehlo.multiply %v5369, %v5376 : tensor<32x240x28x28xf32>
    %v5378 = stablehlo.reshape %v5327 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5379 = stablehlo.multiply %v5378, %v5377 : tensor<32x240x28x28xf32>
    %v5380 = stablehlo.reduce(%v5379 init: %v5364) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5381 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5382 = stablehlo.multiply %v5380, %v5381 : tensor<240xf32>
    %v5383 = stablehlo.subtract %b6eg, %v5382 : tensor<240xf32>
    %v5384 = stablehlo.reshape %v5327 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5385 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5386 = stablehlo.reduce(%v5384 init: %v5385) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5387 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5388 = stablehlo.multiply %v5386, %v5387 : tensor<240xf32>
    %v5389 = stablehlo.subtract %b6ebt, %v5388 : tensor<240xf32>
    %v5390 = stablehlo.reshape %v572 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5391 = stablehlo.reshape %v5357 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5392 = stablehlo.transpose %v5390, dims = [1, 0, 2, 3] : (tensor<32x40x28x28xf32>) -> tensor<40x32x28x28xf32>
    %v5393 = stablehlo.transpose %v5391, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %v5394 = stablehlo.convolution(%v5392, %v5393)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<40x32x28x28xf32>, tensor<240x32x28x28xf32>) -> tensor<40x240x1x1xf32>
    %v5395 = stablehlo.transpose %v5394, dims = [1, 0, 2, 3] : (tensor<40x240x1x1xf32>) -> tensor<240x40x1x1xf32>
    %v5396 = stablehlo.constant dense<0.05> : tensor<240x40x1x1xf32>
    %v5397 = stablehlo.multiply %v5395, %v5396 : tensor<240x40x1x1xf32>
    %v5398 = stablehlo.subtract %b6eW, %v5397 : tensor<240x40x1x1xf32>
    %v5399 = stablehlo.reshape %v548 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5400 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5401 = stablehlo.constant dense<25088.0> : tensor<32x40x28x28xf32>
    %v5402 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v5403 = stablehlo.reduce(%v5399 init: %v5400) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5404 = stablehlo.broadcast_in_dim %v5403, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5405 = stablehlo.divide %v5404, %v5401 : tensor<32x40x28x28xf32>
    %v5406 = stablehlo.subtract %v5399, %v5405 : tensor<32x40x28x28xf32>
    %v5407 = stablehlo.multiply %v5406, %v5406 : tensor<32x40x28x28xf32>
    %v5408 = stablehlo.reduce(%v5407 init: %v5400) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5409 = stablehlo.broadcast_in_dim %v5408, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5410 = stablehlo.divide %v5409, %v5401 : tensor<32x40x28x28xf32>
    %v5411 = stablehlo.add %v5410, %v5402 : tensor<32x40x28x28xf32>
    %v5412 = stablehlo.rsqrt %v5411 : tensor<32x40x28x28xf32>
    %v5413 = stablehlo.multiply %v5406, %v5412 : tensor<32x40x28x28xf32>
    %v5414 = stablehlo.broadcast_in_dim %b5pg, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5415 = stablehlo.reshape %v5362 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5416 = stablehlo.multiply %v5414, %v5415 : tensor<32x40x28x28xf32>
    %v5417 = stablehlo.reduce(%v5416 init: %v5400) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5418 = stablehlo.broadcast_in_dim %v5417, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5419 = stablehlo.multiply %v5413, %v5416 : tensor<32x40x28x28xf32>
    %v5420 = stablehlo.reduce(%v5419 init: %v5400) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5421 = stablehlo.broadcast_in_dim %v5420, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5422 = stablehlo.multiply %v5416, %v5401 : tensor<32x40x28x28xf32>
    %v5423 = stablehlo.subtract %v5422, %v5418 : tensor<32x40x28x28xf32>
    %v5424 = stablehlo.multiply %v5413, %v5421 : tensor<32x40x28x28xf32>
    %v5425 = stablehlo.subtract %v5423, %v5424 : tensor<32x40x28x28xf32>
    %v5426 = stablehlo.divide %v5412, %v5401 : tensor<32x40x28x28xf32>
    %v5427 = stablehlo.multiply %v5426, %v5425 : tensor<32x40x28x28xf32>
    %v5428 = stablehlo.reshape %v5427 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v5429 = stablehlo.reshape %v5428 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5430 = stablehlo.reverse %b5pW, dims = [2, 3] : tensor<40x240x1x1xf32>
    %v5431 = stablehlo.transpose %v5430, dims = [1, 0, 2, 3] : (tensor<40x240x1x1xf32>) -> tensor<240x40x1x1xf32>
    %v5432 = stablehlo.convolution(%v5429, %v5431)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<32x240x28x28xf32>
    %v5433 = stablehlo.reshape %v5432 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5434 = stablehlo.reshape %v548 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5435 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5436 = stablehlo.constant dense<25088.0> : tensor<32x40x28x28xf32>
    %v5437 = stablehlo.reduce(%v5434 init: %v5435) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5438 = stablehlo.broadcast_in_dim %v5437, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5439 = stablehlo.divide %v5438, %v5436 : tensor<32x40x28x28xf32>
    %v5440 = stablehlo.subtract %v5434, %v5439 : tensor<32x40x28x28xf32>
    %v5441 = stablehlo.multiply %v5440, %v5440 : tensor<32x40x28x28xf32>
    %v5442 = stablehlo.reduce(%v5441 init: %v5435) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5443 = stablehlo.broadcast_in_dim %v5442, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5444 = stablehlo.divide %v5443, %v5436 : tensor<32x40x28x28xf32>
    %v5445 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v5446 = stablehlo.add %v5444, %v5445 : tensor<32x40x28x28xf32>
    %v5447 = stablehlo.rsqrt %v5446 : tensor<32x40x28x28xf32>
    %v5448 = stablehlo.multiply %v5440, %v5447 : tensor<32x40x28x28xf32>
    %v5449 = stablehlo.reshape %v5362 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5450 = stablehlo.multiply %v5449, %v5448 : tensor<32x40x28x28xf32>
    %v5451 = stablehlo.reduce(%v5450 init: %v5435) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5452 = stablehlo.constant dense<0.05> : tensor<40xf32>
    %v5453 = stablehlo.multiply %v5451, %v5452 : tensor<40xf32>
    %v5454 = stablehlo.subtract %b5pg, %v5453 : tensor<40xf32>
    %v5455 = stablehlo.reshape %v5362 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5456 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5457 = stablehlo.reduce(%v5455 init: %v5456) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5458 = stablehlo.constant dense<0.05> : tensor<40xf32>
    %v5459 = stablehlo.multiply %v5457, %v5458 : tensor<40xf32>
    %v5460 = stablehlo.subtract %b5pbt, %v5459 : tensor<40xf32>
    %v5461 = stablehlo.reshape %v543 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5462 = stablehlo.reshape %v5428 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5463 = stablehlo.transpose %v5461, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %v5464 = stablehlo.transpose %v5462, dims = [1, 0, 2, 3] : (tensor<32x40x28x28xf32>) -> tensor<40x32x28x28xf32>
    %v5465 = stablehlo.convolution(%v5463, %v5464)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<240x32x28x28xf32>, tensor<40x32x28x28xf32>) -> tensor<240x40x1x1xf32>
    %v5466 = stablehlo.transpose %v5465, dims = [1, 0, 2, 3] : (tensor<240x40x1x1xf32>) -> tensor<40x240x1x1xf32>
    %v5467 = stablehlo.constant dense<0.05> : tensor<40x240x1x1xf32>
    %v5468 = stablehlo.multiply %v5466, %v5467 : tensor<40x240x1x1xf32>
    %v5469 = stablehlo.subtract %b5pW, %v5468 : tensor<40x240x1x1xf32>
    %v5470 = stablehlo.reshape %v513 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5471 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5472 = stablehlo.reduce(%v5470 init: %v5471) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v5473 = stablehlo.constant dense<784.0> : tensor<32x240xf32>
    %v5474 = stablehlo.divide %v5472, %v5473 : tensor<32x240xf32>
    %v5475 = stablehlo.dot_general %v5474, %b5zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v5476 = stablehlo.broadcast_in_dim %b5zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v5477 = stablehlo.add %v5475, %v5476 : tensor<32x10xf32>
    %v5478 = stablehlo.logistic %v5477 : tensor<32x10xf32>
    %v5479 = stablehlo.multiply %v5477, %v5478 : tensor<32x10xf32>
    %v5480 = stablehlo.dot_general %v5479, %b5zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v5481 = stablehlo.broadcast_in_dim %b5zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v5482 = stablehlo.add %v5480, %v5481 : tensor<32x240xf32>
    %v5483 = stablehlo.logistic %v5482 : tensor<32x240xf32>
    %v5484 = stablehlo.reshape %v5433 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5485 = stablehlo.broadcast_in_dim %v5483, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x28x28xf32>
    %v5486 = stablehlo.multiply %v5485, %v5484 : tensor<32x240x28x28xf32>
    %v5487 = stablehlo.multiply %v5470, %v5484 : tensor<32x240x28x28xf32>
    %v5488 = stablehlo.reduce(%v5487 init: %v5471) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v5489 = stablehlo.constant dense<1.0> : tensor<32x240xf32>
    %v5490 = stablehlo.subtract %v5489, %v5483 : tensor<32x240xf32>
    %v5491 = stablehlo.multiply %v5483, %v5490 : tensor<32x240xf32>
    %v5492 = stablehlo.multiply %v5488, %v5491 : tensor<32x240xf32>
    %v5493 = stablehlo.dot_general %v5492, %b5zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<10x240xf32>) -> tensor<32x10xf32>
    %v5494 = stablehlo.logistic %v5477 : tensor<32x10xf32>
    %v5495 = stablehlo.constant dense<1.0> : tensor<32x10xf32>
    %v5496 = stablehlo.subtract %v5495, %v5494 : tensor<32x10xf32>
    %v5497 = stablehlo.multiply %v5477, %v5496 : tensor<32x10xf32>
    %v5498 = stablehlo.add %v5495, %v5497 : tensor<32x10xf32>
    %v5499 = stablehlo.multiply %v5494, %v5498 : tensor<32x10xf32>
    %v5500 = stablehlo.multiply %v5493, %v5499 : tensor<32x10xf32>
    %v5501 = stablehlo.dot_general %v5500, %b5zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<240x10xf32>) -> tensor<32x240xf32>
    %v5502 = stablehlo.constant dense<784.0> : tensor<32x240xf32>
    %v5503 = stablehlo.divide %v5501, %v5502 : tensor<32x240xf32>
    %v5504 = stablehlo.broadcast_in_dim %v5503, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x28x28xf32>
    %v5505 = stablehlo.add %v5486, %v5504 : tensor<32x240x28x28xf32>
    %v5506 = stablehlo.reshape %v5505 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5507 = stablehlo.reshape %v513 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5508 = stablehlo.reshape %v5433 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5509 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5510 = stablehlo.multiply %v5507, %v5508 : tensor<32x240x28x28xf32>
    %v5511 = stablehlo.reduce(%v5510 init: %v5509) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v5512 = stablehlo.logistic %v526 : tensor<32x240xf32>
    %v5513 = stablehlo.constant dense<1.0> : tensor<32x240xf32>
    %v5514 = stablehlo.subtract %v5513, %v5512 : tensor<32x240xf32>
    %v5515 = stablehlo.multiply %v5512, %v5514 : tensor<32x240xf32>
    %v5516 = stablehlo.multiply %v5511, %v5515 : tensor<32x240xf32>
    %v5517 = stablehlo.dot_general %v523, %v5516, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<32x240xf32>) -> tensor<10x240xf32>
    %v5518 = stablehlo.constant dense<0.05> : tensor<10x240xf32>
    %v5519 = stablehlo.multiply %v5517, %v5518 : tensor<10x240xf32>
    %v5520 = stablehlo.subtract %b5zW2, %v5519 : tensor<10x240xf32>
    %v5521 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5522 = stablehlo.reduce(%v5516 init: %v5521) applies stablehlo.add across dimensions = [0] : (tensor<32x240xf32>, tensor<f32>) -> tensor<240xf32>
    %v5523 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5524 = stablehlo.multiply %v5522, %v5523 : tensor<240xf32>
    %v5525 = stablehlo.subtract %b5zb2, %v5524 : tensor<240xf32>
    %v5526 = stablehlo.reshape %v5516 : (tensor<32x240xf32>) -> tensor<32x1x240xf32>
    %v5527 = stablehlo.dot_general %v5526, %b5zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x240xf32>, tensor<10x240xf32>) -> tensor<32x1x10xf32>
    %v5528 = stablehlo.reshape %v5527 : (tensor<32x1x10xf32>) -> tensor<32x10xf32>
    %v5529 = stablehlo.logistic %v521 : tensor<32x10xf32>
    %v5530 = stablehlo.constant dense<1.0> : tensor<32x10xf32>
    %v5531 = stablehlo.subtract %v5530, %v5529 : tensor<32x10xf32>
    %v5532 = stablehlo.multiply %v521, %v5531 : tensor<32x10xf32>
    %v5533 = stablehlo.add %v5530, %v5532 : tensor<32x10xf32>
    %v5534 = stablehlo.multiply %v5529, %v5533 : tensor<32x10xf32>
    %v5535 = stablehlo.multiply %v5528, %v5534 : tensor<32x10xf32>
    %v5536 = stablehlo.dot_general %v518, %v5535, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<32x10xf32>) -> tensor<240x10xf32>
    %v5537 = stablehlo.constant dense<0.05> : tensor<240x10xf32>
    %v5538 = stablehlo.multiply %v5536, %v5537 : tensor<240x10xf32>
    %v5539 = stablehlo.subtract %b5zW1, %v5538 : tensor<240x10xf32>
    %v5540 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5541 = stablehlo.reduce(%v5535 init: %v5540) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v5542 = stablehlo.constant dense<0.05> : tensor<10xf32>
    %v5543 = stablehlo.multiply %v5541, %v5542 : tensor<10xf32>
    %v5544 = stablehlo.subtract %b5zb1, %v5543 : tensor<10xf32>
    %v5545 = stablehlo.reshape %v5506 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5546 = stablehlo.reshape %v509 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5547 = stablehlo.logistic %v5546 : tensor<32x240x28x28xf32>
    %v5548 = stablehlo.constant dense<1.0> : tensor<32x240x28x28xf32>
    %v5549 = stablehlo.subtract %v5548, %v5547 : tensor<32x240x28x28xf32>
    %v5550 = stablehlo.multiply %v5546, %v5549 : tensor<32x240x28x28xf32>
    %v5551 = stablehlo.add %v5548, %v5550 : tensor<32x240x28x28xf32>
    %v5552 = stablehlo.multiply %v5547, %v5551 : tensor<32x240x28x28xf32>
    %v5553 = stablehlo.multiply %v5545, %v5552 : tensor<32x240x28x28xf32>
    %v5554 = stablehlo.reshape %v5553 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5555 = stablehlo.reshape %v489 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5556 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5557 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v5558 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v5559 = stablehlo.reduce(%v5555 init: %v5556) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5560 = stablehlo.broadcast_in_dim %v5559, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5561 = stablehlo.divide %v5560, %v5557 : tensor<32x240x28x28xf32>
    %v5562 = stablehlo.subtract %v5555, %v5561 : tensor<32x240x28x28xf32>
    %v5563 = stablehlo.multiply %v5562, %v5562 : tensor<32x240x28x28xf32>
    %v5564 = stablehlo.reduce(%v5563 init: %v5556) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5565 = stablehlo.broadcast_in_dim %v5564, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5566 = stablehlo.divide %v5565, %v5557 : tensor<32x240x28x28xf32>
    %v5567 = stablehlo.add %v5566, %v5558 : tensor<32x240x28x28xf32>
    %v5568 = stablehlo.rsqrt %v5567 : tensor<32x240x28x28xf32>
    %v5569 = stablehlo.multiply %v5562, %v5568 : tensor<32x240x28x28xf32>
    %v5570 = stablehlo.broadcast_in_dim %b5dg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5571 = stablehlo.reshape %v5554 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5572 = stablehlo.multiply %v5570, %v5571 : tensor<32x240x28x28xf32>
    %v5573 = stablehlo.reduce(%v5572 init: %v5556) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5574 = stablehlo.broadcast_in_dim %v5573, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5575 = stablehlo.multiply %v5569, %v5572 : tensor<32x240x28x28xf32>
    %v5576 = stablehlo.reduce(%v5575 init: %v5556) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5577 = stablehlo.broadcast_in_dim %v5576, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5578 = stablehlo.multiply %v5572, %v5557 : tensor<32x240x28x28xf32>
    %v5579 = stablehlo.subtract %v5578, %v5574 : tensor<32x240x28x28xf32>
    %v5580 = stablehlo.multiply %v5569, %v5577 : tensor<32x240x28x28xf32>
    %v5581 = stablehlo.subtract %v5579, %v5580 : tensor<32x240x28x28xf32>
    %v5582 = stablehlo.divide %v5568, %v5557 : tensor<32x240x28x28xf32>
    %v5583 = stablehlo.multiply %v5582, %v5581 : tensor<32x240x28x28xf32>
    %v5584 = stablehlo.reshape %v5583 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5585 = stablehlo.reshape %v5584 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5586 = stablehlo.reverse %b5dW, dims = [2, 3] : tensor<240x1x5x5xf32>
    %v5587 = stablehlo.convolution(%v5585, %v5586)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x5x5xf32>) -> tensor<32x240x28x28xf32>
    %v5588 = stablehlo.reshape %v5587 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5589 = stablehlo.reshape %v489 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5590 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5591 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v5592 = stablehlo.reduce(%v5589 init: %v5590) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5593 = stablehlo.broadcast_in_dim %v5592, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5594 = stablehlo.divide %v5593, %v5591 : tensor<32x240x28x28xf32>
    %v5595 = stablehlo.subtract %v5589, %v5594 : tensor<32x240x28x28xf32>
    %v5596 = stablehlo.multiply %v5595, %v5595 : tensor<32x240x28x28xf32>
    %v5597 = stablehlo.reduce(%v5596 init: %v5590) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5598 = stablehlo.broadcast_in_dim %v5597, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5599 = stablehlo.divide %v5598, %v5591 : tensor<32x240x28x28xf32>
    %v5600 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v5601 = stablehlo.add %v5599, %v5600 : tensor<32x240x28x28xf32>
    %v5602 = stablehlo.rsqrt %v5601 : tensor<32x240x28x28xf32>
    %v5603 = stablehlo.multiply %v5595, %v5602 : tensor<32x240x28x28xf32>
    %v5604 = stablehlo.reshape %v5554 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5605 = stablehlo.multiply %v5604, %v5603 : tensor<32x240x28x28xf32>
    %v5606 = stablehlo.reduce(%v5605 init: %v5590) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5607 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5608 = stablehlo.multiply %v5606, %v5607 : tensor<240xf32>
    %v5609 = stablehlo.subtract %b5dg, %v5608 : tensor<240xf32>
    %v5610 = stablehlo.reshape %v5554 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5611 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5612 = stablehlo.reduce(%v5610 init: %v5611) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5613 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5614 = stablehlo.multiply %v5612, %v5613 : tensor<240xf32>
    %v5615 = stablehlo.subtract %b5dbt, %v5614 : tensor<240xf32>
    %v5616 = stablehlo.reshape %v484 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5617 = stablehlo.reshape %v5584 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5618 = stablehlo.transpose %v5616, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %v5619 = stablehlo.transpose %v5617, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %v5620 = stablehlo.convolution(%v5618, %v5619)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 240 : i64, feature_group_count = 1 : i64} : (tensor<240x32x28x28xf32>, tensor<240x32x28x28xf32>) -> tensor<1x240x5x5xf32>
    %v5621 = stablehlo.reshape %v5620 : (tensor<1x240x5x5xf32>) -> tensor<240x1x5x5xf32>
    %v5622 = stablehlo.constant dense<0.05> : tensor<240x1x5x5xf32>
    %v5623 = stablehlo.multiply %v5621, %v5622 : tensor<240x1x5x5xf32>
    %v5624 = stablehlo.subtract %b5dW, %v5623 : tensor<240x1x5x5xf32>
    %v5625 = stablehlo.reshape %v5588 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5626 = stablehlo.reshape %v480 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5627 = stablehlo.logistic %v5626 : tensor<32x240x28x28xf32>
    %v5628 = stablehlo.constant dense<1.0> : tensor<32x240x28x28xf32>
    %v5629 = stablehlo.subtract %v5628, %v5627 : tensor<32x240x28x28xf32>
    %v5630 = stablehlo.multiply %v5626, %v5629 : tensor<32x240x28x28xf32>
    %v5631 = stablehlo.add %v5628, %v5630 : tensor<32x240x28x28xf32>
    %v5632 = stablehlo.multiply %v5627, %v5631 : tensor<32x240x28x28xf32>
    %v5633 = stablehlo.multiply %v5625, %v5632 : tensor<32x240x28x28xf32>
    %v5634 = stablehlo.reshape %v5633 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5635 = stablehlo.reshape %v460 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5636 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5637 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v5638 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v5639 = stablehlo.reduce(%v5635 init: %v5636) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5640 = stablehlo.broadcast_in_dim %v5639, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5641 = stablehlo.divide %v5640, %v5637 : tensor<32x240x28x28xf32>
    %v5642 = stablehlo.subtract %v5635, %v5641 : tensor<32x240x28x28xf32>
    %v5643 = stablehlo.multiply %v5642, %v5642 : tensor<32x240x28x28xf32>
    %v5644 = stablehlo.reduce(%v5643 init: %v5636) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5645 = stablehlo.broadcast_in_dim %v5644, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5646 = stablehlo.divide %v5645, %v5637 : tensor<32x240x28x28xf32>
    %v5647 = stablehlo.add %v5646, %v5638 : tensor<32x240x28x28xf32>
    %v5648 = stablehlo.rsqrt %v5647 : tensor<32x240x28x28xf32>
    %v5649 = stablehlo.multiply %v5642, %v5648 : tensor<32x240x28x28xf32>
    %v5650 = stablehlo.broadcast_in_dim %b5eg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5651 = stablehlo.reshape %v5634 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5652 = stablehlo.multiply %v5650, %v5651 : tensor<32x240x28x28xf32>
    %v5653 = stablehlo.reduce(%v5652 init: %v5636) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5654 = stablehlo.broadcast_in_dim %v5653, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5655 = stablehlo.multiply %v5649, %v5652 : tensor<32x240x28x28xf32>
    %v5656 = stablehlo.reduce(%v5655 init: %v5636) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5657 = stablehlo.broadcast_in_dim %v5656, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5658 = stablehlo.multiply %v5652, %v5637 : tensor<32x240x28x28xf32>
    %v5659 = stablehlo.subtract %v5658, %v5654 : tensor<32x240x28x28xf32>
    %v5660 = stablehlo.multiply %v5649, %v5657 : tensor<32x240x28x28xf32>
    %v5661 = stablehlo.subtract %v5659, %v5660 : tensor<32x240x28x28xf32>
    %v5662 = stablehlo.divide %v5648, %v5637 : tensor<32x240x28x28xf32>
    %v5663 = stablehlo.multiply %v5662, %v5661 : tensor<32x240x28x28xf32>
    %v5664 = stablehlo.reshape %v5663 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5665 = stablehlo.reshape %v5664 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5666 = stablehlo.reverse %b5eW, dims = [2, 3] : tensor<240x40x1x1xf32>
    %v5667 = stablehlo.transpose %v5666, dims = [1, 0, 2, 3] : (tensor<240x40x1x1xf32>) -> tensor<40x240x1x1xf32>
    %v5668 = stablehlo.convolution(%v5665, %v5667)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x28x28xf32>, tensor<40x240x1x1xf32>) -> tensor<32x40x28x28xf32>
    %v5669 = stablehlo.reshape %v5668 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v5670 = stablehlo.reshape %v460 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5671 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5672 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v5673 = stablehlo.reduce(%v5670 init: %v5671) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5674 = stablehlo.broadcast_in_dim %v5673, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5675 = stablehlo.divide %v5674, %v5672 : tensor<32x240x28x28xf32>
    %v5676 = stablehlo.subtract %v5670, %v5675 : tensor<32x240x28x28xf32>
    %v5677 = stablehlo.multiply %v5676, %v5676 : tensor<32x240x28x28xf32>
    %v5678 = stablehlo.reduce(%v5677 init: %v5671) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5679 = stablehlo.broadcast_in_dim %v5678, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5680 = stablehlo.divide %v5679, %v5672 : tensor<32x240x28x28xf32>
    %v5681 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v5682 = stablehlo.add %v5680, %v5681 : tensor<32x240x28x28xf32>
    %v5683 = stablehlo.rsqrt %v5682 : tensor<32x240x28x28xf32>
    %v5684 = stablehlo.multiply %v5676, %v5683 : tensor<32x240x28x28xf32>
    %v5685 = stablehlo.reshape %v5634 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5686 = stablehlo.multiply %v5685, %v5684 : tensor<32x240x28x28xf32>
    %v5687 = stablehlo.reduce(%v5686 init: %v5671) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5688 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5689 = stablehlo.multiply %v5687, %v5688 : tensor<240xf32>
    %v5690 = stablehlo.subtract %b5eg, %v5689 : tensor<240xf32>
    %v5691 = stablehlo.reshape %v5634 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5692 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5693 = stablehlo.reduce(%v5691 init: %v5692) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5694 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5695 = stablehlo.multiply %v5693, %v5694 : tensor<240xf32>
    %v5696 = stablehlo.subtract %b5ebt, %v5695 : tensor<240xf32>
    %v5697 = stablehlo.reshape %v455 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5698 = stablehlo.reshape %v5664 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5699 = stablehlo.transpose %v5697, dims = [1, 0, 2, 3] : (tensor<32x40x28x28xf32>) -> tensor<40x32x28x28xf32>
    %v5700 = stablehlo.transpose %v5698, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %v5701 = stablehlo.convolution(%v5699, %v5700)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<40x32x28x28xf32>, tensor<240x32x28x28xf32>) -> tensor<40x240x1x1xf32>
    %v5702 = stablehlo.transpose %v5701, dims = [1, 0, 2, 3] : (tensor<40x240x1x1xf32>) -> tensor<240x40x1x1xf32>
    %v5703 = stablehlo.constant dense<0.05> : tensor<240x40x1x1xf32>
    %v5704 = stablehlo.multiply %v5702, %v5703 : tensor<240x40x1x1xf32>
    %v5705 = stablehlo.subtract %b5eW, %v5704 : tensor<240x40x1x1xf32>
    %v5706 = stablehlo.reshape %v5669 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5707 = stablehlo.reshape %v5362 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5708 = stablehlo.add %v5706, %v5707 : tensor<32x40x28x28xf32>
    %v5709 = stablehlo.reshape %v5708 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v5710 = stablehlo.reshape %v435 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5711 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5712 = stablehlo.constant dense<25088.0> : tensor<32x40x28x28xf32>
    %v5713 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v5714 = stablehlo.reduce(%v5710 init: %v5711) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5715 = stablehlo.broadcast_in_dim %v5714, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5716 = stablehlo.divide %v5715, %v5712 : tensor<32x40x28x28xf32>
    %v5717 = stablehlo.subtract %v5710, %v5716 : tensor<32x40x28x28xf32>
    %v5718 = stablehlo.multiply %v5717, %v5717 : tensor<32x40x28x28xf32>
    %v5719 = stablehlo.reduce(%v5718 init: %v5711) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5720 = stablehlo.broadcast_in_dim %v5719, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5721 = stablehlo.divide %v5720, %v5712 : tensor<32x40x28x28xf32>
    %v5722 = stablehlo.add %v5721, %v5713 : tensor<32x40x28x28xf32>
    %v5723 = stablehlo.rsqrt %v5722 : tensor<32x40x28x28xf32>
    %v5724 = stablehlo.multiply %v5717, %v5723 : tensor<32x40x28x28xf32>
    %v5725 = stablehlo.broadcast_in_dim %b4pg, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5726 = stablehlo.reshape %v5709 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5727 = stablehlo.multiply %v5725, %v5726 : tensor<32x40x28x28xf32>
    %v5728 = stablehlo.reduce(%v5727 init: %v5711) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5729 = stablehlo.broadcast_in_dim %v5728, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5730 = stablehlo.multiply %v5724, %v5727 : tensor<32x40x28x28xf32>
    %v5731 = stablehlo.reduce(%v5730 init: %v5711) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5732 = stablehlo.broadcast_in_dim %v5731, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5733 = stablehlo.multiply %v5727, %v5712 : tensor<32x40x28x28xf32>
    %v5734 = stablehlo.subtract %v5733, %v5729 : tensor<32x40x28x28xf32>
    %v5735 = stablehlo.multiply %v5724, %v5732 : tensor<32x40x28x28xf32>
    %v5736 = stablehlo.subtract %v5734, %v5735 : tensor<32x40x28x28xf32>
    %v5737 = stablehlo.divide %v5723, %v5712 : tensor<32x40x28x28xf32>
    %v5738 = stablehlo.multiply %v5737, %v5736 : tensor<32x40x28x28xf32>
    %v5739 = stablehlo.reshape %v5738 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v5740 = stablehlo.reshape %v5739 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5741 = stablehlo.reverse %b4pW, dims = [2, 3] : tensor<40x144x1x1xf32>
    %v5742 = stablehlo.transpose %v5741, dims = [1, 0, 2, 3] : (tensor<40x144x1x1xf32>) -> tensor<144x40x1x1xf32>
    %v5743 = stablehlo.convolution(%v5740, %v5742)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<144x40x1x1xf32>) -> tensor<32x144x28x28xf32>
    %v5744 = stablehlo.reshape %v5743 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v5745 = stablehlo.reshape %v435 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5746 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5747 = stablehlo.constant dense<25088.0> : tensor<32x40x28x28xf32>
    %v5748 = stablehlo.reduce(%v5745 init: %v5746) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5749 = stablehlo.broadcast_in_dim %v5748, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5750 = stablehlo.divide %v5749, %v5747 : tensor<32x40x28x28xf32>
    %v5751 = stablehlo.subtract %v5745, %v5750 : tensor<32x40x28x28xf32>
    %v5752 = stablehlo.multiply %v5751, %v5751 : tensor<32x40x28x28xf32>
    %v5753 = stablehlo.reduce(%v5752 init: %v5746) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5754 = stablehlo.broadcast_in_dim %v5753, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5755 = stablehlo.divide %v5754, %v5747 : tensor<32x40x28x28xf32>
    %v5756 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v5757 = stablehlo.add %v5755, %v5756 : tensor<32x40x28x28xf32>
    %v5758 = stablehlo.rsqrt %v5757 : tensor<32x40x28x28xf32>
    %v5759 = stablehlo.multiply %v5751, %v5758 : tensor<32x40x28x28xf32>
    %v5760 = stablehlo.reshape %v5709 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5761 = stablehlo.multiply %v5760, %v5759 : tensor<32x40x28x28xf32>
    %v5762 = stablehlo.reduce(%v5761 init: %v5746) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5763 = stablehlo.constant dense<0.05> : tensor<40xf32>
    %v5764 = stablehlo.multiply %v5762, %v5763 : tensor<40xf32>
    %v5765 = stablehlo.subtract %b4pg, %v5764 : tensor<40xf32>
    %v5766 = stablehlo.reshape %v5709 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5767 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5768 = stablehlo.reduce(%v5766 init: %v5767) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5769 = stablehlo.constant dense<0.05> : tensor<40xf32>
    %v5770 = stablehlo.multiply %v5768, %v5769 : tensor<40xf32>
    %v5771 = stablehlo.subtract %b4pbt, %v5770 : tensor<40xf32>
    %v5772 = stablehlo.reshape %v430 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5773 = stablehlo.reshape %v5739 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5774 = stablehlo.transpose %v5772, dims = [1, 0, 2, 3] : (tensor<32x144x28x28xf32>) -> tensor<144x32x28x28xf32>
    %v5775 = stablehlo.transpose %v5773, dims = [1, 0, 2, 3] : (tensor<32x40x28x28xf32>) -> tensor<40x32x28x28xf32>
    %v5776 = stablehlo.convolution(%v5774, %v5775)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<144x32x28x28xf32>, tensor<40x32x28x28xf32>) -> tensor<144x40x1x1xf32>
    %v5777 = stablehlo.transpose %v5776, dims = [1, 0, 2, 3] : (tensor<144x40x1x1xf32>) -> tensor<40x144x1x1xf32>
    %v5778 = stablehlo.constant dense<0.05> : tensor<40x144x1x1xf32>
    %v5779 = stablehlo.multiply %v5777, %v5778 : tensor<40x144x1x1xf32>
    %v5780 = stablehlo.subtract %b4pW, %v5779 : tensor<40x144x1x1xf32>
    %v5781 = stablehlo.reshape %v400 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5782 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5783 = stablehlo.reduce(%v5781 init: %v5782) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5784 = stablehlo.constant dense<784.0> : tensor<32x144xf32>
    %v5785 = stablehlo.divide %v5783, %v5784 : tensor<32x144xf32>
    %v5786 = stablehlo.dot_general %v5785, %b4zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v5787 = stablehlo.broadcast_in_dim %b4zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v5788 = stablehlo.add %v5786, %v5787 : tensor<32x6xf32>
    %v5789 = stablehlo.logistic %v5788 : tensor<32x6xf32>
    %v5790 = stablehlo.multiply %v5788, %v5789 : tensor<32x6xf32>
    %v5791 = stablehlo.dot_general %v5790, %b4zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v5792 = stablehlo.broadcast_in_dim %b4zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v5793 = stablehlo.add %v5791, %v5792 : tensor<32x144xf32>
    %v5794 = stablehlo.logistic %v5793 : tensor<32x144xf32>
    %v5795 = stablehlo.reshape %v5744 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5796 = stablehlo.broadcast_in_dim %v5794, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v5797 = stablehlo.multiply %v5796, %v5795 : tensor<32x144x28x28xf32>
    %v5798 = stablehlo.multiply %v5781, %v5795 : tensor<32x144x28x28xf32>
    %v5799 = stablehlo.reduce(%v5798 init: %v5782) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5800 = stablehlo.constant dense<1.0> : tensor<32x144xf32>
    %v5801 = stablehlo.subtract %v5800, %v5794 : tensor<32x144xf32>
    %v5802 = stablehlo.multiply %v5794, %v5801 : tensor<32x144xf32>
    %v5803 = stablehlo.multiply %v5799, %v5802 : tensor<32x144xf32>
    %v5804 = stablehlo.dot_general %v5803, %b4zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<6x144xf32>) -> tensor<32x6xf32>
    %v5805 = stablehlo.logistic %v5788 : tensor<32x6xf32>
    %v5806 = stablehlo.constant dense<1.0> : tensor<32x6xf32>
    %v5807 = stablehlo.subtract %v5806, %v5805 : tensor<32x6xf32>
    %v5808 = stablehlo.multiply %v5788, %v5807 : tensor<32x6xf32>
    %v5809 = stablehlo.add %v5806, %v5808 : tensor<32x6xf32>
    %v5810 = stablehlo.multiply %v5805, %v5809 : tensor<32x6xf32>
    %v5811 = stablehlo.multiply %v5804, %v5810 : tensor<32x6xf32>
    %v5812 = stablehlo.dot_general %v5811, %b4zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<144x6xf32>) -> tensor<32x144xf32>
    %v5813 = stablehlo.constant dense<784.0> : tensor<32x144xf32>
    %v5814 = stablehlo.divide %v5812, %v5813 : tensor<32x144xf32>
    %v5815 = stablehlo.broadcast_in_dim %v5814, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v5816 = stablehlo.add %v5797, %v5815 : tensor<32x144x28x28xf32>
    %v5817 = stablehlo.reshape %v5816 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v5818 = stablehlo.reshape %v400 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5819 = stablehlo.reshape %v5744 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5820 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5821 = stablehlo.multiply %v5818, %v5819 : tensor<32x144x28x28xf32>
    %v5822 = stablehlo.reduce(%v5821 init: %v5820) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5823 = stablehlo.logistic %v413 : tensor<32x144xf32>
    %v5824 = stablehlo.constant dense<1.0> : tensor<32x144xf32>
    %v5825 = stablehlo.subtract %v5824, %v5823 : tensor<32x144xf32>
    %v5826 = stablehlo.multiply %v5823, %v5825 : tensor<32x144xf32>
    %v5827 = stablehlo.multiply %v5822, %v5826 : tensor<32x144xf32>
    %v5828 = stablehlo.dot_general %v410, %v5827, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<32x144xf32>) -> tensor<6x144xf32>
    %v5829 = stablehlo.constant dense<0.05> : tensor<6x144xf32>
    %v5830 = stablehlo.multiply %v5828, %v5829 : tensor<6x144xf32>
    %v5831 = stablehlo.subtract %b4zW2, %v5830 : tensor<6x144xf32>
    %v5832 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5833 = stablehlo.reduce(%v5827 init: %v5832) applies stablehlo.add across dimensions = [0] : (tensor<32x144xf32>, tensor<f32>) -> tensor<144xf32>
    %v5834 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v5835 = stablehlo.multiply %v5833, %v5834 : tensor<144xf32>
    %v5836 = stablehlo.subtract %b4zb2, %v5835 : tensor<144xf32>
    %v5837 = stablehlo.reshape %v5827 : (tensor<32x144xf32>) -> tensor<32x1x144xf32>
    %v5838 = stablehlo.dot_general %v5837, %b4zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x144xf32>, tensor<6x144xf32>) -> tensor<32x1x6xf32>
    %v5839 = stablehlo.reshape %v5838 : (tensor<32x1x6xf32>) -> tensor<32x6xf32>
    %v5840 = stablehlo.logistic %v408 : tensor<32x6xf32>
    %v5841 = stablehlo.constant dense<1.0> : tensor<32x6xf32>
    %v5842 = stablehlo.subtract %v5841, %v5840 : tensor<32x6xf32>
    %v5843 = stablehlo.multiply %v408, %v5842 : tensor<32x6xf32>
    %v5844 = stablehlo.add %v5841, %v5843 : tensor<32x6xf32>
    %v5845 = stablehlo.multiply %v5840, %v5844 : tensor<32x6xf32>
    %v5846 = stablehlo.multiply %v5839, %v5845 : tensor<32x6xf32>
    %v5847 = stablehlo.dot_general %v405, %v5846, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<32x6xf32>) -> tensor<144x6xf32>
    %v5848 = stablehlo.constant dense<0.05> : tensor<144x6xf32>
    %v5849 = stablehlo.multiply %v5847, %v5848 : tensor<144x6xf32>
    %v5850 = stablehlo.subtract %b4zW1, %v5849 : tensor<144x6xf32>
    %v5851 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5852 = stablehlo.reduce(%v5846 init: %v5851) applies stablehlo.add across dimensions = [0] : (tensor<32x6xf32>, tensor<f32>) -> tensor<6xf32>
    %v5853 = stablehlo.constant dense<0.05> : tensor<6xf32>
    %v5854 = stablehlo.multiply %v5852, %v5853 : tensor<6xf32>
    %v5855 = stablehlo.subtract %b4zb1, %v5854 : tensor<6xf32>
    %v5856 = stablehlo.reshape %v5817 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5857 = stablehlo.reshape %v396 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5858 = stablehlo.logistic %v5857 : tensor<32x144x28x28xf32>
    %v5859 = stablehlo.constant dense<1.0> : tensor<32x144x28x28xf32>
    %v5860 = stablehlo.subtract %v5859, %v5858 : tensor<32x144x28x28xf32>
    %v5861 = stablehlo.multiply %v5857, %v5860 : tensor<32x144x28x28xf32>
    %v5862 = stablehlo.add %v5859, %v5861 : tensor<32x144x28x28xf32>
    %v5863 = stablehlo.multiply %v5858, %v5862 : tensor<32x144x28x28xf32>
    %v5864 = stablehlo.multiply %v5856, %v5863 : tensor<32x144x28x28xf32>
    %v5865 = stablehlo.reshape %v5864 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v5866 = stablehlo.reshape %v376 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5867 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5868 = stablehlo.constant dense<25088.0> : tensor<32x144x28x28xf32>
    %v5869 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v5870 = stablehlo.reduce(%v5866 init: %v5867) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5871 = stablehlo.broadcast_in_dim %v5870, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5872 = stablehlo.divide %v5871, %v5868 : tensor<32x144x28x28xf32>
    %v5873 = stablehlo.subtract %v5866, %v5872 : tensor<32x144x28x28xf32>
    %v5874 = stablehlo.multiply %v5873, %v5873 : tensor<32x144x28x28xf32>
    %v5875 = stablehlo.reduce(%v5874 init: %v5867) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5876 = stablehlo.broadcast_in_dim %v5875, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5877 = stablehlo.divide %v5876, %v5868 : tensor<32x144x28x28xf32>
    %v5878 = stablehlo.add %v5877, %v5869 : tensor<32x144x28x28xf32>
    %v5879 = stablehlo.rsqrt %v5878 : tensor<32x144x28x28xf32>
    %v5880 = stablehlo.multiply %v5873, %v5879 : tensor<32x144x28x28xf32>
    %v5881 = stablehlo.broadcast_in_dim %b4dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5882 = stablehlo.reshape %v5865 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5883 = stablehlo.multiply %v5881, %v5882 : tensor<32x144x28x28xf32>
    %v5884 = stablehlo.reduce(%v5883 init: %v5867) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5885 = stablehlo.broadcast_in_dim %v5884, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5886 = stablehlo.multiply %v5880, %v5883 : tensor<32x144x28x28xf32>
    %v5887 = stablehlo.reduce(%v5886 init: %v5867) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5888 = stablehlo.broadcast_in_dim %v5887, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5889 = stablehlo.multiply %v5883, %v5868 : tensor<32x144x28x28xf32>
    %v5890 = stablehlo.subtract %v5889, %v5885 : tensor<32x144x28x28xf32>
    %v5891 = stablehlo.multiply %v5880, %v5888 : tensor<32x144x28x28xf32>
    %v5892 = stablehlo.subtract %v5890, %v5891 : tensor<32x144x28x28xf32>
    %v5893 = stablehlo.divide %v5879, %v5868 : tensor<32x144x28x28xf32>
    %v5894 = stablehlo.multiply %v5893, %v5892 : tensor<32x144x28x28xf32>
    %v5895 = stablehlo.reshape %v5894 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v5896 = stablehlo.reshape %v5895 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5897 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5898 = stablehlo.pad %v5896, %v5897, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144x56x56xf32>
    %v5899 = stablehlo.reverse %b4dW, dims = [2, 3] : tensor<144x1x5x5xf32>
    %v5900 = stablehlo.convolution(%v5898, %v5899)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x5x5xf32>) -> tensor<32x144x56x56xf32>
    %v5901 = stablehlo.reshape %v5900 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v5902 = stablehlo.reshape %v376 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5903 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5904 = stablehlo.constant dense<25088.0> : tensor<32x144x28x28xf32>
    %v5905 = stablehlo.reduce(%v5902 init: %v5903) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5906 = stablehlo.broadcast_in_dim %v5905, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5907 = stablehlo.divide %v5906, %v5904 : tensor<32x144x28x28xf32>
    %v5908 = stablehlo.subtract %v5902, %v5907 : tensor<32x144x28x28xf32>
    %v5909 = stablehlo.multiply %v5908, %v5908 : tensor<32x144x28x28xf32>
    %v5910 = stablehlo.reduce(%v5909 init: %v5903) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5911 = stablehlo.broadcast_in_dim %v5910, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5912 = stablehlo.divide %v5911, %v5904 : tensor<32x144x28x28xf32>
    %v5913 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v5914 = stablehlo.add %v5912, %v5913 : tensor<32x144x28x28xf32>
    %v5915 = stablehlo.rsqrt %v5914 : tensor<32x144x28x28xf32>
    %v5916 = stablehlo.multiply %v5908, %v5915 : tensor<32x144x28x28xf32>
    %v5917 = stablehlo.reshape %v5865 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5918 = stablehlo.multiply %v5917, %v5916 : tensor<32x144x28x28xf32>
    %v5919 = stablehlo.reduce(%v5918 init: %v5903) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5920 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v5921 = stablehlo.multiply %v5919, %v5920 : tensor<144xf32>
    %v5922 = stablehlo.subtract %b4dg, %v5921 : tensor<144xf32>
    %v5923 = stablehlo.reshape %v5865 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5924 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5925 = stablehlo.reduce(%v5923 init: %v5924) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5926 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v5927 = stablehlo.multiply %v5925, %v5926 : tensor<144xf32>
    %v5928 = stablehlo.subtract %b4dbt, %v5927 : tensor<144xf32>
    %v5929 = stablehlo.reshape %v371 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5930 = stablehlo.reshape %v5895 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5931 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5932 = stablehlo.pad %v5930, %v5931, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144x56x56xf32>
    %v5933 = stablehlo.transpose %v5929, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v5934 = stablehlo.transpose %v5932, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v5935 = stablehlo.convolution(%v5933, %v5934)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 144 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<1x144x5x5xf32>
    %v5936 = stablehlo.reshape %v5935 : (tensor<1x144x5x5xf32>) -> tensor<144x1x5x5xf32>
    %v5937 = stablehlo.constant dense<0.05> : tensor<144x1x5x5xf32>
    %v5938 = stablehlo.multiply %v5936, %v5937 : tensor<144x1x5x5xf32>
    %v5939 = stablehlo.subtract %b4dW, %v5938 : tensor<144x1x5x5xf32>
    %v5940 = stablehlo.reshape %v5901 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5941 = stablehlo.reshape %v367 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5942 = stablehlo.logistic %v5941 : tensor<32x144x56x56xf32>
    %v5943 = stablehlo.constant dense<1.0> : tensor<32x144x56x56xf32>
    %v5944 = stablehlo.subtract %v5943, %v5942 : tensor<32x144x56x56xf32>
    %v5945 = stablehlo.multiply %v5941, %v5944 : tensor<32x144x56x56xf32>
    %v5946 = stablehlo.add %v5943, %v5945 : tensor<32x144x56x56xf32>
    %v5947 = stablehlo.multiply %v5942, %v5946 : tensor<32x144x56x56xf32>
    %v5948 = stablehlo.multiply %v5940, %v5947 : tensor<32x144x56x56xf32>
    %v5949 = stablehlo.reshape %v5948 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v5950 = stablehlo.reshape %v347 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5951 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5952 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v5953 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v5954 = stablehlo.reduce(%v5950 init: %v5951) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5955 = stablehlo.broadcast_in_dim %v5954, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5956 = stablehlo.divide %v5955, %v5952 : tensor<32x144x56x56xf32>
    %v5957 = stablehlo.subtract %v5950, %v5956 : tensor<32x144x56x56xf32>
    %v5958 = stablehlo.multiply %v5957, %v5957 : tensor<32x144x56x56xf32>
    %v5959 = stablehlo.reduce(%v5958 init: %v5951) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5960 = stablehlo.broadcast_in_dim %v5959, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5961 = stablehlo.divide %v5960, %v5952 : tensor<32x144x56x56xf32>
    %v5962 = stablehlo.add %v5961, %v5953 : tensor<32x144x56x56xf32>
    %v5963 = stablehlo.rsqrt %v5962 : tensor<32x144x56x56xf32>
    %v5964 = stablehlo.multiply %v5957, %v5963 : tensor<32x144x56x56xf32>
    %v5965 = stablehlo.broadcast_in_dim %b4eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5966 = stablehlo.reshape %v5949 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5967 = stablehlo.multiply %v5965, %v5966 : tensor<32x144x56x56xf32>
    %v5968 = stablehlo.reduce(%v5967 init: %v5951) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5969 = stablehlo.broadcast_in_dim %v5968, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5970 = stablehlo.multiply %v5964, %v5967 : tensor<32x144x56x56xf32>
    %v5971 = stablehlo.reduce(%v5970 init: %v5951) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5972 = stablehlo.broadcast_in_dim %v5971, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5973 = stablehlo.multiply %v5967, %v5952 : tensor<32x144x56x56xf32>
    %v5974 = stablehlo.subtract %v5973, %v5969 : tensor<32x144x56x56xf32>
    %v5975 = stablehlo.multiply %v5964, %v5972 : tensor<32x144x56x56xf32>
    %v5976 = stablehlo.subtract %v5974, %v5975 : tensor<32x144x56x56xf32>
    %v5977 = stablehlo.divide %v5963, %v5952 : tensor<32x144x56x56xf32>
    %v5978 = stablehlo.multiply %v5977, %v5976 : tensor<32x144x56x56xf32>
    %v5979 = stablehlo.reshape %v5978 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v5980 = stablehlo.reshape %v5979 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5981 = stablehlo.reverse %b4eW, dims = [2, 3] : tensor<144x24x1x1xf32>
    %v5982 = stablehlo.transpose %v5981, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %v5983 = stablehlo.convolution(%v5980, %v5982)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v5984 = stablehlo.reshape %v5983 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v5985 = stablehlo.reshape %v347 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5986 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5987 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v5988 = stablehlo.reduce(%v5985 init: %v5986) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5989 = stablehlo.broadcast_in_dim %v5988, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5990 = stablehlo.divide %v5989, %v5987 : tensor<32x144x56x56xf32>
    %v5991 = stablehlo.subtract %v5985, %v5990 : tensor<32x144x56x56xf32>
    %v5992 = stablehlo.multiply %v5991, %v5991 : tensor<32x144x56x56xf32>
    %v5993 = stablehlo.reduce(%v5992 init: %v5986) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5994 = stablehlo.broadcast_in_dim %v5993, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5995 = stablehlo.divide %v5994, %v5987 : tensor<32x144x56x56xf32>
    %v5996 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v5997 = stablehlo.add %v5995, %v5996 : tensor<32x144x56x56xf32>
    %v5998 = stablehlo.rsqrt %v5997 : tensor<32x144x56x56xf32>
    %v5999 = stablehlo.multiply %v5991, %v5998 : tensor<32x144x56x56xf32>
    %v6000 = stablehlo.reshape %v5949 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6001 = stablehlo.multiply %v6000, %v5999 : tensor<32x144x56x56xf32>
    %v6002 = stablehlo.reduce(%v6001 init: %v5986) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6003 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v6004 = stablehlo.multiply %v6002, %v6003 : tensor<144xf32>
    %v6005 = stablehlo.subtract %b4eg, %v6004 : tensor<144xf32>
    %v6006 = stablehlo.reshape %v5949 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6007 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6008 = stablehlo.reduce(%v6006 init: %v6007) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6009 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v6010 = stablehlo.multiply %v6008, %v6009 : tensor<144xf32>
    %v6011 = stablehlo.subtract %b4ebt, %v6010 : tensor<144xf32>
    %v6012 = stablehlo.reshape %v342 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6013 = stablehlo.reshape %v5979 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6014 = stablehlo.transpose %v6012, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v6015 = stablehlo.transpose %v6013, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v6016 = stablehlo.convolution(%v6014, %v6015)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<24x144x1x1xf32>
    %v6017 = stablehlo.transpose %v6016, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %v6018 = stablehlo.constant dense<0.05> : tensor<144x24x1x1xf32>
    %v6019 = stablehlo.multiply %v6017, %v6018 : tensor<144x24x1x1xf32>
    %v6020 = stablehlo.subtract %b4eW, %v6019 : tensor<144x24x1x1xf32>
    %v6021 = stablehlo.reshape %v318 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6022 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6023 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v6024 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v6025 = stablehlo.reduce(%v6021 init: %v6022) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6026 = stablehlo.broadcast_in_dim %v6025, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6027 = stablehlo.divide %v6026, %v6023 : tensor<32x24x56x56xf32>
    %v6028 = stablehlo.subtract %v6021, %v6027 : tensor<32x24x56x56xf32>
    %v6029 = stablehlo.multiply %v6028, %v6028 : tensor<32x24x56x56xf32>
    %v6030 = stablehlo.reduce(%v6029 init: %v6022) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6031 = stablehlo.broadcast_in_dim %v6030, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6032 = stablehlo.divide %v6031, %v6023 : tensor<32x24x56x56xf32>
    %v6033 = stablehlo.add %v6032, %v6024 : tensor<32x24x56x56xf32>
    %v6034 = stablehlo.rsqrt %v6033 : tensor<32x24x56x56xf32>
    %v6035 = stablehlo.multiply %v6028, %v6034 : tensor<32x24x56x56xf32>
    %v6036 = stablehlo.broadcast_in_dim %b3pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6037 = stablehlo.reshape %v5984 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6038 = stablehlo.multiply %v6036, %v6037 : tensor<32x24x56x56xf32>
    %v6039 = stablehlo.reduce(%v6038 init: %v6022) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6040 = stablehlo.broadcast_in_dim %v6039, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6041 = stablehlo.multiply %v6035, %v6038 : tensor<32x24x56x56xf32>
    %v6042 = stablehlo.reduce(%v6041 init: %v6022) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6043 = stablehlo.broadcast_in_dim %v6042, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6044 = stablehlo.multiply %v6038, %v6023 : tensor<32x24x56x56xf32>
    %v6045 = stablehlo.subtract %v6044, %v6040 : tensor<32x24x56x56xf32>
    %v6046 = stablehlo.multiply %v6035, %v6043 : tensor<32x24x56x56xf32>
    %v6047 = stablehlo.subtract %v6045, %v6046 : tensor<32x24x56x56xf32>
    %v6048 = stablehlo.divide %v6034, %v6023 : tensor<32x24x56x56xf32>
    %v6049 = stablehlo.multiply %v6048, %v6047 : tensor<32x24x56x56xf32>
    %v6050 = stablehlo.reshape %v6049 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v6051 = stablehlo.reshape %v6050 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6052 = stablehlo.reverse %b3pW, dims = [2, 3] : tensor<24x144x1x1xf32>
    %v6053 = stablehlo.transpose %v6052, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %v6054 = stablehlo.convolution(%v6051, %v6053)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v6055 = stablehlo.reshape %v6054 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v6056 = stablehlo.reshape %v318 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6057 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6058 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v6059 = stablehlo.reduce(%v6056 init: %v6057) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6060 = stablehlo.broadcast_in_dim %v6059, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6061 = stablehlo.divide %v6060, %v6058 : tensor<32x24x56x56xf32>
    %v6062 = stablehlo.subtract %v6056, %v6061 : tensor<32x24x56x56xf32>
    %v6063 = stablehlo.multiply %v6062, %v6062 : tensor<32x24x56x56xf32>
    %v6064 = stablehlo.reduce(%v6063 init: %v6057) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6065 = stablehlo.broadcast_in_dim %v6064, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6066 = stablehlo.divide %v6065, %v6058 : tensor<32x24x56x56xf32>
    %v6067 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v6068 = stablehlo.add %v6066, %v6067 : tensor<32x24x56x56xf32>
    %v6069 = stablehlo.rsqrt %v6068 : tensor<32x24x56x56xf32>
    %v6070 = stablehlo.multiply %v6062, %v6069 : tensor<32x24x56x56xf32>
    %v6071 = stablehlo.reshape %v5984 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6072 = stablehlo.multiply %v6071, %v6070 : tensor<32x24x56x56xf32>
    %v6073 = stablehlo.reduce(%v6072 init: %v6057) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6074 = stablehlo.constant dense<0.05> : tensor<24xf32>
    %v6075 = stablehlo.multiply %v6073, %v6074 : tensor<24xf32>
    %v6076 = stablehlo.subtract %b3pg, %v6075 : tensor<24xf32>
    %v6077 = stablehlo.reshape %v5984 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6078 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6079 = stablehlo.reduce(%v6077 init: %v6078) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6080 = stablehlo.constant dense<0.05> : tensor<24xf32>
    %v6081 = stablehlo.multiply %v6079, %v6080 : tensor<24xf32>
    %v6082 = stablehlo.subtract %b3pbt, %v6081 : tensor<24xf32>
    %v6083 = stablehlo.reshape %v313 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6084 = stablehlo.reshape %v6050 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6085 = stablehlo.transpose %v6083, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v6086 = stablehlo.transpose %v6084, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v6087 = stablehlo.convolution(%v6085, %v6086)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<24x32x56x56xf32>) -> tensor<144x24x1x1xf32>
    %v6088 = stablehlo.transpose %v6087, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %v6089 = stablehlo.constant dense<0.05> : tensor<24x144x1x1xf32>
    %v6090 = stablehlo.multiply %v6088, %v6089 : tensor<24x144x1x1xf32>
    %v6091 = stablehlo.subtract %b3pW, %v6090 : tensor<24x144x1x1xf32>
    %v6092 = stablehlo.reshape %v283 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6093 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6094 = stablehlo.reduce(%v6092 init: %v6093) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v6095 = stablehlo.constant dense<3136.0> : tensor<32x144xf32>
    %v6096 = stablehlo.divide %v6094, %v6095 : tensor<32x144xf32>
    %v6097 = stablehlo.dot_general %v6096, %b3zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v6098 = stablehlo.broadcast_in_dim %b3zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v6099 = stablehlo.add %v6097, %v6098 : tensor<32x6xf32>
    %v6100 = stablehlo.logistic %v6099 : tensor<32x6xf32>
    %v6101 = stablehlo.multiply %v6099, %v6100 : tensor<32x6xf32>
    %v6102 = stablehlo.dot_general %v6101, %b3zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v6103 = stablehlo.broadcast_in_dim %b3zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v6104 = stablehlo.add %v6102, %v6103 : tensor<32x144xf32>
    %v6105 = stablehlo.logistic %v6104 : tensor<32x144xf32>
    %v6106 = stablehlo.reshape %v6055 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6107 = stablehlo.broadcast_in_dim %v6105, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v6108 = stablehlo.multiply %v6107, %v6106 : tensor<32x144x56x56xf32>
    %v6109 = stablehlo.multiply %v6092, %v6106 : tensor<32x144x56x56xf32>
    %v6110 = stablehlo.reduce(%v6109 init: %v6093) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v6111 = stablehlo.constant dense<1.0> : tensor<32x144xf32>
    %v6112 = stablehlo.subtract %v6111, %v6105 : tensor<32x144xf32>
    %v6113 = stablehlo.multiply %v6105, %v6112 : tensor<32x144xf32>
    %v6114 = stablehlo.multiply %v6110, %v6113 : tensor<32x144xf32>
    %v6115 = stablehlo.dot_general %v6114, %b3zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<6x144xf32>) -> tensor<32x6xf32>
    %v6116 = stablehlo.logistic %v6099 : tensor<32x6xf32>
    %v6117 = stablehlo.constant dense<1.0> : tensor<32x6xf32>
    %v6118 = stablehlo.subtract %v6117, %v6116 : tensor<32x6xf32>
    %v6119 = stablehlo.multiply %v6099, %v6118 : tensor<32x6xf32>
    %v6120 = stablehlo.add %v6117, %v6119 : tensor<32x6xf32>
    %v6121 = stablehlo.multiply %v6116, %v6120 : tensor<32x6xf32>
    %v6122 = stablehlo.multiply %v6115, %v6121 : tensor<32x6xf32>
    %v6123 = stablehlo.dot_general %v6122, %b3zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<144x6xf32>) -> tensor<32x144xf32>
    %v6124 = stablehlo.constant dense<3136.0> : tensor<32x144xf32>
    %v6125 = stablehlo.divide %v6123, %v6124 : tensor<32x144xf32>
    %v6126 = stablehlo.broadcast_in_dim %v6125, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v6127 = stablehlo.add %v6108, %v6126 : tensor<32x144x56x56xf32>
    %v6128 = stablehlo.reshape %v6127 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v6129 = stablehlo.reshape %v283 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6130 = stablehlo.reshape %v6055 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6131 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6132 = stablehlo.multiply %v6129, %v6130 : tensor<32x144x56x56xf32>
    %v6133 = stablehlo.reduce(%v6132 init: %v6131) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v6134 = stablehlo.logistic %v296 : tensor<32x144xf32>
    %v6135 = stablehlo.constant dense<1.0> : tensor<32x144xf32>
    %v6136 = stablehlo.subtract %v6135, %v6134 : tensor<32x144xf32>
    %v6137 = stablehlo.multiply %v6134, %v6136 : tensor<32x144xf32>
    %v6138 = stablehlo.multiply %v6133, %v6137 : tensor<32x144xf32>
    %v6139 = stablehlo.dot_general %v293, %v6138, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<32x144xf32>) -> tensor<6x144xf32>
    %v6140 = stablehlo.constant dense<0.05> : tensor<6x144xf32>
    %v6141 = stablehlo.multiply %v6139, %v6140 : tensor<6x144xf32>
    %v6142 = stablehlo.subtract %b3zW2, %v6141 : tensor<6x144xf32>
    %v6143 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6144 = stablehlo.reduce(%v6138 init: %v6143) applies stablehlo.add across dimensions = [0] : (tensor<32x144xf32>, tensor<f32>) -> tensor<144xf32>
    %v6145 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v6146 = stablehlo.multiply %v6144, %v6145 : tensor<144xf32>
    %v6147 = stablehlo.subtract %b3zb2, %v6146 : tensor<144xf32>
    %v6148 = stablehlo.reshape %v6138 : (tensor<32x144xf32>) -> tensor<32x1x144xf32>
    %v6149 = stablehlo.dot_general %v6148, %b3zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x144xf32>, tensor<6x144xf32>) -> tensor<32x1x6xf32>
    %v6150 = stablehlo.reshape %v6149 : (tensor<32x1x6xf32>) -> tensor<32x6xf32>
    %v6151 = stablehlo.logistic %v291 : tensor<32x6xf32>
    %v6152 = stablehlo.constant dense<1.0> : tensor<32x6xf32>
    %v6153 = stablehlo.subtract %v6152, %v6151 : tensor<32x6xf32>
    %v6154 = stablehlo.multiply %v291, %v6153 : tensor<32x6xf32>
    %v6155 = stablehlo.add %v6152, %v6154 : tensor<32x6xf32>
    %v6156 = stablehlo.multiply %v6151, %v6155 : tensor<32x6xf32>
    %v6157 = stablehlo.multiply %v6150, %v6156 : tensor<32x6xf32>
    %v6158 = stablehlo.dot_general %v288, %v6157, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<32x6xf32>) -> tensor<144x6xf32>
    %v6159 = stablehlo.constant dense<0.05> : tensor<144x6xf32>
    %v6160 = stablehlo.multiply %v6158, %v6159 : tensor<144x6xf32>
    %v6161 = stablehlo.subtract %b3zW1, %v6160 : tensor<144x6xf32>
    %v6162 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6163 = stablehlo.reduce(%v6157 init: %v6162) applies stablehlo.add across dimensions = [0] : (tensor<32x6xf32>, tensor<f32>) -> tensor<6xf32>
    %v6164 = stablehlo.constant dense<0.05> : tensor<6xf32>
    %v6165 = stablehlo.multiply %v6163, %v6164 : tensor<6xf32>
    %v6166 = stablehlo.subtract %b3zb1, %v6165 : tensor<6xf32>
    %v6167 = stablehlo.reshape %v6128 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6168 = stablehlo.reshape %v279 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6169 = stablehlo.logistic %v6168 : tensor<32x144x56x56xf32>
    %v6170 = stablehlo.constant dense<1.0> : tensor<32x144x56x56xf32>
    %v6171 = stablehlo.subtract %v6170, %v6169 : tensor<32x144x56x56xf32>
    %v6172 = stablehlo.multiply %v6168, %v6171 : tensor<32x144x56x56xf32>
    %v6173 = stablehlo.add %v6170, %v6172 : tensor<32x144x56x56xf32>
    %v6174 = stablehlo.multiply %v6169, %v6173 : tensor<32x144x56x56xf32>
    %v6175 = stablehlo.multiply %v6167, %v6174 : tensor<32x144x56x56xf32>
    %v6176 = stablehlo.reshape %v6175 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v6177 = stablehlo.reshape %v259 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6178 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6179 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v6180 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v6181 = stablehlo.reduce(%v6177 init: %v6178) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6182 = stablehlo.broadcast_in_dim %v6181, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6183 = stablehlo.divide %v6182, %v6179 : tensor<32x144x56x56xf32>
    %v6184 = stablehlo.subtract %v6177, %v6183 : tensor<32x144x56x56xf32>
    %v6185 = stablehlo.multiply %v6184, %v6184 : tensor<32x144x56x56xf32>
    %v6186 = stablehlo.reduce(%v6185 init: %v6178) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6187 = stablehlo.broadcast_in_dim %v6186, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6188 = stablehlo.divide %v6187, %v6179 : tensor<32x144x56x56xf32>
    %v6189 = stablehlo.add %v6188, %v6180 : tensor<32x144x56x56xf32>
    %v6190 = stablehlo.rsqrt %v6189 : tensor<32x144x56x56xf32>
    %v6191 = stablehlo.multiply %v6184, %v6190 : tensor<32x144x56x56xf32>
    %v6192 = stablehlo.broadcast_in_dim %b3dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6193 = stablehlo.reshape %v6176 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6194 = stablehlo.multiply %v6192, %v6193 : tensor<32x144x56x56xf32>
    %v6195 = stablehlo.reduce(%v6194 init: %v6178) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6196 = stablehlo.broadcast_in_dim %v6195, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6197 = stablehlo.multiply %v6191, %v6194 : tensor<32x144x56x56xf32>
    %v6198 = stablehlo.reduce(%v6197 init: %v6178) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6199 = stablehlo.broadcast_in_dim %v6198, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6200 = stablehlo.multiply %v6194, %v6179 : tensor<32x144x56x56xf32>
    %v6201 = stablehlo.subtract %v6200, %v6196 : tensor<32x144x56x56xf32>
    %v6202 = stablehlo.multiply %v6191, %v6199 : tensor<32x144x56x56xf32>
    %v6203 = stablehlo.subtract %v6201, %v6202 : tensor<32x144x56x56xf32>
    %v6204 = stablehlo.divide %v6190, %v6179 : tensor<32x144x56x56xf32>
    %v6205 = stablehlo.multiply %v6204, %v6203 : tensor<32x144x56x56xf32>
    %v6206 = stablehlo.reshape %v6205 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v6207 = stablehlo.reshape %v6206 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6208 = stablehlo.reverse %b3dW, dims = [2, 3] : tensor<144x1x3x3xf32>
    %v6209 = stablehlo.convolution(%v6207, %v6208)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x56x56xf32>
    %v6210 = stablehlo.reshape %v6209 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v6211 = stablehlo.reshape %v259 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6212 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6213 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v6214 = stablehlo.reduce(%v6211 init: %v6212) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6215 = stablehlo.broadcast_in_dim %v6214, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6216 = stablehlo.divide %v6215, %v6213 : tensor<32x144x56x56xf32>
    %v6217 = stablehlo.subtract %v6211, %v6216 : tensor<32x144x56x56xf32>
    %v6218 = stablehlo.multiply %v6217, %v6217 : tensor<32x144x56x56xf32>
    %v6219 = stablehlo.reduce(%v6218 init: %v6212) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6220 = stablehlo.broadcast_in_dim %v6219, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6221 = stablehlo.divide %v6220, %v6213 : tensor<32x144x56x56xf32>
    %v6222 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v6223 = stablehlo.add %v6221, %v6222 : tensor<32x144x56x56xf32>
    %v6224 = stablehlo.rsqrt %v6223 : tensor<32x144x56x56xf32>
    %v6225 = stablehlo.multiply %v6217, %v6224 : tensor<32x144x56x56xf32>
    %v6226 = stablehlo.reshape %v6176 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6227 = stablehlo.multiply %v6226, %v6225 : tensor<32x144x56x56xf32>
    %v6228 = stablehlo.reduce(%v6227 init: %v6212) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6229 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v6230 = stablehlo.multiply %v6228, %v6229 : tensor<144xf32>
    %v6231 = stablehlo.subtract %b3dg, %v6230 : tensor<144xf32>
    %v6232 = stablehlo.reshape %v6176 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6233 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6234 = stablehlo.reduce(%v6232 init: %v6233) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6235 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v6236 = stablehlo.multiply %v6234, %v6235 : tensor<144xf32>
    %v6237 = stablehlo.subtract %b3dbt, %v6236 : tensor<144xf32>
    %v6238 = stablehlo.reshape %v254 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6239 = stablehlo.reshape %v6206 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6240 = stablehlo.transpose %v6238, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v6241 = stablehlo.transpose %v6239, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v6242 = stablehlo.convolution(%v6240, %v6241)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 144 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<1x144x3x3xf32>
    %v6243 = stablehlo.reshape %v6242 : (tensor<1x144x3x3xf32>) -> tensor<144x1x3x3xf32>
    %v6244 = stablehlo.constant dense<0.05> : tensor<144x1x3x3xf32>
    %v6245 = stablehlo.multiply %v6243, %v6244 : tensor<144x1x3x3xf32>
    %v6246 = stablehlo.subtract %b3dW, %v6245 : tensor<144x1x3x3xf32>
    %v6247 = stablehlo.reshape %v6210 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6248 = stablehlo.reshape %v250 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6249 = stablehlo.logistic %v6248 : tensor<32x144x56x56xf32>
    %v6250 = stablehlo.constant dense<1.0> : tensor<32x144x56x56xf32>
    %v6251 = stablehlo.subtract %v6250, %v6249 : tensor<32x144x56x56xf32>
    %v6252 = stablehlo.multiply %v6248, %v6251 : tensor<32x144x56x56xf32>
    %v6253 = stablehlo.add %v6250, %v6252 : tensor<32x144x56x56xf32>
    %v6254 = stablehlo.multiply %v6249, %v6253 : tensor<32x144x56x56xf32>
    %v6255 = stablehlo.multiply %v6247, %v6254 : tensor<32x144x56x56xf32>
    %v6256 = stablehlo.reshape %v6255 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v6257 = stablehlo.reshape %v230 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6258 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6259 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v6260 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v6261 = stablehlo.reduce(%v6257 init: %v6258) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6262 = stablehlo.broadcast_in_dim %v6261, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6263 = stablehlo.divide %v6262, %v6259 : tensor<32x144x56x56xf32>
    %v6264 = stablehlo.subtract %v6257, %v6263 : tensor<32x144x56x56xf32>
    %v6265 = stablehlo.multiply %v6264, %v6264 : tensor<32x144x56x56xf32>
    %v6266 = stablehlo.reduce(%v6265 init: %v6258) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6267 = stablehlo.broadcast_in_dim %v6266, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6268 = stablehlo.divide %v6267, %v6259 : tensor<32x144x56x56xf32>
    %v6269 = stablehlo.add %v6268, %v6260 : tensor<32x144x56x56xf32>
    %v6270 = stablehlo.rsqrt %v6269 : tensor<32x144x56x56xf32>
    %v6271 = stablehlo.multiply %v6264, %v6270 : tensor<32x144x56x56xf32>
    %v6272 = stablehlo.broadcast_in_dim %b3eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6273 = stablehlo.reshape %v6256 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6274 = stablehlo.multiply %v6272, %v6273 : tensor<32x144x56x56xf32>
    %v6275 = stablehlo.reduce(%v6274 init: %v6258) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6276 = stablehlo.broadcast_in_dim %v6275, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6277 = stablehlo.multiply %v6271, %v6274 : tensor<32x144x56x56xf32>
    %v6278 = stablehlo.reduce(%v6277 init: %v6258) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6279 = stablehlo.broadcast_in_dim %v6278, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6280 = stablehlo.multiply %v6274, %v6259 : tensor<32x144x56x56xf32>
    %v6281 = stablehlo.subtract %v6280, %v6276 : tensor<32x144x56x56xf32>
    %v6282 = stablehlo.multiply %v6271, %v6279 : tensor<32x144x56x56xf32>
    %v6283 = stablehlo.subtract %v6281, %v6282 : tensor<32x144x56x56xf32>
    %v6284 = stablehlo.divide %v6270, %v6259 : tensor<32x144x56x56xf32>
    %v6285 = stablehlo.multiply %v6284, %v6283 : tensor<32x144x56x56xf32>
    %v6286 = stablehlo.reshape %v6285 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v6287 = stablehlo.reshape %v6286 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6288 = stablehlo.reverse %b3eW, dims = [2, 3] : tensor<144x24x1x1xf32>
    %v6289 = stablehlo.transpose %v6288, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %v6290 = stablehlo.convolution(%v6287, %v6289)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v6291 = stablehlo.reshape %v6290 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v6292 = stablehlo.reshape %v230 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6293 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6294 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v6295 = stablehlo.reduce(%v6292 init: %v6293) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6296 = stablehlo.broadcast_in_dim %v6295, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6297 = stablehlo.divide %v6296, %v6294 : tensor<32x144x56x56xf32>
    %v6298 = stablehlo.subtract %v6292, %v6297 : tensor<32x144x56x56xf32>
    %v6299 = stablehlo.multiply %v6298, %v6298 : tensor<32x144x56x56xf32>
    %v6300 = stablehlo.reduce(%v6299 init: %v6293) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6301 = stablehlo.broadcast_in_dim %v6300, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6302 = stablehlo.divide %v6301, %v6294 : tensor<32x144x56x56xf32>
    %v6303 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v6304 = stablehlo.add %v6302, %v6303 : tensor<32x144x56x56xf32>
    %v6305 = stablehlo.rsqrt %v6304 : tensor<32x144x56x56xf32>
    %v6306 = stablehlo.multiply %v6298, %v6305 : tensor<32x144x56x56xf32>
    %v6307 = stablehlo.reshape %v6256 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6308 = stablehlo.multiply %v6307, %v6306 : tensor<32x144x56x56xf32>
    %v6309 = stablehlo.reduce(%v6308 init: %v6293) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6310 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v6311 = stablehlo.multiply %v6309, %v6310 : tensor<144xf32>
    %v6312 = stablehlo.subtract %b3eg, %v6311 : tensor<144xf32>
    %v6313 = stablehlo.reshape %v6256 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6314 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6315 = stablehlo.reduce(%v6313 init: %v6314) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6316 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v6317 = stablehlo.multiply %v6315, %v6316 : tensor<144xf32>
    %v6318 = stablehlo.subtract %b3ebt, %v6317 : tensor<144xf32>
    %v6319 = stablehlo.reshape %v225 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6320 = stablehlo.reshape %v6286 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6321 = stablehlo.transpose %v6319, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v6322 = stablehlo.transpose %v6320, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v6323 = stablehlo.convolution(%v6321, %v6322)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<24x144x1x1xf32>
    %v6324 = stablehlo.transpose %v6323, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %v6325 = stablehlo.constant dense<0.05> : tensor<144x24x1x1xf32>
    %v6326 = stablehlo.multiply %v6324, %v6325 : tensor<144x24x1x1xf32>
    %v6327 = stablehlo.subtract %b3eW, %v6326 : tensor<144x24x1x1xf32>
    %v6328 = stablehlo.reshape %v6291 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6329 = stablehlo.reshape %v5984 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6330 = stablehlo.add %v6328, %v6329 : tensor<32x24x56x56xf32>
    %v6331 = stablehlo.reshape %v6330 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v6332 = stablehlo.reshape %v205 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6333 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6334 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v6335 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v6336 = stablehlo.reduce(%v6332 init: %v6333) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6337 = stablehlo.broadcast_in_dim %v6336, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6338 = stablehlo.divide %v6337, %v6334 : tensor<32x24x56x56xf32>
    %v6339 = stablehlo.subtract %v6332, %v6338 : tensor<32x24x56x56xf32>
    %v6340 = stablehlo.multiply %v6339, %v6339 : tensor<32x24x56x56xf32>
    %v6341 = stablehlo.reduce(%v6340 init: %v6333) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6342 = stablehlo.broadcast_in_dim %v6341, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6343 = stablehlo.divide %v6342, %v6334 : tensor<32x24x56x56xf32>
    %v6344 = stablehlo.add %v6343, %v6335 : tensor<32x24x56x56xf32>
    %v6345 = stablehlo.rsqrt %v6344 : tensor<32x24x56x56xf32>
    %v6346 = stablehlo.multiply %v6339, %v6345 : tensor<32x24x56x56xf32>
    %v6347 = stablehlo.broadcast_in_dim %b2pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6348 = stablehlo.reshape %v6331 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6349 = stablehlo.multiply %v6347, %v6348 : tensor<32x24x56x56xf32>
    %v6350 = stablehlo.reduce(%v6349 init: %v6333) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6351 = stablehlo.broadcast_in_dim %v6350, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6352 = stablehlo.multiply %v6346, %v6349 : tensor<32x24x56x56xf32>
    %v6353 = stablehlo.reduce(%v6352 init: %v6333) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6354 = stablehlo.broadcast_in_dim %v6353, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6355 = stablehlo.multiply %v6349, %v6334 : tensor<32x24x56x56xf32>
    %v6356 = stablehlo.subtract %v6355, %v6351 : tensor<32x24x56x56xf32>
    %v6357 = stablehlo.multiply %v6346, %v6354 : tensor<32x24x56x56xf32>
    %v6358 = stablehlo.subtract %v6356, %v6357 : tensor<32x24x56x56xf32>
    %v6359 = stablehlo.divide %v6345, %v6334 : tensor<32x24x56x56xf32>
    %v6360 = stablehlo.multiply %v6359, %v6358 : tensor<32x24x56x56xf32>
    %v6361 = stablehlo.reshape %v6360 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v6362 = stablehlo.reshape %v6361 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6363 = stablehlo.reverse %b2pW, dims = [2, 3] : tensor<24x96x1x1xf32>
    %v6364 = stablehlo.transpose %v6363, dims = [1, 0, 2, 3] : (tensor<24x96x1x1xf32>) -> tensor<96x24x1x1xf32>
    %v6365 = stablehlo.convolution(%v6362, %v6364)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<96x24x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v6366 = stablehlo.reshape %v6365 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v6367 = stablehlo.reshape %v205 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6368 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6369 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v6370 = stablehlo.reduce(%v6367 init: %v6368) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6371 = stablehlo.broadcast_in_dim %v6370, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6372 = stablehlo.divide %v6371, %v6369 : tensor<32x24x56x56xf32>
    %v6373 = stablehlo.subtract %v6367, %v6372 : tensor<32x24x56x56xf32>
    %v6374 = stablehlo.multiply %v6373, %v6373 : tensor<32x24x56x56xf32>
    %v6375 = stablehlo.reduce(%v6374 init: %v6368) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6376 = stablehlo.broadcast_in_dim %v6375, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6377 = stablehlo.divide %v6376, %v6369 : tensor<32x24x56x56xf32>
    %v6378 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v6379 = stablehlo.add %v6377, %v6378 : tensor<32x24x56x56xf32>
    %v6380 = stablehlo.rsqrt %v6379 : tensor<32x24x56x56xf32>
    %v6381 = stablehlo.multiply %v6373, %v6380 : tensor<32x24x56x56xf32>
    %v6382 = stablehlo.reshape %v6331 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6383 = stablehlo.multiply %v6382, %v6381 : tensor<32x24x56x56xf32>
    %v6384 = stablehlo.reduce(%v6383 init: %v6368) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6385 = stablehlo.constant dense<0.05> : tensor<24xf32>
    %v6386 = stablehlo.multiply %v6384, %v6385 : tensor<24xf32>
    %v6387 = stablehlo.subtract %b2pg, %v6386 : tensor<24xf32>
    %v6388 = stablehlo.reshape %v6331 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6389 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6390 = stablehlo.reduce(%v6388 init: %v6389) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6391 = stablehlo.constant dense<0.05> : tensor<24xf32>
    %v6392 = stablehlo.multiply %v6390, %v6391 : tensor<24xf32>
    %v6393 = stablehlo.subtract %b2pbt, %v6392 : tensor<24xf32>
    %v6394 = stablehlo.reshape %v200 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6395 = stablehlo.reshape %v6361 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6396 = stablehlo.transpose %v6394, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v6397 = stablehlo.transpose %v6395, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v6398 = stablehlo.convolution(%v6396, %v6397)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<24x32x56x56xf32>) -> tensor<96x24x1x1xf32>
    %v6399 = stablehlo.transpose %v6398, dims = [1, 0, 2, 3] : (tensor<96x24x1x1xf32>) -> tensor<24x96x1x1xf32>
    %v6400 = stablehlo.constant dense<0.05> : tensor<24x96x1x1xf32>
    %v6401 = stablehlo.multiply %v6399, %v6400 : tensor<24x96x1x1xf32>
    %v6402 = stablehlo.subtract %b2pW, %v6401 : tensor<24x96x1x1xf32>
    %v6403 = stablehlo.reshape %v170 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6404 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6405 = stablehlo.reduce(%v6403 init: %v6404) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v6406 = stablehlo.constant dense<3136.0> : tensor<32x96xf32>
    %v6407 = stablehlo.divide %v6405, %v6406 : tensor<32x96xf32>
    %v6408 = stablehlo.dot_general %v6407, %b2zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<96x4xf32>) -> tensor<32x4xf32>
    %v6409 = stablehlo.broadcast_in_dim %b2zb1, dims = [1] : (tensor<4xf32>) -> tensor<32x4xf32>
    %v6410 = stablehlo.add %v6408, %v6409 : tensor<32x4xf32>
    %v6411 = stablehlo.logistic %v6410 : tensor<32x4xf32>
    %v6412 = stablehlo.multiply %v6410, %v6411 : tensor<32x4xf32>
    %v6413 = stablehlo.dot_general %v6412, %b2zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<4x96xf32>) -> tensor<32x96xf32>
    %v6414 = stablehlo.broadcast_in_dim %b2zb2, dims = [1] : (tensor<96xf32>) -> tensor<32x96xf32>
    %v6415 = stablehlo.add %v6413, %v6414 : tensor<32x96xf32>
    %v6416 = stablehlo.logistic %v6415 : tensor<32x96xf32>
    %v6417 = stablehlo.reshape %v6366 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6418 = stablehlo.broadcast_in_dim %v6416, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v6419 = stablehlo.multiply %v6418, %v6417 : tensor<32x96x56x56xf32>
    %v6420 = stablehlo.multiply %v6403, %v6417 : tensor<32x96x56x56xf32>
    %v6421 = stablehlo.reduce(%v6420 init: %v6404) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v6422 = stablehlo.constant dense<1.0> : tensor<32x96xf32>
    %v6423 = stablehlo.subtract %v6422, %v6416 : tensor<32x96xf32>
    %v6424 = stablehlo.multiply %v6416, %v6423 : tensor<32x96xf32>
    %v6425 = stablehlo.multiply %v6421, %v6424 : tensor<32x96xf32>
    %v6426 = stablehlo.dot_general %v6425, %b2zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<4x96xf32>) -> tensor<32x4xf32>
    %v6427 = stablehlo.logistic %v6410 : tensor<32x4xf32>
    %v6428 = stablehlo.constant dense<1.0> : tensor<32x4xf32>
    %v6429 = stablehlo.subtract %v6428, %v6427 : tensor<32x4xf32>
    %v6430 = stablehlo.multiply %v6410, %v6429 : tensor<32x4xf32>
    %v6431 = stablehlo.add %v6428, %v6430 : tensor<32x4xf32>
    %v6432 = stablehlo.multiply %v6427, %v6431 : tensor<32x4xf32>
    %v6433 = stablehlo.multiply %v6426, %v6432 : tensor<32x4xf32>
    %v6434 = stablehlo.dot_general %v6433, %b2zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<96x4xf32>) -> tensor<32x96xf32>
    %v6435 = stablehlo.constant dense<3136.0> : tensor<32x96xf32>
    %v6436 = stablehlo.divide %v6434, %v6435 : tensor<32x96xf32>
    %v6437 = stablehlo.broadcast_in_dim %v6436, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v6438 = stablehlo.add %v6419, %v6437 : tensor<32x96x56x56xf32>
    %v6439 = stablehlo.reshape %v6438 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v6440 = stablehlo.reshape %v170 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6441 = stablehlo.reshape %v6366 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6442 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6443 = stablehlo.multiply %v6440, %v6441 : tensor<32x96x56x56xf32>
    %v6444 = stablehlo.reduce(%v6443 init: %v6442) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v6445 = stablehlo.logistic %v183 : tensor<32x96xf32>
    %v6446 = stablehlo.constant dense<1.0> : tensor<32x96xf32>
    %v6447 = stablehlo.subtract %v6446, %v6445 : tensor<32x96xf32>
    %v6448 = stablehlo.multiply %v6445, %v6447 : tensor<32x96xf32>
    %v6449 = stablehlo.multiply %v6444, %v6448 : tensor<32x96xf32>
    %v6450 = stablehlo.dot_general %v180, %v6449, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<32x96xf32>) -> tensor<4x96xf32>
    %v6451 = stablehlo.constant dense<0.05> : tensor<4x96xf32>
    %v6452 = stablehlo.multiply %v6450, %v6451 : tensor<4x96xf32>
    %v6453 = stablehlo.subtract %b2zW2, %v6452 : tensor<4x96xf32>
    %v6454 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6455 = stablehlo.reduce(%v6449 init: %v6454) applies stablehlo.add across dimensions = [0] : (tensor<32x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v6456 = stablehlo.constant dense<0.05> : tensor<96xf32>
    %v6457 = stablehlo.multiply %v6455, %v6456 : tensor<96xf32>
    %v6458 = stablehlo.subtract %b2zb2, %v6457 : tensor<96xf32>
    %v6459 = stablehlo.reshape %v6449 : (tensor<32x96xf32>) -> tensor<32x1x96xf32>
    %v6460 = stablehlo.dot_general %v6459, %b2zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x96xf32>, tensor<4x96xf32>) -> tensor<32x1x4xf32>
    %v6461 = stablehlo.reshape %v6460 : (tensor<32x1x4xf32>) -> tensor<32x4xf32>
    %v6462 = stablehlo.logistic %v178 : tensor<32x4xf32>
    %v6463 = stablehlo.constant dense<1.0> : tensor<32x4xf32>
    %v6464 = stablehlo.subtract %v6463, %v6462 : tensor<32x4xf32>
    %v6465 = stablehlo.multiply %v178, %v6464 : tensor<32x4xf32>
    %v6466 = stablehlo.add %v6463, %v6465 : tensor<32x4xf32>
    %v6467 = stablehlo.multiply %v6462, %v6466 : tensor<32x4xf32>
    %v6468 = stablehlo.multiply %v6461, %v6467 : tensor<32x4xf32>
    %v6469 = stablehlo.dot_general %v175, %v6468, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<32x4xf32>) -> tensor<96x4xf32>
    %v6470 = stablehlo.constant dense<0.05> : tensor<96x4xf32>
    %v6471 = stablehlo.multiply %v6469, %v6470 : tensor<96x4xf32>
    %v6472 = stablehlo.subtract %b2zW1, %v6471 : tensor<96x4xf32>
    %v6473 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6474 = stablehlo.reduce(%v6468 init: %v6473) applies stablehlo.add across dimensions = [0] : (tensor<32x4xf32>, tensor<f32>) -> tensor<4xf32>
    %v6475 = stablehlo.constant dense<0.05> : tensor<4xf32>
    %v6476 = stablehlo.multiply %v6474, %v6475 : tensor<4xf32>
    %v6477 = stablehlo.subtract %b2zb1, %v6476 : tensor<4xf32>
    %v6478 = stablehlo.reshape %v6439 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6479 = stablehlo.reshape %v166 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6480 = stablehlo.logistic %v6479 : tensor<32x96x56x56xf32>
    %v6481 = stablehlo.constant dense<1.0> : tensor<32x96x56x56xf32>
    %v6482 = stablehlo.subtract %v6481, %v6480 : tensor<32x96x56x56xf32>
    %v6483 = stablehlo.multiply %v6479, %v6482 : tensor<32x96x56x56xf32>
    %v6484 = stablehlo.add %v6481, %v6483 : tensor<32x96x56x56xf32>
    %v6485 = stablehlo.multiply %v6480, %v6484 : tensor<32x96x56x56xf32>
    %v6486 = stablehlo.multiply %v6478, %v6485 : tensor<32x96x56x56xf32>
    %v6487 = stablehlo.reshape %v6486 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v6488 = stablehlo.reshape %v146 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6489 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6490 = stablehlo.constant dense<100352.0> : tensor<32x96x56x56xf32>
    %v6491 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v6492 = stablehlo.reduce(%v6488 init: %v6489) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6493 = stablehlo.broadcast_in_dim %v6492, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v6494 = stablehlo.divide %v6493, %v6490 : tensor<32x96x56x56xf32>
    %v6495 = stablehlo.subtract %v6488, %v6494 : tensor<32x96x56x56xf32>
    %v6496 = stablehlo.multiply %v6495, %v6495 : tensor<32x96x56x56xf32>
    %v6497 = stablehlo.reduce(%v6496 init: %v6489) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6498 = stablehlo.broadcast_in_dim %v6497, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v6499 = stablehlo.divide %v6498, %v6490 : tensor<32x96x56x56xf32>
    %v6500 = stablehlo.add %v6499, %v6491 : tensor<32x96x56x56xf32>
    %v6501 = stablehlo.rsqrt %v6500 : tensor<32x96x56x56xf32>
    %v6502 = stablehlo.multiply %v6495, %v6501 : tensor<32x96x56x56xf32>
    %v6503 = stablehlo.broadcast_in_dim %b2dg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v6504 = stablehlo.reshape %v6487 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6505 = stablehlo.multiply %v6503, %v6504 : tensor<32x96x56x56xf32>
    %v6506 = stablehlo.reduce(%v6505 init: %v6489) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6507 = stablehlo.broadcast_in_dim %v6506, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v6508 = stablehlo.multiply %v6502, %v6505 : tensor<32x96x56x56xf32>
    %v6509 = stablehlo.reduce(%v6508 init: %v6489) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6510 = stablehlo.broadcast_in_dim %v6509, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v6511 = stablehlo.multiply %v6505, %v6490 : tensor<32x96x56x56xf32>
    %v6512 = stablehlo.subtract %v6511, %v6507 : tensor<32x96x56x56xf32>
    %v6513 = stablehlo.multiply %v6502, %v6510 : tensor<32x96x56x56xf32>
    %v6514 = stablehlo.subtract %v6512, %v6513 : tensor<32x96x56x56xf32>
    %v6515 = stablehlo.divide %v6501, %v6490 : tensor<32x96x56x56xf32>
    %v6516 = stablehlo.multiply %v6515, %v6514 : tensor<32x96x56x56xf32>
    %v6517 = stablehlo.reshape %v6516 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v6518 = stablehlo.reshape %v6517 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6519 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6520 = stablehlo.pad %v6518, %v6519, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96x112x112xf32>
    %v6521 = stablehlo.reverse %b2dW, dims = [2, 3] : tensor<96x1x3x3xf32>
    %v6522 = stablehlo.convolution(%v6520, %v6521)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x112x112xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x112x112xf32>
    %v6523 = stablehlo.reshape %v6522 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v6524 = stablehlo.reshape %v146 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6525 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6526 = stablehlo.constant dense<100352.0> : tensor<32x96x56x56xf32>
    %v6527 = stablehlo.reduce(%v6524 init: %v6525) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6528 = stablehlo.broadcast_in_dim %v6527, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v6529 = stablehlo.divide %v6528, %v6526 : tensor<32x96x56x56xf32>
    %v6530 = stablehlo.subtract %v6524, %v6529 : tensor<32x96x56x56xf32>
    %v6531 = stablehlo.multiply %v6530, %v6530 : tensor<32x96x56x56xf32>
    %v6532 = stablehlo.reduce(%v6531 init: %v6525) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6533 = stablehlo.broadcast_in_dim %v6532, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v6534 = stablehlo.divide %v6533, %v6526 : tensor<32x96x56x56xf32>
    %v6535 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v6536 = stablehlo.add %v6534, %v6535 : tensor<32x96x56x56xf32>
    %v6537 = stablehlo.rsqrt %v6536 : tensor<32x96x56x56xf32>
    %v6538 = stablehlo.multiply %v6530, %v6537 : tensor<32x96x56x56xf32>
    %v6539 = stablehlo.reshape %v6487 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6540 = stablehlo.multiply %v6539, %v6538 : tensor<32x96x56x56xf32>
    %v6541 = stablehlo.reduce(%v6540 init: %v6525) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6542 = stablehlo.constant dense<0.05> : tensor<96xf32>
    %v6543 = stablehlo.multiply %v6541, %v6542 : tensor<96xf32>
    %v6544 = stablehlo.subtract %b2dg, %v6543 : tensor<96xf32>
    %v6545 = stablehlo.reshape %v6487 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6546 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6547 = stablehlo.reduce(%v6545 init: %v6546) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6548 = stablehlo.constant dense<0.05> : tensor<96xf32>
    %v6549 = stablehlo.multiply %v6547, %v6548 : tensor<96xf32>
    %v6550 = stablehlo.subtract %b2dbt, %v6549 : tensor<96xf32>
    %v6551 = stablehlo.reshape %v141 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6552 = stablehlo.reshape %v6517 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6553 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6554 = stablehlo.pad %v6552, %v6553, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96x112x112xf32>
    %v6555 = stablehlo.transpose %v6551, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %v6556 = stablehlo.transpose %v6554, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %v6557 = stablehlo.convolution(%v6555, %v6556)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x112x112xf32>, tensor<96x32x112x112xf32>) -> tensor<1x96x3x3xf32>
    %v6558 = stablehlo.reshape %v6557 : (tensor<1x96x3x3xf32>) -> tensor<96x1x3x3xf32>
    %v6559 = stablehlo.constant dense<0.05> : tensor<96x1x3x3xf32>
    %v6560 = stablehlo.multiply %v6558, %v6559 : tensor<96x1x3x3xf32>
    %v6561 = stablehlo.subtract %b2dW, %v6560 : tensor<96x1x3x3xf32>
    %v6562 = stablehlo.reshape %v6523 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6563 = stablehlo.reshape %v137 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6564 = stablehlo.logistic %v6563 : tensor<32x96x112x112xf32>
    %v6565 = stablehlo.constant dense<1.0> : tensor<32x96x112x112xf32>
    %v6566 = stablehlo.subtract %v6565, %v6564 : tensor<32x96x112x112xf32>
    %v6567 = stablehlo.multiply %v6563, %v6566 : tensor<32x96x112x112xf32>
    %v6568 = stablehlo.add %v6565, %v6567 : tensor<32x96x112x112xf32>
    %v6569 = stablehlo.multiply %v6564, %v6568 : tensor<32x96x112x112xf32>
    %v6570 = stablehlo.multiply %v6562, %v6569 : tensor<32x96x112x112xf32>
    %v6571 = stablehlo.reshape %v6570 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v6572 = stablehlo.reshape %v117 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6573 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6574 = stablehlo.constant dense<401408.0> : tensor<32x96x112x112xf32>
    %v6575 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v6576 = stablehlo.reduce(%v6572 init: %v6573) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6577 = stablehlo.broadcast_in_dim %v6576, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v6578 = stablehlo.divide %v6577, %v6574 : tensor<32x96x112x112xf32>
    %v6579 = stablehlo.subtract %v6572, %v6578 : tensor<32x96x112x112xf32>
    %v6580 = stablehlo.multiply %v6579, %v6579 : tensor<32x96x112x112xf32>
    %v6581 = stablehlo.reduce(%v6580 init: %v6573) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6582 = stablehlo.broadcast_in_dim %v6581, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v6583 = stablehlo.divide %v6582, %v6574 : tensor<32x96x112x112xf32>
    %v6584 = stablehlo.add %v6583, %v6575 : tensor<32x96x112x112xf32>
    %v6585 = stablehlo.rsqrt %v6584 : tensor<32x96x112x112xf32>
    %v6586 = stablehlo.multiply %v6579, %v6585 : tensor<32x96x112x112xf32>
    %v6587 = stablehlo.broadcast_in_dim %b2eg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v6588 = stablehlo.reshape %v6571 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6589 = stablehlo.multiply %v6587, %v6588 : tensor<32x96x112x112xf32>
    %v6590 = stablehlo.reduce(%v6589 init: %v6573) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6591 = stablehlo.broadcast_in_dim %v6590, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v6592 = stablehlo.multiply %v6586, %v6589 : tensor<32x96x112x112xf32>
    %v6593 = stablehlo.reduce(%v6592 init: %v6573) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6594 = stablehlo.broadcast_in_dim %v6593, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v6595 = stablehlo.multiply %v6589, %v6574 : tensor<32x96x112x112xf32>
    %v6596 = stablehlo.subtract %v6595, %v6591 : tensor<32x96x112x112xf32>
    %v6597 = stablehlo.multiply %v6586, %v6594 : tensor<32x96x112x112xf32>
    %v6598 = stablehlo.subtract %v6596, %v6597 : tensor<32x96x112x112xf32>
    %v6599 = stablehlo.divide %v6585, %v6574 : tensor<32x96x112x112xf32>
    %v6600 = stablehlo.multiply %v6599, %v6598 : tensor<32x96x112x112xf32>
    %v6601 = stablehlo.reshape %v6600 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v6602 = stablehlo.reshape %v6601 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6603 = stablehlo.reverse %b2eW, dims = [2, 3] : tensor<96x16x1x1xf32>
    %v6604 = stablehlo.transpose %v6603, dims = [1, 0, 2, 3] : (tensor<96x16x1x1xf32>) -> tensor<16x96x1x1xf32>
    %v6605 = stablehlo.convolution(%v6602, %v6604)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x112x112xf32>, tensor<16x96x1x1xf32>) -> tensor<32x16x112x112xf32>
    %v6606 = stablehlo.reshape %v6605 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v6607 = stablehlo.reshape %v117 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6608 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6609 = stablehlo.constant dense<401408.0> : tensor<32x96x112x112xf32>
    %v6610 = stablehlo.reduce(%v6607 init: %v6608) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6611 = stablehlo.broadcast_in_dim %v6610, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v6612 = stablehlo.divide %v6611, %v6609 : tensor<32x96x112x112xf32>
    %v6613 = stablehlo.subtract %v6607, %v6612 : tensor<32x96x112x112xf32>
    %v6614 = stablehlo.multiply %v6613, %v6613 : tensor<32x96x112x112xf32>
    %v6615 = stablehlo.reduce(%v6614 init: %v6608) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6616 = stablehlo.broadcast_in_dim %v6615, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v6617 = stablehlo.divide %v6616, %v6609 : tensor<32x96x112x112xf32>
    %v6618 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v6619 = stablehlo.add %v6617, %v6618 : tensor<32x96x112x112xf32>
    %v6620 = stablehlo.rsqrt %v6619 : tensor<32x96x112x112xf32>
    %v6621 = stablehlo.multiply %v6613, %v6620 : tensor<32x96x112x112xf32>
    %v6622 = stablehlo.reshape %v6571 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6623 = stablehlo.multiply %v6622, %v6621 : tensor<32x96x112x112xf32>
    %v6624 = stablehlo.reduce(%v6623 init: %v6608) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6625 = stablehlo.constant dense<0.05> : tensor<96xf32>
    %v6626 = stablehlo.multiply %v6624, %v6625 : tensor<96xf32>
    %v6627 = stablehlo.subtract %b2eg, %v6626 : tensor<96xf32>
    %v6628 = stablehlo.reshape %v6571 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6629 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6630 = stablehlo.reduce(%v6628 init: %v6629) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6631 = stablehlo.constant dense<0.05> : tensor<96xf32>
    %v6632 = stablehlo.multiply %v6630, %v6631 : tensor<96xf32>
    %v6633 = stablehlo.subtract %b2ebt, %v6632 : tensor<96xf32>
    %v6634 = stablehlo.reshape %v112 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6635 = stablehlo.reshape %v6601 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6636 = stablehlo.transpose %v6634, dims = [1, 0, 2, 3] : (tensor<32x16x112x112xf32>) -> tensor<16x32x112x112xf32>
    %v6637 = stablehlo.transpose %v6635, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %v6638 = stablehlo.convolution(%v6636, %v6637)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x32x112x112xf32>, tensor<96x32x112x112xf32>) -> tensor<16x96x1x1xf32>
    %v6639 = stablehlo.transpose %v6638, dims = [1, 0, 2, 3] : (tensor<16x96x1x1xf32>) -> tensor<96x16x1x1xf32>
    %v6640 = stablehlo.constant dense<0.05> : tensor<96x16x1x1xf32>
    %v6641 = stablehlo.multiply %v6639, %v6640 : tensor<96x16x1x1xf32>
    %v6642 = stablehlo.subtract %b2eW, %v6641 : tensor<96x16x1x1xf32>
    %v6643 = stablehlo.reshape %v92 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6644 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6645 = stablehlo.constant dense<401408.0> : tensor<32x16x112x112xf32>
    %v6646 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v6647 = stablehlo.reduce(%v6643 init: %v6644) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6648 = stablehlo.broadcast_in_dim %v6647, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v6649 = stablehlo.divide %v6648, %v6645 : tensor<32x16x112x112xf32>
    %v6650 = stablehlo.subtract %v6643, %v6649 : tensor<32x16x112x112xf32>
    %v6651 = stablehlo.multiply %v6650, %v6650 : tensor<32x16x112x112xf32>
    %v6652 = stablehlo.reduce(%v6651 init: %v6644) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6653 = stablehlo.broadcast_in_dim %v6652, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v6654 = stablehlo.divide %v6653, %v6645 : tensor<32x16x112x112xf32>
    %v6655 = stablehlo.add %v6654, %v6646 : tensor<32x16x112x112xf32>
    %v6656 = stablehlo.rsqrt %v6655 : tensor<32x16x112x112xf32>
    %v6657 = stablehlo.multiply %v6650, %v6656 : tensor<32x16x112x112xf32>
    %v6658 = stablehlo.broadcast_in_dim %b1pg, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v6659 = stablehlo.reshape %v6606 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6660 = stablehlo.multiply %v6658, %v6659 : tensor<32x16x112x112xf32>
    %v6661 = stablehlo.reduce(%v6660 init: %v6644) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6662 = stablehlo.broadcast_in_dim %v6661, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v6663 = stablehlo.multiply %v6657, %v6660 : tensor<32x16x112x112xf32>
    %v6664 = stablehlo.reduce(%v6663 init: %v6644) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6665 = stablehlo.broadcast_in_dim %v6664, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v6666 = stablehlo.multiply %v6660, %v6645 : tensor<32x16x112x112xf32>
    %v6667 = stablehlo.subtract %v6666, %v6662 : tensor<32x16x112x112xf32>
    %v6668 = stablehlo.multiply %v6657, %v6665 : tensor<32x16x112x112xf32>
    %v6669 = stablehlo.subtract %v6667, %v6668 : tensor<32x16x112x112xf32>
    %v6670 = stablehlo.divide %v6656, %v6645 : tensor<32x16x112x112xf32>
    %v6671 = stablehlo.multiply %v6670, %v6669 : tensor<32x16x112x112xf32>
    %v6672 = stablehlo.reshape %v6671 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v6673 = stablehlo.reshape %v6672 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6674 = stablehlo.reverse %b1pW, dims = [2, 3] : tensor<16x32x1x1xf32>
    %v6675 = stablehlo.transpose %v6674, dims = [1, 0, 2, 3] : (tensor<16x32x1x1xf32>) -> tensor<32x16x1x1xf32>
    %v6676 = stablehlo.convolution(%v6673, %v6675)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x16x112x112xf32>, tensor<32x16x1x1xf32>) -> tensor<32x32x112x112xf32>
    %v6677 = stablehlo.reshape %v6676 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v6678 = stablehlo.reshape %v92 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6679 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6680 = stablehlo.constant dense<401408.0> : tensor<32x16x112x112xf32>
    %v6681 = stablehlo.reduce(%v6678 init: %v6679) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6682 = stablehlo.broadcast_in_dim %v6681, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v6683 = stablehlo.divide %v6682, %v6680 : tensor<32x16x112x112xf32>
    %v6684 = stablehlo.subtract %v6678, %v6683 : tensor<32x16x112x112xf32>
    %v6685 = stablehlo.multiply %v6684, %v6684 : tensor<32x16x112x112xf32>
    %v6686 = stablehlo.reduce(%v6685 init: %v6679) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6687 = stablehlo.broadcast_in_dim %v6686, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v6688 = stablehlo.divide %v6687, %v6680 : tensor<32x16x112x112xf32>
    %v6689 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v6690 = stablehlo.add %v6688, %v6689 : tensor<32x16x112x112xf32>
    %v6691 = stablehlo.rsqrt %v6690 : tensor<32x16x112x112xf32>
    %v6692 = stablehlo.multiply %v6684, %v6691 : tensor<32x16x112x112xf32>
    %v6693 = stablehlo.reshape %v6606 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6694 = stablehlo.multiply %v6693, %v6692 : tensor<32x16x112x112xf32>
    %v6695 = stablehlo.reduce(%v6694 init: %v6679) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6696 = stablehlo.constant dense<0.05> : tensor<16xf32>
    %v6697 = stablehlo.multiply %v6695, %v6696 : tensor<16xf32>
    %v6698 = stablehlo.subtract %b1pg, %v6697 : tensor<16xf32>
    %v6699 = stablehlo.reshape %v6606 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6700 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6701 = stablehlo.reduce(%v6699 init: %v6700) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6702 = stablehlo.constant dense<0.05> : tensor<16xf32>
    %v6703 = stablehlo.multiply %v6701, %v6702 : tensor<16xf32>
    %v6704 = stablehlo.subtract %b1pbt, %v6703 : tensor<16xf32>
    %v6705 = stablehlo.reshape %v87 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6706 = stablehlo.reshape %v6672 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6707 = stablehlo.transpose %v6705, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %v6708 = stablehlo.transpose %v6706, dims = [1, 0, 2, 3] : (tensor<32x16x112x112xf32>) -> tensor<16x32x112x112xf32>
    %v6709 = stablehlo.convolution(%v6707, %v6708)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<16x32x112x112xf32>) -> tensor<32x16x1x1xf32>
    %v6710 = stablehlo.transpose %v6709, dims = [1, 0, 2, 3] : (tensor<32x16x1x1xf32>) -> tensor<16x32x1x1xf32>
    %v6711 = stablehlo.constant dense<0.05> : tensor<16x32x1x1xf32>
    %v6712 = stablehlo.multiply %v6710, %v6711 : tensor<16x32x1x1xf32>
    %v6713 = stablehlo.subtract %b1pW, %v6712 : tensor<16x32x1x1xf32>
    %v6714 = stablehlo.reshape %v57 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6715 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6716 = stablehlo.reduce(%v6714 init: %v6715) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v6717 = stablehlo.constant dense<12544.0> : tensor<32x32xf32>
    %v6718 = stablehlo.divide %v6716, %v6717 : tensor<32x32xf32>
    %v6719 = stablehlo.dot_general %v6718, %b1zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x8xf32>) -> tensor<32x8xf32>
    %v6720 = stablehlo.broadcast_in_dim %b1zb1, dims = [1] : (tensor<8xf32>) -> tensor<32x8xf32>
    %v6721 = stablehlo.add %v6719, %v6720 : tensor<32x8xf32>
    %v6722 = stablehlo.logistic %v6721 : tensor<32x8xf32>
    %v6723 = stablehlo.multiply %v6721, %v6722 : tensor<32x8xf32>
    %v6724 = stablehlo.dot_general %v6723, %b1zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x8xf32>, tensor<8x32xf32>) -> tensor<32x32xf32>
    %v6725 = stablehlo.broadcast_in_dim %b1zb2, dims = [1] : (tensor<32xf32>) -> tensor<32x32xf32>
    %v6726 = stablehlo.add %v6724, %v6725 : tensor<32x32xf32>
    %v6727 = stablehlo.logistic %v6726 : tensor<32x32xf32>
    %v6728 = stablehlo.reshape %v6677 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6729 = stablehlo.broadcast_in_dim %v6727, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v6730 = stablehlo.multiply %v6729, %v6728 : tensor<32x32x112x112xf32>
    %v6731 = stablehlo.multiply %v6714, %v6728 : tensor<32x32x112x112xf32>
    %v6732 = stablehlo.reduce(%v6731 init: %v6715) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v6733 = stablehlo.constant dense<1.0> : tensor<32x32xf32>
    %v6734 = stablehlo.subtract %v6733, %v6727 : tensor<32x32xf32>
    %v6735 = stablehlo.multiply %v6727, %v6734 : tensor<32x32xf32>
    %v6736 = stablehlo.multiply %v6732, %v6735 : tensor<32x32xf32>
    %v6737 = stablehlo.dot_general %v6736, %b1zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<8x32xf32>) -> tensor<32x8xf32>
    %v6738 = stablehlo.logistic %v6721 : tensor<32x8xf32>
    %v6739 = stablehlo.constant dense<1.0> : tensor<32x8xf32>
    %v6740 = stablehlo.subtract %v6739, %v6738 : tensor<32x8xf32>
    %v6741 = stablehlo.multiply %v6721, %v6740 : tensor<32x8xf32>
    %v6742 = stablehlo.add %v6739, %v6741 : tensor<32x8xf32>
    %v6743 = stablehlo.multiply %v6738, %v6742 : tensor<32x8xf32>
    %v6744 = stablehlo.multiply %v6737, %v6743 : tensor<32x8xf32>
    %v6745 = stablehlo.dot_general %v6744, %b1zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x8xf32>, tensor<32x8xf32>) -> tensor<32x32xf32>
    %v6746 = stablehlo.constant dense<12544.0> : tensor<32x32xf32>
    %v6747 = stablehlo.divide %v6745, %v6746 : tensor<32x32xf32>
    %v6748 = stablehlo.broadcast_in_dim %v6747, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v6749 = stablehlo.add %v6730, %v6748 : tensor<32x32x112x112xf32>
    %v6750 = stablehlo.reshape %v6749 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v6751 = stablehlo.reshape %v57 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6752 = stablehlo.reshape %v6677 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6753 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6754 = stablehlo.multiply %v6751, %v6752 : tensor<32x32x112x112xf32>
    %v6755 = stablehlo.reduce(%v6754 init: %v6753) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v6756 = stablehlo.logistic %v70 : tensor<32x32xf32>
    %v6757 = stablehlo.constant dense<1.0> : tensor<32x32xf32>
    %v6758 = stablehlo.subtract %v6757, %v6756 : tensor<32x32xf32>
    %v6759 = stablehlo.multiply %v6756, %v6758 : tensor<32x32xf32>
    %v6760 = stablehlo.multiply %v6755, %v6759 : tensor<32x32xf32>
    %v6761 = stablehlo.dot_general %v67, %v6760, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x8xf32>, tensor<32x32xf32>) -> tensor<8x32xf32>
    %v6762 = stablehlo.constant dense<0.05> : tensor<8x32xf32>
    %v6763 = stablehlo.multiply %v6761, %v6762 : tensor<8x32xf32>
    %v6764 = stablehlo.subtract %b1zW2, %v6763 : tensor<8x32xf32>
    %v6765 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6766 = stablehlo.reduce(%v6760 init: %v6765) applies stablehlo.add across dimensions = [0] : (tensor<32x32xf32>, tensor<f32>) -> tensor<32xf32>
    %v6767 = stablehlo.constant dense<0.05> : tensor<32xf32>
    %v6768 = stablehlo.multiply %v6766, %v6767 : tensor<32xf32>
    %v6769 = stablehlo.subtract %b1zb2, %v6768 : tensor<32xf32>
    %v6770 = stablehlo.reshape %v6760 : (tensor<32x32xf32>) -> tensor<32x1x32xf32>
    %v6771 = stablehlo.dot_general %v6770, %b1zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x32xf32>, tensor<8x32xf32>) -> tensor<32x1x8xf32>
    %v6772 = stablehlo.reshape %v6771 : (tensor<32x1x8xf32>) -> tensor<32x8xf32>
    %v6773 = stablehlo.logistic %v65 : tensor<32x8xf32>
    %v6774 = stablehlo.constant dense<1.0> : tensor<32x8xf32>
    %v6775 = stablehlo.subtract %v6774, %v6773 : tensor<32x8xf32>
    %v6776 = stablehlo.multiply %v65, %v6775 : tensor<32x8xf32>
    %v6777 = stablehlo.add %v6774, %v6776 : tensor<32x8xf32>
    %v6778 = stablehlo.multiply %v6773, %v6777 : tensor<32x8xf32>
    %v6779 = stablehlo.multiply %v6772, %v6778 : tensor<32x8xf32>
    %v6780 = stablehlo.dot_general %v62, %v6779, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x8xf32>) -> tensor<32x8xf32>
    %v6781 = stablehlo.constant dense<0.05> : tensor<32x8xf32>
    %v6782 = stablehlo.multiply %v6780, %v6781 : tensor<32x8xf32>
    %v6783 = stablehlo.subtract %b1zW1, %v6782 : tensor<32x8xf32>
    %v6784 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6785 = stablehlo.reduce(%v6779 init: %v6784) applies stablehlo.add across dimensions = [0] : (tensor<32x8xf32>, tensor<f32>) -> tensor<8xf32>
    %v6786 = stablehlo.constant dense<0.05> : tensor<8xf32>
    %v6787 = stablehlo.multiply %v6785, %v6786 : tensor<8xf32>
    %v6788 = stablehlo.subtract %b1zb1, %v6787 : tensor<8xf32>
    %v6789 = stablehlo.reshape %v6750 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6790 = stablehlo.reshape %v53 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6791 = stablehlo.logistic %v6790 : tensor<32x32x112x112xf32>
    %v6792 = stablehlo.constant dense<1.0> : tensor<32x32x112x112xf32>
    %v6793 = stablehlo.subtract %v6792, %v6791 : tensor<32x32x112x112xf32>
    %v6794 = stablehlo.multiply %v6790, %v6793 : tensor<32x32x112x112xf32>
    %v6795 = stablehlo.add %v6792, %v6794 : tensor<32x32x112x112xf32>
    %v6796 = stablehlo.multiply %v6791, %v6795 : tensor<32x32x112x112xf32>
    %v6797 = stablehlo.multiply %v6789, %v6796 : tensor<32x32x112x112xf32>
    %v6798 = stablehlo.reshape %v6797 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v6799 = stablehlo.reshape %v33 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6800 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6801 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v6802 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v6803 = stablehlo.reduce(%v6799 init: %v6800) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6804 = stablehlo.broadcast_in_dim %v6803, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6805 = stablehlo.divide %v6804, %v6801 : tensor<32x32x112x112xf32>
    %v6806 = stablehlo.subtract %v6799, %v6805 : tensor<32x32x112x112xf32>
    %v6807 = stablehlo.multiply %v6806, %v6806 : tensor<32x32x112x112xf32>
    %v6808 = stablehlo.reduce(%v6807 init: %v6800) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6809 = stablehlo.broadcast_in_dim %v6808, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6810 = stablehlo.divide %v6809, %v6801 : tensor<32x32x112x112xf32>
    %v6811 = stablehlo.add %v6810, %v6802 : tensor<32x32x112x112xf32>
    %v6812 = stablehlo.rsqrt %v6811 : tensor<32x32x112x112xf32>
    %v6813 = stablehlo.multiply %v6806, %v6812 : tensor<32x32x112x112xf32>
    %v6814 = stablehlo.broadcast_in_dim %b1dg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6815 = stablehlo.reshape %v6798 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6816 = stablehlo.multiply %v6814, %v6815 : tensor<32x32x112x112xf32>
    %v6817 = stablehlo.reduce(%v6816 init: %v6800) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6818 = stablehlo.broadcast_in_dim %v6817, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6819 = stablehlo.multiply %v6813, %v6816 : tensor<32x32x112x112xf32>
    %v6820 = stablehlo.reduce(%v6819 init: %v6800) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6821 = stablehlo.broadcast_in_dim %v6820, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6822 = stablehlo.multiply %v6816, %v6801 : tensor<32x32x112x112xf32>
    %v6823 = stablehlo.subtract %v6822, %v6818 : tensor<32x32x112x112xf32>
    %v6824 = stablehlo.multiply %v6813, %v6821 : tensor<32x32x112x112xf32>
    %v6825 = stablehlo.subtract %v6823, %v6824 : tensor<32x32x112x112xf32>
    %v6826 = stablehlo.divide %v6812, %v6801 : tensor<32x32x112x112xf32>
    %v6827 = stablehlo.multiply %v6826, %v6825 : tensor<32x32x112x112xf32>
    %v6828 = stablehlo.reshape %v6827 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v6829 = stablehlo.reshape %v6828 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6830 = stablehlo.reverse %b1dW, dims = [2, 3] : tensor<32x1x3x3xf32>
    %v6831 = stablehlo.convolution(%v6829, %v6830)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 32 : i64} : (tensor<32x32x112x112xf32>, tensor<32x1x3x3xf32>) -> tensor<32x32x112x112xf32>
    %v6832 = stablehlo.reshape %v6831 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v6833 = stablehlo.reshape %v33 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6834 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6835 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v6836 = stablehlo.reduce(%v6833 init: %v6834) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6837 = stablehlo.broadcast_in_dim %v6836, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6838 = stablehlo.divide %v6837, %v6835 : tensor<32x32x112x112xf32>
    %v6839 = stablehlo.subtract %v6833, %v6838 : tensor<32x32x112x112xf32>
    %v6840 = stablehlo.multiply %v6839, %v6839 : tensor<32x32x112x112xf32>
    %v6841 = stablehlo.reduce(%v6840 init: %v6834) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6842 = stablehlo.broadcast_in_dim %v6841, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6843 = stablehlo.divide %v6842, %v6835 : tensor<32x32x112x112xf32>
    %v6844 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v6845 = stablehlo.add %v6843, %v6844 : tensor<32x32x112x112xf32>
    %v6846 = stablehlo.rsqrt %v6845 : tensor<32x32x112x112xf32>
    %v6847 = stablehlo.multiply %v6839, %v6846 : tensor<32x32x112x112xf32>
    %v6848 = stablehlo.reshape %v6798 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6849 = stablehlo.multiply %v6848, %v6847 : tensor<32x32x112x112xf32>
    %v6850 = stablehlo.reduce(%v6849 init: %v6834) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6851 = stablehlo.constant dense<0.05> : tensor<32xf32>
    %v6852 = stablehlo.multiply %v6850, %v6851 : tensor<32xf32>
    %v6853 = stablehlo.subtract %b1dg, %v6852 : tensor<32xf32>
    %v6854 = stablehlo.reshape %v6798 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6855 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6856 = stablehlo.reduce(%v6854 init: %v6855) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6857 = stablehlo.constant dense<0.05> : tensor<32xf32>
    %v6858 = stablehlo.multiply %v6856, %v6857 : tensor<32xf32>
    %v6859 = stablehlo.subtract %b1dbt, %v6858 : tensor<32xf32>
    %v6860 = stablehlo.reshape %v28 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6861 = stablehlo.reshape %v6828 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6862 = stablehlo.transpose %v6860, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %v6863 = stablehlo.transpose %v6861, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %v6864 = stablehlo.convolution(%v6862, %v6863)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 32 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<32x32x112x112xf32>) -> tensor<1x32x3x3xf32>
    %v6865 = stablehlo.reshape %v6864 : (tensor<1x32x3x3xf32>) -> tensor<32x1x3x3xf32>
    %v6866 = stablehlo.constant dense<0.05> : tensor<32x1x3x3xf32>
    %v6867 = stablehlo.multiply %v6865, %v6866 : tensor<32x1x3x3xf32>
    %v6868 = stablehlo.subtract %b1dW, %v6867 : tensor<32x1x3x3xf32>
    %v6869 = stablehlo.reshape %v6832 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6870 = stablehlo.reshape %v24 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6871 = stablehlo.logistic %v6870 : tensor<32x32x112x112xf32>
    %v6872 = stablehlo.constant dense<1.0> : tensor<32x32x112x112xf32>
    %v6873 = stablehlo.subtract %v6872, %v6871 : tensor<32x32x112x112xf32>
    %v6874 = stablehlo.multiply %v6870, %v6873 : tensor<32x32x112x112xf32>
    %v6875 = stablehlo.add %v6872, %v6874 : tensor<32x32x112x112xf32>
    %v6876 = stablehlo.multiply %v6871, %v6875 : tensor<32x32x112x112xf32>
    %v6877 = stablehlo.multiply %v6869, %v6876 : tensor<32x32x112x112xf32>
    %v6878 = stablehlo.reshape %v6877 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v6879 = stablehlo.reshape %v4 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6880 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6881 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v6882 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v6883 = stablehlo.reduce(%v6879 init: %v6880) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6884 = stablehlo.broadcast_in_dim %v6883, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6885 = stablehlo.divide %v6884, %v6881 : tensor<32x32x112x112xf32>
    %v6886 = stablehlo.subtract %v6879, %v6885 : tensor<32x32x112x112xf32>
    %v6887 = stablehlo.multiply %v6886, %v6886 : tensor<32x32x112x112xf32>
    %v6888 = stablehlo.reduce(%v6887 init: %v6880) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6889 = stablehlo.broadcast_in_dim %v6888, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6890 = stablehlo.divide %v6889, %v6881 : tensor<32x32x112x112xf32>
    %v6891 = stablehlo.add %v6890, %v6882 : tensor<32x32x112x112xf32>
    %v6892 = stablehlo.rsqrt %v6891 : tensor<32x32x112x112xf32>
    %v6893 = stablehlo.multiply %v6886, %v6892 : tensor<32x32x112x112xf32>
    %v6894 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6895 = stablehlo.reshape %v6878 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6896 = stablehlo.multiply %v6894, %v6895 : tensor<32x32x112x112xf32>
    %v6897 = stablehlo.reduce(%v6896 init: %v6880) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6898 = stablehlo.broadcast_in_dim %v6897, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6899 = stablehlo.multiply %v6893, %v6896 : tensor<32x32x112x112xf32>
    %v6900 = stablehlo.reduce(%v6899 init: %v6880) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6901 = stablehlo.broadcast_in_dim %v6900, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6902 = stablehlo.multiply %v6896, %v6881 : tensor<32x32x112x112xf32>
    %v6903 = stablehlo.subtract %v6902, %v6898 : tensor<32x32x112x112xf32>
    %v6904 = stablehlo.multiply %v6893, %v6901 : tensor<32x32x112x112xf32>
    %v6905 = stablehlo.subtract %v6903, %v6904 : tensor<32x32x112x112xf32>
    %v6906 = stablehlo.divide %v6892, %v6881 : tensor<32x32x112x112xf32>
    %v6907 = stablehlo.multiply %v6906, %v6905 : tensor<32x32x112x112xf32>
    %v6908 = stablehlo.reshape %v6907 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v6909 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v6910 = stablehlo.reshape %v6908 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6911 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6912 = stablehlo.pad %v6910, %v6911, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32x224x224xf32>
    %v6913 = stablehlo.transpose %v6909, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %v6914 = stablehlo.transpose %v6912, dims = [1, 0, 2, 3] : (tensor<32x32x224x224xf32>) -> tensor<32x32x224x224xf32>
    %v6915 = stablehlo.convolution(%v6913, %v6914)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 2], [0, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<32x32x224x224xf32>) -> tensor<3x32x3x3xf32>
    %v6916 = stablehlo.transpose %v6915, dims = [1, 0, 2, 3] : (tensor<3x32x3x3xf32>) -> tensor<32x3x3x3xf32>
    %v6917 = stablehlo.constant dense<0.05> : tensor<32x3x3x3xf32>
    %v6918 = stablehlo.multiply %v6916, %v6917 : tensor<32x3x3x3xf32>
    %v6919 = stablehlo.subtract %sW, %v6918 : tensor<32x3x3x3xf32>
    %v6920 = stablehlo.reshape %v4 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6921 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6922 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v6923 = stablehlo.reduce(%v6920 init: %v6921) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6924 = stablehlo.broadcast_in_dim %v6923, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6925 = stablehlo.divide %v6924, %v6922 : tensor<32x32x112x112xf32>
    %v6926 = stablehlo.subtract %v6920, %v6925 : tensor<32x32x112x112xf32>
    %v6927 = stablehlo.multiply %v6926, %v6926 : tensor<32x32x112x112xf32>
    %v6928 = stablehlo.reduce(%v6927 init: %v6921) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6929 = stablehlo.broadcast_in_dim %v6928, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6930 = stablehlo.divide %v6929, %v6922 : tensor<32x32x112x112xf32>
    %v6931 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v6932 = stablehlo.add %v6930, %v6931 : tensor<32x32x112x112xf32>
    %v6933 = stablehlo.rsqrt %v6932 : tensor<32x32x112x112xf32>
    %v6934 = stablehlo.multiply %v6926, %v6933 : tensor<32x32x112x112xf32>
    %v6935 = stablehlo.reshape %v6878 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6936 = stablehlo.multiply %v6935, %v6934 : tensor<32x32x112x112xf32>
    %v6937 = stablehlo.reduce(%v6936 init: %v6921) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6938 = stablehlo.constant dense<0.05> : tensor<32xf32>
    %v6939 = stablehlo.multiply %v6937, %v6938 : tensor<32xf32>
    %v6940 = stablehlo.subtract %sg, %v6939 : tensor<32xf32>
    %v6941 = stablehlo.reshape %v6878 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6942 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6943 = stablehlo.reduce(%v6941 init: %v6942) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6944 = stablehlo.constant dense<0.05> : tensor<32xf32>
    %v6945 = stablehlo.multiply %v6943, %v6944 : tensor<32xf32>
    %v6946 = stablehlo.subtract %sbt, %v6945 : tensor<32xf32>
    return %v6919, %v6940, %v6946, %v6868, %v6853, %v6859, %v6783, %v6788, %v6764, %v6769, %v6713, %v6698, %v6704, %v6642, %v6627, %v6633, %v6561, %v6544, %v6550, %v6472, %v6477, %v6453, %v6458, %v6402, %v6387, %v6393, %v6327, %v6312, %v6318, %v6246, %v6231, %v6237, %v6161, %v6166, %v6142, %v6147, %v6091, %v6076, %v6082, %v6020, %v6005, %v6011, %v5939, %v5922, %v5928, %v5850, %v5855, %v5831, %v5836, %v5780, %v5765, %v5771, %v5705, %v5690, %v5696, %v5624, %v5609, %v5615, %v5539, %v5544, %v5520, %v5525, %v5469, %v5454, %v5460, %v5398, %v5383, %v5389, %v5317, %v5300, %v5306, %v5228, %v5233, %v5209, %v5214, %v5158, %v5143, %v5149, %v5083, %v5068, %v5074, %v5002, %v4987, %v4993, %v4917, %v4922, %v4898, %v4903, %v4847, %v4832, %v4838, %v4772, %v4757, %v4763, %v4691, %v4676, %v4682, %v4606, %v4611, %v4587, %v4592, %v4536, %v4521, %v4527, %v4465, %v4450, %v4456, %v4384, %v4369, %v4375, %v4299, %v4304, %v4280, %v4285, %v4229, %v4214, %v4220, %v4154, %v4139, %v4145, %v4073, %v4058, %v4064, %v3988, %v3993, %v3969, %v3974, %v3918, %v3903, %v3909, %v3843, %v3828, %v3834, %v3762, %v3747, %v3753, %v3677, %v3682, %v3658, %v3663, %v3607, %v3592, %v3598, %v3536, %v3521, %v3527, %v3455, %v3438, %v3444, %v3366, %v3371, %v3347, %v3352, %v3296, %v3281, %v3287, %v3221, %v3206, %v3212, %v3140, %v3125, %v3131, %v3055, %v3060, %v3036, %v3041, %v2985, %v2970, %v2976, %v2910, %v2895, %v2901, %v2829, %v2814, %v2820, %v2744, %v2749, %v2725, %v2730, %v2674, %v2659, %v2665, %v2599, %v2584, %v2590, %v2518, %v2503, %v2509, %v2433, %v2438, %v2414, %v2419, %v2363, %v2348, %v2354, %v2292, %v2277, %v2283, %v2211, %v2196, %v2202, %v2126, %v2131, %v2107, %v2112, %v2056, %v2041, %v2047, %v1985, %v1970, %v1976, %v1895, %v1900 : tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x8xf32>, tensor<8xf32>, tensor<8x32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x4xf32>, tensor<4xf32>, tensor<4x96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x6xf32>, tensor<6xf32>, tensor<6x144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x5x5xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x6xf32>, tensor<6xf32>, tensor<6x144xf32>, tensor<144xf32>, tensor<40x144x1x1xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240x40x1x1xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x1x5x5xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x10xf32>, tensor<10xf32>, tensor<10x240xf32>, tensor<240xf32>, tensor<40x240x1x1xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240x40x1x1xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x1x3x3xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x10xf32>, tensor<10xf32>, tensor<10x240xf32>, tensor<240xf32>, tensor<80x240x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x3x3xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<80x480x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x3x3xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<80x480x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x5x5xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<112x480x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<112x672x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<112x672x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<192x672x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x3x3xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<320x1152x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x10xf32>, tensor<10xf32>
  }
}
