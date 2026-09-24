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
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
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
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
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
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
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
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
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
    %v3205 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3206 = stablehlo.pad %v3204, %v3205, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672x14x14xf32>
    %v3207 = stablehlo.reverse %b12dW, dims = [2, 3] : tensor<672x1x5x5xf32>
    %v3208 = stablehlo.convolution(%v3206, %v3207)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %v3209 = stablehlo.reshape %v3208 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3210 = stablehlo.reshape %v1157 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3211 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3212 = stablehlo.constant dense<1568.0> : tensor<32x672x7x7xf32>
    %v3213 = stablehlo.reduce(%v3210 init: %v3211) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v3214 = stablehlo.broadcast_in_dim %v3213, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v3215 = stablehlo.divide %v3214, %v3212 : tensor<32x672x7x7xf32>
    %v3216 = stablehlo.subtract %v3210, %v3215 : tensor<32x672x7x7xf32>
    %v3217 = stablehlo.multiply %v3216, %v3216 : tensor<32x672x7x7xf32>
    %v3218 = stablehlo.reduce(%v3217 init: %v3211) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v3219 = stablehlo.broadcast_in_dim %v3218, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v3220 = stablehlo.divide %v3219, %v3212 : tensor<32x672x7x7xf32>
    %v3221 = stablehlo.constant dense<1.0e-5> : tensor<32x672x7x7xf32>
    %v3222 = stablehlo.add %v3220, %v3221 : tensor<32x672x7x7xf32>
    %v3223 = stablehlo.rsqrt %v3222 : tensor<32x672x7x7xf32>
    %v3224 = stablehlo.multiply %v3216, %v3223 : tensor<32x672x7x7xf32>
    %v3225 = stablehlo.reshape %v3173 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3226 = stablehlo.multiply %v3225, %v3224 : tensor<32x672x7x7xf32>
    %v3227 = stablehlo.reduce(%v3226 init: %v3211) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v3228 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3229 = stablehlo.multiply %v3227, %v3228 : tensor<672xf32>
    %v3230 = stablehlo.subtract %b12dg, %v3229 : tensor<672xf32>
    %v3231 = stablehlo.reshape %v3173 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3232 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3233 = stablehlo.reduce(%v3231 init: %v3232) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v3234 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3235 = stablehlo.multiply %v3233, %v3234 : tensor<672xf32>
    %v3236 = stablehlo.subtract %b12dbt, %v3235 : tensor<672xf32>
    %v3237 = stablehlo.reshape %v1152 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3238 = stablehlo.reshape %v3203 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v3239 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3240 = stablehlo.pad %v3238, %v3239, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672x14x14xf32>
    %v3241 = stablehlo.transpose %v3237, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3242 = stablehlo.transpose %v3240, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3243 = stablehlo.convolution(%v3241, %v3242)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 672 : i64, feature_group_count = 1 : i64} : (tensor<672x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<1x672x5x5xf32>
    %v3244 = stablehlo.reshape %v3243 : (tensor<1x672x5x5xf32>) -> tensor<672x1x5x5xf32>
    %v3245 = stablehlo.constant dense<0.05> : tensor<672x1x5x5xf32>
    %v3246 = stablehlo.multiply %v3244, %v3245 : tensor<672x1x5x5xf32>
    %v3247 = stablehlo.subtract %b12dW, %v3246 : tensor<672x1x5x5xf32>
    %v3248 = stablehlo.reshape %v3209 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3249 = stablehlo.reshape %v1148 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3250 = stablehlo.logistic %v3249 : tensor<32x672x14x14xf32>
    %v3251 = stablehlo.constant dense<1.0> : tensor<32x672x14x14xf32>
    %v3252 = stablehlo.subtract %v3251, %v3250 : tensor<32x672x14x14xf32>
    %v3253 = stablehlo.multiply %v3249, %v3252 : tensor<32x672x14x14xf32>
    %v3254 = stablehlo.add %v3251, %v3253 : tensor<32x672x14x14xf32>
    %v3255 = stablehlo.multiply %v3250, %v3254 : tensor<32x672x14x14xf32>
    %v3256 = stablehlo.multiply %v3248, %v3255 : tensor<32x672x14x14xf32>
    %v3257 = stablehlo.reshape %v3256 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3258 = stablehlo.reshape %v1128 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3259 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3260 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3261 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3262 = stablehlo.reduce(%v3258 init: %v3259) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3263 = stablehlo.broadcast_in_dim %v3262, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3264 = stablehlo.divide %v3263, %v3260 : tensor<32x672x14x14xf32>
    %v3265 = stablehlo.subtract %v3258, %v3264 : tensor<32x672x14x14xf32>
    %v3266 = stablehlo.multiply %v3265, %v3265 : tensor<32x672x14x14xf32>
    %v3267 = stablehlo.reduce(%v3266 init: %v3259) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3268 = stablehlo.broadcast_in_dim %v3267, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3269 = stablehlo.divide %v3268, %v3260 : tensor<32x672x14x14xf32>
    %v3270 = stablehlo.add %v3269, %v3261 : tensor<32x672x14x14xf32>
    %v3271 = stablehlo.rsqrt %v3270 : tensor<32x672x14x14xf32>
    %v3272 = stablehlo.multiply %v3265, %v3271 : tensor<32x672x14x14xf32>
    %v3273 = stablehlo.broadcast_in_dim %b12eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3274 = stablehlo.reshape %v3257 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3275 = stablehlo.multiply %v3273, %v3274 : tensor<32x672x14x14xf32>
    %v3276 = stablehlo.reduce(%v3275 init: %v3259) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3277 = stablehlo.broadcast_in_dim %v3276, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3278 = stablehlo.multiply %v3272, %v3275 : tensor<32x672x14x14xf32>
    %v3279 = stablehlo.reduce(%v3278 init: %v3259) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3280 = stablehlo.broadcast_in_dim %v3279, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3281 = stablehlo.multiply %v3275, %v3260 : tensor<32x672x14x14xf32>
    %v3282 = stablehlo.subtract %v3281, %v3277 : tensor<32x672x14x14xf32>
    %v3283 = stablehlo.multiply %v3272, %v3280 : tensor<32x672x14x14xf32>
    %v3284 = stablehlo.subtract %v3282, %v3283 : tensor<32x672x14x14xf32>
    %v3285 = stablehlo.divide %v3271, %v3260 : tensor<32x672x14x14xf32>
    %v3286 = stablehlo.multiply %v3285, %v3284 : tensor<32x672x14x14xf32>
    %v3287 = stablehlo.reshape %v3286 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3288 = stablehlo.reshape %v3287 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3289 = stablehlo.reverse %b12eW, dims = [2, 3] : tensor<672x112x1x1xf32>
    %v3290 = stablehlo.transpose %v3289, dims = [1, 0, 2, 3] : (tensor<672x112x1x1xf32>) -> tensor<112x672x1x1xf32>
    %v3291 = stablehlo.convolution(%v3288, %v3290)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v3292 = stablehlo.reshape %v3291 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v3293 = stablehlo.reshape %v1128 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3294 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3295 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3296 = stablehlo.reduce(%v3293 init: %v3294) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3297 = stablehlo.broadcast_in_dim %v3296, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3298 = stablehlo.divide %v3297, %v3295 : tensor<32x672x14x14xf32>
    %v3299 = stablehlo.subtract %v3293, %v3298 : tensor<32x672x14x14xf32>
    %v3300 = stablehlo.multiply %v3299, %v3299 : tensor<32x672x14x14xf32>
    %v3301 = stablehlo.reduce(%v3300 init: %v3294) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3302 = stablehlo.broadcast_in_dim %v3301, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3303 = stablehlo.divide %v3302, %v3295 : tensor<32x672x14x14xf32>
    %v3304 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3305 = stablehlo.add %v3303, %v3304 : tensor<32x672x14x14xf32>
    %v3306 = stablehlo.rsqrt %v3305 : tensor<32x672x14x14xf32>
    %v3307 = stablehlo.multiply %v3299, %v3306 : tensor<32x672x14x14xf32>
    %v3308 = stablehlo.reshape %v3257 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3309 = stablehlo.multiply %v3308, %v3307 : tensor<32x672x14x14xf32>
    %v3310 = stablehlo.reduce(%v3309 init: %v3294) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3311 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3312 = stablehlo.multiply %v3310, %v3311 : tensor<672xf32>
    %v3313 = stablehlo.subtract %b12eg, %v3312 : tensor<672xf32>
    %v3314 = stablehlo.reshape %v3257 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3315 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3316 = stablehlo.reduce(%v3314 init: %v3315) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3317 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3318 = stablehlo.multiply %v3316, %v3317 : tensor<672xf32>
    %v3319 = stablehlo.subtract %b12ebt, %v3318 : tensor<672xf32>
    %v3320 = stablehlo.reshape %v1123 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3321 = stablehlo.reshape %v3287 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3322 = stablehlo.transpose %v3320, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %v3323 = stablehlo.transpose %v3321, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3324 = stablehlo.convolution(%v3322, %v3323)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<112x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<112x672x1x1xf32>
    %v3325 = stablehlo.transpose %v3324, dims = [1, 0, 2, 3] : (tensor<112x672x1x1xf32>) -> tensor<672x112x1x1xf32>
    %v3326 = stablehlo.constant dense<0.05> : tensor<672x112x1x1xf32>
    %v3327 = stablehlo.multiply %v3325, %v3326 : tensor<672x112x1x1xf32>
    %v3328 = stablehlo.subtract %b12eW, %v3327 : tensor<672x112x1x1xf32>
    %v3329 = stablehlo.reshape %v1099 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3330 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3331 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v3332 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v3333 = stablehlo.reduce(%v3329 init: %v3330) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3334 = stablehlo.broadcast_in_dim %v3333, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3335 = stablehlo.divide %v3334, %v3331 : tensor<32x112x14x14xf32>
    %v3336 = stablehlo.subtract %v3329, %v3335 : tensor<32x112x14x14xf32>
    %v3337 = stablehlo.multiply %v3336, %v3336 : tensor<32x112x14x14xf32>
    %v3338 = stablehlo.reduce(%v3337 init: %v3330) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3339 = stablehlo.broadcast_in_dim %v3338, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3340 = stablehlo.divide %v3339, %v3331 : tensor<32x112x14x14xf32>
    %v3341 = stablehlo.add %v3340, %v3332 : tensor<32x112x14x14xf32>
    %v3342 = stablehlo.rsqrt %v3341 : tensor<32x112x14x14xf32>
    %v3343 = stablehlo.multiply %v3336, %v3342 : tensor<32x112x14x14xf32>
    %v3344 = stablehlo.broadcast_in_dim %b11pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3345 = stablehlo.reshape %v3292 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3346 = stablehlo.multiply %v3344, %v3345 : tensor<32x112x14x14xf32>
    %v3347 = stablehlo.reduce(%v3346 init: %v3330) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3348 = stablehlo.broadcast_in_dim %v3347, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3349 = stablehlo.multiply %v3343, %v3346 : tensor<32x112x14x14xf32>
    %v3350 = stablehlo.reduce(%v3349 init: %v3330) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3351 = stablehlo.broadcast_in_dim %v3350, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3352 = stablehlo.multiply %v3346, %v3331 : tensor<32x112x14x14xf32>
    %v3353 = stablehlo.subtract %v3352, %v3348 : tensor<32x112x14x14xf32>
    %v3354 = stablehlo.multiply %v3343, %v3351 : tensor<32x112x14x14xf32>
    %v3355 = stablehlo.subtract %v3353, %v3354 : tensor<32x112x14x14xf32>
    %v3356 = stablehlo.divide %v3342, %v3331 : tensor<32x112x14x14xf32>
    %v3357 = stablehlo.multiply %v3356, %v3355 : tensor<32x112x14x14xf32>
    %v3358 = stablehlo.reshape %v3357 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v3359 = stablehlo.reshape %v3358 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3360 = stablehlo.reverse %b11pW, dims = [2, 3] : tensor<112x672x1x1xf32>
    %v3361 = stablehlo.transpose %v3360, dims = [1, 0, 2, 3] : (tensor<112x672x1x1xf32>) -> tensor<672x112x1x1xf32>
    %v3362 = stablehlo.convolution(%v3359, %v3361)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v3363 = stablehlo.reshape %v3362 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3364 = stablehlo.reshape %v1099 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3365 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3366 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v3367 = stablehlo.reduce(%v3364 init: %v3365) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3368 = stablehlo.broadcast_in_dim %v3367, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3369 = stablehlo.divide %v3368, %v3366 : tensor<32x112x14x14xf32>
    %v3370 = stablehlo.subtract %v3364, %v3369 : tensor<32x112x14x14xf32>
    %v3371 = stablehlo.multiply %v3370, %v3370 : tensor<32x112x14x14xf32>
    %v3372 = stablehlo.reduce(%v3371 init: %v3365) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3373 = stablehlo.broadcast_in_dim %v3372, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3374 = stablehlo.divide %v3373, %v3366 : tensor<32x112x14x14xf32>
    %v3375 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v3376 = stablehlo.add %v3374, %v3375 : tensor<32x112x14x14xf32>
    %v3377 = stablehlo.rsqrt %v3376 : tensor<32x112x14x14xf32>
    %v3378 = stablehlo.multiply %v3370, %v3377 : tensor<32x112x14x14xf32>
    %v3379 = stablehlo.reshape %v3292 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3380 = stablehlo.multiply %v3379, %v3378 : tensor<32x112x14x14xf32>
    %v3381 = stablehlo.reduce(%v3380 init: %v3365) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3382 = stablehlo.constant dense<0.05> : tensor<112xf32>
    %v3383 = stablehlo.multiply %v3381, %v3382 : tensor<112xf32>
    %v3384 = stablehlo.subtract %b11pg, %v3383 : tensor<112xf32>
    %v3385 = stablehlo.reshape %v3292 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3386 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3387 = stablehlo.reduce(%v3385 init: %v3386) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3388 = stablehlo.constant dense<0.05> : tensor<112xf32>
    %v3389 = stablehlo.multiply %v3387, %v3388 : tensor<112xf32>
    %v3390 = stablehlo.subtract %b11pbt, %v3389 : tensor<112xf32>
    %v3391 = stablehlo.reshape %v1094 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3392 = stablehlo.reshape %v3358 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3393 = stablehlo.transpose %v3391, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3394 = stablehlo.transpose %v3392, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %v3395 = stablehlo.convolution(%v3393, %v3394)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<672x32x14x14xf32>, tensor<112x32x14x14xf32>) -> tensor<672x112x1x1xf32>
    %v3396 = stablehlo.transpose %v3395, dims = [1, 0, 2, 3] : (tensor<672x112x1x1xf32>) -> tensor<112x672x1x1xf32>
    %v3397 = stablehlo.constant dense<0.05> : tensor<112x672x1x1xf32>
    %v3398 = stablehlo.multiply %v3396, %v3397 : tensor<112x672x1x1xf32>
    %v3399 = stablehlo.subtract %b11pW, %v3398 : tensor<112x672x1x1xf32>
    %v3400 = stablehlo.reshape %v1077 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3401 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3402 = stablehlo.reduce(%v3400 init: %v3401) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3403 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v3404 = stablehlo.divide %v3402, %v3403 : tensor<32x672xf32>
    %v3405 = stablehlo.dot_general %v3404, %b11zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v3406 = stablehlo.broadcast_in_dim %b11zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v3407 = stablehlo.add %v3405, %v3406 : tensor<32x28xf32>
    %v3408 = stablehlo.logistic %v3407 : tensor<32x28xf32>
    %v3409 = stablehlo.multiply %v3407, %v3408 : tensor<32x28xf32>
    %v3410 = stablehlo.dot_general %v3409, %b11zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v3411 = stablehlo.broadcast_in_dim %b11zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v3412 = stablehlo.add %v3410, %v3411 : tensor<32x672xf32>
    %v3413 = stablehlo.logistic %v3412 : tensor<32x672xf32>
    %v3414 = stablehlo.reshape %v3363 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3415 = stablehlo.broadcast_in_dim %v3413, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v3416 = stablehlo.multiply %v3415, %v3414 : tensor<32x672x14x14xf32>
    %v3417 = stablehlo.multiply %v3400, %v3414 : tensor<32x672x14x14xf32>
    %v3418 = stablehlo.reduce(%v3417 init: %v3401) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3419 = stablehlo.constant dense<1.0> : tensor<32x672xf32>
    %v3420 = stablehlo.subtract %v3419, %v3413 : tensor<32x672xf32>
    %v3421 = stablehlo.multiply %v3413, %v3420 : tensor<32x672xf32>
    %v3422 = stablehlo.multiply %v3418, %v3421 : tensor<32x672xf32>
    %v3423 = stablehlo.dot_general %v3422, %b11zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<28x672xf32>) -> tensor<32x28xf32>
    %v3424 = stablehlo.logistic %v3407 : tensor<32x28xf32>
    %v3425 = stablehlo.constant dense<1.0> : tensor<32x28xf32>
    %v3426 = stablehlo.subtract %v3425, %v3424 : tensor<32x28xf32>
    %v3427 = stablehlo.multiply %v3407, %v3426 : tensor<32x28xf32>
    %v3428 = stablehlo.add %v3425, %v3427 : tensor<32x28xf32>
    %v3429 = stablehlo.multiply %v3424, %v3428 : tensor<32x28xf32>
    %v3430 = stablehlo.multiply %v3423, %v3429 : tensor<32x28xf32>
    %v3431 = stablehlo.dot_general %v3430, %b11zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<672x28xf32>) -> tensor<32x672xf32>
    %v3432 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v3433 = stablehlo.divide %v3431, %v3432 : tensor<32x672xf32>
    %v3434 = stablehlo.broadcast_in_dim %v3433, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v3435 = stablehlo.add %v3416, %v3434 : tensor<32x672x14x14xf32>
    %v3436 = stablehlo.reshape %v3435 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3437 = stablehlo.reshape %v1077 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3438 = stablehlo.reshape %v3363 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3439 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3440 = stablehlo.multiply %v3437, %v3438 : tensor<32x672x14x14xf32>
    %v3441 = stablehlo.reduce(%v3440 init: %v3439) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3442 = stablehlo.logistic %v1090 : tensor<32x672xf32>
    %v3443 = stablehlo.constant dense<1.0> : tensor<32x672xf32>
    %v3444 = stablehlo.subtract %v3443, %v3442 : tensor<32x672xf32>
    %v3445 = stablehlo.multiply %v3442, %v3444 : tensor<32x672xf32>
    %v3446 = stablehlo.multiply %v3441, %v3445 : tensor<32x672xf32>
    %v3447 = stablehlo.dot_general %v1087, %v3446, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<32x672xf32>) -> tensor<28x672xf32>
    %v3448 = stablehlo.constant dense<0.05> : tensor<28x672xf32>
    %v3449 = stablehlo.multiply %v3447, %v3448 : tensor<28x672xf32>
    %v3450 = stablehlo.subtract %b11zW2, %v3449 : tensor<28x672xf32>
    %v3451 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3452 = stablehlo.reduce(%v3446 init: %v3451) applies stablehlo.add across dimensions = [0] : (tensor<32x672xf32>, tensor<f32>) -> tensor<672xf32>
    %v3453 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3454 = stablehlo.multiply %v3452, %v3453 : tensor<672xf32>
    %v3455 = stablehlo.subtract %b11zb2, %v3454 : tensor<672xf32>
    %v3456 = stablehlo.reshape %v3446 : (tensor<32x672xf32>) -> tensor<32x1x672xf32>
    %v3457 = stablehlo.dot_general %v3456, %b11zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x672xf32>, tensor<28x672xf32>) -> tensor<32x1x28xf32>
    %v3458 = stablehlo.reshape %v3457 : (tensor<32x1x28xf32>) -> tensor<32x28xf32>
    %v3459 = stablehlo.logistic %v1085 : tensor<32x28xf32>
    %v3460 = stablehlo.constant dense<1.0> : tensor<32x28xf32>
    %v3461 = stablehlo.subtract %v3460, %v3459 : tensor<32x28xf32>
    %v3462 = stablehlo.multiply %v1085, %v3461 : tensor<32x28xf32>
    %v3463 = stablehlo.add %v3460, %v3462 : tensor<32x28xf32>
    %v3464 = stablehlo.multiply %v3459, %v3463 : tensor<32x28xf32>
    %v3465 = stablehlo.multiply %v3458, %v3464 : tensor<32x28xf32>
    %v3466 = stablehlo.dot_general %v1082, %v3465, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<32x28xf32>) -> tensor<672x28xf32>
    %v3467 = stablehlo.constant dense<0.05> : tensor<672x28xf32>
    %v3468 = stablehlo.multiply %v3466, %v3467 : tensor<672x28xf32>
    %v3469 = stablehlo.subtract %b11zW1, %v3468 : tensor<672x28xf32>
    %v3470 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3471 = stablehlo.reduce(%v3465 init: %v3470) applies stablehlo.add across dimensions = [0] : (tensor<32x28xf32>, tensor<f32>) -> tensor<28xf32>
    %v3472 = stablehlo.constant dense<0.05> : tensor<28xf32>
    %v3473 = stablehlo.multiply %v3471, %v3472 : tensor<28xf32>
    %v3474 = stablehlo.subtract %b11zb1, %v3473 : tensor<28xf32>
    %v3475 = stablehlo.reshape %v3436 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3476 = stablehlo.reshape %v1073 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3477 = stablehlo.logistic %v3476 : tensor<32x672x14x14xf32>
    %v3478 = stablehlo.constant dense<1.0> : tensor<32x672x14x14xf32>
    %v3479 = stablehlo.subtract %v3478, %v3477 : tensor<32x672x14x14xf32>
    %v3480 = stablehlo.multiply %v3476, %v3479 : tensor<32x672x14x14xf32>
    %v3481 = stablehlo.add %v3478, %v3480 : tensor<32x672x14x14xf32>
    %v3482 = stablehlo.multiply %v3477, %v3481 : tensor<32x672x14x14xf32>
    %v3483 = stablehlo.multiply %v3475, %v3482 : tensor<32x672x14x14xf32>
    %v3484 = stablehlo.reshape %v3483 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3485 = stablehlo.reshape %v1053 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3486 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3487 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3488 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3489 = stablehlo.reduce(%v3485 init: %v3486) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3490 = stablehlo.broadcast_in_dim %v3489, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3491 = stablehlo.divide %v3490, %v3487 : tensor<32x672x14x14xf32>
    %v3492 = stablehlo.subtract %v3485, %v3491 : tensor<32x672x14x14xf32>
    %v3493 = stablehlo.multiply %v3492, %v3492 : tensor<32x672x14x14xf32>
    %v3494 = stablehlo.reduce(%v3493 init: %v3486) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3495 = stablehlo.broadcast_in_dim %v3494, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3496 = stablehlo.divide %v3495, %v3487 : tensor<32x672x14x14xf32>
    %v3497 = stablehlo.add %v3496, %v3488 : tensor<32x672x14x14xf32>
    %v3498 = stablehlo.rsqrt %v3497 : tensor<32x672x14x14xf32>
    %v3499 = stablehlo.multiply %v3492, %v3498 : tensor<32x672x14x14xf32>
    %v3500 = stablehlo.broadcast_in_dim %b11dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3501 = stablehlo.reshape %v3484 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3502 = stablehlo.multiply %v3500, %v3501 : tensor<32x672x14x14xf32>
    %v3503 = stablehlo.reduce(%v3502 init: %v3486) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3504 = stablehlo.broadcast_in_dim %v3503, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3505 = stablehlo.multiply %v3499, %v3502 : tensor<32x672x14x14xf32>
    %v3506 = stablehlo.reduce(%v3505 init: %v3486) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3507 = stablehlo.broadcast_in_dim %v3506, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3508 = stablehlo.multiply %v3502, %v3487 : tensor<32x672x14x14xf32>
    %v3509 = stablehlo.subtract %v3508, %v3504 : tensor<32x672x14x14xf32>
    %v3510 = stablehlo.multiply %v3499, %v3507 : tensor<32x672x14x14xf32>
    %v3511 = stablehlo.subtract %v3509, %v3510 : tensor<32x672x14x14xf32>
    %v3512 = stablehlo.divide %v3498, %v3487 : tensor<32x672x14x14xf32>
    %v3513 = stablehlo.multiply %v3512, %v3511 : tensor<32x672x14x14xf32>
    %v3514 = stablehlo.reshape %v3513 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3515 = stablehlo.reshape %v3514 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3516 = stablehlo.reverse %b11dW, dims = [2, 3] : tensor<672x1x5x5xf32>
    %v3517 = stablehlo.convolution(%v3515, %v3516)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %v3518 = stablehlo.reshape %v3517 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3519 = stablehlo.reshape %v1053 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3520 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3521 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3522 = stablehlo.reduce(%v3519 init: %v3520) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3523 = stablehlo.broadcast_in_dim %v3522, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3524 = stablehlo.divide %v3523, %v3521 : tensor<32x672x14x14xf32>
    %v3525 = stablehlo.subtract %v3519, %v3524 : tensor<32x672x14x14xf32>
    %v3526 = stablehlo.multiply %v3525, %v3525 : tensor<32x672x14x14xf32>
    %v3527 = stablehlo.reduce(%v3526 init: %v3520) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3528 = stablehlo.broadcast_in_dim %v3527, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3529 = stablehlo.divide %v3528, %v3521 : tensor<32x672x14x14xf32>
    %v3530 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3531 = stablehlo.add %v3529, %v3530 : tensor<32x672x14x14xf32>
    %v3532 = stablehlo.rsqrt %v3531 : tensor<32x672x14x14xf32>
    %v3533 = stablehlo.multiply %v3525, %v3532 : tensor<32x672x14x14xf32>
    %v3534 = stablehlo.reshape %v3484 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3535 = stablehlo.multiply %v3534, %v3533 : tensor<32x672x14x14xf32>
    %v3536 = stablehlo.reduce(%v3535 init: %v3520) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3537 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3538 = stablehlo.multiply %v3536, %v3537 : tensor<672xf32>
    %v3539 = stablehlo.subtract %b11dg, %v3538 : tensor<672xf32>
    %v3540 = stablehlo.reshape %v3484 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3541 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3542 = stablehlo.reduce(%v3540 init: %v3541) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3543 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3544 = stablehlo.multiply %v3542, %v3543 : tensor<672xf32>
    %v3545 = stablehlo.subtract %b11dbt, %v3544 : tensor<672xf32>
    %v3546 = stablehlo.reshape %v1048 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3547 = stablehlo.reshape %v3514 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3548 = stablehlo.transpose %v3546, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3549 = stablehlo.transpose %v3547, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3550 = stablehlo.convolution(%v3548, %v3549)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 672 : i64, feature_group_count = 1 : i64} : (tensor<672x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<1x672x5x5xf32>
    %v3551 = stablehlo.reshape %v3550 : (tensor<1x672x5x5xf32>) -> tensor<672x1x5x5xf32>
    %v3552 = stablehlo.constant dense<0.05> : tensor<672x1x5x5xf32>
    %v3553 = stablehlo.multiply %v3551, %v3552 : tensor<672x1x5x5xf32>
    %v3554 = stablehlo.subtract %b11dW, %v3553 : tensor<672x1x5x5xf32>
    %v3555 = stablehlo.reshape %v3518 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3556 = stablehlo.reshape %v1044 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3557 = stablehlo.logistic %v3556 : tensor<32x672x14x14xf32>
    %v3558 = stablehlo.constant dense<1.0> : tensor<32x672x14x14xf32>
    %v3559 = stablehlo.subtract %v3558, %v3557 : tensor<32x672x14x14xf32>
    %v3560 = stablehlo.multiply %v3556, %v3559 : tensor<32x672x14x14xf32>
    %v3561 = stablehlo.add %v3558, %v3560 : tensor<32x672x14x14xf32>
    %v3562 = stablehlo.multiply %v3557, %v3561 : tensor<32x672x14x14xf32>
    %v3563 = stablehlo.multiply %v3555, %v3562 : tensor<32x672x14x14xf32>
    %v3564 = stablehlo.reshape %v3563 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3565 = stablehlo.reshape %v1024 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3566 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3567 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3568 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3569 = stablehlo.reduce(%v3565 init: %v3566) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3570 = stablehlo.broadcast_in_dim %v3569, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3571 = stablehlo.divide %v3570, %v3567 : tensor<32x672x14x14xf32>
    %v3572 = stablehlo.subtract %v3565, %v3571 : tensor<32x672x14x14xf32>
    %v3573 = stablehlo.multiply %v3572, %v3572 : tensor<32x672x14x14xf32>
    %v3574 = stablehlo.reduce(%v3573 init: %v3566) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3575 = stablehlo.broadcast_in_dim %v3574, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3576 = stablehlo.divide %v3575, %v3567 : tensor<32x672x14x14xf32>
    %v3577 = stablehlo.add %v3576, %v3568 : tensor<32x672x14x14xf32>
    %v3578 = stablehlo.rsqrt %v3577 : tensor<32x672x14x14xf32>
    %v3579 = stablehlo.multiply %v3572, %v3578 : tensor<32x672x14x14xf32>
    %v3580 = stablehlo.broadcast_in_dim %b11eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3581 = stablehlo.reshape %v3564 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3582 = stablehlo.multiply %v3580, %v3581 : tensor<32x672x14x14xf32>
    %v3583 = stablehlo.reduce(%v3582 init: %v3566) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3584 = stablehlo.broadcast_in_dim %v3583, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3585 = stablehlo.multiply %v3579, %v3582 : tensor<32x672x14x14xf32>
    %v3586 = stablehlo.reduce(%v3585 init: %v3566) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3587 = stablehlo.broadcast_in_dim %v3586, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3588 = stablehlo.multiply %v3582, %v3567 : tensor<32x672x14x14xf32>
    %v3589 = stablehlo.subtract %v3588, %v3584 : tensor<32x672x14x14xf32>
    %v3590 = stablehlo.multiply %v3579, %v3587 : tensor<32x672x14x14xf32>
    %v3591 = stablehlo.subtract %v3589, %v3590 : tensor<32x672x14x14xf32>
    %v3592 = stablehlo.divide %v3578, %v3567 : tensor<32x672x14x14xf32>
    %v3593 = stablehlo.multiply %v3592, %v3591 : tensor<32x672x14x14xf32>
    %v3594 = stablehlo.reshape %v3593 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3595 = stablehlo.reshape %v3594 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3596 = stablehlo.reverse %b11eW, dims = [2, 3] : tensor<672x112x1x1xf32>
    %v3597 = stablehlo.transpose %v3596, dims = [1, 0, 2, 3] : (tensor<672x112x1x1xf32>) -> tensor<112x672x1x1xf32>
    %v3598 = stablehlo.convolution(%v3595, %v3597)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v3599 = stablehlo.reshape %v3598 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v3600 = stablehlo.reshape %v1024 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3601 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3602 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3603 = stablehlo.reduce(%v3600 init: %v3601) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3604 = stablehlo.broadcast_in_dim %v3603, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3605 = stablehlo.divide %v3604, %v3602 : tensor<32x672x14x14xf32>
    %v3606 = stablehlo.subtract %v3600, %v3605 : tensor<32x672x14x14xf32>
    %v3607 = stablehlo.multiply %v3606, %v3606 : tensor<32x672x14x14xf32>
    %v3608 = stablehlo.reduce(%v3607 init: %v3601) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3609 = stablehlo.broadcast_in_dim %v3608, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3610 = stablehlo.divide %v3609, %v3602 : tensor<32x672x14x14xf32>
    %v3611 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3612 = stablehlo.add %v3610, %v3611 : tensor<32x672x14x14xf32>
    %v3613 = stablehlo.rsqrt %v3612 : tensor<32x672x14x14xf32>
    %v3614 = stablehlo.multiply %v3606, %v3613 : tensor<32x672x14x14xf32>
    %v3615 = stablehlo.reshape %v3564 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3616 = stablehlo.multiply %v3615, %v3614 : tensor<32x672x14x14xf32>
    %v3617 = stablehlo.reduce(%v3616 init: %v3601) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3618 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3619 = stablehlo.multiply %v3617, %v3618 : tensor<672xf32>
    %v3620 = stablehlo.subtract %b11eg, %v3619 : tensor<672xf32>
    %v3621 = stablehlo.reshape %v3564 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3622 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3623 = stablehlo.reduce(%v3621 init: %v3622) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3624 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3625 = stablehlo.multiply %v3623, %v3624 : tensor<672xf32>
    %v3626 = stablehlo.subtract %b11ebt, %v3625 : tensor<672xf32>
    %v3627 = stablehlo.reshape %v1019 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3628 = stablehlo.reshape %v3594 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3629 = stablehlo.transpose %v3627, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %v3630 = stablehlo.transpose %v3628, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3631 = stablehlo.convolution(%v3629, %v3630)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<112x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<112x672x1x1xf32>
    %v3632 = stablehlo.transpose %v3631, dims = [1, 0, 2, 3] : (tensor<112x672x1x1xf32>) -> tensor<672x112x1x1xf32>
    %v3633 = stablehlo.constant dense<0.05> : tensor<672x112x1x1xf32>
    %v3634 = stablehlo.multiply %v3632, %v3633 : tensor<672x112x1x1xf32>
    %v3635 = stablehlo.subtract %b11eW, %v3634 : tensor<672x112x1x1xf32>
    %v3636 = stablehlo.reshape %v3599 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3637 = stablehlo.reshape %v3292 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3638 = stablehlo.add %v3636, %v3637 : tensor<32x112x14x14xf32>
    %v3639 = stablehlo.reshape %v3638 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v3640 = stablehlo.reshape %v995 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3641 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3642 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v3643 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v3644 = stablehlo.reduce(%v3640 init: %v3641) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3645 = stablehlo.broadcast_in_dim %v3644, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3646 = stablehlo.divide %v3645, %v3642 : tensor<32x112x14x14xf32>
    %v3647 = stablehlo.subtract %v3640, %v3646 : tensor<32x112x14x14xf32>
    %v3648 = stablehlo.multiply %v3647, %v3647 : tensor<32x112x14x14xf32>
    %v3649 = stablehlo.reduce(%v3648 init: %v3641) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3650 = stablehlo.broadcast_in_dim %v3649, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3651 = stablehlo.divide %v3650, %v3642 : tensor<32x112x14x14xf32>
    %v3652 = stablehlo.add %v3651, %v3643 : tensor<32x112x14x14xf32>
    %v3653 = stablehlo.rsqrt %v3652 : tensor<32x112x14x14xf32>
    %v3654 = stablehlo.multiply %v3647, %v3653 : tensor<32x112x14x14xf32>
    %v3655 = stablehlo.broadcast_in_dim %b10pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3656 = stablehlo.reshape %v3639 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3657 = stablehlo.multiply %v3655, %v3656 : tensor<32x112x14x14xf32>
    %v3658 = stablehlo.reduce(%v3657 init: %v3641) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3659 = stablehlo.broadcast_in_dim %v3658, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3660 = stablehlo.multiply %v3654, %v3657 : tensor<32x112x14x14xf32>
    %v3661 = stablehlo.reduce(%v3660 init: %v3641) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3662 = stablehlo.broadcast_in_dim %v3661, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3663 = stablehlo.multiply %v3657, %v3642 : tensor<32x112x14x14xf32>
    %v3664 = stablehlo.subtract %v3663, %v3659 : tensor<32x112x14x14xf32>
    %v3665 = stablehlo.multiply %v3654, %v3662 : tensor<32x112x14x14xf32>
    %v3666 = stablehlo.subtract %v3664, %v3665 : tensor<32x112x14x14xf32>
    %v3667 = stablehlo.divide %v3653, %v3642 : tensor<32x112x14x14xf32>
    %v3668 = stablehlo.multiply %v3667, %v3666 : tensor<32x112x14x14xf32>
    %v3669 = stablehlo.reshape %v3668 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v3670 = stablehlo.reshape %v3669 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3671 = stablehlo.reverse %b10pW, dims = [2, 3] : tensor<112x672x1x1xf32>
    %v3672 = stablehlo.transpose %v3671, dims = [1, 0, 2, 3] : (tensor<112x672x1x1xf32>) -> tensor<672x112x1x1xf32>
    %v3673 = stablehlo.convolution(%v3670, %v3672)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v3674 = stablehlo.reshape %v3673 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3675 = stablehlo.reshape %v995 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3676 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3677 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v3678 = stablehlo.reduce(%v3675 init: %v3676) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3679 = stablehlo.broadcast_in_dim %v3678, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3680 = stablehlo.divide %v3679, %v3677 : tensor<32x112x14x14xf32>
    %v3681 = stablehlo.subtract %v3675, %v3680 : tensor<32x112x14x14xf32>
    %v3682 = stablehlo.multiply %v3681, %v3681 : tensor<32x112x14x14xf32>
    %v3683 = stablehlo.reduce(%v3682 init: %v3676) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3684 = stablehlo.broadcast_in_dim %v3683, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3685 = stablehlo.divide %v3684, %v3677 : tensor<32x112x14x14xf32>
    %v3686 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v3687 = stablehlo.add %v3685, %v3686 : tensor<32x112x14x14xf32>
    %v3688 = stablehlo.rsqrt %v3687 : tensor<32x112x14x14xf32>
    %v3689 = stablehlo.multiply %v3681, %v3688 : tensor<32x112x14x14xf32>
    %v3690 = stablehlo.reshape %v3639 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3691 = stablehlo.multiply %v3690, %v3689 : tensor<32x112x14x14xf32>
    %v3692 = stablehlo.reduce(%v3691 init: %v3676) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3693 = stablehlo.constant dense<0.05> : tensor<112xf32>
    %v3694 = stablehlo.multiply %v3692, %v3693 : tensor<112xf32>
    %v3695 = stablehlo.subtract %b10pg, %v3694 : tensor<112xf32>
    %v3696 = stablehlo.reshape %v3639 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3697 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3698 = stablehlo.reduce(%v3696 init: %v3697) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3699 = stablehlo.constant dense<0.05> : tensor<112xf32>
    %v3700 = stablehlo.multiply %v3698, %v3699 : tensor<112xf32>
    %v3701 = stablehlo.subtract %b10pbt, %v3700 : tensor<112xf32>
    %v3702 = stablehlo.reshape %v990 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3703 = stablehlo.reshape %v3669 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3704 = stablehlo.transpose %v3702, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3705 = stablehlo.transpose %v3703, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %v3706 = stablehlo.convolution(%v3704, %v3705)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<672x32x14x14xf32>, tensor<112x32x14x14xf32>) -> tensor<672x112x1x1xf32>
    %v3707 = stablehlo.transpose %v3706, dims = [1, 0, 2, 3] : (tensor<672x112x1x1xf32>) -> tensor<112x672x1x1xf32>
    %v3708 = stablehlo.constant dense<0.05> : tensor<112x672x1x1xf32>
    %v3709 = stablehlo.multiply %v3707, %v3708 : tensor<112x672x1x1xf32>
    %v3710 = stablehlo.subtract %b10pW, %v3709 : tensor<112x672x1x1xf32>
    %v3711 = stablehlo.reshape %v973 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3712 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3713 = stablehlo.reduce(%v3711 init: %v3712) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3714 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v3715 = stablehlo.divide %v3713, %v3714 : tensor<32x672xf32>
    %v3716 = stablehlo.dot_general %v3715, %b10zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v3717 = stablehlo.broadcast_in_dim %b10zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v3718 = stablehlo.add %v3716, %v3717 : tensor<32x28xf32>
    %v3719 = stablehlo.logistic %v3718 : tensor<32x28xf32>
    %v3720 = stablehlo.multiply %v3718, %v3719 : tensor<32x28xf32>
    %v3721 = stablehlo.dot_general %v3720, %b10zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v3722 = stablehlo.broadcast_in_dim %b10zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v3723 = stablehlo.add %v3721, %v3722 : tensor<32x672xf32>
    %v3724 = stablehlo.logistic %v3723 : tensor<32x672xf32>
    %v3725 = stablehlo.reshape %v3674 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3726 = stablehlo.broadcast_in_dim %v3724, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v3727 = stablehlo.multiply %v3726, %v3725 : tensor<32x672x14x14xf32>
    %v3728 = stablehlo.multiply %v3711, %v3725 : tensor<32x672x14x14xf32>
    %v3729 = stablehlo.reduce(%v3728 init: %v3712) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3730 = stablehlo.constant dense<1.0> : tensor<32x672xf32>
    %v3731 = stablehlo.subtract %v3730, %v3724 : tensor<32x672xf32>
    %v3732 = stablehlo.multiply %v3724, %v3731 : tensor<32x672xf32>
    %v3733 = stablehlo.multiply %v3729, %v3732 : tensor<32x672xf32>
    %v3734 = stablehlo.dot_general %v3733, %b10zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<28x672xf32>) -> tensor<32x28xf32>
    %v3735 = stablehlo.logistic %v3718 : tensor<32x28xf32>
    %v3736 = stablehlo.constant dense<1.0> : tensor<32x28xf32>
    %v3737 = stablehlo.subtract %v3736, %v3735 : tensor<32x28xf32>
    %v3738 = stablehlo.multiply %v3718, %v3737 : tensor<32x28xf32>
    %v3739 = stablehlo.add %v3736, %v3738 : tensor<32x28xf32>
    %v3740 = stablehlo.multiply %v3735, %v3739 : tensor<32x28xf32>
    %v3741 = stablehlo.multiply %v3734, %v3740 : tensor<32x28xf32>
    %v3742 = stablehlo.dot_general %v3741, %b10zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<672x28xf32>) -> tensor<32x672xf32>
    %v3743 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v3744 = stablehlo.divide %v3742, %v3743 : tensor<32x672xf32>
    %v3745 = stablehlo.broadcast_in_dim %v3744, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v3746 = stablehlo.add %v3727, %v3745 : tensor<32x672x14x14xf32>
    %v3747 = stablehlo.reshape %v3746 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3748 = stablehlo.reshape %v973 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3749 = stablehlo.reshape %v3674 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3750 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3751 = stablehlo.multiply %v3748, %v3749 : tensor<32x672x14x14xf32>
    %v3752 = stablehlo.reduce(%v3751 init: %v3750) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v3753 = stablehlo.logistic %v986 : tensor<32x672xf32>
    %v3754 = stablehlo.constant dense<1.0> : tensor<32x672xf32>
    %v3755 = stablehlo.subtract %v3754, %v3753 : tensor<32x672xf32>
    %v3756 = stablehlo.multiply %v3753, %v3755 : tensor<32x672xf32>
    %v3757 = stablehlo.multiply %v3752, %v3756 : tensor<32x672xf32>
    %v3758 = stablehlo.dot_general %v983, %v3757, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<32x672xf32>) -> tensor<28x672xf32>
    %v3759 = stablehlo.constant dense<0.05> : tensor<28x672xf32>
    %v3760 = stablehlo.multiply %v3758, %v3759 : tensor<28x672xf32>
    %v3761 = stablehlo.subtract %b10zW2, %v3760 : tensor<28x672xf32>
    %v3762 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3763 = stablehlo.reduce(%v3757 init: %v3762) applies stablehlo.add across dimensions = [0] : (tensor<32x672xf32>, tensor<f32>) -> tensor<672xf32>
    %v3764 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3765 = stablehlo.multiply %v3763, %v3764 : tensor<672xf32>
    %v3766 = stablehlo.subtract %b10zb2, %v3765 : tensor<672xf32>
    %v3767 = stablehlo.reshape %v3757 : (tensor<32x672xf32>) -> tensor<32x1x672xf32>
    %v3768 = stablehlo.dot_general %v3767, %b10zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x672xf32>, tensor<28x672xf32>) -> tensor<32x1x28xf32>
    %v3769 = stablehlo.reshape %v3768 : (tensor<32x1x28xf32>) -> tensor<32x28xf32>
    %v3770 = stablehlo.logistic %v981 : tensor<32x28xf32>
    %v3771 = stablehlo.constant dense<1.0> : tensor<32x28xf32>
    %v3772 = stablehlo.subtract %v3771, %v3770 : tensor<32x28xf32>
    %v3773 = stablehlo.multiply %v981, %v3772 : tensor<32x28xf32>
    %v3774 = stablehlo.add %v3771, %v3773 : tensor<32x28xf32>
    %v3775 = stablehlo.multiply %v3770, %v3774 : tensor<32x28xf32>
    %v3776 = stablehlo.multiply %v3769, %v3775 : tensor<32x28xf32>
    %v3777 = stablehlo.dot_general %v978, %v3776, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<32x28xf32>) -> tensor<672x28xf32>
    %v3778 = stablehlo.constant dense<0.05> : tensor<672x28xf32>
    %v3779 = stablehlo.multiply %v3777, %v3778 : tensor<672x28xf32>
    %v3780 = stablehlo.subtract %b10zW1, %v3779 : tensor<672x28xf32>
    %v3781 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3782 = stablehlo.reduce(%v3776 init: %v3781) applies stablehlo.add across dimensions = [0] : (tensor<32x28xf32>, tensor<f32>) -> tensor<28xf32>
    %v3783 = stablehlo.constant dense<0.05> : tensor<28xf32>
    %v3784 = stablehlo.multiply %v3782, %v3783 : tensor<28xf32>
    %v3785 = stablehlo.subtract %b10zb1, %v3784 : tensor<28xf32>
    %v3786 = stablehlo.reshape %v3747 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3787 = stablehlo.reshape %v969 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3788 = stablehlo.logistic %v3787 : tensor<32x672x14x14xf32>
    %v3789 = stablehlo.constant dense<1.0> : tensor<32x672x14x14xf32>
    %v3790 = stablehlo.subtract %v3789, %v3788 : tensor<32x672x14x14xf32>
    %v3791 = stablehlo.multiply %v3787, %v3790 : tensor<32x672x14x14xf32>
    %v3792 = stablehlo.add %v3789, %v3791 : tensor<32x672x14x14xf32>
    %v3793 = stablehlo.multiply %v3788, %v3792 : tensor<32x672x14x14xf32>
    %v3794 = stablehlo.multiply %v3786, %v3793 : tensor<32x672x14x14xf32>
    %v3795 = stablehlo.reshape %v3794 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3796 = stablehlo.reshape %v949 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3797 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3798 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3799 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3800 = stablehlo.reduce(%v3796 init: %v3797) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3801 = stablehlo.broadcast_in_dim %v3800, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3802 = stablehlo.divide %v3801, %v3798 : tensor<32x672x14x14xf32>
    %v3803 = stablehlo.subtract %v3796, %v3802 : tensor<32x672x14x14xf32>
    %v3804 = stablehlo.multiply %v3803, %v3803 : tensor<32x672x14x14xf32>
    %v3805 = stablehlo.reduce(%v3804 init: %v3797) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3806 = stablehlo.broadcast_in_dim %v3805, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3807 = stablehlo.divide %v3806, %v3798 : tensor<32x672x14x14xf32>
    %v3808 = stablehlo.add %v3807, %v3799 : tensor<32x672x14x14xf32>
    %v3809 = stablehlo.rsqrt %v3808 : tensor<32x672x14x14xf32>
    %v3810 = stablehlo.multiply %v3803, %v3809 : tensor<32x672x14x14xf32>
    %v3811 = stablehlo.broadcast_in_dim %b10dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3812 = stablehlo.reshape %v3795 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3813 = stablehlo.multiply %v3811, %v3812 : tensor<32x672x14x14xf32>
    %v3814 = stablehlo.reduce(%v3813 init: %v3797) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3815 = stablehlo.broadcast_in_dim %v3814, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3816 = stablehlo.multiply %v3810, %v3813 : tensor<32x672x14x14xf32>
    %v3817 = stablehlo.reduce(%v3816 init: %v3797) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3818 = stablehlo.broadcast_in_dim %v3817, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3819 = stablehlo.multiply %v3813, %v3798 : tensor<32x672x14x14xf32>
    %v3820 = stablehlo.subtract %v3819, %v3815 : tensor<32x672x14x14xf32>
    %v3821 = stablehlo.multiply %v3810, %v3818 : tensor<32x672x14x14xf32>
    %v3822 = stablehlo.subtract %v3820, %v3821 : tensor<32x672x14x14xf32>
    %v3823 = stablehlo.divide %v3809, %v3798 : tensor<32x672x14x14xf32>
    %v3824 = stablehlo.multiply %v3823, %v3822 : tensor<32x672x14x14xf32>
    %v3825 = stablehlo.reshape %v3824 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3826 = stablehlo.reshape %v3825 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3827 = stablehlo.reverse %b10dW, dims = [2, 3] : tensor<672x1x5x5xf32>
    %v3828 = stablehlo.convolution(%v3826, %v3827)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %v3829 = stablehlo.reshape %v3828 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3830 = stablehlo.reshape %v949 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3831 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3832 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3833 = stablehlo.reduce(%v3830 init: %v3831) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3834 = stablehlo.broadcast_in_dim %v3833, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3835 = stablehlo.divide %v3834, %v3832 : tensor<32x672x14x14xf32>
    %v3836 = stablehlo.subtract %v3830, %v3835 : tensor<32x672x14x14xf32>
    %v3837 = stablehlo.multiply %v3836, %v3836 : tensor<32x672x14x14xf32>
    %v3838 = stablehlo.reduce(%v3837 init: %v3831) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3839 = stablehlo.broadcast_in_dim %v3838, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3840 = stablehlo.divide %v3839, %v3832 : tensor<32x672x14x14xf32>
    %v3841 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3842 = stablehlo.add %v3840, %v3841 : tensor<32x672x14x14xf32>
    %v3843 = stablehlo.rsqrt %v3842 : tensor<32x672x14x14xf32>
    %v3844 = stablehlo.multiply %v3836, %v3843 : tensor<32x672x14x14xf32>
    %v3845 = stablehlo.reshape %v3795 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3846 = stablehlo.multiply %v3845, %v3844 : tensor<32x672x14x14xf32>
    %v3847 = stablehlo.reduce(%v3846 init: %v3831) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3848 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3849 = stablehlo.multiply %v3847, %v3848 : tensor<672xf32>
    %v3850 = stablehlo.subtract %b10dg, %v3849 : tensor<672xf32>
    %v3851 = stablehlo.reshape %v3795 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3852 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3853 = stablehlo.reduce(%v3851 init: %v3852) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3854 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3855 = stablehlo.multiply %v3853, %v3854 : tensor<672xf32>
    %v3856 = stablehlo.subtract %b10dbt, %v3855 : tensor<672xf32>
    %v3857 = stablehlo.reshape %v944 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3858 = stablehlo.reshape %v3825 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3859 = stablehlo.transpose %v3857, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3860 = stablehlo.transpose %v3858, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3861 = stablehlo.convolution(%v3859, %v3860)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 672 : i64, feature_group_count = 1 : i64} : (tensor<672x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<1x672x5x5xf32>
    %v3862 = stablehlo.reshape %v3861 : (tensor<1x672x5x5xf32>) -> tensor<672x1x5x5xf32>
    %v3863 = stablehlo.constant dense<0.05> : tensor<672x1x5x5xf32>
    %v3864 = stablehlo.multiply %v3862, %v3863 : tensor<672x1x5x5xf32>
    %v3865 = stablehlo.subtract %b10dW, %v3864 : tensor<672x1x5x5xf32>
    %v3866 = stablehlo.reshape %v3829 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3867 = stablehlo.reshape %v940 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3868 = stablehlo.logistic %v3867 : tensor<32x672x14x14xf32>
    %v3869 = stablehlo.constant dense<1.0> : tensor<32x672x14x14xf32>
    %v3870 = stablehlo.subtract %v3869, %v3868 : tensor<32x672x14x14xf32>
    %v3871 = stablehlo.multiply %v3867, %v3870 : tensor<32x672x14x14xf32>
    %v3872 = stablehlo.add %v3869, %v3871 : tensor<32x672x14x14xf32>
    %v3873 = stablehlo.multiply %v3868, %v3872 : tensor<32x672x14x14xf32>
    %v3874 = stablehlo.multiply %v3866, %v3873 : tensor<32x672x14x14xf32>
    %v3875 = stablehlo.reshape %v3874 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3876 = stablehlo.reshape %v920 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3877 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3878 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3879 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3880 = stablehlo.reduce(%v3876 init: %v3877) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3881 = stablehlo.broadcast_in_dim %v3880, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3882 = stablehlo.divide %v3881, %v3878 : tensor<32x672x14x14xf32>
    %v3883 = stablehlo.subtract %v3876, %v3882 : tensor<32x672x14x14xf32>
    %v3884 = stablehlo.multiply %v3883, %v3883 : tensor<32x672x14x14xf32>
    %v3885 = stablehlo.reduce(%v3884 init: %v3877) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3886 = stablehlo.broadcast_in_dim %v3885, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3887 = stablehlo.divide %v3886, %v3878 : tensor<32x672x14x14xf32>
    %v3888 = stablehlo.add %v3887, %v3879 : tensor<32x672x14x14xf32>
    %v3889 = stablehlo.rsqrt %v3888 : tensor<32x672x14x14xf32>
    %v3890 = stablehlo.multiply %v3883, %v3889 : tensor<32x672x14x14xf32>
    %v3891 = stablehlo.broadcast_in_dim %b10eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3892 = stablehlo.reshape %v3875 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3893 = stablehlo.multiply %v3891, %v3892 : tensor<32x672x14x14xf32>
    %v3894 = stablehlo.reduce(%v3893 init: %v3877) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3895 = stablehlo.broadcast_in_dim %v3894, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3896 = stablehlo.multiply %v3890, %v3893 : tensor<32x672x14x14xf32>
    %v3897 = stablehlo.reduce(%v3896 init: %v3877) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3898 = stablehlo.broadcast_in_dim %v3897, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3899 = stablehlo.multiply %v3893, %v3878 : tensor<32x672x14x14xf32>
    %v3900 = stablehlo.subtract %v3899, %v3895 : tensor<32x672x14x14xf32>
    %v3901 = stablehlo.multiply %v3890, %v3898 : tensor<32x672x14x14xf32>
    %v3902 = stablehlo.subtract %v3900, %v3901 : tensor<32x672x14x14xf32>
    %v3903 = stablehlo.divide %v3889, %v3878 : tensor<32x672x14x14xf32>
    %v3904 = stablehlo.multiply %v3903, %v3902 : tensor<32x672x14x14xf32>
    %v3905 = stablehlo.reshape %v3904 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v3906 = stablehlo.reshape %v3905 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3907 = stablehlo.reverse %b10eW, dims = [2, 3] : tensor<672x112x1x1xf32>
    %v3908 = stablehlo.transpose %v3907, dims = [1, 0, 2, 3] : (tensor<672x112x1x1xf32>) -> tensor<112x672x1x1xf32>
    %v3909 = stablehlo.convolution(%v3906, %v3908)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v3910 = stablehlo.reshape %v3909 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v3911 = stablehlo.reshape %v920 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3912 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3913 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v3914 = stablehlo.reduce(%v3911 init: %v3912) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3915 = stablehlo.broadcast_in_dim %v3914, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3916 = stablehlo.divide %v3915, %v3913 : tensor<32x672x14x14xf32>
    %v3917 = stablehlo.subtract %v3911, %v3916 : tensor<32x672x14x14xf32>
    %v3918 = stablehlo.multiply %v3917, %v3917 : tensor<32x672x14x14xf32>
    %v3919 = stablehlo.reduce(%v3918 init: %v3912) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3920 = stablehlo.broadcast_in_dim %v3919, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v3921 = stablehlo.divide %v3920, %v3913 : tensor<32x672x14x14xf32>
    %v3922 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v3923 = stablehlo.add %v3921, %v3922 : tensor<32x672x14x14xf32>
    %v3924 = stablehlo.rsqrt %v3923 : tensor<32x672x14x14xf32>
    %v3925 = stablehlo.multiply %v3917, %v3924 : tensor<32x672x14x14xf32>
    %v3926 = stablehlo.reshape %v3875 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3927 = stablehlo.multiply %v3926, %v3925 : tensor<32x672x14x14xf32>
    %v3928 = stablehlo.reduce(%v3927 init: %v3912) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3929 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3930 = stablehlo.multiply %v3928, %v3929 : tensor<672xf32>
    %v3931 = stablehlo.subtract %b10eg, %v3930 : tensor<672xf32>
    %v3932 = stablehlo.reshape %v3875 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3933 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3934 = stablehlo.reduce(%v3932 init: %v3933) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v3935 = stablehlo.constant dense<0.05> : tensor<672xf32>
    %v3936 = stablehlo.multiply %v3934, %v3935 : tensor<672xf32>
    %v3937 = stablehlo.subtract %b10ebt, %v3936 : tensor<672xf32>
    %v3938 = stablehlo.reshape %v915 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3939 = stablehlo.reshape %v3905 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v3940 = stablehlo.transpose %v3938, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %v3941 = stablehlo.transpose %v3939, dims = [1, 0, 2, 3] : (tensor<32x672x14x14xf32>) -> tensor<672x32x14x14xf32>
    %v3942 = stablehlo.convolution(%v3940, %v3941)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<112x32x14x14xf32>, tensor<672x32x14x14xf32>) -> tensor<112x672x1x1xf32>
    %v3943 = stablehlo.transpose %v3942, dims = [1, 0, 2, 3] : (tensor<112x672x1x1xf32>) -> tensor<672x112x1x1xf32>
    %v3944 = stablehlo.constant dense<0.05> : tensor<672x112x1x1xf32>
    %v3945 = stablehlo.multiply %v3943, %v3944 : tensor<672x112x1x1xf32>
    %v3946 = stablehlo.subtract %b10eW, %v3945 : tensor<672x112x1x1xf32>
    %v3947 = stablehlo.reshape %v3910 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3948 = stablehlo.reshape %v3639 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3949 = stablehlo.add %v3947, %v3948 : tensor<32x112x14x14xf32>
    %v3950 = stablehlo.reshape %v3949 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v3951 = stablehlo.reshape %v895 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3952 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3953 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v3954 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v3955 = stablehlo.reduce(%v3951 init: %v3952) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3956 = stablehlo.broadcast_in_dim %v3955, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3957 = stablehlo.divide %v3956, %v3953 : tensor<32x112x14x14xf32>
    %v3958 = stablehlo.subtract %v3951, %v3957 : tensor<32x112x14x14xf32>
    %v3959 = stablehlo.multiply %v3958, %v3958 : tensor<32x112x14x14xf32>
    %v3960 = stablehlo.reduce(%v3959 init: %v3952) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3961 = stablehlo.broadcast_in_dim %v3960, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3962 = stablehlo.divide %v3961, %v3953 : tensor<32x112x14x14xf32>
    %v3963 = stablehlo.add %v3962, %v3954 : tensor<32x112x14x14xf32>
    %v3964 = stablehlo.rsqrt %v3963 : tensor<32x112x14x14xf32>
    %v3965 = stablehlo.multiply %v3958, %v3964 : tensor<32x112x14x14xf32>
    %v3966 = stablehlo.broadcast_in_dim %b9pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3967 = stablehlo.reshape %v3950 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3968 = stablehlo.multiply %v3966, %v3967 : tensor<32x112x14x14xf32>
    %v3969 = stablehlo.reduce(%v3968 init: %v3952) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3970 = stablehlo.broadcast_in_dim %v3969, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3971 = stablehlo.multiply %v3965, %v3968 : tensor<32x112x14x14xf32>
    %v3972 = stablehlo.reduce(%v3971 init: %v3952) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3973 = stablehlo.broadcast_in_dim %v3972, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3974 = stablehlo.multiply %v3968, %v3953 : tensor<32x112x14x14xf32>
    %v3975 = stablehlo.subtract %v3974, %v3970 : tensor<32x112x14x14xf32>
    %v3976 = stablehlo.multiply %v3965, %v3973 : tensor<32x112x14x14xf32>
    %v3977 = stablehlo.subtract %v3975, %v3976 : tensor<32x112x14x14xf32>
    %v3978 = stablehlo.divide %v3964, %v3953 : tensor<32x112x14x14xf32>
    %v3979 = stablehlo.multiply %v3978, %v3977 : tensor<32x112x14x14xf32>
    %v3980 = stablehlo.reshape %v3979 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v3981 = stablehlo.reshape %v3980 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3982 = stablehlo.reverse %b9pW, dims = [2, 3] : tensor<112x480x1x1xf32>
    %v3983 = stablehlo.transpose %v3982, dims = [1, 0, 2, 3] : (tensor<112x480x1x1xf32>) -> tensor<480x112x1x1xf32>
    %v3984 = stablehlo.convolution(%v3981, %v3983)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<480x112x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v3985 = stablehlo.reshape %v3984 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v3986 = stablehlo.reshape %v895 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v3987 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3988 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v3989 = stablehlo.reduce(%v3986 init: %v3987) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3990 = stablehlo.broadcast_in_dim %v3989, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3991 = stablehlo.divide %v3990, %v3988 : tensor<32x112x14x14xf32>
    %v3992 = stablehlo.subtract %v3986, %v3991 : tensor<32x112x14x14xf32>
    %v3993 = stablehlo.multiply %v3992, %v3992 : tensor<32x112x14x14xf32>
    %v3994 = stablehlo.reduce(%v3993 init: %v3987) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v3995 = stablehlo.broadcast_in_dim %v3994, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v3996 = stablehlo.divide %v3995, %v3988 : tensor<32x112x14x14xf32>
    %v3997 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v3998 = stablehlo.add %v3996, %v3997 : tensor<32x112x14x14xf32>
    %v3999 = stablehlo.rsqrt %v3998 : tensor<32x112x14x14xf32>
    %v4000 = stablehlo.multiply %v3992, %v3999 : tensor<32x112x14x14xf32>
    %v4001 = stablehlo.reshape %v3950 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v4002 = stablehlo.multiply %v4001, %v4000 : tensor<32x112x14x14xf32>
    %v4003 = stablehlo.reduce(%v4002 init: %v3987) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v4004 = stablehlo.constant dense<0.05> : tensor<112xf32>
    %v4005 = stablehlo.multiply %v4003, %v4004 : tensor<112xf32>
    %v4006 = stablehlo.subtract %b9pg, %v4005 : tensor<112xf32>
    %v4007 = stablehlo.reshape %v3950 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v4008 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4009 = stablehlo.reduce(%v4007 init: %v4008) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v4010 = stablehlo.constant dense<0.05> : tensor<112xf32>
    %v4011 = stablehlo.multiply %v4009, %v4010 : tensor<112xf32>
    %v4012 = stablehlo.subtract %b9pbt, %v4011 : tensor<112xf32>
    %v4013 = stablehlo.reshape %v890 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4014 = stablehlo.reshape %v3980 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v4015 = stablehlo.transpose %v4013, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4016 = stablehlo.transpose %v4014, dims = [1, 0, 2, 3] : (tensor<32x112x14x14xf32>) -> tensor<112x32x14x14xf32>
    %v4017 = stablehlo.convolution(%v4015, %v4016)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<112x32x14x14xf32>) -> tensor<480x112x1x1xf32>
    %v4018 = stablehlo.transpose %v4017, dims = [1, 0, 2, 3] : (tensor<480x112x1x1xf32>) -> tensor<112x480x1x1xf32>
    %v4019 = stablehlo.constant dense<0.05> : tensor<112x480x1x1xf32>
    %v4020 = stablehlo.multiply %v4018, %v4019 : tensor<112x480x1x1xf32>
    %v4021 = stablehlo.subtract %b9pW, %v4020 : tensor<112x480x1x1xf32>
    %v4022 = stablehlo.reshape %v873 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4023 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4024 = stablehlo.reduce(%v4022 init: %v4023) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4025 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v4026 = stablehlo.divide %v4024, %v4025 : tensor<32x480xf32>
    %v4027 = stablehlo.dot_general %v4026, %b9zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v4028 = stablehlo.broadcast_in_dim %b9zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v4029 = stablehlo.add %v4027, %v4028 : tensor<32x20xf32>
    %v4030 = stablehlo.logistic %v4029 : tensor<32x20xf32>
    %v4031 = stablehlo.multiply %v4029, %v4030 : tensor<32x20xf32>
    %v4032 = stablehlo.dot_general %v4031, %b9zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v4033 = stablehlo.broadcast_in_dim %b9zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v4034 = stablehlo.add %v4032, %v4033 : tensor<32x480xf32>
    %v4035 = stablehlo.logistic %v4034 : tensor<32x480xf32>
    %v4036 = stablehlo.reshape %v3985 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4037 = stablehlo.broadcast_in_dim %v4035, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v4038 = stablehlo.multiply %v4037, %v4036 : tensor<32x480x14x14xf32>
    %v4039 = stablehlo.multiply %v4022, %v4036 : tensor<32x480x14x14xf32>
    %v4040 = stablehlo.reduce(%v4039 init: %v4023) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4041 = stablehlo.constant dense<1.0> : tensor<32x480xf32>
    %v4042 = stablehlo.subtract %v4041, %v4035 : tensor<32x480xf32>
    %v4043 = stablehlo.multiply %v4035, %v4042 : tensor<32x480xf32>
    %v4044 = stablehlo.multiply %v4040, %v4043 : tensor<32x480xf32>
    %v4045 = stablehlo.dot_general %v4044, %b9zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<20x480xf32>) -> tensor<32x20xf32>
    %v4046 = stablehlo.logistic %v4029 : tensor<32x20xf32>
    %v4047 = stablehlo.constant dense<1.0> : tensor<32x20xf32>
    %v4048 = stablehlo.subtract %v4047, %v4046 : tensor<32x20xf32>
    %v4049 = stablehlo.multiply %v4029, %v4048 : tensor<32x20xf32>
    %v4050 = stablehlo.add %v4047, %v4049 : tensor<32x20xf32>
    %v4051 = stablehlo.multiply %v4046, %v4050 : tensor<32x20xf32>
    %v4052 = stablehlo.multiply %v4045, %v4051 : tensor<32x20xf32>
    %v4053 = stablehlo.dot_general %v4052, %b9zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<480x20xf32>) -> tensor<32x480xf32>
    %v4054 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v4055 = stablehlo.divide %v4053, %v4054 : tensor<32x480xf32>
    %v4056 = stablehlo.broadcast_in_dim %v4055, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v4057 = stablehlo.add %v4038, %v4056 : tensor<32x480x14x14xf32>
    %v4058 = stablehlo.reshape %v4057 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4059 = stablehlo.reshape %v873 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4060 = stablehlo.reshape %v3985 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4061 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4062 = stablehlo.multiply %v4059, %v4060 : tensor<32x480x14x14xf32>
    %v4063 = stablehlo.reduce(%v4062 init: %v4061) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4064 = stablehlo.logistic %v886 : tensor<32x480xf32>
    %v4065 = stablehlo.constant dense<1.0> : tensor<32x480xf32>
    %v4066 = stablehlo.subtract %v4065, %v4064 : tensor<32x480xf32>
    %v4067 = stablehlo.multiply %v4064, %v4066 : tensor<32x480xf32>
    %v4068 = stablehlo.multiply %v4063, %v4067 : tensor<32x480xf32>
    %v4069 = stablehlo.dot_general %v883, %v4068, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<32x480xf32>) -> tensor<20x480xf32>
    %v4070 = stablehlo.constant dense<0.05> : tensor<20x480xf32>
    %v4071 = stablehlo.multiply %v4069, %v4070 : tensor<20x480xf32>
    %v4072 = stablehlo.subtract %b9zW2, %v4071 : tensor<20x480xf32>
    %v4073 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4074 = stablehlo.reduce(%v4068 init: %v4073) applies stablehlo.add across dimensions = [0] : (tensor<32x480xf32>, tensor<f32>) -> tensor<480xf32>
    %v4075 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4076 = stablehlo.multiply %v4074, %v4075 : tensor<480xf32>
    %v4077 = stablehlo.subtract %b9zb2, %v4076 : tensor<480xf32>
    %v4078 = stablehlo.reshape %v4068 : (tensor<32x480xf32>) -> tensor<32x1x480xf32>
    %v4079 = stablehlo.dot_general %v4078, %b9zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x480xf32>, tensor<20x480xf32>) -> tensor<32x1x20xf32>
    %v4080 = stablehlo.reshape %v4079 : (tensor<32x1x20xf32>) -> tensor<32x20xf32>
    %v4081 = stablehlo.logistic %v881 : tensor<32x20xf32>
    %v4082 = stablehlo.constant dense<1.0> : tensor<32x20xf32>
    %v4083 = stablehlo.subtract %v4082, %v4081 : tensor<32x20xf32>
    %v4084 = stablehlo.multiply %v881, %v4083 : tensor<32x20xf32>
    %v4085 = stablehlo.add %v4082, %v4084 : tensor<32x20xf32>
    %v4086 = stablehlo.multiply %v4081, %v4085 : tensor<32x20xf32>
    %v4087 = stablehlo.multiply %v4080, %v4086 : tensor<32x20xf32>
    %v4088 = stablehlo.dot_general %v878, %v4087, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<32x20xf32>) -> tensor<480x20xf32>
    %v4089 = stablehlo.constant dense<0.05> : tensor<480x20xf32>
    %v4090 = stablehlo.multiply %v4088, %v4089 : tensor<480x20xf32>
    %v4091 = stablehlo.subtract %b9zW1, %v4090 : tensor<480x20xf32>
    %v4092 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4093 = stablehlo.reduce(%v4087 init: %v4092) applies stablehlo.add across dimensions = [0] : (tensor<32x20xf32>, tensor<f32>) -> tensor<20xf32>
    %v4094 = stablehlo.constant dense<0.05> : tensor<20xf32>
    %v4095 = stablehlo.multiply %v4093, %v4094 : tensor<20xf32>
    %v4096 = stablehlo.subtract %b9zb1, %v4095 : tensor<20xf32>
    %v4097 = stablehlo.reshape %v4058 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4098 = stablehlo.reshape %v869 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4099 = stablehlo.logistic %v4098 : tensor<32x480x14x14xf32>
    %v4100 = stablehlo.constant dense<1.0> : tensor<32x480x14x14xf32>
    %v4101 = stablehlo.subtract %v4100, %v4099 : tensor<32x480x14x14xf32>
    %v4102 = stablehlo.multiply %v4098, %v4101 : tensor<32x480x14x14xf32>
    %v4103 = stablehlo.add %v4100, %v4102 : tensor<32x480x14x14xf32>
    %v4104 = stablehlo.multiply %v4099, %v4103 : tensor<32x480x14x14xf32>
    %v4105 = stablehlo.multiply %v4097, %v4104 : tensor<32x480x14x14xf32>
    %v4106 = stablehlo.reshape %v4105 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4107 = stablehlo.reshape %v849 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4108 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4109 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4110 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4111 = stablehlo.reduce(%v4107 init: %v4108) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4112 = stablehlo.broadcast_in_dim %v4111, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4113 = stablehlo.divide %v4112, %v4109 : tensor<32x480x14x14xf32>
    %v4114 = stablehlo.subtract %v4107, %v4113 : tensor<32x480x14x14xf32>
    %v4115 = stablehlo.multiply %v4114, %v4114 : tensor<32x480x14x14xf32>
    %v4116 = stablehlo.reduce(%v4115 init: %v4108) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4117 = stablehlo.broadcast_in_dim %v4116, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4118 = stablehlo.divide %v4117, %v4109 : tensor<32x480x14x14xf32>
    %v4119 = stablehlo.add %v4118, %v4110 : tensor<32x480x14x14xf32>
    %v4120 = stablehlo.rsqrt %v4119 : tensor<32x480x14x14xf32>
    %v4121 = stablehlo.multiply %v4114, %v4120 : tensor<32x480x14x14xf32>
    %v4122 = stablehlo.broadcast_in_dim %b9dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4123 = stablehlo.reshape %v4106 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4124 = stablehlo.multiply %v4122, %v4123 : tensor<32x480x14x14xf32>
    %v4125 = stablehlo.reduce(%v4124 init: %v4108) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4126 = stablehlo.broadcast_in_dim %v4125, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4127 = stablehlo.multiply %v4121, %v4124 : tensor<32x480x14x14xf32>
    %v4128 = stablehlo.reduce(%v4127 init: %v4108) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4129 = stablehlo.broadcast_in_dim %v4128, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4130 = stablehlo.multiply %v4124, %v4109 : tensor<32x480x14x14xf32>
    %v4131 = stablehlo.subtract %v4130, %v4126 : tensor<32x480x14x14xf32>
    %v4132 = stablehlo.multiply %v4121, %v4129 : tensor<32x480x14x14xf32>
    %v4133 = stablehlo.subtract %v4131, %v4132 : tensor<32x480x14x14xf32>
    %v4134 = stablehlo.divide %v4120, %v4109 : tensor<32x480x14x14xf32>
    %v4135 = stablehlo.multiply %v4134, %v4133 : tensor<32x480x14x14xf32>
    %v4136 = stablehlo.reshape %v4135 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4137 = stablehlo.reshape %v4136 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4138 = stablehlo.reverse %b9dW, dims = [2, 3] : tensor<480x1x5x5xf32>
    %v4139 = stablehlo.convolution(%v4137, %v4138)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x5x5xf32>) -> tensor<32x480x14x14xf32>
    %v4140 = stablehlo.reshape %v4139 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4141 = stablehlo.reshape %v849 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4142 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4143 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4144 = stablehlo.reduce(%v4141 init: %v4142) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4145 = stablehlo.broadcast_in_dim %v4144, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4146 = stablehlo.divide %v4145, %v4143 : tensor<32x480x14x14xf32>
    %v4147 = stablehlo.subtract %v4141, %v4146 : tensor<32x480x14x14xf32>
    %v4148 = stablehlo.multiply %v4147, %v4147 : tensor<32x480x14x14xf32>
    %v4149 = stablehlo.reduce(%v4148 init: %v4142) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4150 = stablehlo.broadcast_in_dim %v4149, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4151 = stablehlo.divide %v4150, %v4143 : tensor<32x480x14x14xf32>
    %v4152 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4153 = stablehlo.add %v4151, %v4152 : tensor<32x480x14x14xf32>
    %v4154 = stablehlo.rsqrt %v4153 : tensor<32x480x14x14xf32>
    %v4155 = stablehlo.multiply %v4147, %v4154 : tensor<32x480x14x14xf32>
    %v4156 = stablehlo.reshape %v4106 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4157 = stablehlo.multiply %v4156, %v4155 : tensor<32x480x14x14xf32>
    %v4158 = stablehlo.reduce(%v4157 init: %v4142) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4159 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4160 = stablehlo.multiply %v4158, %v4159 : tensor<480xf32>
    %v4161 = stablehlo.subtract %b9dg, %v4160 : tensor<480xf32>
    %v4162 = stablehlo.reshape %v4106 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4163 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4164 = stablehlo.reduce(%v4162 init: %v4163) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4165 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4166 = stablehlo.multiply %v4164, %v4165 : tensor<480xf32>
    %v4167 = stablehlo.subtract %b9dbt, %v4166 : tensor<480xf32>
    %v4168 = stablehlo.reshape %v844 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4169 = stablehlo.reshape %v4136 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4170 = stablehlo.transpose %v4168, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4171 = stablehlo.transpose %v4169, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4172 = stablehlo.convolution(%v4170, %v4171)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 480 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<1x480x5x5xf32>
    %v4173 = stablehlo.reshape %v4172 : (tensor<1x480x5x5xf32>) -> tensor<480x1x5x5xf32>
    %v4174 = stablehlo.constant dense<0.05> : tensor<480x1x5x5xf32>
    %v4175 = stablehlo.multiply %v4173, %v4174 : tensor<480x1x5x5xf32>
    %v4176 = stablehlo.subtract %b9dW, %v4175 : tensor<480x1x5x5xf32>
    %v4177 = stablehlo.reshape %v4140 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4178 = stablehlo.reshape %v840 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4179 = stablehlo.logistic %v4178 : tensor<32x480x14x14xf32>
    %v4180 = stablehlo.constant dense<1.0> : tensor<32x480x14x14xf32>
    %v4181 = stablehlo.subtract %v4180, %v4179 : tensor<32x480x14x14xf32>
    %v4182 = stablehlo.multiply %v4178, %v4181 : tensor<32x480x14x14xf32>
    %v4183 = stablehlo.add %v4180, %v4182 : tensor<32x480x14x14xf32>
    %v4184 = stablehlo.multiply %v4179, %v4183 : tensor<32x480x14x14xf32>
    %v4185 = stablehlo.multiply %v4177, %v4184 : tensor<32x480x14x14xf32>
    %v4186 = stablehlo.reshape %v4185 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4187 = stablehlo.reshape %v820 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4188 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4189 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4190 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4191 = stablehlo.reduce(%v4187 init: %v4188) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4192 = stablehlo.broadcast_in_dim %v4191, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4193 = stablehlo.divide %v4192, %v4189 : tensor<32x480x14x14xf32>
    %v4194 = stablehlo.subtract %v4187, %v4193 : tensor<32x480x14x14xf32>
    %v4195 = stablehlo.multiply %v4194, %v4194 : tensor<32x480x14x14xf32>
    %v4196 = stablehlo.reduce(%v4195 init: %v4188) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4197 = stablehlo.broadcast_in_dim %v4196, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4198 = stablehlo.divide %v4197, %v4189 : tensor<32x480x14x14xf32>
    %v4199 = stablehlo.add %v4198, %v4190 : tensor<32x480x14x14xf32>
    %v4200 = stablehlo.rsqrt %v4199 : tensor<32x480x14x14xf32>
    %v4201 = stablehlo.multiply %v4194, %v4200 : tensor<32x480x14x14xf32>
    %v4202 = stablehlo.broadcast_in_dim %b9eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4203 = stablehlo.reshape %v4186 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4204 = stablehlo.multiply %v4202, %v4203 : tensor<32x480x14x14xf32>
    %v4205 = stablehlo.reduce(%v4204 init: %v4188) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4206 = stablehlo.broadcast_in_dim %v4205, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4207 = stablehlo.multiply %v4201, %v4204 : tensor<32x480x14x14xf32>
    %v4208 = stablehlo.reduce(%v4207 init: %v4188) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4209 = stablehlo.broadcast_in_dim %v4208, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4210 = stablehlo.multiply %v4204, %v4189 : tensor<32x480x14x14xf32>
    %v4211 = stablehlo.subtract %v4210, %v4206 : tensor<32x480x14x14xf32>
    %v4212 = stablehlo.multiply %v4201, %v4209 : tensor<32x480x14x14xf32>
    %v4213 = stablehlo.subtract %v4211, %v4212 : tensor<32x480x14x14xf32>
    %v4214 = stablehlo.divide %v4200, %v4189 : tensor<32x480x14x14xf32>
    %v4215 = stablehlo.multiply %v4214, %v4213 : tensor<32x480x14x14xf32>
    %v4216 = stablehlo.reshape %v4215 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4217 = stablehlo.reshape %v4216 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4218 = stablehlo.reverse %b9eW, dims = [2, 3] : tensor<480x80x1x1xf32>
    %v4219 = stablehlo.transpose %v4218, dims = [1, 0, 2, 3] : (tensor<480x80x1x1xf32>) -> tensor<80x480x1x1xf32>
    %v4220 = stablehlo.convolution(%v4217, %v4219)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v4221 = stablehlo.reshape %v4220 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v4222 = stablehlo.reshape %v820 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4223 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4224 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4225 = stablehlo.reduce(%v4222 init: %v4223) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4226 = stablehlo.broadcast_in_dim %v4225, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4227 = stablehlo.divide %v4226, %v4224 : tensor<32x480x14x14xf32>
    %v4228 = stablehlo.subtract %v4222, %v4227 : tensor<32x480x14x14xf32>
    %v4229 = stablehlo.multiply %v4228, %v4228 : tensor<32x480x14x14xf32>
    %v4230 = stablehlo.reduce(%v4229 init: %v4223) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4231 = stablehlo.broadcast_in_dim %v4230, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4232 = stablehlo.divide %v4231, %v4224 : tensor<32x480x14x14xf32>
    %v4233 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4234 = stablehlo.add %v4232, %v4233 : tensor<32x480x14x14xf32>
    %v4235 = stablehlo.rsqrt %v4234 : tensor<32x480x14x14xf32>
    %v4236 = stablehlo.multiply %v4228, %v4235 : tensor<32x480x14x14xf32>
    %v4237 = stablehlo.reshape %v4186 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4238 = stablehlo.multiply %v4237, %v4236 : tensor<32x480x14x14xf32>
    %v4239 = stablehlo.reduce(%v4238 init: %v4223) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4240 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4241 = stablehlo.multiply %v4239, %v4240 : tensor<480xf32>
    %v4242 = stablehlo.subtract %b9eg, %v4241 : tensor<480xf32>
    %v4243 = stablehlo.reshape %v4186 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4244 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4245 = stablehlo.reduce(%v4243 init: %v4244) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4246 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4247 = stablehlo.multiply %v4245, %v4246 : tensor<480xf32>
    %v4248 = stablehlo.subtract %b9ebt, %v4247 : tensor<480xf32>
    %v4249 = stablehlo.reshape %v815 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4250 = stablehlo.reshape %v4216 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4251 = stablehlo.transpose %v4249, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %v4252 = stablehlo.transpose %v4250, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4253 = stablehlo.convolution(%v4251, %v4252)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<80x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<80x480x1x1xf32>
    %v4254 = stablehlo.transpose %v4253, dims = [1, 0, 2, 3] : (tensor<80x480x1x1xf32>) -> tensor<480x80x1x1xf32>
    %v4255 = stablehlo.constant dense<0.05> : tensor<480x80x1x1xf32>
    %v4256 = stablehlo.multiply %v4254, %v4255 : tensor<480x80x1x1xf32>
    %v4257 = stablehlo.subtract %b9eW, %v4256 : tensor<480x80x1x1xf32>
    %v4258 = stablehlo.reshape %v791 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4259 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4260 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v4261 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v4262 = stablehlo.reduce(%v4258 init: %v4259) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4263 = stablehlo.broadcast_in_dim %v4262, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4264 = stablehlo.divide %v4263, %v4260 : tensor<32x80x14x14xf32>
    %v4265 = stablehlo.subtract %v4258, %v4264 : tensor<32x80x14x14xf32>
    %v4266 = stablehlo.multiply %v4265, %v4265 : tensor<32x80x14x14xf32>
    %v4267 = stablehlo.reduce(%v4266 init: %v4259) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4268 = stablehlo.broadcast_in_dim %v4267, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4269 = stablehlo.divide %v4268, %v4260 : tensor<32x80x14x14xf32>
    %v4270 = stablehlo.add %v4269, %v4261 : tensor<32x80x14x14xf32>
    %v4271 = stablehlo.rsqrt %v4270 : tensor<32x80x14x14xf32>
    %v4272 = stablehlo.multiply %v4265, %v4271 : tensor<32x80x14x14xf32>
    %v4273 = stablehlo.broadcast_in_dim %b8pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4274 = stablehlo.reshape %v4221 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4275 = stablehlo.multiply %v4273, %v4274 : tensor<32x80x14x14xf32>
    %v4276 = stablehlo.reduce(%v4275 init: %v4259) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4277 = stablehlo.broadcast_in_dim %v4276, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4278 = stablehlo.multiply %v4272, %v4275 : tensor<32x80x14x14xf32>
    %v4279 = stablehlo.reduce(%v4278 init: %v4259) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4280 = stablehlo.broadcast_in_dim %v4279, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4281 = stablehlo.multiply %v4275, %v4260 : tensor<32x80x14x14xf32>
    %v4282 = stablehlo.subtract %v4281, %v4277 : tensor<32x80x14x14xf32>
    %v4283 = stablehlo.multiply %v4272, %v4280 : tensor<32x80x14x14xf32>
    %v4284 = stablehlo.subtract %v4282, %v4283 : tensor<32x80x14x14xf32>
    %v4285 = stablehlo.divide %v4271, %v4260 : tensor<32x80x14x14xf32>
    %v4286 = stablehlo.multiply %v4285, %v4284 : tensor<32x80x14x14xf32>
    %v4287 = stablehlo.reshape %v4286 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v4288 = stablehlo.reshape %v4287 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4289 = stablehlo.reverse %b8pW, dims = [2, 3] : tensor<80x480x1x1xf32>
    %v4290 = stablehlo.transpose %v4289, dims = [1, 0, 2, 3] : (tensor<80x480x1x1xf32>) -> tensor<480x80x1x1xf32>
    %v4291 = stablehlo.convolution(%v4288, %v4290)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v4292 = stablehlo.reshape %v4291 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4293 = stablehlo.reshape %v791 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4294 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4295 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v4296 = stablehlo.reduce(%v4293 init: %v4294) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4297 = stablehlo.broadcast_in_dim %v4296, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4298 = stablehlo.divide %v4297, %v4295 : tensor<32x80x14x14xf32>
    %v4299 = stablehlo.subtract %v4293, %v4298 : tensor<32x80x14x14xf32>
    %v4300 = stablehlo.multiply %v4299, %v4299 : tensor<32x80x14x14xf32>
    %v4301 = stablehlo.reduce(%v4300 init: %v4294) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4302 = stablehlo.broadcast_in_dim %v4301, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4303 = stablehlo.divide %v4302, %v4295 : tensor<32x80x14x14xf32>
    %v4304 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v4305 = stablehlo.add %v4303, %v4304 : tensor<32x80x14x14xf32>
    %v4306 = stablehlo.rsqrt %v4305 : tensor<32x80x14x14xf32>
    %v4307 = stablehlo.multiply %v4299, %v4306 : tensor<32x80x14x14xf32>
    %v4308 = stablehlo.reshape %v4221 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4309 = stablehlo.multiply %v4308, %v4307 : tensor<32x80x14x14xf32>
    %v4310 = stablehlo.reduce(%v4309 init: %v4294) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4311 = stablehlo.constant dense<0.05> : tensor<80xf32>
    %v4312 = stablehlo.multiply %v4310, %v4311 : tensor<80xf32>
    %v4313 = stablehlo.subtract %b8pg, %v4312 : tensor<80xf32>
    %v4314 = stablehlo.reshape %v4221 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4315 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4316 = stablehlo.reduce(%v4314 init: %v4315) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4317 = stablehlo.constant dense<0.05> : tensor<80xf32>
    %v4318 = stablehlo.multiply %v4316, %v4317 : tensor<80xf32>
    %v4319 = stablehlo.subtract %b8pbt, %v4318 : tensor<80xf32>
    %v4320 = stablehlo.reshape %v786 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4321 = stablehlo.reshape %v4287 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4322 = stablehlo.transpose %v4320, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4323 = stablehlo.transpose %v4321, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %v4324 = stablehlo.convolution(%v4322, %v4323)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<80x32x14x14xf32>) -> tensor<480x80x1x1xf32>
    %v4325 = stablehlo.transpose %v4324, dims = [1, 0, 2, 3] : (tensor<480x80x1x1xf32>) -> tensor<80x480x1x1xf32>
    %v4326 = stablehlo.constant dense<0.05> : tensor<80x480x1x1xf32>
    %v4327 = stablehlo.multiply %v4325, %v4326 : tensor<80x480x1x1xf32>
    %v4328 = stablehlo.subtract %b8pW, %v4327 : tensor<80x480x1x1xf32>
    %v4329 = stablehlo.reshape %v769 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4330 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4331 = stablehlo.reduce(%v4329 init: %v4330) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4332 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v4333 = stablehlo.divide %v4331, %v4332 : tensor<32x480xf32>
    %v4334 = stablehlo.dot_general %v4333, %b8zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v4335 = stablehlo.broadcast_in_dim %b8zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v4336 = stablehlo.add %v4334, %v4335 : tensor<32x20xf32>
    %v4337 = stablehlo.logistic %v4336 : tensor<32x20xf32>
    %v4338 = stablehlo.multiply %v4336, %v4337 : tensor<32x20xf32>
    %v4339 = stablehlo.dot_general %v4338, %b8zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v4340 = stablehlo.broadcast_in_dim %b8zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v4341 = stablehlo.add %v4339, %v4340 : tensor<32x480xf32>
    %v4342 = stablehlo.logistic %v4341 : tensor<32x480xf32>
    %v4343 = stablehlo.reshape %v4292 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4344 = stablehlo.broadcast_in_dim %v4342, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v4345 = stablehlo.multiply %v4344, %v4343 : tensor<32x480x14x14xf32>
    %v4346 = stablehlo.multiply %v4329, %v4343 : tensor<32x480x14x14xf32>
    %v4347 = stablehlo.reduce(%v4346 init: %v4330) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4348 = stablehlo.constant dense<1.0> : tensor<32x480xf32>
    %v4349 = stablehlo.subtract %v4348, %v4342 : tensor<32x480xf32>
    %v4350 = stablehlo.multiply %v4342, %v4349 : tensor<32x480xf32>
    %v4351 = stablehlo.multiply %v4347, %v4350 : tensor<32x480xf32>
    %v4352 = stablehlo.dot_general %v4351, %b8zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<20x480xf32>) -> tensor<32x20xf32>
    %v4353 = stablehlo.logistic %v4336 : tensor<32x20xf32>
    %v4354 = stablehlo.constant dense<1.0> : tensor<32x20xf32>
    %v4355 = stablehlo.subtract %v4354, %v4353 : tensor<32x20xf32>
    %v4356 = stablehlo.multiply %v4336, %v4355 : tensor<32x20xf32>
    %v4357 = stablehlo.add %v4354, %v4356 : tensor<32x20xf32>
    %v4358 = stablehlo.multiply %v4353, %v4357 : tensor<32x20xf32>
    %v4359 = stablehlo.multiply %v4352, %v4358 : tensor<32x20xf32>
    %v4360 = stablehlo.dot_general %v4359, %b8zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<480x20xf32>) -> tensor<32x480xf32>
    %v4361 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v4362 = stablehlo.divide %v4360, %v4361 : tensor<32x480xf32>
    %v4363 = stablehlo.broadcast_in_dim %v4362, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v4364 = stablehlo.add %v4345, %v4363 : tensor<32x480x14x14xf32>
    %v4365 = stablehlo.reshape %v4364 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4366 = stablehlo.reshape %v769 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4367 = stablehlo.reshape %v4292 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4368 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4369 = stablehlo.multiply %v4366, %v4367 : tensor<32x480x14x14xf32>
    %v4370 = stablehlo.reduce(%v4369 init: %v4368) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4371 = stablehlo.logistic %v782 : tensor<32x480xf32>
    %v4372 = stablehlo.constant dense<1.0> : tensor<32x480xf32>
    %v4373 = stablehlo.subtract %v4372, %v4371 : tensor<32x480xf32>
    %v4374 = stablehlo.multiply %v4371, %v4373 : tensor<32x480xf32>
    %v4375 = stablehlo.multiply %v4370, %v4374 : tensor<32x480xf32>
    %v4376 = stablehlo.dot_general %v779, %v4375, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<32x480xf32>) -> tensor<20x480xf32>
    %v4377 = stablehlo.constant dense<0.05> : tensor<20x480xf32>
    %v4378 = stablehlo.multiply %v4376, %v4377 : tensor<20x480xf32>
    %v4379 = stablehlo.subtract %b8zW2, %v4378 : tensor<20x480xf32>
    %v4380 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4381 = stablehlo.reduce(%v4375 init: %v4380) applies stablehlo.add across dimensions = [0] : (tensor<32x480xf32>, tensor<f32>) -> tensor<480xf32>
    %v4382 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4383 = stablehlo.multiply %v4381, %v4382 : tensor<480xf32>
    %v4384 = stablehlo.subtract %b8zb2, %v4383 : tensor<480xf32>
    %v4385 = stablehlo.reshape %v4375 : (tensor<32x480xf32>) -> tensor<32x1x480xf32>
    %v4386 = stablehlo.dot_general %v4385, %b8zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x480xf32>, tensor<20x480xf32>) -> tensor<32x1x20xf32>
    %v4387 = stablehlo.reshape %v4386 : (tensor<32x1x20xf32>) -> tensor<32x20xf32>
    %v4388 = stablehlo.logistic %v777 : tensor<32x20xf32>
    %v4389 = stablehlo.constant dense<1.0> : tensor<32x20xf32>
    %v4390 = stablehlo.subtract %v4389, %v4388 : tensor<32x20xf32>
    %v4391 = stablehlo.multiply %v777, %v4390 : tensor<32x20xf32>
    %v4392 = stablehlo.add %v4389, %v4391 : tensor<32x20xf32>
    %v4393 = stablehlo.multiply %v4388, %v4392 : tensor<32x20xf32>
    %v4394 = stablehlo.multiply %v4387, %v4393 : tensor<32x20xf32>
    %v4395 = stablehlo.dot_general %v774, %v4394, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<32x20xf32>) -> tensor<480x20xf32>
    %v4396 = stablehlo.constant dense<0.05> : tensor<480x20xf32>
    %v4397 = stablehlo.multiply %v4395, %v4396 : tensor<480x20xf32>
    %v4398 = stablehlo.subtract %b8zW1, %v4397 : tensor<480x20xf32>
    %v4399 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4400 = stablehlo.reduce(%v4394 init: %v4399) applies stablehlo.add across dimensions = [0] : (tensor<32x20xf32>, tensor<f32>) -> tensor<20xf32>
    %v4401 = stablehlo.constant dense<0.05> : tensor<20xf32>
    %v4402 = stablehlo.multiply %v4400, %v4401 : tensor<20xf32>
    %v4403 = stablehlo.subtract %b8zb1, %v4402 : tensor<20xf32>
    %v4404 = stablehlo.reshape %v4365 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4405 = stablehlo.reshape %v765 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4406 = stablehlo.logistic %v4405 : tensor<32x480x14x14xf32>
    %v4407 = stablehlo.constant dense<1.0> : tensor<32x480x14x14xf32>
    %v4408 = stablehlo.subtract %v4407, %v4406 : tensor<32x480x14x14xf32>
    %v4409 = stablehlo.multiply %v4405, %v4408 : tensor<32x480x14x14xf32>
    %v4410 = stablehlo.add %v4407, %v4409 : tensor<32x480x14x14xf32>
    %v4411 = stablehlo.multiply %v4406, %v4410 : tensor<32x480x14x14xf32>
    %v4412 = stablehlo.multiply %v4404, %v4411 : tensor<32x480x14x14xf32>
    %v4413 = stablehlo.reshape %v4412 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4414 = stablehlo.reshape %v745 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4415 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4416 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4417 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4418 = stablehlo.reduce(%v4414 init: %v4415) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4419 = stablehlo.broadcast_in_dim %v4418, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4420 = stablehlo.divide %v4419, %v4416 : tensor<32x480x14x14xf32>
    %v4421 = stablehlo.subtract %v4414, %v4420 : tensor<32x480x14x14xf32>
    %v4422 = stablehlo.multiply %v4421, %v4421 : tensor<32x480x14x14xf32>
    %v4423 = stablehlo.reduce(%v4422 init: %v4415) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4424 = stablehlo.broadcast_in_dim %v4423, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4425 = stablehlo.divide %v4424, %v4416 : tensor<32x480x14x14xf32>
    %v4426 = stablehlo.add %v4425, %v4417 : tensor<32x480x14x14xf32>
    %v4427 = stablehlo.rsqrt %v4426 : tensor<32x480x14x14xf32>
    %v4428 = stablehlo.multiply %v4421, %v4427 : tensor<32x480x14x14xf32>
    %v4429 = stablehlo.broadcast_in_dim %b8dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4430 = stablehlo.reshape %v4413 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4431 = stablehlo.multiply %v4429, %v4430 : tensor<32x480x14x14xf32>
    %v4432 = stablehlo.reduce(%v4431 init: %v4415) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4433 = stablehlo.broadcast_in_dim %v4432, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4434 = stablehlo.multiply %v4428, %v4431 : tensor<32x480x14x14xf32>
    %v4435 = stablehlo.reduce(%v4434 init: %v4415) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4436 = stablehlo.broadcast_in_dim %v4435, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4437 = stablehlo.multiply %v4431, %v4416 : tensor<32x480x14x14xf32>
    %v4438 = stablehlo.subtract %v4437, %v4433 : tensor<32x480x14x14xf32>
    %v4439 = stablehlo.multiply %v4428, %v4436 : tensor<32x480x14x14xf32>
    %v4440 = stablehlo.subtract %v4438, %v4439 : tensor<32x480x14x14xf32>
    %v4441 = stablehlo.divide %v4427, %v4416 : tensor<32x480x14x14xf32>
    %v4442 = stablehlo.multiply %v4441, %v4440 : tensor<32x480x14x14xf32>
    %v4443 = stablehlo.reshape %v4442 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4444 = stablehlo.reshape %v4443 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4445 = stablehlo.reverse %b8dW, dims = [2, 3] : tensor<480x1x3x3xf32>
    %v4446 = stablehlo.convolution(%v4444, %v4445)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %v4447 = stablehlo.reshape %v4446 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4448 = stablehlo.reshape %v745 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4449 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4450 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4451 = stablehlo.reduce(%v4448 init: %v4449) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4452 = stablehlo.broadcast_in_dim %v4451, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4453 = stablehlo.divide %v4452, %v4450 : tensor<32x480x14x14xf32>
    %v4454 = stablehlo.subtract %v4448, %v4453 : tensor<32x480x14x14xf32>
    %v4455 = stablehlo.multiply %v4454, %v4454 : tensor<32x480x14x14xf32>
    %v4456 = stablehlo.reduce(%v4455 init: %v4449) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4457 = stablehlo.broadcast_in_dim %v4456, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4458 = stablehlo.divide %v4457, %v4450 : tensor<32x480x14x14xf32>
    %v4459 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4460 = stablehlo.add %v4458, %v4459 : tensor<32x480x14x14xf32>
    %v4461 = stablehlo.rsqrt %v4460 : tensor<32x480x14x14xf32>
    %v4462 = stablehlo.multiply %v4454, %v4461 : tensor<32x480x14x14xf32>
    %v4463 = stablehlo.reshape %v4413 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4464 = stablehlo.multiply %v4463, %v4462 : tensor<32x480x14x14xf32>
    %v4465 = stablehlo.reduce(%v4464 init: %v4449) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4466 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4467 = stablehlo.multiply %v4465, %v4466 : tensor<480xf32>
    %v4468 = stablehlo.subtract %b8dg, %v4467 : tensor<480xf32>
    %v4469 = stablehlo.reshape %v4413 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4470 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4471 = stablehlo.reduce(%v4469 init: %v4470) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4472 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4473 = stablehlo.multiply %v4471, %v4472 : tensor<480xf32>
    %v4474 = stablehlo.subtract %b8dbt, %v4473 : tensor<480xf32>
    %v4475 = stablehlo.reshape %v740 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4476 = stablehlo.reshape %v4443 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4477 = stablehlo.transpose %v4475, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4478 = stablehlo.transpose %v4476, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4479 = stablehlo.convolution(%v4477, %v4478)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 480 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<1x480x3x3xf32>
    %v4480 = stablehlo.reshape %v4479 : (tensor<1x480x3x3xf32>) -> tensor<480x1x3x3xf32>
    %v4481 = stablehlo.constant dense<0.05> : tensor<480x1x3x3xf32>
    %v4482 = stablehlo.multiply %v4480, %v4481 : tensor<480x1x3x3xf32>
    %v4483 = stablehlo.subtract %b8dW, %v4482 : tensor<480x1x3x3xf32>
    %v4484 = stablehlo.reshape %v4447 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4485 = stablehlo.reshape %v736 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4486 = stablehlo.logistic %v4485 : tensor<32x480x14x14xf32>
    %v4487 = stablehlo.constant dense<1.0> : tensor<32x480x14x14xf32>
    %v4488 = stablehlo.subtract %v4487, %v4486 : tensor<32x480x14x14xf32>
    %v4489 = stablehlo.multiply %v4485, %v4488 : tensor<32x480x14x14xf32>
    %v4490 = stablehlo.add %v4487, %v4489 : tensor<32x480x14x14xf32>
    %v4491 = stablehlo.multiply %v4486, %v4490 : tensor<32x480x14x14xf32>
    %v4492 = stablehlo.multiply %v4484, %v4491 : tensor<32x480x14x14xf32>
    %v4493 = stablehlo.reshape %v4492 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4494 = stablehlo.reshape %v716 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4495 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4496 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4497 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4498 = stablehlo.reduce(%v4494 init: %v4495) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4499 = stablehlo.broadcast_in_dim %v4498, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4500 = stablehlo.divide %v4499, %v4496 : tensor<32x480x14x14xf32>
    %v4501 = stablehlo.subtract %v4494, %v4500 : tensor<32x480x14x14xf32>
    %v4502 = stablehlo.multiply %v4501, %v4501 : tensor<32x480x14x14xf32>
    %v4503 = stablehlo.reduce(%v4502 init: %v4495) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4504 = stablehlo.broadcast_in_dim %v4503, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4505 = stablehlo.divide %v4504, %v4496 : tensor<32x480x14x14xf32>
    %v4506 = stablehlo.add %v4505, %v4497 : tensor<32x480x14x14xf32>
    %v4507 = stablehlo.rsqrt %v4506 : tensor<32x480x14x14xf32>
    %v4508 = stablehlo.multiply %v4501, %v4507 : tensor<32x480x14x14xf32>
    %v4509 = stablehlo.broadcast_in_dim %b8eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4510 = stablehlo.reshape %v4493 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4511 = stablehlo.multiply %v4509, %v4510 : tensor<32x480x14x14xf32>
    %v4512 = stablehlo.reduce(%v4511 init: %v4495) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4513 = stablehlo.broadcast_in_dim %v4512, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4514 = stablehlo.multiply %v4508, %v4511 : tensor<32x480x14x14xf32>
    %v4515 = stablehlo.reduce(%v4514 init: %v4495) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4516 = stablehlo.broadcast_in_dim %v4515, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4517 = stablehlo.multiply %v4511, %v4496 : tensor<32x480x14x14xf32>
    %v4518 = stablehlo.subtract %v4517, %v4513 : tensor<32x480x14x14xf32>
    %v4519 = stablehlo.multiply %v4508, %v4516 : tensor<32x480x14x14xf32>
    %v4520 = stablehlo.subtract %v4518, %v4519 : tensor<32x480x14x14xf32>
    %v4521 = stablehlo.divide %v4507, %v4496 : tensor<32x480x14x14xf32>
    %v4522 = stablehlo.multiply %v4521, %v4520 : tensor<32x480x14x14xf32>
    %v4523 = stablehlo.reshape %v4522 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4524 = stablehlo.reshape %v4523 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4525 = stablehlo.reverse %b8eW, dims = [2, 3] : tensor<480x80x1x1xf32>
    %v4526 = stablehlo.transpose %v4525, dims = [1, 0, 2, 3] : (tensor<480x80x1x1xf32>) -> tensor<80x480x1x1xf32>
    %v4527 = stablehlo.convolution(%v4524, %v4526)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v4528 = stablehlo.reshape %v4527 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v4529 = stablehlo.reshape %v716 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4530 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4531 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4532 = stablehlo.reduce(%v4529 init: %v4530) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4533 = stablehlo.broadcast_in_dim %v4532, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4534 = stablehlo.divide %v4533, %v4531 : tensor<32x480x14x14xf32>
    %v4535 = stablehlo.subtract %v4529, %v4534 : tensor<32x480x14x14xf32>
    %v4536 = stablehlo.multiply %v4535, %v4535 : tensor<32x480x14x14xf32>
    %v4537 = stablehlo.reduce(%v4536 init: %v4530) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4538 = stablehlo.broadcast_in_dim %v4537, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4539 = stablehlo.divide %v4538, %v4531 : tensor<32x480x14x14xf32>
    %v4540 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4541 = stablehlo.add %v4539, %v4540 : tensor<32x480x14x14xf32>
    %v4542 = stablehlo.rsqrt %v4541 : tensor<32x480x14x14xf32>
    %v4543 = stablehlo.multiply %v4535, %v4542 : tensor<32x480x14x14xf32>
    %v4544 = stablehlo.reshape %v4493 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4545 = stablehlo.multiply %v4544, %v4543 : tensor<32x480x14x14xf32>
    %v4546 = stablehlo.reduce(%v4545 init: %v4530) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4547 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4548 = stablehlo.multiply %v4546, %v4547 : tensor<480xf32>
    %v4549 = stablehlo.subtract %b8eg, %v4548 : tensor<480xf32>
    %v4550 = stablehlo.reshape %v4493 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4551 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4552 = stablehlo.reduce(%v4550 init: %v4551) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4553 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4554 = stablehlo.multiply %v4552, %v4553 : tensor<480xf32>
    %v4555 = stablehlo.subtract %b8ebt, %v4554 : tensor<480xf32>
    %v4556 = stablehlo.reshape %v711 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4557 = stablehlo.reshape %v4523 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4558 = stablehlo.transpose %v4556, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %v4559 = stablehlo.transpose %v4557, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4560 = stablehlo.convolution(%v4558, %v4559)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<80x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<80x480x1x1xf32>
    %v4561 = stablehlo.transpose %v4560, dims = [1, 0, 2, 3] : (tensor<80x480x1x1xf32>) -> tensor<480x80x1x1xf32>
    %v4562 = stablehlo.constant dense<0.05> : tensor<480x80x1x1xf32>
    %v4563 = stablehlo.multiply %v4561, %v4562 : tensor<480x80x1x1xf32>
    %v4564 = stablehlo.subtract %b8eW, %v4563 : tensor<480x80x1x1xf32>
    %v4565 = stablehlo.reshape %v4528 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4566 = stablehlo.reshape %v4221 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4567 = stablehlo.add %v4565, %v4566 : tensor<32x80x14x14xf32>
    %v4568 = stablehlo.reshape %v4567 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v4569 = stablehlo.reshape %v687 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4570 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4571 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v4572 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v4573 = stablehlo.reduce(%v4569 init: %v4570) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4574 = stablehlo.broadcast_in_dim %v4573, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4575 = stablehlo.divide %v4574, %v4571 : tensor<32x80x14x14xf32>
    %v4576 = stablehlo.subtract %v4569, %v4575 : tensor<32x80x14x14xf32>
    %v4577 = stablehlo.multiply %v4576, %v4576 : tensor<32x80x14x14xf32>
    %v4578 = stablehlo.reduce(%v4577 init: %v4570) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4579 = stablehlo.broadcast_in_dim %v4578, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4580 = stablehlo.divide %v4579, %v4571 : tensor<32x80x14x14xf32>
    %v4581 = stablehlo.add %v4580, %v4572 : tensor<32x80x14x14xf32>
    %v4582 = stablehlo.rsqrt %v4581 : tensor<32x80x14x14xf32>
    %v4583 = stablehlo.multiply %v4576, %v4582 : tensor<32x80x14x14xf32>
    %v4584 = stablehlo.broadcast_in_dim %b7pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4585 = stablehlo.reshape %v4568 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4586 = stablehlo.multiply %v4584, %v4585 : tensor<32x80x14x14xf32>
    %v4587 = stablehlo.reduce(%v4586 init: %v4570) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4588 = stablehlo.broadcast_in_dim %v4587, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4589 = stablehlo.multiply %v4583, %v4586 : tensor<32x80x14x14xf32>
    %v4590 = stablehlo.reduce(%v4589 init: %v4570) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4591 = stablehlo.broadcast_in_dim %v4590, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4592 = stablehlo.multiply %v4586, %v4571 : tensor<32x80x14x14xf32>
    %v4593 = stablehlo.subtract %v4592, %v4588 : tensor<32x80x14x14xf32>
    %v4594 = stablehlo.multiply %v4583, %v4591 : tensor<32x80x14x14xf32>
    %v4595 = stablehlo.subtract %v4593, %v4594 : tensor<32x80x14x14xf32>
    %v4596 = stablehlo.divide %v4582, %v4571 : tensor<32x80x14x14xf32>
    %v4597 = stablehlo.multiply %v4596, %v4595 : tensor<32x80x14x14xf32>
    %v4598 = stablehlo.reshape %v4597 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v4599 = stablehlo.reshape %v4598 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4600 = stablehlo.reverse %b7pW, dims = [2, 3] : tensor<80x480x1x1xf32>
    %v4601 = stablehlo.transpose %v4600, dims = [1, 0, 2, 3] : (tensor<80x480x1x1xf32>) -> tensor<480x80x1x1xf32>
    %v4602 = stablehlo.convolution(%v4599, %v4601)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v4603 = stablehlo.reshape %v4602 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4604 = stablehlo.reshape %v687 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4605 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4606 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v4607 = stablehlo.reduce(%v4604 init: %v4605) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4608 = stablehlo.broadcast_in_dim %v4607, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4609 = stablehlo.divide %v4608, %v4606 : tensor<32x80x14x14xf32>
    %v4610 = stablehlo.subtract %v4604, %v4609 : tensor<32x80x14x14xf32>
    %v4611 = stablehlo.multiply %v4610, %v4610 : tensor<32x80x14x14xf32>
    %v4612 = stablehlo.reduce(%v4611 init: %v4605) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4613 = stablehlo.broadcast_in_dim %v4612, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4614 = stablehlo.divide %v4613, %v4606 : tensor<32x80x14x14xf32>
    %v4615 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v4616 = stablehlo.add %v4614, %v4615 : tensor<32x80x14x14xf32>
    %v4617 = stablehlo.rsqrt %v4616 : tensor<32x80x14x14xf32>
    %v4618 = stablehlo.multiply %v4610, %v4617 : tensor<32x80x14x14xf32>
    %v4619 = stablehlo.reshape %v4568 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4620 = stablehlo.multiply %v4619, %v4618 : tensor<32x80x14x14xf32>
    %v4621 = stablehlo.reduce(%v4620 init: %v4605) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4622 = stablehlo.constant dense<0.05> : tensor<80xf32>
    %v4623 = stablehlo.multiply %v4621, %v4622 : tensor<80xf32>
    %v4624 = stablehlo.subtract %b7pg, %v4623 : tensor<80xf32>
    %v4625 = stablehlo.reshape %v4568 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4626 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4627 = stablehlo.reduce(%v4625 init: %v4626) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4628 = stablehlo.constant dense<0.05> : tensor<80xf32>
    %v4629 = stablehlo.multiply %v4627, %v4628 : tensor<80xf32>
    %v4630 = stablehlo.subtract %b7pbt, %v4629 : tensor<80xf32>
    %v4631 = stablehlo.reshape %v682 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4632 = stablehlo.reshape %v4598 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4633 = stablehlo.transpose %v4631, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4634 = stablehlo.transpose %v4632, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %v4635 = stablehlo.convolution(%v4633, %v4634)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<80x32x14x14xf32>) -> tensor<480x80x1x1xf32>
    %v4636 = stablehlo.transpose %v4635, dims = [1, 0, 2, 3] : (tensor<480x80x1x1xf32>) -> tensor<80x480x1x1xf32>
    %v4637 = stablehlo.constant dense<0.05> : tensor<80x480x1x1xf32>
    %v4638 = stablehlo.multiply %v4636, %v4637 : tensor<80x480x1x1xf32>
    %v4639 = stablehlo.subtract %b7pW, %v4638 : tensor<80x480x1x1xf32>
    %v4640 = stablehlo.reshape %v665 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4641 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4642 = stablehlo.reduce(%v4640 init: %v4641) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4643 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v4644 = stablehlo.divide %v4642, %v4643 : tensor<32x480xf32>
    %v4645 = stablehlo.dot_general %v4644, %b7zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v4646 = stablehlo.broadcast_in_dim %b7zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v4647 = stablehlo.add %v4645, %v4646 : tensor<32x20xf32>
    %v4648 = stablehlo.logistic %v4647 : tensor<32x20xf32>
    %v4649 = stablehlo.multiply %v4647, %v4648 : tensor<32x20xf32>
    %v4650 = stablehlo.dot_general %v4649, %b7zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v4651 = stablehlo.broadcast_in_dim %b7zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v4652 = stablehlo.add %v4650, %v4651 : tensor<32x480xf32>
    %v4653 = stablehlo.logistic %v4652 : tensor<32x480xf32>
    %v4654 = stablehlo.reshape %v4603 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4655 = stablehlo.broadcast_in_dim %v4653, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v4656 = stablehlo.multiply %v4655, %v4654 : tensor<32x480x14x14xf32>
    %v4657 = stablehlo.multiply %v4640, %v4654 : tensor<32x480x14x14xf32>
    %v4658 = stablehlo.reduce(%v4657 init: %v4641) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4659 = stablehlo.constant dense<1.0> : tensor<32x480xf32>
    %v4660 = stablehlo.subtract %v4659, %v4653 : tensor<32x480xf32>
    %v4661 = stablehlo.multiply %v4653, %v4660 : tensor<32x480xf32>
    %v4662 = stablehlo.multiply %v4658, %v4661 : tensor<32x480xf32>
    %v4663 = stablehlo.dot_general %v4662, %b7zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<20x480xf32>) -> tensor<32x20xf32>
    %v4664 = stablehlo.logistic %v4647 : tensor<32x20xf32>
    %v4665 = stablehlo.constant dense<1.0> : tensor<32x20xf32>
    %v4666 = stablehlo.subtract %v4665, %v4664 : tensor<32x20xf32>
    %v4667 = stablehlo.multiply %v4647, %v4666 : tensor<32x20xf32>
    %v4668 = stablehlo.add %v4665, %v4667 : tensor<32x20xf32>
    %v4669 = stablehlo.multiply %v4664, %v4668 : tensor<32x20xf32>
    %v4670 = stablehlo.multiply %v4663, %v4669 : tensor<32x20xf32>
    %v4671 = stablehlo.dot_general %v4670, %b7zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<480x20xf32>) -> tensor<32x480xf32>
    %v4672 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v4673 = stablehlo.divide %v4671, %v4672 : tensor<32x480xf32>
    %v4674 = stablehlo.broadcast_in_dim %v4673, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v4675 = stablehlo.add %v4656, %v4674 : tensor<32x480x14x14xf32>
    %v4676 = stablehlo.reshape %v4675 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4677 = stablehlo.reshape %v665 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4678 = stablehlo.reshape %v4603 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4679 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4680 = stablehlo.multiply %v4677, %v4678 : tensor<32x480x14x14xf32>
    %v4681 = stablehlo.reduce(%v4680 init: %v4679) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v4682 = stablehlo.logistic %v678 : tensor<32x480xf32>
    %v4683 = stablehlo.constant dense<1.0> : tensor<32x480xf32>
    %v4684 = stablehlo.subtract %v4683, %v4682 : tensor<32x480xf32>
    %v4685 = stablehlo.multiply %v4682, %v4684 : tensor<32x480xf32>
    %v4686 = stablehlo.multiply %v4681, %v4685 : tensor<32x480xf32>
    %v4687 = stablehlo.dot_general %v675, %v4686, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<32x480xf32>) -> tensor<20x480xf32>
    %v4688 = stablehlo.constant dense<0.05> : tensor<20x480xf32>
    %v4689 = stablehlo.multiply %v4687, %v4688 : tensor<20x480xf32>
    %v4690 = stablehlo.subtract %b7zW2, %v4689 : tensor<20x480xf32>
    %v4691 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4692 = stablehlo.reduce(%v4686 init: %v4691) applies stablehlo.add across dimensions = [0] : (tensor<32x480xf32>, tensor<f32>) -> tensor<480xf32>
    %v4693 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4694 = stablehlo.multiply %v4692, %v4693 : tensor<480xf32>
    %v4695 = stablehlo.subtract %b7zb2, %v4694 : tensor<480xf32>
    %v4696 = stablehlo.reshape %v4686 : (tensor<32x480xf32>) -> tensor<32x1x480xf32>
    %v4697 = stablehlo.dot_general %v4696, %b7zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x480xf32>, tensor<20x480xf32>) -> tensor<32x1x20xf32>
    %v4698 = stablehlo.reshape %v4697 : (tensor<32x1x20xf32>) -> tensor<32x20xf32>
    %v4699 = stablehlo.logistic %v673 : tensor<32x20xf32>
    %v4700 = stablehlo.constant dense<1.0> : tensor<32x20xf32>
    %v4701 = stablehlo.subtract %v4700, %v4699 : tensor<32x20xf32>
    %v4702 = stablehlo.multiply %v673, %v4701 : tensor<32x20xf32>
    %v4703 = stablehlo.add %v4700, %v4702 : tensor<32x20xf32>
    %v4704 = stablehlo.multiply %v4699, %v4703 : tensor<32x20xf32>
    %v4705 = stablehlo.multiply %v4698, %v4704 : tensor<32x20xf32>
    %v4706 = stablehlo.dot_general %v670, %v4705, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<32x20xf32>) -> tensor<480x20xf32>
    %v4707 = stablehlo.constant dense<0.05> : tensor<480x20xf32>
    %v4708 = stablehlo.multiply %v4706, %v4707 : tensor<480x20xf32>
    %v4709 = stablehlo.subtract %b7zW1, %v4708 : tensor<480x20xf32>
    %v4710 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4711 = stablehlo.reduce(%v4705 init: %v4710) applies stablehlo.add across dimensions = [0] : (tensor<32x20xf32>, tensor<f32>) -> tensor<20xf32>
    %v4712 = stablehlo.constant dense<0.05> : tensor<20xf32>
    %v4713 = stablehlo.multiply %v4711, %v4712 : tensor<20xf32>
    %v4714 = stablehlo.subtract %b7zb1, %v4713 : tensor<20xf32>
    %v4715 = stablehlo.reshape %v4676 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4716 = stablehlo.reshape %v661 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4717 = stablehlo.logistic %v4716 : tensor<32x480x14x14xf32>
    %v4718 = stablehlo.constant dense<1.0> : tensor<32x480x14x14xf32>
    %v4719 = stablehlo.subtract %v4718, %v4717 : tensor<32x480x14x14xf32>
    %v4720 = stablehlo.multiply %v4716, %v4719 : tensor<32x480x14x14xf32>
    %v4721 = stablehlo.add %v4718, %v4720 : tensor<32x480x14x14xf32>
    %v4722 = stablehlo.multiply %v4717, %v4721 : tensor<32x480x14x14xf32>
    %v4723 = stablehlo.multiply %v4715, %v4722 : tensor<32x480x14x14xf32>
    %v4724 = stablehlo.reshape %v4723 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4725 = stablehlo.reshape %v641 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4726 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4727 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4728 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4729 = stablehlo.reduce(%v4725 init: %v4726) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4730 = stablehlo.broadcast_in_dim %v4729, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4731 = stablehlo.divide %v4730, %v4727 : tensor<32x480x14x14xf32>
    %v4732 = stablehlo.subtract %v4725, %v4731 : tensor<32x480x14x14xf32>
    %v4733 = stablehlo.multiply %v4732, %v4732 : tensor<32x480x14x14xf32>
    %v4734 = stablehlo.reduce(%v4733 init: %v4726) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4735 = stablehlo.broadcast_in_dim %v4734, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4736 = stablehlo.divide %v4735, %v4727 : tensor<32x480x14x14xf32>
    %v4737 = stablehlo.add %v4736, %v4728 : tensor<32x480x14x14xf32>
    %v4738 = stablehlo.rsqrt %v4737 : tensor<32x480x14x14xf32>
    %v4739 = stablehlo.multiply %v4732, %v4738 : tensor<32x480x14x14xf32>
    %v4740 = stablehlo.broadcast_in_dim %b7dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4741 = stablehlo.reshape %v4724 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4742 = stablehlo.multiply %v4740, %v4741 : tensor<32x480x14x14xf32>
    %v4743 = stablehlo.reduce(%v4742 init: %v4726) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4744 = stablehlo.broadcast_in_dim %v4743, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4745 = stablehlo.multiply %v4739, %v4742 : tensor<32x480x14x14xf32>
    %v4746 = stablehlo.reduce(%v4745 init: %v4726) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4747 = stablehlo.broadcast_in_dim %v4746, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4748 = stablehlo.multiply %v4742, %v4727 : tensor<32x480x14x14xf32>
    %v4749 = stablehlo.subtract %v4748, %v4744 : tensor<32x480x14x14xf32>
    %v4750 = stablehlo.multiply %v4739, %v4747 : tensor<32x480x14x14xf32>
    %v4751 = stablehlo.subtract %v4749, %v4750 : tensor<32x480x14x14xf32>
    %v4752 = stablehlo.divide %v4738, %v4727 : tensor<32x480x14x14xf32>
    %v4753 = stablehlo.multiply %v4752, %v4751 : tensor<32x480x14x14xf32>
    %v4754 = stablehlo.reshape %v4753 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4755 = stablehlo.reshape %v4754 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4756 = stablehlo.reverse %b7dW, dims = [2, 3] : tensor<480x1x3x3xf32>
    %v4757 = stablehlo.convolution(%v4755, %v4756)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %v4758 = stablehlo.reshape %v4757 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4759 = stablehlo.reshape %v641 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4760 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4761 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4762 = stablehlo.reduce(%v4759 init: %v4760) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4763 = stablehlo.broadcast_in_dim %v4762, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4764 = stablehlo.divide %v4763, %v4761 : tensor<32x480x14x14xf32>
    %v4765 = stablehlo.subtract %v4759, %v4764 : tensor<32x480x14x14xf32>
    %v4766 = stablehlo.multiply %v4765, %v4765 : tensor<32x480x14x14xf32>
    %v4767 = stablehlo.reduce(%v4766 init: %v4760) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4768 = stablehlo.broadcast_in_dim %v4767, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4769 = stablehlo.divide %v4768, %v4761 : tensor<32x480x14x14xf32>
    %v4770 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4771 = stablehlo.add %v4769, %v4770 : tensor<32x480x14x14xf32>
    %v4772 = stablehlo.rsqrt %v4771 : tensor<32x480x14x14xf32>
    %v4773 = stablehlo.multiply %v4765, %v4772 : tensor<32x480x14x14xf32>
    %v4774 = stablehlo.reshape %v4724 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4775 = stablehlo.multiply %v4774, %v4773 : tensor<32x480x14x14xf32>
    %v4776 = stablehlo.reduce(%v4775 init: %v4760) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4777 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4778 = stablehlo.multiply %v4776, %v4777 : tensor<480xf32>
    %v4779 = stablehlo.subtract %b7dg, %v4778 : tensor<480xf32>
    %v4780 = stablehlo.reshape %v4724 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4781 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4782 = stablehlo.reduce(%v4780 init: %v4781) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4783 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4784 = stablehlo.multiply %v4782, %v4783 : tensor<480xf32>
    %v4785 = stablehlo.subtract %b7dbt, %v4784 : tensor<480xf32>
    %v4786 = stablehlo.reshape %v636 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4787 = stablehlo.reshape %v4754 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4788 = stablehlo.transpose %v4786, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4789 = stablehlo.transpose %v4787, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4790 = stablehlo.convolution(%v4788, %v4789)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 480 : i64, feature_group_count = 1 : i64} : (tensor<480x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<1x480x3x3xf32>
    %v4791 = stablehlo.reshape %v4790 : (tensor<1x480x3x3xf32>) -> tensor<480x1x3x3xf32>
    %v4792 = stablehlo.constant dense<0.05> : tensor<480x1x3x3xf32>
    %v4793 = stablehlo.multiply %v4791, %v4792 : tensor<480x1x3x3xf32>
    %v4794 = stablehlo.subtract %b7dW, %v4793 : tensor<480x1x3x3xf32>
    %v4795 = stablehlo.reshape %v4758 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4796 = stablehlo.reshape %v632 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4797 = stablehlo.logistic %v4796 : tensor<32x480x14x14xf32>
    %v4798 = stablehlo.constant dense<1.0> : tensor<32x480x14x14xf32>
    %v4799 = stablehlo.subtract %v4798, %v4797 : tensor<32x480x14x14xf32>
    %v4800 = stablehlo.multiply %v4796, %v4799 : tensor<32x480x14x14xf32>
    %v4801 = stablehlo.add %v4798, %v4800 : tensor<32x480x14x14xf32>
    %v4802 = stablehlo.multiply %v4797, %v4801 : tensor<32x480x14x14xf32>
    %v4803 = stablehlo.multiply %v4795, %v4802 : tensor<32x480x14x14xf32>
    %v4804 = stablehlo.reshape %v4803 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4805 = stablehlo.reshape %v612 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4806 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4807 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4808 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4809 = stablehlo.reduce(%v4805 init: %v4806) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4810 = stablehlo.broadcast_in_dim %v4809, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4811 = stablehlo.divide %v4810, %v4807 : tensor<32x480x14x14xf32>
    %v4812 = stablehlo.subtract %v4805, %v4811 : tensor<32x480x14x14xf32>
    %v4813 = stablehlo.multiply %v4812, %v4812 : tensor<32x480x14x14xf32>
    %v4814 = stablehlo.reduce(%v4813 init: %v4806) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4815 = stablehlo.broadcast_in_dim %v4814, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4816 = stablehlo.divide %v4815, %v4807 : tensor<32x480x14x14xf32>
    %v4817 = stablehlo.add %v4816, %v4808 : tensor<32x480x14x14xf32>
    %v4818 = stablehlo.rsqrt %v4817 : tensor<32x480x14x14xf32>
    %v4819 = stablehlo.multiply %v4812, %v4818 : tensor<32x480x14x14xf32>
    %v4820 = stablehlo.broadcast_in_dim %b7eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4821 = stablehlo.reshape %v4804 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4822 = stablehlo.multiply %v4820, %v4821 : tensor<32x480x14x14xf32>
    %v4823 = stablehlo.reduce(%v4822 init: %v4806) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4824 = stablehlo.broadcast_in_dim %v4823, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4825 = stablehlo.multiply %v4819, %v4822 : tensor<32x480x14x14xf32>
    %v4826 = stablehlo.reduce(%v4825 init: %v4806) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4827 = stablehlo.broadcast_in_dim %v4826, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4828 = stablehlo.multiply %v4822, %v4807 : tensor<32x480x14x14xf32>
    %v4829 = stablehlo.subtract %v4828, %v4824 : tensor<32x480x14x14xf32>
    %v4830 = stablehlo.multiply %v4819, %v4827 : tensor<32x480x14x14xf32>
    %v4831 = stablehlo.subtract %v4829, %v4830 : tensor<32x480x14x14xf32>
    %v4832 = stablehlo.divide %v4818, %v4807 : tensor<32x480x14x14xf32>
    %v4833 = stablehlo.multiply %v4832, %v4831 : tensor<32x480x14x14xf32>
    %v4834 = stablehlo.reshape %v4833 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v4835 = stablehlo.reshape %v4834 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4836 = stablehlo.reverse %b7eW, dims = [2, 3] : tensor<480x80x1x1xf32>
    %v4837 = stablehlo.transpose %v4836, dims = [1, 0, 2, 3] : (tensor<480x80x1x1xf32>) -> tensor<80x480x1x1xf32>
    %v4838 = stablehlo.convolution(%v4835, %v4837)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v4839 = stablehlo.reshape %v4838 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v4840 = stablehlo.reshape %v612 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4841 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4842 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v4843 = stablehlo.reduce(%v4840 init: %v4841) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4844 = stablehlo.broadcast_in_dim %v4843, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4845 = stablehlo.divide %v4844, %v4842 : tensor<32x480x14x14xf32>
    %v4846 = stablehlo.subtract %v4840, %v4845 : tensor<32x480x14x14xf32>
    %v4847 = stablehlo.multiply %v4846, %v4846 : tensor<32x480x14x14xf32>
    %v4848 = stablehlo.reduce(%v4847 init: %v4841) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4849 = stablehlo.broadcast_in_dim %v4848, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v4850 = stablehlo.divide %v4849, %v4842 : tensor<32x480x14x14xf32>
    %v4851 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v4852 = stablehlo.add %v4850, %v4851 : tensor<32x480x14x14xf32>
    %v4853 = stablehlo.rsqrt %v4852 : tensor<32x480x14x14xf32>
    %v4854 = stablehlo.multiply %v4846, %v4853 : tensor<32x480x14x14xf32>
    %v4855 = stablehlo.reshape %v4804 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4856 = stablehlo.multiply %v4855, %v4854 : tensor<32x480x14x14xf32>
    %v4857 = stablehlo.reduce(%v4856 init: %v4841) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4858 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4859 = stablehlo.multiply %v4857, %v4858 : tensor<480xf32>
    %v4860 = stablehlo.subtract %b7eg, %v4859 : tensor<480xf32>
    %v4861 = stablehlo.reshape %v4804 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4862 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4863 = stablehlo.reduce(%v4861 init: %v4862) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v4864 = stablehlo.constant dense<0.05> : tensor<480xf32>
    %v4865 = stablehlo.multiply %v4863, %v4864 : tensor<480xf32>
    %v4866 = stablehlo.subtract %b7ebt, %v4865 : tensor<480xf32>
    %v4867 = stablehlo.reshape %v607 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4868 = stablehlo.reshape %v4834 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v4869 = stablehlo.transpose %v4867, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %v4870 = stablehlo.transpose %v4868, dims = [1, 0, 2, 3] : (tensor<32x480x14x14xf32>) -> tensor<480x32x14x14xf32>
    %v4871 = stablehlo.convolution(%v4869, %v4870)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<80x32x14x14xf32>, tensor<480x32x14x14xf32>) -> tensor<80x480x1x1xf32>
    %v4872 = stablehlo.transpose %v4871, dims = [1, 0, 2, 3] : (tensor<80x480x1x1xf32>) -> tensor<480x80x1x1xf32>
    %v4873 = stablehlo.constant dense<0.05> : tensor<480x80x1x1xf32>
    %v4874 = stablehlo.multiply %v4872, %v4873 : tensor<480x80x1x1xf32>
    %v4875 = stablehlo.subtract %b7eW, %v4874 : tensor<480x80x1x1xf32>
    %v4876 = stablehlo.reshape %v4839 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4877 = stablehlo.reshape %v4568 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4878 = stablehlo.add %v4876, %v4877 : tensor<32x80x14x14xf32>
    %v4879 = stablehlo.reshape %v4878 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v4880 = stablehlo.reshape %v587 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4881 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4882 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v4883 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v4884 = stablehlo.reduce(%v4880 init: %v4881) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4885 = stablehlo.broadcast_in_dim %v4884, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4886 = stablehlo.divide %v4885, %v4882 : tensor<32x80x14x14xf32>
    %v4887 = stablehlo.subtract %v4880, %v4886 : tensor<32x80x14x14xf32>
    %v4888 = stablehlo.multiply %v4887, %v4887 : tensor<32x80x14x14xf32>
    %v4889 = stablehlo.reduce(%v4888 init: %v4881) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4890 = stablehlo.broadcast_in_dim %v4889, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4891 = stablehlo.divide %v4890, %v4882 : tensor<32x80x14x14xf32>
    %v4892 = stablehlo.add %v4891, %v4883 : tensor<32x80x14x14xf32>
    %v4893 = stablehlo.rsqrt %v4892 : tensor<32x80x14x14xf32>
    %v4894 = stablehlo.multiply %v4887, %v4893 : tensor<32x80x14x14xf32>
    %v4895 = stablehlo.broadcast_in_dim %b6pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4896 = stablehlo.reshape %v4879 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4897 = stablehlo.multiply %v4895, %v4896 : tensor<32x80x14x14xf32>
    %v4898 = stablehlo.reduce(%v4897 init: %v4881) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4899 = stablehlo.broadcast_in_dim %v4898, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4900 = stablehlo.multiply %v4894, %v4897 : tensor<32x80x14x14xf32>
    %v4901 = stablehlo.reduce(%v4900 init: %v4881) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4902 = stablehlo.broadcast_in_dim %v4901, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4903 = stablehlo.multiply %v4897, %v4882 : tensor<32x80x14x14xf32>
    %v4904 = stablehlo.subtract %v4903, %v4899 : tensor<32x80x14x14xf32>
    %v4905 = stablehlo.multiply %v4894, %v4902 : tensor<32x80x14x14xf32>
    %v4906 = stablehlo.subtract %v4904, %v4905 : tensor<32x80x14x14xf32>
    %v4907 = stablehlo.divide %v4893, %v4882 : tensor<32x80x14x14xf32>
    %v4908 = stablehlo.multiply %v4907, %v4906 : tensor<32x80x14x14xf32>
    %v4909 = stablehlo.reshape %v4908 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v4910 = stablehlo.reshape %v4909 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4911 = stablehlo.reverse %b6pW, dims = [2, 3] : tensor<80x240x1x1xf32>
    %v4912 = stablehlo.transpose %v4911, dims = [1, 0, 2, 3] : (tensor<80x240x1x1xf32>) -> tensor<240x80x1x1xf32>
    %v4913 = stablehlo.convolution(%v4910, %v4912)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<240x80x1x1xf32>) -> tensor<32x240x14x14xf32>
    %v4914 = stablehlo.reshape %v4913 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v4915 = stablehlo.reshape %v587 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4916 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4917 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v4918 = stablehlo.reduce(%v4915 init: %v4916) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4919 = stablehlo.broadcast_in_dim %v4918, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4920 = stablehlo.divide %v4919, %v4917 : tensor<32x80x14x14xf32>
    %v4921 = stablehlo.subtract %v4915, %v4920 : tensor<32x80x14x14xf32>
    %v4922 = stablehlo.multiply %v4921, %v4921 : tensor<32x80x14x14xf32>
    %v4923 = stablehlo.reduce(%v4922 init: %v4916) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4924 = stablehlo.broadcast_in_dim %v4923, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v4925 = stablehlo.divide %v4924, %v4917 : tensor<32x80x14x14xf32>
    %v4926 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v4927 = stablehlo.add %v4925, %v4926 : tensor<32x80x14x14xf32>
    %v4928 = stablehlo.rsqrt %v4927 : tensor<32x80x14x14xf32>
    %v4929 = stablehlo.multiply %v4921, %v4928 : tensor<32x80x14x14xf32>
    %v4930 = stablehlo.reshape %v4879 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4931 = stablehlo.multiply %v4930, %v4929 : tensor<32x80x14x14xf32>
    %v4932 = stablehlo.reduce(%v4931 init: %v4916) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4933 = stablehlo.constant dense<0.05> : tensor<80xf32>
    %v4934 = stablehlo.multiply %v4932, %v4933 : tensor<80xf32>
    %v4935 = stablehlo.subtract %b6pg, %v4934 : tensor<80xf32>
    %v4936 = stablehlo.reshape %v4879 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4937 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4938 = stablehlo.reduce(%v4936 init: %v4937) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v4939 = stablehlo.constant dense<0.05> : tensor<80xf32>
    %v4940 = stablehlo.multiply %v4938, %v4939 : tensor<80xf32>
    %v4941 = stablehlo.subtract %b6pbt, %v4940 : tensor<80xf32>
    %v4942 = stablehlo.reshape %v582 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v4943 = stablehlo.reshape %v4909 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v4944 = stablehlo.transpose %v4942, dims = [1, 0, 2, 3] : (tensor<32x240x14x14xf32>) -> tensor<240x32x14x14xf32>
    %v4945 = stablehlo.transpose %v4943, dims = [1, 0, 2, 3] : (tensor<32x80x14x14xf32>) -> tensor<80x32x14x14xf32>
    %v4946 = stablehlo.convolution(%v4944, %v4945)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<240x32x14x14xf32>, tensor<80x32x14x14xf32>) -> tensor<240x80x1x1xf32>
    %v4947 = stablehlo.transpose %v4946, dims = [1, 0, 2, 3] : (tensor<240x80x1x1xf32>) -> tensor<80x240x1x1xf32>
    %v4948 = stablehlo.constant dense<0.05> : tensor<80x240x1x1xf32>
    %v4949 = stablehlo.multiply %v4947, %v4948 : tensor<80x240x1x1xf32>
    %v4950 = stablehlo.subtract %b6pW, %v4949 : tensor<80x240x1x1xf32>
    %v4951 = stablehlo.reshape %v565 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v4952 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4953 = stablehlo.reduce(%v4951 init: %v4952) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v4954 = stablehlo.constant dense<196.0> : tensor<32x240xf32>
    %v4955 = stablehlo.divide %v4953, %v4954 : tensor<32x240xf32>
    %v4956 = stablehlo.dot_general %v4955, %b6zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v4957 = stablehlo.broadcast_in_dim %b6zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v4958 = stablehlo.add %v4956, %v4957 : tensor<32x10xf32>
    %v4959 = stablehlo.logistic %v4958 : tensor<32x10xf32>
    %v4960 = stablehlo.multiply %v4958, %v4959 : tensor<32x10xf32>
    %v4961 = stablehlo.dot_general %v4960, %b6zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v4962 = stablehlo.broadcast_in_dim %b6zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v4963 = stablehlo.add %v4961, %v4962 : tensor<32x240xf32>
    %v4964 = stablehlo.logistic %v4963 : tensor<32x240xf32>
    %v4965 = stablehlo.reshape %v4914 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v4966 = stablehlo.broadcast_in_dim %v4964, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x14x14xf32>
    %v4967 = stablehlo.multiply %v4966, %v4965 : tensor<32x240x14x14xf32>
    %v4968 = stablehlo.multiply %v4951, %v4965 : tensor<32x240x14x14xf32>
    %v4969 = stablehlo.reduce(%v4968 init: %v4952) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v4970 = stablehlo.constant dense<1.0> : tensor<32x240xf32>
    %v4971 = stablehlo.subtract %v4970, %v4964 : tensor<32x240xf32>
    %v4972 = stablehlo.multiply %v4964, %v4971 : tensor<32x240xf32>
    %v4973 = stablehlo.multiply %v4969, %v4972 : tensor<32x240xf32>
    %v4974 = stablehlo.dot_general %v4973, %b6zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<10x240xf32>) -> tensor<32x10xf32>
    %v4975 = stablehlo.logistic %v4958 : tensor<32x10xf32>
    %v4976 = stablehlo.constant dense<1.0> : tensor<32x10xf32>
    %v4977 = stablehlo.subtract %v4976, %v4975 : tensor<32x10xf32>
    %v4978 = stablehlo.multiply %v4958, %v4977 : tensor<32x10xf32>
    %v4979 = stablehlo.add %v4976, %v4978 : tensor<32x10xf32>
    %v4980 = stablehlo.multiply %v4975, %v4979 : tensor<32x10xf32>
    %v4981 = stablehlo.multiply %v4974, %v4980 : tensor<32x10xf32>
    %v4982 = stablehlo.dot_general %v4981, %b6zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<240x10xf32>) -> tensor<32x240xf32>
    %v4983 = stablehlo.constant dense<196.0> : tensor<32x240xf32>
    %v4984 = stablehlo.divide %v4982, %v4983 : tensor<32x240xf32>
    %v4985 = stablehlo.broadcast_in_dim %v4984, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x14x14xf32>
    %v4986 = stablehlo.add %v4967, %v4985 : tensor<32x240x14x14xf32>
    %v4987 = stablehlo.reshape %v4986 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v4988 = stablehlo.reshape %v565 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v4989 = stablehlo.reshape %v4914 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v4990 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4991 = stablehlo.multiply %v4988, %v4989 : tensor<32x240x14x14xf32>
    %v4992 = stablehlo.reduce(%v4991 init: %v4990) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v4993 = stablehlo.logistic %v578 : tensor<32x240xf32>
    %v4994 = stablehlo.constant dense<1.0> : tensor<32x240xf32>
    %v4995 = stablehlo.subtract %v4994, %v4993 : tensor<32x240xf32>
    %v4996 = stablehlo.multiply %v4993, %v4995 : tensor<32x240xf32>
    %v4997 = stablehlo.multiply %v4992, %v4996 : tensor<32x240xf32>
    %v4998 = stablehlo.dot_general %v575, %v4997, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<32x240xf32>) -> tensor<10x240xf32>
    %v4999 = stablehlo.constant dense<0.05> : tensor<10x240xf32>
    %v5000 = stablehlo.multiply %v4998, %v4999 : tensor<10x240xf32>
    %v5001 = stablehlo.subtract %b6zW2, %v5000 : tensor<10x240xf32>
    %v5002 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5003 = stablehlo.reduce(%v4997 init: %v5002) applies stablehlo.add across dimensions = [0] : (tensor<32x240xf32>, tensor<f32>) -> tensor<240xf32>
    %v5004 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5005 = stablehlo.multiply %v5003, %v5004 : tensor<240xf32>
    %v5006 = stablehlo.subtract %b6zb2, %v5005 : tensor<240xf32>
    %v5007 = stablehlo.reshape %v4997 : (tensor<32x240xf32>) -> tensor<32x1x240xf32>
    %v5008 = stablehlo.dot_general %v5007, %b6zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x240xf32>, tensor<10x240xf32>) -> tensor<32x1x10xf32>
    %v5009 = stablehlo.reshape %v5008 : (tensor<32x1x10xf32>) -> tensor<32x10xf32>
    %v5010 = stablehlo.logistic %v573 : tensor<32x10xf32>
    %v5011 = stablehlo.constant dense<1.0> : tensor<32x10xf32>
    %v5012 = stablehlo.subtract %v5011, %v5010 : tensor<32x10xf32>
    %v5013 = stablehlo.multiply %v573, %v5012 : tensor<32x10xf32>
    %v5014 = stablehlo.add %v5011, %v5013 : tensor<32x10xf32>
    %v5015 = stablehlo.multiply %v5010, %v5014 : tensor<32x10xf32>
    %v5016 = stablehlo.multiply %v5009, %v5015 : tensor<32x10xf32>
    %v5017 = stablehlo.dot_general %v570, %v5016, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<32x10xf32>) -> tensor<240x10xf32>
    %v5018 = stablehlo.constant dense<0.05> : tensor<240x10xf32>
    %v5019 = stablehlo.multiply %v5017, %v5018 : tensor<240x10xf32>
    %v5020 = stablehlo.subtract %b6zW1, %v5019 : tensor<240x10xf32>
    %v5021 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5022 = stablehlo.reduce(%v5016 init: %v5021) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v5023 = stablehlo.constant dense<0.05> : tensor<10xf32>
    %v5024 = stablehlo.multiply %v5022, %v5023 : tensor<10xf32>
    %v5025 = stablehlo.subtract %b6zb1, %v5024 : tensor<10xf32>
    %v5026 = stablehlo.reshape %v4987 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5027 = stablehlo.reshape %v561 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5028 = stablehlo.logistic %v5027 : tensor<32x240x14x14xf32>
    %v5029 = stablehlo.constant dense<1.0> : tensor<32x240x14x14xf32>
    %v5030 = stablehlo.subtract %v5029, %v5028 : tensor<32x240x14x14xf32>
    %v5031 = stablehlo.multiply %v5027, %v5030 : tensor<32x240x14x14xf32>
    %v5032 = stablehlo.add %v5029, %v5031 : tensor<32x240x14x14xf32>
    %v5033 = stablehlo.multiply %v5028, %v5032 : tensor<32x240x14x14xf32>
    %v5034 = stablehlo.multiply %v5026, %v5033 : tensor<32x240x14x14xf32>
    %v5035 = stablehlo.reshape %v5034 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v5036 = stablehlo.reshape %v541 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5037 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5038 = stablehlo.constant dense<6272.0> : tensor<32x240x14x14xf32>
    %v5039 = stablehlo.constant dense<1.0e-5> : tensor<32x240x14x14xf32>
    %v5040 = stablehlo.reduce(%v5036 init: %v5037) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5041 = stablehlo.broadcast_in_dim %v5040, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v5042 = stablehlo.divide %v5041, %v5038 : tensor<32x240x14x14xf32>
    %v5043 = stablehlo.subtract %v5036, %v5042 : tensor<32x240x14x14xf32>
    %v5044 = stablehlo.multiply %v5043, %v5043 : tensor<32x240x14x14xf32>
    %v5045 = stablehlo.reduce(%v5044 init: %v5037) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5046 = stablehlo.broadcast_in_dim %v5045, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v5047 = stablehlo.divide %v5046, %v5038 : tensor<32x240x14x14xf32>
    %v5048 = stablehlo.add %v5047, %v5039 : tensor<32x240x14x14xf32>
    %v5049 = stablehlo.rsqrt %v5048 : tensor<32x240x14x14xf32>
    %v5050 = stablehlo.multiply %v5043, %v5049 : tensor<32x240x14x14xf32>
    %v5051 = stablehlo.broadcast_in_dim %b6dg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v5052 = stablehlo.reshape %v5035 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5053 = stablehlo.multiply %v5051, %v5052 : tensor<32x240x14x14xf32>
    %v5054 = stablehlo.reduce(%v5053 init: %v5037) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5055 = stablehlo.broadcast_in_dim %v5054, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v5056 = stablehlo.multiply %v5050, %v5053 : tensor<32x240x14x14xf32>
    %v5057 = stablehlo.reduce(%v5056 init: %v5037) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5058 = stablehlo.broadcast_in_dim %v5057, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v5059 = stablehlo.multiply %v5053, %v5038 : tensor<32x240x14x14xf32>
    %v5060 = stablehlo.subtract %v5059, %v5055 : tensor<32x240x14x14xf32>
    %v5061 = stablehlo.multiply %v5050, %v5058 : tensor<32x240x14x14xf32>
    %v5062 = stablehlo.subtract %v5060, %v5061 : tensor<32x240x14x14xf32>
    %v5063 = stablehlo.divide %v5049, %v5038 : tensor<32x240x14x14xf32>
    %v5064 = stablehlo.multiply %v5063, %v5062 : tensor<32x240x14x14xf32>
    %v5065 = stablehlo.reshape %v5064 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v5066 = stablehlo.reshape %v5065 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5067 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5068 = stablehlo.pad %v5066, %v5067, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240x28x28xf32>
    %v5069 = stablehlo.reverse %b6dW, dims = [2, 3] : tensor<240x1x3x3xf32>
    %v5070 = stablehlo.convolution(%v5068, %v5069)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x3x3xf32>) -> tensor<32x240x28x28xf32>
    %v5071 = stablehlo.reshape %v5070 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5072 = stablehlo.reshape %v541 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5073 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5074 = stablehlo.constant dense<6272.0> : tensor<32x240x14x14xf32>
    %v5075 = stablehlo.reduce(%v5072 init: %v5073) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5076 = stablehlo.broadcast_in_dim %v5075, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v5077 = stablehlo.divide %v5076, %v5074 : tensor<32x240x14x14xf32>
    %v5078 = stablehlo.subtract %v5072, %v5077 : tensor<32x240x14x14xf32>
    %v5079 = stablehlo.multiply %v5078, %v5078 : tensor<32x240x14x14xf32>
    %v5080 = stablehlo.reduce(%v5079 init: %v5073) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5081 = stablehlo.broadcast_in_dim %v5080, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v5082 = stablehlo.divide %v5081, %v5074 : tensor<32x240x14x14xf32>
    %v5083 = stablehlo.constant dense<1.0e-5> : tensor<32x240x14x14xf32>
    %v5084 = stablehlo.add %v5082, %v5083 : tensor<32x240x14x14xf32>
    %v5085 = stablehlo.rsqrt %v5084 : tensor<32x240x14x14xf32>
    %v5086 = stablehlo.multiply %v5078, %v5085 : tensor<32x240x14x14xf32>
    %v5087 = stablehlo.reshape %v5035 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5088 = stablehlo.multiply %v5087, %v5086 : tensor<32x240x14x14xf32>
    %v5089 = stablehlo.reduce(%v5088 init: %v5073) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5090 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5091 = stablehlo.multiply %v5089, %v5090 : tensor<240xf32>
    %v5092 = stablehlo.subtract %b6dg, %v5091 : tensor<240xf32>
    %v5093 = stablehlo.reshape %v5035 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5094 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5095 = stablehlo.reduce(%v5093 init: %v5094) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v5096 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5097 = stablehlo.multiply %v5095, %v5096 : tensor<240xf32>
    %v5098 = stablehlo.subtract %b6dbt, %v5097 : tensor<240xf32>
    %v5099 = stablehlo.reshape %v536 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5100 = stablehlo.reshape %v5065 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v5101 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5102 = stablehlo.pad %v5100, %v5101, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240x28x28xf32>
    %v5103 = stablehlo.transpose %v5099, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %v5104 = stablehlo.transpose %v5102, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %v5105 = stablehlo.convolution(%v5103, %v5104)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 240 : i64, feature_group_count = 1 : i64} : (tensor<240x32x28x28xf32>, tensor<240x32x28x28xf32>) -> tensor<1x240x3x3xf32>
    %v5106 = stablehlo.reshape %v5105 : (tensor<1x240x3x3xf32>) -> tensor<240x1x3x3xf32>
    %v5107 = stablehlo.constant dense<0.05> : tensor<240x1x3x3xf32>
    %v5108 = stablehlo.multiply %v5106, %v5107 : tensor<240x1x3x3xf32>
    %v5109 = stablehlo.subtract %b6dW, %v5108 : tensor<240x1x3x3xf32>
    %v5110 = stablehlo.reshape %v5071 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5111 = stablehlo.reshape %v532 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5112 = stablehlo.logistic %v5111 : tensor<32x240x28x28xf32>
    %v5113 = stablehlo.constant dense<1.0> : tensor<32x240x28x28xf32>
    %v5114 = stablehlo.subtract %v5113, %v5112 : tensor<32x240x28x28xf32>
    %v5115 = stablehlo.multiply %v5111, %v5114 : tensor<32x240x28x28xf32>
    %v5116 = stablehlo.add %v5113, %v5115 : tensor<32x240x28x28xf32>
    %v5117 = stablehlo.multiply %v5112, %v5116 : tensor<32x240x28x28xf32>
    %v5118 = stablehlo.multiply %v5110, %v5117 : tensor<32x240x28x28xf32>
    %v5119 = stablehlo.reshape %v5118 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5120 = stablehlo.reshape %v512 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5121 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5122 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v5123 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v5124 = stablehlo.reduce(%v5120 init: %v5121) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5125 = stablehlo.broadcast_in_dim %v5124, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5126 = stablehlo.divide %v5125, %v5122 : tensor<32x240x28x28xf32>
    %v5127 = stablehlo.subtract %v5120, %v5126 : tensor<32x240x28x28xf32>
    %v5128 = stablehlo.multiply %v5127, %v5127 : tensor<32x240x28x28xf32>
    %v5129 = stablehlo.reduce(%v5128 init: %v5121) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5130 = stablehlo.broadcast_in_dim %v5129, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5131 = stablehlo.divide %v5130, %v5122 : tensor<32x240x28x28xf32>
    %v5132 = stablehlo.add %v5131, %v5123 : tensor<32x240x28x28xf32>
    %v5133 = stablehlo.rsqrt %v5132 : tensor<32x240x28x28xf32>
    %v5134 = stablehlo.multiply %v5127, %v5133 : tensor<32x240x28x28xf32>
    %v5135 = stablehlo.broadcast_in_dim %b6eg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5136 = stablehlo.reshape %v5119 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5137 = stablehlo.multiply %v5135, %v5136 : tensor<32x240x28x28xf32>
    %v5138 = stablehlo.reduce(%v5137 init: %v5121) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5139 = stablehlo.broadcast_in_dim %v5138, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5140 = stablehlo.multiply %v5134, %v5137 : tensor<32x240x28x28xf32>
    %v5141 = stablehlo.reduce(%v5140 init: %v5121) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5142 = stablehlo.broadcast_in_dim %v5141, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5143 = stablehlo.multiply %v5137, %v5122 : tensor<32x240x28x28xf32>
    %v5144 = stablehlo.subtract %v5143, %v5139 : tensor<32x240x28x28xf32>
    %v5145 = stablehlo.multiply %v5134, %v5142 : tensor<32x240x28x28xf32>
    %v5146 = stablehlo.subtract %v5144, %v5145 : tensor<32x240x28x28xf32>
    %v5147 = stablehlo.divide %v5133, %v5122 : tensor<32x240x28x28xf32>
    %v5148 = stablehlo.multiply %v5147, %v5146 : tensor<32x240x28x28xf32>
    %v5149 = stablehlo.reshape %v5148 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5150 = stablehlo.reshape %v5149 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5151 = stablehlo.reverse %b6eW, dims = [2, 3] : tensor<240x40x1x1xf32>
    %v5152 = stablehlo.transpose %v5151, dims = [1, 0, 2, 3] : (tensor<240x40x1x1xf32>) -> tensor<40x240x1x1xf32>
    %v5153 = stablehlo.convolution(%v5150, %v5152)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x28x28xf32>, tensor<40x240x1x1xf32>) -> tensor<32x40x28x28xf32>
    %v5154 = stablehlo.reshape %v5153 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v5155 = stablehlo.reshape %v512 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5156 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5157 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v5158 = stablehlo.reduce(%v5155 init: %v5156) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5159 = stablehlo.broadcast_in_dim %v5158, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5160 = stablehlo.divide %v5159, %v5157 : tensor<32x240x28x28xf32>
    %v5161 = stablehlo.subtract %v5155, %v5160 : tensor<32x240x28x28xf32>
    %v5162 = stablehlo.multiply %v5161, %v5161 : tensor<32x240x28x28xf32>
    %v5163 = stablehlo.reduce(%v5162 init: %v5156) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5164 = stablehlo.broadcast_in_dim %v5163, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5165 = stablehlo.divide %v5164, %v5157 : tensor<32x240x28x28xf32>
    %v5166 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v5167 = stablehlo.add %v5165, %v5166 : tensor<32x240x28x28xf32>
    %v5168 = stablehlo.rsqrt %v5167 : tensor<32x240x28x28xf32>
    %v5169 = stablehlo.multiply %v5161, %v5168 : tensor<32x240x28x28xf32>
    %v5170 = stablehlo.reshape %v5119 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5171 = stablehlo.multiply %v5170, %v5169 : tensor<32x240x28x28xf32>
    %v5172 = stablehlo.reduce(%v5171 init: %v5156) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5173 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5174 = stablehlo.multiply %v5172, %v5173 : tensor<240xf32>
    %v5175 = stablehlo.subtract %b6eg, %v5174 : tensor<240xf32>
    %v5176 = stablehlo.reshape %v5119 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5177 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5178 = stablehlo.reduce(%v5176 init: %v5177) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5179 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5180 = stablehlo.multiply %v5178, %v5179 : tensor<240xf32>
    %v5181 = stablehlo.subtract %b6ebt, %v5180 : tensor<240xf32>
    %v5182 = stablehlo.reshape %v507 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5183 = stablehlo.reshape %v5149 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5184 = stablehlo.transpose %v5182, dims = [1, 0, 2, 3] : (tensor<32x40x28x28xf32>) -> tensor<40x32x28x28xf32>
    %v5185 = stablehlo.transpose %v5183, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %v5186 = stablehlo.convolution(%v5184, %v5185)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<40x32x28x28xf32>, tensor<240x32x28x28xf32>) -> tensor<40x240x1x1xf32>
    %v5187 = stablehlo.transpose %v5186, dims = [1, 0, 2, 3] : (tensor<40x240x1x1xf32>) -> tensor<240x40x1x1xf32>
    %v5188 = stablehlo.constant dense<0.05> : tensor<240x40x1x1xf32>
    %v5189 = stablehlo.multiply %v5187, %v5188 : tensor<240x40x1x1xf32>
    %v5190 = stablehlo.subtract %b6eW, %v5189 : tensor<240x40x1x1xf32>
    %v5191 = stablehlo.reshape %v483 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5192 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5193 = stablehlo.constant dense<25088.0> : tensor<32x40x28x28xf32>
    %v5194 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v5195 = stablehlo.reduce(%v5191 init: %v5192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5196 = stablehlo.broadcast_in_dim %v5195, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5197 = stablehlo.divide %v5196, %v5193 : tensor<32x40x28x28xf32>
    %v5198 = stablehlo.subtract %v5191, %v5197 : tensor<32x40x28x28xf32>
    %v5199 = stablehlo.multiply %v5198, %v5198 : tensor<32x40x28x28xf32>
    %v5200 = stablehlo.reduce(%v5199 init: %v5192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5201 = stablehlo.broadcast_in_dim %v5200, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5202 = stablehlo.divide %v5201, %v5193 : tensor<32x40x28x28xf32>
    %v5203 = stablehlo.add %v5202, %v5194 : tensor<32x40x28x28xf32>
    %v5204 = stablehlo.rsqrt %v5203 : tensor<32x40x28x28xf32>
    %v5205 = stablehlo.multiply %v5198, %v5204 : tensor<32x40x28x28xf32>
    %v5206 = stablehlo.broadcast_in_dim %b5pg, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5207 = stablehlo.reshape %v5154 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5208 = stablehlo.multiply %v5206, %v5207 : tensor<32x40x28x28xf32>
    %v5209 = stablehlo.reduce(%v5208 init: %v5192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5210 = stablehlo.broadcast_in_dim %v5209, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5211 = stablehlo.multiply %v5205, %v5208 : tensor<32x40x28x28xf32>
    %v5212 = stablehlo.reduce(%v5211 init: %v5192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5213 = stablehlo.broadcast_in_dim %v5212, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5214 = stablehlo.multiply %v5208, %v5193 : tensor<32x40x28x28xf32>
    %v5215 = stablehlo.subtract %v5214, %v5210 : tensor<32x40x28x28xf32>
    %v5216 = stablehlo.multiply %v5205, %v5213 : tensor<32x40x28x28xf32>
    %v5217 = stablehlo.subtract %v5215, %v5216 : tensor<32x40x28x28xf32>
    %v5218 = stablehlo.divide %v5204, %v5193 : tensor<32x40x28x28xf32>
    %v5219 = stablehlo.multiply %v5218, %v5217 : tensor<32x40x28x28xf32>
    %v5220 = stablehlo.reshape %v5219 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v5221 = stablehlo.reshape %v5220 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5222 = stablehlo.reverse %b5pW, dims = [2, 3] : tensor<40x240x1x1xf32>
    %v5223 = stablehlo.transpose %v5222, dims = [1, 0, 2, 3] : (tensor<40x240x1x1xf32>) -> tensor<240x40x1x1xf32>
    %v5224 = stablehlo.convolution(%v5221, %v5223)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<32x240x28x28xf32>
    %v5225 = stablehlo.reshape %v5224 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5226 = stablehlo.reshape %v483 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5227 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5228 = stablehlo.constant dense<25088.0> : tensor<32x40x28x28xf32>
    %v5229 = stablehlo.reduce(%v5226 init: %v5227) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5230 = stablehlo.broadcast_in_dim %v5229, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5231 = stablehlo.divide %v5230, %v5228 : tensor<32x40x28x28xf32>
    %v5232 = stablehlo.subtract %v5226, %v5231 : tensor<32x40x28x28xf32>
    %v5233 = stablehlo.multiply %v5232, %v5232 : tensor<32x40x28x28xf32>
    %v5234 = stablehlo.reduce(%v5233 init: %v5227) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5235 = stablehlo.broadcast_in_dim %v5234, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5236 = stablehlo.divide %v5235, %v5228 : tensor<32x40x28x28xf32>
    %v5237 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v5238 = stablehlo.add %v5236, %v5237 : tensor<32x40x28x28xf32>
    %v5239 = stablehlo.rsqrt %v5238 : tensor<32x40x28x28xf32>
    %v5240 = stablehlo.multiply %v5232, %v5239 : tensor<32x40x28x28xf32>
    %v5241 = stablehlo.reshape %v5154 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5242 = stablehlo.multiply %v5241, %v5240 : tensor<32x40x28x28xf32>
    %v5243 = stablehlo.reduce(%v5242 init: %v5227) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5244 = stablehlo.constant dense<0.05> : tensor<40xf32>
    %v5245 = stablehlo.multiply %v5243, %v5244 : tensor<40xf32>
    %v5246 = stablehlo.subtract %b5pg, %v5245 : tensor<40xf32>
    %v5247 = stablehlo.reshape %v5154 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5248 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5249 = stablehlo.reduce(%v5247 init: %v5248) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5250 = stablehlo.constant dense<0.05> : tensor<40xf32>
    %v5251 = stablehlo.multiply %v5249, %v5250 : tensor<40xf32>
    %v5252 = stablehlo.subtract %b5pbt, %v5251 : tensor<40xf32>
    %v5253 = stablehlo.reshape %v478 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5254 = stablehlo.reshape %v5220 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5255 = stablehlo.transpose %v5253, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %v5256 = stablehlo.transpose %v5254, dims = [1, 0, 2, 3] : (tensor<32x40x28x28xf32>) -> tensor<40x32x28x28xf32>
    %v5257 = stablehlo.convolution(%v5255, %v5256)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<240x32x28x28xf32>, tensor<40x32x28x28xf32>) -> tensor<240x40x1x1xf32>
    %v5258 = stablehlo.transpose %v5257, dims = [1, 0, 2, 3] : (tensor<240x40x1x1xf32>) -> tensor<40x240x1x1xf32>
    %v5259 = stablehlo.constant dense<0.05> : tensor<40x240x1x1xf32>
    %v5260 = stablehlo.multiply %v5258, %v5259 : tensor<40x240x1x1xf32>
    %v5261 = stablehlo.subtract %b5pW, %v5260 : tensor<40x240x1x1xf32>
    %v5262 = stablehlo.reshape %v461 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5263 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5264 = stablehlo.reduce(%v5262 init: %v5263) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v5265 = stablehlo.constant dense<784.0> : tensor<32x240xf32>
    %v5266 = stablehlo.divide %v5264, %v5265 : tensor<32x240xf32>
    %v5267 = stablehlo.dot_general %v5266, %b5zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v5268 = stablehlo.broadcast_in_dim %b5zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v5269 = stablehlo.add %v5267, %v5268 : tensor<32x10xf32>
    %v5270 = stablehlo.logistic %v5269 : tensor<32x10xf32>
    %v5271 = stablehlo.multiply %v5269, %v5270 : tensor<32x10xf32>
    %v5272 = stablehlo.dot_general %v5271, %b5zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v5273 = stablehlo.broadcast_in_dim %b5zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v5274 = stablehlo.add %v5272, %v5273 : tensor<32x240xf32>
    %v5275 = stablehlo.logistic %v5274 : tensor<32x240xf32>
    %v5276 = stablehlo.reshape %v5225 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5277 = stablehlo.broadcast_in_dim %v5275, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x28x28xf32>
    %v5278 = stablehlo.multiply %v5277, %v5276 : tensor<32x240x28x28xf32>
    %v5279 = stablehlo.multiply %v5262, %v5276 : tensor<32x240x28x28xf32>
    %v5280 = stablehlo.reduce(%v5279 init: %v5263) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v5281 = stablehlo.constant dense<1.0> : tensor<32x240xf32>
    %v5282 = stablehlo.subtract %v5281, %v5275 : tensor<32x240xf32>
    %v5283 = stablehlo.multiply %v5275, %v5282 : tensor<32x240xf32>
    %v5284 = stablehlo.multiply %v5280, %v5283 : tensor<32x240xf32>
    %v5285 = stablehlo.dot_general %v5284, %b5zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<10x240xf32>) -> tensor<32x10xf32>
    %v5286 = stablehlo.logistic %v5269 : tensor<32x10xf32>
    %v5287 = stablehlo.constant dense<1.0> : tensor<32x10xf32>
    %v5288 = stablehlo.subtract %v5287, %v5286 : tensor<32x10xf32>
    %v5289 = stablehlo.multiply %v5269, %v5288 : tensor<32x10xf32>
    %v5290 = stablehlo.add %v5287, %v5289 : tensor<32x10xf32>
    %v5291 = stablehlo.multiply %v5286, %v5290 : tensor<32x10xf32>
    %v5292 = stablehlo.multiply %v5285, %v5291 : tensor<32x10xf32>
    %v5293 = stablehlo.dot_general %v5292, %b5zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<240x10xf32>) -> tensor<32x240xf32>
    %v5294 = stablehlo.constant dense<784.0> : tensor<32x240xf32>
    %v5295 = stablehlo.divide %v5293, %v5294 : tensor<32x240xf32>
    %v5296 = stablehlo.broadcast_in_dim %v5295, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x28x28xf32>
    %v5297 = stablehlo.add %v5278, %v5296 : tensor<32x240x28x28xf32>
    %v5298 = stablehlo.reshape %v5297 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5299 = stablehlo.reshape %v461 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5300 = stablehlo.reshape %v5225 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5301 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5302 = stablehlo.multiply %v5299, %v5300 : tensor<32x240x28x28xf32>
    %v5303 = stablehlo.reduce(%v5302 init: %v5301) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v5304 = stablehlo.logistic %v474 : tensor<32x240xf32>
    %v5305 = stablehlo.constant dense<1.0> : tensor<32x240xf32>
    %v5306 = stablehlo.subtract %v5305, %v5304 : tensor<32x240xf32>
    %v5307 = stablehlo.multiply %v5304, %v5306 : tensor<32x240xf32>
    %v5308 = stablehlo.multiply %v5303, %v5307 : tensor<32x240xf32>
    %v5309 = stablehlo.dot_general %v471, %v5308, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<32x240xf32>) -> tensor<10x240xf32>
    %v5310 = stablehlo.constant dense<0.05> : tensor<10x240xf32>
    %v5311 = stablehlo.multiply %v5309, %v5310 : tensor<10x240xf32>
    %v5312 = stablehlo.subtract %b5zW2, %v5311 : tensor<10x240xf32>
    %v5313 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5314 = stablehlo.reduce(%v5308 init: %v5313) applies stablehlo.add across dimensions = [0] : (tensor<32x240xf32>, tensor<f32>) -> tensor<240xf32>
    %v5315 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5316 = stablehlo.multiply %v5314, %v5315 : tensor<240xf32>
    %v5317 = stablehlo.subtract %b5zb2, %v5316 : tensor<240xf32>
    %v5318 = stablehlo.reshape %v5308 : (tensor<32x240xf32>) -> tensor<32x1x240xf32>
    %v5319 = stablehlo.dot_general %v5318, %b5zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x240xf32>, tensor<10x240xf32>) -> tensor<32x1x10xf32>
    %v5320 = stablehlo.reshape %v5319 : (tensor<32x1x10xf32>) -> tensor<32x10xf32>
    %v5321 = stablehlo.logistic %v469 : tensor<32x10xf32>
    %v5322 = stablehlo.constant dense<1.0> : tensor<32x10xf32>
    %v5323 = stablehlo.subtract %v5322, %v5321 : tensor<32x10xf32>
    %v5324 = stablehlo.multiply %v469, %v5323 : tensor<32x10xf32>
    %v5325 = stablehlo.add %v5322, %v5324 : tensor<32x10xf32>
    %v5326 = stablehlo.multiply %v5321, %v5325 : tensor<32x10xf32>
    %v5327 = stablehlo.multiply %v5320, %v5326 : tensor<32x10xf32>
    %v5328 = stablehlo.dot_general %v466, %v5327, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<32x10xf32>) -> tensor<240x10xf32>
    %v5329 = stablehlo.constant dense<0.05> : tensor<240x10xf32>
    %v5330 = stablehlo.multiply %v5328, %v5329 : tensor<240x10xf32>
    %v5331 = stablehlo.subtract %b5zW1, %v5330 : tensor<240x10xf32>
    %v5332 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5333 = stablehlo.reduce(%v5327 init: %v5332) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v5334 = stablehlo.constant dense<0.05> : tensor<10xf32>
    %v5335 = stablehlo.multiply %v5333, %v5334 : tensor<10xf32>
    %v5336 = stablehlo.subtract %b5zb1, %v5335 : tensor<10xf32>
    %v5337 = stablehlo.reshape %v5298 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5338 = stablehlo.reshape %v457 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5339 = stablehlo.logistic %v5338 : tensor<32x240x28x28xf32>
    %v5340 = stablehlo.constant dense<1.0> : tensor<32x240x28x28xf32>
    %v5341 = stablehlo.subtract %v5340, %v5339 : tensor<32x240x28x28xf32>
    %v5342 = stablehlo.multiply %v5338, %v5341 : tensor<32x240x28x28xf32>
    %v5343 = stablehlo.add %v5340, %v5342 : tensor<32x240x28x28xf32>
    %v5344 = stablehlo.multiply %v5339, %v5343 : tensor<32x240x28x28xf32>
    %v5345 = stablehlo.multiply %v5337, %v5344 : tensor<32x240x28x28xf32>
    %v5346 = stablehlo.reshape %v5345 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5347 = stablehlo.reshape %v437 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5348 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5349 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v5350 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v5351 = stablehlo.reduce(%v5347 init: %v5348) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5352 = stablehlo.broadcast_in_dim %v5351, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5353 = stablehlo.divide %v5352, %v5349 : tensor<32x240x28x28xf32>
    %v5354 = stablehlo.subtract %v5347, %v5353 : tensor<32x240x28x28xf32>
    %v5355 = stablehlo.multiply %v5354, %v5354 : tensor<32x240x28x28xf32>
    %v5356 = stablehlo.reduce(%v5355 init: %v5348) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5357 = stablehlo.broadcast_in_dim %v5356, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5358 = stablehlo.divide %v5357, %v5349 : tensor<32x240x28x28xf32>
    %v5359 = stablehlo.add %v5358, %v5350 : tensor<32x240x28x28xf32>
    %v5360 = stablehlo.rsqrt %v5359 : tensor<32x240x28x28xf32>
    %v5361 = stablehlo.multiply %v5354, %v5360 : tensor<32x240x28x28xf32>
    %v5362 = stablehlo.broadcast_in_dim %b5dg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5363 = stablehlo.reshape %v5346 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5364 = stablehlo.multiply %v5362, %v5363 : tensor<32x240x28x28xf32>
    %v5365 = stablehlo.reduce(%v5364 init: %v5348) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5366 = stablehlo.broadcast_in_dim %v5365, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5367 = stablehlo.multiply %v5361, %v5364 : tensor<32x240x28x28xf32>
    %v5368 = stablehlo.reduce(%v5367 init: %v5348) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5369 = stablehlo.broadcast_in_dim %v5368, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5370 = stablehlo.multiply %v5364, %v5349 : tensor<32x240x28x28xf32>
    %v5371 = stablehlo.subtract %v5370, %v5366 : tensor<32x240x28x28xf32>
    %v5372 = stablehlo.multiply %v5361, %v5369 : tensor<32x240x28x28xf32>
    %v5373 = stablehlo.subtract %v5371, %v5372 : tensor<32x240x28x28xf32>
    %v5374 = stablehlo.divide %v5360, %v5349 : tensor<32x240x28x28xf32>
    %v5375 = stablehlo.multiply %v5374, %v5373 : tensor<32x240x28x28xf32>
    %v5376 = stablehlo.reshape %v5375 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5377 = stablehlo.reshape %v5376 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5378 = stablehlo.reverse %b5dW, dims = [2, 3] : tensor<240x1x5x5xf32>
    %v5379 = stablehlo.convolution(%v5377, %v5378)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x5x5xf32>) -> tensor<32x240x28x28xf32>
    %v5380 = stablehlo.reshape %v5379 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5381 = stablehlo.reshape %v437 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5382 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5383 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v5384 = stablehlo.reduce(%v5381 init: %v5382) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5385 = stablehlo.broadcast_in_dim %v5384, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5386 = stablehlo.divide %v5385, %v5383 : tensor<32x240x28x28xf32>
    %v5387 = stablehlo.subtract %v5381, %v5386 : tensor<32x240x28x28xf32>
    %v5388 = stablehlo.multiply %v5387, %v5387 : tensor<32x240x28x28xf32>
    %v5389 = stablehlo.reduce(%v5388 init: %v5382) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5390 = stablehlo.broadcast_in_dim %v5389, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5391 = stablehlo.divide %v5390, %v5383 : tensor<32x240x28x28xf32>
    %v5392 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v5393 = stablehlo.add %v5391, %v5392 : tensor<32x240x28x28xf32>
    %v5394 = stablehlo.rsqrt %v5393 : tensor<32x240x28x28xf32>
    %v5395 = stablehlo.multiply %v5387, %v5394 : tensor<32x240x28x28xf32>
    %v5396 = stablehlo.reshape %v5346 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5397 = stablehlo.multiply %v5396, %v5395 : tensor<32x240x28x28xf32>
    %v5398 = stablehlo.reduce(%v5397 init: %v5382) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5399 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5400 = stablehlo.multiply %v5398, %v5399 : tensor<240xf32>
    %v5401 = stablehlo.subtract %b5dg, %v5400 : tensor<240xf32>
    %v5402 = stablehlo.reshape %v5346 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5403 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5404 = stablehlo.reduce(%v5402 init: %v5403) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5405 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5406 = stablehlo.multiply %v5404, %v5405 : tensor<240xf32>
    %v5407 = stablehlo.subtract %b5dbt, %v5406 : tensor<240xf32>
    %v5408 = stablehlo.reshape %v432 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5409 = stablehlo.reshape %v5376 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5410 = stablehlo.transpose %v5408, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %v5411 = stablehlo.transpose %v5409, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %v5412 = stablehlo.convolution(%v5410, %v5411)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 240 : i64, feature_group_count = 1 : i64} : (tensor<240x32x28x28xf32>, tensor<240x32x28x28xf32>) -> tensor<1x240x5x5xf32>
    %v5413 = stablehlo.reshape %v5412 : (tensor<1x240x5x5xf32>) -> tensor<240x1x5x5xf32>
    %v5414 = stablehlo.constant dense<0.05> : tensor<240x1x5x5xf32>
    %v5415 = stablehlo.multiply %v5413, %v5414 : tensor<240x1x5x5xf32>
    %v5416 = stablehlo.subtract %b5dW, %v5415 : tensor<240x1x5x5xf32>
    %v5417 = stablehlo.reshape %v5380 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5418 = stablehlo.reshape %v428 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5419 = stablehlo.logistic %v5418 : tensor<32x240x28x28xf32>
    %v5420 = stablehlo.constant dense<1.0> : tensor<32x240x28x28xf32>
    %v5421 = stablehlo.subtract %v5420, %v5419 : tensor<32x240x28x28xf32>
    %v5422 = stablehlo.multiply %v5418, %v5421 : tensor<32x240x28x28xf32>
    %v5423 = stablehlo.add %v5420, %v5422 : tensor<32x240x28x28xf32>
    %v5424 = stablehlo.multiply %v5419, %v5423 : tensor<32x240x28x28xf32>
    %v5425 = stablehlo.multiply %v5417, %v5424 : tensor<32x240x28x28xf32>
    %v5426 = stablehlo.reshape %v5425 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5427 = stablehlo.reshape %v408 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5428 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5429 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v5430 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v5431 = stablehlo.reduce(%v5427 init: %v5428) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5432 = stablehlo.broadcast_in_dim %v5431, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5433 = stablehlo.divide %v5432, %v5429 : tensor<32x240x28x28xf32>
    %v5434 = stablehlo.subtract %v5427, %v5433 : tensor<32x240x28x28xf32>
    %v5435 = stablehlo.multiply %v5434, %v5434 : tensor<32x240x28x28xf32>
    %v5436 = stablehlo.reduce(%v5435 init: %v5428) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5437 = stablehlo.broadcast_in_dim %v5436, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5438 = stablehlo.divide %v5437, %v5429 : tensor<32x240x28x28xf32>
    %v5439 = stablehlo.add %v5438, %v5430 : tensor<32x240x28x28xf32>
    %v5440 = stablehlo.rsqrt %v5439 : tensor<32x240x28x28xf32>
    %v5441 = stablehlo.multiply %v5434, %v5440 : tensor<32x240x28x28xf32>
    %v5442 = stablehlo.broadcast_in_dim %b5eg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5443 = stablehlo.reshape %v5426 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5444 = stablehlo.multiply %v5442, %v5443 : tensor<32x240x28x28xf32>
    %v5445 = stablehlo.reduce(%v5444 init: %v5428) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5446 = stablehlo.broadcast_in_dim %v5445, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5447 = stablehlo.multiply %v5441, %v5444 : tensor<32x240x28x28xf32>
    %v5448 = stablehlo.reduce(%v5447 init: %v5428) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5449 = stablehlo.broadcast_in_dim %v5448, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5450 = stablehlo.multiply %v5444, %v5429 : tensor<32x240x28x28xf32>
    %v5451 = stablehlo.subtract %v5450, %v5446 : tensor<32x240x28x28xf32>
    %v5452 = stablehlo.multiply %v5441, %v5449 : tensor<32x240x28x28xf32>
    %v5453 = stablehlo.subtract %v5451, %v5452 : tensor<32x240x28x28xf32>
    %v5454 = stablehlo.divide %v5440, %v5429 : tensor<32x240x28x28xf32>
    %v5455 = stablehlo.multiply %v5454, %v5453 : tensor<32x240x28x28xf32>
    %v5456 = stablehlo.reshape %v5455 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v5457 = stablehlo.reshape %v5456 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5458 = stablehlo.reverse %b5eW, dims = [2, 3] : tensor<240x40x1x1xf32>
    %v5459 = stablehlo.transpose %v5458, dims = [1, 0, 2, 3] : (tensor<240x40x1x1xf32>) -> tensor<40x240x1x1xf32>
    %v5460 = stablehlo.convolution(%v5457, %v5459)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x28x28xf32>, tensor<40x240x1x1xf32>) -> tensor<32x40x28x28xf32>
    %v5461 = stablehlo.reshape %v5460 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v5462 = stablehlo.reshape %v408 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5463 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5464 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v5465 = stablehlo.reduce(%v5462 init: %v5463) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5466 = stablehlo.broadcast_in_dim %v5465, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5467 = stablehlo.divide %v5466, %v5464 : tensor<32x240x28x28xf32>
    %v5468 = stablehlo.subtract %v5462, %v5467 : tensor<32x240x28x28xf32>
    %v5469 = stablehlo.multiply %v5468, %v5468 : tensor<32x240x28x28xf32>
    %v5470 = stablehlo.reduce(%v5469 init: %v5463) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5471 = stablehlo.broadcast_in_dim %v5470, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v5472 = stablehlo.divide %v5471, %v5464 : tensor<32x240x28x28xf32>
    %v5473 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v5474 = stablehlo.add %v5472, %v5473 : tensor<32x240x28x28xf32>
    %v5475 = stablehlo.rsqrt %v5474 : tensor<32x240x28x28xf32>
    %v5476 = stablehlo.multiply %v5468, %v5475 : tensor<32x240x28x28xf32>
    %v5477 = stablehlo.reshape %v5426 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5478 = stablehlo.multiply %v5477, %v5476 : tensor<32x240x28x28xf32>
    %v5479 = stablehlo.reduce(%v5478 init: %v5463) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5480 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5481 = stablehlo.multiply %v5479, %v5480 : tensor<240xf32>
    %v5482 = stablehlo.subtract %b5eg, %v5481 : tensor<240xf32>
    %v5483 = stablehlo.reshape %v5426 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5484 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5485 = stablehlo.reduce(%v5483 init: %v5484) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v5486 = stablehlo.constant dense<0.05> : tensor<240xf32>
    %v5487 = stablehlo.multiply %v5485, %v5486 : tensor<240xf32>
    %v5488 = stablehlo.subtract %b5ebt, %v5487 : tensor<240xf32>
    %v5489 = stablehlo.reshape %v403 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5490 = stablehlo.reshape %v5456 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v5491 = stablehlo.transpose %v5489, dims = [1, 0, 2, 3] : (tensor<32x40x28x28xf32>) -> tensor<40x32x28x28xf32>
    %v5492 = stablehlo.transpose %v5490, dims = [1, 0, 2, 3] : (tensor<32x240x28x28xf32>) -> tensor<240x32x28x28xf32>
    %v5493 = stablehlo.convolution(%v5491, %v5492)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<40x32x28x28xf32>, tensor<240x32x28x28xf32>) -> tensor<40x240x1x1xf32>
    %v5494 = stablehlo.transpose %v5493, dims = [1, 0, 2, 3] : (tensor<40x240x1x1xf32>) -> tensor<240x40x1x1xf32>
    %v5495 = stablehlo.constant dense<0.05> : tensor<240x40x1x1xf32>
    %v5496 = stablehlo.multiply %v5494, %v5495 : tensor<240x40x1x1xf32>
    %v5497 = stablehlo.subtract %b5eW, %v5496 : tensor<240x40x1x1xf32>
    %v5498 = stablehlo.reshape %v5461 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5499 = stablehlo.reshape %v5154 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5500 = stablehlo.add %v5498, %v5499 : tensor<32x40x28x28xf32>
    %v5501 = stablehlo.reshape %v5500 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v5502 = stablehlo.reshape %v383 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5503 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5504 = stablehlo.constant dense<25088.0> : tensor<32x40x28x28xf32>
    %v5505 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v5506 = stablehlo.reduce(%v5502 init: %v5503) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5507 = stablehlo.broadcast_in_dim %v5506, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5508 = stablehlo.divide %v5507, %v5504 : tensor<32x40x28x28xf32>
    %v5509 = stablehlo.subtract %v5502, %v5508 : tensor<32x40x28x28xf32>
    %v5510 = stablehlo.multiply %v5509, %v5509 : tensor<32x40x28x28xf32>
    %v5511 = stablehlo.reduce(%v5510 init: %v5503) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5512 = stablehlo.broadcast_in_dim %v5511, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5513 = stablehlo.divide %v5512, %v5504 : tensor<32x40x28x28xf32>
    %v5514 = stablehlo.add %v5513, %v5505 : tensor<32x40x28x28xf32>
    %v5515 = stablehlo.rsqrt %v5514 : tensor<32x40x28x28xf32>
    %v5516 = stablehlo.multiply %v5509, %v5515 : tensor<32x40x28x28xf32>
    %v5517 = stablehlo.broadcast_in_dim %b4pg, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5518 = stablehlo.reshape %v5501 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5519 = stablehlo.multiply %v5517, %v5518 : tensor<32x40x28x28xf32>
    %v5520 = stablehlo.reduce(%v5519 init: %v5503) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5521 = stablehlo.broadcast_in_dim %v5520, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5522 = stablehlo.multiply %v5516, %v5519 : tensor<32x40x28x28xf32>
    %v5523 = stablehlo.reduce(%v5522 init: %v5503) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5524 = stablehlo.broadcast_in_dim %v5523, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5525 = stablehlo.multiply %v5519, %v5504 : tensor<32x40x28x28xf32>
    %v5526 = stablehlo.subtract %v5525, %v5521 : tensor<32x40x28x28xf32>
    %v5527 = stablehlo.multiply %v5516, %v5524 : tensor<32x40x28x28xf32>
    %v5528 = stablehlo.subtract %v5526, %v5527 : tensor<32x40x28x28xf32>
    %v5529 = stablehlo.divide %v5515, %v5504 : tensor<32x40x28x28xf32>
    %v5530 = stablehlo.multiply %v5529, %v5528 : tensor<32x40x28x28xf32>
    %v5531 = stablehlo.reshape %v5530 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v5532 = stablehlo.reshape %v5531 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5533 = stablehlo.reverse %b4pW, dims = [2, 3] : tensor<40x144x1x1xf32>
    %v5534 = stablehlo.transpose %v5533, dims = [1, 0, 2, 3] : (tensor<40x144x1x1xf32>) -> tensor<144x40x1x1xf32>
    %v5535 = stablehlo.convolution(%v5532, %v5534)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<144x40x1x1xf32>) -> tensor<32x144x28x28xf32>
    %v5536 = stablehlo.reshape %v5535 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v5537 = stablehlo.reshape %v383 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5538 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5539 = stablehlo.constant dense<25088.0> : tensor<32x40x28x28xf32>
    %v5540 = stablehlo.reduce(%v5537 init: %v5538) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5541 = stablehlo.broadcast_in_dim %v5540, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5542 = stablehlo.divide %v5541, %v5539 : tensor<32x40x28x28xf32>
    %v5543 = stablehlo.subtract %v5537, %v5542 : tensor<32x40x28x28xf32>
    %v5544 = stablehlo.multiply %v5543, %v5543 : tensor<32x40x28x28xf32>
    %v5545 = stablehlo.reduce(%v5544 init: %v5538) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5546 = stablehlo.broadcast_in_dim %v5545, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v5547 = stablehlo.divide %v5546, %v5539 : tensor<32x40x28x28xf32>
    %v5548 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v5549 = stablehlo.add %v5547, %v5548 : tensor<32x40x28x28xf32>
    %v5550 = stablehlo.rsqrt %v5549 : tensor<32x40x28x28xf32>
    %v5551 = stablehlo.multiply %v5543, %v5550 : tensor<32x40x28x28xf32>
    %v5552 = stablehlo.reshape %v5501 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5553 = stablehlo.multiply %v5552, %v5551 : tensor<32x40x28x28xf32>
    %v5554 = stablehlo.reduce(%v5553 init: %v5538) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5555 = stablehlo.constant dense<0.05> : tensor<40xf32>
    %v5556 = stablehlo.multiply %v5554, %v5555 : tensor<40xf32>
    %v5557 = stablehlo.subtract %b4pg, %v5556 : tensor<40xf32>
    %v5558 = stablehlo.reshape %v5501 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5559 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5560 = stablehlo.reduce(%v5558 init: %v5559) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v5561 = stablehlo.constant dense<0.05> : tensor<40xf32>
    %v5562 = stablehlo.multiply %v5560, %v5561 : tensor<40xf32>
    %v5563 = stablehlo.subtract %b4pbt, %v5562 : tensor<40xf32>
    %v5564 = stablehlo.reshape %v378 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5565 = stablehlo.reshape %v5531 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v5566 = stablehlo.transpose %v5564, dims = [1, 0, 2, 3] : (tensor<32x144x28x28xf32>) -> tensor<144x32x28x28xf32>
    %v5567 = stablehlo.transpose %v5565, dims = [1, 0, 2, 3] : (tensor<32x40x28x28xf32>) -> tensor<40x32x28x28xf32>
    %v5568 = stablehlo.convolution(%v5566, %v5567)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<144x32x28x28xf32>, tensor<40x32x28x28xf32>) -> tensor<144x40x1x1xf32>
    %v5569 = stablehlo.transpose %v5568, dims = [1, 0, 2, 3] : (tensor<144x40x1x1xf32>) -> tensor<40x144x1x1xf32>
    %v5570 = stablehlo.constant dense<0.05> : tensor<40x144x1x1xf32>
    %v5571 = stablehlo.multiply %v5569, %v5570 : tensor<40x144x1x1xf32>
    %v5572 = stablehlo.subtract %b4pW, %v5571 : tensor<40x144x1x1xf32>
    %v5573 = stablehlo.reshape %v361 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5574 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5575 = stablehlo.reduce(%v5573 init: %v5574) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5576 = stablehlo.constant dense<784.0> : tensor<32x144xf32>
    %v5577 = stablehlo.divide %v5575, %v5576 : tensor<32x144xf32>
    %v5578 = stablehlo.dot_general %v5577, %b4zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v5579 = stablehlo.broadcast_in_dim %b4zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v5580 = stablehlo.add %v5578, %v5579 : tensor<32x6xf32>
    %v5581 = stablehlo.logistic %v5580 : tensor<32x6xf32>
    %v5582 = stablehlo.multiply %v5580, %v5581 : tensor<32x6xf32>
    %v5583 = stablehlo.dot_general %v5582, %b4zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v5584 = stablehlo.broadcast_in_dim %b4zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v5585 = stablehlo.add %v5583, %v5584 : tensor<32x144xf32>
    %v5586 = stablehlo.logistic %v5585 : tensor<32x144xf32>
    %v5587 = stablehlo.reshape %v5536 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5588 = stablehlo.broadcast_in_dim %v5586, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v5589 = stablehlo.multiply %v5588, %v5587 : tensor<32x144x28x28xf32>
    %v5590 = stablehlo.multiply %v5573, %v5587 : tensor<32x144x28x28xf32>
    %v5591 = stablehlo.reduce(%v5590 init: %v5574) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5592 = stablehlo.constant dense<1.0> : tensor<32x144xf32>
    %v5593 = stablehlo.subtract %v5592, %v5586 : tensor<32x144xf32>
    %v5594 = stablehlo.multiply %v5586, %v5593 : tensor<32x144xf32>
    %v5595 = stablehlo.multiply %v5591, %v5594 : tensor<32x144xf32>
    %v5596 = stablehlo.dot_general %v5595, %b4zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<6x144xf32>) -> tensor<32x6xf32>
    %v5597 = stablehlo.logistic %v5580 : tensor<32x6xf32>
    %v5598 = stablehlo.constant dense<1.0> : tensor<32x6xf32>
    %v5599 = stablehlo.subtract %v5598, %v5597 : tensor<32x6xf32>
    %v5600 = stablehlo.multiply %v5580, %v5599 : tensor<32x6xf32>
    %v5601 = stablehlo.add %v5598, %v5600 : tensor<32x6xf32>
    %v5602 = stablehlo.multiply %v5597, %v5601 : tensor<32x6xf32>
    %v5603 = stablehlo.multiply %v5596, %v5602 : tensor<32x6xf32>
    %v5604 = stablehlo.dot_general %v5603, %b4zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<144x6xf32>) -> tensor<32x144xf32>
    %v5605 = stablehlo.constant dense<784.0> : tensor<32x144xf32>
    %v5606 = stablehlo.divide %v5604, %v5605 : tensor<32x144xf32>
    %v5607 = stablehlo.broadcast_in_dim %v5606, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v5608 = stablehlo.add %v5589, %v5607 : tensor<32x144x28x28xf32>
    %v5609 = stablehlo.reshape %v5608 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v5610 = stablehlo.reshape %v361 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5611 = stablehlo.reshape %v5536 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5612 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5613 = stablehlo.multiply %v5610, %v5611 : tensor<32x144x28x28xf32>
    %v5614 = stablehlo.reduce(%v5613 init: %v5612) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5615 = stablehlo.logistic %v374 : tensor<32x144xf32>
    %v5616 = stablehlo.constant dense<1.0> : tensor<32x144xf32>
    %v5617 = stablehlo.subtract %v5616, %v5615 : tensor<32x144xf32>
    %v5618 = stablehlo.multiply %v5615, %v5617 : tensor<32x144xf32>
    %v5619 = stablehlo.multiply %v5614, %v5618 : tensor<32x144xf32>
    %v5620 = stablehlo.dot_general %v371, %v5619, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<32x144xf32>) -> tensor<6x144xf32>
    %v5621 = stablehlo.constant dense<0.05> : tensor<6x144xf32>
    %v5622 = stablehlo.multiply %v5620, %v5621 : tensor<6x144xf32>
    %v5623 = stablehlo.subtract %b4zW2, %v5622 : tensor<6x144xf32>
    %v5624 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5625 = stablehlo.reduce(%v5619 init: %v5624) applies stablehlo.add across dimensions = [0] : (tensor<32x144xf32>, tensor<f32>) -> tensor<144xf32>
    %v5626 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v5627 = stablehlo.multiply %v5625, %v5626 : tensor<144xf32>
    %v5628 = stablehlo.subtract %b4zb2, %v5627 : tensor<144xf32>
    %v5629 = stablehlo.reshape %v5619 : (tensor<32x144xf32>) -> tensor<32x1x144xf32>
    %v5630 = stablehlo.dot_general %v5629, %b4zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x144xf32>, tensor<6x144xf32>) -> tensor<32x1x6xf32>
    %v5631 = stablehlo.reshape %v5630 : (tensor<32x1x6xf32>) -> tensor<32x6xf32>
    %v5632 = stablehlo.logistic %v369 : tensor<32x6xf32>
    %v5633 = stablehlo.constant dense<1.0> : tensor<32x6xf32>
    %v5634 = stablehlo.subtract %v5633, %v5632 : tensor<32x6xf32>
    %v5635 = stablehlo.multiply %v369, %v5634 : tensor<32x6xf32>
    %v5636 = stablehlo.add %v5633, %v5635 : tensor<32x6xf32>
    %v5637 = stablehlo.multiply %v5632, %v5636 : tensor<32x6xf32>
    %v5638 = stablehlo.multiply %v5631, %v5637 : tensor<32x6xf32>
    %v5639 = stablehlo.dot_general %v366, %v5638, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<32x6xf32>) -> tensor<144x6xf32>
    %v5640 = stablehlo.constant dense<0.05> : tensor<144x6xf32>
    %v5641 = stablehlo.multiply %v5639, %v5640 : tensor<144x6xf32>
    %v5642 = stablehlo.subtract %b4zW1, %v5641 : tensor<144x6xf32>
    %v5643 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5644 = stablehlo.reduce(%v5638 init: %v5643) applies stablehlo.add across dimensions = [0] : (tensor<32x6xf32>, tensor<f32>) -> tensor<6xf32>
    %v5645 = stablehlo.constant dense<0.05> : tensor<6xf32>
    %v5646 = stablehlo.multiply %v5644, %v5645 : tensor<6xf32>
    %v5647 = stablehlo.subtract %b4zb1, %v5646 : tensor<6xf32>
    %v5648 = stablehlo.reshape %v5609 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5649 = stablehlo.reshape %v357 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5650 = stablehlo.logistic %v5649 : tensor<32x144x28x28xf32>
    %v5651 = stablehlo.constant dense<1.0> : tensor<32x144x28x28xf32>
    %v5652 = stablehlo.subtract %v5651, %v5650 : tensor<32x144x28x28xf32>
    %v5653 = stablehlo.multiply %v5649, %v5652 : tensor<32x144x28x28xf32>
    %v5654 = stablehlo.add %v5651, %v5653 : tensor<32x144x28x28xf32>
    %v5655 = stablehlo.multiply %v5650, %v5654 : tensor<32x144x28x28xf32>
    %v5656 = stablehlo.multiply %v5648, %v5655 : tensor<32x144x28x28xf32>
    %v5657 = stablehlo.reshape %v5656 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v5658 = stablehlo.reshape %v337 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5659 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5660 = stablehlo.constant dense<25088.0> : tensor<32x144x28x28xf32>
    %v5661 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v5662 = stablehlo.reduce(%v5658 init: %v5659) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5663 = stablehlo.broadcast_in_dim %v5662, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5664 = stablehlo.divide %v5663, %v5660 : tensor<32x144x28x28xf32>
    %v5665 = stablehlo.subtract %v5658, %v5664 : tensor<32x144x28x28xf32>
    %v5666 = stablehlo.multiply %v5665, %v5665 : tensor<32x144x28x28xf32>
    %v5667 = stablehlo.reduce(%v5666 init: %v5659) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5668 = stablehlo.broadcast_in_dim %v5667, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5669 = stablehlo.divide %v5668, %v5660 : tensor<32x144x28x28xf32>
    %v5670 = stablehlo.add %v5669, %v5661 : tensor<32x144x28x28xf32>
    %v5671 = stablehlo.rsqrt %v5670 : tensor<32x144x28x28xf32>
    %v5672 = stablehlo.multiply %v5665, %v5671 : tensor<32x144x28x28xf32>
    %v5673 = stablehlo.broadcast_in_dim %b4dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5674 = stablehlo.reshape %v5657 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5675 = stablehlo.multiply %v5673, %v5674 : tensor<32x144x28x28xf32>
    %v5676 = stablehlo.reduce(%v5675 init: %v5659) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5677 = stablehlo.broadcast_in_dim %v5676, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5678 = stablehlo.multiply %v5672, %v5675 : tensor<32x144x28x28xf32>
    %v5679 = stablehlo.reduce(%v5678 init: %v5659) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5680 = stablehlo.broadcast_in_dim %v5679, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5681 = stablehlo.multiply %v5675, %v5660 : tensor<32x144x28x28xf32>
    %v5682 = stablehlo.subtract %v5681, %v5677 : tensor<32x144x28x28xf32>
    %v5683 = stablehlo.multiply %v5672, %v5680 : tensor<32x144x28x28xf32>
    %v5684 = stablehlo.subtract %v5682, %v5683 : tensor<32x144x28x28xf32>
    %v5685 = stablehlo.divide %v5671, %v5660 : tensor<32x144x28x28xf32>
    %v5686 = stablehlo.multiply %v5685, %v5684 : tensor<32x144x28x28xf32>
    %v5687 = stablehlo.reshape %v5686 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v5688 = stablehlo.reshape %v5687 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5689 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5690 = stablehlo.pad %v5688, %v5689, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144x56x56xf32>
    %v5691 = stablehlo.reverse %b4dW, dims = [2, 3] : tensor<144x1x5x5xf32>
    %v5692 = stablehlo.convolution(%v5690, %v5691)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x5x5xf32>) -> tensor<32x144x56x56xf32>
    %v5693 = stablehlo.reshape %v5692 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v5694 = stablehlo.reshape %v337 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5695 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5696 = stablehlo.constant dense<25088.0> : tensor<32x144x28x28xf32>
    %v5697 = stablehlo.reduce(%v5694 init: %v5695) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5698 = stablehlo.broadcast_in_dim %v5697, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5699 = stablehlo.divide %v5698, %v5696 : tensor<32x144x28x28xf32>
    %v5700 = stablehlo.subtract %v5694, %v5699 : tensor<32x144x28x28xf32>
    %v5701 = stablehlo.multiply %v5700, %v5700 : tensor<32x144x28x28xf32>
    %v5702 = stablehlo.reduce(%v5701 init: %v5695) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5703 = stablehlo.broadcast_in_dim %v5702, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v5704 = stablehlo.divide %v5703, %v5696 : tensor<32x144x28x28xf32>
    %v5705 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v5706 = stablehlo.add %v5704, %v5705 : tensor<32x144x28x28xf32>
    %v5707 = stablehlo.rsqrt %v5706 : tensor<32x144x28x28xf32>
    %v5708 = stablehlo.multiply %v5700, %v5707 : tensor<32x144x28x28xf32>
    %v5709 = stablehlo.reshape %v5657 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5710 = stablehlo.multiply %v5709, %v5708 : tensor<32x144x28x28xf32>
    %v5711 = stablehlo.reduce(%v5710 init: %v5695) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5712 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v5713 = stablehlo.multiply %v5711, %v5712 : tensor<144xf32>
    %v5714 = stablehlo.subtract %b4dg, %v5713 : tensor<144xf32>
    %v5715 = stablehlo.reshape %v5657 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5716 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5717 = stablehlo.reduce(%v5715 init: %v5716) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v5718 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v5719 = stablehlo.multiply %v5717, %v5718 : tensor<144xf32>
    %v5720 = stablehlo.subtract %b4dbt, %v5719 : tensor<144xf32>
    %v5721 = stablehlo.reshape %v332 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5722 = stablehlo.reshape %v5687 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v5723 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5724 = stablehlo.pad %v5722, %v5723, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144x56x56xf32>
    %v5725 = stablehlo.transpose %v5721, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v5726 = stablehlo.transpose %v5724, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v5727 = stablehlo.convolution(%v5725, %v5726)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 144 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<1x144x5x5xf32>
    %v5728 = stablehlo.reshape %v5727 : (tensor<1x144x5x5xf32>) -> tensor<144x1x5x5xf32>
    %v5729 = stablehlo.constant dense<0.05> : tensor<144x1x5x5xf32>
    %v5730 = stablehlo.multiply %v5728, %v5729 : tensor<144x1x5x5xf32>
    %v5731 = stablehlo.subtract %b4dW, %v5730 : tensor<144x1x5x5xf32>
    %v5732 = stablehlo.reshape %v5693 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5733 = stablehlo.reshape %v328 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5734 = stablehlo.logistic %v5733 : tensor<32x144x56x56xf32>
    %v5735 = stablehlo.constant dense<1.0> : tensor<32x144x56x56xf32>
    %v5736 = stablehlo.subtract %v5735, %v5734 : tensor<32x144x56x56xf32>
    %v5737 = stablehlo.multiply %v5733, %v5736 : tensor<32x144x56x56xf32>
    %v5738 = stablehlo.add %v5735, %v5737 : tensor<32x144x56x56xf32>
    %v5739 = stablehlo.multiply %v5734, %v5738 : tensor<32x144x56x56xf32>
    %v5740 = stablehlo.multiply %v5732, %v5739 : tensor<32x144x56x56xf32>
    %v5741 = stablehlo.reshape %v5740 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v5742 = stablehlo.reshape %v308 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5743 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5744 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v5745 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v5746 = stablehlo.reduce(%v5742 init: %v5743) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5747 = stablehlo.broadcast_in_dim %v5746, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5748 = stablehlo.divide %v5747, %v5744 : tensor<32x144x56x56xf32>
    %v5749 = stablehlo.subtract %v5742, %v5748 : tensor<32x144x56x56xf32>
    %v5750 = stablehlo.multiply %v5749, %v5749 : tensor<32x144x56x56xf32>
    %v5751 = stablehlo.reduce(%v5750 init: %v5743) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5752 = stablehlo.broadcast_in_dim %v5751, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5753 = stablehlo.divide %v5752, %v5744 : tensor<32x144x56x56xf32>
    %v5754 = stablehlo.add %v5753, %v5745 : tensor<32x144x56x56xf32>
    %v5755 = stablehlo.rsqrt %v5754 : tensor<32x144x56x56xf32>
    %v5756 = stablehlo.multiply %v5749, %v5755 : tensor<32x144x56x56xf32>
    %v5757 = stablehlo.broadcast_in_dim %b4eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5758 = stablehlo.reshape %v5741 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5759 = stablehlo.multiply %v5757, %v5758 : tensor<32x144x56x56xf32>
    %v5760 = stablehlo.reduce(%v5759 init: %v5743) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5761 = stablehlo.broadcast_in_dim %v5760, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5762 = stablehlo.multiply %v5756, %v5759 : tensor<32x144x56x56xf32>
    %v5763 = stablehlo.reduce(%v5762 init: %v5743) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5764 = stablehlo.broadcast_in_dim %v5763, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5765 = stablehlo.multiply %v5759, %v5744 : tensor<32x144x56x56xf32>
    %v5766 = stablehlo.subtract %v5765, %v5761 : tensor<32x144x56x56xf32>
    %v5767 = stablehlo.multiply %v5756, %v5764 : tensor<32x144x56x56xf32>
    %v5768 = stablehlo.subtract %v5766, %v5767 : tensor<32x144x56x56xf32>
    %v5769 = stablehlo.divide %v5755, %v5744 : tensor<32x144x56x56xf32>
    %v5770 = stablehlo.multiply %v5769, %v5768 : tensor<32x144x56x56xf32>
    %v5771 = stablehlo.reshape %v5770 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v5772 = stablehlo.reshape %v5771 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5773 = stablehlo.reverse %b4eW, dims = [2, 3] : tensor<144x24x1x1xf32>
    %v5774 = stablehlo.transpose %v5773, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %v5775 = stablehlo.convolution(%v5772, %v5774)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v5776 = stablehlo.reshape %v5775 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v5777 = stablehlo.reshape %v308 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5778 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5779 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v5780 = stablehlo.reduce(%v5777 init: %v5778) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5781 = stablehlo.broadcast_in_dim %v5780, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5782 = stablehlo.divide %v5781, %v5779 : tensor<32x144x56x56xf32>
    %v5783 = stablehlo.subtract %v5777, %v5782 : tensor<32x144x56x56xf32>
    %v5784 = stablehlo.multiply %v5783, %v5783 : tensor<32x144x56x56xf32>
    %v5785 = stablehlo.reduce(%v5784 init: %v5778) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5786 = stablehlo.broadcast_in_dim %v5785, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5787 = stablehlo.divide %v5786, %v5779 : tensor<32x144x56x56xf32>
    %v5788 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v5789 = stablehlo.add %v5787, %v5788 : tensor<32x144x56x56xf32>
    %v5790 = stablehlo.rsqrt %v5789 : tensor<32x144x56x56xf32>
    %v5791 = stablehlo.multiply %v5783, %v5790 : tensor<32x144x56x56xf32>
    %v5792 = stablehlo.reshape %v5741 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5793 = stablehlo.multiply %v5792, %v5791 : tensor<32x144x56x56xf32>
    %v5794 = stablehlo.reduce(%v5793 init: %v5778) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5795 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v5796 = stablehlo.multiply %v5794, %v5795 : tensor<144xf32>
    %v5797 = stablehlo.subtract %b4eg, %v5796 : tensor<144xf32>
    %v5798 = stablehlo.reshape %v5741 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5799 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5800 = stablehlo.reduce(%v5798 init: %v5799) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5801 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v5802 = stablehlo.multiply %v5800, %v5801 : tensor<144xf32>
    %v5803 = stablehlo.subtract %b4ebt, %v5802 : tensor<144xf32>
    %v5804 = stablehlo.reshape %v303 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5805 = stablehlo.reshape %v5771 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5806 = stablehlo.transpose %v5804, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v5807 = stablehlo.transpose %v5805, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v5808 = stablehlo.convolution(%v5806, %v5807)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<24x144x1x1xf32>
    %v5809 = stablehlo.transpose %v5808, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %v5810 = stablehlo.constant dense<0.05> : tensor<144x24x1x1xf32>
    %v5811 = stablehlo.multiply %v5809, %v5810 : tensor<144x24x1x1xf32>
    %v5812 = stablehlo.subtract %b4eW, %v5811 : tensor<144x24x1x1xf32>
    %v5813 = stablehlo.reshape %v279 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5814 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5815 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v5816 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v5817 = stablehlo.reduce(%v5813 init: %v5814) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5818 = stablehlo.broadcast_in_dim %v5817, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v5819 = stablehlo.divide %v5818, %v5815 : tensor<32x24x56x56xf32>
    %v5820 = stablehlo.subtract %v5813, %v5819 : tensor<32x24x56x56xf32>
    %v5821 = stablehlo.multiply %v5820, %v5820 : tensor<32x24x56x56xf32>
    %v5822 = stablehlo.reduce(%v5821 init: %v5814) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5823 = stablehlo.broadcast_in_dim %v5822, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v5824 = stablehlo.divide %v5823, %v5815 : tensor<32x24x56x56xf32>
    %v5825 = stablehlo.add %v5824, %v5816 : tensor<32x24x56x56xf32>
    %v5826 = stablehlo.rsqrt %v5825 : tensor<32x24x56x56xf32>
    %v5827 = stablehlo.multiply %v5820, %v5826 : tensor<32x24x56x56xf32>
    %v5828 = stablehlo.broadcast_in_dim %b3pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v5829 = stablehlo.reshape %v5776 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5830 = stablehlo.multiply %v5828, %v5829 : tensor<32x24x56x56xf32>
    %v5831 = stablehlo.reduce(%v5830 init: %v5814) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5832 = stablehlo.broadcast_in_dim %v5831, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v5833 = stablehlo.multiply %v5827, %v5830 : tensor<32x24x56x56xf32>
    %v5834 = stablehlo.reduce(%v5833 init: %v5814) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5835 = stablehlo.broadcast_in_dim %v5834, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v5836 = stablehlo.multiply %v5830, %v5815 : tensor<32x24x56x56xf32>
    %v5837 = stablehlo.subtract %v5836, %v5832 : tensor<32x24x56x56xf32>
    %v5838 = stablehlo.multiply %v5827, %v5835 : tensor<32x24x56x56xf32>
    %v5839 = stablehlo.subtract %v5837, %v5838 : tensor<32x24x56x56xf32>
    %v5840 = stablehlo.divide %v5826, %v5815 : tensor<32x24x56x56xf32>
    %v5841 = stablehlo.multiply %v5840, %v5839 : tensor<32x24x56x56xf32>
    %v5842 = stablehlo.reshape %v5841 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v5843 = stablehlo.reshape %v5842 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5844 = stablehlo.reverse %b3pW, dims = [2, 3] : tensor<24x144x1x1xf32>
    %v5845 = stablehlo.transpose %v5844, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %v5846 = stablehlo.convolution(%v5843, %v5845)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v5847 = stablehlo.reshape %v5846 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v5848 = stablehlo.reshape %v279 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5849 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5850 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v5851 = stablehlo.reduce(%v5848 init: %v5849) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5852 = stablehlo.broadcast_in_dim %v5851, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v5853 = stablehlo.divide %v5852, %v5850 : tensor<32x24x56x56xf32>
    %v5854 = stablehlo.subtract %v5848, %v5853 : tensor<32x24x56x56xf32>
    %v5855 = stablehlo.multiply %v5854, %v5854 : tensor<32x24x56x56xf32>
    %v5856 = stablehlo.reduce(%v5855 init: %v5849) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5857 = stablehlo.broadcast_in_dim %v5856, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v5858 = stablehlo.divide %v5857, %v5850 : tensor<32x24x56x56xf32>
    %v5859 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v5860 = stablehlo.add %v5858, %v5859 : tensor<32x24x56x56xf32>
    %v5861 = stablehlo.rsqrt %v5860 : tensor<32x24x56x56xf32>
    %v5862 = stablehlo.multiply %v5854, %v5861 : tensor<32x24x56x56xf32>
    %v5863 = stablehlo.reshape %v5776 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5864 = stablehlo.multiply %v5863, %v5862 : tensor<32x24x56x56xf32>
    %v5865 = stablehlo.reduce(%v5864 init: %v5849) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5866 = stablehlo.constant dense<0.05> : tensor<24xf32>
    %v5867 = stablehlo.multiply %v5865, %v5866 : tensor<24xf32>
    %v5868 = stablehlo.subtract %b3pg, %v5867 : tensor<24xf32>
    %v5869 = stablehlo.reshape %v5776 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5870 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5871 = stablehlo.reduce(%v5869 init: %v5870) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v5872 = stablehlo.constant dense<0.05> : tensor<24xf32>
    %v5873 = stablehlo.multiply %v5871, %v5872 : tensor<24xf32>
    %v5874 = stablehlo.subtract %b3pbt, %v5873 : tensor<24xf32>
    %v5875 = stablehlo.reshape %v274 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5876 = stablehlo.reshape %v5842 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v5877 = stablehlo.transpose %v5875, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v5878 = stablehlo.transpose %v5876, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v5879 = stablehlo.convolution(%v5877, %v5878)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<24x32x56x56xf32>) -> tensor<144x24x1x1xf32>
    %v5880 = stablehlo.transpose %v5879, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %v5881 = stablehlo.constant dense<0.05> : tensor<24x144x1x1xf32>
    %v5882 = stablehlo.multiply %v5880, %v5881 : tensor<24x144x1x1xf32>
    %v5883 = stablehlo.subtract %b3pW, %v5882 : tensor<24x144x1x1xf32>
    %v5884 = stablehlo.reshape %v257 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5885 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5886 = stablehlo.reduce(%v5884 init: %v5885) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5887 = stablehlo.constant dense<3136.0> : tensor<32x144xf32>
    %v5888 = stablehlo.divide %v5886, %v5887 : tensor<32x144xf32>
    %v5889 = stablehlo.dot_general %v5888, %b3zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v5890 = stablehlo.broadcast_in_dim %b3zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v5891 = stablehlo.add %v5889, %v5890 : tensor<32x6xf32>
    %v5892 = stablehlo.logistic %v5891 : tensor<32x6xf32>
    %v5893 = stablehlo.multiply %v5891, %v5892 : tensor<32x6xf32>
    %v5894 = stablehlo.dot_general %v5893, %b3zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v5895 = stablehlo.broadcast_in_dim %b3zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v5896 = stablehlo.add %v5894, %v5895 : tensor<32x144xf32>
    %v5897 = stablehlo.logistic %v5896 : tensor<32x144xf32>
    %v5898 = stablehlo.reshape %v5847 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5899 = stablehlo.broadcast_in_dim %v5897, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v5900 = stablehlo.multiply %v5899, %v5898 : tensor<32x144x56x56xf32>
    %v5901 = stablehlo.multiply %v5884, %v5898 : tensor<32x144x56x56xf32>
    %v5902 = stablehlo.reduce(%v5901 init: %v5885) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5903 = stablehlo.constant dense<1.0> : tensor<32x144xf32>
    %v5904 = stablehlo.subtract %v5903, %v5897 : tensor<32x144xf32>
    %v5905 = stablehlo.multiply %v5897, %v5904 : tensor<32x144xf32>
    %v5906 = stablehlo.multiply %v5902, %v5905 : tensor<32x144xf32>
    %v5907 = stablehlo.dot_general %v5906, %b3zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<6x144xf32>) -> tensor<32x6xf32>
    %v5908 = stablehlo.logistic %v5891 : tensor<32x6xf32>
    %v5909 = stablehlo.constant dense<1.0> : tensor<32x6xf32>
    %v5910 = stablehlo.subtract %v5909, %v5908 : tensor<32x6xf32>
    %v5911 = stablehlo.multiply %v5891, %v5910 : tensor<32x6xf32>
    %v5912 = stablehlo.add %v5909, %v5911 : tensor<32x6xf32>
    %v5913 = stablehlo.multiply %v5908, %v5912 : tensor<32x6xf32>
    %v5914 = stablehlo.multiply %v5907, %v5913 : tensor<32x6xf32>
    %v5915 = stablehlo.dot_general %v5914, %b3zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<144x6xf32>) -> tensor<32x144xf32>
    %v5916 = stablehlo.constant dense<3136.0> : tensor<32x144xf32>
    %v5917 = stablehlo.divide %v5915, %v5916 : tensor<32x144xf32>
    %v5918 = stablehlo.broadcast_in_dim %v5917, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v5919 = stablehlo.add %v5900, %v5918 : tensor<32x144x56x56xf32>
    %v5920 = stablehlo.reshape %v5919 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v5921 = stablehlo.reshape %v257 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5922 = stablehlo.reshape %v5847 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5923 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5924 = stablehlo.multiply %v5921, %v5922 : tensor<32x144x56x56xf32>
    %v5925 = stablehlo.reduce(%v5924 init: %v5923) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v5926 = stablehlo.logistic %v270 : tensor<32x144xf32>
    %v5927 = stablehlo.constant dense<1.0> : tensor<32x144xf32>
    %v5928 = stablehlo.subtract %v5927, %v5926 : tensor<32x144xf32>
    %v5929 = stablehlo.multiply %v5926, %v5928 : tensor<32x144xf32>
    %v5930 = stablehlo.multiply %v5925, %v5929 : tensor<32x144xf32>
    %v5931 = stablehlo.dot_general %v267, %v5930, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<32x144xf32>) -> tensor<6x144xf32>
    %v5932 = stablehlo.constant dense<0.05> : tensor<6x144xf32>
    %v5933 = stablehlo.multiply %v5931, %v5932 : tensor<6x144xf32>
    %v5934 = stablehlo.subtract %b3zW2, %v5933 : tensor<6x144xf32>
    %v5935 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5936 = stablehlo.reduce(%v5930 init: %v5935) applies stablehlo.add across dimensions = [0] : (tensor<32x144xf32>, tensor<f32>) -> tensor<144xf32>
    %v5937 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v5938 = stablehlo.multiply %v5936, %v5937 : tensor<144xf32>
    %v5939 = stablehlo.subtract %b3zb2, %v5938 : tensor<144xf32>
    %v5940 = stablehlo.reshape %v5930 : (tensor<32x144xf32>) -> tensor<32x1x144xf32>
    %v5941 = stablehlo.dot_general %v5940, %b3zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x144xf32>, tensor<6x144xf32>) -> tensor<32x1x6xf32>
    %v5942 = stablehlo.reshape %v5941 : (tensor<32x1x6xf32>) -> tensor<32x6xf32>
    %v5943 = stablehlo.logistic %v265 : tensor<32x6xf32>
    %v5944 = stablehlo.constant dense<1.0> : tensor<32x6xf32>
    %v5945 = stablehlo.subtract %v5944, %v5943 : tensor<32x6xf32>
    %v5946 = stablehlo.multiply %v265, %v5945 : tensor<32x6xf32>
    %v5947 = stablehlo.add %v5944, %v5946 : tensor<32x6xf32>
    %v5948 = stablehlo.multiply %v5943, %v5947 : tensor<32x6xf32>
    %v5949 = stablehlo.multiply %v5942, %v5948 : tensor<32x6xf32>
    %v5950 = stablehlo.dot_general %v262, %v5949, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<32x6xf32>) -> tensor<144x6xf32>
    %v5951 = stablehlo.constant dense<0.05> : tensor<144x6xf32>
    %v5952 = stablehlo.multiply %v5950, %v5951 : tensor<144x6xf32>
    %v5953 = stablehlo.subtract %b3zW1, %v5952 : tensor<144x6xf32>
    %v5954 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5955 = stablehlo.reduce(%v5949 init: %v5954) applies stablehlo.add across dimensions = [0] : (tensor<32x6xf32>, tensor<f32>) -> tensor<6xf32>
    %v5956 = stablehlo.constant dense<0.05> : tensor<6xf32>
    %v5957 = stablehlo.multiply %v5955, %v5956 : tensor<6xf32>
    %v5958 = stablehlo.subtract %b3zb1, %v5957 : tensor<6xf32>
    %v5959 = stablehlo.reshape %v5920 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5960 = stablehlo.reshape %v253 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5961 = stablehlo.logistic %v5960 : tensor<32x144x56x56xf32>
    %v5962 = stablehlo.constant dense<1.0> : tensor<32x144x56x56xf32>
    %v5963 = stablehlo.subtract %v5962, %v5961 : tensor<32x144x56x56xf32>
    %v5964 = stablehlo.multiply %v5960, %v5963 : tensor<32x144x56x56xf32>
    %v5965 = stablehlo.add %v5962, %v5964 : tensor<32x144x56x56xf32>
    %v5966 = stablehlo.multiply %v5961, %v5965 : tensor<32x144x56x56xf32>
    %v5967 = stablehlo.multiply %v5959, %v5966 : tensor<32x144x56x56xf32>
    %v5968 = stablehlo.reshape %v5967 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v5969 = stablehlo.reshape %v233 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5970 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5971 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v5972 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v5973 = stablehlo.reduce(%v5969 init: %v5970) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5974 = stablehlo.broadcast_in_dim %v5973, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5975 = stablehlo.divide %v5974, %v5971 : tensor<32x144x56x56xf32>
    %v5976 = stablehlo.subtract %v5969, %v5975 : tensor<32x144x56x56xf32>
    %v5977 = stablehlo.multiply %v5976, %v5976 : tensor<32x144x56x56xf32>
    %v5978 = stablehlo.reduce(%v5977 init: %v5970) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5979 = stablehlo.broadcast_in_dim %v5978, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5980 = stablehlo.divide %v5979, %v5971 : tensor<32x144x56x56xf32>
    %v5981 = stablehlo.add %v5980, %v5972 : tensor<32x144x56x56xf32>
    %v5982 = stablehlo.rsqrt %v5981 : tensor<32x144x56x56xf32>
    %v5983 = stablehlo.multiply %v5976, %v5982 : tensor<32x144x56x56xf32>
    %v5984 = stablehlo.broadcast_in_dim %b3dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5985 = stablehlo.reshape %v5968 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v5986 = stablehlo.multiply %v5984, %v5985 : tensor<32x144x56x56xf32>
    %v5987 = stablehlo.reduce(%v5986 init: %v5970) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5988 = stablehlo.broadcast_in_dim %v5987, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5989 = stablehlo.multiply %v5983, %v5986 : tensor<32x144x56x56xf32>
    %v5990 = stablehlo.reduce(%v5989 init: %v5970) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v5991 = stablehlo.broadcast_in_dim %v5990, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v5992 = stablehlo.multiply %v5986, %v5971 : tensor<32x144x56x56xf32>
    %v5993 = stablehlo.subtract %v5992, %v5988 : tensor<32x144x56x56xf32>
    %v5994 = stablehlo.multiply %v5983, %v5991 : tensor<32x144x56x56xf32>
    %v5995 = stablehlo.subtract %v5993, %v5994 : tensor<32x144x56x56xf32>
    %v5996 = stablehlo.divide %v5982, %v5971 : tensor<32x144x56x56xf32>
    %v5997 = stablehlo.multiply %v5996, %v5995 : tensor<32x144x56x56xf32>
    %v5998 = stablehlo.reshape %v5997 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v5999 = stablehlo.reshape %v5998 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6000 = stablehlo.reverse %b3dW, dims = [2, 3] : tensor<144x1x3x3xf32>
    %v6001 = stablehlo.convolution(%v5999, %v6000)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x56x56xf32>
    %v6002 = stablehlo.reshape %v6001 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v6003 = stablehlo.reshape %v233 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6004 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6005 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v6006 = stablehlo.reduce(%v6003 init: %v6004) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6007 = stablehlo.broadcast_in_dim %v6006, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6008 = stablehlo.divide %v6007, %v6005 : tensor<32x144x56x56xf32>
    %v6009 = stablehlo.subtract %v6003, %v6008 : tensor<32x144x56x56xf32>
    %v6010 = stablehlo.multiply %v6009, %v6009 : tensor<32x144x56x56xf32>
    %v6011 = stablehlo.reduce(%v6010 init: %v6004) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6012 = stablehlo.broadcast_in_dim %v6011, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6013 = stablehlo.divide %v6012, %v6005 : tensor<32x144x56x56xf32>
    %v6014 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v6015 = stablehlo.add %v6013, %v6014 : tensor<32x144x56x56xf32>
    %v6016 = stablehlo.rsqrt %v6015 : tensor<32x144x56x56xf32>
    %v6017 = stablehlo.multiply %v6009, %v6016 : tensor<32x144x56x56xf32>
    %v6018 = stablehlo.reshape %v5968 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6019 = stablehlo.multiply %v6018, %v6017 : tensor<32x144x56x56xf32>
    %v6020 = stablehlo.reduce(%v6019 init: %v6004) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6021 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v6022 = stablehlo.multiply %v6020, %v6021 : tensor<144xf32>
    %v6023 = stablehlo.subtract %b3dg, %v6022 : tensor<144xf32>
    %v6024 = stablehlo.reshape %v5968 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6025 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6026 = stablehlo.reduce(%v6024 init: %v6025) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6027 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v6028 = stablehlo.multiply %v6026, %v6027 : tensor<144xf32>
    %v6029 = stablehlo.subtract %b3dbt, %v6028 : tensor<144xf32>
    %v6030 = stablehlo.reshape %v228 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6031 = stablehlo.reshape %v5998 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6032 = stablehlo.transpose %v6030, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v6033 = stablehlo.transpose %v6031, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v6034 = stablehlo.convolution(%v6032, %v6033)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 144 : i64, feature_group_count = 1 : i64} : (tensor<144x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<1x144x3x3xf32>
    %v6035 = stablehlo.reshape %v6034 : (tensor<1x144x3x3xf32>) -> tensor<144x1x3x3xf32>
    %v6036 = stablehlo.constant dense<0.05> : tensor<144x1x3x3xf32>
    %v6037 = stablehlo.multiply %v6035, %v6036 : tensor<144x1x3x3xf32>
    %v6038 = stablehlo.subtract %b3dW, %v6037 : tensor<144x1x3x3xf32>
    %v6039 = stablehlo.reshape %v6002 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6040 = stablehlo.reshape %v224 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6041 = stablehlo.logistic %v6040 : tensor<32x144x56x56xf32>
    %v6042 = stablehlo.constant dense<1.0> : tensor<32x144x56x56xf32>
    %v6043 = stablehlo.subtract %v6042, %v6041 : tensor<32x144x56x56xf32>
    %v6044 = stablehlo.multiply %v6040, %v6043 : tensor<32x144x56x56xf32>
    %v6045 = stablehlo.add %v6042, %v6044 : tensor<32x144x56x56xf32>
    %v6046 = stablehlo.multiply %v6041, %v6045 : tensor<32x144x56x56xf32>
    %v6047 = stablehlo.multiply %v6039, %v6046 : tensor<32x144x56x56xf32>
    %v6048 = stablehlo.reshape %v6047 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v6049 = stablehlo.reshape %v204 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6050 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6051 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v6052 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v6053 = stablehlo.reduce(%v6049 init: %v6050) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6054 = stablehlo.broadcast_in_dim %v6053, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6055 = stablehlo.divide %v6054, %v6051 : tensor<32x144x56x56xf32>
    %v6056 = stablehlo.subtract %v6049, %v6055 : tensor<32x144x56x56xf32>
    %v6057 = stablehlo.multiply %v6056, %v6056 : tensor<32x144x56x56xf32>
    %v6058 = stablehlo.reduce(%v6057 init: %v6050) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6059 = stablehlo.broadcast_in_dim %v6058, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6060 = stablehlo.divide %v6059, %v6051 : tensor<32x144x56x56xf32>
    %v6061 = stablehlo.add %v6060, %v6052 : tensor<32x144x56x56xf32>
    %v6062 = stablehlo.rsqrt %v6061 : tensor<32x144x56x56xf32>
    %v6063 = stablehlo.multiply %v6056, %v6062 : tensor<32x144x56x56xf32>
    %v6064 = stablehlo.broadcast_in_dim %b3eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6065 = stablehlo.reshape %v6048 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6066 = stablehlo.multiply %v6064, %v6065 : tensor<32x144x56x56xf32>
    %v6067 = stablehlo.reduce(%v6066 init: %v6050) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6068 = stablehlo.broadcast_in_dim %v6067, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6069 = stablehlo.multiply %v6063, %v6066 : tensor<32x144x56x56xf32>
    %v6070 = stablehlo.reduce(%v6069 init: %v6050) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6071 = stablehlo.broadcast_in_dim %v6070, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6072 = stablehlo.multiply %v6066, %v6051 : tensor<32x144x56x56xf32>
    %v6073 = stablehlo.subtract %v6072, %v6068 : tensor<32x144x56x56xf32>
    %v6074 = stablehlo.multiply %v6063, %v6071 : tensor<32x144x56x56xf32>
    %v6075 = stablehlo.subtract %v6073, %v6074 : tensor<32x144x56x56xf32>
    %v6076 = stablehlo.divide %v6062, %v6051 : tensor<32x144x56x56xf32>
    %v6077 = stablehlo.multiply %v6076, %v6075 : tensor<32x144x56x56xf32>
    %v6078 = stablehlo.reshape %v6077 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v6079 = stablehlo.reshape %v6078 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6080 = stablehlo.reverse %b3eW, dims = [2, 3] : tensor<144x24x1x1xf32>
    %v6081 = stablehlo.transpose %v6080, dims = [1, 0, 2, 3] : (tensor<144x24x1x1xf32>) -> tensor<24x144x1x1xf32>
    %v6082 = stablehlo.convolution(%v6079, %v6081)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v6083 = stablehlo.reshape %v6082 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v6084 = stablehlo.reshape %v204 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6085 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6086 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v6087 = stablehlo.reduce(%v6084 init: %v6085) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6088 = stablehlo.broadcast_in_dim %v6087, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6089 = stablehlo.divide %v6088, %v6086 : tensor<32x144x56x56xf32>
    %v6090 = stablehlo.subtract %v6084, %v6089 : tensor<32x144x56x56xf32>
    %v6091 = stablehlo.multiply %v6090, %v6090 : tensor<32x144x56x56xf32>
    %v6092 = stablehlo.reduce(%v6091 init: %v6085) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6093 = stablehlo.broadcast_in_dim %v6092, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v6094 = stablehlo.divide %v6093, %v6086 : tensor<32x144x56x56xf32>
    %v6095 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v6096 = stablehlo.add %v6094, %v6095 : tensor<32x144x56x56xf32>
    %v6097 = stablehlo.rsqrt %v6096 : tensor<32x144x56x56xf32>
    %v6098 = stablehlo.multiply %v6090, %v6097 : tensor<32x144x56x56xf32>
    %v6099 = stablehlo.reshape %v6048 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6100 = stablehlo.multiply %v6099, %v6098 : tensor<32x144x56x56xf32>
    %v6101 = stablehlo.reduce(%v6100 init: %v6085) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6102 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v6103 = stablehlo.multiply %v6101, %v6102 : tensor<144xf32>
    %v6104 = stablehlo.subtract %b3eg, %v6103 : tensor<144xf32>
    %v6105 = stablehlo.reshape %v6048 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6106 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6107 = stablehlo.reduce(%v6105 init: %v6106) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v6108 = stablehlo.constant dense<0.05> : tensor<144xf32>
    %v6109 = stablehlo.multiply %v6107, %v6108 : tensor<144xf32>
    %v6110 = stablehlo.subtract %b3ebt, %v6109 : tensor<144xf32>
    %v6111 = stablehlo.reshape %v199 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6112 = stablehlo.reshape %v6078 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v6113 = stablehlo.transpose %v6111, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v6114 = stablehlo.transpose %v6112, dims = [1, 0, 2, 3] : (tensor<32x144x56x56xf32>) -> tensor<144x32x56x56xf32>
    %v6115 = stablehlo.convolution(%v6113, %v6114)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<24x32x56x56xf32>, tensor<144x32x56x56xf32>) -> tensor<24x144x1x1xf32>
    %v6116 = stablehlo.transpose %v6115, dims = [1, 0, 2, 3] : (tensor<24x144x1x1xf32>) -> tensor<144x24x1x1xf32>
    %v6117 = stablehlo.constant dense<0.05> : tensor<144x24x1x1xf32>
    %v6118 = stablehlo.multiply %v6116, %v6117 : tensor<144x24x1x1xf32>
    %v6119 = stablehlo.subtract %b3eW, %v6118 : tensor<144x24x1x1xf32>
    %v6120 = stablehlo.reshape %v6083 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6121 = stablehlo.reshape %v5776 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6122 = stablehlo.add %v6120, %v6121 : tensor<32x24x56x56xf32>
    %v6123 = stablehlo.reshape %v6122 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v6124 = stablehlo.reshape %v179 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6125 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6126 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v6127 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v6128 = stablehlo.reduce(%v6124 init: %v6125) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6129 = stablehlo.broadcast_in_dim %v6128, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6130 = stablehlo.divide %v6129, %v6126 : tensor<32x24x56x56xf32>
    %v6131 = stablehlo.subtract %v6124, %v6130 : tensor<32x24x56x56xf32>
    %v6132 = stablehlo.multiply %v6131, %v6131 : tensor<32x24x56x56xf32>
    %v6133 = stablehlo.reduce(%v6132 init: %v6125) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6134 = stablehlo.broadcast_in_dim %v6133, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6135 = stablehlo.divide %v6134, %v6126 : tensor<32x24x56x56xf32>
    %v6136 = stablehlo.add %v6135, %v6127 : tensor<32x24x56x56xf32>
    %v6137 = stablehlo.rsqrt %v6136 : tensor<32x24x56x56xf32>
    %v6138 = stablehlo.multiply %v6131, %v6137 : tensor<32x24x56x56xf32>
    %v6139 = stablehlo.broadcast_in_dim %b2pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6140 = stablehlo.reshape %v6123 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6141 = stablehlo.multiply %v6139, %v6140 : tensor<32x24x56x56xf32>
    %v6142 = stablehlo.reduce(%v6141 init: %v6125) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6143 = stablehlo.broadcast_in_dim %v6142, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6144 = stablehlo.multiply %v6138, %v6141 : tensor<32x24x56x56xf32>
    %v6145 = stablehlo.reduce(%v6144 init: %v6125) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6146 = stablehlo.broadcast_in_dim %v6145, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6147 = stablehlo.multiply %v6141, %v6126 : tensor<32x24x56x56xf32>
    %v6148 = stablehlo.subtract %v6147, %v6143 : tensor<32x24x56x56xf32>
    %v6149 = stablehlo.multiply %v6138, %v6146 : tensor<32x24x56x56xf32>
    %v6150 = stablehlo.subtract %v6148, %v6149 : tensor<32x24x56x56xf32>
    %v6151 = stablehlo.divide %v6137, %v6126 : tensor<32x24x56x56xf32>
    %v6152 = stablehlo.multiply %v6151, %v6150 : tensor<32x24x56x56xf32>
    %v6153 = stablehlo.reshape %v6152 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v6154 = stablehlo.reshape %v6153 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6155 = stablehlo.reverse %b2pW, dims = [2, 3] : tensor<24x96x1x1xf32>
    %v6156 = stablehlo.transpose %v6155, dims = [1, 0, 2, 3] : (tensor<24x96x1x1xf32>) -> tensor<96x24x1x1xf32>
    %v6157 = stablehlo.convolution(%v6154, %v6156)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<96x24x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v6158 = stablehlo.reshape %v6157 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v6159 = stablehlo.reshape %v179 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6160 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6161 = stablehlo.constant dense<100352.0> : tensor<32x24x56x56xf32>
    %v6162 = stablehlo.reduce(%v6159 init: %v6160) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6163 = stablehlo.broadcast_in_dim %v6162, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6164 = stablehlo.divide %v6163, %v6161 : tensor<32x24x56x56xf32>
    %v6165 = stablehlo.subtract %v6159, %v6164 : tensor<32x24x56x56xf32>
    %v6166 = stablehlo.multiply %v6165, %v6165 : tensor<32x24x56x56xf32>
    %v6167 = stablehlo.reduce(%v6166 init: %v6160) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6168 = stablehlo.broadcast_in_dim %v6167, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v6169 = stablehlo.divide %v6168, %v6161 : tensor<32x24x56x56xf32>
    %v6170 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v6171 = stablehlo.add %v6169, %v6170 : tensor<32x24x56x56xf32>
    %v6172 = stablehlo.rsqrt %v6171 : tensor<32x24x56x56xf32>
    %v6173 = stablehlo.multiply %v6165, %v6172 : tensor<32x24x56x56xf32>
    %v6174 = stablehlo.reshape %v6123 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6175 = stablehlo.multiply %v6174, %v6173 : tensor<32x24x56x56xf32>
    %v6176 = stablehlo.reduce(%v6175 init: %v6160) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6177 = stablehlo.constant dense<0.05> : tensor<24xf32>
    %v6178 = stablehlo.multiply %v6176, %v6177 : tensor<24xf32>
    %v6179 = stablehlo.subtract %b2pg, %v6178 : tensor<24xf32>
    %v6180 = stablehlo.reshape %v6123 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6181 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6182 = stablehlo.reduce(%v6180 init: %v6181) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x24x56x56xf32>, tensor<f32>) -> tensor<24xf32>
    %v6183 = stablehlo.constant dense<0.05> : tensor<24xf32>
    %v6184 = stablehlo.multiply %v6182, %v6183 : tensor<24xf32>
    %v6185 = stablehlo.subtract %b2pbt, %v6184 : tensor<24xf32>
    %v6186 = stablehlo.reshape %v174 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6187 = stablehlo.reshape %v6153 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v6188 = stablehlo.transpose %v6186, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v6189 = stablehlo.transpose %v6187, dims = [1, 0, 2, 3] : (tensor<32x24x56x56xf32>) -> tensor<24x32x56x56xf32>
    %v6190 = stablehlo.convolution(%v6188, %v6189)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<24x32x56x56xf32>) -> tensor<96x24x1x1xf32>
    %v6191 = stablehlo.transpose %v6190, dims = [1, 0, 2, 3] : (tensor<96x24x1x1xf32>) -> tensor<24x96x1x1xf32>
    %v6192 = stablehlo.constant dense<0.05> : tensor<24x96x1x1xf32>
    %v6193 = stablehlo.multiply %v6191, %v6192 : tensor<24x96x1x1xf32>
    %v6194 = stablehlo.subtract %b2pW, %v6193 : tensor<24x96x1x1xf32>
    %v6195 = stablehlo.reshape %v157 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6196 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6197 = stablehlo.reduce(%v6195 init: %v6196) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v6198 = stablehlo.constant dense<3136.0> : tensor<32x96xf32>
    %v6199 = stablehlo.divide %v6197, %v6198 : tensor<32x96xf32>
    %v6200 = stablehlo.dot_general %v6199, %b2zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<96x4xf32>) -> tensor<32x4xf32>
    %v6201 = stablehlo.broadcast_in_dim %b2zb1, dims = [1] : (tensor<4xf32>) -> tensor<32x4xf32>
    %v6202 = stablehlo.add %v6200, %v6201 : tensor<32x4xf32>
    %v6203 = stablehlo.logistic %v6202 : tensor<32x4xf32>
    %v6204 = stablehlo.multiply %v6202, %v6203 : tensor<32x4xf32>
    %v6205 = stablehlo.dot_general %v6204, %b2zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<4x96xf32>) -> tensor<32x96xf32>
    %v6206 = stablehlo.broadcast_in_dim %b2zb2, dims = [1] : (tensor<96xf32>) -> tensor<32x96xf32>
    %v6207 = stablehlo.add %v6205, %v6206 : tensor<32x96xf32>
    %v6208 = stablehlo.logistic %v6207 : tensor<32x96xf32>
    %v6209 = stablehlo.reshape %v6158 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6210 = stablehlo.broadcast_in_dim %v6208, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v6211 = stablehlo.multiply %v6210, %v6209 : tensor<32x96x56x56xf32>
    %v6212 = stablehlo.multiply %v6195, %v6209 : tensor<32x96x56x56xf32>
    %v6213 = stablehlo.reduce(%v6212 init: %v6196) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v6214 = stablehlo.constant dense<1.0> : tensor<32x96xf32>
    %v6215 = stablehlo.subtract %v6214, %v6208 : tensor<32x96xf32>
    %v6216 = stablehlo.multiply %v6208, %v6215 : tensor<32x96xf32>
    %v6217 = stablehlo.multiply %v6213, %v6216 : tensor<32x96xf32>
    %v6218 = stablehlo.dot_general %v6217, %b2zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<4x96xf32>) -> tensor<32x4xf32>
    %v6219 = stablehlo.logistic %v6202 : tensor<32x4xf32>
    %v6220 = stablehlo.constant dense<1.0> : tensor<32x4xf32>
    %v6221 = stablehlo.subtract %v6220, %v6219 : tensor<32x4xf32>
    %v6222 = stablehlo.multiply %v6202, %v6221 : tensor<32x4xf32>
    %v6223 = stablehlo.add %v6220, %v6222 : tensor<32x4xf32>
    %v6224 = stablehlo.multiply %v6219, %v6223 : tensor<32x4xf32>
    %v6225 = stablehlo.multiply %v6218, %v6224 : tensor<32x4xf32>
    %v6226 = stablehlo.dot_general %v6225, %b2zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<96x4xf32>) -> tensor<32x96xf32>
    %v6227 = stablehlo.constant dense<3136.0> : tensor<32x96xf32>
    %v6228 = stablehlo.divide %v6226, %v6227 : tensor<32x96xf32>
    %v6229 = stablehlo.broadcast_in_dim %v6228, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v6230 = stablehlo.add %v6211, %v6229 : tensor<32x96x56x56xf32>
    %v6231 = stablehlo.reshape %v6230 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v6232 = stablehlo.reshape %v157 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6233 = stablehlo.reshape %v6158 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6234 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6235 = stablehlo.multiply %v6232, %v6233 : tensor<32x96x56x56xf32>
    %v6236 = stablehlo.reduce(%v6235 init: %v6234) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v6237 = stablehlo.logistic %v170 : tensor<32x96xf32>
    %v6238 = stablehlo.constant dense<1.0> : tensor<32x96xf32>
    %v6239 = stablehlo.subtract %v6238, %v6237 : tensor<32x96xf32>
    %v6240 = stablehlo.multiply %v6237, %v6239 : tensor<32x96xf32>
    %v6241 = stablehlo.multiply %v6236, %v6240 : tensor<32x96xf32>
    %v6242 = stablehlo.dot_general %v167, %v6241, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<32x96xf32>) -> tensor<4x96xf32>
    %v6243 = stablehlo.constant dense<0.05> : tensor<4x96xf32>
    %v6244 = stablehlo.multiply %v6242, %v6243 : tensor<4x96xf32>
    %v6245 = stablehlo.subtract %b2zW2, %v6244 : tensor<4x96xf32>
    %v6246 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6247 = stablehlo.reduce(%v6241 init: %v6246) applies stablehlo.add across dimensions = [0] : (tensor<32x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v6248 = stablehlo.constant dense<0.05> : tensor<96xf32>
    %v6249 = stablehlo.multiply %v6247, %v6248 : tensor<96xf32>
    %v6250 = stablehlo.subtract %b2zb2, %v6249 : tensor<96xf32>
    %v6251 = stablehlo.reshape %v6241 : (tensor<32x96xf32>) -> tensor<32x1x96xf32>
    %v6252 = stablehlo.dot_general %v6251, %b2zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x96xf32>, tensor<4x96xf32>) -> tensor<32x1x4xf32>
    %v6253 = stablehlo.reshape %v6252 : (tensor<32x1x4xf32>) -> tensor<32x4xf32>
    %v6254 = stablehlo.logistic %v165 : tensor<32x4xf32>
    %v6255 = stablehlo.constant dense<1.0> : tensor<32x4xf32>
    %v6256 = stablehlo.subtract %v6255, %v6254 : tensor<32x4xf32>
    %v6257 = stablehlo.multiply %v165, %v6256 : tensor<32x4xf32>
    %v6258 = stablehlo.add %v6255, %v6257 : tensor<32x4xf32>
    %v6259 = stablehlo.multiply %v6254, %v6258 : tensor<32x4xf32>
    %v6260 = stablehlo.multiply %v6253, %v6259 : tensor<32x4xf32>
    %v6261 = stablehlo.dot_general %v162, %v6260, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<32x4xf32>) -> tensor<96x4xf32>
    %v6262 = stablehlo.constant dense<0.05> : tensor<96x4xf32>
    %v6263 = stablehlo.multiply %v6261, %v6262 : tensor<96x4xf32>
    %v6264 = stablehlo.subtract %b2zW1, %v6263 : tensor<96x4xf32>
    %v6265 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6266 = stablehlo.reduce(%v6260 init: %v6265) applies stablehlo.add across dimensions = [0] : (tensor<32x4xf32>, tensor<f32>) -> tensor<4xf32>
    %v6267 = stablehlo.constant dense<0.05> : tensor<4xf32>
    %v6268 = stablehlo.multiply %v6266, %v6267 : tensor<4xf32>
    %v6269 = stablehlo.subtract %b2zb1, %v6268 : tensor<4xf32>
    %v6270 = stablehlo.reshape %v6231 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6271 = stablehlo.reshape %v153 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6272 = stablehlo.logistic %v6271 : tensor<32x96x56x56xf32>
    %v6273 = stablehlo.constant dense<1.0> : tensor<32x96x56x56xf32>
    %v6274 = stablehlo.subtract %v6273, %v6272 : tensor<32x96x56x56xf32>
    %v6275 = stablehlo.multiply %v6271, %v6274 : tensor<32x96x56x56xf32>
    %v6276 = stablehlo.add %v6273, %v6275 : tensor<32x96x56x56xf32>
    %v6277 = stablehlo.multiply %v6272, %v6276 : tensor<32x96x56x56xf32>
    %v6278 = stablehlo.multiply %v6270, %v6277 : tensor<32x96x56x56xf32>
    %v6279 = stablehlo.reshape %v6278 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v6280 = stablehlo.reshape %v133 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6281 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6282 = stablehlo.constant dense<100352.0> : tensor<32x96x56x56xf32>
    %v6283 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v6284 = stablehlo.reduce(%v6280 init: %v6281) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6285 = stablehlo.broadcast_in_dim %v6284, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v6286 = stablehlo.divide %v6285, %v6282 : tensor<32x96x56x56xf32>
    %v6287 = stablehlo.subtract %v6280, %v6286 : tensor<32x96x56x56xf32>
    %v6288 = stablehlo.multiply %v6287, %v6287 : tensor<32x96x56x56xf32>
    %v6289 = stablehlo.reduce(%v6288 init: %v6281) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6290 = stablehlo.broadcast_in_dim %v6289, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v6291 = stablehlo.divide %v6290, %v6282 : tensor<32x96x56x56xf32>
    %v6292 = stablehlo.add %v6291, %v6283 : tensor<32x96x56x56xf32>
    %v6293 = stablehlo.rsqrt %v6292 : tensor<32x96x56x56xf32>
    %v6294 = stablehlo.multiply %v6287, %v6293 : tensor<32x96x56x56xf32>
    %v6295 = stablehlo.broadcast_in_dim %b2dg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v6296 = stablehlo.reshape %v6279 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6297 = stablehlo.multiply %v6295, %v6296 : tensor<32x96x56x56xf32>
    %v6298 = stablehlo.reduce(%v6297 init: %v6281) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6299 = stablehlo.broadcast_in_dim %v6298, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v6300 = stablehlo.multiply %v6294, %v6297 : tensor<32x96x56x56xf32>
    %v6301 = stablehlo.reduce(%v6300 init: %v6281) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6302 = stablehlo.broadcast_in_dim %v6301, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v6303 = stablehlo.multiply %v6297, %v6282 : tensor<32x96x56x56xf32>
    %v6304 = stablehlo.subtract %v6303, %v6299 : tensor<32x96x56x56xf32>
    %v6305 = stablehlo.multiply %v6294, %v6302 : tensor<32x96x56x56xf32>
    %v6306 = stablehlo.subtract %v6304, %v6305 : tensor<32x96x56x56xf32>
    %v6307 = stablehlo.divide %v6293, %v6282 : tensor<32x96x56x56xf32>
    %v6308 = stablehlo.multiply %v6307, %v6306 : tensor<32x96x56x56xf32>
    %v6309 = stablehlo.reshape %v6308 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v6310 = stablehlo.reshape %v6309 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6311 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6312 = stablehlo.pad %v6310, %v6311, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96x112x112xf32>
    %v6313 = stablehlo.reverse %b2dW, dims = [2, 3] : tensor<96x1x3x3xf32>
    %v6314 = stablehlo.convolution(%v6312, %v6313)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x112x112xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x112x112xf32>
    %v6315 = stablehlo.reshape %v6314 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v6316 = stablehlo.reshape %v133 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6317 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6318 = stablehlo.constant dense<100352.0> : tensor<32x96x56x56xf32>
    %v6319 = stablehlo.reduce(%v6316 init: %v6317) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6320 = stablehlo.broadcast_in_dim %v6319, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v6321 = stablehlo.divide %v6320, %v6318 : tensor<32x96x56x56xf32>
    %v6322 = stablehlo.subtract %v6316, %v6321 : tensor<32x96x56x56xf32>
    %v6323 = stablehlo.multiply %v6322, %v6322 : tensor<32x96x56x56xf32>
    %v6324 = stablehlo.reduce(%v6323 init: %v6317) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6325 = stablehlo.broadcast_in_dim %v6324, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v6326 = stablehlo.divide %v6325, %v6318 : tensor<32x96x56x56xf32>
    %v6327 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v6328 = stablehlo.add %v6326, %v6327 : tensor<32x96x56x56xf32>
    %v6329 = stablehlo.rsqrt %v6328 : tensor<32x96x56x56xf32>
    %v6330 = stablehlo.multiply %v6322, %v6329 : tensor<32x96x56x56xf32>
    %v6331 = stablehlo.reshape %v6279 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6332 = stablehlo.multiply %v6331, %v6330 : tensor<32x96x56x56xf32>
    %v6333 = stablehlo.reduce(%v6332 init: %v6317) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6334 = stablehlo.constant dense<0.05> : tensor<96xf32>
    %v6335 = stablehlo.multiply %v6333, %v6334 : tensor<96xf32>
    %v6336 = stablehlo.subtract %b2dg, %v6335 : tensor<96xf32>
    %v6337 = stablehlo.reshape %v6279 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6338 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6339 = stablehlo.reduce(%v6337 init: %v6338) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v6340 = stablehlo.constant dense<0.05> : tensor<96xf32>
    %v6341 = stablehlo.multiply %v6339, %v6340 : tensor<96xf32>
    %v6342 = stablehlo.subtract %b2dbt, %v6341 : tensor<96xf32>
    %v6343 = stablehlo.reshape %v128 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6344 = stablehlo.reshape %v6309 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v6345 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6346 = stablehlo.pad %v6344, %v6345, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96x112x112xf32>
    %v6347 = stablehlo.transpose %v6343, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %v6348 = stablehlo.transpose %v6346, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %v6349 = stablehlo.convolution(%v6347, %v6348)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x112x112xf32>, tensor<96x32x112x112xf32>) -> tensor<1x96x3x3xf32>
    %v6350 = stablehlo.reshape %v6349 : (tensor<1x96x3x3xf32>) -> tensor<96x1x3x3xf32>
    %v6351 = stablehlo.constant dense<0.05> : tensor<96x1x3x3xf32>
    %v6352 = stablehlo.multiply %v6350, %v6351 : tensor<96x1x3x3xf32>
    %v6353 = stablehlo.subtract %b2dW, %v6352 : tensor<96x1x3x3xf32>
    %v6354 = stablehlo.reshape %v6315 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6355 = stablehlo.reshape %v124 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6356 = stablehlo.logistic %v6355 : tensor<32x96x112x112xf32>
    %v6357 = stablehlo.constant dense<1.0> : tensor<32x96x112x112xf32>
    %v6358 = stablehlo.subtract %v6357, %v6356 : tensor<32x96x112x112xf32>
    %v6359 = stablehlo.multiply %v6355, %v6358 : tensor<32x96x112x112xf32>
    %v6360 = stablehlo.add %v6357, %v6359 : tensor<32x96x112x112xf32>
    %v6361 = stablehlo.multiply %v6356, %v6360 : tensor<32x96x112x112xf32>
    %v6362 = stablehlo.multiply %v6354, %v6361 : tensor<32x96x112x112xf32>
    %v6363 = stablehlo.reshape %v6362 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v6364 = stablehlo.reshape %v104 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6365 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6366 = stablehlo.constant dense<401408.0> : tensor<32x96x112x112xf32>
    %v6367 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v6368 = stablehlo.reduce(%v6364 init: %v6365) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6369 = stablehlo.broadcast_in_dim %v6368, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v6370 = stablehlo.divide %v6369, %v6366 : tensor<32x96x112x112xf32>
    %v6371 = stablehlo.subtract %v6364, %v6370 : tensor<32x96x112x112xf32>
    %v6372 = stablehlo.multiply %v6371, %v6371 : tensor<32x96x112x112xf32>
    %v6373 = stablehlo.reduce(%v6372 init: %v6365) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6374 = stablehlo.broadcast_in_dim %v6373, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v6375 = stablehlo.divide %v6374, %v6366 : tensor<32x96x112x112xf32>
    %v6376 = stablehlo.add %v6375, %v6367 : tensor<32x96x112x112xf32>
    %v6377 = stablehlo.rsqrt %v6376 : tensor<32x96x112x112xf32>
    %v6378 = stablehlo.multiply %v6371, %v6377 : tensor<32x96x112x112xf32>
    %v6379 = stablehlo.broadcast_in_dim %b2eg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v6380 = stablehlo.reshape %v6363 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6381 = stablehlo.multiply %v6379, %v6380 : tensor<32x96x112x112xf32>
    %v6382 = stablehlo.reduce(%v6381 init: %v6365) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6383 = stablehlo.broadcast_in_dim %v6382, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v6384 = stablehlo.multiply %v6378, %v6381 : tensor<32x96x112x112xf32>
    %v6385 = stablehlo.reduce(%v6384 init: %v6365) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6386 = stablehlo.broadcast_in_dim %v6385, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v6387 = stablehlo.multiply %v6381, %v6366 : tensor<32x96x112x112xf32>
    %v6388 = stablehlo.subtract %v6387, %v6383 : tensor<32x96x112x112xf32>
    %v6389 = stablehlo.multiply %v6378, %v6386 : tensor<32x96x112x112xf32>
    %v6390 = stablehlo.subtract %v6388, %v6389 : tensor<32x96x112x112xf32>
    %v6391 = stablehlo.divide %v6377, %v6366 : tensor<32x96x112x112xf32>
    %v6392 = stablehlo.multiply %v6391, %v6390 : tensor<32x96x112x112xf32>
    %v6393 = stablehlo.reshape %v6392 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v6394 = stablehlo.reshape %v6393 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6395 = stablehlo.reverse %b2eW, dims = [2, 3] : tensor<96x16x1x1xf32>
    %v6396 = stablehlo.transpose %v6395, dims = [1, 0, 2, 3] : (tensor<96x16x1x1xf32>) -> tensor<16x96x1x1xf32>
    %v6397 = stablehlo.convolution(%v6394, %v6396)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x112x112xf32>, tensor<16x96x1x1xf32>) -> tensor<32x16x112x112xf32>
    %v6398 = stablehlo.reshape %v6397 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v6399 = stablehlo.reshape %v104 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6400 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6401 = stablehlo.constant dense<401408.0> : tensor<32x96x112x112xf32>
    %v6402 = stablehlo.reduce(%v6399 init: %v6400) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6403 = stablehlo.broadcast_in_dim %v6402, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v6404 = stablehlo.divide %v6403, %v6401 : tensor<32x96x112x112xf32>
    %v6405 = stablehlo.subtract %v6399, %v6404 : tensor<32x96x112x112xf32>
    %v6406 = stablehlo.multiply %v6405, %v6405 : tensor<32x96x112x112xf32>
    %v6407 = stablehlo.reduce(%v6406 init: %v6400) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6408 = stablehlo.broadcast_in_dim %v6407, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v6409 = stablehlo.divide %v6408, %v6401 : tensor<32x96x112x112xf32>
    %v6410 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v6411 = stablehlo.add %v6409, %v6410 : tensor<32x96x112x112xf32>
    %v6412 = stablehlo.rsqrt %v6411 : tensor<32x96x112x112xf32>
    %v6413 = stablehlo.multiply %v6405, %v6412 : tensor<32x96x112x112xf32>
    %v6414 = stablehlo.reshape %v6363 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6415 = stablehlo.multiply %v6414, %v6413 : tensor<32x96x112x112xf32>
    %v6416 = stablehlo.reduce(%v6415 init: %v6400) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6417 = stablehlo.constant dense<0.05> : tensor<96xf32>
    %v6418 = stablehlo.multiply %v6416, %v6417 : tensor<96xf32>
    %v6419 = stablehlo.subtract %b2eg, %v6418 : tensor<96xf32>
    %v6420 = stablehlo.reshape %v6363 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6421 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6422 = stablehlo.reduce(%v6420 init: %v6421) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x112x112xf32>, tensor<f32>) -> tensor<96xf32>
    %v6423 = stablehlo.constant dense<0.05> : tensor<96xf32>
    %v6424 = stablehlo.multiply %v6422, %v6423 : tensor<96xf32>
    %v6425 = stablehlo.subtract %b2ebt, %v6424 : tensor<96xf32>
    %v6426 = stablehlo.reshape %v99 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6427 = stablehlo.reshape %v6393 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v6428 = stablehlo.transpose %v6426, dims = [1, 0, 2, 3] : (tensor<32x16x112x112xf32>) -> tensor<16x32x112x112xf32>
    %v6429 = stablehlo.transpose %v6427, dims = [1, 0, 2, 3] : (tensor<32x96x112x112xf32>) -> tensor<96x32x112x112xf32>
    %v6430 = stablehlo.convolution(%v6428, %v6429)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<16x32x112x112xf32>, tensor<96x32x112x112xf32>) -> tensor<16x96x1x1xf32>
    %v6431 = stablehlo.transpose %v6430, dims = [1, 0, 2, 3] : (tensor<16x96x1x1xf32>) -> tensor<96x16x1x1xf32>
    %v6432 = stablehlo.constant dense<0.05> : tensor<96x16x1x1xf32>
    %v6433 = stablehlo.multiply %v6431, %v6432 : tensor<96x16x1x1xf32>
    %v6434 = stablehlo.subtract %b2eW, %v6433 : tensor<96x16x1x1xf32>
    %v6435 = stablehlo.reshape %v79 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6436 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6437 = stablehlo.constant dense<401408.0> : tensor<32x16x112x112xf32>
    %v6438 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v6439 = stablehlo.reduce(%v6435 init: %v6436) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6440 = stablehlo.broadcast_in_dim %v6439, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v6441 = stablehlo.divide %v6440, %v6437 : tensor<32x16x112x112xf32>
    %v6442 = stablehlo.subtract %v6435, %v6441 : tensor<32x16x112x112xf32>
    %v6443 = stablehlo.multiply %v6442, %v6442 : tensor<32x16x112x112xf32>
    %v6444 = stablehlo.reduce(%v6443 init: %v6436) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6445 = stablehlo.broadcast_in_dim %v6444, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v6446 = stablehlo.divide %v6445, %v6437 : tensor<32x16x112x112xf32>
    %v6447 = stablehlo.add %v6446, %v6438 : tensor<32x16x112x112xf32>
    %v6448 = stablehlo.rsqrt %v6447 : tensor<32x16x112x112xf32>
    %v6449 = stablehlo.multiply %v6442, %v6448 : tensor<32x16x112x112xf32>
    %v6450 = stablehlo.broadcast_in_dim %b1pg, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v6451 = stablehlo.reshape %v6398 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6452 = stablehlo.multiply %v6450, %v6451 : tensor<32x16x112x112xf32>
    %v6453 = stablehlo.reduce(%v6452 init: %v6436) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6454 = stablehlo.broadcast_in_dim %v6453, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v6455 = stablehlo.multiply %v6449, %v6452 : tensor<32x16x112x112xf32>
    %v6456 = stablehlo.reduce(%v6455 init: %v6436) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6457 = stablehlo.broadcast_in_dim %v6456, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v6458 = stablehlo.multiply %v6452, %v6437 : tensor<32x16x112x112xf32>
    %v6459 = stablehlo.subtract %v6458, %v6454 : tensor<32x16x112x112xf32>
    %v6460 = stablehlo.multiply %v6449, %v6457 : tensor<32x16x112x112xf32>
    %v6461 = stablehlo.subtract %v6459, %v6460 : tensor<32x16x112x112xf32>
    %v6462 = stablehlo.divide %v6448, %v6437 : tensor<32x16x112x112xf32>
    %v6463 = stablehlo.multiply %v6462, %v6461 : tensor<32x16x112x112xf32>
    %v6464 = stablehlo.reshape %v6463 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v6465 = stablehlo.reshape %v6464 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6466 = stablehlo.reverse %b1pW, dims = [2, 3] : tensor<16x32x1x1xf32>
    %v6467 = stablehlo.transpose %v6466, dims = [1, 0, 2, 3] : (tensor<16x32x1x1xf32>) -> tensor<32x16x1x1xf32>
    %v6468 = stablehlo.convolution(%v6465, %v6467)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x16x112x112xf32>, tensor<32x16x1x1xf32>) -> tensor<32x32x112x112xf32>
    %v6469 = stablehlo.reshape %v6468 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v6470 = stablehlo.reshape %v79 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6471 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6472 = stablehlo.constant dense<401408.0> : tensor<32x16x112x112xf32>
    %v6473 = stablehlo.reduce(%v6470 init: %v6471) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6474 = stablehlo.broadcast_in_dim %v6473, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v6475 = stablehlo.divide %v6474, %v6472 : tensor<32x16x112x112xf32>
    %v6476 = stablehlo.subtract %v6470, %v6475 : tensor<32x16x112x112xf32>
    %v6477 = stablehlo.multiply %v6476, %v6476 : tensor<32x16x112x112xf32>
    %v6478 = stablehlo.reduce(%v6477 init: %v6471) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6479 = stablehlo.broadcast_in_dim %v6478, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v6480 = stablehlo.divide %v6479, %v6472 : tensor<32x16x112x112xf32>
    %v6481 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v6482 = stablehlo.add %v6480, %v6481 : tensor<32x16x112x112xf32>
    %v6483 = stablehlo.rsqrt %v6482 : tensor<32x16x112x112xf32>
    %v6484 = stablehlo.multiply %v6476, %v6483 : tensor<32x16x112x112xf32>
    %v6485 = stablehlo.reshape %v6398 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6486 = stablehlo.multiply %v6485, %v6484 : tensor<32x16x112x112xf32>
    %v6487 = stablehlo.reduce(%v6486 init: %v6471) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6488 = stablehlo.constant dense<0.05> : tensor<16xf32>
    %v6489 = stablehlo.multiply %v6487, %v6488 : tensor<16xf32>
    %v6490 = stablehlo.subtract %b1pg, %v6489 : tensor<16xf32>
    %v6491 = stablehlo.reshape %v6398 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6492 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6493 = stablehlo.reduce(%v6491 init: %v6492) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x16x112x112xf32>, tensor<f32>) -> tensor<16xf32>
    %v6494 = stablehlo.constant dense<0.05> : tensor<16xf32>
    %v6495 = stablehlo.multiply %v6493, %v6494 : tensor<16xf32>
    %v6496 = stablehlo.subtract %b1pbt, %v6495 : tensor<16xf32>
    %v6497 = stablehlo.reshape %v74 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6498 = stablehlo.reshape %v6464 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v6499 = stablehlo.transpose %v6497, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %v6500 = stablehlo.transpose %v6498, dims = [1, 0, 2, 3] : (tensor<32x16x112x112xf32>) -> tensor<16x32x112x112xf32>
    %v6501 = stablehlo.convolution(%v6499, %v6500)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<16x32x112x112xf32>) -> tensor<32x16x1x1xf32>
    %v6502 = stablehlo.transpose %v6501, dims = [1, 0, 2, 3] : (tensor<32x16x1x1xf32>) -> tensor<16x32x1x1xf32>
    %v6503 = stablehlo.constant dense<0.05> : tensor<16x32x1x1xf32>
    %v6504 = stablehlo.multiply %v6502, %v6503 : tensor<16x32x1x1xf32>
    %v6505 = stablehlo.subtract %b1pW, %v6504 : tensor<16x32x1x1xf32>
    %v6506 = stablehlo.reshape %v57 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6507 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6508 = stablehlo.reduce(%v6506 init: %v6507) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v6509 = stablehlo.constant dense<12544.0> : tensor<32x32xf32>
    %v6510 = stablehlo.divide %v6508, %v6509 : tensor<32x32xf32>
    %v6511 = stablehlo.dot_general %v6510, %b1zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x8xf32>) -> tensor<32x8xf32>
    %v6512 = stablehlo.broadcast_in_dim %b1zb1, dims = [1] : (tensor<8xf32>) -> tensor<32x8xf32>
    %v6513 = stablehlo.add %v6511, %v6512 : tensor<32x8xf32>
    %v6514 = stablehlo.logistic %v6513 : tensor<32x8xf32>
    %v6515 = stablehlo.multiply %v6513, %v6514 : tensor<32x8xf32>
    %v6516 = stablehlo.dot_general %v6515, %b1zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x8xf32>, tensor<8x32xf32>) -> tensor<32x32xf32>
    %v6517 = stablehlo.broadcast_in_dim %b1zb2, dims = [1] : (tensor<32xf32>) -> tensor<32x32xf32>
    %v6518 = stablehlo.add %v6516, %v6517 : tensor<32x32xf32>
    %v6519 = stablehlo.logistic %v6518 : tensor<32x32xf32>
    %v6520 = stablehlo.reshape %v6469 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6521 = stablehlo.broadcast_in_dim %v6519, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v6522 = stablehlo.multiply %v6521, %v6520 : tensor<32x32x112x112xf32>
    %v6523 = stablehlo.multiply %v6506, %v6520 : tensor<32x32x112x112xf32>
    %v6524 = stablehlo.reduce(%v6523 init: %v6507) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v6525 = stablehlo.constant dense<1.0> : tensor<32x32xf32>
    %v6526 = stablehlo.subtract %v6525, %v6519 : tensor<32x32xf32>
    %v6527 = stablehlo.multiply %v6519, %v6526 : tensor<32x32xf32>
    %v6528 = stablehlo.multiply %v6524, %v6527 : tensor<32x32xf32>
    %v6529 = stablehlo.dot_general %v6528, %b1zW2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<8x32xf32>) -> tensor<32x8xf32>
    %v6530 = stablehlo.logistic %v6513 : tensor<32x8xf32>
    %v6531 = stablehlo.constant dense<1.0> : tensor<32x8xf32>
    %v6532 = stablehlo.subtract %v6531, %v6530 : tensor<32x8xf32>
    %v6533 = stablehlo.multiply %v6513, %v6532 : tensor<32x8xf32>
    %v6534 = stablehlo.add %v6531, %v6533 : tensor<32x8xf32>
    %v6535 = stablehlo.multiply %v6530, %v6534 : tensor<32x8xf32>
    %v6536 = stablehlo.multiply %v6529, %v6535 : tensor<32x8xf32>
    %v6537 = stablehlo.dot_general %v6536, %b1zW1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x8xf32>, tensor<32x8xf32>) -> tensor<32x32xf32>
    %v6538 = stablehlo.constant dense<12544.0> : tensor<32x32xf32>
    %v6539 = stablehlo.divide %v6537, %v6538 : tensor<32x32xf32>
    %v6540 = stablehlo.broadcast_in_dim %v6539, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v6541 = stablehlo.add %v6522, %v6540 : tensor<32x32x112x112xf32>
    %v6542 = stablehlo.reshape %v6541 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v6543 = stablehlo.reshape %v57 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6544 = stablehlo.reshape %v6469 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6545 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6546 = stablehlo.multiply %v6543, %v6544 : tensor<32x32x112x112xf32>
    %v6547 = stablehlo.reduce(%v6546 init: %v6545) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v6548 = stablehlo.logistic %v70 : tensor<32x32xf32>
    %v6549 = stablehlo.constant dense<1.0> : tensor<32x32xf32>
    %v6550 = stablehlo.subtract %v6549, %v6548 : tensor<32x32xf32>
    %v6551 = stablehlo.multiply %v6548, %v6550 : tensor<32x32xf32>
    %v6552 = stablehlo.multiply %v6547, %v6551 : tensor<32x32xf32>
    %v6553 = stablehlo.dot_general %v67, %v6552, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x8xf32>, tensor<32x32xf32>) -> tensor<8x32xf32>
    %v6554 = stablehlo.constant dense<0.05> : tensor<8x32xf32>
    %v6555 = stablehlo.multiply %v6553, %v6554 : tensor<8x32xf32>
    %v6556 = stablehlo.subtract %b1zW2, %v6555 : tensor<8x32xf32>
    %v6557 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6558 = stablehlo.reduce(%v6552 init: %v6557) applies stablehlo.add across dimensions = [0] : (tensor<32x32xf32>, tensor<f32>) -> tensor<32xf32>
    %v6559 = stablehlo.constant dense<0.05> : tensor<32xf32>
    %v6560 = stablehlo.multiply %v6558, %v6559 : tensor<32xf32>
    %v6561 = stablehlo.subtract %b1zb2, %v6560 : tensor<32xf32>
    %v6562 = stablehlo.reshape %v6552 : (tensor<32x32xf32>) -> tensor<32x1x32xf32>
    %v6563 = stablehlo.dot_general %v6562, %b1zW2, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x32xf32>, tensor<8x32xf32>) -> tensor<32x1x8xf32>
    %v6564 = stablehlo.reshape %v6563 : (tensor<32x1x8xf32>) -> tensor<32x8xf32>
    %v6565 = stablehlo.logistic %v65 : tensor<32x8xf32>
    %v6566 = stablehlo.constant dense<1.0> : tensor<32x8xf32>
    %v6567 = stablehlo.subtract %v6566, %v6565 : tensor<32x8xf32>
    %v6568 = stablehlo.multiply %v65, %v6567 : tensor<32x8xf32>
    %v6569 = stablehlo.add %v6566, %v6568 : tensor<32x8xf32>
    %v6570 = stablehlo.multiply %v6565, %v6569 : tensor<32x8xf32>
    %v6571 = stablehlo.multiply %v6564, %v6570 : tensor<32x8xf32>
    %v6572 = stablehlo.dot_general %v62, %v6571, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x8xf32>) -> tensor<32x8xf32>
    %v6573 = stablehlo.constant dense<0.05> : tensor<32x8xf32>
    %v6574 = stablehlo.multiply %v6572, %v6573 : tensor<32x8xf32>
    %v6575 = stablehlo.subtract %b1zW1, %v6574 : tensor<32x8xf32>
    %v6576 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6577 = stablehlo.reduce(%v6571 init: %v6576) applies stablehlo.add across dimensions = [0] : (tensor<32x8xf32>, tensor<f32>) -> tensor<8xf32>
    %v6578 = stablehlo.constant dense<0.05> : tensor<8xf32>
    %v6579 = stablehlo.multiply %v6577, %v6578 : tensor<8xf32>
    %v6580 = stablehlo.subtract %b1zb1, %v6579 : tensor<8xf32>
    %v6581 = stablehlo.reshape %v6542 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6582 = stablehlo.reshape %v53 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6583 = stablehlo.logistic %v6582 : tensor<32x32x112x112xf32>
    %v6584 = stablehlo.constant dense<1.0> : tensor<32x32x112x112xf32>
    %v6585 = stablehlo.subtract %v6584, %v6583 : tensor<32x32x112x112xf32>
    %v6586 = stablehlo.multiply %v6582, %v6585 : tensor<32x32x112x112xf32>
    %v6587 = stablehlo.add %v6584, %v6586 : tensor<32x32x112x112xf32>
    %v6588 = stablehlo.multiply %v6583, %v6587 : tensor<32x32x112x112xf32>
    %v6589 = stablehlo.multiply %v6581, %v6588 : tensor<32x32x112x112xf32>
    %v6590 = stablehlo.reshape %v6589 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v6591 = stablehlo.reshape %v33 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6592 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6593 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v6594 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v6595 = stablehlo.reduce(%v6591 init: %v6592) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6596 = stablehlo.broadcast_in_dim %v6595, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6597 = stablehlo.divide %v6596, %v6593 : tensor<32x32x112x112xf32>
    %v6598 = stablehlo.subtract %v6591, %v6597 : tensor<32x32x112x112xf32>
    %v6599 = stablehlo.multiply %v6598, %v6598 : tensor<32x32x112x112xf32>
    %v6600 = stablehlo.reduce(%v6599 init: %v6592) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6601 = stablehlo.broadcast_in_dim %v6600, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6602 = stablehlo.divide %v6601, %v6593 : tensor<32x32x112x112xf32>
    %v6603 = stablehlo.add %v6602, %v6594 : tensor<32x32x112x112xf32>
    %v6604 = stablehlo.rsqrt %v6603 : tensor<32x32x112x112xf32>
    %v6605 = stablehlo.multiply %v6598, %v6604 : tensor<32x32x112x112xf32>
    %v6606 = stablehlo.broadcast_in_dim %b1dg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6607 = stablehlo.reshape %v6590 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6608 = stablehlo.multiply %v6606, %v6607 : tensor<32x32x112x112xf32>
    %v6609 = stablehlo.reduce(%v6608 init: %v6592) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6610 = stablehlo.broadcast_in_dim %v6609, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6611 = stablehlo.multiply %v6605, %v6608 : tensor<32x32x112x112xf32>
    %v6612 = stablehlo.reduce(%v6611 init: %v6592) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6613 = stablehlo.broadcast_in_dim %v6612, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6614 = stablehlo.multiply %v6608, %v6593 : tensor<32x32x112x112xf32>
    %v6615 = stablehlo.subtract %v6614, %v6610 : tensor<32x32x112x112xf32>
    %v6616 = stablehlo.multiply %v6605, %v6613 : tensor<32x32x112x112xf32>
    %v6617 = stablehlo.subtract %v6615, %v6616 : tensor<32x32x112x112xf32>
    %v6618 = stablehlo.divide %v6604, %v6593 : tensor<32x32x112x112xf32>
    %v6619 = stablehlo.multiply %v6618, %v6617 : tensor<32x32x112x112xf32>
    %v6620 = stablehlo.reshape %v6619 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v6621 = stablehlo.reshape %v6620 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6622 = stablehlo.reverse %b1dW, dims = [2, 3] : tensor<32x1x3x3xf32>
    %v6623 = stablehlo.convolution(%v6621, %v6622)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 32 : i64} : (tensor<32x32x112x112xf32>, tensor<32x1x3x3xf32>) -> tensor<32x32x112x112xf32>
    %v6624 = stablehlo.reshape %v6623 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v6625 = stablehlo.reshape %v33 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6626 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6627 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v6628 = stablehlo.reduce(%v6625 init: %v6626) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6629 = stablehlo.broadcast_in_dim %v6628, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6630 = stablehlo.divide %v6629, %v6627 : tensor<32x32x112x112xf32>
    %v6631 = stablehlo.subtract %v6625, %v6630 : tensor<32x32x112x112xf32>
    %v6632 = stablehlo.multiply %v6631, %v6631 : tensor<32x32x112x112xf32>
    %v6633 = stablehlo.reduce(%v6632 init: %v6626) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6634 = stablehlo.broadcast_in_dim %v6633, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6635 = stablehlo.divide %v6634, %v6627 : tensor<32x32x112x112xf32>
    %v6636 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v6637 = stablehlo.add %v6635, %v6636 : tensor<32x32x112x112xf32>
    %v6638 = stablehlo.rsqrt %v6637 : tensor<32x32x112x112xf32>
    %v6639 = stablehlo.multiply %v6631, %v6638 : tensor<32x32x112x112xf32>
    %v6640 = stablehlo.reshape %v6590 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6641 = stablehlo.multiply %v6640, %v6639 : tensor<32x32x112x112xf32>
    %v6642 = stablehlo.reduce(%v6641 init: %v6626) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6643 = stablehlo.constant dense<0.05> : tensor<32xf32>
    %v6644 = stablehlo.multiply %v6642, %v6643 : tensor<32xf32>
    %v6645 = stablehlo.subtract %b1dg, %v6644 : tensor<32xf32>
    %v6646 = stablehlo.reshape %v6590 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6647 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6648 = stablehlo.reduce(%v6646 init: %v6647) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6649 = stablehlo.constant dense<0.05> : tensor<32xf32>
    %v6650 = stablehlo.multiply %v6648, %v6649 : tensor<32xf32>
    %v6651 = stablehlo.subtract %b1dbt, %v6650 : tensor<32xf32>
    %v6652 = stablehlo.reshape %v28 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6653 = stablehlo.reshape %v6620 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6654 = stablehlo.transpose %v6652, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %v6655 = stablehlo.transpose %v6653, dims = [1, 0, 2, 3] : (tensor<32x32x112x112xf32>) -> tensor<32x32x112x112xf32>
    %v6656 = stablehlo.convolution(%v6654, %v6655)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 32 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<32x32x112x112xf32>) -> tensor<1x32x3x3xf32>
    %v6657 = stablehlo.reshape %v6656 : (tensor<1x32x3x3xf32>) -> tensor<32x1x3x3xf32>
    %v6658 = stablehlo.constant dense<0.05> : tensor<32x1x3x3xf32>
    %v6659 = stablehlo.multiply %v6657, %v6658 : tensor<32x1x3x3xf32>
    %v6660 = stablehlo.subtract %b1dW, %v6659 : tensor<32x1x3x3xf32>
    %v6661 = stablehlo.reshape %v6624 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6662 = stablehlo.reshape %v24 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6663 = stablehlo.logistic %v6662 : tensor<32x32x112x112xf32>
    %v6664 = stablehlo.constant dense<1.0> : tensor<32x32x112x112xf32>
    %v6665 = stablehlo.subtract %v6664, %v6663 : tensor<32x32x112x112xf32>
    %v6666 = stablehlo.multiply %v6662, %v6665 : tensor<32x32x112x112xf32>
    %v6667 = stablehlo.add %v6664, %v6666 : tensor<32x32x112x112xf32>
    %v6668 = stablehlo.multiply %v6663, %v6667 : tensor<32x32x112x112xf32>
    %v6669 = stablehlo.multiply %v6661, %v6668 : tensor<32x32x112x112xf32>
    %v6670 = stablehlo.reshape %v6669 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v6671 = stablehlo.reshape %v4 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6672 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6673 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v6674 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v6675 = stablehlo.reduce(%v6671 init: %v6672) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6676 = stablehlo.broadcast_in_dim %v6675, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6677 = stablehlo.divide %v6676, %v6673 : tensor<32x32x112x112xf32>
    %v6678 = stablehlo.subtract %v6671, %v6677 : tensor<32x32x112x112xf32>
    %v6679 = stablehlo.multiply %v6678, %v6678 : tensor<32x32x112x112xf32>
    %v6680 = stablehlo.reduce(%v6679 init: %v6672) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6681 = stablehlo.broadcast_in_dim %v6680, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6682 = stablehlo.divide %v6681, %v6673 : tensor<32x32x112x112xf32>
    %v6683 = stablehlo.add %v6682, %v6674 : tensor<32x32x112x112xf32>
    %v6684 = stablehlo.rsqrt %v6683 : tensor<32x32x112x112xf32>
    %v6685 = stablehlo.multiply %v6678, %v6684 : tensor<32x32x112x112xf32>
    %v6686 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6687 = stablehlo.reshape %v6670 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6688 = stablehlo.multiply %v6686, %v6687 : tensor<32x32x112x112xf32>
    %v6689 = stablehlo.reduce(%v6688 init: %v6672) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6690 = stablehlo.broadcast_in_dim %v6689, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6691 = stablehlo.multiply %v6685, %v6688 : tensor<32x32x112x112xf32>
    %v6692 = stablehlo.reduce(%v6691 init: %v6672) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6693 = stablehlo.broadcast_in_dim %v6692, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6694 = stablehlo.multiply %v6688, %v6673 : tensor<32x32x112x112xf32>
    %v6695 = stablehlo.subtract %v6694, %v6690 : tensor<32x32x112x112xf32>
    %v6696 = stablehlo.multiply %v6685, %v6693 : tensor<32x32x112x112xf32>
    %v6697 = stablehlo.subtract %v6695, %v6696 : tensor<32x32x112x112xf32>
    %v6698 = stablehlo.divide %v6684, %v6673 : tensor<32x32x112x112xf32>
    %v6699 = stablehlo.multiply %v6698, %v6697 : tensor<32x32x112x112xf32>
    %v6700 = stablehlo.reshape %v6699 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v6701 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v6702 = stablehlo.reshape %v6700 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6703 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6704 = stablehlo.pad %v6702, %v6703, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32x224x224xf32>
    %v6705 = stablehlo.transpose %v6701, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %v6706 = stablehlo.transpose %v6704, dims = [1, 0, 2, 3] : (tensor<32x32x224x224xf32>) -> tensor<32x32x224x224xf32>
    %v6707 = stablehlo.convolution(%v6705, %v6706)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 2], [0, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<32x32x224x224xf32>) -> tensor<3x32x3x3xf32>
    %v6708 = stablehlo.transpose %v6707, dims = [1, 0, 2, 3] : (tensor<3x32x3x3xf32>) -> tensor<32x3x3x3xf32>
    %v6709 = stablehlo.constant dense<0.05> : tensor<32x3x3x3xf32>
    %v6710 = stablehlo.multiply %v6708, %v6709 : tensor<32x3x3x3xf32>
    %v6711 = stablehlo.subtract %sW, %v6710 : tensor<32x3x3x3xf32>
    %v6712 = stablehlo.reshape %v4 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6713 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6714 = stablehlo.constant dense<401408.0> : tensor<32x32x112x112xf32>
    %v6715 = stablehlo.reduce(%v6712 init: %v6713) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6716 = stablehlo.broadcast_in_dim %v6715, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6717 = stablehlo.divide %v6716, %v6714 : tensor<32x32x112x112xf32>
    %v6718 = stablehlo.subtract %v6712, %v6717 : tensor<32x32x112x112xf32>
    %v6719 = stablehlo.multiply %v6718, %v6718 : tensor<32x32x112x112xf32>
    %v6720 = stablehlo.reduce(%v6719 init: %v6713) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6721 = stablehlo.broadcast_in_dim %v6720, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v6722 = stablehlo.divide %v6721, %v6714 : tensor<32x32x112x112xf32>
    %v6723 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v6724 = stablehlo.add %v6722, %v6723 : tensor<32x32x112x112xf32>
    %v6725 = stablehlo.rsqrt %v6724 : tensor<32x32x112x112xf32>
    %v6726 = stablehlo.multiply %v6718, %v6725 : tensor<32x32x112x112xf32>
    %v6727 = stablehlo.reshape %v6670 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6728 = stablehlo.multiply %v6727, %v6726 : tensor<32x32x112x112xf32>
    %v6729 = stablehlo.reduce(%v6728 init: %v6713) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6730 = stablehlo.constant dense<0.05> : tensor<32xf32>
    %v6731 = stablehlo.multiply %v6729, %v6730 : tensor<32xf32>
    %v6732 = stablehlo.subtract %sg, %v6731 : tensor<32xf32>
    %v6733 = stablehlo.reshape %v6670 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v6734 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6735 = stablehlo.reduce(%v6733 init: %v6734) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32xf32>
    %v6736 = stablehlo.constant dense<0.05> : tensor<32xf32>
    %v6737 = stablehlo.multiply %v6735, %v6736 : tensor<32xf32>
    %v6738 = stablehlo.subtract %sbt, %v6737 : tensor<32xf32>
    return %v6711, %v6732, %v6738, %v6660, %v6645, %v6651, %v6575, %v6580, %v6556, %v6561, %v6505, %v6490, %v6496, %v6434, %v6419, %v6425, %v6353, %v6336, %v6342, %v6264, %v6269, %v6245, %v6250, %v6194, %v6179, %v6185, %v6119, %v6104, %v6110, %v6038, %v6023, %v6029, %v5953, %v5958, %v5934, %v5939, %v5883, %v5868, %v5874, %v5812, %v5797, %v5803, %v5731, %v5714, %v5720, %v5642, %v5647, %v5623, %v5628, %v5572, %v5557, %v5563, %v5497, %v5482, %v5488, %v5416, %v5401, %v5407, %v5331, %v5336, %v5312, %v5317, %v5261, %v5246, %v5252, %v5190, %v5175, %v5181, %v5109, %v5092, %v5098, %v5020, %v5025, %v5001, %v5006, %v4950, %v4935, %v4941, %v4875, %v4860, %v4866, %v4794, %v4779, %v4785, %v4709, %v4714, %v4690, %v4695, %v4639, %v4624, %v4630, %v4564, %v4549, %v4555, %v4483, %v4468, %v4474, %v4398, %v4403, %v4379, %v4384, %v4328, %v4313, %v4319, %v4257, %v4242, %v4248, %v4176, %v4161, %v4167, %v4091, %v4096, %v4072, %v4077, %v4021, %v4006, %v4012, %v3946, %v3931, %v3937, %v3865, %v3850, %v3856, %v3780, %v3785, %v3761, %v3766, %v3710, %v3695, %v3701, %v3635, %v3620, %v3626, %v3554, %v3539, %v3545, %v3469, %v3474, %v3450, %v3455, %v3399, %v3384, %v3390, %v3328, %v3313, %v3319, %v3247, %v3230, %v3236, %v3158, %v3163, %v3139, %v3144, %v3088, %v3073, %v3079, %v3013, %v2998, %v3004, %v2932, %v2917, %v2923, %v2847, %v2852, %v2828, %v2833, %v2777, %v2762, %v2768, %v2702, %v2687, %v2693, %v2621, %v2606, %v2612, %v2536, %v2541, %v2517, %v2522, %v2466, %v2451, %v2457, %v2391, %v2376, %v2382, %v2310, %v2295, %v2301, %v2225, %v2230, %v2206, %v2211, %v2155, %v2140, %v2146, %v2084, %v2069, %v2075, %v2003, %v1988, %v1994, %v1918, %v1923, %v1899, %v1904, %v1848, %v1833, %v1839, %v1777, %v1762, %v1768, %v1687, %v1692 : tensor<32x3x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x8xf32>, tensor<8xf32>, tensor<8x32xf32>, tensor<32xf32>, tensor<16x32x1x1xf32>, tensor<16xf32>, tensor<16xf32>, tensor<96x16x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x4xf32>, tensor<4xf32>, tensor<4x96xf32>, tensor<96xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x3x3xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x6xf32>, tensor<6xf32>, tensor<6x144xf32>, tensor<144xf32>, tensor<24x144x1x1xf32>, tensor<24xf32>, tensor<24xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x1x5x5xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144x6xf32>, tensor<6xf32>, tensor<6x144xf32>, tensor<144xf32>, tensor<40x144x1x1xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240x40x1x1xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x1x5x5xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x10xf32>, tensor<10xf32>, tensor<10x240xf32>, tensor<240xf32>, tensor<40x240x1x1xf32>, tensor<40xf32>, tensor<40xf32>, tensor<240x40x1x1xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x1x3x3xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240x10xf32>, tensor<10xf32>, tensor<10x240xf32>, tensor<240xf32>, tensor<80x240x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x3x3xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<80x480x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x3x3xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<80x480x1x1xf32>, tensor<80xf32>, tensor<80xf32>, tensor<480x80x1x1xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x1x5x5xf32>, tensor<480xf32>, tensor<480xf32>, tensor<480x20xf32>, tensor<20xf32>, tensor<20x480xf32>, tensor<480xf32>, tensor<112x480x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<112x672x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<112x672x1x1xf32>, tensor<112xf32>, tensor<112xf32>, tensor<672x112x1x1xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x1x5x5xf32>, tensor<672xf32>, tensor<672xf32>, tensor<672x28xf32>, tensor<28xf32>, tensor<28x672xf32>, tensor<672xf32>, tensor<192x672x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x5x5xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<192x1152x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1152x192x1x1xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x1x3x3xf32>, tensor<1152xf32>, tensor<1152xf32>, tensor<1152x48xf32>, tensor<48xf32>, tensor<48x1152xf32>, tensor<1152xf32>, tensor<320x1152x1x1xf32>, tensor<320xf32>, tensor<320xf32>, tensor<1280x320x1x1xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280x10xf32>, tensor<10xf32>
  }
}
