module @m {
  func.func @efficientnet_drop_fwd_eval(%x: tensor<32x150528xf32>, %sW: tensor<32x3x3x3xf32>, %sg: tensor<32xf32>, %sbt: tensor<32xf32>, %b1dW: tensor<32x1x3x3xf32>, %b1dg: tensor<32xf32>, %b1dbt: tensor<32xf32>, %b1zW1: tensor<32x8xf32>, %b1zb1: tensor<8xf32>, %b1zW2: tensor<8x32xf32>, %b1zb2: tensor<32xf32>, %b1pW: tensor<16x32x1x1xf32>, %b1pg: tensor<16xf32>, %b1pbt: tensor<16xf32>, %b2eW: tensor<96x16x1x1xf32>, %b2eg: tensor<96xf32>, %b2ebt: tensor<96xf32>, %b2dW: tensor<96x1x3x3xf32>, %b2dg: tensor<96xf32>, %b2dbt: tensor<96xf32>, %b2zW1: tensor<96x4xf32>, %b2zb1: tensor<4xf32>, %b2zW2: tensor<4x96xf32>, %b2zb2: tensor<96xf32>, %b2pW: tensor<24x96x1x1xf32>, %b2pg: tensor<24xf32>, %b2pbt: tensor<24xf32>, %b3eW: tensor<144x24x1x1xf32>, %b3eg: tensor<144xf32>, %b3ebt: tensor<144xf32>, %b3dW: tensor<144x1x3x3xf32>, %b3dg: tensor<144xf32>, %b3dbt: tensor<144xf32>, %b3zW1: tensor<144x6xf32>, %b3zb1: tensor<6xf32>, %b3zW2: tensor<6x144xf32>, %b3zb2: tensor<144xf32>, %b3pW: tensor<24x144x1x1xf32>, %b3pg: tensor<24xf32>, %b3pbt: tensor<24xf32>, %b4eW: tensor<144x24x1x1xf32>, %b4eg: tensor<144xf32>, %b4ebt: tensor<144xf32>, %b4dW: tensor<144x1x5x5xf32>, %b4dg: tensor<144xf32>, %b4dbt: tensor<144xf32>, %b4zW1: tensor<144x6xf32>, %b4zb1: tensor<6xf32>, %b4zW2: tensor<6x144xf32>, %b4zb2: tensor<144xf32>, %b4pW: tensor<40x144x1x1xf32>, %b4pg: tensor<40xf32>, %b4pbt: tensor<40xf32>, %b5eW: tensor<240x40x1x1xf32>, %b5eg: tensor<240xf32>, %b5ebt: tensor<240xf32>, %b5dW: tensor<240x1x5x5xf32>, %b5dg: tensor<240xf32>, %b5dbt: tensor<240xf32>, %b5zW1: tensor<240x10xf32>, %b5zb1: tensor<10xf32>, %b5zW2: tensor<10x240xf32>, %b5zb2: tensor<240xf32>, %b5pW: tensor<40x240x1x1xf32>, %b5pg: tensor<40xf32>, %b5pbt: tensor<40xf32>, %b6eW: tensor<240x40x1x1xf32>, %b6eg: tensor<240xf32>, %b6ebt: tensor<240xf32>, %b6dW: tensor<240x1x3x3xf32>, %b6dg: tensor<240xf32>, %b6dbt: tensor<240xf32>, %b6zW1: tensor<240x10xf32>, %b6zb1: tensor<10xf32>, %b6zW2: tensor<10x240xf32>, %b6zb2: tensor<240xf32>, %b6pW: tensor<80x240x1x1xf32>, %b6pg: tensor<80xf32>, %b6pbt: tensor<80xf32>, %b7eW: tensor<480x80x1x1xf32>, %b7eg: tensor<480xf32>, %b7ebt: tensor<480xf32>, %b7dW: tensor<480x1x3x3xf32>, %b7dg: tensor<480xf32>, %b7dbt: tensor<480xf32>, %b7zW1: tensor<480x20xf32>, %b7zb1: tensor<20xf32>, %b7zW2: tensor<20x480xf32>, %b7zb2: tensor<480xf32>, %b7pW: tensor<80x480x1x1xf32>, %b7pg: tensor<80xf32>, %b7pbt: tensor<80xf32>, %b8eW: tensor<480x80x1x1xf32>, %b8eg: tensor<480xf32>, %b8ebt: tensor<480xf32>, %b8dW: tensor<480x1x3x3xf32>, %b8dg: tensor<480xf32>, %b8dbt: tensor<480xf32>, %b8zW1: tensor<480x20xf32>, %b8zb1: tensor<20xf32>, %b8zW2: tensor<20x480xf32>, %b8zb2: tensor<480xf32>, %b8pW: tensor<80x480x1x1xf32>, %b8pg: tensor<80xf32>, %b8pbt: tensor<80xf32>, %b9eW: tensor<480x80x1x1xf32>, %b9eg: tensor<480xf32>, %b9ebt: tensor<480xf32>, %b9dW: tensor<480x1x5x5xf32>, %b9dg: tensor<480xf32>, %b9dbt: tensor<480xf32>, %b9zW1: tensor<480x20xf32>, %b9zb1: tensor<20xf32>, %b9zW2: tensor<20x480xf32>, %b9zb2: tensor<480xf32>, %b9pW: tensor<112x480x1x1xf32>, %b9pg: tensor<112xf32>, %b9pbt: tensor<112xf32>, %b10eW: tensor<672x112x1x1xf32>, %b10eg: tensor<672xf32>, %b10ebt: tensor<672xf32>, %b10dW: tensor<672x1x5x5xf32>, %b10dg: tensor<672xf32>, %b10dbt: tensor<672xf32>, %b10zW1: tensor<672x28xf32>, %b10zb1: tensor<28xf32>, %b10zW2: tensor<28x672xf32>, %b10zb2: tensor<672xf32>, %b10pW: tensor<112x672x1x1xf32>, %b10pg: tensor<112xf32>, %b10pbt: tensor<112xf32>, %b11eW: tensor<672x112x1x1xf32>, %b11eg: tensor<672xf32>, %b11ebt: tensor<672xf32>, %b11dW: tensor<672x1x5x5xf32>, %b11dg: tensor<672xf32>, %b11dbt: tensor<672xf32>, %b11zW1: tensor<672x28xf32>, %b11zb1: tensor<28xf32>, %b11zW2: tensor<28x672xf32>, %b11zb2: tensor<672xf32>, %b11pW: tensor<112x672x1x1xf32>, %b11pg: tensor<112xf32>, %b11pbt: tensor<112xf32>, %b12eW: tensor<672x112x1x1xf32>, %b12eg: tensor<672xf32>, %b12ebt: tensor<672xf32>, %b12dW: tensor<672x1x5x5xf32>, %b12dg: tensor<672xf32>, %b12dbt: tensor<672xf32>, %b12zW1: tensor<672x28xf32>, %b12zb1: tensor<28xf32>, %b12zW2: tensor<28x672xf32>, %b12zb2: tensor<672xf32>, %b12pW: tensor<192x672x1x1xf32>, %b12pg: tensor<192xf32>, %b12pbt: tensor<192xf32>, %b13eW: tensor<1152x192x1x1xf32>, %b13eg: tensor<1152xf32>, %b13ebt: tensor<1152xf32>, %b13dW: tensor<1152x1x5x5xf32>, %b13dg: tensor<1152xf32>, %b13dbt: tensor<1152xf32>, %b13zW1: tensor<1152x48xf32>, %b13zb1: tensor<48xf32>, %b13zW2: tensor<48x1152xf32>, %b13zb2: tensor<1152xf32>, %b13pW: tensor<192x1152x1x1xf32>, %b13pg: tensor<192xf32>, %b13pbt: tensor<192xf32>, %b14eW: tensor<1152x192x1x1xf32>, %b14eg: tensor<1152xf32>, %b14ebt: tensor<1152xf32>, %b14dW: tensor<1152x1x5x5xf32>, %b14dg: tensor<1152xf32>, %b14dbt: tensor<1152xf32>, %b14zW1: tensor<1152x48xf32>, %b14zb1: tensor<48xf32>, %b14zW2: tensor<48x1152xf32>, %b14zb2: tensor<1152xf32>, %b14pW: tensor<192x1152x1x1xf32>, %b14pg: tensor<192xf32>, %b14pbt: tensor<192xf32>, %b15eW: tensor<1152x192x1x1xf32>, %b15eg: tensor<1152xf32>, %b15ebt: tensor<1152xf32>, %b15dW: tensor<1152x1x5x5xf32>, %b15dg: tensor<1152xf32>, %b15dbt: tensor<1152xf32>, %b15zW1: tensor<1152x48xf32>, %b15zb1: tensor<48xf32>, %b15zW2: tensor<48x1152xf32>, %b15zb2: tensor<1152xf32>, %b15pW: tensor<192x1152x1x1xf32>, %b15pg: tensor<192xf32>, %b15pbt: tensor<192xf32>, %b16eW: tensor<1152x192x1x1xf32>, %b16eg: tensor<1152xf32>, %b16ebt: tensor<1152xf32>, %b16dW: tensor<1152x1x3x3xf32>, %b16dg: tensor<1152xf32>, %b16dbt: tensor<1152xf32>, %b16zW1: tensor<1152x48xf32>, %b16zb1: tensor<48xf32>, %b16zW2: tensor<48x1152xf32>, %b16zb2: tensor<1152xf32>, %b16pW: tensor<320x1152x1x1xf32>, %b16pg: tensor<320xf32>, %b16pbt: tensor<320xf32>, %hW: tensor<1280x320x1x1xf32>, %hg: tensor<1280xf32>, %hbt: tensor<1280xf32>, %Wd: tensor<1280x10xf32>, %bd: tensor<10xf32>, %stnmu: tensor<32xf32>, %stnvar: tensor<32xf32>, %b1dnmu: tensor<32xf32>, %b1dnvar: tensor<32xf32>, %b1pnmu: tensor<16xf32>, %b1pnvar: tensor<16xf32>, %b2enmu: tensor<96xf32>, %b2envar: tensor<96xf32>, %b2dnmu: tensor<96xf32>, %b2dnvar: tensor<96xf32>, %b2pnmu: tensor<24xf32>, %b2pnvar: tensor<24xf32>, %b3enmu: tensor<144xf32>, %b3envar: tensor<144xf32>, %b3dnmu: tensor<144xf32>, %b3dnvar: tensor<144xf32>, %b3pnmu: tensor<24xf32>, %b3pnvar: tensor<24xf32>, %b4enmu: tensor<144xf32>, %b4envar: tensor<144xf32>, %b4dnmu: tensor<144xf32>, %b4dnvar: tensor<144xf32>, %b4pnmu: tensor<40xf32>, %b4pnvar: tensor<40xf32>, %b5enmu: tensor<240xf32>, %b5envar: tensor<240xf32>, %b5dnmu: tensor<240xf32>, %b5dnvar: tensor<240xf32>, %b5pnmu: tensor<40xf32>, %b5pnvar: tensor<40xf32>, %b6enmu: tensor<240xf32>, %b6envar: tensor<240xf32>, %b6dnmu: tensor<240xf32>, %b6dnvar: tensor<240xf32>, %b6pnmu: tensor<80xf32>, %b6pnvar: tensor<80xf32>, %b7enmu: tensor<480xf32>, %b7envar: tensor<480xf32>, %b7dnmu: tensor<480xf32>, %b7dnvar: tensor<480xf32>, %b7pnmu: tensor<80xf32>, %b7pnvar: tensor<80xf32>, %b8enmu: tensor<480xf32>, %b8envar: tensor<480xf32>, %b8dnmu: tensor<480xf32>, %b8dnvar: tensor<480xf32>, %b8pnmu: tensor<80xf32>, %b8pnvar: tensor<80xf32>, %b9enmu: tensor<480xf32>, %b9envar: tensor<480xf32>, %b9dnmu: tensor<480xf32>, %b9dnvar: tensor<480xf32>, %b9pnmu: tensor<112xf32>, %b9pnvar: tensor<112xf32>, %b10enmu: tensor<672xf32>, %b10envar: tensor<672xf32>, %b10dnmu: tensor<672xf32>, %b10dnvar: tensor<672xf32>, %b10pnmu: tensor<112xf32>, %b10pnvar: tensor<112xf32>, %b11enmu: tensor<672xf32>, %b11envar: tensor<672xf32>, %b11dnmu: tensor<672xf32>, %b11dnvar: tensor<672xf32>, %b11pnmu: tensor<112xf32>, %b11pnvar: tensor<112xf32>, %b12enmu: tensor<672xf32>, %b12envar: tensor<672xf32>, %b12dnmu: tensor<672xf32>, %b12dnvar: tensor<672xf32>, %b12pnmu: tensor<192xf32>, %b12pnvar: tensor<192xf32>, %b13enmu: tensor<1152xf32>, %b13envar: tensor<1152xf32>, %b13dnmu: tensor<1152xf32>, %b13dnvar: tensor<1152xf32>, %b13pnmu: tensor<192xf32>, %b13pnvar: tensor<192xf32>, %b14enmu: tensor<1152xf32>, %b14envar: tensor<1152xf32>, %b14dnmu: tensor<1152xf32>, %b14dnvar: tensor<1152xf32>, %b14pnmu: tensor<192xf32>, %b14pnvar: tensor<192xf32>, %b15enmu: tensor<1152xf32>, %b15envar: tensor<1152xf32>, %b15dnmu: tensor<1152xf32>, %b15dnvar: tensor<1152xf32>, %b15pnmu: tensor<192xf32>, %b15pnvar: tensor<192xf32>, %b16enmu: tensor<1152xf32>, %b16envar: tensor<1152xf32>, %b16dnmu: tensor<1152xf32>, %b16dnvar: tensor<1152xf32>, %b16pnmu: tensor<320xf32>, %b16pnvar: tensor<320xf32>, %hnmu: tensor<1280xf32>, %hnvar: tensor<1280xf32>, %dp2: tensor<32xf32>, %dp4: tensor<32xf32>, %dp6: tensor<32xf32>, %dp7: tensor<32xf32>, %dp9: tensor<32xf32>, %dp10: tensor<32xf32>, %dp12: tensor<32xf32>, %dp13: tensor<32xf32>, %dp14: tensor<32xf32>) -> tensor<32x10xf32> {
    // ── EfficientNet-B0 eval forward (running-stats BN): every line is pretty(verified AST node) ──
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
    %v19 = stablehlo.logistic %v18 : tensor<32x32x112x112xf32>
    %v20 = stablehlo.multiply %v18, %v19 : tensor<32x32x112x112xf32>
    %v21 = stablehlo.reshape %v20 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v22 = stablehlo.reshape %v21 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v23 = stablehlo.convolution(%v22, %b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 32 : i64} : (tensor<32x32x112x112xf32>, tensor<32x1x3x3xf32>) -> tensor<32x32x112x112xf32>
    %v24 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v25 = stablehlo.add %v23, %v24 : tensor<32x32x112x112xf32>
    %v26 = stablehlo.reshape %v25 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v27 = stablehlo.reshape %v26 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v28 = stablehlo.broadcast_in_dim %b1dnmu, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v29 = stablehlo.subtract %v27, %v28 : tensor<32x32x112x112xf32>
    %v30 = stablehlo.broadcast_in_dim %b1dnvar, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v31 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v32 = stablehlo.add %v30, %v31 : tensor<32x32x112x112xf32>
    %v33 = stablehlo.rsqrt %v32 : tensor<32x32x112x112xf32>
    %v34 = stablehlo.multiply %v29, %v33 : tensor<32x32x112x112xf32>
    %v35 = stablehlo.broadcast_in_dim %b1dg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v36 = stablehlo.broadcast_in_dim %b1dbt, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v37 = stablehlo.multiply %v34, %v35 : tensor<32x32x112x112xf32>
    %v38 = stablehlo.add %v37, %v36 : tensor<32x32x112x112xf32>
    %v39 = stablehlo.reshape %v38 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v40 = stablehlo.reshape %v39 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v41 = stablehlo.logistic %v40 : tensor<32x32x112x112xf32>
    %v42 = stablehlo.multiply %v40, %v41 : tensor<32x32x112x112xf32>
    %v43 = stablehlo.reshape %v42 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v44 = stablehlo.reshape %v43 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v45 = stablehlo.constant dense<0.0> : tensor<f32>
    %v46 = stablehlo.reduce(%v44 init: %v45) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v47 = stablehlo.constant dense<12544.0> : tensor<32x32xf32>
    %v48 = stablehlo.divide %v46, %v47 : tensor<32x32xf32>
    %v49 = stablehlo.dot_general %v48, %b1zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x8xf32>) -> tensor<32x8xf32>
    %v50 = stablehlo.broadcast_in_dim %b1zb1, dims = [1] : (tensor<8xf32>) -> tensor<32x8xf32>
    %v51 = stablehlo.add %v49, %v50 : tensor<32x8xf32>
    %v52 = stablehlo.logistic %v51 : tensor<32x8xf32>
    %v53 = stablehlo.multiply %v51, %v52 : tensor<32x8xf32>
    %v54 = stablehlo.dot_general %v53, %b1zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x8xf32>, tensor<8x32xf32>) -> tensor<32x32xf32>
    %v55 = stablehlo.broadcast_in_dim %b1zb2, dims = [1] : (tensor<32xf32>) -> tensor<32x32xf32>
    %v56 = stablehlo.add %v54, %v55 : tensor<32x32xf32>
    %v57 = stablehlo.logistic %v56 : tensor<32x32xf32>
    %v58 = stablehlo.broadcast_in_dim %v57, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v59 = stablehlo.multiply %v44, %v58 : tensor<32x32x112x112xf32>
    %v60 = stablehlo.reshape %v59 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v61 = stablehlo.reshape %v60 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v62 = stablehlo.convolution(%v61, %b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<16x32x1x1xf32>) -> tensor<32x16x112x112xf32>
    %v63 = stablehlo.broadcast_in_dim %zb16, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v64 = stablehlo.add %v62, %v63 : tensor<32x16x112x112xf32>
    %v65 = stablehlo.reshape %v64 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v66 = stablehlo.reshape %v65 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v67 = stablehlo.broadcast_in_dim %b1pnmu, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v68 = stablehlo.subtract %v66, %v67 : tensor<32x16x112x112xf32>
    %v69 = stablehlo.broadcast_in_dim %b1pnvar, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v70 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v71 = stablehlo.add %v69, %v70 : tensor<32x16x112x112xf32>
    %v72 = stablehlo.rsqrt %v71 : tensor<32x16x112x112xf32>
    %v73 = stablehlo.multiply %v68, %v72 : tensor<32x16x112x112xf32>
    %v74 = stablehlo.broadcast_in_dim %b1pg, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v75 = stablehlo.broadcast_in_dim %b1pbt, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v76 = stablehlo.multiply %v73, %v74 : tensor<32x16x112x112xf32>
    %v77 = stablehlo.add %v76, %v75 : tensor<32x16x112x112xf32>
    %v78 = stablehlo.reshape %v77 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v79 = stablehlo.reshape %v78 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v80 = stablehlo.convolution(%v79, %b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x16x112x112xf32>, tensor<96x16x1x1xf32>) -> tensor<32x96x112x112xf32>
    %v81 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v82 = stablehlo.add %v80, %v81 : tensor<32x96x112x112xf32>
    %v83 = stablehlo.reshape %v82 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v84 = stablehlo.reshape %v83 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v85 = stablehlo.broadcast_in_dim %b2enmu, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v86 = stablehlo.subtract %v84, %v85 : tensor<32x96x112x112xf32>
    %v87 = stablehlo.broadcast_in_dim %b2envar, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v88 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v89 = stablehlo.add %v87, %v88 : tensor<32x96x112x112xf32>
    %v90 = stablehlo.rsqrt %v89 : tensor<32x96x112x112xf32>
    %v91 = stablehlo.multiply %v86, %v90 : tensor<32x96x112x112xf32>
    %v92 = stablehlo.broadcast_in_dim %b2eg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v93 = stablehlo.broadcast_in_dim %b2ebt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v94 = stablehlo.multiply %v91, %v92 : tensor<32x96x112x112xf32>
    %v95 = stablehlo.add %v94, %v93 : tensor<32x96x112x112xf32>
    %v96 = stablehlo.reshape %v95 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v97 = stablehlo.reshape %v96 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v98 = stablehlo.logistic %v97 : tensor<32x96x112x112xf32>
    %v99 = stablehlo.multiply %v97, %v98 : tensor<32x96x112x112xf32>
    %v100 = stablehlo.reshape %v99 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v101 = stablehlo.reshape %v100 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v102 = stablehlo.convolution(%v101, %b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x112x112xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x56x56xf32>
    %v103 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v104 = stablehlo.add %v102, %v103 : tensor<32x96x56x56xf32>
    %v105 = stablehlo.reshape %v104 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v106 = stablehlo.reshape %v105 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v107 = stablehlo.broadcast_in_dim %b2dnmu, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v108 = stablehlo.subtract %v106, %v107 : tensor<32x96x56x56xf32>
    %v109 = stablehlo.broadcast_in_dim %b2dnvar, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v110 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v111 = stablehlo.add %v109, %v110 : tensor<32x96x56x56xf32>
    %v112 = stablehlo.rsqrt %v111 : tensor<32x96x56x56xf32>
    %v113 = stablehlo.multiply %v108, %v112 : tensor<32x96x56x56xf32>
    %v114 = stablehlo.broadcast_in_dim %b2dg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v115 = stablehlo.broadcast_in_dim %b2dbt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v116 = stablehlo.multiply %v113, %v114 : tensor<32x96x56x56xf32>
    %v117 = stablehlo.add %v116, %v115 : tensor<32x96x56x56xf32>
    %v118 = stablehlo.reshape %v117 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v119 = stablehlo.reshape %v118 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v120 = stablehlo.logistic %v119 : tensor<32x96x56x56xf32>
    %v121 = stablehlo.multiply %v119, %v120 : tensor<32x96x56x56xf32>
    %v122 = stablehlo.reshape %v121 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v123 = stablehlo.reshape %v122 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v124 = stablehlo.constant dense<0.0> : tensor<f32>
    %v125 = stablehlo.reduce(%v123 init: %v124) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v126 = stablehlo.constant dense<3136.0> : tensor<32x96xf32>
    %v127 = stablehlo.divide %v125, %v126 : tensor<32x96xf32>
    %v128 = stablehlo.dot_general %v127, %b2zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<96x4xf32>) -> tensor<32x4xf32>
    %v129 = stablehlo.broadcast_in_dim %b2zb1, dims = [1] : (tensor<4xf32>) -> tensor<32x4xf32>
    %v130 = stablehlo.add %v128, %v129 : tensor<32x4xf32>
    %v131 = stablehlo.logistic %v130 : tensor<32x4xf32>
    %v132 = stablehlo.multiply %v130, %v131 : tensor<32x4xf32>
    %v133 = stablehlo.dot_general %v132, %b2zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<4x96xf32>) -> tensor<32x96xf32>
    %v134 = stablehlo.broadcast_in_dim %b2zb2, dims = [1] : (tensor<96xf32>) -> tensor<32x96xf32>
    %v135 = stablehlo.add %v133, %v134 : tensor<32x96xf32>
    %v136 = stablehlo.logistic %v135 : tensor<32x96xf32>
    %v137 = stablehlo.broadcast_in_dim %v136, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v138 = stablehlo.multiply %v123, %v137 : tensor<32x96x56x56xf32>
    %v139 = stablehlo.reshape %v138 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v140 = stablehlo.reshape %v139 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v141 = stablehlo.convolution(%v140, %b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<24x96x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v142 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v143 = stablehlo.add %v141, %v142 : tensor<32x24x56x56xf32>
    %v144 = stablehlo.reshape %v143 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v145 = stablehlo.reshape %v144 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v146 = stablehlo.broadcast_in_dim %b2pnmu, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v147 = stablehlo.subtract %v145, %v146 : tensor<32x24x56x56xf32>
    %v148 = stablehlo.broadcast_in_dim %b2pnvar, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v149 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v150 = stablehlo.add %v148, %v149 : tensor<32x24x56x56xf32>
    %v151 = stablehlo.rsqrt %v150 : tensor<32x24x56x56xf32>
    %v152 = stablehlo.multiply %v147, %v151 : tensor<32x24x56x56xf32>
    %v153 = stablehlo.broadcast_in_dim %b2pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v154 = stablehlo.broadcast_in_dim %b2pbt, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v155 = stablehlo.multiply %v152, %v153 : tensor<32x24x56x56xf32>
    %v156 = stablehlo.add %v155, %v154 : tensor<32x24x56x56xf32>
    %v157 = stablehlo.reshape %v156 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v158 = stablehlo.reshape %v157 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v159 = stablehlo.convolution(%v158, %b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v160 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v161 = stablehlo.add %v159, %v160 : tensor<32x144x56x56xf32>
    %v162 = stablehlo.reshape %v161 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v163 = stablehlo.reshape %v162 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v164 = stablehlo.broadcast_in_dim %b3enmu, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v165 = stablehlo.subtract %v163, %v164 : tensor<32x144x56x56xf32>
    %v166 = stablehlo.broadcast_in_dim %b3envar, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v167 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v168 = stablehlo.add %v166, %v167 : tensor<32x144x56x56xf32>
    %v169 = stablehlo.rsqrt %v168 : tensor<32x144x56x56xf32>
    %v170 = stablehlo.multiply %v165, %v169 : tensor<32x144x56x56xf32>
    %v171 = stablehlo.broadcast_in_dim %b3eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v172 = stablehlo.broadcast_in_dim %b3ebt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v173 = stablehlo.multiply %v170, %v171 : tensor<32x144x56x56xf32>
    %v174 = stablehlo.add %v173, %v172 : tensor<32x144x56x56xf32>
    %v175 = stablehlo.reshape %v174 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v176 = stablehlo.reshape %v175 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v177 = stablehlo.logistic %v176 : tensor<32x144x56x56xf32>
    %v178 = stablehlo.multiply %v176, %v177 : tensor<32x144x56x56xf32>
    %v179 = stablehlo.reshape %v178 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v180 = stablehlo.reshape %v179 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v181 = stablehlo.convolution(%v180, %b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x56x56xf32>
    %v182 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v183 = stablehlo.add %v181, %v182 : tensor<32x144x56x56xf32>
    %v184 = stablehlo.reshape %v183 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v185 = stablehlo.reshape %v184 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v186 = stablehlo.broadcast_in_dim %b3dnmu, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v187 = stablehlo.subtract %v185, %v186 : tensor<32x144x56x56xf32>
    %v188 = stablehlo.broadcast_in_dim %b3dnvar, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v189 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v190 = stablehlo.add %v188, %v189 : tensor<32x144x56x56xf32>
    %v191 = stablehlo.rsqrt %v190 : tensor<32x144x56x56xf32>
    %v192 = stablehlo.multiply %v187, %v191 : tensor<32x144x56x56xf32>
    %v193 = stablehlo.broadcast_in_dim %b3dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v194 = stablehlo.broadcast_in_dim %b3dbt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v195 = stablehlo.multiply %v192, %v193 : tensor<32x144x56x56xf32>
    %v196 = stablehlo.add %v195, %v194 : tensor<32x144x56x56xf32>
    %v197 = stablehlo.reshape %v196 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v198 = stablehlo.reshape %v197 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v199 = stablehlo.logistic %v198 : tensor<32x144x56x56xf32>
    %v200 = stablehlo.multiply %v198, %v199 : tensor<32x144x56x56xf32>
    %v201 = stablehlo.reshape %v200 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v202 = stablehlo.reshape %v201 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v203 = stablehlo.constant dense<0.0> : tensor<f32>
    %v204 = stablehlo.reduce(%v202 init: %v203) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v205 = stablehlo.constant dense<3136.0> : tensor<32x144xf32>
    %v206 = stablehlo.divide %v204, %v205 : tensor<32x144xf32>
    %v207 = stablehlo.dot_general %v206, %b3zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v208 = stablehlo.broadcast_in_dim %b3zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v209 = stablehlo.add %v207, %v208 : tensor<32x6xf32>
    %v210 = stablehlo.logistic %v209 : tensor<32x6xf32>
    %v211 = stablehlo.multiply %v209, %v210 : tensor<32x6xf32>
    %v212 = stablehlo.dot_general %v211, %b3zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v213 = stablehlo.broadcast_in_dim %b3zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v214 = stablehlo.add %v212, %v213 : tensor<32x144xf32>
    %v215 = stablehlo.logistic %v214 : tensor<32x144xf32>
    %v216 = stablehlo.broadcast_in_dim %v215, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v217 = stablehlo.multiply %v202, %v216 : tensor<32x144x56x56xf32>
    %v218 = stablehlo.reshape %v217 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v219 = stablehlo.reshape %v218 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v220 = stablehlo.convolution(%v219, %b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v221 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v222 = stablehlo.add %v220, %v221 : tensor<32x24x56x56xf32>
    %v223 = stablehlo.reshape %v222 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v224 = stablehlo.reshape %v223 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v225 = stablehlo.broadcast_in_dim %b3pnmu, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v226 = stablehlo.subtract %v224, %v225 : tensor<32x24x56x56xf32>
    %v227 = stablehlo.broadcast_in_dim %b3pnvar, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v228 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v229 = stablehlo.add %v227, %v228 : tensor<32x24x56x56xf32>
    %v230 = stablehlo.rsqrt %v229 : tensor<32x24x56x56xf32>
    %v231 = stablehlo.multiply %v226, %v230 : tensor<32x24x56x56xf32>
    %v232 = stablehlo.broadcast_in_dim %b3pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v233 = stablehlo.broadcast_in_dim %b3pbt, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v234 = stablehlo.multiply %v231, %v232 : tensor<32x24x56x56xf32>
    %v235 = stablehlo.add %v234, %v233 : tensor<32x24x56x56xf32>
    %v236 = stablehlo.reshape %v235 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v237 = stablehlo.reshape %v236 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v238 = stablehlo.broadcast_in_dim %dp2, dims = [0] : (tensor<32xf32>) -> tensor<32x24x56x56xf32>
    %v239 = stablehlo.multiply %v238, %v237 : tensor<32x24x56x56xf32>
    %v240 = stablehlo.reshape %v239 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v241 = stablehlo.reshape %v240 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v242 = stablehlo.reshape %v157 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v243 = stablehlo.add %v241, %v242 : tensor<32x24x56x56xf32>
    %v244 = stablehlo.reshape %v243 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v245 = stablehlo.reshape %v244 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v246 = stablehlo.convolution(%v245, %b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v247 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v248 = stablehlo.add %v246, %v247 : tensor<32x144x56x56xf32>
    %v249 = stablehlo.reshape %v248 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v250 = stablehlo.reshape %v249 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v251 = stablehlo.broadcast_in_dim %b4enmu, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v252 = stablehlo.subtract %v250, %v251 : tensor<32x144x56x56xf32>
    %v253 = stablehlo.broadcast_in_dim %b4envar, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v254 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v255 = stablehlo.add %v253, %v254 : tensor<32x144x56x56xf32>
    %v256 = stablehlo.rsqrt %v255 : tensor<32x144x56x56xf32>
    %v257 = stablehlo.multiply %v252, %v256 : tensor<32x144x56x56xf32>
    %v258 = stablehlo.broadcast_in_dim %b4eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v259 = stablehlo.broadcast_in_dim %b4ebt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v260 = stablehlo.multiply %v257, %v258 : tensor<32x144x56x56xf32>
    %v261 = stablehlo.add %v260, %v259 : tensor<32x144x56x56xf32>
    %v262 = stablehlo.reshape %v261 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v263 = stablehlo.reshape %v262 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v264 = stablehlo.logistic %v263 : tensor<32x144x56x56xf32>
    %v265 = stablehlo.multiply %v263, %v264 : tensor<32x144x56x56xf32>
    %v266 = stablehlo.reshape %v265 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v267 = stablehlo.reshape %v266 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v268 = stablehlo.convolution(%v267, %b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x5x5xf32>) -> tensor<32x144x28x28xf32>
    %v269 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v270 = stablehlo.add %v268, %v269 : tensor<32x144x28x28xf32>
    %v271 = stablehlo.reshape %v270 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v272 = stablehlo.reshape %v271 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v273 = stablehlo.broadcast_in_dim %b4dnmu, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v274 = stablehlo.subtract %v272, %v273 : tensor<32x144x28x28xf32>
    %v275 = stablehlo.broadcast_in_dim %b4dnvar, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v276 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v277 = stablehlo.add %v275, %v276 : tensor<32x144x28x28xf32>
    %v278 = stablehlo.rsqrt %v277 : tensor<32x144x28x28xf32>
    %v279 = stablehlo.multiply %v274, %v278 : tensor<32x144x28x28xf32>
    %v280 = stablehlo.broadcast_in_dim %b4dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v281 = stablehlo.broadcast_in_dim %b4dbt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v282 = stablehlo.multiply %v279, %v280 : tensor<32x144x28x28xf32>
    %v283 = stablehlo.add %v282, %v281 : tensor<32x144x28x28xf32>
    %v284 = stablehlo.reshape %v283 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v285 = stablehlo.reshape %v284 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v286 = stablehlo.logistic %v285 : tensor<32x144x28x28xf32>
    %v287 = stablehlo.multiply %v285, %v286 : tensor<32x144x28x28xf32>
    %v288 = stablehlo.reshape %v287 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v289 = stablehlo.reshape %v288 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v290 = stablehlo.constant dense<0.0> : tensor<f32>
    %v291 = stablehlo.reduce(%v289 init: %v290) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v292 = stablehlo.constant dense<784.0> : tensor<32x144xf32>
    %v293 = stablehlo.divide %v291, %v292 : tensor<32x144xf32>
    %v294 = stablehlo.dot_general %v293, %b4zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v295 = stablehlo.broadcast_in_dim %b4zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v296 = stablehlo.add %v294, %v295 : tensor<32x6xf32>
    %v297 = stablehlo.logistic %v296 : tensor<32x6xf32>
    %v298 = stablehlo.multiply %v296, %v297 : tensor<32x6xf32>
    %v299 = stablehlo.dot_general %v298, %b4zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v300 = stablehlo.broadcast_in_dim %b4zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v301 = stablehlo.add %v299, %v300 : tensor<32x144xf32>
    %v302 = stablehlo.logistic %v301 : tensor<32x144xf32>
    %v303 = stablehlo.broadcast_in_dim %v302, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v304 = stablehlo.multiply %v289, %v303 : tensor<32x144x28x28xf32>
    %v305 = stablehlo.reshape %v304 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v306 = stablehlo.reshape %v305 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v307 = stablehlo.convolution(%v306, %b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x28x28xf32>, tensor<40x144x1x1xf32>) -> tensor<32x40x28x28xf32>
    %v308 = stablehlo.broadcast_in_dim %zb40, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v309 = stablehlo.add %v307, %v308 : tensor<32x40x28x28xf32>
    %v310 = stablehlo.reshape %v309 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v311 = stablehlo.reshape %v310 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v312 = stablehlo.broadcast_in_dim %b4pnmu, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v313 = stablehlo.subtract %v311, %v312 : tensor<32x40x28x28xf32>
    %v314 = stablehlo.broadcast_in_dim %b4pnvar, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v315 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v316 = stablehlo.add %v314, %v315 : tensor<32x40x28x28xf32>
    %v317 = stablehlo.rsqrt %v316 : tensor<32x40x28x28xf32>
    %v318 = stablehlo.multiply %v313, %v317 : tensor<32x40x28x28xf32>
    %v319 = stablehlo.broadcast_in_dim %b4pg, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v320 = stablehlo.broadcast_in_dim %b4pbt, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v321 = stablehlo.multiply %v318, %v319 : tensor<32x40x28x28xf32>
    %v322 = stablehlo.add %v321, %v320 : tensor<32x40x28x28xf32>
    %v323 = stablehlo.reshape %v322 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v324 = stablehlo.reshape %v323 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v325 = stablehlo.convolution(%v324, %b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<32x240x28x28xf32>
    %v326 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v327 = stablehlo.add %v325, %v326 : tensor<32x240x28x28xf32>
    %v328 = stablehlo.reshape %v327 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v329 = stablehlo.reshape %v328 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v330 = stablehlo.broadcast_in_dim %b5enmu, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v331 = stablehlo.subtract %v329, %v330 : tensor<32x240x28x28xf32>
    %v332 = stablehlo.broadcast_in_dim %b5envar, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v333 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v334 = stablehlo.add %v332, %v333 : tensor<32x240x28x28xf32>
    %v335 = stablehlo.rsqrt %v334 : tensor<32x240x28x28xf32>
    %v336 = stablehlo.multiply %v331, %v335 : tensor<32x240x28x28xf32>
    %v337 = stablehlo.broadcast_in_dim %b5eg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v338 = stablehlo.broadcast_in_dim %b5ebt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v339 = stablehlo.multiply %v336, %v337 : tensor<32x240x28x28xf32>
    %v340 = stablehlo.add %v339, %v338 : tensor<32x240x28x28xf32>
    %v341 = stablehlo.reshape %v340 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v342 = stablehlo.reshape %v341 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v343 = stablehlo.logistic %v342 : tensor<32x240x28x28xf32>
    %v344 = stablehlo.multiply %v342, %v343 : tensor<32x240x28x28xf32>
    %v345 = stablehlo.reshape %v344 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v346 = stablehlo.reshape %v345 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v347 = stablehlo.convolution(%v346, %b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x5x5xf32>) -> tensor<32x240x28x28xf32>
    %v348 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v349 = stablehlo.add %v347, %v348 : tensor<32x240x28x28xf32>
    %v350 = stablehlo.reshape %v349 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v351 = stablehlo.reshape %v350 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v352 = stablehlo.broadcast_in_dim %b5dnmu, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v353 = stablehlo.subtract %v351, %v352 : tensor<32x240x28x28xf32>
    %v354 = stablehlo.broadcast_in_dim %b5dnvar, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v355 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v356 = stablehlo.add %v354, %v355 : tensor<32x240x28x28xf32>
    %v357 = stablehlo.rsqrt %v356 : tensor<32x240x28x28xf32>
    %v358 = stablehlo.multiply %v353, %v357 : tensor<32x240x28x28xf32>
    %v359 = stablehlo.broadcast_in_dim %b5dg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v360 = stablehlo.broadcast_in_dim %b5dbt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v361 = stablehlo.multiply %v358, %v359 : tensor<32x240x28x28xf32>
    %v362 = stablehlo.add %v361, %v360 : tensor<32x240x28x28xf32>
    %v363 = stablehlo.reshape %v362 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v364 = stablehlo.reshape %v363 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v365 = stablehlo.logistic %v364 : tensor<32x240x28x28xf32>
    %v366 = stablehlo.multiply %v364, %v365 : tensor<32x240x28x28xf32>
    %v367 = stablehlo.reshape %v366 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v368 = stablehlo.reshape %v367 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v369 = stablehlo.constant dense<0.0> : tensor<f32>
    %v370 = stablehlo.reduce(%v368 init: %v369) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v371 = stablehlo.constant dense<784.0> : tensor<32x240xf32>
    %v372 = stablehlo.divide %v370, %v371 : tensor<32x240xf32>
    %v373 = stablehlo.dot_general %v372, %b5zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v374 = stablehlo.broadcast_in_dim %b5zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v375 = stablehlo.add %v373, %v374 : tensor<32x10xf32>
    %v376 = stablehlo.logistic %v375 : tensor<32x10xf32>
    %v377 = stablehlo.multiply %v375, %v376 : tensor<32x10xf32>
    %v378 = stablehlo.dot_general %v377, %b5zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v379 = stablehlo.broadcast_in_dim %b5zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v380 = stablehlo.add %v378, %v379 : tensor<32x240xf32>
    %v381 = stablehlo.logistic %v380 : tensor<32x240xf32>
    %v382 = stablehlo.broadcast_in_dim %v381, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x28x28xf32>
    %v383 = stablehlo.multiply %v368, %v382 : tensor<32x240x28x28xf32>
    %v384 = stablehlo.reshape %v383 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v385 = stablehlo.reshape %v384 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v386 = stablehlo.convolution(%v385, %b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x28x28xf32>, tensor<40x240x1x1xf32>) -> tensor<32x40x28x28xf32>
    %v387 = stablehlo.broadcast_in_dim %zb40, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v388 = stablehlo.add %v386, %v387 : tensor<32x40x28x28xf32>
    %v389 = stablehlo.reshape %v388 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v390 = stablehlo.reshape %v389 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v391 = stablehlo.broadcast_in_dim %b5pnmu, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v392 = stablehlo.subtract %v390, %v391 : tensor<32x40x28x28xf32>
    %v393 = stablehlo.broadcast_in_dim %b5pnvar, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v394 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v395 = stablehlo.add %v393, %v394 : tensor<32x40x28x28xf32>
    %v396 = stablehlo.rsqrt %v395 : tensor<32x40x28x28xf32>
    %v397 = stablehlo.multiply %v392, %v396 : tensor<32x40x28x28xf32>
    %v398 = stablehlo.broadcast_in_dim %b5pg, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v399 = stablehlo.broadcast_in_dim %b5pbt, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v400 = stablehlo.multiply %v397, %v398 : tensor<32x40x28x28xf32>
    %v401 = stablehlo.add %v400, %v399 : tensor<32x40x28x28xf32>
    %v402 = stablehlo.reshape %v401 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v403 = stablehlo.reshape %v402 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v404 = stablehlo.broadcast_in_dim %dp4, dims = [0] : (tensor<32xf32>) -> tensor<32x40x28x28xf32>
    %v405 = stablehlo.multiply %v404, %v403 : tensor<32x40x28x28xf32>
    %v406 = stablehlo.reshape %v405 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v407 = stablehlo.reshape %v406 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v408 = stablehlo.reshape %v323 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v409 = stablehlo.add %v407, %v408 : tensor<32x40x28x28xf32>
    %v410 = stablehlo.reshape %v409 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v411 = stablehlo.reshape %v410 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v412 = stablehlo.convolution(%v411, %b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<32x240x28x28xf32>
    %v413 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v414 = stablehlo.add %v412, %v413 : tensor<32x240x28x28xf32>
    %v415 = stablehlo.reshape %v414 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v416 = stablehlo.reshape %v415 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v417 = stablehlo.broadcast_in_dim %b6enmu, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v418 = stablehlo.subtract %v416, %v417 : tensor<32x240x28x28xf32>
    %v419 = stablehlo.broadcast_in_dim %b6envar, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v420 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v421 = stablehlo.add %v419, %v420 : tensor<32x240x28x28xf32>
    %v422 = stablehlo.rsqrt %v421 : tensor<32x240x28x28xf32>
    %v423 = stablehlo.multiply %v418, %v422 : tensor<32x240x28x28xf32>
    %v424 = stablehlo.broadcast_in_dim %b6eg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v425 = stablehlo.broadcast_in_dim %b6ebt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v426 = stablehlo.multiply %v423, %v424 : tensor<32x240x28x28xf32>
    %v427 = stablehlo.add %v426, %v425 : tensor<32x240x28x28xf32>
    %v428 = stablehlo.reshape %v427 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v429 = stablehlo.reshape %v428 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v430 = stablehlo.logistic %v429 : tensor<32x240x28x28xf32>
    %v431 = stablehlo.multiply %v429, %v430 : tensor<32x240x28x28xf32>
    %v432 = stablehlo.reshape %v431 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v433 = stablehlo.reshape %v432 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v434 = stablehlo.convolution(%v433, %b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x3x3xf32>) -> tensor<32x240x14x14xf32>
    %v435 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v436 = stablehlo.add %v434, %v435 : tensor<32x240x14x14xf32>
    %v437 = stablehlo.reshape %v436 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v438 = stablehlo.reshape %v437 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v439 = stablehlo.broadcast_in_dim %b6dnmu, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v440 = stablehlo.subtract %v438, %v439 : tensor<32x240x14x14xf32>
    %v441 = stablehlo.broadcast_in_dim %b6dnvar, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v442 = stablehlo.constant dense<1.0e-5> : tensor<32x240x14x14xf32>
    %v443 = stablehlo.add %v441, %v442 : tensor<32x240x14x14xf32>
    %v444 = stablehlo.rsqrt %v443 : tensor<32x240x14x14xf32>
    %v445 = stablehlo.multiply %v440, %v444 : tensor<32x240x14x14xf32>
    %v446 = stablehlo.broadcast_in_dim %b6dg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v447 = stablehlo.broadcast_in_dim %b6dbt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v448 = stablehlo.multiply %v445, %v446 : tensor<32x240x14x14xf32>
    %v449 = stablehlo.add %v448, %v447 : tensor<32x240x14x14xf32>
    %v450 = stablehlo.reshape %v449 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v451 = stablehlo.reshape %v450 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v452 = stablehlo.logistic %v451 : tensor<32x240x14x14xf32>
    %v453 = stablehlo.multiply %v451, %v452 : tensor<32x240x14x14xf32>
    %v454 = stablehlo.reshape %v453 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v455 = stablehlo.reshape %v454 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v456 = stablehlo.constant dense<0.0> : tensor<f32>
    %v457 = stablehlo.reduce(%v455 init: %v456) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v458 = stablehlo.constant dense<196.0> : tensor<32x240xf32>
    %v459 = stablehlo.divide %v457, %v458 : tensor<32x240xf32>
    %v460 = stablehlo.dot_general %v459, %b6zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v461 = stablehlo.broadcast_in_dim %b6zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v462 = stablehlo.add %v460, %v461 : tensor<32x10xf32>
    %v463 = stablehlo.logistic %v462 : tensor<32x10xf32>
    %v464 = stablehlo.multiply %v462, %v463 : tensor<32x10xf32>
    %v465 = stablehlo.dot_general %v464, %b6zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v466 = stablehlo.broadcast_in_dim %b6zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v467 = stablehlo.add %v465, %v466 : tensor<32x240xf32>
    %v468 = stablehlo.logistic %v467 : tensor<32x240xf32>
    %v469 = stablehlo.broadcast_in_dim %v468, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x14x14xf32>
    %v470 = stablehlo.multiply %v455, %v469 : tensor<32x240x14x14xf32>
    %v471 = stablehlo.reshape %v470 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v472 = stablehlo.reshape %v471 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v473 = stablehlo.convolution(%v472, %b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x14x14xf32>, tensor<80x240x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v474 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v475 = stablehlo.add %v473, %v474 : tensor<32x80x14x14xf32>
    %v476 = stablehlo.reshape %v475 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v477 = stablehlo.reshape %v476 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v478 = stablehlo.broadcast_in_dim %b6pnmu, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v479 = stablehlo.subtract %v477, %v478 : tensor<32x80x14x14xf32>
    %v480 = stablehlo.broadcast_in_dim %b6pnvar, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v481 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v482 = stablehlo.add %v480, %v481 : tensor<32x80x14x14xf32>
    %v483 = stablehlo.rsqrt %v482 : tensor<32x80x14x14xf32>
    %v484 = stablehlo.multiply %v479, %v483 : tensor<32x80x14x14xf32>
    %v485 = stablehlo.broadcast_in_dim %b6pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v486 = stablehlo.broadcast_in_dim %b6pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v487 = stablehlo.multiply %v484, %v485 : tensor<32x80x14x14xf32>
    %v488 = stablehlo.add %v487, %v486 : tensor<32x80x14x14xf32>
    %v489 = stablehlo.reshape %v488 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v490 = stablehlo.reshape %v489 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v491 = stablehlo.convolution(%v490, %b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v492 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v493 = stablehlo.add %v491, %v492 : tensor<32x480x14x14xf32>
    %v494 = stablehlo.reshape %v493 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v495 = stablehlo.reshape %v494 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v496 = stablehlo.broadcast_in_dim %b7enmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v497 = stablehlo.subtract %v495, %v496 : tensor<32x480x14x14xf32>
    %v498 = stablehlo.broadcast_in_dim %b7envar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v499 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v500 = stablehlo.add %v498, %v499 : tensor<32x480x14x14xf32>
    %v501 = stablehlo.rsqrt %v500 : tensor<32x480x14x14xf32>
    %v502 = stablehlo.multiply %v497, %v501 : tensor<32x480x14x14xf32>
    %v503 = stablehlo.broadcast_in_dim %b7eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v504 = stablehlo.broadcast_in_dim %b7ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v505 = stablehlo.multiply %v502, %v503 : tensor<32x480x14x14xf32>
    %v506 = stablehlo.add %v505, %v504 : tensor<32x480x14x14xf32>
    %v507 = stablehlo.reshape %v506 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v508 = stablehlo.reshape %v507 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v509 = stablehlo.logistic %v508 : tensor<32x480x14x14xf32>
    %v510 = stablehlo.multiply %v508, %v509 : tensor<32x480x14x14xf32>
    %v511 = stablehlo.reshape %v510 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v512 = stablehlo.reshape %v511 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v513 = stablehlo.convolution(%v512, %b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %v514 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v515 = stablehlo.add %v513, %v514 : tensor<32x480x14x14xf32>
    %v516 = stablehlo.reshape %v515 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v517 = stablehlo.reshape %v516 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v518 = stablehlo.broadcast_in_dim %b7dnmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v519 = stablehlo.subtract %v517, %v518 : tensor<32x480x14x14xf32>
    %v520 = stablehlo.broadcast_in_dim %b7dnvar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v521 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v522 = stablehlo.add %v520, %v521 : tensor<32x480x14x14xf32>
    %v523 = stablehlo.rsqrt %v522 : tensor<32x480x14x14xf32>
    %v524 = stablehlo.multiply %v519, %v523 : tensor<32x480x14x14xf32>
    %v525 = stablehlo.broadcast_in_dim %b7dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v526 = stablehlo.broadcast_in_dim %b7dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v527 = stablehlo.multiply %v524, %v525 : tensor<32x480x14x14xf32>
    %v528 = stablehlo.add %v527, %v526 : tensor<32x480x14x14xf32>
    %v529 = stablehlo.reshape %v528 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v530 = stablehlo.reshape %v529 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v531 = stablehlo.logistic %v530 : tensor<32x480x14x14xf32>
    %v532 = stablehlo.multiply %v530, %v531 : tensor<32x480x14x14xf32>
    %v533 = stablehlo.reshape %v532 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v534 = stablehlo.reshape %v533 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v535 = stablehlo.constant dense<0.0> : tensor<f32>
    %v536 = stablehlo.reduce(%v534 init: %v535) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v537 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v538 = stablehlo.divide %v536, %v537 : tensor<32x480xf32>
    %v539 = stablehlo.dot_general %v538, %b7zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v540 = stablehlo.broadcast_in_dim %b7zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v541 = stablehlo.add %v539, %v540 : tensor<32x20xf32>
    %v542 = stablehlo.logistic %v541 : tensor<32x20xf32>
    %v543 = stablehlo.multiply %v541, %v542 : tensor<32x20xf32>
    %v544 = stablehlo.dot_general %v543, %b7zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v545 = stablehlo.broadcast_in_dim %b7zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v546 = stablehlo.add %v544, %v545 : tensor<32x480xf32>
    %v547 = stablehlo.logistic %v546 : tensor<32x480xf32>
    %v548 = stablehlo.broadcast_in_dim %v547, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v549 = stablehlo.multiply %v534, %v548 : tensor<32x480x14x14xf32>
    %v550 = stablehlo.reshape %v549 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v551 = stablehlo.reshape %v550 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v552 = stablehlo.convolution(%v551, %b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v553 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v554 = stablehlo.add %v552, %v553 : tensor<32x80x14x14xf32>
    %v555 = stablehlo.reshape %v554 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v556 = stablehlo.reshape %v555 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v557 = stablehlo.broadcast_in_dim %b7pnmu, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v558 = stablehlo.subtract %v556, %v557 : tensor<32x80x14x14xf32>
    %v559 = stablehlo.broadcast_in_dim %b7pnvar, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v560 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v561 = stablehlo.add %v559, %v560 : tensor<32x80x14x14xf32>
    %v562 = stablehlo.rsqrt %v561 : tensor<32x80x14x14xf32>
    %v563 = stablehlo.multiply %v558, %v562 : tensor<32x80x14x14xf32>
    %v564 = stablehlo.broadcast_in_dim %b7pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v565 = stablehlo.broadcast_in_dim %b7pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v566 = stablehlo.multiply %v563, %v564 : tensor<32x80x14x14xf32>
    %v567 = stablehlo.add %v566, %v565 : tensor<32x80x14x14xf32>
    %v568 = stablehlo.reshape %v567 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v569 = stablehlo.reshape %v568 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v570 = stablehlo.broadcast_in_dim %dp6, dims = [0] : (tensor<32xf32>) -> tensor<32x80x14x14xf32>
    %v571 = stablehlo.multiply %v570, %v569 : tensor<32x80x14x14xf32>
    %v572 = stablehlo.reshape %v571 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v573 = stablehlo.reshape %v572 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v574 = stablehlo.reshape %v489 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v575 = stablehlo.add %v573, %v574 : tensor<32x80x14x14xf32>
    %v576 = stablehlo.reshape %v575 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v577 = stablehlo.reshape %v576 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v578 = stablehlo.convolution(%v577, %b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v579 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v580 = stablehlo.add %v578, %v579 : tensor<32x480x14x14xf32>
    %v581 = stablehlo.reshape %v580 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v582 = stablehlo.reshape %v581 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v583 = stablehlo.broadcast_in_dim %b8enmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v584 = stablehlo.subtract %v582, %v583 : tensor<32x480x14x14xf32>
    %v585 = stablehlo.broadcast_in_dim %b8envar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v586 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v587 = stablehlo.add %v585, %v586 : tensor<32x480x14x14xf32>
    %v588 = stablehlo.rsqrt %v587 : tensor<32x480x14x14xf32>
    %v589 = stablehlo.multiply %v584, %v588 : tensor<32x480x14x14xf32>
    %v590 = stablehlo.broadcast_in_dim %b8eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v591 = stablehlo.broadcast_in_dim %b8ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v592 = stablehlo.multiply %v589, %v590 : tensor<32x480x14x14xf32>
    %v593 = stablehlo.add %v592, %v591 : tensor<32x480x14x14xf32>
    %v594 = stablehlo.reshape %v593 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v595 = stablehlo.reshape %v594 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v596 = stablehlo.logistic %v595 : tensor<32x480x14x14xf32>
    %v597 = stablehlo.multiply %v595, %v596 : tensor<32x480x14x14xf32>
    %v598 = stablehlo.reshape %v597 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v599 = stablehlo.reshape %v598 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v600 = stablehlo.convolution(%v599, %b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %v601 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v602 = stablehlo.add %v600, %v601 : tensor<32x480x14x14xf32>
    %v603 = stablehlo.reshape %v602 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v604 = stablehlo.reshape %v603 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v605 = stablehlo.broadcast_in_dim %b8dnmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v606 = stablehlo.subtract %v604, %v605 : tensor<32x480x14x14xf32>
    %v607 = stablehlo.broadcast_in_dim %b8dnvar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v608 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v609 = stablehlo.add %v607, %v608 : tensor<32x480x14x14xf32>
    %v610 = stablehlo.rsqrt %v609 : tensor<32x480x14x14xf32>
    %v611 = stablehlo.multiply %v606, %v610 : tensor<32x480x14x14xf32>
    %v612 = stablehlo.broadcast_in_dim %b8dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v613 = stablehlo.broadcast_in_dim %b8dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v614 = stablehlo.multiply %v611, %v612 : tensor<32x480x14x14xf32>
    %v615 = stablehlo.add %v614, %v613 : tensor<32x480x14x14xf32>
    %v616 = stablehlo.reshape %v615 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v617 = stablehlo.reshape %v616 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v618 = stablehlo.logistic %v617 : tensor<32x480x14x14xf32>
    %v619 = stablehlo.multiply %v617, %v618 : tensor<32x480x14x14xf32>
    %v620 = stablehlo.reshape %v619 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v621 = stablehlo.reshape %v620 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v622 = stablehlo.constant dense<0.0> : tensor<f32>
    %v623 = stablehlo.reduce(%v621 init: %v622) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v624 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v625 = stablehlo.divide %v623, %v624 : tensor<32x480xf32>
    %v626 = stablehlo.dot_general %v625, %b8zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v627 = stablehlo.broadcast_in_dim %b8zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v628 = stablehlo.add %v626, %v627 : tensor<32x20xf32>
    %v629 = stablehlo.logistic %v628 : tensor<32x20xf32>
    %v630 = stablehlo.multiply %v628, %v629 : tensor<32x20xf32>
    %v631 = stablehlo.dot_general %v630, %b8zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v632 = stablehlo.broadcast_in_dim %b8zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v633 = stablehlo.add %v631, %v632 : tensor<32x480xf32>
    %v634 = stablehlo.logistic %v633 : tensor<32x480xf32>
    %v635 = stablehlo.broadcast_in_dim %v634, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v636 = stablehlo.multiply %v621, %v635 : tensor<32x480x14x14xf32>
    %v637 = stablehlo.reshape %v636 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v638 = stablehlo.reshape %v637 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v639 = stablehlo.convolution(%v638, %b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v640 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v641 = stablehlo.add %v639, %v640 : tensor<32x80x14x14xf32>
    %v642 = stablehlo.reshape %v641 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v643 = stablehlo.reshape %v642 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v644 = stablehlo.broadcast_in_dim %b8pnmu, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v645 = stablehlo.subtract %v643, %v644 : tensor<32x80x14x14xf32>
    %v646 = stablehlo.broadcast_in_dim %b8pnvar, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v647 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v648 = stablehlo.add %v646, %v647 : tensor<32x80x14x14xf32>
    %v649 = stablehlo.rsqrt %v648 : tensor<32x80x14x14xf32>
    %v650 = stablehlo.multiply %v645, %v649 : tensor<32x80x14x14xf32>
    %v651 = stablehlo.broadcast_in_dim %b8pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v652 = stablehlo.broadcast_in_dim %b8pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v653 = stablehlo.multiply %v650, %v651 : tensor<32x80x14x14xf32>
    %v654 = stablehlo.add %v653, %v652 : tensor<32x80x14x14xf32>
    %v655 = stablehlo.reshape %v654 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v656 = stablehlo.reshape %v655 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v657 = stablehlo.broadcast_in_dim %dp7, dims = [0] : (tensor<32xf32>) -> tensor<32x80x14x14xf32>
    %v658 = stablehlo.multiply %v657, %v656 : tensor<32x80x14x14xf32>
    %v659 = stablehlo.reshape %v658 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v660 = stablehlo.reshape %v659 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v661 = stablehlo.reshape %v576 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v662 = stablehlo.add %v660, %v661 : tensor<32x80x14x14xf32>
    %v663 = stablehlo.reshape %v662 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v664 = stablehlo.reshape %v663 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v665 = stablehlo.convolution(%v664, %b9eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v666 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v667 = stablehlo.add %v665, %v666 : tensor<32x480x14x14xf32>
    %v668 = stablehlo.reshape %v667 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v669 = stablehlo.reshape %v668 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v670 = stablehlo.broadcast_in_dim %b9enmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v671 = stablehlo.subtract %v669, %v670 : tensor<32x480x14x14xf32>
    %v672 = stablehlo.broadcast_in_dim %b9envar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v673 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v674 = stablehlo.add %v672, %v673 : tensor<32x480x14x14xf32>
    %v675 = stablehlo.rsqrt %v674 : tensor<32x480x14x14xf32>
    %v676 = stablehlo.multiply %v671, %v675 : tensor<32x480x14x14xf32>
    %v677 = stablehlo.broadcast_in_dim %b9eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v678 = stablehlo.broadcast_in_dim %b9ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v679 = stablehlo.multiply %v676, %v677 : tensor<32x480x14x14xf32>
    %v680 = stablehlo.add %v679, %v678 : tensor<32x480x14x14xf32>
    %v681 = stablehlo.reshape %v680 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v682 = stablehlo.reshape %v681 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v683 = stablehlo.logistic %v682 : tensor<32x480x14x14xf32>
    %v684 = stablehlo.multiply %v682, %v683 : tensor<32x480x14x14xf32>
    %v685 = stablehlo.reshape %v684 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v686 = stablehlo.reshape %v685 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v687 = stablehlo.convolution(%v686, %b9dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x5x5xf32>) -> tensor<32x480x14x14xf32>
    %v688 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v689 = stablehlo.add %v687, %v688 : tensor<32x480x14x14xf32>
    %v690 = stablehlo.reshape %v689 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v691 = stablehlo.reshape %v690 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v692 = stablehlo.broadcast_in_dim %b9dnmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v693 = stablehlo.subtract %v691, %v692 : tensor<32x480x14x14xf32>
    %v694 = stablehlo.broadcast_in_dim %b9dnvar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v695 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v696 = stablehlo.add %v694, %v695 : tensor<32x480x14x14xf32>
    %v697 = stablehlo.rsqrt %v696 : tensor<32x480x14x14xf32>
    %v698 = stablehlo.multiply %v693, %v697 : tensor<32x480x14x14xf32>
    %v699 = stablehlo.broadcast_in_dim %b9dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v700 = stablehlo.broadcast_in_dim %b9dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v701 = stablehlo.multiply %v698, %v699 : tensor<32x480x14x14xf32>
    %v702 = stablehlo.add %v701, %v700 : tensor<32x480x14x14xf32>
    %v703 = stablehlo.reshape %v702 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v704 = stablehlo.reshape %v703 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v705 = stablehlo.logistic %v704 : tensor<32x480x14x14xf32>
    %v706 = stablehlo.multiply %v704, %v705 : tensor<32x480x14x14xf32>
    %v707 = stablehlo.reshape %v706 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v708 = stablehlo.reshape %v707 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v709 = stablehlo.constant dense<0.0> : tensor<f32>
    %v710 = stablehlo.reduce(%v708 init: %v709) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v711 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v712 = stablehlo.divide %v710, %v711 : tensor<32x480xf32>
    %v713 = stablehlo.dot_general %v712, %b9zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v714 = stablehlo.broadcast_in_dim %b9zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v715 = stablehlo.add %v713, %v714 : tensor<32x20xf32>
    %v716 = stablehlo.logistic %v715 : tensor<32x20xf32>
    %v717 = stablehlo.multiply %v715, %v716 : tensor<32x20xf32>
    %v718 = stablehlo.dot_general %v717, %b9zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v719 = stablehlo.broadcast_in_dim %b9zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v720 = stablehlo.add %v718, %v719 : tensor<32x480xf32>
    %v721 = stablehlo.logistic %v720 : tensor<32x480xf32>
    %v722 = stablehlo.broadcast_in_dim %v721, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v723 = stablehlo.multiply %v708, %v722 : tensor<32x480x14x14xf32>
    %v724 = stablehlo.reshape %v723 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v725 = stablehlo.reshape %v724 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v726 = stablehlo.convolution(%v725, %b9pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<112x480x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v727 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v728 = stablehlo.add %v726, %v727 : tensor<32x112x14x14xf32>
    %v729 = stablehlo.reshape %v728 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v730 = stablehlo.reshape %v729 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v731 = stablehlo.broadcast_in_dim %b9pnmu, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v732 = stablehlo.subtract %v730, %v731 : tensor<32x112x14x14xf32>
    %v733 = stablehlo.broadcast_in_dim %b9pnvar, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v734 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v735 = stablehlo.add %v733, %v734 : tensor<32x112x14x14xf32>
    %v736 = stablehlo.rsqrt %v735 : tensor<32x112x14x14xf32>
    %v737 = stablehlo.multiply %v732, %v736 : tensor<32x112x14x14xf32>
    %v738 = stablehlo.broadcast_in_dim %b9pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v739 = stablehlo.broadcast_in_dim %b9pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v740 = stablehlo.multiply %v737, %v738 : tensor<32x112x14x14xf32>
    %v741 = stablehlo.add %v740, %v739 : tensor<32x112x14x14xf32>
    %v742 = stablehlo.reshape %v741 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v743 = stablehlo.reshape %v742 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v744 = stablehlo.convolution(%v743, %b10eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v745 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v746 = stablehlo.add %v744, %v745 : tensor<32x672x14x14xf32>
    %v747 = stablehlo.reshape %v746 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v748 = stablehlo.reshape %v747 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v749 = stablehlo.broadcast_in_dim %b10enmu, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v750 = stablehlo.subtract %v748, %v749 : tensor<32x672x14x14xf32>
    %v751 = stablehlo.broadcast_in_dim %b10envar, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v752 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v753 = stablehlo.add %v751, %v752 : tensor<32x672x14x14xf32>
    %v754 = stablehlo.rsqrt %v753 : tensor<32x672x14x14xf32>
    %v755 = stablehlo.multiply %v750, %v754 : tensor<32x672x14x14xf32>
    %v756 = stablehlo.broadcast_in_dim %b10eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v757 = stablehlo.broadcast_in_dim %b10ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v758 = stablehlo.multiply %v755, %v756 : tensor<32x672x14x14xf32>
    %v759 = stablehlo.add %v758, %v757 : tensor<32x672x14x14xf32>
    %v760 = stablehlo.reshape %v759 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v761 = stablehlo.reshape %v760 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v762 = stablehlo.logistic %v761 : tensor<32x672x14x14xf32>
    %v763 = stablehlo.multiply %v761, %v762 : tensor<32x672x14x14xf32>
    %v764 = stablehlo.reshape %v763 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v765 = stablehlo.reshape %v764 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v766 = stablehlo.convolution(%v765, %b10dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %v767 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v768 = stablehlo.add %v766, %v767 : tensor<32x672x14x14xf32>
    %v769 = stablehlo.reshape %v768 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v770 = stablehlo.reshape %v769 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v771 = stablehlo.broadcast_in_dim %b10dnmu, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v772 = stablehlo.subtract %v770, %v771 : tensor<32x672x14x14xf32>
    %v773 = stablehlo.broadcast_in_dim %b10dnvar, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v774 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v775 = stablehlo.add %v773, %v774 : tensor<32x672x14x14xf32>
    %v776 = stablehlo.rsqrt %v775 : tensor<32x672x14x14xf32>
    %v777 = stablehlo.multiply %v772, %v776 : tensor<32x672x14x14xf32>
    %v778 = stablehlo.broadcast_in_dim %b10dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v779 = stablehlo.broadcast_in_dim %b10dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v780 = stablehlo.multiply %v777, %v778 : tensor<32x672x14x14xf32>
    %v781 = stablehlo.add %v780, %v779 : tensor<32x672x14x14xf32>
    %v782 = stablehlo.reshape %v781 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v783 = stablehlo.reshape %v782 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v784 = stablehlo.logistic %v783 : tensor<32x672x14x14xf32>
    %v785 = stablehlo.multiply %v783, %v784 : tensor<32x672x14x14xf32>
    %v786 = stablehlo.reshape %v785 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v787 = stablehlo.reshape %v786 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v788 = stablehlo.constant dense<0.0> : tensor<f32>
    %v789 = stablehlo.reduce(%v787 init: %v788) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v790 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v791 = stablehlo.divide %v789, %v790 : tensor<32x672xf32>
    %v792 = stablehlo.dot_general %v791, %b10zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v793 = stablehlo.broadcast_in_dim %b10zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v794 = stablehlo.add %v792, %v793 : tensor<32x28xf32>
    %v795 = stablehlo.logistic %v794 : tensor<32x28xf32>
    %v796 = stablehlo.multiply %v794, %v795 : tensor<32x28xf32>
    %v797 = stablehlo.dot_general %v796, %b10zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v798 = stablehlo.broadcast_in_dim %b10zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v799 = stablehlo.add %v797, %v798 : tensor<32x672xf32>
    %v800 = stablehlo.logistic %v799 : tensor<32x672xf32>
    %v801 = stablehlo.broadcast_in_dim %v800, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v802 = stablehlo.multiply %v787, %v801 : tensor<32x672x14x14xf32>
    %v803 = stablehlo.reshape %v802 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v804 = stablehlo.reshape %v803 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v805 = stablehlo.convolution(%v804, %b10pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v806 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v807 = stablehlo.add %v805, %v806 : tensor<32x112x14x14xf32>
    %v808 = stablehlo.reshape %v807 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v809 = stablehlo.reshape %v808 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v810 = stablehlo.broadcast_in_dim %b10pnmu, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v811 = stablehlo.subtract %v809, %v810 : tensor<32x112x14x14xf32>
    %v812 = stablehlo.broadcast_in_dim %b10pnvar, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v813 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v814 = stablehlo.add %v812, %v813 : tensor<32x112x14x14xf32>
    %v815 = stablehlo.rsqrt %v814 : tensor<32x112x14x14xf32>
    %v816 = stablehlo.multiply %v811, %v815 : tensor<32x112x14x14xf32>
    %v817 = stablehlo.broadcast_in_dim %b10pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v818 = stablehlo.broadcast_in_dim %b10pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v819 = stablehlo.multiply %v816, %v817 : tensor<32x112x14x14xf32>
    %v820 = stablehlo.add %v819, %v818 : tensor<32x112x14x14xf32>
    %v821 = stablehlo.reshape %v820 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v822 = stablehlo.reshape %v821 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v823 = stablehlo.broadcast_in_dim %dp9, dims = [0] : (tensor<32xf32>) -> tensor<32x112x14x14xf32>
    %v824 = stablehlo.multiply %v823, %v822 : tensor<32x112x14x14xf32>
    %v825 = stablehlo.reshape %v824 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v826 = stablehlo.reshape %v825 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v827 = stablehlo.reshape %v742 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v828 = stablehlo.add %v826, %v827 : tensor<32x112x14x14xf32>
    %v829 = stablehlo.reshape %v828 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v830 = stablehlo.reshape %v829 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v831 = stablehlo.convolution(%v830, %b11eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v832 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v833 = stablehlo.add %v831, %v832 : tensor<32x672x14x14xf32>
    %v834 = stablehlo.reshape %v833 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v835 = stablehlo.reshape %v834 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v836 = stablehlo.broadcast_in_dim %b11enmu, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v837 = stablehlo.subtract %v835, %v836 : tensor<32x672x14x14xf32>
    %v838 = stablehlo.broadcast_in_dim %b11envar, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v839 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v840 = stablehlo.add %v838, %v839 : tensor<32x672x14x14xf32>
    %v841 = stablehlo.rsqrt %v840 : tensor<32x672x14x14xf32>
    %v842 = stablehlo.multiply %v837, %v841 : tensor<32x672x14x14xf32>
    %v843 = stablehlo.broadcast_in_dim %b11eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v844 = stablehlo.broadcast_in_dim %b11ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v845 = stablehlo.multiply %v842, %v843 : tensor<32x672x14x14xf32>
    %v846 = stablehlo.add %v845, %v844 : tensor<32x672x14x14xf32>
    %v847 = stablehlo.reshape %v846 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v848 = stablehlo.reshape %v847 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v849 = stablehlo.logistic %v848 : tensor<32x672x14x14xf32>
    %v850 = stablehlo.multiply %v848, %v849 : tensor<32x672x14x14xf32>
    %v851 = stablehlo.reshape %v850 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v852 = stablehlo.reshape %v851 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v853 = stablehlo.convolution(%v852, %b11dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %v854 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v855 = stablehlo.add %v853, %v854 : tensor<32x672x14x14xf32>
    %v856 = stablehlo.reshape %v855 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v857 = stablehlo.reshape %v856 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v858 = stablehlo.broadcast_in_dim %b11dnmu, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v859 = stablehlo.subtract %v857, %v858 : tensor<32x672x14x14xf32>
    %v860 = stablehlo.broadcast_in_dim %b11dnvar, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v861 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v862 = stablehlo.add %v860, %v861 : tensor<32x672x14x14xf32>
    %v863 = stablehlo.rsqrt %v862 : tensor<32x672x14x14xf32>
    %v864 = stablehlo.multiply %v859, %v863 : tensor<32x672x14x14xf32>
    %v865 = stablehlo.broadcast_in_dim %b11dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v866 = stablehlo.broadcast_in_dim %b11dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v867 = stablehlo.multiply %v864, %v865 : tensor<32x672x14x14xf32>
    %v868 = stablehlo.add %v867, %v866 : tensor<32x672x14x14xf32>
    %v869 = stablehlo.reshape %v868 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v870 = stablehlo.reshape %v869 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v871 = stablehlo.logistic %v870 : tensor<32x672x14x14xf32>
    %v872 = stablehlo.multiply %v870, %v871 : tensor<32x672x14x14xf32>
    %v873 = stablehlo.reshape %v872 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v874 = stablehlo.reshape %v873 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v875 = stablehlo.constant dense<0.0> : tensor<f32>
    %v876 = stablehlo.reduce(%v874 init: %v875) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v877 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v878 = stablehlo.divide %v876, %v877 : tensor<32x672xf32>
    %v879 = stablehlo.dot_general %v878, %b11zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v880 = stablehlo.broadcast_in_dim %b11zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v881 = stablehlo.add %v879, %v880 : tensor<32x28xf32>
    %v882 = stablehlo.logistic %v881 : tensor<32x28xf32>
    %v883 = stablehlo.multiply %v881, %v882 : tensor<32x28xf32>
    %v884 = stablehlo.dot_general %v883, %b11zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v885 = stablehlo.broadcast_in_dim %b11zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v886 = stablehlo.add %v884, %v885 : tensor<32x672xf32>
    %v887 = stablehlo.logistic %v886 : tensor<32x672xf32>
    %v888 = stablehlo.broadcast_in_dim %v887, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v889 = stablehlo.multiply %v874, %v888 : tensor<32x672x14x14xf32>
    %v890 = stablehlo.reshape %v889 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v891 = stablehlo.reshape %v890 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v892 = stablehlo.convolution(%v891, %b11pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v893 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v894 = stablehlo.add %v892, %v893 : tensor<32x112x14x14xf32>
    %v895 = stablehlo.reshape %v894 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v896 = stablehlo.reshape %v895 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v897 = stablehlo.broadcast_in_dim %b11pnmu, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v898 = stablehlo.subtract %v896, %v897 : tensor<32x112x14x14xf32>
    %v899 = stablehlo.broadcast_in_dim %b11pnvar, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v900 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v901 = stablehlo.add %v899, %v900 : tensor<32x112x14x14xf32>
    %v902 = stablehlo.rsqrt %v901 : tensor<32x112x14x14xf32>
    %v903 = stablehlo.multiply %v898, %v902 : tensor<32x112x14x14xf32>
    %v904 = stablehlo.broadcast_in_dim %b11pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v905 = stablehlo.broadcast_in_dim %b11pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v906 = stablehlo.multiply %v903, %v904 : tensor<32x112x14x14xf32>
    %v907 = stablehlo.add %v906, %v905 : tensor<32x112x14x14xf32>
    %v908 = stablehlo.reshape %v907 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v909 = stablehlo.reshape %v908 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v910 = stablehlo.broadcast_in_dim %dp10, dims = [0] : (tensor<32xf32>) -> tensor<32x112x14x14xf32>
    %v911 = stablehlo.multiply %v910, %v909 : tensor<32x112x14x14xf32>
    %v912 = stablehlo.reshape %v911 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v913 = stablehlo.reshape %v912 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v914 = stablehlo.reshape %v829 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v915 = stablehlo.add %v913, %v914 : tensor<32x112x14x14xf32>
    %v916 = stablehlo.reshape %v915 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v917 = stablehlo.reshape %v916 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v918 = stablehlo.convolution(%v917, %b12eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v919 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v920 = stablehlo.add %v918, %v919 : tensor<32x672x14x14xf32>
    %v921 = stablehlo.reshape %v920 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v922 = stablehlo.reshape %v921 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v923 = stablehlo.broadcast_in_dim %b12enmu, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v924 = stablehlo.subtract %v922, %v923 : tensor<32x672x14x14xf32>
    %v925 = stablehlo.broadcast_in_dim %b12envar, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v926 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v927 = stablehlo.add %v925, %v926 : tensor<32x672x14x14xf32>
    %v928 = stablehlo.rsqrt %v927 : tensor<32x672x14x14xf32>
    %v929 = stablehlo.multiply %v924, %v928 : tensor<32x672x14x14xf32>
    %v930 = stablehlo.broadcast_in_dim %b12eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v931 = stablehlo.broadcast_in_dim %b12ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v932 = stablehlo.multiply %v929, %v930 : tensor<32x672x14x14xf32>
    %v933 = stablehlo.add %v932, %v931 : tensor<32x672x14x14xf32>
    %v934 = stablehlo.reshape %v933 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v935 = stablehlo.reshape %v934 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v936 = stablehlo.logistic %v935 : tensor<32x672x14x14xf32>
    %v937 = stablehlo.multiply %v935, %v936 : tensor<32x672x14x14xf32>
    %v938 = stablehlo.reshape %v937 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v939 = stablehlo.reshape %v938 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v940 = stablehlo.convolution(%v939, %b12dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x7x7xf32>
    %v941 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v942 = stablehlo.add %v940, %v941 : tensor<32x672x7x7xf32>
    %v943 = stablehlo.reshape %v942 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v944 = stablehlo.reshape %v943 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v945 = stablehlo.broadcast_in_dim %b12dnmu, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v946 = stablehlo.subtract %v944, %v945 : tensor<32x672x7x7xf32>
    %v947 = stablehlo.broadcast_in_dim %b12dnvar, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v948 = stablehlo.constant dense<1.0e-5> : tensor<32x672x7x7xf32>
    %v949 = stablehlo.add %v947, %v948 : tensor<32x672x7x7xf32>
    %v950 = stablehlo.rsqrt %v949 : tensor<32x672x7x7xf32>
    %v951 = stablehlo.multiply %v946, %v950 : tensor<32x672x7x7xf32>
    %v952 = stablehlo.broadcast_in_dim %b12dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v953 = stablehlo.broadcast_in_dim %b12dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v954 = stablehlo.multiply %v951, %v952 : tensor<32x672x7x7xf32>
    %v955 = stablehlo.add %v954, %v953 : tensor<32x672x7x7xf32>
    %v956 = stablehlo.reshape %v955 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v957 = stablehlo.reshape %v956 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v958 = stablehlo.logistic %v957 : tensor<32x672x7x7xf32>
    %v959 = stablehlo.multiply %v957, %v958 : tensor<32x672x7x7xf32>
    %v960 = stablehlo.reshape %v959 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v961 = stablehlo.reshape %v960 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v962 = stablehlo.constant dense<0.0> : tensor<f32>
    %v963 = stablehlo.reduce(%v961 init: %v962) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v964 = stablehlo.constant dense<49.0> : tensor<32x672xf32>
    %v965 = stablehlo.divide %v963, %v964 : tensor<32x672xf32>
    %v966 = stablehlo.dot_general %v965, %b12zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v967 = stablehlo.broadcast_in_dim %b12zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v968 = stablehlo.add %v966, %v967 : tensor<32x28xf32>
    %v969 = stablehlo.logistic %v968 : tensor<32x28xf32>
    %v970 = stablehlo.multiply %v968, %v969 : tensor<32x28xf32>
    %v971 = stablehlo.dot_general %v970, %b12zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v972 = stablehlo.broadcast_in_dim %b12zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v973 = stablehlo.add %v971, %v972 : tensor<32x672xf32>
    %v974 = stablehlo.logistic %v973 : tensor<32x672xf32>
    %v975 = stablehlo.broadcast_in_dim %v974, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x7x7xf32>
    %v976 = stablehlo.multiply %v961, %v975 : tensor<32x672x7x7xf32>
    %v977 = stablehlo.reshape %v976 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v978 = stablehlo.reshape %v977 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v979 = stablehlo.convolution(%v978, %b12pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x7x7xf32>, tensor<192x672x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v980 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v981 = stablehlo.add %v979, %v980 : tensor<32x192x7x7xf32>
    %v982 = stablehlo.reshape %v981 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v983 = stablehlo.reshape %v982 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v984 = stablehlo.broadcast_in_dim %b12pnmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v985 = stablehlo.subtract %v983, %v984 : tensor<32x192x7x7xf32>
    %v986 = stablehlo.broadcast_in_dim %b12pnvar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v987 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v988 = stablehlo.add %v986, %v987 : tensor<32x192x7x7xf32>
    %v989 = stablehlo.rsqrt %v988 : tensor<32x192x7x7xf32>
    %v990 = stablehlo.multiply %v985, %v989 : tensor<32x192x7x7xf32>
    %v991 = stablehlo.broadcast_in_dim %b12pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v992 = stablehlo.broadcast_in_dim %b12pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v993 = stablehlo.multiply %v990, %v991 : tensor<32x192x7x7xf32>
    %v994 = stablehlo.add %v993, %v992 : tensor<32x192x7x7xf32>
    %v995 = stablehlo.reshape %v994 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v996 = stablehlo.reshape %v995 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v997 = stablehlo.convolution(%v996, %b13eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v998 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v999 = stablehlo.add %v997, %v998 : tensor<32x1152x7x7xf32>
    %v1000 = stablehlo.reshape %v999 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1001 = stablehlo.reshape %v1000 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1002 = stablehlo.broadcast_in_dim %b13enmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1003 = stablehlo.subtract %v1001, %v1002 : tensor<32x1152x7x7xf32>
    %v1004 = stablehlo.broadcast_in_dim %b13envar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1005 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1006 = stablehlo.add %v1004, %v1005 : tensor<32x1152x7x7xf32>
    %v1007 = stablehlo.rsqrt %v1006 : tensor<32x1152x7x7xf32>
    %v1008 = stablehlo.multiply %v1003, %v1007 : tensor<32x1152x7x7xf32>
    %v1009 = stablehlo.broadcast_in_dim %b13eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1010 = stablehlo.broadcast_in_dim %b13ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1011 = stablehlo.multiply %v1008, %v1009 : tensor<32x1152x7x7xf32>
    %v1012 = stablehlo.add %v1011, %v1010 : tensor<32x1152x7x7xf32>
    %v1013 = stablehlo.reshape %v1012 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1014 = stablehlo.reshape %v1013 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1015 = stablehlo.logistic %v1014 : tensor<32x1152x7x7xf32>
    %v1016 = stablehlo.multiply %v1014, %v1015 : tensor<32x1152x7x7xf32>
    %v1017 = stablehlo.reshape %v1016 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1018 = stablehlo.reshape %v1017 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1019 = stablehlo.convolution(%v1018, %b13dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v1020 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1021 = stablehlo.add %v1019, %v1020 : tensor<32x1152x7x7xf32>
    %v1022 = stablehlo.reshape %v1021 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1023 = stablehlo.reshape %v1022 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1024 = stablehlo.broadcast_in_dim %b13dnmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1025 = stablehlo.subtract %v1023, %v1024 : tensor<32x1152x7x7xf32>
    %v1026 = stablehlo.broadcast_in_dim %b13dnvar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1027 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1028 = stablehlo.add %v1026, %v1027 : tensor<32x1152x7x7xf32>
    %v1029 = stablehlo.rsqrt %v1028 : tensor<32x1152x7x7xf32>
    %v1030 = stablehlo.multiply %v1025, %v1029 : tensor<32x1152x7x7xf32>
    %v1031 = stablehlo.broadcast_in_dim %b13dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1032 = stablehlo.broadcast_in_dim %b13dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1033 = stablehlo.multiply %v1030, %v1031 : tensor<32x1152x7x7xf32>
    %v1034 = stablehlo.add %v1033, %v1032 : tensor<32x1152x7x7xf32>
    %v1035 = stablehlo.reshape %v1034 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1036 = stablehlo.reshape %v1035 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1037 = stablehlo.logistic %v1036 : tensor<32x1152x7x7xf32>
    %v1038 = stablehlo.multiply %v1036, %v1037 : tensor<32x1152x7x7xf32>
    %v1039 = stablehlo.reshape %v1038 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1040 = stablehlo.reshape %v1039 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1041 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1042 = stablehlo.reduce(%v1040 init: %v1041) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1043 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1044 = stablehlo.divide %v1042, %v1043 : tensor<32x1152xf32>
    %v1045 = stablehlo.dot_general %v1044, %b13zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1046 = stablehlo.broadcast_in_dim %b13zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1047 = stablehlo.add %v1045, %v1046 : tensor<32x48xf32>
    %v1048 = stablehlo.logistic %v1047 : tensor<32x48xf32>
    %v1049 = stablehlo.multiply %v1047, %v1048 : tensor<32x48xf32>
    %v1050 = stablehlo.dot_general %v1049, %b13zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1051 = stablehlo.broadcast_in_dim %b13zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1052 = stablehlo.add %v1050, %v1051 : tensor<32x1152xf32>
    %v1053 = stablehlo.logistic %v1052 : tensor<32x1152xf32>
    %v1054 = stablehlo.broadcast_in_dim %v1053, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1055 = stablehlo.multiply %v1040, %v1054 : tensor<32x1152x7x7xf32>
    %v1056 = stablehlo.reshape %v1055 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1057 = stablehlo.reshape %v1056 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1058 = stablehlo.convolution(%v1057, %b13pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1059 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1060 = stablehlo.add %v1058, %v1059 : tensor<32x192x7x7xf32>
    %v1061 = stablehlo.reshape %v1060 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1062 = stablehlo.reshape %v1061 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1063 = stablehlo.broadcast_in_dim %b13pnmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1064 = stablehlo.subtract %v1062, %v1063 : tensor<32x192x7x7xf32>
    %v1065 = stablehlo.broadcast_in_dim %b13pnvar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1066 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1067 = stablehlo.add %v1065, %v1066 : tensor<32x192x7x7xf32>
    %v1068 = stablehlo.rsqrt %v1067 : tensor<32x192x7x7xf32>
    %v1069 = stablehlo.multiply %v1064, %v1068 : tensor<32x192x7x7xf32>
    %v1070 = stablehlo.broadcast_in_dim %b13pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1071 = stablehlo.broadcast_in_dim %b13pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1072 = stablehlo.multiply %v1069, %v1070 : tensor<32x192x7x7xf32>
    %v1073 = stablehlo.add %v1072, %v1071 : tensor<32x192x7x7xf32>
    %v1074 = stablehlo.reshape %v1073 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1075 = stablehlo.reshape %v1074 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1076 = stablehlo.broadcast_in_dim %dp12, dims = [0] : (tensor<32xf32>) -> tensor<32x192x7x7xf32>
    %v1077 = stablehlo.multiply %v1076, %v1075 : tensor<32x192x7x7xf32>
    %v1078 = stablehlo.reshape %v1077 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1079 = stablehlo.reshape %v1078 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1080 = stablehlo.reshape %v995 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1081 = stablehlo.add %v1079, %v1080 : tensor<32x192x7x7xf32>
    %v1082 = stablehlo.reshape %v1081 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1083 = stablehlo.reshape %v1082 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1084 = stablehlo.convolution(%v1083, %b14eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1085 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1086 = stablehlo.add %v1084, %v1085 : tensor<32x1152x7x7xf32>
    %v1087 = stablehlo.reshape %v1086 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1088 = stablehlo.reshape %v1087 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1089 = stablehlo.broadcast_in_dim %b14enmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1090 = stablehlo.subtract %v1088, %v1089 : tensor<32x1152x7x7xf32>
    %v1091 = stablehlo.broadcast_in_dim %b14envar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1092 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1093 = stablehlo.add %v1091, %v1092 : tensor<32x1152x7x7xf32>
    %v1094 = stablehlo.rsqrt %v1093 : tensor<32x1152x7x7xf32>
    %v1095 = stablehlo.multiply %v1090, %v1094 : tensor<32x1152x7x7xf32>
    %v1096 = stablehlo.broadcast_in_dim %b14eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1097 = stablehlo.broadcast_in_dim %b14ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1098 = stablehlo.multiply %v1095, %v1096 : tensor<32x1152x7x7xf32>
    %v1099 = stablehlo.add %v1098, %v1097 : tensor<32x1152x7x7xf32>
    %v1100 = stablehlo.reshape %v1099 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1101 = stablehlo.reshape %v1100 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1102 = stablehlo.logistic %v1101 : tensor<32x1152x7x7xf32>
    %v1103 = stablehlo.multiply %v1101, %v1102 : tensor<32x1152x7x7xf32>
    %v1104 = stablehlo.reshape %v1103 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1105 = stablehlo.reshape %v1104 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1106 = stablehlo.convolution(%v1105, %b14dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v1107 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1108 = stablehlo.add %v1106, %v1107 : tensor<32x1152x7x7xf32>
    %v1109 = stablehlo.reshape %v1108 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1110 = stablehlo.reshape %v1109 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1111 = stablehlo.broadcast_in_dim %b14dnmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1112 = stablehlo.subtract %v1110, %v1111 : tensor<32x1152x7x7xf32>
    %v1113 = stablehlo.broadcast_in_dim %b14dnvar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1114 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1115 = stablehlo.add %v1113, %v1114 : tensor<32x1152x7x7xf32>
    %v1116 = stablehlo.rsqrt %v1115 : tensor<32x1152x7x7xf32>
    %v1117 = stablehlo.multiply %v1112, %v1116 : tensor<32x1152x7x7xf32>
    %v1118 = stablehlo.broadcast_in_dim %b14dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1119 = stablehlo.broadcast_in_dim %b14dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1120 = stablehlo.multiply %v1117, %v1118 : tensor<32x1152x7x7xf32>
    %v1121 = stablehlo.add %v1120, %v1119 : tensor<32x1152x7x7xf32>
    %v1122 = stablehlo.reshape %v1121 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1123 = stablehlo.reshape %v1122 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1124 = stablehlo.logistic %v1123 : tensor<32x1152x7x7xf32>
    %v1125 = stablehlo.multiply %v1123, %v1124 : tensor<32x1152x7x7xf32>
    %v1126 = stablehlo.reshape %v1125 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1127 = stablehlo.reshape %v1126 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1128 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1129 = stablehlo.reduce(%v1127 init: %v1128) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1130 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1131 = stablehlo.divide %v1129, %v1130 : tensor<32x1152xf32>
    %v1132 = stablehlo.dot_general %v1131, %b14zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1133 = stablehlo.broadcast_in_dim %b14zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1134 = stablehlo.add %v1132, %v1133 : tensor<32x48xf32>
    %v1135 = stablehlo.logistic %v1134 : tensor<32x48xf32>
    %v1136 = stablehlo.multiply %v1134, %v1135 : tensor<32x48xf32>
    %v1137 = stablehlo.dot_general %v1136, %b14zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1138 = stablehlo.broadcast_in_dim %b14zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1139 = stablehlo.add %v1137, %v1138 : tensor<32x1152xf32>
    %v1140 = stablehlo.logistic %v1139 : tensor<32x1152xf32>
    %v1141 = stablehlo.broadcast_in_dim %v1140, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1142 = stablehlo.multiply %v1127, %v1141 : tensor<32x1152x7x7xf32>
    %v1143 = stablehlo.reshape %v1142 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1144 = stablehlo.reshape %v1143 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1145 = stablehlo.convolution(%v1144, %b14pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1146 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1147 = stablehlo.add %v1145, %v1146 : tensor<32x192x7x7xf32>
    %v1148 = stablehlo.reshape %v1147 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1149 = stablehlo.reshape %v1148 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1150 = stablehlo.broadcast_in_dim %b14pnmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1151 = stablehlo.subtract %v1149, %v1150 : tensor<32x192x7x7xf32>
    %v1152 = stablehlo.broadcast_in_dim %b14pnvar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1153 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1154 = stablehlo.add %v1152, %v1153 : tensor<32x192x7x7xf32>
    %v1155 = stablehlo.rsqrt %v1154 : tensor<32x192x7x7xf32>
    %v1156 = stablehlo.multiply %v1151, %v1155 : tensor<32x192x7x7xf32>
    %v1157 = stablehlo.broadcast_in_dim %b14pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1158 = stablehlo.broadcast_in_dim %b14pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1159 = stablehlo.multiply %v1156, %v1157 : tensor<32x192x7x7xf32>
    %v1160 = stablehlo.add %v1159, %v1158 : tensor<32x192x7x7xf32>
    %v1161 = stablehlo.reshape %v1160 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1162 = stablehlo.reshape %v1161 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1163 = stablehlo.broadcast_in_dim %dp13, dims = [0] : (tensor<32xf32>) -> tensor<32x192x7x7xf32>
    %v1164 = stablehlo.multiply %v1163, %v1162 : tensor<32x192x7x7xf32>
    %v1165 = stablehlo.reshape %v1164 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1166 = stablehlo.reshape %v1165 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1167 = stablehlo.reshape %v1082 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1168 = stablehlo.add %v1166, %v1167 : tensor<32x192x7x7xf32>
    %v1169 = stablehlo.reshape %v1168 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1170 = stablehlo.reshape %v1169 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1171 = stablehlo.convolution(%v1170, %b15eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1172 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1173 = stablehlo.add %v1171, %v1172 : tensor<32x1152x7x7xf32>
    %v1174 = stablehlo.reshape %v1173 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1175 = stablehlo.reshape %v1174 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1176 = stablehlo.broadcast_in_dim %b15enmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1177 = stablehlo.subtract %v1175, %v1176 : tensor<32x1152x7x7xf32>
    %v1178 = stablehlo.broadcast_in_dim %b15envar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1179 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1180 = stablehlo.add %v1178, %v1179 : tensor<32x1152x7x7xf32>
    %v1181 = stablehlo.rsqrt %v1180 : tensor<32x1152x7x7xf32>
    %v1182 = stablehlo.multiply %v1177, %v1181 : tensor<32x1152x7x7xf32>
    %v1183 = stablehlo.broadcast_in_dim %b15eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1184 = stablehlo.broadcast_in_dim %b15ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1185 = stablehlo.multiply %v1182, %v1183 : tensor<32x1152x7x7xf32>
    %v1186 = stablehlo.add %v1185, %v1184 : tensor<32x1152x7x7xf32>
    %v1187 = stablehlo.reshape %v1186 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1188 = stablehlo.reshape %v1187 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1189 = stablehlo.logistic %v1188 : tensor<32x1152x7x7xf32>
    %v1190 = stablehlo.multiply %v1188, %v1189 : tensor<32x1152x7x7xf32>
    %v1191 = stablehlo.reshape %v1190 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1192 = stablehlo.reshape %v1191 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1193 = stablehlo.convolution(%v1192, %b15dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v1194 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1195 = stablehlo.add %v1193, %v1194 : tensor<32x1152x7x7xf32>
    %v1196 = stablehlo.reshape %v1195 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1197 = stablehlo.reshape %v1196 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1198 = stablehlo.broadcast_in_dim %b15dnmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1199 = stablehlo.subtract %v1197, %v1198 : tensor<32x1152x7x7xf32>
    %v1200 = stablehlo.broadcast_in_dim %b15dnvar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1201 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1202 = stablehlo.add %v1200, %v1201 : tensor<32x1152x7x7xf32>
    %v1203 = stablehlo.rsqrt %v1202 : tensor<32x1152x7x7xf32>
    %v1204 = stablehlo.multiply %v1199, %v1203 : tensor<32x1152x7x7xf32>
    %v1205 = stablehlo.broadcast_in_dim %b15dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1206 = stablehlo.broadcast_in_dim %b15dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1207 = stablehlo.multiply %v1204, %v1205 : tensor<32x1152x7x7xf32>
    %v1208 = stablehlo.add %v1207, %v1206 : tensor<32x1152x7x7xf32>
    %v1209 = stablehlo.reshape %v1208 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1210 = stablehlo.reshape %v1209 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1211 = stablehlo.logistic %v1210 : tensor<32x1152x7x7xf32>
    %v1212 = stablehlo.multiply %v1210, %v1211 : tensor<32x1152x7x7xf32>
    %v1213 = stablehlo.reshape %v1212 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1214 = stablehlo.reshape %v1213 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1215 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1216 = stablehlo.reduce(%v1214 init: %v1215) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1217 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1218 = stablehlo.divide %v1216, %v1217 : tensor<32x1152xf32>
    %v1219 = stablehlo.dot_general %v1218, %b15zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1220 = stablehlo.broadcast_in_dim %b15zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1221 = stablehlo.add %v1219, %v1220 : tensor<32x48xf32>
    %v1222 = stablehlo.logistic %v1221 : tensor<32x48xf32>
    %v1223 = stablehlo.multiply %v1221, %v1222 : tensor<32x48xf32>
    %v1224 = stablehlo.dot_general %v1223, %b15zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1225 = stablehlo.broadcast_in_dim %b15zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1226 = stablehlo.add %v1224, %v1225 : tensor<32x1152xf32>
    %v1227 = stablehlo.logistic %v1226 : tensor<32x1152xf32>
    %v1228 = stablehlo.broadcast_in_dim %v1227, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1229 = stablehlo.multiply %v1214, %v1228 : tensor<32x1152x7x7xf32>
    %v1230 = stablehlo.reshape %v1229 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1231 = stablehlo.reshape %v1230 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1232 = stablehlo.convolution(%v1231, %b15pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1233 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1234 = stablehlo.add %v1232, %v1233 : tensor<32x192x7x7xf32>
    %v1235 = stablehlo.reshape %v1234 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1236 = stablehlo.reshape %v1235 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1237 = stablehlo.broadcast_in_dim %b15pnmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1238 = stablehlo.subtract %v1236, %v1237 : tensor<32x192x7x7xf32>
    %v1239 = stablehlo.broadcast_in_dim %b15pnvar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1240 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1241 = stablehlo.add %v1239, %v1240 : tensor<32x192x7x7xf32>
    %v1242 = stablehlo.rsqrt %v1241 : tensor<32x192x7x7xf32>
    %v1243 = stablehlo.multiply %v1238, %v1242 : tensor<32x192x7x7xf32>
    %v1244 = stablehlo.broadcast_in_dim %b15pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1245 = stablehlo.broadcast_in_dim %b15pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1246 = stablehlo.multiply %v1243, %v1244 : tensor<32x192x7x7xf32>
    %v1247 = stablehlo.add %v1246, %v1245 : tensor<32x192x7x7xf32>
    %v1248 = stablehlo.reshape %v1247 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1249 = stablehlo.reshape %v1248 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1250 = stablehlo.broadcast_in_dim %dp14, dims = [0] : (tensor<32xf32>) -> tensor<32x192x7x7xf32>
    %v1251 = stablehlo.multiply %v1250, %v1249 : tensor<32x192x7x7xf32>
    %v1252 = stablehlo.reshape %v1251 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1253 = stablehlo.reshape %v1252 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1254 = stablehlo.reshape %v1169 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1255 = stablehlo.add %v1253, %v1254 : tensor<32x192x7x7xf32>
    %v1256 = stablehlo.reshape %v1255 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1257 = stablehlo.reshape %v1256 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1258 = stablehlo.convolution(%v1257, %b16eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1259 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1260 = stablehlo.add %v1258, %v1259 : tensor<32x1152x7x7xf32>
    %v1261 = stablehlo.reshape %v1260 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1262 = stablehlo.reshape %v1261 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1263 = stablehlo.broadcast_in_dim %b16enmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1264 = stablehlo.subtract %v1262, %v1263 : tensor<32x1152x7x7xf32>
    %v1265 = stablehlo.broadcast_in_dim %b16envar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1266 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1267 = stablehlo.add %v1265, %v1266 : tensor<32x1152x7x7xf32>
    %v1268 = stablehlo.rsqrt %v1267 : tensor<32x1152x7x7xf32>
    %v1269 = stablehlo.multiply %v1264, %v1268 : tensor<32x1152x7x7xf32>
    %v1270 = stablehlo.broadcast_in_dim %b16eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1271 = stablehlo.broadcast_in_dim %b16ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1272 = stablehlo.multiply %v1269, %v1270 : tensor<32x1152x7x7xf32>
    %v1273 = stablehlo.add %v1272, %v1271 : tensor<32x1152x7x7xf32>
    %v1274 = stablehlo.reshape %v1273 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1275 = stablehlo.reshape %v1274 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1276 = stablehlo.logistic %v1275 : tensor<32x1152x7x7xf32>
    %v1277 = stablehlo.multiply %v1275, %v1276 : tensor<32x1152x7x7xf32>
    %v1278 = stablehlo.reshape %v1277 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1279 = stablehlo.reshape %v1278 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1280 = stablehlo.convolution(%v1279, %b16dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x3x3xf32>) -> tensor<32x1152x7x7xf32>
    %v1281 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1282 = stablehlo.add %v1280, %v1281 : tensor<32x1152x7x7xf32>
    %v1283 = stablehlo.reshape %v1282 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1284 = stablehlo.reshape %v1283 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1285 = stablehlo.broadcast_in_dim %b16dnmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1286 = stablehlo.subtract %v1284, %v1285 : tensor<32x1152x7x7xf32>
    %v1287 = stablehlo.broadcast_in_dim %b16dnvar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1288 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1289 = stablehlo.add %v1287, %v1288 : tensor<32x1152x7x7xf32>
    %v1290 = stablehlo.rsqrt %v1289 : tensor<32x1152x7x7xf32>
    %v1291 = stablehlo.multiply %v1286, %v1290 : tensor<32x1152x7x7xf32>
    %v1292 = stablehlo.broadcast_in_dim %b16dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1293 = stablehlo.broadcast_in_dim %b16dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1294 = stablehlo.multiply %v1291, %v1292 : tensor<32x1152x7x7xf32>
    %v1295 = stablehlo.add %v1294, %v1293 : tensor<32x1152x7x7xf32>
    %v1296 = stablehlo.reshape %v1295 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1297 = stablehlo.reshape %v1296 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1298 = stablehlo.logistic %v1297 : tensor<32x1152x7x7xf32>
    %v1299 = stablehlo.multiply %v1297, %v1298 : tensor<32x1152x7x7xf32>
    %v1300 = stablehlo.reshape %v1299 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1301 = stablehlo.reshape %v1300 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1302 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1303 = stablehlo.reduce(%v1301 init: %v1302) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1304 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1305 = stablehlo.divide %v1303, %v1304 : tensor<32x1152xf32>
    %v1306 = stablehlo.dot_general %v1305, %b16zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1307 = stablehlo.broadcast_in_dim %b16zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1308 = stablehlo.add %v1306, %v1307 : tensor<32x48xf32>
    %v1309 = stablehlo.logistic %v1308 : tensor<32x48xf32>
    %v1310 = stablehlo.multiply %v1308, %v1309 : tensor<32x48xf32>
    %v1311 = stablehlo.dot_general %v1310, %b16zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1312 = stablehlo.broadcast_in_dim %b16zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1313 = stablehlo.add %v1311, %v1312 : tensor<32x1152xf32>
    %v1314 = stablehlo.logistic %v1313 : tensor<32x1152xf32>
    %v1315 = stablehlo.broadcast_in_dim %v1314, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1316 = stablehlo.multiply %v1301, %v1315 : tensor<32x1152x7x7xf32>
    %v1317 = stablehlo.reshape %v1316 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1318 = stablehlo.reshape %v1317 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1319 = stablehlo.convolution(%v1318, %b16pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<320x1152x1x1xf32>) -> tensor<32x320x7x7xf32>
    %v1320 = stablehlo.broadcast_in_dim %zb320, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1321 = stablehlo.add %v1319, %v1320 : tensor<32x320x7x7xf32>
    %v1322 = stablehlo.reshape %v1321 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1323 = stablehlo.reshape %v1322 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1324 = stablehlo.broadcast_in_dim %b16pnmu, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1325 = stablehlo.subtract %v1323, %v1324 : tensor<32x320x7x7xf32>
    %v1326 = stablehlo.broadcast_in_dim %b16pnvar, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1327 = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %v1328 = stablehlo.add %v1326, %v1327 : tensor<32x320x7x7xf32>
    %v1329 = stablehlo.rsqrt %v1328 : tensor<32x320x7x7xf32>
    %v1330 = stablehlo.multiply %v1325, %v1329 : tensor<32x320x7x7xf32>
    %v1331 = stablehlo.broadcast_in_dim %b16pg, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1332 = stablehlo.broadcast_in_dim %b16pbt, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1333 = stablehlo.multiply %v1330, %v1331 : tensor<32x320x7x7xf32>
    %v1334 = stablehlo.add %v1333, %v1332 : tensor<32x320x7x7xf32>
    %v1335 = stablehlo.reshape %v1334 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1336 = stablehlo.reshape %v1335 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1337 = stablehlo.convolution(%v1336, %hW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x320x7x7xf32>, tensor<1280x320x1x1xf32>) -> tensor<32x1280x7x7xf32>
    %v1338 = stablehlo.broadcast_in_dim %zb1280, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1339 = stablehlo.add %v1337, %v1338 : tensor<32x1280x7x7xf32>
    %v1340 = stablehlo.reshape %v1339 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1341 = stablehlo.reshape %v1340 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1342 = stablehlo.broadcast_in_dim %hnmu, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1343 = stablehlo.subtract %v1341, %v1342 : tensor<32x1280x7x7xf32>
    %v1344 = stablehlo.broadcast_in_dim %hnvar, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1345 = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %v1346 = stablehlo.add %v1344, %v1345 : tensor<32x1280x7x7xf32>
    %v1347 = stablehlo.rsqrt %v1346 : tensor<32x1280x7x7xf32>
    %v1348 = stablehlo.multiply %v1343, %v1347 : tensor<32x1280x7x7xf32>
    %v1349 = stablehlo.broadcast_in_dim %hg, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1350 = stablehlo.broadcast_in_dim %hbt, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1351 = stablehlo.multiply %v1348, %v1349 : tensor<32x1280x7x7xf32>
    %v1352 = stablehlo.add %v1351, %v1350 : tensor<32x1280x7x7xf32>
    %v1353 = stablehlo.reshape %v1352 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1354 = stablehlo.reshape %v1353 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1355 = stablehlo.logistic %v1354 : tensor<32x1280x7x7xf32>
    %v1356 = stablehlo.multiply %v1354, %v1355 : tensor<32x1280x7x7xf32>
    %v1357 = stablehlo.reshape %v1356 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1358 = stablehlo.reshape %v1357 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1359 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1360 = stablehlo.reduce(%v1358 init: %v1359) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v1361 = stablehlo.constant dense<49.0> : tensor<32x1280xf32>
    %v1362 = stablehlo.divide %v1360, %v1361 : tensor<32x1280xf32>
    %v1363 = stablehlo.dot_general %v1362, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1280xf32>, tensor<1280x10xf32>) -> tensor<32x10xf32>
    %v1364 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v1365 = stablehlo.add %v1363, %v1364 : tensor<32x10xf32>
    return %v1365 : tensor<32x10xf32>
  }
}
