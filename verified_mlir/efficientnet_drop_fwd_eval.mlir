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
    %v57 = stablehlo.reshape %v43 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v58 = stablehlo.constant dense<0.0> : tensor<f32>
    %v59 = stablehlo.reduce(%v57 init: %v58) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v60 = stablehlo.constant dense<12544.0> : tensor<32x32xf32>
    %v61 = stablehlo.divide %v59, %v60 : tensor<32x32xf32>
    %v62 = stablehlo.dot_general %v61, %b1zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x8xf32>) -> tensor<32x8xf32>
    %v63 = stablehlo.broadcast_in_dim %b1zb1, dims = [1] : (tensor<8xf32>) -> tensor<32x8xf32>
    %v64 = stablehlo.add %v62, %v63 : tensor<32x8xf32>
    %v65 = stablehlo.logistic %v64 : tensor<32x8xf32>
    %v66 = stablehlo.multiply %v64, %v65 : tensor<32x8xf32>
    %v67 = stablehlo.dot_general %v66, %b1zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x8xf32>, tensor<8x32xf32>) -> tensor<32x32xf32>
    %v68 = stablehlo.broadcast_in_dim %b1zb2, dims = [1] : (tensor<32xf32>) -> tensor<32x32xf32>
    %v69 = stablehlo.add %v67, %v68 : tensor<32x32xf32>
    %v70 = stablehlo.logistic %v69 : tensor<32x32xf32>
    %v71 = stablehlo.broadcast_in_dim %v70, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v72 = stablehlo.multiply %v57, %v71 : tensor<32x32x112x112xf32>
    %v73 = stablehlo.reshape %v72 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v74 = stablehlo.reshape %v73 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v75 = stablehlo.convolution(%v74, %b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<16x32x1x1xf32>) -> tensor<32x16x112x112xf32>
    %v76 = stablehlo.broadcast_in_dim %zb16, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v77 = stablehlo.add %v75, %v76 : tensor<32x16x112x112xf32>
    %v78 = stablehlo.reshape %v77 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v79 = stablehlo.reshape %v78 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v80 = stablehlo.broadcast_in_dim %b1pnmu, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v81 = stablehlo.subtract %v79, %v80 : tensor<32x16x112x112xf32>
    %v82 = stablehlo.broadcast_in_dim %b1pnvar, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v83 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v84 = stablehlo.add %v82, %v83 : tensor<32x16x112x112xf32>
    %v85 = stablehlo.rsqrt %v84 : tensor<32x16x112x112xf32>
    %v86 = stablehlo.multiply %v81, %v85 : tensor<32x16x112x112xf32>
    %v87 = stablehlo.broadcast_in_dim %b1pg, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v88 = stablehlo.broadcast_in_dim %b1pbt, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v89 = stablehlo.multiply %v86, %v87 : tensor<32x16x112x112xf32>
    %v90 = stablehlo.add %v89, %v88 : tensor<32x16x112x112xf32>
    %v91 = stablehlo.reshape %v90 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v92 = stablehlo.reshape %v91 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v93 = stablehlo.convolution(%v92, %b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x16x112x112xf32>, tensor<96x16x1x1xf32>) -> tensor<32x96x112x112xf32>
    %v94 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v95 = stablehlo.add %v93, %v94 : tensor<32x96x112x112xf32>
    %v96 = stablehlo.reshape %v95 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v97 = stablehlo.reshape %v96 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v98 = stablehlo.broadcast_in_dim %b2enmu, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v99 = stablehlo.subtract %v97, %v98 : tensor<32x96x112x112xf32>
    %v100 = stablehlo.broadcast_in_dim %b2envar, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v101 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v102 = stablehlo.add %v100, %v101 : tensor<32x96x112x112xf32>
    %v103 = stablehlo.rsqrt %v102 : tensor<32x96x112x112xf32>
    %v104 = stablehlo.multiply %v99, %v103 : tensor<32x96x112x112xf32>
    %v105 = stablehlo.broadcast_in_dim %b2eg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v106 = stablehlo.broadcast_in_dim %b2ebt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v107 = stablehlo.multiply %v104, %v105 : tensor<32x96x112x112xf32>
    %v108 = stablehlo.add %v107, %v106 : tensor<32x96x112x112xf32>
    %v109 = stablehlo.reshape %v108 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v110 = stablehlo.reshape %v109 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v111 = stablehlo.logistic %v110 : tensor<32x96x112x112xf32>
    %v112 = stablehlo.multiply %v110, %v111 : tensor<32x96x112x112xf32>
    %v113 = stablehlo.reshape %v112 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v114 = stablehlo.reshape %v113 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v115 = stablehlo.convolution(%v114, %b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x112x112xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x56x56xf32>
    %v116 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v117 = stablehlo.add %v115, %v116 : tensor<32x96x56x56xf32>
    %v118 = stablehlo.reshape %v117 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v119 = stablehlo.reshape %v118 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v120 = stablehlo.broadcast_in_dim %b2dnmu, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v121 = stablehlo.subtract %v119, %v120 : tensor<32x96x56x56xf32>
    %v122 = stablehlo.broadcast_in_dim %b2dnvar, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v123 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v124 = stablehlo.add %v122, %v123 : tensor<32x96x56x56xf32>
    %v125 = stablehlo.rsqrt %v124 : tensor<32x96x56x56xf32>
    %v126 = stablehlo.multiply %v121, %v125 : tensor<32x96x56x56xf32>
    %v127 = stablehlo.broadcast_in_dim %b2dg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v128 = stablehlo.broadcast_in_dim %b2dbt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v129 = stablehlo.multiply %v126, %v127 : tensor<32x96x56x56xf32>
    %v130 = stablehlo.add %v129, %v128 : tensor<32x96x56x56xf32>
    %v131 = stablehlo.reshape %v130 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v132 = stablehlo.reshape %v131 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v133 = stablehlo.logistic %v132 : tensor<32x96x56x56xf32>
    %v134 = stablehlo.multiply %v132, %v133 : tensor<32x96x56x56xf32>
    %v135 = stablehlo.reshape %v134 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v136 = stablehlo.reshape %v135 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v137 = stablehlo.constant dense<0.0> : tensor<f32>
    %v138 = stablehlo.reduce(%v136 init: %v137) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v139 = stablehlo.constant dense<3136.0> : tensor<32x96xf32>
    %v140 = stablehlo.divide %v138, %v139 : tensor<32x96xf32>
    %v141 = stablehlo.dot_general %v140, %b2zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<96x4xf32>) -> tensor<32x4xf32>
    %v142 = stablehlo.broadcast_in_dim %b2zb1, dims = [1] : (tensor<4xf32>) -> tensor<32x4xf32>
    %v143 = stablehlo.add %v141, %v142 : tensor<32x4xf32>
    %v144 = stablehlo.logistic %v143 : tensor<32x4xf32>
    %v145 = stablehlo.multiply %v143, %v144 : tensor<32x4xf32>
    %v146 = stablehlo.dot_general %v145, %b2zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<4x96xf32>) -> tensor<32x96xf32>
    %v147 = stablehlo.broadcast_in_dim %b2zb2, dims = [1] : (tensor<96xf32>) -> tensor<32x96xf32>
    %v148 = stablehlo.add %v146, %v147 : tensor<32x96xf32>
    %v149 = stablehlo.reshape %v135 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v150 = stablehlo.constant dense<0.0> : tensor<f32>
    %v151 = stablehlo.reduce(%v149 init: %v150) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v152 = stablehlo.constant dense<3136.0> : tensor<32x96xf32>
    %v153 = stablehlo.divide %v151, %v152 : tensor<32x96xf32>
    %v154 = stablehlo.dot_general %v153, %b2zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<96x4xf32>) -> tensor<32x4xf32>
    %v155 = stablehlo.broadcast_in_dim %b2zb1, dims = [1] : (tensor<4xf32>) -> tensor<32x4xf32>
    %v156 = stablehlo.add %v154, %v155 : tensor<32x4xf32>
    %v157 = stablehlo.logistic %v156 : tensor<32x4xf32>
    %v158 = stablehlo.multiply %v156, %v157 : tensor<32x4xf32>
    %v159 = stablehlo.dot_general %v158, %b2zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<4x96xf32>) -> tensor<32x96xf32>
    %v160 = stablehlo.broadcast_in_dim %b2zb2, dims = [1] : (tensor<96xf32>) -> tensor<32x96xf32>
    %v161 = stablehlo.add %v159, %v160 : tensor<32x96xf32>
    %v162 = stablehlo.logistic %v161 : tensor<32x96xf32>
    %v163 = stablehlo.broadcast_in_dim %v162, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v164 = stablehlo.multiply %v149, %v163 : tensor<32x96x56x56xf32>
    %v165 = stablehlo.reshape %v164 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v166 = stablehlo.reshape %v165 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v167 = stablehlo.convolution(%v166, %b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<24x96x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v168 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v169 = stablehlo.add %v167, %v168 : tensor<32x24x56x56xf32>
    %v170 = stablehlo.reshape %v169 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v171 = stablehlo.reshape %v170 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v172 = stablehlo.broadcast_in_dim %b2pnmu, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v173 = stablehlo.subtract %v171, %v172 : tensor<32x24x56x56xf32>
    %v174 = stablehlo.broadcast_in_dim %b2pnvar, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v175 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v176 = stablehlo.add %v174, %v175 : tensor<32x24x56x56xf32>
    %v177 = stablehlo.rsqrt %v176 : tensor<32x24x56x56xf32>
    %v178 = stablehlo.multiply %v173, %v177 : tensor<32x24x56x56xf32>
    %v179 = stablehlo.broadcast_in_dim %b2pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v180 = stablehlo.broadcast_in_dim %b2pbt, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v181 = stablehlo.multiply %v178, %v179 : tensor<32x24x56x56xf32>
    %v182 = stablehlo.add %v181, %v180 : tensor<32x24x56x56xf32>
    %v183 = stablehlo.reshape %v182 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v184 = stablehlo.reshape %v183 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v185 = stablehlo.convolution(%v184, %b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v186 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v187 = stablehlo.add %v185, %v186 : tensor<32x144x56x56xf32>
    %v188 = stablehlo.reshape %v187 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v189 = stablehlo.reshape %v188 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v190 = stablehlo.broadcast_in_dim %b3enmu, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v191 = stablehlo.subtract %v189, %v190 : tensor<32x144x56x56xf32>
    %v192 = stablehlo.broadcast_in_dim %b3envar, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v193 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v194 = stablehlo.add %v192, %v193 : tensor<32x144x56x56xf32>
    %v195 = stablehlo.rsqrt %v194 : tensor<32x144x56x56xf32>
    %v196 = stablehlo.multiply %v191, %v195 : tensor<32x144x56x56xf32>
    %v197 = stablehlo.broadcast_in_dim %b3eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v198 = stablehlo.broadcast_in_dim %b3ebt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v199 = stablehlo.multiply %v196, %v197 : tensor<32x144x56x56xf32>
    %v200 = stablehlo.add %v199, %v198 : tensor<32x144x56x56xf32>
    %v201 = stablehlo.reshape %v200 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v202 = stablehlo.reshape %v201 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v203 = stablehlo.logistic %v202 : tensor<32x144x56x56xf32>
    %v204 = stablehlo.multiply %v202, %v203 : tensor<32x144x56x56xf32>
    %v205 = stablehlo.reshape %v204 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v206 = stablehlo.reshape %v205 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v207 = stablehlo.convolution(%v206, %b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x56x56xf32>
    %v208 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v209 = stablehlo.add %v207, %v208 : tensor<32x144x56x56xf32>
    %v210 = stablehlo.reshape %v209 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v211 = stablehlo.reshape %v210 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v212 = stablehlo.broadcast_in_dim %b3dnmu, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v213 = stablehlo.subtract %v211, %v212 : tensor<32x144x56x56xf32>
    %v214 = stablehlo.broadcast_in_dim %b3dnvar, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v215 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v216 = stablehlo.add %v214, %v215 : tensor<32x144x56x56xf32>
    %v217 = stablehlo.rsqrt %v216 : tensor<32x144x56x56xf32>
    %v218 = stablehlo.multiply %v213, %v217 : tensor<32x144x56x56xf32>
    %v219 = stablehlo.broadcast_in_dim %b3dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v220 = stablehlo.broadcast_in_dim %b3dbt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v221 = stablehlo.multiply %v218, %v219 : tensor<32x144x56x56xf32>
    %v222 = stablehlo.add %v221, %v220 : tensor<32x144x56x56xf32>
    %v223 = stablehlo.reshape %v222 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v224 = stablehlo.reshape %v223 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v225 = stablehlo.logistic %v224 : tensor<32x144x56x56xf32>
    %v226 = stablehlo.multiply %v224, %v225 : tensor<32x144x56x56xf32>
    %v227 = stablehlo.reshape %v226 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v228 = stablehlo.reshape %v227 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v229 = stablehlo.constant dense<0.0> : tensor<f32>
    %v230 = stablehlo.reduce(%v228 init: %v229) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v231 = stablehlo.constant dense<3136.0> : tensor<32x144xf32>
    %v232 = stablehlo.divide %v230, %v231 : tensor<32x144xf32>
    %v233 = stablehlo.dot_general %v232, %b3zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v234 = stablehlo.broadcast_in_dim %b3zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v235 = stablehlo.add %v233, %v234 : tensor<32x6xf32>
    %v236 = stablehlo.logistic %v235 : tensor<32x6xf32>
    %v237 = stablehlo.multiply %v235, %v236 : tensor<32x6xf32>
    %v238 = stablehlo.dot_general %v237, %b3zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v239 = stablehlo.broadcast_in_dim %b3zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v240 = stablehlo.add %v238, %v239 : tensor<32x144xf32>
    %v241 = stablehlo.reshape %v227 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v242 = stablehlo.constant dense<0.0> : tensor<f32>
    %v243 = stablehlo.reduce(%v241 init: %v242) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v244 = stablehlo.constant dense<3136.0> : tensor<32x144xf32>
    %v245 = stablehlo.divide %v243, %v244 : tensor<32x144xf32>
    %v246 = stablehlo.dot_general %v245, %b3zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v247 = stablehlo.broadcast_in_dim %b3zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v248 = stablehlo.add %v246, %v247 : tensor<32x6xf32>
    %v249 = stablehlo.logistic %v248 : tensor<32x6xf32>
    %v250 = stablehlo.multiply %v248, %v249 : tensor<32x6xf32>
    %v251 = stablehlo.dot_general %v250, %b3zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v252 = stablehlo.broadcast_in_dim %b3zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v253 = stablehlo.add %v251, %v252 : tensor<32x144xf32>
    %v254 = stablehlo.logistic %v253 : tensor<32x144xf32>
    %v255 = stablehlo.broadcast_in_dim %v254, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v256 = stablehlo.multiply %v241, %v255 : tensor<32x144x56x56xf32>
    %v257 = stablehlo.reshape %v256 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v258 = stablehlo.reshape %v257 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v259 = stablehlo.convolution(%v258, %b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v260 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v261 = stablehlo.add %v259, %v260 : tensor<32x24x56x56xf32>
    %v262 = stablehlo.reshape %v261 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v263 = stablehlo.reshape %v262 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v264 = stablehlo.broadcast_in_dim %b3pnmu, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v265 = stablehlo.subtract %v263, %v264 : tensor<32x24x56x56xf32>
    %v266 = stablehlo.broadcast_in_dim %b3pnvar, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v267 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v268 = stablehlo.add %v266, %v267 : tensor<32x24x56x56xf32>
    %v269 = stablehlo.rsqrt %v268 : tensor<32x24x56x56xf32>
    %v270 = stablehlo.multiply %v265, %v269 : tensor<32x24x56x56xf32>
    %v271 = stablehlo.broadcast_in_dim %b3pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v272 = stablehlo.broadcast_in_dim %b3pbt, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v273 = stablehlo.multiply %v270, %v271 : tensor<32x24x56x56xf32>
    %v274 = stablehlo.add %v273, %v272 : tensor<32x24x56x56xf32>
    %v275 = stablehlo.reshape %v274 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v276 = stablehlo.reshape %v275 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v277 = stablehlo.broadcast_in_dim %dp2, dims = [0] : (tensor<32xf32>) -> tensor<32x24x56x56xf32>
    %v278 = stablehlo.multiply %v277, %v276 : tensor<32x24x56x56xf32>
    %v279 = stablehlo.reshape %v278 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v280 = stablehlo.reshape %v279 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v281 = stablehlo.reshape %v183 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v282 = stablehlo.add %v280, %v281 : tensor<32x24x56x56xf32>
    %v283 = stablehlo.reshape %v282 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v284 = stablehlo.reshape %v283 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v285 = stablehlo.convolution(%v284, %b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v286 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v287 = stablehlo.add %v285, %v286 : tensor<32x144x56x56xf32>
    %v288 = stablehlo.reshape %v287 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v289 = stablehlo.reshape %v288 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v290 = stablehlo.broadcast_in_dim %b4enmu, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v291 = stablehlo.subtract %v289, %v290 : tensor<32x144x56x56xf32>
    %v292 = stablehlo.broadcast_in_dim %b4envar, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v293 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v294 = stablehlo.add %v292, %v293 : tensor<32x144x56x56xf32>
    %v295 = stablehlo.rsqrt %v294 : tensor<32x144x56x56xf32>
    %v296 = stablehlo.multiply %v291, %v295 : tensor<32x144x56x56xf32>
    %v297 = stablehlo.broadcast_in_dim %b4eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v298 = stablehlo.broadcast_in_dim %b4ebt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v299 = stablehlo.multiply %v296, %v297 : tensor<32x144x56x56xf32>
    %v300 = stablehlo.add %v299, %v298 : tensor<32x144x56x56xf32>
    %v301 = stablehlo.reshape %v300 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v302 = stablehlo.reshape %v301 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v303 = stablehlo.logistic %v302 : tensor<32x144x56x56xf32>
    %v304 = stablehlo.multiply %v302, %v303 : tensor<32x144x56x56xf32>
    %v305 = stablehlo.reshape %v304 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v306 = stablehlo.reshape %v305 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v307 = stablehlo.convolution(%v306, %b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x5x5xf32>) -> tensor<32x144x28x28xf32>
    %v308 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v309 = stablehlo.add %v307, %v308 : tensor<32x144x28x28xf32>
    %v310 = stablehlo.reshape %v309 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v311 = stablehlo.reshape %v310 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v312 = stablehlo.broadcast_in_dim %b4dnmu, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v313 = stablehlo.subtract %v311, %v312 : tensor<32x144x28x28xf32>
    %v314 = stablehlo.broadcast_in_dim %b4dnvar, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v315 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v316 = stablehlo.add %v314, %v315 : tensor<32x144x28x28xf32>
    %v317 = stablehlo.rsqrt %v316 : tensor<32x144x28x28xf32>
    %v318 = stablehlo.multiply %v313, %v317 : tensor<32x144x28x28xf32>
    %v319 = stablehlo.broadcast_in_dim %b4dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v320 = stablehlo.broadcast_in_dim %b4dbt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v321 = stablehlo.multiply %v318, %v319 : tensor<32x144x28x28xf32>
    %v322 = stablehlo.add %v321, %v320 : tensor<32x144x28x28xf32>
    %v323 = stablehlo.reshape %v322 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v324 = stablehlo.reshape %v323 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v325 = stablehlo.logistic %v324 : tensor<32x144x28x28xf32>
    %v326 = stablehlo.multiply %v324, %v325 : tensor<32x144x28x28xf32>
    %v327 = stablehlo.reshape %v326 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v328 = stablehlo.reshape %v327 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v329 = stablehlo.constant dense<0.0> : tensor<f32>
    %v330 = stablehlo.reduce(%v328 init: %v329) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v331 = stablehlo.constant dense<784.0> : tensor<32x144xf32>
    %v332 = stablehlo.divide %v330, %v331 : tensor<32x144xf32>
    %v333 = stablehlo.dot_general %v332, %b4zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v334 = stablehlo.broadcast_in_dim %b4zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v335 = stablehlo.add %v333, %v334 : tensor<32x6xf32>
    %v336 = stablehlo.logistic %v335 : tensor<32x6xf32>
    %v337 = stablehlo.multiply %v335, %v336 : tensor<32x6xf32>
    %v338 = stablehlo.dot_general %v337, %b4zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v339 = stablehlo.broadcast_in_dim %b4zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v340 = stablehlo.add %v338, %v339 : tensor<32x144xf32>
    %v341 = stablehlo.reshape %v327 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v342 = stablehlo.constant dense<0.0> : tensor<f32>
    %v343 = stablehlo.reduce(%v341 init: %v342) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v344 = stablehlo.constant dense<784.0> : tensor<32x144xf32>
    %v345 = stablehlo.divide %v343, %v344 : tensor<32x144xf32>
    %v346 = stablehlo.dot_general %v345, %b4zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v347 = stablehlo.broadcast_in_dim %b4zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v348 = stablehlo.add %v346, %v347 : tensor<32x6xf32>
    %v349 = stablehlo.logistic %v348 : tensor<32x6xf32>
    %v350 = stablehlo.multiply %v348, %v349 : tensor<32x6xf32>
    %v351 = stablehlo.dot_general %v350, %b4zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v352 = stablehlo.broadcast_in_dim %b4zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v353 = stablehlo.add %v351, %v352 : tensor<32x144xf32>
    %v354 = stablehlo.logistic %v353 : tensor<32x144xf32>
    %v355 = stablehlo.broadcast_in_dim %v354, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v356 = stablehlo.multiply %v341, %v355 : tensor<32x144x28x28xf32>
    %v357 = stablehlo.reshape %v356 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v358 = stablehlo.reshape %v357 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v359 = stablehlo.convolution(%v358, %b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x28x28xf32>, tensor<40x144x1x1xf32>) -> tensor<32x40x28x28xf32>
    %v360 = stablehlo.broadcast_in_dim %zb40, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v361 = stablehlo.add %v359, %v360 : tensor<32x40x28x28xf32>
    %v362 = stablehlo.reshape %v361 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v363 = stablehlo.reshape %v362 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v364 = stablehlo.broadcast_in_dim %b4pnmu, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v365 = stablehlo.subtract %v363, %v364 : tensor<32x40x28x28xf32>
    %v366 = stablehlo.broadcast_in_dim %b4pnvar, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v367 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v368 = stablehlo.add %v366, %v367 : tensor<32x40x28x28xf32>
    %v369 = stablehlo.rsqrt %v368 : tensor<32x40x28x28xf32>
    %v370 = stablehlo.multiply %v365, %v369 : tensor<32x40x28x28xf32>
    %v371 = stablehlo.broadcast_in_dim %b4pg, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v372 = stablehlo.broadcast_in_dim %b4pbt, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v373 = stablehlo.multiply %v370, %v371 : tensor<32x40x28x28xf32>
    %v374 = stablehlo.add %v373, %v372 : tensor<32x40x28x28xf32>
    %v375 = stablehlo.reshape %v374 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v376 = stablehlo.reshape %v375 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v377 = stablehlo.convolution(%v376, %b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<32x240x28x28xf32>
    %v378 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v379 = stablehlo.add %v377, %v378 : tensor<32x240x28x28xf32>
    %v380 = stablehlo.reshape %v379 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v381 = stablehlo.reshape %v380 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v382 = stablehlo.broadcast_in_dim %b5enmu, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v383 = stablehlo.subtract %v381, %v382 : tensor<32x240x28x28xf32>
    %v384 = stablehlo.broadcast_in_dim %b5envar, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v385 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v386 = stablehlo.add %v384, %v385 : tensor<32x240x28x28xf32>
    %v387 = stablehlo.rsqrt %v386 : tensor<32x240x28x28xf32>
    %v388 = stablehlo.multiply %v383, %v387 : tensor<32x240x28x28xf32>
    %v389 = stablehlo.broadcast_in_dim %b5eg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v390 = stablehlo.broadcast_in_dim %b5ebt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v391 = stablehlo.multiply %v388, %v389 : tensor<32x240x28x28xf32>
    %v392 = stablehlo.add %v391, %v390 : tensor<32x240x28x28xf32>
    %v393 = stablehlo.reshape %v392 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v394 = stablehlo.reshape %v393 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v395 = stablehlo.logistic %v394 : tensor<32x240x28x28xf32>
    %v396 = stablehlo.multiply %v394, %v395 : tensor<32x240x28x28xf32>
    %v397 = stablehlo.reshape %v396 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v398 = stablehlo.reshape %v397 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v399 = stablehlo.convolution(%v398, %b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x5x5xf32>) -> tensor<32x240x28x28xf32>
    %v400 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v401 = stablehlo.add %v399, %v400 : tensor<32x240x28x28xf32>
    %v402 = stablehlo.reshape %v401 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v403 = stablehlo.reshape %v402 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v404 = stablehlo.broadcast_in_dim %b5dnmu, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v405 = stablehlo.subtract %v403, %v404 : tensor<32x240x28x28xf32>
    %v406 = stablehlo.broadcast_in_dim %b5dnvar, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v407 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v408 = stablehlo.add %v406, %v407 : tensor<32x240x28x28xf32>
    %v409 = stablehlo.rsqrt %v408 : tensor<32x240x28x28xf32>
    %v410 = stablehlo.multiply %v405, %v409 : tensor<32x240x28x28xf32>
    %v411 = stablehlo.broadcast_in_dim %b5dg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v412 = stablehlo.broadcast_in_dim %b5dbt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v413 = stablehlo.multiply %v410, %v411 : tensor<32x240x28x28xf32>
    %v414 = stablehlo.add %v413, %v412 : tensor<32x240x28x28xf32>
    %v415 = stablehlo.reshape %v414 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v416 = stablehlo.reshape %v415 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v417 = stablehlo.logistic %v416 : tensor<32x240x28x28xf32>
    %v418 = stablehlo.multiply %v416, %v417 : tensor<32x240x28x28xf32>
    %v419 = stablehlo.reshape %v418 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v420 = stablehlo.reshape %v419 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v421 = stablehlo.constant dense<0.0> : tensor<f32>
    %v422 = stablehlo.reduce(%v420 init: %v421) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v423 = stablehlo.constant dense<784.0> : tensor<32x240xf32>
    %v424 = stablehlo.divide %v422, %v423 : tensor<32x240xf32>
    %v425 = stablehlo.dot_general %v424, %b5zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v426 = stablehlo.broadcast_in_dim %b5zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v427 = stablehlo.add %v425, %v426 : tensor<32x10xf32>
    %v428 = stablehlo.logistic %v427 : tensor<32x10xf32>
    %v429 = stablehlo.multiply %v427, %v428 : tensor<32x10xf32>
    %v430 = stablehlo.dot_general %v429, %b5zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v431 = stablehlo.broadcast_in_dim %b5zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v432 = stablehlo.add %v430, %v431 : tensor<32x240xf32>
    %v433 = stablehlo.reshape %v419 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v434 = stablehlo.constant dense<0.0> : tensor<f32>
    %v435 = stablehlo.reduce(%v433 init: %v434) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v436 = stablehlo.constant dense<784.0> : tensor<32x240xf32>
    %v437 = stablehlo.divide %v435, %v436 : tensor<32x240xf32>
    %v438 = stablehlo.dot_general %v437, %b5zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v439 = stablehlo.broadcast_in_dim %b5zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v440 = stablehlo.add %v438, %v439 : tensor<32x10xf32>
    %v441 = stablehlo.logistic %v440 : tensor<32x10xf32>
    %v442 = stablehlo.multiply %v440, %v441 : tensor<32x10xf32>
    %v443 = stablehlo.dot_general %v442, %b5zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v444 = stablehlo.broadcast_in_dim %b5zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v445 = stablehlo.add %v443, %v444 : tensor<32x240xf32>
    %v446 = stablehlo.logistic %v445 : tensor<32x240xf32>
    %v447 = stablehlo.broadcast_in_dim %v446, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x28x28xf32>
    %v448 = stablehlo.multiply %v433, %v447 : tensor<32x240x28x28xf32>
    %v449 = stablehlo.reshape %v448 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v450 = stablehlo.reshape %v449 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v451 = stablehlo.convolution(%v450, %b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x28x28xf32>, tensor<40x240x1x1xf32>) -> tensor<32x40x28x28xf32>
    %v452 = stablehlo.broadcast_in_dim %zb40, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v453 = stablehlo.add %v451, %v452 : tensor<32x40x28x28xf32>
    %v454 = stablehlo.reshape %v453 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v455 = stablehlo.reshape %v454 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v456 = stablehlo.broadcast_in_dim %b5pnmu, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v457 = stablehlo.subtract %v455, %v456 : tensor<32x40x28x28xf32>
    %v458 = stablehlo.broadcast_in_dim %b5pnvar, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v459 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v460 = stablehlo.add %v458, %v459 : tensor<32x40x28x28xf32>
    %v461 = stablehlo.rsqrt %v460 : tensor<32x40x28x28xf32>
    %v462 = stablehlo.multiply %v457, %v461 : tensor<32x40x28x28xf32>
    %v463 = stablehlo.broadcast_in_dim %b5pg, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v464 = stablehlo.broadcast_in_dim %b5pbt, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v465 = stablehlo.multiply %v462, %v463 : tensor<32x40x28x28xf32>
    %v466 = stablehlo.add %v465, %v464 : tensor<32x40x28x28xf32>
    %v467 = stablehlo.reshape %v466 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v468 = stablehlo.reshape %v467 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v469 = stablehlo.broadcast_in_dim %dp4, dims = [0] : (tensor<32xf32>) -> tensor<32x40x28x28xf32>
    %v470 = stablehlo.multiply %v469, %v468 : tensor<32x40x28x28xf32>
    %v471 = stablehlo.reshape %v470 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v472 = stablehlo.reshape %v471 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v473 = stablehlo.reshape %v375 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v474 = stablehlo.add %v472, %v473 : tensor<32x40x28x28xf32>
    %v475 = stablehlo.reshape %v474 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v476 = stablehlo.reshape %v475 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v477 = stablehlo.convolution(%v476, %b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<32x240x28x28xf32>
    %v478 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v479 = stablehlo.add %v477, %v478 : tensor<32x240x28x28xf32>
    %v480 = stablehlo.reshape %v479 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v481 = stablehlo.reshape %v480 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v482 = stablehlo.broadcast_in_dim %b6enmu, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v483 = stablehlo.subtract %v481, %v482 : tensor<32x240x28x28xf32>
    %v484 = stablehlo.broadcast_in_dim %b6envar, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v485 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v486 = stablehlo.add %v484, %v485 : tensor<32x240x28x28xf32>
    %v487 = stablehlo.rsqrt %v486 : tensor<32x240x28x28xf32>
    %v488 = stablehlo.multiply %v483, %v487 : tensor<32x240x28x28xf32>
    %v489 = stablehlo.broadcast_in_dim %b6eg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v490 = stablehlo.broadcast_in_dim %b6ebt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v491 = stablehlo.multiply %v488, %v489 : tensor<32x240x28x28xf32>
    %v492 = stablehlo.add %v491, %v490 : tensor<32x240x28x28xf32>
    %v493 = stablehlo.reshape %v492 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v494 = stablehlo.reshape %v493 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v495 = stablehlo.logistic %v494 : tensor<32x240x28x28xf32>
    %v496 = stablehlo.multiply %v494, %v495 : tensor<32x240x28x28xf32>
    %v497 = stablehlo.reshape %v496 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v498 = stablehlo.reshape %v497 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v499 = stablehlo.convolution(%v498, %b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x3x3xf32>) -> tensor<32x240x14x14xf32>
    %v500 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v501 = stablehlo.add %v499, %v500 : tensor<32x240x14x14xf32>
    %v502 = stablehlo.reshape %v501 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v503 = stablehlo.reshape %v502 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v504 = stablehlo.broadcast_in_dim %b6dnmu, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v505 = stablehlo.subtract %v503, %v504 : tensor<32x240x14x14xf32>
    %v506 = stablehlo.broadcast_in_dim %b6dnvar, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v507 = stablehlo.constant dense<1.0e-5> : tensor<32x240x14x14xf32>
    %v508 = stablehlo.add %v506, %v507 : tensor<32x240x14x14xf32>
    %v509 = stablehlo.rsqrt %v508 : tensor<32x240x14x14xf32>
    %v510 = stablehlo.multiply %v505, %v509 : tensor<32x240x14x14xf32>
    %v511 = stablehlo.broadcast_in_dim %b6dg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v512 = stablehlo.broadcast_in_dim %b6dbt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v513 = stablehlo.multiply %v510, %v511 : tensor<32x240x14x14xf32>
    %v514 = stablehlo.add %v513, %v512 : tensor<32x240x14x14xf32>
    %v515 = stablehlo.reshape %v514 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v516 = stablehlo.reshape %v515 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v517 = stablehlo.logistic %v516 : tensor<32x240x14x14xf32>
    %v518 = stablehlo.multiply %v516, %v517 : tensor<32x240x14x14xf32>
    %v519 = stablehlo.reshape %v518 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v520 = stablehlo.reshape %v519 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v521 = stablehlo.constant dense<0.0> : tensor<f32>
    %v522 = stablehlo.reduce(%v520 init: %v521) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v523 = stablehlo.constant dense<196.0> : tensor<32x240xf32>
    %v524 = stablehlo.divide %v522, %v523 : tensor<32x240xf32>
    %v525 = stablehlo.dot_general %v524, %b6zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v526 = stablehlo.broadcast_in_dim %b6zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v527 = stablehlo.add %v525, %v526 : tensor<32x10xf32>
    %v528 = stablehlo.logistic %v527 : tensor<32x10xf32>
    %v529 = stablehlo.multiply %v527, %v528 : tensor<32x10xf32>
    %v530 = stablehlo.dot_general %v529, %b6zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v531 = stablehlo.broadcast_in_dim %b6zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v532 = stablehlo.add %v530, %v531 : tensor<32x240xf32>
    %v533 = stablehlo.reshape %v519 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v534 = stablehlo.constant dense<0.0> : tensor<f32>
    %v535 = stablehlo.reduce(%v533 init: %v534) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v536 = stablehlo.constant dense<196.0> : tensor<32x240xf32>
    %v537 = stablehlo.divide %v535, %v536 : tensor<32x240xf32>
    %v538 = stablehlo.dot_general %v537, %b6zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v539 = stablehlo.broadcast_in_dim %b6zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v540 = stablehlo.add %v538, %v539 : tensor<32x10xf32>
    %v541 = stablehlo.logistic %v540 : tensor<32x10xf32>
    %v542 = stablehlo.multiply %v540, %v541 : tensor<32x10xf32>
    %v543 = stablehlo.dot_general %v542, %b6zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v544 = stablehlo.broadcast_in_dim %b6zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v545 = stablehlo.add %v543, %v544 : tensor<32x240xf32>
    %v546 = stablehlo.logistic %v545 : tensor<32x240xf32>
    %v547 = stablehlo.broadcast_in_dim %v546, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x14x14xf32>
    %v548 = stablehlo.multiply %v533, %v547 : tensor<32x240x14x14xf32>
    %v549 = stablehlo.reshape %v548 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v550 = stablehlo.reshape %v549 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v551 = stablehlo.convolution(%v550, %b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x14x14xf32>, tensor<80x240x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v552 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v553 = stablehlo.add %v551, %v552 : tensor<32x80x14x14xf32>
    %v554 = stablehlo.reshape %v553 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v555 = stablehlo.reshape %v554 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v556 = stablehlo.broadcast_in_dim %b6pnmu, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v557 = stablehlo.subtract %v555, %v556 : tensor<32x80x14x14xf32>
    %v558 = stablehlo.broadcast_in_dim %b6pnvar, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v559 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v560 = stablehlo.add %v558, %v559 : tensor<32x80x14x14xf32>
    %v561 = stablehlo.rsqrt %v560 : tensor<32x80x14x14xf32>
    %v562 = stablehlo.multiply %v557, %v561 : tensor<32x80x14x14xf32>
    %v563 = stablehlo.broadcast_in_dim %b6pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v564 = stablehlo.broadcast_in_dim %b6pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v565 = stablehlo.multiply %v562, %v563 : tensor<32x80x14x14xf32>
    %v566 = stablehlo.add %v565, %v564 : tensor<32x80x14x14xf32>
    %v567 = stablehlo.reshape %v566 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v568 = stablehlo.reshape %v567 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v569 = stablehlo.convolution(%v568, %b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v570 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v571 = stablehlo.add %v569, %v570 : tensor<32x480x14x14xf32>
    %v572 = stablehlo.reshape %v571 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v573 = stablehlo.reshape %v572 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v574 = stablehlo.broadcast_in_dim %b7enmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v575 = stablehlo.subtract %v573, %v574 : tensor<32x480x14x14xf32>
    %v576 = stablehlo.broadcast_in_dim %b7envar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v577 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v578 = stablehlo.add %v576, %v577 : tensor<32x480x14x14xf32>
    %v579 = stablehlo.rsqrt %v578 : tensor<32x480x14x14xf32>
    %v580 = stablehlo.multiply %v575, %v579 : tensor<32x480x14x14xf32>
    %v581 = stablehlo.broadcast_in_dim %b7eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v582 = stablehlo.broadcast_in_dim %b7ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v583 = stablehlo.multiply %v580, %v581 : tensor<32x480x14x14xf32>
    %v584 = stablehlo.add %v583, %v582 : tensor<32x480x14x14xf32>
    %v585 = stablehlo.reshape %v584 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v586 = stablehlo.reshape %v585 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v587 = stablehlo.logistic %v586 : tensor<32x480x14x14xf32>
    %v588 = stablehlo.multiply %v586, %v587 : tensor<32x480x14x14xf32>
    %v589 = stablehlo.reshape %v588 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v590 = stablehlo.reshape %v589 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v591 = stablehlo.convolution(%v590, %b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %v592 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v593 = stablehlo.add %v591, %v592 : tensor<32x480x14x14xf32>
    %v594 = stablehlo.reshape %v593 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v595 = stablehlo.reshape %v594 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v596 = stablehlo.broadcast_in_dim %b7dnmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v597 = stablehlo.subtract %v595, %v596 : tensor<32x480x14x14xf32>
    %v598 = stablehlo.broadcast_in_dim %b7dnvar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v599 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v600 = stablehlo.add %v598, %v599 : tensor<32x480x14x14xf32>
    %v601 = stablehlo.rsqrt %v600 : tensor<32x480x14x14xf32>
    %v602 = stablehlo.multiply %v597, %v601 : tensor<32x480x14x14xf32>
    %v603 = stablehlo.broadcast_in_dim %b7dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v604 = stablehlo.broadcast_in_dim %b7dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v605 = stablehlo.multiply %v602, %v603 : tensor<32x480x14x14xf32>
    %v606 = stablehlo.add %v605, %v604 : tensor<32x480x14x14xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v608 = stablehlo.reshape %v607 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v609 = stablehlo.logistic %v608 : tensor<32x480x14x14xf32>
    %v610 = stablehlo.multiply %v608, %v609 : tensor<32x480x14x14xf32>
    %v611 = stablehlo.reshape %v610 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v612 = stablehlo.reshape %v611 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v613 = stablehlo.constant dense<0.0> : tensor<f32>
    %v614 = stablehlo.reduce(%v612 init: %v613) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v615 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v616 = stablehlo.divide %v614, %v615 : tensor<32x480xf32>
    %v617 = stablehlo.dot_general %v616, %b7zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v618 = stablehlo.broadcast_in_dim %b7zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v619 = stablehlo.add %v617, %v618 : tensor<32x20xf32>
    %v620 = stablehlo.logistic %v619 : tensor<32x20xf32>
    %v621 = stablehlo.multiply %v619, %v620 : tensor<32x20xf32>
    %v622 = stablehlo.dot_general %v621, %b7zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v623 = stablehlo.broadcast_in_dim %b7zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v624 = stablehlo.add %v622, %v623 : tensor<32x480xf32>
    %v625 = stablehlo.reshape %v611 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v626 = stablehlo.constant dense<0.0> : tensor<f32>
    %v627 = stablehlo.reduce(%v625 init: %v626) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v628 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v629 = stablehlo.divide %v627, %v628 : tensor<32x480xf32>
    %v630 = stablehlo.dot_general %v629, %b7zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v631 = stablehlo.broadcast_in_dim %b7zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v632 = stablehlo.add %v630, %v631 : tensor<32x20xf32>
    %v633 = stablehlo.logistic %v632 : tensor<32x20xf32>
    %v634 = stablehlo.multiply %v632, %v633 : tensor<32x20xf32>
    %v635 = stablehlo.dot_general %v634, %b7zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v636 = stablehlo.broadcast_in_dim %b7zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v637 = stablehlo.add %v635, %v636 : tensor<32x480xf32>
    %v638 = stablehlo.logistic %v637 : tensor<32x480xf32>
    %v639 = stablehlo.broadcast_in_dim %v638, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v640 = stablehlo.multiply %v625, %v639 : tensor<32x480x14x14xf32>
    %v641 = stablehlo.reshape %v640 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v642 = stablehlo.reshape %v641 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v643 = stablehlo.convolution(%v642, %b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v644 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v645 = stablehlo.add %v643, %v644 : tensor<32x80x14x14xf32>
    %v646 = stablehlo.reshape %v645 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v647 = stablehlo.reshape %v646 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v648 = stablehlo.broadcast_in_dim %b7pnmu, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v649 = stablehlo.subtract %v647, %v648 : tensor<32x80x14x14xf32>
    %v650 = stablehlo.broadcast_in_dim %b7pnvar, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v651 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v652 = stablehlo.add %v650, %v651 : tensor<32x80x14x14xf32>
    %v653 = stablehlo.rsqrt %v652 : tensor<32x80x14x14xf32>
    %v654 = stablehlo.multiply %v649, %v653 : tensor<32x80x14x14xf32>
    %v655 = stablehlo.broadcast_in_dim %b7pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v656 = stablehlo.broadcast_in_dim %b7pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v657 = stablehlo.multiply %v654, %v655 : tensor<32x80x14x14xf32>
    %v658 = stablehlo.add %v657, %v656 : tensor<32x80x14x14xf32>
    %v659 = stablehlo.reshape %v658 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v660 = stablehlo.reshape %v659 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v661 = stablehlo.broadcast_in_dim %dp6, dims = [0] : (tensor<32xf32>) -> tensor<32x80x14x14xf32>
    %v662 = stablehlo.multiply %v661, %v660 : tensor<32x80x14x14xf32>
    %v663 = stablehlo.reshape %v662 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v664 = stablehlo.reshape %v663 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v665 = stablehlo.reshape %v567 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v666 = stablehlo.add %v664, %v665 : tensor<32x80x14x14xf32>
    %v667 = stablehlo.reshape %v666 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v668 = stablehlo.reshape %v667 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v669 = stablehlo.convolution(%v668, %b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v670 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v671 = stablehlo.add %v669, %v670 : tensor<32x480x14x14xf32>
    %v672 = stablehlo.reshape %v671 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v673 = stablehlo.reshape %v672 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v674 = stablehlo.broadcast_in_dim %b8enmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v675 = stablehlo.subtract %v673, %v674 : tensor<32x480x14x14xf32>
    %v676 = stablehlo.broadcast_in_dim %b8envar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v677 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v678 = stablehlo.add %v676, %v677 : tensor<32x480x14x14xf32>
    %v679 = stablehlo.rsqrt %v678 : tensor<32x480x14x14xf32>
    %v680 = stablehlo.multiply %v675, %v679 : tensor<32x480x14x14xf32>
    %v681 = stablehlo.broadcast_in_dim %b8eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v682 = stablehlo.broadcast_in_dim %b8ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v683 = stablehlo.multiply %v680, %v681 : tensor<32x480x14x14xf32>
    %v684 = stablehlo.add %v683, %v682 : tensor<32x480x14x14xf32>
    %v685 = stablehlo.reshape %v684 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v686 = stablehlo.reshape %v685 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v687 = stablehlo.logistic %v686 : tensor<32x480x14x14xf32>
    %v688 = stablehlo.multiply %v686, %v687 : tensor<32x480x14x14xf32>
    %v689 = stablehlo.reshape %v688 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v690 = stablehlo.reshape %v689 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v691 = stablehlo.convolution(%v690, %b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %v692 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v693 = stablehlo.add %v691, %v692 : tensor<32x480x14x14xf32>
    %v694 = stablehlo.reshape %v693 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v695 = stablehlo.reshape %v694 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v696 = stablehlo.broadcast_in_dim %b8dnmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v697 = stablehlo.subtract %v695, %v696 : tensor<32x480x14x14xf32>
    %v698 = stablehlo.broadcast_in_dim %b8dnvar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v699 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v700 = stablehlo.add %v698, %v699 : tensor<32x480x14x14xf32>
    %v701 = stablehlo.rsqrt %v700 : tensor<32x480x14x14xf32>
    %v702 = stablehlo.multiply %v697, %v701 : tensor<32x480x14x14xf32>
    %v703 = stablehlo.broadcast_in_dim %b8dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v704 = stablehlo.broadcast_in_dim %b8dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v705 = stablehlo.multiply %v702, %v703 : tensor<32x480x14x14xf32>
    %v706 = stablehlo.add %v705, %v704 : tensor<32x480x14x14xf32>
    %v707 = stablehlo.reshape %v706 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v708 = stablehlo.reshape %v707 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v709 = stablehlo.logistic %v708 : tensor<32x480x14x14xf32>
    %v710 = stablehlo.multiply %v708, %v709 : tensor<32x480x14x14xf32>
    %v711 = stablehlo.reshape %v710 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v712 = stablehlo.reshape %v711 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v713 = stablehlo.constant dense<0.0> : tensor<f32>
    %v714 = stablehlo.reduce(%v712 init: %v713) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v715 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v716 = stablehlo.divide %v714, %v715 : tensor<32x480xf32>
    %v717 = stablehlo.dot_general %v716, %b8zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v718 = stablehlo.broadcast_in_dim %b8zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v719 = stablehlo.add %v717, %v718 : tensor<32x20xf32>
    %v720 = stablehlo.logistic %v719 : tensor<32x20xf32>
    %v721 = stablehlo.multiply %v719, %v720 : tensor<32x20xf32>
    %v722 = stablehlo.dot_general %v721, %b8zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v723 = stablehlo.broadcast_in_dim %b8zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v724 = stablehlo.add %v722, %v723 : tensor<32x480xf32>
    %v725 = stablehlo.reshape %v711 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v726 = stablehlo.constant dense<0.0> : tensor<f32>
    %v727 = stablehlo.reduce(%v725 init: %v726) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v728 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v729 = stablehlo.divide %v727, %v728 : tensor<32x480xf32>
    %v730 = stablehlo.dot_general %v729, %b8zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v731 = stablehlo.broadcast_in_dim %b8zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v732 = stablehlo.add %v730, %v731 : tensor<32x20xf32>
    %v733 = stablehlo.logistic %v732 : tensor<32x20xf32>
    %v734 = stablehlo.multiply %v732, %v733 : tensor<32x20xf32>
    %v735 = stablehlo.dot_general %v734, %b8zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v736 = stablehlo.broadcast_in_dim %b8zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v737 = stablehlo.add %v735, %v736 : tensor<32x480xf32>
    %v738 = stablehlo.logistic %v737 : tensor<32x480xf32>
    %v739 = stablehlo.broadcast_in_dim %v738, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v740 = stablehlo.multiply %v725, %v739 : tensor<32x480x14x14xf32>
    %v741 = stablehlo.reshape %v740 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v742 = stablehlo.reshape %v741 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v743 = stablehlo.convolution(%v742, %b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v744 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v745 = stablehlo.add %v743, %v744 : tensor<32x80x14x14xf32>
    %v746 = stablehlo.reshape %v745 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v747 = stablehlo.reshape %v746 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v748 = stablehlo.broadcast_in_dim %b8pnmu, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v749 = stablehlo.subtract %v747, %v748 : tensor<32x80x14x14xf32>
    %v750 = stablehlo.broadcast_in_dim %b8pnvar, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v751 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v752 = stablehlo.add %v750, %v751 : tensor<32x80x14x14xf32>
    %v753 = stablehlo.rsqrt %v752 : tensor<32x80x14x14xf32>
    %v754 = stablehlo.multiply %v749, %v753 : tensor<32x80x14x14xf32>
    %v755 = stablehlo.broadcast_in_dim %b8pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v756 = stablehlo.broadcast_in_dim %b8pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v757 = stablehlo.multiply %v754, %v755 : tensor<32x80x14x14xf32>
    %v758 = stablehlo.add %v757, %v756 : tensor<32x80x14x14xf32>
    %v759 = stablehlo.reshape %v758 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v760 = stablehlo.reshape %v759 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v761 = stablehlo.broadcast_in_dim %dp7, dims = [0] : (tensor<32xf32>) -> tensor<32x80x14x14xf32>
    %v762 = stablehlo.multiply %v761, %v760 : tensor<32x80x14x14xf32>
    %v763 = stablehlo.reshape %v762 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v764 = stablehlo.reshape %v763 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v765 = stablehlo.reshape %v667 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v766 = stablehlo.add %v764, %v765 : tensor<32x80x14x14xf32>
    %v767 = stablehlo.reshape %v766 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v768 = stablehlo.reshape %v767 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v769 = stablehlo.convolution(%v768, %b9eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v770 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v771 = stablehlo.add %v769, %v770 : tensor<32x480x14x14xf32>
    %v772 = stablehlo.reshape %v771 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v773 = stablehlo.reshape %v772 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v774 = stablehlo.broadcast_in_dim %b9enmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v775 = stablehlo.subtract %v773, %v774 : tensor<32x480x14x14xf32>
    %v776 = stablehlo.broadcast_in_dim %b9envar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v777 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v778 = stablehlo.add %v776, %v777 : tensor<32x480x14x14xf32>
    %v779 = stablehlo.rsqrt %v778 : tensor<32x480x14x14xf32>
    %v780 = stablehlo.multiply %v775, %v779 : tensor<32x480x14x14xf32>
    %v781 = stablehlo.broadcast_in_dim %b9eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v782 = stablehlo.broadcast_in_dim %b9ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v783 = stablehlo.multiply %v780, %v781 : tensor<32x480x14x14xf32>
    %v784 = stablehlo.add %v783, %v782 : tensor<32x480x14x14xf32>
    %v785 = stablehlo.reshape %v784 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v786 = stablehlo.reshape %v785 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v787 = stablehlo.logistic %v786 : tensor<32x480x14x14xf32>
    %v788 = stablehlo.multiply %v786, %v787 : tensor<32x480x14x14xf32>
    %v789 = stablehlo.reshape %v788 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v790 = stablehlo.reshape %v789 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v791 = stablehlo.convolution(%v790, %b9dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x5x5xf32>) -> tensor<32x480x14x14xf32>
    %v792 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v793 = stablehlo.add %v791, %v792 : tensor<32x480x14x14xf32>
    %v794 = stablehlo.reshape %v793 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v795 = stablehlo.reshape %v794 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v796 = stablehlo.broadcast_in_dim %b9dnmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v797 = stablehlo.subtract %v795, %v796 : tensor<32x480x14x14xf32>
    %v798 = stablehlo.broadcast_in_dim %b9dnvar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v799 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v800 = stablehlo.add %v798, %v799 : tensor<32x480x14x14xf32>
    %v801 = stablehlo.rsqrt %v800 : tensor<32x480x14x14xf32>
    %v802 = stablehlo.multiply %v797, %v801 : tensor<32x480x14x14xf32>
    %v803 = stablehlo.broadcast_in_dim %b9dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v804 = stablehlo.broadcast_in_dim %b9dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v805 = stablehlo.multiply %v802, %v803 : tensor<32x480x14x14xf32>
    %v806 = stablehlo.add %v805, %v804 : tensor<32x480x14x14xf32>
    %v807 = stablehlo.reshape %v806 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v808 = stablehlo.reshape %v807 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v809 = stablehlo.logistic %v808 : tensor<32x480x14x14xf32>
    %v810 = stablehlo.multiply %v808, %v809 : tensor<32x480x14x14xf32>
    %v811 = stablehlo.reshape %v810 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v812 = stablehlo.reshape %v811 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v813 = stablehlo.constant dense<0.0> : tensor<f32>
    %v814 = stablehlo.reduce(%v812 init: %v813) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v815 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v816 = stablehlo.divide %v814, %v815 : tensor<32x480xf32>
    %v817 = stablehlo.dot_general %v816, %b9zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v818 = stablehlo.broadcast_in_dim %b9zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v819 = stablehlo.add %v817, %v818 : tensor<32x20xf32>
    %v820 = stablehlo.logistic %v819 : tensor<32x20xf32>
    %v821 = stablehlo.multiply %v819, %v820 : tensor<32x20xf32>
    %v822 = stablehlo.dot_general %v821, %b9zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v823 = stablehlo.broadcast_in_dim %b9zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v824 = stablehlo.add %v822, %v823 : tensor<32x480xf32>
    %v825 = stablehlo.reshape %v811 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v826 = stablehlo.constant dense<0.0> : tensor<f32>
    %v827 = stablehlo.reduce(%v825 init: %v826) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v828 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v829 = stablehlo.divide %v827, %v828 : tensor<32x480xf32>
    %v830 = stablehlo.dot_general %v829, %b9zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v831 = stablehlo.broadcast_in_dim %b9zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v832 = stablehlo.add %v830, %v831 : tensor<32x20xf32>
    %v833 = stablehlo.logistic %v832 : tensor<32x20xf32>
    %v834 = stablehlo.multiply %v832, %v833 : tensor<32x20xf32>
    %v835 = stablehlo.dot_general %v834, %b9zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v836 = stablehlo.broadcast_in_dim %b9zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v837 = stablehlo.add %v835, %v836 : tensor<32x480xf32>
    %v838 = stablehlo.logistic %v837 : tensor<32x480xf32>
    %v839 = stablehlo.broadcast_in_dim %v838, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v840 = stablehlo.multiply %v825, %v839 : tensor<32x480x14x14xf32>
    %v841 = stablehlo.reshape %v840 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v842 = stablehlo.reshape %v841 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v843 = stablehlo.convolution(%v842, %b9pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<112x480x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v844 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v845 = stablehlo.add %v843, %v844 : tensor<32x112x14x14xf32>
    %v846 = stablehlo.reshape %v845 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v847 = stablehlo.reshape %v846 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v848 = stablehlo.broadcast_in_dim %b9pnmu, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v849 = stablehlo.subtract %v847, %v848 : tensor<32x112x14x14xf32>
    %v850 = stablehlo.broadcast_in_dim %b9pnvar, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v851 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v852 = stablehlo.add %v850, %v851 : tensor<32x112x14x14xf32>
    %v853 = stablehlo.rsqrt %v852 : tensor<32x112x14x14xf32>
    %v854 = stablehlo.multiply %v849, %v853 : tensor<32x112x14x14xf32>
    %v855 = stablehlo.broadcast_in_dim %b9pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v856 = stablehlo.broadcast_in_dim %b9pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v857 = stablehlo.multiply %v854, %v855 : tensor<32x112x14x14xf32>
    %v858 = stablehlo.add %v857, %v856 : tensor<32x112x14x14xf32>
    %v859 = stablehlo.reshape %v858 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v860 = stablehlo.reshape %v859 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v861 = stablehlo.convolution(%v860, %b10eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v862 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v863 = stablehlo.add %v861, %v862 : tensor<32x672x14x14xf32>
    %v864 = stablehlo.reshape %v863 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v865 = stablehlo.reshape %v864 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v866 = stablehlo.broadcast_in_dim %b10enmu, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v867 = stablehlo.subtract %v865, %v866 : tensor<32x672x14x14xf32>
    %v868 = stablehlo.broadcast_in_dim %b10envar, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v869 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v870 = stablehlo.add %v868, %v869 : tensor<32x672x14x14xf32>
    %v871 = stablehlo.rsqrt %v870 : tensor<32x672x14x14xf32>
    %v872 = stablehlo.multiply %v867, %v871 : tensor<32x672x14x14xf32>
    %v873 = stablehlo.broadcast_in_dim %b10eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v874 = stablehlo.broadcast_in_dim %b10ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v875 = stablehlo.multiply %v872, %v873 : tensor<32x672x14x14xf32>
    %v876 = stablehlo.add %v875, %v874 : tensor<32x672x14x14xf32>
    %v877 = stablehlo.reshape %v876 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v878 = stablehlo.reshape %v877 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v879 = stablehlo.logistic %v878 : tensor<32x672x14x14xf32>
    %v880 = stablehlo.multiply %v878, %v879 : tensor<32x672x14x14xf32>
    %v881 = stablehlo.reshape %v880 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v882 = stablehlo.reshape %v881 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v883 = stablehlo.convolution(%v882, %b10dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %v884 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v885 = stablehlo.add %v883, %v884 : tensor<32x672x14x14xf32>
    %v886 = stablehlo.reshape %v885 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v887 = stablehlo.reshape %v886 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v888 = stablehlo.broadcast_in_dim %b10dnmu, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v889 = stablehlo.subtract %v887, %v888 : tensor<32x672x14x14xf32>
    %v890 = stablehlo.broadcast_in_dim %b10dnvar, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v891 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v892 = stablehlo.add %v890, %v891 : tensor<32x672x14x14xf32>
    %v893 = stablehlo.rsqrt %v892 : tensor<32x672x14x14xf32>
    %v894 = stablehlo.multiply %v889, %v893 : tensor<32x672x14x14xf32>
    %v895 = stablehlo.broadcast_in_dim %b10dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v896 = stablehlo.broadcast_in_dim %b10dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v897 = stablehlo.multiply %v894, %v895 : tensor<32x672x14x14xf32>
    %v898 = stablehlo.add %v897, %v896 : tensor<32x672x14x14xf32>
    %v899 = stablehlo.reshape %v898 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v900 = stablehlo.reshape %v899 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v901 = stablehlo.logistic %v900 : tensor<32x672x14x14xf32>
    %v902 = stablehlo.multiply %v900, %v901 : tensor<32x672x14x14xf32>
    %v903 = stablehlo.reshape %v902 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v904 = stablehlo.reshape %v903 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v905 = stablehlo.constant dense<0.0> : tensor<f32>
    %v906 = stablehlo.reduce(%v904 init: %v905) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v907 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v908 = stablehlo.divide %v906, %v907 : tensor<32x672xf32>
    %v909 = stablehlo.dot_general %v908, %b10zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v910 = stablehlo.broadcast_in_dim %b10zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v911 = stablehlo.add %v909, %v910 : tensor<32x28xf32>
    %v912 = stablehlo.logistic %v911 : tensor<32x28xf32>
    %v913 = stablehlo.multiply %v911, %v912 : tensor<32x28xf32>
    %v914 = stablehlo.dot_general %v913, %b10zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v915 = stablehlo.broadcast_in_dim %b10zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v916 = stablehlo.add %v914, %v915 : tensor<32x672xf32>
    %v917 = stablehlo.reshape %v903 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v918 = stablehlo.constant dense<0.0> : tensor<f32>
    %v919 = stablehlo.reduce(%v917 init: %v918) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v920 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v921 = stablehlo.divide %v919, %v920 : tensor<32x672xf32>
    %v922 = stablehlo.dot_general %v921, %b10zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v923 = stablehlo.broadcast_in_dim %b10zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v924 = stablehlo.add %v922, %v923 : tensor<32x28xf32>
    %v925 = stablehlo.logistic %v924 : tensor<32x28xf32>
    %v926 = stablehlo.multiply %v924, %v925 : tensor<32x28xf32>
    %v927 = stablehlo.dot_general %v926, %b10zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v928 = stablehlo.broadcast_in_dim %b10zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v929 = stablehlo.add %v927, %v928 : tensor<32x672xf32>
    %v930 = stablehlo.logistic %v929 : tensor<32x672xf32>
    %v931 = stablehlo.broadcast_in_dim %v930, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v932 = stablehlo.multiply %v917, %v931 : tensor<32x672x14x14xf32>
    %v933 = stablehlo.reshape %v932 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v934 = stablehlo.reshape %v933 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v935 = stablehlo.convolution(%v934, %b10pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v936 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v937 = stablehlo.add %v935, %v936 : tensor<32x112x14x14xf32>
    %v938 = stablehlo.reshape %v937 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v939 = stablehlo.reshape %v938 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v940 = stablehlo.broadcast_in_dim %b10pnmu, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v941 = stablehlo.subtract %v939, %v940 : tensor<32x112x14x14xf32>
    %v942 = stablehlo.broadcast_in_dim %b10pnvar, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v943 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v944 = stablehlo.add %v942, %v943 : tensor<32x112x14x14xf32>
    %v945 = stablehlo.rsqrt %v944 : tensor<32x112x14x14xf32>
    %v946 = stablehlo.multiply %v941, %v945 : tensor<32x112x14x14xf32>
    %v947 = stablehlo.broadcast_in_dim %b10pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v948 = stablehlo.broadcast_in_dim %b10pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v949 = stablehlo.multiply %v946, %v947 : tensor<32x112x14x14xf32>
    %v950 = stablehlo.add %v949, %v948 : tensor<32x112x14x14xf32>
    %v951 = stablehlo.reshape %v950 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v952 = stablehlo.reshape %v951 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v953 = stablehlo.broadcast_in_dim %dp9, dims = [0] : (tensor<32xf32>) -> tensor<32x112x14x14xf32>
    %v954 = stablehlo.multiply %v953, %v952 : tensor<32x112x14x14xf32>
    %v955 = stablehlo.reshape %v954 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v956 = stablehlo.reshape %v955 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v957 = stablehlo.reshape %v859 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v958 = stablehlo.add %v956, %v957 : tensor<32x112x14x14xf32>
    %v959 = stablehlo.reshape %v958 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v960 = stablehlo.reshape %v959 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v961 = stablehlo.convolution(%v960, %b11eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v962 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v963 = stablehlo.add %v961, %v962 : tensor<32x672x14x14xf32>
    %v964 = stablehlo.reshape %v963 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v965 = stablehlo.reshape %v964 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v966 = stablehlo.broadcast_in_dim %b11enmu, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v967 = stablehlo.subtract %v965, %v966 : tensor<32x672x14x14xf32>
    %v968 = stablehlo.broadcast_in_dim %b11envar, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v969 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v970 = stablehlo.add %v968, %v969 : tensor<32x672x14x14xf32>
    %v971 = stablehlo.rsqrt %v970 : tensor<32x672x14x14xf32>
    %v972 = stablehlo.multiply %v967, %v971 : tensor<32x672x14x14xf32>
    %v973 = stablehlo.broadcast_in_dim %b11eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v974 = stablehlo.broadcast_in_dim %b11ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v975 = stablehlo.multiply %v972, %v973 : tensor<32x672x14x14xf32>
    %v976 = stablehlo.add %v975, %v974 : tensor<32x672x14x14xf32>
    %v977 = stablehlo.reshape %v976 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v978 = stablehlo.reshape %v977 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v979 = stablehlo.logistic %v978 : tensor<32x672x14x14xf32>
    %v980 = stablehlo.multiply %v978, %v979 : tensor<32x672x14x14xf32>
    %v981 = stablehlo.reshape %v980 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v982 = stablehlo.reshape %v981 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v983 = stablehlo.convolution(%v982, %b11dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %v984 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v985 = stablehlo.add %v983, %v984 : tensor<32x672x14x14xf32>
    %v986 = stablehlo.reshape %v985 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v987 = stablehlo.reshape %v986 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v988 = stablehlo.broadcast_in_dim %b11dnmu, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v989 = stablehlo.subtract %v987, %v988 : tensor<32x672x14x14xf32>
    %v990 = stablehlo.broadcast_in_dim %b11dnvar, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v991 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v992 = stablehlo.add %v990, %v991 : tensor<32x672x14x14xf32>
    %v993 = stablehlo.rsqrt %v992 : tensor<32x672x14x14xf32>
    %v994 = stablehlo.multiply %v989, %v993 : tensor<32x672x14x14xf32>
    %v995 = stablehlo.broadcast_in_dim %b11dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v996 = stablehlo.broadcast_in_dim %b11dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v997 = stablehlo.multiply %v994, %v995 : tensor<32x672x14x14xf32>
    %v998 = stablehlo.add %v997, %v996 : tensor<32x672x14x14xf32>
    %v999 = stablehlo.reshape %v998 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1000 = stablehlo.reshape %v999 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1001 = stablehlo.logistic %v1000 : tensor<32x672x14x14xf32>
    %v1002 = stablehlo.multiply %v1000, %v1001 : tensor<32x672x14x14xf32>
    %v1003 = stablehlo.reshape %v1002 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1004 = stablehlo.reshape %v1003 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1005 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1006 = stablehlo.reduce(%v1004 init: %v1005) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v1007 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v1008 = stablehlo.divide %v1006, %v1007 : tensor<32x672xf32>
    %v1009 = stablehlo.dot_general %v1008, %b11zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v1010 = stablehlo.broadcast_in_dim %b11zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v1011 = stablehlo.add %v1009, %v1010 : tensor<32x28xf32>
    %v1012 = stablehlo.logistic %v1011 : tensor<32x28xf32>
    %v1013 = stablehlo.multiply %v1011, %v1012 : tensor<32x28xf32>
    %v1014 = stablehlo.dot_general %v1013, %b11zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v1015 = stablehlo.broadcast_in_dim %b11zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v1016 = stablehlo.add %v1014, %v1015 : tensor<32x672xf32>
    %v1017 = stablehlo.reshape %v1003 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1018 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1019 = stablehlo.reduce(%v1017 init: %v1018) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v1020 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v1021 = stablehlo.divide %v1019, %v1020 : tensor<32x672xf32>
    %v1022 = stablehlo.dot_general %v1021, %b11zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v1023 = stablehlo.broadcast_in_dim %b11zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v1024 = stablehlo.add %v1022, %v1023 : tensor<32x28xf32>
    %v1025 = stablehlo.logistic %v1024 : tensor<32x28xf32>
    %v1026 = stablehlo.multiply %v1024, %v1025 : tensor<32x28xf32>
    %v1027 = stablehlo.dot_general %v1026, %b11zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v1028 = stablehlo.broadcast_in_dim %b11zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v1029 = stablehlo.add %v1027, %v1028 : tensor<32x672xf32>
    %v1030 = stablehlo.logistic %v1029 : tensor<32x672xf32>
    %v1031 = stablehlo.broadcast_in_dim %v1030, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v1032 = stablehlo.multiply %v1017, %v1031 : tensor<32x672x14x14xf32>
    %v1033 = stablehlo.reshape %v1032 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1034 = stablehlo.reshape %v1033 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1035 = stablehlo.convolution(%v1034, %b11pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v1036 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1037 = stablehlo.add %v1035, %v1036 : tensor<32x112x14x14xf32>
    %v1038 = stablehlo.reshape %v1037 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1039 = stablehlo.reshape %v1038 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1040 = stablehlo.broadcast_in_dim %b11pnmu, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1041 = stablehlo.subtract %v1039, %v1040 : tensor<32x112x14x14xf32>
    %v1042 = stablehlo.broadcast_in_dim %b11pnvar, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1043 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v1044 = stablehlo.add %v1042, %v1043 : tensor<32x112x14x14xf32>
    %v1045 = stablehlo.rsqrt %v1044 : tensor<32x112x14x14xf32>
    %v1046 = stablehlo.multiply %v1041, %v1045 : tensor<32x112x14x14xf32>
    %v1047 = stablehlo.broadcast_in_dim %b11pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1048 = stablehlo.broadcast_in_dim %b11pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1049 = stablehlo.multiply %v1046, %v1047 : tensor<32x112x14x14xf32>
    %v1050 = stablehlo.add %v1049, %v1048 : tensor<32x112x14x14xf32>
    %v1051 = stablehlo.reshape %v1050 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1052 = stablehlo.reshape %v1051 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1053 = stablehlo.broadcast_in_dim %dp10, dims = [0] : (tensor<32xf32>) -> tensor<32x112x14x14xf32>
    %v1054 = stablehlo.multiply %v1053, %v1052 : tensor<32x112x14x14xf32>
    %v1055 = stablehlo.reshape %v1054 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1056 = stablehlo.reshape %v1055 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1057 = stablehlo.reshape %v959 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1058 = stablehlo.add %v1056, %v1057 : tensor<32x112x14x14xf32>
    %v1059 = stablehlo.reshape %v1058 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1060 = stablehlo.reshape %v1059 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1061 = stablehlo.convolution(%v1060, %b12eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v1062 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1063 = stablehlo.add %v1061, %v1062 : tensor<32x672x14x14xf32>
    %v1064 = stablehlo.reshape %v1063 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1065 = stablehlo.reshape %v1064 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1066 = stablehlo.broadcast_in_dim %b12enmu, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1067 = stablehlo.subtract %v1065, %v1066 : tensor<32x672x14x14xf32>
    %v1068 = stablehlo.broadcast_in_dim %b12envar, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1069 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v1070 = stablehlo.add %v1068, %v1069 : tensor<32x672x14x14xf32>
    %v1071 = stablehlo.rsqrt %v1070 : tensor<32x672x14x14xf32>
    %v1072 = stablehlo.multiply %v1067, %v1071 : tensor<32x672x14x14xf32>
    %v1073 = stablehlo.broadcast_in_dim %b12eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1074 = stablehlo.broadcast_in_dim %b12ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1075 = stablehlo.multiply %v1072, %v1073 : tensor<32x672x14x14xf32>
    %v1076 = stablehlo.add %v1075, %v1074 : tensor<32x672x14x14xf32>
    %v1077 = stablehlo.reshape %v1076 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1078 = stablehlo.reshape %v1077 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1079 = stablehlo.logistic %v1078 : tensor<32x672x14x14xf32>
    %v1080 = stablehlo.multiply %v1078, %v1079 : tensor<32x672x14x14xf32>
    %v1081 = stablehlo.reshape %v1080 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1082 = stablehlo.reshape %v1081 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1083 = stablehlo.convolution(%v1082, %b12dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x7x7xf32>
    %v1084 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1085 = stablehlo.add %v1083, %v1084 : tensor<32x672x7x7xf32>
    %v1086 = stablehlo.reshape %v1085 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v1087 = stablehlo.reshape %v1086 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1088 = stablehlo.broadcast_in_dim %b12dnmu, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1089 = stablehlo.subtract %v1087, %v1088 : tensor<32x672x7x7xf32>
    %v1090 = stablehlo.broadcast_in_dim %b12dnvar, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1091 = stablehlo.constant dense<1.0e-5> : tensor<32x672x7x7xf32>
    %v1092 = stablehlo.add %v1090, %v1091 : tensor<32x672x7x7xf32>
    %v1093 = stablehlo.rsqrt %v1092 : tensor<32x672x7x7xf32>
    %v1094 = stablehlo.multiply %v1089, %v1093 : tensor<32x672x7x7xf32>
    %v1095 = stablehlo.broadcast_in_dim %b12dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1096 = stablehlo.broadcast_in_dim %b12dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1097 = stablehlo.multiply %v1094, %v1095 : tensor<32x672x7x7xf32>
    %v1098 = stablehlo.add %v1097, %v1096 : tensor<32x672x7x7xf32>
    %v1099 = stablehlo.reshape %v1098 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v1100 = stablehlo.reshape %v1099 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1101 = stablehlo.logistic %v1100 : tensor<32x672x7x7xf32>
    %v1102 = stablehlo.multiply %v1100, %v1101 : tensor<32x672x7x7xf32>
    %v1103 = stablehlo.reshape %v1102 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v1104 = stablehlo.reshape %v1103 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1105 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1106 = stablehlo.reduce(%v1104 init: %v1105) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v1107 = stablehlo.constant dense<49.0> : tensor<32x672xf32>
    %v1108 = stablehlo.divide %v1106, %v1107 : tensor<32x672xf32>
    %v1109 = stablehlo.dot_general %v1108, %b12zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v1110 = stablehlo.broadcast_in_dim %b12zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v1111 = stablehlo.add %v1109, %v1110 : tensor<32x28xf32>
    %v1112 = stablehlo.logistic %v1111 : tensor<32x28xf32>
    %v1113 = stablehlo.multiply %v1111, %v1112 : tensor<32x28xf32>
    %v1114 = stablehlo.dot_general %v1113, %b12zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v1115 = stablehlo.broadcast_in_dim %b12zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v1116 = stablehlo.add %v1114, %v1115 : tensor<32x672xf32>
    %v1117 = stablehlo.reshape %v1103 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1118 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1119 = stablehlo.reduce(%v1117 init: %v1118) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v1120 = stablehlo.constant dense<49.0> : tensor<32x672xf32>
    %v1121 = stablehlo.divide %v1119, %v1120 : tensor<32x672xf32>
    %v1122 = stablehlo.dot_general %v1121, %b12zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v1123 = stablehlo.broadcast_in_dim %b12zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v1124 = stablehlo.add %v1122, %v1123 : tensor<32x28xf32>
    %v1125 = stablehlo.logistic %v1124 : tensor<32x28xf32>
    %v1126 = stablehlo.multiply %v1124, %v1125 : tensor<32x28xf32>
    %v1127 = stablehlo.dot_general %v1126, %b12zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v1128 = stablehlo.broadcast_in_dim %b12zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v1129 = stablehlo.add %v1127, %v1128 : tensor<32x672xf32>
    %v1130 = stablehlo.logistic %v1129 : tensor<32x672xf32>
    %v1131 = stablehlo.broadcast_in_dim %v1130, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x7x7xf32>
    %v1132 = stablehlo.multiply %v1117, %v1131 : tensor<32x672x7x7xf32>
    %v1133 = stablehlo.reshape %v1132 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v1134 = stablehlo.reshape %v1133 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1135 = stablehlo.convolution(%v1134, %b12pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x7x7xf32>, tensor<192x672x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1136 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1137 = stablehlo.add %v1135, %v1136 : tensor<32x192x7x7xf32>
    %v1138 = stablehlo.reshape %v1137 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1139 = stablehlo.reshape %v1138 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1140 = stablehlo.broadcast_in_dim %b12pnmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1141 = stablehlo.subtract %v1139, %v1140 : tensor<32x192x7x7xf32>
    %v1142 = stablehlo.broadcast_in_dim %b12pnvar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1143 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1144 = stablehlo.add %v1142, %v1143 : tensor<32x192x7x7xf32>
    %v1145 = stablehlo.rsqrt %v1144 : tensor<32x192x7x7xf32>
    %v1146 = stablehlo.multiply %v1141, %v1145 : tensor<32x192x7x7xf32>
    %v1147 = stablehlo.broadcast_in_dim %b12pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1148 = stablehlo.broadcast_in_dim %b12pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1149 = stablehlo.multiply %v1146, %v1147 : tensor<32x192x7x7xf32>
    %v1150 = stablehlo.add %v1149, %v1148 : tensor<32x192x7x7xf32>
    %v1151 = stablehlo.reshape %v1150 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1152 = stablehlo.reshape %v1151 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1153 = stablehlo.convolution(%v1152, %b13eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1154 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1155 = stablehlo.add %v1153, %v1154 : tensor<32x1152x7x7xf32>
    %v1156 = stablehlo.reshape %v1155 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1157 = stablehlo.reshape %v1156 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1158 = stablehlo.broadcast_in_dim %b13enmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1159 = stablehlo.subtract %v1157, %v1158 : tensor<32x1152x7x7xf32>
    %v1160 = stablehlo.broadcast_in_dim %b13envar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1161 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1162 = stablehlo.add %v1160, %v1161 : tensor<32x1152x7x7xf32>
    %v1163 = stablehlo.rsqrt %v1162 : tensor<32x1152x7x7xf32>
    %v1164 = stablehlo.multiply %v1159, %v1163 : tensor<32x1152x7x7xf32>
    %v1165 = stablehlo.broadcast_in_dim %b13eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1166 = stablehlo.broadcast_in_dim %b13ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1167 = stablehlo.multiply %v1164, %v1165 : tensor<32x1152x7x7xf32>
    %v1168 = stablehlo.add %v1167, %v1166 : tensor<32x1152x7x7xf32>
    %v1169 = stablehlo.reshape %v1168 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1170 = stablehlo.reshape %v1169 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1171 = stablehlo.logistic %v1170 : tensor<32x1152x7x7xf32>
    %v1172 = stablehlo.multiply %v1170, %v1171 : tensor<32x1152x7x7xf32>
    %v1173 = stablehlo.reshape %v1172 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1174 = stablehlo.reshape %v1173 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1175 = stablehlo.convolution(%v1174, %b13dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v1176 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1177 = stablehlo.add %v1175, %v1176 : tensor<32x1152x7x7xf32>
    %v1178 = stablehlo.reshape %v1177 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1179 = stablehlo.reshape %v1178 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1180 = stablehlo.broadcast_in_dim %b13dnmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1181 = stablehlo.subtract %v1179, %v1180 : tensor<32x1152x7x7xf32>
    %v1182 = stablehlo.broadcast_in_dim %b13dnvar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1183 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1184 = stablehlo.add %v1182, %v1183 : tensor<32x1152x7x7xf32>
    %v1185 = stablehlo.rsqrt %v1184 : tensor<32x1152x7x7xf32>
    %v1186 = stablehlo.multiply %v1181, %v1185 : tensor<32x1152x7x7xf32>
    %v1187 = stablehlo.broadcast_in_dim %b13dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1188 = stablehlo.broadcast_in_dim %b13dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1189 = stablehlo.multiply %v1186, %v1187 : tensor<32x1152x7x7xf32>
    %v1190 = stablehlo.add %v1189, %v1188 : tensor<32x1152x7x7xf32>
    %v1191 = stablehlo.reshape %v1190 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1192 = stablehlo.reshape %v1191 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1193 = stablehlo.logistic %v1192 : tensor<32x1152x7x7xf32>
    %v1194 = stablehlo.multiply %v1192, %v1193 : tensor<32x1152x7x7xf32>
    %v1195 = stablehlo.reshape %v1194 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1196 = stablehlo.reshape %v1195 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1197 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1198 = stablehlo.reduce(%v1196 init: %v1197) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1199 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1200 = stablehlo.divide %v1198, %v1199 : tensor<32x1152xf32>
    %v1201 = stablehlo.dot_general %v1200, %b13zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1202 = stablehlo.broadcast_in_dim %b13zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1203 = stablehlo.add %v1201, %v1202 : tensor<32x48xf32>
    %v1204 = stablehlo.logistic %v1203 : tensor<32x48xf32>
    %v1205 = stablehlo.multiply %v1203, %v1204 : tensor<32x48xf32>
    %v1206 = stablehlo.dot_general %v1205, %b13zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1207 = stablehlo.broadcast_in_dim %b13zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1208 = stablehlo.add %v1206, %v1207 : tensor<32x1152xf32>
    %v1209 = stablehlo.reshape %v1195 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1210 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1211 = stablehlo.reduce(%v1209 init: %v1210) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1212 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1213 = stablehlo.divide %v1211, %v1212 : tensor<32x1152xf32>
    %v1214 = stablehlo.dot_general %v1213, %b13zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1215 = stablehlo.broadcast_in_dim %b13zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1216 = stablehlo.add %v1214, %v1215 : tensor<32x48xf32>
    %v1217 = stablehlo.logistic %v1216 : tensor<32x48xf32>
    %v1218 = stablehlo.multiply %v1216, %v1217 : tensor<32x48xf32>
    %v1219 = stablehlo.dot_general %v1218, %b13zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1220 = stablehlo.broadcast_in_dim %b13zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1221 = stablehlo.add %v1219, %v1220 : tensor<32x1152xf32>
    %v1222 = stablehlo.logistic %v1221 : tensor<32x1152xf32>
    %v1223 = stablehlo.broadcast_in_dim %v1222, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1224 = stablehlo.multiply %v1209, %v1223 : tensor<32x1152x7x7xf32>
    %v1225 = stablehlo.reshape %v1224 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1226 = stablehlo.reshape %v1225 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1227 = stablehlo.convolution(%v1226, %b13pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1228 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1229 = stablehlo.add %v1227, %v1228 : tensor<32x192x7x7xf32>
    %v1230 = stablehlo.reshape %v1229 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1231 = stablehlo.reshape %v1230 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1232 = stablehlo.broadcast_in_dim %b13pnmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1233 = stablehlo.subtract %v1231, %v1232 : tensor<32x192x7x7xf32>
    %v1234 = stablehlo.broadcast_in_dim %b13pnvar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1235 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1236 = stablehlo.add %v1234, %v1235 : tensor<32x192x7x7xf32>
    %v1237 = stablehlo.rsqrt %v1236 : tensor<32x192x7x7xf32>
    %v1238 = stablehlo.multiply %v1233, %v1237 : tensor<32x192x7x7xf32>
    %v1239 = stablehlo.broadcast_in_dim %b13pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1240 = stablehlo.broadcast_in_dim %b13pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1241 = stablehlo.multiply %v1238, %v1239 : tensor<32x192x7x7xf32>
    %v1242 = stablehlo.add %v1241, %v1240 : tensor<32x192x7x7xf32>
    %v1243 = stablehlo.reshape %v1242 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1244 = stablehlo.reshape %v1243 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1245 = stablehlo.broadcast_in_dim %dp12, dims = [0] : (tensor<32xf32>) -> tensor<32x192x7x7xf32>
    %v1246 = stablehlo.multiply %v1245, %v1244 : tensor<32x192x7x7xf32>
    %v1247 = stablehlo.reshape %v1246 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1248 = stablehlo.reshape %v1247 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1249 = stablehlo.reshape %v1151 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1250 = stablehlo.add %v1248, %v1249 : tensor<32x192x7x7xf32>
    %v1251 = stablehlo.reshape %v1250 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1252 = stablehlo.reshape %v1251 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1253 = stablehlo.convolution(%v1252, %b14eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1254 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1255 = stablehlo.add %v1253, %v1254 : tensor<32x1152x7x7xf32>
    %v1256 = stablehlo.reshape %v1255 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1257 = stablehlo.reshape %v1256 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1258 = stablehlo.broadcast_in_dim %b14enmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1259 = stablehlo.subtract %v1257, %v1258 : tensor<32x1152x7x7xf32>
    %v1260 = stablehlo.broadcast_in_dim %b14envar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1261 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1262 = stablehlo.add %v1260, %v1261 : tensor<32x1152x7x7xf32>
    %v1263 = stablehlo.rsqrt %v1262 : tensor<32x1152x7x7xf32>
    %v1264 = stablehlo.multiply %v1259, %v1263 : tensor<32x1152x7x7xf32>
    %v1265 = stablehlo.broadcast_in_dim %b14eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1266 = stablehlo.broadcast_in_dim %b14ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1267 = stablehlo.multiply %v1264, %v1265 : tensor<32x1152x7x7xf32>
    %v1268 = stablehlo.add %v1267, %v1266 : tensor<32x1152x7x7xf32>
    %v1269 = stablehlo.reshape %v1268 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1270 = stablehlo.reshape %v1269 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1271 = stablehlo.logistic %v1270 : tensor<32x1152x7x7xf32>
    %v1272 = stablehlo.multiply %v1270, %v1271 : tensor<32x1152x7x7xf32>
    %v1273 = stablehlo.reshape %v1272 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1274 = stablehlo.reshape %v1273 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1275 = stablehlo.convolution(%v1274, %b14dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v1276 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1277 = stablehlo.add %v1275, %v1276 : tensor<32x1152x7x7xf32>
    %v1278 = stablehlo.reshape %v1277 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1279 = stablehlo.reshape %v1278 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1280 = stablehlo.broadcast_in_dim %b14dnmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1281 = stablehlo.subtract %v1279, %v1280 : tensor<32x1152x7x7xf32>
    %v1282 = stablehlo.broadcast_in_dim %b14dnvar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1283 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1284 = stablehlo.add %v1282, %v1283 : tensor<32x1152x7x7xf32>
    %v1285 = stablehlo.rsqrt %v1284 : tensor<32x1152x7x7xf32>
    %v1286 = stablehlo.multiply %v1281, %v1285 : tensor<32x1152x7x7xf32>
    %v1287 = stablehlo.broadcast_in_dim %b14dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1288 = stablehlo.broadcast_in_dim %b14dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1289 = stablehlo.multiply %v1286, %v1287 : tensor<32x1152x7x7xf32>
    %v1290 = stablehlo.add %v1289, %v1288 : tensor<32x1152x7x7xf32>
    %v1291 = stablehlo.reshape %v1290 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1292 = stablehlo.reshape %v1291 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1293 = stablehlo.logistic %v1292 : tensor<32x1152x7x7xf32>
    %v1294 = stablehlo.multiply %v1292, %v1293 : tensor<32x1152x7x7xf32>
    %v1295 = stablehlo.reshape %v1294 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1296 = stablehlo.reshape %v1295 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1297 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1298 = stablehlo.reduce(%v1296 init: %v1297) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1299 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1300 = stablehlo.divide %v1298, %v1299 : tensor<32x1152xf32>
    %v1301 = stablehlo.dot_general %v1300, %b14zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1302 = stablehlo.broadcast_in_dim %b14zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1303 = stablehlo.add %v1301, %v1302 : tensor<32x48xf32>
    %v1304 = stablehlo.logistic %v1303 : tensor<32x48xf32>
    %v1305 = stablehlo.multiply %v1303, %v1304 : tensor<32x48xf32>
    %v1306 = stablehlo.dot_general %v1305, %b14zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1307 = stablehlo.broadcast_in_dim %b14zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1308 = stablehlo.add %v1306, %v1307 : tensor<32x1152xf32>
    %v1309 = stablehlo.reshape %v1295 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1310 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1311 = stablehlo.reduce(%v1309 init: %v1310) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1312 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1313 = stablehlo.divide %v1311, %v1312 : tensor<32x1152xf32>
    %v1314 = stablehlo.dot_general %v1313, %b14zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1315 = stablehlo.broadcast_in_dim %b14zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1316 = stablehlo.add %v1314, %v1315 : tensor<32x48xf32>
    %v1317 = stablehlo.logistic %v1316 : tensor<32x48xf32>
    %v1318 = stablehlo.multiply %v1316, %v1317 : tensor<32x48xf32>
    %v1319 = stablehlo.dot_general %v1318, %b14zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1320 = stablehlo.broadcast_in_dim %b14zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1321 = stablehlo.add %v1319, %v1320 : tensor<32x1152xf32>
    %v1322 = stablehlo.logistic %v1321 : tensor<32x1152xf32>
    %v1323 = stablehlo.broadcast_in_dim %v1322, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1324 = stablehlo.multiply %v1309, %v1323 : tensor<32x1152x7x7xf32>
    %v1325 = stablehlo.reshape %v1324 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1326 = stablehlo.reshape %v1325 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1327 = stablehlo.convolution(%v1326, %b14pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1328 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1329 = stablehlo.add %v1327, %v1328 : tensor<32x192x7x7xf32>
    %v1330 = stablehlo.reshape %v1329 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1331 = stablehlo.reshape %v1330 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1332 = stablehlo.broadcast_in_dim %b14pnmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1333 = stablehlo.subtract %v1331, %v1332 : tensor<32x192x7x7xf32>
    %v1334 = stablehlo.broadcast_in_dim %b14pnvar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1335 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1336 = stablehlo.add %v1334, %v1335 : tensor<32x192x7x7xf32>
    %v1337 = stablehlo.rsqrt %v1336 : tensor<32x192x7x7xf32>
    %v1338 = stablehlo.multiply %v1333, %v1337 : tensor<32x192x7x7xf32>
    %v1339 = stablehlo.broadcast_in_dim %b14pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1340 = stablehlo.broadcast_in_dim %b14pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1341 = stablehlo.multiply %v1338, %v1339 : tensor<32x192x7x7xf32>
    %v1342 = stablehlo.add %v1341, %v1340 : tensor<32x192x7x7xf32>
    %v1343 = stablehlo.reshape %v1342 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1344 = stablehlo.reshape %v1343 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1345 = stablehlo.broadcast_in_dim %dp13, dims = [0] : (tensor<32xf32>) -> tensor<32x192x7x7xf32>
    %v1346 = stablehlo.multiply %v1345, %v1344 : tensor<32x192x7x7xf32>
    %v1347 = stablehlo.reshape %v1346 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1348 = stablehlo.reshape %v1347 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1349 = stablehlo.reshape %v1251 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1350 = stablehlo.add %v1348, %v1349 : tensor<32x192x7x7xf32>
    %v1351 = stablehlo.reshape %v1350 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1352 = stablehlo.reshape %v1351 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1353 = stablehlo.convolution(%v1352, %b15eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1354 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1355 = stablehlo.add %v1353, %v1354 : tensor<32x1152x7x7xf32>
    %v1356 = stablehlo.reshape %v1355 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1357 = stablehlo.reshape %v1356 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1358 = stablehlo.broadcast_in_dim %b15enmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1359 = stablehlo.subtract %v1357, %v1358 : tensor<32x1152x7x7xf32>
    %v1360 = stablehlo.broadcast_in_dim %b15envar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1361 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1362 = stablehlo.add %v1360, %v1361 : tensor<32x1152x7x7xf32>
    %v1363 = stablehlo.rsqrt %v1362 : tensor<32x1152x7x7xf32>
    %v1364 = stablehlo.multiply %v1359, %v1363 : tensor<32x1152x7x7xf32>
    %v1365 = stablehlo.broadcast_in_dim %b15eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1366 = stablehlo.broadcast_in_dim %b15ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1367 = stablehlo.multiply %v1364, %v1365 : tensor<32x1152x7x7xf32>
    %v1368 = stablehlo.add %v1367, %v1366 : tensor<32x1152x7x7xf32>
    %v1369 = stablehlo.reshape %v1368 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1370 = stablehlo.reshape %v1369 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1371 = stablehlo.logistic %v1370 : tensor<32x1152x7x7xf32>
    %v1372 = stablehlo.multiply %v1370, %v1371 : tensor<32x1152x7x7xf32>
    %v1373 = stablehlo.reshape %v1372 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1374 = stablehlo.reshape %v1373 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1375 = stablehlo.convolution(%v1374, %b15dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v1376 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1377 = stablehlo.add %v1375, %v1376 : tensor<32x1152x7x7xf32>
    %v1378 = stablehlo.reshape %v1377 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1379 = stablehlo.reshape %v1378 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1380 = stablehlo.broadcast_in_dim %b15dnmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1381 = stablehlo.subtract %v1379, %v1380 : tensor<32x1152x7x7xf32>
    %v1382 = stablehlo.broadcast_in_dim %b15dnvar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1383 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1384 = stablehlo.add %v1382, %v1383 : tensor<32x1152x7x7xf32>
    %v1385 = stablehlo.rsqrt %v1384 : tensor<32x1152x7x7xf32>
    %v1386 = stablehlo.multiply %v1381, %v1385 : tensor<32x1152x7x7xf32>
    %v1387 = stablehlo.broadcast_in_dim %b15dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1388 = stablehlo.broadcast_in_dim %b15dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1389 = stablehlo.multiply %v1386, %v1387 : tensor<32x1152x7x7xf32>
    %v1390 = stablehlo.add %v1389, %v1388 : tensor<32x1152x7x7xf32>
    %v1391 = stablehlo.reshape %v1390 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1392 = stablehlo.reshape %v1391 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1393 = stablehlo.logistic %v1392 : tensor<32x1152x7x7xf32>
    %v1394 = stablehlo.multiply %v1392, %v1393 : tensor<32x1152x7x7xf32>
    %v1395 = stablehlo.reshape %v1394 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1396 = stablehlo.reshape %v1395 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1397 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1398 = stablehlo.reduce(%v1396 init: %v1397) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1399 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1400 = stablehlo.divide %v1398, %v1399 : tensor<32x1152xf32>
    %v1401 = stablehlo.dot_general %v1400, %b15zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1402 = stablehlo.broadcast_in_dim %b15zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1403 = stablehlo.add %v1401, %v1402 : tensor<32x48xf32>
    %v1404 = stablehlo.logistic %v1403 : tensor<32x48xf32>
    %v1405 = stablehlo.multiply %v1403, %v1404 : tensor<32x48xf32>
    %v1406 = stablehlo.dot_general %v1405, %b15zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1407 = stablehlo.broadcast_in_dim %b15zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1408 = stablehlo.add %v1406, %v1407 : tensor<32x1152xf32>
    %v1409 = stablehlo.reshape %v1395 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1410 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1411 = stablehlo.reduce(%v1409 init: %v1410) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1412 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1413 = stablehlo.divide %v1411, %v1412 : tensor<32x1152xf32>
    %v1414 = stablehlo.dot_general %v1413, %b15zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1415 = stablehlo.broadcast_in_dim %b15zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1416 = stablehlo.add %v1414, %v1415 : tensor<32x48xf32>
    %v1417 = stablehlo.logistic %v1416 : tensor<32x48xf32>
    %v1418 = stablehlo.multiply %v1416, %v1417 : tensor<32x48xf32>
    %v1419 = stablehlo.dot_general %v1418, %b15zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1420 = stablehlo.broadcast_in_dim %b15zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1421 = stablehlo.add %v1419, %v1420 : tensor<32x1152xf32>
    %v1422 = stablehlo.logistic %v1421 : tensor<32x1152xf32>
    %v1423 = stablehlo.broadcast_in_dim %v1422, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1424 = stablehlo.multiply %v1409, %v1423 : tensor<32x1152x7x7xf32>
    %v1425 = stablehlo.reshape %v1424 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1426 = stablehlo.reshape %v1425 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1427 = stablehlo.convolution(%v1426, %b15pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1428 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1429 = stablehlo.add %v1427, %v1428 : tensor<32x192x7x7xf32>
    %v1430 = stablehlo.reshape %v1429 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1431 = stablehlo.reshape %v1430 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1432 = stablehlo.broadcast_in_dim %b15pnmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1433 = stablehlo.subtract %v1431, %v1432 : tensor<32x192x7x7xf32>
    %v1434 = stablehlo.broadcast_in_dim %b15pnvar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1435 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1436 = stablehlo.add %v1434, %v1435 : tensor<32x192x7x7xf32>
    %v1437 = stablehlo.rsqrt %v1436 : tensor<32x192x7x7xf32>
    %v1438 = stablehlo.multiply %v1433, %v1437 : tensor<32x192x7x7xf32>
    %v1439 = stablehlo.broadcast_in_dim %b15pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1440 = stablehlo.broadcast_in_dim %b15pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1441 = stablehlo.multiply %v1438, %v1439 : tensor<32x192x7x7xf32>
    %v1442 = stablehlo.add %v1441, %v1440 : tensor<32x192x7x7xf32>
    %v1443 = stablehlo.reshape %v1442 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1444 = stablehlo.reshape %v1443 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1445 = stablehlo.broadcast_in_dim %dp14, dims = [0] : (tensor<32xf32>) -> tensor<32x192x7x7xf32>
    %v1446 = stablehlo.multiply %v1445, %v1444 : tensor<32x192x7x7xf32>
    %v1447 = stablehlo.reshape %v1446 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1448 = stablehlo.reshape %v1447 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1449 = stablehlo.reshape %v1351 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1450 = stablehlo.add %v1448, %v1449 : tensor<32x192x7x7xf32>
    %v1451 = stablehlo.reshape %v1450 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1452 = stablehlo.reshape %v1451 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1453 = stablehlo.convolution(%v1452, %b16eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1454 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1455 = stablehlo.add %v1453, %v1454 : tensor<32x1152x7x7xf32>
    %v1456 = stablehlo.reshape %v1455 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1457 = stablehlo.reshape %v1456 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1458 = stablehlo.broadcast_in_dim %b16enmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1459 = stablehlo.subtract %v1457, %v1458 : tensor<32x1152x7x7xf32>
    %v1460 = stablehlo.broadcast_in_dim %b16envar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1461 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1462 = stablehlo.add %v1460, %v1461 : tensor<32x1152x7x7xf32>
    %v1463 = stablehlo.rsqrt %v1462 : tensor<32x1152x7x7xf32>
    %v1464 = stablehlo.multiply %v1459, %v1463 : tensor<32x1152x7x7xf32>
    %v1465 = stablehlo.broadcast_in_dim %b16eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1466 = stablehlo.broadcast_in_dim %b16ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1467 = stablehlo.multiply %v1464, %v1465 : tensor<32x1152x7x7xf32>
    %v1468 = stablehlo.add %v1467, %v1466 : tensor<32x1152x7x7xf32>
    %v1469 = stablehlo.reshape %v1468 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1470 = stablehlo.reshape %v1469 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1471 = stablehlo.logistic %v1470 : tensor<32x1152x7x7xf32>
    %v1472 = stablehlo.multiply %v1470, %v1471 : tensor<32x1152x7x7xf32>
    %v1473 = stablehlo.reshape %v1472 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1474 = stablehlo.reshape %v1473 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1475 = stablehlo.convolution(%v1474, %b16dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x3x3xf32>) -> tensor<32x1152x7x7xf32>
    %v1476 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1477 = stablehlo.add %v1475, %v1476 : tensor<32x1152x7x7xf32>
    %v1478 = stablehlo.reshape %v1477 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1479 = stablehlo.reshape %v1478 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1480 = stablehlo.broadcast_in_dim %b16dnmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1481 = stablehlo.subtract %v1479, %v1480 : tensor<32x1152x7x7xf32>
    %v1482 = stablehlo.broadcast_in_dim %b16dnvar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1483 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1484 = stablehlo.add %v1482, %v1483 : tensor<32x1152x7x7xf32>
    %v1485 = stablehlo.rsqrt %v1484 : tensor<32x1152x7x7xf32>
    %v1486 = stablehlo.multiply %v1481, %v1485 : tensor<32x1152x7x7xf32>
    %v1487 = stablehlo.broadcast_in_dim %b16dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1488 = stablehlo.broadcast_in_dim %b16dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1489 = stablehlo.multiply %v1486, %v1487 : tensor<32x1152x7x7xf32>
    %v1490 = stablehlo.add %v1489, %v1488 : tensor<32x1152x7x7xf32>
    %v1491 = stablehlo.reshape %v1490 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1492 = stablehlo.reshape %v1491 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1493 = stablehlo.logistic %v1492 : tensor<32x1152x7x7xf32>
    %v1494 = stablehlo.multiply %v1492, %v1493 : tensor<32x1152x7x7xf32>
    %v1495 = stablehlo.reshape %v1494 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1496 = stablehlo.reshape %v1495 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1497 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1498 = stablehlo.reduce(%v1496 init: %v1497) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1499 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1500 = stablehlo.divide %v1498, %v1499 : tensor<32x1152xf32>
    %v1501 = stablehlo.dot_general %v1500, %b16zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1502 = stablehlo.broadcast_in_dim %b16zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1503 = stablehlo.add %v1501, %v1502 : tensor<32x48xf32>
    %v1504 = stablehlo.logistic %v1503 : tensor<32x48xf32>
    %v1505 = stablehlo.multiply %v1503, %v1504 : tensor<32x48xf32>
    %v1506 = stablehlo.dot_general %v1505, %b16zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1507 = stablehlo.broadcast_in_dim %b16zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1508 = stablehlo.add %v1506, %v1507 : tensor<32x1152xf32>
    %v1509 = stablehlo.reshape %v1495 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1510 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1511 = stablehlo.reduce(%v1509 init: %v1510) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1512 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1513 = stablehlo.divide %v1511, %v1512 : tensor<32x1152xf32>
    %v1514 = stablehlo.dot_general %v1513, %b16zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1515 = stablehlo.broadcast_in_dim %b16zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1516 = stablehlo.add %v1514, %v1515 : tensor<32x48xf32>
    %v1517 = stablehlo.logistic %v1516 : tensor<32x48xf32>
    %v1518 = stablehlo.multiply %v1516, %v1517 : tensor<32x48xf32>
    %v1519 = stablehlo.dot_general %v1518, %b16zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1520 = stablehlo.broadcast_in_dim %b16zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1521 = stablehlo.add %v1519, %v1520 : tensor<32x1152xf32>
    %v1522 = stablehlo.logistic %v1521 : tensor<32x1152xf32>
    %v1523 = stablehlo.broadcast_in_dim %v1522, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1524 = stablehlo.multiply %v1509, %v1523 : tensor<32x1152x7x7xf32>
    %v1525 = stablehlo.reshape %v1524 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1526 = stablehlo.reshape %v1525 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1527 = stablehlo.convolution(%v1526, %b16pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<320x1152x1x1xf32>) -> tensor<32x320x7x7xf32>
    %v1528 = stablehlo.broadcast_in_dim %zb320, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1529 = stablehlo.add %v1527, %v1528 : tensor<32x320x7x7xf32>
    %v1530 = stablehlo.reshape %v1529 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1531 = stablehlo.reshape %v1530 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1532 = stablehlo.broadcast_in_dim %b16pnmu, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1533 = stablehlo.subtract %v1531, %v1532 : tensor<32x320x7x7xf32>
    %v1534 = stablehlo.broadcast_in_dim %b16pnvar, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1535 = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %v1536 = stablehlo.add %v1534, %v1535 : tensor<32x320x7x7xf32>
    %v1537 = stablehlo.rsqrt %v1536 : tensor<32x320x7x7xf32>
    %v1538 = stablehlo.multiply %v1533, %v1537 : tensor<32x320x7x7xf32>
    %v1539 = stablehlo.broadcast_in_dim %b16pg, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1540 = stablehlo.broadcast_in_dim %b16pbt, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1541 = stablehlo.multiply %v1538, %v1539 : tensor<32x320x7x7xf32>
    %v1542 = stablehlo.add %v1541, %v1540 : tensor<32x320x7x7xf32>
    %v1543 = stablehlo.reshape %v1542 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1544 = stablehlo.reshape %v1543 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1545 = stablehlo.convolution(%v1544, %hW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x320x7x7xf32>, tensor<1280x320x1x1xf32>) -> tensor<32x1280x7x7xf32>
    %v1546 = stablehlo.broadcast_in_dim %zb1280, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1547 = stablehlo.add %v1545, %v1546 : tensor<32x1280x7x7xf32>
    %v1548 = stablehlo.reshape %v1547 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1549 = stablehlo.reshape %v1548 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1550 = stablehlo.broadcast_in_dim %hnmu, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1551 = stablehlo.subtract %v1549, %v1550 : tensor<32x1280x7x7xf32>
    %v1552 = stablehlo.broadcast_in_dim %hnvar, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1553 = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %v1554 = stablehlo.add %v1552, %v1553 : tensor<32x1280x7x7xf32>
    %v1555 = stablehlo.rsqrt %v1554 : tensor<32x1280x7x7xf32>
    %v1556 = stablehlo.multiply %v1551, %v1555 : tensor<32x1280x7x7xf32>
    %v1557 = stablehlo.broadcast_in_dim %hg, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1558 = stablehlo.broadcast_in_dim %hbt, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1559 = stablehlo.multiply %v1556, %v1557 : tensor<32x1280x7x7xf32>
    %v1560 = stablehlo.add %v1559, %v1558 : tensor<32x1280x7x7xf32>
    %v1561 = stablehlo.reshape %v1560 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1562 = stablehlo.reshape %v1561 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1563 = stablehlo.logistic %v1562 : tensor<32x1280x7x7xf32>
    %v1564 = stablehlo.multiply %v1562, %v1563 : tensor<32x1280x7x7xf32>
    %v1565 = stablehlo.reshape %v1564 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1566 = stablehlo.reshape %v1565 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1567 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1568 = stablehlo.reduce(%v1566 init: %v1567) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v1569 = stablehlo.constant dense<49.0> : tensor<32x1280xf32>
    %v1570 = stablehlo.divide %v1568, %v1569 : tensor<32x1280xf32>
    %v1571 = stablehlo.dot_general %v1570, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1280xf32>, tensor<1280x10xf32>) -> tensor<32x10xf32>
    %v1572 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v1573 = stablehlo.add %v1571, %v1572 : tensor<32x10xf32>
    return %v1573 : tensor<32x10xf32>
  }
}
