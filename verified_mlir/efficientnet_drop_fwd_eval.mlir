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
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
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
    %v18 = stablehlo.logistic %v17 : tensor<32x401408xf32>
    %v19 = stablehlo.multiply %v17, %v18 : tensor<32x401408xf32>
    %v20 = stablehlo.reshape %v19 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v21 = stablehlo.convolution(%v20, %b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 32 : i64} : (tensor<32x32x112x112xf32>, tensor<32x1x3x3xf32>) -> tensor<32x32x112x112xf32>
    %v22 = stablehlo.broadcast_in_dim %zb32, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v23 = stablehlo.add %v21, %v22 : tensor<32x32x112x112xf32>
    %v24 = stablehlo.reshape %v23 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v25 = stablehlo.reshape %v24 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v26 = stablehlo.broadcast_in_dim %b1dnmu, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v27 = stablehlo.subtract %v25, %v26 : tensor<32x32x112x112xf32>
    %v28 = stablehlo.broadcast_in_dim %b1dnvar, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v29 = stablehlo.constant dense<1.0e-5> : tensor<32x32x112x112xf32>
    %v30 = stablehlo.add %v28, %v29 : tensor<32x32x112x112xf32>
    %v31 = stablehlo.rsqrt %v30 : tensor<32x32x112x112xf32>
    %v32 = stablehlo.multiply %v27, %v31 : tensor<32x32x112x112xf32>
    %v33 = stablehlo.broadcast_in_dim %b1dg, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v34 = stablehlo.broadcast_in_dim %b1dbt, dims = [1] : (tensor<32xf32>) -> tensor<32x32x112x112xf32>
    %v35 = stablehlo.multiply %v32, %v33 : tensor<32x32x112x112xf32>
    %v36 = stablehlo.add %v35, %v34 : tensor<32x32x112x112xf32>
    %v37 = stablehlo.reshape %v36 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v38 = stablehlo.logistic %v37 : tensor<32x401408xf32>
    %v39 = stablehlo.multiply %v37, %v38 : tensor<32x401408xf32>
    %v40 = stablehlo.reshape %v39 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v41 = stablehlo.constant dense<0.0> : tensor<f32>
    %v42 = stablehlo.reduce(%v40 init: %v41) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v43 = stablehlo.constant dense<12544.0> : tensor<32x32xf32>
    %v44 = stablehlo.divide %v42, %v43 : tensor<32x32xf32>
    %v45 = stablehlo.dot_general %v44, %b1zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x8xf32>) -> tensor<32x8xf32>
    %v46 = stablehlo.broadcast_in_dim %b1zb1, dims = [1] : (tensor<8xf32>) -> tensor<32x8xf32>
    %v47 = stablehlo.add %v45, %v46 : tensor<32x8xf32>
    %v48 = stablehlo.logistic %v47 : tensor<32x8xf32>
    %v49 = stablehlo.multiply %v47, %v48 : tensor<32x8xf32>
    %v50 = stablehlo.dot_general %v49, %b1zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x8xf32>, tensor<8x32xf32>) -> tensor<32x32xf32>
    %v51 = stablehlo.broadcast_in_dim %b1zb2, dims = [1] : (tensor<32xf32>) -> tensor<32x32xf32>
    %v52 = stablehlo.add %v50, %v51 : tensor<32x32xf32>
    %v53 = stablehlo.reshape %v39 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v54 = stablehlo.constant dense<0.0> : tensor<f32>
    %v55 = stablehlo.reduce(%v53 init: %v54) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x32x112x112xf32>, tensor<f32>) -> tensor<32x32xf32>
    %v56 = stablehlo.constant dense<12544.0> : tensor<32x32xf32>
    %v57 = stablehlo.divide %v55, %v56 : tensor<32x32xf32>
    %v58 = stablehlo.dot_general %v57, %b1zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x8xf32>) -> tensor<32x8xf32>
    %v59 = stablehlo.broadcast_in_dim %b1zb1, dims = [1] : (tensor<8xf32>) -> tensor<32x8xf32>
    %v60 = stablehlo.add %v58, %v59 : tensor<32x8xf32>
    %v61 = stablehlo.logistic %v60 : tensor<32x8xf32>
    %v62 = stablehlo.multiply %v60, %v61 : tensor<32x8xf32>
    %v63 = stablehlo.dot_general %v62, %b1zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x8xf32>, tensor<8x32xf32>) -> tensor<32x32xf32>
    %v64 = stablehlo.broadcast_in_dim %b1zb2, dims = [1] : (tensor<32xf32>) -> tensor<32x32xf32>
    %v65 = stablehlo.add %v63, %v64 : tensor<32x32xf32>
    %v66 = stablehlo.logistic %v65 : tensor<32x32xf32>
    %v67 = stablehlo.broadcast_in_dim %v66, dims = [0, 1] : (tensor<32x32xf32>) -> tensor<32x32x112x112xf32>
    %v68 = stablehlo.multiply %v53, %v67 : tensor<32x32x112x112xf32>
    %v69 = stablehlo.reshape %v68 : (tensor<32x32x112x112xf32>) -> tensor<32x401408xf32>
    %v70 = stablehlo.reshape %v69 : (tensor<32x401408xf32>) -> tensor<32x32x112x112xf32>
    %v71 = stablehlo.convolution(%v70, %b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x32x112x112xf32>, tensor<16x32x1x1xf32>) -> tensor<32x16x112x112xf32>
    %v72 = stablehlo.broadcast_in_dim %zb16, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v73 = stablehlo.add %v71, %v72 : tensor<32x16x112x112xf32>
    %v74 = stablehlo.reshape %v73 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v75 = stablehlo.reshape %v74 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v76 = stablehlo.broadcast_in_dim %b1pnmu, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v77 = stablehlo.subtract %v75, %v76 : tensor<32x16x112x112xf32>
    %v78 = stablehlo.broadcast_in_dim %b1pnvar, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v79 = stablehlo.constant dense<1.0e-5> : tensor<32x16x112x112xf32>
    %v80 = stablehlo.add %v78, %v79 : tensor<32x16x112x112xf32>
    %v81 = stablehlo.rsqrt %v80 : tensor<32x16x112x112xf32>
    %v82 = stablehlo.multiply %v77, %v81 : tensor<32x16x112x112xf32>
    %v83 = stablehlo.broadcast_in_dim %b1pg, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v84 = stablehlo.broadcast_in_dim %b1pbt, dims = [1] : (tensor<16xf32>) -> tensor<32x16x112x112xf32>
    %v85 = stablehlo.multiply %v82, %v83 : tensor<32x16x112x112xf32>
    %v86 = stablehlo.add %v85, %v84 : tensor<32x16x112x112xf32>
    %v87 = stablehlo.reshape %v86 : (tensor<32x16x112x112xf32>) -> tensor<32x200704xf32>
    %v88 = stablehlo.reshape %v87 : (tensor<32x200704xf32>) -> tensor<32x16x112x112xf32>
    %v89 = stablehlo.convolution(%v88, %b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x16x112x112xf32>, tensor<96x16x1x1xf32>) -> tensor<32x96x112x112xf32>
    %v90 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v91 = stablehlo.add %v89, %v90 : tensor<32x96x112x112xf32>
    %v92 = stablehlo.reshape %v91 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v93 = stablehlo.reshape %v92 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v94 = stablehlo.broadcast_in_dim %b2enmu, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v95 = stablehlo.subtract %v93, %v94 : tensor<32x96x112x112xf32>
    %v96 = stablehlo.broadcast_in_dim %b2envar, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v97 = stablehlo.constant dense<1.0e-5> : tensor<32x96x112x112xf32>
    %v98 = stablehlo.add %v96, %v97 : tensor<32x96x112x112xf32>
    %v99 = stablehlo.rsqrt %v98 : tensor<32x96x112x112xf32>
    %v100 = stablehlo.multiply %v95, %v99 : tensor<32x96x112x112xf32>
    %v101 = stablehlo.broadcast_in_dim %b2eg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v102 = stablehlo.broadcast_in_dim %b2ebt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x112x112xf32>
    %v103 = stablehlo.multiply %v100, %v101 : tensor<32x96x112x112xf32>
    %v104 = stablehlo.add %v103, %v102 : tensor<32x96x112x112xf32>
    %v105 = stablehlo.reshape %v104 : (tensor<32x96x112x112xf32>) -> tensor<32x1204224xf32>
    %v106 = stablehlo.logistic %v105 : tensor<32x1204224xf32>
    %v107 = stablehlo.multiply %v105, %v106 : tensor<32x1204224xf32>
    %v108 = stablehlo.reshape %v107 : (tensor<32x1204224xf32>) -> tensor<32x96x112x112xf32>
    %v109 = stablehlo.convolution(%v108, %b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x112x112xf32>, tensor<96x1x3x3xf32>) -> tensor<32x96x56x56xf32>
    %v110 = stablehlo.broadcast_in_dim %zb96, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v111 = stablehlo.add %v109, %v110 : tensor<32x96x56x56xf32>
    %v112 = stablehlo.reshape %v111 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v113 = stablehlo.reshape %v112 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v114 = stablehlo.broadcast_in_dim %b2dnmu, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v115 = stablehlo.subtract %v113, %v114 : tensor<32x96x56x56xf32>
    %v116 = stablehlo.broadcast_in_dim %b2dnvar, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v117 = stablehlo.constant dense<1.0e-5> : tensor<32x96x56x56xf32>
    %v118 = stablehlo.add %v116, %v117 : tensor<32x96x56x56xf32>
    %v119 = stablehlo.rsqrt %v118 : tensor<32x96x56x56xf32>
    %v120 = stablehlo.multiply %v115, %v119 : tensor<32x96x56x56xf32>
    %v121 = stablehlo.broadcast_in_dim %b2dg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v122 = stablehlo.broadcast_in_dim %b2dbt, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v123 = stablehlo.multiply %v120, %v121 : tensor<32x96x56x56xf32>
    %v124 = stablehlo.add %v123, %v122 : tensor<32x96x56x56xf32>
    %v125 = stablehlo.reshape %v124 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v126 = stablehlo.logistic %v125 : tensor<32x301056xf32>
    %v127 = stablehlo.multiply %v125, %v126 : tensor<32x301056xf32>
    %v128 = stablehlo.reshape %v127 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v129 = stablehlo.constant dense<0.0> : tensor<f32>
    %v130 = stablehlo.reduce(%v128 init: %v129) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v131 = stablehlo.constant dense<3136.0> : tensor<32x96xf32>
    %v132 = stablehlo.divide %v130, %v131 : tensor<32x96xf32>
    %v133 = stablehlo.dot_general %v132, %b2zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<96x4xf32>) -> tensor<32x4xf32>
    %v134 = stablehlo.broadcast_in_dim %b2zb1, dims = [1] : (tensor<4xf32>) -> tensor<32x4xf32>
    %v135 = stablehlo.add %v133, %v134 : tensor<32x4xf32>
    %v136 = stablehlo.logistic %v135 : tensor<32x4xf32>
    %v137 = stablehlo.multiply %v135, %v136 : tensor<32x4xf32>
    %v138 = stablehlo.dot_general %v137, %b2zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<4x96xf32>) -> tensor<32x96xf32>
    %v139 = stablehlo.broadcast_in_dim %b2zb2, dims = [1] : (tensor<96xf32>) -> tensor<32x96xf32>
    %v140 = stablehlo.add %v138, %v139 : tensor<32x96xf32>
    %v141 = stablehlo.reshape %v127 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v142 = stablehlo.constant dense<0.0> : tensor<f32>
    %v143 = stablehlo.reduce(%v141 init: %v142) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96xf32>
    %v144 = stablehlo.constant dense<3136.0> : tensor<32x96xf32>
    %v145 = stablehlo.divide %v143, %v144 : tensor<32x96xf32>
    %v146 = stablehlo.dot_general %v145, %b2zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x96xf32>, tensor<96x4xf32>) -> tensor<32x4xf32>
    %v147 = stablehlo.broadcast_in_dim %b2zb1, dims = [1] : (tensor<4xf32>) -> tensor<32x4xf32>
    %v148 = stablehlo.add %v146, %v147 : tensor<32x4xf32>
    %v149 = stablehlo.logistic %v148 : tensor<32x4xf32>
    %v150 = stablehlo.multiply %v148, %v149 : tensor<32x4xf32>
    %v151 = stablehlo.dot_general %v150, %b2zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf32>, tensor<4x96xf32>) -> tensor<32x96xf32>
    %v152 = stablehlo.broadcast_in_dim %b2zb2, dims = [1] : (tensor<96xf32>) -> tensor<32x96xf32>
    %v153 = stablehlo.add %v151, %v152 : tensor<32x96xf32>
    %v154 = stablehlo.logistic %v153 : tensor<32x96xf32>
    %v155 = stablehlo.broadcast_in_dim %v154, dims = [0, 1] : (tensor<32x96xf32>) -> tensor<32x96x56x56xf32>
    %v156 = stablehlo.multiply %v141, %v155 : tensor<32x96x56x56xf32>
    %v157 = stablehlo.reshape %v156 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v158 = stablehlo.reshape %v157 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v159 = stablehlo.convolution(%v158, %b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<24x96x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v160 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v161 = stablehlo.add %v159, %v160 : tensor<32x24x56x56xf32>
    %v162 = stablehlo.reshape %v161 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v163 = stablehlo.reshape %v162 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v164 = stablehlo.broadcast_in_dim %b2pnmu, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v165 = stablehlo.subtract %v163, %v164 : tensor<32x24x56x56xf32>
    %v166 = stablehlo.broadcast_in_dim %b2pnvar, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v167 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v168 = stablehlo.add %v166, %v167 : tensor<32x24x56x56xf32>
    %v169 = stablehlo.rsqrt %v168 : tensor<32x24x56x56xf32>
    %v170 = stablehlo.multiply %v165, %v169 : tensor<32x24x56x56xf32>
    %v171 = stablehlo.broadcast_in_dim %b2pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v172 = stablehlo.broadcast_in_dim %b2pbt, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v173 = stablehlo.multiply %v170, %v171 : tensor<32x24x56x56xf32>
    %v174 = stablehlo.add %v173, %v172 : tensor<32x24x56x56xf32>
    %v175 = stablehlo.reshape %v174 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v176 = stablehlo.reshape %v175 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v177 = stablehlo.convolution(%v176, %b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v178 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v179 = stablehlo.add %v177, %v178 : tensor<32x144x56x56xf32>
    %v180 = stablehlo.reshape %v179 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v181 = stablehlo.reshape %v180 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v182 = stablehlo.broadcast_in_dim %b3enmu, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v183 = stablehlo.subtract %v181, %v182 : tensor<32x144x56x56xf32>
    %v184 = stablehlo.broadcast_in_dim %b3envar, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v185 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v186 = stablehlo.add %v184, %v185 : tensor<32x144x56x56xf32>
    %v187 = stablehlo.rsqrt %v186 : tensor<32x144x56x56xf32>
    %v188 = stablehlo.multiply %v183, %v187 : tensor<32x144x56x56xf32>
    %v189 = stablehlo.broadcast_in_dim %b3eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v190 = stablehlo.broadcast_in_dim %b3ebt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v191 = stablehlo.multiply %v188, %v189 : tensor<32x144x56x56xf32>
    %v192 = stablehlo.add %v191, %v190 : tensor<32x144x56x56xf32>
    %v193 = stablehlo.reshape %v192 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v194 = stablehlo.logistic %v193 : tensor<32x451584xf32>
    %v195 = stablehlo.multiply %v193, %v194 : tensor<32x451584xf32>
    %v196 = stablehlo.reshape %v195 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v197 = stablehlo.convolution(%v196, %b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x3x3xf32>) -> tensor<32x144x56x56xf32>
    %v198 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v199 = stablehlo.add %v197, %v198 : tensor<32x144x56x56xf32>
    %v200 = stablehlo.reshape %v199 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v201 = stablehlo.reshape %v200 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v202 = stablehlo.broadcast_in_dim %b3dnmu, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v203 = stablehlo.subtract %v201, %v202 : tensor<32x144x56x56xf32>
    %v204 = stablehlo.broadcast_in_dim %b3dnvar, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v205 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v206 = stablehlo.add %v204, %v205 : tensor<32x144x56x56xf32>
    %v207 = stablehlo.rsqrt %v206 : tensor<32x144x56x56xf32>
    %v208 = stablehlo.multiply %v203, %v207 : tensor<32x144x56x56xf32>
    %v209 = stablehlo.broadcast_in_dim %b3dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v210 = stablehlo.broadcast_in_dim %b3dbt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v211 = stablehlo.multiply %v208, %v209 : tensor<32x144x56x56xf32>
    %v212 = stablehlo.add %v211, %v210 : tensor<32x144x56x56xf32>
    %v213 = stablehlo.reshape %v212 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v214 = stablehlo.logistic %v213 : tensor<32x451584xf32>
    %v215 = stablehlo.multiply %v213, %v214 : tensor<32x451584xf32>
    %v216 = stablehlo.reshape %v215 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v217 = stablehlo.constant dense<0.0> : tensor<f32>
    %v218 = stablehlo.reduce(%v216 init: %v217) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v219 = stablehlo.constant dense<3136.0> : tensor<32x144xf32>
    %v220 = stablehlo.divide %v218, %v219 : tensor<32x144xf32>
    %v221 = stablehlo.dot_general %v220, %b3zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v222 = stablehlo.broadcast_in_dim %b3zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v223 = stablehlo.add %v221, %v222 : tensor<32x6xf32>
    %v224 = stablehlo.logistic %v223 : tensor<32x6xf32>
    %v225 = stablehlo.multiply %v223, %v224 : tensor<32x6xf32>
    %v226 = stablehlo.dot_general %v225, %b3zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v227 = stablehlo.broadcast_in_dim %b3zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v228 = stablehlo.add %v226, %v227 : tensor<32x144xf32>
    %v229 = stablehlo.reshape %v215 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v230 = stablehlo.constant dense<0.0> : tensor<f32>
    %v231 = stablehlo.reduce(%v229 init: %v230) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v232 = stablehlo.constant dense<3136.0> : tensor<32x144xf32>
    %v233 = stablehlo.divide %v231, %v232 : tensor<32x144xf32>
    %v234 = stablehlo.dot_general %v233, %b3zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v235 = stablehlo.broadcast_in_dim %b3zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v236 = stablehlo.add %v234, %v235 : tensor<32x6xf32>
    %v237 = stablehlo.logistic %v236 : tensor<32x6xf32>
    %v238 = stablehlo.multiply %v236, %v237 : tensor<32x6xf32>
    %v239 = stablehlo.dot_general %v238, %b3zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v240 = stablehlo.broadcast_in_dim %b3zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v241 = stablehlo.add %v239, %v240 : tensor<32x144xf32>
    %v242 = stablehlo.logistic %v241 : tensor<32x144xf32>
    %v243 = stablehlo.broadcast_in_dim %v242, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x56x56xf32>
    %v244 = stablehlo.multiply %v229, %v243 : tensor<32x144x56x56xf32>
    %v245 = stablehlo.reshape %v244 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v246 = stablehlo.reshape %v245 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v247 = stablehlo.convolution(%v246, %b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x56x56xf32>, tensor<24x144x1x1xf32>) -> tensor<32x24x56x56xf32>
    %v248 = stablehlo.broadcast_in_dim %zb24, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v249 = stablehlo.add %v247, %v248 : tensor<32x24x56x56xf32>
    %v250 = stablehlo.reshape %v249 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v251 = stablehlo.reshape %v250 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v252 = stablehlo.broadcast_in_dim %b3pnmu, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v253 = stablehlo.subtract %v251, %v252 : tensor<32x24x56x56xf32>
    %v254 = stablehlo.broadcast_in_dim %b3pnvar, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v255 = stablehlo.constant dense<1.0e-5> : tensor<32x24x56x56xf32>
    %v256 = stablehlo.add %v254, %v255 : tensor<32x24x56x56xf32>
    %v257 = stablehlo.rsqrt %v256 : tensor<32x24x56x56xf32>
    %v258 = stablehlo.multiply %v253, %v257 : tensor<32x24x56x56xf32>
    %v259 = stablehlo.broadcast_in_dim %b3pg, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v260 = stablehlo.broadcast_in_dim %b3pbt, dims = [1] : (tensor<24xf32>) -> tensor<32x24x56x56xf32>
    %v261 = stablehlo.multiply %v258, %v259 : tensor<32x24x56x56xf32>
    %v262 = stablehlo.add %v261, %v260 : tensor<32x24x56x56xf32>
    %v263 = stablehlo.reshape %v262 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v264 = stablehlo.broadcast_in_dim %dp2, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v265 = stablehlo.multiply %v264, %v263 : tensor<32x75264xf32>
    %v266 = stablehlo.add %v265, %v175 : tensor<32x75264xf32>
    %v267 = stablehlo.reshape %v266 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v268 = stablehlo.convolution(%v267, %b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v269 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v270 = stablehlo.add %v268, %v269 : tensor<32x144x56x56xf32>
    %v271 = stablehlo.reshape %v270 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v272 = stablehlo.reshape %v271 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v273 = stablehlo.broadcast_in_dim %b4enmu, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v274 = stablehlo.subtract %v272, %v273 : tensor<32x144x56x56xf32>
    %v275 = stablehlo.broadcast_in_dim %b4envar, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v276 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v277 = stablehlo.add %v275, %v276 : tensor<32x144x56x56xf32>
    %v278 = stablehlo.rsqrt %v277 : tensor<32x144x56x56xf32>
    %v279 = stablehlo.multiply %v274, %v278 : tensor<32x144x56x56xf32>
    %v280 = stablehlo.broadcast_in_dim %b4eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v281 = stablehlo.broadcast_in_dim %b4ebt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v282 = stablehlo.multiply %v279, %v280 : tensor<32x144x56x56xf32>
    %v283 = stablehlo.add %v282, %v281 : tensor<32x144x56x56xf32>
    %v284 = stablehlo.reshape %v283 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v285 = stablehlo.logistic %v284 : tensor<32x451584xf32>
    %v286 = stablehlo.multiply %v284, %v285 : tensor<32x451584xf32>
    %v287 = stablehlo.reshape %v286 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v288 = stablehlo.convolution(%v287, %b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x5x5xf32>) -> tensor<32x144x28x28xf32>
    %v289 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v290 = stablehlo.add %v288, %v289 : tensor<32x144x28x28xf32>
    %v291 = stablehlo.reshape %v290 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v292 = stablehlo.reshape %v291 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v293 = stablehlo.broadcast_in_dim %b4dnmu, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v294 = stablehlo.subtract %v292, %v293 : tensor<32x144x28x28xf32>
    %v295 = stablehlo.broadcast_in_dim %b4dnvar, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v296 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v297 = stablehlo.add %v295, %v296 : tensor<32x144x28x28xf32>
    %v298 = stablehlo.rsqrt %v297 : tensor<32x144x28x28xf32>
    %v299 = stablehlo.multiply %v294, %v298 : tensor<32x144x28x28xf32>
    %v300 = stablehlo.broadcast_in_dim %b4dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v301 = stablehlo.broadcast_in_dim %b4dbt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v302 = stablehlo.multiply %v299, %v300 : tensor<32x144x28x28xf32>
    %v303 = stablehlo.add %v302, %v301 : tensor<32x144x28x28xf32>
    %v304 = stablehlo.reshape %v303 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v305 = stablehlo.logistic %v304 : tensor<32x112896xf32>
    %v306 = stablehlo.multiply %v304, %v305 : tensor<32x112896xf32>
    %v307 = stablehlo.reshape %v306 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v308 = stablehlo.constant dense<0.0> : tensor<f32>
    %v309 = stablehlo.reduce(%v307 init: %v308) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v310 = stablehlo.constant dense<784.0> : tensor<32x144xf32>
    %v311 = stablehlo.divide %v309, %v310 : tensor<32x144xf32>
    %v312 = stablehlo.dot_general %v311, %b4zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v313 = stablehlo.broadcast_in_dim %b4zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v314 = stablehlo.add %v312, %v313 : tensor<32x6xf32>
    %v315 = stablehlo.logistic %v314 : tensor<32x6xf32>
    %v316 = stablehlo.multiply %v314, %v315 : tensor<32x6xf32>
    %v317 = stablehlo.dot_general %v316, %b4zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v318 = stablehlo.broadcast_in_dim %b4zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v319 = stablehlo.add %v317, %v318 : tensor<32x144xf32>
    %v320 = stablehlo.reshape %v306 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v321 = stablehlo.constant dense<0.0> : tensor<f32>
    %v322 = stablehlo.reduce(%v320 init: %v321) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v323 = stablehlo.constant dense<784.0> : tensor<32x144xf32>
    %v324 = stablehlo.divide %v322, %v323 : tensor<32x144xf32>
    %v325 = stablehlo.dot_general %v324, %b4zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v326 = stablehlo.broadcast_in_dim %b4zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v327 = stablehlo.add %v325, %v326 : tensor<32x6xf32>
    %v328 = stablehlo.logistic %v327 : tensor<32x6xf32>
    %v329 = stablehlo.multiply %v327, %v328 : tensor<32x6xf32>
    %v330 = stablehlo.dot_general %v329, %b4zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v331 = stablehlo.broadcast_in_dim %b4zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v332 = stablehlo.add %v330, %v331 : tensor<32x144xf32>
    %v333 = stablehlo.logistic %v332 : tensor<32x144xf32>
    %v334 = stablehlo.broadcast_in_dim %v333, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v335 = stablehlo.multiply %v320, %v334 : tensor<32x144x28x28xf32>
    %v336 = stablehlo.reshape %v335 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v337 = stablehlo.reshape %v336 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v338 = stablehlo.convolution(%v337, %b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x28x28xf32>, tensor<40x144x1x1xf32>) -> tensor<32x40x28x28xf32>
    %v339 = stablehlo.broadcast_in_dim %zb40, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v340 = stablehlo.add %v338, %v339 : tensor<32x40x28x28xf32>
    %v341 = stablehlo.reshape %v340 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v342 = stablehlo.reshape %v341 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v343 = stablehlo.broadcast_in_dim %b4pnmu, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v344 = stablehlo.subtract %v342, %v343 : tensor<32x40x28x28xf32>
    %v345 = stablehlo.broadcast_in_dim %b4pnvar, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v346 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v347 = stablehlo.add %v345, %v346 : tensor<32x40x28x28xf32>
    %v348 = stablehlo.rsqrt %v347 : tensor<32x40x28x28xf32>
    %v349 = stablehlo.multiply %v344, %v348 : tensor<32x40x28x28xf32>
    %v350 = stablehlo.broadcast_in_dim %b4pg, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v351 = stablehlo.broadcast_in_dim %b4pbt, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v352 = stablehlo.multiply %v349, %v350 : tensor<32x40x28x28xf32>
    %v353 = stablehlo.add %v352, %v351 : tensor<32x40x28x28xf32>
    %v354 = stablehlo.reshape %v353 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v355 = stablehlo.reshape %v354 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v356 = stablehlo.convolution(%v355, %b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<32x240x28x28xf32>
    %v357 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v358 = stablehlo.add %v356, %v357 : tensor<32x240x28x28xf32>
    %v359 = stablehlo.reshape %v358 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v360 = stablehlo.reshape %v359 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v361 = stablehlo.broadcast_in_dim %b5enmu, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v362 = stablehlo.subtract %v360, %v361 : tensor<32x240x28x28xf32>
    %v363 = stablehlo.broadcast_in_dim %b5envar, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v364 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v365 = stablehlo.add %v363, %v364 : tensor<32x240x28x28xf32>
    %v366 = stablehlo.rsqrt %v365 : tensor<32x240x28x28xf32>
    %v367 = stablehlo.multiply %v362, %v366 : tensor<32x240x28x28xf32>
    %v368 = stablehlo.broadcast_in_dim %b5eg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v369 = stablehlo.broadcast_in_dim %b5ebt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v370 = stablehlo.multiply %v367, %v368 : tensor<32x240x28x28xf32>
    %v371 = stablehlo.add %v370, %v369 : tensor<32x240x28x28xf32>
    %v372 = stablehlo.reshape %v371 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v373 = stablehlo.logistic %v372 : tensor<32x188160xf32>
    %v374 = stablehlo.multiply %v372, %v373 : tensor<32x188160xf32>
    %v375 = stablehlo.reshape %v374 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v376 = stablehlo.convolution(%v375, %b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x5x5xf32>) -> tensor<32x240x28x28xf32>
    %v377 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v378 = stablehlo.add %v376, %v377 : tensor<32x240x28x28xf32>
    %v379 = stablehlo.reshape %v378 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v380 = stablehlo.reshape %v379 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v381 = stablehlo.broadcast_in_dim %b5dnmu, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v382 = stablehlo.subtract %v380, %v381 : tensor<32x240x28x28xf32>
    %v383 = stablehlo.broadcast_in_dim %b5dnvar, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v384 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v385 = stablehlo.add %v383, %v384 : tensor<32x240x28x28xf32>
    %v386 = stablehlo.rsqrt %v385 : tensor<32x240x28x28xf32>
    %v387 = stablehlo.multiply %v382, %v386 : tensor<32x240x28x28xf32>
    %v388 = stablehlo.broadcast_in_dim %b5dg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v389 = stablehlo.broadcast_in_dim %b5dbt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v390 = stablehlo.multiply %v387, %v388 : tensor<32x240x28x28xf32>
    %v391 = stablehlo.add %v390, %v389 : tensor<32x240x28x28xf32>
    %v392 = stablehlo.reshape %v391 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v393 = stablehlo.logistic %v392 : tensor<32x188160xf32>
    %v394 = stablehlo.multiply %v392, %v393 : tensor<32x188160xf32>
    %v395 = stablehlo.reshape %v394 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v396 = stablehlo.constant dense<0.0> : tensor<f32>
    %v397 = stablehlo.reduce(%v395 init: %v396) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v398 = stablehlo.constant dense<784.0> : tensor<32x240xf32>
    %v399 = stablehlo.divide %v397, %v398 : tensor<32x240xf32>
    %v400 = stablehlo.dot_general %v399, %b5zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v401 = stablehlo.broadcast_in_dim %b5zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v402 = stablehlo.add %v400, %v401 : tensor<32x10xf32>
    %v403 = stablehlo.logistic %v402 : tensor<32x10xf32>
    %v404 = stablehlo.multiply %v402, %v403 : tensor<32x10xf32>
    %v405 = stablehlo.dot_general %v404, %b5zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v406 = stablehlo.broadcast_in_dim %b5zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v407 = stablehlo.add %v405, %v406 : tensor<32x240xf32>
    %v408 = stablehlo.reshape %v394 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v409 = stablehlo.constant dense<0.0> : tensor<f32>
    %v410 = stablehlo.reduce(%v408 init: %v409) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v411 = stablehlo.constant dense<784.0> : tensor<32x240xf32>
    %v412 = stablehlo.divide %v410, %v411 : tensor<32x240xf32>
    %v413 = stablehlo.dot_general %v412, %b5zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v414 = stablehlo.broadcast_in_dim %b5zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v415 = stablehlo.add %v413, %v414 : tensor<32x10xf32>
    %v416 = stablehlo.logistic %v415 : tensor<32x10xf32>
    %v417 = stablehlo.multiply %v415, %v416 : tensor<32x10xf32>
    %v418 = stablehlo.dot_general %v417, %b5zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v419 = stablehlo.broadcast_in_dim %b5zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v420 = stablehlo.add %v418, %v419 : tensor<32x240xf32>
    %v421 = stablehlo.logistic %v420 : tensor<32x240xf32>
    %v422 = stablehlo.broadcast_in_dim %v421, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x28x28xf32>
    %v423 = stablehlo.multiply %v408, %v422 : tensor<32x240x28x28xf32>
    %v424 = stablehlo.reshape %v423 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v425 = stablehlo.reshape %v424 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v426 = stablehlo.convolution(%v425, %b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x28x28xf32>, tensor<40x240x1x1xf32>) -> tensor<32x40x28x28xf32>
    %v427 = stablehlo.broadcast_in_dim %zb40, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v428 = stablehlo.add %v426, %v427 : tensor<32x40x28x28xf32>
    %v429 = stablehlo.reshape %v428 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v430 = stablehlo.reshape %v429 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v431 = stablehlo.broadcast_in_dim %b5pnmu, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v432 = stablehlo.subtract %v430, %v431 : tensor<32x40x28x28xf32>
    %v433 = stablehlo.broadcast_in_dim %b5pnvar, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v434 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v435 = stablehlo.add %v433, %v434 : tensor<32x40x28x28xf32>
    %v436 = stablehlo.rsqrt %v435 : tensor<32x40x28x28xf32>
    %v437 = stablehlo.multiply %v432, %v436 : tensor<32x40x28x28xf32>
    %v438 = stablehlo.broadcast_in_dim %b5pg, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v439 = stablehlo.broadcast_in_dim %b5pbt, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v440 = stablehlo.multiply %v437, %v438 : tensor<32x40x28x28xf32>
    %v441 = stablehlo.add %v440, %v439 : tensor<32x40x28x28xf32>
    %v442 = stablehlo.reshape %v441 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v443 = stablehlo.broadcast_in_dim %dp4, dims = [0] : (tensor<32xf32>) -> tensor<32x31360xf32>
    %v444 = stablehlo.multiply %v443, %v442 : tensor<32x31360xf32>
    %v445 = stablehlo.add %v444, %v354 : tensor<32x31360xf32>
    %v446 = stablehlo.reshape %v445 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v447 = stablehlo.convolution(%v446, %b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<32x240x28x28xf32>
    %v448 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v449 = stablehlo.add %v447, %v448 : tensor<32x240x28x28xf32>
    %v450 = stablehlo.reshape %v449 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v451 = stablehlo.reshape %v450 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v452 = stablehlo.broadcast_in_dim %b6enmu, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v453 = stablehlo.subtract %v451, %v452 : tensor<32x240x28x28xf32>
    %v454 = stablehlo.broadcast_in_dim %b6envar, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v455 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v456 = stablehlo.add %v454, %v455 : tensor<32x240x28x28xf32>
    %v457 = stablehlo.rsqrt %v456 : tensor<32x240x28x28xf32>
    %v458 = stablehlo.multiply %v453, %v457 : tensor<32x240x28x28xf32>
    %v459 = stablehlo.broadcast_in_dim %b6eg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v460 = stablehlo.broadcast_in_dim %b6ebt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v461 = stablehlo.multiply %v458, %v459 : tensor<32x240x28x28xf32>
    %v462 = stablehlo.add %v461, %v460 : tensor<32x240x28x28xf32>
    %v463 = stablehlo.reshape %v462 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v464 = stablehlo.logistic %v463 : tensor<32x188160xf32>
    %v465 = stablehlo.multiply %v463, %v464 : tensor<32x188160xf32>
    %v466 = stablehlo.reshape %v465 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v467 = stablehlo.convolution(%v466, %b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x3x3xf32>) -> tensor<32x240x14x14xf32>
    %v468 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v469 = stablehlo.add %v467, %v468 : tensor<32x240x14x14xf32>
    %v470 = stablehlo.reshape %v469 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v471 = stablehlo.reshape %v470 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v472 = stablehlo.broadcast_in_dim %b6dnmu, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v473 = stablehlo.subtract %v471, %v472 : tensor<32x240x14x14xf32>
    %v474 = stablehlo.broadcast_in_dim %b6dnvar, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v475 = stablehlo.constant dense<1.0e-5> : tensor<32x240x14x14xf32>
    %v476 = stablehlo.add %v474, %v475 : tensor<32x240x14x14xf32>
    %v477 = stablehlo.rsqrt %v476 : tensor<32x240x14x14xf32>
    %v478 = stablehlo.multiply %v473, %v477 : tensor<32x240x14x14xf32>
    %v479 = stablehlo.broadcast_in_dim %b6dg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v480 = stablehlo.broadcast_in_dim %b6dbt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v481 = stablehlo.multiply %v478, %v479 : tensor<32x240x14x14xf32>
    %v482 = stablehlo.add %v481, %v480 : tensor<32x240x14x14xf32>
    %v483 = stablehlo.reshape %v482 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v484 = stablehlo.logistic %v483 : tensor<32x47040xf32>
    %v485 = stablehlo.multiply %v483, %v484 : tensor<32x47040xf32>
    %v486 = stablehlo.reshape %v485 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v487 = stablehlo.constant dense<0.0> : tensor<f32>
    %v488 = stablehlo.reduce(%v486 init: %v487) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v489 = stablehlo.constant dense<196.0> : tensor<32x240xf32>
    %v490 = stablehlo.divide %v488, %v489 : tensor<32x240xf32>
    %v491 = stablehlo.dot_general %v490, %b6zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v492 = stablehlo.broadcast_in_dim %b6zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v493 = stablehlo.add %v491, %v492 : tensor<32x10xf32>
    %v494 = stablehlo.logistic %v493 : tensor<32x10xf32>
    %v495 = stablehlo.multiply %v493, %v494 : tensor<32x10xf32>
    %v496 = stablehlo.dot_general %v495, %b6zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v497 = stablehlo.broadcast_in_dim %b6zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v498 = stablehlo.add %v496, %v497 : tensor<32x240xf32>
    %v499 = stablehlo.reshape %v485 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v500 = stablehlo.constant dense<0.0> : tensor<f32>
    %v501 = stablehlo.reduce(%v499 init: %v500) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v502 = stablehlo.constant dense<196.0> : tensor<32x240xf32>
    %v503 = stablehlo.divide %v501, %v502 : tensor<32x240xf32>
    %v504 = stablehlo.dot_general %v503, %b6zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v505 = stablehlo.broadcast_in_dim %b6zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v506 = stablehlo.add %v504, %v505 : tensor<32x10xf32>
    %v507 = stablehlo.logistic %v506 : tensor<32x10xf32>
    %v508 = stablehlo.multiply %v506, %v507 : tensor<32x10xf32>
    %v509 = stablehlo.dot_general %v508, %b6zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v510 = stablehlo.broadcast_in_dim %b6zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v511 = stablehlo.add %v509, %v510 : tensor<32x240xf32>
    %v512 = stablehlo.logistic %v511 : tensor<32x240xf32>
    %v513 = stablehlo.broadcast_in_dim %v512, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x14x14xf32>
    %v514 = stablehlo.multiply %v499, %v513 : tensor<32x240x14x14xf32>
    %v515 = stablehlo.reshape %v514 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v516 = stablehlo.reshape %v515 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v517 = stablehlo.convolution(%v516, %b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x14x14xf32>, tensor<80x240x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v518 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v519 = stablehlo.add %v517, %v518 : tensor<32x80x14x14xf32>
    %v520 = stablehlo.reshape %v519 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v521 = stablehlo.reshape %v520 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v522 = stablehlo.broadcast_in_dim %b6pnmu, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v523 = stablehlo.subtract %v521, %v522 : tensor<32x80x14x14xf32>
    %v524 = stablehlo.broadcast_in_dim %b6pnvar, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v525 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v526 = stablehlo.add %v524, %v525 : tensor<32x80x14x14xf32>
    %v527 = stablehlo.rsqrt %v526 : tensor<32x80x14x14xf32>
    %v528 = stablehlo.multiply %v523, %v527 : tensor<32x80x14x14xf32>
    %v529 = stablehlo.broadcast_in_dim %b6pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v530 = stablehlo.broadcast_in_dim %b6pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v531 = stablehlo.multiply %v528, %v529 : tensor<32x80x14x14xf32>
    %v532 = stablehlo.add %v531, %v530 : tensor<32x80x14x14xf32>
    %v533 = stablehlo.reshape %v532 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v534 = stablehlo.reshape %v533 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v535 = stablehlo.convolution(%v534, %b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v536 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v537 = stablehlo.add %v535, %v536 : tensor<32x480x14x14xf32>
    %v538 = stablehlo.reshape %v537 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v539 = stablehlo.reshape %v538 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v540 = stablehlo.broadcast_in_dim %b7enmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v541 = stablehlo.subtract %v539, %v540 : tensor<32x480x14x14xf32>
    %v542 = stablehlo.broadcast_in_dim %b7envar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v543 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v544 = stablehlo.add %v542, %v543 : tensor<32x480x14x14xf32>
    %v545 = stablehlo.rsqrt %v544 : tensor<32x480x14x14xf32>
    %v546 = stablehlo.multiply %v541, %v545 : tensor<32x480x14x14xf32>
    %v547 = stablehlo.broadcast_in_dim %b7eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v548 = stablehlo.broadcast_in_dim %b7ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v549 = stablehlo.multiply %v546, %v547 : tensor<32x480x14x14xf32>
    %v550 = stablehlo.add %v549, %v548 : tensor<32x480x14x14xf32>
    %v551 = stablehlo.reshape %v550 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v552 = stablehlo.logistic %v551 : tensor<32x94080xf32>
    %v553 = stablehlo.multiply %v551, %v552 : tensor<32x94080xf32>
    %v554 = stablehlo.reshape %v553 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v555 = stablehlo.convolution(%v554, %b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %v556 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v557 = stablehlo.add %v555, %v556 : tensor<32x480x14x14xf32>
    %v558 = stablehlo.reshape %v557 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v559 = stablehlo.reshape %v558 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v560 = stablehlo.broadcast_in_dim %b7dnmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v561 = stablehlo.subtract %v559, %v560 : tensor<32x480x14x14xf32>
    %v562 = stablehlo.broadcast_in_dim %b7dnvar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v563 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v564 = stablehlo.add %v562, %v563 : tensor<32x480x14x14xf32>
    %v565 = stablehlo.rsqrt %v564 : tensor<32x480x14x14xf32>
    %v566 = stablehlo.multiply %v561, %v565 : tensor<32x480x14x14xf32>
    %v567 = stablehlo.broadcast_in_dim %b7dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v568 = stablehlo.broadcast_in_dim %b7dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v569 = stablehlo.multiply %v566, %v567 : tensor<32x480x14x14xf32>
    %v570 = stablehlo.add %v569, %v568 : tensor<32x480x14x14xf32>
    %v571 = stablehlo.reshape %v570 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v572 = stablehlo.logistic %v571 : tensor<32x94080xf32>
    %v573 = stablehlo.multiply %v571, %v572 : tensor<32x94080xf32>
    %v574 = stablehlo.reshape %v573 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v575 = stablehlo.constant dense<0.0> : tensor<f32>
    %v576 = stablehlo.reduce(%v574 init: %v575) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v577 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v578 = stablehlo.divide %v576, %v577 : tensor<32x480xf32>
    %v579 = stablehlo.dot_general %v578, %b7zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v580 = stablehlo.broadcast_in_dim %b7zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v581 = stablehlo.add %v579, %v580 : tensor<32x20xf32>
    %v582 = stablehlo.logistic %v581 : tensor<32x20xf32>
    %v583 = stablehlo.multiply %v581, %v582 : tensor<32x20xf32>
    %v584 = stablehlo.dot_general %v583, %b7zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v585 = stablehlo.broadcast_in_dim %b7zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v586 = stablehlo.add %v584, %v585 : tensor<32x480xf32>
    %v587 = stablehlo.reshape %v573 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v588 = stablehlo.constant dense<0.0> : tensor<f32>
    %v589 = stablehlo.reduce(%v587 init: %v588) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v590 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v591 = stablehlo.divide %v589, %v590 : tensor<32x480xf32>
    %v592 = stablehlo.dot_general %v591, %b7zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v593 = stablehlo.broadcast_in_dim %b7zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v594 = stablehlo.add %v592, %v593 : tensor<32x20xf32>
    %v595 = stablehlo.logistic %v594 : tensor<32x20xf32>
    %v596 = stablehlo.multiply %v594, %v595 : tensor<32x20xf32>
    %v597 = stablehlo.dot_general %v596, %b7zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v598 = stablehlo.broadcast_in_dim %b7zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v599 = stablehlo.add %v597, %v598 : tensor<32x480xf32>
    %v600 = stablehlo.logistic %v599 : tensor<32x480xf32>
    %v601 = stablehlo.broadcast_in_dim %v600, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v602 = stablehlo.multiply %v587, %v601 : tensor<32x480x14x14xf32>
    %v603 = stablehlo.reshape %v602 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v604 = stablehlo.reshape %v603 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v605 = stablehlo.convolution(%v604, %b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v606 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v607 = stablehlo.add %v605, %v606 : tensor<32x80x14x14xf32>
    %v608 = stablehlo.reshape %v607 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v609 = stablehlo.reshape %v608 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v610 = stablehlo.broadcast_in_dim %b7pnmu, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v611 = stablehlo.subtract %v609, %v610 : tensor<32x80x14x14xf32>
    %v612 = stablehlo.broadcast_in_dim %b7pnvar, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v613 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v614 = stablehlo.add %v612, %v613 : tensor<32x80x14x14xf32>
    %v615 = stablehlo.rsqrt %v614 : tensor<32x80x14x14xf32>
    %v616 = stablehlo.multiply %v611, %v615 : tensor<32x80x14x14xf32>
    %v617 = stablehlo.broadcast_in_dim %b7pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v618 = stablehlo.broadcast_in_dim %b7pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v619 = stablehlo.multiply %v616, %v617 : tensor<32x80x14x14xf32>
    %v620 = stablehlo.add %v619, %v618 : tensor<32x80x14x14xf32>
    %v621 = stablehlo.reshape %v620 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v622 = stablehlo.broadcast_in_dim %dp6, dims = [0] : (tensor<32xf32>) -> tensor<32x15680xf32>
    %v623 = stablehlo.multiply %v622, %v621 : tensor<32x15680xf32>
    %v624 = stablehlo.add %v623, %v533 : tensor<32x15680xf32>
    %v625 = stablehlo.reshape %v624 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v626 = stablehlo.convolution(%v625, %b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v627 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v628 = stablehlo.add %v626, %v627 : tensor<32x480x14x14xf32>
    %v629 = stablehlo.reshape %v628 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v630 = stablehlo.reshape %v629 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v631 = stablehlo.broadcast_in_dim %b8enmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v632 = stablehlo.subtract %v630, %v631 : tensor<32x480x14x14xf32>
    %v633 = stablehlo.broadcast_in_dim %b8envar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v634 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v635 = stablehlo.add %v633, %v634 : tensor<32x480x14x14xf32>
    %v636 = stablehlo.rsqrt %v635 : tensor<32x480x14x14xf32>
    %v637 = stablehlo.multiply %v632, %v636 : tensor<32x480x14x14xf32>
    %v638 = stablehlo.broadcast_in_dim %b8eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v639 = stablehlo.broadcast_in_dim %b8ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v640 = stablehlo.multiply %v637, %v638 : tensor<32x480x14x14xf32>
    %v641 = stablehlo.add %v640, %v639 : tensor<32x480x14x14xf32>
    %v642 = stablehlo.reshape %v641 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v643 = stablehlo.logistic %v642 : tensor<32x94080xf32>
    %v644 = stablehlo.multiply %v642, %v643 : tensor<32x94080xf32>
    %v645 = stablehlo.reshape %v644 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v646 = stablehlo.convolution(%v645, %b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %v647 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v648 = stablehlo.add %v646, %v647 : tensor<32x480x14x14xf32>
    %v649 = stablehlo.reshape %v648 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v650 = stablehlo.reshape %v649 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v651 = stablehlo.broadcast_in_dim %b8dnmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v652 = stablehlo.subtract %v650, %v651 : tensor<32x480x14x14xf32>
    %v653 = stablehlo.broadcast_in_dim %b8dnvar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v654 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v655 = stablehlo.add %v653, %v654 : tensor<32x480x14x14xf32>
    %v656 = stablehlo.rsqrt %v655 : tensor<32x480x14x14xf32>
    %v657 = stablehlo.multiply %v652, %v656 : tensor<32x480x14x14xf32>
    %v658 = stablehlo.broadcast_in_dim %b8dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v659 = stablehlo.broadcast_in_dim %b8dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v660 = stablehlo.multiply %v657, %v658 : tensor<32x480x14x14xf32>
    %v661 = stablehlo.add %v660, %v659 : tensor<32x480x14x14xf32>
    %v662 = stablehlo.reshape %v661 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v663 = stablehlo.logistic %v662 : tensor<32x94080xf32>
    %v664 = stablehlo.multiply %v662, %v663 : tensor<32x94080xf32>
    %v665 = stablehlo.reshape %v664 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v666 = stablehlo.constant dense<0.0> : tensor<f32>
    %v667 = stablehlo.reduce(%v665 init: %v666) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v668 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v669 = stablehlo.divide %v667, %v668 : tensor<32x480xf32>
    %v670 = stablehlo.dot_general %v669, %b8zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v671 = stablehlo.broadcast_in_dim %b8zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v672 = stablehlo.add %v670, %v671 : tensor<32x20xf32>
    %v673 = stablehlo.logistic %v672 : tensor<32x20xf32>
    %v674 = stablehlo.multiply %v672, %v673 : tensor<32x20xf32>
    %v675 = stablehlo.dot_general %v674, %b8zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v676 = stablehlo.broadcast_in_dim %b8zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v677 = stablehlo.add %v675, %v676 : tensor<32x480xf32>
    %v678 = stablehlo.reshape %v664 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v679 = stablehlo.constant dense<0.0> : tensor<f32>
    %v680 = stablehlo.reduce(%v678 init: %v679) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v681 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v682 = stablehlo.divide %v680, %v681 : tensor<32x480xf32>
    %v683 = stablehlo.dot_general %v682, %b8zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v684 = stablehlo.broadcast_in_dim %b8zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v685 = stablehlo.add %v683, %v684 : tensor<32x20xf32>
    %v686 = stablehlo.logistic %v685 : tensor<32x20xf32>
    %v687 = stablehlo.multiply %v685, %v686 : tensor<32x20xf32>
    %v688 = stablehlo.dot_general %v687, %b8zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v689 = stablehlo.broadcast_in_dim %b8zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v690 = stablehlo.add %v688, %v689 : tensor<32x480xf32>
    %v691 = stablehlo.logistic %v690 : tensor<32x480xf32>
    %v692 = stablehlo.broadcast_in_dim %v691, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v693 = stablehlo.multiply %v678, %v692 : tensor<32x480x14x14xf32>
    %v694 = stablehlo.reshape %v693 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v695 = stablehlo.reshape %v694 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v696 = stablehlo.convolution(%v695, %b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v697 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v698 = stablehlo.add %v696, %v697 : tensor<32x80x14x14xf32>
    %v699 = stablehlo.reshape %v698 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v700 = stablehlo.reshape %v699 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v701 = stablehlo.broadcast_in_dim %b8pnmu, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v702 = stablehlo.subtract %v700, %v701 : tensor<32x80x14x14xf32>
    %v703 = stablehlo.broadcast_in_dim %b8pnvar, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v704 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v705 = stablehlo.add %v703, %v704 : tensor<32x80x14x14xf32>
    %v706 = stablehlo.rsqrt %v705 : tensor<32x80x14x14xf32>
    %v707 = stablehlo.multiply %v702, %v706 : tensor<32x80x14x14xf32>
    %v708 = stablehlo.broadcast_in_dim %b8pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v709 = stablehlo.broadcast_in_dim %b8pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v710 = stablehlo.multiply %v707, %v708 : tensor<32x80x14x14xf32>
    %v711 = stablehlo.add %v710, %v709 : tensor<32x80x14x14xf32>
    %v712 = stablehlo.reshape %v711 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v713 = stablehlo.broadcast_in_dim %dp7, dims = [0] : (tensor<32xf32>) -> tensor<32x15680xf32>
    %v714 = stablehlo.multiply %v713, %v712 : tensor<32x15680xf32>
    %v715 = stablehlo.add %v714, %v624 : tensor<32x15680xf32>
    %v716 = stablehlo.reshape %v715 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v717 = stablehlo.convolution(%v716, %b9eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v718 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v719 = stablehlo.add %v717, %v718 : tensor<32x480x14x14xf32>
    %v720 = stablehlo.reshape %v719 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v721 = stablehlo.reshape %v720 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v722 = stablehlo.broadcast_in_dim %b9enmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v723 = stablehlo.subtract %v721, %v722 : tensor<32x480x14x14xf32>
    %v724 = stablehlo.broadcast_in_dim %b9envar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v725 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v726 = stablehlo.add %v724, %v725 : tensor<32x480x14x14xf32>
    %v727 = stablehlo.rsqrt %v726 : tensor<32x480x14x14xf32>
    %v728 = stablehlo.multiply %v723, %v727 : tensor<32x480x14x14xf32>
    %v729 = stablehlo.broadcast_in_dim %b9eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v730 = stablehlo.broadcast_in_dim %b9ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v731 = stablehlo.multiply %v728, %v729 : tensor<32x480x14x14xf32>
    %v732 = stablehlo.add %v731, %v730 : tensor<32x480x14x14xf32>
    %v733 = stablehlo.reshape %v732 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v734 = stablehlo.logistic %v733 : tensor<32x94080xf32>
    %v735 = stablehlo.multiply %v733, %v734 : tensor<32x94080xf32>
    %v736 = stablehlo.reshape %v735 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v737 = stablehlo.convolution(%v736, %b9dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x5x5xf32>) -> tensor<32x480x14x14xf32>
    %v738 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v739 = stablehlo.add %v737, %v738 : tensor<32x480x14x14xf32>
    %v740 = stablehlo.reshape %v739 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v741 = stablehlo.reshape %v740 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v742 = stablehlo.broadcast_in_dim %b9dnmu, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v743 = stablehlo.subtract %v741, %v742 : tensor<32x480x14x14xf32>
    %v744 = stablehlo.broadcast_in_dim %b9dnvar, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v745 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v746 = stablehlo.add %v744, %v745 : tensor<32x480x14x14xf32>
    %v747 = stablehlo.rsqrt %v746 : tensor<32x480x14x14xf32>
    %v748 = stablehlo.multiply %v743, %v747 : tensor<32x480x14x14xf32>
    %v749 = stablehlo.broadcast_in_dim %b9dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v750 = stablehlo.broadcast_in_dim %b9dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v751 = stablehlo.multiply %v748, %v749 : tensor<32x480x14x14xf32>
    %v752 = stablehlo.add %v751, %v750 : tensor<32x480x14x14xf32>
    %v753 = stablehlo.reshape %v752 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v754 = stablehlo.logistic %v753 : tensor<32x94080xf32>
    %v755 = stablehlo.multiply %v753, %v754 : tensor<32x94080xf32>
    %v756 = stablehlo.reshape %v755 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v757 = stablehlo.constant dense<0.0> : tensor<f32>
    %v758 = stablehlo.reduce(%v756 init: %v757) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v759 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v760 = stablehlo.divide %v758, %v759 : tensor<32x480xf32>
    %v761 = stablehlo.dot_general %v760, %b9zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v762 = stablehlo.broadcast_in_dim %b9zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v763 = stablehlo.add %v761, %v762 : tensor<32x20xf32>
    %v764 = stablehlo.logistic %v763 : tensor<32x20xf32>
    %v765 = stablehlo.multiply %v763, %v764 : tensor<32x20xf32>
    %v766 = stablehlo.dot_general %v765, %b9zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v767 = stablehlo.broadcast_in_dim %b9zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v768 = stablehlo.add %v766, %v767 : tensor<32x480xf32>
    %v769 = stablehlo.reshape %v755 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v770 = stablehlo.constant dense<0.0> : tensor<f32>
    %v771 = stablehlo.reduce(%v769 init: %v770) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v772 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v773 = stablehlo.divide %v771, %v772 : tensor<32x480xf32>
    %v774 = stablehlo.dot_general %v773, %b9zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v775 = stablehlo.broadcast_in_dim %b9zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v776 = stablehlo.add %v774, %v775 : tensor<32x20xf32>
    %v777 = stablehlo.logistic %v776 : tensor<32x20xf32>
    %v778 = stablehlo.multiply %v776, %v777 : tensor<32x20xf32>
    %v779 = stablehlo.dot_general %v778, %b9zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v780 = stablehlo.broadcast_in_dim %b9zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v781 = stablehlo.add %v779, %v780 : tensor<32x480xf32>
    %v782 = stablehlo.logistic %v781 : tensor<32x480xf32>
    %v783 = stablehlo.broadcast_in_dim %v782, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v784 = stablehlo.multiply %v769, %v783 : tensor<32x480x14x14xf32>
    %v785 = stablehlo.reshape %v784 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v786 = stablehlo.reshape %v785 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v787 = stablehlo.convolution(%v786, %b9pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<112x480x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v788 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v789 = stablehlo.add %v787, %v788 : tensor<32x112x14x14xf32>
    %v790 = stablehlo.reshape %v789 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v791 = stablehlo.reshape %v790 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v792 = stablehlo.broadcast_in_dim %b9pnmu, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v793 = stablehlo.subtract %v791, %v792 : tensor<32x112x14x14xf32>
    %v794 = stablehlo.broadcast_in_dim %b9pnvar, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v795 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v796 = stablehlo.add %v794, %v795 : tensor<32x112x14x14xf32>
    %v797 = stablehlo.rsqrt %v796 : tensor<32x112x14x14xf32>
    %v798 = stablehlo.multiply %v793, %v797 : tensor<32x112x14x14xf32>
    %v799 = stablehlo.broadcast_in_dim %b9pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v800 = stablehlo.broadcast_in_dim %b9pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v801 = stablehlo.multiply %v798, %v799 : tensor<32x112x14x14xf32>
    %v802 = stablehlo.add %v801, %v800 : tensor<32x112x14x14xf32>
    %v803 = stablehlo.reshape %v802 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v804 = stablehlo.reshape %v803 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v805 = stablehlo.convolution(%v804, %b10eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v806 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v807 = stablehlo.add %v805, %v806 : tensor<32x672x14x14xf32>
    %v808 = stablehlo.reshape %v807 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v809 = stablehlo.reshape %v808 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v810 = stablehlo.broadcast_in_dim %b10enmu, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v811 = stablehlo.subtract %v809, %v810 : tensor<32x672x14x14xf32>
    %v812 = stablehlo.broadcast_in_dim %b10envar, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v813 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v814 = stablehlo.add %v812, %v813 : tensor<32x672x14x14xf32>
    %v815 = stablehlo.rsqrt %v814 : tensor<32x672x14x14xf32>
    %v816 = stablehlo.multiply %v811, %v815 : tensor<32x672x14x14xf32>
    %v817 = stablehlo.broadcast_in_dim %b10eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v818 = stablehlo.broadcast_in_dim %b10ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v819 = stablehlo.multiply %v816, %v817 : tensor<32x672x14x14xf32>
    %v820 = stablehlo.add %v819, %v818 : tensor<32x672x14x14xf32>
    %v821 = stablehlo.reshape %v820 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v822 = stablehlo.logistic %v821 : tensor<32x131712xf32>
    %v823 = stablehlo.multiply %v821, %v822 : tensor<32x131712xf32>
    %v824 = stablehlo.reshape %v823 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v825 = stablehlo.convolution(%v824, %b10dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %v826 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v827 = stablehlo.add %v825, %v826 : tensor<32x672x14x14xf32>
    %v828 = stablehlo.reshape %v827 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v829 = stablehlo.reshape %v828 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v830 = stablehlo.broadcast_in_dim %b10dnmu, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v831 = stablehlo.subtract %v829, %v830 : tensor<32x672x14x14xf32>
    %v832 = stablehlo.broadcast_in_dim %b10dnvar, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v833 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v834 = stablehlo.add %v832, %v833 : tensor<32x672x14x14xf32>
    %v835 = stablehlo.rsqrt %v834 : tensor<32x672x14x14xf32>
    %v836 = stablehlo.multiply %v831, %v835 : tensor<32x672x14x14xf32>
    %v837 = stablehlo.broadcast_in_dim %b10dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v838 = stablehlo.broadcast_in_dim %b10dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v839 = stablehlo.multiply %v836, %v837 : tensor<32x672x14x14xf32>
    %v840 = stablehlo.add %v839, %v838 : tensor<32x672x14x14xf32>
    %v841 = stablehlo.reshape %v840 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v842 = stablehlo.logistic %v841 : tensor<32x131712xf32>
    %v843 = stablehlo.multiply %v841, %v842 : tensor<32x131712xf32>
    %v844 = stablehlo.reshape %v843 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v845 = stablehlo.constant dense<0.0> : tensor<f32>
    %v846 = stablehlo.reduce(%v844 init: %v845) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v847 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v848 = stablehlo.divide %v846, %v847 : tensor<32x672xf32>
    %v849 = stablehlo.dot_general %v848, %b10zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v850 = stablehlo.broadcast_in_dim %b10zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v851 = stablehlo.add %v849, %v850 : tensor<32x28xf32>
    %v852 = stablehlo.logistic %v851 : tensor<32x28xf32>
    %v853 = stablehlo.multiply %v851, %v852 : tensor<32x28xf32>
    %v854 = stablehlo.dot_general %v853, %b10zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v855 = stablehlo.broadcast_in_dim %b10zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v856 = stablehlo.add %v854, %v855 : tensor<32x672xf32>
    %v857 = stablehlo.reshape %v843 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v858 = stablehlo.constant dense<0.0> : tensor<f32>
    %v859 = stablehlo.reduce(%v857 init: %v858) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v860 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v861 = stablehlo.divide %v859, %v860 : tensor<32x672xf32>
    %v862 = stablehlo.dot_general %v861, %b10zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v863 = stablehlo.broadcast_in_dim %b10zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v864 = stablehlo.add %v862, %v863 : tensor<32x28xf32>
    %v865 = stablehlo.logistic %v864 : tensor<32x28xf32>
    %v866 = stablehlo.multiply %v864, %v865 : tensor<32x28xf32>
    %v867 = stablehlo.dot_general %v866, %b10zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v868 = stablehlo.broadcast_in_dim %b10zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v869 = stablehlo.add %v867, %v868 : tensor<32x672xf32>
    %v870 = stablehlo.logistic %v869 : tensor<32x672xf32>
    %v871 = stablehlo.broadcast_in_dim %v870, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v872 = stablehlo.multiply %v857, %v871 : tensor<32x672x14x14xf32>
    %v873 = stablehlo.reshape %v872 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v874 = stablehlo.reshape %v873 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v875 = stablehlo.convolution(%v874, %b10pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v876 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v877 = stablehlo.add %v875, %v876 : tensor<32x112x14x14xf32>
    %v878 = stablehlo.reshape %v877 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v879 = stablehlo.reshape %v878 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v880 = stablehlo.broadcast_in_dim %b10pnmu, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v881 = stablehlo.subtract %v879, %v880 : tensor<32x112x14x14xf32>
    %v882 = stablehlo.broadcast_in_dim %b10pnvar, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v883 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v884 = stablehlo.add %v882, %v883 : tensor<32x112x14x14xf32>
    %v885 = stablehlo.rsqrt %v884 : tensor<32x112x14x14xf32>
    %v886 = stablehlo.multiply %v881, %v885 : tensor<32x112x14x14xf32>
    %v887 = stablehlo.broadcast_in_dim %b10pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v888 = stablehlo.broadcast_in_dim %b10pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v889 = stablehlo.multiply %v886, %v887 : tensor<32x112x14x14xf32>
    %v890 = stablehlo.add %v889, %v888 : tensor<32x112x14x14xf32>
    %v891 = stablehlo.reshape %v890 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v892 = stablehlo.broadcast_in_dim %dp9, dims = [0] : (tensor<32xf32>) -> tensor<32x21952xf32>
    %v893 = stablehlo.multiply %v892, %v891 : tensor<32x21952xf32>
    %v894 = stablehlo.add %v893, %v803 : tensor<32x21952xf32>
    %v895 = stablehlo.reshape %v894 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v896 = stablehlo.convolution(%v895, %b11eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v897 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v898 = stablehlo.add %v896, %v897 : tensor<32x672x14x14xf32>
    %v899 = stablehlo.reshape %v898 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v900 = stablehlo.reshape %v899 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v901 = stablehlo.broadcast_in_dim %b11enmu, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v902 = stablehlo.subtract %v900, %v901 : tensor<32x672x14x14xf32>
    %v903 = stablehlo.broadcast_in_dim %b11envar, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v904 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v905 = stablehlo.add %v903, %v904 : tensor<32x672x14x14xf32>
    %v906 = stablehlo.rsqrt %v905 : tensor<32x672x14x14xf32>
    %v907 = stablehlo.multiply %v902, %v906 : tensor<32x672x14x14xf32>
    %v908 = stablehlo.broadcast_in_dim %b11eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v909 = stablehlo.broadcast_in_dim %b11ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v910 = stablehlo.multiply %v907, %v908 : tensor<32x672x14x14xf32>
    %v911 = stablehlo.add %v910, %v909 : tensor<32x672x14x14xf32>
    %v912 = stablehlo.reshape %v911 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v913 = stablehlo.logistic %v912 : tensor<32x131712xf32>
    %v914 = stablehlo.multiply %v912, %v913 : tensor<32x131712xf32>
    %v915 = stablehlo.reshape %v914 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v916 = stablehlo.convolution(%v915, %b11dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %v917 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v918 = stablehlo.add %v916, %v917 : tensor<32x672x14x14xf32>
    %v919 = stablehlo.reshape %v918 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v920 = stablehlo.reshape %v919 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v921 = stablehlo.broadcast_in_dim %b11dnmu, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v922 = stablehlo.subtract %v920, %v921 : tensor<32x672x14x14xf32>
    %v923 = stablehlo.broadcast_in_dim %b11dnvar, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v924 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v925 = stablehlo.add %v923, %v924 : tensor<32x672x14x14xf32>
    %v926 = stablehlo.rsqrt %v925 : tensor<32x672x14x14xf32>
    %v927 = stablehlo.multiply %v922, %v926 : tensor<32x672x14x14xf32>
    %v928 = stablehlo.broadcast_in_dim %b11dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v929 = stablehlo.broadcast_in_dim %b11dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v930 = stablehlo.multiply %v927, %v928 : tensor<32x672x14x14xf32>
    %v931 = stablehlo.add %v930, %v929 : tensor<32x672x14x14xf32>
    %v932 = stablehlo.reshape %v931 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v933 = stablehlo.logistic %v932 : tensor<32x131712xf32>
    %v934 = stablehlo.multiply %v932, %v933 : tensor<32x131712xf32>
    %v935 = stablehlo.reshape %v934 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v936 = stablehlo.constant dense<0.0> : tensor<f32>
    %v937 = stablehlo.reduce(%v935 init: %v936) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v938 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v939 = stablehlo.divide %v937, %v938 : tensor<32x672xf32>
    %v940 = stablehlo.dot_general %v939, %b11zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v941 = stablehlo.broadcast_in_dim %b11zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v942 = stablehlo.add %v940, %v941 : tensor<32x28xf32>
    %v943 = stablehlo.logistic %v942 : tensor<32x28xf32>
    %v944 = stablehlo.multiply %v942, %v943 : tensor<32x28xf32>
    %v945 = stablehlo.dot_general %v944, %b11zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v946 = stablehlo.broadcast_in_dim %b11zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v947 = stablehlo.add %v945, %v946 : tensor<32x672xf32>
    %v948 = stablehlo.reshape %v934 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v949 = stablehlo.constant dense<0.0> : tensor<f32>
    %v950 = stablehlo.reduce(%v948 init: %v949) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v951 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v952 = stablehlo.divide %v950, %v951 : tensor<32x672xf32>
    %v953 = stablehlo.dot_general %v952, %b11zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v954 = stablehlo.broadcast_in_dim %b11zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v955 = stablehlo.add %v953, %v954 : tensor<32x28xf32>
    %v956 = stablehlo.logistic %v955 : tensor<32x28xf32>
    %v957 = stablehlo.multiply %v955, %v956 : tensor<32x28xf32>
    %v958 = stablehlo.dot_general %v957, %b11zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v959 = stablehlo.broadcast_in_dim %b11zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v960 = stablehlo.add %v958, %v959 : tensor<32x672xf32>
    %v961 = stablehlo.logistic %v960 : tensor<32x672xf32>
    %v962 = stablehlo.broadcast_in_dim %v961, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v963 = stablehlo.multiply %v948, %v962 : tensor<32x672x14x14xf32>
    %v964 = stablehlo.reshape %v963 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v965 = stablehlo.reshape %v964 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v966 = stablehlo.convolution(%v965, %b11pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v967 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v968 = stablehlo.add %v966, %v967 : tensor<32x112x14x14xf32>
    %v969 = stablehlo.reshape %v968 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v970 = stablehlo.reshape %v969 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v971 = stablehlo.broadcast_in_dim %b11pnmu, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v972 = stablehlo.subtract %v970, %v971 : tensor<32x112x14x14xf32>
    %v973 = stablehlo.broadcast_in_dim %b11pnvar, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v974 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v975 = stablehlo.add %v973, %v974 : tensor<32x112x14x14xf32>
    %v976 = stablehlo.rsqrt %v975 : tensor<32x112x14x14xf32>
    %v977 = stablehlo.multiply %v972, %v976 : tensor<32x112x14x14xf32>
    %v978 = stablehlo.broadcast_in_dim %b11pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v979 = stablehlo.broadcast_in_dim %b11pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v980 = stablehlo.multiply %v977, %v978 : tensor<32x112x14x14xf32>
    %v981 = stablehlo.add %v980, %v979 : tensor<32x112x14x14xf32>
    %v982 = stablehlo.reshape %v981 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v983 = stablehlo.broadcast_in_dim %dp10, dims = [0] : (tensor<32xf32>) -> tensor<32x21952xf32>
    %v984 = stablehlo.multiply %v983, %v982 : tensor<32x21952xf32>
    %v985 = stablehlo.add %v984, %v894 : tensor<32x21952xf32>
    %v986 = stablehlo.reshape %v985 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v987 = stablehlo.convolution(%v986, %b12eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v988 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v989 = stablehlo.add %v987, %v988 : tensor<32x672x14x14xf32>
    %v990 = stablehlo.reshape %v989 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v991 = stablehlo.reshape %v990 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v992 = stablehlo.broadcast_in_dim %b12enmu, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v993 = stablehlo.subtract %v991, %v992 : tensor<32x672x14x14xf32>
    %v994 = stablehlo.broadcast_in_dim %b12envar, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v995 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v996 = stablehlo.add %v994, %v995 : tensor<32x672x14x14xf32>
    %v997 = stablehlo.rsqrt %v996 : tensor<32x672x14x14xf32>
    %v998 = stablehlo.multiply %v993, %v997 : tensor<32x672x14x14xf32>
    %v999 = stablehlo.broadcast_in_dim %b12eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1000 = stablehlo.broadcast_in_dim %b12ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1001 = stablehlo.multiply %v998, %v999 : tensor<32x672x14x14xf32>
    %v1002 = stablehlo.add %v1001, %v1000 : tensor<32x672x14x14xf32>
    %v1003 = stablehlo.reshape %v1002 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1004 = stablehlo.logistic %v1003 : tensor<32x131712xf32>
    %v1005 = stablehlo.multiply %v1003, %v1004 : tensor<32x131712xf32>
    %v1006 = stablehlo.reshape %v1005 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1007 = stablehlo.convolution(%v1006, %b12dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x7x7xf32>
    %v1008 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1009 = stablehlo.add %v1007, %v1008 : tensor<32x672x7x7xf32>
    %v1010 = stablehlo.reshape %v1009 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v1011 = stablehlo.reshape %v1010 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1012 = stablehlo.broadcast_in_dim %b12dnmu, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1013 = stablehlo.subtract %v1011, %v1012 : tensor<32x672x7x7xf32>
    %v1014 = stablehlo.broadcast_in_dim %b12dnvar, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1015 = stablehlo.constant dense<1.0e-5> : tensor<32x672x7x7xf32>
    %v1016 = stablehlo.add %v1014, %v1015 : tensor<32x672x7x7xf32>
    %v1017 = stablehlo.rsqrt %v1016 : tensor<32x672x7x7xf32>
    %v1018 = stablehlo.multiply %v1013, %v1017 : tensor<32x672x7x7xf32>
    %v1019 = stablehlo.broadcast_in_dim %b12dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1020 = stablehlo.broadcast_in_dim %b12dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1021 = stablehlo.multiply %v1018, %v1019 : tensor<32x672x7x7xf32>
    %v1022 = stablehlo.add %v1021, %v1020 : tensor<32x672x7x7xf32>
    %v1023 = stablehlo.reshape %v1022 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v1024 = stablehlo.logistic %v1023 : tensor<32x32928xf32>
    %v1025 = stablehlo.multiply %v1023, %v1024 : tensor<32x32928xf32>
    %v1026 = stablehlo.reshape %v1025 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1027 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1028 = stablehlo.reduce(%v1026 init: %v1027) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v1029 = stablehlo.constant dense<49.0> : tensor<32x672xf32>
    %v1030 = stablehlo.divide %v1028, %v1029 : tensor<32x672xf32>
    %v1031 = stablehlo.dot_general %v1030, %b12zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v1032 = stablehlo.broadcast_in_dim %b12zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v1033 = stablehlo.add %v1031, %v1032 : tensor<32x28xf32>
    %v1034 = stablehlo.logistic %v1033 : tensor<32x28xf32>
    %v1035 = stablehlo.multiply %v1033, %v1034 : tensor<32x28xf32>
    %v1036 = stablehlo.dot_general %v1035, %b12zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v1037 = stablehlo.broadcast_in_dim %b12zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v1038 = stablehlo.add %v1036, %v1037 : tensor<32x672xf32>
    %v1039 = stablehlo.reshape %v1025 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1040 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1041 = stablehlo.reduce(%v1039 init: %v1040) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v1042 = stablehlo.constant dense<49.0> : tensor<32x672xf32>
    %v1043 = stablehlo.divide %v1041, %v1042 : tensor<32x672xf32>
    %v1044 = stablehlo.dot_general %v1043, %b12zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v1045 = stablehlo.broadcast_in_dim %b12zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v1046 = stablehlo.add %v1044, %v1045 : tensor<32x28xf32>
    %v1047 = stablehlo.logistic %v1046 : tensor<32x28xf32>
    %v1048 = stablehlo.multiply %v1046, %v1047 : tensor<32x28xf32>
    %v1049 = stablehlo.dot_general %v1048, %b12zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v1050 = stablehlo.broadcast_in_dim %b12zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v1051 = stablehlo.add %v1049, %v1050 : tensor<32x672xf32>
    %v1052 = stablehlo.logistic %v1051 : tensor<32x672xf32>
    %v1053 = stablehlo.broadcast_in_dim %v1052, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x7x7xf32>
    %v1054 = stablehlo.multiply %v1039, %v1053 : tensor<32x672x7x7xf32>
    %v1055 = stablehlo.reshape %v1054 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v1056 = stablehlo.reshape %v1055 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1057 = stablehlo.convolution(%v1056, %b12pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x7x7xf32>, tensor<192x672x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1058 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1059 = stablehlo.add %v1057, %v1058 : tensor<32x192x7x7xf32>
    %v1060 = stablehlo.reshape %v1059 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1061 = stablehlo.reshape %v1060 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1062 = stablehlo.broadcast_in_dim %b12pnmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1063 = stablehlo.subtract %v1061, %v1062 : tensor<32x192x7x7xf32>
    %v1064 = stablehlo.broadcast_in_dim %b12pnvar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1065 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1066 = stablehlo.add %v1064, %v1065 : tensor<32x192x7x7xf32>
    %v1067 = stablehlo.rsqrt %v1066 : tensor<32x192x7x7xf32>
    %v1068 = stablehlo.multiply %v1063, %v1067 : tensor<32x192x7x7xf32>
    %v1069 = stablehlo.broadcast_in_dim %b12pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1070 = stablehlo.broadcast_in_dim %b12pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1071 = stablehlo.multiply %v1068, %v1069 : tensor<32x192x7x7xf32>
    %v1072 = stablehlo.add %v1071, %v1070 : tensor<32x192x7x7xf32>
    %v1073 = stablehlo.reshape %v1072 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1074 = stablehlo.reshape %v1073 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1075 = stablehlo.convolution(%v1074, %b13eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1076 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1077 = stablehlo.add %v1075, %v1076 : tensor<32x1152x7x7xf32>
    %v1078 = stablehlo.reshape %v1077 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1079 = stablehlo.reshape %v1078 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1080 = stablehlo.broadcast_in_dim %b13enmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1081 = stablehlo.subtract %v1079, %v1080 : tensor<32x1152x7x7xf32>
    %v1082 = stablehlo.broadcast_in_dim %b13envar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1083 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1084 = stablehlo.add %v1082, %v1083 : tensor<32x1152x7x7xf32>
    %v1085 = stablehlo.rsqrt %v1084 : tensor<32x1152x7x7xf32>
    %v1086 = stablehlo.multiply %v1081, %v1085 : tensor<32x1152x7x7xf32>
    %v1087 = stablehlo.broadcast_in_dim %b13eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1088 = stablehlo.broadcast_in_dim %b13ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1089 = stablehlo.multiply %v1086, %v1087 : tensor<32x1152x7x7xf32>
    %v1090 = stablehlo.add %v1089, %v1088 : tensor<32x1152x7x7xf32>
    %v1091 = stablehlo.reshape %v1090 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1092 = stablehlo.logistic %v1091 : tensor<32x56448xf32>
    %v1093 = stablehlo.multiply %v1091, %v1092 : tensor<32x56448xf32>
    %v1094 = stablehlo.reshape %v1093 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1095 = stablehlo.convolution(%v1094, %b13dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v1096 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1097 = stablehlo.add %v1095, %v1096 : tensor<32x1152x7x7xf32>
    %v1098 = stablehlo.reshape %v1097 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1099 = stablehlo.reshape %v1098 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1100 = stablehlo.broadcast_in_dim %b13dnmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1101 = stablehlo.subtract %v1099, %v1100 : tensor<32x1152x7x7xf32>
    %v1102 = stablehlo.broadcast_in_dim %b13dnvar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1103 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1104 = stablehlo.add %v1102, %v1103 : tensor<32x1152x7x7xf32>
    %v1105 = stablehlo.rsqrt %v1104 : tensor<32x1152x7x7xf32>
    %v1106 = stablehlo.multiply %v1101, %v1105 : tensor<32x1152x7x7xf32>
    %v1107 = stablehlo.broadcast_in_dim %b13dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1108 = stablehlo.broadcast_in_dim %b13dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1109 = stablehlo.multiply %v1106, %v1107 : tensor<32x1152x7x7xf32>
    %v1110 = stablehlo.add %v1109, %v1108 : tensor<32x1152x7x7xf32>
    %v1111 = stablehlo.reshape %v1110 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1112 = stablehlo.logistic %v1111 : tensor<32x56448xf32>
    %v1113 = stablehlo.multiply %v1111, %v1112 : tensor<32x56448xf32>
    %v1114 = stablehlo.reshape %v1113 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1115 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1116 = stablehlo.reduce(%v1114 init: %v1115) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1117 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1118 = stablehlo.divide %v1116, %v1117 : tensor<32x1152xf32>
    %v1119 = stablehlo.dot_general %v1118, %b13zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1120 = stablehlo.broadcast_in_dim %b13zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1121 = stablehlo.add %v1119, %v1120 : tensor<32x48xf32>
    %v1122 = stablehlo.logistic %v1121 : tensor<32x48xf32>
    %v1123 = stablehlo.multiply %v1121, %v1122 : tensor<32x48xf32>
    %v1124 = stablehlo.dot_general %v1123, %b13zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1125 = stablehlo.broadcast_in_dim %b13zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1126 = stablehlo.add %v1124, %v1125 : tensor<32x1152xf32>
    %v1127 = stablehlo.reshape %v1113 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1128 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1129 = stablehlo.reduce(%v1127 init: %v1128) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1130 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1131 = stablehlo.divide %v1129, %v1130 : tensor<32x1152xf32>
    %v1132 = stablehlo.dot_general %v1131, %b13zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1133 = stablehlo.broadcast_in_dim %b13zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1134 = stablehlo.add %v1132, %v1133 : tensor<32x48xf32>
    %v1135 = stablehlo.logistic %v1134 : tensor<32x48xf32>
    %v1136 = stablehlo.multiply %v1134, %v1135 : tensor<32x48xf32>
    %v1137 = stablehlo.dot_general %v1136, %b13zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1138 = stablehlo.broadcast_in_dim %b13zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1139 = stablehlo.add %v1137, %v1138 : tensor<32x1152xf32>
    %v1140 = stablehlo.logistic %v1139 : tensor<32x1152xf32>
    %v1141 = stablehlo.broadcast_in_dim %v1140, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1142 = stablehlo.multiply %v1127, %v1141 : tensor<32x1152x7x7xf32>
    %v1143 = stablehlo.reshape %v1142 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1144 = stablehlo.reshape %v1143 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1145 = stablehlo.convolution(%v1144, %b13pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1146 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1147 = stablehlo.add %v1145, %v1146 : tensor<32x192x7x7xf32>
    %v1148 = stablehlo.reshape %v1147 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1149 = stablehlo.reshape %v1148 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1150 = stablehlo.broadcast_in_dim %b13pnmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1151 = stablehlo.subtract %v1149, %v1150 : tensor<32x192x7x7xf32>
    %v1152 = stablehlo.broadcast_in_dim %b13pnvar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1153 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1154 = stablehlo.add %v1152, %v1153 : tensor<32x192x7x7xf32>
    %v1155 = stablehlo.rsqrt %v1154 : tensor<32x192x7x7xf32>
    %v1156 = stablehlo.multiply %v1151, %v1155 : tensor<32x192x7x7xf32>
    %v1157 = stablehlo.broadcast_in_dim %b13pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1158 = stablehlo.broadcast_in_dim %b13pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1159 = stablehlo.multiply %v1156, %v1157 : tensor<32x192x7x7xf32>
    %v1160 = stablehlo.add %v1159, %v1158 : tensor<32x192x7x7xf32>
    %v1161 = stablehlo.reshape %v1160 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1162 = stablehlo.broadcast_in_dim %dp12, dims = [0] : (tensor<32xf32>) -> tensor<32x9408xf32>
    %v1163 = stablehlo.multiply %v1162, %v1161 : tensor<32x9408xf32>
    %v1164 = stablehlo.add %v1163, %v1073 : tensor<32x9408xf32>
    %v1165 = stablehlo.reshape %v1164 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1166 = stablehlo.convolution(%v1165, %b14eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1167 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1168 = stablehlo.add %v1166, %v1167 : tensor<32x1152x7x7xf32>
    %v1169 = stablehlo.reshape %v1168 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1170 = stablehlo.reshape %v1169 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1171 = stablehlo.broadcast_in_dim %b14enmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1172 = stablehlo.subtract %v1170, %v1171 : tensor<32x1152x7x7xf32>
    %v1173 = stablehlo.broadcast_in_dim %b14envar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1174 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1175 = stablehlo.add %v1173, %v1174 : tensor<32x1152x7x7xf32>
    %v1176 = stablehlo.rsqrt %v1175 : tensor<32x1152x7x7xf32>
    %v1177 = stablehlo.multiply %v1172, %v1176 : tensor<32x1152x7x7xf32>
    %v1178 = stablehlo.broadcast_in_dim %b14eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1179 = stablehlo.broadcast_in_dim %b14ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1180 = stablehlo.multiply %v1177, %v1178 : tensor<32x1152x7x7xf32>
    %v1181 = stablehlo.add %v1180, %v1179 : tensor<32x1152x7x7xf32>
    %v1182 = stablehlo.reshape %v1181 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1183 = stablehlo.logistic %v1182 : tensor<32x56448xf32>
    %v1184 = stablehlo.multiply %v1182, %v1183 : tensor<32x56448xf32>
    %v1185 = stablehlo.reshape %v1184 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1186 = stablehlo.convolution(%v1185, %b14dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v1187 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1188 = stablehlo.add %v1186, %v1187 : tensor<32x1152x7x7xf32>
    %v1189 = stablehlo.reshape %v1188 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1190 = stablehlo.reshape %v1189 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1191 = stablehlo.broadcast_in_dim %b14dnmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1192 = stablehlo.subtract %v1190, %v1191 : tensor<32x1152x7x7xf32>
    %v1193 = stablehlo.broadcast_in_dim %b14dnvar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1194 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1195 = stablehlo.add %v1193, %v1194 : tensor<32x1152x7x7xf32>
    %v1196 = stablehlo.rsqrt %v1195 : tensor<32x1152x7x7xf32>
    %v1197 = stablehlo.multiply %v1192, %v1196 : tensor<32x1152x7x7xf32>
    %v1198 = stablehlo.broadcast_in_dim %b14dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1199 = stablehlo.broadcast_in_dim %b14dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1200 = stablehlo.multiply %v1197, %v1198 : tensor<32x1152x7x7xf32>
    %v1201 = stablehlo.add %v1200, %v1199 : tensor<32x1152x7x7xf32>
    %v1202 = stablehlo.reshape %v1201 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1203 = stablehlo.logistic %v1202 : tensor<32x56448xf32>
    %v1204 = stablehlo.multiply %v1202, %v1203 : tensor<32x56448xf32>
    %v1205 = stablehlo.reshape %v1204 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1206 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1207 = stablehlo.reduce(%v1205 init: %v1206) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1208 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1209 = stablehlo.divide %v1207, %v1208 : tensor<32x1152xf32>
    %v1210 = stablehlo.dot_general %v1209, %b14zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1211 = stablehlo.broadcast_in_dim %b14zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1212 = stablehlo.add %v1210, %v1211 : tensor<32x48xf32>
    %v1213 = stablehlo.logistic %v1212 : tensor<32x48xf32>
    %v1214 = stablehlo.multiply %v1212, %v1213 : tensor<32x48xf32>
    %v1215 = stablehlo.dot_general %v1214, %b14zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1216 = stablehlo.broadcast_in_dim %b14zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1217 = stablehlo.add %v1215, %v1216 : tensor<32x1152xf32>
    %v1218 = stablehlo.reshape %v1204 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1219 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1220 = stablehlo.reduce(%v1218 init: %v1219) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1221 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1222 = stablehlo.divide %v1220, %v1221 : tensor<32x1152xf32>
    %v1223 = stablehlo.dot_general %v1222, %b14zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1224 = stablehlo.broadcast_in_dim %b14zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1225 = stablehlo.add %v1223, %v1224 : tensor<32x48xf32>
    %v1226 = stablehlo.logistic %v1225 : tensor<32x48xf32>
    %v1227 = stablehlo.multiply %v1225, %v1226 : tensor<32x48xf32>
    %v1228 = stablehlo.dot_general %v1227, %b14zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1229 = stablehlo.broadcast_in_dim %b14zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1230 = stablehlo.add %v1228, %v1229 : tensor<32x1152xf32>
    %v1231 = stablehlo.logistic %v1230 : tensor<32x1152xf32>
    %v1232 = stablehlo.broadcast_in_dim %v1231, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1233 = stablehlo.multiply %v1218, %v1232 : tensor<32x1152x7x7xf32>
    %v1234 = stablehlo.reshape %v1233 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1235 = stablehlo.reshape %v1234 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1236 = stablehlo.convolution(%v1235, %b14pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1237 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1238 = stablehlo.add %v1236, %v1237 : tensor<32x192x7x7xf32>
    %v1239 = stablehlo.reshape %v1238 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1240 = stablehlo.reshape %v1239 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1241 = stablehlo.broadcast_in_dim %b14pnmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1242 = stablehlo.subtract %v1240, %v1241 : tensor<32x192x7x7xf32>
    %v1243 = stablehlo.broadcast_in_dim %b14pnvar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1244 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1245 = stablehlo.add %v1243, %v1244 : tensor<32x192x7x7xf32>
    %v1246 = stablehlo.rsqrt %v1245 : tensor<32x192x7x7xf32>
    %v1247 = stablehlo.multiply %v1242, %v1246 : tensor<32x192x7x7xf32>
    %v1248 = stablehlo.broadcast_in_dim %b14pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1249 = stablehlo.broadcast_in_dim %b14pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1250 = stablehlo.multiply %v1247, %v1248 : tensor<32x192x7x7xf32>
    %v1251 = stablehlo.add %v1250, %v1249 : tensor<32x192x7x7xf32>
    %v1252 = stablehlo.reshape %v1251 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1253 = stablehlo.broadcast_in_dim %dp13, dims = [0] : (tensor<32xf32>) -> tensor<32x9408xf32>
    %v1254 = stablehlo.multiply %v1253, %v1252 : tensor<32x9408xf32>
    %v1255 = stablehlo.add %v1254, %v1164 : tensor<32x9408xf32>
    %v1256 = stablehlo.reshape %v1255 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1257 = stablehlo.convolution(%v1256, %b15eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1258 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1259 = stablehlo.add %v1257, %v1258 : tensor<32x1152x7x7xf32>
    %v1260 = stablehlo.reshape %v1259 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1261 = stablehlo.reshape %v1260 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1262 = stablehlo.broadcast_in_dim %b15enmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1263 = stablehlo.subtract %v1261, %v1262 : tensor<32x1152x7x7xf32>
    %v1264 = stablehlo.broadcast_in_dim %b15envar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1265 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1266 = stablehlo.add %v1264, %v1265 : tensor<32x1152x7x7xf32>
    %v1267 = stablehlo.rsqrt %v1266 : tensor<32x1152x7x7xf32>
    %v1268 = stablehlo.multiply %v1263, %v1267 : tensor<32x1152x7x7xf32>
    %v1269 = stablehlo.broadcast_in_dim %b15eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1270 = stablehlo.broadcast_in_dim %b15ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1271 = stablehlo.multiply %v1268, %v1269 : tensor<32x1152x7x7xf32>
    %v1272 = stablehlo.add %v1271, %v1270 : tensor<32x1152x7x7xf32>
    %v1273 = stablehlo.reshape %v1272 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1274 = stablehlo.logistic %v1273 : tensor<32x56448xf32>
    %v1275 = stablehlo.multiply %v1273, %v1274 : tensor<32x56448xf32>
    %v1276 = stablehlo.reshape %v1275 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1277 = stablehlo.convolution(%v1276, %b15dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v1278 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1279 = stablehlo.add %v1277, %v1278 : tensor<32x1152x7x7xf32>
    %v1280 = stablehlo.reshape %v1279 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1281 = stablehlo.reshape %v1280 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1282 = stablehlo.broadcast_in_dim %b15dnmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1283 = stablehlo.subtract %v1281, %v1282 : tensor<32x1152x7x7xf32>
    %v1284 = stablehlo.broadcast_in_dim %b15dnvar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1285 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1286 = stablehlo.add %v1284, %v1285 : tensor<32x1152x7x7xf32>
    %v1287 = stablehlo.rsqrt %v1286 : tensor<32x1152x7x7xf32>
    %v1288 = stablehlo.multiply %v1283, %v1287 : tensor<32x1152x7x7xf32>
    %v1289 = stablehlo.broadcast_in_dim %b15dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1290 = stablehlo.broadcast_in_dim %b15dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1291 = stablehlo.multiply %v1288, %v1289 : tensor<32x1152x7x7xf32>
    %v1292 = stablehlo.add %v1291, %v1290 : tensor<32x1152x7x7xf32>
    %v1293 = stablehlo.reshape %v1292 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1294 = stablehlo.logistic %v1293 : tensor<32x56448xf32>
    %v1295 = stablehlo.multiply %v1293, %v1294 : tensor<32x56448xf32>
    %v1296 = stablehlo.reshape %v1295 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1297 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1298 = stablehlo.reduce(%v1296 init: %v1297) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1299 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1300 = stablehlo.divide %v1298, %v1299 : tensor<32x1152xf32>
    %v1301 = stablehlo.dot_general %v1300, %b15zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1302 = stablehlo.broadcast_in_dim %b15zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1303 = stablehlo.add %v1301, %v1302 : tensor<32x48xf32>
    %v1304 = stablehlo.logistic %v1303 : tensor<32x48xf32>
    %v1305 = stablehlo.multiply %v1303, %v1304 : tensor<32x48xf32>
    %v1306 = stablehlo.dot_general %v1305, %b15zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1307 = stablehlo.broadcast_in_dim %b15zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1308 = stablehlo.add %v1306, %v1307 : tensor<32x1152xf32>
    %v1309 = stablehlo.reshape %v1295 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1310 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1311 = stablehlo.reduce(%v1309 init: %v1310) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1312 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1313 = stablehlo.divide %v1311, %v1312 : tensor<32x1152xf32>
    %v1314 = stablehlo.dot_general %v1313, %b15zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1315 = stablehlo.broadcast_in_dim %b15zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1316 = stablehlo.add %v1314, %v1315 : tensor<32x48xf32>
    %v1317 = stablehlo.logistic %v1316 : tensor<32x48xf32>
    %v1318 = stablehlo.multiply %v1316, %v1317 : tensor<32x48xf32>
    %v1319 = stablehlo.dot_general %v1318, %b15zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1320 = stablehlo.broadcast_in_dim %b15zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1321 = stablehlo.add %v1319, %v1320 : tensor<32x1152xf32>
    %v1322 = stablehlo.logistic %v1321 : tensor<32x1152xf32>
    %v1323 = stablehlo.broadcast_in_dim %v1322, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1324 = stablehlo.multiply %v1309, %v1323 : tensor<32x1152x7x7xf32>
    %v1325 = stablehlo.reshape %v1324 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1326 = stablehlo.reshape %v1325 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1327 = stablehlo.convolution(%v1326, %b15pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1328 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1329 = stablehlo.add %v1327, %v1328 : tensor<32x192x7x7xf32>
    %v1330 = stablehlo.reshape %v1329 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1331 = stablehlo.reshape %v1330 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1332 = stablehlo.broadcast_in_dim %b15pnmu, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1333 = stablehlo.subtract %v1331, %v1332 : tensor<32x192x7x7xf32>
    %v1334 = stablehlo.broadcast_in_dim %b15pnvar, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1335 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1336 = stablehlo.add %v1334, %v1335 : tensor<32x192x7x7xf32>
    %v1337 = stablehlo.rsqrt %v1336 : tensor<32x192x7x7xf32>
    %v1338 = stablehlo.multiply %v1333, %v1337 : tensor<32x192x7x7xf32>
    %v1339 = stablehlo.broadcast_in_dim %b15pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1340 = stablehlo.broadcast_in_dim %b15pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1341 = stablehlo.multiply %v1338, %v1339 : tensor<32x192x7x7xf32>
    %v1342 = stablehlo.add %v1341, %v1340 : tensor<32x192x7x7xf32>
    %v1343 = stablehlo.reshape %v1342 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1344 = stablehlo.broadcast_in_dim %dp14, dims = [0] : (tensor<32xf32>) -> tensor<32x9408xf32>
    %v1345 = stablehlo.multiply %v1344, %v1343 : tensor<32x9408xf32>
    %v1346 = stablehlo.add %v1345, %v1255 : tensor<32x9408xf32>
    %v1347 = stablehlo.reshape %v1346 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1348 = stablehlo.convolution(%v1347, %b16eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1349 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1350 = stablehlo.add %v1348, %v1349 : tensor<32x1152x7x7xf32>
    %v1351 = stablehlo.reshape %v1350 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1352 = stablehlo.reshape %v1351 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1353 = stablehlo.broadcast_in_dim %b16enmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1354 = stablehlo.subtract %v1352, %v1353 : tensor<32x1152x7x7xf32>
    %v1355 = stablehlo.broadcast_in_dim %b16envar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1356 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1357 = stablehlo.add %v1355, %v1356 : tensor<32x1152x7x7xf32>
    %v1358 = stablehlo.rsqrt %v1357 : tensor<32x1152x7x7xf32>
    %v1359 = stablehlo.multiply %v1354, %v1358 : tensor<32x1152x7x7xf32>
    %v1360 = stablehlo.broadcast_in_dim %b16eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1361 = stablehlo.broadcast_in_dim %b16ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1362 = stablehlo.multiply %v1359, %v1360 : tensor<32x1152x7x7xf32>
    %v1363 = stablehlo.add %v1362, %v1361 : tensor<32x1152x7x7xf32>
    %v1364 = stablehlo.reshape %v1363 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1365 = stablehlo.logistic %v1364 : tensor<32x56448xf32>
    %v1366 = stablehlo.multiply %v1364, %v1365 : tensor<32x56448xf32>
    %v1367 = stablehlo.reshape %v1366 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1368 = stablehlo.convolution(%v1367, %b16dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x3x3xf32>) -> tensor<32x1152x7x7xf32>
    %v1369 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1370 = stablehlo.add %v1368, %v1369 : tensor<32x1152x7x7xf32>
    %v1371 = stablehlo.reshape %v1370 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1372 = stablehlo.reshape %v1371 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1373 = stablehlo.broadcast_in_dim %b16dnmu, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1374 = stablehlo.subtract %v1372, %v1373 : tensor<32x1152x7x7xf32>
    %v1375 = stablehlo.broadcast_in_dim %b16dnvar, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1376 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1377 = stablehlo.add %v1375, %v1376 : tensor<32x1152x7x7xf32>
    %v1378 = stablehlo.rsqrt %v1377 : tensor<32x1152x7x7xf32>
    %v1379 = stablehlo.multiply %v1374, %v1378 : tensor<32x1152x7x7xf32>
    %v1380 = stablehlo.broadcast_in_dim %b16dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1381 = stablehlo.broadcast_in_dim %b16dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1382 = stablehlo.multiply %v1379, %v1380 : tensor<32x1152x7x7xf32>
    %v1383 = stablehlo.add %v1382, %v1381 : tensor<32x1152x7x7xf32>
    %v1384 = stablehlo.reshape %v1383 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1385 = stablehlo.logistic %v1384 : tensor<32x56448xf32>
    %v1386 = stablehlo.multiply %v1384, %v1385 : tensor<32x56448xf32>
    %v1387 = stablehlo.reshape %v1386 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1388 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1389 = stablehlo.reduce(%v1387 init: %v1388) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1390 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1391 = stablehlo.divide %v1389, %v1390 : tensor<32x1152xf32>
    %v1392 = stablehlo.dot_general %v1391, %b16zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1393 = stablehlo.broadcast_in_dim %b16zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1394 = stablehlo.add %v1392, %v1393 : tensor<32x48xf32>
    %v1395 = stablehlo.logistic %v1394 : tensor<32x48xf32>
    %v1396 = stablehlo.multiply %v1394, %v1395 : tensor<32x48xf32>
    %v1397 = stablehlo.dot_general %v1396, %b16zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1398 = stablehlo.broadcast_in_dim %b16zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1399 = stablehlo.add %v1397, %v1398 : tensor<32x1152xf32>
    %v1400 = stablehlo.reshape %v1386 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1401 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1402 = stablehlo.reduce(%v1400 init: %v1401) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1403 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1404 = stablehlo.divide %v1402, %v1403 : tensor<32x1152xf32>
    %v1405 = stablehlo.dot_general %v1404, %b16zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1406 = stablehlo.broadcast_in_dim %b16zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1407 = stablehlo.add %v1405, %v1406 : tensor<32x48xf32>
    %v1408 = stablehlo.logistic %v1407 : tensor<32x48xf32>
    %v1409 = stablehlo.multiply %v1407, %v1408 : tensor<32x48xf32>
    %v1410 = stablehlo.dot_general %v1409, %b16zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1411 = stablehlo.broadcast_in_dim %b16zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1412 = stablehlo.add %v1410, %v1411 : tensor<32x1152xf32>
    %v1413 = stablehlo.logistic %v1412 : tensor<32x1152xf32>
    %v1414 = stablehlo.broadcast_in_dim %v1413, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1415 = stablehlo.multiply %v1400, %v1414 : tensor<32x1152x7x7xf32>
    %v1416 = stablehlo.reshape %v1415 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1417 = stablehlo.reshape %v1416 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1418 = stablehlo.convolution(%v1417, %b16pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<320x1152x1x1xf32>) -> tensor<32x320x7x7xf32>
    %v1419 = stablehlo.broadcast_in_dim %zb320, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1420 = stablehlo.add %v1418, %v1419 : tensor<32x320x7x7xf32>
    %v1421 = stablehlo.reshape %v1420 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1422 = stablehlo.reshape %v1421 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1423 = stablehlo.broadcast_in_dim %b16pnmu, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1424 = stablehlo.subtract %v1422, %v1423 : tensor<32x320x7x7xf32>
    %v1425 = stablehlo.broadcast_in_dim %b16pnvar, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1426 = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %v1427 = stablehlo.add %v1425, %v1426 : tensor<32x320x7x7xf32>
    %v1428 = stablehlo.rsqrt %v1427 : tensor<32x320x7x7xf32>
    %v1429 = stablehlo.multiply %v1424, %v1428 : tensor<32x320x7x7xf32>
    %v1430 = stablehlo.broadcast_in_dim %b16pg, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1431 = stablehlo.broadcast_in_dim %b16pbt, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1432 = stablehlo.multiply %v1429, %v1430 : tensor<32x320x7x7xf32>
    %v1433 = stablehlo.add %v1432, %v1431 : tensor<32x320x7x7xf32>
    %v1434 = stablehlo.reshape %v1433 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1435 = stablehlo.reshape %v1434 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1436 = stablehlo.convolution(%v1435, %hW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x320x7x7xf32>, tensor<1280x320x1x1xf32>) -> tensor<32x1280x7x7xf32>
    %v1437 = stablehlo.broadcast_in_dim %zb1280, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1438 = stablehlo.add %v1436, %v1437 : tensor<32x1280x7x7xf32>
    %v1439 = stablehlo.reshape %v1438 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1440 = stablehlo.reshape %v1439 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1441 = stablehlo.broadcast_in_dim %hnmu, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1442 = stablehlo.subtract %v1440, %v1441 : tensor<32x1280x7x7xf32>
    %v1443 = stablehlo.broadcast_in_dim %hnvar, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1444 = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %v1445 = stablehlo.add %v1443, %v1444 : tensor<32x1280x7x7xf32>
    %v1446 = stablehlo.rsqrt %v1445 : tensor<32x1280x7x7xf32>
    %v1447 = stablehlo.multiply %v1442, %v1446 : tensor<32x1280x7x7xf32>
    %v1448 = stablehlo.broadcast_in_dim %hg, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1449 = stablehlo.broadcast_in_dim %hbt, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1450 = stablehlo.multiply %v1447, %v1448 : tensor<32x1280x7x7xf32>
    %v1451 = stablehlo.add %v1450, %v1449 : tensor<32x1280x7x7xf32>
    %v1452 = stablehlo.reshape %v1451 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1453 = stablehlo.logistic %v1452 : tensor<32x62720xf32>
    %v1454 = stablehlo.multiply %v1452, %v1453 : tensor<32x62720xf32>
    %v1455 = stablehlo.reshape %v1454 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1456 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1457 = stablehlo.reduce(%v1455 init: %v1456) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v1458 = stablehlo.constant dense<49.0> : tensor<32x1280xf32>
    %v1459 = stablehlo.divide %v1457, %v1458 : tensor<32x1280xf32>
    %v1460 = stablehlo.dot_general %v1459, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1280xf32>, tensor<1280x10xf32>) -> tensor<32x10xf32>
    %v1461 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v1462 = stablehlo.add %v1460, %v1461 : tensor<32x10xf32>
    return %v1462 : tensor<32x10xf32>
  }
}
