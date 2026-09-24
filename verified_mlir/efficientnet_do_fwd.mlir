module @m {
  func.func @efficientnet_do_fwd(%x: tensor<32x150528xf32>, %sW: tensor<32x3x3x3xf32>, %sg: tensor<32xf32>, %sbt: tensor<32xf32>, %b1dW: tensor<32x1x3x3xf32>, %b1dg: tensor<32xf32>, %b1dbt: tensor<32xf32>, %b1zW1: tensor<32x8xf32>, %b1zb1: tensor<8xf32>, %b1zW2: tensor<8x32xf32>, %b1zb2: tensor<32xf32>, %b1pW: tensor<16x32x1x1xf32>, %b1pg: tensor<16xf32>, %b1pbt: tensor<16xf32>, %b2eW: tensor<96x16x1x1xf32>, %b2eg: tensor<96xf32>, %b2ebt: tensor<96xf32>, %b2dW: tensor<96x1x3x3xf32>, %b2dg: tensor<96xf32>, %b2dbt: tensor<96xf32>, %b2zW1: tensor<96x4xf32>, %b2zb1: tensor<4xf32>, %b2zW2: tensor<4x96xf32>, %b2zb2: tensor<96xf32>, %b2pW: tensor<24x96x1x1xf32>, %b2pg: tensor<24xf32>, %b2pbt: tensor<24xf32>, %b3eW: tensor<144x24x1x1xf32>, %b3eg: tensor<144xf32>, %b3ebt: tensor<144xf32>, %b3dW: tensor<144x1x3x3xf32>, %b3dg: tensor<144xf32>, %b3dbt: tensor<144xf32>, %b3zW1: tensor<144x6xf32>, %b3zb1: tensor<6xf32>, %b3zW2: tensor<6x144xf32>, %b3zb2: tensor<144xf32>, %b3pW: tensor<24x144x1x1xf32>, %b3pg: tensor<24xf32>, %b3pbt: tensor<24xf32>, %b4eW: tensor<144x24x1x1xf32>, %b4eg: tensor<144xf32>, %b4ebt: tensor<144xf32>, %b4dW: tensor<144x1x5x5xf32>, %b4dg: tensor<144xf32>, %b4dbt: tensor<144xf32>, %b4zW1: tensor<144x6xf32>, %b4zb1: tensor<6xf32>, %b4zW2: tensor<6x144xf32>, %b4zb2: tensor<144xf32>, %b4pW: tensor<40x144x1x1xf32>, %b4pg: tensor<40xf32>, %b4pbt: tensor<40xf32>, %b5eW: tensor<240x40x1x1xf32>, %b5eg: tensor<240xf32>, %b5ebt: tensor<240xf32>, %b5dW: tensor<240x1x5x5xf32>, %b5dg: tensor<240xf32>, %b5dbt: tensor<240xf32>, %b5zW1: tensor<240x10xf32>, %b5zb1: tensor<10xf32>, %b5zW2: tensor<10x240xf32>, %b5zb2: tensor<240xf32>, %b5pW: tensor<40x240x1x1xf32>, %b5pg: tensor<40xf32>, %b5pbt: tensor<40xf32>, %b6eW: tensor<240x40x1x1xf32>, %b6eg: tensor<240xf32>, %b6ebt: tensor<240xf32>, %b6dW: tensor<240x1x3x3xf32>, %b6dg: tensor<240xf32>, %b6dbt: tensor<240xf32>, %b6zW1: tensor<240x10xf32>, %b6zb1: tensor<10xf32>, %b6zW2: tensor<10x240xf32>, %b6zb2: tensor<240xf32>, %b6pW: tensor<80x240x1x1xf32>, %b6pg: tensor<80xf32>, %b6pbt: tensor<80xf32>, %b7eW: tensor<480x80x1x1xf32>, %b7eg: tensor<480xf32>, %b7ebt: tensor<480xf32>, %b7dW: tensor<480x1x3x3xf32>, %b7dg: tensor<480xf32>, %b7dbt: tensor<480xf32>, %b7zW1: tensor<480x20xf32>, %b7zb1: tensor<20xf32>, %b7zW2: tensor<20x480xf32>, %b7zb2: tensor<480xf32>, %b7pW: tensor<80x480x1x1xf32>, %b7pg: tensor<80xf32>, %b7pbt: tensor<80xf32>, %b8eW: tensor<480x80x1x1xf32>, %b8eg: tensor<480xf32>, %b8ebt: tensor<480xf32>, %b8dW: tensor<480x1x3x3xf32>, %b8dg: tensor<480xf32>, %b8dbt: tensor<480xf32>, %b8zW1: tensor<480x20xf32>, %b8zb1: tensor<20xf32>, %b8zW2: tensor<20x480xf32>, %b8zb2: tensor<480xf32>, %b8pW: tensor<80x480x1x1xf32>, %b8pg: tensor<80xf32>, %b8pbt: tensor<80xf32>, %b9eW: tensor<480x80x1x1xf32>, %b9eg: tensor<480xf32>, %b9ebt: tensor<480xf32>, %b9dW: tensor<480x1x5x5xf32>, %b9dg: tensor<480xf32>, %b9dbt: tensor<480xf32>, %b9zW1: tensor<480x20xf32>, %b9zb1: tensor<20xf32>, %b9zW2: tensor<20x480xf32>, %b9zb2: tensor<480xf32>, %b9pW: tensor<112x480x1x1xf32>, %b9pg: tensor<112xf32>, %b9pbt: tensor<112xf32>, %b10eW: tensor<672x112x1x1xf32>, %b10eg: tensor<672xf32>, %b10ebt: tensor<672xf32>, %b10dW: tensor<672x1x5x5xf32>, %b10dg: tensor<672xf32>, %b10dbt: tensor<672xf32>, %b10zW1: tensor<672x28xf32>, %b10zb1: tensor<28xf32>, %b10zW2: tensor<28x672xf32>, %b10zb2: tensor<672xf32>, %b10pW: tensor<112x672x1x1xf32>, %b10pg: tensor<112xf32>, %b10pbt: tensor<112xf32>, %b11eW: tensor<672x112x1x1xf32>, %b11eg: tensor<672xf32>, %b11ebt: tensor<672xf32>, %b11dW: tensor<672x1x5x5xf32>, %b11dg: tensor<672xf32>, %b11dbt: tensor<672xf32>, %b11zW1: tensor<672x28xf32>, %b11zb1: tensor<28xf32>, %b11zW2: tensor<28x672xf32>, %b11zb2: tensor<672xf32>, %b11pW: tensor<112x672x1x1xf32>, %b11pg: tensor<112xf32>, %b11pbt: tensor<112xf32>, %b12eW: tensor<672x112x1x1xf32>, %b12eg: tensor<672xf32>, %b12ebt: tensor<672xf32>, %b12dW: tensor<672x1x5x5xf32>, %b12dg: tensor<672xf32>, %b12dbt: tensor<672xf32>, %b12zW1: tensor<672x28xf32>, %b12zb1: tensor<28xf32>, %b12zW2: tensor<28x672xf32>, %b12zb2: tensor<672xf32>, %b12pW: tensor<192x672x1x1xf32>, %b12pg: tensor<192xf32>, %b12pbt: tensor<192xf32>, %b13eW: tensor<1152x192x1x1xf32>, %b13eg: tensor<1152xf32>, %b13ebt: tensor<1152xf32>, %b13dW: tensor<1152x1x5x5xf32>, %b13dg: tensor<1152xf32>, %b13dbt: tensor<1152xf32>, %b13zW1: tensor<1152x48xf32>, %b13zb1: tensor<48xf32>, %b13zW2: tensor<48x1152xf32>, %b13zb2: tensor<1152xf32>, %b13pW: tensor<192x1152x1x1xf32>, %b13pg: tensor<192xf32>, %b13pbt: tensor<192xf32>, %b14eW: tensor<1152x192x1x1xf32>, %b14eg: tensor<1152xf32>, %b14ebt: tensor<1152xf32>, %b14dW: tensor<1152x1x5x5xf32>, %b14dg: tensor<1152xf32>, %b14dbt: tensor<1152xf32>, %b14zW1: tensor<1152x48xf32>, %b14zb1: tensor<48xf32>, %b14zW2: tensor<48x1152xf32>, %b14zb2: tensor<1152xf32>, %b14pW: tensor<192x1152x1x1xf32>, %b14pg: tensor<192xf32>, %b14pbt: tensor<192xf32>, %b15eW: tensor<1152x192x1x1xf32>, %b15eg: tensor<1152xf32>, %b15ebt: tensor<1152xf32>, %b15dW: tensor<1152x1x5x5xf32>, %b15dg: tensor<1152xf32>, %b15dbt: tensor<1152xf32>, %b15zW1: tensor<1152x48xf32>, %b15zb1: tensor<48xf32>, %b15zW2: tensor<48x1152xf32>, %b15zb2: tensor<1152xf32>, %b15pW: tensor<192x1152x1x1xf32>, %b15pg: tensor<192xf32>, %b15pbt: tensor<192xf32>, %b16eW: tensor<1152x192x1x1xf32>, %b16eg: tensor<1152xf32>, %b16ebt: tensor<1152xf32>, %b16dW: tensor<1152x1x3x3xf32>, %b16dg: tensor<1152xf32>, %b16dbt: tensor<1152xf32>, %b16zW1: tensor<1152x48xf32>, %b16zb1: tensor<48xf32>, %b16zW2: tensor<48x1152xf32>, %b16zb2: tensor<1152xf32>, %b16pW: tensor<320x1152x1x1xf32>, %b16pg: tensor<320xf32>, %b16pbt: tensor<320xf32>, %hW: tensor<1280x320x1x1xf32>, %hg: tensor<1280xf32>, %hbt: tensor<1280xf32>, %Wd: tensor<1280x10xf32>, %bd: tensor<10xf32>, %do: tensor<32x1280xf32>) -> tensor<32x10xf32> {
    // ── EfficientNet-B0 forward: every line is pretty(verified AST node) ──
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
    %v1670 = stablehlo.multiply %do, %v1669 : tensor<32x1280xf32>
    %v1671 = stablehlo.dot_general %v1670, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1280xf32>, tensor<1280x10xf32>) -> tensor<32x10xf32>
    %v1672 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v1673 = stablehlo.add %v1671, %v1672 : tensor<32x10xf32>
    return %v1673 : tensor<32x10xf32>
  }
}
