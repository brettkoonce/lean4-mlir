module @m {
  func.func @efficientnet_drop_fwd(%x: tensor<32x150528xf32>, %sW: tensor<32x3x3x3xf32>, %sg: tensor<32xf32>, %sbt: tensor<32xf32>, %b1dW: tensor<32x1x3x3xf32>, %b1dg: tensor<32xf32>, %b1dbt: tensor<32xf32>, %b1zW1: tensor<32x8xf32>, %b1zb1: tensor<8xf32>, %b1zW2: tensor<8x32xf32>, %b1zb2: tensor<32xf32>, %b1pW: tensor<16x32x1x1xf32>, %b1pg: tensor<16xf32>, %b1pbt: tensor<16xf32>, %b2eW: tensor<96x16x1x1xf32>, %b2eg: tensor<96xf32>, %b2ebt: tensor<96xf32>, %b2dW: tensor<96x1x3x3xf32>, %b2dg: tensor<96xf32>, %b2dbt: tensor<96xf32>, %b2zW1: tensor<96x4xf32>, %b2zb1: tensor<4xf32>, %b2zW2: tensor<4x96xf32>, %b2zb2: tensor<96xf32>, %b2pW: tensor<24x96x1x1xf32>, %b2pg: tensor<24xf32>, %b2pbt: tensor<24xf32>, %b3eW: tensor<144x24x1x1xf32>, %b3eg: tensor<144xf32>, %b3ebt: tensor<144xf32>, %b3dW: tensor<144x1x3x3xf32>, %b3dg: tensor<144xf32>, %b3dbt: tensor<144xf32>, %b3zW1: tensor<144x6xf32>, %b3zb1: tensor<6xf32>, %b3zW2: tensor<6x144xf32>, %b3zb2: tensor<144xf32>, %b3pW: tensor<24x144x1x1xf32>, %b3pg: tensor<24xf32>, %b3pbt: tensor<24xf32>, %b4eW: tensor<144x24x1x1xf32>, %b4eg: tensor<144xf32>, %b4ebt: tensor<144xf32>, %b4dW: tensor<144x1x5x5xf32>, %b4dg: tensor<144xf32>, %b4dbt: tensor<144xf32>, %b4zW1: tensor<144x6xf32>, %b4zb1: tensor<6xf32>, %b4zW2: tensor<6x144xf32>, %b4zb2: tensor<144xf32>, %b4pW: tensor<40x144x1x1xf32>, %b4pg: tensor<40xf32>, %b4pbt: tensor<40xf32>, %b5eW: tensor<240x40x1x1xf32>, %b5eg: tensor<240xf32>, %b5ebt: tensor<240xf32>, %b5dW: tensor<240x1x5x5xf32>, %b5dg: tensor<240xf32>, %b5dbt: tensor<240xf32>, %b5zW1: tensor<240x10xf32>, %b5zb1: tensor<10xf32>, %b5zW2: tensor<10x240xf32>, %b5zb2: tensor<240xf32>, %b5pW: tensor<40x240x1x1xf32>, %b5pg: tensor<40xf32>, %b5pbt: tensor<40xf32>, %b6eW: tensor<240x40x1x1xf32>, %b6eg: tensor<240xf32>, %b6ebt: tensor<240xf32>, %b6dW: tensor<240x1x3x3xf32>, %b6dg: tensor<240xf32>, %b6dbt: tensor<240xf32>, %b6zW1: tensor<240x10xf32>, %b6zb1: tensor<10xf32>, %b6zW2: tensor<10x240xf32>, %b6zb2: tensor<240xf32>, %b6pW: tensor<80x240x1x1xf32>, %b6pg: tensor<80xf32>, %b6pbt: tensor<80xf32>, %b7eW: tensor<480x80x1x1xf32>, %b7eg: tensor<480xf32>, %b7ebt: tensor<480xf32>, %b7dW: tensor<480x1x3x3xf32>, %b7dg: tensor<480xf32>, %b7dbt: tensor<480xf32>, %b7zW1: tensor<480x20xf32>, %b7zb1: tensor<20xf32>, %b7zW2: tensor<20x480xf32>, %b7zb2: tensor<480xf32>, %b7pW: tensor<80x480x1x1xf32>, %b7pg: tensor<80xf32>, %b7pbt: tensor<80xf32>, %b8eW: tensor<480x80x1x1xf32>, %b8eg: tensor<480xf32>, %b8ebt: tensor<480xf32>, %b8dW: tensor<480x1x3x3xf32>, %b8dg: tensor<480xf32>, %b8dbt: tensor<480xf32>, %b8zW1: tensor<480x20xf32>, %b8zb1: tensor<20xf32>, %b8zW2: tensor<20x480xf32>, %b8zb2: tensor<480xf32>, %b8pW: tensor<80x480x1x1xf32>, %b8pg: tensor<80xf32>, %b8pbt: tensor<80xf32>, %b9eW: tensor<480x80x1x1xf32>, %b9eg: tensor<480xf32>, %b9ebt: tensor<480xf32>, %b9dW: tensor<480x1x5x5xf32>, %b9dg: tensor<480xf32>, %b9dbt: tensor<480xf32>, %b9zW1: tensor<480x20xf32>, %b9zb1: tensor<20xf32>, %b9zW2: tensor<20x480xf32>, %b9zb2: tensor<480xf32>, %b9pW: tensor<112x480x1x1xf32>, %b9pg: tensor<112xf32>, %b9pbt: tensor<112xf32>, %b10eW: tensor<672x112x1x1xf32>, %b10eg: tensor<672xf32>, %b10ebt: tensor<672xf32>, %b10dW: tensor<672x1x5x5xf32>, %b10dg: tensor<672xf32>, %b10dbt: tensor<672xf32>, %b10zW1: tensor<672x28xf32>, %b10zb1: tensor<28xf32>, %b10zW2: tensor<28x672xf32>, %b10zb2: tensor<672xf32>, %b10pW: tensor<112x672x1x1xf32>, %b10pg: tensor<112xf32>, %b10pbt: tensor<112xf32>, %b11eW: tensor<672x112x1x1xf32>, %b11eg: tensor<672xf32>, %b11ebt: tensor<672xf32>, %b11dW: tensor<672x1x5x5xf32>, %b11dg: tensor<672xf32>, %b11dbt: tensor<672xf32>, %b11zW1: tensor<672x28xf32>, %b11zb1: tensor<28xf32>, %b11zW2: tensor<28x672xf32>, %b11zb2: tensor<672xf32>, %b11pW: tensor<112x672x1x1xf32>, %b11pg: tensor<112xf32>, %b11pbt: tensor<112xf32>, %b12eW: tensor<672x112x1x1xf32>, %b12eg: tensor<672xf32>, %b12ebt: tensor<672xf32>, %b12dW: tensor<672x1x5x5xf32>, %b12dg: tensor<672xf32>, %b12dbt: tensor<672xf32>, %b12zW1: tensor<672x28xf32>, %b12zb1: tensor<28xf32>, %b12zW2: tensor<28x672xf32>, %b12zb2: tensor<672xf32>, %b12pW: tensor<192x672x1x1xf32>, %b12pg: tensor<192xf32>, %b12pbt: tensor<192xf32>, %b13eW: tensor<1152x192x1x1xf32>, %b13eg: tensor<1152xf32>, %b13ebt: tensor<1152xf32>, %b13dW: tensor<1152x1x5x5xf32>, %b13dg: tensor<1152xf32>, %b13dbt: tensor<1152xf32>, %b13zW1: tensor<1152x48xf32>, %b13zb1: tensor<48xf32>, %b13zW2: tensor<48x1152xf32>, %b13zb2: tensor<1152xf32>, %b13pW: tensor<192x1152x1x1xf32>, %b13pg: tensor<192xf32>, %b13pbt: tensor<192xf32>, %b14eW: tensor<1152x192x1x1xf32>, %b14eg: tensor<1152xf32>, %b14ebt: tensor<1152xf32>, %b14dW: tensor<1152x1x5x5xf32>, %b14dg: tensor<1152xf32>, %b14dbt: tensor<1152xf32>, %b14zW1: tensor<1152x48xf32>, %b14zb1: tensor<48xf32>, %b14zW2: tensor<48x1152xf32>, %b14zb2: tensor<1152xf32>, %b14pW: tensor<192x1152x1x1xf32>, %b14pg: tensor<192xf32>, %b14pbt: tensor<192xf32>, %b15eW: tensor<1152x192x1x1xf32>, %b15eg: tensor<1152xf32>, %b15ebt: tensor<1152xf32>, %b15dW: tensor<1152x1x5x5xf32>, %b15dg: tensor<1152xf32>, %b15dbt: tensor<1152xf32>, %b15zW1: tensor<1152x48xf32>, %b15zb1: tensor<48xf32>, %b15zW2: tensor<48x1152xf32>, %b15zb2: tensor<1152xf32>, %b15pW: tensor<192x1152x1x1xf32>, %b15pg: tensor<192xf32>, %b15pbt: tensor<192xf32>, %b16eW: tensor<1152x192x1x1xf32>, %b16eg: tensor<1152xf32>, %b16ebt: tensor<1152xf32>, %b16dW: tensor<1152x1x3x3xf32>, %b16dg: tensor<1152xf32>, %b16dbt: tensor<1152xf32>, %b16zW1: tensor<1152x48xf32>, %b16zb1: tensor<48xf32>, %b16zW2: tensor<48x1152xf32>, %b16zb2: tensor<1152xf32>, %b16pW: tensor<320x1152x1x1xf32>, %b16pg: tensor<320xf32>, %b16pbt: tensor<320xf32>, %hW: tensor<1280x320x1x1xf32>, %hg: tensor<1280xf32>, %hbt: tensor<1280xf32>, %Wd: tensor<1280x10xf32>, %bd: tensor<10xf32>, %dp2: tensor<32xf32>, %dp4: tensor<32xf32>, %dp6: tensor<32xf32>, %dp7: tensor<32xf32>, %dp9: tensor<32xf32>, %dp10: tensor<32xf32>, %dp12: tensor<32xf32>, %dp13: tensor<32xf32>, %dp14: tensor<32xf32>) -> tensor<32x10xf32> {
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
    %v301 = stablehlo.broadcast_in_dim %dp2, dims = [0] : (tensor<32xf32>) -> tensor<32x24x56x56xf32>
    %v302 = stablehlo.multiply %v301, %v300 : tensor<32x24x56x56xf32>
    %v303 = stablehlo.reshape %v302 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v304 = stablehlo.reshape %v303 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v305 = stablehlo.reshape %v199 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v306 = stablehlo.add %v304, %v305 : tensor<32x24x56x56xf32>
    %v307 = stablehlo.reshape %v306 : (tensor<32x24x56x56xf32>) -> tensor<32x75264xf32>
    %v308 = stablehlo.reshape %v307 : (tensor<32x75264xf32>) -> tensor<32x24x56x56xf32>
    %v309 = stablehlo.convolution(%v308, %b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x24x56x56xf32>, tensor<144x24x1x1xf32>) -> tensor<32x144x56x56xf32>
    %v310 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v311 = stablehlo.add %v309, %v310 : tensor<32x144x56x56xf32>
    %v312 = stablehlo.reshape %v311 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v313 = stablehlo.reshape %v312 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v314 = stablehlo.constant dense<0.0> : tensor<f32>
    %v315 = stablehlo.constant dense<100352.0> : tensor<32x144x56x56xf32>
    %v316 = stablehlo.constant dense<1.0e-5> : tensor<32x144x56x56xf32>
    %v317 = stablehlo.reduce(%v313 init: %v314) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v318 = stablehlo.broadcast_in_dim %v317, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v319 = stablehlo.divide %v318, %v315 : tensor<32x144x56x56xf32>
    %v320 = stablehlo.subtract %v313, %v319 : tensor<32x144x56x56xf32>
    %v321 = stablehlo.multiply %v320, %v320 : tensor<32x144x56x56xf32>
    %v322 = stablehlo.reduce(%v321 init: %v314) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x56x56xf32>, tensor<f32>) -> tensor<144xf32>
    %v323 = stablehlo.broadcast_in_dim %v322, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v324 = stablehlo.divide %v323, %v315 : tensor<32x144x56x56xf32>
    %v325 = stablehlo.add %v324, %v316 : tensor<32x144x56x56xf32>
    %v326 = stablehlo.rsqrt %v325 : tensor<32x144x56x56xf32>
    %v327 = stablehlo.multiply %v320, %v326 : tensor<32x144x56x56xf32>
    %v328 = stablehlo.broadcast_in_dim %b4eg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v329 = stablehlo.broadcast_in_dim %b4ebt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x56x56xf32>
    %v330 = stablehlo.multiply %v327, %v328 : tensor<32x144x56x56xf32>
    %v331 = stablehlo.add %v330, %v329 : tensor<32x144x56x56xf32>
    %v332 = stablehlo.reshape %v331 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v333 = stablehlo.reshape %v332 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v334 = stablehlo.logistic %v333 : tensor<32x144x56x56xf32>
    %v335 = stablehlo.multiply %v333, %v334 : tensor<32x144x56x56xf32>
    %v336 = stablehlo.reshape %v335 : (tensor<32x144x56x56xf32>) -> tensor<32x451584xf32>
    %v337 = stablehlo.reshape %v336 : (tensor<32x451584xf32>) -> tensor<32x144x56x56xf32>
    %v338 = stablehlo.convolution(%v337, %b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 144 : i64} : (tensor<32x144x56x56xf32>, tensor<144x1x5x5xf32>) -> tensor<32x144x28x28xf32>
    %v339 = stablehlo.broadcast_in_dim %zb144, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v340 = stablehlo.add %v338, %v339 : tensor<32x144x28x28xf32>
    %v341 = stablehlo.reshape %v340 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v342 = stablehlo.reshape %v341 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v343 = stablehlo.constant dense<0.0> : tensor<f32>
    %v344 = stablehlo.constant dense<25088.0> : tensor<32x144x28x28xf32>
    %v345 = stablehlo.constant dense<1.0e-5> : tensor<32x144x28x28xf32>
    %v346 = stablehlo.reduce(%v342 init: %v343) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v347 = stablehlo.broadcast_in_dim %v346, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v348 = stablehlo.divide %v347, %v344 : tensor<32x144x28x28xf32>
    %v349 = stablehlo.subtract %v342, %v348 : tensor<32x144x28x28xf32>
    %v350 = stablehlo.multiply %v349, %v349 : tensor<32x144x28x28xf32>
    %v351 = stablehlo.reduce(%v350 init: %v343) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<144xf32>
    %v352 = stablehlo.broadcast_in_dim %v351, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v353 = stablehlo.divide %v352, %v344 : tensor<32x144x28x28xf32>
    %v354 = stablehlo.add %v353, %v345 : tensor<32x144x28x28xf32>
    %v355 = stablehlo.rsqrt %v354 : tensor<32x144x28x28xf32>
    %v356 = stablehlo.multiply %v349, %v355 : tensor<32x144x28x28xf32>
    %v357 = stablehlo.broadcast_in_dim %b4dg, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v358 = stablehlo.broadcast_in_dim %b4dbt, dims = [1] : (tensor<144xf32>) -> tensor<32x144x28x28xf32>
    %v359 = stablehlo.multiply %v356, %v357 : tensor<32x144x28x28xf32>
    %v360 = stablehlo.add %v359, %v358 : tensor<32x144x28x28xf32>
    %v361 = stablehlo.reshape %v360 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v362 = stablehlo.reshape %v361 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v363 = stablehlo.logistic %v362 : tensor<32x144x28x28xf32>
    %v364 = stablehlo.multiply %v362, %v363 : tensor<32x144x28x28xf32>
    %v365 = stablehlo.reshape %v364 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v366 = stablehlo.reshape %v365 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v367 = stablehlo.constant dense<0.0> : tensor<f32>
    %v368 = stablehlo.reduce(%v366 init: %v367) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x144x28x28xf32>, tensor<f32>) -> tensor<32x144xf32>
    %v369 = stablehlo.constant dense<784.0> : tensor<32x144xf32>
    %v370 = stablehlo.divide %v368, %v369 : tensor<32x144xf32>
    %v371 = stablehlo.dot_general %v370, %b4zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x144xf32>, tensor<144x6xf32>) -> tensor<32x6xf32>
    %v372 = stablehlo.broadcast_in_dim %b4zb1, dims = [1] : (tensor<6xf32>) -> tensor<32x6xf32>
    %v373 = stablehlo.add %v371, %v372 : tensor<32x6xf32>
    %v374 = stablehlo.logistic %v373 : tensor<32x6xf32>
    %v375 = stablehlo.multiply %v373, %v374 : tensor<32x6xf32>
    %v376 = stablehlo.dot_general %v375, %b4zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x6xf32>, tensor<6x144xf32>) -> tensor<32x144xf32>
    %v377 = stablehlo.broadcast_in_dim %b4zb2, dims = [1] : (tensor<144xf32>) -> tensor<32x144xf32>
    %v378 = stablehlo.add %v376, %v377 : tensor<32x144xf32>
    %v379 = stablehlo.logistic %v378 : tensor<32x144xf32>
    %v380 = stablehlo.broadcast_in_dim %v379, dims = [0, 1] : (tensor<32x144xf32>) -> tensor<32x144x28x28xf32>
    %v381 = stablehlo.multiply %v366, %v380 : tensor<32x144x28x28xf32>
    %v382 = stablehlo.reshape %v381 : (tensor<32x144x28x28xf32>) -> tensor<32x112896xf32>
    %v383 = stablehlo.reshape %v382 : (tensor<32x112896xf32>) -> tensor<32x144x28x28xf32>
    %v384 = stablehlo.convolution(%v383, %b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x144x28x28xf32>, tensor<40x144x1x1xf32>) -> tensor<32x40x28x28xf32>
    %v385 = stablehlo.broadcast_in_dim %zb40, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v386 = stablehlo.add %v384, %v385 : tensor<32x40x28x28xf32>
    %v387 = stablehlo.reshape %v386 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v388 = stablehlo.reshape %v387 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v389 = stablehlo.constant dense<0.0> : tensor<f32>
    %v390 = stablehlo.constant dense<25088.0> : tensor<32x40x28x28xf32>
    %v391 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v392 = stablehlo.reduce(%v388 init: %v389) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v393 = stablehlo.broadcast_in_dim %v392, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v394 = stablehlo.divide %v393, %v390 : tensor<32x40x28x28xf32>
    %v395 = stablehlo.subtract %v388, %v394 : tensor<32x40x28x28xf32>
    %v396 = stablehlo.multiply %v395, %v395 : tensor<32x40x28x28xf32>
    %v397 = stablehlo.reduce(%v396 init: %v389) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v398 = stablehlo.broadcast_in_dim %v397, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v399 = stablehlo.divide %v398, %v390 : tensor<32x40x28x28xf32>
    %v400 = stablehlo.add %v399, %v391 : tensor<32x40x28x28xf32>
    %v401 = stablehlo.rsqrt %v400 : tensor<32x40x28x28xf32>
    %v402 = stablehlo.multiply %v395, %v401 : tensor<32x40x28x28xf32>
    %v403 = stablehlo.broadcast_in_dim %b4pg, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v404 = stablehlo.broadcast_in_dim %b4pbt, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v405 = stablehlo.multiply %v402, %v403 : tensor<32x40x28x28xf32>
    %v406 = stablehlo.add %v405, %v404 : tensor<32x40x28x28xf32>
    %v407 = stablehlo.reshape %v406 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v408 = stablehlo.reshape %v407 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v409 = stablehlo.convolution(%v408, %b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<32x240x28x28xf32>
    %v410 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v411 = stablehlo.add %v409, %v410 : tensor<32x240x28x28xf32>
    %v412 = stablehlo.reshape %v411 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v413 = stablehlo.reshape %v412 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v414 = stablehlo.constant dense<0.0> : tensor<f32>
    %v415 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v416 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v417 = stablehlo.reduce(%v413 init: %v414) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v418 = stablehlo.broadcast_in_dim %v417, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v419 = stablehlo.divide %v418, %v415 : tensor<32x240x28x28xf32>
    %v420 = stablehlo.subtract %v413, %v419 : tensor<32x240x28x28xf32>
    %v421 = stablehlo.multiply %v420, %v420 : tensor<32x240x28x28xf32>
    %v422 = stablehlo.reduce(%v421 init: %v414) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v423 = stablehlo.broadcast_in_dim %v422, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v424 = stablehlo.divide %v423, %v415 : tensor<32x240x28x28xf32>
    %v425 = stablehlo.add %v424, %v416 : tensor<32x240x28x28xf32>
    %v426 = stablehlo.rsqrt %v425 : tensor<32x240x28x28xf32>
    %v427 = stablehlo.multiply %v420, %v426 : tensor<32x240x28x28xf32>
    %v428 = stablehlo.broadcast_in_dim %b5eg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v429 = stablehlo.broadcast_in_dim %b5ebt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v430 = stablehlo.multiply %v427, %v428 : tensor<32x240x28x28xf32>
    %v431 = stablehlo.add %v430, %v429 : tensor<32x240x28x28xf32>
    %v432 = stablehlo.reshape %v431 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v433 = stablehlo.reshape %v432 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v434 = stablehlo.logistic %v433 : tensor<32x240x28x28xf32>
    %v435 = stablehlo.multiply %v433, %v434 : tensor<32x240x28x28xf32>
    %v436 = stablehlo.reshape %v435 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v437 = stablehlo.reshape %v436 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v438 = stablehlo.convolution(%v437, %b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x5x5xf32>) -> tensor<32x240x28x28xf32>
    %v439 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v440 = stablehlo.add %v438, %v439 : tensor<32x240x28x28xf32>
    %v441 = stablehlo.reshape %v440 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v442 = stablehlo.reshape %v441 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v443 = stablehlo.constant dense<0.0> : tensor<f32>
    %v444 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v445 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v446 = stablehlo.reduce(%v442 init: %v443) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v447 = stablehlo.broadcast_in_dim %v446, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v448 = stablehlo.divide %v447, %v444 : tensor<32x240x28x28xf32>
    %v449 = stablehlo.subtract %v442, %v448 : tensor<32x240x28x28xf32>
    %v450 = stablehlo.multiply %v449, %v449 : tensor<32x240x28x28xf32>
    %v451 = stablehlo.reduce(%v450 init: %v443) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v452 = stablehlo.broadcast_in_dim %v451, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v453 = stablehlo.divide %v452, %v444 : tensor<32x240x28x28xf32>
    %v454 = stablehlo.add %v453, %v445 : tensor<32x240x28x28xf32>
    %v455 = stablehlo.rsqrt %v454 : tensor<32x240x28x28xf32>
    %v456 = stablehlo.multiply %v449, %v455 : tensor<32x240x28x28xf32>
    %v457 = stablehlo.broadcast_in_dim %b5dg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v458 = stablehlo.broadcast_in_dim %b5dbt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v459 = stablehlo.multiply %v456, %v457 : tensor<32x240x28x28xf32>
    %v460 = stablehlo.add %v459, %v458 : tensor<32x240x28x28xf32>
    %v461 = stablehlo.reshape %v460 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v462 = stablehlo.reshape %v461 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v463 = stablehlo.logistic %v462 : tensor<32x240x28x28xf32>
    %v464 = stablehlo.multiply %v462, %v463 : tensor<32x240x28x28xf32>
    %v465 = stablehlo.reshape %v464 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v466 = stablehlo.reshape %v465 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v467 = stablehlo.constant dense<0.0> : tensor<f32>
    %v468 = stablehlo.reduce(%v466 init: %v467) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v469 = stablehlo.constant dense<784.0> : tensor<32x240xf32>
    %v470 = stablehlo.divide %v468, %v469 : tensor<32x240xf32>
    %v471 = stablehlo.dot_general %v470, %b5zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v472 = stablehlo.broadcast_in_dim %b5zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v473 = stablehlo.add %v471, %v472 : tensor<32x10xf32>
    %v474 = stablehlo.logistic %v473 : tensor<32x10xf32>
    %v475 = stablehlo.multiply %v473, %v474 : tensor<32x10xf32>
    %v476 = stablehlo.dot_general %v475, %b5zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v477 = stablehlo.broadcast_in_dim %b5zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v478 = stablehlo.add %v476, %v477 : tensor<32x240xf32>
    %v479 = stablehlo.logistic %v478 : tensor<32x240xf32>
    %v480 = stablehlo.broadcast_in_dim %v479, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x28x28xf32>
    %v481 = stablehlo.multiply %v466, %v480 : tensor<32x240x28x28xf32>
    %v482 = stablehlo.reshape %v481 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v483 = stablehlo.reshape %v482 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v484 = stablehlo.convolution(%v483, %b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x28x28xf32>, tensor<40x240x1x1xf32>) -> tensor<32x40x28x28xf32>
    %v485 = stablehlo.broadcast_in_dim %zb40, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v486 = stablehlo.add %v484, %v485 : tensor<32x40x28x28xf32>
    %v487 = stablehlo.reshape %v486 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v488 = stablehlo.reshape %v487 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v489 = stablehlo.constant dense<0.0> : tensor<f32>
    %v490 = stablehlo.constant dense<25088.0> : tensor<32x40x28x28xf32>
    %v491 = stablehlo.constant dense<1.0e-5> : tensor<32x40x28x28xf32>
    %v492 = stablehlo.reduce(%v488 init: %v489) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v493 = stablehlo.broadcast_in_dim %v492, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v494 = stablehlo.divide %v493, %v490 : tensor<32x40x28x28xf32>
    %v495 = stablehlo.subtract %v488, %v494 : tensor<32x40x28x28xf32>
    %v496 = stablehlo.multiply %v495, %v495 : tensor<32x40x28x28xf32>
    %v497 = stablehlo.reduce(%v496 init: %v489) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x40x28x28xf32>, tensor<f32>) -> tensor<40xf32>
    %v498 = stablehlo.broadcast_in_dim %v497, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v499 = stablehlo.divide %v498, %v490 : tensor<32x40x28x28xf32>
    %v500 = stablehlo.add %v499, %v491 : tensor<32x40x28x28xf32>
    %v501 = stablehlo.rsqrt %v500 : tensor<32x40x28x28xf32>
    %v502 = stablehlo.multiply %v495, %v501 : tensor<32x40x28x28xf32>
    %v503 = stablehlo.broadcast_in_dim %b5pg, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v504 = stablehlo.broadcast_in_dim %b5pbt, dims = [1] : (tensor<40xf32>) -> tensor<32x40x28x28xf32>
    %v505 = stablehlo.multiply %v502, %v503 : tensor<32x40x28x28xf32>
    %v506 = stablehlo.add %v505, %v504 : tensor<32x40x28x28xf32>
    %v507 = stablehlo.reshape %v506 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v508 = stablehlo.reshape %v507 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v509 = stablehlo.broadcast_in_dim %dp4, dims = [0] : (tensor<32xf32>) -> tensor<32x40x28x28xf32>
    %v510 = stablehlo.multiply %v509, %v508 : tensor<32x40x28x28xf32>
    %v511 = stablehlo.reshape %v510 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v512 = stablehlo.reshape %v511 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v513 = stablehlo.reshape %v407 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v514 = stablehlo.add %v512, %v513 : tensor<32x40x28x28xf32>
    %v515 = stablehlo.reshape %v514 : (tensor<32x40x28x28xf32>) -> tensor<32x31360xf32>
    %v516 = stablehlo.reshape %v515 : (tensor<32x31360xf32>) -> tensor<32x40x28x28xf32>
    %v517 = stablehlo.convolution(%v516, %b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x40x28x28xf32>, tensor<240x40x1x1xf32>) -> tensor<32x240x28x28xf32>
    %v518 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v519 = stablehlo.add %v517, %v518 : tensor<32x240x28x28xf32>
    %v520 = stablehlo.reshape %v519 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v521 = stablehlo.reshape %v520 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v522 = stablehlo.constant dense<0.0> : tensor<f32>
    %v523 = stablehlo.constant dense<25088.0> : tensor<32x240x28x28xf32>
    %v524 = stablehlo.constant dense<1.0e-5> : tensor<32x240x28x28xf32>
    %v525 = stablehlo.reduce(%v521 init: %v522) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v526 = stablehlo.broadcast_in_dim %v525, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v527 = stablehlo.divide %v526, %v523 : tensor<32x240x28x28xf32>
    %v528 = stablehlo.subtract %v521, %v527 : tensor<32x240x28x28xf32>
    %v529 = stablehlo.multiply %v528, %v528 : tensor<32x240x28x28xf32>
    %v530 = stablehlo.reduce(%v529 init: %v522) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x28x28xf32>, tensor<f32>) -> tensor<240xf32>
    %v531 = stablehlo.broadcast_in_dim %v530, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v532 = stablehlo.divide %v531, %v523 : tensor<32x240x28x28xf32>
    %v533 = stablehlo.add %v532, %v524 : tensor<32x240x28x28xf32>
    %v534 = stablehlo.rsqrt %v533 : tensor<32x240x28x28xf32>
    %v535 = stablehlo.multiply %v528, %v534 : tensor<32x240x28x28xf32>
    %v536 = stablehlo.broadcast_in_dim %b6eg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v537 = stablehlo.broadcast_in_dim %b6ebt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x28x28xf32>
    %v538 = stablehlo.multiply %v535, %v536 : tensor<32x240x28x28xf32>
    %v539 = stablehlo.add %v538, %v537 : tensor<32x240x28x28xf32>
    %v540 = stablehlo.reshape %v539 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v541 = stablehlo.reshape %v540 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v542 = stablehlo.logistic %v541 : tensor<32x240x28x28xf32>
    %v543 = stablehlo.multiply %v541, %v542 : tensor<32x240x28x28xf32>
    %v544 = stablehlo.reshape %v543 : (tensor<32x240x28x28xf32>) -> tensor<32x188160xf32>
    %v545 = stablehlo.reshape %v544 : (tensor<32x188160xf32>) -> tensor<32x240x28x28xf32>
    %v546 = stablehlo.convolution(%v545, %b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 240 : i64} : (tensor<32x240x28x28xf32>, tensor<240x1x3x3xf32>) -> tensor<32x240x14x14xf32>
    %v547 = stablehlo.broadcast_in_dim %zb240, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v548 = stablehlo.add %v546, %v547 : tensor<32x240x14x14xf32>
    %v549 = stablehlo.reshape %v548 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v550 = stablehlo.reshape %v549 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v551 = stablehlo.constant dense<0.0> : tensor<f32>
    %v552 = stablehlo.constant dense<6272.0> : tensor<32x240x14x14xf32>
    %v553 = stablehlo.constant dense<1.0e-5> : tensor<32x240x14x14xf32>
    %v554 = stablehlo.reduce(%v550 init: %v551) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v555 = stablehlo.broadcast_in_dim %v554, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v556 = stablehlo.divide %v555, %v552 : tensor<32x240x14x14xf32>
    %v557 = stablehlo.subtract %v550, %v556 : tensor<32x240x14x14xf32>
    %v558 = stablehlo.multiply %v557, %v557 : tensor<32x240x14x14xf32>
    %v559 = stablehlo.reduce(%v558 init: %v551) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<240xf32>
    %v560 = stablehlo.broadcast_in_dim %v559, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v561 = stablehlo.divide %v560, %v552 : tensor<32x240x14x14xf32>
    %v562 = stablehlo.add %v561, %v553 : tensor<32x240x14x14xf32>
    %v563 = stablehlo.rsqrt %v562 : tensor<32x240x14x14xf32>
    %v564 = stablehlo.multiply %v557, %v563 : tensor<32x240x14x14xf32>
    %v565 = stablehlo.broadcast_in_dim %b6dg, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v566 = stablehlo.broadcast_in_dim %b6dbt, dims = [1] : (tensor<240xf32>) -> tensor<32x240x14x14xf32>
    %v567 = stablehlo.multiply %v564, %v565 : tensor<32x240x14x14xf32>
    %v568 = stablehlo.add %v567, %v566 : tensor<32x240x14x14xf32>
    %v569 = stablehlo.reshape %v568 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v570 = stablehlo.reshape %v569 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v571 = stablehlo.logistic %v570 : tensor<32x240x14x14xf32>
    %v572 = stablehlo.multiply %v570, %v571 : tensor<32x240x14x14xf32>
    %v573 = stablehlo.reshape %v572 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v574 = stablehlo.reshape %v573 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v575 = stablehlo.constant dense<0.0> : tensor<f32>
    %v576 = stablehlo.reduce(%v574 init: %v575) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x240x14x14xf32>, tensor<f32>) -> tensor<32x240xf32>
    %v577 = stablehlo.constant dense<196.0> : tensor<32x240xf32>
    %v578 = stablehlo.divide %v576, %v577 : tensor<32x240xf32>
    %v579 = stablehlo.dot_general %v578, %b6zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x240xf32>, tensor<240x10xf32>) -> tensor<32x10xf32>
    %v580 = stablehlo.broadcast_in_dim %b6zb1, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v581 = stablehlo.add %v579, %v580 : tensor<32x10xf32>
    %v582 = stablehlo.logistic %v581 : tensor<32x10xf32>
    %v583 = stablehlo.multiply %v581, %v582 : tensor<32x10xf32>
    %v584 = stablehlo.dot_general %v583, %b6zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<10x240xf32>) -> tensor<32x240xf32>
    %v585 = stablehlo.broadcast_in_dim %b6zb2, dims = [1] : (tensor<240xf32>) -> tensor<32x240xf32>
    %v586 = stablehlo.add %v584, %v585 : tensor<32x240xf32>
    %v587 = stablehlo.logistic %v586 : tensor<32x240xf32>
    %v588 = stablehlo.broadcast_in_dim %v587, dims = [0, 1] : (tensor<32x240xf32>) -> tensor<32x240x14x14xf32>
    %v589 = stablehlo.multiply %v574, %v588 : tensor<32x240x14x14xf32>
    %v590 = stablehlo.reshape %v589 : (tensor<32x240x14x14xf32>) -> tensor<32x47040xf32>
    %v591 = stablehlo.reshape %v590 : (tensor<32x47040xf32>) -> tensor<32x240x14x14xf32>
    %v592 = stablehlo.convolution(%v591, %b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x240x14x14xf32>, tensor<80x240x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v593 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v594 = stablehlo.add %v592, %v593 : tensor<32x80x14x14xf32>
    %v595 = stablehlo.reshape %v594 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v596 = stablehlo.reshape %v595 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v597 = stablehlo.constant dense<0.0> : tensor<f32>
    %v598 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v599 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v600 = stablehlo.reduce(%v596 init: %v597) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v601 = stablehlo.broadcast_in_dim %v600, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v602 = stablehlo.divide %v601, %v598 : tensor<32x80x14x14xf32>
    %v603 = stablehlo.subtract %v596, %v602 : tensor<32x80x14x14xf32>
    %v604 = stablehlo.multiply %v603, %v603 : tensor<32x80x14x14xf32>
    %v605 = stablehlo.reduce(%v604 init: %v597) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v606 = stablehlo.broadcast_in_dim %v605, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v607 = stablehlo.divide %v606, %v598 : tensor<32x80x14x14xf32>
    %v608 = stablehlo.add %v607, %v599 : tensor<32x80x14x14xf32>
    %v609 = stablehlo.rsqrt %v608 : tensor<32x80x14x14xf32>
    %v610 = stablehlo.multiply %v603, %v609 : tensor<32x80x14x14xf32>
    %v611 = stablehlo.broadcast_in_dim %b6pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v612 = stablehlo.broadcast_in_dim %b6pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v613 = stablehlo.multiply %v610, %v611 : tensor<32x80x14x14xf32>
    %v614 = stablehlo.add %v613, %v612 : tensor<32x80x14x14xf32>
    %v615 = stablehlo.reshape %v614 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v616 = stablehlo.reshape %v615 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v617 = stablehlo.convolution(%v616, %b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v618 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v619 = stablehlo.add %v617, %v618 : tensor<32x480x14x14xf32>
    %v620 = stablehlo.reshape %v619 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v621 = stablehlo.reshape %v620 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v622 = stablehlo.constant dense<0.0> : tensor<f32>
    %v623 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v624 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v625 = stablehlo.reduce(%v621 init: %v622) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v626 = stablehlo.broadcast_in_dim %v625, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v627 = stablehlo.divide %v626, %v623 : tensor<32x480x14x14xf32>
    %v628 = stablehlo.subtract %v621, %v627 : tensor<32x480x14x14xf32>
    %v629 = stablehlo.multiply %v628, %v628 : tensor<32x480x14x14xf32>
    %v630 = stablehlo.reduce(%v629 init: %v622) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v631 = stablehlo.broadcast_in_dim %v630, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v632 = stablehlo.divide %v631, %v623 : tensor<32x480x14x14xf32>
    %v633 = stablehlo.add %v632, %v624 : tensor<32x480x14x14xf32>
    %v634 = stablehlo.rsqrt %v633 : tensor<32x480x14x14xf32>
    %v635 = stablehlo.multiply %v628, %v634 : tensor<32x480x14x14xf32>
    %v636 = stablehlo.broadcast_in_dim %b7eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v637 = stablehlo.broadcast_in_dim %b7ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v638 = stablehlo.multiply %v635, %v636 : tensor<32x480x14x14xf32>
    %v639 = stablehlo.add %v638, %v637 : tensor<32x480x14x14xf32>
    %v640 = stablehlo.reshape %v639 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v641 = stablehlo.reshape %v640 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v642 = stablehlo.logistic %v641 : tensor<32x480x14x14xf32>
    %v643 = stablehlo.multiply %v641, %v642 : tensor<32x480x14x14xf32>
    %v644 = stablehlo.reshape %v643 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v645 = stablehlo.reshape %v644 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v646 = stablehlo.convolution(%v645, %b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %v647 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v648 = stablehlo.add %v646, %v647 : tensor<32x480x14x14xf32>
    %v649 = stablehlo.reshape %v648 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v650 = stablehlo.reshape %v649 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v651 = stablehlo.constant dense<0.0> : tensor<f32>
    %v652 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v653 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v654 = stablehlo.reduce(%v650 init: %v651) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v655 = stablehlo.broadcast_in_dim %v654, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v656 = stablehlo.divide %v655, %v652 : tensor<32x480x14x14xf32>
    %v657 = stablehlo.subtract %v650, %v656 : tensor<32x480x14x14xf32>
    %v658 = stablehlo.multiply %v657, %v657 : tensor<32x480x14x14xf32>
    %v659 = stablehlo.reduce(%v658 init: %v651) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v660 = stablehlo.broadcast_in_dim %v659, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v661 = stablehlo.divide %v660, %v652 : tensor<32x480x14x14xf32>
    %v662 = stablehlo.add %v661, %v653 : tensor<32x480x14x14xf32>
    %v663 = stablehlo.rsqrt %v662 : tensor<32x480x14x14xf32>
    %v664 = stablehlo.multiply %v657, %v663 : tensor<32x480x14x14xf32>
    %v665 = stablehlo.broadcast_in_dim %b7dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v666 = stablehlo.broadcast_in_dim %b7dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v667 = stablehlo.multiply %v664, %v665 : tensor<32x480x14x14xf32>
    %v668 = stablehlo.add %v667, %v666 : tensor<32x480x14x14xf32>
    %v669 = stablehlo.reshape %v668 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v670 = stablehlo.reshape %v669 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v671 = stablehlo.logistic %v670 : tensor<32x480x14x14xf32>
    %v672 = stablehlo.multiply %v670, %v671 : tensor<32x480x14x14xf32>
    %v673 = stablehlo.reshape %v672 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v674 = stablehlo.reshape %v673 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v675 = stablehlo.constant dense<0.0> : tensor<f32>
    %v676 = stablehlo.reduce(%v674 init: %v675) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v677 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v678 = stablehlo.divide %v676, %v677 : tensor<32x480xf32>
    %v679 = stablehlo.dot_general %v678, %b7zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v680 = stablehlo.broadcast_in_dim %b7zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v681 = stablehlo.add %v679, %v680 : tensor<32x20xf32>
    %v682 = stablehlo.logistic %v681 : tensor<32x20xf32>
    %v683 = stablehlo.multiply %v681, %v682 : tensor<32x20xf32>
    %v684 = stablehlo.dot_general %v683, %b7zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v685 = stablehlo.broadcast_in_dim %b7zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v686 = stablehlo.add %v684, %v685 : tensor<32x480xf32>
    %v687 = stablehlo.logistic %v686 : tensor<32x480xf32>
    %v688 = stablehlo.broadcast_in_dim %v687, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v689 = stablehlo.multiply %v674, %v688 : tensor<32x480x14x14xf32>
    %v690 = stablehlo.reshape %v689 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v691 = stablehlo.reshape %v690 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v692 = stablehlo.convolution(%v691, %b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v693 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v694 = stablehlo.add %v692, %v693 : tensor<32x80x14x14xf32>
    %v695 = stablehlo.reshape %v694 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v696 = stablehlo.reshape %v695 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v697 = stablehlo.constant dense<0.0> : tensor<f32>
    %v698 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v699 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v700 = stablehlo.reduce(%v696 init: %v697) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v701 = stablehlo.broadcast_in_dim %v700, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v702 = stablehlo.divide %v701, %v698 : tensor<32x80x14x14xf32>
    %v703 = stablehlo.subtract %v696, %v702 : tensor<32x80x14x14xf32>
    %v704 = stablehlo.multiply %v703, %v703 : tensor<32x80x14x14xf32>
    %v705 = stablehlo.reduce(%v704 init: %v697) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v706 = stablehlo.broadcast_in_dim %v705, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v707 = stablehlo.divide %v706, %v698 : tensor<32x80x14x14xf32>
    %v708 = stablehlo.add %v707, %v699 : tensor<32x80x14x14xf32>
    %v709 = stablehlo.rsqrt %v708 : tensor<32x80x14x14xf32>
    %v710 = stablehlo.multiply %v703, %v709 : tensor<32x80x14x14xf32>
    %v711 = stablehlo.broadcast_in_dim %b7pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v712 = stablehlo.broadcast_in_dim %b7pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v713 = stablehlo.multiply %v710, %v711 : tensor<32x80x14x14xf32>
    %v714 = stablehlo.add %v713, %v712 : tensor<32x80x14x14xf32>
    %v715 = stablehlo.reshape %v714 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v716 = stablehlo.reshape %v715 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v717 = stablehlo.broadcast_in_dim %dp6, dims = [0] : (tensor<32xf32>) -> tensor<32x80x14x14xf32>
    %v718 = stablehlo.multiply %v717, %v716 : tensor<32x80x14x14xf32>
    %v719 = stablehlo.reshape %v718 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v720 = stablehlo.reshape %v719 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v721 = stablehlo.reshape %v615 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v722 = stablehlo.add %v720, %v721 : tensor<32x80x14x14xf32>
    %v723 = stablehlo.reshape %v722 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v724 = stablehlo.reshape %v723 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v725 = stablehlo.convolution(%v724, %b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
    %v726 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v727 = stablehlo.add %v725, %v726 : tensor<32x480x14x14xf32>
    %v728 = stablehlo.reshape %v727 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v729 = stablehlo.reshape %v728 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v730 = stablehlo.constant dense<0.0> : tensor<f32>
    %v731 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v732 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v733 = stablehlo.reduce(%v729 init: %v730) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v734 = stablehlo.broadcast_in_dim %v733, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v735 = stablehlo.divide %v734, %v731 : tensor<32x480x14x14xf32>
    %v736 = stablehlo.subtract %v729, %v735 : tensor<32x480x14x14xf32>
    %v737 = stablehlo.multiply %v736, %v736 : tensor<32x480x14x14xf32>
    %v738 = stablehlo.reduce(%v737 init: %v730) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v739 = stablehlo.broadcast_in_dim %v738, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v740 = stablehlo.divide %v739, %v731 : tensor<32x480x14x14xf32>
    %v741 = stablehlo.add %v740, %v732 : tensor<32x480x14x14xf32>
    %v742 = stablehlo.rsqrt %v741 : tensor<32x480x14x14xf32>
    %v743 = stablehlo.multiply %v736, %v742 : tensor<32x480x14x14xf32>
    %v744 = stablehlo.broadcast_in_dim %b8eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v745 = stablehlo.broadcast_in_dim %b8ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v746 = stablehlo.multiply %v743, %v744 : tensor<32x480x14x14xf32>
    %v747 = stablehlo.add %v746, %v745 : tensor<32x480x14x14xf32>
    %v748 = stablehlo.reshape %v747 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v749 = stablehlo.reshape %v748 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v750 = stablehlo.logistic %v749 : tensor<32x480x14x14xf32>
    %v751 = stablehlo.multiply %v749, %v750 : tensor<32x480x14x14xf32>
    %v752 = stablehlo.reshape %v751 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v753 = stablehlo.reshape %v752 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v754 = stablehlo.convolution(%v753, %b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x3x3xf32>) -> tensor<32x480x14x14xf32>
    %v755 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v756 = stablehlo.add %v754, %v755 : tensor<32x480x14x14xf32>
    %v757 = stablehlo.reshape %v756 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v758 = stablehlo.reshape %v757 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v759 = stablehlo.constant dense<0.0> : tensor<f32>
    %v760 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v761 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v762 = stablehlo.reduce(%v758 init: %v759) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v763 = stablehlo.broadcast_in_dim %v762, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v764 = stablehlo.divide %v763, %v760 : tensor<32x480x14x14xf32>
    %v765 = stablehlo.subtract %v758, %v764 : tensor<32x480x14x14xf32>
    %v766 = stablehlo.multiply %v765, %v765 : tensor<32x480x14x14xf32>
    %v767 = stablehlo.reduce(%v766 init: %v759) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v768 = stablehlo.broadcast_in_dim %v767, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v769 = stablehlo.divide %v768, %v760 : tensor<32x480x14x14xf32>
    %v770 = stablehlo.add %v769, %v761 : tensor<32x480x14x14xf32>
    %v771 = stablehlo.rsqrt %v770 : tensor<32x480x14x14xf32>
    %v772 = stablehlo.multiply %v765, %v771 : tensor<32x480x14x14xf32>
    %v773 = stablehlo.broadcast_in_dim %b8dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v774 = stablehlo.broadcast_in_dim %b8dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v775 = stablehlo.multiply %v772, %v773 : tensor<32x480x14x14xf32>
    %v776 = stablehlo.add %v775, %v774 : tensor<32x480x14x14xf32>
    %v777 = stablehlo.reshape %v776 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v778 = stablehlo.reshape %v777 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v779 = stablehlo.logistic %v778 : tensor<32x480x14x14xf32>
    %v780 = stablehlo.multiply %v778, %v779 : tensor<32x480x14x14xf32>
    %v781 = stablehlo.reshape %v780 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v782 = stablehlo.reshape %v781 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v783 = stablehlo.constant dense<0.0> : tensor<f32>
    %v784 = stablehlo.reduce(%v782 init: %v783) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v785 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v786 = stablehlo.divide %v784, %v785 : tensor<32x480xf32>
    %v787 = stablehlo.dot_general %v786, %b8zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v788 = stablehlo.broadcast_in_dim %b8zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v789 = stablehlo.add %v787, %v788 : tensor<32x20xf32>
    %v790 = stablehlo.logistic %v789 : tensor<32x20xf32>
    %v791 = stablehlo.multiply %v789, %v790 : tensor<32x20xf32>
    %v792 = stablehlo.dot_general %v791, %b8zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v793 = stablehlo.broadcast_in_dim %b8zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v794 = stablehlo.add %v792, %v793 : tensor<32x480xf32>
    %v795 = stablehlo.logistic %v794 : tensor<32x480xf32>
    %v796 = stablehlo.broadcast_in_dim %v795, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v797 = stablehlo.multiply %v782, %v796 : tensor<32x480x14x14xf32>
    %v798 = stablehlo.reshape %v797 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v799 = stablehlo.reshape %v798 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v800 = stablehlo.convolution(%v799, %b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<80x480x1x1xf32>) -> tensor<32x80x14x14xf32>
    %v801 = stablehlo.broadcast_in_dim %zb80, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v802 = stablehlo.add %v800, %v801 : tensor<32x80x14x14xf32>
    %v803 = stablehlo.reshape %v802 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v804 = stablehlo.reshape %v803 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v805 = stablehlo.constant dense<0.0> : tensor<f32>
    %v806 = stablehlo.constant dense<6272.0> : tensor<32x80x14x14xf32>
    %v807 = stablehlo.constant dense<1.0e-5> : tensor<32x80x14x14xf32>
    %v808 = stablehlo.reduce(%v804 init: %v805) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v809 = stablehlo.broadcast_in_dim %v808, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v810 = stablehlo.divide %v809, %v806 : tensor<32x80x14x14xf32>
    %v811 = stablehlo.subtract %v804, %v810 : tensor<32x80x14x14xf32>
    %v812 = stablehlo.multiply %v811, %v811 : tensor<32x80x14x14xf32>
    %v813 = stablehlo.reduce(%v812 init: %v805) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x80x14x14xf32>, tensor<f32>) -> tensor<80xf32>
    %v814 = stablehlo.broadcast_in_dim %v813, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v815 = stablehlo.divide %v814, %v806 : tensor<32x80x14x14xf32>
    %v816 = stablehlo.add %v815, %v807 : tensor<32x80x14x14xf32>
    %v817 = stablehlo.rsqrt %v816 : tensor<32x80x14x14xf32>
    %v818 = stablehlo.multiply %v811, %v817 : tensor<32x80x14x14xf32>
    %v819 = stablehlo.broadcast_in_dim %b8pg, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v820 = stablehlo.broadcast_in_dim %b8pbt, dims = [1] : (tensor<80xf32>) -> tensor<32x80x14x14xf32>
    %v821 = stablehlo.multiply %v818, %v819 : tensor<32x80x14x14xf32>
    %v822 = stablehlo.add %v821, %v820 : tensor<32x80x14x14xf32>
    %v823 = stablehlo.reshape %v822 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v824 = stablehlo.reshape %v823 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v825 = stablehlo.broadcast_in_dim %dp7, dims = [0] : (tensor<32xf32>) -> tensor<32x80x14x14xf32>
    %v826 = stablehlo.multiply %v825, %v824 : tensor<32x80x14x14xf32>
    %v827 = stablehlo.reshape %v826 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v828 = stablehlo.reshape %v827 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v829 = stablehlo.reshape %v723 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v830 = stablehlo.add %v828, %v829 : tensor<32x80x14x14xf32>
    %v831 = stablehlo.reshape %v830 : (tensor<32x80x14x14xf32>) -> tensor<32x15680xf32>
    %v832 = stablehlo.reshape %v831 : (tensor<32x15680xf32>) -> tensor<32x80x14x14xf32>
    %v833 = stablehlo.convolution(%v832, %b9eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x80x14x14xf32>, tensor<480x80x1x1xf32>) -> tensor<32x480x14x14xf32>
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
    %v852 = stablehlo.broadcast_in_dim %b9eg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v853 = stablehlo.broadcast_in_dim %b9ebt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v854 = stablehlo.multiply %v851, %v852 : tensor<32x480x14x14xf32>
    %v855 = stablehlo.add %v854, %v853 : tensor<32x480x14x14xf32>
    %v856 = stablehlo.reshape %v855 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v857 = stablehlo.reshape %v856 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v858 = stablehlo.logistic %v857 : tensor<32x480x14x14xf32>
    %v859 = stablehlo.multiply %v857, %v858 : tensor<32x480x14x14xf32>
    %v860 = stablehlo.reshape %v859 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v861 = stablehlo.reshape %v860 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v862 = stablehlo.convolution(%v861, %b9dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 480 : i64} : (tensor<32x480x14x14xf32>, tensor<480x1x5x5xf32>) -> tensor<32x480x14x14xf32>
    %v863 = stablehlo.broadcast_in_dim %zb480, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v864 = stablehlo.add %v862, %v863 : tensor<32x480x14x14xf32>
    %v865 = stablehlo.reshape %v864 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v866 = stablehlo.reshape %v865 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v867 = stablehlo.constant dense<0.0> : tensor<f32>
    %v868 = stablehlo.constant dense<6272.0> : tensor<32x480x14x14xf32>
    %v869 = stablehlo.constant dense<1.0e-5> : tensor<32x480x14x14xf32>
    %v870 = stablehlo.reduce(%v866 init: %v867) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v871 = stablehlo.broadcast_in_dim %v870, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v872 = stablehlo.divide %v871, %v868 : tensor<32x480x14x14xf32>
    %v873 = stablehlo.subtract %v866, %v872 : tensor<32x480x14x14xf32>
    %v874 = stablehlo.multiply %v873, %v873 : tensor<32x480x14x14xf32>
    %v875 = stablehlo.reduce(%v874 init: %v867) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<480xf32>
    %v876 = stablehlo.broadcast_in_dim %v875, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v877 = stablehlo.divide %v876, %v868 : tensor<32x480x14x14xf32>
    %v878 = stablehlo.add %v877, %v869 : tensor<32x480x14x14xf32>
    %v879 = stablehlo.rsqrt %v878 : tensor<32x480x14x14xf32>
    %v880 = stablehlo.multiply %v873, %v879 : tensor<32x480x14x14xf32>
    %v881 = stablehlo.broadcast_in_dim %b9dg, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v882 = stablehlo.broadcast_in_dim %b9dbt, dims = [1] : (tensor<480xf32>) -> tensor<32x480x14x14xf32>
    %v883 = stablehlo.multiply %v880, %v881 : tensor<32x480x14x14xf32>
    %v884 = stablehlo.add %v883, %v882 : tensor<32x480x14x14xf32>
    %v885 = stablehlo.reshape %v884 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v886 = stablehlo.reshape %v885 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v887 = stablehlo.logistic %v886 : tensor<32x480x14x14xf32>
    %v888 = stablehlo.multiply %v886, %v887 : tensor<32x480x14x14xf32>
    %v889 = stablehlo.reshape %v888 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v890 = stablehlo.reshape %v889 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v891 = stablehlo.constant dense<0.0> : tensor<f32>
    %v892 = stablehlo.reduce(%v890 init: %v891) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x480x14x14xf32>, tensor<f32>) -> tensor<32x480xf32>
    %v893 = stablehlo.constant dense<196.0> : tensor<32x480xf32>
    %v894 = stablehlo.divide %v892, %v893 : tensor<32x480xf32>
    %v895 = stablehlo.dot_general %v894, %b9zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x480xf32>, tensor<480x20xf32>) -> tensor<32x20xf32>
    %v896 = stablehlo.broadcast_in_dim %b9zb1, dims = [1] : (tensor<20xf32>) -> tensor<32x20xf32>
    %v897 = stablehlo.add %v895, %v896 : tensor<32x20xf32>
    %v898 = stablehlo.logistic %v897 : tensor<32x20xf32>
    %v899 = stablehlo.multiply %v897, %v898 : tensor<32x20xf32>
    %v900 = stablehlo.dot_general %v899, %b9zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x20xf32>, tensor<20x480xf32>) -> tensor<32x480xf32>
    %v901 = stablehlo.broadcast_in_dim %b9zb2, dims = [1] : (tensor<480xf32>) -> tensor<32x480xf32>
    %v902 = stablehlo.add %v900, %v901 : tensor<32x480xf32>
    %v903 = stablehlo.logistic %v902 : tensor<32x480xf32>
    %v904 = stablehlo.broadcast_in_dim %v903, dims = [0, 1] : (tensor<32x480xf32>) -> tensor<32x480x14x14xf32>
    %v905 = stablehlo.multiply %v890, %v904 : tensor<32x480x14x14xf32>
    %v906 = stablehlo.reshape %v905 : (tensor<32x480x14x14xf32>) -> tensor<32x94080xf32>
    %v907 = stablehlo.reshape %v906 : (tensor<32x94080xf32>) -> tensor<32x480x14x14xf32>
    %v908 = stablehlo.convolution(%v907, %b9pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x480x14x14xf32>, tensor<112x480x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v909 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v910 = stablehlo.add %v908, %v909 : tensor<32x112x14x14xf32>
    %v911 = stablehlo.reshape %v910 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v912 = stablehlo.reshape %v911 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v913 = stablehlo.constant dense<0.0> : tensor<f32>
    %v914 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v915 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v916 = stablehlo.reduce(%v912 init: %v913) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v917 = stablehlo.broadcast_in_dim %v916, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v918 = stablehlo.divide %v917, %v914 : tensor<32x112x14x14xf32>
    %v919 = stablehlo.subtract %v912, %v918 : tensor<32x112x14x14xf32>
    %v920 = stablehlo.multiply %v919, %v919 : tensor<32x112x14x14xf32>
    %v921 = stablehlo.reduce(%v920 init: %v913) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v922 = stablehlo.broadcast_in_dim %v921, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v923 = stablehlo.divide %v922, %v914 : tensor<32x112x14x14xf32>
    %v924 = stablehlo.add %v923, %v915 : tensor<32x112x14x14xf32>
    %v925 = stablehlo.rsqrt %v924 : tensor<32x112x14x14xf32>
    %v926 = stablehlo.multiply %v919, %v925 : tensor<32x112x14x14xf32>
    %v927 = stablehlo.broadcast_in_dim %b9pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v928 = stablehlo.broadcast_in_dim %b9pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v929 = stablehlo.multiply %v926, %v927 : tensor<32x112x14x14xf32>
    %v930 = stablehlo.add %v929, %v928 : tensor<32x112x14x14xf32>
    %v931 = stablehlo.reshape %v930 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v932 = stablehlo.reshape %v931 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v933 = stablehlo.convolution(%v932, %b10eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v934 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v935 = stablehlo.add %v933, %v934 : tensor<32x672x14x14xf32>
    %v936 = stablehlo.reshape %v935 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v937 = stablehlo.reshape %v936 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v938 = stablehlo.constant dense<0.0> : tensor<f32>
    %v939 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v940 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v941 = stablehlo.reduce(%v937 init: %v938) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v942 = stablehlo.broadcast_in_dim %v941, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v943 = stablehlo.divide %v942, %v939 : tensor<32x672x14x14xf32>
    %v944 = stablehlo.subtract %v937, %v943 : tensor<32x672x14x14xf32>
    %v945 = stablehlo.multiply %v944, %v944 : tensor<32x672x14x14xf32>
    %v946 = stablehlo.reduce(%v945 init: %v938) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v947 = stablehlo.broadcast_in_dim %v946, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v948 = stablehlo.divide %v947, %v939 : tensor<32x672x14x14xf32>
    %v949 = stablehlo.add %v948, %v940 : tensor<32x672x14x14xf32>
    %v950 = stablehlo.rsqrt %v949 : tensor<32x672x14x14xf32>
    %v951 = stablehlo.multiply %v944, %v950 : tensor<32x672x14x14xf32>
    %v952 = stablehlo.broadcast_in_dim %b10eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v953 = stablehlo.broadcast_in_dim %b10ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v954 = stablehlo.multiply %v951, %v952 : tensor<32x672x14x14xf32>
    %v955 = stablehlo.add %v954, %v953 : tensor<32x672x14x14xf32>
    %v956 = stablehlo.reshape %v955 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v957 = stablehlo.reshape %v956 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v958 = stablehlo.logistic %v957 : tensor<32x672x14x14xf32>
    %v959 = stablehlo.multiply %v957, %v958 : tensor<32x672x14x14xf32>
    %v960 = stablehlo.reshape %v959 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v961 = stablehlo.reshape %v960 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v962 = stablehlo.convolution(%v961, %b10dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %v963 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v964 = stablehlo.add %v962, %v963 : tensor<32x672x14x14xf32>
    %v965 = stablehlo.reshape %v964 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v966 = stablehlo.reshape %v965 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v967 = stablehlo.constant dense<0.0> : tensor<f32>
    %v968 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v969 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v970 = stablehlo.reduce(%v966 init: %v967) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v971 = stablehlo.broadcast_in_dim %v970, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v972 = stablehlo.divide %v971, %v968 : tensor<32x672x14x14xf32>
    %v973 = stablehlo.subtract %v966, %v972 : tensor<32x672x14x14xf32>
    %v974 = stablehlo.multiply %v973, %v973 : tensor<32x672x14x14xf32>
    %v975 = stablehlo.reduce(%v974 init: %v967) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v976 = stablehlo.broadcast_in_dim %v975, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v977 = stablehlo.divide %v976, %v968 : tensor<32x672x14x14xf32>
    %v978 = stablehlo.add %v977, %v969 : tensor<32x672x14x14xf32>
    %v979 = stablehlo.rsqrt %v978 : tensor<32x672x14x14xf32>
    %v980 = stablehlo.multiply %v973, %v979 : tensor<32x672x14x14xf32>
    %v981 = stablehlo.broadcast_in_dim %b10dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v982 = stablehlo.broadcast_in_dim %b10dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v983 = stablehlo.multiply %v980, %v981 : tensor<32x672x14x14xf32>
    %v984 = stablehlo.add %v983, %v982 : tensor<32x672x14x14xf32>
    %v985 = stablehlo.reshape %v984 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v986 = stablehlo.reshape %v985 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v987 = stablehlo.logistic %v986 : tensor<32x672x14x14xf32>
    %v988 = stablehlo.multiply %v986, %v987 : tensor<32x672x14x14xf32>
    %v989 = stablehlo.reshape %v988 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v990 = stablehlo.reshape %v989 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v991 = stablehlo.constant dense<0.0> : tensor<f32>
    %v992 = stablehlo.reduce(%v990 init: %v991) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v993 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v994 = stablehlo.divide %v992, %v993 : tensor<32x672xf32>
    %v995 = stablehlo.dot_general %v994, %b10zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v996 = stablehlo.broadcast_in_dim %b10zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v997 = stablehlo.add %v995, %v996 : tensor<32x28xf32>
    %v998 = stablehlo.logistic %v997 : tensor<32x28xf32>
    %v999 = stablehlo.multiply %v997, %v998 : tensor<32x28xf32>
    %v1000 = stablehlo.dot_general %v999, %b10zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v1001 = stablehlo.broadcast_in_dim %b10zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v1002 = stablehlo.add %v1000, %v1001 : tensor<32x672xf32>
    %v1003 = stablehlo.logistic %v1002 : tensor<32x672xf32>
    %v1004 = stablehlo.broadcast_in_dim %v1003, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v1005 = stablehlo.multiply %v990, %v1004 : tensor<32x672x14x14xf32>
    %v1006 = stablehlo.reshape %v1005 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1007 = stablehlo.reshape %v1006 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1008 = stablehlo.convolution(%v1007, %b10pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v1009 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1010 = stablehlo.add %v1008, %v1009 : tensor<32x112x14x14xf32>
    %v1011 = stablehlo.reshape %v1010 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1012 = stablehlo.reshape %v1011 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1013 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1014 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v1015 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v1016 = stablehlo.reduce(%v1012 init: %v1013) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1017 = stablehlo.broadcast_in_dim %v1016, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1018 = stablehlo.divide %v1017, %v1014 : tensor<32x112x14x14xf32>
    %v1019 = stablehlo.subtract %v1012, %v1018 : tensor<32x112x14x14xf32>
    %v1020 = stablehlo.multiply %v1019, %v1019 : tensor<32x112x14x14xf32>
    %v1021 = stablehlo.reduce(%v1020 init: %v1013) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1022 = stablehlo.broadcast_in_dim %v1021, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1023 = stablehlo.divide %v1022, %v1014 : tensor<32x112x14x14xf32>
    %v1024 = stablehlo.add %v1023, %v1015 : tensor<32x112x14x14xf32>
    %v1025 = stablehlo.rsqrt %v1024 : tensor<32x112x14x14xf32>
    %v1026 = stablehlo.multiply %v1019, %v1025 : tensor<32x112x14x14xf32>
    %v1027 = stablehlo.broadcast_in_dim %b10pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1028 = stablehlo.broadcast_in_dim %b10pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1029 = stablehlo.multiply %v1026, %v1027 : tensor<32x112x14x14xf32>
    %v1030 = stablehlo.add %v1029, %v1028 : tensor<32x112x14x14xf32>
    %v1031 = stablehlo.reshape %v1030 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1032 = stablehlo.reshape %v1031 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1033 = stablehlo.broadcast_in_dim %dp9, dims = [0] : (tensor<32xf32>) -> tensor<32x112x14x14xf32>
    %v1034 = stablehlo.multiply %v1033, %v1032 : tensor<32x112x14x14xf32>
    %v1035 = stablehlo.reshape %v1034 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1036 = stablehlo.reshape %v1035 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1037 = stablehlo.reshape %v931 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1038 = stablehlo.add %v1036, %v1037 : tensor<32x112x14x14xf32>
    %v1039 = stablehlo.reshape %v1038 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1040 = stablehlo.reshape %v1039 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1041 = stablehlo.convolution(%v1040, %b11eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v1042 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1043 = stablehlo.add %v1041, %v1042 : tensor<32x672x14x14xf32>
    %v1044 = stablehlo.reshape %v1043 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1045 = stablehlo.reshape %v1044 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1046 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1047 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v1048 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v1049 = stablehlo.reduce(%v1045 init: %v1046) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1050 = stablehlo.broadcast_in_dim %v1049, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1051 = stablehlo.divide %v1050, %v1047 : tensor<32x672x14x14xf32>
    %v1052 = stablehlo.subtract %v1045, %v1051 : tensor<32x672x14x14xf32>
    %v1053 = stablehlo.multiply %v1052, %v1052 : tensor<32x672x14x14xf32>
    %v1054 = stablehlo.reduce(%v1053 init: %v1046) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1055 = stablehlo.broadcast_in_dim %v1054, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1056 = stablehlo.divide %v1055, %v1047 : tensor<32x672x14x14xf32>
    %v1057 = stablehlo.add %v1056, %v1048 : tensor<32x672x14x14xf32>
    %v1058 = stablehlo.rsqrt %v1057 : tensor<32x672x14x14xf32>
    %v1059 = stablehlo.multiply %v1052, %v1058 : tensor<32x672x14x14xf32>
    %v1060 = stablehlo.broadcast_in_dim %b11eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1061 = stablehlo.broadcast_in_dim %b11ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1062 = stablehlo.multiply %v1059, %v1060 : tensor<32x672x14x14xf32>
    %v1063 = stablehlo.add %v1062, %v1061 : tensor<32x672x14x14xf32>
    %v1064 = stablehlo.reshape %v1063 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1065 = stablehlo.reshape %v1064 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1066 = stablehlo.logistic %v1065 : tensor<32x672x14x14xf32>
    %v1067 = stablehlo.multiply %v1065, %v1066 : tensor<32x672x14x14xf32>
    %v1068 = stablehlo.reshape %v1067 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1069 = stablehlo.reshape %v1068 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1070 = stablehlo.convolution(%v1069, %b11dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x14x14xf32>
    %v1071 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1072 = stablehlo.add %v1070, %v1071 : tensor<32x672x14x14xf32>
    %v1073 = stablehlo.reshape %v1072 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1074 = stablehlo.reshape %v1073 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1075 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1076 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v1077 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v1078 = stablehlo.reduce(%v1074 init: %v1075) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1079 = stablehlo.broadcast_in_dim %v1078, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1080 = stablehlo.divide %v1079, %v1076 : tensor<32x672x14x14xf32>
    %v1081 = stablehlo.subtract %v1074, %v1080 : tensor<32x672x14x14xf32>
    %v1082 = stablehlo.multiply %v1081, %v1081 : tensor<32x672x14x14xf32>
    %v1083 = stablehlo.reduce(%v1082 init: %v1075) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1084 = stablehlo.broadcast_in_dim %v1083, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1085 = stablehlo.divide %v1084, %v1076 : tensor<32x672x14x14xf32>
    %v1086 = stablehlo.add %v1085, %v1077 : tensor<32x672x14x14xf32>
    %v1087 = stablehlo.rsqrt %v1086 : tensor<32x672x14x14xf32>
    %v1088 = stablehlo.multiply %v1081, %v1087 : tensor<32x672x14x14xf32>
    %v1089 = stablehlo.broadcast_in_dim %b11dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1090 = stablehlo.broadcast_in_dim %b11dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1091 = stablehlo.multiply %v1088, %v1089 : tensor<32x672x14x14xf32>
    %v1092 = stablehlo.add %v1091, %v1090 : tensor<32x672x14x14xf32>
    %v1093 = stablehlo.reshape %v1092 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1094 = stablehlo.reshape %v1093 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1095 = stablehlo.logistic %v1094 : tensor<32x672x14x14xf32>
    %v1096 = stablehlo.multiply %v1094, %v1095 : tensor<32x672x14x14xf32>
    %v1097 = stablehlo.reshape %v1096 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1098 = stablehlo.reshape %v1097 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1099 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1100 = stablehlo.reduce(%v1098 init: %v1099) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v1101 = stablehlo.constant dense<196.0> : tensor<32x672xf32>
    %v1102 = stablehlo.divide %v1100, %v1101 : tensor<32x672xf32>
    %v1103 = stablehlo.dot_general %v1102, %b11zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v1104 = stablehlo.broadcast_in_dim %b11zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v1105 = stablehlo.add %v1103, %v1104 : tensor<32x28xf32>
    %v1106 = stablehlo.logistic %v1105 : tensor<32x28xf32>
    %v1107 = stablehlo.multiply %v1105, %v1106 : tensor<32x28xf32>
    %v1108 = stablehlo.dot_general %v1107, %b11zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v1109 = stablehlo.broadcast_in_dim %b11zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v1110 = stablehlo.add %v1108, %v1109 : tensor<32x672xf32>
    %v1111 = stablehlo.logistic %v1110 : tensor<32x672xf32>
    %v1112 = stablehlo.broadcast_in_dim %v1111, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x14x14xf32>
    %v1113 = stablehlo.multiply %v1098, %v1112 : tensor<32x672x14x14xf32>
    %v1114 = stablehlo.reshape %v1113 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1115 = stablehlo.reshape %v1114 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1116 = stablehlo.convolution(%v1115, %b11pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x14x14xf32>, tensor<112x672x1x1xf32>) -> tensor<32x112x14x14xf32>
    %v1117 = stablehlo.broadcast_in_dim %zb112, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1118 = stablehlo.add %v1116, %v1117 : tensor<32x112x14x14xf32>
    %v1119 = stablehlo.reshape %v1118 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1120 = stablehlo.reshape %v1119 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1121 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1122 = stablehlo.constant dense<6272.0> : tensor<32x112x14x14xf32>
    %v1123 = stablehlo.constant dense<1.0e-5> : tensor<32x112x14x14xf32>
    %v1124 = stablehlo.reduce(%v1120 init: %v1121) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1125 = stablehlo.broadcast_in_dim %v1124, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1126 = stablehlo.divide %v1125, %v1122 : tensor<32x112x14x14xf32>
    %v1127 = stablehlo.subtract %v1120, %v1126 : tensor<32x112x14x14xf32>
    %v1128 = stablehlo.multiply %v1127, %v1127 : tensor<32x112x14x14xf32>
    %v1129 = stablehlo.reduce(%v1128 init: %v1121) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x112x14x14xf32>, tensor<f32>) -> tensor<112xf32>
    %v1130 = stablehlo.broadcast_in_dim %v1129, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1131 = stablehlo.divide %v1130, %v1122 : tensor<32x112x14x14xf32>
    %v1132 = stablehlo.add %v1131, %v1123 : tensor<32x112x14x14xf32>
    %v1133 = stablehlo.rsqrt %v1132 : tensor<32x112x14x14xf32>
    %v1134 = stablehlo.multiply %v1127, %v1133 : tensor<32x112x14x14xf32>
    %v1135 = stablehlo.broadcast_in_dim %b11pg, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1136 = stablehlo.broadcast_in_dim %b11pbt, dims = [1] : (tensor<112xf32>) -> tensor<32x112x14x14xf32>
    %v1137 = stablehlo.multiply %v1134, %v1135 : tensor<32x112x14x14xf32>
    %v1138 = stablehlo.add %v1137, %v1136 : tensor<32x112x14x14xf32>
    %v1139 = stablehlo.reshape %v1138 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1140 = stablehlo.reshape %v1139 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1141 = stablehlo.broadcast_in_dim %dp10, dims = [0] : (tensor<32xf32>) -> tensor<32x112x14x14xf32>
    %v1142 = stablehlo.multiply %v1141, %v1140 : tensor<32x112x14x14xf32>
    %v1143 = stablehlo.reshape %v1142 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1144 = stablehlo.reshape %v1143 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1145 = stablehlo.reshape %v1039 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1146 = stablehlo.add %v1144, %v1145 : tensor<32x112x14x14xf32>
    %v1147 = stablehlo.reshape %v1146 : (tensor<32x112x14x14xf32>) -> tensor<32x21952xf32>
    %v1148 = stablehlo.reshape %v1147 : (tensor<32x21952xf32>) -> tensor<32x112x14x14xf32>
    %v1149 = stablehlo.convolution(%v1148, %b12eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x112x14x14xf32>, tensor<672x112x1x1xf32>) -> tensor<32x672x14x14xf32>
    %v1150 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1151 = stablehlo.add %v1149, %v1150 : tensor<32x672x14x14xf32>
    %v1152 = stablehlo.reshape %v1151 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1153 = stablehlo.reshape %v1152 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1154 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1155 = stablehlo.constant dense<6272.0> : tensor<32x672x14x14xf32>
    %v1156 = stablehlo.constant dense<1.0e-5> : tensor<32x672x14x14xf32>
    %v1157 = stablehlo.reduce(%v1153 init: %v1154) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1158 = stablehlo.broadcast_in_dim %v1157, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1159 = stablehlo.divide %v1158, %v1155 : tensor<32x672x14x14xf32>
    %v1160 = stablehlo.subtract %v1153, %v1159 : tensor<32x672x14x14xf32>
    %v1161 = stablehlo.multiply %v1160, %v1160 : tensor<32x672x14x14xf32>
    %v1162 = stablehlo.reduce(%v1161 init: %v1154) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x14x14xf32>, tensor<f32>) -> tensor<672xf32>
    %v1163 = stablehlo.broadcast_in_dim %v1162, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1164 = stablehlo.divide %v1163, %v1155 : tensor<32x672x14x14xf32>
    %v1165 = stablehlo.add %v1164, %v1156 : tensor<32x672x14x14xf32>
    %v1166 = stablehlo.rsqrt %v1165 : tensor<32x672x14x14xf32>
    %v1167 = stablehlo.multiply %v1160, %v1166 : tensor<32x672x14x14xf32>
    %v1168 = stablehlo.broadcast_in_dim %b12eg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1169 = stablehlo.broadcast_in_dim %b12ebt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x14x14xf32>
    %v1170 = stablehlo.multiply %v1167, %v1168 : tensor<32x672x14x14xf32>
    %v1171 = stablehlo.add %v1170, %v1169 : tensor<32x672x14x14xf32>
    %v1172 = stablehlo.reshape %v1171 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1173 = stablehlo.reshape %v1172 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1174 = stablehlo.logistic %v1173 : tensor<32x672x14x14xf32>
    %v1175 = stablehlo.multiply %v1173, %v1174 : tensor<32x672x14x14xf32>
    %v1176 = stablehlo.reshape %v1175 : (tensor<32x672x14x14xf32>) -> tensor<32x131712xf32>
    %v1177 = stablehlo.reshape %v1176 : (tensor<32x131712xf32>) -> tensor<32x672x14x14xf32>
    %v1178 = stablehlo.convolution(%v1177, %b12dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 672 : i64} : (tensor<32x672x14x14xf32>, tensor<672x1x5x5xf32>) -> tensor<32x672x7x7xf32>
    %v1179 = stablehlo.broadcast_in_dim %zb672, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1180 = stablehlo.add %v1178, %v1179 : tensor<32x672x7x7xf32>
    %v1181 = stablehlo.reshape %v1180 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v1182 = stablehlo.reshape %v1181 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1183 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1184 = stablehlo.constant dense<1568.0> : tensor<32x672x7x7xf32>
    %v1185 = stablehlo.constant dense<1.0e-5> : tensor<32x672x7x7xf32>
    %v1186 = stablehlo.reduce(%v1182 init: %v1183) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v1187 = stablehlo.broadcast_in_dim %v1186, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1188 = stablehlo.divide %v1187, %v1184 : tensor<32x672x7x7xf32>
    %v1189 = stablehlo.subtract %v1182, %v1188 : tensor<32x672x7x7xf32>
    %v1190 = stablehlo.multiply %v1189, %v1189 : tensor<32x672x7x7xf32>
    %v1191 = stablehlo.reduce(%v1190 init: %v1183) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<672xf32>
    %v1192 = stablehlo.broadcast_in_dim %v1191, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1193 = stablehlo.divide %v1192, %v1184 : tensor<32x672x7x7xf32>
    %v1194 = stablehlo.add %v1193, %v1185 : tensor<32x672x7x7xf32>
    %v1195 = stablehlo.rsqrt %v1194 : tensor<32x672x7x7xf32>
    %v1196 = stablehlo.multiply %v1189, %v1195 : tensor<32x672x7x7xf32>
    %v1197 = stablehlo.broadcast_in_dim %b12dg, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1198 = stablehlo.broadcast_in_dim %b12dbt, dims = [1] : (tensor<672xf32>) -> tensor<32x672x7x7xf32>
    %v1199 = stablehlo.multiply %v1196, %v1197 : tensor<32x672x7x7xf32>
    %v1200 = stablehlo.add %v1199, %v1198 : tensor<32x672x7x7xf32>
    %v1201 = stablehlo.reshape %v1200 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v1202 = stablehlo.reshape %v1201 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1203 = stablehlo.logistic %v1202 : tensor<32x672x7x7xf32>
    %v1204 = stablehlo.multiply %v1202, %v1203 : tensor<32x672x7x7xf32>
    %v1205 = stablehlo.reshape %v1204 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v1206 = stablehlo.reshape %v1205 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1207 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1208 = stablehlo.reduce(%v1206 init: %v1207) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x672x7x7xf32>, tensor<f32>) -> tensor<32x672xf32>
    %v1209 = stablehlo.constant dense<49.0> : tensor<32x672xf32>
    %v1210 = stablehlo.divide %v1208, %v1209 : tensor<32x672xf32>
    %v1211 = stablehlo.dot_general %v1210, %b12zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x672xf32>, tensor<672x28xf32>) -> tensor<32x28xf32>
    %v1212 = stablehlo.broadcast_in_dim %b12zb1, dims = [1] : (tensor<28xf32>) -> tensor<32x28xf32>
    %v1213 = stablehlo.add %v1211, %v1212 : tensor<32x28xf32>
    %v1214 = stablehlo.logistic %v1213 : tensor<32x28xf32>
    %v1215 = stablehlo.multiply %v1213, %v1214 : tensor<32x28xf32>
    %v1216 = stablehlo.dot_general %v1215, %b12zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x28xf32>, tensor<28x672xf32>) -> tensor<32x672xf32>
    %v1217 = stablehlo.broadcast_in_dim %b12zb2, dims = [1] : (tensor<672xf32>) -> tensor<32x672xf32>
    %v1218 = stablehlo.add %v1216, %v1217 : tensor<32x672xf32>
    %v1219 = stablehlo.logistic %v1218 : tensor<32x672xf32>
    %v1220 = stablehlo.broadcast_in_dim %v1219, dims = [0, 1] : (tensor<32x672xf32>) -> tensor<32x672x7x7xf32>
    %v1221 = stablehlo.multiply %v1206, %v1220 : tensor<32x672x7x7xf32>
    %v1222 = stablehlo.reshape %v1221 : (tensor<32x672x7x7xf32>) -> tensor<32x32928xf32>
    %v1223 = stablehlo.reshape %v1222 : (tensor<32x32928xf32>) -> tensor<32x672x7x7xf32>
    %v1224 = stablehlo.convolution(%v1223, %b12pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x672x7x7xf32>, tensor<192x672x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1225 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1226 = stablehlo.add %v1224, %v1225 : tensor<32x192x7x7xf32>
    %v1227 = stablehlo.reshape %v1226 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1228 = stablehlo.reshape %v1227 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1229 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1230 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v1231 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1232 = stablehlo.reduce(%v1228 init: %v1229) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1233 = stablehlo.broadcast_in_dim %v1232, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1234 = stablehlo.divide %v1233, %v1230 : tensor<32x192x7x7xf32>
    %v1235 = stablehlo.subtract %v1228, %v1234 : tensor<32x192x7x7xf32>
    %v1236 = stablehlo.multiply %v1235, %v1235 : tensor<32x192x7x7xf32>
    %v1237 = stablehlo.reduce(%v1236 init: %v1229) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1238 = stablehlo.broadcast_in_dim %v1237, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1239 = stablehlo.divide %v1238, %v1230 : tensor<32x192x7x7xf32>
    %v1240 = stablehlo.add %v1239, %v1231 : tensor<32x192x7x7xf32>
    %v1241 = stablehlo.rsqrt %v1240 : tensor<32x192x7x7xf32>
    %v1242 = stablehlo.multiply %v1235, %v1241 : tensor<32x192x7x7xf32>
    %v1243 = stablehlo.broadcast_in_dim %b12pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1244 = stablehlo.broadcast_in_dim %b12pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1245 = stablehlo.multiply %v1242, %v1243 : tensor<32x192x7x7xf32>
    %v1246 = stablehlo.add %v1245, %v1244 : tensor<32x192x7x7xf32>
    %v1247 = stablehlo.reshape %v1246 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1248 = stablehlo.reshape %v1247 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1249 = stablehlo.convolution(%v1248, %b13eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1250 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1251 = stablehlo.add %v1249, %v1250 : tensor<32x1152x7x7xf32>
    %v1252 = stablehlo.reshape %v1251 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1253 = stablehlo.reshape %v1252 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1254 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1255 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1256 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1257 = stablehlo.reduce(%v1253 init: %v1254) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1258 = stablehlo.broadcast_in_dim %v1257, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1259 = stablehlo.divide %v1258, %v1255 : tensor<32x1152x7x7xf32>
    %v1260 = stablehlo.subtract %v1253, %v1259 : tensor<32x1152x7x7xf32>
    %v1261 = stablehlo.multiply %v1260, %v1260 : tensor<32x1152x7x7xf32>
    %v1262 = stablehlo.reduce(%v1261 init: %v1254) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1263 = stablehlo.broadcast_in_dim %v1262, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1264 = stablehlo.divide %v1263, %v1255 : tensor<32x1152x7x7xf32>
    %v1265 = stablehlo.add %v1264, %v1256 : tensor<32x1152x7x7xf32>
    %v1266 = stablehlo.rsqrt %v1265 : tensor<32x1152x7x7xf32>
    %v1267 = stablehlo.multiply %v1260, %v1266 : tensor<32x1152x7x7xf32>
    %v1268 = stablehlo.broadcast_in_dim %b13eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1269 = stablehlo.broadcast_in_dim %b13ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1270 = stablehlo.multiply %v1267, %v1268 : tensor<32x1152x7x7xf32>
    %v1271 = stablehlo.add %v1270, %v1269 : tensor<32x1152x7x7xf32>
    %v1272 = stablehlo.reshape %v1271 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1273 = stablehlo.reshape %v1272 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1274 = stablehlo.logistic %v1273 : tensor<32x1152x7x7xf32>
    %v1275 = stablehlo.multiply %v1273, %v1274 : tensor<32x1152x7x7xf32>
    %v1276 = stablehlo.reshape %v1275 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1277 = stablehlo.reshape %v1276 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1278 = stablehlo.convolution(%v1277, %b13dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v1279 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1280 = stablehlo.add %v1278, %v1279 : tensor<32x1152x7x7xf32>
    %v1281 = stablehlo.reshape %v1280 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1282 = stablehlo.reshape %v1281 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1283 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1284 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1285 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1286 = stablehlo.reduce(%v1282 init: %v1283) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1287 = stablehlo.broadcast_in_dim %v1286, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1288 = stablehlo.divide %v1287, %v1284 : tensor<32x1152x7x7xf32>
    %v1289 = stablehlo.subtract %v1282, %v1288 : tensor<32x1152x7x7xf32>
    %v1290 = stablehlo.multiply %v1289, %v1289 : tensor<32x1152x7x7xf32>
    %v1291 = stablehlo.reduce(%v1290 init: %v1283) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1292 = stablehlo.broadcast_in_dim %v1291, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1293 = stablehlo.divide %v1292, %v1284 : tensor<32x1152x7x7xf32>
    %v1294 = stablehlo.add %v1293, %v1285 : tensor<32x1152x7x7xf32>
    %v1295 = stablehlo.rsqrt %v1294 : tensor<32x1152x7x7xf32>
    %v1296 = stablehlo.multiply %v1289, %v1295 : tensor<32x1152x7x7xf32>
    %v1297 = stablehlo.broadcast_in_dim %b13dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1298 = stablehlo.broadcast_in_dim %b13dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1299 = stablehlo.multiply %v1296, %v1297 : tensor<32x1152x7x7xf32>
    %v1300 = stablehlo.add %v1299, %v1298 : tensor<32x1152x7x7xf32>
    %v1301 = stablehlo.reshape %v1300 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1302 = stablehlo.reshape %v1301 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1303 = stablehlo.logistic %v1302 : tensor<32x1152x7x7xf32>
    %v1304 = stablehlo.multiply %v1302, %v1303 : tensor<32x1152x7x7xf32>
    %v1305 = stablehlo.reshape %v1304 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1306 = stablehlo.reshape %v1305 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1307 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1308 = stablehlo.reduce(%v1306 init: %v1307) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1309 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1310 = stablehlo.divide %v1308, %v1309 : tensor<32x1152xf32>
    %v1311 = stablehlo.dot_general %v1310, %b13zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1312 = stablehlo.broadcast_in_dim %b13zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1313 = stablehlo.add %v1311, %v1312 : tensor<32x48xf32>
    %v1314 = stablehlo.logistic %v1313 : tensor<32x48xf32>
    %v1315 = stablehlo.multiply %v1313, %v1314 : tensor<32x48xf32>
    %v1316 = stablehlo.dot_general %v1315, %b13zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1317 = stablehlo.broadcast_in_dim %b13zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1318 = stablehlo.add %v1316, %v1317 : tensor<32x1152xf32>
    %v1319 = stablehlo.logistic %v1318 : tensor<32x1152xf32>
    %v1320 = stablehlo.broadcast_in_dim %v1319, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1321 = stablehlo.multiply %v1306, %v1320 : tensor<32x1152x7x7xf32>
    %v1322 = stablehlo.reshape %v1321 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1323 = stablehlo.reshape %v1322 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1324 = stablehlo.convolution(%v1323, %b13pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1325 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1326 = stablehlo.add %v1324, %v1325 : tensor<32x192x7x7xf32>
    %v1327 = stablehlo.reshape %v1326 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1328 = stablehlo.reshape %v1327 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1329 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1330 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v1331 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1332 = stablehlo.reduce(%v1328 init: %v1329) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1333 = stablehlo.broadcast_in_dim %v1332, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1334 = stablehlo.divide %v1333, %v1330 : tensor<32x192x7x7xf32>
    %v1335 = stablehlo.subtract %v1328, %v1334 : tensor<32x192x7x7xf32>
    %v1336 = stablehlo.multiply %v1335, %v1335 : tensor<32x192x7x7xf32>
    %v1337 = stablehlo.reduce(%v1336 init: %v1329) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1338 = stablehlo.broadcast_in_dim %v1337, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1339 = stablehlo.divide %v1338, %v1330 : tensor<32x192x7x7xf32>
    %v1340 = stablehlo.add %v1339, %v1331 : tensor<32x192x7x7xf32>
    %v1341 = stablehlo.rsqrt %v1340 : tensor<32x192x7x7xf32>
    %v1342 = stablehlo.multiply %v1335, %v1341 : tensor<32x192x7x7xf32>
    %v1343 = stablehlo.broadcast_in_dim %b13pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1344 = stablehlo.broadcast_in_dim %b13pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1345 = stablehlo.multiply %v1342, %v1343 : tensor<32x192x7x7xf32>
    %v1346 = stablehlo.add %v1345, %v1344 : tensor<32x192x7x7xf32>
    %v1347 = stablehlo.reshape %v1346 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1348 = stablehlo.reshape %v1347 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1349 = stablehlo.broadcast_in_dim %dp12, dims = [0] : (tensor<32xf32>) -> tensor<32x192x7x7xf32>
    %v1350 = stablehlo.multiply %v1349, %v1348 : tensor<32x192x7x7xf32>
    %v1351 = stablehlo.reshape %v1350 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1352 = stablehlo.reshape %v1351 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1353 = stablehlo.reshape %v1247 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1354 = stablehlo.add %v1352, %v1353 : tensor<32x192x7x7xf32>
    %v1355 = stablehlo.reshape %v1354 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1356 = stablehlo.reshape %v1355 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1357 = stablehlo.convolution(%v1356, %b14eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1358 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1359 = stablehlo.add %v1357, %v1358 : tensor<32x1152x7x7xf32>
    %v1360 = stablehlo.reshape %v1359 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1361 = stablehlo.reshape %v1360 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1362 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1363 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1364 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1365 = stablehlo.reduce(%v1361 init: %v1362) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1366 = stablehlo.broadcast_in_dim %v1365, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1367 = stablehlo.divide %v1366, %v1363 : tensor<32x1152x7x7xf32>
    %v1368 = stablehlo.subtract %v1361, %v1367 : tensor<32x1152x7x7xf32>
    %v1369 = stablehlo.multiply %v1368, %v1368 : tensor<32x1152x7x7xf32>
    %v1370 = stablehlo.reduce(%v1369 init: %v1362) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1371 = stablehlo.broadcast_in_dim %v1370, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1372 = stablehlo.divide %v1371, %v1363 : tensor<32x1152x7x7xf32>
    %v1373 = stablehlo.add %v1372, %v1364 : tensor<32x1152x7x7xf32>
    %v1374 = stablehlo.rsqrt %v1373 : tensor<32x1152x7x7xf32>
    %v1375 = stablehlo.multiply %v1368, %v1374 : tensor<32x1152x7x7xf32>
    %v1376 = stablehlo.broadcast_in_dim %b14eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1377 = stablehlo.broadcast_in_dim %b14ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1378 = stablehlo.multiply %v1375, %v1376 : tensor<32x1152x7x7xf32>
    %v1379 = stablehlo.add %v1378, %v1377 : tensor<32x1152x7x7xf32>
    %v1380 = stablehlo.reshape %v1379 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1381 = stablehlo.reshape %v1380 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1382 = stablehlo.logistic %v1381 : tensor<32x1152x7x7xf32>
    %v1383 = stablehlo.multiply %v1381, %v1382 : tensor<32x1152x7x7xf32>
    %v1384 = stablehlo.reshape %v1383 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1385 = stablehlo.reshape %v1384 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1386 = stablehlo.convolution(%v1385, %b14dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v1387 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1388 = stablehlo.add %v1386, %v1387 : tensor<32x1152x7x7xf32>
    %v1389 = stablehlo.reshape %v1388 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1390 = stablehlo.reshape %v1389 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1391 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1392 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1393 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1394 = stablehlo.reduce(%v1390 init: %v1391) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1395 = stablehlo.broadcast_in_dim %v1394, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1396 = stablehlo.divide %v1395, %v1392 : tensor<32x1152x7x7xf32>
    %v1397 = stablehlo.subtract %v1390, %v1396 : tensor<32x1152x7x7xf32>
    %v1398 = stablehlo.multiply %v1397, %v1397 : tensor<32x1152x7x7xf32>
    %v1399 = stablehlo.reduce(%v1398 init: %v1391) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1400 = stablehlo.broadcast_in_dim %v1399, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1401 = stablehlo.divide %v1400, %v1392 : tensor<32x1152x7x7xf32>
    %v1402 = stablehlo.add %v1401, %v1393 : tensor<32x1152x7x7xf32>
    %v1403 = stablehlo.rsqrt %v1402 : tensor<32x1152x7x7xf32>
    %v1404 = stablehlo.multiply %v1397, %v1403 : tensor<32x1152x7x7xf32>
    %v1405 = stablehlo.broadcast_in_dim %b14dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1406 = stablehlo.broadcast_in_dim %b14dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1407 = stablehlo.multiply %v1404, %v1405 : tensor<32x1152x7x7xf32>
    %v1408 = stablehlo.add %v1407, %v1406 : tensor<32x1152x7x7xf32>
    %v1409 = stablehlo.reshape %v1408 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1410 = stablehlo.reshape %v1409 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1411 = stablehlo.logistic %v1410 : tensor<32x1152x7x7xf32>
    %v1412 = stablehlo.multiply %v1410, %v1411 : tensor<32x1152x7x7xf32>
    %v1413 = stablehlo.reshape %v1412 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1414 = stablehlo.reshape %v1413 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1415 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1416 = stablehlo.reduce(%v1414 init: %v1415) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1417 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1418 = stablehlo.divide %v1416, %v1417 : tensor<32x1152xf32>
    %v1419 = stablehlo.dot_general %v1418, %b14zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1420 = stablehlo.broadcast_in_dim %b14zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1421 = stablehlo.add %v1419, %v1420 : tensor<32x48xf32>
    %v1422 = stablehlo.logistic %v1421 : tensor<32x48xf32>
    %v1423 = stablehlo.multiply %v1421, %v1422 : tensor<32x48xf32>
    %v1424 = stablehlo.dot_general %v1423, %b14zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1425 = stablehlo.broadcast_in_dim %b14zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1426 = stablehlo.add %v1424, %v1425 : tensor<32x1152xf32>
    %v1427 = stablehlo.logistic %v1426 : tensor<32x1152xf32>
    %v1428 = stablehlo.broadcast_in_dim %v1427, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1429 = stablehlo.multiply %v1414, %v1428 : tensor<32x1152x7x7xf32>
    %v1430 = stablehlo.reshape %v1429 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1431 = stablehlo.reshape %v1430 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1432 = stablehlo.convolution(%v1431, %b14pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1433 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1434 = stablehlo.add %v1432, %v1433 : tensor<32x192x7x7xf32>
    %v1435 = stablehlo.reshape %v1434 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1436 = stablehlo.reshape %v1435 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1437 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1438 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v1439 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1440 = stablehlo.reduce(%v1436 init: %v1437) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1441 = stablehlo.broadcast_in_dim %v1440, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1442 = stablehlo.divide %v1441, %v1438 : tensor<32x192x7x7xf32>
    %v1443 = stablehlo.subtract %v1436, %v1442 : tensor<32x192x7x7xf32>
    %v1444 = stablehlo.multiply %v1443, %v1443 : tensor<32x192x7x7xf32>
    %v1445 = stablehlo.reduce(%v1444 init: %v1437) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1446 = stablehlo.broadcast_in_dim %v1445, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1447 = stablehlo.divide %v1446, %v1438 : tensor<32x192x7x7xf32>
    %v1448 = stablehlo.add %v1447, %v1439 : tensor<32x192x7x7xf32>
    %v1449 = stablehlo.rsqrt %v1448 : tensor<32x192x7x7xf32>
    %v1450 = stablehlo.multiply %v1443, %v1449 : tensor<32x192x7x7xf32>
    %v1451 = stablehlo.broadcast_in_dim %b14pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1452 = stablehlo.broadcast_in_dim %b14pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1453 = stablehlo.multiply %v1450, %v1451 : tensor<32x192x7x7xf32>
    %v1454 = stablehlo.add %v1453, %v1452 : tensor<32x192x7x7xf32>
    %v1455 = stablehlo.reshape %v1454 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1456 = stablehlo.reshape %v1455 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1457 = stablehlo.broadcast_in_dim %dp13, dims = [0] : (tensor<32xf32>) -> tensor<32x192x7x7xf32>
    %v1458 = stablehlo.multiply %v1457, %v1456 : tensor<32x192x7x7xf32>
    %v1459 = stablehlo.reshape %v1458 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1460 = stablehlo.reshape %v1459 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1461 = stablehlo.reshape %v1355 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1462 = stablehlo.add %v1460, %v1461 : tensor<32x192x7x7xf32>
    %v1463 = stablehlo.reshape %v1462 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1464 = stablehlo.reshape %v1463 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1465 = stablehlo.convolution(%v1464, %b15eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1466 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1467 = stablehlo.add %v1465, %v1466 : tensor<32x1152x7x7xf32>
    %v1468 = stablehlo.reshape %v1467 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1469 = stablehlo.reshape %v1468 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1470 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1471 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1472 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1473 = stablehlo.reduce(%v1469 init: %v1470) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1474 = stablehlo.broadcast_in_dim %v1473, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1475 = stablehlo.divide %v1474, %v1471 : tensor<32x1152x7x7xf32>
    %v1476 = stablehlo.subtract %v1469, %v1475 : tensor<32x1152x7x7xf32>
    %v1477 = stablehlo.multiply %v1476, %v1476 : tensor<32x1152x7x7xf32>
    %v1478 = stablehlo.reduce(%v1477 init: %v1470) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1479 = stablehlo.broadcast_in_dim %v1478, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1480 = stablehlo.divide %v1479, %v1471 : tensor<32x1152x7x7xf32>
    %v1481 = stablehlo.add %v1480, %v1472 : tensor<32x1152x7x7xf32>
    %v1482 = stablehlo.rsqrt %v1481 : tensor<32x1152x7x7xf32>
    %v1483 = stablehlo.multiply %v1476, %v1482 : tensor<32x1152x7x7xf32>
    %v1484 = stablehlo.broadcast_in_dim %b15eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1485 = stablehlo.broadcast_in_dim %b15ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1486 = stablehlo.multiply %v1483, %v1484 : tensor<32x1152x7x7xf32>
    %v1487 = stablehlo.add %v1486, %v1485 : tensor<32x1152x7x7xf32>
    %v1488 = stablehlo.reshape %v1487 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1489 = stablehlo.reshape %v1488 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1490 = stablehlo.logistic %v1489 : tensor<32x1152x7x7xf32>
    %v1491 = stablehlo.multiply %v1489, %v1490 : tensor<32x1152x7x7xf32>
    %v1492 = stablehlo.reshape %v1491 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1493 = stablehlo.reshape %v1492 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1494 = stablehlo.convolution(%v1493, %b15dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x5x5xf32>) -> tensor<32x1152x7x7xf32>
    %v1495 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1496 = stablehlo.add %v1494, %v1495 : tensor<32x1152x7x7xf32>
    %v1497 = stablehlo.reshape %v1496 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1498 = stablehlo.reshape %v1497 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1499 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1500 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1501 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1502 = stablehlo.reduce(%v1498 init: %v1499) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1503 = stablehlo.broadcast_in_dim %v1502, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1504 = stablehlo.divide %v1503, %v1500 : tensor<32x1152x7x7xf32>
    %v1505 = stablehlo.subtract %v1498, %v1504 : tensor<32x1152x7x7xf32>
    %v1506 = stablehlo.multiply %v1505, %v1505 : tensor<32x1152x7x7xf32>
    %v1507 = stablehlo.reduce(%v1506 init: %v1499) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1508 = stablehlo.broadcast_in_dim %v1507, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1509 = stablehlo.divide %v1508, %v1500 : tensor<32x1152x7x7xf32>
    %v1510 = stablehlo.add %v1509, %v1501 : tensor<32x1152x7x7xf32>
    %v1511 = stablehlo.rsqrt %v1510 : tensor<32x1152x7x7xf32>
    %v1512 = stablehlo.multiply %v1505, %v1511 : tensor<32x1152x7x7xf32>
    %v1513 = stablehlo.broadcast_in_dim %b15dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1514 = stablehlo.broadcast_in_dim %b15dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1515 = stablehlo.multiply %v1512, %v1513 : tensor<32x1152x7x7xf32>
    %v1516 = stablehlo.add %v1515, %v1514 : tensor<32x1152x7x7xf32>
    %v1517 = stablehlo.reshape %v1516 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1518 = stablehlo.reshape %v1517 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1519 = stablehlo.logistic %v1518 : tensor<32x1152x7x7xf32>
    %v1520 = stablehlo.multiply %v1518, %v1519 : tensor<32x1152x7x7xf32>
    %v1521 = stablehlo.reshape %v1520 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1522 = stablehlo.reshape %v1521 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1523 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1524 = stablehlo.reduce(%v1522 init: %v1523) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1525 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1526 = stablehlo.divide %v1524, %v1525 : tensor<32x1152xf32>
    %v1527 = stablehlo.dot_general %v1526, %b15zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1528 = stablehlo.broadcast_in_dim %b15zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1529 = stablehlo.add %v1527, %v1528 : tensor<32x48xf32>
    %v1530 = stablehlo.logistic %v1529 : tensor<32x48xf32>
    %v1531 = stablehlo.multiply %v1529, %v1530 : tensor<32x48xf32>
    %v1532 = stablehlo.dot_general %v1531, %b15zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1533 = stablehlo.broadcast_in_dim %b15zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1534 = stablehlo.add %v1532, %v1533 : tensor<32x1152xf32>
    %v1535 = stablehlo.logistic %v1534 : tensor<32x1152xf32>
    %v1536 = stablehlo.broadcast_in_dim %v1535, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1537 = stablehlo.multiply %v1522, %v1536 : tensor<32x1152x7x7xf32>
    %v1538 = stablehlo.reshape %v1537 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1539 = stablehlo.reshape %v1538 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1540 = stablehlo.convolution(%v1539, %b15pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<192x1152x1x1xf32>) -> tensor<32x192x7x7xf32>
    %v1541 = stablehlo.broadcast_in_dim %zb192, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1542 = stablehlo.add %v1540, %v1541 : tensor<32x192x7x7xf32>
    %v1543 = stablehlo.reshape %v1542 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1544 = stablehlo.reshape %v1543 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1545 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1546 = stablehlo.constant dense<1568.0> : tensor<32x192x7x7xf32>
    %v1547 = stablehlo.constant dense<1.0e-5> : tensor<32x192x7x7xf32>
    %v1548 = stablehlo.reduce(%v1544 init: %v1545) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1549 = stablehlo.broadcast_in_dim %v1548, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1550 = stablehlo.divide %v1549, %v1546 : tensor<32x192x7x7xf32>
    %v1551 = stablehlo.subtract %v1544, %v1550 : tensor<32x192x7x7xf32>
    %v1552 = stablehlo.multiply %v1551, %v1551 : tensor<32x192x7x7xf32>
    %v1553 = stablehlo.reduce(%v1552 init: %v1545) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x7x7xf32>, tensor<f32>) -> tensor<192xf32>
    %v1554 = stablehlo.broadcast_in_dim %v1553, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1555 = stablehlo.divide %v1554, %v1546 : tensor<32x192x7x7xf32>
    %v1556 = stablehlo.add %v1555, %v1547 : tensor<32x192x7x7xf32>
    %v1557 = stablehlo.rsqrt %v1556 : tensor<32x192x7x7xf32>
    %v1558 = stablehlo.multiply %v1551, %v1557 : tensor<32x192x7x7xf32>
    %v1559 = stablehlo.broadcast_in_dim %b15pg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1560 = stablehlo.broadcast_in_dim %b15pbt, dims = [1] : (tensor<192xf32>) -> tensor<32x192x7x7xf32>
    %v1561 = stablehlo.multiply %v1558, %v1559 : tensor<32x192x7x7xf32>
    %v1562 = stablehlo.add %v1561, %v1560 : tensor<32x192x7x7xf32>
    %v1563 = stablehlo.reshape %v1562 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1564 = stablehlo.reshape %v1563 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1565 = stablehlo.broadcast_in_dim %dp14, dims = [0] : (tensor<32xf32>) -> tensor<32x192x7x7xf32>
    %v1566 = stablehlo.multiply %v1565, %v1564 : tensor<32x192x7x7xf32>
    %v1567 = stablehlo.reshape %v1566 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1568 = stablehlo.reshape %v1567 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1569 = stablehlo.reshape %v1463 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1570 = stablehlo.add %v1568, %v1569 : tensor<32x192x7x7xf32>
    %v1571 = stablehlo.reshape %v1570 : (tensor<32x192x7x7xf32>) -> tensor<32x9408xf32>
    %v1572 = stablehlo.reshape %v1571 : (tensor<32x9408xf32>) -> tensor<32x192x7x7xf32>
    %v1573 = stablehlo.convolution(%v1572, %b16eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x7x7xf32>, tensor<1152x192x1x1xf32>) -> tensor<32x1152x7x7xf32>
    %v1574 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1575 = stablehlo.add %v1573, %v1574 : tensor<32x1152x7x7xf32>
    %v1576 = stablehlo.reshape %v1575 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1577 = stablehlo.reshape %v1576 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1578 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1579 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1580 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1581 = stablehlo.reduce(%v1577 init: %v1578) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1582 = stablehlo.broadcast_in_dim %v1581, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1583 = stablehlo.divide %v1582, %v1579 : tensor<32x1152x7x7xf32>
    %v1584 = stablehlo.subtract %v1577, %v1583 : tensor<32x1152x7x7xf32>
    %v1585 = stablehlo.multiply %v1584, %v1584 : tensor<32x1152x7x7xf32>
    %v1586 = stablehlo.reduce(%v1585 init: %v1578) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1587 = stablehlo.broadcast_in_dim %v1586, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1588 = stablehlo.divide %v1587, %v1579 : tensor<32x1152x7x7xf32>
    %v1589 = stablehlo.add %v1588, %v1580 : tensor<32x1152x7x7xf32>
    %v1590 = stablehlo.rsqrt %v1589 : tensor<32x1152x7x7xf32>
    %v1591 = stablehlo.multiply %v1584, %v1590 : tensor<32x1152x7x7xf32>
    %v1592 = stablehlo.broadcast_in_dim %b16eg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1593 = stablehlo.broadcast_in_dim %b16ebt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1594 = stablehlo.multiply %v1591, %v1592 : tensor<32x1152x7x7xf32>
    %v1595 = stablehlo.add %v1594, %v1593 : tensor<32x1152x7x7xf32>
    %v1596 = stablehlo.reshape %v1595 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1597 = stablehlo.reshape %v1596 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1598 = stablehlo.logistic %v1597 : tensor<32x1152x7x7xf32>
    %v1599 = stablehlo.multiply %v1597, %v1598 : tensor<32x1152x7x7xf32>
    %v1600 = stablehlo.reshape %v1599 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1601 = stablehlo.reshape %v1600 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1602 = stablehlo.convolution(%v1601, %b16dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1152 : i64} : (tensor<32x1152x7x7xf32>, tensor<1152x1x3x3xf32>) -> tensor<32x1152x7x7xf32>
    %v1603 = stablehlo.broadcast_in_dim %zb1152, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1604 = stablehlo.add %v1602, %v1603 : tensor<32x1152x7x7xf32>
    %v1605 = stablehlo.reshape %v1604 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1606 = stablehlo.reshape %v1605 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1607 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1608 = stablehlo.constant dense<1568.0> : tensor<32x1152x7x7xf32>
    %v1609 = stablehlo.constant dense<1.0e-5> : tensor<32x1152x7x7xf32>
    %v1610 = stablehlo.reduce(%v1606 init: %v1607) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1611 = stablehlo.broadcast_in_dim %v1610, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1612 = stablehlo.divide %v1611, %v1608 : tensor<32x1152x7x7xf32>
    %v1613 = stablehlo.subtract %v1606, %v1612 : tensor<32x1152x7x7xf32>
    %v1614 = stablehlo.multiply %v1613, %v1613 : tensor<32x1152x7x7xf32>
    %v1615 = stablehlo.reduce(%v1614 init: %v1607) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<1152xf32>
    %v1616 = stablehlo.broadcast_in_dim %v1615, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1617 = stablehlo.divide %v1616, %v1608 : tensor<32x1152x7x7xf32>
    %v1618 = stablehlo.add %v1617, %v1609 : tensor<32x1152x7x7xf32>
    %v1619 = stablehlo.rsqrt %v1618 : tensor<32x1152x7x7xf32>
    %v1620 = stablehlo.multiply %v1613, %v1619 : tensor<32x1152x7x7xf32>
    %v1621 = stablehlo.broadcast_in_dim %b16dg, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1622 = stablehlo.broadcast_in_dim %b16dbt, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1623 = stablehlo.multiply %v1620, %v1621 : tensor<32x1152x7x7xf32>
    %v1624 = stablehlo.add %v1623, %v1622 : tensor<32x1152x7x7xf32>
    %v1625 = stablehlo.reshape %v1624 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1626 = stablehlo.reshape %v1625 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1627 = stablehlo.logistic %v1626 : tensor<32x1152x7x7xf32>
    %v1628 = stablehlo.multiply %v1626, %v1627 : tensor<32x1152x7x7xf32>
    %v1629 = stablehlo.reshape %v1628 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1630 = stablehlo.reshape %v1629 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1631 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1632 = stablehlo.reduce(%v1630 init: %v1631) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1152x7x7xf32>, tensor<f32>) -> tensor<32x1152xf32>
    %v1633 = stablehlo.constant dense<49.0> : tensor<32x1152xf32>
    %v1634 = stablehlo.divide %v1632, %v1633 : tensor<32x1152xf32>
    %v1635 = stablehlo.dot_general %v1634, %b16zW1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1152xf32>, tensor<1152x48xf32>) -> tensor<32x48xf32>
    %v1636 = stablehlo.broadcast_in_dim %b16zb1, dims = [1] : (tensor<48xf32>) -> tensor<32x48xf32>
    %v1637 = stablehlo.add %v1635, %v1636 : tensor<32x48xf32>
    %v1638 = stablehlo.logistic %v1637 : tensor<32x48xf32>
    %v1639 = stablehlo.multiply %v1637, %v1638 : tensor<32x48xf32>
    %v1640 = stablehlo.dot_general %v1639, %b16zW2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x48xf32>, tensor<48x1152xf32>) -> tensor<32x1152xf32>
    %v1641 = stablehlo.broadcast_in_dim %b16zb2, dims = [1] : (tensor<1152xf32>) -> tensor<32x1152xf32>
    %v1642 = stablehlo.add %v1640, %v1641 : tensor<32x1152xf32>
    %v1643 = stablehlo.logistic %v1642 : tensor<32x1152xf32>
    %v1644 = stablehlo.broadcast_in_dim %v1643, dims = [0, 1] : (tensor<32x1152xf32>) -> tensor<32x1152x7x7xf32>
    %v1645 = stablehlo.multiply %v1630, %v1644 : tensor<32x1152x7x7xf32>
    %v1646 = stablehlo.reshape %v1645 : (tensor<32x1152x7x7xf32>) -> tensor<32x56448xf32>
    %v1647 = stablehlo.reshape %v1646 : (tensor<32x56448xf32>) -> tensor<32x1152x7x7xf32>
    %v1648 = stablehlo.convolution(%v1647, %b16pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1152x7x7xf32>, tensor<320x1152x1x1xf32>) -> tensor<32x320x7x7xf32>
    %v1649 = stablehlo.broadcast_in_dim %zb320, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1650 = stablehlo.add %v1648, %v1649 : tensor<32x320x7x7xf32>
    %v1651 = stablehlo.reshape %v1650 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1652 = stablehlo.reshape %v1651 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1653 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1654 = stablehlo.constant dense<1568.0> : tensor<32x320x7x7xf32>
    %v1655 = stablehlo.constant dense<1.0e-5> : tensor<32x320x7x7xf32>
    %v1656 = stablehlo.reduce(%v1652 init: %v1653) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1657 = stablehlo.broadcast_in_dim %v1656, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1658 = stablehlo.divide %v1657, %v1654 : tensor<32x320x7x7xf32>
    %v1659 = stablehlo.subtract %v1652, %v1658 : tensor<32x320x7x7xf32>
    %v1660 = stablehlo.multiply %v1659, %v1659 : tensor<32x320x7x7xf32>
    %v1661 = stablehlo.reduce(%v1660 init: %v1653) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x320x7x7xf32>, tensor<f32>) -> tensor<320xf32>
    %v1662 = stablehlo.broadcast_in_dim %v1661, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1663 = stablehlo.divide %v1662, %v1654 : tensor<32x320x7x7xf32>
    %v1664 = stablehlo.add %v1663, %v1655 : tensor<32x320x7x7xf32>
    %v1665 = stablehlo.rsqrt %v1664 : tensor<32x320x7x7xf32>
    %v1666 = stablehlo.multiply %v1659, %v1665 : tensor<32x320x7x7xf32>
    %v1667 = stablehlo.broadcast_in_dim %b16pg, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1668 = stablehlo.broadcast_in_dim %b16pbt, dims = [1] : (tensor<320xf32>) -> tensor<32x320x7x7xf32>
    %v1669 = stablehlo.multiply %v1666, %v1667 : tensor<32x320x7x7xf32>
    %v1670 = stablehlo.add %v1669, %v1668 : tensor<32x320x7x7xf32>
    %v1671 = stablehlo.reshape %v1670 : (tensor<32x320x7x7xf32>) -> tensor<32x15680xf32>
    %v1672 = stablehlo.reshape %v1671 : (tensor<32x15680xf32>) -> tensor<32x320x7x7xf32>
    %v1673 = stablehlo.convolution(%v1672, %hW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x320x7x7xf32>, tensor<1280x320x1x1xf32>) -> tensor<32x1280x7x7xf32>
    %v1674 = stablehlo.broadcast_in_dim %zb1280, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1675 = stablehlo.add %v1673, %v1674 : tensor<32x1280x7x7xf32>
    %v1676 = stablehlo.reshape %v1675 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1677 = stablehlo.reshape %v1676 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1678 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1679 = stablehlo.constant dense<1568.0> : tensor<32x1280x7x7xf32>
    %v1680 = stablehlo.constant dense<1.0e-5> : tensor<32x1280x7x7xf32>
    %v1681 = stablehlo.reduce(%v1677 init: %v1678) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1682 = stablehlo.broadcast_in_dim %v1681, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1683 = stablehlo.divide %v1682, %v1679 : tensor<32x1280x7x7xf32>
    %v1684 = stablehlo.subtract %v1677, %v1683 : tensor<32x1280x7x7xf32>
    %v1685 = stablehlo.multiply %v1684, %v1684 : tensor<32x1280x7x7xf32>
    %v1686 = stablehlo.reduce(%v1685 init: %v1678) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<1280xf32>
    %v1687 = stablehlo.broadcast_in_dim %v1686, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1688 = stablehlo.divide %v1687, %v1679 : tensor<32x1280x7x7xf32>
    %v1689 = stablehlo.add %v1688, %v1680 : tensor<32x1280x7x7xf32>
    %v1690 = stablehlo.rsqrt %v1689 : tensor<32x1280x7x7xf32>
    %v1691 = stablehlo.multiply %v1684, %v1690 : tensor<32x1280x7x7xf32>
    %v1692 = stablehlo.broadcast_in_dim %hg, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1693 = stablehlo.broadcast_in_dim %hbt, dims = [1] : (tensor<1280xf32>) -> tensor<32x1280x7x7xf32>
    %v1694 = stablehlo.multiply %v1691, %v1692 : tensor<32x1280x7x7xf32>
    %v1695 = stablehlo.add %v1694, %v1693 : tensor<32x1280x7x7xf32>
    %v1696 = stablehlo.reshape %v1695 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1697 = stablehlo.reshape %v1696 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1698 = stablehlo.logistic %v1697 : tensor<32x1280x7x7xf32>
    %v1699 = stablehlo.multiply %v1697, %v1698 : tensor<32x1280x7x7xf32>
    %v1700 = stablehlo.reshape %v1699 : (tensor<32x1280x7x7xf32>) -> tensor<32x62720xf32>
    %v1701 = stablehlo.reshape %v1700 : (tensor<32x62720xf32>) -> tensor<32x1280x7x7xf32>
    %v1702 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1703 = stablehlo.reduce(%v1701 init: %v1702) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x1280x7x7xf32>, tensor<f32>) -> tensor<32x1280xf32>
    %v1704 = stablehlo.constant dense<49.0> : tensor<32x1280xf32>
    %v1705 = stablehlo.divide %v1703, %v1704 : tensor<32x1280xf32>
    %v1706 = stablehlo.dot_general %v1705, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1280xf32>, tensor<1280x10xf32>) -> tensor<32x10xf32>
    %v1707 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v1708 = stablehlo.add %v1706, %v1707 : tensor<32x10xf32>
    return %v1708 : tensor<32x10xf32>
  }
}
